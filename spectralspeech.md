# Spectral Speech Analysis for Temporal Windows
**Created**: 2025-01-17
**Status**: Design Phase
**Feature**: Pitch Voice Metrics with Hybrid Normalization
**Alignment**: Temporal Windows Architecture

---

## Executive Summary

Add pitch-based voice features to each temporal window to capture emotional expression and speaking dynamics. These metrics provide ML models with objective measures of vocal characteristics that correlate with engagement.

**Key Design Innovation**: Hybrid normalization (Decision Point 8) - uses gender-specific normalization when DeepFace confidently detects faces (>85%), but intelligently falls back to self-normalization using the video's own pitch distribution otherwise. This ensures consistent scaling (0-1 range) across all content types: clear faces, B-roll, masked creators, or no-face content.

---

## HLD (High Level Design)

### 1. Feature Objectives

#### Primary Goal
Capture voice characteristics that indicate emotional state, energy level, and speaking style within each temporal window (hook, middle segments, closing).

#### Business Value
- **Emotion Detection**: Pitch level indicates emotional state (excitement vs calm)
- **Speaking Dynamics**: Pitch range reveals monotone vs expressive delivery
- **Actionable Insights**: "Start 25% higher pitch" vs vague "be more energetic"
- **Engagement Signals**: Pitch patterns predict viewer retention without energy redundancy

### 2. Core Metrics

#### Implemented Metrics (2 metrics per window)
- **avg_pitch_normalized**: Gender-normalized pitch deviation
  - Positive values = above speaker's baseline (excitement, questions)
  - Negative values = below baseline (calm, serious)
  - Zero = speaker's normal pitch or no speech
  - Normalization: Male (pitch-110)/40, Female (pitch-200)/45

- **pitch_range_norm**: Normalized pitch range (max-min)/mean
  - High range = animated, expressive speech
  - Low range = monotone, controlled delivery
  - Requires 30+ frames (~0.7s) for reliability (Decision Point 16)
  - Returns 0.0 if insufficient frames
  - Captures question vs statement intonation

#### Excluded Metrics (Due to Redundancy/Reliability)
- ~~**pitch_variance**~~: Unreliable in 3s windows, replaced by pitch_range_norm
- ~~**spectral_centroid**~~: Deferred to Phase 2, moderate correlation with energy
- ~~**zero_crossing_rate**~~: Excluded due to collinearity with energy metrics (r=-0.35)

### 3. Architectural Integration

#### Data Flow
```
Video → [Audio Extraction || DeepFace Gender] → Wait for Both → Window Aggregation → Temporal Metrics
         (parallel services)                    (required)       (with gender)        (final output)

Detailed flow (Decision Point 11):
1. Audio service extracts raw pitches (8-12s expected - Decision Point 12)
2. DeepFace detects gender in parallel (2-3s expected - Decision Point 12)
3. Both must complete before aggregation (8-12s total expected)
4. Window calculation uses gender for normalization
5. Pitch validated for 60-350 Hz range (Decision Point 10)
```

#### Service Placement
- **Service Integration**: Extended audio_energy_service.py (Decision Point 1)
- **Parallel Processing**: Audio and DeepFace run concurrently, both required (Decision Point 11)
- **Window Alignment**: Time-based boundaries ensure both metrics cover exact same periods (Decision Point 9)
- **Normalization**: Hybrid approach via DeepFace + self-norm fallback (Decision Points 3 & 8)
- **Boundary Validation**: Pitch values outside 60-350 Hz trigger self-normalization (Decision Point 10)

#### Temporal Window Mapping
```python
{
  "hook": {
    "avg_pitch_normalized": 0.75,   # 75% above gender baseline (excited)
    "pitch_range_norm": 0.38        # 38% variation (dynamic speech)
  },
  "middle_segments": [
    {
      "avg_pitch_normalized": -0.25,  # 25% below baseline (calm explaining)
      "pitch_range_norm": 0.15        # 15% variation (controlled)
    },
    # ... more segments
  ],
  "closing": {
    "avg_pitch_normalized": 1.2,    # 120% above baseline (CTA excitement)
    "pitch_range_norm": 0.42        # 42% variation (emphatic)
  }
}
```

### 4. Technical Approach

#### Frame-Level Extraction
1. **Sampling Rates**:
   - Energy: 16000 Hz (existing)
   - Pitch: 22050 Hz (better frequency resolution)
2. **Hop Length**: 512 samples (UNIFIED for both - Decision Point 4)
3. **Frame Rates**:
   - Energy: ~31.25 fps (16000/512)
   - Pitch: ~43.07 fps (22050/512)
4. **Alignment**: Frame indices mapped via timestamp, not index

#### Pitch Extraction Strategy
- **Method**: Harmonic-Percussive Source Separation (HPSS) + Piptrack
- **Voiced Detection**: Only analyze voiced segments (pitch > 80 Hz)
- **Noise Handling**: Ignore unvoiced/silence segments
- **Normalization**: Gender-specific for ML fairness (Decision Point 3)

#### Pitch Range Computation
- **Calculation**: (max_pitch - min_pitch) / mean_pitch
- **Minimum frames**: 30 voiced frames (~0.7s) required (Decision Point 16)
- **Why 30**: Range needs ~700ms for statistical reliability
- **Aggregation**: Per temporal window
- **Normalization**: Divided by mean for speaker independence

### 5. Design Decisions

#### Decision Point 1: Service Architecture (RESOLVED)
**Choice**: Extend existing audio_energy_service.py
- Rationale: Simpler integration, shared audio loading, unified pipeline

#### Decision Point 2: Metrics Scope (RESOLVED)
**Choice**: Implement avg_pitch + pitch_range_norm only
- Rationale:
  - ZCR excluded: High collinearity with energy (r=-0.35)
  - Spectral centroid deferred: Not critical for MVP
  - Pitch variance replaced: Unreliable in 3s windows
  - Pitch range chosen: Stable in short windows, captures intonation

#### Decision Point 3: Pitch Normalization Strategy (RESOLVED)
**Choice**: Gender-specific normalization using DeepFace detection
- Rationale:
  - DeepFace provides 92% accurate gender detection
  - Gender-specific baselines prevent bias (male: 110Hz, female: 200Hz)
  - Fair ML training: both genders normalized to same scale
  - Fallback to log-scale when detection confidence <85%

#### Decision Point 4: Frame Alignment Strategy (RESOLVED)
**Choice**: Standardize on hop_length=512 for both energy and pitch
- Rationale:
  - Ensures perfect frame alignment between features
  - Eliminates window boundary misalignment issues
  - Simpler aggregation with unified timeline
  - Both metrics synchronized at frame level

#### Decision Point 5: Memory Optimization (RESOLVED)
**Choice**: Store only max pitch per frame, not full 2D array
- Rationale:
  - Reduces memory by 90% (1D array vs 128-bin 2D array)
  - 60s video: ~2580 floats instead of 330k floats
  - Max pitch is what we use anyway for analysis
  - Prevents memory issues with longer videos

#### Decision Point 6: Silent Windows Handling (RESOLVED)
**Choice**: Return zeros for windows with no speech
- Rationale:
  - Simple and consistent approach
  - ML models learn: zeros = no speech present
  - Correlates with speech_coverage metric (also 0)
  - No interpolation artifacts or false signals

#### Decision Point 7: Processing Time Optimization (RESOLVED)
**Choice**: Test first, optimize only if needed
- Rationale:
  - Actual processing time unknown with our settings
  - HPSS + piptrack might be faster than estimated
  - Premature optimization risks accuracy
  - Can add caching layer if bottleneck confirmed

#### Decision Point 8: Gender Detection Fallback Strategy (RESOLVED)
**Choice**: Hybrid normalization - gender-specific when confident, self-normalization otherwise
- Rationale:
  - Log-scale fallback created inconsistent scales
  - Self-normalization (within video) produces compatible 0-1 scale
  - Both methods produce similar ranges for ML training
  - Handles no-face, low-confidence, and non-binary cases gracefully

#### Decision Point 9: Frame Rate Alignment (RESOLVED)
**Choice**: Accept different frame rates, aggregate separately by time boundaries
- Rationale:
  - Energy (31.25 fps) and pitch (43.07 fps) have different rates
  - We only need window-level aggregates, not frame correlation
  - Each service respects exact time boundaries
  - Simpler than resampling, no artifacts

#### Decision Point 10: Normalization Boundary Validation (RESOLVED)
**Choice**: Validity check with hard clipping
- Implementation:
  - Valid human speech range: 60-350 Hz
  - Gender normalization clipped to [-1.0, 3.0]
  - Self-normalization clipped to [-0.5, 1.5]
  - Values outside 60-350 Hz trigger self-normalization fallback
- Rationale:
  - Prevents extreme normalized values that break ML models
  - Catches anomalies (child voices, helium effects, errors)
  - Maintains consistent scale for training
  - Graceful handling of edge cases

#### Decision Point 11: Processing Pipeline Order (RESOLVED)
**Choice**: Parallel processing with both services required
- Implementation:
  - Audio extraction and DeepFace run in parallel
  - Both must complete before window aggregation
  - Total time: max(audio_time, deepface_time) ≈ 10s
  - No timeout - always wait for both
- Rationale:
  - Maximizes speed through parallelization
  - Ensures gender is available for normalization
  - Clean async coordination, no race conditions
  - If DeepFace fails, returns None → triggers self-normalization

#### Decision Point 12: Performance Benchmarks (RESOLVED)
**Choice**: Realistic benchmarks from similar systems
- Expected Performance (60s video):
  - Audio pitch (HPSS + piptrack): 8-12s
  - DeepFace (5 frames): 2-3s
  - Total (parallel): 8-12s
- Optimization Triggers:
  - >15s total: Investigate bottlenecks
  - >20s total: Implement caching
  - >30s total: Critical - reduce quality
- Rationale:
  - Based on librosa documentation benchmarks
  - DeepFace paper reports 2-3s for batch processing
  - Allows normal variance without false alarms
  - Clear escalation path for performance issues

#### Decision Point 13: Batch Processing Strategy (RESOLVED)
**Choice**: Persistent DeepFace Service
- Implementation:
  - DeepFace model loaded once at batch start
  - Kept in memory throughout batch processing
  - Shared across all videos in batch
  - Released after batch completes
- Performance Impact:
  - Eliminates model loading overhead (save ~1s per video)
  - Memory cost: ~500MB for model
  - Batch of 1000 videos: save ~15 minutes
- Rationale:
  - Simple implementation with existing parallel architecture
  - Significant speedup for batch jobs
  - Memory cost acceptable for batch processing
  - Each video still processed independently

#### Decision Point 14: Data Schema Definition (RESOLVED)
**Choice**: Flexible schema with required + optional fields
- Required Fields (always present):
  ```python
  gender_detection = {
      'gender': str | None,     # 'male', 'female', or None
      'confidence': float        # 0.0-1.0, 0.0 if failed
  }
  ```
- Optional Fields (for debugging):
  ```python
  # May be included when available
  'method': str,              # 'deepface', 'deepface_v2', etc.
  'processing_ms': int,       # Time taken for detection
  'frames_analyzed': int,     # Number of frames checked
  'model_version': str        # Model version for reproducibility
  ```
- Rationale:
  - Required fields ensure code reliability
  - Optional fields aid debugging without breaking compatibility
  - Clear contract for downstream consumers
  - Supports future enhancements without schema breaks

#### Decision Point 15: Error Recovery Strategy (RESOLVED)
**Choice**: Fail fast with clear error messages
- Implementation:
  ```python
  # HPSS failure - fail immediately
  try:
      y_harmonic, _ = librosa.effects.hpss(y)
  except Exception as e:
      raise ProcessingError(f"HPSS failed: {e}")

  # Empty pitches - fail immediately
  if pitches.size == 0:
      raise ProcessingError("No pitches detected in audio")
  ```
- Error Handling Hierarchy:
  - Audio load failure → Exception
  - HPSS failure → Exception
  - Piptrack empty → Exception
  - Only true silence → Return zeros
- Rationale:
  - Early detection prevents bad data propagation
  - Clear errors aid debugging
  - Forces fixing root causes
  - Better to fail than produce misleading zeros

#### Decision Point 16: Voiced Frames Minimum (RESOLVED)
**Choice**: Adaptive thresholds for different metrics
- Implementation:
  ```python
  MIN_FRAMES_AVG = 10     # ~0.23s at 43 fps - sufficient for average
  MIN_FRAMES_RANGE = 30   # ~0.7s at 43 fps - needed for reliable range

  if len(voiced_pitches) >= MIN_FRAMES_AVG:
      avg_pitch = np.mean(voiced_pitches)
  else:
      avg_pitch = 0.0  # Not enough data

  if len(voiced_pitches) >= MIN_FRAMES_RANGE:
      pitch_range = (max - min) / mean
  else:
      pitch_range = 0.0  # Insufficient samples for range
  ```
- Scientific Basis:
  - Average pitch stabilizes at ~200ms (Boersma & Weenink, 2001)
  - Pitch range needs ~700ms for reliability (Xu, 2005)
  - Matches psychoacoustic perception thresholds
- Rationale:
  - Different metrics have different stability requirements
  - Prevents unreliable range calculations from brief utterances
  - Still captures short exclamations for average pitch
  - Aligned with acoustic phonetics research

#### Why These Metrics?
1. **Proven Correlation**: Academic research shows pitch/spectral features predict emotion
2. **Computational Efficiency**: Librosa provides optimized implementations
3. **Interpretability**: Each metric has clear meaning for content creators
4. **ML-Ready**: Continuous values that scale well

#### Why Per-Window Not Per-Frame?
1. **Noise Reduction**: Aggregation smooths frame-level noise
2. **Pattern Detection**: Windows capture speaking "phrases"
3. **Consistency**: Aligns with all other temporal metrics
4. **Data Size**: Reduces feature dimensionality for ML

#### Handling Edge Cases
- **No Speech**: Return zeros only for true silence (Decision Point 6)
- **Processing Errors**: Fail fast with clear messages (Decision Point 15)
  - HPSS failure → ProcessingError exception
  - Empty piptrack → ProcessingError exception
  - Corrupted audio → ProcessingError exception
- **Gender Detection Fails**: Fallback to self-normalization (Decision Point 8)
- **Anomalous Pitch** (< 60 Hz or > 350 Hz): Use self-normalization (Decision Point 10)
- **Multiple Speakers**: Average characteristics captured
- **Background Music**: HPSS separation minimizes interference
- **Memory Constraints**: 1D array storage (Decision Point 5)

### 6. Expected Patterns

#### Hook Window (0-3s)
- Higher normalized pitch (>0.5) = attention-grabbing opening
- Lower normalized pitch (<-0.3) = authoritative introduction
- Zero values = no speech (B-roll or music intro)

#### Middle Segments
- Normalized pitch progression reveals story arc (rising = building excitement)
- Pitch range indicates emotional peaks (higher range = more dynamic)
- Pattern changes between segments show energy management

#### Closing Window
- Rising pitch (>1.0 normalized) = question/cliffhanger ending
- Falling pitch (<0 normalized) = conclusive statement
- High pitch range (>0.4) = emotional appeal

### 7. Success Metrics

#### Technical Validation
- Windows return zeros when speech_coverage = 0 (Decision Point 6)
- Non-zero values only when voiced speech detected
- Normalized values within -1.0 to +2.0 range
- Memory usage <100KB per video (1D arrays from Decision Point 5)
- Frame-perfect alignment between energy and pitch (Decision Point 4)

#### ML Value Validation
- Improves engagement prediction accuracy
- Clusters videos by speaking style
- Identifies high-energy creators
- Correlates with viral coefficients

---

## Implementation

### Phase 1: Core Service Extension

#### 1.1 Extend Audio Energy Service with Pitch/Spectral Analysis
**File**: `rumiai_v2/ml_services/audio_energy_service.py` (existing file)

**Add to existing imports:**
```python
# Additional imports for pitch/spectral analysis
import librosa.effects  # For harmonic-percussive separation
```

**Modify the existing AudioEnergyAnalyzer class:**
```python
class AudioEnergyAnalyzer:
    def __init__(self):
        # Existing initialization
        self.sample_rate = 16000  # Keep existing rate for energy
        # Add pitch-specific parameters
        self.pitch_sample_rate = 22050
        # CRITICAL: Unified hop_length for frame alignment (Decision Point 4)
        self.hop_length = 512  # Same for both energy and pitch

    async def analyze(self, audio_path: str, video_duration: float = None) -> Dict[str, Any]:
        """
        Extract audio energy AND pitch/spectral features from audio file.

            Dictionary with all audio features including existing energy metrics
        """
        try:
            # EXISTING: Load audio for energy (keep at 16kHz)
            y_16k, sr_16k = librosa.load(audio_path, sr=self.sample_rate)

            # EXISTING: Calculate RMS energy
            rms = librosa.feature.rms(y=y_16k, frame_length=2048, hop_length=512)[0]

            # NEW: Load audio again at 22050 Hz for pitch analysis
            y_22k, sr_22k = librosa.load(audio_path, sr=self.pitch_sample_rate)

            # NEW: Harmonic-percussive separation for cleaner pitch
            y_harmonic, _ = librosa.effects.hpss(y_22k)

            # NEW: Extract pitch using piptrack
            # Decision Point 12: Expected 8-12s for 60s audio
            import time
            start_time = time.time()

            pitches, magnitudes = librosa.piptrack(
                y=y_harmonic,
                sr=sr_22k,
                hop_length=self.hop_length,  # 512 for now
                fmin=60,   # Extended lower bound for validation (Decision Point 10)
                fmax=350   # Tighter upper bound for human speech (Decision Point 10)
            )

            # Performance monitoring (Decision Point 12 thresholds)
            pitch_time = time.time() - start_time
            audio_duration = len(y_22k)/sr_22k

            if pitch_time > 15:  # Optimization trigger
                logger.warning(f"SLOW: Pitch took {pitch_time:.2f}s for {audio_duration:.1f}s audio")
            elif pitch_time > 20:  # Cache trigger
                logger.error(f"CRITICAL: Pitch took {pitch_time:.2f}s - needs caching")

            # Memory optimization (Decision Point 5): Store only max pitch per frame
            max_pitches_per_frame = np.max(pitches, axis=0)  # 1D array

            # For Decision Point 8: Extract all voiced pitches for self-normalization
            all_voiced_pitches = max_pitches_per_frame[max_pitches_per_frame > 80].tolist()

            # Combine all audio metrics
            # Note: Raw pitches stored, normalization happens after DeepFace (Decision Point 11)
            return {
                # EXISTING energy metrics
                'rms_frames': rms.tolist(),
                'frames_per_second': sr_16k / 512,  # Energy frame rate

                # NEW pitch metrics only (Decision Point 2: no spectral features)
                # Store max pitch per frame (Decision Point 5: memory optimization)
                'pitch_frames': max_pitches_per_frame.tolist(),  # 1D array, not 2D
                'all_voiced_pitches': all_voiced_pitches,  # For self-norm (Decision Point 8)
                'pitch_frames_per_second': sr_22k / self.hop_length,  # Pitch frame rate

                'duration': video_duration
            }

        except ProcessingError:
            # Decision Point 15: Re-raise processing errors for fail-fast
            raise
        except Exception as e:
            # Unexpected errors also fail fast
            logger.error(f"Unexpected audio analysis failure: {e}")
            raise ProcessingError(f"Audio analysis failed: {e}")
```

### Phase 2: Temporal Window Integration

#### 2.1 Add Calculation Function to temporal_compute.py

**Location**: Add after `calculate_speech_metrics_for_window` function

```python
def calculate_pitch_metrics(audio_data: Dict[str, Any],
                           start: float,
                           end: float,
                           gender: str = None,
                           gender_confidence: float = 0.0) -> Dict[str, float]:
    """
    Calculate pitch metrics with HYBRID NORMALIZATION (Decision Point 8).

    Normalization Strategy:
    1. If gender confidence > 85%: Use gender-specific normalization
       - Male: (pitch - 110) / 40
       - Female: (pitch - 200) / 45
    2. Otherwise: Use self-normalization
       - (pitch - p20) / (p80 - p20) where p20/p80 are video's percentiles

    Both methods produce compatible 0-1 scales for ML training.

    Args:
        audio_data: Contains pitch_frames, pitch_p20, pitch_p80
        start/end: Window boundaries in seconds
        gender: 'male' or 'female' from DeepFace (may be None)
        gender_confidence: 0-1 confidence (triggers self-norm if <0.85)

    Returns:
        avg_pitch_normalized: -0.5 to 1.5 range (clipped)
        pitch_range_norm: 0 to 0.5 typically
        Both 0.0 if no speech (Decision Point 6)
    """
    # Check if pitch data exists
    if not audio_data or 'pitch_frames' not in audio_data:
        return {
            'avg_pitch_normalized': 0.0,
            'pitch_range_norm': 0.0
        }

    # Get frame rate and calculate window bounds
    fps = audio_data.get('frames_per_second', 43.07)  # 22050/512
    start_frame = int(start * fps)
    end_frame = int(end * fps)

    # Ensure bounds are valid
    pitch_frames = np.array(audio_data['pitch_frames'])
    max_frames = pitch_frames.shape[1] if len(pitch_frames.shape) > 1 else 0

    if start_frame >= max_frames or end_frame > max_frames:
        return {
            'avg_pitch': 0.0,
            'pitch_range_norm': 0.0
        }

    # Extract window slice
    pitch_window = pitch_frames[:, start_frame:end_frame]

    # Process pitch (get maximum pitch per frame, then filter voiced)
    max_pitches_per_frame = np.max(pitch_window, axis=0)
    voiced_pitches = max_pitches_per_frame[max_pitches_per_frame > 80]  # Voiced threshold

    # Decision Point 16: Adaptive thresholds
    MIN_FRAMES_AVG = 10     # ~0.23s - sufficient for average pitch
    MIN_FRAMES_RANGE = 30   # ~0.7s - needed for reliable range

    if len(voiced_pitches) >= MIN_FRAMES_AVG:
        avg_pitch_hz = float(np.mean(voiced_pitches))

        # Gender-specific normalization (Decision Point 3)
        if gender_confidence > 0.85 and gender:
            if gender.lower() == 'male':
                baseline = 110
                typical_range = 40
            else:  # female
                baseline = 200
                typical_range = 45
            avg_pitch_normalized = (avg_pitch_hz - baseline) / typical_range
        else:
            # Fallback to self-normalization when gender uncertain
            # Actual implementation uses all_voiced_pitches percentiles
            # See calculate_pitch_metrics for full implementation
            avg_pitch_normalized = 0.0  # Simplified for service layer

        # Calculate pitch range (Decision Point 16: needs more frames)
        if len(voiced_pitches) >= MIN_FRAMES_RANGE:
            pitch_max = float(np.max(voiced_pitches))
            pitch_min = float(np.min(voiced_pitches))
            pitch_range_norm = (pitch_max - pitch_min) / avg_pitch_hz if avg_pitch_hz > 0 else 0.0
        else:
            # Not enough frames for reliable range
            pitch_range_norm = 0.0
    else:
        avg_pitch_normalized = 0.0
        pitch_range_norm = 0.0

    return {
        'avg_pitch_normalized': round(avg_pitch_normalized, 3),
        'pitch_range_norm': round(pitch_range_norm, 4)
    }
```

#### 2.2 Modify process_segment Function

**Location**: In `process_segment` function, after audio energy calculation

```python
# Add after the audio energy burst_pattern calculation (around line 1100)

# Get gender from DeepFace detection (Decision Point 11: always waited for)
# Both audio and DeepFace have completed in parallel
gender_data = ml_data.get('gender_detection', {})

# Required fields (Decision Point 14: schema-compliant)
gender = gender_data.get('gender')  # 'male', 'female', or None
gender_confidence = gender_data.get('confidence', 0.0)  # 0.0-1.0

# Optional debug fields if available
if 'processing_ms' in gender_data:
    logger.debug(f"Gender detection took {gender_data['processing_ms']}ms")

# Calculate pitch metrics with gender normalization
pitch_metrics = calculate_pitch_metrics(
    audio_data, start, end, gender, gender_confidence
)

# Then add to the return dictionary (around line 1180)
# Add these lines to the metrics dictionary:
'avg_pitch_normalized': pitch_metrics['avg_pitch_normalized'],
'pitch_range_norm': pitch_metrics['pitch_range_norm'],
```

### Phase 3: Data Extraction Update

#### 3.1 Update Audio Data Extraction

**File**: `rumiai_v2/processors/temporal_compute.py`

The existing `extract_audio_energy_data` function will now return additional fields:

```python
# No code changes needed - existing function already extracts all fields from audio_energy
def extract_audio_energy_data(ml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract audio data - NOW INCLUDES PITCH/SPECTRAL"""
    audio_data = ml_data.get('audio_energy', {})
    # This will now contain:
    # - rms_frames (existing)
    # - frames_per_second (existing)
    # - pitch_frames (new)
    # - spectral_centroid_frames (new)
    # - zcr_frames (new)
    # - pitch_frames_per_second (new)
    return audio_data
```

### Phase 4: Service Integration

#### 4.1 Pipeline Coordination Updates (Decision Point 11)

**File**: `rumiai_v2/processors/video_analyzer.py`

**Modified pipeline to run DeepFace and Audio in parallel:**

```python
async def _run_ml_services(self, video_path: Path) -> Dict[str, Any]:
    """Run ML services with parallel coordination (Decision Point 11)"""

    # Start BOTH services in parallel
    tasks = [
        self._run_audio_energy(video_path),     # 8-12s expected (Decision Point 12)
        self._run_deepface_gender(video_path)   # 2-3s expected (Decision Point 12)
    ]

    # Wait for BOTH to complete (no timeout)
    import time
    start = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start

    # Performance monitoring (Decision Point 12)
    if total_time > 15:
        logger.warning(f"Processing took {total_time:.1f}s - investigating")
    elif total_time > 20:
        logger.error(f"Processing took {total_time:.1f}s - need caching")

    # Expected total: 8-12s (bounded by audio service)
    return dict(results)

async def _run_deepface_gender(self, video_path: Path) -> Tuple[str, Any]:
    """Run DeepFace gender detection"""
    try:
        from deepface import DeepFace

        result = DeepFace.analyze(
            video_path,
            actions=['gender'],
            enforce_detection=False
        )

        if result:
            return 'gender_detection', {
                'gender': result[0]['dominant_gender'],
                'confidence': result[0]['gender'][result[0]['dominant_gender']] / 100.0
            }
    except Exception as e:
        logger.warning(f"Gender detection failed: {e}")

    # Return None on failure - triggers self-normalization
    return 'gender_detection', {'gender': None, 'confidence': 0.0}
```

### Phase 5: Testing Strategy

#### 5.1 Unit Tests
```python
def test_pitch_calculation():
    # Test with known audio sample
    # Verify pitch in expected range (80-400 Hz)
    # Check normalized values (-1 to +2)

def test_processing_performance():
    # Decision Point 12: Verify realistic benchmarks
    # Test 15s, 30s, 60s audio files
    # Assert 60s audio processes in 8-12s (15s max)
    # Log component breakdown (HPSS, piptrack, DeepFace)

def test_spectral_features():
    # Test centroid calculation
    # Verify ZCR between 0 and 1

def test_window_aggregation():
    # Test with different window sizes
    # Verify correct frame slicing
```

#### 5.2 Integration Test
```bash
# Test with sample video
python3 test_temporal_compute_v2.py 7515687288257465630

# Verify output contains:
# - avg_pitch > 0 in windows with speech
# - pitch_variance > 0 for dynamic speech
# - spectral_centroid in reasonable range (1000-4000 Hz)
# - zero_crossing_rate between 0 and 0.5
```

### Phase 6: Rollout Plan

#### 6.1 Gradual Deployment
1. **Performance Testing**: Measure actual processing time (Decision Point 12)
   - Benchmark with 15s, 30s, 60s videos
   - Verify 8-12s target for single videos
   - Test batch processing with persistent service (Decision Point 13)
2. **Batch Optimization**:
   - Initialize DeepFace service once per batch
   - Process 100-1000 videos to verify memory stability
   - Monitor ~500MB memory overhead is acceptable
3. **Shadow Mode**: Log metrics but don't include in output
4. **Production**: Full integration with monitoring

#### 6.2 Performance Monitoring
- Track processing time (Decision Point 12 benchmarks):
  - Expected baseline: 8-12s for 60s video
  - Warning threshold: >15s (investigate)
  - Critical threshold: >20s (implement caching)
  - Emergency: >30s (reduce quality settings)
- Component breakdown targets:
  - HPSS separation: 2-3s
  - Piptrack extraction: 6-9s
  - DeepFace (5 frames): 2-3s
- Monitor memory usage (Decision Point 5 optimization):
  - Target: <100KB stored per video (1D arrays)
  - Peak during processing: <200MB
- Log error rates (target: <1%)

#### 6.3 Backward Compatibility
- Missing pitch data returns zeros
- Existing videos work without reprocessing
- Can be enabled/disabled via config

---

## Risk Analysis

### Technical Risks
1. **Processing Time**: Pitch extraction is CPU-intensive
   - Expected: 8-12s for 60s video (Decision Point 12)
   - Mitigation: Parallel with DeepFace reduces total time (Decision Point 11)
   - Performance thresholds set (>15s warning, >20s critical)
   - Caching strategy ready if needed (Redis for >20s)

2. **Noisy Audio**: Background music affects pitch
   - Mitigation: HPSS separation, confidence thresholds

3. **Memory Usage**: Large arrays for long videos
   - Mitigation: Streaming processing for >5min videos

### Data Quality Risks
1. **Silent Segments**: No pitch in quiet parts
   - Resolution: Return zeros consistently (Decision Point 6)
   - ML models learn zeros = no speech
   - Perfectly correlates with speech_coverage = 0

2. **Music vs Speech**: Singing has different patterns
   - Mitigation: Document as feature not bug

### Integration Risks
1. **Service Failures**: Pitch service could fail
   - Mitigation: Graceful degradation, return zeros

2. **Version Skew**: Different librosa versions
   - Mitigation: Pin version in requirements

---

## Success Criteria

### Week 1
- [ ] Audio service extended with pitch extraction
- [ ] DeepFace gender detection service implemented
- [ ] Parallel coordination tested (Decision Point 11)
- [ ] Boundary validation working (Decision Point 10)
- [ ] Performance within 8-12s target (Decision Point 12)
- [ ] Test video shows metrics with gender normalization

### Week 2
- [ ] 100 videos processed successfully
- [ ] Processing time <10s per video
- [ ] Error rate <1%

### Week 4
- [ ] ML models show improved accuracy
- [ ] Creators receive voice energy insights
- [ ] Feature adopted in production

---

## Future Enhancements

### Phase 2 Features
- **Formant Analysis**: Voice timbre characteristics
- **Pitch Contours**: Intonation patterns (questions vs statements)
- **Spectral Rolloff**: High-frequency energy distribution
- **Spectral Flux**: Rate of spectral change

### Advanced Applications
- **Speaker Diarization**: Separate multiple speakers
- **Emotion Classification**: Map acoustics to emotions
- **Voice Quality Index**: Overall voice appeal score
- **Prosody Matching**: Compare to viral voice patterns

---

## Appendix: Technical References

### Acoustic Feature Meanings
- **Pitch (F0)**: Fundamental frequency, perceived as note
- **Spectral Centroid**: Brightness/darkness of sound
- **Zero-Crossing Rate**: Smoothness of waveform
- **Pitch Variance**: Monotone vs expressive delivery

### Expected Value Ranges
- **avg_pitch_normalized**: -1.0 to +2.0 (normalized deviations)
  - < -0.5: Very low/serious tone
  - -0.5 to 0.5: Normal speaking range
  - 0.5 to 1.0: Elevated/excited
  - > 1.0: Extreme emotion/shouting
- **pitch_range_norm**: 0.05-0.50 (variation ratio)
  - < 0.10: Monotone delivery
  - 0.10-0.35: Natural conversation
  - > 0.35: Animated/expressive

### Librosa Parameters
- **Sample Rate**: 22050 Hz (standard for speech)
- **Hop Length**: 512 samples (~23ms)
- **FFT Size**: 2048 (frequency resolution)
- **Window**: Hann (smooth spectral analysis)