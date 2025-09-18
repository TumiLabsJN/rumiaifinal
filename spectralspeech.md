# Spectral Speech Analysis for Temporal Windows
**Created**: 2025-01-17
**Status**: PLANNING DOCUMENT - Pitch features NOT YET IMPLEMENTED
**Prerequisites**: DeepFace gender detection (✅ ALREADY IMPLEMENTED)
**Feature**: Pitch Voice Metrics with Hybrid Normalization (FUTURE)
**Alignment**: Temporal Windows Architecture

---

## Executive Summary

Add pitch-based voice features to each temporal window to capture emotional expression and speaking dynamics. These metrics provide ML models with objective measures of vocal characteristics that correlate with engagement.

**Key Design Innovation**: Hybrid normalization (Decision Point 8) - uses gender-specific normalization when DeepFace confidently detects faces (>85%), but intelligently falls back to self-normalization using the video's own pitch distribution otherwise. This ensures consistent scaling (0-1 range) across all content types: clear faces, B-roll, masked creators, or no-face content.

### Current State: What's Already Implemented

**DeepFace Gender Detection (✅ COMPLETE - DO NOT REBUILD):**
1. **Already in Pipeline**:
   - Added to `required_models` in `analysis.py`
   - Integrated in `video_analyzer.py` via `DeepFaceGenderServiceSimple`
   - Data flows through `ml_data['deepface_gender']`
   - Extracted to metadata in `temporal_compute.py` lines 1498-1504

2. **Subprocess Architecture** (Due to TensorFlow conflicts):
   - Service: `rumiai_v2/ml_services/deepface_gender_service_simple.py`
   - Script: `scripts/run_deepface_gender.py`
   - Face size filtering: 120x120 pixels minimum

3. **Data Location** (AUTHORITATIVE):
   ```python
   # Source of truth:
   ml_data['deepface_gender']  # Where DeepFace data lives

   # Convenience copy for temporal windows:
   metadata['gender_detection']  # Copied in temporal_compute.py
   ```

**Pitch Analysis (🔮 FUTURE - THIS DOCUMENT):**
- Not yet implemented
- Will extend audio_energy_service.py
- Depends on DeepFace data for normalization

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

### 2. Temporal Window Metrics

#### Metrics Added to Each Window (Decision Point 2: Two pitch metrics only)
- **avg_pitch_normalized**: Gender-normalized pitch deviation (Decision Point 3)
  - Positive values = above speaker's baseline (excitement, questions)
  - Negative values = below baseline (calm, serious)
  - Zero = speaker's normal pitch or no speech
  - Normalization approach (Decision Point 3 confirmed):
    - Primary: Male (pitch-110)/40, Female (pitch-200)/45
    - Fallback: Self-normalization when gender unknown/multiple people
  - Added to hook, each middle_segment, and closing sections

- **pitch_range_norm**: Normalized pitch range (max-min)/mean
  - High range = animated, expressive speech
  - Low range = monotone, controlled delivery
  - Requires 30+ frames (~0.7s) for reliability (Decision Point 16)
  - Returns 0.0 if insufficient frames
  - Captures question vs statement intonation
  - Added alongside avg_pitch_normalized in all windows

#### Excluded from Implementation (Decision Point 2 confirmed)
- ~~**pitch_variance**~~: Unreliable in 3s windows, replaced by pitch_range_norm
- ~~**spectral_centroid**~~: Deferred to Phase 2
- ~~**zero_crossing_rate**~~: Excluded due to collinearity with energy (r=-0.35)

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
- **Service Integration**: Extended audio_energy_service.py (Decision Point 1 - confirmed to extend existing rather than create new service)
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

#### Decision Point 1: Service Architecture (RESOLVED ✅ CONFIRMED)
**Choice**: Extend existing audio_energy_service.py
- Rationale: Pragmatic approach that works today without refactoring
- Benefits: Single audio load, no breaking changes, simpler deployment
- Trade-off accepted: Service name becomes slightly misleading but functionality is sound

#### Decision Point 2: Metrics Scope (RESOLVED ✅ CONFIRMED)
**Choice**: Implement avg_pitch_normalized + pitch_range_norm only
- Confirmed metrics:
  - avg_pitch_normalized: Gender-aware average pitch deviation
  - pitch_range_norm: Pitch variation (max-min)/mean for expressiveness
- Excluded from MVP:
  - ZCR: High collinearity with energy (r=-0.35), redundant
  - Spectral centroid: Deferred to Phase 2, not critical
  - Pitch variance: Replaced by pitch_range_norm (more reliable in 3s windows)

#### Decision Point 3: Pitch Normalization Strategy (RESOLVED ✅ CONFIRMED)
**Choice**: Gender-specific normalization (REQUIRED per Decision Point 11)
- Primary approach:
  - Male: (pitch - 110Hz) / 40
  - Female: (pitch - 200Hz) / 45
  - Provides fair comparison across genders
- Special case: 'multiple_people':
  - Use self-normalization for multi-person videos
  - This is valid data, not an error
- No fallback for missing gender:
  - If DeepFace fails → pipeline fails (Decision Point 11)
  - Gender detection is required, not optional

#### Decision Point 4: Frame Alignment Strategy (RESOLVED ✅ REVISED)
**Choice**: Time-based alignment, not frame-based
- Each service uses optimal parameters:
  - Energy: 16kHz, hop=512 → 31.25 fps
  - Pitch: 22.05kHz, hop=512 → 43.07 fps (different fps is OK)
- Alignment happens at window aggregation:
  - We only need window-level metrics (3+ seconds)
  - Use timestamps to extract correct window segments
  - No need for frame-perfect alignment
- Benefits:
  - Simpler - no forced parameters
  - Each feature optimized independently
  - Window boundaries respected by time, not frame index

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

#### Decision Point 11: Processing Pipeline Order (RESOLVED ✅ REVISED)
**Choice**: Sequential processing - fail fast
- Implementation:
  - DeepFace MUST complete first (already in pipeline)
  - If DeepFace fails → entire pipeline fails (fail fast)
  - Only then run pitch analysis with gender data
  - No parallel coordination complexity
- Rationale:
  - DeepFace failure is a bug to fix, not hide
  - Sequential is simpler and more debuggable
  - Gender data is required, not optional
  - Fail fast prevents bad data propagation
- Error handling:
  - DeepFace timeout/error → raise exception
  - Don't hide failures with fallbacks
  - Force fixing root cause issues

#### Decision Point 12: Performance Benchmarks (NEEDS VALIDATION)
**WARNING**: These are estimates - MUST benchmark before production
- Estimated Performance (60s video):
  - Current audio_energy (just RMS): Already takes 3-5s
  - Adding HPSS: +2-3s expected
  - Adding piptrack: +5-8s expected
  - **Realistic total: 10-16s** (not 8-12s as originally claimed)
  - DeepFace (already implemented): 3-4s via subprocess
- Performance validation required:
  ```python
  # MUST TEST before implementing:
  import time
  start = time.time()
  pitches = librosa.piptrack(y, sr=22050, hop_length=512)
  print(f"Actual time: {time.time()-start:.2f}s")
  ```
- Optimization triggers TBD based on actual measurements

#### Decision Point 13: Batch Processing Strategy (RESOLVED - REVISED)
**Choice**: Subprocess Isolation via DeepFaceGenderServiceSimple
- Implementation:
  - DeepFace runs in subprocess due to TensorFlow memory conflicts
  - Standalone script: `scripts/run_deepface_gender.py`
  - Wrapper service: `DeepFaceGenderServiceSimple`
  - Face size filtering: Minimum 120x120 pixels to avoid false positives
- Performance Impact:
  - Subprocess overhead: ~100ms per video
  - Total processing: 3-4s per video (7 frames sampled)
  - Memory isolation prevents corruption
- Rationale:
  - TensorFlow 2.16 + ThreadPoolExecutor + Python 3.12 = memory corruption
  - Subprocess is architecturally correct solution, not band-aid
  - Prevents false positives from logos/watermarks
  - Each video processes independently with clean memory state

#### Decision Point 14: Data Schema Definition (RESOLVED - IMPLEMENTED)
**Choice**: Flexible schema with required + optional fields
- Actual Implementation:
  ```python
  # From scripts/run_deepface_gender.py
  gender_detection = {
      'gender': str | None,        # 'male', 'female', 'multiple_people', or None
      'confidence': float,          # 0.0-1.0, 0.0 if failed
      'method': 'deepface',         # Always included
      'frames_analyzed': int,       # Number of frames checked
      'detector_backend': 'opencv', # Face detector used
      'processing_ms': int          # Time taken for detection
  }
  ```
- Special Cases:
  - `'multiple_people'`: Detected when faces >120x120 pixels in multiple frames
  - Confidence 0.0: Indicates self-normalization should be used
  - Face size filtering: Prevents logos/watermarks from triggering multi-person
- Integration:
  - Data flows through `ml_data['deepface_gender']`
  - Must be extracted in `temporal_compute.py` for unified JSON visibility
  - Appears in `insights/*_temporal_windows_updated.json` metadata section

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

## Implementation Plan

### Architecture Notes

#### Service Patterns
1. **All ML services are async** - This is the established pattern in RumiAI
2. **DeepFaceGenderServiceSimple is the ACTUAL service** - Uses subprocess isolation
3. **DeepFaceGenderService is DEAD CODE** - Should be removed, never actually used
4. **Async for I/O, sync for math** - Services load data async, calculation functions are sync

#### Data Flow Pattern
```python
# Step 1: Async services load and process data (I/O operations)
ml_data['audio_energy'] = await audio_service.analyze(video_path)  # Async I/O
ml_data['deepface_gender'] = await deepface_service.analyze(video_path)  # Subprocess

# Step 2: Synchronous functions do math on loaded data (no I/O)
pitch_metrics = calculate_pitch_metrics(
    ml_data['audio_energy'],  # Already loaded
    ml_data['deepface_gender'],  # Already loaded
    start, end
)  # Just math, no async needed
```

### Phase 0: Cleanup Dead Code

#### 0.1 Remove Unused DeepFace Service
```bash
# Remove the dead code service that's never used
rm rumiai_v2/ml_services/deepface_gender_service.py

# Verify nothing imports it
grep -r "from.*deepface_gender_service import DeepFaceGenderService" rumiai_v2/
# Expected: No results (confirming it's not imported anywhere)

# Run tests to ensure nothing breaks
python -m pytest tests/test_deepface*.py -v

# Commit the cleanup
git add -A
git commit -m "Remove dead code: unused DeepFaceGenderService"
```

**Why**: `DeepFaceGenderService` is dead code. The actual service used is `DeepFaceGenderServiceSimple` which runs via subprocess to avoid TensorFlow memory corruption.

### Prerequisites Check
```python
# Verify DeepFace subprocess service is working before implementing pitch
# NOTE: DeepFaceGenderServiceSimple is the ACTUAL service used
from rumiai_v2.ml_services.deepface_gender_service_simple import DeepFaceGenderServiceSimple
service = DeepFaceGenderServiceSimple()
result = await service.analyze('test_video.mp4')
assert result['gender'] in ['male', 'female', 'multiple_people']
print(f"✅ DeepFace subprocess service working: {result}")
```

### Phase 1: Core Service Extension

#### 1.0 Configuration Usage Examples
```python
# Default configuration (high quality)
analyzer = AudioEnergyAnalyzer()

# Custom configuration for faster processing
fast_config = PITCH_CONFIG.copy()
fast_config['quality'] = 'low'
fast_config['max_voiced_samples'] = 200  # Less memory
analyzer = AudioEnergyAnalyzer(config=fast_config)

# Disable pitch extraction entirely
no_pitch_config = PITCH_CONFIG.copy()
no_pitch_config['enabled'] = False
analyzer = AudioEnergyAnalyzer(config=no_pitch_config)

# Override specific settings
custom_config = PITCH_CONFIG.copy()
custom_config['fmin'] = 50  # Lower minimum for deep voices
custom_config['fmax'] = 400  # Higher maximum for children
analyzer = AudioEnergyAnalyzer(config=custom_config)
```

#### 1.1 Extend Audio Energy Service with Pitch Analysis
**File**: `rumiai_v2/ml_services/audio_energy_service.py`

**Implementation mirrors HLD Decision Points 1, 2, 4:**

**Add to existing imports:**
```python
# Additional imports for pitch/spectral analysis
import librosa.effects  # For harmonic-percussive separation
```

**Complete service implementation with error handling:**
```python
import logging
import time
import numpy as np
from typing import Dict, Any, Optional
import librosa
import librosa.effects  # For harmonic-percussive separation
from librosa.util.exceptions import ParameterError  # For proper error handling

logger = logging.getLogger(__name__)

# CONFIGURATION: Centralized pitch extraction settings
PITCH_CONFIG = {
    # Core settings
    'enabled': True,                    # Set False to disable pitch extraction
    'sample_rate': 22050,               # Hz - optimal for pitch detection
    'hop_length': 512,                  # Samples - controls time resolution

    # Pitch detection parameters
    'fmin': 60,                         # Hz - minimum human voice frequency
    'fmax': 350,                        # Hz - maximum human voice frequency
    'voiced_threshold': 80,             # Hz - below this is considered unvoiced

    # Performance settings
    'quality': 'high',                  # 'high', 'medium', 'low'
    'max_processing_time': 15,         # Seconds - warning threshold
    'timeout': 30,                      # Seconds - hard timeout

    # Memory optimization
    'max_voiced_samples': 500,         # Maximum voiced samples to store
    'cache_percentiles': True,         # Pre-compute statistics for efficiency

    # Normalization thresholds
    'min_frames_avg': 10,              # Minimum frames for average pitch
    'min_frames_range': 30,            # Minimum frames for pitch range

    # Quality presets
    'quality_presets': {
        'high': {'hop_length': 512},    # Best quality, slower
        'medium': {'hop_length': 768},  # Balanced
        'low': {'hop_length': 1024}     # Fast, lower quality
    }
}

class AudioEnergyAnalyzer:
    """
    Audio analysis service for energy AND pitch metrics.
    Decision Point 1: Extended existing service (not renamed for stability).

    NOTE: This service is async (like all ML services in RumiAI).
    The async pattern is: services do I/O → store in ml_data → sync functions do math.
    """
    def __init__(self, config: Dict[str, Any] = None):
        # Use provided config or defaults
        self.config = config or PITCH_CONFIG

        # Existing energy parameters
        self.sample_rate = 16000
        self.energy_hop = 512  # → 31.25 fps

        # Pitch parameters from config
        self.pitch_sample_rate = self.config['sample_rate']

        # Apply quality preset if specified
        quality = self.config.get('quality', 'high')
        if quality in self.config['quality_presets']:
            preset = self.config['quality_presets'][quality]
            self.pitch_hop = preset['hop_length']
        else:
            self.pitch_hop = self.config['hop_length']

    async def analyze(self, audio_path: str, video_duration: float = None) -> Dict[str, Any]:
        """
        Extract energy + pitch features (Decision Point 2: only avg_pitch & range).
        Fail fast on errors (Decision Point 11: no hiding failures).

        Performance optimization: Audio is loaded once at 22050 Hz, then resampled
        to 16000 Hz for energy. This saves ~30% I/O time vs loading twice.
        """
        try:
            # Performance tracking
            import time
            start_time = time.time()

            # OPTIMIZED: Load audio once at higher sample rate
            # Load at 22050 Hz (optimal for pitch detection)
            y_22k, sr_22k = librosa.load(audio_path, sr=self.pitch_sample_rate)

            # Resample to 16000 Hz for energy analysis
            # This is faster than loading twice from disk
            y_16k = librosa.resample(
                y_22k,
                orig_sr=self.pitch_sample_rate,
                target_sr=self.sample_rate
            )
            sr_16k = self.sample_rate

            logger.info(f"Audio loaded once and resampled - saved I/O time")

            # EXISTING: Energy extraction (now using resampled audio)
            rms = librosa.feature.rms(y=y_16k, frame_length=2048, hop_length=self.energy_hop)[0]

            # NEW: Pitch extraction (using original 22050 Hz audio)

            # Harmonic separation for cleaner pitch (error = fail fast)
            try:
                y_harmonic, _ = librosa.effects.hpss(y_22k)
            except ParameterError as e:
                raise RuntimeError(f"HPSS failed with invalid parameters: {e}")
            except ValueError as e:
                raise RuntimeError(f"HPSS failed with value error (likely corrupted audio): {e}")
            except Exception as e:
                # Unexpected error - log type for debugging
                logger.error(f"Unexpected HPSS error type: {type(e).__name__}")
                raise RuntimeError(f"HPSS failed unexpectedly: {e}")

            # Extract pitch with validation (using config values)
            try:
                pitches, magnitudes = librosa.piptrack(
                    y=y_harmonic,
                    sr=sr_22k,
                    hop_length=self.pitch_hop,
                    fmin=self.config['fmin'],     # From config
                    fmax=self.config['fmax']      # From config
                )
            except ParameterError as e:
                raise RuntimeError(f"Piptrack failed with invalid parameters: {e}")
            except Exception as e:
                logger.error(f"Unexpected piptrack error type: {type(e).__name__}")
                raise RuntimeError(f"Pitch extraction failed: {e}")

            # Validate pitch extraction succeeded
            if pitches.size == 0:
                raise RuntimeError("Pitch extraction returned empty - no pitched content detected")

            # Memory optimization: Store only max pitch per frame (1D not 2D)
            max_pitches_per_frame = np.max(pitches, axis=0)

            # Extract voiced pitches for self-normalization
            # (only used for multi-person videos per Decision Point 3)
            voiced_threshold = self.config['voiced_threshold']
            all_voiced_pitches = max_pitches_per_frame[max_pitches_per_frame > voiced_threshold]

            # MEMORY OPTIMIZATION: Cap sample + compute statistics
            # Store limited sample for self-normalization
            max_samples = self.config['max_voiced_samples']
            voiced_sample = all_voiced_pitches[:max_samples].tolist() if len(all_voiced_pitches) > 0 else []

            # Pre-compute statistics for self-normalization
            pitch_stats = {}
            if len(all_voiced_pitches) >= 20:  # Need minimum samples for percentiles
                pitch_stats = {
                    'p20': float(np.percentile(all_voiced_pitches, 20)),
                    'p50': float(np.percentile(all_voiced_pitches, 50)),
                    'p80': float(np.percentile(all_voiced_pitches, 80)),
                    'mean': float(np.mean(all_voiced_pitches)),
                    'std': float(np.std(all_voiced_pitches)),
                    'total_voiced_frames': len(all_voiced_pitches)
                }

            logger.info(f"Pitch extraction: {len(all_voiced_pitches)} voiced frames, "
                       f"storing {len(voiced_sample)} samples + statistics")

            # Performance check
            processing_time = time.time() - start_time
            max_time = self.config['max_processing_time']
            if processing_time > max_time:
                logger.warning(f"Pitch extraction slow: {processing_time:.1f}s (threshold: {max_time}s)")

            # Build return dictionary with CONSISTENT naming
            return {
                # Energy metrics (existing)
                'rms_frames': rms.tolist(),
                'energy_fps': sr_16k / self.energy_hop,  # 31.25 fps

                # Pitch metrics (new - Decision Point 2)
                'pitch_frames': max_pitches_per_frame.tolist(),
                'pitch_fps': sr_22k / self.pitch_hop,  # 43.07 fps

                # Memory-optimized self-normalization data (capped + stats)
                'voiced_pitch_sample': voiced_sample,  # Max 500 values
                'pitch_statistics': pitch_stats,  # Pre-computed percentiles

                'duration': video_duration or (len(y_16k) / sr_16k),
                'processing_ms': int(processing_time * 1000)
            }

        except RuntimeError:
            # Our explicit errors - reraise as-is
            raise
        except ParameterError as e:
            # Librosa parameter errors
            raise RuntimeError(f"Audio analysis failed with invalid parameters: {e}")
        except (OSError, IOError) as e:
            # File access errors
            raise RuntimeError(f"Audio file access failed: {e}")
        except Exception as e:
            # Unexpected errors - log type and fail fast
            logger.error(f"Unexpected error in audio analysis: {type(e).__name__}")
            raise RuntimeError(f"Audio analysis failed unexpectedly: {e}")
```

### Phase 2: Temporal Window Integration

#### 2.1 Add Calculation Function to temporal_compute.py

**Location**: Add after `calculate_speech_metrics_for_window` function

```python
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_pitch_metrics(audio_data: Dict[str, Any],
                           ml_data: Dict[str, Any],
                           start: float,
                           end: float) -> Dict[str, float]:
    """
    Calculate pitch metrics for a temporal window.

    Decision Point 3: Gender-specific normalization (required).
    Decision Point 11: Fail fast if gender missing.
    Decision Point 2: Only avg_pitch_normalized and pitch_range_norm.

    Returns:
        Dictionary with exactly 2 metrics (zeros if no speech)
    """
    import numpy as np

    # Step 1: Gender validation (Decision Point 11: fail fast)
    gender_data = ml_data.get('deepface_gender')
    if not gender_data:
        raise ValueError("DeepFace data missing - pipeline error")

    gender = gender_data.get('gender')
    if gender is None:
        raise ValueError(f"DeepFace failed: {gender_data}")

    gender_confidence = gender_data.get('confidence', 0.0)

    # Step 2: Check audio data exists
    if not audio_data or 'pitch_frames' not in audio_data:
        return {'avg_pitch_normalized': 0.0, 'pitch_range_norm': 0.0}

    # Step 3: Extract window frames (Decision Point 4: time-based)
    pitch_fps = audio_data.get('pitch_fps', 43.07)
    pitch_frames = audio_data['pitch_frames']

    start_frame = int(start * pitch_fps)
    end_frame = int(end * pitch_fps)

    # Bounds check
    if end_frame > len(pitch_frames):
        end_frame = len(pitch_frames)

    # Extract window and get voiced frames
    window_pitches = pitch_frames[start_frame:end_frame]
    voiced_pitches = [p for p in window_pitches if p > 80]

    # Step 4: Calculate metrics (Decision Point 16: minimum frames)
    MIN_FRAMES_AVG = 10
    MIN_FRAMES_RANGE = 30

    if len(voiced_pitches) < MIN_FRAMES_AVG:
        # Not enough speech - return zeros
        return {'avg_pitch_normalized': 0.0, 'pitch_range_norm': 0.0}

    # Step 5: Calculate average pitch
    avg_pitch_hz = float(np.mean(voiced_pitches))

    # Step 6: Normalization (Decision Point 3)
    if gender == 'multiple_people':
        # Self-normalization for multi-person videos
        # Use pre-computed statistics for efficiency
        pitch_stats = audio_data.get('pitch_statistics', {})
        if pitch_stats and 'p20' in pitch_stats and 'p80' in pitch_stats:
            p20 = pitch_stats['p20']
            p80 = pitch_stats['p80']
            if p80 > p20:
                avg_pitch_normalized = (avg_pitch_hz - p20) / (p80 - p20)
            else:
                avg_pitch_normalized = 0.0
        else:
            # Fallback to sample if stats not available
            voiced_sample = audio_data.get('voiced_pitch_sample', [])
            if len(voiced_sample) >= 20:
                p20 = np.percentile(voiced_sample, 20)
                p80 = np.percentile(voiced_sample, 80)
                if p80 > p20:
                    avg_pitch_normalized = (avg_pitch_hz - p20) / (p80 - p20)
                else:
                    avg_pitch_normalized = 0.0
            else:
                logger.warning("Insufficient pitch data for self-normalization")
                avg_pitch_normalized = 0.0
    else:
        # Gender-specific normalization
        if gender.lower() == 'male':
            avg_pitch_normalized = (avg_pitch_hz - 110) / 40
        else:  # female
            avg_pitch_normalized = (avg_pitch_hz - 200) / 45

    # Clip to reasonable range
    avg_pitch_normalized = np.clip(avg_pitch_normalized, -1.0, 3.0)

    # Step 7: Calculate range (needs more frames)
    if len(voiced_pitches) >= MIN_FRAMES_RANGE:
        pitch_range_norm = (max(voiced_pitches) - min(voiced_pitches)) / avg_pitch_hz
        pitch_range_norm = min(pitch_range_norm, 1.0)  # Cap at 1.0
    else:
        pitch_range_norm = 0.0

    return {
        'avg_pitch_normalized': round(float(avg_pitch_normalized), 3),
        'pitch_range_norm': round(float(pitch_range_norm), 4)
    }
```

#### 2.2 Modify process_segment Function - CRITICAL INTEGRATION

**Location**: In `process_segment` function, after audio energy calculation

**CRITICAL**: For Direct ML Data services like DeepFace, you MUST also add extraction in the metadata section for unified JSON visibility!

```python
# In process_segment function, after energy metrics calculation:

# Calculate pitch metrics (Decision Points 2, 3, 11)
# Gender is REQUIRED - will raise error if missing
try:
    pitch_metrics = calculate_pitch_metrics(
        audio_data, ml_data, start, end
    )
except ValueError as e:
    # Gender detection failed - this is a pipeline error
    logger.error(f"Pipeline error in segment {start}-{end}: {e}")
    raise  # Fail fast

# Add to segment metrics dictionary:
segment_metrics = {
    # ... existing metrics ...
    'energy_level': energy_level,
    'energy_variance': energy_variance,
    'energy_max': energy_max,
    'burst_pattern': burst_pattern,

    # NEW pitch metrics (Decision Point 2: exactly these two)
    'avg_pitch_normalized': pitch_metrics['avg_pitch_normalized'],
    'pitch_range_norm': pitch_metrics['pitch_range_norm']
}
```

### Phase 3: Data Extraction and Flow

#### 3.1 Audio Data Extraction (No Changes Required)

**File**: `rumiai_v2/processors/temporal_compute.py`

The existing `extract_audio_energy_data` function automatically extracts all fields:

```python
# EXISTING FUNCTION - No changes needed!
def extract_audio_energy_data(ml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract audio data - NOW INCLUDES PITCH automatically"""
    audio_data = ml_data.get('audio_energy', {})
    # Will automatically extract new fields from extended service:
    # - rms_frames (existing)
    # - energy_fps (existing)
    # - pitch_frames (new - added by extended service)
    # - pitch_fps (new - added by extended service)
    # - voiced_pitch_sample (new - max 500 values for memory optimization)
    # - pitch_statistics (new - pre-computed p20, p50, p80, mean, std)
    return audio_data
```

#### 3.2 Gender Data Flow (CRITICAL)

**File**: `rumiai_v2/processors/temporal_compute.py`

Gender detection data must be accessible in two places:

```python
# Lines 1498-1504 in temporal_compute.py (ALREADY IMPLEMENTED)
# Extract gender from ml_data to metadata for visibility
if 'deepface_gender' in ml_data:
    metadata['gender_detection'] = {
        'gender': ml_data['deepface_gender'].get('gender'),
        'confidence': ml_data['deepface_gender'].get('confidence'),
        'method': 'deepface'
    }

# This ensures:
# 1. ml_data['deepface_gender'] - Source of truth from service
# 2. metadata['gender_detection'] - Convenience copy in final JSON
```

### Phase 4: Service Integration

#### 4.1 Pipeline Coordination Updates (Decision Point 11) - ACTUAL IMPLEMENTATION

**File**: `rumiai_v2/processors/video_analyzer.py`

**IMPORTANT**: Uses `DeepFaceGenderServiceSimple` (subprocess isolation), NOT the unused `DeepFaceGenderService`.

**Subprocess Coordination Mechanism**:

The coordination happens in 3 layers:

1. **video_analyzer.py** - Orchestration layer
2. **DeepFaceGenderServiceSimple** - Subprocess management layer
3. **scripts/run_deepface_gender.py** - Actual DeepFace execution

**Layer 1: Orchestration (video_analyzer.py)**
```python
# In video_analyzer.py __init__:
self.deepface_service = None  # Lazy load DeepFace service

# In schedule_analyses method:
if 'deepface_gender' in analyses_to_run:
    analyses['deepface_gender'] = self._run_deepface_analysis(video_path)

async def _run_deepface_analysis(self, video_path: Path) -> Any:
    """Run DeepFace gender detection via subprocess isolation.

    ARCHITECTURE: Uses DeepFaceGenderServiceSimple which calls scripts/run_deepface_gender.py
    in a subprocess to avoid TensorFlow memory corruption.
    NOTE: DeepFaceGenderService (without 'Simple') is DEAD CODE and should be removed.
    """
    try:
        # Lazy load the subprocess service (Simple version is the one actually used)
        if self.deepface_service is None:
            from rumiai_v2.ml_services.deepface_gender_service_simple import (
                DeepFaceGenderServiceSimple
            )
            self.deepface_service = DeepFaceGenderServiceSimple()

        # Run via subprocess (avoids TensorFlow memory corruption)
        result = await self.deepface_service.analyze(str(video_path))

        # Save results
        if result:
            output_path = Path('gender_detection_outputs') / video_id
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / f'{video_id}_gender.json', 'w') as f:
                json.dump(result, f, indent=2)

        return result
    except Exception as e:
        logger.error(f"DeepFace analysis failed: {e}")
        return {'gender': None, 'confidence': 0.0, 'error': str(e)}
```

**Layer 2: Subprocess Management (DeepFaceGenderServiceSimple)**
```python
# rumiai_v2/ml_services/deepface_gender_service_simple.py

class DeepFaceGenderServiceSimple:
    """Manages subprocess execution with proper async coordination."""

    async def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        COORDINATION DETAILS:
        1. Creates subprocess using asyncio.create_subprocess_exec
        2. Waits for completion with 30-second timeout
        3. Returns parsed JSON result or error dict
        """

        # STEP 1: Launch subprocess asynchronously
        process = await asyncio.create_subprocess_exec(
            'python3', 'scripts/run_deepface_gender.py', video_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # STEP 2: Wait for completion with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),  # This blocks until subprocess completes
                timeout=30  # 30 second hard timeout
            )
        except asyncio.TimeoutError:
            # STEP 3a: Handle timeout - kill subprocess
            process.kill()
            await process.wait()  # Clean up zombie process
            return {'gender': None, 'confidence': 0.0, 'error': 'timeout_30s'}

        # STEP 3b: Parse result from subprocess
        if process.returncode == 0:
            result = json.loads(stdout.decode('utf-8'))
            return result
        else:
            # Subprocess failed
            return {'gender': None, 'confidence': 0.0, 'error': stderr.decode('utf-8')}
```

**Coordination Flow Diagram**:
```
Main Process                    Subprocess
------------                    ----------
video_analyzer.py
    |
    v
await deepface_service.analyze()
    |
    v
DeepFaceGenderServiceSimple
    |
    ├─> asyncio.create_subprocess_exec() ──> Launch python3 run_deepface_gender.py
    |                                              |
    ├─> await asyncio.wait_for(30s) <────────────┤ Process video
    |                                              |
    ├─> Parse JSON result <───────────────────────┤ Print JSON to stdout
    |                                              |
    v                                              v
Return result dict                            Exit with code 0 or 1
```

**Layer 3: Subprocess Script**
```python
# scripts/run_deepface_gender.py
# This script is invoked by DeepFaceGenderServiceSimple.analyze() via subprocess.
# It's NOT called directly - the service manages the subprocess execution.

# Key implementation details:
# 1. Face size filtering (>120x120 pixels)
if width >= 120 and height >= 120:
    valid_faces.append(face)

# 2. Multi-person detection
if len(valid_faces) > 1:
    return {'gender': 'multiple_people', 'confidence': 0.0, ...}

# 3. Adaptive frame sampling based on video duration
if duration < 5:
    num_frames = 2
elif duration < 15:
    num_frames = 3
elif duration < 30:
    num_frames = 5
else:
    num_frames = 7
```

### Phase 5: Validation & Benchmarking

#### 5.1 Performance Benchmark (REQUIRED before production)
```python
# benchmark_pitch_extraction.py
import time
import librosa
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_synthetic_speech(duration_seconds: float, sr: int = 22050) -> np.ndarray:
    """Generate synthetic audio with speech-like harmonic content."""
    t = np.linspace(0, duration_seconds, int(sr * duration_seconds))

    # Fundamental frequency variations (100-250 Hz, typical speech range)
    f0_variation = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Varies at 0.5 Hz

    # Generate harmonic series (fundamental + overtones)
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):  # First 5 harmonics
        phase_variation = np.cumsum(2 * np.pi * f0_variation * harmonic / sr)
        amplitude = 1.0 / harmonic  # Natural harmonic decay
        signal += amplitude * np.sin(phase_variation)

    # Add slight noise for realism
    signal += 0.05 * np.random.randn(len(signal))

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    logger.info(f"Generated {duration_seconds}s synthetic speech-like audio")
    return signal

def benchmark_pitch_extraction(duration_seconds: float, audio_path: str = None):
    """
    Validate Decision Point 12 performance estimates with real or synthetic audio.

    NOTE: Production implementation loads audio once and resamples, saving ~30% I/O time.
    This benchmark tests the core pitch extraction performance.
    """

    sr = 22050

    # Try to use real audio if provided, otherwise synthetic
    if audio_path and Path(audio_path).exists():
        logger.info(f"Using real audio file: {audio_path}")
        y, sr = librosa.load(audio_path, sr=sr, duration=duration_seconds)
        audio_type = "real"
    else:
        logger.info(f"Using synthetic speech-like audio ({duration_seconds}s)")
        y = generate_synthetic_speech(duration_seconds, sr)
        audio_type = "synthetic"

    print(f"\nBenchmarking {duration_seconds}s {audio_type} audio:")

    # Benchmark HPSS (Harmonic-Percussive Source Separation)
    start = time.time()
    y_harmonic, _ = librosa.effects.hpss(y)
    hpss_time = time.time() - start

    # Benchmark piptrack (Pitch detection)
    start = time.time()
    pitches, magnitudes = librosa.piptrack(
        y=y_harmonic,
        sr=sr,
        hop_length=512,
        fmin=60,   # Human voice lower bound
        fmax=350   # Human voice upper bound
    )
    piptrack_time = time.time() - start

    # Check if pitches were actually detected
    detected_pitches = np.sum(pitches > 0)

    total_time = hpss_time + piptrack_time
    print(f"  Audio type: {audio_type}")
    print(f"  HPSS: {hpss_time:.2f}s")
    print(f"  Piptrack: {piptrack_time:.2f}s")
    print(f"  Total: {total_time:.2f}s")
    print(f"  Pitches detected: {detected_pitches:,} frames")

    # Validate against Decision Point 12 thresholds
    if total_time > 20:
        print("  ❌ CRITICAL: Exceeds 20s threshold - caching required")
    elif total_time > 15:
        print("  ⚠️  WARNING: Exceeds 15s threshold - investigate optimization")
    else:
        print("  ✅ PASS: Within 15s threshold")

    return {
        'duration': duration_seconds,
        'audio_type': audio_type,
        'hpss_time': hpss_time,
        'piptrack_time': piptrack_time,
        'total_time': total_time,
        'pitches_detected': detected_pitches
    }

# Run benchmarks with fallback
if __name__ == "__main__":
    # Try to find a test audio file
    test_files = list(Path('.').glob('tests/fixtures/*.wav'))
    test_audio = str(test_files[0]) if test_files else None

    if test_audio:
        print(f"Found test audio: {test_audio}")
    else:
        print("No test audio found, will use synthetic speech")

    results = []
    for duration in [15, 30, 60]:
        result = benchmark_pitch_extraction(duration, test_audio)
        results.append(result)

    # Summary
    print("\n=== Performance Summary ===")
    print(f"{'Duration':<10} {'Type':<10} {'Total Time':<12} {'Status'}")
    print("-" * 50)
    for r in results:
        status = "✅" if r['total_time'] < 15 else "⚠️" if r['total_time'] < 20 else "❌"
        print(f"{r['duration']:>8}s  {r['audio_type']:<10} {r['total_time']:>8.2f}s   {status}")
```

#### 5.2 Output Validation
```python
# validate_pitch_output.py
def validate_pitch_metrics(result):
    """Validate pitch metrics are in expected ranges."""

    # Check structure
    assert 'avg_pitch_normalized' in result
    assert 'pitch_range_norm' in result

    # Check ranges (Decision Point 10: clipping)
    assert -1.0 <= result['avg_pitch_normalized'] <= 3.0
    assert 0.0 <= result['pitch_range_norm'] <= 1.0

    # Check types
    assert isinstance(result['avg_pitch_normalized'], float)
    assert isinstance(result['pitch_range_norm'], float)

    print("✅ Pitch metrics valid")

# Test with real video
from rumiai_v2.processors.temporal_compute import calculate_pitch_metrics
result = calculate_pitch_metrics(audio_data, ml_data, 0, 3)
validate_pitch_metrics(result)
```

#### 5.3 Integration Test with Error Handling
```python
# test_pitch_integration.py
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def test_full_pipeline(video_path: str):
    """Complete pipeline test with comprehensive error handling."""

    errors = []

    # 1. Verify gender detection works
    try:
        gender_files = list(Path('gender_detection_outputs').glob('*/gender.json'))
        if not gender_files:
            errors.append("No gender detection output found")
        else:
            with open(gender_files[0]) as f:
                gender_data = json.load(f)
                if gender_data['gender'] not in ['male', 'female', 'multiple_people']:
                    errors.append(f"Invalid gender: {gender_data['gender']}")
                print(f"✅ Gender detection: {gender_data['gender']}")
    except Exception as e:
        errors.append(f"Gender detection error: {e}")

    # 2. Verify pitch extraction with error handling
    try:
        from rumiai_v2.ml_services.audio_energy_service import AudioEnergyAnalyzer
        audio_service = AudioEnergyAnalyzer()
        audio_result = await audio_service.analyze(video_path)

        # Validate required fields
        if 'pitch_frames' not in audio_result:
            errors.append("Missing pitch_frames in audio result")
        if 'pitch_fps' not in audio_result:
            errors.append("Missing pitch_fps in audio result")
        if 'all_voiced_pitches' not in audio_result:
            errors.append("Missing all_voiced_pitches for self-normalization")

        print(f"✅ Pitch extraction: {len(audio_result.get('pitch_frames', []))} frames")
    except RuntimeError as e:
        # Expected errors from service (HPSS fail, no pitches)
        errors.append(f"Pitch extraction failed: {e}")
    except Exception as e:
        # Unexpected errors
        errors.append(f"Unexpected audio service error: {e}")

    # 3. Verify temporal windows with pitch metrics
    try:
        window_files = list(Path('insights').glob('*_temporal_windows_updated.json'))
        if not window_files:
            errors.append("No temporal windows output found")
        else:
            with open(window_files[0]) as f:
                windows = json.load(f)
                hook = windows['temporal_windows']['hook']

                # Validate pitch metrics exist
                if 'avg_pitch_normalized' not in hook:
                    errors.append("Missing avg_pitch_normalized in hook")
                if 'pitch_range_norm' not in hook:
                    errors.append("Missing pitch_range_norm in hook")

                # Validate ranges
                avg_pitch = hook.get('avg_pitch_normalized', 0)
                if not -1.0 <= avg_pitch <= 3.0:
                    errors.append(f"avg_pitch_normalized out of range: {avg_pitch}")

                pitch_range = hook.get('pitch_range_norm', 0)
                if not 0.0 <= pitch_range <= 1.0:
                    errors.append(f"pitch_range_norm out of range: {pitch_range}")

                print(f"✅ Temporal windows: pitch metrics present")
    except Exception as e:
        errors.append(f"Temporal windows error: {e}")

    # Report results
    if errors:
        print("\n❌ Integration test failed with errors:")
        for error in errors:
            print(f"  - {error}")
        raise RuntimeError(f"{len(errors)} errors found")
    else:
        print("\n✅ All integration tests passed!")

# Run test
if __name__ == "__main__":
    test_full_pipeline("test_video.mp4")
```

### Phase 6: Production Deployment

#### 6.1 Pre-Deployment Checklist
```bash
# 1. Verify DeepFace subprocess isolation working
python scripts/run_deepface_gender.py test_video.mp4
# Expected: JSON with gender detection result

# 2. Benchmark pitch extraction performance
python benchmark_pitch_extraction.py
# Expected: <15s for 60s video

# 3. Run integration tests
python test_pitch_integration.py
# Expected: All tests pass

# 4. Memory profiling
python -m memory_profiler rumiai_v2/ml_services/audio_energy_service.py
# Expected: <200MB peak usage

# 5. Verify librosa version
pip show librosa
# Expected: 0.10.1 (pinned version)
```

#### 6.2 Deployment Steps
1. **Code Deployment**:
   ```bash
   # Deploy extended audio_energy_service.py
   # Deploy updated temporal_compute.py with pitch metrics
   # No database migrations needed (JSON storage)
   ```

2. **Configuration**:
   ```python
   # config.yaml - No changes needed, service auto-extends
   ml_services:
     audio_energy:
       enabled: true  # Automatically includes pitch
   ```

3. **Direct to Production**:
   - No feature flags or gradual rollout
   - Pitch metrics become core pipeline feature
   - All new videos get pitch analysis
   - No rollback mechanism (per user requirement)

#### 6.3 Production Monitoring

```python
# monitoring/pitch_metrics_monitor.py
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any
import psutil  # For memory monitoring

logger = logging.getLogger(__name__)

@dataclass
class PitchMetricsMonitor:
    """Production monitoring for pitch extraction performance."""

    # Performance thresholds (Decision Point 12)
    WARNING_THRESHOLD_S = 15
    CRITICAL_THRESHOLD_S = 20
    EMERGENCY_THRESHOLD_S = 30

    def monitor_extraction(self, video_id: str, duration: float, start_time: float):
        """Monitor single extraction and alert if needed."""
        processing_time = time.time() - start_time

        # Component timing breakdown
        components = {
            'audio_load': 0.5,  # Baseline
            'hpss': 0,  # Will be measured
            'piptrack': 0,  # Will be measured
            'aggregation': 0  # Will be measured
        }

        # Alert logic
        if processing_time > self.EMERGENCY_THRESHOLD_S:
            self.alert_emergency(video_id, processing_time, duration)
        elif processing_time > self.CRITICAL_THRESHOLD_S:
            self.alert_critical(video_id, processing_time, duration)
        elif processing_time > self.WARNING_THRESHOLD_S:
            self.alert_warning(video_id, processing_time, duration)

        # Log metrics for dashboard
        self.log_metrics({
            'video_id': video_id,
            'duration_s': duration,
            'processing_time_s': processing_time,
            'fps_achieved': duration / processing_time if processing_time > 0 else 0,
            'memory_peak_mb': self.get_memory_usage_mb()
        })

    def get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to monitoring system."""
        logger.info(f"Pitch metrics: {metrics}")
        # Would also send to DataDog/CloudWatch/etc in production

    def alert_emergency(self, video_id: str, time_s: float, duration: float):
        """Emergency: >30s processing time - immediate action needed."""
        logger.critical(
            f"EMERGENCY: Pitch extraction took {time_s:.1f}s for {duration:.1f}s video. "
            f"Video: {video_id}. ACTION: Reduce quality settings immediately."
        )
        # Trigger PagerDuty/Slack alert

    def alert_critical(self, video_id: str, time_s: float, duration: float):
        """Critical: >20s - implement caching."""
        logger.error(
            f"CRITICAL: Pitch extraction slow: {time_s:.1f}s for {duration:.1f}s video. "
            f"Video: {video_id}. ACTION: Enable Redis caching."
        )

    def alert_warning(self, video_id: str, time_s: float, duration: float):
        """Warning: >15s - investigate."""
        logger.warning(
            f"WARNING: Pitch extraction degraded: {time_s:.1f}s for {duration:.1f}s video. "
            f"Video: {video_id}. ACTION: Investigate performance."
        )
```

#### 6.4 Troubleshooting Guide

```python
# troubleshooting/pitch_diagnostics.py
import logging
from typing import Dict, Any
import librosa

logger = logging.getLogger(__name__)

def diagnose_pitch_failure(video_path: str, error: Exception) -> Dict[str, Any]:
    """Diagnose common pitch extraction failures."""

    diagnostics = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'likely_cause': None,
        'recommended_action': None
    }

    # Common failure patterns
    if "HPSS failed" in str(error):
        diagnostics['likely_cause'] = "Audio file corrupted or unusual format"
        diagnostics['recommended_action'] = "Re-encode audio to standard format"

    elif "No pitches detected" in str(error):
        diagnostics['likely_cause'] = "No voiced speech in audio"
        diagnostics['recommended_action'] = "Check if video has speech content"

    elif "DeepFace data missing" in str(error):
        diagnostics['likely_cause'] = "Gender detection service failed"
        diagnostics['recommended_action'] = "Check DeepFace subprocess logs"

    elif "Memory" in str(error):
        diagnostics['likely_cause'] = "Insufficient memory for long video"
        diagnostics['recommended_action'] = "Enable streaming mode for videos >5min"

    # Run additional checks
    import librosa
    try:
        # Test basic audio load
        y, sr = librosa.load(video_path, duration=1)
        diagnostics['audio_loadable'] = True
        diagnostics['sample_rate'] = sr
        diagnostics['audio_length_s'] = len(y) / sr
    except:
        diagnostics['audio_loadable'] = False
        diagnostics['recommended_action'] = "Fix audio extraction first"

    return diagnostics

# Usage in error handler:
# if pitch_extraction_fails:
#     diagnostics = diagnose_pitch_failure(video_path, error)
#     logger.error(f"Pitch extraction diagnostics: {diagnostics}")
```

#### 6.5 Performance Optimization

```python
# NOTE: Caching removed - not applicable for single-video processing
"""
REMOVED: Video result caching strategy

The original implementation included a cache for pitch analysis results,
but this doesn't make sense for our architecture because:

1. We process videos one at a time (not in batches)
2. The same video is rarely processed twice
3. Each video gets unique pitch analysis

If performance becomes an issue, consider these alternatives:
- Cache model loading (one-time cost per session)
- Use faster pitch extraction algorithms
- Process in lower quality mode for previews
"""

# Simple performance monitoring without caching
def log_performance_metrics(video_id: str, processing_time: float):
    """Log performance for monitoring - no caching needed."""
    if processing_time > 20:
        logger.critical(f"Video {video_id} took {processing_time:.1f}s - investigate!")
    elif processing_time > 15:
        logger.warning(f"Video {video_id} took {processing_time:.1f}s - degraded performance")
    else:
        logger.info(f"Video {video_id} processed in {processing_time:.1f}s")
```

#### 6.6 Runtime Validation

```python
# NOTE: Daily validation removed - not needed with fail-fast architecture
"""
REMOVED: Daily output validation

The original implementation included daily sampling and validation of outputs,
but this is unnecessary because:

1. Our fail-fast design means errors stop the pipeline immediately
2. Invalid values cause exceptions in calculate_pitch_metrics
3. Each video runs independently - no drift or degradation over time
4. No silent failures possible - all errors are loud (exceptions)

Runtime validation happens automatically through:
- Value clipping in calculate_pitch_metrics (ensures ranges)
- Exception raising for missing data
- DeepFace requirement (pipeline fails if gender detection fails)
"""

# Integration tests are sufficient for validation
def test_pitch_metrics_integration():
    """
    Run during development/testing, not in production.
    Our fail-fast architecture makes runtime validation redundant.
    """
    # Test with known inputs
    test_audio_data = {
        'pitch_frames': [150] * 50,
        'pitch_fps': 43.07,
        'pitch_statistics': {'p20': 140, 'p80': 160}
    }

    test_ml_data = {
        'deepface_gender': {'gender': 'male', 'confidence': 0.95}
    }

    # This will raise exceptions if anything is wrong
    result = calculate_pitch_metrics(test_audio_data, test_ml_data, 0, 1)

    # Assertions enforce correctness at test time
    assert -1.0 <= result['avg_pitch_normalized'] <= 3.0
    assert 0.0 <= result['pitch_range_norm'] <= 1.0

    print("✅ Pitch metrics integration test passed")
```

#### 6.7 Emergency Response Procedures

```python
# emergency_response.py
"""
Generic emergency response for pitch extraction performance issues.
Infrastructure-agnostic - works with any deployment.
"""
import os
import logging
import time

logger = logging.getLogger(__name__)

class PitchEmergencyResponse:
    """Handle pitch extraction performance degradation."""

    def __init__(self):
        self.quality_level = "high"
        self.pitch_required = True

    def activate_emergency_mode(self, avg_processing_time: float):
        """Activate appropriate emergency response based on severity."""

        if avg_processing_time > 30:
            logger.critical("EMERGENCY: Activating bypass mode")
            self.bypass_pitch_extraction()
        elif avg_processing_time > 20:
            logger.error("CRITICAL: Reducing to low quality")
            self.reduce_quality("low")
        elif avg_processing_time > 15:
            logger.warning("WARNING: Reducing to medium quality")
            self.reduce_quality("medium")

    def bypass_pitch_extraction(self):
        """Allow pipeline to continue without pitch metrics."""
        self.pitch_required = False
        os.environ['PITCH_REQUIRED'] = 'false'
        logger.info("Pitch extraction bypassed - returning zeros")

    def reduce_quality(self, level: str):
        """Reduce pitch extraction quality for speed."""
        self.quality_level = level
        os.environ['PITCH_QUALITY'] = level

        # Adjust parameters based on quality level
        if level == "low":
            # Increase hop_length for faster processing
            os.environ['PITCH_HOP_LENGTH'] = '1024'  # Double the hop
            logger.info("Pitch quality set to LOW - 2x faster")
        elif level == "medium":
            os.environ['PITCH_HOP_LENGTH'] = '768'
            logger.info("Pitch quality set to MEDIUM - 1.5x faster")

    def gradual_recovery(self):
        """Gradually restore quality as performance improves."""
        logger.info("Starting gradual recovery...")

        # Monitor and adjust every 30 minutes
        time.sleep(1800)  # 30 minutes
        if self.get_avg_processing_time() < 15:
            self.reduce_quality("medium")

        time.sleep(1800)  # Another 30 minutes
        if self.get_avg_processing_time() < 12:
            self.reduce_quality("high")
            logger.info("Fully recovered to HIGH quality")

    def get_avg_processing_time(self) -> float:
        """Get recent average processing time."""
        # In production, query from monitoring system
        # For now, return mock value
        return 10.0

# Usage in main pipeline
def handle_pitch_performance(processing_time: float):
    """Called after each video to check performance."""
    if processing_time > 15:
        emergency = PitchEmergencyResponse()
        emergency.activate_emergency_mode(processing_time)
```

#### 6.8 Backward Compatibility
- **Missing pitch data**: Returns zeros (not errors)
- **Existing videos**: Continue working without pitch metrics
- **Gradual adoption**: New videos get pitch, old videos unchanged
- **No breaking changes**: Service name unchanged (audio_energy)

### Phase 7: Testing Strategy

#### 7.1 Unit Tests
```python
# tests/test_pitch_extraction.py
import pytest
import numpy as np
from rumiai_v2.ml_services.audio_energy_service import AudioEnergyAnalyzer

class TestPitchExtraction:
    """Unit tests for pitch extraction functionality."""

    @pytest.fixture
    def analyzer(self):
        return AudioEnergyAnalyzer()

    def test_pitch_extraction_success(self, analyzer, sample_audio_file):
        """Test successful pitch extraction."""
        result = await analyzer.analyze(sample_audio_file)

        assert 'pitch_frames' in result
        assert 'pitch_fps' in result
        assert result['pitch_fps'] == pytest.approx(43.07, rel=0.01)
        assert len(result['pitch_frames']) > 0

    def test_silent_audio_handling(self, analyzer, silent_audio_file):
        """Test handling of silent audio."""
        result = await analyzer.analyze(silent_audio_file)

        # Should not fail, but return empty/zero pitches
        assert 'pitch_frames' in result
        assert all(p == 0 for p in result['pitch_frames'][:100])

    def test_corrupted_audio_fails_fast(self, analyzer, corrupted_file):
        """Test fail-fast on corrupted audio."""
        with pytest.raises(RuntimeError, match="(HPSS failed|value error)"):
            await analyzer.analyze(corrupted_file)

    def test_invalid_parameters_handled(self, analyzer):
        """Test handling of invalid parameters."""
        analyzer.pitch_hop = -1  # Invalid hop length
        with pytest.raises(RuntimeError, match="invalid parameters"):
            await analyzer.analyze("test.wav")

    def test_memory_optimization(self, analyzer, long_audio_file):
        """Test memory usage stays within bounds."""
        import tracemalloc
        tracemalloc.start()

        result = await analyzer.analyze(long_audio_file)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should use <200MB for 5-minute video
        assert peak / 1024 / 1024 < 200  # MB
```

#### 7.2 Integration Tests
```python
# tests/test_pitch_integration.py
import json
from pathlib import Path
from rumiai_v2.processors.temporal_compute import calculate_pitch_metrics

class TestPitchIntegration:
    """Integration tests for pitch in temporal windows."""

    def test_gender_normalization(self):
        """Test gender-specific normalization."""
        # Male normalization
        male_ml_data = {
            'deepface_gender': {'gender': 'male', 'confidence': 0.95}
        }
        male_audio = {
            'pitch_frames': [110] * 50,  # At male baseline
            'pitch_fps': 43.07
        }

        result = calculate_pitch_metrics(male_audio, male_ml_data, 0, 1)
        assert result['avg_pitch_normalized'] == pytest.approx(0.0, abs=0.01)

        # Female normalization
        female_ml_data = {
            'deepface_gender': {'gender': 'female', 'confidence': 0.90}
        }
        female_audio = {
            'pitch_frames': [200] * 50,  # At female baseline
            'pitch_fps': 43.07
        }

        result = calculate_pitch_metrics(female_audio, female_ml_data, 0, 1)
        assert result['avg_pitch_normalized'] == pytest.approx(0.0, abs=0.01)

    def test_multi_person_self_normalization(self):
        """Test self-normalization for multi-person videos."""
        ml_data = {
            'deepface_gender': {'gender': 'multiple_people', 'confidence': 0.0}
        }
        audio_data = {
            'pitch_frames': list(range(100, 200)),
            'pitch_fps': 43.07,
            'all_voiced_pitches': list(range(80, 250))
        }

        result = calculate_pitch_metrics(audio_data, ml_data, 0, 2)

        # Should use percentile-based normalization
        assert 0.0 <= result['avg_pitch_normalized'] <= 1.0

    def test_minimum_frames_requirement(self):
        """Test minimum frames for reliable metrics."""
        ml_data = {
            'deepface_gender': {'gender': 'male', 'confidence': 0.9}
        }

        # Too few frames for range calculation
        short_audio = {
            'pitch_frames': [120, 125, 130],  # Only 3 frames
            'pitch_fps': 43.07
        }

        result = calculate_pitch_metrics(short_audio, ml_data, 0, 0.07)
        assert result['avg_pitch_normalized'] == 0.0  # Not enough frames
        assert result['pitch_range_norm'] == 0.0  # Definitely not enough
```

#### 7.3 End-to-End Tests
```python
# tests/test_pitch_e2e.py
import subprocess
import json
from pathlib import Path

def test_full_pipeline_with_pitch():
    """End-to-end test of complete pipeline with pitch."""

    test_video = "tests/fixtures/sample_video.mp4"

    # Run full pipeline
    result = subprocess.run(
        ["python", "rumiai_v2/main.py", test_video],
        capture_output=True,
        text=True,
        timeout=60
    )

    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

    # Check output file
    video_id = Path(test_video).stem
    output_path = Path(f"insights/{video_id}_temporal_windows_updated.json")
    assert output_path.exists(), "Output file not created"

    # Validate pitch metrics in output
    with open(output_path) as f:
        data = json.load(f)

    # Check all temporal windows have pitch
    hook = data['temporal_windows']['hook']
    assert 'avg_pitch_normalized' in hook
    assert 'pitch_range_norm' in hook

    for segment in data['temporal_windows']['middle_segments']:
        assert 'avg_pitch_normalized' in segment
        assert 'pitch_range_norm' in segment

    closing = data['temporal_windows']['closing']
    assert 'avg_pitch_normalized' in closing
    assert 'pitch_range_norm' in closing

    # Check gender detection present
    assert 'gender_detection' in data['metadata']

    print("✅ E2E test passed - pitch metrics integrated successfully")
```

### Phase 8: Documentation & Training

#### 8.1 API Documentation
```python
"""
Pitch Metrics API Documentation

New fields added to temporal windows:

avg_pitch_normalized : float
    Gender-normalized average pitch deviation.
    Range: [-1.0, 3.0]
    - Negative: Below speaker's baseline (serious, calm)
    - Zero: At speaker's typical pitch or no speech
    - Positive: Above baseline (excited, questioning)
    - >1.0: Extreme emotion or shouting

    Normalization formulas:
    - Male: (pitch_hz - 110) / 40
    - Female: (pitch_hz - 200) / 45
    - Multiple people: Percentile-based self-normalization

pitch_range_norm : float
    Normalized pitch variation (max - min) / mean.
    Range: [0.0, 1.0]
    - <0.10: Monotone delivery
    - 0.10-0.35: Natural conversational variation
    - >0.35: Highly expressive or animated speech
    - 0.0: No speech or insufficient voiced frames (<30)

Example JSON structure:
{
  "temporal_windows": {
    "hook": {
      "avg_pitch_normalized": 0.75,
      "pitch_range_norm": 0.38,
      ... other metrics ...
    }
  }
}
"""
```

#### 8.2 Migration Guide
```markdown
# Pitch Metrics Migration Guide

## For Existing Systems
No action required - pitch metrics are additive and backward compatible.

## For ML Models
1. New features available:
   - `avg_pitch_normalized`: Use for emotion/energy detection
   - `pitch_range_norm`: Use for expressiveness scoring

2. Feature engineering suggestions:
   ```python
   # Combine with energy for compound features
   excitement_score = (
       window['avg_pitch_normalized'] * 0.4 +
       window['energy_level'] * 0.6
   )

   # Detect questions vs statements
   is_question = window['avg_pitch_normalized'] > 0.5 and position == 'closing'
   ```

## For Content Creators
New insights available:
- "Your hook pitch is 25% below average - try starting with more energy"
- "Your closing has flat pitch (range: 0.08) - add emphasis to your CTA"
```

### Phase 9: Post-Deployment Validation

#### 9.1 Success Metrics Tracking
```python
# metrics/pitch_success_tracker.py
import logging
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def track_pitch_metrics_adoption(days_since_launch: int):
    """Track adoption and success of pitch metrics."""

    metrics = {
        'coverage': 0,  # % of videos with pitch metrics
        'quality': 0,   # % passing validation
        'performance': 0,  # Avg processing time
        'ml_improvement': 0  # Engagement prediction improvement
    }

    # 1. Coverage metric
    total_videos = count_videos_processed(days_since_launch)
    with_pitch = count_videos_with_pitch(days_since_launch)
    metrics['coverage'] = (with_pitch / total_videos) * 100 if total_videos > 0 else 0

    # 2. Quality metric
    validation_results = run_validation_on_sample()
    metrics['quality'] = validation_results.get('pass_rate', 0)

    # 3. Performance metric
    processing_times = get_processing_times(days_since_launch)
    metrics['performance'] = np.mean(processing_times) if processing_times else 0

    # 4. ML improvement (A/B test)
    baseline_accuracy = 0.72  # Before pitch
    with_pitch_accuracy = measure_current_accuracy()
    metrics['ml_improvement'] = (
        (with_pitch_accuracy - baseline_accuracy) / baseline_accuracy * 100
    )

    # Success criteria
    SUCCESS_THRESHOLDS = {
        'coverage': 95,  # 95% videos have pitch
        'quality': 99,   # 99% pass validation
        'performance': 15,  # <15s average
        'ml_improvement': 5  # >5% accuracy gain
    }

    for metric, value in metrics.items():
        threshold = SUCCESS_THRESHOLDS[metric]
        status = "✅" if value >= threshold else "❌"
        print(f"{status} {metric}: {value:.1f} (target: {threshold})")

    return all(
        metrics[k] >= v for k, v in SUCCESS_THRESHOLDS.items()
    )

# Helper functions for metrics collection
def count_videos_processed(days: int) -> int:
    """Count total videos processed in last N days."""
    # In production, query from database
    from pathlib import Path
    cutoff = datetime.now() - timedelta(days=days)
    videos = Path('insights').glob('*_temporal_windows_updated.json')
    return sum(1 for v in videos if v.stat().st_mtime > cutoff.timestamp())

def count_videos_with_pitch(days: int) -> int:
    """Count videos with pitch metrics in last N days."""
    # In production, query from database
    count = 0
    from pathlib import Path
    cutoff = datetime.now() - timedelta(days=days)
    for video_file in Path('insights').glob('*_temporal_windows_updated.json'):
        if video_file.stat().st_mtime > cutoff.timestamp():
            with open(video_file) as f:
                data = json.load(f)
                if 'avg_pitch_normalized' in data.get('temporal_windows', {}).get('hook', {}):
                    count += 1
    return count

def get_processing_times(days: int) -> list:
    """Get pitch processing times from last N days."""
    # In production, query from metrics database
    # For now, return mock data
    return [8.5, 9.2, 10.1, 11.3, 8.9, 12.5, 9.8]

def run_validation_on_sample() -> Dict[str, Any]:
    """Run validation on sample of recent videos."""
    # Calls verify_pitch_output from earlier
    return {'pass_rate': 98.5}  # Mock for now

def measure_current_accuracy() -> float:
    """Measure current ML model accuracy with pitch features."""
    # In production, run evaluation on test set
    return 0.76  # Mock improvement
```

#### 9.2 Rollback Procedures (Not Applicable)
```python
# Per user requirement: "We do not need rollback. Its all in beibi"
# This section intentionally left as documentation only

"""
NO ROLLBACK PLAN - This is a permanent addition to the pipeline.
Once deployed, pitch metrics are part of core functionality.

If issues arise:
1. Fix forward (patch the issue)
2. Use feature degradation (return zeros) if needed
3. But never remove the fields from output
"""
```

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