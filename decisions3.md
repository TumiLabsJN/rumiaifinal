# Decisions for Missing Implementation Details

These decisions address the missing pieces needed to execute the refactortemporal.md plan.

## Point 1: Complete extract_timelines_for_temporal() Implementation

**Issue**: The current extraction function in refactortemporal.md is incomplete. We need to verify the actual ML service output structure.

**Current Code**:
```python
timelines['gesture_timeline'] = mediapipe_data.get('gestures', [])
timelines['expression_timeline'] = mediapipe_data.get('expressions', [])
```

**Questions to resolve**:
1. What is the actual structure of ML service outputs?
2. What keys do they use?
3. How are timestamps formatted?

**Finding**: 
I found the existing extraction functions in precompute_functions.py. They show ML data can be in two formats:
1. Direct: `ml_data['ocr']['textAnnotations']`
2. Nested: `ml_data['ocr']['data']['textAnnotations']`

The existing code handles both formats defensively.

**Decision 1: Use Defensive Extraction Approach**

Copy the defensive extraction pattern that handles both formats (direct and nested).

**Rationale**:
- Production robustness - handles format variations without breaking
- Already proven in current code
- Minimal overhead (3-4 extra lines per service)
- Maintains self-contained architecture
- Prevents runtime failures from format changes

**Implementation**:
```python
def extract_timelines_for_temporal(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    ml_data = analysis_dict.get('ml_data', {})
    timelines = {}
    
    # OCR - defensive extraction
    ocr_data = ml_data.get('ocr', {})
    if 'textAnnotations' in ocr_data:
        annotations = ocr_data['textAnnotations']
    elif 'data' in ocr_data and 'textAnnotations' in ocr_data['data']:
        annotations = ocr_data['data']['textAnnotations']
    else:
        annotations = []
    
    timelines['text_overlay_timeline'] = [
        {'timestamp': ann.get('timestamp', 0), 'text': ann.get('text', '')}
        for ann in annotations
    ]
    
    # Similar defensive patterns for other services...
```

---

## Point 2: Audio Energy Function Modification

**Issue**: Need exact implementation for calculate_audio_energy_for_windows using RMS frames from ML data (Decision 2 from decisions2.md).

**Current function signature**:
```python
def calculate_audio_energy_for_windows(
    audio_path: Optional[Path], 
    video_duration: float
) -> Dict[str, Any]
```

**Needs to become**:
```python
def calculate_audio_energy_for_windows(
    ml_data: Dict,
    video_duration: float  
) -> Dict[str, Any]
```

**Decision 2A: Extend Output Format with Burst Patterns**

Extend the output format to include burst patterns per window as required by Decision 2 in refactortemporal.md.

**Output format**:
```python
{
    'hook_energy_level': float,
    'hook_energy_variance': float,
    'hook_energy_max': float,
    'hook_burst_pattern': str,  # NEW: "front_loaded", "back_loaded", "middle_peak", or "steady"
    'middle_energy_level': float,
    'middle_energy_variance': float,
    'middle_energy_max': float,
    'middle_burst_pattern': str,  # NEW
    'closing_energy_level': float,
    'closing_energy_variance': float,
    'closing_energy_max': float,
    'closing_burst_pattern': str  # NEW
}
```

**Rationale**:
- Required by Decision 2 (incorporated in refactortemporal.md)
- P0TemporalWindows.md spec requires complete energy metrics
- Maintains clean naming pattern: `{window_name}_{metric_type}`
- Provides richer temporal energy features for ML

**Decision 2B: Return Zeros for Missing RMS Frames**

Return zeros for all metrics when RMS frames are missing (Option A).

**Rationale**:
- Aligns with refactortemporal.md philosophy: "accept empty results"
- Audio energy service must run (validated) but can have empty data
- Silent videos are legitimate use cases
- Consistent with how whisper/speech segments are handled
- Log as info, not error

**Decision 2C: Calculate Burst Patterns Per Window**

Calculate burst patterns from RMS frames for each temporal window.

**Rationale**:
- Required by Decision 2 in refactortemporal.md
- Global burst pattern doesn't capture per-window dynamics
- Enables ML to learn window-specific energy patterns
- Can reuse logic from audio_energy_service's `_determine_burst_pattern` method

**Implementation approach**:
```python
def calculate_burst_pattern_for_window(window_rms: np.array) -> str:
    """Determine burst pattern for a specific window's RMS frames"""
    if len(window_rms) < 2:
        return "steady"
    
    # Divide window into thirds and compare energy levels
    third_size = len(window_rms) // 3
    front_avg = np.mean(window_rms[:third_size])
    middle_avg = np.mean(window_rms[third_size:2*third_size])
    back_avg = np.mean(window_rms[2*third_size:])
    
    # Determine pattern based on which third has highest energy
    # Returns: "front_loaded", "back_loaded", "middle_peak", or "steady"
```

---

## Point 3: Integration with rumiai_runner.py

**Issue**: How will temporal_compute.py be called from rumiai_runner.py?

**Current situation**:
- rumiai_runner.py calls functions from COMPUTE_FUNCTIONS registry
- temporal_compute is not in the registry
- We want to replace all 7 analyses with just temporal_windows

**Decision 3: Direct Integration (No Registry)**

Call temporal_compute directly from rumiai_runner.py, bypassing COMPUTE_FUNCTIONS entirely.

**Rationale** (from refactortemporal.md):
- "No integration with COMPUTE_FUNCTIONS needed"
- "Update rumiai_runner.py: Use only temporal_compute"
- "Delete old code: Remove precompute_functions.py"
- Single output replaces 21 files

**Implementation**:
```python
# In rumiai_runner.py - REPLACE the analysis loop
# OLD:
for analysis_name in COMPUTE_FUNCTIONS:
    result = COMPUTE_FUNCTIONS[analysis_name](analysis_dict)
    save_analysis_result(result, analysis_name)

# NEW:
from rumiai_v2.processors.temporal_compute import compute_temporal_windows
result = compute_temporal_windows(analysis_dict)
save_path = output_dir / f"{video_id}_temporal_windows.json"
with open(save_path, 'w') as f:
    json.dump(result, f, indent=2)
```

**No COMPUTE_FUNCTIONS registry needed** - entire precompute_functions.py will be deleted.

---

## Point 4: process_segment Function

**Issue**: The refactored main function references `process_segment()` but it's not defined.

From refactortemporal.md line 400:
```python
seg_data = process_segment(seg_bounds, timelines, speech_segments, audio_energy)
```

**Decision 4: Calculate Full Metrics for All Windows**

Calculate the SAME comprehensive metrics for middle segments as for hook/closing windows.

**Issue**: Spec shows limited metrics for segments but this creates inconsistent ML features.

**Decision**: All windows and segments get identical metric sets for ML consistency.

**Full metrics for EVERY window/segment**:
1. Basic counts (text_count, element_count, total_elements)
2. Density metrics (element_density, changes_per_second)
3. Speech metrics (speech_coverage, word_count)
4. Emotion distribution (happy_ratio, sad_ratio, surprised_ratio, etc.)
5. Framing distribution (closeup_ratio, medium_ratio, wide_ratio)
6. Variance metrics (gaze_variance, pacing_variation, scene_duration_variance)
7. Diversity metrics (vocabulary_diversity)
8. Audio energy (energy_level, energy_variance, energy_max, burst_pattern)

**Implementation**:
```python
def process_segment(seg_bounds: Dict, timelines: Dict, 
                   speech_segments: List, audio_energy: Dict) -> Dict:
    """Process segment with FULL metrics matching hook/closing windows"""
    start = seg_bounds['start']
    end = seg_bounds['end']
    
    # Calculate ALL the same metrics as hook/closing windows
    # Reuse all existing helper functions
    
    return {
        'start': start,
        'end': end,
        'duration': end - start,
        # All metrics identical to main windows
        **element_counts,
        **density_metrics,
        'speech_coverage': speech_coverage,
        'word_count': word_count,
        **emotion_dist,
        **framing_dist,
        'gaze_variance': gaze_var,
        'pacing_variation': pacing_var,
        'scene_duration_variance': scene_var,
        'vocabulary_diversity': vocab_div,
        **energy_metrics
    }
```

**Rationale**:
- **ML consistency**: Same features across all temporal positions
- **No information loss**: Capture all signals for every time window
- **Fair comparison**: Model can learn patterns without position bias
- **Reuses all helpers**: Leverages existing tested functions

---

## Point 5: ML Data Structure Verification

**Decision 5: Mixed Extraction Strategy - Timeline Entries and ML Data**

Extract data from appropriate sources based on where each service stores its results.

**Discovery**:
- Some data comes from timeline entries (emotions, scenes)
- Some data comes directly from ml_data (OCR text, stickers, MediaPipe, YOLO, Whisper, audio)
- Current architecture varies by service type

**Implementation**:
```python
def extract_timelines_for_temporal(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    ml_data = analysis_dict.get('ml_data', {})
    timeline = analysis_dict.get('timeline', {})
    timelines = {}
    
    # 1. Extract from TIMELINE ENTRIES
    
    # Emotion data from timeline entries (FEAT results)
    expression_timeline = []
    for entry in timeline.get('entries', []):
        if entry.get('entry_type') == 'emotion':
            expression_timeline.append({
                'timestamp': entry.get('start', 0),
                'emotion': entry.get('data', {}).get('emotion', 'neutral'),
                'confidence': entry.get('data', {}).get('confidence', 0),
                'source': 'feat'
            })
    timelines['expression_timeline'] = expression_timeline
    
    # Scene data from timeline entries (PySceneDetect results)
    scene_change_timeline = []
    for entry in timeline.get('entries', []):
        if entry.get('entry_type') in ['scene_change', 'scene']:
            scene_change_timeline.append({
                'timestamp': entry.get('start', 0),
                'scene_number': entry.get('data', {}).get('scene_number', 0),
                'confidence': entry.get('data', {}).get('confidence', 0),
                'source': 'pyscenedetect'
            })
    timelines['scene_change_timeline'] = scene_change_timeline
    
    # 2. Extract from ML_DATA directly
    
    # OCR data (text and stickers) - defensive extraction
    ocr_data = ml_data.get('ocr', {})
    if 'textAnnotations' in ocr_data:
        annotations = ocr_data['textAnnotations']
    elif 'data' in ocr_data and 'textAnnotations' in ocr_data['data']:
        annotations = ocr_data['data']['textAnnotations']
    else:
        annotations = []
    
    # Text overlays
    timelines['text_overlay_timeline'] = [
        {'timestamp': ann.get('timestamp', 0), 'text': ann.get('text', '')}
        for ann in annotations
    ]
    
    # Stickers (from OCR service, inline detection)
    timelines['sticker_timeline'] = ocr_data.get('stickers', [])
    
    # Whisper speech segments from ml_data
    whisper_data = ml_data.get('whisper', {})
    timelines['speech_segments'] = whisper_data.get('segments', [])
    
    # YOLO object detections from ml_data
    yolo_data = ml_data.get('yolo', {})
    timelines['object_timeline'] = yolo_data.get('objectAnnotations', [])
    
    # MediaPipe raw data from ml_data
    mediapipe_data = ml_data.get('mediapipe', {})
    timelines['pose_timeline'] = mediapipe_data.get('poses', [])
    timelines['face_timeline'] = mediapipe_data.get('faces', [])
    # Note: MediaPipe gaze is raw, but we also need timeline gaze entries
    
    # Audio energy RMS frames from ml_data (Decision 2)
    audio_data = ml_data.get('audio_energy', {})
    timelines['rms_frames'] = audio_data.get('rms_frames', [])
    timelines['frames_per_second'] = audio_data.get('frames_per_second', 30)
    
    # 3. MORE from TIMELINE ENTRIES
    
    # Gesture data from timeline entries (processed from MediaPipe)
    gesture_timeline = []
    for entry in timeline.get('entries', []):
        if entry.get('entry_type') == 'gesture':
            gesture_timeline.append({
                'timestamp': entry.get('start', 0),
                'gesture': entry.get('data', {}).get('gesture', 'unknown'),
                'confidence': entry.get('data', {}).get('confidence', 0)
            })
    timelines['gesture_timeline'] = gesture_timeline
    
    # Gaze data from timeline entries (processed from MediaPipe)
    gaze_timeline = []
    for entry in timeline.get('entries', []):
        if entry.get('entry_type') == 'gaze':
            gaze_timeline.append({
                'timestamp': entry.get('start', 0),
                'looking_at_camera': entry.get('data', {}).get('looking_at_camera', False),
                'confidence': entry.get('data', {}).get('confidence', 0)
            })
    timelines['gaze_timeline'] = gaze_timeline
    
    return timelines
```

**Rationale**:
- Aligns with current data flow architecture
- Timeline entries are the processed, authoritative source
- FEAT and PySceneDetect data aren't directly in ml_data
- Consistent with how existing precompute_functions work

**Services Actually in ML Data**:
1. **ocr** - text annotations AND stickers (inline detection during OCR)
2. **mediapipe** - poses, faces, hands, gaze, gestures (NOT emotions)
3. **yolo** - object detection
4. **whisper** - speech transcription
5. **audio_energy** - RMS frames, energy metrics

**Data From Timeline Entries**:
- **emotion** - FEAT emotion detection results
- **scene_change/scene** - PySceneDetect scene boundaries
- **gesture** - Processed gesture recognition from MediaPipe
- **gaze** - Processed gaze detection from MediaPipe

**Complete Extraction Summary Table**:

| Data Type | Source | Location | Format |
|-----------|--------|----------|--------|
| Text annotations | ml_data | `ml_data['ocr']['textAnnotations']` | Array of objects |
| Stickers | ml_data | `ml_data['ocr']['stickers']` | Array of objects |
| Speech segments | ml_data | `ml_data['whisper']['segments']` | Array of segments |
| Object detections | ml_data | `ml_data['yolo']['objectAnnotations']` | Array of objects |
| Pose landmarks | ml_data | `ml_data['mediapipe']['poses']` | Array of poses |
| Face landmarks | ml_data | `ml_data['mediapipe']['faces']` | Array of faces |
| RMS frames | ml_data | `ml_data['audio_energy']['rms_frames']` | Array of floats |
| Emotions | timeline | `timeline.entries` with `entry_type: 'emotion'` | FEAT processed |
| Scene changes | timeline | `timeline.entries` with `entry_type: 'scene_change'` | PySceneDetect |
| Gestures | timeline | `timeline.entries` with `entry_type: 'gesture'` | Processed |
| Gaze | timeline | `timeline.entries` with `entry_type: 'gaze'` | Processed |

**Sticker Detection Key Findings** (from VisualOverlay.md):
1. Stickers are detected inline during OCR processing (not a separate service)
2. Uses HSV color space analysis (high saturation = graphics/stickers)
3. Adds only 3-5ms overhead per frame
4. Stored in OCR results under 'stickers' key alongside 'textAnnotations'

**Scene Detection Key Findings** (from ScenePacing.md):
1. Scene detection runs in ml_services.py (not ml_services_unified.py)
2. Uses PySceneDetect with adaptive thresholds [20.0, 15.0, 10.0]
3. Data flow:
   - PySceneDetect → scene data
   - → Timeline builder (creates scene_change and scene entries)
   - → Timeline extraction → sceneChangeTimeline
   - → temporal_compute

This completes our ML data structure verification and clarifies the extraction approach.