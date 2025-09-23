# Post-Refactor Data Flow Architecture

## Overview
After the temporal windows refactor, RumiAI v2 uses a single compute function (`compute_temporal_windows`) that replaces 7 separate analysis functions. The system employs a **mixed data strategy** with three distinct flow patterns.

---

## Data Flow Patterns

### 1. Timeline-Based Flow
**Path:** `ML Services → TimelineBuilder → timeline.entries → compute_temporal_windows()`

Services that create semantic timeline entries:

| Service | Timeline Entry Types | Description |
|---------|---------------------|-------------|
| **Emotion Detection** | `emotion` | FEAT-detected emotions with confidence scores |
| **Scene Detection** | `scene_change`, `scene` | PySceneDetect boundary events |
| **MediaPipe** | `pose`, `gaze`, `face`, `gesture` | Human behavior interpretations |
| **OCR** | `text`, `sticker` | Text overlay and sticker appearances |
| **Whisper** | `speech` | Speech transcription segments |
| **YOLO** | `object` | Object detection events |

**Timeline Entry Structure:**
```python
{
    'entry_type': 'emotion',  # or 'gesture', 'scene_change', etc.
    'start': float,           # Timestamp in seconds
    'end': float,             # Optional end time
    'data': {                 # Entry-specific payload
        'emotion': 'joy',
        'confidence': 0.85
    }
}
```

### 2. Direct ML Data Flow
**Path:** `ML Services → ml_data → compute_temporal_windows()`

Data that bypasses timeline processing:

| Data Type | Location | Reason for Bypass |
|-----------|----------|-------------------|
| **Audio Energy** | `ml_data['audio_energy']` | Continuous signal (RMS frames), not discrete events |
| **Video Metadata** | `metadata` dict | Video-level attributes (likes, views, duration) |
| **Raw Detections** | `ml_data[service_name]` | Coordinates, landmarks, bounding boxes |

**Audio Energy Structure:**
```python
ml_data['audio_energy'] = {
    'rms_frames': [...],        # Raw RMS values per frame
    'frames_per_second': 30.0,  # Frame rate
    'energy_variance': 0.0,
    'burst_pattern': 'steady'   # or 'front_loaded', 'back_loaded', 'middle_peak'
}
```

### 3. Mixed Strategy Flow
**Path:** Both timeline AND ml_data paths used together

Services providing both raw data and semantic interpretations:

```python
# Raw detection data (bypasses timeline)
ml_data['mediapipe'] = {
    'poses': [...],   # Raw pose landmarks
    'faces': [...],   # Face bounding boxes
    'hands': [...]    # Hand landmarks
}

# Semantic interpretations (uses timeline)
timeline.entries = [
    {'entry_type': 'gesture', 'data': {'gesture': 'pointing'}},
    {'entry_type': 'gaze', 'data': {'looking_at_camera': True}}
]
```

---

## Complete Processing Pipeline

### Step 1: ML Service Execution
```python
# video_analyzer.py (lines 39-47)
analyses = {
    'yolo': self._run_yolo,
    'whisper': self._run_whisper,
    'mediapipe': self._run_mediapipe,
    'ocr': self._run_ocr,
    'scene_detection': self._run_scene_detection,
    'audio_energy': self._run_audio_energy,
    'emotion_detection': self._run_emotion_detection
}
```

### Step 2: Timeline Building
```python
# rumiai_runner.py (lines 269-273)
unified_analysis = self.timeline_builder.build_timeline(
    video_id, 
    video_metadata.to_dict(), 
    ml_results
)
```

### Step 3: Analysis Dictionary Creation
```python
# Structure passed to compute_temporal_windows
analysis_dict = {
    'video_id': str,
    'duration': float,
    'metadata': {
        'video_id': str,
        'likes': int,    # diggCount
        'views': int,    # playCount
        'saves': int,    # collectCount
        'shares': int,   # shareCount
        'comments': int  # commentCount
    },
    'ml_data': {
        'yolo': {...},           # Raw YOLO detections
        'whisper': {...},        # Speech segments
        'ocr': {...},            # Text/sticker detections
        'mediapipe': {...},      # Pose/face/hand data
        'scene_detection': {...}, # Scene boundaries
        'audio_energy': {...},    # RMS frames
        'emotion_detection': {...} # FEAT emotions
    },
    'timeline': {
        'entries': [...]  # Semantic timeline entries
    }
}
```

### Step 4: Temporal Window Computation
```python
# rumiai_runner.py (line 293)
temporal_windows = compute_temporal_windows(unified_analysis.to_dict())
```

---

## Data Extraction in compute_temporal_windows

### Mixed Strategy Implementation
The function uses **Decision 5** (temporal_compute.py lines 158-182):

1. **Timeline entries** → Semantic interpretations
   - Emotions, gestures, gaze events
   - Scene changes
   - Already interpreted/classified data

2. **ML data** → Raw detection results  
   - Bounding boxes, landmarks
   - RMS audio frames
   - Continuous measurements

### Extraction Functions

| Function | Source | Extracted Data |
|----------|--------|----------------|
| `extract_timelines_for_temporal()` | Mixed | All timeline data + scene durations |
| `extract_audio_energy_data()` | ml_data | RMS frames, frames_per_second |
| `extract_metadata()` | metadata | Engagement metrics |
| `extract_speech_segments()` | ml_data or timeline | Speech timing and text |

---

## Key Design Decisions

### Why Audio Energy Bypasses Timeline
- **Continuous signal** vs discrete events
- **Frame-level granularity** (30 fps = 30 values/second)
- **Direct statistical analysis** on raw RMS values
- **Performance optimization** - no need for event conversion

### Why Mixed Strategy
- **Semantic clarity**: Timeline entries for interpreted events
- **Raw precision**: ML data for measurements and coordinates
- **Flexibility**: Can compute new metrics from raw data
- **Backward compatibility**: Supports both old and new data formats

---

## Implications for Testing

### Test Script Requirements
1. **Load from correct locations**:
   - Timeline from `temporal_markers/*.json` if available
   - Audio from `audio_energy_outputs/`
   - Emotions from `emotion_detection_outputs/`
   - Speech from `speech_transcriptions/`

2. **Build correct structure**:
   - Must match `UnifiedAnalysis.to_dict()` output
   - Include both `ml_data` and `timeline` keys
   - Preserve metadata structure

3. **Handle format variations**:
   - Old emotion format (time-windowed dict)
   - New timeline format (entry list)
   - Missing services (set defaults)

---

## File References

| Component | File | Key Lines |
|-----------|------|-----------|
| Runner | `scripts/rumiai_runner.py` | 269-293 |
| Timeline Builder | `rumiai_v2/processors/timeline_builder.py` | 25-410 |
| Temporal Compute | `rumiai_v2/processors/temporal_compute.py` | 158-372, 774-1071 |
| Video Analyzer | `rumiai_v2/processors/video_analyzer.py` | 30-387 |
| Analysis Model | `rumiai_v2/models/analysis.py` | 92-143 |

---

## Summary

The post-refactor architecture achieves:
- **Single entry point**: `compute_temporal_windows()`
- **Unified processing**: All temporal features in one function
- **Mixed data strategy**: Optimal for both events and signals
- **Backward compatibility**: Supports multiple data formats
- **Performance optimization**: Direct access where appropriate