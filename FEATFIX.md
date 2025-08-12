# FEAT Integration Fix - Timeline Builder Integration

## Problem Statement

FEAT emotion detection is currently running on GPU and producing high-quality emotion analysis with Action Units (AUs), but the data is not being integrated into the unified timeline. This creates a gap where:
- FEAT generates detailed emotion data and saves it to `emotion_detection_outputs/`
- The timeline extraction doesn't include FEAT data in the `expressionTimeline`
- Precompute functions can't access FEAT results from the timeline
- Emotional journey analysis may use MediaPipe expressions instead of superior FEAT data

## Current State Analysis

### What's Working ✅
- FEAT emotion detection runs successfully on GPU
- Generates comprehensive emotion data including:
  - 7 basic emotions as output in JSON (anger, disgust, fear, joy, sadness, surprise, neutral)
    - Note: FEAT library uses 'happiness' but our service maps it to 'joy'
  - Action Units (AU) activations
  - Confidence scores for each detection
  - Face bounding boxes and landmarks
- Saves output to `emotion_detection_outputs/{video_id}/{video_id}_emotions.json`

### What's Missing ❌
- Timeline extraction in `_extract_timelines_from_analysis()` doesn't process FEAT data
- No validation for FEAT emotion data in MLDataValidator
- `expressionTimeline` is only populated from MediaPipe faces
- Precompute functions can't access FEAT's rich emotion data with Action Units

## Proposed Solution

### Proper Timeline Integration Approach (Following Established Pattern)

We'll integrate FEAT emotion detection following the same pattern used by all other ML services (YOLO, Whisper, OCR, MediaPipe, Scene Detection) by adding it to the timeline_builder.py. This ensures consistency and proper data flow through the timeline system.

### 1. Add FEAT Validation to MLDataValidator

First, add a new validation method to `/rumiai_v2/core/validators/ml_data_validator.py` following the established pattern:

```python
@staticmethod
def validate_emotion_data(data: Dict[str, Any], video_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate FEAT emotion detection data with fail-fast approach.
    Raises exception on any data quality issues rather than normalizing.
    
    Expected structure:
    {
        'emotions': [
            {
                'timestamp': float,
                'emotion': str,
                'confidence': float,
                'all_scores': {...},
                'action_units': [...],
                'au_intensities': {...}
            }
        ]
    }
    """
    # FAIL if no emotions key
    if 'emotions' not in data:
        raise ValueError(f"FATAL: FEAT data missing 'emotions' key for video {video_id}")
    
    if not isinstance(data['emotions'], list):
        raise ValueError(f"FATAL: FEAT 'emotions' is not a list for video {video_id}")
    
    valid_emotions = {'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'}
    
    for i, emotion_entry in enumerate(data['emotions']):
        if not isinstance(emotion_entry, dict):
            raise ValueError(f"FATAL: FEAT emotion entry {i} is not a dict for video {video_id}")
        
        # Required fields - FAIL if missing
        if 'timestamp' not in emotion_entry:
            raise ValueError(f"FATAL: FEAT emotion entry {i} missing timestamp for video {video_id}")
        
        if 'emotion' not in emotion_entry:
            raise ValueError(f"FATAL: FEAT emotion entry {i} missing emotion for video {video_id}")
        
        # Validate timestamp
        timestamp = emotion_entry['timestamp']
        if not isinstance(timestamp, (int, float)) or timestamp < 0:
            raise ValueError(f"FATAL: Invalid timestamp {timestamp} at entry {i} for video {video_id}")
        
        # Validate emotion - FAIL if unknown
        emotion = emotion_entry['emotion']
        if emotion not in valid_emotions:
            raise ValueError(f"FATAL: Unknown emotion '{emotion}' at {timestamp}s for video {video_id}. "
                           f"Valid emotions: {valid_emotions}")
        
        # Validate confidence - FAIL if out of range
        if 'confidence' in emotion_entry:
            confidence = emotion_entry['confidence']
            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"FATAL: Invalid confidence {confidence} at {timestamp}s for video {video_id}")
        
        # Validate action_units if present - FAIL if invalid
        if 'action_units' in emotion_entry:
            aus = emotion_entry['action_units']
            if not isinstance(aus, list):
                raise ValueError(f"FATAL: action_units is not a list at {timestamp}s for video {video_id}")
            for au in aus:
                if not isinstance(au, int) or not 1 <= au <= 45:
                    raise ValueError(f"FATAL: Invalid AU {au} at {timestamp}s for video {video_id}")
        
        # Validate au_intensities if present
        if 'au_intensities' in emotion_entry:
            intensities = emotion_entry['au_intensities']
            if not isinstance(intensities, dict):
                raise ValueError(f"FATAL: au_intensities is not a dict at {timestamp}s for video {video_id}")
            for au, intensity in intensities.items():
                if not isinstance(intensity, (int, float)) or not 0.0 <= intensity <= 1.0:
                    raise ValueError(f"FATAL: Invalid AU intensity {intensity} for AU {au} at {timestamp}s")
    
    # Return data AS-IS if all validation passes
    return data
```

### 2. Add Emotion Detection to Timeline Builder

First, add emotion_detection to the builders dictionary in `timeline_builder.py` and implement the `_add_emotion_entries()` method:

```python
# In timeline_builder.py, update the builders dictionary (around line 45):
builders = {
    'yolo': self._add_yolo_entries,
    'whisper': self._add_whisper_entries,
    'ocr': self._add_ocr_entries,
    'mediapipe': self._add_mediapipe_entries,
    'scene_detection': self._add_scene_entries,
    'emotion_detection': self._add_emotion_entries  # ADD THIS LINE
}

# Then implement the _add_emotion_entries method:
def _add_emotion_entries(self, timeline: Timeline, emotion_data: Dict[str, Any]) -> None:
    """
    Add FEAT emotion detection entries to timeline.
    
    Transforms FEAT emotion data into timeline entries with Action Units.
    """
    # Validate emotion data
    emotion_data = self.ml_validator.validate_emotion_data(emotion_data, timeline.video_id)
    
    # Process each emotion detection
    for emotion_entry in emotion_data.get('emotions', []):
        timestamp = emotion_entry['timestamp']  # Trust validator - no default needed
        
        # Create timeline entry with full FEAT data
        entry = TimelineEntry(
            entry_type='emotion',
            start=Timestamp(timestamp),
            end=Timestamp(timestamp + 1),  # 1-second window
            data={
                'emotion': emotion_entry['emotion'],  # Trust validator - required field
                'confidence': emotion_entry.get('confidence', 0.0),  # Optional field, keep default
                'all_scores': emotion_entry.get('all_scores', {}),  # Optional field
                'action_units': emotion_entry.get('action_units', []),  # Optional field
                'au_intensities': emotion_entry.get('au_intensities', {}),  # Optional field
                'source': 'feat'  # Mark source for downstream validation
            },
            confidence=emotion_entry.get('confidence', 0.0),  # Optional, keep default
            metadata={'model': 'feat', 'has_action_units': bool(emotion_entry.get('action_units'))}
        )
        
        timeline.add_entry(entry)
    
    logger.info(f"Added {len(emotion_data.get('emotions', []))} FEAT emotion entries to timeline")
```

### 3. Update Timeline Extraction in Precompute Functions

Modify `_extract_timelines_from_analysis()` in `precompute_functions.py` to extract FEAT emotion entries from the timeline:

```python
def _extract_timelines_from_analysis(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract timeline data from unified analysis."""
    timeline_data = analysis_dict.get('timeline', {})
    
    # Build timelines dictionary expected by compute functions
    timelines = {
        'textOverlayTimeline': {},
        'stickerTimeline': {},
        'speechTimeline': {},
        'objectTimeline': {},
        'gestureTimeline': {},
        'expressionTimeline': {},  # Will contain FEAT data from timeline
        'sceneTimeline': {},
        'sceneChangeTimeline': {},
        'personTimeline': {},
        'cameraDistanceTimeline': {}
    }
    
    # ... existing extraction code for other timelines ...
    
    # Extract FEAT emotion entries from timeline
    timeline_entries = timeline_data.get('entries', [])
    for entry in timeline_entries:
        if entry.get('entry_type') == 'emotion':
            # Extract timestamp range
            start = entry.get('start', 0)
            end = entry.get('end', start + 1)
            timestamp_key = f"{int(start)}-{int(end)}s"
            
            # Add FEAT data to expressionTimeline
            timelines['expressionTimeline'][timestamp_key] = entry.get('data', {})
    
    logger.info(f"ExpressionTimeline populated with {len(timelines['expressionTimeline'])} FEAT entries")
    
    return timelines
```

### 4. Update Precompute Functions to Use FEAT Data

Modify emotion-related precompute functions to leverage FEAT's richer data:

```python
def compute_emotional_journey_analysis_professional(timelines, duration):
    """
    Professional emotional journey analysis using FEAT data ONLY.
    Fails fast if FEAT data is not available - no fallback to estimates.
    """
    expression_timeline = timelines.get('expressionTimeline', {})
    
    if not expression_timeline:
        # No FEAT data available - return minimal/empty analysis
        logger.warning("No FEAT emotion data available - returning minimal emotional analysis")
        return generate_minimal_emotional_analysis()
    
    # Verify this is FEAT data (fail-fast if not)
    first_entry = next(iter(expression_timeline.values()), {})
    data_source = first_entry.get('source', 'unknown')
    
    if data_source != 'feat':
        # This should never happen with the new extraction logic
        raise ValueError(f"Expected FEAT emotion data but got {data_source}. Emotional analysis requires FEAT.")
    
    # Confirmed FEAT data with Action Units
    logger.info(f"Computing emotional journey with FEAT data ({len(expression_timeline)} time points)")
    
    # Use Action Units for professional emotion analysis
    au_patterns = analyze_action_unit_patterns(expression_timeline)
    emotion_authenticity = calculate_emotion_authenticity_from_aus(expression_timeline)
    micro_expressions = detect_micro_expressions(expression_timeline)
    
    # Professional analysis using FEAT's rich data
    enhanced_metrics = {
        'action_unit_patterns': au_patterns,
        'emotion_authenticity': emotion_authenticity,
        'micro_expressions': micro_expressions,
        'data_quality': 'professional',  # Always professional with FEAT
        'source': 'feat'
    }
    
    # Rest of the emotional journey computation...
    # Uses FEAT's detailed emotion scores and AU data
```

## Benefits of This Fail-Fast Approach

**Data Quality Guarantees:**
- **No emotional estimates** - Only professional-grade FEAT data is used
- **Fail-fast architecture** - System explicitly signals when emotion data is unavailable
- **High confidence analysis** - Action Units ensure emotion authenticity
- **No confusion** - Single source (FEAT) for all emotional data

**Architectural Advantages:**
- **Follows established pattern** - Consistent with all other ML services
- **Proper data flow** - ML services → timeline builder → timeline entries → precompute
- **Timeline integration** - FEAT data available in timeline for all downstream processing
- **Clean separation** - Timeline builder handles data transformation, precompute handles analysis

**Why No MediaPipe Fallback:**
- MediaPipe provides rough expression estimates, not true emotion detection
- FEAT uses validated emotion models with Action Unit analysis
- Mixing data sources would compromise analysis quality
- Better to have no emotional analysis than inaccurate estimates

### 5. Implementation Steps

1. **Add `validate_emotion_data` method** to `/rumiai_v2/core/validators/ml_data_validator.py`
2. **Update timeline_builder.py**:
   - Add `'emotion_detection': self._add_emotion_entries` to builders dictionary
   - Implement `_add_emotion_entries()` method to process FEAT data
3. **Update `_extract_timelines_from_analysis()`** in `precompute_functions.py`:
   - Extract emotion entries from timeline (not ml_data)
   - Build expressionTimeline from FEAT timeline entries
4. **Existing precompute functions** automatically get validated FEAT data from timeline
5. **Remove workarounds** in professional functions that directly call FEAT



## Summary of Implementation

This solution integrates FEAT emotion detection into the RumiAI pipeline through:

1. **Data Validation** - Adding proper validation to MLDataValidator class
2. **Timeline Integration** - Adding FEAT to timeline_builder.py following established pattern
3. **Fail-Fast Policy** - No fallback to MediaPipe estimates
4. **Proper Data Flow** - ML service → timeline → precompute (like all other services)
5. **Consistent Architecture** - Following the same pattern as YOLO, Whisper, OCR, etc.

The approach ensures professional-grade emotion analysis with Action Units while maintaining clean architecture.

## Testing Strategy

### Timeline Extraction Test
```python
def test_feat_extraction_from_timeline():
    """Test that FEAT data properly populates expressionTimeline."""
    # Create test data with both FEAT and MediaPipe
    analysis_dict = {
        'ml_data': {
            'emotion_detection': {
                'emotions': [
                    {
                        'timestamp': 0.0,
                        'emotion': 'joy',
                        'confidence': 0.9,
                        'all_scores': {'joy': 0.9, 'neutral': 0.1},
                        'action_units': [6, 12],
                        'au_intensities': {'6': 0.8, '12': 0.9}
                    }
                ]
            },
            'mediapipe': {
                'faces': [
                    {'timestamp': 0, 'expression': 'neutral', 'confidence': 0.7}
                ]
            }
        },
        'timeline': {'duration': 60},
        'video_id': 'test_video'
    }
    
    timelines = _extract_timelines_from_analysis(analysis_dict)
    
    # Verify FEAT data takes priority over MediaPipe
    assert timelines['expressionTimeline']['0-1s']['source'] == 'feat'
    assert timelines['expressionTimeline']['0-1s']['emotion'] == 'joy'
    assert 'action_units' in timelines['expressionTimeline']['0-1s']
    # MediaPipe expression should be ignored when FEAT exists
```

### Timeline Builder Test
```python
def test_feat_timeline_integration():
    """Test FEAT data is properly added to timeline."""
    # Load sample FEAT output
    feat_data = {
        'emotions': [
            {
                'timestamp': 0.0,
                'emotion': 'joy',
                'confidence': 0.9,
                'all_scores': {'joy': 0.9, 'neutral': 0.1},
                'action_units': [6, 12],
                'au_intensities': {'6': 0.8, '12': 0.9}
            }
        ]
    }
    
    # Build timeline using timeline builder
    timeline = Timeline(video_id='test', duration=60)
    builder = TimelineBuilder()
    builder._add_emotion_entries(timeline, feat_data)
    
    # Verify emotion entries exist in timeline
    emotion_entries = [e for e in timeline.entries if e.entry_type == 'emotion']
    assert len(emotion_entries) > 0
    assert emotion_entries[0].data['emotion'] == 'joy'
    assert emotion_entries[0].data['source'] == 'feat'
    assert 'action_units' in emotion_entries[0].data
```

### Integration Test
```bash
# Run full pipeline on test video
python scripts/rumiai_runner.py "test_video.mp4"

# Check that emotion timeline is populated
grep "emotionTimeline" insights/*/emotional_journey/emotional_journey_complete_*.json
```

## Expected Outcomes

After implementing this fix:

1. ✅ FEAT emotion data will be available in the unified timeline
2. ✅ Precompute functions will use high-quality FEAT emotions instead of MediaPipe expressions
3. ✅ Emotional journey analysis will include Action Units and detailed emotion scores
4. ✅ Expression timeline will contain professional-grade emotion data with Action Units
5. ✅ Emotion features from ML_FEATURES_DOCUMENTATION_V2.md will be properly populated

## Performance Impact

- **No performance degradation**: FEAT already runs during ML analysis phase
- **No additional API calls**: Using existing FEAT output files
- **Improved analysis quality**: Better emotion data for all downstream analysis
- **Memory usage**: Minimal increase (~100KB per video for emotion timeline)


## Additional Considerations

### Future Enhancements
- Add emotion smoothing to reduce frame-to-frame noise
- Implement emotion transition detection
- Add cross-modal emotion validation (speech sentiment vs facial expression)
- Cache emotion timelines for faster reprocessing

### Data Quality Monitoring
- Log when FEAT data is missing or incomplete
- Track confidence scores to identify low-quality detections
- Monitor AU activation patterns for anomalies

This fix will fully integrate FEAT's superior emotion detection capabilities into the RumiAI pipeline, enhancing the quality of emotional journey analysis while maintaining the zero-cost Python-only processing architecture.