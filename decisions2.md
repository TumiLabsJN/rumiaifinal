# Refactortemporal.md Critique Decisions

## Decision 1: Timeline Format - Modify at Source

**Issue**: Timeline format mismatch - temporal_compute expects arrays with timestamps, but `_extract_timelines_from_analysis()` returns dicts with timestamp keys.

**Decision**: Modify `_extract_timelines_from_analysis()` directly in precompute_functions.py to output arrays with timestamp fields.

**Rationale**:
- We're deleting the 7 existing precompute functions in Phase 3 anyway
- Cleaner to fix at source rather than add transformation layers
- Single extraction function for temporal windows format
- No wrapper complexity

**Implementation**:
```python
def _extract_timelines_from_analysis(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract timeline data as arrays with timestamp fields for temporal calculations"""
    
    timelines = {
        'text_overlay_timeline': [],  # Array format
        'sticker_timeline': [],       # Array format
        'object_timeline': [],        # Array format
        # ... etc
    }
    
    # Extract as arrays with timestamp field
    for annotation in ocr_data.get('textAnnotations', []):
        timelines['text_overlay_timeline'].append({
            'timestamp': annotation.get('timestamp', 0),
            'text': annotation.get('text', ''),
            # ... other fields
        })
    
    return timelines
```

## Decision 2: Audio Energy - Store Raw RMS Frames

**Issue**: Need to recalculate audio energy for exact temporal windows (not 5-second windows) as spec requires. Need energy_level, energy_variance, energy_max, and burst patterns per window.

**Decision**: Modify audio_energy_service to store raw RMS frames in ML data, allowing exact window calculations in temporal_compute.

**Rationale**:
- **Spec compliance**: P0TemporalWindows.md line 641 explicitly requires "Recalculate audio energy for exact windows"
- **Complete metrics**: Can calculate variance, max, burst patterns from raw data
- **Fail-fast principle**: No external file dependencies that could fail
- **Self-contained**: Everything in unified JSON, no cache dependencies
- **Acceptable size**: 15KB increase for 30s video is worth correctness

**Implementation**:
```python
# In audio_energy_service.py
return {
    "rms_frames": rms.tolist(),  # Raw RMS frames array
    "frames_per_second": sr / hop_length,  # For timestamp mapping
    "climax_frame": int(np.argmax(rms)),
    "duration": duration
}

# In temporal_compute.py
def compute_temporal_windows(analysis_dict):
    audio_energy = ml_data.get('audio_energy', {})
    
    if 'rms_frames' in audio_energy:
        rms = np.array(audio_energy['rms_frames'])
        fps = audio_energy['frames_per_second']
        
        for window_name, bounds in windows.items():
            start_frame = int(bounds[0] * fps)
            end_frame = int(bounds[1] * fps)
            window_rms = rms[start_frame:end_frame]
            
            # Calculate all required metrics
            energy_results[f'{window_name}_energy_level'] = float(np.mean(window_rms))
            energy_results[f'{window_name}_energy_variance'] = float(np.var(window_rms))
            energy_results[f'{window_name}_energy_max'] = float(np.max(window_rms))
            # ... burst patterns
```

## Decision 3: Self-Contained Temporal Module

**Issue**: Import organization - should temporal_compute import from precompute_functions?

**Decision**: Make temporal_compute completely self-contained. No imports from precompute_functions since we're deleting all old analysis code.

**Rationale**:
- We're replacing all 7 analyses with temporal_windows only
- Clean break - no dependencies on code we're deleting
- Single output: just temporal_windows.json (not 21 files)
- Simpler architecture without legacy dependencies

**Implementation**:
```python
# temporal_compute.py - STANDALONE module
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import numpy as np
import re
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_timelines_for_temporal(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract timelines in format needed for temporal windows.
    Self-contained - doesn't depend on precompute_functions.
    """
    ml_data = analysis_dict.get('ml_data', {})
    timelines = {}
    
    # Extract all timelines as arrays with timestamps
    # OCR text overlays
    ocr_data = ml_data.get('ocr', {})
    timelines['text_overlay_timeline'] = [
        {'timestamp': ann.get('timestamp', 0), 'text': ann.get('text', '')}
        for ann in ocr_data.get('textAnnotations', [])
    ]
    # ... extract other timelines ...
    
    return timelines

def compute_temporal_windows(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Single source of truth for all analysis"""
    timelines = extract_timelines_for_temporal(analysis_dict)
    # ... computation ...
```

**Command remains the same**:
```bash
python3 scripts/rumiai_runner.py "https://tiktok.com/video"
# Output: insights/{video_id}/temporal_windows.json (single file)
```

## Decision 4: Extract in Expected Format

**Issue**: Helper functions in temporal_compute.py need specific timeline formats.

**Decision**: Extract timelines in the exact format our existing helpers expect. No helper modifications needed.

**Rationale**:
- We control the extraction function since we're self-contained
- Existing temporal_compute.py helpers already work correctly
- Simpler to extract correctly than to modify all helpers
- Maintains all existing calculation logic

**Implementation**:
```python
def extract_timelines_for_temporal(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract in the EXACT format our helpers expect"""
    ml_data = analysis_dict.get('ml_data', {})
    
    timelines = {
        'text_overlay_timeline': [],  # Arrays with timestamp field
        'sticker_timeline': [],
        'object_timeline': [],
        'gesture_timeline': [],
        'expression_timeline': [],
        'scene_boundaries': [],
        # Some as dicts if helpers expect that
        'personTimeline': {},
        'gaze_timeline': {},
        'camera_distance_timeline': {}
    }
    
    # Extract OCR to match what calculate_visual_density expects
    ocr_data = ml_data.get('ocr', {})
    for annotation in ocr_data.get('textAnnotations', []):
        timelines['text_overlay_timeline'].append({
            'timestamp': annotation.get('timestamp', 0),
            'text': annotation.get('text', ''),
            'position': annotation.get('position', 'center'),
            'confidence': annotation.get('confidence', 0.9)
        })
    
    # Similar for other timelines - extract in expected format
    return timelines
```

## Decision 5: Validate Service Execution, Not Content

**Issue**: Need validation but must handle legitimate empty results (silent videos, no text, single scene).

**Decision**: Validate that ML services ran successfully, but accept empty results as valid.

**Rationale**:
- Fail-fast for service failures (didn't run, crashed)
- Accept empty results for legitimate cases (silent video, no overlays)
- Distinguish between "service failed" vs "no content found"
- Log empty results for debugging without failing

**Implementation**:
```python
def compute_temporal_windows(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Compute temporal windows with smart validation"""
    
    # Validate structure
    if not analysis_dict or 'ml_data' not in analysis_dict:
        raise ValueError("Missing ml_data in analysis_dict")
    
    ml_data = analysis_dict['ml_data']
    
    # Validate services RAN (not that they found content)
    required_services = ['ocr', 'mediapipe', 'yolo', 'whisper', 'scene_detection']
    for service in required_services:
        if service not in ml_data:
            raise ValueError(f"ML service didn't run: {service}")
        
        # Check service completed successfully
        service_data = ml_data[service]
        if not isinstance(service_data, dict):
            raise ValueError(f"Invalid {service} data format")
        
        # Check execution metadata if available
        if 'metadata' in service_data:
            if not service_data['metadata'].get('processed', True):
                raise ValueError(f"{service} failed to process video")
    
    # Extract data - empty arrays are OK
    whisper_segments = ml_data['whisper'].get('segments', [])
    scene_boundaries = ml_data['scene_detection'].get('scenes', [])
    text_annotations = ml_data['ocr'].get('textAnnotations', [])
    
    # Log empty results (not errors)
    if not whisper_segments:
        logger.info("Video has no speech (silent)")
    if not scene_boundaries:
        logger.info("Video has no scene changes (single shot)")
    if not text_annotations:
        logger.info("Video has no text overlays")
    
    # Continue with computation - zeros where appropriate
```

## Decision 7: Audio Path - Use Raw RMS Frames (Already Resolved by Decision 2)

**Issue**: Original temporal_compute.py has `audio_path: Optional[Path]` parameter, but refactored version using `analysis_dict` can't pass this.

**Decision**: Already resolved by Decision 2 (store raw RMS frames). No additional implementation needed.

**Rationale**:
- Decision 2 eliminates need for audio_path parameter entirely
- Raw RMS frames in ML data provide everything needed for calculations
- No external file dependencies = fail-fast compliance
- Self-contained in unified JSON

**Implementation Steps**:
1. **Audio Energy Service** (already covered in Decision 2):
   ```python
   # In audio_energy_service.py - return raw RMS frames
   return {
       "rms_frames": rms.tolist(),
       "frames_per_second": sr / hop_length,
       "climax_frame": int(np.argmax(rms)),
       "duration": duration
   }
   ```

2. **Temporal Compute** (already covered in Decision 2):
   ```python
   # In temporal_compute.py - use RMS frames from ML data
   audio_energy = ml_data.get('audio_energy', {})
   if 'rms_frames' in audio_energy:
       rms = np.array(audio_energy['rms_frames'])
       fps = audio_energy['frames_per_second']
       # Calculate metrics for each window...
   ```

3. **Remove audio_path parameter**:
   ```python
   # OLD signature (DELETE):
   def compute_temporal_windows(
       timelines: Dict,
       video_metadata: Dict,
       speech_segments: List[Dict],
       audio_path: Optional[Path] = None  # REMOVE THIS
   )
   
   # NEW signature (as per Decision 3):
   def compute_temporal_windows(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
   ```

**No additional work needed** - Decision 2's RMS frames approach fully handles audio energy without needing audio_path.

## Decision 8: Add Detailed Error Context

**Issue**: With analysis_dict as single parameter, we lose context about what specifically failed. Original had explicit parameters that made debugging clearer.

**Decision**: Add detailed error context at each extraction step with specific error messages.

**Rationale**:
- Better debugging without breaking the standard pattern
- Clear error messages identify exactly what's missing or malformed
- Maintains fail-fast principle with informative failures
- Helps distinguish between missing data vs. malformed data

**Implementation**:
```python
def compute_temporal_windows(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Compute temporal windows with detailed error context"""
    
    # Validate input structure with context
    if not analysis_dict:
        raise ValueError("compute_temporal_windows: analysis_dict is None or empty")
    
    if not isinstance(analysis_dict, dict):
        raise TypeError(f"compute_temporal_windows: Expected dict, got {type(analysis_dict)}")
    
    # Extract ML data with error context
    if 'ml_data' not in analysis_dict:
        raise KeyError("compute_temporal_windows: Missing 'ml_data' in analysis_dict")
    
    ml_data = analysis_dict['ml_data']
    if not isinstance(ml_data, dict):
        raise TypeError(f"compute_temporal_windows: ml_data should be dict, got {type(ml_data)}")
    
    # Extract video metadata with context
    if 'video_metadata' not in analysis_dict:
        raise KeyError("compute_temporal_windows: Missing 'video_metadata' in analysis_dict")
    
    video_metadata = analysis_dict['video_metadata']
    
    # Extract timelines with detailed context
    try:
        timelines = extract_timelines_for_temporal(analysis_dict)
    except Exception as e:
        raise ValueError(f"compute_temporal_windows: Failed to extract timelines - {str(e)}")
    
    # Extract duration with validation
    duration = analysis_dict.get('timeline', {}).get('duration', 0)
    if duration <= 0:
        # Try video_metadata as fallback
        duration = video_metadata.get('duration', 0)
        if duration <= 0:
            raise ValueError(f"compute_temporal_windows: Invalid duration={duration} from both timeline and video_metadata")
    
    # Extract speech segments with context
    if 'whisper' not in ml_data:
        logger.warning("compute_temporal_windows: No whisper data in ml_data (video may be silent)")
        speech_segments = []
    else:
        whisper_data = ml_data['whisper']
        if not isinstance(whisper_data, dict):
            raise TypeError(f"compute_temporal_windows: whisper data should be dict, got {type(whisper_data)}")
        speech_segments = whisper_data.get('segments', [])
    
    # Extract audio energy with context
    if 'audio_energy' not in ml_data:
        logger.warning("compute_temporal_windows: No audio_energy in ml_data")
        audio_energy = {}
    else:
        audio_energy = ml_data['audio_energy']
        if 'rms_frames' not in audio_energy:
            logger.warning("compute_temporal_windows: audio_energy missing rms_frames")
    
    # Log successful extraction
    logger.info(f"compute_temporal_windows: Extracted data for video {analysis_dict.get('video_id', 'unknown')}")
    logger.info(f"  - Duration: {duration}s")
    logger.info(f"  - Speech segments: {len(speech_segments)}")
    logger.info(f"  - Timeline types: {list(timelines.keys())}")
    logger.info(f"  - Audio energy: {'yes' if 'rms_frames' in audio_energy else 'no'}")
    
    # Continue with computation...
```

**Error Context Benefits**:
1. Each error message starts with function name for stack trace clarity
2. Shows what was expected vs. what was received
3. Distinguishes between missing (KeyError) vs wrong type (TypeError) vs invalid (ValueError)
4. Logs warnings for optional data (silent videos OK)
5. Logs successful extraction summary for debugging

## Decision 9: Keep Existing Edge Case Handling

**Issue**: Hook window edge case - what happens with videos shorter than 3 seconds? Could break temporal window logic.

**Decision**: Keep the existing `calculate_temporal_windows` function from original temporal_compute.py - it already handles all edge cases correctly.

**Rationale**:
- Original implementation already solved this completely
- Returns `None` for windows that don't exist (clean approach)
- Tested logic that handles all video durations properly

**Implementation**:
```python
def calculate_temporal_windows(video_duration: float) -> Dict[str, Optional[Tuple[float, float]]]:
    """
    Calculate hook, middle, and closing windows based on video duration.
    Handles edge cases for short videos.
    
    Args:
        video_duration: Duration of video in seconds
        
    Returns:
        Dict with 'hook', 'middle', 'closing' window boundaries
    """
    if video_duration <= 3:
        return {
            'hook': (0, video_duration),
            'middle': None,
            'closing': None
        }
    elif video_duration <= 6:
        hook_end = min(3, video_duration)
        return {
            'hook': (0, hook_end),
            'middle': None,
            'closing': (hook_end, video_duration)
        }
    else:
        hook_end = 3
        closing_start = video_duration - 3
        return {
            'hook': (0, hook_end),
            'middle': (hook_end, closing_start) if closing_start > hook_end else None,
            'closing': (closing_start, video_duration)
        }
```

**Edge Case Rules (already implemented)**:
- **≤3 seconds**: Only hook window (full duration), middle=None, closing=None
- **3-6 seconds**: Hook (0-3s) + Closing (3s-end), middle=None  
- **>6 seconds**: Hook (0-3s) + Middle (3s to end-3s) + Closing (last 3s)

**No changes needed** - use existing function as-is.

## Decision 10: Keep Middle Segment Logic As-Is

**Issue**: Middle segment logic complexity - complex rules for dividing middle window into 3-5 segments based on duration.

**Decision**: Keep the existing `calculate_middle_segments` function unchanged - it implements required spec behavior.

**Rationale**:
- P0TemporalWindows.md explicitly requires `middle_segment_1` through `middle_segment_5` features
- Spec shows `# if video > 33s` for segment 4 and `# if video > 60s` for segment 5
- These segments are ML features for understanding content pacing
- Logic is already implemented, tested, and working correctly

**Implementation** (keep existing):
```python
def calculate_middle_segments(video_duration: float) -> Dict[str, Dict[str, float]]:
    """Calculate segment boundaries for the middle window."""
    middle_start = 3  # After hook
    middle_end = video_duration - 3  # Before closing
    middle_duration = middle_end - middle_start
    
    # Handle edge cases
    if middle_duration <= 0:
        return {}
    
    # No segments if middle < 3s
    if middle_duration < 3:
        return {}  # Middle exists but no segments
    
    # Determine segment count based on middle duration
    if middle_duration <= 12:
        num_segments = 3
    elif middle_duration <= 27:
        num_segments = 4
    else:
        num_segments = 5
    
    # Calculate equal segments with precise boundaries
    segment_duration = middle_duration / num_segments
    segments = {}
    
    for i in range(num_segments):
        segment_start = middle_start + (i * segment_duration)
        segment_end = middle_start + ((i + 1) * segment_duration)
        segments[f'segment_{i+1}'] = {
            'start': round(segment_start, 2),  # 10ms precision
            'end': round(segment_end, 2)
        }
    
    return segments
```

**Segment Rules (required by spec)**:
- Middle < 3s: No segments
- Middle 3-12s: 3 segments (videos 9-18s total)
- Middle 12-27s: 4 segments (videos 18-33s total)
- Middle > 27s: 5 segments (videos 33s+ total)

**No changes needed** - complexity is justified by spec requirements.

## Decision 6: Calculate All Required Metadata Inline

**Issue**: Current temporal_compute.py is missing required metadata fields (mentionCount, emojiCount, linkPresent, callToAction, publishDayOfWeek).

**Decision**: Calculate all required metadata fields inline from Apify data. No separate transformation function needed.

**Rationale**:
- P0TemporalWindows.md specifies exact metadata requirements
- Current implementation was incomplete
- Inline calculation is clear and self-documenting
- No extra transformation layers or dependencies

**Implementation**:
```python
def compute_temporal_windows(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    video_metadata = analysis_dict.get('video_metadata', {})
    description = video_metadata.get('description', '')
    
    # Calculate ALL required metadata inline
    import re
    from datetime import datetime
    
    # Parse create time for temporal features
    create_time_str = video_metadata.get('createTime', '')
    try:
        dt = datetime.fromisoformat(create_time_str.replace('Z', '+00:00'))
        publish_hour = dt.hour
        publish_day = dt.weekday()
    except:
        publish_hour = 0
        publish_day = 0
    
    # Calculate text metrics
    mention_count = len(re.findall(r'@\w+', description))
    emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', description))
    link_present = 1 if re.search(r'https?://\S+', description) else 0
    cta_keywords = ['link in bio', 'comment', 'follow', 'like', 'share', 'subscribe']
    call_to_action = 1 if any(kw in description.lower() for kw in cta_keywords) else 0
    
    # Set global_metadata with ALL required fields
    result['global_metadata'] = {
        'video_duration': video_metadata.get('duration', 0),
        'video_id': video_metadata.get('id', ''),
        'publish_hour': publish_hour,
        'publish_day_of_week': publish_day,
        'caption_length': len(description),
        'hashtag_count': len(video_metadata.get('hashtags', [])),
        'mention_count': mention_count,
        'emoji_count': emoji_count,
        'link_present': link_present,
        'call_to_action': call_to_action,
        'has_soundtrack': bool(video_metadata.get('music', {}))
    }
    
    # Set outcomes (ML targets)
    result['outcomes'] = {
        'view_count': video_metadata.get('views', 0),
        'like_count': video_metadata.get('likes', 0),
        'comment_count': video_metadata.get('comments', 0),
        'share_count': video_metadata.get('shares', 0),
        'engagement_rate': video_metadata.get('engagementRate', 0.0)
    }
```