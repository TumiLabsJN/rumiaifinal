# Person Count Fix - YOLO Data Format Issue

## Problem Statement

### Symptom
- `person_count` feature in temporal windows shows incorrect values
- Example: Video10TwoPeople.mp4 shows `person_count=3` in segment_3 despite only 2 people being in video
- YOLO detects persons correctly (85-91% confidence) but detections don't appear in timeline

### Root Cause Analysis (Verified Through Tracing)

#### 1. Data Format Mismatch - CONFIRMED
YOLO service outputs flat list format:
```json
{
  "objectAnnotations": [
    {
      "trackId": "obj_0_1",
      "className": "person",    // Note: "className" not "class"
      "confidence": 0.858,
      "timestamp": 0.0,
      "bbox": [231.7, 419.8, 573.7, 1020.3],
      "frame_number": 0,
      "tracked": true
    },
    {
      "trackId": "obj_0_2",
      "className": "person",
      "confidence": 0.847,
      "timestamp": 0.0,
      "bbox": [0.0, 289.9, 390.2, 1024.0],
      "frame_number": 0,
      "tracked": true
    }
  ]
}
```

Timeline Builder expects grouped format:
```json
{
  "objectAnnotations": [
    {
      "class": "person",        // Note: "class" not "className"
      "frames": [               // Grouped by class with frames array
        {
          "timestamp": 0.0,
          "confidence": 0.858,
          "bbox": [231.7, 419.8, 573.7, 1020.3],
          "trackId": "obj_0_1"
        },
        {
          "timestamp": 0.0,
          "confidence": 0.847,
          "bbox": [0.0, 289.9, 390.2, 1024.0],
          "trackId": "obj_0_2"
        }
      ]
    }
  ]
}
```

#### 2. Validator Processing - TRACED
Actual behavior discovered through logging:
1. `ml_data_validator.py` receives 104 separate YOLO detections (flat list)
2. For each detection:
   - Finds `className` field → converts to `class: "person"` ✅
   - Detects flat format (has timestamp & bbox) → wraps as single-frame array ✅
3. Result: 104 separate annotations, each with `class: "person"` and `frames: [single_detection]`
4. **Problem**: Should be 1 annotation with 104 frames, not 104 annotations with 1 frame each

#### 3. Cascading Effects
- Timeline builder expects grouped format (1 annotation per class with multiple frames)
- Processing 104 separate single-frame annotations causes undefined behavior
- Person counting logic may be double-counting or miscounting track IDs
- Temporal windows show incorrect person counts (e.g., 3 instead of 2)

## Solution Design

### Complete Fix Required (3 Issues Found)

After implementation and testing, we discovered THREE interconnected issues:

1. **Validator Issue**: Creates 104 separate annotations instead of 1 grouped annotation
2. **Timeline Builder Issue**: Looks for trackId in wrong location after grouping
3. **Temporal Compute Issue**: Uses raw YOLO data instead of timeline entries

### Implementation Approach
Fix all three issues to restore proper person counting:
1. Group detections by class in validator
2. Update timeline builder to get trackId from frame data
3. Fix temporal_compute to use timeline entries for objects

**Why this comprehensive fix:**
- Each component was making incorrect assumptions
- Partial fixes resulted in person_count = 0
- All three must work together for correct counting

## Implementation Plan

### 1. Update Validator Grouping Logic
File: `/home/jorge/rumiaifinal/rumiai_v2/core/validators/ml_data_validator.py`

After processing individual annotations (around line 99), add grouping logic:

```python
# After the for loop that processes annotations (line 99)
# Group annotations by class to fix flat YOLO format issue
grouped_by_class = {}
for annotation in data['objectAnnotations']:
    class_name = annotation.get('class', 'unknown')
    if class_name not in grouped_by_class:
        grouped_by_class[class_name] = {
            'class': class_name,
            'frames': []
        }

    # If this annotation has frames, add them to the grouped annotation
    if 'frames' in annotation and annotation['frames']:
        # For flat format, frames will be [annotation.copy()]
        # For already grouped format, frames will be the actual frames array
        for frame in annotation['frames']:
            # Remove redundant class field from frame if present
            if 'class' in frame:
                del frame['class']
            if 'className' in frame:
                del frame['className']
            grouped_by_class[class_name]['frames'].append(frame)

# Replace the annotations with grouped version
data['objectAnnotations'] = list(grouped_by_class.values())
logger.info(f"[TRACE] Grouped {len(original_annotations)} annotations into {len(data['objectAnnotations'])} classes")
```

### 2. Remove Debug Logging
After fixing the issue, remove the TRACE logging added during investigation to avoid log spam.

### 3. Verification Steps

#### A. Check YOLO Output Format
```bash
python3 test_manual_videos.py Video10TwoPeople.mp4
cat object_detection_outputs/*/812144131642935_yolo_detections.json | head -50
# Should show grouped format with 'class' and 'frames' fields
```

#### B. Verify Timeline Integration
```bash
grep -A5 '"entry_type": "object"' unified_analysis/812144131642935.json
# Should now show object entries with person detections
```

#### C. Validate Person Count
```bash
jq '.temporal_windows.middle_segments[2].person_count' \
  insights/812144131642935_temporal_windows_updated.json
# Should show 2 (not 3) for two-person video
```

### 4. Edge Cases to Consider

1. **Empty Detections**: Handle case where YOLO detects nothing
```python
if not results:
    return {'objectAnnotations': []}
```

2. **Single Frame Videos**: Ensure single detection still creates frames array
```python
# Even single detection should be in frames array:
{'class': 'person', 'frames': [single_detection]}
```

3. **Multiple Classes**: Ensure each class gets its own group
```python
# person, bottle, etc. each get separate entries in objectAnnotations
```

## Testing Plan

### Unit Test Coverage
1. Test flat → grouped transformation with multiple classes
2. Test empty detection handling
3. Test single detection creates array
4. Test trackId preservation

### Integration Tests
1. Run on Video10TwoPeople.mp4 → verify person_count=2
2. Run on Video08GenderMale.mp4 → verify person_count=1
3. Run on video with objects → verify object_count > 0

### Regression Tests
1. Ensure other services still work (MediaPipe, OCR, etc.)
2. Verify temporal window computation unchanged
3. Check that existing features remain stable

## Rollback Plan
If issues arise:
1. Revert changes to `ml_services_unified.py`
2. Clear cached YOLO outputs: `rm -rf object_detection_outputs/*`
3. Re-run affected videos

## Future Improvements
1. Add face-based person counting as fallback when YOLO fails
2. Implement confidence-based filtering for person detections
3. Add validation tests to catch format mismatches earlier
4. Consider updating validator to handle both formats flexibly

## References
- Issue discovered: 2025-09-30
- Test videos: Video10TwoPeople.mp4, Video08GenderMale.mp4
- Related files:
  - `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py`
  - `/home/jorge/rumiaifinal/rumiai_v2/processors/timeline_builder.py`
  - `/home/jorge/rumiaifinal/rumiai_v2/core/validators/ml_data_validator.py`
  - `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`