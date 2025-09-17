# Fix Person Count: Unique Person Detection Issue
**Created**: 2025-01-16
**Status**: Ready for Implementation
**Issue**: person_count inflated by counting frame detections instead of unique individuals
**Solution**: Count maximum unique persons visible at any single moment

---

## Executive Summary

The `person_count` metric currently reports 9 people for a single-person video because it counts total YOLO detections across all frames. This fix changes it to count the maximum number of unique individuals visible at any moment, using trackId deduplication with confidence filtering.

---

## Problem Analysis

### Observed Behavior
Video 7515687288257465630 (single person throughout):
- **Hook (0-3s)**: Reports `person_count: 9`
- **Segment 1 (3-10.6s)**: Reports `person_count: 24`
- **Ground Truth**: Only 1 person visible

### Root Cause
The implementation incorrectly aggregates detections:
```python
# Current: Counts every detection across all frames
person_count = sum(1 for o in segment_objects if o.get('className') == 'person')
```

With YOLO processing ~3 fps on a 3-second window:
- 9 frames analyzed → 9 detections of the same person
- Result: `person_count: 9` instead of `1`

### Data Investigation Results
Analysis of YOLO output revealed:
- 108 total person detections across 44-second video
- 100 unique trackIds (not consistent across frames)
- Some timestamps have multiple detections with same trackId (partial/duplicate detections)
- Confidence filtering (>0.5) eliminates most false positives

---

## Solution Design

### Core Approach
Count the maximum number of unique persons visible at any single timestamp within the temporal window.

### Algorithm
1. **Filter by confidence** (>0.5) to eliminate false positives
2. **Group detections by timestamp** to analyze each frame
3. **Count unique trackIds per timestamp** (handles duplicate detections)
4. **Return maximum across window** (represents peak occupancy)

### Why This Works
- **Single-person videos**: Max = 1 across all timestamps
- **Multi-person videos**: Max = actual number of people when all are visible
- **Duplicate detections**: Same trackId counted once per timestamp
- **False positives**: Filtered by confidence threshold

### Implementation Strategy
Following the established pattern from `calculate_speech_metrics_for_window`, we create a modular helper function that encapsulates the complexity while keeping `process_segment` clean.

---

## Implementation Details

### File Location
`/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`

### Step 1: Add Helper Function
Insert before `process_segment` function (following the pattern of `calculate_speech_metrics_for_window`):

```python
def calculate_max_unique_persons(segment_objects, confidence_threshold=0.5):
    """
    Calculate the maximum number of unique persons visible at any point in the segment.

    This counts unique persons by finding the maximum number of unique trackIds
    at any single timestamp, after filtering by confidence.

    Args:
        segment_objects: List of object detections from YOLO
        confidence_threshold: Minimum confidence to consider a detection valid

    Returns:
        int: Maximum number of unique persons visible at any point
    """
    # Filter for person detections with sufficient confidence
    person_detections = [
        obj for obj in segment_objects
        if (obj.get('className') == 'person' or obj.get('label') == 'person')
        and obj.get('confidence', 0) >= confidence_threshold
    ]

    if not person_detections:
        return 0

    # Group by timestamp
    from collections import defaultdict
    detections_by_timestamp = defaultdict(list)

    for detection in person_detections:
        timestamp = detection.get('timestamp', 0)
        track_id = detection.get('trackId', f"unknown_{id(detection)}")
        detections_by_timestamp[timestamp].append(track_id)

    # Find maximum unique persons at any timestamp
    max_persons = 0
    for timestamp, track_ids in detections_by_timestamp.items():
        unique_persons = len(set(track_ids))
        max_persons = max(max_persons, unique_persons)

    return max_persons
```

### Step 2: Replace Existing Logic

**Current Code (line ~878):**
```python
# MVP: Count person detections specifically
person_count = sum(1 for o in segment_objects if o.get('label') == 'person' or o.get('className') == 'person')
```

**New Code:**
```python
# MVP: Count unique persons (maximum at any timestamp)
person_count = calculate_max_unique_persons(segment_objects)
```

---

## Testing & Validation

### Primary Test Case
```bash
python3 test_temporal_compute_v2.py 7515687288257465630
```

### Expected Results
| Window | Before (Wrong) | After (Correct) |
|--------|---------------|-----------------|
| Hook (0-3s) | 9 | 1 |
| Segment 1 | 24 | 1 |
| Segment 2 | 30 | 1 |
| All segments | Inflated | 1 |

### Multi-Person Validation
For videos with multiple people, the metric should show the actual maximum number of people visible at once, not the sum of all detections.

---

## Critical Analysis & Edge Cases

### Strengths of This Approach
✅ **Accurate for common cases**: Single-person and group videos correctly identified
✅ **Handles YOLO quirks**: Duplicate detections filtered by unique trackId per timestamp
✅ **Performance efficient**: O(n) complexity where n = detections in window
✅ **Consistent with user expectations**: Shows "1" for one person, "3" for three people

### Limitations & Considerations
⚠️ **TrackId reliability**: YOLO's trackIds reset between frames, but this doesn't affect our approach since we analyze per-timestamp
⚠️ **Occlusion handling**: Temporarily hidden person might get new trackId when reappearing (acceptable - we want visible count)
⚠️ **Confidence threshold**: Fixed at 0.5 might need tuning for different video qualities
⚠️ **Crowd scenes**: Very dense crowds might hit YOLO's detection limit

### Future Improvements (Post-MVP)
- Dynamic confidence threshold based on video quality
- Temporal smoothing to handle brief occlusions
- Separate metrics for `average_persons` vs `max_persons`
- Track person persistence across window

---

## Implementation Checklist

- [ ] Add `calculate_max_unique_persons` function to temporal_compute.py
- [ ] Replace existing person_count calculation line
- [ ] Run test with video 7515687288257465630
- [ ] Verify hook shows person_count: 1
- [ ] Verify all segments show person_count: 1
- [ ] Test with a multi-person video if available
- [ ] Update any documentation referencing person_count behavior

---

## Breaking Change Notice

This fix fundamentally changes what `person_count` represents:
- **Before**: Total person detections across all frames
- **After**: Maximum unique persons visible at any moment

Videos processed before this fix will have inflated person_count values. Reprocess if accurate counts are needed.