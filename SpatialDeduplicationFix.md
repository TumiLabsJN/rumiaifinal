# Spatial Deduplication Fix for Person Counting

## Problem Statement

**Person counting is completely broken due to YOLO tracking system creating excessive track IDs for single individuals.**

### Evidence: Single-Person Video Analysis

**Video 7480428850522950920** contains only 1 person throughout entire duration, yet temporal windows show:

| Window | Expected | Actual | Issue |
|--------|----------|--------|-------|
| Hook (0-3s) | 1 | 1 | ✅ Correct |
| Segment_1 (3-14.8s) | 1 | 2 | ❌ Wrong |
| Segment_2 (14.8-26.6s) | 1 | 2 | ❌ Wrong |
| Segment_3 (26.6-38.4s) | 1 | 2 | ❌ Wrong |
| Segment_4 (38.4-50.2s) | 1 | 2 | ❌ Wrong |
| Segment_5 (50.2-62s) | 1 | 2 | ❌ Wrong |
| Closing (62-65s) | 1 | 1 | ✅ Correct |

**Pattern**: ALL middle segments incorrectly show 2 people, while hook/closing show correct count of 1.

## Root Cause Analysis

### YOLO Tracking System Breakdown

**Current Algorithm** (in `temporal_compute.py`):
```python
# Count unique track IDs across time window
unique_track_ids = set()
for person_detection in window_detections:
    track_id = person_detection['track_id']
    unique_track_ids.add(track_id)

person_count = len(unique_track_ids)  # ← BROKEN
```

**What's Actually Happening:**

**Segment_1 (3.0s - 14.8s)**:
- **Expected**: 1 person → 1 consistent track ID
- **Actual**: 1 person → **32 different track IDs**!
- **Track IDs**: `obj_105_1`, `obj_120_1`, `obj_135_1`, `obj_330_1`, `obj_330_6`, `obj_345_1`, `obj_345_6`, etc.

**Critical Evidence - Double Detection at Same Timestamp:**
```
15.0s: 2 people [('obj_450_1', 0.738), ('obj_450_6', 0.593)]
15.5s: 2 people [('obj_465_1', 0.805), ('obj_465_6', 0.628)]
16.0s: 2 people [('obj_480_1', 0.852), ('obj_480_6', 0.604)]
```

**The same person is detected twice with different track IDs at identical timestamps.**

### Why Tracking Fails

1. **Track ID Explosion**: Object tracker creates new track IDs every frame instead of maintaining consistency
2. **Multiple Detection**: Same person gets multiple bounding boxes with different track IDs
3. **Confidence Variations**: Different parts of person (face vs body) get separate track IDs
4. **Temporal Inconsistency**: No correlation between track IDs across time

### Impact on ML Pipeline

**Data Quality Corruption:**
- ✅ **Single-person content** misclassified as **multi-person content**
- ❌ **ML features** based on person interaction become meaningless
- ❌ **Content categorization** fails for recommendation algorithms
- ❌ **Behavioral analysis** metrics become unreliable

## Proposed Solution: Spatial Deduplication

### Algorithm: Max Simultaneous People

**Replace track ID counting with spatial-temporal analysis:**

```python
def calculate_person_count_fixed(person_detections, window_start, window_end):
    """Calculate max simultaneous people using spatial deduplication."""

    # Group detections by timestamp
    detections_by_time = {}
    for detection in person_detections:
        timestamp = detection['start']
        if window_start <= timestamp < window_end:
            if timestamp not in detections_by_time:
                detections_by_time[timestamp] = []
            detections_by_time[timestamp].append(detection)

    max_people = 0

    # For each timestamp, count non-overlapping people
    for timestamp, detections in detections_by_time.items():
        # Apply spatial deduplication
        unique_people = spatial_dedup_people(detections)
        max_people = max(max_people, len(unique_people))

    return max_people

def spatial_dedup_people(detections):
    """Remove overlapping detections that represent the same person."""
    if not detections:
        return []

    # Sort by confidence (keep highest confidence detections)
    detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)

    unique_people = []
    for detection in detections:
        bbox = detection.get('bbox', [])
        if not bbox:
            continue

        # Check if this detection overlaps significantly with existing people
        is_duplicate = False
        for existing in unique_people:
            existing_bbox = existing.get('bbox', [])
            if bbox_overlap_ratio(bbox, existing_bbox) > 0.5:  # 50% overlap threshold
                is_duplicate = True
                break

        if not is_duplicate:
            unique_people.append(detection)

    return unique_people

def bbox_overlap_ratio(bbox1, bbox2):
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)

    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0  # No overlap

    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
```

### Why This Approach Works

**Spatial Deduplication Logic:**
1. **Group by timestamp**: Analyze each moment independently
2. **Bounding box overlap**: Detections with >50% overlap = same person
3. **Confidence prioritization**: Keep highest confidence detection when duplicates found
4. **Max aggregation**: Take peak simultaneous people across all timestamps

**Expected Results:**
- **Video 7480428850522950920**: All segments → person_count = 1 ✅
- **True multi-person content**: Correctly count max simultaneous people
- **No double-counting**: Overlapping detections merged

## Implementation Strategy

### Phase 1: Core Fix (30 minutes)

**File**: `/rumiai_v2/processors/temporal_compute.py`

**Location**: Person counting logic around line 1400-1500

**Changes**:
1. Add `spatial_dedup_people()` function
2. Add `bbox_overlap_ratio()` function
3. Replace track ID counting with spatial deduplication in person count calculation

### Phase 2: Validation (15 minutes)

**Test Cases**:
1. **Video 7480428850522950920**: Verify all segments show person_count = 1
2. **Multi-person video**: Verify accurate peak people counting
3. **Edge cases**: Test detection gaps, confidence variations

### Phase 3: Monitoring (Ongoing)

**Metrics to Track**:
- **Person count distribution**: Before vs after fix
- **Single-person content percentage**: Should increase significantly
- **ML model accuracy**: Improved content categorization

## Risk Assessment

### Low-Risk Implementation

**What Changes**:
- ✅ **Person counting algorithm only** - isolated change
- ✅ **Same data structures** - no pipeline modifications
- ✅ **Backward compatible** - doesn't break existing code

**What Doesn't Change**:
- ✅ **Object detection system** - still gets same detections
- ✅ **Track ID generation** - underlying tracking unchanged
- ✅ **Other features** - gesture, gaze, emotion analysis unaffected

### Potential Edge Cases

**Scenario 1: Brief Overlaps**
```
Problem: Person A + Person B appear together for 1 second in 10-second segment
Current: person_count = 2 (overstates multi-person nature)
Solution: Accept this - peak occupancy is meaningful metric
```

**Scenario 2: Detection Gaps**
```
Problem: Person temporarily undetected due to motion blur
Impact: Minimal - spatial deduplication doesn't depend on continuous tracking
```

**Scenario 3: Low Confidence Detections**
```
Problem: Multiple low-confidence detections of same person
Solution: Confidence prioritization ensures highest quality detection kept
```

## Expected Impact

### Before Fix (Broken State)
```json
{
  "temporal_windows": {
    "middle_segments": [
      {"person_count": 2},  // ❌ Wrong - should be 1
      {"person_count": 2},  // ❌ Wrong - should be 1
      {"person_count": 2}   // ❌ Wrong - should be 1
    ]
  }
}
```

### After Fix (Expected Results)
```json
{
  "temporal_windows": {
    "middle_segments": [
      {"person_count": 1},  // ✅ Correct
      {"person_count": 1},  // ✅ Correct
      {"person_count": 1}   // ✅ Correct
    ]
  }
}
```

### Data Quality Improvements

**Content Classification**:
- **Single-person content**: Correctly identified (major improvement)
- **Multi-person content**: More accurate peak people counting
- **ML training data**: Clean person count features for better model performance

**Behavioral Features**:
- **Person interaction metrics**: Only calculated when truly multi-person
- **Individual vs group behaviors**: Proper categorization basis
- **Content recommendation**: Better understanding of social vs solo content

## Alternative Solutions Considered

### Option A: Fix YOLO Tracking System
**Pros**: Addresses root cause, enables individual tracking
**Cons**: Major system change, weeks of development, high risk

### Option B: Track ID Clustering
**Pros**: Maintains some tracking information
**Cons**: Complex algorithm, still relies on broken tracking data

### Option C: Confidence-Based Filtering
**Pros**: Simple implementation
**Cons**: Doesn't solve double-detection at same timestamp

## Decision Rationale

**Why Spatial Deduplication**:
1. **Immediate fix**: Solves the core problem in 30 minutes
2. **Low risk**: Isolated change with clear boundaries
3. **Accurate results**: Better than broken tracking system
4. **ML-focused**: Optimized for feature accuracy over tracking detail

The person_count metric is primarily used for **content categorization** (single vs multi-person) rather than detailed tracking analysis, making spatial deduplication the optimal solution.

## Implementation Timeline

**Total: 45 Minutes**

**Minutes 1-30: Core Implementation**
- Add spatial deduplication functions to temporal_compute.py
- Replace track ID counting logic with spatial analysis
- Handle edge cases (empty detections, missing bbox data)

**Minutes 31-45: Validation & Testing**
- Test on video 7480428850522950920
- Verify all segments show person_count = 1
- Spot check multi-person video for accuracy

**No rollback preparation needed** - change is isolated and easily reversible.

---

## Status: Ready for Implementation

**All requirements identified**:
- ✅ Problem diagnosed (YOLO tracking creates excessive track IDs)
- ✅ Solution designed (spatial deduplication with IoU-based overlap detection)
- ✅ Implementation plan detailed (45-minute timeline)
- ✅ Risk assessment completed (low risk, high accuracy improvement)
- ✅ Testing strategy defined (validation video identified)

**Next Step**: Implement spatial deduplication fix in temporal_compute.py to resolve person counting accuracy issues.