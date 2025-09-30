# ObjectCountFix.md - Fix Object Overcounting Due to Track Fragmentation

## 1. Bug Outline: Track Fragmentation Causes Object Over-Counting

### The Core Problem
ByteTrack assigns new IDs when it loses and reacquires tracking of the same object, causing temporal_compute to count them as multiple distinct objects. This is the **same fragmentation issue** we fixed for person counting in PersonFix2.md, but we incorrectly assumed "objects don't have fragmentation issues like moving people."

### Evidence: E4ExtremeDensity.mp4 Segment_1 (3s-11.8s)

**Manual Count**: 5 physical objects (apple, bottle, 2 cups, toilet paper roll)
**YOLO Detections**: 4 objects detected (apple, bottle, 2 cups — toilet paper not in YOLO training dataset)
**Current System**: 8 unique track IDs counted ❌ (double-counting due to fragmentation)
**Actual Physical Objects**: 4 (matches YOLO capability, but each tracked with 2 IDs)

#### Tracking Timeline:
```
8.56s: Scene change detected (visual transition/cut)
       ↓ ByteTrack loses tracking context

8.76s: Brief detections with IDs [10000, 10001, 10002, 10004] — 4 objects
       (0.2s after scene change, ByteTrack attempting re-acquisition)
       - Apple ID 10000:  center=(287,922), conf=0.902
       - Cup #1 ID 10001: center=(472,828), conf=0.888
       - Cup #2 ID 10002: center=(382,428), conf=0.858
       - Bottle ID 10004: center=(95,672),  conf=0.694
       - Each detected for 1 frame only (0.00s duration)

[0.31 second gap - tracking lost, re-initialization in progress]

9.07s: Persistent detections with NEW IDs [2, 3, 4, 6] — SAME 4 physical objects
       (ByteTrack stabilizes with final ID assignments)
       - Apple ID 2:  center=(286,921), conf=0.885 ← SAME APPLE (1px movement)
       - Cup #1 ID 3: center=(472,827), conf=0.855 ← SAME CUP (1px movement)
       - Cup #2 ID 4: center=(391,410), conf=0.860 ← SAME CUP (9px movement)
       - Bottle ID 6: center=(96,671),  conf=0.709 ← SAME BOTTLE (1px movement)
       - Tracked for 2+ seconds

Result: 8 unique track IDs ([10000, 10001, 10002, 10004] + [2, 3, 4, 6])
        but only 4 physical objects (apple, bottle, cup, cup)

Root Cause: Scene change at 8.56s disrupted ByteTrack's tracking context, causing
            temporary ID assignments (10000-series) before stabilizing with final
            IDs (2, 3, 4, 6) at 9.07s.
```

**Bounding Box Analysis**:
- Apple: 1 pixel movement in center, identical size → Same object
- Bottle: 1 pixel movement in center, 1-3 pixel size difference → Same object
- Cup #1: 1 pixel movement in center, 2 pixel size difference → Same object
- Cup #2: 9 pixel movement in center (minor shift), similar size → Same object

**Conclusion**: ByteTrack lost tracking for 0.31 seconds and reassigned new IDs to the same 4 physical objects (apple, bottle, 2 cups) in nearly identical positions.

**Note**: A potted plant was also detected in this video at 23s (segment_3), but is unrelated to the segment_1 fragmentation bug. The 8 overcounted objects in segment_1 are solely the 4 objects listed above, each counted twice due to track fragmentation.

### Current Behavior
**Note**: PersonFix2.md already implemented overlapping window logic for **person counting** in temporal_compute.py. Objects still use simple counting across the entire segment.

```python
# temporal_compute.py lines 1384-1392 (simple object counting - CURRENT)
for obj in segment_objects:
    if obj.get('className') != 'person':
        timestamp = obj.get('timestamp', 0)
        if start <= timestamp < end:
            instance_id = extract_instance_id(obj.get('trackId', ''))
            if instance_id is not None and obj.get('tracked', True):
                all_object_instances.add(instance_id)

object_count = len(all_object_instances)  # Bug: counts fragmented IDs

# Lines 1394-1426: Person counting uses overlapping windows (from PersonFix2.md)
# This fix extends the windowing approach to objects as well
```

Result: `object_count = 8` when there are only 4 physical objects (apple, bottle, cup, cup).

### Why PersonFix2 Assumed Objects Don't Fragment
From PersonFix2.md line 90:
> "Objects don't have fragmentation issues like moving people"

**This assumption was wrong**. While stationary objects are less prone to fragmentation, they still experience:
1. **Scene changes/cuts** (visual transitions disrupt tracking context - seen in E4ExtremeDensity at 8.56s)
2. **Brief occlusions** (person's hand passes in front)
3. **Camera movement/shake** in handheld TikTok videos
4. **Lighting changes** during transitions
5. **ByteTrack's conservative re-ID strategy** (prefers new ID over risky association)

## 2. Proposed Fix: Apply Overlapping Window Logic to Objects

### Core Insight (from PersonFix2.md)
Physical objects don't teleport. The maximum number of unique objects visible within overlapping temporal windows provides robust object counting that handles track fragmentation.

### Implementation Strategy
Apply the **same overlapping window approach** from PersonFix2.md to objects:

```python
def process_segment(seg_bounds, timelines, audio_data, ml_data, video_duration, video_id=None):
    """
    Process segment with FULL metrics.
    Updated to use overlapping windows for BOTH persons AND objects.
    """
    start = seg_bounds['start']
    end = seg_bounds['end']

    # Filter objects to segment bounds
    segment_objects = [o for o in timelines.get('object_timeline', [])
                      if start <= o.get('timestamp', 0) < end]

    # NEW: Calculate both person and object counts using overlapping windows
    WINDOW_SIZE = 0.2  # Matches 5 FPS sampling rate (0.2s per frame)
    STRIDE = 0.1       # 50% overlap between windows

    max_persons = 0
    max_objects = 0  # NEW: Track max objects per window

    # Overlapping temporal windows
    window_start = start

    while window_start < end:
        window_end = min(window_start + WINDOW_SIZE, end)
        unique_persons_in_window = set()
        unique_objects_in_window = set()  # NEW: Track objects per window

        for obj in segment_objects:
            timestamp = obj.get('timestamp', 0)

            # Check if detection falls within this window
            if window_start <= timestamp < window_end:
                instance_id = extract_instance_id(obj.get('trackId', ''))
                tracked = obj.get('tracked')
                if tracked is None:
                    logger.warning(f"Missing 'tracked' flag at {timestamp}s, defaulting to True")
                    tracked = True

                # Only count tracked detections (excludes fallbacks)
                if instance_id is not None and tracked:
                    if obj.get('className') == 'person':
                        unique_persons_in_window.add(instance_id)
                    else:
                        unique_objects_in_window.add(instance_id)  # NEW: Add to objects

        # Update maximums
        max_persons = max(max_persons, len(unique_persons_in_window))
        max_objects = max(max_objects, len(unique_objects_in_window))  # NEW

        # Slide window by STRIDE
        window_start += STRIDE

    person_count = max_persons
    object_count = max_objects  # NEW: Use windowed count instead of set length
```

### Why This Works
1. **Handles Track Fragmentation**: Objects with multiple IDs (10000→2) won't appear in same window due to 0.31s gap > 0.2s window
2. **Boundary Robustness**: 50% overlap ensures detections near window edges are captured
3. **Conservative Counting**: Takes maximum across windows, accounting for brief occlusions
4. **Consistent Logic**: Same approach proven to work for person counting
5. **No False Negatives**: All legitimate objects in any window get counted
6. **Distinguishes Fragmentation vs. Brief Objects**: Timing separation naturally filters fragmentation while preserving legitimate brief detections (see Risk 3.2 analysis)

### Expected Results
**E4ExtremeDensity.mp4 Segment_1**:
- Window covering 8.76s: [10000, 10001, 10002, 10004] → 4 objects (apple, bottle, 2 cups)
- Windows covering 9.07s+: [2, 3, 4, 6] → 4 objects (same physical objects, new IDs)
- **Max = 4 objects** ✅ (down from 8, correctly counts physical objects)

## 3. Deep Discovery: Implementation Risks and Mitigation

### 3.1 Risk: Under-Counting with Rapid Object Changes

**Scenario**: What if objects genuinely enter/exit rapidly and are never all visible simultaneously?
```
Window 1 (3.0-3.2s): Objects [A, B] → count = 2
Window 2 (3.1-3.3s): Objects [B, C] → count = 2
Window 3 (3.2-3.4s): Objects [C, D] → count = 2
Max = 2, but 4 distinct objects existed
```

**Likelihood**: VERY LOW
- Objects in TikTok videos are typically **persistent** (held, placed on table, worn)
- Unlike people who move constantly, objects tend to stay in frame
- 50% window overlap (0.1s stride) catches most transitions
- E4ExtremeDensity data shows objects persist for **2+ seconds** once tracked

**Mitigation**:
- This is the **fundamental tradeoff**: Over-counting (current) vs. Under-counting (rare edge case)
- Under-counting is **preferable** for ML training - avoids false signals
- Real-world TikTok content rarely has objects appearing/disappearing in <0.2s intervals

**Evidence from Test Data**:
```
E4ExtremeDensity.mp4 segment_1 (8.8 seconds):
- 4 physical objects tracked: apple, bottle, 2 cups
- Persistent detections (IDs 2, 3, 4, 6): 2+ seconds visibility
- Brief detections (IDs 10000, 10001, 10002, 10004): 1 frame only (fragmentation artifacts)
- Fragmentation gap: 0.31 seconds (larger than 0.2s window size)
```

### 3.1.1 Edge Case: Empty Windows (object_count = 0)

**Scenario**: What if no objects are detected in any window?
- Very short segment (e.g., 0.15s closing window)
- Empty scene (person talking, no objects held/visible)
- All detections filtered due to `tracked=False`

**Decision**: **Accept 0 as valid**
- If no tracked objects are detected, `max_objects = 0` is **correct behavior**
- Empty scenes are legitimate (many TikTok videos show just a person speaking)
- No validation logging or fallback logic needed
- Keeps implementation simple and predictable

**Example**:
```python
# All windows have no objects
max_objects = 0
object_count = 0  # Correct: no objects in segment
```

### 3.2 Risk: Legitimate Brief Object Detections Ignored

**Scenario**: Object appears briefly (e.g., phone quickly shown to camera) and gets filtered out because it's only in one window with lower count.
```
Window 1 (5.0-5.2s): [apple, bottle, cup] → 3 objects
Window 2 (5.1-5.3s): [apple, bottle, cup, phone] → 4 objects
Window 3 (5.2-5.4s): [apple, bottle, cup] → 3 objects
Max = 4 (correct!)
```

**Likelihood**: NOT A RISK
- The max() function captures brief but legitimate detections
- As long as the object appears in **any window**, it contributes to the max count
- The windowing approach naturally handles two distinct cases:

**Case 1 - Fragmentation (handled correctly)**:
```
8.76s: 4 objects with IDs [10000, 10001, 10002, 10004] (brief, 1 frame each)
       Apple, bottle, 2 cups - fragmentation artifacts
[0.31s gap - different windows]
9.07s: Same 4 objects with NEW IDs [2, 3, 4, 6] (persistent, 20+ frames)
       Apple, bottle, 2 cups - stable tracking resumes
→ Different windows capture different ID sets
→ Max = 4 (both windows show same 4 physical objects)
```

**Case 2 - Legitimate Brief Object (handled correctly)**:
```
5.10s: Phone ID 5 briefly visible (wave gesture)
5.12s: Phone leaves frame
Window covering 5.0-5.2s: [apple, bottle, cup, phone] → 4 objects
→ Max captures this window, phone counted ✓
```

**Why This Works Without Additional Logic**:
- Fragmentation artifacts have 0.31s gaps (from E4ExtremeDensity data) which place duplicate IDs in **different windows**
- The max() function naturally selects the window with the persistent detections (higher confidence = more frames = more detections)
- Legitimate brief objects appear in their own window and contribute to max
- **No bounding box validation needed** - timing separation is sufficient

### 3.3 Risk: Objects Re-IDed Within Same Window

**Scenario**: Object loses tracking and gets reassigned new ID within 0.2s window
```
Window [9.9-10.1s]:
  - Apple tracked as ID 2 at 9.95s
  - Apple loses and regains tracking
  - Apple tracked as ID 4 at 10.05s
  Result: Count = 2 (should be 1) ❌
```

**Likelihood**: VERY LOW (from PersonFix2.md analysis)
- Requires track loss and re-acquisition within 200ms
- ByteTrack typically takes 300-500ms to reassign IDs (preliminary evidence: E4ExtremeDensity shows 0.31s gap)
- Would require extreme occlusion + rapid movement + immediate reappearance

**Evidence** (Limited - needs validation):
- E4ExtremeDensity shows 0.31s gap between re-IDs (1.5x window size)
  - Scene change at 8.56s triggered fragmentation
  - Temporary IDs at 8.76s (0.2s later)
  - Final IDs at 9.07s (0.31s after temporary IDs)
- PersonFix2.md documented this as VERY LOW likelihood for 0.2s windows
- **Note**: This is based on one data point - Phase 2.5 will validate across multiple videos

**Mitigation**:
- This is an **acceptable edge case** given the overwhelming benefit
- Alternative approaches (bbox IoU matching, trajectory prediction) are far more complex
- Risk/benefit strongly favors windowing approach
- **Phase 2.5 testing** will validate re-ID timing assumptions before production deployment

### 3.4 Risk: Breaking Change for ML Training Data

**Impact Assessment**: ✅ DESIRED CHANGE
- Current object counts are **inflated by ~2x** due to fragmentation
- ML model is learning from **incorrect ground truth** (8 objects when there are 4)
- Fixing this **improves training data quality**

**Backwards Compatibility**: NOT A CONCERN (per requirements)
- No rollback option needed
- Immediate aggressive implementation
- Reprocessing existing videos will improve ML training

**Data Migration**: NONE REQUIRED
- Changes affect only `temporal_compute.py` processing
- Existing stored videos can be reprocessed on-demand
- New videos get correct counts immediately

### 3.5 Risk: Performance Impact

**Window Processing Cost**:
```python
# Current: O(n) single pass through objects
for obj in segment_objects:
    all_object_instances.add(instance_id)

# New: O(n * w) where w = number of windows
for window in windows:
    for obj in segment_objects:
        if window_start <= timestamp < window_end:
            unique_objects_in_window.add(instance_id)
```

**Analysis**:
- Segment duration: Typically 2-9 seconds
- Window count: `(duration - 0.2) / 0.1 + 1` = 18-88 windows
- Object detections: 10-50 per segment (sampled at 5 FPS)
- Total comparisons: 180-4400 (negligible for modern CPU)

**Measured Impact** (from PersonFix2.md implementation):
- Person counting already uses this approach with no performance issues
- Adding objects to same loop has **zero additional overhead**
- Total temporal_compute time: <1 second per video (unchanged)

**Conclusion**: Performance impact is **negligible**.

## 4. Implementation Plan

### Phase 1: Update temporal_compute.py
**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`
**Lines**: 1373-1429

**Context**: PersonFix2.md already implemented overlapping window logic for person counting (lines 1394-1426). This fix **extends the existing windowing loop** to also handle objects, rather than creating new infrastructure.

**Changes**:
1. Remove separate object counting logic (lines 1384-1392)
2. Add `max_objects = 0` alongside existing `max_persons = 0` (line ~1381)
3. Add `unique_objects_in_window = set()` inside the existing window loop (line ~1403)
4. Modify object detection logic to add to window set instead of global set (line ~1420)
5. Add `max_objects = max(max_objects, len(unique_objects_in_window))` in window loop (line ~1424)
6. Change `object_count = len(all_object_instances)` to `object_count = max_objects` (line ~1429)

### Phase 2: Test with Known Cases
**Test Videos**:
1. E4ExtremeDensity.mp4 - Expect object_count to drop from 8→4 in segment_1 (apple, bottle, 2 cups correctly counted once each)
2. Video05ObjectsGestures.mp4 - Verify no regression
3. Video10TwoPeople.mp4 - Verify person counting still works

### Phase 2.5: Validate Re-ID Timing Assumptions
**Purpose**: Verify the 300-500ms re-ID timing claim used in Risk 3.3 analysis.

**Method**:
1. Process 5-10 videos with object tracking fragmentation
2. For each fragmentation event, measure the gap between duplicate IDs
3. Calculate distribution: min, max, median, 90th percentile
4. Verify that <5% of re-IDs happen within 0.2s window

**Test Videos**:
- E4ExtremeDensity.mp4 (known fragmentation at 8.76s→9.07s = 0.31s)
- Video05ObjectsGestures.mp4
- Video11SpeakerSpeech.mp4
- 15SpeechTest.mp4
- 2-3 additional videos with handheld camera movement

**Success Criteria**:
- ✅ ≥95% of re-ID gaps are >0.2s (outside single window)
- ✅ Median re-ID gap is 0.3-0.5s (validates 300-500ms claim)
- ⚠️ If >5% are <0.2s, consider reducing window size to 0.15s

**Script for Analysis**:
```python
# Extract all object detections, identify fragmentation events
# (same class, similar bbox, different IDs within short time)
# Measure time gap between last detection of old ID and first of new ID
```

### Phase 3: Validate with Manual Counts
Compare system output against manual frame-by-frame counting for 3-5 videos to ensure accuracy.

## 5. Code Changes

**Important Context**: The code below shows the **current state** after PersonFix2.md implementation. Persons already use overlapping windows (lines 1394-1426). This fix modifies the existing windowing loop to also process objects, eliminating the separate simple object counting.

### Before (Current Implementation):
```python
# Lines 1380-1429 in temporal_compute.py (AFTER PersonFix2.md)
max_persons = 0
all_object_instances = set()  # Simple set for objects - TO BE REMOVED

# Process objects SEPARATELY with simple counting (lines 1384-1392)
for obj in segment_objects:
    if obj.get('className') != 'person':
        timestamp = obj.get('timestamp', 0)
        if start <= timestamp < end:
            instance_id = extract_instance_id(obj.get('trackId', ''))
            if instance_id is not None and obj.get('tracked', True):
                all_object_instances.add(instance_id)

# Process persons with overlapping windows (PersonFix2.md - lines 1394-1426)
window_start = start
while window_start < end:
    window_end = min(window_start + WINDOW_SIZE, end)
    unique_persons_in_window = set()

    for obj in segment_objects:
        if obj.get('className') == 'person':
            timestamp = obj.get('timestamp', 0)
            if window_start <= timestamp < window_end:
                instance_id = extract_instance_id(obj.get('trackId', ''))
                tracked = obj.get('tracked')
                if tracked is None:
                    logger.warning(f"Missing 'tracked' flag at {timestamp}s, defaulting to True")
                    tracked = True
                if instance_id is not None and tracked:
                    unique_persons_in_window.add(instance_id)

    max_persons = max(max_persons, len(unique_persons_in_window))
    window_start += STRIDE

person_count = max_persons
object_count = len(all_object_instances)  # Bug: counts fragmented IDs
```

### After (Fixed Implementation):
```python
# Lines 1380-1429 in temporal_compute.py (UPDATED)
# Note: Extends PersonFix2.md's existing windowing loop to handle objects
max_persons = 0
max_objects = 0  # NEW: Track max objects per window instead of separate set

# REMOVED: Separate object counting loop (lines 1384-1392 deleted)

# Process BOTH persons and objects with overlapping windows (MODIFIED EXISTING LOOP)
window_start = start
while window_start < end:
    window_end = min(window_start + WINDOW_SIZE, end)
    unique_persons_in_window = set()
    unique_objects_in_window = set()  # NEW

    for obj in segment_objects:
        timestamp = obj.get('timestamp', 0)

        # Check if detection falls within this window
        if window_start <= timestamp < window_end:
            instance_id = extract_instance_id(obj.get('trackId', ''))

            # Check tracked flag with logging for missing values
            tracked = obj.get('tracked')
            if tracked is None:
                logger.warning(f"Missing 'tracked' flag at {timestamp}s, defaulting to True")
                tracked = True

            # Only count tracked detections (excludes fallbacks)
            if instance_id is not None and tracked:
                if obj.get('className') == 'person':
                    unique_persons_in_window.add(instance_id)
                else:
                    unique_objects_in_window.add(instance_id)  # NEW

    # Track maximum across all windows
    max_persons = max(max_persons, len(unique_persons_in_window))
    max_objects = max(max_objects, len(unique_objects_in_window))  # NEW

    window_start += STRIDE

person_count = max_persons
object_count = max_objects  # FIXED: Use windowed max instead of set length
```

### Key Differences:
1. **Removed**: `all_object_instances = set()` (line 1382)
2. **Added**: `max_objects = 0` (line 1381)
3. **Added**: `unique_objects_in_window = set()` inside loop (line 1403)
4. **Modified**: Object detection now adds to window set, not global set (line 1420)
5. **Added**: `max_objects = max(max_objects, len(unique_objects_in_window))` (line 1424)
6. **Modified**: `object_count = max_objects` instead of `len(all_object_instances)` (line 1429)

## 6. Expected Impact

### Test Case: E4ExtremeDensity.mp4 Segment_1 (3s-11.8s)

**Before**:
```json
{
  "start": 3.0,
  "end": 11.8,
  "object_count": 8,  // ❌ Counting 8 unique track IDs [10000, 10001, 10002, 10004, 2, 3, 4, 6]
  "person_count": 1
}
```

**After**:
```json
{
  "start": 3.0,
  "end": 11.8,
  "object_count": 4,  // ✅ Correctly counts 4 physical objects (apple, bottle, cup, cup)
  "person_count": 1   // ✅ Unchanged (person logic already uses windows)
}
```

**Physical Objects in Video**: Apple, bottle, 2 cups (toilet paper roll visible but not in YOLO training dataset)

### Broader Impact:
- Videos with object tracking fragmentation will see **reduced object counts** (more accurate)
- Videos with stable tracking will see **no change** (already correct)
- ML training data quality **improves** (removes false positive object detections)
- Person counting **unchanged** (already using windowed approach)

## 7. Validation Criteria

### Success Metrics:
1. ✅ E4ExtremeDensity segment_1 drops from 8→4 objects (apple, bottle, 2 cups)
2. ✅ Bounding box analysis confirms 4 YOLO-detectable objects (toilet paper not in training dataset)
3. ✅ Person counts remain unchanged across all test videos
4. ✅ No performance degradation (temporal_compute < 1s per video)
5. ✅ Phase 2.5: Re-ID timing validation shows ≥95% gaps >0.2s
6. ✅ Manual spot-checks on 3-5 videos confirm accuracy

### Failure Conditions:
- Object counts increase (would indicate logic error)
- Person counts change (would indicate regression)
- Processing time > 2x current baseline
- Manual verification shows under-counting of legitimate objects

## 8. Rollback Plan

**Per requirements**: No rollback option, immediate aggressive implementation.

**Justification**:
- Current behavior is objectively wrong (counts duplicate IDs as distinct objects)
- Fix is low-risk (same proven approach as PersonFix2.md)
- Benefits outweigh edge case risks
- Backwards compatibility not a concern

## 9. Summary

**Problem**: ByteTrack track fragmentation causes object overcounting (8 unique track IDs when only 4 physical objects exist)

**Root Cause**: temporal_compute.py counts all unique instance IDs across segment without accounting for re-identification of same objects

**Evidence**: E4ExtremeDensity.mp4 segment_1 shows 4 objects (apple, bottle, 2 cups) tracked twice with IDs [10000, 10001, 10002, 10004] → [2, 3, 4, 6] due to scene change at 8.56s

**Solution**: Apply overlapping window approach (proven for person counting) to objects

**Risk Level**: LOW
- Same approach already working for person counting
- E4ExtremeDensity data shows 0.31s re-ID gap (larger than 0.2s window)
- No evidence of sub-window fragmentation in test data

**Impact**: Improved ML training data quality, no breaking changes, negligible performance cost

**Next Step**: Implement Phase 1 changes to temporal_compute.py
