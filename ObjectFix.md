# Object Count and Tracking Bug Documentation

**Date:** 2025-10-21
**Severity:** 🟡 MEDIUM - Affects object_count metric reliability
**Status:** Root cause identified, fix proposed, not yet implemented
**Related:** PersonCountFix.md (same root cause)

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Discovery](#problem-discovery)
3. [Critical Discovery: Fallback IDs Have Two Meanings](#critical-discovery-fallback-ids-have-two-meanings)
4. [Root Cause Analysis](#root-cause-analysis)
5. [Impact Assessment](#impact-assessment)
6. [Proposed Fix (Revised)](#proposed-fix-revised)
7. [Testing & Validation](#testing--validation)
8. [Implementation Notes](#implementation-notes)

---

## Executive Summary

**Problem:** The `object_count` metric counts all detected object classes including those without successful tracking, which reduces metric reliability.

**Root Cause:** Same as person_count bug - the `tracked` field from YOLO is dropped by timeline_builder.py, and temporal_compute.py defaults missing field to `True`, causing all objects (including untracked detections) to be counted.

**Impact:**
- Lower severity than person_count (class deduplication limits inflation)
- However, misclassified objects and noise get counted
- Reduces reliability of object_count metric for ML training

**Fix (Revised):** Change default from `True` to `False` and preserve `tracked` field through pipeline. When tracking fails completely, return object_count = 0 (honest about data quality) rather than returning inflated/noisy counts.

**Why the revision?** Initial proposal to filter fallback IDs (obj_10000+) was too aggressive - it would remove legitimate objects when ByteTrack fails completely. The real issue is the missing field propagation, not the presence of fallback IDs.

---

## Problem Discovery

### Initial Context

While investigating the person_count inflation bug (see PersonCountFix.md), discovered that object_count uses the same logic and has the same missing `tracked` field issue.

### Example Video Analysis #1

**Video:** `7531126454034189599` (Hammond's Candies factory video)

**Metadata:**
```json
{
  "description": "Let's see what creative names you come up with #hammondscandies #candy #sweets #satisfying #candyfactory",
  "author": "hammondscandies"
}
```

**Output:**
```json
{
  "temporal_windows": {
    "hook": {
      "object_count": 14,
      "person_count": 124
    }
  }
}
```

### What YOLO Detected

**Hook window (0-3s) object classes:**
```
person: 124 detections
baseball glove: 25 detections
motorcycle: 22 detections
banana: 10 detections
skateboard: 9 detections
hot dog: 6 detections
fire hydrant: 4 detections
apple: 2 detections
bowl: 2 detections
toothbrush: 2 detections
bottle: 1 detection
chair: 1 detection
sink: 1 detection
train: 1 detection
vase: 1 detection
```

**Total:** 15 unique object classes (excluding person)

### Tracking Analysis

**Skateboard detections (example):**
```bash
cat unified/7531126454034189599.json | \
  jq '[.timeline.entries[] | select(.entry_type == "object" and .data.class == "skateboard" and .start < 3)] |
      group_by(.data.track_id) |
      map({track: .[0].data.track_id, count: length})'
```

**Result:**
```json
[
  {"track": "obj_10000", "count": 1},
  {"track": "obj_10002", "count": 1},
  {"track": "obj_10006", "count": 1},
  {"track": "obj_10008", "count": 1},
  {"track": "obj_10016", "count": 1},
  {"track": "obj_10181", "count": 1},
  {"track": "obj_10191", "count": 1},
  {"track": "obj_10194", "count": 1},
  {"track": "obj_10197", "count": 1}
]
```

**Initial Analysis:** 9 skateboard detections, ALL with fallback tracking IDs (obj_10000+). Each detection appears in only 1 frame, indicating:
1. Tracking completely failed
2. These are likely misclassifications of candy/factory equipment
3. YOLO incorrectly classified candy as "skateboard", "motorcycle", "banana", etc.

---

### Example Video Analysis #2

**Video:** `7555692040944749855` (Candy taste test video)

**Metadata:**
```json
{
  "author": "astonsplayroom",
  "description": "#fyp #candy #tastetest #foryoupage"
}
```

**Objects detected:**
```
cake:          45 detections → ALL fallback IDs (obj_10006 to obj_10069)
dining table:  19 detections → mostly obj_12 (16x), some fallback
bed:           12 detections → ALL obj_5 (good tracking)
bottle:        10 detections → ALL obj_6 (good tracking)
suitcase:       6 detections → fallback IDs
baseball glove: 2 detections → fallback IDs
toothbrush:     1 detection  → fallback IDs
```

**Current object_count:** 7 classes

---

### Example Video Analysis #3

**Video:** `7489503844997647646` (Celery juice video - GOOD tracking)

**Objects detected:**
```
refrigerator: 114 detections → obj_5 (113x), obj_3 (1x) - good tracking
banana:        60 detections → good tracking
bottle:        38 detections → good tracking
toothbrush:     6 detections → obj_4 (4x), obj_10002 (1x), obj_10009 (1x)
remote:         6 detections → good tracking
```

**Key finding**: Even in a video with GOOD tracking, we see fallback IDs:
- **toothbrush**: 4 detections with obj_4 (real tracking), 2 detections with fallback IDs
- These fallback IDs might be legitimate transient objects (person picks up toothbrush briefly)

---

## Critical Discovery: Fallback IDs Have Two Meanings

### The Question That Changed Everything

**Should we filter out ALL objects with fallback tracking IDs (obj_10000+)?**

Initial hypothesis: YES - fallback IDs indicate misclassifications
After deeper analysis: **NO** - the situation is more nuanced

### Scenario A: ByteTrack Failed Completely (Frame 0)

**Example:** Videos 7531126454034189599, 7555692040944749855

**Pattern:**
- Frame 0 already has obj_10000, obj_10001, obj_10002
- ByteTrack never initialized successfully
- **ALL detections** across the entire video get fallback IDs
- **Even legitimate objects** show fallback IDs

**Video 7555692040944749855 analysis:**
- **"dining table"**: 16 detections with obj_12 (REAL tracking) ✅
- **"bed"**: 12 detections with obj_5 (REAL tracking) ✅
- **"bottle"**: 10 detections with obj_6 (REAL tracking) ✅
- **"cake"**: 45 detections, ALL fallback IDs (probably candy misclassified)

**Critical insight:**
- When ByteTrack fails from frame 0, SOME objects still get real tracking IDs
- But MANY legitimate objects get fallback IDs
- Fallback IDs don't necessarily mean "bad object" - they mean "tracking failed, but object might be real"
- **Filtering ALL fallback IDs would incorrectly remove legitimate objects**

---

### Scenario B: ByteTrack Working, Object Appears Briefly

**Example:** Video 7489503844997647646 toothbrush

**Pattern:**
- Most objects have real tracking (obj_1 to obj_999)
- A few objects appear in 1-2 frames and get fallback IDs
- These could be:
  - Transient real objects (person picks up something briefly)
  - Misclassifications (edge of object looks like something else)

**In this case:**
- Fallback IDs could represent either real objects or noise
- Cannot definitively say fallback = bad

---

### What About the "Cake" Problem?

**Issue:** Candy video shows candy, YOLO detects "cake" 45 times with fallback IDs

**Is this a tracking problem or classification problem?**

**Answer:** This is a **classification problem**, not a tracking problem.

**Evidence:**
1. YOLO classified candy as "cake" (wrong but understandable - both are sweets)
2. It did so consistently (45 detections)
3. Tracking failed, so each detection got a unique fallback ID
4. But the underlying issue is: **YOLO doesn't have a "candy" class in COCO dataset**

**Implication:** Even if tracking worked perfectly, YOLO would still misclassify candy as "cake", "donut", or "hot dog". This is a limitation of YOLO's COCO training data, not a tracking bug.

---

## Root Cause Analysis

### The Code Path

#### Step 1: YOLO Generates Tracking Data

**File:** `rumiai_v2/api/ml_services_unified.py`
**Lines:** 315-337

```python
for box in detection.boxes:
    # Try to get real tracking ID, fall back if needed
    if hasattr(box, 'id') and box.id is not None:
        instance_id = int(box.id)
        is_tracked = True
    else:
        # Generate fallback ID for untracked detection
        instance_id = self.next_fallback_id
        self.next_fallback_id += 1
        is_tracked = False

    results.append({
        'trackId': f"obj_{instance_id}",
        'className': model.names[int(box.cls)],
        'confidence': float(box.conf),
        'timestamp': frame_data.timestamp,
        'bbox': box.xyxy[0].tolist(),
        'frame_number': frame_data.frame_number,
        'tracked': is_tracked  # ✅ Field generated here
    })
```

**Key point:** `tracked` field indicates whether ByteTrack successfully tracked the object (`True`) or if it's a fallback ID (`False`).

---

#### Step 2: Timeline Builder Drops the Field

**File:** `rumiai_v2/processors/timeline_builder.py`
**Lines:** 110-122

```python
entry = TimelineEntry(
    start=timestamp,
    end=None,
    entry_type='object',
    data={
        'class': obj_class,
        'confidence': frame_data.get('confidence', 0),
        'bbox': frame_data.get('bbox', []),
        'track_id': frame_data.get('trackId', frame_data.get('track_id', None))
        # ❌ MISSING: 'tracked' field
    }
)
```

**Impact:** The `tracked` field is NOT included in timeline entries.

---

#### Step 3: Temporal Compute Extracts Objects

**File:** `rumiai_v2/processors/temporal_compute.py`
**Lines:** 896-906

```python
# Convert timeline entries to object_timeline
object_timeline = []
for entry in timeline_entries:
    if entry.get('entry_type') == 'object':
        object_timeline.append({
            'timestamp': entry.get('start', 0),
            'className': entry.get('data', {}).get('class', 'unknown'),
            'confidence': entry.get('data', {}).get('confidence', 0),
            'bbox': entry.get('data', {}).get('bbox', []),
            'trackId': entry.get('data', {}).get('track_id', None)
            # ❌ 'tracked' field not copied
        })
```

**Impact:** Even if the field existed in timeline entries, it wouldn't be copied here.

---

#### Step 4: Object Count Calculation

**File:** `rumiai_v2/processors/temporal_compute.py`
**Lines:** 2060-2103

```python
unique_object_classes = set()

# Process objects first: count unique class names (not instances)
for obj in segment_objects:
    if obj.get('className') != 'person':
        timestamp = obj.get('timestamp', 0)
        if start <= timestamp < end:
            tracked = obj.get('tracked', True)  # ❌ Defaults to True
            if tracked:
                unique_object_classes.add(obj.get('className'))

# Later...
object_count = len(unique_object_classes)
```

**The Bug:**
1. Gets `tracked` field (which doesn't exist)
2. Defaults to `True` (wrong assumption)
3. ALL objects counted, including untracked detections

---

## Impact Assessment

### Severity Comparison: object_count vs person_count

| Metric | Bug Impact | Severity |
|--------|-----------|----------|
| **person_count** | Each fallback ID = separate person<br/>124 tracks → person_count = 124 | 🔴 CRITICAL |
| **object_count** | Counts unique classes only<br/>Multiple fallback IDs of same class → counted once | 🟡 MEDIUM |

### Why object_count Impact is Lower

**Example:** Video with tracking failure

**Person detections:**
```
obj_10001: person (1 frame)
obj_10002: person (1 frame)
obj_10003: person (1 frame)
... 121 more fallback IDs, all class="person"
```
- person_count = 124 (massively inflated) ❌

**Object detections:**
```
obj_10000: skateboard (1 frame)
obj_10002: skateboard (1 frame)
obj_10006: skateboard (1 frame)
... 6 more fallback IDs, all class="skateboard"
```
- All added to set as "skateboard"
- object_count increments by 1 (not 9) ✅
- **Deduplication by class name limits inflation**

---

### The Real Problem

While object_count isn't massively inflated, **the reliability is compromised:**

**Candy factory video should have:**
- Expected objects: candy, hands, conveyor belt, machinery
- Expected object_count: ~2-4

**What YOLO detected:**
- Detected: skateboard, motorcycle, banana, hot dog, fire hydrant, apple, bowl, toothbrush, bottle, chair, sink, train, vase
- object_count = 14
- **Mix of YOLO misclassifications and legitimate objects**

**Key issue:** We cannot distinguish between:
1. Real objects that were tracked
2. Real objects that weren't tracked (but exist)
3. Misclassifications that got fallback IDs

---

### Impact on ML Training

**Feature quality degradation:**
- object_count used as feature in ML models
- Noisy values reduce training data quality
- Model may learn incorrect patterns
- Creative insights based on object presence are unreliable

**Examples of potentially bad insights:**
- "Videos with skateboards perform better" (actually candy)
- "Hot dogs appear in 40% of viral videos" (actually misclassified food products)
- object_count distribution may not reflect actual content

---

## Proposed Fix (Revised)

### Why the Original Fix Was Wrong

**Original proposal (Option A from initial version):**
```python
# Filter out all fallback IDs
if track_id and not track_id.startswith('obj_100'):
    unique_object_classes.add(obj.get('className'))
```

**Problem discovered:** This is too aggressive. When ByteTrack fails completely:
- Some legitimate objects still get real tracking (obj_5, obj_6, obj_12)
- Other legitimate objects get fallback IDs
- Filtering ALL fallback IDs removes real objects

**Example impact on video 7555692040944749855:**
- Has "dining table", "bed", "bottle" with good tracking
- Has "cake" with only fallback IDs (but might be legitimate, just misclassified)
- Filtering removes "cake" (which might be fine since it's candy)
- But in other videos, could remove legitimate objects

---

### Revised Fix: Field-Based Only (Conservative)

**Rationale:**
- Object misclassification is a YOLO limitation (COCO dataset doesn't have all objects)
- Filtering fallback IDs is too aggressive when ByteTrack fails completely
- The real bug is the missing `tracked` field propagation
- Better to return object_count = 0 when tracking fails than return noisy data

---

### Fix Location #1: Timeline Builder

**File:** `rumiai_v2/processors/timeline_builder.py`
**Line:** 118

**Current code:**
```python
entry = TimelineEntry(
    start=timestamp,
    end=None,
    entry_type='object',
    data={
        'class': obj_class,
        'confidence': frame_data.get('confidence', 0),
        'bbox': frame_data.get('bbox', []),
        'track_id': frame_data.get('trackId', frame_data.get('track_id', None))
        # ❌ MISSING
    }
)
```

**Fixed code:**
```python
entry = TimelineEntry(
    start=timestamp,
    end=None,
    entry_type='object',
    data={
        'class': obj_class,
        'confidence': frame_data.get('confidence', 0),
        'bbox': frame_data.get('bbox', []),
        'track_id': frame_data.get('trackId', frame_data.get('track_id', None)),
        'tracked': frame_data.get('tracked', True)  # ✅ ADD THIS
    }
)
```

---

### Fix Location #2: Object Timeline Extraction

**File:** `rumiai_v2/processors/temporal_compute.py`
**Lines:** 896-906

**Current code:**
```python
object_timeline = []
for entry in timeline_entries:
    if entry.get('entry_type') == 'object':
        object_timeline.append({
            'timestamp': entry.get('start', 0),
            'className': entry.get('data', {}).get('class', 'unknown'),
            'confidence': entry.get('data', {}).get('confidence', 0),
            'bbox': entry.get('data', {}).get('bbox', []),
            'trackId': entry.get('data', {}).get('track_id', None)
            # ❌ MISSING
        })
```

**Fixed code:**
```python
object_timeline = []
for entry in timeline_entries:
    if entry.get('entry_type') == 'object':
        object_timeline.append({
            'timestamp': entry.get('start', 0),
            'className': entry.get('data', {}).get('class', 'unknown'),
            'confidence': entry.get('data', {}).get('confidence', 0),
            'bbox': entry.get('data', {}).get('bbox', []),
            'trackId': entry.get('data', {}).get('track_id', None),
            'tracked': entry.get('data', {}).get('tracked', True)  # ✅ ADD THIS
        })
```

---

### Fix Location #3: Object Count Calculation

**File:** `rumiai_v2/processors/temporal_compute.py`
**Lines:** 2063-2071

**Current code:**
```python
unique_object_classes = set()

# Process objects first: count unique class names (not instances)
for obj in segment_objects:
    if obj.get('className') != 'person':
        timestamp = obj.get('timestamp', 0)
        if start <= timestamp < end:
            tracked = obj.get('tracked', True)  # ❌ Wrong default
            if tracked:
                unique_object_classes.add(obj.get('className'))
```

**Fixed code:**
```python
unique_object_classes = set()

# Process objects first: count unique class names (not instances)
for obj in segment_objects:
    if obj.get('className') != 'person':
        timestamp = obj.get('timestamp', 0)
        if start <= timestamp < end:
            tracked = obj.get('tracked', False)  # ✅ Change default to False

            # Only count objects that were successfully tracked
            if tracked:
                unique_object_classes.add(obj.get('className'))
```

---

### What This Fix Does

**When tracking works:**
- object_count reflects only tracked objects ✅
- Filters out transient noise and misclassifications
- Data quality is high

**When tracking fails completely:**
- object_count = 0 ✅
- Signals "no reliable object data available"
- This is HONEST about data quality
- Better than returning inflated/noisy counts

**Why this is correct:**
- If ByteTrack fails completely, we CANNOT trust object detections
- Misclassifications and noise dominate the data
- Returning 0 is more honest than returning 14 garbage classes

---

### Alternative: Confidence-Based Fallback (Phase 2)

**If object_count = 0 too often after Phase 1, consider:**

```python
unique_object_classes = set()

for obj in segment_objects:
    if obj.get('className') != 'person':
        timestamp = obj.get('timestamp', 0)
        if start <= timestamp < end:
            tracked = obj.get('tracked', False)
            confidence = obj.get('confidence', 0)

            # Keep if: tracked OR very high confidence (0.85+)
            if tracked or confidence >= 0.85:
                unique_object_classes.add(obj.get('className'))
```

**Rationale:**
- High-confidence detections (0.85+) are more likely real, even without tracking
- Allows some object data when tracking fails
- Threshold of 0.85 is conservative (vs 0.8 which might let too much noise through)

**Tradeoff:**
- More permissive than tracked-only approach
- Still allows some misclassifications through
- Should only be implemented if Phase 1 results in too many zeros

---

## Testing & Validation

### Test Dataset

**Video 1:** `7531126454034189599` (Hammond's Candies factory)

**Current Output:**
```json
{
  "hook": {
    "object_count": 14,
    "person_count": 124
  }
}
```

**Expected After Fix:**
- object_count: 0 (all fallback IDs, tracking failed)
- All "skateboard", "motorcycle", "banana", etc. filtered out
- This is CORRECT - signals unreliable data

---

**Video 2:** `7555692040944749855` (Candy taste test)

**Current Output:**
```json
{
  "hook": {
    "object_count": 7
  }
}
```

**Expected After Fix:**
- object_count: 3 (only tracked: dining table, bed, bottle)
- "cake" filtered out (45 fallback IDs)
- This is MORE ACCURATE - counts only tracked objects

---

**Video 3:** `7489503844997647646` (Celery juice - good tracking)

**Expected After Fix:**
- object_count: should remain similar (good tracking)
- Transient fallback IDs filtered, but shouldn't affect much
- No regression on working videos

---

### Validation Steps

#### 1. Analyze Tracking Quality

```python
# test_object_tracking.py

def analyze_object_tracking(video_id, unified_path):
    """Analyze object tracking quality."""
    with open(unified_path) as f:
        data = json.load(f)

    # Extract object detections in hook window
    object_entries = [
        e for e in data['timeline']['entries']
        if e.get('entry_type') == 'object'
        and e.get('data', {}).get('class') != 'person'
        and 0 <= e.get('start', 0) < 3
    ]

    # Group by class
    by_class = {}
    for entry in object_entries:
        obj_class = entry['data'].get('class')
        track_id = entry['data'].get('track_id')

        if obj_class not in by_class:
            by_class[obj_class] = {'total': 0, 'fallback': 0, 'real': 0}

        by_class[obj_class]['total'] += 1

        if track_id and track_id.startswith('obj_100'):
            by_class[obj_class]['fallback'] += 1
        else:
            by_class[obj_class]['real'] += 1

    print(f"\n{video_id}:")
    print(f"  Unique object classes: {len(by_class)}")

    for obj_class, counts in sorted(by_class.items(), key=lambda x: -x[1]['total']):
        print(f"    {obj_class}: {counts['total']} total, "
              f"{counts['real']} real tracking, {counts['fallback']} fallback")

    # Calculate what object_count would be after fix
    real_objects = [cls for cls, counts in by_class.items() if counts['real'] > 0]
    print(f"  Current object_count: {len(by_class)}")
    print(f"  After fix object_count: {len(real_objects)}")

    return by_class
```

---

#### 2. Test Fix Implementation

```bash
# Reprocess one video with the fix
python scripts/rumiai_runner.py "https://www.tiktok.com/@hammondscandies/video/7531126454034189599"

# Check results
cat insights/7531126454034189599_temporal_windows_updated.json | \
  jq '{object_count: .temporal_windows.hook.object_count, person_count: .temporal_windows.hook.person_count}'

# Expected:
# {
#   "object_count": 0,      # Was 14
#   "person_count": 0-2     # Was 124 (if PersonCountFix also applied)
# }
```

---

#### 3. Monitor object_count = 0 Frequency

After deploying, analyze how often object_count = 0:

```python
import json
from pathlib import Path

zero_count = 0
total_count = 0

for file in Path('insights').glob('*_temporal_windows_updated.json'):
    with open(file) as f:
        data = json.load(f)

    for window_name, window_data in data['temporal_windows'].items():
        if window_name != 'metadata':
            total_count += 1
            if window_data.get('object_count', 0) == 0:
                zero_count += 1

print(f"object_count = 0: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%)")

# If > 40%, consider Phase 2 confidence-based fallback
```

---

### Success Criteria

**Phase 1 fix is successful if:**

1. ✅ object_count for video 7531126454034189599 = 0 (currently 14)
2. ✅ object_count for video 7555692040944749855 = 3 (currently 7)
3. ✅ Only tracked objects counted
4. ✅ No regression on videos with good tracking
5. ✅ object_count more accurately reflects trackable objects
6. ⚠️ If object_count = 0 frequency > 40%, proceed to Phase 2

**Phase 2 (confidence fallback) may be needed if:**
- Too many videos have object_count = 0
- ML training suffers from missing object data
- Analysis shows high-confidence untracked detections are often correct

---

## Implementation Notes

### Coordination with PersonCountFix

**Both fixes modify the same file and nearby lines:**

**PersonCountFix:** Lines 2074-2102 (person_count calculation)
**ObjectFix:** Lines 2063-2071 (object_count calculation)

**Recommendation:** Implement both fixes together in a single PR.

---

### Implementation Order

#### Phase 1: Field-Based Fix (Conservative)

**Effort:** 2-3 hours
**Files Modified:** 2 files (timeline_builder.py, temporal_compute.py)

**Steps:**
1. Fix timeline_builder.py to preserve `tracked` field (line 118)
2. Fix temporal_compute.py to copy `tracked` field (line 904)
3. Fix temporal_compute.py to default `tracked` to False (line 2070)
4. Implement same fix for person_count (line 2082)
5. Test on 10-20 videos
6. Deploy

**Expected result:**
- Videos with good tracking: accurate object_count
- Videos with failed tracking: object_count = 0 (honest)

---

#### Phase 2: Confidence Fallback (If Needed)

**Trigger:** If Phase 1 results in object_count = 0 for > 40% of windows

**Effort:** 1 hour
**Files Modified:** 1 file (temporal_compute.py)

**Steps:**
1. Add confidence-based fallback (threshold 0.85)
2. Test on videos with tracking failures
3. Verify reduces object_count = 0 frequency
4. Monitor false positive rate (bad objects passing through)
5. Deploy if improvement outweighs cost

---

### Testing Recommendations

**Before deploying Phase 1:**

1. Test on videos with known tracking issues
2. Test on videos with good tracking (ensure no regression)
3. Manually verify object_count matches expectations for 20-30 videos
4. Check object_count = 0 frequency on test dataset

**After deploying Phase 1:**

1. Monitor object_count distribution across full dataset
2. Compare before/after distributions
3. Track object_count = 0 frequency
4. Decide if Phase 2 is needed
5. Validate ML feature quality improves

---

## Appendix

### A. Related Files

**Files to Modify (Phase 1):**
1. `rumiai_v2/processors/timeline_builder.py` (line 118)
2. `rumiai_v2/processors/temporal_compute.py` (lines 904, 2070)

**Files for Testing:**
1. `data/clients/test_production/hashtags/test_candy/top_contrastive/buckets/bucket_13-18s/analysis/unified/7531126454034189599.json`
2. `data/clients/test_production/hashtags/test_candy/top_contrastive/buckets/bucket_60-90s/analysis/unified/7555692040944749855.json`
3. `unified_analysis/7489503844997647646.json`

---

### B. Related Documentation

- **PersonCountFix.md** - Same root cause, fix person_count metric
- **VisionServices.md** - YOLO service and tracking implementation
- **SystemArchitecturev2.md** - Overall pipeline architecture
- **YOLO_ByteTrack_CodeComparison.md** - Investigation of tracking code history

---

### C. Commands Reference

**Check object classes in video:**
```bash
cat unified_analysis/VIDEO_ID.json | \
  jq '[.timeline.entries[] | select(.entry_type == "object" and .start < 3)] |
      group_by(.data.class) |
      map({class: .[0].data.class, count: length}) |
      sort_by(-.count)'
```

**Check tracking quality by object class:**
```bash
cat unified_analysis/VIDEO_ID.json | \
  jq '[.timeline.entries[] | select(.entry_type == "object" and .data.class == "skateboard" and .start < 3)] |
      group_by(.data.track_id) |
      map({track: .[0].data.track_id, count: length})'
```

**Check object_count in temporal windows:**
```bash
cat insights/VIDEO_ID_temporal_windows_updated.json | \
  jq '.temporal_windows | to_entries[] | {window: .key, object_count: .value.object_count}'
```

**Count fallback vs real tracking IDs:**
```bash
cat unified_analysis/VIDEO_ID.json | \
  jq '[.timeline.entries[] | select(.entry_type == "object" and .data.class != "person")] |
      group_by(.data.track_id | startswith("obj_100")) |
      map({is_fallback: .[0].data.track_id | startswith("obj_100"), count: length})'
```

---

### D. Expected Results After Phase 1 Fix

**Video 7531126454034189599 (Candy Factory):**

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| object_count | 14 | 0 | 100% - honest about unreliable data |
| person_count | 124 | 1-2 | 98% reduction |
| Data Quality | Garbage | Honest signal | ✅ |

**Video 7555692040944749855 (Candy taste test):**

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| object_count | 7 | 3 | Counts only tracked objects |
| Objects kept | All 7 classes | dining table, bed, bottle | ✅ More accurate |

**Overall Dataset Impact:**

- Videos with good tracking: accurate object_count
- Videos with failed tracking: object_count = 0 (honest)
- Improved reliability for ML training
- May need Phase 2 if too many zeros

---

**Document Version:** 2.0 (Revised after reconsidering fallback ID filtering)
**Last Updated:** 2025-10-21
**Author:** Claude (Sonnet 4.5)
**Reviewed By:** [Pending]
