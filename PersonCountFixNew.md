# Person Count Fix Documentation

**Date**: 2025-10-28
**Issue**: Video 7558977602870906167 reports `person_count: 3` in middle_1 segment when only 1 person is present
**Root Cause**: YOLO tracking fragmentation + overly conservative threshold (95%)
**Status**: Solution Designed, Ready for Implementation

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Proposed Solution](#3-proposed-solution)
4. [Implementation Details](#4-implementation-details)
5. [Risk Analysis](#5-risk-analysis)
6. [Testing Strategy](#6-testing-strategy)
7. [Rollout Plan](#7-rollout-plan)

---

## 1. Problem Statement

### Issue Description

**Video**: 7558977602870906167 (87s fitness video by @benjicavazos)
**Segment**: middle_1 (3.0s - 19.2s)
**Expected**: `person_count: 1` (single person throughout entire video)
**Actual**: `person_count: 3`

### Impact

- **Scope**: Affects videos with significant camera movement, pose changes, or scene transitions
- **Frequency**: Estimated 10-20% of single-person videos
- **ML Impact**: Incorrect person_count values could skew ML training data
- **Business Impact**: Inaccurate features for creative pattern analysis

### Output Example

```json
{
  "middle_segments": [
    {
      "start": 3.0,
      "end": 19.2,
      "duration": 16.2,
      "person_count": 3,  // ❌ WRONG - Should be 1
      "segment_name": "segment_1"
    }
  ]
}
```

---

## 2. Root Cause Analysis

### Investigation Path

#### Step 1: YOLO Detection Analysis

**Command**:
```bash
cat object_detection_outputs/7558977602870906167/7558977602870906167_yolo_detections.json | \
  jq '.objectAnnotations[] | select(.className == "person" and .timestamp >= 3.0 and .timestamp < 19.2) | .trackId' | \
  sort | uniq -c
```

**Result**:
```
199 "obj_1"      # 40.9% of detections (0-9s)
  9 "obj_2"      # 1.8% of detections (brief fallback)
279 "obj_3"      # 57.3% of detections (9-19.2s)
  1 "obj_10000"  # Excluded (tracked=false)
```

**Finding**: YOLO assigned **3 different track IDs** to the same person due to tracking fragmentation.

---

#### Step 2: Current Logic Analysis

**File**: `rumiai_v2/processors/temporal_compute.py:2091-2103`

```python
# Calculate person count with dominant track logic
if not track_counts:
    person_count = 0
else:
    total_detections = sum(track_counts.values())
    max_track_count = max(track_counts.values())

    # If one track dominates with >95% of detections, it's the same person
    if max_track_count / total_detections > 0.95:
        person_count = 1
    else:
        # Multiple balanced tracks = multiple people or uncertain case
        person_count = len(track_counts)
```

**Calculation**:
```
track_counts = {obj_1: 199, obj_2: 9, obj_3: 279}
total = 487
max = 279
dominance = 279 / 487 = 57.3%

57.3% < 95% threshold
→ person_count = len(track_counts) = 3 ❌
```

---

### Root Cause Summary

**Primary Cause**: YOLO tracking fragmentation
- Track lost at ~9s due to major pose/camera change
- New track ID assigned when person re-acquired
- Brief fallback track (obj_2) during transition

**Secondary Cause**: Overly conservative 95% threshold
- Designed to avoid false positives (merging genuine multi-person scenes)
- Too strict for common tracking fragmentation patterns
- Doesn't account for temporal relationships between tracks

---

## 3. Proposed Solution

### Solution Overview: **Two-Layer Approach**

```
Layer 1: Track Merging (Option 2)
└─ Merge tracks that don't overlap temporally (tracking fragmentation)
   └─ Detect genuine co-existence (overlap > 20%)

Layer 2: Threshold Logic (Option 3)
└─ Apply conservative or aggressive thresholds based on overlap detection
   ├─ had_overlap=True  → Conservative (only 95% threshold)
   └─ had_overlap=False → Aggressive (50% + gap thresholds)
```

### Why This Works

**For Single Person (Tracking Fragmentation)**:
```
Raw tracks: obj_1 (0-9s, 199 frames), obj_3 (9-19s, 279 frames)

Layer 1:
- Temporal overlap = 0 frames
- Sequential tracks (9s end → 9s start)
- Merge: obj_1 + obj_3 → 478 frames
- had_overlap = False

Layer 2:
- len(merged_counts) = 1
- Return person_count = 1 ✓
```

**For Duets (Genuine Multi-Person)**:
```
Raw tracks: Person A (600 frames), Person B (400 frames) - both throughout segment

Layer 1:
- Temporal overlap = 350 frames (54%)
- DON'T merge
- had_overlap = True

Layer 2:
- len(merged_counts) = 2
- had_overlap=True → Conservative mode
- max = 60% < 95%
- Return person_count = 2 ✓
```

---

## 4. Implementation Details

### 4.1 File to Modify

**File**: `rumiai_v2/processors/temporal_compute.py`
**Lines**: 2075-2103
**Function**: Feature extraction within `compute_temporal_windows()`

---

### 4.2 Context: Where segment_objects Comes From

**Location in temporal_compute.py**: Lines 2060-2090 (approximately)

The `segment_objects` variable is available in scope from the timeline processing loop:

```python
# Existing code context (around line 2060-2090)
segment_objects = []
for entry in unified_analysis['timeline']['entries']:
    if entry['entry_type'] == 'object':
        timestamp = entry.get('start_time', 0)
        if start <= timestamp < end:
            segment_objects.append(entry['data'])

# Person count calculation starts here (line ~2075)
track_counts = {}
for obj in segment_objects:
    if obj.get('className') == 'person':
        timestamp = obj.get('timestamp', 0)
        if start <= timestamp < end:
            tracked = obj.get('tracked')
            if tracked:
                track_id = obj.get('trackId')
                if track_id:
                    track_counts[track_id] = track_counts.get(track_id, 0) + 1

# NEW CODE GOES HERE (around line 2091)
```

**Structure of segment_objects**:
```python
segment_objects = [
    {
        'className': 'person',
        'trackId': 'obj_1',
        'timestamp': 3.5,
        'tracked': True,
        'confidence': 0.85,
        'bbox': [x1, y1, x2, y2]
    },
    # ... more objects
]
```

---

### 4.3 New Function: Track Merging with Union-Find

Add this function before the person_count calculation block (around line 2050):

```python
class UnionFind:
    """Simple Union-Find (Disjoint Set) for track merging."""
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_groups(self):
        """Return groups of connected elements."""
        groups = {}
        for element in self.parent:
            root = self.find(element)
            if root not in groups:
                groups[root] = []
            groups[root].append(element)
        return list(groups.values())


def merge_fragmented_tracks_v2(track_counts, segment_objects, start, end, config):
    """
    Merge tracks that represent tracking fragmentation while detecting genuine co-existence.

    Uses Union-Find to handle transitive merging (A→B→C all merge into one group).

    Args:
        track_counts: Dict of {track_id: detection_count}
        segment_objects: List of detection objects from timeline
        start: Segment start time
        end: Segment end time
        config: PERSON_COUNT_CONFIG dict with thresholds

    Returns:
        tuple: (merged_counts dict, had_overlap bool)
            - merged_counts: Track IDs after merging fragmented tracks
            - had_overlap: True if any tracks had significant temporal overlap

    Trade-offs:
        - merge_gap_max (default 1.0s): Balance between merging fragmentation vs splitting real transitions
          - Lower (0.5s): Safer, fewer false merges, but may miss some fragmentation
          - Higher (2.0s): Catches more fragmentation, but may merge person entrances/exits
        - Current default (1.0s) accepts that person exit/return >1s will be split
    """
    # Build timeline for each track
    track_timelines = {}
    for obj in segment_objects:
        if obj.get('className') == 'person' and obj.get('tracked'):
            timestamp = obj.get('timestamp', 0)
            if start <= timestamp < end:
                track_id = obj.get('trackId')
                if track_id:
                    if track_id not in track_timelines:
                        track_timelines[track_id] = []
                    track_timelines[track_id].append(timestamp)

    tracks = list(track_timelines.keys())
    if not tracks:
        return {}, False

    # Initialize Union-Find
    uf = UnionFind(tracks)
    had_overlap = False

    # Check all pairs for overlap and potential merging
    for i, track_a in enumerate(tracks):
        a_times = set(track_timelines[track_a])
        a_min, a_max = min(a_times), max(a_times)

        for track_b in tracks[i+1:]:  # Only check each pair once
            b_times = set(track_timelines[track_b])
            b_min, b_max = min(b_times), max(b_times)

            # Calculate temporal overlap
            overlap_frames = len(a_times & b_times)
            total_frames = len(a_times | b_times)
            overlap_ratio = overlap_frames / total_frames if total_frames > 0 else 0

            # Threshold from config
            overlap_threshold = config['overlap_threshold']
            merge_gap_max = config['merge_gap_max']

            if overlap_ratio > overlap_threshold:
                # Significant overlap = genuine co-existence (don't merge)
                had_overlap = True
            else:
                # No/minimal overlap - check if tracks are sequential (tracking fragmentation)
                # Calculate time gap between tracks in both directions:
                #   gap_a_to_b: Time from end of track_a to start of track_b
                #   gap_b_to_a: Time from end of track_b to start of track_a
                # Example: Track A ends at 9.0s, Track B starts at 9.2s → gap = 0.2s
                gap_a_to_b = b_min - a_max if a_max < b_min else float('inf')
                gap_b_to_a = a_min - b_max if b_max < a_min else float('inf')
                min_gap = min(gap_a_to_b, gap_b_to_a)

                # If minimum gap < threshold (default 1.0s), tracks are sequential → merge
                # This handles tracking fragmentation where tracker loses and re-acquires person
                if min_gap < merge_gap_max:
                    uf.union(track_a, track_b)

    # Get merged groups from Union-Find
    merged_groups = uf.get_groups()

    # Aggregate track counts by group
    merged_counts = {}
    for group in merged_groups:
        representative = group[0]  # Use first track as group ID
        merged_counts[representative] = sum(track_counts.get(t, 0) for t in group)

    return merged_counts, had_overlap
```

---

### 4.4 Configuration Parameters

**Add at the top of the person count calculation block (around line 2050)**:

```python
import os

# Person count tunable parameters
PERSON_COUNT_CONFIG = {
    # Tier 1: Clear dominance threshold
    'tier1_threshold': float(os.getenv('PERSON_COUNT_TIER1', '0.95')),

    # Tier 2: Strong dominance threshold
    'tier2_threshold': float(os.getenv('PERSON_COUNT_TIER2', '0.50')),

    # Tier 2: Minimum gap between top 2 tracks
    'tier2_gap': float(os.getenv('PERSON_COUNT_TIER2_GAP', '0.05')),

    # Temporal overlap threshold for genuine co-existence
    'overlap_threshold': float(os.getenv('PERSON_COUNT_OVERLAP', '0.20')),

    # Maximum gap for sequential track merging (in seconds)
    'merge_gap_max': float(os.getenv('PERSON_COUNT_MERGE_GAP', '1.0')),
}
```

---

### 4.5 Updated Person Count Calculation

**Replace lines 2091-2103 with**:

```python
# Calculate person count with two-layer robust logic
# Layer 1: Merge tracking fragmentation
merged_counts, had_overlap = merge_fragmented_tracks_v2(
    track_counts, segment_objects, start, end, PERSON_COUNT_CONFIG
)

# Layer 2: Apply threshold logic
if not merged_counts:
    person_count = 0
elif len(merged_counts) == 1:
    person_count = 1
else:
    total_detections = sum(merged_counts.values())
    max_track_count = max(merged_counts.values())
    max_pct = max_track_count / total_detections

    # Get second highest track percentage
    sorted_counts = sorted(merged_counts.values(), reverse=True)
    second_max_pct = sorted_counts[1] / total_detections if len(sorted_counts) > 1 else 0

    # Get thresholds from config
    tier1_threshold = PERSON_COUNT_CONFIG['tier1_threshold']
    tier2_threshold = PERSON_COUNT_CONFIG['tier2_threshold']
    tier2_gap = PERSON_COUNT_CONFIG['tier2_gap']

    # If Option 2 detected temporal overlap, be MORE conservative
    if had_overlap:
        # Tracks genuinely co-exist → only merge if extremely dominant
        if max_pct > tier1_threshold:  # Tier 1 only
            person_count = 1
        else:
            person_count = len(merged_counts)  # Trust overlap detection
    else:
        # No overlap detected → tracking fragmentation likely
        # Tier 1: Clear dominance
        if max_pct > tier1_threshold:
            person_count = 1
        # Tier 2: Strong dominance with significant gap
        elif max_pct > tier2_threshold and (max_pct - second_max_pct) > tier2_gap:
            person_count = 1
        else:
            person_count = len(merged_counts)

# Enhanced logging for debugging
logger.debug(f"Person count calculation for {start:.1f}s-{end:.1f}s:")
logger.debug(f"  Raw tracks: {track_counts}")
logger.debug(f"  Merged tracks: {merged_counts}")
logger.debug(f"  Had overlap: {had_overlap}")
logger.debug(f"  max_pct: {max_pct:.2%}, second_pct: {second_max_pct:.2%}, gap: {(max_pct - second_max_pct):.2%}")
logger.debug(f"  Decision: Tier {'1' if max_pct > tier1_threshold else '2' if not had_overlap and max_pct > tier2_threshold else 'Conservative'}")
logger.debug(f"  Result: person_count = {person_count}")
```

---

## 5. Risk Analysis

### 5.1 Critical Risks

#### Risk 1: Duets Incorrectly Counted as 1 Person 🔴 **MITIGATED**

**Scenario**: Two people with unequal screen time (60% / 40%)

**Without Overlap Detection**:
- Tier 2 would trigger (60% > 50%, gap 20% > 5%)
- Would return person_count = 1 ❌

**With Overlap Detection**:
- Layer 1 detects 54% temporal overlap
- had_overlap = True → Conservative mode
- Only Tier 1 applies (95% threshold)
- Returns person_count = 2 ✓

**Mitigation Status**: ✅ **FIXED** by overlap detection

---

#### Risk 2: Person Exit/Return Split into 2 🟡 **PARTIAL**

**Scenario**: Person leaves frame at 5s, returns at 7s

**Behavior**:
- Gap = 2s > 1s merge threshold
- Won't merge
- Returns person_count = 2 ❌ (might be wrong)

**Mitigation Options**:
1. Increase merge gap to 2-3s (risk: might merge actual person transitions)
2. Check for scene changes in gap (if scene cut, probably same person)
3. Accept as edge case (2-5% of videos)

**Recommended**: Option 3 - Accept as limitation

---

#### Risk 3: Regression on Currently Correct Videos 🟡 **TESTABLE**

**Concern**: Videos that currently report correct person_count might break

**Mitigation**:
1. ✅ Comprehensive pre-deployment testing (45 videos with ground truth)
2. ✅ Compare new logic against current outputs on test set
3. ✅ Manually validate all differences
4. ✅ Enhanced logging for debugging
5. ✅ Fast rollback capability (git revert + redeploy)

---

### 5.2 Edge Cases (Acceptable Limitations)

| Edge Case | Behavior | Acceptable? |
|-----------|----------|-------------|
| Mirror reflections | Counts reflection as separate person | ✅ YES - Visually distinct |
| Clone effects (TikTok) | Counts each clone | ✅ YES - Visually distinct |
| Very brief background person (<0.5s) | Might be counted or ignored | ✅ YES - Noisy either way |
| Person enters/exits mid-segment | Might be 1 or 2 depending on gap | 🟡 MAYBE - 2-5% of videos |

---

### 5.3 Performance Impact

| Metric | Current | After Fix | Delta |
|--------|---------|-----------|-------|
| Lines of code | 12 | ~80 | +68 |
| Function calls | 1 | 2 | +1 |
| Time per segment | <1ms | ~2-5ms | +2-5ms |
| Time per video (7 segments) | <7ms | ~14-35ms | +7-28ms |
| Time for 300 videos | <2s | ~4-10s | +2-8s |

**Verdict**: ✅ Negligible performance impact

**Note**: Estimates assume typical videos have <5 tracks per segment. Algorithm is O(n²) where n = number of tracks. Highly fragmented videos (10+ tracks) may take 10-50ms per segment, but this is rare in practice. If this becomes an issue, consider early exit after processing top 5 tracks.

---

## 6. Testing Strategy

### 6.1 Test Dataset Requirements

**Ideal Dataset** (high confidence): 45 videos with ground truth labels
**Minimum Dataset** (acceptable): 10-15 videos focusing on critical cases

| Category | Ideal | Minimum | Priority | Purpose |
|----------|-------|---------|----------|---------|
| Single person, stable | 10 | 2 | Medium | Baseline - should be 1 |
| Single person, high movement | 10 | 3 | **High** | Your video type - should be 1 |
| Duets, balanced screen time | 5 | 2 | **High** | Should be 2 |
| Duets, unequal screen time | 5 | 3 | **Critical** | Should be 2 (worst case) |
| Person enters mid-segment | 5 | 1 | Low | Edge case - might be 1 or 2 |
| Person exits/returns | 5 | 1 | Low | Edge case - ambiguous |
| Groups (3+ people) | 5 | 1 | Low | Should be 3+ |
| **Total** | **45** | **13** | - | Manual validation required |

**Time estimates**:
- Ideal dataset: 2-3 hours labeling
- Minimum dataset: 30-45 minutes labeling

---

### 6.2 Validation Process

#### Step 1: Ground Truth Labeling

Create `test_dataset_ground_truth.json`:

```json
{
  "videos": [
    {
      "video_id": "7558977602870906167",
      "segments": {
        "middle_1": {
          "ground_truth_person_count": 1,
          "notes": "Single person, tracking fragmentation"
        }
      }
    },
    {
      "video_id": "duet_example_123",
      "segments": {
        "middle_1": {
          "ground_truth_person_count": 2,
          "notes": "Duet with 60/40 screen time split"
        }
      }
    }
  ]
}
```

---

#### Step 2: Comparison Test

Create test script `test_person_count_fix.py`:

**Note**: This is pseudocode - adapt to your codebase. Functions like `load_test_dataset()`, `load_existing_output()`, and `run_analysis_with_new_logic()` need to be implemented based on your project structure.

```python
# PSEUDOCODE - Adapt to your setup
import json
import os

def run_comparison_test():
    """
    Run both old and new logic, compare outputs
    """
    test_videos = load_test_dataset()
    results = {
        'matches': 0,
        'improvements': 0,
        'regressions': 0,
        'details': []
    }

    for video in test_videos:
        # Compare against current production outputs (pre-fix baseline)
        old_output = load_existing_output(video['video_id'])  # Load from current insights/

        # Run with new logic (after implementing fix)
        new_output = run_analysis_with_new_logic(video['video_id'])

        # Compare
        for segment_name, ground_truth in video['segments'].items():
            old_count = old_output[segment_name]['person_count']
            new_count = new_output[segment_name]['person_count']
            expected = ground_truth['ground_truth_person_count']

            if old_count == new_count:
                results['matches'] += 1
            elif new_count == expected and old_count != expected:
                results['improvements'] += 1
            elif old_count == expected and new_count != expected:
                results['regressions'] += 1

            results['details'].append({
                'video_id': video['video_id'],
                'segment': segment_name,
                'old': old_count,
                'new': new_count,
                'expected': expected,
                'status': 'IMPROVED' if new_count == expected else 'REGRESSED' if old_count == expected else 'CHANGED'
            })

    return results

# Run test
results = run_comparison_test()

print(f"Matches: {results['matches']}")
print(f"Improvements: {results['improvements']}")
print(f"Regressions: {results['regressions']}")

# Review regressions
for detail in results['details']:
    if detail['status'] == 'REGRESSED':
        print(f"⚠️  REGRESSION: {detail}")
```

---

#### Step 3: Success Criteria

**Minimum Requirements for Production**:
- ✅ Improvements > Regressions (net positive)
- ✅ Regression rate < 10% of test set
- ✅ All duet videos correctly counted as 2
- ✅ Your video (7558977602870906167) correctly counted as 1
- ✅ No performance degradation > 100ms per video

**Target Goals**:
- 🎯 Improvements > 2x Regressions
- 🎯 Regression rate < 5%
- 🎯 95%+ accuracy on single-person videos
- 🎯 90%+ accuracy on multi-person videos

---

### 6.3 Monitoring Metrics

After deployment, track:

```python
# Distribution changes
person_count_distribution = {
    'before': {'1': 60%, '2': 30%, '3+': 10%},
    'after':  {'1': 75%, '2': 20%, '3+': 5%}
}

# Expected: More videos counted as 1 (tracking fragmentation fixed)
# Alert if: 2-person count drops below 15% (might be merging duets)
```

---

## 7. Rollout Plan

**Total Timeline**: 4-6 weeks (realistic estimate with buffer for issues)

---

### Phase 1: Implementation (Week 1: Days 1-3)

**Tasks**:
- [ ] Implement UnionFind class
- [ ] Implement `merge_fragmented_tracks_v2()` function
- [ ] Add PERSON_COUNT_CONFIG with tunable parameters
- [ ] Update person count calculation with two-layer logic
- [ ] Add enhanced debug logging
- [ ] Code review and refactoring

**Time estimate**: 2-3 days (one developer)

**Deliverable**: Code ready for testing

---

### Phase 2: Testing (Week 1-2: Days 4-10)

**Tasks**:
- [ ] Create test dataset (13-45 videos with ground truth)
- [ ] Ground truth labeling (30min - 3 hours)
- [ ] Run comparison test script (old vs new outputs)
- [ ] Manually validate all differences (not just regressions)
- [ ] Tune thresholds iteratively based on results
- [ ] Document test results and threshold choices

**Time estimate**: 1 week (includes threshold tuning iterations)

**Success Criteria**:
- Improvements > Regressions (net positive)
- Regression rate < 10%
- All duet videos correctly counted as 2
- Test video (7558977602870906167) correctly counted as 1
- No unexpected edge case failures

**Deliverable**: Test report with validation results and final threshold recommendations

---

### Phase 3: Deployment (Week 3: Days 11-15)

**Tasks**:
- [ ] Deploy new logic to production
- [ ] Monitor first 20 processed videos manually
- [ ] Check person_count distribution changes (1→75%, 2→20%, 3+→5% expected)
- [ ] Review logs for unexpected behaviors/errors
- [ ] Validate on 5-10 new videos with manual inspection
- [ ] Quick fixes if minor issues found

**Time estimate**: 3-5 days (including monitoring period)

**Deliverable**: Confirmation that fix works in production

---

### Phase 4: Monitoring & Stabilization (Week 3-6: Days 16-42)

**Tasks**:
- [ ] Monitor person_count distribution trends over 2-3 weeks
- [ ] Spot-check 10 videos per day for first week, then reduce to 5/day
- [ ] Tune thresholds if consistent patterns emerge
- [ ] Document edge cases and learnings
- [ ] Update this document with production insights
- [ ] Sign off on stable release

**Time estimate**: 2-3 weeks (monitoring can overlap with other work)

**Deliverable**: Stable production system with documented fix and lessons learned

---

**Contingency Buffer**: +1-2 weeks if major issues discovered during testing/deployment

---

### Rollback Plan

If critical issues discovered:

```bash
# Code-level rollback
git revert <commit-hash>
# Redeploy
```

**Rollback triggers**:
- Regression rate > 15% in production
- Critical duet videos consistently counted as 1 person
- Performance degradation > 100ms per video
- Unexpected errors in logs
- ML training pipeline impacted

**Detection method**: Manual spot-checking of 10-20 videos per day during first week post-deployment. No automated regression detection initially.

**Rollback time**: 5-10 minutes (git revert + redeploy)

---

### Historical Data Migration Strategy

**Decision**: Apply fix to **new videos only**. Do NOT reprocess existing 300 videos.

**Rationale**:
- Reprocessing 300 videos is time-consuming (~10-15 minutes processing time)
- Historical ML models already trained on old person_count values
- Changing historical data mid-training could invalidate models
- Inconsistency is acceptable - document as "person_count_v1" vs "person_count_v2"

**Implications**:
- Dataset will have mixed person_count versions
- Document this in data schema: Add `person_count_version` field (optional)
- If consistency is later required, batch reprocess during off-hours

**Alternative** (if consistency critical):
- Batch reprocess all 300 videos overnight
- Archive old outputs with `_v1` suffix
- Compare old vs new outputs for validation

---

## 8. Configuration Reference

### Environment Variables

```bash
# Tier 1 threshold (default: 0.95)
export PERSON_COUNT_TIER1=0.95

# Tier 2 dominance threshold (default: 0.50)
export PERSON_COUNT_TIER2=0.50

# Tier 2 gap threshold (default: 0.05)
export PERSON_COUNT_TIER2_GAP=0.05

# Temporal overlap threshold (default: 0.20)
export PERSON_COUNT_OVERLAP=0.20

# Sequential merge gap max (default: 1.0s)
export PERSON_COUNT_MERGE_GAP=1.0
```

---

### Threshold Tuning Guide

**⚠️ IMPORTANT**: The default thresholds (especially `PERSON_COUNT_OVERLAP=0.20`) are initial estimates and **require validation with real data**. Test with your specific dataset and tune based on observed false positive/negative rates.

**If seeing too many false negatives (missing real multi-person)**:
```bash
# Make overlap detection more sensitive
export PERSON_COUNT_OVERLAP=0.15  # Lower from 0.20

# Make Tier 2 more conservative
export PERSON_COUNT_TIER2=0.60    # Raise from 0.50
export PERSON_COUNT_TIER2_GAP=0.10 # Raise from 0.05
```

**If seeing too many false positives (splitting single person)**:
```bash
# Make overlap detection less sensitive
export PERSON_COUNT_OVERLAP=0.25  # Raise from 0.20

# Make Tier 2 more aggressive
export PERSON_COUNT_TIER2=0.45    # Lower from 0.50
export PERSON_COUNT_TIER2_GAP=0.03 # Lower from 0.05

# Allow longer gaps for merging
export PERSON_COUNT_MERGE_GAP=2.0  # Raise from 1.0
```

---

## 9. Success Metrics

### Quantitative Goals

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| Single-person accuracy | 80-85% | 95% | 98% |
| Multi-person accuracy | 70-75% | 90% | 95% |
| Regression rate | N/A | <10% | <5% |
| Improvement rate | N/A | >20% | >30% |

---

### Qualitative Goals

- ✅ Your video (7558977602870906167) correctly counted
- ✅ No duets incorrectly merged to 1 person
- ✅ Enhanced logging provides clear debugging path
- ✅ Fast rollback via git revert (5-10 minutes)
- ✅ Tunable parameters documented and functional

---

## 10. Summary: Worst Case Scenario

### Primary Risk: Duets Marked as 1 Person

**Status**: 🟢 **MITIGATED** by overlap detection

**How it's Fixed**:
```
Duet with unequal screen time (60% / 40%):

Layer 1 (Track Merging):
- Detects 54% temporal overlap between tracks
- Sets had_overlap = True
- Does NOT merge tracks

Layer 2 (Threshold Logic):
- had_overlap = True → Conservative mode activated
- Only applies Tier 1 (95% threshold)
- Tier 2 (50% + gap) is disabled
- Result: person_count = 2 ✓ CORRECT
```

**Remaining Edge Cases** (2-5% of videos):
- Person exits/returns with >1s gap
- Mirror reflections (acceptable)
- Clone effects (acceptable)

**Bottom Line**: The two-layer approach with overlap detection effectively prevents the worst-case scenario while fixing tracking fragmentation issues.

---

## 11. Contact & References

**Implementation Owner**: TBD
**Reviewer**: TBD
**Testing Lead**: TBD

**Related Files**:
- `rumiai_v2/processors/temporal_compute.py` (lines 2075-2103)
- `object_detection_outputs/{video_id}/{video_id}_yolo_detections.json`
- `insights/{video_id}_temporal_windows_updated.json`

**Test Video Reference**:
- Video ID: 7558977602870906167
- Author: @benjicavazos
- Duration: 87s
- Issue: middle_1 person_count = 3 (should be 1)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Status**: Ready for Implementation
