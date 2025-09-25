# Timestamp Bugs Fix Documentation

## Overview
This document outlines all timestamp-related bugs found in the RumiAI pipeline and provides specific fixes for each issue.

---

## 🔴 CRITICAL BUG 1: 6-Second Video State Inconsistency

### The Problem
Videos exactly 6 seconds long create an inconsistent state where closing window exists but middle segments are empty.

### Current Code
```python
# temporal_compute.py Line 496-500
elif video_duration <= (HOOK_WINDOW_DURATION + CLOSING_WINDOW_DURATION):  # <= 6
    return {
        'hook': (0, HOOK_WINDOW_DURATION),
        'middle': None,
        'closing': (HOOK_WINDOW_DURATION, video_duration)
    }

# Line 516-517
def calculate_middle_segments(video_duration: float):
    if video_duration <= 6:
        return {}  # Returns empty dict, not None!
```

### The Fix
```python
# Option 1: Make 6-second videos have NO middle (consistent with window calculation)
def calculate_middle_segments(video_duration: float):
    if video_duration <= 6:
        return None  # Return None instead of {}

# Option 2: Change boundary to exclude exactly 6 seconds
elif video_duration < (HOOK_WINDOW_DURATION + CLOSING_WINDOW_DURATION):  # < 6 instead of <= 6
    return {
        'hook': (0, HOOK_WINDOW_DURATION),
        'middle': None,
        'closing': (HOOK_WINDOW_DURATION, video_duration)
    }
```

### Recommendation
Use **Option 1** - simpler and maintains consistency. Videos ≤6s should have no middle segments.

---

## 🔴 CRITICAL BUG 2: Eye Contact Double-Counting at Boundaries

### The Problem
Eye contact events at exact boundaries (3.0s, video end) are counted in both adjacent windows.

### Current Code
```python
# temporal_compute.py Lines 1183, 1215
if start <= entry_start <= end:  # WRONG: inclusive upper bound
    eye_contact = entry.get('data', {}).get('eye_contact', 0)
```

### The Fix
```python
# Change to exclusive upper bound to match all other filtering
if start <= entry_start < end:  # Fixed: exclusive upper bound
    eye_contact = entry.get('data', {}).get('eye_contact', 0)
```

### Impact
- Events at 3.0s will only count in middle segment (not hook)
- Events at video end will only count in closing (not double-counted)

---

## 🟡 MEDIUM BUG 3: Frame Number Truncation Loses Precision

### The Problem
Converting timestamps to frame numbers using `int()` truncates decimal frames, losing temporal precision.

### Current Code
```python
# temporal_compute.py Lines 453, 1493
start_frame = int(start * fps)  # Truncates: 3.03s * 30fps = 90.9 → 90
end_frame = int(end * fps)
```

### The Fix
```python
# Use round() instead of int() for nearest frame
start_frame = round(start * fps)  # 3.03s * 30fps = 90.9 → 91
end_frame = round(end * fps)

# Or for more precision, use floor/ceil appropriately
import math
start_frame = math.floor(start * fps)  # Inclusive start
end_frame = math.ceil(end * fps)       # Exclusive end
```

### Recommendation
Use `round()` for both - simpler and preserves temporal accuracy better than truncation.

---

## 🟡 MEDIUM BUG 4: OCR Text Duration Hardcoded to Minimum 1 Second

### The Problem
Short text like "Hi" gets artificial 1-second duration, potentially spanning window boundaries incorrectly.

### Current Code
```python
# timeline_builder.py Line 197
duration = max(1.0, len(text) * 0.1)  # Forces minimum 1 second
```

### The Fix
```python
# Option 1: Remove minimum, allow natural short durations
duration = max(0.1, len(text) * 0.1)  # Minimum 0.1s instead of 1.0s

# Option 2: Make duration proportional but reasonable
duration = min(3.0, max(0.3, len(text) * 0.08))  # 0.3-3.0 second range

# Option 3: Use constant duration for all OCR (simplest)
duration = 0.5  # Fixed 0.5 second duration for all text
```

### Recommendation
Use **Option 2** - provides reasonable range without artificial boundaries spanning.

---

## 🟡 MEDIUM BUG 5: Segment Timeline Uses Different Boundary Logic

### The Problem
Segment timeline uses inclusive upper bound while all other timelines use exclusive.

### Current Code
```python
# temporal_compute.py Line 596
if seg_start <= timestamp <= seg_end:  # Both boundaries inclusive
```

### The Fix
```python
# Make consistent with other timelines
if seg_start <= timestamp < seg_end:  # Exclusive upper bound
```

### Note
Verify this doesn't break existing segment logic before applying.

---

## 🟡 MEDIUM BUG 6: Float Precision in Window Boundaries

### The Problem
Floating point arithmetic can create boundaries like 41.0000001 that don't match events at 41.0.

### Current Code
```python
# temporal_compute.py Line 506
'closing': (video_duration - CLOSING_WINDOW_DURATION, video_duration)
# If video_duration = 44.0000001, start = 41.0000001
```

### The Fix
```python
# Round boundaries to reasonable precision (3 decimal places = millisecond)
def round_boundary(value: float) -> float:
    return round(value, 3)

'closing': (
    round_boundary(video_duration - CLOSING_WINDOW_DURATION),
    round_boundary(video_duration)
)
```

### Alternative Fix
```python
# Use epsilon comparison in filtering
EPSILON = 1e-6
if start <= timestamp < end + EPSILON:  # Allow tiny differences
```

### Recommendation
Use boundary rounding - cleaner and more predictable.

---

## 🟢 LOW BUG 7: No FPS Validation

### The Problem
Code assumes fps is valid, could divide by zero or negative.

### Current Code
```python
# video_analyzer.py Line 734
timestamps.append(frame_count / fps)  # No validation
```

### The Fix
```python
# Validate FPS before use
DEFAULT_FPS = 30.0  # TikTok standard

if fps <= 0 or fps > 1000:  # Sanity check
    logger.warning(f"Invalid FPS {fps}, using default {DEFAULT_FPS}")
    fps = DEFAULT_FPS

timestamps.append(frame_count / fps)
```

---

## 🟢 LOW BUG 8: Edge Case for Videos < 3 Seconds

### The Problem
Not really a bug, but unusual behavior where entire video becomes hook window.

### Current Code
```python
# temporal_compute.py Line 490-495
if video_duration <= HOOK_WINDOW_DURATION:
    return {'hook': (0, video_duration), 'middle': None, 'closing': None}
```

### The Fix
No fix needed - this is correct behavior. Just document it clearly.

```python
# Add clear documentation
if video_duration <= HOOK_WINDOW_DURATION:
    # For very short videos (<3s), entire video is treated as hook
    # This is intentional - hook is most important for short content
    return {'hook': (0, video_duration), 'middle': None, 'closing': None}
```

---

## Implementation Priority

### Immediate (Fix Now)
1. **Bug 2**: Eye contact double-counting - Data corruption
2. **Bug 1**: 6-second video crash - Pipeline failures

### Soon (Next Sprint)
3. **Bug 3**: Frame truncation - Precision loss
4. **Bug 5**: Segment boundary inconsistency - Consistency

### When Convenient
5. **Bug 4**: OCR duration - Minor impact
6. **Bug 6**: Float precision - Rare edge case
7. **Bug 7**: FPS validation - Defensive programming
8. **Bug 8**: Documentation only - No code change

---

## Testing After Fixes

### Create Test Videos
1. **6-second video**: Test Bug 1 fix
2. **Video with events at 3.0s, 41.0s**: Test Bug 2 boundary fix
3. **Video with OCR text**: Test Bug 4 duration fix
4. **Corrupted metadata video**: Test Bug 7 validation

### Automated Tests
```python
def test_boundary_filtering():
    # Test that events at boundaries aren't double-counted
    assert event_at_3_seconds in middle_segment
    assert event_at_3_seconds not in hook

def test_six_second_video():
    # Test that 6-second video processes without crash
    windows = calculate_temporal_windows(6.0)
    assert windows['middle'] is None or windows['middle'] == []

def test_frame_precision():
    # Test that frame conversion maintains precision
    timestamp = 3.03
    frame = round(timestamp * 30)
    assert frame == 91  # Not 90
```

---

---

## 🔵 ALIGNMENT: Production vs BucketsPlan.md Discrepancies

### Overview
The production code does NOT match BucketsPlan.md specifications. This section outlines the changes needed to align them.

### Current Discrepancies

| Duration | BucketsPlan.md Spec | Current Production | Issue |
|----------|-------------------|-------------------|--------|
| 0-3s | Hook only | Hook only | ✅ Correct |
| 3-9s | Hook + Closing only | 3-6s: No middle<br>6-9s: HAS middle | ❌ Wrong boundary |
| 9-18s | Hook + 3 Middle + Closing | Based on middle duration, not video duration | ❌ Wrong logic |
| 18-33s | Hook + 4 Middle + Closing | Boundaries don't align | ❌ Wrong boundaries |

### Root Cause
Production uses **middle segment duration** for decisions, while BucketsPlan uses **total video duration**.

### Changes Required for Alignment

#### Change 1: Fix 3-9 Second Boundary
```python
# CURRENT (temporal_compute.py line 496)
elif video_duration <= (HOOK_WINDOW_DURATION + CLOSING_WINDOW_DURATION):  # 6 seconds
    return {
        'hook': (0, HOOK_WINDOW_DURATION),
        'middle': None,
        'closing': (HOOK_WINDOW_DURATION, video_duration)
    }

# CHANGE TO:
elif video_duration <= 9:  # Match BucketsPlan.md
    return {
        'hook': (0, min(3, video_duration)),
        'middle': None,
        'closing': (min(3, video_duration), video_duration)
    }
```

#### Change 2: Update Middle Segment Calculation
```python
# CURRENT (temporal_compute.py line 516)
def calculate_middle_segments(video_duration: float):
    if video_duration <= 6:
        return {}

# CHANGE TO:
def calculate_middle_segments(video_duration: float):
    if video_duration <= 9:  # No middle for 3-9s videos per BucketsPlan
        return None
```

#### Change 3: Redefine Segment Count Thresholds
```python
# CURRENT (lines 31-36)
SEGMENT_THRESHOLDS = {
    'min_duration_for_segments': 3,     # Based on middle duration
    'three_segments_max': 12,           # Based on middle duration
    'four_segments_max': 27,            # Based on middle duration
}

# CHANGE TO (based on total video duration per BucketsPlan):
BUCKET_THRESHOLDS = {
    'no_middle_max': 9,           # 0-9s: No middle segments
    'three_segments_max': 18,     # 9-18s: 3 middle segments
    'four_segments_max': 33,      # 18-33s: 4 middle segments
    'five_segments_max': 75,      # 33-75s: 5 middle segments
    # >75s: 5 segments (capped)
}
```

#### Change 4: Rewrite Segment Count Logic
```python
# CURRENT (lines 532-537) - uses middle_duration
if middle_duration <= SEGMENT_THRESHOLDS['three_segments_max']:
    num_segments = 3
elif middle_duration <= SEGMENT_THRESHOLDS['four_segments_max']:
    num_segments = 4
else:
    num_segments = 5

# CHANGE TO (use video_duration directly):
def calculate_middle_segments(video_duration: float):
    # No middle for short videos
    if video_duration <= BUCKET_THRESHOLDS['no_middle_max']:
        return None

    # Determine segment count based on TOTAL video duration
    if video_duration <= BUCKET_THRESHOLDS['three_segments_max']:
        num_segments = 3  # 9-18s videos
    elif video_duration <= BUCKET_THRESHOLDS['four_segments_max']:
        num_segments = 4  # 18-33s videos
    elif video_duration <= BUCKET_THRESHOLDS['five_segments_max']:
        num_segments = 5  # 33-75s videos
    else:
        num_segments = 5  # Cap at 5 for very long videos

    # Calculate segment boundaries
    middle_start = HOOK_WINDOW_DURATION
    middle_end = video_duration - CLOSING_WINDOW_DURATION
    middle_duration = middle_end - middle_start

    # Safety check
    if middle_duration <= 0:
        return None

    segment_duration = middle_duration / num_segments
    segments = {}

    for i in range(num_segments):
        segment_start = middle_start + (i * segment_duration)
        segment_end = segment_start + segment_duration
        segments[f'segment_{i+1}'] = {
            'start': segment_start,
            'end': segment_end
        }

    return segments
```

### Impact Analysis

#### Videos Affected by Changes
| Duration | Current Output | New Output | Impact |
|----------|---------------|------------|--------|
| 6-9s | 3 segments | No segments | **Major** - loses middle analysis |
| 12-18s | 3 segments | 3 segments | None - same count |
| 27-33s | 4 segments | 4 segments | None - same count |

#### Backward Compatibility Issues
1. **Existing processed videos**: 6-9s videos will have different structure
2. **ML models**: Trained on current buckets, will need retraining
3. **Downstream consumers**: Code expecting middle segments for 6-9s videos will break

### Migration Strategy: OPTION 1 - HARD CUT-OVER (SELECTED)

We have decided to implement **Option 1: Hard Cut-Over** for a clean, consistent system.

#### Implementation Steps

##### Step 1: Pre-Deployment Testing
1. **Create comprehensive test suite** with videos at all boundary conditions
2. **Run tests on development environment** with new logic
3. **Validate output structure** matches BucketsPlan.md exactly
4. **Document all changes** in release notes

##### Step 2: Production Deployment
1. **Schedule maintenance window** (recommend low-traffic period)
2. **Backup current code** and configuration
3. **Deploy all changes atomically**:
   - Update temporal window calculation (9s boundary)
   - Update segment calculation (video duration based)
   - Update threshold constants
4. **Run smoke tests** immediately after deployment

##### Step 3: Data Reprocessing
1. **Identify affected videos** (all 6-9s videos minimum)
2. **Create reprocessing script**:
```python
# reprocess_for_bucketplan.py
import glob
import json

def needs_reprocessing(video_path):
    """Check if video needs reprocessing based on duration"""
    with open(video_path, 'r') as f:
        data = json.load(f)
    duration = data.get('duration', 0)
    # All 6-9s videos definitely need reprocessing
    # Consider reprocessing all for consistency
    return 6 <= duration <= 9 or REPROCESS_ALL

def reprocess_video(video_id):
    """Re-run temporal compute for video"""
    # Load unified_analysis
    # Run new compute_temporal_windows
    # Save updated output
    pass
```
3. **Execute reprocessing** in batches to avoid overload
4. **Validate reprocessed outputs**

##### Step 4: ML Model Updates
1. **Retrain models** with new bucket structure
2. **Update model configs** to expect new bucket boundaries
3. **Validate model performance** before production use

#### Risk Mitigation

Despite choosing the hard cut-over, we'll implement these safety measures:

1. **Rollback Plan**:
   - Keep previous version tagged and ready
   - Document rollback procedure
   - Test rollback in staging

2. **Monitoring**:
   - Alert on processing failures
   - Track bucket distribution changes
   - Monitor performance metrics

3. **Validation Checklist**:
   - [ ] All tests pass with new logic
   - [ ] 6-9s videos have no middle segments
   - [ ] 9-18s videos have exactly 3 middle segments
   - [ ] No crashes at boundary conditions
   - [ ] Performance acceptable (< 10% degradation)

### Testing Requirements for Hard Cut-Over

#### Priority Test Videos
Create these videos BEFORE deployment:
- **6.0s exactly**: Currently has middle, should have none after fix
- **9.0s exactly**: Critical boundary - should have no middle
- **9.1s**: Should have 3 middle segments
- **18.0s exactly**: Should have 3 middle segments
- **18.1s**: Should have 4 middle segments

#### Comprehensive Test Suite
Full boundary testing:
- 2.9s, 3.0s, 3.1s (bucket 1-2 boundary)
- 5.9s, 6.0s, 6.1s (current problem area)
- 8.9s, 9.0s, 9.1s (bucket 2-3 boundary)
- 17.9s, 18.0s, 18.1s (bucket 3-4 boundary)
- 32.9s, 33.0s, 33.1s (bucket 4-5 boundary)

#### Validation Script
```python
def validate_bucketplan_alignment(video_id, duration):
    """Verify video follows BucketsPlan.md exactly"""
    output = load_temporal_output(video_id)

    if duration <= 3:
        assert output['temporal_windows']['middle_segments'] is None
        assert output['temporal_windows']['closing'] is None
    elif duration <= 9:
        assert output['temporal_windows']['middle_segments'] is None
        assert output['temporal_windows']['closing'] is not None
    elif duration <= 18:
        assert len(output['temporal_windows']['middle_segments']) == 3
    elif duration <= 33:
        assert len(output['temporal_windows']['middle_segments']) == 4
    else:
        assert len(output['temporal_windows']['middle_segments']) == 5
```

---

## Summary

### Phase 1: Bug Fixes (Original 8)
- **2 CRITICAL** bugs causing crashes and data corruption
- **4 MEDIUM** bugs causing precision loss and inconsistencies
- **2 LOW** priority improvements for edge cases
- **Timeline**: 2-3 hours
- **Risk**: Low - mostly one-line changes

### Phase 2: BucketsPlan Alignment (Hard Cut-Over)
- **Major structural changes** to match BucketsPlan.md exactly
- **Affects ALL videos**, especially 6-9 second range
- **Requires complete reprocessing** of existing data
- **Timeline**:
  - Development & Testing: 4-6 hours
  - Deployment & Reprocessing: 2-4 hours
  - Total: 6-10 hours
- **Risk**: High - but mitigated with thorough testing

### Deployment Order

1. **Fix critical bugs first** (Bug 1 & 2) - Immediate
2. **Test alignment changes** - Next day
3. **Deploy alignment changes** - After validation
4. **Reprocess all videos** - Post-deployment
5. **Fix remaining bugs** - When convenient

### Success Criteria

After Option 1 implementation:
- ✅ No 6-second video crashes
- ✅ No double-counting at boundaries
- ✅ Perfect alignment with BucketsPlan.md
- ✅ All existing videos reprocessed
- ✅ Clean, consistent codebase