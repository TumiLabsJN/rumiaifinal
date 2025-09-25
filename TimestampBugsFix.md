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

## Summary

### Bug Fixes to Implement
- **2 CRITICAL** bugs causing crashes and data corruption
- **4 MEDIUM** bugs causing precision loss and inconsistencies
- **2 LOW** priority improvements for edge cases
- **Timeline**: 2-3 hours total
- **Risk**: Low - mostly one-line changes

### Deployment Order

1. **Fix critical bugs** (Bug 1 & 2) - Immediate
   - Eye contact double-counting (line 1183, 1215)
   - 6-second video state issue (line 516)
2. **Fix medium bugs** (Bug 3-6) - Next sprint
   - Frame truncation
   - OCR duration
   - Segment boundary consistency
   - Float precision
3. **Fix low priority items** (Bug 7-8) - When convenient
   - FPS validation
   - Documentation for <3s videos

### Success Criteria

After bug fixes:
- ✅ No 6-second video crashes
- ✅ No double-counting at boundaries
- ✅ Improved precision in frame calculations
- ✅ Consistent boundary logic throughout
- ✅ Better error handling for edge cases

### Note on BucketsPlan Alignment
BucketsPlan alignment has been moved to a separate document (BucketUpdate.md) due to its complexity and higher risk profile. These bug fixes can and should be implemented independently of bucket alignment decisions.