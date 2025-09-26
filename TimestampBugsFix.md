# Timestamp Bugs Fix Documentation

## Overview
This document outlines all timestamp-related bugs found in the RumiAI pipeline and provides specific fixes for each issue.

**Note**: The previous 6-second boundary issue has been resolved by the bucket alignment changes (boundary moved from 6s to 9s). This document focuses on remaining active bugs.

---

## 🔴 CRITICAL BUG 1: Eye Contact Double-Counting at Boundaries

### The Problem
Eye contact events at exact boundaries (3.0s, video end) are counted in both adjacent windows. This is the ONLY timeline entry using inclusive upper bounds.

### Comprehensive Analysis of Boundary Logic
We searched the entire codebase for boundary comparisons and found:

| Location | Pattern | Type | Status |
|----------|---------|------|--------|
| Line 638 | `if start <= timestamp < end:` | Text timeline | ✅ Correct |
| Line 1191 | `if start <= entry_start <= end:` | Eye contact | ❌ BUG |
| Line 1223 | `if start <= entry_start <= end:` | Eye contact | ❌ BUG |
| Line 1262 | `if start <= o.get('timestamp', 0) < end]` | Objects | ✅ Correct |
| Line 1264 | `if start <= g.get('timestamp', 0) < end]` | Gestures | ✅ Correct |
| Line 1266 | `if start <= e.get('timestamp', 0) < end]` | Expressions | ✅ Correct |
| Line 1268 | `if start <= s.get('timestamp', 0) < end]` | Scenes | ✅ Correct |
| Line 1270 | `if start <= c.get('timestamp', 0) < end]` | Camera | ✅ Correct |
| Line 1338 | `if start <= timestamp < end:` | Text (2nd) | ✅ Correct |
| Line 1460 | `if start <= f.get('timestamp', 0) < end]` | Faces | ✅ Correct |
| Line 604 | `if seg_start <= timestamp <= seg_end:` | Speech segments | ⚠️ Different* |

*Speech segments use inclusive bounds intentionally - they represent continuous ranges, not point events.

### Current Code
```python
# temporal_compute.py Lines 1191, 1223 (same bug in two places)
if start <= entry_start <= end:  # WRONG: inclusive upper bound
    eye_contact = entry.get('data', {}).get('eye_contact', 0)
```

### The Fix
```python
# Change to exclusive upper bound to match all other timeline filtering
if start <= entry_start < end:  # Fixed: exclusive upper bound
    eye_contact = entry.get('data', {}).get('eye_contact', 0)
```

### Why This Is The Only Fix Needed
- **10 out of 12** boundary checks already use exclusive upper bound correctly
- **Only eye contact** (2 instances) uses inclusive, causing double-counting
- **Speech segments** (line 604) use inclusive intentionally for continuous ranges

### Impact
- Events at 3.0s will only count in middle segment (not hook)
- Events at video end will only count in closing (not double-counted)
- Aligns with Python convention: `[start, end)` (inclusive start, exclusive end)

---

## 🟡 MEDIUM BUG 2: Frame Number Truncation Loses Precision

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

## 🟡 MEDIUM BUG 3: OCR Text Duration Hardcoded to Minimum 1 Second

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

## 🟡 MEDIUM BUG 4: Segment Timeline Uses Different Boundary Logic

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

## 🟡 MEDIUM BUG 5: Float Precision in Window Boundaries

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

## 🟢 LOW BUG 6: No FPS Validation

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

## 🟢 LOW BUG 7: Edge Case for Videos < 3 Seconds

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
1. **Bug 1**: Eye contact double-counting - Data corruption

### Soon (Next Sprint)
2. **Bug 2**: Frame truncation - Precision loss
3. **Bug 4**: Segment boundary inconsistency - Consistency

### When Convenient
4. **Bug 3**: OCR duration - Minor impact
5. **Bug 5**: Float precision - Rare edge case
6. **Bug 6**: FPS validation - Defensive programming
7. **Bug 7**: Documentation only - No code change

---

## Testing After Fixes

### Create Test Videos
1. **Video with events at 3.0s, 41.0s**: Test Bug 1 boundary fix
2. **Video with OCR text**: Test Bug 3 duration fix
3. **Corrupted metadata video**: Test Bug 6 validation

### Note on Speech Segment Boundaries
```python
# Line 604 uses inclusive boundaries intentionally:
if seg_start <= timestamp <= seg_end:  # Checking if timestamp is IN speech segment
    # This is correct for continuous ranges (speech segments)
    # Different from point events (detections, gestures, etc.)
```

### Automated Tests
```python
def test_boundary_filtering():
    # Test that events at boundaries aren't double-counted
    assert event_at_3_seconds in middle_segment
    assert event_at_3_seconds not in hook

    # Test that all timeline entries use consistent boundaries
    for timeline_type in ['object', 'gesture', 'expression', 'face', 'gaze']:
        events = filter_by_window(timeline_type, start=3.0, end=6.0)
        # Event at 3.0 should be included (inclusive start)
        assert any(e.timestamp == 3.0 for e in events)
        # Event at 6.0 should NOT be included (exclusive end)
        assert not any(e.timestamp == 6.0 for e in events)

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
- **1 CRITICAL** bug causing data corruption (eye contact double-counting)
  - Only 2 lines need changing (1191, 1223)
  - 10 other boundary checks already correct
- **4 MEDIUM** bugs causing precision loss and inconsistencies
- **2 LOW** priority improvements for edge cases
- **Timeline**: 2-3 hours total
- **Risk**: Very Low - only 2 lines to change for critical bug

### Deployment Order

1. **Fix critical bug** (Bug 1) - Immediate
   - Eye contact double-counting (line 1191, 1223)
2. **Fix medium bugs** (Bug 2-5) - Next sprint
   - Frame truncation
   - OCR duration
   - Segment boundary consistency
   - Float precision
3. **Fix low priority items** (Bug 6-7) - When convenient
   - FPS validation
   - Documentation for <3s videos

### Success Criteria

After bug fixes:
- ✅ No double-counting at boundaries
- ✅ Improved precision in frame calculations
- ✅ Consistent boundary logic throughout
- ✅ Better error handling for edge cases

### Note on BucketsPlan Alignment
BucketsPlan alignment has been implemented, moving the boundary from 6s to 9s. This resolved the previous 6-second video state inconsistency issue. The remaining bugs in this document can be implemented independently.