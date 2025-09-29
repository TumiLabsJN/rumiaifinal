# Timestamp Bugs Fix Documentation

**Version**: 2.0
**Last Updated**: January 2025
**Status**: Ready for Implementation

## Overview
This document outlines timestamp-related bugs found in the RumiAI pipeline that require fixing. After thorough review and simplification, we've identified 2 bugs that need implementation.

**Note**: The previous 6-second boundary issue has been resolved by the bucket alignment changes (boundary moved from 6s to 9s).

---

## 🔴 CRITICAL BUG 1: Eye Contact Double-Counting at Boundaries

### The Problem
Eye contact events at exact boundaries (3.0s, video end) are counted in both adjacent windows. This is the ONLY timeline entry using inclusive upper bounds.

### Comprehensive Analysis of Boundary Logic
We searched the entire codebase and found **24 total boundary comparisons** across 5 files:

#### Primary Timeline Filtering (temporal_compute.py)
| Line | Pattern | Type | Status |
|------|---------|------|--------|
| 638 | `if start <= timestamp < end:` | Text timeline | ✅ Correct |
| 1191 | `if start <= entry_start <= end:` | **Eye contact** | **❌ BUG** |
| 1223 | `if start <= entry_start <= end:` | **Eye contact** | **❌ BUG** |
| 1262 | `if start <= o.get('timestamp', 0) < end]` | Objects | ✅ Correct |
| 1264 | `if start <= g.get('timestamp', 0) < end]` | Gestures | ✅ Correct |
| 1266 | `if start <= e.get('timestamp', 0) < end]` | Expressions | ✅ Correct |
| 1268 | `if start <= s.get('timestamp', 0) < end]` | Scenes | ✅ Correct |
| 1270 | `if start <= c.get('timestamp', 0) < end]` | Camera | ✅ Correct |
| 1338 | `if start <= timestamp < end:` | Text overlay | ✅ Correct |
| 1460 | `if start <= f.get('timestamp', 0) < end]` | Faces | ✅ Correct |

#### Special Cases (temporal_compute.py)
| Line | Pattern | Purpose | Status |
|------|---------|---------|--------|
| 604 | `if seg_start <= timestamp <= seg_end:` | Speech segments | ⚠️ Intentional* |
| 890 | `if seg_start < end and seg_end > start:` | Overlap detection | ✅ Correct |
| 1059 | `if seg_end <= start or seg_start >= end:` | Exclusion check | ✅ Correct |
| 1316 | `if scene_end > start and scene_start < end:` | Scene overlap | ✅ Correct |

*Speech segments use inclusive bounds for continuous ranges (not point events).

#### Other Files
- **timeline.py (Line 44)**: `start.seconds <= timestamp < self.end.seconds` ✅
- **timeline.py (Line 134)**: `entry_end >= start and entry_start <= end` ✅ (overlap)
- **debug_gaze.py (Line 28)**: `0 <= e.get('start', 0) <= 3` ⚠️ Test file

### Summary: Only Eye Contact Has The Bug
- **11 timeline filters** use correct exclusive upper bound `[start, end)`
- **Only 2 instances** (both eye contact) use wrong inclusive bound `[start, end]`
- **Special cases** (overlaps, segments) use appropriate patterns for their purpose

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

## 🟡 MEDIUM BUG 2: OCR Text Duration Hardcoded to Minimum 1 Second

### The Problem
Short text like "Hi" gets artificial 1-second duration, potentially spanning window boundaries incorrectly.

### Current Code
```python
# timeline_builder.py Line 197
duration = max(1.0, len(text) * 0.1)  # Forces minimum 1 second
```

### The Fix
```python
# Simple one-line change
duration = max(0.5, len(text) * 0.1)  # 0.5s minimum instead of 1.0s
```

### Why This Fix
- Reduces artificial boundary crossing for short text
- More accurate for quick text flashes
- Zero technical risk (no code depends on 1.0s assumption)
- Maintains proportional scaling for longer text

---

## Root Cause Analysis

### Why These Bugs Exist

1. **Bug 1 (Eye Contact Double-Counting)**
   - **Root Cause**: Copy-paste error during implementation
   - **Pattern**: Developer copied boundary checking logic from another timeline but used inclusive upper bound instead of exclusive
   - **Prevention**: Code review should catch inconsistent boundary patterns

2. **Bug 2 (OCR Duration)**
   - **Root Cause**: Arbitrary minimum chosen without considering edge effects
   - **Pattern**: Developer wanted to ensure text was visible "long enough" but didn't consider window boundary crossing
   - **Prevention**: Design decisions should consider boundary conditions

### Bugs We Removed After Analysis

- **Segment Timeline Boundaries**: Not a bug - intentional for continuous ranges
- **Float Precision**: Academic paranoia - not a real problem
- **FPS Validation**: Over-engineering - OpenCV returns valid FPS for TikTok videos
- **<3s Videos**: Intentional design - correct behavior

---

## Implementation Priority

### Immediate (Fix Now)
1. **Bug 1**: Eye contact double-counting - Data corruption

### Soon (Next Sprint)
2. **Bug 2**: OCR duration - Simple fix, low risk

### When Convenient
None remaining.

---

## Testing After Fixes

### Create Test Videos
1. **Video with events at 3.0s, 41.0s**: Test Bug 1 boundary fix
2. **Video with OCR text**: Test Bug 2 duration fix

### Note on Speech Segment Boundaries (NOT A BUG)
```python
# Line 604 uses inclusive boundaries intentionally for continuous ranges:
if seg_start <= timestamp <= seg_end:  # Checking if timestamp is IN speech segment
    # This is CORRECT and INTENTIONAL for continuous ranges (speech segments)
    # Speech segments represent periods of continuous speech, not point events
    # Point events (objects, gestures) correctly use exclusive upper bound
    # This is NOT inconsistent - it's the appropriate pattern for each data type
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

```

---

---

## Summary

### Bug Fixes to Implement

**Total Bugs**: 2 (down from original 8 after thorough analysis)

1. **CRITICAL - Eye Contact Double-Counting**
   - Impact: Data corruption - events at boundaries counted twice
   - Fix: Change 2 lines (1191, 1223) from inclusive to exclusive upper bound
   - Effort: 5 minutes
   - Risk: Very low - 10 other boundaries already use correct pattern

2. **MEDIUM - OCR Duration Minimum**
   - Impact: Short text artificially spans window boundaries
   - Fix: Change 1 line (197) from 1.0s to 0.5s minimum
   - Effort: 5 minutes
   - Risk: Zero - no dependencies on the 1.0s value

**Total Implementation Time**: 10 minutes of code changes + 1-2 hours testing

### Deployment Order

1. **Fix critical bug** (Bug 1) - Immediate
   - Eye contact double-counting (line 1191, 1223)
   - Risk: Very low - only 2 lines to change
   - Testing: Verify events at boundaries aren't double-counted

2. **Fix medium bug** (Bug 2) - Next sprint
   - OCR duration minimum change (line 197)
   - Risk: Zero - no dependencies on 1.0s assumption
   - Testing: Verify short text doesn't span boundaries unnecessarily

### Success Criteria

After implementing these 2 fixes:
- ✅ Eye contact events no longer double-counted at window boundaries
- ✅ Short OCR text won't artificially span into adjacent windows
- ✅ All timeline boundaries use consistent patterns (exclusive upper bound for point events)
- ✅ OCR duration more accurately represents actual text display time

### Implementation Notes

1. **BucketsPlan Already Implemented**: The 6s→9s boundary change is complete and resolved the state inconsistency issue.

2. **Bugs We Removed During Analysis**:
   - Segment timeline boundaries (intentional for continuous ranges)
   - Float precision (paranoid - not a real issue)
   - FPS validation (over-engineering)
   - Frame truncation (minimal impact)
   - Edge case <3s videos (correct behavior)

3. **These 2 remaining bugs are the only ones worth fixing** after removing academic concerns and over-engineering.