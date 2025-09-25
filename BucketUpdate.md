# BucketPlan Alignment Update

## Overview
This document outlines the changes needed to align production code with BucketsPlan.md specifications, including risks and implementation strategy.

## Current Discrepancies

| Duration | BucketsPlan.md Spec | Current Production | Issue |
|----------|-------------------|-------------------|--------|
| 0-3s | Hook only | Hook only | ✅ Correct |
| 3-9s | Hook + Closing only | 3-6s: No middle<br>6-9s: HAS middle | ❌ Wrong boundary |
| 9-18s | Hook + 3 Middle + Closing | Based on middle duration, not video duration | ❌ Wrong logic |
| 18-33s | Hook + 4 Middle + Closing | Boundaries don't align | ❌ Wrong boundaries |

## Root Cause
Production uses **middle segment duration** for decisions, while BucketsPlan uses **total video duration**.

## ⚠️ RISK ASSESSMENT - COMPREHENSIVE ANALYSIS

### Previously Underestimated Risks

After exhaustive dependency analysis, we've identified critical risks that were initially overlooked:

#### 1. Confirmed Code Dependencies
**Test Files Expecting Current Structure:**
- `test_full_integration.py`: Iterates through `middle_segments` assuming they exist
- `test_temporal_compute_v2.py`: Has assertions about temporal window structure
- `test_rumiai_with_temporal.py`: Tests expect middle segments for certain durations
- `analyze_text_distribution.py`: Aggregates middle segment metrics, will break on None
- `test_average_face_size.py`: Accesses temporal windows directly

**Production Code Dependencies:**
- `cold_start_performance_test.py`: Looks for `temporal_windows_updated.json`
- Multiple analysis scripts access `temporal_windows.get('middle_segments', [])`
- These will need defensive coding for None middle_segments

#### 2. Mathematical Complexity
The middle duration calculation is used throughout the codebase:
```python
middle_duration = video_duration - 6  # This assumption is everywhere
segment_duration = middle_duration / num_segments  # Affects all segment sizes
```

Changing this affects:
- Segment size calculations across all videos
- Frame extraction alignment
- Potential division by zero edge cases
- Feature density within segments

#### 3. Data Gap for 6-9s Videos
- Current: 6-9s videos have 0-3s of middle content analyzed
- After change: This content would be UNANALYZED
- **Impact**: Loss of data coverage for these videos

#### 4. No Rollback Path
- Once videos are reprocessed with new boundaries, old data is gone
- No way to compare or validate without parallel processing
- Historical data becomes inconsistent with new data

#### 5. Performance Implications
- Fewer segments = less granular temporal data
- Could impact ML model performance
- No way to predict impact without testing

#### 6. Specific Breaking Changes Found
**Files That Will Break:**
1. **analyze_text_distribution.py (Lines 50-54)**:
   ```python
   if middle_segments:  # Will be None for 6-9s videos
       overlay_counts = [seg.get('overlay_unique_count', 0) for seg in middle_segments]
   ```
   - Currently expects empty list, not None
   - Will throw TypeError on None

2. **test_full_integration.py (Line 27)**:
   ```python
   for i, segment in enumerate(result['temporal_windows']['middle_segments'], 1):
   ```
   - Will crash on None middle_segments
   - No None check before iteration

3. **All Downstream Analysis Scripts**:
   - Expect `middle_segments` to be list (empty or populated)
   - None value will break list operations

### Risk Level: **HIGH** (Upgraded from MEDIUM-HIGH)

## Changes Required for Alignment

### Change 1: Fix 3-9 Second Boundary
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

### Change 2: Update Middle Segment Calculation
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

### Change 3: Redefine Segment Count Thresholds
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

### Change 4: Rewrite Segment Count Logic
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

## Impact Analysis

### Videos Affected by Changes
| Duration | Current Output | New Output | Impact |
|----------|---------------|------------|--------|
| 6-9s | 3 segments | No segments | **Major** - loses middle analysis |
| 12-18s | 3 segments | 3 segments | None - same count |
| 27-33s | 4 segments | 4 segments | None - same count |

### Backward Compatibility Issues
1. **Existing processed videos**: 6-9s videos will have different structure
2. **ML models**: Trained on current buckets, will need retraining
3. **Downstream consumers**: Code expecting middle segments for 6-9s videos will break
4. **Unknown systems**: We don't know all dependencies on current structure

## Migration Strategy: OPTION 1 - HARD CUT-OVER (SELECTED)

We have decided to implement **Option 1: Hard Cut-Over** for a clean, consistent system.

### ⚠️ Pre-Implementation Requirements (CRITICAL)

Given the HIGH risk level and confirmed breaking changes, BEFORE implementing:

1. **Fix All Downstream Consumers**
   ```python
   # Update all code to handle None middle_segments:
   middle_segments = temporal_windows.get('middle_segments')
   if middle_segments is not None:  # Explicit None check
       # Process segments
   ```

   **Files requiring updates:**
   - analyze_text_distribution.py
   - test_full_integration.py
   - test_temporal_compute_v2.py
   - test_rumiai_with_temporal.py
   - test_average_face_size.py
   - cold_start_performance_test.py

2. **Impact Testing**
   - Process 10-20 videos in each duration range
   - Compare outputs between current and new logic
   - Measure data loss for 6-9s videos

3. **Parallel Validation**
   - Run BOTH versions for a subset of videos
   - Compare ML model performance on both structures
   - Validate no critical features are lost

### Implementation Steps

#### Step 1: Pre-Deployment Testing
1. **Create comprehensive test suite** with videos at all boundary conditions
2. **Run tests on development environment** with new logic
3. **Validate output structure** matches BucketsPlan.md exactly
4. **Document all changes** in release notes
5. **Run parallel comparison** to understand impact

#### Step 2: Production Deployment
1. **Schedule maintenance window** (recommend low-traffic period)
2. **Backup current code** and configuration
3. **Deploy all changes atomically**:
   - Update temporal window calculation (9s boundary)
   - Update segment calculation (video duration based)
   - Update threshold constants
4. **Run smoke tests** immediately after deployment

#### Step 3: Data Reprocessing
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

#### Step 4: ML Model Updates
1. **Retrain models** with new bucket structure
2. **Update model configs** to expect new bucket boundaries
3. **Validate model performance** before production use

### Risk Mitigation

Despite choosing the hard cut-over, we'll implement these safety measures:

1. **Rollback Plan**:
   - Keep previous version tagged and ready
   - Document rollback procedure
   - Test rollback in staging

2. **Monitoring**:
   - Alert on processing failures
   - Track bucket distribution changes
   - Monitor performance metrics
   - **NEW**: Track ML model performance degradation
   - **NEW**: Monitor for unexpected null/empty segments

3. **Validation Checklist**:
   - [ ] All tests pass with new logic
   - [ ] 6-9s videos have no middle segments
   - [ ] 9-18s videos have exactly 3 middle segments
   - [ ] No crashes at boundary conditions
   - [ ] Performance acceptable (< 10% degradation)
   - [ ] **NEW**: ML models show acceptable performance
   - [ ] **NEW**: No downstream systems break
   - [ ] **NEW**: Data coverage analysis complete

## Testing Requirements for Hard Cut-Over

### Priority Test Videos
Create these videos BEFORE deployment:
- **6.0s exactly**: Currently has middle, should have none after fix
- **9.0s exactly**: Critical boundary - should have no middle
- **9.1s**: Should have 3 middle segments
- **18.0s exactly**: Should have 3 middle segments
- **18.1s**: Should have 4 middle segments

### Comprehensive Test Suite
Full boundary testing:
- 2.9s, 3.0s, 3.1s (bucket 1-2 boundary)
- 5.9s, 6.0s, 6.1s (current problem area)
- 8.9s, 9.0s, 9.1s (bucket 2-3 boundary)
- 17.9s, 18.0s, 18.1s (bucket 3-4 boundary)
- 32.9s, 33.0s, 33.1s (bucket 4-5 boundary)

### Validation Script
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

## Summary

### BucketsPlan Alignment
- **Major structural changes** to match BucketsPlan.md exactly
- **Affects ALL videos**, especially 6-9 second range
- **Requires complete reprocessing** of existing data
- **Risk Level**: **HIGH** (upgraded after comprehensive dependency analysis)
- **Breaking Changes Confirmed**: 6+ files will crash without updates

### Timeline
- **Pre-implementation fixes**: 4-6 hours (update all downstream consumers)
- **Pre-implementation audit**: 2-3 days
- **Development & Testing**: 4-6 hours
- **Parallel validation**: 1-2 days
- **Deployment & Reprocessing**: 2-4 hours
- **Total**: 5-7 days (significantly increased from initial estimate)

### Success Criteria

After Option 1 implementation:
- ✅ Perfect alignment with BucketsPlan.md
- ✅ All existing videos reprocessed
- ✅ Clean, consistent codebase
- ✅ No degradation in ML model performance
- ✅ No broken downstream systems
- ✅ Clear documentation of changes and impacts