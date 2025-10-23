# Stage 6 ML Analysis Generation - Test Results

**Date**: 2025-10-23
**Tested by**: Claude CLI
**Status**: ❌ FAILED - Code bugs discovered in Stage 6 implementation

---

## Executive Summary

Stage 6 testing revealed **2 critical bugs** in the `ml_analysis_generation.py` script that prevent successful execution. All 3 test buckets failed during JSON generation. The bugs are:

1. **Bug #1**: Boolean/object column handling in quantile computation
2. **Bug #2**: Undefined variable in conditional code path

---

## Test Execution Results

### Bucket-by-Bucket Summary

| Bucket | Status | Exit Code | Duration | Error Type |
|--------|--------|-----------|----------|------------|
| **bucket_18-33s** | ❌ FAILED | 2 | 0.71s | TypeError (quantile on boolean) |
| **bucket_13-18s** | ❌ FAILED | 2 | 0.03s | TypeError (quantile on boolean) |
| **bucket_60-90s** | ❌ FAILED | 2 | 0.04s | UnboundLocalError (video_count) |

**Note**: Exit code 2 = "Generation failure" according to Stage 6 exit codes.

### Expected vs Actual File Counts

| Bucket | Expected JSONs | Actual JSONs | Notes |
|--------|---------------|--------------|-------|
| bucket_18-33s | 13 | 0 | Rolled back (atomic failure) |
| bucket_13-18s | 7 | 0 | Rolled back (atomic failure) |
| bucket_60-90s | 15 | 0 | Rolled back (atomic failure, 1 temp file deleted) |

**Good News**: The atomic failure pattern worked correctly - no partial output was left behind.

---

## Bug Analysis

### Bug #1: Boolean/Object Column Quantile Computation

**Location**: `ml_pipeline/stage6_analysis/ml_analysis_generation.py:243`
**Function**: `generate_video_rf_json()`
**Severity**: 🔴 **CRITICAL** - Blocks all buckets

**Error Message**:
```
TypeError: numpy boolean subtract, the `-` operator, is not supported,
use the bitwise_xor, the `^` operator, or the logical_xor function instead.
```

**Root Cause**:
The code attempts to compute percentile thresholds on ALL features in `aggregated_features.csv`, including:
- **Boolean columns**: `hook_has_captions`, `middle_1_has_captions`, etc. (6 total)
- **Object columns**: `create_time`, `gender` (2 total)

When pandas tries to run `.quantile()` on boolean or object data, numpy throws this error.

**Problematic Code** (lines 235-244):
```python
# This fails when feature_name is a boolean column
top_performers = df[df['is_top_performer'] == 1][feature_name]
bottom_performers = df[df['is_top_performer'] == 0][feature_name]

top_avg = float(top_performers.mean())
bottom_avg = float(bottom_performers.mean())
gap = abs(top_avg - bottom_avg)

# ❌ FAILS HERE if feature is boolean/object
high_threshold = float(top_performers.quantile(HIGH_PERCENTILE))
low_threshold = float(top_performers.quantile(LOW_PERCENTILE))
```

**Data Evidence**:
```
bucket_18-33s aggregated_features.csv:
- Total columns: 129
- Boolean columns: 6 (hook_has_captions, middle_*_has_captions, closing_has_captions)
- Object columns: 2 (create_time, gender)
- Numeric columns: 121
```

**Recommended Fix**:
Add data type check before computing quantiles:
```python
# Before line 243, add:
if top_performers.dtype == 'bool' or top_performers.dtype == 'object':
    # Skip quantile computation for non-numeric features
    feature_data['top_performer_avg'] = None
    feature_data['bottom_performer_avg'] = None
    feature_data['gap'] = None
    feature_data['distribution'] = None
    logger.debug(f"Skipping distribution for non-numeric feature: {feature_name}")
    continue

# Or use pandas dtype check:
if not pd.api.types.is_numeric_dtype(top_performers):
    # Skip non-numeric features
    continue
```

**Alternative Fix** (more robust):
Filter features to only include numeric columns BEFORE extracting top features:
```python
# In generate_video_rf_json(), after loading aggregated_features.csv
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Then when extracting top features:
for idx in importance_indices:
    feature_name = feature_names[idx]
    if feature_name not in numeric_features:
        continue  # Skip non-numeric features entirely
    # ... rest of code
```

---

### Bug #2: Undefined Variable in Conditional Path

**Location**: `ml_pipeline/stage6_analysis/ml_analysis_generation.py:378`
**Function**: `generate_window_rf_json()`
**Severity**: 🟡 **HIGH** - Blocks bucket_60-90s only (other buckets fail earlier on Bug #1)

**Error Message**:
```
UnboundLocalError: cannot access local variable 'video_count'
where it is not associated with a value
```

**Root Cause**:
The variable `video_count` is only defined inside the `if` branch (line 347) when `is_top_performer` is missing from the CSV. However, it's used outside the conditional block at line 378.

**Problematic Code** (lines 345-352 and 378):
```python
if 'is_top_performer' not in df.columns:
    logger.warning(f"{window}_rf_transformed.csv missing is_top_performer column, calculating fallback")
    video_count = len(df)  # ✅ Defined here
    top_count = int(video_count * TOP_PERFORMER_PERCENTAGE)
    df['is_top_performer'] = [1] * top_count + [0] * (video_count - top_count)
else:
    logger.debug(f"Using existing is_top_performer column from {window}_rf_transformed.csv")
    # ❌ video_count NOT defined in else branch

# Lines later...
analysis_json = {
    'model_type': 'window_level_rf',
    'window_type': window,
    'bucket': bucket,
    'total_videos': video_count,  # ❌ FAILS HERE if else branch taken
    # ...
}
```

**Why bucket_60-90s Failed Here**:
- bucket_18-33s and bucket_13-18s failed earlier on Bug #1
- bucket_60-90s **passed** the video-level RF generation (Bug #1 didn't trigger)
- Then failed on window-level RF generation because `is_top_performer` WAS present in the CSV (else branch taken)

**Recommended Fix**:
Define `video_count` before the conditional:
```python
# Before line 345, add:
video_count = len(df)

# Then the conditional becomes:
if 'is_top_performer' not in df.columns:
    logger.warning(f"{window}_rf_transformed.csv missing is_top_performer column, calculating fallback")
    top_count = int(video_count * TOP_PERFORMER_PERCENTAGE)
    df['is_top_performer'] = [1] * top_count + [0] * (video_count - top_count)
else:
    logger.debug(f"Using existing is_top_performer column from {window}_rf_transformed.csv")
```

---

## Pre-Flight Validation Results

✅ **All pre-flight checks passed successfully**:

### bucket_18-33s
- Stage 4 files: ✅ 14 files exist
- Stage 5 files: ✅ 26 files exist
- Windows: `['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']` (6 windows)

### bucket_13-18s
- Stage 4 files: ✅ 8 files exist
- Stage 5 files: ✅ 14 files exist
- Windows: `['hook', 'middle_aggregate', 'closing']` (3 windows)

### bucket_60-90s
- Stage 4 files: ✅ 16 files exist
- Stage 5 files: ✅ 30 files exist
- Windows: `['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing']` (7 windows)

**Conclusion**: Stage 4 and Stage 5 outputs are complete and valid. The failures are purely due to Stage 6 code bugs.

---

## What Worked Well

1. ✅ **Atomic Failure Pattern**: All temp files were properly rolled back on failure
2. ✅ **Pre-Flight Validation**: Successfully detected all Stage 4 and Stage 5 dependencies
3. ✅ **Error Logging**: Clear error messages with stack traces for debugging
4. ✅ **Exit Codes**: Proper exit code 2 (generation failure) returned

---

## Next Steps

### Immediate Actions Required

1. **Fix Bug #1**: Add numeric data type check before quantile computation
   - Priority: 🔴 CRITICAL
   - Estimated effort: 5-10 minutes
   - Test: Run bucket_18-33s to verify fix

2. **Fix Bug #2**: Move `video_count = len(df)` before conditional
   - Priority: 🟡 HIGH
   - Estimated effort: 2 minutes
   - Test: Run bucket_60-90s to verify fix

3. **Re-run Stage 6 Testing**: Execute all 3 buckets after fixes
   - Expected outcome: All 3 buckets should pass
   - Expected file counts: 13 + 7 + 15 = 35 JSON files total

### Testing Recommendations

After fixes are applied:

```bash
# Re-run Stage 6 for all buckets
cd /home/jorge/rumiaifinal
/home/jorge/rumiaifinal/venv/bin/python3 -c "
from ml_pipeline.stage6_analysis.ml_analysis_generation import generate_ml_analysis_jsons
from config.bucket_definitions import BUCKET_WINDOWS

for bucket_name, bucket_id in [('bucket_18-33s', '18-33s'), ('bucket_13-18s', '13-18s'), ('bucket_60-90s', '60-90s')]:
    bucket_path = f'data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/{bucket_name}'
    windows = BUCKET_WINDOWS[bucket_id]
    exit_code = generate_ml_analysis_jsons(bucket_path, bucket_id, windows)
    print(f'{bucket_name}: exit_code={exit_code}')
"

# Then validate outputs:
ls data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_*/ml_analysis/*_analysis.json | wc -l
# Expected: 35 files
```

---

## Impact Assessment

### Blocked Stages
- ❌ **Stage 6**: Cannot complete (current stage)
- ❌ **Stage 7**: Blocked (depends on Stage 6 JSON outputs)

### Overall Pipeline Status
- ✅ Stage 1-2: Complete
- ✅ Stage 3: Complete (111 videos aggregated)
- ✅ Stage 4: Complete (feature transformation)
- ✅ Stage 5: Complete (ML model training)
- ❌ Stage 6: **BLOCKED** (code bugs)
- ⏳ Stage 7: Pending (awaiting Stage 6 fix)

---

## Technical Notes

### Why bucket_60-90s Behaved Differently

**bucket_60-90s passed Bug #1 but failed on Bug #2**, while bucket_18-33s and bucket_13-18s failed on Bug #1.

**Hypothesis**: The RandomForest model for bucket_60-90s may have ranked numeric features higher in importance, so the top 10 features didn't include any boolean columns. This allowed it to pass the video-level RF generation, only to fail later on the window-level RF generation due to Bug #2.

**Evidence**: Partial output before failure:
```
bucket_60-90s:
  ✓ Generated rf_video_analysis.json.tmp  ← Video-level succeeded
  ❌ Failed on window-level RF (video_count error)
```

This is consistent with the hypothesis that the video-level top 10 features were all numeric.

---

## Conclusion

Stage 6 testing successfully identified 2 critical code bugs that prevent JSON generation. The bugs are straightforward to fix and do not represent architectural issues - just missing edge case handling (non-numeric features) and a variable scoping oversight.

Once these fixes are applied, Stage 6 should complete successfully for all 3 buckets, producing 35 JSON files ready for Stage 7 LLM analysis.

**Estimated time to fix and re-test**: 15-20 minutes
