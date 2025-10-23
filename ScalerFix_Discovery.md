# Scaler Fix - Pre-Implementation Discovery

**Date**: 2025-10-23
**Status**: Discovery Complete - Ready for Implementation Decision

---

## Executive Summary

### Critical Findings

1. ✅ **Stage 4 uses manual scaling** (not sklearn MinMaxScaler objects)
2. ✅ **No scaler infrastructure exists** - need to build from scratch
3. ✅ **Stage 6 not implemented yet** - no downstream code to break
4. ✅ **Original data available** - can recreate scalers perfectly
5. ⚠️ **Implementation more complex than expected** - need to refactor Stage 4

---

## Discovery 1: Stage 4 Current Implementation

### Code Analysis

**File**: `rumiai_v2/processors/feature_transformation.py`
**Function**: `transform_window_level_kmeans()` (lines 635-724)

### What Stage 4 Currently Does

**Manual MinMax Scaling** (lines 671-698):
```python
# Current implementation - NO scaler objects!
for feature in features:
    min_val = df_km[feature].min()
    max_val = df_km[feature].max()
    if max_val > min_val:
        df_km[f'{feature}_scaled'] = (df_km[feature] - min_val) / (max_val - min_val)
    else:
        df_km[f'{feature}_scaled'] = 0.5

    df_km.drop(columns=[feature], inplace=True)  # ← Original values LOST!
```

**Problems**:
- ❌ No `MinMaxScaler` objects created
- ❌ `min_val` and `max_val` calculated but NOT saved
- ❌ Original unscaled columns DROPPED after transformation
- ❌ No way to reproduce scaling for new data

### What Gets Saved

**Output Files** (lines 963-979):
```python
output_files = {'rf_transformed.csv': df_rf}
for window in windows:
    output_files[f'{window}_rf_transformed.csv'] = window_rf_dfs[window]
    output_files[f'{window}_km_transformed.csv'] = window_km_dfs[window]  # ← Only CSVs!

# Write CSVs to disk
for filename, df_output in output_files.items():
    output_path = os.path.join(output_dir, filename)
    df_output.to_csv(output_path, index=False)
```

**Result**: Only CSV files saved, no .pkl files at all

---

## Discovery 2: Downstream Impact Analysis

### Stage 5 (ML Model Training)

**File**: `rumiai_v2/processors/model_training.py`
**Status**: ✅ **Implemented**

**What it expects** (lines 927-935):
```python
# Expects scalers from Stage 4
scaler_source = os.path.join(bucket_base, f'ml_analysis/{window}_scalers.pkl')
scaler_dest = os.path.join(bucket_base, f'models/{window}_scalers_{bucket}.pkl')

if os.path.exists(scaler_source):
    joblib.dump(joblib.load(scaler_source), scaler_dest)  # COPY
    trained_models.append(scaler_dest)
else:
    logger.warning(f"Scaler file missing for {window}: {scaler_source}. Skipping scaler copy.")
```

**Conclusion**:
- ✅ Handles missing scalers gracefully (logs warning)
- ✅ If we create scalers in Stage 4, Stage 5 will automatically copy them
- ✅ **No breaking changes** - Stage 5 already has infrastructure to handle scalers

---

### Stage 6 (ML Analysis Generation)

**File**: DOES NOT EXIST
**Status**: ❌ **Not Implemented Yet**

**Search Results**:
```bash
$ ls rumiai_v2/processors/
feature_transformation.py  # Stage 4
model_training.py          # Stage 5
temporal_compute.py        # Stage 3
timeline_builder.py        # Stage 2
video_analyzer.py          # Stage 2
# NO Stage 6 file!
```

**Implications**:
- ✅ **No existing downstream code to break**
- ✅ Can design scaler format however we want
- ✅ When Stage 6 is implemented, it will expect scalers (per TI docs)
- ⚠️ Need to ensure our scaler format matches what Stage 6 TI expects

---

### Stage 7+ (LLM Analysis, Reports)

**Status**: ❌ Not implemented

**Conclusion**: No downstream dependencies beyond Stage 5

---

## Discovery 3: Upstream Impact Analysis

### Stage 3 (Feature Aggregation)

**File**: `rumiai_v2/processors/temporal_compute.py`
**Output**: `aggregated_features.csv`

**Status**: ✅ **No changes needed**

**Verification**:
```bash
$ ls data/.../ml_analysis/aggregated_features.csv
-rw-r--r-- 1 jorge jorge 55K Oct 22 18:42 aggregated_features.csv
```

**Columns Available**:
- `hook_scene_count`, `hook_word_count`, etc. (RAW unscaled values)
- These are the values we need to fit scalers on

**Conclusion**:
- ✅ Stage 3 output unchanged
- ✅ Has all data needed to fit scalers
- ✅ **No upstream impact**

---

## Discovery 4: Implementation Complexity Analysis

### Current vs Required Implementation

#### **Current State** (Manual Scaling)

```python
# Inline calculation - min/max lost after use
min_val = df[feature].min()
max_val = df[feature].max()
df[f'{feature}_scaled'] = (df[feature] - min_val) / (max_val - min_val)
df.drop(columns=[feature], inplace=True)  # ← Can't recreate scaler!
```

#### **Required State** (sklearn Scalers)

```python
from sklearn.preprocessing import MinMaxScaler
import joblib

# Fit scaler (preserves min/max)
scaler = MinMaxScaler()
scaler.fit(df[[feature]])

# Apply transformation
df[f'{feature}_scaled'] = scaler.transform(df[[feature]]).flatten()
df.drop(columns=[feature], inplace=True)

# Save scaler for inference
scalers[feature] = scaler

# After all features processed:
joblib.dump(scalers, f'ml_analysis/{window}_scalers.pkl')
```

### Changes Needed

**1. Refactor `transform_window_level_kmeans()` function**:
```python
# OLD signature
def transform_window_level_kmeans(df: pd.DataFrame, window_type: str) -> pd.DataFrame

# NEW signature
def transform_window_level_kmeans(df: pd.DataFrame, window_type: str) -> Tuple[pd.DataFrame, dict]
    # Returns: (transformed_df, scalers_dict)
```

**2. Update caller to handle scalers**:
```python
# OLD (line 957)
df_window_km = transform_window_level_kmeans(df, window)

# NEW
df_window_km, scalers = transform_window_level_kmeans(df, window)
window_scalers[window] = scalers  # Store for saving
```

**3. Add scaler saving to output files** (line 966):
```python
# Add to output_files dict (or save separately as .pkl)
for window in windows:
    scaler_path = os.path.join(output_dir, f'{window}_scalers.pkl')
    joblib.dump(window_scalers[window], scaler_path)
```

**Complexity**: Medium (requires function signature change + caller updates)

---

## Discovery 5: Data Consistency Analysis

### Question: Will re-running Stage 4 produce identical results?

**Determinism Check**:

**Deterministic Operations** ✅:
- MinMax scaling: `(x - min) / (max - min)` (pure math)
- Log transform: `log1p(x)` (pure math)
- One-hot encoding: Deterministic mapping

**Non-Deterministic Operations** ❌:
- None found!

**Random Seeds**: None used in Stage 4

**Conclusion**:
- ✅ Re-running Stage 4 will produce **IDENTICAL** output
- ✅ Safe to re-run without invalidating downstream work
- ✅ Scalers will have same min/max as current data

---

## Discovery 6: Test Impact Analysis

### Question: Will adding scalers break existing tests?

**Current Test Files**:
```bash
$ find . -name "*test*.py" -o -name "test_*"
# (check if any Stage 4 tests exist)
```

**File Count Expectations**:
- Current: 13 CSV files per bucket (rf + kmeans)
- After fix: 13 CSV files + 6-7 .pkl files per bucket

**Validation Logic**:
- Stage 4 validation: Checks CSV files only (no .pkl checks)
- Stage 5 validation: Expects scalers (currently fails)

**Conclusion**:
- ⚠️ Need to verify no hardcoded file counts
- ✅ Adding .pkl files shouldn't break CSV-based validation
- ✅ Will FIX Stage 5 validation (currently failing)

---

## Discovery 7: Scaler Format Analysis

### Question: What format should scalers be saved in?

**Stage 5 Expectations** (from code):
```python
scalers = joblib.load('ml_analysis/hook_scalers.pkl')  # ← Expects joblib format
```

**Stage 6 Expectations** (from TI doc):
```markdown
File: hook_scalers.pkl
Format: joblib pickle (dict of MinMaxScaler objects)
Structure:
{
    'scene_count': MinMaxScaler(data_min=[1], data_max=[20]),
    'word_count': MinMaxScaler(data_min=[50], data_max=[800]),
    ...
}
```

**Required Format**:
```python
# Dictionary of sklearn MinMaxScaler objects
scalers = {
    'scene_count': MinMaxScaler(),  # fitted on training data
    'word_count': MinMaxScaler(),
    # ... 25 more features
}

# Save as pickle
joblib.dump(scalers, 'hook_scalers.pkl')
```

**Conclusion**:
- ✅ Format is well-specified
- ✅ Standard sklearn + joblib (no custom serialization)
- ✅ Easy to implement and test

---

## Discovery 8: Alternative Approaches Re-evaluated

### Option 2A: Minimal Refactor (Use sklearn in-place)

**What**: Replace manual scaling with sklearn, minimal code changes

```python
from sklearn.preprocessing import MinMaxScaler

def transform_window_level_kmeans(df, window_type):
    scalers = {}

    # Replace manual scaling (lines 671-698)
    for feature in log_scale_features:
        if feature in df_km.columns:
            # Log transform
            df_km[feature] = np.log1p(df_km[feature])

            # Use sklearn scaler
            scaler = MinMaxScaler()
            df_km[f'{feature}_scaled'] = scaler.fit_transform(df_km[[feature]]).flatten()
            scalers[feature] = scaler  # ← SAVE

            df_km.drop(columns=[feature], inplace=True)

    return df_km, scalers  # ← Return both
```

**Pros**:
- ✅ Minimal code changes (~30 lines modified)
- ✅ Function signature change only (backward compatible if using keyword args)
- ✅ Same output data (just different internal mechanism)

**Cons**:
- ⚠️ Need to update caller (line 957)
- ⚠️ Need to add scaler saving logic (new code)

**Estimated Time**: 45-60 minutes

---

### Option 2B: Major Refactor (Separate scaling and saving)

**What**: Extract scaling logic into separate function

```python
def fit_scalers(df, features):
    """Fit scalers on raw data."""
    scalers = {}
    for feature in features:
        scaler = MinMaxScaler()
        scaler.fit(df[[feature]])
        scalers[feature] = scaler
    return scalers

def apply_scalers(df, scalers):
    """Apply pre-fitted scalers to data."""
    for feature, scaler in scalers.items():
        df[f'{feature}_scaled'] = scaler.transform(df[[feature]]).flatten()
    return df

def transform_window_level_kmeans(df, window_type):
    # Extract features
    # Fit scalers
    scalers = fit_scalers(df_km, all_features)
    # Apply transformations
    df_km = apply_scalers(df_km, scalers)
    return df_km, scalers
```

**Pros**:
- ✅ Cleaner separation of concerns
- ✅ Easier to test scaling independently
- ✅ Easier to reuse fit_scalers for inference

**Cons**:
- ❌ More code changes (~100 lines)
- ❌ Longer implementation time

**Estimated Time**: 90-120 minutes

---

## Discovery 9: Checkpoint/Resume Impact

### Question: Will adding scalers affect checkpoint/resume?

**Current Checkpoint** (feature_transformation.py):
```python
def validate_outputs_and_checkpoint(output_files, bucket, total_videos, bucket_path):
    # Validates CSV files
    # Saves checkpoint JSON
    checkpoint = {
        'stage': 'stage_4',
        'bucket': bucket,
        'total_videos': total_videos,
        'files': list(output_files.keys())  # ← Only CSVs listed
    }
```

**Impact**:
- ⚠️ Checkpoint lists CSV files only
- ⚠️ Adding .pkl files might need checkpoint update
- ✅ But checkpoint is for resume detection, not file validation
- ✅ Stage 5 does its own validation (doesn't use Stage 4 checkpoint for file list)

**Conclusion**:
- ✅ Adding scalers won't break checkpoint/resume
- ⚠️ Consider adding .pkl files to checkpoint for completeness

---

## Recommendations

### Recommended Approach: Option 2A (Minimal Refactor)

**Why**:
1. ✅ **Lowest Risk**: Minimal code changes
2. ✅ **Fastest**: 45-60 min implementation
3. ✅ **No Downstream Impact**: Stage 6 doesn't exist yet
4. ✅ **No Upstream Impact**: Stage 3 unchanged
5. ✅ **Fixes Stage 5**: Provides expected scalers
6. ✅ **Production Ready**: Standard sklearn + joblib

### Implementation Plan (Refined)

**Phase 1: Code Changes** (45-60 min)

1. **Import sklearn** (top of file):
   ```python
   from sklearn.preprocessing import MinMaxScaler
   import joblib
   ```

2. **Refactor `transform_window_level_kmeans()`** (lines 635-724):
   - Replace manual scaling with `MinMaxScaler`
   - Return `(df, scalers)` tuple
   - **Estimated**: 30-40 min

3. **Update caller** (lines 956-960):
   - Handle returned scalers
   - Store in `window_scalers` dict
   - **Estimated**: 5 min

4. **Add scaler saving** (after line 979):
   - Save each window's scalers as .pkl
   - **Estimated**: 10-15 min

**Phase 2: Testing** (15-20 min)

1. Test on single bucket first
2. Verify scaler files created
3. Verify scalers loadable
4. Check scaler values make sense (min/max match data)

**Phase 3: Documentation** (15-20 min)

1. Update FeatureTransformationTI.md
2. Add scaler schema
3. Document output files

**Phase 4: Re-run Stage 4** (20-30 min)

1. Re-run for all 3 test buckets
2. Verify all scaler files created

**Phase 5: Verify Stage 5** (10-15 min)

1. Run Stage 5 training
2. Confirm scalers copied successfully
3. Verify validation passes

**Total Time**: ~105-145 minutes (~2-2.5 hours)

---

## Risk Assessment

### Low Risk Items ✅

- No downstream code to break (Stage 6 not implemented)
- No upstream changes needed (Stage 3 unchanged)
- Deterministic output (safe to re-run)
- Standard libraries (sklearn + joblib)
- Stage 5 already handles scalers gracefully

### Medium Risk Items ⚠️

- Function signature change (but internal to Stage 4)
- Need to test with real data
- Checkpoint might need update (optional)

### High Risk Items ❌

- None identified!

---

## Decision Matrix

| Factor | Option 1 (Stage 5 Creates) | Option 2A (Minimal Refactor) | Option 2B (Major Refactor) |
|--------|-------------------------|----------------------------|---------------------------|
| **Implementation Time** | 30 min | 60 min | 120 min |
| **Re-run Time** | 0 | 30 min | 30 min |
| **Risk Level** | Medium | Low | Medium |
| **Technical Debt** | High | None | None |
| **Downstream Impact** | None | None | None |
| **Upstream Impact** | Violates boundaries | None | None |
| **Maintainability** | Low | High | Very High |
| **Production Ready** | ⚠️ Yes | ✅ Yes | ✅ Yes |
| **Total Cost** | 30 min | 90 min | 150 min |

**Recommendation**: **Option 2A** - Best balance of time vs correctness

---

## Next Steps

### Immediate Actions

1. ✅ Discovery complete (this document)
2. ⏸️ **Awaiting decision**: Proceed with Option 2A?
3. ⏸️ If approved: Implement Phase 1 (code changes)

### Questions for User

1. **Approve Option 2A implementation?**
   - Total time: ~2-2.5 hours
   - Low risk, architecturally correct

2. **Test on all 3 buckets or just one first?**
   - Recommend: Test on bucket_18-33s first, verify, then do others

3. **Update documentation first or after code works?**
   - Recommend: Code first, documentation second

---

## Appendix: Key Code Locations

### Files to Modify

1. **`rumiai_v2/processors/feature_transformation.py`**
   - Line 635: Function definition
   - Lines 671-698: Scaling logic
   - Line 957: Function call
   - Line 979: Output file writing

### Files to Create

1. **Scaler files** (per bucket, per window):
   - `ml_analysis/hook_scalers.pkl`
   - `ml_analysis/middle_1_scalers.pkl`
   - etc.

### Files to Update (Documentation)

1. **`documentation_migration/FutureDevelopments/ChildDocs/FeatureTransformationTI.md`**
   - Section 2.2: Output contract
   - Section 3: Add scaler schema
   - Section 4: Add scaler creation logic

---

**End of Discovery Document**
