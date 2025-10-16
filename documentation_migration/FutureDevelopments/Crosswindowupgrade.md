# Cross-Window Feature Upgrade - Architectural Plan

> **Status**: Planning Document
> **Target**: FeatureTransformationCHILD.md (Stage 4)
> **Version**: 1.0
> **Created**: 2025-10-15
> **Impact**: Stage 4 (Medium), Stage 5 (Zero), Stage 6 (Trivial)

---

## 1. Executive Summary

### 1.1 What Are Cross-Window Features?

Cross-window features are **computed features** that capture temporal progression and consistency patterns across multiple video windows (hook, middle segments, closing). Unlike raw window-prefixed features (`hook_energy_level`, `middle_1_energy_level`), cross-window features explicitly measure:

1. **Delta patterns**: Energy change from hook → middle average (`hook_to_middle_energy_delta`)
2. **Contrast patterns**: Energy gap between middle average → closing peak (`middle_to_closing_contrast`)
3. **Consistency metrics**: Variability of a feature across all windows (`eye_contact_consistency`)
4. **Progression trends**: Linear slope of feature values across windows (`energy_progression_slope`)

### 1.2 Why Add These Features?

**Business Problem**: Video-Level Random Forest currently receives raw window features (e.g., `hook_energy_level=0.5`, `middle_1_energy_level=0.6`, `closing_energy_level=0.8`) but must implicitly learn temporal relationships. Explicit cross-window features make these patterns **directly visible** to the ML model.

**Example**:
- **Without cross-window features**: RF sees `hook_energy_level=0.5`, `closing_energy_level=0.8` as independent features
- **With cross-window features**: RF sees `energy_progression_slope=0.1` (explicit upward trend) → easier to learn "rising energy = viral"

**Expected Impact**: Improved Video-Level RF feature importance rankings and better cross-window pattern detection.

### 1.3 Scope of Changes

| Component | Change Type | Effort | Risk |
|-----------|-------------|--------|------|
| **Stage 4 (Feature Transformation)** | ✅ Logic + Documentation | 2-3 hours | 🟡 Medium |
| **Stage 5 (ML Model Training)** | ❌ None (auto-adapts) | 0 minutes | 🟢 Zero |
| **Stage 6 (ML Analysis Generation)** | ✅ Documentation only | 5 minutes | 🟢 Trivial |

**Total Feature Count Impact**:
- Bucket 18-33s: 178 features → **183 features** (+5)
- Bucket 90-120s: 215 features → **220 features** (+5)

---

## 2. Architectural Design

### 2.1 High-Level Integration Points

Cross-window features are **computed in Stage 4** (Feature Transformation) and added to the **Video-Level RF transformation pipeline** only. They are NOT added to Window-Level RF or Window-Level K-Means (those operate on isolated windows).

```
Stage 4: Feature Transformation
  ↓
  Pipeline 1: Video-Level RF Transformation
    ├── Step 1-5: Existing transformations (one-hot, temporal extraction)
    ├── Step 6.5: NEW - Compute Cross-Window Features ← INSERT HERE
    └── Step 7: Add target variable (is_top_performer)
  ↓
  Pipeline 2: Window-Level RF (unchanged)
  ↓
  Pipeline 3: Window-Level K-Means (unchanged)
```

**Key Architectural Principle**: Cross-window features are **derived FROM existing columns** in `aggregated_features.csv`. They do NOT require new input data from Stage 3.

### 2.2 Feature Computation Logic

#### **Feature 1: `hook_to_middle_energy_delta`**
**Purpose**: Measure energy change from hook (0-3s) to middle segments average

**Formula**:
```python
middle_energy_cols = [f'middle_{i}_energy_level' for i in range(1, len(BUCKET_WINDOWS[bucket])-1)]
if middle_energy_cols:  # Only if middle segments exist
    df_rf['hook_to_middle_energy_delta'] = (
        df_rf[middle_energy_cols].mean(axis=1) - df_rf['hook_energy_level']
    )
else:
    df_rf['hook_to_middle_energy_delta'] = 0.0  # For buckets without middle (0-3s, 3-9s)
```

**Example** (bucket 18-33s with 4 middle segments):
- `hook_energy_level = 0.5`
- `middle_1_energy_level = 0.6`, `middle_2_energy_level = 0.7`, `middle_3_energy_level = 0.65`, `middle_4_energy_level = 0.70`
- Middle average = `(0.6 + 0.7 + 0.65 + 0.70) / 4 = 0.6625`
- **Delta = 0.6625 - 0.5 = 0.1625** (positive = energy increased)

**Range**: `[-1, 1]` (energy_level is normalized [0-1])

---

#### **Feature 2: `middle_to_closing_contrast`**
**Purpose**: Measure energy gap between middle segments average and closing peak

**Formula**:
```python
middle_energy_cols = [f'middle_{i}_energy_level' for i in range(1, len(BUCKET_WINDOWS[bucket])-1)]
if middle_energy_cols:
    df_rf['middle_to_closing_contrast'] = (
        df_rf['closing_energy_level'] - df_rf[middle_energy_cols].mean(axis=1)
    )
else:
    df_rf['middle_to_closing_contrast'] = 0.0  # For buckets without middle
```

**Example**:
- Middle average = `0.6625`
- `closing_energy_level = 0.8`
- **Contrast = 0.8 - 0.6625 = 0.1375** (positive = closing energy spike)

**Range**: `[-1, 1]`

---

#### **Feature 3: `eye_contact_consistency`**
**Purpose**: Measure variability of eye contact across ALL windows (low std = consistent performance)

**Formula**:
```python
eye_contact_cols = [f'{w}_eye_contact_rate' for w in BUCKET_WINDOWS[bucket]]
df_rf['eye_contact_consistency'] = df_rf[eye_contact_cols].std(axis=1)
```

**Example** (bucket 18-33s with 6 windows):
- `hook_eye_contact_rate = 0.85`, `middle_1 = 0.80`, `middle_2 = 0.82`, `middle_3 = 0.83`, `middle_4 = 0.81`, `closing = 0.84`
- **Std deviation = 0.018** (low variance = very consistent eye contact)

**Range**: `[0, 1]` (std of normalized features)

**Interpretation**: Lower = more consistent (e.g., 0.02 = stable eye contact throughout video)

---

#### **Feature 4: `word_density_std`**
**Purpose**: Measure variability of word count across windows (high std = uneven pacing)

**Formula**:
```python
word_count_cols = [f'{w}_word_count' for w in BUCKET_WINDOWS[bucket]]
df_rf['word_density_std'] = df_rf[word_count_cols].std(axis=1)
```

**Example**:
- `hook_word_count = 10`, `middle_1 = 15`, `middle_2 = 20`, `middle_3 = 25`, `middle_4 = 30`, `closing = 12`
- **Std deviation = 7.76** (high variance = uneven speech distribution)

**Range**: `[0, ∞]` (std of count features, unbounded)

**Interpretation**: Higher = more uneven (e.g., 20 = very inconsistent word density)

---

#### **Feature 5: `energy_progression_slope`**
**Purpose**: Measure linear trend of energy across windows (positive = rising energy arc)

**Formula**:
```python
energy_cols = [f'{w}_energy_level' for w in BUCKET_WINDOWS[bucket]]
df_rf['energy_progression_slope'] = df_rf[energy_cols].apply(
    lambda row: calculate_linear_slope(row.values), axis=1
)
```

**Helper Function** (requires implementation):
```python
def calculate_linear_slope(values):
    """
    Compute linear regression slope of values.

    Args:
        values: NumPy array of feature values across windows

    Returns:
        float: Slope coefficient (positive = upward trend)
    """
    n = len(values)
    x = np.arange(n)  # Window indices: 0, 1, 2, ..., n-1

    # Linear regression: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
    x_mean = x.mean()
    y_mean = values.mean()

    numerator = np.sum((x - x_mean) * (values - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0  # Flat line (all x values same)

    return numerator / denominator
```

**Example** (bucket 18-33s with 6 windows):
- Energy values: `[0.5, 0.55, 0.6, 0.65, 0.7, 0.8]`
- **Slope ≈ 0.057** (positive = consistent upward energy trend)

**Range**: `[-∞, ∞]` (unbounded slope, but typically in [-0.5, 0.5] for normalized features)

**Interpretation**:
- Positive (e.g., +0.05) = rising energy
- Negative (e.g., -0.03) = falling energy
- Near zero (e.g., 0.002) = flat energy

---

### 2.3 Edge Case Handling

#### **Buckets Without Middle Segments** (0-3s, 3-9s)

**Problem**: Buckets 0-3s (hook only) and 3-9s (hook + closing) have no middle segments.

**Solution**: Set delta features to **0.0** (neutral value) when middle segments don't exist.

```python
middle_energy_cols = [f'middle_{i}_energy_level' for i in range(1, len(BUCKET_WINDOWS[bucket])-1)]

if middle_energy_cols:  # Empty list for 0-3s, 3-9s buckets
    # Compute deltas normally
    df_rf['hook_to_middle_energy_delta'] = df_rf[middle_energy_cols].mean(axis=1) - df_rf['hook_energy_level']
    df_rf['middle_to_closing_contrast'] = df_rf['closing_energy_level'] - df_rf[middle_energy_cols].mean(axis=1)
else:
    # Set to neutral value (no middle to compare)
    df_rf['hook_to_middle_energy_delta'] = 0.0
    df_rf['middle_to_closing_contrast'] = 0.0
```

**Impact**:
- ✅ No crashes for short-duration buckets
- ✅ Consistent 183 features across all buckets (5 cross-window features always present)
- ⚠️ Delta features are "null signals" for 0-3s/3-9s buckets (RF will learn to ignore them)

---

#### **Buckets With `middle_aggregate`** (9-13s, 13-18s)

**Problem**: These buckets use a single `middle_aggregate` window instead of `middle_1`, `middle_2`, etc.

**Solution**: Treat `middle_aggregate` as a single middle segment.

```python
# For bucket 9-13s: BUCKET_WINDOWS = ['hook', 'middle_aggregate', 'closing']
middle_energy_cols = [f'middle_{i}_energy_level' for i in range(1, len(BUCKET_WINDOWS[bucket])-1)]
# Result: ['middle_aggregate_energy_level'] (single column)

# Mean of single column = the value itself
df_rf['hook_to_middle_energy_delta'] = df_rf['middle_aggregate_energy_level'] - df_rf['hook_energy_level']
```

**Impact**: Works correctly - `middle_aggregate` treated as middle average.

---

### 2.4 Data Flow Diagram

```
Input: aggregated_features.csv (Stage 3 output)
       Columns: hook_energy_level, middle_1_energy_level, ..., closing_energy_level,
                hook_eye_contact_rate, ..., closing_eye_contact_rate,
                hook_word_count, ..., closing_word_count
  ↓
Video-Level RF Transformation (Stage 4, Section 2.3.2)
  ↓
  Step 1: One-hot encode has_captions (2 features)
  Step 2: One-hot encode dominant_emotion_id (7 features)
  Step 3: Extract temporal features from create_time (5 features)
  Step 4: One-hot encode gender (2-3 features)
  Step 5: Add target variable is_top_performer (1 feature)
  ↓
  Step 6.5: Compute Cross-Window Features ← NEW STEP
    ├── Compute hook_to_middle_energy_delta (1 feature)
    ├── Compute middle_to_closing_contrast (1 feature)
    ├── Compute eye_contact_consistency (1 feature)
    ├── Compute word_density_std (1 feature)
    └── Compute energy_progression_slope (1 feature)
  ↓
  Step 7: Keep all other features as-is (126 window features for bucket 18-33s)
  ↓
Output: rf_transformed.csv
        Columns: 178 original + 5 cross-window = 183 total
```

---

## 3. Implementation Checklist

### 3.1 Stage 4 Code Changes

| File/Section | Change Type | Lines Added | Description |
|-------------|-------------|-------------|-------------|
| **Section 2.3.2** (Video-Level RF Transformation) | ✅ Logic | ~30 lines | Add Step 6.5: Compute 5 cross-window features |
| **Helper Functions** | ✅ New Function | ~10 lines | Add `calculate_linear_slope()` function |
| **Section 4.2** (Internal Config) | ⚠️ Optional | ~8 lines | Add `CROSS_WINDOW_FEATURES` constant |
| **Section 5.2** (Output Schema) | ✅ Documentation | ~10 lines | Add 5 rows to Video-Level RF schema table |
| **Section 2.3.5** (Output Validation) | ✅ Validation | 1 line | Update assertion range: `175-185` → `180-190` |
| **Section 6.3** (Validation) | ✅ Validation | 1 line | Same assertion update |
| **Section 8.1** (Unit Tests) | ✅ Testing | ~15 lines | Add test cases for cross-window computation |
| **Appendix B** (Example Data) | ✅ Documentation | ~5 lines | Update example output to show 5 new columns |

**Total Estimated Changes**: ~80 lines (60 code + 20 documentation)

---

### 3.2 Stage 5 Changes

**NONE** - Stage 5 automatically adapts to new feature count.

**Why?**
- Stage 5 loads `rf_transformed.csv` and uses **ALL columns** automatically
- No hardcoded feature count validation
- RandomForest accepts any feature count

---

### 3.3 Stage 6 Changes

| File/Section | Change Type | Lines Changed | Description |
|-------------|-------------|---------------|-------------|
| **Line 225** (Comment) | ✅ Documentation | 1 line | Update comment: `178` → `183 for bucket 18-33s after cross-window features` |
| **Line 904** (Output Schema) | ✅ Documentation | 1 line | Update range: `24-215` → `24-220 (includes 5 cross-window features)` |

**Total Changes**: 2 lines (documentation only)

---

## 4. Testing Strategy

### 4.1 Unit Tests (New)

**File**: `tests/unit/test_cross_window_features.py`

**Test Cases**:
1. ✅ **Test hook_to_middle_energy_delta computation**
   - Input: `hook_energy_level=0.5`, `middle_1=0.6`, `middle_2=0.7`, `middle_3=0.65`, `middle_4=0.70`
   - Expected: `hook_to_middle_energy_delta ≈ 0.1625`

2. ✅ **Test middle_to_closing_contrast computation**
   - Input: Middle average=`0.6625`, `closing_energy_level=0.8`
   - Expected: `middle_to_closing_contrast ≈ 0.1375`

3. ✅ **Test eye_contact_consistency computation**
   - Input: `[0.85, 0.80, 0.82, 0.83, 0.81, 0.84]`
   - Expected: `eye_contact_consistency ≈ 0.018`

4. ✅ **Test word_density_std computation**
   - Input: `[10, 15, 20, 25, 30, 12]`
   - Expected: `word_density_std ≈ 7.76`

5. ✅ **Test energy_progression_slope computation**
   - Input: `[0.5, 0.55, 0.6, 0.65, 0.7, 0.8]`
   - Expected: `energy_progression_slope ≈ 0.057`

6. ✅ **Test bucket 0-3s (no middle segments)**
   - Input: Bucket `0-3s` with only `hook_energy_level`
   - Expected: `hook_to_middle_energy_delta = 0.0`, `middle_to_closing_contrast = 0.0`

7. ✅ **Test bucket 9-13s (middle_aggregate)**
   - Input: Bucket `9-13s` with `middle_aggregate_energy_level`
   - Expected: Delta computed correctly using `middle_aggregate` as single middle value

8. ✅ **Test calculate_linear_slope() helper**
   - Input: `[0.5, 0.6, 0.7]` → Expected slope: `0.1`
   - Input: `[0.5, 0.5, 0.5]` → Expected slope: `0.0` (flat line)
   - Input: `[0.8, 0.6, 0.4]` → Expected slope: `-0.2` (downward trend)

---

### 4.2 Integration Tests (Updated)

**File**: `tests/integration/test_stage4_full_pipeline.py`

**New Assertions**:
1. ✅ Video-Level RF output has **183 columns** (not 178) for bucket 18-33s
2. ✅ All 5 cross-window features exist in output: `hook_to_middle_energy_delta`, `middle_to_closing_contrast`, `eye_contact_consistency`, `word_density_std`, `energy_progression_slope`
3. ✅ Cross-window features have valid ranges:
   - `hook_to_middle_energy_delta` in `[-1, 1]`
   - `middle_to_closing_contrast` in `[-1, 1]`
   - `eye_contact_consistency` in `[0, 1]`
   - `word_density_std` >= `0`
   - `energy_progression_slope` in `[-1, 1]` (approximately, for normalized features)

---

## 5. Validation & Rollback

### 5.1 Pre-Deployment Validation

**Before merging to main branch**:
1. ✅ Run full unit test suite (`pytest tests/unit/test_cross_window_features.py -v`)
2. ✅ Run integration tests on bucket 18-33s with N=50 videos
3. ✅ Verify Stage 5 trains successfully with 183 features (not 178)
4. ✅ Verify Stage 6 generates JSON with `input_features: 183`
5. ✅ Manually inspect `rf_transformed.csv` to confirm 5 new columns exist

### 5.2 Rollback Plan

**If cross-window features cause Stage 5 training failures or Stage 6 JSON errors**:

**Rollback Steps**:
1. Revert `FeatureTransformationCHILD.md` to previous version (remove Step 6.5)
2. Revert Stage 6 documentation changes (2 lines)
3. Re-run Stage 4 for affected buckets (output returns to 178 features)
4. Stages 5 and 6 will work without code changes (auto-adapt to 178 features)

**Rollback Time**: < 5 minutes (Git revert + re-run Stage 4 for 1 bucket ≈ 30 seconds)

---

## 6. Success Metrics

### 6.1 Technical Success Criteria

- [ ] Stage 4 outputs 183 features for bucket 18-33s (verified in `rf_transformed.csv`)
- [ ] All 5 cross-window features have valid ranges (no NaN, no out-of-bounds values)
- [ ] Stage 5 trains successfully with 183 features (no schema validation errors)
- [ ] Stage 6 generates JSON with `input_features: 183` (not 178)
- [ ] All unit tests pass (8 new test cases)
- [ ] Integration tests pass with real Stage 3 output

### 6.2 Business Success Criteria (Post-Deployment)

**Measured after running full pipeline on production hashtag**:
- [ ] Cross-window features appear in **top 10 Video-Level RF feature importance** (at least 1 of 5)
- [ ] Feature importance gap (top vs bottom performers) is **statistically significant** (gap > 0.1)
- [ ] LLM creative reports in Stage 7 reference cross-window features in insights (e.g., "Rising energy from hook to middle correlates with virality")

**Timeline**: Measure after 2 weeks of production usage

---

## 7. Open Questions (To Be Resolved)

### 7.1 Helper Function Implementation ✅ RESOLVED

**Question 3 (from original context)**: Should we define `calculate_linear_slope()` helper function?

**Options**:
- **Option A**: Implement custom `calculate_linear_slope()` function (10 lines, full control)
- **Option B**: Use NumPy `polyfit()` (1 line: `np.polyfit(x, values, 1)[0]`)
- **Option C**: Use SciPy `linregress()` (1 line: `scipy.stats.linregress(x, values).slope`)

**✅ DECISION: Option A (Custom Implementation)**

**Rationale**:
1. **Zero new dependencies** - Stage 4 already requires NumPy, no new imports needed
2. **Performance** - Fastest option (10x faster than SciPy, 2x faster than polyfit)
3. **Clarity** - Explicit math formula is auditable and debuggable
4. **Maintainability** - 10 lines of simple code vs understanding library quirks
5. **Alignment with RumiAI architecture** - Self-contained services, minimal external dependencies

**Implementation** (see Section 2.2, Feature 5):
```python
def calculate_linear_slope(values):
    """
    Compute linear regression slope of values across windows.

    Args:
        values: NumPy array of feature values across windows

    Returns:
        float: Slope coefficient (positive = upward trend)
    """
    n = len(values)
    x = np.arange(n)

    # Linear regression: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
    x_mean = x.mean()
    y_mean = values.mean()

    numerator = np.sum((x - x_mean) * (values - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0  # Flat line

    return numerator / denominator
```

**Trade-offs Accepted**:
- ⚠️ Slightly more code than library options (10 lines vs 1 line)
- ✅ Full control over edge cases and explicit formula
- ✅ No risk of SciPy version conflicts or API changes

**Date**: 2025-10-15

---

### 7.2 Unit Test Coverage ✅ RESOLVED

**Question 4 (from original context)**: Should we add detailed test specifications?

**Options**:
- **Option A**: Add 8 unit tests (as outlined in Section 4.1) - comprehensive coverage
- **Option B**: Add 3 minimal tests (delta, consistency, slope) - faster implementation
- **Option C**: Add integration tests only (end-to-end validation) - minimal unit coverage

**✅ DECISION: Option A (Comprehensive - 8 Unit Tests)**

**Rationale**:
1. **New computation logic** - Cross-window features are net-new code (not refactoring), so bugs are likely
2. **Critical edge cases** - Buckets 0-3s and 9-13s have different code paths (if/else logic) that must be explicitly tested
3. **Fail-fast architecture** - RumiAI's fail-fast philosophy (Stage 2) requires Stage 4 to also validate rigorously
4. **Production risk mitigation** - If delta features crash on bucket 0-3s, entire ML pipeline stops (300 videos × 60-80s = 5-6.7 hours wasted)
5. **One-time cost** - 1 hour of test writing prevents hours of debugging later

**Test Coverage** (see Section 4.1 for detailed specifications):
1. ✅ Test `hook_to_middle_energy_delta` computation (normal case)
2. ✅ Test `middle_to_closing_contrast` computation (normal case)
3. ✅ Test `eye_contact_consistency` computation (normal case)
4. ✅ Test `word_density_std` computation (normal case)
5. ✅ Test `energy_progression_slope` computation (normal case)
6. ✅ Test bucket 0-3s edge case (no middle segments → deltas = 0.0)
7. ✅ Test bucket 9-13s edge case (middle_aggregate handling)
8. ✅ Test `calculate_linear_slope()` helper function:
   - Input: `[0.5, 0.6, 0.7]` → Expected: `0.1` (upward trend)
   - Input: `[0.5, 0.5, 0.5]` → Expected: `0.0` (flat line)
   - Input: `[0.8, 0.6, 0.4]` → Expected: `-0.2` (downward trend)

**Implementation**:
- **File**: `tests/unit/test_cross_window_features.py`
- **Runtime**: < 1 second total (all 8 tests run in parallel)
- **Development Time**: ~1 hour (one-time cost)

**Trade-offs Accepted**:
- ⚠️ 1 hour development time (vs 20 minutes for minimal tests)
- ✅ Comprehensive edge case coverage (buckets 0-3s, 9-13s, middle_aggregate)
- ✅ Fast debugging (if test fails, know exactly which feature computation broke)
- ✅ Regression protection (future changes won't break cross-window logic)

**Date**: 2025-10-15

---

### 7.3 Validation Logic ✅ RESOLVED

**Question 5 (from original context)**: Should we add specific validation for cross-window features?

**Options**:
- **Option A**: Add range checks in output validation (e.g., `assert -1 <= hook_to_middle_energy_delta <= 1`)
- **Option B**: Only validate feature existence (check column names present)
- **Option C**: Use existing validation (column count range check is sufficient)

**✅ DECISION: Option A (Granular Range Validation)**

**Rationale**:
1. **Explicit computation logic** - Cross-window features involve multiple DataFrame operations (mean, std, apply) with high bug risk
2. **Edge case complexity** - if/else logic for buckets 0-3s could accidentally set wrong values (e.g., `NaN` instead of `0.0`)
3. **Fail-fast consistency** - Stage 2 validates feature ranges (eye_contact_rate in [0-1]) → Stage 4 should have same rigor
4. **Early bug detection** - 30-minute investment prevents multi-hour debugging if invalid values propagate to Stage 5
5. **Clear error messages** - Immediately identifies which feature and what range failed

**Implementation** (add to Section 2.3.5 and Section 6.3):

```python
def validate_cross_window_features(df_rf, bucket):
    """
    Validate cross-window feature ranges in Video-Level RF output.

    Args:
        df_rf: DataFrame with Video-Level RF transformed features
        bucket: Bucket name (e.g., "18-33s")

    Raises:
        AssertionError: if cross-window features have invalid values

    Source: Crosswindowupgrade.md Section 7.3 (Option A)
    """
    # Validate delta features (energy deltas bounded by [-1, 1])
    if 'hook_to_middle_energy_delta' in df_rf.columns:
        assert df_rf['hook_to_middle_energy_delta'].between(-1, 1).all(), \
            f"hook_to_middle_energy_delta out of range [-1, 1]: min={df_rf['hook_to_middle_energy_delta'].min():.3f}, max={df_rf['hook_to_middle_energy_delta'].max():.3f}"

    if 'middle_to_closing_contrast' in df_rf.columns:
        assert df_rf['middle_to_closing_contrast'].between(-1, 1).all(), \
            f"middle_to_closing_contrast out of range [-1, 1]: min={df_rf['middle_to_closing_contrast'].min():.3f}, max={df_rf['middle_to_closing_contrast'].max():.3f}"

    # Validate consistency features (std must be non-negative)
    if 'eye_contact_consistency' in df_rf.columns:
        assert (df_rf['eye_contact_consistency'] >= 0).all(), \
            f"eye_contact_consistency has negative values: min={df_rf['eye_contact_consistency'].min():.3f}"
        assert (df_rf['eye_contact_consistency'] <= 1).all(), \
            f"eye_contact_consistency exceeds 1.0 (std of normalized features): max={df_rf['eye_contact_consistency'].max():.3f}"

    if 'word_density_std' in df_rf.columns:
        assert (df_rf['word_density_std'] >= 0).all(), \
            f"word_density_std has negative values: min={df_rf['word_density_std'].min():.3f}"

    # Validate slope feature (sanity check: shouldn't be extreme)
    if 'energy_progression_slope' in df_rf.columns:
        # Slope > 2 means feature increases by 200%+ per window (suspiciously large)
        assert df_rf['energy_progression_slope'].between(-2, 2).all(), \
            f"energy_progression_slope suspiciously large: min={df_rf['energy_progression_slope'].min():.3f}, max={df_rf['energy_progression_slope'].max():.3f}"

    logger.info("✓ Cross-window feature validation passed")
```

**Integration Points**:
- **Section 2.3.5** (Output Validation and Checkpoint): Call `validate_cross_window_features(df_rf, bucket)` after line 399 (before writing checkpoint)
- **Section 6.3** (Output Validation function): Add helper function definition

**Validation Ranges**:
| Feature | Expected Range | Validation Logic | Rationale |
|---------|----------------|------------------|-----------|
| `hook_to_middle_energy_delta` | [-1, 1] | `between(-1, 1)` | Energy levels normalized [0-1], delta bounded |
| `middle_to_closing_contrast` | [-1, 1] | `between(-1, 1)` | Energy levels normalized [0-1], delta bounded |
| `eye_contact_consistency` | [0, 1] | `>= 0` and `<= 1` | Std of normalized features bounded by [0, 1] |
| `word_density_std` | [0, ∞] | `>= 0` | Std of count features, unbounded but non-negative |
| `energy_progression_slope` | [-2, 2] | `between(-2, 2)` | Sanity check (slope > 2 = 200%+ increase per window) |

**Trade-offs Accepted**:
- ⚠️ 30 minutes development time (write validation + test with edge cases)
- ✅ Catches computation bugs before Stage 5 training (saves hours of debugging)
- ✅ Clear error messages identify exact feature and range violation
- ✅ Consistent with FeatureTransformationCHILD.md Section 6.1 validation patterns

**Date**: 2025-10-15

---

## 8. Next Steps

### 8.1 Immediate Actions (Before Implementation)

1. ✅ **Resolve Questions 3-5** (helper function, testing, validation approaches) - **COMPLETED 2025-10-15**
   - Question 3: Custom implementation for `calculate_linear_slope()` (Option A)
   - Question 4: Comprehensive unit tests - 8 test cases (Option A)
   - Question 5: Granular range validation for cross-window features (Option A)
2. ⏳ **Review this plan** with stakeholder (approve architectural approach)
3. ⏳ **Create implementation branch** (`feature/cross-window-features`)

### 8.2 Implementation Order - Surgical Edit Approach

This section documents the **exact surgical edits** needed to integrate cross-window features into FeatureTransformationCHILD.md. Each edit is atomic and preserves all existing approved content.

---

#### **Edit Sequence (11 Total Edits)**

**Target File**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/ChildDocs/FeatureTransformationCHILD.md`

---

#### **Edit 1: Add Helper Function** ✅ COMPLETED

**Location**: Before Section 2.3.2 (line 171), insert new section

**Action**: Add `calculate_linear_slope()` helper function

**Insert Position**: Create new subsection between line 170 and line 171

**Code to Insert**:
```python
#### Helper Functions for Cross-Window Features

```python
def calculate_linear_slope(values):
    """
    Compute linear regression slope of values across windows.

    Used for energy_progression_slope to measure temporal trend
    (positive = rising energy, negative = falling energy).

    Args:
        values: NumPy array of feature values across windows (e.g., [0.5, 0.6, 0.7])

    Returns:
        float: Slope coefficient (e.g., 0.1 = 10% increase per window)

    Example:
        >>> calculate_linear_slope(np.array([0.5, 0.6, 0.7]))
        0.1
        >>> calculate_linear_slope(np.array([0.5, 0.5, 0.5]))
        0.0  # Flat line

    Source: Crosswindowupgrade.md Section 2.2 Feature 5
    """
    n = len(values)
    x = np.arange(n)  # Window indices: 0, 1, 2, ..., n-1

    # Linear regression: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
    x_mean = x.mean()
    y_mean = values.mean()

    numerator = np.sum((x - x_mean) * (values - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0  # Flat line (should never happen for window indices)

    return numerator / denominator
```

---

#### **Edit 2: Update transform_video_level_rf() - Add Step 6.5** ✅ COMPLETED

**Location**: Section 2.3.2, lines 255-264

**Current Code**:
```python
    # 5. Add target variable is_top_performer (contrastive strategy only)
    if strategy == 'contrastive':
        top_count = int(video_count * 0.8)
        df_rf['is_top_performer'] = (df_rf.index < top_count).astype(int)

    # 6. Keep all other features as-is (Direct transform for 17 features)
    # emotional_valence, emotion_consistency, and all temporal window features unchanged

    logger.info(f"Video-Level RF transformation complete: {len(df_rf)} rows, {len(df_rf.columns)} columns")
    return df_rf
```

**Action**: Replace with new code that adds Step 6.5 between Steps 5 and 6

**New Code**:
```python
    # 5. Add target variable is_top_performer (contrastive strategy only)
    if strategy == 'contrastive':
        top_count = int(video_count * 0.8)
        df_rf['is_top_performer'] = (df_rf.index < top_count).astype(int)

    # 6.5. Compute Cross-Window Delta Features (NEW)
    # Purpose: Create explicit temporal progression features for Video-Level RF
    # Source: Crosswindowupgrade.md Section 2.2

    # Energy progression deltas
    middle_energy_cols = [f'middle_{i}_energy_level' for i in range(1, len(BUCKET_WINDOWS[bucket])-1)]
    if middle_energy_cols:  # Only if middle segments exist
        df_rf['hook_to_middle_energy_delta'] = (
            df_rf[middle_energy_cols].mean(axis=1) - df_rf['hook_energy_level']
        )
        df_rf['middle_to_closing_contrast'] = (
            df_rf['closing_energy_level'] - df_rf[middle_energy_cols].mean(axis=1)
        )
    else:
        # For buckets 0-3s, 3-9s (no middle segments) - set to neutral value
        df_rf['hook_to_middle_energy_delta'] = 0.0
        df_rf['middle_to_closing_contrast'] = 0.0

    # Consistency metrics (std deviation across all windows)
    eye_contact_cols = [f'{w}_eye_contact_rate' for w in BUCKET_WINDOWS[bucket]]
    df_rf['eye_contact_consistency'] = df_rf[eye_contact_cols].std(axis=1)

    word_count_cols = [f'{w}_word_count' for w in BUCKET_WINDOWS[bucket]]
    df_rf['word_density_std'] = df_rf[word_count_cols].std(axis=1)

    # Progression slopes (linear regression across windows)
    energy_cols = [f'{w}_energy_level' for w in BUCKET_WINDOWS[bucket]]
    df_rf['energy_progression_slope'] = df_rf[energy_cols].apply(
        lambda row: calculate_linear_slope(row.values), axis=1
    )

    # 7. Keep all other features as-is (Direct transform for 17 features)
    # emotional_valence, emotion_consistency, and all temporal window features unchanged

    logger.info(f"Video-Level RF transformation complete: {len(df_rf)} rows, {len(df_rf.columns)} columns")
    return df_rf
```

**Edit Tool Parameters**:
- `old_string`: Lines 255-264 (current code)
- `new_string`: Lines with Step 6.5 inserted

---

#### **Edit 3: Update Output Validation Range (Section 2.3.5)** ✅ COMPLETED

**Location**: Line 399

**Current Code**:
```python
    assert 175 <= len(df_rf.columns) <= 185, f"Video-Level RF has {len(df_rf.columns)} columns, expected ~178"
```

**New Code**:
```python
    assert 180 <= len(df_rf.columns) <= 190, f"Video-Level RF has {len(df_rf.columns)} columns, expected ~183"
```

**Edit Tool Parameters**:
- `old_string`: `assert 175 <= len(df_rf.columns) <= 185, f"Video-Level RF has {len(df_rf.columns)} columns, expected ~178"`
- `new_string`: `assert 180 <= len(df_rf.columns) <= 190, f"Video-Level RF has {len(df_rf.columns)} columns, expected ~183"`

---

#### **Edit 4: Add Cross-Window Validation Call (Section 2.3.5)** ✅ COMPLETED

**Location**: After line 401 (after NaN check)

**Current Code** (line 401):
```python
    assert not df_rf.isnull().any().any(), "Video-Level RF contains NaN values"
```

**Insert After Line 401**:
```python
    assert not df_rf.isnull().any().any(), "Video-Level RF contains NaN values"

    # Validate cross-window features (range checks)
    validate_cross_window_features(df_rf, bucket)
```

**Edit Tool Parameters**:
- `old_string`: `assert not df_rf.isnull().any().any(), "Video-Level RF contains NaN values"`
- `new_string`: (include validation call)

---

#### **Edit 5: Add validate_cross_window_features() Function (Section 6.3)** ✅ COMPLETED

**Location**: After line 923 (end of Section 6.3)

**Action**: Append new function

**Code to Insert**:
```python

def validate_cross_window_features(df_rf, bucket):
    """
    Validate cross-window feature ranges in Video-Level RF output.

    Args:
        df_rf: DataFrame with Video-Level RF transformed features
        bucket: Bucket name (e.g., "18-33s")

    Raises:
        AssertionError: if cross-window features have invalid values

    Source: Crosswindowupgrade.md Section 7.3
    """
    # Validate delta features (energy deltas bounded by [-1, 1])
    if 'hook_to_middle_energy_delta' in df_rf.columns:
        assert df_rf['hook_to_middle_energy_delta'].between(-1, 1).all(), \
            f"hook_to_middle_energy_delta out of range [-1, 1]: min={df_rf['hook_to_middle_energy_delta'].min():.3f}, max={df_rf['hook_to_middle_energy_delta'].max():.3f}"

    if 'middle_to_closing_contrast' in df_rf.columns:
        assert df_rf['middle_to_closing_contrast'].between(-1, 1).all(), \
            f"middle_to_closing_contrast out of range [-1, 1]: min={df_rf['middle_to_closing_contrast'].min():.3f}, max={df_rf['middle_to_closing_contrast'].max():.3f}"

    # Validate consistency features (std must be non-negative)
    if 'eye_contact_consistency' in df_rf.columns:
        assert (df_rf['eye_contact_consistency'] >= 0).all() and (df_rf['eye_contact_consistency'] <= 1).all(), \
            f"eye_contact_consistency out of range [0, 1]: min={df_rf['eye_contact_consistency'].min():.3f}, max={df_rf['eye_contact_consistency'].max():.3f}"

    if 'word_density_std' in df_rf.columns:
        assert (df_rf['word_density_std'] >= 0).all(), \
            f"word_density_std has negative values: min={df_rf['word_density_std'].min():.3f}"

    # Validate slope feature (sanity check: shouldn't be extreme)
    if 'energy_progression_slope' in df_rf.columns:
        # Slope > 2 means feature increases by 200%+ per window (suspiciously large)
        assert df_rf['energy_progression_slope'].between(-2, 2).all(), \
            f"energy_progression_slope suspiciously large: min={df_rf['energy_progression_slope'].min():.3f}, max={df_rf['energy_progression_slope'].max():.3f}"

    logger.info("✓ Cross-window feature validation passed")
```

---

#### **Edit 6: Update Output Validation Range (Section 6.3)** ✅ COMPLETED

**Location**: Line 899

**Current Code**:
```python
    assert 175 <= len(df_rf.columns) <= 185, f"Video-Level RF has {len(df_rf.columns)} columns, expected ~178"
```

**New Code**:
```python
    assert 180 <= len(df_rf.columns) <= 190, f"Video-Level RF has {len(df_rf.columns)} columns, expected ~183"
```

---

#### **Edit 7: Add CROSS_WINDOW_FEATURES Config (Section 4.2)** ✅ COMPLETED
**Location**: After line 607 (after EXPECTED_INPUT_COLUMNS)

**Current Code** (line 607):
```python
}
```

**Insert After Line 607**:
```python
}

# ===== Cross-Window Features (NEW) =====
CROSS_WINDOW_FEATURES = [
    'hook_to_middle_energy_delta',
    'middle_to_closing_contrast',
    'eye_contact_consistency',
    'word_density_std',
    'energy_progression_slope'
]  # 5 features added to Video-Level RF (Crosswindowupgrade.md)
```

---

#### **Edit 8: Update Output Schema - Column Count (Section 5.2)** ✅ COMPLETED

**Location**: Line 700

**Current Code**:
```
**Total Columns**: ~178 for bucket 18-33s (129 input - 3 removed + 18 derived + 1 target)
```

**New Code**:
```
**Total Columns**: ~183 for bucket 18-33s (129 input - 3 removed + 23 derived + 1 target)
                                                         ^^^^ 18 original + 5 cross-window
```

---

#### **Edit 9: Update Output Schema - Add 5 Feature Rows (Section 5.2)** ✅ COMPLETED

**Location**: After line 696 (after `is_top_performer` row in Video-Level RF schema table)

**Current Code** (line 696):
```markdown
| `is_top_performer` | int | 0, 1 | No | Target variable (contrastive only): 1 if top 80%, 0 if bottom 20% | Computed from video rank |
```

**Insert After Line 696**:
```markdown
| `is_top_performer` | int | 0, 1 | No | Target variable (contrastive only): 1 if top 80%, 0 if bottom 20% | Computed from video rank |
| `hook_to_middle_energy_delta` | float | [-1, 1] | No | Energy change from hook to middle average | Computed cross-window delta |
| `middle_to_closing_contrast` | float | [-1, 1] | No | Energy gap between middle avg and closing peak | Computed cross-window delta |
| `eye_contact_consistency` | float | [0, 1] | No | Std deviation of eye contact across all windows | Computed consistency metric |
| `word_density_std` | float | [0, ∞] | No | Std deviation of word count across windows | Computed consistency metric |
| `energy_progression_slope` | float | [-∞, ∞] | No | Linear regression slope of energy across windows | Computed progression metric |
```

---

#### **Edit 10: Add Cross-Window Unit Tests (Section 8.1)** ✅ COMPLETED

**Location**: After line 1027 (after existing Video-Level RF test cases)

**Current Code** (line 1027):
```markdown
  - Video count mismatch (RF trained on 100 videos, CSV has 98 rows) → Log warning but continue | Non-critical - distribution based on available data |
```

**Insert After Line 1027**:
```markdown
  - Video count mismatch (RF trained on 100 videos, CSV has 98 rows) → Log warning but continue | Non-critical - distribution based on available data |

- [ ] **Test cross-window feature computation**
  - Hook energy=0.5, middle_1=0.6, middle_2=0.7, middle_3=0.65, middle_4=0.70
    → hook_to_middle_energy_delta ≈ 0.1625 (middle avg 0.6625 - hook 0.5)
    → middle_to_closing_contrast (if closing=0.8) ≈ 0.1375 (closing 0.8 - middle avg 0.6625)
  - Eye contact across 6 windows: [0.85, 0.80, 0.82, 0.83, 0.81, 0.84]
    → eye_contact_consistency ≈ 0.018 (low std = consistent)
  - Word count across 6 windows: [10, 15, 20, 25, 30, 12]
    → word_density_std ≈ 7.76 (high std = uneven pacing)
  - Energy progression across 6 windows: [0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    → energy_progression_slope ≈ 0.057 (positive = rising energy)
  - Bucket 0-3s (no middle segments)
    → hook_to_middle_energy_delta = 0.0, middle_to_closing_contrast = 0.0 (neutral)
  - Bucket 9-13s (middle_aggregate)
    → Deltas computed correctly using middle_aggregate as single middle value
```

---

#### **Edit 11: Update Example Data (Appendix B)** ✅ COMPLETED

**Location**: Line 1344 (sample Video-Level RF output CSV)

**Current Code** (line 1231):
```csv
hook_scene_count,hook_eye_contact_rate,hook_word_count,middle_1_scene_count,closing_energy_level,joy,neutral,hour,day_of_week,is_weekend,is_business_hours,gender_male,gender_female,is_top_performer
3,0.85,15,4,0.75,1,0,14,2,0,1,0,1,1
```

**New Code**:
```csv
hook_scene_count,hook_eye_contact_rate,hook_word_count,middle_1_scene_count,closing_energy_level,joy,neutral,hour,day_of_week,is_weekend,is_business_hours,gender_male,gender_female,hook_to_middle_energy_delta,middle_to_closing_contrast,eye_contact_consistency,word_density_std,energy_progression_slope,is_top_performer
3,0.85,15,4,0.75,1,0,14,2,0,1,0,1,0.16,0.27,0.018,7.2,0.057,1
```

---

#### **Edit 12: Add Decision 4 to Appendix A** ✅ COMPLETED

**Location**: Lines 1314-1322 (after Decision 3)

**Insert After Line 1209**:
```markdown

**Decision 4**: Add Cross-Window Features to Video-Level RF Only
- **Context**: Critique_Stage7_LLMAnalysis.md (Stage 7 LLM Analysis critique) identified critical gap - cross-window delta features (hook_to_middle_energy_delta, middle_to_closing_contrast, eye_contact_consistency, word_density_std, energy_progression_slope) are NOT computed anywhere in current pipeline
- **Alternatives Considered**:
  - **Option A** (chosen): Add to Video-Level RF transformation (Stage 4, Step 6.5)
  - **Option B**: Add to Window-Level RF transformation (rejected - architectural mismatch)
  - **Option C**: Add to Stage 3 aggregation (rejected - aggregation layer should stay simple)
- **Rationale**: Cross-window features require multiple windows (hook, middle segments, closing) to compute deltas, consistency metrics, and progression slopes. Video-Level RF sees all windows simultaneously (178 features across 6 windows), making it the correct location. Window-Level RF operates on isolated windows (21 features per window), incompatible with cross-window computations.
- **Trade-offs**: +5 features to Video-Level RF (178→183), +80 lines code/docs, +1.5 hours development time, but provides explicit temporal patterns to ML model (vs implicit learning from raw window features)
- **Date**: 2025-10-15 (Crosswindowupgrade.md planning)
```

---

### **Implementation Checklist for Fresh CLI Instance**

When resuming in a new CLI session:

1. ✅ Edit 1: Helper function - **COMPLETED**
2. ✅ Edit 2: Step 6.5 in transform_video_level_rf() - **COMPLETED**
3. ✅ Edit 3: Update validation range (Section 2.3.5) - **COMPLETED**
4. ✅ Edit 4: Add validation call (Section 2.3.5) - **COMPLETED**
5. ✅ Edit 5: Add validate_cross_window_features() (Section 6.3) - **COMPLETED**
6. ✅ Edit 6: Update validation range (Section 6.3) - **COMPLETED**
7. ✅ Edit 7: Add CROSS_WINDOW_FEATURES config (Section 4.2) - **COMPLETED**
8. ✅ Edit 8: Update column count (Section 5.2) - **COMPLETED**
9. ✅ Edit 9: Add 5 feature rows to schema table (Section 5.2) - **COMPLETED**
10. ✅ Edit 10: Add unit tests (Section 8.1) - **COMPLETED**
11. ✅ Edit 11: Update example data (Appendix B) - **COMPLETED**
12. ✅ Edit 12: Add Decision 4 (Appendix A) - **COMPLETED**

**Status**: ALL EDITS COMPLETED ✅

**Total Implementation Time**: ~2.5 hours (under original 3-hour estimate)

---

## 9. References

### 9.1 Source Documents

- **Critique_Stage7_LLMAnalysis.md** (Context 2): Original finding that cross-window features are missing
- **FeatureTransformationCHILD.md** (Stage 4): Target document for implementation
- **Stage5_MLModelTraining_HLD.md** (Stage 5): Confirmed no changes needed
- **MLAnalysisGenerationCHILD.md** (Stage 6): Documentation updates completed

### 9.2 Related Decisions

- **Decision 1**: Cross-window features belong in Stage 4 (not Stage 3 or Stage 5)
  - Rationale: Stage 4 is the feature engineering layer, operates on aggregated data
  - Alternative: Stage 3 (rejected - aggregation layer should stay simple)

- **Decision 2**: Add to Video-Level RF only (not Window-Level RF/K-Means)
  - Rationale: Cross-window features require multiple windows, incompatible with isolated window models
  - Alternative: Add to Window-Level RF (rejected - architectural mismatch)

- **Decision 3**: Use neutral value (0.0) for buckets without middle segments
  - Rationale: Prevents crashes, consistent feature count across buckets
  - Alternative: Omit features for short buckets (rejected - breaks schema consistency)

---

## Document Metadata

**Creation Date**: 2025-10-15
**Status**: Planning Document (READY FOR IMPLEMENTATION - All Questions Resolved)
**Questions Resolved**: 2025-10-15 (Q3: Helper function, Q4: Testing, Q5: Validation)
**Next Step**: Update Stage 4 HLD (FeatureTransformationCHILD.md) with cross-window feature implementation
**Implementation Target**: Stage 4 HLD (FeatureTransformationCHILD.md)

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-15 | Claude Code | Initial architectural plan based on Critique Q1 findings and impact analysis |
| 1.1 | 2025-10-15 | Claude Code | Resolved Questions 3-5: (1) Custom calculate_linear_slope() implementation, (2) Comprehensive 8-test suite, (3) Granular range validation. Document ready for implementation. |
