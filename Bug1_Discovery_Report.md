# Bug #1 Discovery Report - Boolean Quantile Computation

**Date**: 2025-10-23
**Investigation by**: Claude CLI
**Bug Location**: `ml_pipeline/stage6_analysis/ml_analysis_generation.py:243`
**Function**: `generate_video_rf_json()`

---

## Executive Summary

Bug #1 is caused by attempting to compute percentile statistics (`.quantile()`) on **boolean data types**, which fails in NumPy 1.26.4. The issue affects **2 of 3 test buckets** (bucket_18-33s and bucket_13-18s) because their RandomForest models ranked `closing_has_captions` (a boolean feature) in the **top 10 most important features**.

**Root Cause**: Missing data type validation before quantile computation
**Impact**: Critical - blocks 2/3 buckets from completing Stage 6
**Affected Buckets**: bucket_18-33s (rank #6), bucket_13-18s (rank #10)
**Unaffected Bucket**: bucket_60-90s (no boolean in top 10)

---

## Discovery Timeline

### 1. Initial Error

```
TypeError: numpy boolean subtract, the `-` operator, is not supported,
use the bitwise_xor, the `^` operator, or the logical_xor function instead.
```

**Location**: Line 243 in `generate_video_rf_json()`
```python
high_threshold = float(top_performers.quantile(HIGH_PERCENTILE))
```

### 2. Investigation Findings

#### Finding #1: Pandas Considers Boolean as "Numeric"

```python
col = df['closing_has_captions']
print(pd.api.types.is_numeric_dtype(col))  # → True (misleading!)
print(col.dtype)                            # → bool
```

**Implication**: Standard pandas numeric checks (`is_numeric_dtype()`) return `True` for boolean columns, so they cannot be used to filter out problematic data types.

#### Finding #2: NumPy Quantile Fails on Boolean

```python
top_performers = df[df['is_top_performer'] == 1]['closing_has_captions']
top_performers.quantile(0.66)  # ❌ FAILS with TypeError
```

**Environment**:
- Pandas: 2.3.3
- NumPy: 1.26.4

**Why it fails**: NumPy's percentile implementation uses the `-` operator internally, which is not supported for boolean arrays in NumPy 1.26.4.

#### Finding #3: Boolean Features in Top 10 (Bucket Analysis)

| Bucket | Boolean in Top 10? | Feature Name | Rank | Importance |
|--------|-------------------|--------------|------|------------|
| **bucket_18-33s** | ✅ YES | `closing_has_captions` | #6 | 0.026319 |
| **bucket_13-18s** | ✅ YES | `closing_has_captions` | #10 | 0.022280 |
| **bucket_60-90s** | ❌ NO | — | — | — |

**Why bucket_60-90s passed Bug #1**:
The RandomForest model for this bucket ranked `emotional_valence` features (float64) highest, pushing boolean features outside the top 10. It then failed on Bug #2 instead.

#### Finding #4: All Boolean Features in Dataset

There are **6 boolean features** total across all buckets:

| Feature | bucket_18-33s Rank | bucket_13-18s Rank | bucket_60-90s Rank |
|---------|-------------------|-------------------|-------------------|
| `closing_has_captions` | #6 (0.026) | #10 (0.022) | #11 (0.019) |
| `middle_2_has_captions` | #16 (0.019) | #12 (0.022) | #26 (0.010) |
| `hook_has_captions` | #26 (0.012) | #21 (0.015) | #40 (0.008) |
| `middle_1_has_captions` | #41 (0.007) | #40 (0.006) | #61 (0.005) |
| `middle_3_has_captions` | #89 (0.002) | — | #48 (0.007) |
| `middle_4_has_captions` | #143 (0.000) | — | #99 (0.002) |

**Pattern**: `closing_has_captions` is consistently the highest-ranked boolean feature, appearing in the top 10-11 across all buckets.

---

## Design Context Discovery

### The HLD Document Says...

From `MLAnalysisGenerationCHILD.md` lines 240-243:

```python
# DESIGN DECISION: Video-level uses aggregated_features.csv (Stage 3) not rf_transformed.csv (Stage 4)
# Rationale: Video-level RF includes cross-window features (e.g., hook_to_middle_energy_delta)
# computed from raw aggregated values. Distribution percentiles must match raw data source.
# See CrossHLDalignment2do.md Issue #5 for full rationale.
```

### What This Means

**Intentional Design**: Using `aggregated_features.csv` instead of `rf_transformed.csv` is deliberate.

**Rationale**: Cross-window derived features (like `word_density_std`) are computed from raw aggregated values, so distribution analysis should use the source data.

**The Problem**: This design assumes all features in `aggregated_features.csv` are numeric and can have quantiles computed. **This assumption is false** - there are 6 boolean columns.

---

## CSV Comparison Analysis

### aggregated_features.csv (Stage 3 output)
- **Rows**: 47 (videos)
- **Columns**: 129
- **Boolean columns**: 6 (`*_has_captions`)
- **Object columns**: 2 (`create_time`, `gender`)
- **Has `is_top_performer`**: ❌ NO (computed on-the-fly in Bug #1 code)

### rf_transformed.csv (Stage 4 output)
- **Rows**: 47 (videos)
- **Columns**: 147
- **Boolean columns**: 6 (same as aggregated)
- **Object columns**: 0 (encoded to `gender_female`, `gender_male`, `gender_nan`)
- **Has `is_top_performer`**: ✅ YES (from Stage 4)
- **Additional features**: 20 derived features (e.g., `word_density_std`, `month`, `day_of_week`)

### Feature Presence Check

| Feature | aggregated_features.csv | rf_transformed.csv |
|---------|------------------------|--------------------|
| `word_density_std` (Rank #5) | ❌ NO | ✅ YES |
| `closing_has_captions` (Rank #6) | ✅ YES (bool) | ✅ YES (bool) |
| `gender_female` | ❌ NO | ✅ YES |
| `month` (Rank #14) | ❌ NO | ✅ YES |

**Conclusion**:
- `rf_transformed.csv` has **all 145 features** the model was trained on
- `aggregated_features.csv` is missing **16 features** (derived/encoded)
- **Both CSVs have the 6 boolean features** (not encoded/transformed)

---

## Why Boolean Features Rank High

### Feature: `closing_has_captions`

**What it measures**: Whether captions/text overlays are present in the last 3 seconds (closing window).

**Why RF ranks it highly**:
1. **Discriminative power**: Videos with captions in closing likely correlate with higher engagement
2. **CTA signal**: Captions in closing often contain calls-to-action ("Follow for more!", "Link in bio")
3. **Binary clarity**: Unlike continuous features, boolean features provide clean splits for decision trees

**Distribution** (bucket_18-33s):
```
closing_has_captions:
  False: 34 videos (72%)
  True:  13 videos (28%)
```

**Interpretation**: 28% of videos use captions in closing. If these videos correlate with top performers, the RF model will rank this feature highly.

---

## Impact on Different Buckets

### bucket_18-33s (18-33 second videos)
- **Top 10 Features**: Rank #6 = `closing_has_captions`
- **Trigger Bug #1**: ✅ YES (attempts quantile on boolean)
- **Missing in CSV**: 1 feature (`word_density_std` at rank #5)
- **Expected behavior**: Fail on rank #6 feature

### bucket_13-18s (13-18 second videos)
- **Top 10 Features**: Rank #10 = `closing_has_captions`
- **Trigger Bug #1**: ✅ YES (attempts quantile on boolean)
- **Missing in CSV**: 1 feature (`word_density_std` at rank #13)
- **Expected behavior**: Fail on rank #10 feature

### bucket_60-90s (60-90 second videos)
- **Top 10 Features**: All numeric (emotional_valence features dominate)
- **Trigger Bug #1**: ❌ NO (no boolean in top 10)
- **Missing in CSV**: 0 features in top 20
- **Expected behavior**: **Pass Bug #1**, fail on Bug #2 instead

**Why is this bucket different?**

Longer videos (60-90s) have more time for emotional storytelling. The RF model identified that `middle_X_emotional_valence` features (positions 1-6 in top 10) are strong predictors, pushing boolean features out of the top 10.

---

## Code Analysis

### Current Implementation (Buggy)

```python
# Line 235-244 in generate_video_rf_json()
top_performers = df[df['is_top_performer'] == 1][feature_name]
bottom_performers = df[df['is_top_performer'] == 0][feature_name]

top_avg = float(top_performers.mean())
bottom_avg = float(bottom_performers.mean())
gap = abs(top_avg - bottom_avg)

# ❌ FAILS HERE if feature is boolean
high_threshold = float(top_performers.quantile(HIGH_PERCENTILE))
low_threshold = float(top_performers.quantile(LOW_PERCENTILE))
```

### What Happens

1. `feature_name = 'closing_has_captions'` (boolean)
2. `top_performers` is a pandas Series with dtype `bool`
3. `.mean()` works fine (converts bool to 0/1, computes average)
4. `.quantile(0.66)` **fails** because NumPy can't compute percentiles on boolean

### Edge Case Handling (Existing)

```python
# Line 257 - This handles MISSING features but not BOOLEAN features
if feature_name not in df.columns:
    # Feature not in CSV (e.g., derived features)
    feature_data['top_performer_avg'] = None
    feature_data['bottom_performer_avg'] = None
    feature_data['gap'] = None
    feature_data['distribution'] = None
    continue
```

**What's handled**: Missing features (like `word_density_std`)
**What's NOT handled**: Boolean features (like `closing_has_captions`)

---

## Root Cause Summary

### Primary Root Cause
**Missing data type validation** before quantile computation.

The code assumes all features in `aggregated_features.csv` are numeric with continuous distributions. This is false for:
1. **Boolean features**: 6 columns (`*_has_captions`)
2. **Object features**: 2 columns (`create_time`, `gender`) - but these are never in top 10

### Secondary Issue (Design Flaw)
Using `aggregated_features.csv` instead of `rf_transformed.csv` creates a mismatch:
- Model was trained on 145 features (rf_transformed.csv)
- Distribution analysis uses 129 features (aggregated_features.csv)
- 16 features are missing and get `distribution: None`

**However**, the HLD document explicitly justifies this design decision, so it's not a "bug" per se, but an **incomplete implementation** of that design.

---

## Why This Bug Matters

### Impact on LLM Analysis (Stage 7)

If Bug #1 is fixed but boolean features still lack distribution data, Stage 7 LLM prompts will receive:

```json
{
  "feature": "closing_has_captions",
  "importance": 0.026319,
  "top_performer_avg": null,
  "bottom_performer_avg": null,
  "gap": null,
  "distribution": null
}
```

**Problem**: The LLM cannot generate actionable insights like:
- "80% of top performers use captions in closing vs 20% of bottom performers"
- "Videos with closing captions have 4x higher viral rate"

### What SHOULD Be Computed (Boolean Distribution)

For boolean features, distribution analysis should compute **percentage breakdowns**:

```json
{
  "feature": "closing_has_captions",
  "importance": 0.026319,
  "distribution": {
    "top_performers": {
      "true_percentage": 0.65,   // 65% of top performers have captions
      "false_percentage": 0.35
    },
    "bottom_performers": {
      "true_percentage": 0.20,   // 20% of bottom performers have captions
      "false_percentage": 0.80
    },
    "gap": 0.45  // 45 percentage point difference
  }
}
```

This gives the LLM clear, actionable data: "Use captions in closing - 65% of top performers do vs only 20% of bottom performers."

---

## Fix Strategies

### Strategy 1: Add Boolean Check (Minimal Fix)
**Complexity**: Low
**Impact**: Prevents crash, but boolean features get `distribution: null`

```python
# Before line 243, add:
if pd.api.types.is_bool_dtype(top_performers):
    # Skip quantile for boolean features
    feature_data['distribution'] = None
    logger.debug(f"Skipping distribution for boolean feature: {feature_name}")
    continue
```

**Pros**:
- Quick fix (5 minutes)
- Prevents crash
- Minimal code change

**Cons**:
- Boolean features lose distribution insights
- LLM gets incomplete data for important features (closing_has_captions is rank #6!)

---

### Strategy 2: Compute Boolean-Specific Distribution (Recommended)
**Complexity**: Medium
**Impact**: Provides rich distribution data for boolean features

```python
# Before line 243, add:
if pd.api.types.is_bool_dtype(top_performers):
    # Compute boolean-specific distribution
    top_true_pct = (top_performers == True).sum() / len(top_performers)
    bottom_true_pct = (bottom_performers == True).sum() / len(bottom_performers)

    feature_data['distribution'] = {
        'type': 'boolean',
        'top_performers': {
            'true_percentage': float(top_true_pct),
            'false_percentage': float(1 - top_true_pct)
        },
        'bottom_performers': {
            'true_percentage': float(bottom_true_pct),
            'false_percentage': float(1 - bottom_true_pct)
        },
        'gap': float(abs(top_true_pct - bottom_true_pct))
    }
    logger.debug(f"Computed boolean distribution for {feature_name}")
    continue  # Skip numeric quantile logic
```

**Pros**:
- Provides actionable insights for boolean features
- LLM can generate specific recommendations
- Matches the importance of these features (rank #6!)

**Cons**:
- Slightly more code
- Different JSON schema for boolean vs numeric features

---

### Strategy 3: Use rf_transformed.csv Instead (Architectural Change)
**Complexity**: High
**Impact**: Fixes missing features, but doesn't solve boolean issue

**Not recommended** because:
1. Contradicts documented HLD design decision
2. Requires updating documentation
3. Still doesn't fix boolean quantile issue
4. Boolean features exist in BOTH CSVs

---

## Recommendation

**Use Strategy 2: Boolean-Specific Distribution**

**Rationale**:
1. `closing_has_captions` is **rank #6** in bucket_18-33s - this is a critical feature
2. Boolean distribution provides **actionable insights** for content creators
3. Minimal complexity increase (20 lines of code)
4. Aligns with the goal of Stage 6: "LLM-consumable insights"

**Additional**: Add object type check as well (for `create_time`, `gender` if they ever appear in top 10):

```python
if pd.api.types.is_bool_dtype(top_performers):
    # Boolean distribution logic...
elif pd.api.types.is_object_dtype(top_performers) or pd.api.types.is_string_dtype(top_performers):
    # Skip object features (can't compute distributions)
    feature_data['distribution'] = None
    continue
```

---

## Test Plan (After Fix)

### 1. Unit Test: Boolean Quantile
```python
def test_boolean_feature_distribution():
    """Test that boolean features get proper distribution stats."""
    # Setup mock data with closing_has_captions
    # Run generate_video_rf_json()
    # Assert distribution has 'type': 'boolean'
    # Assert top_performers.true_percentage exists
```

### 2. Integration Test: All Buckets
```bash
# Re-run Stage 6 for all buckets after fix
python stage6_test.py

# Expected:
# - bucket_18-33s: ✅ PASS (13 JSONs)
# - bucket_13-18s: ✅ PASS (7 JSONs)
# - bucket_60-90s: ❌ FAIL (Bug #2 - different issue)
```

### 3. Validation: JSON Schema
```python
# Check closing_has_captions in output JSON
with open('bucket_18-33s/ml_analysis/rf_video_analysis.json') as f:
    data = json.load(f)

closing_feat = next(f for f in data['feature_importance'] if f['feature'] == 'closing_has_captions')

assert closing_feat['distribution'] is not None
assert closing_feat['distribution']['type'] == 'boolean'
assert 0 <= closing_feat['distribution']['top_performers']['true_percentage'] <= 1
```

---

## Appendix: Full Feature Rankings

### bucket_18-33s Top 20 Features

| Rank | Feature | Importance | Data Type | In CSV? |
|------|---------|------------|-----------|---------|
| 1 | middle_2_gesture_count | 0.0474 | int64 | ✅ |
| 2 | middle_4_word_count | 0.0457 | int64 | ✅ |
| 3 | closing_word_count | 0.0386 | int64 | ✅ |
| 4 | middle_3_word_count | 0.0364 | int64 | ✅ |
| 5 | word_density_std | 0.0272 | — | ❌ MISSING |
| 6 | **closing_has_captions** | **0.0263** | **bool** | ✅ **BUG TRIGGER** |
| 7 | closing_speech_coverage | 0.0263 | float64 | ✅ |
| 8 | middle_1_word_count | 0.0259 | int64 | ✅ |
| 9 | middle_1_gesture_count | 0.0254 | int64 | ✅ |
| 10 | middle_4_speech_coverage | 0.0252 | float64 | ✅ |

### bucket_13-18s Top 20 Features

| Rank | Feature | Importance | Data Type | In CSV? |
|------|---------|------------|-----------|---------|
| 1 | closing_pitch_scatter_ratio | 0.0927 | float64 | ✅ |
| 2 | hook_overlay_unique_count | 0.0710 | int64 | ✅ |
| 3 | hook_word_count | 0.0667 | int64 | ✅ |
| 4 | closing_overlay_unique_count | 0.0480 | int64 | ✅ |
| 5 | closing_gaze_variance | 0.0464 | float64 | ✅ |
| 6 | closing_scene_count | 0.0372 | int64 | ✅ |
| 7 | middle_aggregate_average_face_size | 0.0363 | float64 | ✅ |
| 8 | middle_aggregate_scene_duration_variance | 0.0359 | float64 | ✅ |
| 9 | middle_aggregate_pitch_scatter_ratio | 0.0340 | float64 | ✅ |
| 10 | **closing_has_captions** | **0.0223** | **bool** | ✅ **BUG TRIGGER** |

---

## Conclusion

Bug #1 is a **critical but fixable issue** caused by missing data type validation. The recommended fix (Strategy 2) not only prevents the crash but also provides richer insights for boolean features, which are important predictors in the RF models (rank #6 and #10).

**Estimated time to fix**: 30 minutes (implementation + testing)
**Estimated time to test**: 5 minutes (re-run Stage 6 for all buckets)
