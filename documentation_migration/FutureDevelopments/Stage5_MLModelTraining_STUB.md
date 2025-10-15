# Stage 5: ML Model Training - High-Level Design (STUB)

> **Status**: STUB - Ready for Phase 2 expansion
> **Parent**: MLPlanningv2.md - Stage 5: ML Model Training (Lines 1588-1945)
> **Critical Dependencies**: Stage5Tests.md (testing spec), Critique_MLModelTraining.md (business critique)
> **Date Created**: 2025-10-14

---

## ⚠️ FOR FUTURE CLI INSTANCE: READ THIS FIRST

This is a STUB document created during Phase 1 Business Critique.

**CRITICAL sections have been pre-filled** based on bug analysis. When you expand this into a full HLD (Phase 2/3):

1. ✅ **Section 3: Critical Implementation Warnings** is COMPLETE (do NOT delete or modify)
2. ⏸️ **Other sections** need to be filled in (see TODOs below)
3. 🔗 **Reference docs**:
   - `Stage5Tests.md` (comprehensive testing requirements)
   - `Critique_MLModelTraining.md` (Phase 1 Q&A decisions, especially Q3 validation protocol)

**DO NOT start from scratch!** This stub contains critical bug prevention logic that MUST be preserved.

---

## Document Structure (TODO: Fill in Phases 2-3)

### 1. Context & Business Goal
[TODO: Fill in Phase 2 - Why does Stage 5 exist? What problem does it solve?]

### 2. Architecture & Design
[TODO: Fill in Phase 2 - 90 model architecture, training flow, data dependencies]

---

## 3. Critical Implementation Warnings ✅ COMPLETE

> ⚠️ **READ THIS BEFORE IMPLEMENTING**: This section documents HIGH-RISK components
> identified during Phase 1 Business Critique (2025-10-14) and testing analysis.
>
> **Failure to handle these correctly WILL result in bugs.**
>
> Each warning includes:
> - Bug risk level (CRITICAL/HIGH/MEDIUM/LOW)
> - Problem description
> - Correct solution with code examples
> - Testing requirements

---

### 🔴 CRITICAL: Feature Name Mismatch (Priority P0)

**Bug Risk**: **GUARANTEED** if not handled (100% probability)

**Problem**:
Stage 4 creates different feature naming conventions for K-Means vs RF models:

- **K-Means features** (from `hook_km_transformed.csv`):
  - Have transformation suffixes: `eye_contact_rate_scaled`, `scene_count_scaled`, `has_captions_encoded`
  - Total: 39 features per window (all numerical, scaled [0-1])

- **RF features** (from `hook_rf_transformed.csv`):
  - NO suffixes: `eye_contact_rate`, `scene_count`, `has_captions`
  - Total: 22 features per window (21 base + 1 target)

**Impact**:
Without normalization, overlap calculation returns **0/5 features** (no matches):
```python
kmeans_top5 = ['eye_contact_rate_scaled', 'scene_count_scaled', ...]
rf_top5 = ['eye_contact_rate', 'scene_count', ...]

overlap = set(kmeans_top5) & set(rf_top5)  # → EMPTY SET!
len(overlap)  # → 0 (BROKEN VALIDATION)
```

This breaks the entire validation system → all clusters marked EXPLORATORY.

**Solution**: Implement feature name normalization

```python
def normalize_feature_name(feature_name):
    """
    Normalize K-Means feature names for comparison with RF feature names.

    Removes K-Means transformation suffixes from Stage 4:
    - '_scaled' (from MinMax scaling)
    - '_log' (from log transformation - intermediate, usually removed)
    - '_encoded' (from label encoding)

    Args:
        feature_name: str, e.g., 'eye_contact_rate_scaled'

    Returns:
        str, e.g., 'eye_contact_rate'

    Examples:
        >>> normalize_feature_name('eye_contact_rate_scaled')
        'eye_contact_rate'
        >>> normalize_feature_name('has_captions_encoded')
        'has_captions'
        >>> normalize_feature_name('scene_count')  # Already normalized
        'scene_count'
    """
    normalized = feature_name

    # Remove suffixes in order (some features may have multiple)
    suffixes = ['_scaled', '_log', '_encoded']
    for suffix in suffixes:
        normalized = normalized.replace(suffix, '')

    return normalized


# Usage in validation
kmeans_top5 = get_top_cluster_features(kmeans_model, feature_names, n=5)
rf_top5 = get_top_rf_features(rf_model, n=5)

# Normalize K-Means features before comparison
kmeans_normalized = [normalize_feature_name(f) for f in kmeans_top5]

# Now overlap works!
overlap = set(kmeans_normalized) & set(rf_top5)
overlap_count = len(overlap)  # Should be 2-5 (not 0)
```

**Testing Requirements**:
- **Unit Test**: `tests/unit/test_feature_normalization.py` (MUST pass 100%)
- **Integration Test**: `tests/integration/test_stage4_to_stage5_integration.py::test_feature_name_overlap_with_real_data`
  - Uses REAL Stage 4 output files
  - Validates overlap ≥ 15/21 features

**Test Status**: ✅ Comprehensive tests written in Stage5Tests.md Test #2

**Reference**:
- Stage5Tests.md Test #2 (complete test suite)
- Critique_MLModelTraining.md Q&A (bug discovery)

---

### 🔴 HIGH: K-Means Feature Ranking Logic (Priority P0)

**Bug Risk**: **HIGH** (60% testable - conceptually complex, easy to implement wrong)

**Problem**:
Determining which features "define" a K-Means cluster is NOT straightforward. There are multiple plausible but WRONG approaches.

**Wrong approaches** (do NOT use):

**❌ Wrong Approach #1: Ranking by centroid magnitude**
```python
# WRONG: Measures absolute values, not distinctiveness
cluster_centroid = kmeans_model.cluster_centers_[cluster_id]
top_features = np.argsort(cluster_centroid)[::-1][:5]
# Problem: Feature with high value in ALL clusters ranks high, but it doesn't distinguish clusters
```

**❌ Wrong Approach #2: Ranking by distance from global mean**
```python
# WRONG: Measures outliers, not cluster definition
cluster_centroid = centroids[cluster_id]
mean_centroid = centroids.mean(axis=0)
feature_deviations = np.abs(cluster_centroid - mean_centroid)
top_features = np.argsort(feature_deviations)[::-1][:5]
# Problem: Measures THIS cluster vs mean, not what distinguishes ALL clusters
```

**❌ Wrong Approach #3: Using RF feature importance**
```python
# WRONG: K-Means and RF answer different questions
top_features = rf_model.feature_importances_.argsort()[::-1][:5]
# Problem: RF importance is for PREDICTION, K-Means is for CLUSTERING
```

**Correct approach** (use THIS):
✅ Features with highest VARIANCE across all cluster centroids

**Rationale**:
- If Feature A varies greatly between clusters:
  - Cluster 1 centroid: 0.9 (high)
  - Cluster 2 centroid: 0.3 (low)
  - Cluster 3 centroid: 0.6 (medium)
  - Variance: 0.09 (high) → Feature DEFINES clusters

- If Feature B is constant across clusters:
  - Cluster 1 centroid: 0.5
  - Cluster 2 centroid: 0.5
  - Cluster 3 centroid: 0.5
  - Variance: 0.0 → Feature does NOT distinguish clusters

**Implementation**:

```python
import numpy as np


def get_top_cluster_features(kmeans_model, feature_names, n=5):
    """
    Extract top N cluster-defining features from K-Means model.

    Uses variance across cluster centroids as the ranking metric.
    Features with high variance distinguish clusters; features with
    low variance do not.

    Args:
        kmeans_model: Trained sklearn.cluster.KMeans object
        feature_names: List of feature names (must match order of centroids)
        n: Number of top features to return (default 5)

    Returns:
        List of top N feature names, sorted by importance (highest variance first)

    Example:
        >>> kmeans = KMeans(n_clusters=3, random_state=42)
        >>> kmeans.fit(X)  # X shape: (100, 39)
        >>> feature_names = ['eye_contact_rate_scaled', 'scene_count_scaled', ...]
        >>> top_5 = get_top_cluster_features(kmeans, feature_names, n=5)
        >>> print(top_5)
        ['eye_contact_rate_scaled', 'scene_count_scaled', 'energy_level_scaled', ...]
    """
    # Get cluster centroids
    centroids = kmeans_model.cluster_centers_  # Shape: (n_clusters, n_features)

    # Calculate variance of each feature across centroids
    # High variance = feature values differ across clusters = cluster-defining
    feature_variances = np.var(centroids, axis=0)  # Shape: (n_features,)

    # Rank features by variance (highest first)
    top_indices = np.argsort(feature_variances)[::-1][:n]

    # Map indices to feature names
    top_features = [feature_names[i] for i in top_indices]

    return top_features


# Usage
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

feature_names = list(X.columns)  # From hook_km_transformed.csv
top_5 = get_top_cluster_features(kmeans, feature_names, n=5)

print(f"Top 5 cluster-defining features: {top_5}")
# Output: ['eye_contact_rate_scaled', 'scene_count_scaled', 'energy_level_scaled', ...]
```

**Edge Cases**:

1. **All features have same variance** (rare but possible):
   - Ranking order is arbitrary
   - Should not crash
   - Test: `tests/unit/test_kmeans_feature_ranking.py::test_edge_case_all_features_same_variance`

2. **n > number of features**:
   - Should return all features (no error)
   - Test: Handle gracefully with `top_indices = np.argsort(feature_variances)[::-1][:min(n, len(feature_names))]`

**Testing Requirements**:
- **Unit Tests** (60% coverage): `tests/unit/test_kmeans_feature_ranking.py`
  - ✅ Can validate: High-variance features rank above low-variance features
  - ✅ Can validate: Correct number of features returned
  - ⚠️ Cannot fully validate: Conceptual correctness (requires human judgment)

- **Manual Validation** (REQUIRED on first production run):
  - See Stage5Tests.md Layer 3: Manual Validation Checklist
  - Human reviews top 5 features and confirms they "make sense"
  - Questions to ask: "Do these features visually distinguish clusters?" "Are they intuitive?"

**Test Status**:
- ✅ Unit tests written (Stage5Tests.md Test #3)
- ⏸️ Manual validation checklist provided (Stage5Tests.md Layer 3)

**Reference**:
- Stage5Tests.md Test #3 (unit test suite)
- Critique_MLModelTraining.md Q&A (logic discussion)
- Feature extraction logic analysis (Phase 1 critique)

---

### 🟡 MEDIUM: Statistical Test Baseline (Priority P1)

**Bug Risk**: MEDIUM (easy to use wrong baseline if not careful)

**Problem**:
In contrastive mode, the input dataset has 80 top performers + 20 bottom performers.
This means the **baseline success rate is 80%, not 50%**.

If K-Means finds NOTHING (random clustering), each cluster would still have ~80% success rate.

**Wrong baseline** (do NOT use):
```python
# WRONG: Using 50% baseline (assumes balanced dataset)
p_value = binomtest(n_success, n_total, 0.50, alternative='greater')

# Example: Cluster with 27/33 (82%) success rate
p_value = binomtest(27, 33, 0.50, alternative='greater')  # p ≈ 0.0001 (highly significant!)
# WRONG: 82% appears highly significant, but it's only 2% above baseline (80%)
```

**Correct baseline** (use THIS):
```python
# CORRECT: Using 80% baseline (matches input distribution)
p_value = binomtest(n_success, n_total, 0.80, alternative='greater')

# Example: Cluster with 27/33 (82%) success rate
p_value = binomtest(27, 33, 0.80, alternative='greater')  # p ≈ 0.40 (NOT significant)
# CORRECT: 82% is only 2% above 80% baseline → not statistically different from random
```

**Impact of wrong baseline**:
| Success Rate | Wrong (p vs 50%) | Correct (p vs 80%) | Interpretation |
|--------------|------------------|--------------------|-----------------|
| 82% (27/33) | p=0.0001 (significant!) | p=0.40 (not significant) | Wrong baseline creates false confidence |
| 88% (29/33) | p<0.0001 (significant) | p=0.04 (significant) | Both agree, but correct baseline is honest |
| 97% (32/33) | p<0.0001 (significant) | p=0.002 (significant) | Both agree, very strong pattern |

**Implementation**:
```python
from scipy.stats import binomtest


def validate_cluster_statistical(cluster_videos, baseline_rate=0.80):
    """
    Validate cluster using binomial test against contrastive baseline.

    H0 (null hypothesis): Cluster success rate = baseline (80%)
    H1 (alternative): Cluster success rate > baseline

    Args:
        cluster_videos: DataFrame with 'is_top_performer' column
        baseline_rate: float, expected baseline (default 0.80 for contrastive mode)

    Returns:
        tuple: (validation_status, success_rate, p_value)

    Example:
        >>> cluster = df[df['cluster_id'] == 0]
        >>> status, rate, p = validate_cluster_statistical(cluster)
        >>> print(f"{status}: {rate:.1%} success rate (p={p:.4f})")
        'STATISTICALLY VALIDATED': 94.0% success rate (p=0.0021)
    """
    n_total = len(cluster_videos)
    n_success = (cluster_videos['is_top_performer'] == 1).sum()
    success_rate = n_success / n_total

    # Binomial test: Is success rate significantly ABOVE baseline?
    p_value = binomtest(n_success, n_total, baseline_rate, alternative='greater').pvalue

    # Tier assignment based on p-value
    if p_value < 0.05:
        validation_status = "STATISTICALLY VALIDATED (HIGH CONFIDENCE)"
    elif p_value < 0.10:
        validation_status = "MODERATELY VALIDATED (MODERATE CONFIDENCE)"
    else:
        validation_status = "EXPLORATORY (LOW CONFIDENCE)"

    return validation_status, success_rate, p_value
```

**Testing Requirements**:
- **Unit Test**: `tests/unit/test_binomial_test.py`
  - `test_binomial_test_at_baseline()` - Verifies 82% is NOT significant vs 80%
  - `test_binomial_test_significantly_above_baseline()` - Verifies 97% IS significant

**Test Status**: ✅ Complete test suite in Stage5Tests.md Test #1

**Reference**:
- Stage5Tests.md Test #1
- Critique_MLModelTraining.md "Critical Flaw #1" from ultrathink analysis

---

### 🟡 MEDIUM: Silhouette Score Requires Correct X Matrix (Priority P1)

**Bug Risk**: MEDIUM (data passing complexity)

**Problem**:
Silhouette score calculation requires the EXACT feature matrix X that K-Means was trained on.

```python
from sklearn.metrics import silhouette_samples

# Silhouette calculation
silhouette_scores = silhouette_samples(X, kmeans_model.labels_)
#                                      ^^^ MUST be same X used in training!
```

**Common mistakes**:

1. **Wrong feature order** (columns shuffled):
   ```python
   # Training
   X_train = df[['eye_contact_rate', 'scene_count', 'word_count']]
   kmeans.fit(X_train)

   # Validation (WRONG ORDER!)
   X_val = df[['scene_count', 'eye_contact_rate', 'word_count']]
   silhouette = silhouette_samples(X_val, kmeans.labels_)  # WRONG! Meaningless scores
   ```

2. **Wrong scaling** (X vs X_scaled):
   ```python
   # Training
   X_scaled = scaler.fit_transform(X)
   kmeans.fit(X_scaled)

   # Validation (WRONG SCALING!)
   silhouette = silhouette_samples(X, kmeans.labels_)  # WRONG! Used unscaled X
   ```

3. **Subset of features**:
   ```python
   # Training (39 features)
   kmeans.fit(X)  # X has 39 columns

   # Validation (WRONG FEATURES!)
   X_subset = X[['eye_contact_rate', 'scene_count']]  # Only 2 columns
   silhouette = silhouette_samples(X_subset, kmeans.labels_)  # CRASH or wrong results
   ```

**Solution**: Store X alongside kmeans_model during training

```python
import joblib


# ===== During Training (Stage 5) =====
# Load K-Means transformed data from Stage 4
X = pd.read_csv('bucket_18-33s/ml_analysis/hook_km_transformed.csv')

# Train K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Save BOTH model and X matrix
model_path = 'bucket_18-33s/models/hook_kmeans_18-33s.pkl'
X_path = 'bucket_18-33s/models/hook_X_data_18-33s.pkl'

joblib.dump(kmeans, model_path)
joblib.dump(X, X_path)  # Also save X for validation!


# ===== During Validation =====
# Load BOTH model and X
kmeans = joblib.load('bucket_18-33s/models/hook_kmeans_18-33s.pkl')
X = joblib.load('bucket_18-33s/models/hook_X_data_18-33s.pkl')

# Now silhouette calculation is correct (same X used in training)
from sklearn.metrics import silhouette_samples

silhouette_scores = silhouette_samples(X, kmeans.labels_)
cluster_0_silhouette = silhouette_scores[kmeans.labels_ == 0].mean()

print(f"Cluster 0 silhouette: {cluster_0_silhouette:.3f}")
```

**File Structure**:
```
bucket_18-33s/
└── models/
    ├── hook_kmeans_18-33s.pkl          # K-Means model
    ├── hook_X_data_18-33s.pkl          # Feature matrix X (for silhouette)
    ├── hook_rf_18-33s.pkl              # Window-level RF model
    └── rf_video_18-33s.pkl             # Video-level RF model
```

**Storage Impact**: ~50 KB per X file (100 videos × 39 features × 8 bytes) = 410 KB total for all windows

**Testing Requirements**:
- **Unit Test**: `tests/unit/test_silhouette_score.py`
  - Tests well-separated vs overlapping clusters
  - Tests single-element cluster edge case

**Test Status**: ✅ Complete test suite in Stage5Tests.md Test #5

**Reference**: Stage5Tests.md Test #5

---

### 🟢 LOW: Confidence Scoring Threshold Tuning (Priority P2)

**Bug Risk**: LOW (arithmetic is simple, but thresholds are somewhat arbitrary)

**Note**:
The multi-dimensional confidence scoring thresholds were determined during Phase 1 Business Critique (2025-10-14) based on:
- Statistical reasoning (p<0.05 for high confidence, p<0.10 for moderate)
- Feature overlap heuristics (3/5 = 60% minimum)
- Success rate magnitude (>85% for high confidence)
- Cluster quality (silhouette >0.5 for high quality)

**Current thresholds**:
```python
# Statistical significance (0-40 points)
if p_value < 0.01:
    stat_score = 40
elif p_value < 0.05:
    stat_score = 30
elif p_value < 0.10:
    stat_score = 20

# Feature overlap (0-30 points)
overlap_score = overlap_count * 6  # 6 points per feature (out of 5)

# Success rate magnitude (0-20 points)
if success_rate >= 0.95:
    magnitude_score = 20
elif success_rate >= 0.90:
    magnitude_score = 15
elif success_rate >= 0.85:
    magnitude_score = 10

# Cluster quality (0-10 points)
if silhouette >= 0.5:
    quality_score = 10
elif silhouette >= 0.3:
    quality_score = 5

# Tier boundaries
GOLD: 75+ points
SILVER: 55-74 points
BRONZE: 35-54 points
EXPLORATORY: <35 points
```

**If adjusting thresholds**:
1. Document rationale in decision log (why change?)
2. Run `tests/unit/test_confidence_scoring.py` to verify tier boundaries still work
3. Re-run manual validation checklist on production data
4. Update this section with new thresholds and rationale

**Testing Requirements**:
- **Unit Test**: `tests/unit/test_confidence_scoring.py`
  - Tests all tier boundaries (GOLD/SILVER/BRONZE/EXPLORATORY)
  - Tests edge cases (score never negative, boundary conditions)

**Test Status**: ✅ Complete test suite in Stage5Tests.md Test #6

**Reference**:
- Stage5Tests.md Test #6
- Critique_MLModelTraining.md Q3 (confidence score design)

---

## 4. Dependencies & Integration
[TODO: Fill in Phase 2]

**Pre-filled guidance**:
- Stage 4 outputs: `rf_transformed.csv`, `{window}_rf_transformed.csv`, `{window}_km_transformed.csv`
- scipy >= 1.7.0 (for binomtest)
- sklearn (RandomForestClassifier, KMeans, silhouette_samples)
- See Section 3 for critical data dependencies (X matrix storage, feature name normalization)

---

## 5. Data Schemas
[TODO: Fill in Phase 2]

**Pre-filled guidance**:
- Input schemas from Stage 4 (see FeatureTransformationCHILD.md Section 5.2)
- Output: validation_results.json (see Critique Q3 for structure)
- Model files: .pkl format (joblib)

---

## 6. Error Handling & Validation
[TODO: Fill in Phase 2]

**Pre-filled guidance**:
- Fail-fast on missing Stage 4 files
- Validate feature name overlap ≥ 15/21 (catches normalization bug)
- Handle empty clusters (see Section 3)

---

## 7. Performance & Scalability
[TODO: Fill in Phase 2]

**Pre-filled guidance**:
- Training time: ~2-3 seconds per window-level model
- Total: ~120 seconds for all 90 models per hashtag (8 buckets)
- See Stage 5 MLPlanningv2.md lines 1604-1799 for model count breakdown

---

## 8. Testing Strategy ✅ COMPLETE

**Comprehensive testing specification**: `/documentation_migration/FutureDevelopments/Stage5Tests.md`

### Testing Requirements (MANDATORY before production)

**Layer 1: Unit Tests** (6 tests, ~10 seconds total)
- [ ] `tests/unit/test_binomial_test.py` (statistical significance)
- [ ] `tests/unit/test_feature_normalization.py` (CRITICAL: catches name mismatch)
- [ ] `tests/unit/test_kmeans_feature_ranking.py` (HIGH: validates logic)
- [ ] `tests/unit/test_success_rate.py` (simple calculation)
- [ ] `tests/unit/test_silhouette_score.py` (cluster quality)
- [ ] `tests/unit/test_confidence_scoring.py` (multi-dimensional scoring)

**Layer 2: Integration Tests** (1 test, ~30 seconds)
- [ ] `tests/integration/test_stage4_to_stage5_integration.py` (CRITICAL: uses REAL Stage 4 data)

**Layer 3: Manual Validation** (~30 minutes, human review)
- [ ] Complete manual validation checklist on first production run
- [ ] Review top features for intuitive sense
- [ ] Sign off on validation results

**Test Execution**:
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run integration tests (requires Stage 4 fixtures)
pytest tests/integration/ -v

# Manual validation
# Open Stage5Tests.md Layer 3 and follow checklist
```

**Estimated Total Testing Time**: 4-6 hours (implementation + debugging + manual review)

**Test Documentation**: See Stage5Tests.md for:
- Complete test implementations (copy-paste ready)
- Expected outputs
- Fixture setup instructions
- Troubleshooting guide

---

## 9. Future Enhancements
[TODO: Fill in Phase 2]

---

## 10. References & Related Docs

### Parent Documents
- **MLPlanningv2.md Stage 5** (Lines 1588-1945) - Architectural specification
- **Critique_MLModelTraining.md** - Phase 1 Business Critique with Q&A decisions
- **Stage5Tests.md** - Comprehensive testing specification

### Related Child Docs
- **FeatureTransformationCHILD.md** (Stage 4) - Produces inputs for Stage 5

### Key Decisions from Phase 1 Critique
- **Q1**: Foundation model count updated from 16 → 90 models (resolved 2025-10-14)
- **Q2**: Window-level RF necessity justified (creator testing guidelines)
- **Q3**: Multi-dimensional confidence score designed (statistical + overlap + magnitude + quality)

---

## Appendix A: Decision Log

**Purpose**: Record major design decisions with rationale

### Decision 1: Multi-Dimensional Confidence Score (Over Single Metric)

**Date**: 2025-10-14 (Phase 1 Critique Q3)

**Context**: How to validate which K-Means clusters are trustworthy for creator guidelines?

**Alternatives Considered**:
- Option 1 (Top 3, 2/3 overlap): Too strict, high false negative rate
- Option 2 (Top 5, 3/5 overlap + 70% success rate): Wrong baseline (70% < 80%)
- Option 3 (Statistical significance only): Doesn't leverage RF insights
- **Option 4 (Multi-dimensional score)**: Combines 4 signals into 0-100 score

**Decision**: Multi-dimensional confidence score (Option 4)

**Rationale**:
- No single metric is make-or-break (handles edge cases better)
- Gradual confidence spectrum (not binary pass/fail)
- Easy for Stage 7 LLM to interpret (GOLD/SILVER/BRONZE/EXPLORATORY)
- Combines statistical rigor (p-values) with practical validation (feature overlap)

**Trade-offs**: More complex than single metric, but handles sample size issues and edge cases gracefully

**Reference**: Critique_MLModelTraining.md Q3 ultrathink analysis

---

### Decision 2: Feature Name Normalization Required

**Date**: 2025-10-14 (Phase 1 Critique)

**Context**: K-Means and RF features have different naming conventions from Stage 4

**Decision**: Implement `normalize_feature_name()` to strip suffixes before comparison

**Rationale**: Without normalization, overlap = 0 (guaranteed bug). Normalization is simple (3 string replacements) and fully testable.

**Trade-offs**: Adds one function, but prevents critical bug.

**Reference**: Section 3, Critical Warning #1

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 (STUB) | 2025-10-14 | Claude (Phase 1 Critique) | Created stub with Section 3 (Critical Implementation Warnings) pre-filled from testing analysis and critique Q&A |

---

## Next Steps for Future CLI Instance

**When expanding this stub into full HLD (Phase 2/3)**:

1. ✅ **Keep Section 3 intact** - DO NOT MODIFY Critical Implementation Warnings
2. Fill in TODOs in Sections 1, 2, 4, 5, 6, 7, 9 (Context, Architecture, Dependencies, etc.)
3. Reference **Critique_MLModelTraining.md** for Q&A decisions (especially Q3 validation protocol)
4. Reference **Stage5Tests.md** for testing requirements (already complete)
5. Reference **MLPlanningv2.md Stage 5** (lines 1588-1945) for architectural details
6. Update **Change Log** with Phase 2 expansion date
7. Remove **(STUB)** from title when complete
8. Update **Status** at top from "STUB" to "Draft" or "Complete"

**Estimated effort for Phase 2 expansion**: 6-8 hours (filling in architecture, schemas, dependencies)

**DO NOT start from scratch** - this stub already contains critical bug prevention logic.
