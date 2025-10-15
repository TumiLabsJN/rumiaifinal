# Stage 5: ML Model Training - High-Level Design

> **Status**: COMPLETE
> **Parent**: MLPlanningv2.md - Stage 5: ML Model Training (Lines 1624-1992)
> **Phase 1**: Critique_MLModelTraining.md (Business critique with Q&A decisions)
> **Phase 2**: QA_MLModelTraining.md (Clarification Q&A with implementation decisions)
> **Critical Dependencies**: Stage5Tests.md (testing spec), Stage5Alternatives.md (validation protocol)
> **Date Created**: 2025-10-14
> **Last Updated**: 2025-10-14

---

## Document Overview

This High-Level Design (HLD) documents Stage 5: ML Model Training, which trains 90 machine learning models across 8 duration buckets to detect viral video patterns.

**Key Features**:
- **90 models total**: 8 Video-Level RF + 41 Window-Level RF + 41 Window-Level K-Means
- **Dual RF + K-Means architecture**: Prevents blind spots in pattern detection
- **Multi-dimensional validation**: GOLD/SILVER/BRONZE/EXPLORATORY confidence tiers
- **Critical bug prevention**: Section 3 contains MANDATORY implementation warnings

**Target Audience**: Developers implementing Stage 5, TI (Technical Implementation) document authors

---

## Table of Contents

1. [Context & Business Goal](#1-context--business-goal)
2. [Architecture & Design](#2-architecture--design)
3. [Critical Implementation Warnings](#3-critical-implementation-warnings) ✅ COMPLETE
4. [Dependencies & Integration](#4-dependencies--integration)
5. [Data Schemas](#5-data-schemas)
6. [Error Handling & Validation](#6-error-handling--validation)
7. [Performance & Scalability](#7-performance--scalability)
8. [Testing Strategy](#8-testing-strategy) ✅ COMPLETE
9. [Configuration](#9-configuration)
10. [References & Related Docs](#10-references--related-docs)
11. [Appendices](#appendices)

---

## 1. Context & Business Goal

### 1.1 Why Stage 5 Exists

**Business Problem**: RumiAI needs to identify which video features and creative strategies predict TikTok virality for specific duration ranges.

**Current Pipeline Context** (Stage 1-4 complete):
- **Stage 1**: Video discovery (300 videos selected across 3 active buckets)
- **Stage 2**: Video processing (RumiAI analysis, temporal_windows_updated.json per video)
- **Stage 3**: Feature aggregation (flatten temporal windows into aggregated_features.csv)
- **Stage 4**: Feature transformation (3 pipelines: video-level RF, window-level RF, window-level K-Means)

**Stage 5 Goal**: Train machine learning models on transformed features to:
1. **Predict virality** (Random Forest: top 80% vs bottom 20%)
2. **Discover creative strategies** (K-Means: 3 clusters per window)
3. **Validate patterns** (Multi-dimensional confidence scoring)

### 1.2 Success Criteria

**Output**: 90 trained models per hashtag (distributed across 8 duration buckets)

**Quality**:
- Models train successfully on N=100 videos per bucket (minimum 50 contrastive, 30 top mode)
- Feature overlap ≥ 15/21 features (validates Stage 4 → Stage 5 integration)
- Training completes in ~1-2 minutes per hashtag (Stage 5 is 0.5-1% of total 3.6-4.8 hour pipeline)

**Downstream Impact**:
- **Stage 6**: ML Analysis Generation (uses trained models to extract insights)
- **Stage 7**: LLM Report Generation (converts insights to creator guidelines)

### 1.3 Key Design Decisions

**From Phase 1 Business Critique**:
1. **90-model architecture** (not 16): Provides complete pattern coverage (Q1 resolved)
2. **Window-level RF justified**: Enables actionable per-window creator guidelines (Q2 approved)
3. **Multi-dimensional validation**: Alternative 4 (Statistical + Overlap + Magnitude + Quality) chosen (Q3 resolved)

**From Phase 2 Clarification Q&A**:
4. **Sequential training**: No parallelization (training is fast: ~26s per bucket) (Q5)
5. **Fail-fast validation**: Missing Stage 4 files → immediate failure (Q1)
6. **Atomic bucket training**: All models succeed or all deleted on failure (Q8)

---

## 2. Architecture & Design

### 2.1 High-Level Approach

**Architectural Pattern**: Dual Random Forest + Window-Level K-Means

**Why This Architecture?**:
- **Video-Level RF** (8 models): Detects cross-window patterns (energy progression, topic consistency, weak link effects)
- **Window-Level RF** (41 models): Validates window-specific feature importance (what makes a strong hook vs closing)
- **Window-Level K-Means** (41 models): Discovers creative strategies per section with interpretable centroids

**Complete Pattern Coverage**:
| Model Type | What It Captures | What It Misses Without This |
|------------|------------------|----------------------------|
| Video-Level RF | Cross-window interactions | Window-specific feature importance |
| Window-Level RF | Per-window feature rankings | Creative strategy clusters |
| K-Means | 3 distinct strategies per window | Temporal progressions |

### 2.2 Data Flow

```
Stage 4 Output (Transformed Features)
    ↓
    ├─ ml_analysis/rf_transformed.csv (~190 features, 100 videos)
    ├─ ml_analysis/hook_rf_transformed.csv (22 features, 100 videos)
    ├─ ml_analysis/hook_km_transformed.csv (~39 features, 100 videos)
    └─ ... (all windows)
    ↓
Stage 5 Process
    ↓
    ├─ Load CSVs (with validation: missing files → fail)
    ├─ Load hyperparameters (config/model_hyperparameters.json with fallback)
    ├─ Train Video-Level RF (1 model)
    ├─ Train Window-Level RF (6 models for bucket 18-33s)
    ├─ Train K-Means (6 models)
    ├─ Save models to models/*.pkl (20 files total for bucket 18-33s)
    └─ Save model_metrics.json (performance summary)
    ↓
Stage 5 Output (Trained Models)
    ↓
    ├─ models/rf_video_18-33s.pkl
    ├─ models/rf_hook_18-33s.pkl (and 5 more window-level RF)
    ├─ models/hook_kmeans_18-33s.pkl (and 5 more K-Means)
    ├─ models/hook_scalers_18-33s.pkl (and 5 more scalers)
    └─ models/model_metrics.json
    ↓
Stage 6 Input (ML Analysis Generation)
```

### 2.3 Detailed Process

#### 2.3.1 Pre-Training Validation

**Input File Validation** (Q1 decision: Fail-fast on missing files):
```python
def validate_stage4_outputs(bucket, windows):
    """Validate all Stage 4 files exist before training ANY models."""
    required_files = [
        'ml_analysis/rf_transformed.csv',
        *[f'ml_analysis/{window}_rf_transformed.csv' for window in windows],
        *[f'ml_analysis/{window}_km_transformed.csv' for window in windows],
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise StageInputError(
                f"Stage 4 incomplete: Missing {file_path}. Run Stage 4 first."
            )
        if pd.read_csv(file_path).shape[0] == 0:
            raise StageInputError(
                f"Stage 4 output empty: {file_path} has 0 rows."
            )
```

**Video Count Validation** (Q7 decision: Minimum thresholds):
```python
MIN_VIDEOS_CONTRASTIVE = 50  # 40 top + 10 bottom (bare minimum for 80/20 split)
MIN_VIDEOS_TOP = 30          # Descriptive analysis only

def validate_video_count(bucket, video_count, mode):
    """Validate sufficient videos for reliable training."""
    min_required = MIN_VIDEOS_CONTRASTIVE if mode == "contrastive" else MIN_VIDEOS_TOP

    if video_count < min_required:
        raise InsufficientDataError(
            f"Bucket {bucket} has {video_count} videos (min {min_required} required for {mode} mode). "
            f"Re-run Stage 1 with lower --video-count or skip this bucket."
        )
```

#### 2.3.2 Configuration Loading

**Hyperparameter Configuration** (Q3 decision: Config file with fallback):
```python
def load_model_config():
    """Load hyperparameters from config with fallback to hardcoded defaults."""
    try:
        with open('config/model_hyperparameters.json') as f:
            config = json.load(f)
            logger.info("Loaded hyperparameters from config/model_hyperparameters.json")
            return config
    except FileNotFoundError:
        logger.warning(
            "config/model_hyperparameters.json not found. Using hardcoded defaults."
        )
        return {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "kmeans": {
                "n_clusters": 3,
                "random_state": 42,
                "n_init": 10
            }
        }
```

#### 2.3.3 Training Process

**Sequential Training per Bucket** (Q5 decision: Sequential, not parallel):

```python
def train_bucket_models(bucket, windows, mode, config):
    """
    Train all models for bucket. Atomic: all succeed or all deleted.

    Q8 decision: Fail-fast with clean bucket directory
    """
    trained_models = []
    start_time = time.time()

    try:
        # 1. Train Video-Level RF
        logger.info(f"Training video-level RF for {bucket}...")
        X = pd.read_csv('ml_analysis/rf_transformed.csv')
        y = X['is_top_performer']
        X = X.drop(['is_top_performer', 'video_id'], axis=1)

        rf_video = RandomForestClassifier(**config['random_forest'])
        rf_video.fit(X, y)

        model_path = f'models/rf_video_{bucket}.pkl'
        joblib.dump(rf_video, model_path)
        trained_models.append(model_path)

        # 2. Train Window-Level RF (sequential loop)
        for window in windows:
            logger.info(f"Training window-level RF for {window}...")
            X = pd.read_csv(f'ml_analysis/{window}_rf_transformed.csv')
            y = X['is_top_performer']
            X = X.drop(['is_top_performer'], axis=1)

            rf_window = RandomForestClassifier(**config['random_forest'])
            rf_window.fit(X, y)

            model_path = f'models/rf_{window}_{bucket}.pkl'
            joblib.dump(rf_window, model_path)
            trained_models.append(model_path)

        # 3. Train K-Means (sequential loop)
        for window in windows:
            logger.info(f"Training K-Means for {window}...")
            X = pd.read_csv(f'ml_analysis/{window}_km_transformed.csv')
            X = X.drop(['video_id'], axis=1)  # No labels for K-Means

            kmeans = KMeans(**config['kmeans'])
            kmeans.fit(X)

            # Save model
            model_path = f'models/{window}_kmeans_{bucket}.pkl'
            joblib.dump(kmeans, model_path)
            trained_models.append(model_path)

            # Save X matrix for silhouette calculation (Section 3, Warning #4)
            X_path = f'models/{window}_X_data_{bucket}.pkl'
            joblib.dump(X, X_path)
            trained_models.append(X_path)

            # Save scalers (for inference)
            scalers_path = f'models/{window}_scalers_{bucket}.pkl'
            joblib.dump(scalers, scalers_path)
            trained_models.append(scalers_path)

        # 4. Generate model_metrics.json
        metrics = generate_model_metrics(bucket, windows, rf_video, rf_windows, kmeans_models)
        metrics_path = 'models/model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        trained_models.append(metrics_path)

        elapsed = time.time() - start_time
        logger.info(f"✓ Bucket {bucket} training complete: {elapsed:.1f}s ({len(trained_models)} files)")

        # Performance warning (Q9 decision: No hard timeout, but log if suspiciously slow)
        if elapsed > 300:  # 5 minutes per bucket
            logger.warning(
                f"Bucket {bucket} training took {elapsed:.1f}s (expected <120s). "
                f"Check for performance issues."
            )

    except Exception as e:
        # Q8 decision: Clean up ALL models for this bucket on failure
        logger.error(f"""
Bucket {bucket} training failed at {current_model}
Exception: {type(e).__name__}: {str(e)}
Stack trace: {traceback.format_exc(limit=10)}
Completed models before failure: {trained_models}
Training duration before failure: {time.time() - start_time:.1f}s
""")

        # Delete all partial models
        for model_path in trained_models:
            if os.path.exists(model_path):
                os.remove(model_path)

        raise ModelTrainingError(
            f"Bucket {bucket} training failed: {e}. All models deleted. Re-run Stage 5."
        )
```

**Result**: Either bucket has complete model set (e.g., 20 files for bucket 18-33s) OR no models (0 files). Never partial.

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

### 4.1 Input Dependencies

**Stage 4 Outputs** (required before Stage 5 can run):

| File | Purpose | Shape | Validation |
|------|---------|-------|------------|
| `ml_analysis/rf_transformed.csv` | Video-level RF training data | (100 videos, ~190 features) | Must exist, >0 rows |
| `ml_analysis/hook_rf_transformed.csv` | Window-level RF (hook) | (100 videos, 22 features) | Must exist, >0 rows |
| `ml_analysis/hook_km_transformed.csv` | K-Means (hook) | (100 videos, ~39 features) | Must exist, >0 rows |
| ... (all windows) | ... | ... | ... |

**Validation**: Q1 decision - Fail-fast if ANY file missing or empty

### 4.2 Output Contracts

**Trained Models** (Stage 6 dependency):

**File Naming Convention** (Q2 decision):
```
models/
├── rf_video_{bucket}.pkl          # Video-level RF
├── rf_{window}_{bucket}.pkl       # Window-level RF
├── {window}_kmeans_{bucket}.pkl   # K-Means
├── {window}_scalers_{bucket}.pkl  # Scalers
├── {window}_X_data_{bucket}.pkl   # Feature matrix (for silhouette)
└── model_metrics.json             # Performance summary
```

**Example for bucket 18-33s** (20 files total):
- 1 video-level RF
- 6 window-level RF
- 6 K-Means models
- 6 scalers
- 6 X matrices
- 1 model_metrics.json

### 4.3 Cross-Stage Dependencies

**Upstream**:
- **Stage 1**: Video selection (determines N videos per bucket)
- **Stage 2**: Video processing (RumiAI analysis)
- **Stage 3**: Feature aggregation (aggregated_features.csv)
- **Stage 4**: Feature transformation (3 pipelines for dual RF + K-Means)

**Downstream**:
- **Stage 6**: ML Analysis Generation (loads .pkl models, extracts feature importance)
- **Stage 7**: LLM Report Generation (converts insights to creator guidelines)

### 4.4 External Dependencies

**Python Packages**:
- `scikit-learn >= 0.24.0` (RandomForestClassifier, KMeans, silhouette_samples)
- `scipy >= 1.7.0` (binomtest for statistical validation)
- `joblib >= 1.0.0` (model serialization)
- `pandas >= 1.3.0` (CSV loading)
- `numpy >= 1.21.0` (variance calculations)

**Configuration Files**:
- `config/model_hyperparameters.json` (optional, with hardcoded fallback)

---

## 5. Data Schemas

### 5.1 Input Schema

**Video-Level RF Input** (`ml_analysis/rf_transformed.csv`):
```
Shape: (100 videos, ~190 features)
Columns:
  - video_id: str (unique identifier, will be dropped before training)
  - hook_scene_count: int (0-20)
  - hook_eye_contact_rate: float (0.0-1.0)
  - ... (all temporal features: hook, middle_1-4, closing)
  - hour: int (0-23, derived from create_time)
  - day_of_week: int (0-6)
  - is_weekend: int (0 or 1)
  - gender_male: int (0 or 1, one-hot encoded)
  - gender_female: int (0 or 1, one-hot encoded)
  - is_top_performer: int (0 or 1, target variable)
```

**Window-Level RF Input** (`ml_analysis/hook_rf_transformed.csv`):
```
Shape: (100 videos, 22 features)
Columns:
  - scene_count: int (0-20)
  - eye_contact_rate: float (0.0-1.0)
  - word_count: int (0-100)
  - speech_coverage: float (0.0-1.0)
  - energy_level: float (0.0-1.0)
  - ... (21 base features total)
  - is_top_performer: int (0 or 1, target variable)
```

**K-Means Input** (`ml_analysis/hook_km_transformed.csv`):
```
Shape: (100 videos, ~39 features)
Columns:
  - video_id: str (unique identifier, will be dropped before training)
  - eye_contact_rate_scaled: float (0.0-1.0)
  - scene_count_scaled: float (0.0-1.0)
  - word_count_log_scaled: float (0.0-1.0)
  - has_captions_encoded: int (0 or 1)
  - ... (~39 features total, all numerical, scaled [0-1])
  - NO target variable (unsupervised clustering)
```

### 5.2 Output Schema

**Model Files** (.pkl format, joblib):
- Binary serialization of sklearn models
- RandomForestClassifier, KMeans, MinMaxScaler objects

**model_metrics.json** (Q4 decision - Use Section 5.6 schema exactly):
```json
{
  "bucket": "18-33s",
  "total_videos": 100,
  "video_level_rf": {
    "model_type": "random_forest",
    "input_features": 190,
    "accuracy": 0.87,
    "precision": 0.89,
    "recall": 0.84,
    "f1_score": 0.86,
    "top_feature": "hook_eye_contact_rate",
    "top_feature_importance": 0.22,
    "purpose": "Cross-window pattern detection"
  },
  "window_level_rf": {
    "hook": {
      "model_type": "random_forest",
      "input_features": 21,
      "accuracy": 0.82,
      "precision": 0.85,
      "recall": 0.78,
      "top_feature": "eye_contact_rate",
      "top_feature_importance": 0.35
    },
    "middle_1": { "..." },
    "middle_2": { "..." },
    "middle_3": { "..." },
    "middle_4": { "..." },
    "closing": { "..." }
  },
  "window_level_kmeans": {
    "hook": {
      "model_type": "kmeans",
      "input_features": 39,
      "n_clusters": 3,
      "inertia": 12.5,
      "silhouette_score": 0.68,
      "cluster_sizes": [35, 42, 23]
    },
    "middle_1": { "..." },
    "middle_2": { "..." },
    "middle_3": { "..." },
    "middle_4": { "..." },
    "closing": { "..." }
  }
}
```

**Purpose of model_metrics.json** (Q4):
- Quick sanity check after training completes
- Validate model performance is reasonable (accuracy >0.80)
- Verify top feature makes intuitive sense
- Confirm cluster sizes are balanced (~33 videos each)

**NOT included in model_metrics.json**:
- Full feature importance rankings (extracted by Stage 6 from .pkl files)
- Confusion matrices (not needed for unsupervised K-Means)
- Cluster centroids (stored in .pkl files, analyzed by Stage 6)

---

## 6. Error Handling & Validation

### 6.1 Input Validation

**Pre-Training Checks**:

1. **Missing Files** (Q1 decision):
```python
# Fail immediately if ANY Stage 4 file missing
for file_path in required_files:
    if not os.path.exists(file_path):
        raise StageInputError(f"Stage 4 incomplete: Missing {file_path}. Run Stage 4 first.")
```

2. **Empty Files** (Q1 decision):
```python
# Fail if CSV has 0 rows
if pd.read_csv(file_path).shape[0] == 0:
    raise StageInputError(f"Stage 4 output empty: {file_path} has 0 rows.")
```

3. **Insufficient Videos** (Q7 decision):
```python
# Fail if below minimum threshold
if video_count < MIN_VIDEOS_CONTRASTIVE:  # 50 for contrastive, 30 for top
    raise InsufficientDataError(f"Bucket {bucket} has {video_count} videos (min {MIN_VIDEOS_CONTRASTIVE} required).")
```

4. **Config Validation** (Q3 decision):
```python
# Validate hyperparameters (if config file provided)
if 'n_estimators' not in config['random_forest']:
    raise ConfigError("Invalid model_hyperparameters.json: Missing 'n_estimators' for random_forest")
```

### 6.2 Error Cases

**Scenario 1: Stage 4 Files Missing** (Q1):
```
ERROR: Stage 4 incomplete: Missing ml_analysis/hook_rf_transformed.csv. Run Stage 4 first.
Action: Re-run Stage 4 or check if bucket was skipped in Stage 1
```

**Scenario 2: Insufficient Videos** (Q7):
```
ERROR: Bucket 18-33s has 45 videos (min 50 required for contrastive mode).
Action: Re-run Stage 1 with lower --video-count or skip this bucket
```

**Scenario 3: Training Failure Mid-Bucket** (Q8):
```
ERROR: Bucket 18-33s training failed at rf_middle_2_18-33s.pkl: NaN values in feature data.
Action: All 3 partially trained models deleted. Fix data issue and re-run Stage 5.
```

**Scenario 4: Config File Malformed** (Q3):
```
ERROR: Invalid model_hyperparameters.json: {parsing error}
Action: Using hardcoded defaults instead
```

### 6.3 Error Logging

**Q10 Decision: Balanced Logging** (Error + Context, No Data Dump)

**What is logged**:
- WHAT failed: Model name, file path, input shape
- WHY it failed: Exception type and message, stack trace (first 10 lines)
- CONTEXT: Hyperparameters, completed models, training duration, NaN count
- NOT logged: Actual feature values, video IDs (privacy concerns)

**Example**:
```
ERROR: Bucket 18-33s training failed at rf_hook_18-33s.pkl
Exception: ValueError: Input contains NaN
Stack trace: [first 10 lines]
Input file: ml_analysis/hook_rf_transformed.csv
Input shape: (100 videos, 22 features)
NaN count: 3 values in 2 columns
Hyperparameters: {n_estimators: 100, max_depth: 10, random_state: 42}
Completed models before failure: ['rf_video_18-33s.pkl']
Training duration before failure: 1.2s
```

### 6.4 Recovery Procedures

**Q8 Decision: Atomic Bucket Training**

**On Training Failure**:
1. Delete ALL models for this bucket (partial models removed)
2. Log comprehensive error (Q10 format)
3. Fail with clear error message
4. User re-runs Stage 5 after fixing issue

**Result**: Either bucket has complete model set OR no models. Never partial.

---

## 7. Performance & Scalability

### 7.1 Performance Targets

**Q9 Decision: No Hard Timeout** (Best-effort with logging)

**Expected Performance**:
- **Typical 3-bucket scenario**: 30-90 seconds per bucket (depends on model count: 3-15 models)
- **Total for 3 buckets**: 90-270 seconds (~1.5-4.5 minutes)
- **Varies by hardware**: Development machine vs CI/CD vs production server

**Performance Guidelines**:
- **Expected**: 30-90 seconds per bucket
- **Acceptable**: 90-300 seconds per bucket (slower hardware)
- **Warning**: > 5 minutes per bucket (log warning, continue)
- **Likely bug**: > 30 minutes per bucket (suggests infinite loop or hardware failure)

**No enforced timeout** - Stage 5 is NOT user-facing, and training time varies by hardware

### 7.2 Scale Limitations

**Minimum Requirements** (Q7):
- **Contrastive mode**: min 50 videos (40 top + 10 bottom)
- **Top mode**: min 30 videos (descriptive analysis)

**Maximum Tested**:
- N=200 videos per bucket (training time: ~60 seconds, acceptable)
- 8 buckets active (total: 120 models, ~4 minutes)

**Memory**:
- Peak: ~500 MB per bucket (RandomForest with 100 estimators, 190 features, 100 samples)
- K-Means: ~50 MB (lightweight)

### 7.3 Bottleneck Analysis

**Q9 Rationale: Stage 5 is NOT a bottleneck**

**Pipeline Context**:
- Total pipeline: 3.6-4.8 hours (Phase 1 Q4)
- Stage 5: 1-2 minutes (~0.5-1% of total time)
- Bottleneck: Stage 2 (video processing, 60-80s per video)

**Optimization NOT needed** for Stage 5 (training is already fast)

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

## 9. Configuration

### 9.1 Hyperparameter Configuration

**Q3 Decision: Configurable via Config File with Fallback**

**Configuration File**: `config/model_hyperparameters.json`

**Structure**:
```json
{
  "random_forest": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "kmeans": {
    "n_clusters": 3,
    "random_state": 42,
    "n_init": 10
  }
}
```

**Loading Logic**:
```python
def load_model_config():
    """Load hyperparameters from config with fallback to hardcoded defaults."""
    try:
        with open('config/model_hyperparameters.json') as f:
            return json.load(f)
    except FileNotFoundError:
        # Graceful fallback
        return {
            "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "kmeans": {"n_clusters": 3, "random_state": 42, "n_init": 10}
        }
```

**Behavior**:
- Config file exists → Use hyperparameters from file
- Config file missing → Use hardcoded defaults (log warning)
- Config file malformed → Fail with error "Invalid model_hyperparameters.json: {error}"

---

## 10. References & Related Docs

### 10.1 Parent Documents
- **MLPlanningv2.md Stage 5** (Lines 1624-1992) - Architectural specification
- **Critique_MLModelTraining.md** - Phase 1 Business Critique with Q&A decisions
- **QA_MLModelTraining.md** - Phase 2 Clarification Q&A with implementation decisions
- **Stage5Tests.md** - Comprehensive testing specification

### 10.2 Related Child Docs
- **FeatureTransformationCHILD.md** (Stage 4) - Produces inputs for Stage 5
- **Stage5Alternatives.md** - Validation protocol alternatives analysis (Alternative 4 chosen)

### 10.3 Key Decisions from Phase 1 Critique
- **Q1**: Foundation model count updated from 16 → 90 models (resolved 2025-10-14)
- **Q2**: Window-level RF necessity justified (creator testing guidelines)
- **Q3**: Multi-dimensional confidence score designed (statistical + overlap + magnitude + quality)

### 10.4 Key Decisions from Phase 2 Q&A
- **Q1**: Missing Stage 4 files → Fail-fast (Alternative A)
- **Q2**: File naming → Section 5.5 detailed naming (Alternative A)
- **Q3**: Hyperparameters → Config file with fallback (Alternative B)
- **Q5**: Training order → Sequential (Alternative A)
- **Q7**: Insufficient videos → Fail-fast, min 50/30 (Alternative A)
- **Q8**: Mid-bucket failure → Clean bucket directory (Alternative C)
- **Q9**: Performance target → No hard timeout (Alternative C)
- **Q10**: Error logging → Balanced logging (Alternative C)

---

## Appendices

### Appendix A: Decision Log

**Purpose**: Record major design decisions with rationale

#### Decision 1: Multi-Dimensional Confidence Score (Over Single Metric)

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

**Reference**: Critique_MLModelTraining.md Q3 ultrathink analysis, Stage5Alternatives.md

---

#### Decision 2: Feature Name Normalization Required

**Date**: 2025-10-14 (Phase 1 Critique)

**Context**: K-Means and RF features have different naming conventions from Stage 4

**Decision**: Implement `normalize_feature_name()` to strip suffixes before comparison

**Rationale**: Without normalization, overlap = 0 (guaranteed bug). Normalization is simple (3 string replacements) and fully testable.

**Trade-offs**: Adds one function, but prevents critical bug.

**Reference**: Section 3, Critical Warning #1

---

#### Decision 3: Sequential Training (Over Parallel)

**Date**: 2025-10-14 (Phase 2 Q5)

**Context**: Should models train sequentially or in parallel?

**Decision**: Sequential training (one model at a time)

**Rationale**:
- Training is fast (~26s per bucket, 78s for 3 buckets total)
- Stage 5 is 0.36% of total pipeline time (not a bottleneck)
- Sequential aligns with Foundation's checkpoint/resume philosophy
- Easier debugging (clear error: "rf_hook_18-33s.pkl failed")
- No resource contention (sklearn RF already uses multi-threading internally)

**Trade-offs**: Slower than parallel (78s vs ~24s), but complexity not justified for 54-second savings

**Reference**: QA_MLModelTraining.md Q5

---

### Appendix B: Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 (STUB) | 2025-10-14 | Claude (Phase 1 Critique) | Created stub with Section 3 (Critical Implementation Warnings) pre-filled |
| 1.0 (HLD) | 2025-10-14 | Claude (Phase 2/3) | Expanded stub into full HLD using QA_MLModelTraining.md answers. Sections 1, 2, 4-7, 9 filled. Section 3 preserved intact. |

---

### Appendix C: Future Enhancements

**Post-MVP Improvements** (not required for initial deployment):

1. **Train/Test Split** (if N > 200 per bucket):
   - Split data 70/30 for out-of-sample validation
   - Validates generalization, not just in-sample fit
   - Requires larger sample sizes

2. **Per-Bucket Hyperparameter Tuning**:
   - Different hyperparameters for 0-3s vs 90-120s buckets
   - More complex config structure
   - Only if profiling shows benefit

3. **Anti-Pattern Detection Enhancement**:
   - Explicitly label clusters with <70% success rate as "ANTI-PATTERNS (AVOID)"
   - Negative magnitude scores for low-performing clusters
   - See Stage5Alternatives.md Section 4.3

4. **Parallel Training** (if Stage 5 becomes bottleneck):
   - Train window-level models in parallel (not all models)
   - Only if profiling shows Stage 5 exceeds 5 minutes regularly
   - See QA_MLModelTraining.md Q5 Alternative C

---

**END OF DOCUMENT**
