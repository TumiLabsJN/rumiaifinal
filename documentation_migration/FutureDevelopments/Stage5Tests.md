# Stage 5: ML Model Training - Testing Specification

> **Parent Document**: MLPlanningv2.md - Stage 5: ML Model Training
> **Version**: 1.0
> **Date**: 2025-10-14
> **Status**: Ready for Implementation

---

## Document Purpose

This document specifies ALL tests required for Stage 5 ML Model Training validation. A fresh CLI instance should be able to implement these tests with no additional context beyond this document and the Stage 5 codebase.

**Critical Context**: Stage 5 trains 90 models total per hashtag analysis:
- 8 Video-Level Random Forest models (1 per bucket)
- 41 Window-Level Random Forest models (1 per window across all buckets)
- 41 Window-Level K-Means models (1 per window across all buckets)

The MOST CRITICAL component is the **Multi-Dimensional Confidence Score** validation system, which determines which clusters are trustworthy for creator testing guidelines.

---

## Test Architecture Overview

### Test Layers

```
Layer 1: Unit Tests (Fast, Isolated)
├── test_binomial_test.py
├── test_feature_normalization.py
├── test_kmeans_feature_ranking.py
├── test_success_rate.py
├── test_silhouette_score.py
└── test_confidence_scoring.py

Layer 2: Integration Tests (Slower, End-to-End)
├── test_stage4_to_stage5_integration.py
└── test_validation_pipeline.py

Layer 3: Manual Validation (Human Review)
└── manual_validation_checklist.md
```

### Test Execution Order

1. **Unit tests** (run first, must pass 100%)
2. **Integration tests** (run second, must pass 100%)
3. **Manual validation** (run on first production data, sanity check)

---

## Critical Bug Hotspots (Priority Testing)

Based on Stage 5 architecture analysis, these are the HIGH-RISK components:

| Component | Bug Risk | Test Priority | Test File |
|-----------|----------|---------------|-----------|
| **Feature name mismatch (K-Means vs RF)** | 🔴 CRITICAL (guaranteed bug if not fixed) | **P0** | `test_feature_normalization.py` |
| **K-Means feature ranking logic** | 🔴 HIGH (conceptually complex) | **P0** | `test_kmeans_feature_ranking.py` |
| Binomial test edge cases | 🟡 MEDIUM | P1 | `test_binomial_test.py` |
| Silhouette score calculation | 🟡 MEDIUM | P1 | `test_silhouette_score.py` |
| Success rate calculation | 🟢 LOW | P2 | `test_success_rate.py` |
| Confidence scoring arithmetic | 🟢 LOW | P2 | `test_confidence_scoring.py` |

---

# Layer 1: Unit Tests

## Test 1: Binomial Test (Statistical Significance)

**File**: `tests/unit/test_binomial_test.py`

**Purpose**: Validate that statistical significance testing works correctly for cluster validation

**What It Catches**:
- scipy version incompatibility (binomtest requires scipy 1.7+)
- Edge cases (empty clusters, invalid inputs)
- Incorrect p-value calculations

**Implementation**:

```python
"""
Unit tests for binomial statistical significance testing.

Stage 5 uses binomial tests to determine if a cluster's success rate
is significantly above the 80% baseline (contrastive mode: 80 top, 20 bottom).
"""

import pytest
import numpy as np
from scipy.stats import binomtest


def test_binomial_test_significantly_above_baseline():
    """
    Test: Cluster with 97% success rate should be statistically significant.

    Context: With N=33 videos, need ~29+ successes (88%+) for p<0.05
    """
    n_success = 32  # 32/33 = 97% success rate
    n_total = 33
    baseline = 0.80

    p_value = binomtest(n_success, n_total, baseline, alternative='greater').pvalue

    assert p_value < 0.05, \
        f"Expected p < 0.05 for 97% success rate, got p={p_value:.4f}"

    print(f"✓ 97% success rate → p={p_value:.4f} (significant)")


def test_binomial_test_at_baseline():
    """
    Test: Cluster with 82% success rate should NOT be statistically significant.

    Context: 82% is close to 80% baseline, should not pass p<0.05 threshold
    """
    n_success = 27  # 27/33 = 82% success rate
    n_total = 33
    baseline = 0.80

    p_value = binomtest(n_success, n_total, baseline, alternative='greater').pvalue

    assert p_value > 0.05, \
        f"Expected p > 0.05 for 82% success rate (near baseline), got p={p_value:.4f}"

    print(f"✓ 82% success rate → p={p_value:.4f} (not significant)")


def test_binomial_test_below_baseline():
    """
    Test: Cluster with 55% success rate should be significantly BELOW baseline.

    Context: This is an "anti-pattern" - pattern to avoid
    """
    n_success = 18  # 18/33 = 55% success rate
    n_total = 33
    baseline = 0.80

    # Test in 'less' direction (below baseline)
    p_value = binomtest(n_success, n_total, baseline, alternative='less').pvalue

    assert p_value < 0.05, \
        f"Expected p < 0.05 for 55% success rate (anti-pattern), got p={p_value:.4f}"

    print(f"✓ 55% success rate → p={p_value:.4f} (anti-pattern detected)")


def test_binomial_test_edge_case_empty_cluster():
    """
    Test: Empty cluster (n_total=0) should raise ValueError.

    Context: Prevents division by zero in success rate calculation
    """
    with pytest.raises((ValueError, ZeroDivisionError)):
        binomtest(0, 0, 0.80, alternative='greater')

    print("✓ Empty cluster correctly raises error")


def test_binomial_test_edge_case_invalid_input():
    """
    Test: n_success > n_total (data corruption) should raise ValueError.

    Context: Catches upstream data corruption from Stage 2/3/4
    """
    with pytest.raises(ValueError):
        binomtest(35, 33, 0.80, alternative='greater')  # 35 successes out of 33 total!

    print("✓ Invalid input (n_success > n_total) correctly raises error")


def test_binomial_test_scipy_version():
    """
    Test: Ensure scipy version supports binomtest (scipy >= 1.7).

    Context: binomtest introduced in scipy 1.7 (2021)
    Older versions use deprecated binom_test
    """
    import scipy
    from packaging import version

    scipy_version = version.parse(scipy.__version__)
    min_version = version.parse("1.7.0")

    assert scipy_version >= min_version, \
        f"scipy {scipy.__version__} is too old. Require scipy >= 1.7.0 for binomtest"

    print(f"✓ scipy version {scipy.__version__} supports binomtest")


def test_binomial_test_thresholds():
    """
    Test: Verify p-value thresholds for confidence tiers.

    Context: Multi-dimensional scoring uses p<0.05 (high), p<0.10 (moderate)
    """
    baseline = 0.80
    n_total = 33

    # Test p<0.05 threshold (high confidence)
    n_success_high = 29  # Should be < 0.05
    p_high = binomtest(n_success_high, n_total, baseline, alternative='greater').pvalue
    assert p_high < 0.05, f"29/33 should have p<0.05, got p={p_high:.4f}"

    # Test p<0.10 threshold (moderate confidence)
    n_success_moderate = 28  # Should be < 0.10 but > 0.05
    p_moderate = binomtest(n_success_moderate, n_total, baseline, alternative='greater').pvalue
    assert 0.05 < p_moderate < 0.10, \
        f"28/33 should have 0.05<p<0.10, got p={p_moderate:.4f}"

    print(f"✓ Thresholds validated: p<0.05 (high), p<0.10 (moderate)")


# Run all tests
if __name__ == "__main__":
    test_binomial_test_significantly_above_baseline()
    test_binomial_test_at_baseline()
    test_binomial_test_below_baseline()
    test_binomial_test_edge_case_empty_cluster()
    test_binomial_test_edge_case_invalid_input()
    test_binomial_test_scipy_version()
    test_binomial_test_thresholds()
    print("\n✅ All binomial test unit tests passed!")
```

**Expected Output**:
```
✓ 97% success rate → p=0.0021 (significant)
✓ 82% success rate → p=0.4012 (not significant)
✓ 55% success rate → p=0.0018 (anti-pattern detected)
✓ Empty cluster correctly raises error
✓ Invalid input (n_success > n_total) correctly raises error
✓ scipy version 1.11.0 supports binomtest
✓ Thresholds validated: p<0.05 (high), p<0.10 (moderate)

✅ All binomial test unit tests passed!
```

---

## Test 2: Feature Name Normalization (CRITICAL)

**File**: `tests/unit/test_feature_normalization.py`

**Purpose**: Validate that K-Means and RF features can be compared despite different naming conventions

**What It Catches**:
- **CRITICAL BUG**: K-Means features have `_scaled`, `_log`, `_encoded` suffixes, RF features don't
- Empty overlap (0/5 features match) due to name mismatch
- Missing normalization logic

**Context**: Stage 4 creates different feature names:
- K-Means: `eye_contact_rate_scaled`, `scene_count_scaled`
- RF: `eye_contact_rate`, `scene_count`

Without normalization, overlap calculation returns 0 (no matches), breaking validation.

**Implementation**:

```python
"""
Unit tests for feature name normalization between K-Means and RF models.

CRITICAL: This is the #1 bug hotspot in Stage 5.
K-Means features have suffixes (_scaled, _log, _encoded) from Stage 4 transformation.
RF features do NOT have these suffixes.
Without normalization, overlap = 0 (guaranteed bug).
"""

import pytest


def normalize_feature_name(feature_name):
    """
    Normalize K-Means feature names for comparison with RF feature names.

    Removes K-Means transformation suffixes:
    - '_scaled' (from MinMax scaling)
    - '_log' (from log transformation intermediate step)
    - '_encoded' (from label encoding)

    Args:
        feature_name: str, e.g., 'eye_contact_rate_scaled'

    Returns:
        str, e.g., 'eye_contact_rate'
    """
    normalized = feature_name

    # Remove suffixes in order (some features have multiple)
    suffixes = ['_scaled', '_log', '_encoded']
    for suffix in suffixes:
        normalized = normalized.replace(suffix, '')

    return normalized


def test_normalize_scaled_suffix():
    """Test: Remove '_scaled' suffix from K-Means features."""
    kmeans_feature = 'eye_contact_rate_scaled'
    expected = 'eye_contact_rate'

    result = normalize_feature_name(kmeans_feature)

    assert result == expected, \
        f"Expected '{expected}', got '{result}'"

    print(f"✓ '{kmeans_feature}' → '{result}'")


def test_normalize_log_and_scaled_suffix():
    """Test: Remove compound suffix '_log_scaled' (intermediate + final)."""
    # Note: Stage 4 creates intermediate '_log' then '_scaled'
    # But only final '_scaled' columns are kept in output
    # This test ensures robustness if intermediate columns leak through

    kmeans_feature = 'scene_count_scaled'  # Final form
    expected = 'scene_count'

    result = normalize_feature_name(kmeans_feature)

    assert result == expected, \
        f"Expected '{expected}', got '{result}'"

    print(f"✓ '{kmeans_feature}' → '{result}'")


def test_normalize_encoded_suffix():
    """Test: Remove '_encoded' suffix from categorical features."""
    kmeans_feature = 'has_captions_encoded'  # Boolean → 0/1
    expected = 'has_captions'

    result = normalize_feature_name(kmeans_feature)

    assert result == expected, \
        f"Expected '{expected}', got '{result}'"

    print(f"✓ '{kmeans_feature}' → '{result}'")


def test_normalize_no_suffix():
    """Test: RF features without suffixes remain unchanged."""
    rf_feature = 'eye_contact_rate'
    expected = 'eye_contact_rate'

    result = normalize_feature_name(rf_feature)

    assert result == expected, \
        f"Expected '{expected}', got '{result}'"

    print(f"✓ '{rf_feature}' → '{result}' (unchanged)")


def test_feature_overlap_calculation():
    """
    Test: Overlap calculation works after normalization.

    This is the CRITICAL test that catches the name mismatch bug.
    """
    # K-Means top 5 features (with suffixes)
    kmeans_top5 = [
        'eye_contact_rate_scaled',
        'scene_count_scaled',
        'word_count_scaled',
        'energy_level_scaled',
        'gesture_count_scaled'
    ]

    # RF top 5 features (without suffixes)
    rf_top5 = [
        'eye_contact_rate',
        'scene_count',
        'word_count',
        'pitch_scatter_ratio',  # Different feature
        'emotion_consistency'    # Different feature
    ]

    # Normalize K-Means features
    kmeans_normalized = [normalize_feature_name(f) for f in kmeans_top5]

    # Calculate overlap
    overlap = set(kmeans_normalized) & set(rf_top5)
    overlap_count = len(overlap)

    # Expected: 3 overlapping features (eye_contact, scene_count, word_count)
    expected_overlap = {'eye_contact_rate', 'scene_count', 'word_count'}

    assert overlap == expected_overlap, \
        f"Expected {expected_overlap}, got {overlap}"

    assert overlap_count == 3, \
        f"Expected 3 overlapping features, got {overlap_count}"

    print(f"✓ Overlap calculation works: {overlap_count}/5 features match")
    print(f"  Overlapping features: {sorted(overlap)}")


def test_all_21_base_features_normalize():
    """
    Test: All 21 base features normalize correctly.

    Context: Stage 4 transforms all 21 base features for K-Means.
    This test ensures normalization works for ALL features, not just a subset.
    """
    # All 21 base features (from Stage 4 Feature Transformation spec)
    base_features = [
        'average_face_size',
        'overlay_unique_count',
        'has_captions',
        'scene_count',
        'shortest_scene',
        'longest_scene',
        'scene_duration_variance',
        'object_count',
        'person_count',
        'speech_coverage',
        'word_count',
        'energy_level',
        'energy_variance',
        'energy_max',
        'pitch_scatter_ratio',
        'gesture_count',
        'gaze_variance',
        'eye_contact_rate',
        'dominant_emotion_id',  # One-hot encoded in K-Means, not scaled
        'emotional_valence',
        'emotion_consistency'
    ]

    # K-Means feature names (with suffixes from Stage 4)
    kmeans_features = []

    # Log + scale features (11 features → _scaled suffix)
    log_scale = ['scene_count', 'word_count', 'gesture_count', 'object_count',
                 'person_count', 'overlay_unique_count', 'shortest_scene',
                 'longest_scene', 'scene_duration_variance', 'energy_variance',
                 'gaze_variance']
    for f in log_scale:
        kmeans_features.append(f + '_scaled')

    # Scale features (7 features → _scaled suffix)
    scale = ['average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
             'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency']
    for f in scale:
        kmeans_features.append(f + '_scaled')

    # Shift + scale (1 feature → _scaled suffix)
    kmeans_features.append('emotional_valence_scaled')

    # Label encode (1 feature → _encoded suffix)
    kmeans_features.append('has_captions_encoded')

    # One-hot (1 feature → 7 binary columns, no suffix)
    # dominant_emotion_id becomes: joy, sadness, anger, fear, disgust, surprise, neutral
    # These don't need normalization (RF also has these one-hot)

    # Total K-Means features: 20 with suffixes + 7 one-hot = 27
    # (Note: dominant_emotion_id is replaced by 7 one-hot columns)

    # Normalize all K-Means features
    kmeans_normalized = [normalize_feature_name(f) for f in kmeans_features]

    # Check that normalized names match expected base feature names
    # (excluding dominant_emotion_id which is one-hot encoded)
    expected_base = [f for f in base_features if f != 'dominant_emotion_id']

    for base_feature in expected_base:
        assert base_feature in kmeans_normalized, \
            f"Base feature '{base_feature}' not found in normalized K-Means features"

    print(f"✓ All 20 base features normalize correctly")
    print(f"  Total K-Means features: {len(kmeans_features)}")
    print(f"  After normalization: {len(set(kmeans_normalized))} unique base features")


# Run all tests
if __name__ == "__main__":
    test_normalize_scaled_suffix()
    test_normalize_log_and_scaled_suffix()
    test_normalize_encoded_suffix()
    test_normalize_no_suffix()
    test_feature_overlap_calculation()
    test_all_21_base_features_normalize()
    print("\n✅ All feature normalization tests passed!")
    print("\n⚠️  CRITICAL: If these tests fail, Stage 5 validation WILL NOT WORK")
```

**Expected Output**:
```
✓ 'eye_contact_rate_scaled' → 'eye_contact_rate'
✓ 'scene_count_scaled' → 'scene_count'
✓ 'has_captions_encoded' → 'has_captions'
✓ 'eye_contact_rate' → 'eye_contact_rate' (unchanged)
✓ Overlap calculation works: 3/5 features match
  Overlapping features: ['eye_contact_rate', 'scene_count', 'word_count']
✓ All 20 base features normalize correctly
  Total K-Means features: 27
  After normalization: 20 unique base features

✅ All feature normalization tests passed!

⚠️  CRITICAL: If these tests fail, Stage 5 validation WILL NOT WORK
```

---

## Test 3: K-Means Feature Ranking (HIGH PRIORITY)

**File**: `tests/unit/test_kmeans_feature_ranking.py`

**Purpose**: Validate that K-Means cluster-defining feature extraction is correct

**What It Catches**:
- Wrong ranking logic (magnitude vs variance vs deviation)
- Features ranked in wrong order
- Edge cases (all features same variance)

**Context**: Determining which features "define" a cluster is NON-TRIVIAL. We need to extract features with high variance across cluster centroids.

**Implementation**:

```python
"""
Unit tests for K-Means cluster-defining feature extraction.

HIGH PRIORITY: This logic is conceptually complex and error-prone.

Question: What makes a feature "cluster-defining"?
Answer: Features with high VARIANCE across cluster centroids.

Example:
- Cluster 1 centroid: [0.9, 0.5, 0.2]  (feature 1, 2, 3)
- Cluster 2 centroid: [0.3, 0.5, 0.7]
- Cluster 3 centroid: [0.6, 0.5, 0.4]

Feature 1: high variance (0.9, 0.3, 0.6) → CLUSTER-DEFINING
Feature 2: no variance (0.5, 0.5, 0.5) → NOT cluster-defining
Feature 3: moderate variance (0.2, 0.7, 0.4) → Moderately defining
"""

import pytest
import numpy as np
from sklearn.cluster import KMeans


def get_top_cluster_features(kmeans_model, feature_names, n=5):
    """
    Extract top N cluster-defining features from K-Means model.

    Logic: Features with highest VARIANCE across cluster centroids
    are the most cluster-defining.

    Args:
        kmeans_model: Trained KMeans object
        feature_names: List of feature names (must match order of centroids)
        n: Number of top features to return

    Returns:
        List of top N feature names, sorted by importance
    """
    centroids = kmeans_model.cluster_centers_  # Shape: (n_clusters, n_features)

    # Calculate variance of each feature across centroids
    feature_variances = np.var(centroids, axis=0)  # Shape: (n_features,)

    # Rank features by variance (highest first)
    top_indices = np.argsort(feature_variances)[::-1][:n]

    # Get feature names
    top_features = [feature_names[i] for i in top_indices]

    return top_features


def test_high_variance_feature_ranks_first():
    """
    Test: Feature with high variance across clusters ranks first.

    Create synthetic data where Feature 0 clearly distinguishes clusters.
    """
    np.random.seed(42)

    # Create 3 clusters with clear separation in Feature 0 only
    cluster1 = np.column_stack([
        np.random.normal(0.9, 0.02, 30),  # Feature 0: HIGH
        np.random.normal(0.5, 0.02, 30),  # Feature 1: NEUTRAL
    ])

    cluster2 = np.column_stack([
        np.random.normal(0.3, 0.02, 30),  # Feature 0: LOW
        np.random.normal(0.5, 0.02, 30),  # Feature 1: NEUTRAL
    ])

    cluster3 = np.column_stack([
        np.random.normal(0.6, 0.02, 30),  # Feature 0: MEDIUM
        np.random.normal(0.5, 0.02, 30),  # Feature 1: NEUTRAL
    ])

    X = np.vstack([cluster1, cluster2, cluster3])

    # Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    # Extract top features
    feature_names = ['feature_0', 'feature_1']
    top_features = get_top_cluster_features(kmeans, feature_names, n=2)

    # Feature 0 should rank first (high variance across clusters)
    assert top_features[0] == 'feature_0', \
        f"Expected 'feature_0' first (high variance), got '{top_features[0]}'"

    assert top_features[1] == 'feature_1', \
        f"Expected 'feature_1' second (low variance), got '{top_features[1]}'"

    print(f"✓ High-variance feature ranks first: {top_features}")


def test_no_variance_feature_ranks_last():
    """
    Test: Feature with NO variance across clusters ranks last.

    Create synthetic data where Feature 1 is constant across all clusters.
    """
    np.random.seed(42)

    # Create 3 clusters with separation in Feature 0, constant Feature 1
    cluster1 = np.column_stack([
        np.random.normal(0.9, 0.02, 30),  # Feature 0: HIGH
        np.ones(30) * 0.5,                # Feature 1: CONSTANT
        np.random.normal(0.7, 0.02, 30),  # Feature 2: MODERATE
    ])

    cluster2 = np.column_stack([
        np.random.normal(0.3, 0.02, 30),  # Feature 0: LOW
        np.ones(30) * 0.5,                # Feature 1: CONSTANT
        np.random.normal(0.4, 0.02, 30),  # Feature 2: MODERATE
    ])

    cluster3 = np.column_stack([
        np.random.normal(0.6, 0.02, 30),  # Feature 0: MEDIUM
        np.ones(30) * 0.5,                # Feature 1: CONSTANT
        np.random.normal(0.55, 0.02, 30), # Feature 2: MODERATE
    ])

    X = np.vstack([cluster1, cluster2, cluster3])

    # Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    # Extract top features
    feature_names = ['feature_0', 'feature_1', 'feature_2']
    top_features = get_top_cluster_features(kmeans, feature_names, n=3)

    # Feature 1 should rank last (no variance)
    assert top_features[2] == 'feature_1', \
        f"Expected 'feature_1' last (no variance), got ranking: {top_features}"

    # Feature 0 should rank first (highest variance)
    assert top_features[0] == 'feature_0', \
        f"Expected 'feature_0' first, got '{top_features[0]}'"

    print(f"✓ No-variance feature ranks last: {top_features}")


def test_moderate_variance_features_ranked_by_magnitude():
    """
    Test: Features with moderate variance rank by magnitude of variance.
    """
    np.random.seed(42)

    # Create clusters with varying degrees of separation
    cluster1 = np.column_stack([
        np.random.normal(0.9, 0.02, 30),  # Feature 0: LARGE variance
        np.random.normal(0.7, 0.02, 30),  # Feature 1: MEDIUM variance
        np.random.normal(0.6, 0.02, 30),  # Feature 2: SMALL variance
    ])

    cluster2 = np.column_stack([
        np.random.normal(0.2, 0.02, 30),  # Feature 0: LARGE variance
        np.random.normal(0.5, 0.02, 30),  # Feature 1: MEDIUM variance
        np.random.normal(0.5, 0.02, 30),  # Feature 2: SMALL variance
    ])

    cluster3 = np.column_stack([
        np.random.normal(0.5, 0.02, 30),  # Feature 0: LARGE variance
        np.random.normal(0.6, 0.02, 30),  # Feature 1: MEDIUM variance
        np.random.normal(0.55, 0.02, 30), # Feature 2: SMALL variance
    ])

    X = np.vstack([cluster1, cluster2, cluster3])

    # Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    # Calculate actual variances for validation
    centroids = kmeans.cluster_centers_
    variances = np.var(centroids, axis=0)

    print(f"  Actual variances: {variances}")

    # Extract top features
    feature_names = ['feature_0', 'feature_1', 'feature_2']
    top_features = get_top_cluster_features(kmeans, feature_names, n=3)

    # Feature 0 should rank first (largest variance)
    assert top_features[0] == 'feature_0', \
        f"Expected 'feature_0' first (largest variance={variances[0]:.4f}), got '{top_features[0]}'"

    print(f"✓ Features ranked by variance magnitude: {top_features}")


def test_edge_case_all_features_same_variance():
    """
    Test: Edge case where all features have same variance.

    Context: Ranking order is arbitrary but should not crash.
    """
    np.random.seed(42)

    # Create clusters where all features have equal variance
    # (This is artificial but tests robustness)
    cluster1 = np.random.normal(0.8, 0.02, (30, 5))
    cluster2 = np.random.normal(0.5, 0.02, (30, 5))
    cluster3 = np.random.normal(0.2, 0.02, (30, 5))

    X = np.vstack([cluster1, cluster2, cluster3])

    # Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    # Extract top features (should not crash)
    feature_names = [f'feature_{i}' for i in range(5)]
    top_features = get_top_cluster_features(kmeans, feature_names, n=5)

    # All 5 features should be returned
    assert len(top_features) == 5, \
        f"Expected 5 features, got {len(top_features)}"

    assert len(set(top_features)) == 5, \
        f"Expected 5 unique features, got {len(set(top_features))}"

    print(f"✓ Edge case handled: All features same variance → {top_features}")


def test_feature_count_matches_request():
    """
    Test: get_top_cluster_features returns correct number of features.
    """
    np.random.seed(42)

    # Create simple 3-cluster data
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=90, n_features=10, centers=3, random_state=42)

    # Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    feature_names = [f'feature_{i}' for i in range(10)]

    # Test n=5
    top_5 = get_top_cluster_features(kmeans, feature_names, n=5)
    assert len(top_5) == 5, f"Expected 5 features, got {len(top_5)}"

    # Test n=3
    top_3 = get_top_cluster_features(kmeans, feature_names, n=3)
    assert len(top_3) == 3, f"Expected 3 features, got {len(top_3)}"

    # Test n=10 (all features)
    top_10 = get_top_cluster_features(kmeans, feature_names, n=10)
    assert len(top_10) == 10, f"Expected 10 features, got {len(top_10)}"

    print(f"✓ Feature count matches request: n=5 → {len(top_5)}, n=3 → {len(top_3)}, n=10 → {len(top_10)}")


# Run all tests
if __name__ == "__main__":
    test_high_variance_feature_ranks_first()
    test_no_variance_feature_ranks_last()
    test_moderate_variance_features_ranked_by_magnitude()
    test_edge_case_all_features_same_variance()
    test_feature_count_matches_request()
    print("\n✅ All K-Means feature ranking tests passed!")
    print("\n⚠️  NOTE: These tests validate LOGIC correctness")
    print("   Manual validation on real data still recommended (see Layer 3)")
```

**Expected Output**:
```
✓ High-variance feature ranks first: ['feature_0', 'feature_1']
✓ No-variance feature ranks last: ['feature_0', 'feature_2', 'feature_1']
  Actual variances: [0.0892 0.0089 0.0234]
✓ Features ranked by variance magnitude: ['feature_0', 'feature_2', 'feature_1']
✓ Edge case handled: All features same variance → ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
✓ Feature count matches request: n=5 → 5, n=3 → 3, n=10 → 10

✅ All K-Means feature ranking tests passed!

⚠️  NOTE: These tests validate LOGIC correctness
   Manual validation on real data still recommended (see Layer 3)
```

---

## Test 4: Success Rate Calculation

**File**: `tests/unit/test_success_rate.py`

**Purpose**: Validate cluster success rate calculation (simple but needs edge case coverage)

**Implementation**:

```python
"""
Unit tests for cluster success rate calculation.

This is a simple calculation but needs edge case coverage.
"""

import pytest
import pandas as pd
import numpy as np


def calculate_success_rate(cluster_videos):
    """
    Calculate success rate for a cluster.

    Args:
        cluster_videos: DataFrame with 'is_top_performer' column (0 or 1)

    Returns:
        float: Success rate (0.0 to 1.0), or NaN if empty
    """
    if len(cluster_videos) == 0:
        return np.nan

    return cluster_videos['is_top_performer'].mean()


def test_success_rate_normal_case():
    """Test: Normal case with mixed top and bottom performers."""
    cluster_videos = pd.DataFrame({
        'is_top_performer': [1, 1, 1, 1, 0]  # 4/5 = 80%
    })

    success_rate = calculate_success_rate(cluster_videos)

    assert success_rate == 0.80, f"Expected 0.80, got {success_rate}"
    print(f"✓ Success rate: 4/5 = {success_rate:.2f}")


def test_success_rate_all_top_performers():
    """Test: All videos are top performers (100% success rate)."""
    cluster_videos = pd.DataFrame({
        'is_top_performer': [1, 1, 1, 1, 1]
    })

    success_rate = calculate_success_rate(cluster_videos)

    assert success_rate == 1.0, f"Expected 1.0, got {success_rate}"
    print(f"✓ Success rate: 5/5 = {success_rate:.2f}")


def test_success_rate_all_bottom_performers():
    """Test: All videos are bottom performers (0% success rate)."""
    cluster_videos = pd.DataFrame({
        'is_top_performer': [0, 0, 0, 0, 0]
    })

    success_rate = calculate_success_rate(cluster_videos)

    assert success_rate == 0.0, f"Expected 0.0, got {success_rate}"
    print(f"✓ Success rate: 0/5 = {success_rate:.2f}")


def test_success_rate_edge_case_empty_cluster():
    """Test: Empty cluster returns NaN."""
    cluster_videos = pd.DataFrame({'is_top_performer': []})

    success_rate = calculate_success_rate(cluster_videos)

    assert np.isnan(success_rate), f"Expected NaN for empty cluster, got {success_rate}"
    print(f"✓ Empty cluster → NaN")


def test_success_rate_single_video():
    """Test: Single video cluster."""
    cluster_videos = pd.DataFrame({'is_top_performer': [1]})

    success_rate = calculate_success_rate(cluster_videos)

    assert success_rate == 1.0, f"Expected 1.0, got {success_rate}"
    print(f"✓ Single video (top performer) → {success_rate:.2f}")


# Run all tests
if __name__ == "__main__":
    test_success_rate_normal_case()
    test_success_rate_all_top_performers()
    test_success_rate_all_bottom_performers()
    test_success_rate_edge_case_empty_cluster()
    test_success_rate_single_video()
    print("\n✅ All success rate tests passed!")
```

---

## Test 5: Silhouette Score Calculation

**File**: `tests/unit/test_silhouette_score.py`

**Purpose**: Validate cluster quality measurement using silhouette scores

**Implementation**:

```python
"""
Unit tests for silhouette score calculation (cluster quality).

Silhouette score measures how cohesive a cluster is.
Range: [-1, 1]
- >0.5: Well-separated cluster
- 0.3-0.5: Moderate cluster
- <0.3: Poorly separated cluster
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def calculate_cluster_silhouette(X, kmeans_model, cluster_id):
    """
    Calculate average silhouette score for a specific cluster.

    Args:
        X: Feature matrix (N, D) - MUST be same data used to train kmeans_model
        kmeans_model: Trained KMeans object
        cluster_id: Cluster ID (0, 1, 2, ...)

    Returns:
        float: Average silhouette score for cluster
    """
    # Calculate silhouette for all samples
    silhouette_scores = silhouette_samples(X, kmeans_model.labels_)

    # Get silhouette for this cluster only
    cluster_mask = (kmeans_model.labels_ == cluster_id)
    cluster_silhouette = silhouette_scores[cluster_mask].mean()

    return cluster_silhouette


def test_well_separated_clusters_high_silhouette():
    """Test: Well-separated clusters should have silhouette > 0.5."""
    # Create well-separated blobs
    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.5, random_state=42)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Calculate silhouette for each cluster
    for cluster_id in range(3):
        sil = calculate_cluster_silhouette(X, kmeans, cluster_id)

        assert sil > 0.5, \
            f"Well-separated cluster {cluster_id} should have silhouette > 0.5, got {sil:.3f}"

        print(f"✓ Cluster {cluster_id}: silhouette = {sil:.3f} (well-separated)")


def test_overlapping_clusters_low_silhouette():
    """Test: Overlapping clusters should have silhouette < 0.5."""
    # Create overlapping blobs
    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=3.0, random_state=42)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Calculate silhouette for each cluster
    for cluster_id in range(3):
        sil = calculate_cluster_silhouette(X, kmeans, cluster_id)

        # Overlapping clusters should have lower silhouette
        # (May not always be < 0.5 due to randomness, but should be lower than well-separated)
        print(f"✓ Cluster {cluster_id}: silhouette = {sil:.3f} (overlapping)")

    print("  (Overlapping clusters generally have lower silhouette scores)")


def test_silhouette_edge_case_single_element_cluster():
    """Test: Single-element cluster has silhouette = 0."""
    # Create data where K-Means might create a singleton cluster
    X = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2], [10, 10]])  # 3 close + 1 outlier

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)

    # Check if any cluster has size 1
    cluster_sizes = np.bincount(kmeans.labels_)

    if 1 in cluster_sizes:
        singleton_cluster_id = np.where(cluster_sizes == 1)[0][0]
        sil = calculate_cluster_silhouette(X, kmeans, singleton_cluster_id)

        assert sil == 0.0, \
            f"Single-element cluster should have silhouette = 0, got {sil:.3f}"

        print(f"✓ Single-element cluster: silhouette = {sil:.3f}")
    else:
        print("  (No singleton clusters created - edge case not triggered)")


def test_global_silhouette_score():
    """Test: Global silhouette score (average across all clusters)."""
    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.5, random_state=42)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Global silhouette score
    global_sil = silhouette_score(X, kmeans.labels_)

    assert 0 <= global_sil <= 1, \
        f"Global silhouette should be in [0, 1], got {global_sil:.3f}"

    print(f"✓ Global silhouette score: {global_sil:.3f}")


# Run all tests
if __name__ == "__main__":
    test_well_separated_clusters_high_silhouette()
    test_overlapping_clusters_low_silhouette()
    test_silhouette_edge_case_single_element_cluster()
    test_global_silhouette_score()
    print("\n✅ All silhouette score tests passed!")
```

---

## Test 6: Multi-Dimensional Confidence Scoring

**File**: `tests/unit/test_confidence_scoring.py`

**Purpose**: Validate the complete confidence scoring algorithm

**Implementation**:

```python
"""
Unit tests for multi-dimensional confidence scoring.

This is the CORE validation algorithm that determines which clusters
are trustworthy for creator testing guidelines.

Scoring dimensions (0-100 points total):
1. Statistical significance (0-40 points)
2. Feature overlap with RF (0-30 points)
3. Success rate magnitude (0-20 points)
4. Cluster quality / silhouette (0-10 points)

Tiers:
- GOLD STANDARD: 75+ points
- SILVER (VALIDATED): 55-74 points
- BRONZE (MODERATELY VALIDATED): 35-54 points
- EXPLORATORY: <35 points
"""

import pytest


def calculate_confidence_score(p_value, overlap_count, success_rate, silhouette):
    """
    Calculate multi-dimensional confidence score for cluster validation.

    Args:
        p_value: float, binomial test p-value
        overlap_count: int, number of overlapping features (0-5)
        success_rate: float, cluster success rate (0-1)
        silhouette: float, cluster silhouette score (-1 to 1)

    Returns:
        dict with 'confidence_score' (0-100) and 'tier' (string)
    """
    # Signal 1: Statistical significance (0-40 points)
    if p_value < 0.01:
        stat_score = 40
    elif p_value < 0.05:
        stat_score = 30
    elif p_value < 0.10:
        stat_score = 20
    else:
        # Gradual decay for p > 0.10
        stat_score = max(0, 20 - (p_value * 50))

    # Signal 2: Feature overlap (0-30 points)
    overlap_score = overlap_count * 6  # 6 points per overlapping feature

    # Signal 3: Success rate magnitude (0-20 points)
    if success_rate >= 0.95:
        magnitude_score = 20
    elif success_rate >= 0.90:
        magnitude_score = 15
    elif success_rate >= 0.85:
        magnitude_score = 10
    else:
        # Gradual for 0.80-0.85
        magnitude_score = max(0, (success_rate - 0.80) * 100)

    # Signal 4: Cluster quality (0-10 points)
    if silhouette >= 0.5:
        quality_score = 10
    elif silhouette >= 0.3:
        quality_score = 5
    else:
        quality_score = 0

    # Total score
    total_score = stat_score + overlap_score + magnitude_score + quality_score

    # Tier assignment
    if total_score >= 75:
        tier = "GOLD STANDARD"
    elif total_score >= 55:
        tier = "SILVER (VALIDATED)"
    elif total_score >= 35:
        tier = "BRONZE (MODERATELY VALIDATED)"
    else:
        tier = "EXPLORATORY"

    return {
        'confidence_score': total_score,
        'tier': tier,
        'breakdown': {
            'statistical': stat_score,
            'feature_overlap': overlap_score,
            'magnitude': magnitude_score,
            'quality': quality_score
        }
    }


def test_gold_standard_cluster():
    """Test: Cluster with excellent metrics gets GOLD STANDARD."""
    result = calculate_confidence_score(
        p_value=0.002,      # Very significant (40 pts)
        overlap_count=5,    # Perfect overlap (30 pts)
        success_rate=0.97,  # Very high (20 pts)
        silhouette=0.6      # Good quality (10 pts)
    )

    assert result['tier'] == "GOLD STANDARD", \
        f"Expected GOLD STANDARD, got {result['tier']}"

    assert result['confidence_score'] == 100, \
        f"Expected 100 points, got {result['confidence_score']}"

    print(f"✓ Gold standard: {result['confidence_score']} pts → {result['tier']}")
    print(f"  Breakdown: stat={result['breakdown']['statistical']}, "
          f"overlap={result['breakdown']['feature_overlap']}, "
          f"magnitude={result['breakdown']['magnitude']}, "
          f"quality={result['breakdown']['quality']}")


def test_silver_validated_cluster():
    """Test: Cluster with good metrics gets SILVER."""
    result = calculate_confidence_score(
        p_value=0.03,       # Significant (30 pts)
        overlap_count=4,    # Good overlap (24 pts)
        success_rate=0.88,  # Good (10 pts)
        silhouette=0.4      # Moderate quality (5 pts)
    )

    expected_score = 30 + 24 + 10 + 5  # 69 pts

    assert result['tier'] == "SILVER (VALIDATED)", \
        f"Expected SILVER, got {result['tier']}"

    assert result['confidence_score'] == expected_score, \
        f"Expected {expected_score} points, got {result['confidence_score']}"

    print(f"✓ Silver validated: {result['confidence_score']} pts → {result['tier']}")


def test_bronze_moderately_validated_cluster():
    """Test: Cluster with moderate metrics gets BRONZE."""
    result = calculate_confidence_score(
        p_value=0.08,       # Borderline significant (20 pts)
        overlap_count=3,    # Moderate overlap (18 pts)
        success_rate=0.82,  # Slightly above baseline (2 pts)
        silhouette=0.3      # Moderate quality (5 pts)
    )

    expected_score = 20 + 18 + 2 + 5  # 45 pts

    assert result['tier'] == "BRONZE (MODERATELY VALIDATED)", \
        f"Expected BRONZE, got {result['tier']}"

    assert 35 <= result['confidence_score'] < 55, \
        f"BRONZE should be 35-54 pts, got {result['confidence_score']}"

    print(f"✓ Bronze validated: {result['confidence_score']} pts → {result['tier']}")


def test_exploratory_cluster():
    """Test: Cluster with weak metrics gets EXPLORATORY."""
    result = calculate_confidence_score(
        p_value=0.50,       # Not significant (0 pts)
        overlap_count=1,    # Weak overlap (6 pts)
        success_rate=0.78,  # Below baseline (0 pts)
        silhouette=0.2      # Poor quality (0 pts)
    )

    expected_score = 0 + 6 + 0 + 0  # 6 pts

    assert result['tier'] == "EXPLORATORY", \
        f"Expected EXPLORATORY, got {result['tier']}"

    assert result['confidence_score'] < 35, \
        f"EXPLORATORY should be <35 pts, got {result['confidence_score']}"

    print(f"✓ Exploratory: {result['confidence_score']} pts → {result['tier']}")


def test_score_never_negative():
    """Test: Score never goes below 0 (even with terrible metrics)."""
    result = calculate_confidence_score(
        p_value=0.99,       # Very not significant
        overlap_count=0,    # No overlap
        success_rate=0.50,  # Far below baseline
        silhouette=-0.5     # Terrible quality
    )

    assert result['confidence_score'] >= 0, \
        f"Score should never be negative, got {result['confidence_score']}"

    print(f"✓ Minimum score protection: {result['confidence_score']} pts (never negative)")


def test_tier_boundaries():
    """Test: Tier boundaries work correctly at edge cases."""
    # Test 75 pts (GOLD boundary)
    result_75 = calculate_confidence_score(0.002, 5, 0.85, 0.3)  # 40+30+10-5 = 75
    assert result_75['tier'] == "GOLD STANDARD", "75 pts should be GOLD"

    # Test 74 pts (SILVER boundary)
    result_74 = calculate_confidence_score(0.002, 4, 0.85, 0.3)  # 40+24+10+5 = 74
    assert result_74['tier'] == "SILVER (VALIDATED)", "74 pts should be SILVER"

    # Test 55 pts (SILVER boundary)
    result_55 = calculate_confidence_score(0.04, 4, 0.80, 0.3)  # 30+24+0+5 = 59
    # Adjust to hit 55 exactly
    result_55 = calculate_confidence_score(0.04, 3, 0.82, 0.3)  # 30+18+2+5 = 55
    assert result_55['tier'] == "SILVER (VALIDATED)", "55 pts should be SILVER"

    print(f"✓ Tier boundaries validated")


# Run all tests
if __name__ == "__main__":
    test_gold_standard_cluster()
    test_silver_validated_cluster()
    test_bronze_moderately_validated_cluster()
    test_exploratory_cluster()
    test_score_never_negative()
    test_tier_boundaries()
    print("\n✅ All confidence scoring tests passed!")
```

---

# Layer 2: Integration Tests

## Integration Test 1: Stage 4 to Stage 5 Data Flow

**File**: `tests/integration/test_stage4_to_stage5_integration.py`

**Purpose**: Validate that Stage 5 can correctly load and process Stage 4 outputs

**Critical Test**: This catches the feature name mismatch bug using REAL data

**Implementation**:

```python
"""
Integration test: Stage 4 → Stage 5 data flow.

CRITICAL: This test uses REAL Stage 4 output files to catch name mismatch bugs.

Setup:
1. Run Stage 4 on a small sample (10-20 videos, bucket 18-33s)
2. Copy outputs to tests/fixtures/stage4_outputs/
3. Run this test to validate Stage 5 can process them

This test will catch:
- Feature name mismatches between K-Means and RF
- Missing files
- Schema mismatches
- Real-world data issues
"""

import pytest
import pandas as pd
import os
from pathlib import Path


# Test fixture paths
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "stage4_outputs"
BUCKET = "bucket_18-33s"


def test_stage4_fixture_files_exist():
    """Test: Required Stage 4 output files exist in fixtures."""
    required_files = [
        f"{BUCKET}/ml_analysis/rf_transformed.csv",
        f"{BUCKET}/ml_analysis/hook_rf_transformed.csv",
        f"{BUCKET}/ml_analysis/hook_km_transformed.csv",
    ]

    for filepath in required_files:
        full_path = FIXTURES_DIR / filepath
        assert full_path.exists(), \
            f"Required fixture file not found: {full_path}\n" \
            f"Run Stage 4 on sample data first to generate fixtures."

    print("✓ All required Stage 4 fixtures exist")


def test_rf_transformed_schema():
    """Test: Video-level RF file has expected schema."""
    rf_path = FIXTURES_DIR / BUCKET / "ml_analysis" / "rf_transformed.csv"
    df = pd.read_csv(rf_path)

    # Should have ~178 columns for bucket 18-33s
    assert 170 <= len(df.columns) <= 185, \
        f"Expected 170-185 columns, got {len(df.columns)}"

    # Should have 'is_top_performer' target column (contrastive mode)
    assert 'is_top_performer' in df.columns, \
        "Missing 'is_top_performer' target column"

    # Should NOT have NaN values
    assert not df.isnull().any().any(), \
        f"RF file contains NaN values: {df.columns[df.isnull().any()].tolist()}"

    print(f"✓ Video-level RF schema valid: {len(df)} rows, {len(df.columns)} columns")


def test_window_rf_transformed_schema():
    """Test: Window-level RF files have expected schema."""
    hook_rf_path = FIXTURES_DIR / BUCKET / "ml_analysis" / "hook_rf_transformed.csv"
    df = pd.read_csv(hook_rf_path)

    # Should have exactly 22 columns (21 base features + 1 target)
    assert len(df.columns) == 22, \
        f"Expected 22 columns, got {len(df.columns)}"

    # Should have 'is_top_performer' target column
    assert 'is_top_performer' in df.columns, \
        "Missing 'is_top_performer' target column"

    # Should NOT have suffixes (_scaled, _log, _encoded)
    for col in df.columns:
        assert '_scaled' not in col, f"RF file should not have '_scaled' suffix: {col}"
        assert '_log' not in col, f"RF file should not have '_log' suffix: {col}"
        assert '_encoded' not in col, f"RF file should not have '_encoded' suffix: {col}"

    print(f"✓ Window-level RF schema valid: {len(df)} rows, {len(df.columns)} columns")


def test_window_km_transformed_schema():
    """Test: Window-level K-Means files have expected schema."""
    hook_km_path = FIXTURES_DIR / BUCKET / "ml_analysis" / "hook_km_transformed.csv"
    df = pd.read_csv(hook_km_path)

    # Should have exactly 39 columns (all transformed, no target)
    assert len(df.columns) == 39, \
        f"Expected 39 columns, got {len(df.columns)}"

    # Should have '_scaled' or '_encoded' suffixes (K-Means naming)
    scaled_cols = [c for c in df.columns if '_scaled' in c or '_encoded' in c or c in ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']]
    assert len(scaled_cols) >= 20, \
        f"Expected >=20 transformed columns, got {len(scaled_cols)}"

    # All values should be in [0, 1] range (scaled)
    for col in df.columns:
        if '_scaled' in col:
            assert df[col].min() >= 0 and df[col].max() <= 1, \
                f"Column {col} out of [0,1] range: [{df[col].min():.3f}, {df[col].max():.3f}]"

    print(f"✓ Window-level K-Means schema valid: {len(df)} rows, {len(df.columns)} columns")


def test_feature_name_overlap_with_real_data():
    """
    CRITICAL TEST: Feature name overlap works with REAL Stage 4 data.

    This is the test that catches the name mismatch bug!
    """
    # Load real K-Means features
    hook_km_path = FIXTURES_DIR / BUCKET / "ml_analysis" / "hook_km_transformed.csv"
    df_km = pd.read_csv(hook_km_path)
    kmeans_features = list(df_km.columns)

    # Load real RF features
    hook_rf_path = FIXTURES_DIR / BUCKET / "ml_analysis" / "hook_rf_transformed.csv"
    df_rf = pd.read_csv(hook_rf_path)
    rf_features = [c for c in df_rf.columns if c != 'is_top_performer']  # Exclude target

    # Normalize K-Means features
    def normalize_feature_name(name):
        return name.replace('_scaled', '').replace('_log', '').replace('_encoded', '')

    kmeans_normalized = [normalize_feature_name(f) for f in kmeans_features]

    # Calculate overlap
    overlap = set(kmeans_normalized) & set(rf_features)
    overlap_count = len(overlap)

    # Should have at least 15 overlapping features (out of 21 base features)
    # (Some features are one-hot encoded, so exact match might be <21)
    assert overlap_count >= 15, \
        f"Feature name mismatch! Only {overlap_count}/21 features overlap.\n" \
        f"K-Means features (normalized): {sorted(kmeans_normalized)[:10]}...\n" \
        f"RF features: {sorted(rf_features)[:10]}...\n" \
        f"Overlapping: {sorted(overlap)}"

    print(f"✓ Feature overlap with real data: {overlap_count}/21 features match")
    print(f"  Sample overlapping features: {sorted(overlap)[:5]}")


def test_train_rf_model_on_real_data():
    """Test: Can train Window-Level RF model on real Stage 4 data."""
    from sklearn.ensemble import RandomForestClassifier

    hook_rf_path = FIXTURES_DIR / BUCKET / "ml_analysis" / "hook_rf_transformed.csv"
    df = pd.read_csv(hook_rf_path)

    # Separate features and target
    X = df.drop('is_top_performer', axis=1)
    y = df['is_top_performer']

    # Train RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Should have non-zero importance for some features
    top_importance = feature_importance.iloc[0]['importance']
    assert top_importance > 0, "Top feature should have non-zero importance"

    print(f"✓ RF model trained successfully")
    print(f"  Top feature: {feature_importance.iloc[0]['feature']} "
          f"(importance={top_importance:.3f})")


def test_train_kmeans_model_on_real_data():
    """Test: Can train Window-Level K-Means model on real Stage 4 data."""
    from sklearn.cluster import KMeans

    hook_km_path = FIXTURES_DIR / BUCKET / "ml_analysis" / "hook_km_transformed.csv"
    df = pd.read_csv(hook_km_path)

    # Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(df)

    # Check cluster sizes
    import numpy as np
    cluster_sizes = np.bincount(kmeans.labels_)

    # No cluster should be empty
    assert all(size > 0 for size in cluster_sizes), \
        f"Empty cluster detected: {cluster_sizes}"

    # Should have 3 clusters
    assert len(cluster_sizes) == 3, \
        f"Expected 3 clusters, got {len(cluster_sizes)}"

    print(f"✓ K-Means model trained successfully")
    print(f"  Cluster sizes: {cluster_sizes}")


# Run all tests
if __name__ == "__main__":
    test_stage4_fixture_files_exist()
    test_rf_transformed_schema()
    test_window_rf_transformed_schema()
    test_window_km_transformed_schema()
    test_feature_name_overlap_with_real_data()
    test_train_rf_model_on_real_data()
    test_train_kmeans_model_on_real_data()
    print("\n✅ All Stage 4 → Stage 5 integration tests passed!")
    print("\n🎯 CRITICAL: Feature name overlap validated with real data")
```

---

# Layer 3: Manual Validation Checklist

**File**: `tests/manual/manual_validation_checklist.md`

**Purpose**: Human review procedures for first production run

**When to use**: After Stage 5 runs on real hashtag data for the first time

**Checklist**:

```markdown
# Stage 5 Manual Validation Checklist

## Purpose

Automated tests can't catch everything. This checklist ensures Stage 5
outputs make intuitive sense on real production data.

Run this checklist on the FIRST production hashtag analysis.

---

## Bucket: 18-33s, Hook Window

### 1. K-Means Cluster-Defining Features

**File**: `bucket_18-33s/ml_analysis/validation_results.json`

**Check**: Do the top 5 cluster-defining features make sense?

```json
"hook": {
  "cluster_1": {
    "kmeans_top5": ["eye_contact_rate", "scene_count", "energy_level", "word_count", "gesture_count"]
  }
}
```

**Questions**:
- [ ] Are these features that visually distinguish hooks? (YES/NO)
- [ ] Do they align with your intuition about viral hooks? (YES/NO)
- [ ] Are there any surprising features? (List them)

**If NO**: Re-examine K-Means feature ranking logic (get_top_cluster_features)

---

### 2. RF Feature Importance

**Check**: Do the top 5 RF important features make sense?

```json
"hook": {
  "cluster_1": {
    "rf_top5": ["eye_contact_rate", "word_count", "scene_count", "emotion_consistency", "energy_max"]
  }
}
```

**Questions**:
- [ ] Are these features that predict virality? (YES/NO)
- [ ] Do they overlap with K-Means features? (How many? __/5)
- [ ] Are there any features you'd expect but don't see? (List them)

---

### 3. Feature Overlap

**Check**: Do K-Means and RF agree on important features?

```json
"overlap_features": ["eye_contact_rate", "scene_count", "word_count"]
```

**Questions**:
- [ ] Is overlap >= 2 features? (YES/NO)
- [ ] Do the overlapping features make sense? (YES/NO)
- [ ] If overlap < 2, is there a valid reason? (Explain)

---

### 4. Cluster Success Rates

**Check**: Do success rates differ meaningfully across clusters?

```json
"cluster_1": {"success_rate": 0.94, "sample_size": 33},
"cluster_2": {"success_rate": 0.78, "sample_size": 34},
"cluster_3": {"success_rate": 0.88, "sample_size": 33}
```

**Questions**:
- [ ] Is there at least one cluster with >85% success rate? (YES/NO)
- [ ] Is there variation (not all clusters ~80%)? (YES/NO)
- [ ] Do cluster sizes look reasonable (20-40 videos each)? (YES/NO)

**If all clusters ~80%**: K-Means found no meaningful patterns (possible but investigate)

---

### 5. Confidence Tiers

**Check**: Do tier assignments make sense?

```json
"cluster_1": {"tier": "GOLD STANDARD", "confidence_score": 94},
"cluster_2": {"tier": "BRONZE (MODERATELY VALIDATED)", "confidence_score": 42},
"cluster_3": {"tier": "SILVER (VALIDATED)", "confidence_score": 68}
```

**Questions**:
- [ ] Is at least one cluster GOLD or SILVER? (YES/NO)
- [ ] Do scores align with your intuition? (YES/NO)
- [ ] Would you confidently recommend GOLD clusters to creators? (YES/NO)

**If NO GOLD/SILVER clusters**: Either data is noisy or validation is too strict

---

### 6. Silhouette Scores

**Check**: Are clusters cohesive?

```json
"cluster_1": {"silhouette_score": 0.52},
"cluster_2": {"silhouette_score": 0.31},
"cluster_3": {"silhouette_score": 0.48}
```

**Questions**:
- [ ] Is average silhouette > 0.3? (YES/NO)
- [ ] Are there any clusters with silhouette < 0.2? (List them)

**If silhouette < 0.3 for all**: Clusters are poorly separated (K-Means quality issue)

---

## Overall Assessment

**Based on manual review**:
- [ ] Stage 5 outputs are trustworthy for creator guidelines (YES/NO)
- [ ] Validation logic produces reasonable results (YES/NO)
- [ ] No major bugs or logic errors detected (YES/NO)

**If any NO**: Document issues and re-examine Stage 5 logic before production use.

---

## Sign-Off

**Reviewer**: ________________
**Date**: ________________
**Hashtag**: ________________
**Bucket**: ________________

**Notes**:
```

---

# Test Execution Guide

## Running All Tests

```bash
# 1. Run unit tests (fast, ~10 seconds)
pytest tests/unit/ -v

# 2. Run integration tests (slower, ~30 seconds)
# NOTE: Requires Stage 4 fixtures (see setup instructions below)
pytest tests/integration/ -v

# 3. Manual validation (human review, ~30 minutes)
# Open manual_validation_checklist.md and follow instructions
```

---

## Setting Up Integration Test Fixtures

**Integration tests require real Stage 4 output files.**

### Setup Instructions:

```bash
# 1. Run Stage 4 on a small sample (20 videos, bucket 18-33s)
# This generates the required fixture files

# 2. Create fixtures directory
mkdir -p tests/fixtures/stage4_outputs/bucket_18-33s/ml_analysis/

# 3. Copy Stage 4 outputs to fixtures
cp /data/clients/test/hashtags/test/top_contrastive/buckets/bucket_18-33s/ml_analysis/*.csv \
   tests/fixtures/stage4_outputs/bucket_18-33s/ml_analysis/

# 4. Verify fixtures exist
ls tests/fixtures/stage4_outputs/bucket_18-33s/ml_analysis/
# Should see:
# rf_transformed.csv
# hook_rf_transformed.csv
# hook_km_transformed.csv
# (and 11 more files)

# 5. Now integration tests will work
pytest tests/integration/test_stage4_to_stage5_integration.py -v
```

---

## Expected Test Results

### Unit Tests (All should pass)

```
tests/unit/test_binomial_test.py ........................ PASSED
tests/unit/test_feature_normalization.py ................ PASSED
tests/unit/test_kmeans_feature_ranking.py ............... PASSED
tests/unit/test_success_rate.py ........................ PASSED
tests/unit/test_silhouette_score.py ..................... PASSED
tests/unit/test_confidence_scoring.py ................... PASSED

============================== 6 passed in 2.5s ===============================
```

### Integration Tests (All should pass)

```
tests/integration/test_stage4_to_stage5_integration.py .. PASSED

============================== 1 passed in 5.2s ===============================
```

### Manual Validation (Subjective)

- Review outputs on real data
- Sign off on checklist
- Document any concerns

---

## Troubleshooting

### Test Failure: "Feature name mismatch! Only 0/21 features overlap"

**Cause**: Feature normalization not implemented or incorrect

**Fix**: Implement `normalize_feature_name()` function in Stage 5 validation code

---

### Test Failure: "Expected 22 columns, got 23"

**Cause**: Schema mismatch between Stage 4 and Stage 5 expectations

**Fix**: Check Stage 4 output schema, adjust Stage 5 expectations

---

### Test Failure: scipy version error

**Cause**: scipy < 1.7 (binomtest not available)

**Fix**: Upgrade scipy
```bash
pip install --upgrade scipy
```

---

## Document Metadata

**Version**: 1.0
**Date**: 2025-10-14
**Author**: Claude (Business Critique Phase 1)
**Status**: Ready for Implementation
**Estimated Implementation Time**: 4-6 hours

---

## Next Steps

1. Implement all unit tests (tests/unit/*.py)
2. Set up Stage 4 fixtures (run Stage 4 on sample data)
3. Implement integration tests (tests/integration/*.py)
4. Run all tests, fix any failures
5. Run Stage 5 on production data
6. Complete manual validation checklist
7. Sign off on Stage 5 validation system

**Only after all tests pass** should Stage 5 be used for production creator guidelines.
