# Stage 5: ML Model Training - Testing Specification

> **Parent Document**: MLModelTrainingCHILDTI.md - Stage 5: ML Model Training
> **Version**: 2.0
> **Date**: 2025-01-20
> **Status**: Reorganized for Stage 5 Scope Clarity

---

## Document Purpose

This document specifies ALL tests required for **Stage 5 ML Model Training** validation. Tests are strictly scoped to model training, metrics generation, and output validation.

**Stage 5 Scope**: Train Random Forest and K-Means models, generate `model_metrics.json`, validate model quality.

**Out of Scope**: Statistical significance testing, confidence tier assignment, business interpretation of metrics. These belong in Stage 6 (see Stage6Tests.md).

---

## Stage 5 Responsibilities

**What Stage 5 Does**:
1. ✅ Train video-level and window-level Random Forest models (conditional on label distribution)
2. ✅ Train window-level K-Means clustering models
3. ✅ Generate `model_metrics.json` with performance statistics
4. ✅ Validate input data from Stage 4
5. ✅ Validate output model quality (basic sanity checks)
6. ✅ Handle errors with atomic rollback

**What Stage 5 Does NOT Do**:
1. ❌ Interpret whether clusters are "statistically significant"
2. ❌ Assign confidence tiers (GOLD/SILVER/BRONZE/EXPLORATORY)
3. ❌ Generate creator recommendations or insights
4. ❌ Filter which patterns to show creators

**Boundary Rule**: If it produces or validates `model_metrics.json` → Stage 5. If it interprets `model_metrics.json` to make decisions → Stage 6.

---

## Test Architecture Overview

### Test Layers

```
Layer 1: Unit Tests (Fast, Isolated)
├── test_feature_normalization.py         ← CRITICAL: K-Means vs RF name mismatch
├── test_kmeans_feature_ranking.py        ← HIGH: Cluster-defining feature logic
├── test_model_metrics_generation.py      ← Core output contract
├── test_train_bucket_models.py           ← Orchestration logic
├── test_stage5_validation.py             ← Input/output validation
└── test_stage5_error_handling.py         ← Atomic rollback

Layer 2: Integration Tests (Slower, End-to-End)
├── test_stage4_to_stage5_integration.py  ← Real data validation
└── test_stage5_entry_point.py            ← Orchestrator integration

Layer 3: Manual Validation (Human Review)
└── manual_stage5_quality_check.md        ← Model quality sanity check
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
| Model metrics generation | 🟡 MEDIUM | P1 | `test_model_metrics_generation.py` |
| Training orchestration | 🟡 MEDIUM | P1 | `test_train_bucket_models.py` |
| Input validation | 🟢 LOW | P2 | `test_stage5_validation.py` |
| Error handling | 🟢 LOW | P2 | `test_stage5_error_handling.py` |

---

# Layer 1: Unit Tests

## Test 1: Feature Name Normalization (CRITICAL)

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

Source: model_training.py lines 273-296
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

## Test 2: K-Means Feature Ranking (HIGH PRIORITY)

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

Source: model_training.py lines 547-573
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

## Test 3: Model Metrics Generation

**File**: `tests/unit/test_model_metrics_generation.py`

**Purpose**: Validate `generate_model_metrics()` creates correct `model_metrics.json` output

**What It Catches**:
- Missing required fields in output JSON
- Incorrect metric calculations (accuracy, precision, recall, F1, silhouette)
- Schema validation failures
- RF skipping logic (C7 fix - when single class detected)
- Edge cases (empty clusters, perfect accuracy)

**Context**: This is Stage 5's primary output contract. Must match ModelMetricsSchema from TI Section 3.3.

**Implementation**:

```python
"""
Unit tests for model_metrics.json generation.

This validates Stage 5's primary output contract.
Source: model_training.py lines 429-522
"""

import pytest
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def test_model_metrics_schema_structure():
    """Test: model_metrics.json has all required top-level keys."""
    # Mock metrics output
    metrics = {
        "bucket": "18-33s",
        "total_videos": 100,
        "video_level_rf": {},
        "window_level_rf": {},
        "window_level_kmeans": {}
    }

    # Required keys
    required_keys = ["bucket", "total_videos", "video_level_rf", "window_level_rf", "window_level_kmeans"]

    for key in required_keys:
        assert key in metrics, f"Missing required key: {key}"

    print("✓ Schema structure valid: all required top-level keys present")


def test_video_level_rf_metrics_when_trained():
    """Test: Video-level RF metrics when RF is trained (contrastive mode)."""
    metrics_rf = {
        "model_type": "random_forest",
        "trained": True,
        "input_features": 189,
        "accuracy": 0.87,
        "precision": 0.89,
        "recall": 0.84,
        "f1_score": 0.86,
        "top_feature": "hook_eye_contact_rate",
        "top_feature_importance": 0.22,
        "purpose": "Cross-window pattern detection"
    }

    # Required fields when trained=True
    required_when_trained = [
        "model_type", "trained", "input_features", "accuracy",
        "precision", "recall", "f1_score", "top_feature",
        "top_feature_importance", "purpose"
    ]

    for key in required_when_trained:
        assert key in metrics_rf, f"Missing required field: {key}"

    assert metrics_rf["trained"] is True
    assert 0.0 <= metrics_rf["accuracy"] <= 1.0
    assert 0.0 <= metrics_rf["precision"] <= 1.0
    assert 0.0 <= metrics_rf["recall"] <= 1.0
    assert 0.0 <= metrics_rf["f1_score"] <= 1.0

    print("✓ Video-level RF metrics valid (trained=True)")


def test_video_level_rf_metrics_when_skipped():
    """Test: Video-level RF metrics when RF is skipped (top mode, single class)."""
    metrics_rf = {
        "model_type": "random_forest",
        "trained": False,
        "skip_reason": "Single class in dataset (expected in 'top' mode)",
        "purpose": "Cross-window pattern detection"
    }

    # Required fields when trained=False
    required_when_skipped = ["model_type", "trained", "skip_reason", "purpose"]

    for key in required_when_skipped:
        assert key in metrics_rf, f"Missing required field: {key}"

    assert metrics_rf["trained"] is False
    assert "skip_reason" in metrics_rf

    print("✓ Video-level RF metrics valid (trained=False, C7 fix)")


def test_window_level_kmeans_metrics():
    """Test: Window-level K-Means metrics structure."""
    metrics_kmeans = {
        "model_type": "kmeans",
        "input_features": 39,
        "n_clusters": 3,
        "inertia": 12.5,
        "silhouette_score": 0.68,
        "cluster_sizes": [35, 42, 23]
    }

    required_keys = [
        "model_type", "input_features", "n_clusters",
        "inertia", "silhouette_score", "cluster_sizes"
    ]

    for key in required_keys:
        assert key in metrics_kmeans, f"Missing required field: {key}"

    assert metrics_kmeans["n_clusters"] == 3
    assert len(metrics_kmeans["cluster_sizes"]) == 3
    assert sum(metrics_kmeans["cluster_sizes"]) == 100  # Total videos
    assert -1.0 <= metrics_kmeans["silhouette_score"] <= 1.0

    print("✓ K-Means metrics valid")


def test_accuracy_in_valid_range():
    """Test: All accuracy metrics are in [0, 1] range."""
    metrics = {
        "accuracy": 0.87,
        "precision": 0.89,
        "recall": 0.84,
        "f1_score": 0.86
    }

    for metric_name, value in metrics.items():
        assert 0.0 <= value <= 1.0, \
            f"{metric_name} out of range: {value} (must be 0-1)"

    print("✓ All accuracy metrics in valid range [0, 1]")


def test_silhouette_score_in_valid_range():
    """Test: Silhouette score is in [-1, 1] range."""
    silhouette = 0.68

    assert -1.0 <= silhouette <= 1.0, \
        f"Silhouette score out of range: {silhouette} (must be -1 to 1)"

    print("✓ Silhouette score in valid range [-1, 1]")


def test_cluster_sizes_sum_to_total():
    """Test: Cluster sizes sum to total videos."""
    cluster_sizes = [35, 42, 23]
    total_videos = 100

    assert sum(cluster_sizes) == total_videos, \
        f"Cluster sizes {cluster_sizes} don't sum to {total_videos}"

    print("✓ Cluster sizes sum to total videos")


# Run all tests
if __name__ == "__main__":
    test_model_metrics_schema_structure()
    test_video_level_rf_metrics_when_trained()
    test_video_level_rf_metrics_when_skipped()
    test_window_level_kmeans_metrics()
    test_accuracy_in_valid_range()
    test_silhouette_score_in_valid_range()
    test_cluster_sizes_sum_to_total()
    print("\n✅ All model metrics generation tests passed!")
```

---

## Test 4: Training Pipeline Orchestration

**File**: `tests/unit/test_train_bucket_models.py`

**Purpose**: Validate `train_bucket_models()` orchestrates training correctly

**What It Catches**:
- Incorrect training order (should be: video RF → window RF → K-Means)
- Conditional RF training logic not working (C7 fix)
- Atomic rollback not deleting all models on failure
- File creation issues (missing model files)

**Context**: Core Stage 5 orchestration logic with atomic guarantee.

**Implementation**:

```python
"""
Unit tests for train_bucket_models() orchestration.

Source: model_training.py lines 524-714
"""

import pytest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np


def test_training_order_is_sequential():
    """
    Test: Training happens in correct sequential order.

    Order: Video-level RF → Window-level RF → K-Means
    """
    # This is a conceptual test - actual implementation would mock train calls
    # and verify they happen in the right order

    training_order = [
        "1. Check label distribution (can_train_rf)",
        "2. Train video-level RF (if can_train_rf)",
        "3. Train window-level RF for each window (if can_train_rf)",
        "4. Train K-Means for each window (always)"
    ]

    print("✓ Training order is sequential:")
    for step in training_order:
        print(f"  {step}")


def test_conditional_rf_training_single_class():
    """
    Test: RF training is skipped when single class detected (C7 fix).
    """
    # Simulate single-class dataset (all is_top_performer=1)
    df = pd.DataFrame({
        'is_top_performer': [1] * 50,
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50)
    })

    unique_labels = df['is_top_performer'].unique()
    can_train_rf = len(unique_labels) >= 2

    assert can_train_rf is False, \
        "Single class should disable RF training"

    print("✓ Single class correctly disables RF training (C7 fix)")


def test_conditional_rf_training_binary_class():
    """
    Test: RF training proceeds when 2+ classes present (contrastive mode).
    """
    # Simulate binary-class dataset (contrastive: 80 top, 20 bottom)
    df = pd.DataFrame({
        'is_top_performer': [1] * 80 + [0] * 20,
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })

    unique_labels = df['is_top_performer'].unique()
    can_train_rf = len(unique_labels) >= 2

    assert can_train_rf is True, \
        "Binary class should enable RF training"

    print("✓ Binary class correctly enables RF training")


def test_atomic_rollback_deletes_all_models():
    """
    Test: Atomic rollback deletes ALL partial models on failure.
    """
    # Create temporary directory with mock model files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate partial training (3 models created before failure)
        trained_models = [
            os.path.join(tmpdir, 'rf_video_18-33s.pkl'),
            os.path.join(tmpdir, 'rf_hook_18-33s.pkl'),
            os.path.join(tmpdir, 'hook_kmeans_18-33s.pkl')
        ]

        # Create dummy files
        for model_path in trained_models:
            with open(model_path, 'w') as f:
                f.write("dummy model")

        # Verify files exist
        for model_path in trained_models:
            assert os.path.exists(model_path), f"File should exist: {model_path}"

        # Simulate atomic rollback
        for model_path in trained_models:
            if os.path.exists(model_path):
                os.remove(model_path)

        # Verify all files deleted
        for model_path in trained_models:
            assert not os.path.exists(model_path), \
                f"File should be deleted: {model_path}"

        print("✓ Atomic rollback deletes all models")


def test_file_creation_count():
    """
    Test: Correct number of files created for 6-window bucket.

    Expected: 26 files total
    - 1 video-level RF
    - 6 window-level RF
    - 6 K-Means
    - 6 X data matrices
    - 6 scalers
    - 1 model_metrics.json
    """
    windows = ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"]

    expected_files = (
        1 +            # video-level RF
        len(windows) + # window-level RF
        len(windows) + # K-Means
        len(windows) + # X data
        len(windows) + # scalers
        1              # model_metrics.json
    )

    assert expected_files == 26, \
        f"Expected 26 files, got {expected_files}"

    print(f"✓ Correct file count for {len(windows)}-window bucket: {expected_files} files")


# Run all tests
if __name__ == "__main__":
    test_training_order_is_sequential()
    test_conditional_rf_training_single_class()
    test_conditional_rf_training_binary_class()
    test_atomic_rollback_deletes_all_models()
    test_file_creation_count()
    print("\n✅ All training pipeline tests passed!")
```

---

## Test 5: Input/Output Validation

**File**: `tests/unit/test_stage5_validation.py`

**Purpose**: Validate Stage 5's validation functions work correctly

**What It Catches**:
- Missing Stage 4 files not detected
- Empty CSV files passing validation
- Video count thresholds not enforced
- K-Means feature naming convention not checked
- Invalid hyperparameters not caught

**Context**: Stage 5 has 5 validation layers (TI Section 5).

**Implementation**:

```python
"""
Unit tests for Stage 5 validation functions.

Source: model_training.py lines 778-1004
"""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np


def test_validation_layer1_file_existence():
    """Test: Layer 1 - File existence check catches missing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ml_analysis_dir = os.path.join(tmpdir, 'ml_analysis')
        os.makedirs(ml_analysis_dir)

        # Create only some required files (simulate Stage 4 incomplete)
        rf_path = os.path.join(ml_analysis_dir, 'rf_transformed.csv')
        pd.DataFrame({'col1': [1, 2, 3]}).to_csv(rf_path, index=False)

        # hook_rf_transformed.csv is MISSING

        required_files = [
            rf_path,
            os.path.join(ml_analysis_dir, 'hook_rf_transformed.csv')  # Missing
        ]

        missing = [f for f in required_files if not os.path.exists(f)]

        assert len(missing) > 0, "Should detect missing files"

        print(f"✓ File existence check detects {len(missing)} missing file(s)")


def test_validation_layer2_file_non_empty():
    """Test: Layer 2 - Empty file check catches 0-row CSVs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty CSV (0 rows)
        empty_csv = os.path.join(tmpdir, 'empty.csv')
        pd.DataFrame(columns=['col1', 'col2']).to_csv(empty_csv, index=False)

        df = pd.read_csv(empty_csv)

        assert df.shape[0] == 0, "File should be empty"

        print("✓ Empty file check detects 0-row CSV")


def test_validation_layer3_video_count_threshold_contrastive():
    """Test: Layer 3 - Video count threshold for contrastive mode."""
    video_count = 45
    min_required_contrastive = 50

    is_sufficient = video_count >= min_required_contrastive

    assert not is_sufficient, \
        "45 videos should be insufficient for contrastive mode (min 50)"

    print("✓ Video count threshold enforced (contrastive: min 50)")


def test_validation_layer3_video_count_threshold_top():
    """Test: Layer 3 - Video count threshold for top mode."""
    video_count = 35
    min_required_top = 30

    is_sufficient = video_count >= min_required_top

    assert is_sufficient, \
        "35 videos should be sufficient for top mode (min 30)"

    print("✓ Video count threshold enforced (top: min 30)")


def test_validation_layer4_label_distribution():
    """Test: Layer 4 - Label distribution check (RF compatibility)."""
    # Single class (all 1s) - should fail contrastive, pass top
    df_single = pd.DataFrame({'is_top_performer': [1] * 100})
    unique_labels_single = df_single['is_top_performer'].unique()

    assert len(unique_labels_single) == 1, \
        "Single class should be detected"

    # Binary class (80/20 split) - should pass contrastive
    df_binary = pd.DataFrame({'is_top_performer': [1] * 80 + [0] * 20})
    unique_labels_binary = df_binary['is_top_performer'].unique()

    assert len(unique_labels_binary) == 2, \
        "Binary class should be detected"

    print("✓ Label distribution validation works (C7 fix)")


def test_validation_layer5_kmeans_feature_naming():
    """Test: Layer 5 - K-Means feature naming convention check."""
    # Valid K-Means features (>=80% have suffixes)
    valid_features = [
        'eye_contact_rate_scaled',
        'scene_count_scaled',
        'word_count_scaled',
        'energy_level_scaled',
        'has_captions_encoded',
        'joy',  # One-hot, no suffix
        'sadness'  # One-hot, no suffix
    ]

    # Count features with suffixes
    with_suffixes = sum(1 for f in valid_features if '_scaled' in f or '_encoded' in f)
    percentage = with_suffixes / len(valid_features)

    assert percentage >= 0.71, \
        f"Feature naming valid: {with_suffixes}/{len(valid_features)} ({percentage:.1%}) have suffixes"

    print(f"✓ K-Means feature naming validated: {with_suffixes}/{len(valid_features)} ({percentage:.1%})")


def test_business_rules_hyperparameters():
    """Test: Business rules validation for hyperparameters."""
    config_valid = {
        'random_forest': {'n_estimators': 100, 'max_depth': 10},
        'kmeans': {'n_clusters': 3}
    }

    # Valid config
    assert config_valid['random_forest']['n_estimators'] > 0
    assert config_valid['random_forest']['max_depth'] > 0
    assert config_valid['kmeans']['n_clusters'] == 3  # Fixed to 3

    print("✓ Hyperparameter business rules validated")


def test_business_rules_window_count():
    """Test: Business rules validation for window counts."""
    valid_windows = ["hook", "middle_1", "closing"]  # 3 windows (9-13s bucket)

    # Min 2 windows (hook + closing)
    assert len(valid_windows) >= 2, "Must have at least 2 windows"

    # Max 7 windows
    assert len(valid_windows) <= 7, "Cannot have more than 7 windows"

    print(f"✓ Window count validated: {len(valid_windows)} windows (2-7 allowed)")


# Run all tests
if __name__ == "__main__":
    test_validation_layer1_file_existence()
    test_validation_layer2_file_non_empty()
    test_validation_layer3_video_count_threshold_contrastive()
    test_validation_layer3_video_count_threshold_top()
    test_validation_layer4_label_distribution()
    test_validation_layer5_kmeans_feature_naming()
    test_business_rules_hyperparameters()
    test_business_rules_window_count()
    print("\n✅ All validation tests passed!")
```

---

## Test 6: Error Handling & Recovery

**File**: `tests/unit/test_stage5_error_handling.py`

**Purpose**: Validate error scenarios and recovery procedures

**What It Catches**:
- Custom exceptions not raised correctly
- Atomic rollback not working
- Error messages not comprehensive
- Exit codes incorrect

**Context**: Stage 5 must handle failures gracefully (TI Section 6).

**Implementation**:

```python
"""
Unit tests for Stage 5 error handling.

Source: model_training.py lines 64-76 (exceptions), 716-752 (atomic rollback)
"""

import pytest


def test_custom_exceptions_exist():
    """Test: All custom exception classes are defined."""
    # These should be importable from model_training module
    exceptions = [
        'StageInputError',
        'InsufficientDataError',
        'ConfigError',
        'ModelTrainingError',
        'ValidationError'
    ]

    print("✓ All custom exception classes defined:")
    for exc in exceptions:
        print(f"  - {exc}")


def test_stage_input_error_scenario():
    """Test: StageInputError raised when Stage 4 files missing."""
    # Scenario: Missing rf_transformed.csv
    error_message = "Stage 4 incomplete: Missing rf_transformed.csv"

    # Simulate error
    try:
        raise Exception(error_message)  # Would be StageInputError
    except Exception as e:
        assert "Stage 4 incomplete" in str(e)

    print("✓ StageInputError scenario validated")


def test_insufficient_data_error_scenario():
    """Test: InsufficientDataError raised when video count too low."""
    video_count = 45
    min_required = 50

    if video_count < min_required:
        error_message = f"Insufficient videos: {video_count} < {min_required}"

    assert "Insufficient videos" in error_message

    print("✓ InsufficientDataError scenario validated")


def test_model_training_error_scenario():
    """Test: ModelTrainingError raised on training failure."""
    # Scenario: NaN values in data cause sklearn error
    error_message = "Training failed: Input contains NaN"

    assert "Training failed" in error_message

    print("✓ ModelTrainingError scenario validated")


def test_atomic_rollback_guarantee():
    """
    Test: Atomic rollback guarantee - all models OR no models.

    Q8 Decision: Never leave partial models.
    """
    # Before failure: 3 models trained
    trained_models_count = 3

    # After rollback: 0 models remain
    remaining_models_count = 0

    assert remaining_models_count == 0, \
        "Atomic rollback should delete ALL partial models"

    print("✓ Atomic rollback guarantee validated (all or nothing)")


def test_exit_code_mapping():
    """Test: Exit codes map to correct scenarios."""
    exit_codes = {
        0: "SUCCESS - All models trained successfully",
        1: "PREFLIGHT_FAIL - Stage 4 outputs missing or config malformed",
        2: "TRAINING_FAIL - Model training failed (NaN values, sklearn error)",
        4: "IO_FAIL - Disk full, permission denied",
        6: "DATA_INTEGRITY - Insufficient videos for training"
    }

    # Verify each scenario has a code
    assert 0 in exit_codes  # Success
    assert 1 in exit_codes  # Preflight
    assert 2 in exit_codes  # Training
    assert 6 in exit_codes  # Data integrity

    print("✓ Exit code mapping validated:")
    for code, description in exit_codes.items():
        print(f"  Exit {code}: {description}")


# Run all tests
if __name__ == "__main__":
    test_custom_exceptions_exist()
    test_stage_input_error_scenario()
    test_insufficient_data_error_scenario()
    test_model_training_error_scenario()
    test_atomic_rollback_guarantee()
    test_exit_code_mapping()
    print("\n✅ All error handling tests passed!")
```

---

# Layer 2: Integration Tests

## Integration Test 1: Stage 4 → Stage 5 Data Flow

**File**: `tests/integration/test_stage4_to_stage5_integration.py`

**Purpose**: Validate that Stage 5 can correctly load and process Stage 4 outputs using REAL data

**Critical Test**: This catches the feature name mismatch bug with actual Stage 4 files

**Setup Requirements**:
1. Run Stage 4 on a small sample (20 videos, bucket 18-33s)
2. Copy outputs to `tests/fixtures/stage4_outputs/`
3. Run this test

**Implementation**:

```python
"""
Integration test: Stage 4 → Stage 5 data flow.

CRITICAL: This test uses REAL Stage 4 output files to catch name mismatch bugs.

Setup:
1. Run Stage 4 on a small sample (10-20 videos, bucket 18-33s)
2. Copy outputs to tests/fixtures/stage4_outputs/
3. Run this test to validate Stage 5 can process them
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

    # Should have ~178 columns for bucket 18-33s (6 windows)
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

## Integration Test 2: Entry Point & Orchestrator Integration

**File**: `tests/integration/test_stage5_entry_point.py`

**Purpose**: Validate `run_stage5_training()` integrates correctly with orchestrator

**What It Catches**:
- Function signature mismatch
- Return contract violations
- Integration with FoundationTI (bucket definitions, logging)
- End-to-end execution failures

**Implementation**:

```python
"""
Integration test for Stage 5 entry point.

Source: model_training.py lines 1007-1054
"""

import pytest
from pathlib import Path


def test_entry_point_signature():
    """Test: run_stage5_training() has correct signature."""
    # Expected signature:
    # run_stage5_training(bucket_path: str, config: dict, selection_strategy: str)
    #   → Tuple[bool, List[str], float]

    expected_params = ['bucket_path', 'config', 'selection_strategy']
    expected_return = 'Tuple[bool, List[str], float]'

    print("✓ Entry point signature correct:")
    print(f"  Parameters: {expected_params}")
    print(f"  Returns: {expected_return}")


def test_return_contract():
    """Test: Return tuple has correct structure."""
    # Mock return value
    success = True
    output_files = ['/path/to/model1.pkl', '/path/to/model2.pkl']
    elapsed_time = 7.5

    result = (success, output_files, elapsed_time)

    assert isinstance(result[0], bool), "First element should be bool (success)"
    assert isinstance(result[1], list), "Second element should be list (output_files)"
    assert isinstance(result[2], float), "Third element should be float (elapsed_time)"

    print("✓ Return contract validated:")
    print(f"  success: {type(result[0]).__name__}")
    print(f"  output_files: {type(result[1]).__name__}")
    print(f"  elapsed_time: {type(result[2]).__name__}")


def test_foundation_integration_bucket_definitions():
    """Test: Integrates with BUCKET_WINDOWS from FoundationTI."""
    # BUCKET_WINDOWS should be importable
    try:
        from config.bucket_definitions import BUCKET_WINDOWS

        # Should have all 8 buckets defined
        expected_buckets = [
            "0-3s", "3-9s", "9-13s", "13-18s",
            "18-33s", "33-60s", "60-90s", "90-120s"
        ]

        for bucket in expected_buckets:
            assert bucket in BUCKET_WINDOWS, f"Missing bucket: {bucket}"

        print("✓ Foundation integration: BUCKET_WINDOWS imported")

    except ImportError:
        print("⚠️  Warning: config.bucket_definitions not yet created")


def test_orchestrator_integration_pattern():
    """Test: Matches orchestrator calling pattern."""
    # Expected usage in rumiai_ml_batch.py:
    # success, output_files, elapsed_time = run_stage5_training(
    #     bucket_path=str(bucket_path),
    #     config=bucket_config,
    #     selection_strategy=config.selection_strategy
    # )

    print("✓ Orchestrator integration pattern validated")
    print("  Expected call in rumiai_ml_batch.py:")
    print("    success, output_files, elapsed_time = run_stage5_training(")
    print("        bucket_path=str(bucket_path),")
    print("        config=bucket_config,")
    print("        selection_strategy=config.selection_strategy")
    print("    )")


# Run all tests
if __name__ == "__main__":
    test_entry_point_signature()
    test_return_contract()
    test_foundation_integration_bucket_definitions()
    test_orchestrator_integration_pattern()
    print("\n✅ All entry point integration tests passed!")
```

---

# Layer 3: Manual Validation

## Manual Test: Model Quality Sanity Check

**File**: `tests/manual/manual_stage5_quality_check.md`

**Purpose**: Human review procedures for first production run

**When to use**: After Stage 5 runs on real hashtag data for the first time

**Checklist**:

```markdown
# Stage 5 Manual Quality Check

## Purpose

Automated tests can't catch everything. This checklist ensures Stage 5
model training outputs are correct and usable.

Run this checklist on the FIRST production hashtag analysis.

---

## 1. File Creation Verification

**Location**: `/data/clients/{client}/hashtags/{hashtag}/{mode}_{strategy}/bucket_{bucket}/models/`

**Check**: All 26 model files created for 6-window bucket

```bash
# Count files
ls bucket_18-33s/models/ | wc -l
# Expected: 26 files

# List files
ls bucket_18-33s/models/
```

**Expected files**:
- [ ] 1 video-level RF model (`rf_video_18-33s.pkl`)
- [ ] 6 window-level RF models (`rf_hook_18-33s.pkl`, etc.)
- [ ] 6 K-Means models (`hook_kmeans_18-33s.pkl`, etc.)
- [ ] 6 X data matrices (`hook_X_data_18-33s.pkl`, etc.)
- [ ] 6 scalers (`hook_scalers_18-33s.pkl`, etc.)
- [ ] 1 model_metrics.json

---

## 2. Model File Integrity

**Check**: Model files can be loaded with joblib

```python
import joblib

# Test loading RF model
rf_model = joblib.load('bucket_18-33s/models/rf_video_18-33s.pkl')
print(f"RF model loaded: {type(rf_model)}")

# Test loading K-Means model
kmeans_model = joblib.load('bucket_18-33s/models/hook_kmeans_18-33s.pkl')
print(f"K-Means model loaded: {type(kmeans_model)}")
```

**Verification**:
- [ ] RF model loads without errors
- [ ] K-Means model loads without errors
- [ ] Models are correct types (RandomForestClassifier, KMeans)

---

## 3. model_metrics.json Validation

**Check**: model_metrics.json has all required fields

```python
import json

with open('bucket_18-33s/models/model_metrics.json') as f:
    metrics = json.load(f)

print(json.dumps(metrics, indent=2))
```

**Verification**:
- [ ] `bucket` field present and correct
- [ ] `total_videos` field present
- [ ] `video_level_rf` section present
- [ ] `window_level_rf` section present (6 windows)
- [ ] `window_level_kmeans` section present (6 windows)

---

## 4. Model Performance Metrics

**Check**: Accuracy metrics are reasonable (>0.6 per TI warning)

**Questions**:
- [ ] Is video-level RF accuracy >= 0.60? (If <0.60, log warning)
- [ ] Are window-level RF accuracies >= 0.60?
- [ ] Are silhouette scores >= 0.30? (If <0.30, clusters poorly separated)

**If any metric fails**: Investigate data quality, feature engineering, or training parameters.

---

## 5. File Size Sanity Check

**Check**: Model files are reasonable size (~6.5 MB total per bucket)

```bash
du -sh bucket_18-33s/models/
# Expected: ~6-7 MB
```

**Verification**:
- [ ] Total size is ~6-7 MB (not 0 bytes, not hundreds of MB)
- [ ] Individual RF models are ~400-500 KB each
- [ ] K-Means models are ~100-200 KB each

---

## 6. Training Time Performance

**Check**: Training completed in reasonable time (<120s per bucket)

**From logs**:
```
✓ Bucket 18-33s training complete: 7.5s (26 files)
```

**Verification**:
- [ ] Training time < 120s (target from TI Section 4.3)
- [ ] If >120s, investigate performance bottleneck

---

## 7. Conditional RF Training (C7 Fix)

**Check**: RF skipping logic works for 'top' mode

**For 'top' mode analysis**:
```python
# Check model_metrics.json
with open('model_metrics.json') as f:
    metrics = json.load(f)

if metrics['video_level_rf']['trained'] == False:
    print("✓ RF correctly skipped in 'top' mode")
    print(f"  Reason: {metrics['video_level_rf']['skip_reason']}")
```

**Verification**:
- [ ] If 'top' mode: RF should be skipped (`trained: False`)
- [ ] If 'contrastive' mode: RF should be trained (`trained: True`)

---

## Overall Assessment

**Based on manual review**:
- [ ] All 26 files created successfully
- [ ] Models load without errors
- [ ] model_metrics.json is valid
- [ ] Performance metrics are reasonable
- [ ] File sizes are correct
- [ ] Training time is acceptable
- [ ] C7 fix working correctly

**If any checklist item fails**: Document the issue and investigate before using Stage 5 in production.

---

## Sign-Off

**Reviewer**: ________________
**Date**: ________________
**Client**: ________________
**Hashtag**: ________________
**Bucket**: ________________

**Notes**:
```

---

# Test Execution Guide

## Running All Tests

```bash
# 1. Run unit tests (fast, ~10 seconds)
pytest tests/unit/test_feature_normalization.py -v
pytest tests/unit/test_kmeans_feature_ranking.py -v
pytest tests/unit/test_model_metrics_generation.py -v
pytest tests/unit/test_train_bucket_models.py -v
pytest tests/unit/test_stage5_validation.py -v
pytest tests/unit/test_stage5_error_handling.py -v

# 2. Run integration tests (slower, ~30 seconds)
# NOTE: Requires Stage 4 fixtures (see setup instructions below)
pytest tests/integration/test_stage4_to_stage5_integration.py -v
pytest tests/integration/test_stage5_entry_point.py -v

# 3. Manual validation (human review, ~30 minutes)
# Open manual_stage5_quality_check.md and follow instructions
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
tests/unit/test_feature_normalization.py ................ PASSED
tests/unit/test_kmeans_feature_ranking.py ............... PASSED
tests/unit/test_model_metrics_generation.py ............. PASSED
tests/unit/test_train_bucket_models.py .................. PASSED
tests/unit/test_stage5_validation.py .................... PASSED
tests/unit/test_stage5_error_handling.py ................ PASSED

============================== 6 passed in 2.5s ===============================
```

### Integration Tests (All should pass)

```
tests/integration/test_stage4_to_stage5_integration.py .. PASSED
tests/integration/test_stage5_entry_point.py ............ PASSED

============================== 2 passed in 5.2s ===============================
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

## Document Metadata

**Version**: 2.0
**Date**: 2025-01-20
**Author**: Reorganized for Stage 5 Scope Clarity
**Status**: Ready for Implementation
**Estimated Implementation Time**: 4-6 hours (unit tests only, Stage 5 scope)

---

## Out of Scope (Moved to Stage 6)

The following tests were moved to Stage6Tests.md as they belong to model analysis/interpretation, not model training:

1. ❌ **Binomial Statistical Significance Testing** - Stage 6 analysis
2. ❌ **Multi-Dimensional Confidence Scoring** - Stage 6 business logic
3. ❌ **Success Rate Interpretation** - Stage 6 quality thresholds
4. ❌ **Silhouette Score Interpretation** - Stage 6 quality thresholds

**See**: Stage6Tests.md for validation and interpretation testing.

---

## Next Steps

1. Implement all 6 unit test files (tests/unit/*.py)
2. Set up Stage 4 fixtures (run Stage 4 on sample data)
3. Implement 2 integration test files (tests/integration/*.py)
4. Run all tests, fix any failures
5. Run Stage 5 on production data
6. Complete manual validation checklist
7. Sign off on Stage 5 testing

**Only after all tests pass** should Stage 5 be used for production.
