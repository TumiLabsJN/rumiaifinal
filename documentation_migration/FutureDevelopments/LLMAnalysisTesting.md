# LLM Analysis (Stage 7) - Quality Assurance Strategy

> **Parent**: QA_LLMAnalysis.md - Q6: Testing Strategy
> **Purpose**: In-depth testing strategy for Stage 7 LLM Analysis, referenced by TI and HLD documents
> **Version**: 1.0
> **Last Updated**: 2025-01-28
> **Status**: Draft

---

## 1. Overview

### 1.1 Testing Philosophy

**Core Principle**: Test **deterministic pre-LLM logic exhaustively** with synthetic data BEFORE testing expensive, non-deterministic LLM integration.

### 1.2 Why This Approach?

**Problem**: Stage 7 has two distinct components:
1. **Pre-LLM Logic** (Deterministic): Cluster path extraction, frequency calculation, 10% threshold filtering, confidence classification
2. **LLM Integration** (Non-Deterministic): Anthropic API calls, prompt engineering, JSON parsing

**Risk**: If pre-LLM logic is broken, LLM receives garbage input → garbage output → wasted API costs + unreliable reports

**Solution**: Testing pyramid prioritizing pre-LLM logic

```
        ┌──────────────────────┐
        │  LLM Integration     │  ← Test LAST (expensive, non-deterministic)
        │  (Real API calls)    │  ← 5-10 integration tests
        └──────────────────────┘
              ▲
              │
        ┌──────────────────────┐
        │  Pre-LLM Logic       │  ← Test FIRST (free, deterministic)
        │  (Cluster paths,     │  ← 30-50 unit tests
        │   frequency calc,    │
        │   threshold filter,  │
        │   confidence levels, │
        │   fallback logic)    │
        └──────────────────────┘
```

---

## 2. Test Data Strategy

### 2.1 Two-Phase Test Data Approach

**Phase 1: Synthetic JSONs** (For Pre-LLM Logic Testing)
- **Purpose**: Test edge cases, validate logic correctness with controlled inputs
- **Scope**: All 8 buckets (0-3s through 90-120s)
- **Scale**: Realistic-scale fixtures (50-100 videos per bucket to test 10% threshold logic)
- **Cost**: FREE (no API calls)

**Phase 2: Real Stage 6 Outputs** (For Integration Testing)
- **Purpose**: Test with actual ML data from real video processing
- **Source**: Videos processed through Stage 2.6 → Stage 3 → Stage 4 → Stage 5 → Stage 6
- **Scope**: 1-2 representative buckets (e.g., 18-33s, 33-60s)
- **Scale**: 10-20 videos with full 13 Stage 6 JSONs
- **Cost**: API calls required (~$0.10-0.26 per test run)

### 2.2 Test Fixture Structure

```
tests/
├── fixtures/
│   ├── synthetic/                          # Phase 1: Controlled test data
│   │   ├── bucket_0-3s/
│   │   │   ├── ml_analysis/
│   │   │   │   ├── rf_video_analysis.json           # 50 videos, known feature values
│   │   │   │   ├── hook_rf_analysis.json            # Controlled RF importance
│   │   │   │   └── hook_kmeans_analysis.json        # Known cluster assignments
│   │   │   └── README.md                            # Documents test scenario
│   │   ├── bucket_3-9s/                   # 2 windows, 50 videos
│   │   ├── bucket_9-13s/                  # 3 windows (middle_aggregate)
│   │   ├── bucket_13-18s/                 # 3 windows (middle_aggregate)
│   │   ├── bucket_18-33s/                 # 6 windows, 100 videos
│   │   ├── bucket_33-60s/                 # 7 windows, 100 videos
│   │   ├── bucket_60-90s/                 # 7 windows, 100 videos
│   │   └── bucket_90-120s/                # 7 windows, 100 videos
│   │
│   └── real/                               # Phase 2: Real ML outputs
│       ├── bucket_18-33s_10videos/
│       │   ├── ml_analysis/
│       │   │   ├── rf_video_analysis.json           # From real Stage 6
│       │   │   ├── hook_rf_analysis.json
│       │   │   ├── hook_kmeans_analysis.json
│       │   │   └── ... (13 files total)
│       │   └── README.md                            # Source: videos from hashtag X, date Y
│       └── bucket_33-60s_10videos/
│           └── ml_analysis/
│               └── ... (13 files)
```

### 2.3 Synthetic Test Data Design Principles

**Principle 1: Controlled Cluster Path Distributions**

Example for bucket 18-33s (100 videos):
```python
# Designed distribution for testing 10% threshold
SYNTHETIC_CLUSTER_PATHS = {
    # Path formulas (meet 10% threshold)
    "[0,1,1,2,0,1]": 22,  # 22% - very_high confidence
    "[1,0,0,1,1,0]": 18,  # 18% - high confidence
    "[0,0,1,1,2,2]": 12,  # 12% - moderate confidence

    # Below threshold (should be excluded)
    "[2,2,0,0,1,1]": 8,   # 8% - excluded
    "[1,1,1,1,1,1]": 7,   # 7% - excluded
    "[0,2,1,0,2,1]": 6,   # 6% - excluded

    # Rare patterns (fragmentation)
    "other_patterns": 27  # 27 different paths with 1 video each
}

# Expected Stage 7 behavior:
# - 3 creative_reports (path-based, frequencies: 22%, 18%, 12%)
# - Confidence levels correctly assigned
# - supplementary_insights includes universal RF features
```

**Principle 2: Edge Case Coverage**

**Edge Case 1: Insufficient Paths for 3 Reports**
```python
# Only 2 paths meet 10% threshold
SYNTHETIC_PATHS_INSUFFICIENT = {
    "[0,1,1,2,0,1]": 15,  # 15% - high confidence
    "[1,0,0,1,1,0]": 12,  # 12% - moderate confidence
    "other_patterns": 73  # 73 videos in paths <10%
}

# Expected behavior:
# - 2 path-based reports
# - 1 feature-based fallback report
```

**Edge Case 2: All Paths Below 10% Threshold**
```python
# Maximum path frequency is 9%
SYNTHETIC_PATHS_FRAGMENTED = {
    "[0,1,1,2,0,1]": 9,   # 9% - excluded
    "[1,0,0,1,1,0]": 8,   # 8% - excluded
    "[0,0,1,1,2,2]": 7,   # 7% - excluded
    "other_patterns": 76  # All other paths <7%
}

# Expected behavior:
# - 3 feature-based fallback reports (all based on universal RF features)
# - No path-based reports
```

**Edge Case 3: Bucket 0-3s (Single Window)**
```python
# Only 1 window (hook), no closing
SYNTHETIC_0_3S = {
    "windows": ["hook"],
    "total_videos": 50,
    "hook_kmeans": {
        "cluster_0": 18,  # 36%
        "cluster_1": 20,  # 40%
        "cluster_2": 12   # 24%
    }
}

# Expected behavior:
# - Phase 1: 1 LLM call (hook_analysis.json)
# - Phase 2: SKIPPED (no temporal progression)
# - Output: bucket_summary_0-3s.json with 3 hook strategies
```

---

## 3. Pre-LLM Logic Testing (Unit Tests)

### 3.1 Test Categories

**Total Expected**: 30-50 unit tests covering all deterministic logic

#### **Category 1: Cluster Path Extraction** (5-8 tests)

**Test 1.1: Basic Path Extraction (6 windows)**
```python
def test_cluster_path_extraction_6_windows():
    """
    Test: Extract cluster paths from 6-window bucket (18-33s)

    Input:
    - 100 videos
    - 6 windows: hook, middle_1, middle_2, middle_3, middle_4, closing
    - Known cluster assignments per video

    Expected Output:
    - 100 cluster paths extracted
    - Each path has 6 positions [c1, c2, c3, c4, c5, c6]
    - path_str formatted correctly: "Hook-C0 → M1-C1 → M2-C1 → M3-C2 → M4-C0 → Closing-C1"
    """
    # Load synthetic K-Means JSONs with known assignments
    kmeans_outputs = load_synthetic_kmeans('bucket_18-33s')

    # Expected: video_0 has path [0, 1, 1, 2, 0, 1]
    expected_video_0_path = [0, 1, 1, 2, 0, 1]

    # Run extraction
    video_paths = extract_cluster_paths(window_types=['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing'],
                                        kmeans_outputs=kmeans_outputs)

    # Assertions
    assert len(video_paths) == 100, "Should extract 100 paths"
    assert video_paths[0]['path'] == expected_video_0_path, "Video 0 path incorrect"
    assert video_paths[0]['path_str'] == "Hook-C0 → M1-C1 → M2-C1 → M3-C2 → M4-C0 → Closing-C1"

    # All paths have 6 positions
    assert all(len(vp['path']) == 6 for vp in video_paths)
```

**Test 1.2: Path Extraction - 3 Windows with middle_aggregate**
```python
def test_cluster_path_extraction_middle_aggregate():
    """
    Test: Extract paths from bucket with middle_aggregate (9-13s)

    Input:
    - 50 videos
    - 3 windows: hook, middle_aggregate, closing

    Expected Output:
    - 50 cluster paths with 3 positions [c1, c2, c3]
    - path_str includes "Middle_Aggregate-C1" (not "Middle_1")
    """
    kmeans_outputs = load_synthetic_kmeans('bucket_9-13s')

    video_paths = extract_cluster_paths(window_types=['hook', 'middle_aggregate', 'closing'],
                                        kmeans_outputs=kmeans_outputs)

    assert len(video_paths) == 50
    assert all(len(vp['path']) == 3 for vp in video_paths)
    assert 'Middle_Aggregate' in video_paths[0]['path_str']
```

**Test 1.3: Path Extraction - 2 Windows (3-9s)**
```python
def test_cluster_path_extraction_2_windows():
    """
    Test: Extract paths from bucket with only hook + closing

    Input:
    - 50 videos
    - 2 windows: hook, closing

    Expected Output:
    - 50 cluster paths with 2 positions [c1, c2]
    - 3^2 = 9 possible unique paths
    """
    kmeans_outputs = load_synthetic_kmeans('bucket_3-9s')

    video_paths = extract_cluster_paths(window_types=['hook', 'closing'],
                                        kmeans_outputs=kmeans_outputs)

    assert len(video_paths) == 50
    assert all(len(vp['path']) == 2 for vp in video_paths)

    # Should have ≤9 unique paths (3^2 maximum)
    unique_paths = set(tuple(vp['path']) for vp in video_paths)
    assert len(unique_paths) <= 9
```

**Test 1.4: Path Extraction - Single Window (0-3s)**
```python
def test_cluster_path_extraction_single_window():
    """
    Test: Bucket 0-3s has only 1 window - no path extraction needed

    Input:
    - 50 videos
    - 1 window: hook

    Expected Output:
    - extract_cluster_paths should return empty or raise NotApplicable
    - This bucket skips Phase 2 entirely
    """
    kmeans_outputs = load_synthetic_kmeans('bucket_0-3s')

    # Should either return empty or raise NotApplicableError
    result = extract_cluster_paths(window_types=['hook'], kmeans_outputs=kmeans_outputs)

    assert result is None or len(result) == 0, "Single-window buckets don't have paths"
```

**Test 1.5: Path Extraction - Missing Window Data**
```python
def test_cluster_path_extraction_missing_window():
    """
    Test: Error handling when K-Means data missing for a window

    Input:
    - 6 windows expected
    - middle_3_kmeans_analysis.json is missing

    Expected Output:
    - Raise MissingDataError with specific window name
    """
    kmeans_outputs = load_synthetic_kmeans('bucket_18-33s')
    del kmeans_outputs['middle_3']  # Simulate missing data

    with pytest.raises(MissingDataError) as exc_info:
        extract_cluster_paths(window_types=['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing'],
                              kmeans_outputs=kmeans_outputs)

    assert 'middle_3' in str(exc_info.value)
```

---

#### **Category 2: Path Frequency Calculation** (8-12 tests)

**Test 2.1: Basic Frequency Calculation**
```python
def test_path_frequency_calculation_basic():
    """
    Test: Calculate frequencies for known distribution

    Input:
    - 100 videos with controlled path distribution
    - Path [0,1,1,2,0,1] appears 22 times
    - Path [1,0,0,1,1,0] appears 18 times
    - Path [0,0,1,1,2,2] appears 12 times

    Expected Output:
    - Top 3 paths with frequencies: 22, 18, 12
    - Percentages: 22.0%, 18.0%, 12.0%
    """
    video_paths = create_synthetic_paths(distribution={
        (0,1,1,2,0,1): 22,
        (1,0,0,1,1,0): 18,
        (0,0,1,1,2,2): 12,
        (2,2,0,0,1,1): 8,
        # ... rest distributed across other paths
    }, total=100)

    path_frequencies = analyze_path_frequencies(video_paths)

    assert len(path_frequencies) >= 3
    assert path_frequencies[0]['frequency'] == 22
    assert path_frequencies[0]['percentage'] == 22.0
    assert path_frequencies[1]['frequency'] == 18
    assert path_frequencies[2]['frequency'] == 12
```

**Test 2.2: Frequency Calculation - Ties**
```python
def test_path_frequency_ties():
    """
    Test: Handle tied frequencies correctly

    Input:
    - Path A: 15 videos (15%)
    - Path B: 15 videos (15%) - tied with A
    - Path C: 12 videos (12%)

    Expected Output:
    - Both tied paths included
    - Stable sorting (deterministic order for ties)
    """
    video_paths = create_synthetic_paths(distribution={
        (0,1,1,2,0,1): 15,
        (1,0,0,1,1,0): 15,  # Tied
        (0,0,1,1,2,2): 12,
    }, total=100)

    path_frequencies = analyze_path_frequencies(video_paths)

    # Top 2 should both have frequency 15
    assert path_frequencies[0]['frequency'] == 15
    assert path_frequencies[1]['frequency'] == 15
    assert path_frequencies[2]['frequency'] == 12
```

**Test 2.3: Frequency Calculation - All Paths Unique**
```python
def test_path_frequency_all_unique():
    """
    Test: Extreme fragmentation - every video has unique path

    Input:
    - 100 videos, 100 unique paths (1 video each)

    Expected Output:
    - All paths have frequency 1 (1%)
    - None meet 10% threshold
    """
    video_paths = create_100_unique_paths()

    path_frequencies = analyze_path_frequencies(video_paths)

    assert len(path_frequencies) == 100
    assert all(pf['frequency'] == 1 for pf in path_frequencies)
    assert all(pf['percentage'] == 1.0 for pf in path_frequencies)
```

**Test 2.4: Frequency Calculation - Single Dominant Path**
```python
def test_path_frequency_single_dominant():
    """
    Test: 90% of videos follow same path

    Input:
    - Path [0,1,1,2,0,1]: 90 videos (90%)
    - Other paths: 10 videos distributed

    Expected Output:
    - Dominant path has 90% frequency
    - Confidence level: very_high
    """
    video_paths = create_synthetic_paths(distribution={
        (0,1,1,2,0,1): 90,
        # ... rest distributed
    }, total=100)

    path_frequencies = analyze_path_frequencies(video_paths)

    assert path_frequencies[0]['frequency'] == 90
    assert path_frequencies[0]['percentage'] == 90.0
```

**Test 2.5: Path Frequency - Small Sample Size**
```python
def test_path_frequency_small_sample():
    """
    Test: Bucket with only 50 videos (below typical 100)

    Input:
    - 50 videos total
    - Path A: 10 videos (20%)
    - Path B: 8 videos (16%)
    - Path C: 6 videos (12%)

    Expected Output:
    - Frequencies calculated correctly
    - 10% threshold = 5 videos minimum (not 10)
    """
    video_paths = create_synthetic_paths(distribution={
        (0,1,1,2,0,1): 10,
        (1,0,0,1,1,0): 8,
        (0,0,1,1,2,2): 6,
    }, total=50)

    path_frequencies = analyze_path_frequencies(video_paths)

    assert path_frequencies[0]['percentage'] == 20.0  # 10/50
    assert path_frequencies[1]['percentage'] == 16.0  # 8/50
    assert path_frequencies[2]['percentage'] == 12.0  # 6/50
```

---

#### **Category 3: 10% Threshold Filtering** (6-10 tests)

**Test 3.1: Basic Threshold Filtering**
```python
def test_threshold_filtering_10_percent():
    """
    Test: Filter paths to only include ≥10%

    Input:
    - 5 paths with frequencies: 22%, 18%, 12%, 8%, 5%

    Expected Output:
    - 3 paths returned (22%, 18%, 12%)
    - 8% and 5% excluded
    """
    all_paths = [
        {"frequency": 22, "percentage": 22.0, "path": [0,1,1,2,0,1]},
        {"frequency": 18, "percentage": 18.0, "path": [1,0,0,1,1,0]},
        {"frequency": 12, "percentage": 12.0, "path": [0,0,1,1,2,2]},
        {"frequency": 8, "percentage": 8.0, "path": [2,2,0,0,1,1]},
        {"frequency": 5, "percentage": 5.0, "path": [1,1,1,1,1,1]},
    ]

    filtered = filter_paths_by_threshold(all_paths, min_percentage=10.0)

    assert len(filtered) == 3
    assert all(p['percentage'] >= 10.0 for p in filtered)
    assert filtered[0]['percentage'] == 22.0
    assert filtered[2]['percentage'] == 12.0
```

**Test 3.2: Threshold Filtering - Exactly 10%**
```python
def test_threshold_filtering_exact_10_percent():
    """
    Test: Path with exactly 10% frequency is INCLUDED

    Input:
    - Path with 10 videos out of 100 (10.0%)

    Expected Output:
    - Path is included (≥10%, not >10%)
    """
    paths = [
        {"frequency": 10, "percentage": 10.0, "path": [0,1,1,2,0,1]},
        {"frequency": 9, "percentage": 9.0, "path": [1,0,0,1,1,0]},
    ]

    filtered = filter_paths_by_threshold(paths, min_percentage=10.0)

    assert len(filtered) == 1
    assert filtered[0]['percentage'] == 10.0
```

**Test 3.3: Threshold Filtering - No Paths Meet Threshold**
```python
def test_threshold_filtering_no_paths_qualify():
    """
    Test: All paths below 10% threshold

    Input:
    - Maximum path frequency is 9%

    Expected Output:
    - Empty list (no paths meet threshold)
    - Fallback logic will handle this in later stage
    """
    paths = [
        {"frequency": 9, "percentage": 9.0, "path": [0,1,1,2,0,1]},
        {"frequency": 8, "percentage": 8.0, "path": [1,0,0,1,1,0]},
        {"frequency": 7, "percentage": 7.0, "path": [0,0,1,1,2,2]},
    ]

    filtered = filter_paths_by_threshold(paths, min_percentage=10.0)

    assert len(filtered) == 0
```

**Test 3.4: Threshold Filtering - All Paths Meet Threshold**
```python
def test_threshold_filtering_all_qualify():
    """
    Test: All paths have ≥10% frequency (low fragmentation)

    Input:
    - Path A: 40%, Path B: 35%, Path C: 25%

    Expected Output:
    - All 3 paths returned
    """
    paths = [
        {"frequency": 40, "percentage": 40.0, "path": [0,1,1,2,0,1]},
        {"frequency": 35, "percentage": 35.0, "path": [1,0,0,1,1,0]},
        {"frequency": 25, "percentage": 25.0, "path": [0,0,1,1,2,2]},
    ]

    filtered = filter_paths_by_threshold(paths, min_percentage=10.0)

    assert len(filtered) == 3
```

---

#### **Category 4: Confidence Level Classification** (4-6 tests)

**Test 4.1: Confidence Level Assignment**
```python
def test_confidence_level_classification():
    """
    Test: Assign confidence levels based on frequency percentages

    Input:
    - Path A: 25% (very_high: ≥20%)
    - Path B: 17% (high: 15-20%)
    - Path C: 11% (moderate: 10-15%)

    Expected Output:
    - Correct confidence_level assigned to each
    """
    paths = [
        {"frequency": 25, "percentage": 25.0},
        {"frequency": 17, "percentage": 17.0},
        {"frequency": 11, "percentage": 11.0},
    ]

    classified = assign_confidence_levels(paths)

    assert classified[0]['confidence_level'] == 'very_high'
    assert classified[1]['confidence_level'] == 'high'
    assert classified[2]['confidence_level'] == 'moderate'
```

**Test 4.2: Confidence Boundaries**
```python
def test_confidence_level_boundaries():
    """
    Test: Boundary values for confidence classification

    Input:
    - 20.0%: very_high (boundary)
    - 15.0%: high (boundary)
    - 10.0%: moderate (boundary)

    Expected Output:
    - Boundaries correctly classified
    """
    paths = [
        {"frequency": 20, "percentage": 20.0},  # Exactly 20%
        {"frequency": 15, "percentage": 15.0},  # Exactly 15%
        {"frequency": 10, "percentage": 10.0},  # Exactly 10%
    ]

    classified = assign_confidence_levels(paths)

    assert classified[0]['confidence_level'] == 'very_high'  # ≥20%
    assert classified[1]['confidence_level'] == 'high'       # ≥15%
    assert classified[2]['confidence_level'] == 'moderate'   # ≥10%
```

**Test 4.3: All Very High Confidence**
```python
def test_confidence_all_very_high():
    """
    Test: Low fragmentation - all paths have ≥20%

    Input:
    - Path A: 40%, Path B: 35%, Path C: 25%

    Expected Output:
    - All classified as very_high
    """
    paths = [
        {"frequency": 40, "percentage": 40.0},
        {"frequency": 35, "percentage": 35.0},
        {"frequency": 25, "percentage": 25.0},
    ]

    classified = assign_confidence_levels(paths)

    assert all(p['confidence_level'] == 'very_high' for p in classified)
```

---

#### **Category 5: Fallback Logic** (6-8 tests)

**Test 5.1: Fallback - Only 2 Paths Meet Threshold**
```python
def test_fallback_logic_two_paths():
    """
    Test: Generate 3 reports when only 2 paths meet 10% threshold

    Input:
    - 2 paths with ≥10% frequency
    - rf_video_data available for feature-based fallback

    Expected Output:
    - 3 creative_reports total
    - First 2 are path-based (type='path_based')
    - Third is feature-based fallback (type='feature_based')
    """
    paths = [
        {"frequency": 15, "percentage": 15.0, "confidence_level": "high", "path": [0,1,1,2,0,1]},
        {"frequency": 12, "percentage": 12.0, "confidence_level": "moderate", "path": [1,0,0,1,1,0]},
    ]

    rf_video_data = load_synthetic_rf_video_analysis()

    reports = generate_creative_reports(paths, rf_video_data, bucket='18-33s')

    # Should return exactly 3 reports
    assert len(reports) == 3

    # First 2 are path-based
    assert reports[0]['type'] == 'path_based'
    assert reports[0]['frequency'] == 15
    assert reports[1]['type'] == 'path_based'
    assert reports[1]['frequency'] == 12

    # Third is feature-based fallback
    assert reports[2]['type'] == 'feature_based'
    assert 'universal_features' in reports[2]
    assert len(reports[2]['universal_features']) >= 5  # Top 5-7 RF features
```

**Test 5.2: Fallback - Only 1 Path Meets Threshold**
```python
def test_fallback_logic_one_path():
    """
    Test: Generate 3 reports when only 1 path meets 10% threshold

    Input:
    - 1 path with ≥10% frequency

    Expected Output:
    - 1 path-based report
    - 2 feature-based fallback reports
    """
    paths = [
        {"frequency": 12, "percentage": 12.0, "confidence_level": "moderate", "path": [0,1,1,2,0,1]},
    ]

    rf_video_data = load_synthetic_rf_video_analysis()

    reports = generate_creative_reports(paths, rf_video_data, bucket='18-33s')

    assert len(reports) == 3
    assert reports[0]['type'] == 'path_based'
    assert reports[1]['type'] == 'feature_based'
    assert reports[2]['type'] == 'feature_based'
```

**Test 5.3: Fallback - No Paths Meet Threshold (Extreme Fragmentation)**
```python
def test_fallback_logic_no_paths():
    """
    Test: Generate 3 feature-based reports when all paths <10%

    Input:
    - Maximum path frequency is 9%
    - Extreme fragmentation (100 videos, ~50 unique paths)

    Expected Output:
    - 3 feature-based reports (all based on universal RF features)
    - No path-based reports
    - supplementary_insights includes cross-window patterns
    """
    paths = []  # No paths meet 10% threshold

    rf_video_data = load_synthetic_rf_video_analysis()

    reports = generate_creative_reports(paths, rf_video_data, bucket='18-33s')

    assert len(reports) == 3
    assert all(r['type'] == 'feature_based' for r in reports)

    # Each report focuses on different RF feature subsets
    assert reports[0]['primary_feature'] != reports[1]['primary_feature']
    assert reports[1]['primary_feature'] != reports[2]['primary_feature']
```

**Test 5.4: Fallback - Feature-Based Report Structure**
```python
def test_fallback_feature_based_structure():
    """
    Test: Feature-based fallback reports have correct structure

    Expected Structure:
    - type: 'feature_based'
    - primary_feature: Top RF feature used for this report
    - universal_features: Top 5-7 video-level RF features
    - creator_recommendations: Actionable steps based on RF data
    """
    rf_video_data = load_synthetic_rf_video_analysis()

    report = generate_feature_based_report(rf_video_data, report_id=1)

    # Validate structure
    assert report['type'] == 'feature_based'
    assert 'primary_feature' in report
    assert 'universal_features' in report
    assert len(report['universal_features']) >= 5
    assert 'creator_recommendations' in report

    # Universal features come from top RF importance
    top_rf_feature = rf_video_data['feature_importance'][0]['feature']
    assert top_rf_feature in [uf['feature'] for uf in report['universal_features']]
```

---

#### **Category 6: Edge Case Buckets** (4-6 tests)

**Test 6.1: Bucket 0-3s - Single Window, No Phase 2**
```python
def test_bucket_0_3s_single_window_no_phase2():
    """
    Test: Bucket 0-3s has only hook window, skips Phase 2

    Input:
    - 1 window: hook
    - 50 videos

    Expected Output:
    - Phase 1: hook_analysis.json generated
    - Phase 2: SKIPPED (no winning_formulas.json)
    - Output: bucket_summary_0-3s.json with 3 hook strategies
    """
    bucket_path = 'tests/fixtures/synthetic/bucket_0-3s'

    result = run_stage7_llm_analysis(bucket_path, bucket='0-3s')

    # Phase 1 output exists
    assert os.path.exists(f'{bucket_path}/ml_analysis/llm/hook_analysis.json')

    # Phase 2 output does NOT exist
    assert not os.path.exists(f'{bucket_path}/ml_analysis/llm/winning_formulas.json')

    # Summary file exists
    assert os.path.exists(f'{bucket_path}/ml_analysis/llm/bucket_summary_0-3s.json')

    # Summary contains 3 hook strategies (from 3 clusters)
    summary = load_json(f'{bucket_path}/ml_analysis/llm/bucket_summary_0-3s.json')
    assert len(summary['hook_strategies']) == 3
```

**Test 6.2: Bucket 3-9s - 2 Windows, Minimal Paths**
```python
def test_bucket_3_9s_two_windows():
    """
    Test: Bucket 3-9s has hook + closing, 9 possible paths (3^2)

    Input:
    - 2 windows: hook, closing
    - 50 videos

    Expected Output:
    - Phase 1: 2 JSONs (hook_analysis.json, closing_analysis.json)
    - Phase 2: winning_formulas.json with paths of length 2
    - Maximum 9 unique paths possible
    """
    bucket_path = 'tests/fixtures/synthetic/bucket_3-9s'

    result = run_stage7_llm_analysis(bucket_path, bucket='3-9s')

    # Phase 1 outputs
    assert os.path.exists(f'{bucket_path}/ml_analysis/llm/hook_analysis.json')
    assert os.path.exists(f'{bucket_path}/ml_analysis/llm/closing_analysis.json')

    # Phase 2 output
    winning_formulas = load_json(f'{bucket_path}/ml_analysis/llm/winning_formulas.json')

    # All paths have 2 positions
    for formula in winning_formulas['winning_formulas']:
        assert len(formula['cluster_path']) == 2

    # Cannot exceed 9 unique paths (3^2)
    total_unique_paths = len(set(tuple(f['cluster_path']) for f in winning_formulas['winning_formulas']))
    assert total_unique_paths <= 9
```

**Test 6.3: Bucket with middle_aggregate**
```python
def test_bucket_middle_aggregate_9_13s():
    """
    Test: Buckets 9-13s and 13-18s use middle_aggregate window

    Input:
    - 3 windows: hook, middle_aggregate, closing
    - 50 videos

    Expected Output:
    - Phase 1: 3 JSONs (hook, middle_aggregate, closing)
    - Phase 2: winning_formulas with 3-position paths
    - path_str includes "Middle_Aggregate-C1" (not "Middle_1")
    """
    bucket_path = 'tests/fixtures/synthetic/bucket_9-13s'

    result = run_stage7_llm_analysis(bucket_path, bucket='9-13s')

    # Phase 1: middle_aggregate analysis exists
    assert os.path.exists(f'{bucket_path}/ml_analysis/llm/middle_aggregate_analysis.json')

    # Phase 2: paths reference middle_aggregate
    winning_formulas = load_json(f'{bucket_path}/ml_analysis/llm/winning_formulas.json')

    first_formula = winning_formulas['winning_formulas'][0]
    assert 'Middle_Aggregate' in first_formula['structure']['middle_pattern']
    assert len(first_formula['cluster_path']) == 3
```

---

#### **Category 7: Hybrid Output Structure** (4-5 tests)

**Test 7.1: Hybrid Output - creative_reports + supplementary_insights**
```python
def test_hybrid_output_structure():
    """
    Test: Phase 2 output includes both creative_reports and supplementary_insights

    Expected Structure:
    {
      "creative_reports": [3 reports],
      "supplementary_insights": {
        "universal_principles": [...],
        "cross_window_patterns": [...]
      }
    }
    """
    paths = [
        {"frequency": 22, "percentage": 22.0, "confidence_level": "very_high", "path": [0,1,1,2,0,1]},
        {"frequency": 18, "percentage": 18.0, "confidence_level": "high", "path": [1,0,0,1,1,0]},
        {"frequency": 12, "percentage": 12.0, "confidence_level": "moderate", "path": [0,0,1,1,2,2]},
    ]

    rf_video_data = load_synthetic_rf_video_analysis()

    output = generate_phase2_output(paths, rf_video_data, bucket='18-33s')

    # Validate structure
    assert 'creative_reports' in output
    assert 'supplementary_insights' in output

    # creative_reports has 3 reports
    assert len(output['creative_reports']) == 3

    # supplementary_insights has universal principles
    assert 'universal_principles' in output['supplementary_insights']
    assert len(output['supplementary_insights']['universal_principles']) >= 5

    # supplementary_insights has cross-window patterns
    assert 'cross_window_patterns' in output['supplementary_insights']
```

**Test 7.2: Supplementary Insights - Universal Principles**
```python
def test_supplementary_insights_universal_principles():
    """
    Test: supplementary_insights.universal_principles contains top 5-7 RF features

    Expected Content:
    - Top video-level RF features (e.g., hook_eye_contact_rate)
    - Each principle has: feature, importance, top_avg, bottom_avg, gap
    - Sorted by RF importance (descending)
    """
    rf_video_data = load_synthetic_rf_video_analysis()

    supplementary = generate_supplementary_insights(rf_video_data)

    principles = supplementary['universal_principles']

    # Should have 5-7 features
    assert 5 <= len(principles) <= 7

    # First principle should be highest RF importance
    assert principles[0]['importance'] > principles[1]['importance']

    # Each principle has required fields
    for principle in principles:
        assert 'feature' in principle
        assert 'importance' in principle
        assert 'top_performer_avg' in principle
        assert 'bottom_performer_avg' in principle
        assert 'gap' in principle
```

**Test 7.3: Supplementary Insights - Cross-Window Patterns**
```python
def test_supplementary_insights_cross_window_patterns():
    """
    Test: supplementary_insights.cross_window_patterns contains cross-window RF features

    Expected Content:
    - Features like: hook_to_middle_energy_delta, eye_contact_consistency
    - Only features marked as pattern_type='cross_window' in RF data
    """
    rf_video_data = load_synthetic_rf_video_analysis()

    supplementary = generate_supplementary_insights(rf_video_data)

    cross_window = supplementary['cross_window_patterns']

    # Should have 2-4 cross-window patterns
    assert 2 <= len(cross_window) <= 4

    # Each pattern is labeled as cross_window in RF data
    for pattern in cross_window:
        feature_name = pattern['feature']
        # Find in RF data
        rf_feature = next(f for f in rf_video_data['feature_importance'] if f['feature'] == feature_name)
        assert rf_feature.get('pattern_type') == 'cross_window'
```

---

### 3.2 Test Execution Strategy

**Order of Execution**:
1. **Category 1-2** (Cluster Path Extraction, Frequency Calculation): Foundation tests
2. **Category 3-4** (Threshold Filtering, Confidence Levels): Core logic tests
3. **Category 5** (Fallback Logic): Integration of earlier components
4. **Category 6** (Edge Case Buckets): Special case handling
5. **Category 7** (Hybrid Output Structure): Output validation

**Running Tests**:
```bash
# Run all pre-LLM logic tests
pytest tests/unit/test_pre_llm_logic.py -v

# Run specific category
pytest tests/unit/test_pre_llm_logic.py::TestClusterPathExtraction -v

# Run with coverage
pytest tests/unit/test_pre_llm_logic.py --cov=stage7_llm_analysis --cov-report=html

# Run only fast tests (no API calls)
pytest tests/unit/ -v -m "not api_call"
```

---

## 4. LLM Integration Testing (Integration Tests)

### 4.1 Test Categories

**Total Expected**: 5-10 integration tests with real API calls

#### **Test 4.1: End-to-End Phase 1 (Single Window)**
```python
@pytest.mark.api_call  # Marks test as requiring API (costs money)
def test_phase1_single_window_real_api():
    """
    Test: Phase 1 analysis for single window with real Anthropic API

    Input:
    - Real Stage 6 outputs (hook_kmeans_analysis.json + hook_rf_analysis.json)
    - Bucket: 18-33s
    - Window: hook

    Expected Output:
    - hook_analysis.json generated
    - JSON structure valid (has 'window_type', 'clusters' fields)
    - 3 clusters with named strategies
    - creator_recommendations include actionable steps

    Cost: ~$0.02-0.04 per run
    """
    bucket_path = 'tests/fixtures/real/bucket_18-33s_10videos'

    # Load real Stage 6 data
    kmeans_data = load_json(f'{bucket_path}/ml_analysis/hook_kmeans_analysis.json')
    rf_data = load_json(f'{bucket_path}/ml_analysis/hook_rf_analysis.json')

    # Make real API call
    analysis = analyze_window(
        window_type='hook',
        kmeans_data=kmeans_data,
        rf_data=rf_data,
        bucket='18-33s',
        hashtag='#nutrition'
    )

    # Validate structure
    assert analysis['window_type'] == 'hook'
    assert len(analysis['clusters']) == 3

    # Each cluster has required fields
    for cluster in analysis['clusters']:
        assert 'cluster_id' in cluster
        assert 'name' in cluster  # LLM generated name
        assert 'defining_features' in cluster
        assert 'creator_recommendations' in cluster
        assert len(cluster['creator_recommendations']) >= 3
```

#### **Test 4.2: End-to-End Phase 1 (All Windows, Parallel)**
```python
@pytest.mark.api_call
def test_phase1_all_windows_parallel_real_api():
    """
    Test: Phase 1 for all windows in parallel with real API

    Input:
    - Real Stage 6 outputs (all 13 JSONs)
    - Bucket: 18-33s (6 windows)

    Expected Output:
    - 6 window analysis JSONs generated in parallel
    - All complete within 15-20 seconds (wall-clock time)
    - No failures (100% success rate)

    Cost: ~$0.12-0.18 per run (6 windows × ~$0.02)
    """
    bucket_path = 'tests/fixtures/real/bucket_18-33s_10videos'

    # Run Phase 1 in parallel
    start_time = time.time()
    window_analyses = run_phase1_parallel(
        bucket_path=bucket_path,
        bucket='18-33s',
        hashtag='#nutrition',
        window_types=['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']
    )
    elapsed_time = time.time() - start_time

    # All windows completed
    assert len(window_analyses) == 6

    # Parallel execution is fast (<20s wall-clock, not 60s sequential)
    assert elapsed_time < 20

    # All outputs valid
    for window_type, analysis in window_analyses.items():
        assert analysis['window_type'] == window_type
        assert len(analysis['clusters']) == 3
```

#### **Test 4.3: End-to-End Phase 2 (Cross-Window Synthesis)**
```python
@pytest.mark.api_call
def test_phase2_synthesis_real_api():
    """
    Test: Phase 2 cross-window synthesis with real API

    Input:
    - 6 Phase 1 analyses (from previous test or pre-generated)
    - Real Stage 6 rf_video_analysis.json

    Expected Output:
    - winning_formulas.json generated
    - 3-5 winning formulas (filtered to ≥10% frequency)
    - Each formula has temporal_progressions
    - rf_cross_window_validation included

    Cost: ~$0.08-0.12 per run
    """
    bucket_path = 'tests/fixtures/real/bucket_18-33s_10videos'

    # Load Phase 1 outputs (or run Phase 1 first)
    window_analyses = load_phase1_analyses(bucket_path)
    rf_video_data = load_json(f'{bucket_path}/ml_analysis/rf_video_analysis.json')
    kmeans_outputs = load_kmeans_outputs(bucket_path)

    # Run Phase 2
    synthesis = run_phase2_synthesis(
        window_analyses=window_analyses,
        kmeans_outputs=kmeans_outputs,
        rf_video_data=rf_video_data,
        bucket='18-33s',
        hashtag='#nutrition'
    )

    # Validate structure
    assert 'winning_formulas' in synthesis
    assert len(synthesis['winning_formulas']) >= 3  # At least 3 (may have fallback)

    # Each formula has required fields
    for formula in synthesis['winning_formulas']:
        assert 'name' in formula
        assert 'cluster_path' in formula
        assert 'frequency' in formula
        assert 'temporal_progressions' in formula
        assert 'rf_cross_window_validation' in formula
```

#### **Test 4.4: Complete Stage 7 Pipeline (Phase 1 + Phase 2)**
```python
@pytest.mark.api_call
@pytest.mark.slow  # Marks as slow test (takes 30-40s)
def test_complete_stage7_pipeline_real_api():
    """
    Test: Full Stage 7 execution (Phase 1 + Phase 2) with real API

    Input:
    - Real Stage 6 outputs (bucket 18-33s, 10 videos)

    Expected Output:
    - 8 output files generated:
      - 6 Phase 1 JSONs (window analyses)
      - 1 Phase 2 JSON (winning_formulas)
      - 1 complete_analysis JSON (combined)
    - Total time: 25-35 seconds
    - Total cost: ~$0.20-0.30

    Cost: ~$0.20-0.30 per run
    """
    bucket_path = 'tests/fixtures/real/bucket_18-33s_10videos'

    # Run complete Stage 7
    result = run_stage7_llm_analysis(
        bucket_path=bucket_path,
        bucket='18-33s',
        hashtag='#nutrition'
    )

    # Validate all outputs exist
    llm_dir = f'{bucket_path}/ml_analysis/llm'
    assert os.path.exists(f'{llm_dir}/hook_analysis.json')
    assert os.path.exists(f'{llm_dir}/middle_1_analysis.json')
    assert os.path.exists(f'{llm_dir}/middle_2_analysis.json')
    assert os.path.exists(f'{llm_dir}/middle_3_analysis.json')
    assert os.path.exists(f'{llm_dir}/middle_4_analysis.json')
    assert os.path.exists(f'{llm_dir}/closing_analysis.json')
    assert os.path.exists(f'{llm_dir}/winning_formulas.json')
    assert os.path.exists(f'{llm_dir}/complete_analysis_18-33s.json')

    # Validate execution metrics
    assert result['execution_metrics']['total_time_seconds'] < 40
    assert result['execution_metrics']['api_calls'] == 7  # 6 Phase 1 + 1 Phase 2
```

#### **Test 4.5: API Retry Logic (Simulated Failure)**
```python
@pytest.mark.api_call
def test_api_retry_logic_with_mock_failure():
    """
    Test: Smart retry logic when window API call fails

    Scenario:
    - Phase 1: 6 parallel calls
    - Window 3 (middle_2) fails with 503 error
    - Retry only middle_2 (not all 6)
    - Second attempt succeeds

    Expected:
    - Total API calls: 7 (6 initial + 1 retry)
    - All 6 windows complete
    - Execution time: ~12-15s (includes 2s backoff wait)
    """
    bucket_path = 'tests/fixtures/real/bucket_18-33s_10videos'

    # Mock failure for middle_2
    with mock_api_failure(window='middle_2', status_code=503, fail_count=1):
        window_analyses = run_phase1_parallel(
            bucket_path=bucket_path,
            bucket='18-33s',
            hashtag='#nutrition',
            window_types=['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']
        )

    # All 6 windows completed despite middle_2 failure
    assert len(window_analyses) == 6

    # Check retry log
    logs = get_test_logs()
    assert 'Retry attempt 1' in logs
    assert 'middle_2' in logs
```

---

### 4.2 Test Execution Strategy

**Running Integration Tests**:
```bash
# Run all integration tests (costs ~$1-2 total)
pytest tests/integration/test_llm_integration.py -v -m "api_call"

# Run single test
pytest tests/integration/test_llm_integration.py::test_phase1_single_window_real_api -v

# Skip expensive tests (run only pre-LLM logic)
pytest tests/ -v -m "not api_call"

# Run only fast integration tests
pytest tests/integration/ -v -m "api_call and not slow"
```

**CI/CD Strategy**:
- **PR Validation**: Run only pre-LLM logic tests (FREE, fast)
- **Nightly Builds**: Run all integration tests with real API (costs ~$2, runs once per day)
- **Pre-Release**: Run full integration suite on all 8 buckets (costs ~$10-15)

---

## 5. Output Validation Testing

### 5.1 Schema Validation (Deterministic)

**Test 5.1: Phase 1 Output Schema**
```python
def test_phase1_output_schema_validation():
    """
    Test: Validate Phase 1 output matches expected JSON schema

    Cannot test exact string content (LLM non-deterministic),
    but CAN test structure and required fields.
    """
    analysis = load_json('tests/fixtures/real/bucket_18-33s/ml_analysis/llm/hook_analysis.json')

    # Required top-level fields
    assert 'window_type' in analysis
    assert 'bucket' in analysis
    assert 'hashtag' in analysis
    assert 'clusters' in analysis

    # Exactly 3 clusters
    assert len(analysis['clusters']) == 3

    # Each cluster has required fields
    for cluster in analysis['clusters']:
        assert 'cluster_id' in cluster
        assert 'size' in cluster
        assert 'name' in cluster  # LLM-generated
        assert 'defining_features' in cluster
        assert 'strategy_description' in cluster
        assert 'creator_recommendations' in cluster

        # defining_features is non-empty list
        assert isinstance(cluster['defining_features'], list)
        assert len(cluster['defining_features']) >= 3

        # creator_recommendations is non-empty list
        assert isinstance(cluster['creator_recommendations'], list)
        assert len(cluster['creator_recommendations']) >= 3
```

**Test 5.2: Phase 2 Output Schema**
```python
def test_phase2_output_schema_validation():
    """
    Test: Validate Phase 2 output matches expected JSON schema
    """
    synthesis = load_json('tests/fixtures/real/bucket_18-33s/ml_analysis/llm/winning_formulas.json')

    # Required top-level fields
    assert 'creative_reports' in synthesis
    assert 'supplementary_insights' in synthesis
    assert 'bucket' in synthesis
    assert 'hashtag' in synthesis

    # creative_reports has 3 reports
    assert len(synthesis['creative_reports']) == 3

    # Each report has required fields
    for report in synthesis['creative_reports']:
        assert 'report_id' in report
        assert 'type' in report  # 'path_based' or 'feature_based'
        assert report['type'] in ['path_based', 'feature_based']

        if report['type'] == 'path_based':
            assert 'frequency' in report
            assert 'percentage' in report
            assert 'confidence_level' in report
            assert report['confidence_level'] in ['very_high', 'high', 'moderate']

    # supplementary_insights has required fields
    assert 'universal_principles' in synthesis['supplementary_insights']
    assert 'cross_window_patterns' in synthesis['supplementary_insights']
```

### 5.2 Semantic Content Validation (Semi-Deterministic)

**Test 5.3: RF Feature Mention Validation**
```python
def test_llm_output_mentions_top_rf_features():
    """
    Test: LLM output mentions top RF features from input data

    Rationale:
    - If top RF feature is "eye_contact_rate" (importance 0.35),
      LLM SHOULD mention it in recommendations
    - Not exact string matching, but check feature is referenced
    """
    analysis = load_json('tests/fixtures/real/bucket_18-33s/ml_analysis/llm/hook_analysis.json')
    rf_data = load_json('tests/fixtures/real/bucket_18-33s/ml_analysis/hook_rf_analysis.json')

    # Top 3 RF features
    top_features = [f['feature'] for f in rf_data['feature_importance'][:3]]
    # e.g., ['eye_contact_rate', 'energy_level', 'word_count']

    # Convert analysis to text for searching
    analysis_text = json.dumps(analysis).lower()

    # At least 2 of top 3 features should be mentioned
    mentions = sum(1 for feature in top_features if feature.lower() in analysis_text)
    assert mentions >= 2, f"Expected at least 2 top RF features mentioned, found {mentions}"
```

**Test 5.4: Cluster Size Consistency**
```python
def test_cluster_sizes_match_kmeans_input():
    """
    Test: LLM reports cluster sizes that match K-Means input data

    Rationale:
    - K-Means input says Cluster 0 has 35 videos
    - LLM output MUST report "size: 35" (this is factual, not interpretive)
    """
    analysis = load_json('tests/fixtures/real/bucket_18-33s/ml_analysis/llm/hook_analysis.json')
    kmeans_data = load_json('tests/fixtures/real/bucket_18-33s/ml_analysis/hook_kmeans_analysis.json')

    # Compare cluster sizes
    for i in range(3):
        llm_size = analysis['clusters'][i]['size']
        kmeans_size = kmeans_data['clusters'][i]['size']
        assert llm_size == kmeans_size, f"Cluster {i} size mismatch: LLM={llm_size}, K-Means={kmeans_size}"
```

---

## 6. Automated Validation Layer (Critique Q3 - Layer 1)

### 6.1 Post-LLM Validation Checks

**Test 6.1: Feature Value Contradiction Detection**
```python
def test_automated_validation_feature_contradictions():
    """
    Test: Detect when LLM reports feature value contradicting source data

    Example Contradiction:
    - K-Means centroid: eye_contact_rate = 0.22 (low)
    - LLM says: "high eye contact (0.85)"

    Expected: ValidationError raised
    """
    # Mock LLM response with contradiction
    llm_output = {
        "cluster_id": 0,
        "defining_features": [
            "eye_contact_rate: 0.85 (high eye contact)"  # WRONG! Centroid shows 0.22
        ]
    }

    kmeans_centroid = {
        "eye_contact_rate": 0.22,  # Actual value
        "word_count": 14.5,
    }

    # Run validation
    with pytest.raises(ValidationError) as exc_info:
        validate_feature_values(llm_output, kmeans_centroid)

    assert "eye_contact_rate" in str(exc_info.value)
    assert "contradiction" in str(exc_info.value).lower()
```

**Test 6.2: Invented Feature Detection**
```python
def test_automated_validation_invented_features():
    """
    Test: Detect when LLM references features not in source data

    Example Invented Feature:
    - LLM mentions: "background_blur: 0.75"
    - But K-Means centroid has no such feature

    Expected: ValidationError raised
    """
    llm_output = {
        "cluster_id": 0,
        "defining_features": [
            "eye_contact_rate: 0.87",
            "background_blur: 0.75"  # INVENTED! Not in K-Means data
        ]
    }

    kmeans_centroid = {
        "eye_contact_rate": 0.87,
        "word_count": 14.5,
        # No "background_blur" feature
    }

    with pytest.raises(ValidationError) as exc_info:
        validate_no_invented_features(llm_output, kmeans_centroid)

    assert "background_blur" in str(exc_info.value)
    assert "invented" in str(exc_info.value).lower()
```

**Test 6.3: RF Priority Validation**
```python
def test_automated_validation_rf_priority():
    """
    Test: Detect when LLM ignores top RF features in recommendations

    Example RF Misalignment:
    - Top RF feature: eye_contact_rate (importance 0.35, rank #1)
    - LLM recommendations: No mention of eye_contact

    Expected: Warning logged (not fatal error, but suspicious)
    """
    llm_output = {
        "cluster_id": 0,
        "creator_recommendations": [
            "Use 3-4 scene cuts in first 3 seconds",
            "Add text overlays within 2 seconds",
            "Speak quickly - 45-50 words"
            # NO mention of eye_contact (top RF feature!)
        ]
    }

    rf_data = {
        "feature_importance": [
            {"feature": "eye_contact_rate", "importance": 0.35, "rank": 1},
            {"feature": "scene_count", "importance": 0.22, "rank": 2},
            {"feature": "word_count", "importance": 0.18, "rank": 3}
        ]
    }

    # Run validation
    warnings = validate_rf_feature_priority(llm_output, rf_data, top_n=3)

    # Should warn that top RF feature is not mentioned
    assert len(warnings) > 0
    assert "eye_contact_rate" in warnings[0]
```

---

## 7. Test Data Maintenance

### 7.1 Synthetic Data Generation

**Script**: `tests/fixtures/synthetic/generate_synthetic_fixtures.py`

```python
def generate_synthetic_bucket(bucket: str, num_videos: int, path_distribution: dict):
    """
    Generate synthetic Stage 6 outputs for testing.

    Args:
        bucket: Bucket name (e.g., "18-33s")
        num_videos: Total videos (e.g., 100)
        path_distribution: Dict mapping path tuples to frequencies
            e.g., {(0,1,1,2,0,1): 22, (1,0,0,1,1,0): 18, ...}

    Outputs:
        - rf_video_analysis.json
        - {window}_rf_analysis.json (6-7 files)
        - {window}_kmeans_analysis.json (6-7 files)
    """
    # Implementation details...
```

**Usage**:
```bash
# Generate all synthetic fixtures
python tests/fixtures/synthetic/generate_synthetic_fixtures.py --all

# Generate specific bucket
python tests/fixtures/synthetic/generate_synthetic_fixtures.py --bucket 18-33s --videos 100

# Generate edge case (extreme fragmentation)
python tests/fixtures/synthetic/generate_synthetic_fixtures.py --bucket 18-33s --videos 100 --fragmentation high
```

### 7.2 Real Data Collection

**Process**:
1. Run video processing through Stage 2.6
2. Continue through Stage 3 → Stage 4 → Stage 5 → Stage 6
3. Copy Stage 6 outputs to `tests/fixtures/real/`
4. Document source (hashtag, date, video count) in README.md

**Example**:
```bash
# After processing real videos
cp -r /data/clients/acme/buckets/bucket_18-33s/ml_analysis tests/fixtures/real/bucket_18-33s_10videos/

# Document source
echo "Source: #nutrition hashtag, processed 2025-01-28, 10 videos" > tests/fixtures/real/bucket_18-33s_10videos/README.md
```

---

## 8. Testing Checklist

### 8.1 Pre-LLM Logic Tests (Must Pass Before LLM Integration)

- [ ] **Cluster Path Extraction** (8 tests)
  - [ ] 6-window bucket (18-33s)
  - [ ] 3-window bucket with middle_aggregate (9-13s, 13-18s)
  - [ ] 2-window bucket (3-9s)
  - [ ] 1-window bucket (0-3s) - no paths
  - [ ] Missing window data error handling

- [ ] **Path Frequency Calculation** (12 tests)
  - [ ] Basic frequency calculation
  - [ ] Tied frequencies
  - [ ] All paths unique (extreme fragmentation)
  - [ ] Single dominant path (90%)
  - [ ] Small sample size (50 videos)

- [ ] **10% Threshold Filtering** (10 tests)
  - [ ] Basic threshold filtering
  - [ ] Exactly 10% boundary
  - [ ] No paths meet threshold
  - [ ] All paths meet threshold

- [ ] **Confidence Level Classification** (6 tests)
  - [ ] Confidence level assignment (very_high/high/moderate)
  - [ ] Boundary values (20%, 15%, 10%)
  - [ ] All very high confidence

- [ ] **Fallback Logic** (8 tests)
  - [ ] 2 paths meet threshold (1 fallback report)
  - [ ] 1 path meets threshold (2 fallback reports)
  - [ ] 0 paths meet threshold (3 fallback reports)
  - [ ] Feature-based report structure

- [ ] **Edge Case Buckets** (6 tests)
  - [ ] Bucket 0-3s (single window, no Phase 2)
  - [ ] Bucket 3-9s (2 windows)
  - [ ] Bucket with middle_aggregate

- [ ] **Hybrid Output Structure** (5 tests)
  - [ ] creative_reports + supplementary_insights
  - [ ] Universal principles
  - [ ] Cross-window patterns

### 8.2 LLM Integration Tests (Run After Pre-LLM Tests Pass)

- [ ] **Phase 1 Integration** (3 tests)
  - [ ] Single window analysis (real API)
  - [ ] All windows parallel (real API)
  - [ ] API retry logic

- [ ] **Phase 2 Integration** (2 tests)
  - [ ] Cross-window synthesis (real API)
  - [ ] Complete pipeline (Phase 1 + Phase 2)

- [ ] **Output Validation** (4 tests)
  - [ ] Phase 1 schema validation
  - [ ] Phase 2 schema validation
  - [ ] RF feature mention validation
  - [ ] Cluster size consistency

- [ ] **Automated Validation Layer** (3 tests)
  - [ ] Feature value contradiction detection
  - [ ] Invented feature detection
  - [ ] RF priority validation

---

## 9. Success Criteria

**Stage 7 is ready for production when**:

1. **Pre-LLM Logic**: All 50 unit tests pass (100% success rate)
2. **LLM Integration**: 8/10 integration tests pass (80% success rate acceptable due to LLM non-determinism)
3. **Automated Validation**: Hallucination detection catches >90% of feature contradictions/invented features
4. **Real Data Testing**: Successfully processes 10 real videos from Stage 2.6 → Stage 7 without errors
5. **All 8 Buckets**: Edge case buckets (0-3s, 3-9s, 9-13s) tested and working correctly

---

## 10. References

**Related Documents**:
- QA_LLMAnalysis.md - Q6: Testing Strategy (parent document)
- Critique_Stage7_LLMAnalysis.md - Q3: Automated Validation Layer (Layer 1)
- Critique_Stage7_LLMAnalysis.md - Q4: Smart Retry Logic
- Critique_Stage7_LLMAnalysis.md - Q5: 10% Threshold and Confidence Levels

**Test Fixtures**:
- `tests/fixtures/synthetic/` - Controlled test data for pre-LLM logic
- `tests/fixtures/real/` - Real Stage 6 outputs from actual video processing

**Test Execution**:
- `tests/unit/test_pre_llm_logic.py` - Pre-LLM logic tests (FREE, fast)
- `tests/integration/test_llm_integration.py` - LLM integration tests (costs ~$1-2 per run)

---

**Document Metadata**:
- **Creation Date**: 2025-01-28
- **Last Modified**: 2025-01-28
- **Authors**: Claude Code (QA Strategy Document Generator)
- **Reviewers**: [Pending]
- **Next Review Date**: [Pending]
