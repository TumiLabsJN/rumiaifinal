# Stage 7 LLM Analysis - Testing Documentation

**Document Version**: 1.0
**Date**: 2025-10-21
**Status**: Test Plan - Ready for Implementation
**Source**: LLMAnalysisCHILD.md Section 7 & 8 + Gap Analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Testing Strategy Overview](#2-testing-strategy-overview)
3. [HLD-Specified Tests](#3-hld-specified-tests)
4. [Gap Analysis & Recommendations](#4-gap-analysis--recommendations)
5. [Complete Test Suite](#5-complete-test-suite)
6. [Test Data Requirements](#6-test-data-requirements)
7. [Test Execution Plan](#7-test-execution-plan)
8. [Success Criteria](#8-success-criteria)

---

## 1. Executive Summary

### 1.1 Coverage Assessment

| Test Category | HLD Coverage | Recommended Coverage | Priority |
|--------------|--------------|---------------------|----------|
| **Unit Tests** | ✅ Good (60%) | 🟡 Add 40% | HIGH |
| **Integration Tests** | ✅ Good (70%) | 🟢 Add 30% | MEDIUM |
| **Edge Cases** | ✅ Good (50%) | 🟡 Add 50% | HIGH |
| **Checkpoint/Resume** | ❌ Missing (0%) | 🔴 Add 100% | **CRITICAL** |
| **Parallel Execution** | ❌ Missing (0%) | 🔴 Add 100% | **CRITICAL** |
| **Validation** | 🟡 Basic (40%) | 🟡 Add 60% | HIGH |

**Overall Assessment**: HLD tests cover ~55-60% of critical functionality. **Additional 40-45% coverage needed** for production readiness.

### 1.2 Risk Assessment

**High-Risk Untested Areas**:
1. ⚠️ **Checkpoint/Resume Logic** - Cost-critical feature ($4.18/client at risk)
2. ⚠️ **Parallel Execution** - Thread safety, race conditions
3. ⚠️ **Retry Logic** - Exponential backoff timing, retryable vs non-retryable errors
4. ⚠️ **Feature-Based Report Generation** - Python-generated JSON schema compliance
5. ⚠️ **Output Validation** - Hallucination detection (3-layer validation per TI Section 5)

### 1.3 Recommendations

**Before Production Deployment**:
- ✅ Implement all HLD-specified tests (~9 hours)
- 🔴 Implement HIGH priority gap tests (~3 hours)
- 🟡 Implement MEDIUM priority gap tests (~1 hour)

**Total Testing Effort**: ~13 hours

---

## 2. Testing Strategy Overview

### 2.1 Test Pyramid

```
                    ┌─────────────────┐
                    │  Integration    │  20% (4 hours)
                    │  Tests (E2E)    │
                    └─────────────────┘
                  ┌───────────────────────┐
                  │   Edge Case Tests     │  30% (3 hours)
                  │   (API, Buckets)      │
                  └───────────────────────┘
              ┌─────────────────────────────────┐
              │      Unit Tests                 │  50% (6 hours)
              │  (Preprocessing, Prompts, API)  │
              └─────────────────────────────────┘
```

### 2.2 Test Execution Modes

**Mode 1: Mock API (Fast, No Cost)**
- Unit tests with pre-recorded JSON fixtures
- Preprocessing function tests
- Prompt builder tests
- Cost: $0, Time: ~5 minutes

**Mode 2: Live API (Slow, ~$0.26/bucket)**
- Integration tests with real Anthropic API
- Full pipeline E2E tests
- Cost: ~$0.26 per bucket, Time: ~60 seconds per bucket

**Mode 3: Synthetic Data (Controlled)**
- Edge case testing with manipulated data
- Scenario B/C/D forcing
- Cost: $0 (mock) or ~$0.09 (live)

### 2.3 Test Data Sources

1. **Synthetic JSON Fixtures** (10-20 videos per bucket):
   - Controlled edge cases
   - Scenario forcing (A/B/C/D)
   - Malformed data testing

2. **Real Stage 6 Outputs** (100 videos):
   - Pilot hashtag data (#nutrition recommended)
   - 1-2 priority buckets (18-33s, 33-60s)
   - Full pipeline integration

3. **Mock API Responses**:
   - Pre-recorded Claude API responses
   - Error responses (429, 503, 401, 400)
   - Truncated responses (max_tokens exceeded)

---

## 3. HLD-Specified Tests

### 3.1 Unit Tests: Phase 1 Preprocessing

**Source**: LLMAnalysisCHILD.md §7.3, §8.1

#### Test 1.1: Bimodal Feature Detection
**File**: `tests/test_phase1_preprocessing.py::test_bimodal_detection`
**Function**: `detect_bimodal_pattern()`
**TI Reference**: §4.1

**Input Data**:
```json
{
  "feature": "word_count",
  "distribution": {
    "top_performers": {
      "high_percentage": 0.40,
      "low_percentage": 0.35
    }
  },
  "top_performer_avg": 52,
  "bottom_performer_avg": 18
}
```

**Expected Output**:
```python
{
  "is_bimodal": True,
  "high_percentage": 0.40,
  "low_percentage": 0.35,
  "interpretation": "BOTH strategies work",
  "pattern_label": "BIMODAL"
}
```

**Validation**:
```python
bimodal_info = detect_bimodal_pattern(feature_data)
assert bimodal_info['is_bimodal'] == True
assert bimodal_info['pattern_label'] == "BIMODAL"
assert "BOTH strategies work" in bimodal_info['interpretation']
```

**Boundary Cases**:
- `high=30%, low=30%` → `is_bimodal=True` (exactly at threshold)
- `high=29%, low=31%` → `is_bimodal=False` (just below threshold)
- `high=72%, low=15%` → `is_bimodal=False` (unimodal high)

---

#### Test 1.2: High-Contrast Feature Filtering
**File**: `tests/test_phase1_preprocessing.py::test_high_contrast_filtering`
**Function**: `identify_high_contrast_features()`
**TI Reference**: §4.2

**Input Data**: 21 features, 8 have ≥0.20 contrast between clusters

**Expected Output**:
- Python filters to 8 high-contrast features
- All 8 have `max_contrast >= 0.20`

**Validation**:
```python
high_contrast = identify_high_contrast_features(kmeans_data, threshold=0.20)
assert len(high_contrast['clusters'][0]['high_contrast_features']) == 8

for feature in high_contrast['clusters'][0]['high_contrast_features']:
    assert feature['max_contrast'] >= 0.20
```

**Boundary Cases**:
- Feature with `max_contrast=0.20` → Included (≥ threshold)
- Feature with `max_contrast=0.199` → Excluded (< threshold)
- Empty cluster → Returns empty list, no error

---

#### Test 1.3: RF Alignment Scoring
**File**: `tests/test_phase1_preprocessing.py::test_rf_alignment`
**Function**: `compute_rf_alignment()`
**TI Reference**: §4.3

**Input Data**: Cluster with 2/5 top RF features aligned (eye_contact_rate, energy_level)

**Expected Output**:
```python
{
  "alignment_score": "2/5",
  "alignment_count": 2,
  "matched_features": [
    {"feature": "eye_contact_rate", "status": "matches"},
    {"feature": "energy_level", "status": "close"}
  ]
}
```

**Validation**:
```python
alignment = compute_rf_alignment(cluster_features, rf_features, tolerance=0.15)
assert alignment['alignment_score'] == "2/5"
assert alignment['alignment_count'] == 2
assert len(alignment['matched_features']) == 2
```

**Boundary Cases**:
- 0/5 alignment (creative novelty cluster)
- 5/5 alignment (perfect RF match)
- Tolerance boundary testing (14.9% vs 15.0% vs 15.1%)

---

#### Test 1.4: Feature Enrichment
**File**: `tests/test_phase1_preprocessing.py::test_feature_enrichment`
**Function**: `enrich_high_contrast_features()`
**TI Reference**: §4.4

**Input Data**: High-contrast features + RF metadata

**Expected Output**: Enriched features with formatted strings for LLM prompt

**Validation**:
```python
enriched = enrich_high_contrast_features(high_contrast_features, rf_features, centroid)
assert len(enriched) == len(high_contrast_features)

for feature in enriched:
    assert 'formatted_string' in feature
    assert 'rf_importance' in feature or feature['rf_importance'] is None
```

**Edge Cases**:
- Feature in high-contrast but not in RF top 10 → `rf_importance=None`
- Bimodal feature → Formatted string includes Strategy A/B
- Missing centroid value → Graceful degradation

---

### 3.2 Unit Tests: Phase 2 Preprocessing

**Source**: LLMAnalysisCHILD.md §7.3, §8.2

#### Test 2.1: Scenario A - 3+ Paths Above 10% Threshold
**File**: `tests/test_phase2_preprocessing.py::test_scenario_a`
**Function**: `prepare_path_data_for_llm()`
**TI Reference**: §4.5

**Input Data**:
```python
cluster_paths = [
    {"path": "C2→C1→C3→C2→C1→C3", "count": 22, "percentage": 22.0},
    {"path": "C1→C2→C1→C3→C2→C1", "count": 18, "percentage": 18.0},
    {"path": "C3→C2→C3→C1→C2→C3", "count": 15, "percentage": 15.0},
    {"path": "C2→C3→C1→C2→C3→C1", "count": 12, "percentage": 12.0},
    {"path": "C1→C3→C2→C1→C3→C2", "count": 11, "percentage": 11.0}
    # ... 30 paths below 10%
]
total_videos = 100
```

**Expected Output**:
```python
{
  "scenario": "A",
  "paths_above_threshold": 5,
  "paths_for_prompt": [
    {"path": "...", "percentage": 22.0, "confidence": "very_high", "threshold_label": "✅ ABOVE THRESHOLD"},
    {"path": "...", "percentage": 18.0, "confidence": "high", "threshold_label": "✅ ABOVE THRESHOLD"},
    {"path": "...", "percentage": 15.0, "confidence": "high", "threshold_label": "✅ ABOVE THRESHOLD"}
    # Only top 3 included
  ]
}
```

**Validation**:
```python
path_data = prepare_path_data_for_llm(cluster_paths, total_videos, threshold=0.10)
assert path_data['scenario'] == 'A'
assert path_data['paths_above_threshold'] == 5
assert len(path_data['paths_for_prompt']) == 3  # Top 3 only

# Check confidence levels
assert path_data['paths_for_prompt'][0]['confidence'] == 'very_high'  # 22%
assert path_data['paths_for_prompt'][1]['confidence'] == 'high'       # 18%
assert path_data['paths_for_prompt'][2]['confidence'] == 'high'       # 15%
```

---

#### Test 2.2: Scenario B - 2 Paths Above Threshold
**File**: `tests/test_phase2_preprocessing.py::test_scenario_b`

**Input Data**: Force exactly 2 paths ≥10%
```python
cluster_paths = [
    {"path": "C2→C1→C3→C2→C1→C3", "count": 18, "percentage": 18.0},
    {"path": "C1→C2→C1→C3→C2→C1", "count": 12, "percentage": 12.0},
    {"path": "C3→C2→C3→C1→C2→C3", "count": 8, "percentage": 8.0},
    # ... rest below 10%
]
```

**Expected Output**:
```python
{
  "scenario": "B",
  "paths_above_threshold": 2,
  "feature_based_reports_needed": 1
}
```

**Validation**:
- 2 path-based reports generated
- 1 feature-based report generated (type="feature_based", frequency=null)

---

#### Test 2.3: Scenario C - 1 Path Above Threshold
**File**: `tests/test_phase2_preprocessing.py::test_scenario_c`

**Input Data**: Force exactly 1 path ≥10%

**Expected Output**:
```python
{
  "scenario": "C",
  "paths_above_threshold": 1,
  "feature_based_reports_needed": 2
}
```

---

#### Test 2.4: Scenario D - 0 Paths Above Threshold (High Fragmentation)
**File**: `tests/test_phase2_preprocessing.py::test_scenario_d`

**Input Data**: 40+ unique paths, all below 10%

**Expected Output**:
```python
{
  "scenario": "D",
  "paths_above_threshold": 0,
  "feature_based_reports_needed": 3,
  "fragmentation_level": "high"
}
```

**Validation**:
- All 3 reports are feature-based
- `supplementary_insights` becomes primary guidance
- No cluster_path arrays in reports

---

#### Test 2.5: Confidence Level Classification
**File**: `tests/test_phase2_preprocessing.py::test_confidence_classification`
**Function**: `classify_confidence_level()`
**TI Reference**: §4.6

**Input/Output Table**:
| Percentage | Expected Confidence |
|-----------|-------------------|
| 25.0% | very_high |
| 20.0% | very_high (boundary) |
| 19.9% | high |
| 17.5% | high |
| 15.0% | high (boundary) |
| 14.9% | moderate |
| 12.0% | moderate |
| 10.0% | moderate |

**Validation**:
```python
assert classify_confidence_level(25.0) == 'very_high'
assert classify_confidence_level(20.0) == 'very_high'  # Boundary
assert classify_confidence_level(19.9) == 'high'       # Just below
assert classify_confidence_level(15.0) == 'high'       # Boundary
assert classify_confidence_level(14.9) == 'moderate'   # Just below
```

---

#### Test 2.6: Universal Principles Generation
**File**: `tests/test_phase2_preprocessing.py::test_universal_principles`
**Function**: `generate_universal_principles()`
**TI Reference**: §4.7

**Input Data**: RF video-level data with 15 features

**Expected Output**: List of 5-7 universal principle strings

**Validation**:
```python
principles = generate_universal_principles(rf_video_data, top_n=7)
assert 5 <= len(principles) <= 7
assert all(isinstance(p, str) for p in principles)
assert all(len(p) > 0 for p in principles)
```

**Edge Cases**:
- RF data with <7 features → Return all available
- RF data with missing importance scores → Skip or use default
- RF data with duplicate features → Deduplicate

---

#### Test 2.7: Cross-Window Patterns
**File**: `tests/test_phase2_preprocessing.py::test_cross_window_patterns`
**Function**: `generate_cross_window_patterns()`
**TI Reference**: §4.8

**Input Data**: Window analyses from Phase 1 (6 windows)

**Expected Output**: List of temporal progression patterns

**Validation**:
```python
patterns = generate_cross_window_patterns(window_analyses, rf_video_data)
assert isinstance(patterns, list)
assert len(patterns) >= 1  # At least one pattern found
```

**Edge Cases**:
- 2 windows only (bucket 3-9s) → Should still work
- 1 window only (bucket 0-3s) → Return empty list or skip
- Missing RF data in some windows → Graceful degradation

---

#### Test 2.8: Feature-Based Report Generation
**File**: `tests/test_phase2_preprocessing.py::test_feature_based_reports`
**Function**: `generate_feature_based_reports()`
**TI Reference**: §4.9

**Input Data**: RF features + K-Means data, request 3 reports

**Expected Output**: 3 feature-based reports with distinct categories

**Validation**:
```python
reports = generate_feature_based_reports(rf_features, kmeans_data, num_reports=3)
assert len(reports) == 3

# Check schema compliance
for report in reports:
    assert 'title' in report
    assert 'description' in report
    assert 'key_features' in report
    assert 'type' in report and report['type'] == 'feature_based'
    assert 'frequency' in report and report['frequency'] is None

# Check category diversity
categories = [r['category'] for r in reports]
assert len(set(categories)) >= 2  # At least 2 different categories
```

**Edge Cases**:
- Request 3 reports but only 2 feature categories available
- Request 5 reports (more than features available)
- No RF features available (empty input)

---

### 3.3 Prompt Builder Tests

**Source**: LLMAnalysisCHILD.md §7.4

#### Test 3.1: Phase 1 Prompt Builder
**File**: `tests/test_prompts.py::test_phase1_prompt_builder`
**Function**: `build_phase1_prompt()`
**TI Reference**: §4.13

**Test Cases**:

**3.1.1: Complete Prompt Generation**
```python
prompt = build_phase1_prompt("hook", kmeans_data, rf_data, "18-33s", "nutrition")

# Check required sections present
assert "## Your Task" in prompt
assert "## Data Provided" in prompt
assert "## Output Requirements" in prompt
assert "Random Forest Feature Importance" in prompt
assert "K-Means Clusters" in prompt
```

**3.1.2: Bimodal Feature Presentation**
```python
# Input: RF feature with bimodal pattern (40% high, 35% low)
prompt = build_phase1_prompt(...)

assert "Strategy A:" in prompt
assert "Strategy B:" in prompt
assert "BIMODAL" in prompt or "BOTH strategies work" in prompt
```

**3.1.3: RF Alignment Display**
```python
# Input: Cluster with 2/5 RF alignment
prompt = build_phase1_prompt(...)

assert "RF alignment: 2/5" in prompt or "2 of the top 5" in prompt
assert "✅" in prompt  # Check marks for aligned features
```

**3.1.4: Prompt Length Validation**
```python
prompt = build_phase1_prompt(...)
token_count = len(prompt.split())  # Rough estimate
assert 2000 <= token_count <= 3000  # Within Claude context limits
```

---

#### Test 3.2: Phase 2 Prompt Builder
**File**: `tests/test_prompts.py::test_phase2_prompt_builder`
**Function**: `build_phase2_prompt()`
**TI Reference**: §4.14

**Test Cases**:

**3.2.1: Scenario A Prompt**
```python
prompt = build_phase2_prompt(window_analyses, cluster_paths, rf_video_data,
                             "18-33s", "nutrition", scenario="A")

assert "Generate exactly 3 path-based reports" in prompt
assert "✅ ABOVE THRESHOLD" in prompt
assert "feature-based" not in prompt  # No fallback needed
```

**3.2.2: Scenario B Prompt**
```python
prompt = build_phase2_prompt(..., scenario="B")

assert "2 path-based" in prompt
assert "1 feature-based" in prompt
assert "FALLBACK REPORT" in prompt or "feature_based_reports" in prompt
```

**3.2.3: Scenario D Prompt (High Fragmentation)**
```python
prompt = build_phase2_prompt(..., scenario="D")

assert "3 feature-based reports" in prompt
assert "high fragmentation" in prompt.lower()
assert "supplementary_insights" in prompt
```

**3.2.4: Feature-Based Report Embedding**
```python
# Scenario B/C/D - Check Python-generated reports embedded correctly
prompt = build_phase2_prompt(..., scenario="B")

# Should contain JSON-formatted feature-based report
assert '{"title":' in prompt or '"type": "feature_based"' in prompt
```

**3.2.5: Prompt Length Validation**
```python
prompt = build_phase2_prompt(...)
token_count = len(prompt.split())
assert 4000 <= token_count <= 6000  # Scenario D has longest prompt
```

---

### 3.4 API Integration Tests

**Source**: LLMAnalysisCHILD.md §7.5

#### Test 4.1: Phase 1 Window Analysis with Retry
**File**: `tests/test_api_integration.py::test_phase1_window_analysis`
**Function**: `analyze_window_with_retry()`
**Mode**: Mock API (fast) + Live API (slow)

**Test Cases**:

**4.1.1: Successful Analysis (Mock API)**
```python
# Mock successful Claude API response
with mock_anthropic_api(response=valid_window_analysis_json):
    result = analyze_window_with_retry(bucket_path, "hook", "18-33s", "nutrition")

    assert result is not None
    assert 'clusters' in result
    assert len(result['clusters']) >= 2
```

**4.1.2: Retry on 503 Error (Mock API)**
```python
# Mock 503 error on first call, success on second
with mock_anthropic_api(responses=[503_error, success_response]):
    result = analyze_window_with_retry(...)

    assert result is not None
    # Check logs: should show "Retry attempt 1/3"
```

**4.1.3: Retry on 429 Rate Limit (Mock API)**
```python
# Mock 429 error, then success
with mock_anthropic_api(responses=[429_error, success_response]):
    result = analyze_window_with_retry(...)

    assert result is not None
    # Verify exponential backoff timing (2s delay)
```

**4.1.4: Non-Retryable Error (401 Auth Failure)**
```python
# Mock 401 error - should fail immediately without retry
with mock_anthropic_api(response=401_error):
    with pytest.raises(Exception) as exc_info:
        analyze_window_with_retry(...)

    assert "Authentication" in str(exc_info.value)
    # Check logs: NO retry attempts
```

**4.1.5: Timeout Handling**
```python
# Mock timeout error
with mock_anthropic_api(response=timeout_error):
    with pytest.raises(Exception):
        analyze_window_with_retry(...)

    # Check logs: should show 3 retry attempts before final failure
```

**4.1.6: Live API Test (Small Sample)**
```python
@pytest.mark.live_api
@pytest.mark.slow
def test_phase1_live_api():
    # Use 10-video test sample
    result = analyze_window_with_retry(test_bucket_path, "hook", "18-33s", "nutrition")

    assert result is not None
    assert len(result['clusters']) >= 2

    # Validate output schema
    validate_phase1_schema(result)
```

---

#### Test 4.2: Phase 2 Synthesis
**File**: `tests/test_api_integration.py::test_phase2_synthesis`
**Function**: `run_phase2_synthesis()`
**Mode**: Mock API + Live API

**Test Cases**:

**4.2.1: Scenario A Synthesis (Mock)**
```python
with mock_anthropic_api(response=scenario_a_synthesis):
    result = run_phase2_synthesis(bucket_path, window_analyses, "18-33s", "nutrition")

    assert 'creative_reports' in result
    assert len(result['creative_reports']) == 3
    assert all(r['type'] == 'path_based' for r in result['creative_reports'])
```

**4.2.2: Scenario D Synthesis (Mock)**
```python
with mock_anthropic_api(response=scenario_d_synthesis):
    result = run_phase2_synthesis(...)

    assert len(result['creative_reports']) == 3
    assert all(r['type'] == 'feature_based' for r in result['creative_reports'])
    assert 'supplementary_insights' in result
```

**4.2.3: Confidence Level Validation**
```python
result = run_phase2_synthesis(...)

# Check confidence levels match frequency thresholds
for report in result['creative_reports']:
    if report['frequency'] >= 20.0:
        assert report['confidence_level'] == 'very_high'
    elif report['frequency'] >= 15.0:
        assert report['confidence_level'] == 'high'
    else:
        assert report['confidence_level'] == 'moderate'
```

---

### 3.5 Integration Tests (End-to-End)

**Source**: LLMAnalysisCHILD.md §7.6

#### Test 5.1: Full Pipeline Test
**File**: `tests/test_integration.py::test_full_pipeline`
**Mode**: Live API (slow, ~$0.73 for 18-33s bucket)
**Test Data**: 100-video sample from pilot hashtag (#nutrition)

**Test Steps**:
1. Pre-flight validation passes
2. Phase 1: All 6 windows complete successfully
3. Phase 2: Generates exactly 3 reports
4. Output files: 6 window JSONs + 1 synthesis JSON + 1 complete JSON = 8 files

**Validation Checkpoints**:
```python
@pytest.mark.integration
@pytest.mark.live_api
@pytest.mark.slow
def test_full_pipeline_18_33s_bucket():
    # Run complete Stage 7
    exit_code = main(bucket_path=test_bucket_path, bucket="18-33s", hashtag="nutrition")

    assert exit_code == 0

    # Check output files created
    output_dir = os.path.join(test_bucket_path, "ml_analysis/llm")
    assert os.path.exists(os.path.join(output_dir, "hook_analysis.json"))
    assert os.path.exists(os.path.join(output_dir, "middle_1_analysis.json"))
    # ... check all 6 windows
    assert os.path.exists(os.path.join(output_dir, "winning_formulas.json"))
    assert os.path.exists(os.path.join(output_dir, "complete_analysis_18-33s.json"))

    # Validate Phase 1 outputs
    for window in ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"]:
        with open(os.path.join(output_dir, f"{window}_analysis.json")) as f:
            analysis = json.load(f)

        # Check defining_features has exactly 3 items
        for cluster in analysis['clusters']:
            assert len(cluster['defining_features']) == 3

            # Check RF validation present
            assert 'rf_validation' in cluster
            assert 'insight' in cluster['rf_validation']
            assert 'alignment' in cluster['rf_validation']['insight'].lower()

    # Validate Phase 2 output
    with open(os.path.join(output_dir, "winning_formulas.json")) as f:
        synthesis = json.load(f)

    assert len(synthesis['creative_reports']) == 3

    for report in synthesis['creative_reports']:
        assert 'confidence_level' in report
        assert report['confidence_level'] in ['very_high', 'high', 'moderate']

    assert 'supplementary_insights' in synthesis
    assert 'universal_principles' in synthesis['supplementary_insights']
    assert 'cross_window_patterns' in synthesis['supplementary_insights']
```

---

#### Test 5.2: Scenario Testing (All 4 Scenarios)
**File**: `tests/test_integration.py::test_all_scenarios`

**5.2.1: Scenario A Test (#nutrition, 100 videos)**
```python
@pytest.mark.integration
def test_scenario_a_nutrition():
    # Expected: 5+ paths ≥10%
    result = main(bucket_path=nutrition_bucket_path, bucket="18-33s", hashtag="nutrition")

    with open(f"{nutrition_bucket_path}/ml_analysis/llm/winning_formulas.json") as f:
        synthesis = json.load(f)

    # Validate: 3 path-based reports, all with cluster_path arrays
    assert len(synthesis['creative_reports']) == 3

    for report in synthesis['creative_reports']:
        assert report['type'] == 'path_based'
        assert 'cluster_path' in report
        assert len(report['cluster_path']) >= 1
```

**5.2.2: Scenario B Test (Synthetic - Force 2 Paths)**
```python
@pytest.mark.integration
def test_scenario_b_synthetic():
    # Manipulate cluster paths to force exactly 2 paths ≥10%
    result = main(bucket_path=synthetic_b_bucket_path, bucket="18-33s")

    with open(f"{synthetic_b_bucket_path}/ml_analysis/llm/winning_formulas.json") as f:
        synthesis = json.load(f)

    # Validate: 2 path-based + 1 feature-based
    path_based_count = sum(1 for r in synthesis['creative_reports'] if r['type'] == 'path_based')
    feature_based_count = sum(1 for r in synthesis['creative_reports'] if r['type'] == 'feature_based')

    assert path_based_count == 2
    assert feature_based_count == 1

    # Check: Report #3 has type="feature_based", frequency=null
    feature_report = [r for r in synthesis['creative_reports'] if r['type'] == 'feature_based'][0]
    assert feature_report['frequency'] is None
```

**5.2.3: Scenario C Test (Fragmented Hashtag)**
```python
@pytest.mark.integration
def test_scenario_c_fragmented():
    # Expected: 1-2 paths ≥10%
    result = main(bucket_path=fragmented_bucket_path, bucket="18-33s")

    with open(f"{fragmented_bucket_path}/ml_analysis/llm/winning_formulas.json") as f:
        synthesis = json.load(f)

    path_based_count = sum(1 for r in synthesis['creative_reports'] if r['type'] == 'path_based')
    feature_based_count = sum(1 for r in synthesis['creative_reports'] if r['type'] == 'feature_based')

    # Validate: 1 path + 2 feature OR 2 path + 1 feature
    assert (path_based_count == 1 and feature_based_count == 2) or \
           (path_based_count == 2 and feature_based_count == 1)
```

**5.2.4: Scenario D Test (Highly Fragmented - 40+ Paths)**
```python
@pytest.mark.integration
def test_scenario_d_highly_fragmented():
    # Expected: 0 paths ≥10%
    result = main(bucket_path=highly_fragmented_bucket_path, bucket="18-33s")

    with open(f"{highly_fragmented_bucket_path}/ml_analysis/llm/winning_formulas.json") as f:
        synthesis = json.load(f)

    # Validate: 3 feature-based reports
    assert len(synthesis['creative_reports']) == 3
    assert all(r['type'] == 'feature_based' for r in synthesis['creative_reports'])

    # Check: supplementary_insights becomes primary guidance
    assert 'supplementary_insights' in synthesis
    assert len(synthesis['supplementary_insights']['universal_principles']) >= 5
```

---

### 3.6 Edge Case Tests

**Source**: LLMAnalysisCHILD.md §8.1

#### Test 6.1: Bucket 0-3s (Single Window, No Phase 2)
**File**: `tests/test_edge_cases.py::test_bucket_0_3s`

```python
def test_bucket_0_3s_single_window():
    result = main(bucket_path=bucket_0_3s_path, bucket="0-3s", hashtag="nutrition")

    output_dir = os.path.join(bucket_0_3s_path, "ml_analysis/llm")

    # Only hook_analysis.json should be created
    assert os.path.exists(os.path.join(output_dir, "hook_analysis.json"))

    # No Phase 2 files (winning_formulas.json should NOT exist)
    # Note: TI specifies Phase 2 is skipped for bucket 0-3s
    # Verify implementation behavior
```

---

#### Test 6.2: Bucket 3-9s (2 Windows, Minimal Paths)
**File**: `tests/test_edge_cases.py::test_bucket_3_9s`

```python
def test_bucket_3_9s_minimal_paths():
    result = main(bucket_path=bucket_3_9s_path, bucket="3-9s", hashtag="nutrition")

    output_dir = os.path.join(bucket_3_9s_path, "ml_analysis/llm")

    # Check 2 window files
    assert os.path.exists(os.path.join(output_dir, "hook_analysis.json"))
    assert os.path.exists(os.path.join(output_dir, "closing_analysis.json"))

    # Phase 2 runs but with limited path data (likely Scenario C or D)
    assert os.path.exists(os.path.join(output_dir, "winning_formulas.json"))

    # Verify cross_window_patterns handles <3 windows gracefully
    with open(os.path.join(output_dir, "winning_formulas.json")) as f:
        synthesis = json.load(f)

    # Should have supplementary_insights even with 2 windows
    assert 'supplementary_insights' in synthesis
```

---

#### Test 6.3: Bucket 9-13s (middle_aggregate Window)
**File**: `tests/test_edge_cases.py::test_bucket_9_13s`

```python
def test_bucket_9_13s_middle_aggregate():
    result = main(bucket_path=bucket_9_13s_path, bucket="9-13s", hashtag="nutrition")

    output_dir = os.path.join(bucket_9_13s_path, "ml_analysis/llm")

    # Check for middle_aggregate window (not middle_1, middle_2)
    assert os.path.exists(os.path.join(output_dir, "middle_aggregate_analysis.json"))
```

---

#### Test 6.4: API Failure Scenarios
**File**: `tests/test_edge_cases.py::test_api_failures`

**6.4.1: 429 Rate Limit**
```python
def test_rate_limit_handling():
    with mock_anthropic_api(responses=[429_error] * 3):  # Fail all retries
        with pytest.raises(Exception) as exc_info:
            analyze_window_with_retry(...)

        assert "Rate limit" in str(exc_info.value)
        # Verify 3 retry attempts in logs
```

**6.4.2: 503 Service Unavailable**
```python
def test_service_unavailable_retry():
    with mock_anthropic_api(responses=[503_error, 503_error, success_response]):
        result = analyze_window_with_retry(...)

        assert result is not None
        # Verify 2 retries before success
```

**6.4.3: Timeout**
```python
def test_timeout_handling():
    with mock_anthropic_api(timeout=True):
        with pytest.raises(Exception) as exc_info:
            analyze_window_with_retry(..., timeout=90)

        assert "timeout" in str(exc_info.value).lower()
```

---

#### Test 6.5: JSON Truncation (max_tokens Exceeded)
**File**: `tests/test_edge_cases.py::test_json_truncation`

```python
def test_truncated_json_response():
    # Mock API returns truncated JSON (missing closing braces)
    truncated_json = '{"clusters": [{"id": "C1", "defining_features": ["feature1"'

    with mock_anthropic_api(response=truncated_json):
        with pytest.raises(json.JSONDecodeError):
            analyze_window_with_retry(...)

        # Verify error logged with helpful message
```

---

## 4. Gap Analysis & Recommendations

### 4.1 Critical Gaps (MUST Fix Before Production)

#### Gap 1: Checkpoint/Resume Logic - NOT TESTED ⚠️
**Risk Level**: 🔴 **CRITICAL**
**Impact**: Cost-critical feature - bugs could cause expensive re-runs ($4.18/client at risk)

**Missing Tests**:
1. Checkpoint creation after first window completes
2. Resume from checkpoint - skip completed windows
3. Checkpoint corruption (invalid JSON)
4. Checkpoint with partial completion (3/6 windows)
5. Checkpoint cleanup after full completion

**Recommended Tests**:
```python
# Test CP.1: Checkpoint Creation
def test_checkpoint_creation():
    """Verify .phase1_status.json created after first window"""
    run_phase1_parallel(bucket_path, "18-33s", "nutrition", ["hook"])

    status_file = os.path.join(bucket_path, "ml_analysis/llm/.phase1_status.json")
    assert os.path.exists(status_file)

    with open(status_file) as f:
        status = json.load(f)

    assert 'completed_windows' in status
    assert 'hook' in status['completed_windows']

# Test CP.2: Resume from Checkpoint
def test_resume_from_checkpoint():
    """Verify Stage 7 resumes from checkpoint, skips completed windows"""
    # Step 1: Run 3/6 windows, then simulate crash
    windows_phase1 = ["hook", "middle_1", "middle_2"]
    run_phase1_parallel(bucket_path, "18-33s", "nutrition", windows_phase1)

    # Checkpoint should have 3 completed windows
    status_file = os.path.join(bucket_path, "ml_analysis/llm/.phase1_status.json")
    with open(status_file) as f:
        status = json.load(f)
    assert len(status['completed_windows']) == 3

    # Step 2: Resume with all 6 windows (should skip first 3)
    with mock_api_call_counter() as counter:
        windows_all = ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"]
        run_phase1_parallel(bucket_path, "18-33s", "nutrition", windows_all)

        # Should make only 3 API calls (not 6)
        assert counter.call_count == 3

# Test CP.3: Checkpoint Corruption
def test_checkpoint_corruption_handling():
    """Verify graceful handling of corrupted checkpoint file"""
    status_file = os.path.join(bucket_path, "ml_analysis/llm/.phase1_status.json")

    # Write invalid JSON
    with open(status_file, 'w') as f:
        f.write("{invalid json")

    # Should re-run all windows (ignore corrupted checkpoint)
    with mock_api_call_counter() as counter:
        run_phase1_parallel(bucket_path, "18-33s", "nutrition", ["hook", "middle_1"])

        assert counter.call_count == 2  # Both windows re-run

# Test CP.4: Partial Completion
def test_partial_completion_3_of_6_windows():
    """Test checkpoint with 3/6 windows completed"""
    # Manually create checkpoint
    status = {
        "completed_windows": ["hook", "middle_1", "middle_2"],
        "timestamp": "2025-10-21T10:30:00Z"
    }
    status_file = os.path.join(bucket_path, "ml_analysis/llm/.phase1_status.json")
    with open(status_file, 'w') as f:
        json.dump(status, f)

    # Resume should only run remaining 3 windows
    with mock_api_call_counter() as counter:
        run_phase1_parallel(bucket_path, "18-33s", "nutrition",
                          ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"])

        assert counter.call_count == 3  # Only middle_3, middle_4, closing
```

**Estimated Time**: 45 minutes

---

#### Gap 2: Parallel Execution - NOT TESTED ⚠️
**Risk Level**: 🔴 **CRITICAL**
**Impact**: Thread safety issues, race conditions, or sequential execution masquerading as parallel

**Missing Tests**:
1. Verify windows actually run in parallel (timing test)
2. Thread safety of status file writes
3. Behavior when 1 window fails but others succeed
4. Verify max_workers configuration

**Recommended Tests**:
```python
# Test PE.1: Parallel Execution Timing
def test_parallel_execution_timing():
    """Verify windows run in parallel, not sequentially"""
    import time

    # Mock API calls with 5-second delay each
    def mock_slow_api_call(*args, **kwargs):
        time.sleep(5)
        return valid_analysis_response

    with mock.patch('anthropic.Client.messages.create', side_effect=mock_slow_api_call):
        start_time = time.time()

        # Run 6 windows
        run_phase1_parallel(bucket_path, "18-33s", "nutrition",
                          ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"])

        elapsed = time.time() - start_time

        # If sequential: 6 windows * 5s = 30s
        # If parallel: max(5s, 5s, ...) = ~5-7s (with overhead)
        assert elapsed < 10, f"Execution took {elapsed}s - likely sequential, not parallel"

# Test PE.2: Thread Safety of Status File
def test_thread_safety_status_file():
    """Verify status file writes are thread-safe (no race conditions)"""
    # Run with many windows to increase likelihood of race conditions
    windows = ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"]

    run_phase1_parallel(bucket_path, "18-33s", "nutrition", windows)

    # Checkpoint should have ALL 6 windows (no lost writes)
    status_file = os.path.join(bucket_path, "ml_analysis/llm/.phase1_status.json")
    with open(status_file) as f:
        status = json.load(f)

    assert len(status['completed_windows']) == 6
    assert set(status['completed_windows']) == set(windows)

# Test PE.3: Partial Failure Handling
def test_partial_failure_in_parallel_execution():
    """Verify behavior when 1 window fails but others succeed"""
    def mock_api_call_with_failure(window_type, *args, **kwargs):
        if window_type == "middle_2":
            raise Exception("API failure for middle_2")
        return valid_analysis_response

    with mock.patch('analyze_window_with_retry', side_effect=mock_api_call_with_failure):
        with pytest.raises(Exception):
            run_phase1_parallel(bucket_path, "18-33s", "nutrition",
                              ["hook", "middle_1", "middle_2"])

        # Checkpoint should have 2 completed windows (hook, middle_1)
        status_file = os.path.join(bucket_path, "ml_analysis/llm/.phase1_status.json")
        with open(status_file) as f:
            status = json.load(f)

        assert "hook" in status['completed_windows']
        assert "middle_1" in status['completed_windows']
        assert "middle_2" not in status['completed_windows']
```

**Estimated Time**: 30 minutes

---

#### Gap 3: Retry Logic Timing - SUPERFICIAL ⚠️
**Risk Level**: 🔴 **HIGH**
**Impact**: Incorrect backoff could hammer API or fail to retry properly

**Missing Tests**:
1. Exponential backoff timing (0s, 2s, 4s)
2. Retryable vs non-retryable error distinction
3. Max attempts configuration

**Recommended Tests**:
```python
# Test RT.1: Exponential Backoff Timing
def test_exponential_backoff_timing():
    """Verify backoff delays are 0s, 2s, 4s"""
    import time

    call_times = []

    def mock_failing_api(*args, **kwargs):
        call_times.append(time.time())
        raise Exception("503 Service Unavailable")

    with mock.patch('anthropic.Client.messages.create', side_effect=mock_failing_api):
        with pytest.raises(Exception):
            analyze_window_with_retry(bucket_path, "hook", "18-33s", "nutrition", max_attempts=3)

    # Verify 3 attempts made
    assert len(call_times) == 3

    # Verify delays: 0s before attempt 1, ~2s before attempt 2, ~4s before attempt 3
    delay_1_to_2 = call_times[1] - call_times[0]
    delay_2_to_3 = call_times[2] - call_times[1]

    assert 1.8 <= delay_1_to_2 <= 2.5, f"Delay 1→2: {delay_1_to_2}s (expected ~2s)"
    assert 3.8 <= delay_2_to_3 <= 4.5, f"Delay 2→3: {delay_2_to_3}s (expected ~4s)"

# Test RT.2: Retryable vs Non-Retryable Errors
def test_retryable_vs_non_retryable_errors():
    """Verify 401/400 fail immediately, 429/503 retry"""
    call_count = 0

    def mock_401_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise Exception("401 Unauthorized")

    # 401 should fail immediately (no retry)
    with mock.patch('anthropic.Client.messages.create', side_effect=mock_401_error):
        with pytest.raises(Exception):
            analyze_window_with_retry(bucket_path, "hook", "18-33s", "nutrition", max_attempts=3)

    assert call_count == 1, "401 error should NOT retry"

    # Reset counter
    call_count = 0

    def mock_429_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise Exception("429 Rate Limit Exceeded")

    # 429 should retry 3 times
    with mock.patch('anthropic.Client.messages.create', side_effect=mock_429_error):
        with pytest.raises(Exception):
            analyze_window_with_retry(bucket_path, "hook", "18-33s", "nutrition", max_attempts=3)

    assert call_count == 3, "429 error should retry 3 times"
```

**Estimated Time**: 30 minutes

---

#### Gap 4: Feature-Based Report Generation - NOT TESTED ⚠️
**Risk Level**: 🔴 **HIGH**
**Impact**: If reports don't match schema, Phase 2 LLM may fail to parse or hallucinate

**Missing Tests**:
1. JSON schema compliance
2. Category grouping (visual, audio, behavioral)
3. Edge cases (insufficient features, missing data)

**Recommended Tests**:
```python
# Test FB.1: JSON Schema Compliance
def test_feature_based_report_schema_compliance():
    """Verify generated reports match exact Phase 2 schema"""
    reports = generate_feature_based_reports(rf_features, kmeans_data, num_reports=3)

    for report in reports:
        # Required fields
        assert 'title' in report
        assert 'description' in report
        assert 'key_features' in report
        assert 'type' in report
        assert 'frequency' in report
        assert 'confidence_level' in report

        # Field types
        assert isinstance(report['title'], str)
        assert isinstance(report['description'], str)
        assert isinstance(report['key_features'], list)
        assert report['type'] == 'feature_based'
        assert report['frequency'] is None
        assert report['confidence_level'] in ['very_high', 'high', 'moderate']

        # key_features structure
        assert len(report['key_features']) >= 1
        for feature in report['key_features']:
            assert 'feature' in feature
            assert 'value' in feature or 'description' in feature

# Test FB.2: Category Diversity
def test_feature_based_report_category_diversity():
    """Verify reports use different feature categories (visual, audio, behavioral)"""
    reports = generate_feature_based_reports(rf_features, kmeans_data, num_reports=3)

    categories = [r.get('category', 'unknown') for r in reports]

    # At least 2 different categories
    assert len(set(categories)) >= 2, f"All reports used same category: {categories}"

# Test FB.3: Insufficient Features
def test_feature_based_report_insufficient_features():
    """Verify graceful handling when <3 feature categories available"""
    # RF data with only 2 features (both visual)
    minimal_rf_features = [
        {"feature": "eye_contact_rate", "importance": 0.15, "category": "visual"},
        {"feature": "hand_gesture_count", "importance": 0.12, "category": "visual"}
    ]

    reports = generate_feature_based_reports(minimal_rf_features, kmeans_data, num_reports=3)

    # Should still generate reports (even if categories overlap)
    assert len(reports) <= 3  # May generate fewer if truly insufficient
```

**Estimated Time**: 30 minutes

---

#### Gap 5: Output Validation (3-Layer) - BASIC ⚠️
**Risk Level**: 🔴 **HIGH**
**Impact**: TI Section 5 specifies 3-layer validation, current tests only check Layer 1

**Missing Tests**:
1. Hallucination detection (invented features/videos)
2. Bimodal feature formatting in output
3. RF validation insight accuracy

**Recommended Tests**:
```python
# Test OV.1: Hallucination Detection - Features
def test_no_hallucinated_features():
    """Verify LLM doesn't invent features not in input data"""
    result = analyze_window_with_retry(bucket_path, "hook", "18-33s", "nutrition")

    # Get all features from input data
    with open(f"{bucket_path}/ml_analysis/hook_kmeans_analysis.json") as f:
        kmeans_data = json.load(f)
    with open(f"{bucket_path}/ml_analysis/hook_rf_analysis.json") as f:
        rf_data = json.load(f)

    all_input_features = set()
    for feature in rf_data['feature_importance']:
        all_input_features.add(feature['feature'])
    for cluster in kmeans_data['clusters']:
        for feature in cluster.get('features', []):
            all_input_features.add(feature['feature'])

    # Check LLM output features
    for cluster in result['clusters']:
        for defining_feature in cluster['defining_features']:
            # Extract feature name (format: "feature_name: description")
            feature_name = defining_feature.split(':')[0].strip()

            assert feature_name in all_input_features, \
                f"LLM hallucinated feature '{feature_name}' not in input data"

# Test OV.2: Hallucination Detection - Videos
def test_no_hallucinated_video_ids():
    """Verify LLM doesn't invent video IDs not in cluster"""
    result = analyze_window_with_retry(bucket_path, "hook", "18-33s", "nutrition")

    # Get actual video IDs per cluster
    with open(f"{bucket_path}/ml_analysis/hook_kmeans_analysis.json") as f:
        kmeans_data = json.load(f)

    for idx, cluster in enumerate(result['clusters']):
        actual_video_ids = set(kmeans_data['clusters'][idx]['videos'])

        # Check video IDs in LLM output (if present)
        if 'example_videos' in cluster:
            for video_id in cluster['example_videos']:
                assert video_id in actual_video_ids, \
                    f"LLM hallucinated video ID '{video_id}' not in cluster {idx}"

# Test OV.3: Bimodal Feature Formatting
def test_bimodal_feature_formatting_in_output():
    """Verify bimodal features correctly formatted in LLM output"""
    # Use input with known bimodal feature (word_count: 40% high, 35% low)
    result = analyze_window_with_retry(bucket_path, "hook", "18-33s", "nutrition")

    # Find cluster with bimodal feature
    for cluster in result['clusters']:
        for defining_feature in cluster['defining_features']:
            if 'ALTERNATIVE STRATEGIES' in defining_feature or \
               'Strategy A' in defining_feature or 'Strategy B' in defining_feature:
                # Found bimodal feature - verify both strategies mentioned
                assert 'Strategy A' in defining_feature or 'BRIEF' in defining_feature
                assert 'Strategy B' in defining_feature or 'DENSE' in defining_feature
                break

# Test OV.4: RF Validation Insight Accuracy
def test_rf_validation_insight_accuracy():
    """Verify RF alignment scores in insights match computed scores"""
    result = analyze_window_with_retry(bucket_path, "hook", "18-33s", "nutrition")

    # Load RF data to compute expected alignment
    with open(f"{bucket_path}/ml_analysis/hook_rf_analysis.json") as f:
        rf_data = json.load(f)
    with open(f"{bucket_path}/ml_analysis/hook_kmeans_analysis.json") as f:
        kmeans_data = json.load(f)

    for idx, cluster in enumerate(result['clusters']):
        # Compute expected alignment
        cluster_features = [f['feature'] for f in kmeans_data['clusters'][idx]['features']]
        expected_alignment = compute_rf_alignment(cluster_features, rf_data['feature_importance'])

        # Check insight mentions alignment score
        insight = cluster['rf_validation']['insight']

        # Should mention "X/5" or "X of the top 5"
        assert expected_alignment['alignment_score'] in insight or \
               f"{expected_alignment['alignment_count']} of the top 5" in insight
```

**Estimated Time**: 45 minutes

---

### 4.2 High Priority Gaps (Recommended Before Production)

#### Gap 6: Cluster Path Extraction - MINIMAL
**Risk Level**: 🟡 **MEDIUM**
**Impact**: Path extraction errors directly affect scenario detection (A/B/C/D)

**Missing Tests**:
1. Path extraction with videos missing from some windows
2. Path frequency calculation accuracy
3. Edge cases (0 videos, 1 video)

**Estimated Time**: 30 minutes

---

#### Gap 7: Cross-Window Pattern Detection - NOT TESTED
**Risk Level**: 🟡 **MEDIUM**
**Impact**: Affects Phase 2 supplementary insights quality

**Missing Tests**:
1. Graceful degradation with 2 windows
2. Graceful degradation with 1 window
3. Temporal progression logic validation

**Estimated Time**: 20 minutes

---

#### Gap 8: Universal Principles - NOT TESTED
**Risk Level**: 🟡 **MEDIUM**
**Impact**: Affects Phase 2 supplementary insights quality

**Missing Tests**:
1. Edge case with <7 features
2. Duplicate features handling
3. Missing importance scores

**Estimated Time**: 15 minutes

---

#### Gap 9: Prompt Builder Edge Cases - MINIMAL
**Risk Level**: 🟡 **MEDIUM**
**Impact**: Malformed prompts could degrade LLM output quality

**Missing Tests**:
1. Empty preprocessing outputs (0 bimodal features, 0 high-contrast features)
2. Malformed RF data (missing fields)
3. Feature-based report embedding validation

**Estimated Time**: 30 minutes

---

### 4.3 Medium Priority Gaps (Nice to Have)

#### Gap 10: Cost Tracking - NOT TESTED
**Risk Level**: 🟢 **LOW**
**Impact**: No visibility into actual vs estimated costs

**Estimated Time**: 10 minutes

---

#### Gap 11: Logging - NOT TESTED
**Risk Level**: 🟢 **LOW**
**Impact**: Difficult debugging in production

**Estimated Time**: 15 minutes

---

### 4.4 Summary of Gaps

| Gap # | Category | Risk | HLD Coverage | Recommended Coverage | Time |
|-------|----------|------|--------------|---------------------|------|
| 1 | Checkpoint/Resume | 🔴 CRITICAL | 0% | +100% | 45 min |
| 2 | Parallel Execution | 🔴 CRITICAL | 0% | +100% | 30 min |
| 3 | Retry Logic Timing | 🔴 HIGH | 20% | +80% | 30 min |
| 4 | Feature-Based Reports | 🔴 HIGH | 0% | +100% | 30 min |
| 5 | Output Validation (3-Layer) | 🔴 HIGH | 40% | +60% | 45 min |
| 6 | Cluster Path Extraction | 🟡 MEDIUM | 30% | +70% | 30 min |
| 7 | Cross-Window Patterns | 🟡 MEDIUM | 0% | +100% | 20 min |
| 8 | Universal Principles | 🟡 MEDIUM | 0% | +100% | 15 min |
| 9 | Prompt Builder Edge Cases | 🟡 MEDIUM | 40% | +60% | 30 min |
| 10 | Cost Tracking | 🟢 LOW | 0% | +100% | 10 min |
| 11 | Logging | 🟢 LOW | 0% | +100% | 15 min |

**Total Additional Testing Time**: ~4.5 hours

---

## 5. Complete Test Suite

### 5.1 Recommended Test File Structure

```
tests/
├── test_phase1_preprocessing.py        # HLD Tests 1.1-1.4 (Unit tests for §4.1-4.4)
├── test_phase2_preprocessing.py        # HLD Tests 2.1-2.8 (Unit tests for §4.5-4.9)
├── test_prompts.py                     # HLD Tests 3.1-3.2 (Prompt builders §4.13-4.14)
│                                       # + Gap 9 (Edge cases)
├── test_api_integration.py             # HLD Tests 4.1-4.2 (API with retry)
│                                       # + Gap 3 (Retry timing)
├── test_checkpoint_resume.py           # ⭐ NEW - Gap 1 (CRITICAL)
├── test_parallel_execution.py          # ⭐ NEW - Gap 2 (CRITICAL)
├── test_output_validation.py           # ⭐ NEW - Gap 5 (HIGH)
├── test_feature_based_reports.py       # ⭐ NEW - Gap 4 (HIGH)
├── test_integration.py                 # HLD Tests 5.1-5.2 (E2E pipeline)
│                                       # + Gap 6 (Cluster paths)
├── test_edge_cases.py                  # HLD Tests 6.1-6.5 (Buckets, API failures)
│                                       # + Gap 7, 8 (Cross-window, Universal)
└── fixtures/
    ├── sample_rf_data.json
    ├── sample_kmeans_data.json
    ├── sample_cluster_paths.json
    ├── mock_api_responses/
    │   ├── hook_analysis_success.json
    │   ├── winning_formulas_scenario_a.json
    │   └── error_responses.json
    └── test_buckets/
        ├── bucket_0-3s/              # 1-window test data
        ├── bucket_3-9s/              # 2-window test data
        ├── bucket_18-33s/            # 6-window test data
        └── synthetic_scenario_b/     # Forced 2-path scenario
```

### 5.2 Complete Test Coverage Matrix

| Function | HLD Unit Test | HLD Integration Test | Gap Test | Total Coverage |
|----------|--------------|---------------------|----------|----------------|
| `detect_bimodal_pattern()` | ✅ Test 1.1 | - | - | ✅ 100% |
| `identify_high_contrast_features()` | ✅ Test 1.2 | - | - | ✅ 100% |
| `compute_rf_alignment()` | ✅ Test 1.3 | - | - | ✅ 100% |
| `enrich_high_contrast_features()` | ✅ Test 1.4 | - | - | ✅ 100% |
| `prepare_path_data_for_llm()` | ✅ Test 2.1-2.4 | - | 🟡 Gap 6 | 🟡 85% |
| `classify_confidence_level()` | ✅ Test 2.5 | - | - | ✅ 100% |
| `generate_universal_principles()` | ✅ Test 2.6 | - | 🟡 Gap 8 | 🟡 70% |
| `generate_cross_window_patterns()` | ✅ Test 2.7 | - | 🟡 Gap 7 | 🟡 60% |
| `generate_feature_based_reports()` | ✅ Test 2.8 | - | 🔴 Gap 4 | 🟡 50% |
| `build_phase1_prompt()` | ✅ Test 3.1 | - | 🟡 Gap 9 | 🟡 75% |
| `build_phase2_prompt()` | ✅ Test 3.2 | - | 🟡 Gap 9 | 🟡 75% |
| `analyze_window_with_retry()` | ✅ Test 4.1 | ✅ Test 5.1 | 🔴 Gap 3 | 🟡 80% |
| `run_phase1_parallel()` | - | ✅ Test 5.1 | 🔴 Gap 1, 2 | 🔴 40% |
| `run_phase2_synthesis()` | ✅ Test 4.2 | ✅ Test 5.1 | 🟡 Gap 6 | 🟡 80% |
| `main()` | - | ✅ Test 5.1-5.2 | - | ✅ 90% |

**Legend**:
- ✅ 100%: Fully tested
- 🟡 60-90%: Partially tested, gaps identified
- 🔴 <60%: Significant gaps, high risk

---

## 6. Test Data Requirements

### 6.1 Synthetic JSON Fixtures

**Required Fixtures** (stored in `tests/fixtures/`):

1. **sample_rf_data.json** (~2KB):
   - 15 features with importance scores
   - Includes 1 bimodal feature (40% high, 35% low)
   - Mix of visual, audio, behavioral categories
   - Covers all RF data fields

2. **sample_kmeans_data.json** (~5KB):
   - 3 clusters with 10-15 videos each
   - 21 features per cluster (8 with ≥0.20 contrast)
   - Centroid values for all features
   - Videos distributed across clusters

3. **sample_cluster_paths.json** (~1KB):
   - 35 unique paths for 100 videos
   - Top 5 paths: [22%, 18%, 15%, 12%, 11%] (Scenario A)
   - Includes path for each video

4. **sample_window_analyses.json** (~12KB):
   - Phase 1 outputs for 6 windows
   - Each window: 3 clusters, 3 defining features, RF validation

5. **mock_api_responses/** (directory):
   - `hook_analysis_success.json`: Valid Phase 1 response
   - `winning_formulas_scenario_a.json`: Valid Phase 2 Scenario A response
   - `winning_formulas_scenario_d.json`: Valid Phase 2 Scenario D response
   - `error_429_rate_limit.json`: 429 error response
   - `error_503_service_unavailable.json`: 503 error response
   - `error_401_unauthorized.json`: 401 error response
   - `truncated_json.json`: Incomplete JSON (max_tokens exceeded)

### 6.2 Real Stage 6 Outputs

**Required Test Data** (from pilot hashtag testing):

1. **bucket_18-33s_nutrition_100videos/** (~150KB):
   - Complete Stage 6 outputs (13 JSONs)
   - 100 videos from #nutrition hashtag
   - All 6 windows (hook, middle_1-4, closing)
   - Expected: Scenario A (5+ paths ≥10%)

2. **bucket_3-9s_nutrition_100videos/** (~50KB):
   - Complete Stage 6 outputs (13 JSONs)
   - 100 videos from #nutrition hashtag
   - 2 windows only (hook, closing)
   - Expected: Scenario C or D (limited paths)

3. **bucket_0-3s_nutrition_100videos/** (~25KB):
   - Complete Stage 6 outputs (13 JSONs)
   - 100 videos from #nutrition hashtag
   - 1 window only (hook)
   - No Phase 2 (validation test)

### 6.3 Synthetic Scenario-Forcing Data

**Required Synthetic Test Buckets**:

1. **synthetic_scenario_b/** (~100KB):
   - Manipulated cluster paths: exactly 2 paths ≥10%
   - Path distribution: [18%, 12%, 8%, 7%, 6%, ...]
   - Forces Scenario B (2 path + 1 feature-based)

2. **synthetic_scenario_c/** (~100KB):
   - Manipulated cluster paths: exactly 1 path ≥10%
   - Path distribution: [15%, 9%, 8%, 7%, 6%, ...]
   - Forces Scenario C (1 path + 2 feature-based)

3. **synthetic_scenario_d/** (~100KB):
   - Highly fragmented: 40+ unique paths, all <10%
   - Path distribution: [8%, 7%, 6%, 5%, 4%, ...]
   - Forces Scenario D (3 feature-based)

### 6.4 Test Data Generation Scripts

**Recommended Scripts** (to be created):

```python
# tests/generate_test_data.py

def generate_sample_rf_data() -> dict:
    """Generate synthetic RF data with bimodal feature"""
    # ...

def generate_sample_kmeans_data(n_clusters: int = 3) -> dict:
    """Generate synthetic K-Means data with controlled contrast"""
    # ...

def generate_scenario_b_cluster_paths() -> List[dict]:
    """Generate cluster paths forcing Scenario B (exactly 2 paths ≥10%)"""
    # ...

def manipulate_real_stage6_output(
    source_bucket_path: str,
    target_scenario: str
) -> None:
    """Manipulate real Stage 6 output to force specific scenario"""
    # ...
```

---

## 7. Test Execution Plan

### 7.1 Local Development Testing

**Phase 1: Unit Tests (Fast, No Cost)**
```bash
# Install test dependencies
pip install pytest pytest-mock pytest-cov

# Run all unit tests with mocked API
pytest tests/test_phase1_preprocessing.py -v
pytest tests/test_phase2_preprocessing.py -v
pytest tests/test_prompts.py -v

# Run with coverage report
pytest tests/test_*.py --cov=ml_pipeline.stage7_llm_analysis --cov-report=html

# Target: >80% line coverage
```

**Expected Runtime**: ~2-3 minutes
**Expected Cost**: $0

---

**Phase 2: API Integration Tests (Mock Mode)**
```bash
# Run API tests with mocked Anthropic client
pytest tests/test_api_integration.py -v --mock-api

# Run checkpoint/resume tests
pytest tests/test_checkpoint_resume.py -v

# Run parallel execution tests
pytest tests/test_parallel_execution.py -v
```

**Expected Runtime**: ~3-4 minutes
**Expected Cost**: $0

---

**Phase 3: Edge Case Tests (Mock Mode)**
```bash
# Run all edge case tests
pytest tests/test_edge_cases.py -v --mock-api
pytest tests/test_output_validation.py -v --mock-api
pytest tests/test_feature_based_reports.py -v
```

**Expected Runtime**: ~2-3 minutes
**Expected Cost**: $0

---

### 7.2 Live API Testing (Slow, ~$0.26/bucket)

**Phase 4: Small Sample Live API Test**
```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Run Phase 1 live test (10 videos, 1 window)
pytest tests/test_api_integration.py::test_phase1_live_api -v --live-api

# Expected cost: ~$0.09
```

**Expected Runtime**: ~30 seconds
**Expected Cost**: ~$0.09

---

**Phase 5: Integration Test (100 videos, 6 windows)**
```bash
# Run full pipeline test with 18-33s bucket
pytest tests/test_integration.py::test_full_pipeline_18_33s_bucket -v --live-api

# Expected cost: ~$0.73
```

**Expected Runtime**: ~60 seconds
**Expected Cost**: ~$0.73

---

**Phase 6: Scenario Testing (All 4 Scenarios)**
```bash
# Test all scenarios with real/synthetic data
pytest tests/test_integration.py::test_scenario_a_nutrition -v --live-api
pytest tests/test_integration.py::test_scenario_b_synthetic -v --live-api
pytest tests/test_integration.py::test_scenario_c_fragmented -v --live-api
pytest tests/test_integration.py::test_scenario_d_highly_fragmented -v --live-api

# Expected cost: ~$2.92 (4 buckets * $0.73 avg)
```

**Expected Runtime**: ~4 minutes
**Expected Cost**: ~$2.92

---

### 7.3 CI/CD Testing (GitHub Actions)

**Recommended Workflow** (.github/workflows/stage7_tests.yml):

```yaml
name: Stage 7 Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r ml_pipeline/stage7_llm_analysis/requirements.txt
          pip install pytest pytest-mock pytest-cov
      - name: Run unit tests
        run: pytest tests/test_phase*.py tests/test_prompts.py -v --cov
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests-mock:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Run integration tests (mock API)
        run: pytest tests/test_api_integration.py tests/test_edge_cases.py -v --mock-api

  integration-tests-live:
    runs-on: ubuntu-latest
    # Only run on main branch (save API costs)
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Run integration tests (live API)
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: pytest tests/test_integration.py::test_full_pipeline_18_33s_bucket -v --live-api
```

---

### 7.4 Test Execution Checklist

**Before Merging to Main**:
- [ ] All unit tests pass (mock API)
- [ ] All edge case tests pass (mock API)
- [ ] Code coverage >80%
- [ ] At least 1 live API test passes (10-video sample)

**Before Production Deployment**:
- [ ] All HLD-specified tests pass (100%)
- [ ] All HIGH priority gap tests pass (Gaps 1-5)
- [ ] At least 2 full integration tests pass (live API, 100 videos)
- [ ] All 4 scenario tests pass (A/B/C/D)
- [ ] Checkpoint/resume tested successfully
- [ ] Parallel execution timing validated (<10s for 6 windows)

---

## 8. Success Criteria

### 8.1 Unit Test Success Criteria

✅ **Preprocessing Functions**:
- All boundary cases tested (thresholds: 0.10, 0.15, 0.20, 0.30)
- Edge cases handled gracefully (empty inputs, missing fields)
- Return types match specifications
- No unhandled exceptions

✅ **Prompt Builders**:
- All required sections present in prompts
- Prompt length within Claude limits (2K-3K for Phase 1, 4K-6K for Phase 2)
- Preprocessing outputs correctly embedded
- Bimodal features show Strategy A/B
- Feature-based reports embedded as valid JSON

### 8.2 Integration Test Success Criteria

✅ **Full Pipeline (E2E)**:
- All windows complete successfully
- Exactly 8 output files created (6 windows + 1 synthesis + 1 complete)
- All defining_features arrays have exactly 3 items
- All rf_validation.insight fields include alignment scores
- All creative_reports have confidence_level field
- supplementary_insights present with both subsections

✅ **Scenario Testing**:
- Scenario A: 3 path-based reports generated
- Scenario B: 2 path + 1 feature-based
- Scenario C: 1 path + 2 feature-based OR 2 path + 1 feature-based
- Scenario D: 3 feature-based reports
- Confidence levels match frequency thresholds
- No hallucinated features or video IDs

### 8.3 Performance Success Criteria

✅ **Timing**:
- Parallel execution: 6 windows in <10s (vs ~30s sequential)
- Phase 1 API call: <15s per window
- Phase 2 API call: <30s
- Full pipeline (18-33s bucket): <90s

✅ **Cost**:
- Phase 1 window: $0.08-0.10 per call
- Phase 2 synthesis: $0.15-0.20 per call
- Full bucket (18-33s): $0.65-0.80
- Full pipeline (8 buckets): $3.50-4.50 per client

### 8.4 Code Quality Success Criteria

✅ **Coverage**:
- Line coverage: >80%
- Branch coverage: >70%
- All critical paths tested (retry, checkpoint, scenarios)

✅ **Maintainability**:
- All tests have clear docstrings
- Test data fixtures well-organized
- Mock API responses match real API format
- Tests are reproducible (no flaky tests)

---

## Appendix A: Test Implementation Priority

### Week 1: HLD Tests + Critical Gaps (13 hours)

**Day 1-2: HLD Unit Tests** (6 hours)
- Test 1.1-1.4: Phase 1 preprocessing
- Test 2.1-2.8: Phase 2 preprocessing
- Test 3.1-3.2: Prompt builders

**Day 3: Critical Gaps** (3.5 hours)
- Gap 1: Checkpoint/resume (45 min)
- Gap 2: Parallel execution (30 min)
- Gap 3: Retry logic timing (30 min)
- Gap 4: Feature-based reports (30 min)
- Gap 5: Output validation (45 min)

**Day 4-5: HLD Integration Tests** (4 hours)
- Test 4.1-4.2: API integration
- Test 5.1-5.2: Full pipeline + scenarios
- Test 6.1-6.5: Edge cases

### Week 2: Medium Priority Gaps + Live API Testing (3 hours)

**Day 1: Medium Priority Gaps** (1.5 hours)
- Gap 6: Cluster path extraction (30 min)
- Gap 7: Cross-window patterns (20 min)
- Gap 8: Universal principles (15 min)
- Gap 9: Prompt edge cases (30 min)

**Day 2: Live API Testing** (1.5 hours)
- Small sample tests (10 videos)
- Full integration test (100 videos)
- Scenario testing (4 scenarios)

**Day 3: CI/CD Setup + Documentation**
- GitHub Actions workflow
- Test data preparation
- README updates

---

## Appendix B: Mock API Response Templates

```python
# tests/fixtures/mock_api_responses/hook_analysis_success.json
{
  "clusters": [
    {
      "id": "C1",
      "size": 12,
      "percentage": 40.0,
      "defining_features": [
        "eye_contact_rate: HIGH (0.85) - Direct gaze 85% of time",
        "word_count: BRIEF (≤20 words) - Ultra-concise messaging",
        "hook_type: QUESTION - Curiosity-driven opening"
      ],
      "rf_validation": {
        "insight": "Strong RF alignment (3/5): eye_contact_rate, word_count, hook_type all match top 5 RF features",
        "alignment_score": "3/5"
      },
      "creative_strategy": "High-energy, brief, question-based hooks with direct eye contact"
    },
    {
      "id": "C2",
      "size": 10,
      "percentage": 33.3,
      "defining_features": [
        "text_overlay_present: TRUE - On-screen text reinforcement",
        "music_energy: HIGH (0.78) - Upbeat background music",
        "transition_count: MODERATE (2-3 transitions) - Visual variety"
      ],
      "rf_validation": {
        "insight": "Moderate RF alignment (2/5): text_overlay matches RF, music_energy shows creative novelty",
        "alignment_score": "2/5"
      },
      "creative_strategy": "Text-heavy, high-energy hooks with moderate visual transitions"
    },
    {
      "id": "C3",
      "size": 8,
      "percentage": 26.7,
      "defining_features": [
        "word_count: DENSE (≥80 words) - Information-rich opening (BIMODAL - ALTERNATIVE STRATEGY to C1)",
        "speaking_rate: FAST (180+ WPM) - Rapid delivery",
        "hand_gesture_count: HIGH (12+ gestures) - Animated presentation"
      ],
      "rf_validation": {
        "insight": "Low RF alignment (1/5): word_count diverges from RF trend (creative novelty cluster)",
        "alignment_score": "1/5"
      },
      "creative_strategy": "Dense, fast-paced, highly animated hooks with information overload"
    }
  ],
  "metadata": {
    "window_type": "hook",
    "bucket": "18-33s",
    "hashtag": "nutrition",
    "total_videos": 30,
    "n_clusters": 3,
    "timestamp": "2025-10-21T12:30:45Z"
  }
}
```

---

**End of Stage7Tests.md**

**Next Steps**:
1. Review this test plan with team
2. Create test fixtures (Appendix B templates)
3. Implement Week 1 tests (HLD + Critical Gaps)
4. Run live API tests on pilot data
5. Deploy to production after all tests pass
