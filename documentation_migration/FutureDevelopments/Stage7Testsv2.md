# Stage 7: LLM Analysis - Comprehensive Testing Strategy v2

> **Parent Document**: LLMAnalysisCHILD.md Section 8 (Testing Strategy)
> **TI Document**: LLMAnalysisCHILDTI.md Section 7 (Complete Example Traces)
> **Version**: 2.0
> **Date**: 2025-10-21
> **Status**: ACTIVE - Multi-CLI Instance Workflow

---

## 🚀 Section 0: Quick Start for New CLI Instance

**Last Updated**: 2025-10-21
**Current Phase**: Phase 0 - HLD Tests
**Resume Point**: HLD-01 (Section 3.1.1)
**Total Progress**: 0/50 tests (0%)

### 0.1 Resume Instructions

**For the FIRST CLI Instance** (Fixtures Not Generated Yet):
1. **Generate test fixtures** (one-time setup):
   ```bash
   cd /home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/tests
   python scripts/generate_test_fixtures.py --all --skip-live-api
   ```
   - Takes ~2 minutes
   - Generates ~5 MB of test data
   - Update Section 13.3 (Fixture Update Log) after completion

2. Read this section (Section 0) - 3 minutes
3. Check Section 0.2 (Test Status Summary) - identify current phase
4. Jump to Section 0.3 (Find Next Test) - get test ID (HLD-01)
5. Read detailed test spec in Section 3.1.1
6. Create `tests/test_phase1_preprocessing.py` and implement test
7. Run test: `pytest tests/test_phase1_preprocessing.py::test_bimodal_detection -v`
8. Document result in Section 11 (Test Execution Log)
9. Update checklist in Section 2 (mark test complete with ✅ and timestamp)
10. Commit changes to this file before ending session

**For SUBSEQUENT CLI Instances** (Fixtures Already Generated):
1. Read this section (Section 0) - 2 minutes
2. Check Section 0.2 (Test Status Summary) - identify current phase
3. Jump to Section 0.3 (Find Next Test) - get test ID
4. Read detailed test spec in Sections 3-6
5. Implement test in appropriate file
6. Run test, document result in Section 11 (Test Execution Log)
7. Update checklist in Section 2 (mark test complete)
8. If bug found → Log in Section 10 (Bug Discovery Log)
9. Commit changes to this file before ending session
10. Next CLI instance resumes from updated checklist

**Critical Files to Update**:
- `Stage7Testsv2.md` Section 0.2 (progress bars)
- `Stage7Testsv2.md` Section 0.4 (files modified this session)
- `Stage7Testsv2.md` Section 2 (checklist - mark tests complete)
- `Stage7Testsv2.md` Section 11 (execution log - add new run entry)
- `Stage7Testsv2.md` Section 13.3 (fixture update log - if generating fixtures)
- Test implementation files (`.py` files)
- `Stage7Testsv2.md` Section 10 (if bugs found)

---

### 0.2 Test Status Summary (Auto-Update This Section)

```
Phase 0: HLD Tests (Baseline)
[----------] 0/30 (0%)

Phase 1: P0 Critical Tests
[----------] 0/14 (0%)

Phase 2: P1 High Priority Tests
[----------] 0/6 (0%)

Overall Progress
[----------] 0/50 (0%)
```

**Legend**: `[####------]` = 40% complete, `[##########]` = 100% complete

**Current Session Info**:
- CLI Instance: #1
- Started: 2025-10-21
- Working On: HLD-01 (Bimodal detection test)
- Blockers: None

---

### 0.3 Find Next Test (Jump-to-Work)

**Phase 0 - HLD Tests**: ⏳ Not Started
- **Next Test**: HLD-01 (Section 3.1.1)
- **File**: `tests/test_phase1_preprocessing.py`
- **Complexity**: Simple
- **Est. Time**: 10 minutes

**Phase 1 - P0 Critical Tests**: ⏳ Pending
- **Next Test**: P0-01 (Section 4.1.1)
- **File**: `tests/test_checkpoint_resume.py`
- **Complexity**: Medium
- **Est. Time**: 30 minutes

**Phase 2 - P1 High Priority Tests**: ⏳ Pending
- **Next Test**: P1-01 (Section 5.1.1)

---

### 0.4 Files Modified This Session

**Session Start**: 2025-10-21 (Current Session)
**Session End**: (Ongoing)

- `Stage7Testsv2.md` (updated: Sections 0.2, 0.3, 0.4, 2, 11, 13.3)
- No test files created yet

**Note**: Update this list before ending CLI session.

---

## 1. Testing Philosophy & Strategy

### 1.1 Multi-CLI Instance Workflow

This test suite is designed for **iterative implementation** across multiple Claude Code CLI sessions. Each CLI instance:
- Reads the current test status
- Picks up the next unchecked test
- Implements and runs the test
- Updates the checklist and logs results
- Commits changes before ending

**Benefits**:
- ✅ No context loss between sessions
- ✅ Clear progress tracking
- ✅ Parallel work possible (different test files)
- ✅ Bug discovery documented immediately

**Workflow Diagram**:
```
CLI Instance 1          CLI Instance 2          CLI Instance 3
     |                       |                       |
  Read Status            Read Status            Read Status
     |                       |                       |
  Test HLD-01            Test HLD-05            Test P0-01
     |                       |                       |
  Update Checklist       Update Checklist       Discover Bug #1
     |                       |                       |
  Commit Changes         Commit Changes         Log Bug + Commit
     |                       |                       |
  (Session End)          (Session End)          (Session End)
```

---

### 1.2 Bidirectional Sync Protocol

**Tests → HLD**: When tests discover implementation bugs
1. Log bug in Section 10 (Bug Discovery Log)
2. Mark bug for HLD review (flag: `[HLD-REVIEW-NEEDED]`)
3. Next CLI instance reviews bug log
4. If bug is P0/P1 and affects HLD correctness → **Manual update** to HLD Section 11
5. If bug is implementation-only → No HLD update needed

**HLD → Tests**: When HLD is updated with new edge cases
1. Add new test to Section 12 (Deferred Tests)
2. Prioritize test (P0/P1/P2)
3. Move to appropriate phase checklist (Section 2)

**Manual Review Gate**: All bugs require human review before HLD updates to prevent noise.

---

### 1.3 Test Categories

| Category | Description | Test Count | Priority |
|----------|-------------|------------|----------|
| **HLD Baseline** | Original 30 tests from HLD Section 8 | 30 tests | P0 |
| **Checkpoint/Resume** | Cost-critical checkpoint logic | 4 tests | P0 |
| **Parallel Execution** | Thread safety, race conditions | 3 tests | P0 |
| **Retry Logic** | Exponential backoff, error handling | 2 tests | P0 |
| **Feature-Based Reports** | JSON schema compliance | 3 tests | P0 |
| **Output Validation** | Hallucination detection, 3-layer validation | 2 tests | P0 |
| **Cluster Path Extraction** | Path frequency accuracy | 1 test | P1 |
| **Cross-Window Patterns** | Temporal progression logic | 1 test | P1 |
| **Universal Principles** | Edge cases (< 7 features) | 1 test | P1 |
| **Prompt Edge Cases** | Empty preprocessing outputs | 3 tests | P1 |

---

### 1.4 Exit Criteria

**Phase 0 (HLD Tests) Complete**:
- ✅ All 30 HLD tests pass
- ✅ No P0 bugs discovered
- ✅ Fixtures generated for HLD tests

**Phase 1 (P0 Critical) Complete**:
- ✅ All 14 P0 tests pass
- ✅ All P0 bugs fixed (none in "OPEN" status)
- ✅ Code coverage ≥70%

**Phase 2 (P1 High Priority) Complete**:
- ✅ All 6 P1 tests pass
- ✅ All P1 bugs fixed
- ✅ Code coverage ≥85%

**Final Sign-Off Criteria**:
- ✅ All 50 tests pass
- ✅ No P0/P1 bugs in "OPEN" status
- ✅ HLD Section 8 (Testing Strategy) updated with final test count
- ✅ All fixtures committed to repo
- ✅ Manual review of Section 14 (HLD Sync Checklist) complete

---

## 2. Test Inventory (Master Checklist)

### 2.1 Overview

| Phase | Test Count | Status | Progress |
|-------|------------|--------|----------|
| Phase 0: HLD Baseline | 30 | ⏳ Not Started | 0/30 (0%) |
| Phase 1: P0 Critical | 14 | ⏳ Not Started | 0/14 (0%) |
| Phase 2: P1 High Priority | 6 | ⏳ Not Started | 0/6 (0%) |
| **Total** | **50** | ⏳ **Not Started** | **0/50 (0%)** |

**Estimated Total Time**: ~16 hours (50 tests × 20 min avg)

---

### 2.2 Phase 0: HLD Tests (30 tests - BASELINE)

**Purpose**: Implement the 30 tests specified in LLMAnalysisCHILD.md Section 8.

**Progress**: [----------] 0/30 (0%)

#### Phase 1 Preprocessing Tests (8 tests)

- [ ] **HLD-01**: Bimodal detection - 30%/30% split
  - File: `test_phase1_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 10 min
  - Fixture: None (unit test)
  - Status: ⏳ Not Started
  - Test Function: `test_bimodal_detection`

- [ ] **HLD-02**: High-contrast filtering - 21 features → 8 with ≥0.20 contrast
  - File: `test_phase1_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 10 min
  - Fixture: `fixtures/phase1/kmeans_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_high_contrast_filtering`

- [ ] **HLD-03**: RF alignment scoring - 2/5 alignment
  - File: `test_phase1_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 10 min
  - Fixture: `fixtures/phase1/rf_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_rf_alignment_scoring`

- [ ] **HLD-04**: Feature enrichment with RF metadata
  - File: `test_phase1_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 10 min
  - Fixture: `fixtures/phase1/rf_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_feature_enrichment`

- [ ] **HLD-05**: Bimodal detection boundary - 29.9% vs 30.0%
  - File: `test_phase1_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 5 min
  - Fixture: None (unit test)
  - Status: ⏳ Not Started
  - Test Function: `test_bimodal_boundary_cases`

- [ ] **HLD-06**: High-contrast threshold - 19.9% vs 20.0%
  - File: `test_phase1_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 5 min
  - Fixture: `fixtures/phase1/kmeans_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_high_contrast_threshold`

- [ ] **HLD-07**: RF alignment tolerance - 14.9% vs 15.0%
  - File: `test_phase1_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 5 min
  - Fixture: `fixtures/phase1/rf_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_rf_alignment_tolerance`

- [ ] **HLD-08**: Feature enrichment fallback (feature not in RF top 10)
  - File: `test_phase1_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 10 min
  - Fixture: `fixtures/phase1/rf_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_feature_enrichment_fallback`

#### Phase 2 Preprocessing Tests (8 tests)

- [ ] **HLD-09**: Scenario A - 5+ paths above 10% threshold
  - File: `test_phase2_preprocessing.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/phase2/cluster_paths_scenario_a.json`
  - Status: ⏳ Not Started
  - Test Function: `test_scenario_a`

- [ ] **HLD-10**: Scenario B - Exactly 2 paths above threshold
  - File: `test_phase2_preprocessing.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/phase2/cluster_paths_scenario_b.json`
  - Status: ⏳ Not Started
  - Test Function: `test_scenario_b`

- [ ] **HLD-11**: Scenario C - Exactly 1 path above threshold
  - File: `test_phase2_preprocessing.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/phase2/cluster_paths_scenario_c.json`
  - Status: ⏳ Not Started
  - Test Function: `test_scenario_c`

- [ ] **HLD-12**: Scenario D - 0 paths above threshold (high fragmentation)
  - File: `test_phase2_preprocessing.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/phase2/cluster_paths_scenario_d.json`
  - Status: ⏳ Not Started
  - Test Function: `test_scenario_d`

- [ ] **HLD-13**: Confidence level classification - 19.9% vs 20.0% vs 15.0%
  - File: `test_phase2_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 5 min
  - Fixture: None (unit test)
  - Status: ⏳ Not Started
  - Test Function: `test_confidence_classification`

- [ ] **HLD-14**: Universal principles - top 5-7 features
  - File: `test_phase2_preprocessing.py`
  - Complexity: Simple
  - Est. Time: 10 min
  - Fixture: `fixtures/phase2/rf_video_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_universal_principles`

- [ ] **HLD-15**: Cross-window patterns - 6 windows
  - File: `test_phase2_preprocessing.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/phase2/window_analyses.json`
  - Status: ⏳ Not Started
  - Test Function: `test_cross_window_patterns`

- [ ] **HLD-16**: Feature-based reports - 3 reports, distinct categories
  - File: `test_phase2_preprocessing.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/phase2/rf_features.json`
  - Status: ⏳ Not Started
  - Test Function: `test_feature_based_reports`

#### Prompt Builder Tests (5 tests)

- [ ] **HLD-17**: Phase 1 prompt - all sections present
  - File: `test_prompts.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/prompts/phase1_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_phase1_prompt_sections`

- [ ] **HLD-18**: Phase 1 prompt - bimodal feature formatting
  - File: `test_prompts.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/prompts/bimodal_feature.json`
  - Status: ⏳ Not Started
  - Test Function: `test_phase1_bimodal_formatting`

- [ ] **HLD-19**: Phase 1 prompt - RF alignment display
  - File: `test_prompts.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/prompts/phase1_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_phase1_rf_alignment`

- [ ] **HLD-20**: Phase 2 prompt - Scenario A instructions
  - File: `test_prompts.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/prompts/scenario_a_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_phase2_scenario_a_prompt`

- [ ] **HLD-21**: Phase 2 prompt - Scenario D with feature-based embedding
  - File: `test_prompts.py`
  - Complexity: Complex
  - Est. Time: 20 min
  - Fixture: `fixtures/prompts/scenario_d_data.json`
  - Status: ⏳ Not Started
  - Test Function: `test_phase2_scenario_d_prompt`

#### API Integration Tests (6 tests)

- [ ] **HLD-22**: Phase 1 successful analysis (mock API)
  - File: `test_api_integration.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/api/success_response.json`
  - Status: ⏳ Not Started
  - Test Function: `test_phase1_success`

- [ ] **HLD-23**: Retry on 503 error (mock API)
  - File: `test_api_integration.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/api/503_error.json`
  - Status: ⏳ Not Started
  - Test Function: `test_retry_on_503`

- [ ] **HLD-24**: Retry on 429 rate limit (mock API)
  - File: `test_api_integration.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/api/429_error.json`
  - Status: ⏳ Not Started
  - Test Function: `test_retry_on_429`

- [ ] **HLD-25**: Non-retryable 401 error (mock API)
  - File: `test_api_integration.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/api/401_error.json`
  - Status: ⏳ Not Started
  - Test Function: `test_non_retryable_401`

- [ ] **HLD-26**: Phase 2 synthesis success (mock API)
  - File: `test_api_integration.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/api/synthesis_response.json`
  - Status: ⏳ Not Started
  - Test Function: `test_phase2_synthesis`

- [ ] **HLD-27**: JSON truncation handling (max_tokens exceeded)
  - File: `test_api_integration.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/api/truncated_json.json`
  - Status: ⏳ Not Started
  - Test Function: `test_json_truncation`

#### Integration Tests (3 tests)

- [ ] **HLD-28**: Full pipeline E2E (18-33s bucket, mock API)
  - File: `test_integration.py`
  - Complexity: Complex
  - Est. Time: 30 min
  - Fixture: `fixtures/integration/bucket_18-33s/`
  - Status: ⏳ Not Started
  - Test Function: `test_full_pipeline`

- [ ] **HLD-29**: All 4 scenarios produce 3 reports
  - File: `test_integration.py`
  - Complexity: Complex
  - Est. Time: 30 min
  - Fixture: `fixtures/integration/all_scenarios/`
  - Status: ⏳ Not Started
  - Test Function: `test_all_scenarios_3_reports`

- [ ] **HLD-30**: Output validation - schema compliance
  - File: `test_integration.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/integration/bucket_18-33s/`
  - Status: ⏳ Not Started
  - Test Function: `test_output_schema_compliance`

---

### 2.3 Phase 1: P0 Critical Tests (14 tests)

**Purpose**: Tests that prevent **production crashes** and **cost overruns**.

**Progress**: [----------] 0/14 (0%)

**Must Pass Before Production Deployment**: ✅ Required

#### Category: Checkpoint/Resume Logic (4 tests)

- [ ] **P0-01**: Checkpoint creation after first window
  - File: `test_checkpoint_resume.py`
  - Complexity: Medium
  - Est. Time: 30 min
  - Fixture: `fixtures/checkpoint/partial_run/`
  - Status: ⏳ Not Started
  - **Critical**: Prevents cost overruns from re-running completed windows

- [ ] **P0-02**: Resume from checkpoint - skip completed windows
  - File: `test_checkpoint_resume.py`
  - Complexity: Medium
  - Est. Time: 30 min
  - Fixture: `fixtures/checkpoint/partial_run/`
  - Status: ⏳ Not Started
  - **Critical**: Verifies cost-saving resume logic

- [ ] **P0-03**: Checkpoint corruption handling
  - File: `test_checkpoint_resume.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/checkpoint/corrupted_checkpoint/`
  - Status: ⏳ Not Started
  - **Critical**: Prevents crash on invalid checkpoint file

- [ ] **P0-04**: Partial completion (3/6 windows)
  - File: `test_checkpoint_resume.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/checkpoint/partial_run/`
  - Status: ⏳ Not Started

#### Category: Parallel Execution (3 tests)

- [ ] **P0-05**: Parallel execution timing (6 windows in <10s)
  - File: `test_parallel_execution.py`
  - Complexity: Complex
  - Est. Time: 30 min
  - Fixture: Mock API with 5s delay
  - Status: ⏳ Not Started
  - **Critical**: Verifies actual parallel execution

- [ ] **P0-06**: Thread safety of status file writes
  - File: `test_parallel_execution.py`
  - Complexity: Complex
  - Est. Time: 30 min
  - Fixture: `fixtures/parallel/concurrent_writes/`
  - Status: ⏳ Not Started
  - **Critical**: Prevents race conditions

- [ ] **P0-07**: Partial failure handling (1 window fails, others succeed)
  - File: `test_parallel_execution.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: Mock API with partial failure
  - Status: ⏳ Not Started

#### Category: Retry Logic (2 tests)

- [ ] **P0-08**: Exponential backoff timing (0s, 2s, 4s)
  - File: `test_retry_logic.py`
  - Complexity: Medium
  - Est. Time: 30 min
  - Fixture: Mock API with controlled delays
  - Status: ⏳ Not Started
  - **Critical**: Prevents API hammering

- [ ] **P0-09**: Retryable vs non-retryable error distinction
  - File: `test_retry_logic.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: Mock API with various error codes
  - Status: ⏳ Not Started
  - **Critical**: Ensures correct error handling

#### Category: Feature-Based Reports (3 tests)

- [ ] **P0-10**: JSON schema compliance
  - File: `test_feature_based_reports.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/feature_reports/rf_features.json`
  - Status: ⏳ Not Started
  - **Critical**: Prevents LLM parsing failures

- [ ] **P0-11**: Category diversity (visual, audio, behavioral)
  - File: `test_feature_based_reports.py`
  - Complexity: Simple
  - Est. Time: 15 min
  - Fixture: `fixtures/feature_reports/rf_features.json`
  - Status: ⏳ Not Started

- [ ] **P0-12**: Insufficient features handling
  - File: `test_feature_based_reports.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/feature_reports/minimal_features.json`
  - Status: ⏳ Not Started

#### Category: Output Validation (2 tests)

- [ ] **P0-13**: No hallucinated features
  - File: `test_output_validation.py`
  - Complexity: Complex
  - Est. Time: 30 min
  - Fixture: `fixtures/validation/input_output_pairs.json`
  - Status: ⏳ Not Started
  - **Critical**: Prevents data integrity issues

- [ ] **P0-14**: No hallucinated video IDs
  - File: `test_output_validation.py`
  - Complexity: Complex
  - Est. Time: 30 min
  - Fixture: `fixtures/validation/input_output_pairs.json`
  - Status: ⏳ Not Started
  - **Critical**: Prevents data integrity issues

---

### 2.4 Phase 2: P1 High Priority Tests (6 tests)

**Purpose**: Tests for **edge cases** and **cross-bucket compatibility**.

**Progress**: [----------] 0/6 (0%)

**Can Deploy Without These**: ⚠️ Not Recommended (deploy with risk acknowledgment)

#### Category: Edge Cases (6 tests)

- [ ] **P1-01**: Cluster path extraction - videos missing from some windows
  - File: `test_cluster_paths.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/edge_cases/missing_videos.json`
  - Status: ⏳ Not Started

- [ ] **P1-02**: Cross-window patterns - 2 windows only
  - File: `test_cross_window.py`
  - Complexity: Medium
  - Est. Time: 15 min
  - Fixture: `fixtures/edge_cases/two_windows.json`
  - Status: ⏳ Not Started

- [ ] **P1-03**: Universal principles - <7 features
  - File: `test_universal_principles.py`
  - Complexity: Simple
  - Est. Time: 10 min
  - Fixture: `fixtures/edge_cases/few_features.json`
  - Status: ⏳ Not Started

- [ ] **P1-04**: Prompt builder - empty preprocessing outputs
  - File: `test_prompts_edge_cases.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/edge_cases/empty_outputs.json`
  - Status: ⏳ Not Started

- [ ] **P1-05**: Prompt builder - malformed RF data
  - File: `test_prompts_edge_cases.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/edge_cases/malformed_rf.json`
  - Status: ⏳ Not Started

- [ ] **P1-06**: Feature-based report embedding in Phase 2 prompt
  - File: `test_prompts_edge_cases.py`
  - Complexity: Medium
  - Est. Time: 20 min
  - Fixture: `fixtures/edge_cases/scenario_b_data.json`
  - Status: ⏳ Not Started

---

## 3. HLD Test Specifications (Phase 0)

### 3.1 Phase 1 Preprocessing Tests

#### 3.1.1 Test HLD-01: Bimodal Detection - 30%/30% Split

**Purpose**: Validate that bimodal pattern detection correctly identifies features with both high and low strategies.

**Implementation**:
```python
def test_bimodal_detection():
    """Test: 30%/30% split detected as bimodal."""
    from ml_pipeline.stage7_llm_analysis import detect_bimodal_pattern

    distribution = {
        'top_performers': {
            'high_percentage': 0.40,
            'low_percentage': 0.35
        }
    }

    result = detect_bimodal_pattern(distribution)

    assert result['is_bimodal'] == True
    assert result['pattern_label'] == "BIMODAL"
    assert "BOTH strategies work" in result['interpretation']

    print("✓ Bimodal detection passed: 40%/35% split")
```

**Pass Criteria**:
- ✅ `is_bimodal=True` when both ≥30%
- ✅ Pattern label correctly set
- ✅ Interpretation explains both strategies

**Failure Modes**:
- ❌ `is_bimodal=False` when should be True
- ❌ Boundary case failure (29.9% vs 30.0%)

---

#### 3.1.2 Test HLD-02: High-Contrast Filtering

**Purpose**: Validate that only features with ≥0.20 contrast are kept.

**Implementation**:
```python
def test_high_contrast_filtering():
    """Test: Filter 21 features to 8 with ≥0.20 contrast."""
    from ml_pipeline.stage7_llm_analysis import identify_high_contrast_features

    kmeans_data = load_fixture('fixtures/phase1/kmeans_data.json')

    result = identify_high_contrast_features(kmeans_data, threshold=0.20)

    assert len(result['clusters'][0]['high_contrast_features']) == 8

    for feature in result['clusters'][0]['high_contrast_features']:
        assert feature['max_contrast'] >= 0.20

    print("✓ High-contrast filtering passed: 8/21 features kept")
```

---

*(Continue similar detailed structure for HLD-03 through HLD-30)*

---

## 4. P0 Critical Test Specifications (Phase 1)

### 4.1 Checkpoint/Resume Tests

#### 4.1.1 Test P0-01: Checkpoint Creation After First Window

**Purpose**: Validate that `.phase1_status.json` is created after first window completes.

**Context**: This is a **cost-critical** feature. If checkpoint logic fails, re-running Stage 7 could cost an additional $4.18 per client.

**Implementation**:
```python
def test_checkpoint_creation():
    """Test: .phase1_status.json created after first window."""
    from ml_pipeline.stage7_llm_analysis import run_phase1_parallel

    bucket_path = 'tests/fixtures/checkpoint/partial_run/'

    # Run only first window
    run_phase1_parallel(bucket_path, "18-33s", "nutrition", ["hook"])

    status_file = os.path.join(bucket_path, "ml_analysis/llm/.phase1_status.json")
    assert os.path.exists(status_file), "Checkpoint file not created"

    with open(status_file) as f:
        status = json.load(f)

    assert 'completed_windows' in status
    assert 'hook' in status['completed_windows']
    assert 'timestamp' in status

    print("✓ Checkpoint created successfully")
```

**Pass Criteria**:
- ✅ `.phase1_status.json` exists
- ✅ Contains `completed_windows` list
- ✅ `hook` in completed list
- ✅ Timestamp present

**Failure Modes**:
- ❌ Checkpoint file not created
- ❌ Checkpoint created but missing fields
- ❌ Checkpoint created but empty

**Fixture Required**:
- `fixtures/checkpoint/partial_run/` (Stage 6 outputs for 18-33s bucket)

---

#### 4.1.2 Test P0-02: Resume from Checkpoint

**Purpose**: Validate that Stage 7 resumes from checkpoint and skips completed windows.

**Implementation**:
```python
def test_resume_from_checkpoint():
    """Test: Resume skips completed windows."""
    from ml_pipeline.stage7_llm_analysis import run_phase1_parallel

    bucket_path = 'tests/fixtures/checkpoint/partial_run/'

    # Step 1: Run 3/6 windows
    run_phase1_parallel(bucket_path, "18-33s", "nutrition",
                       ["hook", "middle_1", "middle_2"])

    # Verify checkpoint
    status_file = os.path.join(bucket_path, "ml_analysis/llm/.phase1_status.json")
    with open(status_file) as f:
        status = json.load(f)
    assert len(status['completed_windows']) == 3

    # Step 2: Resume with all 6 windows (should skip first 3)
    with mock_api_call_counter() as counter:
        run_phase1_parallel(bucket_path, "18-33s", "nutrition",
                          ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"])

        # Should make only 3 API calls (not 6)
        assert counter.call_count == 3, f"Expected 3 API calls, got {counter.call_count}"

    print("✓ Resume from checkpoint passed")
```

**Pass Criteria**:
- ✅ Only 3 API calls made (not 6)
- ✅ Final checkpoint has all 6 windows
- ✅ No errors during resume

**Cost Impact**: **CRITICAL** - Failing this test means potential 2x cost overruns

---

*(Continue similar detailed structure for P0-03 through P0-14)*

---

## 10. Bug Discovery Log

**Purpose**: Track all bugs found during test implementation. These feed back into HLD Section 11.

**Instructions**:
1. Log bug immediately when discovered
2. Assign severity (P0/P1/P2)
3. Flag for HLD review if P0/P1 and affects HLD correctness
4. Update bug status when fixed
5. Cross-reference bug in test execution log

**Current Bug Count**: 0

---

### Bug Entry Template

```markdown
#### Bug #[N]: [Short Description]

**Discovered By**: Test [ID] ([test function name])
**Date**: YYYY-MM-DD HH:MM
**Severity**: P0 (Production Crash) | P1 (Incorrect Output) | P2 (Minor Issue)
**Status**: OPEN | IN PROGRESS | FIXED | DEFERRED
**HLD Review**: [HLD-REVIEW-NEEDED] | [HLD-UPDATE-NOT-REQUIRED]

**Description**:
[Detailed explanation of the bug]

**Root Cause**:
[Why the bug exists - trace back to code logic]

**Reproduction Steps**:
1. [Step 1]
2. [Step 2]
3. Observe: [Error]

**Code Location**:
- File: `[filename]`
- Function: `[function_name]`
- Line: [line number]
- Code Snippet:
```python
[problematic code]
```

**Fix Applied**:
```python
# Before:
[old code]

# After:
[new code]
```

**Impact**:
- Users Affected: [Production | Testing Only]
- Data Corruption Risk: [Yes | No]
- Crash Risk: [Yes | No]
- Cost Impact: [$ amount if applicable]

**HLD Update Required** (if flagged for review):
- [ ] Update HLD Section 6.2 (Error Handling) with edge case
- [ ] Update HLD Section 8.1 (Unit Tests) to add this test
- [ ] Update TI Section 4.X pseudocode with fix

**Related Tests**:
- [Test ID] - [relationship]

**Lessons Learned**:
[What we learned from this bug]

**Fixed By**: CLI Instance #[N]
**Fix Verified**: [Date] by Test [ID]
```

---

*(Bug entries will be added here as tests discover issues)*

---

## 11. Test Execution Log

**Purpose**: Chronological log of test runs across CLI instances. Helps debug flaky tests and track progress.

**Instructions**:
1. Add new log entry at the start of each CLI session
2. Update entry at the end of session with results
3. Include failed test details with error messages
4. Document environment (Python version, package versions)
5. List all files modified during session

**Current Run Count**: 0

---

### Run Entry Template

```markdown
### Run #[N]: [Date] [Time]

**CLI Instance**: #[N]
**Phase**: Phase [0/1/2] - [Phase Name]
**Tests Run**: [Start Test ID] to [End Test ID] ([N] tests)
**Duration**: [HH:MM]
**Session Start**: YYYY-MM-DD HH:MM
**Session End**: YYYY-MM-DD HH:MM

#### Results Summary
- ✅ Passed: [N] tests ([Test IDs])
- ❌ Failed: [N] tests ([Test IDs])
- ⏭️ Skipped: [N] tests ([Test IDs])

#### Test Results Details

**Passed Tests**:
- [Test ID]: [Test Name] - [Duration] - [Notes if any]

**Failed Tests**:

**[Test ID]: [Test Name]**
- Error: `[Error message]`
- Traceback:
```
[Full traceback]
```
- Line: [filename]:[line number]
- Root Cause: [Analysis]
- Bug Logged: Bug #[N]
- Action: [Fix implemented | Investigation needed | Deferred]

#### Environment
- Python: [version]
- anthropic: [version]
- pytest: [version]
- OS: [Linux/Windows/Mac] [version]

#### Fixtures Used
- [Fixture name] - [status: created/modified/used]

#### Files Modified
- `[filename]` ([description of changes])

#### Bugs Discovered
- Bug #[N]: [Brief description]

#### Next Steps
1. [Action item 1]
2. [Action item 2]

#### Notes
[Any additional context for next CLI instance]
```

---

*(Run entries will be added here as tests are executed)*

---

## 12. Deferred Tests & Future Work

**Purpose**: Tests that are out of scope but should be added later.

**Current Deferred Count**: 0

---

### 12.1 Deferred to Post-MVP

*(No deferred tests yet)*

---

### 12.2 Tests Added from HLD Updates

**Purpose**: When HLD is updated with new edge cases, add corresponding tests here.

*(No HLD updates yet)*

---

## 13. Fixture Management

### 13.1 Fixture Inventory

**Total Fixture Sets**: 7

#### Fixture Set 1: Phase 1 Preprocessing
- **Location**: `tests/fixtures/phase1/`
- **Size**: ~500 KB (estimated)
- **Description**: RF data, K-Means data for preprocessing tests
- **Files**:
  - `rf_data.json` (10 features with importance scores, distributions)
  - `kmeans_data.json` (3 clusters, 21 features, 8 with ≥0.20 contrast)
  - `bimodal_feature.json` (40% high, 35% low)
- **Used By**: HLD-01 to HLD-08
- **Status**: ⏳ Pending Generation

#### Fixture Set 2: Phase 2 Preprocessing
- **Location**: `tests/fixtures/phase2/`
- **Size**: ~800 KB (estimated)
- **Description**: Cluster paths for all 4 scenarios, window analyses
- **Files**:
  - `cluster_paths_scenario_a.json` (5+ paths ≥10%)
  - `cluster_paths_scenario_b.json` (2 paths ≥10%)
  - `cluster_paths_scenario_c.json` (1 path ≥10%)
  - `cluster_paths_scenario_d.json` (0 paths ≥10%)
  - `window_analyses.json` (6 windows)
  - `rf_video_data.json` (15 features)
  - `rf_features.json` (for feature-based reports)
- **Used By**: HLD-09 to HLD-16
- **Status**: ⏳ Pending Generation

#### Fixture Set 3: Prompts
- **Location**: `tests/fixtures/prompts/`
- **Size**: ~1 MB (estimated)
- **Description**: Complete data for prompt building tests
- **Files**:
  - `phase1_data.json` (kmeans + RF data)
  - `bimodal_feature.json` (for bimodal formatting test)
  - `scenario_a_data.json` (5+ paths)
  - `scenario_d_data.json` (0 paths, feature-based reports)
- **Used By**: HLD-17 to HLD-21
- **Status**: ⏳ Pending Generation

#### Fixture Set 4: API Responses
- **Location**: `tests/fixtures/api/`
- **Size**: ~200 KB (estimated)
- **Description**: Mock API responses for all test scenarios
- **Files**:
  - `success_response.json` (valid Phase 1 output)
  - `synthesis_response.json` (valid Phase 2 output)
  - `503_error.json` (service unavailable)
  - `429_error.json` (rate limit)
  - `401_error.json` (unauthorized)
  - `truncated_json.json` (incomplete JSON)
- **Used By**: HLD-22 to HLD-27
- **Status**: ⏳ Pending Generation

#### Fixture Set 5: Integration
- **Location**: `tests/fixtures/integration/`
- **Size**: ~2 MB (estimated)
- **Description**: Complete Stage 6 outputs for E2E tests
- **Files**:
  - `bucket_18-33s/` (6 windows, 13 JSON files from Stage 6)
  - `all_scenarios/` (4 buckets, one for each scenario)
- **Used By**: HLD-28 to HLD-30
- **Status**: ⏳ Pending Generation

#### Fixture Set 6: Checkpoint/Resume
- **Location**: `tests/fixtures/checkpoint/`
- **Size**: ~1 MB (estimated)
- **Description**: Partial runs, corrupted checkpoints
- **Files**:
  - `partial_run/` (3/6 windows complete with checkpoint)
  - `corrupted_checkpoint/` (invalid JSON checkpoint)
- **Used By**: P0-01 to P0-04
- **Status**: ⏳ Pending Generation

#### Fixture Set 7: Edge Cases
- **Location**: `tests/fixtures/edge_cases/`
- **Size**: ~500 KB (estimated)
- **Description**: Edge case data for P1 tests
- **Files**:
  - `missing_videos.json` (videos missing from some windows)
  - `two_windows.json` (for cross-window test)
  - `few_features.json` (<7 features)
  - `empty_outputs.json` (0 bimodal features, 0 high-contrast)
  - `malformed_rf.json` (missing fields)
  - `scenario_b_data.json` (for feature-based embedding test)
- **Used By**: P1-01 to P1-06
- **Status**: ⏳ Pending Generation

---

### 13.2 Fixture Generation Scripts

**Script**: `tests/scripts/generate_test_fixtures.py`

```python
"""
Generate all test fixtures for Stage 7 tests.

Usage:
    python tests/scripts/generate_test_fixtures.py --fixture-set phase1
    python tests/scripts/generate_test_fixtures.py --fixture-set checkpoint
    python tests/scripts/generate_test_fixtures.py --all
"""

def generate_phase1_fixtures():
    """Generate Fixture Set 1: Phase 1 Preprocessing."""
    # TODO: Implement
    pass

def generate_phase2_fixtures():
    """Generate Fixture Set 2: Phase 2 Preprocessing."""
    # TODO: Implement
    pass

def generate_prompts_fixtures():
    """Generate Fixture Set 3: Prompts."""
    # TODO: Implement
    pass

def generate_api_fixtures():
    """Generate Fixture Set 4: API Responses."""
    # TODO: Implement
    pass

def generate_integration_fixtures():
    """Generate Fixture Set 5: Integration."""
    # TODO: Implement
    pass

def generate_checkpoint_fixtures():
    """Generate Fixture Set 6: Checkpoint/Resume."""
    # TODO: Implement
    pass

def generate_edge_case_fixtures():
    """Generate Fixture Set 7: Edge Cases."""
    # TODO: Implement
    pass
```

**Status**: Script template created, implementations pending

---

### 13.3 Fixture Update Log

**Purpose**: Track fixture creation and modifications.

- **2025-10-21**: Created fixture directory structure
- *(Updates will be added as fixtures are generated)*

---

## 14. HLD Synchronization Checklist

**Purpose**: Ensure bidirectional sync between Stage7Testsv2.md and HLD.

### 14.1 When to Update HLD (Manual Review Required)

**Trigger 1: Bug Discovery (P0 or P1)**
- [ ] Bug logged in Section 10 with `[HLD-REVIEW-NEEDED]` flag
- [ ] Bug affects HLD correctness (not just implementation detail)
- [ ] CLI instance reviews bug impact on HLD
- [ ] Update HLD Section 11 (Implementation Log) with bug entry
- [ ] Update HLD Section 6.2 (Error Handling) if edge case not documented
- [ ] Remove `[HLD-REVIEW-NEEDED]` flag, mark as `[HLD-UPDATED]`

**Trigger 2: Test Count Change (>5 new tests)**
- [ ] Added >5 new tests beyond the original 30 HLD tests
- [ ] Update HLD Section 8.1 (Unit Tests) with new test count
- [ ] Update HLD Section 8.3 (Test Data) with new fixtures

**Trigger 3: Test Strategy Change**
- [ ] Changed test approach (e.g., switched from mocking to real fixtures)
- [ ] Update HLD Section 8 (Testing Strategy) with rationale

---

### 14.2 When to Update Stage7Testsv2.md

**Trigger 1: HLD Edge Case Added**
- [ ] HLD Section 2.4 (Detailed Process) documents new edge case
- [ ] Add test to Section 12.2 (Tests Added from HLD Updates)
- [ ] Prioritize test (P0/P1/P2)
- [ ] Move to appropriate phase checklist when ready to implement

**Trigger 2: HLD Function Signature Changed**
- [ ] HLD Section 4 (Algorithmic Specifications) updates function
- [ ] Update affected test specifications in Sections 3-6
- [ ] Re-run tests to catch breaking changes

---

### 14.3 Final Sync Verification (Before Marking "Complete")

Run this checklist before marking testing phase "Complete":

- [ ] All P0/P1 bugs logged in Stage7Testsv2.md Section 10
- [ ] All P0/P1 bugs cross-referenced in HLD Section 11
- [ ] Test count in HLD Section 8.1 matches Stage7Testsv2.md Section 2 (50 tests)
- [ ] All fixtures documented in HLD Section 8.3
- [ ] No orphaned tests (tests without corresponding test file)
- [ ] No orphaned fixtures (fixtures not used by any test)
- [ ] HLD Section 8 (Testing Strategy) reflects final test structure
- [ ] TI Section 11.3 (Implementation Progress) marked complete

---

## Appendix A: Test File Structure

```
tests/
├── __init__.py
├── test_phase1_preprocessing.py    # HLD-01 to HLD-08
├── test_phase2_preprocessing.py    # HLD-09 to HLD-16
├── test_prompts.py                 # HLD-17 to HLD-21
├── test_api_integration.py         # HLD-22 to HLD-27
├── test_integration.py             # HLD-28 to HLD-30
├── test_checkpoint_resume.py       # P0-01 to P0-04
├── test_parallel_execution.py      # P0-05 to P0-07
├── test_retry_logic.py             # P0-08 to P0-09
├── test_feature_based_reports.py   # P0-10 to P0-12
├── test_output_validation.py       # P0-13 to P0-14
├── test_cluster_paths.py           # P1-01
├── test_cross_window.py            # P1-02
├── test_universal_principles.py    # P1-03
├── test_prompts_edge_cases.py      # P1-04 to P1-06
├── fixtures/
│   ├── phase1/
│   ├── phase2/
│   ├── prompts/
│   ├── api/
│   ├── integration/
│   ├── checkpoint/
│   └── edge_cases/
└── scripts/
    ├── __init__.py
    └── generate_test_fixtures.py
```

---

## Appendix B: Quick Reference Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific phase
pytest tests/test_phase1_preprocessing.py -v
pytest tests/test_checkpoint_resume.py -v

# Run specific test
pytest tests/test_phase1_preprocessing.py::test_bimodal_detection -v

# Run with coverage
pytest tests/ --cov=ml_pipeline.stage7_llm_analysis --cov-report=html

# Run only P0 tests (when marked with decorator)
pytest tests/ -v -m p0

# Generate fixtures
python tests/scripts/generate_test_fixtures.py --fixture-set phase1
python tests/scripts/generate_test_fixtures.py --all
```

---

## Document Metadata

**Version**: 2.0
**Created**: 2025-10-21
**Last Updated**: 2025-10-21
**Status**: ACTIVE - Ready for Test Implementation
**Total Tests**: 50 (30 HLD + 20 new)
**Estimated Total Time**: ~16 hours
**Next CLI Instance**: Start with fixture generation + HLD-01 (Section 3.1.1)

---

## 🎯 IMMEDIATE NEXT STEPS (For Next CLI Instance)

### Step 1: Generate Test Fixtures (One-Time Setup)
```bash
cd /home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/tests
python scripts/generate_test_fixtures.py --all --skip-live-api
```

**Expected Output**:
```
========================================
Generating Fixture Set 1: Phase 1 Preprocessing
========================================
✓ Created directories
✓ Created: rf_data.json (Size: 12 KB)
✓ Created: kmeans_data.json (Size: 45 KB)
...
✅ Fixture Set 1 Complete! (~500 KB)

========================================
All Fixtures Generated Successfully!
========================================
Total Size: ~5 MB
Total Files: 27
```

**After Generation**:
- Update Section 13.3 (Fixture Update Log) with timestamp
- Commit fixture files to repo

---

### Step 2: Implement First Test (HLD-01)

**File to Create**: `tests/test_phase1_preprocessing.py`

**Test Specification**: Section 3.1.1 (jump to that section)

**Test Code** (from Section 3.1.1):
```python
def test_bimodal_detection():
    """Test: 30%/30% split detected as bimodal."""
    from ml_pipeline.stage7_llm_analysis import detect_bimodal_pattern

    distribution = {
        'top_performers': {
            'high_percentage': 0.40,
            'low_percentage': 0.35
        }
    }

    result = detect_bimodal_pattern(distribution)

    assert result['is_bimodal'] == True
    assert result['pattern_label'] == "BIMODAL"
    assert "BOTH strategies work" in result['interpretation']

    print("✓ Bimodal detection passed: 40%/35% split")
```

**Run Test**:
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
PYTHONPATH=/home/jorge/rumiaifinal pytest ml_pipeline/stage7_llm_analysis/tests/test_phase1_preprocessing.py::test_bimodal_detection -v
```

---

### Step 3: Update Stage7Testsv2.md

After running test, update these sections:

**Section 0.2** (Test Status Summary):
```
Phase 0: HLD Tests (Baseline)
[#---------] 1/30 (3%)   ← Update this

Current Session Info:
- CLI Instance: #1                    ← Keep or increment
- Started: 2025-10-21                 ← Current date
- Working On: HLD-02                  ← Next test
- Blockers: None
```

**Section 0.4** (Files Modified This Session):
```
- tests/test_phase1_preprocessing.py (created: 2025-10-21)
- tests/fixtures/ (generated: 2025-10-21)
- Stage7Testsv2.md (updated: 2025-10-21)
```

**Section 2.2** (Phase 0 Checklist):
```
- [x] **HLD-01**: Bimodal detection - 30%/30% split
  - Status: ✅ PASSED
  - Last Updated: 2025-10-21
```

**Section 11** (Test Execution Log):
Add new run entry (see template in Section 11)

---

### Step 4: Commit Changes

```bash
git add tests/
git add documentation_migration/FutureDevelopments/Stage7Testsv2.md
git commit -m "Test: HLD-01 implemented and passing

- Generated test fixtures (Sets 1-7, ~5 MB)
- Implemented test_bimodal_detection
- Updated Stage7Testsv2.md progress tracking

Progress: 1/50 tests (2%)"
```

---

### Step 5: Continue to Next Test

- Next test: **HLD-02** (Section 3.1.2)
- File: `tests/test_phase1_preprocessing.py` (same file)
- Complexity: Simple
- Est. Time: 10 minutes

---

## 📋 Quick Navigation

- **Resume Instructions**: Section 0.1
- **Test Status**: Section 0.2
- **Find Next Test**: Section 0.3
- **Test Checklist**: Section 2
- **Test Specifications**: Sections 3-6
- **Bug Log**: Section 10
- **Execution Log**: Section 11
- **Fixtures**: Section 13

---

**End of Document**
