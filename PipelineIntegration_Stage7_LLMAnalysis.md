# Pipeline Integration: Stage 7 LLM Analysis

> **Stage**: Stage 7: LLM Analysis - Hybrid Two-Phase Approach
> **TI Document**: LLMAnalysisCHILDTI.md
> **Integration Date**: 2025-10-22
> **Status**: Pending Review

---

## 1. Integration Overview

**Stage Position**:
```
Stage 6: ML Analysis Generation
    ↓
Stage 7: LLM Analysis - Hybrid Two-Phase Approach ← NEW
    ↓
Stage 8: Report Delivery (Future)
```

**Metadata**:
```yaml
Stage_Name: Stage 7: LLM Analysis - Hybrid Two-Phase Approach
Pipeline_Position: After Stage 6, Before Stage 8 (Future)
Integration_Type: New Stage
Breaking_Changes: No
Backward_Compatible: Yes
Implementation_Priority: HIGH
```

**Auto-Detected Patterns from rumiai_ml_batch.py**:
```
✓ BASE_PATH_VAR = bucket_path (from analysis_base / f"buckets/bucket_{bucket_name}")
✓ BUCKET_ITERATION = for bucket_name in winning_buckets
✓ CONFIG_ACCESS = cli_args.target (hashtag from CLI)
✓ UPSTREAM_PATH_PATTERN = bucket_path / "{file}"
✓ STAGE_IDENTIFIER = "Stage {N}"
✓ Entry Function: main() from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis

Note: Stage 7 uses internal checkpoint (.phase1_status.json), not external checkpoint file
```

---

## 2. rumiai_ml_batch.py Modifications

### 2.1 Import Statement

**Location**: Top of rumiai_ml_batch.py, after Stage 6 import (after `from ml_pipeline.stage6_analysis` import)

**Action**: ADD

```python
# Stage 7: LLM Analysis - Hybrid Two-Phase Approach
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_llm_analysis_main
from config.bucket_definitions import BUCKET_WINDOWS  # Already imported at top (line 65)
# Source: Actual implementation in /home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py:536
```

---

### 2.2 Pipeline Orchestration

**Location**: `main()` function, after Stage 6 block (after Stage 6 completion message)

**Action**: ADD

```python
    # ===== Stage 7: LLM Analysis - Hybrid Two-Phase Approach =====
    logger.info("Starting Stage 7: LLM Analysis")

    # Source: Pattern from Stage 3-6 (analysis_base and winning_buckets already defined)
    for bucket_name in winning_buckets:
        logger.info(f"Starting Stage 7 for bucket: {bucket_name}")
        bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

        # Stage 7 does not use traditional checkpoints - it has internal .phase1_status.json tracking
        # Check if Stage 7 outputs already exist
        llm_output_dir = bucket_path / "ml_analysis/llm"
        complete_analysis_file = llm_output_dir / "complete_analysis.json"

        if complete_analysis_file.exists():
            logger.info(f"✓ Stage 7 already completed for bucket {bucket_name} (complete_analysis.json found)")
            continue

        try:
            # Validate prerequisites (Stage 6 outputs must exist)
            validate_stage7_prerequisites(str(bucket_path), bucket_name)

            # Execute Stage 7 LLM Analysis
            # Source: LLMAnalysisCHILDTI.md Section 2 StageInput
            stage7_llm_analysis_main(
                bucket_path=str(bucket_path),  # Convert Path to str for Stage 7 API
                bucket=bucket_name,             # e.g., "18-33s"
                hashtag=cli_args.target         # From CLI args (e.g., "#nutrition")
            )

            # Validate outputs (Phase 1 + Phase 2 outputs)
            validate_stage7_outputs(str(bucket_path), bucket_name)

            logger.info(f"✓ Stage 7 complete for bucket {bucket_name}: LLM Analysis")

        except FileNotFoundError as e:
            logger.error(f"✗ Stage 7 failed for bucket {bucket_name}: Missing prerequisite - {e}")
            logger.error("Action: Ensure Stage 6 completed successfully and generated all ML analysis JSONs")
            raise
        except ValueError as e:
            logger.error(f"✗ Stage 7 failed for bucket {bucket_name}: Invalid input - {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Stage 7 failed for bucket {bucket_name}: {e}")
            handle_stage7_error(e, str(bucket_path))
            raise

    print("\n✓ Stage 7: LLM Analysis - COMPLETE")
```

**Key Integration Points**:
- Entry function: `stage7_llm_analysis_main()` (actual implementation, not theoretical)
- Parameters match TI Section 2 StageInput exactly:
  - `bucket_path`: str - Path to bucket directory
  - `bucket`: str - Duration bucket identifier (e.g., "18-33s")
  - `hashtag`: Optional[str] - Hashtag context for creative insights
- Stage 7 uses internal checkpoint (`.phase1_status.json`), not external
- Output validation checks `complete_analysis.json` existence

---

### 2.3 CLI Arguments

**No stage-specific CLI parameters required**

Stage 7 uses existing Foundation parameters:
- `--client` (inherited from Foundation)
- `--target` (inherited from Foundation, passed as `hashtag`)
- No new CLI args needed per TI Section 9 CONFIG_SCHEMA

---

## 3. Helper Functions

### 3.1 Prerequisite Validation

**Location**: New function in rumiai_ml_batch.py (before `run_pipeline()`)

**Action**: ADD

```python
def validate_stage7_prerequisites(bucket_path: str, bucket: str) -> None:
    """
    Validate Stage 7 input dependencies exist.
    Source: LLMAnalysisCHILDTI.md Section 12.2 (Upstream TI Requirements)

    Stage 7 requires Stage 6 outputs:
    - rf_video_analysis.json (video-level RF features)
    - {window}_rf_analysis.json (per window RF features)
    - {window}_kmeans_analysis.json (per window K-Means clusters)

    Raises:
        FileNotFoundError: If required upstream output missing
    """
    ml_analysis_dir = os.path.join(bucket_path, "ml_analysis")

    # BUCKET_WINDOWS already imported at top (Section 2.1)
    window_types = BUCKET_WINDOWS.get(bucket, [])

    if not window_types:
        raise ValueError(f"Invalid bucket: {bucket}")

    required_files = [
        # Video-level RF (cross-window features)
        os.path.join(ml_analysis_dir, "rf_video_analysis.json"),
    ]

    # Window-level files (RF + K-Means for each window)
    for window in window_types:
        required_files.append(os.path.join(ml_analysis_dir, f"{window}_rf_analysis.json"))
        required_files.append(os.path.join(ml_analysis_dir, f"{window}_kmeans_analysis.json"))

    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        raise FileNotFoundError(
            f"Stage 7 prerequisites missing ({len(missing)} files):\n" +
            "\n".join(f"  - {os.path.basename(f)}" for f in missing) +
            f"\n\nAction: Ensure Stage 6 (ML Analysis Generation) completed successfully for bucket {bucket}"
        )

    logger.info(f"✓ Stage 7 prerequisites validated for bucket {bucket} ({len(required_files)} files)")
```

**Source**: TI Section 12.2 UPSTREAM_OUTPUTS_REQUIRED (Stage 6 dependencies)

---

### 3.2 Output Validation

**Location**: New function in rumiai_ml_batch.py

**Action**: ADD

```python
def validate_stage7_outputs(bucket_path: str, bucket: str) -> None:
    """
    Validate Stage 7 outputs created correctly.
    Source: LLMAnalysisCHILDTI.md Section 3 (Stage Contract - StageOutput)

    Validates:
    - Phase 1 window analyses (hook, middle_X, closing)
    - Phase 2 synthesis (cross-window insights)
    - Complete analysis (combined Phase 1 + Phase 2)

    Raises:
        AssertionError: If output validation fails
        FileNotFoundError: If required output missing
    """
    llm_output_dir = os.path.join(bucket_path, "ml_analysis/llm")

    # BUCKET_WINDOWS already imported at top (Section 2.1)
    window_types = BUCKET_WINDOWS.get(bucket, [])

    if not window_types:
        raise ValueError(f"Invalid bucket: {bucket}")

    # Validate Phase 1 outputs (one per window)
    for window in window_types:
        window_file = os.path.join(llm_output_dir, f"{window}_analysis.json")

        assert os.path.exists(window_file), \
            f"Stage 7 Phase 1 output missing: {window}_analysis.json"

        # Validate JSON structure
        with open(window_file, 'r') as f:
            data = json.load(f)
            assert 'window_type' in data, f"{window}_analysis.json missing 'window_type' field"
            assert 'clusters' in data, f"{window}_analysis.json missing 'clusters' field"
            assert len(data['clusters']) == 3, f"{window}_analysis.json must have exactly 3 clusters"

    # Validate Phase 2 synthesis output
    synthesis_file = os.path.join(llm_output_dir, "synthesis.json")
    assert os.path.exists(synthesis_file), \
        "Stage 7 Phase 2 output missing: synthesis.json"

    with open(synthesis_file, 'r') as f:
        synthesis = json.load(f)
        assert 'winning_formulas' in synthesis, "synthesis.json missing 'winning_formulas'"
        assert 'scenario' in synthesis, "synthesis.json missing 'scenario' field"

    # Validate complete analysis (combined output)
    complete_file = os.path.join(llm_output_dir, "complete_analysis.json")
    assert os.path.exists(complete_file), \
        "Stage 7 complete analysis output missing: complete_analysis.json"

    with open(complete_file, 'r') as f:
        complete = json.load(f)
        assert 'phase1_window_analyses' in complete
        assert 'phase2_synthesis' in complete
        assert 'bucket' in complete
        assert complete['bucket'] == bucket

    logger.info(f"✓ Stage 7 outputs validated for bucket {bucket} (Phase 1: {len(window_types)} windows, Phase 2: synthesis, Complete: 1 file)")
```

**Source**: TI Section 3 StageOutput validation rules

---

### 3.3 Error Handling

**Location**: New function in rumiai_ml_batch.py

**Action**: ADD

```python
def handle_stage7_error(error: Exception, bucket_path: str) -> None:
    """
    Handle Stage 7 errors.
    Source: LLMAnalysisCHILDTI.md Section 6 (Error Handling)

    Stage 7 has 3 main error categories:
    - LLMValidationError: Invalid LLM response (wrong schema, missing fields)
    - Phase1ExecutionError: Phase 1 window analysis failure
    - InsufficientDataError: <3 cluster paths meet 10% threshold (fallback to feature-based reports)

    Note: Stage 7 raises exceptions with error codes in message strings.
    This error matching pattern is specific to current Stage 7 implementation.
    """
    # LLM Validation Errors (malformed API responses)
    if "Missing 'clusters' key" in str(error) or "Expected 3 clusters" in str(error):
        logger.error("Stage 7 Error: LLM response validation failed")
        logger.error("Issue: Claude API returned malformed response (wrong schema)")
        logger.error("Action: Check LLM prompt construction and API response parsing")
        logger.error("Retry Policy: Automatic retry with exponential backoff [0s, 2s, 4s]")

    # Phase 1 Execution Errors (window analysis failures)
    elif "Phase1ExecutionError" in str(type(error).__name__):
        logger.error("Stage 7 Error: Phase 1 window analysis failed")
        logger.error(f"Details: {error}")
        logger.error("Action: Check .phase1_status.json for partial progress, resume from checkpoint")

    # Insufficient Data Errors (fallback scenario)
    elif "InsufficientDataError" in str(error):
        logger.error("Stage 7 Error: Insufficient cluster paths (<3 paths meet 10% threshold)")
        logger.error("Issue: Not enough common viral patterns detected")
        logger.error("Action: System will automatically use feature-based fallback reports")
        logger.info("Note: This is not a failure - fallback strategy will generate insights from RF features")

    # API Errors (authentication, rate limits, timeouts)
    elif "401" in str(error):
        logger.error("Stage 7 Error: Claude API authentication failed")
        logger.error("Issue: ANTHROPIC_API_KEY invalid or missing")
        logger.error("Action: Verify ANTHROPIC_API_KEY in .env file")
        logger.error("Retry Policy: NO RETRY (non-retryable error)")

    elif "429" in str(error) or "503" in str(error):
        logger.error(f"Stage 7 Error: Claude API {error}")
        logger.error("Issue: Rate limit exceeded or service unavailable")
        logger.error("Retry Policy: Automatic retry with exponential backoff [0s, 2s, 4s]")

    else:
        logger.error(f"Stage 7 Error: Unexpected error - {error}")
        logger.error("Action: Check logs for full traceback")

    # Cleanup partial outputs (only if catastrophic failure)
    if not isinstance(error, (FileNotFoundError, ValueError)):
        cleanup_stage7_partial_outputs(bucket_path)


def cleanup_stage7_partial_outputs(bucket_path: str) -> None:
    """
    Remove partial outputs from failed Stage 7 execution.
    Source: LLMAnalysisCHILDTI.md Section 3 StageOutput

    Note: Stage 7 has checkpoint/resume capability (.phase1_status.json).
    Only cleanup if catastrophic failure (not for recoverable errors).
    """
    llm_output_dir = os.path.join(bucket_path, "ml_analysis/llm")

    if not os.path.exists(llm_output_dir):
        return

    # List of output files to clean up
    partial_files = [
        # Phase 1 outputs (variable count based on bucket)
        # Phase 2 outputs
        os.path.join(llm_output_dir, "synthesis.json"),
        # Complete analysis
        os.path.join(llm_output_dir, "complete_analysis.json"),
    ]

    # Add window-specific files (dynamically based on bucket)
    for file in os.listdir(llm_output_dir):
        if file.endswith("_analysis.json") and file != "complete_analysis.json":
            partial_files.append(os.path.join(llm_output_dir, file))

    removed_count = 0
    for f in partial_files:
        if os.path.exists(f):
            os.remove(f)
            logger.info(f"Cleaned up partial output: {os.path.basename(f)}")
            removed_count += 1

    # Keep .phase1_status.json for resume capability
    # Do NOT delete: .phase1_status.json (checkpoint file)

    if removed_count > 0:
        logger.info(f"✓ Cleaned up {removed_count} partial outputs (kept .phase1_status.json for resume)")
```

**Source**: TI Section 6 ERROR_CONDITIONS dict

**Key Error Types** (from TI):
- `LLMValidationError`: LLM response validation failure
- `Phase1ExecutionError`: Phase 1 window analysis failure
- `InsufficientDataError`: <3 cluster paths (uses fallback)
- API errors: 401 (auth), 429 (rate limit), 503 (unavailable)

---

## 4. Checkpoint Schema

**Note**: Stage 7 uses internal checkpoint system, not external checkpoint file.

**Internal Checkpoint**: `.phase1_status.json` (managed by Stage 7 implementation)

```python
# Internal checkpoint schema for Stage 7 (managed by stage7_llm_analysis.py)
# Location: bucket_path/ml_analysis/llm/.phase1_status.json

Phase1StatusSchema = {
    "phase1_complete": bool,           # True if all windows analyzed
    "total_windows": int,              # Total windows for this bucket
    "completed_windows": list[str],    # ["hook", "middle_1", ...]
    "failed_windows": list[dict],      # [{window: str, error: str, timestamp: str}]
    "started_at": str,                 # ISO 8601 timestamp
    "last_updated": str,               # ISO 8601 timestamp
    "completed_at": str | None         # ISO 8601 timestamp or null
}
```

**External Checkpoint**: Not used (Stage 7 manages its own resume logic)

**Integration**: rumiai_ml_batch.py checks for `complete_analysis.json` existence to determine if Stage 7 completed

---

## 5. Integration Tests

**File**: `tests/integration/test_pipeline_stage7.py`

**Action**: CREATE

**Note**: Tests require Anthropic API mocking. See `ml_pipeline/stage7_llm_analysis/tests/test_parallel_execution.py` for mocking pattern using `unittest.mock.patch`.

```python
import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, Mock
from rumiai_ml_batch import main  # Entry point from rumiai_ml_batch.py

def test_stage7_happy_path(tmp_path):
    """
    Test Stage 7 integration (happy path).
    Source: LLMAnalysisCHILDTI.md Section 7 (Complete Example Traces)

    Validates:
    - Stage 7 processes all windows (hook, middle_X, closing)
    - Phase 1 outputs created for each window
    - Phase 2 synthesis created
    - Complete analysis JSON created
    """
    # Setup test data (Stage 6 outputs)
    bucket_path = tmp_path / "buckets/bucket_18-33s"
    ml_analysis_dir = bucket_path / "ml_analysis"
    ml_analysis_dir.mkdir(parents=True)

    # Create mock Stage 6 outputs
    # rf_video_analysis.json
    with open(ml_analysis_dir / "rf_video_analysis.json", 'w') as f:
        json.dump({
            "total_videos": 100,
            "top_performer_count": 80,
            "features": [{"feature": "energy_level", "importance": 0.35}]
        }, f)

    # Window-level files (18-33s bucket: hook, middle_1-4, closing)
    windows = ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"]
    for window in windows:
        # RF analysis (complete schema required per TI Section 5)
        with open(ml_analysis_dir / f"{window}_rf_analysis.json", 'w') as f:
            json.dump({
                "feature_importance": [
                    {
                        "feature": "eye_contact_rate",
                        "importance": 0.35,
                        "rank": 1,
                        "top_performer_avg": 0.88,
                        "bottom_performer_avg": 0.45,
                        "gap": 0.43,
                        "distribution": {
                            "top_performers": {
                                "high_percentage": 0.75,
                                "low_percentage": 0.15
                            }
                        }
                    }
                ]
            }, f)

        # K-Means analysis
        with open(ml_analysis_dir / f"{window}_kmeans_analysis.json", 'w') as f:
            json.dump({
                "n_clusters": 3,
                "clusters": [
                    {"cluster_id": i, "size": 30, "centroid": {}}
                    for i in range(3)
                ]
            }, f)

    # Configure mock ANTHROPIC_API_KEY
    os.environ["ANTHROPIC_API_KEY"] = "test-key-12345"

    # Mock CLI args for main()
    import sys
    sys.argv = [
        "rumiai_ml_batch.py",
        "--client", "test_client",
        "--target", "#nutrition"
    ]

    # Mock Anthropic API to prevent real API calls
    mock_llm_response = {
        'clusters': [
            {'cluster_id': i, 'name': f'Cluster {i}', 'defining_features': []}
            for i in range(3)
        ]
    }

    with patch('ml_pipeline.stage7_llm_analysis.stage7_llm_analysis.Anthropic') as mock_anthropic:
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.content = [Mock(text=json.dumps(mock_llm_response))]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            # Execute pipeline (mocked to run Stage 7 only)
            # Note: In production, modify main() to support stage filtering
            main()  # Runs full pipeline

    # Validate outputs
    llm_output_dir = bucket_path / "ml_analysis/llm"

    # Phase 1 outputs (one per window)
    for window in windows:
        output_file = llm_output_dir / f"{window}_analysis.json"
        assert output_file.exists(), f"Missing Phase 1 output: {window}_analysis.json"

        with open(output_file) as f:
            data = json.load(f)
            assert data['window_type'] == window
            assert len(data['clusters']) == 3

    # Phase 2 synthesis
    synthesis_file = llm_output_dir / "synthesis.json"
    assert synthesis_file.exists(), "Missing Phase 2 output: synthesis.json"

    with open(synthesis_file) as f:
        synthesis = json.load(f)
        assert 'winning_formulas' in synthesis
        assert 'scenario' in synthesis

    # Complete analysis
    complete_file = llm_output_dir / "complete_analysis.json"
    assert complete_file.exists(), "Missing complete analysis"

    with open(complete_file) as f:
        complete = json.load(f)
        assert complete['bucket'] == "18-33s"
        assert 'phase1_window_analyses' in complete
        assert 'phase2_synthesis' in complete


def test_stage7_checkpoint_resume(tmp_path):
    """
    Test Stage 7 resume capability (internal .phase1_status.json).

    Scenario:
    - Phase 1 partially complete (3/6 windows done)
    - Re-run Stage 7
    - Verify: Resumes from checkpoint, skips completed windows
    """
    # Create partial Phase 1 status
    bucket_path = tmp_path / "buckets/bucket_18-33s"
    llm_output_dir = bucket_path / "ml_analysis/llm"
    llm_output_dir.mkdir(parents=True)
    status_file = llm_output_dir / ".phase1_status.json"

    with open(status_file, 'w') as f:
        json.dump({
            "phase1_complete": False,
            "total_windows": 6,
            "completed_windows": ["hook", "middle_1", "middle_2"],
            "failed_windows": [],
            "started_at": "2025-10-22T10:00:00",
            "last_updated": "2025-10-22T10:05:00",
            "completed_at": None
        }, f)

    # Create completed window outputs
    for window in ["hook", "middle_1", "middle_2"]:
        with open(llm_output_dir / f"{window}_analysis.json", 'w') as f:
            json.dump({"window_type": window, "clusters": []}, f)

    # Re-run Stage 7 (with mocking)
    with patch('ml_pipeline.stage7_llm_analysis.stage7_llm_analysis.Anthropic'):
        # Call stage7_llm_analysis_main directly for unit test
        from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_main
        stage7_main(bucket_path=str(bucket_path), bucket="18-33s", hashtag="#nutrition")

    # Verify: Only remaining 3 windows processed
    assert (llm_output_dir / "middle_3_analysis.json").exists()
    assert (llm_output_dir / "middle_4_analysis.json").exists()
    assert (llm_output_dir / "closing_analysis.json").exists()


def test_stage7_error_handling_missing_prerequisites(tmp_path):
    """
    Test Stage 7 error handling (missing Stage 6 outputs).
    Source: LLMAnalysisCHILDTI.md Section 7 Trace 3 (Error Scenario)
    """
    # Setup: Missing rf_video_analysis.json
    bucket_path = tmp_path / "buckets/bucket_18-33s"
    ml_analysis_dir = bucket_path / "ml_analysis"
    ml_analysis_dir.mkdir(parents=True)

    # No Stage 6 outputs created

    # Should raise FileNotFoundError when validating prerequisites
    from rumiai_ml_batch import validate_stage7_prerequisites

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_stage7_prerequisites(str(bucket_path), "18-33s")

    # Verify error message
    assert "Stage 7 prerequisites missing" in str(exc_info.value)
    assert "rf_video_analysis.json" in str(exc_info.value)


def test_stage7_skip_when_complete():
    """
    Test Stage 7 skips execution when complete_analysis.json exists.
    """
    # Setup: Create complete_analysis.json (indicates Stage 7 done)
    llm_output_dir = bucket_path / "ml_analysis/llm"
    llm_output_dir.mkdir(parents=True)

    with open(llm_output_dir / "complete_analysis.json", 'w') as f:
        json.dump({"bucket": "18-33s", "status": "complete"}, f)

    # Run pipeline
    run_pipeline(config, start_stage=7, stop_at_stage=7)

    # Verify: Stage 7 skipped (log message)
    assert "Stage 7 already completed" in captured_logs
```

**Test Coverage**:
- test_happy_path: Full Stage 7 execution (Phase 1 + Phase 2)
- test_checkpoint_resume: Internal checkpoint resume capability
- test_error_handling: Missing prerequisites validation
- test_skip_when_complete: Skip logic when outputs exist

---

## 6. Documentation Updates

### 6.1 Pipeline Diagram Update

**BEFORE**:
```
Stage 6: ML Analysis Generation
    ↓
Stage 8: Report Delivery (Future)
```

**AFTER**:
```
Stage 6: ML Analysis Generation
    ↓
Stage 7: LLM Analysis - Hybrid Two-Phase Approach ← NEW
    ↓
Stage 8: Report Delivery (Future)
```

**Update Location**: README.md, MLPlanningv2.md Section 2 (Pipeline Architecture)

---

### 6.2 CHANGELOG Entry

```markdown
## [Unreleased]

### Added
- Stage 7: LLM Analysis - Hybrid Two-Phase Approach
  - Entry point: `stage7_llm_analysis_main()` (from `ml_pipeline.stage7_llm_analysis.stage7_llm_analysis.main`)
  - Outputs:
    - Phase 1: Window-level cluster analyses (hook, middle_X, closing)
    - Phase 2: Cross-window synthesis with winning formulas
    - Complete: Combined Phase 1 + Phase 2 analysis
  - CLI parameters: None (uses Foundation `--client` and `--target`)
  - Features:
    - Hybrid two-phase approach (parallel window analysis + cross-window synthesis)
    - Internal checkpoint/resume (.phase1_status.json)
    - Exponential backoff retry for LLM API (0s, 2s, 4s)
    - Fallback to feature-based reports when <3 cluster paths exist
  - Cost: ~$4.18 per client (6-7 windows × $0.60 average per window)
```

**Constraints**:
- Brief description: "Hybrid two-phase LLM analysis generating creative insights from ML patterns"
- Max 80 characters for title

---

## 7. Pre-Merge Validation Checklist

**Before merging Stage 7 integration:**

### Code Modifications
- [ ] rumiai_ml_batch.py import added (Section 2.1)
- [ ] Pipeline orchestration code added (Section 2.2)
- [ ] CLI arguments: None required (confirmed)
- [ ] Helper functions added (Section 3.1-3.3):
  - [ ] `validate_stage7_prerequisites()`
  - [ ] `validate_stage7_outputs()`
  - [ ] `handle_stage7_error()`
  - [ ] `cleanup_stage7_partial_outputs()`

### Testing
- [ ] Integration tests pass (Section 5):
  - [ ] `test_stage7_happy_path`
  - [ ] `test_stage7_checkpoint_resume`
  - [ ] `test_stage7_error_handling_missing_prerequisites`
  - [ ] `test_stage7_skip_when_complete`
- [ ] Manual testing completed:
  - [ ] Run full pipeline with Stage 7 on real data (test client + hashtag)
  - [ ] Verify outputs match TI Section 3 StageOutput
  - [ ] Test error scenario: missing Stage 6 outputs
  - [ ] Test resume scenario: partial Phase 1 completion
  - [ ] Verify cost tracking: ~$4.18 per client for 6-window bucket

### Documentation
- [ ] Pipeline diagram updated (Section 6.1)
- [ ] CHANGELOG entry created (Section 6.2)
- [ ] README.md updated with Stage 7 description

### Code Review
- [ ] All code derived from TI document (no hallucinated features)
- [ ] Entry function name: `main()` from `ml_pipeline.stage7_llm_analysis.stage7_llm_analysis` (verified)
- [ ] Entry function is from actual implementation (line 536), not theoretical
- [ ] Error messages match TI Section 6 error categories
- [ ] No modifications to existing stages (integration is additive)
- [ ] ANTHROPIC_API_KEY environment variable required (verified in .env)

### Cost Management
- [ ] Verify ANTHROPIC_API_KEY configured in .env
- [ ] Cost tracking logged per window (~$0.60 average)
- [ ] Budget alert threshold: >$8.00 per client triggers warning
- [ ] Checkpoint/resume prevents duplicate API costs

---

## 8. Rollback Plan

**If Stage 7 integration causes issues:**

### Step 1: Revert rumiai_ml_batch.py
```bash
git revert <commit_hash>
```

### Step 2: Remove Stage 7 outputs
```bash
# Option A: Full cleanup (deletes checkpoint - partial work not recoverable)
find /data/clients -type d -name "llm" -exec rm -rf {} +

# Option B: Preserve checkpoint for resume (recommended for partial failures)
# Remove outputs but keep .phase1_status.json
find /data/clients -type d -name "llm" -exec sh -c '
    for dir; do
        find "$dir" -type f ! -name ".phase1_status.json" -delete
    done
' sh {} +

# Or for specific client/hashtag:
# Full cleanup:
rm -rf /data/clients/{client_id}/hashtags/{target}/*/buckets/*/ml_analysis/llm/

# Preserve checkpoint:
find /data/clients/{client_id}/hashtags/{target}/*/buckets/*/ml_analysis/llm/ \
    -type f ! -name ".phase1_status.json" -delete
```

**Recommendation**: Use Option B to preserve partial Phase 1 work if rolling back due to Phase 2 issues.

### Step 3: Document issues
- Create bug report with logs
- Note which test failed (happy path, error handling, resume)
- Identify root cause:
  - Integration issue (rumiai_ml_batch.py changes)
  - TI spec issue (LLMAnalysisCHILDTI.md)
  - Implementation issue (stage7_llm_analysis.py)
  - API issue (Claude API errors)

### Step 4: Verify rollback
```bash
# Verify Stage 6 still works
python rumiai_ml_batch.py --client test_client --target "#test" --stop-stage 6

# Verify no Stage 7 artifacts remain
find /data/clients -name ".phase1_status.json" -o -name "complete_analysis.json"
```

---

## 9. Integration History

| Date | Action | Author | Notes |
|------|--------|--------|-------|
| 2025-10-22 | Created | Claude Code | From LLMAnalysisCHILDTI.md, auto-detected patterns from rumiai_ml_batch.py |
| | Reviewed | | Pending |
| | Merged | | Pending |

---

## 10. References

**Source Documents**:
- **TI Document**: LLMAnalysisCHILDTI.md (all code specifications)
- **HLD Document**: LLMAnalysisCHILD.md (high-level design)
- **Mother HLD**: MLPlanningv2.md Stage 7 (lines 1847-2023)
- **Foundation**: FoundationCHILD.md (directory structure, CLI params)
- **Pipeline**: rumiai_ml_batch.py (current version before integration)

**Traceability**:
- Section 2.1-2.2: Actual implementation (`ml_pipeline.stage7_llm_analysis.stage7_llm_analysis.main:536`)
- Section 2.2: TI Section 2 StageInput (bucket_path, bucket, hashtag)
- Section 3.1: TI Section 12.2 (Stage 6 upstream requirements)
- Section 3.2: TI Section 3 StageOutput (Phase 1, Phase 2, Complete outputs)
- Section 3.3: TI Section 6 ERROR_CONDITIONS (LLMValidationError, Phase1ExecutionError, etc.)
- Section 4: Internal checkpoint (.phase1_status.json, TI Section 4.10)
- Section 5: TI Section 7 (Complete Example Traces)

**Auto-Detected Patterns** (from rumiai_ml_batch.py):
- BASE_PATH_VAR: `bucket_path`
- Entry function: `main()` from line 536 of stage7_llm_analysis.py
- Stage pattern: "Stage {N}: {Name}"
- Logging pattern: `logger.info(f"Starting Stage {N}: {Name} for bucket {bucket_name}")`
- Bucket iteration: `for bucket_name in sorted(active_buckets):`
- Output check: File existence (not checkpoint JSON)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-22
**Status**: Pending Review

---

## Appendix A: Entry Point Verification

**Step 0.4 Entry Point Verification - COMPLETED**

```
=== ENTRY POINT VERIFICATION ===

Step 1: Check TI Section 11.4 Implementation Log
  ✗ Section 11.4 is EMPTY (implementation not yet logged in TI)
  → Proceed to Step 2

Step 2: Check Actual Implementation Files
  ✓ Found implementation: /home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py
  ✓ Entry function: main() (line 536)
  ✓ Signature: def main(bucket_path: str, bucket: str, hashtag: Optional[str] = None):
  → Implementation exists, use actual function

Step 3: Verify NOT using Section 8.3 theoretical examples
  ✓ Entry function is from actual implementation (stage7_llm_analysis.py:536)
  ✓ NOT from theoretical/placeholder code
  ✓ NOT from commented examples

FINAL EXTRACTION:
  Source: Actual implementation file
  Entry Function: main
  Module Path: ml_pipeline.stage7_llm_analysis.stage7_llm_analysis
  Import Statement: from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_llm_analysis_main

  Verified: Actual implementation, not theoretical
  Status: ✅ VERIFIED
```

---

## Appendix B: Cost Management Integration

**Stage 7 Cost Tracking** (per TI Section 9.9):

```python
# Add to rumiai_ml_batch.py after Stage 7 execution

# Track Stage 7 costs (optional enhancement)
def log_stage7_costs(bucket: str, window_count: int):
    """
    Log Stage 7 API costs for monitoring.
    Source: LLMAnalysisCHILDTI.md Section 9.9 Cost Management
    """
    # Average cost per window: $0.60 (Phase 1)
    # Phase 2 cost: $0.82 (synthesis)
    phase1_cost = window_count * 0.60
    phase2_cost = 0.82
    total_cost = phase1_cost + phase2_cost

    logger.info(f"Stage 7 Cost Estimate for bucket {bucket}:")
    logger.info(f"  Phase 1: {window_count} windows × $0.60 = ${phase1_cost:.2f}")
    logger.info(f"  Phase 2: 1 synthesis × $0.82 = ${phase2_cost:.2f}")
    logger.info(f"  Total: ${total_cost:.2f}")

    # Budget alert (per TI Section 9.9.2)
    MAX_COST_PER_BUCKET = 5.00  # Allows 6-window buckets (18-33s costs ~$4.18)
    if total_cost > MAX_COST_PER_BUCKET:
        logger.warning(f"⚠️ Stage 7 cost ${total_cost:.2f} exceeds budget ${MAX_COST_PER_BUCKET:.2f} for bucket {bucket}")

# Usage: Add after stage7_llm_analysis_main() call
log_stage7_costs(bucket_name, len(window_types))
```

**Cost Budget Guardrails** (from TI):
- MAX_COST_PER_BUCKET: $5.00 (allows 6-window buckets like 18-33s @ ~$4.18)
- MAX_COST_PER_CLIENT: $8.00 (full pipeline, all 8 buckets)
- Checkpoint/resume prevents duplicate costs (reuses completed windows)

---

**END OF INTEGRATION DOCUMENT**
