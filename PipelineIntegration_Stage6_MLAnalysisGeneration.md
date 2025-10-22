# Pipeline Integration: Stage 6 ML Analysis Generation

> **Stage**: Stage 6: ML Analysis Generation
> **TI Document**: MLAnalysisGenerationCHILDTI.md
> **Integration Date**: 2025-10-21
> **Status**: Pending Review

---

## 1. Integration Overview

**Stage Position**:
```
Stage 5: ML Model Training
    ↓
Stage 6: ML Analysis Generation ← NEW
    ↓
Stage 7: LLM Report Generation (TODO)
```

**Metadata**:
```yaml
Stage_Name: Stage 6: ML Analysis Generation
Pipeline_Position: After Stage 5, Before Stage 7
Integration_Type: New Stage
Breaking_Changes: No
Backward_Compatible: Yes
Implementation_Priority: HIGH
```

**Purpose**: Extract insights from trained ML models and generate structured JSON files for LLM consumption.

**Auto-Detected Patterns** (from rumiai_ml_batch.py):
```
✓ BASE_PATH_VAR = bucket_path
✓ CHECKPOINT_LOCATION = bucket_path / "checkpoints/"
✓ UPSTREAM_PATH_PATTERN = bucket_path / "{file}"
✓ STAGE_IDENTIFIER = "Stage {N}"
✓ Existing stages: 0, 1, 2, 2.5, 2.6, 2.7, 3, 3.4, 4, 5
✓ New stage number: 6
```

---

## 2. rumiai_ml_batch.py Modifications

### 2.1 Import Statement

**Location**: Top of rumiai_ml_batch.py, after Stage 5 import (after line 62)

**Action**: ADD

```python
# Stage 6: ML Analysis Generation
from ml_pipeline.stage6_analysis.ml_analysis_generation import generate_ml_analysis_jsons
from config.bucket_definitions import BUCKET_WINDOWS
```

**Source**: MLAnalysisGenerationCHILDTI.md Section 8.2 (module structure)

**Note**: Verify `from pathlib import Path` already exists in rumiai_ml_batch.py (line 20) - required by helper functions

---

### 2.2 Pipeline Orchestration

**Location**: After Stage 5 block (after line 999), before "FINAL STATUS" section

**Action**: ADD

```python
        # ===== STAGE 6: ML ANALYSIS GENERATION =====
        logger.info("Starting Stage 6: ML Analysis Generation")
        print("\n" + "="*80)
        print("STAGE 6: ML ANALYSIS GENERATION")
        print("="*80)

        # Process each winning bucket
        stage6_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 6 for bucket: {bucket_name}")
            print(f"\n--- Generating ML analysis for bucket: {bucket_name} ---")

            bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

            try:
                # Validate Stage 5 completed successfully
                stage5_checkpoint = bucket_path / "checkpoints" / "stage_5_checkpoint.json"
                if not stage5_checkpoint.exists():
                    logger.error(f"Bucket {bucket_name}: Stage 5 checkpoint missing")
                    print(f"✗ Bucket {bucket_name}: Stage 5 not complete (skipping)")
                    continue

                with open(stage5_checkpoint) as f:
                    stage5_status = json.load(f)

                if stage5_status.get("status") != "completed":
                    logger.error(f"Bucket {bucket_name}: Stage 5 status={stage5_status.get('status')}")
                    print(f"✗ Bucket {bucket_name}: Stage 5 incomplete (skipping)")
                    continue

                # Check if Stage 6 already complete for this bucket
                checkpoint_path = bucket_path / "checkpoints" / "stage_6_checkpoint.json"
                if checkpoint_path.exists():
                    logger.info(f"Bucket {bucket_name}: Stage 6 already complete (checkpoint exists)")
                    print(f"✓ Bucket {bucket_name}: Analysis already complete (skipping)")

                    # Load checkpoint to get output count
                    with open(checkpoint_path) as f:
                        checkpoint = json.load(f)

                    stage6_summaries[bucket_name] = {
                        "json_files_generated": len(checkpoint["output_files"]),
                        "elapsed_time": 0.0  # Skipped, no time
                    }
                    continue

                # Validate prerequisites
                validate_stage_6_prerequisites(str(bucket_path))

                # Get windows configuration for this bucket
                # Source: MLAnalysisGenerationCHILDTI.md Section 2.1 (bucket parameter)
                windows = BUCKET_WINDOWS[bucket_name]

                # Execute Stage 6 analysis generation
                # Source: MLAnalysisGenerationCHILDTI.md Section 8.2 (entry function)
                # Entry point: generate_ml_analysis_jsons(bucket_path, bucket, windows)
                # Returns: exit_code (0 = success, non-zero = failure)
                import time
                start_time = time.time()

                exit_code = generate_ml_analysis_jsons(
                    bucket_path=str(bucket_path),  # TI Section 2.1 StageInput param 1
                    bucket=bucket_name,            # TI Section 2.1 StageInput param 2
                    windows=windows                # TI Section 2.1 StageInput param 3
                )

                elapsed_time = time.time() - start_time

                if exit_code != 0:
                    raise ValueError(f"Stage 6 failed with exit code {exit_code}")

                # Validate outputs (includes JSON count validation)
                validate_stage_6_outputs(str(bucket_path))

                # Count generated JSON files for checkpoint
                # Expected: 1 video RF + N window RF + N window K-Means = 1 + (2 × N)
                ml_analysis_dir = bucket_path / "ml_analysis"
                json_files = list(ml_analysis_dir.glob("*.json"))
                json_count = len(json_files)

                # Create checkpoint
                # Source: rumiai_ml_batch.py Stage 5 pattern (lines 894-909)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                checkpoint_data = {
                    "stage": "stage_6_ml_analysis_generation",
                    "bucket": bucket_name,
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "json_files_generated": json_count,
                    "output_files": [str(f.relative_to(bucket_path)) for f in json_files]
                }

                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                stage6_summaries[bucket_name] = {
                    "json_files_generated": json_count,
                    "elapsed_time": elapsed_time
                }

                logger.info(
                    f"Bucket {bucket_name} complete: "
                    f"{json_count} JSON files generated in {elapsed_time:.1f}s"
                )
                print(
                    f"✓ Bucket {bucket_name}: {json_count} analysis JSONs → "
                    f"{elapsed_time:.1f}s"
                )

            except FileNotFoundError as e:
                # Missing upstream file (Stage 5 models or Stage 4 CSVs)
                # Source: MLAnalysisGenerationCHILDTI.md Section 6 (Error Case 1)
                # Strategy: Skip bucket, continue pipeline
                logger.error(f"Stage 6 prerequisite missing for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Prerequisites missing (skipping)")
                print("   Ensure Stage 5 completed successfully for this bucket")
                print("   Other buckets will continue processing")
                continue

            except ValueError as e:
                # Input validation failed (corrupted PKL, invalid data)
                # Source: MLAnalysisGenerationCHILDTI.md Section 6 (Error Case 2)
                # Strategy: Skip bucket, continue pipeline
                logger.error(f"Stage 6 validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Data validation failed (skipping)")
                print("   This indicates Stage 5 produced invalid models")
                print("   Other buckets will continue processing")
                continue

            except AssertionError as e:
                # Output validation failed (JSON count mismatch, schema error)
                # Source: MLAnalysisGenerationCHILDTI.md Section 6 (Error Case 3)
                # Strategy: Skip bucket, continue pipeline
                logger.error(f"Stage 6 output validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Output validation failed (skipping)")
                print("   This indicates incorrect JSON generation")
                print("   Other buckets will continue processing")
                continue

            except (IOError, OSError) as e:
                # I/O failure (disk full, permission denied)
                # Source: MLAnalysisGenerationCHILDTI.md Section 6 (Error Case 4)
                # Strategy: Exit pipeline (system-wide issue)
                logger.error(f"Stage 6 I/O error for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: I/O error (exiting pipeline)")
                print("   Check disk space and permissions")
                print("   This is a system-wide issue - stopping pipeline")
                return 4  # Exit code 4 = I/O failure

            except Exception as e:
                # Unexpected error
                logger.error(
                    f"Stage 6 unexpected error for bucket {bucket_name}: {e}",
                    exc_info=True
                )
                print(f"✗ Bucket {bucket_name} unexpected error: {e}")
                return 99  # Exit code 99 = unexpected error

        logger.info("Stage 6 completed for all buckets")
        print("\n✓ Stage 6: ML Analysis Generation - COMPLETE")

        # Log Stage 6 summary
        if stage6_summaries:
            total_jsons = sum(s['json_files_generated'] for s in stage6_summaries.values())
            avg_time = sum(s['elapsed_time'] for s in stage6_summaries.values()) / len(stage6_summaries)
            logger.info(
                f"Stage 6 Summary: {total_jsons} JSON files generated "
                f"across {len(stage6_summaries)} buckets (avg {avg_time:.1f}s per bucket)"
            )
            print(
                f"Summary: {total_jsons} analysis JSONs across "
                f"{len(stage6_summaries)} buckets"
            )
        else:
            logger.warning("Stage 6 completed but no buckets were processed")
            print("⚠️  No buckets processed in Stage 6 (all skipped)")
```

**Parameters Extracted from TI Section 2.1 (Stage6Input)**:
- `bucket_path`: str - Base bucket directory path
- `bucket`: str - Bucket name (e.g., "18-33s")
- `windows`: list[str] - Window list from BUCKET_WINDOWS config

**Outputs Listed from TI Section 2.2 (Stage6Output)**:
- `ml_analysis/rf_video_analysis.json` (1 file)
- `ml_analysis/{window}_rf_analysis.json` (N files, where N = window count)
- `ml_analysis/{window}_kmeans_analysis.json` (N files)
- **Total**: 1 + (2 × N) JSON files per bucket

---

### 2.3 CLI Arguments

**Status**: No stage-specific CLI parameters

Stage 6 uses Foundation parameters only (client, bucket, etc.). No new CLI arguments needed.

**Source**: MLAnalysisGenerationCHILDTI.md Section 2.1 - all parameters derived from Foundation

---

## 3. Helper Functions

### 3.1 Prerequisite Validation

**Location**: New function in rumiai_ml_batch.py (before Stage 6 orchestration block)

**Action**: ADD

```python
def validate_stage_6_prerequisites(bucket_path: str) -> None:
    """
    Validate Stage 6 input dependencies exist.
    Source: MLAnalysisGenerationCHILDTI.md Section 3.1 (INPUT FILES)

    Raises:
        FileNotFoundError: If required upstream output missing
    """
    # Get bucket name from path
    bucket_name = Path(bucket_path).name.replace('bucket_', '')

    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket_name]

    bucket_path_obj = Path(bucket_path)

    # Required files from TI Section 2.1 (Stage6Input)
    required_files = []

    # Stage 3 output (1 file)
    required_files.append(bucket_path_obj / "ml_analysis" / "aggregated_features.csv")

    # Stage 4 outputs (1 + 2×N files)
    required_files.append(bucket_path_obj / "ml_analysis" / "rf_transformed.csv")
    for window in windows:
        required_files.append(bucket_path_obj / "ml_analysis" / f"{window}_rf_transformed.csv")
        required_files.append(bucket_path_obj / "ml_analysis" / f"{window}_km_transformed.csv")

    # Stage 5 outputs (2 + 4×N files)
    required_files.append(bucket_path_obj / "models" / f"rf_video_{bucket_name}.pkl")
    required_files.append(bucket_path_obj / "models" / "model_metrics.json")
    for window in windows:
        required_files.append(bucket_path_obj / "models" / f"rf_{window}_{bucket_name}.pkl")
        required_files.append(bucket_path_obj / "models" / f"{window}_kmeans_{bucket_name}.pkl")
        required_files.append(bucket_path_obj / "models" / f"{window}_scalers_{bucket_name}.pkl")
        required_files.append(bucket_path_obj / "models" / f"{window}_X_data_{bucket_name}.pkl")

    missing = [str(f) for f in required_files if not f.exists()]

    if missing:
        raise FileNotFoundError(
            f"Stage 6 prerequisites missing ({len(missing)} files):\n" +
            "\n".join(f"  - {f}" for f in missing[:5]) +  # Show first 5
            (f"\n  ... and {len(missing)-5} more" if len(missing) > 5 else "") +
            f"\n\nAction: Ensure Stages 3, 4, and 5 completed successfully"
        )
```

---

### 3.2 Output Validation

**Location**: New function in rumiai_ml_batch.py

**Action**: ADD

```python
def validate_stage_6_outputs(bucket_path: str) -> None:
    """
    Validate Stage 6 outputs created correctly.
    Source: MLAnalysisGenerationCHILDTI.md Section 5 (validate_stage_output)

    Raises:
        AssertionError: If output validation fails
    """
    # Get bucket name and windows
    bucket_name = Path(bucket_path).name.replace('bucket_', '')
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket_name]

    bucket_path_obj = Path(bucket_path)
    ml_analysis_dir = bucket_path_obj / "ml_analysis"

    # Validate ml_analysis directory exists
    assert ml_analysis_dir.exists(), \
        f"Stage 6 output directory missing: {ml_analysis_dir}"

    # Validate video-level RF JSON
    video_rf_json = ml_analysis_dir / "rf_video_analysis.json"
    assert video_rf_json.exists(), \
        f"Stage 6 video RF JSON missing: {video_rf_json}"

    # Validate video RF JSON structure
    with open(video_rf_json) as f:
        video_rf_data = json.load(f)

    assert "bucket" in video_rf_data, "Video RF JSON missing 'bucket' field"
    assert "feature_importance" in video_rf_data, "Video RF JSON missing 'feature_importance' field"
    assert isinstance(video_rf_data["feature_importance"], list), \
        "Video RF 'feature_importance' must be a list"
    assert len(video_rf_data["feature_importance"]) <= 10, \
        f"Video RF has {len(video_rf_data['feature_importance'])} features (max 10)"

    # Validate window-level JSONs (2 per window)
    for window in windows:
        # Validate window RF JSON
        window_rf_json = ml_analysis_dir / f"{window}_rf_analysis.json"
        assert window_rf_json.exists(), \
            f"Stage 6 window RF JSON missing: {window_rf_json}"

        with open(window_rf_json) as f:
            window_rf_data = json.load(f)

        assert window_rf_data.get("window_type") == window, \
            f"Window RF JSON has wrong window_type: {window_rf_data.get('window_type')} (expected {window})"

        # Validate K-Means JSON
        window_km_json = ml_analysis_dir / f"{window}_kmeans_analysis.json"
        assert window_km_json.exists(), \
            f"Stage 6 K-Means JSON missing: {window_km_json}"

        with open(window_km_json) as f:
            window_km_data = json.load(f)

        assert "clusters" in window_km_data, f"K-Means JSON missing 'clusters' field"
        assert len(window_km_data["clusters"]) == 3, \
            f"K-Means has {len(window_km_data['clusters'])} clusters (expected 3)"

    # All validations passed
    print(f"  ✓ Output validation passed: {1 + len(windows)*2} JSON files")
```

---

### 3.3 Error Handling

**Location**: Error handling is inline in Section 2.2 orchestration block

**Implemented Errors** (from MLAnalysisGenerationCHILDTI.md Section 6):
- `FileNotFoundError`: Missing upstream files (Stage 4/5 outputs)
- `ValueError`: Input validation failed (corrupted models, invalid data)
- `AssertionError`: Output validation failed (wrong JSON count, schema error)
- `IOError/OSError`: I/O failure (disk full, permissions)
- `Exception`: Unexpected errors

**Cleanup**: No partial outputs to clean (Stage 6 uses atomic write pattern with temp files internally)

---

## 4. Checkpoint Schema

**Source**: TI Section 2.2 (StageOutput), rumiai_ml_batch.py Stage 5 pattern

```python
# Checkpoint schema for Stage 6: ML Analysis Generation

Checkpoint_Stage6Schema = {
    "stage": "stage_6_ml_analysis_generation",  # Fixed identifier
    "bucket": str,                               # Bucket name (e.g., "18-33s")
    "status": str,                               # "completed" | "in_progress" | "failed"
    "timestamp": str,                            # ISO 8601 format

    # Stage completion markers (from TI Section 2.2 StageOutput)
    "json_files_generated": int,                 # Count of JSON files (1 + 2×N)
    "output_files": [
        # Example for bucket "18-33s" (6 windows):
        "ml_analysis/rf_video_analysis.json",
        "ml_analysis/hook_rf_analysis.json",
        "ml_analysis/hook_kmeans_analysis.json",
        "ml_analysis/middle_1_rf_analysis.json",
        "ml_analysis/middle_1_kmeans_analysis.json",
        # ... (total 13 files for 18-33s)
    ]
}
```

**Validation**:
- [x] Schema follows rumiai_ml_batch.py checkpoint format (Stage 5 pattern)
- [x] outputs_created list matches TI Section 2.2 StageOutput exactly
- [x] Stage identifier format: `stage_6_ml_analysis_generation`

---

## 5. Integration Tests

**File**: `tests/integration/test_pipeline_stage_6.py`

**Action**: CREATE

```python
import pytest
import os
import json
from pathlib import Path

def test_stage_6_happy_path(tmp_path):
    """
    Test Stage 6 integration (happy path).
    Source: MLAnalysisGenerationCHILDTI.md Section 7 Trace 1

    Note: Requires mock data setup or validation mocking.
    Test creates checkpoint structure but not the 40 input files Stage 6 needs.
    Consider mocking validate_stage_6_prerequisites() for unit testing.
    """
    # Setup test bucket with Stage 5 outputs
    bucket_path = tmp_path / "buckets" / "bucket_18-33s"
    bucket_path.mkdir(parents=True)

    # Create mock Stage 5 checkpoint
    checkpoint_dir = bucket_path / "checkpoints"
    checkpoint_dir.mkdir()
    stage5_checkpoint = {
        "stage": "stage_5_ml_model_training",
        "bucket": "18-33s",
        "status": "completed",
        "models_trained": 13
    }
    with open(checkpoint_dir / "stage_5_checkpoint.json", 'w') as f:
        json.dump(stage5_checkpoint, f)

    # Run Stage 6 (simulated - actual test would call pipeline)
    # Expected: 13 JSON files (1 video + 6 window RF + 6 window K-Means)

    # Verify outputs
    ml_analysis_dir = bucket_path / "ml_analysis"
    assert ml_analysis_dir.exists()

    # Verify video RF JSON
    assert (ml_analysis_dir / "rf_video_analysis.json").exists()

    # Verify window JSONs (example for hook)
    assert (ml_analysis_dir / "hook_rf_analysis.json").exists()
    assert (ml_analysis_dir / "hook_kmeans_analysis.json").exists()


@pytest.mark.skip(reason="Implementation pending - add before merge")
def test_stage_6_checkpoint_skip():
    """Test Stage 6 skips when checkpoint exists."""
    # Create checkpoint
    # Run pipeline
    # Verify stage skipped
    pass


@pytest.mark.skip(reason="Implementation pending - add before merge")
def test_stage_6_error_handling():
    """
    Test Stage 6 error handling.
    Source: MLAnalysisGenerationCHILDTI.md Section 7 Trace 3
    """
    # Simulate missing Stage 5 models
    # Run pipeline
    # Verify FileNotFoundError raised
    # Verify error message matches TI Section 6
    # Verify bucket skipped, pipeline continues
    pass


@pytest.mark.skip(reason="Implementation pending - add before merge")
def test_stage_6_resume_after_error():
    """Test Stage 6 resume after fixing error."""
    # Simulate previous failure
    # Fix missing models
    # Re-run pipeline
    # Verify stage completes successfully
    pass
```

---

## 6. Documentation Updates

### 6.1 Pipeline Diagram

**BEFORE**:
```
Stage 5: ML Model Training
    ↓
Stage 7: LLM Report Generation (TODO)
```

**AFTER**:
```
Stage 5: ML Model Training
    ↓
Stage 6: ML Analysis Generation ← NEW
    ↓
Stage 7: LLM Report Generation (TODO)
```

**Update Locations**:
1. rumiai_ml_batch.py docstring (lines 183-184)
2. rumiai_ml_batch.py FINAL STATUS section (lines 1013-1014)

**Action**: UPDATE

```python
# Location 1: Docstring (lines 183-184)
    # Stage 5: ML Model Training
    # Stage 6: ML Analysis Generation
    # Stage 7: LLM Report Generation (TODO)

# Location 2: FINAL STATUS section (lines 1013-1014)
- print("⧗ Stage 6: ML Analysis Generation - TODO")
+ print("✓ Stage 6: ML Analysis Generation - COMPLETE")
```

---

### 6.2 CHANGELOG Entry

```markdown
## [Unreleased]

### Added
- Stage 6: ML Analysis Generation - Extract insights from trained ML models and generate structured JSON files
  - Entry point: `generate_ml_analysis_jsons` (from MLAnalysisGenerationCHILDTI.md Section 8.2)
  - Outputs:
    - 1 video-level RF JSON (cross-window feature importance)
    - N window-level RF JSONs (per-window feature importance with rank)
    - N window-level K-Means JSONs (cluster centroids, top videos per cluster)
    - Total: 1 + (2 × N) JSON files per bucket
  - CLI parameters: None (uses Foundation params only)
  - Checkpoint: stage_6_checkpoint.json (atomic write pattern)
  - Error handling: Skip-on-fail for bucket-specific errors, exit for system-wide errors
```

---

## 7. Pre-Merge Validation Checklist

**Before merging Stage 6 integration:**

### Code Modifications
- [ ] rumiai_ml_batch.py import added (Section 2.1)
- [ ] Pipeline orchestration code added (Section 2.2)
- [ ] CLI arguments: N/A (uses Foundation params only)
- [ ] Helper functions added (Section 3.1-3.2)

### Testing
- [ ] Integration tests pass (Section 5):
  - [ ] `test_stage_6_happy_path`
  - [ ] `test_stage_6_checkpoint_skip`
  - [ ] `test_stage_6_error_handling`
  - [ ] `test_stage_6_resume_after_error`
- [ ] Manual testing completed:
  - [ ] Run full pipeline with Stage 6 on real data
  - [ ] Verify outputs match TI Section 7 Trace 1 (13 JSONs for bucket 18-33s)
  - [ ] Test error scenario: missing Stage 5 models
  - [ ] Test checkpoint/resume: verify skip when checkpoint exists

### Documentation
- [ ] Pipeline diagram updated (Section 6.1)
- [ ] CHANGELOG entry created (Section 6.2)
- [ ] rumiai_ml_batch.py docstring updated (Stage 6 added to list)

### Code Review
- [ ] All code derived from TI document (no hallucinated features)
- [ ] Entry function name matches TI Section 8.2 exactly: `generate_ml_analysis_jsons`
- [ ] Error messages match TI Section 6 exactly
- [ ] Checkpoint schema follows rumiai_ml_batch.py format (Stage 5 pattern)
- [ ] No modifications to existing stages (integration is additive)
- [ ] All 40 prerequisite files validated (4 + 6×N for N windows)
- [ ] All 13 output files validated (1 + 2×N for N windows)

---

## 8. Rollback Plan

**If Stage 6 integration causes issues:**

### Step 1: Revert rumiai_ml_batch.py
```bash
git revert <commit_hash>
```

### Step 2: Remove Stage 6 checkpoints
```bash
find /data/clients -name "stage_6_checkpoint.json" -delete
```

### Step 3: Remove partial Stage 6 outputs (if any)
```bash
# Remove only Stage 6-specific JSON files
# Video-level RF analysis
find /data/clients -path "*/ml_analysis/rf_video_analysis.json" -delete

# Window-level RF analysis (all windows)
find /data/clients -path "*/ml_analysis/*_rf_analysis.json" -delete

# Window-level K-Means analysis (all windows)
find /data/clients -path "*/ml_analysis/*_kmeans_analysis.json" -delete
```

### Step 4: Document issues
- Create bug report with logs
- Note which test failed
- Identify root cause (integration issue vs TI spec issue vs implementation bug)

---

## 9. Integration History

| Date | Action | Author | Notes |
|------|--------|--------|-------|
| 2025-10-21 | Created | Claude (Sonnet 4.5) | From MLAnalysisGenerationCHILDTI.md |
| TBD | Reviewed | TBD | TBD |
| TBD | Merged | TBD | PR #TBD |

---

## 10. References

**Source Documents**:
- **TI Document**: MLAnalysisGenerationCHILDTI.md (all code specifications)
- **HLD Document**: MLAnalysisGenerationCHILD.md (design rationale)
- **Mother HLD**: MLPlanningv2.md Stage 6 section
- **Foundation**: FoundationCHILD.md (client architecture, directory structure)
- **Pipeline**: rumiai_ml_batch.py (current version before integration)

**Traceability**:
- Section 2.1: TI Section 8.2 (Entry Function), TI Section 2.1 (Stage6Input parameters)
- Section 2.2: TI Section 2.2 (Stage6Output file list)
- Section 3.1: TI Section 2.1 (40 prerequisite files: 1 + 13 + 26)
- Section 3.2: TI Section 5 (Output Validation)
- Section 4: TI Section 2.2 (StageOutput), rumiai_ml_batch.py Stage 5 checkpoint pattern
- Section 5: TI Section 7 (Traces 1, 3)

**Key Metrics**:
- **Input Files**: 40 files for bucket 18-33s (4 + 6×N, where N=6 windows)
  - Stage 3: 1 CSV
  - Stage 4: 13 CSVs
  - Stage 5: 26 files (models + metrics)
- **Output Files**: 13 JSON files for bucket 18-33s (1 + 2×N)
  - 1 video-level RF JSON
  - 6 window-level RF JSONs
  - 6 window-level K-Means JSONs
- **Expected Duration**: ~1-5 seconds per bucket (lightweight JSON generation from pre-trained models)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-21
**Status**: Pending Review
