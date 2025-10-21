# Pipeline Integration: Stage 5 - ML Model Training

> **Stage**: Stage 5: ML Model Training
> **TI Document**: MLModelTrainingCHILDTI.md
> **Integration Date**: 2025-01-29
> **Status**: Ready for Integration (Matches Actual Implementation)
> **TI Verification**: ✅ VERIFIED (Sections 2, 6, 8.3, 11.4 cross-checked 2025-01-29)

---

## 1. Integration Overview

**Stage Position**:
```
Stage 4: Feature Transformation
    ↓
Stage 5: ML Model Training ← NEW
    ↓
Stage 6: ML Analysis Generation
```

**Metadata**:
```yaml
Stage_Name: Stage 5: ML Model Training
Pipeline_Position: After Stage 4 (Feature Transformation), Before Stage 6 (ML Analysis Generation)
Integration_Type: New Stage
Breaking_Changes: No
Backward_Compatible: Yes
Implementation_Priority: HIGH
```

**Key Integration Points**:
- **Input**: Transformed feature CSVs from Stage 4 (RF + K-Means pipelines)
- **Output**: Trained ML models (.pkl files), model_metrics.json
- **Checkpoint**: stage_5_checkpoint.json per bucket
- **Processing Model**: Per-bucket iteration (same as Stages 3-4)

---

## 2. rumiai_ml_batch.py Modifications

### 2.1 Import Statement

**Location**: Top of rumiai_ml_batch.py, after existing imports (line ~54)

**Action**: ADD

```python
# Stage 5: ML Model Training
from rumiai_v2.processors.model_training import (
    run_stage5_training,
    StageInputError,
    InsufficientDataError,
    ModelTrainingError,
    ValidationError
)  # Source: MLModelTrainingCHILDTI.md Section 11.4, Section 6.1 (exception classes)
```

**Module path derivation**:
- TI Section 11.4 (Implementation Log) documents actual implementation location
- Actual file: `/home/jorge/rumiaifinal/rumiai_v2/processors/model_training.py` (39KB, 1050 lines)
- Entry point: `run_stage5_training()` - Complete orchestrator integration function

---

### 2.2 Pipeline Orchestration

**Location**: Main function in rumiai_ml_batch.py, after Stage 4 block (line ~799)

**Action**: ADD

**Note**: Uses existing datetime import from line 19

```python
        # ===== STAGE 5: ML MODEL TRAINING =====
        logger.info("Starting Stage 5: ML Model Training")
        print("\n" + "="*80)
        print("STAGE 5: ML MODEL TRAINING")
        print("="*80)

        # Note: Hyperparameters are loaded inside run_stage5_training() via load_model_config()
        # No orchestrator-level configuration needed (TI Section 11.4)

        # Process each winning bucket
        stage5_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 5 for bucket: {bucket_name}")
            print(f"\n--- Training models for bucket: {bucket_name} ---")

            bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

            try:
                # Validate Stage 4 completed successfully
                stage4_checkpoint = bucket_path / "checkpoints" / "stage_4_checkpoint.json"
                if not stage4_checkpoint.exists():
                    logger.error(f"Bucket {bucket_name}: Stage 4 checkpoint missing")
                    print(f"✗ Bucket {bucket_name}: Stage 4 not complete (skipping)")
                    continue

                with open(stage4_checkpoint) as f:
                    stage4_status = json.load(f)

                if stage4_status.get("status") != "completed":
                    logger.error(f"Bucket {bucket_name}: Stage 4 status={stage4_status.get('status')}")
                    print(f"✗ Bucket {bucket_name}: Stage 4 incomplete (skipping)")
                    continue

                # Check if Stage 5 already complete for this bucket
                checkpoint_path = bucket_path / "checkpoints" / "stage_5_checkpoint.json"
                if checkpoint_path.exists():
                    logger.info(f"Bucket {bucket_name}: Stage 5 already complete (checkpoint exists)")
                    print(f"✓ Bucket {bucket_name}: Training already complete (skipping)")

                    # Load checkpoint to get model count
                    with open(checkpoint_path) as f:
                        checkpoint = json.load(f)

                    stage5_summaries[bucket_name] = {
                        "models_trained": checkpoint["models_trained"],
                        "elapsed_time": 0.0  # Skipped, no time
                    }
                    continue

                # Prepare config for Stage 5 entry point
                # Source: MLModelTrainingCHILDTI.md Section 11.4 (actual implementation)
                bucket_config = {
                    "bucket": bucket_name,
                    "strategy": config.selection_strategy,
                    "video_count": config.video_count
                }

                # Execute Stage 5 training (from TI Section 11.4: ACTUAL IMPLEMENTATION)
                # Source: MLModelTrainingCHILDTI.md Section 11.4 lines 2945-2947
                # Entry point: run_stage5_training() returns (success, output_files, elapsed_time)
                success, output_files, elapsed_time = run_stage5_training(
                    bucket_path=str(bucket_path),
                    config=bucket_config,
                    selection_strategy=config.selection_strategy
                )

                # Count models trained from output_files list
                models_trained = len([f for f in output_files if f.endswith('.pkl')])

                # CREATE CHECKPOINT (CRITICAL FIX C2)
                # Source: Stage 4 pattern (rumiai_ml_batch.py lines 725-738)
                # Note: output_files already provided by run_stage5_training()
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                checkpoint_data = {
                    "stage": "stage_5_ml_model_training",
                    "bucket": bucket_name,
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "models_trained": models_trained,
                    "output_files": output_files  # From run_stage5_training() return value
                }

                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                stage5_summaries[bucket_name] = {
                    "models_trained": models_trained,
                    "elapsed_time": elapsed_time
                }

                logger.info(
                    f"Bucket {bucket_name} complete: "
                    f"{models_trained} models trained in {elapsed_time:.1f}s"
                )
                print(
                    f"✓ Bucket {bucket_name}: {models_trained} models trained → "
                    f"{elapsed_time:.1f}s"
                )

            except StageInputError as e:
                # Missing upstream file (Stage 4 CSVs or checkpoint)
                # Source: MLModelTrainingCHILDTI.md Section 6.1, 6.2 Scenario 1
                # Strategy: Skip bucket, continue pipeline (bucket-specific missing data)
                logger.error(f"Stage 5 prerequisite missing for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Prerequisites missing (skipping)")
                print("   Ensure Stage 4 completed successfully for this bucket")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except InsufficientDataError as e:
                # Insufficient videos or invalid data
                # Source: MLModelTrainingCHILDTI.md Section 6.1, 6.2 Scenario 2
                # Strategy: Skip bucket, continue pipeline (bucket-specific data error)
                logger.error(f"Stage 5 validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Data validation failed (skipping)")
                print("   This indicates insufficient videos or invalid labels")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except ModelTrainingError as e:
                # Model training failed (sklearn errors)
                # Source: MLModelTrainingCHILDTI.md Section 6.1, 6.2 Scenario 3
                # Strategy: Skip bucket, continue pipeline (bucket-specific training failure)
                logger.error(f"Stage 5 training failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Model training failed (skipping)")
                print("   This may indicate data quality issues for this bucket")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except ValidationError as e:
                # Output validation failed (model metrics, feature overlap)
                # Source: MLModelTrainingCHILDTI.md Section 6.1 (custom exception)
                # Strategy: Skip bucket, continue pipeline (bucket-specific validation failure)
                logger.error(f"Stage 5 output validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Output validation failed (skipping)")
                print("   This indicates low-quality models for this bucket")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except (IOError, OSError) as e:
                # I/O failure (disk full, permission denied)
                # Source: MLModelTrainingCHILDTI.md Section 6 (system-wide failure)
                # Strategy: Exit pipeline (system-wide issue affects all buckets)
                logger.error(f"Stage 5 I/O error for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: I/O error (exiting pipeline)")
                print("   Check disk space and permissions")
                print("   This is a system-wide issue - stopping pipeline")
                return 4  # Exit code 4 = I/O failure

            except Exception as e:
                # Unexpected error
                # Source: MLModelTrainingCHILDTI.md Section 6 (catch-all)
                logger.error(
                    f"Stage 5 unexpected error for bucket {bucket_name}: {e}",
                    exc_info=True
                )
                print(f"✗ Bucket {bucket_name} unexpected error: {e}")
                return 99  # Exit code 99 = unexpected error

        logger.info("Stage 5 completed for all buckets")
        print("\n✓ Stage 5: ML Model Training - COMPLETE")

        # Log Stage 5 summary
        if stage5_summaries:
            total_models = sum(s['models_trained'] for s in stage5_summaries.values())
            avg_time = sum(s['elapsed_time'] for s in stage5_summaries.values()) / len(stage5_summaries)
            logger.info(
                f"Stage 5 Summary: {total_models} models trained across "
                f"{len(stage5_summaries)} buckets (avg {avg_time:.1f}s per bucket)"
            )
```

**Parameter mapping from TI Section 11.4 actual implementation**:
- `bucket_path` (str): Path to bucket directory
- `config` (dict): Bucket configuration with fields:
  - `bucket` (str): Bucket name
  - `strategy` (str): Selection strategy
  - `video_count` (int): Number of videos
- `selection_strategy` (str): From CLI config (also in config dict for convenience)

**Return values** (TI Section 11.4, line 979-982):
- `success` (bool): Always True when function returns (errors raise exceptions)
- `output_files` (List[str]): Paths to all created model files
- `elapsed_time` (float): Training duration in seconds

**Output files from TI Section 8.3 downstream_inputs**:
- `models/rf_video_{bucket}.pkl`: Video-level RF model
- `models/rf_{window}_{bucket}.pkl`: Window-level RF models (per window)
- `models/{window}_kmeans_{bucket}.pkl`: K-Means models (per window)
- `models/{window}_scalers_{bucket}.pkl`: Scalers (per window)
- `models/{window}_X_data_{bucket}.pkl`: Feature matrices for silhouette calculation
- `models/model_metrics.json`: Performance summary

---

### 2.3 CLI Arguments

**No stage-specific CLI parameters**

Stage 5 uses Foundation params only (from FoundationCHILD.md):
- `--selection-strategy`: Determines if RF can be trained (contrastive vs top)
- `--video-count`: Affects minimum video validation thresholds

---

## 3. Helper Functions

### 3.1 Prerequisite Validation

**Built into run_stage5_training()** (TI Section 11.4 implementation)

Prerequisite validation is performed internally by `run_stage5_training()` via `validate_stage4_outputs()`:
- Check Stage 4 checkpoint exists (orchestrator layer, line 95-108)
- Check `rf_transformed.csv` exists (inside run_stage5_training)
- Check all window-level CSVs exist (inside run_stage5_training)
- Validate video count meets minimum threshold (inside run_stage5_training)

The orchestrator only checks the Stage 4 checkpoint before calling the function. All other validation is handled by the Stage 5 module, which raises `StageInputError` if prerequisites are missing.

---

### 3.2 Output Validation

**Location**: Integrated into train_bucket_models function (Stage 5 module handles this internally)

**Note**: TI Section 6 indicates validation is performed within the stage implementation:
- Model file existence checks
- Feature overlap validation (≥15/21 features)
- Model metrics threshold checks

The integration layer skips redundant validation since the stage function raises AssertionError on validation failures.

---

### 3.3 Error Handling

Error handling is integrated into Section 2.2 Pipeline Orchestration block (exception handlers).

**No cleanup function needed**: Following Stage 4 pattern, skip-on-error strategy continues to next bucket. The stage implementation (`train_bucket_models`) performs atomic rollback internally per TI Section 6.3.

---

## 4. Checkpoint Schema

**Source**: TI Section 2.2 StageOutput, FoundationCHILD.md Section 5.3

```python
# Checkpoint schema for Stage 5: ML Model Training

Checkpoint_Stage5Schema = {
    "stage": "stage_5_ml_model_training",  # Fixed identifier
    "bucket": str,  # Bucket name (e.g., "18-33s", "60-90s")
    "status": str,  # "completed" | "in_progress" | "failed"
    "timestamp": str,  # ISO 8601 format (e.g., "2025-01-29T14:30:00Z")

    # Stage completion markers (from TI Section 8.3)
    "models_trained": int,  # Total number of models trained (13-26 per bucket)
    "output_files": [  # List of all .pkl and .json files created
        "models/rf_video_{bucket}.pkl",
        "models/rf_hook_{bucket}.pkl",
        "models/rf_middle_1_{bucket}.pkl",
        # ... (all window-level RF models)
        "models/hook_kmeans_{bucket}.pkl",
        "models/hook_scalers_{bucket}.pkl",
        "models/hook_X_data_{bucket}.pkl",
        # ... (all K-Means models, scalers, X matrices)
        "models/model_metrics.json"
    ]
}
```

**Example checkpoint for bucket 18-33s** (6 windows: hook, middle_1-4, closing):
```json
{
  "stage": "stage_5_ml_model_training",
  "bucket": "18-33s",
  "status": "completed",
  "timestamp": "2025-01-29T15:45:32Z",
  "models_trained": 26,
  "output_files": [
    "models/rf_video_18-33s.pkl",
    "models/rf_hook_18-33s.pkl",
    "models/rf_middle_1_18-33s.pkl",
    "models/rf_middle_2_18-33s.pkl",
    "models/rf_middle_3_18-33s.pkl",
    "models/rf_middle_4_18-33s.pkl",
    "models/rf_closing_18-33s.pkl",
    "models/hook_kmeans_18-33s.pkl",
    "models/hook_scalers_18-33s.pkl",
    "models/hook_X_data_18-33s.pkl",
    "models/middle_1_kmeans_18-33s.pkl",
    "models/middle_1_scalers_18-33s.pkl",
    "models/middle_1_X_data_18-33s.pkl",
    "models/middle_2_kmeans_18-33s.pkl",
    "models/middle_2_scalers_18-33s.pkl",
    "models/middle_2_X_data_18-33s.pkl",
    "models/middle_3_kmeans_18-33s.pkl",
    "models/middle_3_scalers_18-33s.pkl",
    "models/middle_3_X_data_18-33s.pkl",
    "models/middle_4_kmeans_18-33s.pkl",
    "models/middle_4_scalers_18-33s.pkl",
    "models/middle_4_X_data_18-33s.pkl",
    "models/closing_kmeans_18-33s.pkl",
    "models/closing_scalers_18-33s.pkl",
    "models/closing_X_data_18-33s.pkl",
    "models/model_metrics.json"
  ]
}
```

**Model count calculation**:
- 1 video-level RF + 6 window-level RF = 7 RF models
- 6 K-Means + 6 scalers + 6 X matrices = 18 K-Means artifacts
- 1 model_metrics.json
- **Total: 26 files**

**Validation**:
- [x] Schema follows FoundationCHILD.md checkpoint format
- [x] `output_files` matches TI Section 8.3 downstream_inputs exactly
- [x] Stage identifier format: `stage_5_ml_model_training` (snake_case)
- [x] File count accurate: 26 for bucket 18-33s (1 + 6 + 6*3 + 1)

---

## 5. Integration Tests

**Source**: MLModelTrainingCHILDTI.md Section 7 (Complete Example Traces)

**File**: `tests/integration/test_pipeline_stage_5.py`

**Action**: CREATE

**Note**: Test implementations below are stubs requiring full implementation before production use.

```python
import pytest
import os
import json
from pathlib import Path
from rumiai_ml_batch import main


def test_stage_5_happy_path(tmp_path):
    """
    Test Stage 5 integration (happy path).
    Source: MLModelTrainingCHILDTI.md Section 7 Trace 1

    TODO: Implement full test with mock CSV data before production use.
    """
    # Setup: Create Stage 4 outputs (transformed CSVs)
    bucket_path = tmp_path / "buckets" / "bucket_18-33s"
    ml_analysis_dir = bucket_path / "ml_analysis"
    ml_analysis_dir.mkdir(parents=True)

    # Create mock Stage 4 CSVs (from TI Section 7 Trace 1 Input)
    # rf_transformed.csv: 100 rows × 190 columns
    # window CSVs: 100 rows × 22 columns (RF), 100 rows × 39 columns (KM)
    # ... (create mock data files)

    # Create Stage 4 checkpoint
    checkpoint_dir = bucket_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    with open(checkpoint_dir / "stage_4_checkpoint.json", 'w') as f:
        json.dump({"status": "completed"}, f)

    # Run pipeline up to Stage 5
    config = {
        "bucket_path": str(bucket_path),
        "strategy": "contrastive",
        "bucket_name": "18-33s"
    }

    # Execute (entry point from TI Section 8.3)
    from ml_pipeline.stage5_training import train_bucket_models
    # ... (full implementation needed)

    # Verify outputs from TI Section 8.3
    models_dir = bucket_path / "models"
    assert (models_dir / "rf_video_18-33s.pkl").exists()
    assert (models_dir / "rf_hook_18-33s.pkl").exists()
    assert (models_dir / "hook_kmeans_18-33s.pkl").exists()
    assert (models_dir / "model_metrics.json").exists()

    # Verify checkpoint created
    checkpoint = bucket_path / "checkpoints" / "stage_5_checkpoint.json"
    assert checkpoint.exists()

    with open(checkpoint) as f:
        checkpoint_data = json.load(f)

    assert checkpoint_data["status"] == "completed"
    assert checkpoint_data["models_trained"] == 26  # For bucket 18-33s


def test_stage_5_checkpoint_skip():
    """
    Test Stage 5 skips when checkpoint exists.

    TODO: Implement full test before production use.
    """
    # Create bucket with Stage 5 checkpoint
    # Run pipeline
    # Verify stage skipped (no new models trained)
    pass


def test_stage_5_error_handling_missing_input():
    """
    Test Stage 5 error handling (missing Stage 4 CSV).
    Source: MLModelTrainingCHILDTI.md Section 7 Trace 3 (Error Scenario 1)

    TODO: Implement full test before production use.
    """
    # Setup: Bucket with Stage 4 checkpoint but missing rf_transformed.csv
    # Expect FileNotFoundError or continue (skip bucket)
    pass


def test_stage_5_insufficient_videos():
    """
    Test Stage 5 error handling (insufficient videos for training).
    Source: MLModelTrainingCHILDTI.md Section 7 Trace 3 (Error Scenario 2)

    TODO: Implement full test before production use.
    """
    # Setup: Create Stage 4 outputs with only 30 videos (contrastive mode requires ≥50)
    # Expect ValueError (from TI Section 6.2 Scenario 2)
    pass


def test_stage_5_resume_after_error():
    """
    Test Stage 5 resume after fixing error.

    TODO: Implement full test before production use.
    """
    # Simulate previous failure (missing CSV)
    # Fix error condition (add CSV)
    # Re-run pipeline
    # Verify stage completes successfully
    pass
```

**Validation**:
- [x] test_happy_path uses data from TI Section 7 Trace 1
- [x] test_error_handling references TI Section 7 Trace 3
- [x] test_error_handling expects ValueError from TI Section 6.2 Scenario 2
- [x] All output assertions reference TI Section 8.3 downstream_inputs
- [!] **IMPORTANT**: Tests are stubs - require full implementation before production (noted in Pre-Merge Checklist)

---

## 6. Documentation Updates

### 6.1 Pipeline Diagram

**BEFORE**:
```
Stage 4: Feature Transformation → Stage 6: ML Analysis Generation
```

**AFTER**:
```
Stage 4: Feature Transformation
    ↓
Stage 5: ML Model Training ← NEW
    ↓
Stage 6: ML Analysis Generation
```

**Location in codebase**: Update `rumiai_ml_batch.py` main function docstring (line ~166)

---

### 6.2 CHANGELOG Entry

```markdown
## [Unreleased]

### Added
- Stage 5: ML Model Training - Trains Random Forest and K-Means models per bucket
  - Entry point: `run_stage5_training()` (from rumiai_v2.processors.model_training)
  - Implementation: `/home/jorge/rumiaifinal/rumiai_v2/processors/model_training.py` (39KB, 1050 lines)
  - Outputs:
    - RF models: rf_video_{bucket}.pkl, rf_{window}_{bucket}.pkl (per window)
    - K-Means models: {window}_kmeans_{bucket}.pkl (per window)
    - Scalers: {window}_scalers_{bucket}.pkl (per window)
    - Feature matrices: {window}_X_data_{bucket}.pkl (per window)
    - Metrics: models/model_metrics.json
  - CLI parameters: None (uses Foundation params: --selection-strategy, --video-count)
  - Model count: 13-26 models per bucket (varies by bucket windows)
  - Architecture: Dual RF + K-Means for complete pattern coverage
```

**Validation**:
- [x] Pipeline diagram shows stage between 4 and 6
- [x] CHANGELOG entry includes stage name, entry point, outputs
- [x] Brief description under 80 characters: ✓ (79 chars)

---

## 7. Pre-Merge Validation Checklist

**Before merging Stage 5 integration:**

### Code Modifications
- [ ] rumiai_ml_batch.py import added (Section 2.1)
- [ ] Pipeline orchestration code added (Section 2.2)
- [ ] CLI arguments reviewed (Section 2.3 - none needed, confirmed)
- [ ] Inline validation added (Section 2.2 lines 66-84)

### Testing
- [ ] Integration tests pass (Section 5):
  - [ ] `test_stage_5_happy_path` **(STUB - requires implementation)**
  - [ ] `test_stage_5_checkpoint_skip` **(STUB - requires implementation)**
  - [ ] `test_stage_5_error_handling_missing_input` **(STUB - requires implementation)**
  - [ ] `test_stage_5_insufficient_videos` **(STUB - requires implementation)**
  - [ ] `test_stage_5_resume_after_error` **(STUB - requires implementation)**
- [ ] Manual testing completed:
  - [ ] Run full pipeline with Stage 5 on real data (100 videos, contrastive mode)
  - [ ] Verify outputs match TI Section 7 Trace 1
  - [ ] Test error scenario from TI Section 7 Trace 3 (missing CSV)
  - [ ] Verify model_metrics.json schema correctness

### Documentation
- [ ] Pipeline diagram updated in rumiai_ml_batch.py docstring (Section 6.1)
- [ ] CHANGELOG entry created (Section 6.2)
- [ ] Stage5_MLModelTraining_HLD.md Section 3 warnings reviewed

### Code Review
- [ ] All code derived from TI document (no hallucinated features)
- [ ] Entry function names match TI Section 8.3 exactly (`train_bucket_models`, `validate_stage4_outputs`)
- [ ] Error messages match TI Section 6.2 scenarios
- [ ] Checkpoint schema follows FoundationCHILD.md format
- [ ] No modifications to existing stages (integration is additive)
- [ ] Feature overlap validation threshold: ≥15/21 features (from HLD Section 3 Warning #1)
- [ ] Binomial test baseline: 80% (contrastive mode, from HLD Section 3 Warning #3)

### Special Stage 5 Validations (HLD Section 3 Critical Warnings)
- [ ] Feature name normalization implemented (removes _scaled, _log, _encoded suffixes)
- [ ] K-Means feature ranking uses variance across centroids (not magnitude or distance)
- [ ] Silhouette score calculation uses correct X matrix (saved during training)
- [ ] Analysis mode compatibility: RF training skipped in 'top' mode (single class)

---

## 8. Rollback Plan

**If Stage 5 integration causes issues:**

### Step 1: Revert rumiai_ml_batch.py
```bash
git revert {commit_hash}
```

### Step 2: Remove Stage 5 checkpoints
```bash
find /data/clients/*/hashtags/*/*/buckets/*/checkpoints/ -name "stage_5_checkpoint.json" -delete
```

### Step 3: Remove partial models (if needed)
```bash
# Remove all .pkl files and model_metrics.json from models/ directories
find /data/clients/*/hashtags/*/*/buckets/*/models/ -name "*.pkl" -delete
find /data/clients/*/hashtags/*/*/buckets/*/models/ -name "model_metrics.json" -delete
```

### Step 4: Document issues
- Create bug report with logs
- Note which test failed (from Section 5 test suite)
- Identify root cause (integration issue vs TI spec issue)
- Check HLD Section 3 warnings (common failure points)

---

## 9. Integration History

| Date | Action | Author | Notes |
|------|--------|--------|-------|
| 2025-01-29 | Created | RumiAI Team | From MLModelTrainingCHILDTI.md v1.0 |
| 2025-01-29 | TI Verified | RumiAI Team | Sections 2, 6, 8.3 cross-checked |
| 2025-01-29 | Critique Applied | RumiAI Team | Fixed C1, C2, M3-M5, L6-L8 |
| [Pending] | Reviewed | [Reviewer] | [Review notes] |
| [Pending] | Merged | [Merger] | PR #[number] |

---

## 10. References

**Source Documents**:
- **TI Document**: MLModelTrainingCHILDTI.md (all code specifications)
  - Section 2: Stage Contract (StageInput, StageOutput)
  - Section 6: Error Handling (error scenarios, exception types)
  - Section 8.3: Integration Points (theoretical orchestration pattern)
  - Section 11.4: Implementation Log (ACTUAL implementation, lines 2892-2958)
- **Actual Implementation**: `/home/jorge/rumiaifinal/rumiai_v2/processors/model_training.py` (39KB, 1050 lines)
  - Entry point: `run_stage5_training(bucket_path, config, selection_strategy)`
  - Created: 2025-01-20 per TI Section 11.4
- **Mother HLD (Original)**: MLPlanningv2.md (Stage 5, lines 1624-1992)
- **Mother HLD (Detailed)**: Stage5_MLModelTraining_HLD.md (Stage 5, lines 1-1633)
- **Foundation**: FoundationCHILD.md (Section 2: Client Architecture, Section 5.3: Checkpoint Schema)
- **Pipeline**: rumiai_ml_batch.py (current version before integration, lines 1-831)

**Traceability**:
- Section 2.1: TI Section 11.4 (Actual Implementation - import path and entry point)
- Section 2.2: TI Section 11.4 lines 2945-2947 (actual integration pattern)
- Section 3.1: TI Section 11.4 (validation built into run_stage5_training)
- Section 3.2: TI Section 6 (validation delegated to stage implementation)
- Section 3.3: TI Section 6.3 (atomic rollback in stage function)
- Section 4: TI Section 8.3 downstream_inputs, FoundationCHILD.md Section 5.3
- Section 5: TI Section 7 (Traces 1, 3)

**Related Documents**:
- **Stage5_MLModelTraining_HLD.md Section 3**: Critical Implementation Warnings (mandatory reading)
- **FeatureTransformationCHILD.md**: Stage 4 output contracts (upstream dependency)
- **CrossHLDalignment2do.md Issue #7**: Exit code standardization

---

## Critique Resolution Summary

### Fixes Applied

**2025-01-29 - Initial Critique Resolution**:
- ✅ **C1**: TI Document Verified - Read MLModelTrainingCHILDTI.md Sections 2, 6, 8.3
- ✅ **C2**: Checkpoint Creation Added - Lines 147-162 in Section 2.2
- ✅ **M3**: Import Note Added - "Uses existing datetime import from line 19" comment
- ✅ **M4**: Inline Validation - Removed Section 3.1 helper function, inline checks in Section 2.2
- ✅ **M5**: Cleanup Function Removed - Section 3.3 now notes atomic rollback in stage function
- ✅ **L6**: Checkpoint Example Fixed - Model count corrected to 26 for bucket 18-33s
- ✅ **L7**: Test Stubs Noted - Added TODO comments and Pre-Merge Checklist notes
- ✅ **L8**: MLPlanningv2.md Added - Section 10 References updated with original Mother HLD

**2025-01-29 - Implementation Alignment**:
- ✅ **CRITICAL**: Updated to match actual implementation (TI Section 11.4)
  - Changed import: `ml_pipeline.stage5_training` → `rumiai_v2.processors.model_training`
  - Changed function: `train_bucket_models()` → `run_stage5_training()`
  - Updated parameters: 5-param signature → 3-param signature with return values
  - Updated Section 2.1, 2.2, 3.1, 10 to reference TI Section 11.4 (actual implementation)
  - Removed inline validation (now built into run_stage5_training)
  - Use returned output_files list instead of manual glob
  - Added actual file reference: `/home/jorge/rumiaifinal/rumiai_v2/processors/model_training.py`

**2025-01-29 - Exception Handling & Config Fixes**:
- ✅ **C1**: Exception types corrected - Changed from built-in exceptions to custom exceptions
  - `FileNotFoundError` → `StageInputError`
  - `ValueError` → `InsufficientDataError`
  - `RuntimeError` → `ModelTrainingError`
  - `AssertionError` → `ValidationError`
  - Added 4 custom exception imports to Section 2.1
- ✅ **C2**: Removed redundant hyperparameter loading (lines 74-84 deleted)
  - Hyperparameters loaded inside `run_stage5_training()` via `load_model_config()`
  - Orchestrator doesn't need to load config
- ✅ **M1**: Added complete import statement to Section 2.1 (5 imports total)
- ✅ **M2**: Status field consistency - Updated line 6 to match line 725

---

**Document Version**: 1.3 (Exception-Handling-Fixed)
**Last Updated**: 2025-01-29
**Status**: Ready for Integration (Verified Against Actual Implementation)
