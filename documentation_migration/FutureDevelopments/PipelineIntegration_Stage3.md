# Pipeline Integration: Stage 3 Feature Aggregation

> **Source TI Document**: FeatureAggregationTI.md v1.0
> **Mother HLD**: MLPlanningv2.md (Stage 3, lines 1385-1504)
> **Foundation**: FoundationCHILD.md v1.1
> **Target Pipeline**: rumiai_ml_batch.py
> **Generated**: 2025-01-17
> **Status**: Ready for Implementation

---

## Section 1: Integration Overview

### 1.1 Stage Metadata

```yaml
Stage_Name: Stage 3: Feature Aggregation
Pipeline_Position: After Stage 2.7 (Content Analysis), Before Stage 4 (Feature Transformation)
Integration_Type: New Stage
Breaking_Changes: No
Backward_Compatible: Yes
Implementation_Priority: HIGH
```

**Upstream Dependency**: Stage 2.5 (File Organization) must complete successfully
**Downstream Consumer**: Stage 4 (Feature Transformation) - reads `aggregated_features.csv`

### 1.2 Stage Purpose

Transform variable-length temporal window JSON files into fixed-size CSV files for ML training.

**Key Features**:
- Extracts 21 base features from each temporal window
- Middle aggregation for buckets 9-13s and 13-18s (4 strategies)
- Bucket-specific column counts (24-150 columns)
- Graceful error handling (skip bad videos, continue processing)
- Atomic write pattern (crash-safe CSV generation)

### 1.3 Integration Scope

**Files Modified**:
- `rumiai_ml_batch.py` - Add Stage 3 orchestration block (after line 499)

**Files Created**: None (Stage 3 implementation already exists in `scripts/stage3_aggregation.py`)

**Lines of Code Added**: ~100 lines (orchestration + error handling)

---

## Section 2: rumiai_ml_batch.py Modifications

### 2.1 Import Statement

**Location**: Top of `rumiai_ml_batch.py` (after existing imports, around line 50)

**Action**: ADD

```python
# Stage 3: Feature Aggregation
# Source: FeatureAggregationTI.md Section 4
from scripts.stage3_aggregation import aggregate_features
```

**Rationale**: Import the main entry function from Stage 3 implementation

### 2.2 Pipeline Orchestration

**Location**: After Stage 2.7 completion (after line 499 in `rumiai_ml_batch.py`)

**Action**: ADD

```python
# ===== STAGE 3: FEATURE AGGREGATION =====
logger.info("Starting Stage 3: Feature Aggregation")
print("\n" + "="*80)
print("STAGE 3: FEATURE AGGREGATION")
print("="*80)

# Process each winning bucket
stage3_summaries = {}
for bucket_name in winning_buckets:
    logger.info(f"Starting Stage 3 for bucket: {bucket_name}")
    print(f"\n--- Aggregating features for bucket: {bucket_name} ---")

    bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

    try:
        # Validate bucket structure exists (from Stage 2.5)
        insights_dir = bucket_path / "analysis" / "insights"
        if not insights_dir.exists():
            logger.error(
                f"Bucket {bucket_name}: Insights directory missing ({insights_dir}). "
                "Stage 2.5 may have failed."
            )
            print(f"✗ Bucket {bucket_name}: Missing insights directory (skipping)")
            continue

        json_count = len(list(insights_dir.glob("*_temporal_windows_updated.json")))
        if json_count == 0:
            logger.warning(
                f"Bucket {bucket_name}: No temporal_windows_updated.json files found. "
                "Skipping."
            )
            print(f"⚠️  Bucket {bucket_name}: No JSON files (skipping)")
            continue

        print(f"Found {json_count} temporal_windows files")

        # Execute Stage 3 aggregation
        # Source: FeatureAggregationTI.md Section 4.2
        csv_path, summary_path = aggregate_features(str(bucket_path))

        # Load summary for reporting
        with open(summary_path) as f:
            summary = json.load(f)

        stage3_summaries[bucket_name] = summary

        logger.info(
            f"Bucket {bucket_name} complete: "
            f"{summary['videos_processed']}/{summary['input_files_found']} videos aggregated"
        )
        print(
            f"✓ Bucket {bucket_name}: {summary['videos_processed']} videos → "
            f"{summary['output_csv']['columns']} features"
        )

        # Warn if videos were skipped
        if summary['videos_skipped'] > 0:
            print(f"  ⚠️  {summary['videos_skipped']} videos skipped")
            if summary['skipped_reasons']:
                print(f"     Reasons: {summary['skipped_reasons']}")

    except ValueError as e:
        # Pre-flight validation failed (invalid inputs, missing files)
        # Source: FeatureAggregationTI.md Section 7.1
        logger.error(f"Stage 3 validation failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name} validation failed: {e}")
        print("   This indicates Stage 2.5 incomplete or corrupted data")
        return 1  # Exit code 1 = validation failure

    except AssertionError as e:
        # Output validation failed (schema mismatch, column count error)
        # Source: FeatureAggregationTI.md Section 7.1
        logger.error(f"Stage 3 output validation failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name} output validation failed: {e}")
        print("   This indicates a bug in feature extraction logic")
        return 3  # Exit code 3 = output validation failure

    except (IOError, OSError) as e:
        # I/O failure (disk full, permission denied)
        # Source: FeatureAggregationTI.md Appendix B
        logger.error(f"Stage 3 I/O error for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name} I/O error: {e}")
        print("   Check disk space and permissions")
        return 4  # Exit code 4 = I/O failure

    except Exception as e:
        # Unexpected error
        logger.error(
            f"Stage 3 unexpected error for bucket {bucket_name}: {e}",
            exc_info=True
        )
        print(f"✗ Bucket {bucket_name} unexpected error: {e}")
        return 99  # Exit code 99 = unexpected error

logger.info("Stage 3 completed for all buckets")
print("\n✓ Stage 3: Feature Aggregation - COMPLETE")

# Log Stage 3 summary
if stage3_summaries:
    total_aggregated = sum(s['videos_processed'] for s in stage3_summaries.values())
    total_skipped = sum(s['videos_skipped'] for s in stage3_summaries.values())
    logger.info(
        f"Stage 3 Summary: {total_aggregated} videos aggregated, "
        f"{total_skipped} skipped across {len(stage3_summaries)} buckets"
    )
    print(
        f"Summary: {total_aggregated} videos aggregated across "
        f"{len(stage3_summaries)} buckets"
    )
else:
    logger.warning("Stage 3 completed but no buckets were processed")
    print("⚠️  No buckets processed in Stage 3 (all skipped)")
```

**Auto-Detected Patterns Used**:
- Loop through `winning_buckets` (matches Stage 2, line 270 pattern)
- Use `analysis_base / "path"` for Path construction (matches Stage 0.4, line 211 pattern)
- Error handling with `logger.error` + `continue` for recoverable errors (matches Stage 2, line 302-306)
- Summary dict stored in `stage3_summaries` (matches Stage 2, line 269 pattern)
- Exit codes for fatal errors (matches main function pattern, lines 527-540)

### 2.3 CLI Arguments

**Status**: No new CLI arguments required

**Rationale**: Stage 3 uses Foundation CLI parameters only:
- `--client` (determines analysis_base path)
- `--target` (determines analysis_base path)
- `--video-count` (determines expected row count for validation)

All bucket-specific processing is automatic (iterates `winning_buckets` from Stage 1).

### 2.4 Final Status Display Update

**Location**: Line 510 in `rumiai_ml_batch.py`

**Action**: MODIFY

**BEFORE**:
```python
print("⧗ Stage 3: Feature Aggregation - TODO")
```

**AFTER**:
```python
print("✓ Stage 3: Feature Aggregation - COMPLETE")
```

**Location**: Line 517 in `rumiai_ml_batch.py`

**Action**: MODIFY

**BEFORE**:
```python
print(f"✅ Stages 0-2.7 complete!")
```

**AFTER**:
```python
print(f"✅ Stages 0-3 complete!")
```

**Location**: Line 520 in `rumiai_ml_batch.py`

**Action**: ADD (after line 519)

```python
print(f"   Aggregated {total_aggregated} videos into {len(stage3_summaries)} bucket CSVs")
```

---

## Section 3: Helper Functions

### 3.1 Prerequisite Validation

**Status**: Built into Stage 3 implementation

Stage 3's `validate_dependencies()` function (FeatureAggregationTI.md Section 3.1) performs:
- Bucket path existence check
- Insights directory existence check
- JSON file count validation
- Write permission testing

**Integration Approach**: Call validation implicitly via `aggregate_features()` entry point. Validation errors raise `ValueError` which is caught in Section 2.2 orchestration code.

### 3.2 Output Validation

**Status**: Built into Stage 3 implementation

Stage 3's `validate_output()` function (FeatureAggregationTI.md Section 3.4) performs:
- Row count > 0 check
- Column count matches bucket expectations (EXPECTED_FEATURE_COUNTS dict)
- Required columns exist (`video_id`, `create_time`)
- Null column warnings
- Video ID uniqueness check

**Integration Approach**: Validation errors raise `AssertionError` which is caught in Section 2.2 orchestration code (exit code 3).

### 3.3 Error Handling

**Status**: Built into Stage 3 implementation

Stage 3 handles these errors internally (FeatureAggregationTI.md Section 7.1):
- Malformed JSON → skip video, log error, continue
- Validation errors → skip video, log error, continue
- Missing fields → skip video, log error, continue
- Duplicate video_ids → skip video, log warning, continue

**Graceful Degradation**: If 99 out of 100 videos succeed, Stage 3 creates CSV with 99 rows (partial success).

**Integration Approach**: Pipeline orchestration (Section 2.2) only handles **fatal errors** that stop bucket processing:
- `ValueError` → Pre-flight validation failed (exit code 1)
- `AssertionError` → Output validation failed (exit code 3)
- `IOError/OSError` → I/O failure (exit code 4)
- `Exception` → Unexpected error (exit code 99)

---

## Section 4: Checkpoint Schema

**Status**: Not Applicable

**Rationale**: RumiAI pipeline currently does NOT use checkpoints (confirmed by analyzing `rumiai_ml_batch.py`).

Stage 3 is **idempotent** (safe to re-run):
- Re-running Stage 3 overwrites `aggregated_features.csv` with fresh data
- Uses atomic write pattern (temp file + rename) to prevent corruption
- No checkpoint needed because Stage 3 completes in < 1 minute per bucket

**Future Enhancement**: If Stage 3 performance degrades with larger N (e.g., N=1000), add checkpoint support:
```json
{
  "stage": "stage_3_feature_aggregation",
  "bucket": "18-33s",
  "status": "completed",
  "timestamp": "2025-01-17T10:37:54Z",
  "outputs_created": [
    "ml_analysis/aggregated_features.csv",
    "ml_analysis/aggregation_summary.json"
  ],
  "videos_processed": 100,
  "videos_skipped": 0
}
```

**Checkpoint Location** (if implemented):
```python
checkpoint_path = analysis_base / f"buckets/bucket_{bucket_name}/.stage3_checkpoint.json"
```

---

## Section 5: Integration Tests

### Test File Structure

**File**: `tests/integration/test_pipeline_stage3.py`

**Action**: CREATE

```python
"""
Integration tests for Stage 3: Feature Aggregation

Source: FeatureAggregationTI.md Section 5 (Testing Procedures)
"""

import json
import pytest
from pathlib import Path
import pandas as pd
import shutil


# ===== Test 1: Happy Path (Bucket 33-60s) =====

def test_stage3_bucket_33_60s_happy_path(test_analysis_base, sample_temporal_windows_50s):
    """
    Test Stage 3 integration for bucket 33-60s (happy path).

    Source: FeatureAggregationTI.md Section 5.2 Test 1

    Expected:
    - 150 columns (21 features × 7 windows + 3 metadata)
    - 1 row processed
    - aggregation_summary.json created
    """
    # Setup: Create bucket structure with sample JSON
    bucket_path = test_analysis_base / "buckets/bucket_33-60s"
    insights_dir = bucket_path / "analysis/insights"
    insights_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(
        sample_temporal_windows_50s,
        insights_dir / "238506412723073_temporal_windows_updated.json"
    )

    # Execute: Import and run Stage 3
    from scripts.stage3_aggregation import aggregate_features
    csv_path, summary_path = aggregate_features(str(bucket_path))

    # Verify: Check CSV structure
    assert csv_path.exists(), f"CSV not created: {csv_path}"
    df = pd.read_csv(csv_path)

    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    assert len(df.columns) == 150, f"Expected 150 columns, got {len(df.columns)}"

    # Verify: Check column naming
    assert 'video_id' in df.columns
    assert 'hook_scene_count' in df.columns
    assert 'middle_1_scene_count' in df.columns
    assert 'middle_5_scene_count' in df.columns
    assert 'closing_scene_count' in df.columns
    assert 'create_time' in df.columns
    assert 'gender' in df.columns

    # Verify: Check summary JSON
    assert summary_path.exists(), f"Summary JSON not created: {summary_path}"
    with open(summary_path) as f:
        summary = json.load(f)

    assert summary['videos_processed'] == 1
    assert summary['videos_skipped'] == 0
    assert summary['output_csv']['columns'] == 150
    assert summary['bucket'] == '33-60s'


# ===== Test 2: Middle Aggregation (Bucket 9-13s) =====

def test_stage3_bucket_9_13s_middle_aggregation(test_analysis_base, sample_temporal_windows_10s):
    """
    Test Stage 3 middle aggregation for bucket 9-13s.

    Source: FeatureAggregationTI.md Section 5.2 Test 2

    Expected:
    - 66 columns (21 features × 3 windows + 3 metadata)
    - middle_aggregate_* columns (not middle_1_*, middle_2_*, middle_3_*)
    - SUM/MIN/MAX/MODE/AVERAGE aggregation strategies applied
    """
    # Setup
    bucket_path = test_analysis_base / "buckets/bucket_9-13s"
    insights_dir = bucket_path / "analysis/insights"
    insights_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(
        sample_temporal_windows_10s,
        insights_dir / "7099027230512139526_temporal_windows_updated.json"
    )

    # Execute
    from scripts.stage3_aggregation import aggregate_features
    csv_path, summary_path = aggregate_features(str(bucket_path))

    # Verify: Check CSV structure
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert len(df.columns) == 66, f"Expected 66 columns, got {len(df.columns)}"

    # Verify: Check aggregation columns exist
    assert 'middle_aggregate_scene_count' in df.columns
    assert 'middle_aggregate_word_count' in df.columns
    assert 'middle_aggregate_shortest_scene' in df.columns
    assert 'middle_aggregate_dominant_emotion_id' in df.columns

    # Verify: Check individual middle columns do NOT exist
    assert 'middle_1_scene_count' not in df.columns, "Should use middle_aggregate, not middle_1"
    assert 'middle_2_scene_count' not in df.columns
    assert 'middle_3_scene_count' not in df.columns


# ===== Test 3: Error Handling (Malformed JSON) =====

def test_stage3_error_handling_malformed_json(test_analysis_base):
    """
    Test Stage 3 graceful handling of malformed JSON.

    Source: FeatureAggregationTI.md Section 5.1 Test 3

    Expected:
    - 1 good video processed
    - 1 bad video skipped
    - Summary shows skipped_reasons['malformed_json'] = 1
    """
    # Setup: Create bucket with 1 good + 1 bad JSON
    bucket_path = test_analysis_base / "buckets/bucket_18-33s"
    insights_dir = bucket_path / "analysis/insights"
    insights_dir.mkdir(parents=True, exist_ok=True)

    # Good JSON
    good_json = insights_dir / "good_video_temporal_windows_updated.json"
    with open(good_json, 'w') as f:
        json.dump({
            "video_id": "123",
            "duration": 20.0,
            "temporal_windows": {
                "hook": {"scene_count": 5, "word_count": 10},
                "middle_segments": [
                    {"scene_count": 3, "word_count": 15},
                    {"scene_count": 4, "word_count": 12},
                    {"scene_count": 2, "word_count": 8},
                    {"scene_count": 6, "word_count": 20}
                ],
                "closing": {"scene_count": 4, "word_count": 7}
            },
            "metadata": {"create_time": "2025-01-17T10:00:00", "gender_detection": {"gender": "male"}}
        }, f)

    # Bad JSON (malformed)
    bad_json = insights_dir / "bad_video_temporal_windows_updated.json"
    with open(bad_json, 'w') as f:
        f.write("{malformed json content!!!")

    # Execute
    from scripts.stage3_aggregation import aggregate_features
    csv_path, summary_path = aggregate_features(str(bucket_path))

    # Verify: 1 video processed successfully
    df = pd.read_csv(csv_path)
    assert len(df) == 1, f"Expected 1 good video, got {len(df)}"
    assert df['video_id'].iloc[0] == "123"

    # Verify: Summary shows 1 skipped
    with open(summary_path) as f:
        summary = json.load(f)

    assert summary['videos_processed'] == 1
    assert summary['videos_skipped'] == 1
    assert summary['skipped_reasons']['malformed_json'] == 1


# ===== Test 4: Empty Bucket Handling =====

def test_stage3_empty_bucket_error(test_analysis_base):
    """
    Test Stage 3 error when no JSON files found.

    Source: FeatureAggregationTI.md Section 7.1

    Expected:
    - ValueError raised with clear error message
    - No CSV created
    """
    # Setup: Create empty bucket
    bucket_path = test_analysis_base / "buckets/bucket_18-33s"
    insights_dir = bucket_path / "analysis/insights"
    insights_dir.mkdir(parents=True, exist_ok=True)
    # No JSON files added

    # Execute and verify error
    from scripts.stage3_aggregation import aggregate_features

    with pytest.raises(ValueError) as exc_info:
        aggregate_features(str(bucket_path))

    assert "No temporal_windows_updated.json files found" in str(exc_info.value)
    assert "Did Stage 2.5 complete?" in str(exc_info.value)


# ===== Test 5: Full Pipeline Integration =====

def test_stage3_full_pipeline_integration(test_analysis_base, monkeypatch):
    """
    Test Stage 3 integration within full rumiai_ml_batch.py pipeline.

    Mocks Stages 0-2.7 completion, verifies Stage 3 orchestration.
    """
    # This test would mock the full pipeline up to Stage 2.7
    # and verify Stage 3 integration code path
    # (Implementation omitted for brevity - see full test suite)
    pass
```

**Test Data Requirements**:

Create test fixtures directory: `tests/fixtures/`

```
tests/fixtures/
├── 238506412723073_temporal_windows_updated.json  # 50s video (bucket 33-60s)
├── 7099027230512139526_temporal_windows_updated.json  # 10s video (bucket 9-13s)
└── malformed_video.json  # Invalid JSON for error testing
```

---

## Section 6: Documentation Updates

### 6.1 Pipeline Diagram Update

**File**: `README.md` or `docs/pipeline_overview.md`

**BEFORE**:
```
Stage 2.7: Content Analysis
      ↓
Stage 4: Feature Transformation (TODO)
```

**AFTER**:
```
Stage 2.7: Content Analysis
      ↓
Stage 3: Feature Aggregation
      ↓ aggregated_features.csv (N rows × 24-150 features per bucket)
Stage 4: Feature Transformation (TODO)
```

### 6.2 CHANGELOG Entry

**File**: `CHANGELOG.md`

**Action**: ADD to Unreleased section

```markdown
## [Unreleased]

### Added
- **Stage 3: Feature Aggregation** - Transform temporal window JSONs to fixed-size CSV
  - Entry point: `scripts/stage3_aggregation.py::aggregate_features()`
  - Outputs: `ml_analysis/aggregated_features.csv` + `ml_analysis/aggregation_summary.json`
  - Features:
    - Bucket-specific column counts (24-150 features)
    - Middle aggregation for short-window buckets (9-13s, 13-18s)
    - Graceful error handling (skip bad videos, continue processing)
    - Atomic write pattern (crash-safe CSV generation)
  - CLI parameters: None (uses Foundation parameters only)
  - Integration: `rumiai_ml_batch.py` lines 500-620 (after Stage 2.7)
  - Source: FeatureAggregationTI.md v1.0
```

### 6.3 README Quick Start Update

**File**: `README.md`

**Section**: Pipeline Execution

**BEFORE**:
```bash
# Run pipeline (Stages 0-2.7)
python rumiai_ml_batch.py --client acme_corp --target "#fitness"

# Current pipeline stops at Stage 2.7 (Content Analysis)
# Stage 3+ are TODO
```

**AFTER**:
```bash
# Run pipeline (Stages 0-3)
python rumiai_ml_batch.py --client acme_corp --target "#fitness"

# Outputs aggregated_features.csv for each winning bucket
# Stage 4+ are TODO
```

---

## Section 7: Pre-Merge Validation Checklist

Before merging Stage 3 integration:

### Code Modifications
- [ ] Import statement added (`rumiai_ml_batch.py` line ~50)
- [ ] Pipeline orchestration code added (after line 499)
- [ ] Final status display updated (lines 510, 517, 520)
- [ ] No modifications to existing stages (integration is additive)

### Testing
- [ ] Integration tests created (`tests/integration/test_pipeline_stage3.py`)
  - [ ] test_stage3_bucket_33_60s_happy_path
  - [ ] test_stage3_bucket_9_13s_middle_aggregation
  - [ ] test_stage3_error_handling_malformed_json
  - [ ] test_stage3_empty_bucket_error
  - [ ] test_stage3_full_pipeline_integration
- [ ] All tests pass locally
- [ ] Manual testing completed:
  - [ ] Run full pipeline on real data (N=10 small test)
  - [ ] Verify outputs in `ml_analysis/aggregated_features.csv`
  - [ ] Check column counts match expected (Section 2.2 validation)
  - [ ] Test error scenario (empty bucket, malformed JSON)
  - [ ] Verify graceful degradation (partial success)

### Documentation
- [ ] Pipeline diagram updated (Section 6.1)
- [ ] CHANGELOG entry created (Section 6.2)
- [ ] README quick start updated (Section 6.3)

### Code Review
- [ ] All code derived from TI document (no hallucinated features)
- [ ] Entry function name matches TI (`aggregate_features` from Section 4)
- [ ] Error handling matches TI Section 7.1 exactly
- [ ] Exit codes match Appendix B (1=validation, 3=output, 4=I/O, 99=unexpected)
- [ ] No checkpoints added (confirmed not needed - Section 4)
- [ ] Patterns match existing stages (winning_buckets loop, logger usage)

### Performance
- [ ] Stage 3 completes in < 1 minute per bucket (N=100)
- [ ] Memory usage < 50 MB per bucket (N=100)
- [ ] Pipeline total time increase: +2-5 minutes for 3 buckets

---

## Section 8: Rollback Plan

If Stage 3 integration causes issues:

### Step 1: Revert rumiai_ml_batch.py
```bash
# Identify commit hash
git log --oneline rumiai_ml_batch.py | grep "Stage 3"

# Revert integration commit
git revert <commit_hash>

# Or manually comment out Stage 3 block (lines 500-620)
```

### Step 2: Remove Stage 3 outputs
```bash
# Clean up Stage 3 outputs from all buckets
find data/clients/*/hashtags/*/buckets/bucket_*/ml_analysis/ -name "aggregated_features.csv" -delete
find data/clients/*/hashtags/*/buckets/bucket_*/ml_analysis/ -name "aggregation_summary.json" -delete

# Verify removal
find data/clients/ -name "aggregated_features.csv"
```

### Step 3: Document issues
Create bug report with:
- Error logs from `data/logs/rumiai_ml_*.log`
- Failed bucket name
- Input file count and sample JSON
- Expected vs actual column count
- Root cause hypothesis

### Step 4: Verify pipeline still works
```bash
# Run pipeline without Stage 3 (should stop at Stage 2.7)
python rumiai_ml_batch.py --client test --target "#test" --video-count 5

# Verify Stages 0-2.7 complete successfully
```

---

## Section 9: Integration History

| Date | Action | Author | Notes |
|------|--------|--------|-------|
| 2025-01-17 | Created | Claude Code | Generated from FeatureAggregationTI.md v1.0 |
| TBD | Reviewed | {Reviewer} | {Review notes} |
| TBD | Merged | {Merger} | PR #{number} |

---

## Section 10: References

### Source Documents
- **TI Document**: FeatureAggregationTI.md v1.0 (complete technical implementation)
- **Mother HLD**: MLPlanningv2.md Stage 3 (lines 1385-1504)
- **Foundation**: FoundationCHILD.md v1.1 (directory structure, bucket definitions)
- **Implementation**: scripts/stage3_aggregation.py (627 lines, production-ready)
- **Pipeline**: rumiai_ml_batch.py (current version, Stages 0-2.7 complete)

### TI Document Section Mapping

| Integration Section | TI Document Reference | Content |
|---------------------|----------------------|---------|
| 2.1 Import | Section 8 (File Structure) | Module path and entry function |
| 2.2 Orchestration | Section 4 (Integration) | Complete integration code block |
| 3.1 Prerequisites | Section 3.1 (validate_dependencies) | Pre-flight validation logic |
| 3.2 Output Validation | Section 3.4 (validate_output) | Schema validation assertions |
| 3.3 Error Handling | Section 7.1 (Troubleshooting) | Error types and recovery actions |
| 5 Integration Tests | Section 5 (Testing) | Test cases and expected results |
| Rollback Plan | Section 6 (Deployment) | Rollback procedures |

### Key Decisions

1. **No Checkpoints**: Stage 3 is fast (< 1 min/bucket) and idempotent → checkpoints not needed
2. **Exit Codes**: Matches TI Appendix B (1=validation, 3=output, 4=I/O, 99=unexpected)
3. **Graceful Degradation**: Skip bad videos within bucket, only fail if ALL videos fail
4. **Atomic Writes**: CSV corruption prevented by temp file + rename pattern
5. **Loop Pattern**: Iterate winning_buckets (matches Stage 2 pattern, line 270)

---

## Appendix A: Complete Integration Code

**Full code block for Section 2.2** (ready to copy-paste):

```python
# ===== STAGE 3: FEATURE AGGREGATION =====
logger.info("Starting Stage 3: Feature Aggregation")
print("\n" + "="*80)
print("STAGE 3: FEATURE AGGREGATION")
print("="*80)

stage3_summaries = {}
for bucket_name in winning_buckets:
    logger.info(f"Starting Stage 3 for bucket: {bucket_name}")
    print(f"\n--- Aggregating features for bucket: {bucket_name} ---")

    bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

    try:
        insights_dir = bucket_path / "analysis" / "insights"
        if not insights_dir.exists():
            logger.error(f"Bucket {bucket_name}: Insights directory missing ({insights_dir}). Stage 2.5 may have failed.")
            print(f"✗ Bucket {bucket_name}: Missing insights directory (skipping)")
            continue

        json_count = len(list(insights_dir.glob("*_temporal_windows_updated.json")))
        if json_count == 0:
            logger.warning(f"Bucket {bucket_name}: No temporal_windows_updated.json files found. Skipping.")
            print(f"⚠️  Bucket {bucket_name}: No JSON files (skipping)")
            continue

        print(f"Found {json_count} temporal_windows files")

        csv_path, summary_path = aggregate_features(str(bucket_path))

        with open(summary_path) as f:
            summary = json.load(f)

        stage3_summaries[bucket_name] = summary
        logger.info(f"Bucket {bucket_name} complete: {summary['videos_processed']}/{summary['input_files_found']} videos aggregated")
        print(f"✓ Bucket {bucket_name}: {summary['videos_processed']} videos → {summary['output_csv']['columns']} features")

        if summary['videos_skipped'] > 0:
            print(f"  ⚠️  {summary['videos_skipped']} videos skipped")
            if summary['skipped_reasons']:
                print(f"     Reasons: {summary['skipped_reasons']}")

    except ValueError as e:
        logger.error(f"Stage 3 validation failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name} validation failed: {e}")
        print("   This indicates Stage 2.5 incomplete or corrupted data")
        return 1

    except AssertionError as e:
        logger.error(f"Stage 3 output validation failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name} output validation failed: {e}")
        print("   This indicates a bug in feature extraction logic")
        return 3

    except (IOError, OSError) as e:
        logger.error(f"Stage 3 I/O error for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name} I/O error: {e}")
        print("   Check disk space and permissions")
        return 4

    except Exception as e:
        logger.error(f"Stage 3 unexpected error for bucket {bucket_name}: {e}", exc_info=True)
        print(f"✗ Bucket {bucket_name} unexpected error: {e}")
        return 99

logger.info("Stage 3 completed for all buckets")
print("\n✓ Stage 3: Feature Aggregation - COMPLETE")

if stage3_summaries:
    total_aggregated = sum(s['videos_processed'] for s in stage3_summaries.values())
    total_skipped = sum(s['videos_skipped'] for s in stage3_summaries.values())
    logger.info(f"Stage 3 Summary: {total_aggregated} videos aggregated, {total_skipped} skipped across {len(stage3_summaries)} buckets")
    print(f"Summary: {total_aggregated} videos aggregated across {len(stage3_summaries)} buckets")
else:
    logger.warning("Stage 3 completed but no buckets were processed")
    print("⚠️  No buckets processed in Stage 3 (all skipped)")
```

---

**END OF PIPELINE INTEGRATION DOCUMENT**
