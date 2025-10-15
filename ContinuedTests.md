# Continued Testing Guide - Stage 3+ Resume Strategy

---

## 🚨 FOR FRESH CLI INSTANCE

**If you're a new CLI session being told to read this document:**

This document describes how to **continue testing from Stage 3 onwards** using the 111 videos that were already processed through Stage 2. You don't need to re-scrape or re-process videos - the temporal_windows JSON files are ready and waiting.

**Your task**: Implement and test subsequent ML pipeline stages (3, 4, 5, etc.) using existing Stage 2 outputs.

---

## Current State: After Stage 2 Completion

### Test Run Summary (2025-10-14)

**Completed**: Stage 0, 1, 2, 2.5
**Duration**: 4 hours 18 minutes
**Videos Processed**: 111/150 (74%)
**Output Files**: 111 `*_temporal_windows_updated.json` files

### Data Location

```
/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/
├── config.json                          ✅ Pipeline configuration
├── winner_analysis.json                 ✅ Winning buckets metadata (top 3)
└── buckets/
    ├── bucket_18-33s/
    │   ├── selected_videos.json         ✅ 50 video metadata records
    │   ├── checkpoints/
    │   │   └── stage_2_checkpoint.json  ✅ Stage 2 completion status
    │   └── analysis/insights/           ✅ 50 temporal_windows files
    │       ├── 7545713916584774968_temporal_windows_updated.json
    │       ├── 7544734155570105656_temporal_windows_updated.json
    │       └── ... (48 more files)
    │
    ├── bucket_13-18s/
    │   ├── selected_videos.json         ✅ 29 video metadata records
    │   ├── checkpoints/
    │   │   └── stage_2_checkpoint.json  ✅ Stage 2 completion status
    │   └── analysis/insights/           ✅ 26 temporal_windows files
    │
    └── bucket_60-90s/
        ├── selected_videos.json         ✅ 35 video metadata records
        ├── checkpoints/
        │   └── stage_2_checkpoint.json  ✅ Stage 2 completion status
        └── analysis/insights/           ✅ 35 temporal_windows files
```

### Bucket Breakdown

| Bucket | Videos Selected | Successfully Processed | Temporal Windows Files |
|--------|----------------|------------------------|------------------------|
| 18-33s | 50 | 50 (100%) | 50 files |
| 13-18s | 29 | 26 (90%) | 26 files |
| 60-90s | 35 | 35 (100%) | 35 files |
| **Total** | **114** | **111 (97.4%)** | **111 files** |

**Note**: 3 videos failed in bucket 13-18s due to temporal computation errors (see `TemporalBug.md`).

---

## Approach 1: Stage-Specific Resume Scripts

### Overview

Create individual `run_stageN_only.py` scripts that:
1. Load outputs from previous stage
2. Validate prerequisites exist
3. Run only that specific stage
4. Save outputs for next stage

### Benefits

✅ **Isolation**: Test each stage independently
✅ **Fast Iteration**: No need to re-run earlier stages
✅ **Clear Dependencies**: Explicit validation of inputs
✅ **Debugging**: Easy to identify which stage failed
✅ **Development**: Build and test stages incrementally

---

## Template: `run_stage3_only.py`

### Stage 3: Feature Aggregation

**Purpose**: Aggregate 111 temporal_windows JSON files into bucket-level feature matrices.

**Input**: `buckets/bucket_*/analysis/insights/*_temporal_windows_updated.json` (from Stage 2)
**Output**: `buckets/bucket_*/aggregated_features.csv` (for Stage 4)

```python
#!/usr/bin/env python3
"""
Stage 3-Only: Feature Aggregation

Resume testing from Stage 3 using existing Stage 2 outputs.

Usage:
    python3 run_stage3_only.py

Prerequisites:
    - Stage 2 completed successfully
    - Temporal windows files exist in buckets/bucket_*/analysis/insights/
    - winner_analysis.json and config.json exist

Outputs:
    - buckets/bucket_*/aggregated_features.csv
    - Stage 3 completion checkpoint
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
def load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

load_env()

from ml_pipeline.stage3_aggregation import stage_3_feature_aggregation_main


def setup_logging():
    """Setup logging for Stage 3-only run"""
    data_root = os.getenv("DATA_ROOT", str(Path(__file__).parent / "data"))
    log_dir = Path(data_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"stage3_only_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Stage 3-only logging initialized: {log_file}")
    return logger


def validate_stage2_outputs(analysis_base: Path) -> dict:
    """
    Validate Stage 2 completed successfully.

    Returns:
        dict with validation results and details
    """
    errors = []
    warnings = []

    # Check required files
    if not (analysis_base / "winner_analysis.json").exists():
        errors.append("Missing winner_analysis.json (from Stage 1)")

    if not (analysis_base / "config.json").exists():
        errors.append("Missing config.json (from Stage 0)")

    # Check buckets directory
    buckets_dir = analysis_base / "buckets"
    if not buckets_dir.exists():
        errors.append("Missing buckets/ directory")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check each bucket has insights
    bucket_stats = {}
    for bucket_dir in buckets_dir.iterdir():
        if not bucket_dir.is_dir():
            continue

        bucket_name = bucket_dir.name.replace("bucket_", "")
        insights_dir = bucket_dir / "analysis/insights"

        if not insights_dir.exists():
            warnings.append(f"Bucket {bucket_name} missing analysis/insights/")
            continue

        # Count temporal_windows files
        temporal_files = list(insights_dir.glob("*_temporal_windows_updated.json"))
        bucket_stats[bucket_name] = {
            "insights_dir": insights_dir,
            "temporal_files": len(temporal_files),
            "files": temporal_files
        }

    if not bucket_stats:
        errors.append("No buckets with temporal_windows files found")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "bucket_stats": bucket_stats
    }


def main():
    """Run Stage 3: Feature Aggregation"""
    try:
        print("="*80)
        print("STAGE 3-ONLY: FEATURE AGGREGATION")
        print("="*80)
        print()

        logger = setup_logging()

        # Define analysis base path
        data_root = Path(os.getenv("DATA_ROOT", str(Path(__file__).parent / "data")))
        analysis_base = data_root / "clients/test_final/hashtags/test_vitamin/top_contrastive"

        print(f"Analysis base: {analysis_base}")
        logger.info(f"Analysis base: {analysis_base}")

        # Validate Stage 2 outputs exist
        print("\n" + "="*80)
        print("VALIDATING STAGE 2 OUTPUTS")
        print("="*80)

        validation = validate_stage2_outputs(analysis_base)

        if not validation["valid"]:
            print("\n✗ VALIDATION FAILED:")
            for error in validation["errors"]:
                print(f"  ERROR: {error}")
            for warning in validation["warnings"]:
                print(f"  WARNING: {warning}")
            logger.error("Stage 2 validation failed")
            return 1

        print("✓ Stage 2 outputs validated")

        # Show bucket statistics
        print("\nBucket Statistics:")
        for bucket_name, stats in validation["bucket_stats"].items():
            print(f"  {bucket_name}: {stats['temporal_files']} temporal_windows files")
            logger.info(f"Bucket {bucket_name}: {stats['temporal_files']} files")

        if validation["warnings"]:
            print("\nWarnings:")
            for warning in validation["warnings"]:
                print(f"  ⚠️  {warning}")

        # Load winner analysis
        with open(analysis_base / "winner_analysis.json") as f:
            winner_analysis = json.load(f)

        # Load config
        with open(analysis_base / "config.json") as f:
            config = json.load(f)

        winning_buckets = winner_analysis['top_3_buckets']
        print(f"\nWinning buckets: {', '.join(winning_buckets)}")
        logger.info(f"Processing {len(winning_buckets)} winning buckets: {winning_buckets}")

        # ===== STAGE 3: FEATURE AGGREGATION =====
        print("\n" + "="*80)
        print("STAGE 3: FEATURE AGGREGATION")
        print("="*80)

        stage3_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 3 for bucket: {bucket_name}")
            print(f"\n--- Processing bucket: {bucket_name} ---")

            # Check if bucket has temporal_windows files
            if bucket_name not in validation["bucket_stats"]:
                logger.warning(f"Bucket {bucket_name} has no temporal_windows files, skipping")
                print(f"⚠️  No temporal_windows files found for {bucket_name}, skipping")
                continue

            bucket_stats = validation["bucket_stats"][bucket_name]
            insights_dir = bucket_stats["insights_dir"]
            file_count = bucket_stats["temporal_files"]

            print(f"✓ Found {file_count} temporal_windows files")
            print(f"Processing feature aggregation for bucket {bucket_name}...")

            # Run Stage 3 feature aggregation
            try:
                summary = stage_3_feature_aggregation_main(
                    config=config,
                    bucket_name=bucket_name,
                    insights_dir=str(insights_dir)
                )

                stage3_summaries[bucket_name] = summary
                logger.info(f"Bucket {bucket_name} complete: {summary['videos_processed']} videos aggregated")
                print(f"✓ Bucket {bucket_name}: {summary['videos_processed']} videos aggregated")
                if summary.get('errors', 0) > 0:
                    print(f"  ⚠️  {summary['errors']} errors occurred")

            except Exception as e:
                logger.error(f"Stage 3 failed for bucket {bucket_name}: {e}", exc_info=True)
                print(f"✗ Bucket {bucket_name} failed: {e}")
                # Continue with other buckets
                continue

        logger.info("Stage 3 completed for all buckets")
        print("\n✓ Stage 3: Feature Aggregation - COMPLETE")

        # Log Stage 3 summary
        total_videos = sum(s.get('videos_processed', 0) for s in stage3_summaries.values())
        logger.info(f"Stage 3 Summary: {total_videos} videos aggregated across {len(stage3_summaries)} buckets")
        print(f"Summary: {total_videos} videos aggregated across {len(stage3_summaries)} buckets")

        # ===== FINAL STATUS =====
        print("\n" + "="*80)
        print("STAGE 3-ONLY EXECUTION COMPLETE")
        print("="*80)
        print(f"✅ Aggregated {total_videos} videos across {len(winning_buckets)} buckets")
        print(f"   Output location: {analysis_base}/buckets/bucket_*/aggregated_features.csv")
        print("="*80)

        logger.info("="*80)
        logger.info("STAGE 3-ONLY EXECUTION COMPLETE")
        logger.info("="*80)

        return 0

    except KeyboardInterrupt:
        print("\n\n✗ Stage 3 interrupted by user (Ctrl+C)")
        return 130

    except Exception as e:
        print(f"\n✗ Stage 3 failed: {e}")
        if 'logger' in locals():
            logger.error(f"Stage 3 failed: {e}", exc_info=True)
        else:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Template: `run_stage4_only.py`

### Stage 4: Feature Transformation

**Purpose**: Transform aggregated features for ML training (normalization, encoding, etc.).

**Input**: `buckets/bucket_*/aggregated_features.csv` (from Stage 3)
**Output**: `buckets/bucket_*/transformed_features.csv` (for Stage 5)

```python
#!/usr/bin/env python3
"""
Stage 4-Only: Feature Transformation

Resume testing from Stage 4 using existing Stage 3 outputs.

Usage:
    python3 run_stage4_only.py

Prerequisites:
    - Stage 3 completed successfully
    - aggregated_features.csv exists for each bucket

Outputs:
    - buckets/bucket_*/transformed_features.csv
    - Stage 4 completion checkpoint
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline.stage4_transformation import stage_4_feature_transformation_main


def validate_stage3_outputs(analysis_base: Path) -> dict:
    """Validate Stage 3 completed successfully"""
    errors = []
    bucket_stats = {}

    buckets_dir = analysis_base / "buckets"
    if not buckets_dir.exists():
        errors.append("Missing buckets/ directory")
        return {"valid": False, "errors": errors}

    for bucket_dir in buckets_dir.iterdir():
        if not bucket_dir.is_dir():
            continue

        bucket_name = bucket_dir.name.replace("bucket_", "")
        aggregated_file = bucket_dir / "aggregated_features.csv"

        if aggregated_file.exists():
            # Check file has content
            file_size = aggregated_file.stat().st_size
            bucket_stats[bucket_name] = {
                "aggregated_file": aggregated_file,
                "file_size_mb": file_size / 1024 / 1024
            }
        else:
            errors.append(f"Bucket {bucket_name} missing aggregated_features.csv")

    if not bucket_stats:
        errors.append("No buckets with aggregated_features.csv found")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "bucket_stats": bucket_stats
    }


def main():
    """Run Stage 4: Feature Transformation"""
    # Similar structure to run_stage3_only.py
    # Load Stage 3 outputs, transform features, save to transformed_features.csv
    pass


if __name__ == "__main__":
    sys.exit(main())
```

---

## Template: `run_stage5_only.py`

### Stage 5: ML Model Training

**Purpose**: Train Random Forest and K-means models per bucket.

**Input**: `buckets/bucket_*/transformed_features.csv` (from Stage 4)
**Output**: `buckets/bucket_*/models/` (RF + K-means models)

```python
#!/usr/bin/env python3
"""
Stage 5-Only: ML Model Training

Resume testing from Stage 5 using existing Stage 4 outputs.

Usage:
    python3 run_stage5_only.py

Prerequisites:
    - Stage 4 completed successfully
    - transformed_features.csv exists for each bucket

Outputs:
    - buckets/bucket_*/models/rf_model.pkl
    - buckets/bucket_*/models/kmeans_model.pkl
    - Stage 5 completion checkpoint
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline.stage5_training import stage_5_ml_training_main


def validate_stage4_outputs(analysis_base: Path) -> dict:
    """Validate Stage 4 completed successfully"""
    errors = []
    bucket_stats = {}

    buckets_dir = analysis_base / "buckets"
    for bucket_dir in buckets_dir.iterdir():
        if not bucket_dir.is_dir():
            continue

        bucket_name = bucket_dir.name.replace("bucket_", "")
        transformed_file = bucket_dir / "transformed_features.csv"

        if transformed_file.exists():
            import pandas as pd
            df = pd.read_csv(transformed_file)
            bucket_stats[bucket_name] = {
                "transformed_file": transformed_file,
                "num_samples": len(df),
                "num_features": len(df.columns)
            }
        else:
            errors.append(f"Bucket {bucket_name} missing transformed_features.csv")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "bucket_stats": bucket_stats
    }


def main():
    """Run Stage 5: ML Model Training"""
    # Similar structure to run_stage3_only.py
    # Load transformed features, train RF + K-means, save models
    pass


if __name__ == "__main__":
    sys.exit(main())
```

---

## Validation Quick Reference

### Check Stage 2 Outputs Ready

```bash
# Count temporal_windows files per bucket
for bucket in 18-33s 13-18s 60-90s; do
    count=$(ls data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_$bucket/analysis/insights/*.json 2>/dev/null | wc -l)
    echo "Bucket $bucket: $count files"
done

# Expected output:
# Bucket 18-33s: 50 files
# Bucket 13-18s: 26 files
# Bucket 60-90s: 35 files
```

### Check Stage 3 Outputs Ready

```bash
# Check aggregated_features.csv exists
ls data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_*/aggregated_features.csv
```

### Check Stage 4 Outputs Ready

```bash
# Check transformed_features.csv exists
ls data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_*/transformed_features.csv
```

### Check Stage 5 Outputs Ready

```bash
# Check ML models exist
ls data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_*/models/*.pkl
```

---

## Usage Examples

### Scenario 1: Test Stage 3 Independently

```bash
# Assumes Stage 2 completed successfully (111 temporal_windows files exist)
export DATA_ROOT=/home/jorge/rumiaifinal/data
python3 run_stage3_only.py
```

**Expected Output**:
```
================================================================================
STAGE 3-ONLY: FEATURE AGGREGATION
================================================================================

Analysis base: /home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive

================================================================================
VALIDATING STAGE 2 OUTPUTS
================================================================================
✓ Stage 2 outputs validated

Bucket Statistics:
  18-33s: 50 temporal_windows files
  13-18s: 26 temporal_windows files
  60-90s: 35 temporal_windows files

Winning buckets: 18-33s, 13-18s, 60-90s

================================================================================
STAGE 3: FEATURE AGGREGATION
================================================================================

--- Processing bucket: 18-33s ---
✓ Found 50 temporal_windows files
Processing feature aggregation for bucket 18-33s...
✓ Bucket 18-33s: 50 videos aggregated

--- Processing bucket: 13-18s ---
✓ Found 26 temporal_windows files
Processing feature aggregation for bucket 13-18s...
✓ Bucket 13-18s: 26 videos aggregated

--- Processing bucket: 60-90s ---
✓ Found 35 temporal_windows files
Processing feature aggregation for bucket 60-90s...
✓ Bucket 60-90s: 35 videos aggregated

✓ Stage 3: Feature Aggregation - COMPLETE
Summary: 111 videos aggregated across 3 buckets

================================================================================
STAGE 3-ONLY EXECUTION COMPLETE
================================================================================
✅ Aggregated 111 videos across 3 buckets
   Output location: /home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_*/aggregated_features.csv
================================================================================
```

### Scenario 2: Chain Stages 3 → 4 → 5

```bash
# Run Stage 3
python3 run_stage3_only.py
# Verify: ls data/.../buckets/bucket_*/aggregated_features.csv

# Run Stage 4
python3 run_stage4_only.py
# Verify: ls data/.../buckets/bucket_*/transformed_features.csv

# Run Stage 5
python3 run_stage5_only.py
# Verify: ls data/.../buckets/bucket_*/models/*.pkl
```

### Scenario 3: Re-run Single Stage After Bug Fix

```bash
# Stage 3 failed due to bug, fix the bug, then re-run
python3 run_stage3_only.py

# No need to re-run Stage 2 or earlier
```

---

## Implementation Checklist

### For Each New Stage (e.g., Stage 3):

- [ ] Create `run_stage3_only.py` script
- [ ] Implement `validate_stage2_outputs()` function
- [ ] Load required inputs (previous stage outputs)
- [ ] Import and call stage main function (e.g., `stage_3_feature_aggregation_main`)
- [ ] Handle errors per bucket (continue on failure)
- [ ] Log summary statistics
- [ ] Save outputs for next stage
- [ ] Test with existing Stage 2 data (111 videos)

### Testing Protocol:

1. **Validate Prerequisites**: Run validation function, ensure no errors
2. **Test on Small Sample**: Test with 1 bucket first
3. **Test Full Dataset**: Run on all 3 buckets (111 videos)
4. **Verify Outputs**: Check output files exist and have content
5. **Check Logs**: Review logs for errors/warnings
6. **Prepare Next Stage**: Ensure outputs are in correct format for next stage

---

## File Locations Reference

### Current Test Data (Post-Stage 2)

```
Base: /home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/

Stage 1 Outputs:
├── winner_analysis.json           # Top 3 buckets metadata
├── buckets/bucket_*/selected_videos.json  # 50/29/35 video metadata

Stage 2 Outputs:
├── buckets/bucket_18-33s/analysis/insights/  # 50 temporal_windows files
├── buckets/bucket_13-18s/analysis/insights/  # 26 temporal_windows files
└── buckets/bucket_60-90s/analysis/insights/  # 35 temporal_windows files

Stage 3 Outputs (TODO):
├── buckets/bucket_18-33s/aggregated_features.csv
├── buckets/bucket_13-18s/aggregated_features.csv
└── buckets/bucket_60-90s/aggregated_features.csv

Stage 4 Outputs (TODO):
├── buckets/bucket_18-33s/transformed_features.csv
├── buckets/bucket_13-18s/transformed_features.csv
└── buckets/bucket_60-90s/transformed_features.csv

Stage 5 Outputs (TODO):
├── buckets/bucket_18-33s/models/rf_model.pkl
├── buckets/bucket_18-33s/models/kmeans_model.pkl
├── buckets/bucket_13-18s/models/rf_model.pkl
├── buckets/bucket_13-18s/models/kmeans_model.pkl
├── buckets/bucket_60-90s/models/rf_model.pkl
└── buckets/bucket_60-90s/models/kmeans_model.pkl
```

---

## Benefits of Stage-Specific Scripts

### Development Speed ⚡
- Test each stage in isolation without waiting for previous stages
- Iterate quickly on single stage bugs
- No need to re-run expensive stages (Stage 2 took 4+ hours)

### Debugging Clarity 🔍
- Clear failure point (which stage failed)
- Easy to add debug logging to specific stage
- Can manually inspect input/output files between stages

### Testing Flexibility 🧪
- Test with subset of data (1 bucket vs all 3)
- Test with synthetic data
- Test edge cases per stage

### Production Safety 🛡️
- Validate prerequisites before running
- Fail fast if inputs missing
- Continue processing other buckets on failure

---

## Next Steps

1. **Implement Stage 3** (`ml_pipeline/stage3_aggregation/main.py`)
2. **Create `run_stage3_only.py`** using template above
3. **Test with 111 existing videos** from Stage 2
4. **Verify outputs** (`aggregated_features.csv` per bucket)
5. **Repeat for Stage 4, 5, etc.**

---

## Related Documentation

- **NewTests.md**: Original bug found during Stage 2 testing
- **TemporalBug.md**: 3 failed videos investigation
- **StageTests.md**: Full end-to-end test plan (if exists)
- **MLRoadmap.md**: Overall ML pipeline architecture

---

**Last Updated**: 2025-10-15
**Test Data**: 111 videos ready from Stage 2 (2025-10-14 run)
**Status**: Ready for Stage 3+ implementation
