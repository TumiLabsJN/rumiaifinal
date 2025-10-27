#!/usr/bin/env python3
"""
Stage 3: Feature Aggregation

Transforms variable-length temporal window JSON files into fixed-size CSV files for ML training.

Source: FeatureAggregationCHILD.md v1.1
Status: Production-Ready

Key Features:
- Extracts 21 base features from each temporal window (hook, middle segments, closing)
- Middle aggregation for buckets 9-13s and 13-18s (4 strategies: SUM, MIN/MAX, MODE, AVERAGE)
- Bucket 0-3s special case: Hook only, no closing window
- Graceful error handling: Skip bad videos, log errors, continue processing
- Atomic write pattern: temp file + rename to prevent corruption
- Output: aggregated_features.csv + aggregation_summary.json

Usage:
    python3 scripts/stage3_aggregation.py --bucket-path="data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s"
"""

import argparse
import json
import logging
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ===== CONFIGURATION =====

# Base features (21 per window) - Source: FeatureAggregationCHILD.md Section 4.2
BASE_FEATURES = [
    'average_face_size',        # Float, [0-1]
    'overlay_unique_count',     # Integer, count
    'has_captions',             # Boolean
    'scene_count',              # Integer, count
    'shortest_scene',           # Float, seconds
    'longest_scene',            # Float, seconds
    'scene_duration_variance',  # Float
    'object_count',             # Integer, count
    'person_count',             # Integer, count
    'dominant_emotion_id',      # Categorical, 1-7
    'speech_coverage',          # Float, [0-1]
    'word_count',               # Integer, count
    'energy_level',             # Float, [0-1]
    'energy_variance',          # Float
    'energy_max',               # Float, [0-1]
    'pitch_scatter_ratio',      # Float, [0-1]
    'gesture_count',            # Integer, count
    'gaze_variance',            # Float
    'eye_contact_rate',         # Float, [0-1]
    'emotional_valence',        # Float, [-1, 1]
    'emotion_consistency'       # Float, [0-1]
]

# Metadata fields (2 video-level fields)
METADATA_FIELDS = ['create_time', 'gender']

# Bucket configurations (window counts) - Source: FoundationCHILD.md Section 6
BUCKET_MIDDLE_SEGMENTS = {
    '0-3s': 0,
    '3-9s': 0,
    '9-13s': 3,
    '13-18s': 3,
    '18-33s': 4,
    '33-60s': 5,
    '60-90s': 5,
    '90-120s': 5
}

# Buckets that aggregate middle segments (short windows) - Source: FeatureAggregationCHILD.md Section 4.2
AGGREGATE_MIDDLE_BUCKETS = ['9-13s', '13-18s']

# Feature aggregation strategies - Source: FeatureAggregationCHILD.md Decision 7
SUM_FEATURES = [
    'scene_count', 'word_count', 'object_count',
    'person_count', 'overlay_unique_count', 'gesture_count'
]

MIN_FEATURES = ['shortest_scene']
MAX_FEATURES = ['longest_scene']
CATEGORICAL_FEATURES = ['dominant_emotion_id', 'has_captions']

# Expected feature counts (for validation) - Source: FeatureAggregationCHILD.md Section 4.2
EXPECTED_FEATURE_COUNTS = {
    '0-3s': 24,    # 21 × 1 window (hook only) + 3 metadata
    '3-9s': 45,    # 21 × 2 windows + 3 metadata
    '9-13s': 66,   # 21 × 3 windows (hook + middle_aggregate + closing) + 3 metadata
    '13-18s': 66,  # 21 × 3 windows (hook + middle_aggregate + closing) + 3 metadata
    '18-33s': 129, # 21 × 6 windows + 3 metadata
    '33-60s': 150, # 21 × 7 windows + 3 metadata
    '60-90s': 150,
    '90-120s': 150
}

# File paths (relative to bucket directory)
INSIGHTS_DIR = "analysis/insights"
OUTPUT_DIR = "ml_analysis"
OUTPUT_CSV = "aggregated_features.csv"
SUMMARY_JSON = "aggregation_summary.json"

# Logging configuration
LOG_PROGRESS_INTERVAL = 10  # Log every N videos

# ===== LOGGING SETUP =====

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== CORE FUNCTIONS =====

def validate_dependencies(bucket_path: Path) -> int:
    """
    Validate all prerequisites before processing (fail fast with clear error messages).

    Source: FeatureAggregationCHILD.md Section 2.3.1

    Args:
        bucket_path: Path to bucket directory (e.g., bucket_18-33s/)

    Returns:
        int: Number of JSON files found

    Raises:
        ValueError: if validation fails with specific error message
    """
    # 1. Check bucket path exists
    if not bucket_path.exists():
        raise ValueError(f"Bucket path does not exist: {bucket_path}")

    # 2. Check insights directory exists and has files
    insights_dir = bucket_path / "analysis" / "insights"
    if not insights_dir.exists():
        raise ValueError(
            f"Insights directory missing: {insights_dir}. "
            "Did Stage 2.5 complete?"
        )

    json_files = list(insights_dir.glob("*_temporal_windows_updated.json"))
    if len(json_files) == 0:
        raise ValueError(
            f"No temporal_windows_updated.json files found in {insights_dir}. "
            "Did Stage 2.5 complete?"
        )

    # 3. Check ml_analysis directory is writable
    ml_analysis_dir = bucket_path / "ml_analysis"
    ml_analysis_dir.mkdir(parents=True, exist_ok=True)

    # Test write permissions
    test_file = ml_analysis_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        raise ValueError(
            f"Cannot write to {ml_analysis_dir}. Check permissions."
        )

    return len(json_files)


def extract_features(temporal_windows_json: Path, bucket: str) -> Dict:
    """
    Extract features from temporal_windows_updated.json.

    Source: FeatureAggregationCHILD.md Section 2.3.2

    Args:
        temporal_windows_json: Path to JSON file
        bucket: Bucket name (e.g., "18-33s") for validation

    Returns:
        dict: Feature dictionary with snake_case column names

    Raises:
        ValueError: if JSON structure invalid or middle_segments missing
    """
    # Load JSON
    with open(temporal_windows_json) as f:
        data = json.load(f)

    # Extract video_id from filename
    video_id = temporal_windows_json.stem.replace('_temporal_windows_updated', '')

    # Validate required fields exist
    if 'metadata' not in data:
        raise ValueError(f"Video {video_id}: Missing 'metadata' field")
    if 'temporal_windows' not in data:
        raise ValueError(f"Video {video_id}: Missing 'temporal_windows' field")

    windows = data['temporal_windows']
    metadata = data['metadata']

    # Initialize feature dictionary
    video_features = {'video_id': video_id}

    # Hook features (1 window - always present)
    for feature in BASE_FEATURES:
        video_features[f'hook_{feature}'] = windows['hook'].get(feature)

    # Middle features (0-5 segments depending on bucket)
    middle_segments = windows.get('middle_segments')

    if middle_segments is None or len(middle_segments) == 0:
        # For buckets 0-3s, 3-9s: middle_segments is null (expected)
        if bucket not in ['0-3s', '3-9s']:
            raise ValueError(
                f"Video {video_id}: null or empty middle_segments "
                f"(bucket {bucket} requires middle segments)"
            )
    else:
        # Aggregate middle segments for short-window buckets
        if bucket in AGGREGATE_MIDDLE_BUCKETS:
            # Aggregate all middle segments into single "middle_aggregate"
            # Reason: Short windows (1-4s) produce unreliable scene/speech features
            # Aggregation creates longer window (4.5-9.3s) for reliable measurements

            for feature in BASE_FEATURES:
                # Collect non-null feature values from all middle segments
                feature_values = [
                    seg.get(feature)
                    for seg in middle_segments
                    if seg.get(feature) is not None
                ]

                # Skip if all values are None
                if len(feature_values) == 0:
                    video_features[f'middle_aggregate_{feature}'] = None
                    continue

                # Apply aggregation strategy based on feature type
                if feature in SUM_FEATURES:
                    # Cumulative features: sum across segments
                    video_features[f'middle_aggregate_{feature}'] = sum(feature_values)
                elif feature in MIN_FEATURES:
                    # Extreme value features: pick minimum
                    video_features[f'middle_aggregate_{feature}'] = min(feature_values)
                elif feature in MAX_FEATURES:
                    # Extreme value features: pick maximum
                    video_features[f'middle_aggregate_{feature}'] = max(feature_values)
                elif feature in CATEGORICAL_FEATURES:
                    # Categorical features: use mode (most common value)
                    mode_series = pd.Series(feature_values).mode()
                    video_features[f'middle_aggregate_{feature}'] = mode_series[0] if len(mode_series) > 0 else None
                else:
                    # Default: average for continuous/ratio features
                    video_features[f'middle_aggregate_{feature}'] = np.mean(feature_values)

            logger.debug(
                f"Video {video_id}: Aggregated {len(middle_segments)} middle segments "
                f"into middle_aggregate (bucket {bucket} has short windows)"
            )

        else:
            # Keep separate middle segments for longer buckets
            # Buckets 18-33s, 33-60s, 60-90s, 90-120s have longer windows (3-22.8s)
            # All features reliable at this duration
            for i, segment in enumerate(middle_segments, start=1):
                for feature in BASE_FEATURES:
                    video_features[f'middle_{i}_{feature}'] = segment.get(feature)

    # Closing features (1 window - skip for bucket 0-3s)
    if bucket != '0-3s':
        for feature in BASE_FEATURES:
            video_features[f'closing_{feature}'] = windows['closing'].get(feature)

    # Metadata (2 fields - video-level, not per-window)
    video_features['create_time'] = metadata.get('create_time')

    # Gender detection (optional field - use .get() with None default)
    gender_data = metadata.get('gender_detection', {})
    video_features['gender'] = gender_data.get('gender')

    return video_features


def validate_input(data: dict, video_id: str, bucket: str):
    """
    Validate temporal_windows_updated.json before feature extraction.

    Source: FeatureAggregationCHILD.md Section 6.1

    Args:
        data: Parsed JSON data
        video_id: Video identifier
        bucket: Bucket name (e.g., "18-33s")

    Raises:
        ValueError: if validation fails with specific error message
    """
    # 1. Check required top-level fields exist
    if 'metadata' not in data:
        raise ValueError(f"Video {video_id}: Missing 'metadata' field")

    if 'temporal_windows' not in data:
        raise ValueError(f"Video {video_id}: Missing 'temporal_windows' field")

    # 2. Validate temporal_windows structure
    windows = data['temporal_windows']

    if 'hook' not in windows:
        raise ValueError(f"Video {video_id}: Missing 'hook' window")

    # Closing window validation (not required for bucket 0-3s)
    if bucket != '0-3s':
        if 'closing' not in windows:
            raise ValueError(f"Video {video_id}: Missing 'closing' window")

    # 3. Validate middle_segments (bucket-specific)
    middle_segments = windows.get('middle_segments')
    expected_middle_count = BUCKET_MIDDLE_SEGMENTS.get(bucket, 0)

    if expected_middle_count > 0:
        # Bucket requires middle segments
        if middle_segments is None or len(middle_segments) == 0:
            raise ValueError(
                f"Video {video_id}: null or empty middle_segments "
                f"(bucket {bucket} requires {expected_middle_count} segments)"
            )

        # Flexible validation for aggregation buckets
        # Warn if segment count doesn't match expected, but proceed with aggregation
        if len(middle_segments) != expected_middle_count:
            logger.warning(
                f"Video {video_id}: Expected {expected_middle_count} middle segments, "
                f"found {len(middle_segments)}. "
                f"{'Aggregation will proceed with available segments.' if bucket in AGGREGATE_MIDDLE_BUCKETS else 'Proceeding anyway.'}"
            )
    else:
        # Bucket 0-3s, 3-9s - middle_segments should be null
        if middle_segments is not None and len(middle_segments) > 0:
            logger.warning(
                f"Video {video_id}: Unexpected middle segments in bucket {bucket}"
            )

    # 4. Validate metadata required fields
    metadata = data['metadata']

    if 'create_time' not in metadata:
        raise ValueError(f"Video {video_id}: Missing metadata.create_time")


def process_bucket(bucket_path: Path, bucket: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Process all videos in bucket directory.

    Source: FeatureAggregationCHILD.md Section 2.3.3

    Args:
        bucket_path: Path to bucket directory
        bucket: Bucket name (e.g., "18-33s")

    Returns:
        tuple: (DataFrame with aggregated features, dict with skip reasons)

    Raises:
        ValueError: if zero valid videos processed
    """
    insights_dir = bucket_path / "analysis" / "insights"
    json_files = list(insights_dir.glob("*_temporal_windows_updated.json"))

    aggregated_data = []
    seen_video_ids = set()
    skipped_reasons = defaultdict(int)

    for i, video_file in enumerate(json_files, start=1):
        try:
            # Extract video_id from filename
            video_id = video_file.stem.replace('_temporal_windows_updated', '')

            # Check for duplicate video_ids
            if video_id in seen_video_ids:
                logger.warning(
                    f"Duplicate video_id {video_id} found in {video_file.name}. "
                    "Skipping."
                )
                skipped_reasons['duplicate_video_id'] += 1
                continue

            seen_video_ids.add(video_id)

            # Load JSON
            with open(video_file) as f:
                data = json.load(f)

            # Validate input
            validate_input(data, video_id, bucket)

            # Extract features
            features = extract_features(video_file, bucket)
            aggregated_data.append(features)

            # Log progress every 10 videos
            if i % LOG_PROGRESS_INTERVAL == 0:
                logger.info(f"Processed {i}/{len(json_files)} videos")

        except json.JSONDecodeError as e:
            logger.error(
                f"Video {video_file.name}: Malformed JSON - {e}. Skipping."
            )
            skipped_reasons['malformed_json'] += 1
            continue

        except ValueError as e:
            # Validation errors (missing fields, null middle_segments)
            logger.error(f"Video {video_file.name}: {e}. Skipping.")
            skipped_reasons['validation_error'] += 1
            continue

        except Exception as e:
            logger.error(
                f"Video {video_file.name}: Unexpected error - {e}. Skipping."
            )
            skipped_reasons['unexpected_error'] += 1
            continue

    # Check if we have ANY valid videos
    if len(aggregated_data) == 0:
        raise ValueError(
            f"No valid videos processed in bucket {bucket_path}. Check logs."
        )

    # Log completion summary
    logger.info(
        f"Successfully processed {len(aggregated_data)}/{len(json_files)} videos "
        f"({len(json_files) - len(aggregated_data)} skipped)"
    )
    if skipped_reasons:
        logger.info(f"Skipped reasons: {dict(skipped_reasons)}")

    # Convert to DataFrame
    df = pd.DataFrame(aggregated_data)

    return df, skipped_reasons


def validate_output(df: pd.DataFrame, bucket: str):
    """
    Validate aggregated DataFrame before saving.

    Source: FeatureAggregationCHILD.md Section 6.3

    Args:
        df: Aggregated features DataFrame
        bucket: Bucket name (e.g., "18-33s")

    Raises:
        AssertionError: if output schema invalid
    """
    # 1. Check row count > 0
    assert len(df) > 0, "DataFrame has 0 rows (no valid videos processed)"

    # 2. Check column count matches expected for bucket
    expected_cols = EXPECTED_FEATURE_COUNTS[bucket]
    actual_cols = len(df.columns)

    assert actual_cols == expected_cols, \
        f"Column count mismatch: expected {expected_cols}, got {actual_cols}"

    # 3. Check required columns exist
    required_cols = ['video_id', 'create_time']
    missing_cols = [c for c in required_cols if c not in df.columns]

    assert len(missing_cols) == 0, f"Missing required columns: {missing_cols}"

    # 4. Check for completely null columns (indicates extraction error)
    null_cols = df.columns[df.isnull().all()].tolist()
    if len(null_cols) > 0:
        logger.warning(
            f"Found {len(null_cols)} completely null columns: {null_cols[:5]}... "
            "(may indicate feature extraction issues)"
        )

    # 5. Validate video_id uniqueness
    duplicate_ids = df['video_id'].duplicated().sum()
    assert duplicate_ids == 0, f"Found {duplicate_ids} duplicate video_ids in output"


def save_aggregated_csv(df: pd.DataFrame, output_path: Path):
    """
    Save DataFrame to CSV with atomic write pattern.

    Source: FeatureAggregationCHILD.md Section 2.3.4

    Args:
        df: Aggregated features DataFrame
        output_path: Final CSV path

    Raises:
        IOError: if write fails
    """
    temp_path = output_path.with_suffix('.tmp')

    try:
        # Write to temporary file first
        df.to_csv(temp_path, index=False)

        # Atomic rename (only if write succeeded)
        shutil.move(str(temp_path), str(output_path))

        logger.info(
            f"Created {output_path.name} - "
            f"{len(df)} rows × {len(df.columns)} columns"
        )

    finally:
        # Clean up temp file if rename failed
        if temp_path.exists():
            temp_path.unlink()


def aggregate_features(bucket_path: str) -> Tuple[Path, Path]:
    """
    Complete Stage 3 feature aggregation pipeline.

    Source: FeatureAggregationCHILD.md Appendix C

    Args:
        bucket_path: str, path to bucket directory (e.g., "bucket_18-33s")

    Returns:
        tuple: (csv_path, summary_path)

    Raises:
        ValueError: if validation fails or zero valid videos processed
    """
    start_time = time.time()
    bucket_path = Path(bucket_path)

    # Extract bucket name from path
    bucket = bucket_path.name.replace('bucket_', '')

    # ===== 1. Validate Dependencies =====
    logger.info(f"Stage 3 Feature Aggregation starting")
    logger.info(f"Bucket path: {bucket_path}")
    logger.info(f"Bucket: {bucket}")

    # Pre-flight validation - raises ValueError if fails
    total_files = validate_dependencies(bucket_path)
    logger.info(f"Found {total_files} temporal_windows_updated.json files")
    logger.info(f"Validation complete - ml_analysis/ directory writable")

    # ===== 2. Load and Extract Features from All Videos =====
    # process_bucket raises ValueError if zero valid videos
    # Let exceptions propagate to caller (orchestrator or main())
    df, skipped_reasons = process_bucket(bucket_path, bucket)

    # ===== 3. Validate Output Schema =====
    logger.info("Validating output schema")
    try:
        validate_output(df, bucket)
    except AssertionError as e:
        # Convert AssertionError to ValueError for consistent exception handling
        raise ValueError(f"Output validation failed: {e}") from e

    # ===== 4. Atomic Write to CSV =====
    ml_analysis_dir = bucket_path / "ml_analysis"
    output_path = ml_analysis_dir / OUTPUT_CSV

    # save_aggregated_csv raises IOError/OSError if write fails
    # Let exceptions propagate to caller
    save_aggregated_csv(df, output_path)

    # ===== 5. Generate Summary JSON =====
    end_time = time.time()

    summary = {
        "bucket_path": str(bucket_path),
        "bucket": bucket,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(end_time - start_time, 2),
        "input_files_found": total_files,
        "videos_processed": len(df),
        "videos_skipped": total_files - len(df),
        "skipped_reasons": dict(skipped_reasons),
        "output_csv": {
            "path": OUTPUT_CSV,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        },
        "stage_version": "3.0.0"
    }

    summary_path = ml_analysis_dir / SUMMARY_JSON
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Stage 3 complete - Duration: {summary['duration_seconds']}s")
    logger.info(f"Summary saved to: {summary_path}")

    # ===== 6. Write Checkpoint (for orchestrator skip logic) =====
    # Source: Stage 4-7 checkpoint pattern in rumiai_ml_batch.py
    # Checkpoint enables Stage 3 skip logic (resume without re-aggregating)
    checkpoint_dir = bucket_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "stage_3_checkpoint.json"

    checkpoint = {
        "stage": "feature_aggregation",
        "status": "completed",
        "total_videos": len(df),
        "output_files": [OUTPUT_CSV, SUMMARY_JSON],
        "completion_time": datetime.now(timezone.utc).isoformat(),
        "videos_processed": summary['videos_processed'],
        "videos_skipped": summary['videos_skipped'],
        "bucket": bucket,
        "duration_seconds": summary['duration_seconds'],
        "feature_count": len(df.columns)  # Add for orchestrator summary reporting
    }

    # Write checkpoint with graceful degradation
    # Philosophy: Checkpoint is performance optimization, CSV is source of truth
    # If checkpoint write fails, Stage 3 outputs are still valid
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint written: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Checkpoint write failed: {e}")
        logger.warning(
            "Stage 3 outputs ARE VALID (CSV and summary created successfully). "
            "Pipeline will continue using these outputs. "
            "NOTE: Stage 3 will re-run on next pipeline execution (checkpoint missing for skip logic)."
        )
        # Don't raise - graceful degradation allows pipeline to continue
        # CSV is the source of truth; checkpoint is only for performance optimization

    # ===== 7. Return Paths =====
    return output_path, summary_path


# ===== CLI ENTRY POINT =====

def main():
    """
    CLI entry point for Stage 3 feature aggregation.

    Catches exceptions from aggregate_features() and converts to exit codes:
    - Exit 0: Success
    - Exit 1: Validation failures (pre-flight, processing, output)
    - Exit 4: I/O failures (CSV write)
    - Exit 99: Unexpected errors or KeyboardInterrupt
    """
    parser = argparse.ArgumentParser(
        description="Stage 3: Feature Aggregation - Transform temporal window JSONs to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single bucket
  python3 scripts/stage3_aggregation.py --bucket-path="data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s"

  # Parallel processing (3 buckets simultaneously)
  python3 scripts/stage3_aggregation.py --bucket-path="...bucket_18-33s" &
  python3 scripts/stage3_aggregation.py --bucket-path="...bucket_33-60s" &
  python3 scripts/stage3_aggregation.py --bucket-path="...bucket_60-90s" &
        """
    )
    parser.add_argument(
        "--bucket-path",
        required=True,
        help="Path to bucket directory (e.g., data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s)"
    )

    args = parser.parse_args()

    try:
        csv_path, summary_path = aggregate_features(args.bucket_path)

        print(f"\n✓ Aggregated features saved to: {csv_path}")
        print(f"✓ Summary saved to: {summary_path}")
        sys.exit(0)

    except ValueError as e:
        # Pre-flight validation, processing, or output validation failures
        logger.error(f"Validation or processing failed: {e}")
        print(f"\n✗ Stage 3 failed: {e}")
        sys.exit(1)

    except (IOError, OSError) as e:
        # CSV write failures
        logger.error(f"I/O error: {e}")
        print(f"\n✗ Stage 3 failed: I/O error - {e}")
        sys.exit(4)

    except KeyboardInterrupt:
        logger.error("Stage 3 interrupted by user")
        print("\n✗ Stage 3 interrupted by user")
        sys.exit(99)

    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        print(f"\n✗ Stage 3 failed: Unexpected error - {type(e).__name__}: {e}")
        sys.exit(99)


if __name__ == "__main__":
    main()
