#!/usr/bin/env python3
"""
Stage 4: Feature Transformation

Transforms aggregated features into ML-ready formats for Random Forest and K-Means training.

Source: FeatureTransformationCHILD.md
Status: Production-Ready (25/25 unit tests passing)

Key Features:
- Video-Level RF: Gender encoding, temporal extraction, emotion one-hot, cross-window features (~146 features)
- Window-Level RF: Raw features per window (22 features each)
- Window-Level K-Means: Log+scale, shift+scale, label encoding (27 features each)
- Handles bucket-specific schemas (66/129/150 columns)
- Outputs 13 CSV files per bucket

Usage:
    python3 scripts/stage4_transformation.py --bucket-path="data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s"
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rumiai_v2.processors.feature_transformation import (
    transform_video_level_rf,
    transform_window_level_rf,
    transform_window_level_kmeans
)

# ===== LOGGING SETUP =====

def setup_logging():
    """Configure logging for Stage 4 transformation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ===== HELPER FUNCTIONS =====

def parse_bucket_name(bucket_path: Path) -> str:
    """Extract bucket name from path (e.g., 'bucket_18-33s' -> '18-33s')"""
    bucket_dir_name = bucket_path.name
    if bucket_dir_name.startswith('bucket_'):
        return bucket_dir_name.replace('bucket_', '')
    return bucket_dir_name

def get_window_names(bucket_name: str, num_middle: int) -> list:
    """Get list of window names for this bucket"""
    windows = ['hook']

    # Add middle segments
    if num_middle == 0:
        pass  # No middle segments (0-3s, 3-9s)
    elif bucket_name in ['9-13s', '13-18s']:
        windows.append('middle_aggregate')
    else:
        for i in range(1, num_middle + 1):
            windows.append(f'middle_{i}')

    # Add closing (except for 0-3s bucket)
    if bucket_name != '0-3s':
        windows.append('closing')

    return windows

def detect_bucket_structure(df: pd.DataFrame) -> Tuple[str, int]:
    """Detect bucket type and number of middle segments from DataFrame columns"""

    # Check for middle_aggregate (9-13s, 13-18s buckets)
    if 'middle_aggregate_average_face_size' in df.columns:
        return 'aggregate', 0

    # Count middle_N segments
    middle_count = 0
    for i in range(1, 10):  # Check up to 9 middle segments
        if f'middle_{i}_average_face_size' in df.columns:
            middle_count = i
        else:
            break

    return 'individual', middle_count

# ===== MAIN PROCESSING =====

def process_bucket(bucket_path: Path) -> Dict:
    """
    Process a single bucket: transform aggregated features into ML-ready formats.

    Args:
        bucket_path: Path to bucket directory (e.g., .../bucket_18-33s)

    Returns:
        Summary dict with processing statistics
    """
    start_time = time.time()
    bucket_name = parse_bucket_name(bucket_path)

    logger.info(f"Stage 4 Feature Transformation starting")
    logger.info(f"Bucket path: {bucket_path}")
    logger.info(f"Bucket: {bucket_name}")

    # ===== STEP 1: Validate inputs =====

    ml_analysis_dir = bucket_path / 'ml_analysis'
    aggregated_file = ml_analysis_dir / 'aggregated_features.csv'

    if not aggregated_file.exists():
        raise FileNotFoundError(
            f"Aggregated features not found: {aggregated_file}\n"
            f"Run Stage 3 first: python3 scripts/stage3_aggregation.py --bucket-path={bucket_path}"
        )

    logger.info(f"Loading aggregated features from: {aggregated_file}")

    # ===== STEP 2: Load data and config =====

    df = pd.read_csv(aggregated_file)
    num_videos = len(df)
    logger.info(f"Loaded {num_videos} videos")

    # Load config.json for strategy and video_count
    config_file = bucket_path.parent.parent / 'config.json'
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found at {config_file}")

    with open(config_file) as f:
        config = json.load(f)

    strategy = config.get('selection_strategy', 'contrastive')
    video_count = config.get('video_count', 50)
    logger.info(f"Config: strategy={strategy}, video_count={video_count}")

    # Detect bucket structure
    structure_type, num_middle = detect_bucket_structure(df)
    logger.info(f"Detected structure: {structure_type} (middle segments: {num_middle})")

    # ===== STEP 3: Transform Video-Level RF =====

    logger.info("Transforming video-level features for Random Forest...")
    try:
        rf_video_df = transform_video_level_rf(df, bucket_name, strategy, video_count)
        rf_video_file = ml_analysis_dir / 'rf_transformed.csv'
        rf_video_df.to_csv(rf_video_file, index=False)
        logger.info(f"✓ Video-level RF: {len(rf_video_df)} rows × {len(rf_video_df.columns)} columns -> {rf_video_file.name}")
    except ValueError as e:
        logger.error(f"Video-level RF transformation failed: {e}")
        raise

    # ===== STEP 4: Transform Window-Level RF =====

    logger.info("Transforming window-level features for Random Forest...")
    window_names = get_window_names(bucket_name, num_middle)

    rf_window_files = []
    for window_name in window_names:
        try:
            window_rf_df = transform_window_level_rf(df, window_name, strategy, video_count)
            window_rf_file = ml_analysis_dir / f'{window_name}_rf_transformed.csv'
            window_rf_df.to_csv(window_rf_file, index=False)
            rf_window_files.append(window_rf_file.name)
            logger.info(f"✓ {window_name} RF: {len(window_rf_df)} rows × {len(window_rf_df.columns)} columns")
        except ValueError as e:
            logger.error(f"Window RF transformation failed for {window_name}: {e}")
            raise

    # ===== STEP 5: Transform Window-Level K-Means =====

    logger.info("Transforming window-level features for K-Means...")

    km_window_files = []
    for window_name in window_names:
        try:
            window_km_df = transform_window_level_kmeans(df, window_name)
            window_km_file = ml_analysis_dir / f'{window_name}_km_transformed.csv'
            window_km_df.to_csv(window_km_file, index=False)
            km_window_files.append(window_km_file.name)
            logger.info(f"✓ {window_name} K-Means: {len(window_km_df)} rows × {len(window_km_df.columns)} columns")
        except ValueError as e:
            logger.error(f"Window K-Means transformation failed for {window_name}: {e}")
            raise

    # ===== STEP 6: Save summary =====

    duration = time.time() - start_time

    summary = {
        'bucket_path': str(bucket_path),
        'bucket': bucket_name,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': round(duration, 2),
        'input_file': aggregated_file.name,
        'videos_processed': num_videos,
        'structure_type': structure_type,
        'num_middle_segments': num_middle,
        'outputs': {
            'rf_video': {
                'file': 'rf_transformed.csv',
                'rows': len(rf_video_df),
                'columns': len(rf_video_df.columns)
            },
            'rf_windows': [
                {'window': w, 'file': f, 'columns': 22}
                for w, f in zip(window_names, rf_window_files)
            ],
            'km_windows': [
                {'window': w, 'file': f, 'columns': 27}
                for w, f in zip(window_names, km_window_files)
            ]
        },
        'stage_version': '4.0.0'
    }

    summary_file = ml_analysis_dir / 'transformation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Stage 4 complete - Duration: {duration:.2f}s")
    logger.info(f"Summary saved to: {summary_file}")

    return summary

# ===== CLI ENTRY POINT =====

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Stage 4: Feature Transformation - Transform aggregated features into ML-ready formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform features for a single bucket
  python3 scripts/stage4_transformation.py --bucket-path="data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s"

  # Process all buckets in a hashtag
  for bucket in bucket_18-33s bucket_13-18s bucket_60-90s; do
    python3 scripts/stage4_transformation.py --bucket-path="data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/$bucket"
  done

Output Files (per bucket):
  - rf_transformed.csv (~146 features for video-level RF training)
  - hook_rf_transformed.csv (22 features)
  - middle_*_rf_transformed.csv (22 features each)
  - closing_rf_transformed.csv (22 features)
  - hook_km_transformed.csv (27 features)
  - middle_*_km_transformed.csv (27 features each)
  - closing_km_transformed.csv (27 features)
  - transformation_summary.json
        """
    )

    parser.add_argument(
        '--bucket-path',
        type=str,
        required=True,
        help='Path to bucket directory (e.g., data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s)'
    )

    args = parser.parse_args()
    bucket_path = Path(args.bucket_path)

    # Validate bucket path
    if not bucket_path.exists():
        logger.error(f"Bucket path does not exist: {bucket_path}")
        return 1

    if not bucket_path.is_dir():
        logger.error(f"Bucket path is not a directory: {bucket_path}")
        return 1

    # Process bucket
    try:
        summary = process_bucket(bucket_path)

        # Print success summary
        print("\n" + "="*80)
        print("✓ STAGE 4 TRANSFORMATION COMPLETE")
        print("="*80)
        print(f"Bucket: {summary['bucket']}")
        print(f"Videos: {summary['videos_processed']}")
        print(f"Duration: {summary['duration_seconds']}s")
        print(f"\nOutputs saved to: {bucket_path}/ml_analysis/")
        print(f"  - rf_transformed.csv ({summary['outputs']['rf_video']['columns']} features)")
        print(f"  - {len(summary['outputs']['rf_windows'])} window RF files (22 features each)")
        print(f"  - {len(summary['outputs']['km_windows'])} window K-Means files (27 features each)")
        print(f"  - transformation_summary.json")
        print("="*80)

        return 0

    except Exception as e:
        logger.error(f"Stage 4 transformation failed: {e}", exc_info=True)
        print(f"\n✗ Stage 4 transformation failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
