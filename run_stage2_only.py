#!/usr/bin/env python3
"""
Stage 2-Only Script - Bypass Stage 1

This script runs Stage 2 video processing using existing Stage 1 data.
Use this when Stage 1 data already exists and you only need to process videos.

Usage:
    python3 run_stage2_only.py

Expects existing data at:
    /data/clients/test_final/hashtags/test_vitamin/top_contrastive/
    ├── winner_analysis.json
    ├── config.json
    └── buckets/bucket_*/selected_videos.json
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

from ml_pipeline.stage2_processing import stage_2_video_processing_main
from ml_pipeline.stage2_5_organize import stage_2_5_file_organization_main


def setup_logging():
    """Setup logging for Stage 2-only run"""
    data_root = os.getenv("DATA_ROOT", str(Path(__file__).parent / "data"))
    log_dir = Path(data_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"stage2_only_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Stage 2-only logging initialized: {log_file}")
    return logger


def main():
    """Run Stage 2 + 2.5 using existing Stage 1 data"""
    try:
        print("="*80)
        print("STAGE 2-ONLY: VIDEO PROCESSING (Bypassing Stage 1)")
        print("="*80)
        print()

        logger = setup_logging()

        # Define the analysis base path
        data_root = Path(os.getenv("DATA_ROOT", str(Path(__file__).parent / "data")))
        analysis_base = data_root / "clients/test_final/hashtags/test_vitamin/top_contrastive"

        print(f"Analysis base: {analysis_base}")
        logger.info(f"Analysis base: {analysis_base}")

        # Verify required files exist
        winner_analysis_path = analysis_base / "winner_analysis.json"
        config_path = analysis_base / "config.json"

        if not winner_analysis_path.exists():
            print(f"✗ ERROR: {winner_analysis_path} not found")
            print("Stage 1 data is missing. Please run the full pipeline first.")
            return 1

        if not config_path.exists():
            print(f"✗ ERROR: {config_path} not found")
            return 1

        print(f"✓ Found winner_analysis.json")
        print(f"✓ Found config.json")

        # Load winner analysis
        with open(winner_analysis_path) as f:
            winner_analysis = json.load(f)

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        winning_buckets = winner_analysis['top_3_buckets']
        print(f"\nWinning buckets: {', '.join(winning_buckets)}")
        logger.info(f"Processing {len(winning_buckets)} winning buckets: {winning_buckets}")

        # ===== STAGE 2: VIDEO PROCESSING =====
        print("\n" + "="*80)
        print("STAGE 2: VIDEO PROCESSING")
        print("="*80)

        stage2_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 2 for bucket: {bucket_name}")
            print(f"\n--- Processing bucket: {bucket_name} ---")

            # Load selected videos for this bucket
            bucket_videos_path = analysis_base / f"buckets/bucket_{bucket_name}/selected_videos.json"
            if not bucket_videos_path.exists():
                logger.warning(f"No selected videos found for bucket {bucket_name}, skipping")
                print(f"⚠️  No selected videos found for {bucket_name}, skipping")
                continue

            # Load videos with the FIX applied
            with open(bucket_videos_path) as f:
                video_data = json.load(f)
                video_list = video_data['videos']  # Extract just the videos array

            print(f"✓ Loaded {len(video_list)} videos")
            print(f"Processing {len(video_list)} videos for bucket {bucket_name}...")

            # Run Stage 2 video processing for this bucket
            try:
                summary = stage_2_video_processing_main(
                    config=config,
                    video_list=video_list,
                    bucket_name=bucket_name,
                    enable_pause_support=False  # Use video_processor.py with working hybrid approach
                )

                stage2_summaries[bucket_name] = summary
                logger.info(f"Bucket {bucket_name} complete: {summary['completed']}/{summary['total']} videos processed")
                print(f"✓ Bucket {bucket_name}: {summary['completed']}/{summary['total']} videos processed")
                if summary['failed'] > 0:
                    print(f"  ⚠️  {summary['failed']} videos failed")

            except Exception as e:
                logger.error(f"Stage 2 failed for bucket {bucket_name}: {e}", exc_info=True)
                print(f"✗ Bucket {bucket_name} failed: {e}")
                # Continue with other buckets
                continue

        logger.info("Stage 2 completed for all buckets")
        print("\n✓ Stage 2: Video Processing - COMPLETE")

        # Log Stage 2 summary
        total_videos = sum(s['total'] for s in stage2_summaries.values())
        completed_videos = sum(s['completed'] for s in stage2_summaries.values())
        failed_videos = sum(s['failed'] for s in stage2_summaries.values())
        logger.info(f"Stage 2 Summary: {completed_videos}/{total_videos} videos processed, {failed_videos} failed")
        print(f"Summary: {completed_videos}/{total_videos} videos processed, {failed_videos} failed")

        # ===== STAGE 2.5: FILE ORGANIZATION =====
        logger.info("Starting Stage 2.5: File Organization")
        print("\n" + "="*80)
        print("STAGE 2.5: FILE ORGANIZATION")
        print("="*80)

        try:
            organization_summary = stage_2_5_file_organization_main(
                analysis_base=str(analysis_base)
            )

            logger.info("Stage 2.5 completed successfully")
            logger.info(f"Moved: {organization_summary['moved_count']} files")
            logger.info(f"Skipped: {organization_summary['skipped_already_organized']} files")
            logger.info(f"Missing: {organization_summary['missing_count']} files")

            print(f"\n✓ Stage 2.5: File Organization - COMPLETE")
            print(f"  Organized {organization_summary['moved_count']} temporal_windows files")
            if organization_summary['skipped_already_organized'] > 0:
                print(f"  Skipped {organization_summary['skipped_already_organized']} files (already organized)")
            if organization_summary['missing_count'] > 0:
                print(f"  ⚠️  {organization_summary['missing_count']} files missing")

        except Exception as e:
            logger.error(f"Stage 2.5 failed: {e}", exc_info=True)
            print(f"\n✗ Stage 2.5 failed: {e}")
            return 1

        # ===== FINAL STATUS =====
        print("\n" + "="*80)
        print("STAGE 2-ONLY EXECUTION COMPLETE")
        print("="*80)
        print(f"✅ Processed {completed_videos}/{total_videos} videos across {len(winning_buckets)} buckets")
        print(f"   Output location: {analysis_base}/buckets/bucket_*/analysis/insights/")
        print("="*80)

        logger.info("="*80)
        logger.info("STAGE 2-ONLY EXECUTION COMPLETE")
        logger.info("="*80)

        return 0

    except KeyboardInterrupt:
        print("\n\n✗ Stage 2 interrupted by user (Ctrl+C)")
        return 130

    except Exception as e:
        print(f"\n✗ Stage 2 failed: {e}")
        if 'logger' in locals():
            logger.error(f"Stage 2 failed: {e}", exc_info=True)
        else:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
