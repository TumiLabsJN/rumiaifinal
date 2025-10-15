#!/usr/bin/env python3
"""
RumiAI ML Batch Processing Pipeline - Main Entry Point

Orchestrates the complete ML pipeline from video discovery to report generation.

Usage:
    python rumiai_ml_batch.py --client acme_corp --target "#nutrition"

For full help:
    python rumiai_ml_batch.py --help

Source: VideoDiscoveryCHILDTI.md + FoundationCHILD.md
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
def load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

load_env()

from foundation.cli import parse_args
from foundation.config import ConfigManager
from foundation.paths import PathBuilder, sanitize_client_id
from ml_pipeline.stage1_discovery import VideoDiscovery
from ml_pipeline.stage2_processing import stage_2_video_processing_main
from ml_pipeline.stage2_5_organize import stage_2_5_file_organization_main
from pathlib import Path
import json


def setup_logging(client_id: str, target: str):
    """
    Configure logging for the ML pipeline.

    Logs to both console and file.
    """
    # Create logs directory (use local path if /data not accessible)
    data_root = os.getenv("DATA_ROOT", str(Path(__file__).parent / "data"))
    log_dir = Path(data_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_target = target.replace('#', '').replace('@', '')
    log_file = log_dir / f"rumiai_ml_{client_id}_{sanitized_target}_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")

    return logger


def main():
    """
    Main entry point for RumiAI ML Batch Pipeline.

    Pipeline Stages:
    - Stage 0: Foundation (CLI parsing, config, directory setup)
    - Stage 1: Video Discovery & Selection
    - Stage 2: Video Processing (TODO)
    - Stage 3: Feature Aggregation (TODO)
    - Stage 4: Feature Transformation (TODO)
    - Stage 5: ML Model Training (TODO)
    - Stage 6: ML Analysis Generation (TODO)
    - Stage 7: LLM Report Generation (TODO)
    """
    try:
        # ===== STAGE 0: FOUNDATION =====
        print("="*80)
        print("RumiAI ML BATCH PIPELINE")
        print("="*80)
        print()

        # Stage 0.1: Parse CLI arguments
        print("Stage 0: Foundation - Parsing CLI arguments...")
        cli_args = parse_args()
        print(f"✓ Parsed CLI arguments")

        # Stage 0.2: Validate environment
        print("Stage 0: Foundation - Validating environment...")
        apify_api_key = os.getenv("APIFY_API_KEY")
        if not apify_api_key:
            print("✗ ERROR: APIFY_API_KEY environment variable not set")
            print("  Obtain API key from: https://console.apify.com/account/integrations")
            print("  Set with: export APIFY_API_KEY='your_key_here'")
            return 1
        print(f"✓ Environment validated")

        # Stage 0.3: Setup logging
        logger = setup_logging(cli_args.client, cli_args.target)
        logger.info("="*80)
        logger.info("RumiAI ML BATCH PIPELINE STARTED")
        logger.info("="*80)

        # Stage 0.4: Create Config object and paths
        print("Stage 0: Foundation - Creating configuration and paths...")
        config = ConfigManager.from_cli_args(cli_args)

        path_builder = PathBuilder()
        analysis_base = path_builder.get_target_dir(
            client_id=sanitize_client_id(cli_args.client),
            analysis_type=cli_args.analysis_type,
            target=cli_args.target,
            analysis_mode=cli_args.analysis_mode,
            selection_strategy=cli_args.selection_strategy
        )
        analysis_base.mkdir(parents=True, exist_ok=True)

        print(f"✓ Created directory structure: {analysis_base}")
        logger.info(f"Created directory structure: {analysis_base}")

        # Stage 0.5: Save configuration
        print("Stage 0: Foundation - Saving configuration...")
        config_path = analysis_base / "config.json"
        ConfigManager.save(config, config_path)
        print(f"✓ Saved configuration: {config_path}")
        logger.info(f"Saved configuration: {config_path}")

        print()

        # ===== STAGE 1: VIDEO DISCOVERY & SELECTION =====
        logger.info("Starting Stage 1: Video Discovery & Selection")

        video_discovery = VideoDiscovery(
            config=config.model_dump(),  # Convert pydantic model to dict
            apify_api_key=apify_api_key,
            path_builder=path_builder
        )

        exit_code = video_discovery.run()

        if exit_code != 0:
            logger.error(f"Stage 1 failed with exit code {exit_code}")
            return exit_code

        logger.info("Stage 1 completed successfully")

        # ===== STAGE 2: VIDEO PROCESSING =====
        logger.info("Starting Stage 2: Video Processing")
        print("\n" + "="*80)
        print("STAGE 2: VIDEO PROCESSING")
        print("="*80)

        # Load winner_analysis.json to get winning buckets
        winner_analysis_path = analysis_base / "winner_analysis.json"
        if not winner_analysis_path.exists():
            logger.error("winner_analysis.json not found - Stage 1 may have failed")
            return 1

        with open(winner_analysis_path) as f:
            winner_analysis = json.load(f)

        winning_buckets = winner_analysis['top_3_buckets']
        logger.info(f"Processing {len(winning_buckets)} winning buckets: {winning_buckets}")
        print(f"\nWinning buckets: {', '.join(winning_buckets)}")

        # Process each winning bucket
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

            with open(bucket_videos_path) as f:
                video_data = json.load(f)
                video_list = video_data['videos']  # Extract just the videos array

            print(f"Processing {len(video_list)} videos for bucket {bucket_name}...")

            # Run Stage 2 video processing for this bucket
            try:
                summary = stage_2_video_processing_main(
                    config=config.model_dump(),
                    video_list=video_list,
                    bucket_name=bucket_name,
                    enable_pause_support=True  # Allow Ctrl+C graceful pause
                )

                stage2_summaries[bucket_name] = summary
                logger.info(f"Bucket {bucket_name} complete: {summary['completed']}/{summary['total']} videos processed")
                print(f"✓ Bucket {bucket_name}: {summary['completed']}/{summary['total']} videos processed")
                if summary['failed'] > 0:
                    print(f"  ⚠️  {summary['failed']} videos failed")

            except Exception as e:
                logger.error(f"Stage 2 failed for bucket {bucket_name}: {e}", exc_info=True)
                print(f"✗ Bucket {bucket_name} failed: {e}")
                # Continue with other buckets (skip-on-fail policy)
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
            # Stage 2.5 organizes all temporal_windows files from flat directory
            # into bucket-specific directories
            organization_summary = stage_2_5_file_organization_main(
                analysis_base=str(analysis_base)
            )

            logger.info("Stage 2.5 completed successfully")
            logger.info(f"Moved: {organization_summary['moved_count']} files")
            logger.info(f"Skipped (already organized): {organization_summary['skipped_already_organized']} files")
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
        print("PIPELINE STATUS")
        print("="*80)
        print("✓ Stage 0: Foundation - COMPLETE")
        print("✓ Stage 1: Video Discovery & Selection - COMPLETE")
        print("✓ Stage 2: Video Processing - COMPLETE")
        print("✓ Stage 2.5: File Organization - COMPLETE")
        print("⧗ Stage 3: Feature Aggregation - TODO")
        print("⧗ Stage 4: Feature Transformation - TODO")
        print("⧗ Stage 5: ML Model Training - TODO")
        print("⧗ Stage 6: ML Analysis Generation - TODO")
        print("⧗ Stage 7: LLM Report Generation - TODO")
        print("="*80)
        print()
        print(f"✅ Stages 0-2.5 complete!")
        print(f"   Processed {completed_videos} videos across {len(winning_buckets)} buckets")
        print(f"   Output location: {analysis_base}")
        print()

        logger.info("="*80)
        logger.info("PIPELINE EXECUTION COMPLETE (Stage 0-2.5)")
        logger.info("="*80)

        return 0

    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user (Ctrl+C)")
        return 130

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        if 'logger' in locals():
            logger.error(f"Pipeline failed: {e}", exc_info=True)
        else:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
