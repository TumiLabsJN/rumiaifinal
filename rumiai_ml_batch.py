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
from pathlib import Path


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

        # ===== STAGE 2+: TODO =====
        print("\n" + "="*80)
        print("PIPELINE STATUS")
        print("="*80)
        print("✓ Stage 0: Foundation - COMPLETE")
        print("✓ Stage 1: Video Discovery & Selection - COMPLETE")
        print("⧗ Stage 2: Video Processing - TODO")
        print("⧗ Stage 3: Feature Aggregation - TODO")
        print("⧗ Stage 4: Feature Transformation - TODO")
        print("⧗ Stage 5: ML Model Training - TODO")
        print("⧗ Stage 6: ML Analysis Generation - TODO")
        print("⧗ Stage 7: LLM Report Generation - TODO")
        print("="*80)
        print()
        print("Stage 1 complete! Next: Implement Stage 2 (Video Processing)")
        print()

        logger.info("="*80)
        logger.info("PIPELINE EXECUTION COMPLETE (Stage 0-1)")
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
