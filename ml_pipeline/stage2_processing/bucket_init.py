"""
Bucket directory initialization for Stage 2: Video Processing

Creates all 8 bucket directories with complete subdirectory structure.

Source: VideoProcessingTI.md Section 4 (Function 1)
"""

import os
import logging
from typing import Dict

from ml_pipeline.stage2_processing.utils import get_bucket_path

logger = logging.getLogger(__name__)


def initialize_bucket_directories(config: dict) -> Dict[str, str]:
    """
    Create all 8 bucket directories with complete subdirectory structure.

    Args:
        config: dict, loaded from config.json (FoundationCHILD Section 5.1)

    Returns:
        dict: Created paths mapped to bucket names

    Raises:
        OSError: If directory creation fails (permissions, disk space)

    Source: VideoProcessingTI.md Section 4 (Function 1)
    """

    # Step 1: Define all 8 bucket names
    BUCKET_NAMES = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]

    # Step 2: Construct base path from config
    data_root = os.getenv('DATA_ROOT', '/data')
    analysis_base = (
        f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/"
        f"{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"
    )

    # Step 3: Define complete subdirectory structure per bucket (15 subdirectories)
    SUBDIRECTORIES = [
        "videos/", "analysis/", "analysis/insights/", "analysis/unified/",
        "analysis/service_debug/", "validation/", "flagged_videos/", "ml_analysis/",
        "models/", "llm_reports/", "llm_reports/analysis/", "llm_reports/formatted/",
        "reports/", "checkpoints/", "logs/"
    ]

    # Step 4: Initialize tracking dictionary
    created_paths = {}

    # Step 5: Create all 8 buckets with full subdirectory structure
    for bucket_name in BUCKET_NAMES:
        bucket_path = f"{analysis_base}buckets/bucket_{bucket_name}/"
        logger.info(f"Creating bucket directory structure: bucket_{bucket_name}")

        try:
            os.makedirs(bucket_path, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create bucket directory {bucket_path}: {e}")

        for subdir in SUBDIRECTORIES:
            subdir_path = f"{bucket_path}{subdir}"
            try:
                os.makedirs(subdir_path, exist_ok=True)
            except OSError as e:
                raise OSError(f"Failed to create subdirectory {subdir_path}: {e}")

        created_paths[bucket_name] = bucket_path
        logger.debug(f"  ✓ Created {len(SUBDIRECTORIES)} subdirectories for bucket_{bucket_name}")

    logger.info(f"✓ Successfully created all 8 bucket directories with complete subdirectory structure")
    return created_paths
