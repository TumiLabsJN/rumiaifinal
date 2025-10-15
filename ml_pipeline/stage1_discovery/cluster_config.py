"""
Cluster Configuration Module

Handles loading and management of hashtag cluster configurations.

Source: HashtagVolumeV2_TI.md Section 3.1, 3.6
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

from .constants import CLUSTER_CONFIG_PATH_TEMPLATE, EXIT_CODE_CLUSTER_CONFIG_NOT_FOUND

logger = logging.getLogger(__name__)


def load_cluster_config(cluster_id: str) -> dict:
    """
    Load cluster configuration from JSON file.

    Source: HashtagVolumeV2_TI.md Section 3.1

    Args:
        cluster_id: str, cluster identifier (e.g., "nutrition")
                    Validation: alphanumeric + underscore only

    Returns:
        dict: Cluster configuration object
              Schema: ClusterConfigSchema (Section 2.2)

    Raises:
        FileNotFoundError: if cluster config file doesn't exist
        ValueError: if cluster config validation fails
        json.JSONDecodeError: if config file is invalid JSON
    """
    # Get project root directory (parent of rumiaifinal directory)
    project_root = Path(__file__).parent.parent.parent

    # Build config file path (use absolute path from project root)
    cluster_path = project_root / "config" / "hashtag_clusters" / f"{cluster_id}.json"

    logger.debug(f"Looking for cluster config at: {cluster_path}")

    # Check file exists
    if not cluster_path.exists():
        raise FileNotFoundError(
            f"Cluster config not found: {cluster_path}\n"
            f"Create cluster config with: python generate_cluster.py"
        )

    # Load JSON
    try:
        with open(cluster_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in cluster config: {cluster_path}\n"
            f"Error: {str(e)}",
            e.doc,
            e.pos
        )

    # Import validation function here to avoid circular imports
    from .cluster_validation import validate_cluster_config

    # Validate config schema
    validate_cluster_config(config, str(cluster_path))

    logger.info(f"Loaded cluster config: {cluster_id}")
    logger.info(f"  Primary: {config['primary_hashtag']}")
    logger.info(f"  Variants: {len(config['variant_hashtags'])} hashtags")

    total_hashtags = len(config['variant_hashtags']) + 1
    runs_per_hashtag = config['scrape_config']['runs_per_hashtag']
    total_scrapes = total_hashtags * runs_per_hashtag

    logger.info(f"  Scrape config: {runs_per_hashtag} runs × {total_hashtags} hashtags = {total_scrapes} scrapes")

    return config


def detect_target_type(target: str, analysis_type: str) -> Tuple[str, Optional[dict]]:
    """
    Detect if target is a cluster or single hashtag/profile.

    MODIFIES: VideoDiscoveryCHILDTI.md target handling logic
    - ADDED: Cluster detection (if target doesn't start with # or @)
    - ADDED: Cluster config loading
    - PRESERVED: Single hashtag/profile handling
    - ADDED: Error on single hashtag (deprecated per DECISION 6)

    Source: HashtagVolumeV2_TI.md Section 3.6

    Args:
        target: str, target from CLI parameter (e.g., "nutrition" or "#nutrition")
        analysis_type: str, "hashtag" or "competitor" or "creator"

    Returns:
        tuple:
            - target_type: str, "cluster" or "single"
            - config: dict, cluster config if cluster, None if single

    Raises:
        ValueError: if single hashtag used (deprecated per DECISION 6)
        FileNotFoundError: if cluster config not found
    """
    if analysis_type == "hashtag":
        if target.startswith("#"):
            # Single hashtag mode (DEPRECATED per DECISION 6)
            raise ValueError(
                f"Single hashtag scraping is deprecated as of 2025-10-10.\n"
                f"Please create a cluster configuration:\n"
                f"  1. Run: python generate_cluster.py\n"
                f"  2. Enter primary hashtag: {target[1:]}\n"
                f"  3. Configure cluster settings\n"
                f"  4. Run: python rumiai_ml_batch.py --client YOUR_CLIENT --target {target[1:]} --analysis-type hashtag\n\n"
                f"Rationale: Cluster strategy provides 2-3x more unique videos "
                f"with rich analytics for optimization."
            )
        else:
            # Cluster mode (target is cluster name)
            cluster_config = load_cluster_config(target)
            return ("cluster", cluster_config)

    else:
        # Competitor or Creator - single profile mode (unchanged)
        # Source: VideoDiscoveryCHILDTI.md (INHERITED)
        return ("single", None)


def get_all_hashtags(cluster_config: dict) -> list:
    """
    Get all hashtags from cluster config (primary + variants).

    Args:
        cluster_config: dict, loaded cluster configuration

    Returns:
        list: All hashtags in cluster
    """
    return [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']
