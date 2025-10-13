"""
Foundation package for RumiAI ML Pipeline.

Provides shared infrastructure for CLI parsing, configuration management,
directory structure, and schema validation.

Source: FoundationCHILD.md
"""

from foundation.cli import parse_args, CLIArgs
from foundation.config import ConfigManager
from foundation.paths import PathBuilder, sanitize_target, sanitize_client_id
from foundation.buckets import assign_bucket, get_bucket_bounds
from foundation.schemas import Config, ApifyVideoMetadata, CheckpointSchema, CheckpointFailedVideo
from foundation.constants import (
    BUCKET_DEFINITIONS,
    VALID_ANALYSIS_TYPES,
    VALID_MODES,
    VALID_STRATEGIES,
    VALID_REPORT_TYPES,
    VALID_REPORT_AUDIENCES,
    DEFAULT_VIDEO_COUNTS,
    DEFAULT_ANALYSIS_MODES,
    DEFAULT_SELECTION_STRATEGIES,
    DEFAULT_REPORT_AUDIENCES,
    MIN_RECOMMENDED_N,
    ENGAGEMENT_SHARE_WEIGHT,
    CONFIG_FILENAME,
    CHECKPOINT_FILENAME,
    SELECTED_VIDEOS_FILENAME,
)

__version__ = "1.0.0"

__all__ = [
    # CLI
    "parse_args",
    "CLIArgs",
    # Configuration
    "ConfigManager",
    "Config",
    # Paths
    "PathBuilder",
    "sanitize_target",
    "sanitize_client_id",
    # Buckets
    "assign_bucket",
    "get_bucket_bounds",
    # Schemas
    "ApifyVideoMetadata",
    "CheckpointSchema",
    "CheckpointFailedVideo",
    # Constants
    "BUCKET_DEFINITIONS",
    "VALID_ANALYSIS_TYPES",
    "VALID_MODES",
    "VALID_STRATEGIES",
    "VALID_REPORT_TYPES",
    "VALID_REPORT_AUDIENCES",
    "DEFAULT_VIDEO_COUNTS",
    "DEFAULT_ANALYSIS_MODES",
    "DEFAULT_SELECTION_STRATEGIES",
    "DEFAULT_REPORT_AUDIENCES",
    "MIN_RECOMMENDED_N",
    "ENGAGEMENT_SHARE_WEIGHT",
    "CONFIG_FILENAME",
    "CHECKPOINT_FILENAME",
    "SELECTED_VIDEOS_FILENAME",
]
