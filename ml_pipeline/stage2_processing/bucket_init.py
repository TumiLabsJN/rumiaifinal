"""
Bucket directory initialization for Stage 2: Video Processing

Ensures required bucket directory structure exists for processing.

DESIGN PHILOSOPHY - Defensive Programming:
----------------------------------------------
This module implements a "defensive" approach where Stage 2 ensures
its prerequisites exist, rather than assuming Stage 1 created them.

Why defensive instead of fail-fast?
  1. Resilience: Recovers from partial Stage 1 failures automatically
  2. Independent operation: Stage 2 can run without Stage 1 (testing, reprocessing)
  3. Self-healing: Fixes missing subdirectories from Stage 1 inconsistencies
  4. Idempotency: exist_ok=True makes creation safe to repeat

Stage 1 vs Stage 2 bucket creation:
  - Stage 1: Creates directories for winning buckets (3 buckets)
  - Stage 2: Ensures ITS bucket exists (1 bucket per call)
  - Both creating the same directories is FINE (idempotent)
  - Think: "defensive redundancy" not "wasteful duplication"

Alternative considered:
  - Validation-only approach (fail if bucket missing)
  - Rejected because: Breaks independent Stage 2 execution, painful testing,
    forces full Stage 1 rerun for minor directory issues

Real-world analogy:
  - Database migrations use "CREATE TABLE IF NOT EXISTS"
  - Kubernetes uses "livenessProbe" auto-restart
  - Systems design for recovery, not assume perfection

Source: VideoProcessingTI.md Section 4 (Function 1)
"""

import os
import logging

from ml_pipeline.stage2_processing.utils import get_bucket_path

logger = logging.getLogger(__name__)

# Valid bucket names (RumiAI duration buckets)
VALID_BUCKETS = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]

# Complete subdirectory structure per bucket (15 subdirectories)
SUBDIRECTORIES = [
    "videos/",
    "analysis/",
    "analysis/insights/",
    "analysis/unified/",
    "analysis/service_debug/",
    "validation/",
    "flagged_videos/",
    "ml_analysis/",
    "models/",
    "llm_reports/",
    "llm_reports/analysis/",
    "llm_reports/formatted/",
    "reports/",
    "checkpoints/",
    "logs/"
]


def validate_bucket_name(bucket_name: str):
    """
    Validate bucket name is in expected format.

    Args:
        bucket_name: Bucket name to validate (e.g., "18-33s")

    Raises:
        ValueError: If bucket name invalid
    """
    if bucket_name not in VALID_BUCKETS:
        raise ValueError(
            f"Invalid bucket name: '{bucket_name}'\n"
            f"Expected one of: {VALID_BUCKETS}\n\n"
            f"Bucket names must match RumiAI duration buckets defined in "
            f"config/bucket_definitions.py"
        )

    logger.debug(f"Bucket name validated: {bucket_name}")


def ensure_bucket_exists(config: dict, bucket_name: str) -> str:
    """
    Ensure directory structure exists for a single bucket.

    Defensive programming: Creates directories if missing, no-op if they exist.
    Idempotent: Safe to call multiple times (exist_ok=True).
    Focused: Only creates the bucket being processed (not all 8).

    This function is called by Stage 2 for each bucket being processed.
    Stage 1 also creates bucket directories, which is fine - idempotency
    means both stages can safely create directories without conflicts.

    Args:
        config: dict, loaded from config.json (FoundationCHILD Section 5.1)
        bucket_name: str, single bucket to ensure exists (e.g., "18-33s")

    Returns:
        str: Path to bucket directory

    Raises:
        ValueError: If bucket_name invalid
        OSError: If directory creation fails (permissions, disk space)

    Source: VideoProcessingTI.md Section 4 (Function 1)
    Modified: Changed from creating all 8 buckets to creating only specified bucket

    Design rationale:
        - Stage 2 is called per-bucket, so it should only touch that bucket
        - Defensive: Recovers from partial Stage 1 failures or manual deletions
        - Resilient: Supports reprocessing individual buckets
        - Testable: Can run Stage 2 tests without Stage 1 setup
    """

    # Step 0: Validate bucket name format
    validate_bucket_name(bucket_name)

    # Step 1: Construct bucket path using PathBuilder
    # REFACTORED (FixCheckpointBug.md): Use PathBuilder for consistent path construction
    # This ensures target sanitization (strips @ and #) matches Stage 0 behavior
    from foundation.paths import PathBuilder

    path_builder = PathBuilder()
    bucket_path_obj = path_builder.get_bucket_path(
        client_id=config['client_id'],
        analysis_type=config['analysis_type'],
        target=config['target'],
        analysis_mode=config['analysis_mode'],
        selection_strategy=config['selection_strategy'],
        bucket_name=bucket_name
    )

    # Convert Path to string for os.makedirs compatibility
    bucket_path = str(bucket_path_obj) + "/"

    # Step 2: Create bucket base directory (idempotent)
    try:
        os.makedirs(bucket_path, exist_ok=True)
        logger.debug(f"Ensured bucket directory exists: bucket_{bucket_name}")
    except OSError as e:
        raise OSError(
            f"Failed to create bucket directory {bucket_path}: {e}\n"
            f"Check disk space and permissions."
        )

    # Step 3: Create all subdirectories (idempotent)
    for subdir in SUBDIRECTORIES:
        subdir_path = f"{bucket_path}{subdir}"
        try:
            os.makedirs(subdir_path, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Failed to create subdirectory {subdir_path}: {e}\n"
                f"Check disk space and permissions."
            )

    # Step 4: Log success
    logger.info(f"✓ Bucket directory ready: bucket_{bucket_name}")
    logger.debug(f"  Path: {bucket_path}")
    logger.debug(f"  Subdirectories: {len(SUBDIRECTORIES)} created/verified")

    return bucket_path
