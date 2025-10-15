"""
Input/output validation for Stage 2.5

Source: FileOrganizationCHILDTI.md Section 5 (Validation Rules)
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# Constants
ALL_BUCKETS = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
SOURCE_DIR = "/home/jorge/rumiaifinal/insights/"


def validate_inputs(analysis_base: str, winning_buckets: List[str]) -> None:
    """
    Validate inputs before starting file organization.

    Args:
        analysis_base: str, path to analysis directory
        winning_buckets: list, bucket names from winner_analysis.json

    Raises:
        ValueError: if validation fails

    Source: FileOrganizationCHILDTI.md Section 5 (Input Validation)
    """
    # Validation 1: Check analysis_base exists
    if not os.path.exists(analysis_base):
        raise ValueError(
            f"Analysis base directory does not exist: {analysis_base}. "
            f"Did Foundation setup run?"
        )

    # Validation 2: Check winning_buckets is valid
    if not winning_buckets or len(winning_buckets) == 0:
        raise ValueError("No winning buckets provided. Check winner_analysis.json.")

    # Validation 3: Validate bucket names are in expected set
    for bucket in winning_buckets:
        if bucket not in ALL_BUCKETS:
            raise ValueError(f"Invalid bucket name: {bucket}. Expected one of {ALL_BUCKETS}")

    # Validation 4: Check source directory exists
    if not os.path.exists(SOURCE_DIR):
        raise ValueError(
            f"Source directory does not exist: {SOURCE_DIR}. Did Stage 2 complete?"
        )

    # Validation 5: Check write permissions to analysis_base
    test_file = f"{analysis_base}/test_write.tmp"
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise ValueError(f"No write permission to {analysis_base}: {e}")


def validate_output(analysis_base: str, winning_buckets: List[str], moved_count: int) -> None:
    """
    Validate output after file organization completes.

    Args:
        analysis_base: str, path to analysis directory
        winning_buckets: list, bucket names processed
        moved_count: int, number of files moved

    Source: FileOrganizationCHILDTI.md Section 5 (Output Validation)
    """
    # Validation 1: Check files exist in target locations
    total_organized = 0
    for bucket in winning_buckets:
        target_dir = f"{analysis_base}/buckets/bucket_{bucket}/analysis/insights/"
        if os.path.exists(target_dir):
            organized_count = len([f for f in os.listdir(target_dir) if f.endswith('.json')])
            total_organized += organized_count
            logger.info(f"Bucket {bucket}: {organized_count} files organized")

    # Validation 2: Log summary
    logger.info(f"Total files organized: {total_organized}")
    logger.info(f"Files moved this run: {moved_count}")

    # Validation 3: Warn if mismatch (some files may have been organized in previous run)
    if total_organized < moved_count:
        logger.warning(
            f"Organized count ({total_organized}) less than moved count ({moved_count}). "
            f"Some files may be missing."
        )
