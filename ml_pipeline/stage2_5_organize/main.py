"""
Main orchestration for Stage 2.5: File Organization

Coordinates all substeps (load winning buckets, build file list, detect duplicates, organize files).

Source: FileOrganizationCHILDTI.md
"""

import logging
from typing import Dict

from ml_pipeline.stage2_5_organize.file_organizer import (
    load_winning_buckets,
    build_file_list,
    detect_duplicates_across_buckets,
    organize_files_with_detection
)
from ml_pipeline.stage2_5_organize.validation import validate_inputs, validate_output

logger = logging.getLogger(__name__)


def stage_2_5_file_organization_main(analysis_base: str) -> Dict[str, any]:
    """
    Main orchestration function for Stage 2.5: File Organization.

    Coordinates all substeps in sequence:
    1. Load winning buckets from winner_analysis.json
    2. Build file list from Stage 2 checkpoints
    3. Detect duplicate video IDs across buckets
    4. Organize files with detection-based resume

    Args:
        analysis_base: str, path to analysis directory
                      Example: /data/clients/acme/hashtags/nutrition/top_contrastive/

    Returns:
        dict: Organization summary
            {
                "moved_count": int,
                "skipped_already_organized": int,
                "missing_count": int,
                "total_processed": int,
                "winning_buckets": list[str]
            }

    Raises:
        FileNotFoundError: if winner_analysis.json or checkpoints missing
        ValueError: if validation fails or data corruption detected

    Source: FileOrganizationCHILDTI.md Section 4 (Main Orchestration)
    """
    logger.info("Starting File Organization (Stage 2.5)")

    # Step 1: Load winning buckets from winner_analysis.json
    logger.info(f"Step 1: Loading winning buckets from {analysis_base}/winner_analysis.json")
    winning_buckets = load_winning_buckets(analysis_base)

    # Step 2: Validate inputs before processing
    logger.info("Step 2: Validating inputs")
    validate_inputs(analysis_base, winning_buckets)

    # Step 3: Build file list from Stage 2 checkpoints
    logger.info("Step 3: Building file list from Stage 2 checkpoints")
    files_to_process = build_file_list(analysis_base, winning_buckets)

    # Handle empty file list gracefully
    if len(files_to_process) == 0:
        logger.warning("No files to organize. All checkpoints have 0 completed videos.")
        return {
            "moved_count": 0,
            "skipped_already_organized": 0,
            "missing_count": 0,
            "total_processed": 0,
            "winning_buckets": winning_buckets
        }

    # Step 4: Detect duplicate video IDs across buckets
    logger.info("Step 4: Validating for duplicate video IDs across buckets")
    detect_duplicates_across_buckets(files_to_process)

    # Step 5: Organize files with detection-based resume
    logger.info("Step 5: Starting file organization")
    summary = organize_files_with_detection(files_to_process)

    # Step 6: Validate outputs
    logger.info("Step 6: Validating stage outputs")
    validate_output(analysis_base, winning_buckets, summary['moved_count'])

    # Step 7: Return summary with winning buckets
    logger.info("File Organization (Stage 2.5) complete")
    summary['winning_buckets'] = winning_buckets

    return summary
