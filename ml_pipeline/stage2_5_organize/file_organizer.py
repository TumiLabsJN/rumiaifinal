"""
Core file organization functions for Stage 2.5

Implements checkpoint-driven batch file organization with detection-based resume.

Source: FileOrganizationCHILDTI.md Section 4 (Algorithmic Specifications)
"""

import os
import json
import shutil
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Constants from FileOrganizationCHILDTI.md Section 9
SOURCE_DIR = "/home/jorge/rumiaifinal/insights/"
ALL_BUCKETS = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]


def load_winning_buckets(analysis_base: str) -> List[str]:
    """
    Load winning buckets from winner_analysis.json.

    Args:
        analysis_base: str, path to analysis directory
                      Example: /data/clients/acme/hashtags/nutrition/top_contrastive/

    Returns:
        list: winning bucket names (e.g., ["18-33s", "33-60s", "13-18s"])

    Raises:
        FileNotFoundError: if winner_analysis.json doesn't exist
        ValueError: if file is corrupted or has invalid schema

    Source: FileOrganizationCHILDTI.md Section 4 (Function 1)
    """
    # Step 1: Construct winner_analysis.json path
    winner_analysis_path = f"{analysis_base}/winner_analysis.json"

    # Step 2: Validate file exists
    if not os.path.exists(winner_analysis_path):
        raise FileNotFoundError(
            f"winner_analysis.json not found at:\n"
            f"  {winner_analysis_path}\n\n"
            f"This file is created by Stage 1.3 (Winner Analysis).\n"
            f"Stage 2.5 requires this file to know which buckets to organize.\n\n"
            f"Solutions:\n"
            f"  1. Complete Stage 1 (Video Discovery & Winner Analysis)\n"
            f"  2. Check if Stage 1 completed successfully\n"
            f"  3. Verify analysis_base path is correct: {analysis_base}"
        )

    # Step 3: Load JSON file
    with open(winner_analysis_path) as f:
        winner_analysis = json.load(f)

    # Step 4: Validate schema - check 'top_3_buckets' field exists
    if 'top_3_buckets' not in winner_analysis:
        raise ValueError(f"winner_analysis.json missing 'top_3_buckets' field")

    # Step 5: Validate 'top_3_buckets' is a list
    if not isinstance(winner_analysis['top_3_buckets'], list):
        raise TypeError(
            f"'top_3_buckets' must be list, got {type(winner_analysis['top_3_buckets'])}"
        )

    # Step 6: Validate list is not empty
    if len(winner_analysis['top_3_buckets']) == 0:
        raise ValueError("'top_3_buckets' is empty - no winning buckets identified")

    # Step 7: Log success and return
    logger.info(f"Loaded winning buckets: {winner_analysis['top_3_buckets']}")
    return winner_analysis['top_3_buckets']


def validate_checkpoint(checkpoint: Dict[str, Any], bucket: str) -> List[str]:
    """
    Validate checkpoint schema and extract completed_video_ids.

    Args:
        checkpoint: dict, loaded from stage_2_checkpoint.json
        bucket: str, bucket name for error messages

    Returns:
        list: completed_video_ids to process

    Raises:
        ValueError: if checkpoint schema is invalid

    Source: FileOrganizationCHILDTI.md Section 4 (Function 2 - validate_checkpoint helper)
    """
    # Step 1: Strict schema validation - check required fields
    required_fields = ['stage', 'bucket', 'completed_video_ids', 'status', 'total_videos']
    missing = [f for f in required_fields if f not in checkpoint]

    if missing:
        raise ValueError(
            f"Checkpoint for {bucket} has invalid schema (missing {missing}). "
            f"Re-run Stage 2 to regenerate checkpoint."
        )

    # Step 2: Validate field types
    if not isinstance(checkpoint['completed_video_ids'], list):
        raise ValueError(f"Checkpoint for {bucket}: 'completed_video_ids' must be list")

    # Step 3: Allow partial completion - warn if status != "completed"
    if checkpoint['status'] != 'completed':
        logger.warning(
            f"Checkpoint for {bucket} status is '{checkpoint['status']}' (not 'completed'). "
            f"Processing {len(checkpoint['completed_video_ids'])}/{checkpoint['total_videos']} videos."
        )

    # Step 4: Handle zero completions gracefully
    if len(checkpoint['completed_video_ids']) == 0:
        logger.info(f"Bucket {bucket} has 0 completed videos. Skipping bucket.")
        return []

    # Step 5: Return completed video IDs
    return checkpoint['completed_video_ids']


def build_file_list(analysis_base: str, winning_buckets: List[str]) -> List[Dict[str, str]]:
    """
    Build list of files to organize from Stage 2 checkpoints.

    Args:
        analysis_base: str, path to analysis directory
        winning_buckets: list, bucket names from winner_analysis.json

    Returns:
        list: file info dicts with keys: video_id, bucket, source_path, target_path

    Source: FileOrganizationCHILDTI.md Section 4 (Function 2)
    """
    files_to_process = []

    # Step 1: Iterate through each winning bucket
    for bucket in winning_buckets:
        # Step 2: Construct checkpoint path for this bucket
        checkpoint_path = f"{analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json"

        # Step 3: Validate checkpoint file exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint missing for bucket {bucket}: {checkpoint_path}")
            raise FileNotFoundError(
                f"Checkpoint not found for bucket {bucket}. Did Stage 2 complete?"
            )

        # Step 4: Load checkpoint JSON
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Step 5: Validate checkpoint schema and extract video IDs
        video_ids = validate_checkpoint(checkpoint, bucket)

        # Step 6: Skip if no completed videos
        if len(video_ids) == 0:
            logger.info(f"Bucket {bucket} has 0 completed videos. Skipping.")
            continue

        # Step 7: Build file info for each video
        for video_id in video_ids:
            # Construct source path (flat directory)
            source_path = f"{SOURCE_DIR}{video_id}_temporal_windows_updated.json"

            # Construct target path (bucket-specific directory)
            target_path = f"{analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json"

            # Add to file list
            files_to_process.append({
                'video_id': video_id,
                'bucket': bucket,
                'source_path': source_path,
                'target_path': target_path
            })

    # Step 8: Log summary
    logger.info(f"Built file list: {len(files_to_process)} files across {len(winning_buckets)} buckets")
    return files_to_process


def detect_duplicates_across_buckets(files_to_process: List[Dict[str, str]]) -> None:
    """
    Detect if same video_id appears in multiple buckets.

    Args:
        files_to_process: list of dict with keys: video_id, bucket, source_path

    Raises:
        ValueError: if duplicate video_id detected

    Source: FileOrganizationCHILDTI.md Section 4 (Function 3)
    """
    # Step 1: Initialize tracking dictionary
    video_id_to_buckets = {}

    # Step 2: Iterate through all files to process
    for file_info in files_to_process:
        video_id = file_info['video_id']
        bucket = file_info['bucket']

        # Step 3: Check if video_id already seen
        if video_id in video_id_to_buckets:
            previous_bucket = video_id_to_buckets[video_id]

            # Step 4: Duplicate detected - raise error with detailed message
            raise ValueError(
                f"Video ID '{video_id}' appears in multiple buckets:\n"
                f"  - Bucket: {previous_bucket}\n"
                f"  - Bucket: {bucket}\n\n"
                f"This indicates checkpoint corruption or Stage 2 bug.\n"
                f"Each video should belong to exactly one bucket based on duration.\n\n"
                f"Solutions:\n"
                f"  1. Re-run Stage 2 to regenerate checkpoints\n"
                f"  2. Manually inspect checkpoints and remove duplicate entries"
            )

        # Step 5: Record video_id → bucket mapping
        video_id_to_buckets[video_id] = bucket

    # Step 6: Log validation success
    logger.info(f"Validation passed: {len(video_id_to_buckets)} unique videos")


def organize_files_with_detection(files_to_process: List[Dict[str, str]]) -> Dict[str, int]:
    """
    Organize files with automatic resume detection (no checkpoint needed).

    Args:
        files_to_process: list of dict with keys: video_id, bucket, source_path, target_path

    Returns:
        dict: Summary statistics (moved_count, skipped_count, missing_count)

    Source: FileOrganizationCHILDTI.md Section 4 (Function 4)
    """
    # Step 1: Initialize counters
    moved_count = 0
    skipped_already_organized = 0
    missing_count = 0

    # Step 2: Iterate through each file to process
    for file_info in files_to_process:
        source = file_info['source_path']
        target = file_info['target_path']
        video_id = file_info['video_id']
        bucket = file_info['bucket']

        # Step 3: Check file existence states
        source_exists = os.path.exists(source)
        target_exists = os.path.exists(target)

        # Step 4: Case 1 - Already moved in previous run
        if target_exists and not source_exists:
            logger.debug(f"Already organized: {video_id} → {bucket}")
            skipped_already_organized += 1
            continue

        # Step 5: Case 2 - Missing entirely
        if not source_exists and not target_exists:
            logger.warning(
                f"Missing source and target for video {video_id}. "
                f"Stage 2 checkpoint indicated completion, but file doesn't exist."
            )
            missing_count += 1
            continue

        # Step 6: Case 3 - Source exists (move it)
        try:
            # Step 6a: Ensure target directory exists
            target_dir = os.path.dirname(target)
            os.makedirs(target_dir, exist_ok=True)

            # Step 6b: Move file (atomic within same filesystem)
            shutil.move(source, target)
            moved_count += 1

            # Step 6c: Log success
            logger.info(f"Moved: {video_id} → {bucket} ({moved_count}/{len(files_to_process)})")

        except Exception as e:
            # Step 6d: Log error but continue processing other files
            logger.error(f"Failed to move {video_id}: {e}")
            # Non-fatal error - continue with other files
            continue

    # Step 7: Calculate summary statistics
    total_processed = moved_count + skipped_already_organized + missing_count

    # Step 8: Log summary
    logger.info(
        f"\nOrganization complete:\n"
        f"  Total files:  {len(files_to_process)}\n"
        f"  Moved:        {moved_count}\n"
        f"  Already done: {skipped_already_organized}\n"
        f"  Missing:      {missing_count}\n"
        f"  Processed:    {total_processed}/{len(files_to_process)}"
    )

    # Step 9: Warn if missing files detected
    if missing_count > 0:
        logger.warning(f"{missing_count} files missing despite checkpoint indicating completion.")

    # Step 10: Return summary dictionary
    return {
        'moved_count': moved_count,
        'skipped_already_organized': skipped_already_organized,
        'missing_count': missing_count,
        'total_processed': total_processed
    }
