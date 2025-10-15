"""
Checkpoint management for Stage 2: Video Processing

Implements checkpoint initialization, save/load with backup, and recovery.

Source: VideoProcessingTI.md Section 4 (Functions 2 & Helper Functions)
"""

import os
import json
import shutil
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any

from ml_pipeline.stage2_processing.utils import (
    get_bucket_path,
    save_json,
    load_json
)
from ml_pipeline.stage2_processing.exceptions import CheckpointCorruptionError

logger = logging.getLogger(__name__)


def save_checkpoint_with_backup(checkpoint_path: str, checkpoint: dict):
    """
    Save checkpoint with automatic backup to prevent data loss from corruption.

    Process:
    1. Copy existing checkpoint to .backup.json (if exists)
    2. Write new checkpoint to .json
    3. Handle errors gracefully

    Args:
        checkpoint_path: str, path to checkpoint file
        checkpoint: dict, checkpoint data to save

    Source: VideoProcessingTI.md Section 4 (Helper Functions)
    """
    backup_path = checkpoint_path.replace('.json', '.backup.json')

    # Backup existing checkpoint before overwriting
    if os.path.exists(checkpoint_path):
        try:
            shutil.copy2(checkpoint_path, backup_path)
        except Exception as e:
            logger.warning(f"Failed to backup checkpoint: {e}")

    # Write new checkpoint
    save_json(checkpoint_path, checkpoint)


def load_checkpoint_with_recovery(checkpoint_path: str) -> dict:
    """
    Load checkpoint with automatic recovery from backup on corruption.

    Recovery strategy:
    1. Try loading main checkpoint
    2. If corrupted (JSONDecodeError), try loading .backup.json
    3. If both corrupted, fail with clear recovery instructions

    Args:
        checkpoint_path: str, path to checkpoint file

    Returns:
        dict: loaded checkpoint data

    Raises:
        CheckpointCorruptionError: if both checkpoint and backup are corrupted

    Source: VideoProcessingTI.md Section 4 (Helper Functions)
    """
    backup_path = checkpoint_path.replace('.json', '.backup.json')

    # Try loading main checkpoint
    try:
        return load_json(checkpoint_path)
    except json.JSONDecodeError as e:
        logger.error(f"Checkpoint corrupted: {e}")

        # Try loading backup
        if os.path.exists(backup_path):
            logger.info("Attempting to restore from backup...")
            try:
                checkpoint = load_json(backup_path)
                logger.info("✓ Successfully restored from backup")
                return checkpoint
            except json.JSONDecodeError:
                logger.error("Backup also corrupted")

        # Both corrupted - raise with recovery instructions
        raise CheckpointCorruptionError(
            checkpoint_path=checkpoint_path,
            backup_path=backup_path,
            original_error=e
        )


def initialize_checkpoint(
    bucket_name: str,
    video_list: list,
    config: dict
) -> Tuple[dict, list]:
    """
    Initialize checkpoint for video processing stage.

    FIXED: Now uses bucket_name (not full path) and constructs full path using get_bucket_path()

    Args:
        bucket_name: str, bucket name only (e.g., "18-33s"), NOT full path
        video_list: list, videos selected by Stage 1
        config: dict, loaded from config.json

    Returns:
        checkpoint: dict, checkpoint data (new or existing)
        remaining_videos: list, videos to process (excludes completed)

    Source: VideoProcessingTI.md Section 4 (Function 2)
    """

    def validate_config_match(checkpoint_config: dict, current_config: dict):
        """Validates that checkpoint config matches current run config."""
        critical_fields = ['video_count', 'selection_strategy', 'date_filter']
        mismatches = []

        for field in critical_fields:
            if checkpoint_config.get(field) != current_config.get(field):
                mismatches.append(
                    f"{field}: checkpoint={checkpoint_config.get(field)}, "
                    f"current={current_config.get(field)}"
                )

        if mismatches:
            raise ValueError(
                f"Config mismatch detected. Cannot resume with different parameters:\n" +
                "\n".join(mismatches)
            )

    # FIXED: Construct full bucket path using helper function
    bucket_path = get_bucket_path(config, bucket_name)
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    # Check if checkpoint exists (auto-resume scenario)
    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint_with_recovery(checkpoint_path)
        validate_config_match(checkpoint['config'], config)

        completed_ids = set(checkpoint['completed_video_ids'])
        remaining_videos = [v for v in video_list if v['id'] not in completed_ids]

        logger.info(f"Checkpoint detected: {checkpoint['completed']}/{checkpoint['total_videos']} completed")
        logger.info(f"Auto-resuming from bucket {bucket_name} ({len(remaining_videos)} remaining)")

        return checkpoint, remaining_videos

    # Create new checkpoint
    else:
        checkpoint = {
            "stage": "video_processing",
            "bucket": bucket_name,
            "total_videos": len(video_list),
            "completed": 0,
            "failed": 0,
            "remaining": len(video_list),
            "last_checkpoint": datetime.utcnow().isoformat(),
            "completed_video_ids": [],
            "failed_video_ids": [],
            "config": config,
            "status": "in_progress",
            "pause_reason": None,
            "pause_timestamp": None
        }

        save_checkpoint_with_backup(checkpoint_path, checkpoint)
        logger.info(f"Created new checkpoint for bucket {bucket_name} ({len(video_list)} videos)")

        return checkpoint, video_list


def finalize_checkpoint(checkpoint: dict, config: dict):
    """
    Mark video processing stage as complete.

    FIXED: Now accepts config to construct full bucket path

    Args:
        checkpoint: dict, checkpoint data after all videos processed
        config: dict, configuration

    Source: VideoProcessingTI.md Section 4 (Function 6)
    """
    checkpoint['status'] = 'completed'
    checkpoint['completion_time'] = datetime.utcnow().isoformat()

    # FIXED: Construct full bucket path
    bucket_path = get_bucket_path(config, checkpoint['bucket'])
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    save_json(checkpoint_path, checkpoint)

    logger.info(f"Video processing complete for bucket {checkpoint['bucket']}")
    logger.info(f"  Total: {checkpoint['total_videos']}")
    logger.info(f"  Completed: {checkpoint['completed']}")
    logger.info(f"  Failed: {checkpoint['failed']}")

    if checkpoint['failed'] > 0:
        logger.warning(f"  {checkpoint['failed']} videos failed (see failed_video_ids in checkpoint)")
