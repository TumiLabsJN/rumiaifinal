"""
Video processing through RumiAI pipeline for Stage 2

Integrates with existing rumiai_runner.py production script.

Source: VideoProcessingTI.md Section 4 (Function 4 & run_rumiai_pipeline)
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any

from ml_pipeline.stage2_processing.utils import get_bucket_path, load_json
from ml_pipeline.stage2_processing.exceptions import ProcessingError
from ml_pipeline.stage2_processing.validation import validate_temporal_windows_schema
from ml_pipeline.stage2_processing.checkpoint import save_checkpoint_with_backup

logger = logging.getLogger(__name__)

# IMPORTANT: rumiai_runner.py outputs to a HARDCODED flat directory (no bucket awareness)
# Stage 2.5 (FileOrganizationCHILD.md) will organize these files into bucket directories later
RUMIAI_OUTPUT_DIR = "/home/jorge/rumiaifinal/insights/"


def run_rumiai_pipeline(video_path: str, video_id: str, output_dir: str, timeout: int = 300) -> Dict[str, Any]:
    """
    Run existing RumiAI pipeline (scripts/rumiai_runner.py) with timeout enforcement.

    Success validation strategy:
    1. Subprocess exits with code 0 (check=True raises CalledProcessError if exit != 0)
    2. Insights file exists at expected path
    3. Insights file passes schema validation (handled by caller)

    IMPORTANT: rumiai_runner.py does NOT output JSON to stdout. Contract is file-based.
    Stdout/stderr are logged for debugging only (warnings, progress, errors).

    HYBRID MODE: Accepts both local MP4 paths and TikTok URLs.
    - If video_path is local file: processes directly
    - If video_path is TikTok URL: rumiai_runner.py scrapes, downloads, then processes

    Args:
        video_path: str, path to downloaded MP4 OR TikTok URL
        video_id: str, TikTok video ID
        output_dir: str, base output directory for analysis
        timeout: int, maximum processing time in seconds (default: 300s)

    Returns:
        dict: {"status": "success", "video_id": video_id, "insights_path": path}

    Raises:
        ProcessingError: if RumiAI pipeline fails or insights file missing
        TimeoutError: if processing exceeds timeout

    Source: VideoProcessingTI.md Section 2.3.3
    """
    # FIXED: rumiai_runner.py outputs to hardcoded flat directory, not bucket-specific paths
    insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"

    cmd = [
        sys.executable,  # python3
        'scripts/rumiai_runner.py',
        video_path  # rumiai_runner.py accepts URL or path as positional arg
    ]

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,  # Kill process if exceeds 300s
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError if exit code != 0
        )

        # Log stdout/stderr for debugging (don't parse - no JSON contract)
        if result.stdout:
            logger.debug(f"RumiAI stdout: {result.stdout[:500]}")
        if result.stderr:
            logger.warning(f"RumiAI stderr: {result.stderr[:500]}")

        # Validate insights file exists (ground truth for success)
        if not os.path.exists(insights_path):
            raise ProcessingError(
                video_id=video_id,
                stage="output_validation",
                message=f"RumiAI exited successfully (code 0) but no insights file at {insights_path}"
            )

        return {
            "status": "success",
            "video_id": video_id,
            "insights_path": insights_path
        }

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"RumiAI processing exceeded {timeout}s timeout for video {video_id}")
    except subprocess.CalledProcessError as e:
        raise ProcessingError(
            video_id=video_id,
            stage="rumiai_pipeline",
            message=f"RumiAI pipeline failed (exit code {e.returncode}). "
                   f"Stderr: {e.stderr[:200] if e.stderr else 'none'}"
        )


def process_videos_sequential(
    remaining_videos: list,
    bucket_name: str,
    checkpoint: dict,
    config: dict
) -> Dict[str, int]:
    """
    Process videos sequentially through RumiAI pipeline with checkpointing.

    FIXED: Now uses bucket_name and constructs full path, calls error handlers

    Args:
        remaining_videos: list, videos to process
        bucket_name: str, bucket name (e.g., "18-33s")
        checkpoint: dict, checkpoint data
        config: dict, configuration for path construction

    Returns:
        dict: processing statistics

    Source: VideoProcessingTI.md Section 4 (Function 4)
    """
    # FIXED: Construct full bucket path
    bucket_path = get_bucket_path(config, bucket_name)
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    for i, video in enumerate(remaining_videos, start=1):
        video_id = video['id']
        local_video_path = f"{bucket_path}videos/{video_id}.mp4"

        # Hybrid approach: Use local file if exists, otherwise use TikTok URL
        if os.path.exists(local_video_path):
            video_path = local_video_path
            logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id} (local file)")
        elif 'webVideoUrl' in video:
            video_path = video['webVideoUrl']
            logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id} (TikTok URL)")
        else:
            logger.error(f"Video {video_id} not found locally and no webVideoUrl available")
            handle_video_processing_error(
                ValueError(f"Video {video_id} missing: no local file and no webVideoUrl"),
                video_id, checkpoint, checkpoint_path
            )
            continue

        try:
            # Run RumiAI pipeline with timeout
            result = run_rumiai_pipeline(
                video_path=video_path,
                video_id=video_id,
                output_dir=f"{bucket_path}analysis/",  # Not used, but kept for API consistency
                timeout=300
            )

            # FIXED: Validate output exists at hardcoded RumiAI output directory
            insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
            if not os.path.exists(insights_path):
                raise ProcessingError(
                    video_id=video_id,
                    stage="output_validation",
                    message=f"RumiAI did not generate insights file at {insights_path}"
                )

            # Validate output schema
            insights = load_json(insights_path)
            validate_temporal_windows_schema(insights)

            # Mark as completed
            checkpoint['completed'] += 1
            checkpoint['remaining'] -= 1
            checkpoint['completed_video_ids'].append(video_id)
            checkpoint['last_checkpoint'] = datetime.utcnow().isoformat()

            save_checkpoint_with_backup(checkpoint_path, checkpoint)
            logger.info(f"Successfully processed video {video_id} ({checkpoint['completed']}/{checkpoint['total_videos']})")

        except Exception as e:
            # FIXED: Call error handler from Section 6
            handle_video_processing_error(e, video_id, checkpoint, checkpoint_path)
            continue

    return {
        "total": checkpoint['total_videos'],
        "completed": checkpoint['completed'],
        "failed": checkpoint['failed']
    }


def handle_video_processing_error(
    error: Exception,
    video_id: str,
    checkpoint: dict,
    checkpoint_path: str
) -> str:
    """
    Handle errors during video processing with skip-on-fail policy.

    ENHANCED: Now includes metrics logging and critical error detection

    Args:
        error: Exception, the error that occurred
        video_id: str, TikTok video ID
        checkpoint: dict, current checkpoint data
        checkpoint_path: str, path to checkpoint file

    Returns:
        str: Action taken ("failed", "critical")

    Source: VideoProcessingTI.md Section 6
    """
    error_type = type(error).__name__
    error_message = str(error)

    # Check for critical errors that should halt processing
    if isinstance(error, (OSError, MemoryError)) and "disk full" in error_message.lower():
        logger.critical(f"CRITICAL ERROR: Disk full. Cannot continue processing.")
        raise error

    # Log error with full context
    logger.error(f"Failed to process video {video_id}: {error_message}")
    logger.error(f"Error type: {error_type}")

    # Update checkpoint with failure
    checkpoint['failed'] += 1
    checkpoint['remaining'] -= 1
    checkpoint['failed_video_ids'].append({
        "video_id": video_id,
        "error": error_message,
        "error_type": error_type,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Save checkpoint with failure
    save_checkpoint_with_backup(checkpoint_path, checkpoint)

    # Log skip-on-fail action
    logger.warning(f"Video {video_id} marked as failed. Continuing batch processing.")

    return "failed"
