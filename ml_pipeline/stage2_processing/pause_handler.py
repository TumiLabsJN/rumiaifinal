"""
Graceful pause handling for Stage 2: Video Processing

Implements SIGINT (Ctrl+C) handling for graceful pause between videos.

Source: VideoProcessingTI.md Section 4 (Function 5)
"""

import os
import sys
import signal
import platform
import logging
from datetime import datetime

from ml_pipeline.stage2_processing.checkpoint import save_checkpoint_with_backup
from ml_pipeline.stage2_processing.utils import get_bucket_path

logger = logging.getLogger(__name__)

# Global pause flag (signal-safe for single-threaded processing)
pause_requested = False


def request_pause(signum: int, frame):
    """
    Signal handler for graceful pause.

    Thread-safety note: Global flag is safe for single-threaded sequential
    video processing. Flag is only checked between videos, not during processing.

    Args:
        signum: Signal number
        frame: Current stack frame

    Source: VideoProcessingTI.md Section 4 (Function 5)
    """
    global pause_requested

    if not pause_requested:
        pause_requested = True
        logger.info("\n⏸️  Pause requested. Will pause after current video completes...")
        logger.info("   Press Ctrl+C again to force quit (current video will be lost)")
    else:
        # Second Ctrl+C = force quit
        logger.warning("\n❌ Force quit requested. Current video will NOT be saved.")
        logger.warning("   Checkpoint may be inconsistent if killed mid-write.")
        sys.exit(1)


def process_videos_with_pause_support(
    remaining_videos: list,
    bucket_name: str,
    checkpoint: dict,
    config: dict
):
    """
    Process videos with graceful pause support.

    FIXED: Now uses bucket_name and constructs full path

    Args:
        remaining_videos: list, videos to process
        bucket_name: str, bucket name
        checkpoint: dict, checkpoint data
        config: dict, configuration

    Source: VideoProcessingTI.md Section 4 (Function 5)
    """
    global pause_requested

    # Register signal handler (cross-platform SIGINT support)
    signal.signal(signal.SIGINT, request_pause)   # Ctrl+C

    # Platform compatibility note
    if platform.system() == 'Windows':
        logger.warning("Graceful pause (Ctrl+C) has limited support on Windows")
        logger.warning("Checkpoint auto-save after each video provides recovery")

    # FIXED: Construct full bucket path
    bucket_path = get_bucket_path(config, bucket_name)
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    # Process each video
    for i, video in enumerate(remaining_videos, start=1):
        # Check pause flag BEFORE starting next video
        if pause_requested:
            checkpoint['status'] = 'paused'
            checkpoint['pause_reason'] = 'user_requested'
            checkpoint['pause_timestamp'] = datetime.utcnow().isoformat()
            save_checkpoint_with_backup(checkpoint_path, checkpoint)

            logger.info(f"\n⏸️  Paused gracefully after {checkpoint['completed']}/{checkpoint['total_videos']} videos")
            logger.info(f"   Resume anytime by re-running the same command")
            logger.info(f"   Checkpoint: {checkpoint_path}")
            return

        # FIXED: Use hybrid approach (same as video_processor.py)
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
            from ml_pipeline.stage2_processing.video_processor import handle_video_processing_error
            handle_video_processing_error(
                ValueError(f"Video {video_id} missing: no local file and no webVideoUrl"),
                video_id, checkpoint, checkpoint_path
            )
            continue

        # Process video through RumiAI pipeline (delegate to video_processor logic)
        try:
            from ml_pipeline.stage2_processing.video_processor import (
                run_rumiai_pipeline, validate_temporal_windows_schema, load_json,
                handle_video_processing_error, RUMIAI_OUTPUT_DIR, ProcessingError
            )

            # Run RumiAI pipeline with timeout
            result = run_rumiai_pipeline(
                video_path=video_path,
                video_id=video_id,
                output_dir=f"{bucket_path}analysis/",
                timeout=300
            )

            # Validate output exists at hardcoded RumiAI output directory
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
            # Handle any processing error
            handle_video_processing_error(e, video_id, checkpoint, checkpoint_path)
            continue

    # If we get here, all videos processed (no pause requested)
    logger.info(f"All videos processed successfully")
