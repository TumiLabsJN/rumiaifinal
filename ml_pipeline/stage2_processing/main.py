"""
Main orchestration for Stage 2: Video Processing

Coordinates all substeps (bucket init, checkpoint, download, processing).

Source: VideoProcessingTI.md Section 4 (Main Orchestration Function)
"""

import logging
from typing import Dict, List

from ml_pipeline.stage2_processing.bucket_init import ensure_bucket_exists
from ml_pipeline.stage2_processing.checkpoint import (
    initialize_checkpoint,
    finalize_checkpoint
)
from ml_pipeline.stage2_processing.video_download import download_video
from ml_pipeline.stage2_processing.video_processor import (
    process_videos_sequential,
    RUMIAI_OUTPUT_DIR
)
from ml_pipeline.stage2_processing.pause_handler import process_videos_with_pause_support
from ml_pipeline.stage2_processing.utils import get_bucket_path

logger = logging.getLogger(__name__)


def stage_2_video_processing_main(
    config: dict,
    video_list: List[dict],
    bucket_name: str,
    enable_pause_support: bool = True
) -> Dict[str, any]:
    """
    Main orchestration function for Stage 2: Video Processing.

    Coordinates all substeps (2.3.0 through 2.3.5) in sequence.

    Args:
        config: dict, loaded from config.json
        video_list: list, selected videos from Stage 1
        bucket_name: str, bucket to process (e.g., "18-33s")
        enable_pause_support: bool, enable Ctrl+C graceful pause (default: True)

    Returns:
        dict: final processing statistics
            {"total": int, "completed": int, "failed": int, "status": str}

    Raises:
        ValueError: if input validation fails
        OSError: if disk/permission errors occur

    Source: VideoProcessingTI.md Section 4 (Main Orchestration Function)
    """

    # Step 0: Ensure this bucket's directory structure exists
    # Defensive: Creates if missing (e.g., reprocessing after manual deletion)
    # Idempotent: No-op if Stage 1 already created it (exist_ok=True)
    logger.info(f"Step 0: Ensuring bucket directory exists for {bucket_name}")
    bucket_path = ensure_bucket_exists(config, bucket_name)
    logger.debug(f"Bucket directory ready: {bucket_path}")

    # Step 1: Load or create checkpoint
    logger.info(f"Step 1: Initializing checkpoint for bucket {bucket_name}")
    checkpoint, remaining_videos = initialize_checkpoint(bucket_name, video_list, config)

    if not remaining_videos:
        logger.info("No videos to process (all videos already completed)")
        finalize_checkpoint(checkpoint, config)
        return {
            "total": checkpoint['total_videos'],
            "completed": checkpoint['completed'],
            "failed": checkpoint['failed'],
            "status": "completed"
        }

    # Step 2: Download videos if downloadAddr available (optional pre-download)
    logger.info(f"Step 2: Checking for pre-downloadable videos")
    bucket_path = get_bucket_path(config, bucket_name)
    videos_dir = f"{bucket_path}videos/"

    downloadable_count = 0
    for i, video in enumerate(remaining_videos, start=1):
        video_id = video['id']

        # Check if video has download URL (try multiple API formats - API changed Oct 2025)
        has_download_url = False
        video_meta = video.get('videoMeta', {})

        # Check old API format (downloadAddr)
        if video_meta and 'downloadAddr' in video_meta:
            has_download_url = True
        # Check mediaUrls
        elif 'mediaUrls' in video and video.get('mediaUrls'):
            has_download_url = True

        if has_download_url:
            logger.info(f"Pre-downloading video {i}/{len(remaining_videos)}: {video_id}")
            try:
                download_video(video, videos_dir)
                downloadable_count += 1
            except Exception as e:
                logger.warning(f"Pre-download failed for {video_id}: {e}")
                logger.info(f"Will use webVideoUrl during processing instead")
        else:
            # No download URL available, will use webVideoUrl during processing
            logger.debug(f"Video {video_id} has no direct download URL, will use webVideoUrl during processing")

    logger.info(f"Pre-downloaded {downloadable_count}/{len(remaining_videos)} videos")
    if downloadable_count == 0:
        logger.info("No videos pre-downloaded. Will use webVideoUrl for all videos during processing.")

    # Step 3-4: Process videos with optional pause support
    logger.info(f"Step 3-4: Processing {len(remaining_videos)} videos")

    if enable_pause_support:
        # Use pause handler (checks for Ctrl+C between videos)
        process_videos_with_pause_support(remaining_videos, bucket_name, checkpoint, config)
    else:
        # Use direct processing (no pause handling)
        stats = process_videos_sequential(remaining_videos, bucket_name, checkpoint, config)

    # Step 5: Finalize checkpoint
    logger.info("Step 5: Finalizing checkpoint")
    finalize_checkpoint(checkpoint, config)

    # Step 6: Validate outputs (basic validation)
    logger.info("Step 6: Validating stage outputs")
    validate_stage_output(bucket_name, checkpoint, config)

    return {
        "total": checkpoint['total_videos'],
        "completed": checkpoint['completed'],
        "failed": checkpoint['failed'],
        "status": checkpoint['status']
    }


def validate_stage_output(bucket_name: str, checkpoint: dict, config: dict):
    """
    Validate stage outputs after all videos processed.

    FIXED: Validates outputs at hardcoded RumiAI output directory (flat structure).
    Stage 2.5 (FileOrganizationCHILD.md) will organize these into bucket directories later.

    Args:
        bucket_name: str, bucket name
        checkpoint: dict, final checkpoint data
        config: dict, configuration

    Raises:
        AssertionError: if output validation fails

    Source: VideoProcessingTI.md Section 6.3
    """
    import os

    # FIXED: Check files in hardcoded RumiAI output directory (not bucket-specific yet)
    insights_dir = RUMIAI_OUTPUT_DIR

    # 1. Check insights directory exists
    assert os.path.exists(insights_dir), f"RumiAI insights directory not found: {insights_dir}"

    # 2. Check each completed video has insights file at hardcoded location
    missing_files = []
    for video_id in checkpoint['completed_video_ids']:
        insights_path = f"{insights_dir}{video_id}_temporal_windows_updated.json"
        if not os.path.exists(insights_path):
            missing_files.append(video_id)

    assert len(missing_files) == 0, \
        f"Missing insights files for {len(missing_files)} completed videos: {missing_files[:5]}"

    # 3. Check checkpoint saved
    bucket_path = get_bucket_path(config, bucket_name)
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"
    assert os.path.exists(checkpoint_path), "Checkpoint file not saved"

    # 4. Log summary
    logger.info(f"Stage 2 validation passed for bucket {bucket_name}")
    logger.info(f"  Completed videos: {checkpoint['completed']}")
    logger.info(f"  Failed videos: {checkpoint['failed']}")
    logger.info(f"  Insights location: {insights_dir} (flat structure)")
    logger.info(f"  Note: Stage 2.5 will organize files into bucket directories")
