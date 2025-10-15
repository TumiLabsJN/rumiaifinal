"""
Video download with retry logic for Stage 2: Video Processing

Downloads videos from Apify download URLs with exponential backoff retry.

Source: VideoProcessingTI.md Section 4 (Function 3)
"""

import os
import time
import logging
import requests
from typing import Dict, Any

from ml_pipeline.stage2_processing.exceptions import DownloadError
from ml_pipeline.stage2_processing.utils import MIN_VIDEO_SIZE

logger = logging.getLogger(__name__)


def download_video(video_metadata: Dict[str, Any], output_dir: str, max_attempts: int = 3) -> str:
    """
    Download video MP4 from Apify download URL with retry logic.

    Args:
        video_metadata: dict, Apify metadata
        output_dir: str, path to bucket/videos/ directory
        max_attempts: int, max download attempts (default: 3)

    Returns:
        str: path to downloaded MP4

    Raises:
        DownloadError: if download fails after max_attempts

    Source: VideoProcessingTI.md Section 4 (Function 3)
    """

    video_id = video_metadata['id']
    download_url = video_metadata['videoMeta']['downloadAddr']
    output_path = f"{output_dir}/{video_id}.mp4"

    # Check if video already downloaded (resume optimization)
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)

        if file_size < MIN_VIDEO_SIZE:
            logger.warning(
                f"Existing file for {video_id} too small ({file_size} bytes), "
                f"removing and re-downloading"
            )
            os.remove(output_path)
        else:
            logger.info(f"Video {video_id} already downloaded and valid ({file_size} bytes), skipping")
            return output_path

    # Retry loop with exponential backoff
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Downloading video {video_id} (attempt {attempt}/{max_attempts})")

            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(output_path)
            if file_size < MIN_VIDEO_SIZE:
                raise DownloadError(
                    video_id=video_id,
                    attempts=attempt,
                    original_error=Exception(f"Downloaded file too small: {file_size} bytes (minimum: {MIN_VIDEO_SIZE})")
                )

            logger.info(f"Successfully downloaded video {video_id} ({file_size / 1024 / 1024:.2f} MB)")
            return output_path

        except (requests.exceptions.RequestException, DownloadError) as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")

            if os.path.exists(output_path):
                os.remove(output_path)

            if attempt < max_attempts:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise DownloadError(
                    video_id=video_id,
                    attempts=max_attempts,
                    original_error=e
                )
