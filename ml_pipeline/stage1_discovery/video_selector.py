"""
Stage 1.4: Video Selection Per Bucket

Selects videos per winning bucket using contrastive or top strategy.

Source: VideoDiscoveryCHILDTI.md Section 4.4
"""

import logging
from typing import List, Dict
from datetime import datetime, timezone

from .constants import (
    CONTRASTIVE_TOP_SPLIT,
    MIN_VIDEOS_PER_BUCKET,
)

# Import bucket utility from foundation
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from foundation.buckets import assign_bucket


logger = logging.getLogger(__name__)


class VideoSelector:
    """
    Handles video selection per bucket using different strategies.

    Implements:
    - Contrastive strategy (80% top + 20% bottom)
    - Top strategy (100% top performers)
    - Per-bucket video grouping
    - Selection count validation
    """

    def select_videos_per_bucket(
        self,
        videos: List[Dict],
        winning_buckets: List[str],
        strategy: str,
        video_count: int
    ) -> Dict[str, Dict]:
        """
        Select videos for each winning bucket using specified strategy.

        Args:
            videos: All filtered videos (sorted by engagement DESC)
            winning_buckets: List of top 3 bucket names (e.g., ["18-33s", "33-60s", "13-18s"])
            strategy: "contrastive" or "top"
            video_count: N videos to select per bucket

        Returns:
            Dict mapping bucket → selected_videos.json structure
            {
                "18-33s": {
                    "bucket": "18-33s",
                    "strategy": "contrastive",
                    "video_count": 100,
                    "selected_count": 100,
                    "top_count": 80,
                    "bottom_count": 20,
                    "videos": [...],
                    "selection_date": "2025-01-28T10:30:00Z"
                },
                ...
            }
        """
        logger.info(
            f"Starting video selection: strategy={strategy}, video_count={video_count} per bucket"
        )

        # Step 1: Group all videos by bucket
        bucket_videos = self._group_videos_by_bucket(videos)

        # Step 2: Filter to winning buckets only
        winning_bucket_videos = {
            bucket: vids for bucket, vids in bucket_videos.items()
            if bucket in winning_buckets
        }

        # Step 3: Select videos per bucket using strategy
        selected_per_bucket = {}
        for bucket in winning_buckets:
            if bucket not in winning_bucket_videos:
                logger.warning(f"Bucket {bucket} has no videos. Skipping.")
                continue

            bucket_vids = winning_bucket_videos[bucket]

            if strategy == "contrastive":
                selection = self._select_contrastive(bucket, bucket_vids, video_count)
            elif strategy == "top":
                selection = self._select_top(bucket, bucket_vids, video_count)
            else:
                raise ValueError(f"Invalid strategy: {strategy}")

            selected_per_bucket[bucket] = selection

            logger.info(
                f"Bucket {bucket}: Selected {selection['selected_count']} videos "
                f"(top: {selection['top_count']}, bottom: {selection['bottom_count']})"
            )

        return selected_per_bucket

    def _group_videos_by_bucket(self, videos: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group all videos by duration bucket.

        Returns:
            Dict mapping bucket name → list of videos
        """
        bucket_videos = {}

        for video in videos:
            duration = video.get("duration")
            if duration is None:
                logger.warning(f"Video {video.get('id', 'unknown')} has no duration. Skipping.")
                continue

            try:
                bucket = assign_bucket(duration)
                if bucket not in bucket_videos:
                    bucket_videos[bucket] = []
                bucket_videos[bucket].append(video)
            except ValueError as e:
                logger.warning(f"Video {video.get('id', 'unknown')}: {e}")
                continue

        # Log bucket distribution
        logger.info("Video distribution across all buckets:")
        for bucket in sorted(bucket_videos.keys()):
            logger.info(f"  - {bucket}: {len(bucket_videos[bucket])} videos")

        return bucket_videos

    def _select_contrastive(
        self,
        bucket: str,
        bucket_videos: List[Dict],
        video_count: int
    ) -> Dict:
        """
        Select videos using contrastive strategy (80% top + 20% bottom).

        Args:
            bucket: Bucket name
            bucket_videos: All videos in this bucket (sorted by engagement DESC)
            video_count: Target N videos to select

        Returns:
            selected_videos.json structure for this bucket
        """
        # Calculate split
        top_count = int(video_count * CONTRASTIVE_TOP_SPLIT)  # 80%
        bottom_count = video_count - top_count  # 20%

        # Handle case where bucket has fewer videos than requested
        available_count = len(bucket_videos)
        if available_count < video_count:
            logger.warning(
                f"Bucket {bucket} has only {available_count} videos (requested {video_count}). "
                f"Adjusting selection proportionally."
            )
            # Adjust proportionally
            actual_top = int(available_count * CONTRASTIVE_TOP_SPLIT)
            actual_bottom = available_count - actual_top
        else:
            actual_top = top_count
            actual_bottom = bottom_count

        # Select top performers (first N videos, already sorted DESC)
        top_videos = bucket_videos[:actual_top]

        # Tag top performers with is_top_performer status
        for video in top_videos:
            video['is_top_performer'] = True

        # Select bottom performers (last N videos)
        bottom_videos = bucket_videos[-actual_bottom:] if actual_bottom > 0 else []

        # Tag bottom performers with is_top_performer status
        for video in bottom_videos:
            video['is_top_performer'] = False

        # Combine
        selected_videos = top_videos + bottom_videos

        return {
            "bucket": bucket,
            "strategy": "contrastive",
            "video_count": video_count,
            "selected_count": len(selected_videos),
            "top_count": len(top_videos),
            "bottom_count": len(bottom_videos),
            "videos": selected_videos,
            "selection_date": datetime.now(timezone.utc).isoformat()
        }

    def _select_top(
        self,
        bucket: str,
        bucket_videos: List[Dict],
        video_count: int
    ) -> Dict:
        """
        Select videos using top strategy (100% top performers).

        Args:
            bucket: Bucket name
            bucket_videos: All videos in this bucket (sorted by engagement DESC)
            video_count: Target N videos to select

        Returns:
            selected_videos.json structure for this bucket
        """
        # Handle case where bucket has fewer videos than requested
        available_count = len(bucket_videos)
        if available_count < video_count:
            logger.warning(
                f"Bucket {bucket} has only {available_count} videos (requested {video_count}). "
                f"Selecting all available."
            )
            actual_count = available_count
        else:
            actual_count = video_count

        # Select top N performers
        selected_videos = bucket_videos[:actual_count]

        return {
            "bucket": bucket,
            "strategy": "top",
            "video_count": video_count,
            "selected_count": len(selected_videos),
            "top_count": len(selected_videos),
            "bottom_count": 0,
            "videos": selected_videos,
            "selection_date": datetime.now(timezone.utc).isoformat()
        }
