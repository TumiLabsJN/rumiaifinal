"""
Stage 1.2: Date Filtering

Filters scraped videos by publication date using UTC timezone.

Source: VideoDiscoveryCHILDTI.md Section 4.2
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict

from .constants import (
    DATE_FILTER_TIMEZONE,
    CLOCK_SKEW_TOLERANCE_HOURS,
    MIN_VIDEOS_FOR_ANALYSIS,
    EXIT_CODE_INVALID_DATE_FILTER,
    EXIT_CODE_INSUFFICIENT_VIDEOS,
)


logger = logging.getLogger(__name__)


class DateFilter:
    """
    Handles date-based filtering of video metadata.

    Implements:
    - UTC-based date filtering
    - Timestamp validation
    - Clock skew tolerance
    - Degraded mode handling (< 100 videos)
    """

    def filter_by_date(
        self,
        videos: List[Dict],
        date_filter: str
    ) -> List[Dict]:
        """
        Filter videos by publication date using UTC timezone.

        Args:
            videos: Apify video metadata objects (already sorted)
            date_filter: "last_N_days" (e.g., "last_90_days")

        Returns:
            Filtered videos within date range

        Raises:
            ValueError: If date_filter format invalid
            ValueError: If < 10 videos remain after filtering (insufficient for analysis)
        """
        # Step 1: Parse date_filter parameter
        days = self._parse_date_filter(date_filter)

        # Step 2: Calculate cutoff date (always use UTC for consistency)
        cutoff_date = datetime.now(DATE_FILTER_TIMEZONE) - timedelta(days=days)
        logger.info(f"Date filtering: last {days} days (cutoff: {cutoff_date.isoformat()})")

        # Step 3: Filter with robust timestamp validation
        filtered_videos, skipped_reasons = self._filter_videos_by_date(
            videos, cutoff_date
        )

        # Step 4: Log filtering results
        self._log_filtering_results(videos, filtered_videos, days, skipped_reasons)

        # Step 5: Handle edge cases
        self._validate_filtered_count(filtered_videos)

        return filtered_videos

    def _parse_date_filter(self, date_filter: str) -> int:
        """
        Parse date_filter parameter to extract number of days.

        Format: "last_N_days" where N = 1-365

        Returns:
            Number of days as integer

        Raises:
            ValueError: If format invalid
        """
        try:
            days = int(date_filter.replace("last_", "").replace("_days", ""))
            if days < 1 or days > 365:
                raise ValueError(f"Days must be between 1-365, got {days}")
            return days
        except (ValueError, AttributeError) as e:
            logger.error(f"Invalid date_filter format: {date_filter}")
            raise ValueError(
                f"Invalid date_filter format: {date_filter}. "
                f"Expected: last_N_days where N = 1-365"
            ) from e

    def _filter_videos_by_date(
        self,
        videos: List[Dict],
        cutoff_date: datetime
    ) -> tuple:
        """
        Filter videos with robust timestamp validation.

        Returns:
            tuple: (filtered_videos, skipped_reasons)
        """
        filtered_videos = []
        skipped_reasons = {
            "null_or_zero": 0,
            "invalid_conversion": 0,
            "future_timestamp": 0,
        }

        for video in videos:
            video_id = video.get("id", "unknown")
            create_time = video.get("createTime")

            # Validation 1: Check create_time exists and is non-zero
            if create_time is None or create_time == 0:
                logger.warning(f"Video {video_id} has invalid create_time (null/zero). Skipping.")
                skipped_reasons["null_or_zero"] += 1
                continue

            # Validation 2: Convert Unix timestamp to UTC datetime
            try:
                video_date = datetime.fromtimestamp(create_time, tz=DATE_FILTER_TIMEZONE)
            except (ValueError, OSError) as e:
                logger.warning(
                    f"Video {video_id} has invalid timestamp {create_time}. Skipping. Error: {e}"
                )
                skipped_reasons["invalid_conversion"] += 1
                continue

            # Validation 3: Handle future timestamps (clock skew tolerance)
            future_threshold = datetime.now(DATE_FILTER_TIMEZONE) + timedelta(
                hours=CLOCK_SKEW_TOLERANCE_HOURS
            )
            if video_date > future_threshold:
                logger.warning(
                    f"Video {video_id} has future timestamp {video_date.isoformat()}. Skipping."
                )
                skipped_reasons["future_timestamp"] += 1
                continue

            # Validation 4: Apply date filter
            if video_date >= cutoff_date:
                filtered_videos.append(video)

        return filtered_videos, skipped_reasons

    def _log_filtering_results(
        self,
        original_videos: List[Dict],
        filtered_videos: List[Dict],
        days: int,
        skipped_reasons: Dict[str, int]
    ):
        """Log filtering results with detailed skip reasons."""
        logger.info(
            f"Date filtering: {len(original_videos)} → {len(filtered_videos)} videos (last {days} days)"
        )

        total_skipped = sum(skipped_reasons.values())
        if total_skipped > 0:
            logger.info(f"Skipped {total_skipped} videos due to invalid timestamps:")
            for reason, count in skipped_reasons.items():
                if count > 0:
                    logger.info(f"  - {reason}: {count}")

    def _validate_filtered_count(self, filtered_videos: List[Dict]):
        """
        Validate filtered video count and handle edge cases.

        Raises:
            ValueError: If < 10 videos (insufficient for analysis)
        """
        video_count = len(filtered_videos)

        # Handle edge case - insufficient videos after filtering
        if video_count < MIN_VIDEOS_FOR_ANALYSIS:
            logger.error(
                f"Insufficient videos for analysis. Need ≥{MIN_VIDEOS_FOR_ANALYSIS}, got {video_count}"
            )
            raise ValueError(
                f"Insufficient videos for analysis. Need ≥{MIN_VIDEOS_FOR_ANALYSIS}, got {video_count}. "
                f"Try different target or relax date filter."
            )

        # Handle edge case - degraded mode (10-99 videos)
        if video_count < 100:
            logger.warning(
                f"Small dataset ({video_count} videos). "
                f"Statistical validity may be limited. Recommended: ≥100 videos."
            )
