"""
Stage 1.3: Winner Distribution Analysis

Analyzes top performers to identify top 3 buckets where winners cluster.

Source: VideoDiscoveryCHILDTI.md Section 4.3
"""

import logging
from typing import List, Dict, Tuple

from .constants import (
    MIN_VIDEOS_FOR_ANALYSIS,
    TOP_PERFORMERS_FOR_ANALYSIS,
    TOP_BUCKETS_TO_PROCESS,
    MIN_WINNER_PERCENTAGE,
)

# Import bucket utility from foundation
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from foundation.buckets import assign_bucket


logger = logging.getLogger(__name__)


class WinnerAnalyzer:
    """
    Analyzes winner distribution across duration buckets.

    Implements:
    - Top performer bucket analysis
    - Winner concentration calculation
    - Qualified bucket filtering (≥5% winners)
    - Top 3 bucket selection
    """

    def analyze_winner_distribution(
        self,
        videos: List[Dict]
    ) -> Tuple[List[str], Dict[str, int], Dict[str, float]]:
        """
        Identify winning buckets by analyzing where top performers cluster.

        Args:
            videos: Filtered videos (sorted by engagement DESC)

        Returns:
            tuple:
                - list[str]: Top 3 bucket names where winners cluster
                - dict[str, int]: Winner distribution {bucket: count}
                - dict[str, float]: All qualified buckets {bucket: percentage}

        Raises:
            ValueError: If < 10 videos total (insufficient for analysis)
            ValueError: If no buckets qualify (≥5% winners required)
        """
        # Step 1: Validate minimum dataset size
        if len(videos) < MIN_VIDEOS_FOR_ANALYSIS:
            logger.error(
                f"Insufficient videos for analysis. Need ≥{MIN_VIDEOS_FOR_ANALYSIS}, got {len(videos)}"
            )
            raise ValueError(
                f"Insufficient videos for analysis. Need ≥{MIN_VIDEOS_FOR_ANALYSIS}, got {len(videos)}. "
                f"Try different target or relax date filter."
            )

        # Step 2: Determine analysis mode based on dataset size
        top_performers = self._select_top_performers(videos)

        # Step 3: Bucket videos by duration
        winner_distribution = self._bucket_top_performers(top_performers)

        # Step 4: Calculate winner concentration percentages
        winner_percentages = self._calculate_winner_percentages(
            winner_distribution, len(top_performers)
        )

        # Step 5: Filter buckets - only keep those with ≥ MIN_WINNER_PERCENTAGE (5%)
        qualified_buckets = self._filter_qualified_buckets(winner_percentages)

        # Step 6: Sort qualified buckets by winner concentration and select top 3
        top_buckets = self._select_top_buckets(qualified_buckets)

        # Step 7: Log winner distribution
        self._log_winner_distribution(
            top_buckets, winner_distribution, len(top_performers)
        )

        # Step 8: Return top bucket names (without percentages)
        selected_bucket_names = [bucket for bucket, _ in top_buckets]

        return selected_bucket_names, winner_distribution, qualified_buckets

    def _select_top_performers(self, videos: List[Dict]) -> List[Dict]:
        """
        Select top performers for analysis.

        Normal mode: top 100 if ≥100 available
        Degraded mode: all videos if < 100 available
        """
        if len(videos) < TOP_PERFORMERS_FOR_ANALYSIS:
            # Degraded mode
            logger.warning(
                f"Small dataset ({len(videos)} videos). Analyzing all available. "
                f"Statistical validity may be limited. Recommended: ≥100 videos."
            )
            return videos
        else:
            # Normal mode
            logger.info(f"Analyzing top {TOP_PERFORMERS_FOR_ANALYSIS} performers")
            return videos[:TOP_PERFORMERS_FOR_ANALYSIS]

    def _bucket_top_performers(self, top_performers: List[Dict]) -> Dict[str, int]:
        """
        Bucket top performers by duration.

        Returns:
            dict[str, int]: Winner distribution {bucket: count}
        """
        winner_distribution = {}
        skipped_count = 0

        for video in top_performers:
            duration = video.get("duration")
            if duration is None:
                logger.warning(f"Video {video.get('id', 'unknown')} has no duration. Skipping.")
                skipped_count += 1
                continue

            try:
                bucket = assign_bucket(duration)
                winner_distribution[bucket] = winner_distribution.get(bucket, 0) + 1
            except ValueError as e:
                # Skip videos with invalid duration (e.g., > 120s)
                logger.warning(
                    f"Video {video.get('id', 'unknown')} has invalid duration ({duration}s): {e}. Skipping."
                )
                skipped_count += 1
                continue

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} videos with invalid/missing duration during winner analysis")

        return winner_distribution

    def _calculate_winner_percentages(
        self,
        winner_distribution: Dict[str, int],
        total_winners: int
    ) -> Dict[str, float]:
        """
        Calculate winner concentration percentages.

        Uses len(top_performers) dynamically (not hardcoded 100) to handle degraded mode.
        """
        winner_percentages = {
            bucket: (count / total_winners) * 100
            for bucket, count in winner_distribution.items()
        }
        return winner_percentages

    def _filter_qualified_buckets(
        self,
        winner_percentages: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Filter buckets - only keep those with ≥ MIN_WINNER_PERCENTAGE (5%).

        This prevents processing buckets with only 1-4 winners (wasteful).

        Raises:
            ValueError: If no buckets qualify
        """
        qualified_buckets = {
            bucket: percentage
            for bucket, percentage in winner_percentages.items()
            if percentage >= MIN_WINNER_PERCENTAGE
        }

        if len(qualified_buckets) == 0:
            logger.error("No buckets qualified (≥5% winners required)")
            raise ValueError(
                f"No buckets qualified (≥{MIN_WINNER_PERCENTAGE}% winners required). "
                f"Winner distribution too fragmented. Try different target or broader date range."
            )

        return qualified_buckets

    def _select_top_buckets(
        self,
        qualified_buckets: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Sort qualified buckets by winner concentration (DESC) and select top 3.

        Returns:
            List of (bucket, percentage) tuples
        """
        top_buckets = sorted(
            qualified_buckets.items(),
            key=lambda x: x[1],  # Sort by percentage
            reverse=True
        )[:TOP_BUCKETS_TO_PROCESS]

        # Handle edge case - < 3 qualified buckets
        if len(top_buckets) < TOP_BUCKETS_TO_PROCESS:
            logger.warning(
                f"Only {len(top_buckets)} bucket(s) qualified (≥{MIN_WINNER_PERCENTAGE}% winners). "
                f"Processing {len(top_buckets)} bucket(s) instead of {TOP_BUCKETS_TO_PROCESS}."
            )

        return top_buckets

    def _log_winner_distribution(
        self,
        top_buckets: List[Tuple[str, float]],
        winner_distribution: Dict[str, int],
        total_winners: int
    ):
        """Log winner distribution with coverage statistics."""
        logger.info(f"Winner distribution ({total_winners} top performers):")
        for bucket, percentage in top_buckets:
            count = winner_distribution[bucket]
            logger.info(f"  - {bucket}: {count} videos ({percentage:.1f}%)")

        # Calculate total coverage
        total_coverage = sum(winner_distribution[b] for b, _ in top_buckets)
        coverage_percentage = (total_coverage / total_winners) * 100
        logger.info(
            f"Total winner coverage: {total_coverage}/{total_winners} ({coverage_percentage:.1f}%)"
        )
