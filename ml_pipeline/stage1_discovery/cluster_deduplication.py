"""
Cluster Deduplication with Provenance Tracking

Deduplicates videos while tracking all source hashtags and runs.

Source: HashtagVolumeV2_TI.md Section 3.4
"""

import logging
from typing import List, Dict, Tuple

from .constants import EXIT_CODE_ALL_SCRAPES_FAILED

logger = logging.getLogger(__name__)


def deduplicate_with_provenance(
    all_videos: List[Dict],
    cluster_config: dict,
    failed_scrapes: List[Dict]
) -> Tuple[List[Dict], Dict]:
    """
    Deduplicate videos while tracking full source provenance.

    EXTENDS: VideoDiscoveryCHILDTI.md deduplication logic
    - PRESERVED: Video ID-based deduplication (keeps first occurrence)
    - ADDED: source_hashtags tracking (ALL hashtags that found this video)
    - ADDED: source_runs tracking (ALL runs that found this video)
    - ADDED: Cluster analytics generation

    Source: HashtagVolumeV2_TI.md Section 3.4

    Args:
        all_videos: list[dict], all scraped videos from run_cluster_scraping()
                    Example: 1,939 videos from 8 scrapes (4 hashtags × 2 runs)
                    Each video has: source_hashtags=[hashtag], source_runs=[run]
        cluster_config: dict, cluster configuration (for analytics generation)
                       Schema: ClusterConfigSchema (Section 2.2)
        failed_scrapes: list[dict], failed scrape details from run_cluster_scraping()
                       Schema: [{"hashtag": str, "run": int, "error": str}, ...]

    Returns:
        tuple:
            - unique_videos: list[dict], deduplicated videos with complete provenance
                           Example: 1,400 unique videos
                           Each video has: source_hashtags=[list of all hashtags], source_runs=[list of all runs]
            - analytics: dict, cluster health analytics
                        Schema: ClusterAnalyticsSchema (Section 2.2)

    Raises:
        ValueError: if all_videos is empty (all scrapes failed)
    """
    # DEFENSIVE CHECK: Handle empty input
    if len(all_videos) == 0:
        logger.error("No videos to deduplicate. All scrapes failed.")
        logger.error(f"Failed scrapes: {len(failed_scrapes)}/{len(failed_scrapes)}")
        for failure in failed_scrapes:
            logger.error(f"  - {failure['hashtag']} run {failure['run']}: {failure['error']}")

        raise ValueError(
            f"All {len(failed_scrapes)} scrapes failed. No videos available for analysis.\n"
            f"Check Apify API key, network connectivity, and target validity."
        )

    print(f"\nDeduplicating {len(all_videos)} videos...", end=" ", flush=True)
    logger.info(f"Starting deduplication: {len(all_videos)} total videos")

    unique_videos_map = {}

    for video in all_videos:
        video_id = video['id']
        source_hashtag = video['source_hashtags'][0]  # Current scrape's hashtag (single-element list)
        source_run = video['source_runs'][0]          # Current scrape's run (single-element list)

        if video_id not in unique_videos_map:
            # First occurrence - initialize tracking
            # video already has source_hashtags and source_runs from run_cluster_scraping()
            unique_videos_map[video_id] = video
        else:
            # Duplicate - append to tracking arrays
            existing = unique_videos_map[video_id]

            # Append source_hashtag if not already tracked
            if source_hashtag not in existing['source_hashtags']:
                existing['source_hashtags'].append(source_hashtag)

            # Append source_run if not already tracked
            if source_run not in existing['source_runs']:
                existing['source_runs'].append(source_run)

    # Convert map to list
    unique_videos = list(unique_videos_map.values())

    # Calculate duplication rate
    duplication_rate = (len(all_videos) - len(unique_videos)) / len(all_videos) * 100

    print(f"✅ {len(unique_videos)} unique ({duplication_rate:.1f}% overlap)")
    logger.info(f"Deduplication complete: {len(unique_videos)} unique videos ({duplication_rate:.1f}% overlap)")
    logger.info(f"  Total scraped: {len(all_videos)}")
    logger.info(f"  Duplicates removed: {len(all_videos) - len(unique_videos)}")

    # Generate cluster analytics (will be implemented in cluster_analytics.py)
    from .cluster_analytics import generate_cluster_analytics
    analytics = generate_cluster_analytics(all_videos, unique_videos, cluster_config, failed_scrapes)

    return unique_videos, analytics
