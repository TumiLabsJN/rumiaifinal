"""
Cluster Scraping Orchestration Module

Handles multi-hashtag, multi-run scraping with error recovery and provenance tracking.

Source: HashtagVolumeV2_TI.md Section 3.2, 3.3, 3.4
"""

import logging
import time
from typing import List, Dict, Tuple

from .apify_scraper import ApifyScraper
from .constants import (
    RETRY_MAX_ATTEMPTS,
    RETRY_BACKOFF_DELAYS,
    EXIT_CODE_ALL_SCRAPES_FAILED
)

logger = logging.getLogger(__name__)


def run_cluster_scraping(
    cluster_config: dict,
    analysis_mode: str,
    country_code: str,
    date_filter: str,
    apify_scraper: ApifyScraper
) -> Tuple[List[Dict], List[Dict]]:
    """
    Orchestrate multi-hashtag, multi-run scraping with error recovery and progress logging.

    EXTENDS: VideoDiscoveryCHILDTI.md scrape_videos() function
    - ADDED: Multi-hashtag support (cluster_config.all_hashtags)
    - ADDED: Multi-run orchestration with delays
    - ADDED: Provenance tracking (source_hashtags, source_runs)
    - ADDED: Progress logging per scrape
    - ADDED: Retry logic with exponential backoff
    - PRESERVED: Apify scraper selection logic (from VideoDiscoveryCHILDTI.md)

    Source: HashtagVolumeV2_TI.md Section 3.2

    Args:
        cluster_config: dict, loaded from load_cluster_config()
                       Schema: ClusterConfigSchema (Section 2.2)
        analysis_mode: str, "top" or "recent" (from CLI parameter)
        country_code: str, "US" or "BR" or "global" (from CLI parameter)
        date_filter: str, "last_N_days" format (e.g., "last_90_days")
        apify_scraper: ApifyScraper, initialized scraper instance

    Returns:
        tuple[list[dict], list[dict]]:
            - all_videos: All scraped videos with provenance tracking
                         Each video has ExtendedVideoMetadataSchema (Section 2.3)
                         Videos NOT deduplicated yet (deduplication happens in Section 3.4)
            - failed_scrapes: List of failed scrape details
                             Schema: [{"hashtag": str, "run": int, "error": str}, ...]
                             Used by generate_cluster_analytics() for accurate reporting

    Raises:
        ValueError: if cluster_config invalid (caught before this function)
        Exception: if all scrapes fail (logs error, raises exception)
    """
    # Extract cluster parameters
    cluster_id = cluster_config['cluster_id']
    all_hashtags = [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']
    runs_per_hashtag = cluster_config['scrape_config']['runs_per_hashtag']
    delay_ms = cluster_config['scrape_config']['delay_between_runs_ms']
    results_per_page = cluster_config['scrape_config']['results_per_page']

    # Calculate total scrapes
    total_scrapes = len(all_hashtags) * runs_per_hashtag

    logger.info(f"\n{'='*80}")
    logger.info(f"CLUSTER SCRAPING: {cluster_id}")
    logger.info(f"{'='*80}")
    logger.info(f"Hashtags: {len(all_hashtags)} (primary + {len(all_hashtags)-1} variants)")
    logger.info(f"Runs per hashtag: {runs_per_hashtag}")
    logger.info(f"Total scrapes: {total_scrapes}")
    logger.info(f"Delay between scrapes: {delay_ms/1000:.0f}s ({delay_ms/60000:.1f} min)")
    logger.info(f"Results per scrape: {results_per_page}")
    logger.info(f"{'='*80}\n")

    print(f"\nCluster: {cluster_id} ({len(all_hashtags)} hashtags × {runs_per_hashtag} runs = {total_scrapes} scrapes)\n")

    # Initialize results
    all_videos = []
    scrape_num = 0
    failed_scrapes = []

    # Loop: hashtags × runs
    for hashtag in all_hashtags:
        for run in range(1, runs_per_hashtag + 1):
            scrape_num += 1
            print(f"[{scrape_num}/{total_scrapes}] Scraping {hashtag} (run {run})...", end=" ", flush=True)
            logger.info(f"Scraping {hashtag} (run {run}) - {scrape_num}/{total_scrapes}")

            # Error recovery: Retry 3x with exponential backoff
            videos = _scrape_with_retry(
                apify_scraper=apify_scraper,
                hashtag=hashtag,
                run_num=run,
                analysis_mode=analysis_mode,
                country_code=country_code,
                date_filter=date_filter,
                results_per_page=results_per_page,
                max_retries=RETRY_MAX_ATTEMPTS
            )

            if videos:
                print(f"✅ {len(videos)} videos")
                logger.info(f"✅ Scrape successful: {len(videos)} videos from {hashtag} run {run}")

                # Tag videos with source provenance
                for video in videos:
                    video['source_hashtags'] = [hashtag]  # Initialize as list
                    video['source_runs'] = [run]          # Initialize as list

                # Append to master list (provenance added by loop above)
                all_videos.extend(videos)
            else:
                print(f"❌ Failed after {RETRY_MAX_ATTEMPTS} retries")
                logger.error(f"❌ Scrape failed after {RETRY_MAX_ATTEMPTS} retries: {hashtag} run {run}")
                failed_scrapes.append({
                    "hashtag": hashtag,
                    "run": run,
                    "error": "Failed after retries"
                })

            # Delay between scrapes (except after last scrape)
            if scrape_num < total_scrapes:
                delay_seconds = delay_ms / 1000
                print(f"    Waiting {delay_seconds / 60:.0f} min before next scrape...", flush=True)
                logger.debug(f"Delaying {delay_seconds}s before next scrape")
                time.sleep(delay_seconds)

    # Log summary
    print()  # Blank line after all scrapes
    logger.info(f"\n{'='*80}")
    logger.info(f"CLUSTER SCRAPING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total videos scraped: {len(all_videos)}")
    logger.info(f"Successful scrapes: {total_scrapes - len(failed_scrapes)}/{total_scrapes}")

    if failed_scrapes:
        logger.warning(f"⚠️  {len(failed_scrapes)} scrape(s) failed:")
        for failure in failed_scrapes:
            logger.warning(f"   - {failure['hashtag']} run {failure['run']}: {failure['error']}")
    else:
        logger.info("✅ All scrapes successful!")

    logger.info(f"{'='*80}\n")

    # Check if all scrapes failed
    if len(all_videos) == 0:
        error_msg = (
            f"All {len(failed_scrapes)} scrapes failed. No videos available for analysis.\n"
            f"Check Apify API key, network connectivity, and target validity.\n"
            f"Failed scrapes:\n"
        )
        for failure in failed_scrapes:
            error_msg += f"  - {failure['hashtag']} run {failure['run']}: {failure['error']}\n"

        logger.error(error_msg)
        raise Exception(error_msg)

    return all_videos, failed_scrapes


def _scrape_with_retry(
    apify_scraper: ApifyScraper,
    hashtag: str,
    run_num: int,
    analysis_mode: str,
    country_code: str,
    date_filter: str,
    results_per_page: int,
    max_retries: int = 3
) -> List[Dict]:
    """
    Scrape single hashtag with automatic retry on failure.

    Source: HashtagVolumeV2_TI.md Section 3.3

    Retry policy: Exponential backoff (5s, 15s, 45s)
    Failure handling: Skip scrape after max retries, continue with cluster

    Args:
        apify_scraper: ApifyScraper, initialized scraper instance
        hashtag: str, hashtag to scrape (e.g., "#nutrition")
        run_num: int, run number (1-5)
        analysis_mode: str, "top" or "recent"
        country_code: str, "US" or "BR" or "global"
        date_filter: str, "last_N_days" format
        results_per_page: int, videos per scrape (100-800)
        max_retries: int, maximum retry attempts (default: 3)

    Returns:
        list[dict]: Scraped videos (ApifyVideoMetadataSchema)
                    Empty list if all retries fail

    Raises:
        None - all errors caught and logged, returns [] on failure
    """
    backoff_delays = RETRY_BACKOFF_DELAYS  # [5, 15, 45] seconds

    for attempt in range(max_retries):
        try:
            # Use the updated scrape_videos method that accepts results_per_page
            # We need to temporarily override APIFY_SCRAPE_COUNT
            from . import constants
            original_count = constants.APIFY_SCRAPE_COUNT
            constants.APIFY_SCRAPE_COUNT = results_per_page

            try:
                # Call Apify scraper (inherited from VideoDiscoveryCHILDTI.md)
                videos = apify_scraper.scrape_videos(
                    analysis_type="hashtag",  # Always hashtag for cluster scraping
                    target=hashtag,
                    analysis_mode=analysis_mode,
                    date_filter=date_filter,
                    country_code=country_code
                )
            finally:
                # Restore original count
                constants.APIFY_SCRAPE_COUNT = original_count

            # Success - return immediately
            logger.debug(f"Successfully scraped {hashtag} run {run_num}: {len(videos)} videos")
            return videos

        except Exception as e:
            if attempt < max_retries - 1:
                # Retry with backoff
                delay = backoff_delays[attempt]
                logger.warning(f"  Retry {attempt+1}/{max_retries} in {delay}s... (Error: {str(e)})")
                print(f" (retry {attempt+1}/{max_retries} in {delay}s...)", end="", flush=True)
                time.sleep(delay)
            else:
                # All retries exhausted
                logger.error(f"  Skipping {hashtag} run {run_num} after {max_retries} failed attempts")
                logger.error(f"  Final error: {str(e)}")
                return []  # Return empty list, continue with remaining scrapes

    # Should never reach here (return in loop), but safety fallback
    return []
