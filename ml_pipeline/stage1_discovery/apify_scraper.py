"""
Stage 1.1: Apify Scraping

Scrapes videos from TikTok using Apify API with deduplication and engagement sorting.

Source: VideoDiscoveryCHILDTI.md Section 4.1
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Dict

from .constants import (
    APIFY_PROFILE_SCRAPER_ID,
    APIFY_HASHTAG_SCRAPER_ID,
    APIFY_SCRAPE_COUNT,
    APIFY_TIMEOUT,
    APIFY_RETRY_COUNT,
    APIFY_RETRY_BACKOFF,
    EXIT_CODE_APIFY_KEY_MISSING,
    EXIT_CODE_APIFY_TIMEOUT,
    EXIT_CODE_ALL_DUPLICATES,
)


logger = logging.getLogger(__name__)


class ApifyScraper:
    """
    Handles video scraping from TikTok via Apify API.

    Implements:
    - Actor selection based on analysis type
    - Retry logic with exponential backoff
    - Deduplication by video ID
    - Engagement-based sorting
    """

    def __init__(self, apify_api_key: str):
        """
        Initialize Apify scraper.

        Args:
            apify_api_key: Apify account API key

        Raises:
            ValueError: If apify_api_key is None or empty
        """
        if not apify_api_key:
            logger.error("APIFY_API_KEY environment variable not set")
            raise ValueError(
                "APIFY_API_KEY environment variable required. "
                "Obtain from: https://console.apify.com/account/integrations"
            )

        self.api_key = apify_api_key
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Apify client."""
        if self._client is None:
            from apify_client import ApifyClient
            self._client = ApifyClient(self.api_key)
        return self._client

    def scrape_videos(
        self,
        analysis_type: str,
        target: str,
        analysis_mode: str,
        date_filter: str = "last_90_days",
        country_code: str = "US"
    ) -> List[Dict]:
        """
        Scrape videos from TikTok via Apify API with deduplication and engagement sorting.

        Args:
            analysis_type: "hashtag", "competitor", or "creator"
            target: Target identifier (#nutrition, @handle)
            analysis_mode: "top" or "recent"
            date_filter: "last_N_days" format (e.g., "last_90_days")
            country_code: "US", "BR", or "global" (geographic filtering)

        Returns:
            List of unique video metadata objects sorted by playCount DESC

        Raises:
            ValueError: If analysis_type invalid or all videos are duplicates
            TimeoutError: If Apify scraping times out after retries
            ConnectionError: If Apify rate limit exceeded
        """
        logger.info(
            f"Starting Apify scraping: type={analysis_type}, target={target}, mode={analysis_mode}"
        )

        # Step 1: Select scraper and build input params
        actor_id, input_params = self._build_scraper_config(
            analysis_type, target, analysis_mode, date_filter, country_code
        )

        # Step 2: Run Apify scraper with retry logic
        videos = self._run_scraper_with_retry(actor_id, input_params, target, analysis_mode)

        # Step 3: Deduplicate by video ID
        unique_videos = self._deduplicate_videos(videos)

        # Step 4: Client-side engagement sorting
        sorted_videos = self._sort_by_engagement(unique_videos)

        # Step 5: Record scrape timestamp
        scrape_timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"Scrape timestamp: {scrape_timestamp}")

        return sorted_videos

    def _build_scraper_config(
        self,
        analysis_type: str,
        target: str,
        analysis_mode: str,
        date_filter: str,
        country_code: str
    ) -> tuple:
        """
        Build unified Profile Scraper config for both hashtags and profiles.

        Uses Profile Scraper (GdWCkxBtKWOsKjdch) for all analysis types.
        Supports native date filtering and geography filtering.

        Returns:
            tuple: (actor_id, input_params)
        """
        # Use unified Profile Scraper for both hashtags and profiles
        actor_id = APIFY_PROFILE_SCRAPER_ID

        # Base parameters (common to all analysis types)
        input_params = {
            "resultsPerPage": APIFY_SCRAPE_COUNT,
            "shouldDownloadCovers": False,
            "shouldDownloadVideos": False,
            "shouldDownloadSubtitles": False,
            "shouldDownloadSlideshowImages": False,
        }

        # Add target-specific parameters
        if analysis_type == "hashtag":
            input_params["hashtags"] = [target]  # e.g., ["#nutrition"] - keep # prefix
        elif analysis_type in ["competitor", "creator"]:
            input_params["profiles"] = [target]  # e.g., ["@rival_brand"] - keep @ prefix
        else:
            raise ValueError(f"Invalid analysis_type: {analysis_type}")

        # Add date filtering (native Apify parameters)
        oldest_date, newest_date = self._calculate_date_range(date_filter)
        input_params["oldestPostDateUnified"] = oldest_date
        input_params["newestPostDate"] = newest_date

        # Add geography filtering (proxyCountryCode)
        if country_code != "global":
            input_params["proxyCountryCode"] = country_code  # "US" or "BR"
        # If "global", omit proxyCountryCode (default Apify routing)

        # Add sorting parameter for analysis_mode
        if analysis_mode == "recent":
            input_params["profileSorting"] = "latest"
        # If "top", omit profileSorting (Apify defaults to engagement-based)

        logger.info(f"Selected actor: {actor_id} (unified Profile Scraper)")
        logger.info(f"Input params: {input_params}")

        return actor_id, input_params

    def _calculate_date_range(self, date_filter: str) -> tuple:
        """
        Calculate Apify date range parameters from date_filter.

        Args:
            date_filter: "last_N_days" format (e.g., "last_90_days")

        Returns:
            tuple: (oldestPostDateUnified, newestPostDate) in YYYY-MM-DD format
        """
        from datetime import datetime, timedelta, timezone

        # Parse days from date_filter (e.g., "last_90_days" → 90)
        days = int(date_filter.replace("last_", "").replace("_days", ""))

        # Calculate dates
        newest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        oldest_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        logger.info(f"Date range: {oldest_date} to {newest_date} ({days} days)")

        return oldest_date, newest_date

    def _run_scraper_with_retry(
        self,
        actor_id: str,
        input_params: dict,
        target: str,
        analysis_mode: str
    ) -> List[Dict]:
        """
        Run Apify scraper with retry logic and exponential backoff.

        Returns:
            List of video metadata objects (may contain duplicates)
        """
        videos = []

        for attempt in range(APIFY_RETRY_COUNT):
            try:
                logger.info(f"Starting Apify scraper (attempt {attempt + 1}/{APIFY_RETRY_COUNT})")

                # Run actor and wait for completion
                run = self.client.actor(actor_id).call(
                    run_input=input_params,
                    timeout_secs=APIFY_TIMEOUT
                )

                # Fetch results from dataset
                dataset_items = self.client.dataset(run["defaultDatasetId"]).list_items().items
                videos = dataset_items

                logger.info(f"Apify scraping complete: {len(videos)} videos returned")

                # Debug: Log first video's fields to see what we got
                if videos and len(videos) > 0:
                    sample = videos[0]
                    logger.info(f"Sample video fields: {list(sample.keys())}")

                    # Check if duration is nested in videoMeta
                    if 'videoMeta' in sample and sample['videoMeta']:
                        logger.info(f"videoMeta fields: {list(sample['videoMeta'].keys())}")
                        if 'duration' in sample['videoMeta']:
                            logger.info(f"Duration found in videoMeta: {sample['videoMeta']['duration']}")

                # Flatten duration from videoMeta to top level for all videos
                for video in videos:
                    if 'videoMeta' in video and video['videoMeta'] and 'duration' in video['videoMeta']:
                        video['duration'] = video['videoMeta']['duration']

                break  # Success, exit retry loop

            except TimeoutError as e:
                wait_time = APIFY_RETRY_BACKOFF[attempt] if attempt < len(APIFY_RETRY_BACKOFF) else 45
                logger.warning(f"Apify timeout (attempt {attempt + 1}). Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

                if attempt == APIFY_RETRY_COUNT - 1:
                    # Final retry failed
                    logger.error(f"Apify scraping timeout after {APIFY_RETRY_COUNT} retries")
                    raise TimeoutError(
                        f"Apify scraping timeout after {APIFY_RETRY_COUNT} retries. "
                        f"Check network connection or increase APIFY_TIMEOUT."
                    ) from e

            except Exception as e:
                error_str = str(e).lower()
                if "429" in str(e) or "rate limit" in error_str:
                    # Rate limit exceeded
                    logger.warning("Apify rate limit exceeded. Waiting 60s before retry...")
                    time.sleep(60)
                    continue
                else:
                    # Unknown error, fail-fast
                    logger.error(f"Apify scraping error: {e}")
                    raise

        # Validate scraper returned data
        if len(videos) < 100:
            logger.warning(
                f"Only {len(videos)} videos scraped (expected {APIFY_SCRAPE_COUNT}). "
                f"Proceeding with available data."
            )

        return videos

    def _deduplicate_videos(self, videos: List[Dict]) -> List[Dict]:
        """
        Deduplicate videos by video ID (keep first occurrence).

        TikTok videos can appear multiple times due to:
        - Reposts
        - Cross-hashtag appearances
        - Apify duplicates
        """
        seen_ids = set()
        unique_videos = []

        for video in videos:
            video_id = video.get("id")
            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                unique_videos.append(video)

        # Log deduplication stats
        duplicate_count = len(videos) - len(unique_videos)
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(videos)) * 100
            logger.info(f"Removed {duplicate_count} duplicate videos ({duplicate_percentage:.1f}%)")

        # Check for extreme duplication (data quality issue)
        if len(videos) > 0 and (duplicate_count / len(videos)) > 0.95:
            logger.error(
                f"All scraped videos are duplicates ({len(unique_videos)} unique from {len(videos)} scraped)"
            )
            raise ValueError(
                f"All scraped videos are duplicates ({len(unique_videos)} unique from {len(videos)} scraped). "
                f"Data quality issue. Check target or Apify scraper configuration."
            )

        logger.info(f"Scraped {len(videos)} videos → {len(unique_videos)} unique")

        return unique_videos

    def _sort_by_engagement(self, videos: List[Dict]) -> List[Dict]:
        """
        Sort videos by playCount (views) descending for success-based analysis.

        Apify returns videos in default order (not sorted by engagement).
        """
        sorted_videos = sorted(videos, key=lambda v: v.get("playCount", 0), reverse=True)

        logger.info(f"Sorted {len(sorted_videos)} videos by engagement (playCount DESC)")

        return sorted_videos
