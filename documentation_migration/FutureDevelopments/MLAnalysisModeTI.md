# Analysis Mode System - Technical Implementation

**Related HLD**: [MLAnalysisMode.md](./MLAnalysisMode.md)
**Status**: Implementation Ready
**Last Updated**: 2025-10-01

---

## Overview

This document contains the technical implementation details for the Analysis Mode System (top vs recent). For high-level design, business context, and decision rationale, see [MLAnalysisMode.md](./MLAnalysisMode.md).

---

## 1. Apify Client Implementation

### 1.1 Dual-Scraper Architecture

**File**: `/home/jorge/rumiaifinal/rumiai_v2/api/apify_client.py`

```python
"""
Apify API client for RumiAI v2 - ML Batch Processing
Supports both hashtag and profile scraping with dual-mode analysis
"""
import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

from ..core.exceptions import APIError
from ..core.models import VideoMetadata

logger = logging.getLogger(__name__)


class ApifyClient:
    """
    Apify API client for TikTok scraping.

    Supports:
    - Hashtag scraping (clockworks/tiktok-hashtag-scraper)
    - Profile scraping (clockworks/tiktok-scraper)
    - Top mode (engagement-based sorting)
    - Recent mode (date-based sorting)
    """

    def __init__(self, api_token: str):
        self.api_token = api_token

        # Two actor IDs based on use case
        self.profile_scraper_id = "GdWCkxBtKWOsKjdch"  # clockworks/tiktok-scraper
        self.hashtag_scraper_id = "TBD_FROM_APIFY_DASHBOARD"  # clockworks/tiktok-hashtag-scraper

        self.base_url = "https://api.apify.com/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}"
        }

    async def scrape_hashtag(
        self,
        hashtag: str,
        video_count: int = 800,
        analysis_mode: str = "top"  # "top" or "recent"
    ) -> List[VideoMetadata]:
        """
        Scrape videos from hashtag for ML batch processing.

        Args:
            hashtag: Hashtag to scrape (with or without #)
            video_count: Number of videos to scrape (default: 800)
            analysis_mode: "top" (engagement) or "recent" (date)

        Returns:
            List of VideoMetadata objects

        Example:
            videos = await client.scrape_hashtag("#nutrition", 800, "top")
        """
        logger.info(f"Scraping hashtag {hashtag} - mode: {analysis_mode}, count: {video_count}")

        # Determine sorting parameter
        sort_by = "engagement" if analysis_mode == "top" else "date"

        # Prepare actor input
        actor_input = {
            "hashtagsUrls": [f"https://www.tiktok.com/tag/{hashtag.lstrip('#')}"],
            "resultsPerPage": video_count,
            "shouldDownloadVideos": True,
            "sortBy": sort_by,
            "sortOrder": "desc",
            "proxyConfiguration": {
                "useApifyProxy": True
            }
        }

        try:
            # Run scraper with hashtag-specific actor
            videos = await self._run_scraper(
                self.hashtag_scraper_id,
                actor_input
            )

            logger.info(f"Successfully scraped {len(videos)} videos from {hashtag}")
            return videos

        except Exception as e:
            raise APIError(
                'Apify',
                0,
                f"Hashtag scraping failed for {hashtag}: {str(e)}",
                hashtag
            )

    async def scrape_profile(
        self,
        handle: str,
        video_count: int,
        analysis_mode: str = "top"  # "top" or "recent"
    ) -> List[VideoMetadata]:
        """
        Scrape videos from TikTok profile for competitor/creator analysis.

        Args:
            handle: TikTok handle (with or without @)
            video_count: Number of videos to scrape
            analysis_mode: "top" (engagement) or "recent" (date)

        Returns:
            List of VideoMetadata objects

        Example:
            videos = await client.scrape_profile("@rival_brand", 150, "top")
        """
        logger.info(f"Scraping profile {handle} - mode: {analysis_mode}, count: {video_count}")

        # Determine sorting parameter
        sort_by = "engagement" if analysis_mode == "top" else "date"

        # Prepare actor input
        actor_input = {
            "profilesUrls": [f"https://www.tiktok.com/{handle.lstrip('@')}"],
            "resultsPerPage": video_count,
            "shouldDownloadVideos": True,
            "sortBy": sort_by,
            "sortOrder": "desc",
            "proxyConfiguration": {
                "useApifyProxy": True
            }
        }

        try:
            # Run scraper with profile-specific actor
            videos = await self._run_scraper(
                self.profile_scraper_id,
                actor_input
            )

            logger.info(f"Successfully scraped {len(videos)} videos from {handle}")
            return videos

        except Exception as e:
            raise APIError(
                'Apify',
                0,
                f"Profile scraping failed for {handle}: {str(e)}",
                handle
            )

    async def _run_scraper(
        self,
        actor_id: str,
        actor_input: Dict[str, Any]
    ) -> List[VideoMetadata]:
        """
        Run Apify actor and return parsed VideoMetadata objects.

        Internal method used by scrape_hashtag() and scrape_profile().
        """
        # Start actor run
        run_info = await self._start_actor_run(actor_id, actor_input)

        # Handle both 'id' and 'data' response formats
        if 'data' in run_info and 'id' in run_info['data']:
            run_id = run_info['data']['id']
        elif 'id' in run_info:
            run_id = run_info['id']
        else:
            raise KeyError(f"No run ID found in response: {run_info}")

        logger.info(f"Apify run started: {run_id}")

        # Wait for completion
        run_result = await self._wait_for_run(run_id, timeout=600)  # 10 min timeout for large batches

        if run_result['status'] != 'SUCCEEDED':
            raise APIError(
                'Apify',
                0,
                f"Actor run failed with status: {run_result['status']}"
            )

        # Get dataset items
        dataset_id = run_result['defaultDatasetId']
        items = await self._get_dataset_items(dataset_id)

        if not items:
            raise APIError('Apify', 0, "No video data returned")

        # Convert to VideoMetadata objects
        videos = []
        for item in items:
            try:
                metadata = VideoMetadata.from_apify_data(item)
                videos.append(metadata)
            except Exception as e:
                logger.error(f"Failed to parse video data: {e}")
                # Continue processing other videos

        return videos

    async def _start_actor_run(
        self,
        actor_id: str,
        actor_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start an Apify actor run."""
        url = f"{self.base_url}/acts/{actor_id}/runs"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self.headers,
                json=actor_input
            ) as response:
                text = await response.text()
                logger.info(f"Apify API response status: {response.status}")

                if response.status not in [200, 201]:
                    raise APIError('Apify', response.status, f"Failed to start actor: {text}")

                try:
                    data = json.loads(text)
                    return data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {text[:500]}")
                    raise APIError('Apify', response.status, f"Invalid JSON response")

    async def _wait_for_run(self, run_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Wait for actor run to complete."""
        url = f"{self.base_url}/actor-runs/{run_id}"
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    raise APIError('Apify', 0, f"Run {run_id} timed out after {timeout}s")

                # Get run status
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise APIError('Apify', response.status, f"Failed to get run status: {text}")

                    run_data = await response.json()
                    status = run_data['data']['status']

                    logger.info(f"Run {run_id} status: {status}")

                    if status in ['SUCCEEDED', 'FAILED', 'ABORTED']:
                        return run_data['data']

                # Wait before next check
                await asyncio.sleep(5)  # Check every 5 seconds for large batches

    async def _get_dataset_items(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get items from Apify dataset."""
        url = f"{self.base_url}/datasets/{dataset_id}/items"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise APIError('Apify', response.status, f"Failed to get dataset items: {text}")

                return await response.json()

    # Keep existing single video methods for backward compatibility
    async def scrape_video(self, video_url: str) -> VideoMetadata:
        """
        Scrape single video metadata from TikTok.

        Legacy method for single video scraping (rumiai_runner.py).
        For batch processing, use scrape_hashtag() or scrape_profile().
        """
        logger.info(f"Scraping video: {video_url}")

        actor_input = {
            "postURLs": [video_url],
            "resultsPerPage": 1,
            "shouldDownloadVideos": True,
            "shouldDownloadCovers": True,
            "shouldDownloadSubtitles": True,
            "proxyConfiguration": {
                "useApifyProxy": True
            }
        }

        try:
            videos = await self._run_scraper(self.profile_scraper_id, actor_input)
            if not videos:
                raise APIError('Apify', 0, "No video data returned", video_url)

            logger.info(f"Successfully scraped video {videos[0].video_id}")
            return videos[0]

        except Exception as e:
            if isinstance(e, APIError):
                raise
            else:
                raise APIError('Apify', 0, f"Scraping failed: {str(e)}", video_url)

    async def download_video(
        self,
        download_url: str,
        video_id: str,
        output_dir: Path = Path("temp")
    ) -> Path:
        """
        Download video file from Apify storage.

        Returns path to downloaded video file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_id}.mp4"

        # Check if already downloaded
        if output_path.exists():
            logger.info(f"Video already downloaded: {output_path}")
            return output_path

        logger.info(f"Downloading video {video_id} from {download_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url, headers=self.headers) as response:
                    if response.status != 200:
                        raise APIError(
                            'Apify',
                            response.status,
                            f"Failed to download video: {response.status}",
                            video_id
                        )

                    # Download in chunks
                    with open(output_path, 'wb') as f:
                        chunk_size = 8192
                        downloaded = 0

                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress every 10MB
                            if downloaded % (10 * 1024 * 1024) == 0:
                                logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB")

            logger.info(f"Successfully downloaded video to {output_path}")
            return output_path

        except Exception as e:
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()

            if isinstance(e, APIError):
                raise
            else:
                raise APIError('Apify', 0, f"Download failed: {str(e)}", video_id)
```

---

## 2. Client-Side Date Filtering

### 2.1 Date Filter Implementation

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/video_filters.py` (NEW)

```python
"""
Video filtering utilities for ML batch processing
"""
from typing import List
from datetime import datetime, timedelta
import logging

from ..core.models import VideoMetadata

logger = logging.getLogger(__name__)


def filter_by_date(
    videos: List[VideoMetadata],
    date_filter: str
) -> List[VideoMetadata]:
    """
    Client-side date filtering (required for hashtag scraping).

    Args:
        videos: List of VideoMetadata objects from Apify
        date_filter: Date constraint in one of two formats:
            - "last_N_days" (e.g., "last_90_days")
            - "YYYY-MM-DD:YYYY-MM-DD" (e.g., "2024-01-01:2025-01-01")

    Returns:
        Filtered list of videos matching date criteria

    Examples:
        # Keep videos from last 90 days
        filtered = filter_by_date(videos, "last_90_days")

        # Keep videos in specific date range
        filtered = filter_by_date(videos, "2024-06-01:2024-12-01")
    """
    logger.info(f"Filtering {len(videos)} videos by date: {date_filter}")

    # Parse date filter
    if date_filter.startswith("last_"):
        # Extract number of days
        days_str = date_filter.replace("last_", "").replace("_days", "")
        try:
            days = int(days_str)
        except ValueError:
            raise ValueError(f"Invalid date filter format: {date_filter}")

        min_date = datetime.now() - timedelta(days=days)
        max_date = datetime.now()

        logger.info(f"Filtering for videos from {min_date.date()} to {max_date.date()}")

    elif ":" in date_filter:
        # Date range format
        try:
            start_str, end_str = date_filter.split(":")
            min_date = datetime.fromisoformat(start_str)
            max_date = datetime.fromisoformat(end_str)
        except ValueError:
            raise ValueError(f"Invalid date range format: {date_filter}")

        logger.info(f"Filtering for videos from {min_date.date()} to {max_date.date()}")

    else:
        raise ValueError(
            f"Invalid date filter: {date_filter}. "
            "Use 'last_N_days' or 'YYYY-MM-DD:YYYY-MM-DD'"
        )

    # Filter videos
    filtered = [
        v for v in videos
        if min_date <= v.create_time <= max_date
    ]

    retention_rate = len(filtered) / len(videos) * 100 if videos else 0
    logger.info(
        f"Date filtering complete: {len(filtered)}/{len(videos)} videos retained "
        f"({retention_rate:.1f}%)"
    )

    return filtered


def filter_by_engagement_threshold(
    videos: List[VideoMetadata],
    min_views: int = 1000,
    min_engagement_rate: float = 0.02
) -> List[VideoMetadata]:
    """
    Filter out low-quality videos before ML processing.

    Args:
        videos: List of VideoMetadata objects
        min_views: Minimum view count (default: 1000)
        min_engagement_rate: Minimum engagement rate (default: 2%)

    Returns:
        Filtered list of videos meeting quality thresholds
    """
    logger.info(
        f"Filtering {len(videos)} videos by engagement "
        f"(min views: {min_views}, min rate: {min_engagement_rate})"
    )

    filtered = []
    for video in videos:
        # Check minimum views
        if video.views < min_views:
            continue

        # Calculate engagement rate
        total_engagement = video.likes + video.comments + video.shares
        engagement_rate = total_engagement / video.views if video.views > 0 else 0

        # Check minimum engagement rate
        if engagement_rate < min_engagement_rate:
            continue

        filtered.append(video)

    retention_rate = len(filtered) / len(videos) * 100 if videos else 0
    logger.info(
        f"Engagement filtering complete: {len(filtered)}/{len(videos)} videos retained "
        f"({retention_rate:.1f}%)"
    )

    return filtered


def calculate_engagement_score(video: VideoMetadata) -> float:
    """
    Calculate composite engagement score with share boost factor.

    Formula: views × (1 + share_rate × 10)

    Args:
        video: VideoMetadata object

    Returns:
        Engagement score (higher = more viral potential)

    See MLAnalysisMode.md for detailed rationale.
    """
    views = video.views
    shares = video.shares

    # Calculate share rate (shares as % of views)
    share_rate = shares / max(views, 1)

    # Share boost factor: 10x multiplier means 1% share rate = 10% boost
    share_boost = 1 + (share_rate * 10)

    # Final score
    engagement_score = views * share_boost

    return engagement_score


def sort_by_engagement(videos: List[VideoMetadata]) -> List[VideoMetadata]:
    """
    Sort videos by composite engagement score (descending).

    Args:
        videos: List of VideoMetadata objects

    Returns:
        Sorted list (highest engagement first)
    """
    # Calculate engagement scores
    videos_with_scores = [
        (video, calculate_engagement_score(video))
        for video in videos
    ]

    # Sort by score (descending)
    sorted_videos = sorted(
        videos_with_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Return just the videos
    return [video for video, score in sorted_videos]


def sort_by_date(videos: List[VideoMetadata]) -> List[VideoMetadata]:
    """
    Sort videos by publish date (newest first).

    Args:
        videos: List of VideoMetadata objects

    Returns:
        Sorted list (most recent first)
    """
    return sorted(videos, key=lambda v: v.create_time, reverse=True)
```

---

## 3. CLI Argument Parsing

### 3.1 Analysis Mode Flag Handling

**File**: `/home/jorge/rumiaifinal/scripts/rumiai_ml_batch.py` (NEW)

```python
"""
ML Batch Processing CLI - Main entry point for bulk video analysis
"""
import argparse
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse CLI arguments for ML batch processing.

    Supports three analysis types: hashtag, competitor, creator
    Each with top/recent mode support
    """
    parser = argparse.ArgumentParser(
        description="RumiAI ML Batch Processing - Analyze videos at scale"
    )

    # Required arguments
    parser.add_argument(
        '--client',
        required=True,
        help='Client identifier (e.g., "client_acme_corp")'
    )

    parser.add_argument(
        '--analysis-type',
        required=True,
        choices=['hashtag', 'competitor', 'creator'],
        help='Type of analysis to perform'
    )

    parser.add_argument(
        '--target',
        required=True,
        help='Analysis target: #hashtag for hashtag, @handle for competitor/creator'
    )

    # Optional arguments
    parser.add_argument(
        '--video-count',
        type=int,
        default=None,
        help='Number of videos to scrape (default: 300 for hashtag, 150 for competitor, 40 for creator)'
    )

    parser.add_argument(
        '--date-filter',
        default='last_90_days',
        help='Date filter: "last_N_days" or "YYYY-MM-DD:YYYY-MM-DD" (default: last_90_days)'
    )

    parser.add_argument(
        '--analysis-mode',
        choices=['top', 'recent'],
        default=None,
        help='Analysis mode: "top" (engagement) or "recent" (date). Auto-defaults per analysis type if not specified.'
    )

    parser.add_argument(
        '--compare-to',
        help='For creator analysis: what to compare against (format: "hashtag:nutrition" or "competitor:rival_brand")'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force restart (discard existing checkpoint)'
    )

    args = parser.parse_args()

    # Apply smart defaults
    if args.video_count is None:
        if args.analysis_type == 'hashtag':
            args.video_count = 300
        elif args.analysis_type == 'competitor':
            args.video_count = 150
        else:  # creator
            args.video_count = 40

    if args.analysis_mode is None:
        if args.analysis_type == 'creator':
            args.analysis_mode = 'recent'  # Creator vetting needs natural style
        else:
            args.analysis_mode = 'top'  # Hashtag/competitor default to best work

    return args


async def main():
    """Main entry point for ML batch processing."""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("RumiAI ML Batch Processing")
    logger.info("=" * 80)
    logger.info(f"Client: {args.client}")
    logger.info(f"Analysis Type: {args.analysis_type}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Video Count: {args.video_count}")
    logger.info(f"Date Filter: {args.date_filter}")
    logger.info(f"Analysis Mode: {args.analysis_mode}")
    logger.info("=" * 80)

    # TODO: Implement batch processing pipeline
    # 1. Initialize Apify client
    # 2. Scrape videos (hashtag or profile based on analysis_type)
    # 3. Apply date filtering
    # 4. Bucket by duration
    # 5. Select videos for processing
    # 6. Run RumiAI analysis with checkpoint/resume
    # 7. Train ML models
    # 8. Generate reports

    logger.info("✅ Batch processing complete!")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Checkpoint Integration

### 4.1 Using CheckpointManager with Analysis Mode

**CheckpointManager validates `analysis_mode`** on resume to prevent mixing top/recent modes.

**Critical**: Analysis mode is stored in checkpoint config and must match when resuming. Prevents data corruption from switching between engagement-sorted and date-sorted video sets.

**Implementation**: See [MLCheckpointResumeTI.md](./MLCheckpointResumeTI.md) for complete `CheckpointManager` class and usage examples.

**Usage Example**:

```python
from rumiai_v2.processors.checkpoint_manager import CheckpointManager

# Initialize checkpoint
checkpoint = CheckpointManager(client_id, "hashtag", "#nutrition")

# Save config including analysis_mode
config = {
    "video_count": 300,
    "date_filter": "last_90_days",
    "analysis_mode": "top"  # ← Validated on resume
}
checkpoint.save_config(config)

# On resume - this will validate mode matches
checkpoint.validate_config(config)  # Raises ValueError if mismatch
```

**See Also**: [MLCheckpointResume.md](./MLCheckpointResume.md) for high-level checkpoint design

---

## 5. Integration Example

### 5.1 End-to-End Workflow

**Example usage combining all components:**

```python
"""
Example: Complete ML batch processing workflow
"""
import asyncio
from pathlib import Path

from rumiai_v2.api.apify_client import ApifyClient
from rumiai_v2.processors.video_filters import (
    filter_by_date,
    filter_by_engagement_threshold,
    sort_by_engagement
)
from rumiai_v2.processors.checkpoint_manager import CheckpointManager


async def process_hashtag_analysis(
    client_id: str,
    hashtag: str,
    video_count: int = 300,
    date_filter: str = "last_90_days",
    analysis_mode: str = "top",
    force_restart: bool = False
):
    """
    Complete hashtag analysis workflow.

    Steps:
    1. Initialize checkpoint
    2. Scrape videos from Apify
    3. Apply date filtering
    4. Apply engagement filtering
    5. Bucket by duration
    6. Process videos with RumiAI (with checkpoint/resume)
    7. Train ML models
    8. Generate reports
    """

    # 1. Initialize checkpoint manager
    checkpoint = CheckpointManager(client_id, "hashtag", hashtag)

    if force_restart:
        checkpoint.clear_checkpoint()

    # Check for existing checkpoint
    resume_position, last_bucket = checkpoint.get_resume_point()

    if resume_position > 0:
        print(f"✓ Resuming from position {resume_position}")
        print(f"✓ Loading {resume_position} completed videos from checkpoint")

        # Validate config matches
        new_config = {
            "video_count": video_count,
            "date_filter": date_filter,
            "analysis_mode": analysis_mode
        }
        checkpoint.validate_config(new_config)
    else:
        print("Starting fresh batch processing")

        # Save initial config
        config = {
            "video_count": video_count,
            "date_filter": date_filter,
            "analysis_mode": analysis_mode
        }
        checkpoint.save_config(config)

    # 2. Scrape videos from Apify (skip if resuming)
    if resume_position == 0:
        print(f"\n[1/7] Scraping {video_count} videos from {hashtag} (mode: {analysis_mode})")

        apify = ApifyClient(api_token="YOUR_APIFY_TOKEN")
        videos = await apify.scrape_hashtag(
            hashtag=hashtag,
            video_count=800,  # Scrape more for filtering headroom
            analysis_mode=analysis_mode
        )

        print(f"✓ Scraped {len(videos)} videos from Apify")

        # 3. Apply date filtering
        print(f"\n[2/7] Applying date filter: {date_filter}")
        videos = filter_by_date(videos, date_filter)
        print(f"✓ {len(videos)} videos after date filtering")

        # 4. Apply engagement filtering
        print(f"\n[3/7] Applying engagement thresholds")
        videos = filter_by_engagement_threshold(
            videos,
            min_views=1000,
            min_engagement_rate=0.02
        )
        print(f"✓ {len(videos)} videos after engagement filtering")

        # 5. Sort by engagement (if top mode)
        if analysis_mode == "top":
            print(f"\n[4/7] Sorting by engagement score")
            videos = sort_by_engagement(videos)

        # TODO: Bucket by duration (8 buckets)
        # TODO: Select top 40 + bottom 20 per bucket

        print(f"\n✓ Ready to process {len(videos)} videos")

    # 6. Process videos with RumiAI (with checkpoint/resume)
    # TODO: Implement sequential processing with checkpoint.save_progress()

    # 7. Train ML models
    # TODO: Load features with checkpoint.load_completed_features()

    # 8. Generate reports
    # TODO: Create creative strategy reports

    print("\n✅ Hashtag analysis complete!")


# Run example
if __name__ == "__main__":
    asyncio.run(process_hashtag_analysis(
        client_id="client_acme_corp",
        hashtag="#nutrition",
        video_count=300,
        date_filter="last_90_days",
        analysis_mode="top",
        force_restart=False
    ))
```

---

## 6. Testing Scripts

### 6.1 Apify Scraper Validation Test

**File**: `/home/jorge/rumiaifinal/tests/test_apify_hashtag.py` (NEW)

```python
"""
Test script for validating Apify hashtag scraper integration
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rumiai_v2.api.apify_client import ApifyClient
from rumiai_v2.processors.video_filters import (
    filter_by_date,
    calculate_engagement_score,
    sort_by_engagement
)


async def test_hashtag_scraping():
    """Test hashtag scraping with both modes."""

    print("=" * 80)
    print("Testing Apify Hashtag Scraper")
    print("=" * 80)

    # Initialize client
    apify = ApifyClient(api_token="YOUR_APIFY_TOKEN")

    # Test 1: Top mode (engagement sorting)
    print("\n[TEST 1] Scraping #nutrition - TOP MODE (engagement)")
    print("-" * 80)

    videos_top = await apify.scrape_hashtag(
        hashtag="#nutrition",
        video_count=100,
        analysis_mode="top"
    )

    print(f"✓ Scraped {len(videos_top)} videos")
    print(f"\nTop 5 videos by engagement:")

    sorted_top = sort_by_engagement(videos_top[:5])
    for i, video in enumerate(sorted_top, 1):
        score = calculate_engagement_score(video)
        print(f"  {i}. Views: {video.views:,} | Shares: {video.shares:,} | Score: {score:.0f}")

    # Test 2: Recent mode (date sorting)
    print("\n[TEST 2] Scraping #nutrition - RECENT MODE (date)")
    print("-" * 80)

    videos_recent = await apify.scrape_hashtag(
        hashtag="#nutrition",
        video_count=100,
        analysis_mode="recent"
    )

    print(f"✓ Scraped {len(videos_recent)} videos")
    print(f"\nMost recent 5 videos:")

    for i, video in enumerate(videos_recent[:5], 1):
        print(f"  {i}. Posted: {video.create_time.date()} | Views: {video.views:,}")

    # Test 3: Date filtering
    print("\n[TEST 3] Date filtering - last_30_days")
    print("-" * 80)

    filtered = filter_by_date(videos_top, "last_30_days")
    print(f"✓ Filtered: {len(filtered)}/{len(videos_top)} videos retained")

    # Test 4: Verify metadata fields
    print("\n[TEST 4] Metadata field validation")
    print("-" * 80)

    sample = videos_top[0]
    required_fields = [
        ('video_id', sample.video_id),
        ('views', sample.views),
        ('likes', sample.likes),
        ('comments', sample.comments),
        ('shares', sample.shares),
        ('duration', sample.duration),
        ('create_time', sample.create_time),
        ('download_url', sample.download_url)
    ]

    all_present = True
    for field_name, field_value in required_fields:
        status = "✓" if field_value else "✗"
        print(f"  {status} {field_name}: {field_value}")
        if not field_value:
            all_present = False

    if all_present:
        print("\n✅ All required metadata fields present!")
    else:
        print("\n⚠️ Some metadata fields missing!")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_hashtag_scraping())
```

---

## 7. Next Steps

### 7.1 Implementation Checklist

**IMMEDIATE** (Required for ML batch MVP):
- [ ] Get hashtag scraper actor ID from Apify dashboard
- [ ] Update `apify_client.py` with dual-scraper support
- [ ] Create `video_filters.py` with date filtering functions
- [ ] Create `checkpoint_manager.py` with config validation
- [ ] Test with real hashtag to validate sorting and metadata

**NEXT** (Before ML training):
- [ ] Implement duration bucketing logic (8 buckets)
- [ ] Implement top 40 + bottom 20 selection per bucket
- [ ] Add sequential video processing with checkpoint integration
- [ ] Measure actual retention rate after date filtering
- [ ] Document minimum sample size thresholds

**FUTURE** (Post-MVP optimization):
- [ ] Add retry logic for failed Apify scrapes
- [ ] Implement rate limiting for large batches
- [ ] Add progress bars for long-running operations
- [ ] Create unified batch processing orchestrator

---

## 8. Related Documentation

- **High-Level Design**: [MLAnalysisMode.md](./MLAnalysisMode.md)
- **Video Selection Strategy**: [VideoSelection.md](./VideoSelection.md) (lines 19-45)
- **Checkpoint/Resume System**: [MLCheckpointResume.md](./MLCheckpointResume.md)
- **ML Planning**: [MLPlanning.md](../../MLPlanning.md) (section 3.1)
