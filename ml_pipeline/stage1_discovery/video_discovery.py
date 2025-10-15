"""
Stage 1: Video Discovery & Selection - Main Orchestrator

Orchestrates the complete Stage 1 pipeline from scraping to bucket selection.

Source: VideoDiscoveryCHILDTI.md
"""

import logging
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from .apify_scraper import ApifyScraper
from .date_filter import DateFilter
from .winner_analyzer import WinnerAnalyzer
from .video_selector import VideoSelector
from .confirmation import InteractiveConfirmation
from .cluster_config import detect_target_type, load_cluster_config
from .cluster_scraper import run_cluster_scraping
from .cluster_deduplication import deduplicate_with_provenance
from .cluster_analytics import save_cluster_analytics
from .constants import (
    EXIT_CODE_SUCCESS,
    EXIT_CODE_USER_ABORT,
)

# Import foundation modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from foundation.paths import PathBuilder
from foundation.config import ConfigManager


logger = logging.getLogger(__name__)


class VideoDiscovery:
    """
    Main orchestrator for Stage 1: Video Discovery & Selection.

    Coordinates all Stage 1 sub-processes:
    1. Apify scraping
    2. Date filtering
    3. Winner analysis
    4. Video selection per bucket
    5. Interactive confirmation
    6. Output file creation
    """

    def __init__(
        self,
        config: Dict,
        apify_api_key: str,
        path_builder: Optional[PathBuilder] = None
    ):
        """
        Initialize Video Discovery orchestrator.

        Args:
            config: Configuration dict from Foundation Stage 0
            apify_api_key: Apify API key from environment
            path_builder: Optional PathBuilder instance (created if not provided)
        """
        self.config = config
        self.apify_api_key = apify_api_key

        # Initialize path builder
        if path_builder is None:
            self.path_builder = PathBuilder()
        else:
            self.path_builder = path_builder

        # Initialize sub-components
        self.scraper = ApifyScraper(apify_api_key)
        self.date_filter = DateFilter()
        self.winner_analyzer = WinnerAnalyzer()
        self.video_selector = VideoSelector()
        self.confirmation = InteractiveConfirmation()

    def run(self) -> int:
        """
        Execute complete Stage 1 pipeline.

        Returns:
            Exit code (0 = success, >0 = error)
        """
        try:
            logger.info("="*80)
            logger.info("STAGE 1: VIDEO DISCOVERY & SELECTION")
            logger.info("="*80)
            logger.info(f"Client: {self.config['client_id']}")
            logger.info(f"Target: {self.config['target']} ({self.config['analysis_type']})")
            logger.info(f"Mode: {self.config['analysis_mode']}")
            logger.info(f"Strategy: {self.config['selection_strategy']}")
            logger.info(f"Video count: {self.config['video_count']} per bucket")
            logger.info(f"Date filter: {self.config['date_filter']}")
            logger.info("")

            # Stage 1.1: Detect cluster mode and scrape accordingly
            logger.info("Stage 1.1: Detecting target type...")
            target_type, cluster_config = detect_target_type(
                self.config['target'],
                self.config['analysis_type']
            )

            if target_type == "cluster":
                logger.info(f"✓ Cluster mode detected: {cluster_config['cluster_id']}")
                logger.info(f"  Primary: {cluster_config['primary_hashtag']}")
                logger.info(f"  Variants: {len(cluster_config['variant_hashtags'])} hashtags")
                logger.info("")

                # Stage 1.1a: Cluster Scraping
                logger.info("Stage 1.1a: Running cluster scraping...")
                all_videos, failed_scrapes = run_cluster_scraping(
                    cluster_config=cluster_config,
                    analysis_mode=self.config['analysis_mode'],
                    country_code=self.config.get('country_code', 'US'),
                    date_filter=self.config['date_filter'],
                    apify_scraper=self.scraper
                )
                logger.info(f"✓ Cluster scraping complete: {len(all_videos)} videos")
                logger.info("")

                # Stage 1.1b: Deduplication with Provenance
                logger.info("Stage 1.1b: Deduplicating with provenance tracking...")
                videos, analytics = deduplicate_with_provenance(
                    all_videos=all_videos,
                    cluster_config=cluster_config,
                    failed_scrapes=failed_scrapes
                )
                logger.info(f"✓ Deduplicated to {len(videos)} unique videos")
                logger.info("")

                # Stage 1.1c: Save Cluster Analytics
                logger.info("Stage 1.1c: Saving cluster analytics...")
                analytics_path = save_cluster_analytics(
                    analytics=analytics,
                    client_id=self.config['client_id'],
                    cluster_id=cluster_config['cluster_id']
                )
                logger.info(f"✓ Cluster analytics saved: {analytics_path}")
                logger.info("")

            else:
                logger.info(f"✓ Single target mode: {self.config['target']}")
                logger.info("")

                # Stage 1.1: Apify Scraping (Single Mode)
                logger.info("Stage 1.1: Scraping videos from Apify...")
                videos = self.scraper.scrape_videos(
                    analysis_type=self.config['analysis_type'],
                    target=self.config['target'],
                    analysis_mode=self.config['analysis_mode'],
                    date_filter=self.config['date_filter'],
                    country_code=self.config.get('country_code', 'US')  # Default to US if not provided
                )
                logger.info(f"✓ Scraped {len(videos)} unique videos")
                logger.info("")

            # Stage 1.2: Date Filtering
            logger.info("Stage 1.2: Filtering by publication date...")
            filtered_videos = self.date_filter.filter_by_date(
                videos=videos,
                date_filter=self.config['date_filter']
            )
            logger.info(f"✓ Filtered to {len(filtered_videos)} videos ({self.config['date_filter']})")
            logger.info("")

            # Stage 1.3: Winner Analysis
            logger.info("Stage 1.3: Analyzing winner distribution...")
            winning_buckets, winner_distribution, qualified_buckets = \
                self.winner_analyzer.analyze_winner_distribution(filtered_videos)
            logger.info(f"✓ Identified {len(winning_buckets)} winning bucket(s): {winning_buckets}")
            logger.info("")

            # Stage 1.4: Video Selection Per Bucket
            logger.info("Stage 1.4: Selecting videos per bucket...")
            selected_per_bucket = self.video_selector.select_videos_per_bucket(
                videos=filtered_videos,
                winning_buckets=winning_buckets,
                strategy=self.config['selection_strategy'],
                video_count=self.config['video_count']
            )
            total_selected = sum(s['selected_count'] for s in selected_per_bucket.values())
            logger.info(f"✓ Selected {total_selected} videos across {len(selected_per_bucket)} bucket(s)")
            logger.info("")

            # Stage 1.5: Interactive Confirmation
            logger.info("Stage 1.5: Awaiting user confirmation...")
            confirmed = self.confirmation.confirm_bucket_selection(
                selected_per_bucket=selected_per_bucket,
                auto_confirm=self.config.get('auto_confirm', False)
            )

            if not confirmed:
                logger.info("User aborted. Exiting without creating output files.")
                return EXIT_CODE_USER_ABORT

            # Stage 1.6: Create Output Files
            logger.info("Stage 1.6: Creating output files...")
            self._create_output_files(
                selected_per_bucket=selected_per_bucket,
                winner_distribution=winner_distribution,
                winning_buckets=winning_buckets
            )
            logger.info("✓ Output files created successfully")
            logger.info("")

            logger.info("="*80)
            logger.info("STAGE 1 COMPLETE")
            logger.info("="*80)

            return EXIT_CODE_SUCCESS

        except KeyboardInterrupt:
            logger.info("User aborted with Ctrl+C")
            return EXIT_CODE_USER_ABORT

        except Exception as e:
            logger.error(f"Stage 1 failed: {e}", exc_info=True)
            raise

    def _create_output_files(
        self,
        selected_per_bucket: Dict[str, Dict],
        winner_distribution: Dict[str, int],
        winning_buckets: list
    ):
        """
        Create output files for Stage 1.

        Creates:
        1. selected_videos.json per bucket
        2. winner_analysis.json
        3. Bucket directory structure
        """
        # Get analysis base directory using PathBuilder
        analysis_base = self.path_builder.get_target_dir(
            client_id=self.config['client_id'],
            analysis_type=self.config['analysis_type'],
            target=self.config['target'],
            analysis_mode=self.config['analysis_mode'],
            selection_strategy=self.config['selection_strategy']
        )

        # Create winner_analysis.json
        winner_analysis_path = os.path.join(str(analysis_base), "winner_analysis.json")
        winner_analysis = {
            "top_100_distribution": winner_distribution,
            "top_3_buckets": winning_buckets,
            "winner_coverage": sum(
                winner_distribution.get(b, 0) for b in winning_buckets
            ) / sum(winner_distribution.values()) * 100,
            "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_date": datetime.now(timezone.utc).isoformat()
        }

        os.makedirs(os.path.dirname(winner_analysis_path), exist_ok=True)
        with open(winner_analysis_path, 'w') as f:
            json.dump(winner_analysis, f, indent=2)

        logger.info(f"Created winner_analysis.json: {winner_analysis_path}")

        # Create selected_videos.json per bucket
        for bucket, selection in selected_per_bucket.items():
            bucket_base = self.path_builder.get_bucket_dir(analysis_base, bucket)

            # Create bucket directory structure
            bucket_base_str = str(bucket_base)
            os.makedirs(os.path.join(bucket_base_str, "videos"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "analysis", "insights"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "analysis", "unified"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "analysis", "service_debug"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "validation"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "flagged_videos"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "ml_analysis"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "models"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "llm_reports", "analysis"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "llm_reports", "formatted"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "reports"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(bucket_base_str, "logs"), exist_ok=True)

            # Write selected_videos.json
            selected_videos_path = os.path.join(bucket_base_str, "selected_videos.json")
            with open(selected_videos_path, 'w') as f:
                json.dump(selection, f, indent=2)

            logger.info(
                f"Created bucket {bucket}: {selection['selected_count']} videos → {selected_videos_path}"
            )
