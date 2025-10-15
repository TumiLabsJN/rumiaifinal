"""
Cluster Analytics Generation Module

Generates comprehensive cluster health analytics reports.

Source: HashtagVolumeV2_TI.md Section 3.5, 3.7
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict
from pathlib import Path

from .constants import CLUSTER_ANALYTICS_PATH_TEMPLATE

logger = logging.getLogger(__name__)


def generate_cluster_analytics(
    all_videos: List[Dict],
    unique_videos: List[Dict],
    cluster_config: Dict,
    failed_scrapes: List[Dict]
) -> Dict:
    """
    Generate cluster health analytics report.

    Source: HashtagVolumeV2_TI.md Section 3.5

    Args:
        all_videos: list[dict], all scraped videos (before deduplication)
        unique_videos: list[dict], deduplicated videos with provenance
        cluster_config: dict, cluster configuration (for calculating total attempts)
                       Schema: ClusterConfigSchema (Section 2.2)
        failed_scrapes: list[dict], failed scrape details from run_cluster_scraping()
                       Schema: [{"hashtag": str, "run": int, "error": str}, ...]

    Returns:
        dict: Cluster analytics report
              Schema: ClusterAnalyticsSchema (Section 2.2)

    Raises:
        None
    """
    logger.info("Generating cluster analytics...")

    # Extract cluster parameters
    all_hashtags = [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']
    runs_per_hashtag = cluster_config['scrape_config']['runs_per_hashtag']
    cluster_id = cluster_config['cluster_id']

    # Calculate scrape attempts from CONFIG (not derived from video data)
    total_scrapes_attempted = len(all_hashtags) * runs_per_hashtag
    total_scrapes_succeeded = total_scrapes_attempted - len(failed_scrapes)

    # ===== SCRAPE SUMMARY =====
    scrape_summary = {
        "total_scrapes_attempted": total_scrapes_attempted,
        "total_scrapes_succeeded": total_scrapes_succeeded,
        "total_scraped_videos": len(all_videos),
        "total_unique_videos": len(unique_videos),
        "overall_duplication_rate": (len(all_videos) - len(unique_videos)) / len(all_videos) * 100 if all_videos else 0,
        "failed_scrapes": failed_scrapes
    }

    logger.info(f"  Scrape summary: {total_scrapes_succeeded}/{total_scrapes_attempted} successful")
    logger.info(f"  Total videos: {len(all_videos)} scraped → {len(unique_videos)} unique")
    logger.info(f"  Duplication rate: {scrape_summary['overall_duplication_rate']:.1f}%")

    # ===== PER-HASHTAG CONTRIBUTION =====
    logger.info("  Calculating per-hashtag contribution...")
    per_hashtag_contribution = {}

    for hashtag in all_hashtags:
        # Videos found by this hashtag
        found_by_hashtag = [
            v for v in unique_videos if hashtag in v['source_hashtags']
        ]

        # Videos exclusive to this hashtag (not found by any other)
        exclusive_to_hashtag = [
            v for v in unique_videos
            if hashtag in v['source_hashtags'] and len(v['source_hashtags']) == 1
        ]

        # Calculate overlap (found by this hashtag AND at least one other)
        overlap_videos = len(found_by_hashtag) - len(exclusive_to_hashtag)

        per_hashtag_contribution[hashtag] = {
            "total_found": len(found_by_hashtag),
            "unique_videos": len(found_by_hashtag),  # Same as total_found (all unique)
            "overlap_videos": overlap_videos,
            "exclusive_videos": len(exclusive_to_hashtag),
            "contribution_percentage": len(found_by_hashtag) / len(unique_videos) * 100 if unique_videos else 0
        }

        logger.debug(f"    {hashtag}: {len(found_by_hashtag)} videos ({per_hashtag_contribution[hashtag]['contribution_percentage']:.1f}%)")

    # ===== PAIRWISE OVERLAPS =====
    logger.info("  Calculating pairwise overlaps...")
    pairwise_overlaps = {}

    for i, hashtag1 in enumerate(all_hashtags):
        for hashtag2 in all_hashtags[i+1:]:  # Only pairs (no self-comparison)
            # Videos found by BOTH hashtags
            overlap = [
                v for v in unique_videos
                if hashtag1 in v['source_hashtags'] and hashtag2 in v['source_hashtags']
            ]

            # Smaller set size (for percentage calculation)
            set1_size = len([v for v in unique_videos if hashtag1 in v['source_hashtags']])
            set2_size = len([v for v in unique_videos if hashtag2 in v['source_hashtags']])
            smaller_size = min(set1_size, set2_size)

            # Overlap percentage
            overlap_pct = len(overlap) / smaller_size * 100 if smaller_size > 0 else 0

            # Key format: alphabetical order, underscores instead of hashes
            key = f"{hashtag1.replace('#', '')}_{hashtag2.replace('#', '')}"
            pairwise_overlaps[key] = round(overlap_pct, 1)

    # ===== RUN EFFECTIVENESS =====
    logger.info("  Calculating run effectiveness...")
    run_effectiveness = {}

    for hashtag in all_hashtags:
        # Get all runs for this hashtag
        runs = sorted(list(set([
            run for v in unique_videos
            if hashtag in v['source_hashtags']
            for run in v['source_runs']
        ])))

        if len(runs) >= 2:
            # Videos from run 1
            run_1_videos = [
                v for v in unique_videos
                if hashtag in v['source_hashtags'] and 1 in v['source_runs']
            ]

            # Videos from run 2
            run_2_videos = [
                v for v in unique_videos
                if hashtag in v['source_hashtags'] and 2 in v['source_runs']
            ]

            # Videos NEW in run 2 (not in run 1)
            run_2_new = [
                v for v in run_2_videos
                if 1 not in v['source_runs']  # Only found in run 2
            ]

            run_effectiveness[hashtag] = {
                "run_1_videos": len(run_1_videos),
                "run_2_videos": len(run_2_videos),
                "run_2_new_videos": len(run_2_new),
                "run_2_new_percentage": len(run_2_new) / len(run_2_videos) * 100 if run_2_videos else 0
            }

            logger.debug(f"    {hashtag}: run 2 added {len(run_2_new)} new videos ({run_effectiveness[hashtag]['run_2_new_percentage']:.1f}%)")

    # ===== BUCKET DISTRIBUTION BY SOURCE =====
    # (Optional - only if bucket assignment happens in Stage 1)
    # For now, leave empty as buckets are assigned later in the pipeline
    bucket_distribution_by_source = {}

    # ===== ASSEMBLE ANALYTICS =====
    analytics = {
        "cluster_id": cluster_id,
        "execution_date": datetime.now(timezone.utc).isoformat(),
        "scrape_summary": scrape_summary,
        "per_hashtag_contribution": per_hashtag_contribution,
        "pairwise_overlaps": pairwise_overlaps,
        "run_effectiveness": run_effectiveness,
        "bucket_distribution_by_source": bucket_distribution_by_source
    }

    logger.info("✅ Cluster analytics generation complete")

    return analytics


def save_cluster_analytics(analytics: Dict, client_id: str, cluster_id: str) -> str:
    """
    Save cluster analytics to JSON file.

    Source: HashtagVolumeV2_TI.md Section 3.7

    Args:
        analytics: dict, cluster analytics report
                  Schema: ClusterAnalyticsSchema (Section 2.2)
        client_id: str, client identifier
        cluster_id: str, cluster identifier

    Returns:
        str: Path to saved analytics file

    Raises:
        OSError: if file write fails
    """
    # Get project root (parent of rumiaifinal directory)
    project_root = Path(__file__).parent.parent.parent

    # Build analytics file path
    # Template: "/data/clients/{client_id}/hashtag/{cluster_id}/cluster_analytics.json"
    analytics_path = project_root / "data" / "clients" / client_id / "hashtag" / cluster_id / "cluster_analytics.json"

    logger.info(f"Saving cluster analytics to: {analytics_path}")

    # Create directory if doesn't exist
    analytics_path.parent.mkdir(parents=True, exist_ok=True)

    # Write analytics
    with open(analytics_path, 'w') as f:
        json.dump(analytics, f, indent=2)

    logger.info(f"✅ Saved cluster analytics: {analytics_path}")

    return str(analytics_path)
