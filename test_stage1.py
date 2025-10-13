#!/usr/bin/env python3
"""
Test Stage 1: Video Discovery & Selection

Tests Stage 1 implementation with mock data.
"""

import sys
import os
import json
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline.stage1_discovery.date_filter import DateFilter
from ml_pipeline.stage1_discovery.winner_analyzer import WinnerAnalyzer
from ml_pipeline.stage1_discovery.video_selector import VideoSelector
from ml_pipeline.stage1_discovery.confirmation import InteractiveConfirmation


def create_mock_videos(count: int, base_timestamp: int) -> list:
    """Create mock video metadata objects."""
    videos = []
    for i in range(count):
        # Create videos with decreasing engagement (sorted DESC)
        playCount = 100000 - (i * 1000)

        # Distribute across different durations
        if i % 5 == 0:
            duration = 20  # 18-33s bucket
        elif i % 5 == 1:
            duration = 40  # 33-60s bucket
        elif i % 5 == 2:
            duration = 15  # 13-18s bucket
        elif i % 5 == 3:
            duration = 25  # 18-33s bucket
        else:
            duration = 50  # 33-60s bucket

        videos.append({
            "id": f"video_{i:04d}",
            "createTime": base_timestamp - (i * 86400),  # Spread across days
            "duration": duration,
            "playCount": playCount,
            "shareCount": playCount // 100,
            "commentCount": playCount // 200,
            "likeCount": playCount // 50,
            "webVideoUrl": f"https://tiktok.com/@user/video/{i:04d}",
            "videoMeta": {
                "downloadAddr": f"https://download.url/{i:04d}",
                "width": 1080,
                "height": 1920
            },
            "authorMeta": {
                "id": f"author_{i % 10}",
                "name": f"@creator_{i % 10}"
            }
        })

    return videos


def test_date_filter():
    """Test Stage 1.2: Date Filtering."""
    print("\n" + "="*80)
    print("TEST: Stage 1.2 - Date Filter")
    print("="*80)

    # Create mock videos spanning 100 days
    now = int(datetime.now(timezone.utc).timestamp())
    videos = create_mock_videos(count=150, base_timestamp=now)

    print(f"Created {len(videos)} mock videos")

    # Test filtering
    date_filter = DateFilter()
    filtered = date_filter.filter_by_date(videos, "last_90_days")

    print(f"After filtering (last_90_days): {len(filtered)} videos")
    print("✓ Date filter test passed")


def test_winner_analyzer():
    """Test Stage 1.3: Winner Analysis."""
    print("\n" + "="*80)
    print("TEST: Stage 1.3 - Winner Analyzer")
    print("="*80)

    # Create mock videos (already sorted by engagement DESC)
    now = int(datetime.now(timezone.utc).timestamp())
    videos = create_mock_videos(count=150, base_timestamp=now)

    print(f"Created {len(videos)} mock videos")

    # Test winner analysis
    analyzer = WinnerAnalyzer()
    winning_buckets, winner_dist, qualified = analyzer.analyze_winner_distribution(videos)

    print(f"Winning buckets: {winning_buckets}")
    print(f"Winner distribution: {winner_dist}")
    print("✓ Winner analyzer test passed")

    return videos, winning_buckets


def test_video_selector():
    """Test Stage 1.4: Video Selection."""
    print("\n" + "="*80)
    print("TEST: Stage 1.4 - Video Selector")
    print("="*80)

    # Get videos and winning buckets from previous test
    now = int(datetime.now(timezone.utc).timestamp())
    videos = create_mock_videos(count=150, base_timestamp=now)

    analyzer = WinnerAnalyzer()
    winning_buckets, _, _ = analyzer.analyze_winner_distribution(videos)

    # Test contrastive strategy
    selector = VideoSelector()
    selected = selector.select_videos_per_bucket(
        videos=videos,
        winning_buckets=winning_buckets,
        strategy="contrastive",
        video_count=50
    )

    print("\nContrastive Strategy Results:")
    for bucket, selection in selected.items():
        print(f"  {bucket}: {selection['selected_count']} videos")
        print(f"    Top: {selection['top_count']}, Bottom: {selection['bottom_count']}")

    # Test top strategy
    selected_top = selector.select_videos_per_bucket(
        videos=videos,
        winning_buckets=winning_buckets,
        strategy="top",
        video_count=40
    )

    print("\nTop Strategy Results:")
    for bucket, selection in selected_top.items():
        print(f"  {bucket}: {selection['selected_count']} videos")
        print(f"    Top: {selection['top_count']}, Bottom: {selection['bottom_count']}")

    print("✓ Video selector test passed")

    return selected


def test_confirmation():
    """Test Stage 1.5: Interactive Confirmation."""
    print("\n" + "="*80)
    print("TEST: Stage 1.5 - Interactive Confirmation")
    print("="*80)

    # Get selected videos from previous test
    now = int(datetime.now(timezone.utc).timestamp())
    videos = create_mock_videos(count=150, base_timestamp=now)

    analyzer = WinnerAnalyzer()
    winning_buckets, _, _ = analyzer.analyze_winner_distribution(videos)

    selector = VideoSelector()
    selected = selector.select_videos_per_bucket(
        videos=videos,
        winning_buckets=winning_buckets,
        strategy="contrastive",
        video_count=50
    )

    # Test auto-confirm mode
    confirmation = InteractiveConfirmation()
    result = confirmation.confirm_bucket_selection(
        selected_per_bucket=selected,
        auto_confirm=True  # Skip interactive prompt for testing
    )

    print(f"Confirmation result (auto-confirm): {result}")
    print("✓ Confirmation test passed")


def main():
    """Run all Stage 1 tests."""
    print("\n" + "="*80)
    print("STAGE 1 IMPLEMENTATION TEST SUITE")
    print("="*80)

    try:
        test_date_filter()
        test_winner_analyzer()
        test_video_selector()
        test_confirmation()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print()

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
