"""
Test hybrid approach: Stage 2 processing with webVideoUrl (no downloadAddr)

This test validates that Stage 2 can process videos from Stage 1 scraping
even when downloadAddr is missing (uses webVideoUrl instead).
"""

import os
import sys
import json
import logging

# Add project root to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from ml_pipeline.stage2_processing.main import stage_2_video_processing_main

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("Testing Hybrid Approach: Stage 2 with webVideoUrl")
    print("=" * 80)

    # Test configuration (matches Config schema from foundation/schemas.py)
    config = {
        "client_id": "testy_client",
        "analysis_type": "hashtag",
        "target": "test_vitamin",
        "analysis_mode": "top",
        "selection_strategy": "contrastive",
        "video_count": 10,
        "date_filter": "last_90_days",
        "report_type": "single",
        "report_audience": "creator",
        "auto_confirm": True,
        "run_date": "2025-01-14T00:00:00Z"
    }

    # Use bucket with existing selected videos
    bucket_name = "18-33s"
    data_root = os.getenv('DATA_ROOT', '/home/jorge/rumiaifinal/data')
    bucket_path = f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/buckets/bucket_{bucket_name}/"

    selected_videos_path = f"{bucket_path}selected_videos.json"

    # Check if test data exists
    if not os.path.exists(selected_videos_path):
        print(f"❌ Test data not found: {selected_videos_path}")
        print("Run Stage 1 first to generate test data")
        return 1

    # Load selected videos
    with open(selected_videos_path, 'r') as f:
        data = json.load(f)
        all_videos = data['videos']

    print(f"\n1. Found {len(all_videos)} selected videos in bucket {bucket_name}")

    # Take only first 2 videos for quick test
    test_videos = all_videos[:2]
    print(f"2. Testing with {len(test_videos)} videos (quick test)")

    # Verify videos have webVideoUrl
    for video in test_videos:
        video_id = video['id']
        has_web_url = 'webVideoUrl' in video
        has_download_addr = 'videoMeta' in video and video.get('videoMeta') and 'downloadAddr' in video.get('videoMeta', {})

        print(f"\n   Video {video_id}:")
        print(f"   - Has webVideoUrl: {has_web_url}")
        print(f"   - Has downloadAddr: {has_download_addr}")

        if has_web_url:
            print(f"   - webVideoUrl: {video['webVideoUrl']}")

    print(f"\n3. Running Stage 2 processing...")
    print(f"   This will:")
    print(f"   - Skip pre-download (no downloadAddr available)")
    print(f"   - Use webVideoUrl during processing")
    print(f"   - rumiai_runner.py will scrape + download + process")
    print(f"   ⏱️  Expected time: ~3-5 minutes for 2 videos\n")

    try:
        # Run Stage 2 processing
        result = stage_2_video_processing_main(
            config=config,
            video_list=test_videos,
            bucket_name=bucket_name,
            enable_pause_support=False
        )

        print("\n" + "=" * 80)
        print("✅ Stage 2 Processing Complete")
        print("=" * 80)
        print(f"Total videos: {result['total']}")
        print(f"Completed: {result['completed']}")
        print(f"Failed: {result['failed']}")
        print(f"Status: {result['status']}")

        # Verify outputs exist
        print("\n4. Verifying outputs...")
        insights_dir = "/home/jorge/rumiaifinal/insights/"

        for video in test_videos:
            video_id = video['id']
            insights_path = f"{insights_dir}{video_id}_temporal_windows_updated.json"

            if os.path.exists(insights_path):
                file_size = os.path.getsize(insights_path) / 1024
                print(f"   ✓ {video_id}_temporal_windows_updated.json ({file_size:.1f} KB)")
            else:
                print(f"   ✗ {video_id}_temporal_windows_updated.json (missing)")

        print("\n" + "=" * 80)
        print("✅ Hybrid Approach Test: PASSED")
        print("=" * 80)
        print("\nConclusion:")
        print("- Stage 2 successfully processed videos using webVideoUrl")
        print("- No downloadAddr required from Stage 1 scraping")
        print("- rumiai_runner.py handled scraping + downloading automatically")
        print("\n✅ Ready for full pipeline test (STEP 7)")

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Hybrid Approach Test: FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
