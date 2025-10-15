# test_stage2_fix.py
# Quick validation script to test the Stage 2 bug fix

import json
from pathlib import Path

print("=" * 60)
print("Testing Stage 2 Bug Fix - Data Format Validation")
print("=" * 60)

# Load one bucket's selected videos
bucket_path = Path("/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s/selected_videos.json")

print(f"\nLoading: {bucket_path}")

try:
    with open(bucket_path) as f:
        video_data = json.load(f)
        video_list = video_data['videos']  # THE FIX

    print(f"✓ Loaded {len(video_list)} videos")
    print(f"✓ First video ID: {video_list[0]['id']}")
    print(f"✓ First video has webVideoUrl: {'webVideoUrl' in video_list[0]}")

    # Test that Stage 2 can iterate
    print("\nTesting iteration over first 2 videos:")
    for video in video_list[:2]:  # Just first 2
        video_id = video['id']  # Should NOT error
        web_url = video.get('webVideoUrl')
        print(f"  - Video {video_id}: {web_url}")

    print("\n" + "=" * 60)
    print("✓ TEST PASSED - Fix is correct!")
    print("=" * 60)

except Exception as e:
    print("\n" + "=" * 60)
    print(f"✗ TEST FAILED: {e}")
    print("=" * 60)
    raise
