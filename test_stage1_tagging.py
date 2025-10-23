#!/usr/bin/env python3
"""
Test: Verify Stage 1 adds is_top_performer tags to selected videos
"""
import json
from pathlib import Path

# Test with existing data
bucket_path = Path("/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s")
selected_videos_path = bucket_path / "selected_videos.json"

print("=" * 60)
print("TEST 1: Verify Stage 1 Video Tagging")
print("=" * 60)

if not selected_videos_path.exists():
    print(f"❌ FAIL: selected_videos.json not found at {selected_videos_path}")
    exit(1)

with open(selected_videos_path) as f:
    data = json.load(f)

print(f"\n📊 File Structure:")
print(f"  - Bucket: {data['bucket']}")
print(f"  - Strategy: {data['strategy']}")
print(f"  - Total videos: {len(data['videos'])}")
print(f"  - Metadata top_count: {data.get('top_count')}")
print(f"  - Metadata bottom_count: {data.get('bottom_count')}")

print(f"\n🔍 Checking video tags...")

# Check if videos have is_top_performer tag
videos_with_tag = [v for v in data['videos'] if 'is_top_performer' in v]
print(f"  - Videos with is_top_performer tag: {len(videos_with_tag)}/{len(data['videos'])}")

if len(videos_with_tag) == 0:
    print(f"\n❌ FAIL: NO videos have is_top_performer tag!")
    print(f"   This means Stage 1 fix was not applied.")
    print(f"\n   First video keys: {list(data['videos'][0].keys())[:15]}")
    exit(1)

# Count top vs bottom
top_count = sum(1 for v in data['videos'] if v.get('is_top_performer') == True)
bottom_count = sum(1 for v in data['videos'] if v.get('is_top_performer') == False)

print(f"\n📈 Performer Distribution:")
print(f"  - Top performers (True): {top_count}")
print(f"  - Bottom performers (False): {bottom_count}")
print(f"  - Total: {top_count + bottom_count}")

# Validate 80/20 split
expected_top = data.get('top_count', 0)
expected_bottom = data.get('bottom_count', 0)

print(f"\n✓ Expected Split:")
print(f"  - Expected top: {expected_top}")
print(f"  - Expected bottom: {expected_bottom}")

if top_count == expected_top and bottom_count == expected_bottom:
    print(f"\n✅ PASS: Tags match expected 80/20 split!")
else:
    print(f"\n⚠️  WARNING: Tag count mismatch!")
    print(f"   Expected: {expected_top} top / {expected_bottom} bottom")
    print(f"   Actual: {top_count} top / {bottom_count} bottom")

# Verify order (first N should be top, last M should be bottom)
print(f"\n🔍 Verifying tag order...")
first_n_are_top = all(v.get('is_top_performer') == True for v in data['videos'][:expected_top])
last_m_are_bottom = all(v.get('is_top_performer') == False for v in data['videos'][-expected_bottom:])

if first_n_are_top and last_m_are_bottom:
    print(f"✅ PASS: Tag order is correct (first {expected_top} are top, last {expected_bottom} are bottom)")
else:
    print(f"❌ FAIL: Tag order is incorrect!")
    if not first_n_are_top:
        print(f"   - First {expected_top} videos are NOT all tagged as top performers")
    if not last_m_are_bottom:
        print(f"   - Last {expected_bottom} videos are NOT all tagged as bottom performers")

print("\n" + "=" * 60)
print("TEST 1 COMPLETE")
print("=" * 60)
