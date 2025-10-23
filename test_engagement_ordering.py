#!/usr/bin/env python3
"""
Test: Verify top performers have higher engagement than bottom performers
"""
import json
from pathlib import Path
import statistics

print("=" * 60)
print("TEST 3: Verify Engagement-Based Ordering")
print("=" * 60)

bucket_path = Path("/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s")
selected_videos_path = bucket_path / "selected_videos.json"

with open(selected_videos_path) as f:
    data = json.load(f)

print(f"\n📊 Analyzing {len(data['videos'])} videos")

# Separate top and bottom performers
top_performers = [v for v in data['videos'] if v.get('is_top_performer') == True]
bottom_performers = [v for v in data['videos'] if v.get('is_top_performer') == False]

print(f"  - Top performers: {len(top_performers)}")
print(f"  - Bottom performers: {len(bottom_performers)}")

if len(top_performers) == 0 or len(bottom_performers) == 0:
    print(f"\n❌ FAIL: Missing top or bottom performers!")
    exit(1)

# Extract engagement metrics (views)
top_views = [v.get('playCount', 0) for v in top_performers]
bottom_views = [v.get('playCount', 0) for v in bottom_performers]

print(f"\n📈 Engagement Analysis (Views):")
print(f"\n  Top Performers:")
print(f"    - Min: {min(top_views):,}")
print(f"    - Max: {max(top_views):,}")
print(f"    - Mean: {statistics.mean(top_views):,.0f}")
print(f"    - Median: {statistics.median(top_views):,.0f}")

print(f"\n  Bottom Performers:")
print(f"    - Min: {min(bottom_views):,}")
print(f"    - Max: {max(bottom_views):,}")
print(f"    - Mean: {statistics.mean(bottom_views):,.0f}")
print(f"    - Median: {statistics.median(bottom_views):,.0f}")

# Test 1: Top performers should have higher MINIMUM than bottom MAXIMUM
# (or at least higher median)
top_min = min(top_views)
bottom_max = max(bottom_views)
top_median = statistics.median(top_views)
bottom_median = statistics.median(bottom_views)

print(f"\n🔍 Test: Engagement Separation")

if top_min >= bottom_max:
    print(f"✅ PERFECT: Top min ({top_min:,}) >= Bottom max ({bottom_max:,})")
    print(f"   Perfect separation - no overlap!")
elif top_median > bottom_median:
    print(f"✅ PASS: Top median ({top_median:,}) > Bottom median ({bottom_median:,})")
    print(f"   Clear separation despite some overlap")
    
    # Calculate overlap
    overlap_count = sum(1 for v in top_views if v < bottom_max)
    print(f"   - Overlap: {overlap_count}/{len(top_views)} top videos have views < bottom max")
else:
    print(f"❌ FAIL: Top performers do NOT have higher engagement!")
    print(f"   Top median: {top_median:,}")
    print(f"   Bottom median: {bottom_median:,}")
    exit(1)

# Test 2: Verify list is sorted descending
print(f"\n🔍 Test: Video List Sorted by Engagement")
all_views = [v.get('playCount', 0) for v in data['videos']]
is_sorted_desc = all(all_views[i] >= all_views[i+1] for i in range(len(all_views)-1))

if is_sorted_desc:
    print(f"✅ PASS: Videos are sorted by views descending")
    print(f"   First video: {all_views[0]:,} views")
    print(f"   Last video: {all_views[-1]:,} views")
else:
    print(f"⚠️  WARNING: Videos NOT fully sorted by views")
    print(f"   This may indicate Stage 1 sorting issue")

print("\n" + "=" * 60)
print("✅ TEST 3 COMPLETE")
print("=" * 60)
