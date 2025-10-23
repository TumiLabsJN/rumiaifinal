#!/usr/bin/env python3
"""
Test: Verify Stage 4 reads is_top_performer tags from selected_videos.json
"""
import pandas as pd
import json
from pathlib import Path

print("=" * 60)
print("TEST 2: Verify Stage 4 Label Reading")
print("=" * 60)

# Test with bucket 18-33s
bucket_path = Path("/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s")

# Load selected_videos.json (Stage 1 output)
selected_videos_path = bucket_path / "selected_videos.json"
with open(selected_videos_path) as f:
    selected = json.load(f)

print(f"\n📊 Stage 1 Output (selected_videos.json):")
print(f"  - Total videos: {len(selected['videos'])}")
print(f"  - Expected top: {selected['top_count']}")
print(f"  - Expected bottom: {selected['bottom_count']}")

# Create expected mapping from Stage 1
stage1_map = {str(v['id']): v.get('is_top_performer') for v in selected['videos']}
stage1_top_count = sum(1 for v in stage1_map.values() if v == True)
stage1_bottom_count = sum(1 for v in stage1_map.values() if v == False)

print(f"  - Actual top (tagged True): {stage1_top_count}")
print(f"  - Actual bottom (tagged False): {stage1_bottom_count}")

# Load rf_transformed.csv (Stage 4 output)
rf_csv_path = bucket_path / "ml_analysis" / "rf_transformed.csv"

if not rf_csv_path.exists():
    print(f"\n❌ FAIL: rf_transformed.csv not found!")
    print(f"   Run Stage 4 first: python3 rumiai_ml_batch.py --stage 4 ...")
    exit(1)

df_rf = pd.read_csv(rf_csv_path)

print(f"\n📊 Stage 4 Output (rf_transformed.csv):")
print(f"  - Total rows: {len(df_rf)}")
print(f"  - Has is_top_performer column: {'is_top_performer' in df_rf.columns}")

if 'is_top_performer' not in df_rf.columns:
    print(f"\n❌ FAIL: rf_transformed.csv missing is_top_performer column!")
    exit(1)

stage4_top_count = (df_rf['is_top_performer'] == 1).sum()
stage4_bottom_count = (df_rf['is_top_performer'] == 0).sum()

print(f"  - Top performers (1): {stage4_top_count}")
print(f"  - Bottom performers (0): {stage4_bottom_count}")

# Test 1: Validate distribution
print(f"\n🔍 Test 1: Validate 80/20 Distribution")
total = len(df_rf)
top_pct = (stage4_top_count / total) * 100
bottom_pct = (stage4_bottom_count / total) * 100

print(f"  - Top %: {top_pct:.1f}%")
print(f"  - Bottom %: {bottom_pct:.1f}%")

if 75 <= top_pct <= 85 and 15 <= bottom_pct <= 25:
    print(f"✅ PASS: Distribution is approximately 80/20")
else:
    print(f"❌ FAIL: Distribution is NOT 80/20!")
    exit(1)

# Test 2: Verify Stage 4 matches Stage 1
print(f"\n🔍 Test 2: Verify Stage 4 Labels Match Stage 1")

# Check each video in CSV matches Stage 1 tag
mismatches = []
for idx, row in df_rf.iterrows():
    video_id = str(row['video_id'])
    stage4_label = row['is_top_performer']
    stage1_label = stage1_map.get(video_id)
    
    if stage1_label is not None:
        expected = 1 if stage1_label == True else 0
        if stage4_label != expected:
            mismatches.append((video_id, expected, stage4_label))

if len(mismatches) == 0:
    print(f"✅ PASS: All {len(df_rf)} videos match Stage 1 tags!")
else:
    print(f"❌ FAIL: {len(mismatches)} videos have mismatched labels!")
    print(f"\n   First 5 mismatches:")
    for video_id, expected, actual in mismatches[:5]:
        print(f"   - Video {video_id}: Expected {expected}, Got {actual}")
    exit(1)

# Test 3: Verify both classes present (critical for RF training)
print(f"\n🔍 Test 3: Verify Both Classes Present")
unique_labels = df_rf['is_top_performer'].unique()
print(f"  - Unique labels: {sorted(unique_labels)}")

if len(unique_labels) >= 2 and 0 in unique_labels and 1 in unique_labels:
    print(f"✅ PASS: Both classes (0 and 1) present - RF can train!")
else:
    print(f"❌ FAIL: Only {len(unique_labels)} class(es) present - RF cannot train!")
    exit(1)

print("\n" + "=" * 60)
print("✅ TEST 2 COMPLETE - ALL CHECKS PASSED!")
print("=" * 60)
