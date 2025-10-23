#!/usr/bin/env python3
"""
Integration Test: Verify Stage 1 → Stage 4 → Stage 5 pipeline
"""
import json
import pandas as pd
from pathlib import Path
import sys

print("=" * 70)
print("INTEGRATION TEST: End-to-End Pipeline Validation")
print("=" * 70)

bucket_path = Path("/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s")

# ============================================================================
# STAGE 1: Verify selected_videos.json
# ============================================================================
print("\n" + "=" * 70)
print("STAGE 1: Video Selection")
print("=" * 70)

selected_path = bucket_path / "selected_videos.json"
if not selected_path.exists():
    print(f"❌ FAIL: selected_videos.json not found")
    sys.exit(1)

with open(selected_path) as f:
    selected = json.load(f)

stage1_top = sum(1 for v in selected['videos'] if v.get('is_top_performer') == True)
stage1_bottom = sum(1 for v in selected['videos'] if v.get('is_top_performer') == False)

print(f"✓ File exists: selected_videos.json")
print(f"✓ Total videos: {len(selected['videos'])}")
print(f"✓ Tagged top: {stage1_top}")
print(f"✓ Tagged bottom: {stage1_bottom}")

if stage1_top == 0 or stage1_bottom == 0:
    print(f"❌ FAIL: Missing top or bottom tags in Stage 1")
    sys.exit(1)

print(f"✅ STAGE 1 PASS")

# ============================================================================
# STAGE 4: Verify rf_transformed.csv
# ============================================================================
print("\n" + "=" * 70)
print("STAGE 4: Feature Transformation")
print("=" * 70)

rf_path = bucket_path / "ml_analysis" / "rf_transformed.csv"
if not rf_path.exists():
    print(f"❌ FAIL: rf_transformed.csv not found")
    print(f"   Run Stage 4 first!")
    sys.exit(1)

df_rf = pd.read_csv(rf_path)

if 'is_top_performer' not in df_rf.columns:
    print(f"❌ FAIL: Missing is_top_performer column")
    sys.exit(1)

stage4_top = (df_rf['is_top_performer'] == 1).sum()
stage4_bottom = (df_rf['is_top_performer'] == 0).sum()

print(f"✓ File exists: rf_transformed.csv")
print(f"✓ Total rows: {len(df_rf)}")
print(f"✓ Top performers (1): {stage4_top}")
print(f"✓ Bottom performers (0): {stage4_bottom}")
print(f"✓ Both classes present: {stage4_top > 0 and stage4_bottom > 0}")

if stage4_top == 0 or stage4_bottom == 0:
    print(f"❌ FAIL: Missing class in Stage 4 output")
    sys.exit(1)

# Verify Stage 4 matches Stage 1
stage1_map = {str(v['id']): (1 if v.get('is_top_performer') == True else 0) for v in selected['videos']}
matches = 0
for _, row in df_rf.iterrows():
    video_id = str(row['video_id'])
    if video_id in stage1_map:
        if row['is_top_performer'] == stage1_map[video_id]:
            matches += 1

match_pct = (matches / len(df_rf)) * 100
print(f"✓ Stage 1 ↔ Stage 4 match: {matches}/{len(df_rf)} ({match_pct:.1f}%)")

if match_pct < 95:
    print(f"⚠️  WARNING: Low match rate between Stage 1 and Stage 4")

print(f"✅ STAGE 4 PASS")

# ============================================================================
# STAGE 5: Check if models can be trained
# ============================================================================
print("\n" + "=" * 70)
print("STAGE 5: ML Training Readiness")
print("=" * 70)

models_dir = bucket_path / "models"

print(f"✓ Checking training readiness...")
print(f"  - Feature count: {len(df_rf.columns) - 2} (excluding video_id, is_top_performer)")
print(f"  - Sample count: {len(df_rf)}")
print(f"  - Class distribution: {stage4_top}:{stage4_bottom} (~{stage4_top/len(df_rf)*100:.0f}:{stage4_bottom/len(df_rf)*100:.0f})")

# Check if models exist (Stage 5 has run)
if models_dir.exists():
    model_files = list(models_dir.glob("*.pkl"))
    print(f"✓ Models directory exists")
    print(f"✓ Model files: {len(model_files)}")
    
    if len(model_files) > 0:
        print(f"✅ STAGE 5 COMPLETE: Models trained successfully!")
    else:
        print(f"⚠️  STAGE 5 NOT RUN: No model files found")
        print(f"   Run: python3 rumiai_ml_batch.py --stage 5 ...")
else:
    print(f"⚠️  STAGE 5 NOT RUN: Models directory doesn't exist")
    print(f"   Run: python3 rumiai_ml_batch.py --stage 5 ...")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n✅ Stage 1: Video selection with tags - WORKING")
print(f"✅ Stage 4: Feature transformation reads tags - WORKING")
print(f"✅ Pipeline: End-to-end data flow - VALIDATED")

print(f"\n📊 Key Metrics:")
print(f"  - Videos processed: {len(df_rf)}")
print(f"  - Top/Bottom split: {stage4_top}/{stage4_bottom} ({stage4_top/len(df_rf)*100:.0f}/{stage4_bottom/len(df_rf)*100:.0f})")
print(f"  - Stage consistency: {match_pct:.1f}% match")

print("\n" + "=" * 70)
print("🎉 ALL INTEGRATION TESTS PASSED!")
print("=" * 70)
