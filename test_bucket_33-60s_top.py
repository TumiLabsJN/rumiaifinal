#!/usr/bin/env python3
# test_bucket_33-60s_top.py
import os, sys
from pathlib import Path

# Load .env
env_file = Path("/home/jorge/rumiaifinal/.env")
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value.strip().strip('"').strip("'")

sys.path.insert(0, "/home/jorge/rumiaifinal")

BUCKET_PATH = "data/clients/influencer1/competitors/mandanazarghami/top_top/buckets/bucket_33-60s"
BUCKET = "33-60s"
STRATEGY = "top"  # ← IMPORTANT: TOP mode

print("=" * 80)
print("Testing bucket_33-60s (TOP mode, 3 videos, 5 xwin features)")
print("This tests the minimum video threshold (3) and TOP mode behavior")
print("=" * 80)

# Stage 3
print("\n--- Stage 3 ---")
from scripts.stage3_aggregation import aggregate_features
csv_path, _ = aggregate_features(BUCKET_PATH, STRATEGY)
import pandas as pd
df = pd.read_csv(csv_path)
print(f"✓ Columns: {len(df.columns)} (expected 156)")
print(f"✓ Videos: {len(df)}")
print(f"✓ is_top_performer unique values: {df['is_top_performer'].unique()} (should be [1] for TOP mode)")
assert len(df.columns) == 156
assert all(df['is_top_performer'] == 1), "TOP mode should have all is_top_performer=1"

# Stage 4
print("\n--- Stage 4 ---")
from rumiai_v2.processors.feature_transformation import run_stage4_transformation
config = {'bucket': BUCKET, 'strategy': STRATEGY}
success, files, elapsed = run_stage4_transformation(BUCKET_PATH, config)
print(f"✓ Transformed {len(files)} files in {elapsed:.2f}s")

# Stage 5
print("\n--- Stage 5 (3 videos minimum test) ---")
from rumiai_v2.processors.model_training import run_stage5_training
config = {'bucket': BUCKET, 'video_count': len(df)}
try:
    success, models, elapsed = run_stage5_training(BUCKET_PATH, config, STRATEGY)
    print(f"✓ Models trained: {len(models)} in {elapsed:.2f}s")
    print(f"  NOTE: Models trained on only 3 videos (minimum threshold)")
except Exception as e:
    print(f"✗ Stage 5 FAILED: {e}")
    print("  This indicates MIN_VIDEOS threshold is still too high")
    sys.exit(1)

# Stage 6
print("\n--- Stage 6 ---")
from ml_pipeline.stage6_analysis.ml_analysis_generation import generate_ml_analysis_jsons
from config.bucket_definitions import BUCKET_WINDOWS
json_count = generate_ml_analysis_jsons(BUCKET_PATH, BUCKET, BUCKET_WINDOWS[BUCKET])
print(f"✓ JSONs generated: {json_count}")

# Stage 7
print("\n--- Stage 7 ---")
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_main
stage7_main(BUCKET_PATH, BUCKET, "mandanazarghami")
print(f"✓ Stage 7 complete")

import json
with open(f"{BUCKET_PATH}/ml_analysis/llm/winning_formulas.json") as f:
    s7 = json.load(f)

principles = s7.get('supplementary_insights', {}).get('universal_principles', [])
xwin_principles = [p for p in principles if p.startswith('xwin_')]
print(f"\nResults:")
print(f"  Total principles: {len(principles)}")
print(f"  xwin principles: {len(xwin_principles)}")
for x in xwin_principles:
    print(f"    {x}")

print("\n" + "=" * 80)
print("✓ Test 3 (bucket_33-60s, TOP mode, 3 videos) PASSED")
print("=" * 80)
