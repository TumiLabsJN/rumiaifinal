#!/usr/bin/env python3
# test_bucket_60-90s.py
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

BUCKET_PATH = "data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_60-90s"
BUCKET = "60-90s"
STRATEGY = "contrastive"

print("=" * 80)
print("Testing bucket_60-90s (contrastive, 38 videos, 5 xwin features)")
print("=" * 80)

# Stage 3
print("\n--- Stage 3 ---")
from scripts.stage3_aggregation import aggregate_features
csv_path, _ = aggregate_features(BUCKET_PATH, STRATEGY)
import pandas as pd
df = pd.read_csv(csv_path)
print(f"✓ Columns: {len(df.columns)} (expected 156)")
print(f"✓ Videos: {len(df)}")
xwin_cols = [c for c in df.columns if c.startswith('xwin_')]
print(f"✓ xwin features: {xwin_cols}")
assert len(df.columns) == 156, f"Expected 156, got {len(df.columns)}"
assert len(xwin_cols) == 5, f"Expected 5 xwin, got {len(xwin_cols)}"

# Stage 4
print("\n--- Stage 4 ---")
from rumiai_v2.processors.feature_transformation import run_stage4_transformation
config = {'bucket': BUCKET, 'strategy': STRATEGY}
success, _, _ = run_stage4_transformation(BUCKET_PATH, config)
rf_df = pd.read_csv(f"{BUCKET_PATH}/ml_analysis/rf_transformed.csv")
print(f"✓ RF columns: {len(rf_df.columns)} (expected 168)")
xwin_in_rf = [c for c in rf_df.columns if c.startswith('xwin_')]
print(f"✓ xwin in RF: {xwin_in_rf}")
assert len(rf_df.columns) == 168, f"Expected 168 (includes gender_nan), got {len(rf_df.columns)}"
assert len(xwin_in_rf) == 5, f"Expected 5 xwin in RF, got {len(xwin_in_rf)}"

# Stage 5
print("\n--- Stage 5 ---")
from rumiai_v2.processors.model_training import run_stage5_training
config = {'bucket': BUCKET, 'video_count': len(df)}
success, models, _ = run_stage5_training(BUCKET_PATH, config, STRATEGY)
print(f"✓ Models trained: {len(models)}")

# Stage 6
print("\n--- Stage 6 ---")
from ml_pipeline.stage6_analysis.ml_analysis_generation import generate_ml_analysis_jsons
from config.bucket_definitions import BUCKET_WINDOWS
json_count = generate_ml_analysis_jsons(BUCKET_PATH, BUCKET, BUCKET_WINDOWS[BUCKET])
print(f"✓ JSONs generated: {json_count}")

import json
with open(f"{BUCKET_PATH}/ml_analysis/rf_video_analysis.json") as f:
    s6 = json.load(f)
xwin_s6 = [f for f in s6['feature_importance'] if f['feature'].startswith('xwin_')]
print(f"  xwin in top 10 RF features: {len(xwin_s6)}")
for x in xwin_s6:
    print(f"    ✓ {x['feature']}: importance={x['importance']:.4f}")

# Stage 7
print("\n--- Stage 7 ---")
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_main
stage7_main(BUCKET_PATH, BUCKET, "wellness")
print(f"✓ Stage 7 complete")

with open(f"{BUCKET_PATH}/ml_analysis/llm/winning_formulas.json") as f:
    s7 = json.load(f)
principles = s7.get('supplementary_insights', {}).get('universal_principles', [])
xwin_s7 = [p for p in principles if 'xwin_' in p]
print(f"  xwin in universal_principles: {len(xwin_s7)}")
for x in xwin_s7:
    print(f"    - {x}")

print("\n" + "=" * 80)
print("✓ Test 2 (bucket_60-90s) PASSED")
print("=" * 80)
