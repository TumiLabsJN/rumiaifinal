#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Load .env
env_file = Path("/home/jorge/rumiaifinal/.env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

sys.path.insert(0, "/home/jorge/rumiaifinal")

# Test parameters
BUCKET_PATH = "data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s"
BUCKET = "3-9s"
STRATEGY = "contrastive"

print("=" * 80)
print(f"Testing bucket_3-9s (contrastive mode, 32 videos, 3 xwin features)")
print("=" * 80)

# Stage 3
print("\n--- Stage 3: Feature Aggregation ---")
from scripts.stage3_aggregation import aggregate_features
csv_path, summary_path = aggregate_features(BUCKET_PATH, STRATEGY)
print(f"✓ Stage 3 complete: {csv_path}")

# Verify Stage 3 output
import pandas as pd
df = pd.read_csv(csv_path)
print(f"  Columns: {len(df.columns)} (expected 49)")
print(f"  Rows: {len(df)}")
xwin_cols = [c for c in df.columns if c.startswith('xwin_')]
print(f"  xwin features: {xwin_cols}")
assert len(df.columns) == 49, f"Expected 49 columns, got {len(df.columns)}"
assert len(xwin_cols) == 3, f"Expected 3 xwin features, got {len(xwin_cols)}"

# Stage 4
print("\n--- Stage 4: Feature Transformation ---")
from rumiai_v2.processors.feature_transformation import run_stage4_transformation
config = {'bucket': BUCKET, 'strategy': STRATEGY}
success, output_files, elapsed = run_stage4_transformation(BUCKET_PATH, config)
print(f"✓ Stage 4 complete: {len(output_files)} files in {elapsed:.2f}s")

# Verify Stage 4 output
rf_df = pd.read_csv(f"{BUCKET_PATH}/ml_analysis/rf_transformed.csv")
print(f"  Video RF columns: {len(rf_df.columns)} (expected ~65)")
xwin_in_rf = [c for c in rf_df.columns if c.startswith('xwin_')]
print(f"  xwin in video RF: {xwin_in_rf}")
assert len(xwin_in_rf) == 3, f"Expected 3 xwin in RF, got {len(xwin_in_rf)}"

# Stage 5
print("\n--- Stage 5: Model Training ---")
from rumiai_v2.processors.model_training import run_stage5_training
config = {'bucket': BUCKET, 'video_count': len(df)}
success, models, elapsed = run_stage5_training(BUCKET_PATH, config, STRATEGY)
print(f"✓ Stage 5 complete: {len(models)} models in {elapsed:.2f}s")

# Stage 6
print("\n--- Stage 6: Analysis Generation ---")
from ml_pipeline.stage6_analysis.ml_analysis_generation import generate_ml_analysis_jsons
from config.bucket_definitions import BUCKET_WINDOWS
windows = BUCKET_WINDOWS[BUCKET]
json_count = generate_ml_analysis_jsons(BUCKET_PATH, BUCKET, windows)
print(f"✓ Stage 6 complete: {json_count} JSONs generated")

# Verify Stage 6 output
import json
with open(f"{BUCKET_PATH}/ml_analysis/rf_video_analysis.json") as f:
    s6_data = json.load(f)
xwin_in_top10 = [f for f in s6_data['feature_importance'] if f['feature'].startswith('xwin_')]
print(f"  xwin in top 10 RF features: {len(xwin_in_top10)}")
for feat in xwin_in_top10:
    dist_status = "✓" if feat.get('distribution') else "✗"
    print(f"    {dist_status} {feat['feature']}: importance={feat['importance']:.4f}")

# Stage 7
print("\n--- Stage 7: LLM Analysis ---")
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_main
stage7_main(BUCKET_PATH, BUCKET, "wellness")
print(f"✓ Stage 7 complete")

# Verify Stage 7 output
with open(f"{BUCKET_PATH}/ml_analysis/llm/winning_formulas.json") as f:
    s7_data = json.load(f)
principles = s7_data.get('supplementary_insights', {}).get('universal_principles', [])
xwin_principles = [p for p in principles if 'xwin_' in p]
print(f"  xwin in universal_principles: {len(xwin_principles)}")
for p in xwin_principles:
    print(f"    - {p}")

print("\n" + "=" * 80)
print("✓ Test 1 (bucket_3-9s) PASSED")
print("=" * 80)
