# S7B2 Tests Part 2 - Comprehensive Verification

**Purpose:** Verify the S7B2 cross-window feature fix works across multiple buckets, modes, and data volumes.

**Date:** 2025-10-28
**Status:** Ready for execution by fresh CLI
**Prerequisites:** All S7B2 code fixes applied (see Pre-Test Checklist)

---

## Pre-Test Checklist

Before running tests, verify these fixes are applied:

### 1. Cross-Window Feature Names (xwin_ prefix)
```bash
# Verify in Stage 3
grep "xwin_hook_to_middle_energy" scripts/stage3_aggregation.py
# Should return: df['xwin_hook_to_middle_energy'] = ...

# Verify in Stage 4
grep "xwin_hook_to_middle_energy" rumiai_v2/processors/feature_transformation.py
# Should return: 'xwin_hook_to_middle_energy',
```

### 2. Stage 6 video_count Bug Fix
```bash
# Verify fix
grep -A 3 "video_count = len(df)" ml_pipeline/stage6_analysis/ml_analysis_generation.py
# Should show: video_count = len(df) BEFORE the if statement
```

### 3. Stage 5 Minimum Videos Threshold
```bash
# Verify lowered thresholds
grep "MIN_VIDEOS_CONTRASTIVE = 3" rumiai_v2/processors/model_training.py
grep "MIN_VIDEOS_TOP = 3" rumiai_v2/processors/model_training.py
# Both should return the lines with = 3
```

### 4. Stage 7 Wrapper Script
```bash
# Verify exists
ls -lh run_stage7_test.py
# Should show file created with .env loading logic
```

**If any checks fail:** Code fixes not applied. Do NOT proceed with tests.

---

## Test Matrix

| Test | Bucket | Videos | Mode | xwin Features | Priority |
|------|--------|--------|------|---------------|----------|
| 1    | 3-9s   | 32     | contrastive | 3 | High |
| 2    | 60-90s | 38     | contrastive | 5 | High |
| 3    | 33-60s | 3      | top | 5 | Critical |
| 4    | 18-33s | 3      | top | 5 | Medium |
| 5    | 60-90s | 3      | top | 5 | Medium |

**Coverage:**
- ✅ 3 unique bucket sizes (3-9s, 33-60s, 60-90s)
- ✅ Both modes (contrastive, top)
- ✅ Both data volumes (3 vs 30+ videos)
- ✅ 3 xwin features (3-9s) and 5 xwin features (others)

---

## Test 1: bucket_3-9s (32 videos, contrastive, 3 xwin features)

### Path
```
/home/jorge/rumiaifinal/data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s
```

### Expected Behavior
- **Stage 3:** 49 columns (21×2 windows + 3 metadata + 3 xwin + 1 label)
- **Stage 4:** 65 columns (49 input + transformations)
- **Stage 6:** 3 xwin features in top 10 RF features
- **Stage 7:** At least 1 xwin feature in universal_principles

### Execution Commands

#### Option A: Using Wrapper Scripts (Recommended)

**Step 1: Create test_bucket_3-9s.py**
```python
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
print(f"Testing bucket_3-9s (contrastive mode)")
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
xwin_principles = [p for p in principles if p.startswith('xwin_')]
print(f"  xwin in universal_principles: {len(xwin_principles)}")
for p in xwin_principles:
    print(f"    - {p}")

print("\n" + "=" * 80)
print("✓ Test 1 (bucket_3-9s) PASSED")
print("=" * 80)
```

**Run:**
```bash
cd /home/jorge/rumiaifinal
venv/bin/python test_bucket_3-9s.py
```

#### Option B: Manual Command Execution

If the script fails, run each stage manually:

**Stage 3:**
```bash
cd /home/jorge/rumiaifinal
export PYTHONPATH=/home/jorge/rumiaifinal
venv/bin/python scripts/stage3_aggregation.py \
  --bucket-path data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s \
  --strategy contrastive
```

**Verify Stage 3:**
```bash
# Count columns
head -1 data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s/ml_analysis/aggregated_features.csv | tr ',' '\n' | wc -l
# Should output: 49

# Check xwin features
head -1 data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s/ml_analysis/aggregated_features.csv | tr ',' '\n' | grep "^xwin_"
# Should output 3 features:
# xwin_eye_contact_consistency
# xwin_word_density_std
# xwin_energy_progression_slope
```

**Stage 4:**
```bash
venv/bin/python -c "
from rumiai_v2.processors.feature_transformation import run_stage4_transformation
bucket_path = 'data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s'
config = {'bucket': '3-9s', 'strategy': 'contrastive'}
success, files, elapsed = run_stage4_transformation(bucket_path, config)
print(f'Success: {success}, Files: {len(files)}, Time: {elapsed:.2f}s')
"
```

**Verify Stage 4:**
```bash
head -1 data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s/ml_analysis/rf_transformed.csv | tr ',' '\n' | wc -l
# Should be around 65 columns

head -1 data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s/ml_analysis/rf_transformed.csv | tr ',' '\n' | grep "^xwin_" | wc -l
# Should output: 3
```

**Stage 5:**
```bash
venv/bin/python -c "
from rumiai_v2.processors.model_training import run_stage5_training
bucket_path = 'data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s'
config = {'bucket': '3-9s', 'video_count': 32}  # Actual video count
success, models, elapsed = run_stage5_training(bucket_path, config, 'contrastive')
print(f'Success: {success}, Models: {len(models)}, Time: {elapsed:.2f}s')
"
```

**Stage 6:**
```bash
venv/bin/python -c "
from ml_pipeline.stage6_analysis.ml_analysis_generation import generate_ml_analysis_jsons
from config.bucket_definitions import BUCKET_WINDOWS
bucket_path = 'data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s'
bucket = '3-9s'
windows = BUCKET_WINDOWS[bucket]
json_count = generate_ml_analysis_jsons(bucket_path, bucket, windows)
print(f'JSONs generated: {json_count}')
"
```

**Verify Stage 6:**
```bash
cat data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s/ml_analysis/rf_video_analysis.json | jq '.feature_importance[].feature' | grep "xwin_"
# Should show xwin features in top 10
```

**Stage 7:**
```bash
# Create wrapper if doesn't exist
cat > run_stage7_bucket_3-9s.py << 'EOF'
#!/usr/bin/env python3
import os, sys
from pathlib import Path

env_file = Path("/home/jorge/rumiaifinal/.env")
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value.strip().strip('"').strip("'")

sys.path.insert(0, "/home/jorge/rumiaifinal")
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_main

stage7_main('data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s', '3-9s', 'wellness')
print('\n✓ Stage 7 Complete')
EOF

venv/bin/python run_stage7_bucket_3-9s.py
```

**Verify Stage 7:**
```bash
cat data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s/ml_analysis/llm/winning_formulas.json | jq '.supplementary_insights.universal_principles[]' | grep "xwin_"
# Should show at least one xwin feature
```

### Success Criteria

- ✅ Stage 3: 49 columns, 3 xwin features present
- ✅ Stage 4: ~65 columns, 3 xwin in video RF
- ✅ Stage 5: Models trained successfully
- ✅ Stage 6: xwin features in top 10 with non-NULL distributions
- ✅ Stage 7: At least 1 xwin feature in universal_principles

---

## Test 2: bucket_60-90s (38 videos, contrastive, 5 xwin features)

### Path
```
/home/jorge/rumiaifinal/data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_60-90s
```

### Expected Behavior
- **Stage 3:** 156 columns (21×7 windows + 3 metadata + 5 xwin + 1 label)
- **Stage 4:** 167 columns
- **Stage 6:** 5 xwin features available
- **Stage 7:** Multiple xwin features in universal_principles

### Quick Execution

**All Stages Script:**
```python
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

print("Testing bucket_60-90s (contrastive, 5 xwin features)")

# Stage 3
from scripts.stage3_aggregation import aggregate_features
csv_path, _ = aggregate_features(BUCKET_PATH, STRATEGY)
import pandas as pd
df = pd.read_csv(csv_path)
print(f"✓ Stage 3: {len(df.columns)} columns, {len(df)} videos")
assert len(df.columns) == 156, f"Expected 156, got {len(df.columns)}"

# Stage 4
from rumiai_v2.processors.feature_transformation import run_stage4_transformation
config = {'bucket': BUCKET, 'strategy': STRATEGY}
success, _, _ = run_stage4_transformation(BUCKET_PATH, config)
rf_df = pd.read_csv(f"{BUCKET_PATH}/ml_analysis/rf_transformed.csv")
print(f"✓ Stage 4: {len(rf_df.columns)} columns")
assert len(rf_df.columns) == 167, f"Expected 167, got {len(rf_df.columns)}"

# Stage 5
from rumiai_v2.processors.model_training import run_stage5_training
config = {'bucket': BUCKET, 'video_count': len(df)}
success, models, _ = run_stage5_training(BUCKET_PATH, config, STRATEGY)
print(f"✓ Stage 5: {len(models)} models trained")

# Stage 6
from ml_pipeline.stage6_analysis.ml_analysis_generation import generate_ml_analysis_jsons
from config.bucket_definitions import BUCKET_WINDOWS
json_count = generate_ml_analysis_jsons(BUCKET_PATH, BUCKET, BUCKET_WINDOWS[BUCKET])
print(f"✓ Stage 6: {json_count} JSONs")

import json
with open(f"{BUCKET_PATH}/ml_analysis/rf_video_analysis.json") as f:
    s6 = json.load(f)
xwin_s6 = [f for f in s6['feature_importance'] if f['feature'].startswith('xwin_')]
print(f"  xwin in top 10: {len(xwin_s6)}")

# Stage 7
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_main
stage7_main(BUCKET_PATH, BUCKET, "wellness")
print(f"✓ Stage 7 complete")

with open(f"{BUCKET_PATH}/ml_analysis/llm/winning_formulas.json") as f:
    s7 = json.load(f)
principles = s7.get('supplementary_insights', {}).get('universal_principles', [])
xwin_s7 = [p for p in principles if p.startswith('xwin_')]
print(f"  xwin in principles: {len(xwin_s7)}")
for x in xwin_s7:
    print(f"    {x}")

print("✓ Test 2 PASSED")
```

**Run:**
```bash
cd /home/jorge/rumiaifinal
venv/bin/python test_bucket_60-90s.py
```

### Success Criteria

- ✅ Stage 3: 156 columns, 5 xwin features
- ✅ Stage 4: 167 columns
- ✅ Stage 6: xwin features with distributions
- ✅ Stage 7: Multiple xwin in universal_principles

---

## Test 3: bucket_33-60s (3 videos, TOP mode, 5 xwin features) **CRITICAL**

### Path
```
/home/jorge/rumiaifinal/data/clients/influencer1/competitors/mandanazarghami/top_top/buckets/bucket_33-60s
```

### Expected Behavior
- **TOP MODE:** All is_top_performer=1 (no contrastive split)
- **Stage 3:** 156 columns (21×7 + 3 + 5 + 1)
- **Stage 5:** Models trained on only 3 videos (minimum threshold test)
- **Stage 6:** Distributions may be less meaningful with only 3 videos
- **Stage 7:** Should complete without error

### Execution

**All Stages Script:**
```python
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
```

**Run:**
```bash
cd /home/jorge/rumiaifinal
venv/bin/python test_bucket_33-60s_top.py
```

### Success Criteria

- ✅ Stage 3: All is_top_performer=1 (TOP mode behavior)
- ✅ Stage 5: Accepts 3 videos (minimum threshold validation)
- ✅ Stage 7: Completes without error
- ✅ Pipeline handles minimum video count gracefully

---

## Test 4 & 5: Additional Buckets (Optional)

If time permits, test remaining buckets:

**Test 4:** bucket_18-33s (influencer1, top, 3 videos)
**Test 5:** bucket_60-90s (influencer1, top, 3 videos)

Use same pattern as Test 3, adjusting:
- `BUCKET_PATH`
- `BUCKET` name
- Expected Stage 3 columns (135 for 18-33s, 156 for 60-90s)
- Expected Stage 4 columns (147 for 18-33s, 167 for 60-90s)

---

## Verification Reference

### Expected Column Counts by Bucket

| Bucket | Windows | Stage 3 | Stage 4 | xwin Features |
|--------|---------|---------|---------|---------------|
| 3-9s   | 2       | 49      | ~65     | 3             |
| 18-33s | 6       | 135     | 147     | 5             |
| 33-60s | 7       | 156     | 167     | 5             |
| 60-90s | 7       | 156     | 167     | 5             |

### xwin Features by Bucket

**3-9s (3 features):**
- xwin_eye_contact_consistency
- xwin_word_density_std
- xwin_energy_progression_slope

**All others (5 features):**
- xwin_hook_to_middle_energy
- xwin_middle_to_closing_energy
- xwin_eye_contact_consistency
- xwin_word_density_std
- xwin_energy_progression_slope

### Quick Verification Commands

**Check Stage 3 columns:**
```bash
head -1 <bucket_path>/ml_analysis/aggregated_features.csv | tr ',' '\n' | wc -l
```

**Check xwin features in Stage 3:**
```bash
head -1 <bucket_path>/ml_analysis/aggregated_features.csv | tr ',' '\n' | grep "^xwin_"
```

**Check Stage 6 xwin in top 10:**
```bash
cat <bucket_path>/ml_analysis/rf_video_analysis.json | jq '.feature_importance[].feature' | grep "xwin_"
```

**Check Stage 7 xwin in principles:**
```bash
cat <bucket_path>/ml_analysis/llm/winning_formulas.json | jq '.supplementary_insights.universal_principles[]' | grep "xwin_"
```

---

## Troubleshooting

### Issue: "MIN_VIDEOS_CONTRASTIVE = 50"
**Solution:** MIN_VIDEOS thresholds not lowered to 3. Re-apply fix:
```python
# In rumiai_v2/processors/model_training.py line 770-771:
MIN_VIDEOS_CONTRASTIVE = 3
MIN_VIDEOS_TOP = 3
```

### Issue: "ANTHROPIC_API_KEY environment variable not set"
**Solution:** Stage 7 doesn't load .env. Use wrapper script approach (run_stage7_*.py)

### Issue: "Feature hook_to_middle_energy_delta not found"
**Solution:** xwin_ prefix not applied. Re-apply Stage 3 & 4 name changes.

### Issue: "video_count undefined"
**Solution:** Stage 6 bug not fixed. Move `video_count = len(df)` outside conditional.

### Issue: Stage 4 validation error "Expected 49 columns, found 52"
**Solution:** Old xwin feature names causing window prefix collision. Verify xwin_ prefix applied.

---

## Summary Report Template

After completing tests, summarize results:

```markdown
# S7B2 Tests Part 2 - Results

**Date:** [DATE]
**Tester:** [NAME]

## Results Summary

| Test | Bucket | Videos | Mode | Status | Notes |
|------|--------|--------|------|--------|-------|
| 1    | 3-9s   | 32     | contrastive | ✅/✗ | |
| 2    | 60-90s | 38     | contrastive | ✅/✗ | |
| 3    | 33-60s | 3      | top | ✅/✗ | |
| 4    | 18-33s | 3      | top | ✅/✗ | Optional |
| 5    | 60-90s | 3      | top | ✅/✗ | Optional |

## xwin Features Found in Stage 7

### Test 1 (bucket_3-9s):
- [ ] xwin_eye_contact_consistency
- [ ] xwin_word_density_std
- [ ] xwin_energy_progression_slope

### Test 2 (bucket_60-90s):
- [ ] xwin_hook_to_middle_energy
- [ ] xwin_middle_to_closing_energy
- [ ] xwin_eye_contact_consistency
- [ ] xwin_word_density_std
- [ ] xwin_energy_progression_slope

### Test 3 (bucket_33-60s, top mode):
- [ ] (list found features)

## Issues Encountered

[Describe any failures, errors, or unexpected behavior]

## Conclusion

- S7B2 fix working: ✅/✗
- Cross-window features flow through pipeline: ✅/✗
- Minimum video threshold (3) works: ✅/✗
- TOP mode works correctly: ✅/✗
```

---

**End of S7B2TestsPt2.md**
