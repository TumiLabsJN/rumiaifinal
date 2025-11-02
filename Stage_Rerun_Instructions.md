# Stage Re-run Instructions (Stages 3-7)

**Purpose:** Re-run specific stages of the ML pipeline WITHOUT redoing expensive video processing (Stages 1-2) or taxonomy discovery (Stage 2.6).

**Supported Stages:** 3 (Feature Aggregation), 4 (Feature Transformation), 5 (ML Training), 6 (ML Analysis), 7 (LLM Analysis)

**Use Case Examples:**
- Fixed a bug in feature aggregation logic → Re-run from Stage 3
- Updated transformation code → Re-run from Stage 4
- Changed ML model hyperparameters → Re-run from Stage 5
- Updated analysis generation → Re-run from Stage 6
- Modified LLM prompts → Re-run from Stage 7

---

## ⚠️ **CRITICAL: What Gets Preserved**

**These are NEVER deleted (expensive/time-consuming):**
- ✅ Stage 1 checkpoint (`checkpoints/stage_1_checkpoint.json`) - Video scraping/discovery
- ✅ Stage 2 checkpoints (`buckets/bucket_*/checkpoints/stage_2_checkpoint.json`) - Video processing
- ✅ Stage 2.5 organized files (`buckets/bucket_*/analysis/`) - Temporal windows
- ✅ Stage 2.5.1 validation cache (`content_taxonomies/transcript_validation_cache.json`)
- ✅ Stage 2.6 taxonomy (`content_taxonomies/{TARGET}_taxonomy.json`) - Manual curation
- ✅ Stage 2.7 classification (`content_analysis/validated/`) - LLM classifications
- ✅ Raw video files, transcripts, metadata

---

## 📋 **Quick Reference: What Each Stage Does**

| Stage | Name | Inputs | Outputs | Typical Duration |
|-------|------|--------|---------|------------------|
| 3 | Feature Aggregation | temporal_windows JSONs + classifications | aggregated_features.json | ~30s |
| 4 | Feature Transformation | aggregated_features.json | normalized CSVs, correlation matrices | ~1min |
| 5 | ML Model Training | transformed features | trained models (RF, K-Means) | ~2-5min |
| 6 | ML Analysis | trained models | contrastive analysis JSONs | ~1min |
| 7 | LLM Analysis | ML analysis + classifications | complete_analysis JSONs | ~5-10min |

---

## 🚀 **Quick Start (Concise Input Format)**

If you receive a concise rerun request like:

```
Instructions: /home/jorge/rumiaifinal/Stage_Rerun_Instructions.md
Test: /home/jorge/rumiaifinal/data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive
Stage: 3
```

### **Extract Parameters from Test Path:**

**Path Structure**: `/home/jorge/rumiaifinal/data/clients/{CLIENT_ID}/{ANALYSIS_TYPE}s/{TARGET}/{ANALYSIS_MODE}_{SELECTION_STRATEGY}`

**Example Parsing**:
- Path: `/home/jorge/rumiaifinal/data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive`
- **CLIENT_ID**: `rollo_test5`
- **ANALYSIS_TYPE**: `hashtag` (from `hashtag**s**` directory, remove the 's')
- **TARGET**: `wellnesspt2_test5`
- **ANALYSIS_MODE**: `top` (first part of `top_contrastive`)
- **SELECTION_STRATEGY**: `contrastive` (second part of `top_contrastive`)

### **Then:**
1. Use extracted parameters to fill placeholders in commands below
2. Jump to the section for your specified stage (e.g., "Stage 3: Feature Aggregation")
3. Execute the steps with your extracted parameters

---

## 🚀 **How to Use This Document**

### **Step 1: Identify Your Starting Stage**

Determine which stage needs to be re-run based on what code changed.

### **Step 2: Follow Stage-Specific Instructions Below**

Jump to the section for your starting stage (e.g., "Re-run from Stage 4").

### **Step 3: Run Pipeline**

Execute the pipeline command - it will skip earlier stages and re-run from your chosen stage onwards.

---

## 📍 **Stage 3: Feature Aggregation**

### **When to Use:**
- Modified feature aggregation logic
- Changed which features to aggregate
- Fixed bugs in `stage3_aggregation/` code

### **What Will Happen:**
- Stages 1, 2, 2.5, 2.6, 2.7: **SKIP** (checkpoints exist)
- **Stages 3, 4, 5, 6, 7: RE-RUN** (cascade)

### **Step 1: Navigate to Target Directory**

```bash
cd /home/jorge/rumiaifinal/data/clients/{CLIENT_ID}/{ANALYSIS_TYPE}s/{TARGET}/{ANALYSIS_MODE}_{SELECTION_STRATEGY}

# Example:
# cd /home/jorge/rumiaifinal/data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive
```

### **Step 2: Delete Stage 3+ Checkpoints and Outputs**

```bash
# Delete Stage 3 checkpoints (all buckets)
rm -f buckets/bucket_*/checkpoints/stage_3_checkpoint.json

# Delete Stage 3 outputs
rm -rf buckets/bucket_*/analysis/aggregated_features.json

# Delete Stage 4 checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_4_checkpoint.json
rm -rf buckets/bucket_*/analysis/transformed_features/

# Delete Stage 5 checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_5_checkpoint.json
rm -rf buckets/bucket_*/analysis/models/

# Delete Stage 6 checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_6_checkpoint.json
rm -rf buckets/bucket_*/analysis/contrastive_analysis/

 # Delete Stage 7 outputs (no checkpoint file - both Phase 1 and Phase 2)
  rm -rf buckets/bucket_*/ml_analysis/llm/

echo "✓ Cleaned Stages 3-7"
```

### **Step 3: Verify Expensive Data Preserved**

```bash
echo "=== VERIFYING PRESERVED DATA ==="

# Stage 1 checkpoint
ls -la checkpoints/stage_1_checkpoint.json && echo "  ✅ Stage 1 preserved" || echo "  ❌ Stage 1 MISSING"

# Stage 2 checkpoints
ls -la buckets/bucket_60-90s/checkpoints/stage_2_checkpoint.json && echo "  ✅ Stage 2 preserved" || echo "  ❌ Stage 2 MISSING"

# Taxonomy
ls -la content_taxonomies/{TARGET}_taxonomy.json && echo "  ✅ Taxonomy preserved" || echo "  ❌ Taxonomy MISSING"

# Classifications
find content_analysis/validated -name "*_content.json" | wc -l | xargs -I {} echo "  ✅ {} classification files preserved"
```

### **Step 4: Run Pipeline**

```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

python rumiai_ml_batch.py \
  --client {CLIENT_ID} \
  --target {TARGET} \
  --analysis-type {ANALYSIS_TYPE} \
  --analysis-mode {ANALYSIS_MODE} \
  --selection-strategy {SELECTION_STRATEGY}

# Example:
# python rumiai_ml_batch.py \
#   --client rollo_test5 \
#   --target wellnesspt2_test5 \
#   --analysis-type hashtag \
#   --analysis-mode top \
#   --selection-strategy contrastive
```

---

## 📍 **Stage 4: Feature Transformation**

### **When to Use:**
- Modified transformation logic
- Changed normalization methods
- Fixed bugs in `stage4_transformation/` code

### **What Will Happen:**
- Stages 1, 2, 2.5, 2.6, 2.7, 3: **SKIP** (checkpoints exist)
- **Stages 4, 5, 6, 7: RE-RUN** (cascade)

### **Step 1: Navigate to Target Directory**

```bash
cd /home/jorge/rumiaifinal/data/clients/{CLIENT_ID}/{ANALYSIS_TYPE}s/{TARGET}/{ANALYSIS_MODE}_{SELECTION_STRATEGY}
```

### **Step 2: Delete Stage 4+ Checkpoints and Outputs**

```bash
# Delete Stage 4 checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_4_checkpoint.json
rm -rf buckets/bucket_*/analysis/transformed_features/

# Delete Stage 5 checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_5_checkpoint.json
rm -rf buckets/bucket_*/analysis/models/

# Delete Stage 6 checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_6_checkpoint.json
rm -rf buckets/bucket_*/analysis/contrastive_analysis/

# Delete Stage 7 outputs (no checkpoint file)
rm -f buckets/bucket_*/ml_analysis/llm/complete_analysis_*.json

echo "✓ Cleaned Stages 4-7"
```

### **Step 3: Verify Prerequisites Exist**

```bash
echo "=== VERIFYING PREREQUISITES ==="

# Stage 3 aggregated features
find buckets/bucket_*/analysis -name "aggregated_features.json" | wc -l | xargs -I {} echo "  {} aggregated feature files"

# Expect 3 files (one per bucket)
```

### **Step 4: Run Pipeline**

```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

python rumiai_ml_batch.py \
  --client {CLIENT_ID} \
  --target {TARGET} \
  --analysis-type {ANALYSIS_TYPE} \
  --analysis-mode {ANALYSIS_MODE} \
  --selection-strategy {SELECTION_STRATEGY}
```

---

## 📍 **Stage 5: ML Model Training**

### **When to Use:**
- Changed ML model hyperparameters
- Modified Random Forest or K-Means configuration
- Fixed bugs in `stage5_training/` code

### **What Will Happen:**
- Stages 1, 2, 2.5, 2.6, 2.7, 3, 4: **SKIP** (checkpoints exist)
- **Stages 5, 6, 7: RE-RUN** (cascade)

### **Step 1: Navigate to Target Directory**

```bash
cd /home/jorge/rumiaifinal/data/clients/{CLIENT_ID}/{ANALYSIS_TYPE}s/{TARGET}/{ANALYSIS_MODE}_{SELECTION_STRATEGY}
```

### **Step 2: Delete Stage 5+ Checkpoints and Outputs**

```bash
# Delete Stage 5 checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_5_checkpoint.json
rm -rf buckets/bucket_*/analysis/models/

# Delete Stage 6 checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_6_checkpoint.json
rm -rf buckets/bucket_*/analysis/contrastive_analysis/

# Delete Stage 7 outputs (no checkpoint file)
rm -f buckets/bucket_*/ml_analysis/llm/complete_analysis_*.json

echo "✓ Cleaned Stages 5-7"
```

### **Step 3: Verify Prerequisites Exist**

```bash
echo "=== VERIFYING PREREQUISITES ==="

# Stage 4 transformed features
find buckets/bucket_*/analysis/transformed_features -name "*.csv" | wc -l | xargs -I {} echo "  {} transformed feature files"

# Expect ~20-30 CSV files per bucket
```

### **Step 4: Run Pipeline**

```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

python rumiai_ml_batch.py \
  --client {CLIENT_ID} \
  --target {TARGET} \
  --analysis-type {ANALYSIS_TYPE} \
  --analysis-mode {ANALYSIS_MODE} \
  --selection-strategy {SELECTION_STRATEGY}
```

---

## 📍 **Stage 6: ML Analysis Generation**

### **When to Use:**
- Modified contrastive analysis logic
- Changed which insights to generate
- Fixed bugs in `stage6_analysis/` code

### **What Will Happen:**
- Stages 1, 2, 2.5, 2.6, 2.7, 3, 4, 5: **SKIP** (checkpoints exist)
- **Stages 6, 7: RE-RUN** (cascade)

### **Step 1: Navigate to Target Directory**

```bash
cd /home/jorge/rumiaifinal/data/clients/{CLIENT_ID}/{ANALYSIS_TYPE}s/{TARGET}/{ANALYSIS_MODE}_{SELECTION_STRATEGY}
```

### **Step 2: Delete Stage 6+ Checkpoints and Outputs**

```bash
# Delete Stage 6 checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_6_checkpoint.json
rm -rf buckets/bucket_*/analysis/contrastive_analysis/

# Delete Stage 7 outputs (no checkpoint file)
rm -f buckets/bucket_*/ml_analysis/llm/complete_analysis_*.json

echo "✓ Cleaned Stages 6-7"
```

### **Step 3: Verify Prerequisites Exist**

```bash
echo "=== VERIFYING PREREQUISITES ==="

# Stage 5 trained models
find buckets/bucket_*/analysis/models -name "*.pkl" | wc -l | xargs -I {} echo "  {} model files"

# Expect ~20-30 model files per bucket
```

### **Step 4: Run Pipeline**

```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

python rumiai_ml_batch.py \
  --client {CLIENT_ID} \
  --target {TARGET} \
  --analysis-type {ANALYSIS_TYPE} \
  --analysis-mode {ANALYSIS_MODE} \
  --selection-strategy {SELECTION_STRATEGY}
```

---

## 📍 **Stage 7: LLM Analysis**

### **When to Use:**
- Modified LLM prompts
- Changed narrative structure
- Fixed bugs in `stage7_llm/` code
- Want to regenerate final reports with new insights

### **What Will Happen:**
- Stages 1, 2, 2.5, 2.6, 2.7, 3, 4, 5, 6: **SKIP** (checkpoints exist)
- **Stage 7: RE-RUN** (only this stage)

### **Step 1: Navigate to Target Directory**

```bash
cd /home/jorge/rumiaifinal/data/clients/{CLIENT_ID}/{ANALYSIS_TYPE}s/{TARGET}/{ANALYSIS_MODE}_{SELECTION_STRATEGY}
```

### **Step 2: Delete Stage 7 Outputs**

```bash
# Delete Stage 7 outputs (no checkpoint file)
rm -f buckets/bucket_*/ml_analysis/llm/complete_analysis_*.json

echo "✓ Cleaned Stage 7"
```

**Note:** Stage 7 has no checkpoint file - it's identified by the presence of `complete_analysis_{BUCKET}.json` files.

### **Step 3: Verify Prerequisites Exist**

```bash
echo "=== VERIFYING PREREQUISITES ==="

# Stage 6 contrastive analysis
find buckets/bucket_*/analysis/contrastive_analysis -name "*.json" | wc -l | xargs -I {} echo "  {} analysis files"

# Expect ~10-15 JSON files per bucket
```

### **Step 4: Run Pipeline**

```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

python rumiai_ml_batch.py \
  --client {CLIENT_ID} \
  --target {TARGET} \
  --analysis-type {ANALYSIS_TYPE} \
  --analysis-mode {ANALYSIS_MODE} \
  --selection-strategy {SELECTION_STRATEGY}
```

---

## 🔧 **Troubleshooting**

### **Issue: Pipeline still skips stages I deleted checkpoints for**

**Cause:** Checkpoint validation may be checking output files, not just checkpoint existence.

**Solution:**
1. Verify you deleted BOTH checkpoint files AND output directories
2. Check for hidden state files (`.state.json`)
3. Try deleting parent directory and letting pipeline recreate it

---

### **Issue: "Stage X requires Stage Y outputs"**

**Cause:** You deleted outputs from a prerequisite stage.

**Solution:**
- Re-run from the earlier stage that provides the missing outputs
- Example: If Stage 5 fails due to missing Stage 4 outputs, re-run from Stage 4

---

### **Issue: Pipeline re-runs Stage 1 or 2 (expensive!)**

**Cause:** Accidentally deleted Stage 1/2 checkpoints or video files.

**Solution:**
1. **STOP THE PIPELINE IMMEDIATELY** (Ctrl+C)
2. Verify checkpoints exist:
   ```bash
   ls -la checkpoints/stage_1_checkpoint.json
   ls -la buckets/bucket_*/checkpoints/stage_2_checkpoint.json
   ```
3. If missing, restore from backup or accept re-scraping cost

---

### **Issue: "Config mismatch" errors for Stage 2**

**Cause:** Pipeline parameters differ from original run (e.g., `date_filter`).

**Expected behavior:** This is normal! Stage 2 will report config mismatch but skip processing because videos already exist. Pipeline will continue to later stages.

**When to worry:** Only if Stage 2 actually starts processing videos (you'll see "Processing video X/Y" messages).

---

## 📊 **Cost & Time Estimates**

| Starting Stage | Stages Re-run | Typical Duration | Anthropic API Cost |
|----------------|---------------|------------------|-------------------|
| Stage 3 | 3, 4, 5, 6, 7 | ~10-20 min | ~$0.50-$1.00 (Stage 7) |
| Stage 4 | 4, 5, 6, 7 | ~10-20 min | ~$0.50-$1.00 (Stage 7) |
| Stage 5 | 5, 6, 7 | ~8-15 min | ~$0.50-$1.00 (Stage 7) |
| Stage 6 | 6, 7 | ~6-12 min | ~$0.50-$1.00 (Stage 7) |
| Stage 7 | 7 only | ~5-10 min | ~$0.50-$1.00 |

**Note:** API costs are primarily from Stage 7 LLM calls. Stages 3-6 are pure computation (no API calls).

---

## ✅ **Success Indicators**

After pipeline completes, verify:

```bash
# Check all stages show COMPLETE
tail -50 data/logs/rumiai_ml_{CLIENT_ID}_{TARGET}_*.log | grep "COMPLETE"

# Verify final outputs exist
ls -la buckets/bucket_*/ml_analysis/llm/complete_analysis_*.json

# Should see 3 files (one per bucket)
```

---

## 📝 **Summary for Fresh Agent**

**TL;DR:**
1. Navigate to target directory
2. Delete checkpoints + outputs for chosen stage and all downstream stages
3. Verify expensive data (Stages 1, 2, 2.6, 2.7) still exists
4. Run pipeline - it will skip preserved stages and re-run from your chosen stage

**Key Safety:**
- ✅ NEVER delete Stage 1/2 checkpoints (video scraping/processing)
- ✅ NEVER delete taxonomy files (Stage 2.6 manual curation)
- ✅ NEVER delete classification outputs (Stage 2.7 LLM calls)
- ✅ ALWAYS verify prerequisites exist before running

**Common Pattern:**
```bash
# 1. Navigate
cd /home/jorge/rumiaifinal/data/clients/{CLIENT}/{TYPE}s/{TARGET}/{MODE}_{STRATEGY}

# 2. Delete stage N+ checkpoints and outputs
rm -f buckets/bucket_*/checkpoints/stage_{N}_checkpoint.json
rm -rf buckets/bucket_*/analysis/{STAGE_N_OUTPUTS}/
# ... repeat for stages N+1, N+2, etc.

# 3. Run pipeline
cd /home/jorge/rumiaifinal && source venv/bin/activate
python rumiai_ml_batch.py --client {CLIENT} --target {TARGET} --analysis-type {TYPE} --analysis-mode {MODE} --selection-strategy {STRATEGY}
```
