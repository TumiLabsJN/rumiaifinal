# Stage 2.7 Classification Re-Test Instructions

**Purpose:** Re-run Stage 2.7 (Classification) with the fixed brace-counting code WITHOUT redoing expensive video processing or taxonomy discovery.

**Flow:** rollo_test5/wellnesspt2_test5

**Context:** This flow already has:
- ✅ Stage 2 complete (300 videos processed)
- ✅ Stage 2.5.1 complete (validation cache)
- ✅ Stage 2.6 complete (raw discovery)
- ✅ Curated taxonomy (wellnesspt2_test5_taxonomy.json)
- ⚠️ Stage 2.7 complete (BUT with OLD buggy code - 241/300 = 80.3% success)

**Goal:** Re-run ONLY Stage 2.7 with FIXED code to achieve ~95% success rate (285/300)

---

## Quick Start (For Fresh CLI Instance)

### Step 1: Delete Classification Outputs

```bash
cd /home/jorge/rumiaifinal/data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive

# Delete classification checkpoint (tells pipeline to re-run Stage 2.7)
rm .checkpoints/classification_checkpoint.json

# Delete classification outputs (clean slate)
rm -rf content_analysis/

# Leave everything else intact!
```

### Step 2: Verify What Remains (Important!)

```bash
# These should still exist:
ls -la checkpoints/stage_1_checkpoint.json  # ✅ Should exist
ls -la buckets/bucket_60-90s/checkpoints/stage_2_checkpoint.json  # ✅ Should exist
ls -la selection_manifest.json  # ✅ Should exist
ls -la content_taxonomies/transcript_validation_cache.json  # ✅ Should exist
ls -la content_taxonomies/wellnesspt2_test5_taxonomy.json  # ✅ Should exist

# These should be deleted:
ls -la .checkpoints/classification_checkpoint.json  # ❌ Should NOT exist
ls -la content_analysis/  # ❌ Should NOT exist
```

### Step 3: Re-run Pipeline

```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

python rumiai_ml_batch.py \
  --client rollo_test5 \
  --target wellnesspt2_test5 \
  --analysis-type hashtag \
  --analysis-mode top \
  --selection-strategy contrastive
```

**What will happen:**
1. ⏭️ Skip Stage 1 (checkpoint exists)
2. ⏭️ Skip Stage 2 (checkpoints exist)
3. ⏭️ Skip Stage 2.5 (files organized)
4. ⏭️ Skip Stage 2.5.1 (validation cache exists)
5. ⏭️ Skip Stage 2.6 (taxonomy exists)
6. ✅ **RUN Stage 2.7** (classification checkpoint deleted)

### Step 4: Monitor Progress

```bash
# In another terminal, tail the log
tail -f data/logs/rumiai_ml_rollo_test5_wellnesspt2_test5_*.log | grep -E "Stage 2.7|Classified|Failed|Complete"
```

**Expected time:** ~10-15 minutes (300 videos × 2-3 seconds per API call)

### Step 5: Analyze Results (CORRECT METHOD)

```bash
cd /home/jorge/rumiaifinal/data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive

# Check checkpoint (ground truth)
cat .checkpoints/classification_checkpoint.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
completed = len(data['completed'])
failed = len(data['failed'])
total = completed + failed
print(f'=== RESULTS ===')
print(f'Completed: {completed}/{total} ({completed/total*100:.1f}%)')
print(f'Failed: {failed}/{total} ({failed/total*100:.1f}%)')
print()
print('Expected with fix: ~285/300 (95%)')
print('Original buggy run: 241/300 (80.3%)')
"

# Count actual output files
echo "=== FILE COUNTS ==="
echo "bucket_60-90s: $(ls content_analysis/validated/bucket_60-90s/*.json 2>/dev/null | wc -l)"
echo "bucket_18-33s: $(ls content_analysis/validated/bucket_18-33s/*.json 2>/dev/null | wc -l)"
echo "bucket_33-60s: $(ls content_analysis/validated/bucket_33-60s/*.json 2>/dev/null | wc -l)"
echo "Total: $(find content_analysis/validated -name "*_content.json" | wc -l)"

# Check error breakdown
LOG_FILE=$(ls -t data/logs/rumiai_ml_rollo_test5_wellnesspt2_test5_*.log | head -1)
echo "=== ERROR BREAKDOWN ==="
grep "❌ Failed" $LOG_FILE | sed 's/.*Failed [0-9]*\/[0-9]*: [0-9]* - \(.*\)/\1/' | sed 's/: line.*//' | sort | uniq -c | sort -rn
```

---

## Expected Results

### BEFORE Fix (Original Run - 80.3% success)
```
Completed: 241/300 (80.3%)
Failed: 59/300 (19.7%)

Error Breakdown:
  44 Extra data (multiple JSON objects)
  15 Expecting value (empty responses)
```

### AFTER Fix (New Run - ~95% success)
```
Completed: ~285/300 (95%)
Failed: ~15/300 (5%)

Error Breakdown:
   0 Extra data (FIXED by brace-counting)
  15 Expecting value (still need investigation)
```

---

## What Gets Preserved

✅ **All video processing** (Stage 2 - temporal_windows files)
✅ **Validation cache** (Stage 2.5.1 - transcript_validation_cache.json)
✅ **Taxonomy** (Stage 2.6 - curated taxonomy)
✅ **All checkpoints** (Stage 1, Stage 2)
✅ **Manifest** (selection_manifest.json)

**Estimated cost savings:** $0 (no re-processing, no re-discovery)
**Time savings:** ~4-5 hours (skip video processing)

---

## What Gets Deleted & Recreated

❌ **Classification checkpoint** (.checkpoints/classification_checkpoint.json)
❌ **Classification outputs** (content_analysis/validated/)

🆕 **New classification results** (with fixed code)

---

## Verification Checklist

After pipeline completes, verify:

- [ ] Classification checkpoint shows ~285/300 completed (not 241/300)
- [ ] "Extra data" errors reduced from 44 to ~0
- [ ] "Expecting value" errors remain at ~15 (different issue)
- [ ] File counts per bucket increased:
  - bucket_60-90s: 89 → ~97
  - bucket_18-33s: 82 → ~94
  - bucket_33-60s: 70 → ~94
- [ ] Log shows no "Extra data" errors
- [ ] Success rate improved from 80.3% → ~95%

---

## Troubleshooting

### Issue: Pipeline re-runs Stage 2 (video processing)

**Cause:** Stage 2 checkpoints were accidentally deleted

**Solution:**
```bash
# Check if checkpoints exist
ls -la buckets/bucket_*/checkpoints/stage_2_checkpoint.json

# If missing, you'll need to restore from backup or accept re-processing
```

### Issue: Pipeline re-runs Stage 2.6 (taxonomy discovery)

**Cause:** Taxonomy file was deleted or state file corrupted

**Solution:**
```bash
# Check taxonomy exists
ls -la content_taxonomies/wellnesspt2_test5_taxonomy.json

# Check state file
cat .content_analysis_state.json
# Should show: "taxonomy_curated": true
```

### Issue: "Stage 2.5.1 must complete before Stage 2.7"

**Cause:** Validation cache missing

**Solution:**
```bash
# Check validation cache exists
ls -la content_taxonomies/transcript_validation_cache.json

# If missing, re-run pipeline to recreate it
```

---

## Files Modified by the Fix

The brace-counting fix was applied to:

1. `ml_pipeline/stage2_content_analysis/classification.py:67-100`
   - Updated `extract_json()` function
   - Changed from `rfind('}')` to brace-counting algorithm

2. `ml_pipeline/stage2_content_analysis/classification.py:1466`
   - Changed helper function to use `extract_json()`

3. `ml_pipeline/stage2_content_analysis/classification.py:1571`
   - Changed `classify_caption_only()` to use `extract_json()`

---

## Summary for Fresh Agent

**TL;DR:**
1. Delete `.checkpoints/classification_checkpoint.json`
2. Delete `content_analysis/` directory
3. Run full pipeline command
4. Verify success rate improved from 80% → 95%
5. Confirm "Extra data" errors dropped from 44 → 0

**Time:** 10-15 minutes
**Cost:** ~$0.30 (classification API calls only)
**No re-processing needed!**
