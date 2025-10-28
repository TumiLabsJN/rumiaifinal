# 🚀 Jumpstart Prompt for Stage 7 Validation Bug Fix

## Context: RumiAI ML Pipeline End-to-End Test

You are helping fix a validation bug in the RumiAI ML pipeline that occurred during **Test 4** - an end-to-end test of the complete ML training pipeline for TikTok video analysis.

### What is RumiAI?
- **Purpose**: Internal ML tool for Tumi Labs' RippleOS consultancy
- **Function**: Analyzes TikTok videos through 9 ML services to extract 60+ features per video
- **Pipeline**: 7-stage ML training pipeline that processes videos → trains models → generates creative insights
- **Current Test**: Test 4 (Client: Rollo_Test4, Hashtag: wellness_test4, 157 videos)

### Test 4 Context
This test was running the complete pipeline from video discovery through ML model training and LLM-generated creative reports. We recently fixed a **Stage 2.7 classification bug** (97.5% success rate achieved), and the pipeline successfully completed **Stages 0-6**. However, Stage 7 failed partway through with a validation error.

---

## The Bug You Need to Fix

**Read the full bug report first**: `Stage7E2EBug.md` in the current directory.

### Quick Summary
- **Stage**: 7 (LLM Analysis - Creative Report Generation)
- **Error**: `AssertionError: Stage 7 Phase 2 output missing: synthesis.json`
- **Status**: Bucket 3-9s completed successfully, but validation failed before processing buckets 60-90s and 18-33s
- **Root Cause**: Validation expects `synthesis.json` but Stage 7 actually generates `winning_formulas.json`

### What Completed Successfully
- ✅ Stages 0-6: Complete for all 3 buckets (3-9s, 60-90s, 18-33s)
- ✅ Stage 7 for bucket 3-9s: Generated 4 JSON files including `winning_formulas.json`
- ❌ Validation failed with schema mismatch before continuing to remaining buckets

### The Specific Problem
```
Expected file:    synthesis.json           ❌ Does not exist
Actual file:      winning_formulas.json    ✅ Successfully generated (6543 bytes)
```

---

## Your Mission

Fix the validation logic so Stage 7 can complete for all 3 buckets without errors.

### Success Criteria
1. Pipeline validates Stage 7 output correctly for bucket 3-9s (already complete)
2. Pipeline proceeds to process buckets 60-90s and 18-33s
3. All 3 buckets generate complete LLM analysis
4. No validation errors in logs
5. Pipeline exits with code 0 (success)

---

## Required Discovery (Do This FIRST)

### Phase 1: Understand Current Stage 7 Output Schema

**1.1 Examine what bucket 3-9s actually generated:**
```bash
ls -la /home/jorge/rumiaifinal/data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive/buckets/bucket_3-9s/ml_analysis/llm/
```

**Expected files:**
- `hook_analysis.json` (Phase 1 output)
- `closing_analysis.json` (Phase 1 output)
- `winning_formulas.json` (Phase 2 output - 6543 bytes)
- `complete_analysis_3-9s.json` (Full analysis)

**1.2 Read one of these files to understand the schema:**
```bash
cat /home/jorge/rumiaifinal/data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive/buckets/bucket_3-9s/ml_analysis/llm/winning_formulas.json | jq '.' | head -50
```

**Questions to answer:**
- What is the structure of `winning_formulas.json`?
- Does it contain creative reports?
- Is this the final Stage 7 output format?

---

### Phase 2: Locate the Validation Code

**2.1 Search for the validation that's failing:**
```bash
cd /home/jorge/rumiaifinal
grep -rn "synthesis.json" --include="*.py"
```

**2.2 Search for the assertion error:**
```bash
grep -rn "Stage 7 Phase 2 output missing" --include="*.py"
```

**2.3 Find where validation happens:**
```bash
grep -rn "AssertionError.*Stage 7" --include="*.py"
```

**Expected locations:**
- `rumiai_ml_batch.py` - Main pipeline orchestration (most likely)
- `rumiai/stage7_llm_analysis.py` - Stage 7 implementation
- `ml_pipeline/stage7_*/validation.py` - If validation is modularized

**Questions to answer:**
- Where is the validation code located?
- What file(s) does it check for?
- Does it validate after each bucket or at the end of Stage 7?
- Is validation in the main orchestrator or in Stage 7 module?

---

### Phase 3: Understand Stage 7 File Generation

**3.1 Find where winning_formulas.json is created:**
```bash
grep -rn "winning_formulas.json" --include="*.py"
```

**3.2 Examine the Stage 7 LLM analysis code:**
```bash
# Based on logs, this file exists:
cat rumiai/stage7_llm_analysis.py | grep -A 10 -B 5 "winning_formulas"
```

**3.3 Check if synthesis.json was ever used:**
```bash
# Search git history (if available)
git log --all --oneline --grep="synthesis"
git log --all --oneline --grep="winning_formulas"
```

**Questions to answer:**
- When does Stage 7 save `winning_formulas.json`?
- Was `synthesis.json` the old naming convention?
- Did a recent refactor change the output schema?
- Are there comments explaining the schema change?

---

### Phase 4: Trace the Error in Logs

**4.1 Find the exact error context in the latest log:**
```bash
grep -A 10 -B 10 "synthesis.json" /home/jorge/rumiaifinal/data/logs/rumiai_ml_Rollo_Test4_wellness_test4_20251028_095929.log
```

**4.2 Check what happened right before the error:**
```bash
grep -B 20 "Stage 7 unexpected error" /home/jorge/rumiaifinal/data/logs/rumiai_ml_Rollo_Test4_wellness_test4_20251028_095929.log
```

**4.3 Look for Stage 7 completion logs:**
```bash
grep "Stage 7 Complete" /home/jorge/rumiaifinal/data/logs/rumiai_ml_Rollo_Test4_wellness_test4_20251028_095929.log
```

**Questions to answer:**
- Where exactly in the code flow does validation happen?
- Is it after Phase 2 completes?
- Is it before starting the next bucket?
- What module logs the error? (`__main__` suggests orchestrator)

---

### Phase 5: Check for Schema Documentation

**5.1 Search for Stage 7 documentation:**
```bash
find /home/jorge/rumiaifinal -name "*stage7*" -o -name "*Stage7*" | grep -E "\.(md|txt|rst)$"
```

**5.2 Check if there's a schema definition:**
```bash
grep -rn "synthesis" /home/jorge/rumiaifinal/documentation_migration/
grep -rn "winning_formulas" /home/jorge/rumiaifinal/documentation_migration/
```

**5.3 Look for validation schemas:**
```bash
find /home/jorge/rumiaifinal -name "*validation*" -o -name "*schema*" | grep -v __pycache__
```

**Questions to answer:**
- Is there documentation explaining Stage 7 output schema?
- Are there schema validation files (JSON schema, Pydantic models)?
- Is the expected output documented anywhere?

---

## Recommended Fix Strategy

After completing discovery, implement **Option 1 from Stage7E2EBug.md** (Update Validation):

### Step 1: Locate Validation Code
Based on discovery, find the exact line(s) checking for `synthesis.json`.

### Step 2: Understand Intent
Determine what the validation is checking:
- File existence?
- File schema?
- Complete Stage 7 output?

### Step 3: Update Validation Logic
Replace check for `synthesis.json` with check for `winning_formulas.json`:

**Current (broken):**
```python
assert os.path.exists(f"{llm_dir}/synthesis.json"), "Stage 7 Phase 2 output missing: synthesis.json"
```

**Fixed:**
```python
assert os.path.exists(f"{llm_dir}/winning_formulas.json"), "Stage 7 Phase 2 output missing: winning_formulas.json"
assert os.path.exists(f"{llm_dir}/complete_analysis_{bucket}.json"), f"Stage 7 complete analysis missing for {bucket}"
```

### Step 4: Verify Phase 1 Outputs (If Needed)
If validation also checks Phase 1 outputs, ensure it checks for:
```python
assert os.path.exists(f"{llm_dir}/hook_analysis.json"), "Stage 7 Phase 1 output missing: hook_analysis.json"
assert os.path.exists(f"{llm_dir}/closing_analysis.json"), "Stage 7 Phase 1 output missing: closing_analysis.json"
```

### Step 5: Test the Fix
After updating validation:
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

# Re-run pipeline (will skip completed stages 0-6)
python rumiai_ml_batch.py \
  --client Rollo_Test4 \
  --target wellness_test4 \
  --analysis-type hashtag \
  --selection-strategy contrastive \
  --video-count 100 \
  --date-filter last_270_days \
  --country-code US \
  --report-type single \
  --report-audience client
```

**Expected behavior:**
- Skips Stages 0-6 (already complete)
- Validates bucket 3-9s Stage 7 output ✅
- Proceeds to bucket 60-90s Stage 7
- Completes bucket 60-90s Stage 7 ✅
- Proceeds to bucket 18-33s Stage 7
- Completes bucket 18-33s Stage 7 ✅
- Exits with code 0

---

## Important Notes

### Do NOT Re-Run Stages 0-6
The checkpoint system will automatically skip:
- Stage 1: Video discovery (157 videos already selected)
- Stage 2: Video processing (157/157 already processed)
- Stage 2.7: Classification (153/157 already classified)
- Stages 3-6: ML models already trained

### Stage 7 Bucket 3-9s
Bucket 3-9s already completed successfully. The validation fix should:
1. Recognize bucket 3-9s is complete
2. Skip re-processing bucket 3-9s
3. Start with bucket 60-90s

If the pipeline tries to re-process bucket 3-9s, it should be safe (Phase 2 is idempotent), but wasteful (~25 seconds + $0.15 API cost).

### Classification Bug Context
We recently fixed a Stage 2.7 classification bug where Claude Haiku was appending extra text after JSON. The fix achieved 97.5% success rate (153/157 videos). This Stage 7 validation bug is **completely separate** - it's a file naming mismatch, not an LLM output issue.

---

## Alternative Approaches (If Discovery Reveals Issues)

### If validation is more complex than file existence:
- Check what else validation does (schema validation? content checks?)
- Update all validation logic to match current Stage 7 output format

### If synthesis.json is referenced in multiple places:
- Create an alias: `ln -s winning_formulas.json synthesis.json`
- Or do a comprehensive rename across the codebase

### If this is part of a larger schema evolution:
- Check if other stages also have legacy validation
- Update documentation to reflect new schema
- Add migration notes for future developers

---

## Key Files & Directories

### Working Directory
```
/home/jorge/rumiaifinal/
```

### Test 4 Data Directory
```
/home/jorge/rumiaifinal/data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive/
```

### Stage 7 Output (Bucket 3-9s - Already Complete)
```
buckets/bucket_3-9s/ml_analysis/llm/
├── hook_analysis.json
├── closing_analysis.json
├── winning_formulas.json         ← Generated by Stage 7
└── complete_analysis_3-9s.json
```

### Buckets Pending Stage 7
```
buckets/bucket_60-90s/ml_analysis/llm/   ← Empty, waiting for Stage 7
buckets/bucket_18-33s/ml_analysis/llm/   ← Empty, waiting for Stage 7
```

### Logs
```
data/logs/rumiai_ml_Rollo_Test4_wellness_test4_20251028_095929.log
```

### Code Files to Check (Likely Locations)
```
rumiai_ml_batch.py                    ← Main orchestrator (likely has validation)
rumiai/stage7_llm_analysis.py         ← Stage 7 implementation
ml_pipeline/stage7_*/                 ← Stage 7 modules (if they exist)
```

---

## Questions to Ask During Discovery

1. **Where is the validation code?** (File and line number)
2. **What does it validate?** (Just file existence, or content too?)
3. **When does validation run?** (After each bucket? End of Stage 7? Before next stage?)
4. **Why the name mismatch?** (Was synthesis.json renamed to winning_formulas.json?)
5. **Are there other validation checks?** (Do they also need updating?)
6. **Is there documentation?** (Schema definitions, comments, docstrings?)
7. **Is this a known issue?** (Git history, TODO comments, issue tracker?)

---

## Success Checklist

After implementing the fix, verify:

- [ ] Validation code updated to check for `winning_formulas.json`
- [ ] Validation passes for bucket 3-9s (already complete)
- [ ] Pipeline proceeds to bucket 60-90s
- [ ] Bucket 60-90s Stage 7 completes successfully
- [ ] Pipeline proceeds to bucket 18-33s
- [ ] Bucket 18-33s Stage 7 completes successfully
- [ ] All 3 buckets have complete LLM analysis files
- [ ] Pipeline exits with code 0
- [ ] No assertion errors in logs
- [ ] Log shows "Stage 7 complete" for all 3 buckets

---

## Emergency Fallback

If validation is deeply embedded and hard to fix:

**Quick workaround:**
```bash
# For each bucket, create alias
cd /home/jorge/rumiaifinal/data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive/buckets/bucket_3-9s/ml_analysis/llm/
ln -s winning_formulas.json synthesis.json

# This satisfies validation without changing code
```

Then re-run the pipeline. This is **not recommended** as a permanent fix, but can unblock testing while you implement proper validation updates.

---

**Status**: Ready for discovery and fix
**Priority**: Medium (Stage 7 is post-ML-training, not blocking core pipeline)
**Estimated Time**: 30-60 minutes (discovery + fix + test)
