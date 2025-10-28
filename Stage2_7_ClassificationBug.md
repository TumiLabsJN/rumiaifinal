# Stage 2.7 Classification Bug Report

## 🚨 Critical Issue: 100% Classification Failure Rate

**Date**: 2025-10-28
**Test**: Test 4 (Rollo_Test4, wellness_test4)
**Stage**: 2.7 (Video Classification)
**Status**: STOPPED - Systemic failure

---

## Issue Summary

Stage 2.7 classification failed with **100% failure rate** (39/39 videos failed) due to Claude Haiku API returning invalid JSON format.

### Error Pattern
```
❌ Failed X/157: {video_id} - LLM returned invalid JSON: Extra data: line X column 1 (char XXX)
```

**Common error types:**
- `Extra data: line 25 column 1 (char 672)` - Most frequent
- `Extra data: line 27 column 1 (char 697)`
- `Extra data: line 35 column 1 (char 852)`
- `Expecting ',' delimiter: line 13 column 3 (char 403)` - Less frequent

---

## Test 4 State When Stopped

### ✅ Completed Stages:
- **Stage 0**: Foundation
- **Stage 1**: Video Discovery
  - 16/16 scrapes successful
  - 3,679 raw → 1,080 unique → 302 after 270-day filter
  - Winner buckets: `3-9s` (59), `60-90s` (47), `18-33s` (51)
  - **Total selected: 157 videos**
- **Stage 2**: Video Processing - All 157 videos processed through ML pipeline
- **Stage 2.5**: File Organization - Complete
- **Stage 2.6**: Pattern Discovery - Complete
  - Raw discovery: `wellness_test4_raw_discovery.json`
  - Curated taxonomy: `wellness_test4_taxonomy.json` (6 categories, 5 hooks)

### ❌ Failed Stage:
- **Stage 2.7**: Video Classification (STOPPED)
  - Progress: 39/157 videos attempted
  - Success: 0 videos
  - Failed: 39 videos (100% failure rate)
  - Reason: Invalid JSON from Claude Haiku API

### ⏸️ Not Started:
- Stage 3-7: Pending classification fix

---

## Root Cause Analysis

### Problem
Claude Haiku model is returning JSON with **extra content after the closing brace**, causing parsing failures.

### Why This Matters
The JSON parser expects **only** valid JSON, but Haiku is appending:
- Explanatory text
- Comments
- Additional context

Example structure causing failure:
```json
{
  "content_category": "supplement_recommendation",
  ...
}
Here's my reasoning for this classification...
```

The text after `}` causes: `Extra data: line X column 1`

### System Behavior
- Each video gets **3 retry attempts**
- All retries fail with same error
- System marks video as failed and continues
- With 100% failure rate, classification completes with 0 usable results

---

## Impact Assessment

### Immediate Impact
- ❌ No classification data for any videos
- ❌ Stages 3-7 cannot proceed meaningfully without classification
- ❌ ML models will lack content_category and hook_strategy features

### Data Preserved
- ✅ All 157 temporal_windows files intact (350+ ML features per video)
- ✅ Taxonomy files created and validated
- ✅ All checkpoints saved through Stage 2.6

---

## Next Steps for Investigation

### 1. **Examine Classification Prompt**
**File to check**: `ml_pipeline/stage2_content_analysis/classification.py`

**What to look for:**
- Prompt structure sent to Haiku
- JSON output instructions
- Whether prompt uses `response_format: "json"` or similar constraints

**Hypothesis**: Prompt may not be strict enough about JSON-only output

### 2. **Inspect Failed API Response**
**Action**: Add debug logging to capture raw Haiku response

**What to check:**
```python
# Before JSON parsing, log the raw response
logger.debug(f"Raw API response: {raw_response}")
```

**Goal**: See exact format of Haiku's output to understand what extra content is being added

### 3. **Review JSON Parsing Logic**
**File to check**: Classification module's response parsing

**What to look for:**
- Current parsing: `json.loads(response_text)`
- Potential fix: Strip everything after last `}`
- Alternative: Use regex to extract JSON block

**Potential fix pattern:**
```python
# Find last closing brace, ignore everything after
last_brace = response_text.rfind('}')
if last_brace != -1:
    json_text = response_text[:last_brace + 1]
    parsed = json.loads(json_text)
```

### 4. **Test Alternative Approaches**

**Option A: Stricter Prompt**
```
Return ONLY valid JSON. No explanations, no additional text, no markdown formatting.
Your response must start with { and end with }
```

**Option B: Switch Model**
- Try Claude Sonnet instead of Haiku
- More reliable for structured output
- Higher cost but better accuracy

**Option C: Pre-processing Response**
- Strip markdown code fences if present
- Extract JSON between first `{` and last `}`
- Handle common formatting issues

### 5. **Verify Taxonomy Structure**
**File to check**: `wellness_test4_taxonomy.json`

**Action**: Ensure taxonomy validation passed
```bash
cat /home/jorge/rumiaifinal/data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive/content_taxonomies/wellness_test4_taxonomy.json
```

**Confirm:**
- All category names are snake_case
- All definitions are >10 characters
- No special characters causing prompt issues

---

## Recommended Fix Priority

### Priority 1: Quick Fix (Try First)
**Add JSON extraction logic** to handle extra content:

```python
def extract_json(response_text):
    """Extract JSON from response that may have extra content."""
    # Strip markdown code fences
    text = response_text.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]

    # Find JSON boundaries
    first_brace = text.find('{')
    last_brace = text.rfind('}')

    if first_brace != -1 and last_brace != -1:
        json_text = text[first_brace:last_brace + 1]
        return json.loads(json_text)

    raise ValueError("No valid JSON found in response")
```

### Priority 2: Prompt Enhancement
Add to classification prompt:
```
CRITICAL: Return ONLY the JSON object. Do not include:
- Explanations before or after the JSON
- Reasoning or commentary
- Markdown formatting (no ```)
Your entire response must be valid JSON starting with { and ending with }
```

### Priority 3: Model Switch
If above fails, switch from Haiku to Sonnet in classification config.

---

## How to Resume Test 4

### Step 1: Apply Fix
Implement Priority 1 or 2 fix above in classification code.

### Step 2: Clear Failed Classifications
```bash
# Remove partial classification checkpoint (will retry all videos)
rm /home/jorge/rumiaifinal/data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive/buckets/*/.classification_checkpoint.json
```

### Step 3: Resume Pipeline
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

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

Pipeline will:
- Skip Stages 0-2.6 (checkpoints valid)
- Retry Stage 2.7 with fix applied
- Continue to Stages 3-7 if classification succeeds

---

## Files & Locations

### Logs
- Main log: `/home/jorge/rumiaifinal/data/logs/rumiai_ml_Rollo_Test4_wellness_test4_20251028_085043.log`
- Resume log: `/home/jorge/rumiaifinal/test4_execution_resume.log`

### Data Directory
```
/home/jorge/rumiaifinal/data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive/
├── buckets/
│   ├── bucket_3-9s/      (59 videos processed)
│   ├── bucket_60-90s/    (47 videos processed)
│   └── bucket_18-33s/    (51 videos processed)
├── content_taxonomies/
│   ├── wellness_test4_raw_discovery.json
│   └── wellness_test4_taxonomy.json
├── checkpoints/
│   └── stage_1_checkpoint.json
└── config.json
```

### Code Files to Investigate
- `ml_pipeline/stage2_content_analysis/classification.py` - Main classification logic
- `ml_pipeline/stage2_content_analysis/llm_client.py` - API interaction
- `ml_pipeline/stage2_content_analysis/taxonomy_validation.py` - Taxonomy validation

---

## Test 4 Anomalies (Separate from Bug)

### Low Video Count Issue
- Expected: ~2,800 videos after 270-day filter
- Actual: 302 videos (89% loss)
- Likely cause: Date filter calculation bug (filtering future dates)
- Impact: Only 157 videos selected instead of 300

This is a **separate issue** from the classification bug and should be investigated independently.

---

## Success Criteria for Fix

✅ Classification stage completes with:
- Success rate: >80% (some failures acceptable)
- At least 125/157 videos classified successfully
- All classification JSON files created
- Pipeline proceeds to Stage 3 automatically

---

**Status**: Ready for investigation and fix
**Next Owner**: Fresh CLI instance with this documentation
