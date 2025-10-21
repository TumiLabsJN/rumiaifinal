# Stage 2.6 Test Completion Workflow

**Date**: 2025-10-16
**Purpose**: Guide for completing Stage 2.6 testing and transitioning to Stage 2.7
**Current Status**: Stage 2.6 discovery complete, awaiting manual curation

---

## Current State

### Completed ✅

1. **Stage 2.5 Enhancement**: Created `selection_manifest.json`
   - File: `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/selection_manifest.json`
   - Contains: 88 top performers + 23 bottom performers across 3 buckets

2. **Stage 2.6 Pattern Discovery**: Ran LLM discovery on 77 valid transcripts
   - File: `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_raw_discovery.json`
   - Discovered: 3 content categories, 2 hook strategies, 5 pain points, 5 keywords, 4 engagement drivers, 4 content tactics

3. **Improvements Made**:
   - Frequency threshold: 10% → 5% (to capture more patterns)
   - Sample size: 50 → 100 (uses all 88 available top performers)
   - Smart sampling: Uses all available, doesn't fail if fewer than requested

### Pending ⏳

1. **Manual Curation**: You need to curate `test_vitamin_raw_discovery.json`
2. **Transcript Filtering**: Optional improvement documented in `2.6FilterImprovement.md` (not blocking)
3. **Stage 2.7 Testing**: Video classification with curated taxonomy

---

## What You Need To Do: Manual Curation

### Step 1: Review Raw Discovery File

**Location:**
```
/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_raw_discovery.json
```

**What to look for:**

#### Categories 1-2: Semantic Categories (Need Definitions)

Check `content_categories` and `hook_strategies`:
- ✅ Keep patterns with reasonable frequency (5%+)
- ❌ Remove patterns with <5% frequency
- 🔄 Merge similar patterns (e.g., "supplement_review" + "product_review" → "supplement_review")
- ✏️ **Add definitions** (minimum 10 characters) - this is required!
- ✏️ Ensure names are `snake_case` (lowercase, underscores only)

**Example transformation:**

**Before (raw discovery):**
```json
{
  "name": "supplement_review",
  "frequency": 8,
  "examples": ["vitamin D supplementates", "all the supplements I take"],
  "representative_video_ids": ["7533996320898420023", "7560443128885366046"],
  "percentage": 10.4
}
```

**After (curated - ADD definition):**
```json
{
  "name": "supplement_review",
  "definition": "Videos reviewing or listing specific vitamin/supplement products taken by the creator"
}
```

#### Categories 3-6: Simple Lists

Check `audience_pain_points`, `trending_keywords`, `engagement_drivers`, `content_tactics`:
- ✅ Keep items that are clear and specific
- ❌ Remove duplicates
- ❌ Remove items that are too vague or generic
- 🔄 Merge similar items if needed

**These are just string arrays, no definitions needed.**

### Step 2: Create Curated Taxonomy File

**Target file:**
```
/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_taxonomy.json
```

**Required structure:**

```json
{
  "content_categories": [
    {
      "name": "supplement_review",
      "definition": "Videos reviewing or listing specific vitamin/supplement products taken by the creator"
    },
    {
      "name": "medical_education",
      "definition": "Educational content explaining vitamin benefits, deficiency symptoms, or health impacts"
    },
    {
      "name": "consultation_simulation",
      "definition": "Videos simulating a doctor-patient consultation discussing vitamin needs"
    }
  ],
  "hook_strategies": [
    {
      "name": "symptom_list",
      "definition": "Opens by listing symptoms or signs of deficiency"
    },
    {
      "name": "authority_statement",
      "definition": "Opens with creator establishing credibility or personal health practices"
    }
  ],
  "audience_pain_points": [
    "vitamin deficiency",
    "low energy",
    "poor sleep",
    "hormonal imbalance",
    "chronic fatigue"
  ],
  "trending_keywords": [
    "vitamin D3",
    "vitamin B12",
    "vitamin K2",
    "gut health",
    "apple cider vinegar"
  ],
  "engagement_drivers": [
    "expert explanation",
    "personal experience",
    "scientific backing",
    "symptom checklist"
  ],
  "content_tactics": [
    "benefits list",
    "problem solution format",
    "medical consultation style",
    "educational breakdown"
  ]
}
```

**Key requirements:**
- ✅ All 6 top-level fields present
- ✅ Categories 1-2 have `name` + `definition` (NOT `examples`, `frequency`, etc.)
- ✅ Categories 3-6 are simple string arrays
- ✅ Names are `snake_case`
- ✅ Definitions are at least 10 characters
- ✅ No duplicate names or items

### Step 3: Validate Your Taxonomy (Optional but Recommended)

You can validate your curated file before telling me to proceed:

```bash
source venv/bin/activate
python -c "
from ml_pipeline.stage2_content_analysis.taxonomy_validation import validate_curated_taxonomy

taxonomy_path = '/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_taxonomy.json'

try:
    validate_curated_taxonomy(taxonomy_path)
    print('✅ Taxonomy validation passed!')
except Exception as e:
    print(f'❌ Validation failed: {e}')
"
```

**Common validation errors:**
- Missing `name` or `definition` fields
- Names not in `snake_case` (e.g., "Supplement Review" instead of "supplement_review")
- Definitions too short (<10 chars)
- Duplicate names or items
- Empty arrays in categories 3-6

---

## What To Tell Me When Ready

### Trigger Phrase Options

Once you've curated the taxonomy file, tell me **any of these**:

**Option 1: Simple trigger**
```
"Curation complete, test Stage 2.7"
```

**Option 2: Explicit path**
```
"I've curated the taxonomy at:
/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_taxonomy.json

Test Stage 2.7"
```

**Option 3: With validation request**
```
"Curation done. Validate taxonomy then test Stage 2.7"
```

**Option 4: Ask for review first**
```
"Curation done. Review my taxonomy before testing Stage 2.7"
```

### What Happens Next

When you trigger Stage 2.7 testing, I will:

1. **Validate your curated taxonomy**
   - Check file exists
   - Validate structure and schema
   - Ensure all requirements met

2. **Run Stage 2.7 classification**
   - Classify all 111 videos (88 top + 23 bottom performers)
   - Use your curated taxonomy as the classification schema
   - Generate `{video_id}_content.json` files for each video

3. **Show classification results**
   - Summary statistics (completed/failed counts)
   - Sample classifications for review
   - Coverage metrics (high/medium/low confidence)

4. **Verify Stage 2.6/2.7 integration**
   - Confirm full pipeline works end-to-end
   - Validate output format for downstream stages

---

## Alternative Paths

### Path A: Skip Curation, Use Raw Discovery

If you want to test Stage 2.7 **without manual curation** (quick test):

**Trigger:**
```
"Skip curation, convert raw discovery to taxonomy format and test Stage 2.7"
```

**What I'll do:**
- Auto-convert `test_vitamin_raw_discovery.json` to taxonomy format
- Add minimal definitions (10 chars minimum)
- Save as `test_vitamin_taxonomy.json`
- Proceed with Stage 2.7 test

**Pros:**
- Fast (no manual work)
- Tests the integration immediately

**Cons:**
- Lower quality taxonomy
- May have patterns that should be removed
- Not production-representative

### Path B: Implement Transcript Filtering First

If you want to implement the transcript quality filter before proceeding:

**Trigger:**
```
"Implement 2.6FilterImprovement.md, then re-run discovery"
```

**What I'll do:**
1. Add `is_valid_transcript()` function
2. Update sampling logic to filter noise
3. Re-run Stage 2.6 discovery
4. Wait for you to curate the improved results
5. Then test Stage 2.7

**Pros:**
- More accurate coverage metrics
- Better quality patterns
- Production-ready filtering

**Cons:**
- Takes longer (~35 min implementation + re-run discovery)
- Another round of manual curation needed

---

## Expected Stage 2.7 Test Flow

### What I'll Run

```python
from ml_pipeline.stage2_content_analysis.classification import run_classification_stage

result = run_classification_stage(
    client_id='test_final',
    hashtag='test_vitamin',
    analysis_mode='top',
    selection_strategy='contrastive',
    parallel=False,  # Sequential mode (safer for testing)
    max_workers=5,   # If parallel enabled
    checkpoint_enabled=True  # Resume support
)
```

### What You'll See

```
================================================================================
STAGE 2.7: VIDEO CLASSIFICATION TEST
================================================================================

Step 1/3: Validating inputs...
✓ Taxonomy validation passed
✓ Selection manifest found

Step 2/3: Loading taxonomy and manifest...
✓ Files loaded

Step 3/3: Classifying all videos (Claude Haiku)...
Classification mode: sequential
Classifying 111 videos across 3 buckets...

✅ Classified (1/111): 7545713916584774968
✅ Classified (2/111): 7544734155570105656
...
✅ Classified (111/111): 7560964241084271903

✓ Stage 2.7: Classified 111/111 videos in 125.3s

================================================================================
CLASSIFICATION COMPLETE
================================================================================
Mode: sequential
Total videos: 111
Completed: 111
Failed: 0
Duration: 125.3s
```

### Output Files Created

```
/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_analysis/
├── 7545713916584774968_content.json
├── 7544734155570105656_content.json
├── 7554856008615529783_content.json
├── ... (111 files total)
```

**Each file contains:**
```json
{
  "video_id": "7545713916584774968",
  "taxonomy_version": "stage2.6_output",
  "content_category": "supplement_review",
  "hook_strategy": "authority_statement",
  "pain_points": ["vitamin deficiency", "low energy"],
  "keywords": ["vitamin D3", "gut health"],
  "engagement_drivers": ["personal experience"],
  "content_tactics": ["benefits list"],
  "confidence": "high",
  "transcript_available": true,
  "note": null,
  "caption_analysis": {
    "hook_type": "statement",
    "cta_type": "link_in_bio",
    "brand_mention_present": true,
    "influencer_tag_present": false,
    "emoji_usage": "some",
    "caption_length": "long",
    "hashtag_count": 8,
    "hashtag_placement": "end"
  }
}
```

---

## Success Criteria

### Stage 2.6 Test Complete When:

- ✅ Raw discovery file created with valid patterns
- ✅ Manual curation completed (or deliberately skipped)
- ✅ Curated taxonomy file validated successfully

### Stage 2.7 Test Complete When:

- ✅ All 111 videos classified without errors
- ✅ Classification files created (111 `*_content.json` files)
- ✅ Output format validated for downstream stages
- ✅ Confidence distribution is reasonable (not all "low")

### Full Integration Validated When:

- ✅ Stage 2.5 → 2.6 → 2.7 flow works seamlessly
- ✅ No manual file intervention needed (except curation)
- ✅ All intermediate files in correct locations
- ✅ Checkpoint/resume works (if classification interrupted)

---

## FAQ

### Q: How long does Stage 2.7 take?

**A:** ~2 minutes for 111 videos in sequential mode
- Sequential: ~1 second per video (safe, default)
- Parallel (5 workers): ~30 seconds total (faster, optional)

### Q: What if I make a mistake in curation?

**A:** Just tell me and I'll help you fix it before running Stage 2.7:
```
"I made a mistake in the taxonomy, help me fix [specific issue]"
```

### Q: Can I test Stage 2.7 multiple times?

**A:** Yes! Classification is idempotent (safe to re-run):
- Re-running overwrites previous classification files
- Checkpoint allows interruption and resume
- No side effects on other stages

### Q: What if Stage 2.7 fails?

**A:** I'll show you:
1. Exact error message
2. Which video failed (if specific video issue)
3. Validation errors (if taxonomy issue)
4. How to fix and retry

---

## Quick Reference Commands

### View Raw Discovery
```bash
cat /home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_raw_discovery.json
```

### View Curated Taxonomy (after you create it)
```bash
cat /home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_taxonomy.json
```

### Validate Taxonomy
```bash
source venv/bin/activate
python -c "
from ml_pipeline.stage2_content_analysis.taxonomy_validation import validate_curated_taxonomy
validate_curated_taxonomy('/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_taxonomy.json')
print('✅ Valid')
"
```

### Check Classification Progress (if running)
```bash
ls /home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_analysis/*.json | wc -l
```

---

## Next Document Suggestions

After Stage 2.7 completes, we should create:
- **Stage2.7TestCompletion.md** - How to validate classification results and proceed to Stage 3
- **Stage2IntegrationSummary.md** - Full Stage 2 → 2.5 → 2.6 → 2.7 end-to-end test report

---

**Last Updated**: 2025-10-16
**Status**: Awaiting manual curation
**Ready for**: Your taxonomy curation work
