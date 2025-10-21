# METAPROMPT: Discovery Validation & Test Pipeline Continuation

**Purpose**: Run full blind recreation validation of Stage 2.6 discovery output, then continue with test pipeline (curation + classification)

**Context**: We implemented transcript validation filtering (2.6FilterImprovement.md) and re-ran Stage 2.6 discovery with filtered data (34 valid transcripts). A spot-check validation found the output is mostly accurate (7/10 quality) but needs full blind validation for maximum rigor.

---

## PART 1: Full Blind Recreation Validation (Option C)

### Your Task:

You will independently recreate the Stage 2.6 discovery analysis from scratch without seeing the existing LLM output, then compare your results to validate quality.

### Step 1: Load Context Documents

Read these files to understand the system:
1. `/home/jorge/rumiaifinal/QUICK_REFERENCE.md` - RumiAI system overview
2. `/home/jorge/rumiaifinal/SystemArchitecturev2.md` - Technical architecture
3. `/home/jorge/rumiaifinal/2.6FilterImprovement.md` - Transcript filtering enhancement
4. `/home/jorge/rumiaifinal/validation_report_test_vitamin.md` - Spot-check validation results (DO NOT read the discovery output section yet)

### Step 2: Read ALL 34 Transcripts (Blind)

**CRITICAL**: Do NOT read the existing discovery output yet. Read transcripts independently.

Load the validation cache to get the list of 34 valid video IDs:
```bash
cat /home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/transcript_validation_cache.json
```

For each valid video ID, read the transcript from:
```
/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json
```

**Reading Strategy**:
- Read all 34 transcripts systematically
- Take notes on patterns you observe
- Do NOT look at `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_raw_discovery.json` yet

### Step 3: Apply the EXACT Discovery Prompt

Use the discovery prompt from `/home/jorge/rumiaifinal/ml_pipeline/stage2_content_analysis/discovery.py` (lines 59-306).

**Key instructions from the prompt**:
- Identify patterns appearing in AT LEAST 5% of videos (minimum 2 videos, but prefer 10%+ = 3+ videos)
- 6 categories: content_categories, hook_strategies, audience_pain_points, trending_keywords, engagement_drivers, content_tactics
- Categories 1-2: Include frequency, examples, representative_video_ids
- Categories 3-6: Simple string lists
- Use snake_case naming
- Be objective and data-driven

### Step 4: Produce Your Discovery JSON

Create your independent analysis with this EXACT format:

```json
{
  "hashtag": "test_vitamin",
  "analysis_date": "{current_timestamp}",
  "sample_size": 34,
  "discovered_patterns": {
    "content_categories": [
      {
        "name": "your_category_name",
        "frequency": 8,
        "examples": ["example phrase 1", "example phrase 2"],
        "representative_video_ids": ["video_id_1", "video_id_2"]
      }
    ],
    "hook_strategies": [
      {
        "name": "your_hook_name",
        "frequency": 6,
        "examples": ["hook example 1", "hook example 2"],
        "representative_video_ids": ["video_id_1", "video_id_2"]
      }
    ],
    "audience_pain_points": ["pain_point_1", "pain_point_2"],
    "trending_keywords": ["keyword_1", "keyword_2"],
    "engagement_drivers": ["driver_1", "driver_2"],
    "content_tactics": ["tactic_1", "tactic_2"]
  }
}
```

Save your output to:
```
/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_BLIND_RECREATION.json
```

### Step 5: Load and Compare

NOW read the original LLM output:
```
/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_raw_discovery.json
```

Create a detailed comparison report:

**Comparison Metrics**:
1. **Pattern Overlap**: Which patterns appear in both outputs?
2. **Unique to Original**: Patterns LLM found that you didn't
3. **Unique to Yours**: Patterns you found that LLM didn't
4. **Frequency Differences**: For overlapping patterns, compare frequencies
5. **Quality Assessment**: Which output is more accurate/useful?

**Save comparison report to**:
```
/home/jorge/rumiaifinal/validation_report_BLIND_RECREATION.md
```

**Report Structure**:
```markdown
# Blind Recreation Validation Report

## Executive Summary
- Agreement rate: X%
- Major discrepancies: Y patterns
- Quality verdict: [BETTER/SIMILAR/WORSE than original]

## Pattern-by-Pattern Comparison

### Content Categories
| Pattern Name | Original Freq | Your Freq | Status |
|--------------|---------------|-----------|--------|
| health_education | 12 | X | [MATCH/DIFFER/MISSING] |

### Analysis
- Patterns both found: [list]
- Patterns only original found: [list] (possible hallucinations?)
- Patterns only you found: [list] (missed by original?)

## Conclusions
- Is original output trustworthy? YES/NO
- Recommended action: [ACCEPT/REVISE/REJECT original output]
```

---

## PART 2: Continue Test Pipeline

After validation, proceed with the test pipeline regardless of validation results (we can use either output for testing).

### Step 6: Manual Curation

**Input**: Choose which discovery output to curate:
- Original: `test_vitamin_raw_discovery.json` (from filtered LLM)
- Yours: `test_vitamin_BLIND_RECREATION.json` (your independent analysis)
- Hybrid: Merge best patterns from both

**Task**: Create curated taxonomy file

**Output**:
```
/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_taxonomy.json
```

**Curation Requirements**:

1. **Add definitions** to content_categories and hook_strategies:
```json
{
  "name": "health_education",
  "definition": "Educational content explaining vitamin benefits, deficiency symptoms, or health impacts"
}
```

2. **Remove invalid fields** (frequency, examples, representative_video_ids, percentage)

3. **Verify pattern thresholds**:
   - Each pattern must appear in ≥3 videos (10% of 34)
   - Remove patterns with frequency <3

4. **Check for duplicates**:
   - Merge "vitamin K2" and "k2 supplement" if both exist
   - Ensure case consistency (all lowercase)

5. **Validate snake_case naming**:
   - All pattern names must be lowercase with underscores
   - No spaces, capitals, or hyphens

**Example curated taxonomy**:
```json
{
  "content_categories": [
    {
      "name": "health_education",
      "definition": "Educational content explaining vitamin benefits or deficiency symptoms"
    },
    {
      "name": "supplement_review",
      "definition": "Videos listing or reviewing specific supplements taken by creator"
    },
    {
      "name": "consultation_roleplay",
      "definition": "Videos simulating doctor-patient consultations about vitamins"
    }
  ],
  "hook_strategies": [
    {
      "name": "symptom_listing",
      "definition": "Opens by listing symptoms or signs of deficiency"
    },
    {
      "name": "expertise_statement",
      "definition": "Opens with creator establishing health credentials or practices"
    }
  ],
  "audience_pain_points": [
    "vitamin deficiency",
    "chronic fatigue",
    "hormonal imbalance",
    "poor sleep",
    "low energy"
  ],
  "trending_keywords": [
    "vitamin d3",
    "vitamin b12",
    "k2 supplement",
    "gut health",
    "blood sugar"
  ],
  "engagement_drivers": [
    "expert explanation",
    "personal experience",
    "scientific backing",
    "simplified information"
  ],
  "content_tactics": [
    "symptom checklist",
    "product demonstration",
    "consultation format",
    "educational breakdown"
  ]
}
```

### Step 7: Validate Curated Taxonomy

Run validation to ensure proper format:

```bash
cd /home/jorge/rumiaifinal
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

**If validation fails**: Fix issues and re-run validation until it passes.

### Step 8: Run Stage 2.7 Classification

Classify all 111 videos using the curated taxonomy:

```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
set -a && source .env && set +a
DATA_ROOT=/home/jorge/rumiaifinal/data python -c "
import sys
sys.path.insert(0, '/home/jorge/rumiaifinal')

from ml_pipeline.stage2_content_analysis.classification import run_classification_stage

print('=' * 80)
print('STAGE 2.7: VIDEO CLASSIFICATION TEST')
print('=' * 80)
print()

# Run classification
result = run_classification_stage(
    client_id='test_final',
    hashtag='test_vitamin',
    analysis_mode='top',
    selection_strategy='contrastive',
    parallel=False,  # Sequential mode for testing
    checkpoint_enabled=True
)

print()
print('Classification Results:')
print(f'  Total videos: {result[\"total\"]}')
print(f'  Completed: {result[\"completed\"]}')
print(f'  Failed: {result[\"failed\"]}')
print(f'  Duration: {result[\"duration_seconds\"]}s')
print(f'  Mode: {result[\"mode\"]}')

if result['failed'] > 0:
    print()
    print(f'Failed video IDs: {result[\"failed_ids\"]}')
"
```

**Expected Output**:
- 111 classification files created
- Location: `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_analysis/`
- Files: `{video_id}_content.json` (one per video)

**Each classification file contains**:
```json
{
  "video_id": "7545713916584774968",
  "taxonomy_version": "stage2.6_output",
  "content_category": "supplement_review",
  "hook_strategy": "expertise_statement",
  "pain_points": ["vitamin deficiency", "low energy"],
  "keywords": ["vitamin d3", "gut health"],
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

### Step 9: Validate Classification Results

Spot-check 5-10 random classification files:

```bash
cd /home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_analysis

# Show random sample
ls *_content.json | shuf -n 5 | while read file; do
  echo "File: $file"
  cat "$file" | python3 -m json.tool
  echo "---"
done
```

**Check for**:
- ✅ All 12 required fields present
- ✅ Content category matches taxonomy (exact string match)
- ✅ Hook strategy matches taxonomy (exact string match)
- ✅ Multi-select fields are arrays (can be empty)
- ✅ Caption_analysis has all 8 subfields
- ✅ Confidence is "high", "medium", or "low"

### Step 10: Generate Test Summary Report

Create final test summary:

```bash
python3 -c "
import json
import os
from collections import Counter

content_dir = '/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_analysis'

# Count classification files
files = [f for f in os.listdir(content_dir) if f.endswith('_content.json')]

# Analyze classifications
categories = []
hooks = []
confidences = []
transcript_availability = []

for file in files:
    with open(os.path.join(content_dir, file), 'r') as f:
        data = json.load(f)
        categories.append(data.get('content_category'))
        hooks.append(data.get('hook_strategy'))
        confidences.append(data.get('confidence'))
        transcript_availability.append(data.get('transcript_available'))

print('=' * 80)
print('STAGE 2.7 CLASSIFICATION TEST SUMMARY')
print('=' * 80)
print()
print(f'Total videos classified: {len(files)}/111')
print()

print('Content Category Distribution:')
for cat, count in Counter(categories).most_common():
    print(f'  {cat}: {count} ({count/len(files)*100:.1f}%)')
print()

print('Hook Strategy Distribution:')
for hook, count in Counter(hooks).most_common():
    print(f'  {hook}: {count} ({count/len(files)*100:.1f}%)')
print()

print('Confidence Distribution:')
for conf, count in Counter(confidences).most_common():
    print(f'  {conf}: {count} ({count/len(files)*100:.1f}%)')
print()

print('Transcript Availability:')
for avail, count in Counter(transcript_availability).most_common():
    print(f'  {avail}: {count} ({count/len(files)*100:.1f}%)')
print()

print('=' * 80)
print('TEST PIPELINE COMPLETE')
print('=' * 80)
"
```

---

## Expected Timeline

**PART 1: Blind Recreation Validation**
- Step 1: Load context (5 min)
- Step 2-3: Read 34 transcripts + analyze (30-40 min)
- Step 4: Produce discovery JSON (10 min)
- Step 5: Compare outputs (15 min)
- **Total: ~60-70 minutes**

**PART 2: Test Pipeline Continuation**
- Step 6: Manual curation (10-15 min)
- Step 7: Validate taxonomy (2 min)
- Step 8: Run classification (2-3 min for 111 videos)
- Step 9: Spot-check results (5 min)
- Step 10: Generate summary (2 min)
- **Total: ~20-30 minutes**

**GRAND TOTAL: ~90-100 minutes**

---

## Success Criteria

### PART 1: Validation Success
- ✅ Blind recreation produces ≥70% pattern overlap with original
- ✅ Major patterns (top 3 categories/hooks) match
- ✅ Frequency differences <20% for overlapping patterns
- ✅ No major hallucinations detected in original
- **Verdict**: Original output is trustworthy

### PART 2: Pipeline Success
- ✅ Curated taxonomy passes validation
- ✅ All 111 videos classified successfully (0 failures)
- ✅ Classification files have proper schema
- ✅ Confidence distribution is reasonable (not all "low")
- ✅ Coverage: ≥70% of videos match at least one pattern
- **Verdict**: Stage 2.6 → 2.7 pipeline works end-to-end

---

## Files You Will Create

**PART 1 Outputs**:
1. `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_BLIND_RECREATION.json`
2. `/home/jorge/rumiaifinal/validation_report_BLIND_RECREATION.md`

**PART 2 Outputs**:
3. `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_taxonomies/test_vitamin_taxonomy.json`
4. 111 classification files: `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/content_analysis/{video_id}_content.json`

---

## Important Notes

1. **DO NOT skip the blind recreation** - This is the most rigorous validation method
2. **Read ALL 34 transcripts** - Don't sample, read every single one
3. **Follow the EXACT prompt** - Use the discovery prompt from discovery.py
4. **Be objective** - Don't bias yourself based on spot-check validation results
5. **Environment variables**: Remember to set DATA_ROOT and load .env for API key

---

## If You Encounter Issues

**Issue: Validation cache not found**
- Run Stage 2.5.5 first: `validate_all_transcripts("test_final", "test_vitamin")`

**Issue: ANTHROPIC_API_KEY not set**
- Ensure you run: `source .env` before classification

**Issue: Classification fails for some videos**
- Check checkpoint file: `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/.checkpoints/classification_checkpoint.json`
- Resume by re-running classification (checkpoint will skip completed videos)

**Issue: Too many context tokens**
- Read transcripts in batches (e.g., 10 at a time)
- Summarize patterns incrementally

---

## Questions to Answer in Your Report

**PART 1 Validation**:
1. What is the pattern overlap percentage?
2. Are there patterns the original LLM hallucinated (not grounded in transcripts)?
3. Are there obvious patterns the original LLM missed?
4. Which output is higher quality? Why?
5. Should we trust the original output for Stage 2.7?

**PART 2 Pipeline**:
6. Did all 111 videos classify successfully?
7. What's the confidence distribution? (% high/medium/low)
8. Do classifications make sense? (spot-check 5 examples)
9. Does the validation filter improve classification quality?
10. Is the Stage 2.6 → 2.7 pipeline ready for production?

---

**END OF METAPROMPT**

Good luck! Execute methodically and document everything.
