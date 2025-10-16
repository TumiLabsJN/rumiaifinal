# Stage 2: Content Analysis (Stages 2.6 & 2.7)

Pattern discovery and video classification using LLM for TikTok video analysis.

**Source**: ContentAnalysisCHILDTI.md
**Version**: 1.0.0
**Total Lines**: 1,975 (across 7 Python files)

---

## Overview

This module implements two-stage content analysis:

1. **Stage 2.6: Pattern Discovery** - Discover content patterns from sample transcripts using Claude Sonnet
2. **Stage 2.7: Video Classification** - Classify all videos using curated taxonomy + Claude Haiku

---

## Module Structure

```
stage2_content_analysis/
├── __init__.py                 # Module initialization (81 lines)
├── utils.py                    # Shared utilities (204 lines)
├── validation.py               # Input/output validation (356 lines)
├── error_handlers.py           # Error handling utilities (195 lines)
├── discovery.py                # Stage 2.6 implementation (464 lines)
├── classification.py           # Stage 2.7 implementation (504 lines)
├── test_integration.py         # Integration tests (171 lines)
└── README.md                   # This file
```

---

## Quick Start

### Stage 2.6: Pattern Discovery

```python
from ml_pipeline.stage2_content_analysis import run_discovery_stage

# Run discovery for a hashtag
raw_taxonomy = run_discovery_stage(
    client_id="acme_corp",
    hashtag="nutrition",
    analysis_mode="top",
    selection_strategy="contrastive",
    sample_size=50
)

# Output: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/content_taxonomies/nutrition_raw_discovery.json
```

**What it does:**
1. Samples 50 transcripts from top 3 buckets (stratified)
2. Calls Claude Sonnet to discover 6 pattern categories
3. Saves raw discovery JSON for manual curation

**Manual step required**: Curate `raw_discovery.json` → `taxonomy.json`

---

### Stage 2.7: Video Classification

```python
from ml_pipeline.stage2_content_analysis import run_classification_stage

# Run classification for all videos
stats = run_classification_stage(
    client_id="acme_corp",
    hashtag="nutrition",
    analysis_mode="top",
    selection_strategy="contrastive"
)

# Output: 120 files at /data/.../buckets/bucket_*/content_analysis/*_content.json
```

**What it does:**
1. Loads curated taxonomy from Stage 2.6
2. Classifies all videos (40 per bucket × 3 buckets) using Claude Haiku
3. Saves individual classification files per video

---

## Environment Requirements

### Required Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Python Dependencies

```
anthropic>=0.20.0
```

---

## API Usage

### Discovery Functions

```python
from ml_pipeline.stage2_content_analysis import (
    sample_transcripts_for_discovery,
    discover_patterns_llm,
    calculate_percentages,
    run_discovery_stage
)

# Sample transcripts (Step 1)
transcripts = sample_transcripts_for_discovery(
    manifest_path="/path/to/selection_manifest.json",
    sample_size=50
)

# Discover patterns with LLM (Step 2)
raw_taxonomy = discover_patterns_llm(
    transcripts=transcripts,
    hashtag="nutrition",
    client_id="acme_corp"
)

# Or run complete pipeline
raw_taxonomy = run_discovery_stage(
    client_id="acme_corp",
    hashtag="nutrition",
    sample_size=50
)
```

### Classification Functions

```python
from ml_pipeline.stage2_content_analysis import (
    classify_video_llm,
    classify_all_videos,
    run_classification_stage
)

# Classify single video
import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

classification = classify_video_llm(
    video_id="7428596413707144481",
    transcript={"text": "...", "available": True},
    caption="Amazing nutrition tip!",
    hashtags=["nutrition", "health"],
    taxonomy=taxonomy,
    client=client
)

# Or run complete pipeline
stats = run_classification_stage(
    client_id="acme_corp",
    hashtag="nutrition"
)
```

### Utility Functions

```python
from ml_pipeline.stage2_content_analysis import (
    load_json,
    save_json,
    construct_path
)

# Load JSON with error handling
data = load_json("/path/to/file.json")

# Save JSON with atomic writes
save_json("/path/to/output.json", data)

# Construct standardized paths
path = construct_path(
    client_id="acme",
    hashtag="nutrition",
    file_type="taxonomy"
)
# Returns: /data/clients/acme/hashtags/nutrition/top_contrastive/content_taxonomies/nutrition_taxonomy.json
```

---

## Output Schemas

### Discovery Output (Stage 2.6)

```json
{
  "hashtag": "nutrition",
  "analysis_date": "2025-01-28T10:30:00Z",
  "sample_size": 50,
  "discovered_patterns": {
    "content_categories": [
      {
        "name": "recipe_tutorial",
        "frequency": 32,
        "percentage": 64.0,
        "examples": ["step by step", "here's how to make"],
        "representative_video_ids": ["123", "456"]
      }
    ],
    "hook_strategies": [...],
    "audience_pain_points": ["bloating", "low energy"],
    "trending_keywords": ["protein intake", "gut health"],
    "engagement_drivers": ["before after reveal"],
    "content_tactics": ["direct to camera"]
  }
}
```

### Classification Output (Stage 2.7)

```json
{
  "video_id": "7428596413707144481",
  "taxonomy_version": "stage2.6_output",
  "content_category": "recipe_tutorial",
  "hook_strategy": "question_hook",
  "pain_points": ["bloating", "low energy"],
  "keywords": ["protein intake"],
  "engagement_drivers": ["before after reveal"],
  "content_tactics": ["direct to camera"],
  "caption_analysis": {
    "hook_type": "statement",
    "cta_type": "follow",
    "brand_mention_present": false,
    "influencer_tag_present": false,
    "emoji_usage": "some",
    "caption_length": "short",
    "hashtag_count": 8,
    "hashtag_placement": "end"
  },
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

---

## Error Handling

The module implements comprehensive error handling:

- **E1-E12**: Catalog of error cases with recovery strategies
- **Retry Logic**: 3 attempts with exponential backoff for API failures
- **Fail-Fast**: Validation errors fail immediately with clear messages
- **Graceful Skip**: Missing transcripts logged but pipeline continues

### Example Error Messages

```
❌ Required input not found: /data/.../selection_manifest.json
This file should have been created by Stage 2.5.
Action: Verify Stage 2.5 completed successfully.

⏰ Discovery timeout (>120s). Retry 1/3 in 1s...

⚠️ Skipping video 123: Transcript not found
```

---

## Testing

### Run Unit Tests

```bash
cd /home/jorge/rumiaifinal
python -m pytest ml_pipeline/stage2_content_analysis/test_integration.py -v
```

### Manual Integration Test

1. Ensure Stage 2.5 completed (selection_manifest.json exists)
2. Set `ANTHROPIC_API_KEY` environment variable
3. Run discovery:
   ```python
   from ml_pipeline.stage2_content_analysis import run_discovery_stage
   run_discovery_stage("acme", "nutrition")
   ```
4. Manually curate `nutrition_raw_discovery.json` → `nutrition_taxonomy.json`
5. Run classification:
   ```python
   from ml_pipeline.stage2_content_analysis import run_classification_stage
   run_classification_stage("acme", "nutrition")
   ```
6. Verify 120 classification files created

---

## Performance

### Stage 2.6 (Discovery)

- **Model**: Claude 3.5 Sonnet
- **Input**: 50 transcripts (~5000 chars prompt)
- **Time**: ~45-60 seconds
- **Cost**: ~$0.50 per discovery
- **Output**: 1 raw discovery JSON

### Stage 2.7 (Classification)

- **Model**: Claude 3 Haiku
- **Input**: 1 video (transcript + caption + hashtags)
- **Time**: ~3-5 seconds per video
- **Cost**: ~$0.02 per video
- **Throughput**: 120 videos = ~6-10 minutes total
- **Output**: 120 classification JSONs

---

## Dependencies

### Input Dependencies (Stage 2.5)

- `selection_manifest.json` - Top 3 buckets + video selection
- Transcript files: `/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json`

### Output Feeds Into (Stage 7)

- LLM Report Generation - Uses classifications for contrastive analysis

---

## Logging

The module uses Python logging with clear status indicators:

```
[2025-01-28 10:30:00] [INFO] [stage2_content_analysis.discovery] ================================================================================
[2025-01-28 10:30:00] [INFO] [stage2_content_analysis.discovery] STAGE 2.6: CONTENT PATTERN DISCOVERY
[2025-01-28 10:30:00] [INFO] [stage2_content_analysis.discovery] Client: acme_corp, Hashtag: #nutrition, Sample Size: 50
[2025-01-28 10:30:00] [INFO] [stage2_content_analysis.discovery] Step 1/3: Validating inputs...
[2025-01-28 10:30:00] [INFO] [stage2_content_analysis.discovery] ✓ Input validation passed
[2025-01-28 10:30:01] [INFO] [stage2_content_analysis.discovery] Step 2/3: Sampling transcripts from top 3 buckets...
[2025-01-28 10:30:02] [INFO] [stage2_content_analysis.discovery] ✓ Sampled 50 transcripts
[2025-01-28 10:30:02] [INFO] [stage2_content_analysis.discovery] Step 3/3: Running LLM pattern discovery (Claude Sonnet)...
[2025-01-28 10:30:47] [INFO] [stage2_content_analysis.discovery] ✅ Discovery complete: .../nutrition_raw_discovery.json
[2025-01-28 10:30:47] [INFO] [stage2_content_analysis.discovery] 📝 Next: Manually curate and save to nutrition_taxonomy.json
```

---

## Troubleshooting

### Issue: `ANTHROPIC_API_KEY not set`

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Issue: `selection_manifest.json not found`

Verify Stage 2.5 completed successfully. The manifest should be at:
```
/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/selection_manifest.json
```

### Issue: `Insufficient transcripts sampled`

Check that Stage 2 (Whisper transcription) has processed videos. Transcripts should be at:
```
/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json
```

### Issue: `LLM returned invalid JSON`

This is usually transient. The module retries 3 times automatically. If it persists:
1. Check https://status.anthropic.com
2. Verify prompt formatting hasn't been modified
3. Report to Anthropic if recurring

---

## Implementation Notes

### Key Design Decisions

1. **Two-Stage Approach**: Discovery (Sonnet) → Classification (Haiku)
   - Rationale: Discovery requires reasoning (Sonnet), classification is repetitive (Haiku cheaper)

2. **Manual Curation Step**: Raw discovery requires human review before classification
   - Rationale: Ensures taxonomy quality, prevents downstream errors

3. **Fail-Fast Validation**: All inputs validated upfront
   - Rationale: Catch errors early, save API costs

4. **Atomic Writes**: Save JSON to temp file, then rename
   - Rationale: Prevents partial writes if process crashes

5. **Stratified Sampling**: Even distribution across top 3 buckets
   - Rationale: Ensures patterns representative of all durations

---

## Version History

- **v1.0.0** (2025-01-28): Initial implementation
  - Stage 2.6: Pattern Discovery
  - Stage 2.7: Video Classification
  - Comprehensive validation and error handling
  - Integration tests

---

## Contact

**Project**: RumiAI - TikTok Video Analysis
**Company**: Tumi Labs
**Source Documentation**: ContentAnalysisCHILDTI.md
