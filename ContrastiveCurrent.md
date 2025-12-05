# ContrastiveCurrent.md - Running Contrastive Mode on Existing Data

**Created**: 2025-12-04
**Purpose**: Instructions for running contrastive analysis on @shopbyjake and @ayunonutricion
**Audience**: LLM agents implementing this task after context compaction
**Working Directory**: `/home/jorge/rumiaifinal`

---

## Context From Previous Session

### What Was Accomplished
1. **ReportFix.md implemented**: Fixed duplicate classification entries in reports
   - Modified 5 files to normalize `engagement_drivers`, `content_tactics`, etc.
   - Files: `classification.py`, `extract_competitor_data.py`, `extract_client_data.py`, `extract_creator_data.py`, `extract_multi_competitor_data.py`
   - Fix verified on @shopbyjake report - no duplicates

2. **Contrastive mode research completed**: Determined both competitors have all data needed

### Why Contrastive Mode Is Needed
- Current runs used `--selection-strategy top` which only analyzes top performers
- `winning_formulas.json` shows `"creative_reports": []` (empty) because RF analysis requires contrastive mode
- Contrastive mode compares top 80% vs bottom 20% to find what makes winners different

---

## Executive Summary

**Both competitors have ALL data needed. No video re-downloads required.**

The `selected_videos.json` files already contain ALL videos from each creator (sorted by engagement). The "bottom 20%" are simply the lowest-engagement videos already in the data - they just weren't labeled as bottom performers.

| Competitor | Videos in Buckets | All Transcripts Present? | Ready? |
|------------|-------------------|--------------------------|--------|
| @shopbyjake | 100 + 100 + 24 = 224 | YES (verified) | YES |
| @ayunonutricion | 100 + 92 + 60 = 252 | YES (verified) | YES |

---

## Commands to Execute

### Prerequisites
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
```

### Run Contrastive Analysis

```bash
# @shopbyjake (run first)
python rumiai_ml_batch.py \
  --client statesidegrowers \
  --target @shopbyjake \
  --analysis-type competitor \
  --selection-strategy contrastive \
  --video-count 100 \
  --date-filter last_90_days \
  --country-code US \
  --report-type single \
  --report-audience client

# @ayunonutricion (run second or in parallel)
python rumiai_ml_batch.py \
  --client statesidegrowers \
  --target @ayunonutricion \
  --analysis-type competitor \
  --selection-strategy contrastive \
  --video-count 100 \
  --date-filter last_90_days \
  --country-code US \
  --report-type single \
  --report-audience client
```

**Key difference from original runs**: `--selection-strategy contrastive` instead of `top`

---

## Expected Behavior

### What Should Happen
1. **Stage 1**: May re-scrape TikTok metadata (~30s) or skip if cached
2. **Stage 2**: SKIP video downloads (videos already exist)
3. **Stage 2**: SKIP transcription (whisper files exist in `/home/jorge/rumiaifinal/speech_transcriptions/`)
4. **Stage 2.5-2.7**: Run content analysis - videos labeled as `top` (80%) or `bottom` (20%)
5. **Stage 7**: Generate `winning_formulas.json` WITH `creative_reports` populated
6. **Stage 8**: Generate reports

### Output Location
New directory created: `top_contrastive/` (separate from existing `top_top/`)
```
data/clients/statesidegrowers/competitors/shopbyjake/top_contrastive/
data/clients/statesidegrowers/competitors/ayunonutricion/top_contrastive/
```

---

## Handling Taxonomy Curation Pause

**IMPORTANT**: Stage 2.6 may pause and require manual taxonomy curation.

If you see exit code 2 or message about taxonomy curation:

### Option 1: Copy Existing Taxonomy (Recommended)
```bash
# For shopbyjake
mkdir -p data/clients/statesidegrowers/competitors/shopbyjake/top_contrastive/content_taxonomies/
cp data/clients/statesidegrowers/competitors/shopbyjake/top_top/content_taxonomies/shopbyjake_taxonomy.json \
   data/clients/statesidegrowers/competitors/shopbyjake/top_contrastive/content_taxonomies/

# Create state file marking taxonomy as curated
echo '{"discovery_complete": true, "taxonomy_curated": true, "taxonomy_version": "1.0"}' > \
   data/clients/statesidegrowers/competitors/shopbyjake/top_contrastive/.content_analysis_state.json

# Re-run the command
```

### Option 2: Manual Curation
1. Review `{competitor}_raw_discovery.json`
2. Edit `{competitor}_taxonomy.json`
3. Set `taxonomy_curated: true` in `.content_analysis_state.json`
4. Re-run command

---

## Verification After Running

### Check 1: Directory Created
```bash
ls -la data/clients/statesidegrowers/competitors/shopbyjake/
# Should show both top_top/ and top_contrastive/
```

### Check 2: Bottom Performers Exist
```bash
source venv/bin/activate && python3 << 'EOF'
import json
import glob

for competitor in ['shopbyjake', 'ayunonutricion']:
    files = glob.glob(f'data/clients/statesidegrowers/competitors/{competitor}/top_contrastive/content_analysis/validated/bucket_*/*_content.json')
    bottom_count = 0
    for f in files:
        with open(f) as fp:
            if json.load(fp).get('performer_type') == 'bottom':
                bottom_count += 1
    print(f"{competitor}: {bottom_count} bottom performers, {len(files) - bottom_count} top performers")
EOF
```

### Check 3: Winning Formulas Populated
```bash
source venv/bin/activate && python3 << 'EOF'
import json
for competitor in ['shopbyjake', 'ayunonutricion']:
    try:
        with open(f'data/clients/statesidegrowers/competitors/{competitor}/top_contrastive/winning_formulas.json') as f:
            d = json.load(f)
            cr = d.get('creative_reports', [])
            print(f"{competitor}: creative_reports has {len(cr)} entries")
    except FileNotFoundError:
        print(f"{competitor}: winning_formulas.json not found yet")
EOF
# Should be > 0 (was 0 with top strategy)
```

---

## What Contrastive Strategy Does

### Video Selection (per bucket)
| Strategy | Selection | Result |
|----------|-----------|--------|
| `top` | Top N by engagement | All videos labeled `is_top_performer: true` |
| `contrastive` | Top 80% + Bottom 20% | 80% labeled `top`, 20% labeled `bottom` |

### Why This Matters
- `top` strategy: Can only say "here's what top performers do"
- `contrastive` strategy: Can say "top performers do X, bottom performers do Y, the difference is Z"

The `creative_reports` in `winning_formulas.json` contains this comparative analysis.

---

## Existing Data Locations (Reference)

### @shopbyjake
```
/home/jorge/rumiaifinal/data/clients/statesidegrowers/competitors/shopbyjake/top_top/
├── buckets/
│   ├── bucket_33-60s/selected_videos.json  # 100 videos, sorted by engagement
│   ├── bucket_60-90s/selected_videos.json  # 100 videos
│   └── bucket_90-120s/selected_videos.json # 24 videos
├── content_taxonomies/shopbyjake_taxonomy.json  # Can copy to top_contrastive/
└── .content_analysis_state.json
```

### @ayunonutricion
```
/home/jorge/rumiaifinal/data/clients/statesidegrowers/competitors/ayunonutricion/top_top/
├── buckets/
│   ├── bucket_33-60s/selected_videos.json  # 100 videos
│   ├── bucket_60-90s/selected_videos.json  # 92 videos
│   └── bucket_90-120s/selected_videos.json # 60 videos
├── content_taxonomies/ayunonutricion_taxonomy.json
└── .content_analysis_state.json
```

### Global Transcripts
```
/home/jorge/rumiaifinal/speech_transcriptions/
└── {video_id}_whisper.json  # 2,812+ transcripts (shared across all analyses)
```

---

## Implementation Checklist

- [ ] Activate venv: `source venv/bin/activate`
- [ ] Run contrastive command for @shopbyjake
- [ ] Handle taxonomy curation pause if triggered (copy from top_top/)
- [ ] Run contrastive command for @ayunonutricion
- [ ] Handle taxonomy curation pause if triggered
- [ ] Verify `top_contrastive/` directories created
- [ ] Verify bottom performers labeled in content_analysis files
- [ ] Verify `winning_formulas.json` has `creative_reports` populated (> 0 entries)

---

## Related Documentation

| File | Purpose |
|------|---------|
| `ReportFix.md` | Duplicate classification fix (implemented this session) |
| `ContrastiveMode.md` | Original research document |
| `documentation_migration/rumiaibatch/STAGE_7_IMPL.md` | Winning formula generation details |
