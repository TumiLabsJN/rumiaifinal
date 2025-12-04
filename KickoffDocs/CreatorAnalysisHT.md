# Creator Analysis - Handoff Template

## Purpose
This document guides an LLM to run a **creator-focused hashtag analysis** on the RumiAI pipeline. The pipeline scrapes videos from a hashtag/niche, processes them through ML services, and generates actionable creative intelligence reports designed **for content creators** to improve their content strategy.

**Key Distinction**: This is NOT analyzing a specific creator's profile (@handle). Instead, it analyzes a **hashtag/niche** and generates reports tailored for creators who want to succeed in that niche.

---

## When You Receive This Task

The user will provide:
- **Hashtag/Niche**: TikTok hashtag (e.g., `#guthealth`, `#supplements`, `#wellness`)
- **Client name**: Project identifier (e.g., `wellness_creator_jan`)

### Your First Response Should Be:

1. **Confirm standard settings**:
   - Video count: `100` per bucket (contrastive strategy needs more data)
   - Date filter: `last_270_days` (9 months)
   - Selection strategy: `contrastive` (compares top vs bottom performers)
   - Country: `US`

2. **Check if client exists**:
   ```bash
   ls /home/jorge/rumiaifinal/data/clients/
   ```
   - If client folder exists, confirm: "I see this client exists, is this the same project?"
   - If new client, confirm: "This will create a new client folder."

3. **Ask for confirmation**:
   "Should we run with standard settings, or would you like to adjust anything?"

---

## Standard Settings

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--video-count` | `100` | Videos per winning bucket |
| `--date-filter` | `last_270_days` | 9-month lookback window |
| `--selection-strategy` | `contrastive` | Top vs Bottom comparison (key for creator insights) |
| `--country-code` | `US` | Geographic filter |
| `--analysis-type` | `hashtag` | Hashtag-based analysis |
| `--report-type` | `single` | Single hashtag report |
| `--report-audience` | `creator` | Report formatted for content creators |

### Why Contrastive Strategy?

For creator reports, we use `contrastive` (not `top`) because:
- Compares **top performers** vs **bottom performers** in the same niche
- Reveals what separates viral content from underperforming content
- Provides actionable "do this, not that" insights for creators

---

## Command to Run

Once confirmed, activate the virtual environment and run:

```bash
cd /home/jorge/rumiaifinal && source venv/bin/activate && python rumiai_ml_batch.py \
  --client [CLIENT_NAME] \
  --target "#[HASHTAG]" \
  --analysis-type hashtag \
  --selection-strategy contrastive \
  --video-count 100 \
  --date-filter last_270_days \
  --country-code US \
  --report-type single \
  --report-audience creator
```

**Important**:
- The `--target` parameter MUST include the `#` symbol and be quoted (e.g., `"#guthealth"`)
- Client name will be lowercased automatically
- Run in background for long-running pipelines

---

## Pipeline Stages

The pipeline runs through these stages automatically:

| Stage | Name | Duration | Notes |
|-------|------|----------|-------|
| 0 | Foundation | <1 min | Creates config and directories |
| 1 | Video Discovery | 2-5 min | Scrapes TikTok via Apify |
| 2 | Video Processing | 2-3 min/video | ML analysis (YOLO, Whisper, etc.) |
| 2.5 | File Organization | <1 min | Organizes outputs by bucket |
| 2.5.1 | Transcript Validation | <1 min | Filters music-only videos |
| 2.6 | Taxonomy Discovery | 1-2 min | **PAUSES FOR MANUAL CURATION** |
| 2.7 | Classification | 5-15 min | Classifies all videos |
| 3-7 | ML & Reports | 10-20 min | Training and report generation |

### Manual Curation Pause (Stage 2.6)

After Stage 2.6, the pipeline **exits with code 2** and requires manual action:

1. Review the raw taxonomy at:
   ```
   /home/jorge/rumiaifinal/data/clients/{client}/hashtags/{hashtag}/{mode}_{strategy}/content_taxonomies/{hashtag}_raw_discovery.json
   ```

2. Edit the curated taxonomy:
   ```
   /home/jorge/rumiaifinal/data/clients/{client}/hashtags/{hashtag}/{mode}_{strategy}/content_taxonomies/{hashtag}_taxonomy.json
   ```

3. Set `taxonomy_curated: true` in:
   ```
   /home/jorge/rumiaifinal/data/clients/{client}/hashtags/{hashtag}/{mode}_{strategy}/.content_analysis_state.json
   ```

4. Re-run the same command to continue from Stage 2.7

---

## Report Generation (Stage 8)

After pipeline completes, generate the creator report:

```bash
python extract_creator_data.py \
  --client [CLIENT_NAME] \
  --hashtag [HASHTAG] \
  --mode top \
  --strategy contrastive
```

### Creator Report Output

| Output | Description |
|--------|-------------|
| **Excel file** | 3 tabs (one per winning bucket) |
| **QR codes** | 12 total (4 per bucket: 2 top + 2 bottom examples) |
| **Content** | Actionable creative formulas, hooks, closings, what works vs what doesn't |

---

## Output Location

Final outputs will be in:
```
/home/jorge/rumiaifinal/data/clients/{client}/hashtags/{hashtag}/top_contrastive/
├── winner_analysis.json           # Bucket distribution
├── content_taxonomies/            # Taxonomy files
├── buckets/                       # Per-bucket video data
│   └── bucket_{duration}/
│       ├── selected_videos.json
│       └── ml_analysis/
├── reports/
│   └── creator/
│       ├── {hashtag}_creator_report.xlsx
│       └── qr_codes/
└── checkpoints/                   # Resume points
```

---

## Checkpoint System

The pipeline creates checkpoints after each stage. If it fails or is interrupted:
- Simply re-run the same command
- It will automatically skip completed stages and resume from the last checkpoint

---

## Common Issues

### Issue: "Only X videos scraped"
- **Cause**: Niche hashtag with limited content
- **Solution**: Try a broader hashtag or extend to `last_365_days`

### Issue: "Insufficient valid transcripts"
- **Cause**: Many videos in this niche are music-only
- **Solution**: Pipeline now warns but proceeds. Taxonomy quality may be limited.

### Issue: "Bucket has only X top performers"
- **Cause**: Small dataset spread across buckets
- **Solution**: Pipeline now warns but proceeds with available data.

---

## Creator Report vs Client Report vs Competitor Report

| Report Type | Target | Audience | Key Insights |
|-------------|--------|----------|--------------|
| **Creator** | #hashtag | Content creators | "What makes content go viral in this niche" |
| **Client** | #hashtag | Brand executives | "Market overview and trends summary" |
| **Competitor** | @handle | Brand intel team | "What a specific competitor is doing" |

---

## Example Workflow

**User says**: "I want to help a wellness creator understand what content performs best in the gut health niche"

**You respond**:
1. "I'll run a creator analysis on #guthealth. Standard settings are 100 videos/bucket, 270 days, contrastive strategy. Does this work?"
2. After confirmation, run the pipeline
3. At Stage 2.6, curate the taxonomy (or auto-curate if appropriate)
4. Generate the creator report with `extract_creator_data.py`
5. Share the Excel location with the user

---

## Reference Documentation

For deeper understanding, read:
- `/home/jorge/rumiaifinal/documentation_migration/rumiaibatch/STAGE_2.6_2.7_IMPL.md` - Content analysis details
- `/home/jorge/rumiaifinal/documentation_migration/rumiaibatch/STAGE_8_IMPL.md` - Report generation (Section 6: Creator Report)
