# Single Competitor Analysis - Handoff Template

## Purpose
This document guides an LLM to run a single competitor analysis on the RumiAI pipeline. The pipeline scrapes a TikTok competitor's videos, processes them through ML services, and generates content taxonomy and creative intelligence reports.

---

## When You Receive This Task

The user will provide:
- **Competitor handle**: TikTok handle (e.g., `@balanceofnature`)
- **Client name**: Project identifier (e.g., `Statesidegrowers`)

### Your First Response Should Be:

1. **Confirm standard settings**:
   - Video count: `80` per bucket
   - Date filter: `last_270_days` (9 months)
   - Selection strategy: `top` (for competitor analysis)
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
| `--video-count` | `80` | Videos per winning bucket |
| `--date-filter` | `last_270_days` | 9-month lookback window |
| `--selection-strategy` | `top` | Top performers only (standard for competitor analysis) |
| `--country-code` | `US` | Geographic filter |
| `--analysis-type` | `competitor` | Handle-based analysis |
| `--report-type` | `single` | Single competitor report |
| `--report-audience` | `client` | Report format |

---

## Command to Run

Once confirmed, activate the virtual environment and run:

```bash
cd /home/jorge/rumiaifinal && source venv/bin/activate && python rumiai_ml_batch.py \
  --client [CLIENT_NAME] \
  --target @[COMPETITOR_HANDLE] \
  --analysis-type competitor \
  --selection-strategy top \
  --video-count 80 \
  --date-filter last_270_days \
  --country-code US \
  --report-type single \
  --report-audience client
```

**Important**:
- The `--target` parameter MUST include the `@` symbol (e.g., `@balanceofnature`)
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
   /home/jorge/rumiaifinal/data/clients/{client}/competitors/{competitor}/top_top/content_taxonomies/{competitor}_raw_discovery.json
   ```

2. Edit the curated taxonomy:
   ```
   /home/jorge/rumiaifinal/data/clients/{client}/competitors/{competitor}/top_top/content_taxonomies/{competitor}_taxonomy.json
   ```

3. Set `taxonomy_curated: true` in:
   ```
   /home/jorge/rumiaifinal/data/clients/{client}/competitors/{competitor}/top_top/.content_analysis_state.json
   ```

4. Re-run the same command to continue from Stage 2.7

---

## Checkpoint System

The pipeline creates checkpoints after each stage. If it fails or is interrupted:
- Simply re-run the same command
- It will automatically skip completed stages and resume from the last checkpoint

Checkpoints are stored in:
```
/home/jorge/rumiaifinal/data/clients/{client}/competitors/{competitor}/top_top/checkpoints/
```

---

## Common Issues

### Issue: "Only X videos scraped"
- **Cause**: Competitor doesn't post frequently
- **Solution**: Proceed if >30 videos, or try `last_365_days` for more data

### Issue: "Insufficient valid transcripts"
- **Cause**: Most videos are music-only (no speech)
- **Solution**: Pipeline now warns but proceeds. Taxonomy quality may be limited.

### Issue: "Bucket has only X top performers"
- **Cause**: Small dataset spread across buckets
- **Solution**: Pipeline now warns but proceeds with available data.

---

## Output Location

Final outputs will be in:
```
/home/jorge/rumiaifinal/data/clients/{client}/competitors/{competitor}/top_top/
├── winner_analysis.json           # Bucket distribution
├── content_taxonomies/            # Taxonomy files
├── buckets/                       # Per-bucket video data
│   └── bucket_{duration}/
│       ├── selected_videos.json
│       └── ml_analysis/
└── checkpoints/                   # Resume points
```

---

## Reference Documentation

For deeper understanding, read:
- `/home/jorge/rumiaifinal/E2ETest_Wellness_Rollo.md` - Full test examples
- `/home/jorge/rumiaifinal/documentation_migration/rumiaibatch/STAGE_2.6_2.7_IMPL.md` - Content analysis details
- `/home/jorge/rumiaifinal/documentation_migration/rumiaibatch/STAGE_8_IMPL.md` - Report generation
