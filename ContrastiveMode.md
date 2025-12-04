# ContrastiveMode.md - Research Plan for Running Contrastive Mode

**Created**: 2025-12-04
**Goal**: Determine if contrastive mode can be run on @shopbyjake and @ayunonutricion without re-downloading videos
**Current State**: Both competitors have `top_top` analysis directories with processed videos

---

## Background

### Current Setup
- `--selection-strategy top` was used (creates `top_top` directory)
- `winning_formulas.json` shows `"creative_reports": []` (empty) because RF analysis requires contrastive mode
- We want `--selection-strategy contrastive` (creates `top_contrastive` directory)

### Key Question
Can we reuse Stage 2 video processing outputs (ML insights, transcripts) when running contrastive mode?

---

## Discovery Tasks

### Task 1: Understand Directory Structure Difference

**Files to read**:
```
/home/jorge/rumiaifinal/foundation/paths.py (lines 51-77)
```

**Question**: What directory structure does `top_contrastive` vs `top_top` create?

**Expected finding**: Directory name is `{mode}_{strategy}` so contrastive creates separate directory.

---

### Task 2: Identify Reusable Outputs

**Files to check**:
```
/home/jorge/rumiaifinal/speech_transcriptions/  # Global, not per-analysis
/home/jorge/rumiaifinal/ml_insights/            # Global or per-analysis?
/home/jorge/rumiaifinal/temp/                   # Downloaded videos location?
```

**Question**: Which Stage 2 outputs are stored globally vs per-analysis-directory?

**Grep to run**:
```bash
grep -r "speech_transcriptions" rumiai_ml_batch.py ml_pipeline/ --include="*.py" -l
grep -r "ml_insights" rumiai_ml_batch.py ml_pipeline/ --include="*.py" -l
```

---

### Task 3: Check Video Selection Difference

**Files to read**:
```
/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/video_selector.py (lines 1-100)
/home/jorge/rumiaifinal/documentation_migration/rumiaibatch/STAGE_1_IMPL.md (search "contrastive")
```

**Question**: Does contrastive mode select DIFFERENT videos or just LABEL them differently?

**Key distinction**:
- If same videos, different labels → Can reuse all processing
- If different videos selected → Must re-download and re-process

---

### Task 4: Check Stage 2 Processing Reuse Logic

**Files to read**:
```
/home/jorge/rumiaifinal/rumiai_ml_batch.py (lines 700-850, Stage 2 section)
/home/jorge/rumiaifinal/ml_pipeline/stage2_processing/main.py (checkpoint logic)
```

**Question**: Does Stage 2 check for existing transcripts/insights before processing?

**Grep to run**:
```bash
grep -n "speech_transcriptions" ml_pipeline/stage2_processing/*.py
grep -n "already.*exist\|skip.*exist\|reuse" ml_pipeline/stage2_processing/*.py
```

---

### Task 5: Check Contrastive vs Top Video Count Requirements

**File to read**:
```
/home/jorge/rumiaifinal/foundation/constants.py (lines 41-57)
```

**Expected content**:
```python
DEFAULT_SELECTION_STRATEGIES = {
    "hashtag": "contrastive",
    "competitor": "contrastive",
    "creator": "top"
}
MIN_RECOMMENDED_N = {
    "contrastive": 50,  # Ensures 10 bottom performers (20%)
    "top": 30
}
```

**Question**: Do we have enough videos (100) for contrastive mode (needs 50 minimum)?

---

### Task 6: Check winning_formulas.json Generation Trigger

**Files to read**:
```
/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/ (find relevant file)
/home/jorge/rumiaifinal/documentation_migration/rumiaibatch/STAGE_7_IMPL.md
```

**Grep to run**:
```bash
grep -rn "winning_formulas\|creative_reports\|RF analysis\|contrastive" ml_pipeline/stage7_llm_analysis/
```

**Question**: What exactly triggers `creative_reports` population vs empty array?

---

## Expected Outcomes

### Scenario A: Full Reuse Possible
If transcripts and ML insights are stored globally, contrastive run would:
1. Create new `top_contrastive` directory
2. Stage 1: Re-scrape metadata (fast, ~30s)
3. Stage 2: Skip video download, reuse existing transcripts/insights
4. Stage 3+: Run fresh analysis with contrastive labels

### Scenario B: Partial Reuse
If some outputs are per-directory:
1. Transcripts reusable (global)
2. ML insights may need regeneration
3. Video downloads can be skipped if in temp/

### Scenario C: No Reuse
If everything is per-analysis-directory, full re-run required.

---

## Commands to Execute

After discovery, if reuse is possible, the command would be:

```bash
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
```

Key change: `--selection-strategy contrastive` instead of `top`

---

## Files Summary for Discovery

| Priority | File | Lines | Purpose |
|----------|------|-------|---------|
| 1 | `foundation/paths.py` | 51-77 | Directory naming logic |
| 2 | `foundation/constants.py` | 41-57 | Strategy defaults and minimums |
| 3 | `ml_pipeline/stage1_discovery/video_selector.py` | 1-100 | Selection logic difference |
| 4 | `rumiai_ml_batch.py` | 700-850 | Stage 2 orchestration |
| 5 | `STAGE_7_IMPL.md` | Search "RF\|contrastive" | Formula generation trigger |

---

## Implementation Checklist

- [ ] Run Task 1-6 discovery
- [ ] Document which outputs are reusable
- [ ] Determine if video re-download required
- [ ] Test contrastive run on one competitor
- [ ] Verify `winning_formulas.json` populates with `creative_reports`
