# End-to-End Test: Wellness Cluster (Client: Rollo)

## 📋 Test Overview

**Test ID:** E2E-WELLNESS-001
**Test Type:** Full ML Pipeline Validation
**Client:** Rollo
**Cluster:** wellness (4 hashtags)
**Objective:** Validate complete pipeline from video discovery through ML model training and report generation

---

## 🎯 Test Objectives

This E2E test validates:

1. **Cluster Scraping:** Multi-hashtag scraping with deduplication and analytics
2. **Video Processing:** ML service execution across all winning buckets
3. **Content Analysis:** Pattern discovery (Stage 2.6) and video classification (Stage 2.7)
4. **Feature Engineering:** Temporal feature aggregation and transformation
5. **ML Training:** Random Forest + K-Means model training per bucket
6. **Report Generation:** LLM-powered analysis and PDF report creation

**Success Criteria:** All pipeline stages complete successfully with expected output files generated

---

## 🔧 Prerequisites

### Required Environment Variables
```bash
export APIFY_API_KEY="your_apify_key"
export ANTHROPIC_API_KEY="your_claude_key"
```

### Required Tools
- Python 3.8+
- RumiAI dependencies installed
- Apify account with credits
- Claude API access

### Initial State
- Clean test environment (no existing Rollo client data)
- Cluster config exists: `/config/hashtag_clusters/wellness.json`

---

## 📦 Test Configuration

### Cluster Configuration
**File:** `/home/jorge/rumiaifinal/config/hashtag_clusters/wellness.json`

```json
{
  "cluster_id": "wellness",
  "description": "Wellness cluster focused on supplement-related content - 4 hashtags for semantic coverage",
  "primary_hashtag": "#wellness",
  "variant_hashtags": [
    "#wellnesssupplements",
    "#healthandwellness",
    "#wellnessjourney"
  ],
  "scrape_config": {
    "runs_per_hashtag": 2,
    "delay_between_runs_ms": 120000,
    "results_per_page": 600
  }
}
```

### CLI Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--client` | Rollo | Test client identifier |
| `--target` | wellness | Cluster name (not hashtag) |
| `--analysis-type` | hashtag | Cluster analysis mode |
| `--selection-strategy` | contrastive | Top 80% + Bottom 20% per bucket |
| `--video-count` | 80 | Videos per winning bucket (64 top + 16 bottom) |
| `--date-filter` | last_150_days | 5-month window for recency |
| `--country-code` | US | Geographic filter |
| `--report-type` | single | Single hashtag analysis |
| `--report-audience` | client | Report format for brand/client |

**Expected Scraping:**
- 4 hashtags × 2 runs × 600 results = 8 scrapes
- ~4,800 videos before deduplication
- ~1,500-2,000 unique videos after deduplication
- ~800-1,200 videos after 150-day filter

---

## 🚀 Test Execution

### Command

```bash
cd /home/jorge/rumiaifinal

python rumiai_ml_batch.py \
  --client Rollo \
  --target wellness \
  --analysis-type hashtag \
  --selection-strategy contrastive \
  --video-count 80 \
  --date-filter last_150_days \
  --country-code US \
  --report-type single \
  --report-audience client
```

### Expected Runtime

| Stage | Expected Duration | Notes |
|-------|------------------|-------|
| Stage 0 | <5 seconds | Foundation setup |
| Stage 1 | 30-45 minutes | Cluster scraping (8 scrapes × 2 min delay) |
| Stage 2 | 2-4 hours | Video processing (depends on video count/duration) |
| Stage 2.5 | <10 seconds | File organization |
| Stage 2.6 | 3-5 minutes | Pattern discovery (LLM call) |
| **⏸️ PAUSE** | **Manual** | **Taxonomy curation (1-3 hours)** |
| Stage 2.7 | 15-30 minutes | Video classification (LLM calls) |
| Stage 3 | 10-20 minutes | Feature aggregation |
| Stage 4 | 5-10 minutes | Feature transformation |
| Stage 5 | 10-15 minutes | ML model training |
| Stage 6 | 5-10 minutes | ML analysis generation |
| Stage 7 | 20-30 minutes | LLM report generation |
| **TOTAL** | **4-6 hours** | **Excluding manual curation time** |

---

## 📊 Pipeline Flow & Expected Behavior

### Stage 0: Foundation
**Action:** Creates directory structure and saves configuration

**Expected Output:**
```
✓ Created directory structure: data/clients/Rollo/hashtag/wellness/top_contrastive
✓ Saved configuration: data/clients/Rollo/hashtag/wellness/top_contrastive/config.json
```

### Stage 1: Video Discovery & Selection
**Action:** Cluster scraping with deduplication and winner analysis

**Sub-stages:**
1. **1.1a:** Cluster scraping (8 scrapes)
2. **1.1b:** Deduplication with provenance tracking
3. **1.1c:** Hashtag validation (false positive removal)
4. **1.1d:** Save cluster analytics
5. **1.2:** Date filtering (last 150 days)
6. **1.3:** Duration bucketing (8 buckets)
7. **1.4:** Winner analysis (select top 3 buckets)
8. **1.5:** Video selection (contrastive: 64 top + 16 bottom per bucket)

**Expected Output:**
```
✓ Stage 1: Video Discovery - COMPLETE
  Scraping: 8/8 scrapes successful
  Deduplication: 4,800 → 1,850 unique videos (61.5% retained)
  Date filter: 1,850 → 1,120 videos (last 150 days)
  Winner buckets: bucket_13-18s, bucket_18-33s, bucket_33-60s
  Selected: 240 videos (80 per bucket)
```

**Key Output Files:**
- `cluster_analytics.json` - Scraping health metrics
- `winner_analysis.json` - Bucket distribution and winners
- `buckets/bucket_{duration}/selected_videos.json` - Selected videos per bucket

### Stage 2: Video Processing
**Action:** Process selected videos through RumiAI ML pipeline

**Per Bucket Processing:**
- Downloads video from TikTok
- Runs 9 ML services (YOLO, Whisper, MediaPipe, OCR, Scene Detection, FEAT, etc.)
- Generates `temporal_windows_updated.json` with 350+ features
- Checkpoint/resume enabled

**Expected Output:**
```
✓ Stage 2: Video Processing - COMPLETE
  bucket_13-18s: 80/80 videos processed (0 failed)
  bucket_18-33s: 80/80 videos processed (0 failed)
  bucket_33-60s: 80/80 videos processed (0 failed)
  Summary: 240/240 videos processed
```

**Key Output Files:**
- `insights/{video_id}_temporal_windows_updated.json` (240 files)
- `.stage2_checkpoint.json` (per bucket)

### Stage 2.5: File Organization
**Action:** Organize temporal_windows files into bucket directories

**Expected Output:**
```
✓ Stage 2.5: File Organization - COMPLETE
  Organized 240 temporal_windows files
  Skipped 0 files (already organized)
  Missing 0 files
```

**Key Output Files:**
- `buckets/bucket_{duration}/insights/{video_id}_temporal_windows_updated.json`
- `buckets/bucket_{duration}/selection_manifest.json`

### Stage 2.6: Pattern Discovery (⏸️ MANUAL PAUSE)
**Action:** LLM discovers content patterns from sample videos

**Expected Output:**
```
--- Stage 2.6: Pattern Discovery (One-Time Setup) ---
Discovering content patterns from sample transcripts...

✅ Stage 2.6 Discovery Complete!
================================================================================

📋 NEXT STEP: Manual curation required (estimated time: 1-3 hours depending on complexity)

After curation, re-run this command to continue:
  python rumiai_ml_batch.py --client Rollo --target wellness
```

**Raw Discovery File Created:**
- `content_taxonomies/wellness_raw_discovery.json`

**Contains:**
```json
{
  "hashtag": "wellness",
  "analysis_date": "2025-10-22T...",
  "sample_size": 50,
  "discovered_patterns": {
    "content_categories": [...],    // 5-10 categories with frequency, examples
    "hook_strategies": [...],        // 5-10 hook types
    "audience_pain_points": [...],  // 10-15 pain points
    "trending_keywords": [...],      // 15-25 keywords
    "engagement_drivers": [...],     // 10-15 drivers
    "content_tactics": [...]         // 10-15 tactics
  }
}
```

**⏸️ PIPELINE PAUSED - Exit Code 2**

---

## 📝 Manual Curation Instructions (Stage 2.6)

### Step 1: Open Raw Discovery File

```bash
cd /home/jorge/rumiaifinal
cat data/clients/Rollo/hashtag/wellness/top_contrastive/content_taxonomies/wellness_raw_discovery.json
```

### Step 2: Review and Curate Patterns

**Curation Guidelines:**

1. **Remove low-frequency patterns** (<10% occurrence)
2. **Merge similar categories**
   - Example: "supplement_review" + "product_demo" → "supplement_showcase"
3. **Ensure snake_case naming**
   - ✅ Good: `wellness_routine`, `morning_supplements`
   - ❌ Bad: `Wellness Routine`, `morning supplements`
4. **Add clear definitions** (minimum 10 characters)
   - Transform `examples` into concise `definition`
   - Example:
     ```json
     // Raw discovery:
     "examples": ["I take vitamin D every morning", "My supplement routine"]

     // Curated definition:
     "definition": "Videos showcasing daily supplement routines and wellness practices"
     ```
5. **Remove duplicates**

### Step 3: Create Curated Taxonomy File

**Required Structure:**

```json
{
  "content_categories": [
    {
      "name": "supplement_routine",
      "definition": "Videos showcasing daily supplement routines and wellness practices"
    },
    {
      "name": "health_education",
      "definition": "Educational content explaining wellness benefits, deficiency symptoms, or health science"
    }
  ],
  "hook_strategies": [
    {
      "name": "relatable_symptom",
      "definition": "Opens with common health symptoms or wellness challenges viewers relate to"
    }
  ],
  "audience_pain_points": [
    "chronic fatigue",
    "stress management",
    "poor sleep quality"
  ],
  "trending_keywords": [
    "wellness journey",
    "holistic health",
    "natural supplements"
  ],
  "engagement_drivers": [
    "personal transformation story",
    "expert authority"
  ],
  "content_tactics": [
    "morning routine walkthrough",
    "product demonstration"
  ]
}
```

**Save as:**
```bash
data/clients/Rollo/hashtag/wellness/top_contrastive/content_taxonomies/wellness_taxonomy.json
```

### Step 4: Validate Taxonomy (Optional but Recommended)

```bash
python -c "
from ml_pipeline.stage2_content_analysis.taxonomy_validation import validate_curated_taxonomy
validate_curated_taxonomy('data/clients/Rollo/hashtag/wellness/top_contrastive/content_taxonomies/wellness_taxonomy.json')
print('✓ Taxonomy validation passed')
"
```

### Step 5: Resume Pipeline

```bash
# Re-run the SAME command
python rumiai_ml_batch.py \
  --client Rollo \
  --target wellness \
  --analysis-type hashtag \
  --selection-strategy contrastive \
  --video-count 80 \
  --date-filter last_150_days \
  --country-code US \
  --report-type single \
  --report-audience client
```

**Expected Behavior:**
- Stages 0-2.5 will check checkpoints and skip (fast)
- Stage 2.6 will detect taxonomy exists and skip
- Stage 2.7 will run (classification)

---

### Stage 2.7: Video Classification
**Action:** Classify all videos using curated taxonomy

**Expected Output:**
```
--- Stage 2.7: Video Classification ---
Validating taxonomy...
✓ Taxonomy validation passed
Classification mode: sequential
Classifying videos across 3 buckets...

✓ Stage 2.7: Classified 240/240 videos in 1,234.56s
```

**Key Output Files:**
- `buckets/bucket_{duration}/classification/{video_id}_classification.json` (240 files)
- `.classification_checkpoint.json` (per bucket)

### Stage 3: Feature Aggregation
**Action:** Aggregate temporal features + classifications into bucket-level CSVs

**Expected Output:**
```
✓ Stage 3: Feature Aggregation - COMPLETE
  bucket_13-18s: 80 videos aggregated
  bucket_18-33s: 80 videos aggregated
  bucket_33-60s: 80 videos aggregated
```

**Key Output Files:**
- `buckets/bucket_{duration}/aggregated_features.csv`
- `buckets/bucket_{duration}/checkpoints/stage_3_checkpoint.json`

### Stage 4: Feature Transformation
**Action:** Transform features for ML training (scaling, encoding, splits)

**Expected Output:**
```
✓ Stage 4: Feature Transformation - COMPLETE
  bucket_13-18s: 80 samples transformed (64 train, 16 test)
  bucket_18-33s: 80 samples transformed (64 train, 16 test)
  bucket_33-60s: 80 samples transformed (64 train, 16 test)
```

**Key Output Files:**
- `buckets/bucket_{duration}/transformed_features.pkl`
- `buckets/bucket_{duration}/checkpoints/stage_4_checkpoint.json`

### Stage 5: ML Model Training
**Action:** Train Random Forest + K-Means models per bucket

**Expected Output:**
```
✓ Stage 5: ML Model Training - COMPLETE
  bucket_13-18s: RF accuracy 0.85, K-Means silhouette 0.42
  bucket_18-33s: RF accuracy 0.82, K-Means silhouette 0.38
  bucket_33-60s: RF accuracy 0.88, K-Means silhouette 0.45
```

**Key Output Files:**
- `buckets/bucket_{duration}/models/random_forest_model.pkl`
- `buckets/bucket_{duration}/models/kmeans_model.pkl`
- `buckets/bucket_{duration}/models/training_summary.json`

### Stage 6: ML Analysis Generation
**Action:** Generate ML insights from trained models

**Expected Output:**
```
✓ Stage 6: ML Analysis Generation - COMPLETE
  Generated 3 bucket-level analysis JSONs
```

**Key Output Files:**
- `buckets/bucket_{duration}/ml_analysis.json`

### Stage 7: LLM Report Generation
**Action:** Transform ML insights into natural language reports using Claude API

**Expected Output:**
```
✓ Stage 7: LLM Report Generation - COMPLETE
  Generated client report: wellness_client_report.pdf
```

**Key Output Files:**
- `reports/wellness_client_report.pdf`
- `reports/wellness_client_report.json` (structured data)

---

## 📂 Complete Output File Tree

```
/home/jorge/rumiaifinal/data/clients/Rollo/hashtag/wellness/
│
├── cluster_analytics.json                    # Stage 1: Cluster health metrics
│
└── top_contrastive/
    ├── config.json                           # Stage 0: Pipeline configuration
    ├── winner_analysis.json                  # Stage 1: Bucket winners
    │
    ├── buckets/
    │   ├── bucket_13-18s/
    │   │   ├── selected_videos.json          # Stage 1: Selected 80 videos
    │   │   ├── selection_manifest.json       # Stage 2.5: Organized video list
    │   │   ├── insights/
    │   │   │   ├── {video_id}_temporal_windows_updated.json  # Stage 2 (×80)
    │   │   │   └── ...
    │   │   ├── classification/
    │   │   │   ├── {video_id}_classification.json           # Stage 2.7 (×80)
    │   │   │   └── ...
    │   │   ├── aggregated_features.csv       # Stage 3
    │   │   ├── transformed_features.pkl      # Stage 4
    │   │   ├── ml_analysis.json              # Stage 6
    │   │   ├── models/
    │   │   │   ├── random_forest_model.pkl   # Stage 5
    │   │   │   ├── kmeans_model.pkl          # Stage 5
    │   │   │   └── training_summary.json     # Stage 5
    │   │   └── checkpoints/
    │   │       ├── .stage2_checkpoint.json
    │   │       ├── .classification_checkpoint.json
    │   │       ├── stage_3_checkpoint.json
    │   │       ├── stage_4_checkpoint.json
    │   │       └── stage_5_checkpoint.json
    │   │
    │   ├── bucket_18-33s/                    # (Same structure as above)
    │   └── bucket_33-60s/                    # (Same structure as above)
    │
    ├── content_taxonomies/
    │   ├── wellness_raw_discovery.json       # Stage 2.6: LLM raw output
    │   └── wellness_taxonomy.json            # Stage 2.6: Manually curated
    │
    └── reports/
        ├── wellness_client_report.pdf        # Stage 7: Final client report
        └── wellness_client_report.json       # Stage 7: Structured report data
```

---

## ✅ Success Criteria

### Primary Success Criteria

**Pipeline Completion:**
- ✅ All stages complete without errors
- ✅ Exit code 0 on final completion
- ✅ No "CRITICAL ERROR" logs

**Output File Validation:**

| File | Expected Count | Validation |
|------|---------------|------------|
| `cluster_analytics.json` | 1 | Contains scrape_summary, per_hashtag_contribution |
| `winner_analysis.json` | 1 | Contains top_3_buckets array |
| `selected_videos.json` | 3 | 80 videos per bucket |
| `temporal_windows_updated.json` | 240 | All have temporal_windows.hook/closing |
| `classification.json` | 240 | All have content_category, hook_strategy |
| `aggregated_features.csv` | 3 | Rows match video count (80 per bucket) |
| `ml_analysis.json` | 3 | Contains rf_insights, kmeans_insights |
| `wellness_client_report.pdf` | 1 | Readable PDF, >10 pages |

### Secondary Success Criteria

**Cluster Analytics Quality:**
```json
{
  "scrape_summary": {
    "total_scrapes_attempted": 8,
    "total_scrapes_succeeded": 8,          // ✅ 100% success rate
    "total_unique_videos": "1500-2000",    // ✅ Within expected range
    "overall_duplication_rate": "20-40"    // ✅ Healthy overlap (not too high/low)
  },
  "per_hashtag_contribution": {
    "#wellness": {
      "contribution_percentage": ">15"     // ✅ All hashtags contribute meaningfully
    }
  }
}
```

**ML Model Quality:**
```json
{
  "random_forest": {
    "accuracy": ">0.75",                   // ✅ Reasonable classification performance
    "feature_importance_top5": "exists"    // ✅ Feature importance calculated
  },
  "kmeans": {
    "silhouette_score": ">0.30",          // ✅ Decent cluster separation
    "cluster_sizes": "balanced"            // ✅ No cluster has <10% samples
  }
}
```

**Taxonomy Quality:**
```json
{
  "content_categories": "5-10 items",      // ✅ Not too sparse/dense
  "hook_strategies": "5-10 items",
  "validation": "passes snake_case check"  // ✅ All names lowercase + underscore
}
```

---

## ❌ Failure Scenarios & Troubleshooting

### Failure 1: Stage 1 - Cluster Scraping Fails

**Symptoms:**
```
✗ Stage 1.1a: Cluster scraping failed
  #wellness: 0/2 runs successful
  Error: Apify API rate limit exceeded
```

**Root Causes:**
- Apify API key invalid or expired
- Insufficient Apify credits
- TikTok rate limiting

**Troubleshooting:**
```bash
# Check Apify API key
echo $APIFY_API_KEY

# Check Apify account credits
# Visit: https://console.apify.com/billing

# Reduce scraping intensity (edit cluster config)
{
  "scrape_config": {
    "runs_per_hashtag": 1,           # Reduce from 2 to 1
    "results_per_page": 400          # Reduce from 600 to 400
  }
}
```

**Expected Outcome:**
- Fewer scrapes (4 instead of 8)
- Lower cost but less data

---

### Failure 2: Stage 2 - Video Processing Timeout

**Symptoms:**
```
✗ Bucket bucket_33-60s: Video 7421234567890123456 failed
  Error: ML service timeout after 300s
```

**Root Causes:**
- FEAT emotion detection timeout (long videos)
- Network issues downloading video
- Insufficient system resources

**Troubleshooting:**
```bash
# Check Stage 2 checkpoint
cat data/clients/Rollo/hashtag/wellness/top_contrastive/buckets/bucket_33-60s/.stage2_checkpoint.json

# Resume will skip completed videos automatically
python rumiai_ml_batch.py ...

# If repeated failures, check logs
tail -100 data/logs/rumiai_ml_Rollo_wellness_*.log
```

**Expected Outcome:**
- Checkpoint/resume skips successful videos
- Only retries failed videos
- May need to skip problematic videos manually

---

### Failure 3: Stage 2.6 - Taxonomy Discovery Fails

**Symptoms:**
```
✗ Stage 2.6 failed: Claude API timeout
```

**Root Causes:**
- Claude API key invalid
- Insufficient Claude credits
- Sample size too large (>100 videos)

**Troubleshooting:**
```bash
# Check Claude API key
echo $ANTHROPIC_API_KEY

# Check sample transcripts exist
ls data/clients/Rollo/hashtag/wellness/top_contrastive/buckets/*/insights/*.json | wc -l

# Reduce sample size in code (if needed)
# discovery.py: sample_size=50 (default) → sample_size=30
```

**Expected Outcome:**
- Retry with valid API key succeeds
- Smaller sample size = faster discovery

---

### Failure 4: Stage 2.6 - Taxonomy Validation Fails

**Symptoms:**
```
✗ Taxonomy validation failed
  content_categories[2] name 'Wellness Routine' must be snake_case
```

**Root Causes:**
- Manual curation mistakes (capitals, spaces)
- Missing required fields
- Definitions too short (<10 chars)

**Troubleshooting:**
```bash
# Common mistakes:
❌ "Wellness Routine"     → ✅ "wellness_routine"
❌ "morning supplements"  → ✅ "morning_supplements"
❌ "definition": "Tips"   → ✅ "definition": "Wellness tips and advice"

# Validate manually
python -c "
from ml_pipeline.stage2_content_analysis.taxonomy_validation import validate_curated_taxonomy
validate_curated_taxonomy('data/clients/Rollo/hashtag/wellness/top_contrastive/content_taxonomies/wellness_taxonomy.json')
"
```

**Expected Outcome:**
- Fix validation errors
- Re-run pipeline (Stage 2.6 will skip, 2.7 will run)

---

### Failure 5: Stage 2.7 - Classification Fails

**Symptoms:**
```
✗ Stage 2.7: Classification failed
  bucket_13-18s: 45/80 videos classified (35 failed)
  Error: Claude API rate limit
```

**Root Causes:**
- Claude API rate limiting (too many requests)
- Taxonomy has too many categories (>15)
- Network interruption

**Troubleshooting:**
```bash
# Check classification checkpoint
cat data/clients/Rollo/hashtag/wellness/top_contrastive/buckets/bucket_13-18s/.classification_checkpoint.json

# Resume classification (automatic checkpoint/resume)
python rumiai_ml_batch.py ...

# Enable parallel classification (faster but more API usage)
export ENABLE_PARALLEL_CLASSIFICATION=true
export MAX_CLASSIFICATION_WORKERS=3
```

**Expected Outcome:**
- Checkpoint/resume completes remaining videos
- Parallel mode trades speed for API rate limits

---

### Failure 6: Stage 5 - ML Training Fails

**Symptoms:**
```
✗ Stage 5: ML training failed
  bucket_13-18s: Insufficient data (12 samples, minimum 20)
```

**Root Causes:**
- Too few videos passed earlier stages
- Stage 2 failures reduced sample size
- Contrastive selection needs minimum 20 videos

**Troubleshooting:**
```bash
# Check how many videos reached Stage 5
wc -l data/clients/Rollo/hashtag/wellness/top_contrastive/buckets/bucket_13-18s/aggregated_features.csv

# If <20 videos:
# Option A: Re-run with higher --video-count (e.g., 100)
# Option B: Use --selection-strategy top (no bottom 20% split)
```

**Expected Outcome:**
- Increase video count ensures sufficient training data
- Minimum 20 videos per bucket for ML training

---

### Failure 7: Stage 1 Re-Run (Resume Issue)

**Symptoms:**
```
Stage 1: Re-scraping videos (wasting time/money)
  Even though winner_analysis.json exists
```

**Root Cause:**
- Stage 1 has NO skip logic (known limitation)
- Re-running command re-scrapes all videos

**Troubleshooting:**
```bash
# Prevention: Don't re-run full command after Stage 2.6 pause

# If you need to resume after manual curation:
# Option A: Use --skip-taxonomy-curation flag (future feature)
# Option B: Manually skip to Stage 2.7 only (not yet implemented)

# Workaround: Complete pipeline in one run
# (curate taxonomy quickly, <5 min, to avoid long delay)
```

**Expected Outcome:**
- Minimize delays between Stage 2.6 pause and resume
- Future: Add Stage 1 skip logic or --skip-taxonomy-curation flag

---

## 📊 Performance Benchmarks

### Expected Costs

| Resource | Unit Cost | Units | Total |
|----------|-----------|-------|-------|
| Apify (scraping) | $0.10/scrape | 8 scrapes | **$0.80** |
| Claude API (discovery) | $0.015/1K tokens | ~50K tokens | **$0.75** |
| Claude API (classification) | $0.015/1K tokens | ~300K tokens | **$4.50** |
| Claude API (reports) | $0.015/1K tokens | ~100K tokens | **$1.50** |
| **TOTAL** | | | **~$7.55** |

### Expected Timeline (240 videos)

| Stage | Optimistic | Realistic | Pessimistic |
|-------|-----------|-----------|-------------|
| Stage 0-1 | 30 min | 40 min | 60 min |
| Stage 2 | 2.0 hrs | 2.5 hrs | 4.0 hrs |
| Stage 2.5-2.6 | 5 min | 10 min | 20 min |
| **⏸️ Manual** | **5 min** | **30 min** | **3 hrs** |
| Stage 2.7 | 15 min | 20 min | 40 min |
| Stage 3-7 | 45 min | 60 min | 90 min |
| **TOTAL** | **3.5 hrs** | **4.5 hrs** | **8.5 hrs** |

---

## 📝 Test Validation Checklist

### Pre-Test Checklist
- [ ] Environment variables set (APIFY_API_KEY, ANTHROPIC_API_KEY)
- [ ] Cluster config exists: `config/hashtag_clusters/wellness.json`
- [ ] No existing Rollo/wellness data (clean test)
- [ ] Apify credits available (>$1)
- [ ] Claude credits available (>$7)

### Stage Completion Checklist
- [ ] Stage 0: config.json created
- [ ] Stage 1: cluster_analytics.json generated (8/8 scrapes successful)
- [ ] Stage 1: winner_analysis.json shows 3 winning buckets
- [ ] Stage 1: 240 videos selected (80 per bucket)
- [ ] Stage 2: 240 temporal_windows files created
- [ ] Stage 2.5: Files organized into bucket directories
- [ ] Stage 2.6: wellness_raw_discovery.json created
- [ ] Stage 2.6: Manual curation complete, wellness_taxonomy.json created
- [ ] Stage 2.7: 240 classification.json files created
- [ ] Stage 3: 3 aggregated_features.csv files (one per bucket)
- [ ] Stage 4: 3 transformed_features.pkl files
- [ ] Stage 5: 6 model files (RF + K-Means per bucket)
- [ ] Stage 6: 3 ml_analysis.json files
- [ ] Stage 7: wellness_client_report.pdf generated

### Quality Validation Checklist
- [ ] cluster_analytics.json: Duplication rate 20-40%
- [ ] cluster_analytics.json: All hashtags contribute >10%
- [ ] temporal_windows files: All have hook/closing windows
- [ ] classification files: All have valid content_category
- [ ] ML models: RF accuracy >0.75, K-Means silhouette >0.30
- [ ] Client report: PDF opens, >10 pages, contains insights

---

## 🎯 Test Execution Log Template

```markdown
# Test Execution: E2E-WELLNESS-001

**Date:** YYYY-MM-DD
**Tester:** [Name]
**Environment:** Production / Staging

## Execution Timeline

| Stage | Start Time | End Time | Duration | Status |
|-------|-----------|----------|----------|--------|
| Stage 0 | HH:MM | HH:MM | X min | ✅ / ❌ |
| Stage 1 | HH:MM | HH:MM | X min | ✅ / ❌ |
| Stage 2 | HH:MM | HH:MM | X hrs | ✅ / ❌ |
| Stage 2.5 | HH:MM | HH:MM | X sec | ✅ / ❌ |
| Stage 2.6 | HH:MM | HH:MM | X min | ✅ / ❌ |
| Manual Curation | HH:MM | HH:MM | X hrs | ✅ / ❌ |
| Stage 2.7 | HH:MM | HH:MM | X min | ✅ / ❌ |
| Stage 3 | HH:MM | HH:MM | X min | ✅ / ❌ |
| Stage 4 | HH:MM | HH:MM | X min | ✅ / ❌ |
| Stage 5 | HH:MM | HH:MM | X min | ✅ / ❌ |
| Stage 6 | HH:MM | HH:MM | X min | ✅ / ❌ |
| Stage 7 | HH:MM | HH:MM | X min | ✅ / ❌ |

## Key Metrics

- **Total Videos Scraped:** [number]
- **Unique Videos After Dedup:** [number]
- **Videos After Date Filter:** [number]
- **Winning Buckets:** [bucket names]
- **Videos Processed:** [completed]/[total]
- **Videos Classified:** [completed]/[total]
- **ML Model Accuracy (avg):** [percentage]

## Issues Encountered

| Stage | Issue | Resolution | Impact |
|-------|-------|------------|--------|
| [stage] | [description] | [how fixed] | Low/Med/High |

## Test Result

**Overall Status:** ✅ PASS / ❌ FAIL
**Success Criteria Met:** [X]/[Y]
**Notes:** [Any observations, recommendations, or follow-ups]
```

---

## 📚 Related Documentation

- **Cluster Setup:** `CLUSTER_QUICK_START.md`
- **Cluster Implementation:** `HASHTAG_CLUSTER_IMPLEMENTATION.md`
- **System Architecture:** `SystemArchitecturev2.md`
- **ML Roadmap:** `MLROADMAP.md`
- **Quick Reference:** `QUICK_REFERENCE.md`

---

## 📞 Support & Debugging

**Logs Location:**
```bash
tail -f data/logs/rumiai_ml_Rollo_wellness_*.log
```

**Common Debug Commands:**
```bash
# Check Stage 1 cluster analytics
cat data/clients/Rollo/hashtag/wellness/cluster_analytics.json | jq '.scrape_summary'

# Check Stage 2 checkpoint status
cat data/clients/Rollo/hashtag/wellness/top_contrastive/buckets/*/.[stage]*checkpoint.json | jq

# Count processed videos
find data/clients/Rollo/hashtag/wellness -name "*temporal_windows_updated.json" | wc -l

# Count classified videos
find data/clients/Rollo/hashtag/wellness -name "*classification.json" | wc -l

# Check ML model files
ls -lh data/clients/Rollo/hashtag/wellness/top_contrastive/buckets/*/models/
```

---

**Test Document Version:** 1.0
**Last Updated:** 2025-10-22
**Author:** RumiAI Testing Team
