# End-to-End Test: Wellness Cluster (Client: Rollo)

---

## 📋 Test Index

### Hashtag Tests (Wellness Cluster)
- [Test 1: 150-Day Date Filter](#test-1-150-day-date-filter) - Original baseline test
- [Test 2: 270-Day Date Filter](#test-2-270-day-date-filter) - Extended 9-month window
- [Test 3: 180-Minute Scrape Delay + 100 Videos](#test-3-180-minute-scrape-delay--100-videos) - Extended delay with larger sample size
- [Test 4: 4 Runs Per Hashtag + 100 Videos](#test-4-4-runs-per-hashtag--100-videos) - Doubled scraping runs with larger sample size
- [Test 5: Healthy + Fitness Cluster (9 Hashtags)](#test-5-healthy--fitness-cluster-9-hashtags) - Comprehensive fitness-focused cluster, 36 scrapes, 100 videos per bucket

### Competitor Tests
- [CompetitorTest: @nutrachampssupplement](#competitortest-nutrachampssupplement) - Single creator analysis, top 60 per bucket, 270-day window

---

# Test 1: 150-Day Date Filter

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
| `--video-count` | 100 | Videos per winning bucket (64 top + 16 bottom) |
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

---
---
---

# Test 2: 270-Day Date Filter

## 📋 Test Overview

**Test ID:** E2E-WELLNESS-002
**Test Type:** Extended Date Range Validation (9-Month Window)
**Client:** Rollo_Test2
**Cluster:** wellness (4 hashtags)
**Objective:** Validate pipeline performance with extended 270-day date filter to increase video pool size and improve ML model training data

**Key Differences from Test 1:**
- Client name: `Rollo` → `Rollo_Test2` (natural test isolation)
- Date filter: `last_150_days` → `last_270_days` (9 months vs 5 months)
- Taxonomy: Fresh discovery (not reusing Test 1)
- Expected video pool: ~800-1,200 videos → **~1,400-2,200 videos**
- Expected selected videos: 240 → **potentially more per bucket**

---

## 🎯 Test Objectives

This E2E test validates:

1. **Extended Date Range Impact:** Assess how 9-month window affects video diversity and quality
2. **Larger Training Sets:** Validate ML model performance with increased training data
3. **Cluster Coverage:** Determine if older videos still maintain hashtag relevance
4. **Processing Scalability:** Confirm pipeline handles larger video counts without degradation
5. **ML Model Quality:** Compare model metrics against Test 1 baseline (expected improvement)

**Success Criteria:**
- All pipeline stages complete successfully
- ML model accuracy ≥ Test 1 metrics (RF accuracy >0.75, K-Means silhouette >0.30)
- No significant increase in failed/skipped videos despite larger pool

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

**Test Isolation Strategy:** Use different client name (`Rollo_Test2`) for automatic separation

**Prerequisites:**
- Test 1 data exists in `/data/clients/Rollo/` (preserved, untouched)
- Test 2 will create fresh `/data/clients/Rollo_Test2/` directory
- No archiving or manual cleanup needed
- Fresh taxonomy discovery (Stage 2.6 will run and pause for curation)

**Why Different Client Name?**
- ✅ Automatic test isolation (no data mixing)
- ✅ No checkpoint conflicts (fresh start guaranteed)
- ✅ Test 1 results preserved for comparison
- ✅ Simpler than archiving (no manual `mv` commands)
- ✅ Side-by-side comparison after Test 2 completes

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

| Parameter | Value | Change from Test 1 | Rationale |
|-----------|-------|---------------------|-----------|
| `--client` | Rollo_Test2 | **CHANGED** (was: Rollo) | **Natural test isolation, prevents data mixing** |
| `--target` | wellness | **No change** | Same cluster |
| `--analysis-type` | hashtag | **No change** | Cluster analysis mode |
| `--selection-strategy` | contrastive | **No change** | Top 80% + Bottom 20% |
| `--video-count` | 80 | **No change** | Videos per winning bucket |
| `--date-filter` | last_270_days | **CHANGED** (was: last_150_days) | **9-month window for more data** |
| `--country-code` | US | **No change** | Geographic filter |
| `--report-type` | single | **No change** | Single hashtag analysis |
| `--report-audience` | client | **No change** | Report format for brand/client |

**Expected Scraping (Identical to Test 1):**
- 4 hashtags × 2 runs × 600 results = 8 scrapes
- ~4,800 videos before deduplication
- ~1,500-2,000 unique videos after deduplication

**Expected Date Filter Impact (New):**
- Test 1: ~800-1,200 videos after 150-day filter (60-65% retention)
- **Test 2: ~1,400-2,200 videos after 270-day filter (75-85% retention)**
- **Impact:** +75-83% more videos available for bucket selection

---

## 🚀 Test Execution

### Command

```bash
cd /home/jorge/rumiaifinal

python rumiai_ml_batch.py \
  --client Rollo_Test2 \
  --target wellness \
  --analysis-type hashtag \
  --selection-strategy contrastive \
  --video-count 100 \
  --date-filter last_270_days \
  --country-code US \
  --report-type single \
  --report-audience client
```

### Expected Runtime

| Stage | Test 1 Duration | Test 2 Expected Duration | Delta | Notes |
|-------|----------------|--------------------------|-------|-------|
| Stage 0 | <5 seconds | <5 seconds | No change | Foundation setup |
| Stage 1 | 30-45 minutes | 30-45 minutes | No change | Scraping identical (8 scrapes) |
| Stage 2 | 2-4 hours | **2-4 hours** | No change | Same 240 videos processed |
| Stage 2.5 | <10 seconds | <10 seconds | No change | File organization |
| Stage 2.6 | 3-5 minutes | 3-5 minutes | No change | Pattern discovery (fresh) |
| **⏸️ PAUSE** | **Manual** | **1-3 hours** | **Same** | **Fresh taxonomy curation** |
| Stage 2.7 | 15-30 minutes | 15-30 minutes | No change | Same 240 videos classified |
| Stage 3 | 10-20 minutes | 10-20 minutes | No change | Aggregation |
| Stage 4 | 5-10 minutes | 5-10 minutes | No change | Transformation |
| Stage 5 | 10-15 minutes | 10-15 minutes | No change | ML training |
| Stage 6 | 5-10 minutes | 5-10 minutes | No change | ML analysis |
| Stage 7 | 20-30 minutes | 20-30 minutes | No change | LLM reports |
| **TOTAL** | **4-6 hours** | **4-6 hours** | **Same** | **(excluding manual curation)** |

**Key Insights:**
- Runtime is **identical to Test 1** because `--video-count 100` still selects only 240 videos
- The larger date window improves **selection quality** (more candidates to choose from), not quantity
- Fresh taxonomy discovery tests if 270-day window reveals different content patterns

---

## 📊 Expected Differences from Test 1

### 1. Video Pool Size (Stage 1)

**Test 1 (150 days):**
```
Date filter: 1,850 → 1,120 videos (last 150 days)
Retention: ~60%
```

**Test 2 (270 days):**
```
Date filter: 1,850 → 1,600 videos (last 270 days)
Retention: ~86%
Expected delta: +480 more videos (+43%)
```

### 2. Bucket Distribution Quality (Stage 1)

**Hypothesis:** Longer date range → more balanced bucket distribution

**Test 1 Example:**
```json
{
  "bucket_13-18s": { "video_count": 180, "top_80_selected": 64 },
  "bucket_18-33s": { "video_count": 350, "top_80_selected": 64 },
  "bucket_33-60s": { "video_count": 420, "top_80_selected": 64 }
}
```

**Test 2 Expected:**
```json
{
  "bucket_13-18s": { "video_count": 280, "top_80_selected": 64 },  // +100 videos (55% increase)
  "bucket_18-33s": { "video_count": 520, "top_80_selected": 64 },  // +170 videos (49% increase)
  "bucket_33-60s": { "video_count": 600, "top_80_selected": 64 }   // +180 videos (43% increase)
}
```

**Impact:** More competitive selection (top 80 out of 280 vs top 80 out of 180) → higher quality training data

### 3. Content Diversity (Stage 2.6/2.7)

**Hypothesis:** 9-month window captures seasonal trends + content evolution

**Expected Taxonomy Differences:**
- Test 1 (5 months): May miss seasonal patterns (e.g., "New Year wellness resolutions")
- Test 2 (9 months): Captures 3 full seasons → more diverse content categories

**Example:**
```
Test 1 content_categories: ["supplement_routine", "health_education", "transformation_story"]
Test 2 content_categories: ["supplement_routine", "health_education", "transformation_story", "seasonal_wellness", "holiday_recovery"]
```

**Validation Strategy:**
1. Compare `wellness_raw_discovery.json` between Test 1 and Test 2
2. Measure taxonomy overlap (expected 70-90% core categories preserved)
3. Identify unique categories in Test 2 (seasonal/temporal patterns)

**Important Note:** Fresh taxonomy discovery means Test 2 has **two variables**:
- Variable 1: Date filter (150d → 270d)
- Variable 2: Taxonomy (may differ from Test 1)

This tests both date range impact AND whether longer windows reveal different content patterns.

### 4. ML Model Performance (Stage 5)

**Hypothesis:** Better data quality → improved model metrics

**Test 1 Baseline:**
```json
{
  "bucket_13-18s": { "rf_accuracy": 0.82, "kmeans_silhouette": 0.38 },
  "bucket_18-33s": { "rf_accuracy": 0.85, "kmeans_silhouette": 0.42 },
  "bucket_33-60s": { "rf_accuracy": 0.88, "kmeans_silhouette": 0.45 }
}
```

**Test 2 Expected (Optimistic):**
```json
{
  "bucket_13-18s": { "rf_accuracy": 0.85, "kmeans_silhouette": 0.42 },  // +3% / +10%
  "bucket_18-33s": { "rf_accuracy": 0.87, "kmeans_silhouette": 0.45 },  // +2% / +7%
  "bucket_33-60s": { "rf_accuracy": 0.90, "kmeans_silhouette": 0.48 }   // +2% / +7%
}
```

**Validation Metric:** Calculate average improvement across all buckets

### 5. LLM Report Quality (Stage 7)

**Hypothesis:** More diverse content → richer creative insights

**Expected Differences:**
- Test 1: Insights based on recent trends (5 months)
- Test 2: Insights capture longer-term patterns + seasonal trends

**Example:**
```
Test 1: "85% of top videos use relatable symptoms in hook"
Test 2: "85% of top videos use relatable symptoms in hook (consistent across all 3 seasons)"
```

---

## 📊 Success Criteria

### Primary Success Criteria (Same as Test 1)

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

### Secondary Success Criteria (Test 2 Specific)

**Date Filter Impact:**
```json
{
  "videos_after_filter": ">1400",           // ✅ 9-month window increases pool
  "filter_retention_rate": ">75%",          // ✅ Higher retention than Test 1 (60%)
  "bucket_depth_improvement": ">40%"        // ✅ More videos per bucket for selection
}
```

**ML Model Quality (Comparison):**
```json
{
  "rf_accuracy_improvement": "≥0% vs Test 1",     // ✅ At minimum, no degradation
  "kmeans_silhouette_improvement": "≥0% vs Test 1", // ✅ Ideally +5-10%
  "feature_importance_stability": "top 5 consistent" // ✅ Same key features as Test 1
}
```

**Content Diversity:**
```json
{
  "content_categories_discovered": "≥5 categories",  // ✅ Rich taxonomy
  "seasonal_patterns_detected": "≥1 seasonal trend", // ✅ Unique to 9-month window
  "taxonomy_overlap_with_test1": "70-90%"           // ✅ Core categories preserved
}
```

---

## 📊 Comparative Analysis: Test 1 vs Test 2

### Video Pool Comparison

| Metric | Test 1 (150d) | Test 2 (270d) | Delta | % Change |
|--------|---------------|---------------|-------|----------|
| **Scraped (raw)** | ~4,800 | ~4,800 | 0 | 0% |
| **Unique (dedup)** | ~1,850 | ~1,850 | 0 | 0% |
| **After date filter** | ~1,120 | ~1,600 | +480 | **+43%** |
| **Bucket depth (avg)** | ~373 | ~533 | +160 | **+43%** |
| **Videos selected** | 240 | 240 | 0 | 0% |
| **Selection competitiveness** | Top 21% | Top 15% | -6% | **More selective** |

**Key Insight:** Test 2 selects from a **more competitive pool** (top 15% vs top 21%), potentially improving quality.

### Expected Cost Comparison

| Resource | Test 1 Cost | Test 2 Cost | Delta | Notes |
|----------|-------------|-------------|-------|-------|
| Apify (scraping) | $0.80 | $0.80 | $0.00 | Same 8 scrapes |
| Claude API (discovery) | $0.75 | $0.75 | $0.00 | Fresh taxonomy discovery |
| Claude API (classification) | $4.50 | $4.50 | $0.00 | Same 240 videos |
| Claude API (reports) | $1.50 | $1.50 | $0.00 | Same report count |
| **TOTAL** | **$7.55** | **$7.55** | **$0.00** | **Identical cost** |

**Note:** Fresh taxonomy discovery means same cost as Test 1, but allows testing if 270-day window reveals different content patterns.

### Timeline Comparison

| Scenario | Test 1 Duration | Test 2 Duration | Notes |
|----------|----------------|-----------------|-------|
| **Fresh taxonomy** | 4-6 hrs | 4-6 hrs | Same (manual curation 1-3 hrs) |
| **Manual curation time** | 1-3 hrs | 1-3 hrs | Same effort (depends on pattern complexity) |

---

## 🧪 Test Validation Checklist

### Pre-Test Checklist
- [ ] Environment variables set (APIFY_API_KEY, ANTHROPIC_API_KEY)
- [ ] Cluster config exists: `config/hashtag_clusters/wellness.json`
- [ ] Test 1 results exist in `data/clients/Rollo/` (preserved for comparison)
- [ ] No existing `data/clients/Rollo_Test2/` directory (fresh start)
- [ ] Apify credits available (>$1)
- [ ] Claude credits available (>$8)

### Stage Completion Checklist (Same as Test 1)
- [ ] Stage 0: config.json created
- [ ] Stage 1: cluster_analytics.json generated (8/8 scrapes successful)
- [ ] Stage 1: winner_analysis.json shows 3 winning buckets
- [ ] Stage 1: **>1,400 videos after date filter** (vs ~1,120 in Test 1)
- [ ] Stage 1: 240 videos selected (80 per bucket)
- [ ] Stage 2: 240 temporal_windows files created
- [ ] Stage 2.5: Files organized into bucket directories
- [ ] Stage 2.6: **Fresh taxonomy discovered** (wellness_raw_discovery.json created)
- [ ] Manual curation: wellness_taxonomy.json created from raw discovery
- [ ] Stage 2.7: 240 classification.json files created
- [ ] Stage 3: 3 aggregated_features.csv files (one per bucket)
- [ ] Stage 4: 3 transformed_features.pkl files
- [ ] Stage 5: 6 model files (RF + K-Means per bucket)
- [ ] Stage 6: 3 ml_analysis.json files
- [ ] Stage 7: wellness_client_report.pdf generated

### Quality Validation Checklist (Test 2 Specific)
- [ ] cluster_analytics.json: **>1,400 videos after 270-day filter**
- [ ] cluster_analytics.json: Duplication rate 20-40%
- [ ] winner_analysis.json: **Bucket counts +40-50% vs Test 1**
- [ ] Taxonomy comparison: 70-90% overlap with Test 1 (core categories preserved)
- [ ] Taxonomy unique patterns: ≥1 seasonal/temporal pattern detected in Test 2
- [ ] ML models: RF accuracy ≥ Test 1 baseline
- [ ] ML models: K-Means silhouette ≥ Test 1 baseline
- [ ] Client report: Contains insights spanning 9-month window

---

## 🎯 Test Execution Log Template

```markdown
# Test Execution: E2E-WELLNESS-002

**Date:** YYYY-MM-DD
**Tester:** [Name]
**Environment:** Production / Staging
**Taxonomy Source:** Fresh Discovery (9-month window)

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
- **Videos After Date Filter (270d):** [number] (Compare to Test 1: ~1,120)
- **Filter Retention Rate:** [percentage]%
- **Winning Buckets:** [bucket names]
- **Bucket Depth (avg):** [number] (Compare to Test 1: ~373)
- **Videos Processed:** [completed]/[total]
- **Videos Classified:** [completed]/[total]
- **ML Model Accuracy (avg):** [percentage] (Compare to Test 1 baseline)
- **K-Means Silhouette (avg):** [score] (Compare to Test 1 baseline)
- **Taxonomy Overlap with Test 1:** [percentage]% ([X]/[Y] categories match)
- **Unique Test 2 Patterns:** [list seasonal/temporal patterns]

## Test 1 vs Test 2 Comparison

| Metric | Test 1 (150d) | Test 2 (270d) | Delta | Improvement |
|--------|---------------|---------------|-------|-------------|
| Videos after filter | ~1,120 | [actual] | [delta] | [%] |
| Bucket depth (avg) | ~373 | [actual] | [delta] | [%] |
| RF accuracy (avg) | 0.85 | [actual] | [delta] | [%] |
| K-Means silhouette (avg) | 0.42 | [actual] | [delta] | [%] |

## Issues Encountered

| Stage | Issue | Resolution | Impact |
|-------|-------|------------|--------|
| [stage] | [description] | [how fixed] | Low/Med/High |

## Test Result

**Overall Status:** ✅ PASS / ❌ FAIL
**Success Criteria Met:** [X]/[Y]
**Comparison to Test 1:** Better / Same / Worse
**Recommendation:** [Use Test 1 or Test 2 settings for production? Why?]

**Notes:** [Any observations, particularly regarding 9-month window impact]
```

---

## 📚 Related Documentation

- **Test 1 (Baseline):** See above in this document
- **Cluster Setup:** `CLUSTER_QUICK_START.md`
- **Cluster Implementation:** `HASHTAG_CLUSTER_IMPLEMENTATION.md`
- **System Architecture:** `SystemArchitecturev2.md`
- **ML Roadmap:** `MLROADMAP.md`

---

## 📞 Support & Debugging

**Logs Location:**
```bash
tail -f data/logs/rumiai_ml_Rollo_wellness_*.log
```

**Common Debug Commands:**
```bash
# Compare Test 1 vs Test 2 date filter results
echo "Test 1:" && cat data/clients/Rollo/hashtag/wellness/top_contrastive/winner_analysis.json | jq '.bucket_stats'
echo "Test 2:" && cat data/clients/Rollo_Test2/hashtag/wellness/top_contrastive/winner_analysis.json | jq '.bucket_stats'

# Check Stage 1 video counts
echo "Test 1:" && cat data/clients/Rollo/hashtag/wellness/cluster_analytics.json | jq '.scrape_summary'
echo "Test 2:" && cat data/clients/Rollo_Test2/hashtag/wellness/cluster_analytics.json | jq '.scrape_summary'

# Compare taxonomies
echo "Test 1:" && cat data/clients/Rollo/hashtag/wellness/top_contrastive/content_taxonomies/wellness_taxonomy.json | jq '.content_categories[].name'
echo "Test 2:" && cat data/clients/Rollo_Test2/hashtag/wellness/top_contrastive/content_taxonomies/wellness_taxonomy.json | jq '.content_categories[].name'

# Compare ML model metrics
echo "Test 1 models:" && cat data/clients/Rollo/hashtag/wellness/top_contrastive/buckets/*/models/training_summary.json | jq '.metrics'
echo "Test 2 models:" && cat data/clients/Rollo_Test2/hashtag/wellness/top_contrastive/buckets/*/models/training_summary.json | jq '.metrics'

# Count processed videos (should be 240 for both tests)
echo "Test 1:" && find data/clients/Rollo -name "*temporal_windows_updated.json" | wc -l
echo "Test 2:" && find data/clients/Rollo_Test2 -name "*temporal_windows_updated.json" | wc -l
```

---

**Test 2 Added:** 2025-10-25
**Author:** RumiAI Testing Team

---
---
---

# Test 3: 180-Minute Scrape Delay + 100 Videos

## 🚀 Quick Start for Fresh CLI Instance

**If you're running this test for the first time or restarting after a failure, follow these steps:**

### Step 1: Clean Up Previous Attempts (if any)
```bash
rm -rf /home/jorge/rumiaifinal/data/clients/rollo_test3
```

### Step 2: Disable Windows Sleep
- Windows Settings → System → Power & Sleep
- Set "When plugged in, PC goes to sleep after" → **Never**

### Step 3: Verify Code Changes
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
python -c "from ml_pipeline.stage1_discovery.constants import MAX_DELAY_BETWEEN_RUNS_MS; print(f'✅ Delays supported up to {MAX_DELAY_BETWEEN_RUNS_MS/60000:.0f} minutes')"
```

### Step 4: Run Test 3
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

python rumiai_ml_batch.py \
  --client Rollo_Test3 \
  --target wellness_test3 \
  --analysis-type hashtag \
  --selection-strategy contrastive \
  --video-count 100 \
  --date-filter last_270_days \
  --country-code US \
  --report-type single \
  --report-audience client
```

**Expected Duration:** 16-18 hours (excluding 1-3 hour manual taxonomy curation)

---

## 📋 Test Overview

**Test ID:** E2E-WELLNESS-003
**Test Type:** Extended Scrape Delay + Larger Sample Size Validation
**Client:** Rollo_Test3
**Cluster:** wellness_test3 (4 hashtags, custom delay config)
**Objective:** Validate pipeline performance with extended 180-minute scrape delays and 100 videos per bucket to test rate-limiting avoidance and improved ML model training with larger datasets

**Key Differences from Test 2:**
- Client name: `Rollo_Test2` → `Rollo_Test3` (natural test isolation)
- Video count: `80` → `100` (+25% more videos per bucket)
- Scrape delay: `120,000 ms (2 min)` → `10,800,000 ms (180 min)` (+90x longer delays)
- Cluster config: Uses `wellness_test3.json` (custom config with 180-min delay)
- Expected Stage 1 duration: ~30-45 min → **~12+ hours** (due to 180-min delays)
- Expected total pipeline duration: ~4-6 hrs → **~16-18 hours** (excluding manual curation)

---

## 🎯 Test Objectives

This E2E test validates:

1. **Extended Scrape Delays:** Test system handling of 180-minute delays between scrapes (rate limiting avoidance)
2. **Larger Training Sets:** Validate ML model performance with 100 videos per bucket (vs 80 in Test 2)
3. **Long-Running Pipeline Stability:** Confirm system stability over 16-18 hour execution window
4. **Video Quality Impact:** Assess if extended delays improve video diversity/quality
5. **ML Model Quality:** Compare model metrics against Test 2 baseline (expected improvement with more data)

**Success Criteria:**
- All pipeline stages complete successfully over extended timeline
- ML model accuracy ≥ Test 2 metrics (RF accuracy >0.75, K-Means silhouette >0.30)
- No timeout or stability issues during 180-minute delays
- Video processing completes for all 300 videos (100 per bucket)

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

### ⚠️ **CRITICAL: Prevent System Sleep During Test**

**Before starting this test, you MUST disable computer sleep to prevent connection failures during long delays.**

**On Windows (for WSL2):**
1. Open Windows Settings → System → Power & Sleep
2. Set "When plugged in, PC goes to sleep after" → **Never**
3. Set "When plugged in, turn off screen after" → **Never** (optional, but recommended)

**Why:** The 180-minute delays use long-running HTTP connections to Apify. If your computer sleeps:
- ❌ Network connections will break ("Connection reset by peer" errors)
- ❌ Apify scrapes will fail even though they completed successfully on Apify's side
- ❌ Test will fail and need to restart from scratch

**After test completes:** Remember to re-enable sleep settings!

---

### Required Code Changes

**✅ ALREADY APPLIED:** Test 3 code changes have been implemented:

1. **File:** `ml_pipeline/stage1_discovery/constants.py` (Line 137)
   ```python
   MAX_DELAY_BETWEEN_RUNS_MS = 10800000  # 180 minutes (supports long delays)
   ```

2. **File:** `ml_pipeline/stage1_discovery/cluster_scraper.py` (Lines 136-154)
   ```python
   # Chunked sleep implementation (prevents hang if system suspends/resumes)
   start_time = time.time()
   end_time = start_time + delay_seconds
   while time.time() < end_time:
       remaining = end_time - time.time()
       if remaining > 0:
           time.sleep(min(60, remaining))  # Sleep in 60-second chunks
   ```

**Validation:** Verify the code changes are present:
```bash
cd /home/jorge/rumiaifinal
python -c "
from ml_pipeline.stage1_discovery.constants import MAX_DELAY_BETWEEN_RUNS_MS
assert MAX_DELAY_BETWEEN_RUNS_MS >= 10800000, 'MAX_DELAY_BETWEEN_RUNS_MS must be ≥ 10800000'
print(f'✅ MAX_DELAY_BETWEEN_RUNS_MS = {MAX_DELAY_BETWEEN_RUNS_MS} ({MAX_DELAY_BETWEEN_RUNS_MS/60000:.0f} minutes)')
"
```

### Initial State

**Test Isolation Strategy:** Use different client name (`Rollo_Test3`) for automatic separation

**Prerequisites:**
- Test 1 data exists in `/data/clients/Rollo/` (preserved)
- Test 2 data exists in `/data/clients/Rollo_Test2/` (preserved)
- Test 3 will create fresh `/data/clients/Rollo_Test3/` directory
- Custom cluster config exists: `config/hashtag_clusters/wellness_test3.json` ✅ (already created)
- Fresh taxonomy discovery (Stage 2.6 will run and pause for curation)

**Cleaning Up Failed Test Attempts:**

If you need to restart Test 3 (e.g., after a failure), simply delete the client data directory:

```bash
# Safe to delete - only removes Test 3 data, preserves Test 1 & Test 2
rm -rf /home/jorge/rumiaifinal/data/clients/rollo_test3

# Verify it's deleted
ls /home/jorge/rumiaifinal/data/clients/

# Should show: Rollo  Rollo_Test2  (but NOT rollo_test3)
```

**Why this is safe:**
- The pipeline will recreate the directory automatically
- No checkpoints exist from failed test (test never completed Stage 1)
- Test 1 and Test 2 data remain untouched

---

## 📦 Test Configuration

### Cluster Configuration
**File:** `/home/jorge/rumiaifinal/config/hashtag_clusters/wellness_test3.json`

**⚠️ MUST CREATE THIS FILE BEFORE RUNNING TEST**

```json
{
  "cluster_id": "wellness_test3",
  "description": "Wellness cluster for Test 3 - Extended 180-minute scrape delays to avoid rate limiting",
  "primary_hashtag": "#wellness",
  "variant_hashtags": [
    "#wellnesssupplements",
    "#healthandwellness",
    "#wellnessjourney"
  ],
  "scrape_config": {
    "runs_per_hashtag": 2,
    "delay_between_runs_ms": 10800000,
    "results_per_page": 600
  }
}
```

**Key Change:** `delay_between_runs_ms: 10800000` (180 minutes) vs `120000` (2 minutes) in Test 2

### CLI Parameters

| Parameter | Value | Change from Test 2 | Rationale |
|-----------|-------|---------------------|-----------|
| `--client` | Rollo_Test3 | **CHANGED** (was: Rollo_Test2) | **Natural test isolation** |
| `--target` | wellness_test3 | **CHANGED** (was: wellness) | **Custom cluster config with 180-min delay** |
| `--analysis-type` | hashtag | **No change** | Cluster analysis mode |
| `--selection-strategy` | contrastive | **No change** | Top 80% + Bottom 20% |
| `--video-count` | 100 | **CHANGED** (was: 80) | **+25% more videos per bucket** |
| `--date-filter` | last_270_days | **No change** | 9-month window (same as Test 2) |
| `--country-code` | US | **No change** | Geographic filter |
| `--report-type` | single | **No change** | Single hashtag analysis |
| `--report-audience` | client | **No change** | Report format for brand/client |

**Expected Scraping (Same volume, different timing):**
- 4 hashtags × 2 runs × 600 results = 8 scrapes
- ~4,800 videos before deduplication
- ~1,500-2,000 unique videos after deduplication
- ~1,400-2,200 videos after 270-day filter (same as Test 2)

**Expected Stage 1 Duration:**
- Test 2: ~30-45 minutes (2-min delays)
- **Test 3: ~12+ hours** (180-min delays between 8 scrapes)
- Calculation: 7 delays × 180 min = 1,260 min (21 hours) + scrape time

**Expected Video Processing:**
- Test 2: 240 videos (80 per bucket)
- **Test 3: 300 videos (100 per bucket)** (+25% increase)
- Contrastive split: **80 top + 20 bottom** per bucket

---

## 🚀 Test Execution

### Command

```bash
cd /home/jorge/rumiaifinal

python rumiai_ml_batch.py \
  --client Rollo_Test3 \
  --target wellness_test3 \
  --analysis-type hashtag \
  --selection-strategy contrastive \
  --video-count 100 \
  --date-filter last_270_days \
  --country-code US \
  --report-type single \
  --report-audience client
```

### Expected Runtime

| Stage | Test 2 Duration | Test 3 Expected Duration | Delta | Notes |
|-------|----------------|--------------------------|-------|-------|
| Stage 0 | <5 seconds | <5 seconds | No change | Foundation setup |
| Stage 1 | 30-45 minutes | **12-21 hours** | **+12-20 hours** | **180-min delays × 7 + scrape time** |
| Stage 2 | 2-4 hours | **2.5-5 hours** | **+0.5-1 hour** | **+25% more videos (300 vs 240)** |
| Stage 2.5 | <10 seconds | <10 seconds | No change | File organization |
| Stage 2.6 | 3-5 minutes | 3-5 minutes | No change | Pattern discovery (fresh) |
| **⏸️ PAUSE** | **Manual** | **1-3 hours** | **Same** | **Fresh taxonomy curation** |
| Stage 2.7 | 15-30 minutes | **20-40 minutes** | **+5-10 min** | **+25% more videos to classify** |
| Stage 3 | 10-20 minutes | **12-25 minutes** | **+2-5 min** | **+25% aggregation work** |
| Stage 4 | 5-10 minutes | **6-12 minutes** | **+1-2 min** | **+25% transformation work** |
| Stage 5 | 10-15 minutes | **12-18 minutes** | **+2-3 min** | **+25% training data** |
| Stage 6 | 5-10 minutes | 5-10 minutes | No change | ML analysis |
| Stage 7 | 20-30 minutes | 20-30 minutes | No change | LLM reports |
| **TOTAL** | **4-6 hours** | **16-18 hours** | **+12 hours** | **(excluding manual curation)** |

**⚠️ CRITICAL TIMELINE NOTE:**
- **Stage 1 alone takes 12-21 hours** due to 180-minute delays
- **Recommendation:** Run overnight or over weekend
- **Monitoring:** Check logs periodically during Stage 1 to verify progress

---

## 📊 Expected Differences from Test 2

### 1. Stage 1 Execution Time (Massive Increase)

**Test 2 (2-minute delays):**
```
Stage 1 duration: ~30-45 minutes
8 scrapes with 7 delays × 2 min = 14 minutes of waiting
```

**Test 3 (180-minute delays):**
```
Stage 1 duration: ~12-21 hours
8 scrapes with 7 delays × 180 min = 1,260 minutes (21 hours) of waiting
Actual duration depends on Apify scrape execution time per run
```

**Impact:** Stage 1 becomes the bottleneck (70-80% of total pipeline time)

### 2. Video Pool Size (Same as Test 2)

**Test 2:**
```
Date filter: 1,850 → ~1,600 videos (last 270 days)
```

**Test 3:**
```
Date filter: 1,850 → ~1,600 videos (last 270 days)
Expected: IDENTICAL to Test 2 (same 270-day filter)
```

**Hypothesis:** Extended scrape delays MAY improve video diversity (TikTok feed refresh), but pool size will be similar.

### 3. Videos Processed (25% Increase)

**Test 2:**
```
Selected: 240 videos (80 per bucket)
Contrastive split: 64 top + 16 bottom per bucket
```

**Test 3:**
```
Selected: 300 videos (100 per bucket)
Contrastive split: 80 top + 20 bottom per bucket
Impact: +25% more training data
```

### 4. ML Model Performance (Expected Improvement)

**Hypothesis:** 100 videos per bucket → better model performance than Test 2 (80 videos)

**Test 2 Baseline:**
```json
{
  "bucket_13-18s": { "rf_accuracy": 0.85, "kmeans_silhouette": 0.42 },
  "bucket_18-33s": { "rf_accuracy": 0.87, "kmeans_silhouette": 0.45 },
  "bucket_33-60s": { "rf_accuracy": 0.90, "kmeans_silhouette": 0.48 }
}
```

**Test 3 Expected (Optimistic):**
```json
{
  "bucket_13-18s": { "rf_accuracy": 0.87, "kmeans_silhouette": 0.45 },  // +2% / +7%
  "bucket_18-33s": { "rf_accuracy": 0.89, "kmeans_silhouette": 0.48 },  // +2% / +7%
  "bucket_33-60s": { "rf_accuracy": 0.92, "kmeans_silhouette": 0.51 }   // +2% / +6%
}
```

**Rationale:** 25% more training data (100 vs 80 videos) should improve model generalization.

### 5. Pipeline Stability Test

**Test 2:** 4-6 hour execution (short-term stability)
**Test 3:** 16-18 hour execution (long-term stability)

**Validation Points:**
- No memory leaks during 180-minute delays
- Checkpoint/resume works after extended delays
- Apify scrapes complete successfully despite long waits
- No network timeouts or connection drops

---

## 📊 Expected Costs

| Resource | Test 2 Cost | Test 3 Cost | Delta | Notes |
|----------|-------------|-------------|-------|-------|
| Apify (scraping) | $0.80 | $0.80 | $0.00 | Same 8 scrapes (delays don't affect cost) |
| Claude API (discovery) | $0.75 | $0.75 | $0.00 | Fresh taxonomy discovery |
| Claude API (classification) | $4.50 | **$5.65** | **+$1.15** | +25% more videos (300 vs 240) |
| Claude API (reports) | $1.50 | $1.50 | $0.00 | Same report count |
| **TOTAL** | **$7.55** | **~$8.70** | **+$1.15** | **+15% cost increase** |

**Cost Efficiency:** 25% more training data for only 15% more cost (good value).

---

## ✅ Success Criteria

### Primary Success Criteria (Same as Test 2)

**Pipeline Completion:**
- ✅ All stages complete without errors over 16-18 hour window
- ✅ Exit code 0 on final completion
- ✅ No "CRITICAL ERROR" logs
- ✅ No timeout errors during 180-minute delays

**Output File Validation:**

| File | Expected Count | Validation |
|------|---------------|------------|
| `cluster_analytics.json` | 1 | Contains scrape_summary, per_hashtag_contribution |
| `winner_analysis.json` | 1 | Contains top_3_buckets array |
| `selected_videos.json` | 3 | **100 videos per bucket** (vs 80 in Test 2) |
| `temporal_windows_updated.json` | **300** | All have temporal_windows.hook/closing |
| `classification.json` | **300** | All have content_category, hook_strategy |
| `aggregated_features.csv` | 3 | Rows match video count (**100 per bucket**) |
| `ml_analysis.json` | 3 | Contains rf_insights, kmeans_insights |
| `wellness_client_report.pdf` | 1 | Readable PDF, >10 pages |

### Secondary Success Criteria (Test 3 Specific)

**Extended Delay Handling:**
```json
{
  "stage1_completion": "no errors during 180-min delays",
  "total_stage1_duration": "12-21 hours (as expected)",
  "scrape_success_rate": "8/8 scrapes successful",
  "no_timeout_errors": true
}
```

**ML Model Quality (Comparison to Test 2):**
```json
{
  "rf_accuracy_improvement": "≥0% vs Test 2",     // At minimum, no degradation
  "kmeans_silhouette_improvement": "≥0% vs Test 2", // Ideally +5-10%
  "training_data_increase": "+25% (300 vs 240 videos)"
}
```

**System Stability:**
```json
{
  "no_memory_leaks": true,
  "checkpoint_resume_works": true,
  "no_connection_drops": true
}
```

---

## 📊 Comparative Analysis: Test 2 vs Test 3

### Timeline Comparison

| Stage | Test 2 (2-min delays, 80 vids) | Test 3 (180-min delays, 100 vids) | Delta |
|-------|--------------------------------|-----------------------------------|-------|
| **Stage 1** | 30-45 min | **12-21 hours** | **+12-20 hours** |
| **Stage 2** | 2-4 hours | **2.5-5 hours** | **+0.5-1 hour** |
| **Stage 2.7** | 15-30 min | **20-40 min** | **+5-10 min** |
| **TOTAL** | 4-6 hours | **16-18 hours** | **+12 hours** |

### Cost Comparison

| Metric | Test 2 | Test 3 | Delta | % Change |
|--------|--------|--------|-------|----------|
| **Apify cost** | $0.80 | $0.80 | $0.00 | 0% |
| **Claude API cost** | $6.75 | $7.90 | +$1.15 | +17% |
| **Total cost** | $7.55 | **$8.70** | **+$1.15** | **+15%** |
| **Videos processed** | 240 | **300** | +60 | **+25%** |
| **Cost per video** | $0.031 | **$0.029** | -$0.002 | **-7% (more efficient)** |

**Key Insight:** Test 3 is more cost-efficient per video ($0.029 vs $0.031).

### ML Training Data Comparison

| Metric | Test 2 | Test 3 | Delta | Impact |
|--------|--------|--------|-------|--------|
| **Videos per bucket** | 80 | **100** | +20 | +25% |
| **Top videos (80%)** | 64 | **80** | +16 | +25% |
| **Bottom videos (20%)** | 16 | **20** | +4 | +25% |
| **Train/test split** | 64/16 | **80/20** | +16/+4 | +25% each |

---

## 🧪 Test Validation Checklist

### Pre-Test Checklist
- [ ] Environment variables set (APIFY_API_KEY, ANTHROPIC_API_KEY)
- [ ] **Code change verified:** `MAX_DELAY_BETWEEN_RUNS_MS = 10800000` in constants.py
- [ ] **Cluster config created:** `config/hashtag_clusters/wellness_test3.json` with 180-min delay
- [ ] Test 1 data exists in `data/clients/Rollo/` (preserved)
- [ ] Test 2 data exists in `data/clients/Rollo_Test2/` (preserved)
- [ ] No existing `data/clients/Rollo_Test3/` directory (fresh start)
- [ ] Apify credits available (>$1)
- [ ] Claude credits available (>$9)
- [ ] **Timeline cleared:** 16-18 hour window available (recommend overnight/weekend run)

### Stage Completion Checklist
- [ ] Stage 0: config.json created
- [ ] **Stage 1: 8/8 scrapes successful (12-21 hour duration)**
- [ ] Stage 1: cluster_analytics.json generated
- [ ] Stage 1: winner_analysis.json shows 3 winning buckets
- [ ] Stage 1: ~1,600 videos after 270-day filter (same as Test 2)
- [ ] **Stage 1: 300 videos selected (100 per bucket)**
- [ ] **Stage 2: 300 temporal_windows files created**
- [ ] Stage 2.5: Files organized into bucket directories
- [ ] Stage 2.6: Fresh taxonomy discovered (wellness_test3_raw_discovery.json)
- [ ] Manual curation: wellness_test3_taxonomy.json created
- [ ] **Stage 2.7: 300 classification.json files created**
- [ ] Stage 3: 3 aggregated_features.csv files (**100 rows each**)
- [ ] Stage 4: 3 transformed_features.pkl files
- [ ] Stage 5: 6 model files (RF + K-Means per bucket)
- [ ] Stage 6: 3 ml_analysis.json files
- [ ] Stage 7: wellness_test3_client_report.pdf generated

### Quality Validation Checklist (Test 3 Specific)
- [ ] **Stage 1: No timeout errors during 180-minute delays**
- [ ] **Stage 1: Total duration 12-21 hours (as expected)**
- [ ] cluster_analytics.json: Duplication rate 20-40%
- [ ] winner_analysis.json: Bucket counts similar to Test 2 (~1,600 videos after filter)
- [ ] Taxonomy: Fresh discovery completed successfully
- [ ] **ML models: RF accuracy ≥ Test 2 baseline** (expected improvement with 100 videos)
- [ ] **ML models: K-Means silhouette ≥ Test 2 baseline**
- [ ] **System stability: No memory leaks or connection issues over 16-18 hours**
- [ ] Client report: Contains insights from 300 videos

---

## 🎯 Test Execution Log Template

```markdown
# Test Execution: E2E-WELLNESS-003

**Date:** YYYY-MM-DD
**Tester:** [Name]
**Environment:** Production / Staging
**Taxonomy Source:** Fresh Discovery (9-month window, 180-min delays)

## Execution Timeline

| Stage | Start Time | End Time | Duration | Status |
|-------|-----------|----------|----------|--------|
| Stage 0 | HH:MM | HH:MM | X min | ✅ / ❌ |
| **Stage 1** | **HH:MM** | **HH:MM** | **X hours** | **✅ / ❌** |
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
- **Videos After Date Filter (270d):** [number] (Compare to Test 2: ~1,600)
- **Stage 1 Total Duration:** [hours] (Expected: 12-21 hours)
- **180-Minute Delays Completed:** [7/7 or X/7]
- **Winning Buckets:** [bucket names]
- **Videos Selected:** [completed]/300 (100 per bucket)
- **Videos Processed:** [completed]/300
- **Videos Classified:** [completed]/300
- **ML Model Accuracy (avg):** [percentage] (Compare to Test 2 baseline)
- **K-Means Silhouette (avg):** [score] (Compare to Test 2 baseline)

## Test 2 vs Test 3 Comparison

| Metric | Test 2 (80 vids) | Test 3 (100 vids) | Delta | Improvement |
|--------|------------------|-------------------|-------|-------------|
| Videos processed | 240 | [actual] | [delta] | [%] |
| Stage 1 duration | 30-45 min | [actual] | [delta] | [hours] |
| RF accuracy (avg) | 0.87 | [actual] | [delta] | [%] |
| K-Means silhouette (avg) | 0.45 | [actual] | [delta] | [%] |
| Cost per video | $0.031 | [actual] | [delta] | [%] |

## Issues Encountered

| Stage | Issue | Resolution | Impact |
|-------|-------|------------|--------|
| [stage] | [description] | [how fixed] | Low/Med/High |

## Test Result

**Overall Status:** ✅ PASS / ❌ FAIL
**Success Criteria Met:** [X]/[Y]
**Comparison to Test 2:** Better / Same / Worse
**180-Min Delay Handling:** Success / Failure
**System Stability (16-18 hrs):** Stable / Unstable
**Recommendation:** [Use Test 2 or Test 3 settings for production? Why?]

**Notes:** [Observations on 180-min delays, 100 videos per bucket, long-term stability]
```

---

## 📚 Related Documentation

- **Test 1 (Baseline):** See above in this document
- **Test 2 (270-day filter):** See above in this document
- **Cluster Setup:** `CLUSTER_QUICK_START.md`
- **Cluster Implementation:** `HASHTAG_CLUSTER_IMPLEMENTATION.md`
- **System Architecture:** `SystemArchitecturev2.md`
- **ML Roadmap:** `MLROADMAP.md`

---

## 📞 Support & Debugging

**Logs Location:**
```bash
tail -f data/logs/rumiai_ml_Rollo_Test3_wellness_test3_*.log
```

**Common Debug Commands:**
```bash
# Monitor Stage 1 progress during 180-minute delays
tail -f data/logs/rumiai_ml_Rollo_Test3_wellness_test3_*.log | grep -E "(Scrape|Waiting|Delay)"

# Check if 180-minute delay is configured correctly
cat config/hashtag_clusters/wellness_test3.json | jq '.scrape_config.delay_between_runs_ms'
# Should output: 10800000

# Compare Test 2 vs Test 3 video counts
echo "Test 2:" && find data/clients/Rollo_Test2 -name "*temporal_windows_updated.json" | wc -l
echo "Test 3:" && find data/clients/Rollo_Test3 -name "*temporal_windows_updated.json" | wc -l
# Test 2 should be 240, Test 3 should be 300

# Compare Test 2 vs Test 3 ML model metrics
echo "Test 2 models:" && cat data/clients/Rollo_Test2/hashtag/wellness/top_contrastive/buckets/*/models/training_summary.json | jq '.metrics'
echo "Test 3 models:" && cat data/clients/Rollo_Test3/hashtag/wellness_test3/top_contrastive/buckets/*/models/training_summary.json | jq '.metrics'

# Check Stage 1 cluster analytics
cat data/clients/Rollo_Test3/hashtag/wellness_test3/cluster_analytics.json | jq '.scrape_summary'

# Verify Stage 1 winner analysis
cat data/clients/Rollo_Test3/hashtag/wellness_test3/top_contrastive/winner_analysis.json | jq '.bucket_stats'
```

**Monitoring During Extended Stage 1:**
```bash
# Run in separate terminal to monitor Stage 1 progress
watch -n 300 'tail -20 data/logs/rumiai_ml_Rollo_Test3_wellness_test3_*.log | grep -E "(Scrape|Waiting|Delay|completed)"'
# Updates every 5 minutes to show scrape progress
```

---

**Test 3 Added:** 2025-10-25
**Author:** RumiAI Testing Team

---
---
---

# Test 4: 4 Runs Per Hashtag + 100 Videos

## 📋 Test Overview

**Test ID:** E2E-WELLNESS-004
**Test Type:** Doubled Scraping Runs + Larger Sample Size Validation
**Client:** Rollo_Test4
**Cluster:** wellness_test4 (4 hashtags, 4 runs per hashtag)
**Objective:** Validate pipeline performance with doubled scraping runs (4 vs 2) and 100 videos per bucket to assess impact of larger video pool on deduplication, selection quality, and ML model training

**Key Differences from Test 2:**
- Client name: `Rollo_Test2` → `Rollo_Test4` (natural test isolation)
- Runs per hashtag: `2` → `4` (+100% more scrapes)
- Video count: `80` → `100` (+25% more videos per bucket)
- Cluster config: Uses `wellness_test4.json` (custom config with 4 runs)
- Expected scrapes: 8 scrapes → **16 scrapes** (4 hashtags × 4 runs)
- Expected raw videos: ~4,800 → **~9,600** (before deduplication)
- Expected Stage 1 duration: ~30-45 min → **~60-90 min** (16 scrapes + 15 delays)
- Expected total pipeline duration: ~4-6 hrs → **~5-7 hours** (excluding manual curation)

---

## 🎯 Test Objectives

This E2E test validates:

1. **Doubled Scraping Impact:** Assess how 4 runs per hashtag (vs 2) affects video pool diversity and deduplication rates
2. **Larger Training Sets:** Validate ML model performance with 100 videos per bucket (vs 80 in Test 2)
3. **Deduplication Efficiency:** Measure overlap rates with doubled scraping (expected higher duplication)
4. **Selection Quality:** Determine if larger pool (post-dedup) improves top/bottom video selection
5. **ML Model Quality:** Compare model metrics against Test 2 baseline (expected improvement with more data)
6. **Cost Efficiency:** Evaluate if 2x scraping cost justifies potential quality gains

**Success Criteria:**
- All pipeline stages complete successfully
- ML model accuracy ≥ Test 2 metrics (RF accuracy >0.75, K-Means silhouette >0.30)
- Deduplication rate analysis shows reasonable overlap (40-60% duplication expected)
- Video processing completes for all 300 videos (100 per bucket)

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

**Test Isolation Strategy:** Use different client name (`Rollo_Test4`) for automatic separation

**Prerequisites:**
- Test 1 data exists in `/data/clients/Rollo/` (preserved)
- Test 2 data exists in `/data/clients/Rollo_Test2/` (preserved)
- Test 3 data exists in `/data/clients/Rollo_Test3/` (preserved)
- Test 4 will create fresh `/data/clients/Rollo_Test4/` directory
- Cluster config exists: `config/hashtag_clusters/wellness_test4.json` ✅ (already created)
- Fresh taxonomy discovery (Stage 2.6 will run and pause for curation)

**Why Different Client Name?**
- ✅ Automatic test isolation (no data mixing)
- ✅ No checkpoint conflicts (fresh start guaranteed)
- ✅ Previous test results preserved for comparison
- ✅ Simpler than archiving (no manual cleanup needed)
- ✅ Side-by-side comparison after Test 4 completes

---

## 📦 Test Configuration

### Cluster Configuration
**File:** `/home/jorge/rumiaifinal/config/hashtag_clusters/wellness_test4.json`

```json
{
  "cluster_id": "wellness_test4",
  "description": "Wellness cluster for Test 4 - 4 runs per hashtag (doubled from Test 2) with 100 videos per bucket",
  "primary_hashtag": "#wellness",
  "variant_hashtags": [
    "#wellnesssupplements",
    "#healthandwellness",
    "#wellnessjourney"
  ],
  "scrape_config": {
    "runs_per_hashtag": 4,
    "delay_between_runs_ms": 120000,
    "results_per_page": 600
  }
}
```

### CLI Parameters

| Parameter | Value | Change from Test 2 | Rationale |
|-----------|-------|---------------------|-----------|
| `--client` | Rollo_Test4 | **CHANGED** (was: Rollo_Test2) | **Natural test isolation** |
| `--target` | wellness_test4 | **CHANGED** (was: wellness) | **Custom cluster config with 4 runs** |
| `--analysis-type` | hashtag | **No change** | Cluster analysis mode |
| `--selection-strategy` | contrastive | **No change** | Top 80% + Bottom 20% |
| `--video-count` | 100 | **CHANGED** (was: 80) | **+25% more videos per bucket** |
| `--date-filter` | last_270_days | **No change** | 9-month window (same as Test 2) |
| `--country-code` | US | **No change** | Geographic filter |
| `--report-type` | single | **No change** | Single hashtag analysis |
| `--report-audience` | client | **No change** | Report format for brand/client |

**Expected Scraping (Doubled volume):**
- 4 hashtags × 4 runs × 600 results = **16 scrapes** (vs Test 2's 8 scrapes)
- **~9,600 videos before deduplication** (vs Test 2's ~4,800)
- **~3,000-4,000 unique videos after deduplication** (vs Test 2's ~1,850)
- ~2,400-3,200 videos after 270-day filter (vs Test 2's ~1,600)

**Expected Date Filter Impact:**
- Test 2: ~1,850 → ~1,600 videos after 270-day filter
- **Test 4: ~3,500 → ~2,800 videos after 270-day filter** (+75% increase)
- **Impact:** Significantly larger pool for bucket selection

**Expected Video Processing:**
- Test 2: 240 videos (80 per bucket)
- **Test 4: 300 videos (100 per bucket)** (+25% increase)
- Contrastive split: **80 top + 20 bottom** per bucket

---

## 🚀 Test Execution

### Command

```bash
cd /home/jorge/rumiaifinal

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

### Expected Runtime

| Stage | Test 2 Duration | Test 4 Expected Duration | Delta | Notes |
|-------|----------------|--------------------------|-------|-------|
| Stage 0 | <5 seconds | <5 seconds | No change | Foundation setup |
| Stage 1 | 30-45 minutes | **60-90 minutes** | **+30-45 min** | **16 scrapes (2x) + 15 delays × 2 min** |
| Stage 2 | 2-4 hours | **2.5-5 hours** | **+0.5-1 hour** | **+25% more videos (300 vs 240)** |
| Stage 2.5 | <10 seconds | <10 seconds | No change | File organization |
| Stage 2.6 | 3-5 minutes | 3-5 minutes | No change | Pattern discovery (fresh) |
| **⏸️ PAUSE** | **Manual** | **1-3 hours** | **Same** | **Fresh taxonomy curation** |
| Stage 2.7 | 15-30 minutes | **20-40 minutes** | **+5-10 min** | **+25% more videos to classify** |
| Stage 3 | 10-20 minutes | **12-25 minutes** | **+2-5 min** | **+25% aggregation work** |
| Stage 4 | 5-10 minutes | **6-12 minutes** | **+1-2 min** | **+25% transformation work** |
| Stage 5 | 10-15 minutes | **12-18 minutes** | **+2-3 min** | **+25% training data** |
| Stage 6 | 5-10 minutes | 5-10 minutes | No change | ML analysis |
| Stage 7 | 20-30 minutes | 20-30 minutes | No change | LLM reports |
| **TOTAL** | **4-6 hours** | **5-7 hours** | **+1 hour** | **(excluding manual curation)** |

**Key Insights:**
- Stage 1 increases by ~30-45 min due to doubled scraping (16 vs 8 scrapes)
- Stage 2+ increases by ~30-45 min due to 25% more videos (300 vs 240)
- Total increase: ~1 hour (+20% longer than Test 2)

---

## 📊 Expected Differences from Test 2

### 1. Scraping Volume (Stage 1 - Doubled)

**Test 2 (2 runs per hashtag):**
```
Scrapes: 8 (4 hashtags × 2 runs)
Raw videos: ~4,800
Delays: 7 × 2 min = 14 minutes
Stage 1 duration: ~30-45 minutes
```

**Test 4 (4 runs per hashtag):**
```
Scrapes: 16 (4 hashtags × 4 runs)
Raw videos: ~9,600 (+100% increase)
Delays: 15 × 2 min = 30 minutes
Stage 1 duration: ~60-90 minutes (+100% increase)
```

**Impact:** Doubled scraping volume to test if more runs improve video diversity.

### 2. Deduplication Analysis (Stage 1 - Critical Metric)

**Test 2 (8 scrapes):**
```
Raw videos: ~4,800
Unique after dedup: ~1,850
Duplication rate: ~61% (expected for 2 runs)
```

**Test 4 (16 scrapes - Hypothesis):**
```
Raw videos: ~9,600
Unique after dedup: ~3,500 (estimated)
Duplication rate: ~64% (expected higher with 4 runs)
Incremental unique videos: +1,650 (+89% increase vs Test 2)
```

**Key Question:** Does doubling scrapes from 2→4 runs provide sufficient new unique videos to justify the cost?

**Expected Outcome:**
- Diminishing returns: 4 runs won't double unique videos (higher overlap)
- But should still provide +70-90% more unique videos for selection

### 3. Video Pool Size After Date Filter (Stage 1)

**Test 2:**
```
After dedup: ~1,850 videos
After 270-day filter: ~1,600 videos
Retention: ~86%
```

**Test 4:**
```
After dedup: ~3,500 videos (estimated)
After 270-day filter: ~2,800 videos (estimated)
Retention: ~80% (similar to Test 2)
Delta: +1,200 videos (+75% increase)
```

**Impact:** Significantly larger pool for bucket selection.

### 4. Bucket Distribution Quality (Stage 1)

**Hypothesis:** Larger pool → more balanced bucket distribution + higher quality top/bottom selection

**Test 2 Example:**
```json
{
  "bucket_13-18s": { "video_count": 280, "top_80_selected": 80, "bottom_20_selected": 20 },
  "bucket_18-33s": { "video_count": 520, "top_80_selected": 80, "bottom_20_selected": 20 },
  "bucket_33-60s": { "video_count": 600, "top_80_selected": 80, "bottom_20_selected": 20 }
}
```

**Test 4 Expected:**
```json
{
  "bucket_13-18s": { "video_count": 450, "top_80_selected": 80, "bottom_20_selected": 20 },  // +170 videos (61% increase)
  "bucket_18-33s": { "video_count": 850, "top_80_selected": 80, "bottom_20_selected": 20 },  // +330 videos (63% increase)
  "bucket_33-60s": { "video_count": 1000, "top_80_selected": 80, "bottom_20_selected": 20 }  // +400 videos (67% increase)
}
```

**Impact:** More competitive selection (top 80 out of 450 vs top 80 out of 280) → higher quality training data.

### 5. Videos Processed (25% Increase - Same as Test 3)

**Test 2:**
```
Selected: 240 videos (80 per bucket)
Contrastive split: 64 top + 16 bottom per bucket
```

**Test 4:**
```
Selected: 300 videos (100 per bucket)
Contrastive split: 80 top + 20 bottom per bucket
Impact: +25% more training data
```

### 6. ML Model Performance (Expected Improvement)

**Hypothesis:** 100 videos per bucket + larger selection pool → better model performance than Test 2

**Test 2 Baseline:**
```json
{
  "bucket_13-18s": { "rf_accuracy": 0.85, "kmeans_silhouette": 0.42 },
  "bucket_18-33s": { "rf_accuracy": 0.87, "kmeans_silhouette": 0.45 },
  "bucket_33-60s": { "rf_accuracy": 0.90, "kmeans_silhouette": 0.48 }
}
```

**Test 4 Expected (Optimistic):**
```json
{
  "bucket_13-18s": { "rf_accuracy": 0.87, "kmeans_silhouette": 0.45 },  // +2% / +7%
  "bucket_18-33s": { "rf_accuracy": 0.89, "kmeans_silhouette": 0.48 },  // +2% / +7%
  "bucket_33-60s": { "rf_accuracy": 0.92, "kmeans_silhouette": 0.51 }   // +2% / +6%
}
```

**Rationale:**
- 25% more training data (100 vs 80 videos)
- Higher quality selection (more competitive pool)
- Should improve model generalization

### 7. Content Diversity (Stage 2.6/2.7)

**Hypothesis:** Larger, more diverse video pool → richer content taxonomy

**Expected Taxonomy Differences:**
- Test 2 (8 scrapes, ~1,600 videos): May miss niche patterns
- Test 4 (16 scrapes, ~2,800 videos): Captures broader range of content styles

**Validation Strategy:**
1. Compare `wellness_test4_raw_discovery.json` to Test 2's discovery
2. Measure taxonomy richness (expected 10-20% more unique patterns)
3. Identify unique categories in Test 4 (niche content types)

---

## 📊 Expected Costs

| Resource | Test 2 Cost | Test 4 Cost | Delta | Notes |
|----------|-------------|-------------|-------|-------|
| Apify (scraping) | $0.80 | **$1.60** | **+$0.80** | **16 scrapes vs 8 scrapes (2x cost)** |
| Claude API (discovery) | $0.75 | $0.75 | $0.00 | Fresh taxonomy discovery (same sample) |
| Claude API (classification) | $4.50 | **$5.65** | **+$1.15** | +25% more videos (300 vs 240) |
| Claude API (reports) | $1.50 | $1.50 | $0.00 | Same report count |
| **TOTAL** | **$7.55** | **~$9.50** | **+$1.95** | **+26% cost increase** |

**Cost Analysis:**
- Apify doubles (+$0.80) due to 2x scrapes
- Claude API increases (+$1.15) due to 25% more videos
- **Total cost efficiency:** 25% more training data for 26% more cost (reasonable)

---

## ✅ Success Criteria

### Primary Success Criteria (Same as Test 2)

**Pipeline Completion:**
- ✅ All stages complete without errors
- ✅ Exit code 0 on final completion
- ✅ No "CRITICAL ERROR" logs

**Output File Validation:**

| File | Expected Count | Validation |
|------|---------------|------------|
| `cluster_analytics.json` | 1 | Contains scrape_summary, per_hashtag_contribution |
| `winner_analysis.json` | 1 | Contains top_3_buckets array |
| `selected_videos.json` | 3 | **100 videos per bucket** (vs 80 in Test 2) |
| `temporal_windows_updated.json` | **300** | All have temporal_windows.hook/closing |
| `classification.json` | **300** | All have content_category, hook_strategy |
| `aggregated_features.csv` | 3 | Rows match video count (**100 per bucket**) |
| `ml_analysis.json` | 3 | Contains rf_insights, kmeans_insights |
| `wellness_test4_client_report.pdf` | 1 | Readable PDF, >10 pages |

### Secondary Success Criteria (Test 4 Specific)

**Scraping & Deduplication:**
```json
{
  "scrapes_completed": "16/16 successful",
  "raw_videos_scraped": ">9000",
  "unique_after_dedup": ">3000",
  "deduplication_rate": "60-70%",  // Expected higher than Test 2 (61%)
  "incremental_unique_videos": ">1500 vs Test 2"  // +70-90% more unique videos
}
```

**Video Pool Quality:**
```json
{
  "videos_after_filter": ">2500",           // 270-day filter
  "filter_retention_rate": "75-85%",        // Similar to Test 2
  "bucket_depth_improvement": ">60%"        // More videos per bucket for selection
}
```

**ML Model Quality (Comparison to Test 2):**
```json
{
  "rf_accuracy_improvement": "≥0% vs Test 2",     // At minimum, no degradation
  "kmeans_silhouette_improvement": "≥0% vs Test 2", // Ideally +5-10%
  "training_data_increase": "+25% (300 vs 240 videos)"
}
```

**Content Diversity:**
```json
{
  "content_categories_discovered": "≥5 categories",
  "taxonomy_richness_vs_test2": "+10-20% more patterns",
  "niche_patterns_detected": "≥2 unique to Test 4"
}
```

---

## 📊 Comparative Analysis: Test 2 vs Test 4

### Scraping & Deduplication Comparison

| Metric | Test 2 (2 runs) | Test 4 (4 runs) | Delta | % Change |
|--------|-----------------|-----------------|-------|----------|
| **Scrapes** | 8 | **16** | +8 | **+100%** |
| **Raw videos** | ~4,800 | **~9,600** | +4,800 | **+100%** |
| **Unique (dedup)** | ~1,850 | **~3,500** | +1,650 | **+89%** |
| **Dedup rate** | ~61% | **~64%** | +3% | **Higher overlap** |
| **After date filter** | ~1,600 | **~2,800** | +1,200 | **+75%** |
| **Videos selected** | 240 | **300** | +60 | **+25%** |

**Key Insight:** Doubling scrapes (2→4 runs) yields ~89% more unique videos, not 100% (diminishing returns due to higher overlap).

### Timeline Comparison

| Stage | Test 2 (2 runs, 80 vids) | Test 4 (4 runs, 100 vids) | Delta |
|-------|--------------------------|---------------------------|-------|
| **Stage 1** | 30-45 min | **60-90 min** | **+30-45 min** |
| **Stage 2** | 2-4 hours | **2.5-5 hours** | **+0.5-1 hour** |
| **Stage 2.7** | 15-30 min | **20-40 min** | **+5-10 min** |
| **TOTAL** | 4-6 hours | **5-7 hours** | **+1 hour** |

### Cost Comparison

| Metric | Test 2 | Test 4 | Delta | % Change |
|--------|--------|--------|-------|----------|
| **Apify cost** | $0.80 | **$1.60** | +$0.80 | **+100%** |
| **Claude API cost** | $6.75 | **$7.90** | +$1.15 | **+17%** |
| **Total cost** | $7.55 | **$9.50** | **+$1.95** | **+26%** |
| **Videos processed** | 240 | **300** | +60 | **+25%** |
| **Cost per video** | $0.031 | **$0.032** | +$0.001 | **+3% (slightly less efficient)** |

**Key Insight:** Test 4 costs 26% more but delivers 25% more training data, making it slightly less cost-efficient per video than Test 2.

### ML Training Data Comparison

| Metric | Test 2 | Test 4 | Delta | Impact |
|--------|--------|--------|-------|--------|
| **Videos per bucket** | 80 | **100** | +20 | +25% |
| **Top videos (80%)** | 64 | **80** | +16 | +25% |
| **Bottom videos (20%)** | 16 | **20** | +4 | +25% |
| **Train/test split** | 64/16 | **80/20** | +16/+4 | +25% each |
| **Selection pool depth** | ~373 avg | **~600 avg** | +227 | **+61% more competitive** |

---

## 🧪 Test Validation Checklist

### Pre-Test Checklist
- [ ] Environment variables set (APIFY_API_KEY, ANTHROPIC_API_KEY)
- [ ] Cluster config exists: `config/hashtag_clusters/wellness_test4.json` ✅
- [ ] Test 1 data exists in `data/clients/Rollo/` (preserved)
- [ ] Test 2 data exists in `data/clients/Rollo_Test2/` (preserved)
- [ ] Test 3 data exists in `data/clients/Rollo_Test3/` (preserved)
- [ ] No existing `data/clients/Rollo_Test4/` directory (fresh start)
- [ ] Apify credits available (>$2)
- [ ] Claude credits available (>$10)

### Stage Completion Checklist
- [ ] Stage 0: config.json created
- [ ] **Stage 1: 16/16 scrapes successful** (vs 8/8 in Test 2)
- [ ] Stage 1: cluster_analytics.json generated
- [ ] Stage 1: **~9,600 raw videos scraped** (vs ~4,800 in Test 2)
- [ ] Stage 1: **~3,500 unique videos after dedup** (vs ~1,850 in Test 2)
- [ ] Stage 1: **Deduplication rate 60-70%** (vs ~61% in Test 2)
- [ ] Stage 1: **~2,800 videos after 270-day filter** (vs ~1,600 in Test 2)
- [ ] Stage 1: winner_analysis.json shows 3 winning buckets
- [ ] **Stage 1: 300 videos selected (100 per bucket)** (vs 240 in Test 2)
- [ ] **Stage 2: 300 temporal_windows files created**
- [ ] Stage 2.5: Files organized into bucket directories
- [ ] Stage 2.6: Fresh taxonomy discovered (wellness_test4_raw_discovery.json)
- [ ] Manual curation: wellness_test4_taxonomy.json created
- [ ] **Stage 2.7: 300 classification.json files created**
- [ ] Stage 3: 3 aggregated_features.csv files (**100 rows each**)
- [ ] Stage 4: 3 transformed_features.pkl files
- [ ] Stage 5: 6 model files (RF + K-Means per bucket)
- [ ] Stage 6: 3 ml_analysis.json files
- [ ] Stage 7: wellness_test4_client_report.pdf generated

### Quality Validation Checklist (Test 4 Specific)
- [ ] **Stage 1: 16 scrapes completed successfully** (doubled from Test 2)
- [ ] **Stage 1: ~9,600 raw videos scraped** (100% increase)
- [ ] **Stage 1: ~3,500 unique videos** (89% increase vs Test 2)
- [ ] **Stage 1: Deduplication rate analysis shows reasonable overlap** (60-70%)
- [ ] cluster_analytics.json: Per-hashtag contribution shows all 4 hashtags contributing
- [ ] winner_analysis.json: **Bucket counts +60-75% vs Test 2** (~600 avg vs ~373 avg)
- [ ] Taxonomy: Fresh discovery completed successfully
- [ ] **Taxonomy richness: +10-20% more patterns vs Test 2** (due to larger pool)
- [ ] **ML models: RF accuracy ≥ Test 2 baseline** (expected improvement with 100 videos)
- [ ] **ML models: K-Means silhouette ≥ Test 2 baseline**
- [ ] Client report: Contains insights from 300 videos

---

## 🎯 Test Execution Log Template

```markdown
# Test Execution: E2E-WELLNESS-004

**Date:** YYYY-MM-DD
**Tester:** [Name]
**Environment:** Production / Staging
**Taxonomy Source:** Fresh Discovery (9-month window, 4 runs per hashtag)

## Execution Timeline

| Stage | Start Time | End Time | Duration | Status |
|-------|-----------|----------|----------|--------|
| Stage 0 | HH:MM | HH:MM | X min | ✅ / ❌ |
| **Stage 1** | **HH:MM** | **HH:MM** | **X min** | **✅ / ❌** |
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

- **Total Scrapes:** [completed]/16 (Expected: 16/16)
- **Total Videos Scraped (raw):** [number] (Expected: ~9,600)
- **Unique Videos After Dedup:** [number] (Expected: ~3,500)
- **Deduplication Rate:** [percentage]% (Expected: 60-70%)
- **Videos After Date Filter (270d):** [number] (Expected: ~2,800)
- **Filter Retention Rate:** [percentage]%
- **Winning Buckets:** [bucket names]
- **Bucket Depth (avg):** [number] (Expected: ~600, vs Test 2: ~373)
- **Videos Selected:** [completed]/300 (100 per bucket)
- **Videos Processed:** [completed]/300
- **Videos Classified:** [completed]/300
- **ML Model Accuracy (avg):** [percentage] (Compare to Test 2 baseline)
- **K-Means Silhouette (avg):** [score] (Compare to Test 2 baseline)
- **Taxonomy Richness vs Test 2:** [number] new patterns / [total] patterns

## Deduplication Analysis (Critical for Test 4)

| Metric | Value | Comparison to Test 2 |
|--------|-------|----------------------|
| Raw videos scraped | [number] | [delta] (+X%) |
| Unique after dedup | [number] | [delta] (+X%) |
| Deduplication rate | [percentage]% | [delta] (+X%) |
| Incremental unique videos | [number] | [delta] (Expected: +1,500 to +1,700) |

**Did doubling scrapes justify the cost?** [Yes/No - Analysis]

## Test 2 vs Test 4 Comparison

| Metric | Test 2 (2 runs, 80 vids) | Test 4 (4 runs, 100 vids) | Delta | Improvement |
|--------|--------------------------|---------------------------|-------|-------------|
| Raw videos | ~4,800 | [actual] | [delta] | [%] |
| Unique videos | ~1,850 | [actual] | [delta] | [%] |
| After filter | ~1,600 | [actual] | [delta] | [%] |
| Videos processed | 240 | [actual] | [delta] | [%] |
| Stage 1 duration | 30-45 min | [actual] | [delta] | [minutes] |
| RF accuracy (avg) | 0.87 | [actual] | [delta] | [%] |
| K-Means silhouette (avg) | 0.45 | [actual] | [delta] | [%] |
| Total cost | $7.55 | [actual] | [delta] | [%] |
| Cost per video | $0.031 | [actual] | [delta] | [%] |

## Issues Encountered

| Stage | Issue | Resolution | Impact |
|-------|-------|------------|--------|
| [stage] | [description] | [how fixed] | Low/Med/High |

## Test Result

**Overall Status:** ✅ PASS / ❌ FAIL
**Success Criteria Met:** [X]/[Y]
**Comparison to Test 2:** Better / Same / Worse
**Deduplication Efficiency:** [Good/Fair/Poor - based on unique video gain]
**Recommendation:** [Use Test 2 or Test 4 settings for production? Why?]

**Notes:** [Observations on 4 runs per hashtag, deduplication rates, 100 videos per bucket impact]
```

---

## 📚 Related Documentation

- **Test 1 (Baseline):** See above in this document
- **Test 2 (270-day filter):** See above in this document
- **Test 3 (180-min delays):** See above in this document
- **Cluster Setup:** `CLUSTER_QUICK_START.md`
- **Cluster Implementation:** `HASHTAG_CLUSTER_IMPLEMENTATION.md`
- **System Architecture:** `SystemArchitecturev2.md`
- **ML Roadmap:** `MLROADMAP.md`

---

## 📞 Support & Debugging

**Logs Location:**
```bash
tail -f data/logs/rumiai_ml_Rollo_Test4_wellness_test4_*.log
```

**Common Debug Commands:**
```bash
# Check cluster config (verify 4 runs)
cat config/hashtag_clusters/wellness_test4.json | jq '.scrape_config'
# Should output: "runs_per_hashtag": 4

# Monitor Stage 1 scraping progress
tail -f data/logs/rumiai_ml_Rollo_Test4_wellness_test4_*.log | grep -E "(Scrape|completed|Waiting)"

# Compare Test 2 vs Test 4 scraping results
echo "Test 2 cluster analytics:" && cat data/clients/Rollo_Test2/hashtag/wellness/cluster_analytics.json | jq '.scrape_summary'
echo "Test 4 cluster analytics:" && cat data/clients/Rollo_Test4/hashtag/wellness_test4/cluster_analytics.json | jq '.scrape_summary'

# Analyze deduplication efficiency
echo "Test 2 dedup:" && cat data/clients/Rollo_Test2/hashtag/wellness/cluster_analytics.json | jq '.scrape_summary | {total_scraped: .total_videos_scraped, unique: .total_unique_videos, dedup_rate: .overall_duplication_rate}'
echo "Test 4 dedup:" && cat data/clients/Rollo_Test4/hashtag/wellness_test4/cluster_analytics.json | jq '.scrape_summary | {total_scraped: .total_videos_scraped, unique: .total_unique_videos, dedup_rate: .overall_duplication_rate}'

# Compare winner analysis (bucket depth)
echo "Test 2 buckets:" && cat data/clients/Rollo_Test2/hashtag/wellness/top_contrastive/winner_analysis.json | jq '.bucket_stats'
echo "Test 4 buckets:" && cat data/clients/Rollo_Test4/hashtag/wellness_test4/top_contrastive/winner_analysis.json | jq '.bucket_stats'

# Compare video counts
echo "Test 2 processed:" && find data/clients/Rollo_Test2 -name "*temporal_windows_updated.json" | wc -l
echo "Test 4 processed:" && find data/clients/Rollo_Test4 -name "*temporal_windows_updated.json" | wc -l
# Test 2 should be 240, Test 4 should be 300

# Compare ML model metrics
echo "Test 2 models:" && cat data/clients/Rollo_Test2/hashtag/wellness/top_contrastive/buckets/*/models/training_summary.json | jq '.metrics'
echo "Test 4 models:" && cat data/clients/Rollo_Test4/hashtag/wellness_test4/top_contrastive/buckets/*/models/training_summary.json | jq '.metrics'

# Compare taxonomies (richness analysis)
echo "Test 2 taxonomy:" && cat data/clients/Rollo_Test2/hashtag/wellness/top_contrastive/content_taxonomies/wellness_taxonomy.json | jq '.content_categories | length'
echo "Test 4 taxonomy:" && cat data/clients/Rollo_Test4/hashtag/wellness_test4/top_contrastive/content_taxonomies/wellness_test4_taxonomy.json | jq '.content_categories | length'
```

**Deduplication Analysis Commands:**
```bash
# Calculate incremental unique videos from Test 4
python3 << EOF
import json

# Load Test 2 analytics
with open('data/clients/Rollo_Test2/hashtag/wellness/cluster_analytics.json') as f:
    test2 = json.load(f)
    test2_unique = test2['scrape_summary']['total_unique_videos']

# Load Test 4 analytics
with open('data/clients/Rollo_Test4/hashtag/wellness_test4/cluster_analytics.json') as f:
    test4 = json.load(f)
    test4_unique = test4['scrape_summary']['total_unique_videos']

# Calculate incremental gain
incremental = test4_unique - test2_unique
pct_gain = (incremental / test2_unique) * 100

print(f"Test 2 unique videos: {test2_unique}")
print(f"Test 4 unique videos: {test4_unique}")
print(f"Incremental gain: +{incremental} videos (+{pct_gain:.1f}%)")
print(f"Diminishing returns: {incremental / (test2_unique)} (should be <1.0 for 2x scrapes)")
EOF
```

---

**Test 4 Added:** 2025-10-27
**Author:** RumiAI Testing Team

---
---


---

# Test 5: Healthy + Fitness Cluster (9 Hashtags)

## 📋 Test Overview

**Test ID:** E2E-WELLNESS-005  
**Test Type:** Comprehensive Fitness-Focused Multi-Hashtag Cluster  
**Client:** Rollo  
**Cluster:** wellnesspt2_test5 (9 hashtags: #healthy + 8 variants)  
**Objective:** Validate fitness-heavy cluster with maximum hashtag coverage and cross-cluster comparison capability

---

## 🎯 Test Objectives

This test expands on the wellness cluster approach with:

1. **Maximum Hashtag Coverage:** 9 hashtags (vs 4 in Test 4) for comprehensive content diversity
2. **Fitness Emphasis:** 35% fitness content (#fitnesstips, #fitnesslifestyle, #healthandfit)
3. **Cross-Cluster Comparison:** Includes #wellness for direct comparison with Test 4
4. **Behavioral Content:** Adds #healthyhabits for routine/habit-focused patterns
5. **Larger Scraping Scale:** 36 scrapes (vs 16 in Test 4) for deeper data pool

**Success Criteria:** Achieve 5,000-6,500 unique videos with balanced fitness/wellness/nutrition coverage

---

## 📦 Test Configuration

### Cluster Configuration
**File:** `/home/jorge/rumiaifinal/config/hashtag_clusters/wellnesspt2_test5.json`

```json
{
  "cluster_id": "wellnesspt2_test5",
  "description": "Comprehensive healthy lifestyle cluster - 9 hashtags covering fitness, wellness, nutrition, and habits",
  "primary_hashtag": "#healthy",
  "variant_hashtags": [
    "#healthylifestyle",
    "#healthyeating",
    "#healthyliving",
    "#fitnesstips",
    "#fitnesslifestyle",
    "#healthandfit",
    "#wellness",
    "#healthyhabits"
  ],
  "scrape_config": {
    "runs_per_hashtag": 4,
    "delay_between_runs_ms": 180000,
    "results_per_page": 600
  }
}
```

### Hashtag Content Distribution

| Category | Hashtags | Expected % |
|----------|----------|------------|
| **Fitness/Exercise** | #fitnesstips, #fitnesslifestyle, #healthandfit | 35% |
| **Wellness/Mindfulness** | #wellness, #healthyhabits | 20% |
| **Food/Nutrition** | #healthyeating | 15% |
| **Lifestyle/Habits** | #healthylifestyle, #healthyliving | 15% |
| **General/Mixed** | #healthy (primary) | 15% |

### CLI Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--client` | Rollo | Test client identifier |
| `--target` | wellnesspt2_test5 | Cluster ID from config file |
| `--analysis-type` | hashtag | Cluster analysis mode |
| `--selection-strategy` | contrastive | Top 80% + Bottom 20% per bucket |
| `--video-count` | 100 | Videos per winning bucket |
| `--date-filter` | last_270_days | 9-month window (same as Test 4) |
| `--country-code` | US | Geographic filter |
| `--report-type` | single | Single hashtag analysis |
| `--report-audience` | client | Report format for brand/client |

**Expected Scraping:**
- 9 hashtags × 4 runs × 600 results = **36 scrapes**
- ~21,600 videos before deduplication
- ~5,000-6,500 unique videos after deduplication (~70-75% retention)
- ~3,000-4,500 videos after 270-day filter
- **300 videos selected** (100 per winning bucket × 3 buckets)

**Scraping Duration:** ~108 minutes (36 scrapes × 3 min/scrape) = **1.8 hours**

---

## 🚀 Test Execution

### Command

```bash
cd /home/jorge/rumiaifinal

python rumiai_ml_batch.py \
  --client Rollo \
  --target wellnesspt2_test5 \
  --analysis-type hashtag \
  --selection-strategy contrastive \
  --video-count 100 \
  --date-filter last_270_days \
  --country-code US \
  --report-type single \
  --report-audience client
```

### Expected Runtime

| Stage | Expected Duration | Notes |
|-------|------------------|-------|
| Stage 0 | <5 seconds | Foundation setup |
| Stage 1 | **1.8-2 hours** | **36 scrapes** (9 hashtags × 4 runs) |
| Stage 2 | 3-5 hours | Video processing (300 videos) |
| Stage 2.5 | <10 seconds | File organization |
| Stage 2.6 | 3-5 minutes | Pattern discovery |
| **⏸️ PAUSE** | **Manual** | **Taxonomy curation** |
| Stage 2.7 | 20-30 minutes | Video classification (300 videos) |
| Stage 3 | 10-20 minutes | Feature aggregation |
| Stage 4 | 5-10 minutes | Feature transformation |
| Stage 5 | 10-15 minutes | ML model training |
| Stage 6 | 5-10 minutes | ML analysis generation |
| Stage 7 | 20-30 minutes | LLM report generation |
| **TOTAL** | **5-8 hours** | **Excluding manual curation** |

**Longer than Test 4 due to:**
- More scrapes (36 vs 16)
- Same video processing count (300 videos)

---

## 📊 Key Differences from Test 4

| Aspect | Test 4 (wellness) | Test 5 (wellnesspt2) |
|--------|-------------------|----------------------|
| **Hashtags** | 4 hashtags | **9 hashtags** |
| **Scrapes** | 16 scrapes | **36 scrapes** |
| **Scraping Time** | ~30-45 min | **~1.8-2 hours** |
| **Focus** | Pure wellness/supplements | **Fitness + wellness mix** |
| **Expected Unique Videos** | 1,500-2,000 | **5,000-6,500** |
| **Content Mix** | Supplements-heavy | **Fitness-heavy (35%)** |
| **Cross-Cluster** | Standalone | **Includes #wellness overlap** |

---

## 🔍 Strategic Value

### 1. Cross-Cluster Comparison
**#wellness appears in both Test 4 and Test 5:**
- Test 4: #wellness as primary focus (supplements, holistic wellness)
- Test 5: #wellness as 1 of 9 hashtags (fitness + wellness context)

**Analysis Opportunity:** Compare how #wellness performs in different cluster contexts

### 2. Fitness Content ML Training
Test 5 is first test with significant fitness content:
- Workout demonstrations
- Exercise form/technique
- Training programs
- Fitness lifestyle vlogs

**ML Value:** Train models on fitness-specific creative patterns

### 3. Behavioral Content (#healthyhabits)
Captures routine-building and habit-stacking content:
- Morning routines
- Daily habits
- Productivity + health
- Behavior change frameworks

### 4. Maximum Diversity
9 hashtags provide:
- Broader creator diversity
- More content style variety
- Better coverage of wellness spectrum

---

## ✅ Success Criteria

### Primary Success Criteria

1. **Stage 1: Video Discovery**
   - [ ] 36/36 scrapes complete successfully
   - [ ] 5,000-6,500 unique videos after deduplication
   - [ ] Deduplication rate: 70-75% retention
   - [ ] 3 winning buckets identified
   - [ ] 300 videos selected (100 per bucket)

2. **Stage 2: Video Processing**
   - [ ] 300/300 videos processed
   - [ ] 0 processing failures
   - [ ] All videos have temporal_windows_updated.json

3. **Stage 2.7: Classification**
   - [ ] >95% classification success (285+/300 videos)
   - [ ] Taxonomy reflects fitness + wellness + nutrition mix

4. **Stage 5: ML Training**
   - [ ] 6 models trained (RF + K-Means × 3 buckets)
   - [ ] Training accuracy >80%
   - [ ] No class imbalance errors

5. **Stage 7: LLM Analysis**
   - [ ] 3 bucket analyses complete
   - [ ] Creative reports generated
   - [ ] Exit code: 0

### Comparison Metrics vs Test 4

| Metric | Test 4 Target | Test 5 Target |
|--------|---------------|---------------|
| Unique videos | 1,500-2,000 | 5,000-6,500 |
| Deduplication rate | 60-65% | 70-75% |
| Scraping time | 30-45 min | 1.8-2 hours |
| Fitness content % | <10% | ~35% |
| Videos processed | 300 | 300 |
| Classification success | >95% | >95% |

---

## 📂 Output Directory Structure

```
data/clients/Rollo/
└── hashtags/
    └── wellnesspt2_test5/
        └── top_contrastive/
            ├── config.json
            ├── cluster_analytics.json  # 36 scrapes documented
            ├── winner_analysis.json
            ├── content_taxonomies/
            │   └── wellnesspt2_test5_taxonomy.json
            └── buckets/
                ├── bucket_{winning_bucket_1}/
                │   ├── selected_videos.json  # 100 videos
                │   ├── top/  # 80 videos
                │   ├── bottom/  # 20 videos
                │   ├── validation/
                │   │   └── video_review.csv
                │   ├── ml_analysis/
                │   │   ├── aggregated_features.json
                │   │   ├── rf_transformed.csv
                │   │   ├── kmeans_transformed.csv
                │   │   ├── *_analysis.json
                │   │   └── llm/
                │   │       ├── hook_analysis.json
                │   │       ├── closing_analysis.json
                │   │       ├── winning_formulas.json
                │   │       └── complete_analysis_{bucket}.json
                │   └── models/
                │       ├── rf_classifier.pkl
                │       ├── kmeans_hook.pkl
                │       └── training_summary.json
                ├── bucket_{winning_bucket_2}/
                └── bucket_{winning_bucket_3}/
```

---
# COMPETITOR ANALYSIS TESTS

# CompetitorTest: @nutrachampssupplement

## 📋 Test Overview

**Test ID:** E2E-COMPETITOR-001  
**Test Type:** Competitor Creative Analysis (Handle-based)  
**Client:** Rollo_test6
**Target:** @nutrachampssupplement (TikTok Handle)  
**Analysis Type:** Competitor  
**Objective:** Analyze competitor's top-performing creative patterns across winning duration buckets

---

## 🎯 Key Differences from Hashtag Tests

| Aspect | Hashtag Tests (Tests 1-4) | Competitor Test |
|--------|---------------------------|-----------------|
| **Analysis Type** | `--analysis-type hashtag` | `--analysis-type competitor` |
| **Target** | Hashtag cluster (e.g., wellness) | TikTok handle (e.g., @nutrachampssupplement) |
| **Selection Strategy** | Contrastive (top 80% + bottom 20%) | **Top performers only** |
| **Video Source** | Multiple hashtags with deduplication | Single creator (all unique videos) |
| **Deduplication** | High (~40-60% duplicates across hashtags) | None (0% - all videos are unique) |
| **Scraping** | Multiple runs with delays | Single comprehensive scrape |
| **Content Diversity** | High (many creators) | Low (single creator style) |

---

## 📦 Test Configuration

### CLI Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--client` | Rollo_test6 | Test client identifier |
| `--target` | nutrachampssupplement | TikTok handle (without @) |
| `--analysis-type` | competitor | Handle-based analysis |
| `--selection-strategy` | top | Top performers only (no contrastive) |
| `--video-count` | 60 | Videos per winning bucket |
| `--date-filter` | last_270_days | 9-month window |
| `--country-code` | US | Geographic filter |
| `--report-type` | single | Single creator analysis |
| `--report-audience` | client | Report format for brand/client |

**Expected Behavior:**
- Single scrape of @nutrachampssupplement's profile
- All videos are unique (0% deduplication)
- Date filter: Last 270 days
- Duration bucketing: Videos distributed across 8 buckets
- Winner analysis: Select top 3 buckets by video count
- Video selection: **Top 60 performers per winning bucket** (no bottom performers)

---

## 🚀 Test Execution

### Command

```bash
cd /home/jorge/rumiaifinal

python rumiai_ml_batch.py \
  --client Rollo \
  --target nutrachampssupplement \
  --analysis-type competitor \
  --selection-strategy top \
  --video-count 60 \
  --date-filter last_270_days \
  --country-code US \
  --report-type single \
  --report-audience client
```
    t5_retention = (t5_unique / t5_scraped) * 100

print(f"Test 4: {t4_scraped} scraped → {t4_unique} unique ({t4_retention:.1f}% retention)")
print(f"Test 5: {t5_scraped} scraped → {t5_unique} unique ({t5_retention:.1f}% retention)")
print(f"Incremental gain: +{t5_unique - t4_unique} unique videos")


### Expected Runtime

| Stage | Expected Duration | Notes |
|-------|------------------|-------|
| Stage 0 | <5 seconds | Foundation setup |
| Stage 1 | 3-5 minutes | Single profile scrape (no multi-run delays) |
| Stage 2 | 1.5-3 hours | Video processing (180 videos = 60 × 3 buckets) |
| Stage 2.5 | <10 seconds | File organization |
| Stage 2.6 | 3-5 minutes | Pattern discovery (competitor-specific taxonomy) |
| **⏸️ PAUSE** | **Manual** | **Taxonomy curation (30-60 min)** |
| Stage 2.7 | 10-20 minutes | Video classification (180 videos) |
| Stage 3 | 5-10 minutes | Feature aggregation |
| Stage 4 | 3-5 minutes | Feature transformation |
| Stage 5 | 5-10 minutes | ML model training |
| Stage 6 | 3-5 minutes | ML analysis generation |
| Stage 7 | 15-20 minutes | LLM report generation |
| **TOTAL** | **2-4 hours** | **Excluding manual curation** |

**Faster than Hashtag Tests because:**
- No multi-run scraping delays (single scrape)
- Fewer total videos (180 vs 240-300)
- No deduplication overhead

---

## 📊 Pipeline Flow & Key Differences

### Stage 1: Video Discovery (Competitor Mode)

**Differences from Hashtag Mode:**

1. **No Cluster Scraping** - Single handle scrape
2. **No Deduplication** - All videos are unique
3. **No Hashtag Validation** - Creator's videos are pre-validated
4. **No Cluster Analytics** - Single-source data

**Expected Output:**
```
✓ Stage 1: Video Discovery - COMPLETE
  Scraping: 1/1 profile scrape successful
  Total videos: ~300-500 (depends on creator)
  Date filter: 300-500 → 180-250 videos (last 270 days)
  Winner buckets: [e.g., bucket_13-18s, bucket_18-33s, bucket_33-60s]
  Selected: 180 videos (60 per bucket, top performers only)
```

**Key Output Files:**
- `winner_analysis.json` - Bucket distribution
- `buckets/bucket_{duration}/selected_videos.json` - Top 60 videos per bucket
- **No cluster_analytics.json** (not applicable for competitor analysis)

### Stage 2.6: Pattern Discovery (Competitor Taxonomy)

**Competitor-Specific Patterns:**
- Single creator's content style
- Consistent visual/audio branding
- Recurring creative frameworks
- Hook patterns specific to creator
- CTA strategies

**Expected Taxonomy Depth:**
- Fewer content categories (5-10 vs 15-25 for hashtags)
- More specific to creator's niche
- Higher pattern consistency

### Stage 5: ML Model Training (Top-Only Strategy)

**Key Difference:**
- Models trained **only on top performers** (no contrastive learning)
- Random Forest predicts "top tier" vs "mid tier" within top 60
- K-Means clusters top performers by creative style
- Insights focus on "what makes their best content work"

---

## 📂 Output Directory Structure

```
data/clients/Rollo/
└── competitor/
    └── nutrachampssupplement/
        └── top/  # Note: "top" not "top_contrastive"
            ├── config.json
            ├── winner_analysis.json
            └── buckets/
                ├── bucket_18-33s/  # Example winning bucket
                │   ├── selected_videos.json  # Top 60 videos
                │   ├── top/
                │   │   └── [60 video JSONs]
                │   ├── validation/
                │   │   └── video_review.csv
                │   ├── ml_analysis/
                │   │   ├── aggregated_features.json
                │   │   ├── rf_transformed.csv
                │   │   ├── kmeans_transformed.csv
                │   │   ├── *_analysis.json  # Stage 6 ML analysis
                │   │   └── llm/  # Stage 7 LLM reports
                │   │       ├── hook_analysis.json
                │   │       ├── closing_analysis.json
                │   │       ├── winning_formulas.json
                │   │       └── complete_analysis_18-33s.json
                │   └── models/
                │       ├── rf_classifier.pkl
                │       ├── kmeans_hook.pkl
                │       └── training_summary.json
                ├── bucket_33-60s/  # Example winning bucket
                │   └── [same structure]
                └── bucket_60-90s/  # Example winning bucket
                    └── [same structure]
```

**Key Directory Differences:**
- Path includes `/competitor/` not `/hashtag/`
- Subdirectory is `/top/` not `/top_contrastive/`
- No `cluster_analytics.json`
- Only `top/` folder per bucket (no `bottom/` folder)

---

## ✅ Success Criteria

### Primary Success Criteria

1. **Stage 1: Video Discovery**
   - [ ] Single profile scrape completes successfully
   - [ ] Videos filtered to last 270 days
   - [ ] 3 winning buckets identified
   - [ ] 60 videos selected per winning bucket (top performers)
   - [ ] `winner_analysis.json` generated

2. **Stage 2: Video Processing**
   - [ ] All 180 videos processed (60 × 3 buckets)
   - [ ] All videos have `temporal_windows_updated.json`
   - [ ] 0 processing failures
   - [ ] Checkpoints enable resume if interrupted

3. **Stage 2.7: Classification**
   - [ ] 180/180 videos classified successfully
   - [ ] Classification matches competitor's content style
   - [ ] Taxonomy reflects single creator patterns

4. **Stage 5: ML Training**
   - [ ] 6 models trained (RF + K-Means × 3 buckets)
   - [ ] Models trained on top performers only
   - [ ] No class imbalance errors
   - [ ] Training metrics > 80% accuracy

5. **Stage 7: LLM Analysis**
   - [ ] 3 bucket analyses generated
   - [ ] Creative reports identify competitor's winning patterns
   - [ ] `complete_analysis_{bucket}.json` for each bucket
   - [ ] Exit code: 0

### Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Videos processed | 180 (100%) | 60 per bucket × 3 buckets |
| Classification success | >95% | 171+/180 videos |
| ML training success | 6/6 models | RF + K-Means per bucket |
| Pipeline completion | Exit code 0 | No errors |
| Processing time | <4 hours | Excluding manual curation |

---

## 🔍 Analysis Use Cases

### What This Test Tells You

1. **Competitor's Winning Duration Strategy**
   - Which video lengths perform best for this creator?
   - Do they focus on short-form (<30s) or long-form (60-90s)?

2. **Creative Patterns Within Top Performers**
   - What creative frameworks do they reuse?
   - Hook patterns, CTA strategies, visual styles
   - Temporal progressions (hook → closing)

3. **Content Strategy Insights**
   - Publishing frequency (via date filter)
   - Topic diversity (via taxonomy)
   - Engagement patterns (via bucket winners)

4. **Benchmarking Against Hashtag Tests**
   - Compare competitor's patterns vs industry (Test 4 wellness cluster)
   - Identify unique strategies vs common patterns
   - Find gaps in competitor's approach

---

## 📊 Comparison Commands

### Compare Competitor vs Hashtag Test 4

```bash
# Video counts
echo "Test 4 (hashtag):" && find data/clients/Rollo_Test4 -name "*temporal_windows_updated.json" | wc -l
echo "CompetitorTest:" && find data/clients/Rollo/competitor/nutrachampssupplement -name "*temporal_windows_updated.json" | wc -l

# Winning buckets
echo "Test 4 winners:" && cat data/clients/Rollo_Test4/hashtags/wellness_test4/top_contrastive/winner_analysis.json | jq '.winner_buckets'
echo "Competitor winners:" && cat data/clients/Rollo/competitor/nutrachampssupplement/top/winner_analysis.json | jq '.winner_buckets'

# ML model performance
echo "Test 4 models:" && cat data/clients/Rollo_Test4/hashtags/wellness_test4/top_contrastive/buckets/*/models/training_summary.json | jq '.metrics'
echo "Competitor models:" && cat data/clients/Rollo/competitor/nutrachampssupplement/top/buckets/*/models/training_summary.json | jq '.metrics'

# LLM creative reports
echo "Test 4 reports:" && cat data/clients/Rollo_Test4/hashtags/wellness_test4/top_contrastive/buckets/*/ml_analysis/llm/winning_formulas.json | jq '.creative_reports | length'
echo "Competitor reports:" && cat data/clients/Rollo/competitor/nutrachampssupplement/top/buckets/*/ml_analysis/llm/winning_formulas.json | jq '.creative_reports | length'
```

---

## ❌ Failure Scenarios & Troubleshooting

### Failure 1: Stage 1 - Profile Not Found

**Error:**
```
ERROR: TikTok profile @nutrachampssupplement not found
```

**Solutions:**
```bash
# Verify handle spelling (common mistake: including @)
# Correct:   --target nutrachampssupplement
# Incorrect: --target @nutrachampssupplement

# Check if profile is public
# Visit: https://www.tiktok.com/@nutrachampssupplement

# Try alternative scraper if primary fails
# (Implementation-specific fallback logic)
```

### Failure 2: Stage 1 - Too Few Videos After Date Filter

**Error:**
```
WARNING: Only 45 videos after date filter (need 60 per bucket × 3 = 180 minimum)
```

**Solutions:**
```bash
# Option 1: Extend date filter
python rumiai_ml_batch.py \
  --date-filter last_365_days \
  # ... other params

# Option 2: Reduce video count per bucket
python rumiai_ml_batch.py \
  --video-count 40 \
  # ... other params

# Option 3: Analyze top 2 buckets instead of 3
# (Automatic: winner analysis will select top N buckets with sufficient videos)
```

### Failure 3: Stage 5 - Not Enough Class Diversity (Top-Only Strategy)

**Error:**
```
ERROR: ML training requires at least 2 classes, but top-only strategy has 1
```

**Note:** This should NOT happen because:
- Top 60 videos are split by RF into "top tier" vs "mid tier"
- K-Means creates natural clusters within top performers
- If this error occurs, it's a bug in the implementation

**Workaround:**
```bash
# Switch to contrastive strategy (not recommended for competitor analysis)
python rumiai_ml_batch.py \
  --selection-strategy contrastive \
  # ... other params
```

---

## 📝 Test Execution Log Template

```markdown
## CompetitorTest Execution Log

**Date:** [YYYY-MM-DD]
**Executed By:** [Name]
**Command:** 
```bash
python rumiai_ml_batch.py --client Rollo --target nutrachampssupplement --analysis-type competitor --selection-strategy top --video-count 60 --date-filter last_270_days --country-code US --report-type single --report-audience client
```

### Stage Results

| Stage | Status | Duration | Notes |
|-------|--------|----------|-------|
| Stage 0 | ✅ PASS | [time] | Foundation setup |
| Stage 1 | ✅ PASS | [time] | [X] videos scraped, [Y] after date filter, [Z] buckets |
| Stage 2 | ✅ PASS | [time] | [X]/[Y] videos processed |
| Stage 2.5 | ✅ PASS | [time] | File organization |
| Stage 2.6 | ✅ PASS | [time] | [X] patterns discovered |
| Stage 2.7 | ✅ PASS | [time] | [X]/[Y] videos classified |
| Stage 3 | ✅ PASS | [time] | Features aggregated |
| Stage 4 | ✅ PASS | [time] | Features transformed |
| Stage 5 | ✅ PASS | [time] | [X] models trained |
| Stage 6 | ✅ PASS | [time] | ML analysis generated |
| Stage 7 | ✅ PASS | [time] | LLM reports generated |

### Winning Buckets

| Bucket | Video Count | Top 60 Selected |
|--------|-------------|-----------------|
| [duration] | [count] | ✅ Yes |
| [duration] | [count] | ✅ Yes |
| [duration] | [count] | ✅ Yes |

### Final Output

- **Videos Processed:** [X]/[Y]
- **Classification Success:** [X]% ([Y]/[Z] videos)
- **Models Trained:** [X]/6
- **Creative Reports:** [X] reports across [Y] buckets
- **Exit Code:** 0 ✅
- **Total Duration:** [X] hours [Y] minutes

### Issues Encountered

| Issue | Stage | Severity | Resolution |
|-------|-------|----------|------------|
| [description] | [stage] | Low/Med/High | [how fixed] |

### Key Insights

**Competitor's Winning Patterns:**
- [Pattern 1]
- [Pattern 2]
- [Pattern 3]

**Comparison to Test 4 (Wellness Hashtag):**
- [Similarity/Difference 1]
- [Similarity/Difference 2]

**Recommendations:**
- [Recommendation 1]
- [Recommendation 2]
```

---

## 📚 Related Documentation

- **Competitor Analysis Implementation:** `COMPETITOR_ANALYSIS_IMPLEMENTATION.md` (if exists)
- **Test 4 (Hashtag Baseline):** See above in this document
- **System Architecture:** `SystemArchitecturev2.md`
- **ML Roadmap:** `MLROADMAP.md`

---

**CompetitorTest Added:** 2025-10-28  
**Author:** RumiAI Testing Team  
**Status:** Ready for execution


