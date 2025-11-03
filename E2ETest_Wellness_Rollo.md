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
- [CompetitorTest: @vitalproteins](#competitortest-vitalproteins) - Vital Proteins collagen brand analysis, top 80 per bucket, 270-day window

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
## 📚 Related Documentation

- **Cluster Setup:** `CLUSTER_QUICK_START.md`
- **Cluster Implementation:** `HASHTAG_CLUSTER_IMPLEMENTATION.md`
- **System Architecture:** `SystemArchitecturev2.md`
- **ML Roadmap:** `MLROADMAP.md`
- **Quick Reference:** `QUICK_REFERENCE.md`

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

# Test 3: 180-Minute Scrape Delay + 100 Videos

## 📋 Test Overview

**Test ID:** E2E-WELLNESS-003
**Test Type:** Extended Scrape Delay + Larger Sample Size Validation
**Client:** Rollo_Test3
**Cluster:** wellness_test3 (4 hashtags, custom delay config)
**Objective:** Validate pipeline performance with extended 180-minute scrape delays and 100 videos per bucket to test rate-limiting avoidance and improved ML model training with larger datasets

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


## 📦 Test Configuration

### Cluster Configuration
**File:** `/home/jorge/rumiaifinal/config/hashtag_clusters/wellness_test3.json`

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

## CompetitorTest: @nutrachampssupplement

### 📋 Test Overview

**Test ID:** E2E-COMPETITOR-001  
**Test Type:** Competitor Creative Analysis (Handle-based)  
**Client:** Rollo_test6
**Target:** @nutrachampssupplement (TikTok Handle)  
**Analysis Type:** Competitor  
**Objective:** Analyze competitor's top-performing creative patterns across winning duration buckets

---

### 📦 Test Configuration

#### CLI Parameters

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

## CompetitorTest: vitalproteins

### 📋 Test Overview

**Test ID:** E2E-COMPETITOR-002
**Test Type:** Competitor Creative Analysis (Handle-based)
**Client:** Rollo_test7
**Target:** @vitalproteins (TikTok Handle)
**Analysis Type:** Competitor
**Objective:** Analyze competitor's top-performing creative patterns across winning duration buckets

---

### 📦 Test Configuration

#### CLI Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--client` | Rollo_test7 | Test client identifier |
| `--target` | vitalproteins | TikTok handle (without @) |
| `--analysis-type` | competitor | Handle-based analysis |
| `--selection-strategy` | top | Top performers only (no contrastive) |
| `--video-count` | 80 | Videos per winning bucket |
| `--date-filter` | last_270_days | 9-month window |
| `--country-code` | US | Geographic filter |
| `--report-type` | single | Single creator analysis |
| `--report-audience` | client | Report format for brand/client |

**Expected Behavior:**
- Single scrape of @vitalproteins's profile
- All videos are unique (0% deduplication)
- Date filter: Last 270 days
- Duration bucketing: Videos distributed across 8 buckets
- Winner analysis: Select top 3 buckets by video count
- Video selection: **Top 80 performers per winning bucket** (no bottom performers)

---

## CompetitorTest: drinkolipop

### 📋 Test Overview

**Test ID:** E2E-COMPETITOR-003
**Test Type:** Competitor Creative Analysis (Handle-based)
**Client:** Rollo_test8
**Target:** @drinkolipop (TikTok Handle)
**Analysis Type:** Competitor
**Objective:** Analyze competitor's top-performing creative patterns across winning duration buckets

---

### 📦 Test Configuration

#### CLI Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--client` | Rollo_test8 | Test client identifier |
| `--target` | drinkolipop | TikTok handle (without @) |
| `--analysis-type` | competitor | Handle-based analysis |
| `--selection-strategy` | top | Top performers only (no contrastive) |
| `--video-count` | 80 | Videos per winning bucket |
| `--date-filter` | last_270_days | 9-month window |
| `--country-code` | US | Geographic filter |
| `--report-type` | single | Single creator analysis |
| `--report-audience` | client | Report format for brand/client |

**Expected Behavior:**
- Single scrape of @drinkolipop's profile
- All videos are unique (0% deduplication)
- Date filter: Last 270 days
- Duration bucketing: Videos distributed across 8 buckets
- Winner analysis: Select top 3 buckets by video count
- Video selection: **Top 80 performers per winning bucket** (no bottom performers)

---
