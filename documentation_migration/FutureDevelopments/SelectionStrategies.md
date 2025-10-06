# Selection Strategies

**Document Type**: High-Level Design (HLD)
**Parent Document**: MLPlanning.md
**Technical Implementation**: [SelectionStrategiesTI.md](./SelectionStrategiesTI.md) (future)
**Last Updated**: 2025-01-06

---

## Overview

### What are Selection Strategies?

Selection strategies determine **what videos to analyze** after sorting. They control which subset of videos enters the ML training pipeline, directly impacting the type of insights generated, processing resources required, and statistical validity of results.

Selection strategies are **orthogonal to analysis modes**. Analysis mode (top/recent) controls HOW videos are sorted, while selection strategy controls WHAT subset is analyzed. This separation allows flexible combinations: you can analyze top-performing videos with contrastive selection, or recent videos with all-inclusive selection.

Three strategies exist because different business questions require different analytical approaches. The choice involves trade-offs between insight depth (contrastive learning), processing speed (top-only), and comprehensiveness (complete distribution).

### Why Three Strategies?

**Business Rationale**:
- **Different questions require different data**: "What makes winners different?" needs contrastive data; "What do winners do?" only needs top performers
- **Resource optimization**: Processing 40 videos (top) is 33% faster than 60 videos (contrastive), enabling budget-conscious analyses
- **Statistical validity**: Not all markets have enough videos for contrastive analysis; strategies adapt to data availability
- **ML model requirements**: Classification models (Random Forest) require contrastive data; clustering models work with top-only data

---

## The Three Strategies

### Contrastive Strategy

**Purpose**: Identify creative patterns that differentiate success from failure

**What It Does**:
- **Phase 1**: Scrape 800 videos sorted by engagement (captures full performance spectrum)
- **Phase 2**: Analyze top 100 performers to identify winning formats (which buckets do winners cluster in?)
- **Phase 3**: Select top 3 buckets where winners concentrate (success-based distribution, not volume-based)
- **Phase 4**: Per winning bucket selection - Top 80% + Bottom 20% of N videos in that bucket
- Enables contrastive learning: "What makes top videos different from less successful videos?"
- Trains classification models (Random Forest)

**Example** (Success-based bucket selection from #nutrition):
```
Top 100 performers (out of 800 scraped):
- Bucket 18-33s: 45 videos (45% of winners) ← Selected bucket #1
- Bucket 33-60s: 30 videos (30% of winners) ← Selected bucket #2
- Bucket 13-18s: 20 videos (20% of winners) ← Selected bucket #3
- Bucket 9-13s:  5 videos  (5% of winners)  ← Not selected (despite high volume)

Result: Process 18-33s, 33-60s, 13-18s (where winners cluster)

Per bucket (e.g., 18-33s bucket, N=100):
- Top 80: Rank #1-80 in bucket (500K-200K views - top performers)
- Bottom 20: Rank #81-100 in bucket (50K-100K views - less successful within winners)
```

**What "Bottom" Actually Means**:
- **NOT true failures** (1K-5K views) - Apify sorts by engagement DESC, so we don't get low performers
- **"Less successful within successful sample"** - Videos that made it to top 800 but ranked lower
- **Good enough contrast**: 500K vs 20K views is meaningful (differentiates "great" from "average")
- **Actionable patterns**: Still captures "what separates top performers from the rest"

**Business Value**:
- **Actionable insights**: "High eye contact correlates with top performance (0.85 vs 0.45)"
- **Causal patterns**: Not just "what top videos do" but "what top videos do DIFFERENTLY"
- **Replicable success**: Patterns that distinguish winners from average performers
- **What to avoid**: Bottom performer analysis reveals patterns associated with lower engagement
- **Focus on winning formats**: Only process top 3 buckets where winners cluster (success-based, not volume-based)
- **Skip underperforming formats**: Ignore high-volume buckets if winners don't cluster there

**Resource Requirements**:
- Single scrape: 800 videos (engagement sorted, full performance spectrum)
- Winner analysis: Top 100 performers analyzed to identify winning bucket distribution
- Per bucket: Variable (depends on where winners cluster)
  - Typical: Top 3 winning buckets selected
  - Selection: Top 80% + Bottom 20% of N per bucket (N from --video-count, default 100)
  - May process with fewer videos if winning bucket has insufficient volume (user warned)
- Processing time: Highest (more videos, complex ML)
- Statistical validity: Requires sufficient samples for both groups (≥ N per bucket ideal, flexible for winners)
- Cost: Highest - $4 Apify (800 videos) + processing + LLM calls

**ML Output**:
- Random Forest classification model per bucket
- Feature importance rankings (which features separate top from bottom)
- Top vs bottom pattern analysis
- Predictive capability: Can classify new videos as likely top/bottom performers

**Use Cases**:
- Hashtag analysis (market research)
- Training content creators (what to replicate vs avoid)
- A/B testing insights (what differentiates winning variants)

**When to Use**:
- Market has sufficient video volume (≥ N per bucket, where N from --video-count)
- Goal is understanding "what works vs what doesn't"
- Resources available for comprehensive analysis
- Need classification models for future prediction

**User Configuration**:

**CLI Parameter**: `--video-count N` (default: 100)

**How It Works**:
- System scrapes 800 videos from hashtag (engagement sorted DESC via Apify)
- Analyzes top 100 performers: Buckets them by duration to see where winners cluster
- Identifies top 3 buckets by success concentration (what % of top 100 fall into each bucket?)
- Per winning bucket: Selects top 80% + bottom 20% of N from all videos in that bucket
  - Example: N=100 → 80 top + 20 bottom
  - Example: N=150 → 120 top + 30 bottom

**Flexibility**:
- N is fully configurable by user via CLI
- Success-based selection: Focuses on buckets where winners cluster (not just high-volume buckets)
- Adaptive threshold: If winning bucket has < N total videos, process anyway with warning
  - Example: Bucket 33-60s has 40% of winners but only 50 total videos (< N=100)
  - System processes all 50 videos and warns user about smaller sample size

---

### Top Strategy

**Purpose**: Identify best practices from successful content only

**What It Does**:
- Analyzes top 100 performers to identify winning formats (same logic as Contrastive)
- Selects top 3 buckets where winners cluster (success-based, not volume-based)
- Per winning bucket: Selects top N performers only (no bottom performers)
- Analyzes patterns in successful content
- Clustering/descriptive analysis (no classification)

**Business Value**:
- **Faster processing**: Analyzes only top performers (no bottom group)
- **Focus on winning formats**: Only processes buckets where winners cluster
- **Success patterns**: "What do top performers have in common?"
- **Benchmark standards**: Identify industry best practices in winning formats
- **Cost efficiency**: Lower processing costs than contrastive (fewer videos per bucket)

**Resource Requirements**:
- Single scrape: 800 videos (engagement sorted, same as Contrastive)
- Winner analysis: Top 100 performers analyzed to identify winning bucket distribution
- Per bucket: Top N performers only (N from --video-count, default 40)
  - Typical: Top 3 winning buckets selected
  - May process with fewer videos if winning bucket has insufficient volume (user warned)
- Processing time: Medium (fewer videos per bucket than contrastive)
- Statistical validity: Sufficient for pattern identification
- Cost: Medium - $4 Apify (800 videos, same as Contrastive) + lower processing (fewer videos/bucket)

**ML Output**:
- K-Means clustering (creative pattern groups)
- Descriptive statistics (averages, distributions)
- Success pattern analysis
- No predictive models (clustering is descriptive, not predictive)

**Use Cases**:
- Competitor analysis (understand rivals' best work)
- Quick market scans (faster than contrastive)
- Budget-conscious analyses (lower cost)
- Benchmarking exercises (industry standards)

**When to Use**:
- Market has moderate video volume (≥ N per bucket, where N from --video-count)
- Goal is "what do winners do?" not "what makes winners different?"
- Speed/cost priority over depth
- Classification not needed (clustering sufficient)

**User Configuration**:

**CLI Parameter**: `--video-count N` (default: 40)

**How It Works**:
- System scrapes 800 videos from target (engagement sorted DESC via Apify)
- Analyzes top 100 performers: Buckets them by duration to see where winners cluster
- Identifies top 3 buckets by success concentration (what % of top 100 fall into each bucket?)
- Per winning bucket: Selects top N performers only (no bottom performers)
  - Example: N=40 → top 40 videos
  - Example: N=60 → top 60 videos

**Flexibility**:
- N is fully configurable by user via CLI
- Success-based selection: Focuses on buckets where winners cluster (not just high-volume buckets)
- Adaptive threshold: If winning bucket has < N total videos, process anyway with warning
  - Example: Bucket 33-60s has 40% of winners but only 50 total videos (< N=40)
  - System processes all 50 videos and warns user about smaller sample size

**Limitation**:
- Cannot answer "what to AVOID" (no bottom performers analyzed)
- Less statistically rigorous than contrastive (no control group)
- No predictive models (cannot classify future videos)

---

## Adaptive Bucket Processing

### The Business Problem

**Market Reality**: Winners cluster in specific formats
- Typical hashtag: 75% of top 100 performers fall into 3 duration buckets
- Processing all 8 buckets equally misses the winning formats

**Example** (Success-based distribution):
```
Top 100 performers from 800 scraped:
- Bucket 4 (18-33s):  45 videos (45% of winners) → Process ✅
- Bucket 5 (33-60s):  30 videos (30% of winners) → Process ✅
- Bucket 3 (13-18s):  20 videos (20% of winners) → Process ✅
- Bucket 2 (9-13s):    5 videos  (5% of winners)  → Skip ❌ (despite high volume)
```

**Problems with volume-based processing**:
1. **Wrong focus**: High-volume formats may underperform (9-13s: 400 videos but only 5 winners)
2. **Resource waste**: Analyzing formats that don't drive results
3. **Misleading insights**: "What works in 9-13s" when winners cluster in 18-60s
4. **Business misalignment**: Focus should be where winners cluster, not where volume is

**A/B Testing Analogy**:
- In A/B testing, you optimize winning variants and kill losing variants
- Applied to buckets: Focus on winning formats (where top 100 cluster), skip underperforming formats
- Why train ML models for 9-13s if only 5% of winners use it?

---

### Solution Approach

**Principle**: Focus resources on formats where winners cluster (success-based, not volume-based)

**Strategy-Specific Thresholds**:

| Strategy | Minimum Videos | Rationale |
|----------|---------------|-----------|
| **Contrastive** | Flexible | Ideally ≥ N for 80/20 selection. If winning bucket has < N videos, process anyway with warning |
| **Top** | Flexible | Ideally ≥ N top performers. If winning bucket has < N videos, process anyway with warning |

**Note**: Both Contrastive and Top strategies use success-based bucket selection (analyze where winners cluster, not volume)

**Adaptive Logic**:
1. Scrape 800 videos (engagement sorted DESC)
2. Analyze top 100 performers: Bucket them by duration
3. Calculate winner concentration: What % of top 100 fall into each bucket?
4. Select top 3 buckets by success concentration (where winners cluster)
5. Per winning bucket: Process with requested N (or all available videos if < N, with warning)
6. Report explains which buckets were selected and why others were skipped

**Example Outcome** (Success-based selection):
```
Scraped: 800 videos from #nutrition

Top 100 performers distribution:
- Bucket 18-33s: 45 videos (45% of winners) → Process ✅
- Bucket 33-60s: 30 videos (30% of winners) → Process ✅
- Bucket 13-18s: 20 videos (20% of winners) → Process ✅
- Bucket 9-13s:  5 videos  (5% of winners)  → Skip ❌

✓ Processing bucket_18-33s: 45% of winners cluster here (120 total videos in bucket)
✓ Processing bucket_33-60s: 30% of winners cluster here (50 total videos in bucket)
  ⚠ Warning: Only 50 videos available (requested N=100). Processing all 50.
✓ Processing bucket_13-18s: 20% of winners cluster here (80 total videos in bucket)
✗ Skipping bucket_9-13s: Only 5% of winners (high volume but low success rate)

Result: 95% winner coverage with 3 buckets (vs 8 buckets)
```

---

### Business Impact

**Cost & Time Savings**:
- Typical hashtag: Process 2-3 buckets instead of 8
- **75% cost reduction**: $2.40 vs $7.60 (LLM calls)
- **75% time reduction**: 2 hours vs 8 hours
- **Same or better insights**: Focus on statistically valid data

**Quality Improvement**:
- No misleading insights from underperforming formats
- Clear communication about winning formats
- Strategic guidance: "Focus on 18-60s where winners cluster" (not where volume is)
- Avoid analyzing high-volume but low-success formats

**Resource Optimization**:
- Skip video processing for underperforming buckets (even if high volume)
- Skip RumiAI processing for formats that don't drive results
- Skip LLM calls for buckets with < 5% winner concentration

**Market Intelligence**:
- Winner analysis reveals actual success patterns (not just popularity)
- Success concentration = strong performance signal ("18-60s drives results")
- Clients can align strategy with winning formats, not just popular ones
- Identify blue ocean opportunities (underutilized high-performing formats)

---

## Limitations & Trade-offs

### Contrastive
**Pros**:
- Deepest insights (what makes winners different)
- Causal patterns (not just correlation)
- Classification models (predictive capability)
- "What to avoid" insights (bottom performer analysis)

**Cons**:
- Highest resource requirements (N videos/bucket, default N=100)
- Needs large samples (≥ N per bucket)
- Slowest processing (most videos, both top and bottom)
- Highest cost (most LLM calls)

---

### Top
**Pros**:
- Faster processing (analyzes only top performers, no bottom group)
- Cheaper (fewer videos than contrastive, typical N=40 vs N=100)
- Still actionable (success patterns identified)
- Configurable sample size (N from --video-count, default 40)

**Cons**:
- No "what to avoid" insights (no bottom performers)
- No classification models (clustering only)
- Less statistically rigorous (no control group)
- Cannot predict future performance

---

## Future Enhancements

### Performance-Weighted Selection
**Goal**: Process high-engagement buckets even if lower volume

**Logic**:
- Calculate average engagement per bucket
- Process buckets with engagement > 1.5x median, even if volume < threshold
- Example: "33-60s has only 5% of videos but 2x avg engagement → process it"

**Business Value**: Identify underserved high-performing formats (market gaps)

---

### Tiered Processing
**Goal**: Different analysis depth based on video volume

**Tiers**:
- **HIGH** (≥80 videos): Full contrastive analysis
- **MEDIUM** (40-79 videos): Top-only analysis
- **LOW** (10-39 videos): Descriptive stats only (no ML)
- **SKIP** (< 10 videos): No processing

**Business Value**: Balanced approach (deep analysis where it matters, light touch on edges)

---

### Dynamic Thresholds
**Goal**: Adjust thresholds based on total video count

**Logic**:
- If scraping 1000 videos, raise threshold to 100 (higher bar for processing)
- If scraping 100 videos, lower threshold to 30 (adapt to data availability)

**Business Value**: Context-aware processing (what's "sufficient" depends on dataset size)

---

## Cross-References

**Related Documents**:
- [MLPlanning.md](./MLPlanning.md) - Parent document (CLI Configuration section)
- [MLAnalysisMode.md](./MLAnalysisMode.md) - Analysis mode implementation (how videos are sorted)
- [SelectionStrategiesTI.md](./SelectionStrategiesTI.md) - Technical implementation (future)
- [AdaptiveBucketProcessing.md](./AdaptiveBucketProcessing.md) - Detailed adaptive logic (future TI)

**Implementation Details** (future TI docs):
- Video selection algorithms
- Threshold calculation logic
- Distribution analysis code
- Edge case handling


# Sorting Strategies

## Apify Integration

**Two Scrapers Required**:
- **Hashtag scraper**: clockworks/tiktok-hashtag-scraper
- **Profile scraper**: clockworks/tiktok-scraper (current)

**Sorting Strategy**:
- Top mode → `sortBy: engagement`

**Apify Scraper Parameters**:
```json
{
  "hashtagsUrls": ["#nutrition"],
  "resultsPerPage": 800,
  "shouldDownloadVideos": true,
  "sortBy": "engagement",
  "sortOrder": "desc"
}
```

**Date Filtering**:
- **Server-side**: Not supported by Apify hashtag scraper
- **Client-side**: System filters scraped videos by `create_time` field after retrieval
- **Format**: `--date-filter last_N_days` (e.g., `last_90_days`, `last_30_days`)
- **Process**: Scrape 800 videos → filter by date → bucket by duration → select top N per bucket
- **Default**: `last_90_days` for hashtag/competitor, `last_30_days` for creator

## Engagement Score Calculation

**Post-Processing**:
- Calculate composite engagement score: `views × (1 + share_rate × 10)`
- Sort videos by engagement score descending
- Select top N videos

**Implementation**: See [MLAnalysisModeTI.md - Engagement Score Calculation](./MLAnalysisModeTI.md#21-date-filter-implementation)


### Overview

For `--analysis-mode top`, videos are ranked by **engagement score** - a composite metric that identifies viral potential beyond simple view counts.

### Formula

**Composite Engagement Score**:
```
engagement_score = views × (1 + share_rate × 10)

where:
  share_rate = shares / views
  share_boost = 1 + (share_rate × 10)
```

### 3.1 Apify Scraper Investigation & Recommendations

**Date**: 2025-10-01
**Investigation Goal**: Determine optimal Apify scraper(s) for ML batch processing requirements

#### Current Scraper Setup

**Actor ID**: `GdWCkxBtKWOsKjdch` (clockworks/tiktok-scraper)
**Current Usage**: Single video scraping via `postURLs` parameter
**Integration**: `/home/jorge/rumiaifinal/rumiai_v2/api/apify_client.py`

#### Requirements Analysis

Based on MLAnalysisMode.md and system architecture:

**Video Volume per Analysis**:
- Hashtag analysis: 300+ videos → 480 processed (60 per bucket × 8 buckets)
- Competitor analysis: 150 videos → all processed
- Creator analysis: 40 videos → all processed

**Required Apify Parameters**:
1. **Sorting capability**: `sortBy: engagement` (top mode) OR `sortBy: date` (recent mode)
2. **Volume support**: 400-800 videos per scrape (hashtag volume with filtering headroom)
3. **Target types**: Both hashtag URLs (`#nutrition`) AND profile URLs (`@handle`)
4. **Video downloads**: `shouldDownloadVideos: true`
5. **Date filtering**: Client-side filtering required (no server-side for hashtags)

**Required Metadata Fields** (from video.py:35-84):
- ✅ `playCount` → views
- ✅ `diggCount` → likes
- ✅ `commentCount` → comments
- ✅ `shareCount` → shares
- ✅ `createTime` / `createTimeISO` → create_time
- ✅ `videoUrl` / `downloadAddr` → download_url
- ✅ `duration` → duration
- ✅ `authorMeta.name` → username

#### Scraper Comparison Matrix

| Feature | clockworks/tiktok-scraper | clockworks/tiktok-hashtag-scraper | Recommendation |
|---------|---------------------------|-----------------------------------|----------------|
| **Profile scraping** (@handle) | ✅ Yes (`profilesUrls`) | ❌ No | Need current scraper |
| **Hashtag scraping** (#tag) | ⚠️ Limited (`postURLs` only) | ✅ Yes (`hashtagsUrls`) | Need hashtag scraper |
| **Sorting by engagement** | ❓ Unknown (needs testing) | ✅ Yes (`sortBy: engagement`) | Hashtag scraper better |
| **Sorting by date** | ❓ Unknown (needs testing) | ✅ Yes (`sortBy: date`) | Hashtag scraper better |
| **Volume limit** | 400-800 per hashtag | 400-800 per hashtag | Both sufficient |
| **Metadata fields** | ✅ All required fields | ✅ All required fields | Both sufficient |
| **Video downloads** | ✅ Yes | ✅ Yes | Both sufficient |
| **Cost per video** | ~$0.005 | ~$0.005 | Same |

#### Critical Findings

**✅ GOOD NEWS - Existing Integration Works**:
- Current `VideoMetadata.from_apify_data()` already handles all required fields
- Engagement scoring formula ready (MLAnalysisMode.md lines 153-176)
- Service contracts guarantee valid data structures

**⚠️ LIMITATION DISCOVERED - Hashtag Volume**:
- **Hard limit**: 400-800 videos per hashtag (TikTok platform limitation, not Apify)
- **Impact**: When scraping 800 videos for hashtags, after client-side date filtering, may have <480 videos needed
- **Mitigation**: Accept fewer videos per bucket OR relax date constraints

**❌ CRITICAL GAP - Date Filtering**:
- ❌ No server-side date filtering for hashtag searches
- ✅ Date filtering only available for profile scraping
- **Workaround**: Must scrape 800 videos, then filter client-side