# Analysis Mode System

**Document Type**: High-Level Design (HLD)
**Technical Implementation**: See [MLAnalysisModeTI.md](./MLAnalysisModeTI.md) for code, integration details, and testing
**Last Updated**: 2025-10-01

---

## Overview

### Business Problem
Different business questions require analyzing different video sets:
- **"What works?"** → Analyze top-performing content (highest engagement)
- **"What's happening now?"** → Analyze most recent content (current trends/strategy)

Without dual-mode support, users would need to run separate analyses or manually filter videos, reducing flexibility and insight quality.

### Solution
Implement `--analysis-mode` flag with two options:
- `top`: Sort by engagement, analyze highest-performing videos
- `recent`: Sort by publish date, analyze most recent videos

### Stakeholder Value
- **Tumi Labs**: Single system handles multiple analysis types (ML training, trend monitoring, strategy tracking)
- **Brands**: Understand both "what works historically" and "what's trending now"
- **Competitive Intelligence**: Track rival's best work AND detect strategy shifts

---

## The Two Modes

### Top Mode (`--analysis-mode top`)

**What it does**: Analyzes highest-engagement videos within date filter

**How it works**:
1. Apify scrapes videos with `sortBy: engagement`
2. Videos sorted by composite score: `views × share_boost_factor`
3. Takes top N videos by engagement
4. Analyzes patterns that correlate with success

**Use cases**:
- **Hashtag**: Train ML models on viral patterns ("what makes #nutrition content go viral?")
- **Competitor**: Benchmark rival's best-performing content ("what works for @rival_brand?")
- **Creator**: Understand creator's peak performance style (for coaching)

**Output insight**: "These creative patterns correlate with high engagement"

---

### Recent Mode (`--analysis-mode recent`)

**What it does**: Analyzes most recently published videos within date filter

**How it works**:
1. Apify scrapes videos with `sortBy: date`
2. Videos sorted by `createTime` (newest first)
3. Takes most recent N videos
4. Analyzes current content strategy

**Use cases**:
- **Hashtag**: Detect trend shifts ("are #nutrition creators posting more long-form now?")
- **Competitor**: Track strategy changes ("has @rival_brand changed their approach?")
- **Creator**: Understand natural production style (for vetting)

**Output insight**: "This is what's being produced right now"

---

## Default Modes per Analysis Type

| Analysis Type | Default Mode | Reasoning |
|---------------|--------------|-----------|
| **Hashtag** | `top` | ML training needs viral patterns, not just recent posts |
| **Competitor** | `top` | Benchmark rival's best work first, track trends optionally |
| **Creator** | `recent` | Vetting needs natural style, not cherry-picked best |

**CLI behavior**:
```bash
# If --analysis-mode not specified, uses defaults above
python rumiai_ml_batch.py --analysis-type hashtag --target "#nutrition"
# Automatically uses: --analysis-mode top

python rumiai_ml_batch.py --analysis-type creator --target "@affiliate"
# Automatically uses: --analysis-mode recent
```

---

## Apify Integration

### Top Mode - Engagement Sorting

**Apify Scraper Parameters**:
```json
{
  "hashtagsUrls": ["#nutrition"],
  "resultsPerPage": 300,
  "shouldDownloadVideos": true,
  "sortBy": "engagement",
  "sortOrder": "desc"
}
```

**Post-Processing**:
- Calculate composite engagement score: `views × (1 + share_rate × 10)`
- Sort videos by engagement score descending
- Select top N videos

**Implementation**: See [MLAnalysisModeTI.md - Engagement Score Calculation](./MLAnalysisModeTI.md#21-date-filter-implementation)

---

### Recent Mode - Date Sorting

**Apify Scraper Parameters**:
```json
{
  "hashtagsUrls": ["#nutrition"],
  "resultsPerPage": 300,
  "shouldDownloadVideos": true,
  "sortBy": "date",
  "sortOrder": "desc"
}
```

**Post-Processing**:
- Sort videos by `createTime` descending (newest first)
- Select most recent N videos

**Implementation**: See [MLAnalysisModeTI.md - Date Sorting](./MLAnalysisModeTI.md#21-date-filter-implementation)

---

## Engagement Score Calculation

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

### Rationale

**Why not just use views?**
- High views don't always mean replicable patterns
- Some videos get views from paid promotion, not organic virality
- Shares indicate content people actively want to spread

**Why weight shares heavily?**
- **Shares = Viral Indicator**: People only share content they believe others will engage with
- **Quality Signal**: Shares require more commitment than likes (social risk)
- **Amplification Factor**: Shared content reaches new audiences organically
- **TikTok Algorithm**: Share rate heavily influences "For You" page placement

**The 10x multiplier:**
- A video with 1% share rate (10,000 shares / 1M views) gets 10% boost
- A video with 2% share rate gets 20% boost
- Typical TikTok share rates: 0.5-2% (viral videos: 3-5%+)

### Examples

| Video | Views | Shares | Share Rate | Share Boost | Engagement Score |
|-------|-------|--------|------------|-------------|------------------|
| **A** (High views, low shares) | 1,000,000 | 2,000 | 0.2% | 1.02 | 1,020,000 |
| **B** (Medium views, high shares) | 500,000 | 15,000 | 3% | 1.30 | 650,000 |
| **C** (Lower views, exceptional shares) | 300,000 | 15,000 | 5% | 1.50 | 450,000 |

**Key Insight**: Video A ranks highest by engagement score, but videos B and C have stronger viral indicators (share rate). The formula balances both reach (views) and viral potential (shares).

### Sorting Logic

1. Calculate engagement score for each video
2. Sort by engagement score descending (highest first)
3. Use `createTime` as tiebreaker for identical scores

**Implementation**: See [MLAnalysisModeTI.md - Engagement Sorting](./MLAnalysisModeTI.md#21-date-filter-implementation)

### Data Source

All metrics come from Apify TikTok scraper:

```json
{
  "playCount": 3200000,      // → views
  "diggCount": 346500,       // → likes
  "commentCount": 872,        // → comments
  "shareCount": 15500         // → shares
}
```

**Reliability**:
- ✅ Apify always provides these metrics (core TikTok data)
- ✅ If missing, video is skipped (not processed)
- ✅ Point-in-time snapshot (sufficient for pattern analysis)

### Alternative Considerations

**Simple Engagement Rate (Not Used)**:
- Formula: `(likes + comments + shares) / views`
- **Why not**: Treats all interactions equally; doesn't weight shares higher; lower correlation with viral patterns

**Likes + Comments Weighting (Future Enhancement)**:
- Add weighted multipliers: shares (10x), comments (5x), likes (1x)
- **Status**: Not implemented yet (keeping formula simple for MVP)

### Usage in ML Pipeline

**Hashtag Analysis** (contrastive):
1. Calculate engagement scores for all videos
2. Sort by engagement score
3. Bucket by duration (8 buckets)
4. Select top 40 + bottom 20 per bucket

**Competitor Analysis** (all videos):
1. Calculate engagement scores for all videos
2. Sort by engagement score
3. Process all videos (no top/bottom selection)

**Implementation**: See [MLAnalysisModeTI.md - Video Selection](./MLAnalysisModeTI.md#5-integration-example)

### Quality Filters

Before calculating engagement score, apply minimum thresholds:

**Thresholds**:
- **Minimum 1,000 views**: Ensures statistical significance
- **Minimum 2% basic engagement rate**: Filters "dead" content (bots, low-quality)

**Implementation**: See [MLAnalysisModeTI.md - Engagement Filtering](./MLAnalysisModeTI.md#21-date-filter-implementation)

### Validation & Testing

**Test Cases**:
- Zero shares → no boost (score = views × 1.0)
- 1% share rate → 10% boost (score = views × 1.10)
- 5% share rate → 50% boost (score = views × 1.50)
- Zero views → handled gracefully (score = 0)

**Implementation**: See [MLAnalysisModeTI.md - Testing Scripts](./MLAnalysisModeTI.md#6-testing-scripts)

---

## How Mode Affects Each Analysis Type

### Hashtag Analysis

#### Top Mode (Default)
```bash
python rumiai_ml_batch.py \
  --analysis-type hashtag \
  --target "#nutrition" \
  --video-count 300 \
  --date-filter "last_90_days" \
  --analysis-mode top
```

**Process**:
1. Scrape 300 highest-engagement #nutrition videos from last 90 days
2. Bucket by duration (8 buckets)
3. Select top 40 + bottom 20 per bucket (contrastive analysis)
4. Train ML models on viral patterns
5. Generate reports: "What makes #nutrition content go viral?"

**Insight**: "15-18s videos with joy_ratio > 0.6 and fast cuts have 3x engagement"

---

#### Recent Mode (Optional)
```bash
--analysis-mode recent
```

**Process**: Same but analyzes most recent 300 posts (regardless of engagement)

**Insight**: "Trend shift detected: 60% of recent #nutrition posts are 60-90s storytelling (was 20% three months ago)"

**Use case**: Quarterly trend reports, detect market shifts

---

### Competitor Analysis

#### Top Mode (Default)
```bash
python rumiai_ml_batch.py \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode top
```

**Process**:
1. Scrape 150 highest-engagement videos from @rival_brand
2. Bucket by duration
3. Process ALL videos (no top/bottom selection)
4. Analyze creative patterns in their best work
5. Generate report: "What works for @rival_brand?"

**Insight**: "@rival_brand's top videos average 0.75 energy_level and 42 words in hook"

---

#### Recent Mode
```bash
--analysis-mode recent
```

**Process**: Same but analyzes most recent 150 posts

**Insight**: "Strategy shift: @rival_brand recently increased 13-18s content from 30% to 65%"

**Use case**: Monthly competitor monitoring, detect strategy changes

---

### Creator Analysis

#### Recent Mode (Default)
```bash
python rumiai_ml_batch.py \
  --analysis-type creator \
  --target "@potential_affiliate" \
  --video-count 40 \
  --analysis-mode recent \
  --compare-to hashtag:nutrition
```

**Process**:
1. Scrape most recent 40 videos from @potential_affiliate
2. Bucket by duration
3. Calculate distribution (what they naturally produce)
4. Compare to client's hashtag patterns
5. Generate compatibility score + hiring recommendation

**Insight**: "Creator naturally produces 55% in 13-18s bucket (matches client's 45% viral distribution) - STRONG FIT"

---

#### Top Mode (Optional)
```bash
--analysis-mode top
```

**Process**: Same but analyzes top 40 videos by engagement

**Insight**: "Creator's best work: 60-90s storytelling (avg 250K views), but rarely produces this format (10% of content)"

**Use case**: Understanding creator's peak performance for coaching (less useful for vetting)

---

## Mode Comparison Examples

### Scenario: Competitor Strategy Shift Detection

**Client**: Acme Nutrition brand wants to track rival's strategy

**Approach**: Run both modes monthly

```bash
# Month 1: Baseline (Top Mode)
python rumiai_ml_batch.py \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode top

# Output: Best work is 13-18s with high energy (0.75)
```

```bash
# Month 2: Track recent strategy (Recent Mode)
python rumiai_ml_batch.py \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode recent

# Output: Recent posts shifted to 60-90s storytelling (strategy change detected)
```

**Action**: Acme adjusts their content strategy based on rival's pivot

---

### Scenario: Creator Vetting (Recent) vs Coaching (Top)

**Client**: Wants to vet @fitness_jane for affiliate program

**Step 1: Vetting (Recent Mode)**
```bash
python rumiai_ml_batch.py \
  --analysis-type creator \
  --target "@fitness_jane" \
  --video-count 40 \
  --analysis-mode recent \
  --compare-to hashtag:nutrition

# Output: Natural style = 55% in 13-18s bucket
# Compatibility score: 0.82 (STRONG FIT)
# Recommendation: Tier 1 - Immediate Hire
```

**Step 2: Coaching (Top Mode)** - After hiring
```bash
--analysis-mode top

# Output: Best performing videos = 0.68 joy_ratio, 15 text overlays
# Coach creator: "Your viral videos have higher joy_ratio than your average (0.35)"
```

---

## Implementation Design

### CLI Flag Handling

**Default Mode Selection**:
- Hashtag/Competitor → `top` (analyze best work)
- Creator → `recent` (analyze natural style)
- User can override with explicit `--analysis-mode` flag

**Implementation**: See [MLAnalysisModeTI.md - CLI Argument Parsing](./MLAnalysisModeTI.md#3-cli-argument-parsing)

---

### Apify Scraper Integration

**Two Scrapers Required**:
- **Hashtag scraper**: clockworks/tiktok-hashtag-scraper
- **Profile scraper**: clockworks/tiktok-scraper (current)

**Sorting Strategy**:
- Top mode → `sortBy: engagement`
- Recent mode → `sortBy: date`

**Implementation**: See [MLAnalysisModeTI.md - Apify Client](./MLAnalysisModeTI.md#1-apify-client-implementation)

---

### Checkpoint Integration

**Config Validation on Resume**:
- Checkpoint stores: `video_count`, `date_filter`, `analysis_mode`
- Resume validation: Must match or error with `--force` suggestion
- Prevents resuming with different mode than started with

**Implementation**: See [MLCheckpointResumeTI.md - CheckpointManager](./MLCheckpointResumeTI.md)

---

## Edge Cases & Handling

### Case 1: Not Enough High-Engagement Videos
**Scenario**: Requesting top 300 videos, but only 150 meet thresholds
**Handling**: Use all available videos, log warning

### Case 2: Creator Has Deleted Recent Videos
**Scenario**: Recent mode requests 40 videos, but only 30 available
**Handling**: Process all available, adjust compatibility confidence score

### Case 3: Date Filter Eliminates All Videos
**Scenario**: Date filter too restrictive (e.g., last_7_days but no recent posts)
**Handling**: Error with helpful suggestions (expand date range, check account status)

### Case 4: Engagement Ties
**Scenario**: Multiple videos with identical engagement scores
**Handling**: Use `createTime` as tiebreaker (newer first)

**Implementation**: See [MLAnalysisModeTI.md - Edge Case Handling](./MLAnalysisModeTI.md#21-date-filter-implementation)

---

## Reporting Differences by Mode

### Top Mode Reports

**Focus**: "What works" (pattern → outcome)

**Example excerpt**:
```
## Hook Analysis (13-18s bucket)

Top-performing videos (avg 250K views) show:
- Joy ratio: 0.68 (vs bottom 20: 0.32) → +112% engagement
- Text overlays in first 3s: 85% (vs bottom: 45%) → +89% retention
- Energy level: 0.75 (vs bottom: 0.42) → +78% shares

Recommendation: Prioritize joyful energy and immediate text overlays in 13-18s content.
```

---

### Recent Mode Reports

**Focus**: "What's happening" (trend → strategy)

**Example excerpt**:
```
## Recent Content Strategy Shift (Last 30 days)

Duration distribution change:
- 13-18s: 65% (was 45% three months ago) → +44% increase
- 60-90s: 15% (was 35% three months ago) → -57% decrease

Creative pattern changes:
- Average energy level: 0.82 (was 0.65) → More dynamic editing
- Text overlay density: +40% increase → Heavier use of captions

Interpretation: Market shifting toward short-form, high-energy content with heavy text overlays.
```

---

## Testing Strategy

### Unit Tests Required

**Test Coverage**:
- ✅ Analysis mode defaults per analysis type (hashtag→top, creator→recent)
- ✅ Engagement score calculation (various share rates)
- ✅ Date sorting (newest first)
- ✅ Engagement sorting (highest first)
- ✅ Edge cases (zero views, missing data)

**Implementation**: See [MLAnalysisModeTI.md - Testing Scripts](./MLAnalysisModeTI.md#6-testing-scripts)

---

## Future Enhancements

### Mixed Mode Analysis
Compare top vs recent in single run:
```bash
--analysis-mode both
```
Output: Side-by-side comparison showing strategy shifts

---

### Custom Sorting
Allow custom engagement formulas:
```bash
--analysis-mode top --engagement-formula "views * 0.5 + shares * 2 + comments * 1.5"
```

---

### Time-Weighted Recent
Exponentially decay older videos in recent mode:
```bash
--analysis-mode recent --time-decay 0.95
```

---

## Summary

### Key Decisions
- ✅ **Two modes**: `top` (engagement) and `recent` (date)
- ✅ **Smart defaults**: hashtag/competitor use `top`, creator uses `recent`
- ✅ **Apify integration**: `sortBy` parameter controls video ordering
- ✅ **Flexible use**: Same infrastructure, different business questions

### Success Metrics
- **Flexibility**: One command switch changes analysis purpose
- **Intelligence**: Different insights based on mode (patterns vs trends)
- **Business value**: Answers "what works?" AND "what's happening now?"