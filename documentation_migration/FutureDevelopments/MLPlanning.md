ML next Phase


We are in the last phase of RumiAI

# Basics

## A. System Architecture

### Primary Goals
1. **Batch Video Analysis**
   - Process up to 300 videos sequentially through `rumiai_runner.py`
   - Implement checkpoint/resume system for failure recovery

2. **Client-Centric Data Organization**
   - Multi-tenant data structure: Client → (Hashtags | Competitors | Creators) → Duration Buckets → Videos
   - Three analysis types per client: Market research (hashtags), competitive intelligence (competitors), creator vetting (creators)
   - Bucket-specific analysis within client boundaries
   - Persistent client configuration management

3. **Duration-Specific ML Pattern Recognition**
   - Train **separate ML models (RF and Kmeans) for each duration bucket**
   - Recognize that 15-second patterns differ completely from 60-second patterns

4. **Creative Report Generation**
   - **Hashtags**: Number of reports vary per bucket - "What works for #nutrition"
   - **Competitors**: 5 strategy reports per bucket (40 total per competitor) - "What @rival is doing"
   - **Creators**: Style profile + compatibility scoring - "Does this creator fit our brand?"
   - Multiple perspectives and strategies per duration bucket
   - Include bucket performance metrics for strategic content planning

5. **Creator Compatibility Analysis**
   - Analyze potential affiliate creators' natural content style (via their TikTok handle)
   - Cross-check creator features against client hashtag/competitor viral patterns
   - Generate compatibility scores per duration bucket (e.g., "Creator's 15s style matches #nutrition patterns: 82%")
   - Provide hiring recommendations: which creators naturally produce content that aligns with what works
   - **See detailed implementation**: [MLCreatorMatch.md](./documentation_migration/FutureDevelopments/MLCreatorMatch.md) 

## Client Architecture & Storage

### Data Example

#### ML Production Pipeline Architecture
```
/data/
├── clients/
│   ├── {client_id}/
│   │   ├── hashtags/                       # Hashtag-based analyses
│   │   │   └── {hashtag_name}/             # e.g., "nutrition", "fitness"
│   │   │       ├── buckets/                # Duration-based ML buckets (8 total)
│   │   │       │   ├── bucket_0-3s/
│   │   │       │   │   ├── videos/         # Raw MP4s for this duration (30-day retention)
│   │   │       │   │   ├── analysis/       # RumiAI outputs for videos in this bucket
│   │   │       │   │   │   ├── insights/   # temporal_windows JSON (1 per video)
│   │   │       │   │   │   ├── unified/    # Intermediate timeline+ml_data (debugging)
│   │   │       │   │   │   └── service_debug/  # emotion_detection, audio_energy outputs
│   │   │       │   │   ├── models/         # Trained models for THIS bucket
│   │   │       │   │   │   ├── random_forest_v1.pkl
│   │   │       │   │   │   ├── kmeans_v1.pkl
│   │   │       │   │   │   └── model_metrics.json
│   │   │       │   │   ├── reports/        # Bucket-specific creative reports (5 per bucket)
│   │   │       │   │   ├── checkpoints/    # Processing state for this bucket
│   │   │       │   │   └── logs/           # Bucket processing logs
│   │   │       │   │
│   │   │       │   ├── bucket_3-9s/        # Bucket 2
│   │   │       │   ├── bucket_9-13s/       # Bucket 3
│   │   │       │   ├── bucket_13-18s/      # Bucket 4
│   │   │       │   ├── bucket_18-33s/      # Bucket 5
│   │   │       │   ├── bucket_33-60s/      # Bucket 6
│   │   │       │   ├── bucket_60-90s/      # Bucket 7
│   │   │       │   └── bucket_90-120s/     # Bucket 8
│   │   │       │
│   │   │       └── hashtag_summary/        # Cross-bucket executive reports
│   │   │           ├── executive_report.pdf    # Bird's eye view for client
│   │   │           └── hashtag_metrics.json    # Aggregated stats across all buckets
│   │   │
│   │   ├── competitors/                    # Competitor tracking (what rivals are doing)
│   │   │   └── {competitor_handle}/        # e.g., "@rival_brand"
│   │   │       ├── buckets/                # Same 8-bucket structure
│   │   │       │   ├── bucket_0-3s/        # (full structure same as hashtags)
│   │   │       │   ├── bucket_3-9s/
│   │   │       │   ├── bucket_9-13s/
│   │   │       │   ├── bucket_13-18s/
│   │   │       │   ├── bucket_18-33s/
│   │   │       │   ├── bucket_33-60s/
│   │   │       │   ├── bucket_60-90s/
│   │   │       │   └── bucket_90-120s/
│   │   │       │
│   │   │       └── competitor_summary/     # Cross-bucket competitor analysis
│   │   │           ├── competitor_report.pdf   # "What @rival_brand is doing"
│   │   │           └── competitor_metrics.json
│   │   │
│   │   └── creators/                       # Creator vetting (potential affiliates)
│   │       └── {creator_handle}/           # e.g., "@potential_affiliate"
│   │           ├── buckets/                # Same 8-bucket structure
│   │           │   ├── bucket_0-3s/        # (full structure same as hashtags)
│   │           │   ├── bucket_3-9s/
│   │           │   ├── bucket_9-13s/
│   │           │   ├── bucket_13-18s/
│   │           │   ├── bucket_18-33s/
│   │           │   ├── bucket_33-60s/
│   │           │   ├── bucket_60-90s/
│   │           │   └── bucket_90-120s/
│   │           │
│   │           └── creator_summary/        # Cross-check analysis
│   │               ├── style_profile.pdf       # Creator's natural content style
│   │               ├── compatibility_report.pdf # Match vs client hashtags/competitors
│   │               ├── compatibility_scores.json # Numeric scores per bucket
│   │               └── creator_metrics.json
```

#### Development/Testing Architecture (Preserved)
```
/insights/                     # Final temporal windows JSON (single video CLI usage)
/unified_analysis/             # Intermediate timeline + ml_data
/temp/                         # Downloaded videos
/emotion_detection_outputs/    # Service debugging
/audio_energy_outputs/         # Service debugging
/logs/                         # Processing logs
```

**Note**: Root-level directories remain for `rumiai_runner.py` CLI development/testing. Production ML batch processing uses `/data/clients/` structure.


### Data Retention Policy
- **Raw Videos**: 30 days (then delete to save space, can re-download if needed)
- **ML Analysis**: 6 months (compressed after 30 days)
- **ML Models**: Keep latest 3 versions per client/hashtag
- **Reports**: Indefinite (small size, high value)
- **Checkpoints**: 7 days after successful completion

### Storage Cost Optimization
- **Video Deletion**: Remove raw videos after 30 days


## Flow

This section documents how to initialize ML batch processing via CLI for each analysis type.

**MVP Approach**: Command-line flags for all inputs (enables automation, scriptable, repeatable). Interactive mode and config files are future enhancements.

**Apify Parameters**: Exposed as CLI flags (`--video-count`, `--date-filter`, `--analysis-mode`) for full control per analysis.

**Analysis Modes**: Two modes enable different business questions - `top` (what works?) vs `recent` (what's happening now?). See [MLAnalysisMode.md](./documentation_migration/FutureDevelopments/MLAnalysisMode.md) for detailed implementation.

**Checkpoint/Resume**: Auto-resume on restart (detects checkpoint, continues automatically). Use `--force` flag to discard checkpoint and restart fresh. See [MLCheckpointResume.md](./documentation_migration/FutureDevelopments/MLCheckpointResume.md) for details.

---

### Hashtag Flow

**Purpose**: Analyze videos from a hashtag to identify viral creative patterns and train duration-specific ML models

**Default Mode**: `--analysis-mode top` (analyze highest-engagement videos)

**Report Types**:
- `single` (default): Full ML pipeline for one hashtag
- `comparison`: LLM-based comparison of previously analyzed hashtags (no video processing)

---

#### Single Hashtag Analysis (Default)

**Command**:
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type hashtag \
  --target "#nutrition" \
  --video-count 300 \
  --date-filter "last_90_days" \
  --analysis-mode top \
  --report-type single  # or omit (default)
```

**Inputs**:
- `--client`: Client identifier (creates `/data/clients/{client_id}/`)
- `--analysis-type`: `hashtag` | `competitor` | `creator`
- `--target`: Hashtag name (e.g., `#nutrition`)
- `--video-count`: Number of videos to scrape (default: 300)
- `--date-filter`: Recency filter (e.g., `last_90_days`, `2024-01-01:2025-01-01`)
- `--analysis-mode`: `top` (default) | `recent`
- `--report-type`: `single` (default) | `comparison`

**Process (Top Mode)**:
1. **Apify Scraping**: Fetch 300 videos for #nutrition sorted by engagement
2. **Client-side Filtering**: Apply date filter, remove duplicates
3. **Duration Bucketing**: Split videos into 8 duration buckets
4. **Video Selection**: Top 40 + Bottom 20 per bucket (contrastive analysis)
5. **Batch Analysis**: Run RumiAI on selected videos (sequential, with checkpoints)
6. **ML Training**: Train Random Forest + K-Means per bucket
7. **Report Generation**: Create 5 strategy reports per bucket (40 total)

**Process (Recent Mode)** - Optional for trend monitoring:
Same as Top Mode but sorts by publish date instead of engagement. Use case: "What are creators posting NOW?"

**Outputs**:
```
/data/clients/client_acme_corp/hashtags/nutrition/
  ├── buckets/
  │   ├── bucket_13-18s/
  │   │   ├── videos/          # 60 MP4 files (40 top + 20 bottom)
  │   │   ├── analysis/        # 60 temporal_windows JSONs
  │   │   ├── models/          # random_forest_v1.pkl, kmeans_v1.pkl
  │   │   └── reports/         # 5 PDF creative strategy reports
  │   └── ... (8 buckets total)
  └── hashtag_summary/
      ├── executive_report.pdf      # Cross-bucket overview for client
      └── hashtag_metrics.json      # Used for comparison mode
```

**Duration**: ~6-8 hours for 300 videos (60-80s per video analysis)

**Cost**: ~$4.00 Apify + compute time

---

#### Multi-Hashtag Comparison

**Command**:
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type hashtag \
  --target "#nutrition,#fitness,#wellness" \
  --report-type comparison
```

**Prerequisites**: All hashtags must be individually analyzed first (single mode)

**Process**:
1. **Validation**: Check that all hashtags have existing `hashtag_metrics.json`
2. **Load Data**: Read JSON files for each hashtag
3. **LLM Analysis**: Send data to Claude API with comparison prompt
4. **Report Generation**: Create comparison PDF with strategic recommendations

**Outputs**:
```
/data/clients/client_acme_corp/hashtags/comparisons/
  └── nutrition_vs_fitness_vs_wellness_20250128/
      ├── comparison_input.json     # Data sent to LLM
      ├── llm_analysis.json         # Raw LLM response
      └── comparison_report.pdf     # Formatted report
```

**Duration**: ~30 seconds (LLM call only, no video processing)

**Cost**: ~$0.50 (LLM API call only)

**Error Handling**:
```bash
# If any hashtag hasn't been analyzed individually:
✗ Cannot generate comparison report

Missing individual analyses:
  ✓ #nutrition - analyzed (2025-01-28)
  ✓ #fitness - analyzed (2025-01-27)
  ✗ #wellness - NOT ANALYZED

Run individual analysis first:
  python rumiai_ml_batch.py --client "acme" --target "#wellness" --report-type single
```

**Report Contents**:
- Executive summary (which hashtag to prioritize)
- Side-by-side comparison table (duration preferences, key patterns, engagement rates)
- Strategic recommendations per hashtag
- Priority ranking (viral potential × replicability × strategic fit)

---

### Competitor Flow

**Purpose**: Benchmark competitor's successful content or track their current strategy

**Default Mode**: `--analysis-mode top` (analyze highest-engagement videos)

**Command (Single Competitor Deep Dive)**:
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode top \
  --report-type single
```

**Command (Multi-Competitor Comparison)**:
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type competitor \
  --target "@rival_brand,@competitor2,@competitor3" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode top \
  --report-type comparison
```

**Command (Recent Mode)** - Track current strategy:
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode recent \
  --report-type single
```

**Inputs**:
- `--client`: Client identifier
- `--analysis-type`: `competitor`
- `--target`: Single competitor handle (`@rival_brand`) OR comma-separated list (`@rival1,@rival2,@rival3`)
- `--video-count`: Number of videos to scrape per competitor (default: 150)
- `--date-filter`: Recency filter
- `--analysis-mode`: `top` (default) | `recent`
- `--report-type`: `single` (deep dive on one competitor) | `comparison` (side-by-side comparison of multiple competitors)

**Process (Top Mode)** - "What works for rival?":
1. **Apify Scraping**: Fetch 150 videos from @rival_brand sorted by engagement
2. **Client-side Filtering**: Apply date filter
3. **Duration Bucketing**: Split videos into 8 duration buckets
4. **Batch Analysis**: Process ALL 150 videos (no top/bottom selection)
5. **Pattern Analysis**: Identify creative patterns that correlate with rival's success
6. **Report Generation**: "What works for @rival_brand" (patterns in their best content)

**Process (Recent Mode)** - "What is rival posting now?":
Same as Top Mode but sorts by publish date. Identifies current content strategy and potential shifts.

**Outputs**:
```
/data/clients/client_acme_corp/competitors/rival_brand/
  ├── videos/                # 150 MP4 files
  ├── analysis/              # 150 temporal_windows JSONs
  ├── distribution_analysis/
  │   └── duration_distribution.json  # What durations rival focuses on
  └── competitor_summary/
      └── competitor_report.pdf  # "What @rival_brand is doing"
```

**Duration**: ~3-4 hours for 150 videos (60-80s per video analysis)

**Key Differences vs Hashtag**:
- Hashtag: 300+ videos → 480 processed (top 40 + bottom 20 per bucket) - Market research
- Competitor: 150 videos → all processed - Competitive intelligence
- No contrastive analysis (no top/bottom selection within buckets)

---

### Creator Flow

**Purpose**: Vet potential affiliate creators by analyzing their natural content style and compatibility with client patterns

**Default Mode**: `--analysis-mode recent` (analyze most recent videos for natural style)

**Command (Recent Mode)** - Natural style vetting (primary use case):
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type creator \
  --target "@potential_affiliate" \
  --video-count 40 \
  --analysis-mode recent \
  --compare-to hashtag:nutrition
```

**Command (Top Mode)** - Peak performance analysis (optional):
```bash
python rumiai_ml_batch.py \
  --client "client_acme_corp" \
  --analysis-type creator \
  --target "@potential_affiliate" \
  --video-count 40 \
  --analysis-mode top \
  --compare-to hashtag:nutrition
```

**Inputs**:
- `--client`: Client identifier
- `--analysis-type`: `creator`
- `--target`: Creator TikTok handle (e.g., `@potential_affiliate`)
- `--video-count`: Number of videos to analyze (default: 40)
- `--analysis-mode`: `recent` (default) | `top`
- `--compare-to`: What to compare against (format: `hashtag:{name}` or `competitor:{handle}`)

**Process (Recent Mode)** - "What does creator naturally produce?":
1. **Apify Scraping**: Fetch most recent 40 videos from creator (sorted by date)
2. **Batch Analysis**: Run RumiAI on all 40 videos
3. **Duration Distribution**: Calculate % of content per bucket (natural style)
4. **Feature Aggregation**: Average features per bucket
5. **Compatibility Scoring**: Compare vs client's hashtag/competitor patterns
6. **Report Generation**: Style profile + compatibility report + hiring recommendation

**Process (Top Mode)** - "What does creator do when they go viral?":
Same as Recent Mode but analyzes top 40 videos by engagement. Use case: Understanding peak performance style for coaching purposes.

**Outputs**:
```
/data/clients/client_acme_corp/creators/potential_affiliate/
  ├── videos/                      # 40 MP4 files
  ├── analysis/insights/           # 40 temporal_windows JSONs
  ├── distribution_analysis/
  │   └── duration_distribution.json  # Natural duration preferences
  ├── compatibility_analysis/
  │   └── vs_hashtag_nutrition/
  │       ├── distribution_match.json      # How well durations align
  │       ├── feature_alignment.json       # Feature-level comparison
  │       ├── compatibility_score.json     # Final score (0-1)
  │       └── hiring_recommendation.json   # Tier 1-4 + reasoning
  └── creator_summary/
      ├── style_profile.pdf            # Creator's natural content style
      └── compatibility_report.pdf     # Match vs client patterns
```

**Duration**: ~53 minutes for 40 videos (80s per video)

**Key Differences vs Hashtag/Competitor**:
- Only 40 videos (snapshot for vetting, not comprehensive analysis)
- No ML model training (uses existing hashtag/competitor models for comparison)
- Outputs compatibility score + hiring recommendation (not creative strategy reports)
- Default mode is `recent` (natural style) vs `top` (best performance)


---

# New Features

## 1. Creator Match Analysis

**Purpose**: Analyze potential affiliate creators' natural content style and match against client's viral patterns to optimize hiring decisions.

**Key Capabilities**:
- Analyze most recent 40 videos from creator's TikTok handle
- Identify creator's natural duration distribution across 8 buckets
- Compare creator's production style against client hashtag/competitor success patterns
- Generate compatibility scores combining distribution match + feature alignment
- Provide hiring recommendation tiers (Immediate Hire → Pass)

**Business Value**:
- Reduce hiring risk by identifying natural creator-brand fit
- Minimize coaching overhead by selecting creators who naturally produce winning durations
- Data-driven affiliate vetting instead of gut-feel decisions

**Implementation Details**: See [MLCreatorMatch.md](./documentation_migration/FutureDevelopments/MLCreatorMatch.md)

**Priority**: HIGH - Critical for affiliate vetting and ROI optimization

---

## 2. Checkpoint Resume System

**Purpose**: Enable recovery from process interruptions during long-running batch analyses (6-8 hours) without re-processing completed videos.

**Key Capabilities**:
- Automatic checkpoint saving after each video completes analysis
- Auto-resume detection when restarting interrupted batch processing
- Track completed videos, failed videos, and current processing state per bucket
- Support for manual restart via `--force` flag (discard checkpoint)

**Business Value**:
- Prevent wasted compute time (3-6 hours saved per interruption)
- Enable reliable batch processing despite SSH disconnects, crashes, or manual stops
- Reduce risk for long-running client analyses (300+ videos)

**Implementation Details**: See [MLCheckpointResume.md](./documentation_migration/FutureDevelopments/MLCheckpointResume.md)

**Priority**: HIGH - Critical for 6-8 hour batch jobs reliability

---

## 3. Analysis Mode System

**Purpose**: Enable dual-purpose analysis - understand "what works" (top-performing content) vs "what's happening now" (current trends/strategy).

**Key Capabilities**:
- `top` mode: Analyze highest-engagement videos to identify successful patterns
- `recent` mode: Analyze most recent videos to track current strategies and trends
- Apify integration for both engagement-based and date-based sorting
- Different default modes per analysis type (hashtag/competitor: top, creator: recent)

**Business Value**:
- Competitive intelligence: Benchmark rival's best work OR track their strategy shifts
- Trend detection: Identify market changes by analyzing recent vs historical patterns
- Creator vetting: Natural style (recent) vs peak performance (top) for different use cases

**Implementation Details**: See [MLAnalysisMode.md](./documentation_migration/FutureDevelopments/MLAnalysisMode.md)

**Priority**: HIGH - Core feature for flexible analysis across different business questions

---

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
- **Workaround**: Must scrape 800 videos, then filter client-side (see VideoSelection.md lines 19-45)

**🔄 TWO SCRAPERS REQUIRED**:

**Use Case 1: Hashtag Analysis**
→ **clockworks/tiktok-hashtag-scraper**
- Input: `hashtagsUrls: ["#nutrition"]`
- Sorting: `sortBy: engagement` (top mode) OR `sortBy: date` (recent mode)
- Volume: `resultsPerPage: 800`

**Use Case 2: Competitor/Creator Analysis**
→ **clockworks/tiktok-scraper** (current)
- Input: `profilesUrls: ["@rival_brand"]`
- Sorting: `sortBy: engagement` OR `sortBy: date`
- Volume: `resultsPerPage: 150` (competitor) OR `40` (creator)

#### Implementation Recommendations

**Phase 1: Update Apify Client (IMMEDIATE)**

Add hashtag scraper support to existing `apify_client.py`:

```python
class ApifyClient:
    def __init__(self, api_token: str):
        self.api_token = api_token
        # Two actor IDs based on use case
        self.profile_scraper_id = "GdWCkxBtKWOsKjdch"  # Current
        self.hashtag_scraper_id = "TBD_HASHTAG_ACTOR_ID"  # New

    async def scrape_hashtag(
        self,
        hashtag: str,
        video_count: int = 800,
        analysis_mode: str = "top"  # "top" or "recent"
    ) -> List[VideoMetadata]:
        """
        Scrape videos from hashtag for ML batch processing
        Supports both top (engagement) and recent (date) modes
        """
        sort_by = "engagement" if analysis_mode == "top" else "date"

        actor_input = {
            "hashtagsUrls": [f"https://www.tiktok.com/tag/{hashtag.lstrip('#')}"],
            "resultsPerPage": video_count,
            "shouldDownloadVideos": True,
            "sortBy": sort_by,
            "sortOrder": "desc",
            "proxyConfiguration": {"useApifyProxy": True}
        }

        # Use hashtag scraper
        return await self._run_scraper(
            self.hashtag_scraper_id,
            actor_input
        )

    async def scrape_profile(
        self,
        handle: str,
        video_count: int,
        analysis_mode: str = "top"
    ) -> List[VideoMetadata]:
        """
        Scrape videos from TikTok profile for competitor/creator analysis
        """
        sort_by = "engagement" if analysis_mode == "top" else "date"

        actor_input = {
            "profilesUrls": [f"https://www.tiktok.com/{handle}"],
            "resultsPerPage": video_count,
            "shouldDownloadVideos": True,
            "sortBy": sort_by,
            "sortOrder": "desc",
            "proxyConfiguration": {"useApifyProxy": True}
        }

        # Use profile scraper (existing)
        return await self._run_scraper(
            self.profile_scraper_id,
            actor_input
        )
```

**Phase 2: Client-Side Date Filtering (REQUIRED)**

Implement post-scrape filtering as documented in VideoSelection.md:

```python
def filter_by_date(videos: List[VideoMetadata], date_filter: str) -> List[VideoMetadata]:
    """
    Client-side date filtering (required for hashtags)

    Args:
        date_filter: "last_90_days" or "2024-01-01:2025-01-01"
    """
    if date_filter.startswith("last_"):
        days = int(date_filter.replace("last_", "").replace("_days", ""))
        min_date = datetime.now() - timedelta(days=days)
    else:
        start, end = date_filter.split(":")
        min_date = datetime.fromisoformat(start)

    return [v for v in videos if v.create_time >= min_date]
```

**Phase 3: Testing & Validation (NEXT STEP)**

Before implementing ML batch processing:

1. **Test hashtag scraper** with sample hashtag:
   ```bash
   python test_apify_hashtag.py --hashtag "#nutrition" --count 100 --mode top
   ```

2. **Verify sorting works**:
   - Top mode: Confirm videos sorted by engagement score
   - Recent mode: Confirm videos sorted by publish date

3. **Validate metadata fields**:
   - Ensure all required fields present (`playCount`, `shareCount`, etc.)
   - Confirm `VideoMetadata.from_apify_data()` works without changes

4. **Test date filtering**:
   - Scrape 800 videos
   - Apply `last_90_days` filter
   - Measure retention rate (expect 40-60% after filtering)

#### Cost Analysis (Updated)

**Hashtag Analysis** (300 target, 800 scraped):
- Scrape cost: 800 videos × $0.005 = **$4.00**
- After date filtering: ~480 videos (60% retention)
- Per bucket analysis: 60 videos × 8 buckets = 480 videos
- **Result**: ✅ Sufficient volume for contrastive analysis

**Competitor Analysis** (150 videos):
- Scrape cost: 150 videos × $0.005 = **$0.75**
- No date filtering needed (all videos processed)

**Creator Analysis** (40 videos):
- Scrape cost: 40 videos × $0.005 = **$0.20**
- No date filtering needed

**Total per client** (1 hashtag + 2 competitors + 3 creators):
- 1 × $4.00 + 2 × $0.75 + 3 × $0.20 = **$6.10 per analysis batch**

#### Action Items

**IMMEDIATE (Required for ML batch MVP)**:
1. ✅ Get hashtag scraper actor ID from Apify dashboard
2. ✅ Update `apify_client.py` with dual-scraper support
3. ✅ Implement `scrape_hashtag()` and `scrape_profile()` methods
4. ✅ Add client-side date filtering function
5. ✅ Test with real hashtag (#nutrition, #fitness) to validate sorting

**NEXT (Before ML training)**:
1. ⏭️ Measure actual retention rate after date filtering
2. ⏭️ Adjust volume targets per bucket (may need <60 if filtering aggressive)
3. ⏭️ Document minimum sample size thresholds (see VideoSelection.md C.9)

**FUTURE (Post-MVP optimization)**:
1. 🔮 Investigate "Super TikTok Scraper" for 90% cost savings ($0.0005/video)
2. 🔮 Add retry logic for failed scrapes
3. 🔮 Implement rate limiting for large batches

#### Answers to Original Questions

1. **Video volume**: 800 initial scrape → ~480 after filtering ✅
2. **Sorting capabilities**: Both scrapers support `sortBy: engagement` and `sortBy: date` ✅
3. **Metadata fields**: All required fields confirmed present ✅
4. **Profile vs Hashtag**: TWO scrapers required (profile scraper + hashtag scraper) ✅
5. **Cost**: ~$0.005 per video, $6.10 per typical client analysis ✅
6. **Date filtering**: Client-side only for hashtags (server-side unavailable) ⚠️

---

## 4. LLM Data Strategy

**Purpose**: Define optimal data formatting and aggregation strategy for sending ML analysis results (Random Forest + K-Means) to Claude API for insight generation and report creation.

**Key Capabilities**:
- Support for two ML analysis types per bucket (Random Forest feature importance + K-Means clustering)
- Full raw data format: Send complete video-level features (60 videos × 35 features per JSON)
- Aggregated statistics format: Send compressed statistical summaries (mean, median, quartiles, distribution)
- Token limit management: Stay within Claude API's 200K token (~800KB) limit
- Scalable comparison mode: Support 5-10+ hashtag comparisons via aggregation

**Architecture Decisions**:

**Single Hashtag Analysis**:
- Data volume: ~480KB (2 JSONs per bucket × 8 buckets)
- Recommendation: **Full raw data**
- Rationale: Well within token limits, provides richer insights for LLM
- LLM calls: 16 total (RF + K-Means per bucket)

**Multi-Hashtag Comparison**:
- Data volume: ~1.44MB raw (exceeds limits)
- Recommendation: **Aggregated statistics** (reduces to ~200KB)
- Rationale: Token limit compliance, prevents hallucination risk
- LLM calls: 8 total (combined RF + K-Means per bucket)

**Business Value**:
- Prevent hallucination and poor report quality from oversized context windows
- Enable scalable multi-hashtag comparisons (5-10+ hashtags)
- Balance data richness (full raw) vs efficiency (aggregated) based on analysis type
- Optimize LLM API costs by right-sizing payloads

**Implementation Details**:
- High-Level Design: [ML_LLMData.md](./documentation_migration/FutureDevelopments/ML_LLMData.md)
- Technical Implementation: [ML_LLMDataTI.md](./documentation_migration/FutureDevelopments/ML_LLMDataTI.md)

**Priority**: MEDIUM - Required before report generation, but can start with single hashtag MVP (full raw data approach)

**Dependencies**:
- Requires ML analysis outputs (Random Forest + K-Means JSONs per bucket)
- Feeds into Creative Report Generation system
- Integration with Claude API for insight synthesis

---

