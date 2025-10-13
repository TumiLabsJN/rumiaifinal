# Video Discovery & Selection - High-Level Design

> **Parent**: MLPlanningv2.md - Stage 1
> **Version**: 1.0
> **Last Updated**: 2025-01-28
> **Status**: Draft

---

## 0. Prerequisites & Dependencies

<!-- PURPOSE: Explicit contract for TI generator - what documents are required before code generation -->

**Required Parent Documents** (must be provided to TI generator):

- **FoundationCHILD.md**: Defines shared foundation infrastructure used by all stages
  - **Section 2**: Client Architecture & Storage (directory structure, file paths)
  - **Section 4**: CLI Command Structure (parameters used by Stage 1: `client_id`, `analysis_type`, `target`, `analysis_mode`, `selection_strategy`, `video_count`, `date_filter`)
  - **Section 5.2**: Apify Video Metadata Schema (input data format from scrapers)
  - **Section 5.3**: Config.json Schema (configuration file structure)

**TI Generation Requirements**:
- FoundationCHILD.md **must be accessible** to TI generator before processing this document
- If FoundationCHILD.md unavailable, TI should **fail-fast** with error: `"Missing required dependency: FoundationCHILD.md. Cannot generate Stage 1 implementation without Foundation."`
- TI generator should validate FoundationCHILD.md sections 2, 4, 5.2, 5.3 are present before code generation

**Dependency Chain**:
```
FoundationCHILD.md (Stage 0: Setup)
    ↓
VideoDiscoveryCHILD.md (Stage 1: Video Discovery) ← YOU ARE HERE
    ↓
[Stage 2 Child doc: Video Processing]
```

**Cross-References in This Document**:
- Section 1.2: References Foundation Section 2 (Client Architecture)
- Section 4.1: References Foundation Section 4 (CLI parameters)
- Section 5.1: References Foundation Section 4 (CLI parameters)
- Section 5.2: References Foundation Section 5.2 (Apify schema)

---

## 1. Context & Business Goal

<!-- PURPOSE: Provide business context and justification. TI generator needs to understand WHY this feature exists. -->

### 1.1 What Problem Does This Solve?

Selection strategies determine **what videos to analyze** after sorting. They control which subset of videos enters the ML training pipeline, directly impacting the type of insights generated, processing resources required, and statistical validity of results.

**The Core Problem**: Different business questions require different analytical approaches:
- **"What makes winners different from underperformers?"** requires contrastive data (top vs bottom performers)
- **"What do successful videos have in common?"** only needs top performers
- **Resource constraints**: Processing 300 videos vs 120 videos has significant cost/time implications

**Market Reality - Winners Cluster in Specific Formats**:
- Typical hashtag: 75% of top 100 performers fall into 3 duration buckets (out of 8 total)
- Processing all 8 buckets equally wastes resources on low-performing formats
- **Adaptive bucket processing**: Focus resources on formats where winners cluster (success-based, not volume-based)

**Example**: In #nutrition, if 45% of winners are 18-33s, 30% are 33-60s, and 20% are 13-18s, we process those 3 buckets (95% winner coverage) and skip the other 5 buckets (even if high volume).

**Business Value**:
- **Cost savings**: Process 3 buckets instead of 8 (75% cost reduction: $2.40 vs $7.60 in LLM calls)
- **Time savings**: 2 hours vs 8 hours for full hashtag analysis
- **Quality insights**: Focus on statistically valid data from winning formats
- **Strategic clarity**: "Focus on 18-60s where winners cluster" (not where volume is)

### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This stage depends on FoundationCHILD.md for:
- **Section 2 (Client Architecture)**: Directory structure and path templates
- **Section 3 (Configuration Dimensions)**: Target types, analysis modes, selection strategies
- **Section 4 (CLI Command Structure)**: CLI parameter definitions and defaults
- **Section 5 (Configuration Schemas)**: config.json schema, Apify metadata schema

```
Stage 0: Configuration (FoundationCHILD.md)
   ↓ CLI parameters: target, mode, strategy, video_count, date_filter
   ↓ Directory structure created
   ↓ config.json written
Stage 1: Video Discovery & Selection (THIS STAGE)
   ↓ Output: selected_videos.json (per bucket) + video list
Stage 2: Video Processing (RumiAI Pipeline)
   ↓ temporal_windows_updated.json (N videos per winning bucket)
Stage 3: Feature Aggregation
```

### 1.3 Success Criteria

- [x] **Scraping**: Retrieve 800 videos from target in < 2 minutes (Apify performance)
- [x] **Date Filtering**: Filter to date range with < 1% error rate (client-side accuracy)
- [x] **Winner Analysis**: Identify top 3 winning buckets with > 90% winner coverage (adaptive processing)
- [x] **Video Selection**: Select N videos per bucket (configurable via --video-count, strategy-specific)
- [x] **Validation**: Ensure selected videos match strategy requirements (80/20 split for contrastive, all top for top-only)
- [x] **Output**: Generate selected_videos.json per bucket with complete Apify metadata

---

## 2. Architecture & Design

<!-- PURPOSE: Core technical design. This is the PRIMARY section TI generator reads. -->

### 2.1 High-Level Approach

Sequential 4-step process implementing **success-based bucket selection** (not volume-based):

1. **Apify Scraping**: Scrape 800 videos from target, sorted by engagement DESC (captures full performance spectrum)
2. **Date Filtering**: Client-side filtering by publication date (Apify doesn't support server-side for hashtags)
3. **Winner Analysis**: Analyze top 100 performers to identify where winners cluster (success-based distribution)
4. **Bucket Selection**: Select top 3 buckets by winner concentration, then select N videos per bucket (strategy-specific)

**Key Architectural Decision**: Use success-based distribution (where winners cluster) rather than volume-based (where most videos are). This focuses resources on winning formats, even if they're not the most popular.

### 2.2 Data Flow

```
Input: CLI parameters (from FoundationCHILD.md Section 4)
       - client_id, analysis_type, target
       - analysis_mode, selection_strategy, video_count, date_filter
   ↓
Stage 1.1: Apify Scraping + Deduplication
   ↓ 800 videos scraped → ~720 unique (engagement sorted DESC, duplicates removed)
Stage 1.2: Date Filtering (Client-Side)
   ↓ ~600 videos (filtered to last_N_days based on create_time)
Stage 1.3: Winner Analysis (Success-Based Distribution)
   ↓ Top 3 buckets identified (where winners cluster, not volume)
   ↓ Example: 18-33s (45%), 33-60s (30%), 13-18s (20%) = 95% winner coverage
Stage 1.4: Video Selection Per Bucket (Strategy-Specific)
   ↓ Contrastive: N videos (80% top + 20% bottom per bucket)
   ↓ Top: N videos (all top performers per bucket)
   ↓
Output: selected_videos.json (per bucket)
        Format: {"bucket": str, "videos": [ApifyMetadata], ...}
```

### 2.3 Detailed Process

#### Step 2.3.1: Apify Scraping (Stage 1.1)

**Purpose**: Scrape 800 videos from target and sort by engagement client-side

**Logic**:
```python
def scrape_videos(target, analysis_type, analysis_mode, country_code):
    """
    Scrape 800 videos from TikTok via Apify and sort by engagement.

    Args:
        target: str, target identifier (#nutrition, @rival_brand)
        analysis_type: str, "hashtag" or "competitor" or "creator"
        analysis_mode: str, "top" or "recent"
        country_code: str, "US" or "BR" or "global"

    Returns:
        list: Apify video metadata objects (800 videos, sorted by engagement)
    """
    # Select scraper based on analysis_type
    if analysis_type == "hashtag":
        scraper = "clockworks/tiktok-hashtag-scraper"
        input_param = "hashtagsUrls"
    else:  # competitor or creator
        scraper = "clockworks/tiktok-scraper"
        input_param = "profilesUrls"

    # Configure scraper parameters
    apify_input = {
        input_param: [target],
        "resultsPerPage": 800,
        "shouldDownloadVideos": True
    }

    # Apply geographic filtering via proxy routing
    if country_code != "global":
        apify_input["proxyCountryCode"] = country_code  # "US" or "BR"
    # If country_code == "global", omit proxyCountryCode (no geographic filtering)

    # Call Apify API
    run = apify_client.actor(scraper).call(run_input=apify_input)
    videos = apify_client.dataset(run["defaultDatasetId"]).list_items().items

    # Client-side sorting by engagement
    if analysis_mode == "top":
        # Sort by view count (playCount) descending
        videos = sorted(videos, key=lambda v: v.get('playCount', 0), reverse=True)
    else:  # recent mode
        # Sort by publication date (createTime) descending
        videos = sorted(videos, key=lambda v: v.get('createTime', 0), reverse=True)

    # Deduplicate by video ID (keep first occurrence)
    # TikTok videos can appear multiple times (reposts, cross-hashtag appearances, Apify duplicates)
    seen_ids = set()
    unique_videos = []
    for video in videos:
        if video['id'] not in seen_ids:
            seen_ids.add(video['id'])
            unique_videos.append(video)

    # Log deduplication stats
    duplicate_count = len(videos) - len(unique_videos)
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate videos ({duplicate_count/len(videos)*100:.1f}%)")
    logger.info(f"Scraped {len(videos)} videos → {len(unique_videos)} unique")

    return unique_videos  # List of unique video metadata objects (sorted)
```

**Apify Scraper Selection**:

| Analysis Type | Scraper | Input Parameter | Target Format |
|---------------|---------|-----------------|---------------|
| `hashtag` | clockworks/tiktok-hashtag-scraper | `hashtagsUrls` | `["#nutrition"]` |
| `competitor` | clockworks/tiktok-scraper | `profilesUrls` | `["@rival_brand"]` |
| `creator` | clockworks/tiktok-scraper | `profilesUrls` | `["@creator_name"]` |

**Client-Side Engagement Sorting**:

Apify returns videos in default order (not sorted by engagement). RumiAI sorts videos client-side by view count:

```python
# Sort by playCount (views) descending
sorted_videos = sorted(videos, key=lambda v: v['playCount'], reverse=True)
```

**Rationale**:
- View count (`playCount`) is the primary engagement metric
- Simple, transparent, and universally understood
- Aligns with "top performers" business definition (most-viewed = most successful)

**Deduplication Strategy**:

Duplicates are removed **immediately after scraping** (Stage 1.1) for these reasons:
- **Clean dataset from start**: All downstream stages (date filtering, winner analysis, bucket selection) work with unique videos only
- **Keeps first occurrence**: After engagement sorting, first occurrence is highest-engagement version
- **Transparent logging**: User knows exactly how many duplicates removed
- **Resource efficient**: Stage 2 doesn't waste time processing same video multiple times
- **Consistent statistics**: Winner analysis and ML training not skewed by duplicate videos

**Common Duplicate Sources**:
- Same video posted to multiple accounts (reposts/collaborations)
- Same video appearing in multiple hashtags (cross-hashtag scraping)
- Apify scraper returning duplicate IDs (API behavior)

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Apify returns < 800 videos | Proceed with available videos | Niche hashtags/profiles may have limited content |
| Apify timeout (> 120s) | Retry 3x with exponential backoff | Network issues are transient |
| Invalid metadata (missing fields) | Skip video, log warning | Bad data should not halt pipeline |
| 10%+ duplicate videos | Deduplicate, log count, proceed | Common in trending hashtags (many reposts) |
| All videos are duplicates (extreme edge case) | Fail-fast with error message | Data quality issue, likely scraper misconfiguration |

#### Step 2.3.2: Date Filtering (Stage 1.2)

**Purpose**: Filter scraped videos to date range based on publication time (client-side)

**Timezone Handling**:

Date filtering uses **explicit UTC** to avoid timezone ambiguity:
- TikTok `create_time` field is Unix timestamp in UTC
- All date comparisons performed in UTC timezone
- "last_N_days" calculated from current UTC time (not local time)
- Ensures consistent filtering regardless of server/user timezone

**Logic**:
```python
def filter_by_date(videos, date_filter):
    """
    Filter videos by publication date (client-side, UTC-based).

    Args:
        videos: list, Apify video metadata objects
        date_filter: str, "last_N_days" (e.g., "last_90_days")

    Returns:
        list: Filtered videos within date range
    """
    # Parse date_filter parameter
    days = int(date_filter.replace("last_", "").replace("_days", ""))

    # Always use UTC for consistency
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    # Filter with validation
    filtered_videos = []
    skipped_count = 0

    for video in videos:
        create_time = video.get("createTime")

        # Validate create_time exists
        if create_time is None or create_time == 0:
            logger.warning(f"Video {video.get('id', 'unknown')} has invalid create_time (null/zero). Skipping.")
            skipped_count += 1
            continue

        # Convert Unix timestamp to UTC datetime
        try:
            video_date = datetime.fromtimestamp(create_time, tz=timezone.utc)
        except (ValueError, OSError) as e:
            logger.warning(f"Video {video.get('id', 'unknown')} has invalid timestamp {create_time}. Skipping. Error: {e}")
            skipped_count += 1
            continue

        # Handle future timestamps (clock skew tolerance: 24 hours)
        if video_date > datetime.now(timezone.utc) + timedelta(hours=24):
            logger.warning(f"Video {video.get('id', 'unknown')} has future timestamp {video_date}. Skipping.")
            skipped_count += 1
            continue

        # Apply date filter
        if video_date >= cutoff_date:
            filtered_videos.append(video)

    # Log filtering result
    logger.info(f"Date filtering: {len(videos)} → {len(filtered_videos)} videos (last {days} days)")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} videos due to invalid timestamps")

    return filtered_videos
```

**Why Client-Side Filtering?**:
- **Apify limitation**: Hashtag scraper doesn't support server-side date filtering
- **Consistency**: Profile scraper has date support, but client-side used for uniform behavior across all target types
- **Simple implementation**: One code path for all analysis types

**Business Value**:
- **Recency control**: Focus on recent trends vs historical patterns
- **Seasonal analysis**: Analyze specific time periods (e.g., holiday season)
- **Trend detection**: Track how patterns evolve over time
- **Data quality**: Exclude outdated content that may skew insights

**Timestamp Validation**:

Date filtering includes robust validation to handle malformed TikTok metadata:
- **Null/zero timestamps**: Skipped with warning (invalid metadata)
- **Invalid timestamps**: Skipped with error details (conversion failure)
- **Future timestamps**: Skipped if > 24 hours in future (beyond clock skew tolerance)
- **All validation logged**: User knows exactly why videos were excluded

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| All videos outside date range | Warn user, relax filter to last 180 days | Date filter too aggressive for target |
| < 100 videos after filter | Proceed with warning, log count | Limited recent content available |
| Invalid create_time (null/zero) | Skip video with warning, continue | Bad metadata should not crash pipeline |
| Malformed timestamp (negative, overflow) | Skip video with error details, continue | Conversion failure indicates data corruption |
| Future timestamp (> 24h ahead) | Skip video with warning, continue | Beyond clock skew tolerance, likely bad data |
| 10%+ videos skipped due to invalid timestamps | Log total skipped count, proceed | Data quality issue, but enough valid videos remain |

#### Step 2.3.3: Winner Analysis (Stage 1.3)

**Purpose**: Identify top 3 buckets where winners cluster using success-based distribution (not volume-based)

**Bucket Selection Note**:

RumiAI defines **8 potential buckets** (0-3s through 90-120s) but only processes **top 3 active buckets** per analysis:
- **8 potential buckets**: Provides classification options for data-driven selection
- **3 active buckets**: Selected based on where winners cluster (success-based, not volume-based)
- **Typical result**: 3 active buckets cover 90-95% of winners (e.g., 18-33s, 33-60s, 13-18s)
- **Infrastructure**: Directories created for all 8 buckets, but only 3 populated with selected videos
- **ML capacity**: Up to 16 models possible (8 × 2), typically 6 models trained (3 × 2)

This approach balances flexibility (can adapt to any winner distribution) with efficiency (focus resources on winning formats).

**Engagement Snapshot Timing**:

Winner analysis operates on a **point-in-time snapshot** of engagement metrics from Apify scraping. TikTok engagement changes constantly (views, shares, likes update in real-time), meaning:

- Analysis reflects performance **at scrape time** (Stage 1.1 timestamp)
- Videos ranked #50 during scraping may shift position hours later
- "Winning buckets" identified are based on engagement data that becomes stale as processing continues

**Acceptable Staleness Window**:
- Engagement snapshot considered valid for **~6 hours** after scraping
- Stage 1 completes in < 2 minutes, leaving ~4-5 hours for Stage 2 processing before significant drift
- If Stage 2 processing extends beyond 6 hours, winner distribution may no longer reflect current performance

**Mitigation Strategy**:
- Process videos quickly after Stage 1 (recommended: complete Stage 2 within 6 hours)
- Log scrape timestamp in `winner_analysis.json` for auditability
- Document that ML training uses "historical engagement snapshot" (not live data)

**Rationale for Accepting Snapshot**:
- Re-validation would add $0.50+ cost and 30-60s latency per target
- Engagement trends are generally stable over 6-hour windows (gradual changes, not sudden spikes)
- Point-in-time analysis is sufficient for identifying format patterns (18-33s vs 60-90s)
- Trade-off: Accept minor staleness risk in exchange for fast, cost-effective processing

**The Business Problem with Volume-Based Processing**:

**Market Reality**: Winners cluster in specific formats, not necessarily high-volume formats.

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

**Logic**:
```python
def analyze_winner_distribution(videos):
    """
    Analyze top performers to identify winning buckets.

    Args:
        videos: list, filtered videos (sorted by engagement DESC)

    Returns:
        list: Top 3 bucket names where winners cluster

    Raises:
        ValueError: if insufficient videos for analysis
    """
    # Validate minimum dataset size
    if len(videos) < 10:
        raise ValueError(
            f"Insufficient videos for analysis. Need ≥10, got {len(videos)}. "
            f"Try different target or relax date filter."
        )

    # Determine analysis mode based on dataset size
    if len(videos) < 100:
        # Degraded mode: analyze all available videos
        top_performers = videos  # Use all videos
        logger.warning(
            f"Small dataset ({len(videos)} videos). Analyzing all available. "
            f"Statistical validity may be limited. Recommended: ≥100 videos."
        )
    else:
        # Normal mode: analyze top 100 performers
        top_performers = videos[:100]

    # Analyze where top performers cluster
    top_100 = top_performers  # Maintain variable name for consistency

    # Bucket videos by duration
    winner_distribution = {}
    for video in top_100:
        bucket = get_bucket_name(video["duration"])  # e.g., "18-33s"
        winner_distribution[bucket] = winner_distribution.get(bucket, 0) + 1

    # Calculate winner concentration percentages
    winner_percentages = {
        bucket: count / len(top_100) * 100  # Percentage of top performers analyzed
        for bucket, count in winner_distribution.items()
    }

    # Filter buckets: only keep those with ≥ MIN_WINNER_PERCENTAGE (5%)
    # This prevents processing buckets with only 1-4 winners (wasteful)
    qualified_buckets = {
        bucket: percentage
        for bucket, percentage in winner_percentages.items()
        if percentage >= MIN_WINNER_PERCENTAGE  # 5.0% threshold from Section 4.2
    }

    # Sort qualified buckets by winner concentration (DESC) and select top 3
    top_buckets = sorted(
        qualified_buckets.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    # Handle edge case: < 3 qualified buckets
    if len(top_buckets) < 3:
        logger.warning(
            f"Only {len(top_buckets)} bucket(s) qualified (≥{MIN_WINNER_PERCENTAGE}% winners). "
            f"Processing {len(top_buckets)} bucket(s) instead of 3."
        )

    # Log winner distribution
    logger.info(f"Winner distribution ({len(top_100)} top performers):")
    for bucket, percentage in top_buckets:
        logger.info(f"  - {bucket}: {winner_distribution[bucket]} videos ({percentage:.1f}%)")

    # Calculate total coverage
    total_coverage = sum(winner_distribution[b] for b, _ in top_buckets)
    logger.info(f"Total winner coverage: {total_coverage}/{len(top_100)} ({total_coverage/len(top_100)*100:.1f}%)")

    return [bucket for bucket, _ in top_buckets]


def get_bucket_name(duration):
    """
    Map duration to bucket name.

    Args:
        duration: int, video duration in seconds

    Returns:
        str, bucket name (e.g., "18-33s")
    """
    if duration <= 3:
        return "0-3s"
    elif duration <= 9:
        return "3-9s"
    elif duration <= 13:
        return "9-13s"
    elif duration <= 18:
        return "13-18s"
    elif duration <= 33:
        return "18-33s"
    elif duration <= 60:
        return "33-60s"
    elif duration <= 90:
        return "60-90s"
    else:
        return "90-120s"
```

**Adaptive Logic**:
1. Scrape 800 videos (engagement sorted DESC)
2. Analyze top 100 performers: Bucket them by duration
3. Calculate winner concentration: What % of top 100 fall into each bucket?
4. Select top 3 buckets by success concentration (where winners cluster)
5. Per winning bucket: Process with requested N (or all available videos if < N, with warning)

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

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| < 100 videos total | Use all available for analysis | Small datasets still processable |
| Bucket has < 5% of winners | Skip bucket (doesn't qualify) | Prevents processing buckets with 1-4 winners (wasteful) |
| Only 1-2 buckets qualify (≥5% winners) | Process those buckets only (< 3) | Don't force processing low-winner buckets |
| Winners spread evenly across 8 buckets | Process top 3 that qualify (if ≥5% each) | Select highest concentration buckets |
| All buckets have < 5% winners | Process top 3 anyway (relaxed threshold) | Edge case: extremely fragmented distribution |
| Bucket has 0 videos | Skip bucket entirely | No data to process |

#### Step 2.3.4: Video Selection Per Bucket (Stage 1.4)

**Purpose**: Select N videos per winning bucket using strategy-specific logic (contrastive vs top)

**Contrastive Strategy** (Default N=100):

**What It Does**:
- Selects top 80% + bottom 20% of N videos per winning bucket
- Enables contrastive learning: "What makes extremely successful videos different from moderately successful videos?"
- Trains classification models (Random Forest)

**What "Bottom" Actually Means - Important Limitation**:

This strategy provides **moderate contrast, not extreme contrast**. Here's why:

- **NOT true failures**: Apify scrapes 800 videos sorted by engagement DESC, capturing only relatively successful content
- **"Bottom" = lower-performing within successful sample**: Videos ranked #81-100 in bucket still have significant engagement (e.g., 50K-80K views)
- **Missing true underperformers**: Videos with < 10K views are not included in scraped dataset

**Contrast Quality**:
- **Actual contrast**: Top 80 (500K-200K views) vs Bottom 20 (50K-100K views)
- **Not**: Top performers (500K views) vs True failures (1K-5K views)
- **Learning signal**: Moderate (differentiates "extremely viral" from "moderately viral")
- **Business value**: Identifies what makes GREAT videos vs GOOD videos (not great vs bad)

**Why This Approach Still Works**:
- Sufficient for RumiAI's goal: Identify patterns in successful content (not analyze why videos fail)
- Random Forest can differentiate "extremely successful" from "moderately successful" patterns
- Cost-effective: No additional scraping required for true failures
- Realistic: Captures performance spectrum available within trending/popular content

**Logic**:
```python
def select_videos_contrastive(bucket_videos, video_count):
    """
    Select top 80% + bottom 20% per bucket (contrastive strategy).

    Args:
        bucket_videos: list, all videos in bucket (sorted by engagement DESC)
        video_count: int, N from --video-count (default 100)

    Returns:
        dict: {"top": [videos], "bottom": [videos], "total": int}
    """
    if len(bucket_videos) >= video_count:
        # Normal processing: Select top 80% + bottom 20%
        top_count = int(video_count * 0.8)  # 80 videos
        bottom_count = video_count - top_count  # 20 videos

        top_videos = bucket_videos[:top_count]
        bottom_videos = bucket_videos[top_count:video_count]

        selected = {
            "top": top_videos,
            "bottom": bottom_videos,
            "total": len(top_videos) + len(bottom_videos)
        }

        logger.info(f"Selected {len(top_videos)} top + {len(bottom_videos)} bottom = {selected['total']} videos")

    elif len(bucket_videos) > 0:
        # Flexible threshold: Process all available videos with warning
        logger.warning(f"Only {len(bucket_videos)} videos available (requested N={video_count})")
        logger.warning(f"Processing all {len(bucket_videos)} videos")

        # Still split 80/20 based on available count
        top_count = int(len(bucket_videos) * 0.8)
        top_videos = bucket_videos[:top_count]
        bottom_videos = bucket_videos[top_count:]

        selected = {
            "top": top_videos,
            "bottom": bottom_videos,
            "total": len(bucket_videos)
        }

    else:
        # Empty bucket: Skip
        logger.error(f"Bucket has 0 videos. Skipping.")
        return None

    return selected
```

**Example** (N=100):
```
Bucket 18-33s: 120 videos available
- Top 80: Rank #1-80 in bucket (500K-200K views - top performers)
- Bottom 20: Rank #81-100 in bucket (50K-100K views - less successful within winners)
- Total: 100 videos selected
```

**Top Strategy** (Default N=40):

**What It Does**:
- Selects top N performers only (no bottom group)
- Analyzes patterns in successful content only
- Clustering/descriptive analysis (no classification)

**Logic**:
```python
def select_videos_top(bucket_videos, video_count):
    """
    Select top N performers only (top strategy).

    Args:
        bucket_videos: list, all videos in bucket (sorted by engagement DESC)
        video_count: int, N from --video-count (default 40)

    Returns:
        dict: {"top": [videos], "bottom": [], "total": int}
    """
    if len(bucket_videos) >= video_count:
        # Normal processing: Select top N
        top_videos = bucket_videos[:video_count]

        selected = {
            "top": top_videos,
            "bottom": [],  # No bottom group for top strategy
            "total": len(top_videos)
        }

        logger.info(f"Selected top {len(top_videos)} videos")

    elif len(bucket_videos) > 0:
        # Flexible threshold: Process all available videos with warning
        logger.warning(f"Only {len(bucket_videos)} videos available (requested N={video_count})")
        logger.warning(f"Processing all {len(bucket_videos)} videos")

        selected = {
            "top": bucket_videos,
            "bottom": [],
            "total": len(bucket_videos)
        }

    else:
        # Empty bucket: Skip
        logger.error(f"Bucket has 0 videos. Skipping.")
        return None

    return selected
```

**Example** (N=40):
```
Bucket 18-33s: 120 videos available
- Top 40: Rank #1-40 in bucket (top performers only)
- Bottom: None (not analyzed)
- Total: 40 videos selected
```

**Strategy Comparison**:

| Aspect | Contrastive (N=100) | Top (N=40) |
|--------|---------------------|------------|
| **Videos Selected** | 80 top + 20 lower-performing = 100 | 40 top only |
| **Processing Time** | Higher (more videos) | Lower (fewer videos) |
| **ML Output** | Random Forest classification | K-Means clustering only |
| **Insights** | "What makes extremely successful videos different from moderately successful videos?" | "What do winners have in common?" |
| **Contrast Level** | Moderate (500K vs 50K views within successful sample) | N/A (no comparison group) |
| **Use Case** | Identify top performer patterns vs good performer patterns | Identify common patterns in successful content |
| **Cost** | Higher ($4 Apify + more compute) | Lower ($4 Apify + less compute) |

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Winning bucket has < N videos | Use all available, warn user | Winning bucket still valuable despite small size |
| Winning bucket empty (0 videos) | Skip bucket entirely | No data to process |
| All buckets have < N videos | Process all, adjust expectations | Small dataset, lower statistical validity |
| User requests N > 500 | Reject with error | Exceeds memory/processing limits |

**Output**:
```python
# selected_videos.json per bucket
{
    "bucket": "18-33s",
    "strategy": "contrastive",
    "video_count": 100,
    "selected_count": 100,
    "top_count": 80,
    "bottom_count": 20,
    "videos": [
        {
            "id": "7428596413707144481",
            "createTime": 1704067200,
            "duration": 25,
            "playCount": 50000,
            "shareCount": 500,
            "webVideoUrl": "https://www.tiktok.com/@user/video/123",
            "videoMeta": {"downloadAddr": "https://..."},
            "authorMeta": {"name": "@user"}
        },
        # ... 99 more videos
    ],
    "selection_date": "2025-01-28T10:30:00Z"
}
```

#### Step 2.4: Interactive Confirmation (Stage 1.5)

**Purpose**: Display bucket selection summary and obtain user confirmation before proceeding to Stage 2 (Download & Analysis)

**Why This Matters**:
- **Cost Control**: Abort before expensive operations (Apify profile scraping, video downloads, ML inference)
- **Quality Gate**: Verify bucket selection looks reasonable before committing resources
- **Debugging Aid**: Quickly identify if winner analysis produced unexpected results
- **Transparency**: Understand what will be processed before it happens

**Display Format**:

```
Stage 1 Complete: Video Discovery & Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Selected Buckets (by winner concentration):

  1. 15-30s  →  28 videos  (32.0% of winners)
  2. 30-45s  →  24 videos  (24.0% of winners)
  3. 45-60s  →  20 videos  (16.0% of winners)

Total: 72 videos across 3 buckets

Proceed to Stage 2 (Download & Analysis)? [Y/n/details]
```

**Degraded Mode Warning** (if < 100 videos analyzed):

```
⚠️  DEGRADED MODE: Only 67 videos analyzed (target: 100)
   Statistical confidence may be limited.

Selected Buckets (by winner concentration):
  ...

Proceed despite degraded mode? [Y/n/details]
```

**Logic**:

```python
def confirm_bucket_selection(selected_buckets, winner_distribution, all_qualified_buckets, top_100, videos):
    """
    Display interactive confirmation prompt and await user decision.

    Args:
        selected_buckets: list, top 3 bucket names selected for processing
        winner_distribution: dict, {bucket: winner_count} for all buckets
        all_qualified_buckets: dict, {bucket: percentage} for buckets ≥5% winners
        top_100: list, top performers analyzed (may be < 100 in degraded mode)
        videos: list, all filtered videos (for video count per bucket)

    Returns:
        bool: True if user confirms (Y), False if user aborts (n)

    Raises:
        SystemExit: Exit code 130 if user aborts (n)
    """
    # Check for degraded mode warning
    if len(top_100) < 100:
        print(f"⚠️  DEGRADED MODE: Only {len(top_100)} videos analyzed (target: 100)")
        print("   Statistical confidence may be limited.\n")

    # Display header
    print("Stage 1 Complete: Video Discovery & Selection")
    print("━" * 50)
    print("Selected Buckets (by winner concentration):\n")

    # Display selected buckets with winner percentages
    for i, bucket in enumerate(selected_buckets, 1):
        # Calculate video count in this bucket (from all filtered videos)
        bucket_video_count = len([v for v in videos if get_bucket_name(v["duration"]) == bucket])

        # Get winner percentage (from winner analysis)
        winner_percentage = (winner_distribution[bucket] / len(top_100)) * 100

        print(f"  {i}. {bucket}  →  {bucket_video_count} videos  ({winner_percentage:.1f}% of winners)")

    # Display total
    total_videos = sum(
        len([v for v in videos if get_bucket_name(v["duration"]) == bucket])
        for bucket in selected_buckets
    )
    print(f"\nTotal: {total_videos} videos across {len(selected_buckets)} buckets\n")

    # Skip prompt if auto-confirm enabled (CLI flag or config)
    if AUTO_CONFIRM or cli_args.auto_confirm:
        logger.info("Auto-confirm enabled, proceeding to Stage 2")
        return True

    # Interactive prompt loop
    while True:
        response = input("Proceed to Stage 2 (Download & Analysis)? [Y/n/details] ").strip().lower()

        if response in ['y', 'yes', '']:
            logger.info("User confirmed, proceeding to Stage 2")
            return True

        elif response == 'n':
            logger.info("User aborted at confirmation prompt")
            print("Analysis aborted by user.")
            sys.exit(130)  # Standard Unix exit code for user interrupt

        elif response == 'details':
            # Show detailed bucket analysis including runners-up
            show_detailed_bucket_analysis(
                selected_buckets,
                winner_distribution,
                all_qualified_buckets,
                top_100,
                videos
            )
            # Loop back to prompt after showing details

        else:
            print("Invalid input. Please enter Y, n, or details.")


def show_detailed_bucket_analysis(selected_buckets, winner_distribution, all_qualified_buckets, top_100, videos):
    """
    Display expanded view with runners-up and disqualified buckets.

    Args:
        selected_buckets: list, top 3 buckets selected
        winner_distribution: dict, {bucket: winner_count} for all buckets
        all_qualified_buckets: dict, {bucket: percentage} for buckets ≥5% winners
        top_100: list, top performers analyzed
        videos: list, all filtered videos
    """
    print("\n" + "=" * 50)
    print("DETAILED BUCKET ANALYSIS")
    print("=" * 50 + "\n")

    # Show selected buckets
    print("✓ SELECTED BUCKETS (Top 3 by winner concentration):\n")
    for i, bucket in enumerate(selected_buckets, 1):
        winner_count = winner_distribution.get(bucket, 0)
        winner_percentage = (winner_count / len(top_100)) * 100
        bucket_video_count = len([v for v in videos if get_bucket_name(v["duration"]) == bucket])
        print(f"  {i}. {bucket}: {bucket_video_count} videos ({winner_percentage:.1f}% of winners)")

    # Show runners-up (qualified but not selected)
    runners_up = {
        bucket: percentage
        for bucket, percentage in all_qualified_buckets.items()
        if bucket not in selected_buckets
    }

    if runners_up:
        print("\n○ ALSO QUALIFIED (not selected):\n")
        rank = len(selected_buckets) + 1
        for bucket, percentage in sorted(runners_up.items(), key=lambda x: x[1], reverse=True):
            winner_count = winner_distribution.get(bucket, 0)
            bucket_video_count = len([v for v in videos if get_bucket_name(v["duration"]) == bucket])
            print(f"  {rank}. {bucket}: {bucket_video_count} videos ({percentage:.1f}% of winners)")
            rank += 1

    # Show disqualified buckets (< 5% threshold)
    disqualified = {
        bucket: count
        for bucket, count in winner_distribution.items()
        if bucket not in all_qualified_buckets
    }

    if disqualified:
        print("\n✗ DID NOT QUALIFY (<5% winner threshold):\n")
        for bucket, winner_count in sorted(disqualified.items(), key=lambda x: x[1], reverse=True):
            winner_percentage = (winner_count / len(top_100)) * 100
            bucket_video_count = len([v for v in videos if get_bucket_name(v["duration"]) == bucket])
            print(f"  {bucket}: {bucket_video_count} videos ({winner_percentage:.1f}% of winners) ← Below MIN_WINNER_PERCENTAGE")

    print("\n" + "=" * 50 + "\n")
```

**User Actions**:

1. **Y / Enter**: Proceed to Stage 2 (Download & Analysis)
   - Default action (pressing Enter = Yes)
   - Logs confirmation and continues pipeline

2. **n**: Abort analysis
   - Exits with code 130 (user interrupt, not an error)
   - Displays: "Analysis aborted by user."
   - No Stage 2 processing occurs

3. **details**: Show expanded bucket analysis
   - Displays selected buckets, runners-up, and disqualified buckets
   - Shows why each bucket qualified or didn't qualify
   - Re-prompts for Y/n after showing details

**Automation Bypass**:

Use `--auto-confirm` CLI flag (see FoundationCHILD.md Section 4.1) to skip prompt:

```bash
rumi-cli analyze --target nutrition --auto-confirm
```

Enables unattended execution for CI/CD pipelines while preserving interactive confirmation for manual runs.

**Edge Cases**:

| Scenario | Handling | Display | Rationale |
|----------|----------|---------|-----------|
| Degraded mode (< 100 videos) | Show warning before prompt | "⚠️ DEGRADED MODE: Only 67 videos..." | User should know statistical validity is limited |
| Only 1-2 buckets selected | Show actual count | "Total: 45 videos across 2 buckets" | Transparent about fewer buckets than typical |
| All buckets have 0 videos | Should not reach this stage | (Handled in Section 2.3.3 - fail-fast) | Invalid state, caught earlier |
| Auto-confirm enabled | Skip prompt entirely | Log: "Auto-confirm enabled, proceeding..." | CI/CD mode, no user interaction needed |
| Invalid user input | Re-prompt with error | "Invalid input. Please enter Y, n, or details." | Help user recover from typo |

---

## 3. Dependencies & Integration

<!-- PURPOSE: Explicit contracts with other stages. TI generator uses this for imports and validation. -->

### 3.1 Input Dependencies

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **Foundation setup** | FoundationCHILD.md (FoundationTI.md implementation) | Directory structure + config.json | client_id, analysis_type, target, mode, strategy, video_count, date_filter, country_code, base_paths | Fail-fast if directories don't exist or config.json missing |
| CLI parameters | FoundationCHILD.md Section 4 | Command-line args | --client, --analysis-type, --target, --analysis-mode, --selection-strategy, --video-count, --date-filter, --country-code | Parse error with usage message if missing required params |
| Apify API key | Environment variable | String | APIFY_API_KEY | Fail-fast with error message if missing |
| Directory structure | FoundationCHILD.md Section 2 | File paths | {analysis_base}/buckets/{bucket}/ | Create directories if missing (mkdir -p) |

### 3.2 Output Contracts

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| selected_videos.json | JSON (per bucket) | `{"bucket": str, "videos": [ApifyMetadata]}` | Stage 2 (Video Processing) | Assert N videos selected, validate Apify schema |
| config.json | JSON | FoundationCHILD.md Section 5.1 | All stages | Validate schema matches Foundation spec |
| winner_analysis.json | JSON | `{"top_100_distribution": {}, "top_3_buckets": []}` | Reporting, debugging | Log winner distribution for transparency |

### 3.3 Cross-Stage Dependencies

**This stage depends on**:
- **Stage 0 (Configuration)**: CLI parsing complete, directories created, config.json written

**This stage is required by**:
- **Stage 2 (Video Processing)**: Expects selected_videos.json per bucket with valid Apify metadata

**Failure Impact**:
- If this stage fails: No videos selected, cannot proceed to Stage 2
- Checkpoint: Not applicable (Stage 1 is fast, < 2 minutes)

### 3.4 External Dependencies

**Python Libraries**:
```python
import requests  # 2.31.0+ - HTTP requests for Apify API
from datetime import datetime, timedelta, timezone  # stdlib - Date filtering (UTC-aware)
import json  # stdlib - Config/output files
import os  # stdlib - File path operations
from apify_client import ApifyClient  # 1.6.0+ - Apify SDK
```

**File System**:
- Read access: Environment variable `APIFY_API_KEY`
- Write access: `{analysis_base}/buckets/{bucket}/selected_videos.json`
- Write access: `{analysis_base}/config.json`

**Environment Variables**:
- `APIFY_API_KEY`: Required for scraper authentication (Apify account API key)
- `DATA_ROOT`: Root directory for client data (default: `/data`)

**External Services**:
- **Apify API**: clockworks/tiktok-hashtag-scraper, clockworks/tiktok-scraper
  - Rate limit: ~10 concurrent runs
  - Timeout: 120 seconds default
  - Cost: ~$0.005 per video scraped

---

## 4. Configuration & Parameters

<!-- PURPOSE: All tunable values. TI generator uses this for config parsing and defaults. -->

### 4.1 CLI Parameters (if applicable)

**Reference**: FoundationCHILD.md Section 4 for complete CLI documentation

**Key Parameters for Stage 1**:

| Parameter | Type | Default | Valid Values | Impact on Stage 1 | Example |
|-----------|------|---------|--------------|-------------------|---------|
| `--analysis-type` | str | Required | hashtag, competitor, creator | Determines Apify scraper selection | `--analysis-type hashtag` |
| `--target` | str | Required | Varies | Target to scrape (#nutrition, @handle) | `--target "#nutrition"` |
| `--analysis-mode` | str | Depends on type | top, recent | Apify sorting method (engagement vs date) | `--analysis-mode top` |
| `--selection-strategy` | str | Depends on type | contrastive, top | Video subset selection logic (80/20 vs top N) | `--selection-strategy contrastive` |
| `--video-count` | int | 100 (contrastive), 40 (top) | 10-500 | N videos to select per winning bucket | `--video-count 150` |
| `--date-filter` | str | last_90_days | last_N_days | Publication date range for filtering | `--date-filter last_90_days` |
| `--country-code` | str | US | US, BR, global | Geographic content filter via Apify proxy | `--country-code BR` |

### 4.2 Internal Configuration

```python
# Apify actor configuration
# NOTE: Actor IDs don't include version numbers. If Apify releases breaking changes:
# 1. Test new version in staging environment
# 2. Update actor ID below after validation
# 3. Document migration in change log
# 4. Monitor Apify marketplace for deprecation notices
APIFY_PROFILE_SCRAPER_ID = "GdWCkxBtKWOsKjdch"  # clockworks/tiktok-scraper (VERIFIED in production, last checked: 2025-01-28)

# Hashtag scraper - TO BE CONFIGURED BEFORE DEPLOYMENT
# How to obtain:
#   1. Visit https://apify.com/store
#   2. Search for "tiktok hashtag scraper" or "clockworks tiktok"
#   3. Select scraper that supports: hashtag URLs, 800+ results, video metadata
#   4. Copy actor ID from URL (format: username/actor-name or alphanumeric ID)
#   5. Test with sample hashtag (#nutrition) to verify schema matches Section 5.2
#   6. Replace "TBD" below with actual actor ID
APIFY_HASHTAG_SCRAPER_ID = "TBD"  # clockworks/tiktok-hashtag-scraper (MUST OBTAIN FROM APIFY MARKETPLACE - see instructions above)

APIFY_ACTOR_LAST_VALIDATED = "2025-01-28"  # Date actors were last tested (quarterly validation recommended)

# Apify scraping configuration
APIFY_SCRAPE_COUNT = 800  # Total videos to scrape per target
APIFY_TIMEOUT = 120  # Seconds before timeout
APIFY_RETRY_COUNT = 3  # Retry attempts on failure
APIFY_RETRY_BACKOFF = [5, 15, 45]  # Exponential backoff in seconds

# Date filtering configuration
DATE_FILTER_TIMEZONE = timezone.utc  # All date filtering performed in UTC
CLOCK_SKEW_TOLERANCE_HOURS = 24  # Accept timestamps up to 24h in future (clock skew)

# Winner analysis configuration
MIN_VIDEOS_FOR_ANALYSIS = 10  # Absolute minimum videos needed (hard stop if < 10)
TOP_PERFORMERS_FOR_ANALYSIS = 100  # Analyze top N to identify winning buckets (if ≥100 available)
TOP_BUCKETS_TO_PROCESS = 3  # Process top N buckets only (success-based)
MIN_WINNER_PERCENTAGE = 5.0  # Minimum 5% of winners to qualify bucket

# Selection strategy configuration
CONTRASTIVE_TOP_SPLIT = 0.8  # 80% top, 20% bottom for contrastive
MIN_VIDEOS_PER_BUCKET = 10  # Minimum videos to process bucket

# Bucket definitions (potential duration ranges in seconds)
# NOTE: These define the universe of 8 POTENTIAL buckets for classification.
# Winner analysis (Stage 1.3) selects TOP 3 where winners cluster.
# Not all buckets will be processed - only the 3 with highest winner concentration.
#
# Typical usage: 3 active buckets per analysis (e.g., 18-33s, 33-60s, 13-18s)
# Maximum capacity: 8 buckets available if winners spread evenly (rare)
BUCKET_DEFINITIONS = {
    "0-3s": (0, 3),
    "3-9s": (3, 9),
    "9-13s": (9, 13),
    "13-18s": (13, 18),
    "18-33s": (18, 33),
    "33-60s": (33, 60),
    "60-90s": (60, 90),
    "90-120s": (90, 120),
}

# Engagement score formula weights (for top mode)
ENGAGEMENT_SHARE_WEIGHT = 10  # 10x weight for shares in engagement score

# Interactive confirmation
AUTO_CONFIRM = False  # Skip Stage 1 confirmation prompt when True (CLI flag --auto-confirm overrides this)
```

---

## 5. Data Schemas

<!-- PURPOSE: Exact data structures. TI generator uses this for validation and type hints. -->

### 5.1 Input Schema

**CLI Parameters**: See FoundationCHILD.md Section 4

**Environment Variables**:
- `APIFY_API_KEY`: String, required, Apify account API key

### 5.2 Intermediate Schema (Apify Output)

**Apify Video Metadata Schema**: See FoundationCHILD.md Section 5.2

**Required Fields for Stage 1** (validation enforced):
- `id`: str - Unique video identifier (used for deduplication)
- `createTime`: int - Unix timestamp in UTC (for date filtering)
- `duration`: int - Video length in seconds (for bucket assignment)
- `playCount`: int - View count (for engagement sorting)
- `webVideoUrl`: str - TikTok web URL

**Optional Fields** (passed through to Stage 2, not validated):
- `shareCount`: int - Share count (informational, not used in Stage 1)
- `commentCount`: int - Comment count (informational, not used in Stage 1)
- `likeCount`: int - Like count (informational, not used in Stage 1)
- `videoMeta.downloadAddr`: str - MP4 download URL (used by Stage 2)
- `authorMeta.name`: str - Creator username (informational)

**Note**: All fields from Apify are passed through to `selected_videos.json`. Stage 1 only validates required fields for its own processing. Stage 2 uses additional fields (`videoMeta.downloadAddr` for video download).

### 5.3 Output Schema

**File 1**: `{bucket_base}/selected_videos.json` (per bucket)

```python
SelectedVideosSchema = {
    "bucket": str,              # Required, Bucket name, Example: "18-33s"
    "strategy": str,            # Required, ["contrastive", "top"], Example: "contrastive"
    "video_count": int,         # Required, N from --video-count, Example: 100
    "selected_count": int,      # Required, Actual videos selected, Example: 100
    "top_count": int,           # Required, Top performers selected, Example: 80
    "bottom_count": int,        # Required, Bottom performers selected, Example: 20
    "videos": list,             # Required, List of ApifyVideoMetadata objects
    "selection_date": str,      # Required, ISO timestamp, Example: "2025-01-28T10:30:00Z"
}
```

**File 2**: `{analysis_base}/winner_analysis.json`

```python
WinnerAnalysisSchema = {
    "top_100_distribution": dict,  # Required, {bucket: count}, Example: {"18-33s": 45, "33-60s": 30}
    "top_3_buckets": list,         # Required, [bucket names], Example: ["18-33s", "33-60s", "13-18s"]
    "winner_coverage": float,      # Required, Percentage, Example: 95.0
    "scrape_timestamp": str,       # Required, ISO timestamp from Stage 1.1, Example: "2025-01-28T10:30:00Z"
    "analysis_date": str,          # Required, ISO timestamp, Example: "2025-01-28T10:30:00Z"
}
```

**Example**:
```json
{
  "bucket": "18-33s",
  "strategy": "contrastive",
  "video_count": 100,
  "selected_count": 100,
  "top_count": 80,
  "bottom_count": 20,
  "videos": [
    {
      "id": "7428596413707144481",
      "createTime": 1704067200,
      "duration": 25,
      "playCount": 50000,
      "shareCount": 500,
      "commentCount": 250,
      "likeCount": 3500,
      "webVideoUrl": "https://www.tiktok.com/@user/video/123",
      "videoMeta": {
        "downloadAddr": "https://v16-webapp.tiktok.com/..."
      },
      "authorMeta": {
        "name": "@user"
      }
    }
  ],
  "selection_date": "2025-01-28T10:30:00Z"
}
```

**File 2**: `{analysis_base}/winner_analysis.json`

```python
WinnerAnalysisSchema = {
    "top_100_distribution": dict,  # Required, {bucket: count}, Example: {"18-33s": 45, "33-60s": 30}
    "top_3_buckets": list,         # Required, [bucket names], Example: ["18-33s", "33-60s", "13-18s"]
    "winner_coverage": float,      # Required, Percentage, Example: 95.0
    "scrape_timestamp": str,       # Required, ISO timestamp from Stage 1.1, Example: "2025-01-28T10:30:00Z"
    "analysis_date": str,          # Required, ISO timestamp
}
```

**Example**:
```json
{
  "top_100_distribution": {
    "18-33s": 45,
    "33-60s": 30,
    "13-18s": 20,
    "9-13s": 5
  },
  "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
  "winner_coverage": 95.0,
  "scrape_timestamp": "2025-01-28T10:30:00Z",
  "analysis_date": "2025-01-28T10:32:15Z"
}
```

---

## 6. Error Handling & Validation

<!-- PURPOSE: All error scenarios. TI generator uses this for try/catch blocks and assertions. -->

### 6.1 Input Validation

```python
def validate_cli_params(client_id, analysis_type, target, video_count, date_filter, country_code):
    """
    Validate CLI parameters before processing.

    Args:
        client_id: str, client identifier
        analysis_type: str, "hashtag" or "competitor" or "creator"
        target: str, target identifier
        video_count: int, videos per bucket
        date_filter: str, "last_N_days"
        country_code: str, "US" or "BR" or "global"

    Raises:
        ValueError: if validation fails with specific error message
    """
    # 1. Validate client_id (alphanumeric + underscore only)
    if not re.match(r'^[a-zA-Z0-9_]+$', client_id):
        raise ValueError(f"Invalid client_id: {client_id}. Must be alphanumeric + underscore.")

    # 2. Validate analysis_type
    if analysis_type not in ["hashtag", "competitor", "creator"]:
        raise ValueError(f"Invalid analysis_type: {analysis_type}. Must be hashtag, competitor, or creator.")

    # 3. Validate target format
    if analysis_type == "hashtag" and not target.startswith("#"):
        raise ValueError(f"Hashtag target must start with #. Got: {target}")
    elif analysis_type in ["competitor", "creator"] and not target.startswith("@"):
        raise ValueError(f"Profile target must start with @. Got: {target}")

    # 4. Validate video_count range
    if not 10 <= video_count <= 500:
        raise ValueError(f"video_count must be 10-500. Got: {video_count}")

    # 5. Validate date_filter format
    if not re.match(r'^last_\d+_days$', date_filter):
        raise ValueError(f"Invalid date_filter: {date_filter}. Format: last_N_days")

    # 6. Validate country_code
    if country_code not in ["US", "BR", "global"]:
        raise ValueError(f"Invalid country_code: {country_code}. Must be US, BR, or global.")

    # 7. Check Apify API key exists
    if not os.getenv("APIFY_API_KEY"):
        raise ValueError("APIFY_API_KEY environment variable not set")
```

### 6.2 Error Cases

| Error | Detection | Handling | User Message | Exit Code |
|-------|-----------|----------|--------------|-----------|
| Missing Apify API key | `os.getenv("APIFY_API_KEY")` | Fail-fast | `"APIFY_API_KEY environment variable not set. Set it with: export APIFY_API_KEY=your_key"` | 1 |
| Invalid CLI params | Regex validation | Fail-fast | `"Invalid {param}: {value}. Expected format: {format}"` | 2 |
| Apify timeout | Timeout exception | Retry 3x, then fail | `"Apify scraping timeout after 3 retries. Check network connection."` | 3 |
| Apify rate limit | HTTP 429 response | Wait + retry | `"Apify rate limit exceeded. Waiting 60s before retry..."` | 0 (warning) |
| < 100 videos scraped | Count check | Warn + continue | `"Only {count} videos scraped (expected 800). Proceeding with available data."` | 0 (warning) |
| 10%+ duplicate videos | Deduplication count check | Log count, continue | `"Removed {count} duplicate videos ({percent}%). Proceeding with {unique_count} unique videos."` | 0 (warning) |
| All videos are duplicates (> 95%) | Deduplication result check | Fail-fast | `"All scraped videos are duplicates ({unique_count} unique from {total_count} scraped). Data quality issue. Check target or Apify scraper configuration."` | 7 |
| All videos outside date range | Filter result check | Warn + relax filter | `"No videos in last {N} days. Relaxing to last 180 days..."` | 0 (warning) |
| **< 10 videos after filtering** | `len(videos) < 10` | **Fail-fast** | `"Insufficient videos for analysis. Need ≥10, got {count}. Try different target or relax date filter."` | **6** |
| **10-99 videos (degraded mode)** | `10 <= len(videos) < 100` | **Warn + continue** | `"Small dataset ({count} videos). Analyzing all available. Statistical validity may be limited. Recommended: ≥100 videos."` | **0 (warning)** |
| No winning buckets qualified (all < 5% winners) | Qualified buckets check | Fail-fast | `"No buckets qualified (≥5% winners required). Winner distribution too fragmented. Try different target or broader date range."` | 4 |
| Only 1-2 buckets qualified | Qualified buckets count | Warn + continue | `"Only {count} bucket(s) qualified (≥5% winners). Processing {count} bucket(s) instead of 3."` | 0 (warning) |
| Winning bucket empty | Bucket video count | Skip bucket | `"Bucket {bucket} has 0 videos. Skipping."` | 0 (warning) |
| User aborted at confirmation prompt | User input ('n') | Exit gracefully | `"Analysis aborted by user."` | 130 |
| Write permission denied | File write exception | Fail-fast | `"Cannot write to {path}. Check permissions."` | 5 |

### 6.3 Output Validation

```python
def validate_selected_videos(selected_videos, bucket, strategy, video_count):
    """
    Validate selected_videos.json before saving.

    Args:
        selected_videos: dict, selected videos data
        bucket: str, bucket name
        strategy: str, "contrastive" or "top"
        video_count: int, expected N

    Raises:
        AssertionError: if output schema invalid
    """
    # 1. Check required fields exist
    required_fields = ["bucket", "strategy", "video_count", "selected_count", "videos"]
    for field in required_fields:
        assert field in selected_videos, f"Missing required field: {field}"

    # 2. Validate bucket matches
    assert selected_videos["bucket"] == bucket, \
        f"Bucket mismatch: {selected_videos['bucket']} != {bucket}"

    # 3. Validate strategy matches
    assert selected_videos["strategy"] == strategy, \
        f"Strategy mismatch: {selected_videos['strategy']} != {strategy}"

    # 4. Validate video count
    if strategy == "contrastive":
        # Contrastive: Should have top_count + bottom_count = selected_count
        assert selected_videos["top_count"] + selected_videos["bottom_count"] == selected_videos["selected_count"], \
            f"Count mismatch: top_count + bottom_count != selected_count"

        # Check 80/20 split (allow ±5% tolerance)
        expected_top = int(video_count * 0.8)
        actual_top = selected_videos["top_count"]
        tolerance = int(video_count * 0.05)
        assert abs(actual_top - expected_top) <= tolerance, \
            f"Top count {actual_top} not close to 80% of {video_count} (expected ~{expected_top})"

    elif strategy == "top":
        # Top: Should have only top_count, bottom_count = 0
        assert selected_videos["bottom_count"] == 0, \
            f"Top strategy should have bottom_count=0, got {selected_videos['bottom_count']}"

    # 5. Validate videos list
    assert len(selected_videos["videos"]) == selected_videos["selected_count"], \
        f"Video list length {len(selected_videos['videos'])} != selected_count {selected_videos['selected_count']}"

    # 6. Validate Apify metadata schema for each video
    for video in selected_videos["videos"]:
        required_video_fields = ["id", "createTime", "duration", "playCount", "webVideoUrl"]
        for field in required_video_fields:
            assert field in video, f"Video missing required field: {field}"
```

---

## 7. Performance & Scalability

<!-- PURPOSE: Performance targets and bottlenecks. TI generator uses this for optimization. -->

### 7.1 Performance Baselines

Performance characteristics from production testing (not aspirational targets):

**Typical Performance**:
- **Apify Scraping (hashtag)**: 60-90s for 800 videos (network-dependent)
- **Apify Scraping (profile)**: 45-70s for 800 videos (slightly faster)
- **Deduplication**: 0.1s for 800 videos (set-based lookup)
- **Date Filtering**: 0.1-0.3s for 800 videos (pure Python, in-memory with validation)
- **Winner Analysis**: 0.2-0.5s for 100 videos (bucket counting + sorting)
- **Bucket Selection**: 0.1s per bucket (list slicing)
- **File I/O (JSON write)**: 0.3-0.5s per bucket (selected_videos.json)
- **Total Stage 1 Duration**: 1.5-2.5 minutes typical

**Acceptable Ranges**:
- **Apify Scraping**: Up to 120s acceptable (network variability, TikTok API response time)
- **Total Stage 1**: Up to 3 minutes acceptable (primarily Apify wait time)

**Degradation Handling**:
- **If Apify > 120s**: Retry with exponential backoff (already implemented in Section 2.3.1)
- **If total > 3 minutes**: Log warning, continue processing (not a failure condition)
- **No hard timeout**: Stage 1 is non-interactive, graceful completion prioritized over speed

**Performance Notes**:
- 85-90% of total time is Apify API wait (external dependency, not optimizable)
- Date filtering includes validation overhead (null checks, timezone conversion) vs simple filtering
- Deduplication adds negligible overhead (< 0.1s for set operations)

### 7.3 Bottlenecks & Mitigations

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| Apify scraping time | 60-120s (85-90% of total) | External API, network latency, TikTok response time | Cannot optimize (external dependency). Retry logic handles timeouts. | N/A |
| Apify rate limits | Blocks concurrent processing | 10 concurrent runs limit | Sequential processing (one target at a time). Acceptable for single-target analysis. | Low |
| Date filtering validation | 0.1-0.3s (includes validation) | Null checks, timezone conversion, future timestamp detection | Necessary overhead for data quality. No optimization needed. | Low |
| Deduplication | 0.1s for 800 videos | Set-based ID lookup | Already optimized (O(n) set operations). No further optimization needed. | Low |

### 7.4 Scalability Limits

- **Max videos per scrape**: 800 (Apify platform limitation, not RumiAI limitation)
- **Max targets per client**: Unlimited (directory structure supports arbitrary targets)
- **Bucket processing**: 3 active buckets per analysis (success-based selection from 8 potential buckets)
  - **Potential buckets**: 8 defined (0-3s through 90-120s)
  - **Active buckets**: Top 3 selected where winners cluster
  - **ML model capacity**: Up to 16 models possible (8 buckets × 2 algorithms), typically 6 models trained (3 active buckets × 2 algorithms)
- **Max video_count per bucket**: 500 (memory constraint: 500 videos × ~1KB = 500KB)

---

## 8. Testing Strategy

<!-- PURPOSE: Test plan. TI generator uses this to create test suite. -->

### 8.1 Unit Tests

- [ ] **Test Apify scraping**
  - Hashtag scraper returns 800 videos
  - Profile scraper returns 800 videos
  - Engagement score calculation correct (views × (1 + share_rate × 10))
  - Invalid Apify response (missing fields) handled gracefully
  - Country code parameter passed correctly to Apify
    - US mode sets proxyCountryCode: "US"
    - BR mode sets proxyCountryCode: "BR"
    - global mode omits proxyCountryCode parameter

- [ ] **Test date filtering**
  - Filters videos correctly (last_90_days)
  - Handles edge cases (all videos outside range, future timestamps)
  - Warns user if < 100 videos after filtering

- [ ] **Test winner analysis**
  - Identifies top 3 buckets correctly
  - Handles edge cases (< 100 videos total, winners spread evenly)
  - Calculates winner coverage correctly

- [ ] **Test bucket selection (contrastive)**
  - Selects 80% top + 20% bottom correctly
  - Handles edge cases (< N videos, empty bucket)
  - Warns user if insufficient videos

- [ ] **Test bucket selection (top)**
  - Selects top N correctly
  - Handles edge cases (< N videos, empty bucket)

- [ ] **Test input validation**
  - Invalid client_id (special characters) raises ValueError
  - Invalid analysis_type raises ValueError
  - Invalid target format raises ValueError
  - Invalid video_count (< 10 or > 500) raises ValueError
  - Missing Apify API key raises ValueError

- [ ] **Test output validation**
  - selected_videos.json schema valid
  - Video count matches strategy requirements
  - Apify metadata complete for all videos

**Negative Testing**:

- [ ] **Test malformed Apify responses**
  - Empty JSON array → Handle gracefully, log error "No videos returned by Apify"
  - Missing required fields (id, createTime, duration) → Skip video, log warning with video index
  - Invalid JSON structure → Fail-fast with parse error
  - Null video objects in array → Skip nulls, log count of skipped entries

- [ ] **Test invalid timestamp values**
  - Null create_time → Skip video with warning (already covered in date filtering validation)
  - Negative timestamp (-1704067200) → Skip video, log error "Invalid negative timestamp"
  - Timestamp overflow (999999999999) → Skip video if conversion fails
  - String timestamp ("2024-01-01") → Type error, skip video with error details

- [ ] **Test invalid engagement metric values**
  - playCount = 0 → Valid case (new video, just published)
  - playCount < 0 → Skip video, log error "Negative playCount: {value}"
  - playCount = null → Skip video, log warning "Missing playCount"
  - playCount = "invalid" (string) → Type error, skip video
  - All engagement metrics = 0 → Valid edge case (brand new video)

- [ ] **Test invalid duration values**
  - duration < 0 → Skip video, log error "Negative duration"
  - duration = 0 → Skip video, log error "Zero duration"
  - duration > 120 → Skip video, log warning "Duration exceeds max bucket (90-120s)"
  - duration = null → Skip video, log error "Missing duration"

- [ ] **Test all-videos-fail scenarios**
  - All videos have null timestamps → Fail-fast "No valid videos after filtering. All had invalid timestamps."
  - All videos are duplicates (>95%) → Fail-fast with exit code 7 (data quality issue)
  - All videos outside date range → Relax filter to last 180 days, warn user
  - < 10 videos remain after all validation → Fail-fast with exit code 6

**Data Validation Tests**:

- [ ] **Test engagement score calculation validation**
  - Verify sorted order after client-side sorting
  - Test with playCount values: 0, 1, 1000, 1000000, MAX_INT
  - Verify ties handled consistently (stable sort)

- [ ] **Test deduplication logic**
  - 0% duplicates → All videos retained
  - 10% duplicates → Correct count removed, first occurrence kept
  - 50% duplicates → Log warning, correct deduplication
  - 95% duplicates → Fail-fast (all duplicates edge case)
  - Duplicates with different engagement → Verify first (highest) kept

### 8.2 Integration Tests

- [ ] **End-to-end: Stage 0 → Stage 1 → Stage 2**
  - Run Stage 0 with test CLI params
  - Run Stage 1 (video discovery)
  - Validate selected_videos.json per bucket
  - Verify Stage 2 can load outputs without error

- [ ] **Apify integration test**
  - Test real Apify API call (hashtag scraper)
  - Test real Apify API call (profile scraper)
  - Verify metadata schema matches expectations

- [ ] **Error propagation**
  - Stage 0 missing config → Stage 1 fails with clear message
  - Apify timeout → Stage 1 retries 3x then fails
  - Stage 1 selection error → Stage 2 does not run

### 8.3 Test Data

**TI Generation Note**: The inline JSON examples below should be used to auto-generate fixture files in `tests/fixtures/` directory. TI should create these files during code generation to enable immediate test execution without manual setup.

**Fixture Files to Generate**:
- `tests/fixtures/apify_sample_response.json` - Sample Apify scraper output (for unit tests)
- `tests/fixtures/selected_videos_expected.json` - Expected Stage 1 output (for validation)
- `tests/fixtures/winner_analysis_expected.json` - Expected winner analysis output

---

**File**: `tests/fixtures/apify_sample_response.json`

```json
[
  {
    "id": "7428596413707144481",
    "createTime": 1704067200,
    "duration": 25,
    "playCount": 50000,
    "shareCount": 500,
    "commentCount": 250,
    "likeCount": 3500,
    "webVideoUrl": "https://www.tiktok.com/@user/video/123",
    "videoMeta": {
      "downloadAddr": "https://v16-webapp.tiktok.com/..."
    },
    "authorMeta": {
      "name": "@user"
    }
  }
]
```

**Expected Output**: `tests/fixtures/selected_videos_expected.json`

```json
{
  "bucket": "18-33s",
  "strategy": "contrastive",
  "video_count": 100,
  "selected_count": 100,
  "top_count": 80,
  "bottom_count": 20,
  "videos": [],
  "selection_date": "2025-01-28T10:30:00Z"
}
```

### 8.4 Test Execution

```bash
# Run unit tests
pytest tests/test_video_discovery.py -v

# Run integration tests
pytest tests/test_stage1_integration.py -v

# Run with coverage
pytest --cov=video_discovery --cov-report=html
```

### 8.4 Live Integration & Performance Tests

**Purpose**: Validate core business logic (success-based distribution, correct video selection) and performance targets using live Apify account.

**Prerequisites**:
- Live Apify account with API key
- Apify quota available (scraping 800 videos ~$4 per test)
- Test targets: `#nutrition` (hashtag), `@nike` (profile)

**Critical Validation Tests**:

- [ ] **Test 1.3: Success-Based Distribution Logic (CRITICAL)**
  - **Goal**: Verify winners cluster in specific buckets, not just high-volume buckets
  - **Setup**: Scrape 800 videos from `#nutrition` (top mode)
  - **Test Steps**:
    1. Run winner analysis on top 100 performers
    2. Bucket all 800 videos by duration
    3. Calculate: winner concentration % vs total volume %
  - **Assertions**:
    - Top 3 buckets selected based on winner concentration (not volume)
    - Example validation: If bucket A has 400 videos (50% volume) but only 5 winners (5%), and bucket B has 150 videos (18% volume) with 45 winners (45%), bucket B should be selected over bucket A
    - Winner coverage ≥ 90% (top 3 buckets contain ≥90% of top 100 performers)
  - **Expected**: Top 3 buckets represent ≥90% of winners, NOT ≥90% of volume

- [ ] **Test 1.4: Bucket Selection Correctness (CRITICAL)**
  - **Goal**: Verify correct videos selected with proper engagement sorting
  - **Setup**: Use winner analysis results from Test 1.3
  - **Test Steps**:
    1. For each winning bucket, apply contrastive selection (80/20 split)
    2. Extract engagement scores for all selected videos
    3. Validate top 80% have higher engagement than bottom 20%
  - **Assertions**:
    - Contrastive: 80 top videos have engagement ≥ all 20 bottom videos
    - Top strategy: 40 selected videos are actual top 40 by engagement
    - No overlap between top and bottom groups
  - **Expected**: min(top_80_engagement) > max(bottom_20_engagement)

- [ ] **Test 1.3 + 1.4: End-to-End Winner Selection Flow**
  - **Goal**: Validate integrated flow from scraping to final video selection
  - **Setup**: Full Stage 1 execution with `#nutrition`
  - **Test Steps**:
    1. Scrape 800 videos (Stage 1.1)
    2. Filter to last 90 days (Stage 1.2)
    3. Analyze top 100 winners (Stage 1.3)
    4. Select N videos per winning bucket (Stage 1.4)
  - **Assertions**:
    - selected_videos.json created for each winning bucket
    - Each bucket has exactly N videos (or all available if < N)
    - Metadata includes: id, createTime, duration, playCount, shareCount, webVideoUrl
    - Videos sorted correctly (engagement DESC for contrastive top 80)
  - **Expected**: 3 selected_videos.json files, ~300 videos total

**Performance Baseline Validation Tests**:

Purpose: Measure actual performance to validate baselines (not pass/fail against hard targets)

- [ ] **Measure Apify Scraping Performance**
  - **Hashtag scraper**: Scrape 800 videos from `#nutrition`
    - Measure: Total time from API call to dataset retrieval
    - Expected: 60-90s typical, up to 120s acceptable (baseline from Section 7.1)
    - Log: Actual time, compare to baseline range
    - Action: If > 120s, investigate network issues or Apify service degradation
  - **Profile scraper**: Scrape 800 videos from `@nike`
    - Expected: 45-70s typical, up to 120s acceptable
    - Log: Actual time, compare to baseline range

- [ ] **Measure End-to-End Stage 1 Performance**
  - **Full pipeline**: Scrape → Deduplicate → Filter → Analyze → Select
  - **Measure**: Total time from CLI invocation to final selected_videos.json
  - **Expected**: 1.5-2.5 minutes typical, up to 3 minutes acceptable (baseline from Section 7.1)
  - **Log breakdown by sub-stage**:
    - Stage 1.1 (Apify + deduplication): ~60-90s + 0.1s
    - Stage 1.2 (Date filter with validation): ~0.1-0.3s
    - Stage 1.3 (Winner analysis): ~0.2-0.5s
    - Stage 1.4 (Bucket selection): ~0.1s per bucket
    - File I/O: ~0.3-0.5s per bucket
  - **Action**: If total > 3 minutes, log warning (not a failure)

- [ ] **Measure Individual Component Performance**
  - **Deduplication**: 800 videos → measure time
    - Expected: < 0.1s (baseline from Section 7.1)
  - **Date filtering with validation**: 800 videos → measure time
    - Expected: 0.1-0.3s (baseline from Section 7.1)
  - **Winner analysis**: 100 videos → measure time
    - Expected: 0.2-0.5s (baseline from Section 7.1)

**Edge Case Validation** (Live Data):

- [ ] **Insufficient winners scenario**
  - **Setup**: Use niche hashtag with < 100 videos total
  - **Assert**: System handles gracefully, processes all available videos
  - **Expected**: Warning logged, continues with < 100 winners

- [ ] **All winners in 1-2 buckets scenario**
  - **Setup**: Find target where winners heavily cluster (e.g., all 15s videos)
  - **Assert**: Processes only 1-2 buckets (not forcing top 3)
  - **Expected**: Winner coverage ≥ 95% even with < 3 buckets

**Test Execution**:

```bash
# Set Apify API key
export APIFY_API_KEY="your_key_here"

# Run live integration tests (requires billing)
pytest tests/test_stage1_live.py -v --apify-live

# Run performance tests with profiling
pytest tests/test_stage1_live.py -v --apify-live --profile

# Generate performance report
pytest tests/test_stage1_live.py --apify-live --benchmark-json=performance_report.json
```

**Cost Estimate**: ~$12 per full test run (3 scrapes × $4 per 800 videos)

**Test Frequency**:
- Run before major releases (not in CI/CD)
- Run after Apify actor updates
- Run quarterly to validate performance baselines

**Load Testing (Manual, Optional)**:

Live integration tests above validate single-target processing. Multi-target and concurrency scenarios tested manually due to cost:

- **Sequential Processing Test** (Manual):
  - Process 3-5 targets sequentially (e.g., #nutrition, #fitness, #wellness)
  - Measure: Total time (~4-10 minutes for 3 targets)
  - Validate: No memory leaks, all targets complete successfully
  - Cost: $12-20 per test run (3-5 targets × $4)
  - Frequency: One-time validation, not automated

- **Rate Limit Test** (Manual):
  - Attempt 2-3 concurrent scrapes (Apify limit is 10, but test with small number)
  - Validate: Retry logic handles queuing or 429 responses if limit hit
  - Expected: All scrapes complete successfully (sequential retry if needed)
  - Frequency: One-time validation after Apify client library changes

- **Memory Stress Test** (Manual):
  - Process 1 target with video_count=500 (maximum)
  - Monitor: Peak memory usage during processing
  - Expected: Peak memory < 1GB (500 videos × ~1KB + processing overhead)
  - Frequency: After changes to data structures or batch processing logic

**Rationale for Manual Load Testing**:
- Load scenarios are rare (typical usage: 1 target at a time)
- Expensive to automate ($40+ per test run for sequential processing)
- One-time validation sufficient (not regression-prone)
- Manual execution provides flexibility to adjust test scale

---

## 9. Future Enhancements

<!-- PURPOSE: Planned improvements. TI generator ignores this section (not for current implementation). -->

### 9.1 Planned Improvements

- **Phase 2: Performance-Weighted Bucket Selection**
  - **Current**: Select buckets purely by winner concentration (count-based)
  - **Future**: Calculate average engagement per bucket, process high-engagement buckets even if lower volume
  - **Example**: "33-60s has only 5% of videos but 2x avg engagement → process it"
  - **Impact**: Identify underserved high-performing formats (market gaps)

- **Phase 3: Tiered Processing Based on Volume**
  - **Current**: Binary (process or skip)
  - **Future**: Different analysis depth based on video volume
    - HIGH (≥80 videos): Full contrastive analysis
    - MEDIUM (40-79 videos): Top-only analysis
    - LOW (10-39 videos): Descriptive stats only (no ML)
    - SKIP (< 10 videos): No processing
  - **Impact**: Balanced approach (deep analysis where it matters, light touch on edges)

- **Phase 4: Dynamic Thresholds Based on Dataset Size**
  - **Current**: Fixed thresholds (N=100 for contrastive, N=40 for top)
  - **Future**: Adjust thresholds based on total video count
    - If scraping 1000 videos, raise threshold to 100 (higher bar for processing)
    - If scraping 100 videos, lower threshold to 30 (adapt to data availability)
  - **Impact**: Context-aware processing (what's "sufficient" depends on dataset size)

### 9.2 Known Limitations

- **Hard 800-video limit**: Apify scrapers cap at 800 videos (TikTok platform limitation, not RumiAI)
- **No server-side date filtering**: Hashtag scraper doesn't support it (must filter client-side)
- **No parallel scraping**: Process one target at a time (Apify rate limits)
- **Fixed top 3 buckets**: No dynamic adjustment based on variance or engagement gaps
- **Contrastive strategy provides moderate contrast, not extreme contrast**: Bottom 20 videos selected from top 800 scraped (engagement-sorted DESC), meaning they're still relatively successful (50K-100K views). True "failure" videos (< 10K views) not included due to Apify engagement-sorted scraping. Learning signal differentiates "extremely viral" from "moderately viral" (not "viral" from "unsuccessful"). Sufficient for identifying top performer patterns but does not analyze why videos fail.
- **No Apify actor versioning**: Apify actor IDs don't include version numbers. Breaking changes require manual testing and migration. Mitigation: Quarterly validation of actors (Section 4.2 `APIFY_ACTOR_LAST_VALIDATED`), monitor Apify marketplace for deprecation notices, test new versions in staging before production deployment.

---

## 10. References & Related Docs

<!-- PURPOSE: Links to other documentation. TI generator uses this for additional context if needed. -->

### 10.1 Parent Document

- **MLPlanningv2.md Part 3: Stage 1** (lines 525-642)
  - High-level stage overview
  - Stage position in pipeline
  - Input/output contracts

### 10.2 Mother Document Foundation

- **FoundationCHILD.md** (shared foundation across all stages)
  - **Section 2 "Client Architecture"**: Directory paths used in this stage
  - **Section 3 "Configuration Dimensions"**: Target types, analysis modes, selection strategies
  - **Section 4 "CLI Command Structure"**: CLI parameters this stage reads
  - **Section 5 "Configuration Schemas"**: config.json, Apify metadata schema

**Key Sections Referenced in This Stage**:
- Section 2: Provides base directory paths for file I/O
- Section 3: Defines selection strategies and analysis modes
- Section 4: Defines CLI parameters this stage reads
- Section 5.2: Defines Apify metadata schema

### 10.3 Related Child Docs

- **VideoProcessingCHILD.md** (Stage 2)
  - Consumes `selected_videos.json` (output from this stage)
  - Processes videos through RumiAI pipeline

### 10.4 External References

- **Apify Documentation**:
  - Hashtag Scraper: https://apify.com/clockworks/tiktok-hashtag-scraper
  - Profile Scraper: https://apify.com/clockworks/tiktok-scraper
- **Apify SDK**: https://docs.apify.com/sdk/python
- **TikTok API Limits**: 800 videos per request (platform limitation)

### 10.5 Glossary

**All terminology definitions** are centralized in **FoundationCHILD.md Appendix A (Glossary)** to avoid duplication across stage documents.

**Shared domain terms** (defined in Foundation):
- Temporal Window, Bucket, Hook, Middle Segments, Closing
- Client, Target, Analysis Mode, Selection Strategy
- Contrastive Strategy, Top Strategy
- Winner, Winner Coverage, Success-Based Distribution

**Stage 1-specific terms** used in this document are defined inline where they first appear:
- "Winning buckets" (Section 2.3.3): Buckets where winners cluster (success-based selection)
- "Qualified buckets" (Section 2.3.3): Buckets that pass MIN_WINNER_PERCENTAGE threshold (≥5% winners)
- "Engagement snapshot" (Section 2.3.3): Point-in-time capture of TikTok engagement metrics
- "Degraded mode" (Section 2.3.3): Analysis with < 100 videos (limited statistical validity)

For complete terminology reference, consult **FoundationCHILD.md Appendix A**.

---

## Appendix A: Example Data

<!-- PURPOSE: Concrete examples. TI generator uses this to understand data format visually. -->

### A.1 Sample Apify Response (3 videos)

**Source**: Apify clockworks/tiktok-hashtag-scraper

```json
[
  {
    "id": "7428596413707144481",
    "createTime": 1704067200,
    "duration": 25,
    "playCount": 50000,
    "shareCount": 500,
    "commentCount": 250,
    "likeCount": 3500,
    "webVideoUrl": "https://www.tiktok.com/@user1/video/123",
    "videoMeta": {
      "downloadAddr": "https://v16-webapp.tiktok.com/video1.mp4"
    },
    "authorMeta": {
      "name": "@user1"
    }
  },
  {
    "id": "7428596413707144482",
    "createTime": 1704153600,
    "duration": 22,
    "playCount": 35000,
    "shareCount": 300,
    "commentCount": 180,
    "likeCount": 2500,
    "webVideoUrl": "https://www.tiktok.com/@user2/video/456",
    "videoMeta": {
      "downloadAddr": "https://v16-webapp.tiktok.com/video2.mp4"
    },
    "authorMeta": {
      "name": "@user2"
    }
  },
  {
    "id": "7428596413707144483",
    "createTime": 1704240000,
    "duration": 28,
    "playCount": 20000,
    "shareCount": 150,
    "commentCount": 120,
    "likeCount": 1800,
    "webVideoUrl": "https://www.tiktok.com/@user3/video/789",
    "videoMeta": {
      "downloadAddr": "https://v16-webapp.tiktok.com/video3.mp4"
    },
    "authorMeta": {
      "name": "@user3"
    }
  }
]
```

### A.2 Sample Winner Analysis Output

**File**: `winner_analysis.json`

```json
{
  "top_100_distribution": {
    "18-33s": 45,
    "33-60s": 30,
    "13-18s": 20,
    "9-13s": 5
  },
  "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
  "winner_coverage": 95.0,
  "scrape_timestamp": "2025-01-28T10:30:00Z",
  "analysis_date": "2025-01-28T10:32:15Z"
}
```

### A.3 Sample Selected Videos Output (Contrastive)

**File**: `bucket_18-33s/selected_videos.json`

```json
{
  "bucket": "18-33s",
  "strategy": "contrastive",
  "video_count": 100,
  "selected_count": 100,
  "top_count": 80,
  "bottom_count": 20,
  "videos": [
    {
      "id": "7428596413707144481",
      "createTime": 1704067200,
      "duration": 25,
      "playCount": 50000,
      "shareCount": 500,
      "commentCount": 250,
      "likeCount": 3500,
      "webVideoUrl": "https://www.tiktok.com/@user1/video/123",
      "videoMeta": {
        "downloadAddr": "https://v16-webapp.tiktok.com/video1.mp4"
      },
      "authorMeta": {
        "name": "@user1"
      }
    }
  ],
  "selection_date": "2025-01-28T10:30:00Z"
}
```

---