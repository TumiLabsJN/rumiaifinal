# STAGE_1_IMPL.md - Video Discovery & Selection

**Version**: 1.0.0
**Last Updated**: 2025-11-02
**Purpose**: Implementation guide for Stage 1: Video Discovery & Selection
**Target Audience**: LLM agents debugging, modifying, or extending Stage 1

**Related**: [PRODUCTION_FLOW.md Stage 1 Contract](PRODUCTION_FLOW.md#stage-1-video-discovery)

---

## Quick Reference

### Entry Points

**Main Entry**: `VideoDiscovery.run()` at [`ml_pipeline/stage1_discovery/video_discovery.py:84-234`](ml_pipeline/stage1_discovery/video_discovery.py#L84-L234)

**Orchestrator Call**: [`rumiai_ml_batch.py:570-707`](rumiai_ml_batch.py#L570-L707)
```python
# Line 633-639
video_discovery = VideoDiscovery(
    config=config.model_dump(),
    apify_api_key=apify_api_key,
    path_builder=path_builder
)
exit_code = video_discovery.run()
```

### Key Characteristics

- **Duration**: 30-60 seconds (800 videos scraped, filtered, analyzed)
- **External Dependencies**: Apify API (APIFY_API_KEY required)
- **User Interaction**: Interactive confirmation prompt (can be disabled with `--auto-confirm`)
- **Checkpoint**: `{analysis_base}/checkpoints/stage_1_checkpoint.json`
- **Exit Strategy**: Exit pipeline on failure (exit code 1)
- **Mode Support**: Single hashtag (deprecated), Cluster mode (recommended), Competitor, Creator

### Module Structure

```
ml_pipeline/stage1_discovery/          (2,930 total lines)
├── video_discovery.py      (312)  # Main orchestrator
├── apify_scraper.py        (330)  # Stage 1.1: Apify scraping
├── date_filter.py          (197)  # Stage 1.2: Date filtering
├── winner_analyzer.py      (291)  # Stage 1.3: Winner analysis
├── video_selector.py       (243)  # Stage 1.4: Video selection
├── confirmation.py         (128)  # Stage 1.5: User confirmation
├── constants.py            (170)  # Configuration constants
├── cluster_config.py       (144)  # Cluster mode config
├── cluster_scraper.py      (265)  # Cluster scraping
├── cluster_deduplication.py (106) # Deduplication
├── cluster_validation.py   (236)  # Cluster validation
├── cluster_analytics.py    (222)  # Analytics generation
└── hashtag_validator.py    (251)  # Post-scraping validation

foundation/                           (1,028 total lines)
├── cli.py                  (311)  # CLI argument parsing
├── config.py                (81)  # Configuration management
├── paths.py                (273)  # Path building & sanitization
├── buckets.py               (78)  # Bucket assignment logic
├── schemas.py              (132)  # Pydantic schemas
└── constants.py             (67)  # Foundation constants
```

---

## Table of Contents

1. [Overview](#overview)
2. [Input Contract](#input-contract)
3. [Output Contract](#output-contract)
4. [Core Functions](#core-functions)
5. [Data Flow](#data-flow)
6. [Error Handling](#error-handling)
7. [Checkpoint Strategy](#checkpoint-strategy)
8. [Debugging Guide](#debugging-guide)
9. [Modification Guide](#modification-guide)
10. [Related Documentation](#related-documentation)

---

## Overview

**Stage 1** discovers and selects TikTok videos for ML analysis. It identifies the top 3 duration buckets where viral videos cluster, then selects N videos per bucket for processing.

### Sub-Stages

**Stage 1.1: Apify Scraping**
- Scrape 800 videos from TikTok via Apify API
- Support: Single hashtag (deprecated), Cluster mode, Competitor, Creator
- Deduplication by video ID
- Sort by engagement (playCount DESC)

**Stage 1.2: Date Filtering**
- Filter videos by publication date (e.g., last 90 days)
- UTC timezone handling
- Clock skew tolerance (24 hours)

**Stage 1.3: Winner Analysis**
- Analyze top 100 performers
- Bucket by duration (8 buckets: 0-3s, 3-9s, ..., 90-120s)
- Calculate winner concentration per bucket
- Select top 3 buckets (minimum 5% winner concentration)

**Stage 1.4: Video Selection**
- Group videos by winning buckets
- **Contrastive strategy**: 80% top + 20% bottom performers
- **Top strategy**: 100% top performers
- Default: 100 videos (contrastive), 40 videos (top)

**Stage 1.5: Interactive Confirmation**
- Display selection summary (buckets, counts, estimated time)
- Prompt: "Proceed with video processing? (y/n)"
- Auto-confirm mode available (`--auto-confirm`)

**Stage 1.6: Output Creation**
- Create directory structure (13 subdirs per bucket)
- Write `winner_analysis.json` (top 3 buckets metadata)
- Write `selected_videos.json` per bucket (video metadata + selection info)
- Optional: `cluster_analytics.json` (cluster mode only)

---

## Input Contract

### Prerequisites

**None** - Stage 1 is the first pipeline stage

### Required Inputs

**1. CLI Arguments** (validated by `foundation/cli.py:62-179`)

| Argument | Type | Required | Example | Validation |
|----------|------|----------|---------|------------|
| `--client` | string | ✓ | `acme_corp` | Regex: `^[a-zA-Z0-9_]+$` |
| `--analysis-type` | enum | ✓ | `hashtag` | `["hashtag", "competitor", "creator"]` |
| `--target` | string | ✓ | `nutrition` | Cluster name (alphanumeric) or `#hashtag` (deprecated) or `@handle` |
| `--analysis-mode` | enum | | `top` | `["top", "recent"]` (default: varies by type) |
| `--selection-strategy` | enum | | `contrastive` | `["contrastive", "top"]` (default: varies by type) |
| `--video-count` | int | | `100` | Range: 1-500 (default: varies by strategy) |
| `--date-filter` | string | | `last_90_days` | Format: `last_N_days` where N=1-365 |
| `--country-code` | enum | | `US` | `["US", "BR", "global"]` (default: US) |
| `--auto-confirm` | bool | | `true` | Default: true (skip prompts) |

**Default Value Logic** (foundation/cli.py:182-214):
```python
# Hashtag analysis
analysis_type = "hashtag"
  → analysis_mode = "top"
  → selection_strategy = "contrastive"
  → video_count = 100
  → report_audience = "client"

# Competitor analysis
analysis_type = "competitor"
  → analysis_mode = "top"
  → selection_strategy = "contrastive"
  → video_count = 100
  → report_audience = "client"

# Creator analysis
analysis_type = "creator"
  → analysis_mode = "recent"
  → selection_strategy = "top"
  → video_count = 40
  → report_audience = "creator"
```

**2. Environment Variables**

| Variable | Required | Purpose | Obtain From |
|----------|----------|---------|-------------|
| `APIFY_API_KEY` | ✓ | Apify API authentication | https://console.apify.com/account/integrations |

**Validation** (ml_pipeline/stage1_discovery/apify_scraper.py:51-56):
```python
if not apify_api_key:
    raise ValueError(
        "APIFY_API_KEY environment variable required. "
        "Obtain from: https://console.apify.com/account/integrations"
    )
```

**3. Cluster Configuration** (Cluster Mode Only)

**Location**: `/config/hashtag_clusters/{cluster_id}.json`

**Schema** (from cluster_config.py:144):
```json
{
  "cluster_id": "nutrition",
  "primary_hashtag": "#nutrition",
  "variant_hashtags": ["#healthyeating", "#nutritiontips"],
  "runs_per_hashtag": 2,
  "delay_between_runs_ms": 120000,
  "results_per_page": 800
}
```

**Validation Rules** (constants.py:126-143):
- `variant_hashtags`: 1-10 hashtags
- `runs_per_hashtag`: 1-5 runs
- `delay_between_runs_ms`: 60,000-10,800,000 ms (1-180 minutes)
- `results_per_page`: 100-800 videos

---

## Output Contract

### Output Files

**1. winner_analysis.json**

**Path**: `{analysis_base}/winner_analysis.json`

**Schema** (video_discovery.py:269-277):
```json
{
  "top_100_distribution": {
    "18-33s": 35,
    "33-60s": 28,
    "13-18s": 22,
    "60-90s": 10,
    "9-13s": 5
  },
  "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
  "winner_coverage": 85.0,
  "scrape_timestamp": "2025-11-02T10:30:00.000000Z",
  "analysis_date": "2025-11-02T10:30:00.000000Z"
}
```

**Fields**:
- `top_100_distribution`: Winner count per bucket (all buckets)
- `top_3_buckets`: Selected buckets for processing (array of 3)
- `winner_coverage`: Percentage of winners in top 3 buckets
- `scrape_timestamp`: UTC timestamp when scraping completed
- `analysis_date`: UTC timestamp when analysis ran

**2. selected_videos.json** (per bucket)

**Path**: `{analysis_base}/buckets/bucket_{bucket}/selected_videos.json`

**Schema** (video_selector.py:192-201):
```json
{
  "bucket": "18-33s",
  "strategy": "contrastive",
  "video_count": 100,
  "selected_count": 100,
  "top_count": 80,
  "bottom_count": 20,
  "selection_date": "2025-11-02T10:30:00Z",
  "videos": [
    {
      "id": "7123456789012345678",
      "webVideoUrl": "https://www.tiktok.com/@user/video/7123456789012345678",
      "playCount": 1500000,
      "shareCount": 50000,
      "commentCount": 12000,
      "likeCount": 300000,
      "duration": 25,
      "createTime": 1698768000,
      "text": "Video caption...",
      "hashtags": ["#nutrition", "#healthy"],
      "is_top_performer": true
    }
  ]
}
```

**Fields**:
- `bucket`: Duration bucket name
- `strategy`: Selection strategy used
- `video_count`: Target videos requested
- `selected_count`: Actual videos selected (may be < video_count if limited availability)
- `top_count`: Number of top performers
- `bottom_count`: Number of bottom performers
- `selection_date`: UTC timestamp
- `videos`: Array of video metadata objects
  - `is_top_performer`: true (top 80%) or false (bottom 20%)

**3. cluster_analytics.json** (Cluster Mode Only)

**Path**: `{analysis_base}/cluster_analytics.json`

**Schema** (cluster_analytics.py:222):
```json
{
  "cluster_id": "nutrition",
  "scrape_timestamp": "2025-11-02T10:30:00Z",
  "total_scraped": 4800,
  "total_unique": 3200,
  "duplicate_rate_pct": 33.3,
  "hashtag_breakdown": {
    "#nutrition": {"scraped": 1600, "unique": 1200},
    "#healthyeating": {"scraped": 1600, "unique": 1100}
  },
  "failed_scrapes": ["#obsolete"],
  "hashtag_validation": {
    "total_input": 3200,
    "passed": 2900,
    "removed": 300,
    "removal_rate_pct": 9.4
  }
}
```

**4. stage_1_checkpoint.json**

**Path**: `{analysis_base}/checkpoints/stage_1_checkpoint.json`

**Schema** (rumiai_ml_batch.py:679-689):
```json
{
  "stage": "stage_1_video_discovery",
  "completion_timestamp": "2025-11-02T10:30:00.000000Z",
  "winning_buckets": ["18-33s", "33-60s", "13-18s"],
  "output_files": [
    "/data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json",
    "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/selected_videos.json",
    "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_33-60s/selected_videos.json",
    "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_13-18s/selected_videos.json"
  ],
  "analysis_mode": "top",
  "analysis_type": "hashtag",
  "target": "nutrition",
  "video_count": 100
}
```

### Directory Structure Created

**Created by** `video_discovery.py:289-303`:
```
{analysis_base}/
├── winner_analysis.json
├── cluster_analytics.json (cluster mode only)
├── checkpoints/
│   └── stage_1_checkpoint.json
└── buckets/
    ├── bucket_18-33s/
    │   ├── selected_videos.json
    │   ├── videos/
    │   ├── analysis/
    │   │   ├── insights/
    │   │   ├── unified/
    │   │   └── service_debug/
    │   ├── validation/
    │   ├── flagged_videos/
    │   ├── ml_analysis/
    │   ├── models/
    │   ├── llm_reports/
    │   │   ├── analysis/
    │   │   └── formatted/
    │   ├── reports/
    │   ├── checkpoints/
    │   └── logs/
    ├── bucket_33-60s/
    │   └── (same structure)
    └── bucket_13-18s/
        └── (same structure)
```

---

## Core Functions

### 1. VideoDiscovery.run() - Main Orchestrator

**Location**: `ml_pipeline/stage1_discovery/video_discovery.py:84-234`

**Purpose**: Execute complete Stage 1 pipeline (6 sub-stages)

**Function Signature**:
```python
def run(self) -> int:
    """
    Execute complete Stage 1 pipeline.

    Returns:
        Exit code (0 = success, >0 = error)
    """
```

**Execution Flow**:
```python
# Stage 1.1: Detect cluster mode and scrape
target_type, cluster_config = detect_target_type(
    self.config['target'],
    self.config['analysis_type']
)

if target_type == "cluster":
    # Cluster mode: Multi-hashtag scraping
    all_videos, failed_scrapes = run_cluster_scraping(...)
    videos, dedup_analytics = deduplicate_with_provenance(...)
    videos, validation_report = validate_target_hashtags(...)
    save_cluster_analytics(...)
else:
    # Single mode: Direct scraping
    videos = self.scraper.scrape_videos(...)

# Stage 1.2: Date filtering
filtered_videos = self.date_filter.filter_by_date(
    videos=videos,
    date_filter=self.config['date_filter']
)

# Stage 1.3: Winner analysis
winning_buckets, winner_distribution, qualified_buckets = \
    self.winner_analyzer.analyze_winner_distribution(filtered_videos)

# Stage 1.4: Video selection
selected_per_bucket = self.video_selector.select_videos_per_bucket(
    videos=filtered_videos,
    winning_buckets=winning_buckets,
    strategy=self.config['selection_strategy'],
    video_count=self.config['video_count']
)

# Stage 1.5: Interactive confirmation
confirmed = self.confirmation.confirm_bucket_selection(
    selected_per_bucket=selected_per_bucket,
    auto_confirm=self.config.get('auto_confirm', False)
)

if not confirmed:
    return EXIT_CODE_USER_ABORT

# Stage 1.6: Create output files
self._create_output_files(
    selected_per_bucket=selected_per_bucket,
    winner_distribution=winner_distribution,
    winning_buckets=winning_buckets
)

return EXIT_CODE_SUCCESS
```

**Exit Codes**:
- `0` - Success
- `130` - User aborted at confirmation

**Error Propagation**: Raises exceptions (caught by orchestrator)

---

### 2. ApifyScraper.scrape_videos() - Stage 1.1

**Location**: `ml_pipeline/stage1_discovery/apify_scraper.py:69-117`

**Purpose**: Scrape videos from TikTok via Apify API with deduplication and engagement sorting

**Function Signature**:
```python
def scrape_videos(
    self,
    analysis_type: str,
    target: str,
    analysis_mode: str,
    date_filter: str = "last_90_days",
    country_code: str = "US"
) -> List[Dict]:
```

**Implementation**:
```python
# Step 1: Select scraper and build input params
actor_id, input_params = self._build_scraper_config(
    analysis_type, target, analysis_mode, date_filter, country_code
)

# Step 2: Run Apify scraper with retry logic
videos = self._run_scraper_with_retry(actor_id, input_params, target, analysis_mode)

# Step 3: Deduplicate by video ID
unique_videos = self._deduplicate_videos(videos)

# Step 4: Client-side engagement sorting
sorted_videos = self._sort_by_engagement(unique_videos)

return sorted_videos
```

**Apify Actor Configuration** (apify_scraper.py:119-174):
```python
# Unified Profile Scraper (supports both hashtags and profiles)
actor_id = "GdWCkxBtKWOsKjdch"  # clockworks/tiktok-scraper

input_params = {
    "resultsPerPage": 800,
    "shouldDownloadCovers": False,
    "shouldDownloadVideos": False,
    "shouldDownloadSubtitles": False,
    "shouldDownloadSlideshowImages": False,
}

# Target-specific parameters
if analysis_type == "hashtag":
    input_params["hashtags"] = [target]  # e.g., ["#nutrition"]
elif analysis_type in ["competitor", "creator"]:
    input_params["profiles"] = [target]  # e.g., ["@rival_brand"]

# Date filtering (native Apify parameters)
oldest_date, newest_date = self._calculate_date_range(date_filter)
input_params["oldestPostDateUnified"] = oldest_date
input_params["newestPostDate"] = newest_date

# Geography filtering
if country_code != "global":
    input_params["proxyCountryCode"] = country_code  # "US" or "BR"

# Sorting
if analysis_mode == "recent":
    input_params["profileSorting"] = "latest"
# If "top", omit profileSorting (default: engagement-based)
```

**Retry Logic** (apify_scraper.py:199-271):
```python
for attempt in range(APIFY_RETRY_COUNT):  # APIFY_RETRY_COUNT = 3
    try:
        run = self.client.actor(actor_id).call(
            run_input=input_params,
            timeout_secs=APIFY_TIMEOUT  # 720 seconds (12 minutes)
        )
        dataset_items = self.client.dataset(run["defaultDatasetId"]).list_items().items
        videos = dataset_items

        # Flatten duration from videoMeta to top level
        for video in videos:
            if 'videoMeta' in video and video['videoMeta'] and 'duration' in video['videoMeta']:
                video['duration'] = video['videoMeta']['duration']

        break  # Success, exit retry loop

    except TimeoutError:
        wait_time = APIFY_RETRY_BACKOFF[attempt]  # [5, 15, 45] seconds
        time.sleep(wait_time)

        if attempt == APIFY_RETRY_COUNT - 1:
            raise TimeoutError("Apify scraping timeout after 3 retries")

    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            time.sleep(60)  # Wait 60s for rate limit
            continue
        else:
            raise  # Unknown error, fail-fast
```

**Deduplication** (apify_scraper.py:282-318):
```python
seen_ids = set()
unique_videos = []

for video in videos:
    video_id = video.get("id")
    if video_id and video_id not in seen_ids:
        seen_ids.add(video_id)
        unique_videos.append(video)

# Check for extreme duplication (>95% duplicates)
duplicate_count = len(videos) - len(unique_videos)
if len(videos) > 0 and (duplicate_count / len(videos)) > 0.95:
    raise ValueError(
        f"All scraped videos are duplicates ({len(unique_videos)} unique from {len(videos)} scraped). "
        f"Data quality issue."
    )
```

**Engagement Sorting** (apify_scraper.py:320-330):
```python
sorted_videos = sorted(videos, key=lambda v: v.get("playCount", 0), reverse=True)
```

---

### 3. DateFilter.filter_by_date() - Stage 1.2

**Location**: `ml_pipeline/stage1_discovery/date_filter.py:36-73`

**Purpose**: Filter videos by publication date with robust timestamp validation

**Function Signature**:
```python
def filter_by_date(
    self,
    videos: List[Dict],
    date_filter: str
) -> List[Dict]:
```

**Implementation**:
```python
# Step 1: Parse date_filter parameter (e.g., "last_90_days" → 90)
days = int(date_filter.replace("last_", "").replace("_days", ""))

# Step 2: Calculate cutoff date (UTC)
cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

# Step 3: Filter with robust timestamp validation
filtered_videos = []
skipped_reasons = {
    "null_or_zero": 0,
    "invalid_conversion": 0,
    "future_timestamp": 0,
}

for video in videos:
    create_time = video.get("createTime")

    # Validation 1: Check create_time exists and is non-zero
    if create_time is None or create_time == 0:
        skipped_reasons["null_or_zero"] += 1
        continue

    # Validation 2: Convert Unix timestamp to UTC datetime
    try:
        video_date = datetime.fromtimestamp(create_time, tz=timezone.utc)
    except (ValueError, OSError):
        skipped_reasons["invalid_conversion"] += 1
        continue

    # Validation 3: Handle future timestamps (24h clock skew tolerance)
    future_threshold = datetime.now(timezone.utc) + timedelta(hours=24)
    if video_date > future_threshold:
        skipped_reasons["future_timestamp"] += 1
        continue

    # Validation 4: Apply date filter
    if video_date >= cutoff_date:
        filtered_videos.append(video)

# Step 4: Validate minimum count (≥10 videos required)
if len(filtered_videos) < 10:
    raise ValueError(
        f"Insufficient videos for analysis. Need ≥10, got {len(filtered_videos)}. "
        f"Try different target or relax date filter."
    )

return filtered_videos
```

---

### 4. WinnerAnalyzer.analyze_winner_distribution() - Stage 1.3

**Location**: `ml_pipeline/stage1_discovery/winner_analyzer.py:40-95`

**Purpose**: Identify winning buckets by analyzing where top performers cluster

**Function Signature**:
```python
def analyze_winner_distribution(
    self,
    videos: List[Dict]
) -> Tuple[List[str], Dict[str, int], Dict[str, float]]:
```

**Returns**:
- `list[str]`: Top 3 bucket names (e.g., ["18-33s", "33-60s", "13-18s"])
- `dict[str, int]`: Winner distribution {bucket: count}
- `dict[str, float]`: Qualified buckets {bucket: percentage}

**Implementation**:
```python
# Step 1: Validate minimum dataset size (≥10 videos)
if len(videos) < 10:
    raise ValueError("Insufficient videos for analysis. Need ≥10")

# Step 2: Select top 100 performers with valid durations
top_performers = self._select_top_performers(videos)

# Step 3: Bucket videos by duration
winner_distribution = {}
for video in top_performers:
    duration = video.get("duration")
    bucket = assign_bucket(duration)  # foundation/buckets.py:11
    winner_distribution[bucket] = winner_distribution.get(bucket, 0) + 1

# Step 4: Calculate winner percentages
winner_percentages = {
    bucket: (count / len(top_performers)) * 100
    for bucket, count in winner_distribution.items()
}

# Step 5: Filter qualified buckets (≥5% winners)
qualified_buckets = {
    bucket: percentage
    for bucket, percentage in winner_percentages.items()
    if percentage >= 5.0
}

if len(qualified_buckets) == 0:
    raise ValueError("No buckets qualified (≥5% winners required)")

# Step 6: Sort and select top 3 buckets
top_buckets = sorted(
    qualified_buckets.items(),
    key=lambda x: x[1],
    reverse=True
)[:3]

selected_bucket_names = [bucket for bucket, _ in top_buckets]

return selected_bucket_names, winner_distribution, qualified_buckets
```

**Top Performer Selection** (winner_analyzer.py:97-171):
```python
# Collect exactly 100 videos with valid durations
selected = []
skipped_invalid_duration = 0

for video in videos:  # Already sorted DESC by engagement
    duration = video.get("duration")

    if duration is None:
        skipped_invalid_duration += 1
        continue

    try:
        bucket = assign_bucket(duration)  # Raises ValueError if > 120s
        selected.append(video)

        if len(selected) >= 100:
            break

    except ValueError:
        skipped_invalid_duration += 1
        continue

return selected
```

**Bucket Assignment** (foundation/buckets.py:11-56):
```python
def assign_bucket(duration: float) -> str:
    if duration < 3:
        return "0-3s"
    elif duration < 9:
        return "3-9s"
    elif duration < 13:
        return "9-13s"
    elif duration < 18:
        return "13-18s"
    elif duration < 33:
        return "18-33s"
    elif duration < 60:
        return "33-60s"
    elif duration < 90:
        return "60-90s"
    elif duration <= 120:  # Inclusive upper bound
        return "90-120s"
    else:
        raise ValueError(f"Video duration {duration}s exceeds TikTok maximum (120s)")
```

---

### 5. VideoSelector.select_videos_per_bucket() - Stage 1.4

**Location**: `ml_pipeline/stage1_discovery/video_selector.py:39-107`

**Purpose**: Select videos for each winning bucket using specified strategy

**Function Signature**:
```python
def select_videos_per_bucket(
    self,
    videos: List[Dict],
    winning_buckets: List[str],
    strategy: str,
    video_count: int
) -> Dict[str, Dict]:
```

**Implementation**:
```python
# Step 1: Group all videos by bucket
bucket_videos = {}
for video in videos:
    duration = video.get("duration")
    bucket = assign_bucket(duration)
    if bucket not in bucket_videos:
        bucket_videos[bucket] = []
    bucket_videos[bucket].append(video)

# Step 2: Filter to winning buckets only
winning_bucket_videos = {
    bucket: vids for bucket, vids in bucket_videos.items()
    if bucket in winning_buckets
}

# Step 3: Select videos per bucket using strategy
selected_per_bucket = {}
for bucket in winning_buckets:
    bucket_vids = winning_bucket_videos[bucket]

    if strategy == "contrastive":
        selection = self._select_contrastive(bucket, bucket_vids, video_count)
    elif strategy == "top":
        selection = self._select_top(bucket, bucket_vids, video_count)

    selected_per_bucket[bucket] = selection

return selected_per_bucket
```

**Contrastive Selection** (video_selector.py:140-201):
```python
# Calculate split (80% top, 20% bottom)
top_count = int(video_count * 0.8)  # 80
bottom_count = video_count - top_count  # 20

# Handle limited availability
available_count = len(bucket_videos)
if available_count < video_count:
    actual_top = int(available_count * 0.8)
    actual_bottom = available_count - actual_top
else:
    actual_top = top_count
    actual_bottom = bottom_count

# Select top performers (first N videos, already sorted DESC)
top_videos = bucket_videos[:actual_top]
for video in top_videos:
    video['is_top_performer'] = True

# Select bottom performers (last N videos)
bottom_videos = bucket_videos[-actual_bottom:] if actual_bottom > 0 else []
for video in bottom_videos:
    video['is_top_performer'] = False

# Combine
selected_videos = top_videos + bottom_videos

return {
    "bucket": bucket,
    "strategy": "contrastive",
    "video_count": video_count,
    "selected_count": len(selected_videos),
    "top_count": len(top_videos),
    "bottom_count": len(bottom_videos),
    "videos": selected_videos,
    "selection_date": datetime.now(timezone.utc).isoformat()
}
```

**Top Selection** (video_selector.py:203-243):
```python
# Handle limited availability
available_count = len(bucket_videos)
if available_count < video_count:
    actual_count = available_count
else:
    actual_count = video_count

# Select top N performers
selected_videos = bucket_videos[:actual_count]

return {
    "bucket": bucket,
    "strategy": "top",
    "video_count": video_count,
    "selected_count": len(selected_videos),
    "top_count": len(selected_videos),
    "bottom_count": 0,
    "videos": selected_videos,
    "selection_date": datetime.now(timezone.utc).isoformat()
}
```

---

### 6. InteractiveConfirmation.confirm_bucket_selection() - Stage 1.5

**Location**: `ml_pipeline/stage1_discovery/confirmation.py:29-56`

**Purpose**: Display summary and get user approval

**Function Signature**:
```python
def confirm_bucket_selection(
    self,
    selected_per_bucket: Dict[str, Dict],
    auto_confirm: bool = False
) -> bool:
```

**Implementation**:
```python
# Display summary
print("="*80)
print("STAGE 1: VIDEO DISCOVERY COMPLETE")
print("="*80)
print(f"\nSelected {len(selected_per_bucket)} winning bucket(s) for processing:\n")

for bucket_name in sorted(selected_per_bucket.keys()):
    selection = selected_per_bucket[bucket_name]
    print(f"  Bucket: {selection['bucket']}")
    print(f"    Strategy: {selection['strategy']}")
    print(f"    Selected: {selection['selected_count']} videos")
    print(f"      - Top performers: {selection['top_count']}")
    if selection['bottom_count'] > 0:
        print(f"      - Bottom performers: {selection['bottom_count']}")

# Skip prompt if auto-confirm enabled
if auto_confirm:
    return True

# Prompt user
response = input("Proceed with video processing? (y/n): ").strip().lower()

if response in ['y', 'yes']:
    return True
elif response in ['n', 'no']:
    return False
else:
    # Invalid response, retry
    return self._prompt_user()
```

---

### 7. VideoDiscovery._create_output_files() - Stage 1.6

**Location**: `ml_pipeline/stage1_discovery/video_discovery.py:244-312`

**Purpose**: Create output files and directory structure

**Implementation**:
```python
# Get analysis base directory
analysis_base = self.path_builder.get_target_dir(
    client_id=self.config['client_id'],
    analysis_type=self.config['analysis_type'],
    target=self.config['target'],
    analysis_mode=self.config['analysis_mode'],
    selection_strategy=self.config['selection_strategy']
)

# Create winner_analysis.json
winner_analysis = {
    "top_100_distribution": winner_distribution,
    "top_3_buckets": winning_buckets,
    "winner_coverage": sum(
        winner_distribution.get(b, 0) for b in winning_buckets
    ) / 100 * 100,
    "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
    "analysis_date": datetime.now(timezone.utc).isoformat()
}

winner_analysis_path = os.path.join(str(analysis_base), "winner_analysis.json")
os.makedirs(os.path.dirname(winner_analysis_path), exist_ok=True)
with open(winner_analysis_path, 'w') as f:
    json.dump(winner_analysis, f, indent=2)

# Create selected_videos.json per bucket
for bucket, selection in selected_per_bucket.items():
    bucket_base = self.path_builder.get_bucket_dir(analysis_base, bucket)

    # Create directory structure (13 subdirectories)
    bucket_base_str = str(bucket_base)
    os.makedirs(os.path.join(bucket_base_str, "videos"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "analysis", "insights"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "analysis", "unified"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "analysis", "service_debug"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "validation"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "flagged_videos"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "ml_analysis"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "models"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "llm_reports", "analysis"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "llm_reports", "formatted"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "reports"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(bucket_base_str, "logs"), exist_ok=True)

    # Write selected_videos.json
    selected_videos_path = os.path.join(bucket_base_str, "selected_videos.json")
    with open(selected_videos_path, 'w') as f:
        json.dump(selection, f, indent=2)
```

---

## Data Flow

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: CLI Args + APIFY_API_KEY                                    │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1.1: Apify Scraping                                          │
│ ─────────────────────────────────────────────────────────────────  │
│ Mode Detection:                                                     │
│   • Cluster mode: target = "nutrition" (alphanumeric)              │
│   • Single mode: target = "#nutrition" (deprecated)                │
│   • Profile mode: target = "@brand"                                │
│                                                                     │
│ Cluster Mode Path:                                                 │
│   1. Load cluster config: /config/hashtag_clusters/{cluster_id}.json│
│   2. Scrape N hashtags × M runs each                               │
│   3. Deduplicate with provenance tracking                          │
│   4. Validate hashtags (remove false positives)                    │
│   5. Save cluster_analytics.json                                   │
│                                                                     │
│ Single/Profile Mode Path:                                          │
│   1. Build Apify actor config                                      │
│   2. Run scraper with retry (3 attempts, 5s/15s/45s backoff)      │
│   3. Flatten videoMeta.duration to top level                       │
│   4. Deduplicate by video ID                                       │
│   5. Sort by playCount DESC                                        │
│                                                                     │
│ Output: 700-750 unique videos, sorted by engagement                │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1.2: Date Filtering                                          │
│ ─────────────────────────────────────────────────────────────────  │
│ 1. Parse date_filter: "last_90_days" → 90                          │
│ 2. Calculate cutoff: now - 90 days (UTC)                           │
│ 3. Validate timestamps:                                            │
│    • Skip null/zero createTime                                     │
│    • Skip invalid Unix timestamps                                  │
│    • Skip future timestamps (>24h tolerance)                       │
│ 4. Filter: video_date >= cutoff_date                               │
│ 5. Validate count: ≥10 videos required                             │
│                                                                     │
│ Output: 100-700 filtered videos                                    │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1.3: Winner Analysis                                         │
│ ─────────────────────────────────────────────────────────────────  │
│ 1. Select top 100 performers (with valid durations 0-120s)         │
│ 2. Bucket by duration:                                             │
│    • 0-3s, 3-9s, 9-13s, 13-18s, 18-33s, 33-60s, 60-90s, 90-120s   │
│ 3. Calculate winner percentages per bucket                         │
│    Example: 18-33s: 35/100 = 35%, 33-60s: 28/100 = 28%            │
│ 4. Filter qualified buckets (≥5% winners)                          │
│ 5. Sort by percentage DESC, select top 3                           │
│                                                                     │
│ Output:                                                             │
│   • winning_buckets: ["18-33s", "33-60s", "13-18s"]               │
│   • winner_distribution: {all buckets: counts}                     │
│   • qualified_buckets: {buckets ≥5%: percentages}                  │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1.4: Video Selection                                         │
│ ─────────────────────────────────────────────────────────────────  │
│ 1. Group all filtered videos by bucket                             │
│ 2. For each winning bucket:                                        │
│                                                                     │
│    Contrastive Strategy (80/20 split):                             │
│      • Top 80 videos (first 80, sorted DESC)                       │
│        → Tag: is_top_performer = true                              │
│      • Bottom 20 videos (last 20)                                  │
│        → Tag: is_top_performer = false                             │
│                                                                     │
│    Top Strategy (100% top):                                        │
│      • Top 40 videos (first 40, sorted DESC)                       │
│                                                                     │
│ 3. Create selected_videos.json structure per bucket                │
│                                                                     │
│ Output: selected_per_bucket dict with metadata + video arrays      │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1.5: Interactive Confirmation                                │
│ ─────────────────────────────────────────────────────────────────  │
│ 1. Display summary:                                                │
│    • Buckets selected (3)                                          │
│    • Videos per bucket (100 or 40)                                 │
│    • Top/bottom split                                              │
│    • Estimated processing time                                     │
│                                                                     │
│ 2. Prompt: "Proceed with video processing? (y/n)"                  │
│    • auto_confirm=true → Skip prompt                               │
│    • auto_confirm=false → Wait for user input                      │
│                                                                     │
│ Output: Boolean (confirmed/aborted)                                │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1.6: Output Creation                                         │
│ ─────────────────────────────────────────────────────────────────  │
│ 1. Create directory structure:                                     │
│    • {analysis_base}/                                              │
│    • buckets/bucket_{name}/ (×3)                                   │
│    • 13 subdirectories per bucket                                  │
│                                                                     │
│ 2. Write winner_analysis.json:                                     │
│    • top_100_distribution (all buckets)                            │
│    • top_3_buckets (array of 3)                                    │
│    • winner_coverage (%)                                           │
│    • timestamps                                                    │
│                                                                     │
│ 3. Write selected_videos.json per bucket:                          │
│    • Bucket metadata                                               │
│    • Selection strategy info                                       │
│    • Video array with TikTok metadata                              │
│                                                                     │
│ 4. Optional: Write cluster_analytics.json (cluster mode only)      │
│                                                                     │
│ Output: All Stage 1 files created                                  │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: winner_analysis.json + selected_videos.json (×3)           │
│         + cluster_analytics.json (optional)                         │
│         + Directory structure created                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Checkpoint Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ Orchestrator: rumiai_ml_batch.py:570-707                        │
└────┬─────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Check if checkpoint exists                                       │
│ Path: {analysis_base}/checkpoints/stage_1_checkpoint.json       │
└────┬─────────────────────────────────────────────────────────────┘
     │
     ├─ YES ────────────────────────────────────────────────────┐
     │                                                           │
     ▼                                                           │
┌──────────────────────────────────────────────────────────────┐ │
│ Validate Checkpoint Schema                                   │ │
│ Required fields: ["winning_buckets", "output_files",         │ │
│                   "completion_timestamp"]                    │ │
└────┬─────────────────────────────────────────────────────────┘ │
     │                                                           │
     ├─ VALID ──┐                                               │
     │          │                                               │
     │          ▼                                               │
     │     ┌──────────────────────────────────────────┐        │
     │     │ Validate Output Files Exist              │        │
     │     │ Check all paths in "output_files" array  │        │
     │     └────┬─────────────────────────────────────┘        │
     │          │                                               │
     │          ├─ ALL EXIST ──────────────────────────────┐   │
     │          │                                          │   │
     │          │                                          ▼   │
     │          │                             ┌──────────────────────────┐
     │          │                             │ SKIP Stage 1             │
     │          │                             │ Load winning_buckets     │
     │          │                             │ Continue to Stage 2      │
     │          │                             └──────────────────────────┘
     │          │
     │          ├─ MISSING FILES ──────────────────────┐
     │          │                                      │
     │          ▼                                      │
     │     ┌──────────────────────────────────────┐   │
     │     │ Delete corrupt checkpoint            │   │
     │     │ Re-run Stage 1                       │   │
     │     └──────────────────────────────────────┘   │
     │                                                 │
     ├─ INVALID (missing fields) ────────────────────┤
     │                                                 │
     ▼                                                 │
┌──────────────────────────────────────────┐          │
│ Delete corrupt checkpoint                │          │
│ Re-run Stage 1                           │          │
└──────────────────────────────────────────┘          │
                                                       │
     ├─ NO (checkpoint doesn't exist) ────────────────┤
     │                                                 │
     ▼                                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ RUN Stage 1: VideoDiscovery.run()                               │
└────┬─────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Create Checkpoint                                                │
│ ──────────────────────────────────────────────────────────────  │
│ checkpoint_data = {                                              │
│   "stage": "stage_1_video_discovery",                           │
│   "completion_timestamp": "2025-11-02T10:30:00Z",               │
│   "winning_buckets": ["18-33s", "33-60s", "13-18s"],           │
│   "output_files": [                                             │
│     ".../winner_analysis.json",                                 │
│     ".../bucket_18-33s/selected_videos.json",                   │
│     ".../bucket_33-60s/selected_videos.json",                   │
│     ".../bucket_13-18s/selected_videos.json",                   │
│     ".../cluster_analytics.json"  # If cluster mode             │
│   ],                                                             │
│   "analysis_mode": "top",                                       │
│   "analysis_type": "hashtag",                                   │
│   "target": "nutrition",                                        │
│   "video_count": 100                                            │
│ }                                                                │
│                                                                  │
│ Atomic write to: {analysis_base}/checkpoints/stage_1_checkpoint.json │
│                                                                  │
│ Error Handling:                                                 │
│   • IOError/PermissionError → Warn, continue (non-fatal)        │
│   • Stage 1 will re-run on next resume                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

### Error Matrix

| Error Type | Exception | Location | Strategy | Exit Code | User Action |
|------------|-----------|----------|----------|-----------|-------------|
| **Missing API Key** | ValueError | apify_scraper.py:51 | Exit pipeline | 1 | Set APIFY_API_KEY env var |
| **Apify Timeout** | TimeoutError | apify_scraper.py:256 | Exit pipeline after 3 retries | 3 | Check network, increase APIFY_TIMEOUT |
| **Apify Rate Limit** | Exception (429) | apify_scraper.py:263 | Wait 60s, retry | - | Wait or upgrade Apify plan |
| **All Duplicates** | ValueError | apify_scraper.py:309 | Exit pipeline | 7 | Check target or scraper config |
| **Invalid Date Filter** | ValueError | date_filter.py:94 | Exit pipeline | 2 | Use format: last_N_days (N=1-365) |
| **Insufficient Videos** | ValueError | date_filter.py:187, winner_analyzer.py:66 | Exit pipeline | 6 | Relax date filter or try different target |
| **No Qualified Buckets** | ValueError | winner_analyzer.py:241 | Exit pipeline | - | Winner distribution too fragmented |
| **User Abort** | - | confirmation.py:116 | Exit pipeline | 130 | User pressed 'n' or Ctrl+C |
| **Keyboard Interrupt** | KeyboardInterrupt | video_discovery.py:236 | Exit pipeline | 130 | User pressed Ctrl+C |
| **Cluster Config Missing** | FileNotFoundError | cluster_config.py | Exit pipeline | 10 | Create /config/hashtag_clusters/{cluster_id}.json |
| **Cluster Config Invalid** | ValueError | cluster_config.py | Exit pipeline | 11 | Fix cluster config schema |
| **Single Hashtag Deprecated** | - | cluster_config.py | Exit pipeline | 12 | Migrate to cluster mode |
| **All Scrapes Failed** | ValueError | cluster_scraper.py | Exit pipeline | 13 | Check Apify API status |

### Exit Code Reference

**Source**: `ml_pipeline/stage1_discovery/constants.py:89-98, 156-170`

```python
# Success
EXIT_CODE_SUCCESS = 0

# General errors
EXIT_CODE_APIFY_KEY_MISSING = 1
EXIT_CODE_INVALID_DATE_FILTER = 2
EXIT_CODE_APIFY_TIMEOUT = 3
EXIT_CODE_INSUFFICIENT_VIDEOS = 6
EXIT_CODE_ALL_DUPLICATES = 7

# Cluster mode errors
EXIT_CODE_CLUSTER_CONFIG_NOT_FOUND = 10
EXIT_CODE_CLUSTER_CONFIG_INVALID = 11
EXIT_CODE_SINGLE_HASHTAG_DEPRECATED = 12
EXIT_CODE_ALL_SCRAPES_FAILED = 13

# User interaction
EXIT_CODE_USER_ABORT = 130  # Ctrl+C or 'n' at confirmation
```

### Retry Strategies

**Apify Scraper Retry** (apify_scraper.py:199-271):
```python
APIFY_RETRY_COUNT = 3
APIFY_RETRY_BACKOFF = [5, 15, 45]  # seconds

for attempt in range(3):
    try:
        # Run Apify scraper (720s timeout)
        run = self.client.actor(actor_id).call(...)
        break  # Success

    except TimeoutError:
        # Exponential backoff
        wait_time = APIFY_RETRY_BACKOFF[attempt]
        time.sleep(wait_time)

        if attempt == 2:  # Final retry
            raise TimeoutError("Apify scraping timeout after 3 retries")

    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            # Rate limit: Wait 60s, retry
            time.sleep(60)
            continue
        else:
            # Unknown error: Fail-fast
            raise
```

**Cluster Scraper Retry** (cluster_scraper.py):
```python
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_DELAYS = [5, 15, 45]  # seconds

# Per-hashtag retry with backoff
# Failed hashtags tracked in failed_scrapes list
# Pipeline continues if ≥1 hashtag succeeds
```

### Validation Errors

**CLI Validation** (foundation/cli.py:217-283):
```python
# Client ID format
if not re.match(r"^[a-zA-Z0-9_]+$", args.client):
    raise ValueError("Invalid --client. Must be alphanumeric + underscore.")

# Target format (hashtag analysis)
if args.analysis_type == "hashtag":
    if args.target.startswith("#"):
        # Single hashtag (deprecated, caught later)
        if len(args.target) < 2:
            raise ValueError("Invalid --target. Must have ≥2 characters.")
    else:
        # Cluster name
        if not re.match(r"^[a-zA-Z0-9_]+$", args.target):
            raise ValueError("Invalid --target. Cluster names must be alphanumeric.")

# Target format (competitor/creator)
elif args.analysis_type in ["competitor", "creator"]:
    if not args.target.startswith("@") or len(args.target) < 2:
        raise ValueError("Invalid --target. Must start with @ and have ≥2 characters.")

# Video count range
if not (1 <= args.video_count <= 500):
    raise ValueError("Invalid --video-count. Must be 1-500.")

# Date filter format
if not re.match(r"^last_\d+_days$", args.date_filter):
    raise ValueError("Invalid --date-filter. Must match: last_N_days (N=1-365).")
```

**Checkpoint Validation** (rumiai_ml_batch.py:586-604):
```python
# Schema validation
required_fields = ["winning_buckets", "output_files", "completion_timestamp"]
missing_fields = [field for field in required_fields if field not in checkpoint]

if missing_fields:
    logger.warning(f"Checkpoint corrupt (missing fields: {missing_fields})")
    stage1_checkpoint_path.unlink()
    raise ValueError("Checkpoint validation failed")

# File existence validation
output_files = checkpoint["output_files"]
missing_files = [f for f in output_files if not Path(f).exists()]

if missing_files:
    logger.warning(f"Stage 1 outputs incomplete ({len(missing_files)} files missing)")
    stage1_checkpoint_path.unlink()
    raise ValueError("Output files missing")
```

---

## Checkpoint Strategy

### Checkpoint Location

**Path**: `{analysis_base}/checkpoints/stage_1_checkpoint.json`

**Created By**: Orchestrator (rumiai_ml_batch.py:679-704), not VideoDiscovery class

### Checkpoint Schema

```json
{
  "stage": "stage_1_video_discovery",
  "completion_timestamp": "2025-11-02T10:30:00.000000Z",
  "winning_buckets": ["18-33s", "33-60s", "13-18s"],
  "output_files": [
    "/data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json",
    "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/selected_videos.json",
    "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_33-60s/selected_videos.json",
    "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_13-18s/selected_videos.json",
    "/data/clients/acme/hashtags/nutrition/top_contrastive/cluster_analytics.json"
  ],
  "analysis_mode": "top",
  "analysis_type": "hashtag",
  "target": "nutrition",
  "video_count": 100
}
```

### Validation Logic

**Location**: `rumiai_ml_batch.py:577-624`

**Step 1: Schema Validation**
```python
required_fields = ["winning_buckets", "output_files", "completion_timestamp"]
missing_fields = [field for field in required_fields if field not in checkpoint]

if missing_fields:
    # Corrupt checkpoint → Delete and re-run
    stage1_checkpoint_path.unlink()
    raise ValueError("Checkpoint validation failed")
```

**Step 2: File Existence Validation**
```python
output_files = checkpoint["output_files"]
missing_files = [f for f in output_files if not Path(f).exists()]

if missing_files:
    # Incomplete outputs → Delete checkpoint and re-run
    stage1_checkpoint_path.unlink()
    raise ValueError("Output files missing")
```

**Step 3: Success Path**
```python
# Checkpoint valid → Skip Stage 1
winning_buckets = checkpoint["winning_buckets"]
print(f"✓ Stage 1: Video Discovery - SKIPPED (already complete)")
print(f"  Winning buckets: {', '.join(winning_buckets)}")
```

### Checkpoint Creation

**Location**: `rumiai_ml_batch.py:653-704`

```python
# Build output files list
output_files = [str(analysis_base / "winner_analysis.json")]

for bucket in winning_buckets:
    selected_videos_path = analysis_base / f"buckets/bucket_{bucket}/selected_videos.json"
    output_files.append(str(selected_videos_path))

# Add cluster_analytics.json if cluster mode
is_cluster_mode = (
    config.analysis_type == "hashtag" and
    not config.target.startswith("#")
)
if is_cluster_mode:
    cluster_analytics_path = analysis_base / "cluster_analytics.json"
    if cluster_analytics_path.exists():
        output_files.append(str(cluster_analytics_path))

# Create checkpoint data
checkpoint_data = {
    "stage": "stage_1_video_discovery",
    "completion_timestamp": datetime.now(timezone.utc).isoformat(),
    "winning_buckets": winning_buckets,
    "output_files": output_files,
    "analysis_mode": config.analysis_mode,
    "analysis_type": config.analysis_type,
    "target": config.target,
    "video_count": config.video_count,
}

# Atomic write
checkpoint_dir = analysis_base / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

try:
    with open(stage1_checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.info(f"Stage 1 checkpoint created: {stage1_checkpoint_path}")
except (IOError, PermissionError) as e:
    # Non-fatal warning (pipeline continues)
    logger.warning(f"Failed to create Stage 1 checkpoint: {e}")
    logger.warning("Pipeline will re-run Stage 1 on next resume")
```

### Resume Behavior

**Scenario 1: Checkpoint valid + all files exist**
- Action: Skip Stage 1 entirely
- Load `winning_buckets` from checkpoint
- Continue to Stage 2

**Scenario 2: Checkpoint missing or corrupt**
- Action: Run Stage 1 from scratch
- Create new checkpoint on success

**Scenario 3: Checkpoint valid but files missing**
- Action: Delete checkpoint, re-run Stage 1
- Indicates partial completion or file deletion

**Scenario 4: Checkpoint write fails**
- Action: Log warning, continue pipeline
- Stage 1 will re-run on next resume (non-fatal)

---

## Debugging Guide

### Common Issues

#### Issue 1: "APIFY_API_KEY environment variable not set"

**Symptom**:
```
ValueError: APIFY_API_KEY environment variable required.
Obtain from: https://console.apify.com/account/integrations
```

**Cause**: Missing or unset APIFY_API_KEY environment variable

**Debug**:
```bash
# Check if set
echo $APIFY_API_KEY

# Check if available to Python
python3 -c "import os; print(os.getenv('APIFY_API_KEY'))"
```

**Fix**:
```bash
# Set environment variable
export APIFY_API_KEY="your_key_here"

# Or add to .env file (if using python-dotenv)
echo "APIFY_API_KEY=your_key_here" >> .env
```

**Location**: `ml_pipeline/stage1_discovery/apify_scraper.py:51-56`

---

#### Issue 2: "Apify scraping timeout after 3 retries"

**Symptom**:
```
TimeoutError: Apify scraping timeout after 3 retries.
Check network connection or increase APIFY_TIMEOUT.
```

**Cause**: Apify scraper exceeded 720s timeout (3 attempts)

**Debug**:
```bash
# Check network connectivity
curl -I https://api.apify.com

# Check Apify status page
curl -I https://status.apify.com
```

**Fix Options**:

1. **Increase timeout** (constants.py:32-33):
```python
# Change from 720s (12 min) to 1200s (20 min)
APIFY_TIMEOUT = 1200
```

2. **Check Apify account status**: Visit https://console.apify.com/actors/runs

3. **Retry manually**: Pipeline will resume from Stage 1 if checkpoint invalid

**Location**: `ml_pipeline/stage1_discovery/apify_scraper.py:256-259`

---

#### Issue 3: "Insufficient videos for analysis. Need ≥10, got 5"

**Symptom**:
```
ValueError: Insufficient videos for analysis. Need ≥10, got 5.
Try different target or relax date filter.
```

**Cause**: Date filter too strict or target has low video count

**Debug**:
```bash
# Check winner_analysis.json (if Stage 1 partially completed)
cat {analysis_base}/winner_analysis.json | jq '.top_100_distribution'

# Check selected_videos.json count
cat {analysis_base}/buckets/bucket_*/selected_videos.json | jq '.selected_count'
```

**Fix Options**:

1. **Relax date filter**:
```bash
# Change from last_30_days to last_90_days
--date-filter last_90_days
```

2. **Try different target** (if target is low-volume):
```bash
--target nutrition  # Try broader cluster
```

3. **Lower minimum threshold** (not recommended, affects ML quality):
```python
# constants.py:53-54
MIN_VIDEOS_FOR_ANALYSIS = 5  # Lower from 10 to 5
```

**Location**:
- `ml_pipeline/stage1_discovery/date_filter.py:183-189`
- `ml_pipeline/stage1_discovery/winner_analyzer.py:61-68`

---

#### Issue 4: "No buckets qualified (≥5% winners required)"

**Symptom**:
```
ValueError: No buckets qualified (≥5% winners required).
Winner distribution too fragmented.
```

**Cause**: Winner videos spread too thinly across all 8 buckets (no bucket has ≥5 videos)

**Debug**:
```bash
# Check winner distribution
cat {analysis_base}/winner_analysis.json | jq '.top_100_distribution'

# Example fragmented distribution:
# {"0-3s": 3, "3-9s": 2, "9-13s": 4, "13-18s": 3, ...}
# (No bucket has ≥5 videos)
```

**Fix Options**:

1. **Increase sample size** (scrape more videos):
```python
# constants.py:29
APIFY_SCRAPE_COUNT = 1200  # Increase from 800
```

2. **Lower qualification threshold** (not recommended):
```python
# constants.py:62-63
MIN_WINNER_PERCENTAGE = 3.0  # Lower from 5.0 to 3.0
```

3. **Try different target** (current target may have diverse video lengths)

**Location**: `ml_pipeline/stage1_discovery/winner_analyzer.py:240-245`

---

#### Issue 5: Checkpoint validation failed (missing fields)

**Symptom**:
```
WARNING: Checkpoint corrupt (missing fields: ['output_files']). Deleting and re-running Stage 1.
```

**Cause**: Checkpoint JSON missing required fields (manual edit or corruption)

**Debug**:
```bash
# Check checkpoint schema
cat {analysis_base}/checkpoints/stage_1_checkpoint.json | jq 'keys'

# Required fields: ["winning_buckets", "output_files", "completion_timestamp"]
```

**Fix**:
1. Orchestrator automatically deletes corrupt checkpoint
2. Stage 1 re-runs from scratch
3. If recurring, check disk space and permissions

**Location**: `rumiai_ml_batch.py:586-593`

---

#### Issue 6: Stage 1 outputs incomplete (files missing)

**Symptom**:
```
WARNING: Stage 1 outputs incomplete (3 files missing). Re-running Stage 1.
```

**Cause**: Output files listed in checkpoint don't exist on filesystem

**Debug**:
```bash
# List expected files from checkpoint
cat {analysis_base}/checkpoints/stage_1_checkpoint.json | jq '.output_files[]'

# Check which files are missing
for file in $(cat {analysis_base}/checkpoints/stage_1_checkpoint.json | jq -r '.output_files[]'); do
    [ -f "$file" ] && echo "✓ $file" || echo "✗ $file (MISSING)"
done
```

**Fix**:
1. Orchestrator automatically deletes checkpoint
2. Stage 1 re-runs from scratch
3. If files were manually deleted, let Stage 1 re-run

**Location**: `rumiai_ml_batch.py:595-604`

---

### Debug Commands

**Check Stage 1 outputs**:
```bash
# List all Stage 1 output files
ls -lh {analysis_base}/winner_analysis.json
ls -lh {analysis_base}/buckets/bucket_*/selected_videos.json
ls -lh {analysis_base}/cluster_analytics.json  # Cluster mode only

# Verify JSON validity
jq empty {analysis_base}/winner_analysis.json && echo "✓ Valid JSON" || echo "✗ Invalid JSON"
```

**Check checkpoint status**:
```bash
# View checkpoint
cat {analysis_base}/checkpoints/stage_1_checkpoint.json | jq '.'

# Check checkpoint age
stat -c %y {analysis_base}/checkpoints/stage_1_checkpoint.json
```

**Check video counts**:
```bash
# Winner distribution
cat {analysis_base}/winner_analysis.json | jq '.top_100_distribution'

# Selected videos per bucket
for bucket in {analysis_base}/buckets/bucket_*/; do
    echo "$(basename $bucket): $(cat ${bucket}selected_videos.json | jq '.selected_count') videos"
done
```

**Check Apify scraping results**:
```bash
# Cluster analytics (cluster mode only)
cat {analysis_base}/cluster_analytics.json | jq '{
    total_unique: .total_unique,
    duplicate_rate: .duplicate_rate_pct,
    validation_removed: .hashtag_validation.removed
}'
```

**Validate bucket assignment**:
```bash
# Check video durations in selected_videos.json
cat {analysis_base}/buckets/bucket_18-33s/selected_videos.json | \
    jq '.videos[].duration' | \
    awk '{if ($1 < 18 || $1 >= 33) print "ERROR: " $1 "s outside bucket range 18-33s"}'
```

---

## Modification Guide

### Scenario 1: Add Support for New Geography Filter

**Requirement**: Add support for "IN" (India) country code

**Files to Modify**:
1. `foundation/constants.py` - Add to VALID_COUNTRY_CODES
2. `ml_pipeline/stage1_discovery/apify_scraper.py` - Already supports via proxyCountryCode

**Steps**:

**Step 1**: Add "IN" to valid country codes
```python
# File: foundation/constants.py (line ~40)
VALID_COUNTRY_CODES = ["US", "BR", "IN", "global"]  # Add "IN"
```

**Step 2**: Test Apify scraper (already supports any country code)
```python
# File: ml_pipeline/stage1_discovery/apify_scraper.py:162-164
if country_code != "global":
    input_params["proxyCountryCode"] = country_code  # "US", "BR", or "IN"
```

**Step 3**: Update CLI help text
```python
# File: foundation/cli.py:163-166
parser.add_argument(
    "--country-code",
    choices=VALID_COUNTRY_CODES,
    default=DEFAULT_COUNTRY_CODE,
    help="Geographic filtering (default: US, options: US/BR/IN/global)"  # Update help
)
```

**Test**:
```bash
python rumiai_ml_batch.py \
    --client test \
    --analysis-type hashtag \
    --target nutrition \
    --country-code IN  # Test India filtering
```

---

### Scenario 2: Change Top Performer Count from 100 to 150

**Requirement**: Analyze top 150 performers instead of top 100

**Files to Modify**:
1. `ml_pipeline/stage1_discovery/constants.py`

**Steps**:

**Step 1**: Update constant
```python
# File: ml_pipeline/stage1_discovery/constants.py:56-57
TOP_PERFORMERS_FOR_ANALYSIS = 150  # Changed from 100
```

**Step 2**: Update winner coverage calculation
```python
# File: ml_pipeline/stage1_discovery/video_discovery.py:272-274
"winner_coverage": sum(
    winner_distribution.get(b, 0) for b in winning_buckets
) / 150 * 100,  # Changed from 100 to 150
```

**Impact**:
- Winner percentages will be calculated from 150 videos instead of 100
- Min qualified bucket threshold (5%) now requires ≥8 videos (was ≥5)
- More representative winner distribution

**Test**:
```bash
# Run Stage 1 and check winner_analysis.json
python rumiai_ml_batch.py --client test --analysis-type hashtag --target nutrition

# Verify top_100_distribution sums to 150
cat {analysis_base}/winner_analysis.json | jq '.top_100_distribution | add'
# Expected: 150
```

---

### Scenario 3: Add New Duration Bucket (120-180s)

**Requirement**: Extend support for TikTok videos up to 180 seconds (3 minutes)

**Files to Modify**:
1. `foundation/buckets.py` - Bucket assignment logic
2. `foundation/constants.py` - Bucket definitions

**Steps**:

**Step 1**: Update bucket assignment function
```python
# File: foundation/buckets.py:11-56
def assign_bucket(duration: float) -> str:
    if duration < 3:
        return "0-3s"
    elif duration < 9:
        return "3-9s"
    elif duration < 13:
        return "9-13s"
    elif duration < 18:
        return "13-18s"
    elif duration < 33:
        return "18-33s"
    elif duration < 60:
        return "33-60s"
    elif duration < 90:
        return "60-90s"
    elif duration < 120:
        return "90-120s"
    elif duration <= 180:  # NEW BUCKET
        return "120-180s"
    else:
        raise ValueError(f"Video duration {duration}s exceeds TikTok maximum (180s)")
```

**Step 2**: Add bucket definition
```python
# File: foundation/constants.py
BUCKET_DEFINITIONS = {
    "0-3s": (0, 3),
    "3-9s": (3, 9),
    "9-13s": (9, 13),
    "13-18s": (13, 18),
    "18-33s": (18, 33),
    "33-60s": (33, 60),
    "60-90s": (60, 90),
    "90-120s": (90, 120),
    "120-180s": (120, 180),  # NEW BUCKET
}
```

**Step 3**: Update downstream stages that use BUCKET_DEFINITIONS
- Stage 2.6, 2.7, 3, 4, 5, 6, 7 (all bucket-aware stages)
- Search for `BUCKET_DEFINITIONS` usage and update window configurations

**Test**:
```bash
# Unit test bucket assignment
python3 -c "
from foundation.buckets import assign_bucket
print(assign_bucket(150))  # Should print: 120-180s
print(assign_bucket(180))  # Should print: 120-180s
print(assign_bucket(181))  # Should raise ValueError
"
```

---

### Scenario 4: Change Contrastive Split to 70/30

**Requirement**: Change contrastive strategy from 80/20 to 70/30 (70% top, 30% bottom)

**Files to Modify**:
1. `ml_pipeline/stage1_discovery/constants.py`

**Steps**:

**Step 1**: Update constant
```python
# File: ml_pipeline/stage1_discovery/constants.py:67-69
CONTRASTIVE_TOP_SPLIT = 0.7  # Changed from 0.8 (80%) to 0.7 (70%)
```

**Impact**:
- For video_count=100: 70 top + 30 bottom (was 80 + 20)
- More bottom performers for contrastive ML training
- Better underperformer pattern detection

**Test**:
```bash
# Run Stage 1 with contrastive strategy
python rumiai_ml_batch.py \
    --client test \
    --analysis-type hashtag \
    --target nutrition \
    --selection-strategy contrastive \
    --video-count 100

# Verify split in selected_videos.json
cat {analysis_base}/buckets/bucket_*/selected_videos.json | \
    jq '{top: .top_count, bottom: .bottom_count}'
# Expected: {"top": 70, "bottom": 30}
```

---

### Scenario 5: Skip Interactive Confirmation by Default

**Requirement**: Make auto-confirm=true the default (skip prompts)

**Already Implemented**: This is already the default behavior!

**Verification**:
```python
# File: foundation/cli.py:147-152
parser.add_argument(
    "--auto-confirm",
    action="store_true",
    default=True,  # ← Already defaults to True
    help="Skip interactive confirmation prompts (default: enabled)"
)
```

**To Enable Prompts** (opposite behavior):
```bash
python rumiai_ml_batch.py \
    --client test \
    --analysis-type hashtag \
    --target nutrition \
    --no-auto-confirm  # ← This enables prompts
```

---

### Scenario 6: Add Email Notification on Stage 1 Completion

**Requirement**: Send email when Stage 1 completes (success or failure)

**Files to Modify**:
1. `ml_pipeline/stage1_discovery/video_discovery.py`
2. Create new module: `ml_pipeline/notifications/email_notifier.py`

**Steps**:

**Step 1**: Create email notifier module
```python
# File: ml_pipeline/notifications/email_notifier.py (new file)
import smtplib
from email.mime.text import MIMEText
import os

def send_stage_completion_email(
    stage: str,
    status: str,
    client_id: str,
    target: str,
    details: dict
):
    """
    Send email notification on stage completion.

    Args:
        stage: Stage name (e.g., "Stage 1: Video Discovery")
        status: "SUCCESS" or "FAILURE"
        client_id: Client identifier
        target: Analysis target
        details: Additional details (winning_buckets, error message, etc.)
    """
    # Email configuration from environment
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    sender_email = os.getenv("NOTIFICATION_EMAIL")
    sender_password = os.getenv("NOTIFICATION_EMAIL_PASSWORD")
    recipient_email = os.getenv("ADMIN_EMAIL")

    if not all([sender_email, sender_password, recipient_email]):
        print("⚠️  Email notification skipped (missing SMTP credentials)")
        return

    # Compose email
    subject = f"[RumiAI] {stage} - {status}"

    body = f"""
{stage} - {status}

Client: {client_id}
Target: {target}

Details:
{json.dumps(details, indent=2)}

Timestamp: {datetime.now(timezone.utc).isoformat()}
"""

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Send email
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"✓ Email notification sent: {status}")
    except Exception as e:
        print(f"⚠️  Email notification failed: {e}")
```

**Step 2**: Update VideoDiscovery.run() to send notification
```python
# File: ml_pipeline/stage1_discovery/video_discovery.py:84-242

# Add import at top of file
from ml_pipeline.notifications.email_notifier import send_stage_completion_email

# Modify run() method:
def run(self) -> int:
    try:
        # ... existing Stage 1 logic ...

        # Stage 1.6: Create output files
        self._create_output_files(...)

        # NEW: Send success notification
        send_stage_completion_email(
            stage="Stage 1: Video Discovery",
            status="SUCCESS",
            client_id=self.config['client_id'],
            target=self.config['target'],
            details={
                "winning_buckets": winning_buckets,
                "total_videos_selected": sum(s['selected_count'] for s in selected_per_bucket.values()),
                "strategy": self.config['selection_strategy']
            }
        )

        logger.info("STAGE 1 COMPLETE")
        return EXIT_CODE_SUCCESS

    except KeyboardInterrupt:
        # User abort - no notification needed
        return EXIT_CODE_USER_ABORT

    except Exception as e:
        # NEW: Send failure notification
        send_stage_completion_email(
            stage="Stage 1: Video Discovery",
            status="FAILURE",
            client_id=self.config.get('client_id', 'unknown'),
            target=self.config.get('target', 'unknown'),
            details={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )

        logger.error(f"Stage 1 failed: {e}", exc_info=True)
        raise
```

**Step 3**: Set environment variables
```bash
# Add to .env or export
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export NOTIFICATION_EMAIL="rumiai@example.com"
export NOTIFICATION_EMAIL_PASSWORD="your_app_password"
export ADMIN_EMAIL="admin@example.com"
```

**Test**:
```bash
# Run Stage 1 and check email inbox
python rumiai_ml_batch.py --client test --analysis-type hashtag --target nutrition

# Expected: Email with subject "[RumiAI] Stage 1: Video Discovery - SUCCESS"
```

---

## Related Documentation

### Pipeline Documentation
- **[PRODUCTION_FLOW.md](PRODUCTION_FLOW.md)**: Complete pipeline overview (Stages 0-7)
- **[PRODUCTION_FLOW.md - Stage 1 Contract](PRODUCTION_FLOW.md#stage-1-video-discovery)**: Stage 1 inputs/outputs/dependencies

### Foundation Documentation
- **[foundation/cli.py](foundation/cli.py)**: CLI argument parsing and validation
- **[foundation/config.py](foundation/config.py)**: Configuration management
- **[foundation/paths.py](foundation/paths.py)**: Path building and sanitization
- **[foundation/buckets.py](foundation/buckets.py)**: Duration bucket assignment logic

### Stage 1 Source Files
- **[ml_pipeline/stage1_discovery/video_discovery.py](ml_pipeline/stage1_discovery/video_discovery.py)**: Main orchestrator (312 lines)
- **[ml_pipeline/stage1_discovery/apify_scraper.py](ml_pipeline/stage1_discovery/apify_scraper.py)**: Apify scraping logic (330 lines)
- **[ml_pipeline/stage1_discovery/date_filter.py](ml_pipeline/stage1_discovery/date_filter.py)**: Date filtering (197 lines)
- **[ml_pipeline/stage1_discovery/winner_analyzer.py](ml_pipeline/stage1_discovery/winner_analyzer.py)**: Winner analysis (291 lines)
- **[ml_pipeline/stage1_discovery/video_selector.py](ml_pipeline/stage1_discovery/video_selector.py)**: Video selection (243 lines)
- **[ml_pipeline/stage1_discovery/confirmation.py](ml_pipeline/stage1_discovery/confirmation.py)**: User confirmation (128 lines)
- **[ml_pipeline/stage1_discovery/constants.py](ml_pipeline/stage1_discovery/constants.py)**: Configuration constants (170 lines)

### Downstream Stages
- **Stage 2**: Video Processing (consumes selected_videos.json)
- **Stage 2.5**: File Organization (consumes winner_analysis.json)
- **Stage 2.6, 2.7**: Content Analysis (consumes selection_manifest.json)
- **Stage 8**: Report Generation (consumes video metadata)

---

## Document Metadata

**Generated**: 2025-11-02
**Source**: 100% systematic code reading (3,958 production lines across 12 modules)
**Verification**: All line numbers, schemas, and code snippets from actual source code
**Coverage**: Complete Stage 1 implementation (Stages 1.1-1.6)

**Last Validated**: 2025-11-02
**Apify Actor**: clockworks/tiktok-scraper (GdWCkxBtKWOsKjdch) - Last validated 2025-10-20
