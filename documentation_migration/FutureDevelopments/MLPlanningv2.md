# ML Production Pipeline Planning v2

> **Version**: 2.0 (Stage-Based Architecture)
> **Parent Document**: MLPlanning.md (legacy)
> **Structure**: Organized by processing stages for easy TI doc creation

---

## Document Overview

### Purpose

This document provides comprehensive planning for RumiAI's ML production pipeline, organized by processing stages to facilitate:
- End-to-end pipeline understanding
- Technical Implementation (TI) document creation
- Future feature impact analysis
- Cross-document consistency verification

### Quick Navigation

- **Part 1: Foundation** - System overview, client architecture, data retention
- **Part 2: Configuration (Stage 0)** - CLI parameters and configuration dimensions
- **Part 3: Processing Pipeline (Stages 1-7)** - Linear video processing to report generation
- **Part 4: Future Enhancements** - Planned features organized by stage impact

### Stage-Based Organization

Each processing stage includes:
- **Purpose**: One-line goal
- **Input**: What data enters this stage
- **Process**: Step-by-step transformation logic
- **Output**: What data exits this stage
- **Child Documents**: Related HLD docs that expand on this stage
- **Future TI Document**: Technical implementation doc to be created
- **Related Future Features**: Enhancements that impact this stage

---

# Part 1: Foundation

## System Goals & Success Criteria

### Primary Goals

1. **Batch Video Analysis**
   - Process up to 300 videos sequentially through `rumiai_runner.py`
   - Implement checkpoint/resume system for failure recovery

2. **Client-Centric Data Organization**
   - Multi-tenant data structure: Client → Hashtags → Duration Buckets → Videos
   - Bucket-specific analysis within client/hashtag boundaries
   - Persistent client/hashtag/duration configuration management

3. **Duration-Specific ML Pattern Recognition**
   - Train separate ML models for each duration bucket
   - Recognize that 15-second patterns differ completely from 60-second patterns
   - Generate bucket-specific insights (no universal patterns across durations)

4. **Creative Report Generation**
   - Number of Reports to be created is being determined.

### Success Criteria

**Processing Capability**:
- Successfully analyze up to 300 videos per hashtag sequentially
- Support multiple hashtags per client (e.g., 4-10+ hashtags)
- Checkpoint/resume system enables recovery without data loss
- Complete end-to-end processing or clear failure identification

**ML Insight Generation**:
- Generate meaningful trends and patterns from analyzed videos
- Include confidence scores and pattern validation

**Report Delivery**:
- Produce concise, actionable creative strategy reports
- Focus on "easy to replicate" format with clear steps
- Identical reports for both UGC Factories and individual creators

**Client Executive Reporting**:
- Bird's eye view reports covering minimum 5 hashtags per client
- Demonstrate scope and value through breadth of research

### Key Metrics

- **Input Scale**: User-configurable via --video-count N per qualified bucket
  - Contrastive default: N=100 per bucket (80 top + 20 bottom), top 3 buckets = ~300 videos
  - Top default: N=40 per bucket, top 3 buckets = ~120 videos
  - Only top 3 most active buckets are processed (adaptive bucket processing)
- **ML Models**: Random Forest and K-means with a capacity of 16 models total (2 algorithms × 8 duration buckets)
- **Output**: Duration-specific creative recommendations (Total Creative Reports being defined)
- **Processing**: Sequential (one-by-one) with resumption capability

---

## Client Architecture & Storage

### Directory Structure

```
/data/
├── clients/
│   ├── {client_id}/
│   │   ├── hashtags/                       # Hashtag-based analyses
│   │   │   └── {hashtag_name}/             # e.g., "nutrition", "fitness"
│   │   │       ├── top_contrastive/        # Analysis: top mode + contrastive strategy
│   │   │       │   ├── config.json         # {mode: "top", strategy: "contrastive", date_filter: "last_90_days", run_date: "2025-01-28", video_count: 300}
│   │   │       │   ├── buckets/            # Duration-based ML buckets (8 total)
│   │   │       │   │   ├── bucket_0-3s/
│   │   │       │   │   │   ├── videos/     # Raw MP4s: N files (configurable via --video-count)
│   │   │       │   │   │   │                # Contrastive: 80% top + 20% bottom of N (e.g., 80+20 if N=100)
│   │   │       │   │   │   ├── analysis/   # RumiAI outputs for N videos
│   │   │       │   │   │   │   ├── insights/   # temporal_windows JSON (1 per video)
│   │   │       │   │   │   │   ├── unified/    # Intermediate timeline+ml_data (debugging)
│   │   │       │   │   │   │   └── service_debug/  # emotion_detection, audio_energy outputs
│   │   │       │   │   │   ├── ml_analysis/    # ML pipeline outputs
│   │   │       │   │   │   │   ├── aggregated_features.csv          # Aggregated temporal windows (N videos)
│   │   │       │   │   │   │   ├── rf_transformed.csv               # RF-ready features
│   │   │       │   │   │   │   ├── km_transformed.csv               # KMeans-ready features
│   │   │       │   │   │   │   ├── random_forest_analysis.json      # ~30KB - Input to LLM Call 1
│   │   │       │   │   │   │   └── kmeans_analysis.json             # ~30KB - Input to LLM Call 1
│   │   │       │   │   │   ├── models/     # Trained models for THIS bucket
│   │   │       │   │   │   │   ├── random_forest_v1.pkl  # Classification model
│   │   │       │   │   │   │   ├── kmeans_v1.pkl         # Clustering model
│   │   │       │   │   │   │   ├── scalers.pkl            # MinMaxScalers for KMeans
│   │   │       │   │   │   │   └── model_metrics.json
│   │   │       │   │   │   ├── llm_reports/  # LLM outputs
│   │   │       │   │   │   │   ├── analysis/              # LLM Call 1 outputs (insight extraction)
│   │   │       │   │   │   │   │   ├── call_1_rf_prompt.txt
│   │   │       │   │   │   │   │   ├── call_1_rf_raw_response.json
│   │   │       │   │   │   │   │   ├── call_1_kmeans_prompt.txt
│   │   │       │   │   │   │   │   ├── call_1_kmeans_raw_response.json
│   │   │       │   │   │   │   │   └── insights.json              # Structured insights (parsed) → Input to LLM Call 2
│   │   │       │   │   │   │   └── formatted/            # LLM Call 2 outputs (report generation)
│   │   │       │   │   │   │       ├── call_2_prompt.txt
│   │   │       │   │   │   │       ├── call_2_raw_response.json
│   │   │       │   │   │   │       ├── rf_feature_importance.md
│   │   │       │   │   │   │       ├── strategy_1_the_educator.md
│   │   │       │   │   │   │       ├── strategy_2_visual_storyteller.md
│   │   │       │   │   │   │       ├── strategy_3_personal_journey.md
│   │   │       │   │   │   │       └── bucket_summary.md
│   │   │       │   │   │   ├── reports/    # Final PDFs (no LLM - converted from markdown)
│   │   │       │   │   │   ├── checkpoints/ # Processing state for this bucket
│   │   │       │   │   │   └── logs/       # Bucket processing logs
│   │   │       │   │   ├── bucket_3-9s/
│   │   │       │   │   ├── bucket_9-13s/
│   │   │       │   │   ├── bucket_13-18s/
│   │   │       │   │   ├── bucket_18-33s/
│   │   │       │   │   ├── bucket_33-60s/
│   │   │       │   │   ├── bucket_60-90s/
│   │   │       │   │   └── bucket_90-120s/
│   │   │       │   └── hashtag_summary/   # Cross-bucket executive reports
│   │   │       │       ├── executive_report.pdf
│   │   │       │       └── hashtag_metrics.json
│   │   │       │
│   │   │       ├── top_top/               # Analysis: top mode + top strategy (OPTIONAL)
│   │   │       │   ├── config.json        # {mode: "top", strategy: "top", date_filter: "last_90_days", run_date: "2025-02-01", video_count: 300}
│   │   │       │   ├── buckets/           # Same structure
│   │   │       │   │   └── bucket_0-3s/
│   │   │       │   │       ├── videos/    # N files (top only, N from --video-count, default 40)
│   │   │       │   │       ├── analysis/  # N JSONs
│   │   │       │   │       └── reports/   # Best practices reports (no classification models)
│   │   │       │   └── hashtag_summary/
│   │   │
│   │   ├── competitors/                   # Competitor tracking (what rivals are doing)
│   │   │   └── {competitor_handle}/       # e.g., "@rival_brand"
│   │   │       ├── top_top/               # Analysis: top mode + top strategy
│   │   │       │   ├── config.json        # {mode: "top", strategy: "top", date_filter: "last_90_days", run_date: "2025-01-28", video_count: 150}
│   │   │       │   ├── buckets/           # Same 8-bucket structure
│   │   │       │   │   └── bucket_0-3s/
│   │   │       │   │       ├── videos/    # N files (top only, N from --video-count, default 40)
│   │   │       │   │       ├── analysis/
│   │   │       │   │       └── reports/   # Best practices reports
│   │   │       │   └── competitor_summary/
│   │   │       │       ├── competitor_report.pdf
│   │   │       │       └── competitor_metrics.json
│   │   │       │
│   │   │       ├── recent_top/            # Analysis: recent mode + top strategy (OPTIONAL)
│   │   │       │   ├── config.json        # Track current strategy shifts
│   │   │       │   └── buckets/
│   │   │       │
│   │   │       └── top_contrastive/       # Analysis: top mode + contrastive strategy (OPTIONAL)
│   │   │           ├── config.json        # Full contrastive competitor analysis
│   │   │           └── buckets/
│   │   │
│   │   └── creators/                      # Creator vetting (potential affiliates)
│   │       └── {creator_handle}/          # e.g., "@potential_affiliate"
│   │           ├── recent_top/            # Analysis: recent mode + top strategy
│   │           │   ├── config.json        # {mode: "recent", strategy: "top", date_filter: "last_30_days", run_date: "2025-01-28", video_count: 40}
│   │           │   ├── buckets/           # Same 8-bucket structure
│   │           │   │   └── bucket_0-3s/
│   │           │   │       ├── videos/    # N files (top recent videos, N from --video-count, default 40)
│   │           │   │       └── analysis/
│   │           │   └── creator_summary/
│   │           │       ├── style_profile.pdf
│   │           │       ├── compatibility_report.pdf
│   │           │       ├── compatibility_scores.json
│   │           │       └── creator_metrics.json
│   │           │
│   │           └── top_top/               # Analysis: top mode + top strategy (OPTIONAL)
│   │               ├── config.json        # Peak performance analysis
│   │               └── buckets/
```

### Architecture Notes

- **Analysis Directories**: Each `{mode}_{strategy}/` directory is a complete, independent analysis run
- **config.json**: Stores run parameters (mode, strategy, date_filter, run_date, video_count) for reproducibility
- **Coexistence**: Multiple analyses can exist simultaneously without overwriting
- **Default Paths**: Each flow type writes to its default analysis directory
  - Hashtag → `top_contrastive/`
  - Competitor → `top_top/`
  - Creator → `recent_top/`
- **Video Counts**: User-configurable via --video-count N
  - Contrastive: N per qualified bucket (80/20 split, default N=100)
  - Top: N per qualified bucket (all top, default N=40)
  - Only top 3 most active buckets are processed (adaptive bucket processing)

### Data Retention Policy

| Asset | Retention | Rationale |
|-------|-----------|-----------|
| **Raw Videos** | 30 days | Can re-download if needed, saves space |
| **ML Analysis** | 6 months | Compressed after 30 days |
| **ML Models** | Latest 3 versions | Per client/hashtag |
| **Reports** | Indefinite | Small size, high value |
| **Checkpoints** | 7 days after completion | Enable resume capability |

### Storage Cost Optimization

- **Video Deletion**: Remove raw videos after 30 days
- **ML Analysis Compression**: Compress after 60 days
- **Model Versioning**: Keep only latest 3 versions per client/hashtag

---

# Part 2: Configuration (Pre-Pipeline)

All RumiAI analyses use **multiple orthogonal configuration dimensions**. These flags work independently and can be combined in any valid way.

## Command Structure

```bash
python rumiai_ml_batch.py \
  --client "client_name" \
  --analysis-type {hashtag|competitor|creator} \    # Stage 0.1: Target Type
  --target "{target}" \
  --analysis-mode {top|recent} \                     # Stage 0.2: Analysis Mode
  --selection-strategy {contrastive|top} \           # Stage 0.3: Selection Strategy
  --video-count N \                                  # Stage 0.4: Video Count
  --date-filter last_N_days \                        # Stage 0.5: Date Filter
  --report-type {single|comparison}                  # Stage 0.6: Report Type
```

**Design Principles**:
- **Orthogonal**: Each dimension is independent
- **Composable**: Valid combinations work across all target types
- **Default-aware**: Each target type has sensible defaults
- **Explicit**: All parameters exposed as CLI flags (automation-friendly)

---

## Stage 0.1: Target Types

**Purpose**: Determines what source to analyze videos from

**Available Types**:

| Type | CLI Flag | Target Format | Data Source | Primary Use Case |
|------|----------|---------------|-------------|------------------|
| `hashtag` | `--analysis-type hashtag` | `#nutrition` | TikTok hashtag search | Market research - identify viral patterns |
| `competitor` | `--analysis-type competitor` | `@rival_brand` | TikTok profile | Competitive intelligence - understand rivals |
| `creator` | `--analysis-type creator` | `@potential_affiliate` | TikTok profile | Creator vetting - assess fit for hiring |

**CLI Usage**:
```bash
--analysis-type hashtag     # Analyze hashtag content
--analysis-type competitor  # Analyze competitor profile
--analysis-type creator     # Analyze creator profile
```

**Key Differences**:

| Aspect | Hashtag | Competitor | Creator |
|--------|---------|------------|---------|
| **Apify Scraper** | clockworks/tiktok-hashtag-scraper | clockworks/tiktok-scraper | clockworks/tiktok-scraper |
| **Video Source** | All TikTok users posting with hashtag | Single profile's content | Single profile's content |
| **ML Training** | Yes (classification models) | Optional (descriptive only) | No (uses existing models) |
| **Default Mode** | `top` | `top` | `recent` |
| **Default Strategy** | `contrastive` | `top` | `top` |
| **Default Date Filter** | `last_90_days` | `last_90_days` | `last_30_days` |
| **Default Video Count** | 100 | 40 | 40 |

**Why Target Type Matters**:
- Different scrapers (hashtag scraper vs profile scraper)
- Different business questions (market trends vs competitor benchmarking vs hiring decisions)
- Different default configurations (optimized per use case)
- Same underlying processing pipeline (RumiAI → Buckets → ML → Reports)

**Child Documents**: None (native to MLPlanning)

**Future TI Document**: None (CLI configuration only)

---

## Stage 0.2: Analysis Modes

**Purpose**: Controls how Apify sorts and selects videos

**Available Modes**:

| Mode | Sort By | Use Case | Default For |
|------|---------|----------|-------------|
| `top` | Engagement (composite score) | "What works?" - Identify successful patterns | Hashtag, Competitor |
| `recent` | Publish date (newest first) | "What's happening now?" - Track current trends | Creator |

**CLI Usage**:
```bash
--analysis-mode top     # Analyze highest-performing content
--analysis-mode recent  # Analyze most recent content
```

**Engagement Score Formula** (Top Mode):
```
engagement_score = views × (1 + share_rate × 10)

where:
  share_rate = shares / views
  share_boost = 1 + (share_rate × 10)
```

**Why This Formula**:
- Shares signal viral potential (10x weight reflects their importance)
- Views alone can be misleading (high views, low engagement)
- Captures "share-worthy" content beyond just popularity

**Child Documents**:
- SelectionStrategies.md (Sorting Strategies section)

**Future TI Document**: VideoDiscoveryTI.md (Apify integration, sorting implementation)

---

## Stage 0.3: Selection Strategies

**Purpose**: Determines what videos to select after sorting (orthogonal to analysis mode)

**Available Strategies**:

| Strategy | Videos Selected | Use Case | Default For |
|----------|----------------|----------|-------------|
| `contrastive` | Top 80% + Bottom 20% per bucket (N configurable, default 100) | ML training - identify pattern differences through contrast | Hashtag |
| `top` | Top N per bucket only (N configurable, default 40) | Best practices analysis - learn from success only | Competitor, Creator |

**Why Separate from Analysis Mode?**
These are orthogonal dimensions:
- **Analysis Mode** (top/recent): Controls HOW videos are sorted
- **Selection Strategy** (contrastive/top): Controls WHAT subset is analyzed

**Example**:
- `top mode + contrastive strategy` = Top performers analyzed with contrastive learning
- `recent mode + top strategy` = Most recent high-quality content only

**Child Documents**: SelectionStrategies.md (comprehensive strategy documentation)

**Future TI Document**: VideoDiscoveryTI.md (selection logic implementation)

---

## Stage 0.4: Video Count

**Purpose**: Controls how many videos to analyze per winning bucket

**CLI Parameter**: `--video-count N`

**Strategy-Specific Behavior**:

| Strategy | Default N | Interpretation | Example (N=100) |
|----------|-----------|----------------|-----------------|
| **Contrastive** | 100 | 80% top + 20% bottom | 80 top + 20 bottom = 100 total |
| **Top** | 40 | All top performers | 40 top performers |

**Key Concepts**:
- **Success-based bucket selection**: Analyzes top 100 performers to identify where winners cluster
- **Adaptive processing**: Only top 3 buckets where winners concentrate are processed (not volume-based)
- **Flexible threshold**: If winning bucket has < N total videos, process anyway with warning

**Examples**:
```bash
# Contrastive with 150 videos per bucket (120 top + 30 bottom)
--selection-strategy contrastive --video-count 150

# Top with 60 videos per bucket
--selection-strategy top --video-count 60
```

**Child Documents**: SelectionStrategies.md (adaptive processing logic)

**Future TI Document**: VideoDiscoveryTI.md (bucket selection, threshold handling)

---

## Stage 0.5: Date Filtering

**Purpose**: Controls when videos were published - filters scraped videos by publication date

**CLI Parameter**: `--date-filter last_N_days`

**Format**: Relative date range only
- `last_N_days` where N is the number of days to look back from today
- Examples: `last_30_days`, `last_90_days`, `last_180_days`

**Default**: `last_90_days`

**CLI Usage**:
```bash
--date-filter last_90_days   # Last 90 days (default)
--date-filter last_30_days   # Last 30 days
--date-filter last_180_days  # Last 180 days
```

**How It Works**:
1. Apify scrapes 800 videos from target (no server-side date filtering available)
2. **Client-side filtering**: System filters videos by `create_time` based on date filter
3. Filtered videos proceed to bucketing and selection

**Why Client-Side?**
- Apify's hashtag scraper doesn't support server-side date filtering
- Profile scraper has date support, but client-side used for consistency across all target types
- Ensures uniform behavior regardless of scraper type

**Business Value**:
- **Recency control**: Focus on recent trends vs historical patterns
- **Seasonal analysis**: Analyze specific time periods (e.g., holiday season)
- **Trend detection**: Track how patterns evolve over time
- **Data quality**: Exclude outdated content that may skew insights

**Interaction with Analysis Modes**:
- **Top Mode + Date Filter**: "What worked recently?" (best practices from last N days)
- **Recent Mode + Date Filter**: "What's happening now?" (most recent content within last N days)
- Both dimensions are orthogonal - date filters WHEN, mode filters HOW

**Default Per Target Type**:
- **Hashtag**: `last_90_days` (quarterly trends for market research)
- **Competitor**: `last_90_days` (current competitive strategies)
- **Creator**: `last_30_days` (recent natural style for vetting)

**Child Documents**: 

**Future TI Document**: VideoDiscoveryTI.md (date filtering logic)

---

## Stage 0.6: Report Types

**Purpose**: Determines what type of output is generated

**Available Types**:

| Type | What It Does | Prerequisites | Output | Applies To |
|------|-------------|---------------|--------|------------|
| `single` | Deep analysis of one target | None | Full ML analysis + creative reports | All target types (default) |
| `comparison` | Side-by-side comparison of 2+ targets | All targets must have existing single analyses | LLM-synthesized comparison report | All target types |

**CLI Usage**:
```bash
--report-type single       # Deep dive on one target (default)
--report-type comparison   # Compare multiple targets
```

**Single Mode Process**:
1. Scrape videos from target (Apify)
2. Run full ML pipeline (RumiAI → Buckets → ML Training)
3. Generate creative reports per bucket
4. Output: Models + Reports + Analysis data

**Comparison Mode Process**:
1. **No video processing** - uses existing analyses
2. Load data from completed single analyses
3. Send to Claude API with comparison prompt
4. Generate comparison report
5. Output: Single comparison PDF

**Key Differences**:

| Aspect | Single | Comparison |
|--------|--------|------------|
| **Video Processing** | Full ML pipeline (hours) | No processing (seconds) |
| **Prerequisites** | None | Requires existing single analyses |
| **LLM Calls** | 24 per target (analysis + formatting) | 1 per comparison group |
| **Output** | ML models + 40 creative reports | 1 comparison PDF |
| **Duration** | 6-8 hours (hashtag, 300 videos) | ~30 seconds (LLM only) |
| **Cost** | Apify ($4) + Compute + LLM ($3.60) | LLM only (~$0.50) |

**Child Documents**: None (native to MLPlanning)

**Future TI Document**: ReportGenerationTI.md (comparison report logic)

---

# Part 3: Processing Pipeline (Linear Stages)

End-to-end pipeline from video discovery to final reports.

## Pipeline Overview

```
Stage 1: Video Discovery & Selection
    ↓ Selected video list (per bucket)
Stage 2: Video Processing (RumiAI Pipeline)
    ↓ temporal_windows_updated.json (N videos per qualified bucket)
Stage 3: Feature Aggregation
    ↓ aggregated_features.csv (N rows × ~35 columns)
Stage 4: Feature Transformation
    ↓ rf_transformed.csv + km_transformed.csv
Stage 5: ML Model Training
    ↓ random_forest_analysis.json + kmeans_analysis.json (~30KB each)
Stage 6: ML Analysis Generation
    ↓ insights.json (structured ML insights)
Stage 7: LLM Report Generation
    ↓ 5 PDF creative strategy reports per bucket
```

---

## Stage 1: Video Discovery & Selection

**Purpose**: Identify and select winning videos for analysis

**Input**:
- CLI parameters (target, mode, strategy, date_filter, video_count)
- Stage 0 configuration values

**Process**:

### 1.1: Apify Scraping
```json
{
  "hashtagsUrls": ["#nutrition"],
  "resultsPerPage": 800,
  "shouldDownloadVideos": true,
  "sortBy": "engagement",
  "sortOrder": "desc"
}
```
- Scrapes 800 videos from target
- Engagement sorted DESC (top performers first)
- No server-side date filtering available

### 1.2: Date Filtering (Client-Side)
```python
# Filter videos by create_time
filtered_videos = [
    v for v in scraped_videos
    if v.create_time >= (now - timedelta(days=N))
]
```
- Apply `--date-filter last_N_days` parameter
- Filters by `create_time` field
- Typical result: 800 → 600 videos (last 90 days)

### 1.3: Winner Analysis (Success-Based Distribution)
```python
# Analyze top 100 performers
top_100 = filtered_videos[:100]

# Bucket by duration
winner_distribution = count_by_bucket(top_100)
# Result: {
#   "18-33s": 45,  # 45% of winners
#   "33-60s": 30,  # 30% of winners
#   "13-18s": 20,  # 20% of winners
#   "9-13s": 5     # 5% of winners
# }

# Select top 3 buckets by winner concentration
top_3_buckets = rank_by_concentration(winner_distribution)[:3]
# Result: ["18-33s", "33-60s", "13-18s"]
```
- **NOT volume-based**: Ignores where most creators post
- **Success-based**: Focuses on where winners cluster
- Example: Skip 9-13s (400 videos, 5 winners), Process 18-60s (150 videos, 75 winners)

### 1.4: Video Selection Per Bucket (Strategy-Specific)

**Contrastive Strategy** (N=100):
```python
bucket_videos = filter_by_duration(filtered_videos, bucket)

if len(bucket_videos) >= N:
    # Normal processing
    top_80_percent = bucket_videos[:int(N * 0.8)]  # 80 videos
    bottom_20_percent = bucket_videos[int(N * 0.8):N]  # 20 videos
    selected = top_80_percent + bottom_20_percent
elif bucket in top_3_buckets and bucket.winner_concentration > threshold:
    # Flexible threshold for winning buckets
    selected = bucket_videos  # Use all available
    warn(f"Only {len(bucket_videos)} videos (< N={N})")
else:
    # Skip bucket
    skip(bucket)
```

**Top Strategy** (N=40):
```python
bucket_videos = filter_by_duration(filtered_videos, bucket)

if len(bucket_videos) >= N:
    selected = bucket_videos[:N]  # Top N only
elif bucket in top_3_buckets:
    selected = bucket_videos  # Use all available
    warn(f"Only {len(bucket_videos)} videos (< N={N})")
else:
    skip(bucket)
```

**Output**:
- Selected video list (per bucket)
- Typical: ~300 videos total (3 buckets × ~100 videos each)
- Format: List of video URLs/IDs for Stage 2 processing

**Example Workflow**:
```
Scraped: 800 videos (all-time)
↓ Apply date_filter: last_90_days
Filtered: 600 videos (within date range)
↓ Analyze top 100 performers (success-based distribution)
Top 100 winners: 18-33s (45%), 33-60s (30%), 13-18s (20%), 9-13s (5%)
↓ Select top 3 winning buckets
Process: 18-33s, 33-60s, 13-18s (95% of winners)
↓ Apply selection strategy (contrastive, N=100)
Per bucket: 100 videos (80 top + 20 bottom)
```

**Child Documents**:
- SelectionStrategies.md (comprehensive strategy design, adaptive processing logic)

**Future TI Document**:

**Related Future Features**:
- Phase 2: Top Video Selection Formula Validation (Stage 1 improvement)

---

## Stage 2: Video Processing (RumiAI Pipeline)

**Purpose**: Process selected videos through RumiAI analysis pipeline

**Input**: Selected video URLs/IDs (from Stage 1)

**Process**:

### 2.1: Video Download
```python
# Download videos via Apify
for video_url in selected_videos:
    download_video(video_url, target_dir="bucket_{duration}/videos/")
```
- Downloads raw MP4 files
- Stores in bucket-specific `videos/` directory

### 2.2: Sequential RumiAI Processing
```python
for video_file in bucket_videos:
    # Run rumiai_runner.py
    result = run_rumiai_pipeline(video_file)

    # Save temporal windows output
    save(f"analysis/insights/{video_id}_temporal_windows_updated.json", result)

    # Checkpoint after each video (failure recovery)
    checkpoint.mark_complete(video_id)
```
- Processes one video at a time (sequential)
- Generates `temporal_windows_updated.json` per video
- Automatic checkpoint after each successful video

### 2.3: Checkpoint System
```json
{
  "bucket": "18-33s",
  "total_videos": 100,
  "completed": 45,
  "failed": 2,
  "remaining": 53,
  "last_checkpoint": "2025-01-28T14:32:15Z",
  "completed_video_ids": ["123", "456", ...]
}
```
- Saves processing state after each video
- Enables resume on interruption (SSH disconnect, crash, manual stop)
- No data loss, no reprocessing

**Output**:
- `temporal_windows_updated.json` (N files per bucket)
- Each JSON contains features per temporal window (hook, middle segments, closing)
- Checkpoints in `checkpoints/` directory
- Processing logs in `logs/` directory

**Child Documents**:
- MLCheckpointResume.md (checkpoint/resume system design)

**Future TI Document**:
- VideoProcessingTI.md (rumiai_runner integration, checkpoint logic, error handling)

**Related Future Features**:
- Phase 1: Checkpoint Resume System (Stage 2 core feature)

---

## Stage 3: Feature Aggregation

**Purpose**: Aggregate temporal windows to video-level features

**Input**: N × `temporal_windows_updated.json` files per qualified bucket (N from --video-count)

**Process**:

### 3.1: Temporal Window Aggregation
```python
for temporal_windows_json in bucket_jsons:
    video_features = {}

    # Hook features: Use directly (always 1 hook window)
    video_features['hook_scene_count'] = windows['hook']['scene_count']
    video_features['hook_eye_contact_rate'] = windows['hook']['eye_contact_rate']

    # Middle features: Average across all middle segments
    middle_windows = windows['middle']  # Variable count (2-7 segments)
    video_features['middle_avg_word_count'] = mean([w['word_count'] for w in middle_windows])
    video_features['middle_avg_scene_count'] = mean([w['scene_count'] for w in middle_windows])

    # Closing features: Use directly (always 1 closing window)
    video_features['closing_energy_level'] = windows['closing']['energy_level']

    # Global features: Sum or derive from all windows
    video_features['duration'] = windows['metadata']['duration']
    video_features['total_scene_count'] = sum_all_scene_counts(windows)

    aggregated_rows.append(video_features)
```

### 3.2: Create Aggregated CSV
```python
import pandas as pd

df = pd.DataFrame(aggregated_rows)
df.to_csv("ml_analysis/aggregated_features.csv", index=False)
```

**Output**: `ml_analysis/aggregated_features.csv`
- Shape: (N videos, ~35 aggregated features)
- Example columns:
  - `hook_scene_count`, `hook_eye_contact_rate`, `hook_word_count`
  - `middle_avg_word_count`, `middle_avg_scene_count`, `middle_avg_energy_level`
  - `closing_energy_level`, `closing_word_count`
  - `duration`, `total_scene_count`, `create_time`

**Why Aggregation**:
- ML algorithms need fixed-size feature vectors (one row per video)
- Middle segments vary by duration (2-7 windows) → average handles variable count
- Hook and closing always have 1 window → use directly

**Child Documents**:
- FeatureTransformation.md ("Temporal Features to ML Training Input" section)

**Future TI Document**:
- FeatureAggregationTI.md (aggregation algorithms, validation logic)

---

## Stage 4: Feature Transformation

**Purpose**: Transform aggregated features for ML algorithms (RF and K-Means have different requirements)

**Input**: `ml_analysis/aggregated_features.csv`

**Process**:

### 4.1: Random Forest Transformation
```python
# RF is scale-invariant, needs categorical encoding
df_rf = df.copy()

# One-hot encoding for categorical features
df_rf = pd.get_dummies(df_rf, columns=['dominant_emotion_id'])

# Extract temporal features from create_time
df_rf['hour'] = df_rf['create_time'].dt.hour
df_rf['day_of_week'] = df_rf['create_time'].dt.dayofweek
df_rf['is_weekend'] = (df_rf['day_of_week'] >= 5).astype(int)
df_rf['is_business_hours'] = ((df_rf['hour'] >= 9) & (df_rf['hour'] <= 17)).astype(int)

# Numerical features: Use directly (scale-invariant)
# No scaling needed for Random Forest

# Add target variable (for Contrastive strategy)
df_rf['is_top_performer'] = (df_rf.index < int(N * 0.8)).astype(int)

df_rf.to_csv("ml_analysis/rf_transformed.csv", index=False)
```

### 4.2: K-Means Transformation
```python
# K-Means is scale-sensitive, needs normalization
df_km = df.copy()

# Log + scale for right-skewed features (counts, variances)
skewed_features = ['hook_scene_count', 'middle_avg_word_count', 'total_scene_count']
for feature in skewed_features:
    df_km[f'{feature}_log'] = np.log1p(df_km[feature])  # log(1 + x)
    df_km[f'{feature}_scaled'] = (df_km[f'{feature}_log'] - df_km[f'{feature}_log'].min()) / \
                                  (df_km[f'{feature}_log'].max() - df_km[f'{feature}_log'].min())

# Scale [0-1] for already-normalized features (rates, percentages)
normalized_features = ['hook_eye_contact_rate', 'middle_avg_emotion_consistency']
for feature in normalized_features:
    df_km[f'{feature}_scaled'] = (df_km[feature] - df_km[feature].min()) / \
                                  (df_km[feature].max() - df_km[feature].min())

# Cyclical encoding for create_time (time is circular)
df_km['hour_sin'] = np.sin(2 * np.pi * df_km['create_time'].dt.hour / 24)
df_km['hour_cos'] = np.cos(2 * np.pi * df_km['create_time'].dt.hour / 24)

# One-hot encoding for dominant_emotion_id
df_km = pd.get_dummies(df_km, columns=['dominant_emotion_id'])

df_km.to_csv("ml_analysis/km_transformed.csv", index=False)
```

**Outputs**:
- `ml_analysis/rf_transformed.csv` (N videos, ~39 features)
- `ml_analysis/km_transformed.csv` (N videos, ~40 features)

**Why Different Transformations**:
- **Random Forest**: Scale-invariant, handles raw values well, needs categorical encoding
- **K-Means**: Scale-sensitive, needs normalization, sensitive to feature scale differences

**Child Documents**:
- FeatureTransformation.md (complete transformation specifications, feature lists)

**Future TI Document**:
- FeatureTransformationTI.md (transformation code, validation, edge cases)

---

## Stage 5: ML Model Training

**Purpose**: Train Random Forest and K-Means models per bucket

**Input**:
- `ml_analysis/rf_transformed.csv` (with `is_top_performer` labels)
- `ml_analysis/km_transformed.csv`

**Process**:

### 5.1: Random Forest Training (Classification)
```python
# Classification: Top 80% vs Bottom 20% performers
X = rf_transformed.drop(['is_top_performer'], axis=1)  # (N, 39)
y = rf_transformed['is_top_performer']  # (N,) - binary labels

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
).fit(X, y)

# Extract feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Generate predictions
predictions = rf_model.predict_proba(X)[:, 1]  # Probability of top performer

# Save model
joblib.dump(rf_model, "models/random_forest_v1.pkl")
```

### 5.2: K-Means Training (Clustering)
```python
# Clustering: Identify creative patterns
X = km_transformed  # (N, 40)

# Fit scalers per bucket (save for inference)
scalers = {}
for feature in X.columns:
    scaler = MinMaxScaler()
    scalers[feature] = scaler.fit(X[[feature]])

X_scaled = pd.DataFrame({
    col: scalers[col].transform(X[[col]]).flatten()
    for col in X.columns
})

# Train K-Means
kmeans_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
).fit(X_scaled)

cluster_assignments = kmeans_model.labels_
cluster_centroids = kmeans_model.cluster_centers_

# Save models
joblib.dump(kmeans_model, "models/kmeans_v1.pkl")
joblib.dump(scalers, "models/scalers.pkl")
```

### 5.3: Model Metrics
```json
{
  "random_forest": {
    "accuracy": 0.88,
    "precision": 0.85,
    "recall": 0.92,
    "f1_score": 0.88,
    "top_feature": "hook_eye_contact_rate",
    "top_feature_importance": 0.22
  },
  "kmeans": {
    "n_clusters": 3,
    "inertia": 45.2,
    "silhouette_score": 0.65,
    "cluster_sizes": [22, 35, 43]
  }
}
```

**Outputs**:
- `models/random_forest_v1.pkl`
- `models/kmeans_v1.pkl`
- `models/scalers.pkl` (for K-Means inference)
- `models/model_metrics.json`

**Why Two Models**:
- **Random Forest**: Answers "what differentiates top from bottom?" (contrastive learning)
- **K-Means**: Answers "what are the creative patterns?" (segmentation)

**Child Documents**:
- Kmeans.md (K-Means specific design, scaler fitting details)
- KMValidation.md (validation approach)

**Future TI Document**:
- MLModelTrainingTI.md (training code, hyperparameter tuning, cross-validation)

**Related Future Features**:
- Phase 2: ML Model Validation Framework (Stage 5 improvement)

---

## Stage 6: ML Analysis Generation

**Purpose**: Generate ML analysis JSONs for LLM consumption

**Input**:
- Trained models + transformed features
- `ml_analysis/aggregated_features.csv` (raw features for context)

**Process**:

### 6.1: Random Forest Analysis JSON
```python
# Extract feature importance and video-level predictions
rf_analysis = {
    "analysis_type": "random_forest",
    "bucket": bucket,
    "hashtag": hashtag,
    "video_count": N,

    "feature_importance": [
        {
            "feature": "hook_eye_contact_rate",
            "importance": 0.22,
            "top_performer_avg": 0.88,
            "bottom_performer_avg": 0.45,
            "gap": 0.43
        },
        # ... top 10 features
    ],

    "videos": [
        {
            "video_id": "123",
            "is_top_performer": 1,
            "prediction_confidence": 0.92,
            "features": {
                "hook_scene_count": 3,
                "middle_avg_word_count": 55,
                # ... all features
            }
        },
        # ... all N videos
    ]
}

save("ml_analysis/random_forest_analysis.json", rf_analysis)
```

### 6.2: K-Means Analysis JSON
```python
# Extract cluster assignments and centroids
kmeans_analysis = {
    "analysis_type": "kmeans",
    "bucket": bucket,
    "hashtag": hashtag,
    "n_clusters": 3,

    "cluster_summary": [
        {
            "cluster_id": 0,
            "cluster_name": "The Educator Pattern",
            "video_count": 22,
            "avg_engagement": 125000,
            "top_performer_percentage": 0.68,
            "defining_features": {
                "hook": {"high_eye_contact": 0.85, "moderate_scene_count": 3.2},
                "middle": {"high_word_count": 55, "consistent_emotion": 0.8},
                "closing": {"high_energy": 0.8}
            }
        },
        # ... 3 clusters
    ],

    "videos": [
        {
            "video_id": "123",
            "cluster_id": 0,
            "distance_to_centroid": 0.12,
            "features": {
                # ... all features
            }
        },
        # ... all N videos
    ]
}

save("ml_analysis/kmeans_analysis.json", kmeans_analysis)
```

**Output Size**: ~30KB per JSON (2 JSONs per bucket = ~60KB)

**Outputs**:
- `ml_analysis/random_forest_analysis.json` (~30KB)
- `ml_analysis/kmeans_analysis.json` (~30KB)

**Why Separate JSONs**:
- Smaller context per LLM call (30KB vs 60KB)
- More focused analysis prompts
- Easier to debug if one analysis fails
- Can run LLM calls in parallel

**Child Documents**:
- ML_LLMData.md (JSON format strategy, schema specifications)
- ML_LLMDataTI.md (JSON generation technical specs)

**Future TI Document**:
- MLAnalysisGenerationTI.md (JSON generation code, schema validation)

**Related Future Features**:
- Phase 2: LLM Data Strategy (Stage 6 core feature)

---

## Stage 7: LLM Report Generation

**Purpose**: Generate creative strategy reports via Claude API

**Input**:
- `ml_analysis/random_forest_analysis.json` (~30KB)
- `ml_analysis/kmeans_analysis.json` (~30KB)

**Process**:

### 7.1: Analysis LLM Calls (Insight Extraction)

**Purpose**: Extract structured insights from ML analysis without formatting concerns

**RF Analysis Call**:
```python
prompt = f"""
You are an ML analysis expert. Analyze Random Forest feature importance data.

HASHTAG: {hashtag}
BUCKET: {bucket}

INPUT DATA:
{rf_analysis_json}

TASK:
Extract insights in JSON format:

{{
  "top_features": [
    {{
      "feature": "hook_eye_contact_rate",
      "importance": 0.22,
      "interpretation": "Why this feature matters",
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "actionable_advice": "Maintain direct eye contact throughout hook"
    }},
    ... (top 5 features)
  ],
  "improvement_opportunities": [...],
  "model_performance": {{...}}
}}

Focus on INSIGHTS, not formatting. Be precise with numbers.
"""

rf_insights = claude_api.generate(prompt)
save("llm_reports/analysis/call_1_rf_raw_response.json", rf_insights)
```

**K-Means Analysis Call**:
```python
prompt = f"""
You are an ML analysis expert. Analyze K-Means clustering results.

HASHTAG: {hashtag}
BUCKET: {bucket}

INPUT DATA:
{kmeans_analysis_json}

TASK:
Extract insights in JSON format:

{{
  "clusters": [
    {{
      "cluster_id": 0,
      "name": "The Educator",
      "video_count": 22,
      "avg_engagement": 125000,
      "top_performer_percentage": 0.68,
      "defining_features": {{...}},
      "strategy_summary": "2-3 sentence description"
    }},
    ... (3 clusters)
  ],
  "recommendations": {{...}}
}}

Focus on INSIGHTS, not formatting. Be precise with numbers.
"""

kmeans_insights = claude_api.generate(prompt)
save("llm_reports/analysis/call_1_kmeans_raw_response.json", kmeans_insights)
```

**Consolidate Insights**:
```python
insights = {
    "bucket": bucket,
    "hashtag": hashtag,
    "rf_insights": json.loads(rf_insights),
    "kmeans_insights": json.loads(kmeans_insights),
    "generated_at": timestamp
}
save("llm_reports/analysis/insights.json", insights)
```

**Outputs**:
- `llm_reports/analysis/call_1_rf_prompt.txt`
- `llm_reports/analysis/call_1_rf_raw_response.json`
- `llm_reports/analysis/call_1_kmeans_prompt.txt`
- `llm_reports/analysis/call_1_kmeans_raw_response.json`
- `llm_reports/analysis/insights.json` (consolidated)

**LLM Calls per Bucket**: 2 (RF + K-Means)

---

### 7.2: Formatting LLM Call (Report Generation)

**Purpose**: Convert structured insights into polished, actionable creative strategy reports

**Input**: `llm_reports/analysis/insights.json`

```python
prompt = f"""
You are a creative strategy consultant. Generate polished markdown reports.

HASHTAG: {hashtag}
BUCKET: {bucket}

INPUT INSIGHTS:
{insights_json}

TASK:
Generate 5 markdown reports with professional formatting.
Each report should be clearly separated with "---REPORT: filename---" markers.

---REPORT: rf_feature_importance.md---

# Random Forest Feature Importance: {hashtag} ({bucket})

## Executive Summary
[2-3 sentences on what drives success for this bucket]

## Top 5 Success Drivers

### 1. {{feature.name}} (Importance: {{feature.importance}})

**What it means**: {{feature.interpretation}}

**The Gap**:
- Top performers: {{feature.top_performer_avg}}
- Bottom performers: {{feature.bottom_performer_avg}}
- Difference: {{feature.gap}}

**Action**: {{feature.actionable_advice}}

[Repeat for features 2-5]

---REPORT: strategy_1_the_educator.md---

# Strategy 1: The Educator

## Overview
{{cluster.strategy_summary}}

**Success Metrics**:
- Videos using this pattern: {{cluster.video_count}}
- Average engagement: {{cluster.avg_engagement}}
- Top performer rate: {{cluster.top_performer_percentage * 100}}%

## Key Features

### Hook (First 3 seconds)
[List defining characteristics]

### Middle Segments
[List defining characteristics]

### Closing (Last 3 seconds)
[List defining characteristics]

## Creation Checklist
- [ ] Hook: Maintain direct eye contact (0.85+ rate)
- [ ] Hook: Use moderate pacing (3-4 scene cuts)
- [ ] Middle: High word density (50-60 words)
...

---REPORT: strategy_2_visual_storyteller.md---
[Similar structure for cluster 1]

---REPORT: strategy_3_personal_journey.md---
[Similar structure for cluster 2]

---REPORT: bucket_summary.md---

# Bucket Summary: {hashtag} ({bucket})

## Pattern Overview

We identified 3 distinct creative strategies:

1. **{{cluster_1.name}}** ({{cluster_1.percentage}}%) - {{top_performer_rate}}% success rate
2. **{{cluster_2.name}}** ({{cluster_2.percentage}}%) - {{top_performer_rate}}% success rate
3. **{{cluster_3.name}}** ({{cluster_3.percentage}}%) - {{top_performer_rate}}% success rate

## Recommendations

**Start here**: {{recommendations.best_for_beginners}}

**Why**: {{recommendations.reasoning}}

IMPORTANT:
- Use markdown formatting (##, ###, bold, lists)
- Keep language actionable and specific
- Include exact numbers from insights
- Each report should be standalone
"""

reports = claude_api.generate(prompt)

# Parse LLM response (split by ---REPORT: markers)
parsed_reports = parse_multiple_reports(reports)

save("llm_reports/formatted/rf_feature_importance.md", parsed_reports['rf'])
save("llm_reports/formatted/strategy_1_the_educator.md", parsed_reports['strategy_1'])
save("llm_reports/formatted/strategy_2_visual_storyteller.md", parsed_reports['strategy_2'])
save("llm_reports/formatted/strategy_3_personal_journey.md", parsed_reports['strategy_3'])
save("llm_reports/formatted/bucket_summary.md", parsed_reports['summary'])
```

**Outputs**:
- `llm_reports/formatted/call_2_prompt.txt`
- `llm_reports/formatted/call_2_raw_response.json`
- `llm_reports/formatted/rf_feature_importance.md`
- `llm_reports/formatted/strategy_1_the_educator.md`
- `llm_reports/formatted/strategy_2_visual_storyteller.md`
- `llm_reports/formatted/strategy_3_personal_journey.md`
- `llm_reports/formatted/bucket_summary.md`

**LLM Calls per Bucket**: 1 (all 5 reports in single call)

---

### 7.3: PDF Generation (No LLM)

**Purpose**: Convert markdown reports to professional PDFs

**Input**: `llm_reports/formatted/*.md` (5 markdown files per bucket)

```python
from markdown2pdf import convert_markdown_to_pdf

reports = [
    "rf_feature_importance",
    "strategy_1_the_educator",
    "strategy_2_visual_storyteller",
    "strategy_3_personal_journey",
    "bucket_summary"
]

for report in reports:
    md_path = f"llm_reports/formatted/{report}.md"
    pdf_path = f"reports/{report}.pdf"

    convert_markdown_to_pdf(
        md_path,
        pdf_path,
        css_template="creative_report.css"  # Custom styling
    )
```

**Outputs** (`reports/` directory):
- `rf_feature_importance.pdf`
- `strategy_1_the_educator.pdf`
- `strategy_2_visual_storyteller.pdf`
- `strategy_3_personal_journey.pdf`
- `bucket_summary.pdf`

**Report Structure**:
- Professional formatting (headers, lists, bold text)
- Branded styling (Tumi Labs colors, fonts)
- Actionable checklists
- Specific numbers and metrics
- Example video references

**No LLM Calls**: Pure formatting (no API costs, fast execution)

---

### Stage 7 Summary

**LLM Calls per Bucket**:
- Analysis calls: 2 (RF + K-Means)
- Formatting calls: 1 (all 5 reports)
- **Total**: 3 calls per bucket

**LLM Calls per Hashtag** (3 qualified buckets):
- Analysis calls: 6 (2 × 3)
- Formatting calls: 3 (1 × 3)
- **Total**: 9 calls per hashtag

**Cost Estimate** (assuming $0.15 per call):
- Per bucket: $0.45
- Per hashtag: $1.35

**Duration Estimate**:
- Analysis calls: ~30 seconds each
- Formatting calls: ~45 seconds each
- PDF generation: ~5 seconds per report
- **Total per hashtag**: ~8-12 minutes (if sequential), ~3-5 minutes (if parallel)

**Why Two-Call Approach**:
- Better quality through separation of concerns
- Easier iteration (change formatting without re-analysis)
- Intermediate insights valuable for debugging and reuse
- Only moderate cost increase vs single-call approach

**Child Documents**:
- MLCreativeReports.md (report design & LLM prompts)

**Future TI Document**:
- LLMReportGenerationTI.md (LLM integration code, prompt templates, PDF generation)

**Related Future Features**:
- Phase 1: Creative Report Output (Stage 7 core feature)

---