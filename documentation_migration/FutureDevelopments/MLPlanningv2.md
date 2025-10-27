# ML Production Pipeline Planning v2

> **Version**: 2.0 (Stage-Based Architecture)
> **Parent Document**: MLPlanning.md (legacy)
> **Structure**: Organized by processing stages for easy TI doc creation

---

## Implementation Status Tracker

| Feature | Status | Implementation Date | Notes |
|---------|--------|-------------------|-------|
| **Hashtag Cluster Strategy** | ✅ IMPLEMENTED | 2025-10-13 | Multi-hashtag scraping with provenance tracking |
| Checkpoint Resume System | 📋 PLANNED | TBD | Stage 2 resume capability |
| Pipeline Validation | 📋 PLANNED | TBD | Stage 2.4 outlier detection |
| ML Model Validation Framework | 📋 PLANNED | TBD | Stage 5 improvement |

**Legend**:
- ✅ IMPLEMENTED: Feature is live in production
- 🔄 IN PROGRESS: Active development
- 📋 PLANNED: Documented, not yet started

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
   - Multi-tenant data structure: Client → Analysis Type → Target → Mode+Strategy → Buckets → Videos
   - Bucket-specific analysis within client/target boundaries
   - Persistent configuration management via config.json

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
- **ML Models**: 90 models total across 8 duration buckets for complete pattern coverage:
  - 8 Video-Level Random Forest models (1 per bucket) - Detects cross-window patterns and temporal progressions
  - 41 Window-Level Random Forest models (1 per window per bucket) - Validates window-specific feature importance
  - 41 Window-Level K-Means models (1 per window per bucket) - Discovers creative strategies within each video section
  - Architecture rationale: Dual RF + window-level K-Means prevents blind spots (video-level RF captures "hook→middle consistency", window-level captures "what makes a strong hook", K-Means discovers "3 ways to do strong hooks")
- **Output**: Duration-specific creative recommendations (Total Creative Reports being defined)
- **Processing**: Sequential (one-by-one) with resumption capability

---

## Client Architecture & Storage

### Directory Structure

```
/config/
├── hashtag_clusters/                       # Cluster configuration files (NEW: 2025-10-13)
│   ├── nutrition.json                      # Multi-hashtag cluster definition
│   ├── fitness.json                        # Example cluster config
│   └── wellness.json                       # Example cluster config
│
/data/
├── clients/
│   ├── {client_id}/
│   │   ├── hashtags/                       # Hashtag-based analyses (supports cluster mode)
│   │   │   └── {cluster_id}/               # e.g., "nutrition", "fitness" (cluster name, NOT "#nutrition")
│   │   │       ├── cluster_analytics.json  # Cluster scraping metrics (NEW: 2025-10-13)
│   │   │       │                           # Provenance, per-hashtag contribution, overlaps
│   │   │       ├── top_contrastive/        # Analysis: top mode + contrastive strategy
│   │   │       │   ├── config.json         # Stage 0 output: {mode: "top", strategy: "contrastive", date_filter: "last_90_days", run_date: "2025-01-28", video_count: 300}
│   │   │       │   ├── winner_analysis.json  # Stage 1.3 output: Bucket distribution analysis (top_100_distribution, top_3_buckets)
│   │   │       │   ├── buckets/            # Duration-based ML buckets (8 total)
│   │   │       │   │   ├── bucket_0-3s/
│   │   │       │   │   │   ├── videos/     # Raw MP4s: N files (configurable via --video-count)
│   │   │       │   │   │   │                # Contrastive: 80% top + 20% bottom of N (e.g., 80+20 if N=100)
│   │   │       │   │   │   ├── analysis/   # RumiAI outputs for N videos
│   │   │       │   │   │   │   ├── insights/   # temporal_windows JSON (1 per video)
│   │   │       │   │   │   │   ├── unified/    # Intermediate timeline+ml_data (debugging)
│   │   │       │   │   │   │   └── service_debug/  # emotion_detection, audio_energy outputs
│   │   │       │   │   │   ├── validation/     # Pipeline validation outputs (Stage 2.4)
│   │   │       │   │   │   │   ├── rolling_stats.json           # Running statistics per feature
│   │   │       │   │   │   │   └── validation_summary.json      # Summary of anomalies
│   │   │       │   │   │   ├── flagged_videos/ # Investigation packages for anomalies
│   │   │       │   │   │   │   └── {video_id}/ # Centralized troubleshooting folder
│   │   │       │   │   │   │       ├── video.mp4
│   │   │       │   │   │   │       ├── temporal_windows_updated.json
│   │   │       │   │   │   │       ├── unified_analysis.json
│   │   │       │   │   │   │       ├── service_debug/
│   │   │       │   │   │   │       └── validation_report.json
│   │   │       │   │   │   ├── ml_analysis/    # ML pipeline outputs
│   │   │       │   │   │   │   ├── aggregated_features.csv              # Stage 3: Aggregated temporal windows (N videos)
│   │   │       │   │   │   │   ├── rf_transformed.csv                   # Stage 4: Video-level RF (~190 features)
│   │   │       │   │   │   │   ├── hook_rf_transformed.csv              # Stage 4: Window-level RF (22 features)
│   │   │       │   │   │   │   ├── middle_1_rf_transformed.csv          # Stage 4: Window-level RF (22 features)
│   │   │       │   │   │   │   ├── middle_2_rf_transformed.csv          # Stage 4: Window-level RF (22 features)
│   │   │       │   │   │   │   ├── middle_3_rf_transformed.csv          # Stage 4: Window-level RF (22 features)
│   │   │       │   │   │   │   ├── middle_4_rf_transformed.csv          # Stage 4: Window-level RF (22 features)
│   │   │       │   │   │   │   ├── closing_rf_transformed.csv           # Stage 4: Window-level RF (22 features)
│   │   │       │   │   │   │   ├── hook_km_transformed.csv              # Stage 4: Window-level K-Means (~39 features)
│   │   │       │   │   │   │   ├── middle_1_km_transformed.csv          # Stage 4: Window-level K-Means (~39 features)
│   │   │       │   │   │   │   ├── middle_2_km_transformed.csv          # Stage 4: Window-level K-Means (~39 features)
│   │   │       │   │   │   │   ├── middle_3_km_transformed.csv          # Stage 4: Window-level K-Means (~39 features)
│   │   │       │   │   │   │   ├── middle_4_km_transformed.csv          # Stage 4: Window-level K-Means (~39 features)
│   │   │       │   │   │   │   ├── closing_km_transformed.csv           # Stage 4: Window-level K-Means (~39 features)
│   │   │       │   │   │   │   ├── random_forest_analysis.json          # Stage 6: ~30KB - Input to LLM Call 1
│   │   │       │   │   │   │   └── kmeans_analysis.json                 # Stage 6: ~30KB - Input to LLM Call 1
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
| **ML Analysis** | 6 months | Compressed after 60 days |
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
  --country-code {US|BR|global} \                    # Stage 0.6: Country Code
  --report-type {single|comparison} \                # Stage 0.7: Report Type
  --report-audience {client|internal|creator} \      # Stage 0.8: Report Audience
  --auto-confirm                                     # Skip interactive prompts (CI/CD)
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

| Type | CLI Flag | Target Format | Data Source | Primary Use Case | Scraping Mode |
|------|----------|---------------|-------------|------------------|---------------|
| `hashtag` | `--analysis-type hashtag` | `nutrition` (cluster ID) | TikTok hashtag search (multi-hashtag cluster) | Market research - identify viral patterns | **Cluster mode** (4 hashtags × 2 runs) |
| `competitor` | `--analysis-type competitor` | `@rival_brand` | TikTok profile | Competitive intelligence - understand rivals | Single mode (1 profile, 800 videos) |
| `creator` | `--analysis-type creator` | `@potential_affiliate` | TikTok profile | Creator vetting - assess fit for hiring | Single mode (1 profile, 800 videos) |

**CLI Usage**:
```bash
--analysis-type hashtag     # Analyze hashtag content (cluster mode)
--analysis-type competitor  # Analyze competitor profile (single mode)
--analysis-type creator     # Analyze creator profile (single mode)
```

**Cluster Mode (Hashtag Analysis Only)**:

**What it is**: Multi-hashtag scraping strategy that combines semantically related hashtags to maximize unique video discovery.

**Why it matters**: Single hashtag scraping with US geographic filtering reduces video volume by 57%. Cluster mode provides 2-3x more unique videos while maintaining semantic relevance.

**Architecture**:
- **Cluster config**: `/config/hashtag_clusters/{cluster_id}.json`
- **Example**: `nutrition` cluster = `#nutrition` + `#nutritionist` + `#nutritiontips` + `#nutritioncoach`
- **Scraping**: 4 hashtags × 2 runs = 8 scrapes with provenance tracking
- **Result**: ~1,900 videos scraped → ~1,400 unique videos after deduplication

**Key Concepts**:
- **Narrow semantic clustering**: Related hashtags with 20-30% overlap (not too broad, not too narrow)
- **Provenance tracking**: System tracks which hashtags/runs found each video (`source_hashtags`, `source_runs`)
- **Cluster analytics**: Per-hashtag contribution, pairwise overlaps, run effectiveness metrics
- **Single hashtag deprecated**: As of 2025-10-10, single hashtag scraping (e.g., `--target "#nutrition"`) is deprecated. Use cluster mode instead.

**When to use**:
- **Hashtag analysis**: Use cluster mode (default, recommended)
- **Competitor/Creator analysis**: Use single profile mode (no clusters needed)

**Key Differences**:

| Aspect | Hashtag | Competitor | Creator |
|--------|---------|------------|---------|
| **Apify Scraper** | clockworks/tiktok-scraper (GdWCkxBtKWOsKjdch) | clockworks/tiktok-scraper (GdWCkxBtKWOsKjdch) | clockworks/tiktok-scraper (GdWCkxBtKWOsKjdch) |
| **Video Source** | All TikTok users posting with hashtag | Single profile's content | Single profile's content |
| **ML Training** | Yes (classification models) | Optional (descriptive only) | No (uses existing models) |
| **Default Mode** | `top` | `top` | `recent` |
| **Default Strategy** | `contrastive` | `top` | `top` |
| **Default Date Filter** | `last_90_days` | `last_90_days` | `last_30_days` |
| **Default Video Count** | 100 | 40 | 40 |

**Why Target Type Matters**:
- Same unified scraper (supports both hashtags and profiles)
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

**Business Rationale**:
- Shares are 10x more valuable than views alone (viral indicator)
- Formula prioritizes "share-worthy" content over passive consumption
- Example: Video A (100K views, 100 shares, score=110K) outranks Video B (105K views, 10 shares, score=106.05K)
- Validated through initial client feedback showing share rate correlates with campaign success

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
- Apify's unified Profile Scraper doesn't support server-side date filtering for hashtags
- Client-side filtering used for consistency across all target types (hashtag, competitor, creator)
- Ensures uniform behavior regardless of analysis type

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

## Stage 0.6: Country Code

**Purpose**: Controls geographic content filtering via Apify proxy routing

**CLI Parameter**: `--country-code {country}`

**Available Values**:

| Value | Proxy Behavior | Content Returned | Use Case |
|-------|----------------|------------------|----------|
| `US` | proxyCountryCode: "US" | US-specific trending content | US market analysis (default) |
| `BR` | proxyCountryCode: "BR" | Brazil-specific trending content | Brazilian market analysis |
| `global` | No proxy parameter | Unfiltered global content | Cross-market comparison |

**Default**: `US`

**CLI Usage**:
```bash
--country-code US      # US market analysis (default)
--country-code BR      # Brazilian market analysis
--country-code global  # Global content (no filtering)
```

**How It Works**:
- Apify's `proxyCountryCode` parameter routes scraping through country-specific proxies
- TikTok's algorithm returns region-specific trending content based on proxy location
- "global" mode omits the parameter, letting Apify use default routing (mixed results)

**Business Value**:
- **Market-specific insights**: Analyze what works in specific geographic markets
- **Localization research**: Identify region-specific creative patterns
- **Multi-market comparison**: Compare US vs BR vs global trends
- **International expansion**: Understand new market requirements before launch

**Default Per Target Type**:
- **Hashtag**: `US` (US market research default)
- **Competitor**: `US` (analyze US-based competitors)
- **Creator**: `US` (vet US-based creators)

**Child Documents**: None (native to MLPlanning)

**Future TI Document**: VideoDiscoveryTI.md (country code implementation)

---

## Stage 0.7: Report Types

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

## Stage 0.8: Report Audience

**Purpose**: Determines target audience for generated reports (affects language, detail level, formatting)

**Available Audiences**:

| Audience | Who It's For | Language Style | Detail Level | Use Case |
|----------|-------------|----------------|--------------|----------|
| `client` | Brand stakeholders, marketing teams | Formal, business-oriented | High-level insights, strategic recommendations | Hashtag, Competitor analysis |
| `internal` | Tumi Labs team, data scientists | Technical, analytical | Full technical details, model metrics, raw data | Debugging, model validation, research |
| `creator` | Content creators, influencers | Casual, actionable, encouraging | Practical tips, specific actions to take | Creator monitoring, coaching |

**CLI Usage**:
```bash
--report-audience client     # Business report for brand stakeholders (default for hashtag/competitor)
--report-audience internal   # Technical report for Tumi Labs team
--report-audience creator    # Actionable tips for content creators (default for creator analysis)
```

**Default Logic**:
- **Hashtag analysis**: `client` (brands analyzing market trends)
- **Competitor analysis**: `client` (brands benchmarking against rivals)
- **Creator analysis**: `creator` (coaching individual creators)

**Impact on Report Generation** (Stage 5):
- **Client reports**: Focus on "What's working in the market?" with strategic insights
- **Internal reports**: Include model performance metrics, feature importance, statistical confidence
- **Creator reports**: Frame insights as "How to improve your content" with specific actionable steps

**Example Output Differences**:

| Insight Type | Client | Internal | Creator |
|-------------|--------|----------|---------|
| **Pattern found** | "Top performers use dynamic camera angles" | "Feature: camera_movement, importance: 0.23, p<0.01" | "Try moving your camera! 75% of viral videos use this" |
| **Timing insight** | "Hook engagement peaks at 1.2s" | "Temporal window: hook, peak_engagement: 1.2s ±0.3s" | "Grab attention in the first 1-2 seconds - that's when viewers decide to stay" |
| **Music pattern** | "Trending audio increases shareability" | "Audio feature correlation: 0.45 with share_rate" | "Use trending sounds! They get 2x more shares" |

**Child Documents**: None (native to MLPlanning)

**Future TI Document**: ReportGenerationTI.md (audience-specific formatting logic)

---

# Part 3: Processing Pipeline (Linear Stages)

End-to-end pipeline from video discovery to final reports.

## Pipeline Overview

```
Stage 1: Video Discovery & Selection
    ↓ Selected video list (per bucket)
Stage 2: Video Processing (RumiAI Pipeline)
    ↓ temporal_windows_updated.json (N videos per qualified bucket)
    ↓ Stage 2.4: Pipeline Validation
    ↓ rolling_stats.json + flagged_videos/ (if anomalies detected)
Stage 3: Feature Aggregation
    ↓ aggregated_features.csv (N rows × ~65-215 features per bucket)
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

### 1.1: Cluster Orchestration (Hashtag Mode)

**For hashtag analysis**, the system uses **cluster mode** to maximize unique video discovery:

**Step 1: Load Cluster Configuration**
```python
# Load cluster config from /config/hashtag_clusters/{cluster_id}.json
cluster_config = load_json(f"/config/hashtag_clusters/{target}.json")

# Example cluster config:
# {
#   "cluster_id": "nutrition",
#   "primary_hashtag": "#nutrition",
#   "variant_hashtags": ["#nutritionist", "#nutritiontips", "#nutritioncoach"],
#   "scrape_config": {
#     "runs_per_hashtag": 2,
#     "delay_between_runs_ms": 120000,  # 2 minutes
#     "results_per_page": 800
#   }
# }
```

**Step 2: Multi-Hashtag Scraping Loop**
```python
all_hashtags = [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']
# Result: ["#nutrition", "#nutritionist", "#nutritiontips", "#nutritioncoach"]

all_videos = []
failed_scrapes = []

for hashtag in all_hashtags:
    for run in range(1, runs_per_hashtag + 1):
        # Scrape with retry logic (3 attempts, exponential backoff [5s, 15s, 45s])
        videos = scrape_with_retry(hashtag, run, max_retries=3)

        if videos:
            # Tag videos with provenance (which hashtags/runs found them)
            for video in videos:
                video['source_hashtags'] = [hashtag]
                video['source_runs'] = [run]

            all_videos.extend(videos)
        else:
            failed_scrapes.append({"hashtag": hashtag, "run": run, "error": "..."})

        # Delay between scrapes to avoid rate limiting
        time.sleep(delay_between_runs_ms / 1000)

# Result: 4 hashtags × 2 runs = 8 scrapes with provenance tracking
# Example: ~1,900 videos scraped (before deduplication)
```

**For competitor/creator analysis**, the system uses **single mode** (one profile, 800 videos, no clusters).

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

### 1.3: Cluster Deduplication with Provenance (Hashtag Mode Only)

**For hashtag cluster mode**, videos may appear in multiple hashtag scrapes. The system deduplicates while preserving provenance:

```python
def deduplicate_with_provenance(all_videos):
    """
    Deduplicate videos while merging source_hashtags and source_runs.

    Input: ~1,900 videos (4 hashtags × 2 runs × ~240 videos each)
    Output: ~1,400 unique videos with merged provenance
    """
    unique_map = {}

    for video in all_videos:
        video_id = video['id']

        if video_id in unique_map:
            # Duplicate found - merge provenance
            existing = unique_map[video_id]
            existing['source_hashtags'].extend(video['source_hashtags'])
            existing['source_runs'].extend(video['source_runs'])
        else:
            # New video - add to map
            unique_map[video_id] = video

    unique_videos = list(unique_map.values())

    # Example result:
    # Video 123 found by: ["#nutrition", "#nutritiontips"] in runs [1, 2]
    # Video 456 found by: ["#nutrition"] in run [1] (exclusive to one hashtag)

    return unique_videos

unique_videos = deduplicate_with_provenance(all_videos)
# Result: ~1,400 unique videos (27.8% deduplication rate)
```

**For competitor/creator analysis**, no deduplication needed (single profile, no duplicates).

### 1.4: Cluster Analytics Generation (Hashtag Mode Only)

**For hashtag cluster mode**, the system generates health metrics for cluster optimization:

```python
def generate_cluster_analytics(all_videos, unique_videos, cluster_config, failed_scrapes):
    """
    Generate cluster health analytics.

    5 Analytics Sections:
    1. Scrape summary: Total attempts, successes, duplication rate
    2. Per-hashtag contribution: Which hashtags contribute most unique videos
    3. Pairwise overlaps: Overlap percentage between hashtag pairs
    4. Run effectiveness: Does run 2 add significant new videos?
    5. Bucket distribution by source: Which hashtags populate which buckets
    """
    analytics = {
        "cluster_id": cluster_config['cluster_id'],
        "execution_date": datetime.now(timezone.utc).isoformat(),
        "scrape_summary": {
            "total_scrapes_attempted": 8,
            "total_scrapes_succeeded": 8,
            "total_scraped_videos": len(all_videos),      # 1,900
            "total_unique_videos": len(unique_videos),    # 1,400
            "overall_duplication_rate": 27.8,             # %
            "failed_scrapes": failed_scrapes
        },
        "per_hashtag_contribution": {
            "#nutrition": {
                "total_found": 782,
                "exclusive_videos": 450,
                "contribution_percentage": 55.9
            },
            # ... other hashtags
        },
        "pairwise_overlaps": {
            "nutrition_nutritionist": 18.2,  # % overlap
            # ... other pairs
        },
        "run_effectiveness": {
            "#nutrition": {
                "run_1_videos": 690,
                "run_2_videos": 720,
                "run_2_new_videos": 92,
                "run_2_new_percentage": 12.8
            },
            # ... other hashtags
        },
        "bucket_distribution_by_source": {}  # Empty for now (populated in Stage 1.6)
    }

    # Save to: /data/clients/{client_id}/hashtag/{cluster_id}/cluster_analytics.json
    save_cluster_analytics(analytics, client_id, cluster_id)

    return analytics

analytics = generate_cluster_analytics(all_videos, unique_videos, cluster_config, failed_scrapes)
```

**Output**: `cluster_analytics.json` (used for cluster optimization and cost savings)

**For competitor/creator analysis**, no cluster analytics needed.

### 1.5: Winner Analysis (Success-Based Distribution)
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

### 1.6: Video Selection Per Bucket (Strategy-Specific)

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

### 1.7: Interactive Confirmation

After bucket selection completes, CLI displays summary and prompts user to confirm before proceeding to Stage 2:

```
Selected Buckets (by winner concentration):
  1. 15-30s  →  28 videos  (32.0% of winners)
  2. 30-45s  →  24 videos  (24.0% of winners)
  3. 45-60s  →  20 videos  (16.0% of winners)

Total: 72 videos across 3 buckets

Proceed to Stage 2 (Download & Analysis)? [Y/n/details]
```

**Purpose**: Allow user to review bucket selection and abort before expensive operations (downloads, ML inference).

**User Options**:
- `Y` or `Enter`: Proceed to Stage 2
- `n`: Abort (exit code 130)
- `details`: Show full bucket analysis including runners-up

**Bypass**: Use `--auto-confirm` flag to skip prompt for CI/CD pipelines.

**Child Document**: VideoDiscoveryCHILD.md Step 2.4 (Interactive Confirmation - full implementation details)

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

## Stage 2.5: File Organization (Bucket Assignment)

**Purpose**: Organize temporal_windows_updated.json files from flat /insights/ directory into bucket-specific directories

**Why Separate from Stage 2?**:
- Stage 2 (rumiai_runner.py) processes videos one-at-a-time with no bucket awareness
- Stage 2 saves all outputs to flat `/insights/` directory (mixed durations)
- Stage 2.5 is a BATCH operation that runs ONCE after all Stage 2 processing completes
- Separates concerns: video analysis (Stage 2) vs file organization (Stage 2.5)
- Stage 3 requires bucket-organized inputs for efficient processing

**Input**:
- `/insights/{video_id}_temporal_windows_updated.json` (N files, flat structure, mixed durations)
- Each JSON contains `metadata.duration` field

**Process**:

### 2.5.1: Batch File Organization
```python
# Read all JSON files from flat insights directory
for json_file in glob("/insights/*_temporal_windows_updated.json"):
    # Extract duration from JSON
    data = load_json(json_file)
    duration = data['metadata']['duration']

    # Determine bucket assignment
    bucket = assign_bucket(duration)

    # Move to bucket-specific directory
    target_path = f"bucket_{bucket}/analysis/insights/{json_file.name}"
    move_file(json_file, target_path)
```

### 2.5.2: Bucket Assignment Logic
```python
def assign_bucket(duration):
    """
    Assign duration to bucket using lower-inclusive boundaries (matches upstream).

    Convention: [lower, upper) - includes lower bound, excludes upper bound

    Examples:
        0.0s  → bucket_0-3s
        2.99s → bucket_0-3s
        3.0s  → bucket_3-9s   (NOT bucket_0-3s)
        9.0s  → bucket_9-13s  (NOT bucket_3-9s)
        18.0s → bucket_18-33s
        119.99s → bucket_90-120s
        120.0s → ERROR (exceeds maximum)
    """
    if duration < 0:
        raise ValueError(f"Duration {duration}s is negative")
    if duration >= 120.0:
        raise ValueError(f"Duration {duration}s exceeds maximum 120s")

    # Lower-inclusive, upper-exclusive boundaries [lower, upper)
    if duration < 3.0:
        return '0-3s'
    elif duration < 9.0:
        return '3-9s'
    elif duration < 13.0:
        return '9-13s'
    elif duration < 18.0:
        return '13-18s'
    elif duration < 33.0:
        return '18-33s'
    elif duration < 60.0:
        return '33-60s'
    elif duration < 90.0:
        return '60-90s'
    else:  # duration < 120.0
        return '90-120s'
```

### 2.5.3: Validation & Logging
```python
# Track organization results
organization_summary = {
    'total_files': len(all_files),
    'organized_by_bucket': {},
    'skipped_files': [],
    'errors': []
}

# Verify all files organized
for bucket in BUCKETS:
    count = len(glob(f"bucket_{bucket}/analysis/insights/*.json"))
    organization_summary['organized_by_bucket'][bucket] = count

# Log summary
logger.info(f"Organized {total_files} files into {len(BUCKETS)} buckets")
logger.info(f"Distribution: {organization_summary['organized_by_bucket']}")
```

**Output**:
- Bucket directories populated with organized JSON files:
  ```
  bucket_18-33s/analysis/insights/
  ├── 7428596_temporal_windows_updated.json
  ├── 238506_temporal_windows_updated.json
  └── ... (N files with 18.0 <= duration < 33.0)

  bucket_33-60s/analysis/insights/
  ├── 9876543_temporal_windows_updated.json
  └── ... (M files with 33.0 <= duration < 60.0)
  ```
- Organization summary log (files per bucket, processing time, any errors)

**Error Handling**:
- **Missing duration field**: Skip file, log error, continue processing
- **Invalid duration (< 0 or >= 120)**: Skip file, log error, continue processing
- **Malformed JSON**: Skip file, log error, continue processing
- **File move failure**: Retry once, then log error and continue
- **No files to organize**: Log warning, exit gracefully (not an error)
- **All files fail**: Log critical error, exit with error code

**When This Runs**:
- **AFTER**: All Stage 2 video processing completes (batch operation)
- **BEFORE**: Stage 3 Feature Aggregation begins
- **FREQUENCY**: Once per hashtag analysis run

**Invocation**:
```bash
python3 scripts/stage2_5_organize.py \
  --source-dir="/insights" \
  --client="test_run" \
  --target-type="hashtags" \
  --target-name="fitness" \
  --strategy="top_contrastive"
```

**Child Documents**:
- FileOrganizationCHILD.md (complete HLD with edge cases, validation, error handling)

**Future TI Document**:
- FileOrganizationTI.md (implementation of batch file organization logic)

**Related Future Features**:
- None (core pipeline component, not an enhancement)

---

## Stage 2.6 & 2.7: Content Analysis (Discovery & Classification)

**Purpose**: Extract qualitative content insights (hook strategies, pain points, engagement drivers) from video transcripts/captions using LLM-powered taxonomy classification

**Why Separate from Stage 2?**:
- Stage 2 (RumiAI) extracts quantitative ML features (eye_contact, energy_level, scene_count)
- Stage 2.6/2.7 analyzes semantic content patterns that can't be captured by ML features alone
- Enables Stage 7 reports to combine quantitative insights ("eye_contact importance: 0.23") with qualitative insights ("60% of top videos use problem_solution hook")
- Two-step process requires human curation between discovery and classification
- Uses different AI models (Sonnet for discovery, Haiku for classification)

**Input**:
- **Stage 2.6 Discovery**:
  - `selection_manifest.json` (from Stage 2.5 - top 3 buckets, video lists)
  - `speech_transcriptions/{video_id}_whisper.json` (50 sampled transcripts)
- **Stage 2.7 Classification**:
  - `content_taxonomies/{hashtag}_taxonomy.json` (manually curated after 2.6)
  - `speech_transcriptions/{video_id}_whisper.json` (120 videos: 20 top + 20 bottom per bucket)
  - `unified_analysis/{video_id}.json` (captions and hashtags)

**Process**:

### 2.6.1: Discovery Sampling (One-Time per Hashtag)
```python
# Sample 50 transcripts stratified across top 3 buckets
def sample_transcripts_for_discovery(manifest_path, sample_size=50):
    manifest = load_json(manifest_path)
    top_3_buckets = manifest['selected_buckets']  # ["33_60s", "60_90s", "90_120s"]

    samples_per_bucket = sample_size // 3  # ~17 per bucket
    sampled_transcripts = []

    for bucket in top_3_buckets:
        top_performers = manifest['videos_by_bucket'][bucket]['top_performers']
        sampled_ids = random.sample(top_performers, min(samples_per_bucket, len(top_performers)))

        for video_id in sampled_ids:
            transcript_data = load_json(f"speech_transcriptions/{video_id}_whisper.json")
            sampled_transcripts.append({
                "video_id": video_id,
                "text": transcript_data['text'],
                "bucket": bucket
            })

    return sampled_transcripts
```

### 2.6.2: LLM Pattern Discovery
```python
# Discover content patterns using Claude 3.5 Sonnet
def discover_patterns_llm(transcripts, hashtag):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Build discovery prompt with 50 transcripts
    prompt = f"""Analyze these {len(transcripts)} transcripts from #{hashtag}.

    Identify recurring patterns across 6 categories:
    1. Content Categories (format types: recipe_tutorial, supplement_review, etc.)
    2. Hook Strategies (opening patterns: problem_solution, direct_statement, etc.)
    3. Audience Pain Points (problems mentioned: bloating, low_energy, etc.)
    4. Trending Keywords (topics: protein, gut_health, holistic, etc.)
    5. Engagement Drivers (shareability tactics: before_after, specific_metrics, etc.)
    6. Content Tactics (presentation styles: personal_story, direct_to_camera, etc.)

    Only include patterns appearing in 10%+ of videos (minimum 3 videos).
    Return JSON with name, frequency, examples per pattern.
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        timeout=120,
        system="You are an expert content analyst...",
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)
```

### 2.6.3: Manual Curation (HUMAN STEP)
**MANUAL INTERVENTION REQUIRED** (~2 hours per hashtag):
1. Review `{hashtag}_raw_discovery.json`
2. Filter patterns: remove <10% frequency, brand-specific, hyper-granular
3. Add definitions for semantic categories (content_categories, hook_strategies)
4. Save curated taxonomy to `{hashtag}_taxonomy.json`
5. Resume pipeline with `--resume-from classification`

### 2.7.1: Video Classification
```python
# Classify 120 videos (20 top + 20 bottom per bucket × 3 buckets)
def classify_videos(manifest_path, taxonomy_path, hashtag, client_id):
    taxonomy = load_json(taxonomy_path)
    manifest = load_json(manifest_path)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    for bucket in manifest['selected_buckets']:
        # Select 20 top + 20 bottom per bucket
        videos = (
            manifest['videos_by_bucket'][bucket]['top_performers'][:20] +
            manifest['videos_by_bucket'][bucket]['bottom_performers'][:20]
        )

        for video_id in videos:
            # Load inputs
            transcript = load_transcript(video_id)
            caption, hashtags = load_caption_and_hashtags(video_id)

            # Classify with Haiku using 3-zone prompt structure
            classification = classify_video_llm(
                video_id, transcript, caption, hashtags, taxonomy, client
            )

            # Save classification
            output_path = f"bucket_{bucket}/content_analysis/{video_id}_content.json"
            save_json(output_path, classification)
```

**Output**:
- **Stage 2.6**: `content_taxonomies/{hashtag}_raw_discovery.json` (LLM output, needs curation)
- **Manual Step**: `content_taxonomies/{hashtag}_taxonomy.json` (curated, production-ready)
- **Stage 2.7**: `bucket_{bucket}/content_analysis/{video_id}_content.json` × 120 files

Example classification output:
```json
{
  "video_id": "7526250443832331550",
  "taxonomy_version": "stage2.6_output",
  "content_category": "wellness_practice",
  "hook_strategy": "direct_statement",
  "pain_points": ["menstrual_discomfort"],
  "keywords": ["holistic", "wellness"],
  "engagement_drivers": ["personal_testimony", "product_link"],
  "content_tactics": ["direct_to_camera", "product_demonstration"],
  "caption_analysis": {
    "hook_type": "statement",
    "cta_type": "link_in_bio",
    "brand_mention_present": true,
    "emoji_usage": "some",
    "caption_length": "long",
    "hashtag_count": 9,
    "hashtag_placement": "end"
  },
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

**Error Handling**:
- **Missing transcript**: Classify using caption/hashtags only, set `transcript_available=false`
- **LLM API timeout (>120s discovery, >30s per video)**: Retry 3x with exponential backoff, then fail
- **Invalid JSON from LLM**: Retry 3x, then fail with error
- **Missing taxonomy file**: Fail-fast with message "Run Stage 2.6 discovery and complete manual curation first"
- **Empty transcript**: Graceful degradation (caption/hashtag-only classification)

**When This Runs**:
- **Stage 2.6 Discovery**: AFTER Stage 2.5 (needs selection_manifest.json), ONCE per hashtag (initial setup)
- **Manual Curation**: Human completes before Stage 2.7 can run (~2 hours)
- **Stage 2.7 Classification**: Every run after taxonomy exists, BEFORE Stage 3
- **FREQUENCY**: Discovery once per hashtag, Classification every run

**Invocation**:
```bash
# Stage 2.6: Discovery (one-time)
python3 scripts/content_analysis.py \
  --client="test_run" \
  --hashtag="nutrition" \
  --stop-after=discovery

# [MANUAL CURATION STEP - Human reviews and curates taxonomy]

# Stage 2.7: Classification (every run)
python3 scripts/content_analysis.py \
  --client="test_run" \
  --hashtag="nutrition" \
  --resume-from=classification
```

**Child Documents**:
- ContentAnalysisCHILD.md (complete HLD with refined prompts, 3-zone structure, grounding rules)
- 2.6HashtagCritique.md (Discovery prompt refinement decisions)
- 2.7ClassificationCritique.md (Classification prompt refinement decisions)

**Future TI Document**:
- ContentAnalysisCHILDTI.md (implementation with API calls, validation, error handling)

**Related Future Features**:
- Phase 2: Semi-automated taxonomy curation (reduce 2h to 30min using learned heuristics)
- Phase 3: Universal taxonomy with hashtag extensions (if 70%+ overlap across hashtags)
- Phase 4: Upgrade Haiku→Sonnet for classification if misclassification rate >20%

---

## Stage 3: Feature Aggregation

**Purpose**: Extract fixed-size feature vectors from temporal windows (bucket-specific structure)

**Input**: N × `temporal_windows_updated.json` files per qualified bucket (N from --video-count)

**Key Architectural Insight**:
Each bucket processes videos with **identical window structures**, eliminating the ragged array problem:
- Bucket 0-3s: All videos have 1 window (Hook only - no closing)
- Bucket 3-9s: All videos have 2 windows (Hook + Closing only)
- Bucket 9-13s, 13-18s: All videos have 3 windows (Hook + Middle Aggregate + Closing)
  - Note: Middle segments aggregated for feature reliability in short windows
- Bucket 18-33s: All videos have 6 windows (Hook + 4 Middle + Closing)
- Bucket 33-60s, 60-90s, 90-120s: All videos have 7 windows (Hook + 5 Middle + Closing)

This enables **full temporal granularity** for most buckets, preserving narrative structure and pacing patterns. Buckets 9-13s and 13-18s use middle aggregation to ensure all 21 features are reliably measured (see Section 3.1.1).

**Process**:

### 3.1: Temporal Window Extraction (Bucket-Specific)

**Bucket 18-33s Example** (4 middle segments - fixed for all videos in this bucket):

```python
# Base features per window (actual RumiAI features - 21 total)
BASE_FEATURES = [
    'average_face_size', 'overlay_unique_count', 'has_captions',
    'scene_count', 'shortest_scene', 'longest_scene', 'scene_duration_variance',
    'object_count', 'person_count', 'dominant_emotion_id',
    'speech_coverage', 'word_count', 'energy_level', 'energy_variance',
    'energy_max', 'pitch_scatter_ratio', 'gesture_count', 'gaze_variance',
    'eye_contact_rate', 'emotional_valence', 'emotion_consistency'
]

for temporal_windows_json in bucket_jsons:
    windows = load_json(temporal_windows_json)
    video_features = {}

    # Hook features (1 window - all 21 base features)
    for feature in BASE_FEATURES:
        video_features[f'hook_{feature}'] = windows['hook'][feature]

    # Middle features (4 segments - FIXED for bucket 18-33s)
    # Full temporal granularity for 18-33s+ buckets (no aggregation)
    middle_segments = windows['middle_segments']  # Always 4 segments
    for i, segment in enumerate(middle_segments, start=1):
        for feature in BASE_FEATURES:
            video_features[f'middle_{i}_{feature}'] = segment[feature]

    # Closing features (1 window - all 21 base features)
    for feature in BASE_FEATURES:
        video_features[f'closing_{feature}'] = windows['closing'][feature]

    # Metadata (non-temporal, non-collinear) - 3 fields total
    video_features['video_id'] = temporal_windows_json.stem.replace('_temporal_windows_updated', '')
    video_features['create_time'] = windows['metadata']['create_time']
    video_features['gender'] = windows['metadata'].get('gender_detection', {}).get('gender')

    # Note: duration removed (redundant with bucket assignment)
    # Note: gender_confidence removed (not needed for ML training)

    # ❌ NO global features that sum temporal features (avoids collinearity)
    # DO NOT ADD: total_scene_count, total_word_count, total_energy, etc.
    # These would be collinear with hook + middle_1 + middle_2 + ... + closing

    aggregated_rows.append(video_features)
```

### 3.1.1: Middle Segment Aggregation (Buckets 9-13s, 13-18s)

**Rationale**: Short middle windows (1-4s) in buckets 9-13s and 13-18s produce unreliable measurements for 8 out of 21 features (38%): `scene_count`, `shortest_scene`, `longest_scene`, `scene_duration_variance`, `speech_coverage`, `word_count`, `gesture_count`, `gaze_variance`. Aggregating the 3 middle segments into a single "middle_aggregate" window creates 4.5-9.3s windows where ALL 21 features are reliably measured.

**Aggregation Strategies** (4 strategies applied based on feature semantics):
- **SUM**: Count features (cumulative) → `scene_count`, `word_count`, `object_count`, `person_count`, `overlay_unique_count`, `gesture_count`
- **MIN**: Minimum extreme values → `shortest_scene`
- **MAX**: Maximum extreme values → `longest_scene`
- **MODE**: Categorical features (most common) → `dominant_emotion_id`, `has_captions`
- **AVERAGE** (default): All other continuous/ratio features → `energy_level`, `eye_contact_rate`, `speech_coverage`, etc.

**Example for Bucket 9-13s**:
```python
# Instead of: middle_1_*, middle_2_*, middle_3_* (3 × 21 = 63 columns)
# Create: middle_aggregate_* (1 × 21 = 21 columns)

import numpy as np
import pandas as pd

middle_segments = windows['middle_segments']  # 3 segments for 9-13s bucket

for feature in BASE_FEATURES:
    # Collect non-null values from all 3 middle segments
    values = [seg[feature] for seg in middle_segments if seg.get(feature) is not None]

    if len(values) == 0:
        video_features[f'middle_aggregate_{feature}'] = None
        continue

    # Apply strategy based on feature type
    if feature in ['scene_count', 'word_count', 'gesture_count']:  # SUM
        video_features[f'middle_aggregate_{feature}'] = sum(values)
    elif feature == 'shortest_scene':  # MIN
        video_features[f'middle_aggregate_{feature}'] = min(values)
    elif feature == 'longest_scene':  # MAX
        video_features[f'middle_aggregate_{feature}'] = max(values)
    elif feature in ['dominant_emotion_id', 'has_captions']:  # MODE
        video_features[f'middle_aggregate_{feature}'] = pd.Series(values).mode()[0]
    else:  # AVERAGE (default for continuous features)
        video_features[f'middle_aggregate_{feature}'] = np.mean(values)
```

**Result**: Buckets 9-13s and 13-18s have **3 windows total** (hook + middle_aggregate + closing) instead of 5 windows with unreliable middle segment features.

**Why This Works**:

- **No ragged arrays**: All videos in a bucket have IDENTICAL window counts
- **Limited averaging**: Only buckets 9-13s and 13-18s aggregate middle segments (for feature reliability). All other buckets preserve full temporal evolution (e.g., emotional arc: neutral → happy → sad)
- **No collinearity**: Global features (like `total_scene_count`) would be mathematical sums of temporal features
- **Fixed-size vectors**: Required for ML algorithms (one row per video)

### 3.2: Bucket-Specific Feature Counts

**Output Feature Counts by Bucket** (Actual RumiAI values):

| Bucket | Middle Segments | Middle Type | Total Windows | Base Features × Windows | Metadata | **Total Features** |
|--------|-----------------|-------------|---------------|-------------------------|----------|-------------------|
| 0-3s | 0 (null) | N/A | 1 (hook only) | 21 × 1 = 21 | 3 | **24** |
| 3-9s | 0 (null) | N/A | 2 | 21 × 2 = 42 | 3 | **45** |
| 9-13s | 3 | Aggregated | 3 | 21 × 3 = 63 | 3 | **66** |
| 13-18s | 3 | Aggregated | 3 | 21 × 3 = 63 | 3 | **66** |
| 18-33s | 4 | Separate | 6 | 21 × 6 = 126 | 3 | **129** |
| 33-60s | 5 | Separate | 7 | 21 × 7 = 147 | 3 | **150** |
| 60-90s | 5 | Separate | 7 | 21 × 7 = 147 | 3 | **150** |
| 90-120s | 5 | Separate | 7 | 21 × 7 = 147 | 3 | **150** |

**Note**:
- **Base features**: 21 features per window (see Section 3.1 for complete list)
- **Metadata**: 3 fields (video_id, create_time, gender)
- **Middle Type**: "Aggregated" means middle segments combined into single middle_aggregate window (see Section 3.1.1)

### 3.3: Create Aggregated CSV
```python
import pandas as pd

df = pd.DataFrame(aggregated_rows)
df.to_csv("ml_analysis/aggregated_features.csv", index=False)

# Example output shape for bucket 18-33s with N=100 videos
# Shape: (100 videos, 129 features)
```

**Output**: `ml_analysis/aggregated_features.csv`
- Shape: **(N videos, 24-150 features)** depending on bucket (see table in Section 3.2)
- Example columns for **bucket 18-33s** (129 features total):
  - Metadata: `video_id`, `create_time`, `gender` (3 features)
  - Hook: `hook_scene_count`, `hook_eye_contact_rate`, `hook_word_count`, ... (21 features)
  - Middle 1: `middle_1_scene_count`, `middle_1_eye_contact_rate`, ... (21 features)
  - Middle 2: `middle_2_scene_count`, `middle_2_eye_contact_rate`, ... (21 features)
  - Middle 3: `middle_3_scene_count`, `middle_3_eye_contact_rate`, ... (21 features)
  - Middle 4: `middle_4_scene_count`, `middle_4_eye_contact_rate`, ... (21 features)
  - Closing: `closing_scene_count`, `closing_energy_level`, ... (21 features)
- Example columns for **bucket 9-13s** (66 features total):
  - Metadata: `video_id`, `create_time`, `gender` (3 features)
  - Hook: `hook_scene_count`, `hook_eye_contact_rate`, ... (21 features)
  - Middle Aggregate: `middle_aggregate_scene_count`, `middle_aggregate_eye_contact_rate`, ... (21 features)
  - Closing: `closing_scene_count`, `closing_energy_level`, ... (21 features)

**Collinearity Prevention**:
```python
# ❌ WRONG - Creates collinear features
total_scene_count = hook_scene_count + middle_1_scene_count + ... + closing_scene_count

# ✅ CORRECT - Use only temporal features
# ML models can learn relationships between temporal features naturally
# No need for explicit global aggregates
```

**Child Documents**:
- FeatureTransformation.md ("Temporal Features to ML Training Input" section)

**Future TI Document**:
- FeatureAggregationTI.md (extraction logic, bucket-specific handling, validation)

---

## Stage 3.4: Review CSV Generation

**Purpose**: Generate video_review.csv for manual outlier investigation in Excel

**Why Separate from Stage 3.1-3.3?**:
- Stage 3.1-3.3 generates `aggregated_features.csv` (ML training input, ~65-215 columns)
- Stage 3.4 generates `video_review.csv` (human review, same features + url column)
- Review CSV is OPTIONAL - deleting it doesn't impact ML pipeline

**Input**:
- `temporal_windows_updated.json` (N files per bucket, with metadata.url)
- Note: Requires Stage 2 modification - temporal_compute.py must include `url` in calculated_metadata

**Process**:
1. Load all temporal_windows_updated.json files for bucket
2. Extract features (same logic as aggregated_features.csv)
3. Check metadata.url presence (skip videos with missing url, log warning)
4. Build CSV rows: [video_id, url, duration, all_features]
5. Save as `bucket_{duration}/validation/video_review.csv`

**Output**:
- `bucket_{duration}/validation/video_review.csv`
- Row count: N videos (same as aggregated_features.csv, minus videos with missing url)
- Column count: ~67-217 columns (video_id + url + duration + all temporal features)

**User Workflow**:
1. Open video_review.csv in Excel
2. Apply conditional formatting to highlight outliers (Excel built-in feature)
3. Click `url` column to watch flagged videos on TikTok
4. Investigate why outliers occurred (encoding issues, edge cases, RumiAI bugs)
5. All videos still proceed to ML training (no exclusions)

**Stage 2 Prerequisite**:
Modify `temporal_compute.py` (line ~2650) to pass url through metadata:
```python
calculated_metadata = {
    'video_id': video_id,
    'duration': video_duration,
    'url': metadata.get('url'),  # ← ADD THIS LINE
    'digg_count': metadata.get('likes', 0),
    ...
}
```

**Error Handling**:
- Videos with missing url: Skip from review CSV, log warning, still included in aggregated_features.csv
- All videos missing url: Log error, skip video_review.csv generation, continue pipeline
- Disk full: Fail fast

**Child Documents**:
- ReviewCSVGenerationCHILD.md (complete HLD with schemas, tests, pseudocode)

**Future TI Document**:
- ReviewCSVGenerationTI.md (implementation of dual CSV generation logic)

**Related Features**:
- Phase 1: Manual Outlier Investigation (simplified from automated Pipeline Validation)

---

## Stage 4: Feature Transformation

**Purpose**: Transform aggregated features into three distinct formats for dual Random Forest + window-level K-Means architecture

**Input**: `ml_analysis/aggregated_features.csv` (bucket-specific feature count: ~65-215 features)

**Output Count** (varies by bucket): Reference `config.bucket_definitions.get_stage4_output_count(bucket)` for counts
- **Formula**: `1 + 3N` files (N = window count)
- **Example bucket_18-33s**: 19 files (1 Video RF + 6 Window RF + 6 Window KM + 6 Scalers)
- See FeatureTransformationCHILD.md for schemas

**Architectural Decision**: This stage creates **3 transformation pipelines** to support:
1. **Video-Level RF** (cross-window patterns)
2. **Window-Level RF** (within-window validation)
3. **Window-Level K-Means** (creative strategies per window)

**Process**:

### 4.1: Video-Level Random Forest Transformation

**Purpose**: Detect cross-window interactions and temporal progressions

**Key Principle**: Random Forest is scale-invariant but benefits from categorical encoding and derived temporal features.

**Bucket 18-33s Example** (6 windows: Hook + 4 Middle + Closing):

```python
import pandas as pd
import numpy as np

df_rf_video = df.copy()  # Input: (N videos, ~185 features)

# ===== 1. Categorical Encoding =====
# One-hot encoding for gender (if available)
if 'gender' in df_rf_video.columns:
    df_rf_video = pd.get_dummies(df_rf_video, columns=['gender'], prefix='gender')

# ===== 2. Temporal Features from create_time =====
df_rf_video['hour'] = df_rf_video['create_time'].dt.hour
df_rf_video['day_of_week'] = df_rf_video['create_time'].dt.dayofweek
df_rf_video['is_weekend'] = (df_rf_video['day_of_week'] >= 5).astype(int)
df_rf_video['is_business_hours'] = ((df_rf_video['hour'] >= 9) & (df_rf_video['hour'] <= 17)).astype(int)

# ===== 3. All Temporal Features Used As-Is =====
# All temporal features (hook_*, middle_1_*, ..., closing_*) used as-is
# RF is scale-invariant - no normalization needed

# Example features already in df_rf_video:
# - hook_scene_count, hook_eye_contact_rate, hook_word_count
# - middle_1_scene_count, middle_1_eye_contact_rate, middle_1_word_count
# - middle_2_*, middle_3_*, middle_4_*
# - closing_scene_count, closing_eye_contact_rate, closing_word_count

# ===== 4. Add Target Variable (Contrastive Strategy Only) =====
# For contrastive: Label top 80% as 1, bottom 20% as 0
df_rf_video['is_top_performer'] = (df_rf_video.index < int(N * 0.8)).astype(int)

# For top strategy: No target variable (descriptive analysis only)

# ===== 5. Save Transformed Features =====
df_rf_video.to_csv("ml_analysis/rf_transformed.csv", index=False)

# Output shape: (N videos, ~190 features) = 185 original + 5 temporal features
```

**Output**: `ml_analysis/rf_transformed.csv` (N videos, ~190 features for bucket 18-33s)

**What It Captures**: Cross-window patterns like energy progression, topic consistency, contrast effects, weak link detection

---

### 4.2: Window-Level Random Forest Transformation

**Purpose**: Validate which features matter within each specific window type

**Key Principle**: Separate models per window (hook, middle_1, middle_2, ..., closing) with 21 base features each

**Base Features Per Window** (~21 total):
- `scene_count`, `eye_contact_rate`, `word_count`, `speech_coverage`
- `energy_level`, `gesture_count`, `emotional_valence`, `emotion_consistency`
- `average_face_size`, `overlay_unique_count`, `has_captions`
- `shortest_scene`, `longest_scene`, `scene_duration_variance`
- `object_count`, `person_count`, `energy_variance`, `energy_max`
- `pitch_scatter_ratio`, `gaze_variance`, `dominant_emotion_id`

**Bucket 18-33s Example** (6 window types):

```python
# For each window type in bucket
for window_type in ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']:
    # Extract window-specific features from aggregated_features.csv
    window_features = df[[f'{window_type}_{feat}' for feat in BASE_FEATURES]]
    window_features.columns = BASE_FEATURES  # Remove prefix

    # Add target variable
    window_features['is_top_performer'] = (window_features.index < int(N * 0.8)).astype(int)

    # Save per-window transformed data
    window_features.to_csv(f'ml_analysis/{window_type}_rf_transformed.csv', index=False)

# Output shape per window: (N videos, 21 base features + 1 target = 22 columns)
```

**Outputs** (for bucket 18-33s):
- `ml_analysis/hook_rf_transformed.csv` (100 videos, 22 features)
- `ml_analysis/middle_1_rf_transformed.csv` (100 videos, 22 features)
- `ml_analysis/middle_2_rf_transformed.csv` (100 videos, 22 features)
- `ml_analysis/middle_3_rf_transformed.csv` (100 videos, 22 features)
- `ml_analysis/middle_4_rf_transformed.csv` (100 videos, 22 features)
- `ml_analysis/closing_rf_transformed.csv` (100 videos, 22 features)

**What It Captures**: Within-window feature importance - which features define a "strong hook" vs "weak hook", etc.

---

### 4.3: Window-Level K-Means Transformation

**Purpose**: Cluster windows to discover creative strategies per video section

**Key Principle**: K-Means is scale-sensitive, requiring normalization. Window-level clustering (21 features) avoids curse of dimensionality.

**Bucket 18-33s Example** (6 window types):

```python
# For each window type in bucket
for window_type in ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']:
    # Extract window-specific features from aggregated_features.csv
    window_features = df[[f'{window_type}_{feat}' for feat in BASE_FEATURES]]
    window_features.columns = BASE_FEATURES  # Remove prefix

    df_km_window = window_features.copy()

    # ===== 1. Log + Scale for Right-Skewed Features (Counts + Variances) =====
    count_features = ['scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count',
                      'overlay_unique_count', 'shortest_scene', 'longest_scene', 'scene_duration_variance',
                      'energy_variance', 'gaze_variance']  # 11 features total

    for feature in count_features:
        if feature in df_km_window.columns:
            # Log transform to reduce skewness
            df_km_window[f'{feature}_log'] = np.log1p(df_km_window[feature])
            # MinMax scale to [0, 1]
            df_km_window[f'{feature}_scaled'] = (
                (df_km_window[f'{feature}_log'] - df_km_window[f'{feature}_log'].min()) /
                (df_km_window[f'{feature}_log'].max() - df_km_window[f'{feature}_log'].min())
            )
            # Drop original raw feature and intermediate log feature
            df_km_window.drop(columns=[feature, f'{feature}_log'], inplace=True)

    # ===== 2. MinMax Scale for Already-Normalized Features (Rates, Ratios) =====
    rate_features = ['average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
                     'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency']  # 7 features total

    for feature in rate_features:
        if feature in df_km_window.columns:
            # MinMax scale to [0, 1] (no log needed - already normalized)
            df_km_window[f'{feature}_scaled'] = (
                (df_km_window[feature] - df_km_window[feature].min()) /
                (df_km_window[feature].max() - df_km_window[feature].min())
            )
            # Drop original feature
            df_km_window.drop(columns=[feature], inplace=True)

    # ===== 3. Shift + Scale for emotional_valence =====
    # emotional_valence is in [-1, 1] range, shift to [0, 1]
    if 'emotional_valence' in df_km_window.columns:
        df_km_window['emotional_valence_scaled'] = (df_km_window['emotional_valence'] + 1) / 2
        df_km_window.drop(columns=['emotional_valence'], inplace=True)

    # ===== 4. Label Encode for has_captions =====
    if 'has_captions' in df_km_window.columns:
        df_km_window['has_captions_encoded'] = df_km_window['has_captions'].astype(int)  # True→1, False→0
        df_km_window.drop(columns=['has_captions'], inplace=True)

    # ===== 5. One-hot for dominant_emotion_id =====
    if 'dominant_emotion_id' in df_km_window.columns:
        for emotion_id, emotion_name in enumerate(['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'], start=1):
            df_km_window[emotion_name] = (df_km_window['dominant_emotion_id'] == emotion_id).astype(int)
        df_km_window.drop(columns=['dominant_emotion_id'], inplace=True)

    # ===== 6. Save per-window K-Means transformed data =====
    df_km_window.to_csv(f'ml_analysis/{window_type}_km_transformed.csv', index=False)

# Output shape per window: (N videos, 39 transformed features)
# - 11 count features × 2 (log + scaled) = 22 features
# - 7 rate features × 1 (scaled) = 7 features
# - 1 emotional_valence (shift + scaled) = 1 feature
# - 1 has_captions (label encoded) = 1 feature
# - 1 dominant_emotion_id (one-hot to 7 binary features) = 7 features
# - Total: 22 + 7 + 1 + 1 + 7 = 38 features (+ metadata if included)
```

**Outputs** (for bucket 18-33s):
- `ml_analysis/hook_km_transformed.csv` (100 videos, 39 features)
- `ml_analysis/middle_1_km_transformed.csv` (100 videos, 39 features)
- `ml_analysis/middle_2_km_transformed.csv` (100 videos, 39 features)
- `ml_analysis/middle_3_km_transformed.csv` (100 videos, 39 features)
- `ml_analysis/middle_4_km_transformed.csv` (100 videos, 39 features)
- `ml_analysis/closing_km_transformed.csv` (100 videos, 39 features)

**What It Captures**: Creative patterns per window - "3 distinct hook strategies that all lead to viral success"

---

### 4.4: Complete Outputs Summary

**For bucket 18-33s** (6 windows):

| Transformation Type | Files Generated | Features per File | Purpose |
|-------------------|-----------------|-------------------|---------|
| **Video-Level RF** | 1 file | ~190 features | Cross-window pattern detection |
| **Window-Level RF** | 6 files | 22 features each | Within-window feature validation |
| **Window-Level K-Means** | 6 files | 39 features each | Per-window creative strategies |
| **Total** | **13 files** | — | Complete ML architecture |

**File Structure**:
```
bucket_18-33s/ml_analysis/
├── aggregated_features.csv              # Stage 3 output (input to Stage 4)
├── rf_transformed.csv                   # Video-level RF (190 features)
├── hook_rf_transformed.csv              # Window-level RF (22 features)
├── middle_1_rf_transformed.csv          # Window-level RF (22 features)
├── middle_2_rf_transformed.csv          # Window-level RF (22 features)
├── middle_3_rf_transformed.csv          # Window-level RF (22 features)
├── middle_4_rf_transformed.csv          # Window-level RF (22 features)
├── closing_rf_transformed.csv           # Window-level RF (22 features)
├── hook_km_transformed.csv              # Window-level K-Means (39 features)
├── middle_1_km_transformed.csv          # Window-level K-Means (39 features)
├── middle_2_km_transformed.csv          # Window-level K-Means (39 features)
├── middle_3_km_transformed.csv          # Window-level K-Means (39 features)
├── middle_4_km_transformed.csv          # Window-level K-Means (39 features)
└── closing_km_transformed.csv           # Window-level K-Means (39 features)
```

**Why Three Transformation Pipelines**:
- **Video-Level RF**: Captures cross-window interactions (energy progression, topic consistency)
- **Window-Level RF**: Validates per-window feature importance (direct K-Means validation)
- **Window-Level K-Means**: Discovers creative strategies with interpretable centroids (21 features vs 150)

**Architectural Advantages**:
- **Complete Pattern Coverage**: Cross-window AND within-window patterns captured
- **LLM-Friendly**: Window-level centroids (21 features) vs video-level (150 features) = 7x smaller context
- **Actionable Insights**: Per-section strategies ("Use this hook type") vs abstract patterns
- **No Blind Spots**: Dual RF ensures both temporal progressions AND window-specific features are validated

**Child Documents**:
- FeatureTransformation.md (complete transformation specifications, feature lists)
- KmeansClusteringStage6.md (dual RF + window-level K-Means architecture rationale)

**Future TI Document**:
- FeatureTransformationTI.md (transformation code, validation, edge cases)

---

## Stage 5: ML Model Training

**Purpose**: Train dual Random Forest models (video-level + window-level) and window-level K-Means models per bucket

> ⚠️ **CRITICAL IMPLEMENTATION WARNINGS**: This stage has HIGH-RISK bug hotspots identified during Phase 1 Business Critique (2025-10-14):
> - **Feature name mismatch** (K-Means vs RF) → guaranteed bug if not handled
> - **K-Means feature ranking logic** → conceptually complex, easy to implement wrong
>
> **BEFORE implementing**, read:
> - **Stage5_MLModelTraining_STUB.md** (ChildDocs/ - Section 3: Critical Implementation Warnings - PRE-FILLED)
> - **Stage5Tests.md** (comprehensive testing spec with bug prevention tests)
> - **Critique_MLModelTraining.md** (Phase 1 Q&A decisions, especially Q3 validation protocol)
>
> Do NOT start from scratch. Expand the STUB HLD which has critical warnings pre-written.

**Architectural Decision**: This stage trains **90 models total** across 8 buckets:
1. **8 Video-Level RF models** (1 per bucket) - Cross-window patterns
2. **41 Window-Level RF models** (1 per window per bucket) - Within-window validation
3. **41 Window-Level K-Means models** (1 per window per bucket) - Creative strategies

**Input** (from Stage 4):
- `ml_analysis/rf_transformed.csv` (video-level RF, ~190 features)
- `ml_analysis/{window}_rf_transformed.csv` (window-level RF, 22 features × 6 windows)
- `ml_analysis/{window}_km_transformed.csv` (window-level K-Means, ~30 features × 6 windows)

**Process**:

### 5.1: Video-Level Random Forest Training (Cross-Window Patterns)

**Purpose**: Detect cross-window interactions, temporal progressions, and weak link effects

**Bucket 18-33s Example**:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load video-level transformed data
X = pd.read_csv('ml_analysis/rf_transformed.csv')  # (100 videos, ~190 features)
y = X['is_top_performer']

# Check label distribution (C7 Compatibility Fix - added 2025-10-20)
unique_labels = y.unique()
can_train_rf = len(unique_labels) >= 2

if not can_train_rf:
    # 'top' mode produces single class - RF training impossible
    print("Skipping RF: Single class detected (expected in 'top' mode)")
    rf_video = None
else:
    # Binary classification possible - train RF
    X = X.drop(['is_top_performer'], axis=1)

    # Train video-level Random Forest
    rf_video = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_video.fit(X, y)

# Extract feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_video.feature_importances_
}).sort_values('importance', ascending=False)

# Generate predictions
predictions = rf_video.predict_proba(X)[:, 1]  # Probability of top performer

# Calculate model metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = rf_video.predict(X)
metrics = {
    'accuracy': accuracy_score(y, y_pred),
    'precision': precision_score(y, y_pred),
    'recall': recall_score(y, y_pred),
    'f1_score': f1_score(y, y_pred)
}

# Save model
joblib.dump(rf_video, 'models/rf_video_18-33s.pkl')
```

**Output**: `models/rf_video_18-33s.pkl` (or None if RF skipped)

**Mode Compatibility Note** (added 2025-10-20):
- **'contrastive' mode**: Trains RF + K-Means (binary labels: top 80% vs bottom 20%)
- **'top' mode**: Trains K-Means only (single class: all videos are winners)
- RF requires 2+ classes for binary classification - impossible in 'top' mode
- See TI Document MLModelTrainingCHILDTI.md Section 11.5 Change #G001 for technical details

**What It Captures** (when RF trained):
- **Sequential patterns**: "Energy builds from hook → middle → closing predicts virality"
- **Consistency patterns**: "Hook topic matches middle topic increases viral rate by 35%"
- **Contrast effects**: "Large energy gap between middle avg and closing peak predicts virality"
- **Weak link detection**: "Videos with strong hooks and middles but weak closings still fail"

---

### 5.2: Window-Level Random Forest Training (Within-Window Validation)

**Purpose**: Validate which features matter within each specific window type (hook, middle_1, ..., closing)

**Bucket 18-33s Example** (6 window types):

```python
# Check if RF training is possible (same logic as video-level - added 2025-10-20)
can_train_rf = len(y.unique()) >= 2  # From video-level check above

if not can_train_rf:
    print("Skipping all window-level RF models (single class)")
else:
    # For each window type in bucket
    for window_type in ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']:
        # Load window-specific transformed data
        X = pd.read_csv(f'ml_analysis/{window_type}_rf_transformed.csv')  # (100 videos, 22 features)
        y = X['is_top_performer']  # Binary labels
        X = X.drop(['is_top_performer'], axis=1)

        # Train window-level Random Forest
        rf_window = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_window.fit(X, y)

    # Extract feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_window.feature_importances_
    }).sort_values('importance', ascending=False)

    # Save model
    joblib.dump(rf_window, f'models/rf_{window_type}_18-33s.pkl')

# Total: 6 window-level RF models for bucket 18-33s
```

**Outputs** (for bucket 18-33s):
- `models/rf_hook_18-33s.pkl` (or none if RF skipped)
- `models/rf_middle_1_18-33s.pkl` (or none if RF skipped)
- `models/rf_middle_2_18-33s.pkl` (or none if RF skipped)
- `models/rf_middle_3_18-33s.pkl` (or none if RF skipped)
- `models/rf_middle_4_18-33s.pkl` (or none if RF skipped)
- `models/rf_closing_18-33s.pkl` (or none if RF skipped)

**Note** (added 2025-10-20): Window-level RF training follows same mode compatibility rules as video-level RF (see Section 5.1 Mode Compatibility Note).

**What It Captures** (when RF trained):
- Which features define a "strong hook" vs "weak hook"
- Which features define a "strong middle" vs "weak middle"
- Which features define a "strong closing" vs "weak closing"
- Direct validation for K-Means cluster defining features

---

### 5.3: Window-Level K-Means Training (Creative Strategies)

**Purpose**: Discover creative patterns per video section with interpretable centroids

**Key Principle**: Window-level clustering (21 features) produces interpretable centroids that can be sent directly to LLM

**Bucket 18-33s Example** (6 window types):

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# For each window type in bucket
for window_type in ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']:
    # Load window-specific K-Means transformed data
    X = pd.read_csv(f'ml_analysis/{window_type}_km_transformed.csv')  # (100 videos, ~30 features)

    # Fit scalers per feature (save for inference)
    scalers = {}
    for feature in X.columns:
        scaler = MinMaxScaler()
        scalers[feature] = scaler.fit(X[[feature]])

    # Apply scaling
    X_scaled = pd.DataFrame({
        col: scalers[col].transform(X[[col]]).flatten()
        for col in X.columns
    })

    # Train K-Means (3 clusters per window)
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)

    # Get cluster assignments and centroids
    cluster_assignments = kmeans.labels_
    cluster_centroids = kmeans.cluster_centers_  # Shape: (3 clusters, ~30 features)

    # Calculate metrics
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(X_scaled, cluster_assignments)

    # Save models
    joblib.dump(kmeans, f'models/{window_type}_kmeans_18-33s.pkl')
    joblib.dump(scalers, f'models/{window_type}_scalers_18-33s.pkl')

# Total: 6 K-Means models + 6 scaler sets for bucket 18-33s
```

**Outputs** (for bucket 18-33s):
- `models/hook_kmeans_18-33s.pkl`
- `models/middle_1_kmeans_18-33s.pkl`
- `models/middle_2_kmeans_18-33s.pkl`
- `models/middle_3_kmeans_18-33s.pkl`
- `models/middle_4_kmeans_18-33s.pkl`
- `models/closing_kmeans_18-33s.pkl`
- `models/hook_scalers_18-33s.pkl` (and 5 more scaler files)

**What It Captures**:
- **3 distinct hook strategies** that all lead to viral success
- **3 distinct middle strategies** for sustained engagement
- **3 distinct closing strategies** for CTAs and completion

**Centroid Interpretability**:
- Centroids: 21 features (not 150!) = **LLM-friendly**
- Example: "3 clusters × 21 features = 63 numbers per window" (manageable for LLM)
- No complex pre-processing needed - all features can be sent to LLM

---

### 5.4: Model Count Per Bucket

**Window Count by Bucket** (determines model count):

| Bucket | Windows | Video-Level RF | Window-Level RF | Window-Level K-Means | **Total Models** |
|--------|---------|----------------|-----------------|----------------------|------------------|
| 0-3s | 1 (hook only) | 1 | 1 | 1 | **3** |
| 3-9s | 2 (hook, closing) | 1 | 2 | 2 | **5** |
| 9-13s | 3 (hook, middle_agg, closing) | 1 | 3 | 3 | **7** |
| 13-18s | 3 (hook, middle_agg, closing) | 1 | 3 | 3 | **7** |
| 18-33s | 6 (hook, middle_1-4, closing) | 1 | 6 | 6 | **13** |
| 33-60s | 7 (hook, middle_1-5, closing) | 1 | 7 | 7 | **15** |
| 60-90s | 7 | 1 | 7 | 7 | **15** |
| 90-120s | 7 | 1 | 7 | 7 | **15** |
| **TOTAL (All 8 buckets)** | **36 windows** | **8** | **36** | **36** | **80 models** |

**Note**: These are theoretical maximums if all 8 buckets were trained. In practice, Stage 1 selects only the **top 3 winning buckets** per analysis, resulting in **~33-45 actual models trained** (15-21 per model type).

**Why Dual RF + K-Means Architecture?**:
- Provides **complete pattern coverage** with no blind spots:
  - Cross-window patterns (video-level RF)
  - Within-window patterns (window-level RF)
  - Creative strategies (window-level K-Means)

**Typical Model Count (3-Bucket Analysis)**:
- 3 video-level RF
- 15-21 window-level RF (depends on which 3 buckets selected)
- 15-21 window-level K-Means
- **Total: ~33-45 models per analysis**

---

### 5.5: Complete File Architecture (Bucket 18-33s Example)

```
bucket_18-33s/
├── ml_analysis/
│   ├── aggregated_features.csv              # Stage 3 output
│   ├── rf_transformed.csv                   # Stage 4 output (video-level RF)
│   ├── hook_rf_transformed.csv              # Stage 4 output (window-level RF)
│   ├── middle_1_rf_transformed.csv          # Stage 4 output
│   ├── middle_2_rf_transformed.csv          # Stage 4 output
│   ├── middle_3_rf_transformed.csv          # Stage 4 output
│   ├── middle_4_rf_transformed.csv          # Stage 4 output
│   ├── closing_rf_transformed.csv           # Stage 4 output
│   ├── hook_km_transformed.csv              # Stage 4 output (window-level K-Means)
│   ├── middle_1_km_transformed.csv          # Stage 4 output
│   ├── middle_2_km_transformed.csv          # Stage 4 output
│   ├── middle_3_km_transformed.csv          # Stage 4 output
│   ├── middle_4_km_transformed.csv          # Stage 4 output
│   └── closing_km_transformed.csv           # Stage 4 output
│
└── models/
    ├── rf_video_18-33s.pkl                  # Stage 5 output (video-level RF)
    │
    ├── rf_hook_18-33s.pkl                   # Stage 5 output (window-level RF)
    ├── rf_middle_1_18-33s.pkl               # Stage 5 output
    ├── rf_middle_2_18-33s.pkl               # Stage 5 output
    ├── rf_middle_3_18-33s.pkl               # Stage 5 output
    ├── rf_middle_4_18-33s.pkl               # Stage 5 output
    ├── rf_closing_18-33s.pkl                # Stage 5 output
    │
    ├── hook_kmeans_18-33s.pkl               # Stage 5 output (K-Means models)
    ├── middle_1_kmeans_18-33s.pkl           # Stage 5 output
    ├── middle_2_kmeans_18-33s.pkl           # Stage 5 output
    ├── middle_3_kmeans_18-33s.pkl           # Stage 5 output
    ├── middle_4_kmeans_18-33s.pkl           # Stage 5 output
    ├── closing_kmeans_18-33s.pkl            # Stage 5 output
    │
    ├── hook_scalers_18-33s.pkl              # Stage 5 output (K-Means scalers)
    ├── middle_1_scalers_18-33s.pkl          # Stage 5 output
    ├── middle_2_scalers_18-33s.pkl          # Stage 5 output
    ├── middle_3_scalers_18-33s.pkl          # Stage 5 output
    ├── middle_4_scalers_18-33s.pkl          # Stage 5 output
    ├── closing_scalers_18-33s.pkl           # Stage 5 output
    │
    └── model_metrics.json                   # Stage 5 output (performance metrics)
```

**Total Files for Bucket 18-33s**: 13 models + 6 scaler files + 1 metrics file = **20 files**

---

### 5.6: Model Metrics Summary

```json
{
  "bucket": "18-33s",
  "total_videos": 100,

  "video_level_rf": {
    "model_type": "random_forest",
    "input_features": 190,
    "accuracy": 0.87,
    "precision": 0.89,
    "recall": 0.84,
    "f1_score": 0.86,
    "top_feature": "hook_eye_contact_rate",
    "top_feature_importance": 0.22,
    "purpose": "Cross-window pattern detection"
  },

  "window_level_rf": {
    "hook": {
      "model_type": "random_forest",
      "input_features": 21,
      "accuracy": 0.82,
      "precision": 0.85,
      "recall": 0.78,
      "top_feature": "eye_contact_rate",
      "top_feature_importance": 0.35
    },
    "middle_1": {...},
    "middle_2": {...},
    "middle_3": {...},
    "middle_4": {...},
    "closing": {...}
  },

  "window_level_kmeans": {
    "hook": {
      "model_type": "kmeans",
      "input_features": 30,
      "n_clusters": 3,
      "inertia": 12.5,
      "silhouette_score": 0.68,
      "cluster_sizes": [35, 42, 23]
    },
    "middle_1": {...},
    "middle_2": {...},
    "middle_3": {...},
    "middle_4": {...},
    "closing": {...}
  }
}
```

---

### 5.7: Architectural Summary

**Why Dual RF + Window-Level K-Means?**:

| Benefit | Video-Level RF | Window-Level RF | Window-Level K-Means |
|---------|----------------|-----------------|----------------------|
| **Cross-window patterns** | ✅ Captures | ❌ Misses | ❌ Misses |
| **Within-window validation** | ❌ Mixed signal | ✅ Perfect alignment | — |
| **K-Means validation** | ⚠️ Indirect | ✅ Direct (same granularity) | — |
| **Temporal progressions** | ✅ Quantified | ❌ Not visible | ❌ Not visible |
| **Feature importance clarity** | ⚠️ Fragmented | ✅ Per-window clarity | — |
| **Creative strategies** | — | — | ✅ Interpretable centroids (21 features) |
| **LLM context size** | Large (190 features) | Small (21 features) | **Optimal (21 features)** |

**Complete Pattern Coverage**: All three model types work together to ensure no blind spots in viral video analysis.

**Child Documents**:
- Kmeans.md (K-Means specific design, scaler fitting details)
- KMValidation.md (validation approach)
- KmeansClusteringStage6.md (dual RF + window-level K-Means architecture rationale)

**Future TI Document**:
- MLModelTrainingTI.md (training code, hyperparameter tuning, cross-validation)

**Related Future Features**:
- Phase 2: ML Model Validation Framework (Stage 5 improvement)

---

## Stage 6: ML Analysis Generation

**Status**: ✅ **COMPLETE** (2025-10-24) - Production-ready after bug fixes

**Recent Updates** (2025-10-24):
- ✅ Bug #1 Resolved: Boolean features TypeError fixed via Stage 4 encoding
- ✅ Bug #2 Resolved: video_count scoping issue fixed
- ✅ Validation: 35/35 JSON files generated successfully across 3 test buckets
- ✅ Quality: 83.3% distribution coverage (exceeds 60% threshold)

**Purpose**: Generate ML analysis JSONs for LLM consumption (13 JSON files per bucket for dual RF + window-level K-Means architecture)

**Architectural Decision**: This stage generates **13 JSON files per bucket**:
1. **1 Video-Level RF JSON** (~30KB) - Cross-window patterns
2. **6 Window-Level RF JSONs** (~5KB each) - Within-window feature importance
3. **6 Window-Level K-Means JSONs** (~5KB each) - Cluster centroids per window

**Total per bucket**: ~95KB across 13 files

**Critical Dependency**: Stage 6 expects all features from Stage 4 to be numeric (int64/float64). Boolean features (has_captions) are encoded to int64 [0, 1] in Stage 4 before reaching Stage 6.

**Input**:
- Trained models (90 models total from Stage 5)
- Transformed features (from Stage 4)
- `ml_analysis/aggregated_features.csv` (raw features for distribution analysis)

**Process**:

### 6.1: Video-Level Random Forest Analysis JSON
```python
# Extract feature importance and video-level predictions
# Load data for distribution analysis
top_videos = aggregated_features[aggregated_features['is_top_performer'] == 1]
bottom_videos = aggregated_features[aggregated_features['is_top_performer'] == 0]

rf_analysis = {
    "analysis_type": "random_forest",
    "bucket": bucket,
    "hashtag": hashtag,
    "video_count": N,
    "input_features": 178,  # Video-Level RF: 129 temporal + 18 derived + 1 target (from Stage 4)

    "feature_importance": [
        {
            "feature": "hook_eye_contact_rate",
            "importance": 0.22,
            "top_performer_avg": 0.88,
            "bottom_performer_avg": 0.45,
            "gap": 0.43,

            # NEW: Distribution percentages for actionable insights
            "distribution": {
                "thresholds": {
                    "high": 0.6,    # Determined by 66th percentile or domain knowledge
                    "low": 0.4      # Determined by 33rd percentile
                },
                "top_performers": {
                    "high_percentage": 0.70,    # 70% of top have >= 0.6
                    "medium_percentage": 0.25,  # 25% have 0.4-0.6
                    "low_percentage": 0.05      # 5% have < 0.4
                },
                "bottom_performers": {
                    "high_percentage": 0.05,    # 5% of bottom have >= 0.6
                    "medium_percentage": 0.15,  # 15% have 0.4-0.6
                    "low_percentage": 0.80      # 80% have < 0.4
                }
            }
        },
        # ... top 10 features (each with distribution data)
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

**Distribution Computation Logic**:
```python
def compute_feature_distribution(feature_name, top_videos, bottom_videos):
    """
    Compute distribution percentages for top and bottom performers.

    Returns thresholds and percentage breakdowns (high/medium/low).
    """
    # Combine all videos to determine global thresholds
    all_values = pd.concat([top_videos[feature_name], bottom_videos[feature_name]])

    # Determine thresholds (66th and 33rd percentile)
    threshold_high = all_values.quantile(0.66)
    threshold_low = all_values.quantile(0.33)

    # Compute percentages for top performers
    top_high = len(top_videos[top_videos[feature_name] >= threshold_high]) / len(top_videos)
    top_medium = len(top_videos[(top_videos[feature_name] >= threshold_low) &
                                 (top_videos[feature_name] < threshold_high)]) / len(top_videos)
    top_low = len(top_videos[top_videos[feature_name] < threshold_low]) / len(top_videos)

    # Compute percentages for bottom performers
    bottom_high = len(bottom_videos[bottom_videos[feature_name] >= threshold_high]) / len(bottom_videos)
    bottom_medium = len(bottom_videos[(bottom_videos[feature_name] >= threshold_low) &
                                       (bottom_videos[feature_name] < threshold_high)]) / len(bottom_videos)
    bottom_low = len(bottom_videos[bottom_videos[feature_name] < threshold_low]) / len(bottom_videos)

    return {
        "thresholds": {"high": threshold_high, "low": threshold_low},
        "top_performers": {
            "high_percentage": top_high,
            "medium_percentage": top_medium,
            "low_percentage": top_low
        },
        "bottom_performers": {
            "high_percentage": bottom_high,
            "medium_percentage": bottom_medium,
            "low_percentage": bottom_low
        }
    }

# Apply to top 10 features
for feature_data in feature_importance_list[:10]:
    feature_name = feature_data['feature']
    feature_data['distribution'] = compute_feature_distribution(
        feature_name, top_videos, bottom_videos
    )
```

**Output**: `ml_analysis/rf_video_analysis.json` (~30KB)

**What It Contains**: Cross-window feature importance with pattern type labels (cross_window vs single_window)

---

### 6.2: Window-Level Random Forest Analysis JSONs

**Purpose**: Generate per-window feature importance for direct validation of K-Means clusters

**⚠️ CRITICAL: Bucket-Specific Window Configuration**

Different buckets have different window structures. Implementation MUST use bucket-aware iteration:

```python
# IMPLEMENTATION: Import from shared config (single source of truth)
from config.bucket_definitions import BUCKET_WINDOWS

# BUCKET_WINDOWS contains bucket-specific window configurations:
# {
#     '0-3s': ['hook'],  # Only hook (no closing - video too short)
#     '3-9s': ['hook', 'closing'],
#     '9-13s': ['hook', 'middle_aggregate', 'closing'],  # Aggregated middle (not middle_1/2/3)
#     '13-18s': ['hook', 'middle_aggregate', 'closing'],  # Aggregated middle (not middle_1/2/3)
#     '18-33s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing'],
#     '33-60s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
#     '60-90s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
#     '90-120s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
# }
```

**Note**: Stage 4 (FeatureTransformationCHILD.md Section 4.2) also imports from this shared config.

**Bucket 18-33s Example** (6 window types):

```python
# CORRECT: Bucket-aware iteration (works for ALL 8 bucket types)
bucket = "18-33s"  # Example bucket
for window_type in BUCKET_WINDOWS[bucket]:
    # Load trained window-level RF model
    rf_window = joblib.load(f'models/rf_{window_type}_18-33s.pkl')

    # Load window-level transformed data
    X = pd.read_csv(f'ml_analysis/{window_type}_rf_transformed.csv')
    y = X['is_top_performer']
    X = X.drop(['is_top_performer'], axis=1)

    # Extract feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_window.feature_importances_
    }).sort_values('importance', ascending=False)

    # Calculate top/bottom performer averages
    top_videos = X[y == 1]
    bottom_videos = X[y == 0]

    # Build analysis JSON
    window_rf_analysis = {
        "model_type": "window_level_rf",
        "window_type": window_type,
        "bucket": "18-33s",
        "total_videos": len(X),
        "input_features": len(X.columns),
        "model_performance": {
            "accuracy": accuracy_score(y, rf_window.predict(X)),
            "precision": precision_score(y, rf_window.predict(X)),
            "recall": recall_score(y, rf_window.predict(X))
        },
        "feature_importance": [
            {
                "feature": row['feature'],
                "importance": row['importance'],
                "top_performer_avg": top_videos[row['feature']].mean(),
                "bottom_performer_avg": bottom_videos[row['feature']].mean(),
                "gap": top_videos[row['feature']].mean() - bottom_videos[row['feature']].mean(),
                "rank": idx + 1
            }
            for idx, row in feature_importance.head(10).iterrows()
        ]
    }

    # Save per-window RF analysis
    with open(f'ml_analysis/{window_type}_rf_analysis.json', 'w') as f:
        json.dump(window_rf_analysis, f, indent=2)
```

**Outputs** (for bucket 18-33s):
- `ml_analysis/hook_rf_analysis.json` (~5KB)
- `ml_analysis/middle_1_rf_analysis.json` (~5KB)
- `ml_analysis/middle_2_rf_analysis.json` (~5KB)
- `ml_analysis/middle_3_rf_analysis.json` (~5KB)
- `ml_analysis/middle_4_rf_analysis.json` (~5KB)
- `ml_analysis/closing_rf_analysis.json` (~5KB)

**What It Contains**: Top 10 features per window with importance scores, top/bottom averages, and gaps

---

### 6.3: Window-Level K-Means Analysis JSONs

**Purpose**: Generate cluster centroids and assignments per window (21 features = LLM-friendly)

**Bucket 18-33s Example** (6 window types):

```python
# CORRECT: Bucket-aware iteration (uses BUCKET_WINDOWS config from Section 6.2)
bucket = "18-33s"  # Example bucket
for window_type in BUCKET_WINDOWS[bucket]:
    # Load trained K-Means model and scalers
    kmeans = joblib.load(f'models/{window_type}_kmeans_18-33s.pkl')
    scalers = joblib.load(f'models/{window_type}_scalers_18-33s.pkl')

    # Load window-specific transformed data
    X = pd.read_csv(f'ml_analysis/{window_type}_km_transformed.csv')

    # Apply scaling
    X_scaled = pd.DataFrame({
        col: scalers[col].transform(X[[col]]).flatten()
        for col in X.columns
    })

    # Get cluster assignments
    labels = kmeans.predict(X_scaled)
    centroids = kmeans.cluster_centers_  # Shape: (3 clusters, ~30 features)

    # Calculate distances to centroids
    from sklearn.metrics import euclidean_distances
    distances = euclidean_distances(X_scaled, centroids)

    # Build analysis JSON
    kmeans_analysis = {
        "window_type": window_type,
        "bucket": "18-33s",
        "total_videos": len(X),
        "n_clusters": 3,
        "clusters": []
    }

    for cluster_id in range(3):
        cluster_videos = X[labels == cluster_id]
        cluster_indices = np.where(labels == cluster_id)[0]

        kmeans_analysis['clusters'].append({
            "cluster_id": cluster_id,
            "size": len(cluster_videos),
            "centroid": dict(zip(X.columns, centroids[cluster_id])),  # All 21-30 features
            "videos": [
                {
                    "video_id": f"video_{idx}",
                    "distance_to_centroid": float(distances[idx, cluster_id])
                }
                for idx in cluster_indices
            ]
        })

    # Save per-window K-Means analysis
    with open(f'ml_analysis/{window_type}_kmeans_analysis.json', 'w') as f:
        json.dump(kmeans_analysis, f, indent=2)
```

**Outputs** (for bucket 18-33s):
- `ml_analysis/hook_kmeans_analysis.json` (~5KB)
- `ml_analysis/middle_1_kmeans_analysis.json` (~5KB)
- `ml_analysis/middle_2_kmeans_analysis.json` (~5KB)
- `ml_analysis/middle_3_kmeans_analysis.json` (~5KB)
- `ml_analysis/middle_4_kmeans_analysis.json` (~5KB)
- `ml_analysis/closing_kmeans_analysis.json` (~5KB)

**What It Contains**:
- 3 clusters per window
- 21-30 dimensional centroids (ALL features sent to LLM - no summarization needed!)
- Video assignments with distances to centroids

**Centroid Interpretability**:
- LLM receives: 3 clusters × 21 features = **63 numbers per window** (manageable!)
- No complex pre-processing - LLM can identify defining features directly
- Example: "Cluster 0: high eye_contact_rate (0.87), low word_count (14.2)"

---

### 6.3.1: Special Case - middle_aggregate Window (Buckets 9-13s, 13-18s)

For short-duration buckets (9-13s, 13-18s), middle segments are aggregated into a single `middle_aggregate` window. This window receives the same JSON structure as other windows:

**Example Output**: `ml_analysis/middle_aggregate_rf_analysis.json`

```json
{
  "model_type": "window_level_rf",
  "window_type": "middle_aggregate",
  "bucket": "9-13s",
  "total_videos": 100,
  "input_features": 21,
  "feature_importance": [
    {
      "feature": "scene_count",
      "importance": 0.22,
      "top_performer_avg": 5.2,
      "bottom_performer_avg": 3.1,
      "gap": 2.1,
      "rank": 1
    },
    ...
  ],
  "note": "Aggregated from 3 short middle segments (1-4s each) to ensure reliable feature measurements"
}
```

**Rationale**: Individual 1-4s middle segments produce unreliable measurements for scene_count, speech_coverage, and word_count. Aggregation creates 4.5-9.3s windows where all 21 features are reliable. See FeatureAggregationCHILD.md Decision 7 (lines 1137-1165) for full justification.

**Implementation Note**: The `middle_aggregate` window is treated identically to other windows in Stage 6 - same RF/K-Means analysis, same JSON schema, same LLM processing in Stage 7.

---

### 6.4: Complete Outputs Summary

**For Bucket 18-33s** (6 windows):

| JSON Type | Count | Size Each | Total Size | LLM Consumer |
|-----------|-------|-----------|------------|--------------|
| **Video-Level RF** | 1 | ~30KB | ~30KB | Stage 7 Phase 2 (cross-window patterns) |
| **Window-Level RF** | 6 | ~5KB | ~30KB | Stage 7 Phase 1 (feature validation) |
| **Window-Level K-Means** | 6 | ~5KB | ~30KB | Stage 7 Phase 1 (cluster insights) |
| **TOTAL** | **13 files** | — | **~95KB** | — |

**File Structure** (bucket 18-33s/ml_analysis/):
```
bucket_18-33s/ml_analysis/
├── rf_video_analysis.json               # Video-level RF (cross-window patterns)
├── hook_rf_analysis.json                # Window-level RF
├── middle_1_rf_analysis.json            # Window-level RF
├── middle_2_rf_analysis.json            # Window-level RF
├── middle_3_rf_analysis.json            # Window-level RF
├── middle_4_rf_analysis.json            # Window-level RF
├── closing_rf_analysis.json             # Window-level RF
├── hook_kmeans_analysis.json            # Window-level K-Means (3 clusters, 21D centroids)
├── middle_1_kmeans_analysis.json        # Window-level K-Means
├── middle_2_kmeans_analysis.json        # Window-level K-Means
├── middle_3_kmeans_analysis.json        # Window-level K-Means
├── middle_4_kmeans_analysis.json        # Window-level K-Means
└── closing_kmeans_analysis.json         # Window-level K-Means
```

---

### 6.5: Why 13 JSON Files?

**Advantages**:
- **LLM-Friendly Context**: Window-level K-Means centroids (21 features) vs video-level (150 features) = 7x smaller
- **Focused Analysis**: Each window type analyzed independently for clarity
- **Direct Validation**: Window-level RF validates K-Means cluster defining features (same granularity)
- **Actionable Insights**: Per-section strategies ("Use this hook type") vs abstract patterns
- **Complete Pattern Coverage**: Video-level RF + Window-level RF + Window-level K-Means = no blind spots

**Trade-offs**:
- More files to manage (13 vs 2)
- Slightly more complex Stage 7 LLM integration
- But: **Better insights, higher actionability, complete pattern coverage**

**Child Documents**:
- ML_LLMData.md (JSON format strategy, schema specifications)
- ML_LLMDataTI.md (JSON generation technical specs)
- KmeansClusteringStage6.md (dual RF + window-level K-Means architecture rationale)

**Future TI Document**:
- MLAnalysisGenerationTI.md (JSON generation code, schema validation)

**Related Future Features**:
- Phase 2: LLM Data Strategy (Stage 6 core feature)

---

## Stage 7: LLM Analysis - Hybrid Two-Phase Approach

**Purpose**: Generate creative insights from K-Means clustering results, validated by dual Random Forest analysis

**Input** (from Stage 6):
- **Video-Level RF** (1 JSON): `rf_video_analysis.json` (~30KB) - Cross-window feature importance
- **Window-Level RF** (6 JSONs): `{window_type}_rf_analysis.json` (~5KB each) - Per-window feature importance
- **Window-Level K-Means** (6 JSONs): `{window_type}_kmeans_analysis.json` (~5KB each) - Per-window cluster centroids
- **Total**: 13 JSON files (~95KB) per bucket

**Architecture**: Two-phase hybrid approach minimizes hallucination risk with small, focused contexts in Phase 1, then combines insights in Phase 2. Random Forest provides both within-window validation (Phase 1) and cross-window pattern detection (Phase 2).

**Process**:

### 7.1: Phase 1 - Per-Window Analysis (Parallel Execution)

**Purpose**: Analyze each window type independently with window-level RF validation

**Execution**: 6-7 parallel API calls (one per window: hook, middle_1-4, closing)

**Input per Window** (Example: Hook):
1. K-Means: `hook_kmeans_analysis.json` - 3 clusters × 21 features = 63 numbers
2. Window-Level RF: `hook_rf_analysis.json` - Top 10 features × 5 metrics = 50 numbers
3. **Combined Context**: 113 numbers total (small, focused)

**LLM Prompt Template** (Phase 1):
```python
def analyze_window(window_type: str, kmeans_data: dict, rf_data: dict, bucket: str, hashtag: str) -> dict:
    """
    Analyze one window type's K-Means clusters with RF validation.

    Returns: Phase 1 analysis JSON
    """
    prompt = f"""
You are analyzing {window_type} segments from 100 viral videos in the {bucket} duration bucket for #{hashtag}.

Context:
- These are all TOP-PERFORMING videos (high engagement)
- You are identifying DIFFERENT STRATEGIES that all lead to success
- Focus on what makes each cluster DISTINCT from the others

K-Means clustering has identified 3 distinct {window_type} patterns:

CLUSTER 0 ({kmeans_data['clusters'][0]['size']} videos):
{format_centroid_features(kmeans_data['clusters'][0]['centroid'])}

CLUSTER 1 ({kmeans_data['clusters'][1]['size']} videos):
{format_centroid_features(kmeans_data['clusters'][1]['centroid'])}

CLUSTER 2 ({kmeans_data['clusters'][2]['size']} videos):
{format_centroid_features(kmeans_data['clusters'][2]['centroid'])}

Random Forest Feature Importance ({window_type}-specific predictive power):

The features that BEST PREDICT viral success within {window_type} segments:

1. {rf_data['feature_importance'][0]['feature']}
   - RF Importance: {rf_data['feature_importance'][0]['importance']} (rank #{rf_data['feature_importance'][0]['rank']})
   - Top performers: avg {rf_data['feature_importance'][0]['top_performer_avg']} ({rf_data['feature_importance'][0]['distribution']['top_performers']['high_percentage']*100:.0f}% have ≥{rf_data['feature_importance'][0]['distribution']['thresholds']['high']})
   - Bottom performers: avg {rf_data['feature_importance'][0]['bottom_performer_avg']} (only {rf_data['feature_importance'][0]['distribution']['bottom_performers']['high_percentage']*100:.0f}% reach {rf_data['feature_importance'][0]['distribution']['thresholds']['high']})
   - Gap: {rf_data['feature_importance'][0]['gap']}

2. {rf_data['feature_importance'][1]['feature']}
   - RF Importance: {rf_data['feature_importance'][1]['importance']} (rank #{rf_data['feature_importance'][1]['rank']})
   - Top performers: avg {rf_data['feature_importance'][1]['top_performer_avg']} ({rf_data['feature_importance'][1]['distribution']['top_performers']['high_percentage']*100:.0f}% have ≥{rf_data['feature_importance'][1]['distribution']['thresholds']['high']})
   - Bottom performers: avg {rf_data['feature_importance'][1]['bottom_performer_avg']} (only {rf_data['feature_importance'][1]['distribution']['bottom_performers']['high_percentage']*100:.0f}% reach {rf_data['feature_importance'][1]['distribution']['thresholds']['high']})
   - Gap: {rf_data['feature_importance'][1]['gap']}

... (top 10 features, each formatted with distribution data)

Your task:
1. **Name each cluster** with a memorable, creator-friendly label (e.g., "The Direct Eye Contact Hook")
2. **Identify 3-5 defining features** per cluster that differentiate it from the others
   - PRIORITIZE features with high RF importance scores (these are most predictive of viral success)
   - Emphasize features with large top/bottom gaps (biggest performance differentiators)
   - **NOTICE distribution patterns**: If top performers show bimodal patterns (e.g., "40% high, 60% low"), this indicates MULTIPLE successful strategies for this feature
3. **Describe the strategy** each cluster represents (what creative approach does it use?)
4. **Generate actionable recommendations** - what should creators DO to replicate this pattern?
   - Focus on high-importance RF features first
   - Include target values based on top_performer_avg from RF data
   - **For bimodal features**: Present both strategies as viable options (e.g., "Use either brief hooks (10-15 words) OR dense hooks (80-90 words) - both work")

Output format: JSON
{{
  "window_type": "{window_type}",
  "clusters": [
    {{
      "cluster_id": 0,
      "name": "Creative strategy name",
      "defining_features": [
        "feature_name: value (interpretation)"
      ],
      "rf_validation": {{
        "top_predictive_features_in_cluster": [...],
        "insight": "How this cluster uses RF-validated features"
      }},
      "strategy_description": "What makes this cluster unique",
      "creator_recommendations": [
        "PRIORITY: Specific actionable step with RF targets",
        "Specific actionable step 2",
        "Specific actionable step 3"
      ]
    }},
    // ... clusters 1 and 2
  ]
}}

Important:
- Be specific and concrete (not generic advice)
- Focus on DIFFERENCES between clusters (not universal best practices)
- Recommendations should be replicable creative techniques
"""

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        temperature=0.3,  # Lower temperature for consistency
        messages=[{"role": "user", "content": prompt}]
    )

    analysis = json.loads(response.content[0].text)
    analysis['analysis_metadata'] = {
        'llm_model': 'claude-sonnet-4',
        'timestamp': datetime.now().isoformat(),
        'api_latency_seconds': response.usage.total_time_seconds
    }

    return analysis


def run_phase1_parallel(bucket: str, hashtag: str, window_types: list) -> dict:
    """
    Run Phase 1 analysis for all windows in parallel.

    Args:
        bucket: '18-33s'
        hashtag: '#nutrition'
        window_types: ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

    Returns: {window_type: analysis_json} for all windows
    """
    window_analyses = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for window_type in window_types:
            # Load K-Means and RF data for this window
            kmeans_data = load_json(f'ml_analysis/{window_type}_kmeans_analysis.json')
            rf_data = load_json(f'ml_analysis/{window_type}_rf_analysis.json')

            # Submit analysis task
            future = executor.submit(
                analyze_window,
                window_type=window_type,
                kmeans_data=kmeans_data,
                rf_data=rf_data,
                bucket=bucket,
                hashtag=hashtag
            )
            futures[window_type] = future

        # Collect results
        for window_type, future in futures.items():
            try:
                analysis = future.result(timeout=60)
                window_analyses[window_type] = analysis
                save_json(f'ml_analysis/llm/{window_type}_analysis.json', analysis)
            except Exception as e:
                logging.error(f"Phase 1 failed for {window_type}: {e}")

    return window_analyses
```

**Phase 1 Output Example** (`hook_analysis.json`):
```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "hashtag": "#nutrition",
  "total_videos": 100,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "name": "The Direct Eye Contact Hook",
      "defining_features": [
        "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43 - HIGHEST PREDICTOR)",
        "word_count: 14 (RF rank #3, importance 0.18, low count strategy)",
        "energy_level: 0.55 (RF rank #2, importance 0.22, moderate-calm approach)"
      ],
      "rf_validation": {
        "top_predictive_features_in_cluster": [
          "eye_contact_rate: Cluster value 0.87 matches top performer avg 0.88 (RF validated)"
        ],
        "insight": "This cluster leverages the #1 most predictive hook feature at optimal levels."
      },
      "strategy_description": "Creator looks directly at camera with minimal speech, establishing immediate connection through eye contact rather than information density.",
      "creator_recommendations": [
        "PRIORITY: Maintain 85-90% eye contact (RF #1 predictor, importance 0.35, gap 0.43)",
        "Keep opening statement under 15 words (RF #3 predictor)",
        "Target moderate energy 0.55-0.60 (RF #2 predictor)"
      ]
    },
    {
      "cluster_id": 1,
      "size": 42,
      "name": "The Text Overlay Hook",
      "defining_features": [
        "overlay_unique_count: 3.5 (high - multiple text overlays)",
        "eye_contact_rate: 0.28 (low - looking away or at product)",
        "word_count: 48 (very high - talking while showing text)"
      ],
      "strategy_description": "Fast-paced, text-heavy opening with multiple scene cuts.",
      "creator_recommendations": [
        "Add 2-3 text overlays in first 3 seconds",
        "Use dynamic cuts (3-4 scenes in hook)",
        "Speak quickly - aim for 45-50 words in 3 seconds"
      ]
    },
    {
      "cluster_id": 2,
      "size": 23,
      "name": "The Action-Driven Hook",
      "defining_features": [
        "object_count: 4.8 (high - multiple props/products visible)",
        "gesture_count: 7.5 (very high - active hand movements)",
        "energy_level: 0.75 (high - dynamic movement)"
      ],
      "strategy_description": "Single continuous shot with high-energy physical action.",
      "creator_recommendations": [
        "Film in one continuous take - avoid cuts in first 3 seconds",
        "Use 6-8 hand gestures (pointing, grabbing, showing products)",
        "Show 4-5 different objects/products early"
      ]
    }
  ]
}
```

**Phase 1 Outputs** (per bucket):
- `ml_analysis/llm/hook_analysis.json`
- `ml_analysis/llm/middle_1_analysis.json`
- `ml_analysis/llm/middle_2_analysis.json`
- `ml_analysis/llm/middle_3_analysis.json`
- `ml_analysis/llm/middle_4_analysis.json`
- `ml_analysis/llm/closing_analysis.json`

**Phase 1 Execution Time**: ~5-10 seconds wall-clock (all 6 calls run in parallel)

**LLM Calls per Bucket (Phase 1)**: 6 parallel calls

---

### 7.2: Phase 2 - Cross-Window Synthesis (Single Call)

**Purpose**: Synthesize cross-window patterns and identify "Winning Formulas" with video-level RF validation

**Input Preparation**:
1. All Phase 1 window analyses (6 JSONs)
2. Video cluster paths across windows (extracted from K-Means outputs)
3. Video-Level RF cross-window feature importance (`rf_video_analysis.json`)

**Extract Video Cluster Paths**:
```python
def extract_cluster_paths(window_analyses: dict, kmeans_outputs: dict) -> list:
    """
    Extract each video's cluster assignment across all windows.

    Returns:
        [
            {'video_id': 'video_001', 'path': [0, 1, 0, 1, 2, 0],
             'path_str': 'Hook-0 → M1-1 → M2-0 → M3-1 → M4-2 → Closing-0'},
            ...
        ]
    """
    video_paths = []
    for video_id in all_video_ids:
        path = []
        for window_type in window_types:
            cluster_id = get_video_cluster(video_id, window_type, kmeans_outputs)
            path.append(cluster_id)

        path_str = format_path(path, window_types)
        video_paths.append({
            'video_id': video_id,
            'path': path,
            'path_str': path_str
        })

    return video_paths


def analyze_path_frequencies(video_paths: list) -> list:
    """
    Identify most common cluster path combinations.

    Returns top 10 most common paths with frequencies.
    """
    path_counts = Counter([tuple(vp['path']) for vp in video_paths])

    top_paths = []
    for path, count in path_counts.most_common(10):
        top_paths.append({
            'path': list(path),
            'frequency': count,
            'percentage': round(count / len(video_paths) * 100, 1),
            'path_str': format_path(path, window_types)
        })

    return top_paths
```

**Video-Level RF Cross-Window Patterns** (Example features):
```json
{
  "feature_importance": [
    {
      "feature": "hook_eye_contact_rate",
      "importance": 0.22,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "rank": 1
    },
    {
      "feature": "hook_to_middle_energy_delta",
      "importance": 0.12,
      "interpretation": "Energy change from hook to middle average",
      "top_performer_avg": 0.15,
      "bottom_performer_avg": -0.08,
      "gap": 0.23,
      "rank": 4,
      "pattern_type": "cross_window"
    },
    {
      "feature": "middle_to_closing_contrast",
      "importance": 0.10,
      "interpretation": "Energy gap between middle avg and closing peak",
      "top_performer_avg": 0.28,
      "bottom_performer_avg": 0.05,
      "gap": 0.23,
      "rank": 5,
      "pattern_type": "cross_window"
    },
    {
      "feature": "eye_contact_consistency",
      "importance": 0.08,
      "interpretation": "Std deviation of eye contact across all windows",
      "top_performer_avg": 0.12,
      "bottom_performer_avg": 0.35,
      "gap": 0.23,
      "rank": 6,
      "pattern_type": "cross_window"
    }
  ]
}
```

**LLM Prompt Template** (Phase 2):
```python
def run_phase2_synthesis(
    window_analyses: dict,
    kmeans_outputs: dict,
    rf_video_data: dict,
    bucket: str,
    hashtag: str
) -> dict:
    """
    Synthesize cross-window patterns from Phase 1 analyses.

    Returns: Phase 2 synthesis JSON (winning formulas)
    """
    # Extract video cluster paths
    video_paths = extract_cluster_paths(window_analyses, kmeans_outputs)
    top_paths = analyze_path_frequencies(video_paths)

    prompt = f"""
You are synthesizing creative insights for viral videos in the {bucket} duration bucket for #{hashtag}.

You have analyzed 100 viral videos across 6 temporal windows. Each window has been clustered into 3 distinct strategies.

## Per-Window Cluster Analyses

### Hook Analysis:
{json.dumps(window_analyses['hook'], indent=2)}

### Middle_1 Analysis:
{json.dumps(window_analyses['middle_1'], indent=2)}

### Middle_2 Analysis:
{json.dumps(window_analyses['middle_2'], indent=2)}

### Middle_3 Analysis:
{json.dumps(window_analyses['middle_3'], indent=2)}

### Middle_4 Analysis:
{json.dumps(window_analyses['middle_4'], indent=2)}

### Closing Analysis:
{json.dumps(window_analyses['closing'], indent=2)}

## Most Common Cluster Paths (Video Journey Patterns)

The 10 most common combinations of window strategies:

{format_top_paths(top_paths)}

## Video-Level Random Forest (Cross-Window Pattern Detection)

The features that BEST PREDICT viral success across the ENTIRE VIDEO JOURNEY:

Top Single-Window Features:
{format_single_window_features(rf_video_data)}

Top Cross-Window Features (these only exist at video-level):
{format_cross_window_features(rf_video_data)}

Key Cross-Window Insights from RF:
- Energy progression matters: Building from hook → middle (delta +0.15) predicts virality
- Closing contrast matters: Large energy gap between middle avg and closing peak (0.28) predicts virality
- Consistency matters: Low variance in eye_contact across windows (std 0.12) predicts virality

## Your Task

Generate exactly 3 creative reports using a frequency-based approach with feature-based fallback.

### STEP 1: Filter Paths by 10% Frequency Threshold

**CRITICAL RULE**: Only consider cluster paths with ≥10% frequency (minimum 10 videos out of 100).

**Why 10% Threshold**:
- Ensures formulas are proven patterns, not statistical noise
- 10% = "1 in 10 videos use this pattern" = reliable for creator replication
- Below 10% = too rare, might not replicate, wastes creator time

**Examples**:
- 22 videos (22%) → INCLUDE ✅ (very high confidence)
- 18 videos (18%) → INCLUDE ✅ (high confidence)
- 12 videos (12%) → INCLUDE ✅ (moderate confidence)
- 8 videos (8%) → EXCLUDE ❌ (below threshold - statistical noise)

**Action**: Count how many paths meet ≥10% threshold from the cluster path data provided above.

---

### STEP 2: Determine Report Mix (Path vs Feature-Based)

Based on number of paths above 10% threshold:

**Scenario A**: 3 or more paths ≥10%
- Generate 3 path-based reports (take top 3 by frequency, ordered descending)

**Scenario B**: Exactly 2 paths ≥10%
- Generate 2 path-based reports (for the 2 paths above threshold)
- Generate 1 feature-based report (using top RF features from video-level analysis)

**Scenario C**: Exactly 1 path ≥10%
- Generate 1 path-based report (for the 1 path above threshold)
- Generate 2 feature-based reports (using top RF features)

**Scenario D**: 0 paths ≥10% (high fragmentation)
- Generate 3 feature-based reports (all based on top RF features)
- Log: "High fragmentation detected: No paths meet 10% threshold. Using feature-based approach."

**ALWAYS output exactly 3 reports total** (never 4, never 2).

---

### STEP 3: Generate Path-Based Reports (for paths ≥10%)

For each cluster path above 10% threshold:

1. **Name**: Creative, memorable name (e.g., "The Educator's Arc")
2. **Structure**: Which cluster combination
   - Hook: Cluster name from Phase 1 (e.g., "The Direct Eye Contact Hook")
   - Middle pattern: Progression description
   - Closing: Cluster name from Phase 1
3. **Frequency & Confidence**:
   - frequency: Video count (e.g., 22)
   - percentage: Frequency percentage (e.g., 22.0)
   - confidence_level: Based on percentage:
     - ≥20%: "very_high" (1 in 5 videos - dominant pattern)
     - 15-19.9%: "high" (1 in 6-7 videos - strong pattern)
     - 10-14.9%: "moderate" (1 in 10 videos - proven pattern)
4. **Temporal Progression**: How key features evolve across windows
   - Show actual values per window (hook: 0.55, middle_avg: 0.65, closing: 0.85)
   - Calculate deltas (hook_to_middle_delta, middle_to_closing_contrast)
   - Describe pattern in words
5. **RF Cross-Window Validation**: How formula matches video-level RF patterns
   - Compare formula's deltas to RF top_performer_avg
   - List matches (e.g., "hook_to_middle_energy_delta: 0.16 matches RF avg 0.15")
   - Provide rf_validation_score (e.g., "9/10" if 3/3 patterns match)
6. **Strategy Description**: Overall creative approach
7. **When to Use**: Content types and creator profiles that fit this formula
8. **Step-by-Step Template**: Concrete replication steps
   - Include window-specific actions (Hook: do X, Middle: do Y)
   - Include cross-window targets (Energy delta: +0.16, Contrast: 0.27)
   - Reference RF-validated features

---

### STEP 4: Generate Feature-Based Reports (fallback when needed)

If fewer than 3 paths meet 10% threshold, generate feature-based reports to reach exactly 3 total.

**Feature-Based Report Structure**:
- **No cluster path** (not based on specific path combination)
- Uses top features from video-level RF analysis
- Focus on universal principles applicable to all videos
- Always classified as "moderate" confidence (not frequency-based)

**How to Create Feature-Based Reports**:
1. Select top RF features (choose from video-level RF feature_importance)
2. Group related features (e.g., eye_contact_rate + eye_contact_consistency = "Eye Contact Strategy")
3. Use top_performer_avg as target values
4. Provide actionable recommendations for each feature group

**Example Feature-Based Report**:
{{
  "report_id": 3,
  "type": "feature_based",
  "frequency": null,
  "percentage": null,
  "confidence_level": "moderate",
  "formula_name": "The High Eye Contact Strategy",
  "strategy_description": "Maintain consistent direct eye contact throughout video journey",
  "key_features": [
    "eye_contact_rate: 0.88 (RF rank #1, importance 0.35, gap 0.43)",
    "eye_contact_consistency: 0.12 std dev (RF rank #6, importance 0.08)"
  ],
  "rf_validation": {{
    "insight": "Leverages #1 and #6 most predictive features across entire video"
  }},
  "when_to_use": "Universal strategy applicable when cluster paths are fragmented. Focus on proven principles.",
  "creator_recommendations": [
    "PRIORITY: Maintain 85-90% eye contact throughout video (RF #1 predictor)",
    "Keep eye contact variance low (<0.15 std dev) across all windows",
    "Use direct-to-camera framing in hook and closing windows"
  ]
}}

**Feature-Based Report Categories** (use these groupings):
1. **Eye Contact & Engagement**: eye_contact_rate, eye_contact_consistency
2. **Energy & Pacing**: energy_level, hook_to_middle_energy_delta, middle_to_closing_contrast
3. **Speech & Density**: word_count, speech_coverage, word_density
4. **Visual Variety**: scene_count, object_count, overlay_unique_count

---

### STEP 5: Generate Supplementary Insights (for all creators)

In addition to the 3 creative reports, provide supplementary insights that apply broadly:

**A. Universal Principles** (5-7 insights):
- Extract from video-level RF feature_importance (top 5-7 features)
- Format: "Feature X (top avg vs bottom avg) - applies to Y% of videos"
- Example: "High eye contact rate (88% vs 45% for top vs bottom) - applies to 78% of videos"
- Purpose: Guidance for creators whose style doesn't match specific path formulas

**B. Cross-Window Patterns** (3-5 insights):
- Extract from video-level RF cross-window features
- Format: Percentage-based insights about temporal evolution
- Example: "78% of high-performing videos use 'bookend' eye contact pattern (high in hook/closing, lower in middle)"
- Purpose: Understanding how features evolve across video journey

---

## Output Format: JSON

{{
  "bucket": "{bucket}",
  "hashtag": "{hashtag or None}",
  "total_videos": 100,
  "total_unique_paths": 45,
  "paths_above_threshold": 5,

  "creative_reports": [
    {{
      "report_id": 1,
      "type": "path_based",  // or "feature_based"
      "path": [0, 1, 1, 1, 2, 0],  // Only for path_based (null for feature_based)
      "frequency": 22,  // Only for path_based (null for feature_based)
      "percentage": 22.0,  // Only for path_based (null for feature_based)
      "confidence_level": "very_high",  // very_high, high, or moderate
      "formula_name": "The Educator's Arc",
      "structure": {{  // Only for path_based
        "hook": "The Direct Eye Contact Hook (Cluster 0)",
        "middle_pattern": "Information Dense Middle (Cluster 1 → 1 → 1 → 2)",
        "closing": "High Energy CTA (Cluster 0)"
      }},
      "temporal_progressions": [  // Only for path_based
        {{
          "feature": "energy_level",
          "hook": 0.55,
          "middle_1": 0.60,
          "middle_2": 0.62,
          "middle_3": 0.68,
          "middle_4": 0.75,
          "closing": 0.85,
          "pattern": "Steady build from moderate to high",
          "hook_to_middle_delta": 0.16,
          "middle_to_closing_contrast": 0.27
        }}
      ],
      "rf_cross_window_validation": {{
        "matches_top_patterns": [
          "hook_to_middle_energy_delta: 0.16 (RF top performer avg: 0.15, RF rank #4)",
          "middle_to_closing_contrast: 0.27 (RF top performer avg: 0.28, RF rank #5)"
        ],
        "insight": "This formula exhibits 2 of 3 major cross-window patterns identified by video-level RF.",
        "rf_validation_score": "8/10"
      }},
      "strategy_description": "Start with intimate eye contact to build trust, deliver dense educational content in middle segments, return to direct eye contact for high-energy call-to-action.",
      "when_to_use": "Educational nutrition content, product explanations, how-to videos.",
      "creator_recommendations": [
        "Hook (0-3s): Direct eye contact (0.87), minimal words (14), moderate energy (0.55)",
        "Middle_1 (3-8s): Shift to product view, increase talking speed (50+ words), build energy to 0.60",
        "Middle_2-4 (8-23s): Continue information delivery, steady energy progression",
        "Closing (23-26s): Return to direct eye contact (0.82), peak energy (0.85), clear CTA",
        "CROSS-WINDOW TARGETS (RF validated):",
        "  - Energy delta hook→middle: +0.16 (RF target: +0.15)",
        "  - Energy contrast middle→closing: 0.27 gap (RF target: 0.28)"
      ]
    }},
    // Report 2
    // Report 3
  ],

  "supplementary_insights": {{
    "universal_principles": [
      "High eye contact rate (88% vs 45% for top vs bottom performers) - applies to 78% of videos",
      "Consistent energy maintenance across windows (std dev ≤0.15) - found in 65% of top performers",
      "Clear CTA in closing window - present in 92% of high-performing videos",
      "Text overlays within first 3 seconds - found in 60% of top performers",
      "Energy builds from hook to closing - 65% of videos use this pattern"
    ],
    "cross_window_patterns": [
      "78% of high-performing videos use 'bookend' eye contact pattern (high in hook/closing, lower in middle)",
      "Energy progression: 65% build energy, 12% maintain consistent energy, 23% variable",
      "Closing energy should match or exceed middle average (85% of top performers follow this)",
      "Videos with energy delta >0.3 from hook to closing had 2x engagement"
    ]
  }},

  "path_statistics": {{
    "total_unique_paths": 45,
    "paths_above_threshold": 5,
    "needs_fallback": false
  }},

  "analysis_metadata": {{
    "llm_model": "claude-sonnet-4-20250514",
    "timestamp": "2025-10-16T...",
    "phase": "phase2_synthesis"
  }}
}}

---

## Important Reminders:

1. **Always output exactly 3 creative reports** (never more, never less)
2. **Apply 10% threshold strictly** (8% paths are excluded)
3. **Classify confidence levels accurately**:
   - very_high: ≥20%
   - high: 15-19.9%
   - moderate: 10-14.9%
   - Feature-based reports: always moderate
4. **Use feature-based fallback when needed** (<3 paths above 10%)
5. **Include supplementary_insights** (universal principles + cross-window patterns)
6. **Focus on actionability**: Concrete steps creators can replicate
7. **Validate against RF data**: Cross-window patterns should match video-level RF features
"""

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}]
    )

    synthesis = json.loads(response.content[0].text)
    synthesis['bucket'] = bucket
    synthesis['hashtag'] = hashtag
    synthesis['total_videos'] = len(video_paths)
    synthesis['analysis_metadata'] = {
        'llm_model': 'claude-sonnet-4',
        'timestamp': datetime.now().isoformat(),
        'api_latency_seconds': response.usage.total_time_seconds
    }

    save_json('ml_analysis/llm/winning_formulas.json', synthesis)

    return synthesis
```

**Phase 2 Output Example** (`winning_formulas.json`):
```json
{
  "bucket": "18-33s",
  "hashtag": "#nutrition",
  "total_videos": 100,
  "winning_formulas": [
    {
      "name": "The Educator's Arc",
      "structure": {
        "hook": "The Direct Eye Contact Hook (Cluster 0)",
        "middle_pattern": "Information Dense Middle (Cluster 1 → 1 → 1 → 2)",
        "closing": "High Energy CTA (Cluster 0)"
      },
      "cluster_path": [0, 1, 0, 1, 2, 0],
      "frequency": 18,
      "percentage": 18.0,
      "temporal_progressions": [
        {
          "feature": "energy_level",
          "hook": 0.55,
          "middle_1": 0.60,
          "middle_2": 0.62,
          "middle_3": 0.68,
          "middle_4": 0.75,
          "closing": 0.85,
          "pattern": "Steady build from moderate (0.55) to high (0.85)",
          "hook_to_middle_delta": 0.16,
          "middle_to_closing_contrast": 0.27
        }
      ],
      "rf_cross_window_validation": {
        "matches_top_patterns": [
          "hook_to_middle_energy_delta: 0.16 (matches RF top performer avg 0.15, RF rank #4)",
          "middle_to_closing_contrast: 0.27 (matches RF top performer avg 0.28, RF rank #5)"
        ],
        "insight": "This formula exhibits ALL THREE major cross-window patterns identified by video-level RF.",
        "rf_validation_score": "9/10"
      },
      "strategy_description": "Start with intimate eye contact to build trust, deliver dense educational content in middle segments, return to direct eye contact for high-energy call-to-action.",
      "when_to_use": "Educational nutrition content, product explanations, how-to videos.",
      "step_by_step_template": [
        "Hook (0-3s): Direct eye contact (0.87), minimal words (14), moderate energy (0.55)",
        "Middle_1 (3-8s): Shift to product view, increase talking speed (50+ words), build energy to 0.60",
        "Middle_2-4 (8-23s): Continue information delivery, steady energy progression",
        "Closing (23-26s): Return to direct eye contact (0.82), peak energy (0.85), clear CTA",
        "CROSS-WINDOW TARGETS (RF validated):",
        "  - Energy delta hook→middle: +0.16 (RF target: +0.15)",
        "  - Energy contrast middle→closing: 0.27 gap (RF target: 0.28)"
      ]
    }
  ],
  "cross_window_insights": [
    "78% of high-performing videos use 'bookend' eye contact pattern (high in hook/closing, lower in middle)",
    "Energy builds are common (65% of videos), but 12% succeed with consistent energy",
    "Closing energy should match or exceed middle average (85% of top performers follow this)"
  ]
}
```

**Phase 2 Output**:
- `ml_analysis/llm/winning_formulas.json`

**Phase 2 Execution Time**: ~15-20 seconds

**LLM Calls per Bucket (Phase 2)**: 1 call

---

### 7.3: Complete Stage 7 Pipeline

**Orchestration Code**:
```python
def run_stage7_llm_analysis(bucket: str, hashtag: str) -> dict:
    """
    Complete Stage 7 pipeline: Phase 1 + Phase 2.

    Returns: Complete creative analysis with window insights + winning formulas
    """
    logger.info(f"Starting Stage 7 LLM Analysis for {bucket} / {hashtag}")

    # Determine window types for this bucket
    window_types = get_window_types_for_bucket(bucket)
    # e.g., ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

    # Phase 1: Analyze each window in parallel
    logger.info("Phase 1: Analyzing each window type...")
    start_time = time.time()

    window_analyses = run_phase1_parallel(
        bucket=bucket,
        hashtag=hashtag,
        window_types=window_types
    )

    phase1_time = time.time() - start_time
    logger.info(f"Phase 1 completed in {phase1_time:.1f}s ({len(window_types)} windows in parallel)")

    # Validate Phase 1 outputs
    if len(window_analyses) != len(window_types):
        logger.warning(f"Phase 1 incomplete: {len(window_analyses)}/{len(window_types)} windows analyzed")

    # Phase 2: Synthesize cross-window patterns
    logger.info("Phase 2: Synthesizing winning formulas...")
    start_time = time.time()

    # Load K-Means outputs for cluster path extraction
    kmeans_outputs = load_kmeans_outputs(bucket, window_types)

    # Load video-level RF for cross-window validation
    rf_video_data = load_json('ml_analysis/rf_video_analysis.json')

    synthesis = run_phase2_synthesis(
        window_analyses=window_analyses,
        kmeans_outputs=kmeans_outputs,
        rf_video_data=rf_video_data,
        bucket=bucket,
        hashtag=hashtag
    )

    phase2_time = time.time() - start_time
    logger.info(f"Phase 2 completed in {phase2_time:.1f}s")

    # Combine Phase 1 + Phase 2 into final output
    complete_analysis = {
        'bucket': bucket,
        'hashtag': hashtag,
        'window_analyses': window_analyses,
        'winning_formulas': synthesis['winning_formulas'],
        'cross_window_insights': synthesis['cross_window_insights'],
        'execution_metrics': {
            'phase1_time_seconds': phase1_time,
            'phase2_time_seconds': phase2_time,
            'total_time_seconds': phase1_time + phase2_time,
            'api_calls': len(window_types) + 1
        }
    }

    # Save complete analysis
    save_json(f'ml_analysis/llm/complete_analysis_{bucket}.json', complete_analysis)

    logger.info(f"Stage 7 complete. Total time: {phase1_time + phase2_time:.1f}s")

    return complete_analysis
```

---

### Stage 7 Summary

**Output File Structure** (per bucket):
```
bucket_18-33s/
└── ml_analysis/
    └── llm/
        ├── hook_analysis.json              # Phase 1 (3 clusters, named strategies)
        ├── middle_1_analysis.json          # Phase 1
        ├── middle_2_analysis.json          # Phase 1
        ├── middle_3_analysis.json          # Phase 1
        ├── middle_4_analysis.json          # Phase 1
        ├── closing_analysis.json           # Phase 1
        ├── winning_formulas.json           # Phase 2 (3-5 formulas)
        └── complete_analysis_18-33s.json   # Combined Phase 1 + Phase 2
```

**LLM Calls per Bucket** (e.g., 18-33s with 6 windows):
- Phase 1: 6 calls (parallel)
- Phase 2: 1 call
- **Total**: 7 calls per bucket

**LLM Calls per Hashtag** (3 qualified buckets):
- Assuming top 3 buckets: 18-33s (6 windows), 33-60s (7 windows), 60-90s (7 windows)
- Phase 1 calls: 6 + 7 + 7 = 20 calls
- Phase 2 calls: 1 + 1 + 1 = 3 calls
- **Total**: 23 calls per hashtag

**Cost Estimate**:
- Per bucket (6 windows): ~$0.26
- Per hashtag (3 buckets): ~$0.78

**Duration Estimate**:
- Phase 1: ~5-10s wall-clock (parallel execution)
- Phase 2: ~15-20s per bucket
- **Total per bucket**: ~25-30s
- **Total per hashtag**: ~90s (if buckets run sequentially), ~30s (if buckets run in parallel)

**Advantages of Dual RF Hybrid Approach**:

| Metric | Dual RF Hybrid (Approved) | Single Call (Old) |
|--------|---------------------------|-------------------|
| **Context per call** | Phase 1: 113 numbers | 1000+ numbers |
| **Hallucination risk** | Low (focused prompts) | Higher (overwhelming) |
| **Parallelization** | Yes (6-7 calls Phase 1) | No |
| **Fault tolerance** | High (window failures don't block others) | Low |
| **Cross-window patterns** | ✅ Video-level RF in Phase 2 | ❌ Limited |
| **Within-window validation** | ✅ Window-level RF in Phase 1 | ❌ Not available |
| **Pattern coverage** | Complete (no blind spots) | Incomplete |
| **API calls** | 7 per bucket | 3 per bucket |
| **Cost** | ~$0.26 per bucket | ~$0.45 per bucket |
| **Total time** | ~25-30s | ~8-12 minutes |

**Key Insights**:
- **Dual RF validation**: Both within-window (Phase 1) AND cross-window (Phase 2) pattern detection
- **Window-level granularity**: Each window analyzed independently with focused context (113 numbers vs 1000+)
- **Winning formulas**: Explicit video journey patterns with RF validation (e.g., "Hook-0 → Middle-1 → Closing-0")
- **Temporal progression**: Track feature evolution across windows (energy builds, eye contact bookends, etc.)
- **Error handling**: Graceful degradation - one window failure doesn't block entire analysis
- **Schema consistency (2025-10-27 update)**: All creative reports (path-based AND feature-based) use identical 13-field schema, ensuring downstream Stage 8 PDF generation compatibility

**Child Documents**:
- LLMAnalysis7.md (complete Stage 7 architecture with dual RF integration)

**Future TI Document**:
- LLMAnalysisTI.md (LLM integration code, prompt templates, Phase 1/2 orchestration)

**Related Future Features**:
- Phase 1: Creative Report Output (Stage 7 core feature)

---