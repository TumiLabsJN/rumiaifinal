# RumiAI ML Pipeline - Foundation & Configuration

> **Parent**: MLPlanningv2.md - Parts 1 & 2
> **Purpose**: Shared foundation document providing cross-cutting system architecture, configuration, and schemas for all pipeline stages
> **Version**: 1.1
> **Last Updated**: 2025-01-28
> **Status**: Draft

---

## Document Overview

**This is a SHARED document** referenced by all stage-specific Child HLDs (VideoDiscoveryCHILD.md, FeatureAggregationCHILD.md, etc.).

**Purpose**: Provides cross-cutting information that all stages depend on:
- System goals and success criteria
- Client directory architecture
- CLI parameter definitions and defaults
- Configuration dimension specifications (target types, analysis modes, selection strategies)
- Cross-cutting data schemas (config.json, Apify metadata, checkpoints)

**How to Use**:
- Stage-specific Child docs reference this document for directory paths, CLI parameters, and config schemas
- TI generation requires BOTH this Foundation document AND the stage-specific Child doc
- Updates to this document cascade to all stage TIs automatically

---

## 1. System Goals & Success Criteria

<!-- Source: MLPlanningv2.md Part 1 - System Goals & Success Criteria -->

### 1.1 Primary Goals

1. **Batch Video Analysis**
   - Process up to 300 videos sequentially through `rumiai_runner.py`
   - Implement checkpoint/resume system for failure recovery

2. **Client-Centric Data Organization**
   - Multi-tenant data structure: Client → Analysis Type → Target → Mode+Strategy → Buckets → Videos
   - Bucket-specific analysis within client/target boundaries
   - Persistent configuration management via config.json

3. **Duration-Specific ML Pattern Recognition**
   - Train separate ML models for each duration bucket (8 buckets total)
   - Recognize that 15-second patterns differ completely from 60-second patterns
   - Generate bucket-specific insights (no universal patterns across durations)

4. **Creative Report Generation**
   - Number of reports per bucket is being determined (design in progress)
   - Report types under consideration: RF feature importance, K-Means cluster strategies, bucket summary

### 1.2 Success Criteria

**Processing Capability**:
- Successfully analyze up to 300 videos per target sequentially
- Support multiple targets per client (e.g., 4-10+ hashtags/competitors/creators)
- Checkpoint/resume system enables recovery without data loss
- Complete end-to-end processing or clear failure identification

**ML Insight Generation**:
- Generate meaningful trends and patterns from analyzed videos
- Include confidence scores and pattern validation

**Report Delivery**:
- Produce concise, actionable creative strategy reports
- Focus on "easy to replicate" format with clear steps
- Identical report structure for all target types

### 1.3 Key Metrics

- **Input Scale**: User-configurable via `--video-count N` per qualified bucket
  - Contrastive default: N=100 per bucket (80 top + 20 bottom), top 3 buckets = ~300 videos
  - Top default: N=40 per bucket, top 3 buckets = ~120 videos
  - Only top 3 most active buckets are processed (adaptive bucket processing)
- **ML Models**: 90 models total across 8 duration buckets for complete pattern coverage:
  - 8 Video-Level Random Forest models (1 per bucket) - Detects cross-window patterns and temporal progressions
  - 41 Window-Level Random Forest models (1 per window per bucket) - Validates window-specific feature importance
  - 41 Window-Level K-Means models (1 per window per bucket) - Discovers creative strategies within each video section
  - Architecture rationale: Dual RF + window-level K-Means prevents blind spots (video-level RF captures "hook→middle consistency", window-level captures "what makes a strong hook", K-Means discovers "3 ways to do strong hooks")
- **Output**: Duration-specific creative recommendations (report count being finalized)
- **Processing**: Sequential (one-by-one) with resumption capability

---

## 2. Client Architecture & Storage

<!-- Source: MLPlanningv2.md Part 1 - Client Architecture & Storage -->

### 2.1 Directory Structure

**Base Path Template**:
```
/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/
```

**Complete Structure**:
```
/config/
├── hashtag_clusters/                      # Cluster configuration directory (NEW)
│   ├── {cluster_id}.json                  # Cluster config (e.g., nutrition.json)
│   ├── test_nutrition.json                # Example: test cluster (2 hashtags × 1 run)
│   └── nutrition_example.json             # Example: full cluster (4 hashtags × 2 runs)

/data/
├── clients/
│   ├── {client_id}/                       # e.g., "acme_corp"
│   │   ├── hashtags/                       # Analysis type: hashtag
│   │   │   └── {cluster_id}/               # Cluster ID (e.g., "nutrition", NOT "#nutrition")
│   │   │       ├── cluster_analytics.json  # Cluster health metrics (NEW)
│   │   │       │                           # Schema: ClusterAnalyticsSchema (Section 5.4)
│   │   │       │                           # Contains: scrape summary, per-hashtag contribution,
│   │   │       │                           #           pairwise overlaps, run effectiveness
│   │   │       ├── top_contrastive/        # Mode: top, Strategy: contrastive
│   │   │       │   ├── config.json         # {client_id, analysis_type, target, analysis_mode, selection_strategy, video_count, date_filter, report_type, report_audience, auto_confirm, run_date}
│   │   │       │   ├── buckets/            # 8 duration buckets
│   │   │       │   │   ├── bucket_0-3s/
│   │   │       │   │   │   ├── videos/     # Raw MP4s (N files)
│   │   │       │   │   │   ├── analysis/   # RumiAI outputs
│   │   │       │   │   │   │   ├── insights/       # temporal_windows_updated.json (1 per video)
│   │   │       │   │   │   │   ├── unified/        # Intermediate timeline+ml_data
│   │   │       │   │   │   │   └── service_debug/  # ML service outputs
│   │   │       │   │   │   ├── validation/         # Pipeline validation outputs
│   │   │       │   │   │   │   ├── rolling_stats.json
│   │   │       │   │   │   │   └── validation_summary.json
│   │   │       │   │   │   ├── flagged_videos/     # Investigation packages
│   │   │       │   │   │   │   └── {video_id}/
│   │   │       │   │   │   │       ├── video.mp4
│   │   │       │   │   │   │       ├── temporal_windows_updated.json
│   │   │       │   │   │   │       ├── unified_analysis.json
│   │   │       │   │   │   │       ├── service_debug/
│   │   │       │   │   │   │       └── validation_report.json
│   │   │       │   │   │   ├── ml_analysis/        # ML pipeline outputs
│   │   │       │   │   │   │   ├── aggregated_features.csv
│   │   │       │   │   │   │   ├── rf_transformed.csv
│   │   │       │   │   │   │   ├── km_transformed.csv
│   │   │       │   │   │   │   ├── random_forest_analysis.json
│   │   │       │   │   │   │   └── kmeans_analysis.json
│   │   │       │   │   │   ├── models/             # Trained models
│   │   │       │   │   │   │   ├── random_forest_v1.pkl
│   │   │       │   │   │   │   ├── kmeans_v1.pkl
│   │   │       │   │   │   │   ├── scalers.pkl
│   │   │       │   │   │   │   └── model_metrics.json
│   │   │       │   │   │   ├── llm_reports/        # LLM outputs
│   │   │       │   │   │   │   ├── analysis/      # LLM Call 1 (insight extraction)
│   │   │       │   │   │   │   │   ├── call_1_rf_prompt.txt
│   │   │       │   │   │   │   │   ├── call_1_rf_raw_response.json
│   │   │       │   │   │   │   │   ├── call_1_kmeans_prompt.txt
│   │   │       │   │   │   │   │   ├── call_1_kmeans_raw_response.json
│   │   │       │   │   │   │   │   └── insights.json
│   │   │       │   │   │   │   └── formatted/     # LLM Call 2 (report generation)
│   │   │       │   │   │   │       ├── call_2_prompt.txt
│   │   │       │   │   │   │       ├── call_2_raw_response.json
│   │   │       │   │   │   │       ├── rf_feature_importance.md
│   │   │       │   │   │   │       ├── strategy_1_the_educator.md
│   │   │       │   │   │   │       ├── strategy_2_visual_storyteller.md
│   │   │       │   │   │   │       ├── strategy_3_personal_journey.md
│   │   │       │   │   │   │       └── bucket_summary.md
│   │   │       │   │   │   ├── reports/            # Final PDFs
│   │   │       │   │   │   │   ├── rf_feature_importance.pdf
│   │   │       │   │   │   │   ├── strategy_1_the_educator.pdf
│   │   │       │   │   │   │   ├── strategy_2_visual_storyteller.pdf
│   │   │       │   │   │   │   ├── strategy_3_personal_journey.pdf
│   │   │       │   │   │   │   └── bucket_summary.pdf
│   │   │       │   │   │   ├── checkpoints/        # Processing state
│   │   │       │   │   │   │   └── stage_{X}_checkpoint.json
│   │   │       │   │   │   └── logs/               # Bucket processing logs
│   │   │       │   │   │       └── processing_{date}.log
│   │   │       │   │   ├── bucket_3-9s/
│   │   │       │   │   ├── bucket_9-13s/
│   │   │       │   │   ├── bucket_13-18s/
│   │   │       │   │   ├── bucket_18-33s/
│   │   │       │   │   ├── bucket_33-60s/
│   │   │       │   │   ├── bucket_60-90s/
│   │   │       │   │   └── bucket_90-120s/
│   │   │       │   └── hashtag_summary/           # Cross-bucket reports
│   │   │       │       ├── executive_report.pdf
│   │   │       │       └── hashtag_metrics.json
│   │   │       │
│   │   │       ├── top_top/                       # Optional: different strategy
│   │   │       ├── recent_top/                    # Optional: different mode
│   │   │       └── ...
│   │   │
│   │   ├── competitors/                           # Analysis type: competitor
│   │   │   └── {competitor_handle}/               # e.g., "rival_brand" (@ removed)
│   │   │       ├── top_top/                       # Default for competitors
│   │   │       ├── top_contrastive/               # Optional
│   │   │       └── ...
│   │   │
│   │   └── creators/                              # Analysis type: creator
│   │       └── {creator_handle}/                  # e.g., "potential_affiliate" (@ removed)
│   │           ├── recent_top/                    # Default for creators
│   │           ├── top_top/                       # Optional
│   │           └── ...
```

### 2.2 Path Templates

**For TI Implementation**:
```python
BASE_PATHS = {
    # ===== CLUSTER CONFIGURATION PATHS (NEW) =====
    "cluster_config_dir": "/config/hashtag_clusters/",
    "cluster_config_file": "/config/hashtag_clusters/{cluster_id}.json",
    "cluster_analytics_file": "/data/clients/{client_id}/hashtag/{cluster_id}/cluster_analytics.json",

    # ===== CLIENT DATA PATHS =====
    "client_base": "/data/clients/{client_id}/",
    "analysis_type_base": "{client_base}/{analysis_type}s/",  # Note: plural form
    "target_base": "{analysis_type_base}/{target}/",
    "analysis_base": "{target_base}/{mode}_{strategy}/",
    "bucket_base": "{analysis_base}/bucket_{bucket}/",

    # Standard subdirectories per bucket
    "videos": "{bucket_base}/videos/",
    "analysis": "{bucket_base}/analysis/",
    "insights": "{bucket_base}/analysis/insights/",
    "unified": "{bucket_base}/analysis/unified/",
    "service_debug": "{bucket_base}/analysis/service_debug/",
    "validation": "{bucket_base}/validation/",
    "flagged_videos": "{bucket_base}/flagged_videos/",
    "ml_analysis": "{bucket_base}/ml_analysis/",
    "models": "{bucket_base}/models/",
    "llm_reports": "{bucket_base}/llm_reports/",
    "llm_analysis": "{bucket_base}/llm_reports/analysis/",
    "llm_formatted": "{bucket_base}/llm_reports/formatted/",
    "reports": "{bucket_base}/reports/",
    "checkpoints": "{bucket_base}/checkpoints/",
    "logs": "{bucket_base}/logs/",
}

# Example usage in stage implementations:
video_path = BASE_PATHS["videos"].format(
    client_id="acme_corp",
    analysis_type="hashtag",
    target="nutrition",  # Remove # or @ prefix
    mode="top",
    strategy="contrastive",
    bucket="18-33s"
)
# Result: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/videos/
```

### 2.2.1 Path Sanitization Rules

**Target Sanitization Algorithm**:
```python
import re

def sanitize_target(target: str, analysis_type: str) -> str:
    """
    Sanitize target for filesystem path usage.

    Rules:
    1. Remove prefix (# for hashtag, @ for competitor/creator)
    2. Convert to lowercase
    3. Replace spaces with underscores
    4. Remove or replace special characters (keep only alphanumeric, underscore, hyphen)
    5. Collapse multiple underscores to single underscore
    6. Strip leading/trailing underscores
    """
    # Step 1: Remove prefix
    if analysis_type == "hashtag" and target.startswith("#"):
        sanitized = target[1:]
    elif analysis_type in ["competitor", "creator"] and target.startswith("@"):
        sanitized = target[1:]
    else:
        sanitized = target

    # Step 2: Lowercase
    sanitized = sanitized.lower()

    # Step 3: Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")

    # Step 4: Remove special characters (keep alphanumeric, underscore, hyphen)
    sanitized = re.sub(r'[^a-z0-9_-]', '', sanitized)

    # Step 5: Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Step 6: Strip leading/trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized
```

**Examples**:
| Input | Output |
|-------|--------|
| `#Fitness & Nutrition!` | `fitness_nutrition` |
| `@My Brand 2024` | `my_brand_2024` |
| `#nutrition` | `nutrition` |
| `@rival__brand` | `rival_brand` |
| `#Weight-Loss` | `weight-loss` |
| `@_special_user_` | `special_user` |

**Client ID Sanitization**:
Client IDs follow the same rules but without prefix removal (already provided without prefixes via `--client` parameter).

### 2.3 Architecture Notes

- **Analysis Directories**: Each `{mode}_{strategy}/` directory is a complete, independent analysis run
- **config.json**: Stores run parameters (mode, strategy, date_filter, run_date, video_count) for reproducibility
- **Coexistence**: Multiple analyses can exist simultaneously without overwriting
- **Default Paths**: Each analysis type writes to its default analysis directory
  - Hashtag → `top_contrastive/`
  - Competitor → `top_top/`
  - Creator → `recent_top/`
- **Video Counts**: User-configurable via `--video-count N`
  - Contrastive: N per qualified bucket (80/20 split, default N=100)
  - Top: N per qualified bucket (all top, default N=40)
  - Only top 3 most active buckets are processed (adaptive bucket processing)

### 2.4 Data Retention Policy

| Asset | Retention | Rationale |
|-------|-----------|-----------|
| **Raw Videos** | 30 days | Can re-download if needed, saves space |
| **ML Analysis** | 6 months | Compressed after 30 days |
| **ML Models** | Latest 3 versions | Per client/target |
| **Reports** | Indefinite | Small size, high value |
| **Checkpoints** | 7 days after completion | Enable resume capability |

### 2.5 Storage Cost Optimization

- **Video Deletion**: Remove raw videos after 30 days
- **ML Analysis Compression**: Compress after 60 days
- **Model Versioning**: Keep only latest 3 versions per client/target

---

## 3. Configuration Dimensions

<!-- Source: MLPlanningv2.md Part 2 - Configuration -->

All RumiAI analyses use **multiple orthogonal configuration dimensions**. These flags work independently and can be combined in any valid way.

### 3.1 Target Types (Stage 0.1)

**Purpose**: Determines what source to analyze videos from

**Available Types**:

| Type | CLI Flag | Target Format | Data Source | Primary Use Case | Scraping Mode |
|------|----------|---------------|-------------|------------------|---------------|
| `hashtag` | `--analysis-type hashtag` | `nutrition` (cluster ID) | TikTok hashtag search (multi-hashtag cluster) | Market research - identify viral patterns | **Cluster mode** (4 hashtags × 2 runs) |
| `competitor` | `--analysis-type competitor` | `@rival_brand` | TikTok profile | Competitive intelligence - understand rivals | Single mode (1 profile, 800 videos) |
| `creator` | `--analysis-type creator` | `@potential_affiliate` | TikTok profile | Creator vetting - assess fit for hiring | Single mode (1 profile, 800 videos) |

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

---

### 3.2 Analysis Modes (Stage 0.2)

**Purpose**: Controls how Apify sorts and selects videos

**Available Modes**:

| Mode | Sort By | Use Case | Default For |
|------|---------|----------|-------------|
| `top` | Engagement (composite score) | "What works?" - Identify successful patterns | Hashtag, Competitor |
| `recent` | Publish date (newest first) | "What's happening now?" - Track current trends | Creator |

**Engagement Score Formula** (Top Mode):
```
engagement_score = views × (1 + share_rate × 10)

where:
  share_rate = shares / views
  share_boost = 1 + (share_rate × 10)
```

**Formula Source**: MLPlanningv2.md Section 0.2 (Stage 0.2: Analysis Modes)

**Why This Formula**:
- Shares signal viral potential (10x weight reflects their importance)
- Views alone can be misleading (high views, low engagement)
- Captures "share-worthy" content beyond just popularity

---

### 3.3 Selection Strategies (Stage 0.3)

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

**Example Combinations**:
- `top mode + contrastive strategy` = Top performers analyzed with contrastive learning
- `recent mode + top strategy` = Most recent high-quality content only
- `top mode + top strategy` = Best-of-best analysis (no bottom performers)
- `recent mode + contrastive strategy` = Recent winners vs recent underperformers

---

### 3.4 Video Count (Stage 0.4)

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

**Valid Range**: 10-500 videos per bucket

**Minimum Recommended N by Strategy**:

| Strategy | Minimum N | Rationale |
|----------|-----------|-----------|
| **Contrastive** | 50 | Ensures at least 10 bottom performers (20%) for statistical validity in classification |
| **Top** | 20 | Minimum sample size for K-Means clustering (3 clusters × ~7 videos per cluster) |

**Warning Thresholds**:
- **N < 50 for contrastive**: System warns "Low bottom performer count (N × 0.2 = {count}). Recommend N ≥ 50 for robust classification."
- **N < 20 for top**: System warns "Low sample size for clustering. Recommend N ≥ 20 for pattern detection."

**Absolute Minimum** (hard limit): N ≥ 10
- Below 10: System rejects with error "Insufficient sample size. Minimum N=10 required."

**Note**: These are recommendations. System allows N=10-49 with warnings, but quality of ML insights may degrade with insufficient data.

**Examples**:
```bash
# Contrastive with 150 videos per bucket (120 top + 30 bottom)
--selection-strategy contrastive --video-count 150

# Top with 60 videos per bucket
--selection-strategy top --video-count 60

# Minimum acceptable (with warning)
--selection-strategy contrastive --video-count 30  # Warning: only 6 bottom performers
```

---

### 3.5 Date Filtering (Stage 0.5)

**Purpose**: Controls when videos were published - filters scraped videos by publication date

**CLI Parameter**: `--date-filter last_N_days`

**Format**: Relative date range only
- `last_N_days` where N is the number of days to look back from today
- Examples: `last_30_days`, `last_90_days`, `last_180_days`

**Default**: `last_90_days`

**How It Works**:
1. Apify scrapes videos from target (`resultsPerPage`: 800 - hard limit enforced by Apify)
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

**Default Per Target Type**:
- **Hashtag**: `last_90_days` (quarterly trends for market research)
- **Competitor**: `last_90_days` (current competitive strategies)
- **Creator**: `last_30_days` (recent natural style for vetting)

---

### 3.6 Country Code (Stage 0.6)

**Purpose**: Controls geographic content filtering via Apify proxy routing

**CLI Parameter**: `--country-code {country}`

**Available Values**:

| Value | Proxy Behavior | Content Returned | Use Case |
|-------|----------------|------------------|----------|
| `US` | proxyCountryCode: "US" | US-specific trending content | US market analysis (default) |
| `BR` | proxyCountryCode: "BR" | Brazil-specific trending content | Brazilian market analysis |
| `global` | No proxy parameter | Unfiltered global content | Cross-market comparison |

**Default**: `US`

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

**Examples**:
```bash
# US market analysis (default)
--country-code US

# Brazilian market analysis
--country-code BR

# Global content (no filtering)
--country-code global
```

---

### 3.7 Report Types (Stage 0.7)

**Purpose**: Determines what type of output is generated

**Available Types**:

| Type | What It Does | Prerequisites | Output | Applies To |
|------|-------------|---------------|--------|------------|
| `single` | Deep analysis of one target | None | Full ML analysis + creative reports | All target types (default) |
| `comparison` | Side-by-side comparison of 2+ targets | All targets must have existing single analyses | LLM-synthesized comparison report | All target types |

**Key Differences**:

| Aspect | Single | Comparison |
|--------|--------|------------|
| **Video Processing** | Full ML pipeline (hours) | No processing (seconds) |
| **Prerequisites** | None | Requires existing single analyses |
| **LLM Calls** | 24 per target (analysis + formatting) | 1 per comparison group |
| **Output** | ML models + 40 creative reports | 1 comparison PDF |
| **Duration** | 6-8 hours (hashtag, 300 videos) | ~30 seconds (LLM only) |
| **Cost** | Apify ($4) + Compute + LLM ($3.60) | LLM only (~$0.50) |

---

### 3.8 Report Audience (Stage 0.8)

**Purpose**: Determines target audience for generated reports (affects language, detail level, formatting)

**Available Audiences**:

| Audience | Who It's For | Language Style | Detail Level | Use Case |
|----------|-------------|----------------|--------------|----------|
| `client` | Brand stakeholders, marketing teams | Formal, business-oriented | High-level insights, strategic recommendations | Hashtag, Competitor analysis |
| `internal` | Tumi Labs team, data scientists | Technical, analytical | Full technical details, model metrics, raw data | Debugging, model validation, research |
| `creator` | Content creators, influencers | Casual, actionable, encouraging | Practical tips, specific actions to take | Creator monitoring, coaching |

**Default Per Target Type**:
- **Hashtag**: `client` (brands analyzing market trends)
- **Competitor**: `client` (brands benchmarking against rivals)
- **Creator**: `creator` (coaching individual creators)

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

---

## 4. CLI Command Structure

<!-- Source: MLPlanningv2.md Part 2 - Command Structure -->

**Complete Command Syntax**:
```bash
python rumiai_ml_batch.py \
  --client "client_name" \
  --analysis-type {hashtag|competitor|creator} \
  --target "{target}" \
  --analysis-mode {top|recent} \
  --selection-strategy {contrastive|top} \
  --video-count N \
  --date-filter last_N_days \
  --country-code {US|BR|global} \
  --report-type {single|comparison} \
  --report-audience {client|internal|creator} \
  --auto-confirm              # Skip interactive prompts (for CI/CD)
```

**Design Principles**:
- **Orthogonal**: Each dimension is independent
- **Composable**: Valid combinations work across all target types
- **Default-aware**: Each target type has sensible defaults
- **Explicit**: All parameters exposed as CLI flags (automation-friendly)

### 4.1 CLI Parameters

| Parameter | Type | Required | Default | Valid Values | Description | Validation Rules |
|-----------|------|----------|---------|--------------|-------------|------------------|
| `--client` | str | Yes | None | Alphanumeric + underscore | Client identifier (e.g., "acme_corp") | Regex: `^[a-zA-Z0-9_]+$` (min 1 char) |
| `--analysis-type` | str | Yes | None | hashtag, competitor, creator | Target type to analyze | Enum: `["hashtag", "competitor", "creator"]` |
| `--target` | str | Yes | None | Varies by type | Target identifier (#nutrition, @rival_brand, @creator_name) | Format by type:<br>- hashtag: must start with `#`, min 2 chars<br>- competitor/creator: must start with `@`, min 2 chars |
| `--analysis-mode` | str | No | Depends on type | top, recent | Sorting method for video selection | Enum: `["top", "recent"]` |
| `--selection-strategy` | str | No | Depends on type | contrastive, top | Video subset selection strategy | Enum: `["contrastive", "top"]` |
| `--video-count` | int | No | Depends on strategy | 10-500 | Videos to analyze per winning bucket | Integer range: 10-500 (inclusive) |
| `--date-filter` | str | No | last_90_days | last_N_days | Publication date filter (relative days) | Regex: `^last_\d+_days$` where `\d+` is 1-365 |
| `--country-code` | str | No | US | US, BR, global | Geographic content filter via Apify proxy routing | Enum: `["US", "BR", "global"]` |
| `--report-type` | str | No | single | single, comparison | Output type | Enum: `["single", "comparison"]` |
| `--report-audience` | str | No | Depends on type | client, internal, creator | Target audience for report (affects language, detail level) | Enum: `["client", "internal", "creator"]` |
| `--auto-confirm` | bool | No | False | True, False | Skip interactive confirmation prompts (for CI/CD automation) | Boolean flag (no value required) |

### 4.2 Default Value Logic

**Hashtag Defaults**:
```python
if analysis_type == "hashtag":
    mode = mode or "top"
    strategy = strategy or "contrastive"
    video_count = video_count or 100
    date_filter = date_filter or "last_90_days"
    country_code = country_code or "US"
```

**Competitor Defaults**:
```python
if analysis_type == "competitor":
    mode = mode or "top"
    strategy = strategy or "top"
    video_count = video_count or 40
    date_filter = date_filter or "last_90_days"
    country_code = country_code or "US"
```

**Creator Defaults**:
```python
if analysis_type == "creator":
    mode = mode or "recent"
    strategy = strategy or "top"
    video_count = video_count or 40
    date_filter = date_filter or "last_30_days"
    country_code = country_code or "US"
```

### 4.3 Example Commands

**Hashtag Analysis (Using Defaults)**:
```bash
python rumiai_ml_batch.py \
  --client "acme" \
  --analysis-type hashtag \
  --target "#nutrition"
# Applies: mode=top, strategy=contrastive, video_count=100, date_filter=last_90_days
```

**Competitor Analysis (Override Defaults)**:
```bash
python rumiai_ml_batch.py \
  --client "acme" \
  --analysis-type competitor \
  --target "@rival_brand" \
  --selection-strategy contrastive \
  --video-count 150
# Applies: mode=top (default), strategy=contrastive (override), video_count=150 (override)
```

**Creator Analysis (Recent + Top)**:
```bash
python rumiai_ml_batch.py \
  --client "acme" \
  --analysis-type creator \
  --target "@potential_affiliate"
# Applies: mode=recent (default), strategy=top (default), video_count=40 (default), date_filter=last_30_days (default)
```

**Brazilian Market Analysis**:
```bash
python rumiai_ml_batch.py \
  --client "acme" \
  --analysis-type hashtag \
  --target "#fitness" \
  --country-code BR
# Applies: mode=top (default), strategy=contrastive (default), country_code=BR (override)
# Returns Brazil-specific trending content via Apify proxy routing
```

**Global Content (No Geographic Filter)**:
```bash
python rumiai_ml_batch.py \
  --client "acme" \
  --analysis-type hashtag \
  --target "#wellness" \
  --country-code global
# Applies: country_code=global (override) - omits proxyCountryCode parameter
# Returns unfiltered global content from mixed geographic sources
```

---

## 5. Configuration Schemas

<!-- Source: MLPlanningv2.md Part 1 & 2 - Cross-cutting schemas -->

### 5.1 config.json Schema

**Location**: `{analysis_base}/config.json`

**Purpose**: Stores run parameters for reproducibility and resumption

**Schema**:
```python
ConfigSchema = {
    "client_id": str,              # Required, alphanumeric + underscore, Example: "acme_corp"
    "analysis_type": str,          # Required, ["hashtag", "competitor", "creator"], Example: "hashtag"
    "target": str,                 # Required, format depends on analysis_type, Example: "#nutrition"
    "analysis_mode": str,          # Required, ["top", "recent"], Example: "top"
    "selection_strategy": str,     # Required, ["contrastive", "top"], Example: "contrastive"
    "video_count": int,            # Required, Range: 10-500, Example: 100
    "date_filter": str,            # Required, "last_N_days", Example: "last_90_days"
    "country_code": str,           # Required, ["US", "BR", "global"], Example: "US"
    "report_type": str,            # Required, ["single", "comparison"], Example: "single"
    "report_audience": str,        # Required, ["client", "internal", "creator"], Example: "client"
    "auto_confirm": bool,          # Required, skip interactive prompts, Example: false
    "run_date": str,               # Required, ISO 8601 format, Example: "2025-01-28T10:30:00Z"
}
```

**Example**:
```json
{
  "client_id": "acme_corp",
  "analysis_type": "hashtag",
  "target": "#nutrition",
  "analysis_mode": "top",
  "selection_strategy": "contrastive",
  "video_count": 100,
  "date_filter": "last_90_days",
  "country_code": "US",
  "report_type": "single",
  "report_audience": "client",
  "auto_confirm": false,
  "run_date": "2025-01-28T10:30:00Z"
}
```

---

### 5.2 Apify Video Metadata Schema

**Source**: Apify scrapers (clockworks/tiktok-hashtag-scraper, clockworks/tiktok-scraper)

**Scraping Configuration**:
- `resultsPerPage`: 800 (hard limit enforced by Apify)
- Defined in: `ml_pipeline/stage1_discovery/constants.py::APIFY_SCRAPE_COUNT`

**Schema**:
```python
ApifyVideoMetadataSchema = {
    "id": str,                     # Required, Unique video ID, Example: "7428596413707144481"
    "createTime": int,             # Required, Unix timestamp, Example: 1704067200
    "duration": int,               # Required, Seconds, Range: 3-120, Example: 25
    "playCount": int,              # Required, Views, >= 0, Example: 50000
    "shareCount": int,             # Required, Shares, >= 0, Example: 500
    "commentCount": int,           # Required, Comments, >= 0, Example: 250
    "likeCount": int,              # Required, Likes, >= 0, Example: 3500
    "webVideoUrl": str,            # Required, TikTok URL, Example: "https://www.tiktok.com/@user/video/123"
    "videoMeta": {                 # Required
        "downloadAddr": str,       # Required, MP4 URL for download
        "width": int,              # Required, Resolution width, Example: 1080
        "height": int,             # Required, Resolution height, Example: 1920
    },
    "authorMeta": {                # Required
        "id": str,                 # Required, Author ID
        "name": str,               # Required, Author username, Example: "@fitness_guru"
    },
}
```

---

### 5.3 Checkpoint Schema

**Location**: `{bucket_base}/checkpoints/stage_{X}_checkpoint.json`

**Purpose**: Enables resume on interruption (SSH disconnect, crash, manual stop)

**Schema**:
```python
CheckpointSchema = {
    "stage": str,                  # Required, Stage name, Example: "video_processing"
    "bucket": str,                 # Required, Bucket name, Example: "18-33s"
    "total_videos": int,           # Required, Total videos to process, Example: 100
    "completed": int,              # Required, Successfully processed, Example: 45
    "failed": int,                 # Required, Failed with errors, Example: 2
    "remaining": int,              # Required, Not yet processed, Example: 53
    "last_checkpoint": str,        # Required, ISO timestamp, Example: "2025-01-28T14:32:15Z"
    "completed_video_ids": list[str],   # Required, List of processed video IDs
    "failed_video_ids": list[dict],     # Required, List of failure records
        # Nested schema for failed_video_ids items:
        # {
        #   "video_id": str,        # Required, Video ID that failed
        #   "error": str,           # Required, Error message/reason
        #   "timestamp": str,       # Optional, ISO timestamp of failure
        #   "stage": str            # Optional, Substage that failed (e.g., "FEAT", "Whisper")
        # }
}
```

**Example**:
```json
{
  "stage": "video_processing",
  "bucket": "18-33s",
  "total_videos": 100,
  "completed": 45,
  "failed": 2,
  "remaining": 53,
  "last_checkpoint": "2025-01-28T14:32:15Z",
  "completed_video_ids": ["123", "456", "789"],
  "failed_video_ids": [
    {"video_id": "321", "error": "FEAT timeout after 120s"}
  ]
}
```

---

### 5.4 Cluster Configuration Schema (NEW)

**Source**: VideoDiscoveryCHILD.md Section 5.2a (from HashtagVolumeV2.md DECISION 1)

**Location**: `/config/hashtag_clusters/{cluster_id}.json`

**Purpose**: Defines multi-hashtag scraping clusters for hashtag analysis

**Schema**:
```python
ClusterConfigSchema = {
    "cluster_id": str,             # Required, cluster identifier (alphanumeric + underscore)
                                   # Example: "nutrition"

    "description": str,            # Required, human-readable description
                                   # Example: "Nutrition niche - narrow semantic cluster"

    "primary_hashtag": str,        # Required, original target hashtag (starts with #)
                                   # Example: "#nutrition"

    "variant_hashtags": list[str], # Required, 1-10 variant hashtags (each starts with #)
                                   # Example: ["#nutritionist", "#nutritiontips", "#nutritioncoach"]

    "scrape_config": {             # Required, scraping parameters
        "runs_per_hashtag": int,   # Required, runs per hashtag (1-5)
                                   # Example: 2

        "delay_between_runs_ms": int,  # Required, delay between scrapes (60000-600000ms)
                                       # Example: 120000 (2 minutes)

        "results_per_page": int,   # Required, videos per scrape (100-800)
                                   # Example: 800
    },

    "metadata": {                  # Optional, user metadata
        "created_date": str,       # Optional, ISO 8601 timestamp
        "created_by": str,         # Optional, creator username
        "notes": str,              # Optional, additional notes
    }
}
```

**Validation Requirements**:
- `cluster_id`: Regex `^[a-zA-Z0-9_]+$` (min 1 char)
- `primary_hashtag`: Regex `^#[a-zA-Z0-9_]+$` (min 2 chars)
- `variant_hashtags`: Array length 1-10, each element matches hashtag regex
- `runs_per_hashtag`: Range 1-5
- `delay_between_runs_ms`: Range 60000-600000 (1-10 minutes)
- `results_per_page`: Range 100-800

**Example**:
```json
{
  "cluster_id": "nutrition",
  "description": "Nutrition niche - narrow semantic cluster (20-30% overlap)",
  "primary_hashtag": "#nutrition",
  "variant_hashtags": ["#nutritionist", "#nutritiontips", "#nutritioncoach"],
  "scrape_config": {
    "runs_per_hashtag": 2,
    "delay_between_runs_ms": 120000,
    "results_per_page": 800
  },
  "metadata": {
    "created_date": "2025-10-10T14:30:00Z",
    "created_by": "RumiAI Team",
    "notes": "4 hashtags × 2 runs = 8 scrapes, ~1400 unique videos expected"
  }
}
```

---

### 5.5 Cluster Analytics Schema (NEW)

**Source**: VideoDiscoveryCHILD.md Section 5.2b (from HashtagVolumeV2.md DECISION 3)

**Location**: `/data/clients/{client_id}/hashtag/{cluster_id}/cluster_analytics.json`

**Purpose**: Cluster health analytics for optimization and monitoring

**Schema**:
```python
ClusterAnalyticsSchema = {
    "cluster_id": str,             # Required, cluster identifier
                                   # Example: "nutrition"

    "execution_date": str,         # Required, ISO 8601 timestamp
                                   # Example: "2025-10-10T14:30:00Z"

    "scrape_summary": {            # Required, overall scraping statistics
        "total_scrapes_attempted": int,  # Example: 8 (4 hashtags × 2 runs)
        "total_scrapes_succeeded": int,  # Example: 8
        "total_scraped_videos": int,     # Example: 1939 (before dedup)
        "total_unique_videos": int,      # Example: 1400 (after dedup)
        "overall_duplication_rate": float,  # Example: 27.8 (percentage)
        "failed_scrapes": list[dict],    # Example: [] or [{"hashtag": str, "run": int, "error": str}]
    },

    "per_hashtag_contribution": dict[str, dict],  # Key: hashtag name
                                                  # Value: {
                                                  #   "total_found": int,
                                                  #   "exclusive_videos": int,
                                                  #   "contribution_percentage": float
                                                  # }

    "pairwise_overlaps": dict[str, float],  # Key: "hashtag1_vs_hashtag2" (alphabetical)
                                            # Value: overlap percentage
                                            # Example: {"nutrition_vs_nutritionist": 18.2}

    "run_effectiveness": dict[str, dict],   # Key: hashtag name
                                            # Value: {
                                            #   "run_1_videos": int,
                                            #   "run_2_videos": int,
                                            #   "run_2_new_videos": int,
                                            #   "run_2_new_percentage": float
                                            # }

    "bucket_distribution_by_source": dict[str, dict],  # Key: bucket name (e.g., "60-90s")
                                                       # Value: {
                                                       #   "total_videos": int,
                                                       #   "by_hashtag": dict[str, int]
                                                       # }
}
```

**Usage**:
- Generated by Stage 1 (Video Discovery) after deduplication
- Used for cluster optimization: identify low-contributing hashtags for removal
- Used for cost savings: analyze run effectiveness (2 runs vs 3 runs)
- Used for root cause analysis: diagnose bucket deficiencies by source hashtag

**Example**:
```json
{
  "cluster_id": "nutrition",
  "execution_date": "2025-10-10T14:45:00Z",
  "scrape_summary": {
    "total_scrapes_attempted": 8,
    "total_scrapes_succeeded": 8,
    "total_scraped_videos": 1939,
    "total_unique_videos": 1400,
    "overall_duplication_rate": 27.8,
    "failed_scrapes": []
  },
  "per_hashtag_contribution": {
    "#nutrition": {
      "total_found": 782,
      "exclusive_videos": 450,
      "contribution_percentage": 55.9
    },
    "#nutritionist": {
      "total_found": 420,
      "exclusive_videos": 180,
      "contribution_percentage": 30.0
    }
  },
  "pairwise_overlaps": {
    "nutrition_nutritionist": 18.2,
    "nutrition_nutritiontips": 25.1
  },
  "run_effectiveness": {
    "#nutrition": {
      "run_1_videos": 690,
      "run_2_videos": 720,
      "run_2_new_videos": 92,
      "run_2_new_percentage": 12.8
    }
  },
  "bucket_distribution_by_source": {}
}
```

---

## 6. Bucket Definitions

**8 Duration Buckets**:

| Bucket | Duration Range | Middle Segments | Total Windows | Use Case |
|--------|----------------|-----------------|---------------|----------|
| 0-3s | 0-3 seconds | 0 (null) | 2 (Hook + Closing) | Ultra-short hooks |
| 3-9s | 3-9 seconds | 0 (null) | 2 (Hook + Closing) | Short hooks |
| 9-13s | 9-13 seconds | 3 | 5 | Short-form content |
| 13-18s | 13-18 seconds | 3 | 5 | TikTok sweet spot (15s) |
| 18-33s | 18-33 seconds | 4 | 6 | Medium-form content |
| 33-60s | 33-60 seconds | 5 | 7 | Long-form content |
| 60-90s | 60-90 seconds | 5 | 7 | Extended content |
| 90-120s | 90-120 seconds | 5 | 7 | Maximum TikTok length |

**Adaptive Processing**:
- Only top 3 buckets where winners cluster are processed
- Success-based selection (not volume-based)
- Example: Skip 9-13s (400 videos, 5 winners), Process 18-60s (150 videos, 75 winners)

**Window Configuration**:
- Bucket-specific window configurations are defined in `config/bucket_definitions.py`
- This shared configuration is imported by Stage 4 (Feature Transformation) and Stage 6 (ML Analysis Generation)
- See `config/bucket_definitions.py` for the BUCKET_WINDOWS dictionary

### 6.1 Bucket Assignment Logic

**Assignment Algorithm**:
```python
def assign_bucket(duration: float) -> str:
    """Assign video to duration bucket based on video length."""
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
    elif duration <= 120:
        return "90-120s"
    else:
        raise ValueError(f"Video duration {duration}s exceeds TikTok maximum (120s)")
```

**Edge Cases**:
- Video exactly 9.0s → assigns to "9-13s" bucket (inclusive lower bound)
- Video exactly 120.0s → assigns to "90-120s" bucket (inclusive upper bound)
- Video >120s → rejected (TikTok platform maximum is 120s)
- Video <3s → assigns to "0-3s" bucket (valid but rare)

**Boundary Behavior**:
- All buckets use inclusive lower bound: `duration >= lower_bound`
- All buckets use exclusive upper bound: `duration < upper_bound`
- Exception: Final bucket "90-120s" uses inclusive upper bound: `duration <= 120`

---

## 7. References

### 7.1 Parent Document

- **MLPlanningv2.md** (lines 1-498)
  - Part 1: Foundation (lines 39-233)
  - Part 2: Configuration (lines 236-498)

### 7.2 Related Documents

- **MLROADMAP.md**: Business context and ML roadmap
- **SystemArchitecturev2.md**: RumiAI processing pipeline architecture
- **ChildTemplate.md**: Template for stage-specific Child HLDs

### 7.3 Usage by Stage-Specific Child Docs

All stage-specific Child HLDs reference this document:
- VideoDiscoveryCHILD.md (Stage 1)
- VideoProcessingCHILD.md (Stage 2)
- PipelineValidationCHILD.md (Stage 2.4)
- FeatureAggregationCHILD.md (Stage 3)
- FeatureTransformationCHILD.md (Stage 4)
- MLModelTrainingCHILD.md (Stage 5)
- MLAnalysisGenerationCHILD.md (Stage 6)
- LLMReportGenerationCHILD.md (Stage 7)

---

## Appendix A: Glossary (Shared Terms)

<!-- PURPOSE: Define domain-specific terminology used across all stages. Child HLDs reference this glossary and only define component-specific terms in their own Appendix A. -->

### Domain-Specific Terms

**Temporal Window**
- A time-bounded segment of a video used for feature extraction
- Each bucket has different temporal window configurations
- Examples: hook window (0-3s), middle_1 window (varies by bucket), closing window (last 3s)
- Used in: Stages 2, 3, 4, 6

**Bucket**
- Duration range grouping for videos
- 8 buckets defined: 0-3s, 3-9s, 9-13s, 13-18s, 18-33s, 33-60s, 60-90s, 90-120s
- Each bucket trains separate ML models
- Rationale: 15-second video patterns differ fundamentally from 60-second patterns
- Used in: All stages (0-7)

**Hook**
- The first temporal window of a video (always 0-3 seconds)
- Most critical window for viewer engagement
- Contains features: scene_count, eye_contact_rate, word_count, emotion metrics, etc.
- Used in: Stages 2, 3, 4, 5, 6

**Middle Segments**
- Temporal windows between hook and closing
- Count varies by bucket: 0 segments (short videos), up to 5 segments (long videos)
- Named: middle_1, middle_2, middle_3, middle_4, middle_5
- Same feature set as hook and closing windows
- Used in: Stages 2, 3, 4, 5, 6

**Closing**
- The final temporal window of a video (always last 3 seconds)
- Critical for call-to-action and viewer retention
- Contains features: scene_count, energy_level, emotion metrics, etc.
- Used in: Stages 2, 3, 4, 5, 6

**FEAT (Facial Expression Analysis Tool)**
- External emotion detection API used in Stage 2
- Provides: joy, surprise, anger, disgust, fear, sadness, neutral emotion scores
- Known issue: 15% timeout rate, mitigated with retry logic
- Used in: Stage 2 (VideoProcessing), referenced in Stage 6 reports

**Checkpoint**
- Saved state allowing pipeline resume without re-processing completed stages
- Format: checkpoint.json with stage completion status
- Strategy: Stage-level granularity (not video-level)
- Used in: Stages 1-7, managed by rumiai_runner.py

**Contrastive Strategy**
- Video selection approach comparing top 80% vs bottom 20% performers
- Target variable: `is_top_performer` (binary classification)
- Used for: Random Forest model training
- Alternative: Top Strategy
- Used in: Stages 0, 1, 4, 5

**Top Strategy**
- Video selection approach using only top-performing videos
- No target variable (unsupervised learning)
- Used for: K-Means clustering
- Alternative: Contrastive Strategy
- Used in: Stages 0, 1, 4, 5

**Engagement Metrics**
- Quantitative performance indicators for video success
- Apify provides: views, likes, shares, comments, watch_time
- Primary metric: view_count (used for ranking)
- Used in: Stages 0, 1, 6, 7

**Feature Aggregation**
- Process of combining temporal window features into single video-level row
- Input: N temporal_windows JSONs per video → Output: 1 row with ~185 columns
- Column naming: `{window_name}_{feature_name}` (e.g., hook_scene_count)
- Used in: Stage 3, consumed by Stages 4-7

**Client**
- Top-level organizational entity (e.g., "nike", "adidas")
- Each client has independent data directory: `/data/clients/{client_id}/`
- Multi-tenant architecture supports multiple clients
- Used in: All stages (0-7)

**Target**
- Analysis subject within a client (hashtag, account, keyword)
- Examples: "#justdoit", "@nike", "running shoes"
- Target type determines discovery method in Stage 1
- Directory: `/data/clients/{client_id}/targets/{target}/`
- Used in: Stages 0, 1, 7

**Cluster Mode (NEW)**
- Multi-hashtag scraping strategy for hashtag analysis
- Combines semantically related hashtags to maximize unique video discovery
- Example: `nutrition` cluster = #nutrition + #nutritionist + #nutritiontips + #nutritioncoach
- Provides 2-3x more unique videos than single hashtag scraping
- Configuration file: `/config/hashtag_clusters/{cluster_id}.json`
- Used in: Stage 1 (Video Discovery) for hashtag analysis only

**Narrow Semantic Clustering (NEW)**
- Strategy for selecting related hashtags with 20-30% overlap
- Not too broad (> 30% overlap = redundant data)
- Not too narrow (< 20% overlap = unrelated content)
- Goal: Maximize unique video discovery while maintaining semantic relevance
- Used in: Cluster configuration design (Stage 1 input)

**Provenance Tracking (NEW)**
- System capability to track which hashtags/runs found each video
- Fields added to video metadata: `source_hashtags` (list), `source_runs` (list)
- Example: Video found by #nutrition run 1 AND #nutritiontips run 2 → `source_hashtags: ["#nutrition", "#nutritiontips"]`, `source_runs: [1, 2]`
- Purpose: Enable cluster optimization (identify low-contributing hashtags)
- Used in: Stage 1 (Video Discovery), cluster analytics generation

**Cluster Analytics (NEW)**
- Health metrics report generated after cluster scraping
- 5 sections: scrape summary, per-hashtag contribution, pairwise overlaps, run effectiveness, bucket distribution by source
- Output file: `/data/clients/{client_id}/hashtag/{cluster_id}/cluster_analytics.json`
- Use cases: Identify low-contributing hashtags, optimize run count, diagnose bucket deficiencies
- Schema: ClusterAnalyticsSchema (Section 5.5)
- Used in: Stage 1 (Video Discovery), cluster optimization

### ML-Specific Terms

**RF (Random Forest)**
- Supervised learning model for feature importance ranking
- Input: transformed features + target variable (contrastive strategy)
- Output: Feature importance scores (which temporal windows drive engagement)
- Used in: Stages 4, 5, 6, 7

**KM (K-Means)**
- Unsupervised clustering algorithm for pattern discovery
- Input: scaled features (no target variable)
- Output: 3 cluster labels + cluster characteristics
- Three cluster strategies: Pattern Archetype, Engagement Archetypes, Hybrid Synthesis
- Used in: Stages 4, 5, 6, 7

**Feature Scaling**
- Normalization technique transforming features to [0,1] range
- Methods: MinMaxScaler for rates, log1p + MinMax for counts
- Purpose: Equalizes feature magnitudes for K-Means clustering
- Used in: Stage 4

**Feature Importance**
- ML-derived scores indicating which features predict engagement
- Source: Random Forest model (Gini importance or permutation importance)
- Output: Ranked list of temporal window features
- Used in: Stage 6 (RF reports), Stage 7 (LLM synthesis)

**BUCKET_WINDOWS**
- Centralized configuration dictionary mapping bucket names to their window structures
- Location: `config/bucket_definitions.py`
- Format: `{'0-3s': ['hook'], '3-9s': ['hook', 'closing'], ...}`
- Purpose: Single source of truth for bucket-specific window configurations across all stages
- Used by: Stage 4 (Feature Transformation), Stage 6 (ML Analysis Generation), Stage 7 (LLM Analysis)
- Prevents configuration desync between stages

### System Architecture Terms

**Stage**
- A discrete processing step in the ML pipeline
- 8 stages total: 0 (Config), 1 (Discovery), 2 (Processing), 3 (Aggregation), 4 (Transformation), 5 (Training), 6 (Analysis), 7 (Reporting)
- Each stage has: input dependencies, output contracts, checkpoint support
- Sequential execution enforced by rumiai_runner.py
- Used in: All documentation, code, and configuration

**Analysis Mode**
- Processing mode: training vs prediction
- Training mode: Uses curated video lists, trains new models
- Prediction mode: Uses new videos, applies existing models (future feature)
- Current implementation: Training mode only
- Used in: Stages 0, 5

**Apify**
- Third-party web scraping service for TikTok video discovery
- Provides: video metadata (URLs, engagement metrics, create_time, duration)
- Rate limits: Managed by Apify subscription tier
- Used in: Stage 1 (VideoDiscovery)

**config.json**
- Central configuration file storing all pipeline parameters
- Location: `/data/clients/{client_id}/targets/{target}/mode_{mode}_strategy_{strategy}/config.json`
- Schema: See Section 5.1 (Configuration Schemas)
- Created by: Stage 0 (CLI invocation)
- Used in: All stages (0-7)

### Abbreviations

**HLD**: High-Level Design
**TI**: Technical Implementation
**ML**: Machine Learning
**LLM**: Large Language Model
**API**: Application Programming Interface
**CSV**: Comma-Separated Values
**JSON**: JavaScript Object Notation
**CLI**: Command-Line Interface
**GCS**: Google Cloud Storage (potential future use)

---

## Document Metadata

**Creation Date**: 2025-01-28
**Last Modified**: 2025-01-28
**Authors**: RumiAI Team
**Reviewers**: [Pending]
**Approved By**: [Pending]
**Next Review Date**: [Pending]

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.1 | 2025-01-28 | RumiAI Team | Updated Section 1.3: Corrected ML model count from 16 to 90 models (detailed architecture breakdown added) |
| 1.0 | 2025-01-28 | RumiAI Team | Initial creation from MLPlanningv2.md Parts 1 & 2 |
