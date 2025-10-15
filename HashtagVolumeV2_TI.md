# Hashtag Volume Strategy V2 - Technical Implementation

> **TI Document**: HashtagVolumeV2_TI.md
> **Generated From**: HashtagVolumeV2.md (TI_Loose method)
> **Source Document Type**: Decision-based
> **Foundation Document**: FoundationCHILD.md
> **Child Documents**: VideoDiscoveryCHILD.md, VideoDiscoveryCHILDTI.md
> **Generation Mode**: Mode B - Extension
> **Generated**: 2025-10-13

---

## Section 1: Document Metadata & Scope

### Feature Name
Narrow Semantic Clustering Strategy for Hashtag Scraping

### Source Document
HashtagVolumeV2.md

### Foundation Document
FoundationCHILD.md

### Child Documents Referenced
- VideoDiscoveryCHILD.md (Stage 1: Video Discovery & Selection)
- VideoDiscoveryCHILDTI.md (Stage 1 Technical Implementation)

### Dependencies
- **FoundationTI.md** (REQUIRED - all stages depend on Foundation)
  - Section 2: Client Architecture & Directory Structure
  - Section 4: CLI Command Structure
  - Section 5: Configuration Schemas
- **VideoDiscoveryCHILDTI.md** (REQUIRED - provides baseline Stage 1 implementation)
  - Extends: Apify scraping logic
  - Modifies: Scraping strategy from single hashtag to cluster orchestration
  - Adds: Deduplication with provenance tracking
  - Adds: Cluster analytics generation

### Feeds Into
- VideoProcessingTI.md (Stage 2) - Consumes selected videos from cluster scraping
- MLModelTrainingTI.md (Stage 5) - Uses cluster-sourced videos for training

### Implementation Priority
CRITICAL

### Rationale
**Business Problem**: US geographic filtering reduces hashtag video volume by 57%, making it impossible to achieve 50+ videos per bucket for contrastive ML analysis with single-hashtag scraping.

**Solution**: Narrow Semantic Clustering Strategy
- Use 4 semantically related hashtags per target niche (e.g., #nutrition, #nutritionist, #nutritiontips, #nutritioncoach)
- Run 2 scrapes per hashtag with 2-minute delays
- Total: 8 scrapes per niche
- Validated Results: 777 unique videos (18.6% overlap) from single run → Projected 1,320-1,380 unique videos after 2nd run

### Implementation Scope

This TI implements 3 major functional changes to Stage 1 (Video Discovery):

1. **Cluster Configuration System** (DECISION 1)
   - New: `/config/hashtag_clusters/{cluster_id}.json` files
   - New: `generate_cluster.py` interactive cluster creation tool
   - Modified: CLI detection logic (cluster-first routing)

2. **Cluster Orchestration** (DECISION 2)
   - Extended: Stage 1 scraping logic for multi-hashtag execution
   - New: 8-scrape loop with 2-minute delays
   - New: Retry logic with exponential backoff (5s, 15s, 45s)
   - New: Progress logging per scrape

3. **Deduplication with Provenance** (DECISION 3)
   - Extended: Deduplication logic to track ALL source hashtags
   - New: Cluster analytics generation (`cluster_analytics.json`)
   - New: Per-hashtag contribution analysis
   - New: Pairwise overlap matrix

---

## Section 2: Data Schemas & Configuration

### 2.1 Foundation Schemas (Inherited)

These schemas are defined in FoundationCHILD.md and used across all stages.

```python
# Source: FoundationCHILD.md Section 5.1 (INHERITED - not modified by HashtagVolumeV2)
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
# Source: FoundationCHILD.md Section 5.1 (INHERITED)

# Source: FoundationCHILD.md Section 5.2 (INHERITED - not modified by HashtagVolumeV2)
ApifyVideoMetadataSchema = {
    "id": str,                     # Required, Unique video ID, Example: "7428596413707144481"
    "createTime": int,             # Required, Unix timestamp (UTC), Example: 1704067200
    "duration": int,               # Required, Seconds, Range: 3-120, Example: 25
    "playCount": int,              # Required, Views, >= 0, Example: 50000
    "shareCount": int,             # Required, Shares, >= 0, Example: 500
    "commentCount": int,           # Required, Comments, >= 0, Example: 250
    "likeCount": int,              # Required, Likes, >= 0, Example: 3500
    "webVideoUrl": str,            # Required, TikTok URL
    "videoMeta": {                 # Required
        "downloadAddr": str,       # Required, MP4 URL for download
    },
    "authorMeta": {                # Required
        "name": str,               # Required, Author username
    },
}
# Source: FoundationCHILD.md Section 5.2 (INHERITED)
```

### 2.2 New Schemas (Added by HashtagVolumeV2)

```python
# ===== CLUSTER CONFIGURATION SCHEMA (NEW) =====
# Source: HashtagVolumeV2.md DECISION 1, lines 213-234
# Location: /config/hashtag_clusters/{cluster_id}.json

ClusterConfigSchema = {
    "cluster_id": str,             # Required, unique identifier, alphanumeric + underscore
                                   # Example: "nutrition"
                                   # Validation: Regex ^[a-zA-Z0-9_]+$ (min 1 char)

    "description": str,            # Required, human-readable description
                                   # Example: "Nutrition niche - narrow semantic cluster"
                                   # Validation: min 1 char, max 500 chars

    "primary_hashtag": str,        # Required, original target hashtag
                                   # Format: starts with #, min 2 chars
                                   # Example: "#nutrition"
                                   # Validation: Regex ^#[a-zA-Z0-9_]+$ (min 2 chars)

    "variant_hashtags": list[str], # Required, array of 1-10 variant hashtags
                                   # Format: each starts with #, min 2 chars
                                   # Example: ["#nutritionist", "#nutritiontips", "#nutritioncoach"]
                                   # Validation:
                                   #   - Array length: 1-10
                                   #   - Each element: Regex ^#[a-zA-Z0-9_]+$ (min 2 chars)
                                   #   - No duplicates (case-insensitive)

    "scrape_config": {             # Required, scraping parameters
        "runs_per_hashtag": int,   # Required, number of scrapes per hashtag
                                   # Range: 1-5
                                   # Example: 2
                                   # Validation: 1 <= value <= 5

        "delay_between_runs_ms": int,  # Required, delay between scrapes in milliseconds
                                       # Range: 60000-600000 (1-10 minutes)
                                       # Example: 120000 (2 minutes)
                                       # Validation: 60000 <= value <= 600000

        "results_per_page": int,   # Required, videos per scrape
                                   # Range: 100-800
                                   # Example: 800
                                   # Validation: 100 <= value <= 800
    },

    "metadata": {                  # Optional, user metadata
        "created_date": str,       # Optional, ISO 8601 timestamp
                                   # Example: "2025-10-09T10:30:00Z"

        "created_by": str,         # Optional, creator username
                                   # Example: "jorge"

        "notes": str,              # Optional, additional notes
                                   # Example: "Validated with 18.6% overlap in Option 1 test"
    }
}
# Source: HashtagVolumeV2.md DECISION 1, lines 213-234 (NEW schema)
# File location: /config/hashtag_clusters/{cluster_id}.json
# Field count: 11 total fields (5 top-level + 3 scrape_config + 3 metadata)
# Required fields: cluster_id, description, primary_hashtag, variant_hashtags, scrape_config (all subfields)
# Validation: See Section 4 for validate_cluster_config() logic

# ===== CLUSTER ANALYTICS SCHEMA (NEW) =====
# Source: HashtagVolumeV2.md DECISION 3, lines 587-686
# Location: /data/{client}/hashtag/{cluster_id}/cluster_analytics.json

ClusterAnalyticsSchema = {
    "cluster_id": str,             # Required, cluster identifier
                                   # Example: "nutrition"

    "execution_date": str,         # Required, ISO 8601 timestamp
                                   # Example: "2025-10-10T14:30:00Z"

    "scrape_summary": {            # Required, overall scraping statistics
        "total_scrapes_attempted": int,  # Required, number of scrape attempts
                                         # Example: 8 (4 hashtags × 2 runs)

        "total_scrapes_succeeded": int,  # Required, successful scrapes
                                         # Example: 8

        "total_scraped_videos": int,     # Required, total videos scraped (before deduplication)
                                         # Example: 1939

        "total_unique_videos": int,      # Required, unique videos (after deduplication)
                                         # Example: 1400

        "overall_duplication_rate": float,  # Required, percentage of duplicates
                                           # Example: 27.8
                                           # Calculation: (total_scraped - total_unique) / total_scraped * 100

        "failed_scrapes": list[dict],    # Required, list of failed scrape details
                                         # Example: [] (empty if all succeeded)
                                         # Schema per failure: {"hashtag": str, "run": int, "error": str}
    },

    "per_hashtag_contribution": dict[str, dict],  # Required, per-hashtag statistics
                                                  # Key: hashtag name (e.g., "#nutrition")
                                                  # Value: contribution details (see schema below)
    # Per-hashtag contribution schema:
    # {
    #   "total_found": int,             # Total videos found by this hashtag (across all runs)
    #   "unique_videos": int,           # Videos unique to this hashtag (no overlap)
    #   "overlap_videos": int,          # Videos shared with other hashtags
    #   "exclusive_videos": int,        # Videos found ONLY by this hashtag
    #   "contribution_percentage": float # Percentage of total unique videos
    # }
    # Example:
    # "#nutrition": {
    #   "total_found": 500,
    #   "unique_videos": 380,
    #   "overlap_videos": 120,
    #   "exclusive_videos": 260,
    #   "contribution_percentage": 27.1
    # }

    "pairwise_overlaps": dict[str, float],  # Required, overlap percentages between hashtag pairs
                                            # Key: "{hashtag1}_vs_{hashtag2}" (alphabetical order)
                                            # Value: overlap percentage
                                            # Example: {"nutrition_vs_nutritionist": 18.2}

    "run_effectiveness": dict[str, dict],   # Required, effectiveness of multiple runs per hashtag
                                            # Key: hashtag name
                                            # Value: run comparison details
    # Per-hashtag run effectiveness schema:
    # {
    #   "run_1_videos": int,            # Videos from first run
    #   "run_2_videos": int,            # Videos from second run
    #   "run_2_new_videos": int,        # NEW videos from second run (not in first)
    #   "run_2_new_percentage": float   # Percentage of new videos in second run
    # }
    # Example:
    # "#nutrition": {
    #   "run_1_videos": 250,
    #   "run_2_videos": 250,
    #   "run_2_new_videos": 130,
    #   "run_2_new_percentage": 52.0
    # }

    "bucket_distribution_by_source": dict[str, dict],  # Required, bucket-level source tracking
                                                       # Key: bucket name (e.g., "60-90s")
                                                       # Value: bucket contribution details
    # Per-bucket distribution schema:
    # {
    #   "total_videos": int,            # Total videos in this bucket
    #   "by_hashtag": dict[str, int]    # Videos per hashtag
    # }
    # Example:
    # "60-90s": {
    #   "total_videos": 115,
    #   "by_hashtag": {
    #     "#nutrition": 42,
    #     "#nutritionist": 38,
    #     "#nutritiontips": 22,
    #     "#nutritioncoach": 13
    #   }
    # }
}
# Source: HashtagVolumeV2.md DECISION 3, lines 587-686 (NEW schema)
# File location: /data/{client}/hashtag/{cluster_id}/cluster_analytics.json
# Purpose: Cluster health monitoring, optimization, root cause analysis
```

### 2.3 Extended Schemas (Modified by HashtagVolumeV2)

```python
# ===== EXTENDED VIDEO METADATA SCHEMA =====
# Source: VideoDiscoveryCHILDTI.md Section 3.4 (BASE) + HashtagVolumeV2.md DECISION 3 (EXTENSION)

ExtendedVideoMetadataSchema = {
    # ===== INHERITED from VideoDiscoveryCHILDTI.md =====
    "id": str,                     # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)
    "createTime": int,             # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)
    "duration": int,               # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)
    "playCount": int,              # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)
    "shareCount": int,             # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)
    "commentCount": int,           # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)
    "likeCount": int,              # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)
    "webVideoUrl": str,            # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)
    "videoMeta": dict,             # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)
    "authorMeta": dict,            # Source: VideoDiscoveryCHILDTI.md Section 3.4 (INHERITED)

    # ===== EXTENDED by HashtagVolumeV2.md DECISION 3 =====
    "source_hashtags": list[str],  # Source: HashtagVolumeV2.md DECISION 3, lines 746-786 (NEW - cluster provenance)
                                   # List of hashtags that found this video
                                   # Example: ["#nutrition", "#nutritiontips"]
                                   # Purpose: Track which hashtags contributed to finding this video
                                   # Updated during deduplication: appends new hashtag if duplicate found

    "source_runs": list[int],      # Source: HashtagVolumeV2.md DECISION 3, lines 746-786 (NEW - run tracking)
                                   # List of run numbers that found this video
                                   # Example: [1, 2]
                                   # Purpose: Track which scrape runs found this video
                                   # Updated during deduplication: appends new run if duplicate found
}
# Schema extended: 10 fields INHERITED + 2 fields ADDED = 12 total
# Source: VideoDiscoveryCHILDTI.md Section 3.4 (base) + HashtagVolumeV2.md DECISION 3 (extension)
# Modification type: EXTENSION (adds fields, doesn't change existing)
```

### 2.4 Configuration Constants

**Source**: Constants imported from `src/config/config_constants.py`

```python
# Source: HashtagVolumeV2.md DECISION 1, DECISION 2, DECISION 3, DECISION 4

# ===== CLUSTER CONFIGURATION PATHS =====
CLUSTER_CONFIG_DIR = "/config/hashtag_clusters/"
# Source: HashtagVolumeV2.md DECISION 1, line 210

CLUSTER_CONFIG_PATH_TEMPLATE = "/config/hashtag_clusters/{cluster_id}.json"
# Source: HashtagVolumeV2.md DECISION 1, line 210

# ===== CLUSTER ANALYTICS PATHS =====
CLUSTER_ANALYTICS_PATH_TEMPLATE = "/data/{client}/hashtag/{cluster_id}/cluster_analytics.json"
# Source: HashtagVolumeV2.md DECISION 3, line 586

# ===== CLUSTER SCRAPING DEFAULTS =====
DEFAULT_RUNS_PER_HASHTAG = 2
# Source: HashtagVolumeV2.md DECISION 1, line 224
# Rationale: 2 runs balance data volume with cost

DEFAULT_DELAY_BETWEEN_RUNS_MS = 120000  # 2 minutes
# Source: HashtagVolumeV2.md DECISION 1, line 225
# Rationale: Avoid rate limiting, allow TikTok feed to refresh

DEFAULT_RESULTS_PER_PAGE = 800
# Source: HashtagVolumeV2.md DECISION 1, line 226
# Rationale: Maximum videos per Apify scrape

# ===== CLUSTER VALIDATION RANGES =====
MIN_VARIANT_HASHTAGS = 1
MAX_VARIANT_HASHTAGS = 10
# Source: HashtagVolumeV2.md DECISION 1, line 238
# Rationale: 1-10 variants provide flexibility while keeping overlap manageable

MIN_RUNS_PER_HASHTAG = 1
MAX_RUNS_PER_HASHTAG = 5
# Source: HashtagVolumeV2.md DECISION 1, lines 335-336
# Rationale: 1-5 runs provide flexibility (1=quick test, 5=maximum data)

MIN_DELAY_BETWEEN_RUNS_MS = 60000   # 1 minute
MAX_DELAY_BETWEEN_RUNS_MS = 600000  # 10 minutes
# Source: HashtagVolumeV2.md DECISION 1, lines 335-336
# Rationale: 1-10 minutes balances rate limiting with execution speed

MIN_RESULTS_PER_PAGE = 100
MAX_RESULTS_PER_PAGE = 800
# Source: HashtagVolumeV2.md DECISION 1, lines 335-336
# Rationale: 100-800 provides flexibility (100=quick test, 800=maximum data)

# ===== RETRY CONFIGURATION =====
RETRY_MAX_ATTEMPTS = 3
# Source: HashtagVolumeV2.md DECISION 2, line 445
# Rationale: 3 retries balance reliability with execution speed

RETRY_BACKOFF_DELAYS = [5, 15, 45]  # seconds
# Source: HashtagVolumeV2.md DECISION 2, line 445
# Rationale: Exponential backoff (5s, 15s, 45s) handles transient network issues
```

---

## Section 3: Implementation Specifications

### 3.0 Required Imports

All functions in this section require the following imports:

```python
# Source: HashtagVolumeV2.md Implementation Requirements

# ===== Standard Library =====
import os                        # File path operations, directory creation
import json                      # JSON config/analytics file I/O
import re                        # Regex validation (cluster_id, hashtag format)
import time                      # Scrape delays (time.sleep)
from datetime import datetime, timezone, timedelta  # Timestamps, date filtering

# ===== Type Hints =====
from typing import Tuple, List, Dict, Optional

# ===== Third-Party =====
from apify_client import ApifyClient  # Apify SDK (inherited from VideoDiscoveryCHILDTI.md)

# ===== Internal Imports =====
from src.config.config_constants import (
    CLUSTER_CONFIG_PATH_TEMPLATE,
    CLUSTER_ANALYTICS_PATH_TEMPLATE,
    RETRY_MAX_ATTEMPTS,
    RETRY_BACKOFF_DELAYS,
    MIN_VARIANT_HASHTAGS,
    MAX_VARIANT_HASHTAGS,
    MIN_RUNS_PER_HASHTAG,
    MAX_RUNS_PER_HASHTAG,
    MIN_DELAY_BETWEEN_RUNS_MS,
    MAX_DELAY_BETWEEN_RUNS_MS,
    MIN_RESULTS_PER_PAGE,
    MAX_RESULTS_PER_PAGE,
)

from src.config.error_codes import (
    EXIT_CODE_CLUSTER_CONFIG_NOT_FOUND,
    EXIT_CODE_CLUSTER_CONFIG_INVALID,
    EXIT_CODE_SINGLE_HASHTAG_DEPRECATED,
    EXIT_CODE_ALL_SCRAPES_FAILED,
)

from src.stage1.apify_client import call_apify_scraper  # Inherited from VideoDiscoveryCHILDTI.md

# ===== Logging =====
import logging
logger = logging.getLogger(__name__)
```

**Note**: All functions in Section 3.1 through 3.7 assume these imports are available.

---

### 3.1 Function 1: Load Cluster Configuration

```python
# Source: HashtagVolumeV2.md DECISION 1, lines 210-234, 311-322
# Extends: VideoDiscoveryCHILDTI.md target handling logic

def load_cluster_config(cluster_id: str) -> dict:
    """
    Load cluster configuration from JSON file.

    Source: HashtagVolumeV2.md DECISION 1, lines 210-234

    Args:
        cluster_id: str, cluster identifier (e.g., "nutrition")
                    Validation: alphanumeric + underscore only

    Returns:
        dict: Cluster configuration object
              Schema: ClusterConfigSchema (Section 2.2)

    Raises:
        FileNotFoundError: if cluster config file doesn't exist
        ValueError: if cluster config validation fails (see Section 4.1)
        json.JSONDecodeError: if config file is invalid JSON
    """
    # Build config file path
    cluster_path = CLUSTER_CONFIG_PATH_TEMPLATE.format(cluster_id=cluster_id)
    # Example: "/config/hashtag_clusters/nutrition.json"

    # Check file exists
    if not os.path.exists(cluster_path):
        raise FileNotFoundError(
            f"Cluster config not found: {cluster_path}\n"
            f"Create cluster config with: python generate_cluster.py"
        )

    # Load JSON
    try:
        with open(cluster_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in cluster config: {cluster_path}\n"
            f"Error: {str(e)}"
        )

    # Validate config schema (see Section 4.1)
    validate_cluster_config(config, cluster_path)

    logger.info(f"Loaded cluster config: {cluster_id}")
    logger.info(f"  Primary: {config['primary_hashtag']}")
    logger.info(f"  Variants: {len(config['variant_hashtags'])} hashtags")
    logger.info(f"  Scrape config: {config['scrape_config']['runs_per_hashtag']} runs × "
                f"{len(config['variant_hashtags']) + 1} hashtags = "
                f"{(len(config['variant_hashtags']) + 1) * config['scrape_config']['runs_per_hashtag']} scrapes")

    return config
```

### 3.2 Function 2: Cluster Scraping Orchestration

```python
# Source: HashtagVolumeV2.md DECISION 2, lines 409-441
# Extends: VideoDiscoveryCHILDTI.md scrape_videos() function

def run_cluster_scraping(cluster_config: dict, analysis_mode: str, country_code: str) -> tuple[list[dict], list[dict]]:
    """
    Orchestrate multi-hashtag, multi-run scraping with error recovery and progress logging.

    EXTENDS: VideoDiscoveryCHILDTI.md scrape_videos() function
    - ADDED: Multi-hashtag support (cluster_config.all_hashtags)
    - ADDED: Multi-run orchestration with delays
    - ADDED: Provenance tracking (source_hashtags, source_runs)
    - ADDED: Progress logging per scrape
    - ADDED: Retry logic with exponential backoff
    - PRESERVED: Apify scraper selection logic (from VideoDiscoveryCHILDTI.md)

    Source: HashtagVolumeV2.md DECISION 2, lines 409-441

    Args:
        cluster_config: dict, loaded from load_cluster_config()
                       Schema: ClusterConfigSchema (Section 2.2)
        analysis_mode: str, "top" or "recent" (from CLI parameter)
        country_code: str, "US" or "BR" or "global" (from CLI parameter)

    Returns:
        tuple[list[dict], list[dict]]:
            - all_videos: All scraped videos with provenance tracking
                         Each video has ExtendedVideoMetadataSchema (Section 2.3)
                         Videos NOT deduplicated yet (deduplication happens in Section 3.4)
            - failed_scrapes: List of failed scrape details
                             Schema: [{"hashtag": str, "run": int, "error": str}, ...]
                             Used by generate_cluster_analytics() for accurate reporting

    Raises:
        ValueError: if cluster_config invalid (caught before this function)
        ApifyError: if all scrapes fail (logged, returns ([], failed_scrapes))
    """
    # Extract cluster parameters
    cluster_id = cluster_config['cluster_id']
    all_hashtags = [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']
    runs_per_hashtag = cluster_config['scrape_config']['runs_per_hashtag']
    delay_ms = cluster_config['scrape_config']['delay_between_runs_ms']
    results_per_page = cluster_config['scrape_config']['results_per_page']

    # Calculate total scrapes
    total_scrapes = len(all_hashtags) * runs_per_hashtag

    logger.info(f"\nCluster: {cluster_id} ({len(all_hashtags)} hashtags × {runs_per_hashtag} runs = {total_scrapes} scrapes)\n")

    # Initialize results
    all_videos = []
    scrape_num = 0
    failed_scrapes = []

    # Loop: hashtags × runs
    for hashtag in all_hashtags:
        for run in range(1, runs_per_hashtag + 1):
            scrape_num += 1
            print(f"[{scrape_num}/{total_scrapes}] Scraping {hashtag} (run {run})...", end=" ", flush=True)

            # Error recovery: Retry 3x with exponential backoff
            videos = scrape_with_retry(
                hashtag=hashtag,
                run_num=run,
                analysis_mode=analysis_mode,
                country_code=country_code,
                results_per_page=results_per_page,
                max_retries=RETRY_MAX_ATTEMPTS
            )

            if videos:
                print(f"✅ {len(videos)} videos")

                # Tag videos with source provenance
                for video in videos:
                    video['source_hashtags'] = [hashtag]  # Initialize as list
                    video['source_runs'] = [run]          # Initialize as list

                # Append to master list (provenance added by loop above)
                all_videos.extend(videos)
            else:
                print(f"❌ Failed after {RETRY_MAX_ATTEMPTS} retries")
                failed_scrapes.append({
                    "hashtag": hashtag,
                    "run": run,
                    "error": "Failed after retries"
                })

            # Delay between scrapes (except after last scrape)
            if scrape_num < total_scrapes:
                delay_seconds = delay_ms / 1000
                print(f"    ({delay_seconds / 60:.0f} min delay)", flush=True)
                time.sleep(delay_seconds)

    # Log summary
    logger.info(f"\nScraping complete: {len(all_videos)} videos from {total_scrapes} scrapes")
    if failed_scrapes:
        logger.warning(f"⚠️  {len(failed_scrapes)} scrape(s) failed:")
        for failure in failed_scrapes:
            logger.warning(f"   - {failure['hashtag']} run {failure['run']}: {failure['error']}")

    return all_videos, failed_scrapes
```

### 3.3 Function 3: Scrape with Retry Logic

```python
# Source: HashtagVolumeV2.md DECISION 2, lines 444-466
# NEW function (not in VideoDiscoveryCHILDTI.md)

def scrape_with_retry(hashtag: str, run_num: int, analysis_mode: str, country_code: str,
                      results_per_page: int, max_retries: int = 3) -> list[dict]:
    """
    Scrape single hashtag with automatic retry on failure.

    Source: HashtagVolumeV2.md DECISION 2, lines 444-466

    Retry policy: Exponential backoff (5s, 15s, 45s)
    Failure handling: Skip scrape after max retries, continue with cluster

    Args:
        hashtag: str, hashtag to scrape (e.g., "#nutrition")
        run_num: int, run number (1-5)
        analysis_mode: str, "top" or "recent"
        country_code: str, "US" or "BR" or "global"
        results_per_page: int, videos per scrape (100-800)
        max_retries: int, maximum retry attempts (default: 3)

    Returns:
        list[dict]: Scraped videos (ApifyVideoMetadataSchema)
                    Empty list if all retries fail

    Raises:
        None - all errors caught and logged, returns [] on failure
    """
    backoff_delays = RETRY_BACKOFF_DELAYS  # [5, 15, 45] seconds

    for attempt in range(max_retries):
        try:
            # Call Apify scraper (inherited from VideoDiscoveryCHILDTI.md)
            videos = call_apify_scraper(
                target=hashtag,
                analysis_type="hashtag",  # Always hashtag for cluster scraping
                analysis_mode=analysis_mode,
                country_code=country_code,
                results_per_page=results_per_page
            )

            # Success - return immediately
            return videos

        except ApifyError as e:
            if attempt < max_retries - 1:
                # Retry with backoff
                delay = backoff_delays[attempt]
                logger.warning(f"  Retry {attempt+1}/{max_retries} in {delay}s... (Error: {str(e)})")
                time.sleep(delay)
            else:
                # All retries exhausted
                logger.error(f"  Skipping {hashtag} run {run_num} after {max_retries} failed attempts")
                logger.error(f"  Final error: {str(e)}")
                return []  # Return empty list, continue with remaining scrapes

        except Exception as e:
            # Unexpected error - log and fail immediately
            logger.error(f"  Unexpected error scraping {hashtag} run {run_num}: {str(e)}")
            return []

    # Should never reach here (return in loop), but safety fallback
    return []
```

### 3.4 Function 4: Deduplication with Provenance Tracking

```python
# Source: HashtagVolumeV2.md DECISION 3, lines 746-786
# Extends: VideoDiscoveryCHILDTI.md deduplication logic

def deduplicate_with_provenance(all_videos: list[dict], cluster_config: dict, failed_scrapes: list[dict]) -> tuple[list[dict], dict]:
    """
    Deduplicate videos while tracking full source provenance.

    EXTENDS: VideoDiscoveryCHILDTI.md deduplication logic
    - PRESERVED: Video ID-based deduplication (keeps first occurrence)
    - ADDED: source_hashtags tracking (ALL hashtags that found this video)
    - ADDED: source_runs tracking (ALL runs that found this video)
    - ADDED: Cluster analytics generation

    Source: HashtagVolumeV2.md DECISION 3, lines 746-786

    Args:
        all_videos: list[dict], all scraped videos from run_cluster_scraping()
                    Example: 1,939 videos from 8 scrapes (4 hashtags × 2 runs)
                    Each video has: source_hashtags=[hashtag], source_runs=[run]
        cluster_config: dict, cluster configuration (for analytics generation)
                       Schema: ClusterConfigSchema (Section 2.2)
        failed_scrapes: list[dict], failed scrape details from run_cluster_scraping()
                       Schema: [{"hashtag": str, "run": int, "error": str}, ...]

    Returns:
        tuple:
            - unique_videos: list[dict], deduplicated videos with complete provenance
                           Example: 1,400 unique videos
                           Each video has: source_hashtags=[list of all hashtags], source_runs=[list of all runs]
            - analytics: dict, cluster health analytics
                        Schema: ClusterAnalyticsSchema (Section 2.2)

    Raises:
        ValueError: if all_videos is empty (all scrapes failed)
    """
    # DEFENSIVE CHECK: Handle empty input
    if len(all_videos) == 0:
        logger.error("No videos to deduplicate. All scrapes failed.")
        logger.error(f"Failed scrapes: {len(failed_scrapes)}/{len(failed_scrapes)}")
        for failure in failed_scrapes:
            logger.error(f"  - {failure['hashtag']} run {failure['run']}: {failure['error']}")

        raise ValueError(
            f"All {len(failed_scrapes)} scrapes failed. No videos available for analysis.\n"
            f"Check Apify API key, network connectivity, and target validity.\n"
            f"Exit code: {EXIT_CODE_ALL_SCRAPES_FAILED}"
        )

    print(f"\nDeduplicating {len(all_videos)} videos...", end=" ", flush=True)

    unique_videos_map = {}

    for video in all_videos:
        video_id = video['id']
        source_hashtag = video['source_hashtags'][0]  # Current scrape's hashtag (single-element list)
        source_run = video['source_runs'][0]          # Current scrape's run (single-element list)

        if video_id not in unique_videos_map:
            # First occurrence - initialize tracking
            # video already has source_hashtags and source_runs from run_cluster_scraping()
            unique_videos_map[video_id] = video
        else:
            # Duplicate - append to tracking arrays
            existing = unique_videos_map[video_id]

            # Append source_hashtag if not already tracked
            if source_hashtag not in existing['source_hashtags']:
                existing['source_hashtags'].append(source_hashtag)

            # Append source_run if not already tracked
            if source_run not in existing['source_runs']:
                existing['source_runs'].append(source_run)

    # Convert map to list
    unique_videos = list(unique_videos_map.values())

    # Calculate duplication rate
    duplication_rate = (len(all_videos) - len(unique_videos)) / len(all_videos) * 100

    print(f"✅ {len(unique_videos)} unique ({duplication_rate:.1f}% overlap)")

    # Generate cluster analytics (see Section 3.5)
    analytics = generate_cluster_analytics(all_videos, unique_videos, cluster_config, failed_scrapes)

    return unique_videos, analytics
```

### 3.5 Function 5: Generate Cluster Analytics

```python
# Source: HashtagVolumeV2.md DECISION 3, lines 587-686
# NEW function (not in VideoDiscoveryCHILDTI.md)

def generate_cluster_analytics(all_videos: list[dict], unique_videos: list[dict], cluster_config: dict, failed_scrapes: list[dict]) -> dict:
    """
    Generate cluster health analytics report.

    Source: HashtagVolumeV2.md DECISION 3, lines 587-686

    Args:
        all_videos: list[dict], all scraped videos (before deduplication)
        unique_videos: list[dict], deduplicated videos with provenance
        cluster_config: dict, cluster configuration (for calculating total attempts)
                       Schema: ClusterConfigSchema (Section 2.2)
        failed_scrapes: list[dict], failed scrape details from run_cluster_scraping()
                       Schema: [{"hashtag": str, "run": int, "error": str}, ...]

    Returns:
        dict: Cluster analytics report
              Schema: ClusterAnalyticsSchema (Section 2.2)

    Raises:
        None
    """
    # Extract cluster parameters
    all_hashtags = [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']
    runs_per_hashtag = cluster_config['scrape_config']['runs_per_hashtag']
    cluster_id = cluster_config['cluster_id']

    # Calculate scrape attempts from CONFIG (not derived from video data)
    total_scrapes_attempted = len(all_hashtags) * runs_per_hashtag
    total_scrapes_succeeded = total_scrapes_attempted - len(failed_scrapes)

    # ===== SCRAPE SUMMARY (FIXED) =====
    scrape_summary = {
        "total_scrapes_attempted": total_scrapes_attempted,      # From config
        "total_scrapes_succeeded": total_scrapes_succeeded,      # Calculated correctly
        "total_scraped_videos": len(all_videos),
        "total_unique_videos": len(unique_videos),
        "overall_duplication_rate": (len(all_videos) - len(unique_videos)) / len(all_videos) * 100 if all_videos else 0,
        "failed_scrapes": failed_scrapes  # Actually populated now!
    }

    # ===== PER-HASHTAG CONTRIBUTION =====
    per_hashtag_contribution = {}

    for hashtag in all_hashtags:
        # Videos found by this hashtag
        found_by_hashtag = [
            v for v in unique_videos if hashtag in v['source_hashtags']
        ]

        # Videos exclusive to this hashtag (not found by any other)
        exclusive_to_hashtag = [
            v for v in unique_videos
            if hashtag in v['source_hashtags'] and len(v['source_hashtags']) == 1
        ]

        # Calculate overlap (found by this hashtag AND at least one other)
        overlap_videos = len(found_by_hashtag) - len(exclusive_to_hashtag)

        per_hashtag_contribution[hashtag] = {
            "total_found": len(found_by_hashtag),
            "unique_videos": len(found_by_hashtag),  # Same as total_found (all unique)
            "overlap_videos": overlap_videos,
            "exclusive_videos": len(exclusive_to_hashtag),
            "contribution_percentage": len(found_by_hashtag) / len(unique_videos) * 100
        }

    # ===== PAIRWISE OVERLAPS =====
    pairwise_overlaps = {}

    for i, hashtag1 in enumerate(all_hashtags):
        for hashtag2 in all_hashtags[i+1:]:  # Only pairs (no self-comparison)
            # Videos found by BOTH hashtags
            overlap = [
                v for v in unique_videos
                if hashtag1 in v['source_hashtags'] and hashtag2 in v['source_hashtags']
            ]

            # Smaller set size (for percentage calculation)
            set1_size = len([v for v in unique_videos if hashtag1 in v['source_hashtags']])
            set2_size = len([v for v in unique_videos if hashtag2 in v['source_hashtags']])
            smaller_size = min(set1_size, set2_size)

            # Overlap percentage
            overlap_pct = len(overlap) / smaller_size * 100 if smaller_size > 0 else 0

            # Key format: alphabetical order, underscores instead of hashes
            key = f"{hashtag1.replace('#', '')}_{hashtag2.replace('#', '')}"
            pairwise_overlaps[key] = round(overlap_pct, 1)

    # ===== RUN EFFECTIVENESS =====
    run_effectiveness = {}

    for hashtag in all_hashtags:
        # Get all runs for this hashtag
        runs = sorted(list(set([
            run for v in unique_videos
            if hashtag in v['source_hashtags']
            for run in v['source_runs']
        ])))

        if len(runs) >= 2:
            # Videos from run 1
            run_1_videos = [
                v for v in unique_videos
                if hashtag in v['source_hashtags'] and 1 in v['source_runs']
            ]

            # Videos from run 2
            run_2_videos = [
                v for v in unique_videos
                if hashtag in v['source_hashtags'] and 2 in v['source_runs']
            ]

            # Videos NEW in run 2 (not in run 1)
            run_2_new = [
                v for v in run_2_videos
                if 1 not in v['source_runs']  # Only found in run 2
            ]

            run_effectiveness[hashtag] = {
                "run_1_videos": len(run_1_videos),
                "run_2_videos": len(run_2_videos),
                "run_2_new_videos": len(run_2_new),
                "run_2_new_percentage": len(run_2_new) / len(run_2_videos) * 100 if run_2_videos else 0
            }

    # ===== BUCKET DISTRIBUTION BY SOURCE =====
    # (Optional - only if bucket assignment happens in Stage 1)
    bucket_distribution_by_source = {}
    # NOTE: This section depends on whether Stage 1 assigns buckets
    # Source: HashtagVolumeV2.md DECISION 3, lines 656-684
    # If buckets assigned in Stage 1, implement bucket distribution tracking
    # If buckets assigned in Stage 2, skip this section

    # ===== ASSEMBLE ANALYTICS =====
    analytics = {
        "cluster_id": cluster_id,  # From cluster_config parameter (not derived from video data)
        "execution_date": datetime.now(timezone.utc).isoformat(),
        "scrape_summary": scrape_summary,
        "per_hashtag_contribution": per_hashtag_contribution,
        "pairwise_overlaps": pairwise_overlaps,
        "run_effectiveness": run_effectiveness,
        "bucket_distribution_by_source": bucket_distribution_by_source
    }

    return analytics
```

### 3.6 Function 6: CLI Detection Logic (Modified)

```python
# Source: HashtagVolumeV2.md DECISION 1, lines 311-322
# Modifies: VideoDiscoveryCHILDTI.md target handling logic

def detect_target_type(target: str, analysis_type: str) -> tuple[str, dict]:
    """
    Detect if target is a cluster or single hashtag/profile.

    MODIFIES: VideoDiscoveryCHILDTI.md target handling logic
    - ADDED: Cluster detection (if target doesn't start with # or @)
    - ADDED: Cluster config loading
    - PRESERVED: Single hashtag/profile handling
    - ADDED: Error on single hashtag (deprecated per DECISION 6)

    Source: HashtagVolumeV2.md DECISION 1, lines 311-322, DECISION 6

    Args:
        target: str, target from CLI parameter (e.g., "nutrition" or "#nutrition")
        analysis_type: str, "hashtag" or "competitor" or "creator"

    Returns:
        tuple:
            - target_type: str, "cluster" or "single"
            - config: dict, cluster config if cluster, None if single

    Raises:
        ValueError: if single hashtag used (deprecated per DECISION 6)
        FileNotFoundError: if cluster config not found
    """
    if analysis_type == "hashtag":
        if target.startswith("#"):
            # Single hashtag mode (DEPRECATED per DECISION 6)
            raise ValueError(
                f"Single hashtag scraping is deprecated as of 2025-10-10.\n"
                f"Please create a cluster configuration:\n"
                f"  1. Run: python generate_cluster.py\n"
                f"  2. Enter primary hashtag: {target[1:]}\n"
                f"  3. Configure cluster settings\n"
                f"  4. Run: python rumiai_ml_batch.py --target {target[1:]}\n\n"
                f"Rationale: Cluster strategy provides 2-3x more unique videos "
                f"with rich analytics for optimization."
            )
        else:
            # Cluster mode (target is cluster name)
            cluster_config = load_cluster_config(target)  # See Section 3.1
            return ("cluster", cluster_config)

    else:
        # Competitor or Creator - single profile mode (unchanged)
        # Source: VideoDiscoveryCHILDTI.md (INHERITED)
        return ("single", None)
```

### 3.7 Function 7: Save Cluster Analytics

```python
# Source: HashtagVolumeV2.md DECISION 3, lines 587-686
# NEW function (not in VideoDiscoveryCHILDTI.md)

def save_cluster_analytics(analytics: dict, client_id: str, cluster_id: str):
    """
    Save cluster analytics to JSON file.

    Source: HashtagVolumeV2.md DECISION 3, lines 587-686

    Args:
        analytics: dict, cluster analytics report
                  Schema: ClusterAnalyticsSchema (Section 2.2)
        client_id: str, client identifier
        cluster_id: str, cluster identifier

    Returns:
        str: Path to saved analytics file

    Raises:
        OSError: if file write fails
    """
    # Build analytics file path
    analytics_path = CLUSTER_ANALYTICS_PATH_TEMPLATE.format(
        client=client_id,
        cluster_id=cluster_id
    )
    # Example: "/data/acme_corp/hashtag/nutrition/cluster_analytics.json"

    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(analytics_path), exist_ok=True)

    # Write analytics
    with open(analytics_path, 'w') as f:
        json.dump(analytics, f, indent=2)

    logger.info(f"Saved cluster analytics: {analytics_path}")

    return analytics_path
```

---

## Section 4: Validation & Error Handling

### 4.1 Cluster Configuration Validation

```python
# Source: HashtagVolumeV2.md DECISION 1, lines 238-257
# NEW validation function (not in VideoDiscoveryCHILDTI.md)

def validate_cluster_config(config: dict, config_path: str):
    """
    Validate cluster configuration schema and constraints.

    Source: HashtagVolumeV2.md DECISION 1, lines 238-257

    Args:
        config: dict, loaded cluster configuration
        config_path: str, path to config file (for error messages)

    Raises:
        ValueError: if validation fails (with specific error message)

    Returns:
        None (raises ValueError on failure)
    """
    # ===== REQUIRED FIELDS =====
    required_fields = [
        "cluster_id",
        "description",
        "primary_hashtag",
        "variant_hashtags",
        "scrape_config"
    ]

    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"Missing required field '{field}' in {config_path}\n"
                f"Schema: ClusterConfigSchema (see Section 2.2)"
            )

    # ===== CLUSTER_ID VALIDATION =====
    cluster_id = config['cluster_id']

    if not re.match(r'^[a-zA-Z0-9_]+$', cluster_id):
        raise ValueError(
            f"Invalid cluster_id: '{cluster_id}'\n"
            f"Must be alphanumeric + underscore only (regex: ^[a-zA-Z0-9_]+$)\n"
            f"Example: 'nutrition', 'fitness_tips'"
        )

    if len(cluster_id) < 1:
        raise ValueError(
            f"Invalid cluster_id: '{cluster_id}'\n"
            f"Must be at least 1 character"
        )

    # ===== DESCRIPTION VALIDATION =====
    description = config['description']

    if not isinstance(description, str) or len(description) < 1:
        raise ValueError(
            f"Invalid description: must be non-empty string"
        )

    if len(description) > 500:
        raise ValueError(
            f"Invalid description: exceeds 500 characters ({len(description)} chars)\n"
            f"Keep descriptions concise."
        )

    # ===== PRIMARY_HASHTAG VALIDATION =====
    primary = config['primary_hashtag']

    if not isinstance(primary, str):
        raise ValueError(f"Invalid primary_hashtag: must be string, got {type(primary)}")

    if not primary.startswith('#'):
        raise ValueError(
            f"Invalid primary_hashtag: '{primary}'\n"
            f"Must start with # (e.g., '#nutrition')"
        )

    if len(primary) < 2:
        raise ValueError(
            f"Invalid primary_hashtag: '{primary}'\n"
            f"Must be at least 2 characters (# + 1 char)"
        )

    if not re.match(r'^#[a-zA-Z0-9_]+$', primary):
        raise ValueError(
            f"Invalid primary_hashtag: '{primary}'\n"
            f"Must be alphanumeric + underscore only (regex: ^#[a-zA-Z0-9_]+$)\n"
            f"Example: '#nutrition', '#fitness_tips'"
        )

    # ===== VARIANT_HASHTAGS VALIDATION =====
    variants = config['variant_hashtags']

    if not isinstance(variants, list):
        raise ValueError(
            f"Invalid variant_hashtags: must be array, got {type(variants)}"
        )

    if len(variants) < MIN_VARIANT_HASHTAGS or len(variants) > MAX_VARIANT_HASHTAGS:
        raise ValueError(
            f"Invalid variant_hashtags: must have {MIN_VARIANT_HASHTAGS}-{MAX_VARIANT_HASHTAGS} variants, got {len(variants)}\n"
            f"Current variants: {variants}"
        )

    # Validate each variant
    for i, variant in enumerate(variants):
        if not isinstance(variant, str):
            raise ValueError(
                f"Invalid variant_hashtags[{i}]: must be string, got {type(variant)}"
            )

        if not variant.startswith('#'):
            raise ValueError(
                f"Invalid variant_hashtags[{i}]: '{variant}'\n"
                f"Must start with # (e.g., '#nutritionist')"
            )

        if len(variant) < 2:
            raise ValueError(
                f"Invalid variant_hashtags[{i}]: '{variant}'\n"
                f"Must be at least 2 characters (# + 1 char)"
            )

        if not re.match(r'^#[a-zA-Z0-9_]+$', variant):
            raise ValueError(
                f"Invalid variant_hashtags[{i}]: '{variant}'\n"
                f"Must be alphanumeric + underscore only (regex: ^#[a-zA-Z0-9_]+$)"
            )

    # Check for duplicates (case-insensitive)
    all_hashtags = [primary.lower()] + [v.lower() for v in variants]
    if len(all_hashtags) != len(set(all_hashtags)):
        duplicates = [h for h in all_hashtags if all_hashtags.count(h) > 1]
        raise ValueError(
            f"Duplicate hashtags found (case-insensitive): {list(set(duplicates))}\n"
            f"Each hashtag must be unique within cluster"
        )

    # ===== SCRAPE_CONFIG VALIDATION =====
    if 'scrape_config' not in config:
        raise ValueError(
            f"Missing required field 'scrape_config' in {config_path}"
        )

    scrape_config = config['scrape_config']

    # Required scrape_config fields
    required_scrape_fields = [
        "runs_per_hashtag",
        "delay_between_runs_ms",
        "results_per_page"
    ]

    for field in required_scrape_fields:
        if field not in scrape_config:
            raise ValueError(
                f"Missing required field 'scrape_config.{field}' in {config_path}"
            )

    # Validate runs_per_hashtag
    runs = scrape_config['runs_per_hashtag']
    if not isinstance(runs, int):
        raise ValueError(
            f"Invalid scrape_config.runs_per_hashtag: must be integer, got {type(runs)}"
        )

    if runs < MIN_RUNS_PER_HASHTAG or runs > MAX_RUNS_PER_HASHTAG:
        raise ValueError(
            f"Invalid scrape_config.runs_per_hashtag: must be {MIN_RUNS_PER_HASHTAG}-{MAX_RUNS_PER_HASHTAG}, got {runs}"
        )

    # Validate delay_between_runs_ms
    delay = scrape_config['delay_between_runs_ms']
    if not isinstance(delay, int):
        raise ValueError(
            f"Invalid scrape_config.delay_between_runs_ms: must be integer, got {type(delay)}"
        )

    if delay < MIN_DELAY_BETWEEN_RUNS_MS or delay > MAX_DELAY_BETWEEN_RUNS_MS:
        raise ValueError(
            f"Invalid scrape_config.delay_between_runs_ms: must be {MIN_DELAY_BETWEEN_RUNS_MS}-{MAX_DELAY_BETWEEN_RUNS_MS} ms, got {delay}\n"
            f"Acceptable range: {MIN_DELAY_BETWEEN_RUNS_MS/60000:.1f}-{MAX_DELAY_BETWEEN_RUNS_MS/60000:.1f} minutes"
        )

    # Validate results_per_page
    results = scrape_config['results_per_page']
    if not isinstance(results, int):
        raise ValueError(
            f"Invalid scrape_config.results_per_page: must be integer, got {type(results)}"
        )

    if results < MIN_RESULTS_PER_PAGE or results > MAX_RESULTS_PER_PAGE:
        raise ValueError(
            f"Invalid scrape_config.results_per_page: must be {MIN_RESULTS_PER_PAGE}-{MAX_RESULTS_PER_PAGE}, got {results}"
        )

    # ===== METADATA VALIDATION (OPTIONAL) =====
    if 'metadata' in config:
        metadata = config['metadata']

        # Validate created_date if present
        if 'created_date' in metadata:
            created_date = metadata['created_date']
            try:
                datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(
                    f"Invalid metadata.created_date: '{created_date}'\n"
                    f"Must be ISO 8601 format (e.g., '2025-10-09T10:30:00Z')"
                )

    # All validations passed
    logger.debug(f"✅ Cluster config validation passed: {cluster_id}")
```

### 4.2 Error Cases & Exit Codes

```python
# Source: HashtagVolumeV2.md DECISION 1, DECISION 2, DECISION 6
# Extends: VideoDiscoveryCHILDTI.md error handling

# ===== NEW ERROR CODES (HashtagVolumeV2-specific) =====
EXIT_CODE_CLUSTER_CONFIG_NOT_FOUND = 10
# Source: HashtagVolumeV2.md DECISION 1, line 311
# Trigger: load_cluster_config() raises FileNotFoundError
# Message: "Cluster config not found: {path}\nCreate cluster config with: python generate_cluster.py"

EXIT_CODE_CLUSTER_CONFIG_INVALID = 11
# Source: HashtagVolumeV2.md DECISION 1, line 238
# Trigger: validate_cluster_config() raises ValueError
# Message: (Specific validation error from Section 4.1)

EXIT_CODE_SINGLE_HASHTAG_DEPRECATED = 12
# Source: HashtagVolumeV2.md DECISION 6, lines 1052-1062
# Trigger: detect_target_type() detects single hashtag (target starts with #)
# Message: "Single hashtag scraping is deprecated as of 2025-10-10..."

EXIT_CODE_ALL_SCRAPES_FAILED = 13
# Source: HashtagVolumeV2.md DECISION 2, line 461
# Trigger: run_cluster_scraping() returns empty list (all scrapes failed after retries)
# Message: "All scrapes failed. Check Apify API key, network, and target validity."

# ===== INHERITED ERROR CODES (from VideoDiscoveryCHILDTI.md) =====
# These exit codes are preserved from VideoDiscoveryCHILDTI.md:
# - EXIT_CODE_MISSING_APIFY_KEY = 1
# - EXIT_CODE_INVALID_CLI_PARAMS = 2
# - EXIT_CODE_APIFY_TIMEOUT = 3
# - EXIT_CODE_INSUFFICIENT_VIDEOS = 6
# - EXIT_CODE_USER_ABORT = 130
# (See VideoDiscoveryCHILDTI.md Section 6 for full list)
```

### 4.3 Error Handling Table

```python
# Source: HashtagVolumeV2.md DECISION 1, DECISION 2, DECISION 3, DECISION 6

"""
Error Handling Reference

| Error Scenario | Detection | Handling | User Message | Exit Code |
|----------------|-----------|----------|--------------|-----------|
| **Cluster config not found** | load_cluster_config() FileNotFoundError | Fail-fast | "Cluster config not found: {path}\\nCreate cluster config with: python generate_cluster.py" | 10 |
| **Cluster config invalid JSON** | json.JSONDecodeError | Fail-fast | "Invalid JSON in cluster config: {path}\\nError: {details}" | 11 |
| **Cluster config schema invalid** | validate_cluster_config() ValueError | Fail-fast | (Specific validation error, e.g., "Invalid cluster_id...") | 11 |
| **Single hashtag used (deprecated)** | detect_target_type() detects # prefix | Fail-fast | "Single hashtag scraping is deprecated as of 2025-10-10...\\nPlease create a cluster configuration..." | 12 |
| **Single scrape fails (retry logic)** | scrape_with_retry() ApifyError | Retry 3x with backoff [5s, 15s, 45s], then skip scrape | "⚠️ Retry {attempt}/{max} in {delay}s... (Error: {e})" | 0 (warning) |
| **All scrapes in cluster fail** | run_cluster_scraping() returns [] | Fail-fast | "All scrapes failed. Check Apify API key, network, and target validity." | 13 |
| **Some scrapes fail (partial success)** | run_cluster_scraping() returns partial results | Continue, log warnings | "⚠️ {N} scrape(s) failed: - {hashtag} run {run}: {error}" | 0 (warning) |
| **Duplicate hashtags in cluster** | validate_cluster_config() checks uniqueness | Fail-fast | "Duplicate hashtags found (case-insensitive): {duplicates}" | 11 |
| **Invalid hashtag format** | validate_cluster_config() regex check | Fail-fast | "Invalid {field}: '{value}'\\nMust be alphanumeric + underscore only..." | 11 |
| **Runs per hashtag out of range** | validate_cluster_config() range check | Fail-fast | "Invalid scrape_config.runs_per_hashtag: must be {MIN}-{MAX}, got {value}" | 11 |
| **Delay out of range** | validate_cluster_config() range check | Fail-fast | "Invalid scrape_config.delay_between_runs_ms: must be {MIN}-{MAX} ms, got {value}" | 11 |
| **Results per page out of range** | validate_cluster_config() range check | Fail-fast | "Invalid scrape_config.results_per_page: must be {MIN}-{MAX}, got {value}" | 11 |
| **0 unique videos after deduplication** | deduplicate_with_provenance() checks length | Fail-fast | "No unique videos after deduplication. All {N} scraped videos were duplicates." | 6 |
| **Cluster analytics write fails** | save_cluster_analytics() OSError | Log error, continue | "⚠️ Failed to save cluster analytics: {error}\\nContinuing..." | 0 (warning) |
"""
```

### 4.4 Input Validation Summary

```python
# Source: HashtagVolumeV2.md DECISION 1, lines 238-257
# NEW validation (extends VideoDiscoveryCHILDTI.md input validation)

def validate_cluster_inputs(target: str, analysis_type: str):
    """
    Validate cluster-related CLI inputs before processing.

    Source: HashtagVolumeV2.md DECISION 1, DECISION 6

    EXTENDS: VideoDiscoveryCHILDTI.md validate_cli_params()
    - ADDED: Cluster target validation (no # or @ prefix)
    - ADDED: Cluster config existence check
    - ADDED: Single hashtag deprecation check

    Args:
        target: str, target from CLI parameter
        analysis_type: str, "hashtag" or "competitor" or "creator"

    Raises:
        ValueError: if validation fails
        FileNotFoundError: if cluster config not found

    Returns:
        None (raises exception on failure)
    """
    if analysis_type == "hashtag":
        # Check if single hashtag (deprecated)
        if target.startswith("#"):
            raise ValueError(
                f"Single hashtag scraping is deprecated as of 2025-10-10.\n"
                f"Please create a cluster configuration:\n"
                f"  1. Run: python generate_cluster.py\n"
                f"  2. Enter primary hashtag: {target[1:]}\n"
                f"  3. Configure cluster settings\n"
                f"  4. Run: python rumiai_ml_batch.py --target {target[1:]}\n\n"
                f"Rationale: Cluster strategy provides 2-3x more unique videos "
                f"with rich analytics for optimization."
            )

        # Validate cluster name format (alphanumeric + underscore)
        if not re.match(r'^[a-zA-Z0-9_]+$', target):
            raise ValueError(
                f"Invalid cluster name: '{target}'\n"
                f"Must be alphanumeric + underscore only (no # prefix)\n"
                f"Example: 'nutrition', 'fitness_tips'"
            )

        # Check cluster config exists
        cluster_path = CLUSTER_CONFIG_PATH_TEMPLATE.format(cluster_id=target)
        if not os.path.exists(cluster_path):
            raise FileNotFoundError(
                f"Cluster config not found: {cluster_path}\n"
                f"Create cluster config with: python generate_cluster.py"
            )

    # Competitor/Creator validation inherited from VideoDiscoveryCHILDTI.md
```

---

## Section 5: Integration & File Structure

### 5.1 Directory Structure (Extended)

```
# Source: FoundationCHILD.md Section 2.1 (BASE) + HashtagVolumeV2.md DECISION 1, DECISION 3 (EXTENSIONS)

/data/
├── clients/
│   └── {client_id}/                           # Source: FoundationCHILD.md Section 2.1 (INHERITED)
│       └── hashtags/                          # Source: FoundationCHILD.md Section 2.1 (INHERITED)
│           └── {cluster_id}/                  # Source: HashtagVolumeV2.md DECISION 1 (MODIFIED - cluster name instead of hashtag)
│               ├── cluster_analytics.json     # Source: HashtagVolumeV2.md DECISION 3 (NEW)
│               └── top_contrastive/           # Source: FoundationCHILD.md Section 2.1 (INHERITED)
│                   ├── config.json            # Source: FoundationCHILD.md Section 5.1 (INHERITED)
│                   ├── winner_analysis.json   # Source: VideoDiscoveryCHILD.md Section 5.3 (INHERITED)
│                   └── buckets/               # Source: FoundationCHILD.md Section 2.1 (INHERITED)
│                       ├── bucket_18-33s/     # Source: FoundationCHILD.md Section 2.1 (INHERITED)
│                       │   └── selected_videos.json  # Source: VideoDiscoveryCHILD.md Section 5.3 (INHERITED)
│                       │                             # Videos have EXTENDED schema (Section 2.3)
│                       ├── bucket_33-60s/
│                       └── bucket_60-90s/

/config/
└── hashtag_clusters/                          # Source: HashtagVolumeV2.md DECISION 1 (NEW directory)
    ├── nutrition.json                         # Source: HashtagVolumeV2.md DECISION 1 (NEW file type)
    ├── fitness.json
    └── wellness.json

/tools/                                         # Source: HashtagVolumeV2.md DECISION 4 (NEW directory)
└── generate_cluster.py                        # Cluster creation tool

/logs/                                          # Execution logs
└── cluster_scraping.log                       # Error logs, scraping logs

# KEY CHANGES:
# 1. NEW: /config/hashtag_clusters/ directory for cluster configs
# 2. MODIFIED: {cluster_id} replaces {target_sanitized} in hashtag path
# 3. NEW: cluster_analytics.json at cluster level
# 4. EXTENDED: selected_videos.json videos now include source_hashtags and source_runs
```

### 5.2 File Specifications

```python
# Source: HashtagVolumeV2.md DECISION 1, DECISION 3

# ===== FILE 1: Cluster Configuration =====
# Location: /config/hashtag_clusters/{cluster_id}.json
# Source: HashtagVolumeV2.md DECISION 1, lines 210-234
# Schema: ClusterConfigSchema (Section 2.2)
# Size: ~1-5 KB (small JSON config)
# Encoding: UTF-8
# Format: JSON (pretty-printed, 2-space indent)
# Created by: generate_cluster.py interactive tool
# Used by: Stage 1 (load_cluster_config function)
# Purpose: Define cluster hashtags and scraping parameters

cluster_config_file = {
    "path": "/config/hashtag_clusters/{cluster_id}.json",
    "example_path": "/config/hashtag_clusters/nutrition.json",
    "schema": "ClusterConfigSchema",
    "size_kb": "1-5",
    "created_by": "generate_cluster.py",
    "used_by": ["Stage 1 (Video Discovery)"],
}

# ===== FILE 2: Cluster Analytics =====
# Location: /data/{client}/hashtags/{cluster_id}/cluster_analytics.json
# Source: HashtagVolumeV2.md DECISION 3, lines 587-686
# Schema: ClusterAnalyticsSchema (Section 2.2)
# Size: ~10-50 KB (depends on cluster size)
# Encoding: UTF-8
# Format: JSON (pretty-printed, 2-space indent)
# Created by: Stage 1 (save_cluster_analytics function)
# Used by: Cluster health monitoring, optimization, debugging
# Purpose: Track hashtag contribution, overlap, run effectiveness

cluster_analytics_file = {
    "path": "/data/{client}/hashtags/{cluster_id}/cluster_analytics.json",
    "example_path": "/data/acme_corp/hashtags/nutrition/cluster_analytics.json",
    "schema": "ClusterAnalyticsSchema",
    "size_kb": "10-50",
    "created_by": "Stage 1 (deduplicate_with_provenance → save_cluster_analytics)",
    "used_by": ["Manual review", "Cluster optimization", "Root cause analysis"],
}

# ===== FILE 3: Selected Videos (Extended) =====
# Location: /data/{client}/hashtags/{cluster_id}/{mode}_{strategy}/buckets/bucket_{bucket}/selected_videos.json
# Source: VideoDiscoveryCHILD.md Section 5.3 (BASE) + HashtagVolumeV2.md DECISION 3 (EXTENSION)
# Schema: SelectedVideosSchema with ExtendedVideoMetadataSchema (Section 2.3)
# Size: ~100-500 KB per bucket (100-500 videos × ~1KB each)
# Encoding: UTF-8
# Format: JSON (pretty-printed, 2-space indent)
# Created by: Stage 1 (video selection logic)
# Used by: Stage 2 (Video Processing)
# Purpose: Videos selected for ML analysis with cluster provenance
# EXTENSION: Each video now includes source_hashtags and source_runs fields

selected_videos_file = {
    "path": "/data/{client}/hashtags/{cluster_id}/{mode}_{strategy}/buckets/bucket_{bucket}/selected_videos.json",
    "example_path": "/data/acme_corp/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/selected_videos.json",
    "schema": "SelectedVideosSchema (with ExtendedVideoMetadataSchema)",
    "size_kb": "100-500",
    "created_by": "Stage 1 (video selection logic)",
    "used_by": ["Stage 2 (Video Processing)"],
    "extension": "Videos include source_hashtags and source_runs fields",
}

# ===== FILE 4: Winner Analysis (Unchanged) =====
# Location: /data/{client}/hashtags/{cluster_id}/{mode}_{strategy}/winner_analysis.json
# Source: VideoDiscoveryCHILD.md Section 5.3 (INHERITED)
# Schema: WinnerAnalysisSchema (unchanged)
# Size: ~5 KB
# Encoding: UTF-8
# Format: JSON (pretty-printed, 2-space indent)
# Created by: Stage 1 (winner analysis logic)
# Used by: Debugging, audit trail, reporting
# Purpose: Track top bucket selection and winner distribution
# NOTE: No changes to this file (inherited from VideoDiscoveryCHILD.md)

winner_analysis_file = {
    "path": "/data/{client}/hashtags/{cluster_id}/{mode}_{strategy}/winner_analysis.json",
    "example_path": "/data/acme_corp/hashtags/nutrition/top_contrastive/winner_analysis.json",
    "schema": "WinnerAnalysisSchema",
    "size_kb": "5",
    "created_by": "Stage 1 (winner analysis logic)",
    "used_by": ["Debugging", "Audit trail", "Reporting"],
    "changes": "None (inherited from VideoDiscoveryCHILD.md)",
}

# ===== FILE 5: Config.json (Unchanged) =====
# Location: /data/{client}/hashtags/{cluster_id}/{mode}_{strategy}/config.json
# Source: FoundationCHILD.md Section 5.1 (INHERITED)
# Schema: ConfigSchema (unchanged)
# Size: ~1 KB
# Encoding: UTF-8
# Format: JSON (pretty-printed, 2-space indent)
# Created by: Stage 0 (Foundation setup)
# Used by: All stages (read-only)
# Purpose: Store run configuration for reproducibility
# NOTE: No changes to this file (inherited from FoundationCHILD.md)

config_file = {
    "path": "/data/{client}/hashtags/{cluster_id}/{mode}_{strategy}/config.json",
    "example_path": "/data/acme_corp/hashtags/nutrition/top_contrastive/config.json",
    "schema": "ConfigSchema",
    "size_kb": "1",
    "created_by": "Stage 0 (Foundation setup)",
    "used_by": ["All stages (read-only)"],
    "changes": "None (inherited from FoundationCHILD.md)",
}
```

### 5.3 Integration Points

```python
# Source: HashtagVolumeV2.md DECISION 1, DECISION 2, DECISION 3

"""
Stage 1 Integration Points

ENTRY POINT: CLI invocation
    Input: python rumiai_ml_batch.py --target nutrition --analysis-type hashtag
    Detection: detect_target_type() checks if target starts with # (see Section 3.6)

    IF cluster mode:
        1. Load cluster config: load_cluster_config("nutrition")
        2. Validate config: validate_cluster_config(config)
        3. Run cluster scraping: run_cluster_scraping(config)
        4. Deduplicate: deduplicate_with_provenance(all_videos)
        5. Generate analytics: generate_cluster_analytics(videos)
        6. Save analytics: save_cluster_analytics(analytics)
        7. Continue to winner analysis (inherited from VideoDiscoveryCHILD.md)

    IF single mode (competitor/creator):
        1. Run single scraping (inherited from VideoDiscoveryCHILDTI.md)
        2. Continue to winner analysis (inherited from VideoDiscoveryCHILD.md)

EXIT POINT: selected_videos.json files (per bucket)
    Output: 3 files (one per winning bucket)
    Schema: SelectedVideosSchema with ExtendedVideoMetadataSchema
    Consumer: Stage 2 (Video Processing)

    Stage 2 reads selected_videos.json files:
        - Iterates over videos list
        - Downloads each video using videoMeta.downloadAddr
        - Processes through RumiAI pipeline
        - NEW: Can access source_hashtags and source_runs if needed for analytics

PARALLEL OUTPUT: cluster_analytics.json
    Output: 1 file (per cluster)
    Schema: ClusterAnalyticsSchema
    Consumer: Manual review, optimization, debugging

    Usage:
        - Review per-hashtag contribution
        - Identify underperforming hashtags
        - Optimize cluster composition
        - Analyze run effectiveness (is 2nd run worth it?)
        - Root cause bucket distribution issues

CONFIGURATION DEPENDENCY: /config/hashtag_clusters/{cluster_id}.json
    Must exist before Stage 1 execution (cluster mode)
    Created by: generate_cluster.py interactive tool
    Validation: validate_cluster_config() at load time
    Failure: EXIT_CODE_CLUSTER_CONFIG_NOT_FOUND (10) or EXIT_CODE_CLUSTER_CONFIG_INVALID (11)
"""
```

### 5.4 Code Integration Locations

```python
# Source: HashtagVolumeV2.md DECISION 1, DECISION 2, DECISION 3, DECISION 4

"""
File Modification Plan

FILE 1: rumiai_ml_batch.py (Stage 1 entry point)
    Location: /src/rumiai_ml_batch.py (VideoDiscoveryCHILDTI.md implementation)

    MODIFICATIONS:
        1. Import new functions (Section 3):
            - load_cluster_config
            - validate_cluster_config
            - run_cluster_scraping
            - deduplicate_with_provenance
            - generate_cluster_analytics
            - save_cluster_analytics
            - detect_target_type (modified)

        2. Replace target detection logic:
            BEFORE (VideoDiscoveryCHILDTI.md):
                if analysis_type == "hashtag":
                    target = target  # Use as-is

            AFTER (HashtagVolumeV2.md):
                target_type, cluster_config = detect_target_type(target, analysis_type)

                if target_type == "cluster":
                    # Run cluster scraping (NEW)
                    all_videos = run_cluster_scraping(cluster_config, analysis_mode, country_code)
                    unique_videos, analytics = deduplicate_with_provenance(all_videos)
                    save_cluster_analytics(analytics, client_id, cluster_config['cluster_id'])
                    videos = unique_videos  # Continue to winner analysis
                else:
                    # Run single scraping (INHERITED)
                    videos = scrape_videos(target, analysis_type, analysis_mode, country_code)

        3. Pass videos to winner analysis (unchanged):
            top_buckets = analyze_winner_distribution(videos)
            select_videos_per_bucket(videos, top_buckets, selection_strategy, video_count)

FILE 2: generate_cluster.py (NEW - cluster creation tool)
    Location: /tools/generate_cluster.py (NEW file)

    PURPOSE: Interactive cluster configuration creation

    FUNCTIONALITY:
        1. Prompt for cluster_id
        2. Prompt for description
        3. Prompt for primary_hashtag
        4. Prompt for variant_hashtags (loop until done)
        5. Prompt for scrape_config parameters (with defaults)
        6. Validate configuration (call validate_cluster_config)
        7. Save to /config/hashtag_clusters/{cluster_id}.json
        8. Display summary and next steps

    SOURCE: HashtagVolumeV2.md DECISION 4, lines 908-955

    IMPLEMENTATION: See Section 6.3 for detailed spec

FILE 3: deduplication.py (modified - provenance tracking)
    Location: /src/stage1/deduplication.py (VideoDiscoveryCHILDTI.md implementation)

    MODIFICATIONS:
        1. Replace deduplicate() function with deduplicate_with_provenance():
            - Add source_hashtags tracking
            - Add source_runs tracking
            - Generate cluster analytics

        2. Keep signature compatible:
            BEFORE: deduplicate(videos) -> unique_videos
            AFTER: deduplicate_with_provenance(videos) -> (unique_videos, analytics)

            NOTE: For single mode (competitor/creator), analytics = None

FILE 4: config_constants.py (extended - cluster constants)
    Location: /src/config/config_constants.py (FoundationCHILD.md implementation)

    ADDITIONS:
        - CLUSTER_CONFIG_DIR
        - CLUSTER_CONFIG_PATH_TEMPLATE
        - CLUSTER_ANALYTICS_PATH_TEMPLATE
        - DEFAULT_RUNS_PER_HASHTAG
        - DEFAULT_DELAY_BETWEEN_RUNS_MS
        - DEFAULT_RESULTS_PER_PAGE
        - MIN_VARIANT_HASHTAGS
        - MAX_VARIANT_HASHTAGS
        - MIN_RUNS_PER_HASHTAG
        - MAX_RUNS_PER_HASHTAG
        - MIN_DELAY_BETWEEN_RUNS_MS
        - MAX_DELAY_BETWEEN_RUNS_MS
        - MIN_RESULTS_PER_PAGE
        - MAX_RESULTS_PER_PAGE
        - RETRY_MAX_ATTEMPTS
        - RETRY_BACKOFF_DELAYS

    SOURCE: Section 2.4

FILE 5: error_codes.py (extended - cluster error codes)
    Location: /src/config/error_codes.py (VideoDiscoveryCHILDTI.md implementation)

    ADDITIONS:
        - EXIT_CODE_CLUSTER_CONFIG_NOT_FOUND = 10
        - EXIT_CODE_CLUSTER_CONFIG_INVALID = 11
        - EXIT_CODE_SINGLE_HASHTAG_DEPRECATED = 12
        - EXIT_CODE_ALL_SCRAPES_FAILED = 13

    SOURCE: Section 4.2
"""
```

### 5.4.1 Backward Compatibility Migration

**⚠️ BREAKING CHANGE**: Function signature changed from single return to tuple return

#### Overview

FILE 3: deduplication.py (modified - provenance tracking)
    Location: /src/stage1/deduplication.py (VideoDiscoveryCHILDTI.md implementation)

**Signature Changes**:

```python
# BEFORE (VideoDiscoveryCHILDTI.md):
def deduplicate(videos: list[dict]) -> list[dict]:
    """Return deduplicated videos."""
    return unique_videos

# Caller code
videos = deduplicate(videos)

# AFTER (HashtagVolumeV2.md):
def deduplicate_with_provenance(
    videos: list[dict],
    cluster_config: dict,
    failed_scrapes: list[dict]
) -> tuple[list[dict], dict]:
    """Return (unique_videos, analytics)."""
    return unique_videos, analytics

# Caller code (MUST UPDATE)
videos, analytics = deduplicate_with_provenance(videos, cluster_config, failed_scrapes)
```

#### Migration Steps

**Step 1: Update rumiai_ml_batch.py (Stage 1 entry point)**

FIND:
```python
if target_type == "cluster":
    all_videos = run_cluster_scraping(cluster_config, analysis_mode, country_code)
    videos = deduplicate(all_videos)  # OLD
```

REPLACE:
```python
if target_type == "cluster":
    all_videos, failed_scrapes = run_cluster_scraping(cluster_config, analysis_mode, country_code)
    videos, analytics = deduplicate_with_provenance(all_videos, cluster_config, failed_scrapes)
    save_cluster_analytics(analytics, client_id, cluster_config['cluster_id'])
```

**Step 2: Handle single mode (competitor/creator)**

Single mode still uses OLD deduplication (no provenance tracking needed):

```python
if target_type == "single":
    videos = scrape_videos(target, analysis_type, analysis_mode, country_code)
    videos = deduplicate(videos)  # OLD function still exists for single mode
    # No analytics generated for single mode
```

**RECOMMENDATION**: Keep BOTH functions:
- `deduplicate()` for single mode (backward compatible)
- `deduplicate_with_provenance()` for cluster mode (new)

**Step 3: Update unit tests**

All tests calling `deduplicate_with_provenance()` must handle tuple unpacking:

BEFORE:
```python
unique = deduplicate(videos)
assert len(unique) == expected_count
```

AFTER:
```python
unique, analytics = deduplicate_with_provenance(videos, config, [])
assert len(unique) == expected_count
assert analytics is not None
```

### 5.5 Dependency Graph

```python
# Source: HashtagVolumeV2.md DECISION 1, DECISION 2, DECISION 3

"""
Function Call Flow (Cluster Mode)

CLI Invocation
    ↓
detect_target_type(target, analysis_type)  [Section 3.6]
    ↓ (if cluster)
load_cluster_config(cluster_id)  [Section 3.1]
    ↓
validate_cluster_config(config, path)  [Section 4.1]
    ↓
run_cluster_scraping(config, mode, country_code)  [Section 3.2]
    ↓ (loop: hashtags × runs)
    ├─→ scrape_with_retry(hashtag, run, ...)  [Section 3.3]
    │       ↓ (retry loop)
    │       └─→ call_apify_scraper(...)  [VideoDiscoveryCHILDTI.md - INHERITED]
    └─→ Tag videos with source_hashtags, source_runs
    ↓
deduplicate_with_provenance(all_videos)  [Section 3.4]
    ↓
generate_cluster_analytics(all_videos, unique_videos)  [Section 3.5]
    ↓
save_cluster_analytics(analytics, client_id, cluster_id)  [Section 3.7]
    ↓
analyze_winner_distribution(unique_videos)  [VideoDiscoveryCHILD.md - INHERITED]
    ↓
select_videos_per_bucket(videos, buckets, strategy, count)  [VideoDiscoveryCHILD.md - INHERITED]
    ↓
save_selected_videos(videos, bucket_path)  [VideoDiscoveryCHILD.md - INHERITED]
    ↓
Stage 2 (Video Processing)
```

---

## Section 6: Testing & Implementation Checklist

### 6.1 Unit Tests

**Test Implementation**: Complete unit tests are provided in `/home/jorge/rumiaifinal/tests/test_hashtag_volume_v2.py`

This test file includes all 7 test groups with full implementations:

**Test Coverage**: 7 test groups covering all cluster functionality

**Test Group Summary**:

1. **Cluster Configuration Loading** (8 tests)
   - Load valid config successfully
   - Raise FileNotFoundError if config doesn't exist
   - Raise JSONDecodeError if config malformed
   - Log config details after successful load

2. **Cluster Configuration Validation** (12 tests)
   - Valid config passes all checks
   - Missing required field raises ValueError
   - Invalid cluster_id format raises ValueError
   - Duplicate hashtags detected (case-insensitive)
   - Out-of-range parameters rejected

3. **Cluster Scraping Orchestration** (6 tests)
   - Scrape all hashtags successfully
   - Handle partial failures (some scrapes fail)
   - Tag videos with source provenance
   - Return failed_scrapes list

4. **Retry Logic** (4 tests)
   - Return videos on first attempt
   - Retry and succeed on 2nd attempt
   - Return empty list after all retries fail
   - Exponential backoff timing correct

5. **Deduplication with Provenance** (5 tests)
   - No duplicates - all videos unique
   - Duplicates found - merge provenance
   - Track run provenance correctly
   - Raise ValueError if all_videos empty

6. **Cluster Analytics Generation** (8 tests)
   - Calculate per-hashtag contribution correctly
   - Calculate pairwise overlaps correctly
   - Calculate run effectiveness correctly
   - Use config for scrape counts (not derived data)
   - Populate failed_scrapes correctly

7. **CLI Detection Logic** (3 tests)
   - Detect cluster target (no # prefix)
   - Raise ValueError for single hashtag (deprecated)
   - Competitor mode uses single scraping

**Running Tests**:

```bash
# Run all HashtagVolumeV2 unit tests
pytest tests/test_hashtag_volume_v2.py -v

# Run specific test group
pytest tests/test_hashtag_volume_v2.py::test_load_cluster_config_success -v

# Run with coverage
pytest tests/test_hashtag_volume_v2.py --cov=src.stage1.cluster --cov-report=html

# Run integration tests (requires Apify API key)
pytest tests/test_hashtag_volume_v2.py -v -m integration
```

**Test Dependencies**:
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock
```

**Fixtures**:
Test fixtures are provided in `tests/fixtures/cluster_configs/`:
- `valid_nutrition.json` - Valid cluster config
- `invalid_duplicates.json` - Duplicate hashtags test case
- `invalid_missing_fields.json` - Missing fields test case

For complete test implementations, see `/home/jorge/rumiaifinal/tests/test_hashtag_volume_v2.py`

### 6.2 Integration Tests

**Test Scope**: End-to-end cluster scraping workflow

**Test File**: `tests/integration/test_cluster_integration.py`

**Test Scenario**:
1. Load cluster config from `tests/fixtures/cluster_configs/valid_nutrition.json`
2. Run `run_cluster_scraping()` with test Apify credentials
3. Verify deduplicated videos returned with provenance
4. Verify analytics file created at expected path
5. Verify all functions in call chain execute without errors

**Prerequisites**:
- Apify API key set in environment: `APIFY_API_TOKEN`
- Test cluster config exists: `tests/fixtures/cluster_configs/valid_nutrition.json`

**Run**: `pytest tests/integration/test_cluster_integration.py -v -m integration`

### 6.3 Validation Tools

**1. Configuration Validator** (`validate_cluster_config.py`)
- **Tool**: `src/stage1/validate_cluster_config.py`
- **Usage**: `python src/stage1/validate_cluster_config.py <cluster_id>`
- **Success**: Exit code 0, outputs "✓ Configuration valid"
- **Failure**: Non-zero exit code (see `src/config/error_codes.py`), outputs error details

**2. Health Reporter** (`report_cluster_health.py`)
- **Tool**: `src/stage1/report_cluster_health.py`
- **Usage**: `python src/stage1/report_cluster_health.py <cluster_id>`
- **Output File**: `data/clusters/<cluster_id>/analytics.json`
- **Success**: Exit code 0, file created

**3. CLI Detector** (`detect_cluster_eligible_run.py`)
- **Tool**: `src/stage1/detect_cluster_eligible_run.py`
- **Usage**: `python src/stage1/detect_cluster_eligible_run.py --analysis-mode <mode> --country-code <code>`
- **Output**: "SINGLE_HASHTAG" or "CLUSTER"
- **Success**: Exit code 0

### 6.4 Implementation Checklist

```markdown
# Source: HashtagVolumeV2.md ALL DECISIONS

## Phase 1: Cluster Configuration System (DECISION 1)
- [ ] Create `/config/hashtag_clusters/` directory
- [ ] Implement `ClusterConfigSchema` (Section 2.2)
- [ ] Implement `load_cluster_config()` function (Section 3.1)
- [ ] Implement `validate_cluster_config()` function (Section 4.1)
- [ ] Implement `detect_target_type()` function (Section 3.6)
- [ ] Add cluster constants to `config_constants.py` (Section 2.4)
- [ ] Add cluster error codes to `error_codes.py` (Section 4.2)
- [ ] Create `generate_cluster.py` tool (Section 6.3)
- [ ] Test cluster config loading (Section 6.1 - Test Group 1)
- [ ] Test cluster config validation (Section 6.1 - Test Group 2)
- [ ] Test CLI detection logic (Section 6.1 - Test Group 7)

## Phase 2: Cluster Orchestration (DECISION 2)
- [ ] Implement `run_cluster_scraping()` function (Section 3.2)
- [ ] Implement `scrape_with_retry()` function (Section 3.3)
- [ ] Add retry constants to `config_constants.py` (Section 2.4)
- [ ] Modify `rumiai_ml_batch.py` entry point (Section 5.4 - FILE 1)
- [ ] Add progress logging per scrape
- [ ] Test cluster scraping orchestration (Section 6.1 - Test Group 3)
- [ ] Test retry logic (Section 6.1 - Test Group 4)

## Phase 3: Deduplication & Analytics (DECISION 3)
- [ ] Implement `ExtendedVideoMetadataSchema` (Section 2.3)
- [ ] Implement `deduplicate_with_provenance()` function (Section 3.4)
- [ ] Implement `generate_cluster_analytics()` function (Section 3.5)
- [ ] Implement `save_cluster_analytics()` function (Section 3.7)
- [ ] Implement `ClusterAnalyticsSchema` (Section 2.2)
- [ ] Modify `deduplication.py` file (Section 5.4 - FILE 3)
- [ ] Test deduplication with provenance (Section 6.1 - Test Group 5)
- [ ] Test cluster analytics generation (Section 6.1 - Test Group 6)

## Phase 4: Integration & Testing (DECISION 4, 5, 6, 7)
- [ ] Run end-to-end integration test (Section 6.2)
- [ ] Verify cluster vs single mode compatibility (Section 6.2)
- [ ] Update documentation (README, user guides)
- [ ] Validate single hashtag deprecation warning (DECISION 6)
- [ ] Confirm Stage 2 compatibility (videos with extended schema)
- [ ] Create sample cluster configs for common niches
- [ ] Performance testing with real Apify account

## Phase 5: Deployment Readiness
- [ ] Code review completed
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Documentation updated
- [ ] generate_cluster.py tool tested
- [ ] Cluster configs created for production niches
- [ ] Stage 2 compatibility verified
- [ ] Apify cost estimation validated
- [ ] Error handling tested (all exit codes)
- [ ] Single hashtag deprecation enforced
```

---

## END OF DOCUMENT

**Document Statistics**:
- Total Sections: 6
- Total Functions Specified: 7
- New Schemas: 2 (ClusterConfigSchema, ClusterAnalyticsSchema)
- Extended Schemas: 1 (ExtendedVideoMetadataSchema)
- New Error Codes: 4
- New Configuration Constants: 15
- Implementation Phases: 5

**Source Traceability**:
- All sections linked to source document line numbers
- All decisions mapped to implementation specifications
- All schemas validated against source requirements
- All functions include source attribution

**Ready for Implementation**: ✅
