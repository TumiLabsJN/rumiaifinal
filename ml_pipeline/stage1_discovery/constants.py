"""
Stage 1 Configuration Constants

Source: VideoDiscoveryCHILDTI.md Section 9.2 (Internal Configuration)
"""

from datetime import timezone

# ===== APIFY ACTOR CONFIGURATION =====

# Apify actor IDs
APIFY_PROFILE_SCRAPER_ID = "GdWCkxBtKWOsKjdch"
# clockworks/tiktok-scraper (profile scraper)
# Use case: Competitor/creator analysis

APIFY_HASHTAG_SCRAPER_ID = "GdWCkxBtKWOsKjdch"
# clockworks/tiktok-scraper (profile scraper - unified scraper for both hashtags + profiles)
# Configured: 2025-10-08
# Use case: Hashtag analysis + Competitor/Creator analysis
# Features: Native date filtering, geography filtering (proxyCountryCode)

APIFY_ACTOR_LAST_VALIDATED = "2025-10-08"
# Date actors were last tested (quarterly validation recommended)


# ===== APIFY SCRAPING CONFIGURATION =====

APIFY_SCRAPE_COUNT = 800
# Total videos to scrape per target (hard limit enforced by Apify)

APIFY_TIMEOUT = 720
# Seconds before timeout (12 minutes - provides 2x safety margin for Profile Scraper)

APIFY_RETRY_COUNT = 3
# Retry attempts on failure

APIFY_RETRY_BACKOFF = [5, 15, 45]
# Exponential backoff in seconds


# ===== DATE FILTERING CONFIGURATION =====

DATE_FILTER_TIMEZONE = timezone.utc
# All date filtering performed in UTC

CLOCK_SKEW_TOLERANCE_HOURS = 24
# Accept timestamps up to N hours in future


# ===== WINNER ANALYSIS CONFIGURATION =====

MIN_VIDEOS_FOR_ANALYSIS = 10
# Absolute minimum videos needed

TOP_PERFORMERS_FOR_ANALYSIS = 100
# Analyze top N to identify winning buckets

TOP_BUCKETS_TO_PROCESS = 3
# Process top N buckets only

MIN_WINNER_PERCENTAGE = 5.0
# Minimum 5% of winners to qualify bucket


# ===== SELECTION STRATEGY CONFIGURATION =====

CONTRASTIVE_TOP_SPLIT = 0.8
# 80% top, 20% bottom for contrastive strategy

MIN_VIDEOS_PER_BUCKET = 10
# Minimum videos to process bucket


# ===== ENGAGEMENT SCORE FORMULA =====

ENGAGEMENT_SHARE_WEIGHT = 10
# 10x weight for shares in engagement score
# Formula: engagement_score = playCount + (shareCount × 10) + commentCount + likeCount
# NOTE: Current implementation uses playCount only


# ===== INTERACTIVE CONFIRMATION =====

AUTO_CONFIRM = False
# Skip Stage 1 confirmation prompt when True


# ===== EXIT CODES =====

EXIT_CODE_SUCCESS = 0
EXIT_CODE_APIFY_KEY_MISSING = 1
EXIT_CODE_INVALID_DATE_FILTER = 2
EXIT_CODE_APIFY_TIMEOUT = 3
EXIT_CODE_INSUFFICIENT_VIDEOS = 6
EXIT_CODE_ALL_DUPLICATES = 7
EXIT_CODE_USER_ABORT = 130

# ===== CLUSTER CONFIGURATION (HashtagVolumeV2_TI.md) =====

# Cluster Configuration Paths
CLUSTER_CONFIG_DIR = "/config/hashtag_clusters/"
# Source: HashtagVolumeV2.md DECISION 1, line 210

CLUSTER_CONFIG_PATH_TEMPLATE = "/config/hashtag_clusters/{cluster_id}.json"
# Source: HashtagVolumeV2.md DECISION 1, line 210

# Cluster Analytics Paths
CLUSTER_ANALYTICS_PATH_TEMPLATE = "/data/clients/{client_id}/hashtag/{cluster_id}/cluster_analytics.json"
# Source: HashtagVolumeV2.md DECISION 3, line 586

# Cluster Scraping Defaults
DEFAULT_RUNS_PER_HASHTAG = 2
# Source: HashtagVolumeV2.md DECISION 1, line 224
# Rationale: 2 runs balance data volume with cost

DEFAULT_DELAY_BETWEEN_RUNS_MS = 120000  # 2 minutes
# Source: HashtagVolumeV2.md DECISION 1, line 225
# Rationale: Avoid rate limiting, allow TikTok feed to refresh

DEFAULT_RESULTS_PER_PAGE = 800
# Source: HashtagVolumeV2.md DECISION 1, line 226
# Rationale: Maximum videos per Apify scrape

# Cluster Validation Ranges
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

# Retry Configuration
RETRY_MAX_ATTEMPTS = 3
# Source: HashtagVolumeV2.md DECISION 2, line 445
# Rationale: 3 retries balance reliability with execution speed

RETRY_BACKOFF_DELAYS = [5, 15, 45]  # seconds
# Source: HashtagVolumeV2.md DECISION 2, line 445
# Rationale: Exponential backoff (5s, 15s, 45s) handles transient network issues

# Cluster Exit Codes
EXIT_CODE_CLUSTER_CONFIG_NOT_FOUND = 10
# Source: HashtagVolumeV2.md DECISION 1, line 311
# Trigger: load_cluster_config() raises FileNotFoundError

EXIT_CODE_CLUSTER_CONFIG_INVALID = 11
# Source: HashtagVolumeV2.md DECISION 1, line 238
# Trigger: validate_cluster_config() raises ValueError

EXIT_CODE_SINGLE_HASHTAG_DEPRECATED = 12
# Source: HashtagVolumeV2.md DECISION 6, lines 1052-1062
# Trigger: detect_target_type() detects single hashtag (target starts with #)

EXIT_CODE_ALL_SCRAPES_FAILED = 13
# Source: HashtagVolumeV2.md DECISION 2, line 461
# Trigger: run_cluster_scraping() returns empty list (all scrapes failed after retries)
