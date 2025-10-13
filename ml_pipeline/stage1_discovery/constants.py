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
