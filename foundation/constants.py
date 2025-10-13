"""
Shared constants for RumiAI ML Pipeline.

Source: FoundationCHILD.md Section 3: Configuration Dimensions, Section 6: Bucket Definitions
"""

# Bucket Definitions (Section 6: Bucket Definitions)
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

# Valid Enum Values (Section 4.1: CLI Parameters)
VALID_ANALYSIS_TYPES = ["hashtag", "competitor", "creator"]
VALID_MODES = ["top", "recent"]
VALID_STRATEGIES = ["contrastive", "top"]
VALID_REPORT_TYPES = ["single", "comparison"]
VALID_REPORT_AUDIENCES = ["client", "internal", "creator"]
VALID_COUNTRY_CODES = ["US", "BR", "global"]  # Geographic filtering options

# Default Values by Target Type (Section 4.2: Default Value Logic)
DEFAULT_VIDEO_COUNTS = {
    "hashtag": 100,
    "competitor": 100,
    "creator": 40,
}

DEFAULT_ANALYSIS_MODES = {
    "hashtag": "top",
    "competitor": "top",
    "creator": "recent",
}

DEFAULT_SELECTION_STRATEGIES = {
    "hashtag": "contrastive",
    "competitor": "contrastive",
    "creator": "top",
}

DEFAULT_REPORT_AUDIENCES = {
    "hashtag": "client",
    "competitor": "client",
    "creator": "creator",
}

# Default Country Code (geographic filtering)
DEFAULT_COUNTRY_CODE = "US"  # US-specific content by default

# Minimum Recommended N (Section 3.4: Video Count)
MIN_RECOMMENDED_N = {
    "contrastive": 50,  # Ensures 10 bottom performers (20%)
    "top": 20,  # Minimum for K-Means clustering
}

# Engagement Score Formula (Section 3.2: Analysis Modes)
ENGAGEMENT_SHARE_WEIGHT = 10  # 10x weight for shares

# File Naming Patterns
CONFIG_FILENAME = "config.json"
CHECKPOINT_FILENAME = "checkpoint.json"
SELECTED_VIDEOS_FILENAME = "selected_videos.json"
