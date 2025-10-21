# Review CSV Generation - Technical Implementation

> **TI Document**: ReviewCSVGenerationTI.md
> **Parent HLD**: ReviewCSVGenerationCHILD.md (Stage 3.4: Review CSV Generation)
> **Foundation HLD**: FoundationCHILD.md (Shared across all stages)
> **Version**: 1.0
> **Last Updated**: 2025-01-17
> **Status**: Draft

---

## 1. Document Metadata

**Feature Name**: Review CSV Generation

**Parent HLD**: ReviewCSVGenerationCHILD.md

**Foundation HLD**: FoundationCHILD.md

**Covers HLD Sections**:

**From ReviewCSVGenerationCHILD.md**:
- Section 1: Context & Business Goal
- Section 1.1: What Problem Does This Solve?
- Section 1.2: Where This Fits in Pipeline
- Section 1.3: Success Criteria
- Section 2: Architecture & Design
- Section 2.1: High-Level Approach
- Section 2.2: Data Flow
- Section 2.3: Detailed Process
- Section 2.3.1: Prerequisite - Modify temporal_compute.py (Stage 2)
- Section 2.3.2: Load Temporal Windows Files
- Section 2.3.3: Extract Features (Mirror aggregated_features.csv Logic)
- Section 2.3.4: URL Validation and Row Filtering
- Section 2.3.5: Generate video_review.csv
- Section 3: Dependencies & Integration
- Section 3.1: Input Dependencies
- Section 3.2: Output Contracts
- Section 3.3: Cross-Stage Dependencies
- Section 3.4: External Dependencies
- Section 4: Configuration & Parameters
- Section 4.1: Configuration Sources
- Section 4.2: Environment Variables
- Section 4.3: Runtime Parameters
- Section 5: Data Schemas
- Section 5.1: Input Schema
- Section 5.2: Output Schema
- Section 6: Error Handling & Validation
- Section 6.1: Input Validation
- Section 6.2: Error Cases
- Section 6.3: Output Validation
- Section 7: Performance & Scalability
- Section 8: Testing Strategy
- Section 10: References & Related Docs
- Appendix A: Glossary
- Appendix B: Decision Log

**From FoundationCHILD.md**:
- Section 2: Client Architecture & Storage
- Section 2.1: Directory Structure
- Section 2.2: Path Templates
- Section 4: CLI Command Structure
- Section 4.1: CLI Parameters
- Section 5: Configuration Schemas
- Section 5.1: config.json Schema

**Related TI Documents**:

**Depends On**:
- FoundationTI.md (REQUIRED - documentation dependency: provides directory structure, CLI parameters, config schemas)
- VideoProcessingTI.md (Stage 2) - Runtime dependency: produces temporal_windows_updated.json files with metadata.url
- FeatureAggregationTI.md (Stage 3) - Runtime dependency: produces aggregated_features.csv for validation

**Feeds Into**:
- NONE - video_review.csv is optional and independent (does not impact ML pipeline)

**Implementation Priority**: MEDIUM

**Rationale**: This component enables rapid outlier investigation during manual review, but does not block ML training. The ML pipeline (aggregated_features.csv) continues unaffected. However, manual review is valuable for debugging RumiAI processing issues (statistical outliers, edge case content) and improving data quality over time. Without this component, outlier investigation requires manually hunting through multiple files (temporal_windows_updated.json, unified_analysis.json) across N videos, significantly slowing debugging workflows.

---

## 2. Stage Contract

### 2.1 Input Contract

```python
# Sources: FoundationCHILD.md Sections 2, 4 | ReviewCSVGenerationCHILD.md Sections 3.1, 5.1

class Stage3_4Input:
    """
    Exact structure Stage 3.4 (Review CSV Generation) receives.

    Sources:
    - CLI parameters: FoundationCHILD.md Section 4.1
    - Directory paths: FoundationCHILD.md Section 2.2
    - Stage-specific inputs: ReviewCSVGenerationCHILD.md Section 3.1
    """

    # ===== CLI PARAMETERS (from FoundationCHILD.md Section 4.1) =====
    client_id: str                  # Required, CLI parameter --client, alphanumeric + underscore
                                    # Example: "acme_corp"
                                    # Validation: Regex ^[a-zA-Z0-9_]+$ (min 1 char)

    analysis_type: str              # Required, CLI parameter --analysis-type
                                    # Valid values: ["hashtag", "competitor", "creator"]
                                    # Example: "hashtag"

    target: str                     # Required, CLI parameter --target
                                    # Format depends on analysis_type (sanitized, no prefix)
                                    # Example: "nutrition" (from "#nutrition")

    analysis_mode: str              # Required, CLI parameter --analysis-mode
                                    # Valid values: ["top", "recent"]
                                    # Example: "top"

    selection_strategy: str         # Required, CLI parameter --selection-strategy
                                    # Valid values: ["contrastive", "top"]
                                    # Example: "contrastive"

    bucket: str                     # Required, bucket identifier
                                    # Format: "{min}-{max}s" (e.g., "18-33s")
                                    # Valid values: ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]

    # ===== DIRECTORY PATHS (from FoundationCHILD.md Section 2.2) =====
    bucket_base: str                # Bucket directory path
                                    # Template: "/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/bucket_{bucket}/"
                                    # Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/"

    insights_dir: str               # Temporal windows output directory
                                    # Template: "{bucket_base}/analysis/insights/"
                                    # Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/analysis/insights/"

    validation_dir: str             # Validation output directory (for video_review.csv)
                                    # Template: "{bucket_base}/validation/"
                                    # Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/validation/"

    # ===== STAGE-SPECIFIC INPUTS (ReviewCSVGenerationCHILD.md Section 3.1) =====
    temporal_windows_files: list[str]  # Paths to temporal_windows_updated.json files
                                       # Location: "{insights_dir}/*_temporal_windows_updated.json"
                                       # Example: [
                                       #   "/data/.../bucket_18-33s/analysis/insights/7428596413707144481_temporal_windows_updated.json",
                                       #   "/data/.../bucket_18-33s/analysis/insights/7428596413707144482_temporal_windows_updated.json"
                                       # ]
                                       # Schema: ReviewCSVGenerationCHILD.md Section 5.1
                                       # Source: Stage 2 (Video Processing)
                                       # Must exist: Yes
                                       # Validation: At least 1 file required

    metadata_url_field: str         # Field name for URL in temporal_windows_updated.json
                                    # Value: "url" (from metadata.url)
                                    # Source: Stage 2 modification to temporal_compute.py (Section 2.3.1)
                                    # Validation: Must be present in metadata dict (may be None value)
```

### 2.2 Output Contract

```python
# Sources: FoundationCHILD.md Section 2.2 | ReviewCSVGenerationCHILD.md Sections 3.2, 5.2

class Stage3_4Output:
    """
    Exact structure Stage 3.4 (Review CSV Generation) produces.

    Sources:
    - Output contracts: ReviewCSVGenerationCHILD.md Section 3.2
    - Output schemas: ReviewCSVGenerationCHILD.md Section 5.2
    - Directory paths: FoundationCHILD.md Section 2.2
    """

    # ===== OUTPUT FILES =====
    video_review_csv_path: str      # Path to video_review.csv
                                    # Location: "{bucket_base}/validation/video_review.csv"
                                    # Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/validation/video_review.csv"
                                    # Schema: ReviewCSVGenerationCHILD.md Section 5.2 (VideoReviewSchema)
                                    # Format: CSV with header row
                                    # Size: ~200KB for N=100 videos (bucket 18-33s, 186 columns)
                                    # Consumers: Human (Excel review for outlier investigation)

    # ===== OUTPUT SCHEMA DETAILS =====
    video_review_schema: dict       # Schema definition (see Section 3.3)
                                    # Columns: video_id, url, duration, all temporal features
                                    # Column count: Bucket-dependent
                                    #   - Bucket 0-3s, 3-9s: ~67 columns (3 metadata + ~64 features)
                                    #   - Bucket 9-13s, 13-18s: ~157 columns (3 metadata + ~154 features)
                                    #   - Bucket 18-33s: ~187 columns (3 metadata + ~184 features)
                                    #   - Bucket 33-60s, 60-90s, 90-120s: ~217 columns (3 metadata + ~214 features)
                                    # Row count: N videos (same as aggregated_features.csv, minus videos with missing url)

    # ===== VALIDATION METRICS =====
    row_count: int                  # Number of videos included in review CSV
                                    # Must be: 1 <= row_count <= N (where N = total videos in bucket)
                                    # Comparison: row_count <= len(aggregated_features.csv)
                                    # Difference reason: Videos missing metadata.url are excluded

    column_count: int               # Number of columns in review CSV
                                    # Must match: column_count == len(aggregated_features.csv columns) + 1
                                    # Extra column: "url" (at position 2)

    videos_with_url: int            # Count of videos with valid url field
                                    # Must be: videos_with_url == row_count

    videos_without_url: int         # Count of videos excluded (missing url)
                                    # Must be: videos_with_url + videos_without_url == total_videos

    # ===== EXIT CODES =====
    exit_code_success: int = 0      # All videos processed, review CSV generated
    exit_code_pre_flight: int = 1   # insights/ directory missing or no JSON files found
    exit_code_no_valid_urls: int = 2  # All videos missing url field (no output generated)
    exit_code_io_failure: int = 4   # Disk full during CSV write or permission denied
```

---

## 3. Data Schemas

### 3.1 Foundation Schemas

These schemas are defined in FoundationCHILD.md and used across all pipeline stages.

```python
# Source: FoundationCHILD.md Section 5.1
ConfigSchema = {
    "client_id": str,              # Required, alphanumeric + underscore, Example: "acme_corp"
    "analysis_type": str,          # Required, ["hashtag", "competitor", "creator"], Example: "hashtag"
    "target": str,                 # Required, sanitized target name, Example: "nutrition"
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
# Note: This stage reads config.json indirectly through directory paths (not direct config access)
```

### 3.2 Stage 3.4 Input Schema

```python
# Source: ReviewCSVGenerationCHILD.md Section 5.1

TemporalWindowsInputSchema = {
    # File: temporal_windows_updated.json (N files per bucket)
    # Location: {bucket_base}/analysis/insights/{video_id}_temporal_windows_updated.json
    # Format: JSON

    # Top-level structure
    "temporal_windows": dict,      # Required, Contains hook, middle_segments, closing
    "metadata": dict,              # Required, Contains video_id, url, duration, engagement metrics

    # temporal_windows.hook (first 3 seconds)
    "temporal_windows.hook": {
        "scene_count": int,        # Required, Range: 0-20, Scene cuts in hook window (0-3s)
        "word_count": int,         # Required, Range: 0-150, Words spoken in hook
        "eye_contact_rate": float, # Required, Range: 0.0-1.0, Gaze camera ratio
        # **FEATURE EXTRACTION IS DYNAMIC**: All keys in the hook dict will be extracted.
        # The 3 features shown above are representative examples only.
        # Implementation (Section 4.3) uses: for feature_name, value in hook.items()
        # Full feature list available in TotalFeatures.md (external reference)
        # All hook features will be prefixed with "hook_" in video_review.csv
    },

    # temporal_windows.middle_segments (3-5 segments OR null)
    "temporal_windows.middle_segments": list | None,  # null for videos ≤9s, list of dicts otherwise
    # **DYNAMIC EXTRACTION**: Each middle segment dict has same features as hook (dynamically extracted)
    # Features will be prefixed with "middle_{i}_" in video_review.csv (i = 1, 2, 3, 4, or 5)
    # Implementation: for i, segment in enumerate(middle_segments): for feature_name, value in segment.items()

    # temporal_windows.closing (last 3 seconds)
    "temporal_windows.closing": {
        "scene_count": int,        # Required, Range: 0-20, Scene cuts in closing window
        "word_count": int,         # Required, Range: 0-150, Words spoken in closing
        # **DYNAMIC EXTRACTION**: All keys in closing dict will be extracted (same structure as hook)
        # Implementation (Section 4.3) uses: for feature_name, value in closing.items()
        # All closing features will be prefixed with "closing_" in video_review.csv
    },

    # metadata (video-level information)
    "metadata.video_id": str,      # Required, Unique TikTok video ID, Example: "7428596413707144481"
    "metadata.url": str | None,    # Required for review CSV (may be None if Stage 2 modification not deployed)
                                   # Format: "https://www.tiktok.com/@user/video/{video_id}"
                                   # Example: "https://www.tiktok.com/@user/video/7428596413707144481"
                                   # Source: Apify "webVideoUrl" → unified_analysis.json → temporal_compute.py
    "metadata.duration": float,    # Required, Range: 0-120, Video length in seconds, Example: 22.5
    "metadata.digg_count": int,    # Required, Range: >=0, TikTok likes
    "metadata.play_count": int,    # Required, Range: >=0, TikTok views
    "metadata.share_count": int,   # Required, Range: >=0, TikTok shares
    "metadata.comment_count": int, # Required, Range: >=0, TikTok comments
    # Additional metadata fields may exist (author, create_time, etc.) but not used by this stage
}
```

### 3.3 Stage 3.4 Output Schema

```python
# Source: ReviewCSVGenerationCHILD.md Section 5.2

VideoReviewCSVSchema = {
    # File: video_review.csv
    # Location: {bucket_base}/validation/video_review.csv
    # Format: CSV with header row
    # Rows: N videos (where N = videos with valid url)
    # Columns: Bucket-dependent (67-217 columns)

    # ===== METADATA COLUMNS (always first 3 columns) =====
    "video_id": str,               # Column 1, Required, TikTok video ID (17-20 digits)
                                   # Example: "7428596413707144481"
                                   # Source: metadata.video_id from temporal_windows_updated.json

    "url": str,                    # Column 2, Required, TikTok video URL
                                   # Format: https://www.tiktok.com/@{user}/video/{video_id}
                                   # Example: "https://www.tiktok.com/@user/video/7428596413707144481"
                                   # Source: metadata.url from temporal_windows_updated.json
                                   # Validation: Must be non-empty string (videos without url are excluded)

    "duration": float,             # Column 3, Required, Video length in seconds
                                   # Range: Bucket-specific (e.g., 18.0-33.0 for bucket 18-33s)
                                   # Example: 22.5
                                   # Source: metadata.duration from temporal_windows_updated.json

    # ===== TEMPORAL FEATURE COLUMNS (columns 4+) =====
    # All features from temporal_windows.hook, middle_segments, closing
    # Naming convention: {window_name}_{feature_name}

    # Hook features (always present, dynamically extracted)
    "hook_scene_count": int,       # Range: 0-20, Scene cuts in first 3s, Example: 3
    "hook_word_count": int,        # Range: 0-150, Words spoken in first 3s, Example: 15
    "hook_eye_contact_rate": float,# Range: 0.0-1.0, Gaze camera ratio, Example: 0.75
    # **DYNAMIC EXTRACTION**: All keys from temporal_windows.hook are extracted with "hook_" prefix
    # The 3 features above are representative examples. Exact count depends on RumiAI temporal_windows output.
    # Implementation: for feature_name, value in hook.items(): features[f'hook_{feature_name}'] = value

    # Middle segment features (bucket-dependent, 0-5 segments)
    # Bucket 0-3s, 3-9s: No middle segments (null) → 0 middle columns
    # Bucket 9-13s, 13-18s: 3 segments × ~30 features = ~90 columns
    # Bucket 18-33s: 4 segments × ~30 features = ~120 columns
    # Bucket 33-60s, 60-90s, 90-120s: 5 segments × ~30 features = ~150 columns
    "middle_1_scene_count": int,   # Range: 0-20, Scene cuts in middle segment 1, Example: 5
    "middle_1_word_count": int,    # Range: 0-150, Words in middle segment 1, Example: 20
    "middle_2_scene_count": int,   # (if applicable for bucket)
    "middle_3_scene_count": int,   # (if applicable for bucket)
    "middle_4_scene_count": int,   # (if applicable for bucket, e.g., 18-33s+)
    "middle_5_scene_count": int,   # (if applicable for bucket, e.g., 33-60s+)
    # ... remaining middle features for applicable segments

    # Closing features (always present, dynamically extracted)
    "closing_scene_count": int,    # Range: 0-20, Scene cuts in last 3s, Example: 4
    "closing_word_count": int,     # Range: 0-150, Words in last 3s, Example: 12
    # **DYNAMIC EXTRACTION**: All keys from temporal_windows.closing are extracted with "closing_" prefix
    # Implementation: for feature_name, value in closing.items(): features[f'closing_{feature_name}'] = value
}

# ===== COLUMN COUNT BY BUCKET =====
# Bucket 0-3s:      ~67 columns  (3 metadata + 64 features: hook + closing, no middle)
# Bucket 3-9s:      ~67 columns  (3 metadata + 64 features: hook + closing, no middle)
# Bucket 9-13s:     ~157 columns (3 metadata + 154 features: hook + 3 middle + closing)
# Bucket 13-18s:    ~157 columns (3 metadata + 154 features: hook + 3 middle + closing)
# Bucket 18-33s:    ~187 columns (3 metadata + 184 features: hook + 4 middle + closing)
# Bucket 33-60s:    ~217 columns (3 metadata + 214 features: hook + 5 middle + closing)
# Bucket 60-90s:    ~217 columns (3 metadata + 214 features: hook + 5 middle + closing)
# Bucket 90-120s:   ~217 columns (3 metadata + 214 features: hook + 5 middle + closing)

# ===== ROW REQUIREMENTS =====
# - All rows must have valid (non-empty) url
# - Row count = Number of videos with valid metadata.url
# - Videos with missing/null/empty url are excluded (logged as warnings)
# - Row count ≤ aggregated_features.csv row count (some videos may be excluded)
```

**Field Count Verification**:
```
Child Section 5.2 Table: Shows example schema for bucket 18-33s (186 columns)
TI Schema 3.3: Documents 3 metadata + ~184 features = ~187 columns ✓

**IMPORTANT - Dynamic Extraction Approach**:
- Feature extraction is DYNAMIC (not hardcoded to specific feature names)
- Implementation (Section 4.3) loops through all dict keys: for feature_name, value in window.items()
- This makes the code future-proof: new features in temporal_windows automatically appear in CSV
- Exact feature count depends on RumiAI temporal_windows output (varies by pipeline version)
- The 3 features shown per window (scene_count, word_count, eye_contact_rate) are examples only
- Full feature enumeration available in TotalFeatures.md (external reference)
```

---

## 4. Algorithmic Specifications

**Source**: ReviewCSVGenerationCHILD.md Section 2.3 (Detailed Process)

---

> ✅ **STATUS NOTICE**: Section 4.1 documents a Stage 2 prerequisite modification that is **already implemented**.
>
> This section is included for reference only (documents the dependency).
>
> **Stage 3.4 implementations begin at Section 4.2** (load_temporal_windows).

---

### 4.1 Function: ensure_url_in_metadata() [PREREQUISITE - Stage 2]

**Status**: ✅ Already Implemented in temporal_compute.py

**Purpose**: Ensure `url` field flows through to temporal_windows_updated.json metadata (Stage 2 modification)

**Implementation** (from ReviewCSVGenerationCHILD.md Section 2.3.1):

```python
# NOTE: This is a Stage 2 modification, not implemented in Stage 3.4
# Documented here for dependency tracking

# File: /home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py
# Location: Line ~2650 (calculated_metadata section)

def compute_temporal_windows(unified_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute temporal windows from unified_analysis (Stage 2 function).

    MODIFICATION REQUIRED: Add 'url' field to calculated_metadata dict.

    Source: ReviewCSVGenerationCHILD.md Section 2.3.1
    """
    # Extract metadata from unified_analysis
    metadata = unified_analysis.get('metadata', {})
    video_id = metadata.get('video_id', 'unknown')

    # BEFORE (current code - incomplete):
    # calculated_metadata = {
    #     'video_id': video_id,
    #     'duration': video_duration,
    #     'digg_count': metadata.get('likes', 0),
    #     'play_count': metadata.get('views', 0),
    #     ...
    # }

    # AFTER (required modification):
    calculated_metadata = {
        'video_id': video_id,
        'duration': video_duration,
        'url': metadata.get('url'),  # ← ADD THIS LINE
        'digg_count': metadata.get('likes', 0),
        'play_count': metadata.get('views', 0),
        'share_count': metadata.get('shares', 0),
        'comment_count': metadata.get('comments', 0),
        # ... other metadata fields
    }

    # Continue with temporal window computation
    # ...

    return {
        'temporal_windows': temporal_windows,
        'metadata': calculated_metadata  # Now includes 'url'
    }
```

**Edge Cases** (from ReviewCSVGenerationCHILD.md Section 2.3.1):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| metadata.url is None | Pass through as None | Stage 3.4 will detect and handle missing urls |
| metadata.url is empty string | Pass through as is | Stage 3.4 will detect and handle |
| metadata key missing entirely | Pass through as None (`.get()`) | Graceful degradation |

**Validation Rules**:
- NONE - This is passthrough logic, validation happens in Stage 3.4

**Error Conditions**:
- NONE - Stage 2 responsibility (not Stage 3.4)

---

### 4.2 Function: load_temporal_windows()

**Purpose**: Read all temporal_windows_updated.json files for the bucket

**Implementation** (from ReviewCSVGenerationCHILD.md Section 2.3.2):

```python
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_temporal_windows(bucket_path: Path) -> List[Dict[str, Any]]:
    """
    Load all temporal_windows_updated.json files for a bucket.

    Args:
        bucket_path: Path to bucket directory (e.g., bucket_18-33s/)
                    Example: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/

    Returns:
        List of temporal windows dicts (one per video)
        Example: [
            {
                'temporal_windows': {...},
                'metadata': {'video_id': '123', 'url': 'https://...', 'duration': 22.5}
            },
            ...
        ]

    Raises:
        FileNotFoundError: If insights/ directory doesn't exist
        ValueError: If no JSON files found
        JSONDecodeError: If malformed JSON encountered

    Source: ReviewCSVGenerationCHILD.md Section 2.3.2
    """
    # Step 1: Construct path to insights directory
    insights_dir = bucket_path / "analysis" / "insights"
    logger.info(f"Loading temporal windows from {insights_dir}")

    # Step 2: Validate insights directory exists
    if not insights_dir.exists():
        error_msg = f"insights/ directory not found: {insights_dir}. Stage 2 must complete first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Step 3: Find all temporal_windows_updated.json files
    json_files = sorted(insights_dir.glob("*_temporal_windows_updated.json"))

    # Step 4: Validate at least one file found
    if not json_files:
        error_msg = f"No temporal_windows_updated.json files found in {insights_dir}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Found {len(json_files)} temporal windows files")

    # Step 5: Load each JSON file
    temporal_data = []
    for json_file in json_files:
        try:
            # Step 5a: Read JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Step 5b: Validate required keys present
            if 'temporal_windows' not in data:
                logger.warning(f"Missing 'temporal_windows' key in {json_file.name}, skipping")
                continue

            if 'metadata' not in data:
                logger.warning(f"Missing 'metadata' key in {json_file.name}, skipping")
                continue

            # Step 5c: Add to results
            temporal_data.append(data)

        except json.JSONDecodeError as e:
            # Malformed JSON - log error and skip file
            logger.error(f"Malformed JSON in {json_file.name}: {e}")
            raise  # Fail fast on corrupted data

    # Step 6: Return loaded data
    logger.info(f"Successfully loaded {len(temporal_data)} temporal windows")
    return temporal_data
```

**Edge Cases** (from ReviewCSVGenerationCHILD.md Section 2.3.2):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| insights/ directory doesn't exist | Raise FileNotFoundError | Stage 2 must complete before Stage 3.4 |
| No JSON files found | Raise ValueError | Cannot generate review CSV without data |
| Malformed JSON | Raise JSONDecodeError | Fail fast on corrupted data |

**Validation Rules**:
```python
assert insights_dir.exists(), f"insights/ directory not found: {insights_dir}"
assert len(json_files) > 0, f"No JSON files found in {insights_dir}"
```

**Error Conditions**:
- Links to Section 6.2: Error "missing_input_file"

**Example Input**:
```
bucket_path = Path("/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/")
```

**Example Output**:
```python
[
    {
        'temporal_windows': {
            'hook': {'scene_count': 3, 'word_count': 15, ...},
            'middle_segments': [{...}, {...}, {...}, {...}],
            'closing': {'scene_count': 4, 'word_count': 12, ...}
        },
        'metadata': {
            'video_id': '7428596413707144481',
            'url': 'https://www.tiktok.com/@user/video/7428596413707144481',
            'duration': 22.5
        }
    },
    # ... more videos
]
```

---

### 4.3 Function: extract_features_with_url()

**Purpose**: Extract same features as aggregated_features.csv to ensure review mirrors ML input

**Implementation** (from ReviewCSVGenerationCHILD.md Section 2.3.3):

```python
def extract_features_with_url(temporal_windows: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from temporal_windows_updated.json.
    Uses SAME logic as aggregated_features.csv generation (Stage 3.1-3.3).

    Args:
        temporal_windows: Single video's temporal windows data
                         Format: {
                             'temporal_windows': {...},
                             'metadata': {'video_id': str, 'url': str, 'duration': float}
                         }

    Returns:
        Dict with video_id, url, duration, and all temporal features
        Example: {
            'video_id': '7428596413707144481',
            'url': 'https://www.tiktok.com/@user/video/...',
            'duration': 22.5,
            'hook_scene_count': 3,
            'hook_word_count': 15,
            'middle_1_scene_count': 5,
            ...
        }

    Source: ReviewCSVGenerationCHILD.md Section 2.3.3
    """
    # Step 1: Extract metadata
    metadata = temporal_windows.get('metadata', {})
    video_id = metadata.get('video_id', 'unknown')
    url = metadata.get('url')  # May be None
    duration = metadata.get('duration', 0)

    logger.debug(f"Extracting features for video {video_id}")

    # Step 2: Initialize features dict with metadata
    features = {
        'video_id': video_id,
        'url': url,
        'duration': duration
    }

    # Step 3: Extract hook features (all ~30 base features)
    hook = temporal_windows['temporal_windows']['hook']
    for feature_name, value in hook.items():
        # Prefix hook features with "hook_"
        features[f'hook_{feature_name}'] = value

    logger.debug(f"Extracted {len(hook)} hook features")

    # Step 4: Extract middle segment features (3-5 segments depending on bucket)
    middle_segments = temporal_windows['temporal_windows']['middle_segments']

    if middle_segments is not None:  # null for videos ≤9s
        # Step 4a: Iterate through middle segments
        for i, segment in enumerate(middle_segments, start=1):
            # Step 4b: Extract all features from segment
            for feature_name, value in segment.items():
                # Prefix with "middle_{i}_"
                features[f'middle_{i}_{feature_name}'] = value

        logger.debug(f"Extracted features from {len(middle_segments)} middle segments")
    else:
        logger.debug("No middle segments (video ≤9s)")

    # Step 5: Extract closing features (all ~30 base features)
    closing = temporal_windows['temporal_windows']['closing']
    for feature_name, value in closing.items():
        # Prefix closing features with "closing_"
        features[f'closing_{feature_name}'] = value

    logger.debug(f"Extracted {len(closing)} closing features")

    # Step 6: Log total feature count
    total_features = len(features) - 3  # Subtract metadata columns
    logger.debug(f"Total features extracted: {total_features}")

    # Step 7: Return feature dict
    return features
```

**Edge Cases** (from ReviewCSVGenerationCHILD.md Section 2.3.3):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| middle_segments is null | Skip middle feature extraction | Videos ≤9s have no middle (expected) |
| Feature value is None | Pass through as None | Preserve data fidelity |
| Unexpected feature keys | Include anyway | Future-proof for schema changes |

**Validation Rules**:
```python
# No validation here - pass through all features as-is
# Validation happens in Stage 3.1-3.3 (Feature Aggregation)
```

**Error Conditions**:
- Links to Section 6.2: NONE (graceful handling of all edge cases)

**Example Input**:
```python
{
    'temporal_windows': {
        'hook': {'scene_count': 3, 'word_count': 15, 'eye_contact_rate': 0.75},
        'middle_segments': [
            {'scene_count': 5, 'word_count': 20},
            {'scene_count': 4, 'word_count': 18},
            {'scene_count': 3, 'word_count': 22},
            {'scene_count': 6, 'word_count': 19}
        ],
        'closing': {'scene_count': 4, 'word_count': 12}
    },
    'metadata': {
        'video_id': '7428596413707144481',
        'url': 'https://www.tiktok.com/@user/video/7428596413707144481',
        'duration': 22.5
    }
}
```

**Example Output**:
```python
{
    'video_id': '7428596413707144481',
    'url': 'https://www.tiktok.com/@user/video/7428596413707144481',
    'duration': 22.5,
    'hook_scene_count': 3,
    'hook_word_count': 15,
    'hook_eye_contact_rate': 0.75,
    'middle_1_scene_count': 5,
    'middle_1_word_count': 20,
    'middle_2_scene_count': 4,
    'middle_2_word_count': 18,
    'middle_3_scene_count': 3,
    'middle_3_word_count': 22,
    'middle_4_scene_count': 6,
    'middle_4_word_count': 19,
    'closing_scene_count': 4,
    'closing_word_count': 12
}
```

---

### 4.4 Function: filter_videos_with_url()

**Purpose**: Exclude videos with missing url from review CSV (cannot investigate without clickable link)

**Implementation** (from ReviewCSVGenerationCHILD.md Section 2.3.4):

```python
def filter_videos_with_url(feature_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out videos missing url field.

    Args:
        feature_rows: List of feature dicts (one per video)
                     Example: [
                         {'video_id': '123', 'url': 'https://...', 'feature1': 5},
                         {'video_id': '456', 'url': None, 'feature1': 10},
                         ...
                     ]

    Returns:
        Filtered list (only videos with valid url)
        Example: [
            {'video_id': '123', 'url': 'https://...', 'feature1': 5}
        ]

    Source: ReviewCSVGenerationCHILD.md Section 2.3.4
    """
    # Step 1: Initialize counters
    valid_rows = []
    skipped_count = 0

    logger.info(f"Filtering {len(feature_rows)} videos for valid urls")

    # Step 2: Iterate through all feature rows
    for row in feature_rows:
        # Step 2a: Extract video_id and url
        video_id = row.get('video_id', 'unknown')
        url = row.get('url')

        # Step 2b: Check if url is valid (non-empty)
        if not url:  # None or empty string
            # Step 2c: Log warning for excluded video
            logger.warning(
                f"Video {video_id} excluded from video_review.csv - missing url"
            )
            skipped_count += 1
            continue

        # Step 2d: Check if url is whitespace-only
        if isinstance(url, str) and url.strip() == '':
            logger.warning(
                f"Video {video_id} excluded from video_review.csv - whitespace-only url"
            )
            skipped_count += 1
            continue

        # Step 2e: Add valid row to results
        valid_rows.append(row)

    # Step 3: Log summary
    if skipped_count > 0:
        logger.info(
            f"Excluded {skipped_count} videos from review CSV (missing url). "
            f"These videos remain in aggregated_features.csv for ML training."
        )

    logger.info(f"Retained {len(valid_rows)} videos with valid urls")

    # Step 4: Return filtered rows
    return valid_rows
```

**Edge Cases** (from ReviewCSVGenerationCHILD.md Section 2.3.4):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| All videos missing url | Return empty list, log error | Review CSV cannot be generated |
| url is whitespace-only string | Treat as missing (not url) | Invalid url = same as missing |
| url format invalid | Pass through anyway | Excel will show invalid link (user can see issue) |

**Validation Rules**:
```python
# Validation happens in calling function (generate_review_csv)
# This function only filters, doesn't fail
```

**Error Conditions**:
- Links to Section 6.2: Error "all_videos_missing_url" (handled by caller)

**Example Input**:
```python
[
    {'video_id': '123', 'url': 'https://tiktok.com/...', 'feature1': 5},
    {'video_id': '456', 'url': None, 'feature1': 10},          # Missing url
    {'video_id': '789', 'url': '', 'feature1': 15},             # Empty url
    {'video_id': '101', 'url': '   ', 'feature1': 20}           # Whitespace
]
```

**Example Output**:
```python
[
    {'video_id': '123', 'url': 'https://tiktok.com/...', 'feature1': 5}
]
# 3 videos excluded (logged as warnings)
```

---

### 4.5 Function: generate_review_csv()

**Purpose**: Write CSV with video_id, url, duration, all features (mirror aggregated_features.csv)

**Implementation** (from ReviewCSVGenerationCHILD.md Section 2.3.5):

```python
import pandas as pd

def generate_review_csv(
    feature_rows: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Generate video_review.csv from feature rows.

    Args:
        feature_rows: List of feature dicts (filtered for valid urls)
                     Example: [
                         {'video_id': '123', 'url': 'https://...', 'duration': 22.5, 'hook_scene_count': 3, ...}
                     ]
        output_path: Path to save video_review.csv
                    Example: /data/clients/.../bucket_18-33s/validation/video_review.csv

    Raises:
        ValueError: If no videos with valid url (empty feature_rows)
        OSError: If disk full during write

    Source: ReviewCSVGenerationCHILD.md Section 2.3.5
    """
    # Step 1: Validate input
    if not feature_rows:
        error_msg = "No videos with valid url - cannot generate review CSV"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Generating review CSV with {len(feature_rows)} videos")

    # Step 2: Convert to DataFrame
    df = pd.DataFrame(feature_rows)
    logger.debug(f"DataFrame created: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 3: Reorder columns (video_id, url, duration first, then features)
    # Source: ReviewCSVGenerationCHILD.md Section 2.3.5 (url at position 2)
    metadata_cols = ['video_id', 'url', 'duration']

    # Step 3a: Get feature columns (all except metadata)
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    # Step 3b: Sort feature columns alphabetically
    feature_cols_sorted = sorted(feature_cols)

    # Step 3c: Combine metadata + sorted features
    final_column_order = metadata_cols + feature_cols_sorted
    df = df[final_column_order]

    logger.debug(f"Columns reordered: video_id, url, duration, then {len(feature_cols_sorted)} features")

    # Step 4: Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory ensured: {output_path.parent}")

    # Step 5: Write CSV to disk
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Generated video_review.csv: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"   Location: {output_path}")

        # Step 5a: Log file size
        file_size_kb = output_path.stat().st_size / 1024
        logger.info(f"   File size: {file_size_kb:.2f} KB")

    except OSError as e:
        # Disk full or permission denied
        error_msg = f"Failed to write video_review.csv: {e}"
        logger.error(error_msg)
        raise
```

**Edge Cases** (from ReviewCSVGenerationCHILD.md Section 2.3.5):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| validation/ directory doesn't exist | Create with mkdir(parents=True) | Auto-create directory structure |
| File already exists | Overwrite | Re-running Stage 3 should replace old data |
| Disk full | Raise OSError | Fail fast on system errors |

**Validation Rules**:
```python
assert len(feature_rows) > 0, "No videos with valid url"
assert 'url' in df.columns, "url column missing from DataFrame"
assert df.columns[1] == 'url', "url must be at column position 2 (index 1)"
```

**Error Conditions**:
- Links to Section 6.2: Error "disk_full_during_csv_write"

**Example Input**:
```python
feature_rows = [
    {'video_id': '123', 'url': 'https://...', 'duration': 20.0, 'hook_scene_count': 3, 'closing_scene_count': 4}
]
output_path = Path("/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/validation/video_review.csv")
```

**Example Output**:
```
File created at: /data/clients/.../bucket_18-33s/validation/video_review.csv

CSV contents:
video_id,url,duration,closing_scene_count,hook_scene_count
123,https://...,20.0,4,3
```

**Logs**:
```
INFO: Generating review CSV with 1 videos
DEBUG: DataFrame created: 1 rows, 5 columns
DEBUG: Columns reordered: video_id, url, duration, then 2 features
DEBUG: Output directory ensured: /data/.../bucket_18-33s/validation/
INFO: ✅ Generated video_review.csv: 1 rows, 5 columns
INFO:    Location: /data/.../bucket_18-33s/validation/video_review.csv
INFO:    File size: 0.15 KB
```

---

### 4.6 Function: generate_review_csv_for_bucket() [ORCHESTRATION]

**Purpose**: Main entry point that orchestrates all subfunctions to generate review CSV

**Implementation** (from ReviewCSVGenerationCHILD.md Sections 2.2, 2.3):

```python
def generate_review_csv_for_bucket(bucket_path: Path) -> None:
    """
    Main orchestration function for Stage 3.4 (Review CSV Generation).
    Wires together all subfunctions: load → extract → filter → generate → validate.

    Args:
        bucket_path: Path to bucket directory
                    Example: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/

    Raises:
        FileNotFoundError: If insights/ directory doesn't exist
        ValueError: If no JSON files found OR all videos missing url
        OSError: If disk full during CSV write or permission denied
        JSONDecodeError: If malformed JSON encountered

    Exit Codes:
        0: Success - review CSV generated
        1: Pre-flight failure - missing input files
        2: Execution failure - all videos missing url
        4: I/O failure - disk full or permission denied

    Source: ReviewCSVGenerationCHILD.md Sections 2.2, 2.3
    """
    logger.info("=" * 80)
    logger.info("STAGE 3.4: Review CSV Generation")
    logger.info(f"Bucket: {bucket_path}")
    logger.info("=" * 80)

    try:
        # Step 1: Validate inputs (Section 5.1)
        logger.info("Step 1/5: Validating inputs...")
        validate_stage_input(bucket_path)
        logger.info("✅ Input validation passed")

        # Step 2: Load temporal windows from JSON files (Section 4.2)
        logger.info("Step 2/5: Loading temporal windows...")
        temporal_data = load_temporal_windows(bucket_path)
        logger.info(f"✅ Loaded {len(temporal_data)} temporal windows")

        # Step 3: Extract features with url for each video (Section 4.3)
        logger.info("Step 3/5: Extracting features...")
        feature_rows = []
        for tw_data in temporal_data:
            features = extract_features_with_url(tw_data)
            feature_rows.append(features)
        logger.info(f"✅ Extracted features from {len(feature_rows)} videos")

        # Step 4: Filter out videos with missing url (Section 4.4)
        logger.info("Step 4/5: Filtering videos with valid urls...")
        valid_rows = filter_videos_with_url(feature_rows)

        # Check if any videos remain after filtering
        if not valid_rows:
            error_msg = (
                "No videos with valid url - cannot generate review CSV. "
                "Check if Stage 2 modification (temporal_compute.py) is deployed."
            )
            logger.error(error_msg)
            logger.error("Process terminated with exit code 2")
            raise ValueError(error_msg)

        logger.info(f"✅ Filtered to {len(valid_rows)} videos with valid urls")

        # Step 5: Generate review CSV (Section 4.5)
        logger.info("Step 5/5: Generating review CSV...")
        output_path = bucket_path / "validation" / "video_review.csv"
        generate_review_csv(valid_rows, output_path)
        logger.info("✅ Review CSV generated")

        # Step 6: Validate output (Section 5.2)
        logger.info("Validating output...")
        validate_stage_output(output_path, valid_rows)
        logger.info("✅ Output validation passed")

        # Success summary
        logger.info("=" * 80)
        logger.info(f"✅ STAGE 3.4 COMPLETED SUCCESSFULLY")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Videos: {len(valid_rows)} (excluded: {len(feature_rows) - len(valid_rows)})")
        logger.info(f"   Exit code: 0")
        logger.info("=" * 80)

    except FileNotFoundError as e:
        # Pre-flight validation failure
        logger.error("=" * 80)
        logger.error("❌ STAGE 3.4 FAILED: Pre-flight Validation")
        logger.error(f"   Error: {e}")
        logger.error(f"   Exit code: 1")
        logger.error("=" * 80)
        raise

    except ValueError as e:
        # Execution failure (all videos missing url)
        logger.error("=" * 80)
        logger.error("❌ STAGE 3.4 FAILED: Execution Failure")
        logger.error(f"   Error: {e}")
        logger.error(f"   Exit code: 2")
        logger.error("=" * 80)
        raise

    except OSError as e:
        # I/O failure (disk full, permission denied)
        logger.error("=" * 80)
        logger.error("❌ STAGE 3.4 FAILED: I/O Failure")
        logger.error(f"   Error: {e}")
        logger.error(f"   Exit code: 4")
        logger.error("=" * 80)
        raise

    except json.JSONDecodeError as e:
        # Malformed JSON
        logger.error("=" * 80)
        logger.error("❌ STAGE 3.4 FAILED: Malformed JSON")
        logger.error(f"   Error: {e}")
        logger.error(f"   Exit code: 1")
        logger.error("=" * 80)
        raise
```

**Workflow Diagram**:
```
bucket_path (input)
      ↓
[validate_stage_input]  ← Section 5.1
      ↓
[load_temporal_windows]  ← Section 4.2
      ↓
temporal_data (List[Dict])
      ↓
[extract_features_with_url (loop)]  ← Section 4.3
      ↓
feature_rows (List[Dict])
      ↓
[filter_videos_with_url]  ← Section 4.4
      ↓
valid_rows (List[Dict])
      ↓ (if empty → ValueError exit code 2)
[generate_review_csv]  ← Section 4.5
      ↓
video_review.csv (output)
      ↓
[validate_stage_output]  ← Section 5.2
      ↓
✅ SUCCESS (exit code 0)
```

**Error Handling Flow**:
```
FileNotFoundError (insights/ missing)     → Exit code 1
ValueError (no JSON files)                → Exit code 1
JSONDecodeError (malformed JSON)          → Exit code 1
ValueError (all videos missing url)       → Exit code 2
OSError (disk full)                       → Exit code 4
PermissionError (permission denied)       → Exit code 4
```

**Edge Cases** (from ReviewCSVGenerationCHILD.md Section 2.3):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Some videos missing url | Filter out, log warnings | Partial success better than failure |
| All videos missing url | ValueError, exit code 2 | Cannot generate review CSV |
| Empty feature_rows after filtering | Caught by `if not valid_rows` check | Same as all videos missing url |

**Validation Rules**:
```python
# Pre-execution validation (Section 5.1)
assert insights_dir.exists(), "insights/ directory must exist"
assert len(json_files) > 0, "At least one JSON file required"

# Post-execution validation (Section 5.2)
assert output_path.exists(), "video_review.csv must be created"
assert len(valid_rows) > 0, "At least one video with valid url required"
```

**Error Conditions**:
- Links to Section 6: All error conditions handled with appropriate exit codes

**Example Usage**:
```python
from pathlib import Path

# Example 1: Normal execution
bucket_path = Path("/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/")
generate_review_csv_for_bucket(bucket_path)
# Output: /data/.../bucket_18-33s/validation/video_review.csv

# Example 2: Error handling
try:
    generate_review_csv_for_bucket(bucket_path)
except FileNotFoundError:
    print("Stage 2 must complete first")
except ValueError:
    print("All videos missing url - deploy Stage 2 modification")
except OSError:
    print("Disk full or permission denied")
```

**Logs** (from Example Trace 1):
```
================================================================================
STAGE 3.4: Review CSV Generation
Bucket: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/
================================================================================
INFO: Step 1/5: Validating inputs...
INFO: ✅ Input validation passed
INFO: Step 2/5: Loading temporal windows...
INFO: Loading temporal windows from /data/.../bucket_18-33s/analysis/insights
INFO: Found 3 temporal windows files
INFO: Successfully loaded 3 temporal windows
INFO: ✅ Loaded 3 temporal windows
INFO: Step 3/5: Extracting features...
DEBUG: Extracting features for video 7428596413707144481
DEBUG: Extracted 3 hook features
DEBUG: Extracted features from 4 middle segments
DEBUG: Extracted 3 closing features
INFO: ✅ Extracted features from 3 videos
INFO: Step 4/5: Filtering videos with valid urls...
INFO: Filtering 3 videos for valid urls
INFO: Retained 3 videos with valid urls
INFO: ✅ Filtered to 3 videos with valid urls
INFO: Step 5/5: Generating review CSV...
INFO: Generating review CSV with 3 videos
INFO: ✅ Generated video_review.csv: 3 rows, 15 columns
INFO: ✅ Review CSV generated
INFO: Validating output...
INFO: ✅ Output validation passed
================================================================================
✅ STAGE 3.4 COMPLETED SUCCESSFULLY
   Output: /data/.../bucket_18-33s/validation/video_review.csv
   Videos: 3 (excluded: 0)
   Exit code: 0
================================================================================
```

---

## 5. Validation Rules

**Source**: ReviewCSVGenerationCHILD.md Sections 6.1, 6.3

### 5.1 Input Validation

```python
# Source: ReviewCSVGenerationCHILD.md Section 6.1

def validate_stage_input(bucket_path: Path) -> None:
    """
    Validate input before processing.
    Source: ReviewCSVGenerationCHILD.md Section 6.1

    Raises:
        FileNotFoundError: If insights/ directory doesn't exist
        ValueError: If no temporal_windows_updated.json files found
    """
    # Validation 1: insights/ directory exists
    insights_dir = bucket_path / "analysis" / "insights"
    assert insights_dir.exists(), \
        f"insights/ directory not found: {insights_dir}. Stage 2 must complete first."

    # Validation 2: At least one JSON file exists
    json_files = list(insights_dir.glob("*_temporal_windows_updated.json"))
    assert len(json_files) > 0, \
        f"No temporal_windows_updated.json files found in {insights_dir}"

    # Validation 3: JSON files are readable (checked in load_temporal_windows)
    # Validation 4: Required keys present (checked in load_temporal_windows)
```

### 5.2 Output Validation

```python
# Source: ReviewCSVGenerationCHILD.md Section 6.3

def validate_stage_output(
    output_path: Path,
    feature_rows: List[Dict[str, Any]]
) -> None:
    """
    Validate output after processing.
    Source: ReviewCSVGenerationCHILD.md Section 6.3

    Args:
        output_path: Path to video_review.csv
                    Example: /data/.../bucket_18-33s/validation/video_review.csv
        feature_rows: List of feature dicts that were written to CSV (for row count validation)

    Raises:
        AssertionError: If validation checks fail
    """
    # Derive bucket_path from output_path
    # output_path format: {bucket_base}/validation/video_review.csv
    # bucket_path format: {bucket_base}
    bucket_path = output_path.parent.parent

    # Construct aggregated_features.csv path
    # Location: {bucket_base}/ml_analysis/aggregated_features.csv (per Foundation Section 2.2)
    aggregated_features_csv_path = bucket_path / "ml_analysis" / "aggregated_features.csv"

    # Validation 1: CSV file exists
    assert output_path.exists(), \
        f"video_review.csv not created at {output_path}"

    # Load generated CSV
    import pandas as pd
    df_review = pd.read_csv(output_path)

    # Validation 2: CSV row count > 0
    assert len(df_review) > 0, \
        f"video_review.csv is empty (0 rows)"

    # Validation 3: url column exists
    assert 'url' in df_review.columns, \
        f"video_review.csv missing 'url' column"

    # Validation 4: url column is at position 2 (index 1)
    assert df_review.columns[1] == 'url', \
        f"url column not at position 2. Found at position {list(df_review.columns).index('url') + 1}"

    # Validation 5: Row count matches input (minus excluded videos)
    expected_row_count = len(feature_rows)
    actual_row_count = len(df_review)
    assert actual_row_count == expected_row_count, \
        f"Row count mismatch: expected {expected_row_count}, got {actual_row_count}"

    # Validation 6: Row count ≤ aggregated_features.csv (some videos may be excluded)
    if aggregated_features_csv_path.exists():
        df_aggregated = pd.read_csv(aggregated_features_csv_path)
        assert len(df_review) <= len(df_aggregated), \
            f"Review CSV has more rows ({len(df_review)}) than aggregated CSV ({len(df_aggregated)})"

        # Warning if difference > 10%
        diff_percentage = (len(df_aggregated) - len(df_review)) / len(df_aggregated) * 100
        if diff_percentage > 10:
            logger.warning(
                f"Review CSV has {diff_percentage:.1f}% fewer rows than aggregated CSV. "
                f"Check if many videos are missing url field."
            )

    # Validation 7: Column count = aggregated columns + 1 (for url)
    if aggregated_features_csv_path.exists():
        df_aggregated = pd.read_csv(aggregated_features_csv_path)
        expected_col_count = len(df_aggregated.columns) + 1  # +1 for url column
        actual_col_count = len(df_review.columns)
        assert actual_col_count == expected_col_count, \
            f"Column count mismatch: expected {expected_col_count} (aggregated + url), got {actual_col_count}"

    # Validation 8: All urls are non-empty strings
    assert df_review['url'].notna().all(), \
        f"Found null values in url column (should have been filtered out)"

    assert (df_review['url'].str.strip() != '').all(), \
        f"Found empty strings in url column (should have been filtered out)"

    logger.info("✅ Output validation passed")
```

---

## 6. Error Handling

**Source**: ReviewCSVGenerationCHILD.md Section 6.2

```python
ERROR_CONDITIONS = {
    # Error 1: Missing Input File (from Child Section 6.2)
    "missing_input_file": {
        "condition": "insights/ directory doesn't exist OR no *_temporal_windows_updated.json files found",
        "error_type": "FileNotFoundError",
        "action": "Fail-fast (raise exception, exit with code 1)",
        "retry_policy": "No retry",
        "user_message": "insights/ directory not found at {path}. Stage 2 must complete first."
    },

    # Error 2: All Videos Missing URL (from Child Section 6.2)
    "all_videos_missing_url": {
        "condition": "filter_videos_with_url() returns empty list (no videos have valid url)",
        "error_type": "ValueError",
        "action": "Skip video_review.csv generation, log error, continue to Stage 4 (ML pipeline unaffected)",
        "retry_policy": "No retry",
        "user_message": "No videos with valid url - cannot generate review CSV. Check if Stage 2 modification (temporal_compute.py) is deployed."
    },

    # Error 3: Disk Full During CSV Write (from Child Section 6.2)
    "disk_full_during_csv_write": {
        "condition": "OSError raised by df.to_csv() - insufficient disk space",
        "error_type": "OSError",
        "action": "Fail-fast (raise exception, exit with code 4)",
        "retry_policy": "No retry",
        "user_message": "Failed to write video_review.csv: {error}. Check disk space at {path}."
    },

    # Error 4: Malformed JSON (from Child Section 6.1 Input Validation)
    "malformed_json": {
        "condition": "json.load() raises JSONDecodeError",
        "error_type": "JSONDecodeError",
        "action": "Fail-fast (raise exception, exit with code 1)",
        "retry_policy": "No retry",
        "user_message": "Malformed JSON in {filename}: {error}. Re-run Stage 2 (Video Processing)."
    },

    # Error 5: Permission Denied (from Child Section 6.2)
    "permission_denied": {
        "condition": "PermissionError during directory creation or file write",
        "error_type": "PermissionError",
        "action": "Fail-fast (raise exception, exit with code 4)",
        "retry_policy": "No retry",
        "user_message": "Permission denied writing to {path}. Check directory permissions."
    },
}
```

---

## 7. Complete Example Traces

**Source**: ReviewCSVGenerationCHILD.md Section 2.2 (Data Flow) and Section 5 (Data Schemas)

### Trace 1: Normal Processing (Happy Path)

**Source**: Derived from ReviewCSVGenerationCHILD.md Sections 2.2, 5.1, 5.2

**Input**:
```
Bucket: 18-33s
Path: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/

Files in analysis/insights/:
- 7428596413707144481_temporal_windows_updated.json (22.5s)
- 7428596413707144482_temporal_windows_updated.json (25.0s)
- 7428596413707144483_temporal_windows_updated.json (19.8s)

Sample temporal_windows_updated.json (video 7428596413707144481):
{
  "temporal_windows": {
    "hook": {"scene_count": 3, "word_count": 15, "eye_contact_rate": 0.75},
    "middle_segments": [
      {"scene_count": 5, "word_count": 20},
      {"scene_count": 4, "word_count": 18},
      {"scene_count": 3, "word_count": 22},
      {"scene_count": 6, "word_count": 19}
    ],
    "closing": {"scene_count": 4, "word_count": 12}
  },
  "metadata": {
    "video_id": "7428596413707144481",
    "url": "https://www.tiktok.com/@user/video/7428596413707144481",
    "duration": 22.5
  }
}
```

**Processing Steps**:
```
Step 1: load_temporal_windows(bucket_path)
        → Found 3 JSON files
        → Loaded 3 videos successfully
        Intermediate: [
          {temporal_windows: {...}, metadata: {video_id: "7428596413707144481", url: "https://...", duration: 22.5}},
          {temporal_windows: {...}, metadata: {video_id: "7428596413707144482", url: "https://...", duration: 25.0}},
          {temporal_windows: {...}, metadata: {video_id: "7428596413707144483", url: "https://...", duration: 19.8}}
        ]

Step 2: extract_features_with_url() for each video
        → Video 1: Extracted 15 features (3 metadata + 12 temporal)
        → Video 2: Extracted 15 features
        → Video 3: Extracted 15 features
        Intermediate: [
          {video_id: "7428596413707144481", url: "https://...", duration: 22.5, hook_scene_count: 3, ...},
          {video_id: "7428596413707144482", url: "https://...", duration: 25.0, hook_scene_count: 2, ...},
          {video_id: "7428596413707144483", url: "https://...", duration: 19.8, hook_scene_count: 4, ...}
        ]

Step 3: filter_videos_with_url(feature_rows)
        → All 3 videos have valid url
        → No videos excluded
        Intermediate: 3 valid rows retained

Step 4: generate_review_csv(valid_rows, output_path)
        → DataFrame created: 3 rows × 15 columns
        → Columns reordered: video_id, url, duration, then 12 features alphabetically
        → Output directory created: /data/.../bucket_18-33s/validation/
        → CSV written successfully
```

**Output**:
```
File: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/validation/video_review.csv

video_id,url,duration,closing_scene_count,closing_word_count,hook_eye_contact_rate,hook_scene_count,hook_word_count,middle_1_scene_count,middle_1_word_count,middle_2_scene_count,middle_2_word_count,middle_3_scene_count,middle_3_word_count,middle_4_scene_count,middle_4_word_count
7428596413707144481,https://www.tiktok.com/@user/video/7428596413707144481,22.5,4,12,0.75,3,15,5,20,4,18,3,22,6,19
7428596413707144482,https://www.tiktok.com/@user/video/7428596413707144482,25.0,5,10,0.65,2,12,6,22,5,20,4,18,7,21
7428596413707144483,https://www.tiktok.com/@user/video/7428596413707144483,19.8,3,14,0.80,4,18,4,19,3,17,5,23,6,20

**Note**: The trace above shows 15 columns for readability. Actual output for bucket 18-33s has ~187 columns:
- 3 metadata columns: video_id, url, duration
- ~30 hook features: hook_scene_count, hook_word_count, hook_eye_contact_rate, ... (27 more)
- ~120 middle features: 4 segments × ~30 features each
- ~30 closing features: closing_scene_count, closing_word_count, ... (28 more)
```

**Files Created**:
- `/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/validation/video_review.csv`: 3 rows, ~187 columns (~15 KB for bucket 18-33s)

**Logs**:
```
INFO: Loading temporal windows from /data/.../bucket_18-33s/analysis/insights
INFO: Found 3 temporal windows files
INFO: Successfully loaded 3 temporal windows
DEBUG: Extracting features for video 7428596413707144481
DEBUG: Extracted 3 hook features
DEBUG: Extracted features from 4 middle segments
DEBUG: Extracted 3 closing features
DEBUG: Total features extracted: 12
INFO: Filtering 3 videos for valid urls
INFO: Retained 3 videos with valid urls
INFO: Generating review CSV with 3 videos
DEBUG: DataFrame created: 3 rows, 15 columns
DEBUG: Columns reordered: video_id, url, duration, then 12 features
DEBUG: Output directory ensured: /data/.../bucket_18-33s/validation/
INFO: ✅ Generated video_review.csv: 3 rows, 15 columns
INFO:    Location: /data/.../bucket_18-33s/validation/video_review.csv
INFO:    File size: 0.52 KB
INFO: ✅ Output validation passed
```

---

### Trace 2: Edge Case - Videos with Missing URL

**Source**: ReviewCSVGenerationCHILD.md Section 2.3.4 Edge Cases

**Input**:
```
Bucket: 18-33s
Path: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/

Files in analysis/insights/:
- 7428596413707144481_temporal_windows_updated.json (url present)
- 7428596413707144482_temporal_windows_updated.json (url: null)
- 7428596413707144483_temporal_windows_updated.json (url: "")

Sample with missing url (video 7428596413707144482):
{
  "temporal_windows": {...},
  "metadata": {
    "video_id": "7428596413707144482",
    "url": null,  ← Missing URL
    "duration": 25.0
  }
}
```

**Processing Steps**:
```
Step 1: load_temporal_windows(bucket_path)
        → Found 3 JSON files
        → Loaded 3 videos successfully
        Intermediate: 3 videos (including 2 with missing url)

Step 2: extract_features_with_url() for each video
        → Video 1: url = "https://..." (valid)
        → Video 2: url = null (invalid)
        → Video 3: url = "" (invalid)
        Intermediate: 3 feature rows extracted

Step 3: filter_videos_with_url(feature_rows)
        → Video 7428596413707144481: url valid, RETAINED
        → Video 7428596413707144482: url is null, EXCLUDED (logged warning)
        → Video 7428596413707144483: url is empty string, EXCLUDED (logged warning)
        Intermediate: 1 valid row retained (2 excluded)

Step 4: generate_review_csv(valid_rows, output_path)
        → DataFrame created: 1 row × 15 columns
        → CSV written successfully
```

**Output**:
```
File: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/validation/video_review.csv

video_id,url,duration,...
7428596413707144481,https://www.tiktok.com/@user/video/7428596413707144481,22.5,...

(Only 1 row - 2 videos excluded due to missing url)
```

**Logs**:
```
INFO: Loading temporal windows from /data/.../bucket_18-33s/analysis/insights
INFO: Found 3 temporal windows files
INFO: Successfully loaded 3 temporal windows
INFO: Filtering 3 videos for valid urls
WARNING: Video 7428596413707144482 excluded from video_review.csv - missing url
WARNING: Video 7428596413707144483 excluded from video_review.csv - whitespace-only url
INFO: Excluded 2 videos from review CSV (missing url). These videos remain in aggregated_features.csv for ML training.
INFO: Retained 1 videos with valid urls
INFO: Generating review CSV with 1 videos
INFO: ✅ Generated video_review.csv: 1 rows, 15 columns
INFO:    Location: /data/.../bucket_18-33s/validation/video_review.csv
WARNING: Review CSV has 66.7% fewer rows than aggregated CSV. Check if many videos are missing url field.
```

---

### Trace 3: Error Case - All Videos Missing URL

**Source**: ReviewCSVGenerationCHILD.md Section 6.2 Error Cases

**Input**:
```
Bucket: 18-33s
Path: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/

Files in analysis/insights/:
- 7428596413707144481_temporal_windows_updated.json (url: null)
- 7428596413707144482_temporal_windows_updated.json (url: null)
- 7428596413707144483_temporal_windows_updated.json (url: null)

All 3 videos have missing url field (Stage 2 modification not deployed)
```

**Processing Steps**:
```
Step 1: load_temporal_windows(bucket_path)
        → Found 3 JSON files
        → Loaded 3 videos successfully
        Intermediate: 3 videos (all missing url)

Step 2: extract_features_with_url() for each video
        → Video 1: url = null
        → Video 2: url = null
        → Video 3: url = null
        Intermediate: 3 feature rows extracted (all with url=null)

Step 3: filter_videos_with_url(feature_rows)
        → Video 7428596413707144481: url is null, EXCLUDED
        → Video 7428596413707144482: url is null, EXCLUDED
        → Video 7428596413707144483: url is null, EXCLUDED
        Intermediate: 0 valid rows (all excluded)

Step 4: generate_review_csv(valid_rows, output_path)
        → ERROR: No videos with valid url
        → ValueError raised
```

**Output**:
```
No output file created
```

**Error**:
```
ValueError: No videos with valid url - cannot generate review CSV. Check if Stage 2 modification (temporal_compute.py) is deployed.
```

**Exit Code**: 2

**Logs**:
```
INFO: Loading temporal windows from /data/.../bucket_18-33s/analysis/insights
INFO: Found 3 temporal windows files
INFO: Successfully loaded 3 temporal windows
INFO: Filtering 3 videos for valid urls
WARNING: Video 7428596413707144481 excluded from video_review.csv - missing url
WARNING: Video 7428596413707144482 excluded from video_review.csv - missing url
WARNING: Video 7428596413707144483 excluded from video_review.csv - missing url
INFO: Excluded 3 videos from review CSV (missing url). These videos remain in aggregated_features.csv for ML training.
INFO: Retained 0 videos with valid urls
ERROR: No videos with valid url - cannot generate review CSV
ERROR: Process terminated with exit code 2
```

**Recovery Action**:
1. Verify Stage 2 modification deployed: Check `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py` line ~2650
2. Ensure `'url': metadata.get('url')` added to calculated_metadata dict
3. Re-run Stage 2 (Video Processing) to regenerate temporal_windows_updated.json files with url field
4. Retry Stage 3.4 (Review CSV Generation)

---

## 8. File Structure & Integration

**Source**: FoundationCHILD.md Section 2, ReviewCSVGenerationCHILD.md Sections 3.1, 3.2, 3.4

### 8.1 Module Location

```
FILE_PATH = "/rumiai_v2/ml_pipeline/stage3_aggregation/review_csv_generator.py"
# Rationale: Part of Stage 3 (Feature Aggregation) sub-stage 3.4
```

### 8.2 Imports

```python
# Source: ReviewCSVGenerationCHILD.md Section 3.4 External Dependencies

IMPORTS = [
    "import json  # Standard library",
    "import logging  # Standard library",
    "from pathlib import Path  # Standard library",
    "from typing import List, Dict, Any  # Standard library",
    "import pandas as pd  # 2.0.0+",
]
```

### 8.3 Entry Point

```python
ENTRY_FUNCTION = "generate_review_csv_for_bucket(bucket_path: Path) -> None"
# Main orchestration function that calls all 5 subfunctions
```

### 8.4 Base Directory Structure

```python
# Source: FoundationCHILD.md Section 2.1

BASE_PATHS = {
    "client_base": "/data/clients/{client_id}/",
    "analysis_type_base": "{client_base}/{analysis_type}s/",
    "target_base": "{analysis_type_base}/{target}/",
    "analysis_base": "{target_base}/{mode}_{strategy}/",
    "bucket_base": "{analysis_base}/bucket_{bucket}/",

    # Stage 3.4 specific paths
    "insights": "{bucket_base}/analysis/insights/",
    "validation": "{bucket_base}/validation/",
}
```

### 8.5 Stage Output Paths

```python
# Source: ReviewCSVGenerationCHILD.md Section 3.2

OUTPUT_PATHS = {
    "video_review_csv": "{bucket_base}/validation/video_review.csv",
}
```

### 8.6 Integration Points

```python
CALLS_TO_EXTERNAL_SYSTEMS = {}
# No external systems - pure local file processing
```

---

## 9. Configuration & Environment

**Source**: FoundationCHILD.md Sections 3, 4 | ReviewCSVGenerationCHILD.md Sections 4.1, 4.2

### 9.1 Environment Variables

```python
# Source: ReviewCSVGenerationCHILD.md Section 4.2

ENV_VARS = {}
# No environment variables required - uses path-based configuration
```

### 9.2 Configuration Object

```python
# Source: ReviewCSVGenerationCHILD.md Section 4.1

CONFIG_SCHEMA = {
    "cli_params": {
        # No stage-specific CLI parameters
        # Inherits all parameters from Foundation (client_id, analysis_type, etc.)
        # This stage is invoked internally by Stage 3 pipeline, not via CLI
    },

    "internal_constants": {
        # No internal constants - behavior is deterministic based on input data
    },
}
```

**Note**: This stage has NO user-configurable parameters (per Child Section 4.1). All configuration comes from directory paths and input data structure.

---

## 10. Logging Specifications

**Source**: ReviewCSVGenerationCHILD.md Section 2.3 pseudocode (no Section 6.4 present)

### 10.1 Log Messages

```python
# Extracted from ReviewCSVGenerationCHILD.md Section 2.3 pseudocode

LOG_MESSAGES = {
    # Loading phase
    "loading_start": ("INFO", "Loading temporal windows from {insights_dir}"),
    "files_found": ("INFO", "Found {count} temporal windows files"),
    "loading_success": ("INFO", "Successfully loaded {count} temporal windows"),

    # Extraction phase
    "extracting_features": ("DEBUG", "Extracting features for video {video_id}"),
    "hook_extracted": ("DEBUG", "Extracted {count} hook features"),
    "middle_extracted": ("DEBUG", "Extracted features from {count} middle segments"),
    "no_middle": ("DEBUG", "No middle segments (video ≤9s)"),
    "closing_extracted": ("DEBUG", "Extracted {count} closing features"),
    "total_features": ("DEBUG", "Total features extracted: {count}"),

    # Filtering phase
    "filtering_start": ("INFO", "Filtering {count} videos for valid urls"),
    "video_excluded_null": ("WARNING", "Video {video_id} excluded from video_review.csv - missing url"),
    "video_excluded_empty": ("WARNING", "Video {video_id} excluded from video_review.csv - whitespace-only url"),
    "excluded_summary": ("INFO", "Excluded {count} videos from review CSV (missing url). These videos remain in aggregated_features.csv for ML training."),
    "retained_summary": ("INFO", "Retained {count} videos with valid urls"),

    # Generation phase
    "generating_start": ("INFO", "Generating review CSV with {count} videos"),
    "dataframe_created": ("DEBUG", "DataFrame created: {rows} rows, {cols} columns"),
    "columns_reordered": ("DEBUG", "Columns reordered: video_id, url, duration, then {count} features"),
    "directory_ensured": ("DEBUG", "Output directory ensured: {path}"),
    "generation_success": ("INFO", "✅ Generated video_review.csv: {rows} rows, {cols} columns"),
    "output_location": ("INFO", "   Location: {path}"),
    "file_size": ("INFO", "   File size: {size_kb} KB"),

    # Validation phase
    "validation_passed": ("INFO", "✅ Output validation passed"),
    "validation_warning_row_diff": ("WARNING", "Review CSV has {percentage}% fewer rows than aggregated CSV. Check if many videos are missing url field."),

    # Error phase
    "no_json_files": ("ERROR", "No temporal_windows_updated.json files found in {path}"),
    "malformed_json": ("ERROR", "Malformed JSON in {filename}: {error}"),
    "no_valid_urls": ("ERROR", "No videos with valid url - cannot generate review CSV"),
    "write_failed": ("ERROR", "Failed to write video_review.csv: {error}"),
}
```

### 10.2 Metrics to Track

```python
# Source: Inferred from ReviewCSVGenerationCHILD.md Section 7.1

METRICS = {
    "videos_processed": "Total videos loaded from temporal_windows_updated.json files",
    "videos_with_url": "Count of videos with valid url field",
    "videos_without_url": "Count of videos excluded (missing url)",
    "processing_time_seconds": "Total processing time in seconds",
    "csv_read_time_seconds": "Time spent loading JSON files",
    "csv_write_time_seconds": "Time spent writing video_review.csv",
    "output_file_size_kb": "Size of generated video_review.csv in kilobytes",
}
```

---

## 11. Implementation Log

**Purpose**: Record any deviations from Sections 1-10 during implementation (Phase 4).

**When to Log**: Any time actual implementation differs from this TI specification.

**Instructions**: See `/DevOps/Phase 4/Implementation_Prompt.md` for enforcement rules.

---

### 11.1 Change Log Entry Template

```markdown
### Change #{XXX} - [{BREAKING|MAJOR|MINOR|TRIVIAL}]
**Date**: YYYY-MM-DD HH:MM
**Component**: {Component name from Section 4}
**TI Reference**: Section {X}.{Y}

**Planned (from TI Section {X}.{Y})**:
\`\`\`python
{Copy exact spec from this TI document}
\`\`\`

**Implemented**:
\`\`\`python
{What was actually coded}
\`\`\`

**Reason for Deviation**:
{Technical reason - bug discovered, performance issue, integration conflict, dependency limitation, etc.}

**Impact Analysis**:
- [ ] TI Updates Needed: {List section numbers from this document that need updates}
- [ ] HLD Updates Needed: {List section numbers from ReviewCSVGenerationCHILD.md}
- [ ] Foundation Updates Needed: {Yes/No - rarely yes}

**Code Reference**:
- File: `{filename}:{line_range}`
- Commit: {git_sha} (if committed)

**Testing Impact**:
- [ ] Unit tests affected: {test file names}
- [ ] Integration tests affected: {test file names}
- [ ] New tests required: {Yes/No}
```

### 11.2 Severity Definitions

**[BREAKING]**: Changes public contracts, breaks downstream stages
- Removes functionality specified in TI
- Changes output schema structure (field removal, type change)
- **Impact**: Stage 4 will break
- **Example**: Removing url column from video_review.csv

**[MAJOR]**: Changes core logic/algorithms, requires HLD update
- Modifies algorithm from TI specification
- Changes validation rules
- **Impact**: Behavior differs from spec, HLD needs update
- **Example**: Changing url column position from 2 to 3

**[MINOR]**: Adds optional features, TI update only
- Adds defensive checks not specified
- Improves error messages beyond TI spec
- **Impact**: Behavior compatible with spec, TI doc update only
- **Example**: Adding null check for duration field

**[TRIVIAL]**: Performance/refactoring, no doc updates needed
- Performance optimizations preserving behavior
- Code refactoring without logic changes
- **Impact**: No doc updates needed, log for awareness
- **Example**: Caching DataFrame operations

### 11.3 When to Log a Change

**MUST LOG** when you:
- ✅ Change function signature from TI Section 4
- ✅ Modify algorithm logic from TI Section 4
- ✅ Change data types/schemas from TI Section 3
- ✅ Add new error cases not in TI Section 6
- ✅ Skip validation rules from TI Section 5
- ✅ Change file paths from TI Section 8

**DO NOT LOG** for:
- ❌ Variable name changes (as long as function signature matches)
- ❌ Code comments/docstrings additions
- ❌ Whitespace/formatting
- ❌ Import order (as long as all imports from TI Section 8 are present)

### 11.4 Implementation Log Entries

*(This section starts EMPTY. Entries are added during Phase 4: Implementation)*

---

## 12. Dependencies & Prerequisites

**Source**: ReviewCSVGenerationCHILD.md Sections 3.3, 3.4, 7.4

### 12.1 External Dependencies

```python
# Source: ReviewCSVGenerationCHILD.md Section 3.4

EXTERNAL_DEPS = {
    "pandas": {
        "version": ">=2.0.0",
        "purpose": "DataFrame operations for CSV I/O and column reordering",
        "pip_install": "pip install pandas>=2.0.0"
    },
    "json": {
        "version": "standard library",
        "purpose": "Load temporal_windows_updated.json files",
        "pip_install": "N/A (standard library)"
    },
    "pathlib": {
        "version": "standard library",
        "purpose": "File path operations",
        "pip_install": "N/A (standard library)"
    },
    "logging": {
        "version": "standard library",
        "purpose": "Logging progress and warnings",
        "pip_install": "N/A (standard library)"
    },
}
```

### 12.2 Upstream TI Requirements

```python
# Source: ReviewCSVGenerationCHILD.md Section 3.3

UPSTREAM_OUTPUTS_REQUIRED = {
    "VideoProcessingTI": [
        "{bucket_base}/analysis/insights/*_temporal_windows_updated.json",
        # Must contain metadata.url field (requires Stage 2 modification)
    ],
    "FeatureAggregationTI": [
        "{bucket_base}/ml_analysis/aggregated_features.csv",
        # Used for validation comparison (row count, column count)
    ],
}
```

### 12.3 System Prerequisites

```python
# Source: ReviewCSVGenerationCHILD.md Section 7.1

SYSTEM_REQUIREMENTS = {
    "disk_space": "1GB minimum (for input/output CSVs + temp files)",
    "memory": "100MB minimum (200MB recommended for N=300 videos)",
    "api_keys": [],  # None required
    "network": "Not required (pure local file processing)",
}
```

---

## 13. HLD Traceability Matrix

**Purpose**: Map every HLD section to TI implementation

| HLD Section | TI Section | Implementation Status |
|-------------|------------|----------------------|
| ReviewCSVGenerationCHILD.md Section 1: Context & Business Goal | Section 1: Document Metadata | To Implement |
| ReviewCSVGenerationCHILD.md Section 1.1: What Problem Does This Solve? | Section 1: Document Metadata (Rationale) | To Implement |
| ReviewCSVGenerationCHILD.md Section 1.2: Where This Fits in Pipeline | Section 1: Document Metadata (Depends On/Feeds Into) | To Implement |
| ReviewCSVGenerationCHILD.md Section 1.3: Success Criteria | Section 2.2: StageOutput (validation metrics) | To Implement |
| ReviewCSVGenerationCHILD.md Section 2.1: High-Level Approach | Section 4: Algorithmic Specifications (intro) | To Implement |
| ReviewCSVGenerationCHILD.md Section 2.2: Data Flow | Section 7: Complete Example Traces | To Implement |
| ReviewCSVGenerationCHILD.md Section 2.3.1: Prerequisite - Modify temporal_compute.py | Section 4.1: ensure_url_in_metadata() | To Implement |
| ReviewCSVGenerationCHILD.md Section 2.3.2: Load Temporal Windows Files | Section 4.2: load_temporal_windows() | To Implement |
| ReviewCSVGenerationCHILD.md Section 2.3.3: Extract Features | Section 4.3: extract_features_with_url() | To Implement |
| ReviewCSVGenerationCHILD.md Section 2.3.4: URL Validation and Row Filtering | Section 4.4: filter_videos_with_url() | To Implement |
| ReviewCSVGenerationCHILD.md Section 2.3.5: Generate video_review.csv | Section 4.5: generate_review_csv() | To Implement |
| ReviewCSVGenerationCHILD.md Section 3.1: Input Dependencies | Section 2.1: StageInput contract | To Implement |
| ReviewCSVGenerationCHILD.md Section 3.2: Output Contracts | Section 2.2: StageOutput contract | To Implement |
| ReviewCSVGenerationCHILD.md Section 3.3: Cross-Stage Dependencies | Section 12.2: Upstream TI Requirements | To Implement |
| ReviewCSVGenerationCHILD.md Section 3.4: External Dependencies | Section 12.1: External Dependencies | To Implement |
| ReviewCSVGenerationCHILD.md Section 4.1: Configuration Sources | Section 9: Configuration & Environment | To Implement |
| ReviewCSVGenerationCHILD.md Section 4.2: Environment Variables | Section 9.1: Environment Variables | To Implement |
| ReviewCSVGenerationCHILD.md Section 4.3: Runtime Parameters | Section 9.2: Configuration Object | To Implement |
| ReviewCSVGenerationCHILD.md Section 5.1: Input Schema | Section 3.2: Stage Input Schema | To Implement |
| ReviewCSVGenerationCHILD.md Section 5.2: Output Schema | Section 3.3: Stage Output Schema | To Implement |
| ReviewCSVGenerationCHILD.md Section 6.1: Input Validation | Section 5.1: Validation Rules (input) | To Implement |
| ReviewCSVGenerationCHILD.md Section 6.2: Error Cases | Section 6: Error Handling | To Implement |
| ReviewCSVGenerationCHILD.md Section 6.3: Output Validation | Section 5.3: Validation Rules (output) | To Implement |
| ReviewCSVGenerationCHILD.md Section 7: Performance & Scalability | Section 10.2: Metrics | To Implement |
| ReviewCSVGenerationCHILD.md Section 8: Testing Strategy | Section 14: Test Specifications (future) | To Implement |
| ReviewCSVGenerationCHILD.md Section 10: References | Section 14: References (this section) | To Implement |
| ReviewCSVGenerationCHILD.md Appendix A: Glossary | Section 3: Data Schemas (terminology) | To Implement |
| ReviewCSVGenerationCHILD.md Appendix B: Decision Log | Section 11: Implementation Log (future deviations) | To Implement |
| FoundationCHILD.md Section 2: Client Architecture | Section 8.4: Base Directory Structure | To Implement |
| FoundationCHILD.md Section 2.2: Path Templates | Section 8.5: Stage Output Paths | To Implement |
| FoundationCHILD.md Section 4: CLI Command Structure | Section 2.1: StageInput (CLI parameters) | To Implement |
| FoundationCHILD.md Section 5.1: config.json Schema | Section 3.1: Foundation Schemas | To Implement |

---

## 14. References

### 14.1 Source Documents

- **ReviewCSVGenerationCHILD.md v1.0**: Parent HLD specification
- **FoundationCHILD.md v1.1**: Directory structure and shared architecture
- **SystemArchitecturev2.md**: RumiAI processing pipeline architecture

### 14.2 Implementation Files

- **`/rumiai_v2/ml_pipeline/stage3_aggregation/review_csv_generator.py`**: Main implementation (to be created)
- **`/rumiai_v2/processors/temporal_compute.py`**: Stage 2 modification (line ~2650)

### 14.3 Related Stages

- **Stage 2 (Video Processing)**: Prerequisite - produces temporal_windows_updated.json with metadata.url
- **Stage 3 (Feature Aggregation)**: Prerequisite - produces aggregated_features.csv for validation comparison
- **Stage 4 (Feature Transformation)**: Consumer - NOT dependent (video_review.csv is optional)

---

## Appendix A: Complete CLI Help Output

*(To be filled during implementation)*

```bash
$ python review_csv_generator.py --help

# Not a CLI tool - invoked internally by Stage 3 pipeline
# No CLI help output
```

---

## Appendix B: Exit Code Reference

| Code | Category | Scenario | Recovery Action |
|------|----------|----------|-----------------|
| 0 | Success | video_review.csv generated successfully | None (proceed to Stage 4) |
| 1 | Pre-flight Validation | insights/ directory missing OR no JSON files found | Re-run Stage 2 (Video Processing) |
| 2 | Execution Failure | All videos missing url field | Deploy Stage 2 modification (temporal_compute.py), re-run Stage 2 |
| 4 | I/O Failure | Disk full OR permission denied | Free disk space, fix permissions, retry Stage 3.4 |

---

## Appendix C: Sample Output Files

### Sample video_review.csv (Bucket 18-33s, 3 videos)

```csv
video_id,url,duration,closing_scene_count,closing_word_count,hook_eye_contact_rate,hook_scene_count,hook_word_count,middle_1_scene_count,middle_1_word_count,middle_2_scene_count,middle_2_word_count,middle_3_scene_count,middle_3_word_count,middle_4_scene_count,middle_4_word_count
7428596413707144481,https://www.tiktok.com/@user/video/7428596413707144481,22.5,4,12,0.75,3,15,5,20,4,18,3,22,6,19
7428596413707144482,https://www.tiktok.com/@user/video/7428596413707144482,25.0,5,10,0.65,2,12,6,22,5,20,4,18,7,21
7428596413707144483,https://www.tiktok.com/@user/video/7428596413707144483,19.8,3,14,0.80,4,18,4,19,3,17,5,23,6,20
```

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-17 | RumiAI Team | Initial TI document creation from ReviewCSVGenerationCHILD.md |

---

**END OF TECHNICAL IMPLEMENTATION DOCUMENT**
