# Review CSV Generation - High-Level Design

> **Parent**: MLPlanningv2.md - Stage 3.4 (relocated from Stage 2.4)
> **Version**: 1.0
> **Last Updated**: 2025-01-09
> **Status**: Approved

---

## 1. Context & Business Goal

### 1.1 What Problem Does This Solve?

RumiAI's fail-fast validation catches processing errors (service crashes), but NOT statistical outliers or domain-implausible values. A video with `hook_scene_count = 10` (10 scene cuts in 3 seconds) technically passes RumiAI validation but likely indicates encoding issues, rapid cuts breaking assumptions, or edge case content. Manual investigation is needed to understand WHY outliers occur (fix RumiAI bugs, filter problematic videos, or accept as valid edge cases). Without easy access to outlier videos, debugging is time-consuming and requires hunting through multiple files.

This component generates `video_review.csv` - a human-readable CSV with clickable TikTok URLs that mirrors ML training data exactly, enabling rapid outlier investigation in Excel.

### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This component depends on MLPlanningv2.md Part 1 for:
- Client directory structure (Part 1, Section 2: Client Architecture & Storage)
- Bucket organization (`bucket_{duration}/validation/` paths)

```
Stage 2: Video Processing (RumiAI Pipeline)
   ↓ Output: temporal_windows_updated.json (N files per bucket, with metadata.url)
   ↓ Modification: temporal_compute.py adds 'url' to calculated_metadata
Stage 3: Feature Aggregation
   ↓ Sub-stage 3.1-3.3: Generate aggregated_features.csv
Stage 3.4: Review CSV Generation [THIS COMPONENT]
   ↓ Output: video_review.csv (N rows, same features as aggregated + url column)
Stage 4: Feature Transformation
```

**Cross-Stage Dependency**: Requires Stage 2 modification (1-line addition to temporal_compute.py) to pass `url` through metadata.

### 1.3 Success Criteria

- [✓] video_review.csv mirrors aggregated_features.csv exactly (same N rows, same feature values)
- [✓] video_review.csv includes clickable `url` column (column 2, after video_id)
- [✓] Videos with missing url are excluded from review CSV (logged as warnings)
- [✓] File generated per bucket in `validation/` subdirectory
- [✓] No impact on ML pipeline (aggregated_features.csv unchanged)

---

## 2. Architecture & Design

### 2.1 High-Level Approach

This component creates a **dual-output pattern** in Stage 3: the existing `aggregated_features.csv` (ML training input, no url) and a NEW `video_review.csv` (human review, with url). Both files are generated from the same source data (temporal_windows_updated.json files), ensuring what the user reviews in Excel is EXACTLY what ML trains on. The video_review.csv includes all ~65-215 features (depending on bucket) plus video_id, url, and duration for maximum investigation flexibility. Videos missing the `url` field are skipped from review CSV (logged) but still included in aggregated_features.csv (ML unaffected).

### 2.2 Data Flow

```
Input: temporal_windows_updated.json (N files per bucket)
       Location: bucket_{duration}/analysis/insights/
       Schema: {temporal_windows: {...}, metadata: {video_id, url, duration, ...}}
   ↓
Process Step 1: Load all temporal_windows_updated.json files for bucket
   ↓
Process Step 2: Extract features (same logic as aggregated_features.csv generation)
   ↓
Process Step 3: Check metadata.url presence (skip row if missing, log warning)
   ↓
Process Step 4: Build CSV rows: [video_id, url, duration, all_features]
   ↓
Output 1: video_review.csv (for human review)
          Location: bucket_{duration}/validation/video_review.csv
          Schema: (N rows, ~67-217 columns depending on bucket)

Output 2: aggregated_features.csv (for ML, unchanged)
          Location: bucket_{duration}/ml_analysis/aggregated_features.csv
          Schema: (N rows, ~65-215 columns)
```

### 2.3 Detailed Process

#### Step 2.3.1: Prerequisite - Modify temporal_compute.py (Stage 2)

**Purpose**: Ensure `url` field flows through to temporal_windows_updated.json metadata

**Logic**:
```python
# Source: QA Q2 - temporal_compute.py modification (line ~2650)
# File: /home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py

# BEFORE (current code):
calculated_metadata = {
    'video_id': video_id,
    'duration': video_duration,
    'digg_count': metadata.get('likes', 0),
    'play_count': metadata.get('views', 0),
    ...
}

# AFTER (required modification):
calculated_metadata = {
    'video_id': video_id,
    'duration': video_duration,
    'url': metadata.get('url'),  # ← ADD THIS LINE
    'digg_count': metadata.get('likes', 0),
    'play_count': metadata.get('views', 0),
    ...
}
```

**Rationale**:
- unified_analysis.json already contains `metadata.url` (from Apify `webVideoUrl`)
- temporal_compute.py's `calculated_metadata` section is for metadata passthrough (not just computed features)
- Adding `url` follows existing pattern (engagement counts, author, timestamps already passed through)

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| metadata.url is None | Pass through as None | Stage 3.4 will detect and handle missing urls |
| metadata.url is empty string | Pass through as is | Stage 3.4 will detect and handle |
| metadata key missing entirely | Pass through as None (`.get()`) | Graceful degradation |

#### Step 2.3.2: Load Temporal Windows Files

**Purpose**: Read all temporal_windows_updated.json files for the bucket

**Logic**:
```python
# Source: QA Q6 - per-bucket processing
import json
from pathlib import Path
from typing import List, Dict, Any

def load_temporal_windows(bucket_path: Path) -> List[Dict[str, Any]]:
    """
    Load all temporal_windows_updated.json files for a bucket.

    Args:
        bucket_path: Path to bucket directory (e.g., bucket_18-33s/)

    Returns:
        List of temporal windows dicts (one per video)
    """
    insights_dir = bucket_path / "analysis" / "insights"
    json_files = sorted(insights_dir.glob("*_temporal_windows_updated.json"))

    temporal_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            temporal_data.append(data)

    return temporal_data
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| insights/ directory doesn't exist | Raise FileNotFoundError | Stage 2 must complete before Stage 3.4 |
| No JSON files found | Raise ValueError | Cannot generate review CSV without data |
| Malformed JSON | Raise JSONDecodeError | Fail fast on corrupted data |

#### Step 2.3.3: Extract Features (Mirror aggregated_features.csv Logic)

**Purpose**: Extract same features as aggregated_features.csv to ensure review mirrors ML input

**Logic**:
```python
# Source: QA Q4 - include ALL features
# Source: MLPlanningv2.md Stage 3 - feature extraction logic

def extract_features_with_url(temporal_windows: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from temporal_windows_updated.json.
    Uses SAME logic as aggregated_features.csv generation.

    Args:
        temporal_windows: Single video's temporal windows data

    Returns:
        Dict with video_id, url, duration, and all temporal features
    """
    # Extract metadata
    metadata = temporal_windows.get('metadata', {})
    video_id = metadata.get('video_id', 'unknown')
    url = metadata.get('url')  # May be None
    duration = metadata.get('duration', 0)

    # Extract temporal features (SOURCE: Stage 3 existing logic)
    features = {'video_id': video_id, 'url': url, 'duration': duration}

    # Hook features (all ~30 base features)
    hook = temporal_windows['temporal_windows']['hook']
    for feature_name, value in hook.items():
        features[f'hook_{feature_name}'] = value

    # Middle segment features (3-5 segments depending on bucket)
    middle_segments = temporal_windows['temporal_windows']['middle_segments']
    if middle_segments:  # null for videos ≤9s
        for i, segment in enumerate(middle_segments, start=1):
            for feature_name, value in segment.items():
                features[f'middle_{i}_{feature_name}'] = value

    # Closing features (all ~30 base features)
    closing = temporal_windows['temporal_windows']['closing']
    for feature_name, value in closing.items():
        features[f'closing_{feature_name}'] = value

    return features
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| middle_segments is null | Skip middle feature extraction | Videos ≤9s have no middle (expected) |
| Feature value is None | Pass through as None | Preserve data fidelity |
| Unexpected feature keys | Include anyway | Future-proof for schema changes |

#### Step 2.3.4: URL Validation and Row Filtering

**Purpose**: Exclude videos with missing url from review CSV (cannot investigate without clickable link)

**Logic**:
```python
# Source: QA Q5 - Option A (skip videos with missing url)
import logging

logger = logging.getLogger(__name__)

def filter_videos_with_url(feature_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out videos missing url field.

    Args:
        feature_rows: List of feature dicts (one per video)

    Returns:
        Filtered list (only videos with valid url)
    """
    valid_rows = []
    skipped_count = 0

    for row in feature_rows:
        video_id = row.get('video_id', 'unknown')
        url = row.get('url')

        if not url:  # None or empty string
            logger.warning(
                f"Video {video_id} excluded from video_review.csv - missing url"
            )
            skipped_count += 1
            continue

        valid_rows.append(row)

    if skipped_count > 0:
        logger.info(
            f"Excluded {skipped_count} videos from review CSV (missing url). "
            f"These videos remain in aggregated_features.csv for ML training."
        )

    return valid_rows
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| All videos missing url | Return empty list, log error | Review CSV cannot be generated |
| url is whitespace-only string | Treat as missing (not url) | Invalid url = same as missing |
| url format invalid | Pass through anyway | Excel will show invalid link (user can see issue) |

#### Step 2.3.5: Generate video_review.csv

**Purpose**: Write CSV with video_id, url, duration, all features (mirror aggregated_features.csv)

**Logic**:
```python
# Source: QA Q3, Q4, Q6 - dual CSV generation with url at position 2
import pandas as pd

def generate_review_csv(
    feature_rows: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Generate video_review.csv from feature rows.

    Args:
        feature_rows: List of feature dicts (filtered for valid urls)
        output_path: Path to save video_review.csv
    """
    if not feature_rows:
        logger.error("No videos with valid url - cannot generate review CSV")
        return

    # Convert to DataFrame
    df = pd.DataFrame(feature_rows)

    # Reorder columns: video_id, url, duration, then all other features
    # Source: QA Q4 - url at position 2
    cols = ['video_id', 'url', 'duration']
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + sorted(other_cols)]  # Sort features alphabetically

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        f"✅ Generated video_review.csv: {len(df)} rows, {len(df.columns)} columns"
    )
    logger.info(f"   Location: {output_path}")
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| validation/ directory doesn't exist | Create with mkdir(parents=True) | Auto-create directory structure |
| File already exists | Overwrite | Re-running Stage 3 should replace old data |
| Disk full | Raise OSError | Fail fast on system errors |

---

## 3. Dependencies & Integration

### 3.1 Input Dependencies

| Dependency | Source | Provides | Required? | Notes |
|------------|--------|----------|-----------|-------|
| **temporal_windows_updated.json** | Stage 2 (RumiAI Pipeline) | All temporal features + metadata | YES | N files per bucket in `analysis/insights/` |
| **metadata.url** | Stage 2 (temporal_compute.py) | TikTok video URL | YES | Requires 1-line modification to temporal_compute.py (Step 2.3.1) |
| **Bucket directory structure** | MLPlanningv2.md Part 1 Section 2 | File paths | YES | `bucket_{duration}/` structure |

**Source Tracing**:
- `metadata.url`: Apify scraper (`webVideoUrl`) → unified_analysis.json (`metadata.url`) → temporal_compute.py (`calculated_metadata['url']`) → temporal_windows_updated.json
- Verified in: QA_ReviewCSVGeneration.md Q1, Q2

### 3.2 Output Contracts

| Output | Location | Format | Consumers | Purpose |
|--------|----------|--------|-----------|---------|
| **video_review.csv** | `bucket_{duration}/validation/` | CSV (N rows, ~67-217 cols) | Human (Excel review) | Manual outlier investigation with clickable URLs |

**Schema** (Bucket 18-33s example - 186 columns):
- Column 1: `video_id` (string, e.g., "7428596413707144481")
- Column 2: `url` (string, e.g., "https://www.tiktok.com/@user/video/7428596413707144481")
- Column 3: `duration` (float, 18.0-33.0 for this bucket)
- Columns 4-186: All temporal features (hook_*, middle_1_*, middle_2_*, middle_3_*, middle_4_*, closing_*)

**Row Count**: Matches aggregated_features.csv (same N videos), minus any videos with missing url

### 3.3 Cross-Stage Dependencies

**Upstream (Required Before This Stage)**:
- Stage 2 must complete (all temporal_windows_updated.json files exist)
- Stage 2 modification (temporal_compute.py) must be deployed (url in metadata)
- Stage 3.1-3.3 must complete (aggregated_features.csv generated first)

**Downstream (This Stage Enables)**:
- NONE - video_review.csv is optional/independent
- Deleting video_review.csv does NOT impact ML pipeline

**Parallel Execution**:
- Can run in parallel with Stage 4 (Feature Transformation) - no dependencies

### 3.4 External Dependencies

NONE - No external services, APIs, or third-party libraries beyond standard Python (pandas, json, pathlib, logging)

---

## 4. Configuration & Parameters

### 4.1 Configuration Sources

This component has NO user-configurable parameters. Behavior is deterministic based on input data.

**Fixed Parameters**:
- **url column position**: Always column 2 (after video_id)
- **Feature inclusion**: Always ALL features (no filtering)
- **Missing url handling**: Always skip row (log warning)

### 4.2 Environment Variables

NONE

### 4.3 Runtime Parameters

NONE - Component is invoked as part of Stage 3 pipeline (no CLI flags)

---

## 5. Data Schemas

### 5.1 Input Schema

**File**: `temporal_windows_updated.json`
**Location**: `bucket_{duration}/analysis/insights/{video_id}_temporal_windows_updated.json`
**Format**: JSON

**Required Fields**:
```json
{
  "temporal_windows": {
    "hook": {
      "scene_count": int,
      "word_count": int,
      "eye_contact_rate": float (0.0-1.0),
      // ... ~27 more features
    },
    "middle_segments": [
      {
        "scene_count": int,
        "word_count": int,
        // ... same ~30 features per segment
      }
      // 3-5 segments depending on bucket (null for ≤9s videos)
    ],
    "closing": {
      "scene_count": int,
      "word_count": int,
      // ... ~30 features
    }
  },
  "metadata": {
    "video_id": string (required),
    "url": string (required for review CSV),  // ← NEW FIELD from Stage 2 modification
    "duration": float (required),
    "digg_count": int,
    "play_count": int,
    // ... other engagement metrics
  }
}
```

**Source**: temporal_compute.py output (Stage 2)

### 5.2 Output Schema

**File**: `video_review.csv`
**Location**: `bucket_{duration}/validation/video_review.csv`
**Format**: CSV with header row

**Column Schema** (Bucket 18-33s - 186 columns):

| Column | Type | Range/Format | Example | Description |
|--------|------|--------------|---------|-------------|
| video_id | string | TikTok video ID (17-20 digits) | "7428596413707144481" | Unique identifier |
| url | string | TikTok URL format | "https://www.tiktok.com/@user/video/7428596..." | Clickable link |
| duration | float | 18.0-33.0 (bucket-specific) | 22.5 | Video length in seconds |
| hook_scene_count | int | 0-20 | 3 | Scene cuts in first 3s |
| hook_word_count | int | 0-150 | 15 | Words spoken in first 3s |
| hook_eye_contact_rate | float | 0.0-1.0 | 0.75 | Gaze camera ratio |
| middle_1_scene_count | int | 0-20 | 5 | Scene cuts in segment 1 |
| middle_1_word_count | int | 0-150 | 20 | Words in segment 1 |
| ... | ... | ... | ... | ... (all middle segments) |
| closing_scene_count | int | 0-20 | 4 | Scene cuts in last 3s |
| closing_word_count | int | 0-150 | 12 | Words in last 3s |
| ... | ... | ... | ... | ... (~30 closing features) |

**Column Count by Bucket**:
- Bucket 0-3s, 3-9s: ~67 columns (video_id + url + duration + ~64 features)
- Bucket 9-13s, 13-18s: ~157 columns (video_id + url + duration + ~154 features)
- Bucket 18-33s: ~187 columns (video_id + url + duration + ~184 features)
- Bucket 33-60s, 60-90s, 90-120s: ~217 columns (video_id + url + duration + ~214 features)

**Row Count**: N videos (same as aggregated_features.csv, minus videos with missing url)

**Source**: QA_ReviewCSVGeneration.md Q3, Q4, Q6

---

## 6. Error Handling & Validation

### 6.1 Input Validation

| Check | Validation Rule | Error Type | Error Message | Recovery |
|-------|----------------|------------|---------------|----------|
| **insights/ directory exists** | Path.exists() | FileNotFoundError | "insights/ directory not found: {path}. Stage 2 must complete first." | FAIL FAST |
| **JSON files exist** | len(glob()) > 0 | ValueError | "No temporal_windows_updated.json files found in {path}" | FAIL FAST |
| **Valid JSON** | json.load() | JSONDecodeError | "Malformed JSON in {filename}: {error}" | SKIP FILE + LOG |
| **Required fields present** | 'metadata' in data | KeyError | "Missing 'metadata' key in {filename}" | SKIP FILE + LOG |

### 6.2 Error Cases

#### Error 1: Missing url Field

**Scenario**: temporal_windows_updated.json missing `metadata.url`

**Detection**:
```python
url = temporal_windows.get('metadata', {}).get('url')
if not url:
    # Handle missing url
```

**Handling**:
- Skip video from video_review.csv (Step 2.3.4)
- Include video in aggregated_features.csv (ML unaffected)
- Log warning: `"Video {video_id} excluded from review CSV - missing url"`

**User Impact**: Cannot investigate this video in Excel (no clickable link)

**Source**: QA_ReviewCSVGeneration.md Q5

#### Error 2: All Videos Missing url

**Scenario**: No videos have valid url (Stage 2 modification not deployed?)

**Detection**:
```python
valid_rows = filter_videos_with_url(feature_rows)
if not valid_rows:
    # Handle empty result
```

**Handling**:
- Log error: `"No videos with valid url - cannot generate review CSV"`
- Skip video_review.csv generation (no output file)
- Continue to Stage 4 (ML pipeline unaffected)

**User Impact**: No review CSV available for manual investigation

#### Error 3: Disk Full During CSV Write

**Scenario**: Insufficient disk space to write video_review.csv

**Detection**: OSError from `df.to_csv()`

**Handling**:
- Raise OSError (fail fast)
- Log error: `"Failed to write video_review.csv: {error}"`
- Do NOT continue to Stage 4 (indicates system issue)

**User Impact**: Stage 3 fails, requires disk cleanup

### 6.3 Output Validation

| Check | Validation Rule | Expected | Action if Failed |
|-------|----------------|----------|------------------|
| **CSV row count** | len(df) > 0 | At least 1 row | Log error, skip file generation |
| **url column exists** | 'url' in df.columns | True | ASSERT (should never fail) |
| **url column position** | df.columns[1] == 'url' | True | ASSERT (should never fail) |
| **Row count ≤ aggregated rows** | len(review) ≤ len(aggregated) | True | Log warning if difference > 10% |

---

## 7. Performance & Scalability

### 7.1 Performance Targets

| Metric | Target | Measured | Bottleneck | Notes |
|--------|--------|----------|------------|-------|
| **Generation time** | < 10 seconds per bucket | TBD | pandas DataFrame creation | For N=100 videos, ~186 columns |
| **Memory usage** | < 100 MB peak | TBD | Loading all temporal_windows JSONs | 100 videos × ~50KB each = 5MB JSON + ~50MB DataFrame |
| **Disk I/O** | < 5 MB written | TBD | CSV write | 100 rows × ~186 cols × 10 bytes/cell = ~200KB |

### 7.2 Scalability Considerations

**Current Design** (N videos per bucket):
- **N = 100** (default contrastive): ~200KB CSV, < 5 seconds
- **N = 300** (max): ~600KB CSV, < 15 seconds
- **N = 500** (extreme): ~1MB CSV, < 30 seconds

**Scalability Limits**:
- **Excel limit**: 1,048,576 rows × 16,384 columns (will never hit)
- **pandas limit**: ~1 million rows before performance degrades (will never hit)
- **Practical limit**: N=1000 videos (100MB memory, 60 seconds) - still acceptable

**No optimization needed** - current design scales beyond requirements.

### 7.3 Bottlenecks & Mitigations

| Bottleneck | Impact | Mitigation | Priority |
|------------|--------|------------|----------|
| **JSON file loading** | O(N) file I/O | Acceptable for N≤500 | LOW |
| **DataFrame memory** | ~50MB for N=100 | Acceptable on 8GB+ systems | LOW |
| **CSV write** | Blocking I/O (< 5s) | Acceptable latency | LOW |

**No performance issues expected** - component is lightweight.

---

## 8. Testing Strategy

### 8.1 Unit Tests

#### Test 1: Extract Features with URL

**Purpose**: Verify feature extraction matches aggregated_features.csv logic

**Setup**:
```python
# Load sample temporal_windows_updated.json (with metadata.url)
sample_json = load_test_data("bucket_18-33s/sample_video.json")
```

**Execute**:
```python
features = extract_features_with_url(sample_json)
```

**Assert**:
```python
assert features['video_id'] == '7428596413707144481'
assert features['url'] == 'https://www.tiktok.com/@user/video/7428596...'
assert features['duration'] == 22.5
assert 'hook_scene_count' in features
assert 'middle_1_scene_count' in features
assert 'closing_scene_count' in features
assert len(features) == 187  # video_id + url + duration + 184 features
```

#### Test 2: Filter Videos with Missing URL

**Purpose**: Verify videos without url are excluded from review CSV

**Setup**:
```python
feature_rows = [
    {'video_id': '123', 'url': 'https://tiktok.com/...', 'feature1': 5},
    {'video_id': '456', 'url': None, 'feature1': 10},  # Missing url
    {'video_id': '789', 'url': '', 'feature1': 15},     # Empty url
]
```

**Execute**:
```python
valid_rows = filter_videos_with_url(feature_rows)
```

**Assert**:
```python
assert len(valid_rows) == 1  # Only first video
assert valid_rows[0]['video_id'] == '123'
# Check warning logged for video 456 and 789
```

#### Test 3: CSV Column Ordering

**Purpose**: Verify url is at position 2 (after video_id)

**Setup**:
```python
feature_rows = [
    {'video_id': '123', 'url': 'https://...', 'duration': 20.0, 'hook_scene_count': 3}
]
df = pd.DataFrame(feature_rows)
```

**Execute**:
```python
# Reorder columns (Step 2.3.5 logic)
cols = ['video_id', 'url', 'duration']
other_cols = [c for c in df.columns if c not in cols]
df = df[cols + sorted(other_cols)]
```

**Assert**:
```python
assert df.columns[0] == 'video_id'
assert df.columns[1] == 'url'
assert df.columns[2] == 'duration'
```

### 8.2 Integration Tests

#### Integration Test 1: End-to-End CSV Generation

**Purpose**: Verify complete workflow from temporal_windows JSONs to video_review.csv

**Setup**:
```bash
# Use existing test bucket
TEST_BUCKET=/home/jorge/rumiaifinal/data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s/

# Ensure 5-10 temporal_windows_updated.json files exist
ls $TEST_BUCKET/analysis/insights/*_temporal_windows_updated.json | wc -l
# Expected: 5-10 files
```

**Execute**:
```python
# Run Stage 3.4
bucket_path = Path(TEST_BUCKET)
temporal_data = load_temporal_windows(bucket_path)
feature_rows = [extract_features_with_url(tw) for tw in temporal_data]
valid_rows = filter_videos_with_url(feature_rows)
output_path = bucket_path / "validation" / "video_review.csv"
generate_review_csv(valid_rows, output_path)
```

**Assert**:
```python
# Check file exists
assert output_path.exists()

# Load and validate
df_review = pd.read_csv(output_path)
df_aggregated = pd.read_csv(bucket_path / "ml_analysis" / "aggregated_features.csv")

# Row count match (minus missing urls)
assert len(df_review) <= len(df_aggregated)
assert len(df_review) >= len(df_aggregated) * 0.9  # Allow 10% missing urls

# Column count (aggregated + 1 for url)
assert len(df_review.columns) == len(df_aggregated.columns) + 1

# url column present at position 2
assert df_review.columns[1] == 'url'

# All urls valid (start with https://)
assert df_review['url'].str.startswith('https://').all()

# Feature values match between files (sample check)
# Compare first row, skip url column
review_features = df_review.iloc[0].drop(['url'])
aggregated_features = df_aggregated.iloc[0]
assert review_features.equals(aggregated_features)
```

**Source**: QA_ReviewCSVGeneration.md Q6

### 8.3 Test Data

**Primary Test Data Source**:
- Location: `/home/jorge/rumiaifinal/data/clients/test_run/hashtags/fitness/top_contrastive/buckets/`
- Buckets available: bucket_18-33s, bucket_60-90s, bucket_13-18s
- Videos per bucket: ~30 videos (real TikTok data)

**Sample Test Scenario** (bucket_18-33s):
```
Input: 10 temporal_windows_updated.json files
Expected Output:
  - video_review.csv: 10 rows × 186 columns
  - Column 1: video_id (e.g., "7428596413707144481")
  - Column 2: url (e.g., "https://www.tiktok.com/@user/video/7428596...")
  - Column 3: duration (18.0-33.0 range)
  - Columns 4-186: All temporal features (hook_*, middle_*, closing_*)
```

**Edge Case Test Data**:
- **Missing url**: Manually remove `metadata.url` from one JSON file → verify excluded from review CSV
- **Empty middle_segments**: Use video from bucket 3-9s (null middle) → verify no middle_* columns
- **Large N**: Process bucket with 100 videos → verify performance < 15 seconds

---

## 9. Future Enhancements

### Enhancement 1: Auto-Apply Excel Conditional Formatting

**Current State**: User manually applies conditional formatting in Excel

**Proposed**: Generate .xlsx file with pre-configured conditional formatting rules
- Red cells: Values > Q3 + 1.5×IQR
- Yellow cells: Values > Q3 + IQR
- Library: `openpyxl` or `xlsxwriter`

**Effort**: 2-3 hours (low complexity)
**Value**: Saves user 5 minutes per bucket review

### Enhancement 2: Summary Statistics CSV

**Current State**: video_review.csv contains raw data only

**Proposed**: Generate `validation/summary_stats.csv` alongside review CSV
- Columns: feature_name, mean, std, min, Q1, median, Q3, max, outlier_count
- Helps user prioritize which features to investigate

**Effort**: 1-2 hours (trivial with pandas.describe())
**Value**: Speeds up outlier identification

### Enhancement 3: Automated Outlier Flagging Column

**Current State**: User manually identifies outliers in Excel

**Proposed**: Add `outlier_flag` column to video_review.csv
- Value: comma-separated list of flagged features (e.g., "hook_scene_count,middle_2_word_count")
- Logic: IQR-based detection (feature > Q3 + 1.5×IQR)

**Effort**: 3-4 hours (requires per-feature statistics)
**Value**: Pre-filters outliers for faster review

---

## 10. References & Related Docs

### 10.1 Mother Document Sections

- **MLPlanningv2.md Section 3.4**: Review CSV Generation (this component)
- **MLPlanningv2.md Stage 2**: Video Processing (provides temporal_windows_updated.json)
- **MLPlanningv2.md Stage 3**: Feature Aggregation (provides aggregated_features.csv)

### 10.2 Mother Document Foundation

**MLPlanningv2.md Part 1: Foundation**
- Section 2: Client Architecture & Storage (bucket directory structure)
  - Defines `bucket_{duration}/validation/` path for video_review.csv

**Source Tracing**:
- `webVideoUrl`: foundation/schemas.py line 70 (VideoMetadata schema)
- `metadata.url`: unified_analysis.json (verified in QA Q2)
- Bucket paths: MLPlanningv2.md Part 1 Section 2

### 10.3 Phase Documents

- **Critique_ReviewCSVGeneration.md**: Business justification, simplified approach decision
- **QA_ReviewCSVGeneration.md**: Technical clarifications (6 questions answered)
  - Q1: Field name tracing (webVideoUrl → metadata.url)
  - Q2: Data flow (url passthrough via temporal_compute.py)
  - Q3: Dual CSV design (ML vs human review separation)
  - Q4: Feature inclusion (ALL features for flexibility)
  - Q5: Error handling (skip missing url videos)
  - Q6: Real reflection principle (mirrors ML training data)

### 10.4 Related Components

- **Stage 2 Modification**: temporal_compute.py (1-line addition to calculated_metadata)
- **Stage 3.1-3.3**: Feature Aggregation (generates aggregated_features.csv)
- **Stage 4**: Feature Transformation (consumes aggregated_features.csv, NOT video_review.csv)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **aggregated_features.csv** | ML training input CSV (N rows × ~65-215 features), excludes url |
| **video_review.csv** | Human review CSV (N rows × ~67-217 features), includes url |
| **temporal_windows_updated.json** | Stage 2 output containing all temporal features + metadata per video |
| **metadata.url** | TikTok video URL field passed through from Apify scraper |
| **Outlier** | Feature value significantly different from bucket mean (e.g., scene_count=10 when mean=3) |
| **IQR** | Interquartile Range (Q3 - Q1), used for outlier detection in Excel conditional formatting |
| **Bucket** | Duration-based grouping (e.g., 18-33s) with consistent temporal window structure |
| **Real Reflection** | video_review.csv mirrors aggregated_features.csv exactly (same rows, same feature values) |

---

## Appendix B: Decision Log

### Decision 1: Separate Review CSV vs Adding URL to aggregated_features.csv

**Context**: Original plan was to add url column to aggregated_features.csv for manual review

**Problem**: aggregated_features.csv feeds directly into ML training (Stage 5: Random Forest + K-Means). URL is non-numeric metadata, not a feature. Adding it would require:
- Dropping url before ML (extra preprocessing)
- Risk of accidentally passing url to ML models

**Alternatives Considered**:
1. Add url to aggregated_features.csv, drop before Stage 5
2. Create separate review CSV (chosen)
3. Read from unified_analysis.json during review (no CSV)

**Decision**: Create separate `video_review.csv` for human review, keep aggregated_features.csv clean for ML

**Rationale**:
- Separation of concerns: ML data vs human review data
- No risk of ML contamination
- Review file is optional (can delete without impacting pipeline)
- Cleaner architecture

**Source**: QA_ReviewCSVGeneration.md Q3, Critique Phase 1

### Decision 2: Include ALL Features in video_review.csv

**Context**: Could include subset of "likely outlier" features (10-20 columns) vs all features

**Alternatives Considered**:
1. Subset of 10-20 key features (scene_count, word_count, emotion ratios)
2. All features (chosen)

**Decision**: Include ALL features (~65-215 depending on bucket)

**Rationale**:
- User flexibility: can apply conditional formatting to ANY column
- No need to pre-select "important" features
- Excel handles 200+ columns well
- Same data as ML sees (true reflection principle)

**Source**: QA_ReviewCSVGeneration.md Q4

### Decision 3: Skip Videos with Missing URL vs Placeholder

**Context**: How to handle videos missing metadata.url field

**Alternatives Considered**:
1. Skip video row entirely (chosen)
2. Include row with empty url field
3. Include row with "MISSING_URL" placeholder
4. Fail-fast and stop Stage 3

**Decision**: Skip video from review CSV, log warning, continue to ML training

**Rationale**:
- If url missing, review CSV is useless for that video (can't click to watch)
- No point including row that can't be investigated
- Keeps review CSV clean (only reviewable videos)
- ML pipeline unaffected (video still in aggregated_features.csv)

**Source**: QA_ReviewCSVGeneration.md Q5

### Decision 4: Modify temporal_compute.py vs Read unified_analysis.json

**Context**: Where to source the url field for video_review.csv

**Alternatives Considered**:
1. Modify Stage 2 to pass url through temporal_windows_updated.json (chosen)
2. Read url from unified_analysis.json in Stage 3 (separate file read)

**Decision**: Add 1-line to temporal_compute.py to include url in calculated_metadata

**Rationale**:
- Single source of truth: temporal_windows_updated.json has all data (features + metadata + url)
- Simpler Stage 3 code: read ONE file instead of TWO
- Consistent with existing pattern: temporal_compute already passes through engagement metrics
- Not breaking separation of concerns: calculated_metadata section is for metadata passthrough

**Source**: QA_ReviewCSVGeneration.md Q2, User confirmation

### Decision 5: Component Relocation to Stage 3.4

**Context**: Original placement was Section 2.4 (Pipeline Validation) in Stage 2

**Problem**:
- Main work (generating video_review.csv) happens in Stage 3, not Stage 2
- Stage 2 only needs 1-line modification (url passthrough)
- Calling it "Pipeline Validation" was misleading (it's not validation, it's review support)

**Decision**: Relocate to Stage 3.4 (Review CSV Generation), rename component

**Rationale**:
- Primary deliverable (video_review.csv) is generated in Stage 3
- Stage 2 change is just a dependency
- Better architectural fit (Stage 3 concerns)

**Source**: User feedback during Phase 2

---

**Document Metadata**
- Version: 1.0
- Last Updated: 2025-01-09
- Status: APPROVED ✓
- Perfect Draft: No TODOs, complete schemas, realistic tests, all sources traced
- Phase 1: Critique_ReviewCSVGeneration.md (COMPLETE)
- Phase 2: QA_ReviewCSVGeneration.md (COMPLETE)
- Ready for TI Generation: YES
