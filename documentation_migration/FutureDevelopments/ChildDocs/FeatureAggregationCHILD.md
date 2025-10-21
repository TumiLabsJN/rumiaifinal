# Feature Aggregation - High-Level Design

> **Parent**: MLPlanningv2.md - Stage 3
> **Version**: 1.1
> **Last Updated**: 2025-01-28
> **Status**: Production-Ready

---

## 1. Context & Business Goal

### 1.1 What Problem Does This Solve?

Machine learning algorithms require fixed-size feature vectors, but raw temporal window data has variable-length structures (2-7 windows depending on video duration). This component eliminates the ragged array problem by organizing videos into duration buckets where all videos share identical temporal window structures. Each bucket processes videos with consistent window counts, enabling direct CSV aggregation while preserving full temporal granularity and narrative pacing patterns critical for creative analysis.

**Middle Segment Aggregation**: For buckets 9-13s and 13-18s, middle segments are aggregated into a single "middle_aggregate" window instead of kept separate (middle_1, middle_2, middle_3). This ensures all 21 base features are reliably measured, as individual middle windows in these buckets are too short (1-4s) for features like scene_count, speech_coverage, and scene_duration_variance to produce stable values. The aggregation uses four strategies: SUM for count features (scene_count, word_count), MIN/MAX for extreme values (shortest_scene, longest_scene), MODE for categorical features (dominant_emotion_id, has_captions), and AVERAGE for all continuous/ratio features. Buckets 18-33s and longer preserve separate middle segments as their windows (3-22.8s each) are long enough for reliable feature extraction.

### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This component depends on FoundationCHILD.md for:
- Client directory structure (Section 2: Client Architecture & Storage)
- Configuration patterns (Section 4: CLI Command Structure & Configuration Dimensions)
- Temporal window definitions (Section 3: Data Schemas - temporal_windows_updated.json)

```
Stage 1: Video Discovery & Selection
   ↓ Output: selected_videos.json (per bucket, N videos)
Stage 2: Video Processing (RumiAI Pipeline)
   ↓ Output: temporal_windows_updated.json files (N files, flat in /insights/)
Stage 2.5: File Organization (NEW DEPENDENCY)
   ↓ Output: temporal_windows_updated.json organized into bucket directories
   ↓         bucket_{duration}/analysis/insights/*.json
Stage 3: Feature Aggregation (THIS COMPONENT)
   ↓ Output: aggregated_features.csv + aggregation_summary.json
Stage 4: Feature Transformation
```

**Critical**: Stage 2.5 must complete before Stage 3 can begin. Stage 2.5 reads duration from each temporal_windows_updated.json and moves files from flat `/insights/` to bucket-specific directories.

### 1.3 Success Criteria

- [ ] Process 30-200 videos per bucket in under 5 minutes with < 2GB memory
- [ ] Generate valid aggregated_features.csv with exact column count matching bucket configuration (44/65/128/149 features)
- [ ] Graceful error handling - skip bad videos with clear logging, fail only if zero valid videos remain
- [ ] Dual output - clean ML training CSV + metadata summary JSON for debugging
- [ ] All output schemas validated before saving (atomic write pattern prevents partial corruption)

---

## 2. Architecture & Design

### 2.1 High-Level Approach

Stage 3 is invoked once per bucket with a bucket directory path. It reads all temporal_windows_updated.json files from the bucket's insights directory, extracts exactly 21 base features from each temporal window (hook, middle segments, closing), and creates a flat CSV with snake_case column names (e.g., `hook_scene_count`, `middle_1_word_count`). The bucket-specific window structure ensures all videos produce identical column counts. Graceful error handling skips corrupted files while continuing batch processing. Atomic writes prevent partial CSV corruption.

### 2.2 Data Flow

```
Input: bucket_{duration}/analysis/insights/{video_id}_temporal_windows_updated.json
       Schema: JSON with temporal_windows.hook, .middle_segments[], .closing + metadata
       Location: Organized by Stage 2.5 into bucket directories
   ↓
Process Step 1: Validate bucket directory exists and contains JSON files (defensive checks)
   ↓
Process Step 2: Load each JSON, extract 21 features × N windows per video
   ↓
Process Step 3: Handle errors gracefully - skip malformed JSON, null middle_segments, duplicates
   ↓
Process Step 4: Convert aggregated list to pandas DataFrame, validate schema
   ↓
Process Step 5: Atomic write - save to temp file, then rename to final CSV
   ↓
Output 1: bucket_{duration}/ml_analysis/aggregated_features.csv
          Schema: (N videos, 45-150 features) - bucket-specific column count
          Location: Same bucket directory, ml_analysis/ subdirectory
Output 2: bucket_{duration}/ml_analysis/aggregation_summary.json
          Schema: Metadata (timestamp, counts, skip reasons, column names)
          Location: Same ml_analysis/ directory (debugging only, not consumed by Stage 4)
```

### 2.3 Detailed Process

#### Step 2.3.1: Dependency Validation

**Purpose**: Verify all prerequisites before processing (fail fast with clear error messages)

**Logic**:
```python
def validate_dependencies(bucket_path: Path):
    """
    Validate all prerequisites before processing.

    Source: QA Q3 (defensive validation strategy)

    Args:
        bucket_path: Path to bucket directory (e.g., bucket_18-33s/)

    Raises:
        ValueError: if validation fails with specific error message
    """
    # 1. Check bucket path exists
    if not bucket_path.exists():
        raise ValueError(f"Bucket path does not exist: {bucket_path}")

    # 2. Check insights directory exists and has files
    insights_dir = bucket_path / "analysis" / "insights"
    if not insights_dir.exists():
        raise ValueError(
            f"Insights directory missing: {insights_dir}. "
            "Did Stage 2.5 complete?"
        )

    json_files = list(insights_dir.glob("*_temporal_windows_updated.json"))
    if len(json_files) == 0:
        raise ValueError(
            f"No temporal_windows_updated.json files found in {insights_dir}. "
            "Did Stage 2.5 complete?"
        )

    # 3. Check ml_analysis directory is writable
    ml_analysis_dir = bucket_path / "ml_analysis"
    ml_analysis_dir.mkdir(parents=True, exist_ok=True)

    # Test write permissions
    test_file = ml_analysis_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        raise ValueError(
            f"Cannot write to {ml_analysis_dir}. Check permissions."
        )

    return len(json_files)  # Return count for logging
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Bucket directory missing | Raise ValueError with clear message | Infrastructure issue - fail fast before processing |
| Insights directory empty | Raise ValueError mentioning Stage 2.5 | User may have forgotten to run Stage 2.5 |
| ml_analysis/ not writable | Raise ValueError about permissions | Fail before processing to avoid wasted work |

#### Step 2.3.2: Feature Extraction Per Video

**Purpose**: Extract 21 base features from each temporal window, handling bucket-specific window counts

**Logic**:
```python
def extract_features(temporal_windows_json: Path, bucket: str):
    """
    Extract features from temporal_windows_updated.json.

    Source: QA Q1, Q2, Q9 (column naming, structure, exact counts)

    Args:
        temporal_windows_json: Path to JSON file
        bucket: Bucket name (e.g., "18-33s") for validation

    Returns:
        dict: Feature dictionary with snake_case column names

    Raises:
        ValueError: if JSON structure invalid or middle_segments missing
    """
    # Load JSON
    with open(temporal_windows_json) as f:
        data = json.load(f)

    # Extract video_id from filename
    video_id = temporal_windows_json.stem.replace('_temporal_windows_updated', '')

    # Validate required fields exist
    if 'metadata' not in data:
        raise ValueError(f"Video {video_id}: Missing 'metadata' field")
    if 'temporal_windows' not in data:
        raise ValueError(f"Video {video_id}: Missing 'temporal_windows' field")

    windows = data['temporal_windows']
    metadata = data['metadata']

    # Initialize feature dictionary
    video_features = {'video_id': video_id}

    # Base features (21 per window)
    BASE_FEATURES = [
        'average_face_size', 'overlay_unique_count', 'has_captions',
        'scene_count', 'shortest_scene', 'longest_scene', 'scene_duration_variance',
        'object_count', 'person_count', 'dominant_emotion_id',
        'speech_coverage', 'word_count', 'energy_level', 'energy_variance',
        'energy_max', 'pitch_scatter_ratio', 'gesture_count', 'gaze_variance',
        'eye_contact_rate', 'emotional_valence', 'emotion_consistency'
    ]

    # Hook features (1 window - always present)
    for feature in BASE_FEATURES:
        video_features[f'hook_{feature}'] = windows['hook'].get(feature)

    # Middle features (0-5 segments depending on bucket)
    middle_segments = windows.get('middle_segments')

    if middle_segments is None or len(middle_segments) == 0:
        # For buckets 0-3s, 3-9s: middle_segments is null (expected)
        if bucket not in ['0-3s', '3-9s']:
            raise ValueError(
                f"Video {video_id}: null or empty middle_segments "
                f"(bucket {bucket} requires middle segments)"
            )
    else:
        # **NEW: Aggregate middle segments for short-window buckets**
        if bucket in AGGREGATE_MIDDLE_BUCKETS:
            # Aggregate all middle segments into single "middle_aggregate"
            # Reason: Short windows (1-4s) produce unreliable scene/speech features
            # Aggregation creates longer window (4.5-9.3s) for reliable measurements

            import numpy as np
            import pandas as pd

            for feature in BASE_FEATURES:
                # Collect non-null feature values from all middle segments
                feature_values = [
                    seg.get(feature)
                    for seg in middle_segments
                    if seg.get(feature) is not None
                ]

                # Skip if all values are None
                if len(feature_values) == 0:
                    video_features[f'middle_aggregate_{feature}'] = None
                    continue

                # Apply aggregation strategy based on feature type
                if feature in SUM_FEATURES:
                    # Cumulative features: sum across segments
                    video_features[f'middle_aggregate_{feature}'] = sum(feature_values)
                elif feature in MIN_FEATURES:
                    # Extreme value features: pick minimum
                    video_features[f'middle_aggregate_{feature}'] = min(feature_values)
                elif feature in MAX_FEATURES:
                    # Extreme value features: pick maximum
                    video_features[f'middle_aggregate_{feature}'] = max(feature_values)
                elif feature in CATEGORICAL_FEATURES:
                    # Categorical features: use mode (most common value)
                    mode_series = pd.Series(feature_values).mode()
                    video_features[f'middle_aggregate_{feature}'] = mode_series[0] if len(mode_series) > 0 else None
                else:
                    # Default: average for continuous/ratio features
                    video_features[f'middle_aggregate_{feature}'] = np.mean(feature_values)

            logger.debug(
                f"Video {video_id}: Aggregated {len(middle_segments)} middle segments "
                f"into middle_aggregate (bucket {bucket} has short windows)"
            )

        else:
            # **ORIGINAL: Keep separate middle segments for longer buckets**
            # Buckets 18-33s, 33-60s, 60-90s, 90-120s have longer windows (3-22.8s)
            # All features reliable at this duration
            for i, segment in enumerate(middle_segments, start=1):
                for feature in BASE_FEATURES:
                    video_features[f'middle_{i}_{feature}'] = segment.get(feature)

    # Closing features (1 window - skip for bucket 0-3s)
    if bucket != '0-3s':
        for feature in BASE_FEATURES:
            video_features[f'closing_{feature}'] = windows['closing'].get(feature)

    # Metadata (2 fields - video-level, not per-window)
    video_features['create_time'] = metadata.get('create_time')

    # Gender detection (optional field - use .get() with None default)
    gender_data = metadata.get('gender_detection', {})
    video_features['gender'] = gender_data.get('gender')

    return video_features
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Missing metadata field | Raise ValueError with video_id | Required for ML - fail fast |
| Null middle_segments in long bucket | Raise ValueError (Stage 2 bug) | Source: QA Q5 - skip video, log error |
| Missing optional gender field | Use None (get() default) | Gender is optional - proceed without it |
| Feature value is null | Keep null in DataFrame | Pandas handles nulls - validation in next stage |

#### Step 2.3.3: Batch Processing with Error Handling

**Purpose**: Process all videos in bucket, handling errors gracefully without failing entire batch

**Logic**:
```python
def process_bucket(bucket_path: Path, bucket: str):
    """
    Process all videos in bucket directory.

    Source: QA Q5, Q11 (error handling, duplicate detection)

    Args:
        bucket_path: Path to bucket directory
        bucket: Bucket name (e.g., "18-33s")

    Returns:
        pd.DataFrame: Aggregated features

    Raises:
        ValueError: if zero valid videos processed
    """
    insights_dir = bucket_path / "analysis" / "insights"
    json_files = list(insights_dir.glob("*_temporal_windows_updated.json"))

    aggregated_data = []
    seen_video_ids = set()
    skipped_reasons = defaultdict(int)

    for i, video_file in enumerate(json_files, start=1):
        try:
            # Extract video_id from filename
            video_id = video_file.stem.replace('_temporal_windows_updated', '')

            # Check for duplicate video_ids (Source: QA Q11)
            if video_id in seen_video_ids:
                logger.warning(
                    f"Duplicate video_id {video_id} found in {video_file.name}. "
                    "Skipping."
                )
                skipped_reasons['duplicate_video_id'] += 1
                continue

            seen_video_ids.add(video_id)

            # Extract features
            features = extract_features(video_file, bucket)
            aggregated_data.append(features)

            # Log progress every 10 videos (Source: QA Q10)
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(json_files)} videos")

        except json.JSONDecodeError as e:
            logger.error(
                f"Video {video_file.name}: Malformed JSON - {e}. Skipping."
            )
            skipped_reasons['malformed_json'] += 1
            continue

        except ValueError as e:
            # Validation errors (missing fields, null middle_segments)
            logger.error(f"Video {video_file.name}: {e}. Skipping.")
            skipped_reasons['validation_error'] += 1
            continue

        except Exception as e:
            logger.error(
                f"Video {video_file.name}: Unexpected error - {e}. Skipping."
            )
            skipped_reasons['unexpected_error'] += 1
            continue

    # Check if we have ANY valid videos (Source: QA Q5)
    if len(aggregated_data) == 0:
        raise ValueError(
            f"No valid videos processed in bucket {bucket_path}. Check logs."
        )

    # Log completion summary
    logger.info(
        f"Successfully processed {len(aggregated_data)}/{len(json_files)} videos "
        f"({len(json_files) - len(aggregated_data)} skipped)"
    )
    if skipped_reasons:
        logger.info(f"Skipped reasons: {dict(skipped_reasons)}")

    # Convert to DataFrame
    df = pd.DataFrame(aggregated_data)

    return df, skipped_reasons
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Malformed JSON file | Skip video, log error, continue | Source: QA Q5 - graceful degradation |
| All videos fail | Raise ValueError (don't create empty CSV) | No ML training possible with 0 videos |
| Duplicate video_ids | Keep first, skip duplicates with warning | Source: QA Q11 - prevent duplicate rows |
| 99 of 100 videos succeed | Create CSV with 99 rows (partial success) | Source: QA Q5 - allow partial success |

#### Step 2.3.4: Atomic Write Pattern

**Purpose**: Prevent partial CSV corruption if write operation fails mid-process

**Logic**:
```python
def save_aggregated_csv(df: pd.DataFrame, output_path: Path):
    """
    Save DataFrame to CSV with atomic write pattern.

    Source: QA Q7 (output integrity via atomic writes)

    Args:
        df: Aggregated features DataFrame
        output_path: Final CSV path

    Raises:
        IOError: if write fails
    """
    temp_path = output_path.with_suffix('.tmp')

    try:
        # Write to temporary file first
        df.to_csv(temp_path, index=False)

        # Atomic rename (only if write succeeded)
        shutil.move(temp_path, output_path)

        logger.info(
            f"Created {output_path.name} - "
            f"{len(df)} rows × {len(df.columns)} columns"
        )

    finally:
        # Clean up temp file if rename failed
        if temp_path.exists():
            temp_path.unlink()
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Disk full during write | temp file exists, final CSV doesn't | Downstream stages won't see corrupted data |
| Process killed mid-write | temp file exists, final CSV doesn't | No partial corruption |
| Rename fails | temp file cleaned up in finally block | No leftover temp files |

---

## 3. Dependencies & Integration

### 3.1 Input Dependencies

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **System setup** | FoundationCHILD.md (Section 2: Client Architecture) | Directory structure + bucket paths | bucket_{duration}/analysis/insights/, ml_analysis/ writable | Fail-fast if directories don't exist or not writable |
| **Stage 2.5 Completion** | File Organization stage | temporal_windows_updated.json files organized into bucket directories | Files must exist in bucket_{duration}/analysis/insights/ | Fail-fast with error: "No JSON files found. Did Stage 2.5 complete?" |
| temporal_windows_updated.json | Stage 2 (RumiAI Pipeline) via Stage 2.5 | JSON with temporal_windows.hook, .middle_segments[], .closing | 21 base features per window, metadata.duration, .create_time | Skip invalid files, log error, continue processing |
| Bucket parameter | CLI invocation | --bucket-path flag | Valid bucket directory path | Fail-fast if path invalid or doesn't exist |

### 3.2 Output Contracts

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| aggregated_features.csv | CSV | (N rows, 45-150 columns) - N = valid videos, columns = bucket-specific feature count | Stage 4 (Feature Transformation) | Assert row count > 0, column count matches expected for bucket |
| aggregation_summary.json | JSON | Metadata: timestamp, counts, skip reasons, column names | Debugging only (NOT consumed by downstream stages) | None - optional metadata file |

**Column Count by Bucket** (UPDATED - Middle Aggregation + 0-3s Hook Only):
- 0-3s: 24 features (21 × 1 window + 3 metadata: video_id, create_time, gender)
- 3-9s: 45 features (21 × 2 windows + 3 metadata)
- 9-13s, 13-18s: 66 features (21 × 3 windows + 3 metadata) [CHANGED - middle segments aggregated]
  - Structure: hook_* (21) + middle_aggregate_* (21) + closing_* (21) + metadata (3)
  - Reason: Short middle windows (1-4s) aggregated for feature reliability
  - Aggregation strategy: SUM for counts, AVG for ratios, MIN/MAX for extremes, MODE for categorical
- 18-33s: 129 features (21 × 6 windows + 3 metadata)
- 33-60s, 60-90s, 90-120s: 150 features (21 × 7 windows + 3 metadata)

### 3.3 Cross-Stage Dependencies

**This feature depends on**:
- **Stage 1 (Video Selection)**: Defines which videos to process (but Stage 3 doesn't cross-reference selected_videos.json - trusts Stage 2.5 organized correct videos)
- **Stage 2 (Video Processing)**: Produces temporal_windows_updated.json files with complete temporal features
- **Stage 2.5 (File Organization)**: CRITICAL - Organizes files from flat /insights/ to bucket directories. Stage 3 cannot run without Stage 2.5 completion.

**This feature is required by**:
- **Stage 4 (Feature Transformation)**: Expects aggregated_features.csv in exact bucket-specific column format
- **Stage 5 (ML Training)**: Indirectly requires this (via Stage 4 transformations)

**Failure Impact**:
- If this stage fails: Stage 4 and Stage 5 cannot run (no aggregated features)
- Checkpoint: Stage 3 is fast (< 5 minutes) - no checkpointing needed, can re-run entire bucket if needed

### 3.4 External Dependencies

**Python Libraries**:
```python
import pandas as pd  # 2.0.0+ (DataFrame creation, CSV I/O, Series.mode for categorical aggregation)
import numpy as np  # 1.24.0+ (mean aggregation for continuous features)
import json  # standard library (JSON parsing)
import shutil  # standard library (atomic move)
from pathlib import Path  # standard library (path operations)
from collections import defaultdict  # standard library (skip reason tracking)
import logging  # standard library (logging)

# Note: scipy NOT required - using pandas.Series.mode() for categorical features instead of scipy.stats.mode()
```

**File System**:
- Read access: `{bucket_path}/analysis/insights/*.json`
- Write access: `{bucket_path}/ml_analysis/` (creates if missing)

**Environment Variables**:
- None (all configuration via CLI parameters)

**External Services**: None (pure computational stage - no API calls)

---

## 4. Configuration & Parameters

### 4.1 CLI Parameters

| Parameter | Type | Default | Valid Values | Impact | Example |
|-----------|------|---------|--------------|--------|---------|
| `--bucket-path` | str | Required | Valid directory path | Determines which bucket to process | `--bucket-path="data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s"` |

**Invocation Pattern**: One invocation per bucket (enables parallelization)
```bash
# Process single bucket
python3 scripts/stage3_aggregation.py \
  --bucket-path="data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s"

# Parallel processing (3 buckets simultaneously)
python3 scripts/stage3_aggregation.py --bucket-path="...bucket_18-33s" &
python3 scripts/stage3_aggregation.py --bucket-path="...bucket_33-60s" &
python3 scripts/stage3_aggregation.py --bucket-path="...bucket_60-90s" &
```

### 4.2 Internal Configuration

```python
# Base features (21 per window) - Source: QA Q1 (exact list from FeatureTransformation.md)
BASE_FEATURES = [
    'average_face_size',        # Float, [0-1]
    'overlay_unique_count',     # Integer, count
    'has_captions',             # Boolean
    'scene_count',              # Integer, count
    'shortest_scene',           # Float, seconds
    'longest_scene',            # Float, seconds
    'scene_duration_variance',  # Float
    'object_count',             # Integer, count
    'person_count',             # Integer, count
    'dominant_emotion_id',      # Categorical, 1-7
    'speech_coverage',          # Float, [0-1]
    'word_count',               # Integer, count
    'energy_level',             # Float, [0-1]
    'energy_variance',          # Float
    'energy_max',               # Float, [0-1]
    'pitch_scatter_ratio',      # Float, [0-1]
    'gesture_count',            # Integer, count
    'gaze_variance',            # Float
    'eye_contact_rate',         # Float, [0-1]
    'emotional_valence',        # Float, [-1, 1]
    'emotion_consistency'       # Float, [0-1]
]

# Metadata fields (2 video-level fields)
METADATA_FIELDS = ['create_time', 'gender']

# Bucket configurations (window counts) - Source: FoundationCHILD.md Section 6 (Bucket Definitions)
BUCKET_MIDDLE_SEGMENTS = {
    '0-3s': 0,
    '3-9s': 0,
    '9-13s': 3,
    '13-18s': 3,
    '18-33s': 4,
    '33-60s': 5,
    '60-90s': 5,
    '90-120s': 5
}

# **NEW: Buckets that aggregate middle segments (short windows)**
# These buckets have middle windows of 1-4s, which produce unreliable measurements
# for scene_count, scene_duration_variance, speech_coverage, word_count, etc.
# Aggregation creates 4.5-9.3s windows where all 21 features are reliable.
AGGREGATE_MIDDLE_BUCKETS = ['9-13s', '13-18s']

# **NEW: Feature aggregation strategies**
# SUM: Cumulative/count features (discrete events)
SUM_FEATURES = [
    'scene_count', 'word_count', 'object_count',
    'person_count', 'overlay_unique_count', 'gesture_count'
]

# MIN: Pick minimum value (shortest scene)
MIN_FEATURES = ['shortest_scene']

# MAX: Pick maximum value (longest scene)
MAX_FEATURES = ['longest_scene']

# MODE: Categorical features (most common value)
CATEGORICAL_FEATURES = ['dominant_emotion_id', 'has_captions']

# AVERAGE (default): All other features use mean
# - speech_coverage, eye_contact_rate, energy_level, energy_variance, energy_max
# - pitch_scatter_ratio, gaze_variance, emotional_valence, emotion_consistency
# - average_face_size, scene_duration_variance

# Expected feature counts (for validation) - UPDATED with middle aggregation and 0-3s hook only
EXPECTED_FEATURE_COUNTS = {
    '0-3s': 24,   # 21 × 1 window (hook only) + 3 metadata (video_id, create_time, gender)
    '3-9s': 45,   # 21 × 2 windows + 3 metadata
    '9-13s': 66,  # 21 × 3 windows (hook + middle_aggregate + closing) + 3 metadata
    '13-18s': 66, # 21 × 3 windows (hook + middle_aggregate + closing) + 3 metadata
    '18-33s': 129, # 21 × 6 windows + 3 metadata
    '33-60s': 150, # 21 × 7 windows + 3 metadata
    '60-90s': 150,
    '90-120s': 150
}

# File paths (relative to bucket directory)
INSIGHTS_DIR = "analysis/insights"
OUTPUT_DIR = "ml_analysis"
OUTPUT_CSV = "aggregated_features.csv"
SUMMARY_JSON = "aggregation_summary.json"

# Logging configuration
LOG_PROGRESS_INTERVAL = 10  # Log every N videos
```

---

## 5. Data Schemas

### 5.1 Input Schema

**File**: `{bucket_path}/analysis/insights/{video_id}_temporal_windows_updated.json`

**Structure**: JSON with temporal_windows object + metadata object

**Key Fields** (subset shown - see example in Appendix B for complete structure):

| Field Path | Type | Range | Nulls? | Description | Example |
|------------|------|-------|--------|-------------|---------|
| `video_id` | str | - | No | Video identifier | "238506412723073" |
| `temporal_windows.hook.scene_count` | int | 0-20 | No | Scene cuts in hook window (0-3s) | 1 |
| `temporal_windows.hook.eye_contact_rate` | float | 0.0-1.0 | No | Eye contact proportion in hook | 0.8673 |
| `temporal_windows.hook.word_count` | int | 0-200 | No | Words spoken in hook | 7 |
| `temporal_windows.middle_segments[0].scene_count` | int | 0-20 | No | Scene cuts in middle segment 1 | 6 |
| `temporal_windows.middle_segments[0].word_count` | int | 0-200 | No | Words in middle segment 1 | 19 |
| `temporal_windows.closing.energy_level` | float | 0.0-1.0 | No | Audio energy in closing window | 0.0163 |
| `metadata.create_time` | str | ISO 8601 | No | Video publish timestamp | "2025-10-02T18:42:05.970516" |
| `metadata.gender_detection.gender` | str | male/female | Yes | Detected gender (optional) | "male" |

**Total Fields per Window**: 21 base features (see Section 4.2 for complete list)

**Windows by Bucket**:
- 0-3s: 1 window (hook only, no middle or closing)
- 3-9s: 2 windows (hook + closing, no middle_segments)
- 9-13s, 13-18s: 5 windows (hook + 3 middle + closing)
- 18-33s: 6 windows (hook + 4 middle + closing)
- 33-60s, 60-90s, 90-120s: 7 windows (hook + 5 middle + closing)

### 5.2 Output Schema

**File 1**: `{bucket_path}/ml_analysis/aggregated_features.csv`

**Format**: CSV with header row, one row per video, flat column naming with underscores

**Column Naming Convention** (Source: QA Q9, updated with aggregation and 0-3s):
```
video_id, create_time, gender,
hook_scene_count, hook_word_count, hook_energy_level, ...,
middle_1_scene_count, middle_1_word_count, middle_1_energy_level, ...  # For buckets 18-33s+
middle_aggregate_scene_count, middle_aggregate_word_count, ...         # For buckets 9-13s, 13-18s
closing_scene_count, closing_word_count, closing_energy_level, ...     # For buckets 3-9s+
```

**Schema (Bucket 18-33s example - 6 windows)**:

| Column | Type | Range | Nulls? | Description | Example |
|--------|------|-------|--------|-------------|---------|
| `video_id` | str | - | No | Primary key | "238506412723073" |
| `create_time` | str | ISO 8601 | No | Publish timestamp | "2025-10-02T18:42:05.970516" |
| `gender` | str | male/female/null | Yes | Detected gender | "male" |
| `hook_scene_count` | int | 0-20 | No | Scene cuts in hook | 1 |
| `hook_eye_contact_rate` | float | 0.0-1.0 | No | Eye contact in hook | 0.8673 |
| `hook_word_count` | int | 0-200 | No | Words in hook | 7 |
| `middle_1_scene_count` | int | 0-20 | No | Scene cuts in middle segment 1 | 6 |
| `middle_1_word_count` | int | 0-200 | No | Words in middle segment 1 | 19 |
| `middle_2_scene_count` | int | 0-20 | No | Scene cuts in middle segment 2 | 4 |
| `middle_2_word_count` | int | 0-200 | No | Words in middle segment 2 | 23 |
| `middle_3_scene_count` | int | 0-20 | No | Scene cuts in middle segment 3 | 3 |
| `middle_3_word_count` | int | 0-200 | No | Words in middle segment 3 | 19 |
| `middle_4_scene_count` | int | 0-20 | No | Scene cuts in middle segment 4 | 2 |
| `middle_4_word_count` | int | 0-200 | No | Words in middle segment 4 | 16 |
| `closing_scene_count` | int | 0-20 | No | Scene cuts in closing | 2 |
| `closing_energy_level` | float | 0.0-1.0 | No | Audio energy in closing | 0.0163 |
| ... | ... | ... | ... | (All 21 base features × 6 windows) | ... |

**Total Columns by Bucket** (UPDATED - Middle Aggregation + 0-3s Hook Only):
- 0-3s: 24 columns (1 window × 21 features + 3 metadata)
- 3-9s: 45 columns (2 windows × 21 features + 3 metadata)
- 9-13s, 13-18s: 66 columns (3 windows × 21 features + 3 metadata)
  - Column naming: hook_*, middle_aggregate_*, closing_*, metadata
  - Note: middle_1_*, middle_2_*, middle_3_* replaced by single middle_aggregate_*
- 18-33s: 129 columns (6 windows × 21 features + 3 metadata)
- 33-60s, 60-90s, 90-120s: 150 columns (7 windows × 21 features + 3 metadata)

**Note on Cross-Window Features**:
This stage outputs window-specific features only (21 per window). Cross-window features (e.g., `hook_to_middle_energy_delta`, `middle_to_closing_contrast`, `eye_contact_consistency`, `energy_trend_slope`, `window_consistency_score`) are computed in **Stage 4 (Feature Transformation)**, not Stage 3. These 5 additional features are added to the video-level RF training dataset by comparing values across temporal windows. See FeatureTransformationCHILD.md Section 2.3.2 for cross-window feature engineering logic.

**Row Count**: N = number of successfully processed videos (may be less than total files if some skipped)

**File 2**: `{bucket_path}/ml_analysis/aggregation_summary.json`

**Purpose**: Metadata about aggregation process (debugging only, NOT consumed by Stage 4)

**Schema**:
```json
{
  "bucket_path": "data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s",
  "timestamp": "2025-01-09T14:32:15Z",
  "duration_seconds": 12.3,
  "input_files_found": 45,
  "videos_processed": 43,
  "videos_skipped": 2,
  "skipped_reasons": {
    "malformed_json": 1,
    "null_middle_segments": 0,
    "duplicate_video_id": 1,
    "validation_error": 0,
    "unexpected_error": 0
  },
  "output_csv": {
    "path": "ml_analysis/aggregated_features.csv",
    "rows": 43,
    "columns": 129,
    "column_names": ["video_id", "duration", "create_time", "gender", "gender_confidence", "hook_scene_count", ...]
  },
  "stage_version": "3.0.0"
}
```

---

## 6. Error Handling & Validation

### 6.1 Input Validation

```python
def validate_input(data: dict, video_id: str, bucket: str):
    """
    Validate temporal_windows_updated.json before feature extraction.

    Source: QA Q5 (graceful error handling)

    Args:
        data: Parsed JSON data
        video_id: Video identifier
        bucket: Bucket name (e.g., "18-33s")

    Raises:
        ValueError: if validation fails with specific error message
    """
    # 1. Check required top-level fields exist
    if 'metadata' not in data:
        raise ValueError(f"Video {video_id}: Missing 'metadata' field")

    if 'temporal_windows' not in data:
        raise ValueError(f"Video {video_id}: Missing 'temporal_windows' field")

    # 2. Validate temporal_windows structure
    windows = data['temporal_windows']

    if 'hook' not in windows:
        raise ValueError(f"Video {video_id}: Missing 'hook' window")

    # Closing window validation (not required for bucket 0-3s)
    if bucket != '0-3s':
        if 'closing' not in windows:
            raise ValueError(f"Video {video_id}: Missing 'closing' window")

    # 3. Validate middle_segments (bucket-specific)
    middle_segments = windows.get('middle_segments')
    expected_middle_count = BUCKET_MIDDLE_SEGMENTS.get(bucket, 0)

    if expected_middle_count > 0:
        # Bucket requires middle segments
        if middle_segments is None or len(middle_segments) == 0:
            raise ValueError(
                f"Video {video_id}: null or empty middle_segments "
                f"(bucket {bucket} requires {expected_middle_count} segments)"
            )

        # **NEW: Flexible validation for aggregation buckets (Option B)**
        # Warn if segment count doesn't match expected, but proceed with aggregation
        if len(middle_segments) != expected_middle_count:
            logger.warning(
                f"Video {video_id}: Expected {expected_middle_count} middle segments, "
                f"found {len(middle_segments)}. "
                f"{'Aggregation will proceed with available segments.' if bucket in AGGREGATE_MIDDLE_BUCKETS else 'Proceeding anyway.'}"
            )
    else:
        # Bucket 0-3s, 3-9s - middle_segments should be null
        if middle_segments is not None and len(middle_segments) > 0:
            logger.warning(
                f"Video {video_id}: Unexpected middle segments in bucket {bucket}"
            )

    # 4. Validate metadata required fields
    metadata = data['metadata']

    if 'create_time' not in metadata:
        raise ValueError(f"Video {video_id}: Missing metadata.create_time")

    # Note: gender_detection is optional - use .get() with None default in extraction
    # Note: duration removed (redundant with bucket assignment)
```

### 6.2 Error Cases

| Error | Detection | Handling | User Message | Exit Code |
|-------|-----------|----------|--------------|-----------|
| Bucket directory missing | `bucket_path.exists()` | Fail-fast | `"Bucket path does not exist: {bucket_path}"` | 1 |
| Insights directory empty | `glob("*.json")` returns empty list | Fail-fast | `"No JSON files found in {insights_dir}. Did Stage 2.5 complete?"` | 2 |
| ml_analysis/ not writable | Test file write raises PermissionError | Fail-fast | `"Cannot write to {ml_analysis_dir}. Check permissions."` | 3 |
| Malformed JSON file | `json.load()` raises JSONDecodeError | Skip video, log error, continue | `"Video {filename}: Malformed JSON - {error}. Skipping."` | 0 (warning) |
| Missing required field | Field validation raises ValueError | Skip video, log error, continue | `"Video {video_id}: Missing 'metadata' field. Skipping."` | 0 (warning) |
| Null middle_segments in long bucket | Middle segment validation | Skip video, log error, continue | `"Video {video_id}: null or empty middle_segments (bucket {bucket} requires {N} segments). Skipping."` | 0 (warning) |
| Duplicate video_id | `video_id in seen_video_ids` | Skip duplicate, log warning, continue | `"Duplicate video_id {video_id} found in {filename}. Skipping."` | 0 (warning) |
| All videos fail | `len(aggregated_data) == 0` | Fail-fast | `"No valid videos processed in bucket {bucket_path}. Check logs."` | 4 |
| Disk full during write | `df.to_csv()` raises IOError | Fail-fast (temp file exists, final CSV doesn't) | `"Failed to write CSV: {error}"` | 5 |

### 6.3 Output Validation

```python
def validate_output(df: pd.DataFrame, bucket: str):
    """
    Validate aggregated DataFrame before saving.

    Source: QA Q7 (output validation before atomic write)

    Args:
        df: Aggregated features DataFrame
        bucket: Bucket name (e.g., "18-33s")

    Raises:
        AssertionError: if output schema invalid
    """
    # 1. Check row count > 0
    assert len(df) > 0, "DataFrame has 0 rows (no valid videos processed)"

    # 2. Check column count matches expected for bucket
    expected_cols = EXPECTED_FEATURE_COUNTS[bucket]
    actual_cols = len(df.columns)

    assert actual_cols == expected_cols, \
        f"Column count mismatch: expected {expected_cols}, got {actual_cols}"

    # 3. Check required columns exist
    required_cols = ['video_id', 'create_time']
    missing_cols = [c for c in required_cols if c not in df.columns]

    assert len(missing_cols) == 0, f"Missing required columns: {missing_cols}"

    # 4. Check for completely null columns (indicates extraction error)
    null_cols = df.columns[df.isnull().all()].tolist()
    if len(null_cols) > 0:
        logger.warning(
            f"Found {len(null_cols)} completely null columns: {null_cols[:5]}... "
            "(may indicate feature extraction issues)"
        )

    # 5. Validate video_id uniqueness
    duplicate_ids = df['video_id'].duplicated().sum()
    assert duplicate_ids == 0, f"Found {duplicate_ids} duplicate video_ids in output"
```

---

## 7. Performance & Scalability

### 7.1 Performance Targets

- **Throughput**: Process 100 videos (bucket 18-33s) in < 2 minutes, 300 videos in < 5 minutes
- **Memory**: Peak usage < 2GB for 300 videos (~20 MB typical for 100 videos)
- **Disk I/O**: < 5s for JSON reads (100 videos), < 2s for CSV write
- **CPU**: < 30% average utilization (single-threaded, I/O bound)

### 7.2 Measured Performance

Performance estimates based on system architecture and pandas benchmarks:

| Metric | Bucket 18-33s (N=100) | Bucket 18-33s (N=300) | Notes |
|--------|------------------------|------------------------|-------|
| JSON loading | 15s | 45s | 100 files × 50-100 KB each |
| Feature extraction | 30s | 90s | Python dict operations |
| DataFrame creation | 2s | 5s | pandas overhead |
| CSV write | 1s | 3s | pandas to_csv |
| **Total time** | ~50s | ~2.5 min | Linear scaling, I/O bound |
| Memory peak | ~20 MB | ~50 MB | In-memory DataFrame |

### 7.3 Bottlenecks & Mitigations

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| JSON file I/O | 45s for N=300 | Sequential file reads (no parallelization) | Acceptable for current scale - parallelize across buckets not within bucket | Low |
| Feature extraction loop | 90s for N=300 | Python dict operations (not vectorized) | Could vectorize with pandas apply(), but adds complexity for minimal gain | Low |
| Memory growth | 50 MB peak for N=300 | Full DataFrame in memory | Trivial memory usage - no optimization needed | Low |
| Error logging overhead | Minimal (< 1s) | File I/O for each log message | Use buffered logging handler if becomes issue | Low |

### 7.4 Scalability Limits

- **Max videos per bucket**: 1000 (memory: ~200 MB, time: ~8 minutes) - well within acceptable limits
- **Max features per bucket**: 150 (bucket 90-120s) - no performance degradation
- **Min videos per bucket**: 1 (edge case handled - creates valid single-row CSV)
- **Parallelization**: Run multiple buckets simultaneously (3-5 parallel invocations typical)

**Scaling Strategy**: Process buckets in parallel (not videos within bucket) using shell scripting or orchestration layer.

---

## 8. Testing Strategy

### 8.1 Unit Tests

- [ ] **Test input validation**
  - Empty insights directory (raises ValueError with Stage 2.5 message)
  - Missing required fields (raises ValueError with field name)
  - Null middle_segments in long bucket (raises ValueError)
  - Missing closing window for bucket 0-3s (passes - closing not required)
  - Missing closing window for bucket 3-9s+ (raises ValueError)
  - Valid input with all fields (passes without error)
  - Optional gender field missing (uses None, no error)

- [ ] **Test feature extraction**
  - Correct column naming: `hook_scene_count`, `middle_1_word_count` (not dotted notation)
  - Correct feature count per bucket (24 for 0-3s, 45 for 3-9s, 66 for 9-18s, 129 for 18-33s, 150 for 33-60s+)
  - Bucket 0-3s: only hook columns, no closing columns
  - Bucket 3-9s+: hook + closing columns (+ middle if applicable)
  - Metadata fields extracted correctly (video_id, create_time, gender)
  - Null values preserved (not replaced with defaults)

- [ ] **Test error handling**
  - Malformed JSON file (skip video, log error, continue)
  - Duplicate video_ids (keep first, skip second with warning)
  - All videos fail (raise ValueError, no CSV created)
  - 99 of 100 succeed (create CSV with 99 rows)

- [ ] **Test output validation**
  - Column count matches expected for bucket
  - No duplicate video_ids in output
  - All required columns present

### 8.2 Integration Tests

- [ ] **End-to-end: Stage 2.5 → Stage 3 → Stage 4**
  - Use real temporal_windows_updated.json files (from test data)
  - Run Stage 3 aggregation
  - Verify aggregated_features.csv exists with correct shape
  - Verify Stage 4 can load CSV without error

- [ ] **Error propagation**
  - Stage 2.5 missing output → Stage 3 fails with clear message
  - Stage 3 aggregation error → Stage 4 does not run

### 8.3 Test Data

**File**: `tests/fixtures/temporal_windows_bucket_33-60s_sample.json`

```json
{
  "video_id": "238506412723073",
  "duration": 50.0,
  "temporal_windows": {
    "hook": {
      "scene_count": 1,
      "eye_contact_rate": 0.8673,
      "word_count": 7,
      "energy_level": 0.0106
    },
    "middle_segments": [
      {"scene_count": 6, "word_count": 19, "energy_level": 0.0088},
      {"scene_count": 4, "word_count": 23, "energy_level": 0.0113},
      {"scene_count": 3, "word_count": 19, "energy_level": 0.0142},
      {"scene_count": 2, "word_count": 16, "energy_level": 0.0147},
      {"scene_count": 3, "word_count": 22, "energy_level": 0.0127}
    ],
    "closing": {
      "scene_count": 2,
      "energy_level": 0.0163
    }
  },
  "metadata": {
    "duration": 50.0,
    "create_time": "2025-10-02T18:42:05.970516",
    "gender_detection": {"gender": "male", "confidence": 0.9863}
  }
}
```

**Expected Output CSV** (bucket 33-60s, 7 windows = 149 columns):
```csv
video_id,create_time,gender,hook_scene_count,hook_eye_contact_rate,hook_word_count,hook_energy_level,...,middle_1_scene_count,middle_1_word_count,...,closing_scene_count,closing_energy_level
238506412723073,2025-10-02T18:42:05.970516,male,1,0.8673,7,0.0106,...,6,19,...,2,0.0163
```

### 8.4 Test Execution

```bash
# Run unit tests
pytest tests/test_feature_aggregation.py -v

# Run integration tests
pytest tests/test_stage3_integration.py -v

# Run with coverage
pytest tests/test_feature_aggregation.py --cov=stage3_aggregation --cov-report=html

# Test with real data
python3 scripts/stage3_aggregation.py \
  --bucket-path="tests/fixtures/bucket_18-33s_sample"
```

---

## 9. Future Enhancements

### 9.1 Planned Improvements

- **Phase 2: Bucket configuration centralization**
  - Current: Hardcoded BUCKET_MIDDLE_SEGMENTS dict in Stage 3 code
  - Future: Shared bucket_config.json read by Stage 2 (temporal compute) and Stage 3 (aggregation)
  - Impact: Single source of truth for bucket definitions, easier to add new buckets

- **Phase 2: Parallel JSON loading**
  - Current: Sequential file reads (acceptable for 100-300 videos)
  - Future: Use multiprocessing to load 4-8 files in parallel
  - Impact: 2-4x speedup for JSON I/O (45s → 12s for N=300)

- **Phase 3: Incremental aggregation**
  - Current: Full bucket reprocessing on re-run
  - Future: Track processed video_ids, only aggregate new videos
  - Impact: Faster re-runs after adding new videos to bucket

### 9.2 Known Limitations

- **No duration validation**: Trusts Stage 2.5 organized videos into correct buckets (doesn't re-check duration ranges)
- **No cross-reference with selected_videos.json**: Doesn't validate that aggregated videos match Stage 1 selection
- **Single-threaded**: No parallelization within bucket (but can run multiple buckets in parallel)
- **No schema evolution handling**: If RumiAI adds/removes features, Stage 3 breaks (requires code update)

---

## 10. References & Related Docs

### 10.1 Parent Document

- **MLPlanningv2.md Section 3.3 "Feature Aggregation"**
  - High-level component overview (lines 799-920)
  - Stage position in pipeline
  - Bucket-specific feature counts

### 10.2 Foundation Dependencies

- **FoundationCHILD.md**
  - Section 2 "Client Architecture & Storage": Directory paths used in this stage (bucket structure, ml_analysis/)
  - Section 3 "Data Schemas": temporal_windows_updated.json schema with window definitions
  - Section 6 "Bucket Definitions": Bucket-specific window counts and middle segment configurations

**Key Sections Referenced in This Stage**:
- Section 2 "Client Architecture": Provides bucket directory structure (bucket_{duration}/analysis/insights/, ml_analysis/)
- Section 6 "Bucket Definitions": Defines middle segment counts per bucket (used in BUCKET_MIDDLE_SEGMENTS configuration)

### 10.3 Related Child Docs

- **FileOrganizationCHILD.md** (Stage 2.5)
  - Produces organized temporal_windows_updated.json files (input to this stage)
  - Defines bucket directory structure

- **FeatureTransformationCHILD.md** (Stage 4)
  - Consumes aggregated_features.csv (output from this stage)
  - Defines expected schema and column names

### 10.4 External References

- **Pandas CSV I/O**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
- **Atomic file writes**: POSIX shutil.move() guarantees (Linux/Mac)
- **JSON parsing**: Python json module documentation

---

## Appendix A: Decision Log

**Purpose**: Record major design decisions with rationale and trade-offs.

**Decision 1**: Stage 2.5 file organization (NEW component required)
- **Context**: Stage 2 produces flat temporal_windows_updated.json files in /insights/. Stage 3 needs files organized by bucket for efficient processing.
- **Alternatives Considered**:
  - Option A: Stage 3 reads all files, filters by duration - Rejected (inefficient, Stage 3 doesn't need bucket knowledge)
  - Option B: Stage 2.5 organizes files into bucket directories - **CHOSEN**
- **Rationale**: Separation of concerns (Stage 2 processes videos, Stage 2.5 organizes by bucket, Stage 3 aggregates). Performance benefit (no duration filtering). Architectural consistency (mirrors Stage 1 bucket organization).
- **Trade-offs**: Added Stage 2.5 complexity, but cleaner Stage 3 implementation and better parallelization support.
- **Date**: 2025-01-09 (Source: QA Q1, Q2)

**Decision 2**: One invocation per bucket (not one for all buckets)
- **Context**: Need to determine CLI invocation pattern for Stage 3.
- **Alternatives Considered**:
  - Option A: One invocation per bucket - **CHOSEN**
  - Option B: One invocation for all buckets - Rejected (no parallelization, harder failure recovery)
- **Rationale**: Enables parallel processing of 3 buckets simultaneously. Better failure resilience (one bucket fails, others continue). Consistent with Stage 1 bucket-centric architecture.
- **Trade-offs**: Requires orchestration layer to invoke Stage 3 multiple times, but enables 3x parallelization speedup.
- **Date**: 2025-01-09 (Source: QA Q2)

**Decision 3**: Graceful error handling (skip bad videos, not fail-fast)
- **Context**: How to handle malformed JSON, missing fields, null middle_segments during aggregation.
- **Alternatives Considered**:
  - Option A: Fail-fast on first error - Rejected (wastes work on 99 good videos)
  - Option B: Skip bad videos, log errors, continue - **CHOSEN**
  - Option C: Fill missing data with defaults - Rejected (pollutes ML training data)
- **Rationale**: Graceful degradation (99 of 100 videos proceed to ML). Data quality preservation (only real data enters pipeline). Actionable logging (clear error messages for debugging).
- **Trade-offs**: Some videos excluded from analysis, but better than pipeline failure or fake data.
- **Date**: 2025-01-09 (Source: QA Q5, Critique final decision)

**Decision 4**: Atomic write pattern (temp file + rename)
- **Context**: Prevent partial CSV corruption if write operation fails mid-process.
- **Alternatives Considered**:
  - Option A: Direct write to final path - Rejected (risk of partial corruption)
  - Option B: Atomic write (temp + rename) - **CHOSEN**
  - Option C: Write + validate - Rejected (doesn't prevent partial writes)
- **Rationale**: Minimal code overhead (2 extra lines). Prevents downstream stages from consuming corrupted data. Standard pattern for critical file writes.
- **Trade-offs**: Slightly more complex code, but critical data integrity benefit.
- **Date**: 2025-01-09 (Source: QA Q7)

**Decision 5**: Flat column naming with underscores (not dotted notation)
- **Context**: Need to specify exact CSV column naming convention.
- **Alternatives Considered**:
  - Option A: Flat with underscores (`hook_scene_count`) - **CHOSEN**
  - Option B: Dotted notation (`hook.scene_count`) - Rejected (pandas access issues)
  - Option C: Prefixed categories (`feat_hook_scene_count`) - Rejected (unnecessary verbosity)
- **Rationale**: Pandas-friendly (no issues with df.hook_scene_count access). Matches Python naming conventions (snake_case). Easy to read in Excel. Consistent with common ML practice.
- **Trade-offs**: None - clear winner for readability and compatibility.
- **Date**: 2025-01-09 (Source: QA Q9)

**Decision 6**: Generate aggregation_summary.json alongside CSV
- **Context**: Whether to output metadata file documenting aggregation process.
- **Alternatives Considered**:
  - Option A: Generate summary JSON - **CHOSEN**
  - Option B: Log to console only - Rejected (logs ephemeral, not queryable)
- **Rationale**: Very low implementation cost (~20 lines). Helpful for debugging (skip reasons, timestamps). Enables future monitoring/dashboards. Self-documenting output.
- **Trade-offs**: One extra file per bucket, but negligible size (~1 KB) and high debugging value.
- **Date**: 2025-01-09 (Source: QA Q12)

**Decision 7**: Aggregate middle segments for buckets 9-13s and 13-18s with 4-strategy approach
- **Context**: Middle windows in buckets 9-13s (1-2.3s each) and 13-18s (2.3-4s each) are too short to produce reliable measurements for 8 out of 21 features (38%): scene_count, shortest_scene, longest_scene, scene_duration_variance, speech_coverage, word_count, gesture_count, gaze_variance.
- **Alternatives Considered**:
  - Option A: Keep separate middle segments, accept unreliable features - Rejected (pollutes ML training data)
  - Option B: Aggregate middle segments into single "middle_aggregate" window - **CHOSEN**
  - Option C: Remove unreliable features for these buckets only - Rejected (creates inconsistent feature sets across buckets)
  - Option D: Do nothing, rely on ML to handle noise - Rejected (high-dimensional noise degrades cluster quality)
- **Rationale**:
  - Feature reliability: 13/21 reliable in 1-2.3s windows → 21/21 reliable in 4.5-9.3s aggregated window
  - Four aggregation strategies preserve data semantics:
    - **SUM** for cumulative features (scene_count, word_count): Total events across segments
    - **MIN/MAX** for extreme values (shortest_scene, longest_scene): True extremes preserved
    - **MODE** for categorical features (dominant_emotion_id, has_captions): Most common value
    - **AVERAGE** for continuous/ratio features (speech_coverage, energy_level): Representative value
  - Bucket-specific models already handle different feature counts (44, 65, 128, 149)
  - Temporal granularity loss acceptable (middle progressions unreliable anyway in short segments)
  - Simpler than feature filtering (maintains consistent 21-feature schema across all windows)
- **Metadata Reduction Decision**: Removed `duration` (redundant with bucket assignment) and `gender_confidence` (not needed for ML). Reduced metadata from 5 to 2 fields (create_time, gender).
- **Trade-offs**:
  - Lose middle segment progression for 9-18s videos (e.g., can't detect "word_count increases middle_1 → middle_3")
  - But: This progression data was unreliable due to short windows (noise, not signal)
  - Feature count reduced from 108 → 65 for these buckets (40% reduction)
  - But: Fewer high-quality features better than more low-quality features for ML
- **Impact**:
  - Stage 3: +60 lines of code (4-strategy aggregation logic + configuration)
  - Stage 4-7: Column name changes only (middle_aggregate_* instead of middle_1_*, middle_2_*, middle_3_*)
  - K-Means clustering: Better quality (21 reliable features instead of 13 reliable + 8 noisy)
  - Downstream stages: No logic changes (just different column names in DataFrame)
- **Date**: 2025-01-10 (Source: FeatureAggregationCHANGE.md analysis of temporal window reliability)

**Decision 8**: Bucket 0-3s has hook only (no closing window)
- **Context**: For videos 0-3s long, the full video is already captured by the hook window (0-3s). A closing window (last 3s) would completely overlap with the hook.
- **Alternatives Considered**:
  - Option A: Hook only (1 window) - **CHOSEN**
  - Option B: Hook + closing (2 windows with complete overlap) - Rejected (creates duplicate data)
- **Rationale**:
  - Temporal impossibility: Cannot extract separate 3s hook + 3s closing from a video shorter than 6s
  - SystemArchitecturev2.md Line 194 explicitly states "0-3s | None (null) | Hook only"
  - No redundant data: Hook already covers 100% of video content
  - Consistent with production temporal_compute.py logic
- **Impact**:
  - Stage 3: +5 lines of code (conditional skip closing for 0-3s)
  - Column count: 0-3s bucket reduced from 45 → 24 columns (21 features, not 42)
  - Stage 4-7: No changes needed (already handle variable window counts per bucket)
- **Date**: 2025-10-14 (Source: Stage 3-4 Compatibility Analysis Q4)

---

## Appendix B: Example Data

### B.1 Sample Input (1 video, bucket 33-60s with 7 windows)

**File**: `/home/jorge/rumiaifinal/insights/238506412723073_temporal_windows_updated.json`

```json
{
  "video_id": "238506412723073",
  "duration": 50.0,
  "temporal_windows": {
    "hook": {
      "start": 0,
      "end": 3.0,
      "duration": 3.0,
      "scene_count": 1,
      "eye_contact_rate": 0.8673,
      "word_count": 7,
      "speech_coverage": 1.0,
      "energy_level": 0.0106,
      "energy_max": 0.0374,
      "pitch_scatter_ratio": 0.5827,
      "dominant_emotion_id": 7,
      "emotional_valence": -0.3333,
      "average_face_size": 0.2073
    },
    "middle_segments": [
      {
        "start": 3.0,
        "end": 11.8,
        "duration": 8.8,
        "scene_count": 6,
        "word_count": 19,
        "energy_level": 0.0088,
        "segment_name": "segment_1"
      },
      {
        "start": 11.8,
        "end": 20.6,
        "duration": 8.8,
        "scene_count": 4,
        "word_count": 23,
        "energy_level": 0.0113,
        "segment_name": "segment_2"
      },
      {
        "start": 20.6,
        "end": 29.4,
        "duration": 8.8,
        "scene_count": 3,
        "word_count": 19,
        "energy_level": 0.0142,
        "segment_name": "segment_3"
      },
      {
        "start": 29.4,
        "end": 38.2,
        "duration": 8.8,
        "scene_count": 2,
        "word_count": 16,
        "energy_level": 0.0147,
        "segment_name": "segment_4"
      },
      {
        "start": 38.2,
        "end": 47.0,
        "duration": 8.8,
        "scene_count": 3,
        "word_count": 22,
        "energy_level": 0.0127,
        "segment_name": "segment_5"
      }
    ],
    "closing": {
      "start": 47.0,
      "end": 50.0,
      "duration": 3.0,
      "scene_count": 2,
      "energy_level": 0.0163,
      "eye_contact_rate": 0.8837
    }
  },
  "metadata": {
    "video_id": "238506412723073",
    "duration": 50.0,
    "create_time": "2025-10-02T18:42:05.970516",
    "gender_detection": {
      "gender": "male",
      "confidence": 0.9863
    }
  }
}
```

### B.2 Sample Output (bucket 33-60s, 7 windows = 149 columns)

**File**: `ml_analysis/aggregated_features.csv`

**Note**: Showing subset of columns for readability. Actual CSV has all 21 base features × 7 windows + 2 metadata = 149 total columns.

```csv
video_id,create_time,gender,hook_scene_count,hook_eye_contact_rate,hook_word_count,hook_speech_coverage,hook_energy_level,middle_1_scene_count,middle_1_word_count,middle_1_energy_level,middle_2_scene_count,middle_2_word_count,middle_2_energy_level,middle_3_scene_count,middle_3_word_count,middle_3_energy_level,middle_4_scene_count,middle_4_word_count,middle_4_energy_level,middle_5_scene_count,middle_5_word_count,middle_5_energy_level,closing_scene_count,closing_energy_level,closing_eye_contact_rate
238506412723073,2025-10-02T18:42:05.970516,male,1,0.8673,7,1.0,0.0106,6,19,0.0088,4,23,0.0113,3,19,0.0142,2,16,0.0147,3,22,0.0127,2,0.0163,0.8837
```

### B.3 Sample Summary JSON

**File**: `ml_analysis/aggregation_summary.json`

```json
{
  "bucket_path": "data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_33-60s",
  "timestamp": "2025-01-09T14:32:15Z",
  "duration_seconds": 12.3,
  "input_files_found": 45,
  "videos_processed": 43,
  "videos_skipped": 2,
  "skipped_reasons": {
    "malformed_json": 1,
    "null_middle_segments": 0,
    "duplicate_video_id": 1,
    "validation_error": 0,
    "unexpected_error": 0
  },
  "output_csv": {
    "path": "ml_analysis/aggregated_features.csv",
    "rows": 43,
    "columns": 149,
    "column_names": [
      "video_id", "create_time", "gender",
      "hook_scene_count", "hook_eye_contact_rate", "hook_word_count",
      "middle_1_scene_count", "middle_1_word_count", "middle_1_energy_level",
      "middle_2_scene_count", "middle_2_word_count", "middle_2_energy_level",
      "middle_3_scene_count", "middle_3_word_count", "middle_3_energy_level",
      "middle_4_scene_count", "middle_4_word_count", "middle_4_energy_level",
      "middle_5_scene_count", "middle_5_word_count", "middle_5_energy_level",
      "closing_scene_count", "closing_energy_level", "closing_eye_contact_rate"
    ]
  },
  "stage_version": "3.0.0"
}
```

---

## Appendix C: Pseudocode (Complete)

### C.1 Full Aggregation Pipeline

```python
def aggregate_features(bucket_path: str):
    """
    Complete Stage 3 feature aggregation pipeline.

    Sources: QA Q1-Q12, Critique decisions

    Args:
        bucket_path: str, path to bucket directory (e.g., "bucket_18-33s")

    Returns:
        tuple: (csv_path, summary_path)

    Raises:
        ValueError: if validation fails or zero valid videos processed
    """
    import pandas as pd
    import json
    import shutil
    from pathlib import Path
    from collections import defaultdict
    from datetime import datetime, timezone

    start_time = time.time()
    bucket_path = Path(bucket_path)

    # Extract bucket name from path
    bucket = bucket_path.name.replace('bucket_', '')

    # ===== 1. Validate Dependencies =====
    logger.info(f"Stage 3 Feature Aggregation starting")
    logger.info(f"Bucket path: {bucket_path}")

    total_files = validate_dependencies(bucket_path)  # See Section 2.3.1
    logger.info(f"Found {total_files} temporal_windows_updated.json files")
    logger.info(f"Validation complete - ml_analysis/ directory writable")

    # ===== 2. Load and Extract Features from All Videos =====
    insights_dir = bucket_path / "analysis" / "insights"
    json_files = list(insights_dir.glob("*_temporal_windows_updated.json"))

    aggregated_data = []
    seen_video_ids = set()
    skipped_reasons = defaultdict(int)

    for i, video_file in enumerate(json_files, start=1):
        try:
            # Load JSON
            with open(video_file) as f:
                data = json.load(f)

            # Extract video_id from filename
            video_id = video_file.stem.replace('_temporal_windows_updated', '')

            # Check for duplicate video_ids (Source: QA Q11)
            if video_id in seen_video_ids:
                logger.warning(
                    f"Duplicate video_id {video_id} found in {video_file.name}. "
                    "Skipping."
                )
                skipped_reasons['duplicate_video_id'] += 1
                continue

            seen_video_ids.add(video_id)

            # Validate input (Source: QA Q5)
            validate_input(data, video_id, bucket)  # See Section 6.1

            # Extract features (Source: QA Q1, Q9)
            features = extract_features(video_file, bucket)  # See Section 2.3.2
            aggregated_data.append(features)

            # Log progress every 10 videos (Source: QA Q10)
            if i % 10 == 0:
                logger.info(f"Processed {i}/{total_files} videos")

        except json.JSONDecodeError as e:
            logger.error(
                f"Video {video_file.name}: Malformed JSON - {e}. Skipping."
            )
            skipped_reasons['malformed_json'] += 1
            continue

        except ValueError as e:
            # Validation errors (missing fields, null middle_segments)
            logger.error(f"Video {video_file.name}: {e}. Skipping.")
            skipped_reasons['validation_error'] += 1
            continue

        except Exception as e:
            logger.error(
                f"Video {video_file.name}: Unexpected error - {e}. Skipping."
            )
            skipped_reasons['unexpected_error'] += 1
            continue

    # Check if we have ANY valid videos (Source: QA Q5)
    if len(aggregated_data) == 0:
        raise ValueError(
            f"No valid videos processed in bucket {bucket_path}. Check logs."
        )

    # Log completion summary
    logger.info(
        f"Successfully processed {len(aggregated_data)}/{total_files} videos "
        f"({total_files - len(aggregated_data)} skipped)"
    )
    if skipped_reasons:
        logger.info(f"Skipped reasons: {dict(skipped_reasons)}")

    # ===== 3. Convert to DataFrame =====
    df = pd.DataFrame(aggregated_data)

    # ===== 4. Validate Output Schema =====
    logger.info("Validating output schema")
    validate_output(df, bucket)  # See Section 6.3

    # ===== 5. Atomic Write to CSV =====
    ml_analysis_dir = bucket_path / "ml_analysis"
    output_path = ml_analysis_dir / "aggregated_features.csv"

    save_aggregated_csv(df, output_path)  # See Section 2.3.4

    # ===== 6. Generate Summary JSON =====
    end_time = time.time()

    summary = {
        "bucket_path": str(bucket_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(end_time - start_time, 2),
        "input_files_found": total_files,
        "videos_processed": len(aggregated_data),
        "videos_skipped": total_files - len(aggregated_data),
        "skipped_reasons": dict(skipped_reasons),
        "output_csv": {
            "path": "ml_analysis/aggregated_features.csv",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        },
        "stage_version": "3.0.0"
    }

    summary_path = ml_analysis_dir / "aggregation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Stage 3 complete - Duration: {summary['duration_seconds']}s")

    # ===== 7. Return Paths =====
    return output_path, summary_path


def extract_features(temporal_windows_json: Path, bucket: str):
    """See Section 2.3.2 for full implementation"""
    # Implementation shown in Section 2.3.2
    pass


def validate_dependencies(bucket_path: Path):
    """See Section 2.3.1 for full implementation"""
    # Implementation shown in Section 2.3.1
    pass


def validate_input(data: dict, video_id: str, bucket: str):
    """See Section 6.1 for full implementation"""
    # Implementation shown in Section 6.1
    pass


def validate_output(df: pd.DataFrame, bucket: str):
    """See Section 6.3 for full implementation"""
    # Implementation shown in Section 6.3
    pass


def save_aggregated_csv(df: pd.DataFrame, output_path: Path):
    """See Section 2.3.4 for full implementation"""
    # Implementation shown in Section 2.3.4
    pass


# ===== CLI Entry Point =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 3: Feature Aggregation"
    )
    parser.add_argument(
        "--bucket-path",
        required=True,
        help="Path to bucket directory (e.g., bucket_18-33s)"
    )

    args = parser.parse_args()

    csv_path, summary_path = aggregate_features(args.bucket_path)

    print(f"✓ Aggregated features saved to: {csv_path}")
    print(f"✓ Summary saved to: {summary_path}")
```

---

## Document Metadata

**Creation Date**: 2025-01-09
**Last Modified**: 2025-01-09
**Authors**: Senior Technical Architect (AI-generated based on Phase 1-2 outputs)
**Status**: Production-Ready (NO TODOs)
**Next Review Date**: 2025-02-09

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.1 | 2025-01-28 | RumiAI Team | Fixed broken references: Updated all Mother HLD references to point to FoundationCHILD.md (4 locations: Lines 20-23, 441, 555, 1047-1056). Enforces three-tier architecture: Mother → Foundation → Components. |
| 1.0 | 2025-01-09 | Technical Architect | Initial production-ready HLD generated from Phase 1 Critique + Phase 2 Q&A |
