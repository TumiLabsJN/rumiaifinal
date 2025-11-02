# Stage 3 & 3.4: Feature Aggregation - Implementation Guide

**Purpose**: Transform variable-length temporal window JSONs into fixed-size CSV files for ML training
**Target Audience**: LLM agents fixing bugs, adding features, or modifying feature aggregation stages
**Related**: [PRODUCTION_FLOW.md Stage 3 Contract](../../PRODUCTION_FLOW.md#stage-3-feature-aggregation)

**Source**: 100% systematic code reading of 4 modules (1,654 production lines)

---

## Quick Reference

### Stage 3: Feature Aggregation
- **Entry Point**: `scripts/stage3_aggregation.py::aggregate_features()` (line 732)
- **Orchestrator Call**: `rumiai_ml_batch.py:1085-1260`
- **Checkpoint**: `{bucket_path}/checkpoints/stage_3_checkpoint.json`
- **Duration**: ~10-30s per bucket (40 videos)
- **Output**: `aggregated_features.csv` (350+ features × N videos)
- **Key Feature**: Cross-window features + is_top_performer label (added in S7B2 fix)

### Stage 3.4: Review CSV Generation
- **Entry Point**: `ml_pipeline/stage3_aggregation/review_csv_generator.py::generate_review_csv_for_bucket()` (line 309)
- **Orchestrator Call**: `rumiai_ml_batch.py:1160-1190`
- **Duration**: ~5-10s per bucket
- **Output**: `video_review.csv` (same features + clickable TikTok URLs)
- **Status**: Optional (failures don't halt pipeline)

### Module Structure
```
scripts/
└── stage3_aggregation.py (931 lines)    # Stage 3 main implementation

ml_pipeline/stage3_aggregation/
└── review_csv_generator.py (347 lines)  # Stage 3.4 review CSV

config/
└── bucket_definitions.py (200 lines)    # Shared bucket configurations
```

---

## Table of Contents

1. [Part A: Stage 3 Feature Aggregation](#part-a-stage-3-feature-aggregation)
2. [Part B: Stage 3.4 Review CSV Generation](#part-b-stage-34-review-csv-generation)
3. [Shared Configuration Reference](#shared-configuration-reference)
4. [Data Flow & Architecture](#data-flow--architecture)
5. [Error Handling Matrix](#error-handling-matrix)
6. [Debugging Guide](#debugging-guide)

---

# Part A: Stage 3 Feature Aggregation

## Overview

**Purpose**: Extract 21 base features from each temporal window, compute cross-window features, add labels
**Input**: Variable-length temporal_windows_updated.json files (per video)
**Output**: Fixed-size CSV with bucket-specific feature count (25-156 columns)

## Input Contract

### Prerequisites
- **Stage 2.5** complete → Files organized in `buckets/bucket_{name}/analysis/insights/`
- **Stage 1** complete → `selected_videos.json` (for is_top_performer labels in contrastive mode)

### Input Files
```
{bucket_path}/
├── analysis/insights/
│   └── {video_id}_temporal_windows_updated.json  # 40 files (one per video)
└── selected_videos.json                           # Optional (Stage 1 output)
```

### Validation
**File**: `stage3_aggregation.py::validate_dependencies()` (line 128-177)

```python
# Pre-flight checks:
1. bucket_path exists
2. analysis/insights/ directory exists
3. At least 1 temporal_windows_updated.json file found
4. ml_analysis/ directory is writable (creates if missing)
```

**Raises**: `ValueError` with specific error message if validation fails

---

## Output Contract

### Files Created
```
{bucket_path}/
├── ml_analysis/
│   ├── aggregated_features.csv           # Main output (N rows × M cols)
│   └── aggregation_summary.json          # Processing metadata
└── checkpoints/
    └── stage_3_checkpoint.json           # For orchestrator skip logic
```

### Output Schema

**aggregated_features.csv** (columns vary by bucket):

| Bucket | Total Columns | Breakdown |
|--------|---------------|-----------|
| 0-3s | 25 | 21×1 + 3 metadata + 0 cross-window + 1 label |
| 3-9s | 49 | 21×2 + 3 + 3 + 1 |
| 9-13s | 72 | 21×3 + 3 + 5 + 1 |
| 13-18s | 72 | 21×3 + 3 + 5 + 1 |
| 18-33s | 135 | 21×6 + 3 + 5 + 1 |
| 33-60s | 156 | 21×7 + 3 + 5 + 1 |
| 60-90s | 156 | 21×7 + 3 + 5 + 1 |
| 90-120s | 156 | 21×7 + 3 + 5 + 1 |

**Column Structure**:
```csv
video_id,hook_average_face_size,hook_overlay_unique_count,...,middle_1_average_face_size,...,closing_average_face_size,...,create_time,gender,xwin_hook_to_middle_energy,xwin_middle_to_closing_energy,xwin_eye_contact_consistency,xwin_word_density_std,xwin_energy_progression_slope,is_top_performer
7428596413707144481,0.42,3,...,0.38,...,0.51,...,1704960000,female,0.12,-0.08,0.15,2.3,0.05,1
```

**Validation**: `stage3_aggregation.py::validate_output()` (line 654-697)

---

## Core Functions

### Function Call Chain

```
aggregate_features()  [Main Entry Point]
  ├─ validate_dependencies()                    # Pre-flight checks
  ├─ process_bucket()                          # Main processing loop
  │   ├─ For each video_file:
  │   │   ├─ validate_input()                   # JSON schema validation
  │   │   └─ extract_features()                 # 21 base features per window
  │   ├─ add_cross_window_features()           # 0-5 cross-window features
  │   └─ add_is_top_performer_label()          # Binary label (contrastive)
  ├─ validate_output()                         # Output schema validation
  ├─ save_aggregated_csv()                     # Atomic write
  └─ Write checkpoint JSON                      # For skip logic
```

---

### 1. aggregate_features()
**File**: `stage3_aggregation.py:732-853`
**Purpose**: Main Stage 3 pipeline orchestration

```python
def aggregate_features(
    bucket_path: str,
    strategy: str = 'contrastive'
) -> Tuple[Path, Path]:
```

**Parameters**:
- `bucket_path`: Path to bucket directory (e.g., "bucket_18-33s")
- `strategy`: "contrastive" (top 80% vs bottom 20%) or "top" (top only)

**Returns**: `(csv_path, summary_path)`

**Flow**:
1. Extract bucket name from path
2. Validate dependencies (raises ValueError)
3. Process all videos (raises ValueError if zero valid)
4. Validate output schema (raises AssertionError)
5. Atomic write to CSV (raises IOError)
6. Generate summary JSON
7. Write checkpoint (graceful degradation if fails)

**Exit Codes** (from orchestrator):
- 0: Success
- 1: ValueError (validation/processing failure)
- 3: AssertionError (output validation failure)
- 4: IOError (CSV write failure)
- 99: Unexpected error

---

### 2. process_bucket()
**File**: `stage3_aggregation.py:551-652`
**Purpose**: Process all videos, extract features, add cross-window features + labels

```python
def process_bucket(
    bucket_path: Path,
    bucket: str,
    strategy: str
) -> Tuple[pd.DataFrame, Dict]:
```

**Processing Loop** (lines 576-626):
```python
for video_file in json_files:
    try:
        # Extract video_id from filename
        video_id = video_file.stem.replace('_temporal_windows_updated', '')

        # Check duplicates
        if video_id in seen_video_ids:
            skipped_reasons['duplicate_video_id'] += 1
            continue

        # Load JSON
        with open(video_file) as f:
            data = json.load(f)

        # Validate input
        validate_input(data, video_id, bucket)

        # Extract features
        features = extract_features(video_file, bucket)
        aggregated_data.append(features)

    except json.JSONDecodeError:
        skipped_reasons['malformed_json'] += 1
        continue
    except ValueError:
        skipped_reasons['validation_error'] += 1
        continue
    except Exception:
        skipped_reasons['unexpected_error'] += 1
        continue
```

**Post-Processing**:
1. Convert list to DataFrame
2. Add 0-5 cross-window features (bucket-dependent)
3. Add is_top_performer label (strategy-dependent)

**Returns**: `(df_complete, skipped_reasons)`

**Raises**: `ValueError` if zero valid videos processed

---

### 3. extract_features()
**File**: `stage3_aggregation.py:179-292`
**Purpose**: Extract 21 base features from each temporal window

**BASE_FEATURES** (21 per window):
```python
[
    'average_face_size',        # Float, [0-1]
    'overlay_unique_count',     # Integer
    'has_captions',             # Boolean
    'scene_count',              # Integer
    'shortest_scene',           # Float, seconds
    'longest_scene',            # Float, seconds
    'scene_duration_variance',  # Float
    'object_count',             # Integer
    'person_count',             # Integer
    'dominant_emotion_id',      # Categorical, 1-7
    'speech_coverage',          # Float, [0-1]
    'word_count',               # Integer
    'energy_level',             # Float, [0-1]
    'energy_variance',          # Float
    'energy_max',               # Float, [0-1]
    'pitch_scatter_ratio',      # Float, [0-1]
    'gesture_count',            # Integer
    'gaze_variance',            # Float
    'eye_contact_rate',         # Float, [0-1]
    'emotional_valence',        # Float, [-1, 1]
    'emotion_consistency'       # Float, [0-1]
]
```

**Feature Extraction Logic**:

**Hook Features** (always present):
```python
for feature in BASE_FEATURES:
    video_features[f'hook_{feature}'] = windows['hook'].get(feature)
```

**Middle Features** (bucket-dependent):

**Case 1: No middle segments** (buckets 0-3s, 3-9s):
```python
if middle_segments is None or len(middle_segments) == 0:
    # Skip middle features
    pass
```

**Case 2: Aggregate middle segments** (buckets 9-13s, 13-18s):
```python
if bucket in ['9-13s', '13-18s']:
    # Aggregate all middle segments into "middle_aggregate_*"
    for feature in BASE_FEATURES:
        feature_values = [
            seg.get(feature)
            for seg in middle_segments
            if seg.get(feature) is not None
        ]

        if feature in SUM_FEATURES:
            video_features[f'middle_aggregate_{feature}'] = sum(feature_values)
        elif feature in MIN_FEATURES:
            video_features[f'middle_aggregate_{feature}'] = min(feature_values)
        elif feature in MAX_FEATURES:
            video_features[f'middle_aggregate_{feature}'] = max(feature_values)
        elif feature in CATEGORICAL_FEATURES:
            mode_series = pd.Series(feature_values).mode()
            video_features[f'middle_aggregate_{feature}'] = mode_series[0]
        else:
            # Default: average
            video_features[f'middle_aggregate_{feature}'] = np.mean(feature_values)
```

**Aggregation Strategies**:
- `SUM_FEATURES`: `['scene_count', 'word_count', 'object_count', 'person_count', 'overlay_unique_count', 'gesture_count']`
- `MIN_FEATURES`: `['shortest_scene']`
- `MAX_FEATURES`: `['longest_scene']`
- `CATEGORICAL_FEATURES`: `['dominant_emotion_id', 'has_captions']`

**Rationale**: Short windows (1-4s each) produce unreliable scene/speech features. Aggregation creates longer window (4.5-9.3s) for reliable measurements.

**Case 3: Keep separate middle segments** (buckets ≥ 18-33s):
```python
else:
    # Buckets 18-33s, 33-60s, 60-90s, 90-120s
    for i, segment in enumerate(middle_segments, start=1):
        for feature in BASE_FEATURES:
            video_features[f'middle_{i}_{feature}'] = segment.get(feature)
```

**Closing Features** (skip for bucket 0-3s):
```python
if bucket != '0-3s':
    for feature in BASE_FEATURES:
        video_features[f'closing_{feature}'] = windows['closing'].get(feature)
```

**Metadata** (2 video-level fields):
```python
video_features['create_time'] = metadata.get('create_time')

gender_data = metadata.get('gender_detection', {})
video_features['gender'] = gender_data.get('gender')
```

**Returns**: `Dict` with all features

---

### 4. validate_input()
**File**: `stage3_aggregation.py:485-549`
**Purpose**: Validate temporal_windows JSON before feature extraction

```python
def validate_input(data: dict, video_id: str, bucket: str):
    # 1. Check required top-level fields
    if 'metadata' not in data:
        raise ValueError(f"Video {video_id}: Missing 'metadata' field")

    if 'temporal_windows' not in data:
        raise ValueError(f"Video {video_id}: Missing 'temporal_windows' field")

    # 2. Validate temporal_windows structure
    windows = data['temporal_windows']

    if 'hook' not in windows:
        raise ValueError(f"Video {video_id}: Missing 'hook' window")

    if bucket != '0-3s' and 'closing' not in windows:
        raise ValueError(f"Video {video_id}: Missing 'closing' window")

    # 3. Validate middle_segments (bucket-specific)
    middle_segments = windows.get('middle_segments')
    expected_middle_count = BUCKET_MIDDLE_SEGMENTS.get(bucket, 0)

    if expected_middle_count > 0:
        if middle_segments is None or len(middle_segments) == 0:
            raise ValueError(
                f"Video {video_id}: null or empty middle_segments "
                f"(bucket {bucket} requires {expected_middle_count} segments)"
            )

        # Flexible validation for aggregation buckets
        if len(middle_segments) != expected_middle_count:
            logger.warning(
                f"Video {video_id}: Expected {expected_middle_count}, "
                f"found {len(middle_segments)}. Proceeding with aggregation."
            )
    else:
        # Buckets 0-3s, 3-9s should have null middle_segments
        if middle_segments is not None and len(middle_segments) > 0:
            logger.warning(
                f"Video {video_id}: Unexpected middle segments in bucket {bucket}"
            )

    # 4. Validate metadata
    if 'create_time' not in data['metadata']:
        raise ValueError(f"Video {video_id}: Missing metadata.create_time")
```

**Expected Middle Segment Counts**:
```python
BUCKET_MIDDLE_SEGMENTS = {
    '0-3s': 0, '3-9s': 0,
    '9-13s': 3, '13-18s': 3,
    '18-33s': 4, '33-60s': 5, '60-90s': 5, '90-120s': 5
}
```

---

### 5. add_cross_window_features()
**File**: `stage3_aggregation.py:294-398`
**Purpose**: Compute 0-5 cross-window derived features (added in S7B2 fix)

```python
def add_cross_window_features(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
```

**Cross-Window Features** (bucket-dependent):

| Bucket | Count | Features Added |
|--------|-------|----------------|
| 0-3s | 0 | None (only hook, no comparisons possible) |
| 3-9s | 3 | eye_contact_consistency, word_density_std, energy_progression_slope |
| 9-13s+ | 5 | All 5 features below |

**Feature Definitions** (for buckets with ≥2 windows):

**1. xwin_hook_to_middle_energy** (if hook + middle exist):
```python
df['xwin_hook_to_middle_energy'] = (
    df[middle_energy_cols].mean(axis=1) - df['hook_energy_level']
)
```
- Measures energy change from hook to middle average
- Positive = energy increases, Negative = energy decreases

**2. xwin_middle_to_closing_energy** (if middle + closing exist):
```python
df['xwin_middle_to_closing_energy'] = (
    df['closing_energy_level'] - df[middle_energy_cols].mean(axis=1)
)
```
- Measures energy change from middle to closing

**3. xwin_eye_contact_consistency** (if ≥2 windows):
```python
eye_contact_cols = [f'{w}_eye_contact_rate' for w in windows]
df['xwin_eye_contact_consistency'] = df[eye_contact_cols].std(axis=1)
```
- Standard deviation of eye_contact_rate across all windows
- Low std = consistent eye contact, High std = variable eye contact

**4. xwin_word_density_std** (if ≥2 windows):
```python
word_count_cols = [f'{w}_word_count' for w in windows]
df['xwin_word_density_std'] = df[word_count_cols].std(axis=1)
```
- Standard deviation of word_count (pacing variability)

**5. xwin_energy_progression_slope** (if ≥2 windows):
```python
# Linear regression slope across windows
for idx, row in df.iterrows():
    y = row[energy_cols].values.astype(float)
    x = np.arange(len(y))

    if pd.isna(y).any():
        slopes.append(np.nan)
    else:
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)

df['xwin_energy_progression_slope'] = slopes
```
- Positive slope = energy increases over time
- Negative slope = energy decreases over time
- Zero slope = consistent energy

**Implementation Notes**:
- Uses `config.bucket_definitions.BUCKET_WINDOWS` to get actual window names
- Handles `middle_aggregate` correctly for buckets 9-13s, 13-18s
- Gracefully handles NaN/None values

**Returns**: DataFrame with 0-5 additional columns

---

### 6. add_is_top_performer_label()
**File**: `stage3_aggregation.py:401-482`
**Purpose**: Add binary label for contrastive analysis (added in S7B2 fix)

```python
def add_is_top_performer_label(
    df: pd.DataFrame,
    bucket_path: Path,
    strategy: str
) -> pd.DataFrame:
```

**Strategy-Dependent Logic**:

**Top Mode**:
```python
if strategy == 'top':
    df['is_top_performer'] = 1
    # All videos are top performers by definition
```

**Contrastive Mode**:
```python
if strategy == 'contrastive':
    selected_videos_path = bucket_path / "selected_videos.json"

    if selected_videos_path.exists():
        # Load from Stage 1 output
        with open(selected_videos_path) as f:
            selected = json.load(f)

        # Create mapping: video_id → is_top_performer
        performer_map = {
            str(v['id']): v.get('is_top_performer', True)
            for v in selected['videos']
        }

        # Map to DataFrame
        df['is_top_performer'] = (
            df['video_id'].astype(str)
            .map(performer_map)
            .fillna(1)  # Default to top if missing
            .astype(int)
        )

        # Log distribution
        top_count = (df['is_top_performer'] == 1).sum()
        bottom_count = (df['is_top_performer'] == 0).sum()
        logger.info(f"{top_count} top, {bottom_count} bottom")

    else:
        # Fallback: Index-based 80/20 split
        top_count = int(len(df) * 0.8)
        df['is_top_performer'] = (df.index < top_count).astype(int)
```

**Purpose**: This label enables Stage 6 to compute distribution comparisons between top and bottom performers for RF feature importance analysis.

**Returns**: DataFrame with `is_top_performer` column (1 = top 80%, 0 = bottom 20%)

---

### 7. validate_output()
**File**: `stage3_aggregation.py:654-697`
**Purpose**: Validate aggregated DataFrame before saving

```python
def validate_output(df: pd.DataFrame, bucket: str):
    from config.bucket_definitions import get_stage3_expected_feature_count

    # 1. Check row count > 0
    assert len(df) > 0, "DataFrame has 0 rows"

    # 2. Check column count (uses config function)
    expected_cols = get_stage3_expected_feature_count(bucket)
    actual_cols = len(df.columns)

    assert actual_cols == expected_cols, \
        f"Column count mismatch: expected {expected_cols}, got {actual_cols}"

    # 3. Check required columns exist
    required_cols = ['video_id', 'create_time', 'is_top_performer']
    missing_cols = [c for c in required_cols if c not in df.columns]
    assert len(missing_cols) == 0, f"Missing: {missing_cols}"

    # 4. Check for completely null columns (warning only)
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        logger.warning(f"{len(null_cols)} null columns: {null_cols[:5]}")

    # 5. Validate video_id uniqueness
    duplicate_ids = df['video_id'].duplicated().sum()
    assert duplicate_ids == 0, f"{duplicate_ids} duplicate video_ids"
```

**Raises**: `AssertionError` with specific message

---

### 8. save_aggregated_csv()
**File**: `stage3_aggregation.py:699-730`
**Purpose**: Atomic write pattern to prevent corruption

```python
def save_aggregated_csv(df: pd.DataFrame, output_path: Path):
    temp_path = output_path.with_suffix('.tmp')

    try:
        # Write to temporary file
        df.to_csv(temp_path, index=False)

        # Atomic rename (only if write succeeded)
        shutil.move(str(temp_path), str(output_path))

        logger.info(f"Created {output_path.name}")

    finally:
        # Clean up temp file if rename failed
        if temp_path.exists():
            temp_path.unlink()
```

**Why Atomic**: Prevents partial CSV writes if process crashes mid-write. Either complete CSV or no CSV (no corrupted file).

**Raises**: `IOError/OSError` if write fails

---

## Checkpoint Strategy

### Checkpoint Schema

**File**: `{bucket_path}/checkpoints/stage_3_checkpoint.json`

```json
{
  "stage": "feature_aggregation",
  "status": "completed",
  "total_videos": 40,
  "output_files": ["aggregated_features.csv", "aggregation_summary.json"],
  "completion_time": "2025-01-28T10:30:00Z",
  "videos_processed": 38,
  "videos_skipped": 2,
  "bucket": "18-33s",
  "duration_seconds": 12.5,
  "feature_count": 135
}
```

### Orchestrator Skip Logic

**File**: `rumiai_ml_batch.py:1098-1128`

```python
checkpoint_path = bucket_path / "checkpoints" / "stage_3_checkpoint.json"

if checkpoint_path.exists():
    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Validate checkpoint status
        if checkpoint.get("status") != "completed":
            logger.warning("Checkpoint status != completed. Re-running.")
            checkpoint_path.unlink()
        else:
            # Load summary for reporting
            stage3_summaries[bucket_name] = {
                "videos_processed": checkpoint.get("videos_processed", 0),
                "videos_skipped": checkpoint.get("videos_skipped", 0),
                "output_csv": {"columns": checkpoint.get("feature_count", "unknown")},
                "duration_seconds": checkpoint.get("duration_seconds", 0)
            }
            continue  # Skip to next bucket

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Invalid checkpoint: {e}. Re-running.")
        checkpoint_path.unlink()
```

### Graceful Degradation

**File**: `stage3_aggregation.py:835-850`

```python
try:
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint written: {checkpoint_path}")

except Exception as e:
    logger.warning(f"Checkpoint write failed: {e}")
    logger.warning(
        "Stage 3 outputs ARE VALID (CSV created successfully). "
        "Pipeline will continue. "
        "NOTE: Stage 3 will re-run on next execution (no checkpoint for skip logic)."
    )
    # Don't raise - CSV is source of truth, checkpoint is optimization
```

**Philosophy**: Checkpoint is performance optimization, not critical. CSV is the source of truth.

---

# Part B: Stage 3.4 Review CSV Generation

## Overview

**Purpose**: Generate video_review.csv with clickable TikTok URLs for manual outlier inspection
**Key Feature**: **IDENTICAL feature extraction** to Stage 3 + url field
**Status**: Optional (failures logged, don't halt pipeline)

## Input Contract

### Prerequisites
- **Stage 2** complete → `analysis/insights/*_temporal_windows_updated.json`

### Input Files
```
{bucket_path}/analysis/insights/
└── {video_id}_temporal_windows_updated.json  # Contains metadata.url field
```

---

## Output Contract

### Files Created
```
{bucket_path}/validation/
└── video_review.csv                  # Clickable URLs + all features
```

### Output Schema

**video_review.csv** (same features as aggregated_features.csv + url):

```csv
video_id,url,duration,hook_average_face_size,...,middle_1_average_face_size,...,closing_average_face_size,...,create_time,gender
="7428596413707144481",https://www.tiktok.com/@user/video/7428596413707144481,18.5,0.42,...,0.38,...,0.51,...,1704960000,female
```

**Differences from aggregated_features.csv**:
1. ✅ Includes `url` column (position 2)
2. ✅ Includes `duration` column (position 3)
3. ❌ Excludes cross-window features (`xwin_*`)
4. ❌ Excludes `is_top_performer` label
5. ✅ video_id formatted as Excel text: `="video_id"` (prevents rounding)

**Column Ordering**: video_id, url, duration, then temporal features in progressive order (hook → middle → closing), then metadata

---

## Core Functions

### Function Call Chain

```
generate_review_csv_for_bucket()  [Main Entry Point]
  ├─ load_temporal_windows()                   # Load all JSONs
  ├─ extract_features_with_url()               # Extract features (loop)
  ├─ filter_videos_with_url()                  # Remove missing URLs
  └─ generate_review_csv()                     # Create CSV with temporal ordering
```

---

### 1. generate_review_csv_for_bucket()
**File**: `review_csv_generator.py:309-347`
**Purpose**: Complete Stage 3.4 pipeline

```python
def generate_review_csv_for_bucket(bucket_path: Path) -> None:
    logger.info(f"Stage 3.4: Review CSV Generation starting")

    # Extract bucket name
    bucket_name = bucket_path.name.replace('bucket_', '')

    # Step 1: Load temporal windows
    temporal_data = load_temporal_windows(bucket_path)

    # Step 2: Extract features with url
    feature_rows = []
    for tw_data in temporal_data:
        features = extract_features_with_url(tw_data, bucket_name)
        feature_rows.append(features)

    # Step 3: Filter videos with valid url
    valid_rows = filter_videos_with_url(feature_rows)

    # Step 4: Generate review CSV
    output_path = bucket_path / "validation" / "video_review.csv"
    generate_review_csv(valid_rows, output_path)
```

**Raises**:
- `FileNotFoundError` - insights/ directory missing
- `ValueError` - No JSON files or all videos missing url
- `IOError` - Cannot write CSV

---

### 2. load_temporal_windows()
**File**: `review_csv_generator.py:54-92`
**Purpose**: Load all temporal_windows JSON files

```python
def load_temporal_windows(bucket_path: Path) -> List[Dict[str, Any]]:
    insights_dir = bucket_path / "analysis" / "insights"

    if not insights_dir.exists():
        raise FileNotFoundError(
            f"insights/ directory not found: {insights_dir}. "
            "Stage 2 must complete before Stage 3.4"
        )

    json_files = sorted(insights_dir.glob("*_temporal_windows_updated.json"))

    if len(json_files) == 0:
        raise ValueError(f"No JSON files in {insights_dir}")

    temporal_data = []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            temporal_data.append(data)

    return temporal_data
```

**Returns**: `List[Dict]` - List of temporal windows data

---

### 3. extract_features_with_url()
**File**: `review_csv_generator.py:94-187`
**Purpose**: **IDENTICAL extraction logic to Stage 3** + url field

**Shared Constants** (lines 34-41):
```python
# Import from Stage 3 to guarantee identical extraction
from stage3_aggregation import (
    BASE_FEATURES,
    AGGREGATE_MIDDLE_BUCKETS,
    SUM_FEATURES,
    MIN_FEATURES,
    MAX_FEATURES,
    CATEGORICAL_FEATURES
)
```

**Feature Extraction** (identical to Stage 3):
```python
def extract_features_with_url(temporal_windows: Dict, bucket: str) -> Dict:
    metadata = temporal_windows.get('metadata', {})
    video_id = metadata.get('video_id', 'unknown')
    url = metadata.get('url')  # May be None
    duration = metadata.get('duration', 0)

    windows = temporal_windows.get('temporal_windows', {})

    # Initialize with metadata
    video_features = {'video_id': video_id, 'url': url, 'duration': duration}

    # Hook features (identical to Stage 3)
    for feature in BASE_FEATURES:
        video_features[f'hook_{feature}'] = windows['hook'].get(feature)

    # Middle features (identical aggregation logic to Stage 3)
    middle_segments = windows.get('middle_segments')

    if middle_segments:
        if bucket in AGGREGATE_MIDDLE_BUCKETS:
            # Aggregate middle segments (9-13s, 13-18s)
            for feature in BASE_FEATURES:
                feature_values = [
                    seg.get(feature) for seg in middle_segments
                    if seg.get(feature) is not None
                ]

                if not feature_values:
                    video_features[f'middle_aggregate_{feature}'] = None
                    continue

                # Apply aggregation strategy (identical to Stage 3)
                if feature in SUM_FEATURES:
                    video_features[f'middle_aggregate_{feature}'] = sum(feature_values)
                elif feature in MIN_FEATURES:
                    video_features[f'middle_aggregate_{feature}'] = min(feature_values)
                elif feature in MAX_FEATURES:
                    video_features[f'middle_aggregate_{feature}'] = max(feature_values)
                elif feature in CATEGORICAL_FEATURES:
                    mode_series = pd.Series(feature_values).mode()
                    video_features[f'middle_aggregate_{feature}'] = mode_series[0]
                else:
                    video_features[f'middle_aggregate_{feature}'] = np.mean(feature_values)

        else:
            # Keep separate middle segments (18-33s+)
            for i, segment in enumerate(middle_segments, start=1):
                for feature in BASE_FEATURES:
                    video_features[f'middle_{i}_{feature}'] = segment.get(feature)

    # Closing features (identical to Stage 3)
    if bucket != '0-3s':
        for feature in BASE_FEATURES:
            video_features[f'closing_{feature}'] = windows['closing'].get(feature)

    # Metadata (identical to Stage 3)
    video_features['create_time'] = metadata.get('create_time')

    gender_data = metadata.get('gender_detection', {})
    video_features['gender'] = gender_data.get('gender')

    return video_features
```

**Key Difference from Stage 3**:
- Takes already-loaded dict (not file path)
- Adds `url` and `duration` fields from metadata
- **Does NOT add cross-window features or is_top_performer label**

---

### 4. filter_videos_with_url()
**File**: `review_csv_generator.py:190-226`
**Purpose**: Remove videos with missing/empty url field

```python
def filter_videos_with_url(feature_rows: List[Dict]) -> List[Dict]:
    valid_rows = []
    skipped_count = 0

    for row in feature_rows:
        video_id = row.get('video_id', 'unknown')
        url = row.get('url')

        # Filter: url must be non-empty string
        if not url or (isinstance(url, str) and url.strip() == ''):
            logger.warning(f"Video {video_id} excluded - missing url")
            skipped_count += 1
            continue

        valid_rows.append(row)

    if skipped_count > 0:
        logger.info(
            f"Excluded {skipped_count} videos (missing url). "
            f"These remain in aggregated_features.csv for ML training."
        )

    return valid_rows
```

**Rationale**: Review CSV is for manual inspection via clickable URLs. Videos without URLs can't be inspected, but should still be included in ML training (aggregated_features.csv).

---

### 5. generate_review_csv()
**File**: `review_csv_generator.py:228-307`
**Purpose**: Create final CSV with temporal column ordering + Excel compatibility

```python
def generate_review_csv(feature_rows: List[Dict], output_path: Path) -> None:
    if not feature_rows:
        raise ValueError("No videos with valid url")

    df = pd.DataFrame(feature_rows)

    # Column ordering: video_id, url, duration, then temporal features
    cols = ['video_id', 'url', 'duration']
    other_cols = [c for c in df.columns if c not in cols]

    # Sort features in temporal order (not alphabetical)
    def temporal_sort_key(col_name):
        if col_name.startswith('hook_'):
            return (0, col_name)

        elif col_name.startswith('middle_'):
            if '_aggregate_' in col_name:
                return (1, 0, col_name)  # Before middle_1
            else:
                parts = col_name.split('_')
                if len(parts) >= 2 and parts[1].isdigit():
                    segment_num = int(parts[1])
                    return (1, segment_num, col_name)
                else:
                    return (1, 999, col_name)

        elif col_name.startswith('closing_'):
            return (2, col_name)

        else:
            # Metadata (create_time, gender) at end
            return (3, col_name)

    other_cols_sorted = sorted(other_cols, key=temporal_sort_key)
    df = df[cols + other_cols_sorted]

    # Excel compatibility: Convert video_id to text formula
    # Prevents Excel from rounding 19-digit TikTok IDs
    df['video_id'] = '="' + df['video_id'].astype(str) + '"'

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL

    logger.info(f"✅ Generated video_review.csv: {len(df)} rows")
```

**Column Ordering Result**:
1. video_id, url, duration
2. hook_* columns (alphabetical within hook)
3. middle_aggregate_* or middle_1_*, middle_2_*, ... (sorted by number)
4. closing_* columns (alphabetical within closing)
5. create_time, gender

**Excel Compatibility**:
- video_id formatted as `="7428596413707144481"` (Excel treats as text, not number)
- Prevents rounding of 19-digit IDs (Excel preserves only 15 digits for numbers)
- When opened in Excel, the `="` prefix is hidden, displays as plain text

---

# Shared Configuration Reference

## config/bucket_definitions.py

### BUCKET_WINDOWS (Master Configuration)

```python
BUCKET_WINDOWS = {
    '0-3s': ['hook'],
    '3-9s': ['hook', 'closing'],
    '9-13s': ['hook', 'middle_aggregate', 'closing'],
    '13-18s': ['hook', 'middle_aggregate', 'closing'],
    '18-33s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing'],
    '33-60s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
    '60-90s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
    '90-120s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
}
```

**Critical**: Note `middle_aggregate` for buckets 9-13s, 13-18s (not middle_1/2/3).

---

### get_stage3_expected_feature_count()

**File**: `bucket_definitions.py:139-200`

**Purpose**: **SINGLE SOURCE OF TRUTH** for Stage 3 output column count

**Formula**: `(21 × window_count) + 3 metadata + X cross-window + 1 label`

```python
def get_stage3_expected_feature_count(bucket: str) -> int:
    window_count = len(BUCKET_WINDOWS[bucket])
    base_features = 21 * window_count
    metadata_cols = 3  # video_id, create_time, gender

    # Cross-window features (bucket-aware)
    if bucket == '0-3s':
        cross_window_features = 0
    elif bucket == '3-9s':
        cross_window_features = 3
    else:
        cross_window_features = 5

    label_cols = 1  # is_top_performer

    return base_features + metadata_cols + cross_window_features + label_cols
```

**Expected Counts**:
- 0-3s: 25 = 21×1 + 3 + 0 + 1
- 3-9s: 49 = 21×2 + 3 + 3 + 1
- 9-13s: 72 = 21×3 + 3 + 5 + 1
- 13-18s: 72 = 21×3 + 3 + 5 + 1
- 18-33s: 135 = 21×6 + 3 + 5 + 1
- 33-60s: 156 = 21×7 + 3 + 5 + 1
- 60-90s: 156 = 21×7 + 3 + 5 + 1
- 90-120s: 156 = 21×7 + 3 + 5 + 1

---

### get_window_count()

**File**: `bucket_definitions.py:33-52`

```python
def get_window_count(bucket: str) -> int:
    if bucket not in BUCKET_WINDOWS:
        raise ValueError(f"Unknown bucket: {bucket}")
    return len(BUCKET_WINDOWS[bucket])
```

**Example**:
```python
>>> get_window_count('18-33s')
6
>>> get_window_count('9-13s')
3
```

---

### get_windows()

**File**: `bucket_definitions.py:54-73`

```python
def get_windows(bucket: str) -> list:
    if bucket not in BUCKET_WINDOWS:
        raise ValueError(f"Unknown bucket: {bucket}")
    return BUCKET_WINDOWS[bucket]
```

**Example**:
```python
>>> get_windows('18-33s')
['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

>>> get_windows('9-13s')
['hook', 'middle_aggregate', 'closing']
```

---

# Data Flow & Architecture

## Complete Pipeline Flow

```
Stage 2.5 (File Organization)
    ↓
    buckets/bucket_{name}/analysis/insights/*.json
    ↓
Stage 3 (Feature Aggregation)
    ├─ validate_dependencies()
    ├─ For each video:
    │   ├─ validate_input()
    │   └─ extract_features()
    │       ├─ Hook: 21 features
    │       ├─ Middle: 0-5 segments (21 features each or aggregated)
    │       ├─ Closing: 21 features (skip for 0-3s)
    │       └─ Metadata: 2 fields
    ├─ add_cross_window_features() → 0-5 features
    ├─ add_is_top_performer_label() → 1 column
    ├─ validate_output()
    ├─ save_aggregated_csv() → aggregated_features.csv
    └─ Write checkpoint
    ↓
Stage 3.4 (Review CSV Generation) - OPTIONAL
    ├─ load_temporal_windows()
    ├─ extract_features_with_url() (identical to Stage 3 + url)
    ├─ filter_videos_with_url()
    └─ generate_review_csv() → video_review.csv
    ↓
Stage 4 (Feature Transformation)
```

---

## File Dependency Graph

```
selected_videos.json (Stage 1)
    └─→ Stage 3: add_is_top_performer_label() (contrastive mode only)

*_temporal_windows_updated.json (Stage 2, organized by Stage 2.5)
    ├─→ Stage 3: extract_features()
    └─→ Stage 3.4: extract_features_with_url()

aggregated_features.csv (Stage 3)
    ├─→ Stage 3.4: For comparison (not directly used)
    ├─→ Stage 4: Feature Transformation
    └─→ Stage 6: ML Analysis (needs is_top_performer for distribution comparisons)

video_review.csv (Stage 3.4)
    └─→ Human manual review (Excel inspection)
```

---

## Middle Segment Aggregation Flow

**Problem**: Short windows (1-4s) produce unreliable scene/speech features

**Solution**: Aggregate all middle segments into single "middle_aggregate" window

**Flow for buckets 9-13s, 13-18s**:

```
Input: temporal_windows.middle_segments = [
  {scene_count: 2, word_count: 5, ...},  # Segment 1 (1.5s)
  {scene_count: 1, word_count: 3, ...},  # Segment 2 (2.0s)
  {scene_count: 3, word_count: 7, ...}   # Segment 3 (1.8s)
]

Aggregation Logic:
├─ SUM_FEATURES (scene_count): 2 + 1 + 3 = 6
├─ SUM_FEATURES (word_count): 5 + 3 + 7 = 15
├─ MIN_FEATURES (shortest_scene): min([0.5, 0.8, 0.6]) = 0.5
├─ MAX_FEATURES (longest_scene): max([1.2, 1.5, 1.0]) = 1.5
├─ CATEGORICAL_FEATURES (dominant_emotion_id): mode([3, 3, 4]) = 3
└─ DEFAULT (average_face_size): mean([0.42, 0.38, 0.45]) = 0.42

Output: middle_aggregate_{feature} columns
```

**Resulting Window**: Aggregated 5.3s window with reliable measurements

---

# Error Handling Matrix

## Stage 3 Errors

### Orchestrator-Level Errors

| Error Type | Cause | Handled By | Action | Exit Code |
|------------|-------|------------|--------|-----------|
| `ValueError` | Pre-flight validation failed | Orchestrator | Exit pipeline | 1 |
| `ValueError` | Zero valid videos processed | Orchestrator | Exit pipeline | 1 |
| `AssertionError` | Output validation failed | Orchestrator | Exit pipeline | 3 |
| `IOError/OSError` | CSV write failure | Orchestrator | Exit pipeline | 4 |
| `Exception` | Unexpected error | Orchestrator | Exit pipeline | 99 |

**Source**: `rumiai_ml_batch.py:1209-1240`

---

### Video-Level Errors (Graceful Skip)

| Error Type | Cause | Handled By | Action | Logged As |
|------------|-------|------------|--------|-----------|
| `json.JSONDecodeError` | Malformed JSON | `process_bucket()` | Skip video | malformed_json |
| `ValueError` | Validation error (missing fields, null middle_segments) | `process_bucket()` | Skip video | validation_error |
| `Exception` | Unexpected error | `process_bucket()` | Skip video | unexpected_error |
| Duplicate video_id | Same video_id in multiple files | `process_bucket()` | Skip video | duplicate_video_id |

**Source**: `stage3_aggregation.py:607-625`

**Philosophy**: Fail gracefully per video, continue processing remaining videos. Only fail pipeline if **zero** valid videos processed.

---

## Stage 3.4 Errors

### Non-Fatal Errors (Logged Warnings)

| Error Type | Cause | Handled By | Action | Pipeline Impact |
|------------|-------|------------|--------|-----------------|
| `ValueError` | All videos missing url | Orchestrator | Log warning | Continue (review CSV optional) |
| `IOError/OSError` | CSV write failure | Orchestrator | Log error | Continue (review CSV optional) |
| `Exception` | Unexpected error | Orchestrator | Log error | Continue (review CSV optional) |

**Source**: `rumiai_ml_batch.py:1169-1190`

**Philosophy**: Stage 3.4 is optional. Failures don't halt pipeline. aggregated_features.csv is the source of truth.

---

### Fatal Errors (Raise Exception)

| Error Type | Cause | Action |
|------------|-------|--------|
| `FileNotFoundError` | insights/ directory missing | Raise (indicates Stage 2 incomplete) |
| `ValueError` | No JSON files found | Raise (indicates Stage 2.5 incomplete) |

**Source**: `review_csv_generator.py:72-83`

---

## Retry Strategies

**Stage 3**: No retries at video level. Skip bad videos, continue processing.

**Stage 3.4**: No retries. Entire stage is optional (non-fatal).

---

# Debugging Guide

## Stage 3 Troubleshooting

### Issue: Column count mismatch

**Symptom**: `AssertionError: Column count mismatch: expected 135, got 140`

**Cause**: Cross-window features or is_top_performer logic changed

**Debug**:
```bash
# Check actual columns in CSV
head -1 data/clients/test/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/ml_analysis/aggregated_features.csv | tr ',' '\n' | wc -l

# Check expected count from config
python3 -c "
from config.bucket_definitions import get_stage3_expected_feature_count
print(get_stage3_expected_feature_count('18-33s'))
"
```

**Fix**:
1. Verify cross-window feature count matches config (0, 3, or 5)
2. Verify is_top_performer column exists
3. Update `config/bucket_definitions.py` if feature count changed

---

### Issue: All videos skipped (zero valid videos)

**Symptom**: `ValueError: No valid videos processed`

**Cause**: Validation failures on all videos

**Debug**:
```bash
# Check logs for validation errors
grep "validation_error" logs/rumiai_*.log | head -10

# Common issues:
# - Missing 'hook' window
# - Missing 'closing' window (for buckets != 0-3s)
# - null/empty middle_segments (for buckets requiring them)
# - Missing metadata.create_time
```

**Fix**:
1. Verify Stage 2 temporal_windows structure is correct
2. Check if bucket name matches actual video durations
3. Validate temporal_windows_updated.json schema manually:
```bash
cat {bucket_path}/analysis/insights/{video_id}_temporal_windows_updated.json | jq '.temporal_windows | keys'
# Should show: ["closing", "hook", "middle_segments"] (or subset)
```

---

### Issue: Middle segment aggregation warning

**Symptom**: `Expected 3 middle segments, found 2. Proceeding with aggregation.`

**Cause**: Video duration at bucket boundary (e.g., 9.0s vs 13.0s)

**Debug**:
```bash
# Check video duration
cat {video_id}_temporal_windows_updated.json | jq '.duration'

# Check middle_segments count
cat {video_id}_temporal_windows_updated.json | jq '.temporal_windows.middle_segments | length'
```

**Fix**: This is a warning, not an error. Aggregation proceeds with available segments. If persistent across many videos, check temporal_compute.py logic for bucket thresholds.

---

### Issue: is_top_performer all 1s (no contrast)

**Symptom**: All videos have `is_top_performer=1` in contrastive mode

**Cause**: selected_videos.json missing or doesn't have `is_top_performer` field

**Debug**:
```bash
# Check if selected_videos.json exists
ls {bucket_path}/selected_videos.json

# Check if it has is_top_performer field
cat {bucket_path}/selected_videos.json | jq '.videos[0] | has("is_top_performer")'

# Check distribution
cat ml_analysis/aggregated_features.csv | cut -d',' -f$(head -1 ml_analysis/aggregated_features.csv | tr ',' '\n' | grep -n "is_top_performer" | cut -d: -f1) | sort | uniq -c
```

**Fix**:
1. If selected_videos.json missing → Stage 1 incomplete, re-run
2. If is_top_performer field missing → Update Stage 1 to add field
3. Fallback behavior: Index-based 80/20 split (logs warning)

---

### Issue: Checkpoint invalid, keeps re-running

**Symptom**: Stage 3 re-runs every time despite completing successfully

**Cause**: Checkpoint schema invalid or status != "completed"

**Debug**:
```bash
# Check checkpoint exists
ls {bucket_path}/checkpoints/stage_3_checkpoint.json

# Check checkpoint schema
cat {bucket_path}/checkpoints/stage_3_checkpoint.json | jq '.'

# Verify status field
cat {bucket_path}/checkpoints/stage_3_checkpoint.json | jq '.status'
```

**Fix**:
1. Delete invalid checkpoint: `rm {bucket_path}/checkpoints/stage_3_checkpoint.json`
2. Re-run Stage 3 (will create valid checkpoint)
3. Verify checkpoint after completion

---

## Stage 3.4 Troubleshooting

### Issue: No videos with valid url

**Symptom**: `ValueError: No videos with valid url for review CSV generation`

**Cause**: metadata.url field missing from all temporal_windows files

**Debug**:
```bash
# Check if url field exists in temporal_windows
cat {bucket_path}/analysis/insights/{video_id}_temporal_windows_updated.json | jq '.metadata.url'

# Count videos with missing url
for f in {bucket_path}/analysis/insights/*_temporal_windows_updated.json; do
  jq -r '.metadata.url // "MISSING"' "$f"
done | grep MISSING | wc -l
```

**Fix**:
1. Verify Stage 2 populates metadata.url (check video_analyzer.py)
2. If url missing from all videos → Stage 2 modification needed
3. Stage 3.4 is optional → Pipeline continues without review CSV

---

### Issue: Review CSV has different features than aggregated CSV

**Symptom**: Feature values don't match between video_review.csv and aggregated_features.csv

**Cause**: Feature extraction logic diverged (not using shared constants)

**Debug**:
```bash
# Compare column counts
head -1 ml_analysis/aggregated_features.csv | tr ',' '\n' | wc -l
head -1 validation/video_review.csv | tr ',' '\n' | wc -l

# Check for cross-window features (should NOT be in review CSV)
head -1 validation/video_review.csv | grep -o "xwin_"
# Should return empty (no cross-window features)

# Check for is_top_performer (should NOT be in review CSV)
head -1 validation/video_review.csv | grep -o "is_top_performer"
# Should return empty
```

**Fix**: Verify review_csv_generator.py imports constants from stage3_aggregation.py (lines 34-41)

---

## Performance Profiling

### Stage 3 Timing

```bash
# Check logs for duration
grep "Stage 3 complete" logs/rumiai_*.log

# Expected timing:
# - 10-20s for 40 videos (bucket 18-33s)
# - 5-10s for 40 videos (bucket 0-3s, fewer features)
```

### Stage 3.4 Timing

```bash
# Check logs for duration
grep "Stage 3.4 complete" logs/rumiai_*.log

# Expected timing:
# - 5-10s for 40 videos (all buckets)
```

---

## Quick Debugging Commands

```bash
# Check aggregated CSV exists
ls {bucket_path}/ml_analysis/aggregated_features.csv

# Check CSV row/column count
wc -l {bucket_path}/ml_analysis/aggregated_features.csv
head -1 {bucket_path}/ml_analysis/aggregated_features.csv | tr ',' '\n' | wc -l

# Check for null columns
python3 -c "
import pandas as pd
df = pd.read_csv('{bucket_path}/ml_analysis/aggregated_features.csv')
null_cols = df.columns[df.isnull().all()].tolist()
print(f'Null columns: {null_cols}')
"

# Check is_top_performer distribution
python3 -c "
import pandas as pd
df = pd.read_csv('{bucket_path}/ml_analysis/aggregated_features.csv')
print(df['is_top_performer'].value_counts())
"

# Check checkpoint status
jq '.status' {bucket_path}/checkpoints/stage_3_checkpoint.json

# Count videos skipped
jq '.videos_skipped' {bucket_path}/checkpoints/stage_3_checkpoint.json

# Check skipped reasons
jq '.skipped_reasons' {bucket_path}/ml_analysis/aggregation_summary.json

# Verify review CSV has URLs
cut -d',' -f2 {bucket_path}/validation/video_review.csv | grep "tiktok.com" | wc -l
```

---

## Modification Guide

### Adding a New Base Feature

**Scenario**: Add "audio_clarity" as 22nd base feature

**Steps**:

1. **Update BASE_FEATURES** (`stage3_aggregation.py:39-61`)
   ```python
   BASE_FEATURES = [
       'average_face_size',
       # ... existing features ...
       'emotion_consistency',
       'audio_clarity'  # ADD
   ]
   ```

2. **Update expected column counts** (`config/bucket_definitions.py:139-200`)
   ```python
   # Formula changes from 21 × window_count to 22 × window_count
   def get_stage3_expected_feature_count(bucket: str) -> int:
       window_count = len(BUCKET_WINDOWS[bucket])
       base_features = 22 * window_count  # CHANGE from 21
       # ... rest unchanged
   ```

3. **Verify temporal_windows JSON includes new feature**
   ```bash
   cat {video_id}_temporal_windows_updated.json | jq '.temporal_windows.hook.audio_clarity'
   ```

4. **Test**:
   ```bash
   # Re-run Stage 3
   python3 rumiai_ml_batch.py --client test --target "#nutrition" --start-stage 3

   # Verify new column exists
   head -1 ml_analysis/aggregated_features.csv | tr ',' '\n' | grep "audio_clarity"
   ```

5. **Expected Changes**:
   - 0-3s: 25 → 26 columns
   - 3-9s: 49 → 51 columns
   - 18-33s: 135 → 141 columns
   - etc.

---

### Adding a New Cross-Window Feature

**Scenario**: Add "xwin_emotion_variance" (variance of dominant_emotion_id across windows)

**Steps**:

1. **Update add_cross_window_features()** (`stage3_aggregation.py:294-398`)
   ```python
   def add_cross_window_features(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
       # ... existing features ...

       # Feature 6: Emotion variance across windows
       emotion_cols = [f'{w}_dominant_emotion_id' for w in windows]
       if len(emotion_cols) >= 2:
           df['xwin_emotion_variance'] = df[emotion_cols].var(axis=1)
           cross_window_count += 1
           logger.debug("  ✓ Computed xwin_emotion_variance")

       return df
   ```

2. **Update expected column counts** (`config/bucket_definitions.py:186-195`)
   ```python
   # Update cross-window feature counts
   if bucket == '0-3s':
       cross_window_features = 0  # No change
   elif bucket == '3-9s':
       cross_window_features = 4  # CHANGE from 3
   else:
       cross_window_features = 6  # CHANGE from 5
   ```

3. **Test**:
   ```bash
   # Re-run Stage 3
   python3 rumiai_ml_batch.py --client test --target "#nutrition" --start-stage 3

   # Verify new column exists
   head -1 ml_analysis/aggregated_features.csv | tr ',' '\n' | grep "xwin_emotion_variance"

   # Check values are computed
   python3 -c "
   import pandas as pd
   df = pd.read_csv('ml_analysis/aggregated_features.csv')
   print(df['xwin_emotion_variance'].describe())
   "
   ```

4. **Expected Changes**:
   - 0-3s: No change (still 25)
   - 3-9s: 49 → 50 columns
   - 9-13s+: 72 → 73 columns (or +1 to current)

---

### Changing Aggregation Strategy for a Feature

**Scenario**: Change "gesture_count" from SUM to AVERAGE for middle segments

**Steps**:

1. **Move feature from SUM_FEATURES to default (average)** (`stage3_aggregation.py:82-89`)
   ```python
   # Before:
   SUM_FEATURES = [
       'scene_count', 'word_count', 'object_count',
       'person_count', 'overlay_unique_count', 'gesture_count'  # REMOVE
   ]

   # After:
   SUM_FEATURES = [
       'scene_count', 'word_count', 'object_count',
       'person_count', 'overlay_unique_count'
   ]
   ```

2. **Update review_csv_generator.py** (imports constants, so automatic)

3. **Test with known data**:
   ```bash
   # Create test JSON with known gesture_count values
   # middle_segments = [{gesture_count: 2}, {gesture_count: 4}, {gesture_count: 6}]

   # Before: middle_aggregate_gesture_count = 2 + 4 + 6 = 12
   # After: middle_aggregate_gesture_count = mean([2, 4, 6]) = 4.0
   ```

4. **Re-run Stage 3 and verify**:
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_csv('ml_analysis/aggregated_features.csv')
   print(df['middle_aggregate_gesture_count'].describe())
   "
   ```

---

### Adding Stage 3.4 URL Validation

**Scenario**: Validate URL format before including in review CSV

**Steps**:

1. **Update filter_videos_with_url()** (`review_csv_generator.py:190-226`)
   ```python
   def filter_videos_with_url(feature_rows: List[Dict]) -> List[Dict]:
       import re

       valid_rows = []
       skipped_count = 0

       for row in feature_rows:
           video_id = row.get('video_id', 'unknown')
           url = row.get('url')

           # Filter 1: url must be non-empty string
           if not url or (isinstance(url, str) and url.strip() == ''):
               logger.warning(f"Video {video_id} excluded - missing url")
               skipped_count += 1
               continue

           # Filter 2: url must match TikTok pattern (NEW)
           if not re.match(r'^https?://.*tiktok\.com/', url):
               logger.warning(f"Video {video_id} excluded - invalid url format: {url}")
               skipped_count += 1
               continue

           valid_rows.append(row)

       return valid_rows
   ```

2. **Test**:
   ```bash
   # Re-run Stage 3.4
   python3 -c "
   from pathlib import Path
   from ml_pipeline.stage3_aggregation.review_csv_generator import generate_review_csv_for_bucket
   generate_review_csv_for_bucket(Path('bucket_18-33s'))
   "

   # Verify all URLs are valid
   cut -d',' -f2 validation/video_review.csv | tail -n +2 | grep -v "tiktok.com"
   # Should return empty (all URLs valid)
   ```

---

## Related Documentation

- **PRODUCTION_FLOW.md**: [Stage 3 Contract](../../PRODUCTION_FLOW.md#stage-3-feature-aggregation)
- **Technical Specs**:
  - `FeatureAggregationCHILD.md` - Original specification
  - `ReviewCSVGenerationCHILD.md` - Stage 3.4 specification
  - `S7B2.md` - Cross-window features + is_top_performer fix
- **Upstream Stages**:
  - [STAGE_2_IMPL.md](STAGE_2_IMPL.md) - ML Processing (creates temporal_windows)
  - [STAGE_2.5_IMPL.md](STAGE_2.5_IMPL.md) - File Organization (organizes files)
- **Downstream Stages**:
  - [STAGE_4_IMPL.md](STAGE_4_IMPL.md) - Feature Transformation (consumes aggregated_features.csv)
  - [STAGE_6_IMPL.md](STAGE_6_IMPL.md) - ML Analysis (uses is_top_performer for RF distribution comparisons)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-28
**Source**: 100% systematic code reading (1,654 production lines across 4 modules)
**Maintainer**: Update when Stage 3 or 3.4 implementation changes
