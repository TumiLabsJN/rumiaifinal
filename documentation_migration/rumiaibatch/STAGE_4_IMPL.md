# Stage 4: Feature Transformation - Implementation Guide

**Purpose**: Transform aggregated features into ML-ready formats for Random Forest and K-Means training
**Target Audience**: LLM agents fixing bugs, adding features, or modifying feature transformation logic
**Related**: [PRODUCTION_FLOW.md Stage 4 Contract](../../PRODUCTION_FLOW.md#stage-4-feature-transformation)

**Source**: 100% systematic code reading of 3 modules (1,569 production lines)

---

## Quick Reference

### Stage 4: Feature Transformation
- **Entry Point**: `rumiai_v2/processors/feature_transformation.py::run_stage4_transformation()` (line 874)
- **Orchestrator Call**: `rumiai_ml_batch.py:1261-1452`
- **Checkpoint**: `{bucket_path}/checkpoints/stage_4_checkpoint.json`
- **Duration**: ~7-30s per bucket (100 videos, bucket 18-33s)
- **Output**: 13 files per bucket (1 Video-Level RF + 6 Window-Level RF + 6 Window-Level K-Means)
- **Key Feature**: Scaler persistence (.pkl files) for inference reproducibility

### Module Structure
```
rumiai_v2/processors/
└── feature_transformation.py (1,088 lines)  # Complete Stage 4 implementation

config/
├── stage4_constants.py (89 lines)          # Stage 4 specific constants
└── bucket_definitions.py (200 lines)       # Shared bucket configurations
```

---

## Table of Contents

1. [Overview](#overview)
2. [Input Contract](#input-contract)
3. [Output Contract](#output-contract)
4. [Core Functions](#core-functions)
5. [Transformation Details](#transformation-details)
6. [Validation & Checkpoint](#validation--checkpoint)
7. [Data Flow & Architecture](#data-flow--architecture)
8. [Error Handling Matrix](#error-handling-matrix)
9. [Debugging Guide](#debugging-guide)
10. [Modification Guide](#modification-guide)

---

# Overview

**Purpose**: Transform Stage 3's aggregated_features.csv (350+ features) into three ML-ready formats:
1. **Video-Level RF**: Cross-window patterns for holistic video analysis (~147 features)
2. **Window-Level RF**: Isolated window analysis for temporal insights (22 features × N windows)
3. **Window-Level K-Means**: Distance-based clustering with normalized features (27 features × N windows)

**Processing Model**: Per-bucket transformation with checkpoint/resume
**Performance**: ~7s for 100 videos (bucket 18-33s, reference hardware)

---

# Input Contract

## Prerequisites
- **Stage 3** complete → `ml_analysis/aggregated_features.csv`
- **Stage 3 checkpoint** (or CSV validation fallback)

## Input Files
```
{bucket_path}/
├── ml_analysis/
│   └── aggregated_features.csv               # Stage 3 output (N videos × M features)
└── checkpoints/
    └── stage_3_checkpoint.json               # Optional (fallback to CSV validation)
```

## Input Schema

**aggregated_features.csv** (bucket 18-33s example):

```csv
video_id,hook_average_face_size,hook_overlay_unique_count,...,middle_1_average_face_size,...,closing_average_face_size,...,create_time,gender,xwin_hook_to_middle_energy,xwin_middle_to_closing_energy,xwin_eye_contact_consistency,xwin_word_density_std,xwin_energy_progression_slope,is_top_performer
7428596413707144481,0.42,3,...,0.38,...,0.51,...,1704960000,female,0.12,-0.08,0.15,2.3,0.05,1
```

**Expected Columns** (bucket-specific):
- 0-3s: 25 columns (21×1 + 3 + 0 + 1)
- 3-9s: 49 columns (21×2 + 3 + 3 + 1)
- 9-13s: 72 columns (21×3 + 3 + 5 + 1)
- 13-18s: 72 columns (21×3 + 3 + 5 + 1)
- 18-33s: 135 columns (21×6 + 3 + 5 + 1)
- 33-60s: 156 columns (21×7 + 3 + 5 + 1)
- 60-90s: 156 columns
- 90-120s: 156 columns

## Validation

**File**: `feature_transformation.py::validate_input()` (line 240-327)

```python
# Pre-flight checks:
1. Column count matches bucket expectations (uses config.bucket_definitions.get_stage3_expected_feature_count())
2. All required columns exist (21 features × N windows + 3 metadata)
3. No NaN values (fail-fast)
4. Normalized features in [0-1] range (eye_contact_rate, speech_coverage, energy_level, etc.)
5. Count features non-negative with sanity bounds (<10,000)
6. Minimum row count ≥3 (MINIMUM_VIDEO_COUNT from stage4_constants)
```

**Raises**: `ValueError` with specific error message

---

# Output Contract

## Files Created

**Per Bucket** (13 files for bucket 18-33s):

```
{bucket_path}/ml_analysis/
├── rf_transformed.csv                        # Video-Level RF (N videos × ~147 features)
├── hook_rf_transformed.csv                   # Window-Level RF (N videos × 22 features)
├── middle_1_rf_transformed.csv
├── middle_2_rf_transformed.csv
├── middle_3_rf_transformed.csv
├── middle_4_rf_transformed.csv
├── closing_rf_transformed.csv
├── hook_km_transformed.csv                   # Window-Level K-Means (N videos × 27 features)
├── middle_1_km_transformed.csv
├── middle_2_km_transformed.csv
├── middle_3_km_transformed.csv
├── middle_4_km_transformed.csv
├── closing_km_transformed.csv
├── hook_scalers.pkl                          # MinMaxScaler objects + metadata
├── middle_1_scalers.pkl
├── middle_2_scalers.pkl
├── middle_3_scalers.pkl
├── middle_4_scalers.pkl
└── closing_scalers.pkl

{bucket_path}/checkpoints/
└── stage_4_checkpoint.json                   # Stage completion marker
```

**File Count Formula**: 1 + 3×N, where N = window count
- Bucket 18-33s (6 windows): 1 + 3×6 = 19 files total (13 CSVs + 6 PKLs)
- Bucket 3-9s (2 windows): 1 + 3×2 = 7 files total (5 CSVs + 2 PKLs)

## Output Schemas

### 1. Video-Level RF (rf_transformed.csv)

**Purpose**: Cross-window patterns for holistic video analysis

**Column Structure** (bucket 18-33s, ~147 columns):

```csv
hook_average_face_size,hook_overlay_unique_count,...,middle_1_average_face_size,...,closing_average_face_size,...,joy,sadness,anger,fear,disgust,surprise,neutral,hour,day_of_week,month,is_weekend,is_business_hours,gender_male,gender_female,gender_nan,xwin_hook_to_middle_energy,xwin_middle_to_closing_energy,xwin_eye_contact_consistency,xwin_word_density_std,xwin_energy_progression_slope,is_top_performer
0.42,3,...,0.38,...,0.51,...,0,0,0,0,0,0,1,14,2,1,0,0,0,1,0,0.12,-0.08,0.15,2.3,0.05,1
```

**Features**:
- **Temporal features**: 21 × N windows (126 for 18-33s)
- **Emotion one-hot**: 7 features (joy, sadness, anger, fear, disgust, surprise, neutral)
- **Temporal extract**: 5 features (hour, day_of_week, month, is_weekend, is_business_hours)
- **Gender encoding**: 3 features (gender_male, gender_female, gender_nan)
- **Cross-window features**: 0-5 features (bucket-dependent: 0-3s=0, 3-9s=3, 9-13s+=5)
- **Target**: 1 feature (is_top_performer)

**Total**: 126 + 7 + 5 + 3 + 5 + 1 = **147 columns** (bucket 18-33s)

### 2. Window-Level RF ({window}_rf_transformed.csv)

**Purpose**: Isolated window analysis for temporal insights

**Column Structure** (22 columns, all windows):

```csv
average_face_size,overlay_unique_count,has_captions,scene_count,shortest_scene,longest_scene,scene_duration_variance,object_count,person_count,dominant_emotion_id,speech_coverage,word_count,energy_level,energy_variance,energy_max,pitch_scatter_ratio,gesture_count,gaze_variance,eye_contact_rate,emotional_valence,emotion_consistency,is_top_performer
0.42,3,1,2,0.5,1.2,0.15,5,1,7,0.85,12,0.72,0.08,0.85,0.35,8,0.12,0.68,-0.2,0.75,1
```

**Features**:
- **Window features**: 21 base features (from single window)
- **Target**: 1 feature (is_top_performer)

**Total**: **22 columns** (consistent across all windows)

### 3. Window-Level K-Means ({window}_km_transformed.csv)

**Purpose**: Distance-based clustering with normalized features [0-1]

**Column Structure** (27 columns, all windows):

```csv
scene_count_scaled,word_count_scaled,gesture_count_scaled,object_count_scaled,person_count_scaled,overlay_unique_count_scaled,shortest_scene_scaled,longest_scene_scaled,scene_duration_variance_scaled,energy_variance_scaled,gaze_variance_scaled,average_face_size_scaled,speech_coverage_scaled,energy_level_scaled,energy_max_scaled,pitch_scatter_ratio_scaled,eye_contact_rate_scaled,emotion_consistency_scaled,emotional_valence_scaled,has_captions_encoded,joy,sadness,anger,fear,disgust,surprise,neutral
0.12,0.45,0.68,0.32,0.15,0.22,0.08,0.75,0.18,0.12,0.05,0.42,0.85,0.72,0.85,0.35,0.68,0.75,0.40,1,0,0,0,0,0,0,1
```

**Features**:
- **Log1p + MinMax scaled**: 11 features (scene_count, word_count, gesture_count, object_count, person_count, overlay_unique_count, shortest_scene, longest_scene, scene_duration_variance, energy_variance, gaze_variance)
- **MinMax scaled**: 7 features (average_face_size, speech_coverage, energy_level, energy_max, pitch_scatter_ratio, eye_contact_rate, emotion_consistency)
- **Shift + Scale**: 1 feature (emotional_valence: [-1,1] → [0,1])
- **Label encoded**: 1 feature (has_captions: Boolean → 0/1)
- **One-hot encoded**: 7 features (dominant_emotion_id → joy, sadness, anger, fear, disgust, surprise, neutral)

**Total**: 11 + 7 + 1 + 1 + 7 = **27 columns** (consistent across all windows)

**Range**: All features normalized to [0-1] for distance-based clustering

### 4. Scaler Files ({window}_scalers.pkl)

**Purpose**: Persist fitted MinMaxScaler objects for inference reproducibility

**Schema** (joblib-serialized dict):

```python
{
    'version': '1.0',                         # Format version
    'sklearn_version': '1.3.0',               # Sklearn version for compatibility
    'scalers': {                              # Fitted MinMaxScaler objects
        'scene_count': MinMaxScaler(...),
        'word_count': MinMaxScaler(...),
        # ... up to 18 scalers (features with variance > 0)
    },
    'constant_features': [                    # Features with zero variance
        'overlay_unique_count',               # Cannot fit scaler (max == min)
        'gaze_variance'                       # Scaled to 0.5 (midpoint)
    ]
}
```

**Usage**: Load with `joblib.load()` for inference transformations

---

# Core Functions

## Function Call Tree

```
run_stage4_transformation()  [ENTRY POINT - Line 874]
  ├─ validate_input()  [Line 240]
  │   ├─ get_stage3_expected_feature_count() [config.bucket_definitions]
  │   ├─ get_required_columns() [Line 57]
  │   └─ MINIMUM_VIDEO_COUNT [config.stage4_constants]
  │
  ├─ transform_video_level_rf()  [Line 434]
  │   ├─ get_base_features() [Line 154]
  │   ├─ calculate_window_midpoint_timestamps() [Line 333]
  │   └─ calculate_linear_slope_with_timestamps() [Line 396]
  │
  ├─ transform_window_level_rf()  [Line 546] (loop for each window)
  │   └─ get_base_features() [Line 154]
  │
  ├─ transform_window_level_kmeans()  [Line 603] (loop for each window)
  │   └─ MinMaxScaler() [sklearn.preprocessing]
  │
  ├─ Save scalers with joblib.dump() [Lines 971-1029]
  │
  ├─ validate_outputs_and_checkpoint()  [Line 774]
  │   ├─ get_expected_output_files() [Line 79]
  │   ├─ get_expected_rf_column_count() [Line 123]
  │   ├─ validate_cross_window_features() [Line 735]
  │   └─ write_checkpoint() [Line 102]
  │
  └─ Write CSV files to disk [Lines 1043-1061]
```

---

## 1. run_stage4_transformation()

**File**: `feature_transformation.py:874-1088`
**Purpose**: Main entry point for Stage 4 transformation pipeline

```python
def run_stage4_transformation(
    bucket_path: str,
    config: dict
) -> Tuple[bool, List[str], float]:
```

**Parameters**:
- `bucket_path`: Full path to bucket directory (e.g., `/data/clients/acme/hashtags/fitness/top_contrastive/buckets/bucket_18-33s`)
- `config`: Dict with `strategy` ("contrastive" or "top") and `video_count`

**Returns**: `(success: bool, output_files: List[str], elapsed_time: float)`

**Flow**:
1. Initialize MetricsCollector
2. Extract bucket name from path
3. Load aggregated_features.csv
4. Validate input schema
5. Transform Video-Level RF
6. Transform Window-Level RF (loop for each window)
7. Transform Window-Level K-Means (loop for each window)
8. Save scaler .pkl files with post-save validation
9. Validate all outputs and write checkpoint
10. Write CSV files to disk
11. Log metrics and performance warnings

**Raises**:
- `FileNotFoundError` - Aggregated CSV missing
- `IOError` - CSV parse failure or scaler save failure
- `ValueError` - Input validation failed
- `AssertionError` - Output validation failed

---

## 2. validate_input()

**File**: `feature_transformation.py:240-327`
**Purpose**: Validate aggregated features before transformation (fail-fast)

```python
def validate_input(df: pd.DataFrame, bucket: str, expected_count: int) -> None:
```

**Validation Checks**:

**1. Column Count** (line 254-262):
```python
from config.bucket_definitions import get_stage3_expected_feature_count
expected_cols = get_stage3_expected_feature_count(bucket)  # 135 for 18-33s
if len(df.columns) != expected_cols:
    raise ValueError(f"Expected {expected_cols} columns, found {len(df.columns)}")
```

**2. Required Columns** (line 264-271):
```python
required_cols = get_required_columns(bucket)  # 21 features × N windows + 3 metadata
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Required columns missing: {missing}")
```

**3. NaN Values** (line 273-280):
```python
nan_cols = df.columns[df.isna().any()].tolist()
if nan_cols:
    nan_count = {col: df[col].isna().sum() for col in nan_cols}
    raise ValueError(f"NaN values detected: {nan_count}")
```

**4. Normalized Features Range** (line 282-294):
```python
normalized_features = [
    'eye_contact_rate', 'speech_coverage', 'energy_level', 'energy_max',
    'pitch_scatter_ratio', 'emotion_consistency', 'average_face_size'
]
for col in df.columns:
    if any(feat in col for feat in normalized_features):
        if (df[col] < 0).any() or (df[col] > 1).any():
            raise ValueError(f"{col} out of range [0.0-1.0]")
```

**5. Count Features Sanity** (line 296-311):
```python
count_features = [
    'scene_count', 'word_count', 'gesture_count', 'object_count',
    'person_count', 'overlay_unique_count'
]
for col in df.columns:
    if any(feat in col for feat in count_features):
        if (df[col] < 0).any():
            raise ValueError(f"{col} has negative values")
        if (df[col] > 10000).any():
            raise ValueError(f"{col} suspiciously high (>10000)")
```

**6. Minimum Row Count** (line 313-324):
```python
from config.stage4_constants import MINIMUM_VIDEO_COUNT
if len(df) < MINIMUM_VIDEO_COUNT:  # Default: 3
    raise ValueError(f"{len(df)} videos found, minimum {MINIMUM_VIDEO_COUNT} required")
```

**Raises**: `ValueError` with specific error message

---

## 3. transform_video_level_rf()

**File**: `feature_transformation.py:434-539`
**Purpose**: Transform aggregated features for Video-Level Random Forest

```python
def transform_video_level_rf(
    df: pd.DataFrame,
    bucket: str,
    strategy: str,
    video_count: int,
    bucket_path: str = None
) -> pd.DataFrame:
```

**Transformations** (in order):

**1. Encode has_captions to 0/1** (line 457-462):
```python
window_columns = [col for col in df_rf.columns if 'has_captions' in col]
for col in window_columns:
    df_rf[col] = df_rf[col].astype(int)  # True → 1, False → 0
```
**Rationale**: Prevents quantile errors in Stage 6 distribution analysis

**2. One-hot encode hook_dominant_emotion_id** (line 464-472):
```python
for emotion_id, emotion_name in enumerate(
    ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'],
    start=1
):
    df_rf[emotion_name] = (df_rf['hook_dominant_emotion_id'] == emotion_id).astype(int)
df_rf.drop(columns=['hook_dominant_emotion_id'], inplace=True)
```
**Rationale**: Use hook emotion as video-level emotion (first 3s = most important)

**3. Extract temporal features from create_time** (line 474-483):
```python
df_rf['create_time'] = pd.to_datetime(df_rf['create_time'])
df_rf['hour'] = df_rf['create_time'].dt.hour  # 0-23
df_rf['day_of_week'] = df_rf['create_time'].dt.dayofweek  # 0=Monday, 6=Sunday
df_rf['month'] = df_rf['create_time'].dt.month  # 1-12
df_rf['is_weekend'] = (df_rf['day_of_week'] >= 5).astype(int)  # 1 if Sat/Sun
df_rf['is_business_hours'] = ((df_rf['hour'] >= 9) & (df_rf['hour'] <= 17)).astype(int)
df_rf.drop(columns=['create_time'], inplace=True)
```

**4. Explicit 3-column gender encoding** (line 485-497):
```python
if 'gender' in df_rf.columns:
    df_rf['gender_male'] = (df_rf['gender'] == 'male').astype(int)
    df_rf['gender_female'] = (df_rf['gender'] == 'female').astype(int)
    df_rf['gender_nan'] = df_rf['gender'].isna().astype(int)
    df_rf.drop(columns=['gender'], inplace=True)
else:
    # Graceful degradation if gender column missing
    df_rf['gender_male'] = 0
    df_rf['gender_female'] = 0
    df_rf['gender_nan'] = 0
```
**Fix**: Always create all 3 columns to ensure consistent schema across buckets

**5. Validate is_top_performer exists** (line 499-508):
```python
if 'is_top_performer' not in df.columns:
    raise ValueError("is_top_performer missing from aggregated_features.csv")
# Pass through unchanged (Stage 3 created it)
```

**6. Validate cross-window features exist** (line 510-533):
```python
expected_cross_window = [
    'xwin_hook_to_middle_energy',
    'xwin_middle_to_closing_energy',
    'xwin_eye_contact_consistency',
    'xwin_word_density_std',
    'xwin_energy_progression_slope'
]
existing_cross_window = [f for f in expected_cross_window if f in df.columns]
# Pass through unchanged (Stage 3 created them)
```

**Returns**: DataFrame with ~147 columns (bucket 18-33s)

---

## 4. transform_window_level_rf()

**File**: `feature_transformation.py:546-596`
**Purpose**: Extract single window features for Window-Level Random Forest

```python
def transform_window_level_rf(
    df: pd.DataFrame,
    window_type: str,
    strategy: str,
    video_count: int,
    bucket_path: str = None
) -> pd.DataFrame:
```

**Transformation Logic**:

**1. Extract window-specific columns** (line 568-574):
```python
window_prefix = f'{window_type}_'  # e.g., "hook_"
window_cols = [c for c in df.columns if c.startswith(window_prefix)]
df_window = df[window_cols].copy()

# Remove prefix: hook_scene_count → scene_count
df_window.columns = [c.replace(window_prefix, '') for c in df_window.columns]
```

**2. Add is_top_performer target** (line 576-583):
```python
if 'is_top_performer' not in df.columns:
    raise ValueError("is_top_performer missing from aggregated_features.csv")
df_window['is_top_performer'] = df['is_top_performer'].copy()
```

**3. Encode has_captions** (line 585-588):
```python
if 'has_captions' in df_window.columns:
    df_window['has_captions'] = df_window['has_captions'].astype(int)  # True → 1, False → 0
```

**Note**:
- `dominant_emotion_id` stays ordinal 1-7 (RF handles ordinal natively)
- `emotional_valence` stays continuous [-1,1] (RF handles continuous natively)

**Returns**: DataFrame with **22 columns** (21 features + 1 target)

---

## 5. transform_window_level_kmeans()

**File**: `feature_transformation.py:603-728`
**Purpose**: Transform features for Window-Level K-Means clustering

```python
def transform_window_level_kmeans(
    df: pd.DataFrame,
    window_type: str
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
```

**Transformation Logic**:

**1. Extract window-specific columns** (line 641-648):
```python
window_prefix = f'{window_type}_'
window_cols = [c for c in df.columns if c.startswith(window_prefix)]
df_km = df[window_cols].copy()
df_km.columns = [c.replace(window_prefix, '') for c in df_km.columns]
```

**2. Log1p + MinMax scale (11 features)** (line 656-681):
```python
log_scale_features = [
    'scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count',
    'overlay_unique_count', 'shortest_scene', 'longest_scene',
    'scene_duration_variance', 'energy_variance', 'gaze_variance'
]
for feature in log_scale_features:
    if feature in df_km.columns:
        # Apply log1p (handles zeros)
        df_km[feature] = np.log1p(df_km[feature])

        # Fit scaler
        scaler = MinMaxScaler()
        min_val = df_km[feature].min()
        max_val = df_km[feature].max()

        if max_val > min_val:
            scaler.fit(df_km[[feature]])
            df_km[f'{feature}_scaled'] = scaler.transform(df_km[[feature]]).flatten()
            scaler_result['fitted'][feature] = scaler
        else:
            # Constant feature (zero variance)
            df_km[f'{feature}_scaled'] = 0.5  # Midpoint
            scaler_result['constant'].append(feature)

        df_km.drop(columns=[feature], inplace=True)
```

**3. MinMax scale only (7 features)** (line 683-702):
```python
scale_features = [
    'average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
    'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency'
]
for feature in scale_features:
    if feature in df_km.columns:
        scaler = MinMaxScaler()
        min_val = df_km[feature].min()
        max_val = df_km[feature].max()

        if max_val > min_val:
            scaler.fit(df_km[[feature]])
            df_km[f'{feature}_scaled'] = scaler.transform(df_km[[feature]]).flatten()
            scaler_result['fitted'][feature] = scaler
        else:
            df_km[f'{feature}_scaled'] = 0.5
            scaler_result['constant'].append(feature)

        df_km.drop(columns=[feature], inplace=True)
```

**4. Shift + Scale for emotional_valence** (line 704-708):
```python
if 'emotional_valence' in df_km.columns:
    # Shift [-1,1] → [0,1]: (x + 1) / 2
    df_km['emotional_valence_scaled'] = (df_km['emotional_valence'] + 1) / 2
    df_km.drop(columns=['emotional_valence'], inplace=True)
```

**5. Label encode has_captions** (line 710-713):
```python
if 'has_captions' in df_km.columns:
    df_km['has_captions_encoded'] = df_km['has_captions'].astype(int)
    df_km.drop(columns=['has_captions'], inplace=True)
```

**6. One-hot encode dominant_emotion_id** (line 715-722):
```python
if 'dominant_emotion_id' in df_km.columns:
    for emotion_id, emotion_name in enumerate(
        ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'],
        start=1
    ):
        df_km[emotion_name] = (df_km['dominant_emotion_id'] == emotion_id).astype(int)
    df_km.drop(columns=['dominant_emotion_id'], inplace=True)
```

**Returns**: `(df_km: DataFrame, scaler_result: Dict)`
- `df_km`: 27 columns, all features normalized to [0-1]
- `scaler_result`: `{'fitted': {feature: MinMaxScaler}, 'constant': [feature_list]}`

---

## 6. Helper Functions

### calculate_window_midpoint_timestamps()

**File**: `feature_transformation.py:333-393`
**Purpose**: Calculate midpoint timestamps for each window programmatically

```python
def calculate_window_midpoint_timestamps(bucket: str, windows: list) -> list:
```

**Logic**:
```python
# Parse bucket duration range (e.g., "18-33s" → min=18, max=33)
total_duration = (min_dur + max_dur) / 2  # Use midpoint

timestamps = []
for window in windows:
    if window == 'hook':
        timestamps.append(1.5)  # Midpoint of [0, 3]
    elif window == 'closing':
        timestamps.append(total_duration - 1.5)  # Midpoint of [duration-3, duration]
    elif window.startswith('middle'):
        middle_start = 3.0
        middle_end = total_duration - 3.0
        middle_duration = middle_end - middle_start

        # Count middle segments
        middle_count = len([w for w in windows if w.startswith('middle')])
        segment_duration = middle_duration / middle_count

        # Extract segment index (middle_1 → 1, middle_2 → 2)
        if window == 'middle_aggregate':
            segment_idx = (middle_count + 1) / 2  # Overall middle midpoint
        else:
            segment_idx = int(window.split('_')[1])

        segment_start = middle_start + (segment_idx - 1) * segment_duration
        segment_midpoint = segment_start + (segment_duration / 2)
        timestamps.append(segment_midpoint)

return timestamps
```

**Example** (bucket 18-33s):
- hook: 1.5s
- middle_1: 6.375s (midpoint of [3, 9.75])
- middle_2: 10.125s
- middle_3: 13.875s
- middle_4: 17.625s
- closing: 24.0s (midpoint of [22.5, 25.5])

---

### calculate_linear_slope_with_timestamps()

**File**: `feature_transformation.py:396-431`
**Purpose**: Calculate linear slope using actual window timestamps (not array indices)

```python
def calculate_linear_slope_with_timestamps(
    values: np.ndarray,
    windows: list,
    bucket: str
) -> float:
```

**Logic**:
```python
if len(values) < 2:
    return 0.0

# Calculate timestamps programmatically
timestamps = calculate_window_midpoint_timestamps(bucket, windows)

# Linear regression using timestamps as x-axis
slope, _ = np.polyfit(timestamps, values, 1)
return slope
```

**Fix**: Uses programmatically calculated timestamps instead of hardcoded dict or array indices [0,1,2,...]

---

### get_expected_rf_column_count()

**File**: `feature_transformation.py:123-151`
**Purpose**: Calculate expected Video-Level RF output column count

```python
def get_expected_rf_column_count(bucket: str) -> int:
```

**Formula**:
```python
from config.bucket_definitions import BUCKET_WINDOWS
window_count = len(BUCKET_WINDOWS[bucket])
temporal_features = 21 * window_count

# Cross-window features (bucket-aware)
if bucket == '0-3s':
    cross_window_features = 0
elif bucket == '3-9s':
    cross_window_features = 3
else:
    cross_window_features = 5

# temporal + emotions(7) + temporal_extract(5) + gender(3) + cross_window(0-5) + target(1)
return temporal_features + 7 + 5 + 3 + cross_window_features + 1
```

**Examples**:
- Bucket 0-3s: 21×1 + 7 + 5 + 3 + 0 + 1 = **37 columns**
- Bucket 3-9s: 21×2 + 7 + 5 + 3 + 3 + 1 = **61 columns**
- Bucket 18-33s: 21×6 + 7 + 5 + 3 + 5 + 1 = **147 columns**

---

## 7. MetricsCollector Class

**File**: `feature_transformation.py:177-234`
**Purpose**: Collect and log Stage 4 performance metrics (thread-safe)

**Methods**:

```python
class MetricsCollector:
    def start_stage(self):
        # Start timer and baseline memory

    def record_input(self, video_count: int, column_count: int, file_size_mb: float):
        # Record input metrics

    def record_transformation_time(self, phase: str, elapsed: float):
        # Record transformation phase timing

    def record_output(self, file_count: int, video_rf_cols: int):
        # Record output metrics

    def finalize(self) -> dict:
        # Finalize metrics and log summary
        # Returns metrics dict with stage_4_duration_seconds, peak_memory_mb, etc.
```

**Thread Safety**: Uses `threading.Lock()` for concurrent metric updates

---

# Validation & Checkpoint

## validate_outputs_and_checkpoint()

**File**: `feature_transformation.py:774-867`
**Purpose**: Validate all transformation outputs and write checkpoint

```python
def validate_outputs_and_checkpoint(
    output_files: dict,
    bucket: str,
    video_count: int,
    bucket_base: str
) -> None:
```

**Validation Checks**:

**1. File count** (line 796-799):
```python
expected_files = get_expected_output_files(bucket)  # 13 for bucket 18-33s
missing_files = [f for f in expected_files if f not in output_files]
if missing_files:
    raise AssertionError(f"Missing output files: {missing_files}")
```

**2. Video-Level RF schema** (line 801-813):
```python
df_rf = output_files['rf_transformed.csv']
expected_cols = get_expected_rf_column_count(bucket)
tolerance = 3  # Allow ±3 for gender variations
assert expected_cols - tolerance <= len(df_rf.columns) <= expected_cols + tolerance
assert len(df_rf) == video_count
assert not df_rf.isnull().any().any()

# Validate cross-window features
validate_cross_window_features(df_rf, bucket)
```

**3. Window-Level RF schemas** (line 815-826):
```python
from config.bucket_definitions import BUCKET_WINDOWS
windows = BUCKET_WINDOWS[bucket]

for window in windows:
    df_window_rf = output_files[f'{window}_rf_transformed.csv']
    assert len(df_window_rf.columns) == 22
    assert len(df_window_rf) == video_count
    assert not df_window_rf.isnull().any().any()
```

**4. Window-Level K-Means schemas** (line 828-846):
```python
for window in windows:
    df_window_km = output_files[f'{window}_km_transformed.csv']
    assert len(df_window_km.columns) == 27
    assert len(df_window_km) == video_count
    assert not df_window_km.isnull().any().any()

    # Validate all _scaled columns in [0,1] range
    scaled_cols = [c for c in df_window_km.columns if c.endswith('_scaled')]
    for col in scaled_cols:
        min_val = df_window_km[col].min()
        max_val = df_window_km[col].max()
        assert min_val >= -1e-10 and max_val <= 1 + 1e-10  # Float precision tolerance
```

**5. Write checkpoint** (line 848-862):
```python
checkpoint = {
    "stage": "feature_transformation",
    "status": "completed",
    "total_videos": video_count,
    "output_files": list(output_files.keys()),
    "completion_time": datetime.now().isoformat()
}

try:
    write_checkpoint(checkpoint, bucket_base)
except Exception as e:
    raise IOError(f"Failed to write checkpoint: {e}") from e
```

**Raises**: `AssertionError` or `IOError`

---

## validate_cross_window_features()

**File**: `feature_transformation.py:735-771`
**Purpose**: Validate cross-window feature ranges in Video-Level RF

```python
def validate_cross_window_features(df_rf: pd.DataFrame, bucket: str) -> None:
```

**Range Checks**:

```python
# Delta features bounded by [-1, 1]
if 'hook_to_middle_energy_delta' in df_rf.columns:
    assert df_rf['hook_to_middle_energy_delta'].between(-1, 1).all()

if 'middle_to_closing_delta' in df_rf.columns:
    assert df_rf['middle_to_closing_delta'].between(-1, 1).all()

# Consistency features (std must be non-negative)
if 'eye_contact_consistency' in df_rf.columns:
    assert (df_rf['eye_contact_consistency'] >= 0).all()

if 'word_density_std' in df_rf.columns:
    assert (df_rf['word_density_std'] >= 0).all()

# Slope feature (sanity check: shouldn't be extreme)
if 'energy_progression_slope' in df_rf.columns:
    assert df_rf['energy_progression_slope'].between(-2, 2).all()
```

**Raises**: `AssertionError` with specific message

---

## Scaler Persistence

**File**: `feature_transformation.py:971-1029`
**Purpose**: Save fitted scalers with post-save validation

**Save Logic**:
```python
for window in windows:
    scaler_path = os.path.join(output_dir, f'{window}_scalers.pkl')

    # Create metadata dict
    scaler_metadata = {
        'version': '1.0',
        'sklearn_version': sklearn.__version__,
        'scalers': window_scalers[window]['fitted'],         # Fitted MinMaxScaler objects
        'constant_features': window_scalers[window]['constant']  # Zero variance features
    }

    # Save with joblib
    try:
        joblib.dump(scaler_metadata, scaler_path)
    except Exception as e:
        raise IOError(f"Scaler save failed for {window}: {e}") from e

    # Post-save validation (verify immediately)
    try:
        loaded = joblib.load(scaler_path)
        assert 'version' in loaded
        assert 'sklearn_version' in loaded
        assert 'scalers' in loaded
        assert 'constant_features' in loaded
        assert isinstance(loaded['scalers'], dict)

        scaler_count = len(loaded['scalers'])
        constant_count = len(loaded['constant_features'])
        file_size_kb = os.path.getsize(scaler_path) / 1024
        logger.info(
            f"✓ Saved {window}_scalers.pkl: {scaler_count} fitted scalers, "
            f"{constant_count} constant features, {file_size_kb:.1f} KB"
        )
    except Exception as e:
        raise IOError(f"Scaler validation failed for {window}: {e}") from e
```

**Philosophy**: Fail-fast approach with immediate post-save verification

---

# Data Flow & Architecture

## Complete Pipeline Flow

```
Stage 3 (Feature Aggregation)
    ↓
    aggregated_features.csv (N videos × 135 columns for 18-33s)
    ↓
Stage 4 (Feature Transformation)
    ├─ validate_input()
    │   ├─ Column count check (135 for 18-33s)
    │   ├─ Required columns check
    │   ├─ NaN check
    │   ├─ Range validation
    │   └─ Minimum row count (≥3)
    │
    ├─ transform_video_level_rf()
    │   ├─ Encode has_captions (Boolean → 0/1)
    │   ├─ One-hot hook_dominant_emotion_id (1 → 7 features)
    │   ├─ Extract temporal from create_time (1 → 5 features)
    │   ├─ Encode gender (1 → 3 features)
    │   ├─ Pass through cross-window features (0-5 features)
    │   └─ Pass through is_top_performer (1 feature)
    │   → rf_transformed.csv (~147 columns)
    │
    ├─ For each window:
    │   ├─ transform_window_level_rf()
    │   │   ├─ Extract window features (21)
    │   │   ├─ Encode has_captions (Boolean → 0/1)
    │   │   └─ Add is_top_performer (1)
    │   │   → {window}_rf_transformed.csv (22 columns)
    │   │
    │   └─ transform_window_level_kmeans()
    │       ├─ Log1p + MinMax scale (11 features)
    │       ├─ MinMax scale (7 features)
    │       ├─ Shift + Scale emotional_valence (1 feature)
    │       ├─ Label encode has_captions (1 feature)
    │       └─ One-hot dominant_emotion_id (7 features)
    │       → {window}_km_transformed.csv (27 columns)
    │       → {window}_scalers.pkl (fitted MinMaxScaler objects)
    │
    ├─ validate_outputs_and_checkpoint()
    │   ├─ File count check (13 files)
    │   ├─ Video-Level RF schema (147 cols ±3)
    │   ├─ Window-Level RF schema (22 cols × N windows)
    │   ├─ Window-Level K-Means schema (27 cols × N windows)
    │   ├─ Cross-window feature validation
    │   └─ Write checkpoint
    │
    └─ Write CSV files to disk
        → stage_4_checkpoint.json
```

---

## File Dependency Graph

```
aggregated_features.csv (Stage 3)
    ├─→ Video-Level RF transformation
    │   └─→ rf_transformed.csv
    │
    ├─→ Window-Level RF transformation (per window)
    │   └─→ {window}_rf_transformed.csv (6 files for 18-33s)
    │
    └─→ Window-Level K-Means transformation (per window)
        ├─→ {window}_km_transformed.csv (6 files for 18-33s)
        └─→ {window}_scalers.pkl (6 files for 18-33s)

stage_4_checkpoint.json
    └─→ Orchestrator skip logic (Stage 4 complete marker)

All outputs → Stage 5 (ML Model Training)
```

---

## Transformation Summary Table

| Transformation | Input Columns | Output Columns | Key Operations |
|----------------|---------------|----------------|----------------|
| **Video-Level RF** | 135 (18-33s) | ~147 | One-hot emotions (7), temporal extract (5), gender encode (3), pass-through cross-window (5), pass-through label (1) |
| **Window-Level RF** | 21 (per window) | 22 | Extract window, encode has_captions, add label |
| **Window-Level K-Means** | 21 (per window) | 27 | Log1p+MinMax (11), MinMax (7), Shift+Scale (1), Label encode (1), One-hot (7) |

---

# Error Handling Matrix

## Orchestrator-Level Errors

**File**: `rumiai_ml_batch.py:1375-1432`

| Error Type | Cause | Handled By | Action | Exit Code |
|------------|-------|------------|--------|-----------|
| `ValueError` | Input validation failed | Orchestrator | Skip bucket, continue | - |
| `AssertionError` | Output validation failed | Orchestrator | Skip bucket, continue | - |
| `FileNotFoundError` | Stage 3 CSV missing | Orchestrator | Skip bucket, continue | - |
| `IOError/OSError` | Disk full, permissions | Orchestrator | **Exit pipeline** | 4 |
| `TimeoutError` | Processing >5 minutes | Orchestrator | **Exit pipeline** | 8 |
| `Exception` | Unexpected error | Orchestrator | **Exit pipeline** | 99 |

**Strategy**: Skip bucket-specific errors (continue processing remaining buckets), exit on system-wide issues

---

## Function-Level Errors

### validate_input() Errors

| Check | Failure Condition | Exception | Message |
|-------|-------------------|-----------|---------|
| Column count | len(df.columns) ≠ expected | `ValueError` | Expected {expected} columns, found {actual} |
| Required columns | Missing temporal/metadata columns | `ValueError` | Required columns missing: {list} |
| NaN values | Any column has NaN | `ValueError` | NaN values detected: {col: count} |
| Normalized range | Feature value outside [0-1] | `ValueError` | {col} out of range [0.0-1.0] |
| Count sanity | Negative or >10,000 | `ValueError` | {col} has negative values / suspiciously high |
| Minimum rows | len(df) < MINIMUM_VIDEO_COUNT | `ValueError` | {len(df)} videos found, minimum {min} required |

---

### transform_video_level_rf() Errors

| Check | Failure Condition | Exception | Message |
|-------|-------------------|-----------|---------|
| is_top_performer | Column missing | `ValueError` | is_top_performer missing from aggregated_features.csv |

---

### transform_window_level_rf() Errors

| Check | Failure Condition | Exception | Message |
|-------|-------------------|-----------|---------|
| is_top_performer | Column missing | `ValueError` | is_top_performer missing from aggregated_features.csv |

---

### validate_outputs_and_checkpoint() Errors

| Check | Failure Condition | Exception | Message |
|-------|-------------------|-----------|---------|
| File count | Missing output files | `AssertionError` | Missing output files: {list} |
| Video-Level RF cols | Column count outside expected ±3 | `AssertionError` | Video-Level RF has {actual} columns, expected {expected} ±3 |
| Video-Level RF rows | Row count ≠ video_count | `AssertionError` | Video-Level RF has {actual} rows, expected {video_count} |
| Video-Level RF NaN | Any NaN values | `AssertionError` | Video-Level RF contains NaN values |
| Window-Level RF cols | Column count ≠ 22 | `AssertionError` | {window} RF has {actual} columns, expected 22 |
| Window-Level RF rows | Row count ≠ video_count | `AssertionError` | {window} RF has {actual} rows, expected {video_count} |
| Window-Level RF NaN | Any NaN values | `AssertionError` | {window} RF contains NaN values |
| Window-Level K-Means cols | Column count ≠ 27 | `AssertionError` | {window} K-Means has {actual} columns, expected 27 |
| Window-Level K-Means rows | Row count ≠ video_count | `AssertionError` | {window} K-Means has {actual} rows, expected {video_count} |
| Window-Level K-Means NaN | Any NaN values | `AssertionError` | {window} K-Means contains NaN values |
| K-Means range | _scaled column outside [0,1] | `AssertionError` | {window} K-Means column {col} outside [0,1] |
| Checkpoint write | File write failure | `IOError` | Failed to write checkpoint: {error} |

---

### Scaler Persistence Errors

| Check | Failure Condition | Exception | Message |
|-------|-------------------|-----------|---------|
| Scaler save | joblib.dump() failure | `IOError` | Scaler save failed for {window}: {error} |
| Post-save validation | joblib.load() failure or missing keys | `IOError` | Scaler validation failed for {window}: {error} |

---

## Performance Warnings

**File**: `feature_transformation.py:1080-1086`

```python
from config.stage4_constants import WARNING_TIME_MULTIPLIER, BASELINE_TIME_SECONDS
warning_threshold = BASELINE_TIME_SECONDS * WARNING_TIME_MULTIPLIER  # 7.2s × 2.0 = 14.4s

if elapsed > warning_threshold:
    logger.warning(f"Stage 4 exceeded warning threshold: {elapsed:.1f}s > {warning_threshold:.1f}s")
```

**Thresholds** (from `stage4_constants.py`):
- `BASELINE_TIME_SECONDS`: 7.2s (reference: N=100, bucket 18-33s)
- `WARNING_TIME_MULTIPLIER`: 2.0 (warn if >14.4s)
- `TIMEOUT_MULTIPLIER`: 10.0 (fail if >72s)
- `MAX_TIMEOUT_SECONDS`: 300s (absolute maximum 5 minutes)

---

# Debugging Guide

## Common Issues

### Issue: Column count mismatch

**Symptom**: `ValueError: Expected 135 columns for bucket 18-33s, found 129`

**Cause**: Stage 3 output missing cross-window features or is_top_performer

**Debug**:
```bash
# Check actual columns
head -1 ml_analysis/aggregated_features.csv | tr ',' '\n' | wc -l

# Check for cross-window features
head -1 ml_analysis/aggregated_features.csv | grep -o "xwin_" | wc -l
# Should return 5 for buckets 9-13s+

# Check for is_top_performer
head -1 ml_analysis/aggregated_features.csv | grep -o "is_top_performer"
# Should return "is_top_performer"
```

**Fix**: Re-run Stage 3 with S7B2 fix (cross-window features + is_top_performer)

---

### Issue: NaN values in input

**Symptom**: `ValueError: NaN values detected: {'hook_scene_count': 5}`

**Cause**: Stage 2 or Stage 3 produced null values

**Debug**:
```bash
# Check for NaN in specific column
python3 -c "
import pandas as pd
df = pd.read_csv('ml_analysis/aggregated_features.csv')
nan_cols = df.columns[df.isna().any()].tolist()
for col in nan_cols:
    print(f'{col}: {df[col].isna().sum()} NaN values')
"
```

**Fix**:
1. Identify source stage (Stage 2 extraction vs Stage 3 aggregation)
2. Check temporal_windows JSON for null values
3. Verify feature extraction logic

---

### Issue: Output validation failed (Video-Level RF column count)

**Symptom**: `AssertionError: Video-Level RF has 144 columns, expected 147 ±3`

**Cause**: Transformation logic dropped or added unexpected columns

**Debug**:
```bash
# Check actual output columns
head -1 ml_analysis/rf_transformed.csv | tr ',' '\n' | wc -l

# Compare expected vs actual
python3 -c "
from rumiai_v2.processors.feature_transformation import get_expected_rf_column_count
print(f'Expected: {get_expected_rf_column_count(\"18-33s\")}')
"

# Identify missing columns (compare with expected formula)
# Expected: 126 temporal + 7 emotion + 5 temporal_extract + 3 gender + 5 cross_window + 1 target = 147
```

**Fix**:
1. Verify emotion one-hot created 7 columns (line 464-472)
2. Verify temporal extract created 5 columns (line 474-483)
3. Verify gender encoding created 3 columns (line 485-497)
4. Verify cross-window features passed through (line 510-533)

---

### Issue: Window-Level K-Means values outside [0,1]

**Symptom**: `AssertionError: hook K-Means column scene_count_scaled outside [0,1]: -0.01-1.02`

**Cause**: MinMaxScaler floating point precision or incorrect scaling

**Debug**:
```python
import pandas as pd
df = pd.read_csv('ml_analysis/hook_km_transformed.csv')

# Check all _scaled columns
scaled_cols = [c for c in df.columns if c.endswith('_scaled')]
for col in scaled_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    print(f'{col}: [{min_val:.10f}, {max_val:.10f}]')
```

**Fix**:
- If values like -1e-10 or 1.00000001 → Float precision issue (acceptable)
- If values like -0.5 or 1.5 → Scaling bug (check MinMaxScaler fit/transform)

---

### Issue: Scaler save failed

**Symptom**: `IOError: Scaler save failed for hook: [Errno 28] No space left on device`

**Cause**: Disk full

**Debug**:
```bash
# Check disk space
df -h | grep "ml_analysis"

# Check scaler file sizes
du -sh ml_analysis/*.pkl
```

**Fix**: Free up disk space or change output directory

---

### Issue: Post-save scaler validation failed

**Symptom**: `IOError: Scaler validation failed for hook: hook_scalers.pkl missing scalers`

**Cause**: Corrupted .pkl file or joblib serialization issue

**Debug**:
```python
import joblib

try:
    loaded = joblib.load('ml_analysis/hook_scalers.pkl')
    print(f"Keys: {loaded.keys()}")
    print(f"Scalers count: {len(loaded['scalers'])}")
    print(f"Constant features: {loaded['constant_features']}")
except Exception as e:
    print(f"Load failed: {e}")
```

**Fix**:
1. Delete corrupted .pkl file
2. Delete entire `ml_analysis/` directory
3. Re-run Stage 4

---

### Issue: is_top_performer all 1s or all 0s

**Symptom**: No variation in target variable

**Cause**: Stage 3 didn't create proper labels (contrastive mode)

**Debug**:
```bash
# Check is_top_performer distribution
python3 -c "
import pandas as pd
df = pd.read_csv('ml_analysis/aggregated_features.csv')
print(df['is_top_performer'].value_counts())
"
```

**Fix**: Re-run Stage 3 with proper strategy parameter

---

### Issue: Stage 4 timeout (>5 minutes)

**Symptom**: `TimeoutError: Stage 4 timeout for bucket 18-33s`

**Cause**: System overload, large video count, or pathological bucket

**Debug**:
```bash
# Check video count
wc -l ml_analysis/aggregated_features.csv

# Check system load
top -n 1 | grep "Cpu(s)"

# Check memory
free -h
```

**Fix**:
1. Reduce `--video-count` parameter
2. Close other processes
3. Check for infinite loops in transformation logic

---

## Performance Profiling

### Timing Breakdown

```bash
# Check logs for phase timing
grep "METRIC.*duration_seconds" logs/rumiai_*.log

# Expected breakdown (N=100, bucket 18-33s):
# - video_rf_duration_seconds: ~1.5s
# - window_rf_duration_seconds: ~1.2s
# - window_km_duration_seconds: ~2.5s
# - scaler_save_duration_seconds: ~0.5s
# - file_io_duration_seconds: ~1.0s
# TOTAL: ~7.2s
```

### Memory Usage

```bash
# Check peak memory
grep "METRIC: peak_memory_mb" logs/rumiai_*.log

# Expected: ~500-1000 MB for N=100 videos
```

---

## Quick Debugging Commands

```bash
# Check Stage 4 checkpoint exists
ls {bucket_path}/checkpoints/stage_4_checkpoint.json

# Check all 13 output files exist (bucket 18-33s)
ls ml_analysis/*.csv ml_analysis/*.pkl | wc -l
# Should return 19 (13 CSVs + 6 PKLs)

# Check Video-Level RF column count
head -1 ml_analysis/rf_transformed.csv | tr ',' '\n' | wc -l

# Check Window-Level RF column count
head -1 ml_analysis/hook_rf_transformed.csv | tr ',' '\n' | wc -l
# Should return 22

# Check Window-Level K-Means column count
head -1 ml_analysis/hook_km_transformed.csv | tr ',' '\n' | wc -l
# Should return 27

# Check for NaN in Video-Level RF
python3 -c "
import pandas as pd
df = pd.read_csv('ml_analysis/rf_transformed.csv')
print(f'NaN count: {df.isnull().sum().sum()}')
"

# Check scaler file integrity
python3 -c "
import joblib
loaded = joblib.load('ml_analysis/hook_scalers.pkl')
print(f'Fitted scalers: {len(loaded[\"scalers\"])}')
print(f'Constant features: {len(loaded[\"constant_features\"])}')
"

# Check K-Means range (all _scaled columns should be [0,1])
python3 -c "
import pandas as pd
df = pd.read_csv('ml_analysis/hook_km_transformed.csv')
scaled_cols = [c for c in df.columns if c.endswith('_scaled')]
for col in scaled_cols:
    print(f'{col}: [{df[col].min():.2f}, {df[col].max():.2f}]')
"

# Check checkpoint status
jq '.status' checkpoints/stage_4_checkpoint.json
# Should return "completed"

# Check output file list in checkpoint
jq '.output_files | length' checkpoints/stage_4_checkpoint.json
# Should return 19 (13 CSVs + 6 PKLs)
```

---

# Modification Guide

## Adding a New Base Feature

**Scenario**: Add "audio_clarity" as 22nd base feature to all transformations

**Steps**:

### 1. Update Stage 3 (prerequisite)
See STAGE_3_IMPL.md modification guide for adding base features

### 2. Update get_base_features()

**File**: `feature_transformation.py:154-170`

```python
def get_base_features() -> List[str]:
    return [
        'average_face_size', 'overlay_unique_count', 'scene_count', 'shortest_scene',
        'longest_scene', 'scene_duration_variance', 'object_count', 'person_count',
        'eye_contact_rate', 'gaze_variance', 'gesture_count', 'speech_coverage',
        'word_count', 'energy_level', 'energy_variance', 'energy_max',
        'pitch_scatter_ratio', 'dominant_emotion_id', 'emotional_valence',
        'emotion_consistency', 'has_captions',
        'audio_clarity'  # ADD
    ]
```

### 3. Update stage4_constants.py

**File**: `config/stage4_constants.py:17-30`

Add to appropriate transformation category:

```python
SCALE_FEATURES = [
    'average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
    'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency',
    'audio_clarity'  # ADD (if normalized [0-1])
]
```

Or:

```python
LOG_SCALE_FEATURES = [
    'scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count',
    'overlay_unique_count', 'shortest_scene', 'longest_scene',
    'scene_duration_variance', 'energy_variance', 'gaze_variance',
    'audio_clarity'  # ADD (if count/variance feature)
]
```

### 4. Update expected column counts

**File**: `feature_transformation.py:123-151`

Update get_expected_rf_column_count() formula:

```python
window_count = len(BUCKET_WINDOWS[bucket])
temporal_features = 22 * window_count  # CHANGE from 21
```

Window-Level RF: 22 → 23 columns
Window-Level K-Means: 27 → 28 columns

### 5. Test

```bash
# Re-run Stage 4
python3 rumiai_ml_batch.py --client test --target "#nutrition" --start-stage 4

# Verify new feature in Video-Level RF
head -1 ml_analysis/rf_transformed.csv | tr ',' '\n' | grep "audio_clarity"

# Verify Window-Level RF has 23 columns
head -1 ml_analysis/hook_rf_transformed.csv | tr ',' '\n' | wc -l

# Verify Window-Level K-Means has 28 columns
head -1 ml_analysis/hook_km_transformed.csv | tr ',' '\n' | wc -l
```

---

## Changing K-Means Scaling Strategy

**Scenario**: Change "gesture_count" from Log1p+MinMax to MinMax only

**Steps**:

### 1. Move feature from LOG_SCALE_FEATURES to SCALE_FEATURES

**File**: `feature_transformation.py:656-702`

```python
# Before:
log_scale_features = [
    'scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count',
    # ...
]

# After:
log_scale_features = [
    'scene_count', 'word_count', 'object_count', 'person_count',
    # ... (removed gesture_count)
]

scale_features = [
    'average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
    'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency',
    'gesture_count'  # ADD
]
```

### 2. Update stage4_constants.py (for consistency)

**File**: `config/stage4_constants.py:33-42`

```python
LOG_SCALE_FEATURES = [
    'scene_count', 'word_count', 'object_count', 'person_count',
    'overlay_unique_count', 'shortest_scene', 'longest_scene',
    'scene_duration_variance', 'energy_variance', 'gaze_variance'
    # Removed 'gesture_count'
]

SCALE_FEATURES = [
    'average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
    'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency',
    'gesture_count'  # ADD
]
```

### 3. Test with known data

```bash
# Create test with known gesture_count values
# Before: gesture_count = [2, 4, 6] → log1p([2,4,6]) = [1.1, 1.6, 1.9] → MinMax
# After: gesture_count = [2, 4, 6] → MinMax directly

python3 -c "
import pandas as pd
df = pd.read_csv('ml_analysis/hook_km_transformed.csv')
print(df['gesture_count_scaled'].describe())
"
```

---

## Adding a New Cross-Window Feature to Video-Level RF

**Scenario**: Add "xwin_scene_complexity_variance" (variance of scene_count across windows)

**Steps**:

### 1. Update Stage 3 (prerequisite)
See STAGE_3_IMPL.md modification guide for adding cross-window features

### 2. Update transform_video_level_rf()

**File**: `feature_transformation.py:510-533`

Add to expected cross-window features list:

```python
expected_cross_window = [
    'xwin_hook_to_middle_energy',
    'xwin_middle_to_closing_energy',
    'xwin_eye_contact_consistency',
    'xwin_word_density_std',
    'xwin_energy_progression_slope',
    'xwin_scene_complexity_variance'  # ADD
]
```

### 3. Update get_expected_rf_column_count()

**File**: `feature_transformation.py:140-147`

Update cross-window count:

```python
if bucket == '0-3s':
    cross_window_features = 0  # No change
elif bucket == '3-9s':
    cross_window_features = 4  # CHANGE from 3
else:
    cross_window_features = 6  # CHANGE from 5
```

### 4. Update validate_cross_window_features()

**File**: `feature_transformation.py:735-771`

Add validation for new feature:

```python
# Validate variance feature (must be non-negative)
if 'xwin_scene_complexity_variance' in df_rf.columns:
    assert (df_rf['xwin_scene_complexity_variance'] >= 0).all(), \
        f"xwin_scene_complexity_variance has negative values"
```

### 5. Test

```bash
# Re-run Stage 3 and Stage 4
python3 rumiai_ml_batch.py --client test --target "#nutrition" --start-stage 3

# Verify new feature exists in Video-Level RF
head -1 ml_analysis/rf_transformed.csv | tr ',' '\n' | grep "xwin_scene_complexity_variance"

# Verify column count increased
head -1 ml_analysis/rf_transformed.csv | tr ',' '\n' | wc -l
# Bucket 18-33s: 147 → 148
# Bucket 3-9s: 61 → 62
```

---

## Customizing Gender Encoding

**Scenario**: Change from 3-column one-hot to single ordinal encoding (male=0, female=1, unknown=2)

**Steps**:

### 1. Update transform_video_level_rf()

**File**: `feature_transformation.py:485-497`

```python
# Before (3 columns):
if 'gender' in df_rf.columns:
    df_rf['gender_male'] = (df_rf['gender'] == 'male').astype(int)
    df_rf['gender_female'] = (df_rf['gender'] == 'female').astype(int)
    df_rf['gender_nan'] = df_rf['gender'].isna().astype(int)
    df_rf.drop(columns=['gender'], inplace=True)

# After (1 column):
if 'gender' in df_rf.columns:
    df_rf['gender_encoded'] = df_rf['gender'].map({
        'male': 0,
        'female': 1
    }).fillna(2).astype(int)  # Unknown = 2
    df_rf.drop(columns=['gender'], inplace=True)
else:
    df_rf['gender_encoded'] = 2  # Default to unknown
```

### 2. Update get_expected_rf_column_count()

**File**: `feature_transformation.py:149`

```python
# temporal + emotions(7) + temporal_extract(5) + gender(1) + cross_window(0-5) + target(1)
return temporal_features + 7 + 5 + 1 + cross_window_features + 1  # CHANGE from +3
```

### 3. Update validate_outputs_and_checkpoint()

**File**: `feature_transformation.py:804`

Reduce tolerance for schema check:

```python
tolerance = 1  # CHANGE from 3 (since gender is now fixed 1 column)
```

### 4. Test

```bash
# Re-run Stage 4
python3 rumiai_ml_batch.py --client test --target "#nutrition" --start-stage 4

# Verify gender encoding
python3 -c "
import pandas as pd
df = pd.read_csv('ml_analysis/rf_transformed.csv')
print(df['gender_encoded'].value_counts())
# Should show: 0 (male), 1 (female), 2 (unknown)
"

# Verify column count decreased
head -1 ml_analysis/rf_transformed.csv | tr ',' '\n' | wc -l
# Bucket 18-33s: 147 → 145 (reduced by 2)
```

---

## Related Documentation

- **PRODUCTION_FLOW.md**: [Stage 4 Contract](../../PRODUCTION_FLOW.md#stage-4-feature-transformation)
- **Technical Specs**:
  - `FeatureTransformationTI.md` - Original specification
  - `S7B2.md` - Cross-window features fix (Stage 3 integration)
- **Upstream Stages**:
  - [STAGE_3_IMPL.md](STAGE_3_IMPL.md) - Feature Aggregation (produces aggregated_features.csv)
- **Downstream Stages**:
  - [STAGE_5_IMPL.md](STAGE_5_IMPL.md) - ML Model Training (consumes transformed CSVs)
  - [STAGE_6_IMPL.md](STAGE_6_IMPL.md) - ML Analysis (uses transformed CSVs for predictions)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-28
**Source**: 100% systematic code reading (1,569 production lines across 3 modules)
**Maintainer**: Update when Stage 4 implementation changes
