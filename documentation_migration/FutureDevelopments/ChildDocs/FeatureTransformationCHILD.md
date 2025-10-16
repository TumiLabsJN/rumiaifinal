# Feature Transformation - High-Level Design

> **Parent**: MLPlanningv2.md - Stage 4: Feature Transformation (Lines 1360-1586)
> **Version**: 1.1
> **Last Updated**: 2025-01-28
> **Status**: Draft

---

## 1. Context & Business Goal

### 1.1 What Problem Does This Solve?

Stage 3 (Feature Aggregation) produces a single CSV with raw temporal window features (hook, middle segments, closing). However, Random Forest and K-Means ML models require different feature formats. Random Forest works with mixed data types (categoricals, temporals) but K-Means requires all numerical features scaled to [0-1] for distance-based clustering. This stage transforms the aggregated features into three distinct formats optimized for the dual Random Forest + window-level K-Means architecture, ensuring ML models receive properly preprocessed data.

### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This stage depends on FoundationCHILD.md for:
- Client directory structure (Section 2: Client Architecture & Storage - path templates and architecture)
- Configuration patterns (Section 4: CLI Command Structure - CLI parameters)
- Checkpoint-based orchestration (Section 1: System Goals & Success Criteria - sequential bucket processing)

```
Stage 3: Feature Aggregation
   ↓ Output: ml_analysis/aggregated_features.csv (N videos, 65-215 features depending on bucket)
Stage 4: Feature Transformation (THIS COMPONENT)
   ↓ Output: 13 transformation files per bucket
      - 1 Video-Level RF file (~178 features)
      - 6 Window-Level RF files (22 features each)
      - 6 Window-Level K-Means files (39 features each)
Stage 5: ML Model Training
```

### 1.3 Success Criteria

- [ ] Process N=100 videos (bucket 18-33s) in < 30 seconds (target) / < 5 minutes (timeout)
- [ ] Peak memory usage < 500 MB (warn at 1 GB, fail at 2 GB)
- [ ] Generate all 13 transformation files with correct schemas (no missing columns)
- [ ] All K-Means scaled columns are in [0-1] range (validated before save)
- [ ] Fail-fast on invalid input data (NaN, missing columns, out-of-range values)
- [ ] No rows dropped during transformation (input N = output N)

---

## 2. Architecture & Design

### 2.1 High-Level Approach

We implement a triple transformation pipeline to support dual Random Forest + window-level K-Means architecture. Video-Level RF receives all temporal windows together (hook_*, middle_*, closing_*) with minimal preprocessing (one-hot encoding for categoricals, temporal extraction from timestamps). Window-Level RF receives raw features per window (21 base features + target) for isolated per-window classification. Window-Level K-Means receives heavily preprocessed features (log transforms for counts, MinMax scaling for rates, cyclical encoding for timestamps, one-hot for categoricals) to ensure all features are numerical and scaled to [0-1] for distance-based clustering.

This architecture choice (approved in Phase 1 Critique) enables both cross-window pattern detection (Video-Level RF) and within-window creative strategy discovery (Window-Level K-Means), with Window-Level RF providing validation that K-Means cluster features are actually predictive.

### 2.2 Data Flow

```
Input: ml_analysis/aggregated_features.csv
       Schema: (N videos, 129 columns for bucket 18-33s)
       Location: /data/clients/{client_id}/hashtags/{cluster_id}/{mode}_{strategy}/buckets/bucket_18-33s/ml_analysis/
   ↓
Validation: Check schema, NaN values, ranges, row count (fail-fast if invalid)
   ↓
Pipeline 1: Video-Level RF Transformation
   - One-hot encode: has_captions → 2 features, dominant_emotion_id → 7 features
   - Extract temporal: create_time → hour, day_of_week, month, is_weekend, is_business_hours (5 features)
   - Extract gender: gender_detection → gender_male, gender_female (2-3 features)
   - Add target: is_top_performer (1 feature, contrastive only)
   - Output: rf_transformed.csv (~178 features)
   ↓
Pipeline 2: Window-Level RF Transformation (6 iterations for bucket 18-33s)
   - For each window (hook, middle_1-4, closing):
     - Extract window features: {window}_{feature} → {feature} (remove prefix)
     - Add target: is_top_performer (1 feature)
     - Output: {window}_rf_transformed.csv (22 features)
   ↓
Pipeline 3: Window-Level K-Means Transformation (6 iterations)
   - For each window (hook, middle_1-4, closing):
     - Log + scale (11 features): scene_count, word_count, gesture_count, object_count, person_count,
       overlay_unique_count, shortest_scene, longest_scene, scene_duration_variance, energy_variance, gaze_variance
       → {feature}_log, {feature}_scaled (22 output columns)
     - Scale [0-1] (7 features): average_face_size, speech_coverage, energy_level, energy_max,
       pitch_scatter_ratio, eye_contact_rate, emotion_consistency
       → {feature}_scaled (7 output columns)
     - Shift + scale (1 feature): emotional_valence [-1,1] → [0,1]
       → emotional_valence_scaled (1 output column)
     - Label encode (1 feature): has_captions True/False → 0/1
       → has_captions_encoded (1 output column)
     - One-hot (1 feature): dominant_emotion_id 1-7 → 7 binary columns
       → joy, sadness, anger, fear, disgust, surprise, neutral (7 output columns)
     - Drop original features, keep only transformed
     - Output: {window}_km_transformed.csv (39 features)
   ↓
Output Validation: Check column counts, scaled ranges [0-1], no NaNs introduced
   ↓
Checkpoint: Write {"stage": "feature_transformation", "status": "completed"}
```

### 2.3 Detailed Process

#### Step 2.3.1: Input Validation

**Purpose**: Fail-fast if aggregated_features.csv has invalid schema, missing data, or corrupted values

**Logic**:
```python
def validate_input(df, bucket, expected_count):
    """
    Validate aggregated features CSV before transformation.
    Source: QA Q4

    Args:
        df: pandas DataFrame from aggregated_features.csv
        bucket: str, bucket name (e.g., "18-33s")
        expected_count: int, expected number of videos from config

    Raises:
        ValueError: if validation fails with specific error message
    """
    # 1. Check column count matches bucket expectations
    expected_cols = get_expected_column_count(bucket)  # 129 for 18-33s
    if len(df.columns) != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns for bucket {bucket}, found {len(df.columns)}")

    # 2. Check all required columns exist
    required_cols = get_required_columns(bucket)  # All 21 base features × windows + metadata
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}. Expected 126 temporal columns (21 features × 6 windows) + 3 metadata.")

    # 3. Check for NaN values (fail-fast - from Q4, Stage 2 fail-fast ensures complete features)
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        nan_count = {col: df[col].isna().sum() for col in nan_cols}
        raise ValueError(f"Invalid input: NaN values detected: {nan_count}. Check Stage 3 aggregation logic.")

    # 4. Validate normalized features are in [0-1] range
    normalized_features = ['eye_contact_rate', 'speech_coverage', 'energy_level', 'energy_max',
                          'pitch_scatter_ratio', 'emotion_consistency', 'average_face_size']
    for col in df.columns:
        if any(feat in col for feat in normalized_features):
            if (df[col] < 0).any() or (df[col] > 1).any():
                invalid_rows = df[(df[col] < 0) | (df[col] > 1)]
                raise ValueError(f"Out of range: {col} has value {invalid_rows[col].max()}, expected [0.0-1.0]. Check Stage 2 calculation.")

    # 5. Validate count features are non-negative with sanity bounds
    count_features = ['scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count']
    for col in df.columns:
        if any(feat in col for feat in count_features):
            if (df[col] < 0).any():
                raise ValueError(f"Out of range: {col} has negative values. Check Stage 2 calculation.")
            if (df[col] > 10000).any():
                raise ValueError(f"Out of range: {col} has suspiciously high values (>10000). Check Stage 2 calculation.")

    # 6. Check minimum row count (N >= 50 required, warn if less than expected)
    if len(df) < 50:
        raise ValueError(f"Insufficient data: {len(df)} videos found, minimum 50 required for ML training.")
    if len(df) < expected_count:
        logger.warning(f"Warning: Expected {expected_count} videos, found {len(df)}. Proceeding with reduced sample size.")

    logger.info(f"Input validation passed: {len(df)} videos, {len(df.columns)} columns")
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Empty CSV (N=0) | Fail-fast with error | No videos to process - pipeline cannot continue |
| Below minimum (N=45) | Fail-fast with error | Insufficient data for reliable ML training (minimum 50) |
| Missing column | Fail-fast with error | Contract violation from Stage 3 |
| NaN in required field | Fail-fast with error | Data corruption from Stage 3 (Stage 2 fail-fast should prevent this) |
| Out-of-range value (e.g., eye_contact_rate=1.5) | Fail-fast with error | Bug in Stage 2 feature calculation |

#### Step 2.3.2: Video-Level RF Transformation

**Purpose**: Create single CSV with all temporal windows + derived features for cross-window Random Forest

**Logic**:
```python
def transform_video_level_rf(df, strategy, video_count):
    """
    Transform aggregated features for Video-Level Random Forest.
    Source: QA Q2a

    Args:
        df: pandas DataFrame from aggregated_features.csv
        strategy: str, "contrastive" or other (affects target variable)
        video_count: int, expected videos for target labeling

    Returns:
        pandas DataFrame with ~178 features for bucket 18-33s
    """
    df_rf = df.copy()

    # 1. One-hot encode has_captions (Boolean → 2 features)
    # Direct use - RF handles Boolean as-is, but one-hot more explicit
    df_rf['no_captions'] = (~df_rf['has_captions']).astype(int)
    df_rf['has_captions'] = df_rf['has_captions'].astype(int)

    # 2. One-hot encode dominant_emotion_id (Categorical 1-7 → 7 features)
    # Create binary column for each emotion category
    for emotion_id, emotion_name in enumerate(['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'], start=1):
        df_rf[emotion_name] = (df_rf['dominant_emotion_id'] == emotion_id).astype(int)
    df_rf.drop(columns=['dominant_emotion_id'], inplace=True)

    # 3. Extract temporal features from create_time (ISO 8601 → 5 features)
    df_rf['hour'] = df_rf['create_time'].dt.hour  # 0-23
    df_rf['day_of_week'] = df_rf['create_time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_rf['month'] = df_rf['create_time'].dt.month  # 1-12
    df_rf['is_weekend'] = (df_rf['day_of_week'] >= 5).astype(int)  # 1 if Sat/Sun
    df_rf['is_business_hours'] = ((df_rf['hour'] >= 9) & (df_rf['hour'] <= 17)).astype(int)  # 1 if 9am-5pm
    df_rf.drop(columns=['create_time'], inplace=True)

    # 4. One-hot encode gender (String → 2-3 features)
    if 'gender' in df_rf.columns:
        # Direct one-hot encoding (gender is already a simple string: 'male', 'female', or null)
        df_rf = pd.get_dummies(df_rf, columns=['gender'], prefix='gender', dummy_na=True)

    # 5. Add target variable is_top_performer (contrastive strategy only)
    if strategy == 'contrastive':
        top_count = int(video_count * 0.8)
        df_rf['is_top_performer'] = (df_rf.index < top_count).astype(int)

    # 6.5. Compute Cross-Window Delta Features (NEW)
    # Purpose: Create explicit temporal progression features for Video-Level RF
    # Source: Crosswindowupgrade.md Section 2.2

    # Energy progression deltas
    middle_energy_cols = [f'middle_{i}_energy_level' for i in range(1, len(BUCKET_WINDOWS[bucket])-1)]
    if middle_energy_cols:  # Only if middle segments exist
        df_rf['hook_to_middle_energy_delta'] = (
            df_rf[middle_energy_cols].mean(axis=1) - df_rf['hook_energy_level']
        )
        df_rf['middle_to_closing_contrast'] = (
            df_rf['closing_energy_level'] - df_rf[middle_energy_cols].mean(axis=1)
        )
    else:
        # For buckets 0-3s, 3-9s (no middle segments) - set to neutral value
        df_rf['hook_to_middle_energy_delta'] = 0.0
        df_rf['middle_to_closing_contrast'] = 0.0

    # Consistency metrics (std deviation across all windows)
    eye_contact_cols = [f'{w}_eye_contact_rate' for w in BUCKET_WINDOWS[bucket]]
    df_rf['eye_contact_consistency'] = df_rf[eye_contact_cols].std(axis=1)

    word_count_cols = [f'{w}_word_count' for w in BUCKET_WINDOWS[bucket]]
    df_rf['word_density_std'] = df_rf[word_count_cols].std(axis=1)

    # Progression slopes (linear regression across windows)
    energy_cols = [f'{w}_energy_level' for w in BUCKET_WINDOWS[bucket]]
    df_rf['energy_progression_slope'] = df_rf[energy_cols].apply(
        lambda row: calculate_linear_slope(row.values), axis=1
    )

    # 7. Keep all other features as-is (Direct transform for 17 features)
    # emotional_valence, emotion_consistency, and all temporal window features unchanged

    logger.info(f"Video-Level RF transformation complete: {len(df_rf)} rows, {len(df_rf.columns)} columns")
    return df_rf
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Missing gender (null) | Create gender_nan=1 column via dummy_na=True | Gender is optional metadata |
| create_time parse error | Fail-fast with error | Invalid timestamp from Stage 3 |
| Unknown emotion_id (not 1-7) | Fail-fast with error | Invalid data from Stage 2 |
| Duplicate one-hot columns | Drop duplicates, keep first | Pandas get_dummies safety |

#### Step 2.3.3: Window-Level RF Transformation

**Purpose**: Extract per-window features (21 base + target) for isolated window classification

**Logic**:
```python
def transform_window_level_rf(df, window_type, strategy, video_count):
    """
    Transform aggregated features for Window-Level Random Forest (one window type).
    Source: QA Q2b

    Args:
        df: pandas DataFrame from aggregated_features.csv
        window_type: str, e.g., "hook", "middle_1", "closing"
        strategy: str, "contrastive" or other
        video_count: int, for target labeling

    Returns:
        pandas DataFrame with 22 features (21 base + 1 target)
    """
    # 1. Extract window-specific features from aggregated CSV
    BASE_FEATURES = get_base_features()  # 21 features from QA Q1
    window_features = df[[f'{window_type}_{feat}' for feat in BASE_FEATURES if f'{window_type}_{feat}' in df.columns]]

    # 2. Remove window prefix from column names (hook_scene_count → scene_count)
    window_features.columns = [col.replace(f'{window_type}_', '') for col in window_features.columns]

    # 3. Add target variable (same as Video-Level RF)
    if strategy == 'contrastive':
        top_count = int(video_count * 0.8)
        window_features['is_top_performer'] = (window_features.index < top_count).astype(int)

    # 4. NO TRANSFORMATION - use raw 21 base features directly
    # RF is scale-invariant, handles Boolean/categorical natively

    logger.info(f"Window-Level RF ({window_type}) transformation complete: {len(window_features)} rows, {len(window_features.columns)} columns")
    return window_features
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Window doesn't exist (e.g., middle_1 for bucket 3-9s) | Skip - not in window list | Bucket-specific window counts from MLPlanningv2.md |
| Column prefix mismatch | Fail-fast with error | Contract violation from Stage 3 |
| Missing base feature for window | Fail-fast with error | Incomplete data from Stage 3 |

#### Step 2.3.4: Window-Level K-Means Transformation

**Purpose**: Create heavily preprocessed, scaled [0-1] features for distance-based K-Means clustering

**Logic**:
```python
def transform_window_level_kmeans(df, window_type):
    """
    Transform aggregated features for Window-Level K-Means (one window type).
    Source: QA Q2c (complete transformation list)

    Args:
        df: pandas DataFrame from aggregated_features.csv
        window_type: str, e.g., "hook", "middle_1", "closing"

    Returns:
        pandas DataFrame with 39 features (all numerical, scaled [0-1])
    """
    # 1. Extract window-specific features
    BASE_FEATURES = get_base_features()
    window_features = df[[f'{window_type}_{feat}' for feat in BASE_FEATURES if f'{window_type}_{feat}' in df.columns]]
    window_features.columns = [col.replace(f'{window_type}_', '') for col in window_features.columns]

    df_km = window_features.copy()

    # 2. Log + Scale for count/variance features (11 features → 22 output columns)
    log_scale_features = ['scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count',
                          'overlay_unique_count', 'shortest_scene', 'longest_scene', 'scene_duration_variance',
                          'energy_variance', 'gaze_variance']  # User confirmed: variances use Log + scale (Q2c)

    for feature in log_scale_features:
        if feature in df_km.columns:
            # Log transform: log(1 + x) to handle zeros
            df_km[f'{feature}_log'] = np.log1p(df_km[feature])

            # MinMax scale to [0, 1]
            min_val = df_km[f'{feature}_log'].min()
            max_val = df_km[f'{feature}_log'].max()
            if max_val > min_val:
                df_km[f'{feature}_scaled'] = (df_km[f'{feature}_log'] - min_val) / (max_val - min_val)
            else:
                df_km[f'{feature}_scaled'] = 0.5  # All same value → midpoint

            # Drop intermediate log column and original
            df_km.drop(columns=[feature, f'{feature}_log'], inplace=True)

    # 3. Scale [0-1] for already-normalized features (7 features → 7 output columns)
    scale_features = ['average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
                     'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency']

    for feature in scale_features:
        if feature in df_km.columns:
            min_val = df_km[feature].min()
            max_val = df_km[feature].max()
            if max_val > min_val:
                df_km[f'{feature}_scaled'] = (df_km[feature] - min_val) / (max_val - min_val)
            else:
                df_km[f'{feature}_scaled'] = 0.5

            df_km.drop(columns=[feature], inplace=True)

    # 4. Shift + Scale for emotional_valence (1 feature → 1 output column)
    if 'emotional_valence' in df_km.columns:
        # Shift [-1,1] → [0,1]: (x + 1) / 2
        df_km['emotional_valence_scaled'] = (df_km['emotional_valence'] + 1) / 2
        df_km.drop(columns=['emotional_valence'], inplace=True)

    # 5. Label Encode for has_captions (1 feature → 1 output column)
    if 'has_captions' in df_km.columns:
        df_km['has_captions_encoded'] = df_km['has_captions'].astype(int)  # True→1, False→0
        df_km.drop(columns=['has_captions'], inplace=True)

    # 6. One-hot for dominant_emotion_id (1 feature → 7 output columns)
    if 'dominant_emotion_id' in df_km.columns:
        for emotion_id, emotion_name in enumerate(['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'], start=1):
            df_km[emotion_name] = (df_km['dominant_emotion_id'] == emotion_id).astype(int)
        df_km.drop(columns=['dominant_emotion_id'], inplace=True)

    logger.info(f"Window-Level K-Means ({window_type}) transformation complete: {len(df_km)} rows, {len(df_km.columns)} columns (expect 39)")
    return df_km
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| All features same value (variance=0) | Set scaled value to 0.5 (midpoint) | Avoid division by zero, midpoint is neutral |
| Negative count (should be impossible) | Fail during input validation | Caught in Step 2.3.1 |
| Unknown emotion_id | Fail during input validation | Caught in Step 2.3.1 |

#### Step 2.3.5: Output Validation and Checkpoint

**Purpose**: Verify all 13 transformation files have correct schemas before marking stage complete

**Logic**:
```python
def validate_outputs_and_checkpoint(output_files, bucket, video_count):
    """
    Validate all transformation outputs and write checkpoint.
    Source: QA Q4, Q6

    Args:
        output_files: dict, {filename: DataFrame} mapping
        bucket: str, bucket name
        video_count: int, expected row count

    Raises:
        AssertionError: if output validation fails
    """
    # 1. Check all 13 files created
    expected_files = get_expected_output_files(bucket)  # 1 video + 6 window RF + 6 window KM
    missing_files = [f for f in expected_files if f not in output_files]
    if missing_files:
        raise AssertionError(f"Missing output files: {missing_files}")

    # 2. Validate Video-Level RF schema
    df_rf = output_files['rf_transformed.csv']
    expected_rf_cols = get_expected_rf_column_count(bucket)  # ~183 for 18-33s
    assert 180 <= len(df_rf.columns) <= 190, f"Video-Level RF has {len(df_rf.columns)} columns, expected ~183"
    assert len(df_rf) == video_count, f"Video-Level RF has {len(df_rf)} rows, expected {video_count}"
    assert not df_rf.isnull().any().any(), "Video-Level RF contains NaN values"

    # Validate cross-window features (range checks)
    validate_cross_window_features(df_rf, bucket)

    # 3. Validate Window-Level RF schemas (6 files)
    for window in get_window_types(bucket):
        df_window_rf = output_files[f'{window}_rf_transformed.csv']
        assert len(df_window_rf.columns) == 22, f"{window} RF has {len(df_window_rf.columns)} columns, expected 22"
        assert len(df_window_rf) == video_count, f"{window} RF has {len(df_window_rf)} rows, expected {video_count}"
        assert not df_window_rf.isnull().any().any(), f"{window} RF contains NaN values"

    # 4. Validate Window-Level K-Means schemas (6 files)
    for window in get_window_types(bucket):
        df_window_km = output_files[f'{window}_km_transformed.csv']
        assert len(df_window_km.columns) == 39, f"{window} K-Means has {len(df_window_km.columns)} columns, expected 39"
        assert len(df_window_km) == video_count, f"{window} K-Means has {len(df_window_km)} rows, expected {video_count}"
        assert not df_window_km.isnull().any().any(), f"{window} K-Means contains NaN values"

        # Validate all _scaled columns are in [0,1] range
        scaled_cols = [c for c in df_window_km.columns if c.endswith('_scaled')]
        for col in scaled_cols:
            assert df_window_km[col].between(0, 1).all(), \
                f"{window} K-Means column {col} has values outside [0,1]: {df_window_km[col].min()}-{df_window_km[col].max()}"

    # 5. Write checkpoint
    checkpoint = {
        "stage": "feature_transformation",
        "status": "completed",
        "total_videos": video_count,
        "output_files": list(output_files.keys()),
        "completion_time": datetime.now().isoformat()
    }
    write_checkpoint(checkpoint, bucket)

    logger.info(f"Output validation passed and checkpoint written: {len(output_files)} files, {video_count} videos")
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Output file write fails (permission denied) | Fail-fast with error, no checkpoint | Cannot proceed to Stage 5 without outputs |
| Checkpoint write fails | Warn but continue | Outputs exist, Stage 5 can still run (checkpoint is for resume only) |
| Row count mismatch (input vs output) | Fail-fast with assertion | Data loss during transformation |

---

## 3. Dependencies & Integration

### 3.1 Input Dependencies

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **Foundation setup** | FoundationCHILD.md (Section 2: Client Architecture) | Directory structure + paths | `/data/clients/{client_id}/hashtags/{cluster_id}/{mode}_{strategy}/buckets/bucket_{duration}/ml_analysis/` | Fail-fast if directory doesn't exist (exit code 2) |
| **aggregated_features.csv** | Stage 3 (Feature Aggregation) | CSV (N rows, 66-216 cols depending on bucket) | All 21 base features × window count + 3 metadata (video_id, create_time, gender) | Fail-fast if file missing or invalid schema (exit code 1) |
| **Bucket context** | Orchestrator (rumiai_ml_batch.py) | Function parameter | bucket_path string (e.g., "/data/clients/nike/hashtags/fitness/top_contrastive/buckets/bucket_18-33s") | Fail-fast if invalid path |
| **Config parameters** | Stage 1 (config.json) | JSON | video_count (int), strategy (str: contrastive/top), client_id, cluster_id, mode, selection_strategy | Read from config.json, fail-fast if missing |
| **Stage 3 checkpoint** | Stage 3 output | JSON (checkpoint file) | status=="completed", aggregated_csv_path exists | Orchestrator validates before calling Stage 4 |

### 3.2 Output Contracts

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| **rf_transformed.csv** | CSV | (N rows, ~178 cols for bucket 18-33s) - All temporal window features + derived temporal features (hour, day_of_week, is_weekend, is_business_hours, month) + one-hot encoded (has_captions, dominant_emotion, gender) + target (is_top_performer) | Stage 5 (ML Model Training) - Video-Level RF | Assert 175 ≤ cols ≤ 185, no NaN, N rows preserved |
| **{window}_rf_transformed.csv** (6 files) | CSV | (N rows, 22 cols) - 21 raw base features + is_top_performer target | Stage 5 (ML Model Training) - Window-Level RF (6 models) | Assert exactly 22 cols, no NaN, N rows preserved |
| **{window}_km_transformed.csv** (6 files) | CSV | (N rows, 39 cols) - All features log/scaled to [0-1], original features dropped | Stage 5 (ML Model Training) - Window-Level K-Means (6 models) | Assert exactly 39 cols, all _scaled cols in [0-1], no NaN, N rows preserved |
| **stage_4_checkpoint.json** | JSON | {"stage": "feature_transformation", "status": "completed", "total_videos": N, "output_files": [...], "completion_time": ISO8601} | Orchestrator (resume logic) | None (informational only) |

**Output File Paths** (all relative to bucket directory):
```
/data/clients/{client_id}/hashtags/{cluster_id}/{mode}_{strategy}/buckets/bucket_18-33s/ml_analysis/
├── rf_transformed.csv
├── hook_rf_transformed.csv
├── middle_1_rf_transformed.csv
├── middle_2_rf_transformed.csv
├── middle_3_rf_transformed.csv
├── middle_4_rf_transformed.csv
├── closing_rf_transformed.csv
├── hook_km_transformed.csv
├── middle_1_km_transformed.csv
├── middle_2_km_transformed.csv
├── middle_3_km_transformed.csv
├── middle_4_km_transformed.csv
└── closing_km_transformed.csv
```

### 3.3 Cross-Stage Dependencies

**This stage depends on**:
- **Stage 1 (Video Discovery)**: Creates config.json with strategy, video_count parameters
- **Stage 3 (Feature Aggregation)**: Must complete successfully (aggregated_features.csv exists with valid schema, checkpoint status=="completed")
- **Stage 2 fail-fast**: Ensures no NaN values from failed video processing (Stage 2 stops pipeline if videos fail)

**This stage is required by**:
- **Stage 5 (ML Model Training)**: Expects all 13 transformation files in exact format (schema validation happens before training)
- Stage 5 Video-Level RF model requires: `rf_transformed.csv` with ~178 features
- Stage 5 Window-Level RF models (6) require: `{window}_rf_transformed.csv` with 22 features each
- Stage 5 Window-Level K-Means models (6) require: `{window}_km_transformed.csv` with 39 features each, all scaled [0-1]

**Failure Impact**:
- If this stage fails: Stage 5 cannot run (no transformed features for ML training)
- If transformation produces wrong schema: Stage 5 fails immediately with schema validation error
- Checkpoint: Resume from this stage without re-running Stages 1-3 (aggregated_features.csv still valid)
- Skip-on-fail: If bucket transformation fails (e.g., corrupted data), orchestrator logs error, skips bucket, continues with next bucket (partial analysis still useful)

### 3.4 External Dependencies

**Python Libraries**:
```python
import pandas as pd  # 2.0.0+ (DataFrame operations, CSV I/O)
import numpy as np   # 1.24.0+ (log1p, sin/cos for cyclical encoding)
import os           # File path operations
import logging      # Performance logging
from datetime import datetime  # Timestamp handling, checkpoint creation
```

**File System**:
- Read access: `/data/clients/{client_id}/hashtags/{cluster_id}/{mode}_{strategy}/buckets/bucket_{duration}/ml_analysis/` (aggregated_features.csv)
- Write access: Same directory (all 13 output files + checkpoint)
- Write access: `/data/clients/{client_id}/hashtags/{cluster_id}/{mode}_{strategy}/buckets/bucket_{duration}/checkpoints/` (checkpoint JSON)

**Environment Variables**: None (paths passed as function parameters from orchestrator)

**External Services**: None (pure computational stage, no API calls, no external ML services)

---

## 4. Configuration & Parameters

### 4.1 CLI Parameters

**Note**: This stage does not have a standalone CLI interface. It is called internally by the pipeline orchestrator (`rumiai_ml_batch.py`) after Stage 3 completes. Parameters are passed as function arguments from config.json.

**Parameters received from orchestrator**:
- `bucket_path` (str): Full path to bucket directory, e.g., "/data/clients/nike/hashtags/fitness/top_contrastive/buckets/bucket_18-33s"
- `config` (dict): Loaded from config.json, contains: {"strategy": "contrastive", "video_count": 100, "client_id": "nike", "cluster_id": "fitness", "mode": "top", "selection_strategy": "contrastive"}

### 4.2 Internal Configuration

```python
# ===== File Paths (relative to bucket directory) =====
AGGREGATED_CSV_PATH = "ml_analysis/aggregated_features.csv"
OUTPUT_DIR = "ml_analysis/"
CHECKPOINT_DIR = "checkpoints/"
CHECKPOINT_FILE = "stage_4_checkpoint.json"

# ===== Base Features (21 total - from FeatureTransformation.md) =====
BASE_FEATURES = [
    # Visual features
    'average_face_size', 'overlay_unique_count', 'scene_count', 'shortest_scene',
    'longest_scene', 'scene_duration_variance', 'object_count', 'person_count',
    'eye_contact_rate', 'gaze_variance', 'gesture_count',
    # Audio features
    'speech_coverage', 'word_count', 'energy_level', 'energy_variance', 'energy_max', 'pitch_scatter_ratio',
    # Emotion features
    'dominant_emotion_id', 'emotional_valence', 'emotion_consistency',
    # Text features
    'has_captions'
]

# ===== Transformation Categories (from Q2c) =====
# Log + Scale features (11 features)
LOG_SCALE_FEATURES = [
    'scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count',
    'overlay_unique_count', 'shortest_scene', 'longest_scene', 'scene_duration_variance',
    'energy_variance', 'gaze_variance'  # Variances use Log + scale (Q2c user decision)
]

# Scale [0-1] features (7 features)
SCALE_FEATURES = [
    'average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
    'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency'
]

# Categorical features
CATEGORICAL_FEATURES = {
    'has_captions': 'boolean',  # Label encode for K-Means, one-hot for RF
    'dominant_emotion_id': 'ordinal_1_7'  # One-hot for both
}

# ===== Cross-Window Features (NEW) =====
CROSS_WINDOW_FEATURES = [
    'hook_to_middle_energy_delta',
    'middle_to_closing_contrast',
    'eye_contact_consistency',
    'word_density_std',
    'energy_progression_slope'
]  # 5 features added to Video-Level RF (Crosswindowupgrade.md)

# ===== Performance Thresholds (from Q5) =====
TARGET_TIME_SECONDS = 30  # Target processing time for N=100 videos
WARNING_TIME_SECONDS = 60  # Warn if exceeds 1 minute
TIMEOUT_SECONDS = 300  # Fail if exceeds 5 minutes

TARGET_MEMORY_MB = 500  # Target peak memory
WARNING_MEMORY_MB = 1024  # Warn if exceeds 1 GB
FAIL_MEMORY_MB = 2048  # Fail if exceeds 2 GB

MINIMUM_VIDEO_COUNT = 50  # Minimum videos for reliable ML training

# ===== Bucket-Specific Window Counts =====
# IMPLEMENTATION: Import from shared config (single source of truth)
from config.bucket_definitions import BUCKET_WINDOWS

# BUCKET_WINDOWS contains bucket-specific window configurations
# See config/bucket_definitions.py for complete definition
# See FoundationCHILD.md Section 6: Bucket Definitions for documentation

# ===== Expected Column Counts (from Q1, Q2a) =====
EXPECTED_INPUT_COLUMNS = {
    '0-3s': 24,    # 21 × 1 + 3 metadata (video_id, create_time, gender)
    '3-9s': 45,    # 21 × 2 + 3 metadata
    '9-13s': 66,   # 21 × 3 + 3 metadata
    '13-18s': 66,  # 21 × 3 + 3 metadata
    '18-33s': 129, # 21 × 6 + 3 metadata
    '33-60s': 150, # 21 × 7 + 3 metadata
    '60-90s': 150,
    '90-120s': 150,
}

# ===== Logging Configuration =====
LOG_PERFORMANCE = True  # Log per-operation timing
LOG_MEMORY_USAGE = True  # Log peak memory
```

---

## 5. Data Schemas

### 5.1 Input Schema

**File**: `ml_analysis/aggregated_features.csv`
**Source**: Stage 3 (Feature Aggregation)

**Complete 21 Base Features** (from Q1, repeated per window):

| Column | Type | Range | Nulls? | Description | Example |
|--------|------|-------|--------|-------------|---------|
| `hook_average_face_size` | float | [0-1] | No | Mean face prominence in hook window (0-3s) | 0.45 |
| `hook_overlay_unique_count` | int | [0-∞] | No | Count of unique text overlays in hook | 2 |
| `hook_has_captions` | bool | True/False | No | Speech-synchronized captions present in hook | True |
| `hook_scene_count` | int | [0-∞] | No | Number of scene changes in hook | 3 |
| `hook_shortest_scene` | float | [0-∞] | No | Duration of shortest scene in hook (seconds) | 0.8 |
| `hook_longest_scene` | float | [0-∞] | No | Duration of longest scene in hook (seconds) | 1.5 |
| `hook_scene_duration_variance` | float | [0-∞] | No | Variance in scene durations in hook | 0.12 |
| `hook_object_count` | int | [0-∞] | No | Non-person objects detected in hook | 5 |
| `hook_person_count` | int | [0-∞] | No | Maximum persons visible simultaneously in hook | 1 |
| `hook_speech_coverage` | float | [0-1] | No | Speech density (% of hook with speech) | 0.85 |
| `hook_word_count` | int | [0-∞] | No | Total words spoken in hook | 15 |
| `hook_energy_level` | float | [0-1] | No | Mean audio intensity in hook | 0.72 |
| `hook_energy_variance` | float | [0-∞] | No | Audio intensity variance in hook | 0.03 |
| `hook_energy_max` | float | [0-1] | No | Peak audio intensity in hook | 0.95 |
| `hook_pitch_scatter_ratio` | float | [0-1] | No | Pitch instability measure in hook | 0.18 |
| `hook_gesture_count` | int | [0-∞] | No | Hand movement count in hook | 8 |
| `hook_gaze_variance` | float | [0-∞] | No | Gaze stability variance in hook | 0.05 |
| `hook_eye_contact_rate` | float | [0-1] | No | Eye contact percentage in hook | 0.85 |
| `hook_dominant_emotion_id` | int | 1-7 | No | Most frequent emotion in hook (1=joy, 2=sadness, 3=anger, 4=fear, 5=disgust, 6=surprise, 7=neutral) | 1 |
| `hook_emotional_valence` | float | [-1, 1] | No | Positive vs negative tone in hook | 0.65 |
| `hook_emotion_consistency` | float | [0, 1] | No | Emotional focus consistency in hook | 0.80 |

**Middle segment columns**: Same 21 features prefixed with `middle_1_`, `middle_2_`, ..., `middle_N_` (N varies by bucket)

**Closing segment columns**: Same 21 features prefixed with `closing_`

**Metadata columns** (video-level, not per-window):

| Column | Type | Range | Nulls? | Description | Example |
|--------|------|-------|--------|-------------|---------|
| `video_id` | string | - | No | Video identifier (not used in transformation, kept for traceability) | "238506412723073" |
| `create_time` | string (ISO 8601) | - | No | Video publish timestamp | "2025-01-15T14:30:00Z" |
| `gender` | string | male/female/null | Yes | Detected gender | "male" |

**Total Columns by Bucket**:
- Bucket 0-3s: 24 columns (21 × 1 + 3 metadata: video_id, create_time, gender)
- Bucket 3-9s: 45 columns (21 × 2 + 3 metadata)
- Bucket 9-13s, 13-18s: 66 columns (21 × 3 + 3 metadata) - middle_aggregate
- Bucket 18-33s: 129 columns (21 × 6 + 3 metadata)
- Bucket 33-60s, 60-90s, 90-120s: 150 columns (21 × 7 + 3 metadata)

### 5.2 Output Schema

#### File 1: `ml_analysis/rf_transformed.csv` (Video-Level RF)

**Transformation**: One-hot encoding + temporal extraction + target variable addition

| Column | Type | Range | Nulls? | Description | Transformation |
|--------|------|-------|--------|-------------|----------------|
| `hook_scene_count` | int | [0-∞] | No | Same as input | Direct (unchanged) |
| `hook_eye_contact_rate` | float | [0-1] | No | Same as input | Direct (unchanged) |
| ... | ... | ... | ... | (All temporal window features preserved) | Direct |
| `no_captions` | int | 0, 1 | No | 1 if has_captions==False | One-hot from has_captions |
| `has_captions` | int | 0, 1 | No | 1 if has_captions==True | One-hot from has_captions |
| `joy` | int | 0, 1 | No | 1 if dominant_emotion_id==1 | One-hot from dominant_emotion_id |
| `sadness` | int | 0, 1 | No | 1 if dominant_emotion_id==2 | One-hot from dominant_emotion_id |
| `anger` | int | 0, 1 | No | 1 if dominant_emotion_id==3 | One-hot from dominant_emotion_id |
| `fear` | int | 0, 1 | No | 1 if dominant_emotion_id==4 | One-hot from dominant_emotion_id |
| `disgust` | int | 0, 1 | No | 1 if dominant_emotion_id==5 | One-hot from dominant_emotion_id |
| `surprise` | int | 0, 1 | No | 1 if dominant_emotion_id==6 | One-hot from dominant_emotion_id |
| `neutral` | int | 0, 1 | No | 1 if dominant_emotion_id==7 | One-hot from dominant_emotion_id |
| `hour` | int | 0-23 | No | Hour of day from create_time | Extracted from create_time |
| `day_of_week` | int | 0-6 | No | Day of week (0=Monday) | Extracted from create_time |
| `month` | int | 1-12 | No | Month from create_time | Extracted from create_time |
| `is_weekend` | int | 0, 1 | No | 1 if Saturday or Sunday | Derived from day_of_week |
| `is_business_hours` | int | 0, 1 | No | 1 if 9am-5pm | Derived from hour |
| `gender_male` | int | 0, 1 | No | 1 if gender=="male" | One-hot from gender |
| `gender_female` | int | 0, 1 | No | 1 if gender=="female" | One-hot from gender |
| `gender_nan` | int | 0, 1 | No | 1 if gender is null | One-hot from gender (dummy_na=True) |
| `is_top_performer` | int | 0, 1 | No | Target variable (contrastive only): 1 if top 80%, 0 if bottom 20% | Computed from video rank |
| `hook_to_middle_energy_delta` | float | [-1, 1] | No | Energy change from hook to middle average | Computed cross-window delta |
| `middle_to_closing_contrast` | float | [-1, 1] | No | Energy gap between middle avg and closing peak | Computed cross-window delta |
| `eye_contact_consistency` | float | [0, 1] | No | Std deviation of eye contact across all windows | Computed consistency metric |
| `word_density_std` | float | [0, ∞] | No | Std deviation of word count across windows | Computed consistency metric |
| `energy_progression_slope` | float | [-∞, ∞] | No | Linear regression slope of energy across windows | Computed progression metric |

**Removed Columns**: `create_time` (replaced with 5 temporal features), `gender` (replaced with 2-3 one-hot features), `dominant_emotion_id` (replaced with 7 one-hot features)

**Total Columns**: ~183 for bucket 18-33s (129 input - 3 removed + 23 derived + 1 target)
                                                         ^^^^ 18 original + 5 cross-window

#### File 2-7: `ml_analysis/{window}_rf_transformed.csv` (Window-Level RF, 6 files)

**Transformation**: Extract window columns, remove prefix, add target (NO feature transformation)

| Column | Type | Range | Nulls? | Description | Transformation |
|--------|------|-------|--------|-------------|----------------|
| `average_face_size` | float | [0-1] | No | Mean face prominence (window-specific) | Extracted from {window}_average_face_size, prefix removed |
| `overlay_unique_count` | int | [0-∞] | No | Count of unique text overlays | Extracted, prefix removed |
| `has_captions` | bool | True/False | No | Speech-synchronized captions present | Extracted, prefix removed (raw Boolean, NO transformation) |
| `scene_count` | int | [0-∞] | No | Number of scene changes | Extracted, prefix removed |
| `shortest_scene` | float | [0-∞] | No | Duration of shortest scene | Extracted, prefix removed |
| `longest_scene` | float | [0-∞] | No | Duration of longest scene | Extracted, prefix removed |
| `scene_duration_variance` | float | [0-∞] | No | Variance in scene durations | Extracted, prefix removed |
| `object_count` | int | [0-∞] | No | Non-person objects detected | Extracted, prefix removed |
| `person_count` | int | [0-∞] | No | Maximum persons visible | Extracted, prefix removed |
| `speech_coverage` | float | [0-1] | No | Speech density | Extracted, prefix removed |
| `word_count` | int | [0-∞] | No | Total words spoken | Extracted, prefix removed |
| `energy_level` | float | [0-1] | No | Mean audio intensity | Extracted, prefix removed |
| `energy_variance` | float | [0-∞] | No | Audio intensity variance | Extracted, prefix removed |
| `energy_max` | float | [0-1] | No | Peak audio intensity | Extracted, prefix removed |
| `pitch_scatter_ratio` | float | [0-1] | No | Pitch instability | Extracted, prefix removed |
| `gesture_count` | int | [0-∞] | No | Hand movement count | Extracted, prefix removed |
| `gaze_variance` | float | [0-∞] | No | Gaze stability variance | Extracted, prefix removed |
| `eye_contact_rate` | float | [0-1] | No | Eye contact percentage | Extracted, prefix removed |
| `dominant_emotion_id` | int | 1-7 | No | Most frequent emotion | Extracted, prefix removed (raw int, NO transformation) |
| `emotional_valence` | float | [-1, 1] | No | Positive vs negative tone | Extracted, prefix removed |
| `emotion_consistency` | float | [0, 1] | No | Emotional focus consistency | Extracted, prefix removed |
| `is_top_performer` | int | 0, 1 | No | Target variable (same as Video-Level RF) | Computed from video rank |

**Total Columns**: Exactly 22 (21 base features + 1 target)

**Files Created** (bucket 18-33s):
- `hook_rf_transformed.csv`
- `middle_1_rf_transformed.csv`
- `middle_2_rf_transformed.csv`
- `middle_3_rf_transformed.csv`
- `middle_4_rf_transformed.csv`
- `closing_rf_transformed.csv`

#### File 8-13: `ml_analysis/{window}_km_transformed.csv` (Window-Level K-Means, 6 files)

**Transformation**: Log + scale, scale [0-1], shift + scale, label encode, one-hot (all features → [0-1] numerical)

**Log + Scale features** (11 features → 22 columns):

| Original Column | Output Column 1 | Output Column 2 | Type | Range | Description |
|-----------------|-----------------|-----------------|------|-------|-------------|
| `scene_count` | `scene_count_log` | `scene_count_scaled` | float | [0-1] | log1p(scene_count), then MinMax scaled |
| `word_count` | `word_count_log` | `word_count_scaled` | float | [0-1] | log1p(word_count), then MinMax scaled |
| `gesture_count` | `gesture_count_log` | `gesture_count_scaled` | float | [0-1] | log1p(gesture_count), then MinMax scaled |
| `object_count` | `object_count_log` | `object_count_scaled` | float | [0-1] | log1p(object_count), then MinMax scaled |
| `person_count` | `person_count_log` | `person_count_scaled` | float | [0-1] | log1p(person_count), then MinMax scaled |
| `overlay_unique_count` | `overlay_unique_count_log` | `overlay_unique_count_scaled` | float | [0-1] | log1p(overlay_unique_count), then MinMax scaled |
| `shortest_scene` | `shortest_scene_log` | `shortest_scene_scaled` | float | [0-1] | log1p(shortest_scene), then MinMax scaled |
| `longest_scene` | `longest_scene_log` | `longest_scene_scaled` | float | [0-1] | log1p(longest_scene), then MinMax scaled |
| `scene_duration_variance` | `scene_duration_variance_log` | `scene_duration_variance_scaled` | float | [0-1] | log1p(scene_duration_variance), then MinMax scaled |
| `energy_variance` | `energy_variance_log` | `energy_variance_scaled` | float | [0-1] | log1p(energy_variance), then MinMax scaled |
| `gaze_variance` | `gaze_variance_log` | `gaze_variance_scaled` | float | [0-1] | log1p(gaze_variance), then MinMax scaled |

**Scale [0-1] features** (7 features → 7 columns):

| Original Column | Output Column | Type | Range | Description |
|-----------------|---------------|------|-------|-------------|
| `average_face_size` | `average_face_size_scaled` | float | [0-1] | MinMax scaled (already normalized) |
| `speech_coverage` | `speech_coverage_scaled` | float | [0-1] | MinMax scaled |
| `energy_level` | `energy_level_scaled` | float | [0-1] | MinMax scaled |
| `energy_max` | `energy_max_scaled` | float | [0-1] | MinMax scaled |
| `pitch_scatter_ratio` | `pitch_scatter_ratio_scaled` | float | [0-1] | MinMax scaled |
| `eye_contact_rate` | `eye_contact_rate_scaled` | float | [0-1] | MinMax scaled |
| `emotion_consistency` | `emotion_consistency_scaled` | float | [0-1] | MinMax scaled |

**Shift + Scale features** (1 feature → 1 column):

| Original Column | Output Column | Type | Range | Description |
|-----------------|---------------|------|-------|-------------|
| `emotional_valence` | `emotional_valence_scaled` | float | [0-1] | Shifted from [-1,1] to [0,1]: (x + 1) / 2 |

**Label Encode features** (1 feature → 1 column):

| Original Column | Output Column | Type | Range | Description |
|-----------------|---------------|------|-------|-------------|
| `has_captions` | `has_captions_encoded` | int | 0, 1 | True→1, False→0 |

**One-hot features** (1 feature → 7 columns):

| Original Column | Output Columns | Type | Range | Description |
|-----------------|----------------|------|-------|-------------|
| `dominant_emotion_id` | `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise`, `neutral` | int (each) | 0, 1 | Binary column per emotion (1 if that emotion is dominant) |

**Removed Columns**: All original 21 base features are dropped after transformation (only transformed versions remain)

**Total Columns**: Exactly 39 (22 log + 7 scale + 1 shift + 1 label + 7 one-hot + 1 target = 39)

**Files Created** (bucket 18-33s): Same 6 files as Window-Level RF, with `_km_transformed.csv` suffix

---

## 6. Error Handling & Validation

### 6.1 Input Validation

```python
def validate_input(df, bucket, expected_count):
    """
    Validate aggregated features CSV before transformation.
    Fail-fast with clear error messages.
    Source: QA Q4

    Args:
        df: pandas DataFrame from aggregated_features.csv
        bucket: str, bucket name (e.g., "18-33s")
        expected_count: int, expected number of videos from config

    Raises:
        ValueError: if validation fails with specific error message
    """
    # 1. Check column count
    expected_cols = EXPECTED_INPUT_COLUMNS[bucket]
    if len(df.columns) != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns for bucket {bucket}, found {len(df.columns)}")

    # 2. Check required columns exist
    required_cols = get_required_columns(bucket)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    # 3. Check for NaN values (fail-fast)
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        nan_count = {col: df[col].isna().sum() for col in nan_cols}
        raise ValueError(f"Invalid input: NaN values detected: {nan_count}. Check Stage 3 aggregation logic.")

    # 4. Check normalized features [0-1]
    for col in df.columns:
        if any(feat in col for feat in SCALE_FEATURES):
            if (df[col] < 0).any() or (df[col] > 1).any():
                invalid = df[(df[col] < 0) | (df[col] > 1)]
                raise ValueError(f"Out of range: {col} has value {invalid[col].max()}, expected [0.0-1.0]. Check Stage 2 calculation.")

    # 5. Check count features non-negative + sanity bounds
    for col in df.columns:
        if any(feat in col for feat in ['scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count']):
            if (df[col] < 0).any():
                raise ValueError(f"Out of range: {col} has negative values. Check Stage 2 calculation.")
            if (df[col] > 10000).any():
                raise ValueError(f"Out of range: {col} has suspiciously high values (>10000). Check Stage 2 calculation.")

    # 6. Check minimum row count
    if len(df) < MINIMUM_VIDEO_COUNT:
        raise ValueError(f"Insufficient data: {len(df)} videos found, minimum {MINIMUM_VIDEO_COUNT} required for ML training.")

    if len(df) < expected_count:
        logger.warning(f"Warning: Expected {expected_count} videos, found {len(df)}. Proceeding with reduced sample size.")

    logger.info(f"Input validation passed: {len(df)} videos, {len(df.columns)} columns")
```

### 6.2 Error Cases

| Error | Detection | Handling | User Message | Exit Code |
|-------|-----------|----------|--------------|-----------|
| Missing input file | `os.path.exists(csv_path)` | Fail-fast | `"Aggregated CSV not found at {path}. Did Stage 3 complete successfully?"` | 1 |
| Invalid CSV format | `pd.read_csv()` exception | Fail-fast | `"Failed to parse CSV: {error}. Check file is valid CSV format."` | 2 |
| Missing required column | Column validation (Step 2.3.1) | Fail-fast | `"Required columns missing: {cols}. Expected 126 temporal columns (21 features × 6 windows), found {actual}."` | 3 |
| NaN values in required fields | `.isnull().any()` | Fail-fast | `"Invalid input: 5 rows contain NaN values in hook_scene_count. Check Stage 3 aggregation logic."` | 4 |
| Invalid duration range | Range validation | Fail-fast | `"Out of range: hook_eye_contact_rate has value 1.5, expected [0.0-1.0]. Check Stage 2 eye contact calculation."` | 5 |
| Insufficient videos (N<50) | Row count check | Fail-fast | `"Insufficient data: 45 videos found, minimum 50 required for ML training."` | 6 |
| Insufficient videos (N<expected but ≥50) | Row count check | Warn + continue | `"Warning: Expected 100 videos, found 95. Proceeding with reduced sample size."` | 0 (warning) |
| Write permission denied | File write exception | Fail-fast | `"Cannot write to {path}. Check permissions."` | 7 |
| Timeout (>5 minutes) | Execution time check | Fail-fast | `"Stage 4 timed out after {elapsed}s (limit: 300s). Check for performance issues."` | 8 |
| Out of memory (>2GB peak) | Memory monitoring | Fail-fast | `"Peak memory {peak_mb}MB exceeds limit 2048MB. Try reducing batch size."` | 9 |

### 6.3 Output Validation

```python
def validate_output(output_files, bucket, video_count):
    """
    Validate transformed CSVs before saving and marking stage complete.
    Source: QA Q4

    Args:
        output_files: dict, {filename: DataFrame} mapping
        bucket: str, bucket name
        video_count: int, expected row count

    Raises:
        AssertionError: if output schema invalid
    """
    # 1. Check all 13 files created
    expected_files = get_expected_output_files(bucket)
    missing = [f for f in expected_files if f not in output_files]
    if missing:
        raise AssertionError(f"Missing output files: {missing}")

    # 2. Validate Video-Level RF
    df_rf = output_files['rf_transformed.csv']
    assert 180 <= len(df_rf.columns) <= 190, f"Video-Level RF has {len(df_rf.columns)} columns, expected ~183"
    assert len(df_rf) == video_count, f"Video-Level RF has {len(df_rf)} rows, expected {video_count}"
    assert not df_rf.isnull().any().any(), "Video-Level RF contains NaN values"

    # 3. Validate Window-Level RF (6 files)
    for window in BUCKET_WINDOWS[bucket]:
        df_w_rf = output_files[f'{window}_rf_transformed.csv']
        assert len(df_w_rf.columns) == 22, f"{window} RF has {len(df_w_rf.columns)} columns, expected 22"
        assert len(df_w_rf) == video_count, f"{window} RF has {len(df_w_rf)} rows, expected {video_count}"
        assert not df_w_rf.isnull().any().any(), f"{window} RF contains NaN values"

    # 4. Validate Window-Level K-Means (6 files)
    for window in BUCKET_WINDOWS[bucket]:
        df_w_km = output_files[f'{window}_km_transformed.csv']
        assert len(df_w_km.columns) == 39, f"{window} K-Means has {len(df_w_km.columns)} columns, expected 39"
        assert len(df_w_km) == video_count, f"{window} K-Means has {len(df_w_km)} rows, expected {video_count}"
        assert not df_w_km.isnull().any().any(), f"{window} K-Means contains NaN values"

        # All _scaled columns must be [0,1]
        scaled_cols = [c for c in df_w_km.columns if c.endswith('_scaled')]
        for col in scaled_cols:
            assert df_w_km[col].between(0, 1).all(), \
                f"{window} K-Means column {col} has values outside [0,1]: {df_w_km[col].min()}-{df_w_km[col].max()}"

    logger.info(f"Output validation passed: {len(output_files)} files, all schemas correct")


def validate_cross_window_features(df_rf, bucket):
    """
    Validate cross-window feature ranges in Video-Level RF output.

    Args:
        df_rf: DataFrame with Video-Level RF transformed features
        bucket: Bucket name (e.g., "18-33s")

    Raises:
        AssertionError: if cross-window features have invalid values

    Source: Crosswindowupgrade.md Section 7.3
    """
    # Validate delta features (energy deltas bounded by [-1, 1])
    if 'hook_to_middle_energy_delta' in df_rf.columns:
        assert df_rf['hook_to_middle_energy_delta'].between(-1, 1).all(), \
            f"hook_to_middle_energy_delta out of range [-1, 1]: min={df_rf['hook_to_middle_energy_delta'].min():.3f}, max={df_rf['hook_to_middle_energy_delta'].max():.3f}"

    if 'middle_to_closing_contrast' in df_rf.columns:
        assert df_rf['middle_to_closing_contrast'].between(-1, 1).all(), \
            f"middle_to_closing_contrast out of range [-1, 1]: min={df_rf['middle_to_closing_contrast'].min():.3f}, max={df_rf['middle_to_closing_contrast'].max():.3f}"

    # Validate consistency features (std must be non-negative)
    if 'eye_contact_consistency' in df_rf.columns:
        assert (df_rf['eye_contact_consistency'] >= 0).all() and (df_rf['eye_contact_consistency'] <= 1).all(), \
            f"eye_contact_consistency out of range [0, 1]: min={df_rf['eye_contact_consistency'].min():.3f}, max={df_rf['eye_contact_consistency'].max():.3f}"

    if 'word_density_std' in df_rf.columns:
        assert (df_rf['word_density_std'] >= 0).all(), \
            f"word_density_std has negative values: min={df_rf['word_density_std'].min():.3f}"

    # Validate slope feature (sanity check: shouldn't be extreme)
    if 'energy_progression_slope' in df_rf.columns:
        # Slope > 2 means feature increases by 200%+ per window (suspiciously large)
        assert df_rf['energy_progression_slope'].between(-2, 2).all(), \
            f"energy_progression_slope suspiciously large: min={df_rf['energy_progression_slope'].min():.3f}, max={df_rf['energy_progression_slope'].max():.3f}"

    logger.info("✓ Cross-window feature validation passed")
```

---

## 7. Performance & Scalability

### 7.1 Performance Targets

- **Throughput**: Process N=100 videos (bucket 18-33s) in < 30 seconds (target) / < 5 minutes (timeout)
- **Memory**: Peak usage < 500 MB (target), warn at 1 GB, fail at 2 GB
- **Disk I/O**: Write 13 CSV files in < 5 seconds (on SSD)
- **CPU**: < 50% average utilization (single-threaded pandas/numpy operations)

### 7.2 Measured Performance

*Note: Measurements will be added after initial implementation and testing*

**Expected performance** (based on pandas/numpy benchmarks for N=100, bucket 18-33s):
- Load CSV: ~1 second
- Validation: <1 second
- Video-Level RF transformation: ~2 seconds (one-hot encoding)
- Window-Level RF transformation (6 iterations): ~1 second (column extraction only)
- Window-Level K-Means transformation (6 iterations): ~4 seconds (log transforms, scaling)
- Write 13 CSVs: ~2 seconds (SSD)
- **Total: ~10-15 seconds** (well under 30-second target)

### 7.3 Bottlenecks & Mitigations

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| **Disk I/O (HDD)** | 5-10s write time | HDD sequential write speed ~50-100 MB/s | Use SSD (500+ MB/s), or buffer writes | Medium |
| **One-hot encoding** | 1-2s for dominant_emotion_id | pandas `get_dummies()` creates 7 new columns | Acceptable performance, no optimization needed | Low |
| **Log + scale iterations** | 3-4s for 6 windows × 11 features | Nested loops: 66 log1p + MinMax operations | Vectorize with numpy broadcasting (future optimization) | Low |
| **Memory growth (N>300)** | 2+ GB peak for N=300 | pandas loads full CSV into memory | Acceptable (N=300 is max per bucket), warn at 1 GB | Low |

### 7.4 Scalability Limits

- **Max videos per bucket**: 300 (design limit from MLPlanningv2.md Part 1)
- **Expected memory at N=300**: ~1.5 GB (3× growth from N=100)
- **Expected time at N=300**: ~30-45 seconds (linear scaling from 10-15s at N=100)
- **Min videos per bucket**: 50 (below this, ML training unreliable)

**Performance logging** (from Q5):
```python
logger.info(f"Stage 4 completed in {elapsed:.1f}s (target: <30s)")
logger.info(f"Peak memory: {peak_mb:.0f}MB (target: <500MB)")
logger.info(f"Video-Level RF: {video_rf_time:.1f}s, Window-Level RF: {window_rf_time:.1f}s, Window-Level K-Means: {window_km_time:.1f}s, I/O: {io_time:.1f}s")
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Source**: QA Q7 (Layer 1 - Unit tests with synthetic data)

**Test fixtures**:
- `tests/fixtures/stage4/test_bucket_18-33s_minimal.csv` (10 videos, 129 columns, synthetic data)
- `tests/fixtures/stage4/test_bucket_9-13s_minimal.csv` (10 videos, 66 columns, middle_aggregate)
- `tests/fixtures/stage4/test_bucket_3-9s_minimal.csv` (10 videos, 45 columns, hook + closing only)

**Test cases**:

- [ ] **Test input validation**
  - Empty CSV (N=0) → raises ValueError "Insufficient data: 0 videos found, minimum 50 required"
  - Missing required columns → raises ValueError with column names
  - NaN values in required fields → raises ValueError with affected columns
  - Out-of-range values (eye_contact_rate=1.5) → raises ValueError with specific column and value
  - Valid input (N=50+, all columns, no NaN) → passes without error

- [ ] **Test Video-Level RF transformations**
  - One-hot encoding: has_captions=True → no_captions=0, has_captions=1
  - One-hot encoding: dominant_emotion_id=1 (joy) → joy=1, others=0
  - Temporal extraction: create_time="2025-01-15T14:30:00" → hour=14, day_of_week=2, is_weekend=0
  - One-hot encoding: gender="female" → gender_male=0, gender_female=1, gender_nan=0
  - Target variable (contrastive): top 80% → is_top_performer=1, bottom 20% → is_top_performer=0

- [ ] **Test Window-Level RF transformations**
  - Column extraction: hook_scene_count=3 → scene_count=3 (prefix removed)
  - No transformation: has_captions stays Boolean (not one-hot)
  - dominant_emotion_id stays int 1-7 (not one-hot)
  - All 6 window files have identical column names (different data)

- [ ] **Test Window-Level K-Means transformations**
  - Log + scale: scene_count=3 → scene_count_log=1.386, scene_count_scaled=0.5 (if min=1, max=5)
  - Scale [0-1]: eye_contact_rate=0.85 → eye_contact_rate_scaled=0.78 (if min=0.45, max=0.95)
  - Shift + scale: emotional_valence=-0.5 → emotional_valence_scaled=0.25 ((-0.5 + 1) / 2)
  - Label encode: has_captions=True → has_captions_encoded=1
  - One-hot: dominant_emotion_id=1 → joy=1, sadness=0, ..., neutral=0

- [ ] **Test edge cases**
  - All features same value (variance=0) → scaled value set to 0.5 (midpoint)
  - Single video (N=1) → raises ValueError "Insufficient data" (below minimum 50)
  - Missing gender (null) → gender_nan=1 via dummy_na=True
  - Bucket 9-13s with middle_aggregate (not middle_1/2/3) → extracts correctly

- [ ] **Test output validation**
  - Video-Level RF: column count in range [175-185] for bucket 18-33s
  - Window-Level RF: exactly 22 columns per file
  - Window-Level K-Means: exactly 39 columns per file
  - All K-Means _scaled columns in [0,1] range
  - No NaN values introduced during transformation
  - Row count preserved (input N = output N for all files)

- [ ] **Test cross-window feature computation**
  - Hook energy=0.5, middle_1=0.6, middle_2=0.7, middle_3=0.65, middle_4=0.70
    → hook_to_middle_energy_delta ≈ 0.1625 (middle avg 0.6625 - hook 0.5)
    → middle_to_closing_contrast (if closing=0.8) ≈ 0.1375 (closing 0.8 - middle avg 0.6625)
  - Eye contact across 6 windows: [0.85, 0.80, 0.82, 0.83, 0.81, 0.84]
    → eye_contact_consistency ≈ 0.018 (low std = consistent)
  - Word count across 6 windows: [10, 15, 20, 25, 30, 12]
    → word_density_std ≈ 7.76 (high std = uneven pacing)
  - Energy progression across 6 windows: [0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    → energy_progression_slope ≈ 0.057 (positive = rising energy)
  - Bucket 0-3s (no middle segments)
    → hook_to_middle_energy_delta = 0.0, middle_to_closing_contrast = 0.0 (neutral)
  - Bucket 9-13s (middle_aggregate)
    → Deltas computed correctly using middle_aggregate as single middle value

**Expected runtime**: <1 second total for all unit tests

### 8.2 Integration Tests

**Source**: QA Q7 (Layer 2 - Integration tests with real Stage 3 output)

**Test fixture creation**:
1. Run small test analysis: `python rumiai_ml_batch.py --client test --target fitness --video-count 50`
2. After Stage 3 completes, copy output: `cp /data/clients/test/hashtags/fitness/top_contrastive/buckets/bucket_18-33s/ml_analysis/aggregated_features.csv tests/fixtures/stage4/real_bucket_18-33s_stage3_output.csv`
3. Commit to git (fixture is ~20 KB)

**Test cases**:

- [ ] **End-to-end: Stage 3 → Stage 4 output generation**
  - Load real Stage 3 output (50 videos, bucket 18-33s)
  - Run Stage 4 transformation
  - Verify all 13 files created
  - Verify schemas match expected (Video-Level RF ~178 cols, Window-Level RF 22 cols, Window-Level K-Means 39 cols)

- [ ] **Output schema validation**
  - Video-Level RF: 175 ≤ cols ≤ 185 (allow flexibility)
  - Window-Level RF: exactly 22 cols per file
  - Window-Level K-Means: exactly 39 cols per file
  - Required columns exist (hook_scene_count, is_top_performer, joy, etc.)

- [ ] **Value range validation**
  - All `_scaled` columns in K-Means are [0-1]
  - All one-hot encoded columns are {0, 1}
  - All log-transformed values are non-negative

- [ ] **Row preservation**
  - Input: 50 rows → Output: 50 rows in all 13 files
  - No rows dropped during transformation

**Expected runtime**: <5 seconds for integration tests

### 8.3 Test Data

**Synthetic fixture** (unit tests):

**File**: `tests/fixtures/stage4/test_bucket_18-33s_minimal.csv` (10 videos, 129 columns)

```csv
hook_scene_count,hook_eye_contact_rate,hook_word_count,hook_energy_level,hook_dominant_emotion_id,hook_emotional_valence,hook_has_captions,...,middle_1_scene_count,...,closing_scene_count,...,create_time,gender
3,0.85,15,0.72,1,0.65,True,...,4,...,2,...,2025-01-15T14:30:00,female
5,0.62,8,0.58,7,-0.15,False,...,3,...,1,...,2025-01-16T09:15:00,male
2,0.91,22,0.80,1,0.80,True,...,5,...,3,...,2025-01-17T18:45:00,female
```

**Real fixture** (integration tests):

**File**: `tests/fixtures/stage4/real_bucket_18-33s_stage3_output.csv` (50 videos, 129 columns, captured from actual Stage 3 run)

*[Real data with actual feature distributions from test analysis run]*

### 8.4 Test Execution

```bash
# Run all unit tests
pytest tests/unit/test_feature_transformation.py -v

# Run integration tests
pytest tests/integration/test_stage4_full_pipeline.py -v

# Run with coverage
pytest tests/ --cov=feature_transformation --cov-report=html

# Run performance benchmarks
pytest tests/performance/test_stage4_timing.py -v --benchmark-only
```

---

## 9. Future Enhancements

### 9.1 Planned Improvements

**Phase 2: Automatic feature scaling optimization**
- Current: Fixed Log + scale for counts, Scale [0-1] for rates
- Future: Analyze distribution skewness per feature per bucket, apply BoxCox/Yeo-Johnson if skewness > threshold
- Impact: Better K-Means clustering quality (more Gaussian-like distributions)
- Estimated effort: 1 week

**Phase 3: Parallel bucket processing**
- Current: Sequential bucket processing (one bucket at a time)
- Future: Process top 3 buckets in parallel (independent transformations)
- Impact: 3× speedup for full hashtag analysis (3 buckets × 30s = 90s → 30s)
- Estimated effort: 3 days
- Constraint: Requires multiprocessing coordination with orchestrator

**Phase 4: Incremental transformation (checkpoint within Stage 4)**
- Current: Stage 4 completes all 13 files or fails entirely
- Future: Checkpoint after each file (e.g., Video-Level RF complete → checkpoint, resume from Window-Level RF if interrupted)
- Impact: Resume faster for very large batches (N=300)
- Estimated effort: 2 days

### 9.2 Known Limitations

**Manual transformation logic**: Transformation rules (Log + scale, Scale [0-1], etc.) are hardcoded per feature. No automatic feature engineering (e.g., polynomial features, interaction terms).

**Fixed output schemas**: Adding new base features (e.g., face_size_variance) requires updating all 3 transformation pipelines + updating EXPECTED_OUTPUT_COLUMNS constants.

**No feature selection**: All 21 base features are included in every transformation. No filtering based on RF importance or correlation analysis.

**Single-threaded**: pandas operations are not parallelized. For N=300, could use Dask for parallel DataFrame operations (estimated 2× speedup).

**Missing MLPlanningv2.md transformation code**: MLPlanningv2.md Section 4.3 has incomplete transformation code (missing 8 features, documented in QA Q2c). This HLD corrects and completes the logic.

---

## 10. References & Related Docs

### 10.1 Parent Document

- **MLPlanningv2.md Section 4 "Stage 4: Feature Transformation"** (Lines 1360-1586)
  - High-level transformation overview
  - Triple pipeline architecture (Video-Level RF + Window-Level RF + Window-Level K-Means)
  - Stage position in pipeline

### 10.2 Foundation Dependencies

- **FoundationCHILD.md**
  - Section 2 "Client Architecture & Storage": Directory paths used in this stage (bucket structure, ml_analysis/)
  - Section 4 "CLI Command Structure": Configuration parameters and defaults
  - Section 6 "Bucket Definitions": Bucket-specific window counts used in transformations
  - Section 1 "System Goals & Success Criteria": Sequential bucket processing model
  - Section 5.3 "Checkpoint Schema": Checkpoint format for pipeline resumption

### 10.3 Related Child Docs

- **FeatureAggregationCHILD.md** (Stage 3)
  - Produces `aggregated_features.csv` (input to this stage)
  - Defines exact column names and temporal window structure
  - Documents middle segment aggregation for buckets 9-13s and 13-18s

- **MLModelTrainingCHILD.md** (Stage 5)
  - Consumes all 13 transformation files (outputs from this stage)
  - Trains 90 models total (8 Video-Level RF + 41 Window-Level RF + 41 Window-Level K-Means)
  - Defines expected transformation schemas for model input validation

### 10.4 External References

- **FeatureTransformation.md**: Complete feature transformation table (21 base features with exact transformation rules)
- **Pandas API**: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html (one-hot encoding)
- **NumPy log1p**: https://numpy.org/doc/stable/reference/generated/numpy.log1p.html (log transformation handling zeros)
- **SystemArchitecturev2.md**: Current production system architecture for context

---

## Appendix A: Decision Log

**Purpose**: Record major design decisions with rationale from Phase 1 Critique

**Decision 1**: Triple Pipeline Architecture (Video-Level RF + Window-Level RF + Window-Level K-Means)
- **Context**: Critique Q1 asked whether Window-Level RF was necessary or redundant with Video-Level RF
- **Alternatives Considered**:
  - Option A: Dual Pipeline (Video-Level RF + Window-Level K-Means only) - 46% fewer files, 2 weeks faster
  - Option B: Triple Pipeline (adds Window-Level RF) - Complete validation coverage, 1 week additional work
- **Rationale**: Window-Level RF provides isolated per-window feature rankings (21 features competing only against each other), enabling comparisons like "Eye contact is 2× more important for hooks (0.35) than closings (0.18)". This cannot be derived from Video-Level RF which ranks features globally (190 features). Window-Level RF also validates K-Means cluster features at the same granularity (both operate on 21 features per window).
- **Trade-offs**: +1 week development time, +50% maintenance burden (3 pipelines vs 2), but eliminates future rework risk and provides 100% pattern coverage vs 70-80% for dual pipeline.
- **Date**: 2025-10-13 (Phase 1 Critique final decision)

**Decision 2**: Variance Features Use Log + Scale (Not Scale [0-1] Only)
- **Context**: Q2c found conflict between MLPlanningv2.md Section 4.3 (lists energy_variance and gaze_variance as "rate features" for Scale [0-1]) and FeatureTransformation.md (says both need "Log + scale" because they're variances)
- **Alternatives Considered**:
  - Option A: Log + scale (consistent with scene_duration_variance treatment)
  - Option B: Scale [0-1] only (as shown in MLPlanningv2.md pseudocode)
- **Rationale**: Variances are squared deviations, inherently right-skewed. Data range is [0-∞] (unbounded), not [0-1]. Log transform compresses outliers (log(10)=2.3, log(100)=4.6) preserving ratio information. Prevents extreme variance outliers from dominating K-Means distance calculations. Mathematically consistent with other variance features.
- **Trade-offs**: MLPlanningv2.md Section 4.3 pseudocode is incorrect and needs updating. This HLD documents the correct transformation.
- **Date**: 2025-10-13 (Phase 2 Q2c user decision)

**Decision 3**: Fail-Fast Validation Strategy (No Imputation or Row Dropping)
- **Context**: Q4 asked how to handle missing/invalid data (NaN, out-of-range values, missing columns)
- **Alternatives Considered**:
  - Option A: Fail-fast with error (halt pipeline, force fix upstream)
  - Option B: Drop rows with NaN (reduces N from 100 to 95)
  - Option C: Impute NaN with mean/median
- **Rationale**: Stage 2 fail-fast ensures complete features exist (if video processing fails, pipeline stops). NaN indicates upstream bug in Stage 2 or Stage 3. Imputing NaN introduces synthetic data into ML training (poor model quality). Dropping rows reduces sample size below minimum threshold (N=100 → N=95 might not be enough). Out-of-range values indicate data corruption or bugs that should be debugged, not hidden.
- **Trade-offs**: Less fault-tolerant (pipeline stops on bad data), but prioritizes data integrity over partial processing. Aligns with RumiAI fail-fast philosophy from SystemArchitecturev2.md.
- **Date**: 2025-10-13 (Phase 2 Q4 user decision)

**Decision 4**: Add Cross-Window Features to Video-Level RF Only
- **Context**: Critique_Stage7_LLMAnalysis.md (Stage 7 LLM Analysis critique) identified critical gap - cross-window delta features (hook_to_middle_energy_delta, middle_to_closing_contrast, eye_contact_consistency, word_density_std, energy_progression_slope) are NOT computed anywhere in current pipeline
- **Alternatives Considered**:
  - **Option A** (chosen): Add to Video-Level RF transformation (Stage 4, Step 6.5)
  - **Option B**: Add to Window-Level RF transformation (rejected - architectural mismatch)
  - **Option C**: Add to Stage 3 aggregation (rejected - aggregation layer should stay simple)
- **Rationale**: Cross-window features require multiple windows (hook, middle segments, closing) to compute deltas, consistency metrics, and progression slopes. Video-Level RF sees all windows simultaneously (178 features across 6 windows), making it the correct location. Window-Level RF operates on isolated windows (21 features per window), incompatible with cross-window computations.
- **Trade-offs**: +5 features to Video-Level RF (178→183), +80 lines code/docs, +1.5 hours development time, but provides explicit temporal patterns to ML model (vs implicit learning from raw window features)
- **Date**: 2025-10-15 (Crosswindowupgrade.md planning)

---

## Appendix B: Example Data

### B.1 Sample Input (3 rows, 12 columns shown out of 129 total)

**File**: `ml_analysis/aggregated_features.csv` (bucket 18-33s)

```csv
hook_scene_count,hook_eye_contact_rate,hook_word_count,hook_energy_level,hook_dominant_emotion_id,hook_emotional_valence,middle_1_scene_count,middle_2_scene_count,closing_energy_level,create_time,gender
3,0.85,15,0.72,1,0.65,4,3,0.75,2025-01-15T14:30:00,female
5,0.62,8,0.58,7,-0.15,3,5,0.82,2025-01-16T09:15:00,male
2,0.91,22,0.80,1,0.80,5,4,0.68,2025-01-17T18:45:00,female
```

### B.2 Sample Video-Level RF Output (3 rows, 20 columns shown out of ~183 total)

**File**: `ml_analysis/rf_transformed.csv`

```csv
hook_scene_count,hook_eye_contact_rate,hook_word_count,middle_1_scene_count,closing_energy_level,joy,neutral,hour,day_of_week,is_weekend,is_business_hours,gender_male,gender_female,hook_to_middle_energy_delta,middle_to_closing_contrast,eye_contact_consistency,word_density_std,energy_progression_slope,is_top_performer
3,0.85,15,4,0.75,1,0,14,2,0,1,0,1,0.16,0.27,0.018,7.2,0.057,1
5,0.62,8,3,0.82,0,1,9,3,0,1,1,0,-0.05,0.15,0.032,5.8,0.042,1
2,0.91,22,5,0.68,1,0,18,4,0,0,0,1,0.12,0.20,0.024,8.1,0.031,1
```

### B.3 Sample Window-Level RF Output (3 rows, hook window)

**File**: `ml_analysis/hook_rf_transformed.csv` (22 columns total)

```csv
scene_count,eye_contact_rate,word_count,energy_level,dominant_emotion_id,emotional_valence,has_captions,...,is_top_performer
3,0.85,15,0.72,1,0.65,True,...,1
5,0.62,8,0.58,7,-0.15,False,...,1
2,0.91,22,0.80,1,0.80,True,...,1
```

### B.4 Sample Window-Level K-Means Output (3 rows, hook window, 12 columns shown out of 39 total)

**File**: `ml_analysis/hook_km_transformed.csv`

```csv
scene_count_log,scene_count_scaled,eye_contact_rate_scaled,word_count_log,word_count_scaled,emotional_valence_scaled,has_captions_encoded,joy,neutral,...
1.386,0.45,0.78,2.773,0.62,0.825,1,1,0,...
1.791,0.85,0.15,2.197,0.35,0.425,0,0,1,...
1.099,0.12,1.0,3.135,0.88,0.90,1,1,0,...
```

**Explanation of transformations** (row 1):
- scene_count=3 → log1p(3)=1.386 → scaled to 0.45 (assuming min=1.099, max=1.791 in this sample)
- eye_contact_rate=0.85 → scaled to 0.78 (assuming min=0.62, max=0.91 in this sample)
- word_count=15 → log1p(15)=2.773 → scaled to 0.62
- emotional_valence=0.65 (originally in [-1,1]) → shifted to [0,1]: (0.65+1)/2 = 0.825
- has_captions=True → encoded to 1
- dominant_emotion_id=1 (joy) → joy=1, neutral=0

---

## Appendix C: Pseudocode (Complete)

### C.1 Full Transformation Pipeline

```python
def run_stage4_transformation(bucket_path, config):
    """
    Stage 4: Feature Transformation

    Complete implementation logic for triple pipeline architecture.
    Sources: Phase 2 Q&A (Q2a, Q2b, Q2c), Phase 1 Critique (triple pipeline decision)

    Args:
        bucket_path: str, full path to bucket directory
        config: dict, loaded from config.json

    Returns:
        tuple: (success: bool, output_files: list, elapsed_time: float)

    Raises:
        ValueError: if input validation fails
        AssertionError: if output validation fails
        IOError: if file I/O fails
    """
    start_time = time.time()
    logger.info(f"Starting Stage 4 transformation for bucket {bucket_path}")

    # ===== 1. Load Configuration =====
    bucket_name = os.path.basename(bucket_path)  # e.g., "bucket_18-33s"
    bucket = bucket_name.replace('bucket_', '')  # "18-33s"
    strategy = config['strategy']  # "contrastive" or "top"
    video_count = config['video_count']  # 100 (expected)

    # ===== 2. Load Input =====
    input_path = os.path.join(bucket_path, AGGREGATED_CSV_PATH)
    if not os.path.exists(input_path):
        raise ValueError(f"Aggregated CSV not found at {input_path}. Did Stage 3 complete successfully?")

    logger.info(f"Loading aggregated features from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} videos, {len(df.columns)} columns")

    # ===== 3. Validate Input =====
    logger.info("Validating input schema and data quality")
    validate_input(df, bucket, video_count)  # See Section 6.1 - raises ValueError if invalid

    output_files = {}

    # ===== 4. Video-Level RF Transformation =====
    logger.info("Transforming features for Video-Level Random Forest")
    rf_start = time.time()
    df_rf = transform_video_level_rf(df, strategy, video_count)  # See Section 2.3.2
    output_files['rf_transformed.csv'] = df_rf
    rf_time = time.time() - rf_start
    logger.info(f"Video-Level RF complete: {len(df_rf)} rows, {len(df_rf.columns)} columns ({rf_time:.1f}s)")

    # ===== 5. Window-Level RF Transformation =====
    logger.info("Transforming features for Window-Level Random Forest")
    window_rf_start = time.time()
    windows = BUCKET_WINDOWS[bucket]  # e.g., ['hook', 'middle_1', ..., 'closing']

    for window_type in windows:
        df_window_rf = transform_window_level_rf(df, window_type, strategy, video_count)  # See Section 2.3.3
        output_files[f'{window_type}_rf_transformed.csv'] = df_window_rf
        logger.info(f"  {window_type} RF: {len(df_window_rf)} rows, {len(df_window_rf.columns)} columns")

    window_rf_time = time.time() - window_rf_start
    logger.info(f"Window-Level RF complete: {len(windows)} files ({window_rf_time:.1f}s)")

    # ===== 6. Window-Level K-Means Transformation =====
    logger.info("Transforming features for Window-Level K-Means")
    window_km_start = time.time()

    for window_type in windows:
        df_window_km = transform_window_level_kmeans(df, window_type)  # See Section 2.3.4
        output_files[f'{window_type}_km_transformed.csv'] = df_window_km
        logger.info(f"  {window_type} K-Means: {len(df_window_km)} rows, {len(df_window_km.columns)} columns")

    window_km_time = time.time() - window_km_start
    logger.info(f"Window-Level K-Means complete: {len(windows)} files ({window_km_time:.1f}s)")

    # ===== 7. Validate Outputs =====
    logger.info("Validating output schemas")
    validate_outputs_and_checkpoint(output_files, bucket, len(df))  # See Section 2.3.5 - raises AssertionError if invalid

    # ===== 8. Write Output Files =====
    logger.info("Writing output files to disk")
    io_start = time.time()
    output_dir = os.path.join(bucket_path, OUTPUT_DIR)

    for filename, df_output in output_files.items():
        output_path = os.path.join(output_dir, filename)
        df_output.to_csv(output_path, index=False)
        logger.info(f"  Wrote {filename}: {os.path.getsize(output_path) / 1024:.1f} KB")

    io_time = time.time() - io_start
    logger.info(f"File I/O complete: {len(output_files)} files ({io_time:.1f}s)")

    # ===== 9. Write Checkpoint =====
    checkpoint = {
        "stage": "feature_transformation",
        "status": "completed",
        "total_videos": len(df),
        "output_files": list(output_files.keys()),
        "completion_time": datetime.now().isoformat(),
        "elapsed_time": time.time() - start_time
    }

    checkpoint_path = os.path.join(bucket_path, CHECKPOINT_DIR, CHECKPOINT_FILE)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    # ===== 10. Performance Summary =====
    elapsed = time.time() - start_time
    logger.info(f"Stage 4 completed in {elapsed:.1f}s (target: <30s)")
    logger.info(f"  Video-Level RF: {rf_time:.1f}s, Window-Level RF: {window_rf_time:.1f}s, Window-Level K-Means: {window_km_time:.1f}s, I/O: {io_time:.1f}s")

    # Check performance targets
    if elapsed > TIMEOUT_SECONDS:
        raise TimeoutError(f"Stage 4 timed out after {elapsed:.0f}s (limit: {TIMEOUT_SECONDS}s). Check for performance issues.")
    if elapsed > WARNING_TIME_SECONDS:
        logger.warning(f"Stage 4 exceeded target time: {elapsed:.1f}s > {WARNING_TIME_SECONDS}s")

    return True, list(output_files.keys()), elapsed


def transform_video_level_rf(df, strategy, video_count):
    """See Section 2.3.2 for complete implementation"""
    # (Full logic documented in Section 2.3.2)
    pass


def transform_window_level_rf(df, window_type, strategy, video_count):
    """See Section 2.3.3 for complete implementation"""
    # (Full logic documented in Section 2.3.3)
    pass


def transform_window_level_kmeans(df, window_type):
    """See Section 2.3.4 for complete implementation"""
    # (Full logic documented in Section 2.3.4)
    pass


def validate_input(df, bucket, expected_count):
    """See Section 6.1 for complete implementation"""
    # (Full logic documented in Section 6.1)
    pass


def validate_outputs_and_checkpoint(output_files, bucket, video_count):
    """See Section 2.3.5 for complete implementation"""
    # (Full logic documented in Section 2.3.5)
    pass
```

---

## Document Metadata

**Creation Date**: 2025-10-13
**Last Modified**: 2025-10-13
**Phase 1 Critique**: Critique_FeatureTransformation.md (APPROVED - Triple Pipeline Architecture)
**Phase 2 Q&A**: QA_FeatureTransformation.md (COMPLETE - 7 questions answered)
**Status**: Draft (ready for Phase 4 review if user requests changes)

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.1 | 2025-01-28 | RumiAI Team | Fixed broken references: Updated all Mother HLD references to point to FoundationCHILD.md (4 locations: Lines 18-21, 451, 589, 1154-1159). Enforces three-tier architecture: Mother → Foundation → Components. |
| 1.0 | 2025-10-13 | Claude (Phase 3 Generation) | Initial complete draft from Phase 1 Critique + Phase 2 Q&A |
