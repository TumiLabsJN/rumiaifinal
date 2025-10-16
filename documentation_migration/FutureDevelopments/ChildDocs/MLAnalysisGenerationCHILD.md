# ML Analysis Generation (Stage 6) - High-Level Design

> **Parent**: MLPlanningv2.md - Stage 6: ML Analysis Generation (Lines 1993-2388)
> **Version**: 1.0
> **Last Updated**: 2025-01-28
> **Status**: Draft

---

## 1. Context & Business Goal

### 1.1 What Problem Does This Solve?

Stage 5 trains 90 ML models (8 Video-Level RF + 41 Window-Level RF + 41 Window-Level K-Means) and stores them as pickle files with minimal metadata. Stage 7 needs **structured, LLM-consumable insights** extracted from these models - not raw pickle files. This component (Stage 6) bridges the gap by:

1. **Extracting model insights**: Feature importance rankings, cluster centroids, performance metrics
2. **Computing distribution statistics**: Percentile thresholds and high/medium/low classifications for top features
3. **Formatting for LLM consumption**: Structured JSON outputs optimized for Claude's context window

Without Stage 6, Stage 7 would need to load 90 pickle files and perform complex ML data extraction, increasing hallucination risk and execution time.

### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This component depends on FoundationCHILD.md for:
- Client directory structure (Section 2: Client Architecture & Storage) - provides `bucket_*/ml_analysis/` paths
- Bucket window configuration (Section 6: Centralized Configuration) - `config/bucket_definitions.py` for window structure

```
Stage 4: Feature Transformation
   ↓ Output: 13 transformed CSVs per bucket (RF + K-Means transformed features)
Stage 5: ML Model Training
   ↓ Output: 90 trained models (PKL files: RF models, K-Means models, scalers, X matrices, metrics)
Stage 6: ML ANALYSIS GENERATION
   ↓ Output: 13 JSON files per bucket (Video RF + Window RF + Window K-Means analysis)
Stage 7: LLM Creative Reports
```

### 1.3 Success Criteria

- [ ] Generate all 13 JSON files per bucket successfully (Video RF + 6-7 Window RF + 6-7 Window K-Means)
- [ ] Complete bucket processing in 3-5 seconds (target), <10 seconds (acceptable)
- [ ] Peak memory usage < 200MB per bucket
- [ ] Atomic output pattern: All 13 JSONs succeed OR all deleted on failure (no partial output)
- [ ] Feature name consistency: K-Means features normalized to match RF naming (no `_scaled` suffixes in output)
- [ ] Pre-flight validation: Fail-fast if Stage 4 or Stage 5 dependencies missing
- [ ] Distribution data included: 66th/33rd percentile thresholds and high/medium/low percentages for top 10 features

---

## 2. Architecture & Design

### 2.1 High-Level Approach

Stage 6 operates as a **pure extraction and formatting layer** - no ML training, no data transformation. It loads pre-trained models from Stage 5, extracts insights from model attributes (`feature_importances_`, `cluster_centers_`), computes distribution statistics from Stage 4 transformed data, and generates 3 types of JSON files:

1. **Video-Level RF JSON** (1 per bucket): Cross-window feature importance with distribution analysis
2. **Window-Level RF JSONs** (6-7 per bucket): Per-window feature importance rankings
3. **Window-Level K-Means JSONs** (6-7 per bucket): Cluster centroids with normalized feature names

All 13 JSONs are written to a temp directory first, validated, then atomically committed to final locations (consistent with Stage 5's atomic pattern). Pre-flight validation ensures all Stage 4 CSVs and Stage 5 models exist before generating any output.

### 2.2 Data Flow

```
Input 1: Stage 5 Models (20 PKL files per bucket)
         Location: bucket_{bucket}/models/
         Files: rf_video_{bucket}.pkl, rf_{window}_{bucket}.pkl, {window}_kmeans_{bucket}.pkl, {window}_scalers_{bucket}.pkl, {window}_X_data_{bucket}.pkl, model_metrics.json
   ↓
Input 2: Stage 4 Transformed CSVs (13 files per bucket)
         Location: bucket_{bucket}/ml_analysis/
         Files: aggregated_features.csv, rf_transformed.csv, {window}_rf_transformed.csv, {window}_km_transformed.csv
   ↓
Process Step 1: PRE-FLIGHT VALIDATION
   - Check all 20 Stage 5 model files exist
   - Check all 13 Stage 4 CSV files exist
   - Fail-fast if any dependencies missing (exit code 1)
   ↓
Process Step 2: GENERATE VIDEO-LEVEL RF JSON
   - Load rf_video_{bucket}.pkl
   - Extract feature_importances_ attribute (top 10 features)
   - Load aggregated_features.csv for distribution analysis
   - Compute 66th/33rd percentile thresholds per feature
   - Calculate high/medium/low percentages for top/bottom performers
   - Write to temp directory: .tmp/rf_video_analysis.json.tmp
   ↓
Process Step 3: GENERATE WINDOW-LEVEL RF JSONs (6-7 files)
   - For each window: Load rf_{window}_{bucket}.pkl
   - Extract feature_importances_ (top 10 features)
   - Load model_metrics.json for accuracy/precision/recall
   - Compute distribution stats from {window}_rf_transformed.csv
   - Write to temp directory: .tmp/{window}_rf_analysis.json.tmp
   ↓
Process Step 4: GENERATE WINDOW-LEVEL K-MEANS JSONs (6-7 files)
   - For each window: Load {window}_kmeans_{bucket}.pkl
   - Extract cluster_centers_ attribute (3 clusters × 21-39 features)
   - Load {window}_X_data_{bucket}.pkl to get feature names
   - NORMALIZE feature names (remove _scaled, _log, _encoded suffixes)
   - Assign videos to clusters, compute distances
   - Write to temp directory: .tmp/{window}_kmeans_analysis.json.tmp
   ↓
Process Step 5: VALIDATE ALL JSON SCHEMAS
   - Check all 13 temp files created
   - Validate JSON structure (parseable)
   - Check feature name consistency (no `_scaled` suffixes)
   - If validation fails: Delete temp directory, exit code 3
   ↓
Process Step 6: ATOMIC COMMIT (RENAME ALL 13 FILES AT ONCE)
   - Rename all .json.tmp files to .json
   - Move from .tmp/ to ml_analysis/
   - Delete temp directory
   - Success: exit code 0
   ↓
Output: 13 JSON files per bucket (~95KB total)
        Location: bucket_{bucket}/ml_analysis/
        Files: rf_video_analysis.json, {window}_rf_analysis.json (×6-7), {window}_kmeans_analysis.json (×6-7)
```

### 2.3 Detailed Process

#### Step 2.3.1: Pre-Flight Validation

**Purpose**: Ensure all Stage 4 and Stage 5 dependencies exist before generating any JSONs (fail-fast principle)

**Logic**:
```python
def validate_stage_dependencies(bucket_path: str, bucket: str, windows: list) -> None:
    """
    Validate Stage 4 and Stage 5 outputs exist.

    Args:
        bucket_path: Absolute path to bucket directory (e.g., /data/clients/acme/buckets/bucket_18-33s)
        bucket: Bucket name (e.g., "18-33s")
        windows: Window list from bucket configuration (e.g., ['hook', 'middle_1', ..., 'closing'])

    Raises:
        PreFlightValidationError: If any required file missing
    """
    missing_files = []

    # ===== Validate Stage 4 CSVs (13 files for bucket 18-33s) =====
    required_stage4_files = [
        'ml_analysis/aggregated_features.csv',         # For distribution analysis
        'ml_analysis/rf_transformed.csv',              # Video-level RF input
        *[f'ml_analysis/{w}_rf_transformed.csv' for w in windows],     # Window-level RF
        *[f'ml_analysis/{w}_km_transformed.csv' for w in windows]      # Window-level K-Means
    ]

    for file_path in required_stage4_files:
        full_path = os.path.join(bucket_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(('Stage 4', file_path))

    # ===== Validate Stage 5 Models (20 files for bucket 18-33s) =====
    required_stage5_files = [
        f'models/rf_video_{bucket}.pkl',                               # Video-level RF model
        *[f'models/rf_{w}_{bucket}.pkl' for w in windows],            # Window-level RF models
        *[f'models/{w}_kmeans_{bucket}.pkl' for w in windows],        # K-Means models
        *[f'models/{w}_scalers_{bucket}.pkl' for w in windows],       # Scalers for K-Means
        *[f'models/{w}_X_data_{bucket}.pkl' for w in windows],        # X matrices for feature names
        'models/model_metrics.json'                                    # Performance metrics
    ]

    for file_path in required_stage5_files:
        full_path = os.path.join(bucket_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(('Stage 5', file_path))

    # ===== Fail-fast if any dependencies missing =====
    if missing_files:
        # Group by stage for clear error message
        stage4_missing = [f for s, f in missing_files if s == 'Stage 4']
        stage5_missing = [f for s, f in missing_files if s == 'Stage 5']

        error_msg = "Pre-flight validation failed:\n"
        if stage4_missing:
            error_msg += f"Stage 4 incomplete ({len(stage4_missing)} files missing):\n"
            error_msg += "\n".join(f"  - {f}" for f in stage4_missing[:5])  # Show first 5
            if len(stage4_missing) > 5:
                error_msg += f"\n  ... and {len(stage4_missing)-5} more"
            error_msg += "\nAction: Re-run Stage 4 (Feature Transformation)\n"

        if stage5_missing:
            error_msg += f"Stage 5 incomplete ({len(stage5_missing)} files missing):\n"
            error_msg += "\n".join(f"  - {f}" for f in stage5_missing[:5])
            if len(stage5_missing) > 5:
                error_msg += f"\n  ... and {len(stage5_missing)-5} more"
            error_msg += "\nAction: Re-run Stage 5 (ML Model Training)\n"

        raise PreFlightValidationError(error_msg)

    logger.info(f"✓ Pre-flight validation passed: All {len(required_stage4_files)} Stage 4 files + {len(required_stage5_files)} Stage 5 files exist")
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Single file missing (e.g., 19 of 20 Stage 5 models exist) | Fail-fast with specific file name | Partial model set is unusable - prevent ambiguous failures later |
| Stage 4 incomplete but Stage 5 complete | Fail with Stage 4 message | Stage 6 needs both stages - clear error prevents confusion |
| Temp directory already exists from previous run | Delete temp directory before starting | Clean state prevents leftover files from failed previous run |

#### Step 2.3.2: Generate Video-Level RF JSON

**Purpose**: Extract cross-window feature importance with distribution analysis for LLM consumption

**Logic**:
```python
def generate_video_rf_json(bucket_path: str, bucket: str) -> dict:
    """
    Generate Video-Level RF analysis JSON with distribution statistics.

    Args:
        bucket_path: Absolute path to bucket directory
        bucket: Bucket name (e.g., "18-33s")

    Returns:
        dict: Video RF analysis JSON structure

    Source: Q&A Q4 (Output Schema), Q1 (Distribution Analysis)
    """
    # ===== 1. Load Video-Level RF Model =====
    model_path = os.path.join(bucket_path, f'models/rf_video_{bucket}.pkl')
    rf_model = joblib.load(model_path)

    # ===== 2. Extract Feature Importance =====
    feature_importances = rf_model.feature_importances_  # NumPy array (length = 183 for bucket 18-33s after cross-window features)
    feature_names = rf_model.feature_names_in_  # From sklearn attribute

    # Sort features by importance, take top 10
    importance_indices = np.argsort(feature_importances)[::-1][:10]
    top_features = [
        {
            'feature': feature_names[idx],
            'importance': float(feature_importances[idx])
        }
        for idx in importance_indices
    ]

    # ===== 3. Load aggregated_features.csv for Distribution Analysis =====
    agg_csv_path = os.path.join(bucket_path, 'ml_analysis/aggregated_features.csv')
    df = pd.read_csv(agg_csv_path)

    # Determine top/bottom performers (contrastive strategy)
    video_count = len(df)
    top_count = int(video_count * 0.8)
    df['is_top_performer'] = [1] * top_count + [0] * (video_count - top_count)

    # ===== 4. Compute Distribution Stats for Each Top Feature =====
    for feature_data in top_features:
        feature_name = feature_data['feature']

        # Skip if feature not in aggregated CSV (e.g., derived features)
        if feature_name not in df.columns:
            # Use default distribution (no stats available)
            feature_data['top_performer_avg'] = None
            feature_data['bottom_performer_avg'] = None
            feature_data['gap'] = None
            feature_data['distribution'] = None
            continue

        # Compute averages
        top_performers = df[df['is_top_performer'] == 1][feature_name]
        bottom_performers = df[df['is_top_performer'] == 0][feature_name]

        top_avg = float(top_performers.mean())
        bottom_avg = float(bottom_performers.mean())
        gap = abs(top_avg - bottom_avg)

        # Compute percentile thresholds (66th, 33rd)
        high_threshold = float(top_performers.quantile(0.66))
        low_threshold = float(top_performers.quantile(0.33))

        # Compute percentage distributions
        top_high_pct = (top_performers >= high_threshold).sum() / len(top_performers)
        top_med_pct = ((top_performers >= low_threshold) & (top_performers < high_threshold)).sum() / len(top_performers)
        top_low_pct = (top_performers < low_threshold).sum() / len(top_performers)

        bottom_high_pct = (bottom_performers >= high_threshold).sum() / len(bottom_performers)
        bottom_med_pct = ((bottom_performers >= low_threshold) & (bottom_performers < high_threshold)).sum() / len(bottom_performers)
        bottom_low_pct = (bottom_performers < low_threshold).sum() / len(bottom_performers)

        # Add to feature data
        feature_data['top_performer_avg'] = top_avg
        feature_data['bottom_performer_avg'] = bottom_avg
        feature_data['gap'] = gap
        feature_data['distribution'] = {
            'thresholds': {
                'high': high_threshold,
                'low': low_threshold
            },
            'top_performers': {
                'high_percentage': float(top_high_pct),
                'medium_percentage': float(top_med_pct),
                'low_percentage': float(top_low_pct)
            },
            'bottom_performers': {
                'high_percentage': float(bottom_high_pct),
                'medium_percentage': float(bottom_med_pct),
                'low_percentage': float(bottom_low_pct)
            }
        }

    # ===== 5. Build Video RF Analysis JSON =====
    analysis_json = {
        'analysis_type': 'random_forest',
        'bucket': bucket,
        'hashtag': None,  # Set by caller if available
        'video_count': video_count,
        'input_features': len(feature_names),
        'feature_importance': top_features
    }

    return analysis_json
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Feature in RF model but not in aggregated CSV (derived feature) | Set distribution fields to `None` | Cannot compute distribution for features not in raw data - LLM can still use importance score |
| All videos have same value for a feature (variance=0) | Percentiles will equal mean | Valid edge case - distribution shows no spread (all values in "medium" range) |
| Video count mismatch (RF trained on 100 videos, CSV has 98 rows) | Log warning but continue | Non-critical - distribution based on available data |

#### Step 2.3.3: Generate Window-Level RF JSONs

**Purpose**: Extract per-window feature importance rankings for focused window analysis in Stage 7 Phase 1

**Logic**:
```python
def generate_window_rf_json(bucket_path: str, bucket: str, window: str) -> dict:
    """
    Generate Window-Level RF analysis JSON.

    Args:
        bucket_path: Absolute path to bucket directory
        bucket: Bucket name (e.g., "18-33s")
        window: Window name (e.g., "hook", "middle_1", "closing")

    Returns:
        dict: Window RF analysis JSON structure

    Source: Q&A Q4 (Output Schema), Q2 (Model file paths)
    """
    # ===== 1. Load Window-Level RF Model =====
    model_path = os.path.join(bucket_path, f'models/rf_{window}_{bucket}.pkl')
    rf_model = joblib.load(model_path)

    # ===== 2. Extract Feature Importance =====
    feature_importances = rf_model.feature_importances_  # Always 21 features per window
    feature_names = rf_model.feature_names_in_  # e.g., ['eye_contact_rate', 'scene_count', ...]

    # Sort by importance, take top 10
    importance_indices = np.argsort(feature_importances)[::-1][:10]
    top_features = [
        {
            'feature': feature_names[idx],
            'importance': float(feature_importances[idx]),
            'rank': rank + 1
        }
        for rank, idx in enumerate(importance_indices)
    ]

    # ===== 3. Load model_metrics.json for Performance Stats =====
    metrics_path = os.path.join(bucket_path, 'models/model_metrics.json')
    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    # Extract metrics for this window
    window_metrics = all_metrics.get(f'rf_{window}', {})

    # ===== 4. Compute Distribution Stats (Same as Video-Level, but per window) =====
    # Load window-specific transformed CSV
    rf_csv_path = os.path.join(bucket_path, f'ml_analysis/{window}_rf_transformed.csv')
    df = pd.read_csv(rf_csv_path)

    # Determine top/bottom performers
    video_count = len(df)
    top_count = int(video_count * 0.8)
    df['is_top_performer'] = [1] * top_count + [0] * (video_count - top_count)

    # Compute distribution stats for each top feature
    for feature_data in top_features:
        feature_name = feature_data['feature']

        if feature_name not in df.columns:
            # Feature not in CSV (shouldn't happen for window-level)
            feature_data['top_performer_avg'] = None
            feature_data['bottom_performer_avg'] = None
            feature_data['gap'] = None
            continue

        top_performers = df[df['is_top_performer'] == 1][feature_name]
        bottom_performers = df[df['is_top_performer'] == 0][feature_name]

        feature_data['top_performer_avg'] = float(top_performers.mean())
        feature_data['bottom_performer_avg'] = float(bottom_performers.mean())
        feature_data['gap'] = abs(feature_data['top_performer_avg'] - feature_data['bottom_performer_avg'])

    # ===== 5. Build Window RF Analysis JSON =====
    analysis_json = {
        'model_type': 'window_level_rf',
        'window_type': window,
        'bucket': bucket,
        'total_videos': video_count,
        'input_features': len(feature_names),  # Always 21
        'model_performance': {
            'accuracy': window_metrics.get('accuracy', None),
            'precision': window_metrics.get('precision', None),
            'recall': window_metrics.get('recall', None)
        },
        'feature_importance': top_features
    }

    return analysis_json
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| model_metrics.json missing performance stats for window | Set accuracy/precision/recall to `None` | Non-critical - feature importance still available for LLM |
| Window has <21 features (buckets 9-13s, 13-18s use middle_aggregate) | Use actual feature count | Valid configuration - distribution varies by bucket |

#### Step 2.3.4: Generate Window-Level K-Means JSONs

**Purpose**: Extract cluster centroids with NORMALIZED feature names for Stage 7 Phase 1 cluster interpretation

**Logic**:
```python
def normalize_feature_name(feature_name: str) -> str:
    """
    Normalize K-Means feature names for consistency with RF feature names.

    Removes transformation suffixes from Stage 4:
    - '_scaled' (from MinMax scaling)
    - '_log' (from log transformation - intermediate, usually removed)
    - '_encoded' (from label encoding)

    Args:
        feature_name: str, e.g., 'eye_contact_rate_scaled'

    Returns:
        str, e.g., 'eye_contact_rate'

    Source: Q&A Q6 (Feature Name Normalization)
    """
    normalized = feature_name

    # Remove suffixes in order (some features may have multiple)
    suffixes = ['_scaled', '_log', '_encoded']
    for suffix in suffixes:
        normalized = normalized.replace(suffix, '')

    return normalized


def generate_window_kmeans_json(bucket_path: str, bucket: str, window: str) -> dict:
    """
    Generate Window-Level K-Means analysis JSON with normalized feature names.

    Args:
        bucket_path: Absolute path to bucket directory
        bucket: Bucket name (e.g., "18-33s")
        window: Window name (e.g., "hook", "middle_1", "closing")

    Returns:
        dict: Window K-Means analysis JSON structure

    Source: Q&A Q4 (Output Schema), Q6 (Feature Normalization - CRITICAL)
    """
    # ===== 1. Load K-Means Model =====
    model_path = os.path.join(bucket_path, f'models/{window}_kmeans_{bucket}.pkl')
    kmeans_model = joblib.load(model_path)

    # ===== 2. Extract Cluster Centroids =====
    centroids = kmeans_model.cluster_centers_  # Shape: (3 clusters, 21-39 features)
    n_clusters = centroids.shape[0]  # Always 3
    n_features = centroids.shape[1]  # 21-39 depending on window

    # ===== 3. Load Feature Names from X_data (Stage 5 output) =====
    x_data_path = os.path.join(bucket_path, f'models/{window}_X_data_{bucket}.pkl')
    X = joblib.load(x_data_path)

    # X is either DataFrame or numpy array - extract feature names
    if hasattr(X, 'columns'):
        # X is DataFrame
        feature_names = X.columns.tolist()  # e.g., ['eye_contact_rate_scaled', 'scene_count_scaled', ...]
    else:
        # X is numpy array - load from K-Means transformed CSV as fallback
        km_csv_path = os.path.join(bucket_path, f'ml_analysis/{window}_km_transformed.csv')
        df_km = pd.read_csv(km_csv_path, nrows=1)  # Just read header
        feature_names = df_km.columns.tolist()

    # ===== 4. Load K-Means Predictions (Cluster Assignments) =====
    km_csv_path = os.path.join(bucket_path, f'ml_analysis/{window}_km_transformed.csv')
    df_km = pd.read_csv(km_csv_path)

    # Predict cluster assignments
    cluster_labels = kmeans_model.predict(df_km[feature_names])

    # ===== 5. Build Clusters with NORMALIZED Feature Names =====
    clusters = []

    for cluster_id in range(n_clusters):
        # Get centroid values
        centroid_values = centroids[cluster_id]

        # CRITICAL: Normalize feature names before creating centroid dict
        # This ensures K-Means features match RF feature names in Stage 7 LLM prompts
        normalized_centroid = {
            normalize_feature_name(name): float(value)
            for name, value in zip(feature_names, centroid_values)
        }

        # Get videos in this cluster
        cluster_videos = df_km[cluster_labels == cluster_id]
        video_ids = cluster_videos.index.tolist()  # Use index as video ID

        # Compute distance to centroid for each video
        videos_list = []
        for video_idx in video_ids:
            video_features = df_km.loc[video_idx][feature_names].values
            distance = np.linalg.norm(video_features - centroid_values)
            videos_list.append({
                'video_id': f'video_{video_idx}',
                'distance_to_centroid': float(distance)
            })

        # Build cluster object
        clusters.append({
            'cluster_id': cluster_id,
            'size': len(video_ids),
            'centroid': normalized_centroid,
            'videos': videos_list
        })

    # ===== 6. Build K-Means Analysis JSON =====
    analysis_json = {
        'window_type': window,
        'bucket': bucket,
        'total_videos': len(df_km),
        'n_clusters': n_clusters,
        'clusters': clusters
    }

    return analysis_json
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Feature already normalized (no suffix) | `normalize_feature_name('scene_count')` → `'scene_count'` (no change) | Valid - some features don't need normalization |
| Multiple suffixes (rare) | `normalize_feature_name('word_count_log_scaled')` → `'word_count'` (both removed) | Edge case handled by sequential replacement |
| X_data is numpy array without feature names | Fallback to loading `{window}_km_transformed.csv` header | Ensures feature names always available |

#### Step 2.3.5: Atomic Output Commit

**Purpose**: Ensure all 13 JSONs succeed or all deleted (no partial output)

**Logic**:
```python
def generate_ml_analysis_jsons(bucket_path: str, bucket: str, windows: list) -> int:
    """
    Generate all ML analysis JSONs using atomic pattern.
    Either all 13 JSONs succeed or all deleted.

    Args:
        bucket_path: Absolute path to bucket directory
        bucket: Bucket name (e.g., "18-33s")
        windows: Window list (e.g., ['hook', 'middle_1', ..., 'closing'])

    Returns:
        int: Exit code (0=success, 1=pre-flight fail, 2=generation fail, 3=validation fail, 4=I/O fail)

    Source: Q&A Q5 (Atomic Output Pattern)
    """
    temp_dir = os.path.join(bucket_path, 'ml_analysis/.tmp/')
    os.makedirs(temp_dir, exist_ok=True)
    generated_files = []

    try:
        # ===== 1. PRE-FLIGHT VALIDATION =====
        logger.info("Pre-flight validation: checking Stage 4 and Stage 5 outputs...")
        validate_stage_dependencies(bucket_path, bucket, windows)
        logger.info("✓ Pre-flight validation passed")

        # ===== 2. GENERATE ALL JSONs TO TEMP DIRECTORY =====
        logger.info("Generating ML analysis JSONs to temp directory...")

        # Generate Video-Level RF JSON
        video_rf_json = generate_video_rf_json(bucket_path, bucket)
        temp_path = os.path.join(temp_dir, 'rf_video_analysis.json.tmp')
        with open(temp_path, 'w') as f:
            json.dump(video_rf_json, f, indent=2)
        generated_files.append(temp_path)
        logger.info(f"  ✓ Generated rf_video_analysis.json.tmp")

        # Generate Window-Level RF JSONs
        for window in windows:
            window_rf_json = generate_window_rf_json(bucket_path, bucket, window)
            temp_path = os.path.join(temp_dir, f'{window}_rf_analysis.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(window_rf_json, f, indent=2)
            generated_files.append(temp_path)
            logger.info(f"  ✓ Generated {window}_rf_analysis.json.tmp")

        # Generate Window-Level K-Means JSONs
        for window in windows:
            window_km_json = generate_window_kmeans_json(bucket_path, bucket, window)
            temp_path = os.path.join(temp_dir, f'{window}_kmeans_analysis.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(window_km_json, f, indent=2)
            generated_files.append(temp_path)
            logger.info(f"  ✓ Generated {window}_kmeans_analysis.json.tmp")

        logger.info(f"✓ All {len(generated_files)} JSONs generated to temp directory")

        # ===== 3. VALIDATE ALL JSON SCHEMAS =====
        logger.info("Validating JSON schemas...")
        validate_all_json_schemas(temp_dir, bucket, windows)
        logger.info("✓ All JSON schemas valid")

        # ===== 4. ATOMIC COMMIT: RENAME ALL TEMP FILES AT ONCE =====
        logger.info("Committing JSONs (atomic rename)...")
        for temp_file in generated_files:
            final_path = temp_file.replace('/.tmp/', '/').replace('.json.tmp', '.json')
            os.rename(temp_file, final_path)
            logger.info(f"  ✓ Committed {os.path.basename(final_path)}")

        # ===== 5. CLEANUP TEMP DIRECTORY =====
        shutil.rmtree(temp_dir)
        logger.info(f"✓ Stage 6 complete: {len(generated_files)} JSONs generated")

        return 0  # SUCCESS

    except PreFlightValidationError as e:
        logger.error(f"Pre-flight validation failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 1  # PRE-FLIGHT VALIDATION FAILED

    except json.JSONDecodeError as e:
        logger.error(f"""
Stage 6 JSON generation failed: Invalid JSON
Exception: {type(e).__name__}: {str(e)}
Stack trace: {traceback.format_exc(limit=10)}
Generated files before failure: {len(generated_files)} of {1 + 2*len(windows)} expected
""")
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning("Rolled back: Deleted all temp files (atomic failure)")
        return 3  # JSON VALIDATION FAILED

    except IOError as e:
        logger.error(f"Disk I/O failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 4  # DISK I/O FAILED

    except Exception as e:
        logger.error(f"""
Stage 6 JSON generation failed
Exception: {type(e).__name__}: {str(e)}
Stack trace: {traceback.format_exc(limit=10)}
Generated files before failure: {len(generated_files)} of {1 + 2*len(windows)} expected
""")
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning("Rolled back: Deleted all temp files (atomic failure)")
        return 2  # JSON GENERATION FAILED
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Mid-generation crash (8 of 13 files created) | Delete all temp files, exit code 2 | Atomic pattern ensures no partial output |
| Disk full during JSON write | Delete all temp files, exit code 4 | Clear error message directs user to disk issue |
| Validation fails after all JSONs generated | Delete all temp files, exit code 3 | Prevent Stage 7 from loading malformed JSONs |

---

## 3. Dependencies & Integration

### 3.1 Input Dependencies

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **System setup** | FoundationCHILD.md (Section 2: Client Architecture, Section 6: Centralized Config) | Directory structure + `config/bucket_definitions.py` | `BUCKET_WINDOWS` config, `bucket_{bucket}/ml_analysis/` directories | Fail-fast if directories don't exist or config missing |
| Stage 5 Video RF model | Stage 5 output | PKL file: `models/rf_video_{bucket}.pkl` | `feature_importances_`, `feature_names_in_` attributes | Pre-flight validation fails (exit code 1) |
| Stage 5 Window RF models | Stage 5 output | PKL files: `models/rf_{window}_{bucket}.pkl` (6-7 files) | `feature_importances_`, `feature_names_in_` attributes | Pre-flight validation fails |
| Stage 5 K-Means models | Stage 5 output | PKL files: `models/{window}_kmeans_{bucket}.pkl` (6-7 files) | `cluster_centers_` attribute, `predict()` method | Pre-flight validation fails |
| Stage 5 Scalers | Stage 5 output | PKL files: `models/{window}_scalers_{bucket}.pkl` (6-7 files) | Not directly used by Stage 6, validated for completeness | Pre-flight validation fails |
| Stage 5 X matrices | Stage 5 output | PKL files: `models/{window}_X_data_{bucket}.pkl` (6-7 files) | Feature names (DataFrame columns or load from CSV fallback) | Pre-flight validation fails |
| Stage 5 Model metrics | Stage 5 output | JSON file: `models/model_metrics.json` | `accuracy`, `precision`, `recall` per window | Pre-flight validation fails |
| Stage 4 Aggregated CSV | Stage 4 output | CSV file: `ml_analysis/aggregated_features.csv` (~15-20 MB) | All feature columns + `video_id`, `create_time`, `gender` metadata | Pre-flight validation fails |
| Stage 4 RF Transformed CSV | Stage 4 output | CSV file: `ml_analysis/rf_transformed.csv` | Video-level features (178 columns for bucket 18-33s) | Pre-flight validation fails |
| Stage 4 Window RF CSVs | Stage 4 output | CSV files: `ml_analysis/{window}_rf_transformed.csv` (6-7 files) | Window-level features (21 columns per window) | Pre-flight validation fails |
| Stage 4 Window K-Means CSVs | Stage 4 output | CSV files: `ml_analysis/{window}_km_transformed.csv` (6-7 files) | Transformed features with `_scaled`, `_log`, `_encoded` suffixes (21-39 columns) | Pre-flight validation fails |

### 3.2 Output Contracts

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| Video RF JSON | JSON: `ml_analysis/rf_video_analysis.json` (~30KB) | `analysis_type` (str), `bucket` (str), `hashtag` (str/null), `video_count` (int), `input_features` (int), `feature_importance` (array of 10 objects with `feature`, `importance`, `top_performer_avg`, `bottom_performer_avg`, `gap`, `distribution`) | Stage 7 Phase 2 (Cross-window synthesis) | Check `feature_importance` has 10 entries, all fields present |
| Window RF JSONs | JSON: `ml_analysis/{window}_rf_analysis.json` (~5KB each, 6-7 files) | `model_type` (str), `window_type` (str), `bucket` (str), `total_videos` (int), `input_features` (int=21), `model_performance` (accuracy/precision/recall), `feature_importance` (array of 10 objects with `feature`, `importance`, `top_performer_avg`, `bottom_performer_avg`, `gap`, `rank`) | Stage 7 Phase 1 (Per-window analysis) | Check `input_features` = 21, all fields present |
| Window K-Means JSONs | JSON: `ml_analysis/{window}_kmeans_analysis.json` (~5KB each, 6-7 files) | `window_type` (str), `bucket` (str), `total_videos` (int), `n_clusters` (int=3), `clusters` (array of 3 objects with `cluster_id`, `size`, `centroid` (dict with 21-39 features), `videos` (array with `video_id`, `distance_to_centroid`)) | Stage 7 Phase 1 (Cluster interpretation) | Check 3 clusters, centroid features normalized (no `_scaled` suffixes), sum of cluster sizes = total_videos |

### 3.3 Cross-Stage Dependencies

**This feature depends on**:
- **Stage 4 (Feature Transformation)**: Must complete successfully - 13 transformed CSVs required for distribution analysis and K-Means cluster assignments
- **Stage 5 (ML Model Training)**: Must complete successfully - 90 trained models (8 Video RF + 41 Window RF + 41 Window K-Means) required for insight extraction

**This feature is required by**:
- **Stage 7 (LLM Creative Reports)**: Expects all 13 JSON files per bucket in exact schema format. Phase 1 loads window-level JSONs (RF + K-Means), Phase 2 loads video-level RF JSON + Phase 1 outputs.

**Failure Impact**:
- If this stage fails: Stage 7 cannot run (no structured insights for LLM)
- Checkpoint: Resume from Stage 6 without re-running Stages 4-5 (atomic pattern prevents partial output)

### 3.4 External Dependencies

**Python Libraries**:
```python
import pandas as pd  # 2.0.0+
import numpy as np  # 1.24.0+
import joblib  # 1.3.0+ (for loading pickle files)
import json  # stdlib
import os  # stdlib
import shutil  # stdlib
import logging  # stdlib
import traceback  # stdlib
```

**File System**:
- Read access: `/data/clients/{client_id}/buckets/{bucket}/models/`, `/data/clients/{client_id}/buckets/{bucket}/ml_analysis/`
- Write access: `/data/clients/{client_id}/buckets/{bucket}/ml_analysis/` (for JSON outputs and temp directory)

**Environment Variables**:
- `DATA_ROOT`: Root directory for client data (default: `/data`) - from FoundationCHILD.md
- `LOG_LEVEL`: Logging verbosity (default: `INFO`)

**External Services**: None (pure computational stage)

---

## 4. Configuration & Parameters

### 4.1 CLI Parameters (if applicable)

This component uses configuration from FoundationCHILD.md Section 4 (CLI Command Structure). No Stage 6-specific CLI parameters - all configuration inherited from earlier stages.

| Parameter | Type | Default | Valid Values | Impact | Source |
|-----------|------|---------|--------------|--------|--------|
| `--bucket` | str | Required | `0-3s`, `3-9s`, `9-13s`, `13-18s`, `18-33s`, `33-60s`, `60-90s`, `90-120s` | Determines which bucket directory to process | FoundationCHILD.md Section 4 |
| `--client` | str | Required | Any string | Determines client directory path | FoundationCHILD.md Section 4 |

### 4.2 Internal Configuration

```python
# Bucket window configuration (centralized)
from config.bucket_definitions import BUCKET_WINDOWS
# BUCKET_WINDOWS = {
#     '0-3s': ['hook'],
#     '3-9s': ['hook', 'closing'],
#     '9-13s': ['hook', 'middle_aggregate', 'closing'],
#     '13-18s': ['hook', 'middle_aggregate', 'closing'],
#     '18-33s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing'],
#     '33-60s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
#     '60-90s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
#     '90-120s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
# }

# Distribution analysis parameters
TOP_PERFORMER_PERCENTAGE = 0.8  # Top 80% vs bottom 20%
HIGH_PERCENTILE = 0.66  # 66th percentile threshold
LOW_PERCENTILE = 0.33  # 33rd percentile threshold

# Feature importance limits
MAX_FEATURES_VIDEO_RF = 10  # Top 10 features for video-level RF
MAX_FEATURES_WINDOW_RF = 10  # Top 10 features for window-level RF

# K-Means parameters
N_CLUSTERS = 3  # Always 3 clusters per window (from Stage 5)

# File paths (relative to bucket directory)
MODELS_DIR = "models"
ML_ANALYSIS_DIR = "ml_analysis"
TEMP_DIR = "ml_analysis/.tmp"

# JSON output filenames
VIDEO_RF_JSON = "rf_video_analysis.json"
WINDOW_RF_JSON_TEMPLATE = "{window}_rf_analysis.json"
WINDOW_KMEANS_JSON_TEMPLATE = "{window}_kmeans_analysis.json"
```

---

## 5. Data Schemas

### 5.1 Input Schema

**File 1**: `ml_analysis/aggregated_features.csv` (from Stage 4)

**Purpose**: Source for distribution analysis (percentile thresholds, high/medium/low classifications)

**Total Columns by Bucket**:
- 0-3s: 24 columns (1 window × 21 features + 3 metadata)
- 3-9s: 45 columns (2 windows × 21 features + 3 metadata)
- 9-13s, 13-18s: 66 columns (3 windows × 21 features + 3 metadata) - middle_aggregate used
- 18-33s: 129 columns (6 windows × 21 features + 3 metadata)
- 33-60s, 60-90s, 90-120s: 150 columns (7 windows × 21 features + 3 metadata)

**Sample Columns** (bucket 18-33s):
| Column | Type | Range | Nulls? | Description | Example |
|--------|------|-------|--------|-------------|---------|
| `video_id` | str | - | No | Unique video identifier | `"7428596413707144481"` |
| `create_time` | datetime | - | No | Video publish timestamp | `"2025-01-15 14:30:00"` |
| `gender` | str | `male`, `female`, `null` | Yes | Detected gender (optional metadata) | `"female"` |
| `hook_scene_count` | int | 0-20 | No | Scene cuts in hook window (0-3s) | `3` |
| `hook_eye_contact_rate` | float | 0.0-1.0 | No | Eye contact proportion in hook | `0.85` |
| `hook_word_count` | int | 0-200 | No | Words spoken in hook | `14` |
| `middle_1_scene_count` | int | 0-20 | No | Scene cuts in middle segment 1 | `5` |
| `middle_1_word_count` | int | 0-200 | No | Words in middle segment 1 | `48` |
| `closing_energy_level` | float | 0.0-1.0 | No | Audio energy in closing window | `0.75` |

**File 2**: `models/rf_video_{bucket}.pkl` (from Stage 5)

**Purpose**: Video-level RF model for cross-window feature importance

**Attributes**:
- `feature_importances_`: NumPy array (length = 178 for bucket 18-33s)
- `feature_names_in_`: NumPy array of strings (feature names)

**File 3**: `models/rf_{window}_{bucket}.pkl` (from Stage 5, 6-7 files per bucket)

**Purpose**: Window-level RF models for per-window feature importance

**Attributes**:
- `feature_importances_`: NumPy array (length = 21 per window)
- `feature_names_in_`: NumPy array of strings (window feature names, no prefix)

**File 4**: `models/{window}_kmeans_{bucket}.pkl` (from Stage 5, 6-7 files per bucket)

**Purpose**: K-Means models for cluster centroids

**Attributes**:
- `cluster_centers_`: NumPy array (shape: 3 clusters × 21-39 features)
- `predict()`: Method for assigning videos to clusters

**File 5**: `models/{window}_X_data_{bucket}.pkl` (from Stage 5, 6-7 files per bucket)

**Purpose**: Feature matrix for extracting feature names

**Format**: DataFrame or NumPy array with feature names (e.g., `eye_contact_rate_scaled`, `scene_count_scaled`)

**File 6**: `ml_analysis/{window}_km_transformed.csv` (from Stage 4, 6-7 files per bucket)

**Purpose**: Transformed K-Means features for cluster assignment and distance computation

**Columns**: 21-39 features with transformation suffixes (`_scaled`, `_log`, `_encoded`)

### 5.2 Output Schema

**File 1**: `ml_analysis/rf_video_analysis.json` (Video-Level RF)

**Complete Structure**:
```json
{
  "analysis_type": "random_forest",
  "bucket": "18-33s",
  "hashtag": null,
  "video_count": 100,
  "input_features": 178,
  "feature_importance": [
    {
      "feature": "hook_eye_contact_rate",
      "importance": 0.22,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "distribution": {
        "thresholds": {
          "high": 0.6,
          "low": 0.4
        },
        "top_performers": {
          "high_percentage": 0.70,
          "medium_percentage": 0.25,
          "low_percentage": 0.05
        },
        "bottom_performers": {
          "high_percentage": 0.05,
          "medium_percentage": 0.15,
          "low_percentage": 0.80
        }
      }
    }
    // ... top 10 features (each with distribution data)
  ]
}
```

**Field Details**:
| Field | Type | Range | Required? | Description |
|-------|------|-------|-----------|-------------|
| `analysis_type` | str | Fixed: `"random_forest"` | Yes | Identifies JSON type |
| `bucket` | str | `"18-33s"`, etc. | Yes | Duration bucket |
| `hashtag` | str/null | Any | No | Hashtag analyzed (set by caller) |
| `video_count` | int | 50-300 | Yes | Total videos in bucket |
| `input_features` | int | 24-220 | Yes | Feature count (varies by bucket, includes 5 cross-window features) |
| `feature_importance` | array | Length 10 | Yes | Top 10 features with stats |
| `feature_importance[].feature` | str | Feature name | Yes | e.g., `"hook_eye_contact_rate"` |
| `feature_importance[].importance` | float | 0.0-1.0 | Yes | RF importance score |
| `feature_importance[].top_performer_avg` | float | Varies | Yes | Mean value in top 80% |
| `feature_importance[].bottom_performer_avg` | float | Varies | Yes | Mean value in bottom 20% |
| `feature_importance[].gap` | float | 0.0+ | Yes | Absolute difference between top/bottom |
| `feature_importance[].distribution` | object | - | Yes | Percentile analysis |
| `distribution.thresholds.high` | float | Varies | Yes | 66th percentile value |
| `distribution.thresholds.low` | float | Varies | Yes | 33rd percentile value |
| `distribution.top_performers.high_percentage` | float | 0.0-1.0 | Yes | % of top performers ≥ high threshold |
| `distribution.top_performers.medium_percentage` | float | 0.0-1.0 | Yes | % of top performers in medium range |
| `distribution.top_performers.low_percentage` | float | 0.0-1.0 | Yes | % of top performers < low threshold |
| `distribution.bottom_performers.*` | float | 0.0-1.0 | Yes | Same structure as top_performers |

**File 2**: `ml_analysis/{window}_rf_analysis.json` (Window-Level RF)

**Complete Structure**:
```json
{
  "model_type": "window_level_rf",
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "input_features": 21,
  "model_performance": {
    "accuracy": 0.82,
    "precision": 0.85,
    "recall": 0.78
  },
  "feature_importance": [
    {
      "feature": "eye_contact_rate",
      "importance": 0.35,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "rank": 1
    }
    // ... top 10 features
  ]
}
```

**Field Details**:
| Field | Type | Range | Required? | Description |
|-------|------|-------|-----------|-------------|
| `model_type` | str | Fixed: `"window_level_rf"` | Yes | Identifies JSON type |
| `window_type` | str | `"hook"`, `"middle_1"`, `"closing"`, etc. | Yes | Window name |
| `bucket` | str | Bucket name | Yes | Duration bucket |
| `total_videos` | int | 50-300 | Yes | Total videos |
| `input_features` | int | 21 | Yes | Always 21 features per window |
| `model_performance.accuracy` | float | 0.0-1.0 | Yes | Model accuracy from Stage 5 metrics |
| `model_performance.precision` | float | 0.0-1.0 | Yes | Model precision |
| `model_performance.recall` | float | 0.0-1.0 | Yes | Model recall |
| `feature_importance` | array | Length 10 | Yes | Top 10 features |
| `feature_importance[].feature` | str | Feature name (no window prefix) | Yes | e.g., `"eye_contact_rate"` NOT `"hook_eye_contact_rate"` |
| `feature_importance[].importance` | float | 0.0-1.0 | Yes | RF importance score |
| `feature_importance[].top_performer_avg` | float | Varies | Yes | Mean value in top 80% |
| `feature_importance[].bottom_performer_avg` | float | Varies | Yes | Mean value in bottom 20% |
| `feature_importance[].gap` | float | 0.0+ | Yes | Absolute difference |
| `feature_importance[].rank` | int | 1-10 | Yes | Rank within top 10 |

**File 3**: `ml_analysis/{window}_kmeans_analysis.json` (Window-Level K-Means)

**Complete Structure**:
```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "n_clusters": 3,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "centroid": {
        "eye_contact_rate": 0.87,
        "scene_count": 0.45,
        "word_count": 0.62,
        "energy_level": 0.73
        // ... all 21-39 features (NORMALIZED - no _scaled suffixes)
      },
      "videos": [
        {
          "video_id": "video_0",
          "distance_to_centroid": 0.15
        }
        // ... all videos in cluster
      ]
    },
    {
      "cluster_id": 1,
      "size": 42,
      "centroid": { /* ... */ },
      "videos": [ /* ... */ ]
    },
    {
      "cluster_id": 2,
      "size": 23,
      "centroid": { /* ... */ },
      "videos": [ /* ... */ ]
    }
  ]
}
```

**Field Details**:
| Field | Type | Range | Required? | Description |
|-------|------|-------|-----------|-------------|
| `window_type` | str | Window name | Yes | e.g., `"hook"`, `"middle_1"` |
| `bucket` | str | Bucket name | Yes | Duration bucket |
| `total_videos` | int | 50-300 | Yes | Total videos |
| `n_clusters` | int | 3 | Yes | Always 3 clusters |
| `clusters` | array | Length 3 | Yes | Cluster data |
| `clusters[].cluster_id` | int | 0, 1, 2 | Yes | Cluster identifier |
| `clusters[].size` | int | 1-300 | Yes | Videos in cluster |
| `clusters[].centroid` | object | 21-39 keys | Yes | Mean feature values for cluster |
| `clusters[].centroid.*` | float | 0.0-1.0 (scaled) | Yes | NORMALIZED feature names (no `_scaled` suffix) |
| `clusters[].videos` | array | Length = size | Yes | Videos in cluster |
| `clusters[].videos[].video_id` | str | `"video_N"` | Yes | Video identifier |
| `clusters[].videos[].distance_to_centroid` | float | 0.0+ | Yes | Euclidean distance to centroid |

---

## 6. Error Handling & Validation

### 6.1 Input Validation

```python
def validate_stage_dependencies(bucket_path: str, bucket: str, windows: list) -> None:
    """
    Validate Stage 4 and Stage 5 outputs exist before generating any JSONs.

    Args:
        bucket_path: Absolute path to bucket directory
        bucket: Bucket name (e.g., "18-33s")
        windows: Window list (e.g., ['hook', 'middle_1', ..., 'closing'])

    Raises:
        PreFlightValidationError: If any required file missing

    Source: Q&A Q5 (Atomic Output Pattern - Pre-Flight Validation)
    """
    missing_files = []

    # ===== Validate Stage 4 CSVs =====
    required_stage4_files = [
        'ml_analysis/aggregated_features.csv',
        'ml_analysis/rf_transformed.csv',
        *[f'ml_analysis/{w}_rf_transformed.csv' for w in windows],
        *[f'ml_analysis/{w}_km_transformed.csv' for w in windows]
    ]

    for file_path in required_stage4_files:
        full_path = os.path.join(bucket_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(('Stage 4', file_path))

    # ===== Validate Stage 5 Models =====
    required_stage5_files = [
        f'models/rf_video_{bucket}.pkl',
        *[f'models/rf_{w}_{bucket}.pkl' for w in windows],
        *[f'models/{w}_kmeans_{bucket}.pkl' for w in windows],
        *[f'models/{w}_scalers_{bucket}.pkl' for w in windows],
        *[f'models/{w}_X_data_{bucket}.pkl' for w in windows],
        'models/model_metrics.json'
    ]

    for file_path in required_stage5_files:
        full_path = os.path.join(bucket_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(('Stage 5', file_path))

    # ===== Fail-fast if any dependencies missing =====
    if missing_files:
        stage4_missing = [f for s, f in missing_files if s == 'Stage 4']
        stage5_missing = [f for s, f in missing_files if s == 'Stage 5']

        error_msg = "Pre-flight validation failed:\n"
        if stage4_missing:
            error_msg += f"Stage 4 incomplete ({len(stage4_missing)} files missing):\n"
            error_msg += "\n".join(f"  - {f}" for f in stage4_missing[:5])
            if len(stage4_missing) > 5:
                error_msg += f"\n  ... and {len(stage4_missing)-5} more"
            error_msg += "\nAction: Re-run Stage 4 (Feature Transformation)\n"

        if stage5_missing:
            error_msg += f"Stage 5 incomplete ({len(stage5_missing)} files missing):\n"
            error_msg += "\n".join(f"  - {f}" for f in stage5_missing[:5])
            if len(stage5_missing) > 5:
                error_msg += f"\n  ... and {len(stage5_missing)-5} more"
            error_msg += "\nAction: Re-run Stage 5 (ML Model Training)\n"

        raise PreFlightValidationError(error_msg)
```

### 6.2 Error Cases

| Error | Detection | Handling | User Message | Exit Code |
|-------|-----------|----------|--------------|-----------|
| Missing Stage 4 CSV | `os.path.exists()` check during pre-flight | Fail-fast before generating any JSONs | `"Stage 4 incomplete (N files missing): [list]. Action: Re-run Stage 4 (Feature Transformation)"` | 1 |
| Missing Stage 5 model | `os.path.exists()` check during pre-flight | Fail-fast before generating any JSONs | `"Stage 5 incomplete (N files missing): [list]. Action: Re-run Stage 5 (ML Model Training)"` | 1 |
| Corrupted pickle file | `joblib.load()` exception | Delete temp directory, fail | `"Failed to load model {path}: {error}. Model may be corrupted. Re-run Stage 5."` | 2 |
| Invalid CSV format | `pd.read_csv()` exception | Delete temp directory, fail | `"Failed to parse CSV {path}: {error}. Check file is valid CSV format."` | 2 |
| Mid-generation crash (8 of 13 JSONs created) | Exception caught in atomic pattern | Delete all temp files | `"Stage 6 JSON generation failed. Exception: {error}. Generated files before failure: 8 of 13 expected. Rolled back: Deleted all temp files (atomic failure)."` | 2 |
| JSON validation fails (invalid schema) | Post-generation validation | Delete all temp files | `"Output validation failed: {error}. All temp files deleted."` | 3 |
| Disk full during JSON write | `IOError` exception | Delete temp directory | `"Disk I/O failed: {error}. Check disk space."` | 4 |
| Feature name mismatch (K-Means vs RF) | Post-generation validation | Delete all temp files, fail | `"Feature name consistency check failed: K-Means JSON contains {count} features with '_scaled' suffixes. Bug in normalization logic."` | 3 |

### 6.3 Output Validation

```python
def validate_all_json_schemas(temp_dir: str, bucket: str, windows: list) -> None:
    """
    Validate all generated JSONs before atomic commit.

    Args:
        temp_dir: Path to temp directory with .json.tmp files
        bucket: Bucket name
        windows: Window list

    Raises:
        ValidationError: If any JSON invalid

    Source: Q&A Q5 (Atomic Output Pattern - Output Validation)
    """
    # ===== 1. Check all expected files exist =====
    expected_files = [
        'rf_video_analysis.json.tmp',
        *[f'{w}_rf_analysis.json.tmp' for w in windows],
        *[f'{w}_kmeans_analysis.json.tmp' for w in windows]
    ]

    for filename in expected_files:
        filepath = os.path.join(temp_dir, filename)
        if not os.path.exists(filepath):
            raise ValidationError(f"Missing temp file: {filename}")

    # ===== 2. Validate JSON parseability =====
    for filename in expected_files:
        filepath = os.path.join(temp_dir, filename)
        try:
            with open(filepath, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in {filename}: {e}")

    # ===== 3. Validate K-Means feature names (no _scaled suffixes) =====
    for window in windows:
        km_filename = f'{window}_kmeans_analysis.json.tmp'
        km_filepath = os.path.join(temp_dir, km_filename)

        with open(km_filepath, 'r') as f:
            km_json = json.load(f)

        # Check centroid feature names
        for cluster in km_json['clusters']:
            centroid = cluster['centroid']
            invalid_features = [
                feat for feat in centroid.keys()
                if '_scaled' in feat or '_log' in feat or '_encoded' in feat
            ]

            if invalid_features:
                raise ValidationError(
                    f"K-Means JSON {km_filename} contains {len(invalid_features)} features with transformation suffixes: {invalid_features[:5]}. "
                    f"Feature normalization failed. Check normalize_feature_name() logic."
                )

    # ===== 4. Validate cluster size consistency =====
    for window in windows:
        km_filename = f'{window}_kmeans_analysis.json.tmp'
        km_filepath = os.path.join(temp_dir, km_filename)

        with open(km_filepath, 'r') as f:
            km_json = json.load(f)

        total_videos = km_json['total_videos']
        cluster_sizes = [cluster['size'] for cluster in km_json['clusters']]
        sum_sizes = sum(cluster_sizes)

        if sum_sizes != total_videos:
            raise ValidationError(
                f"K-Means JSON {km_filename} cluster sizes ({cluster_sizes}) sum to {sum_sizes}, "
                f"but total_videos is {total_videos}. Cluster assignment failed."
            )
```

---

## 7. Performance & Scalability

### 7.1 Performance Targets

- **Throughput**: Process 1 bucket in 3-5 seconds (N=100 videos)
- **Memory**: Peak usage < 200MB per bucket
- **Disk I/O**: < 2 seconds for loading models + CSVs
- **JSON Generation**: < 1 second for writing 13 files (~95KB total)

**Comparison with Other Stages** (for context):
- Stage 5 (ML Training): 30-90 seconds per bucket
- **Stage 6 (Analysis Generation)**: 3-5 seconds per bucket (12x faster - no training)
- Stage 7 (LLM Reports): 25-30 seconds per bucket

### 7.2 Measured Performance

**Expected Performance** (bucket 18-33s with N=100 videos):

| Metric | Target | Acceptable | Warning Threshold | Notes |
|--------|--------|------------|-------------------|-------|
| Total time | 3-5s | 5-10s | >30s | SSD storage assumed |
| Memory peak | 100-150 MB | 150-200 MB | >250 MB | Includes all models + CSVs loaded |
| Pre-flight validation | 0.5-1s | 1-2s | >5s | 33 file existence checks (HDD slower) |
| Model loading | 1-2s | 2-3s | >10s | 20 PKL files × ~50KB each |
| Distribution analysis | 1-2s | 2-3s | >5s | Percentile calculations on 100 videos |
| JSON generation | 0.5-1s | 1-2s | >5s | Write 13 files × ~5-30KB each |

### 7.3 Bottlenecks & Mitigations

**Potential Bottlenecks** (ranked by likelihood):

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| Distribution percentile calculations | 1-2s (acceptable) | NumPy quantile operations on 100 videos × 10 features | Use vectorized numpy operations (already fast) | LOW |
| Disk I/O for loading 20 PKL files | 1-2s (SSD), 5-10s (HDD) | 20 file reads × ~50KB each | Ensure models stored on SSD, or lazy-load only needed models | LOW (SSD), MEDIUM (HDD) |
| CSV loading for distribution analysis | 0.5-1s | pandas.read_csv with 100 rows × 129 columns | Load only needed columns (top 10 features), or cache CSV in memory | VERY LOW |
| JSON serialization | 0.5-1s | Python dict → JSON string conversion, disk writes | Write to temp directory first (atomic pattern already implemented) | VERY LOW |

**NOT bottlenecks**:
- ✅ Feature extraction from models (just attribute access: <0.1s)
- ✅ Feature name normalization (string replace: <0.1s)
- ✅ Validation logic (file existence checks: <0.1s)

### 7.4 Scalability Limits

**Tested Limits**:
- **Maximum**: N=200 videos per bucket (expected: ~5-8s, acceptable)
- **Minimum**: N=50 videos per bucket (expected: ~2-3s)

**Pipeline Context** (from Stage 5 HLD):
- **Total pipeline**: 3.6-4.8 hours for 300 videos
- **Stage 5 (training)**: ~1-2 minutes (~0.5-1% of total)
- **Stage 6 (analysis)**: ~3-5 seconds (~0.02% of total)
- **Stage 7 (LLM reports)**: ~25-30 seconds (~0.2% of total)
- **Bottleneck**: Stage 2 (video processing, 60-80s per video × 300 videos = 5-6.7 hours)

**Conclusion**: Stage 6 is NOT a bottleneck. Optimization NOT needed.

---

## 8. Testing Strategy

### 8.1 Unit Tests

- [ ] **Test pre-flight validation**
  - All dependencies exist (passes without error)
  - Single Stage 4 file missing (raises PreFlightValidationError with file name)
  - Single Stage 5 file missing (raises PreFlightValidationError with file name)
  - Both Stage 4 and Stage 5 incomplete (error message lists both)

- [ ] **Test feature name normalization**
  - `normalize_feature_name('eye_contact_rate_scaled')` → `'eye_contact_rate'`
  - `normalize_feature_name('has_captions_encoded')` → `'has_captions'`
  - `normalize_feature_name('scene_count')` → `'scene_count'` (no change)
  - `normalize_feature_name('word_count_log_scaled')` → `'word_count'` (multiple suffixes)

- [ ] **Test distribution analysis**
  - Compute 66th/33rd percentiles correctly
  - Handle variance=0 (all same value) → percentiles equal mean
  - Handle missing feature in CSV → distribution fields set to `None`

- [ ] **Test JSON generation**
  - Video RF JSON has 10 features with distribution data
  - Window RF JSON has 10 features with rank field
  - K-Means JSON has 3 clusters with normalized feature names

- [ ] **Test output validation**
  - All 13 files created (passes)
  - Missing window file (raises ValidationError)
  - K-Means JSON contains `_scaled` suffix (raises ValidationError)
  - Cluster sizes don't sum to total_videos (raises ValidationError)

### 8.2 Integration Tests

- [ ] **End-to-end: Stage 5 → Stage 6 → Stage 7**
  - Use real Stage 5 models (10 videos, bucket 18-33s)
  - Run Stage 6 analysis generation
  - Validate all 13 JSONs exist and parseable
  - Verify Stage 7 can load outputs without error

- [ ] **Atomic output pattern**
  - Simulate mid-generation crash (8 of 13 files created)
  - Verify all temp files deleted (no partial output)
  - Re-run Stage 6 → all 13 files generated successfully

- [ ] **Feature name consistency**
  - Load Stage 6 outputs (hook_rf_analysis.json + hook_kmeans_analysis.json)
  - Extract feature names from both JSONs
  - Verify overlap ≥ 15 features (high consistency)
  - Verify no `_scaled` suffixes in K-Means centroids

### 8.3 Test Data

**File 1**: `tests/fixtures/stage5_models_bucket_18-33s/` (20 files)

- Minimal Stage 5 models trained on 10 videos
- All required PKL files + model_metrics.json
- Used for unit tests of JSON generation

**File 2**: `tests/fixtures/stage4_csvs_bucket_18-33s/` (13 files)

- Minimal Stage 4 CSVs with 10 videos
- aggregated_features.csv (10 rows × 129 columns)
- rf_transformed.csv (10 rows × 190 columns)
- Window-level CSVs (10 rows × 21 columns each)

**Expected Output**: `tests/fixtures/expected_stage6_outputs/` (13 JSON files)

- rf_video_analysis.json (with distribution data)
- hook_rf_analysis.json, middle_1_rf_analysis.json, ..., closing_rf_analysis.json
- hook_kmeans_analysis.json, middle_1_kmeans_analysis.json, ..., closing_kmeans_analysis.json

### 8.4 Test Execution

```bash
# Run unit tests
pytest tests/unit/test_ml_analysis_generation.py -v

# Run integration tests
pytest tests/integration/test_stage6_integration.py -v

# Run with coverage
pytest tests/test_ml_analysis_generation.py --cov=ml_analysis_generation --cov-report=html

# Test feature normalization specifically
pytest tests/unit/test_feature_normalization.py -v -k "normalize"
```

---

## 9. Future Enhancements

### 9.1 Planned Improvements

**Phase 2: Lazy Model Loading**
- Current: Load all 20 models upfront (1-2s)
- Future: Load models on-demand per JSON generation (reduces memory, faster startup)
- Impact: Memory reduced from 100-150 MB to 30-50 MB peak

**Phase 3: Parallel JSON Generation**
- Current: Generate 13 JSONs sequentially (~3-5s total)
- Future: Generate Video RF, Window RF, Window K-Means in parallel (3 threads)
- Impact: Reduce to ~2-3s total (1.5-2x speedup)

**Phase 4: Distribution Caching**
- Current: Compute distribution stats per bucket (1-2s)
- Future: Cache percentile thresholds across buckets (reuse if video set unchanged)
- Impact: Faster re-runs after model updates (1-2s saved)

### 9.2 Known Limitations

**Manual JSON schema maintenance**: If Stage 7 updates expected schema, Stage 6 must be updated. No automated schema validation.

**No missing data imputation**: If aggregated_features.csv has nulls, distribution analysis fails. Could use mean/median imputation.

**Single-threaded**: JSON generation not parallelized (could use ThreadPoolExecutor for 1.5-2x speedup).

**No incremental updates**: Must regenerate all 13 JSONs even if only 1 model changed (atomic pattern trades efficiency for simplicity).

---

## 10. References & Related Docs

### 10.1 Parent Document

- **MLPlanningv2.md Section 6 "Stage 6: ML Analysis Generation" (Lines 1993-2388)**
  - High-level component overview
  - Stage position in pipeline (between Stage 5 training and Stage 7 LLM reports)
  - Tri-modal JSON output architecture (Video RF + Window RF + Window K-Means)

### 10.2 Mother Document Foundation

- **MLPlanningv2.md Part 1: Foundation** (shared across all stages)
  - Section 2 "Client Architecture": Provides `bucket_{bucket}/ml_analysis/` directory structure used for JSON outputs
  - Section 6 "Centralized Configuration": References `config/bucket_definitions.py` for window structure

**After Foundation Extraction**:
- **FoundationCHILD.md Section 2**: Client directory structure
- **FoundationCHILD.md Section 6**: Centralized bucket configuration (`BUCKET_WINDOWS`)

### 10.3 Related Child Docs

**Upstream Components**:
- **FeatureAggregationCHILD.md** (Stage 3)
  - Produces `aggregated_features.csv` (input to Stage 6 for distribution analysis)
  - Defines exact column names and temporal window structure (21 features per window)

- **FeatureTransformationCHILD.md** (Stage 4)
  - Produces 13 transformed CSVs per bucket (RF + K-Means transformed features)
  - Defines feature naming conventions (`_scaled`, `_log`, `_encoded` suffixes for K-Means)

- **Stage5_MLModelTraining_HLD.md** (Stage 5)
  - Produces 90 trained models (8 Video RF + 41 Window RF + 41 Window K-Means)
  - Defines model file naming conventions and storage structure
  - Section 3 (Feature Name Mismatch): CRITICAL warning about K-Means vs RF feature naming - addressed by Stage 6 normalization

**Downstream Components**:
- **LLMAnalysis7CHILD.md** (Stage 7) - PENDING GENERATION
  - Consumes all 13 JSON files per bucket (Video RF + Window RF + Window K-Means analysis)
  - Phase 1 loads window-level JSONs (RF + K-Means), Phase 2 loads video-level RF JSON + Phase 1 outputs
  - Depends on feature name consistency (K-Means normalized to match RF)

### 10.4 External References

- **Scikit-learn RandomForestClassifier**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  - `feature_importances_` attribute used for extracting importance scores
  - `feature_names_in_` attribute used for feature names

- **Scikit-learn KMeans**: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
  - `cluster_centers_` attribute used for extracting centroids
  - `predict()` method used for cluster assignments

- **Joblib**: https://joblib.readthedocs.io/en/latest/
  - Used for loading pickle files (models, scalers, X matrices)

- **NumPy quantile**: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
  - Used for computing 66th/33rd percentile thresholds in distribution analysis

### 10.5 Code References

- **Stage 5 normalization logic**: Stage5_MLModelTraining_HLD.md Section 3 (lines 358-401) - `normalize_feature_name()` function reused in Stage 6

- **Atomic output pattern**: Stage5_MLModelTraining_HLD.md Section 2.3.3 (lines 208-310) - Atomic bucket training pattern adapted for Stage 6 JSON generation

---

## Appendix A: Decision Log

**Purpose**: Record major design decisions, alternatives considered, and trade-offs accepted.

**Decision 1**: Distribution analysis performed in Stage 6 (not Stage 5)

- **Context**: Stage 6 computes feature distribution percentages (66th/33rd percentile), but Stage 5 already generates `model_metrics.json`. Should distribution analysis be part of model training (Stage 5) or insight extraction (Stage 6)?

- **Alternatives Considered**:
  - **Option A** (chosen): Compute distributions in Stage 6 during JSON generation
  - **Option B**: Compute distributions in Stage 5, store in model_metrics.json, Stage 6 reads pre-computed values

- **Rationale**:
  - Stage 5 focuses on ML training and outputs "summary metrics only" (accuracy, precision, recall, top feature)
  - Stage 6 handles "detailed insight extraction" including distribution analysis (Critique Q1 resolution)
  - Separation of concerns: Training (Stage 5) vs Insight Extraction (Stage 6)
  - Distribution analysis is statistical post-processing, not ML training

- **Trade-offs**:
  - Slight performance cost (1-2s per bucket for percentile calculations in Stage 6)
  - Cleaner separation of concerns (Stage 5 = ML only, Stage 6 = insights only)

- **Date**: 2025-01-28

**Decision 2**: 13 separate JSON files per bucket (not 3 unified files)

- **Context**: Stage 6 generates 13 JSON files per bucket (1 Video RF + 6 Window RF + 6 Window K-Means). Alternative was 3 unified files (1 RF, 1 Window RF, 1 K-Means).

- **Alternatives Considered**:
  - **Option A** (chosen): 13 separate files (1 per window type per analysis)
  - **Option B**: 3 unified files (all windows combined per analysis type)

- **Rationale**:
  - Stage 7 uses two-phase architecture with 6-7 parallel LLM calls in Phase 1 (one per window)
  - Separate files enable parallel execution (each thread loads independent window files)
  - Reduces hallucination risk: Phase 1 focuses on single window (113 numbers) vs unified file (1000+ numbers)
  - Chain-of-thought decomposition: Focused single-window analysis before cross-window synthesis
  - Debugging value: Phase 1 intermediate outputs inspectable per window

- **Trade-offs**:
  - More files to manage (13 vs 3)
  - Stage 7 loads 13 files sequentially (slightly more I/O)
  - Better LLM output quality and parallelization outweighs file management complexity

- **Date**: 2025-01-28 (Critique Q2 resolution)

**Decision 3**: Atomic output pattern (all 13 JSONs succeed or all deleted)

- **Context**: Stage 6 generates 13 JSON files. If crash occurs mid-generation (e.g., 8 of 13 files created), what happens?

- **Alternatives Considered**:
  - **Option A** (chosen): Atomic pattern - write to temp directory, validate all, rename all at once
  - **Option B**: Best-effort pattern - write directly to final location, keep partial output on failure
  - **Option C**: Checkpoint pattern - save progress after each JSON, resume from failure point

- **Rationale**:
  - Consistent with Stage 5's atomic bucket training pattern (architectural consistency)
  - Clean failure states: 0 files = clear signal (no ambiguity)
  - Simple recovery: User just re-runs Stage 6 (no partial state cleanup needed)
  - Low overhead: 13 JSONs × ~5KB = 65KB temp storage, ~5-10s generation time (acceptable to regenerate)

- **Trade-offs**:
  - Must regenerate all 13 JSONs on retry (no partial progress saved)
  - Acceptable: Stage 6 is fast (~3-5s), so regenerating is not a bottleneck

- **Date**: 2025-01-28 (QA Q5 decision)

**Decision 4**: Feature name normalization in Stage 6 (not Stage 4 or Stage 5)

- **Context**: K-Means features have transformation suffixes (`_scaled`, `_log`, `_encoded`) from Stage 4, but RF features don't. LLM in Stage 7 needs consistent feature names to correlate insights.

- **Alternatives Considered**:
  - **Option A** (chosen): Normalize in Stage 6 during K-Means JSON generation
  - **Option B**: Change Stage 4 to not add suffixes (break traceability)
  - **Option C**: Change Stage 5 to normalize during training (extra transformation step)

- **Rationale**:
  - Stage 4 suffixes are correct (indicate transformations applied - good for debugging)
  - Stage 5 should train on actual feature names from Stage 4 (no renaming during training)
  - Stage 6 is the "formatting for LLM consumption" layer - appropriate place for normalization
  - Reuses existing `normalize_feature_name()` function from Stage 5 (code reuse)

- **Trade-offs**:
  - Slight performance cost (<0.1s for string replacements)
  - Critical for Stage 7 LLM correlation (prevents "eye_contact_rate vs eye_contact_rate_scaled" confusion)

- **Date**: 2025-01-28 (QA Q6 decision)

---

## Appendix B: Example Data

### B.1 Sample Video-Level RF JSON (3 features shown)

**File**: `ml_analysis/rf_video_analysis.json`

```json
{
  "analysis_type": "random_forest",
  "bucket": "18-33s",
  "hashtag": "#nutrition",
  "video_count": 100,
  "input_features": 178,
  "feature_importance": [
    {
      "feature": "hook_eye_contact_rate",
      "importance": 0.22,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "distribution": {
        "thresholds": {
          "high": 0.6,
          "low": 0.4
        },
        "top_performers": {
          "high_percentage": 0.70,
          "medium_percentage": 0.25,
          "low_percentage": 0.05
        },
        "bottom_performers": {
          "high_percentage": 0.05,
          "medium_percentage": 0.15,
          "low_percentage": 0.80
        }
      }
    },
    {
      "feature": "middle_avg_word_count",
      "importance": 0.18,
      "top_performer_avg": 55.2,
      "bottom_performer_avg": 28.4,
      "gap": 26.8,
      "distribution": {
        "thresholds": {
          "high": 48.0,
          "low": 32.0
        },
        "top_performers": {
          "high_percentage": 0.65,
          "medium_percentage": 0.30,
          "low_percentage": 0.05
        },
        "bottom_performers": {
          "high_percentage": 0.10,
          "medium_percentage": 0.25,
          "low_percentage": 0.65
        }
      }
    },
    {
      "feature": "closing_energy_level",
      "importance": 0.15,
      "top_performer_avg": 0.75,
      "bottom_performer_avg": 0.42,
      "gap": 0.33,
      "distribution": {
        "thresholds": {
          "high": 0.65,
          "low": 0.45
        },
        "top_performers": {
          "high_percentage": 0.75,
          "medium_percentage": 0.20,
          "low_percentage": 0.05
        },
        "bottom_performers": {
          "high_percentage": 0.15,
          "medium_percentage": 0.25,
          "low_percentage": 0.60
        }
      }
    }
  ]
}
```

### B.2 Sample Window-Level RF JSON (3 features shown)

**File**: `ml_analysis/hook_rf_analysis.json`

```json
{
  "model_type": "window_level_rf",
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "input_features": 21,
  "model_performance": {
    "accuracy": 0.82,
    "precision": 0.85,
    "recall": 0.78
  },
  "feature_importance": [
    {
      "feature": "eye_contact_rate",
      "importance": 0.35,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "rank": 1
    },
    {
      "feature": "energy_level",
      "importance": 0.22,
      "top_performer_avg": 0.55,
      "bottom_performer_avg": 0.35,
      "gap": 0.20,
      "rank": 2
    },
    {
      "feature": "word_count",
      "importance": 0.18,
      "top_performer_avg": 14.2,
      "bottom_performer_avg": 22.5,
      "gap": 8.3,
      "rank": 3
    }
  ]
}
```

### B.3 Sample Window-Level K-Means JSON (1 cluster shown, 5 features)

**File**: `ml_analysis/hook_kmeans_analysis.json`

```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "n_clusters": 3,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "centroid": {
        "eye_contact_rate": 0.87,
        "scene_count": 0.45,
        "word_count": 0.62,
        "energy_level": 0.73,
        "has_captions": 1
      },
      "videos": [
        {
          "video_id": "video_0",
          "distance_to_centroid": 0.15
        },
        {
          "video_id": "video_5",
          "distance_to_centroid": 0.22
        },
        {
          "video_id": "video_12",
          "distance_to_centroid": 0.18
        }
      ]
    }
  ]
}
```

---

## Appendix C: Pseudocode (Complete)

### C.1 Full Stage 6 Pipeline

```python
def run_stage6_ml_analysis_generation(client_id: str, bucket: str) -> int:
    """
    Complete Stage 6 pipeline: Generate all ML analysis JSONs for a bucket.

    Args:
        client_id: Client identifier (e.g., "acme")
        bucket: Bucket name (e.g., "18-33s")

    Returns:
        int: Exit code (0=success, 1=pre-flight fail, 2=generation fail, 3=validation fail, 4=I/O fail)
    """
    logger.info(f"Starting Stage 6: ML Analysis Generation for bucket {bucket}")

    # ===== 1. Setup Paths =====
    bucket_path = f'/data/clients/{client_id}/buckets/bucket_{bucket}'

    # ===== 2. Load Bucket Configuration =====
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket]  # e.g., ['hook', 'middle_1', ..., 'closing']

    logger.info(f"Bucket {bucket} has {len(windows)} windows: {windows}")

    # ===== 3. Generate All JSONs (Atomic Pattern) =====
    exit_code = generate_ml_analysis_jsons(bucket_path, bucket, windows)

    if exit_code == 0:
        logger.info(f"✓ Stage 6 complete: {1 + 2*len(windows)} JSONs generated successfully")
    else:
        logger.error(f"✗ Stage 6 failed with exit code {exit_code}")

    return exit_code


def generate_ml_analysis_jsons(bucket_path: str, bucket: str, windows: list) -> int:
    """
    Generate all ML analysis JSONs using atomic pattern.
    Either all 13 JSONs succeed or all deleted.

    Source: Section 2.3.5 (Atomic Output Commit)
    """
    temp_dir = os.path.join(bucket_path, 'ml_analysis/.tmp/')
    os.makedirs(temp_dir, exist_ok=True)
    generated_files = []

    try:
        # ===== 1. PRE-FLIGHT VALIDATION =====
        logger.info("Pre-flight validation: checking Stage 4 and Stage 5 outputs...")
        validate_stage_dependencies(bucket_path, bucket, windows)
        logger.info("✓ Pre-flight validation passed")

        # ===== 2. GENERATE ALL JSONs TO TEMP DIRECTORY =====
        logger.info("Generating ML analysis JSONs to temp directory...")

        # Generate Video-Level RF JSON
        video_rf_json = generate_video_rf_json(bucket_path, bucket)
        temp_path = os.path.join(temp_dir, 'rf_video_analysis.json.tmp')
        with open(temp_path, 'w') as f:
            json.dump(video_rf_json, f, indent=2)
        generated_files.append(temp_path)
        logger.info(f"  ✓ Generated rf_video_analysis.json.tmp")

        # Generate Window-Level RF JSONs
        for window in windows:
            window_rf_json = generate_window_rf_json(bucket_path, bucket, window)
            temp_path = os.path.join(temp_dir, f'{window}_rf_analysis.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(window_rf_json, f, indent=2)
            generated_files.append(temp_path)
            logger.info(f"  ✓ Generated {window}_rf_analysis.json.tmp")

        # Generate Window-Level K-Means JSONs
        for window in windows:
            window_km_json = generate_window_kmeans_json(bucket_path, bucket, window)
            temp_path = os.path.join(temp_dir, f'{window}_kmeans_analysis.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(window_km_json, f, indent=2)
            generated_files.append(temp_path)
            logger.info(f"  ✓ Generated {window}_kmeans_analysis.json.tmp")

        logger.info(f"✓ All {len(generated_files)} JSONs generated to temp directory")

        # ===== 3. VALIDATE ALL JSON SCHEMAS =====
        logger.info("Validating JSON schemas...")
        validate_all_json_schemas(temp_dir, bucket, windows)
        logger.info("✓ All JSON schemas valid")

        # ===== 4. ATOMIC COMMIT: RENAME ALL TEMP FILES AT ONCE =====
        logger.info("Committing JSONs (atomic rename)...")
        for temp_file in generated_files:
            final_path = temp_file.replace('/.tmp/', '/').replace('.json.tmp', '.json')
            os.rename(temp_file, final_path)
            logger.info(f"  ✓ Committed {os.path.basename(final_path)}")

        # ===== 5. CLEANUP TEMP DIRECTORY =====
        shutil.rmtree(temp_dir)
        logger.info(f"✓ Stage 6 complete: {len(generated_files)} JSONs generated")

        return 0  # SUCCESS

    except PreFlightValidationError as e:
        logger.error(f"Pre-flight validation failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 1  # PRE-FLIGHT VALIDATION FAILED

    except json.JSONDecodeError as e:
        logger.error(f"Stage 6 JSON validation failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 3  # JSON VALIDATION FAILED

    except IOError as e:
        logger.error(f"Disk I/O failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 4  # DISK I/O FAILED

    except Exception as e:
        logger.error(f"Stage 6 JSON generation failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 2  # JSON GENERATION FAILED


# ===== HELPER FUNCTIONS =====
# (See Section 2.3 for complete implementations)

def validate_stage_dependencies(bucket_path, bucket, windows):
    """See Section 2.3.1 for full implementation"""
    pass

def generate_video_rf_json(bucket_path, bucket):
    """See Section 2.3.2 for full implementation"""
    pass

def generate_window_rf_json(bucket_path, bucket, window):
    """See Section 2.3.3 for full implementation"""
    pass

def generate_window_kmeans_json(bucket_path, bucket, window):
    """See Section 2.3.4 for full implementation"""
    pass

def normalize_feature_name(feature_name):
    """See Section 2.3.4 for full implementation"""
    pass

def validate_all_json_schemas(temp_dir, bucket, windows):
    """See Section 6.3 for full implementation"""
    pass
```

---

## Document Metadata

**Creation Date**: 2025-01-28
**Last Modified**: 2025-01-28
**Authors**: Claude Code (Phase 3 HLD Generator)
**Reviewers**: [Pending]
**Approved By**: [Pending]
**Next Review Date**: [Pending]

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-28 | Claude Code | Initial draft (Phase 3 HLD generation) |
