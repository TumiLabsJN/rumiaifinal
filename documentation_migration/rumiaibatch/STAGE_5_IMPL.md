# Stage 5: ML Model Training - Implementation Guide

> **Implementation Document for RumiAI Stage 5**
>
> **Prerequisite**: Read [PRODUCTION_FLOW.md](PRODUCTION_FLOW.md) first for pipeline overview
>
> **Purpose**: Train Random Forest and K-Means models from transformed features (Stage 4 output)
>
> **Version**: 1.0
> **Last Updated**: 2025-01-03
> **Status**: Production

---

## Table of Contents

1. [Quick Reference](#1-quick-reference)
2. [Input Contract](#2-input-contract)
3. [Output Contract](#3-output-contract)
4. [Core Functions](#4-core-functions)
5. [Data Flow](#5-data-flow)
6. [Error Handling](#6-error-handling)
7. [Debugging Guide](#7-debugging-guide)
8. [Modification Guide](#8-modification-guide)
9. [Dependencies](#9-dependencies)
10. [Testing](#10-testing)

---

## 1. Quick Reference

### 1.1 Entry Point

**Primary Entry Point**:
```python
rumiai_v2/processors/model_training.py::run_stage5_training()
Lines: 987-1079 (93 lines)
```

**Orchestrator Integration**:
```python
rumiai_ml_batch.py::main()
Lines: 1453-1627 (175 lines - Stage 5 block)
```

### 1.2 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Duration** | 7-15s | For bucket with 6 windows, 100 videos |
| **Memory Peak** | ~500 MB | RandomForest training (100 estimators) |
| **Disk I/O** | ~6.5 MB | 26 files for 6-window bucket |
| **CPU Usage** | 1+ cores | sklearn uses internal parallelization |

**Performance Breakdown** (bucket 18-33s, 100 videos):
- Video-Level RF: 0.8s (1 model, 190 features)
- Window-Level RF: 1.8s (6 models, 21 features each)
- K-Means: 3.0s (6 models, 39 features each)
- Metrics Generation: 1.4s (compute accuracy, silhouette scores)
- **Total**: ~7s

### 1.3 Model Modes

**Contrastive Mode** (Top 80% vs Bottom 20%):
- Trains: RF + K-Means
- Models: 1 video RF + N window RF + N K-Means = 1 + 2N models
- Use case: "What makes winners different from losers?"

**Top Mode** (Top N videos only):
- Trains: K-Means only (RF skipped - single class)
- Models: N K-Means models
- Use case: "What styles exist among winners?"
- **Critical**: RF cannot train with single class (C7 fix from MLModelTrainingCHILDTI.md Section 11.5)

### 1.4 Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `rumiai_v2/processors/model_training.py` | 1079 | Complete Stage 5 implementation |
| `config/model_hyperparameters.json` | 12 | Hyperparameters (RF, K-Means) |
| `config/bucket_definitions.py` | 200 | BUCKET_WINDOWS mapping (shared) |
| MLModelTrainingCHILDTI.md | 3263 | Technical specification (design rationale) |

**Total Production Code**: 1079 lines (single file implementation)

---

## 2. Input Contract

### 2.1 Prerequisites

**Upstream Stage**: Stage 4 (Feature Transformation)

**Required Checkpoint**:
```json
{bucket_path}/checkpoints/stage_4_checkpoint.json
```

**Validation**:
```python
# Orchestrator validates before calling Stage 5
stage4_checkpoint = bucket_path / "checkpoints" / "stage_4_checkpoint.json"
if not stage4_checkpoint.exists():
    logger.error(f"Bucket {bucket_name}: Stage 4 checkpoint missing")
    continue  # Skip bucket
```

### 2.2 Input Files

**Location**: `{bucket_path}/ml_analysis/`

**Required Files** (13 files for bucket 18-33s):

```python
# Video-Level RF Input
rf_transformed.csv                  # (100 videos × ~190 features)

# Window-Level RF Inputs (N = window count)
hook_rf_transformed.csv             # (100 videos × 22 features)
middle_1_rf_transformed.csv
middle_2_rf_transformed.csv
middle_3_rf_transformed.csv
middle_4_rf_transformed.csv
closing_rf_transformed.csv

# Window-Level K-Means Inputs (N = window count)
hook_km_transformed.csv             # (100 videos × 27 features)
middle_1_km_transformed.csv
middle_2_km_transformed.csv
middle_3_km_transformed.csv
middle_4_km_transformed.csv
closing_km_transformed.csv

# Scalers from Stage 4 (optional but recommended)
hook_scalers.pkl
middle_1_scalers.pkl
middle_2_scalers.pkl
middle_3_scalers.pkl
middle_4_scalers.pkl
closing_scalers.pkl
```

### 2.3 Input Schemas

#### 2.3.1 Video-Level RF Input Schema

**File**: `ml_analysis/rf_transformed.csv`

**Shape**: (100 videos, ~190 features) - varies by bucket

**Schema** (example for 18-33s bucket):
```python
{
    # Temporal window features (126 features = 21 × 6 windows)
    "hook_scene_count": int,              # Range: 0-20
    "hook_eye_contact_rate": float,       # Range: 0.0-1.0
    "hook_word_count": int,               # Range: 0-100
    "hook_speech_coverage": float,        # Range: 0.0-1.0
    "hook_energy_level": float,           # Range: 0.0-1.0
    # ... (16 more hook features)

    "middle_1_scene_count": int,
    # ... (21 middle_1 features)

    "closing_scene_count": int,
    # ... (21 closing features)

    # Video-level derived features (5 features)
    "hour": int,                          # Range: 0-23
    "day_of_week": int,                   # Range: 0-6
    "month": int,                         # Range: 1-12
    "is_weekend": int,                    # 0 or 1
    "is_business_hours": int,             # 0 or 1

    # One-hot encoded emotions (7 features)
    "joy": int,                           # 0 or 1
    "sadness": int,
    "anger": int,
    "fear": int,
    "disgust": int,
    "surprise": int,
    "neutral": int,

    # Gender encoding (3 features)
    "gender_male": int,                   # 0 or 1
    "gender_female": int,                 # 0 or 1
    "gender_nan": int,                    # 0 or 1

    # Cross-window features (0-5 features, bucket-dependent)
    "xwin_hook_to_middle_energy": float,  # Only if bucket has middle
    "xwin_middle_to_closing_energy": float,
    "xwin_eye_contact_consistency": float,
    "xwin_word_density_std": float,
    "xwin_energy_progression_slope": float,

    # Target variable (1 feature)
    "is_top_performer": int,              # 0 or 1 (from Stage 3)
}
```

**Column Count by Bucket**:
- 3-9s: 61 features (2 windows, 3 cross-window)
- 18-33s: 191 features (6 windows, 5 cross-window)

**Source**: feature_transformation.py:434-539

#### 2.3.2 Window-Level RF Input Schema

**Files**: `ml_analysis/{window}_rf_transformed.csv`

**Shape**: (100 videos, 22 features)

**Schema**:
```python
{
    # Base features (21 features)
    "scene_count": int,
    "eye_contact_rate": float,
    "word_count": int,
    "speech_coverage": float,
    "energy_level": float,
    "energy_max": float,
    "has_captions": int,                  # Boolean encoded to 0/1
    "person_count": int,
    "average_face_size": float,
    "object_count": int,
    "joy_ratio": float,
    "surprise_ratio": float,
    "anger_ratio": float,
    "disgust_ratio": float,
    "fear_ratio": float,
    "sadness_ratio": float,
    "neutral_ratio": float,
    "dominant_emotion_id": int,           # 1-7 (ordinal)
    "emotional_valence": float,           # Range: -1.0 to 1.0
    "emotion_consistency": float,
    "pitch_scatter_ratio": float,

    # Target variable (1 feature)
    "is_top_performer": int,              # 0 or 1
}
```

**Source**: feature_transformation.py:546-596

#### 2.3.3 K-Means Input Schema

**Files**: `ml_analysis/{window}_km_transformed.csv`

**Shape**: (100 videos, 27 features)

**Schema**:
```python
{
    # Scaled features (18 features with _scaled suffix)
    "scene_count_scaled": float,          # Range: 0.0-1.0 (log1p + MinMax)
    "word_count_scaled": float,           # Range: 0.0-1.0 (log1p + MinMax)
    "eye_contact_rate_scaled": float,     # Range: 0.0-1.0 (MinMax)
    "energy_level_scaled": float,         # Range: 0.0-1.0 (MinMax)
    # ... (14 more _scaled features)

    # Encoded features (9 features)
    "has_captions_encoded": int,          # 0 or 1
    "joy": int,                           # 0 or 1 (one-hot encoded)
    "sadness": int,
    "anger": int,
    "fear": int,
    "disgust": int,
    "surprise": int,
    "neutral": int,
    "emotional_valence_scaled": float,    # Range: 0.0-1.0 (shifted from [-1,1])
}
```

**Critical**: 80%+ features must have transformation suffixes (`_scaled`, `_log`, `_encoded`)

**Source**: feature_transformation.py:603-729

### 2.4 Configuration

**Hyperparameters File** (optional): `config/model_hyperparameters.json`

```json
{
  "random_forest": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "kmeans": {
    "n_clusters": 3,
    "random_state": 42,
    "n_init": 10
  }
}
```

**Behavior**:
- File exists → Load from file
- File missing → Use hardcoded defaults (log warning, continue)
- File malformed → Fail-fast (ConfigError, exit 1)

**Source**: model_training.py:151-207

---

## 3. Output Contract

### 3.1 Output Files

**Location**: `{bucket_path}/models/`

**Model Files** (26 files for bucket 18-33s):

```python
# Video-Level RF Model (1 file)
rf_video_18-33s.pkl                # 2.3 MB (joblib pickle)

# Window-Level RF Models (6 files)
rf_hook_18-33s.pkl                 # 450 KB each
rf_middle_1_18-33s.pkl
rf_middle_2_18-33s.pkl
rf_middle_3_18-33s.pkl
rf_middle_4_18-33s.pkl
rf_closing_18-33s.pkl

# K-Means Models (6 files)
hook_kmeans_18-33s.pkl             # 180 KB each
middle_1_kmeans_18-33s.pkl
middle_2_kmeans_18-33s.pkl
middle_3_kmeans_18-33s.pkl
middle_4_kmeans_18-33s.pkl
closing_kmeans_18-33s.pkl

# X Data Matrices (6 files) - for silhouette score calculation
hook_X_data_18-33s.pkl             # 45 KB each (DataFrame with feature names)
middle_1_X_data_18-33s.pkl
middle_2_X_data_18-33s.pkl
middle_3_X_data_18-33s.pkl
middle_4_X_data_18-33s.pkl
closing_X_data_18-33s.pkl

# Scalers for Inference (6 files) - copied from Stage 4
hook_scalers_18-33s.pkl            # 8 KB each (MinMaxScaler objects)
middle_1_scalers_18-33s.pkl
middle_2_scalers_18-33s.pkl
middle_3_scalers_18-33s.pkl
middle_4_scalers_18-33s.pkl
closing_scalers_18-33s.pkl

# Model Metrics Summary (1 file)
model_metrics.json                 # 8 KB (performance summary)
```

**File Count by Bucket**:
- 3-9s: 10 files (1 video RF + 2 window RF + 2 K-Means + 2 X_data + 2 scalers + 1 metrics)
- 18-33s: 26 files (1 + 6 + 6 + 6 + 6 + 1)

### 3.2 Output Schema: model_metrics.json

```json
{
  "bucket": "18-33s",
  "total_videos": 100,

  "video_level_rf": {
    "model_type": "random_forest",
    "trained": true,
    "input_features": 190,
    "accuracy": 0.87,
    "precision": 0.89,
    "recall": 0.84,
    "f1_score": 0.86,
    "top_feature": "hook_eye_contact_rate",
    "top_feature_importance": 0.22,
    "purpose": "Cross-window pattern detection"
  },

  "window_level_rf": {
    "hook": {
      "model_type": "random_forest",
      "trained": true,
      "input_features": 21,
      "accuracy": 0.82,
      "precision": 0.85,
      "recall": 0.79,
      "f1_score": 0.82,
      "top_feature": "eye_contact_rate",
      "top_feature_importance": 0.35
    }
  },

  "window_level_kmeans": {
    "hook": {
      "model_type": "kmeans",
      "input_features": 27,
      "n_clusters": 3,
      "inertia": 128.55,
      "silhouette_score": 0.68,
      "cluster_sizes": [35, 42, 23]
    }
  }
}
```

**Mode-Specific Variations**:

**Contrastive Mode** (`trained: true`):
```json
{
  "video_level_rf": {
    "trained": true,
    "accuracy": 0.87,
    // ... full metrics
  }
}
```

**Top Mode** (`trained: false`):
```json
{
  "video_level_rf": {
    "trained": false,
    "skip_reason": "Single class in dataset (expected in 'top' mode)",
    "purpose": "Cross-window pattern detection"
  }
}
```

**Source**: model_training.py:285-384, MLModelTrainingCHILDTI.md:516-573

### 3.3 Checkpoint Schema

**File**: `{bucket_path}/checkpoints/stage_5_checkpoint.json`

```json
{
  "stage": "stage_5_ml_model_training",
  "bucket": "18-33s",
  "status": "completed",
  "timestamp": "2025-10-28T13:10:59.261394Z",
  "models_trained": 13,
  "output_files": [
    "/path/to/models/rf_video_18-33s.pkl",
    "/path/to/models/hook_kmeans_18-33s.pkl",
    "..."
  ]
}
```

**Created By**: Orchestrator (rumiai_ml_batch.py:1525-1537)

---

## 4. Core Functions

### 4.1 Function Reference Table

| Function | Lines | Purpose | Returns | Raises |
|----------|-------|---------|---------|--------|
| `run_stage5_training()` | 987-1079 | **Entry point** | `(bool, list, float)` | Custom exceptions |
| `validate_stage_input()` | 737-806 | Pre-flight validation | `None` | ValidationError |
| `validate_stage4_outputs()` | 85-149 | Check Stage 4 files exist | `None` | StageInputError |
| `load_model_config()` | 151-207 | Load hyperparameters | `dict` | ConfigError |
| `validate_business_rules()` | 862-887 | Validate hyperparameters | `None` | ValidationError |
| `train_bucket_models()` | 422-621 | **Main training loop** | `None` | ModelTrainingError |
| `generate_model_metrics()` | 285-419 | Create metrics JSON | `dict` | None |
| `normalize_feature_name()` | 208-241 | Remove transformation suffixes | `str` | None |
| `get_top_cluster_features()` | 243-283 | Extract cluster-defining features | `list` | None |
| `validate_stage_output()` | 889-984 | Verify all models created | `None` | ValidationError |
| `validate_kmeans_feature_naming()` | 807-860 | Check 80%+ have suffixes | `None` | ValidationError |
| `atomic_rollback()` | 623-665 | Delete all partial models | `None` | None |
| `log_training_error()` | 667-735 | Comprehensive error logging | `None` | None |

### 4.2 Custom Exceptions

**File**: model_training.py:52-71

```python
class StageInputError(Exception):
    """Raised when Stage 4 inputs missing or invalid."""
    pass

class InsufficientDataError(Exception):
    """Raised when video count below minimum threshold."""
    pass

class ConfigError(Exception):
    """Raised when config file malformed."""
    pass

class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass

class ValidationError(Exception):
    """Raised when validation checks fail."""
    pass
```

### 4.3 Entry Point: run_stage5_training()

**Location**: model_training.py:987-1079

**Signature**:
```python
def run_stage5_training(
    bucket_path: str,
    config: dict,
    selection_strategy: str
) -> Tuple[bool, List[str], float]:
```

**Parameters**:
- `bucket_path`: Absolute path to bucket directory (e.g., `/data/clients/acme/buckets/bucket_18-33s`)
- `config`: Configuration dict with keys: `bucket`, `strategy`, `video_count`
- `selection_strategy`: "contrastive" or "top" (determines min video count)

**Returns**:
- `success`: bool (True if all models trained)
- `output_files`: list[str] (list of created file paths)
- `elapsed_time`: float (total execution time in seconds)

**Raises**:
- `StageInputError`: Stage 4 files missing
- `InsufficientDataError`: Video count < minimum
- `ConfigError`: Hyperparameters malformed
- `ModelTrainingError`: Training failed
- `ValidationError`: Output validation failed

**Call Chain**:
```
run_stage5_training()
├─→ validate_stage_input()
│   ├─→ validate_stage4_outputs()
│   └─→ validate_kmeans_feature_naming()
├─→ load_model_config()
├─→ validate_business_rules()
├─→ train_bucket_models()
│   ├─→ generate_model_metrics()
│   │   ├─→ normalize_feature_name()
│   │   └─→ get_top_cluster_features()
│   └─→ atomic_rollback() (on failure)
└─→ validate_stage_output()
```

**Example Usage**:
```python
# Called by orchestrator (rumiai_ml_batch.py)
bucket_config = {
    "bucket": "18-33s",
    "strategy": "contrastive",
    "video_count": 100
}

success, output_files, elapsed_time = run_stage5_training(
    bucket_path="/data/clients/acme/buckets/bucket_18-33s",
    config=bucket_config,
    selection_strategy="contrastive"
)

# success: True
# output_files: ['/path/to/rf_video_18-33s.pkl', ...]
# elapsed_time: 7.5
```

### 4.4 Training Loop: train_bucket_models()

**Location**: model_training.py:422-621

**Algorithm**:
```python
def train_bucket_models(bucket, windows, bucket_base, config, selection_strategy):
    trained_models = []  # Track for atomic rollback

    try:
        # STEP 0: Check if RF training possible
        X_check = pd.read_csv('ml_analysis/rf_transformed.csv')
        unique_labels = X_check['is_top_performer'].unique()
        can_train_rf = len(unique_labels) >= 2  # C7 fix

        # STEP 1: Train Video-Level RF (if binary classification possible)
        if can_train_rf:
            rf_video = RandomForestClassifier(**config['random_forest'])
            rf_video.fit(X, y)
            joblib.dump(rf_video, 'models/rf_video_{bucket}.pkl')
            trained_models.append(model_path)

        # STEP 2: Train Window-Level RF (if possible)
        if can_train_rf:
            for window in windows:
                rf_window = RandomForestClassifier(**config['random_forest'])
                rf_window.fit(X_window, y_window)
                joblib.dump(rf_window, f'models/rf_{window}_{bucket}.pkl')
                trained_models.append(model_path)

        # STEP 3: Train K-Means (always, works with any label distribution)
        for window in windows:
            kmeans = KMeans(**config['kmeans'])
            kmeans.fit(X_km)
            joblib.dump(kmeans, f'models/{window}_kmeans_{bucket}.pkl')
            trained_models.append(model_path)

            # Save X matrix for silhouette score
            joblib.dump(X_km, f'models/{window}_X_data_{bucket}.pkl')
            trained_models.append(X_path)

            # Copy scalers from Stage 4
            shutil.copy(f'ml_analysis/{window}_scalers.pkl',
                       f'models/{window}_scalers_{bucket}.pkl')
            trained_models.append(scaler_path)

        # STEP 4: Generate model_metrics.json
        metrics = generate_model_metrics(...)
        json.dump(metrics, 'models/model_metrics.json')
        trained_models.append(metrics_path)

    except Exception as e:
        # Atomic rollback: Delete ALL partial models
        atomic_rollback(bucket, trained_models, bucket_base)
        raise ModelTrainingError(f"Training failed: {e}")
```

**Key Decisions**:
- **Atomic guarantee**: All models succeed OR all deleted
- **Mode-aware**: Skips RF if single class (TOP mode)
- **Sequential**: Trains one model at a time (not parallel)

**Source**: model_training.py:422-621, MLModelTrainingCHILDTI.md:788-993

### 4.5 Feature Normalization: normalize_feature_name()

**Location**: model_training.py:208-241

**Purpose**: Remove K-Means transformation suffixes for comparison with RF features

**Algorithm**:
```python
def normalize_feature_name(feature_name: str) -> str:
    """
    eye_contact_rate_scaled → eye_contact_rate
    has_captions_encoded → has_captions
    scene_count → scene_count (unchanged)
    """
    normalized = feature_name
    suffixes = ['_scaled', '_log', '_encoded']
    for suffix in suffixes:
        normalized = normalized.replace(suffix, '')
    return normalized
```

**Critical**: Stage 6 has identical function (ml_analysis_generation.py:495-519)

**Use Cases**:
1. Comparing K-Means top features with RF top features
2. Generating normalized centroids for Stage 6 JSONs
3. Feature overlap validation

### 4.6 Cluster Feature Extraction: get_top_cluster_features()

**Location**: model_training.py:243-283

**Purpose**: Extract top N features that differentiate clusters

**Algorithm**:
```python
def get_top_cluster_features(kmeans_model, feature_names, n=5):
    """
    Ranks features by variance across cluster centroids.
    High variance = feature values differ across clusters = cluster-defining.
    """
    centroids = kmeans_model.cluster_centers_  # (3 clusters, 27 features)
    feature_variances = np.var(centroids, axis=0)  # Variance per feature
    top_indices = np.argsort(feature_variances)[::-1][:n]
    return [feature_names[i] for i in top_indices]
```

**Example**:
```python
# Cluster centroids:
# Cluster 0: [eye_contact=0.8, scene_count=0.3, energy=0.6]
# Cluster 1: [eye_contact=0.2, scene_count=0.7, energy=0.5]
# Cluster 2: [eye_contact=0.5, scene_count=0.4, energy=0.9]

# Variances:
# eye_contact: var=0.09 (high variance → cluster-defining)
# scene_count: var=0.04 (medium variance)
# energy: var=0.03 (low variance → similar across clusters)

# Returns: ['eye_contact_rate', 'scene_count', 'energy_level', ...]
```

**Source**: model_training.py:243-283, MLModelTrainingCHILDTI.md:1068-1110

---

## 5. Data Flow

### 5.1 Overview Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      STAGE 4 (Upstream)                     │
│  feature_transformation.py::run_stage4_transformation()      │
│                                                              │
│  Outputs:                                                    │
│  ├─ ml_analysis/rf_transformed.csv (100×190)                │
│  ├─ ml_analysis/hook_rf_transformed.csv (100×22)            │
│  ├─ ml_analysis/hook_km_transformed.csv (100×27)            │
│  └─ ml_analysis/hook_scalers.pkl                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      STAGE 5 (This Stage)                   │
│         model_training.py::run_stage5_training()             │
│                                                              │
│  Processing:                                                 │
│  ├─ Load CSVs                                                │
│  ├─ Train Video-Level RF (1 model)                          │
│  ├─ Train Window-Level RF (6 models) - if can_train_rf      │
│  ├─ Train K-Means (6 models) - always                       │
│  └─ Generate model_metrics.json                             │
│                                                              │
│  Outputs:                                                    │
│  ├─ models/rf_video_18-33s.pkl (2.3 MB)                     │
│  ├─ models/hook_kmeans_18-33s.pkl (180 KB)                  │
│  ├─ models/hook_X_data_18-33s.pkl (45 KB)                   │
│  ├─ models/hook_scalers_18-33s.pkl (8 KB)                   │
│  └─ models/model_metrics.json (8 KB)                        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 6 (Downstream)                     │
│   ml_analysis_generation.py::generate_ml_analysis_jsons()   │
│                                                              │
│  Uses:                                                       │
│  ├─ Loads rf_video_18-33s.pkl → Extract feature importance  │
│  ├─ Loads hook_kmeans_18-33s.pkl → Extract centroids        │
│  ├─ Loads hook_X_data_18-33s.pkl → Get feature names        │
│  └─ Reads model_metrics.json → Check if RF trained          │
│                                                              │
│  Outputs:                                                    │
│  ├─ ml_analysis/rf_video_analysis.json                      │
│  ├─ ml_analysis/hook_rf_analysis.json                       │
│  └─ ml_analysis/hook_kmeans_analysis.json                   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Function Call Chain

```
run_stage5_training(bucket_path, config, selection_strategy)
│
├─ Step 1: PRE-FLIGHT VALIDATION
│  │
│  ├─ validate_stage_input(bucket, windows, bucket_base, selection_strategy, video_count)
│  │  │
│  │  ├─ validate_stage4_outputs(bucket, windows, bucket_base, selection_strategy)
│  │  │  │
│  │  │  ├─ os.path.exists() checks for 13 files
│  │  │  ├─ pd.read_csv() to check non-empty
│  │  │  └─ Video count threshold check (50 contrastive, 30 top)
│  │  │
│  │  └─ validate_kmeans_feature_naming(csv_path, expected_suffix='_scaled')
│  │     │
│  │     ├─ Read CSV header
│  │     ├─ Count features with _scaled, _log, _encoded
│  │     └─ Assert ≥80% have transformation suffixes
│  │
│  ├─ load_model_config()
│  │  │
│  │  ├─ os.path.exists('config/model_hyperparameters.json')
│  │  ├─ json.load() → parse config
│  │  └─ Validate required keys (random_forest, kmeans)
│  │
│  └─ validate_business_rules(bucket, windows, config)
│     │
│     ├─ Check n_estimators > 0
│     ├─ Check max_depth > 0
│     ├─ Check n_clusters == 3
│     └─ Check 2 ≤ len(windows) ≤ 7
│
├─ Step 2: TRAIN ALL MODELS
│  │
│  └─ train_bucket_models(bucket, windows, bucket_base, config, selection_strategy)
│     │
│     ├─ Check can_train_rf (unique labels ≥ 2)
│     │
│     ├─ Train Video-Level RF (if can_train_rf)
│     │  ├─ pd.read_csv('rf_transformed.csv')
│     │  ├─ X = drop(['is_top_performer', 'video_id'])
│     │  ├─ y = df['is_top_performer']
│     │  ├─ rf_video = RandomForestClassifier(**config['random_forest'])
│     │  ├─ rf_video.fit(X, y)
│     │  ├─ joblib.dump(rf_video, model_path)
│     │  └─ trained_models.append(model_path)
│     │
│     ├─ Train Window-Level RF (for each window, if can_train_rf)
│     │  ├─ pd.read_csv(f'{window}_rf_transformed.csv')
│     │  ├─ rf_window = RandomForestClassifier(**config['random_forest'])
│     │  ├─ rf_window.fit(X_window, y_window)
│     │  ├─ joblib.dump(rf_window, model_path)
│     │  └─ trained_models.append(model_path)
│     │
│     ├─ Train K-Means (for each window, always)
│     │  ├─ pd.read_csv(f'{window}_km_transformed.csv')
│     │  ├─ X_km = drop(['video_id'])
│     │  ├─ kmeans = KMeans(**config['kmeans'])
│     │  ├─ kmeans.fit(X_km)
│     │  ├─ joblib.dump(kmeans, model_path)
│     │  ├─ joblib.dump(X_km, X_data_path)  # For silhouette
│     │  ├─ shutil.copy(scaler_source, scaler_dest)
│     │  └─ trained_models.append(...)
│     │
│     └─ generate_model_metrics(bucket, windows, bucket_base, rf_video, rf_window_models, kmeans_models, X_data_matrices, total_videos)
│        │
│        ├─ Compute RF metrics (if trained)
│        │  ├─ y_pred = rf_video.predict(X_video)
│        │  ├─ accuracy_score(y_true, y_pred)
│        │  ├─ precision_score(y_true, y_pred)
│        │  ├─ recall_score(y_true, y_pred)
│        │  ├─ f1_score(y_true, y_pred)
│        │  └─ Extract top feature (np.argmax(feature_importances))
│        │
│        ├─ Compute K-Means metrics (always)
│        │  ├─ labels = kmeans.labels_
│        │  ├─ silhouette_score(X_kmeans, labels)
│        │  ├─ cluster_sizes = [sum(labels == i) for i in range(3)]
│        │  └─ get_top_cluster_features(kmeans, feature_names, n=5)
│        │     ├─ centroids = kmeans.cluster_centers_
│        │     ├─ feature_variances = np.var(centroids, axis=0)
│        │     └─ Return top 5 by variance
│        │
│        └─ Return metrics dict
│
└─ Step 3: POST-TRAINING VALIDATION
   │
   └─ validate_stage_output(bucket, windows, bucket_base)
      │
      ├─ Check all model files exist
      ├─ Validate model_metrics.json schema
      ├─ Check accuracy in [0.0, 1.0] (if RF trained)
      └─ Check cluster sizes balanced (warn if <10%)

ON EXCEPTION:
   │
   └─ atomic_rollback(bucket, trained_models, bucket_base)
      │
      ├─ for model_path in trained_models:
      │     os.remove(model_path)
      │
      └─ log_training_error(bucket, current_model, exception, trained_models, start_time, config, bucket_base)
         │
         ├─ Log WHAT failed (model name, input file, input shape)
         ├─ Log WHY failed (exception type, message, stack trace)
         ├─ Log CONTEXT (hyperparameters, completed models, training duration, NaN count)
         └─ Log NEXT STEPS (check data quality, verify sklearn version, etc.)
```

### 5.3 File Lifecycle

```
INPUT FILES (Stage 4)
├─ ml_analysis/rf_transformed.csv
├─ ml_analysis/hook_rf_transformed.csv
├─ ml_analysis/hook_km_transformed.csv
└─ ml_analysis/hook_scalers.pkl
       │
       │ read_csv()
       │ joblib.load()
       ▼
┌──────────────────────────────────┐
│   IN-MEMORY TRAINING OBJECTS     │
│  ├─ X: DataFrame (100×190)       │
│  ├─ y: Series (100,)             │
│  ├─ rf_video: RandomForest       │
│  ├─ kmeans: KMeans               │
│  └─ metrics: dict                │
└──────────────────────────────────┘
       │
       │ joblib.dump()
       │ json.dump()
       ▼
OUTPUT FILES (Stage 5)
├─ models/rf_video_18-33s.pkl
├─ models/hook_kmeans_18-33s.pkl
├─ models/hook_X_data_18-33s.pkl
├─ models/hook_scalers_18-33s.pkl
└─ models/model_metrics.json
       │
       │ joblib.load() (Stage 6)
       │ json.load() (Stage 6)
       ▼
DOWNSTREAM STAGE 6
├─ ml_analysis/rf_video_analysis.json
├─ ml_analysis/hook_rf_analysis.json
└─ ml_analysis/hook_kmeans_analysis.json
```

---

## 6. Error Handling

### 6.1 Error Scenarios Matrix

| Error Type | Scenario | Action | Exit Code | Recovery |
|------------|----------|--------|-----------|----------|
| **StageInputError** | Stage 4 CSV missing | Skip bucket, continue | N/A (orchestrator) | Re-run Stage 4 |
| **InsufficientDataError** | Video count < 50 (contrastive) | Skip bucket, continue | N/A (orchestrator) | Lower --video-count |
| **ConfigError** | model_hyperparameters.json malformed JSON | Fail-fast, exit pipeline | 1 | Fix JSON or delete file |
| **ModelTrainingError** | NaN values in data | Atomic rollback, skip bucket | N/A (orchestrator) | Fix Stage 4 data |
| **ModelTrainingError** | sklearn ValueError | Atomic rollback, skip bucket | N/A (orchestrator) | Check sklearn version |
| **ValidationError** | Output validation failed | Atomic rollback, skip bucket | N/A (orchestrator) | Debug validation logic |
| **IOError** | Disk full, permission denied | Exit pipeline | 4 | Check disk space |
| **Exception** | Unexpected error | Exit pipeline | 99 | Debug stack trace |

**Source**: model_training.py:52-71, rumiai_ml_batch.py:1553-1611

### 6.2 Orchestrator Error Handling

**Location**: rumiai_ml_batch.py:1553-1611

```python
# Stage 5 execution in orchestrator
for bucket_name in winning_buckets:
    try:
        success, output_files, elapsed_time = run_stage5_training(
            bucket_path=str(bucket_path),
            config=bucket_config,
            selection_strategy=config.selection_strategy
        )

        # Success: Create checkpoint
        checkpoint_data = {
            "stage": "stage_5_ml_model_training",
            "bucket": bucket_name,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "models_trained": len([f for f in output_files if f.endswith('.pkl')]),
            "output_files": output_files
        }

    except StageInputError as e:
        # Missing Stage 4 files - skip bucket, continue pipeline
        logger.error(f"Stage 5 prerequisite missing for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name}: Prerequisites missing (skipping)")
        continue

    except InsufficientDataError as e:
        # Not enough videos - skip bucket, continue pipeline
        logger.error(f"Stage 5 validation failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name}: Insufficient videos (skipping)")
        continue

    except ModelTrainingError as e:
        # Training failed (atomic rollback already performed)
        logger.error(f"Stage 5 training failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name}: Training failed (skipping)")
        continue

    except ValidationError as e:
        # Output validation failed
        logger.error(f"Stage 5 output validation failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name}: Output validation failed (skipping)")
        continue

    except (IOError, OSError) as e:
        # System-wide I/O error - exit pipeline
        logger.error(f"Stage 5 I/O error for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name}: I/O error (exiting pipeline)")
        return 4  # Exit code 4 = I/O failure

    except Exception as e:
        # Unexpected error - exit pipeline
        logger.error(f"Stage 5 unexpected error for bucket {bucket_name}: {e}", exc_info=True)
        print(f"✗ Bucket {bucket_name} unexpected error: {e}")
        return 99  # Exit code 99 = unexpected error
```

**Error Handling Strategy**:
- **Bucket-specific errors** (StageInputError, InsufficientDataError, ModelTrainingError, ValidationError): Skip bucket, continue pipeline
- **System-wide errors** (IOError, OSError): Exit pipeline (affects all buckets)
- **Unexpected errors** (Exception): Exit pipeline (unknown state)

### 6.3 Atomic Rollback

**Location**: model_training.py:623-665

**Purpose**: Delete ALL partial models on failure (never leave partial model sets)

**Algorithm**:
```python
def atomic_rollback(bucket: str, trained_models: List[str], bucket_base: str) -> None:
    """
    Atomic rollback: Delete ALL partial models for this bucket.

    Q8 Decision: All models succeed OR all deleted on failure.
    Result: Either bucket has complete model set OR no models. Never partial.
    """
    logger.info(f"Performing atomic rollback for bucket {bucket}...")
    logger.info(f"Models to delete: {len(trained_models)} files")

    deleted_count = 0
    for model_path in trained_models:
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logger.info(f"  ✓ Deleted: {model_path}")
                deleted_count += 1
            except OSError as e:
                logger.error(f"  ✗ Failed to delete: {model_path} - {e}")

    logger.info(f"Rollback complete: {deleted_count}/{len(trained_models)} files deleted")

    # Verify bucket is clean
    models_dir = os.path.join(bucket_base, 'models')
    if os.path.exists(models_dir):
        remaining_files = [f for f in os.listdir(models_dir) if bucket in f]
        if remaining_files:
            logger.warning(f"Warning: {len(remaining_files)} files still exist after rollback")
        else:
            logger.info("✓ Bucket clean: No partial models remain")
```

**Tracking Pattern**:
```python
trained_models = []  # Initialize before training

try:
    # Train model 1
    joblib.dump(rf_video, model_path)
    trained_models.append(model_path)  # Track immediately after save

    # Train model 2
    joblib.dump(rf_hook, model_path)
    trained_models.append(model_path)

    # ... more models ...

except Exception as e:
    # Rollback deletes ALL files in trained_models list
    atomic_rollback(bucket, trained_models, bucket_base)
    raise ModelTrainingError(f"Training failed: {e}")
```

**Example Scenario**:
```
Training bucket 18-33s (26 models total):
✓ Saved: rf_video_18-33s.pkl
✓ Saved: rf_hook_18-33s.pkl
✓ Saved: rf_middle_1_18-33s.pkl
✗ FAILED: rf_middle_2_18-33s.pkl (NaN values detected)

Atomic rollback triggered:
✓ Deleted: rf_video_18-33s.pkl
✓ Deleted: rf_hook_18-33s.pkl
✓ Deleted: rf_middle_1_18-33s.pkl

Result: 0 models in bucket 18-33s (clean state)
```

**Source**: model_training.py:623-665, MLModelTrainingCHILDTI.md:1712-1751

### 6.4 Comprehensive Error Logging

**Location**: model_training.py:667-735

**Purpose**: Log WHAT/WHY/CONTEXT without sensitive data

**Template**:
```python
def log_training_error(bucket, current_model, exception, trained_models, start_time, config, bucket_base):
    """
    Q10 Decision: Balanced Logging (Error + Context, No Data Dump)
    """
    elapsed = time.time() - start_time

    # Get input file path
    if 'rf_video' in current_model:
        input_file = os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv')
    elif '_rf_' in current_model:
        window = current_model.split('_rf_')[0].replace('models/rf_', '')
        input_file = os.path.join(bucket_base, f'ml_analysis/{window}_rf_transformed.csv')
    elif '_kmeans_' in current_model:
        window = current_model.split('_kmeans_')[0].replace('models/', '')
        input_file = os.path.join(bucket_base, f'ml_analysis/{window}_km_transformed.csv')
    else:
        input_file = "Unknown"

    # Get input shape and NaN count
    input_shape = "Unknown"
    nan_count = "Unknown"
    if os.path.exists(input_file):
        try:
            df = pd.read_csv(input_file)
            input_shape = df.shape
            nan_count = df.isna().sum().sum()
        except Exception:
            pass

    logger.error(f"""
===============================================================================
BUCKET {bucket} TRAINING FAILED
===============================================================================

WHAT FAILED:
  Model name: {current_model}
  Input file: {input_file}
  Input shape: {input_shape}

WHY IT FAILED:
  Exception type: {type(exception).__name__}
  Exception message: {str(exception)}
  Stack trace (first 10 lines):
{traceback.format_exc(limit=10)}

CONTEXT:
  Hyperparameters: {config}
  Completed models before failure: {len(trained_models)} files
  Training duration before failure: {elapsed:.1f}s
  NaN count in input: {nan_count} values

RECOVERY ACTION:
  Atomic rollback: Deleting all {len(trained_models)} partial models
  Bucket state after rollback: Clean (no partial models)

NEXT STEPS:
  1. Check input data quality (NaN values, feature ranges)
  2. Verify sklearn version >= 0.24.0
  3. Check disk space and memory availability
  4. Re-run Stage 5 after fixing issue

===============================================================================
""")
```

**Example Output**:
```
===============================================================================
BUCKET 18-33s TRAINING FAILED
===============================================================================

WHAT FAILED:
  Model name: models/rf_middle_2_18-33s.pkl
  Input file: /data/.../ml_analysis/middle_2_rf_transformed.csv
  Input shape: (100, 22)

WHY IT FAILED:
  Exception type: ValueError
  Exception message: Input contains NaN
  Stack trace (first 10 lines):
    File "train_bucket_models.py", line 52, in train_bucket_models
      rf_window.fit(X, y)
    File "sklearn/ensemble/_forest.py", line 345, in fit
      X, y = self._validate_data(X, y)
    File "sklearn/base.py", line 576, in _validate_data
      X = check_array(X, ...)
    ValueError: Input contains NaN

CONTEXT:
  Hyperparameters: {'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}, ...}
  Completed models before failure: 8 files
  Training duration before failure: 3.2s
  NaN count in input: 5 values

RECOVERY ACTION:
  Atomic rollback: Deleting all 8 partial models
  Bucket state after rollback: Clean (no partial models)

NEXT STEPS:
  1. Check input data quality (NaN values, feature ranges)
  2. Verify sklearn version >= 0.24.0
  3. Check disk space and memory availability
  4. Re-run Stage 5 after fixing issue

===============================================================================
```

**Privacy**: Never logs actual feature values, video IDs, or sensitive data

**Source**: model_training.py:667-735, MLModelTrainingCHILDTI.md:1756-1848

---

## 7. Debugging Guide

### 7.1 Common Issues

#### Issue 1: "Stage 4 checkpoint missing"

**Symptom**:
```
✗ Bucket 18-33s: Stage 4 not complete (skipping)
```

**Cause**: Stage 4 did not complete successfully for this bucket

**Debug Steps**:
1. Check if Stage 4 checkpoint exists:
   ```bash
   ls {bucket_path}/checkpoints/stage_4_checkpoint.json
   ```

2. If missing, check Stage 4 logs:
   ```bash
   grep "Stage 4" {analysis_base}/logs/*.log | tail -50
   ```

3. Verify Stage 4 files exist:
   ```bash
   ls {bucket_path}/ml_analysis/
   # Should see: rf_transformed.csv, hook_rf_transformed.csv, etc.
   ```

**Fix**: Re-run Stage 4 for this bucket

#### Issue 2: "Insufficient videos for training"

**Symptom**:
```
✗ Bucket 18-33s: Insufficient videos (skipping)
ERROR: Bucket 18-33s has 45 videos (min 50 required for contrastive mode)
```

**Cause**: Video count below minimum threshold

**Debug Steps**:
1. Check actual video count:
   ```bash
   wc -l {bucket_path}/ml_analysis/rf_transformed.csv
   # Subtract 1 for header
   ```

2. Check selection strategy:
   ```bash
   grep "selection_strategy" {analysis_base}/config.json
   ```

3. Check minimum thresholds:
   - Contrastive mode: 50 videos (40 top + 10 bottom)
   - Top mode: 30 videos

**Fix**:
- Lower `--video-count` in Stage 1, OR
- Switch to `--selection-strategy top` (lower threshold)

#### Issue 3: "NaN values detected during training"

**Symptom**:
```
✗ Bucket 18-33s: Training failed (skipping)
ERROR: ValueError: Input contains NaN
CONTEXT: NaN count in input: 12 values
```

**Cause**: Stage 4 produced CSV with NaN values

**Debug Steps**:
1. Find which file has NaNs:
   ```python
   import pandas as pd
   df = pd.read_csv('ml_analysis/middle_2_rf_transformed.csv')
   print(df.isna().sum())  # Count NaNs per column
   print(df[df.isna().any(axis=1)])  # Show rows with NaNs
   ```

2. Check Stage 4 validation logs:
   ```bash
   grep "validation" {analysis_base}/logs/*.log | grep "Stage 4"
   ```

3. Trace back to Stage 3:
   ```bash
   # Check aggregated_features.csv
   python -c "import pandas as pd; df = pd.read_csv('ml_analysis/aggregated_features.csv'); print(df.isna().sum())"
   ```

**Fix**:
- Fix Stage 3 aggregation logic (missing feature calculation)
- Re-run Stage 3 and Stage 4

#### Issue 4: "K-Means feature naming validation failed"

**Symptom**:
```
ERROR: K-Means CSV feature naming validation failed: hook_km_transformed.csv
  Total features: 27
  Features with _scaled: 12
  Features with _encoded: 3
  Total transformed: 15/27 (55.6%)
  Expected: >=22 (80%)
```

**Cause**: Stage 4 did not apply transformations correctly

**Debug Steps**:
1. Check K-Means CSV headers:
   ```bash
   head -1 {bucket_path}/ml_analysis/hook_km_transformed.csv
   ```

2. Expected suffixes:
   - `_scaled` (from MinMax scaling)
   - `_log` (from log transformation)
   - `_encoded` (from label encoding)

3. Check Stage 4 transformation logic:
   ```bash
   grep "def transform_window_level_kmeans" feature_transformation.py -A 50
   ```

**Fix**: Re-run Stage 4 (bug in transformation logic)

#### Issue 5: "RF model file exists but metrics says trained=False"

**Symptom**:
```
⚠️ Stale RF model files detected from previous run.
model_metrics.json says trained=False (TOP mode), ignoring stale files.
```

**Cause**: Previous contrastive run left RF models, now running in TOP mode

**Debug Steps**:
1. Check model_metrics.json:
   ```bash
   cat {bucket_path}/models/model_metrics.json | jq '.video_level_rf.trained'
   ```

2. Check if RF files exist:
   ```bash
   ls {bucket_path}/models/rf_*.pkl
   ```

**Fix**: Delete stale RF files (Stage 6 will ignore them anyway):
```bash
rm {bucket_path}/models/rf_*.pkl
```

#### Issue 6: "Atomic rollback failed to delete files"

**Symptom**:
```
Warning: 3 files still exist in models/ after rollback: ['rf_video_18-33s.pkl', ...]
```

**Cause**: Permission issue or file locked by another process

**Debug Steps**:
1. Check file permissions:
   ```bash
   ls -la {bucket_path}/models/
   ```

2. Check if files are locked:
   ```bash
   lsof {bucket_path}/models/rf_video_18-33s.pkl
   ```

**Fix**:
- Manually delete files:
  ```bash
  rm {bucket_path}/models/*_18-33s.pkl
  ```
- Fix permissions:
  ```bash
  chmod 644 {bucket_path}/models/*.pkl
  ```

### 7.2 Debugging Commands

**Check Stage 5 execution status**:
```bash
# Check if Stage 5 checkpoint exists
ls {bucket_path}/checkpoints/stage_5_checkpoint.json

# View checkpoint
cat {bucket_path}/checkpoints/stage_5_checkpoint.json | jq '.'

# Count models trained
cat {bucket_path}/checkpoints/stage_5_checkpoint.json | jq '.models_trained'

# List output files
cat {bucket_path}/checkpoints/stage_5_checkpoint.json | jq '.output_files[]'
```

**Verify model files**:
```bash
# List all model files
ls -lh {bucket_path}/models/

# Count model files
ls {bucket_path}/models/*.pkl | wc -l

# Check model file sizes
du -sh {bucket_path}/models/*

# Verify model loadable
python -c "import joblib; m = joblib.load('{bucket_path}/models/rf_video_18-33s.pkl'); print(m)"
```

**Inspect model_metrics.json**:
```bash
# Pretty-print metrics
cat {bucket_path}/models/model_metrics.json | jq '.'

# Check if RF trained
cat {bucket_path}/models/model_metrics.json | jq '.video_level_rf.trained'

# Get video RF accuracy
cat {bucket_path}/models/model_metrics.json | jq '.video_level_rf.accuracy'

# Get K-Means silhouette scores
cat {bucket_path}/models/model_metrics.json | jq '.window_level_kmeans[].silhouette_score'

# Get cluster sizes
cat {bucket_path}/models/model_metrics.json | jq '.window_level_kmeans[].cluster_sizes'
```

**Check input data quality**:
```python
import pandas as pd
import numpy as np

# Load RF input
df = pd.read_csv('{bucket_path}/ml_analysis/rf_transformed.csv')

print(f"Shape: {df.shape}")
print(f"NaN count: {df.isna().sum().sum()}")
print(f"Columns: {list(df.columns)}")

# Check label distribution
print(f"Label distribution:\n{df['is_top_performer'].value_counts()}")

# Check if binary classification possible
unique_labels = df['is_top_performer'].unique()
print(f"Unique labels: {unique_labels}")
print(f"Can train RF: {len(unique_labels) >= 2}")
```

### 7.3 Performance Profiling

**Measure Stage 5 performance**:
```python
import time
import psutil

process = psutil.Process()
start_time = time.time()
start_memory = process.memory_info().rss / 1024 / 1024  # MB

# Run Stage 5
success, output_files, elapsed_time = run_stage5_training(...)

end_memory = process.memory_info().rss / 1024 / 1024  # MB

print(f"Duration: {elapsed_time:.1f}s")
print(f"Memory: {start_memory:.1f} MB → {end_memory:.1f} MB (Δ {end_memory - start_memory:.1f} MB)")
print(f"Files: {len(output_files)}")
print(f"Throughput: {len(output_files) / elapsed_time:.1f} files/sec")
```

**Expected Performance** (bucket 18-33s, 100 videos):
- Duration: 7-15s
- Memory: ~500 MB peak
- Throughput: ~2-3 files/sec

**Performance Red Flags**:
- Duration > 60s → Check for I/O bottleneck or slow disk
- Memory > 2 GB → Check for memory leak or large feature matrices
- Duration varies wildly → Check for contention (other processes)

---

## 8. Modification Guide

### 8.1 How to Add a New Hyperparameter

**Scenario**: Add `min_samples_leaf` to RandomForest config

**Step 1**: Update config schema
```json
// config/model_hyperparameters.json
{
  "random_forest": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 2,  // NEW
    "random_state": 42
  }
}
```

**Step 2**: Update validation (model_training.py:862-887)
```python
def validate_business_rules(bucket: str, windows: List[str], config: dict) -> None:
    # ... existing validation ...

    # NEW: Validate min_samples_leaf
    if 'min_samples_leaf' in config['random_forest']:
        if config['random_forest']['min_samples_leaf'] < 1:
            raise ValidationError("min_samples_leaf must be >= 1")
```

**Step 3**: No code changes needed in training loop
```python
# train_bucket_models() already passes **config['random_forest']
rf_video = RandomForestClassifier(**config['random_forest'])  # Automatically includes min_samples_leaf
```

**Step 4**: Test
```bash
# Run with new hyperparameter
python rumiai_ml_batch.py --client test --bucket 18-33s
```

### 8.2 How to Change Cluster Count

**Scenario**: Change K-Means from 3 clusters to 5 clusters

**Step 1**: Update config
```json
// config/model_hyperparameters.json
{
  "kmeans": {
    "n_clusters": 5,  // Changed from 3
    "random_state": 42,
    "n_init": 10
  }
}
```

**Step 2**: Update validation (model_training.py:862-887)
```python
def validate_business_rules(bucket: str, windows: List[str], config: dict) -> None:
    # ... existing validation ...

    # MODIFIED: Allow 3-5 clusters
    if config['kmeans']['n_clusters'] not in [3, 4, 5]:
        raise ValidationError("n_clusters must be 3, 4, or 5")
```

**Step 3**: Update Stage 6 validation (ml_analysis_generation.py:55)
```python
# K-Means parameters
N_CLUSTERS = 5  # Changed from 3
```

**Step 4**: Test with both cluster counts
```bash
# Test 3 clusters
jq '.kmeans.n_clusters = 3' config/model_hyperparameters.json > tmp.json && mv tmp.json config/model_hyperparameters.json
python rumiai_ml_batch.py --client test --bucket 18-33s

# Test 5 clusters
jq '.kmeans.n_clusters = 5' config/model_hyperparameters.json > tmp.json && mv tmp.json config/model_hyperparameters.json
python rumiai_ml_batch.py --client test --bucket 18-33s
```

### 8.3 How to Add Model Performance Threshold

**Scenario**: Fail training if RF accuracy < 0.60 (random guessing baseline)

**Step 1**: Add validation to `validate_stage_output()` (model_training.py:889-984)
```python
def validate_stage_output(bucket: str, windows: List[str], bucket_base: str) -> None:
    # ... existing validation ...

    # Load model_metrics.json
    metrics_path = os.path.join(bucket_base, 'models/model_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # NEW: Validate RF accuracy threshold
    if metrics['video_level_rf'].get('trained', True):
        rf_accuracy = metrics['video_level_rf']['accuracy']

        # Hard threshold: Fail if < 0.60
        if rf_accuracy < 0.60:
            raise ValidationError(
                f"Video-level RF accuracy too low: {rf_accuracy:.2f} < 0.60. "
                f"Model is no better than random guessing. "
                f"Increase training data or improve feature quality."
            )

        logger.info(f"✓ RF accuracy meets threshold: {rf_accuracy:.2f} >= 0.60")
```

**Step 2**: Add to orchestrator error handling (rumiai_ml_batch.py:1583-1591)
```python
except ValidationError as e:
    # Output validation failed (includes accuracy threshold)
    logger.error(f"Stage 5 output validation failed for bucket {bucket_name}: {e}")
    print(f"✗ Bucket {bucket_name}: Output validation failed (skipping)")
    print("   This may indicate low-quality models for this bucket")
    continue  # Skip this bucket, continue pipeline
```

**Step 3**: Test with low-quality data
```python
# Create test data with random labels (should fail threshold)
import pandas as pd
import numpy as np

df = pd.read_csv('{bucket_path}/ml_analysis/rf_transformed.csv')
df['is_top_performer'] = np.random.randint(0, 2, len(df))  # Random labels
df.to_csv('{bucket_path}/ml_analysis/rf_transformed.csv', index=False)

# Run Stage 5 - should fail with ValidationError
```

### 8.4 How to Add a New Model Type

**Scenario**: Add GradientBoosting as alternative to RandomForest

**Step 1**: Update config schema
```json
// config/model_hyperparameters.json
{
  "model_type": "gradient_boosting",  // NEW: "random_forest" or "gradient_boosting"
  "random_forest": { /* ... */ },
  "gradient_boosting": {  // NEW
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": 42
  }
}
```

**Step 2**: Update training loop (model_training.py:422-621)
```python
def train_bucket_models(bucket, windows, bucket_base, config, selection_strategy):
    # ... existing code ...

    # MODIFIED: Select model type
    model_type = config.get('model_type', 'random_forest')

    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        rf_video = RandomForestClassifier(**config['random_forest'])
    elif model_type == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        rf_video = GradientBoostingClassifier(**config['gradient_boosting'])
    else:
        raise ConfigError(f"Unknown model_type: {model_type}")

    # Rest of training code unchanged
    rf_video.fit(X, y)
    joblib.dump(rf_video, model_path)
```

**Step 3**: Update metrics generation (model_training.py:285-419)
```python
def generate_model_metrics(...):
    # ... existing code ...

    # MODIFIED: Add model_type to metrics
    metrics["video_level_rf"] = {
        "model_type": config.get('model_type', 'random_forest'),  # NEW
        "trained": True,
        # ... rest unchanged
    }
```

**Step 4**: Test both model types
```bash
# Test RandomForest
jq '.model_type = "random_forest"' config/model_hyperparameters.json > tmp.json && mv tmp.json config/model_hyperparameters.json
python rumiai_ml_batch.py --client test --bucket 18-33s

# Test GradientBoosting
jq '.model_type = "gradient_boosting"' config/model_hyperparameters.json > tmp.json && mv tmp.json config/model_hyperparameters.json
python rumiai_ml_batch.py --client test --bucket 18-33s

# Compare accuracy
cat {bucket_path}/models/model_metrics.json | jq '.video_level_rf.accuracy'
```

---

## 9. Dependencies

### 9.1 Upstream Dependencies

**Stage 4: Feature Transformation**

**File**: `rumiai_v2/processors/feature_transformation.py`

**Required Outputs**:
- `ml_analysis/rf_transformed.csv` (video-level RF input)
- `ml_analysis/{window}_rf_transformed.csv` (window-level RF inputs)
- `ml_analysis/{window}_km_transformed.csv` (K-Means inputs)
- `ml_analysis/{window}_scalers.pkl` (scaler objects)

**Contract Verification**:
```python
# Stage 5 validates Stage 4 outputs (model_training.py:85-149)
validate_stage4_outputs(bucket, windows, bucket_base, selection_strategy)
```

**Stage 4 Performance Impact**:
- Stage 4 duration: ~5-10s
- Stage 5 duration: ~7-15s
- **Total**: ~12-25s for both stages

### 9.2 Downstream Dependencies

**Stage 6: ML Analysis Generation**

**File**: `ml_pipeline/stage6_analysis/ml_analysis_generation.py`

**Required Inputs** (from Stage 5):
- `models/rf_video_{bucket}.pkl` (optional, checked via model_metrics.json)
- `models/{window}_kmeans_{bucket}.pkl` (required)
- `models/{window}_X_data_{bucket}.pkl` (required, for feature names)
- `models/{window}_scalers_{bucket}.pkl` (required, for inference)
- `models/model_metrics.json` (required, checks `trained` field)

**Contract Verification**:
```python
# Stage 6 validates Stage 5 outputs (ml_analysis_generation.py:71-213)
validate_stage_dependencies(bucket_path, bucket, windows)

# Stage 6 reads trained field
with open('models/model_metrics.json', 'r') as f:
    metrics = json.load(f)
    rf_trained = metrics['video_level_rf']['trained']

# Stage 6 handles optional RF models
video_rf_json = generate_video_rf_json(bucket_path, bucket)
if video_rf_json is not None:
    # Generate RF JSON
else:
    logger.info("⏭ Skipped rf_video_analysis.json (RF not trained)")
```

**C7 Compatibility Verified**:
- Stage 5 sets `trained: false` when RF skipped (TOP mode)
- Stage 6 reads `trained` field and skips RF JSON generation
- ✅ No breaking changes when RF models missing

### 9.3 External Dependencies

**Python Packages**:
```python
scikit-learn >= 0.24.0   # RandomForestClassifier, KMeans, metrics
pandas >= 1.3.0          # CSV I/O
numpy >= 1.21.0          # Array operations
joblib >= 1.0.0          # Model serialization
```

**Install Command**:
```bash
pip install scikit-learn>=0.24.0 pandas>=1.3.0 numpy>=1.21.0 joblib>=1.0.0
```

**Version Compatibility**:
- sklearn 0.24.0+: Required for `feature_names_in_` attribute
- pandas 1.3.0+: Required for `pd.read_csv()` performance improvements
- joblib 1.0.0+: Required for pickle protocol 5 support

### 9.4 Shared Configuration

**Bucket Definitions**:
```python
# config/bucket_definitions.py
from config.bucket_definitions import BUCKET_WINDOWS

bucket = "18-33s"
windows = BUCKET_WINDOWS[bucket]  # ['hook', 'middle_1', ..., 'closing']
```

**Used By**:
- Stage 3 (Feature Aggregation)
- Stage 4 (Feature Transformation)
- **Stage 5 (ML Model Training)** ← This stage
- Stage 6 (ML Analysis Generation)
- Stage 7 (LLM Analysis)

**Single Source of Truth**: All stages import from `config/bucket_definitions.py`

---

## 10. Testing

### 10.1 Unit Tests

**Test File** (create): `tests/test_stage5_model_training.py`

```python
import pytest
import pandas as pd
import numpy as np
from rumiai_v2.processors.model_training import (
    normalize_feature_name,
    get_top_cluster_features,
    validate_kmeans_feature_naming,
    load_model_config
)

def test_normalize_feature_name():
    """Test feature name normalization."""
    assert normalize_feature_name('eye_contact_rate_scaled') == 'eye_contact_rate'
    assert normalize_feature_name('has_captions_encoded') == 'has_captions'
    assert normalize_feature_name('scene_count') == 'scene_count'
    assert normalize_feature_name('word_count_log_scaled') == 'word_count'

def test_get_top_cluster_features():
    """Test cluster feature extraction by variance."""
    from sklearn.cluster import KMeans

    # Mock data: 3 clusters, 5 features
    X = np.array([
        [0.8, 0.3, 0.6, 0.5, 0.2],  # Cluster 0
        [0.2, 0.7, 0.5, 0.5, 0.3],  # Cluster 1
        [0.5, 0.4, 0.9, 0.5, 0.2],  # Cluster 2
    ])
    feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Get top 3 features
    top_features = get_top_cluster_features(kmeans, feature_names, n=3)

    # f4 has zero variance (all 0.5) → should be last
    # f1, f2, f3 have high variance → should be top 3
    assert len(top_features) == 3
    assert 'f4' not in top_features

def test_validate_kmeans_feature_naming_pass():
    """Test K-Means feature naming validation (pass case)."""
    # Create mock CSV with 80%+ transformed features
    df = pd.DataFrame({
        'video_id': [1, 2, 3],
        'eye_contact_rate_scaled': [0.5, 0.6, 0.7],
        'scene_count_scaled': [0.3, 0.4, 0.5],
        'has_captions_encoded': [0, 1, 1],
        'energy_level_scaled': [0.2, 0.3, 0.4],
        'joy': [1, 0, 1]  # One-hot encoded (no suffix)
    })

    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name

    # Should pass (4/5 = 80% have suffixes)
    validate_kmeans_feature_naming(csv_path)

    # Cleanup
    import os
    os.remove(csv_path)

def test_validate_kmeans_feature_naming_fail():
    """Test K-Means feature naming validation (fail case)."""
    # Create mock CSV with <80% transformed features
    df = pd.DataFrame({
        'video_id': [1, 2, 3],
        'eye_contact_rate_scaled': [0.5, 0.6, 0.7],
        'scene_count': [1, 2, 3],  # Missing _scaled
        'has_captions': [0, 1, 1],  # Missing _encoded
        'energy_level': [0.2, 0.3, 0.4],  # Missing _scaled
        'joy': [1, 0, 1]
    })

    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name

    # Should fail (1/5 = 20% < 80%)
    with pytest.raises(ValidationError):
        validate_kmeans_feature_naming(csv_path)

    # Cleanup
    import os
    os.remove(csv_path)

def test_load_model_config_file_exists():
    """Test loading config from file."""
    import tempfile
    import json

    # Create temp config
    config = {
        "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "kmeans": {"n_clusters": 3, "random_state": 42, "n_init": 10}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    # Mock config path
    import rumiai_v2.processors.model_training as mt
    original_path = 'config/model_hyperparameters.json'
    mt.CONFIG_PATH = config_path  # Override

    # Load config
    loaded_config = load_model_config()

    assert loaded_config['random_forest']['n_estimators'] == 100
    assert loaded_config['kmeans']['n_clusters'] == 3

    # Cleanup
    import os
    os.remove(config_path)

def test_load_model_config_file_missing():
    """Test fallback to defaults when config missing."""
    # Point to non-existent file
    import rumiai_v2.processors.model_training as mt
    mt.CONFIG_PATH = '/tmp/nonexistent_config.json'

    # Should return defaults (with warning logged)
    config = load_model_config()

    assert config['random_forest']['n_estimators'] == 100
    assert config['kmeans']['n_clusters'] == 3
```

**Run Tests**:
```bash
pytest tests/test_stage5_model_training.py -v
```

### 10.2 Integration Test

**Test Scenario**: Train models on real bucket data

```python
import os
import shutil
from rumiai_v2.processors.model_training import run_stage5_training

def test_stage5_integration():
    """Test Stage 5 with real bucket data."""
    # Setup: Copy test bucket data
    test_bucket_path = '/tmp/test_bucket_18-33s'
    shutil.copytree('tests/fixtures/bucket_18-33s', test_bucket_path)

    # Verify Stage 4 files exist
    assert os.path.exists(f'{test_bucket_path}/ml_analysis/rf_transformed.csv')
    assert os.path.exists(f'{test_bucket_path}/ml_analysis/hook_rf_transformed.csv')

    # Run Stage 5
    config = {
        "bucket": "18-33s",
        "strategy": "contrastive",
        "video_count": 100
    }

    success, output_files, elapsed_time = run_stage5_training(
        bucket_path=test_bucket_path,
        config=config,
        selection_strategy="contrastive"
    )

    # Verify success
    assert success == True
    assert len(output_files) == 26  # 1 + 6 + 6 + 6 + 6 + 1
    assert elapsed_time < 30  # Should complete in <30s

    # Verify model files created
    assert os.path.exists(f'{test_bucket_path}/models/rf_video_18-33s.pkl')
    assert os.path.exists(f'{test_bucket_path}/models/hook_kmeans_18-33s.pkl')
    assert os.path.exists(f'{test_bucket_path}/models/model_metrics.json')

    # Verify model loadable
    import joblib
    rf_model = joblib.load(f'{test_bucket_path}/models/rf_video_18-33s.pkl')
    assert rf_model is not None

    # Verify metrics valid
    import json
    with open(f'{test_bucket_path}/models/model_metrics.json') as f:
        metrics = json.load(f)

    assert metrics['bucket'] == '18-33s'
    assert metrics['video_level_rf']['trained'] == True
    assert 0 <= metrics['video_level_rf']['accuracy'] <= 1.0

    # Cleanup
    shutil.rmtree(test_bucket_path)
```

**Run Integration Test**:
```bash
pytest tests/test_stage5_integration.py -v -s
```

### 10.3 Manual Validation Checklist

**Pre-Flight**:
- [ ] Stage 4 checkpoint exists
- [ ] All 13 Stage 4 CSVs exist
- [ ] Video count ≥ 50 (contrastive) or ≥ 30 (top)
- [ ] K-Means features have 80%+ transformation suffixes

**During Training**:
- [ ] No NaN values in input data
- [ ] RF training completes (or skips in TOP mode)
- [ ] K-Means training completes for all windows
- [ ] model_metrics.json generated

**Post-Training**:
- [ ] All model files created (26 for 18-33s)
- [ ] model_metrics.json has correct schema
- [ ] RF accuracy in [0.0, 1.0] (if trained)
- [ ] Silhouette scores in [-1.0, 1.0]
- [ ] Cluster sizes sum to total_videos
- [ ] Stage 5 checkpoint created

**Stage 6 Compatibility**:
- [ ] Stage 6 can load all models
- [ ] Stage 6 validates model_metrics.json
- [ ] Stage 6 handles RF skipped (TOP mode)
- [ ] Stage 6 generates JSONs successfully

---

## Related Documentation

- **PRODUCTION_FLOW.md**: Complete pipeline overview (Stage 5: lines 550-624)
- **MLModelTrainingCHILDTI.md**: Technical specification (3263 lines, design rationale)
- **STAGE_2.6_2.7_IMPL.md**: Example stage implementation guide (similar structure)
- **SystemArchitecturev2.md**: Overall RumiAI architecture
- **MLROADMAP.md**: ML pipeline future development plans

---

**Document End** | Version 1.0 | Last Updated: 2025-01-03
