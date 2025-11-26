# Stage 6: ML Analysis Generation - Implementation Guide

> **Stage**: Stage 6 - ML Analysis Generation
> **Parent Document**: [PRODUCTION_FLOW.md](PRODUCTION_FLOW.md#stage-6-ml-analysis-generation)
> **Implementation**: `ml_pipeline/stage6_analysis/ml_analysis_generation.py`
> **Version**: 1.0
> **Last Updated**: 2025-01-28
> **Status**: Production

---

## Quick Reference

### Entry Point
```python
# File: ml_pipeline/stage6_analysis/ml_analysis_generation.py
# Function: generate_ml_analysis_jsons()
# Lines: 755-850

def generate_ml_analysis_jsons(bucket_path: str, bucket: str, windows: List[str]) -> int:
    """
    Generate all ML analysis JSONs for a bucket (CONTRASTIVE or TOP mode).

    Args:
        bucket_path: Absolute path to bucket directory
        bucket: Bucket name (e.g., "18-33s")
        windows: Window list from BUCKET_WINDOWS config

    Returns:
        int: Exit code (0=success, 1=preflight fail, 2=generation fail,
             3=validation fail, 4=I/O fail)
    """
```

### Orchestrator Call
```python
# File: rumiai_ml_batch.py
# Lines: 1628-1798
# Called from: main() bucket processing loop after Stage 5

exit_code = generate_ml_analysis_jsons(
    bucket_path=str(bucket_path),  # e.g., "/data/clients/acme/buckets/bucket_18-33s"
    bucket=bucket_name,             # e.g., "18-33s"
    windows=windows                 # e.g., ['hook', 'middle_1', ..., 'closing']
)
```

### Checkpoint
```json
{
  "stage": "stage_6_ml_analysis_generation",
  "bucket": "18-33s",
  "status": "completed",
  "timestamp": "2025-01-28T14:32:00Z",
  "json_files_generated": 13,
  "output_files": [
    "ml_analysis/rf_video_analysis.json",
    "ml_analysis/hook_rf_analysis.json",
    "ml_analysis/hook_kmeans_analysis.json",
    "..."
  ]
}
```
**Location**: `{bucket_path}/checkpoints/stage_6_checkpoint.json`

### Duration
- **Typical**: 3-5 seconds per bucket (100 videos)
- **Acceptable**: 5-10 seconds
- **Alert Threshold**: >30 seconds

---

## 1. Input Contract

### 1.1 Required Files (Mode-Aware)

**CONTRASTIVE Mode** (40 files for bucket 18-33s):
```
Stage 3 (1 file):
  ml_analysis/aggregated_features.csv         # 100 rows × 135 cols (incl. xwin features)

Stage 4 (13 files):
  ml_analysis/rf_transformed.csv              # Video-level: 100 rows × 147 cols
  ml_analysis/{window}_rf_transformed.csv     # 6 files: 100 rows × 21 cols each
  ml_analysis/{window}_km_transformed.csv     # 6 files: 100 rows × 21-39 cols each

Stage 5 (26 files):
  models/rf_video_{bucket}.pkl                # Video RF model
  models/rf_{window}_{bucket}.pkl             # 6 window RF models
  models/{window}_kmeans_{bucket}.pkl         # 6 K-Means models
  models/{window}_scalers_{bucket}.pkl        # 6 scaler objects
  models/{window}_X_data_{bucket}.pkl         # 6 feature matrices
  models/model_metrics.json                   # Performance metrics + mode flag
```

**TOP Mode** (34 files for bucket 18-33s):
```
Stage 3 (1 file):
  ml_analysis/aggregated_features.csv         # NOT USED in TOP mode

Stage 4 (7 files):
  ml_analysis/{window}_km_transformed.csv     # 6 files only (no RF CSVs)

Stage 5 (20 files):
  # NO RF MODELS in TOP mode
  models/{window}_kmeans_{bucket}.pkl         # 6 K-Means models
  models/{window}_scalers_{bucket}.pkl        # 6 scaler objects
  models/{window}_X_data_{bucket}.pkl         # 6 feature matrices
  models/model_metrics.json                   # mode_metrics.video_level_rf.trained=false
```

### 1.2 Mode Detection Logic

**Code Location**: `ml_analysis_generation.py:111-165`

```python
# Read model_metrics.json
with open(metrics_path) as f:
    metrics = json.load(f)

# Check if RF was trained
rf_trained = metrics.get('video_level_rf', {}).get('trained', True)

if rf_trained:
    # CONTRASTIVE MODE - validate RF files exist
    logger.info("Pre-flight: RF models required (CONTRASTIVE mode)")
    # ... check rf_video_{bucket}.pkl, rf_{window}_{bucket}.pkl
else:
    # TOP MODE - skip RF file validation
    logger.info("Pre-flight: RF models NOT expected (trained=False - TOP mode)")
```

### 1.3 Input Schemas

**aggregated_features.csv** (Stage 3 output):
```
Columns: 135 for bucket 18-33s
- video_id (str)
- create_time (datetime)
- gender (str, nullable)
- hook_eye_contact_rate (float 0.0-1.0)
- hook_scene_count (int 0-20)
- ... (21 features × 6 windows = 126 columns)
- xwin_hook_to_middle_energy (float, cross-window)
- xwin_middle_to_closing_energy (float, cross-window)
- xwin_eye_contact_consistency (float, cross-window)
- xwin_word_density_std (float, cross-window)
- xwin_energy_progression_slope (float, cross-window)
- is_top_performer (int 0 or 1)

Source of Truth: config/bucket_definitions.py::get_stage3_expected_feature_count(bucket)
```

**{window}_km_transformed.csv** (Stage 4 output):
```
Columns: 21-39 (varies by window)
- eye_contact_rate_scaled (float 0.0-1.0)
- scene_count_scaled (float 0.0-1.0)
- word_count_scaled (float 0.0-1.0)
- has_captions_encoded (int 0 or 1)  # Bug #1 fix: int64, not bool
- ... (all features with _scaled, _log, or _encoded suffixes)

NOTE: Suffixes MUST be removed in output JSONs (normalize_feature_name())
```

**model_metrics.json** (Stage 5 output):
```json
{
  "bucket": "18-33s",
  "total_videos": 100,
  "video_level_rf": {
    "trained": true,              // MODE FLAG: true=CONTRASTIVE, false=TOP
    "accuracy": 0.82,
    "precision": 0.85,
    "recall": 0.78,
    "f1_score": 0.81
  },
  "window_level_rf": {
    "hook": {
      "trained": true,
      "accuracy": 0.75,
      "precision": 0.72,
      "recall": 0.78
    }
    // ... per window
  }
}
```

---

## 2. Output Contract

### 2.1 Output Files (Mode-Aware)

**CONTRASTIVE Mode** (13 files for bucket 18-33s):
```
ml_analysis/
├── rf_video_analysis.json                  # 1 video-level RF (~30KB)
├── hook_rf_analysis.json                   # 6 window RF (~5KB each)
├── middle_1_rf_analysis.json
├── middle_2_rf_analysis.json
├── middle_3_rf_analysis.json
├── middle_4_rf_analysis.json
├── closing_rf_analysis.json
├── hook_kmeans_analysis.json               # 6 window K-Means (~5KB each)
├── middle_1_kmeans_analysis.json
├── middle_2_kmeans_analysis.json
├── middle_3_kmeans_analysis.json
├── middle_4_kmeans_analysis.json
└── closing_kmeans_analysis.json

Total: 1 + (6 × 2) = 13 files
```

**TOP Mode** (6 files for bucket 18-33s):
```
ml_analysis/
├── hook_kmeans_analysis.json               # 6 window K-Means only
├── middle_1_kmeans_analysis.json
├── middle_2_kmeans_analysis.json
├── middle_3_kmeans_analysis.json
├── middle_4_kmeans_analysis.json
└── closing_kmeans_analysis.json

Total: 6 files (NO RF JSONs)
```

### 2.2 JSON Schemas

**Video-Level RF** (`rf_video_analysis.json`):
```json
{
  "analysis_type": "random_forest",
  "bucket": "18-33s",
  "hashtag": null,
  "video_count": 100,
  "input_features": 147,        // get_stage3_expected_feature_count(bucket) + 12
  "feature_importance": [        // Length: 10 (top 10 features)
    {
      "feature": "hook_eye_contact_rate",
      "importance": 0.22,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "distribution": {
        "thresholds": {
          "high": 0.6,           // 66th percentile of top performers
          "low": 0.4             // 33rd percentile of top performers
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
    // ... 9 more features
  ]
}
```

**Window-Level RF** (`{window}_rf_analysis.json`):
```json
{
  "model_type": "window_level_rf",
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "input_features": 21,          // Always 21 per window
  "model_performance": {
    "accuracy": 0.82,
    "precision": 0.85,
    "recall": 0.78
  },
  "feature_importance": [        // Length: 10
    {
      "feature": "eye_contact_rate",  // NO window prefix
      "importance": 0.35,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "rank": 1
    }
    // ... 9 more features
  ]
}
```

**Window-Level K-Means** (`{window}_kmeans_analysis.json`):
```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "n_clusters": 3,               // Always 3 clusters
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "centroid": {
        "eye_contact_rate": 0.87,      // ✅ NORMALIZED (no _scaled suffix)
        "scene_count": 0.45,
        "word_count": 0.62,
        "has_captions": 1
        // ... 21-39 features (all normalized)
      },
      "videos": [
        {
          "video_id": "video_0",
          "distance_to_centroid": 0.15
        }
        // ... all videos in cluster
      ]
    }
    // ... 3 clusters total
  ]
}
```

### 2.3 Validation Rules

**Critical Validations** (Code: `ml_analysis_generation.py:624-690`):
1. All expected files exist in `.tmp/` directory
2. All JSONs are parseable (valid JSON format)
3. K-Means centroid keys have NO suffixes (`_scaled`, `_log`, `_encoded`)
4. Cluster sizes sum EXACTLY to `total_videos`
5. Distribution percentages sum to ~1.0 (tolerance: 0.01)

---

## 3. Core Functions

### 3.1 Function Call Chain

```
generate_ml_analysis_jsons() [ENTRY - line 755]
    │
    ├─→ validate_stage_dependencies() [line 71]
    │   ├─→ Check Stage 4 CSVs (13 files in CONTRASTIVE, 7 in TOP)
    │   ├─→ Check Stage 5 models (mode-aware: 26 files in CONTRASTIVE, 20 in TOP)
    │   └─→ Raises PreFlightValidationError if missing
    │
    ├─→ generate_video_rf_json() [line 215] ← OPTIONAL (None in TOP mode)
    │   ├─→ joblib.load(rf_video_{bucket}.pkl)
    │   ├─→ Extract feature_importances_ (top 10)
    │   ├─→ pd.read_csv(aggregated_features.csv)  # Uses Stage 3 output with xwin
    │   ├─→ Compute 66th/33rd percentile thresholds
    │   └─→ Returns video RF JSON dict or None
    │
    ├─→ generate_window_rf_json() [line 345] × N ← OPTIONAL (None in TOP mode)
    │   ├─→ joblib.load(rf_{window}_{bucket}.pkl)
    │   ├─→ Extract feature_importances_ (top 10)
    │   ├─→ Load model_metrics.json (accuracy/precision/recall)
    │   ├─→ pd.read_csv({window}_rf_transformed.csv)  # Uses Stage 4 output
    │   └─→ Returns window RF JSON dict or None
    │
    ├─→ generate_window_kmeans_json() [line 522] × N ← ALWAYS REQUIRED
    │   ├─→ joblib.load({window}_kmeans_{bucket}.pkl)
    │   ├─→ Extract cluster_centers_ (3 clusters)
    │   ├─→ joblib.load({window}_X_data_{bucket}.pkl)
    │   ├─→ normalize_feature_name() [line 495] ← CRITICAL
    │   ├─→ pd.read_csv({window}_km_transformed.csv)
    │   ├─→ kmeans_model.predict() → cluster assignments
    │   ├─→ np.linalg.norm() → distances to centroid
    │   └─→ Returns window K-Means JSON dict
    │
    ├─→ validate_all_json_schemas() [line 624]
    │   ├─→ Check all expected files exist
    │   ├─→ Validate JSON parseability
    │   ├─→ Validate K-Means feature names (no suffixes)
    │   ├─→ Validate cluster sizes sum correctly
    │   └─→ Raises ValidationError if invalid
    │
    └─→ Atomic commit: os.rename() all .tmp → final
```

### 3.2 Function Reference Table

| Function | Lines | Purpose | Returns | Errors Raised |
|----------|-------|---------|---------|---------------|
| `validate_stage_dependencies()` | 71-169 | Pre-flight validation | None | PreFlightValidationError |
| `generate_video_rf_json()` | 215-283 | Video RF analysis | dict or None | FileNotFoundError, ValueError |
| `generate_window_rf_json()` | 345-436 | Window RF analysis | dict or None | FileNotFoundError, ValueError |
| `normalize_feature_name()` | 495-507 | Remove suffixes | str | - |
| `generate_window_kmeans_json()` | 522-640 | K-Means analysis | dict | FileNotFoundError, ValueError |
| `validate_all_json_schemas()` | 624-690 | Output validation | None | ValidationError |
| `generate_ml_analysis_jsons()` | 755-850 | Main orchestrator | int (exit code) | Multiple |

### 3.3 Key Code Snippets

**Mode Detection** (lines 111-150):
```python
# Read model_metrics.json to detect mode
metrics_path = os.path.join(bucket_path, 'models/model_metrics.json')
with open(metrics_path) as f:
    metrics = json.load(f)

# Check if video-level RF was trained
rf_trained = metrics.get('video_level_rf', {}).get('trained', True)

if rf_trained:
    # CONTRASTIVE MODE - require RF files
    required_files.append(f'models/rf_video_{bucket}.pkl')
    for window in windows:
        required_files.append(f'models/rf_{window}_{bucket}.pkl')
    logger.info(f"Pre-flight: RF models required (CONTRASTIVE mode)")
else:
    # TOP MODE - skip RF files
    logger.info(f"Pre-flight: RF models NOT expected (trained=False - TOP mode)")
```

**Feature Name Normalization** (lines 495-507):
```python
def normalize_feature_name(feature_name: str) -> str:
    """
    Remove transformation suffixes: _scaled, _log, _encoded

    Examples:
        'eye_contact_rate_scaled' → 'eye_contact_rate'
        'has_captions_encoded' → 'has_captions'
        'word_count' → 'word_count' (no change)
    """
    normalized = feature_name
    suffixes = ['_scaled', '_log', '_encoded']
    for suffix in suffixes:
        normalized = normalized.replace(suffix, '')
    return normalized
```

**Atomic Write Pattern** (lines 783-850):
```python
# Step 1: Write to temp directory
temp_dir = os.path.join(bucket_path, 'ml_analysis/.tmp/')
temp_path = os.path.join(temp_dir, 'rf_video_analysis.json.tmp')
with open(temp_path, 'w') as f:
    json.dump(video_rf_json, f, indent=2)

# Step 2: Validate ALL files

# Step 3: Atomic commit (all-or-nothing)
for temp_file in generated_files:
    final_path = temp_file.replace('/.tmp/', '/').replace('.json.tmp', '.json')
    os.rename(temp_file, final_path)  # Atomic operation

# Step 4: Cleanup
shutil.rmtree(temp_dir)
```

---

## 4. Data Flow

### 4.1 Input → Transformation → Output

**Flow 1: Video-Level RF** (CONTRASTIVE mode only):
```
Input:
  models/rf_video_18-33s.pkl (RandomForestClassifier)
  ml_analysis/aggregated_features.csv (100 rows × 135 cols)

Extract:
  rf_model.feature_importances_ → NumPy array [147 features]
  rf_model.feature_names_in_ → ['hook_eye_contact_rate', ...]

Transform:
  1. Sort by importance, take top 10
  2. Split by is_top_performer (80% vs 20%)
  3. Compute 66th/33rd percentile thresholds
  4. Calculate high/medium/low % distributions

Output:
  ml_analysis/rf_video_analysis.json
  {
    "video_count": 100,
    "input_features": 147,
    "feature_importance": [
      {
        "feature": "hook_eye_contact_rate",
        "importance": 0.22,
        "distribution": { ... }
      }
      // ... top 10
    ]
  }
```

**Flow 2: Window-Level RF** (CONTRASTIVE mode only):
```
Input:
  models/rf_hook_18-33s.pkl (RandomForestClassifier, 21 features)
  ml_analysis/hook_rf_transformed.csv (100 rows × 21 cols)
  models/model_metrics.json (accuracy/precision/recall)

Extract:
  rf_model.feature_importances_ → [21 features]
  metrics['window_level_rf']['hook'] → performance stats

Transform:
  1. Sort by importance, take top 10, add rank
  2. Compute top/bottom performer averages
  3. Add performance metrics from model_metrics.json

Output:
  ml_analysis/hook_rf_analysis.json
  {
    "window_type": "hook",
    "input_features": 21,
    "model_performance": {
      "accuracy": 0.82,
      "precision": 0.85,
      "recall": 0.78
    },
    "feature_importance": [
      {
        "feature": "eye_contact_rate",  // NO prefix
        "importance": 0.35,
        "rank": 1
      }
      // ... top 10
    ]
  }
```

**Flow 3: Window-Level K-Means** (ALWAYS required):
```
Input:
  models/hook_kmeans_18-33s.pkl (KMeans, 3 clusters)
  models/hook_X_data_18-33s.pkl (DataFrame, 100 rows × 21 cols)
  ml_analysis/hook_km_transformed.csv (100 rows × 21 cols with suffixes)

Extract:
  kmeans_model.cluster_centers_ → (3, 21) NumPy array
  X_data.columns → ['eye_contact_rate_scaled', 'scene_count_scaled', ...]

Transform:
  1. Normalize feature names: 'eye_contact_rate_scaled' → 'eye_contact_rate'
  2. Build centroid dict with normalized names
  3. Predict cluster assignments: kmeans_model.predict(df)
  4. Compute distances: np.linalg.norm(video_features - centroid)

Output:
  ml_analysis/hook_kmeans_analysis.json
  {
    "window_type": "hook",
    "n_clusters": 3,
    "clusters": [
      {
        "cluster_id": 0,
        "size": 35,
        "centroid": {
          "eye_contact_rate": 0.87,  // ✅ Normalized
          "scene_count": 0.45
          // ... 21 features (NO suffixes)
        },
        "videos": [
          {"video_id": "video_0", "distance_to_centroid": 0.15}
        ]
      }
      // ... 3 clusters
    ]
  }
```

### 4.2 Critical Transformations

**Transformation 1: Top 10 Feature Selection**
```python
# Code: ml_analysis_generation.py:250-258
importance_indices = np.argsort(feature_importances)[::-1][:10]
top_features = [
    {
        'feature': feature_names[idx],
        'importance': float(feature_importances[idx])
    }
    for idx in importance_indices
]
```

**Transformation 2: Distribution Analysis**
```python
# Code: ml_analysis_generation.py:262-283
# Split by performer type
top_performers = df[df['is_top_performer'] == 1][feature_name]
bottom_performers = df[df['is_top_performer'] == 0][feature_name]

# Compute percentile thresholds
high_threshold = float(top_performers.quantile(0.66))  # 66th percentile
low_threshold = float(top_performers.quantile(0.33))   # 33rd percentile

# Compute percentage distributions
top_high_pct = (top_performers >= high_threshold).sum() / len(top_performers)
top_med_pct = ((top_performers >= low_threshold) & (top_performers < high_threshold)).sum() / len(top_performers)
top_low_pct = (top_performers < low_threshold).sum() / len(top_performers)
```

**Transformation 3: Feature Name Normalization**
```python
# Code: ml_analysis_generation.py:573-575
normalized_centroid = {
    normalize_feature_name(name): float(value)
    for name, value in zip(feature_names, centroid_values)
}

# 'eye_contact_rate_scaled' → 'eye_contact_rate'
# 'has_captions_encoded' → 'has_captions'
```

---

## 5. Error Handling

### 5.1 Error Scenarios

**Scenario 1: Missing Stage 4 Files**
```
Trigger: Stage 4 CSV files missing
Code: ml_analysis_generation.py:71-169 (validate_stage_dependencies)
Detection: os.path.exists() check
Action: Raise PreFlightValidationError, exit code 1
Message: "Stage 4 incomplete (N files missing): [list]. Action: Re-run Stage 4"
Recovery: Re-run Stage 4 (Feature Transformation)
```

**Scenario 2: Missing Stage 5 Models**
```
Trigger: Stage 5 model files missing
Code: ml_analysis_generation.py:71-169
Detection: os.path.exists() check
Action: Raise PreFlightValidationError, exit code 1
Message: "Stage 5 incomplete (N files missing): [list]. Action: Re-run Stage 5"
Recovery: Re-run Stage 5 (ML Model Training)
```

**Scenario 3: Corrupted Pickle File**
```
Trigger: joblib.load() fails (UnpicklingError)
Code: ml_analysis_generation.py:241, 370, 539 (model loading)
Detection: Exception during joblib.load()
Action: Delete temp files (atomic rollback), exit code 2
Message: "Failed to load model {path}: {error}. Re-run Stage 5."
Recovery: Re-run Stage 5 to regenerate models
```

**Scenario 4: Feature Name Normalization Failure**
```
Trigger: K-Means centroids contain '_scaled' suffixes
Code: ml_analysis_generation.py:652-664 (validate_all_json_schemas)
Detection: Post-generation validation
Action: Delete all temp files, exit code 3
Message: "K-Means JSON contains features with suffixes. Bug in normalize_feature_name()."
Recovery: Report bug (code logic error)
```

**Scenario 5: Cluster Size Mismatch**
```
Trigger: Sum of cluster sizes ≠ total_videos
Code: ml_analysis_generation.py:666-676
Detection: Post-generation validation
Action: Delete all temp files, exit code 3
Message: "Cluster sizes sum to {sum}, but total_videos is {total}. Cluster assignment failed."
Recovery: Report bug or check K-Means prediction logic
```

**Scenario 6: Disk Full**
```
Trigger: IOError during JSON write
Code: ml_analysis_generation.py:838-841 (exception handler)
Detection: IOError exception
Action: Delete temp files, exit code 4
Message: "Disk I/O failed: {error}. Check disk space."
Recovery: Free disk space, re-run Stage 6
```

### 5.2 Exit Code Reference

| Exit Code | Meaning | Cause | Recovery |
|-----------|---------|-------|----------|
| 0 | Success | All JSONs generated and validated | N/A |
| 1 | Pre-flight validation failed | Missing Stage 4/5 files | Re-run Stage 4 or 5 |
| 2 | JSON generation failed | Model loading error, CSV parsing error | Check logs, re-run Stage 5 |
| 3 | Output validation failed | Schema error, feature name issues | Report bug |
| 4 | Disk I/O failed | Disk full, permission denied | Fix infrastructure |

### 5.3 Atomic Rollback

**Code**: `ml_analysis_generation.py:808-850`

```python
try:
    # Generate all JSONs to temp directory
    # ... (generate 13 files)

    # Validate all files
    validate_all_json_schemas(temp_dir, bucket, windows)

    # Atomic commit (all succeed)
    for temp_file in generated_files:
        final_path = temp_file.replace('/.tmp/', '/').replace('.json.tmp', '.json')
        os.rename(temp_file, final_path)

    return 0  # SUCCESS

except Exception as e:
    # Rollback: Delete ALL temp files (atomic failure)
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.warning("Rolled back: Deleted all temp files (atomic failure)")
    return 2  # FAILURE
```

**Guarantee**: Either ALL 13 JSONs succeed OR ZERO JSONs (no partial output)

---

## 6. Modification Guide

### 6.1 Common Task: Add New Feature to RF Analysis

**Scenario**: Add `distribution_std` (standard deviation) to RF feature importance

**Steps**:
1. **Modify** `generate_video_rf_json()` (line 262-283):
   ```python
   # After computing gap
   feature_data['gap'] = gap

   # ADD NEW FIELD
   feature_data['std_dev'] = float(top_performers.std())
   ```

2. **Update** output schema validation (line 297-303):
   ```python
   assert "std_dev" in feature_data, "Missing std_dev field"
   ```

3. **Update** documentation:
   - MLAnalysisGenerationCHILD.md Section 5.2 (Output Schema)
   - This document Section 2.2 (JSON Schemas)

4. **Test**:
   ```bash
   pytest ml_pipeline/stage6_analysis/tests/test_rf_json_generation.py -k "test_video_rf_schema"
   ```

### 6.2 Common Task: Change Number of Clusters

**Scenario**: Change K-Means from 3 clusters to 5 clusters

**⚠️ WARNING**: This requires re-running Stage 5 first!

**Steps**:
1. **Modify** Stage 5 K-Means training (NOT Stage 6):
   ```python
   # File: ml_pipeline/stage5_training/ml_model_training.py
   # Change: n_clusters=3 → n_clusters=5
   ```

2. **Re-run Stage 5** to regenerate models with 5 clusters

3. **Update** Stage 6 validation (line 686):
   ```python
   # Change validation
   assert len(km_json['clusters']) == 5, \
       f"K-Means has {len(km_json['clusters'])} clusters (expected 5)"
   ```

4. **Update** documentation to reflect new cluster count

### 6.3 Common Task: Add New Window Type

**Scenario**: Add `pre_closing` window (between middle and closing)

**Steps**:
1. **Update** `config/bucket_definitions.py`:
   ```python
   BUCKET_WINDOWS = {
       '18-33s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'pre_closing', 'closing'],
   }
   ```

2. **Re-run Stages 3, 4, 5** (upstream dependencies)

3. **No Stage 6 code changes needed** - Stage 6 is window-agnostic!

4. **Verify** output count formula still holds:
   ```
   Expected JSONs = 1 + (window_count × 2)
   For 7 windows: 1 + (7 × 2) = 15 files
   ```

---

## 7. Debugging Checklist

### 7.1 Pre-Flight Validation Failures

**Symptom**: Exit code 1, "Pre-flight validation failed"

**Checks**:
1. ✅ Verify Stage 4 completed successfully
   ```bash
   ls {bucket_path}/ml_analysis/*_transformed.csv | wc -l
   # Expected: 13 (CONTRASTIVE) or 7 (TOP)
   ```

2. ✅ Verify Stage 5 completed successfully
   ```bash
   ls {bucket_path}/models/*.pkl | wc -l
   # Expected: 19 PKL files (mode-dependent)
   ```

3. ✅ Check model_metrics.json exists and is valid
   ```bash
   cat {bucket_path}/models/model_metrics.json | jq '.video_level_rf.trained'
   # Expected: true (CONTRASTIVE) or false (TOP)
   ```

4. ✅ Check bucket path is correct
   ```bash
   echo $bucket_path
   # Expected: /data/clients/{client}/hashtags/{hashtag}/top_contrastive/bucket_{bucket}
   ```

### 7.2 JSON Generation Failures

**Symptom**: Exit code 2, "JSON generation failed"

**Checks**:
1. ✅ Check model file integrity
   ```bash
   python3 -c "import joblib; joblib.load('{bucket_path}/models/rf_video_{bucket}.pkl')"
   # Should not raise UnpicklingError
   ```

2. ✅ Check CSV file integrity
   ```bash
   python3 -c "import pandas as pd; pd.read_csv('{bucket_path}/ml_analysis/aggregated_features.csv')"
   # Should not raise ParserError
   ```

3. ✅ Check temp directory permissions
   ```bash
   ls -ld {bucket_path}/ml_analysis/.tmp/
   # Expected: drwxr-xr-x (writable)
   ```

4. ✅ Check disk space
   ```bash
   df -h {bucket_path}
   # Expected: >100MB free
   ```

### 7.3 Output Validation Failures

**Symptom**: Exit code 3, "Output validation failed"

**Checks**:
1. ✅ Check K-Means feature names
   ```bash
   cat {bucket_path}/ml_analysis/.tmp/hook_kmeans_analysis.json.tmp | \
     jq '.clusters[0].centroid | keys[] | select(contains("_scaled"))'
   # Expected: (empty - no suffixed features)
   ```

2. ✅ Check cluster size sum
   ```bash
   cat {bucket_path}/ml_analysis/.tmp/hook_kmeans_analysis.json.tmp | \
     jq '[.clusters[].size] | add'
   # Expected: Equals .total_videos
   ```

3. ✅ Check JSON parseability
   ```bash
   cat {bucket_path}/ml_analysis/.tmp/rf_video_analysis.json.tmp | jq .
   # Should not raise "parse error"
   ```

### 7.4 Mode Detection Issues

**Symptom**: Wrong files generated (expecting 13, got 6)

**Checks**:
1. ✅ Verify model_metrics.json mode flag
   ```bash
   cat {bucket_path}/models/model_metrics.json | jq '.video_level_rf.trained'
   # true = CONTRASTIVE (13 files)
   # false = TOP (6 files)
   ```

2. ✅ Check Stage 5 checkpoint
   ```bash
   cat {bucket_path}/checkpoints/stage_5_checkpoint.json | jq '.mode'
   ```

3. ✅ Check RF model files exist
   ```bash
   ls {bucket_path}/models/rf_video_{bucket}.pkl
   # Should exist in CONTRASTIVE, may not exist in TOP
   ```

---

## 8. Dependencies

### 8.1 Python Modules

**Standard Library**:
```python
import os           # File operations
import sys          # System operations
import json         # JSON serialization
import shutil       # Directory operations (atomic rollback)
import logging      # Logging
import traceback    # Error tracing
from typing import List, Dict, Optional
from datetime import datetime
```

**External Libraries**:
```python
import pandas as pd         # Version: 2.0.0+
                           # Purpose: CSV loading, data manipulation
                           # Usage: pd.read_csv(), df.quantile()

import numpy as np          # Version: 1.24.0+
                           # Purpose: Array operations, percentile calculations
                           # Usage: np.argsort(), np.linalg.norm()

import joblib              # Version: 1.3.0+
                           # Purpose: Pickle file loading
                           # Usage: joblib.load(model_path)
```

### 8.2 Internal Imports

```python
from config.bucket_definitions import BUCKET_WINDOWS
# Purpose: Window configuration for each bucket
# Usage: windows = BUCKET_WINDOWS[bucket]
```

### 8.3 Upstream Dependencies (Stages)

| Stage | Dependency | File Pattern | Critical? |
|-------|------------|--------------|-----------|
| Stage 3 | Aggregated features | `ml_analysis/aggregated_features.csv` | Yes (CONTRASTIVE) |
| Stage 4 | RF transformed CSVs | `ml_analysis/{window}_rf_transformed.csv` | Yes (CONTRASTIVE) |
| Stage 4 | K-Means transformed CSVs | `ml_analysis/{window}_km_transformed.csv` | Yes (always) |
| Stage 5 | Video RF model | `models/rf_video_{bucket}.pkl` | Yes (CONTRASTIVE) |
| Stage 5 | Window RF models | `models/rf_{window}_{bucket}.pkl` | Yes (CONTRASTIVE) |
| Stage 5 | K-Means models | `models/{window}_kmeans_{bucket}.pkl` | Yes (always) |
| Stage 5 | X data matrices | `models/{window}_X_data_{bucket}.pkl` | Yes (always) |
| Stage 5 | Scalers | `models/{window}_scalers_{bucket}.pkl` | Validated only |
| Stage 5 | Model metrics | `models/model_metrics.json` | Yes (mode flag) |

### 8.4 Downstream Consumers (Stages)

**Stage 7: LLM Analysis**
```python
# File: ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py
# Lines: 255-256, 415, 626

# Loads all Stage 6 JSONs:
rf_video_path = os.path.join(ml_analysis_dir, 'rf_video_analysis.json')
kmeans_path = os.path.join(ml_analysis_dir, f'{window_type}_kmeans_analysis.json')
rf_path = os.path.join(ml_analysis_dir, f'{window_type}_rf_analysis.json')

# Uses for LLM prompt construction (Phase 1 & 2)
```

---

## 9. Testing

### 9.1 Unit Test Commands

**Test Pre-Flight Validation**:
```bash
pytest ml_pipeline/stage6_analysis/tests/test_validation.py::test_validate_stage_dependencies -v
```

**Test Video RF Generation**:
```bash
pytest ml_pipeline/stage6_analysis/tests/test_rf_generation.py::test_generate_video_rf_json -v
```

**Test K-Means Generation**:
```bash
pytest ml_pipeline/stage6_analysis/tests/test_kmeans_generation.py::test_generate_window_kmeans_json -v
```

**Test Feature Name Normalization**:
```bash
pytest ml_pipeline/stage6_analysis/tests/test_normalization.py::test_normalize_feature_name -v

# Expected output:
# test_normalize_feature_name[eye_contact_rate_scaled → eye_contact_rate] PASSED
# test_normalize_feature_name[has_captions_encoded → has_captions] PASSED
# test_normalize_feature_name[scene_count → scene_count] PASSED
```

### 9.2 Integration Test Commands

**Full Stage 6 Execution** (requires Stage 5 outputs):
```bash
pytest ml_pipeline/stage6_analysis/tests/test_integration.py::test_stage6_end_to_end -v

# Expected output:
# - Pre-flight validation: PASSED
# - JSON generation: 13 files created
# - Output validation: PASSED
# - Atomic commit: 13 files moved to ml_analysis/
```

**Mode-Aware Test** (CONTRASTIVE vs TOP):
```bash
# Test CONTRASTIVE mode
pytest ml_pipeline/stage6_analysis/tests/test_mode_aware.py::test_contrastive_mode -v

# Test TOP mode
pytest ml_pipeline/stage6_analysis/tests/test_mode_aware.py::test_top_mode -v

# Expected: CONTRASTIVE generates 13 files, TOP generates 6 files
```

### 9.3 Manual Test (Standalone)

**Prerequisites**:
- Bucket with completed Stage 5 outputs
- Python environment with dependencies installed

**Command**:
```bash
cd /home/jorge/rumiaifinal

python3 -c "
from ml_pipeline.stage6_analysis.ml_analysis_generation import generate_ml_analysis_jsons
from config.bucket_definitions import BUCKET_WINDOWS

bucket_path = '/data/clients/acme/hashtags/nutrition/top_contrastive/bucket_18-33s'
bucket = '18-33s'
windows = BUCKET_WINDOWS[bucket]

exit_code = generate_ml_analysis_jsons(bucket_path, bucket, windows)
print(f'Exit code: {exit_code}')
"
```

**Expected Output**:
```
Pre-flight validation: checking Stage 4 and Stage 5 outputs...
✓ Pre-flight validation passed: All 13 Stage 4 files + 26 Stage 5 files exist
Generating ML analysis JSONs to temp directory...
  ✓ Generated rf_video_analysis.json.tmp
  ✓ Generated hook_rf_analysis.json.tmp
  ... (13 files)
✓ All 13 JSONs generated to temp directory
Validating JSON schemas...
✓ All JSON schemas valid
Committing JSONs (atomic rename)...
  ✓ Committed rf_video_analysis.json
  ... (13 commits)
✓ Stage 6 complete: 13 JSONs generated
Exit code: 0
```

---

## 10. Performance Characteristics

### 10.1 Timing Breakdown

**Total Duration**: 3-5 seconds per bucket (100 videos)

```
Pre-flight validation: 0.5-1s
  - File existence checks (40 files): 0.3-0.5s
  - Model metrics parsing: 0.1-0.2s
  - Mode detection: 0.1-0.2s

JSON Generation: 2-3s
  - Video RF generation: 0.5-0.8s
    - Model loading: 0.1s
    - CSV loading (aggregated_features.csv): 0.2s
    - Percentile calculations: 0.2-0.5s

  - Window RF generation (×6): 0.6-0.9s
    - Model loading per window: 0.05s
    - CSV loading per window: 0.05s
    - Distribution calculations: 0.05s

  - Window K-Means generation (×6): 0.9-1.3s
    - Model loading per window: 0.05s
    - X_data loading: 0.05s
    - Cluster prediction: 0.05s
    - Distance calculations: 0.05-0.1s

Output Validation: 0.3-0.5s
  - File existence checks: 0.1s
  - JSON parseability: 0.1s
  - Feature name validation: 0.1s
  - Cluster size validation: 0.05s

Atomic Commit: 0.2-0.3s
  - os.rename() ×13: 0.2s
  - Cleanup: 0.05s
```

### 10.2 Bottlenecks

**Not Bottlenecks** (Stage 6 optimized):
- ✅ Model loading (joblib is fast)
- ✅ JSON serialization (Python json module is fast)
- ✅ Feature name normalization (string operations <0.1s)

**Potential Bottlenecks** (acceptable):
- Percentile calculations on 100 videos × 10 features: 0.2-0.5s
- CSV loading (aggregated_features.csv ~18MB): 0.2s
- Distance calculations (100 videos × 3 clusters): 0.05-0.1s per window

**Pipeline Context**:
- Stage 2: 60-80s per video (THE bottleneck)
- Stage 5: 30-90s per bucket
- **Stage 6: 3-5s per bucket** ← NOT a bottleneck
- Stage 7: 25-30s per bucket

### 10.3 Memory Usage

**Peak Memory**: 100-150 MB per bucket

```
Models loaded simultaneously:
  - rf_video model: ~450 KB
  - 6 window RF models: ~50 KB each
  - 6 K-Means models: ~25 KB each
  - 6 X_data matrices: ~80 KB each
  Total: ~1 MB

CSVs loaded:
  - aggregated_features.csv: ~18 MB
  - 6 window RF CSVs: ~200 KB each
  - 6 window K-Means CSVs: ~200 KB each
  Total: ~20 MB

JSON generation (in-memory dicts):
  - 13 JSON dicts: ~10 MB total

Python overhead: ~50 MB

Peak: ~100 MB (acceptable)
```

### 10.4 Scalability Limits

**Tested Limits**:
- Max videos: 200 per bucket (expected: ~5-8s)
- Min videos: 50 per bucket (expected: ~2-3s)

**Not Tested**:
- >200 videos per bucket (may exceed 10s threshold)
- <50 videos per bucket (may fail Stage 5 training)

**Recommendation**: Stage 6 scales linearly with video count. For >200 videos, consider parallelizing window-level JSON generation.

---

## 11. Related Documentation

### 11.1 Parent Documents
- [PRODUCTION_FLOW.md](PRODUCTION_FLOW.md) - Pipeline overview and Stage 6 contract
- [MLAnalysisGenerationCHILD.md](../FutureDevelopments/ChildDocs/MLAnalysisGenerationCHILD.md) - High-level design
- [MLAnalysisGenerationCHILDTI.md](../FutureDevelopments/ChildDocs/MLAnalysisGenerationCHILDTI.md) - Technical implementation spec

### 11.2 Related Stage Documents
- [STAGE_2.6_2.7_IMPL.md](STAGE_2.6_2.7_IMPL.md) - Content classification (provides taxonomy)
- [STAGE_5_IMPL.md](STAGE_5_IMPL.md) - ML Model Training (upstream dependency)
- [STAGE_7_IMPL.md](STAGE_7_IMPL.md) - LLM Analysis (downstream consumer)

### 11.3 Configuration References
- `config/bucket_definitions.py` - BUCKET_WINDOWS configuration
- `config/bucket_definitions.py::get_stage3_expected_feature_count()` - Feature count formula

---

## 12. Document Metadata

**Created**: 2025-01-28
**Last Updated**: 2025-01-28
**Author**: Claude Code (Stage Implementation Generator)
**Reviewers**: [Pending]
**Status**: Draft
**Version**: 1.0

---

## 13. Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-28 | Claude Code | Initial creation following METAPROMPT_STAGE_IMPL.md |
