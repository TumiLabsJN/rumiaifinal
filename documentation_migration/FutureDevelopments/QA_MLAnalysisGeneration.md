# Clarification Q&A: ML Analysis Generation (Stage 6)

> **Mother Doc**: MLPlanningv2.md Section Stage 6: ML Analysis Generation (Lines 1993-2372)
> **Phase 1**: Critique_MLAnalysisGeneration.md
> **Date**: 2025-10-15
> **Status**: IN PROGRESS

## Questions by Category

### Input/Output Contracts

#### Q1: [CRITICAL] What are the exact columns in aggregated_features.csv input?

**Answer**:
The aggregated_features.csv file is produced by Stage 3 (Feature Aggregation) and has **bucket-specific column counts**:

**Column Structure**:
- 3 metadata columns: `video_id`, `create_time`, `gender`
- 21 base features per temporal window:
  - `average_face_size` (float, 0-1)
  - `overlay_unique_count` (int, count)
  - `has_captions` (bool)
  - `scene_count` (int, 0-20)
  - `shortest_scene` (float, seconds)
  - `longest_scene` (float, seconds)
  - `scene_duration_variance` (float)
  - `object_count` (int, count)
  - `person_count` (int, count)
  - `dominant_emotion_id` (categorical, 1-7)
  - `speech_coverage` (float, 0-1)
  - `word_count` (int, 0-200)
  - `energy_level` (float, 0-1)
  - `energy_variance` (float)
  - `energy_max` (float, 0-1)
  - `pitch_scatter_ratio` (float, 0-1)
  - `gesture_count` (int, count)
  - `gaze_variance` (float)
  - `eye_contact_rate` (float, 0-1)
  - `emotional_valence` (float, -1 to 1)
  - `emotion_consistency` (float, 0-1)

**Total Columns by Bucket**:
- 0-3s: 24 columns (1 window × 21 features + 3 metadata)
- 3-9s: 45 columns (2 windows × 21 features + 3 metadata)
- 9-13s, 13-18s: 66 columns (3 windows × 21 features + 3 metadata)
  - Windows: hook_*, middle_aggregate_*, closing_*
  - Note: Middle segments aggregated into single window for reliability
- 18-33s: 129 columns (6 windows × 21 features + 3 metadata)
  - Windows: hook_*, middle_1_*, middle_2_*, middle_3_*, middle_4_*, closing_*
- 33-60s, 60-90s, 90-120s: 150 columns (7 windows × 21 features + 3 metadata)
  - Windows: hook_*, middle_1_* through middle_5_*, closing_*

**Column Naming Convention**:
```
video_id, create_time, gender,
hook_scene_count, hook_word_count, hook_energy_level, ...,
middle_1_scene_count, middle_1_word_count, ...  # For buckets 18-33s+
middle_aggregate_scene_count, ...                # For buckets 9-13s, 13-18s
closing_scene_count, closing_word_count, ...     # For buckets 3-9s+
```

**For HLD Section**: 5.1 (Input Schema)

**Source**: FeatureAggregationCHILD.md Section 4.2 (lines 527-604), Section 5.2 (lines 655-693)

### Dependencies & Integration

#### Q2: [CRITICAL] What are the exact file paths and naming conventions for Stage 5 model files?

**Answer**:
Stage 5 produces **90 models total** across 8 duration buckets:
- 8 Video-Level RF models (1 per bucket)
- 41 Window-Level RF models (distributed by bucket window count)
- 41 Window-Level K-Means models (same distribution)

**File Naming Convention** (from bucket directory root):
```
models/
├── rf_video_{bucket}.pkl              # Video-level RF
├── rf_{window}_{bucket}.pkl           # Window-level RF
├── {window}_kmeans_{bucket}.pkl       # K-Means model
├── {window}_scalers_{bucket}.pkl      # Scalers for K-Means inference
├── {window}_X_data_{bucket}.pkl       # Feature matrix (for silhouette scores)
└── model_metrics.json                 # Performance summary
```

**Example: Bucket 18-33s** (6 windows → 20 files total):
```
bucket_18-33s/models/
├── rf_video_18-33s.pkl                    # 1 video-level RF
├── rf_hook_18-33s.pkl                     # 6 window-level RF
├── rf_middle_1_18-33s.pkl
├── rf_middle_2_18-33s.pkl
├── rf_middle_3_18-33s.pkl
├── rf_middle_4_18-33s.pkl
├── rf_closing_18-33s.pkl
├── hook_kmeans_18-33s.pkl                 # 6 K-Means models
├── middle_1_kmeans_18-33s.pkl
├── middle_2_kmeans_18-33s.pkl
├── middle_3_kmeans_18-33s.pkl
├── middle_4_kmeans_18-33s.pkl
├── closing_kmeans_18-33s.pkl
├── hook_scalers_18-33s.pkl                # 6 scaler files
├── middle_1_scalers_18-33s.pkl
├── middle_2_scalers_18-33s.pkl
├── middle_3_scalers_18-33s.pkl
├── middle_4_scalers_18-33s.pkl
├── closing_scalers_18-33s.pkl
├── hook_X_data_18-33s.pkl                 # 6 X matrix files (for silhouette)
├── middle_1_X_data_18-33s.pkl
├── middle_2_X_data_18-33s.pkl
├── middle_3_X_data_18-33s.pkl
├── middle_4_X_data_18-33s.pkl
├── closing_X_data_18-33s.pkl
└── model_metrics.json                     # 1 metrics summary
```

**File Count by Bucket** (varies by window structure):
- 0-3s: 4 files (1 RF video + 1 RF hook + 1 KM hook + 1 scaler + 1 X + 1 metrics = 6 total)
- 3-9s: 9 files (1 RF video + 2 RF windows + 2 KM + 2 scalers + 2 X + 1 metrics)
- 9-13s, 13-18s: 14 files (1 RF video + 3 RF windows + 3 KM + 3 scalers + 3 X + 1 metrics)
- 18-33s: 20 files (1 RF video + 6 RF windows + 6 KM + 6 scalers + 6 X + 1 metrics)
- 33-60s, 60-90s, 90-120s: 23 files (1 RF video + 7 RF windows + 7 KM + 7 scalers + 7 X + 1 metrics)

**model_metrics.json**: Contains performance summaries (accuracy, top features, silhouette scores) but NOT full feature importance rankings or cluster centroids (those are extracted from .pkl files by Stage 6).

**For HLD Section**: 3.1 (Input Dependencies), 5.1 (Input Schema), 6.1 (Input Validation - pre-flight checks)

**Source**: Stage5_MLModelTraining_HLD.md Section 4.2 (lines 840-857), Section 2.3.3 (lines 230-274), Section 5.2 (lines 937-997)

#### Q3: [CRITICAL] What are the exact file paths for Stage 4 transformed feature files?

**Answer**:
Stage 4 produces **13 transformed CSV files per bucket** (3 types × variable window counts):

**File Paths** (all relative to bucket directory):
```
bucket_{duration}/ml_analysis/
├── rf_transformed.csv                    # Video-Level RF (1 file)
├── hook_rf_transformed.csv               # Window-Level RF (6-7 files)
├── middle_1_rf_transformed.csv
├── middle_2_rf_transformed.csv
├── middle_3_rf_transformed.csv
├── middle_4_rf_transformed.csv
├── closing_rf_transformed.csv
├── hook_km_transformed.csv               # Window-Level K-Means (6-7 files)
├── middle_1_km_transformed.csv
├── middle_2_km_transformed.csv
├── middle_3_km_transformed.csv
├── middle_4_km_transformed.csv
└── closing_km_transformed.csv
```

**File Count by Bucket** (varies by window structure):
- 0-3s: 3 files (1 RF video + 1 RF hook + 1 KM hook)
- 3-9s: 5 files (1 RF video + 2 RF windows + 2 KM)
- 9-13s, 13-18s: 7 files (1 RF video + 3 RF windows + 3 KM) - uses middle_aggregate
- 18-33s: 13 files (1 RF video + 6 RF windows + 6 KM)
- 33-60s, 60-90s, 90-120s: 15 files (1 RF video + 7 RF windows + 7 KM)

**Usage in Stage 6**:
- **Section 6.1** (line 2007): Loads `ml_analysis/aggregated_features.csv` for distribution analysis (NOT transformed CSVs)
- **Section 6.2** (line 2164): Loads `ml_analysis/{window}_rf_transformed.csv` per window for window-level RF analysis
- **Section 6.3** (line 2234-2236): Loads `ml_analysis/{window}_km_transformed.csv` per window for K-Means cluster assignments

**Pre-Flight Validation**: Stage 6 should validate all required Stage 4 files exist before generating ANY JSONs (similar to Stage 5's validation pattern).

**For HLD Section**: 3.1 (Input Dependencies), 5.1 (Input Schema), 6.1 (Input Validation - pre-flight checks)

**Source**: FeatureTransformationCHILD.md Section 3.2 (lines 466-482), Section 5.2 (lines 670-796)

#### Q4: [CRITICAL] What are the exact JSON schemas for all three output JSON types?

**Answer**:
Stage 6 generates **3 types of JSON files** with the following complete schemas:

### 1. Video-Level RF JSON Schema

**File**: `ml_analysis/rf_video_analysis.json` (~30KB)

**Complete Structure**:
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
    }
    // ... top 10 features (each with distribution data)
  ],

  "videos": [
    {
      "video_id": "123",
      "is_top_performer": 1,
      "prediction_confidence": 0.92,
      "features": {
        "hook_scene_count": 3,
        "middle_avg_word_count": 55
        // ... all features
      }
    }
    // ... all N videos
  ]
}
```

**Field Details**:
- `analysis_type`: string (always "random_forest")
- `bucket`: string (e.g., "18-33s")
- `hashtag`: string (e.g., "#nutrition")
- `video_count`: int (total videos, e.g., 100)
- `input_features`: int (feature count, ~178 for bucket 18-33s)
- `feature_importance`: array of objects (top 10 features)
  - `feature`: string (feature name)
  - `importance`: float (RF importance score, 0-1)
  - `top_performer_avg`: float (mean value in top 80%)
  - `bottom_performer_avg`: float (mean value in bottom 20%)
  - `gap`: float (difference between top and bottom)
  - `distribution`: object (NEW - added per Critique Q3)
    - `thresholds`: object with `high` (66th percentile) and `low` (33rd percentile)
    - `top_performers`: object with `high_percentage`, `medium_percentage`, `low_percentage`
    - `bottom_performers`: object with same structure
- `videos`: array of objects (all N videos)
  - `video_id`: string
  - `is_top_performer`: int (0 or 1)
  - `prediction_confidence`: float (0-1)
  - `features`: object (all feature values)

### 2. Window-Level RF JSON Schema

**Files**: `ml_analysis/{window}_rf_analysis.json` (~5KB each, 6-7 files per bucket)

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
- `model_type`: string (always "window_level_rf")
- `window_type`: string (e.g., "hook", "middle_1", "closing", "middle_aggregate")
- `bucket`: string
- `total_videos`: int
- `input_features`: int (always 21 for window-level)
- `model_performance`: object
  - `accuracy`: float (0-1)
  - `precision`: float (0-1)
  - `recall`: float (0-1)
- `feature_importance`: array of objects (top 10)
  - `feature`: string (no window prefix - e.g., "eye_contact_rate" not "hook_eye_contact_rate")
  - `importance`: float (0-1)
  - `top_performer_avg`: float
  - `bottom_performer_avg`: float
  - `gap`: float
  - `rank`: int (1-10)

### 3. Window-Level K-Means JSON Schema

**Files**: `ml_analysis/{window}_kmeans_analysis.json` (~5KB each, 6-7 files per bucket)

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
        "eye_contact_rate_scaled": 0.87,
        "scene_count_scaled": 0.45,
        "word_count_scaled": 0.62,
        "energy_level_scaled": 0.73
        // ... all 21-39 features (scaled)
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
- `window_type`: string
- `bucket`: string
- `total_videos`: int
- `n_clusters`: int (always 3)
- `clusters`: array of 3 objects
  - `cluster_id`: int (0, 1, or 2)
  - `size`: int (videos in cluster)
  - `centroid`: object (ALL 21-39 features as key-value pairs)
    - Feature names have `_scaled` suffix (e.g., "eye_contact_rate_scaled")
    - All values are float (0-1 range due to scaling)
  - `videos`: array of objects
    - `video_id`: string
    - `distance_to_centroid`: float (Euclidean distance)

**Required vs Optional Fields**:
- **All fields are required** except:
  - Video-Level RF: `videos` array can be omitted for space (only `feature_importance` needed for LLM)
  - K-Means: `videos` array can be summarized (only `size` needed, full video list optional)

**For HLD Section**: 5.2 (Output Schema), 6.3 (Output Validation)

**Source**: MLPlanningv2.md lines 2018-2069 (Video RF), 2179-2201 (Window RF), 2252-2275 (Window K-Means)

#### Q5: [CRITICAL] What is the error handling strategy for Stage 6 JSON generation?

**Answer**:
Stage 6 uses **Atomic Output Pattern** (Option B) - all 13 JSONs succeed or all deleted on failure.

### Chosen Approach: Atomic Pattern

**Rationale**:
1. **Consistency with Stage 5**: Stage 5 uses atomic bucket training (all models succeed OR all deleted). Stage 6 follows same pattern.
2. **Clean failure state**: Partial output (3 of 13 JSONs) creates ambiguity for Stage 7. With atomic pattern: 0 files = clear failure signal.
3. **Pre-flight validation prevents most failures**: Validate all Stage 4 CSVs + Stage 5 PKL files exist before generating ANY JSONs.
4. **Simple recovery**: Failure = 0 JSONs = user re-runs Stage 6 cleanly. No partial state cleanup.
5. **Low overhead**: 13 JSONs × ~5KB = 65KB total. Temp file overhead negligible. Stage 6 is fast (~5-10s per bucket), so regenerating all JSONs on retry is acceptable.

### Implementation Pattern

```python
def generate_ml_analysis_jsons(bucket_path, bucket, windows):
    """
    Generate all ML analysis JSONs using atomic pattern.
    Either all 13 JSONs succeed or all deleted.
    """
    temp_dir = f"{bucket_path}/ml_analysis/.tmp/"
    os.makedirs(temp_dir, exist_ok=True)
    generated_files = []

    try:
        # ===== 1. PRE-FLIGHT VALIDATION (fail before generating anything) =====
        logger.info("Pre-flight validation: checking Stage 4 and Stage 5 outputs...")

        # Validate Stage 4 CSVs exist
        required_stage4_files = [
            'ml_analysis/aggregated_features.csv',
            'ml_analysis/rf_transformed.csv',
            *[f'ml_analysis/{w}_rf_transformed.csv' for w in windows],
            *[f'ml_analysis/{w}_km_transformed.csv' for w in windows]
        ]

        for file_path in required_stage4_files:
            full_path = os.path.join(bucket_path, file_path)
            if not os.path.exists(full_path):
                raise PreFlightValidationError(
                    f"Stage 4 incomplete: Missing {file_path}. Run Stage 4 first."
                )

        # Validate Stage 5 PKL models exist
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
                raise PreFlightValidationError(
                    f"Stage 5 incomplete: Missing {file_path}. Run Stage 5 first."
                )

        logger.info("✓ Pre-flight validation passed: All Stage 4 and Stage 5 files exist")

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

    except Exception as e:
        logger.error(f"""
Stage 6 JSON generation failed
Exception: {type(e).__name__}: {str(e)}
Stack trace: {traceback.format_exc(limit=10)}
Generated files before failure: {len(generated_files)} of 13 expected
""")
        # Delete all temp files on failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning("Rolled back: Deleted all temp files (atomic failure)")
        return 2  # JSON GENERATION FAILED
```

### Pre-Flight Validation Checklist

**Stage 4 Files (13 files for bucket 18-33s)**:
- ✅ `ml_analysis/aggregated_features.csv`
- ✅ `ml_analysis/rf_transformed.csv`
- ✅ `ml_analysis/hook_rf_transformed.csv` (× 6 windows)
- ✅ `ml_analysis/hook_km_transformed.csv` (× 6 windows)

**Stage 5 Files (20 files for bucket 18-33s)**:
- ✅ `models/rf_video_18-33s.pkl`
- ✅ `models/rf_hook_18-33s.pkl` (× 6 windows)
- ✅ `models/hook_kmeans_18-33s.pkl` (× 6 windows)
- ✅ `models/hook_scalers_18-33s.pkl` (× 6 windows)
- ✅ `models/hook_X_data_18-33s.pkl` (× 6 windows)
- ✅ `models/model_metrics.json`

### Exit Codes

| Code | Meaning | User Action |
|------|---------|-------------|
| **0** | Success (all 13 JSONs generated) | Continue to Stage 7 |
| **1** | Pre-flight validation failed (missing Stage 4/5 files) | Re-run Stage 4 or Stage 5 |
| **2** | JSON generation failed (mid-process crash) | Check logs, fix issue, re-run Stage 6 |
| **3** | Output validation failed (malformed JSON schema) | Bug in generation code, report issue |
| **4** | Disk I/O failed (permissions, disk full) | Fix disk/permissions issue, re-run Stage 6 |

### Edge Case: Mid-Failure Scenario

**Scenario**: Bucket 18-33s needs 13 JSONs. Stage 6 successfully creates:
- ✅ `rf_video_analysis.json.tmp`
- ✅ `hook_rf_analysis.json.tmp`
- ✅ `hook_kmeans_analysis.json.tmp`
- ❌ Then crashes generating `middle_1_rf_analysis.json`

**What Happens**:
1. Exception caught in `except` block
2. `shutil.rmtree(temp_dir)` deletes all 3 temp files
3. Stage 6 exits with code 2 (JSON generation failed)
4. User sees: **0 JSONs in ml_analysis/** (clean state)
5. User re-runs Stage 6 → generates all 13 JSONs from scratch

**Result**: No partial output, no ambiguity, clean retry.

### Trade-offs

**Pros**:
- Clean failure state (0 files = clear signal)
- Consistent with Stage 5 atomic pattern
- Simple recovery (just re-run Stage 6)
- No "which files are valid?" confusion

**Cons**:
- Must regenerate all 13 JSONs on retry (no partial progress saved)
- **Acceptable**: Stage 6 is fast (~5-10s per bucket), so regenerating is not a bottleneck

**For HLD Section**: 6.2 (Error Cases), 6.1 (Input Validation - pre-flight), 2.3 (Detailed Process)

**Source**: Design decision based on Stage5_MLModelTraining_HLD.md atomic pattern (lines 208-310), Critique_MLAnalysisGeneration.md lines 224-252

### Edge Cases & Validation

#### Q6: [CRITICAL] What is the feature name normalization logic for K-Means features?

**Answer**:
Stage 6 **MUST normalize K-Means feature names** when generating K-Means JSON outputs (Section 6.3) to maintain consistency with RF feature naming for LLM consumption.

### Problem: Feature Name Mismatch

**Root Cause**: Stage 4 creates different naming conventions for K-Means vs RF features:

- **K-Means features** (from `hook_km_transformed.csv`):
  - Have transformation suffixes: `eye_contact_rate_scaled`, `scene_count_scaled`, `has_captions_encoded`
  - Example centroid: `{"eye_contact_rate_scaled": 0.87, "scene_count_scaled": 0.45, ...}`

- **RF features** (from `hook_rf_transformed.csv`):
  - NO suffixes: `eye_contact_rate`, `scene_count`, `has_captions`
  - Example importance: `{"feature": "eye_contact_rate", "importance": 0.35}`

**Impact**: LLM in Stage 7 receives inconsistent feature names and cannot correlate insights:
```
RF says: "eye_contact_rate has 0.35 importance"
K-Means says: "Cluster 0 has high eye_contact_rate_scaled (0.87)"
LLM thinks: These are DIFFERENT features → cannot synthesize insights
```

### When Normalization Is Needed

**Section 6.3 (K-Means JSON generation)**: Normalize feature names in `centroid` object before writing JSON.

**Example transformation**:
```python
# BEFORE normalization (WRONG - inconsistent with RF)
{
  "window_type": "hook",
  "clusters": [
    {
      "cluster_id": 0,
      "centroid": {
        "eye_contact_rate_scaled": 0.87,      # ← Has _scaled suffix
        "scene_count_scaled": 0.45,           # ← Has _scaled suffix
        "has_captions_encoded": 1             # ← Has _encoded suffix
      }
    }
  ]
}

# AFTER normalization (CORRECT - consistent with RF)
{
  "window_type": "hook",
  "clusters": [
    {
      "cluster_id": 0,
      "centroid": {
        "eye_contact_rate": 0.87,             # ← Suffix removed
        "scene_count": 0.45,                  # ← Suffix removed
        "has_captions": 1                     # ← Suffix removed
      }
    }
  ]
}
```

### Normalization Function

**Reuse from Stage 5** (Stage5_MLModelTraining_HLD.md lines 358-401):

```python
def normalize_feature_name(feature_name):
    """
    Normalize K-Means feature names for comparison with RF feature names.

    Removes K-Means transformation suffixes from Stage 4:
    - '_scaled' (from MinMax scaling)
    - '_log' (from log transformation - intermediate, usually removed)
    - '_encoded' (from label encoding)

    Args:
        feature_name: str, e.g., 'eye_contact_rate_scaled'

    Returns:
        str, e.g., 'eye_contact_rate'

    Examples:
        >>> normalize_feature_name('eye_contact_rate_scaled')
        'eye_contact_rate'
        >>> normalize_feature_name('has_captions_encoded')
        'has_captions'
        >>> normalize_feature_name('scene_count')  # Already normalized
        'scene_count'
    """
    normalized = feature_name

    # Remove suffixes in order (some features may have multiple)
    suffixes = ['_scaled', '_log', '_encoded']
    for suffix in suffixes:
        normalized = normalized.replace(suffix, '')

    return normalized
```

### Implementation in Stage 6

**Section 6.3 - K-Means JSON generation**:

```python
# Load K-Means model
kmeans = joblib.load(f'models/{window}_kmeans_{bucket}.pkl')
centroids = kmeans.cluster_centers_  # Shape: (3 clusters, 39 features)

# Load feature names from Stage 4 K-Means CSV
X = pd.read_csv(f'ml_analysis/{window}_km_transformed.csv')
feature_names = X.columns.tolist()  # ['eye_contact_rate_scaled', 'scene_count_scaled', ...]

# Build K-Means analysis JSON
for cluster_id in range(3):
    centroid_values = centroids[cluster_id]

    # CRITICAL: Normalize feature names before creating centroid dict
    normalized_centroid = {
        normalize_feature_name(name): float(value)
        for name, value in zip(feature_names, centroid_values)
    }

    cluster_data = {
        "cluster_id": cluster_id,
        "size": cluster_size,
        "centroid": normalized_centroid  # ← Now consistent with RF feature names
    }
```

### Why This Matters

**For Stage 7 LLM correlation**:
```
# WITH normalization (LLM can correlate):
RF: "eye_contact_rate importance: 0.35 (top feature)"
K-Means: "Cluster 0 has high eye_contact_rate: 0.87"
LLM: "Eye contact is critical - both RF (top feature) and K-Means (high in winning cluster) agree"

# WITHOUT normalization (LLM cannot correlate):
RF: "eye_contact_rate importance: 0.35 (top feature)"
K-Means: "Cluster 0 has high eye_contact_rate_scaled: 0.87"
LLM: "Eye contact mentioned by RF, but different feature eye_contact_rate_scaled in clusters..."
```

### Edge Cases

1. **Feature already normalized** (no suffix): `normalize_feature_name('scene_count')` → `'scene_count'` (no change)
2. **Multiple suffixes** (rare): `normalize_feature_name('word_count_log_scaled')` → `'word_count'` (both removed)
3. **One-hot features** (from `dominant_emotion_id`): `'joy'`, `'sadness'` → Already normalized (no suffixes)

### Testing

**Unit Test**: Reuse Stage 5's test suite
```python
# tests/unit/test_feature_normalization.py (from Stage 5)
def test_normalize_kmeans_features():
    assert normalize_feature_name('eye_contact_rate_scaled') == 'eye_contact_rate'
    assert normalize_feature_name('has_captions_encoded') == 'has_captions'
    assert normalize_feature_name('scene_count') == 'scene_count'  # No change
```

**Integration Test**: Validate Stage 6 JSON outputs
```python
# tests/integration/test_stage6_feature_consistency.py
def test_kmeans_json_has_normalized_features():
    """Verify K-Means JSON has same feature names as RF JSON."""
    # Load outputs
    rf_json = json.load(open('ml_analysis/hook_rf_analysis.json'))
    km_json = json.load(open('ml_analysis/hook_kmeans_analysis.json'))

    # Extract feature names
    rf_features = {f['feature'] for f in rf_json['feature_importance']}
    km_features = set(km_json['clusters'][0]['centroid'].keys())

    # Assert overlap (should be high, not zero)
    overlap = rf_features & km_features
    assert len(overlap) >= 15, f"Only {len(overlap)} features overlap (expected ≥15)"
```

**For HLD Section**: 3.3 (Critical Implementation Warnings), 2.3 (Detailed Process - K-Means JSON generation)

**Source**: Stage5_MLModelTraining_HLD.md Section 3 (lines 328-401), Critique_MLAnalysisGeneration.md lines 105-110

[Questions will be filled iteratively]

### Performance & Scale

#### Q7: What are the performance targets and potential bottlenecks for Stage 6?

**Answer**:
Stage 6 should be **FAST** - significantly faster than Stage 5 (training) and Stage 7 (LLM calls).

### Performance Targets

**Expected Performance** (N=100 videos, bucket 18-33s):
- **Target**: 3-5 seconds per bucket
- **Acceptable**: 5-10 seconds per bucket (slower hardware)
- **Warning**: > 30 seconds per bucket (log warning, investigate)
- **Likely bug**: > 2 minutes per bucket (suggests I/O issues or logic error)

**Comparison with other stages**:
- **Stage 5 (ML Training)**: 30-90 seconds per bucket (Stage5_MLModelTraining_HLD.md line 1105)
- **Stage 6 (Analysis Generation)**: 3-5 seconds per bucket (MUCH faster - no training)
- **Stage 7 (LLM Reports)**: 25-30 seconds per bucket (MLPlanningv2.md line 3091)

**Why Stage 6 is fast**:
- No ML training (just loads pre-trained models)
- No external API calls (unlike Stage 7's LLM)
- Minimal computation (extract attributes, compute percentiles)
- Small output (13 JSONs × ~5KB = 65KB)

### Detailed Operation Timing Estimates

**For bucket 18-33s with N=100 videos:**

| Operation | Estimated Time | Details |
|-----------|---------------|---------|
| **1. Pre-flight validation** | 0.5-1s | Check 13 Stage 4 CSVs + 20 Stage 5 PKL files exist |
| **2. Load Stage 5 models** | 1-2s | Load 20 PKL files (~50KB each): 1 RF video + 6 RF windows + 6 K-Means + 6 scalers + 6 X matrices |
| **3. Load Stage 4 CSVs** | 0.5-1s | Load aggregated_features.csv (~100KB) for distribution analysis |
| **4. Extract RF feature importance** | <0.5s | Read `model.feature_importances_` attribute (already computed) |
| **5. Compute distribution stats** | 1-2s | Calculate 66th/33rd percentiles for top 10 features × 1 video RF + 6 window RF = 7 operations |
| **6. Extract K-Means centroids** | <0.5s | Read `model.cluster_centers_` attribute (already computed) |
| **7. Normalize K-Means features** | <0.5s | String replacements for 39 features × 6 windows |
| **8. Write 13 JSONs** | 0.5-1s | Write 95KB total (65KB payload + formatting) |
| **9. Output validation** | <0.5s | Check 13 files created, schema validation |
| **TOTAL** | **3-5 seconds** | Sum of all operations |

**At scale (N=200 videos)**:
- Distribution calculations scale linearly: ~2-3s (double the percentile computation)
- Model loading unchanged: Still 1-2s
- **Total: ~5-8 seconds** (acceptable, under 10s target)

### Memory Usage

**Peak Memory** (bucket 18-33s, N=100):
- **Stage 5 PKL models in memory**: ~50-80 MB
  - 1 Video RF model (~5-10 MB)
  - 12 Window models (RF + K-Means): ~3-5 MB each × 12 = ~36-60 MB
  - 6 X matrices: ~50 KB each × 6 = ~300 KB
- **Stage 4 CSVs loaded**: ~20-30 MB
  - aggregated_features.csv: ~15-20 MB (100 videos × 129 features)
  - Transformed CSVs (if needed): ~5-10 MB
- **JSON generation buffers**: ~5-10 MB
- **TOTAL Peak**: **100-150 MB** (much less than Stage 5's 500 MB)

**Memory by operation**:
- Baseline (Python + imports): ~30 MB
- After loading all models: ~80-110 MB
- During JSON generation: ~100-150 MB (peak)
- After cleanup: ~30 MB

### Bottleneck Analysis

**Potential Bottlenecks** (ranked by likelihood):

1. **Distribution Percentile Calculations** (1-2s)
   - **Operation**: Computing 66th/33rd percentiles for top 10 features
   - **Why slow**: NumPy quantile operations on 100 videos × 10 features × 7 windows
   - **Mitigation**: Use vectorized numpy operations (already fast), or cache if computed repeatedly
   - **Severity**: LOW (1-2s is acceptable)

2. **Disk I/O for loading 20 PKL files** (1-2s)
   - **Operation**: Loading Stage 5 models from disk
   - **Why slow**: 20 file reads × ~50KB each = ~1 MB total, but I/O latency dominates
   - **Mitigation**: Ensure models stored on SSD (not HDD), or lazy-load only needed models
   - **Severity**: LOW (1-2s is acceptable, would be 5-10s on HDD)

3. **CSV Loading for distribution analysis** (0.5-1s)
   - **Operation**: Loading aggregated_features.csv (~15-20 MB)
   - **Why slow**: pandas.read_csv with 100 rows × 129 columns
   - **Mitigation**: Load only needed columns (top 10 features), or cache CSV in memory
   - **Severity**: VERY LOW (<1s)

4. **JSON Serialization** (0.5-1s)
   - **Operation**: Writing 13 JSON files with json.dump()
   - **Why slow**: Python dict → JSON string conversion, disk writes
   - **Mitigation**: Write to temp directory first (atomic pattern already implemented)
   - **Severity**: VERY LOW (<1s)

**NOT bottlenecks**:
- ✅ Feature extraction from models (just attribute access: <0.1s)
- ✅ Feature name normalization (string replace: <0.1s)
- ✅ Validation logic (file existence checks: <0.1s)

### Performance Guidelines

**Similar to Stage 5's approach** (Stage5_MLModelTraining_HLD.md lines 1102-1115):

- **No hard timeout**: Stage 6 is NOT user-facing, and execution time varies by hardware
- **Warning threshold**: Log warning if > 30 seconds per bucket
- **Expected range**: 3-10 seconds (depending on hardware: SSD vs HDD, fast vs slow CPU)
- **Suspicious behavior**: > 2 minutes suggests bug (infinite loop, disk failure, corrupted models)

**Performance logging**:
```python
logger.info(f"Stage 6 completed in {elapsed:.1f}s (target: <5s)")
logger.info(f"  Pre-flight validation: {preflight_time:.1f}s")
logger.info(f"  Model loading: {load_time:.1f}s")
logger.info(f"  Distribution analysis: {dist_time:.1f}s")
logger.info(f"  JSON generation: {json_time:.1f}s")
logger.info(f"  Peak memory: {peak_mb:.0f}MB (target: <200MB)")

# Warning for slow execution
if elapsed > 30:
    logger.warning(
        f"Stage 6 took {elapsed:.1f}s (expected <10s). "
        f"Check disk I/O performance or model file corruption."
    )
```

### Scalability

**Pipeline Context** (from Stage 5 HLD line 1136):
- **Total pipeline**: 3.6-4.8 hours
- **Stage 5 (training)**: 1-2 minutes (~0.5-1% of total)
- **Stage 6 (analysis)**: 3-5 seconds (~0.02% of total)
- **Stage 7 (LLM reports)**: 25-30 seconds (~0.2% of total)
- **Bottleneck**: Stage 2 (video processing, 60-80s per video × 300 videos = 5-6.7 hours)

**Conclusion**: Stage 6 is NOT a bottleneck. Optimization NOT needed.

**Tested Limits**:
- **Maximum**: N=200 videos per bucket (expected: ~5-8s, acceptable)
- **Minimum**: N=50 videos per bucket (expected: ~2-3s)

**For HLD Section**: 7.1 (Performance Targets), 7.3 (Bottleneck Analysis)

**Source**: Estimated based on operation analysis, Stage5_MLModelTraining_HLD.md Section 7 (lines 1098-1140), MLPlanningv2.md Section 6.4 (lines 2336-2379)

[Questions will be filled iteratively]

### Error Handling

Covered by Q5 (Atomic Output Pattern with comprehensive error handling strategy)

### Testing

Covered by Q6 (includes unit test and integration test specifications for feature normalization)

## Completeness Check

### Questions Asked: 7 Total

**By Category:**
- ✅ **Input/Output Contracts** (3): Q1, Q3, Q4
- ✅ **Dependencies & Integration** (2): Q2, Q3
- ✅ **Error Handling** (1): Q5
- ✅ **Edge Cases & Validation** (1): Q6
- ✅ **Performance & Scale** (1): Q7

### Coverage Analysis

**✅ COMPLETE Coverage:**
1. **Input Schema**: Q1 (aggregated_features.csv), Q3 (Stage 4 CSVs)
2. **Input Dependencies**: Q2 (Stage 5 models with exact file paths), Q3 (Stage 4 files)
3. **Output Schema**: Q4 (Complete JSON schemas for all 3 types)
4. **Error Handling**: Q5 (Atomic pattern, pre-flight validation, exit codes)
5. **Critical Logic**: Q6 (Feature name normalization - CRITICAL bug prevention)
6. **Performance**: Q7 (Timing estimates, bottlenecks, memory usage)

**📋 Deferred to HLD Phase 3:**
- Testing strategy details (Q6 provides foundation, HLD will expand)
- Configuration management (straightforward - no clarification needed)
- Logging specifications (covered by Q5 error handling)
- Deployment considerations (out of scope for HLD)

### Knowledge Gaps Resolved

**Before Phase 2:**
- ❓ Unknown: Exact columns in aggregated_features.csv
- ❓ Unknown: Stage 5 model file naming conventions
- ❓ Unknown: JSON output schemas
- ❓ Unknown: Error handling strategy
- ❓ Unknown: Feature name normalization requirement
- ❓ Unknown: Performance targets

**After Phase 2:**
- ✅ **Resolved**: All critical schema information documented
- ✅ **Resolved**: All file paths and naming conventions specified
- ✅ **Resolved**: Complete JSON schemas with examples
- ✅ **Resolved**: Atomic output pattern chosen and documented
- ✅ **Resolved**: Feature normalization logic from Stage 5 reused
- ✅ **Resolved**: Performance targets estimated (3-5s per bucket)

### Remaining Ambiguities

**None identified.** All critical implementation questions have clear answers with:
- Exact schemas and file paths
- Code examples and patterns
- Decision rationale and trade-offs
- Error handling strategies
- Performance expectations

## Proceed to Phase 3

**Decision**: ✅ YES - Ready for HLD Generation (Phase 3)

**Rationale**:
1. **All critical questions answered** (7/7 complete)
2. **Sufficient implementation detail** for HLD author
3. **No blocking ambiguities** remain
4. **Clear patterns established** (atomic output, feature normalization, pre-flight validation)
5. **References documented** for all answers (source lines provided)

**Phase 3 HLD Author Instructions**:

Use this QA document to write **Stage 6: ML Analysis Generation - High-Level Design** with the following sections:

### Required HLD Sections

**1. Context & Business Goal** (Q4, Q7)
- Why Stage 6 exists: Extract ML insights from trained models for LLM consumption
- Success criteria: 13 JSONs per bucket, 3-5s execution time, <200MB memory

**2. Architecture & Design** (Q4, Q5, Q6)
- Tri-modal JSON generation (Video RF + Window RF + Window K-Means)
- Atomic output pattern (all 13 JSONs succeed or none)
- Feature name normalization for K-Means centroids

**3. Critical Implementation Warnings** (Q6 - MANDATORY)
- 🔴 **CRITICAL**: Feature name normalization (K-Means `_scaled` suffixes must be stripped)
- Include complete `normalize_feature_name()` function from Q6
- Testing requirements from Q6

**4. Dependencies & Integration** (Q1, Q2, Q3)
- **Input Dependencies**:
  - Q1: aggregated_features.csv schema (21 features × windows + 3 metadata)
  - Q2: Stage 5 models (20 files: PKL models + scalers + X matrices + metrics)
  - Q3: Stage 4 CSVs (13 files: RF + K-Means transformed features)
- **Output Contracts**: Q4 (JSON schemas for all 3 types)

**5. Data Schemas** (Q1, Q3, Q4)
- 5.1: Input Schema (Q1 - aggregated_features.csv, Q3 - Stage 4 CSVs)
- 5.2: Output Schema (Q4 - Complete JSON schemas with field details)

**6. Error Handling & Validation** (Q5)
- 6.1: Pre-flight validation (check Stage 4/5 files exist)
- 6.2: Atomic output pattern (temp dir + rename)
- 6.3: Exit codes (0-4 with user actions)
- 6.4: Edge cases (mid-failure scenario from Q5)

**7. Performance & Scalability** (Q7)
- 7.1: Performance targets (3-5s per bucket, 100-150MB memory)
- 7.2: Bottleneck analysis (distribution calculations, disk I/O)
- 7.3: Pipeline context (Stage 6 is 0.02% of total time - NOT a bottleneck)

**8. Testing Strategy**
- Reuse Q6 testing specs (unit + integration tests for normalization)
- Add JSON schema validation tests

**9. Configuration**
- Minimal config needed (bucket definitions already centralized)

**10. References & Related Docs**
- Link to this QA document
- Link to Critique_MLAnalysisGeneration.md
- Link to Stage5_MLModelTraining_HLD.md (for normalization logic)

### Key Design Decisions to Document in HLD

1. **Atomic Output Pattern** (Q5) - All 13 JSONs or none
2. **Feature Name Normalization** (Q6) - Strip `_scaled`, `_log`, `_encoded` suffixes
3. **No Hard Timeout** (Q7) - Best-effort with warning threshold (>30s)
4. **Distribution Analysis in Stage 6** (not Stage 5) - Architecturally valid per Critique Q1

**Status**: Phase 2 COMPLETE - Proceed to Phase 3 HLD Generation

**Date Completed**: 2025-10-15
