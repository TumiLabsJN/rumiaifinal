# ML Analysis Generation - Technical Implementation Document

> **TI Document**: MLAnalysisGenerationCHILDTI.md
> **Parent HLD**: MLAnalysisGenerationCHILD.md (Stage 6: ML Analysis Generation)
> **Foundation HLD**: FoundationCHILD.md (Shared across all stages)
> **Version**: 1.0
> **Last Updated**: 2025-01-28
> **Status**: Draft

---

## Section 1: Document Metadata

**Feature Name**: ML Analysis Generation

**Parent HLD**: MLAnalysisGenerationCHILD.md

**Foundation HLD**: FoundationCHILD.md

**Covers HLD Sections**:

**From MLAnalysisGenerationCHILD.md**:
- Section 1: Context & Business Goal
- Section 1.1: What Problem Does This Solve?
- Section 1.2: Where This Fits in Pipeline
- Section 1.3: Success Criteria
- Section 2: Architecture & Design
- Section 2.1: High-Level Approach
- Section 2.2: Data Flow
- Section 2.3: Detailed Process
- Section 2.3.1: Pre-Flight Validation
- Section 2.3.2: Generate Video-Level RF JSON
- Section 2.3.3: Generate Window-Level RF JSONs
- Section 2.3.4: Generate Window-Level K-Means JSONs
- Section 2.3.5: Atomic Output Commit
- Section 3: Dependencies & Integration
- Section 3.1: Input Dependencies
- Section 3.2: Output Contracts
- Section 3.3: Cross-Stage Dependencies
- Section 3.4: External Dependencies
- Section 4: Configuration & Parameters
- Section 4.1: CLI Parameters
- Section 4.2: Internal Configuration
- Section 5: Data Schemas
- Section 5.1: Input Schema
- Section 5.2: Output Schema
- Section 6: Error Handling & Validation
- Section 6.1: Input Validation
- Section 6.2: Error Cases
- Section 6.3: Output Validation
- Section 7: Performance & Scalability
- Section 7.1: Performance Targets
- Section 7.2: Measured Performance
- Section 7.3: Bottlenecks & Mitigations
- Section 7.4: Scalability Limits
- Section 8: Testing Strategy
- Appendix A: Decision Log
- Appendix B: Example Data
- Appendix C: Pseudocode

**From FoundationCHILD.md**:
- Section 2: Client Architecture & Storage
- Section 2.1: Directory Structure
- Section 2.2: Path Templates
- Section 4: CLI Command Structure
- Section 4.1: CLI Parameters
- Section 6: Bucket Definitions
- Section 7: Standardized Exit Codes (All Stages)

**Related TI Documents**:

**Depends On**:
- FoundationTI.md (REQUIRED - provides CLI parsing, directory creation, config management)
- FeatureAggregationTI.md (Stage 3) - Produces aggregated_features.csv for distribution analysis
- FeatureTransformationTI.md (Stage 4) - Produces 13 transformed CSVs per bucket
- MLModelTrainingTI.md (Stage 5) - Produces 90 trained models (PKL files + metrics)

**Feeds Into**:
- LLMReportGenerationTI.md (Stage 7) - Consumes all JSON analysis files (count varies by bucket: 3-15 JSONs depending on window count) for LLM creative report generation

**Implementation Priority**: HIGH

**Rationale**: Stage 6 is a critical bridge between ML training (Stage 5) and LLM report generation (Stage 7). Without Stage 6, Stage 7 cannot run because it requires structured JSON insights rather than raw pickle files. This stage extracts model insights, computes distribution statistics, and formats data for LLM consumption, preventing hallucination risk and reducing execution time in Stage 7.

---

## Section 2: Stage Contract

### 2.1 Input Contract

```python
# Sources: FoundationCHILD.md Sections 2, 4 | MLAnalysisGenerationCHILD.md Sections 3.1, 5.1

class Stage6Input:
    """
    Exact structure Stage 6 receives.

    Sources:
    - CLI parameters: FoundationCHILD.md Section 4.1
    - Directory paths: FoundationCHILD.md Section 2.2
    - Stage-specific inputs: MLAnalysisGenerationCHILD.md Section 3.1
    """

    # ===== CLI PARAMETERS (from FoundationCHILD.md Section 4.1) =====
    client_id: str                  # Required, CLI parameter --client, alphanumeric + underscore
                                    # Example: "acme_corp"
                                    # Validation: Regex ^[a-zA-Z0-9_]+$ (min 1 char)

    bucket: str                     # Required, CLI parameter --bucket
                                    # Valid values: "0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"
                                    # Example: "18-33s"

    # ===== DIRECTORY PATHS (from FoundationCHILD.md Section 2.2) =====
    client_base: str                # Base client directory
                                    # Template: "/data/clients/{client_id}/"
                                    # Example: "/data/clients/acme_corp/"

    bucket_base: str                # Bucket directory
                                    # Template: "{analysis_base}/bucket_{bucket}/"
                                    # Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/"

    models_dir: str                 # Models directory
                                    # Template: "{bucket_base}/models/"
                                    # Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/models/"

    ml_analysis_dir: str            # ML analysis directory
                                    # Template: "{bucket_base}/ml_analysis/"
                                    # Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/ml_analysis/"

    # ===== BUCKET CONFIGURATION (from FoundationCHILD.md Section 6) =====
    windows: list[str]              # Window list from BUCKET_WINDOWS configuration
                                    # Source: config/bucket_definitions.py
                                    # Example for bucket "18-33s": ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

    # ===== STAGE 5 MODEL FILES (from MLAnalysisGenerationCHILD.md Section 3.1) =====
    # Video-Level RF Model
    rf_video_model_path: str        # Path to video-level RF model
                                    # Location: "{models_dir}/rf_video_{bucket}.pkl"
                                    # Schema: RandomForestClassifier with feature_importances_, feature_names_in_ attributes
                                    # Source: Stage 5 (ML Model Training)

    # Window-Level RF Models (6-7 files)
    rf_window_model_paths: dict[str, str]  # Window name → model path
                                           # Location: "{models_dir}/rf_{window}_{bucket}.pkl"
                                           # Schema: RandomForestClassifier per window
                                           # Source: Stage 5

    # K-Means Models (6-7 files)
    kmeans_model_paths: dict[str, str]     # Window name → model path
                                           # Location: "{models_dir}/{window}_kmeans_{bucket}.pkl"
                                           # Schema: KMeans with cluster_centers_, predict() method
                                           # Source: Stage 5

    # Scaler Files (6-7 files)
    scaler_paths: dict[str, str]           # Window name → scaler path
                                           # Location: "{models_dir}/{window}_scalers_{bucket}.pkl"
                                           # Schema: MinMaxScaler objects
                                           # Source: Stage 5 (validated for completeness, not directly used)

    # X Data Files (6-7 files)
    x_data_paths: dict[str, str]           # Window name → X data path
                                           # Location: "{models_dir}/{window}_X_data_{bucket}.pkl"
                                           # Schema: DataFrame or NumPy array with feature names
                                           # Source: Stage 5

    # Model Metrics
    model_metrics_path: str         # Path to model metrics JSON
                                    # Location: "{models_dir}/model_metrics.json"
                                    # Schema: accuracy, precision, recall per window
                                    # Source: Stage 5

    # ===== STAGE 4 CSV FILES (from MLAnalysisGenerationCHILD.md Section 3.1) =====
    # Aggregated Features CSV
    aggregated_csv_path: str        # Path to aggregated features CSV
                                    # Location: "{ml_analysis_dir}/aggregated_features.csv"
                                    # Schema: 129 columns for bucket "18-33s" (21 features × 6 windows + 3 metadata)
                                    # Source: Stage 3 (Feature Aggregation)
                                    # Purpose: Distribution analysis for video-level RF

    # RF Transformed CSV
    rf_transformed_csv_path: str    # Path to video-level RF transformed CSV
                                    # Location: "{ml_analysis_dir}/rf_transformed.csv"
                                    # Schema: 178-190 columns (includes cross-window features)
                                    # Source: Stage 4 (Feature Transformation)

    # Window RF Transformed CSVs (6-7 files)
    window_rf_csv_paths: dict[str, str]    # Window name → CSV path
                                           # Location: "{ml_analysis_dir}/{window}_rf_transformed.csv"
                                           # Schema: 21 columns per window
                                           # Source: Stage 4
                                           # Purpose: Distribution analysis for window-level RF

    # Window K-Means Transformed CSVs (6-7 files)
    window_km_csv_paths: dict[str, str]    # Window name → CSV path
                                           # Location: "{ml_analysis_dir}/{window}_km_transformed.csv"
                                           # Schema: 21-39 columns with _scaled, _log, _encoded suffixes
                                           # Source: Stage 4
                                           # Purpose: Cluster assignment and distance computation

# ===== TOTAL INPUT FILE COUNT BY BUCKET =====
# Formula: 1 (Stage 3) + [1 + (2 × N)] (Stage 4) + [2 + (4 × N)] (Stage 5)
#        = 4 + (6 × N), where N = window_count
#
# Breakdown for bucket "18-33s" (N = 6 windows):
#   Stage 3: 1 file
#     - aggregated_features.csv
#
#   Stage 4: 13 files
#     - rf_transformed.csv (1 video-level)
#     - {window}_rf_transformed.csv (6 window-level)
#     - {window}_km_transformed.csv (6 K-Means transformed)
#
#   Stage 5: 26 files
#     - rf_video_{bucket}.pkl (1 video-level model)
#     - rf_{window}_{bucket}.pkl (6 window-level models)
#     - {window}_kmeans_{bucket}.pkl (6 K-Means models)
#     - {window}_scalers_{bucket}.pkl (6 scalers)
#     - {window}_X_data_{bucket}.pkl (6 X data matrices)
#     - model_metrics.json (1 metrics file)
#
#   TOTAL: 1 + 13 + 26 = 40 files
#
# File counts by bucket:
#   0-3s (1 window):     4 + (6×1) = 10 files
#   3-9s (2 windows):    4 + (6×2) = 16 files
#   9-13s (3 windows):   4 + (6×3) = 22 files
#   13-18s (3 windows):  4 + (6×3) = 22 files
#   18-33s (6 windows):  4 + (6×6) = 40 files
#   33-60s (7 windows):  4 + (6×7) = 46 files
#   60-90s (7 windows):  4 + (6×7) = 46 files
#   90-120s (7 windows): 4 + (6×7) = 46 files
```

### 2.2 Output Contract

```python
# Sources: FoundationCHILD.md Section 2.2 | MLAnalysisGenerationCHILD.md Sections 3.2, 5.2

class Stage6Output:
    """
    Exact structure Stage 6 produces for downstream stages.

    Sources:
    - Output contracts: MLAnalysisGenerationCHILD.md Section 3.2
    - Output schemas: MLAnalysisGenerationCHILD.md Section 5.2
    - Directory paths: FoundationCHILD.md Section 2.2
    """

    # ===== OUTPUT FILES =====
    # Video-Level RF JSON (1 file)
    rf_video_json_path: str         # Path to video-level RF analysis JSON
                                    # Location: "{ml_analysis_dir}/rf_video_analysis.json"
                                    # Example: "/data/clients/acme_corp/.../bucket_18-33s/ml_analysis/rf_video_analysis.json"
                                    # Schema: MLAnalysisGenerationCHILD.md Section 5.2 (VideoRFSchema)
                                    # Format: JSON
                                    # Size: ~30KB
                                    # Consumers: Stage 7 Phase 2 (Cross-window synthesis)

    # Window-Level RF JSONs (6-7 files)
    window_rf_json_paths: dict[str, str]   # Window name → JSON path
                                           # Location: "{ml_analysis_dir}/{window}_rf_analysis.json"
                                           # Example: ".../hook_rf_analysis.json"
                                           # Schema: MLAnalysisGenerationCHILD.md Section 5.2 (WindowRFSchema)
                                           # Format: JSON
                                           # Size: ~5KB each
                                           # Consumers: Stage 7 Phase 1 (Per-window analysis)

    # Window-Level K-Means JSONs (6-7 files)
    window_kmeans_json_paths: dict[str, str]  # Window name → JSON path
                                              # Location: "{ml_analysis_dir}/{window}_kmeans_analysis.json"
                                              # Example: ".../hook_kmeans_analysis.json"
                                              # Schema: MLAnalysisGenerationCHILD.md Section 5.2 (WindowKMeansSchema)
                                              # Format: JSON
                                              # Size: ~5KB each
                                              # Consumers: Stage 7 Phase 1 (Cluster interpretation)

    # ===== OUTPUT COUNTS BY BUCKET =====
    total_json_count: int           # Total JSON files generated
                                    # Formula: 1 + (window_count × 2)
                                    # Bucket "18-33s": 1 + (6 × 2) = 13 files
                                    # Bucket "90-120s": 1 + (7 × 2) = 15 files

    # ===== OUTPUT SCHEMA DETAILS =====
    # See Section 3: Data Schemas for complete field definitions

    # ===== EXIT CODES (from FoundationCHILD.md Section 7) =====
    exit_code_success: int = 0      # All 13-15 JSONs generated successfully
    exit_code_preflight_fail: int = 1  # Stage 4 or Stage 5 dependencies missing
    exit_code_generation_fail: int = 2  # JSON generation failed (model loading, CSV parsing, etc.)
    exit_code_validation_fail: int = 3  # Output validation failed (schema errors, feature name issues)
    exit_code_io_fail: int = 4      # Disk I/O error (disk full, permission denied)
```

---

## Section 3: Data Schemas

### 3.1 Foundation Schemas

These schemas are defined in FoundationCHILD.md and used across all pipeline stages.

```python
# Source: FoundationCHILD.md Section 5.1
ConfigSchema = {
    "client_id": str,              # Required, alphanumeric + underscore, Example: "acme_corp"
    "analysis_type": str,          # Required, ["hashtag", "competitor", "creator"], Example: "hashtag"
    "target": str,                 # Required, format depends on analysis_type, Example: "#nutrition"
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

# Source: FoundationCHILD.md Section 5.3
CheckpointSchema = {
    "stage": str,                  # Required, Stage name, Example: "ml_analysis_generation"
    "bucket": str,                 # Required, Bucket name, Example: "18-33s"
    "total_videos": int,           # Required, Total videos to process, Example: 100
    "completed": int,              # Required, Successfully processed, Example: 100
    "failed": int,                 # Required, Failed with errors, Example: 0
    "remaining": int,              # Required, Not yet processed, Example: 0
    "last_checkpoint": str,        # Required, ISO timestamp, Example: "2025-01-28T14:32:15Z"
    "completed_video_ids": list[str],   # Required, List of processed video IDs
    "failed_video_ids": list[dict],     # Required, List of failure records
}
```

### 3.2 Stage 6 Input Schemas

```python
# Source: MLAnalysisGenerationCHILD.md Section 5.1

# ===== INPUT SCHEMA 1: Stage 5 Video-Level RF Model =====
# File: models/rf_video_{bucket}.pkl
VideoLevelRFModelSchema = {
    # Scikit-learn RandomForestClassifier attributes
    "feature_importances_": "numpy.ndarray",  # Required, shape (n_features,), Range: 0.0-1.0
                                              # Length: 183 for bucket "18-33s" (includes cross-window features)
                                              # Example: [0.22, 0.18, 0.15, ...]

    "feature_names_in_": "numpy.ndarray",     # Required, shape (n_features,), dtype: str
                                              # Example: ['hook_eye_contact_rate', 'middle_avg_word_count', ...]
}

# ===== INPUT SCHEMA 2: Stage 5 Window-Level RF Models =====
# Files: models/rf_{window}_{bucket}.pkl (6-7 files per bucket)
WindowLevelRFModelSchema = {
    # Scikit-learn RandomForestClassifier attributes
    "feature_importances_": "numpy.ndarray",  # Required, shape (21,), Range: 0.0-1.0
                                              # Always 21 features per window
                                              # Example: [0.35, 0.22, 0.18, ...]

    "feature_names_in_": "numpy.ndarray",     # Required, shape (21,), dtype: str
                                              # No window prefix (e.g., 'eye_contact_rate' NOT 'hook_eye_contact_rate')
                                              # Example: ['eye_contact_rate', 'scene_count', 'word_count', ...]
}

# ===== INPUT SCHEMA 3: Stage 5 K-Means Models =====
# Files: models/{window}_kmeans_{bucket}.pkl (6-7 files per bucket)
KMeansModelSchema = {
    # Scikit-learn KMeans attributes
    "cluster_centers_": "numpy.ndarray",      # Required, shape (3, n_features)
                                              # 3 clusters × 21-39 features (varies by window)
                                              # Example: [[0.87, 0.45, ...], [0.32, 0.78, ...], [0.65, 0.21, ...]]

    "predict": "method",                      # Required, method for cluster assignment
                                              # Signature: predict(X: numpy.ndarray) -> numpy.ndarray
}

# ===== INPUT SCHEMA 4: Stage 5 X Data Files =====
# Files: models/{window}_X_data_{bucket}.pkl (6-7 files per bucket)
XDataSchema = {
    # Can be DataFrame or NumPy array
    # If DataFrame:
    "columns": "pandas.Index",                # Feature names with suffixes
                                              # Example: ['eye_contact_rate_scaled', 'scene_count_scaled', ...]

    # If NumPy array:
    # Fallback: load from {window}_km_transformed.csv header
}

# ===== INPUT SCHEMA 5: Stage 5 Model Metrics =====
# File: models/model_metrics.json
# Source: MLModelTrainingCHILDTI.md Section 3.3 (Stage 5 TI Output Schema)
# NOTE: This schema was corrected to match Stage 5's actual output structure
ModelMetricsSchema = {
    "bucket": str,                            # Required, bucket name, Example: "18-33s"
    "total_videos": int,                      # Required, number of videos trained on, Example: 100

    "video_level_rf": {                       # Required, video-level RF metrics
        "model_type": str,                    # Required, always "random_forest"
        "trained": bool,                      # Required, True if trained, False if skipped
        "accuracy": float,                    # Required (if trained), Range: 0.0-1.0, Example: 0.82
        "precision": float,                   # Required (if trained), Range: 0.0-1.0, Example: 0.85
        "recall": float,                      # Required (if trained), Range: 0.0-1.0, Example: 0.78
        "f1_score": float,                    # Required (if trained), Range: 0.0-1.0, Example: 0.88
        "top_feature": str,                   # Required (if trained), most important feature, Example: "hook_eye_contact_rate"
        "top_feature_importance": float,      # Required (if trained), Range: 0.0-1.0, Example: 0.22
        "skip_reason": str,                   # Optional (if not trained), Example: "Single class in 'top' mode"
    },

    "window_level_rf": {                      # Required, per-window RF metrics
        "{window}": {                         # Key: hook, middle_1, middle_2, etc. (6-7 entries depending on bucket)
            "model_type": str,                # Required, always "random_forest"
            "trained": bool,                  # Required, True if trained, False if skipped
            "accuracy": float,                # Required (if trained), Range: 0.0-1.0, Example: 0.82
            "precision": float,               # Required (if trained), Range: 0.0-1.0, Example: 0.85
            "recall": float,                  # Required (if trained), Range: 0.0-1.0, Example: 0.78
            "f1_score": float,                # Required (if trained), Range: 0.0-1.0, Example: 0.88
            "top_feature": str,               # Required (if trained), most important feature (no window prefix), Example: "eye_contact_rate"
            "top_feature_importance": float,  # Required (if trained), Range: 0.0-1.0, Example: 0.35
            "skip_reason": str,               # Optional (if not trained), Example: "Insufficient videos"
        }
        # Repeat for each window (1-7 entries)
    },

    # NOTE: Stage 6 only uses window_level_rf.{window}.accuracy/precision/recall for display
    # Actual feature importance comes from model.feature_importances_, not this JSON
}

# ===== INPUT SCHEMA 6: Stage 3 Aggregated Features CSV =====
# File: ml_analysis/aggregated_features.csv
# Source: MLAnalysisGenerationCHILD.md Section 5.1 Table (bucket 18-33s example)
AggregatedFeaturesSchema = {
    # Metadata columns
    "video_id": str,                          # Required, Unique video identifier, Example: "7428596413707144481"
    "create_time": "datetime",                # Required, Video publish timestamp, Example: "2025-01-15 14:30:00"
    "gender": str,                            # Optional (Nulls: Yes), ["male", "female", None], Example: "female"

    # Window feature columns (21 features × 6 windows = 126 columns for bucket 18-33s)
    "hook_scene_count": int,                  # Required, Range: 0-20, Example: 3
    "hook_eye_contact_rate": float,           # Required, Range: 0.0-1.0, Example: 0.85
    "hook_word_count": int,                   # Required, Range: 0-200, Example: 14
    "hook_energy_level": float,               # Required, Range: 0.0-1.0, Example: 0.55
    # ... (18 more hook features)

    "middle_1_scene_count": int,              # Required, Range: 0-20, Example: 5
    "middle_1_word_count": int,               # Required, Range: 0-200, Example: 48
    # ... (19 more middle_1 features)

    # ... (middle_2, middle_3, middle_4 features - 21 each)

    "closing_energy_level": float,            # Required, Range: 0.0-1.0, Example: 0.75
    # ... (20 more closing features)

    # Total: 3 metadata + 126 window features = 129 columns for bucket "18-33s"
}

# ===== INPUT SCHEMA 7: Stage 4 Window RF Transformed CSVs =====
# Files: ml_analysis/{window}_rf_transformed.csv (6-7 files per bucket)
WindowRFTransformedSchema = {
    # 21 features per window (no metadata columns, no window prefix in column names)
    "eye_contact_rate": float,                # Required, Range: 0.0-1.0 (or transformed), Example: 0.88
    "scene_count": int,                       # Required, Range: 0-20 (or transformed), Example: 3
    "word_count": int,                        # Required, Range: 0-200 (or transformed), Example: 14
    "energy_level": float,                    # Required, Range: 0.0-1.0, Example: 0.55
    # ... (17 more features - exact 21 features per window)
}

# ===== INPUT SCHEMA 8: Stage 4 Window K-Means Transformed CSVs =====
# Files: ml_analysis/{window}_km_transformed.csv (6-7 files per bucket)
WindowKMeansTransformedSchema = {
    # 21-39 features with transformation suffixes (_scaled, _log, _encoded)
    "eye_contact_rate_scaled": float,         # Required, Range: 0.0-1.0 (scaled), Example: 0.87
    "scene_count_scaled": float,              # Required, Range: 0.0-1.0 (scaled), Example: 0.45
    "word_count_scaled": float,               # Required, Range: 0.0-1.0 (scaled), Example: 0.62
    "energy_level_scaled": float,             # Required, Range: 0.0-1.0, Example: 0.73
    "has_captions_encoded": int,              # Required, Range: 0-1 (label encoded), Example: 1
    # ... (16-34 more features with suffixes)

    # Note: Feature count varies by window (21-39)
    # CRITICAL: Suffixes must be removed during JSON generation (see Section 4.4)
}
```

### 3.3 Stage 6 Output Schemas

```python
# Source: MLAnalysisGenerationCHILD.md Section 5.2

# ===== OUTPUT SCHEMA 1: Video-Level RF Analysis JSON =====
# File: ml_analysis/rf_video_analysis.json (~30KB)
VideoRFAnalysisSchema = {
    "analysis_type": str,                     # Required, Fixed: "random_forest"
    "bucket": str,                            # Required, Example: "18-33s"
    "hashtag": str,                           # Optional (can be None), Example: "#nutrition"
    "video_count": int,                       # Required, Range: 50-300, Example: 100
    "input_features": int,                    # Required, Range: 24-220, Example: 178
                                              # Varies by bucket (includes cross-window features)

    "feature_importance": list[dict],         # Required, Length: 10 (top 10 features)
    # Each feature_importance entry:
    {
        "feature": str,                       # Required, Example: "hook_eye_contact_rate"
        "importance": float,                  # Required, Range: 0.0-1.0, Example: 0.22
        "top_performer_avg": float,           # Required, Mean value in top 80%, Example: 0.88
        "bottom_performer_avg": float,        # Required, Mean value in bottom 20%, Example: 0.45
        "gap": float,                         # Required, Absolute difference, Example: 0.43

        "distribution": {                     # Required, Percentile analysis
            "thresholds": {
                "high": float,                # Required, 66th percentile, Example: 0.6
                "low": float,                 # Required, 33rd percentile, Example: 0.4
            },
            "top_performers": {
                "high_percentage": float,     # Required, Range: 0.0-1.0, Example: 0.70
                "medium_percentage": float,   # Required, Range: 0.0-1.0, Example: 0.25
                "low_percentage": float,      # Required, Range: 0.0-1.0, Example: 0.05
            },
            "bottom_performers": {
                "high_percentage": float,     # Required, Range: 0.0-1.0, Example: 0.05
                "medium_percentage": float,   # Required, Range: 0.0-1.0, Example: 0.15
                "low_percentage": float,      # Required, Range: 0.0-1.0, Example: 0.80
            }
        }
    }
    # Repeat for 10 features
}

# ===== OUTPUT SCHEMA 2: Window-Level RF Analysis JSON =====
# Files: ml_analysis/{window}_rf_analysis.json (~5KB each, 6-7 files)
WindowRFAnalysisSchema = {
    "model_type": str,                        # Required, Fixed: "window_level_rf"
    "window_type": str,                       # Required, Example: "hook", "middle_1", "closing"
    "bucket": str,                            # Required, Example: "18-33s"
    "total_videos": int,                      # Required, Range: 50-300, Example: 100
    "input_features": int,                    # Required, Fixed: 21 (always 21 features per window)

    "model_performance": {                    # Required
        "accuracy": float,                    # Required, Range: 0.0-1.0, Example: 0.82
        "precision": float,                   # Required, Range: 0.0-1.0, Example: 0.85
        "recall": float,                      # Required, Range: 0.0-1.0, Example: 0.78
    },

    "feature_importance": list[dict],         # Required, Length: 10 (top 10 features)
    # Each feature_importance entry:
    {
        "feature": str,                       # Required, No window prefix, Example: "eye_contact_rate"
        "importance": float,                  # Required, Range: 0.0-1.0, Example: 0.35
        "top_performer_avg": float,           # Required, Example: 0.88
        "bottom_performer_avg": float,        # Required, Example: 0.45
        "gap": float,                         # Required, Absolute difference, Example: 0.43
        "rank": int,                          # Required, Range: 1-10, Example: 1
    }
    # Repeat for 10 features
}

# ===== OUTPUT SCHEMA 3: Window-Level K-Means Analysis JSON =====
# Files: ml_analysis/{window}_kmeans_analysis.json (~5KB each, 6-7 files)
WindowKMeansAnalysisSchema = {
    "window_type": str,                       # Required, Example: "hook", "middle_1", "closing"
    "bucket": str,                            # Required, Example: "18-33s"
    "total_videos": int,                      # Required, Range: 50-300, Example: 100
    "n_clusters": int,                        # Required, Fixed: 3 (always 3 clusters)

    "clusters": list[dict],                   # Required, Length: 3
    # Each cluster entry:
    {
        "cluster_id": int,                    # Required, Range: 0-2, Example: 0
        "size": int,                          # Required, Range: 1-300, Example: 35

        "centroid": dict,                     # Required, 21-39 feature keys
                                              # CRITICAL: Feature names NORMALIZED (no _scaled suffixes)
        # Example centroid keys:
        {
            "eye_contact_rate": float,        # Required, Range: 0.0-1.0, Example: 0.87
            "scene_count": float,             # Required, Range: 0.0-1.0 (scaled), Example: 0.45
            "word_count": float,              # Required, Range: 0.0-1.0 (scaled), Example: 0.62
            "energy_level": float,            # Required, Range: 0.0-1.0, Example: 0.73
            "has_captions": int,              # Required, Range: 0-1, Example: 1
            # ... (16-34 more features, all normalized - NO suffixes)
        },

        "videos": list[dict],                 # Required, Length: cluster size
        # Each video entry:
        {
            "video_id": str,                  # Required, Example: "video_0"
            "distance_to_centroid": float,    # Required, Range: 0.0+, Example: 0.15
        }
    }
    # Repeat for 3 clusters
}
```

---

## Section 4: Algorithmic Specifications

**Source**: MLAnalysisGenerationCHILD.md Section 2.3 (Detailed Process) + Appendix C (Pseudocode)

### 4.1 Function: validate_stage_dependencies()

**Purpose**: Ensure all Stage 4 and Stage 5 dependencies exist before generating any JSONs (fail-fast principle)

**Implementation** (from MLAnalysisGenerationCHILD.md Section 2.3.1):

```python
def validate_stage_dependencies(bucket_path: str, bucket: str, windows: list[str]) -> None:
    """
    Validate Stage 4 and Stage 5 outputs exist before generating any JSONs.

    Args:
        bucket_path: str - Absolute path to bucket directory (e.g., /data/clients/acme/buckets/bucket_18-33s)
        bucket: str - Bucket name (e.g., "18-33s")
        windows: list[str] - Window list from bucket configuration (e.g., ['hook', 'middle_1', ..., 'closing'])

    Returns:
        None

    Raises:
        PreFlightValidationError: If any required file missing

    Source: MLAnalysisGenerationCHILD.md Section 2.3.1
    """
    missing_files = []

    # ===== Step 1: Validate Stage 4 CSVs (13 files for bucket 18-33s) =====
    required_stage4_files = [
        'ml_analysis/aggregated_features.csv',         # For distribution analysis
        'ml_analysis/rf_transformed.csv',              # Video-level RF input
    ]

    # Add window-level RF transformed CSVs
    for window in windows:
        required_stage4_files.append(f'ml_analysis/{window}_rf_transformed.csv')

    # Add window-level K-Means transformed CSVs
    for window in windows:
        required_stage4_files.append(f'ml_analysis/{window}_km_transformed.csv')

    # Check each Stage 4 file exists
    for file_path in required_stage4_files:
        full_path = os.path.join(bucket_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(('Stage 4', file_path))

    # ===== Step 2: Validate Stage 5 Models (20 files for bucket 18-33s) =====
    required_stage5_files = [
        f'models/rf_video_{bucket}.pkl',               # Video-level RF model
        'models/model_metrics.json'                    # Performance metrics
    ]

    # Add window-level RF models
    for window in windows:
        required_stage5_files.append(f'models/rf_{window}_{bucket}.pkl')

    # Add K-Means models
    for window in windows:
        required_stage5_files.append(f'models/{window}_kmeans_{bucket}.pkl')

    # Add scalers
    for window in windows:
        required_stage5_files.append(f'models/{window}_scalers_{bucket}.pkl')

    # Add X data matrices
    for window in windows:
        required_stage5_files.append(f'models/{window}_X_data_{bucket}.pkl')

    # Check each Stage 5 file exists
    for file_path in required_stage5_files:
        full_path = os.path.join(bucket_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(('Stage 5', file_path))

    # ===== Step 3: Fail-fast if any dependencies missing =====
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

    # ===== Step 4: Log success =====
    logger.info(f"✓ Pre-flight validation passed: All {len(required_stage4_files)} Stage 4 files + {len(required_stage5_files)} Stage 5 files exist")
```

**Edge Cases** (from MLAnalysisGenerationCHILD.md Section 2.3.1):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Single file missing (e.g., 19 of 20 Stage 5 models exist) | Fail-fast with specific file name | Partial model set is unusable - prevent ambiguous failures later |
| Stage 4 incomplete but Stage 5 complete | Fail with Stage 4 message | Stage 6 needs both stages - clear error prevents confusion |
| Temp directory already exists from previous run | Delete temp directory before starting | Clean state prevents leftover files from failed previous run |

---

### 4.2 Function: generate_video_rf_json()

**Purpose**: Extract cross-window feature importance with distribution analysis for LLM consumption

**Implementation** (from MLAnalysisGenerationCHILD.md Section 2.3.2):

```python
def generate_video_rf_json(bucket_path: str, bucket: str) -> dict:
    """
    Generate Video-Level RF analysis JSON with distribution statistics.

    Args:
        bucket_path: str - Absolute path to bucket directory
        bucket: str - Bucket name (e.g., "18-33s")

    Returns:
        dict: Video RF analysis JSON structure

    Source: MLAnalysisGenerationCHILD.md Section 2.3.2
    """
    # ===== Step 1: Load Video-Level RF Model =====
    model_path = os.path.join(bucket_path, f'models/rf_video_{bucket}.pkl')
    rf_model = joblib.load(model_path)

    # ===== Step 2: Extract Feature Importance =====
    feature_importances = rf_model.feature_importances_  # NumPy array (length = 183 for bucket 18-33s)
    feature_names = rf_model.feature_names_in_  # From sklearn attribute

    # Sort features by importance, take top 10
    importance_indices = np.argsort(feature_importances)[::-1][:10]
    top_features = []

    for idx in importance_indices:
        top_features.append({
            'feature': feature_names[idx],
            'importance': float(feature_importances[idx])
        })

    # ===== Step 3: Load aggregated_features.csv for Distribution Analysis =====
    # NOTE: Video-level uses aggregated_features.csv (Stage 3) not rf_transformed.csv (Stage 4)
    # Rationale: Video-level RF includes cross-window features computed from raw aggregated values
    agg_csv_path = os.path.join(bucket_path, 'ml_analysis/aggregated_features.csv')
    df = pd.read_csv(agg_csv_path)

    # Determine top/bottom performers (contrastive strategy)
    video_count = len(df)
    top_count = int(video_count * 0.8)
    df['is_top_performer'] = [1] * top_count + [0] * (video_count - top_count)

    # ===== Step 4: Compute Distribution Stats for Each Top Feature =====
    for feature_data in top_features:
        feature_name = feature_data['feature']

        # Skip if feature not in aggregated CSV (e.g., derived features)
        if feature_name not in df.columns:
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

    # ===== Step 5: Build Video RF Analysis JSON =====
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

**Edge Cases** (from MLAnalysisGenerationCHILD.md Section 2.3.2):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Feature in RF model but not in aggregated CSV (derived feature) | Set distribution fields to `None` | Cannot compute distribution for features not in raw data - LLM can still use importance score |
| All videos have same value for a feature (variance=0) | Percentiles will equal mean | Valid edge case - distribution shows no spread (all values in "medium" range) |
| Video count mismatch (RF trained on 100 videos, CSV has 98 rows) | Log warning but continue | Non-critical - distribution based on available data |

---

### 4.3 Function: generate_window_rf_json()

**Purpose**: Extract per-window feature importance rankings for focused window analysis in Stage 7 Phase 1

**Implementation** (from MLAnalysisGenerationCHILD.md Section 2.3.3):

```python
def generate_window_rf_json(bucket_path: str, bucket: str, window: str) -> dict:
    """
    Generate Window-Level RF analysis JSON.

    Args:
        bucket_path: str - Absolute path to bucket directory
        bucket: str - Bucket name (e.g., "18-33s")
        window: str - Window name (e.g., "hook", "middle_1", "closing")

    Returns:
        dict: Window RF analysis JSON structure

    Source: MLAnalysisGenerationCHILD.md Section 2.3.3
    """
    # ===== Step 1: Load Window-Level RF Model =====
    model_path = os.path.join(bucket_path, f'models/rf_{window}_{bucket}.pkl')
    rf_model = joblib.load(model_path)

    # ===== Step 2: Extract Feature Importance =====
    feature_importances = rf_model.feature_importances_  # Always 21 features per window
    feature_names = rf_model.feature_names_in_  # e.g., ['eye_contact_rate', 'scene_count', ...]

    # Sort by importance, take top 10
    importance_indices = np.argsort(feature_importances)[::-1][:10]
    top_features = []

    for rank, idx in enumerate(importance_indices):
        top_features.append({
            'feature': feature_names[idx],
            'importance': float(feature_importances[idx]),
            'rank': rank + 1
        })

    # ===== Step 3: Load model_metrics.json for Performance Stats =====
    metrics_path = os.path.join(bucket_path, 'models/model_metrics.json')
    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    # Extract metrics for this window (nested under 'window_level_rf')
    # Schema: all_metrics['window_level_rf'][window] (e.g., all_metrics['window_level_rf']['hook'])
    window_metrics = all_metrics.get('window_level_rf', {}).get(window, {})

    # ===== Step 4: Compute Distribution Stats =====
    # NOTE: Window-level uses {window}_rf_transformed.csv (Stage 4) not aggregated_features.csv
    # Rationale: Window-level RF was trained on Stage 4 transformed data
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

    # ===== Step 5: Build Window RF Analysis JSON =====
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

**Edge Cases** (from MLAnalysisGenerationCHILD.md Section 2.3.3):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| model_metrics.json missing performance stats for window | Set accuracy/precision/recall to `None` | Non-critical - feature importance still available for LLM |
| Window has <21 features (buckets 9-13s, 13-18s use middle_aggregate) | Use actual feature count | Valid configuration - distribution varies by bucket |

---

### 4.4 Function: generate_window_kmeans_json()

**Purpose**: Extract cluster centroids with NORMALIZED feature names for Stage 7 Phase 1 cluster interpretation

**Implementation** (from MLAnalysisGenerationCHILD.md Section 2.3.4):

```python
def normalize_feature_name(feature_name: str) -> str:
    """
    Normalize K-Means feature names for consistency with RF feature names.

    Removes transformation suffixes from Stage 4:
    - '_scaled' (from MinMax scaling)
    - '_log' (from log transformation)
    - '_encoded' (from label encoding)

    Args:
        feature_name: str - e.g., 'eye_contact_rate_scaled'

    Returns:
        str - e.g., 'eye_contact_rate'

    Source: MLAnalysisGenerationCHILD.md Section 2.3.4
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
        bucket_path: str - Absolute path to bucket directory
        bucket: str - Bucket name (e.g., "18-33s")
        window: str - Window name (e.g., "hook", "middle_1", "closing")

    Returns:
        dict: Window K-Means analysis JSON structure

    Source: MLAnalysisGenerationCHILD.md Section 2.3.4
    """
    # ===== Step 1: Load K-Means Model =====
    model_path = os.path.join(bucket_path, f'models/{window}_kmeans_{bucket}.pkl')
    kmeans_model = joblib.load(model_path)

    # ===== Step 2: Extract Cluster Centroids =====
    centroids = kmeans_model.cluster_centers_  # Shape: (3 clusters, 21-39 features)
    n_clusters = centroids.shape[0]  # Always 3
    n_features = centroids.shape[1]  # 21-39 depending on window

    # ===== Step 3: Load Feature Names from X_data =====
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

    # ===== Step 4: Load K-Means Predictions (Cluster Assignments) =====
    km_csv_path = os.path.join(bucket_path, f'ml_analysis/{window}_km_transformed.csv')
    df_km = pd.read_csv(km_csv_path)

    # Predict cluster assignments
    cluster_labels = kmeans_model.predict(df_km[feature_names])

    # ===== Step 5: Build Clusters with NORMALIZED Feature Names =====
    clusters = []

    for cluster_id in range(n_clusters):
        # Get centroid values
        centroid_values = centroids[cluster_id]

        # CRITICAL: Normalize feature names before creating centroid dict
        normalized_centroid = {}
        for name, value in zip(feature_names, centroid_values):
            normalized_name = normalize_feature_name(name)
            normalized_centroid[normalized_name] = float(value)

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

    # ===== Step 6: Build K-Means Analysis JSON =====
    analysis_json = {
        'window_type': window,
        'bucket': bucket,
        'total_videos': len(df_km),
        'n_clusters': n_clusters,
        'clusters': clusters
    }

    return analysis_json
```

**Edge Cases** (from MLAnalysisGenerationCHILD.md Section 2.3.4):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Feature already normalized (no suffix) | `normalize_feature_name('scene_count')` → `'scene_count'` (no change) | Valid - some features don't need normalization |
| Multiple suffixes (rare) | `normalize_feature_name('word_count_log_scaled')` → `'word_count'` (both removed) | Edge case handled by sequential replacement |
| X_data is numpy array without feature names | Fallback to loading `{window}_km_transformed.csv` header | Ensures feature names always available |

---

### 4.5 Function: generate_ml_analysis_jsons()

**Purpose**: Generate all ML analysis JSONs using atomic pattern - either all JSONs succeed or all deleted (count = 1 video RF + 2×window_count)

**Implementation** (from MLAnalysisGenerationCHILD.md Section 2.3.5):

```python
def generate_ml_analysis_jsons(bucket_path: str, bucket: str, windows: list[str]) -> int:
    """
    Generate all ML analysis JSONs using atomic pattern.
    Either all JSONs succeed or all deleted (count = 1 + 2×len(windows)).

    Args:
        bucket_path: str - Absolute path to bucket directory
        bucket: str - Bucket name (e.g., "18-33s")
        windows: list[str] - Window list (e.g., ['hook', 'middle_1', ..., 'closing'])

    Returns:
        int: Exit code (0=success, 1=pre-flight fail, 2=generation fail, 3=validation fail, 4=I/O fail)

    Source: MLAnalysisGenerationCHILD.md Section 2.3.5
    """
    temp_dir = os.path.join(bucket_path, 'ml_analysis/.tmp/')
    os.makedirs(temp_dir, exist_ok=True)
    generated_files = []

    try:
        # ===== Step 1: PRE-FLIGHT VALIDATION =====
        logger.info("Pre-flight validation: checking Stage 4 and Stage 5 outputs...")
        validate_stage_dependencies(bucket_path, bucket, windows)
        logger.info("✓ Pre-flight validation passed")

        # ===== Step 2: GENERATE ALL JSONs TO TEMP DIRECTORY =====
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

        # ===== Step 3: VALIDATE ALL JSON SCHEMAS =====
        logger.info("Validating JSON schemas...")
        validate_all_json_schemas(temp_dir, bucket, windows)
        logger.info("✓ All JSON schemas valid")

        # ===== Step 4: ATOMIC COMMIT: RENAME ALL TEMP FILES AT ONCE =====
        logger.info("Committing JSONs (atomic rename)...")
        for temp_file in generated_files:
            final_path = temp_file.replace('/.tmp/', '/').replace('.json.tmp', '.json')
            os.rename(temp_file, final_path)
            logger.info(f"  ✓ Committed {os.path.basename(final_path)}")

        # ===== Step 5: CLEANUP TEMP DIRECTORY =====
        shutil.rmtree(temp_dir)
        logger.info(f"✓ Stage 6 complete: {len(generated_files)} JSONs generated")

        return 0  # SUCCESS

    except PreFlightValidationError as e:
        logger.error(f"Pre-flight validation failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 1  # PRE-FLIGHT VALIDATION FAILED

    except json.JSONDecodeError as e:
        logger.error(f"Stage 6 JSON generation failed: Invalid JSON - {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning("Rolled back: Deleted all temp files (atomic failure)")
        return 3  # JSON VALIDATION FAILED

    except IOError as e:
        logger.error(f"Disk I/O failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 4  # DISK I/O FAILED

    except Exception as e:
        logger.error(f"Stage 6 JSON generation failed: {type(e).__name__}: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc(limit=10)}")
        logger.error(f"Generated files before failure: {len(generated_files)} of {1 + 2*len(windows)} expected")
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning("Rolled back: Deleted all temp files (atomic failure)")
        return 2  # JSON GENERATION FAILED
```

**Edge Cases** (from MLAnalysisGenerationCHILD.md Section 2.3.5):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Mid-generation crash (8 of 13 files created) | Delete all temp files, exit code 2 | Atomic pattern ensures no partial output |
| Disk full during JSON write | Delete all temp files, exit code 4 | Clear error message directs user to disk issue |
| Validation fails after all JSONs generated | Delete all temp files, exit code 3 | Prevent Stage 7 from loading malformed JSONs |

---

## Section 5: Validation Rules

**Source**: MLAnalysisGenerationCHILD.md Section 6 (Error Handling & Validation)

### 5.1 Input Validation

```python
# Source: MLAnalysisGenerationCHILD.md Section 6.1

def validate_stage_dependencies(bucket_path: str, bucket: str, windows: list[str]) -> None:
    """
    Pre-flight validation: Ensure all Stage 4 and Stage 5 dependencies exist.

    See Section 4.1 for complete implementation.

    Validation Rules:
    1. All Stage 4 CSV files must exist (13 files for bucket 18-33s)
    2. All Stage 5 model files must exist (20 files for bucket 18-33s)
    3. Fail-fast if ANY file missing (partial sets unusable)

    Source: MLAnalysisGenerationCHILD.md Section 6.1
    """
    pass  # Implementation in Section 4.1
```

**Input Validation Rules** (from MLAnalysisGenerationCHILD.md Section 6.1):

| Rule ID | Validation | Check Method | Failure Action | Exit Code |
|---------|------------|--------------|----------------|-----------|
| **IV-1** | Stage 4 aggregated_features.csv exists | `os.path.exists(bucket_path + '/ml_analysis/aggregated_features.csv')` | Fail-fast with error message listing missing file | 1 |
| **IV-2** | Stage 4 rf_transformed.csv exists | `os.path.exists(bucket_path + '/ml_analysis/rf_transformed.csv')` | Fail-fast with error message | 1 |
| **IV-3** | Stage 4 window RF CSVs exist (6-7 files) | `os.path.exists()` for each `{window}_rf_transformed.csv` | Fail-fast, list all missing files (max 5 shown) | 1 |
| **IV-4** | Stage 4 window K-Means CSVs exist (6-7 files) | `os.path.exists()` for each `{window}_km_transformed.csv` | Fail-fast, list all missing files | 1 |
| **IV-5** | Stage 5 video RF model exists | `os.path.exists(bucket_path + f'/models/rf_video_{bucket}.pkl')` | Fail-fast with error message | 1 |
| **IV-6** | Stage 5 window RF models exist (6-7 files) | `os.path.exists()` for each `rf_{window}_{bucket}.pkl` | Fail-fast, list all missing files | 1 |
| **IV-7** | Stage 5 K-Means models exist (6-7 files) | `os.path.exists()` for each `{window}_kmeans_{bucket}.pkl` | Fail-fast, list all missing files | 1 |
| **IV-8** | Stage 5 scaler files exist (6-7 files) | `os.path.exists()` for each `{window}_scalers_{bucket}.pkl` | Fail-fast, list all missing files | 1 |
| **IV-9** | Stage 5 X data files exist (6-7 files) | `os.path.exists()` for each `{window}_X_data_{bucket}.pkl` | Fail-fast, list all missing files | 1 |
| **IV-10** | Stage 5 model_metrics.json exists | `os.path.exists(bucket_path + '/models/model_metrics.json')` | Fail-fast with error message | 1 |

**Validation Error Messages** (from MLAnalysisGenerationCHILD.md Section 6.1):

```
Pre-flight validation failed:
Stage 4 incomplete (2 files missing):
  - ml_analysis/hook_rf_transformed.csv
  - ml_analysis/middle_1_km_transformed.csv
Action: Re-run Stage 4 (Feature Transformation)

Stage 5 incomplete (3 files missing):
  - models/rf_hook_18-33s.pkl
  - models/hook_kmeans_18-33s.pkl
  - models/hook_X_data_18-33s.pkl
Action: Re-run Stage 5 (ML Model Training)
```

---

### 5.2 Output Validation

```python
# Source: MLAnalysisGenerationCHILD.md Section 6.3

def validate_all_json_schemas(temp_dir: str, bucket: str, windows: list[str]) -> None:
    """
    Validate all generated JSONs before atomic commit.

    Validation Steps:
    1. Check all expected files exist in temp directory
    2. Validate JSON parseability (valid JSON format)
    3. Validate K-Means feature names (no _scaled suffixes)
    4. Validate cluster size consistency (sum of cluster sizes = total_videos)

    Args:
        temp_dir: str - Path to temp directory with .json.tmp files
        bucket: str - Bucket name
        windows: list[str] - Window list

    Raises:
        ValidationError: If any JSON invalid

    Source: MLAnalysisGenerationCHILD.md Section 6.3
    """
    # ===== Step 1: Check all expected files exist =====
    expected_files = [
        'rf_video_analysis.json.tmp',
    ]
    for window in windows:
        expected_files.append(f'{window}_rf_analysis.json.tmp')
    for window in windows:
        expected_files.append(f'{window}_kmeans_analysis.json.tmp')

    for filename in expected_files:
        filepath = os.path.join(temp_dir, filename)
        if not os.path.exists(filepath):
            raise ValidationError(f"Missing temp file: {filename}")

    # ===== Step 2: Validate JSON parseability =====
    for filename in expected_files:
        filepath = os.path.join(temp_dir, filename)
        try:
            with open(filepath, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in {filename}: {e}")

    # ===== Step 3: Validate K-Means feature names (no _scaled suffixes) =====
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

    # ===== Step 4: Validate cluster size consistency =====
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

    # ===== Step 5: Validate distribution percentages sum to 1.0 (OV-8) =====
    # Check video-level RF JSON
    video_rf_filepath = os.path.join(temp_dir, 'rf_video_analysis.json.tmp')
    with open(video_rf_filepath, 'r') as f:
        video_rf_json = json.load(f)

    for feature_data in video_rf_json.get('feature_importance', []):
        distribution = feature_data.get('distribution')
        if distribution:  # Skip if distribution is None
            top_perf = distribution.get('top_performers', {})
            bottom_perf = distribution.get('bottom_performers', {})

            # Check top performers percentages sum to ~1.0
            top_sum = top_perf.get('high_percentage', 0) + top_perf.get('medium_percentage', 0) + top_perf.get('low_percentage', 0)
            if abs(top_sum - 1.0) > 0.01:
                logger.warning(
                    f"Video RF JSON: Feature '{feature_data.get('feature')}' top_performers percentages sum to {top_sum:.3f}, expected ~1.0 (tolerance: 0.01). "
                    f"Likely rounding error - continuing."
                )

            # Check bottom performers percentages sum to ~1.0
            bottom_sum = bottom_perf.get('high_percentage', 0) + bottom_perf.get('medium_percentage', 0) + bottom_perf.get('low_percentage', 0)
            if abs(bottom_sum - 1.0) > 0.01:
                logger.warning(
                    f"Video RF JSON: Feature '{feature_data.get('feature')}' bottom_performers percentages sum to {bottom_sum:.3f}, expected ~1.0 (tolerance: 0.01). "
                    f"Likely rounding error - continuing."
                )
```

**Output Validation Rules** (from MLAnalysisGenerationCHILD.md Section 6.3):

| Rule ID | Validation | Check Method | Failure Action | Exit Code |
|---------|------------|--------------|----------------|-----------|
| **OV-1** | All expected temp files exist | Count files in `.tmp/` directory, check against expected count (1 + 2×window_count) | Delete all temp files, raise ValidationError | 3 |
| **OV-2** | All JSONs are parseable | `json.load()` for each temp file | Delete all temp files, raise ValidationError with filename | 3 |
| **OV-3** | Video RF JSON has 10 features | Check `len(analysis_json['feature_importance']) == 10` | Delete all temp files, raise ValidationError | 3 |
| **OV-4** | Window RF JSONs have 10 features each | Check `len(analysis_json['feature_importance']) == 10` for each window | Delete all temp files, raise ValidationError | 3 |
| **OV-5** | K-Means JSONs have 3 clusters each | Check `analysis_json['n_clusters'] == 3` for each window | Delete all temp files, raise ValidationError | 3 |
| **OV-6** | K-Means feature names normalized | Check no `_scaled`, `_log`, `_encoded` suffixes in centroid keys | Delete all temp files, raise ValidationError with feature list | 3 |
| **OV-7** | Cluster sizes sum to total_videos | Check `sum(cluster['size'] for cluster in clusters) == total_videos` | Delete all temp files, raise ValidationError | 3 |
| **OV-8** | Distribution percentages sum to 1.0 | Check `top_performers.high + medium + low ≈ 1.0` (within 0.01 tolerance) | Log warning, continue (non-critical rounding error) | 0 |

**Validation Error Messages** (from MLAnalysisGenerationCHILD.md Section 6.3):

```
# Missing temp file
Missing temp file: hook_rf_analysis.json.tmp

# Invalid JSON format
Invalid JSON in middle_1_kmeans_analysis.json.tmp: Expecting ',' delimiter: line 42 column 5 (char 1205)

# Feature name normalization failed
K-Means JSON hook_kmeans_analysis.json.tmp contains 3 features with transformation suffixes: ['eye_contact_rate_scaled', 'scene_count_scaled', 'word_count_scaled'].
Feature normalization failed. Check normalize_feature_name() logic.

# Cluster size mismatch
K-Means JSON closing_kmeans_analysis.json.tmp cluster sizes ([35, 42, 22]) sum to 99, but total_videos is 100. Cluster assignment failed.
```

---

### 5.3 Data Validation Rules

**Video-Level RF JSON Validation** (from MLAnalysisGenerationCHILD.md Section 5.2):

| Field | Validation Rule | Example Valid | Example Invalid | Action |
|-------|----------------|---------------|-----------------|--------|
| `analysis_type` | Must equal `"random_forest"` | `"random_forest"` | `"kmeans"` | Reject, exit code 3 |
| `bucket` | Must match input bucket parameter | `"18-33s"` | `"0-3s"` (mismatch) | Reject, exit code 3 |
| `video_count` | Must be > 0 and ≤ 300 | `100` | `0` or `500` | Reject, exit code 3 |
| `input_features` | Must be > 0 and ≤ 220 | `178` | `0` or `500` | Reject, exit code 3 |
| `feature_importance` | Must have exactly 10 entries | `[{...}, {...}, ... 10 items]` | `[{...}, {...}]` (2 items) | Reject, exit code 3 |
| `feature_importance[].importance` | Must be 0.0-1.0 | `0.22` | `-0.1` or `1.5` | Reject, exit code 3 |
| `distribution.thresholds.high` | Must be ≥ `low` threshold | `high=0.6, low=0.4` | `high=0.3, low=0.5` | Reject, exit code 3 |
| `distribution.*.high_percentage` | Must be 0.0-1.0 | `0.70` | `-0.1` or `1.5` | Reject, exit code 3 |

**Window-Level RF JSON Validation** (from MLAnalysisGenerationCHILD.md Section 5.2):

| Field | Validation Rule | Example Valid | Example Invalid | Action |
|-------|----------------|---------------|-----------------|--------|
| `model_type` | Must equal `"window_level_rf"` | `"window_level_rf"` | `"video_level_rf"` | Reject, exit code 3 |
| `window_type` | Must be in window list | `"hook"` | `"invalid_window"` | Reject, exit code 3 |
| `input_features` | Must equal 21 | `21` | `20` or `22` | Reject, exit code 3 |
| `model_performance.accuracy` | Must be 0.0-1.0 or None | `0.82` or `None` | `-0.1` or `1.5` | Reject, exit code 3 |
| `feature_importance[].rank` | Must be 1-10 | `1`, `5`, `10` | `0` or `11` | Reject, exit code 3 |

**Window-Level K-Means JSON Validation** (from MLAnalysisGenerationCHILD.md Section 5.2):

| Field | Validation Rule | Example Valid | Example Invalid | Action |
|-------|----------------|---------------|-----------------|--------|
| `n_clusters` | Must equal 3 | `3` | `2` or `5` | Reject, exit code 3 |
| `clusters` | Must have exactly 3 entries | `[{...}, {...}, {...}]` | `[{...}, {...}]` (2 items) | Reject, exit code 3 |
| `clusters[].cluster_id` | Must be 0, 1, or 2 (unique) | `0`, `1`, `2` | `3` or duplicate `0` | Reject, exit code 3 |
| `clusters[].size` | Must be > 0 and ≤ total_videos | `35` | `0` or `500` | Reject, exit code 3 |
| `clusters[].centroid` keys | No suffixes (`_scaled`, `_log`, `_encoded`) | `"eye_contact_rate"` | `"eye_contact_rate_scaled"` | Reject, exit code 3 |
| `clusters[].videos[].distance_to_centroid` | Must be ≥ 0.0 | `0.15` | `-0.5` | Reject, exit code 3 |

---

### 5.4 Cross-Field Validation

**Validation Rules Across Multiple Fields** (from MLAnalysisGenerationCHILD.md Section 6.3):

| Rule ID | Description | Validation Check | Failure Action |
|---------|-------------|------------------|----------------|
| **CFV-1** | Sum of cluster sizes equals total_videos | `sum(cluster['size'] for cluster in clusters) == total_videos` | Reject with error message showing mismatch |
| **CFV-2** | Distribution percentages sum to ~1.0 | `abs(sum([high_pct, med_pct, low_pct]) - 1.0) < 0.01` | Log warning (non-critical) |
| **CFV-3** | High threshold ≥ Low threshold | `distribution.thresholds.high >= distribution.thresholds.low` | Reject with error message |
| **CFV-4** | Feature importance values sum to ~1.0 | `abs(sum(feature_importances) - 1.0) < 0.01` | Log warning (non-critical, may not sum exactly to 1.0) |
| **CFV-5** | Gap equals |top_avg - bottom_avg| | `abs(gap - abs(top_avg - bottom_avg)) < 0.001` | Log warning (rounding error) |

---

### 5.5 Edge Case Handling

**Special Cases and Validation Adjustments** (from MLAnalysisGenerationCHILD.md Section 2.3):

| Edge Case | Validation Adjustment | Rationale |
|-----------|----------------------|-----------|
| Feature not in aggregated CSV (derived feature) | Allow `distribution: None` | Cannot compute distribution for cross-window features not in raw data |
| All videos have same value (variance=0) | Allow `high_threshold == low_threshold` | Valid edge case - no spread in data |
| model_metrics.json missing window metrics | Allow `accuracy: None, precision: None, recall: None` | Non-critical - feature importance still valid |
| X_data is numpy array (no feature names) | Allow fallback to CSV header read | Ensures feature names always available |
| Cluster sizes don't sum exactly (rounding) | Reject (strict check) | Indicates cluster assignment bug - must be exact match |

---

## Section 6: Error Handling

**Source**: MLAnalysisGenerationCHILD.md Section 6 (Error Handling & Validation)

### 6.1 Error Taxonomy

| Error Category | Exit Code | Recovery Strategy | User Action |
|----------------|-----------|-------------------|-------------|
| **Pre-Flight Validation Failure** | 1 | Fail immediately before generating any JSONs | Re-run Stage 4 or Stage 5 to regenerate missing dependencies |
| **JSON Generation Failure** | 2 | Delete all temp files (atomic rollback) | Check error logs, verify model integrity, re-run Stage 6 |
| **Output Validation Failure** | 3 | Delete all temp files (atomic rollback) | Report bug - indicates code logic error in JSON generation |
| **Disk I/O Failure** | 4 | Delete all temp files (atomic rollback) | Check disk space, verify permissions, retry after fixing |

**Source**: MLAnalysisGenerationCHILD.md Section 6.2, FoundationCHILD.md Section 7

---

### 6.2 Error Cases

**Complete Error Handling Table** (from MLAnalysisGenerationCHILD.md Section 6.2):

| Error | Detection | Handling | User Message | Exit Code |
|-------|-----------|----------|--------------|-----------|
| **Missing Stage 4 CSV** | `os.path.exists()` check during pre-flight | Fail-fast before generating any JSONs | `"Stage 4 incomplete (N files missing): [list]. Action: Re-run Stage 4 (Feature Transformation)"` | 1 |
| **Missing Stage 5 model** | `os.path.exists()` check during pre-flight | Fail-fast before generating any JSONs | `"Stage 5 incomplete (N files missing): [list]. Action: Re-run Stage 5 (ML Model Training)"` | 1 |
| **Corrupted pickle file** | `joblib.load()` exception | Delete temp directory, fail | `"Failed to load model {path}: {error}. Model may be corrupted. Re-run Stage 5."` | 2 |
| **Invalid CSV format** | `pd.read_csv()` exception | Delete temp directory, fail | `"Failed to parse CSV {path}: {error}. Check file is valid CSV format."` | 2 |
| **Mid-generation crash** (partial JSONs created) | Exception caught in atomic pattern | Delete all temp files | `"Stage 6 JSON generation failed. Exception: {error}. Generated files before failure: {N} of {1+2×window_count} expected. Rolled back: Deleted all temp files (atomic failure)."` | 2 |
| **JSON validation fails** (invalid schema) | Post-generation validation | Delete all temp files | `"Output validation failed: {error}. All temp files deleted."` | 3 |
| **Disk full during JSON write** | `IOError` exception | Delete temp directory | `"Disk I/O failed: {error}. Check disk space."` | 4 |
| **Feature name mismatch** (K-Means vs RF) | Post-generation validation | Delete all temp files, fail | `"Feature name consistency check failed: K-Means JSON contains {count} features with '_scaled' suffixes. Bug in normalization logic."` | 3 |

---

### 6.3 Exception Handling Implementation

```python
# Source: MLAnalysisGenerationCHILD.md Section 2.3.5

def generate_ml_analysis_jsons(bucket_path: str, bucket: str, windows: list[str]) -> int:
    """
    Main function with comprehensive exception handling.

    See Section 4.5 for complete implementation.
    """
    temp_dir = os.path.join(bucket_path, 'ml_analysis/.tmp/')
    os.makedirs(temp_dir, exist_ok=True)
    generated_files = []

    try:
        # Step 1: Pre-flight validation (raises PreFlightValidationError)
        validate_stage_dependencies(bucket_path, bucket, windows)

        # Step 2-4: Generate JSONs, validate, commit (may raise various exceptions)
        # ... (see Section 4.5)

        return 0  # SUCCESS

    except PreFlightValidationError as e:
        logger.error(f"Pre-flight validation failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 1  # PRE-FLIGHT VALIDATION FAILED

    except json.JSONDecodeError as e:
        logger.error(f"Stage 6 JSON generation failed: Invalid JSON - {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning("Rolled back: Deleted all temp files (atomic failure)")
        return 3  # JSON VALIDATION FAILED

    except IOError as e:
        logger.error(f"Disk I/O failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 4  # DISK I/O FAILED

    except Exception as e:
        logger.error(f"Stage 6 JSON generation failed: {type(e).__name__}: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc(limit=10)}")
        logger.error(f"Generated files before failure: {len(generated_files)} of {1 + 2*len(windows)} expected")
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning("Rolled back: Deleted all temp files (atomic failure)")
        return 2  # JSON GENERATION FAILED
```

---

### 6.4 Custom Exception Classes

```python
# Source: MLAnalysisGenerationCHILD.md Section 6.1

class PreFlightValidationError(Exception):
    """
    Raised when Stage 4 or Stage 5 dependencies are missing.

    Exit Code: 1
    Recovery: Re-run Stage 4 or Stage 5
    """
    pass


class ValidationError(Exception):
    """
    Raised when output JSON validation fails.

    Exit Code: 3
    Recovery: Report bug - indicates code logic error
    """
    pass
```

---

### 6.5 Error Message Templates

**Pre-Flight Validation Error** (Exit Code 1):
```
Pre-flight validation failed:
Stage 4 incomplete (2 files missing):
  - ml_analysis/hook_rf_transformed.csv
  - ml_analysis/middle_1_km_transformed.csv
Action: Re-run Stage 4 (Feature Transformation)

Stage 5 incomplete (3 files missing):
  - models/rf_hook_18-33s.pkl
  - models/hook_kmeans_18-33s.pkl
  - models/hook_X_data_18-33s.pkl
Action: Re-run Stage 5 (ML Model Training)
```

**JSON Generation Error** (Exit Code 2):
```
Stage 6 JSON generation failed: FileNotFoundError: [Errno 2] No such file or directory: '/data/clients/acme/buckets/bucket_18-33s/models/rf_video_18-33s.pkl'
Stack trace:
  File "ml_analysis_generation.py", line 142, in generate_video_rf_json
    rf_model = joblib.load(model_path)
  File "/usr/local/lib/python3.9/site-packages/joblib/numpy_pickle.py", line 579, in load
    with open(filename, 'rb') as f:
Generated files before failure: 0 of 13 expected
Rolled back: Deleted all temp files (atomic failure)
```

**Output Validation Error** (Exit Code 3):
```
K-Means JSON hook_kmeans_analysis.json.tmp contains 3 features with transformation suffixes: ['eye_contact_rate_scaled', 'scene_count_scaled', 'word_count_scaled'].
Feature normalization failed. Check normalize_feature_name() logic.
```

**Disk I/O Error** (Exit Code 4):
```
Disk I/O failed: [Errno 28] No space left on device: '/data/clients/acme/buckets/bucket_18-33s/ml_analysis/.tmp/rf_video_analysis.json.tmp'
```

---

### 6.6 Rollback Procedures

**Atomic Rollback Pattern** (from MLAnalysisGenerationCHILD.md Section 2.3.5):

```python
def atomic_rollback(temp_dir: str, generated_files: list[str]) -> None:
    """
    Rollback procedure: Delete all temp files on failure.

    Steps:
    1. Delete all generated temp files
    2. Delete temp directory
    3. Log rollback action

    Ensures clean state for retry.

    Source: MLAnalysisGenerationCHILD.md Section 2.3.5
    """
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.warning(f"Rolled back: Deleted all temp files ({len(generated_files)} files) (atomic failure)")
    except Exception as e:
        logger.error(f"Rollback failed: {e}. Manual cleanup required for {temp_dir}")
```

**Rollback Scenarios**:

| Scenario | Files Generated | Rollback Action | Final State |
|----------|----------------|-----------------|-------------|
| Pre-flight validation fails | 0 | Delete empty temp directory | Clean (no partial output) |
| Crash during video RF JSON generation | 0 | Delete temp directory | Clean |
| Crash during window RF JSON generation | 3 (video RF + 2 window RF) | Delete all 3 temp files + directory | Clean |
| Validation fails after all JSONs generated | 13 (all expected files) | Delete all 13 temp files + directory | Clean |
| Disk full during 8th JSON write | 7 | Delete all 7 temp files + directory | Clean |

---

### 6.7 Error Recovery Workflows

**Workflow 1: Missing Dependencies (Exit Code 1)**
```
User runs Stage 6
    ↓
Pre-flight validation detects missing Stage 4 file
    ↓
Stage 6 exits with code 1
    ↓
User reads error message: "Stage 4 incomplete (1 file missing): ml_analysis/hook_rf_transformed.csv"
    ↓
User re-runs Stage 4 (Feature Transformation)
    ↓
User re-runs Stage 6 → SUCCESS (exit code 0)
```

**Workflow 2: Corrupted Model File (Exit Code 2)**
```
User runs Stage 6
    ↓
Pre-flight validation passes (all files exist)
    ↓
JSON generation attempts to load rf_video_18-33s.pkl
    ↓
joblib.load() raises UnpicklingError (corrupted file)
    ↓
Exception caught, atomic rollback deletes temp files
    ↓
Stage 6 exits with code 2
    ↓
User reads error message: "Failed to load model ... Model may be corrupted. Re-run Stage 5."
    ↓
User re-runs Stage 5 (ML Model Training)
    ↓
User re-runs Stage 6 → SUCCESS (exit code 0)
```

**Workflow 3: Feature Normalization Bug (Exit Code 3)**
```
User runs Stage 6
    ↓
Pre-flight validation passes
    ↓
All 13 JSONs generated to temp directory
    ↓
Output validation detects '_scaled' suffixes in K-Means centroids
    ↓
ValidationError raised, atomic rollback deletes all 13 temp files
    ↓
Stage 6 exits with code 3
    ↓
User reads error message: "Feature normalization failed. Check normalize_feature_name() logic."
    ↓
User reports bug to development team (code logic error)
    ↓
Developer fixes normalize_feature_name() function
    ↓
User re-runs Stage 6 → SUCCESS (exit code 0)
```

**Workflow 4: Disk Full (Exit Code 4)**
```
User runs Stage 6
    ↓
Pre-flight validation passes
    ↓
JSON generation writes 7 files successfully
    ↓
8th JSON write fails with "No space left on device"
    ↓
IOError caught, atomic rollback deletes 7 temp files
    ↓
Stage 6 exits with code 4
    ↓
User reads error message: "Disk I/O failed: [Errno 28] No space left on device"
    ↓
User checks disk space: df -h /data
    ↓
User clears disk space or expands storage
    ↓
User re-runs Stage 6 → SUCCESS (exit code 0)
```

---

### 6.8 Logging During Error Handling

**Error Logging Requirements** (see Section 10 for complete logging spec):

```python
# Pre-flight validation error
logger.error(f"Pre-flight validation failed: {error_message}")
# Example: "Pre-flight validation failed: Stage 4 incomplete (2 files missing): ..."

# JSON generation error with stack trace
logger.error(f"Stage 6 JSON generation failed: {type(e).__name__}: {str(e)}")
logger.error(f"Stack trace: {traceback.format_exc(limit=10)}")
logger.error(f"Generated files before failure: {len(generated_files)} of {1 + 2*len(windows)} expected")
# Example: "Stage 6 JSON generation failed: FileNotFoundError: [Errno 2] No such file or directory: ..."

# Rollback action
logger.warning("Rolled back: Deleted all temp files (atomic failure)")
# Example: "Rolled back: Deleted all temp files (8 files) (atomic failure)"

# Output validation error
logger.error(f"Output validation failed: {validation_error}")
# Example: "Output validation failed: K-Means JSON hook_kmeans_analysis.json.tmp contains 3 features with transformation suffixes..."
```

**Log Level Mapping**:

| Error Category | Log Level | Rationale |
|----------------|-----------|-----------|
| Pre-flight validation failure | ERROR | User-fixable - missing dependencies |
| JSON generation failure | ERROR | May indicate bug or corrupted data |
| Output validation failure | ERROR | Indicates code logic error (bug) |
| Disk I/O failure | ERROR | Infrastructure issue |
| Atomic rollback action | WARNING | Recovery action taken |
| Non-critical validation warnings (e.g., distribution percentages don't sum exactly to 1.0) | WARNING | Non-blocking issue |

---

## Section 7: Complete Example Traces

**Source**: MLAnalysisGenerationCHILD.md Sections 2.2 (Data Flow), 7 (Performance), Appendix B (Example Data)

### 7.0 Parameterized Trace Template (Bucket-Agnostic)

**Purpose**: This template applies to ALL buckets - substitute actual values based on bucket window count.

**Variables by Bucket**:
```python
# Bucket configuration (from config/bucket_definitions.py)
BUCKET_WINDOWS = {
    '0-3s':    {'windows': ['hook'], 'window_count': 1, 'json_count': 3},
    '3-9s':    {'windows': ['hook', 'closing'], 'window_count': 2, 'json_count': 5},
    '9-13s':   {'windows': ['hook', 'middle_aggregate', 'closing'], 'window_count': 3, 'json_count': 7},
    '13-18s':  {'windows': ['hook', 'middle_aggregate', 'closing'], 'window_count': 3, 'json_count': 7},
    '18-33s':  {'windows': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing'], 'window_count': 6, 'json_count': 13},
    '33-60s':  {'windows': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'], 'window_count': 7, 'json_count': 15},
    '60-90s':  {'windows': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'], 'window_count': 7, 'json_count': 15},
    '90-120s': {'windows': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'], 'window_count': 7, 'json_count': 15},
}

# Formula: json_count = 1 + (window_count × 2)
```

**Execution Flow** (applies to all buckets):
```
[TIMESTAMP] INFO: Starting Stage 6: ML Analysis Generation for bucket {bucket}
[TIMESTAMP] INFO: Pre-flight validation: checking Stage 4 and Stage 5 outputs...
[TIMESTAMP] INFO: ✓ Pre-flight validation passed: All {input_file_count} files exist
              (input_file_count = 4 + 6×window_count)

[TIMESTAMP] INFO: Generating ML analysis JSONs to temp directory...
[TIMESTAMP] INFO:   ✓ Generated rf_video_analysis.json.tmp
[TIMESTAMP] INFO:   ✓ Generated {window[0]}_rf_analysis.json.tmp
[TIMESTAMP] INFO:   ✓ Generated {window[1]}_rf_analysis.json.tmp
... (window_count RF JSONs)
[TIMESTAMP] INFO:   ✓ Generated {window[0]}_kmeans_analysis.json.tmp
[TIMESTAMP] INFO:   ✓ Generated {window[1]}_kmeans_analysis.json.tmp
... (window_count K-Means JSONs)

[TIMESTAMP] INFO: ✓ All {json_count} JSONs generated to temp directory
              (json_count = 1 + 2×window_count)

[TIMESTAMP] INFO: Validating JSON schemas...
[TIMESTAMP] DEBUG:   ✓ All {json_count} temp files exist
[TIMESTAMP] DEBUG:   ✓ All {json_count} JSONs are valid JSON format
[TIMESTAMP] DEBUG:   ✓ All {window_count} K-Means JSONs have normalized feature names
[TIMESTAMP] DEBUG:   ✓ All cluster sizes sum correctly
[TIMESTAMP] INFO: ✓ All JSON schemas valid

[TIMESTAMP] INFO: Committing JSONs (atomic rename)...
[TIMESTAMP] INFO:   ✓ Committed rf_video_analysis.json
... ({json_count} total commits)

[TIMESTAMP] INFO: ✓ Stage 6 complete: {json_count} JSONs generated

Exit Code: 0
```

**Performance Metrics** (vary by bucket and video count):
- Small buckets (0-3s, 1 window): 1-2 seconds for 100 videos
- Medium buckets (18-33s, 6 windows): 3-5 seconds for 100 videos
- Large buckets (90-120s, 7 windows): 4-6 seconds for 100 videos

---

### 7.1 Concrete Example: Bucket "18-33s" Success Case (Exit Code 0)

**Scenario**: Generate all ML analysis JSONs for bucket "18-33s" with 100 videos (6 windows: hook, middle_1-4, closing)

**Input State**:
```
/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/
├── ml_analysis/
│   ├── aggregated_features.csv (100 rows × 129 columns, ~15 MB)
│   ├── rf_transformed.csv (100 rows × 178 columns)
│   ├── hook_rf_transformed.csv (100 rows × 21 columns)
│   ├── middle_1_rf_transformed.csv (100 rows × 21 columns)
│   ├── middle_2_rf_transformed.csv (100 rows × 21 columns)
│   ├── middle_3_rf_transformed.csv (100 rows × 21 columns)
│   ├── middle_4_rf_transformed.csv (100 rows × 21 columns)
│   ├── closing_rf_transformed.csv (100 rows × 21 columns)
│   ├── hook_km_transformed.csv (100 rows × 21 columns)
│   ├── middle_1_km_transformed.csv (100 rows × 21 columns)
│   ├── middle_2_km_transformed.csv (100 rows × 21 columns)
│   ├── middle_3_km_transformed.csv (100 rows × 21 columns)
│   ├── middle_4_km_transformed.csv (100 rows × 21 columns)
│   └── closing_km_transformed.csv (100 rows × 21 columns)
└── models/
    ├── rf_video_18-33s.pkl (RandomForestClassifier, 178 features)
    ├── rf_hook_18-33s.pkl (RandomForestClassifier, 21 features)
    ├── rf_middle_1_18-33s.pkl (RandomForestClassifier, 21 features)
    ├── rf_middle_2_18-33s.pkl (RandomForestClassifier, 21 features)
    ├── rf_middle_3_18-33s.pkl (RandomForestClassifier, 21 features)
    ├── rf_middle_4_18-33s.pkl (RandomForestClassifier, 21 features)
    ├── rf_closing_18-33s.pkl (RandomForestClassifier, 21 features)
    ├── hook_kmeans_18-33s.pkl (KMeans, n_clusters=3)
    ├── middle_1_kmeans_18-33s.pkl (KMeans, n_clusters=3)
    ├── middle_2_kmeans_18-33s.pkl (KMeans, n_clusters=3)
    ├── middle_3_kmeans_18-33s.pkl (KMeans, n_clusters=3)
    ├── middle_4_kmeans_18-33s.pkl (KMeans, n_clusters=3)
    ├── closing_kmeans_18-33s.pkl (KMeans, n_clusters=3)
    ├── hook_scalers_18-33s.pkl (MinMaxScaler)
    ├── middle_1_scalers_18-33s.pkl (MinMaxScaler)
    ├── middle_2_scalers_18-33s.pkl (MinMaxScaler)
    ├── middle_3_scalers_18-33s.pkl (MinMaxScaler)
    ├── middle_4_scalers_18-33s.pkl (MinMaxScaler)
    ├── closing_scalers_18-33s.pkl (MinMaxScaler)
    ├── hook_X_data_18-33s.pkl (DataFrame, 100 rows × 21 columns)
    ├── middle_1_X_data_18-33s.pkl (DataFrame, 100 rows × 21 columns)
    ├── middle_2_X_data_18-33s.pkl (DataFrame, 100 rows × 21 columns)
    ├── middle_3_X_data_18-33s.pkl (DataFrame, 100 rows × 21 columns)
    ├── middle_4_X_data_18-33s.pkl (DataFrame, 100 rows × 21 columns)
    ├── closing_X_data_18-33s.pkl (DataFrame, 100 rows × 21 columns)
    └── model_metrics.json
```

**Execution Trace**:

```
[2025-01-28 14:30:00] INFO: Starting Stage 6: ML Analysis Generation for bucket 18-33s
[2025-01-28 14:30:00] INFO: Bucket 18-33s has 6 windows: ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

# ===== Step 1: Pre-flight validation (0.8s) =====
[2025-01-28 14:30:00] INFO: Pre-flight validation: checking Stage 4 and Stage 5 outputs...
[2025-01-28 14:30:00] DEBUG: Checking 13 Stage 4 CSV files...
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/aggregated_features.csv exists (15.2 MB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/rf_transformed.csv exists (1.8 MB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/hook_rf_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/middle_1_rf_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/middle_2_rf_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/middle_3_rf_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/middle_4_rf_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/closing_rf_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/hook_km_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/middle_1_km_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/middle_2_km_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/middle_3_km_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/middle_4_km_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ ml_analysis/closing_km_transformed.csv exists (200 KB)
[2025-01-28 14:30:00] DEBUG: Checking 20 Stage 5 model files...
[2025-01-28 14:30:00] DEBUG:   ✓ models/rf_video_18-33s.pkl exists (450 KB)
[2025-01-28 14:30:00] DEBUG:   ✓ models/rf_hook_18-33s.pkl exists (50 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/rf_middle_1_18-33s.pkl exists (50 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/rf_middle_2_18-33s.pkl exists (50 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/rf_middle_3_18-33s.pkl exists (50 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/rf_middle_4_18-33s.pkl exists (50 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/rf_closing_18-33s.pkl exists (50 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/hook_kmeans_18-33s.pkl exists (25 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_1_kmeans_18-33s.pkl exists (25 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_2_kmeans_18-33s.pkl exists (25 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_3_kmeans_18-33s.pkl exists (25 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_4_kmeans_18-33s.pkl exists (25 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/closing_kmeans_18-33s.pkl exists (25 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/hook_scalers_18-33s.pkl exists (10 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_1_scalers_18-33s.pkl exists (10 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_2_scalers_18-33s.pkl exists (10 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_3_scalers_18-33s.pkl exists (10 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_4_scalers_18-33s.pkl exists (10 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/closing_scalers_18-33s.pkl exists (10 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/hook_X_data_18-33s.pkl exists (80 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_1_X_data_18-33s.pkl exists (80 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_2_X_data_18-33s.pkl exists (80 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_3_X_data_18-33s.pkl exists (80 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/middle_4_X_data_18-33s.pkl exists (80 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/closing_X_data_18-33s.pkl exists (80 KB)
[2025-01-28 14:30:01] DEBUG:   ✓ models/model_metrics.json exists (5 KB)
[2025-01-28 14:30:01] INFO: ✓ Pre-flight validation passed: All 13 Stage 4 files + 20 Stage 5 files exist

# ===== Step 2: Generate Video-Level RF JSON (1.2s) =====
[2025-01-28 14:30:01] INFO: Generating ML analysis JSONs to temp directory...
[2025-01-28 14:30:01] DEBUG: Loading video RF model: models/rf_video_18-33s.pkl
[2025-01-28 14:30:01] DEBUG: Extracting feature importance (178 features)
[2025-01-28 14:30:01] DEBUG: Top 10 features: ['hook_eye_contact_rate' (0.22), 'middle_avg_word_count' (0.18), ...]
[2025-01-28 14:30:01] DEBUG: Loading aggregated_features.csv for distribution analysis (100 rows × 129 columns)
[2025-01-28 14:30:02] DEBUG: Computing distribution stats for top 10 features...
[2025-01-28 14:30:02] DEBUG:   Feature 1/10: hook_eye_contact_rate (top_avg=0.88, bottom_avg=0.45, gap=0.43)
[2025-01-28 14:30:02] DEBUG:   Feature 2/10: middle_avg_word_count (top_avg=55.2, bottom_avg=28.4, gap=26.8)
[2025-01-28 14:30:02] DEBUG:   Feature 3/10: closing_energy_level (top_avg=0.75, bottom_avg=0.42, gap=0.33)
[2025-01-28 14:30:02] DEBUG:   Feature 4/10: hook_word_count (top_avg=14.2, bottom_avg=22.5, gap=8.3)
[2025-01-28 14:30:02] DEBUG:   Feature 5/10: middle_1_scene_count (top_avg=5.1, bottom_avg=2.8, gap=2.3)
[2025-01-28 14:30:02] DEBUG:   Feature 6/10: closing_word_count (top_avg=18.5, bottom_avg=12.3, gap=6.2)
[2025-01-28 14:30:02] DEBUG:   Feature 7/10: hook_energy_level (top_avg=0.55, bottom_avg=0.35, gap=0.20)
[2025-01-28 14:30:02] DEBUG:   Feature 8/10: middle_2_eye_contact_rate (top_avg=0.72, bottom_avg=0.48, gap=0.24)
[2025-01-28 14:30:02] DEBUG:   Feature 9/10: middle_avg_energy_level (top_avg=0.68, bottom_avg=0.42, gap=0.26)
[2025-01-28 14:30:02] DEBUG:   Feature 10/10: closing_scene_count (top_avg=4.2, bottom_avg=2.1, gap=2.1)
[2025-01-28 14:30:02] DEBUG: Writing rf_video_analysis.json.tmp (30 KB)
[2025-01-28 14:30:02] INFO:   ✓ Generated rf_video_analysis.json.tmp

# ===== Step 3: Generate Window-Level RF JSONs (6 files, 0.8s each = 4.8s total) =====
[2025-01-28 14:30:02] DEBUG: Loading window RF model: models/rf_hook_18-33s.pkl
[2025-01-28 14:30:02] DEBUG: Extracting feature importance (21 features)
[2025-01-28 14:30:02] DEBUG: Loading model_metrics.json for performance stats
[2025-01-28 14:30:02] DEBUG: Loading hook_rf_transformed.csv for distribution analysis (100 rows × 21 columns)
[2025-01-28 14:30:03] DEBUG: Computing distribution stats for top 10 features...
[2025-01-28 14:30:03] DEBUG: Writing hook_rf_analysis.json.tmp (5 KB)
[2025-01-28 14:30:03] INFO:   ✓ Generated hook_rf_analysis.json.tmp

[2025-01-28 14:30:03] DEBUG: Loading window RF model: models/rf_middle_1_18-33s.pkl
[2025-01-28 14:30:03] DEBUG: Extracting feature importance (21 features)
[2025-01-28 14:30:03] DEBUG: Writing middle_1_rf_analysis.json.tmp (5 KB)
[2025-01-28 14:30:03] INFO:   ✓ Generated middle_1_rf_analysis.json.tmp

[2025-01-28 14:30:04] DEBUG: Loading window RF model: models/rf_middle_2_18-33s.pkl
[2025-01-28 14:30:04] INFO:   ✓ Generated middle_2_rf_analysis.json.tmp

[2025-01-28 14:30:04] DEBUG: Loading window RF model: models/rf_middle_3_18-33s.pkl
[2025-01-28 14:30:05] INFO:   ✓ Generated middle_3_rf_analysis.json.tmp

[2025-01-28 14:30:05] DEBUG: Loading window RF model: models/rf_middle_4_18-33s.pkl
[2025-01-28 14:30:06] INFO:   ✓ Generated middle_4_rf_analysis.json.tmp

[2025-01-28 14:30:06] DEBUG: Loading window RF model: models/rf_closing_18-33s.pkl
[2025-01-28 14:30:07] INFO:   ✓ Generated closing_rf_analysis.json.tmp

# ===== Step 4: Generate Window-Level K-Means JSONs (6 files, 0.5s each = 3.0s total) =====
[2025-01-28 14:30:07] DEBUG: Loading K-Means model: models/hook_kmeans_18-33s.pkl
[2025-01-28 14:30:07] DEBUG: Extracting cluster centroids (3 clusters × 21 features)
[2025-01-28 14:30:07] DEBUG: Loading X_data: models/hook_X_data_18-33s.pkl (DataFrame with 21 feature names)
[2025-01-28 14:30:07] DEBUG: Loading hook_km_transformed.csv for cluster assignments (100 rows)
[2025-01-28 14:30:07] DEBUG: Predicting cluster labels...
[2025-01-28 14:30:07] DEBUG: Cluster 0: 35 videos, Cluster 1: 42 videos, Cluster 2: 23 videos
[2025-01-28 14:30:07] DEBUG: Normalizing feature names (removing _scaled suffixes)...
[2025-01-28 14:30:07] DEBUG:   'eye_contact_rate_scaled' → 'eye_contact_rate'
[2025-01-28 14:30:07] DEBUG:   'scene_count_scaled' → 'scene_count'
[2025-01-28 14:30:07] DEBUG:   'word_count_scaled' → 'word_count'
[2025-01-28 14:30:07] DEBUG:   ... (18 more features)
[2025-01-28 14:30:07] DEBUG: Computing distances to centroid for 35 videos in cluster 0...
[2025-01-28 14:30:07] DEBUG: Writing hook_kmeans_analysis.json.tmp (5 KB)
[2025-01-28 14:30:07] INFO:   ✓ Generated hook_kmeans_analysis.json.tmp

[2025-01-28 14:30:08] DEBUG: Loading K-Means model: models/middle_1_kmeans_18-33s.pkl
[2025-01-28 14:30:08] INFO:   ✓ Generated middle_1_kmeans_analysis.json.tmp

[2025-01-28 14:30:08] DEBUG: Loading K-Means model: models/middle_2_kmeans_18-33s.pkl
[2025-01-28 14:30:09] INFO:   ✓ Generated middle_2_kmeans_analysis.json.tmp

[2025-01-28 14:30:09] DEBUG: Loading K-Means model: models/middle_3_kmeans_18-33s.pkl
[2025-01-28 14:30:09] INFO:   ✓ Generated middle_3_kmeans_analysis.json.tmp

[2025-01-28 14:30:10] DEBUG: Loading K-Means model: models/middle_4_kmeans_18-33s.pkl
[2025-01-28 14:30:10] INFO:   ✓ Generated middle_4_kmeans_analysis.json.tmp

[2025-01-28 14:30:10] DEBUG: Loading K-Means model: models/closing_kmeans_18-33s.pkl
[2025-01-28 14:30:11] INFO:   ✓ Generated closing_kmeans_analysis.json.tmp

[2025-01-28 14:30:11] INFO: ✓ All 13 JSONs generated to temp directory

# ===== Step 5: Validate All JSON Schemas (0.3s) =====
[2025-01-28 14:30:11] INFO: Validating JSON schemas...
[2025-01-28 14:30:11] DEBUG: Checking all 13 expected files exist in .tmp/...
[2025-01-28 14:30:11] DEBUG:   ✓ rf_video_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ hook_rf_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ middle_1_rf_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ middle_2_rf_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ middle_3_rf_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ middle_4_rf_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ closing_rf_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ hook_kmeans_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ middle_1_kmeans_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ middle_2_kmeans_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ middle_3_kmeans_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ middle_4_kmeans_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG:   ✓ closing_kmeans_analysis.json.tmp exists
[2025-01-28 14:30:11] DEBUG: Validating JSON parseability...
[2025-01-28 14:30:11] DEBUG:   ✓ All 13 JSONs are valid JSON format
[2025-01-28 14:30:11] DEBUG: Validating K-Means feature names (no _scaled suffixes)...
[2025-01-28 14:30:11] DEBUG:   ✓ hook_kmeans_analysis.json.tmp: All feature names normalized
[2025-01-28 14:30:11] DEBUG:   ✓ middle_1_kmeans_analysis.json.tmp: All feature names normalized
[2025-01-28 14:30:11] DEBUG:   ✓ middle_2_kmeans_analysis.json.tmp: All feature names normalized
[2025-01-28 14:30:11] DEBUG:   ✓ middle_3_kmeans_analysis.json.tmp: All feature names normalized
[2025-01-28 14:30:11] DEBUG:   ✓ middle_4_kmeans_analysis.json.tmp: All feature names normalized
[2025-01-28 14:30:11] DEBUG:   ✓ closing_kmeans_analysis.json.tmp: All feature names normalized
[2025-01-28 14:30:11] DEBUG: Validating cluster size consistency...
[2025-01-28 14:30:11] DEBUG:   ✓ hook_kmeans: sum([35, 42, 23]) = 100 = total_videos
[2025-01-28 14:30:11] DEBUG:   ✓ middle_1_kmeans: sum([38, 40, 22]) = 100 = total_videos
[2025-01-28 14:30:11] DEBUG:   ✓ middle_2_kmeans: sum([33, 45, 22]) = 100 = total_videos
[2025-01-28 14:30:11] DEBUG:   ✓ middle_3_kmeans: sum([36, 42, 22]) = 100 = total_videos
[2025-01-28 14:30:11] DEBUG:   ✓ middle_4_kmeans: sum([34, 43, 23]) = 100 = total_videos
[2025-01-28 14:30:11] DEBUG:   ✓ closing_kmeans: sum([37, 41, 22]) = 100 = total_videos
[2025-01-28 14:30:11] INFO: ✓ All JSON schemas valid

# ===== Step 6: Atomic Commit (0.2s) =====
[2025-01-28 14:30:11] INFO: Committing JSONs (atomic rename)...
[2025-01-28 14:30:11] DEBUG: Renaming rf_video_analysis.json.tmp → rf_video_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed rf_video_analysis.json
[2025-01-28 14:30:11] DEBUG: Renaming hook_rf_analysis.json.tmp → hook_rf_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed hook_rf_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed middle_1_rf_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed middle_2_rf_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed middle_3_rf_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed middle_4_rf_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed closing_rf_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed hook_kmeans_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed middle_1_kmeans_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed middle_2_kmeans_analysis.json
[2025-01-28 14:30:11] INFO:   ✓ Committed middle_3_kmeans_analysis.json
[2025-01-28 14:30:12] INFO:   ✓ Committed middle_4_kmeans_analysis.json
[2025-01-28 14:30:12] INFO:   ✓ Committed closing_kmeans_analysis.json

# ===== Step 7: Cleanup (0.1s) =====
[2025-01-28 14:30:12] DEBUG: Deleting temp directory: ml_analysis/.tmp/
[2025-01-28 14:30:12] INFO: ✓ Stage 6 complete: 13 JSONs generated

# ===== Final Summary =====
[2025-01-28 14:30:12] INFO: ========================================
[2025-01-28 14:30:12] INFO: Stage 6: ML Analysis Generation COMPLETE
[2025-01-28 14:30:12] INFO: Bucket: 18-33s
[2025-01-28 14:30:12] INFO: Total videos: 100
[2025-01-28 14:30:12] INFO: JSONs generated: 13 (1 video RF + 6 window RF + 6 window K-Means)
[2025-01-28 14:30:12] INFO: Total time: 11.2s
[2025-01-28 14:30:12] INFO: Exit code: 0 (SUCCESS)
[2025-01-28 14:30:12] INFO: ========================================
```

**Output State**:
```
/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/
└── ml_analysis/
    ├── rf_video_analysis.json (30 KB)
    ├── hook_rf_analysis.json (5 KB)
    ├── middle_1_rf_analysis.json (5 KB)
    ├── middle_2_rf_analysis.json (5 KB)
    ├── middle_3_rf_analysis.json (5 KB)
    ├── middle_4_rf_analysis.json (5 KB)
    ├── closing_rf_analysis.json (5 KB)
    ├── hook_kmeans_analysis.json (5 KB)
    ├── middle_1_kmeans_analysis.json (5 KB)
    ├── middle_2_kmeans_analysis.json (5 KB)
    ├── middle_3_kmeans_analysis.json (5 KB)
    ├── middle_4_kmeans_analysis.json (5 KB)
    └── closing_kmeans_analysis.json (5 KB)

Total: 13 JSONs, ~95 KB
```

**Performance Breakdown**:
```
Pre-flight validation:    0.8s  ( 7%)
Video RF JSON:            1.2s  (11%)
Window RF JSONs (6):      4.8s  (43%)
Window K-Means JSONs (6): 3.0s  (27%)
Output validation:        0.3s  ( 3%)
Atomic commit:            0.2s  ( 2%)
Cleanup:                  0.1s  ( 1%)
Other (overhead):         0.8s  ( 7%)
─────────────────────────────────
Total:                   11.2s (100%)
```

---

### 7.2 Failure Case 1: Missing Dependencies (Exit Code 1)

**Scenario**: User runs Stage 6 but Stage 4 hook_rf_transformed.csv is missing

**Input State**:
```
/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/
├── ml_analysis/
│   ├── aggregated_features.csv (100 rows × 129 columns)
│   ├── rf_transformed.csv (100 rows × 178 columns)
│   ├── hook_rf_transformed.csv  ← MISSING
│   ├── middle_1_rf_transformed.csv (100 rows × 21 columns)
│   └── ... (other CSVs present)
└── models/
    └── ... (all 20 model files present)
```

**Execution Trace**:

```
[2025-01-28 14:35:00] INFO: Starting Stage 6: ML Analysis Generation for bucket 18-33s
[2025-01-28 14:35:00] INFO: Bucket 18-33s has 6 windows: ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

# ===== Pre-flight validation detects missing file =====
[2025-01-28 14:35:00] INFO: Pre-flight validation: checking Stage 4 and Stage 5 outputs...
[2025-01-28 14:35:00] DEBUG: Checking 13 Stage 4 CSV files...
[2025-01-28 14:35:00] DEBUG:   ✓ ml_analysis/aggregated_features.csv exists (15.2 MB)
[2025-01-28 14:35:00] DEBUG:   ✓ ml_analysis/rf_transformed.csv exists (1.8 MB)
[2025-01-28 14:35:00] DEBUG:   ✗ ml_analysis/hook_rf_transformed.csv MISSING
[2025-01-28 14:35:00] DEBUG:   ✓ ml_analysis/middle_1_rf_transformed.csv exists (200 KB)
[2025-01-28 14:35:00] DEBUG:   ... (continuing checks)
[2025-01-28 14:35:00] ERROR: Pre-flight validation failed:
Stage 4 incomplete (1 file missing):
  - ml_analysis/hook_rf_transformed.csv
Action: Re-run Stage 4 (Feature Transformation)

[2025-01-28 14:35:00] INFO: ========================================
[2025-01-28 14:35:00] INFO: Stage 6: ML Analysis Generation FAILED
[2025-01-28 14:35:00] INFO: Exit code: 1 (PRE-FLIGHT VALIDATION FAILED)
[2025-01-28 14:35:00] INFO: ========================================
```

**Output State**: No files generated (clean state)

**User Action**: Re-run Stage 4 (Feature Transformation) to regenerate missing file, then retry Stage 6

---

### 7.3 Failure Case 2: Corrupted Model File (Exit Code 2)

**Scenario**: rf_video_18-33s.pkl is corrupted (cannot be loaded)

**Execution Trace**:

```
[2025-01-28 14:40:00] INFO: Starting Stage 6: ML Analysis Generation for bucket 18-33s
[2025-01-28 14:40:01] INFO: ✓ Pre-flight validation passed: All 13 Stage 4 files + 20 Stage 5 files exist

# ===== JSON generation fails on corrupted model =====
[2025-01-28 14:40:01] INFO: Generating ML analysis JSONs to temp directory...
[2025-01-28 14:40:01] DEBUG: Loading video RF model: models/rf_video_18-33s.pkl
[2025-01-28 14:40:01] ERROR: Stage 6 JSON generation failed: UnpicklingError: invalid load key, '\x00'
[2025-01-28 14:40:01] ERROR: Stack trace:
  File "ml_analysis_generation.py", line 142, in generate_video_rf_json
    rf_model = joblib.load(model_path)
  File "/usr/local/lib/python3.9/site-packages/joblib/numpy_pickle.py", line 579, in load
    with open(filename, 'rb') as f:
  File "/usr/local/lib/python3.9/site-packages/joblib/numpy_pickle.py", line 358, in _unpickle
    obj = unpickler.load()
[2025-01-28 14:40:01] ERROR: Generated files before failure: 0 of 13 expected
[2025-01-28 14:40:01] WARNING: Rolled back: Deleted all temp files (atomic failure)

[2025-01-28 14:40:01] INFO: ========================================
[2025-01-28 14:40:01] INFO: Stage 6: ML Analysis Generation FAILED
[2025-01-28 14:40:01] INFO: Exit code: 2 (JSON GENERATION FAILED)
[2025-01-28 14:40:01] INFO: ========================================
```

**Output State**: Temp directory deleted (clean state)

**User Action**: Re-run Stage 5 (ML Model Training) to regenerate corrupted model, then retry Stage 6

---

### 7.4 Failure Case 3: Feature Normalization Bug (Exit Code 3)

**Scenario**: normalize_feature_name() function has a bug and doesn't remove _scaled suffixes

**Execution Trace**:

```
[2025-01-28 14:45:00] INFO: Starting Stage 6: ML Analysis Generation for bucket 18-33s
[2025-01-28 14:45:01] INFO: ✓ Pre-flight validation passed
[2025-01-28 14:45:11] INFO: ✓ All 13 JSONs generated to temp directory

# ===== Validation detects unnormalized feature names =====
[2025-01-28 14:45:11] INFO: Validating JSON schemas...
[2025-01-28 14:45:11] DEBUG: Validating K-Means feature names (no _scaled suffixes)...
[2025-01-28 14:45:11] DEBUG:   ✗ hook_kmeans_analysis.json.tmp: Found 21 features with suffixes
[2025-01-28 14:45:11] ERROR: K-Means JSON hook_kmeans_analysis.json.tmp contains 21 features with transformation suffixes: ['eye_contact_rate_scaled', 'scene_count_scaled', 'word_count_scaled', 'energy_level_scaled', 'has_captions_encoded'].
Feature normalization failed. Check normalize_feature_name() logic.
[2025-01-28 14:45:11] WARNING: Rolled back: Deleted all temp files (13 files) (atomic failure)

[2025-01-28 14:45:11] INFO: ========================================
[2025-01-28 14:45:11] INFO: Stage 6: ML Analysis Generation FAILED
[2025-01-28 14:45:11] INFO: Exit code: 3 (OUTPUT VALIDATION FAILED)
[2025-01-28 14:45:11] INFO: ========================================
```

**Output State**: All 13 temp files deleted (clean state)

**User Action**: Report bug to development team (code logic error in normalize_feature_name())

---

### 7.5 Failure Case 4: Disk Full (Exit Code 4)

**Scenario**: Disk runs out of space while writing 8th JSON file

**Execution Trace**:

```
[2025-01-28 14:50:00] INFO: Starting Stage 6: ML Analysis Generation for bucket 18-33s
[2025-01-28 14:50:01] INFO: ✓ Pre-flight validation passed
[2025-01-28 14:50:02] INFO:   ✓ Generated rf_video_analysis.json.tmp
[2025-01-28 14:50:03] INFO:   ✓ Generated hook_rf_analysis.json.tmp
[2025-01-28 14:50:04] INFO:   ✓ Generated middle_1_rf_analysis.json.tmp
[2025-01-28 14:50:05] INFO:   ✓ Generated middle_2_rf_analysis.json.tmp
[2025-01-28 14:50:06] INFO:   ✓ Generated middle_3_rf_analysis.json.tmp
[2025-01-28 14:50:07] INFO:   ✓ Generated middle_4_rf_analysis.json.tmp
[2025-01-28 14:50:08] INFO:   ✓ Generated closing_rf_analysis.json.tmp

# ===== Disk full during 8th file write =====
[2025-01-28 14:50:08] DEBUG: Loading K-Means model: models/hook_kmeans_18-33s.pkl
[2025-01-28 14:50:08] DEBUG: Writing hook_kmeans_analysis.json.tmp (5 KB)
[2025-01-28 14:50:08] ERROR: Disk I/O failed: [Errno 28] No space left on device: '/data/clients/acme/buckets/bucket_18-33s/ml_analysis/.tmp/hook_kmeans_analysis.json.tmp'
[2025-01-28 14:50:08] DEBUG: Deleting temp directory (atomic rollback)...
[2025-01-28 14:50:08] WARNING: Rolled back: Deleted all temp files (7 files) (atomic failure)

[2025-01-28 14:50:08] INFO: ========================================
[2025-01-28 14:50:08] INFO: Stage 6: ML Analysis Generation FAILED
[2025-01-28 14:50:08] INFO: Exit code: 4 (DISK I/O FAILED)
[2025-01-28 14:50:08] INFO: ========================================
```

**Output State**: 7 temp files deleted (clean state)

**User Action**: Check disk space (`df -h /data`), clear space or expand storage, then retry Stage 6

---

## Section 8: File Structure & Integration

**Source**: FoundationCHILD.md Section 2 (Client Architecture & Storage)

### 8.1 Module Structure

```
rumiai/
├── config/
│   ├── __init__.py
│   ├── bucket_definitions.py          # BUCKET_WINDOWS configuration
│   └── settings.py                     # General configuration
│
├── ml_pipeline/
│   ├── __init__.py
│   ├── stage3_feature_aggregation.py   # Stage 3 (upstream dependency)
│   ├── stage4_feature_transformation.py # Stage 4 (upstream dependency)
│   ├── stage5_model_training.py        # Stage 5 (upstream dependency)
│   ├── stage6_ml_analysis_generation.py # THIS MODULE
│   └── stage7_llm_report_generation.py # Stage 7 (downstream consumer)
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                       # Centralized logging
│   ├── path_utils.py                   # Directory path helpers
│   └── validation.py                   # Common validation functions
│
└── main.py                             # CLI entry point
```

---

### 8.2 Module: stage6_ml_analysis_generation.py

```python
"""
Stage 6: ML Analysis Generation

Extracts insights from trained ML models (Stage 5) and generates structured JSON files
for LLM consumption (Stage 7).

Outputs:
- 1 video-level RF JSON (cross-window feature importance)
- N window-level RF JSONs (per-window feature importance)
- N window-level K-Means JSONs (cluster centroids with normalized feature names)

Where N = number of windows per bucket (6-7 depending on bucket duration)

Source: MLAnalysisGenerationCHILD.md
"""

import os
import sys
import json
import shutil
import logging
import traceback
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import joblib

# Internal imports
from config.bucket_definitions import BUCKET_WINDOWS
from utils.logger import setup_logger
from utils.path_utils import get_bucket_path, ensure_directory_exists
from utils.validation import validate_json_schema


# ===== Module-level Configuration =====
logger = setup_logger(__name__)

# Distribution analysis parameters (from MLAnalysisGenerationCHILD.md Section 4.2)
TOP_PERFORMER_PERCENTAGE = 0.8  # Top 80% vs bottom 20%
HIGH_PERCENTILE = 0.66          # 66th percentile threshold
LOW_PERCENTILE = 0.33           # 33rd percentile threshold

# Feature importance limits
MAX_FEATURES_VIDEO_RF = 10      # Top 10 features for video-level RF
MAX_FEATURES_WINDOW_RF = 10     # Top 10 features for window-level RF

# K-Means parameters
N_CLUSTERS = 3                  # Always 3 clusters per window


# ===== Custom Exceptions =====
class PreFlightValidationError(Exception):
    """Raised when Stage 4 or Stage 5 dependencies are missing."""
    pass


class ValidationError(Exception):
    """Raised when output JSON validation fails."""
    pass


# ===== Core Functions =====
# (See Section 4 for complete implementations)

def validate_stage_dependencies(bucket_path: str, bucket: str, windows: List[str]) -> None:
    """See Section 4.1 for complete implementation."""
    pass


def generate_video_rf_json(bucket_path: str, bucket: str) -> dict:
    """See Section 4.2 for complete implementation."""
    pass


def generate_window_rf_json(bucket_path: str, bucket: str, window: str) -> dict:
    """See Section 4.3 for complete implementation."""
    pass


def normalize_feature_name(feature_name: str) -> str:
    """See Section 4.4 for complete implementation."""
    pass


def generate_window_kmeans_json(bucket_path: str, bucket: str, window: str) -> dict:
    """See Section 4.4 for complete implementation."""
    pass


def validate_all_json_schemas(temp_dir: str, bucket: str, windows: List[str]) -> None:
    """See Section 5.2 for complete implementation."""
    pass


def generate_ml_analysis_jsons(bucket_path: str, bucket: str, windows: List[str]) -> int:
    """See Section 4.5 for complete implementation."""
    pass


# ===== CLI Entry Point =====
def main(client_id: str, bucket: str) -> int:
    """
    Main entry point for Stage 6: ML Analysis Generation.

    Args:
        client_id: Client identifier (e.g., "acme_corp")
        bucket: Bucket name (e.g., "18-33s")

    Returns:
        int: Exit code (0=success, 1-4=failure codes)
    """
    logger.info(f"Starting Stage 6: ML Analysis Generation for bucket {bucket}")

    # Get bucket configuration
    if bucket not in BUCKET_WINDOWS:
        logger.error(f"Invalid bucket: {bucket}. Must be one of {list(BUCKET_WINDOWS.keys())}")
        return 1

    windows = BUCKET_WINDOWS[bucket]
    logger.info(f"Bucket {bucket} has {len(windows)} windows: {windows}")

    # Get bucket directory path
    bucket_path = get_bucket_path(client_id, bucket)

    # Generate all ML analysis JSONs
    exit_code = generate_ml_analysis_jsons(bucket_path, bucket, windows)

    if exit_code == 0:
        logger.info(f"✓ Stage 6 complete: {1 + 2*len(windows)} JSONs generated successfully")
    else:
        logger.error(f"✗ Stage 6 failed with exit code {exit_code}")

    return exit_code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 6: ML Analysis Generation")
    parser.add_argument("--client", required=True, help="Client ID")
    parser.add_argument("--bucket", required=True, help="Bucket name (e.g., 18-33s)")

    args = parser.parse_args()

    sys.exit(main(args.client, args.bucket))
```

---

### 8.3 Integration Points

**Upstream Dependencies** (Stage 6 requires these stages to complete first):

```python
# From Stage 3: Feature Aggregation
# File: ml_analysis/aggregated_features.csv
# Used by: generate_video_rf_json() for distribution analysis
from ml_pipeline.stage3_feature_aggregation import FeatureAggregator

# From Stage 4: Feature Transformation
# Files: ml_analysis/rf_transformed.csv, {window}_rf_transformed.csv, {window}_km_transformed.csv
# Used by: generate_window_rf_json() and generate_window_kmeans_json()
from ml_pipeline.stage4_feature_transformation import FeatureTransformer

# From Stage 5: ML Model Training
# Files: models/rf_video_{bucket}.pkl, rf_{window}_{bucket}.pkl, {window}_kmeans_{bucket}.pkl, etc.
# Used by: All JSON generation functions
from ml_pipeline.stage5_model_training import ModelTrainer
```

**Downstream Consumers** (These stages depend on Stage 6 outputs):

```python
# Stage 7: LLM Report Generation
# Consumes: All JSON analysis files (3-15 JSONs depending on bucket window count)
# File: ml_pipeline/stage7_llm_report_generation.py
from ml_pipeline.stage7_llm_report_generation import LLMReportGenerator

# Example usage in Stage 7:
# video_rf_json = json.load(open('ml_analysis/rf_video_analysis.json'))
# hook_rf_json = json.load(open('ml_analysis/hook_rf_analysis.json'))
# hook_kmeans_json = json.load(open('ml_analysis/hook_kmeans_analysis.json'))
```

---

### 8.4 Shared Utilities

**Utility: config/bucket_definitions.py**

```python
"""
Centralized bucket window configuration.

Source: FoundationCHILD.md Section 6
"""

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

**Utility: utils/path_utils.py**

```python
"""
Path utility functions for bucket directory management.

Source: FoundationCHILD.md Section 2.2
"""

import os

DATA_ROOT = os.getenv('DATA_ROOT', '/data')


def get_bucket_path(client_id: str, bucket: str) -> str:
    """
    Get absolute path to bucket directory.

    Args:
        client_id: Client identifier (e.g., "acme_corp")
        bucket: Bucket name (e.g., "18-33s")

    Returns:
        str: Absolute path to bucket directory
        Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s"
    """
    # Simplified version - actual implementation may include hashtag/analysis_mode in path
    return os.path.join(DATA_ROOT, 'clients', client_id, 'buckets', f'bucket_{bucket}')


def ensure_directory_exists(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
```

**Utility: utils/logger.py**

```python
"""
Centralized logging configuration.

Source: FoundationCHILD.md logging standards
"""

import logging
import sys


def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Setup logger with consistent formatting.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
```

---

### 8.5 Import Dependencies

**Standard Library** (Python 3.9+):
```python
import os           # File path operations
import sys          # Exit codes
import json         # JSON serialization
import shutil       # Directory operations (atomic rollback)
import logging      # Logging
import traceback    # Error stack traces
from typing import List, Dict, Tuple  # Type hints
```

**Third-Party Libraries** (see Section 12 for versions):
```python
import pandas as pd      # CSV loading, DataFrame operations
import numpy as np       # Array operations, percentile calculations
import joblib           # Pickle file loading (models, scalers, X_data)
```

**Internal Modules**:
```python
from config.bucket_definitions import BUCKET_WINDOWS
from utils.logger import setup_logger
from utils.path_utils import get_bucket_path, ensure_directory_exists
from utils.validation import validate_json_schema
```

---

### 8.6 Directory Structure (Runtime)

**Before Stage 6 Execution**:
```
/data/clients/{client_id}/hashtags/{hashtag}/{analysis_mode}/bucket_{bucket}/
├── ml_analysis/
│   ├── aggregated_features.csv       (Stage 3 output)
│   ├── rf_transformed.csv            (Stage 4 output)
│   ├── {window}_rf_transformed.csv   (Stage 4 output, 6-7 files)
│   └── {window}_km_transformed.csv   (Stage 4 output, 6-7 files)
└── models/
    ├── rf_video_{bucket}.pkl         (Stage 5 output)
    ├── rf_{window}_{bucket}.pkl      (Stage 5 output, 6-7 files)
    ├── {window}_kmeans_{bucket}.pkl  (Stage 5 output, 6-7 files)
    ├── {window}_scalers_{bucket}.pkl (Stage 5 output, 6-7 files)
    ├── {window}_X_data_{bucket}.pkl  (Stage 5 output, 6-7 files)
    └── model_metrics.json            (Stage 5 output)
```

**During Stage 6 Execution** (Temp Directory):
```
/data/clients/{client_id}/hashtags/{hashtag}/{analysis_mode}/bucket_{bucket}/
└── ml_analysis/
    └── .tmp/
        ├── rf_video_analysis.json.tmp
        ├── {window}_rf_analysis.json.tmp      (6-7 files)
        └── {window}_kmeans_analysis.json.tmp  (6-7 files)
```

**After Stage 6 Execution** (Success):
```
/data/clients/{client_id}/hashtags/{hashtag}/{analysis_mode}/bucket_{bucket}/
└── ml_analysis/
    ├── rf_video_analysis.json
    ├── {window}_rf_analysis.json      (6-7 files)
    └── {window}_kmeans_analysis.json  (6-7 files)

Total: 13-15 JSON files (~95 KB)
Temp directory deleted
```

**After Stage 6 Execution** (Failure):
```
/data/clients/{client_id}/hashtags/{hashtag}/{analysis_mode}/bucket_{bucket}/
└── ml_analysis/
    (No JSON files - atomic rollback deleted all temp files)
```

---

### 8.7 Cross-Module Data Flow

```
Stage 3: Feature Aggregation
    ↓ aggregated_features.csv (15-20 MB)

Stage 4: Feature Transformation
    ↓ rf_transformed.csv + 12 window CSVs (~2-3 MB total)

Stage 5: ML Model Training
    ↓ 90 trained models (PKL files, ~1-2 MB total)

Stage 6: ML Analysis Generation ← THIS MODULE
    ↓ 13 JSON files (~95 KB total)

Stage 7: LLM Report Generation
    ↓ PDF creative reports
```

**Data Handoff Pattern**:
1. Stage 6 reads Stage 4 CSVs and Stage 5 models (read-only access)
2. Stage 6 generates JSON files in temp directory (atomic pattern)
3. Stage 6 validates all JSONs, then commits atomically (rename operation)
4. Stage 7 reads committed JSON files (Stage 6 guarantees all 13 files exist or none exist)

---

## Section 9: Configuration & Environment

**Source**: FoundationCHILD.md Sections 4, 6 | MLAnalysisGenerationCHILD.md Section 4.2

### 9.1 Environment Variables

```bash
# Required Environment Variables

# Data root directory (default: /data)
export DATA_ROOT="/data"

# Log level (default: INFO)
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
export LOG_LEVEL="INFO"

# Optional: Override specific paths for testing
export ML_MODELS_DIR="${DATA_ROOT}/clients/{client_id}/buckets/{bucket}/models"
export ML_ANALYSIS_DIR="${DATA_ROOT}/clients/{client_id}/buckets/{bucket}/ml_analysis"
```

**Environment Variable Reference**:

| Variable | Required | Default | Valid Values | Purpose |
|----------|----------|---------|--------------|---------|
| `DATA_ROOT` | No | `/data` | Any valid directory path | Root directory for client data storage |
| `LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | Logging verbosity level |
| `ML_MODELS_DIR` | No | `{DATA_ROOT}/clients/{client_id}/buckets/{bucket}/models` | Any valid directory path | Override models directory (testing only) |
| `ML_ANALYSIS_DIR` | No | `{DATA_ROOT}/clients/{client_id}/buckets/{bucket}/ml_analysis` | Any valid directory path | Override ml_analysis directory (testing only) |

---

### 9.2 Configuration Files

**File 1: config/bucket_definitions.py**

```python
"""
Centralized bucket window configuration.

Source: FoundationCHILD.md Section 6
"""

# Bucket window definitions (immutable - do not modify without updating all stages)
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

# Validation: Ensure all buckets are defined
VALID_BUCKETS = list(BUCKET_WINDOWS.keys())

# Helper function to get window count
def get_window_count(bucket: str) -> int:
    """Get number of windows for a bucket."""
    if bucket not in BUCKET_WINDOWS:
        raise ValueError(f"Invalid bucket: {bucket}. Must be one of {VALID_BUCKETS}")
    return len(BUCKET_WINDOWS[bucket])

# Helper function to get expected JSON count
def get_expected_json_count(bucket: str) -> int:
    """Get expected number of JSON files for a bucket (1 video RF + N window RF + N window K-Means)."""
    window_count = get_window_count(bucket)
    return 1 + (window_count * 2)  # 1 video RF + (N windows × 2 types)
```

**File 2: config/settings.py**

```python
"""
General configuration settings.

Source: MLAnalysisGenerationCHILD.md Section 4.2
"""

import os

# ===== Directory Configuration =====
DATA_ROOT = os.getenv('DATA_ROOT', '/data')

# ===== Distribution Analysis Parameters =====
TOP_PERFORMER_PERCENTAGE = 0.8  # Top 80% vs bottom 20% (contrastive strategy)
HIGH_PERCENTILE = 0.66          # 66th percentile threshold
LOW_PERCENTILE = 0.33           # 33rd percentile threshold

# ===== Feature Selection Parameters =====
MAX_FEATURES_VIDEO_RF = 10      # Top 10 features for video-level RF
MAX_FEATURES_WINDOW_RF = 10     # Top 10 features for window-level RF

# ===== K-Means Parameters =====
N_CLUSTERS = 3                  # Always 3 clusters per window (from Stage 5)

# ===== File Naming Conventions =====
VIDEO_RF_JSON_FILENAME = "rf_video_analysis.json"
WINDOW_RF_JSON_TEMPLATE = "{window}_rf_analysis.json"
WINDOW_KMEANS_JSON_TEMPLATE = "{window}_kmeans_analysis.json"

# ===== Temp Directory Configuration =====
TEMP_DIR_NAME = ".tmp"
TEMP_FILE_SUFFIX = ".tmp"

# ===== Validation Tolerances =====
DISTRIBUTION_SUM_TOLERANCE = 0.01  # Allow 1% rounding error in distribution percentages
FEATURE_IMPORTANCE_SUM_TOLERANCE = 0.01  # Allow 1% rounding error in feature importances

# ===== Performance Thresholds =====
MAX_EXECUTION_TIME_SECONDS = 30    # Warning threshold for total execution time
MAX_MEMORY_MB = 200                # Warning threshold for peak memory usage
```

---

### 9.3 CLI Configuration

**Usage**:
```bash
# Basic usage
python -m ml_pipeline.stage6_ml_analysis_generation --client acme_corp --bucket 18-33s

# With custom data root
DATA_ROOT=/mnt/data python -m ml_pipeline.stage6_ml_analysis_generation --client acme_corp --bucket 18-33s

# With debug logging
LOG_LEVEL=DEBUG python -m ml_pipeline.stage6_ml_analysis_generation --client acme_corp --bucket 18-33s
```

**CLI Parameters** (from FoundationCHILD.md Section 4):

| Parameter | Required | Type | Valid Values | Default | Description |
|-----------|----------|------|--------------|---------|-------------|
| `--client` | Yes | str | Alphanumeric + underscore (regex: `^[a-zA-Z0-9_]+$`) | None | Client identifier (e.g., "acme_corp") |
| `--bucket` | Yes | str | One of: `0-3s`, `3-9s`, `9-13s`, `13-18s`, `18-33s`, `33-60s`, `60-90s`, `90-120s` | None | Bucket duration range |

**Exit Codes** (from FoundationCHILD.md Section 7):

| Exit Code | Meaning | User Action |
|-----------|---------|-------------|
| 0 | Success - all JSONs generated | Continue to Stage 7 |
| 1 | Pre-flight validation failed - missing dependencies | Re-run Stage 4 or Stage 5 |
| 2 | JSON generation failed - model loading or data errors | Check logs, re-run Stage 5 if model corrupted |
| 3 | Output validation failed - schema errors | Report bug (code logic error) |
| 4 | Disk I/O failed - insufficient storage | Check disk space, retry after cleanup |

---

### 9.4 Deployment Configuration

**Production Environment**:
```bash
# Production settings (recommended)
export DATA_ROOT="/data"
export LOG_LEVEL="INFO"

# Ensure sufficient disk space for temp files
# Required: ~100 KB for temp JSONs per bucket
# Recommended: 1 GB free space minimum

# Verify directory permissions
sudo chown -R ml_user:ml_group /data/clients
sudo chmod -R 755 /data/clients
```

**Development Environment**:
```bash
# Development settings (verbose logging, custom paths)
export DATA_ROOT="/home/user/dev/rumiai_data"
export LOG_LEVEL="DEBUG"

# Use smaller test datasets
# Test bucket: 18-33s with 10 videos (instead of 100)
```

**Testing Environment**:
```bash
# Testing settings (isolated test data)
export DATA_ROOT="/tmp/rumiai_test_data"
export LOG_LEVEL="DEBUG"

# Use pytest fixtures for test data
# See Section 8 (Testing Strategy) for test data structure
```

---

### 9.5 Resource Requirements

**Minimum System Requirements**:

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| **Python Version** | 3.9 | 3.10+ | Type hints require 3.9+ |
| **RAM** | 500 MB | 1 GB | Peak usage ~200 MB during execution |
| **Disk Space (temp)** | 1 MB | 100 MB | Temp files ~95 KB per bucket, cleared after execution |
| **Disk Space (output)** | 100 KB | 10 MB | 13 JSON files × ~7 KB average = ~95 KB per bucket |
| **CPU** | 1 core | 2+ cores | Single-threaded execution (no parallelization) |

**Performance Scaling**:

| Video Count | Execution Time | Peak Memory | Disk I/O |
|-------------|----------------|-------------|----------|
| 50 videos | 6-8s | 100 MB | 50 MB read + 50 KB write |
| 100 videos | 10-12s | 150 MB | 100 MB read + 95 KB write |
| 200 videos | 18-22s | 200 MB | 200 MB read + 150 KB write |

---

### 9.6 Directory Permissions

**Required Permissions**:

```bash
# Read permissions (Stage 4 and Stage 5 outputs)
chmod 644 /data/clients/{client_id}/buckets/{bucket}/ml_analysis/*.csv
chmod 644 /data/clients/{client_id}/buckets/{bucket}/models/*.pkl
chmod 644 /data/clients/{client_id}/buckets/{bucket}/models/*.json

# Write permissions (Stage 6 outputs and temp directory)
chmod 755 /data/clients/{client_id}/buckets/{bucket}/ml_analysis/
chmod 755 /data/clients/{client_id}/buckets/{bucket}/ml_analysis/.tmp/  # Created by Stage 6

# Output files (after Stage 6 completion)
chmod 644 /data/clients/{client_id}/buckets/{bucket}/ml_analysis/*.json
```

**Ownership**:
```bash
# Ensure ml_user owns all directories
sudo chown -R ml_user:ml_group /data/clients/{client_id}/

# Verify permissions
ls -la /data/clients/{client_id}/buckets/{bucket}/ml_analysis/
# Expected: drwxr-xr-x ml_user ml_group ml_analysis/
#           -rw-r--r-- ml_user ml_group rf_video_analysis.json
```

---

### 9.7 Logging Configuration

**Log File Locations** (optional - if file logging enabled):
```bash
# Console logging (default - always enabled)
# Output: stdout/stderr

# File logging (optional - configure via settings.py)
/var/log/rumiai/stage6_ml_analysis_generation.log

# Rotation: Daily, keep 7 days
# Max size: 100 MB per file
```

**Log Levels**:

| Level | When to Use | Example Scenario |
|-------|-------------|------------------|
| **DEBUG** | Development, troubleshooting | Tracing feature normalization logic, examining distribution calculations |
| **INFO** | Production (default) | Pre-flight validation passed, JSON generation progress |
| **WARNING** | Non-critical issues | Distribution percentages sum to 0.99 instead of 1.0 (rounding error) |
| **ERROR** | Failures requiring user action | Pre-flight validation failed, model loading error, disk full |
| **CRITICAL** | System-level failures | Not used in Stage 6 (reserved for infrastructure failures) |

**Log Output Format**:
```
[2025-01-28 14:30:00] INFO: Starting Stage 6: ML Analysis Generation for bucket 18-33s
[2025-01-28 14:30:01] DEBUG: Loading video RF model: models/rf_video_18-33s.pkl
[2025-01-28 14:30:02] ERROR: Pre-flight validation failed: Stage 4 incomplete (1 file missing)
```

---

### 9.8 Configuration Validation

**Startup Validation**:
```python
def validate_configuration() -> None:
    """
    Validate configuration settings on startup.

    Checks:
    - DATA_ROOT exists and is writable
    - BUCKET_WINDOWS configuration is valid
    - All required Python packages are installed

    Raises:
        ConfigurationError: If any validation check fails
    """
    import os
    from config.bucket_definitions import BUCKET_WINDOWS, VALID_BUCKETS

    # Check DATA_ROOT exists
    data_root = os.getenv('DATA_ROOT', '/data')
    if not os.path.exists(data_root):
        raise ConfigurationError(f"DATA_ROOT does not exist: {data_root}")

    # Check DATA_ROOT is writable
    test_file = os.path.join(data_root, '.write_test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except IOError as e:
        raise ConfigurationError(f"DATA_ROOT is not writable: {data_root} ({e})")

    # Validate BUCKET_WINDOWS configuration
    if len(BUCKET_WINDOWS) != 8:
        raise ConfigurationError(f"BUCKET_WINDOWS must have 8 entries, found {len(BUCKET_WINDOWS)}")

    for bucket, windows in BUCKET_WINDOWS.items():
        if not windows:
            raise ConfigurationError(f"Bucket {bucket} has empty window list")

    # Check required packages are installed
    try:
        import pandas
        import numpy
        import joblib
    except ImportError as e:
        raise ConfigurationError(f"Required package not installed: {e}")

    logger.info("✓ Configuration validation passed")
```

**Runtime Configuration Check**:
```bash
# Verify configuration before running Stage 6
python -m ml_pipeline.stage6_ml_analysis_generation --validate-config

# Expected output:
# ✓ DATA_ROOT exists: /data
# ✓ DATA_ROOT is writable
# ✓ BUCKET_WINDOWS configuration valid (8 buckets)
# ✓ Required packages installed: pandas, numpy, joblib
# ✓ Configuration validation passed
```

---

## Section 10: Logging Specifications

**Source**: MLAnalysisGenerationCHILD.md Section 2.3 (Detailed Process) | Section 6 (Error Handling)

### 10.1 Log Message Catalog

**Format Convention**:
```
[TIMESTAMP] LEVEL: MESSAGE
```

**Timestamp Format**: `YYYY-MM-DD HH:MM:SS` (ISO 8601, 24-hour time)

---

### 10.2 Startup & Initialization Logs

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| INFO | `Starting Stage 6: ML Analysis Generation for bucket {bucket}` | Stage 6 entry point | `Starting Stage 6: ML Analysis Generation for bucket 18-33s` |
| INFO | `Bucket {bucket} has {count} windows: {window_list}` | After loading bucket configuration | `Bucket 18-33s has 6 windows: ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']` |
| DEBUG | `Bucket path: {bucket_path}` | After resolving bucket directory | `Bucket path: /data/clients/acme_corp/buckets/bucket_18-33s` |

---

### 10.3 Pre-Flight Validation Logs

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| INFO | `Pre-flight validation: checking Stage 4 and Stage 5 outputs...` | Start of pre-flight validation | `Pre-flight validation: checking Stage 4 and Stage 5 outputs...` |
| DEBUG | `Checking {count} Stage 4 CSV files...` | Before validating Stage 4 files | `Checking 13 Stage 4 CSV files...` |
| DEBUG | `✓ {file_path} exists ({size})` | Each Stage 4 file found | `✓ ml_analysis/aggregated_features.csv exists (15.2 MB)` |
| DEBUG | `✗ {file_path} MISSING` | Stage 4 file missing | `✗ ml_analysis/hook_rf_transformed.csv MISSING` |
| DEBUG | `Checking {count} Stage 5 model files...` | Before validating Stage 5 files | `Checking 20 Stage 5 model files...` |
| DEBUG | `✓ {file_path} exists ({size})` | Each Stage 5 file found | `✓ models/rf_video_18-33s.pkl exists (450 KB)` |
| DEBUG | `✗ {file_path} MISSING` | Stage 5 file missing | `✗ models/rf_hook_18-33s.pkl MISSING` |
| INFO | `✓ Pre-flight validation passed: All {stage4_count} Stage 4 files + {stage5_count} Stage 5 files exist` | All dependencies present | `✓ Pre-flight validation passed: All 13 Stage 4 files + 20 Stage 5 files exist` |
| ERROR | `Pre-flight validation failed:\nStage 4 incomplete ({count} files missing):\n{file_list}\nAction: Re-run Stage 4 (Feature Transformation)` | Stage 4 dependencies missing | `Pre-flight validation failed:\nStage 4 incomplete (1 file missing):\n  - ml_analysis/hook_rf_transformed.csv\nAction: Re-run Stage 4 (Feature Transformation)` |
| ERROR | `Pre-flight validation failed:\nStage 5 incomplete ({count} files missing):\n{file_list}\nAction: Re-run Stage 5 (ML Model Training)` | Stage 5 dependencies missing | `Pre-flight validation failed:\nStage 5 incomplete (3 files missing):\n  - models/rf_hook_18-33s.pkl\n  - models/hook_kmeans_18-33s.pkl\n  - models/hook_X_data_18-33s.pkl\nAction: Re-run Stage 5 (ML Model Training)` |

---

### 10.4 JSON Generation Logs

**Video-Level RF JSON**:

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| INFO | `Generating ML analysis JSONs to temp directory...` | Start of JSON generation phase | `Generating ML analysis JSONs to temp directory...` |
| DEBUG | `Loading video RF model: {model_path}` | Before loading video RF model | `Loading video RF model: models/rf_video_18-33s.pkl` |
| DEBUG | `Extracting feature importance ({count} features)` | After loading model | `Extracting feature importance (178 features)` |
| DEBUG | `Top {count} features: {feature_list}` | After sorting features | `Top 10 features: ['hook_eye_contact_rate' (0.22), 'middle_avg_word_count' (0.18), ...]` |
| DEBUG | `Loading aggregated_features.csv for distribution analysis ({rows} rows × {cols} columns)` | Before loading CSV | `Loading aggregated_features.csv for distribution analysis (100 rows × 129 columns)` |
| DEBUG | `Computing distribution stats for top {count} features...` | Before distribution loop | `Computing distribution stats for top 10 features...` |
| DEBUG | `Feature {index}/{total}: {feature_name} (top_avg={top_avg}, bottom_avg={bottom_avg}, gap={gap})` | Each feature processed | `Feature 1/10: hook_eye_contact_rate (top_avg=0.88, bottom_avg=0.45, gap=0.43)` |
| DEBUG | `Writing {filename} ({size})` | Before writing temp file | `Writing rf_video_analysis.json.tmp (30 KB)` |
| INFO | `✓ Generated {filename}` | After successful write | `✓ Generated rf_video_analysis.json.tmp` |

**Window-Level RF JSONs**:

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| DEBUG | `Loading window RF model: {model_path}` | Before loading window RF model | `Loading window RF model: models/rf_hook_18-33s.pkl` |
| DEBUG | `Extracting feature importance ({count} features)` | After loading model | `Extracting feature importance (21 features)` |
| DEBUG | `Loading model_metrics.json for performance stats` | Before loading metrics | `Loading model_metrics.json for performance stats` |
| DEBUG | `Loading {csv_path} for distribution analysis ({rows} rows × {cols} columns)` | Before loading CSV | `Loading hook_rf_transformed.csv for distribution analysis (100 rows × 21 columns)` |
| DEBUG | `Computing distribution stats for top {count} features...` | Before distribution loop | `Computing distribution stats for top 10 features...` |
| DEBUG | `Writing {filename} ({size})` | Before writing temp file | `Writing hook_rf_analysis.json.tmp (5 KB)` |
| INFO | `✓ Generated {filename}` | After successful write | `✓ Generated hook_rf_analysis.json.tmp` |

**Window-Level K-Means JSONs**:

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| DEBUG | `Loading K-Means model: {model_path}` | Before loading K-Means model | `Loading K-Means model: models/hook_kmeans_18-33s.pkl` |
| DEBUG | `Extracting cluster centroids ({n_clusters} clusters × {n_features} features)` | After loading model | `Extracting cluster centroids (3 clusters × 21 features)` |
| DEBUG | `Loading X_data: {x_data_path} (DataFrame with {count} feature names)` | Before loading X data | `Loading X_data: models/hook_X_data_18-33s.pkl (DataFrame with 21 feature names)` |
| DEBUG | `Loading {csv_path} for cluster assignments ({rows} rows)` | Before loading CSV | `Loading hook_km_transformed.csv for cluster assignments (100 rows)` |
| DEBUG | `Predicting cluster labels...` | Before cluster prediction | `Predicting cluster labels...` |
| DEBUG | `Cluster {id}: {size} videos, Cluster {id}: {size} videos, ...` | After prediction | `Cluster 0: 35 videos, Cluster 1: 42 videos, Cluster 2: 23 videos` |
| DEBUG | `Normalizing feature names (removing _scaled suffixes)...` | Before normalization | `Normalizing feature names (removing _scaled suffixes)...` |
| DEBUG | `'{original}' → '{normalized}'` | Each feature normalized (first 3 shown) | `'eye_contact_rate_scaled' → 'eye_contact_rate'` |
| DEBUG | `... ({count} more features)` | After showing first 3 | `... (18 more features)` |
| DEBUG | `Computing distances to centroid for {count} videos in cluster {id}...` | Before distance computation | `Computing distances to centroid for 35 videos in cluster 0...` |
| DEBUG | `Writing {filename} ({size})` | Before writing temp file | `Writing hook_kmeans_analysis.json.tmp (5 KB)` |
| INFO | `✓ Generated {filename}` | After successful write | `✓ Generated hook_kmeans_analysis.json.tmp` |

**Summary**:

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| INFO | `✓ All {count} JSONs generated to temp directory` | After all JSONs written | `✓ All 13 JSONs generated to temp directory` |

---

### 10.5 Validation Logs

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| INFO | `Validating JSON schemas...` | Start of validation phase | `Validating JSON schemas...` |
| DEBUG | `Checking all {count} expected files exist in .tmp/...` | Before file existence check | `Checking all 13 expected files exist in .tmp/...` |
| DEBUG | `✓ {filename} exists` | Each temp file found | `✓ rf_video_analysis.json.tmp exists` |
| DEBUG | `Validating JSON parseability...` | Before JSON parsing check | `Validating JSON parseability...` |
| DEBUG | `✓ All {count} JSONs are valid JSON format` | All JSONs parseable | `✓ All 13 JSONs are valid JSON format` |
| DEBUG | `Validating K-Means feature names (no _scaled suffixes)...` | Before feature name check | `Validating K-Means feature names (no _scaled suffixes)...` |
| DEBUG | `✓ {filename}: All feature names normalized` | Each K-Means JSON validated | `✓ hook_kmeans_analysis.json.tmp: All feature names normalized` |
| DEBUG | `✗ {filename}: Found {count} features with suffixes` | K-Means JSON has unnormalized features | `✗ hook_kmeans_analysis.json.tmp: Found 21 features with suffixes` |
| DEBUG | `Validating cluster size consistency...` | Before cluster size check | `Validating cluster size consistency...` |
| DEBUG | `✓ {window}_kmeans: sum({cluster_sizes}) = {sum} = total_videos` | Each K-Means cluster sizes valid | `✓ hook_kmeans: sum([35, 42, 23]) = 100 = total_videos` |
| INFO | `✓ All JSON schemas valid` | All validation passed | `✓ All JSON schemas valid` |

---

### 10.6 Atomic Commit Logs

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| INFO | `Committing JSONs (atomic rename)...` | Start of atomic commit phase | `Committing JSONs (atomic rename)...` |
| DEBUG | `Renaming {temp_file} → {final_file}` | Before each rename | `Renaming rf_video_analysis.json.tmp → rf_video_analysis.json` |
| INFO | `✓ Committed {filename}` | After each rename | `✓ Committed rf_video_analysis.json` |

---

### 10.7 Cleanup Logs

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| DEBUG | `Deleting temp directory: {temp_dir}` | Before temp directory deletion | `Deleting temp directory: ml_analysis/.tmp/` |
| INFO | `✓ Stage 6 complete: {count} JSONs generated` | After cleanup, before exit | `✓ Stage 6 complete: 13 JSONs generated` |

---

### 10.8 Success Summary Logs

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| INFO | `========================================` | Start of summary block | `========================================` |
| INFO | `Stage 6: ML Analysis Generation COMPLETE` | Success status | `Stage 6: ML Analysis Generation COMPLETE` |
| INFO | `Bucket: {bucket}` | Bucket processed | `Bucket: 18-33s` |
| INFO | `Total videos: {count}` | Video count | `Total videos: 100` |
| INFO | `JSONs generated: {count} ({breakdown})` | JSON count breakdown | `JSONs generated: 13 (1 video RF + 6 window RF + 6 window K-Means)` |
| INFO | `Total time: {time}s` | Execution time | `Total time: 11.2s` |
| INFO | `Exit code: {code} ({status})` | Exit code | `Exit code: 0 (SUCCESS)` |
| INFO | `========================================` | End of summary block | `========================================` |

---

### 10.9 Error Logs

**Pre-Flight Validation Errors**:

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| ERROR | `Pre-flight validation failed: {error_message}` | Any pre-flight error | `Pre-flight validation failed: Stage 4 incomplete (2 files missing): ...` |

**JSON Generation Errors**:

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| ERROR | `Stage 6 JSON generation failed: {exception_type}: {exception_message}` | Any generation exception | `Stage 6 JSON generation failed: UnpicklingError: invalid load key, '\x00'` |
| ERROR | `Stack trace:\n{stack_trace}` | After exception message | `Stack trace:\n  File "ml_analysis_generation.py", line 142, in generate_video_rf_json\n    rf_model = joblib.load(model_path)\n...` |
| ERROR | `Generated files before failure: {count} of {expected} expected` | After exception | `Generated files before failure: 7 of 13 expected` |

**Output Validation Errors**:

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| ERROR | `K-Means JSON {filename} contains {count} features with transformation suffixes: {feature_list}.\nFeature normalization failed. Check normalize_feature_name() logic.` | Feature normalization bug | `K-Means JSON hook_kmeans_analysis.json.tmp contains 21 features with transformation suffixes: ['eye_contact_rate_scaled', 'scene_count_scaled', 'word_count_scaled', 'energy_level_scaled', 'has_captions_encoded'].\nFeature normalization failed. Check normalize_feature_name() logic.` |
| ERROR | `K-Means JSON {filename} cluster sizes ({cluster_sizes}) sum to {sum}, but total_videos is {total}. Cluster assignment failed.` | Cluster size mismatch | `K-Means JSON closing_kmeans_analysis.json.tmp cluster sizes ([35, 42, 22]) sum to 99, but total_videos is 100. Cluster assignment failed.` |
| ERROR | `Missing temp file: {filename}` | Temp file not created | `Missing temp file: hook_rf_analysis.json.tmp` |
| ERROR | `Invalid JSON in {filename}: {json_error}` | JSON parsing error | `Invalid JSON in middle_1_kmeans_analysis.json.tmp: Expecting ',' delimiter: line 42 column 5 (char 1205)` |

**Disk I/O Errors**:

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| ERROR | `Disk I/O failed: {io_error}` | Any IOError exception | `Disk I/O failed: [Errno 28] No space left on device: '/data/clients/acme/buckets/bucket_18-33s/ml_analysis/.tmp/hook_kmeans_analysis.json.tmp'` |

---

### 10.10 Rollback Logs

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| DEBUG | `Deleting temp directory (atomic rollback)...` | Before rollback | `Deleting temp directory (atomic rollback)...` |
| WARNING | `Rolled back: Deleted all temp files ({count} files) (atomic failure)` | After rollback | `Rolled back: Deleted all temp files (7 files) (atomic failure)` |
| ERROR | `Rollback failed: {error}. Manual cleanup required for {temp_dir}` | Rollback exception | `Rollback failed: [Errno 13] Permission denied. Manual cleanup required for ml_analysis/.tmp/` |

---

### 10.11 Failure Summary Logs

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| INFO | `========================================` | Start of summary block | `========================================` |
| INFO | `Stage 6: ML Analysis Generation FAILED` | Failure status | `Stage 6: ML Analysis Generation FAILED` |
| INFO | `Exit code: {code} ({status})` | Exit code | `Exit code: 1 (PRE-FLIGHT VALIDATION FAILED)` |
| INFO | `========================================` | End of summary block | `========================================` |

---

### 10.12 Warning Logs (Non-Critical)

| Log Level | Message Template | When Logged | Example |
|-----------|------------------|-------------|---------|
| WARNING | `Distribution percentages sum to {sum} (expected 1.0), rounding error within tolerance` | Distribution percentages don't sum exactly to 1.0 | `Distribution percentages sum to 0.998 (expected 1.0), rounding error within tolerance` |
| WARNING | `Feature importance values sum to {sum} (expected 1.0), rounding error within tolerance` | Feature importances don't sum exactly to 1.0 | `Feature importance values sum to 0.997 (expected 1.0), rounding error within tolerance` |
| WARNING | `Feature {feature_name} not in aggregated CSV, distribution set to None` | Derived feature not in raw data | `Feature hook_middle_avg_word_count not in aggregated CSV, distribution set to None` |
| WARNING | `model_metrics.json missing metrics for {window}, set to None` | Metrics missing for window | `model_metrics.json missing metrics for hook, set to None` |

---

### 10.13 Logging Best Practices

**Implementation Guidelines**:

1. **Always log at INFO level**: Pre-flight validation start/success, JSON generation progress, validation success, atomic commit, final summary
2. **Use DEBUG for details**: File paths, feature counts, distribution calculations, feature name normalization
3. **Use ERROR for failures**: Pre-flight failures, generation exceptions, validation errors, disk I/O errors
4. **Use WARNING for non-critical issues**: Rounding errors, missing optional data

**Performance Logging**:
```python
import time

start_time = time.time()
# ... execution ...
elapsed = time.time() - start_time
logger.info(f"Total time: {elapsed:.1f}s")
```

**Memory Logging** (optional, for debugging):
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / (1024 * 1024)
logger.debug(f"Peak memory usage: {memory_mb:.1f} MB")
```

---

## Section 11: Implementation Log

**Purpose**: This section tracks all changes made during implementation of Stage 6: ML Analysis Generation. It serves as a living document to record deviations from the HLD, bug fixes, optimizations, and design decisions made during coding.

**Update Frequency**: After each significant change during implementation

**Review Protocol**: All Implementation Log entries must be reviewed during code review before merging

---

### 11.1 Change Log Format

Each entry must follow this format:

```markdown
#### Entry {NUMBER}: {SHORT_DESCRIPTION}

**Date**: YYYY-MM-DD
**Author**: {Developer Name}
**Severity**: [CRITICAL | HIGH | MEDIUM | LOW]
**Type**: [DEVIATION | BUG_FIX | OPTIMIZATION | CLARIFICATION]

**HLD Section Affected**: {Section Number and Name}

**Description**:
{Detailed description of the change, including:
- What was changed
- Why it was changed
- Impact on other components
- Alternative approaches considered}

**Code Changes**:
```python
# Before:
{original code snippet}

# After:
{modified code snippet}
```

**Testing**:
{How the change was validated}

**Approved By**: {Reviewer Name}
**Status**: [PENDING | APPROVED | REJECTED]
```

---

### 11.2 Severity Levels

| Severity | Definition | Examples | Approval Required |
|----------|------------|----------|-------------------|
| **CRITICAL** | Changes that affect system correctness, data integrity, or HLD contract | Changing JSON schema structure, modifying exit codes, altering atomic pattern | Tech Lead + Product Owner |
| **HIGH** | Changes that affect performance, API behavior, or stage dependencies | Adding new validation rules, changing distribution calculation logic | Tech Lead |
| **MEDIUM** | Changes that improve code quality or fix non-critical bugs | Refactoring functions, improving error messages, optimizing loops | Senior Developer |
| **LOW** | Clarifications, documentation updates, minor optimizations | Adding comments, fixing typos in log messages, variable renaming | Any Reviewer |

---

### 11.3 Implementation Progress Checklist

**Auto-generated from TI Section 4**
**Created**: 2025-10-20 13:15
**Status**: ✅ COMPLETED (10/10 functions complete)
**Last Updated**: 2025-10-20 13:45

#### Phase 1: Document Reading & Setup
- [x] TI document read and verified (all 14 sections)
- [x] Output directory detected and validated: /home/jorge/rumiaifinal/ml_pipeline/stage6_analysis/
- [x] Progress checklist created in TI Section 11.3
- [x] Ready to begin implementation

#### Phase 2: Function Implementation (from TI Section 4)

**Total Functions**: 10
**Completed**: 10/10
**In Progress**: None
**Pending**: 0

##### Function Checklist:

- [x] **Function 1**: `validate_stage_dependencies()` - TI §4.1
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:30

- [x] **Function 2**: `generate_video_rf_json()` - TI §4.2
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:32

- [x] **Function 3**: `generate_window_rf_json()` - TI §4.3
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:34

- [x] **Function 4**: `normalize_feature_name()` - TI §4.4
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:36

- [x] **Function 5**: `generate_window_kmeans_json()` - TI §4.4
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:38

- [x] **Function 6**: `validate_all_json_schemas()` - TI §5.2
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:40

- [x] **Function 7**: `generate_ml_analysis_jsons()` - TI §4.5
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:42

- [x] **Function 8**: `main()` - TI §8.2
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:44

- [x] **Function 9**: Custom Exceptions (PreFlightValidationError, ValidationError) - TI §8.2
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:45

- [x] **Function 10**: Module-level configuration and imports - TI §8.2
  - Status: ✅ COMPLETED
  - Deviations: None
  - Completion time: 2025-10-20 13:45

#### Phase 3: Validation & QA
- [x] All TI Section 5 validations implemented
- [ ] All TI Section 7 traces executed successfully (requires test data)
- [ ] Unit tests passing (test infrastructure not yet set up)
- [ ] Integration tests passing (test infrastructure not yet set up)

#### Phase 4: Post-Implementation
- [x] All deviations logged in TI Section 11.5 (0 deviations - implementation matches TI spec exactly)
- [ ] QA fixes applied and logged (no QA issues identified during implementation)
- [ ] Ready for reconcile_docs.py
- [x] Implementation complete

#### Resume Instructions (For New CLI Instances)

**If implementation interrupted, resume with:**
1. Read this checklist (TI Section 11.3)
2. Identify last completed function
3. Resume from next unchecked function
4. Continue function-by-function workflow

**Current Resume Point**: All functions complete - ready for QA testing

---

### 11.4 Review Protocol

**Before Implementation**:
1. All deviations from HLD must be discussed with Tech Lead
2. Critical changes require written approval before coding
3. High-severity changes require design review meeting

**During Implementation**:
1. Log all changes in Section 11.5 as they occur
2. Include code snippets showing before/after
3. Document testing approach for each change

**Before Merge**:
1. All Implementation Log entries must have "APPROVED" status
2. Tech Lead must review all HIGH and CRITICAL entries
3. Update HLD if permanent deviations are approved

---

### 11.4 Implementation Log Entries

*This section will be populated during implementation. Currently empty.*

---

**Example Entry (for reference - remove before implementation)**:

#### Entry 001: Change feature normalization to handle edge case

**Date**: 2025-01-28
**Author**: John Doe
**Severity**: MEDIUM
**Type**: BUG_FIX

**HLD Section Affected**: Section 4.4 (normalize_feature_name function)

**Description**:
During testing, discovered that normalize_feature_name() fails when a feature has multiple suffixes (e.g., 'word_count_log_scaled'). The original implementation only removed one suffix per pass.

Changed implementation to iteratively remove all suffixes until no more are found. This ensures features like 'word_count_log_scaled' are correctly normalized to 'word_count'.

Impact: Improves robustness of feature normalization. No impact on other components since this is an internal function.

Alternative approaches:
1. Use regex pattern matching (rejected - less readable)
2. Remove all suffixes in single pass (rejected - order-dependent)

**Code Changes**:
```python
# Before:
def normalize_feature_name(feature_name: str) -> str:
    suffixes = ['_scaled', '_log', '_encoded']
    for suffix in suffixes:
        feature_name = feature_name.replace(suffix, '')
    return feature_name

# After:
def normalize_feature_name(feature_name: str) -> str:
    suffixes = ['_scaled', '_log', '_encoded']
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if suffix in feature_name:
                feature_name = feature_name.replace(suffix, '')
                changed = True
    return feature_name
```

**Testing**:
- Unit test added: test_normalize_feature_name_multiple_suffixes()
- Test cases: 'word_count_log_scaled' → 'word_count', 'energy_level_scaled_log' → 'energy_level'
- All existing tests pass

**Approved By**: Jane Smith (Tech Lead)
**Status**: APPROVED

---

### 11.5 Summary Statistics

*To be populated during implementation*

| Metric | Count |
|--------|-------|
| Total Entries | 0 |
| Critical Severity | 0 |
| High Severity | 0 |
| Medium Severity | 0 |
| Low Severity | 0 |
| Deviations | 0 |
| Bug Fixes | 0 |
| Optimizations | 0 |
| Clarifications | 0 |
| Pending Review | 0 |
| Approved | 0 |
| Rejected | 0 |

---

### 11.6 Cross-Reference to HLD Sections

*To be populated during implementation*

List of HLD sections with implementation changes:

| HLD Section | Entry Numbers | Summary |
|-------------|---------------|---------|
| (Empty - no changes yet) | | |

---

### 11.7 Post-Implementation Review Checklist

Before marking implementation complete, verify:

- [ ] All Implementation Log entries have "APPROVED" status
- [ ] All CRITICAL and HIGH severity changes reviewed by Tech Lead
- [ ] HLD updated if permanent deviations approved
- [ ] All code changes have corresponding unit tests
- [ ] All changes documented with clear rationale
- [ ] No pending review items remain
- [ ] Summary statistics updated
- [ ] Cross-reference table complete

---

**Note to Implementers**:

This section is intentionally empty at TI creation time. As you implement Stage 6, use this section to track:

1. **Deviations**: Any time you need to deviate from the TI specification
2. **Bug Fixes**: Issues discovered during implementation that weren't anticipated in the design
3. **Optimizations**: Performance improvements or code quality enhancements
4. **Clarifications**: Ambiguities in the TI that needed interpretation

**Best Practices**:
- Log changes as you make them (don't wait until code review)
- Be specific about what changed and why
- Include code snippets for clarity
- Document testing approach
- Get timely reviews (don't accumulate pending entries)

---

### 11.5 TI Generation Log Entries

**Purpose**: Record any deviations from HLD specifications discovered during TI generation (Phase 2).

**CRITICAL**: This section MUST be populated after TI generation is complete.

---

**Scenario B: Deviations from HLD detected** (1 deviation found):

⚠️ **1 HLD → TI deviation found** (corrected to align with upstream Stage 5 implementation)

---

### Deviation #1: model_metrics.json Schema Mismatch (Cross-Stage Alignment)

**Affected Section**: TI Section 3.2 (INPUT SCHEMA 5: Stage 5 Model Metrics)

**Original HLD Specification** (MLAnalysisGenerationCHILD.md Section 5.1):
```python
# Simplified schema (inferred from HLD)
ModelMetricsSchema = {
    "rf_{window}": {  # Top-level keys like "rf_hook", "rf_middle_1"
        "accuracy": float,
        "precision": float,
        "recall": float
    }
}
```

**Actual Stage 5 Output** (per MLModelTrainingCHILDTI.md Section 3.3):
```python
# Nested schema with metadata
ModelMetricsSchema = {
    "bucket": str,
    "total_videos": int,
    "video_level_rf": { ... },
    "window_level_rf": {
        "{window}": {  # Keys like "hook", "middle_1" (NO "rf_" prefix)
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1_score": float,
            ...
        }
    }
}
```

**Deviation Type**: Schema mismatch between HLD assumption and upstream TI implementation

**Root Cause**:
- HLD Section 5.1 showed simplified/inferred schema (not verbatim from Stage 5 HLD)
- Stage 5 TI (MLModelTrainingCHILDTI.md) implements more robust nested schema with metadata
- Stage 6 HLD did not reference Stage 5 TI output schema directly

**Impact on Implementation**:
- **Line 881**: Changed `all_metrics.get(f'rf_{window}', {})` → `all_metrics.get('window_level_rf', {}).get(window, {})`
- **Lines 387-424**: Updated INPUT SCHEMA 5 to match Stage 5's actual nested structure
- **Impact Level**: Low - metrics are optional (used only for display), feature importance comes from model attributes

**Justification for Deviation**:
1. **Cross-stage consistency**: Stage 5 TI is already defined; Stage 6 must adapt
2. **Better design**: Stage 5's nested schema includes validation metadata (`bucket`, `total_videos`)
3. **Extensibility**: Nested structure can accommodate future metrics without breaking changes
4. **Non-breaking**: Stage 6 handles missing metrics gracefully (all_metrics.get() returns empty dict if missing)

**Verification**:
- ✅ Updated schema tested against Stage 5 TI Section 3.3 (MLModelTrainingCHILDTI.md lines 511-549)
- ✅ Loading code uses safe `.get().get()` pattern (returns empty dict if nested keys missing)
- ✅ Downstream code handles missing metrics (lines 918-922: sets accuracy/precision/recall to None if not found)

**Approval Status**: ⚠️ Requires Tech Lead approval for cross-stage schema alignment

---

**Coverage Summary** (after deviation correction):
- ✅ MLAnalysisGenerationCHILD.md: 34 sections mapped (100% coverage)
- ✅ FoundationCHILD.md: 10 relevant sections mapped (100% coverage)
- ⚠️ Cross-stage alignment: 1 schema corrected to match MLModelTrainingCHILDTI.md (Stage 5)
- ✅ All HLD Appendix A example data used in TI Section 7 traces
- ✅ All HLD Appendix C pseudocode expanded in TI Section 4

**Generation Date**: 2025-01-28 (updated from 2025-01-20 after deviation fix)
**Generated By**: Claude (Sonnet 4.5)
**Review Status**: ⏳ Pending Tech Lead Review (deviation approval required)

---

## Section 12: Dependencies & Prerequisites

**Source**: MLAnalysisGenerationCHILD.md Sections 3.1, 3.4 | FoundationCHILD.md Section 2

### 12.1 Python Package Dependencies

**Python Version**: 3.9+ (required for type hints and modern features)

**Core Dependencies**:

| Package | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **pandas** | ≥1.3.0 | CSV loading, DataFrame operations | `pip install pandas>=1.3.0` |
| **numpy** | ≥1.21.0 | Array operations, percentile calculations, distance metrics | `pip install numpy>=1.21.0` |
| **scikit-learn** | ≥1.0.0 | RandomForestClassifier, KMeans model interfaces | `pip install scikit-learn>=1.0.0` |
| **joblib** | ≥1.1.0 | Pickle file loading (models, scalers, X_data) | `pip install joblib>=1.1.0` |

**Standard Library** (no installation required):
- `os` - File path operations
- `sys` - Exit codes
- `json` - JSON serialization/deserialization
- `shutil` - Directory operations (atomic rollback)
- `logging` - Logging infrastructure
- `traceback` - Stack trace formatting
- `typing` - Type hints

**Optional Dependencies** (for development/testing):
- `pytest` ≥7.0.0 - Unit testing framework
- `pytest-cov` ≥3.0.0 - Code coverage reporting
- `black` ≥22.0.0 - Code formatting
- `mypy` ≥0.950 - Type checking
- `psutil` ≥5.9.0 - Memory profiling (optional)

**Installation Command**:
```bash
# Minimal installation (production)
pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 joblib>=1.1.0

# Full installation (development)
pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 joblib>=1.1.0 \
            pytest>=7.0.0 pytest-cov>=3.0.0 black>=22.0.0 mypy>=0.950 psutil>=5.9.0
```

**Requirements File** (requirements.txt):
```
# Core dependencies
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0

# Optional dependencies (uncomment for development)
# pytest>=7.0.0
# pytest-cov>=3.0.0
# black>=22.0.0
# mypy>=0.950
# psutil>=5.9.0
```

---

### 12.2 System Dependencies

**Operating System**: Linux (Ubuntu 20.04+ or equivalent)

**System Requirements**:

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| **OS** | Linux (any distro) | Ubuntu 20.04+ | Tested on Ubuntu, should work on other Linux distros |
| **Python** | 3.9 | 3.10+ | Type hints require 3.9+, async features benefit from 3.10+ |
| **RAM** | 500 MB | 1 GB | Peak usage ~200 MB during execution |
| **Disk Space (temp)** | 1 MB | 100 MB | Temp files ~95 KB per bucket, cleared after execution |
| **Disk Space (output)** | 100 KB | 10 MB | 13 JSON files × ~7 KB average = ~95 KB per bucket |
| **CPU** | 1 core | 2+ cores | Single-threaded execution (no parallelization in Stage 6) |

**System Commands** (must be available):
- `chmod` - Setting file permissions (optional - only if permission management needed)
- `chown` - Setting file ownership (optional - only if ownership management needed)

---

### 12.3 Upstream Stage Dependencies

Stage 6 depends on outputs from Stages 3, 4, and 5. All upstream stages must complete successfully before running Stage 6.

**Dependency Graph**:
```
Stage 3: Feature Aggregation
    ↓ aggregated_features.csv
Stage 4: Feature Transformation
    ↓ rf_transformed.csv + 12 window CSVs
Stage 5: ML Model Training
    ↓ 20 model files (PKL) + 1 metrics JSON
Stage 6: ML Analysis Generation ← THIS STAGE
```

**Stage 3: Feature Aggregation** (Required):

| File | Location | Size | Purpose | Source Stage |
|------|----------|------|---------|--------------|
| `aggregated_features.csv` | `ml_analysis/` | ~15 MB | Distribution analysis for video-level RF | Stage 3 |

**Stage 4: Feature Transformation** (Required):

| File Pattern | Count | Location | Size | Purpose | Source Stage |
|--------------|-------|----------|------|---------|--------------|
| `rf_transformed.csv` | 1 | `ml_analysis/` | ~2 MB | Video-level RF input (not directly used by Stage 6, but validated) | Stage 4 |
| `{window}_rf_transformed.csv` | 6-7 | `ml_analysis/` | ~200 KB each | Distribution analysis for window-level RF | Stage 4 |
| `{window}_km_transformed.csv` | 6-7 | `ml_analysis/` | ~200 KB each | Cluster assignment for K-Means | Stage 4 |

**Stage 5: ML Model Training** (Required):

| File Pattern | Count | Location | Size | Purpose | Source Stage |
|--------------|-------|----------|------|---------|--------------|
| `rf_video_{bucket}.pkl` | 1 | `models/` | ~450 KB | Video-level RF model (feature importances) | Stage 5 |
| `rf_{window}_{bucket}.pkl` | 6-7 | `models/` | ~50 KB each | Window-level RF models (feature importances) | Stage 5 |
| `{window}_kmeans_{bucket}.pkl` | 6-7 | `models/` | ~25 KB each | K-Means models (cluster centroids) | Stage 5 |
| `{window}_scalers_{bucket}.pkl` | 6-7 | `models/` | ~10 KB each | MinMaxScaler objects (validated but not used) | Stage 5 |
| `{window}_X_data_{bucket}.pkl` | 6-7 | `models/` | ~80 KB each | Feature names for K-Means | Stage 5 |
| `model_metrics.json` | 1 | `models/` | ~5 KB | Model performance metrics | Stage 5 |

**Total Input Files** (for bucket "18-33s"):
- Stage 3: 1 file (~15 MB)
- Stage 4: 13 files (~2-3 MB)
- Stage 5: 20 files (~1-2 MB)
- **Total: 34 files (~18-20 MB)**

**Pre-Flight Validation** (see Section 4.1):
Stage 6 validates all 34 input files exist before generating any JSONs (fail-fast principle).

---

### 12.4 Downstream Stage Dependencies

**Stage 7: LLM Report Generation** (Consumes Stage 6 outputs):

| File Pattern | Count | Location | Size | Purpose | Consumer Stage |
|--------------|-------|----------|------|---------|----------------|
| `rf_video_analysis.json` | 1 | `ml_analysis/` | ~30 KB | Cross-window feature importance for Phase 2 synthesis | Stage 7 |
| `{window}_rf_analysis.json` | 6-7 | `ml_analysis/` | ~5 KB each | Per-window feature importance for Phase 1 analysis | Stage 7 |
| `{window}_kmeans_analysis.json` | 6-7 | `ml_analysis/` | ~5 KB each | Cluster centroids for Phase 1 cluster interpretation | Stage 7 |

**Total Output Files** (varies by bucket):
- **Formula**: 1 + (2 × window_count) JSON files
- **Examples**:
  - Bucket "0-3s" (1 window): 3 JSONs (~20 KB)
  - Bucket "18-33s" (6 windows): 13 JSONs (~95 KB)
  - Bucket "90-120s" (7 windows): 15 JSONs (~110 KB)

**Output Contract** (see Section 2.2):
Stage 6 guarantees either all JSONs exist (success) or none exist (atomic rollback on failure).

---

### 12.5 Configuration Dependencies

**File: config/bucket_definitions.py** (Required):

```python
# Source: FoundationCHILD.md Section 6
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

**File: config/settings.py** (Required):

```python
# Distribution analysis parameters
TOP_PERFORMER_PERCENTAGE = 0.8  # Top 80% vs bottom 20%
HIGH_PERCENTILE = 0.66          # 66th percentile threshold
LOW_PERCENTILE = 0.33           # 33rd percentile threshold

# Feature importance limits
MAX_FEATURES_VIDEO_RF = 10      # Top 10 features
MAX_FEATURES_WINDOW_RF = 10     # Top 10 features

# K-Means parameters
N_CLUSTERS = 3                  # Always 3 clusters
```

---

### 12.6 Environment Prerequisites

**Environment Variables** (see Section 9.1):

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DATA_ROOT` | No | `/data` | Root directory for client data |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |

**Directory Structure** (must exist before running Stage 6):

```
/data/clients/{client_id}/
└── hashtags/{hashtag}/{analysis_mode}/bucket_{bucket}/
    ├── ml_analysis/          # Must exist (created by Stage 3)
    │   └── (Stage 3, 4 CSVs)
    └── models/               # Must exist (created by Stage 5)
        └── (Stage 5 PKL files)
```

**Directory Permissions** (see Section 9.6):

```bash
# Read permissions (Stage 3, 4, 5 outputs)
chmod 644 /data/clients/{client_id}/buckets/{bucket}/ml_analysis/*.csv
chmod 644 /data/clients/{client_id}/buckets/{bucket}/models/*.pkl
chmod 644 /data/clients/{client_id}/buckets/{bucket}/models/*.json

# Write permissions (Stage 6 outputs and temp directory)
chmod 755 /data/clients/{client_id}/buckets/{bucket}/ml_analysis/
```

---

### 12.7 Verification Checklist

Before running Stage 6, verify:

**Python Environment**:
- [ ] Python 3.9+ installed (`python --version`)
- [ ] All required packages installed (`pip list | grep -E "pandas|numpy|scikit-learn|joblib"`)
- [ ] Virtual environment activated (recommended)

**Upstream Stages Complete**:
- [ ] Stage 3 (Feature Aggregation) completed successfully
- [ ] Stage 4 (Feature Transformation) completed successfully
- [ ] Stage 5 (ML Model Training) completed successfully
- [ ] All 34 input files exist (Stage 6 will validate during pre-flight)

**Directory Structure**:
- [ ] `/data/clients/{client_id}/buckets/{bucket}/ml_analysis/` exists
- [ ] `/data/clients/{client_id}/buckets/{bucket}/models/` exists
- [ ] Directories have correct permissions (755 for directories, 644 for files)

**Configuration**:
- [ ] `config/bucket_definitions.py` exists and has 8 bucket definitions
- [ ] `config/settings.py` exists with distribution parameters
- [ ] Environment variables set (if overriding defaults)

**Disk Space**:
- [ ] At least 1 GB free space in `/data` (recommended)
- [ ] At least 100 MB free in temp directory location

**Run Verification Command**:
```bash
# Quick verification script
python -c "
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
print('✓ All dependencies installed')
print(f'Python version: {sys.version}')
print(f'Pandas version: {pd.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Scikit-learn version: {joblib.__version__}')
"
```

Expected output:
```
✓ All dependencies installed
Python version: 3.10.x
Pandas version: 1.5.x
NumPy version: 1.23.x
Scikit-learn version: 1.1.x
```

---

### 12.8 Troubleshooting Common Dependency Issues

**Issue 1: ImportError for pandas/numpy/scikit-learn**

```bash
# Solution: Install missing packages
pip install pandas numpy scikit-learn joblib

# Or use requirements.txt
pip install -r requirements.txt
```

**Issue 2: Pre-flight validation fails (missing upstream files)**

```bash
# Check which files are missing
ls -la /data/clients/{client_id}/buckets/{bucket}/ml_analysis/
ls -la /data/clients/{client_id}/buckets/{bucket}/models/

# Re-run upstream stages
python -m ml_pipeline.stage3_feature_aggregation --client {client_id} --bucket {bucket}
python -m ml_pipeline.stage4_feature_transformation --client {client_id} --bucket {bucket}
python -m ml_pipeline.stage5_model_training --client {client_id} --bucket {bucket}
```

**Issue 3: Permission denied errors**

```bash
# Fix directory permissions
sudo chown -R $USER:$USER /data/clients/{client_id}/
chmod -R 755 /data/clients/{client_id}/buckets/{bucket}/ml_analysis/
chmod 644 /data/clients/{client_id}/buckets/{bucket}/ml_analysis/*.csv
```

**Issue 4: Python version too old**

```bash
# Check Python version
python --version

# If < 3.9, upgrade Python
sudo apt-get update
sudo apt-get install python3.10

# Or use pyenv
pyenv install 3.10.0
pyenv local 3.10.0
```

---

## Section 13: HLD Traceability Matrix

**Purpose**: This matrix maps every section of this TI document back to the corresponding sections in the parent HLD (MLAnalysisGenerationCHILD.md) and Foundation HLD (FoundationCHILD.md), ensuring complete traceability and coverage.

**Coverage Verification**:
- ✓ All HLD sections mapped to TI sections
- ✓ All TI sections reference source HLD sections
- ✓ Bidirectional traceability maintained

---

### 13.1 TI Section → HLD Section Mapping

| TI Section | TI Section Name | HLD Source(s) | HLD Section Name(s) | Coverage % |
|------------|-----------------|---------------|---------------------|------------|
| **Section 1** | Document Metadata | MLAnalysisGenerationCHILD.md (all) + FoundationCHILD.md | Complete document overview | 100% |
| **Section 2** | Stage Contract | MLAnalysisGenerationCHILD.md Sections 3.1, 3.2, 5.1, 5.2 + FoundationCHILD.md Sections 2, 4, 7 | Input Dependencies, Output Contracts, Input/Output Schemas, CLI Parameters, Exit Codes | 100% |
| **Section 3** | Data Schemas | MLAnalysisGenerationCHILD.md Sections 5.1, 5.2 + FoundationCHILD.md Sections 5.1, 5.3 | Input Schema, Output Schema, Config Schema, Checkpoint Schema | 100% |
| **Section 4** | Algorithmic Specifications | MLAnalysisGenerationCHILD.md Sections 2.3 (all subsections), Appendix C | Detailed Process, Pseudocode | 100% |
| **Section 5** | Validation Rules | MLAnalysisGenerationCHILD.md Sections 6.1, 6.3 | Input Validation, Output Validation | 100% |
| **Section 6** | Error Handling | MLAnalysisGenerationCHILD.md Section 6.2 + FoundationCHILD.md Section 7 | Error Cases, Standardized Exit Codes | 100% |
| **Section 7** | Complete Example Traces | MLAnalysisGenerationCHILD.md Sections 2.2, 7, Appendix B | Data Flow, Performance, Example Data | 100% |
| **Section 8** | File Structure & Integration | FoundationCHILD.md Section 2 + MLAnalysisGenerationCHILD.md Section 3 | Client Architecture, Dependencies & Integration | 100% |
| **Section 9** | Configuration & Environment | FoundationCHILD.md Sections 4, 6 + MLAnalysisGenerationCHILD.md Section 4.2 | CLI Parameters, Bucket Definitions, Internal Configuration | 100% |
| **Section 10** | Logging Specifications | MLAnalysisGenerationCHILD.md Sections 2.3, 6 | Detailed Process logs, Error Handling logs | 100% |
| **Section 11** | Implementation Log | TI Template Section 11 | (Empty - to be filled during implementation) | N/A |
| **Section 12** | Dependencies & Prerequisites | MLAnalysisGenerationCHILD.md Sections 3.1, 3.4 + FoundationCHILD.md Section 2 | Input Dependencies, External Dependencies, Directory Structure | 100% |
| **Section 13** | HLD Traceability Matrix | TI Template Section 13 | (This section) | N/A |
| **Section 14** | References | MLAnalysisGenerationCHILD.md + FoundationCHILD.md (all references) | Complete document references | 100% |

---

### 13.2 HLD Section → TI Section Mapping

**Source: MLAnalysisGenerationCHILD.md**

| HLD Section | HLD Section Name | Mapped to TI Section(s) | Coverage Notes |
|-------------|------------------|-------------------------|----------------|
| **Section 1** | Context & Business Goal | Section 1 (Document Metadata) | Business rationale documented |
| **Section 1.1** | What Problem Does This Solve? | Section 1 (Document Metadata) | Problem statement included |
| **Section 1.2** | Where This Fits in Pipeline | Section 1 (Document Metadata), Section 8.7 (Cross-Module Data Flow) | Pipeline position documented |
| **Section 1.3** | Success Criteria | Section 1 (Document Metadata) | Success criteria referenced |
| **Section 2** | Architecture & Design | Section 4 (Algorithmic Specifications) | Architecture translated to implementation |
| **Section 2.1** | High-Level Approach | Section 4 (Algorithmic Specifications) | Approach documented in function specs |
| **Section 2.2** | Data Flow | Section 7 (Complete Example Traces), Section 8.7 (Cross-Module Data Flow) | Data flow illustrated with examples |
| **Section 2.3** | Detailed Process | Section 4 (Algorithmic Specifications) | All 5 subsections mapped to functions |
| **Section 2.3.1** | Pre-Flight Validation | Section 4.1 (validate_stage_dependencies), Section 5.1 (Input Validation) | Complete implementation + validation rules |
| **Section 2.3.2** | Generate Video-Level RF JSON | Section 4.2 (generate_video_rf_json) | Complete implementation with edge cases |
| **Section 2.3.3** | Generate Window-Level RF JSONs | Section 4.3 (generate_window_rf_json) | Complete implementation with edge cases |
| **Section 2.3.4** | Generate Window-Level K-Means JSONs | Section 4.4 (generate_window_kmeans_json + normalize_feature_name) | Complete implementation with normalization |
| **Section 2.3.5** | Atomic Output Commit | Section 4.5 (generate_ml_analysis_jsons) | Complete implementation with rollback |
| **Section 3** | Dependencies & Integration | Section 2 (Stage Contract), Section 8 (File Structure), Section 12 (Dependencies) | All dependencies documented |
| **Section 3.1** | Input Dependencies | Section 2.1 (Input Contract), Section 12.3 (Upstream Stage Dependencies) | All 34 input files listed |
| **Section 3.2** | Output Contracts | Section 2.2 (Output Contract), Section 12.4 (Downstream Stage Dependencies) | All 13 output files listed |
| **Section 3.3** | Cross-Stage Dependencies | Section 12.3, 12.4 (Stage Dependencies) | Stage 3, 4, 5, 7 dependencies documented |
| **Section 3.4** | External Dependencies | Section 12.1 (Python Package Dependencies) | All packages with versions |
| **Section 4** | Configuration & Parameters | Section 9 (Configuration & Environment) | All config documented |
| **Section 4.1** | CLI Parameters | Section 2.1 (Input Contract), Section 9.3 (CLI Configuration) | All CLI params documented |
| **Section 4.2** | Internal Configuration | Section 9.2 (Configuration Files) | Distribution params, feature limits, N_CLUSTERS |
| **Section 5** | Data Schemas | Section 3 (Data Schemas) | All input/output schemas |
| **Section 5.1** | Input Schema | Section 3.2 (Stage 6 Input Schemas) | All 8 input schemas documented |
| **Section 5.2** | Output Schema | Section 3.3 (Stage 6 Output Schemas) | All 3 output schemas documented |
| **Section 6** | Error Handling & Validation | Section 5 (Validation Rules), Section 6 (Error Handling) | All validation rules + error cases |
| **Section 6.1** | Input Validation | Section 5.1 (Input Validation) | 10 validation rules (IV-1 to IV-10) |
| **Section 6.2** | Error Cases | Section 6.2 (Error Cases) | 8 error scenarios with handling |
| **Section 6.3** | Output Validation | Section 5.2 (Output Validation) | 8 validation rules (OV-1 to OV-8) |
| **Section 7** | Performance & Scalability | Section 7 (Complete Example Traces), Section 9.5 (Resource Requirements) | Performance metrics documented |
| **Section 7.1** | Performance Targets | Section 9.5 (Resource Requirements) | Execution time targets |
| **Section 7.2** | Measured Performance | Section 7.1 (Success Case - Performance Breakdown) | 11.2s for 100 videos documented |
| **Section 7.3** | Bottlenecks & Mitigations | Section 7.1 (Performance Breakdown) | Breakdown by phase |
| **Section 7.4** | Scalability Limits | Section 9.5 (Performance Scaling) | Scaling table (50, 100, 200 videos) |
| **Section 8** | Testing Strategy | (Not included in TI - covered separately in Test Plan) | Test plan separate document |
| **Appendix A** | Decision Log | Section 1 (Document Metadata - Rationale) | Key decisions documented |
| **Appendix B** | Example Data | Section 7 (Complete Example Traces) | 5 complete execution traces |
| **Appendix C** | Pseudocode | Section 4 (Algorithmic Specifications) | Pseudocode converted to Python |

**Source: FoundationCHILD.md**

| HLD Section | HLD Section Name | Mapped to TI Section(s) | Coverage Notes |
|-------------|------------------|-------------------------|----------------|
| **Section 2** | Client Architecture & Storage | Section 2.1 (Input Contract), Section 8.6 (Directory Structure) | All directory paths documented |
| **Section 2.1** | Directory Structure | Section 8.6 (Directory Structure - Runtime) | Before/during/after execution states |
| **Section 2.2** | Path Templates | Section 2.1 (Input Contract - Directory Paths) | All path templates included |
| **Section 4** | CLI Command Structure | Section 9.3 (CLI Configuration) | CLI usage examples |
| **Section 4.1** | CLI Parameters | Section 2.1 (Input Contract - CLI Parameters) | --client, --bucket documented |
| **Section 5.1** | Config Schema | Section 3.1 (Foundation Schemas) | ConfigSchema complete |
| **Section 5.3** | Checkpoint Schema | Section 3.1 (Foundation Schemas) | CheckpointSchema complete |
| **Section 6** | Bucket Definitions | Section 9.2 (Configuration Files - bucket_definitions.py) | BUCKET_WINDOWS complete |
| **Section 7** | Standardized Exit Codes | Section 2.2 (Output Contract - Exit Codes), Section 6.1 (Error Taxonomy) | All 5 exit codes (0-4) documented |

---

### 13.3 Coverage Analysis

**MLAnalysisGenerationCHILD.md Coverage**:

| HLD Section Type | Count | Mapped to TI | Coverage % |
|------------------|-------|--------------|------------|
| Main Sections (1-8) | 8 | 8 | 100% |
| Subsections (1.1-7.4) | 23 | 23 | 100% |
| Appendices (A-C) | 3 | 3 | 100% |
| **Total** | **34** | **34** | **100%** |

**FoundationCHILD.md Coverage** (Stage 6 relevant sections):

| HLD Section Type | Count | Mapped to TI | Coverage % |
|------------------|-------|--------------|------------|
| Main Sections (2, 4-7) | 4 | 4 | 100% |
| Subsections | 6 | 6 | 100% |
| **Total** | **10** | **10** | **100%** |

---

### 13.4 Unmapped HLD Sections

**Intentionally Not Mapped** (with rationale):

| HLD Section | HLD Section Name | Rationale for Exclusion |
|-------------|------------------|-------------------------|
| MLAnalysisGenerationCHILD.md Section 8 | Testing Strategy | Testing covered in separate Test Plan document (not part of TI specification) |

**All other HLD sections successfully mapped to TI.**

---

### 13.5 TI Sections Not in HLD

**TI-Specific Sections** (implementation details not in HLD):

| TI Section | TI Section Name | Rationale for Addition |
|------------|-----------------|------------------------|
| Section 7 | Complete Example Traces | Detailed execution traces with log output (expanded from HLD Appendix B) |
| Section 8 | File Structure & Integration | Module structure and import dependencies (implementation detail) |
| Section 10 | Logging Specifications | Complete log message catalog (extracted from HLD Section 2.3 but greatly expanded) |
| Section 11 | Implementation Log | TI template requirement for tracking implementation changes |
| Section 13 | HLD Traceability Matrix | TI template requirement for verifying coverage |

---

### 13.6 Verification Checklist

**HLD Coverage Verification**:
- [x] All MLAnalysisGenerationCHILD.md sections (1-8) mapped to TI
- [x] All MLAnalysisGenerationCHILD.md subsections (1.1-7.4) mapped to TI
- [x] All MLAnalysisGenerationCHILD.md appendices (A-C) mapped to TI
- [x] All relevant FoundationCHILD.md sections mapped to TI
- [x] All HLD decision log entries captured in TI
- [x] All HLD example data included in TI Section 7

**TI Coverage Verification**:
- [x] All TI sections reference source HLD sections
- [x] All TI algorithmic specifications (Section 4) derived from HLD Section 2.3
- [x] All TI validation rules (Section 5) derived from HLD Section 6
- [x] All TI error cases (Section 6) derived from HLD Section 6.2
- [x] All TI data schemas (Section 3) derived from HLD Section 5

**Bidirectional Traceability**:
- [x] Every HLD section maps to at least one TI section
- [x] Every TI section (except templates) references at least one HLD section
- [x] No orphaned HLD sections (except intentionally excluded Section 8 - Testing)
- [x] No unmapped TI content (all content traceable to HLD or TI template requirements)

---

### 13.7 Change Impact Analysis

**If HLD Section Changes, Update These TI Sections**:

| HLD Section | Affected TI Sections |
|-------------|----------------------|
| MLAnalysisGenerationCHILD.md Section 2.3.1 (Pre-Flight Validation) | TI Sections 4.1, 5.1, 7.2 |
| MLAnalysisGenerationCHILD.md Section 2.3.2 (Video RF JSON) | TI Sections 4.2, 7.1 |
| MLAnalysisGenerationCHILD.md Section 2.3.3 (Window RF JSON) | TI Sections 4.3, 7.1 |
| MLAnalysisGenerationCHILD.md Section 2.3.4 (K-Means JSON) | TI Sections 4.4, 7.1 |
| MLAnalysisGenerationCHILD.md Section 2.3.5 (Atomic Commit) | TI Sections 4.5, 6.6, 7.1 |
| MLAnalysisGenerationCHILD.md Section 5.1 (Input Schema) | TI Sections 2.1, 3.2 |
| MLAnalysisGenerationCHILD.md Section 5.2 (Output Schema) | TI Sections 2.2, 3.3 |
| MLAnalysisGenerationCHILD.md Section 6 (Error Handling) | TI Sections 5, 6, 10 |
| FoundationCHILD.md Section 6 (Bucket Definitions) | TI Sections 2.1, 9.2, 12.5 |
| FoundationCHILD.md Section 7 (Exit Codes) | TI Sections 2.2, 6.1, 9.3 |

---

### 13.8 Review Sign-Off

**Traceability Review**:
- [ ] Tech Lead verified all HLD sections mapped to TI
- [ ] Tech Lead verified all TI sections reference HLD sources
- [ ] Product Owner verified business requirements captured in TI
- [ ] QA verified all validation rules and error cases included

**Review Date**: ___________
**Reviewed By**: ___________
**Approval Status**: [ ] APPROVED [ ] NEEDS REVISION

---

## Section 14: References

### 14.1 Source Documents

- **MLAnalysisGenerationCHILD.md v1.0** (Last Updated: 2025-01-28): Parent HLD specification
  - Defines Stage 6 business context, algorithms, schemas, and validation rules
  - Parent Document: MLPlanningv2.md - Stage 6: ML Analysis Generation (Lines 1993-2388)

- **FoundationCHILD.md v1.1** (Last Updated: 2025-01-28): Directory structure and shared architecture
  - Provides client directory structure (Section 2: Client Architecture & Storage)
  - Defines bucket window configuration (Section 6: Centralized Configuration)
  - Specifies CLI parameter structure (Section 4: CLI Command Structure)

- **MLPlanningv2.md**: System-level ML pipeline architecture
  - High-level overview of all 10 pipeline stages
  - Stage 6 detailed specification (Lines 1993-2388)

### 14.2 Implementation Files

- **ml_pipeline/stage6_ml_analysis_generation.py**: Main implementation module (to be created)
  - Location: `rumiai/ml_pipeline/stage6_ml_analysis_generation.py`
  - Purpose: Extract insights from trained ML models and generate structured JSON analysis files
  - Dependencies: pandas, numpy, joblib, scikit-learn
  - Integration point: Called by pipeline orchestrator after Stage 5 completion

- **config/bucket_definitions.py**: Bucket window configuration
  - Location: `rumiai/config/bucket_definitions.py`
  - Purpose: Centralized BUCKET_WINDOWS mapping (defines 6-7 windows per bucket)
  - Source: FoundationCHILD.md Section 6

- **config/settings.py**: General configuration settings
  - Location: `rumiai/config/settings.py`
  - Purpose: Distribution analysis parameters, feature limits, file naming conventions
  - Source: MLAnalysisGenerationCHILD.md Section 4.2

- **utils/logger.py**: Centralized logging (shared across all stages)
- **utils/path_utils.py**: Directory path helpers (shared across all stages)
- **utils/validation.py**: Common validation functions (shared across all stages)

### 14.3 Related Stages

**Upstream Dependencies**:
- **Stage 3 (Feature Aggregation)**: Produces `aggregated_videos.csv`
  - Implementation: `ml_pipeline/stage3_feature_aggregation.py`
  - Output: Aggregated video metadata with performance metrics
  - Required for: Video-level RF distribution analysis (top 80% vs bottom 20%)

- **Stage 4 (Feature Transformation)**: Produces 13 transformed CSVs per bucket
  - Implementation: `ml_pipeline/stage4_feature_transformation.py`
  - Outputs: RF-transformed CSVs + K-Means-transformed CSVs (6-7 window pairs)
  - Required for: Column metadata extraction (original feature names)

- **Stage 5 (ML Model Training)**: Produces 20 trained model artifacts per bucket
  - Implementation: `ml_pipeline/stage5_model_training.py`
  - Outputs:
    - `rf_video_{bucket}.pkl` (1 video-level RF model)
    - `rf_{window}_{bucket}.pkl` (6-7 window-level RF models)
    - `kmeans_{window}_{bucket}.pkl` (6-7 K-Means models)
    - `X_{window}_{bucket}.pkl` (6-7 feature matrices for K-Means)
    - `scaler_kmeans_{window}_{bucket}.pkl` (6-7 scalers for K-Means)
    - `rf_video_metrics_{bucket}.json` (1 video-level performance metrics)
    - `rf_{window}_metrics_{bucket}.json` (6-7 window-level performance metrics)
  - Required for: All ML insight extraction

**Downstream Consumers**:
- **Stage 7 (LLM Creative Report Generation)**: Consumes all JSON analysis files
  - Implementation: `ml_pipeline/stage7_llm_report_generation.py`
  - Purpose: Generate creative insights using Claude LLM
  - Inputs (count varies by bucket: 3-15 JSONs):
    - `rf_video_analysis.json` (1 file: video-level feature importance)
    - `rf_{window}_analysis.json` (N files: window-level feature importance, where N = window_count)
    - `kmeans_{window}_clusters.json` (N files: window-level cluster centroids, where N = window_count)
  - Expectation: All JSONs present (1 + 2N files) or Stage 7 fails (atomic output pattern)

### 14.4 External Documentation

**Python Libraries**:
- **pandas** (v1.5+): DataFrame operations for CSV loading and distribution analysis
  - Documentation: https://pandas.pydata.org/docs/
  - Used in: `generate_video_rf_json()`, `generate_window_rf_json()`

- **numpy** (v1.24+): Numerical operations for percentile calculations
  - Documentation: https://numpy.org/doc/stable/
  - Used in: Distribution analysis (66th/33rd percentile thresholds)

- **scikit-learn** (v1.3+): ML model compatibility (RandomForest, K-Means)
  - Documentation: https://scikit-learn.org/stable/
  - Used in: Model attribute access (`.feature_importances_`, `.cluster_centers_`)

- **joblib** (v1.3+): Model deserialization from pickle files
  - Documentation: https://joblib.readthedocs.io/
  - Used in: Loading `.pkl` files from Stage 5

**ML Concepts**:
- **Random Forest Feature Importance**: https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation
- **K-Means Clustering**: https://scikit-learn.org/stable/modules/clustering.html#k-means
- **StandardScaler**: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

### 14.5 Configuration References

**Bucket Window Definitions** (FoundationCHILD.md Section 6):
```python
BUCKET_WINDOWS = {
    '0-3s': ['hook'],                                                    # 1 window
    '3-9s': ['hook', 'closing'],                                        # 2 windows
    '9-13s': ['hook', 'middle_aggregate', 'closing'],                   # 3 windows
    '13-18s': ['hook', 'middle_aggregate', 'closing'],                  # 3 windows
    '18-33s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing'],  # 6 windows
    '33-60s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],  # 7 windows
    '60-90s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],  # 7 windows
    '90-120s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'], # 7 windows
}
```

**Expected JSON Counts**:
- `0-3s`: 3 JSONs (1 video RF + 1 window RF + 1 window K-Means)
- `3-9s`: 5 JSONs (1 video RF + 2 window RF + 2 window K-Means)
- `9-13s`, `13-18s`: 7 JSONs (1 video RF + 3 window RF + 3 window K-Means)
- `18-33s`: 13 JSONs (1 video RF + 6 window RF + 6 window K-Means)
- `33-60s`, `60-90s`, `90-120s`: 15 JSONs (1 video RF + 7 window RF + 7 window K-Means)

**CLI Parameters** (FoundationCHILD.md Section 4):
```bash
python ml_pipeline/stage6_ml_analysis_generation.py \
  --client <client_id> \
  --bucket <bucket_name>
```

---

## Appendix A: Complete CLI Help Output
*(To be filled during implementation)*

---

## Appendix B: Exit Code Reference

**Quick Reference for Stage 6 Exit Codes**

| Exit Code | Category | Scenario | Recovery Action |
|-----------|----------|----------|-----------------|
| **0** | Success | All JSON files generated and validated successfully | None - proceed to Stage 7 |
| **1** | Pre-Flight Validation | Stage 4 or Stage 5 dependencies missing | Re-run Stage 4 (Feature Transformation) or Stage 5 (ML Model Training) to regenerate missing files |
| **2** | JSON Generation Failure | Model loading failed, CSV parsing error, or mid-generation crash | Check error logs for stack trace, verify model integrity (re-run Stage 5 if corrupted), re-run Stage 6 |
| **3** | Output Validation Failure | Generated JSONs failed schema validation (e.g., K-Means features not normalized, cluster sizes inconsistent) | Report bug - indicates code logic error in `generate_*_json()` functions or `normalize_feature_name()` |
| **4** | Disk I/O Failure | Disk full, permission denied, or I/O error during file write | Check disk space (`df -h`), verify write permissions on ml_analysis directory, retry after fixing |

**Notes**:
- All exit codes 1-4 trigger atomic rollback (all temp files deleted)
- Exit code 0 guarantees all JSONs exist (count = 1 + 2×window_count)
- See Section 6 for detailed error handling and Section 2.2 for exit code definitions in output contract

---

## Appendix C: Sample Output Files
[TODO - To be filled]

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-28 | RumiAI Team | Initial TI document creation from MLAnalysisGenerationCHILD.md |

---

**END OF TECHNICAL IMPLEMENTATION DOCUMENT**
