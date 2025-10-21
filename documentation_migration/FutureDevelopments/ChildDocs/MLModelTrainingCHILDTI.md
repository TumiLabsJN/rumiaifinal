# ML Model Training - Technical Implementation Document

> **TI Document**: MLModelTrainingCHILDTI.md
> **Parent HLD**: Stage5_MLModelTraining_HLD.md (Stage 5: ML Model Training)
> **Foundation HLD**: FoundationCHILD.md (Shared across all stages)
> **Version**: 1.0
> **Last Updated**: 2025-01-20
> **Status**: Draft

---

## Section 1: Document Metadata

**Feature Name**: ML Model Training

**Parent HLD**: Stage5_MLModelTraining_HLD.md

**Foundation HLD**: FoundationCHILD.md

**Covers HLD Sections**:

**From Stage5_MLModelTraining_HLD.md**:
- Section 1: Context & Business Goal
- Section 1.1: Why Stage 5 Exists
- Section 1.2: Success Criteria
- Section 1.3: Key Design Decisions
- Section 2: Architecture & Design
- Section 2.1: High-Level Approach
- Section 2.2: Data Flow
- Section 2.3: Detailed Process
- Section 2.3.1: Pre-Training Validation
- Section 2.3.2: Configuration Loading
- Section 2.3.3: Training Process
- Section 3: Critical Implementation Warnings
- Section 4: Dependencies & Integration
- Section 4.1: Input Dependencies
- Section 4.2: Output Contracts
- Section 4.3: Cross-Stage Dependencies
- Section 4.4: External Dependencies
- Section 5: Data Schemas
- Section 5.1: Input Schema
- Section 5.2: Output Schema
- Section 6: Error Handling & Validation
- Section 6.1: Input Validation
- Section 6.2: Error Cases
- Section 6.3: Error Logging
- Section 6.4: Recovery Procedures
- Section 7: Performance & Scalability
- Section 8: Testing Strategy
- Section 9: Configuration
- Section 9.1: Hyperparameter Configuration
- Section 10: References & Related Docs
- Appendix A: Decision Log
- Appendix B: Change Log
- Appendix C: Future Enhancements

**From FoundationCHILD.md**:
- Section 2: Client Architecture & Storage
- Section 2.1: Directory Structure
- Section 2.2: Path Templates
- Section 2.3: Architecture Notes
- Section 3: Configuration Dimensions
- Section 3.1: Target Types
- Section 3.2: Analysis Modes
- Section 3.3: Selection Strategies
- Section 4: CLI Command Structure
- Section 4.1: CLI Parameters
- Section 5: Configuration Schemas
- Section 5.1: config.json Schema
- Section 6: Bucket Definitions
- Section 6.1: Bucket Assignment Logic
- Section 7: Standardized Exit Codes

**Related TI Documents**:

**Depends On**:
- FoundationTI.md (REQUIRED - provides CLI parsing, directory creation, config management)
- FeatureTransformationCHILDTI.md (Stage 4) - Produces transformed feature CSVs (rf_transformed.csv, window_rf_transformed.csv, window_km_transformed.csv)

**Feeds Into**:
- MLAnalysisGenerationCHILDTI.md (Stage 6) - Consumes trained model files (.pkl) and model_metrics.json

**Implementation Priority**: HIGH

**Rationale**: Stage 5 trains ML models that power the entire insight generation pipeline. Without trained models, Stage 6 cannot extract feature importance or cluster characteristics, and Stage 7 cannot generate creator guidelines. This stage is critical for ML-driven pattern detection and represents ~0.5-1% of total pipeline time (fast but essential).

## Section 2: Stage Contract

### 2.1 Input Contract

```python
# Sources: FoundationCHILD.md Sections 2, 4 | Stage5_MLModelTraining_HLD.md Sections 4.1, 5.1

class Stage5Input:
    """
    Exact structure Stage 5 receives.

    Sources:
    - CLI parameters: FoundationCHILD.md Section 4.1
    - Directory paths: FoundationCHILD.md Section 2.2
    - Stage-specific inputs: Stage5_MLModelTraining_HLD.md Section 4.1
    """

    # ===== CLI PARAMETERS (from FoundationCHILD.md Section 4.1) =====
    client_id: str                  # Required, CLI parameter --client, alphanumeric + underscore
                                    # Example: "acme_corp"
                                    # Validation: Regex ^[a-zA-Z0-9_]+$ (min 1 char)

    analysis_type: str              # Required, CLI parameter --analysis-type
                                    # Valid values: ["hashtag", "competitor", "creator"]
                                    # Example: "hashtag"

    target: str                     # Required, CLI parameter --target
                                    # Format depends on analysis_type (# for hashtag, @ for competitor/creator)
                                    # Example: "#nutrition" or "@rival_brand"

    analysis_mode: str              # Required, CLI parameter --analysis-mode
                                    # Valid values: ["top", "recent"]
                                    # Example: "top"

    selection_strategy: str         # Required, CLI parameter --selection-strategy
                                    # Valid values: ["contrastive", "top"]
                                    # Example: "contrastive"
                                    # Note: Determines baseline for statistical validation (0.80 for contrastive, N/A for top)

    video_count: int                # Required, CLI parameter --video-count
                                    # Range: 10-500
                                    # Example: 100
                                    # Note: Minimum 50 for contrastive, 30 for top mode (Section 6.1)

    bucket: str                     # Required, current bucket being processed
                                    # Valid values: ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
                                    # Example: "18-33s"

    # ===== DIRECTORY PATHS (from FoundationCHILD.md Section 2.2) =====
    client_base: str                # Base client directory
                                    # Template: "/data/clients/{client_id}/"
                                    # Example: "/data/clients/acme_corp/"

    analysis_base: str              # Analysis run directory
                                    # Template: "{client_base}/{analysis_type}s/{target}/{mode}_{strategy}/"
                                    # Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/"

    bucket_base: str                # Bucket-specific directory
                                    # Template: "{analysis_base}/bucket_{bucket}/"
                                    # Example: "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/"

    # ===== STAGE-SPECIFIC INPUTS (Stage5_MLModelTraining_HLD.md Section 4.1) =====

    # Video-Level RF Input
    rf_transformed_csv: str         # Path to video-level RF training data
                                    # Location: "{bucket_base}/ml_analysis/rf_transformed.csv"
                                    # Schema: VideoLevelRFSchema (Section 3.2)
                                    # Shape: (100 videos, ~190 features)
                                    # Source: Stage 4 (Feature Transformation)
                                    # Validation: Must exist, >0 rows (Section 6.1)

    # Window-Level RF Inputs (one per window)
    hook_rf_csv: str                # Location: "{bucket_base}/ml_analysis/hook_rf_transformed.csv"
                                    # Schema: WindowLevelRFSchema (Section 3.2)
                                    # Shape: (100 videos, 22 features)
    middle_1_rf_csv: str            # Location: "{bucket_base}/ml_analysis/middle_1_rf_transformed.csv"
    middle_2_rf_csv: str            # (and so on for all windows in bucket)
    middle_3_rf_csv: str
    middle_4_rf_csv: str            # Only for buckets with 4+ middle segments
    middle_5_rf_csv: str            # Only for buckets with 5 middle segments
    closing_rf_csv: str             # Location: "{bucket_base}/ml_analysis/closing_rf_transformed.csv"

    # K-Means Inputs (one per window)
    hook_km_csv: str                # Location: "{bucket_base}/ml_analysis/hook_km_transformed.csv"
                                    # Schema: KMeansSchema (Section 3.2)
                                    # Shape: (100 videos, 37-39 features)
                                    # Note: 39 with cross-window features, 37 window-only
    middle_1_km_csv: str            # (and so on for all windows)
    middle_2_km_csv: str
    middle_3_km_csv: str
    middle_4_km_csv: str
    middle_5_km_csv: str
    closing_km_csv: str

    # Scalers from Stage 4 (one per window, optional but recommended)
    hook_scalers: str               # Location: "{bucket_base}/ml_analysis/hook_scalers.pkl"
                                    # Format: joblib pickle (sklearn.preprocessing.MinMaxScaler)
                                    # Purpose: Copied to models/ for Stage 6 inference
                                    # Note: If missing, logged as warning but training continues
    middle_1_scalers: str           # (and so on for all windows)
    middle_2_scalers: str
    middle_3_scalers: str
    middle_4_scalers: str
    middle_5_scalers: str
    closing_scalers: str

    # Configuration
    config_file: str                # Location: "{analysis_base}/config.json"
                                    # Schema: FoundationCHILD.md Section 5.1
                                    # Contains: client_id, analysis_type, target, mode, strategy, video_count, etc.

    hyperparameters_config: str     # Location: "config/model_hyperparameters.json" (optional)
                                    # Schema: Stage5_MLModelTraining_HLD.md Section 9.1
                                    # Contains: random_forest params, kmeans params
                                    # Note: If missing, uses hardcoded defaults (no error)

    # Bucket window configuration
    windows: List[str]              # List of temporal windows for this bucket
                                    # Source: config/bucket_definitions.py::BUCKET_WINDOWS[bucket]
                                    # Example for 18-33s: ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"]
                                    # Example for 3-9s: ["hook", "closing"]
```

### 2.2 Output Contract

```python
# Sources: FoundationCHILD.md Section 2.2 | Stage5_MLModelTraining_HLD.md Sections 4.2, 5.2

class Stage5Output:
    """
    Exact structure Stage 5 produces for downstream stages.

    Sources:
    - Output contracts: Stage5_MLModelTraining_HLD.md Section 4.2
    - Output schemas: Stage5_MLModelTraining_HLD.md Section 5.2
    - Directory paths: FoundationCHILD.md Section 2.2
    """

    # ===== OUTPUT FILES =====

    # Video-Level RF Model
    rf_video_model: str             # Path to video-level RF model
                                    # Location: "{bucket_base}/models/rf_video_{bucket}.pkl"
                                    # Example: "/data/clients/acme_corp/.../bucket_18-33s/models/rf_video_18-33s.pkl"
                                    # Format: joblib pickle (sklearn.ensemble.RandomForestClassifier)
                                    # Size: ~1-5 MB (depends on n_estimators, max_depth)
                                    # Consumers: Stage 6 (ML Analysis Generation)

    # Window-Level RF Models (one per window)
    rf_hook_model: str              # Location: "{bucket_base}/models/rf_hook_{bucket}.pkl"
    rf_middle_1_model: str          # Location: "{bucket_base}/models/rf_middle_1_{bucket}.pkl"
    rf_middle_2_model: str          # (and so on for all windows)
    rf_middle_3_model: str
    rf_middle_4_model: str
    rf_middle_5_model: str
    rf_closing_model: str           # Location: "{bucket_base}/models/rf_closing_{bucket}.pkl"
                                    # Format: joblib pickle (sklearn.ensemble.RandomForestClassifier)
                                    # Size: ~500 KB each

    # K-Means Models (one per window)
    hook_kmeans_model: str          # Location: "{bucket_base}/models/hook_kmeans_{bucket}.pkl"
    middle_1_kmeans_model: str      # (and so on for all windows)
    middle_2_kmeans_model: str
    middle_3_kmeans_model: str
    middle_4_kmeans_model: str
    middle_5_kmeans_model: str
    closing_kmeans_model: str       # Location: "{bucket_base}/models/closing_kmeans_{bucket}.pkl"
                                    # Format: joblib pickle (sklearn.cluster.KMeans)
                                    # Size: ~200 KB each

    # K-Means Feature Matrices (for silhouette calculation)
    hook_X_data: str                # Location: "{bucket_base}/models/hook_X_data_{bucket}.pkl"
    middle_1_X_data: str            # (and so on for all windows)
    middle_2_X_data: str
    middle_3_X_data: str
    middle_4_X_data: str
    middle_5_X_data: str
    closing_X_data: str             # Location: "{bucket_base}/models/closing_X_data_{bucket}.pkl"
                                    # Format: joblib pickle (pandas DataFrame)
                                    # Size: ~50 KB each (100 videos × 39 features × 8 bytes)
                                    # Purpose: Required for silhouette score calculation (Section 3, Warning #4)

    # Scalers (for inference)
    hook_scalers: str               # Location: "{bucket_base}/models/hook_scalers_{bucket}.pkl"
    middle_1_scalers: str           # (and so on for all windows)
    middle_2_scalers: str
    middle_3_scalers: str
    middle_4_scalers: str
    middle_5_scalers: str
    closing_scalers: str            # Location: "{bucket_base}/models/closing_scalers_{bucket}.pkl"
                                    # Format: joblib pickle (sklearn.preprocessing.MinMaxScaler)
                                    # Size: ~10 KB each

    # Model Metrics Summary
    model_metrics_json: str         # Location: "{bucket_base}/models/model_metrics.json"
                                    # Schema: ModelMetricsSchema (Section 3.3)
                                    # Format: JSON
                                    # Size: ~5-10 KB
                                    # Purpose: Quick sanity check of model performance (Section 5.2)
                                    # Consumers: Human validation, Stage 6 (optional)

    # ===== OUTPUT SCHEMA DETAILS =====

    # Model Files Count per Bucket
    total_model_files: int          # Total files created per bucket
                                    # For bucket with 6 windows (18-33s):
                                    #   - 1 video-level RF
                                    #   - 6 window-level RF
                                    #   - 6 K-Means
                                    #   - 6 X matrices
                                    #   - 6 scalers
                                    #   - 1 model_metrics.json
                                    # Total: 26 files
                                    # For bucket with 2 windows (3-9s):
                                    #   Total: 10 files

    # Atomic Guarantee (Section 6.4)
    atomic_bucket_training: bool = True  # All models succeed OR all deleted on failure
                                         # No partial model sets allowed

    # ===== EXIT CODES =====
    exit_code_success: int = 0      # All models trained successfully
    exit_code_preflight_fail: int = 1  # Stage 4 outputs missing or invalid
    exit_code_training_fail: int = 2  # Model training failed (NaN values, sklearn error)
    exit_code_validation_fail: int = 3  # Model metrics below threshold (not used in MVP)
    exit_code_io_fail: int = 4      # Disk full, permission denied
    exit_code_partial: int = 5      # Partial completion (not used - atomic training)
    exit_code_data_integrity: int = 6  # Insufficient videos for training
```


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
```

### 3.2 Stage 5 Input Schemas

```python
# Source: Stage5_MLModelTraining_HLD.md Section 5.1

# ===== Video-Level RF Input Schema =====
VideoLevelRFInputSchema = {
    # Source: Stage5_MLModelTraining_HLD.md Section 5.1
    # File: ml_analysis/rf_transformed.csv
    # Shape: (100 videos, ~190 features)

    "video_id": str,               # Required, unique identifier
                                   # Example: "7428596413707144481"
                                   # Note: Will be dropped before training

    # Temporal window features (hook, middle_1-5, closing)
    # NOTE: Each window has 21 base features (see WindowLevelRFInputSchema below for complete list)
    # These features are prefixed with window name (e.g., "hook_", "middle_1_", "closing_")
    # Full feature list documented in FeatureTransformationCHILDTI.md Section 3.2 (Output Schema)

    "hook_scene_count": int,       # Required, Range: 0-20, Hook window scene cuts
    "hook_eye_contact_rate": float,  # Required, Range: 0.0-1.0, Hook window eye contact
    "hook_word_count": int,        # Required, Range: 0-100, Hook window word count
    "hook_speech_coverage": float, # Required, Range: 0.0-1.0, Hook window speech coverage
    "hook_energy_level": float,    # Required, Range: 0.0-1.0, Hook window audio energy
    # ... (remaining 16 hook features follow WindowLevelRFInputSchema pattern below)

    "middle_1_scene_count": int,   # Required (if bucket has middle segments), Range: 0-20
    "middle_1_eye_contact_rate": float,  # Required (if bucket has middle segments), Range: 0.0-1.0
    # ... (remaining 19 middle_1 features follow same 21-feature pattern)

    "middle_2_scene_count": int,   # Required (if bucket has 2+ middle segments)
    # ... (21-feature pattern continues for middle_2)

    "middle_3_scene_count": int,   # Required (if bucket has 3+ middle segments)
    # ... (21-feature pattern continues for middle_3)

    "middle_4_scene_count": int,   # Required (if bucket has 4+ middle segments)
    # ... (21-feature pattern continues for middle_4)

    "middle_5_scene_count": int,   # Required (if bucket has 5 middle segments)
    # ... (21-feature pattern continues for middle_5)

    "closing_scene_count": int,    # Required, Range: 0-20, Closing window scene cuts
    "closing_eye_contact_rate": float,  # Required, Range: 0.0-1.0, Closing window eye contact
    # ... (remaining 19 closing features follow same 21-feature pattern)

    # Video-level derived features
    "hour": int,                   # Required, Range: 0-23, Derived from create_time
    "day_of_week": int,            # Required, Range: 0-6, Monday=0, Sunday=6
    "is_weekend": int,             # Required, 0 or 1, Derived from day_of_week

    # Gender (one-hot encoded)
    "gender_male": int,            # Required, 0 or 1, One-hot encoded
    "gender_female": int,          # Required, 0 or 1, One-hot encoded

    # Target variable
    "is_top_performer": int,       # Required, 0 or 1, Target variable for classification
                                   # 1 = top 80% (in contrastive mode)
                                   # 0 = bottom 20% (in contrastive mode)
                                   # Note: In "top" mode, all videos are 1 (RF training skipped)
}

# ===== Analysis Mode Behavior =====
# Source: Design decision from planning phase
#
# 'contrastive' mode: Top 80% vs Bottom 20%
#   - Trains both Random Forest and K-Means
#   - RF learns what differentiates high from low performers
#   - K-Means finds content style clusters across both performance groups
#
# 'top' mode: Top N videos only
#   - Trains K-Means only (RF skipped)
#   - RF cannot train with single class (all videos are winners)
#   - K-Means finds content style clusters among successful videos
#   - Use case: "What styles exist among winners?" vs "What makes winners different?"

# ===== Window-Level RF Input Schema =====
WindowLevelRFInputSchema = {
    # Source: Stage5_MLModelTraining_HLD.md Section 5.1
    # File: ml_analysis/hook_rf_transformed.csv (and similar for all windows)
    # Shape: (100 videos, 22 features)

    # Base features (21 features per window)
    "scene_count": int,            # Required, Range: 0-20, Scene cuts in window
    "eye_contact_rate": float,     # Required, Range: 0.0-1.0, Proportion of frames with eye contact
    "word_count": int,             # Required, Range: 0-100, Total words spoken
    "speech_coverage": float,      # Required, Range: 0.0-1.0, Proportion of window with speech
    "energy_level": float,         # Required, Range: 0.0-1.0, Audio RMS energy
    "energy_max": float,           # Required, Range: 0.0-1.0, Maximum energy in window
    "has_captions": int,           # Required, 0 or 1, Presence of text overlays
    "person_count": int,           # Required, Range: 0-10, Average people detected
    "close_ratio": float,          # Required, Range: 0.0-1.0, Close-up shot proportion
    "element_count": int,          # Required, Range: 0-100, Objects detected
    "joy_ratio": float,            # Required, Range: 0.0-1.0, Joy emotion proportion
    "surprise_ratio": float,       # Required, Range: 0.0-1.0, Surprise emotion proportion
    "anger_ratio": float,          # Required, Range: 0.0-1.0, Anger emotion proportion
    "disgust_ratio": float,        # Required, Range: 0.0-1.0, Disgust emotion proportion
    "fear_ratio": float,           # Required, Range: 0.0-1.0, Fear emotion proportion
    "sadness_ratio": float,        # Required, Range: 0.0-1.0, Sadness emotion proportion
    "neutral_ratio": float,        # Required, Range: 0.0-1.0, Neutral emotion proportion
    "has_greeting": int,           # Required, 0 or 1, Presence of greeting words
    "has_cta": int,                # Required, 0 or 1, Presence of call-to-action
    "has_question": int,           # Required, 0 or 1, Presence of questions
    "avg_pitch_normalized": float, # Required, Range: 0.0-2.0, Normalized by gender baseline

    # Target variable
    "is_top_performer": int,       # Required, 0 or 1, Target variable for classification
}

# ===== K-Means Input Schema =====
KMeansInputSchema = {
    # Source: Stage5_MLModelTraining_HLD.md Section 5.1
    # File: ml_analysis/hook_km_transformed.csv (and similar for all windows)
    # Shape: (100 videos, 37-39 features)
    # Note: 39 features includes 2 optional cross-window features (density_change, energy_change)
    #       37 features for window-only analysis (no cross-window comparison)

    "video_id": str,               # Required, unique identifier
                                   # Example: "7428596413707144481"
                                   # Note: Will be dropped before training

    # Scaled features (all numerical, scaled to [0-1])
    "eye_contact_rate_scaled": float,      # Required, Range: 0.0-1.0, MinMax scaled
    "scene_count_scaled": float,           # Required, Range: 0.0-1.0, MinMax scaled
    "word_count_log_scaled": float,        # Required, Range: 0.0-1.0, log1p + MinMax scaled
    "speech_coverage_scaled": float,       # Required, Range: 0.0-1.0, MinMax scaled
    "energy_level_scaled": float,          # Required, Range: 0.0-1.0, MinMax scaled
    "energy_max_scaled": float,            # Required, Range: 0.0-1.0, MinMax scaled
    "person_count_log_scaled": float,      # Required, Range: 0.0-1.0, log1p + MinMax scaled
    "close_ratio_scaled": float,           # Required, Range: 0.0-1.0, MinMax scaled
    "element_count_log_scaled": float,     # Required, Range: 0.0-1.0, log1p + MinMax scaled
    "joy_ratio_scaled": float,             # Required, Range: 0.0-1.0, MinMax scaled
    "surprise_ratio_scaled": float,        # Required, Range: 0.0-1.0, MinMax scaled
    "anger_ratio_scaled": float,           # Required, Range: 0.0-1.0, MinMax scaled
    "disgust_ratio_scaled": float,         # Required, Range: 0.0-1.0, MinMax scaled
    "fear_ratio_scaled": float,            # Required, Range: 0.0-1.0, MinMax scaled
    "sadness_ratio_scaled": float,         # Required, Range: 0.0-1.0, MinMax scaled
    "neutral_ratio_scaled": float,         # Required, Range: 0.0-1.0, MinMax scaled
    "avg_pitch_normalized_scaled": float,  # Required, Range: 0.0-1.0, MinMax scaled

    # Encoded features (one-hot or label encoded)
    "has_captions_encoded": int,   # Required, 0 or 1, Label encoded
    "has_greeting_encoded": int,   # Required, 0 or 1, Label encoded
    "has_cta_encoded": int,        # Required, 0 or 1, Label encoded
    "has_question_encoded": int,   # Required, 0 or 1, Label encoded
    "gender_male": int,            # Required, 0 or 1, One-hot encoded
    "gender_female": int,          # Required, 0 or 1, One-hot encoded
    "hour_sin": float,             # Required, Range: -1.0-1.0, Sine-encoded hour
    "hour_cos": float,             # Required, Range: -1.0-1.0, Cosine-encoded hour
    "day_of_week_sin": float,      # Required, Range: -1.0-1.0, Sine-encoded day
    "day_of_week_cos": float,      # Required, Range: -1.0-1.0, Cosine-encoded day
    "is_weekend_encoded": int,     # Required, 0 or 1, Label encoded

    # Cross-window features (only in video-level K-Means, not window-level)
    "energy_progression": float,   # Optional, Range: -1.0-1.0, hook→closing energy change
    "topic_consistency": float,    # Optional, Range: 0.0-1.0, Word overlap across windows
    "weak_link_energy": float,     # Optional, Range: 0.0-1.0, Minimum energy across windows

    # Note: NO target variable (unsupervised clustering)
}
```

### 3.3 Stage 5 Output Schema

```python
# Source: Stage5_MLModelTraining_HLD.md Section 5.2

# ===== Model Metrics JSON Schema =====
ModelMetricsSchema = {
    # Source: Stage5_MLModelTraining_HLD.md Section 5.2
    # File: models/model_metrics.json
    # Purpose: Quick sanity check of model performance after training

    "bucket": str,                 # Required, bucket name, Example: "18-33s"
    "total_videos": int,           # Required, number of videos trained on, Example: 100

    "video_level_rf": {            # Required, video-level Random Forest metrics
        "model_type": str,         # Required, always "random_forest"
        "trained": bool,           # Required, True if RF trained, False if skipped
        # If trained=True:
        "input_features": int,     # Required (if trained), number of features used, Example: 190
        "accuracy": float,         # Required (if trained), Range: 0.0-1.0, classification accuracy
        "precision": float,        # Required (if trained), Range: 0.0-1.0, precision score
        "recall": float,           # Required (if trained), Range: 0.0-1.0, recall score
        "f1_score": float,         # Required (if trained), Range: 0.0-1.0, F1 score
        "top_feature": str,        # Required (if trained), most important feature name
        "top_feature_importance": float,  # Required (if trained), Range: 0.0-1.0, Gini importance
        "purpose": str,            # Required, always "Cross-window pattern detection"
        # If trained=False:
        "skip_reason": str,        # Required (if not trained), Explanation, Example: "Single class in 'top' mode"
    },

    "window_level_rf": {           # Required, window-level Random Forest metrics (one per window)
        "hook": {
            "model_type": str,     # Required, always "random_forest"
            "trained": bool,       # Required, True if RF trained, False if skipped
            # If trained=True:
            "input_features": int, # Required (if trained), always 21 for window-level
            "accuracy": float,     # Required (if trained), Range: 0.0-1.0
            "precision": float,    # Required (if trained), Range: 0.0-1.0
            "recall": float,       # Required (if trained), Range: 0.0-1.0
            "f1_score": float,     # Required (if trained), Range: 0.0-1.0
            "top_feature": str,    # Required (if trained), most important feature name (no window prefix)
            "top_feature_importance": float,  # Required (if trained), Range: 0.0-1.0
            # If trained=False:
            "skip_reason": str,    # Required (if not trained), Example: "Single class in 'top' mode"
        },
        "middle_1": {},            # Same structure as hook (if bucket has middle segments)
        "middle_2": {},            # (if bucket has 2+ middle segments)
        "middle_3": {},            # (if bucket has 3+ middle segments)
        "middle_4": {},            # (if bucket has 4+ middle segments)
        "middle_5": {},            # (if bucket has 5 middle segments)
        "closing": {},             # Same structure as hook
    },

    "window_level_kmeans": {       # Required, K-Means metrics (one per window)
        "hook": {
            "model_type": str,     # Required, always "kmeans"
            "input_features": int, # Required, ~39 for window-level K-Means
            "n_clusters": int,     # Required, always 3 (from config)
            "inertia": float,      # Required, sum of squared distances to centroids
            "silhouette_score": float,  # Required, Range: -1.0-1.0, cluster quality
            "cluster_sizes": list[int],  # Required, 3 integers, videos per cluster
        },
        "middle_1": {},            # Same structure as hook
        "middle_2": {},
        "middle_3": {},
        "middle_4": {},
        "middle_5": {},
        "closing": {},
    },
}
```

**Field Count Validation**:
- Video-Level RF Input: ~190 fields (varies by bucket window count)
- Window-Level RF Input: 22 fields (21 base + 1 target)
- K-Means Input: ~39 fields (varies slightly based on cross-window features)
- Model Metrics Output: ~60-80 fields (depends on bucket window count)

## Section 4: Algorithmic Specifications

**Source**: Stage5_MLModelTraining_HLD.md Section 2.3 (Detailed Process)

### 4.1 Function: validate_stage4_outputs()

**Purpose**: Validate all Stage 4 files exist before training ANY models (fail-fast on missing files)

**Implementation** (from Stage5_MLModelTraining_HLD.md Section 2.3.1):

```python
def validate_stage4_outputs(bucket: str, windows: List[str], bucket_base: str, selection_strategy: str) -> None:
    """
    Validate all Stage 4 files exist before training ANY models.

    Fail-fast strategy: If ANY file is missing or empty, raise error immediately.
    This prevents partial model training when upstream data is incomplete.

    Args:
        bucket: str - Bucket name (e.g., "18-33s")
        windows: List[str] - List of window names for this bucket (from BUCKET_WINDOWS)
        bucket_base: str - Base path to bucket directory
        selection_strategy: str - "contrastive" or "top" (determines min video count threshold)

    Raises:
        StageInputError: If any required file is missing or empty

    Source: Stage5_MLModelTraining_HLD.md Section 2.3.1
    """
    import os
    import pandas as pd

    # Step 1: Build list of required files
    required_files = [
        # Video-level RF
        os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'),
    ]

    # Window-level RF files (one per window)
    for window in windows:
        required_files.append(
            os.path.join(bucket_base, f'ml_analysis/{window}_rf_transformed.csv')
        )

    # K-Means files (one per window)
    for window in windows:
        required_files.append(
            os.path.join(bucket_base, f'ml_analysis/{window}_km_transformed.csv')
        )

    # Step 2: Check file existence
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise StageInputError(
                f"Stage 4 incomplete: Missing {file_path}. "
                f"Run Stage 4 first or check if bucket {bucket} was skipped in Stage 1."
            )

    # Step 3: Check file is not empty
    for file_path in required_files:
        df = pd.read_csv(file_path)
        if df.shape[0] == 0:
            raise StageInputError(
                f"Stage 4 output empty: {file_path} has 0 rows. "
                f"This indicates Stage 4 failed silently."
            )

    # Step 4: Validate video count meets minimum threshold
    video_count = pd.read_csv(required_files[0]).shape[0]
    min_required = 50 if selection_strategy == "contrastive" else 30

    if video_count < min_required:
        raise InsufficientDataError(
            f"Bucket {bucket} has {video_count} videos "
            f"(min {min_required} required for {selection_strategy} mode). "
            f"Re-run Stage 1 with lower --video-count or skip this bucket."
        )

    # Step 5: Validate K-Means feature naming convention (Section 6.1 validation #5)
    # Note: validate_kmeans_feature_naming() defined in Section 5.2 below
    for window in windows:
        km_csv_path = os.path.join(bucket_base, f'ml_analysis/{window}_km_transformed.csv')
        validate_kmeans_feature_naming(km_csv_path)

    logger.info(f"✓ Stage 4 validation passed: {len(required_files)} files exist, {video_count} videos")
```

**Edge Cases** (from Stage5_MLModelTraining_HLD.md Section 2.3.1):

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Stage 4 file missing | Fail-fast: raise StageInputError | Cannot train without transformed features |
| Stage 4 file empty (0 rows) | Fail-fast: raise StageInputError | Indicates Stage 4 silent failure |
| Insufficient videos (< 50 contrastive, < 30 top) | Fail-fast: raise InsufficientDataError | ML models need minimum sample size for reliability |
| K-Means features missing _scaled suffix | Fail-fast: raise ValidationError | Indicates Stage 4 didn't apply transformations (Critical Warning #1) |

**Validation Rules**:
- ALL files must exist (no partial Stage 4 completion)
- ALL files must have >0 rows
- Video count must meet minimum threshold
- K-Means features must have transformation suffixes (_scaled, _log, _encoded)

---

### 4.2 Function: load_model_config()

**Purpose**: Load hyperparameters from config with fallback to hardcoded defaults

**Implementation** (from Stage5_MLModelTraining_HLD.md Section 2.3.2):

```python
def load_model_config() -> dict:
    """
    Load hyperparameters from config with fallback to hardcoded defaults.

    Graceful degradation: If config file is missing, use hardcoded defaults (log warning).
    If config file is malformed, raise error (fail-fast).

    Returns:
        dict: Hyperparameters for RandomForest and KMeans
              {
                  "random_forest": {"n_estimators": int, "max_depth": int, "random_state": int},
                  "kmeans": {"n_clusters": int, "random_state": int, "n_init": int}
              }

    Raises:
        ConfigError: If config file exists but is malformed (invalid JSON, missing required keys)

    Source: Stage5_MLModelTraining_HLD.md Section 2.3.2
    """
    import json
    import os

    config_path = 'config/model_hyperparameters.json'

    # Step 1: Hardcoded defaults (fallback)
    default_config = {
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

    # Step 2: Try to load config file
    if not os.path.exists(config_path):
        logger.warning(
            f"Config file not found: {config_path}. Using hardcoded defaults."
        )
        return default_config

    # Step 3: Parse config file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Step 4: Validate required keys exist
        if 'random_forest' not in config or 'kmeans' not in config:
            raise ConfigError(
                f"Invalid {config_path}: Missing required keys 'random_forest' or 'kmeans'"
            )

        # Step 5: Validate RandomForest params
        rf_params = config['random_forest']
        if 'n_estimators' not in rf_params or 'max_depth' not in rf_params:
            raise ConfigError(
                f"Invalid {config_path}: RandomForest missing 'n_estimators' or 'max_depth'"
            )

        # Step 6: Validate KMeans params
        km_params = config['kmeans']
        if 'n_clusters' not in km_params:
            raise ConfigError(
                f"Invalid {config_path}: KMeans missing 'n_clusters'"
            )

        logger.info(f"✓ Loaded hyperparameters from {config_path}")
        return config

    except json.JSONDecodeError as e:
        raise ConfigError(
            f"Invalid JSON in {config_path}: {e}. Fix JSON syntax or delete file to use defaults."
        )
```

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Config file missing | Use hardcoded defaults + log warning | Allow Stage 5 to run without config file |
| Config file malformed JSON | Fail-fast: raise ConfigError | Invalid JSON is user error, not recoverable |
| Config missing required keys | Fail-fast: raise ConfigError | Partial config is worse than no config |

---

### 4.3 Function: train_bucket_models()

**Purpose**: Train all models for bucket (atomic: all succeed or all deleted)

**Implementation** (from Stage5_MLModelTraining_HLD.md Section 2.3.3):

```python
def train_bucket_models(
    bucket: str,
    windows: List[str],
    bucket_base: str,
    config: dict,
    selection_strategy: str
) -> None:
    """
    Train all models for bucket. Atomic operation: all succeed or all deleted.

    Training order:
    1. Video-level RF (1 model)
    2. Window-level RF (6 models for bucket 18-33s)
    3. K-Means (6 models + 6 X matrices + 6 scalers)

    On failure: Delete ALL partial models for this bucket (atomic rollback).

    Args:
        bucket: str - Bucket name (e.g., "18-33s")
        windows: List[str] - List of window names (e.g., ["hook", "middle_1", ..., "closing"])
        bucket_base: str - Base path to bucket directory
        config: dict - Hyperparameters from load_model_config()
        selection_strategy: str - "contrastive" or "top" (affects baseline validation)

    Raises:
        ModelTrainingError: If any model training fails

    Source: Stage5_MLModelTraining_HLD.md Section 2.3.3
    """
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    import time
    import os
    import json

    trained_models = []  # Track all created files for atomic rollback
    start_time = time.time()

    # Tracking structures for generate_model_metrics()
    rf_window_models = {}
    kmeans_models = {}
    X_data_matrices = {}
    total_videos = 0
    rf_video = None  # Track if RF was trained (None if skipped)

    try:
        # ===== STEP 0: Check if RF training is possible =====
        # Source: C7 fix - RF requires binary classification (2 classes minimum)
        X_check = pd.read_csv(os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'))
        unique_labels = X_check['is_top_performer'].unique()
        can_train_rf = len(unique_labels) >= 2

        if not can_train_rf:
            logger.info(
                f"Skipping Random Forest for {bucket}: "
                f"Single class detected in '{selection_strategy}' mode (expected for 'top' mode). "
                f"K-Means only."
            )

        # ===== STEP 1: Train Video-Level RF (if binary classification possible) =====
        if can_train_rf:
            logger.info(f"Training video-level RF for {bucket}...")

            # Load data
            X = pd.read_csv(os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'))
            y = X['is_top_performer']
            X = X.drop(['is_top_performer', 'video_id'], axis=1)
            total_videos = len(y)

            # Train model
            rf_video = RandomForestClassifier(**config['random_forest'])
            rf_video.fit(X, y)

            # Save model
            model_path = os.path.join(bucket_base, f'models/rf_video_{bucket}.pkl')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(rf_video, model_path)
            trained_models.append(model_path)

            logger.info(f"✓ Video-level RF trained: {model_path}")
        else:
            # Still need total_videos for metrics
            X = pd.read_csv(os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'))
            total_videos = len(X)

        # ===== STEP 2: Train Window-Level RF (if binary classification possible, sequential loop) =====
        if can_train_rf:
            for window in windows:
                logger.info(f"Training window-level RF for {window}...")

                # Load data
                X = pd.read_csv(os.path.join(bucket_base, f'ml_analysis/{window}_rf_transformed.csv'))
                y = X['is_top_performer']
                X = X.drop(['is_top_performer'], axis=1)

                # Train model
                rf_window = RandomForestClassifier(**config['random_forest'])
                rf_window.fit(X, y)

                # Save model
                model_path = os.path.join(bucket_base, f'models/rf_{window}_{bucket}.pkl')
                joblib.dump(rf_window, model_path)
                trained_models.append(model_path)

                # Store for metrics generation
                rf_window_models[window] = rf_window

                logger.info(f"✓ Window-level RF trained: {window}")

        # ===== STEP 3: Train K-Means (sequential loop) =====
        for window in windows:
            logger.info(f"Training K-Means for {window}...")

            # Load data
            X = pd.read_csv(os.path.join(bucket_base, f'ml_analysis/{window}_km_transformed.csv'))
            X = X.drop(['video_id'], axis=1)  # No labels for K-Means

            # Train model
            kmeans = KMeans(**config['kmeans'])
            kmeans.fit(X)

            # Save model
            model_path = os.path.join(bucket_base, f'models/{window}_kmeans_{bucket}.pkl')
            joblib.dump(kmeans, model_path)
            trained_models.append(model_path)

            # Save X matrix (for silhouette calculation - Critical Warning #4)
            X_path = os.path.join(bucket_base, f'models/{window}_X_data_{bucket}.pkl')
            joblib.dump(X, X_path)
            trained_models.append(X_path)

            # Store for metrics generation
            kmeans_models[window] = kmeans
            X_data_matrices[window] = X

            # Save scalers (for inference - Stage 6 dependency)
            # Note: Scalers are loaded from Stage 4 output, not re-fitted here
            scaler_source = os.path.join(bucket_base, f'ml_analysis/{window}_scalers.pkl')
            scaler_dest = os.path.join(bucket_base, f'models/{window}_scalers_{bucket}.pkl')
            if os.path.exists(scaler_source):
                joblib.dump(joblib.load(scaler_source), scaler_dest)
                trained_models.append(scaler_dest)
            else:
                logger.warning(f"Scaler file missing for {window}: {scaler_source}. Skipping scaler copy.")

            logger.info(f"✓ K-Means trained: {window}")

        # ===== STEP 4: Generate model_metrics.json =====
        metrics = generate_model_metrics(
            bucket=bucket,
            windows=windows,
            bucket_base=bucket_base,
            rf_video_model=rf_video,
            rf_window_models=rf_window_models,
            kmeans_models=kmeans_models,
            X_data_matrices=X_data_matrices,
            total_videos=total_videos
        )
        metrics_path = os.path.join(bucket_base, 'models/model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        trained_models.append(metrics_path)

        # ===== SUCCESS =====
        elapsed = time.time() - start_time
        logger.info(
            f"✓ Bucket {bucket} training complete: {elapsed:.1f}s ({len(trained_models)} files)"
        )

        # Performance warning (no hard timeout, but log if suspiciously slow)
        if elapsed > 300:  # 5 minutes per bucket
            logger.warning(
                f"Bucket {bucket} training took {elapsed:.1f}s (expected <120s). "
                f"Check for performance issues."
            )

    except Exception as e:
        # ===== FAILURE: Atomic rollback =====
        logger.error(f"""
Bucket {bucket} training failed
Exception: {type(e).__name__}: {str(e)}
Stack trace: {traceback.format_exc(limit=10)}
Completed models before failure: {len(trained_models)} files
Training duration before failure: {time.time() - start_time:.1f}s

Performing atomic rollback: Deleting all {len(trained_models)} partial models...
""")

        # Delete all partial models
        for model_path in trained_models:
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"  Deleted: {model_path}")

        logger.error(f"Rollback complete. {len(trained_models)} files deleted.")

        raise ModelTrainingError(
            f"Bucket {bucket} training failed: {e}. "
            f"All {len(trained_models)} partial models deleted. "
            f"Fix data issue and re-run Stage 5."
        )
```

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Training fails mid-bucket (e.g., NaN values) | Atomic rollback: delete ALL models | Prevents partial model sets (Section 6.4) |
| Training takes > 5 minutes per bucket | Log warning, continue | No hard timeout (Q9 decision), but suspicious |
| Scaler file missing from Stage 4 | Skip scaler save (log warning) | Non-critical for training, only needed for inference |

**Validation Rules**:
- All trained models must succeed (atomic guarantee)
- All model files must be saved to disk (no in-memory-only models)
- model_metrics.json must be generated last (indicates completion)

---

### 4.4 Function: normalize_feature_name()

**Purpose**: Normalize K-Means feature names for comparison with RF feature names (Critical Warning #1)

**Implementation** (from Stage5_MLModelTraining_HLD.md Section 3, Critical Warning #1):

```python
def normalize_feature_name(feature_name: str) -> str:
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

    Source: Stage5_MLModelTraining_HLD.md Section 3, Critical Warning #1
    """
    normalized = feature_name

    # Remove suffixes in order (some features may have multiple)
    suffixes = ['_scaled', '_log', '_encoded']
    for suffix in suffixes:
        normalized = normalized.replace(suffix, '')

    return normalized
```

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Feature already normalized (no suffix) | Return unchanged | Idempotent operation |
| Feature has multiple suffixes (e.g., '_log_scaled') | Remove all | Handles Stage 4 edge cases |

---

### 4.5 Function: get_top_cluster_features()

**Purpose**: Extract top N cluster-defining features from K-Means model (Critical Warning #2)

**Implementation** (from Stage5_MLModelTraining_HLD.md Section 3, Critical Warning #2):

```python
def get_top_cluster_features(kmeans_model, feature_names: List[str], n: int = 5) -> List[str]:
    """
    Extract top N cluster-defining features from K-Means model.

    Uses variance across cluster centroids as the ranking metric.
    Features with high variance distinguish clusters; features with
    low variance do not.

    Args:
        kmeans_model: Trained sklearn.cluster.KMeans object
        feature_names: List of feature names (must match order of centroids)
        n: Number of top features to return (default 5)

    Returns:
        List of top N feature names, sorted by importance (highest variance first)

    Example:
        >>> kmeans = KMeans(n_clusters=3, random_state=42)
        >>> kmeans.fit(X)  # X shape: (100, 39)
        >>> feature_names = ['eye_contact_rate_scaled', 'scene_count_scaled', ...]
        >>> top_5 = get_top_cluster_features(kmeans, feature_names, n=5)
        >>> print(top_5)
        ['eye_contact_rate_scaled', 'scene_count_scaled', 'energy_level_scaled', ...]

    Source: Stage5_MLModelTraining_HLD.md Section 3, Critical Warning #2
    """
    import numpy as np

    # Step 1: Get cluster centroids
    centroids = kmeans_model.cluster_centers_  # Shape: (n_clusters, n_features)

    # Step 2: Calculate variance of each feature across centroids
    # High variance = feature values differ across clusters = cluster-defining
    feature_variances = np.var(centroids, axis=0)  # Shape: (n_features,)

    # Step 3: Rank features by variance (highest first)
    top_indices = np.argsort(feature_variances)[::-1][:min(n, len(feature_names))]

    # Step 4: Map indices to feature names
    top_features = [feature_names[i] for i in top_indices]

    return top_features
```

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| All features have same variance | Return arbitrary order (no crash) | Rare edge case, indicates no cluster separation |
| n > number of features | Return all features (no error) | Graceful degradation |
| Empty feature list | Return empty list | Defensive programming |

---

### 4.6 Function: generate_model_metrics()

**Purpose**: Generate model_metrics.json with performance metrics for all trained models

**Implementation**:

```python
def generate_model_metrics(bucket: str, windows: List[str], bucket_base: str,
                          rf_video_model, rf_window_models: dict,
                          kmeans_models: dict, X_data_matrices: dict,
                          total_videos: int) -> dict:
    """
    Generate comprehensive metrics for all trained models.

    Args:
        bucket: str - Bucket name (e.g., "18-33s")
        windows: List[str] - Window names for this bucket
        bucket_base: str - Base path to bucket directory
        rf_video_model: Trained video-level RandomForestClassifier or None (if skipped)
        rf_window_models: dict - {window: trained RF model} or {} (if RF skipped)
        kmeans_models: dict - {window: trained KMeans model}
        X_data_matrices: dict - {window: feature DataFrame}
        total_videos: int - Number of videos trained on

    Returns:
        dict: Metrics matching ModelMetricsSchema (Section 3.3)

    Source: Stage5_MLModelTraining_HLD.md Section 5.2
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import silhouette_score
    import numpy as np

    metrics = {
        "bucket": bucket,
        "total_videos": total_videos,
        "video_level_rf": {},
        "window_level_rf": {},
        "window_level_kmeans": {}
    }

    # ===== Video-Level RF Metrics (if trained) =====
    # Source: C7 fix - RF may be None in 'top' mode
    if rf_video_model is not None:
        # Load training data to compute metrics
        X_video = pd.read_csv(os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'))
        y_video = X_video['is_top_performer']
        X_video = X_video.drop(['is_top_performer', 'video_id'], axis=1)

        y_pred_video = rf_video_model.predict(X_video)

        # Get top feature
        feature_importances = rf_video_model.feature_importances_
        top_feature_idx = np.argmax(feature_importances)
        top_feature_name = X_video.columns[top_feature_idx]

        metrics["video_level_rf"] = {
            "model_type": "random_forest",
            "trained": True,
            "input_features": X_video.shape[1],
            "accuracy": float(accuracy_score(y_video, y_pred_video)),
            "precision": float(precision_score(y_video, y_pred_video, zero_division=0)),
            "recall": float(recall_score(y_video, y_pred_video, zero_division=0)),
            "f1_score": float(f1_score(y_video, y_pred_video, zero_division=0)),
            "top_feature": str(top_feature_name),
            "top_feature_importance": float(feature_importances[top_feature_idx]),
            "purpose": "Cross-window pattern detection"
        }
    else:
        # RF skipped (single class in 'top' mode)
        metrics["video_level_rf"] = {
            "model_type": "random_forest",
            "trained": False,
            "skip_reason": "Single class in dataset (expected in 'top' mode)",
            "purpose": "Cross-window pattern detection"
        }

    # ===== Window-Level RF Metrics (if trained) =====
    # Source: C7 fix - rf_window_models will be empty dict if RF skipped
    if rf_window_models:  # Non-empty dict means RF was trained
        for window in windows:
            X_window = pd.read_csv(os.path.join(bucket_base, f'ml_analysis/{window}_rf_transformed.csv'))
            y_window = X_window['is_top_performer']
            X_window = X_window.drop(['is_top_performer'], axis=1)

            rf_model = rf_window_models[window]
            y_pred_window = rf_model.predict(X_window)

            # Get top feature (no window prefix in feature name)
            feature_importances = rf_model.feature_importances_
            top_feature_idx = np.argmax(feature_importances)
            top_feature_name = X_window.columns[top_feature_idx]

            metrics["window_level_rf"][window] = {
                "model_type": "random_forest",
                "trained": True,
                "input_features": X_window.shape[1],
                "accuracy": float(accuracy_score(y_window, y_pred_window)),
                "precision": float(precision_score(y_window, y_pred_window, zero_division=0)),
                "recall": float(recall_score(y_window, y_pred_window, zero_division=0)),
                "f1_score": float(f1_score(y_window, y_pred_window, zero_division=0)),
                "top_feature": str(top_feature_name),
                "top_feature_importance": float(feature_importances[top_feature_idx])
            }
    else:
        # RF skipped - add placeholder for each window
        for window in windows:
            metrics["window_level_rf"][window] = {
                "model_type": "random_forest",
                "trained": False,
                "skip_reason": "Single class in dataset (expected in 'top' mode)"
            }

    # ===== Window-Level K-Means Metrics =====
    for window in windows:
        kmeans_model = kmeans_models[window]
        X_kmeans = X_data_matrices[window]

        # Compute cluster assignments
        labels = kmeans_model.labels_

        # Compute cluster sizes
        cluster_sizes = [int(np.sum(labels == i)) for i in range(kmeans_model.n_clusters)]

        # Compute silhouette score
        silhouette = float(silhouette_score(X_kmeans, labels))

        metrics["window_level_kmeans"][window] = {
            "model_type": "kmeans",
            "input_features": X_kmeans.shape[1],
            "n_clusters": int(kmeans_model.n_clusters),
            "inertia": float(kmeans_model.inertia_),
            "silhouette_score": silhouette,
            "cluster_sizes": cluster_sizes
        }

    return metrics
```

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| All predictions same class | precision/recall use zero_division=0 | Prevents divide-by-zero |
| Cluster sizes unbalanced | Log in metrics, warn in validation | Indicates potential data issue |
| Negative silhouette score | Log as-is (valid metric) | Negative scores indicate poor clustering |

---

## Section 5: Validation Rules

**Source**: Stage5_MLModelTraining_HLD.md Section 6.1 (Input Validation)

### 5.1 Input Validation

```python
# Source: Stage5_MLModelTraining_HLD.md Section 6.1: Input Validation

def validate_stage_input(bucket: str, windows: List[str], bucket_base: str,
                        selection_strategy: str, video_count: int) -> None:
    """
    Validate input before processing.

    Source: Stage5_MLModelTraining_HLD.md Section 6.1

    Validation layers:
    1. File existence (all Stage 4 outputs present)
    2. File non-empty (>0 rows)
    3. Video count threshold (min 50 contrastive, 30 top)
    4. K-Means feature naming convention (>=80% have transformation suffixes)
    """

    # ===== LAYER 1: File Existence =====
    required_files = [
        os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'),
    ]

    for window in windows:
        required_files.append(
            os.path.join(bucket_base, f'ml_analysis/{window}_rf_transformed.csv')
        )
        required_files.append(
            os.path.join(bucket_base, f'ml_analysis/{window}_km_transformed.csv')
        )

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise ValidationError(
                f"Stage 4 incomplete: Missing {file_path}. Run Stage 4 first."
            )

    # ===== LAYER 2: File Non-Empty =====
    for file_path in required_files:
        df = pd.read_csv(file_path)
        if df.shape[0] == 0:
            raise ValidationError(
                f"Stage 4 output empty: {file_path} has 0 rows. Stage 4 failed silently."
            )

    # ===== LAYER 3: Video Count Threshold =====
    MIN_VIDEOS_CONTRASTIVE = 50  # 40 top + 10 bottom (bare minimum for 80/20 split)
    MIN_VIDEOS_TOP = 30          # Descriptive analysis only

    min_required = MIN_VIDEOS_CONTRASTIVE if selection_strategy == "contrastive" else MIN_VIDEOS_TOP

    if video_count < min_required:
        raise ValidationError(
            f"Bucket {bucket} has {video_count} videos "
            f"(min {min_required} required for {selection_strategy} mode). "
            f"Re-run Stage 1 with lower --video-count or skip this bucket."
        )

    # ===== LAYER 4: Label Distribution Validation (RF Compatibility) =====
    # Source: C7 fix - Validate binary classification is possible
    rf_csv_path = os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv')
    df_rf = pd.read_csv(rf_csv_path)
    unique_labels = df_rf['is_top_performer'].unique()

    if len(unique_labels) < 2:
        if selection_strategy == 'contrastive':
            raise ValidationError(
                f"Bucket {bucket}: Only one class found ({unique_labels[0]}) in 'contrastive' mode. "
                f"Random Forest requires both top and bottom performers. "
                f"Stage 4 data preparation may have failed. Check Stage 4 outputs."
            )
        else:  # selection_strategy == 'top'
            logger.info(
                f"Bucket {bucket}: Single class detected in 'top' mode (expected). "
                f"Random Forest training will be skipped. K-Means only."
            )

    # ===== LAYER 5: K-Means Feature Naming Convention =====
    # Source: Stage5_MLModelTraining_HLD.md Section 6.1, Validation #5
    # Note: validate_kmeans_feature_naming() defined immediately below
    for window in windows:
        km_csv_path = os.path.join(bucket_base, f'ml_analysis/{window}_km_transformed.csv')
        validate_kmeans_feature_naming(km_csv_path)


def validate_kmeans_feature_naming(csv_path: str, expected_suffix: str = '_scaled') -> None:
    """
    Validate K-Means CSV has expected transformation suffixes.

    Args:
        csv_path: Path to K-Means transformed CSV (e.g., hook_kmeans_transformed.csv)
        expected_suffix: Primary suffix to check (default: '_scaled')

    Raises:
        ValidationError: If <80% of features have expected suffix

    Source: Stage5_MLModelTraining_HLD.md Section 6.1, Validation #5
    """
    # Read CSV header only
    df = pd.read_csv(csv_path, nrows=1)
    feature_names = [col for col in df.columns if col not in ['video_id', 'create_time', 'gender']]

    # Count features with transformation suffixes
    scaled_count = sum(1 for f in feature_names if '_scaled' in f)
    log_count = sum(1 for f in feature_names if '_log' in f)
    encoded_count = sum(1 for f in feature_names if '_encoded' in f)
    total_transformed = scaled_count + log_count + encoded_count

    # Expect at least 80% of features to have transformation suffixes
    expected_threshold = len(feature_names) * 0.80

    if total_transformed < expected_threshold:
        raise ValidationError(
            f"K-Means CSV feature naming validation failed: {csv_path}\n"
            f"  Total features: {len(feature_names)}\n"
            f"  Features with _scaled: {scaled_count}\n"
            f"  Features with _log: {log_count}\n"
            f"  Features with _encoded: {encoded_count}\n"
            f"  Total transformed: {total_transformed}/{len(feature_names)} ({total_transformed/len(feature_names)*100:.1f}%)\n"
            f"  Expected: >={expected_threshold:.0f} ({80}%)\n"
            f"\n"
            f"This indicates Stage 4 may not have applied transformations correctly.\n"
            f"Check FeatureTransformationCHILD.md Section 2.3.2 for transformation logic.\n"
            f"Expected suffixes: _scaled (StandardScaler), _log (log transform), _encoded (one-hot encoding)"
        )

    logger.info(
        f"✓ K-Means feature naming validated: {total_transformed}/{len(feature_names)} "
        f"({total_transformed/len(feature_names)*100:.1f}%) features have transformation suffixes"
    )
```

### 5.2 Business Logic Validation

```python
# Source: Stage5_MLModelTraining_HLD.md Section 2.3 Edge Cases tables

def validate_business_rules(bucket: str, windows: List[str], config: dict) -> None:
    """
    Validate business rules during processing.

    Source: Stage5_MLModelTraining_HLD.md Section 2.3.X Edge Cases
    """

    # Edge Case: Config file missing (Section 4.2 edge cases)
    # Handled by load_model_config() - uses defaults with warning

    # Edge Case: Hyperparameters validation
    if config['random_forest']['n_estimators'] <= 0:
        raise ValidationError("n_estimators must be > 0")
    if config['random_forest']['max_depth'] <= 0:
        raise ValidationError("max_depth must be > 0")
    if config['kmeans']['n_clusters'] != 3:
        raise ValidationError("n_clusters must be 3 (fixed for this pipeline)")

    # Edge Case: Window count validation
    if len(windows) < 2:
        raise ValidationError(
            f"Bucket must have at least 2 windows (hook + closing), got {len(windows)}"
        )
    if len(windows) > 7:
        raise ValidationError(
            f"Bucket cannot have more than 7 windows, got {len(windows)}"
        )

    logger.info(f"✓ Business rules validated for bucket {bucket}")
```

### 5.3 Output Validation

```python
# Source: Stage5_MLModelTraining_HLD.md Section 6.3 (not explicitly detailed in Child, inferred from Section 5.2)

def validate_stage_output(bucket: str, windows: List[str], bucket_base: str) -> None:
    """
    Validate output after processing.

    Source: Inferred from Stage5_MLModelTraining_HLD.md Section 5.2 (model_metrics.json schema)
    """

    # ===== VALIDATION 1: All model files exist =====
    required_models = [
        os.path.join(bucket_base, f'models/rf_video_{bucket}.pkl'),
    ]

    for window in windows:
        required_models.extend([
            os.path.join(bucket_base, f'models/rf_{window}_{bucket}.pkl'),
            os.path.join(bucket_base, f'models/{window}_kmeans_{bucket}.pkl'),
            os.path.join(bucket_base, f'models/{window}_X_data_{bucket}.pkl'),
            os.path.join(bucket_base, f'models/{window}_scalers_{bucket}.pkl'),
        ])

    required_models.append(os.path.join(bucket_base, 'models/model_metrics.json'))

    for model_path in required_models:
        if not os.path.exists(model_path):
            raise ValidationError(
                f"Expected output missing: {model_path}. Training incomplete."
            )

    # ===== VALIDATION 2: model_metrics.json schema =====
    metrics_path = os.path.join(bucket_base, 'models/model_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    if 'bucket' not in metrics:
        raise ValidationError("model_metrics.json missing 'bucket' field")
    if metrics['bucket'] != bucket:
        raise ValidationError(
            f"model_metrics.json bucket mismatch: expected {bucket}, got {metrics['bucket']}"
        )

    if 'video_level_rf' not in metrics:
        raise ValidationError("model_metrics.json missing 'video_level_rf'")
    if 'window_level_rf' not in metrics:
        raise ValidationError("model_metrics.json missing 'window_level_rf'")
    if 'window_level_kmeans' not in metrics:
        raise ValidationError("model_metrics.json missing 'window_level_kmeans'")

    # ===== VALIDATION 3: Model performance sanity check =====
    # Note: Only validate if RF was actually trained (C7 fix)
    if metrics['video_level_rf'].get('trained', True):  # Default True for backward compatibility
        rf_accuracy = metrics['video_level_rf']['accuracy']
        if not (0.0 <= rf_accuracy <= 1.0):
            raise ValidationError(
                f"Invalid accuracy: {rf_accuracy} (must be 0.0-1.0)"
            )

        # Warn if accuracy suspiciously low (< 0.60)
        # Source: Threshold from Stage5_MLModelTraining_HLD.md Section 6.2
        # Rationale: Random guessing baseline is 0.50, so 0.60 provides minimum margin for useful model
        if rf_accuracy < 0.60:
            logger.warning(
                f"Video-level RF accuracy is low: {rf_accuracy:.2f}. "
                f"This may indicate insufficient training data or poor feature quality."
            )

    # ===== VALIDATION 4: Cluster size balance check =====
    for window in windows:
        cluster_sizes = metrics['window_level_kmeans'][window]['cluster_sizes']
        total_videos = sum(cluster_sizes)

        # Check no cluster is too small (< 10% of total)
        min_size = min(cluster_sizes)
        if min_size < total_videos * 0.10:
            logger.warning(
                f"Window {window} has imbalanced clusters: {cluster_sizes}. "
                f"Smallest cluster ({min_size}) < 10% of total ({total_videos})."
            )

    logger.info(f"✓ Stage output validated: {len(required_models)} files created")
```

### 5.4 Critical Validation: Feature Name Normalization

```python
# Source: Stage5_MLModelTraining_HLD.md Section 3, Critical Warning #1

def validate_feature_overlap(kmeans_top5: List[str], rf_top5: List[str]) -> int:
    """
    Validate feature overlap between K-Means and RF top features.

    CRITICAL: Must normalize K-Means feature names before comparison.

    Source: Stage5_MLModelTraining_HLD.md Section 3, Critical Warning #1

    Returns:
        int: Overlap count (0-5)
    """
    # Normalize K-Means features (remove _scaled, _log, _encoded suffixes)
    kmeans_normalized = [normalize_feature_name(f) for f in kmeans_top5]

    # Calculate overlap
    overlap = set(kmeans_normalized) & set(rf_top5)
    overlap_count = len(overlap)

    # Validation: Expect at least 2/5 overlap (40%)
    # (Per Stage5_MLModelTraining_HLD.md Appendix A, Alternative 4)
    if overlap_count < 2:
        logger.warning(
            f"Low feature overlap: {overlap_count}/5. "
            f"K-Means top 5: {kmeans_normalized}, RF top 5: {rf_top5}"
        )

    return overlap_count
```

**Validation Summary**:
- **Pre-flight**: All Stage 4 files exist, non-empty, meet minimum video count
- **Feature naming**: K-Means features have transformation suffixes (>=80%)
- **Business rules**: Hyperparameters valid, window count reasonable
- **Output completeness**: All model files created, model_metrics.json valid
- **Model quality**: Accuracy in valid range, cluster sizes balanced
- **Feature overlap**: K-Means and RF top features overlap (normalized comparison)

## Section 6: Error Handling

**Source**: Stage5_MLModelTraining_HLD.md Section 6 (Error Handling & Validation)

### 6.1 Error Scenarios

```python
# Source: Stage5_MLModelTraining_HLD.md Section 6.2 (Error Cases)

class StageInputError(Exception):
    """Raised when Stage 4 inputs are missing or invalid."""
    pass

class InsufficientDataError(Exception):
    """Raised when video count below minimum threshold."""
    pass

class ConfigError(Exception):
    """Raised when config file is malformed."""
    pass

class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass

class ValidationError(Exception):
    """Raised when validation checks fail."""
    pass
```

### 6.2 Error Case Handling

```python
# Source: Stage5_MLModelTraining_HLD.md Section 6.2 (Error Cases)

def handle_error_scenarios():
    """
    Error handling patterns for all failure scenarios.

    Source: Stage5_MLModelTraining_HLD.md Section 6.2
    """

    # ===== SCENARIO 1: Stage 4 Files Missing =====
    # Source: Section 6.2, Scenario 1
    try:
        validate_stage4_outputs(bucket, windows, bucket_base)
    except StageInputError as e:
        logger.error(f"""
ERROR: Stage 4 incomplete
Details: {str(e)}
Action: Re-run Stage 4 or check if bucket was skipped in Stage 1

Failed file check: {e}
Expected location: {bucket_base}/ml_analysis/
Required files:
  - rf_transformed.csv
  - {window}_rf_transformed.csv (for each window)
  - {window}_km_transformed.csv (for each window)
""")
        sys.exit(1)  # exit_code_preflight_fail

    # ===== SCENARIO 2: Insufficient Videos =====
    # Source: Section 6.2, Scenario 2
    try:
        video_count = pd.read_csv(os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv')).shape[0]
        min_required = 50 if selection_strategy == "contrastive" else 30

        if video_count < min_required:
            raise InsufficientDataError(
                f"Bucket {bucket} has {video_count} videos (min {min_required} required)."
            )
    except InsufficientDataError as e:
        logger.error(f"""
ERROR: Insufficient videos for training
Details: {str(e)}
Action: Re-run Stage 1 with lower --video-count or skip this bucket

Current: {video_count} videos
Required: {min_required} videos (for {selection_strategy} mode)
Rationale: ML models need minimum sample size for reliable training
""")
        sys.exit(6)  # exit_code_data_integrity

    # ===== SCENARIO 3: Training Failure Mid-Bucket =====
    # Source: Section 6.2, Scenario 3
    try:
        train_bucket_models(bucket, windows, bucket_base, config, selection_strategy)
    except ModelTrainingError as e:
        logger.error(f"""
ERROR: Bucket {bucket} training failed
Details: {str(e)}
Action: All partial models deleted. Fix data issue and re-run Stage 5.

Common causes:
  - NaN values in feature data (check Stage 4 output quality)
  - Sklearn version mismatch (verify sklearn >= 0.24.0)
  - Disk full (check available space)
  - Memory error (reduce n_estimators or max_depth)

Atomic rollback: All models for this bucket have been deleted.
Bucket state: Clean (no partial models)
""")
        sys.exit(2)  # exit_code_training_fail

    # ===== SCENARIO 4: Config File Malformed =====
    # Source: Section 6.2, Scenario 4
    try:
        config = load_model_config()
    except ConfigError as e:
        logger.error(f"""
ERROR: Invalid model_hyperparameters.json
Details: {str(e)}
Action: Fix JSON syntax or delete file to use hardcoded defaults

Config file location: config/model_hyperparameters.json
Expected structure:
{{
  "random_forest": {{
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  }},
  "kmeans": {{
    "n_clusters": 3,
    "random_state": 42,
    "n_init": 10
  }}
}}

Fallback: Delete config file to use hardcoded defaults (will log warning)
""")
        sys.exit(1)  # exit_code_preflight_fail
```

### 6.3 Error Recovery Procedures

```python
# Source: Stage5_MLModelTraining_HLD.md Section 6.4 (Recovery Procedures)

def atomic_rollback(bucket: str, trained_models: List[str], bucket_base: str) -> None:
    """
    Atomic rollback: Delete ALL partial models for this bucket.

    Source: Stage5_MLModelTraining_HLD.md Section 6.4

    Q8 Decision: All models succeed OR all deleted on failure.
    Result: Either bucket has complete model set OR no models. Never partial.

    Args:
        bucket: str - Bucket name (e.g., "18-33s")
        trained_models: List[str] - Paths to all models created before failure
        bucket_base: str - Base path to bucket directory (for verification)
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

    # Verify bucket is clean (no partial models remain)
    models_dir = os.path.join(bucket_base, 'models')
    if os.path.exists(models_dir):
        remaining_files = [f for f in os.listdir(models_dir) if bucket in f]
        if remaining_files:
            logger.warning(
                f"Warning: {len(remaining_files)} files still exist in models/ after rollback: {remaining_files}"
            )
        else:
            logger.info("✓ Bucket clean: No partial models remain")
```

### 6.4 Error Logging Specification

```python
# Source: Stage5_MLModelTraining_HLD.md Section 6.3 (Error Logging)

def log_training_error(bucket: str, current_model: str, exception: Exception,
                      trained_models: List[str], start_time: float,
                      config: dict, bucket_base: str) -> None:
    """
    Comprehensive error logging for training failures.

    Source: Stage5_MLModelTraining_HLD.md Section 6.3 (Q10 Decision)

    Q10 Decision: Balanced Logging (Error + Context, No Data Dump)

    What is logged:
    - WHAT failed: Model name, file path, input shape
    - WHY it failed: Exception type and message, stack trace (first 10 lines)
    - CONTEXT: Hyperparameters, completed models, training duration, NaN count
    - NOT logged: Actual feature values, video IDs (privacy concerns)

    Args:
        bucket: str - Bucket name
        current_model: str - Model that failed during training
        exception: Exception - The exception that was raised
        trained_models: List[str] - Models successfully trained before failure
        start_time: float - Training start timestamp
        config: dict - Hyperparameters configuration
        bucket_base: str - Base path to bucket directory
    """
    import traceback

    elapsed = time.time() - start_time

    # Get input file path (for error context)
    # String parsing examples:
    #   "models/rf_video_18-33s.pkl" → contains 'rf_video' → rf_transformed.csv
    #   "models/rf_hook_18-33s.pkl" → split by '_rf_' → "models/rf" + "hook_18-33s.pkl" → "hook"
    #   "models/hook_kmeans_18-33s.pkl" → split by '_kmeans_' → "models/hook" + "18-33s.pkl" → "hook"
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

    # Get input shape and NaN count (if file exists)
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

### 6.5 Exit Codes

```python
# Source: FoundationCHILD.md Section 7 (Standardized Exit Codes)

EXIT_CODES = {
    0: "SUCCESS - All models trained successfully",
    1: "PREFLIGHT_FAIL - Stage 4 outputs missing or config malformed",
    2: "TRAINING_FAIL - Model training failed (NaN values, sklearn error)",
    3: "VALIDATION_FAIL - Model metrics below threshold (not used in MVP)",
    4: "IO_FAIL - Disk full, permission denied",
    5: "PARTIAL - Partial completion (NOT USED - atomic training guarantees all or nothing)",
    6: "DATA_INTEGRITY - Insufficient videos for training",
}

def exit_with_code(code: int, message: str = "") -> None:
    """
    Exit with standardized exit code and optional message.

    Source: FoundationCHILD.md Section 7
    """
    logger.info(f"Exiting with code {code}: {EXIT_CODES[code]}")
    if message:
        logger.info(f"Additional context: {message}")
    sys.exit(code)
```

**Error Handling Summary**:
- **Fail-fast**: Stage 4 missing → exit 1
- **Atomic rollback**: Training failure → delete ALL models → exit 2
- **Graceful degradation**: Config missing → use defaults + warning → continue
- **Comprehensive logging**: WHAT/WHY/CONTEXT (no sensitive data)
- **Recovery**: Clean bucket state (no partial models)

## Section 7: Complete Example Traces

**Source**: Stage5_MLModelTraining_HLD.md Section 2.3 (Detailed Process)

### 7.1 Success Case: Bucket 18-33s (Contrastive Mode, 100 Videos)

```
===============================================================================
STAGE 5: ML MODEL TRAINING - BUCKET 18-33s
===============================================================================

INPUT PARAMETERS:
  client_id: acme_corp
  analysis_type: hashtag
  target: #nutrition
  mode: top
  strategy: contrastive
  video_count: 100
  bucket: 18-33s
  windows: ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"]

PATHS:
  client_base: /data/clients/acme_corp/
  analysis_base: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/
  bucket_base: /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/

===============================================================================
STEP 1: PRE-FLIGHT VALIDATION
===============================================================================

[2025-01-20 10:00:00] INFO: Validating Stage 4 outputs...

Checking file existence (13 files):
  ✓ ml_analysis/rf_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/hook_rf_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/middle_1_rf_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/middle_2_rf_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/middle_3_rf_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/middle_4_rf_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/closing_rf_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/hook_km_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/middle_1_km_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/middle_2_km_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/middle_3_km_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/middle_4_km_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/closing_km_transformed.csv (exists, 100 rows)

Checking video count threshold:
  Current: 100 videos
  Required: 50 videos (contrastive mode)
  ✓ Sufficient videos for training

Validating K-Means feature naming:
  ✓ hook_km_transformed.csv: 35/39 (89.7%) features have transformation suffixes
  ✓ middle_1_km_transformed.csv: 35/39 (89.7%) features have transformation suffixes
  ✓ middle_2_km_transformed.csv: 35/39 (89.7%) features have transformation suffixes
  ✓ middle_3_km_transformed.csv: 35/39 (89.7%) features have transformation suffixes
  ✓ middle_4_km_transformed.csv: 35/39 (89.7%) features have transformation suffixes
  ✓ closing_km_transformed.csv: 35/39 (89.7%) features have transformation suffixes

[2025-01-20 10:00:02] INFO: ✓ Stage 4 validation passed: 13 files exist, 100 videos

===============================================================================
STEP 2: LOAD HYPERPARAMETERS
===============================================================================

[2025-01-20 10:00:02] INFO: Loading hyperparameters...
[2025-01-20 10:00:02] INFO: ✓ Loaded hyperparameters from config/model_hyperparameters.json

Hyperparameters loaded:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  kmeans:
    n_clusters: 3
    random_state: 42
    n_init: 10

===============================================================================
STEP 3: TRAIN VIDEO-LEVEL RF
===============================================================================

[2025-01-20 10:00:02] INFO: Training video-level RF for 18-33s...

Loading data:
  Input: ml_analysis/rf_transformed.csv
  Shape: (100, 191) # 190 features + 1 target
  Target distribution: 80 top (1), 20 bottom (0)

Dropping columns: ['is_top_performer', 'video_id']
Final training shape: (100, 189)

Training RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)...
Training complete: 0.8s

Saving model:
  Output: models/rf_video_18-33s.pkl
  Size: 2.3 MB

[2025-01-20 10:00:03] INFO: ✓ Video-level RF trained: models/rf_video_18-33s.pkl

===============================================================================
STEP 4: TRAIN WINDOW-LEVEL RF (6 models)
===============================================================================

[2025-01-20 10:00:03] INFO: Training window-level RF for hook...
  Input: ml_analysis/hook_rf_transformed.csv (100, 22)
  Training: RandomForestClassifier (0.3s)
  Output: models/rf_hook_18-33s.pkl (450 KB)
  ✓ Window-level RF trained: hook

[2025-01-20 10:00:04] INFO: Training window-level RF for middle_1...
  Input: ml_analysis/middle_1_rf_transformed.csv (100, 22)
  Training: RandomForestClassifier (0.3s)
  Output: models/rf_middle_1_18-33s.pkl (450 KB)
  ✓ Window-level RF trained: middle_1

[2025-01-20 10:00:04] INFO: Training window-level RF for middle_2...
  ✓ Window-level RF trained: middle_2 (0.3s)

[2025-01-20 10:00:05] INFO: Training window-level RF for middle_3...
  ✓ Window-level RF trained: middle_3 (0.3s)

[2025-01-20 10:00:05] INFO: Training window-level RF for middle_4...
  ✓ Window-level RF trained: middle_4 (0.3s)

[2025-01-20 10:00:06] INFO: Training window-level RF for closing...
  ✓ Window-level RF trained: closing (0.3s)

===============================================================================
STEP 5: TRAIN K-MEANS (6 models)
===============================================================================

[2025-01-20 10:00:06] INFO: Training K-Means for hook...
  Input: ml_analysis/hook_km_transformed.csv (100, 39)
  Training: KMeans(n_clusters=3, random_state=42, n_init=10) (0.5s)
  Output: models/hook_kmeans_18-33s.pkl (180 KB)
  Saving X matrix: models/hook_X_data_18-33s.pkl (45 KB)
  Copying scalers: models/hook_scalers_18-33s.pkl (8 KB)
  ✓ K-Means trained: hook

[2025-01-20 10:00:07] INFO: Training K-Means for middle_1...
  ✓ K-Means trained: middle_1 (0.5s)

[2025-01-20 10:00:07] INFO: Training K-Means for middle_2...
  ✓ K-Means trained: middle_2 (0.5s)

[2025-01-20 10:00:08] INFO: Training K-Means for middle_3...
  ✓ K-Means trained: middle_3 (0.5s)

[2025-01-20 10:00:08] INFO: Training K-Means for middle_4...
  ✓ K-Means trained: middle_4 (0.5s)

[2025-01-20 10:00:09] INFO: Training K-Means for closing...
  ✓ K-Means trained: closing (0.5s)

===============================================================================
STEP 6: GENERATE MODEL METRICS
===============================================================================

[2025-01-20 10:00:09] INFO: Generating model_metrics.json...

Computing metrics:
  Video-level RF: accuracy=0.87, precision=0.89, recall=0.84, f1=0.86
  Top feature: hook_eye_contact_rate (importance=0.22)

  Window-level RF:
    hook: accuracy=0.82, top_feature=eye_contact_rate (0.35)
    middle_1: accuracy=0.79, top_feature=scene_count (0.28)
    middle_2: accuracy=0.81, top_feature=energy_level (0.31)
    middle_3: accuracy=0.78, top_feature=speech_coverage (0.29)
    middle_4: accuracy=0.80, top_feature=word_count (0.27)
    closing: accuracy=0.83, top_feature=has_cta (0.38)

  Window-level K-Means:
    hook: inertia=12.5, silhouette=0.68, cluster_sizes=[35, 42, 23]
    middle_1: inertia=11.8, silhouette=0.65, cluster_sizes=[33, 38, 29]
    middle_2: inertia=13.2, silhouette=0.62, cluster_sizes=[31, 41, 28]
    middle_3: inertia=12.9, silhouette=0.64, cluster_sizes=[34, 37, 29]
    middle_4: inertia=13.5, silhouette=0.61, cluster_sizes=[32, 39, 29]
    closing: inertia=11.2, silhouette=0.71, cluster_sizes=[36, 40, 24]

Saving metrics:
  Output: models/model_metrics.json
  Size: 8.2 KB

[2025-01-20 10:00:09] INFO: ✓ Model metrics generated

===============================================================================
TRAINING COMPLETE
===============================================================================

[2025-01-20 10:00:09] INFO: ✓ Bucket 18-33s training complete: 7.5s (26 files)

Files created:
  - 1 video-level RF model
  - 6 window-level RF models
  - 6 K-Means models
  - 6 X data matrices
  - 6 scalers
  - 1 model_metrics.json
  Total: 26 files

Performance:
  Total duration: 7.5s
  Expected: <120s ✓

Output directory:
  /data/clients/acme_corp/hashtags/nutrition/top_contrastive/bucket_18-33s/models/

[2025-01-20 10:00:09] INFO: Exiting with code 0: SUCCESS - All models trained successfully
```

### 7.2 Failure Case: Stage 4 Files Missing

```
===============================================================================
STAGE 5: ML MODEL TRAINING - BUCKET 18-33s
===============================================================================

[2025-01-20 10:00:00] INFO: Validating Stage 4 outputs...

Checking file existence (13 files):
  ✓ ml_analysis/rf_transformed.csv (exists, 100 rows)
  ✓ ml_analysis/hook_rf_transformed.csv (exists, 100 rows)
  ✗ ml_analysis/middle_1_rf_transformed.csv (MISSING)

===============================================================================
VALIDATION FAILED
===============================================================================

ERROR: Stage 4 incomplete
Details: Stage 4 incomplete: Missing /data/.../ml_analysis/middle_1_rf_transformed.csv.
Run Stage 4 first or check if bucket 18-33s was skipped in Stage 1.

Failed file check: middle_1_rf_transformed.csv
Expected location: /data/clients/acme_corp/.../bucket_18-33s/ml_analysis/
Required files:
  - rf_transformed.csv
  - hook_rf_transformed.csv (for each window)
  - hook_km_transformed.csv (for each window)

Action: Re-run Stage 4 or check if bucket was skipped in Stage 1

[2025-01-20 10:00:01] ERROR: Exiting with code 1: PREFLIGHT_FAIL - Stage 4 outputs missing
```

### 7.3 Failure Case: Training Failure Mid-Bucket (NaN Values)

```
===============================================================================
STAGE 5: ML MODEL TRAINING - BUCKET 18-33s
===============================================================================

[2025-01-20 10:00:00] INFO: ✓ Stage 4 validation passed: 13 files exist, 100 videos
[2025-01-20 10:00:02] INFO: ✓ Loaded hyperparameters from config/model_hyperparameters.json

===============================================================================
STEP 3: TRAIN VIDEO-LEVEL RF
===============================================================================

[2025-01-20 10:00:02] INFO: Training video-level RF for 18-33s...
[2025-01-20 10:00:03] INFO: ✓ Video-level RF trained: models/rf_video_18-33s.pkl

===============================================================================
STEP 4: TRAIN WINDOW-LEVEL RF (6 models)
===============================================================================

[2025-01-20 10:00:03] INFO: Training window-level RF for hook...
[2025-01-20 10:00:04] INFO: ✓ Window-level RF trained: hook

[2025-01-20 10:00:04] INFO: Training window-level RF for middle_1...
[2025-01-20 10:00:04] ERROR: Training failed for middle_1

===============================================================================
BUCKET 18-33s TRAINING FAILED
===============================================================================

WHAT FAILED:
  Model name: models/rf_middle_1_18-33s.pkl
  Input file: ml_analysis/middle_1_rf_transformed.csv
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
  Completed models before failure: 2 files
  Training duration before failure: 2.1s
  NaN count in input: 5 values

RECOVERY ACTION:
  Atomic rollback: Deleting all 2 partial models

[2025-01-20 10:00:05] INFO: Performing atomic rollback for bucket 18-33s...
[2025-01-20 10:00:05] INFO: Models to delete: 2 files
[2025-01-20 10:00:05] INFO:   ✓ Deleted: models/rf_video_18-33s.pkl
[2025-01-20 10:00:05] INFO:   ✓ Deleted: models/rf_hook_18-33s.pkl
[2025-01-20 10:00:05] INFO: Rollback complete: 2/2 files deleted
[2025-01-20 10:00:05] INFO: ✓ Bucket clean: No partial models remain

===============================================================================

ERROR: Bucket 18-33s training failed
Details: NaN values in feature data
Action: All partial models deleted. Fix data issue and re-run Stage 5.

Common causes:
  - NaN values in feature data (check Stage 4 output quality)
  - Sklearn version mismatch (verify sklearn >= 0.24.0)
  - Disk full (check available space)
  - Memory error (reduce n_estimators or max_depth)

Atomic rollback: All models for this bucket have been deleted.
Bucket state: Clean (no partial models)

NEXT STEPS:
  1. Check input data quality (NaN values, feature ranges)
  2. Verify sklearn version >= 0.24.0
  3. Check disk space and memory availability
  4. Re-run Stage 5 after fixing issue

[2025-01-20 10:00:05] ERROR: Exiting with code 2: TRAINING_FAIL - Model training failed
```

### 7.4 Failure Case: Insufficient Videos

```
===============================================================================
STAGE 5: ML MODEL TRAINING - BUCKET 18-33s
===============================================================================

[2025-01-20 10:00:00] INFO: Validating Stage 4 outputs...

Checking file existence (13 files):
  ✓ All 13 files exist

Checking video count threshold:
  Current: 45 videos
  Required: 50 videos (contrastive mode)
  ✗ Insufficient videos for training

===============================================================================
VALIDATION FAILED
===============================================================================

ERROR: Insufficient videos for training
Details: Bucket 18-33s has 45 videos (min 50 required).
Action: Re-run Stage 1 with lower --video-count or skip this bucket

Current: 45 videos
Required: 50 videos (for contrastive mode)
Rationale: ML models need minimum sample size for reliable training

[2025-01-20 10:00:01] ERROR: Exiting with code 6: DATA_INTEGRITY - Insufficient videos for training
```

**Trace Summary**:
- **Success case**: 7.5s total, 26 files created, all validations passed
- **Stage 4 missing**: Fail-fast at pre-flight (exit 1)
- **Training failure**: Atomic rollback, 2 models deleted (exit 2)
- **Insufficient videos**: Fail-fast at validation (exit 6)

## Section 8: File Structure & Integration

**Source**: FoundationCHILD.md Section 2 (Client Architecture & Storage) + Stage5_MLModelTraining_HLD.md Section 4.2 (Output Contracts)

### 8.1 Directory Structure

```
# Source: FoundationCHILD.md Section 2.1 + Stage5_MLModelTraining_HLD.md Section 4.2

/data/clients/{client_id}/
└── {analysis_type}s/                          # "hashtags", "competitors", or "creators"
    └── {target}/                              # e.g., "nutrition" (sanitized, no # or @)
        └── {mode}_{strategy}/                 # e.g., "top_contrastive"
            ├── config.json                    # Analysis configuration (Foundation)
            └── bucket_{bucket}/               # e.g., "bucket_18-33s"
                ├── ml_analysis/               # Stage 4 outputs (inputs to Stage 5)
                │   ├── rf_transformed.csv                    # Video-level RF input (100, ~190)
                │   ├── hook_rf_transformed.csv               # Window-level RF input (100, 22)
                │   ├── middle_1_rf_transformed.csv           # (100, 22)
                │   ├── middle_2_rf_transformed.csv           # (100, 22)
                │   ├── middle_3_rf_transformed.csv           # (100, 22)
                │   ├── middle_4_rf_transformed.csv           # (100, 22)
                │   ├── closing_rf_transformed.csv            # (100, 22)
                │   ├── hook_km_transformed.csv               # K-Means input (100, ~39)
                │   ├── middle_1_km_transformed.csv           # (100, ~39)
                │   ├── middle_2_km_transformed.csv           # (100, ~39)
                │   ├── middle_3_km_transformed.csv           # (100, ~39)
                │   ├── middle_4_km_transformed.csv           # (100, ~39)
                │   ├── closing_km_transformed.csv            # (100, ~39)
                │   ├── hook_scalers.pkl                      # Scalers from Stage 4 (optional, 8 KB)
                │   ├── middle_1_scalers.pkl                  # (8 KB)
                │   ├── middle_2_scalers.pkl                  # (8 KB)
                │   ├── middle_3_scalers.pkl                  # (8 KB)
                │   ├── middle_4_scalers.pkl                  # (8 KB)
                │   └── closing_scalers.pkl                   # (8 KB)
                │
                └── models/                    # Stage 5 outputs (created by this stage)
                    ├── rf_video_18-33s.pkl                   # Video-level RF (2.3 MB)
                    ├── rf_hook_18-33s.pkl                    # Window-level RF (450 KB)
                    ├── rf_middle_1_18-33s.pkl                # (450 KB)
                    ├── rf_middle_2_18-33s.pkl                # (450 KB)
                    ├── rf_middle_3_18-33s.pkl                # (450 KB)
                    ├── rf_middle_4_18-33s.pkl                # (450 KB)
                    ├── rf_closing_18-33s.pkl                 # (450 KB)
                    ├── hook_kmeans_18-33s.pkl                # K-Means model (180 KB)
                    ├── middle_1_kmeans_18-33s.pkl            # (180 KB)
                    ├── middle_2_kmeans_18-33s.pkl            # (180 KB)
                    ├── middle_3_kmeans_18-33s.pkl            # (180 KB)
                    ├── middle_4_kmeans_18-33s.pkl            # (180 KB)
                    ├── closing_kmeans_18-33s.pkl             # (180 KB)
                    ├── hook_X_data_18-33s.pkl                # X matrix for silhouette (45 KB)
                    ├── middle_1_X_data_18-33s.pkl            # (45 KB)
                    ├── middle_2_X_data_18-33s.pkl            # (45 KB)
                    ├── middle_3_X_data_18-33s.pkl            # (45 KB)
                    ├── middle_4_X_data_18-33s.pkl            # (45 KB)
                    ├── closing_X_data_18-33s.pkl             # (45 KB)
                    ├── hook_scalers_18-33s.pkl               # Scalers for inference (8 KB)
                    ├── middle_1_scalers_18-33s.pkl           # (8 KB)
                    ├── middle_2_scalers_18-33s.pkl           # (8 KB)
                    ├── middle_3_scalers_18-33s.pkl           # (8 KB)
                    ├── middle_4_scalers_18-33s.pkl           # (8 KB)
                    ├── closing_scalers_18-33s.pkl            # (8 KB)
                    └── model_metrics.json                    # Performance summary (8 KB)
```

**Total Files Created** (for bucket 18-33s with 6 windows):
- 1 video-level RF model
- 6 window-level RF models
- 6 K-Means models
- 6 X data matrices
- 6 scalers
- 1 model_metrics.json
- **Total: 26 files (~6.5 MB)**

### 8.2 File Naming Conventions

```python
# Source: Stage5_MLModelTraining_HLD.md Section 4.2 (Q2 decision)

# Video-Level RF Model
naming_convention = "rf_video_{bucket}.pkl"
example = "rf_video_18-33s.pkl"

# Window-Level RF Models
naming_convention = "rf_{window}_{bucket}.pkl"
examples = [
    "rf_hook_18-33s.pkl",
    "rf_middle_1_18-33s.pkl",
    "rf_closing_18-33s.pkl"
]

# K-Means Models
naming_convention = "{window}_kmeans_{bucket}.pkl"
examples = [
    "hook_kmeans_18-33s.pkl",
    "middle_1_kmeans_18-33s.pkl",
    "closing_kmeans_18-33s.pkl"
]

# X Data Matrices (for silhouette calculation)
naming_convention = "{window}_X_data_{bucket}.pkl"
examples = [
    "hook_X_data_18-33s.pkl",
    "middle_1_X_data_18-33s.pkl",
    "closing_X_data_18-33s.pkl"
]

# Scalers (for inference)
naming_convention = "{window}_scalers_{bucket}.pkl"
examples = [
    "hook_scalers_18-33s.pkl",
    "middle_1_scalers_18-33s.pkl",
    "closing_scalers_18-33s.pkl"
]

# Model Metrics Summary
naming_convention = "model_metrics.json"
# Always named this, no bucket suffix
```

### 8.3 Integration Points

```python
# Source: Stage5_MLModelTraining_HLD.md Section 4.3 (Cross-Stage Dependencies)

# ===== UPSTREAM DEPENDENCY: Stage 4 (Feature Transformation) =====
upstream_stage = "Stage 4: Feature Transformation"
upstream_outputs = [
    "ml_analysis/rf_transformed.csv",           # Video-level RF input
    "ml_analysis/{window}_rf_transformed.csv",  # Window-level RF input (per window)
    "ml_analysis/{window}_km_transformed.csv",  # K-Means input (per window)
]
integration_protocol = "File-based (CSV files)"
validation = "Pre-flight validation checks all files exist and non-empty"

# ===== DOWNSTREAM DEPENDENCY: Stage 6 (ML Analysis Generation) =====
downstream_stage = "Stage 6: ML Analysis Generation"
downstream_inputs = [
    "models/rf_video_{bucket}.pkl",            # Load for feature importance extraction
    "models/rf_{window}_{bucket}.pkl",         # Load for window-level feature ranking
    "models/{window}_kmeans_{bucket}.pkl",     # Load for cluster centroid analysis
    "models/{window}_X_data_{bucket}.pkl",     # Load for silhouette score calculation
    "models/{window}_scalers_{bucket}.pkl",    # Load for inverse_transform (optional)
    "models/model_metrics.json",               # Load for performance sanity check
]
integration_protocol = "File-based (joblib pickle + JSON)"
handoff_guarantee = "Atomic: All models exist OR no models (never partial)"

# ===== SHARED DEPENDENCY: FoundationTI =====
foundation_provides = [
    "CLI parameter parsing (client_id, analysis_type, target, mode, strategy, video_count, bucket)",
    "Directory path construction (client_base, analysis_base, bucket_base)",
    "config.json schema and loading",
    "Bucket window definitions (BUCKET_WINDOWS[bucket])",
    "Standardized exit codes",
    "Logging configuration"
]

# ===== ORCHESTRATION PATTERN (for rumiai_ml_batch.py implementation) =====
# Source: rumiai_ml_batch.py Stage 4 pattern (lines 662-796)
# Note: Stage 5 not yet implemented - this shows expected calling pattern

"""
Expected orchestrator implementation in rumiai_ml_batch.py (lines ~817-920):

# ===== STAGE 5: ML MODEL TRAINING =====
logger.info("Starting Stage 5: ML Model Training")
print("\n" + "="*80)
print("STAGE 5: ML MODEL TRAINING")
print("="*80)

# Load hyperparameters
hyperparameters = load_model_config()  # From Section 4.2

stage5_summaries = {}
for bucket_name in winning_buckets:
    logger.info(f"Starting Stage 5 for bucket: {bucket_name}")
    print(f"\n--- Training models for bucket: {bucket_name} ---")

    bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

    try:
        # Validate Stage 4 checkpoint exists
        stage4_checkpoint = bucket_path / "checkpoints" / "stage_4_checkpoint.json"
        if not stage4_checkpoint.exists():
            logger.error(f"Bucket {bucket_name}: Stage 4 checkpoint missing")
            print(f"✗ Bucket {bucket_name}: Stage 4 not complete (skipping)")
            continue

        # Get bucket windows
        from config.bucket_definitions import BUCKET_WINDOWS
        windows = BUCKET_WINDOWS[bucket_name]

        # PRE-FLIGHT VALIDATION: Check all Stage 4 files exist (Section 4.1)
        validate_stage4_outputs(
            bucket=bucket_name,
            windows=windows,
            bucket_base=str(bucket_path),
            selection_strategy=config.selection_strategy  # ← From CLI args (FoundationTI)
        )

        # TRAIN MODELS: Sequential training with atomic rollback (Section 4.3)
        train_bucket_models(
            bucket=bucket_name,
            windows=windows,
            bucket_base=str(bucket_path),
            config=hyperparameters,
            selection_strategy=config.selection_strategy  # ← From CLI args (FoundationTI)
        )

        # Success logging
        stage5_summaries[bucket_name] = {"status": "completed"}
        logger.info(f"Bucket {bucket_name} complete: 26 models trained")
        print(f"✓ Bucket {bucket_name}: 26 models trained")

    except StageInputError as e:
        # Stage 4 files missing or invalid (pre-flight failed)
        logger.error(f"Stage 5 validation failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name}: Stage 4 incomplete (skipping)")
        continue  # Skip this bucket, process remaining buckets

    except InsufficientDataError as e:
        # Video count below threshold
        logger.error(f"Stage 5 data validation failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name}: Insufficient videos (skipping)")
        continue  # Skip this bucket

    except ModelTrainingError as e:
        # Training failed mid-bucket (atomic rollback already performed)
        logger.error(f"Stage 5 training failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name}: Training failed (skipping)")
        continue  # Skip this bucket

    except (IOError, OSError) as e:
        # I/O failure (disk full, permission denied) - exit pipeline
        logger.error(f"Stage 5 I/O error for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name}: I/O error (exiting pipeline)")
        return 4  # Exit code 4 = I/O failure

logger.info("Stage 5 completed for all buckets")
print("\n✓ Stage 5: ML Model Training - COMPLETE")
"""

# Key points for implementer:
# 1. selection_strategy comes from config.selection_strategy (CLI args parsed by FoundationTI)
# 2. validate_stage4_outputs() is called BEFORE train_bucket_models() (pre-flight check)
# 3. Skip-on-fail policy for bucket-specific errors (StageInputError, InsufficientDataError)
# 4. Exit-on-fail policy for system-wide errors (IOError, OSError)
# 5. Atomic rollback happens inside train_bucket_models() on ModelTrainingError
```

### 8.4 Path Construction

```python
# Source: FoundationCHILD.md Section 2.2 (Path Templates)

def construct_stage5_paths(client_id: str, analysis_type: str, target: str,
                          mode: str, strategy: str, bucket: str) -> dict:
    """
    Construct all paths needed for Stage 5.

    Source: FoundationCHILD.md Section 2.2
    """
    # Base paths
    client_base = f"/data/clients/{client_id}/"
    analysis_base = f"{client_base}/{analysis_type}s/{target}/{mode}_{strategy}/"
    bucket_base = f"{analysis_base}/bucket_{bucket}/"

    # Input paths (Stage 4 outputs)
    ml_analysis_dir = f"{bucket_base}/ml_analysis/"
    input_paths = {
        "rf_transformed_csv": f"{ml_analysis_dir}/rf_transformed.csv",
    }

    # Add window-level inputs (dynamic based on BUCKET_WINDOWS)
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket]

    for window in windows:
        input_paths[f"{window}_rf_csv"] = f"{ml_analysis_dir}/{window}_rf_transformed.csv"
        input_paths[f"{window}_km_csv"] = f"{ml_analysis_dir}/{window}_km_transformed.csv"

    # Output paths (Stage 5 outputs)
    models_dir = f"{bucket_base}/models/"
    output_paths = {
        "rf_video_model": f"{models_dir}/rf_video_{bucket}.pkl",
        "model_metrics_json": f"{models_dir}/model_metrics.json",
    }

    # Add window-level outputs
    for window in windows:
        output_paths[f"rf_{window}_model"] = f"{models_dir}/rf_{window}_{bucket}.pkl"
        output_paths[f"{window}_kmeans_model"] = f"{models_dir}/{window}_kmeans_{bucket}.pkl"
        output_paths[f"{window}_X_data"] = f"{models_dir}/{window}_X_data_{bucket}.pkl"
        output_paths[f"{window}_scalers"] = f"{models_dir}/{window}_scalers_{bucket}.pkl"

    return {
        "client_base": client_base,
        "analysis_base": analysis_base,
        "bucket_base": bucket_base,
        "ml_analysis_dir": ml_analysis_dir,
        "models_dir": models_dir,
        "input_paths": input_paths,
        "output_paths": output_paths,
    }
```

### 8.5 Storage Requirements

```python
# Source: Stage5_MLModelTraining_HLD.md Section 4.2 + Section 7.1 example

# Per-bucket storage (example: bucket 18-33s with 6 windows)
storage_breakdown = {
    "video_level_rf": "2.3 MB",              # 1 file
    "window_level_rf": "2.7 MB",             # 6 files × 450 KB
    "kmeans_models": "1.08 MB",              # 6 files × 180 KB
    "X_data_matrices": "270 KB",             # 6 files × 45 KB
    "scalers": "48 KB",                      # 6 files × 8 KB
    "model_metrics_json": "8 KB",            # 1 file
}

total_per_bucket = "6.5 MB"                  # For 6-window bucket (18-33s)

# Full analysis storage (3 active buckets, typical case)
typical_analysis_storage = "19.5 MB"         # 3 buckets × 6.5 MB

# Maximum theoretical storage (all 8 buckets, not used in production)
max_theoretical_storage = "52 MB"            # 8 buckets × 6.5 MB (average)
```

**Storage Summary**:
- **Per bucket**: ~6.5 MB (26 files for 6-window bucket)
- **Typical analysis**: ~19.5 MB (3 active buckets selected by Stage 1)
- **Maximum theoretical**: ~52 MB (all 8 buckets, not used in production)

## Section 9: Configuration & Environment

**Source**: Stage5_MLModelTraining_HLD.md Section 9 (Configuration) + Stage5_MLModelTraining_HLD.md Section 4.4 (External Dependencies)

### 9.1 Configuration Files

```json
// Source: Stage5_MLModelTraining_HLD.md Section 9.1

// File: config/model_hyperparameters.json (OPTIONAL - graceful fallback to defaults)
{
  "random_forest": {
    "n_estimators": 100,           // Number of trees in forest (default: 100)
    "max_depth": 10,                // Maximum depth of each tree (default: 10)
    "random_state": 42              // Random seed for reproducibility (default: 42)
  },
  "kmeans": {
    "n_clusters": 3,                // Number of clusters (FIXED: must be 3)
    "random_state": 42,             // Random seed for reproducibility (default: 42)
    "n_init": 10                    // Number of random initializations (default: 10)
  }
}
```

**Configuration Behavior**:
- **File exists**: Load hyperparameters from file
- **File missing**: Use hardcoded defaults (log warning, continue)
- **File malformed**: Fail with ConfigError (exit 1)

### 9.2 Environment Requirements

```python
# Source: Stage5_MLModelTraining_HLD.md Section 4.4 (External Dependencies)

# Python version
python_version = "3.8+"

# Required packages
dependencies = {
    "scikit-learn": ">=0.24.0",    # RandomForestClassifier, KMeans, silhouette_samples
    "scipy": ">=1.7.0",             # binomtest (for statistical validation)
    "joblib": ">=1.0.0",            # Model serialization (pickle)
    "pandas": ">=1.3.0",            # CSV loading and DataFrame operations
    "numpy": ">=1.21.0",            # Variance calculations, array operations
}

# System resources (typical bucket with 6 windows)
resources = {
    "memory_peak": "~500 MB",       # RandomForest training (100 estimators, 190 features)
    "disk_space": "~6.5 MB",        # Model files per bucket
    "cpu_cores": "1+",              # Sklearn uses internal parallelization
}
```

### 9.3 Environment Variables

```bash
# Source: Inferred from FoundationCHILD.md (not explicitly documented)

# Optional: Override data directory
export RUMIAI_DATA_DIR="/data/clients"

# Optional: Override config directory
export RUMIAI_CONFIG_DIR="./config"

# Optional: Enable debug logging
export RUMIAI_LOG_LEVEL="DEBUG"
```

### 9.4 Bucket Window Definitions

```python
# Source: FoundationCHILD.md Section 6 (Bucket Definitions)

# File: config/bucket_definitions.py
BUCKET_WINDOWS = {
    "0-3s": ["hook"],
    "3-9s": ["hook", "closing"],
    "9-13s": ["hook", "middle_1", "closing"],
    "13-18s": ["hook", "middle_1", "middle_2", "closing"],
    "18-33s": ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"],
    "33-60s": ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"],
    "60-90s": ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "middle_5", "closing"],
    "90-120s": ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "middle_5", "closing"],
}
```

**Usage in Stage 5**:
```python
from config.bucket_definitions import BUCKET_WINDOWS

bucket = "18-33s"
windows = BUCKET_WINDOWS[bucket]  # ["hook", "middle_1", ..., "closing"]

# Determines:
# - Number of window-level RF models to train (6 for 18-33s)
# - Number of K-Means models to train (6 for 18-33s)
# - Total model files created (26 for 18-33s)
```

## Section 10: Logging Specifications

**Source**: Stage5_MLModelTraining_HLD.md Section 6.3 (Error Logging) + Section 7.1 (Success trace)

### 10.1 Log Levels

```python
# Source: Inferred from Section 7 example traces

import logging

# Log levels used in Stage 5
LOG_LEVELS = {
    "DEBUG": "Detailed execution flow (hyperparameters, input shapes, feature names)",
    "INFO": "Progress updates (model training start/complete, validation passed)",
    "WARNING": "Non-fatal issues (config missing, slow training, low accuracy)",
    "ERROR": "Fatal errors (Stage 4 missing, training failure, validation failure)",
}

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO
```

### 10.2 Log Format

```python
# Source: Inferred from Section 7 example traces

# Standard log format (from Foundation)
log_format = "[%(asctime)s] %(levelname)s: %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# Example output:
# [2025-01-20 10:00:00] INFO: Validating Stage 4 outputs...
# [2025-01-20 10:00:02] INFO: ✓ Stage 4 validation passed: 13 files exist, 100 videos
# [2025-01-20 10:00:09] INFO: ✓ Bucket 18-33s training complete: 7.5s (26 files)
```

### 10.3 Key Log Messages

```python
# Source: Stage5_MLModelTraining_HLD.md Section 7 (Complete Example Traces)

# ===== PRE-FLIGHT VALIDATION =====
logger.info("Validating Stage 4 outputs...")
logger.info(f"✓ Stage 4 validation passed: {len(required_files)} files exist, {video_count} videos")

# ===== CONFIGURATION LOADING =====
logger.info("Loading hyperparameters...")
logger.info("✓ Loaded hyperparameters from config/model_hyperparameters.json")
logger.warning("Config file not found: config/model_hyperparameters.json. Using hardcoded defaults.")

# ===== MODEL TRAINING =====
logger.info(f"Training video-level RF for {bucket}...")
logger.info(f"✓ Video-level RF trained: {model_path}")

logger.info(f"Training window-level RF for {window}...")
logger.info(f"✓ Window-level RF trained: {window}")

logger.info(f"Training K-Means for {window}...")
logger.info(f"✓ K-Means trained: {window}")

# ===== METRICS GENERATION =====
logger.info("Generating model_metrics.json...")
logger.info("✓ Model metrics generated")

# ===== COMPLETION =====
logger.info(f"✓ Bucket {bucket} training complete: {elapsed:.1f}s ({len(trained_models)} files)")

# ===== PERFORMANCE WARNINGS =====
logger.warning(
    f"Bucket {bucket} training took {elapsed:.1f}s (expected <120s). "
    f"Check for performance issues."
)

logger.warning(
    f"Video-level RF accuracy is low: {rf_accuracy:.2f}. "
    f"This may indicate insufficient training data or poor feature quality."
)

logger.warning(
    f"Window {window} has imbalanced clusters: {cluster_sizes}. "
    f"Smallest cluster ({min_size}) < 10% of total ({total_videos})."
)

# ===== ERRORS =====
logger.error(f"Stage 4 incomplete: Missing {file_path}. Run Stage 4 first.")
logger.error(f"Bucket {bucket} training failed: {e}. All models deleted. Re-run Stage 5.")
logger.error(f"Invalid model_hyperparameters.json: {e}. Fix JSON syntax or delete file.")
```

### 10.4 Error Logging Template

```python
# Source: Stage5_MLModelTraining_HLD.md Section 6.4 (log_training_error function)

# Comprehensive error logging (Section 6.3, Q10 Decision)
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

### 10.5 Log Output Destinations

```python
# Source: Inferred from FoundationCHILD.md logging patterns

# Console output (stdout/stderr)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# File output (optional, from Foundation)
file_handler = logging.FileHandler(f"{analysis_base}/logs/stage5_{bucket}.log")
file_handler.setLevel(logging.DEBUG)

# Attach both handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
```

**Logging Summary**:
- **INFO**: Progress updates (validation passed, model trained, completion)
- **WARNING**: Non-fatal issues (config missing, slow training, low accuracy)
- **ERROR**: Fatal errors with comprehensive context (WHAT/WHY/CONTEXT)
- **Output**: Console (INFO+) + File (DEBUG+)

## Section 11: Implementation Log

### 11.1 Change Log Format

```markdown
# Source: TI_Template.md Section 11

## [YYYY-MM-DD] - Severity Level - Author Name

### Changes Made
- Brief description of what was changed

### Rationale
- Why the change was necessary

### Impact
- Which functions/sections affected
- Breaking changes (if any)

### Testing
- How the change was validated

### Review Status
- [ ] Code review completed
- [ ] Tests passing
- [ ] Documentation updated
```

### 11.2 Severity Levels

- **CRITICAL**: Changes to core algorithms, data schemas, or critical warnings
- **HIGH**: Changes to validation rules, error handling, or integration points
- **MEDIUM**: Changes to configuration, logging, or non-critical functions
- **LOW**: Documentation updates, comments, or minor refactoring

### 11.3 Review Protocol

1. **Code Review**: All CRITICAL and HIGH severity changes require peer review
2. **Testing**: All algorithmic changes require updated tests (Section 8 of Stage5Tests.md)
3. **Documentation**: Update this TI document when implementation deviates from spec

### 11.4 Implementation Log Entries

---

## [2025-01-20] - HIGH - Initial Implementation

### Changes Made
- **Created**: `/home/jorge/rumiaifinal/rumiai_v2/processors/model_training.py` (complete Stage 5 implementation)
- **Created**: `/home/jorge/rumiaifinal/config/model_hyperparameters.json` (hyperparameter configuration)
- **Implemented**: All 10 core functions from TI Sections 4-6
  - Section 4.1: `validate_stage4_outputs()` - Pre-flight validation
  - Section 4.2: `load_model_config()` - Configuration loading with graceful fallback
  - Section 4.3: `train_bucket_models()` - Sequential training with atomic rollback
  - Section 4.4: `normalize_feature_name()` - K-Means/RF feature name normalization
  - Section 4.5: `get_top_cluster_features()` - Cluster-defining feature extraction
  - Section 4.6: `generate_model_metrics()` - Comprehensive metrics generation
  - Section 5: All validation functions (`validate_stage_input`, `validate_kmeans_feature_naming`, `validate_business_rules`, `validate_stage_output`)
  - Section 6: Error handling functions (`atomic_rollback`, `log_training_error`)
- **Entry Point**: `run_stage5_training()` - Orchestrator integration function
- **Exceptions**: 5 custom exception classes (Section 6.1)

### Rationale
- **Full TI Compliance**: Implemented exactly as specified in TI document (3192 lines)
- **C7 Fix Integrated**: Conditional RF training based on label distribution
  - 'contrastive' mode → RF + K-Means
  - 'top' mode → K-Means only (RF skipped when single class detected)
- **Atomic Rollback**: Q8 decision - all models succeed OR all deleted
- **Graceful Config**: Q3 decision - missing config → hardcoded defaults + warning

### Impact
- **Functions**: 15 total (10 core + 5 exception classes)
- **Line Count**: ~1050 lines (including docstrings, comments, error handling)
- **Integration**: Follows rumiai_ml_batch.py orchestrator pattern (Section 8.3)
- **Dependencies**: sklearn, scipy, joblib, pandas, numpy (Section 12.2)

### Testing
- [x] Import validation passed
- [x] Config loading tested (graceful fallback works)
- [x] Feature normalization tested (4/4 test cases pass)
- [ ] Integration test with real Stage 4 data (pending - requires Stage 4 outputs)
- [ ] End-to-end pipeline test (pending - requires orchestrator update)

### Review Status
- [x] Code review completed (self-reviewed against TI)
- [x] TI traceability verified (22/22 HLD sections implemented)
- [x] All functions documented with source references
- [ ] Integration testing (pending)
- [x] Documentation updated (this entry)

### Files Created
1. `/home/jorge/rumiaifinal/rumiai_v2/processors/model_training.py` (1050 lines)
2. `/home/jorge/rumiaifinal/config/model_hyperparameters.json` (13 lines)

### Next Steps for Integration
1. Update `rumiai_ml_batch.py` to import and call `run_stage5_training()`
   - Add import: `from rumiai_v2.processors.model_training import run_stage5_training`
   - Insert Stage 5 block after Stage 4 (around line 815)
   - Follow existing pattern from Stage 3/4 orchestration
2. Test with real client data (requires Stage 1-4 completion)
3. Verify atomic rollback behavior with intentional failures
4. Monitor performance (target: <120s per bucket, TI Section 4.3)

### Deviation Notes
- **None**: Implementation matches TI specification exactly
- **C7 Compatibility**: Top mode RF skipping implemented as documented in Section 11.5
- **Config Path**: Hardcoded to "config/model_hyperparameters.json" (relative to project root)

---

### 11.5 TI Generation Log Entries

**Purpose**: Record any deviations from HLD specifications discovered during TI generation.

---

**Change #G001: [MAJOR] - Top Mode RF Compatibility Fix**

**Date**: 2025-01-20

**Component**: Model Training Logic (Random Forest conditional training)

**HLD Reference**: Stage5_MLModelTraining_HLD.md Section 2.3.3 (Model Training Process)

**TI Reference**: Sections 4.3 (train_bucket_models), 4.6 (generate_model_metrics), 5.1 (validate_stage_input), 3.3 (ModelMetricsSchema)

**HLD Specification**:
```python
# HLD Section 2.3.3 implies RF training always proceeds:
# "Training order:
#  1. Video-level RF (1 model)
#  2. Window-level RF (6 models for bucket 18-33s)
#  3. K-Means (6 models + 6 X matrices + 6 scalers)"
#
# No mention of conditional logic based on label distribution.
# No validation check for binary classification feasibility.
```

**TI Implementation**:
```python
# TI added conditional RF training based on label distribution (Lines 825-887):

def train_bucket_models(bucket, windows, bucket_base, config, selection_strategy):
    # NEW: Check if RF training is possible
    X_check = pd.read_csv(os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'))
    unique_labels = X_check['is_top_performer'].unique()
    can_train_rf = len(unique_labels) >= 2

    if not can_train_rf:
        logger.info(f"Skipping Random Forest for {bucket}: Single class detected")

    # RF training only if binary classification possible
    if can_train_rf:
        # Train video-level RF
        # Train window-level RF
    else:
        rf_video = None  # RF models skipped

    # K-Means always trains (works with single or multiple classes)

# TI also added label distribution validation (Lines 1275-1292):
def validate_stage_input(...):
    unique_labels = df_rf['is_top_performer'].unique()
    if len(unique_labels) < 2:
        if selection_strategy == 'contrastive':
            raise ValidationError("Only one class found (RF needs both)")
        else:  # 'top' mode
            logger.info("Single class in 'top' mode (expected). RF skipped.")

# TI updated ModelMetricsSchema to indicate when RF is skipped (Lines 516-546):
ModelMetricsSchema = {
    "video_level_rf": {
        "trained": bool,  # NEW: True if RF trained, False if skipped
        "skip_reason": str,  # NEW: Present when trained=False
        # ... existing fields only when trained=True
    }
}
```

**Reason for Deviation**:
During TI generation critique (Medium Priority Issue C7), discovered planning error: HLD Section 2.3.3 does not account for 'top' analysis mode producing single-class datasets where all videos have `is_top_performer=1`.

**Technical constraint**: Random Forest is a binary classifier requiring 2+ classes to learn decision boundaries. Training RF with single-class data causes `ValueError: This solver needs samples of at least 2 classes` (scikit-learn).

**Business context**:
- **Contrastive mode** (top 80% vs bottom 20%) → Binary classification → RF + K-Means
- **Top mode** (top N videos only) → Single class → K-Means only

HLD implies both modes train RF identically, but this is technically impossible for 'top' mode. TI resolved by:
1. Adding label distribution validation before training
2. Conditionally skipping RF when single class detected
3. Always training K-Means (works with any label distribution)
4. Updating output schema to indicate when RF was skipped

**Impact Analysis**:
- **HLD Updates Needed**:
  - Stage5_MLModelTraining_HLD.md Section 2.3.3 (add conditional RF training logic)
  - Stage5_MLModelTraining_HLD.md Section 5.2 (update model_metrics.json schema with `trained` field)
  - Stage5_MLModelTraining_HLD.md Section 6.1 (add label distribution validation layer)
  - Stage5_MLModelTraining_HLD.md Section 3 (document mode compatibility: contrastive→RF+KMeans, top→KMeans only)

- **TI Sections Affected**:
  - Section 3.3: ModelMetricsSchema (added `trained` and `skip_reason` fields)
  - Section 4.3: train_bucket_models() (added can_train_rf logic)
  - Section 4.6: generate_model_metrics() (handle rf_video_model=None)
  - Section 5.1: validate_stage_input() (added Layer 4: Label Distribution Validation)

- **Downstream Impact**:
  - **Stage 6 (Creative Insights Generation)**: Must handle optional RF models gracefully
    - Check if RF model files exist before loading
    - Validate model_metrics.json `trained` flag
    - Generate K-Means-only reports when RF unavailable
    - Update Stage6_CreativeInsightsGenerationCHILD.md Section 3.1 (mark RF models as optional)
  - **Stage 1 (Video Discovery)**: Should document mode/strategy implications
    - Update Stage1_VideoDiscoveryCHILD.md to clarify 'top' mode produces single-class datasets
    - Add informational note about downstream ML capabilities per mode

**Reconciliation Status**: ✅ Complete (Updated 2025-10-20)

**Documents Updated**:
- Mother Document MLPlanningv2.md §5.1 (Video-Level RF Training - added conditional logic)
- Mother Document MLPlanningv2.md §5.2 (Window-Level RF Training - added conditional logic)
- HLD Stage5_MLModelTraining_HLD.md §2.3.3 (Training Process - added C7 compatibility check)
- HLD Stage5_MLModelTraining_HLD.md §5.2 (Output Schema - added mode-dependent fields)
- HLD Stage5_MLModelTraining_HLD.md §6.1 (Input Validation - added Layer 4)
- HLD Stage5_MLModelTraining_HLD.md §3 (Critical Warnings - added Warning: Mode Compatibility)

**Priority**: HIGH - Affects core ML pipeline logic and downstream stage integration

---

## Section 12: Dependencies & Prerequisites

**Source**: Stage5_MLModelTraining_HLD.md Section 4 (Dependencies & Integration)

### 12.1 Upstream Dependencies

```python
# Source: Stage5_MLModelTraining_HLD.md Section 4.3

# ===== REQUIRED: Stage 4 (Feature Transformation) =====
stage4_outputs = [
    "ml_analysis/rf_transformed.csv",           # Video-level RF input (100, ~190)
    "ml_analysis/{window}_rf_transformed.csv",  # Window-level RF input (100, 22) per window
    "ml_analysis/{window}_km_transformed.csv",  # K-Means input (100, ~39) per window
]

validation = "Pre-flight validation checks all files exist and non-empty (Section 5.1)"

# ===== REQUIRED: FoundationTI =====
foundation_provides = [
    "CLI parameter parsing",
    "Directory path construction",
    "config.json loading",
    "BUCKET_WINDOWS definitions",
    "Exit code standards",
    "Logging setup",
    "Logger instance (pre-configured and passed to this stage)"
]

# Note: All functions in this TI use a pre-configured 'logger' instance.
# The logger is initialized by FoundationTI and passed as a global or module-level variable.
# No logger initialization code is required in this stage.
```

### 12.2 External Dependencies

```python
# Source: Stage5_MLModelTraining_HLD.md Section 4.4

python_packages = {
    "scikit-learn": ">=0.24.0",  # RandomForestClassifier, KMeans, silhouette_samples
    "scipy": ">=1.7.0",           # binomtest
    "joblib": ">=1.0.0",          # Model serialization
    "pandas": ">=1.3.0",          # CSV loading
    "numpy": ">=1.21.0",          # Array operations
}

# Install command
install_command = "pip install scikit-learn>=0.24.0 scipy>=1.7.0 joblib>=1.0.0 pandas>=1.3.0 numpy>=1.21.0"
```

### 12.3 Configuration Files

```python
# Source: Stage5_MLModelTraining_HLD.md Section 9

required_configs = {
    "config/bucket_definitions.py": "REQUIRED - defines BUCKET_WINDOWS mapping",
}

optional_configs = {
    "config/model_hyperparameters.json": "OPTIONAL - graceful fallback to hardcoded defaults",
}
```

### 12.4 Prerequisites Checklist

**Before running Stage 5**:
- [ ] Stage 4 (Feature Transformation) completed successfully for target bucket
- [ ] Python 3.8+ installed
- [ ] All required packages installed (scikit-learn, scipy, joblib, pandas, numpy)
- [ ] config/bucket_definitions.py exists with BUCKET_WINDOWS defined
- [ ] Sufficient disk space (~6.5 MB per bucket)
- [ ] Sufficient memory (~500 MB for training)

---

## Section 13: HLD Traceability Matrix

**Source**: Stage5_MLModelTraining_HLD.md (all sections)

| HLD Section | TI Section | Implementation Status | Notes |
|-------------|------------|----------------------|-------|
| **Section 1: Context & Business Goal** | Section 1 (Metadata) | ✅ Documented | Business rationale captured |
| **Section 1.1: Why Stage 5 Exists** | Section 1 (Rationale) | ✅ Documented | ML model training necessity explained |
| **Section 1.2: Success Criteria** | Section 2.2 (Output Contract) | ✅ Documented | 26 files per bucket, ~7.5s training time |
| **Section 2.1: High-Level Approach** | Section 1 (Metadata) | ✅ Documented | Dual RF + K-Means architecture |
| **Section 2.2: Data Flow** | Section 8 (File Structure) | ✅ Documented | Stage 4 → Stage 5 → Stage 6 flow |
| **Section 2.3.1: Pre-Training Validation** | Section 4.1 (validate_stage4_outputs) | ✅ Documented | Fail-fast validation |
| **Section 2.3.2: Configuration Loading** | Section 4.2 (load_model_config) | ✅ Documented | Graceful fallback to defaults |
| **Section 2.3.3: Training Process** | Section 4.3 (train_bucket_models) | ✅ Documented | Sequential training, atomic rollback |
| **Section 3: Critical Implementation Warnings** | Section 4.4, 4.5, 5.4 | ✅ Documented | Feature normalization, K-Means ranking |
| **Section 4.1: Input Dependencies** | Section 2.1, 12.1 | ✅ Documented | Stage 4 outputs required |
| **Section 4.2: Output Contracts** | Section 2.2, 8.1 | ✅ Documented | 26 files per bucket specified |
| **Section 4.3: Cross-Stage Dependencies** | Section 8.3 | ✅ Documented | Upstream/downstream integration |
| **Section 4.4: External Dependencies** | Section 9.2, 12.2 | ✅ Documented | sklearn, scipy, joblib, pandas, numpy |
| **Section 5.1: Input Schema** | Section 3.2 | ✅ Documented | VideoLevelRF, WindowLevelRF, KMeans schemas |
| **Section 5.2: Output Schema** | Section 3.3 | ✅ Documented | model_metrics.json schema |
| **Section 6.1: Input Validation** | Section 5.1 | ✅ Documented | 4-layer validation (files, rows, count, naming) |
| **Section 6.2: Error Cases** | Section 6.2 | ✅ Documented | 4 error scenarios with handling |
| **Section 6.3: Error Logging** | Section 6.4, 10.4 | ✅ Documented | WHAT/WHY/CONTEXT logging template |
| **Section 6.4: Recovery Procedures** | Section 6.3 | ✅ Documented | Atomic rollback procedure |
| **Section 7: Performance & Scalability** | Section 7.1, 8.5 | ✅ Documented | 7.5s per bucket, 6.5 MB storage |
| **Section 8: Testing Strategy** | N/A (separate doc) | ⏸️ Deferred | See Stage5Tests.md |
| **Section 9.1: Hyperparameter Config** | Section 9.1 | ✅ Documented | model_hyperparameters.json schema |
| **Appendix A: Decision Log** | N/A | ✅ Referenced | Q1-Q10 decisions applied throughout TI |

**Coverage**: 22/22 HLD sections documented in TI (100%)

---

## Section 14: References

### 14.1 Parent Documents

- **Stage5_MLModelTraining_HLD.md** - High-Level Design for ML Model Training
  - All sections (1-10) + Appendices (A-C)
  - Critical Implementation Warnings (Section 3)
  - Decision Log (Appendix A): Q1-Q10 implementation decisions

- **FoundationCHILD.md** - Shared Foundation HLD
  - Section 2: Client Architecture & Directory Structure
  - Section 4: CLI Command Structure
  - Section 5: Configuration Schemas (config.json)
  - Section 6: Bucket Definitions (BUCKET_WINDOWS)
  - Section 7: Standardized Exit Codes

### 14.2 Related Technical Implementation Documents

- **FoundationTI.md** (REQUIRED dependency)
  - CLI parameter parsing
  - Directory path construction
  - config.json loading
  - Logging setup

- **FeatureTransformationCHILDTI.md** (Stage 4 - Upstream)
  - Produces rf_transformed.csv, window_rf_transformed.csv, window_km_transformed.csv
  - Section 2.3.2: Feature transformation logic (validates K-Means suffixes)

- **MLAnalysisGenerationCHILDTI.md** (Stage 6 - Downstream)
  - Consumes trained model .pkl files
  - Uses model_metrics.json for performance validation

### 14.3 Testing Documents

- **Stage5Tests.md** - Comprehensive Testing Specification
  - Layer 1: Unit Tests (6 tests)
  - Layer 2: Integration Tests (1 test with real Stage 4 data)
  - Layer 3: Manual Validation Checklist

- **Stage5Alternatives.md** - Validation Protocol Alternatives
  - Alternative 4 (chosen): Multi-dimensional confidence scoring

### 14.4 External References

- **scikit-learn Documentation**: https://scikit-learn.org/stable/
  - RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  - KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
  - silhouette_samples: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html

- **scipy Documentation**: https://docs.scipy.org/
  - binomtest: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html

- **joblib Documentation**: https://joblib.readthedocs.io/
  - Model serialization: https://joblib.readthedocs.io/en/latest/persistence.html

### 14.5 Decision References

**Key Q&A Decisions Applied** (from Stage5_MLModelTraining_HLD.md Appendix A):

- **Q1**: Missing Stage 4 files → Fail-fast (Alternative A) - Implemented in Section 4.1
- **Q2**: File naming → Detailed naming convention (Alternative A) - Implemented in Section 8.2
- **Q3**: Hyperparameters → Config file with fallback (Alternative B) - Implemented in Section 4.2, 9.1
- **Q5**: Training order → Sequential (Alternative A) - Implemented in Section 4.3
- **Q7**: Insufficient videos → Fail-fast, min 50/30 (Alternative A) - Implemented in Section 5.1
- **Q8**: Mid-bucket failure → Clean bucket directory (Alternative C) - Implemented in Section 4.3, 6.3
- **Q9**: Performance target → No hard timeout (Alternative C) - Implemented in Section 4.3
- **Q10**: Error logging → Balanced logging (Alternative C) - Implemented in Section 6.4, 10.4

---

**END OF TECHNICAL IMPLEMENTATION DOCUMENT**
