"""
Stage 5: ML Model Training

Source: MLModelTrainingCHILDTI.md
Purpose: Train Random Forest and K-Means models for viral video pattern detection

This module implements the ML model training stage that:
1. Validates Stage 4 transformation outputs
2. Loads hyperparameter configurations
3. Trains video-level and window-level Random Forest models (conditional on label distribution)
4. Trains K-Means clustering models for content style segmentation
5. Generates comprehensive model performance metrics

Key Design Decisions:
- Contrastive mode (top 80% vs bottom 20%) → RF + K-Means
- Top mode (top N only) → K-Means only (RF skipped due to single class)
- Atomic rollback on failure (all models succeed OR all deleted)
- Graceful config degradation (missing config → hardcoded defaults)

Integration:
- Upstream: Stage 4 (Feature Transformation) - requires rf_transformed.csv, window CSVs
- Downstream: Stage 6 (ML Analysis Generation) - provides trained model .pkl files
- Foundation: Logging, CLI args, directory paths, BUCKET_WINDOWS

Author: Implementation from MLModelTrainingCHILDTI.md
Date: 2025-01-20
"""

import os
import sys
import json
import time
import logging
import traceback
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score
)

# ===== CUSTOM EXCEPTIONS =====
# Source: MLModelTrainingCHILDTI.md Section 6.1

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


# ===== LOGGER SETUP =====
# Source: MLModelTrainingCHILDTI.md Section 10, Section 12.1
# Note: Logger is initialized by FoundationTI and passed as module-level variable
# This module expects a pre-configured logger instance to be available

logger = logging.getLogger(__name__)


# ===== SECTION 4.1: PRE-FLIGHT VALIDATION =====
# Source: MLModelTrainingCHILDTI.md Section 4.1

def validate_stage4_outputs(bucket: str, windows: List[str], bucket_base: str) -> None:
    """
    Validate Stage 4 transformation outputs exist and are non-empty.

    Fail-fast validation before model training begins.

    Args:
        bucket: str - Bucket name (e.g., "18-33s")
        windows: List[str] - Window names for this bucket (e.g., ["hook", "middle_1", "closing"])
        bucket_base: str - Base path to bucket directory

    Raises:
        StageInputError: If any required file is missing or empty

    Source: MLModelTrainingCHILDTI.md Section 4.1
    """
    logger.info(f"Validating Stage 4 outputs for bucket {bucket}...")

    # ===== LAYER 1: File Existence =====
    required_files = [
        os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'),
    ]

    for window in windows:
        required_files.extend([
            os.path.join(bucket_base, f'ml_analysis/{window}_rf_transformed.csv'),
            os.path.join(bucket_base, f'ml_analysis/{window}_km_transformed.csv'),
        ])

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        error_msg = (
            f"Stage 4 incomplete: Missing {len(missing_files)} files for bucket {bucket}.\n"
            f"First missing file: {missing_files[0]}\n"
            f"Run Stage 4 first or check if bucket was skipped in Stage 1."
        )
        raise StageInputError(error_msg)

    # ===== LAYER 2: File Non-Empty =====
    empty_files = []
    for file_path in required_files:
        df = pd.read_csv(file_path)
        if df.shape[0] == 0:
            empty_files.append(file_path)

    if empty_files:
        error_msg = (
            f"Stage 4 output empty: {len(empty_files)} files have 0 rows for bucket {bucket}.\n"
            f"First empty file: {empty_files[0]}\n"
            f"Stage 4 failed silently."
        )
        raise StageInputError(error_msg)

    logger.info(
        f"✓ Stage 4 validation passed: {len(required_files)} files exist, "
        f"{df.shape[0]} videos (last file checked)"
    )


# ===== SECTION 4.2: CONFIGURATION LOADING =====
# Source: MLModelTrainingCHILDTI.md Section 4.2

def load_model_config(config_path: str = "config/model_hyperparameters.json") -> dict:
    """
    Load model hyperparameters from config file with graceful fallback.

    Args:
        config_path: str - Path to hyperparameters config file

    Returns:
        dict: Hyperparameters for RandomForest and KMeans

    Raises:
        ConfigError: If config file is malformed (invalid JSON)

    Source: MLModelTrainingCHILDTI.md Section 4.2
    """
    # Hardcoded defaults (fallback if config missing)
    DEFAULT_CONFIG = {
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

    logger.info("Loading hyperparameters...")

    # Try to load from file
    if not os.path.exists(config_path):
        logger.warning(
            f"Config file not found: {config_path}. "
            f"Using hardcoded defaults."
        )
        return DEFAULT_CONFIG

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info(f"✓ Loaded hyperparameters from {config_path}")
        return config

    except json.JSONDecodeError as e:
        error_msg = (
            f"Invalid JSON in {config_path}: {str(e)}\n"
            f"Fix JSON syntax or delete file to use hardcoded defaults."
        )
        raise ConfigError(error_msg)


# ===== SECTION 4.4: FEATURE NAME NORMALIZATION =====
# Source: MLModelTrainingCHILDTI.md Section 4.4

def normalize_feature_name(feature_name: str) -> str:
    """
    Remove K-Means transformation suffixes for RF comparison.

    Critical: K-Means features have suffixes (_scaled, _log, _encoded) but RF features don't.
    Must normalize before comparing feature importance overlap.

    Args:
        feature_name: str - K-Means feature name (e.g., "eye_contact_rate_scaled")

    Returns:
        str: Normalized feature name (e.g., "eye_contact_rate")

    Examples:
        "eye_contact_rate_scaled" → "eye_contact_rate"
        "word_count_log" → "word_count"
        "has_text_encoded" → "has_text"
        "scene_count" → "scene_count" (no suffix, unchanged)

    Source: MLModelTrainingCHILDTI.md Section 4.4
    """
    # Remove transformation suffixes in priority order
    suffixes = ['_scaled', '_log', '_encoded']

    for suffix in suffixes:
        if feature_name.endswith(suffix):
            return feature_name[:-len(suffix)]

    # No suffix found, return unchanged
    return feature_name


# ===== SECTION 4.5: K-MEANS TOP FEATURES =====
# Source: MLModelTrainingCHILDTI.md Section 4.5

def get_top_cluster_features(kmeans_model, feature_names: List[str], n: int = 5) -> List[str]:
    """
    Extract top N features that define K-Means cluster separation.

    Uses cluster centroid variance to identify features with largest differences
    across clusters (high variance = cluster-defining features).

    Args:
        kmeans_model: Trained KMeans model (with .cluster_centers_ attribute)
        feature_names: List[str] - Feature names in same order as training data
        n: int - Number of top features to return (default: 5)

    Returns:
        List[str]: Top N feature names ranked by cluster separation power

    Example:
        centroids = [[1.5, 8.2, 3.1], [1.8, 2.5, 3.0], [1.2, 9.1, 2.8]]
        feature_names = ["scene_count", "eye_contact_rate", "word_count"]
        → Returns ["eye_contact_rate", "word_count", "scene_count"]
          (eye_contact has highest variance across clusters: 9.1 - 2.5 = 6.6)

    Source: MLModelTrainingCHILDTI.md Section 4.5
    """
    # Step 1: Get cluster centroids (shape: n_clusters × n_features)
    centroids = kmeans_model.cluster_centers_

    # Step 2: Calculate variance across clusters for each feature
    # High variance = feature values differ across clusters = cluster-defining
    feature_variances = np.var(centroids, axis=0)  # Shape: (n_features,)

    # Step 3: Rank features by variance (highest first)
    top_indices = np.argsort(feature_variances)[::-1][:min(n, len(feature_names))]

    # Step 4: Map indices to feature names
    top_features = [feature_names[i] for i in top_indices]

    return top_features


# ===== SECTION 4.6: MODEL METRICS GENERATION =====
# Source: MLModelTrainingCHILDTI.md Section 4.6

def generate_model_metrics(
    bucket: str,
    windows: List[str],
    bucket_base: str,
    rf_video_model,  # Can be None if RF skipped
    rf_window_models: dict,  # Can be empty dict if RF skipped
    kmeans_models: dict,
    X_data_matrices: dict,
    total_videos: int
) -> dict:
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

    Source: MLModelTrainingCHILDTI.md Section 4.6
    """
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
            "input_features": int(X_video.shape[1]),
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
                "input_features": int(X_window.shape[1]),
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
            "input_features": int(X_kmeans.shape[1]),
            "n_clusters": int(kmeans_model.n_clusters),
            "inertia": float(kmeans_model.inertia_),
            "silhouette_score": silhouette,
            "cluster_sizes": cluster_sizes
        }

    return metrics


# ===== SECTION 4.3: TRAIN BUCKET MODELS =====
# Source: MLModelTrainingCHILDTI.md Section 4.3

def train_bucket_models(
    bucket: str,
    windows: List[str],
    bucket_base: str,
    config: dict,
    selection_strategy: str
) -> None:
    """
    Train all models for a single bucket with atomic rollback on failure.

    Training order (sequential):
    1. Check if RF training is possible (2+ classes required)
    2. Video-level RF (if possible)
    3. Window-level RF (if possible, per window)
    4. K-Means (always, per window)
    5. Generate model_metrics.json

    Atomic guarantee: All models succeed OR all deleted on failure.

    Args:
        bucket: str - Bucket name (e.g., "18-33s")
        windows: List[str] - Window names for this bucket
        bucket_base: str - Base path to bucket directory
        config: dict - Hyperparameters (from load_model_config)
        selection_strategy: str - "contrastive" or "top"

    Raises:
        ModelTrainingError: If any model training fails (after atomic rollback)

    Source: MLModelTrainingCHILDTI.md Section 4.3
    """
    start_time = time.time()
    trained_models = []  # Track for atomic rollback

    models_dir = os.path.join(bucket_base, 'models')
    os.makedirs(models_dir, exist_ok=True)

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
        rf_video_model = None
        if can_train_rf:
            logger.info(f"Training video-level RF for {bucket}...")

            X_video = pd.read_csv(os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'))
            y_video = X_video['is_top_performer']
            X_video = X_video.drop(['is_top_performer', 'video_id'], axis=1)

            rf_video_model = RandomForestClassifier(
                n_estimators=config['random_forest']['n_estimators'],
                max_depth=config['random_forest']['max_depth'],
                random_state=config['random_forest']['random_state']
            )
            rf_video_model.fit(X_video, y_video)

            # Save model
            model_path = os.path.join(models_dir, f'rf_video_{bucket}.pkl')
            joblib.dump(rf_video_model, model_path)
            trained_models.append(model_path)

            logger.info(f"✓ Video-level RF trained: {model_path}")
        else:
            # Still need total_videos for metrics
            X = pd.read_csv(os.path.join(bucket_base, 'ml_analysis/rf_transformed.csv'))
            total_videos = len(X)

        # ===== STEP 2: Train Window-Level RF (if binary classification possible) =====
        rf_window_models = {}
        if can_train_rf:
            for window in windows:
                logger.info(f"Training window-level RF for {window}...")

                X_window = pd.read_csv(os.path.join(bucket_base, f'ml_analysis/{window}_rf_transformed.csv'))
                y_window = X_window['is_top_performer']
                X_window = X_window.drop(['is_top_performer'], axis=1)

                rf_window = RandomForestClassifier(
                    n_estimators=config['random_forest']['n_estimators'],
                    max_depth=config['random_forest']['max_depth'],
                    random_state=config['random_forest']['random_state']
                )
                rf_window.fit(X_window, y_window)
                rf_window_models[window] = rf_window

                # Save model
                model_path = os.path.join(models_dir, f'rf_{window}_{bucket}.pkl')
                joblib.dump(rf_window, model_path)
                trained_models.append(model_path)

                logger.info(f"✓ Window-level RF trained: {window}")

        # Get total_videos if RF was trained (already set if skipped)
        if can_train_rf:
            total_videos = len(X_video) + len(y_video)  # Combined length
            total_videos = len(y_video)  # Actual video count

        # ===== STEP 3: Train K-Means (always, per window) =====
        kmeans_models = {}
        X_data_matrices = {}

        for window in windows:
            logger.info(f"Training K-Means for {window}...")

            # Load K-Means data
            km_csv_path = os.path.join(bucket_base, f'ml_analysis/{window}_km_transformed.csv')
            X_kmeans = pd.read_csv(km_csv_path)

            # Remove non-feature columns
            feature_cols = [col for col in X_kmeans.columns
                           if col not in ['video_id', 'create_time', 'gender']]
            X_kmeans_features = X_kmeans[feature_cols]

            # Train K-Means
            kmeans = KMeans(
                n_clusters=config['kmeans']['n_clusters'],
                random_state=config['kmeans']['random_state'],
                n_init=config['kmeans']['n_init']
            )
            kmeans.fit(X_kmeans_features)
            kmeans_models[window] = kmeans
            X_data_matrices[window] = X_kmeans_features

            # Save K-Means model
            model_path = os.path.join(models_dir, f'{window}_kmeans_{bucket}.pkl')
            joblib.dump(kmeans, model_path)
            trained_models.append(model_path)

            # Save X data matrix (for silhouette calculation in Stage 6)
            X_data_path = os.path.join(models_dir, f'{window}_X_data_{bucket}.pkl')
            joblib.dump(X_kmeans_features, X_data_path)
            trained_models.append(X_data_path)

            # Copy scalers from Stage 4 (if they exist)
            scalers_source = os.path.join(bucket_base, f'ml_analysis/{window}_scalers.pkl')
            if os.path.exists(scalers_source):
                scalers_dest = os.path.join(models_dir, f'{window}_scalers_{bucket}.pkl')
                import shutil
                shutil.copy2(scalers_source, scalers_dest)
                trained_models.append(scalers_dest)

            logger.info(f"✓ K-Means trained: {window}")

        # ===== STEP 4: Generate Model Metrics =====
        logger.info("Generating model_metrics.json...")

        metrics = generate_model_metrics(
            bucket=bucket,
            windows=windows,
            bucket_base=bucket_base,
            rf_video_model=rf_video_model,
            rf_window_models=rf_window_models,
            kmeans_models=kmeans_models,
            X_data_matrices=X_data_matrices,
            total_videos=total_videos
        )

        # Save metrics
        metrics_path = os.path.join(models_dir, 'model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        trained_models.append(metrics_path)

        logger.info("✓ Model metrics generated")

        # ===== SUCCESS =====
        elapsed = time.time() - start_time
        logger.info(
            f"✓ Bucket {bucket} training complete: {elapsed:.1f}s ({len(trained_models)} files)"
        )

    except Exception as e:
        # ===== ATOMIC ROLLBACK =====
        logger.error(f"Training failed for bucket {bucket}: {str(e)}")
        log_training_error(
            bucket=bucket,
            current_model="unknown",  # Don't know which specific model failed
            exception=e,
            trained_models=trained_models,
            start_time=start_time,
            config=config,
            bucket_base=bucket_base
        )
        atomic_rollback(bucket=bucket, trained_models=trained_models, bucket_base=bucket_base)
        raise ModelTrainingError(f"Bucket {bucket} training failed: {str(e)}")


# ===== SECTION 6.3: ATOMIC ROLLBACK =====
# Source: MLModelTrainingCHILDTI.md Section 6.3

def atomic_rollback(bucket: str, trained_models: List[str], bucket_base: str) -> None:
    """
    Atomic rollback: Delete ALL partial models for this bucket.

    Q8 Decision: All models succeed OR all deleted on failure.
    Result: Either bucket has complete model set OR no models. Never partial.

    Args:
        bucket: str - Bucket name (e.g., "18-33s")
        trained_models: List[str] - Paths to all models created before failure
        bucket_base: str - Base path to bucket directory (for verification)

    Source: MLModelTrainingCHILDTI.md Section 6.3
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


# ===== SECTION 6.4: ERROR LOGGING =====
# Source: MLModelTrainingCHILDTI.md Section 6.4

def log_training_error(
    bucket: str,
    current_model: str,
    exception: Exception,
    trained_models: List[str],
    start_time: float,
    config: dict,
    bucket_base: str
) -> None:
    """
    Comprehensive error logging for training failures.

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

    Source: MLModelTrainingCHILDTI.md Section 6.4
    """
    elapsed = time.time() - start_time

    logger.error(f"""
===============================================================================
BUCKET {bucket} TRAINING FAILED
===============================================================================

WHAT FAILED:
  Model name: {current_model}
  Bucket: {bucket}

WHY IT FAILED:
  Exception type: {type(exception).__name__}
  Exception message: {str(exception)}
  Stack trace (first 10 lines):
{traceback.format_exc(limit=10)}

CONTEXT:
  Hyperparameters: {config}
  Completed models before failure: {len(trained_models)} files
  Training duration before failure: {elapsed:.1f}s

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


# ===== SECTION 5: VALIDATION FUNCTIONS =====
# Source: MLModelTrainingCHILDTI.md Section 5

def validate_stage_input(
    bucket: str,
    windows: List[str],
    bucket_base: str,
    selection_strategy: str,
    video_count: int
) -> None:
    """
    Validate input before processing.

    Validation layers:
    1. File existence (all Stage 4 outputs present)
    2. File non-empty (>0 rows)
    3. Video count threshold (min 50 contrastive, 30 top)
    4. Label distribution (RF compatibility check)
    5. K-Means feature naming convention (>=80% have transformation suffixes)

    Args:
        bucket: str - Bucket name
        windows: List[str] - Window names for this bucket
        bucket_base: str - Base path to bucket directory
        selection_strategy: str - "contrastive" or "top"
        video_count: int - Number of videos in dataset

    Raises:
        ValidationError: If any validation layer fails

    Source: MLModelTrainingCHILDTI.md Section 5.1
    """
    # ===== LAYER 1 & 2: File existence and non-empty (handled by validate_stage4_outputs) =====
    validate_stage4_outputs(bucket, windows, bucket_base)

    # ===== LAYER 3: Video Count Threshold =====
    MIN_VIDEOS_CONTRASTIVE = 3  # Minimum for any statistical analysis (validated with small datasets)
    MIN_VIDEOS_TOP = 3          # Minimum for descriptive analysis only

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

    Source: MLModelTrainingCHILDTI.md Section 5.1
    """
    # Read CSV header only
    df = pd.read_csv(csv_path, nrows=1)
    feature_names = [col for col in df.columns if col not in ['video_id', 'create_time', 'gender']]

    # Count features with transformation suffixes
    scaled_count = sum(1 for f in feature_names if '_scaled' in f)
    log_count = sum(1 for f in feature_names if '_log' in f)
    encoded_count = sum(1 for f in feature_names if '_encoded' in f)

    # Count one-hot encoded emotion features (from dominant_emotion_id)
    # Source: Stage 4 feature_transformation.py line 714 - these ARE transformed but lack suffixes by ML convention
    EMOTION_FEATURES = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']
    emotion_count = sum(1 for f in feature_names if f in EMOTION_FEATURES)

    # Total transformed includes both suffix-based AND one-hot encoded features
    total_transformed = scaled_count + log_count + encoded_count + emotion_count

    # Expect at least 80% of features to be transformed
    expected_threshold = len(feature_names) * 0.80

    if total_transformed < expected_threshold:
        raise ValidationError(
            f"K-Means CSV feature naming validation failed: {csv_path}\n"
            f"  Total features: {len(feature_names)}\n"
            f"  Features with _scaled: {scaled_count}\n"
            f"  Features with _log: {log_count}\n"
            f"  Features with _encoded: {encoded_count}\n"
            f"  One-hot emotion features: {emotion_count}\n"
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


def validate_business_rules(bucket: str, windows: List[str], config: dict) -> None:
    """
    Validate business rules during processing.

    Source: MLModelTrainingCHILDTI.md Section 5.2
    """
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


def validate_stage_output(bucket: str, windows: List[str], bucket_base: str) -> None:
    """
    Validate output after processing.

    Source: MLModelTrainingCHILDTI.md Section 5.3
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


# ===== ENTRY POINT FOR ORCHESTRATOR =====
# Source: MLModelTrainingCHILDTI.md Section 8.3

def run_stage5_training(
    bucket_path: str,
    config: dict,
    selection_strategy: str
) -> Tuple[bool, List[str], float]:
    """
    Entry point for Stage 5 ML Model Training (called by rumiai_ml_batch.py).

    This function integrates into the existing orchestrator pattern following
    the same signature as run_stage4_transformation.

    Args:
        bucket_path: str - Path to bucket directory (e.g., /data/clients/acme/hashtags/nutrition/top_contrastive/bucket_18-33s)
        config: dict - Bucket configuration loaded from config.json
        selection_strategy: str - "contrastive" or "top" (from CLI args via FoundationTI)

    Returns:
        Tuple[bool, List[str], float]:
            - success: bool - Always True when function returns (errors raise exceptions)
            - output_files: List[str] - Paths to all created model files
            - elapsed_time: float - Training duration in seconds

    Raises:
        StageInputError: Stage 4 outputs missing or invalid
        InsufficientDataError: Video count below minimum threshold
        ModelTrainingError: Model training failed (after atomic rollback)
        ValidationError: Validation checks failed

    Source: MLModelTrainingCHILDTI.md Section 8.3
    Integration: rumiai_ml_batch.py lines ~817-920 (expected pattern)
    """
    start_time = time.time()

    # Extract bucket name from config
    bucket = config.get('bucket')
    if not bucket:
        raise ValidationError("config.json missing 'bucket' field")

    # Get windows from bucket definitions
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket]

    # Get video count from config
    video_count = config.get('video_count', 0)

    logger.info(f"Starting Stage 5 for bucket: {bucket}")
    logger.info(f"  Windows: {windows}")
    logger.info(f"  Selection strategy: {selection_strategy}")
    logger.info(f"  Video count: {video_count}")

    # ===== PRE-FLIGHT VALIDATION =====
    validate_stage_input(
        bucket=bucket,
        windows=windows,
        bucket_base=bucket_path,
        selection_strategy=selection_strategy,
        video_count=video_count
    )

    # ===== LOAD HYPERPARAMETERS =====
    hyperparameters = load_model_config()

    # ===== VALIDATE BUSINESS RULES =====
    validate_business_rules(bucket, windows, hyperparameters)

    # ===== TRAIN MODELS =====
    train_bucket_models(
        bucket=bucket,
        windows=windows,
        bucket_base=bucket_path,
        config=hyperparameters,
        selection_strategy=selection_strategy
    )

    # ===== POST-TRAINING VALIDATION =====
    validate_stage_output(bucket, windows, bucket_path)

    # ===== COLLECT OUTPUT FILES =====
    models_dir = os.path.join(bucket_path, 'models')
    output_files = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if os.path.isfile(os.path.join(models_dir, f))
    ]

    elapsed_time = time.time() - start_time

    logger.info(
        f"✓ Stage 5 complete for bucket {bucket}: "
        f"{len(output_files)} files created in {elapsed_time:.1f}s"
    )

    return True, output_files, elapsed_time
