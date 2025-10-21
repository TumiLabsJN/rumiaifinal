"""
Stage 4: Feature Transformation

Transforms aggregated features from Stage 3 into ML-ready formats:
- Video-Level Random Forest (cross-window patterns)
- Window-Level Random Forest (isolated window analysis)
- Window-Level K-Means (distance-based clustering)

Source: FeatureTransformationTI.md
Version: 1.0
Last Updated: 2025-01-28
"""

import pandas as pd
import numpy as np
import os
import logging
import json
import psutil
import time
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 4.6: HELPER FUNCTIONS
# ============================================================================

def get_expected_column_count(bucket: str) -> int:
    """
    Get expected input column count for bucket.

    Source: FeatureTransformationTI.md Section 4.6

    Args:
        bucket: str, bucket name (e.g., "18-33s")

    Returns:
        int, expected column count
    """
    from config.bucket_definitions import BUCKET_WINDOWS
    window_count = len(BUCKET_WINDOWS[bucket])
    return (21 * window_count) + 3


def get_required_columns(bucket: str) -> List[str]:
    """
    Get list of required column names for bucket.

    Source: FeatureTransformationTI.md Section 4.6

    Args:
        bucket: str, bucket name (e.g., "18-33s")

    Returns:
        list of str, required column names
    """
    from config.bucket_definitions import BUCKET_WINDOWS
    BASE_FEATURES = get_base_features()

    required = []
    for window in BUCKET_WINDOWS[bucket]:
        required.extend([f'{window}_{feat}' for feat in BASE_FEATURES])
    required.extend(['video_id', 'create_time', 'gender'])
    return required


def get_expected_output_files(bucket: str) -> List[str]:
    """
    Get list of expected output filenames.

    Source: FeatureTransformationTI.md Section 4.6

    Args:
        bucket: str, bucket name (e.g., "18-33s")

    Returns:
        list of str, expected output filenames
    """
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket]

    files = ['rf_transformed.csv']
    for window in windows:
        files.append(f'{window}_rf_transformed.csv')
        files.append(f'{window}_km_transformed.csv')
    return files


def write_checkpoint(checkpoint: dict, bucket_base: str) -> None:
    """
    Write stage checkpoint to disk.

    Source: FeatureTransformationTI.md Section 4.6

    Args:
        checkpoint: dict, checkpoint data
        bucket_base: str, base directory for checkpoint

    Raises:
        IOError: if checkpoint write fails
    """
    checkpoint_dir = os.path.join(bucket_base, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, 'stage_4_checkpoint.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def get_expected_rf_column_count(bucket: str) -> int:
    """
    Get expected Video-Level RF output column count.

    Source: FeatureTransformationTI.md Section 4.6

    Args:
        bucket: str, bucket name (e.g., "18-33s")

    Returns:
        int, expected column count
    """
    from config.bucket_definitions import BUCKET_WINDOWS
    window_count = len(BUCKET_WINDOWS[bucket])
    temporal_features = 21 * window_count
    # temporal + emotions(7) + temporal_extract(5) + gender(3) + cross_window(5) + target(1)
    # Note: has_captions NOT one-hot encoded, remains Boolean in temporal features
    return temporal_features + 7 + 5 + 3 + 5 + 1


def get_base_features() -> List[str]:
    """
    Get list of 21 base features.

    Source: FeatureTransformationTI.md Section 4.6

    Returns:
        list of str, 21 base feature names
    """
    return [
        'average_face_size', 'overlay_unique_count', 'scene_count', 'shortest_scene',
        'longest_scene', 'scene_duration_variance', 'object_count', 'person_count',
        'eye_contact_rate', 'gaze_variance', 'gesture_count', 'speech_coverage',
        'word_count', 'energy_level', 'energy_variance', 'energy_max',
        'pitch_scatter_ratio', 'dominant_emotion_id', 'emotional_valence',
        'emotion_consistency', 'has_captions'
    ]


# ============================================================================
# SECTION 10.4: METRICS COLLECTION
# ============================================================================

class MetricsCollector:
    """
    Collects and logs Stage 4 performance metrics (thread-safe).

    Source: FeatureTransformationTI.md Section 10.4
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}
        self.start_time = None
        self.process = psutil.Process()
        self.lock = threading.Lock()  # Thread safety for concurrent metric updates

    def start_stage(self):
        """Start timer and baseline memory."""
        with self.lock:
            self.start_time = time.time()
            self.metrics['baseline_memory_mb'] = self.process.memory_info().rss / 1024 / 1024

    def record_input(self, video_count: int, column_count: int, file_size_mb: float):
        """Record input metrics."""
        with self.lock:
            self.metrics['input_video_count'] = video_count
            self.metrics['input_column_count'] = column_count
            self.metrics['input_file_size_mb'] = file_size_mb
        self.logger.info(f"METRIC: input_video_count={video_count}")
        self.logger.info(f"METRIC: input_column_count={column_count}")

    def record_transformation_time(self, phase: str, elapsed: float):
        """Record transformation phase timing."""
        metric_name = f"{phase}_duration_seconds"
        with self.lock:
            self.metrics[metric_name] = elapsed
        self.logger.info(f"METRIC: {metric_name}={elapsed:.2f}")

    def record_output(self, file_count: int, video_rf_cols: int):
        """Record output metrics."""
        with self.lock:
            self.metrics['output_file_count'] = file_count
            self.metrics['video_rf_column_count'] = video_rf_cols
        self.logger.info(f"METRIC: output_file_count={file_count}")
        self.logger.info(f"METRIC: video_rf_column_count={video_rf_cols}")

    def finalize(self):
        """Finalize metrics and log summary."""
        elapsed = time.time() - self.start_time
        peak_memory = self.process.memory_info().rss / 1024 / 1024

        with self.lock:
            self.metrics['stage_4_duration_seconds'] = elapsed
            self.metrics['peak_memory_mb'] = peak_memory

        self.logger.info(f"METRIC: stage_4_duration_seconds={elapsed:.2f}")
        self.logger.info(f"METRIC: peak_memory_mb={peak_memory:.1f}")

        return self.metrics


# ============================================================================
# SECTION 4.1: VALIDATE INPUT
# ============================================================================

def validate_input(df: pd.DataFrame, bucket: str, expected_count: int) -> None:
    """
    Validate aggregated features CSV before transformation.

    Source: FeatureTransformationTI.md Section 4.1

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
        raise ValueError(
            f"Expected {expected_cols} columns for bucket {bucket}, found {len(df.columns)}"
        )

    # 2. Check all required columns exist
    required_cols = get_required_columns(bucket)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. "
            f"Expected 126 temporal columns (21 features × {len(required_cols)//21 - 3} windows) + 3 metadata."
        )

    # 3. Check for NaN values (fail-fast)
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        nan_count = {col: df[col].isna().sum() for col in nan_cols}
        raise ValueError(
            f"Invalid input: NaN values detected: {nan_count}. "
            f"Check Stage 3 aggregation logic."
        )

    # 4. Validate normalized features are in [0-1] range
    normalized_features = [
        'eye_contact_rate', 'speech_coverage', 'energy_level', 'energy_max',
        'pitch_scatter_ratio', 'emotion_consistency', 'average_face_size'
    ]
    for col in df.columns:
        if any(feat in col for feat in normalized_features):
            if (df[col] < 0).any() or (df[col] > 1).any():
                invalid_rows = df[(df[col] < 0) | (df[col] > 1)]
                raise ValueError(
                    f"Out of range: {col} has value {invalid_rows[col].max()}, "
                    f"expected [0.0-1.0]. Check Stage 2 calculation."
                )

    # 5. Validate count features are non-negative with sanity bounds
    count_features = [
        'scene_count', 'word_count', 'gesture_count', 'object_count',
        'person_count', 'overlay_unique_count'
    ]
    for col in df.columns:
        if any(feat in col for feat in count_features):
            if (df[col] < 0).any():
                raise ValueError(
                    f"Out of range: {col} has negative values. Check Stage 2 calculation."
                )
            if (df[col] > 10000).any():
                raise ValueError(
                    f"Out of range: {col} has suspiciously high values (>10000). "
                    f"Check Stage 2 calculation."
                )

    # 6. Check minimum row count
    from config.stage4_constants import MINIMUM_VIDEO_COUNT
    if len(df) < MINIMUM_VIDEO_COUNT:
        raise ValueError(
            f"Insufficient data: {len(df)} videos found, "
            f"minimum {MINIMUM_VIDEO_COUNT} required for ML training."
        )
    if len(df) < expected_count:
        logger.warning(
            f"Warning: Expected {expected_count} videos, found {len(df)}. "
            f"Proceeding with reduced sample size."
        )

    logger.info(f"Input validation passed: {len(df)} videos, {len(df.columns)} columns")


# ============================================================================
# SECTION 4.2: TRANSFORM VIDEO-LEVEL RF
# ============================================================================

def calculate_window_midpoint_timestamps(bucket: str, windows: list) -> list:
    """
    Calculate midpoint timestamps for each window in a bucket programmatically.

    Source: FeatureTransformationTI.md Section 4.2 (Lines 766-824)

    Args:
        bucket: str, e.g., "18-33s"
        windows: list, e.g., ['hook', 'middle_1', ..., 'closing']

    Returns:
        list of floats, midpoint timestamps in seconds

    Logic:
        - hook: midpoint of [0, 3] = 1.5s
        - middle segments: split remaining duration evenly, use midpoints
        - closing: midpoint of [duration-3, duration]
    """
    # Parse bucket duration range
    duration_range = bucket.replace('s', '').split('-')
    if len(duration_range) == 1:
        # Single value like "0-3s" parsed as ["0-3"]
        min_dur, max_dur = map(float, duration_range[0].split('-'))
    else:
        min_dur, max_dur = map(float, duration_range)

    # Use midpoint of duration range for calculation
    total_duration = (min_dur + max_dur) / 2

    timestamps = []

    for window in windows:
        if window == 'hook':
            timestamps.append(1.5)  # Midpoint of [0, 3]
        elif window == 'closing':
            timestamps.append(total_duration - 1.5)  # Midpoint of [duration-3, duration]
        elif window.startswith('middle'):
            # Middle segments split the duration between hook and closing
            middle_start = 3.0
            middle_end = total_duration - 3.0
            middle_duration = middle_end - middle_start

            # Count middle segments
            middle_count = len([w for w in windows if w.startswith('middle')])
            segment_duration = middle_duration / middle_count

            # Extract segment index (middle_1 -> 1, middle_2 -> 2, etc.)
            if window == 'middle_aggregate':
                # Special case: aggregated middle, use overall middle midpoint
                segment_idx = (middle_count + 1) / 2
            else:
                segment_idx = int(window.split('_')[1])

            # Calculate midpoint of this segment
            segment_start = middle_start + (segment_idx - 1) * segment_duration
            segment_midpoint = segment_start + (segment_duration / 2)
            timestamps.append(segment_midpoint)
        else:
            raise ValueError(f"Unknown window type: {window}")

    return timestamps


def calculate_linear_slope_with_timestamps(
    values: np.ndarray,
    windows: list,
    bucket: str
) -> float:
    """
    Calculate linear slope using actual window timestamps (not array indices).

    FIXED: Use programmatically calculated window midpoint timestamps instead of
    hardcoded WINDOW_TIMESTAMPS dict or array indices [0,1,2,3...].

    Source: FeatureTransformationTI.md Section 4.2 (Lines 827-854)

    Args:
        values: np.ndarray, feature values across windows
        windows: list, window names
        bucket: str, bucket identifier

    Returns:
        float, linear slope coefficient
    """
    if len(values) < 2:
        return 0.0

    # Validate inputs match
    if len(values) != len(windows):
        raise ValueError(
            f"Mismatch: {len(values)} values but {len(windows)} windows. "
            f"Cannot calculate slope."
        )

    # Calculate timestamps programmatically
    timestamps = calculate_window_midpoint_timestamps(bucket, windows)

    slope, _ = np.polyfit(timestamps, values, 1)
    return slope


def transform_video_level_rf(
    df: pd.DataFrame,
    bucket: str,
    strategy: str,
    video_count: int
) -> pd.DataFrame:
    """
    Transform aggregated features for Video-Level Random Forest.

    Source: FeatureTransformationTI.md Section 4.2 (Lines 654-763)

    Args:
        df: pandas DataFrame from aggregated_features.csv
        bucket: str, bucket identifier (e.g., "18-33s")
        strategy: str, "contrastive" or other (affects target variable)
        video_count: int, expected videos for target labeling

    Returns:
        pandas DataFrame with ~147 features for bucket 18-33s
    """
    df_rf = df.copy()

    # 1. Keep has_captions as Boolean (no encoding needed - RF handles Boolean natively)
    # has_captions already in 126 temporal features, preserved as-is

    # 2. One-hot encode hook_dominant_emotion_id as video-level emotion (Categorical 1-7 → 7 features)
    # FIXED: TI references dominant_emotion_id but input only has window-specific emotions
    # Use hook emotion as video-level emotion (first 3 seconds = most important)
    for emotion_id, emotion_name in enumerate(
        ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'],
        start=1
    ):
        df_rf[emotion_name] = (df_rf['hook_dominant_emotion_id'] == emotion_id).astype(int)
    df_rf.drop(columns=['hook_dominant_emotion_id'], inplace=True)

    # 3. Extract temporal features from create_time (ISO 8601 → 5 features)
    df_rf['create_time'] = pd.to_datetime(df_rf['create_time'])
    df_rf['hour'] = df_rf['create_time'].dt.hour  # 0-23
    df_rf['day_of_week'] = df_rf['create_time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_rf['month'] = df_rf['create_time'].dt.month  # 1-12
    df_rf['is_weekend'] = (df_rf['day_of_week'] >= 5).astype(int)  # 1 if Sat/Sun
    df_rf['is_business_hours'] = (
        (df_rf['hour'] >= 9) & (df_rf['hour'] <= 17)
    ).astype(int)  # 1 if 9am-5pm
    df_rf.drop(columns=['create_time'], inplace=True)

    # 4. Explicit 3-column gender encoding (String → 3 features, always)
    # FIXED: Always create all 3 columns to ensure consistent schema across buckets
    # Issue: pd.get_dummies with dummy_na=True creates variable columns (2 vs 3)
    if 'gender' in df_rf.columns:
        df_rf['gender_male'] = (df_rf['gender'] == 'male').astype(int)
        df_rf['gender_female'] = (df_rf['gender'] == 'female').astype(int)
        df_rf['gender_nan'] = df_rf['gender'].isna().astype(int)
        df_rf.drop(columns=['gender'], inplace=True)
    else:
        # If gender column missing entirely, create all 3 as zeros (graceful degradation)
        df_rf['gender_male'] = 0
        df_rf['gender_female'] = 0
        df_rf['gender_nan'] = 0

    # 5. Add target variable is_top_performer (contrastive strategy only)
    if strategy == 'contrastive':
        top_count = int(video_count * 0.8)
        df_rf['is_top_performer'] = (df_rf.index < top_count).astype(int)

    # 6. Compute Cross-Window Delta Features (NEW)
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket]

    # Check which windows exist
    has_closing = 'closing' in windows
    has_middle = any(w.startswith('middle') for w in windows)

    # Energy progression deltas (requires both middle and closing)
    if has_middle and has_closing:
        middle_windows = [w for w in windows if w.startswith('middle')]
        middle_energy_cols = [f'{w}_energy_level' for w in middle_windows]

        df_rf['hook_to_middle_energy_delta'] = (
            df_rf[middle_energy_cols].mean(axis=1) - df_rf['hook_energy_level']
        )
        df_rf['middle_to_closing_delta'] = (
            df_rf['closing_energy_level'] - df_rf[middle_energy_cols].mean(axis=1)
        )
    elif has_closing and not has_middle:
        # Bucket 3-9s: has hook + closing but no middle
        df_rf['hook_to_middle_energy_delta'] = 0.0  # No middle
        df_rf['middle_to_closing_delta'] = 0.0   # No middle
    else:
        # Bucket 0-3s: only hook, no middle or closing
        df_rf['hook_to_middle_energy_delta'] = 0.0
        df_rf['middle_to_closing_delta'] = 0.0

    # Consistency metrics (std deviation across all windows)
    eye_contact_cols = [f'{w}_eye_contact_rate' for w in windows]
    df_rf['eye_contact_consistency'] = df_rf[eye_contact_cols].std(axis=1)

    word_count_cols = [f'{w}_word_count' for w in windows]
    df_rf['word_density_std'] = df_rf[word_count_cols].std(axis=1)

    # Progression slopes (linear regression across windows)
    energy_cols = [f'{w}_energy_level' for w in windows]
    df_rf['energy_progression_slope'] = df_rf[energy_cols].apply(
        lambda row: calculate_linear_slope_with_timestamps(row.values, windows, bucket),
        axis=1
    )

    logger.info(
        f"Video-Level RF transformation complete: "
        f"{len(df_rf)} rows, {len(df_rf.columns)} columns"
    )
    return df_rf


# ============================================================================
# SECTION 4.3: TRANSFORM WINDOW-LEVEL RF
# ============================================================================

def transform_window_level_rf(
    df: pd.DataFrame,
    window_type: str,
    strategy: str,
    video_count: int
) -> pd.DataFrame:
    """
    Transform features for Window-Level Random Forest (single window).

    Source: FeatureTransformationTI.md Section 4.3 (Lines 857-924)

    Args:
        df: pandas DataFrame from aggregated_features.csv
        window_type: str, window identifier (e.g., "hook", "middle_1", "closing")
        strategy: str, "contrastive" or other (affects target variable)
        video_count: int, expected videos for target labeling

    Returns:
        pandas DataFrame with 22 columns (21 features + 1 target)
    """
    # Extract window-specific columns only
    window_prefix = f'{window_type}_'
    window_cols = [c for c in df.columns if c.startswith(window_prefix)]

    df_window = df[window_cols].copy()

    # Remove window prefix from column names (hook_scene_count → scene_count)
    df_window.columns = [c.replace(window_prefix, '') for c in df_window.columns]

    # Add target variable is_top_performer (contrastive strategy only)
    if strategy == 'contrastive':
        top_count = int(video_count * 0.8)
        df_window['is_top_performer'] = (df.index < top_count).astype(int)

    # NOTE: NO encoding transformations here
    # - has_captions stays Boolean (RF handles Boolean natively)
    # - dominant_emotion_id stays ordinal 1-7 (RF handles ordinal natively)
    # - emotional_valence stays continuous [-1,1] (RF handles continuous natively)

    logger.info(
        f"{window_type} RF: {len(df_window)} rows, {len(df_window.columns)} columns"
    )
    return df_window


# ============================================================================
# SECTION 4.4: TRANSFORM WINDOW-LEVEL K-MEANS
# ============================================================================

def transform_window_level_kmeans(
    df: pd.DataFrame,
    window_type: str
) -> pd.DataFrame:
    """
    Transform features for Window-Level K-Means (single window).

    Source: FeatureTransformationTI.md Section 4.4 (Lines 926-1032)

    Args:
        df: pandas DataFrame from aggregated_features.csv
        window_type: str, window identifier (e.g., "hook", "middle_1", "closing")

    Returns:
        pandas DataFrame with 27 columns (all numerical, scaled [0-1])
    """
    # Extract window-specific columns only
    window_prefix = f'{window_type}_'
    window_cols = [c for c in df.columns if c.startswith(window_prefix)]

    df_km = df[window_cols].copy()

    # Remove window prefix from column names
    df_km.columns = [c.replace(window_prefix, '') for c in df_km.columns]

    # 1. Log1p + MinMax Scale for count/variance features (11 features → 11 output columns)
    log_scale_features = [
        'scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count',
        'overlay_unique_count', 'shortest_scene', 'longest_scene',
        'scene_duration_variance', 'energy_variance', 'gaze_variance'
    ]
    for feature in log_scale_features:
        if feature in df_km.columns:
            # Apply log1p (log(1+x) to handle zeros)
            df_km[feature] = np.log1p(df_km[feature])

            # MinMax scale to [0,1]
            min_val = df_km[feature].min()
            max_val = df_km[feature].max()
            if max_val > min_val:
                df_km[f'{feature}_scaled'] = (df_km[feature] - min_val) / (max_val - min_val)
            else:
                # All values same (variance=0), set to midpoint
                df_km[f'{feature}_scaled'] = 0.5

            # Drop original column, keep only scaled version
            df_km.drop(columns=[feature], inplace=True)

    # 2. MinMax Scale only for normalized features (7 features → 7 output columns)
    scale_features = [
        'average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
        'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency'
    ]
    for feature in scale_features:
        if feature in df_km.columns:
            # MinMax scale to [0,1] (already in [0,1] range, but scale for consistency)
            min_val = df_km[feature].min()
            max_val = df_km[feature].max()
            if max_val > min_val:
                df_km[f'{feature}_scaled'] = (df_km[feature] - min_val) / (max_val - min_val)
            else:
                df_km[f'{feature}_scaled'] = 0.5

            df_km.drop(columns=[feature], inplace=True)

    # 3. Shift + Scale for emotional_valence (1 feature → 1 output column)
    if 'emotional_valence' in df_km.columns:
        # Shift [-1,1] → [0,1]: (x + 1) / 2
        df_km['emotional_valence_scaled'] = (df_km['emotional_valence'] + 1) / 2
        df_km.drop(columns=['emotional_valence'], inplace=True)

    # 4. Label Encode for has_captions (1 feature → 1 output column)
    if 'has_captions' in df_km.columns:
        df_km['has_captions_encoded'] = df_km['has_captions'].astype(int)
        df_km.drop(columns=['has_captions'], inplace=True)

    # 5. One-hot for dominant_emotion_id (1 feature → 7 output columns)
    if 'dominant_emotion_id' in df_km.columns:
        for emotion_id, emotion_name in enumerate(
            ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'],
            start=1
        ):
            df_km[emotion_name] = (df_km['dominant_emotion_id'] == emotion_id).astype(int)
        df_km.drop(columns=['dominant_emotion_id'], inplace=True)

    logger.info(
        f"Window-Level K-Means ({window_type}) transformation complete: "
        f"{len(df_km)} rows, {len(df_km.columns)} columns (expect 27)"
    )
    return df_km


# ============================================================================
# SECTION 4.5: VALIDATE OUTPUTS AND CHECKPOINT
# ============================================================================

def validate_cross_window_features(df_rf: pd.DataFrame, bucket: str) -> None:
    """
    Validate cross-window feature ranges in Video-Level RF output.

    Source: FeatureTransformationTI.md Section 4.5 (Lines 1143-1173)

    Args:
        df_rf: Video-Level RF DataFrame
        bucket: str, bucket identifier

    Raises:
        AssertionError: if validation fails
    """
    # Validate delta features (bounded by [-1, 1])
    if 'hook_to_middle_energy_delta' in df_rf.columns:
        assert df_rf['hook_to_middle_energy_delta'].between(-1, 1).all(), \
            f"hook_to_middle_energy_delta out of range [-1, 1]"

    if 'middle_to_closing_delta' in df_rf.columns:
        assert df_rf['middle_to_closing_delta'].between(-1, 1).all(), \
            f"middle_to_closing_delta out of range [-1, 1]"

    # Validate consistency features (std must be non-negative)
    if 'eye_contact_consistency' in df_rf.columns:
        assert (df_rf['eye_contact_consistency'] >= 0).all(), \
            f"eye_contact_consistency has negative values"

    if 'word_density_std' in df_rf.columns:
        assert (df_rf['word_density_std'] >= 0).all(), \
            f"word_density_std has negative values"

    # Validate slope feature (sanity check: shouldn't be extreme)
    if 'energy_progression_slope' in df_rf.columns:
        assert df_rf['energy_progression_slope'].between(-2, 2).all(), \
            f"energy_progression_slope suspiciously large"

    logger.info("✓ Cross-window feature validation passed")


def validate_outputs_and_checkpoint(
    output_files: dict,
    bucket: str,
    video_count: int,
    bucket_base: str
) -> None:
    """
    Validate all transformation outputs and write checkpoint.

    Source: FeatureTransformationTI.md Section 4.5 (Lines 1042-1141)

    Args:
        output_files: dict, {filename: DataFrame} mapping
        bucket: str, bucket name
        video_count: int, expected row count
        bucket_base: str, base directory for checkpoint

    Raises:
        AssertionError: if output validation fails
        IOError: if checkpoint write fails
    """
    # 1. Check all 13 files created
    expected_files = get_expected_output_files(bucket)
    missing_files = [f for f in expected_files if f not in output_files]
    if missing_files:
        raise AssertionError(f"Missing output files: {missing_files}")

    # 2. Validate Video-Level RF schema (bucket-specific)
    df_rf = output_files['rf_transformed.csv']
    expected_cols = get_expected_rf_column_count(bucket)
    tolerance = 3  # Allow ±3 for gender variations (missing entire column)
    assert expected_cols - tolerance <= len(df_rf.columns) <= expected_cols + tolerance, \
        f"Video-Level RF has {len(df_rf.columns)} columns, expected {expected_cols} ±{tolerance}"
    assert len(df_rf) == video_count, \
        f"Video-Level RF has {len(df_rf)} rows, expected {video_count}"
    assert not df_rf.isnull().any().any(), \
        "Video-Level RF contains NaN values"

    # Validate cross-window features
    validate_cross_window_features(df_rf, bucket)

    # 3. Validate Window-Level RF schemas (6 files)
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket]

    for window in windows:
        df_window_rf = output_files[f'{window}_rf_transformed.csv']
        assert len(df_window_rf.columns) == 22, \
            f"{window} RF has {len(df_window_rf.columns)} columns, expected 22"
        assert len(df_window_rf) == video_count, \
            f"{window} RF has {len(df_window_rf)} rows, expected {video_count}"
        assert not df_window_rf.isnull().any().any(), \
            f"{window} RF contains NaN values"

    # 4. Validate Window-Level K-Means schemas (6 files)
    for window in windows:
        df_window_km = output_files[f'{window}_km_transformed.csv']
        assert len(df_window_km.columns) == 27, \
            f"{window} K-Means has {len(df_window_km.columns)} columns, expected 27"
        assert len(df_window_km) == video_count, \
            f"{window} K-Means has {len(df_window_km)} rows, expected {video_count}"
        assert not df_window_km.isnull().any().any(), \
            f"{window} K-Means contains NaN values"

        # Validate all _scaled columns are in [0,1] range
        scaled_cols = [c for c in df_window_km.columns if c.endswith('_scaled')]
        for col in scaled_cols:
            assert df_window_km[col].between(0, 1).all(), \
                f"{window} K-Means column {col} has values outside [0,1]: " \
                f"{df_window_km[col].min()}-{df_window_km[col].max()}"

    # 5. Write checkpoint
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
        logger.error(f"Checkpoint write failed: {e}")
        logger.error("Stage 4 outputs may be valid, but orchestrator cannot verify completion.")
        raise IOError(f"Failed to write checkpoint: {e}") from e

    logger.info(
        f"Output validation passed and checkpoint written: "
        f"{len(output_files)} files, {video_count} videos"
    )


# ============================================================================
# MAIN ENTRY POINT FUNCTION
# ============================================================================

def run_stage4_transformation(
    bucket_path: str,
    config: dict
) -> Tuple[bool, List[str], float]:
    """
    Main entry point for Stage 4: Feature Transformation.

    Source: FeatureTransformationTI.md Appendix A (Lines 2160-2181)

    Args:
        bucket_path: str, full path to bucket directory
            (e.g., /data/clients/acme/hashtags/fitness/top_contrastive/bucket_18-33s)
        config: dict, configuration from config.json

    Returns:
        tuple:
            - success: bool (True if all transformations succeeded)
            - output_files: list[str] (list of 13 generated filenames)
            - elapsed_time: float (total execution time in seconds)

    Raises:
        ValueError: Input validation failed
        AssertionError: Output validation failed
        IOError: File I/O failed
        TimeoutError: Execution exceeded 300s
    """
    # Initialize metrics collector
    metrics = MetricsCollector(logger)
    metrics.start_stage()

    # Extract bucket identifier from path (e.g., "bucket_18-33s" → "18-33s")
    bucket = os.path.basename(bucket_path).replace('bucket_', '')

    logger.info(f"Starting Stage 4 transformation for {bucket}")

    # Load aggregated features CSV
    csv_path = os.path.join(bucket_path, 'ml_analysis', 'aggregated_features.csv')
    logger.info(f"Loading aggregated features from {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[EXIT 1] Aggregated CSV not found at {csv_path}. "
            f"Did Stage 3 complete successfully?"
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise IOError(
            f"[EXIT 2] Failed to parse CSV: {e}. Check file is valid CSV format."
        )

    logger.info(f"Loaded {len(df)} videos, {len(df.columns)} columns")

    # Get file size for metrics
    file_size_mb = os.path.getsize(csv_path) / 1024 / 1024
    metrics.record_input(len(df), len(df.columns), file_size_mb)

    # Extract parameters from config
    strategy = config.get('strategy', 'contrastive')
    video_count = config.get('video_count', len(df))

    # Step 1: Validate input
    logger.info("Validating input schema and data quality")
    validate_input(df, bucket, video_count)

    # Step 2: Transform Video-Level RF
    logger.info("Transforming features for Video-Level Random Forest")
    rf_start = time.time()
    df_rf = transform_video_level_rf(df, bucket, strategy, video_count)
    metrics.record_transformation_time('video_rf', time.time() - rf_start)

    # Step 3: Transform Window-Level RF
    logger.info("Transforming features for Window-Level Random Forest")
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket]

    wrf_start = time.time()
    window_rf_dfs = {}
    for window in windows:
        df_window_rf = transform_window_level_rf(df, window, strategy, video_count)
        window_rf_dfs[window] = df_window_rf
    metrics.record_transformation_time('window_rf', time.time() - wrf_start)
    logger.info(f"Window-Level RF complete: {len(windows)} files ({time.time() - wrf_start:.1f}s)")

    # Step 4: Transform Window-Level K-Means
    logger.info("Transforming features for Window-Level K-Means")
    km_start = time.time()
    window_km_dfs = {}
    for window in windows:
        df_window_km = transform_window_level_kmeans(df, window)
        window_km_dfs[window] = df_window_km
    metrics.record_transformation_time('window_km', time.time() - km_start)
    logger.info(f"Window-Level K-Means complete: {len(windows)} files ({time.time() - km_start:.1f}s)")

    # Step 5: Collect all output files into dict
    output_files = {'rf_transformed.csv': df_rf}
    for window in windows:
        output_files[f'{window}_rf_transformed.csv'] = window_rf_dfs[window]
        output_files[f'{window}_km_transformed.csv'] = window_km_dfs[window]

    # Step 6: Validate outputs and write checkpoint
    logger.info("Validating output schemas")
    validate_outputs_and_checkpoint(output_files, bucket, len(df), bucket_path)

    # Step 7: Write output files to disk
    logger.info("Writing output files to disk")
    output_dir = os.path.join(bucket_path, 'ml_analysis')
    os.makedirs(output_dir, exist_ok=True)

    io_start = time.time()
    for filename, df_output in output_files.items():
        output_path = os.path.join(output_dir, filename)
        df_output.to_csv(output_path, index=False)
        file_size_kb = os.path.getsize(output_path) / 1024
        logger.info(f"  Wrote {filename}: {file_size_kb:.1f} KB")
    metrics.record_transformation_time('file_io', time.time() - io_start)
    logger.info(f"File I/O complete: {len(output_files)} files ({time.time() - io_start:.1f}s)")

    # Step 8: Finalize metrics and log summary
    metrics.record_output(len(output_files), len(df_rf.columns))
    final_metrics = metrics.finalize()

    elapsed = final_metrics['stage_4_duration_seconds']
    logger.info(f"Stage 4 completed in {elapsed:.1f}s (target: <30s)")

    # Log performance breakdown
    logger.info(
        f"  Video-Level RF: {final_metrics.get('video_rf_duration_seconds', 0):.1f}s, "
        f"Window-Level RF: {final_metrics.get('window_rf_duration_seconds', 0):.1f}s, "
        f"Window-Level K-Means: {final_metrics.get('window_km_duration_seconds', 0):.1f}s, "
        f"I/O: {final_metrics.get('file_io_duration_seconds', 0):.1f}s"
    )

    # Performance warnings
    from config.stage4_constants import WARNING_TIME_MULTIPLIER, BASELINE_TIME_SECONDS
    warning_threshold = BASELINE_TIME_SECONDS * WARNING_TIME_MULTIPLIER
    if elapsed > warning_threshold:
        logger.warning(
            f"Stage 4 exceeded warning threshold: {elapsed:.1f}s > {warning_threshold:.1f}s"
        )

    return True, list(output_files.keys()), elapsed
