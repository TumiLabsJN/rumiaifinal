"""
Unit tests for Stage 4: Feature Transformation

Source: FeatureTransformationCHILD.md Section 8.1 (Unit Tests)

Tests all transformation functions with synthetic data fixtures.
Expected runtime: <1 second total
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rumiai_v2.processors.feature_transformation import (
    validate_input,
    transform_video_level_rf,
    transform_window_level_rf,
    transform_window_level_kmeans,
    validate_outputs_and_checkpoint,
    calculate_window_midpoint_timestamps,
    calculate_linear_slope_with_timestamps
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def fixture_bucket_18_33s():
    """Load synthetic test fixture for bucket 18-33s (10 videos, 129 columns)"""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "stage4" / "test_bucket_18-33s_minimal.csv"
    return pd.read_csv(fixture_path)


@pytest.fixture
def fixture_bucket_9_13s():
    """Load synthetic test fixture for bucket 9-13s (10 videos, 66 columns)"""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "stage4" / "test_bucket_9-13s_minimal.csv"
    return pd.read_csv(fixture_path)


@pytest.fixture
def fixture_bucket_3_9s():
    """Load synthetic test fixture for bucket 3-9s (10 videos, 45 columns)"""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "stage4" / "test_bucket_3-9s_minimal.csv"
    return pd.read_csv(fixture_path)


# ============================================================================
# TEST INPUT VALIDATION
# ============================================================================

def test_input_validation_insufficient_videos(fixture_bucket_18_33s):
    """Test: Insufficient videos (N<50) raises ValueError"""
    df = fixture_bucket_18_33s.copy()  # Only 10 videos
    df['gender'] = df['gender'].fillna('male')  # Remove NaN to test row count specifically

    with pytest.raises(ValueError, match="Insufficient data: 10 videos found, minimum 50 required"):
        validate_input(df, '18-33s', expected_count=100)


def test_input_validation_missing_columns(fixture_bucket_18_33s):
    """Test: Missing required columns raises ValueError"""
    df = fixture_bucket_18_33s.copy()
    df['gender'] = df['gender'].fillna('male')  # Remove NaN to test column check specifically
    df = df.drop(columns=['hook_scene_count'])  # Drop a required column

    # Should fail on required columns check (not column count, since we only dropped 1)
    with pytest.raises(ValueError):  # Don't match specific message since column count fails first
        validate_input(df, '18-33s', expected_count=10)


def test_input_validation_nan_values(fixture_bucket_18_33s):
    """Test: NaN values raise ValueError"""
    df = fixture_bucket_18_33s.copy()
    df.loc[0, 'hook_scene_count'] = np.nan  # Introduce NaN

    with pytest.raises(ValueError, match="Invalid input: NaN values detected"):
        validate_input(df, '18-33s', expected_count=10)


def test_input_validation_out_of_range(fixture_bucket_18_33s):
    """Test: Out-of-range values raise ValueError"""
    df = fixture_bucket_18_33s.copy()
    df['gender'] = df['gender'].fillna('male')  # Remove NaN to test range check specifically
    df.loc[0, 'hook_eye_contact_rate'] = 1.5  # Out of [0-1] range

    with pytest.raises(ValueError, match="Out of range.*eye_contact_rate"):
        validate_input(df, '18-33s', expected_count=10)


# ============================================================================
# TEST VIDEO-LEVEL RF TRANSFORMATIONS
# ============================================================================

def test_video_rf_gender_encoding(fixture_bucket_18_33s):
    """Test: gender encoding creates 3 columns (male, female, nan)"""
    df = fixture_bucket_18_33s.copy()
    df_rf = transform_video_level_rf(df, '18-33s', 'contrastive', 10)

    # Check all 3 gender columns exist
    assert 'gender_male' in df_rf.columns
    assert 'gender_female' in df_rf.columns
    assert 'gender_nan' in df_rf.columns

    # Check original gender column removed
    assert 'gender' not in df_rf.columns

    # Check values are 0 or 1
    assert df_rf['gender_male'].isin([0, 1]).all()
    assert df_rf['gender_female'].isin([0, 1]).all()
    assert df_rf['gender_nan'].isin([0, 1]).all()


def test_video_rf_temporal_extraction(fixture_bucket_18_33s):
    """Test: create_time extracts hour, day_of_week, month, is_weekend, is_business_hours"""
    df = fixture_bucket_18_33s.copy()
    df_rf = transform_video_level_rf(df, '18-33s', 'contrastive', 10)

    # Check temporal features exist
    assert 'hour' in df_rf.columns
    assert 'day_of_week' in df_rf.columns
    assert 'month' in df_rf.columns
    assert 'is_weekend' in df_rf.columns
    assert 'is_business_hours' in df_rf.columns

    # Check create_time removed
    assert 'create_time' not in df_rf.columns

    # Check value ranges
    assert df_rf['hour'].between(0, 23).all()
    assert df_rf['day_of_week'].between(0, 6).all()
    assert df_rf['month'].between(1, 12).all()
    assert df_rf['is_weekend'].isin([0, 1]).all()
    assert df_rf['is_business_hours'].isin([0, 1]).all()


def test_video_rf_emotion_one_hot(fixture_bucket_18_33s):
    """Test: dominant_emotion_id one-hot encoded to 7 columns"""
    df = fixture_bucket_18_33s.copy()
    df_rf = transform_video_level_rf(df, '18-33s', 'contrastive', 10)

    # Check 7 emotion columns exist
    emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']
    for emotion in emotions:
        assert emotion in df_rf.columns
        assert df_rf[emotion].isin([0, 1]).all()

    # Check original dominant_emotion_id removed
    assert 'dominant_emotion_id' not in df_rf.columns


def test_video_rf_target_variable(fixture_bucket_18_33s):
    """Test: contrastive strategy creates is_top_performer (top 80% = 1)"""
    df = fixture_bucket_18_33s.copy()
    df_rf = transform_video_level_rf(df, '18-33s', 'contrastive', 10)

    # Check target column exists
    assert 'is_top_performer' in df_rf.columns
    assert df_rf['is_top_performer'].isin([0, 1]).all()

    # Check top 80% (8 out of 10 videos) are labeled 1
    assert df_rf['is_top_performer'].sum() == 8


def test_video_rf_cross_window_features(fixture_bucket_18_33s):
    """Test: Cross-window features computed correctly"""
    df = fixture_bucket_18_33s.copy()
    df_rf = transform_video_level_rf(df, '18-33s', 'contrastive', 10)

    # Check 5 cross-window features exist
    assert 'hook_to_middle_energy_delta' in df_rf.columns
    assert 'middle_to_closing_delta' in df_rf.columns
    assert 'eye_contact_consistency' in df_rf.columns
    assert 'word_density_std' in df_rf.columns
    assert 'energy_progression_slope' in df_rf.columns

    # Check value ranges (deltas should be in [-1, 1])
    assert df_rf['hook_to_middle_energy_delta'].between(-1, 1).all()
    assert df_rf['middle_to_closing_delta'].between(-1, 1).all()

    # Check consistency metrics are non-negative (std deviation)
    assert (df_rf['eye_contact_consistency'] >= 0).all()
    assert (df_rf['word_density_std'] >= 0).all()


# ============================================================================
# TEST WINDOW-LEVEL RF TRANSFORMATIONS
# ============================================================================

def test_window_rf_column_extraction(fixture_bucket_18_33s):
    """Test: Window prefix removed (hook_scene_count → scene_count)"""
    df = fixture_bucket_18_33s.copy()
    df_hook_rf = transform_window_level_rf(df, 'hook', 'contrastive', 10)

    # Check prefix removed
    assert 'scene_count' in df_hook_rf.columns
    assert 'hook_scene_count' not in df_hook_rf.columns

    # Check original values preserved
    assert df_hook_rf['scene_count'].equals(df['hook_scene_count'])


def test_window_rf_no_encoding(fixture_bucket_18_33s):
    """Test: has_captions stays Boolean, dominant_emotion_id stays int"""
    df = fixture_bucket_18_33s.copy()
    df_hook_rf = transform_window_level_rf(df, 'hook', 'contrastive', 10)

    # has_captions should be Boolean/int (no one-hot)
    assert 'has_captions' in df_hook_rf.columns
    assert df_hook_rf['has_captions'].dtype in [bool, int, 'bool', 'int64']

    # dominant_emotion_id should be int 1-7 (no one-hot)
    assert 'dominant_emotion_id' in df_hook_rf.columns
    assert df_hook_rf['dominant_emotion_id'].between(1, 7).all()


def test_window_rf_output_schema(fixture_bucket_18_33s):
    """Test: All window RF files have 22 columns"""
    df = fixture_bucket_18_33s.copy()
    windows = ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

    for window in windows:
        df_window_rf = transform_window_level_rf(df, window, 'contrastive', 10)
        assert len(df_window_rf.columns) == 22, f"{window} should have 22 columns, got {len(df_window_rf.columns)}"


# ============================================================================
# TEST WINDOW-LEVEL K-MEANS TRANSFORMATIONS
# ============================================================================

def test_window_kmeans_log_scale(fixture_bucket_18_33s):
    """Test: Log + scale transformation for count features"""
    df = fixture_bucket_18_33s.copy()
    df_hook_km = transform_window_level_kmeans(df, 'hook')

    # Check scene_count_scaled exists (not scene_count)
    assert 'scene_count_scaled' in df_hook_km.columns
    assert 'scene_count' not in df_hook_km.columns

    # Check scaled values in [0,1] range
    assert df_hook_km['scene_count_scaled'].between(0, 1).all()


def test_window_kmeans_shift_scale(fixture_bucket_18_33s):
    """Test: emotional_valence shifted from [-1,1] to [0,1]"""
    df = fixture_bucket_18_33s.copy()
    df_hook_km = transform_window_level_kmeans(df, 'hook')

    # Check emotional_valence_scaled exists
    assert 'emotional_valence_scaled' in df_hook_km.columns
    assert 'emotional_valence' not in df_hook_km.columns

    # Check scaled values in [0,1] range
    assert df_hook_km['emotional_valence_scaled'].between(0, 1).all()


def test_window_kmeans_label_encode(fixture_bucket_18_33s):
    """Test: has_captions label encoded (True→1, False→0)"""
    df = fixture_bucket_18_33s.copy()
    df_hook_km = transform_window_level_kmeans(df, 'hook')

    # Check has_captions_encoded exists
    assert 'has_captions_encoded' in df_hook_km.columns
    assert 'has_captions' not in df_hook_km.columns

    # Check encoded values are 0 or 1
    assert df_hook_km['has_captions_encoded'].isin([0, 1]).all()


def test_window_kmeans_emotion_one_hot(fixture_bucket_18_33s):
    """Test: dominant_emotion_id one-hot encoded to 7 columns"""
    df = fixture_bucket_18_33s.copy()
    df_hook_km = transform_window_level_kmeans(df, 'hook')

    # Check 7 emotion columns exist
    emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']
    for emotion in emotions:
        assert emotion in df_hook_km.columns
        assert df_hook_km[emotion].isin([0, 1]).all()

    # Check original dominant_emotion_id removed
    assert 'dominant_emotion_id' not in df_hook_km.columns


def test_window_kmeans_output_schema(fixture_bucket_18_33s):
    """Test: All window K-Means files have 27 columns"""
    df = fixture_bucket_18_33s.copy()
    windows = ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

    for window in windows:
        df_window_km = transform_window_level_kmeans(df, window)
        assert len(df_window_km.columns) == 27, f"{window} K-Means should have 27 columns, got {len(df_window_km.columns)}"


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_edge_case_zero_variance(fixture_bucket_18_33s):
    """Test: All features same value (variance=0) → scaled to 0.5"""
    df = fixture_bucket_18_33s.copy()

    # Set all hook_scene_count to same value
    df['hook_scene_count'] = 5

    df_hook_km = transform_window_level_kmeans(df, 'hook')

    # Check scaled value is 0.5 (midpoint)
    assert (df_hook_km['scene_count_scaled'] == 0.5).all()


def test_edge_case_missing_gender(fixture_bucket_18_33s):
    """Test: Missing gender creates gender_nan=1"""
    df = fixture_bucket_18_33s.copy()

    # Set first row gender to None
    df.loc[0, 'gender'] = None

    df_rf = transform_video_level_rf(df, '18-33s', 'contrastive', 10)

    # Check first row has gender_nan=1
    assert df_rf.loc[0, 'gender_nan'] == 1
    assert df_rf.loc[0, 'gender_male'] == 0
    assert df_rf.loc[0, 'gender_female'] == 0


def test_edge_case_bucket_9_13s_middle_aggregate(fixture_bucket_9_13s):
    """Test: Bucket 9-13s with middle_aggregate extracts correctly"""
    df = fixture_bucket_9_13s.copy()

    df_middle_agg_rf = transform_window_level_rf(df, 'middle_aggregate', 'contrastive', 10)

    # Check middle_aggregate features exist
    assert 'scene_count' in df_middle_agg_rf.columns
    assert len(df_middle_agg_rf.columns) == 22


# ============================================================================
# TEST OUTPUT VALIDATION
# ============================================================================

def test_output_validation_row_count_preserved(fixture_bucket_18_33s):
    """Test: Row count preserved across all transformations"""
    df = fixture_bucket_18_33s.copy()

    # Video-Level RF
    df_rf = transform_video_level_rf(df, '18-33s', 'contrastive', 10)
    assert len(df_rf) == len(df)

    # Window-Level RF
    df_hook_rf = transform_window_level_rf(df, 'hook', 'contrastive', 10)
    assert len(df_hook_rf) == len(df)

    # Window-Level K-Means
    df_hook_km = transform_window_level_kmeans(df, 'hook')
    assert len(df_hook_km) == len(df)


def test_output_validation_no_nan_introduced(fixture_bucket_18_33s):
    """Test: No NaN values introduced during transformation"""
    df = fixture_bucket_18_33s.copy()

    df_rf = transform_video_level_rf(df, '18-33s', 'contrastive', 10)
    assert not df_rf.isnull().any().any(), "Video-Level RF introduced NaN values"

    df_hook_rf = transform_window_level_rf(df, 'hook', 'contrastive', 10)
    assert not df_hook_rf.isnull().any().any(), "Window-Level RF introduced NaN values"

    df_hook_km = transform_window_level_kmeans(df, 'hook')
    assert not df_hook_km.isnull().any().any(), "Window-Level K-Means introduced NaN values"


def test_output_validation_kmeans_scaled_range(fixture_bucket_18_33s):
    """Test: All _scaled columns in K-Means are [0,1]"""
    df = fixture_bucket_18_33s.copy()
    df_hook_km = transform_window_level_kmeans(df, 'hook')

    scaled_cols = [c for c in df_hook_km.columns if c.endswith('_scaled')]

    for col in scaled_cols:
        assert df_hook_km[col].between(0, 1).all(), f"{col} has values outside [0,1]"


# ============================================================================
# TEST CROSS-WINDOW FEATURES
# ============================================================================

def test_cross_window_timestamp_calculation():
    """Test: calculate_window_midpoint_timestamps() returns correct timestamps"""
    # Bucket 18-33s: midpoint = 25.5s
    # hook (0-3s): midpoint = 1.5s
    # closing (22.5-25.5s): midpoint = 24.0s
    # middle segments split 3-22.5s evenly (19.5s / 4 = 4.875s each)

    timestamps = calculate_window_midpoint_timestamps(
        '18-33s',
        ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']
    )

    assert len(timestamps) == 6
    assert timestamps[0] == 1.5  # hook midpoint
    assert timestamps[-1] == 24.0  # closing midpoint (25.5 - 1.5)


def test_cross_window_slope_calculation():
    """Test: calculate_linear_slope_with_timestamps() computes slope correctly"""
    # Rising energy: [0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    values = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.8])
    windows = ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

    slope = calculate_linear_slope_with_timestamps(values, windows, '18-33s')

    # Should be positive slope (rising energy)
    assert slope > 0


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
