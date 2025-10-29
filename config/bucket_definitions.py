"""
Bucket-specific window configurations for RumiAI ML Pipeline.

SINGLE SOURCE OF TRUTH for temporal window structure across all stages.

Used by:
- Stage 3: Feature Aggregation (aggregates windows into features)
- Stage 4: Feature Transformation (creates window-specific transformed CSVs + scalers)
- Stage 5: ML Model Training (trains models per window)
- Stage 6: ML Analysis Generation (generates JSONs per window)
- Stage 7: LLM Analysis (analyzes windows)
- Documentation: Reference get_stage4_output_count() instead of hardcoding file counts

DO NOT DUPLICATE: All stages should import from this file.

Last Updated: 2025-10-23 (Added get_stage4_output_count for scaler tracking)
"""

# Bucket-specific window configurations
# Each bucket has a different temporal window structure based on video duration
BUCKET_WINDOWS = {
    '0-3s': ['hook'],  # Only hook (no closing - video too short)
    '3-9s': ['hook', 'closing'],
    '9-13s': ['hook', 'middle_aggregate', 'closing'],  # Aggregated middle (not middle_1/2/3)
    '13-18s': ['hook', 'middle_aggregate', 'closing'],  # Aggregated middle (not middle_1/2/3)
    '18-33s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing'],
    '33-60s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
    '60-90s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
    '90-120s': ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'closing'],
}


def get_window_count(bucket: str) -> int:
    """
    Get number of windows for a bucket.

    Args:
        bucket: Duration bucket (e.g., '18-33s')

    Returns:
        Number of windows

    Example:
        >>> get_window_count('18-33s')
        6
        >>> get_window_count('0-3s')
        1
    """
    if bucket not in BUCKET_WINDOWS:
        raise ValueError(f"Unknown bucket: {bucket}. Valid buckets: {list(BUCKET_WINDOWS.keys())}")
    return len(BUCKET_WINDOWS[bucket])


def get_windows(bucket: str) -> list:
    """
    Get window list for a bucket.

    Args:
        bucket: Duration bucket (e.g., '18-33s')

    Returns:
        List of window names

    Example:
        >>> get_windows('18-33s')
        ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']
        >>> get_windows('9-13s')
        ['hook', 'middle_aggregate', 'closing']
    """
    if bucket not in BUCKET_WINDOWS:
        raise ValueError(f"Unknown bucket: {bucket}. Valid buckets: {list(BUCKET_WINDOWS.keys())}")
    return BUCKET_WINDOWS[bucket]


def get_all_buckets() -> list:
    """
    Get list of all valid bucket names.

    Returns:
        List of bucket names

    Example:
        >>> get_all_buckets()
        ['0-3s', '3-9s', '9-13s', '13-18s', '18-33s', '33-60s', '60-90s', '90-120s']
    """
    return list(BUCKET_WINDOWS.keys())


def get_stage4_output_count(bucket: str) -> dict:
    """
    Get expected Stage 4 output file counts for a bucket.

    Stage 4 (Feature Transformation) produces:
    - 1 Video-Level RF CSV (rf_transformed.csv)
    - N Window-Level RF CSVs ({window}_rf_transformed.csv)
    - N Window-Level K-Means CSVs ({window}_km_transformed.csv)
    - N Scaler PKL files ({window}_scalers.pkl)

    Total = 1 + 3N, where N = number of windows

    Args:
        bucket: Duration bucket (e.g., '18-33s')

    Returns:
        Dictionary with file counts:
        {
            'video_rf_csv': 1,
            'window_rf_csv': N,
            'window_km_csv': N,
            'scaler_pkl': N,
            'total_csv': 1 + 2N,
            'total_files': 1 + 3N
        }

    Example:
        >>> get_stage4_output_count('18-33s')
        {'video_rf_csv': 1, 'window_rf_csv': 6, 'window_km_csv': 6,
         'scaler_pkl': 6, 'total_csv': 13, 'total_files': 19}

        >>> get_stage4_output_count('13-18s')
        {'video_rf_csv': 1, 'window_rf_csv': 3, 'window_km_csv': 3,
         'scaler_pkl': 3, 'total_csv': 7, 'total_files': 10}
    """
    if bucket not in BUCKET_WINDOWS:
        raise ValueError(f"Unknown bucket: {bucket}. Valid buckets: {list(BUCKET_WINDOWS.keys())}")

    window_count = len(BUCKET_WINDOWS[bucket])

    return {
        'video_rf_csv': 1,
        'window_rf_csv': window_count,
        'window_km_csv': window_count,
        'scaler_pkl': window_count,
        'total_csv': 1 + (2 * window_count),
        'total_files': 1 + (3 * window_count)
    }


def get_stage3_expected_feature_count(bucket: str) -> int:
    """
    Get expected Stage 3 output column count for a bucket (SINGLE SOURCE OF TRUTH).

    Stage 3 aggregation produces:
    - 21 base features per temporal window (hook, middle segments, closing)
    - 3 metadata columns (video_id, create_time, gender)
    - 0-5 cross-window features (bucket-dependent, see below)
    - 1 is_top_performer label

    Cross-window features by bucket (bucket-aware Option 2):
    - 0-3s: 0 features (only hook exists, no comparisons possible)
    - 3-9s: 3 features (eye_contact_consistency, word_density_std, energy_progression_slope)
    - 9-13s+: 5 features (all 5: deltas + consistency + std + slope)

    Formula: (21 × window_count) + 3 metadata + X cross-window + 1 label

    Args:
        bucket: Duration bucket (e.g., '18-33s')

    Returns:
        Expected column count for aggregated_features.csv

    Example:
        >>> get_stage3_expected_feature_count('18-33s')
        135  # 21×6 + 3 + 5 + 1
        >>> get_stage3_expected_feature_count('9-13s')
        72   # 21×3 + 3 + 5 + 1
        >>> get_stage3_expected_feature_count('3-9s')
        49   # 21×2 + 3 + 3 + 1
        >>> get_stage3_expected_feature_count('0-3s')
        25   # 21×1 + 3 + 0 + 1

    Source: S7B2.md Root Cause Fix (lines 311-419, 553-575)
    Last Updated: 2025-10-28 (S7B2 fix - moved cross-window + is_top_performer to Stage 3)
    """
    if bucket not in BUCKET_WINDOWS:
        raise ValueError(f"Unknown bucket: {bucket}. Valid buckets: {list(BUCKET_WINDOWS.keys())}")

    # Base temporal features: 21 per window
    window_count = len(BUCKET_WINDOWS[bucket])
    base_features = 21 * window_count

    # Metadata columns: video_id, create_time, gender
    metadata_cols = 3

    # Cross-window features (bucket-aware)
    # - 0-3s: 0 (only hook, can't compute cross-window stats)
    # - 3-9s: 3 (consistency + std + slope, no deltas)
    # - 9-13s+: 5 (all 5 features)
    if bucket == '0-3s':
        cross_window_features = 0
    elif bucket == '3-9s':
        cross_window_features = 3
    else:
        # All buckets from 9-13s onward have all 5 cross-window features
        cross_window_features = 5

    # Label: is_top_performer (1 column)
    label_cols = 1

    return base_features + metadata_cols + cross_window_features + label_cols
