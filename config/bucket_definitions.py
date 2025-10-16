"""
Bucket-specific window configurations for RumiAI ML Pipeline.

SINGLE SOURCE OF TRUTH for temporal window structure across all stages.

Used by:
- Stage 3: Feature Aggregation (aggregates windows into features)
- Stage 4: Feature Transformation (creates window-specific transformed CSVs)
- Stage 5: ML Model Training (trains models per window)
- Stage 6: ML Analysis Generation (generates JSONs per window)
- Stage 7: LLM Analysis (analyzes windows)

DO NOT DUPLICATE: All stages should import from this file.

Last Updated: 2025-01-28
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
