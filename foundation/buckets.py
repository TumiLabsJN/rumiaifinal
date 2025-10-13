"""
Bucket assignment logic for video duration buckets.

Source: FoundationCHILD.md Section 6.1: Bucket Assignment Logic
"""

from typing import Tuple
from foundation.constants import BUCKET_DEFINITIONS


def assign_bucket(duration: float) -> str:
    """
    Assign video to duration bucket based on video length.

    Boundary Behavior (Section 6.1: Bucket Assignment Logic):
    - All buckets use inclusive lower bound: duration >= lower_bound
    - All buckets use exclusive upper bound: duration < upper_bound
    - Exception: Final bucket "90-120s" uses inclusive upper bound: duration <= 120

    Args:
        duration: Video duration in seconds

    Returns:
        Bucket name (e.g., "18-33s")

    Raises:
        ValueError: If duration > 120s (exceeds TikTok maximum)

    Examples:
        >>> assign_bucket(2.5)
        '0-3s'
        >>> assign_bucket(9.0)
        '9-13s'
        >>> assign_bucket(120.0)
        '90-120s'
        >>> assign_bucket(125.0)
        ValueError: Video duration 125.0s exceeds TikTok maximum (120s)
    """
    if duration < 3:
        return "0-3s"
    elif duration < 9:
        return "3-9s"
    elif duration < 13:
        return "9-13s"
    elif duration < 18:
        return "13-18s"
    elif duration < 33:
        return "18-33s"
    elif duration < 60:
        return "33-60s"
    elif duration < 90:
        return "60-90s"
    elif duration <= 120:  # Note: inclusive upper bound for final bucket
        return "90-120s"
    else:
        raise ValueError(f"Video duration {duration}s exceeds TikTok maximum (120s)")


def get_bucket_bounds(bucket: str) -> Tuple[float, float]:
    """
    Get duration bounds for a bucket.

    Args:
        bucket: Bucket name (e.g., "18-33s")

    Returns:
        Tuple of (lower_bound, upper_bound)

    Raises:
        KeyError: If bucket name is invalid

    Example:
        >>> get_bucket_bounds("18-33s")
        (18, 33)
    """
    if bucket not in BUCKET_DEFINITIONS:
        raise KeyError(f"Invalid bucket name: {bucket}")
    return BUCKET_DEFINITIONS[bucket]
