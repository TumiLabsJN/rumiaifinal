"""
Schema validation for Stage 2: Video Processing

Validates temporal_windows_updated.json output from RumiAI.

Source: VideoProcessingTI.md Section 2.3.3 (validate_temporal_windows_schema)
"""

import logging
from typing import Dict, Any

from ml_pipeline.stage2_processing.exceptions import ValidationError

logger = logging.getLogger(__name__)


def validate_temporal_windows_schema(insights: Dict[str, Any]):
    """
    Validate temporal_windows_updated.json structure and completeness.

    Requirements spec for TI implementation. Validates that RumiAI output
    matches expected schema without duplicating feature definitions from
    SystemArchitecturev2.md (lines 395-460).

    Args:
        insights: dict, loaded temporal_windows JSON

    Raises:
        ValidationError: if schema invalid

    Source: VideoProcessingTI.md Section 2.3.3
    """
    video_id = insights.get('video_id', 'unknown')

    # 1. Check required top-level keys
    required_keys = ['temporal_windows', 'metadata', 'processing_timestamp']
    missing = [k for k in required_keys if k not in insights]
    if missing:
        raise ValidationError(video_id, 'top_level_keys', str(required_keys), f"missing: {missing}")

    # 2. Validate temporal_windows structure
    windows = insights['temporal_windows']
    if not isinstance(windows, dict):
        raise ValidationError(video_id, 'temporal_windows', 'dict', type(windows).__name__)

    # 3. Check required window sections exist and are dicts
    required_sections = ['hook', 'closing']
    for section in required_sections:
        if section not in windows:
            raise ValidationError(video_id, f'temporal_windows.{section}', 'present', 'missing')
        if not isinstance(windows[section], dict):
            raise ValidationError(video_id, f'temporal_windows.{section}', 'dict', type(windows[section]).__name__)

        # 4. Check feature count (expect 60+ features per window)
        if len(windows[section]) < 50:
            logger.warning(f"Window section '{section}' has only {len(windows[section])} features (expected 60+)")

    # 5. Validate middle_segments logic (null for short videos <9s, list otherwise)
    video_duration = insights.get('duration', 0)
    middle_segments = windows.get('middle_segments')

    if video_duration < 9:
        # Short videos: middle_segments should be null
        if middle_segments is not None:
            raise ValidationError(video_id, 'middle_segments', 'null', f'not null (duration={video_duration}s)')
    else:
        # Longer videos: middle_segments should be list of dicts
        if not isinstance(middle_segments, list):
            raise ValidationError(video_id, 'middle_segments', 'list', type(middle_segments).__name__)

        # Each middle segment should be a dict with features
        for i, segment in enumerate(middle_segments):
            if not isinstance(segment, dict):
                raise ValidationError(video_id, f'middle_segments[{i}]', 'dict', type(segment).__name__)
            if len(segment) < 50:
                logger.warning(f"Middle segment {i} has only {len(segment)} features (expected 60+)")

    # 6. Validate metadata structure
    metadata = insights['metadata']
    if not isinstance(metadata, dict):
        raise ValidationError(video_id, 'metadata', 'dict', type(metadata).__name__)

    # 7. Validate timestamp format
    timestamp = insights['processing_timestamp']
    if not isinstance(timestamp, (str, int, float)):
        raise ValidationError(video_id, 'processing_timestamp', 'str/int/float', type(timestamp).__name__)

    logger.debug(f"Schema validation passed for video {video_id}")
