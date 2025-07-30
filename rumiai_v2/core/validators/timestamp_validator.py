"""
Timestamp validation utilities.
"""
from typing import Any, Optional, List, Union
from ..models.timestamp import Timestamp
from ..exceptions import ValidationError
import logging

logger = logging.getLogger(__name__)


class TimestampValidator:
    """Validate and normalize timestamp values."""
    
    @staticmethod
    def validate_timestamp(value: Any, field_name: str = "timestamp") -> Optional[Timestamp]:
        """
        Validate and convert a timestamp value.
        
        Returns None if invalid (doesn't raise exception).
        """
        ts = Timestamp.from_value(value)
        if ts is None:
            logger.warning(f"Invalid {field_name}: {repr(value)}")
        return ts
    
    @staticmethod
    def validate_timestamp_range(
        start: Any,
        end: Any,
        max_duration: Optional[float] = None
    ) -> tuple[Optional[Timestamp], Optional[Timestamp]]:
        """
        Validate a timestamp range.
        
        Returns (start_ts, end_ts) or (None, None) if invalid.
        """
        start_ts = Timestamp.from_value(start)
        end_ts = Timestamp.from_value(end)
        
        if start_ts and end_ts:
            # Validate range
            if start_ts.seconds > end_ts.seconds:
                logger.warning(f"Invalid range: start {start_ts} > end {end_ts}")
                return None, None
            
            # Check duration limit
            if max_duration and (end_ts.seconds - start_ts.seconds) > max_duration:
                logger.warning(
                    f"Range duration {end_ts.seconds - start_ts.seconds} exceeds max {max_duration}"
                )
                # Clamp to max duration
                end_ts = Timestamp(start_ts.seconds + max_duration)
        
        return start_ts, end_ts
    
    @staticmethod
    def normalize_timestamp_list(
        timestamps: List[Any],
        video_duration: Optional[float] = None
    ) -> List[Timestamp]:
        """
        Normalize a list of timestamps.
        
        Filters out invalid values and clamps to video duration if provided.
        """
        result = []
        
        for value in timestamps:
            ts = Timestamp.from_value(value)
            if ts:
                # Clamp to video duration if provided
                if video_duration and ts.seconds > video_duration:
                    ts = Timestamp(video_duration)
                result.append(ts)
        
        # Sort by time
        result.sort(key=lambda t: t.seconds)
        
        return result
    
    @staticmethod
    def is_valid_timestamp_format(value: str) -> bool:
        """Check if a string is in a valid timestamp format."""
        ts = Timestamp.from_value(value)
        return ts is not None