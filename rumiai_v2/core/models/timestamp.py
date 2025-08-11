"""
Timestamp model for RumiAI v2.

This module provides a unified timestamp implementation that handles all the
various formats found in the current system.
"""
from dataclasses import dataclass
from typing import Union, Optional
import re
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Timestamp:
    """
    Immutable timestamp in seconds.
    
    This is the ONLY timestamp implementation in the system.
    All timestamps are internally stored as float seconds.
    """
    seconds: float
    
    def __post_init__(self):
        if self.seconds < 0:
            # Don't raise - log and clamp to 0
            logger.warning(f"Negative timestamp {self.seconds} clamped to 0")
            object.__setattr__(self, 'seconds', 0.0)
    
    @classmethod
    def from_value(cls, value: Union[str, int, float, None]) -> Optional['Timestamp']:
        """
        Parse any timestamp format to Timestamp object.
        
        CRITICAL: This function is called thousands of times per video.
        Any exception here cascades to complete pipeline failure.
        
        Supported formats:
        - None -> None
        - int/float -> Timestamp(float(value))
        - "0-1s" -> Timestamp(0.0) (uses start of range)
        - "0s" or "0.5s" -> Timestamp(float)
        - "0:00" -> Timestamp(seconds)
        - "00:00:30" -> Timestamp(seconds)
        
        Returns None for unparseable values instead of raising.
        """
        if value is None:
            return None
            
        try:
            # Handle numeric types
            if isinstance(value, (int, float)):
                return cls(float(value))
                
            if isinstance(value, str):
                # Strip whitespace
                value = value.strip()
                
                # Empty string
                if not value:
                    logger.warning("Empty timestamp string, returning None")
                    return None
                
                # Handle range format: "0-1s" or "1.5-2.5s"
                range_match = re.match(r'(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)s?', value)
                if range_match:
                    start = float(range_match.group(1))
                    return cls(start)  # Use start of range
                    
                # Handle single value: "0s" or "0.5s" or just "5"
                single_match = re.match(r'(\d+(?:\.\d+)?)s?$', value)
                if single_match:
                    return cls(float(single_match.group(1)))
                    
                # Handle MM:SS format
                time_match = re.match(r'(\d+):(\d+)$', value)
                if time_match:
                    minutes = int(time_match.group(1))
                    seconds = int(time_match.group(2))
                    return cls(minutes * 60 + seconds)
                    
                # Handle HH:MM:SS format
                long_time_match = re.match(r'(\d+):(\d+):(\d+)$', value)
                if long_time_match:
                    hours = int(long_time_match.group(1))
                    minutes = int(long_time_match.group(2))
                    seconds = int(long_time_match.group(3))
                    return cls(hours * 3600 + minutes * 60 + seconds)
                
        except Exception as e:
            logger.warning(f"Exception parsing timestamp '{value}': {e}")
        
        # Log but don't crash - return None for unparseable
        logger.warning(f"Cannot parse timestamp: {value}, returning None")
        return None
    
    def to_string(self, format_type: str = "seconds") -> str:
        """
        Convert timestamp to string representation.
        
        Args:
            format_type: "seconds" (default), "range", "time"
            
        Returns:
            String representation of timestamp
        """
        if format_type == "seconds":
            return f"{int(self.seconds)}s"
        elif format_type == "range":
            return f"{int(self.seconds)}-{int(self.seconds + 1)}s"
        elif format_type == "time":
            minutes = int(self.seconds // 60)
            secs = int(self.seconds % 60)
            return f"{minutes}:{secs:02d}"
        else:
            return str(self.seconds)
    
    def to_json(self) -> Union[str, float]:
        """
        Serialize to JSON-compatible format.
        
        Args:
            legacy_mode: If True, return old string format for compatibility
            
        Returns:
            String format if legacy_mode, float otherwise
        """
        if legacy_mode:
            return f"{int(self.seconds)}s"  # Old format
        return self.seconds  # New format
    
    def __lt__(self, other: 'Timestamp') -> bool:
        """Compare timestamps."""
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self.seconds < other.seconds
    
    def __le__(self, other: 'Timestamp') -> bool:
        """Compare timestamps."""
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self.seconds <= other.seconds
    
    def __gt__(self, other: 'Timestamp') -> bool:
        """Compare timestamps."""
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self.seconds > other.seconds
    
    def __ge__(self, other: 'Timestamp') -> bool:
        """Compare timestamps."""
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self.seconds >= other.seconds
    
    def __eq__(self, other: object) -> bool:
        """Compare timestamps."""
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self.seconds == other.seconds
    
    def __hash__(self) -> int:
        """Make timestamp hashable for use in sets/dicts."""
        return hash(self.seconds)
    
    def __str__(self) -> str:
        """String representation."""
        return self.to_string()
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"Timestamp({self.seconds})"