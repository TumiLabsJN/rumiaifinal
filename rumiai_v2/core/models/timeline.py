"""
Timeline model for RumiAI v2.

This module provides the unified timeline structure used throughout the system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
from .timestamp import Timestamp
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimelineEntry:
    """
    Single entry in a timeline.
    
    CRITICAL: This is used by all ML processors and Claude prompts.
    """
    start: Timestamp
    end: Optional[Timestamp]  # None for instantaneous events
    entry_type: str  # "speech", "text", "object", "scene_change", etc.
    data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get duration of entry in seconds."""
        if self.end:
            return self.end.seconds - self.start.seconds
        return 0.0
    
    def overlaps(self, other: 'TimelineEntry') -> bool:
        """Check if two entries overlap in time."""
        if not self.end or not other.end:
            # Instantaneous events don't overlap
            return False
        return (self.start.seconds < other.end.seconds and 
                self.end.seconds > other.start.seconds)
    
    def contains_time(self, timestamp: float) -> bool:
        """Check if a timestamp falls within this entry."""
        if self.end:
            return self.start.seconds <= timestamp < self.end.seconds
        else:
            # For instantaneous events, check exact match
            return abs(self.start.seconds - timestamp) < 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'start': self.start.to_json(legacy_mode),
            'entry_type': self.entry_type,
            'data': self.data
        }
        if self.end:
            result['end'] = self.end.to_json(legacy_mode)
        return result


@dataclass
class Timeline:
    """
    Unified timeline structure.
    
    CRITICAL PATH: This model is used by ALL 7 Claude prompts.
    Any failure here affects everything.
    """
    video_id: str
    duration: float  # Total video duration in seconds
    entries: List[TimelineEntry] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate timeline on creation."""
        if self.duration <= 0:
            logger.warning(f"Invalid duration {self.duration} for video {self.video_id}")
            self.duration = 1.0  # Minimum duration to prevent division by zero
    
    def add_entry(self, entry: TimelineEntry) -> None:
        """
        Add entry maintaining chronological order.
        
        CRITICAL: MediaPipe/YOLO often report timestamps beyond video duration.
        Must handle gracefully to prevent pipeline failure.
        """
        # Clamp timestamps to video duration
        if entry.start.seconds > self.duration:
            logger.warning(
                f"Entry start {entry.start.seconds} exceeds duration {self.duration}, clamping"
            )
            entry.start = Timestamp(self.duration)
        
        if entry.end and entry.end.seconds > self.duration:
            logger.warning(
                f"Entry end {entry.end.seconds} exceeds duration {self.duration}, clamping"
            )
            entry.end = Timestamp(self.duration)
        
        # Insert in chronological order
        insert_idx = 0
        for i, existing in enumerate(self.entries):
            if entry.start.seconds < existing.start.seconds:
                insert_idx = i
                break
            insert_idx = i + 1
        
        self.entries.insert(insert_idx, entry)
    
    def get_entries_in_range(self, start: float, end: float) -> List[TimelineEntry]:
        """
        Get all entries within time range.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            List of entries that overlap with the given range
        """
        # Validate range
        if start < 0:
            start = 0
        if end > self.duration:
            end = self.duration
        if start >= end:
            return []
        
        result = []
        for entry in self.entries:
            # Check if entry overlaps with range
            entry_start = entry.start.seconds
            entry_end = entry.end.seconds if entry.end else entry_start
            
            if entry_end >= start and entry_start <= end:
                result.append(entry)
        
        return result
    
    def get_entries_by_type(self, entry_type: str) -> List[TimelineEntry]:
        """Get all entries of specific type."""
        return [entry for entry in self.entries if entry.entry_type == entry_type]
    
    def get_entries_at_time(self, timestamp: float) -> List[TimelineEntry]:
        """Get all entries active at a specific timestamp."""
        return [entry for entry in self.entries if entry.contains_time(timestamp)]
    
    def get_density_buckets(self, bucket_size: float = 1.0) -> List[int]:
        """
        Calculate entry density in time buckets.
        
        Args:
            bucket_size: Size of each bucket in seconds
            
        Returns:
            List of entry counts per bucket
        """
        num_buckets = int(self.duration / bucket_size) + 1
        buckets = [0] * num_buckets
        
        for entry in self.entries:
            bucket_idx = int(entry.start.seconds / bucket_size)
            if bucket_idx < num_buckets:
                buckets[bucket_idx] += 1
        
        return buckets
    
    def merge_with(self, other: 'Timeline') -> None:
        """
        Merge another timeline into this one.
        
        Used when combining results from multiple ML models.
        """
        if other.video_id != self.video_id:
            logger.warning(
                f"Merging timelines from different videos: {self.video_id} != {other.video_id}"
            )
        
        # Add all entries from other timeline
        for entry in other.entries:
            self.add_entry(entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'video_id': self.video_id,
            'duration': self.duration,
            'entry_count': len(self.entries),
            'entries': [entry.to_dict(legacy_mode) for entry in self.entries]
        }
    
    def to_prompt_format(self, entry_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert timeline to format expected by Claude prompts.
        
        This maintains backward compatibility with existing prompt templates.
        """
        if entry_type:
            entries = self.get_entries_by_type(entry_type)
        else:
            entries = self.entries
        
        # Group by timestamp for old format compatibility
        timeline_dict = {}
        for entry in entries:
            # Use legacy timestamp format
            timestamp_key = entry.start.to_string()
            
            if timestamp_key not in timeline_dict:
                timeline_dict[timestamp_key] = []
            
            timeline_dict[timestamp_key].append(entry.data)
        
        return timeline_dict
    
    def __len__(self) -> int:
        """Get number of entries."""
        return len(self.entries)
    
    def __iter__(self) -> Iterator[TimelineEntry]:
        """Iterate over entries."""
        return iter(self.entries)
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (f"Timeline(video_id={self.video_id}, duration={self.duration}, "
                f"entries={len(self.entries)})")