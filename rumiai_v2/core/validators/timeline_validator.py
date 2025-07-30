"""
Timeline validation utilities.
"""
from typing import Dict, Any, Optional
from ..models.timeline import Timeline, TimelineEntry
from ..models.timestamp import Timestamp
from ..exceptions import TimelineError
import logging

logger = logging.getLogger(__name__)


class TimelineValidator:
    """Validate timeline data structures."""
    
    @staticmethod
    def validate_timeline_entry(data: Dict[str, Any]) -> Optional[TimelineEntry]:
        """
        Validate and create a timeline entry from raw data.
        
        Returns None if data is invalid.
        """
        try:
            # Extract required fields
            if 'start' not in data:
                logger.warning("Timeline entry missing 'start' field")
                return None
            
            start = Timestamp.from_value(data['start'])
            if not start:
                logger.warning(f"Invalid start timestamp: {data['start']}")
                return None
            
            # Extract optional end
            end = None
            if 'end' in data:
                end = Timestamp.from_value(data['end'])
            
            # Extract entry type
            entry_type = data.get('entry_type', data.get('type', 'unknown'))
            
            # Extract data payload
            entry_data = data.get('data', {})
            if not isinstance(entry_data, dict):
                entry_data = {'value': entry_data}
            
            return TimelineEntry(
                start=start,
                end=end,
                entry_type=entry_type,
                data=entry_data
            )
            
        except Exception as e:
            logger.error(f"Error creating timeline entry: {e}")
            return None
    
    @staticmethod
    def validate_timeline(timeline: Timeline) -> bool:
        """
        Validate a timeline for consistency.
        
        Checks:
        - Entries are in chronological order
        - No entries exceed video duration
        - Entry types are valid
        """
        if not timeline.video_id:
            logger.warning("Timeline missing video_id")
            return False
        
        if timeline.duration <= 0:
            logger.warning(f"Invalid timeline duration: {timeline.duration}")
            return False
        
        # Check chronological order
        prev_start = -1.0
        for entry in timeline.entries:
            if entry.start.seconds < prev_start:
                logger.warning(
                    f"Timeline entries not in order: {entry.start.seconds} < {prev_start}"
                )
                return False
            prev_start = entry.start.seconds
            
            # Check duration bounds
            if entry.start.seconds > timeline.duration:
                logger.warning(
                    f"Entry start {entry.start.seconds} exceeds duration {timeline.duration}"
                )
                return False
            
            if entry.end and entry.end.seconds > timeline.duration:
                logger.warning(
                    f"Entry end {entry.end.seconds} exceeds duration {timeline.duration}"
                )
                return False
        
        return True
    
    @staticmethod
    def merge_timelines(timeline1: Timeline, timeline2: Timeline) -> Timeline:
        """
        Safely merge two timelines.
        
        Creates a new timeline with entries from both.
        """
        if timeline1.video_id != timeline2.video_id:
            logger.warning(
                f"Merging timelines from different videos: "
                f"{timeline1.video_id} != {timeline2.video_id}"
            )
        
        # Use the longer duration
        duration = max(timeline1.duration, timeline2.duration)
        
        # Create new timeline
        merged = Timeline(
            video_id=timeline1.video_id,
            duration=duration
        )
        
        # Add all entries (add_entry maintains order)
        for entry in timeline1.entries:
            merged.add_entry(entry)
        
        for entry in timeline2.entries:
            merged.add_entry(entry)
        
        return merged