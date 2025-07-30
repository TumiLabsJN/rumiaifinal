#!/usr/bin/env python3
"""
Test suite for Temporal Marker Processor.

CRITICAL: Tests the component with 90% failure rate.
"""
import unittest
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rumiai_v2.core.models import (
    UnifiedAnalysis, Timeline, TimelineEvent, 
    Timestamp, VideoMetadata
)
from rumiai_v2.processors import TemporalMarkerProcessor


class TestTemporalMarkerProcessor(unittest.TestCase):
    """Test temporal marker generation."""
    
    def setUp(self):
        """Create test data."""
        # Create minimal unified analysis
        self.analysis = UnifiedAnalysis(
            video_id="test_video_123",
            video_metadata={
                "video_id": "test_video_123",
                "duration": 60.0,
                "url": "https://tiktok.com/test",
                "description": "Test video"
            },
            timeline=Timeline(duration=60.0),
            ml_results={},
            temporal_markers=None
        )
        
        # Add test events
        self.analysis.timeline.add_event(TimelineEvent(
            timestamp=Timestamp(5.0),
            event_type="scene_change",
            description="Scene transition",
            source="scene_detection",
            confidence=0.95
        ))
        
        self.analysis.timeline.add_event(TimelineEvent(
            timestamp=Timestamp(10.0),
            event_type="speech",
            description="Hello world",
            source="speech_recognition",
            confidence=0.98,
            metadata={"text": "Hello world"}
        ))
        
        self.processor = TemporalMarkerProcessor()
    
    def test_generate_markers_success(self):
        """Test successful marker generation."""
        markers = self.processor.generate_markers(self.analysis)
        
        # Verify structure
        self.assertIsInstance(markers, dict)
        self.assertIn('video_id', markers)
        self.assertIn('markers', markers)
        self.assertIn('metadata', markers)
        self.assertIn('summary', markers)
        
        # Verify video_id
        self.assertEqual(markers['video_id'], 'test_video_123')
        
        # Verify has markers
        self.assertIsInstance(markers['markers'], list)
        self.assertGreater(len(markers['markers']), 0)
        
        # Verify marker structure
        for marker in markers['markers']:
            self.assertIn('timestamp', marker)
            self.assertIn('type', marker)
            self.assertIn('description', marker)
            self.assertIn('confidence', marker)
    
    def test_generate_markers_empty_timeline(self):
        """Test with empty timeline."""
        # Create analysis with empty timeline
        empty_analysis = UnifiedAnalysis(
            video_id="empty_video",
            video_metadata={"video_id": "empty_video", "duration": 30.0},
            timeline=Timeline(duration=30.0),
            ml_results={},
            temporal_markers=None
        )
        
        markers = self.processor.generate_markers(empty_analysis)
        
        # Should still return valid structure
        self.assertIsInstance(markers, dict)
        self.assertEqual(markers['video_id'], 'empty_video')
        self.assertIsInstance(markers['markers'], list)
        self.assertEqual(len(markers['markers']), 0)
    
    def test_generate_markers_with_errors(self):
        """Test error handling."""
        # Create analysis with problematic data
        bad_analysis = UnifiedAnalysis(
            video_id="bad_video",
            video_metadata={"video_id": "bad_video"},
            timeline=Timeline(duration=None),  # Missing duration
            ml_results={},
            temporal_markers=None
        )
        
        # Add event with None timestamp
        bad_analysis.timeline.events.append(TimelineEvent(
            timestamp=None,  # This will cause issues
            event_type="test",
            description="Bad event",
            source="test",
            confidence=1.0
        ))
        
        # Should not raise, returns empty markers
        markers = self.processor.generate_markers(bad_analysis)
        self.assertIsInstance(markers, dict)
        self.assertEqual(markers['video_id'], 'bad_video')
    
    def test_json_serialization(self):
        """Test that output is always valid JSON."""
        markers = self.processor.generate_markers(self.analysis)
        
        # Should be JSON serializable
        json_str = json.dumps(markers)
        self.assertIsInstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        self.assertEqual(parsed['video_id'], markers['video_id'])
    
    def test_marker_types(self):
        """Test different marker types are handled."""
        # Add various event types
        event_types = [
            ("emotion", "Happy moment", 0.85),
            ("object", "Person detected", 0.92),
            ("audio", "Music starts", 0.88),
            ("text", "Caption appears", 0.95),
        ]
        
        for event_type, desc, conf in event_types:
            self.analysis.timeline.add_event(TimelineEvent(
                timestamp=Timestamp(20.0 + len(event_types)),
                event_type=event_type,
                description=desc,
                source="test",
                confidence=conf
            ))
        
        markers = self.processor.generate_markers(self.analysis)
        
        # Verify all types are included
        marker_types = {m['type'] for m in markers['markers']}
        for event_type, _, _ in event_types:
            self.assertIn(event_type, marker_types)
    
    def test_timestamp_ordering(self):
        """Test markers are ordered by timestamp."""
        # Add events out of order
        self.analysis.timeline.add_event(TimelineEvent(
            timestamp=Timestamp(30.0),
            event_type="late",
            description="Late event",
            source="test",
            confidence=0.9
        ))
        
        self.analysis.timeline.add_event(TimelineEvent(
            timestamp=Timestamp(15.0),
            event_type="middle",
            description="Middle event",
            source="test",
            confidence=0.9
        ))
        
        markers = self.processor.generate_markers(self.analysis)
        
        # Verify ordering
        timestamps = [m['timestamp'] for m in markers['markers']]
        self.assertEqual(timestamps, sorted(timestamps))
    
    def test_confidence_filtering(self):
        """Test low confidence events are filtered."""
        # Add low confidence event
        self.analysis.timeline.add_event(TimelineEvent(
            timestamp=Timestamp(25.0),
            event_type="uncertain",
            description="Low confidence",
            source="test",
            confidence=0.3  # Below threshold
        ))
        
        markers = self.processor.generate_markers(self.analysis)
        
        # Verify high confidence only
        for marker in markers['markers']:
            self.assertGreaterEqual(marker['confidence'], 0.5)
    
    def test_metadata_generation(self):
        """Test metadata is properly generated."""
        markers = self.processor.generate_markers(self.analysis)
        
        metadata = markers['metadata']
        self.assertIn('generated_at', metadata)
        self.assertIn('version', metadata)
        self.assertIn('total_events', metadata)
        self.assertIn('duration', metadata)
        
        # Verify counts
        self.assertEqual(metadata['total_events'], len(self.analysis.timeline.events))
    
    def test_summary_generation(self):
        """Test summary statistics."""
        markers = self.processor.generate_markers(self.analysis)
        
        summary = markers['summary']
        self.assertIn('total_markers', summary)
        self.assertIn('marker_types', summary)
        self.assertIn('density', summary)
        
        # Verify density calculation
        if summary['total_markers'] > 0 and self.analysis.timeline.duration:
            expected_density = summary['total_markers'] / self.analysis.timeline.duration
            self.assertAlmostEqual(summary['density'], expected_density, places=2)


if __name__ == '__main__':
    unittest.main()