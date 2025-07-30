#!/usr/bin/env python3
"""
Test with real fixture data.

Validates system works with realistic data structures.
"""
import unittest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rumiai_v2.core.models import UnifiedAnalysis
from rumiai_v2.processors import TemporalMarkerProcessor, MLDataExtractor
from rumiai_v2.core.models import PromptType


class TestFixtures(unittest.TestCase):
    """Test with fixture data."""
    
    def setUp(self):
        """Load fixture data."""
        fixture_path = Path(__file__).parent / "fixtures" / "sample_unified_analysis.json"
        self.analysis = UnifiedAnalysis.load_from_file(str(fixture_path))
    
    def test_load_fixture(self):
        """Test fixture loads correctly."""
        self.assertEqual(self.analysis.video_id, "test_fixture_123")
        self.assertEqual(self.analysis.timeline.duration, 45.67)
        self.assertEqual(len(self.analysis.timeline.events), 7)
        self.assertEqual(len(self.analysis.ml_results), 3)
    
    def test_temporal_markers_from_fixture(self):
        """Test temporal marker generation from fixture."""
        processor = TemporalMarkerProcessor()
        markers = processor.generate_markers(self.analysis)
        
        # Verify markers generated
        self.assertEqual(markers['video_id'], 'test_fixture_123')
        self.assertGreater(len(markers['markers']), 0)
        
        # Verify marker types from fixture events
        marker_types = {m['type'] for m in markers['markers']}
        expected_types = {'scene_change', 'speech', 'emotion', 'object', 'music'}
        self.assertEqual(marker_types & expected_types, expected_types)
        
        # Verify timestamps preserved
        speech_markers = [m for m in markers['markers'] if m['type'] == 'speech']
        self.assertTrue(any(m['timestamp'] == 10.3 for m in speech_markers))
    
    def test_ml_data_extraction(self):
        """Test ML data extraction for prompts."""
        extractor = MLDataExtractor()
        
        # Test creative density extraction
        context = extractor.extract_for_prompt(
            self.analysis, 
            PromptType.CREATIVE_DENSITY
        )
        
        self.assertIn('timeline_events', context.data)
        self.assertIn('scene_changes', context.data)
        self.assertEqual(len(context.data['scene_changes']), 1)  # One scene change in fixture
        
        # Test emotional journey extraction
        context = extractor.extract_for_prompt(
            self.analysis,
            PromptType.EMOTIONAL_JOURNEY
        )
        
        self.assertIn('emotions', context.data)
        self.assertTrue(any(e['emotion'] == 'happy' for e in context.data['emotions']))
    
    def test_timeline_integrity(self):
        """Test timeline event ordering and integrity."""
        events = self.analysis.timeline.get_events()
        
        # Verify chronological order
        timestamps = [e.timestamp.seconds for e in events]
        self.assertEqual(timestamps, sorted(timestamps))
        
        # Verify start and end events
        self.assertEqual(events[0].event_type, 'start')
        self.assertEqual(events[-1].event_type, 'end')
        
        # Verify all events within duration
        for event in events:
            self.assertLessEqual(event.timestamp.seconds, self.analysis.timeline.duration)
            self.assertGreaterEqual(event.timestamp.seconds, 0)
    
    def test_ml_result_structure(self):
        """Test ML result data structure."""
        # Scene detection
        scene_data = self.analysis.ml_results['scene_detection']
        self.assertTrue(scene_data.success)
        self.assertIn('scenes', scene_data.data)
        self.assertEqual(len(scene_data.data['scenes']), 2)
        
        # Speech recognition
        speech_data = self.analysis.ml_results['speech_recognition']
        self.assertTrue(speech_data.success)
        self.assertIn('transcripts', speech_data.data)
        self.assertEqual(speech_data.data['language'], 'en')
        
        # Emotion recognition
        emotion_data = self.analysis.ml_results['emotion_recognition']
        self.assertTrue(emotion_data.success)
        self.assertEqual(emotion_data.data['dominant_emotion'], 'happy')
    
    def test_metadata_preservation(self):
        """Test video metadata is preserved."""
        metadata = self.analysis.video_metadata
        
        self.assertEqual(metadata['video_id'], 'test_fixture_123')
        self.assertEqual(metadata['duration'], 45.67)
        self.assertIn('stats', metadata)
        self.assertEqual(metadata['stats']['plays'], 12345)
    
    def test_fixture_completeness(self):
        """Test fixture has all required fields."""
        # Test is_complete
        self.assertFalse(self.analysis.is_complete())  # Some ML analyses missing
        
        # Test completion status
        status = self.analysis.get_completion_status()
        self.assertEqual(status['scene_detection'], True)
        self.assertEqual(status['speech_recognition'], True)
        self.assertEqual(status['emotion_recognition'], True)
        
        # Missing analyses should be False
        self.assertEqual(status['object_detection'], False)
        self.assertEqual(status['text_recognition'], False)


if __name__ == '__main__':
    unittest.main()