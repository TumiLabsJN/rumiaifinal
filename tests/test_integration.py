#!/usr/bin/env python3
"""
Integration tests for RumiAI v2.

Tests the full pipeline with mock data.
"""
import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
import sys
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from rumiai_v2.core.models import (
    VideoMetadata, UnifiedAnalysis, Timeline, 
    MLAnalysisResult, PromptResult, PromptType
)
from scripts.rumiai_runner import RumiAIRunner


class TestIntegration(unittest.TestCase):
    """Test full pipeline integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.unified_dir = Path(self.temp_dir) / "unified"
        self.insights_dir = Path(self.temp_dir) / "insights" 
        self.temporal_dir = Path(self.temp_dir) / "temporal"
        
        # Create directories
        for dir_path in [self.output_dir, self.unified_dir, self.insights_dir, self.temporal_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('rumiai_v2.api.ApifyClient')
    @patch('rumiai_v2.api.ClaudeClient')
    @patch('rumiai_v2.api.MLServices')
    @patch('rumiai_v2.config.Settings')
    def test_full_pipeline_success(self, mock_settings, mock_ml, mock_claude, mock_apify):
        """Test successful full pipeline execution."""
        # Configure settings
        mock_settings.return_value.output_dir = self.output_dir
        mock_settings.return_value.unified_dir = self.unified_dir
        mock_settings.return_value.insights_dir = self.insights_dir
        mock_settings.return_value.temporal_dir = self.temporal_dir
        mock_settings.return_value.temp_dir = Path(self.temp_dir) / "temp"
        mock_settings.return_value.prompt_delay = 0
        mock_settings.return_value.apify_token = "test"
        mock_settings.return_value.claude_api_key = "test"
        mock_settings.return_value.claude_model = "claude-3"
        mock_settings.return_value._prompt_templates = {}
        mock_settings.return_value.prompt_timeouts = {}
        
        # Mock Apify responses
        mock_apify_instance = AsyncMock()
        mock_apify_instance.scrape_video = AsyncMock(return_value=VideoMetadata(
            video_id="test123",
            url="https://tiktok.com/test123",
            download_url="https://download.url",
            duration=30.0,
            description="Test video",
            author="testuser",
            created_at="2024-01-01"
        ))
        
        video_path = Path(self.temp_dir) / "test123.mp4"
        video_path.touch()
        mock_apify_instance.download_video = AsyncMock(return_value=video_path)
        mock_apify.return_value = mock_apify_instance
        
        # Mock ML analysis
        mock_ml_instance = AsyncMock()
        mock_ml_instance.analyze_video = AsyncMock(return_value={
            'scene_detection': MLAnalysisResult(
                model_name='scene_detection',
                success=True,
                data={'scenes': [{'timestamp': 5.0, 'confidence': 0.9}]},
                processing_time=1.0
            ),
            'speech_recognition': MLAnalysisResult(
                model_name='speech_recognition', 
                success=True,
                data={'transcripts': [{'timestamp': 10.0, 'text': 'Hello'}]},
                processing_time=2.0
            )
        })
        mock_ml.return_value = mock_ml_instance
        
        # Mock Claude responses
        mock_claude_instance = Mock()
        mock_claude_instance.send_prompt = Mock(return_value=PromptResult(
            prompt_type=PromptType.CREATIVE_DENSITY,
            success=True,
            response="Analysis complete",
            tokens_used=100,
            estimated_cost=0.01,
            processing_time=1.0
        ))
        mock_claude.return_value = mock_claude_instance
        
        # Run pipeline
        runner = RumiAIRunner(legacy_mode=False)
        result = asyncio.run(runner.process_video_url("https://tiktok.com/test123"))
        
        # Verify success
        self.assertTrue(result['success'])
        self.assertEqual(result['video_id'], 'test123')
        
        # Verify outputs exist
        self.assertIn('outputs', result)
        self.assertIn('unified', result['outputs'])
        self.assertIn('temporal', result['outputs'])
        
        # Verify files created
        unified_path = Path(result['outputs']['unified'])
        self.assertTrue(unified_path.exists())
        
        temporal_path = Path(result['outputs']['temporal'])
        self.assertTrue(temporal_path.exists())
    
    @patch('rumiai_v2.config.Settings')
    def test_legacy_mode(self, mock_settings):
        """Test legacy video ID processing mode."""
        # Configure settings
        mock_settings.return_value.output_dir = self.output_dir
        mock_settings.return_value.unified_dir = self.unified_dir
        mock_settings.return_value.insights_dir = self.insights_dir
        mock_settings.return_value.temporal_dir = self.temporal_dir
        mock_settings.return_value.prompt_delay = 0
        mock_settings.return_value._prompt_templates = {}
        
        # Create existing unified analysis
        analysis = UnifiedAnalysis(
            video_id="legacy123",
            video_metadata={"video_id": "legacy123", "duration": 60.0},
            timeline=Timeline(duration=60.0),
            ml_results={},
            temporal_markers=None
        )
        
        unified_path = self.unified_dir / "legacy123.json"
        analysis.save_to_file(str(unified_path))
        
        # Mock Claude
        with patch('rumiai_v2.api.ClaudeClient') as mock_claude:
            mock_claude_instance = Mock()
            mock_claude_instance.send_prompt = Mock(return_value=PromptResult(
                prompt_type=PromptType.CREATIVE_DENSITY,
                success=True,
                response="Legacy analysis",
                tokens_used=50,
                estimated_cost=0.005,
                processing_time=0.5
            ))
            mock_claude.return_value = mock_claude_instance
            
            # Run in legacy mode
            runner = RumiAIRunner(legacy_mode=True)
            result = asyncio.run(runner.process_video_id("legacy123"))
            
            # Verify success
            self.assertTrue(result['success'])
            self.assertEqual(result['video_id'], 'legacy123')
            self.assertGreater(result['prompts_completed'], 0)
    
    def test_temporal_marker_integration(self):
        """Test temporal marker generation integration."""
        from rumiai_v2.processors import TemporalMarkerProcessor
        from rumiai_v2.core.models import TimelineEvent, Timestamp
        
        # Create analysis with events
        analysis = UnifiedAnalysis(
            video_id="marker_test",
            video_metadata={"video_id": "marker_test", "duration": 120.0},
            timeline=Timeline(duration=120.0),
            ml_results={},
            temporal_markers=None
        )
        
        # Add various events
        events = [
            (10.0, "scene_change", "Indoor to outdoor", 0.95),
            (25.0, "emotion", "Joy detected", 0.88),
            (30.0, "speech", "Important dialogue", 0.92),
            (45.0, "object", "Product shown", 0.85),
            (60.0, "music", "Beat drop", 0.90),
        ]
        
        for ts, event_type, desc, conf in events:
            analysis.timeline.add_event(TimelineEvent(
                timestamp=Timestamp(ts),
                event_type=event_type,
                description=desc,
                source="ml_model",
                confidence=conf
            ))
        
        # Generate markers
        processor = TemporalMarkerProcessor()
        markers = processor.generate_markers(analysis)
        
        # Verify structure
        self.assertEqual(markers['video_id'], 'marker_test')
        self.assertEqual(len(markers['markers']), len(events))
        
        # Verify JSON serializable
        json_str = json.dumps(markers)
        parsed = json.loads(json_str)
        self.assertEqual(parsed['video_id'], 'marker_test')
    
    @patch('rumiai_v2.api.ApifyClient')
    def test_error_handling(self, mock_apify):
        """Test pipeline error handling."""
        # Mock Apify to fail
        mock_apify_instance = AsyncMock()
        mock_apify_instance.scrape_video = AsyncMock(
            side_effect=Exception("Network error")
        )
        mock_apify.return_value = mock_apify_instance
        
        with patch('rumiai_v2.config.Settings') as mock_settings:
            mock_settings.return_value.output_dir = self.output_dir
            mock_settings.return_value.unified_dir = self.unified_dir
            mock_settings.return_value.insights_dir = self.insights_dir
            mock_settings.return_value.temporal_dir = self.temporal_dir
            
            runner = RumiAIRunner()
            result = asyncio.run(runner.process_video_url("https://tiktok.com/fail"))
            
            # Should return error result, not raise
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertEqual(result['error'], "Network error")
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old file structure."""
        # Create old-style unified analysis
        old_path = Path("unified_analysis") 
        old_path.mkdir(exist_ok=True)
        
        analysis_data = {
            "video_id": "old_style",
            "video_metadata": {"video_id": "old_style", "duration": 45.0},
            "timeline": {
                "duration": 45.0,
                "events": []
            },
            "ml_results": {},
            "temporal_markers": None
        }
        
        with open(old_path / "old_style.json", 'w') as f:
            json.dump(analysis_data, f)
        
        try:
            with patch('rumiai_v2.config.Settings') as mock_settings:
                mock_settings.return_value.output_dir = self.output_dir
                mock_settings.return_value.unified_dir = self.unified_dir
                mock_settings.return_value.insights_dir = self.insights_dir
                mock_settings.return_value.temporal_dir = self.temporal_dir
                mock_settings.return_value._prompt_templates = {}
                
                with patch('rumiai_v2.api.ClaudeClient') as mock_claude:
                    mock_claude.return_value.send_prompt = Mock(
                        return_value=PromptResult(
                            prompt_type=PromptType.CREATIVE_DENSITY,
                            success=True,
                            response="Old style works",
                            tokens_used=10,
                            estimated_cost=0.001,
                            processing_time=0.1
                        )
                    )
                    
                    runner = RumiAIRunner(legacy_mode=True)
                    result = asyncio.run(runner.process_video_id("old_style"))
                    
                    # Should find and process old file
                    self.assertTrue(result['success'])
                    self.assertEqual(result['video_id'], 'old_style')
        finally:
            # Cleanup
            shutil.rmtree(old_path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()