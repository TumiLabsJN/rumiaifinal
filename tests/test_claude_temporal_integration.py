"""
Test Claude Temporal Integration
"""

import pytest
import json
from python.claude_temporal_integration import ClaudeTemporalIntegration


class TestClaudeTemporalIntegration:
    """Test suite for Claude temporal integration"""
    
    @pytest.fixture
    def temporal_markers(self):
        """Sample temporal marker data"""
        return {
            'first_5_seconds': {
                'density_progression': [3, 2, 4, 1, 2],
                'text_moments': [
                    {'time': 0.5, 'text': 'WAIT FOR IT', 'size': 'L', 'position': 'center'},
                    {'time': 2.0, 'text': 'Amazing', 'size': 'M', 'position': 'bottom'}
                ],
                'emotion_sequence': ['neutral', 'happy', 'surprise', 'happy', 'neutral'],
                'gesture_moments': [
                    {'time': 1.5, 'gesture': 'pointing', 'confidence': 0.9, 'target': 'product'}
                ]
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'cta_appearances': [
                    {'time': 55.0, 'text': 'Follow for more', 'type': 'overlay'}
                ]
            }
        }
    
    def test_basic_integration(self):
        """Test basic integration functionality"""
        integrator = ClaudeTemporalIntegration(enable_temporal_markers=True)
        
        assert integrator.enable_temporal_markers is True
        assert integrator.rollout_percentage == 100.0
    
    def test_rollout_deterministic(self):
        """Test that rollout is deterministic by video_id"""
        integrator = ClaudeTemporalIntegration(
            enable_temporal_markers=True,
            rollout_percentage=50.0
        )
        
        # Same video_id should always get same result
        video_id = "test_video_123"
        result1 = integrator.should_include_temporal_markers(video_id)
        result2 = integrator.should_include_temporal_markers(video_id)
        assert result1 == result2
        
        # Different videos should have different results
        results = []
        for i in range(100):
            vid = f"video_{i}"
            results.append(integrator.should_include_temporal_markers(vid))
        
        # With 50% rollout, should have some True and some False
        true_count = sum(results)
        assert 30 < true_count < 70  # Allow some variance
    
    def test_format_temporal_markers(self, temporal_markers):
        """Test temporal marker formatting"""
        integrator = ClaudeTemporalIntegration()
        
        formatted = integrator.format_temporal_markers_for_claude(temporal_markers)
        
        # Check key sections are present
        assert "TEMPORAL PATTERN DATA:" in formatted
        assert "FIRST 5 SECONDS (Hook Window)" in formatted
        assert "CTA WINDOW" in formatted
        assert "KEY TEMPORAL INSIGHTS:" in formatted
        
        # Check specific data
        assert "WAIT FOR IT" in formatted
        assert "density" in formatted.lower()
        assert "Follow for more" in formatted
    
    def test_compact_mode(self, temporal_markers):
        """Test compact formatting mode"""
        integrator = ClaudeTemporalIntegration()
        
        regular = integrator.format_temporal_markers_for_claude(temporal_markers, compact=False)
        compact = integrator.format_temporal_markers_for_claude(temporal_markers, compact=True)
        
        # Compact should be shorter
        assert len(compact) < len(regular)
        
        # But still contain key info
        assert "WAIT FOR IT" in compact
        assert "Follow for more" in compact
    
    def test_build_context_with_markers(self, temporal_markers):
        """Test building context with temporal markers"""
        integrator = ClaudeTemporalIntegration(enable_temporal_markers=True)
        
        existing_context = {
            'video_stats': {'views': 1000, 'likes': 100},
            'duration': 60
        }
        
        video_id = "test_video"
        result = integrator.build_context_with_temporal_markers(
            existing_context, temporal_markers, video_id
        )
        
        # Should contain both contexts
        assert "CONTEXT DATA:" in result
        assert "video_stats" in result
        assert "TEMPORAL PATTERN DATA:" in result
        assert "WAIT FOR IT" in result
    
    def test_disabled_temporal_markers(self, temporal_markers):
        """Test when temporal markers are disabled"""
        integrator = ClaudeTemporalIntegration(enable_temporal_markers=False)
        
        existing_context = {'test': 'data'}
        video_id = "test_video"
        
        result = integrator.build_context_with_temporal_markers(
            existing_context, temporal_markers, video_id
        )
        
        # Should only have regular context
        assert "CONTEXT DATA:" in result
        assert "TEMPORAL PATTERN DATA:" not in result
    
    def test_size_estimation(self):
        """Test prompt size estimation"""
        integrator = ClaudeTemporalIntegration()
        
        context = "A" * 150000  # 150KB
        prompt = "B" * 50000   # 50KB
        
        size_info = integrator.get_prompt_size_estimate(context, prompt)
        
        assert size_info['total_chars'] == 200000
        assert size_info['size_kb'] > 195  # Close to 200KB
        assert len(size_info['warnings']) > 0
        assert "exceeds 180KB" in size_info['warnings'][0]
    
    def test_insights_generation(self, temporal_markers):
        """Test insight generation from temporal patterns"""
        integrator = ClaudeTemporalIntegration()
        
        formatted = integrator.format_temporal_markers_for_claude(temporal_markers)
        
        # Check insights are generated - density is 2.4 so should be moderate
        assert "MODERATE HOOK DENSITY" in formatted
        
        # Test high density
        temporal_markers['first_5_seconds']['density_progression'] = [4, 5, 4, 3, 4]
        formatted_high = integrator.format_temporal_markers_for_claude(temporal_markers)
        assert "HIGH HOOK DENSITY" in formatted_high
        
        # Test low density
        temporal_markers['first_5_seconds']['density_progression'] = [0, 0, 1, 0, 0]
        formatted_low = integrator.format_temporal_markers_for_claude(temporal_markers)
        assert "LOW HOOK DENSITY" in formatted_low
        
        # Test early CTA
        temporal_markers['first_5_seconds']['text_moments'] = [
            {'time': 1.0, 'text': 'Follow me now!', 'size': 'L'}
        ]
        formatted_cta = integrator.format_temporal_markers_for_claude(temporal_markers)
        assert "EARLY CTA" in formatted_cta
    
    def test_format_options(self):
        """Test configurable format options"""
        integrator = ClaudeTemporalIntegration()
        
        # Disable emotions
        integrator.format_options['include_emotions'] = False
        
        markers = {
            'first_5_seconds': {
                'emotion_sequence': ['happy', 'sad', 'neutral', 'happy', 'surprise'],
                'density_progression': [1, 1, 1, 1, 1]
            }
        }
        
        formatted = integrator.format_temporal_markers_for_claude(markers)
        
        # Emotions should not be in output
        assert "Emotion Flow:" not in formatted
        assert "happy" not in formatted.lower()
    
    def test_empty_markers(self):
        """Test handling of empty temporal markers"""
        integrator = ClaudeTemporalIntegration()
        
        # Empty markers
        formatted = integrator.format_temporal_markers_for_claude({})
        assert formatted == ""
        
        # None markers
        formatted = integrator.format_temporal_markers_for_claude(None)
        assert formatted == ""