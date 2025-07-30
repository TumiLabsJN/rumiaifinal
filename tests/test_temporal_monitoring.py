"""
Test Temporal Monitoring System
"""

import pytest
import json
import tempfile
from pathlib import Path
from python.temporal_monitoring import TemporalMarkerMonitor


class TestTemporalMonitoring:
    """Test suite for temporal monitoring"""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor with temp directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = TemporalMarkerMonitor(metrics_dir=tmpdir)
            yield monitor
    
    def test_record_extraction_success(self, monitor):
        """Test recording successful extraction"""
        monitor.record_extraction(
            video_id="test_123",
            success=True,
            extraction_time=2.5,
            marker_size_kb=45.2
        )
        
        metrics = monitor.get_current_metrics()
        assert metrics['extraction_count'] == 1
        assert metrics['extraction_errors'] == 0
        assert metrics['avg_extraction_time'] == 2.5
        assert metrics['avg_marker_size_kb'] == 45.2
    
    def test_record_extraction_failure(self, monitor):
        """Test recording failed extraction"""
        monitor.record_extraction(
            video_id="test_456",
            success=False,
            extraction_time=0.5,
            error="File not found"
        )
        
        metrics = monitor.get_current_metrics()
        assert metrics['extraction_count'] == 1
        assert metrics['extraction_errors'] == 1
        assert metrics['extraction_error_rate'] == 1.0
    
    def test_record_claude_request(self, monitor):
        """Test recording Claude API request"""
        monitor.record_claude_request(
            video_id="test_789",
            prompt_name="hook_analysis",
            has_temporal_markers=True,
            rollout_decision="included",
            prompt_size_kb=125.5,
            success=True
        )
        
        metrics = monitor.get_current_metrics()
        assert metrics['claude_requests_with_markers'] == 1
        assert metrics['claude_requests_without_markers'] == 0
        assert metrics['temporal_marker_adoption'] == 1.0
        assert metrics['rollout_decisions']['included'] == 1
    
    def test_rollout_health_check(self, monitor):
        """Test rollout health checking"""
        # Add some good data
        for i in range(15):
            monitor.record_extraction(
                video_id=f"test_{i}",
                success=True,
                extraction_time=2.0,
                marker_size_kb=50.0
            )
        
        # Add one failure (should still be healthy)
        monitor.record_extraction(
            video_id="test_fail",
            success=False,
            extraction_time=0.1,
            error="Test error"
        )
        
        health = monitor.check_rollout_health()
        assert health['healthy'] is True
        assert health['checks']['extraction_error_rate_ok'] is True
        assert health['checks']['sufficient_data'] is True
    
    def test_unhealthy_rollout(self, monitor):
        """Test unhealthy rollout conditions"""
        # Add mostly failures
        for i in range(10):
            monitor.record_extraction(
                video_id=f"test_{i}",
                success=i < 5,  # 50% failure rate
                extraction_time=1.0,
                marker_size_kb=50.0 if i < 5 else None,
                error=None if i < 5 else "Failed"
            )
        
        health = monitor.check_rollout_health()
        assert health['healthy'] is False
        assert health['checks']['extraction_error_rate_ok'] is False
        assert len(health['recommendations']) > 0
    
    def test_quality_comparison(self, monitor):
        """Test insight quality comparison"""
        # Initialize the quality scores dict if not present
        if 'insights_quality_scores' not in monitor.historical_metrics:
            monitor.historical_metrics['insights_quality_scores'] = {}
            
        # Record quality scores
        for i in range(5):
            monitor.record_insight_quality(
                video_id=f"test_{i}",
                prompt_name="hook_analysis",
                with_markers=True,
                quality_score=8.5 + i * 0.1,
                specific_patterns_found=["pattern1", "pattern2"]
            )
            
            monitor.record_insight_quality(
                video_id=f"test_{i}",
                prompt_name="hook_analysis",
                with_markers=False,
                quality_score=6.0 + i * 0.1,
                specific_patterns_found=["pattern1"]
            )
        
        comparison = monitor.get_quality_comparison()
        assert 'hook_analysis' in comparison
        assert comparison['hook_analysis']['improvement'] > 0
        assert comparison['hook_analysis']['with_markers']['avg_score'] > 8.0
        assert comparison['hook_analysis']['without_markers']['avg_score'] > 5.0
    
    def test_metrics_persistence(self, monitor):
        """Test saving and loading metrics"""
        # Add some data
        monitor.record_extraction("test_1", True, 2.5, 45.0)
        monitor.record_extraction("test_2", False, 0.5, error="Failed")
        
        # Save session
        monitor.save_session_metrics()
        
        # Create new monitor with same directory
        new_monitor = TemporalMarkerMonitor(metrics_dir=monitor.metrics_dir)
        
        # Check historical data was loaded
        assert new_monitor.historical_metrics['total_extractions'] == 2
        assert new_monitor.historical_metrics['total_errors'] == 1
        assert len(new_monitor.historical_metrics['sessions']) == 1
    
    def test_generate_report(self, monitor):
        """Test report generation"""
        # Add varied data
        monitor.record_extraction("test_1", True, 2.5, 45.0)
        monitor.record_claude_request(
            "test_1", "hook_analysis", True, "included", 125.0, True
        )
        
        report = monitor.generate_report()
        
        # Check report contains key sections
        assert "TEMPORAL MARKER METRICS REPORT" in report
        assert "EXTRACTION METRICS:" in report
        assert "CLAUDE INTEGRATION:" in report
        assert "ROLLOUT DECISIONS:" in report
        assert "HISTORICAL SUMMARY:" in report
    
    def test_size_statistics(self, monitor):
        """Test marker size statistics"""
        sizes = [30.0, 45.0, 60.0, 75.0, 90.0]
        
        for i, size in enumerate(sizes):
            monitor.record_extraction(
                video_id=f"test_{i}",
                success=True,
                extraction_time=2.0,
                marker_size_kb=size
            )
        
        metrics = monitor.get_current_metrics()
        assert metrics['avg_marker_size_kb'] == 60.0  # Average
        assert metrics['min_marker_size_kb'] == 30.0
        assert metrics['max_marker_size_kb'] == 90.0
    
    def test_api_error_tracking(self, monitor):
        """Test API error tracking"""
        # Record some successful requests
        for i in range(8):
            monitor.record_claude_request(
                f"test_{i}", "hook_analysis", True, "included", 100.0, True
            )
        
        # Record some failures
        for i in range(2):
            monitor.record_claude_request(
                f"test_fail_{i}", "hook_analysis", True, "included", 100.0, False,
                error="Rate limit exceeded"
            )
        
        metrics = monitor.get_current_metrics()
        assert metrics['api_error_rate'] == 0.2  # 2/10 = 20%
        assert len(metrics['api_errors']) == 2