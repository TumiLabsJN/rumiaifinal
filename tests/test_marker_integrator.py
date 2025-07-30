"""
Test Temporal Marker Integrator
Verifies the integrator correctly merges markers from all sources
"""

import pytest
import json
from python.temporal_marker_extractors import TemporalMarkerIntegrator


class TestTemporalMarkerIntegrator:
    """Test suite for temporal marker integrator"""
    
    @pytest.fixture
    def video_metadata(self):
        """Standard test video metadata"""
        return {
            'fps': 30.0,
            'extraction_fps': 2.0,
            'frame_count': 1800,
            'duration': 60.0
        }
    
    @pytest.fixture
    def ocr_markers(self):
        """Sample OCR markers"""
        return {
            'first_5_seconds': {
                'density_progression': [2, 1, 0, 1, 0],
                'text_moments': [
                    {'time': 0.0, 'text': 'WAIT FOR IT', 'size': 'L', 'position': 'center'},
                    {'time': 1.0, 'text': 'Amazing', 'size': 'M', 'position': 'bottom'},
                    {'time': 3.5, 'text': 'Watch this', 'size': 'M', 'position': 'top'}
                ]
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'cta_appearances': [
                    {'time': 55.0, 'text': 'Follow for more', 'type': 'overlay', 'size': 'L'}
                ]
            }
        }
    
    @pytest.fixture
    def yolo_markers(self):
        """Sample YOLO markers"""
        return {
            'first_5_seconds': {
                'density_progression': [1, 2, 2, 1, 1],
                'object_appearances': [
                    {'time': 0.0, 'objects': ['person'], 'confidence': [0.99]},
                    {'time': 1.0, 'objects': ['person', 'dog'], 'confidence': [0.98, 0.95]},
                    {'time': 2.0, 'objects': ['person', 'product'], 'confidence': [0.97, 0.90]}
                ]
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'object_focus': [
                    {'time': 52.0, 'object': 'person', 'confidence': 0.96},
                    {'time': 56.0, 'object': 'product', 'confidence': 0.94}
                ]
            }
        }
    
    @pytest.fixture
    def mediapipe_markers(self):
        """Sample MediaPipe markers"""
        return {
            'first_5_seconds': {
                'emotion_sequence': ['neutral', 'happy', 'happy', 'surprise', 'neutral'],
                'gesture_moments': [
                    {'time': 1.5, 'gesture': 'pointing', 'confidence': 0.92, 'target': 'product'},
                    {'time': 3.0, 'gesture': 'wave', 'confidence': 0.88}
                ]
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'gesture_sync': [
                    {'time': 55.5, 'gesture': 'pointing', 'aligns_with_cta': True, 'confidence': 0.91}
                ]
            }
        }
    
    def test_basic_integration(self, video_metadata, ocr_markers, yolo_markers, mediapipe_markers):
        """Test basic marker integration"""
        integrator = TemporalMarkerIntegrator(video_metadata)
        
        unified = integrator.integrate_markers(
            ocr_markers=ocr_markers,
            yolo_markers=yolo_markers,
            mediapipe_markers=mediapipe_markers
        )
        
        # Check structure
        assert 'first_5_seconds' in unified
        assert 'cta_window' in unified
        
        # Check all data was integrated
        assert len(unified['first_5_seconds']['text_moments']) == 3
        assert len(unified['first_5_seconds']['object_appearances']) == 3
        assert unified['first_5_seconds']['emotion_sequence'] == ['neutral', 'happy', 'happy', 'surprise', 'neutral']
        assert len(unified['first_5_seconds']['gesture_moments']) == 2
        
        # Check CTA window
        assert len(unified['cta_window']['cta_appearances']) == 1
        assert len(unified['cta_window']['object_focus']) == 2
        assert len(unified['cta_window']['gesture_sync']) == 1
    
    def test_density_merging(self, video_metadata, ocr_markers, yolo_markers):
        """Test density progression merging"""
        integrator = TemporalMarkerIntegrator(video_metadata)
        
        unified = integrator.integrate_markers(
            ocr_markers=ocr_markers,
            yolo_markers=yolo_markers
        )
        
        # Density should be sum of OCR + YOLO
        expected_density = [3, 3, 2, 2, 1]  # [2+1, 1+2, 0+2, 1+1, 0+1]
        assert unified['first_5_seconds']['density_progression'] == expected_density
    
    def test_partial_integration(self, video_metadata, ocr_markers):
        """Test integration with only some analyzers available"""
        integrator = TemporalMarkerIntegrator(video_metadata)
        
        # Only OCR available
        unified = integrator.integrate_markers(ocr_markers=ocr_markers)
        
        assert len(unified['first_5_seconds']['text_moments']) == 3
        assert unified['first_5_seconds']['emotion_sequence'] == ['neutral'] * 5  # Default
        assert len(unified['first_5_seconds']['gesture_moments']) == 0
        assert len(unified['first_5_seconds']['object_appearances']) == 0
    
    def test_no_analyzers(self, video_metadata):
        """Test integration with no analyzer data"""
        integrator = TemporalMarkerIntegrator(video_metadata)
        
        unified = integrator.integrate_markers()
        
        # Should have valid structure with empty/default data
        assert unified['first_5_seconds']['density_progression'] == [0, 0, 0, 0, 0]
        assert unified['first_5_seconds']['emotion_sequence'] == ['neutral'] * 5
        assert len(unified['first_5_seconds']['text_moments']) == 0
        assert len(unified['first_5_seconds']['gesture_moments']) == 0
        assert len(unified['first_5_seconds']['object_appearances']) == 0
    
    def test_cta_time_range_consistency(self, video_metadata, ocr_markers, yolo_markers, mediapipe_markers):
        """Test that CTA time range is consistent"""
        integrator = TemporalMarkerIntegrator(video_metadata)
        
        unified = integrator.integrate_markers(
            ocr_markers=ocr_markers,
            yolo_markers=yolo_markers,
            mediapipe_markers=mediapipe_markers
        )
        
        assert unified['cta_window']['time_range'] == '51.0-60.0s'
    
    def test_size_limits_applied(self, video_metadata):
        """Test that size reduction is applied during integration"""
        # Create oversized markers
        large_ocr = {
            'first_5_seconds': {
                'text_moments': [
                    {'time': i * 0.1, 'text': f'Text {i}' * 10, 'size': 'L'}
                    for i in range(50)  # Way over limit
                ],
                'density_progression': [10] * 5
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'cta_appearances': []
            }
        }
        
        integrator = TemporalMarkerIntegrator(video_metadata)
        unified = integrator.integrate_markers(ocr_markers=large_ocr)
        
        # Check that final size reduction is applied
        # The integrator applies TemporalMarkerSafety.check_and_reduce_size
        # which will reduce if total size exceeds limits
        json_size_kb = len(json.dumps(unified)) / 1024
        
        # Should be within reasonable limits
        assert json_size_kb < 180  # Hard limit with buffer
        
        # If size reduction was triggered, text moments might be reduced
        # but this depends on total size, not just count
        assert 'first_5_seconds' in unified
        assert 'text_moments' in unified['first_5_seconds']
    
    def test_different_video_durations(self):
        """Test integration with different video durations"""
        test_cases = [
            (30.0, '25.5-30.0s'),
            (60.0, '51.0-60.0s'),
            (120.0, '102.0-120.0s')
        ]
        
        for duration, expected_cta_range in test_cases:
            metadata = {
                'fps': 30.0,
                'extraction_fps': 2.0,
                'duration': duration,
                'frame_count': int(duration * 30)
            }
            
            integrator = TemporalMarkerIntegrator(metadata)
            unified = integrator.integrate_markers()
            
            assert unified['cta_window']['time_range'] == expected_cta_range
    
    def test_density_capping(self, video_metadata):
        """Test that density values are capped at reasonable levels"""
        # Create markers with very high density
        high_density_ocr = {
            'first_5_seconds': {
                'density_progression': [50, 50, 50, 50, 50],  # Unrealistically high
                'text_moments': []
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'cta_appearances': []
            }
        }
        
        integrator = TemporalMarkerIntegrator(video_metadata)
        unified = integrator.integrate_markers(ocr_markers=high_density_ocr)
        
        # Density should be capped at 10
        for density in unified['first_5_seconds']['density_progression']:
            assert density <= 10
    
    def test_json_serializable(self, video_metadata, ocr_markers, yolo_markers, mediapipe_markers):
        """Test that integrated markers are JSON serializable"""
        integrator = TemporalMarkerIntegrator(video_metadata)
        
        unified = integrator.integrate_markers(
            ocr_markers=ocr_markers,
            yolo_markers=yolo_markers,
            mediapipe_markers=mediapipe_markers
        )
        
        # Should not raise exception
        json_str = json.dumps(unified)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Can be loaded back
        loaded = json.loads(json_str)
        assert loaded['first_5_seconds']['emotion_sequence'] == unified['first_5_seconds']['emotion_sequence']