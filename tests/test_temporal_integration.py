"""
Test Temporal Marker Integration
Verifies the main integration pipeline works correctly
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from python.temporal_marker_integration import TemporalMarkerPipeline, extract_temporal_markers


class TestTemporalMarkerIntegration:
    """Test suite for temporal marker integration"""
    
    @pytest.fixture
    def mock_video_metadata(self):
        """Mock video metadata"""
        return {
            'fps': 30.0,
            'extraction_fps': 2.0,
            'duration': 60.0,
            'frame_count': 1800
        }
    
    @pytest.fixture
    def mock_ocr_data(self):
        """Mock OCR analysis data"""
        return {
            'frame_details': [
                {
                    'frame': 'frame_0000_t0.00.jpg',
                    'text_elements': [
                        {
                            'text': 'Welcome',
                            'confidence': 0.99,
                            'bbox': {'x1': 100, 'y1': 200, 'x2': 200, 'y2': 250},
                            'category': 'overlay_text'
                        }
                    ]
                },
                {
                    'frame': 'frame_0102_t51.00.jpg',
                    'text_elements': [
                        {
                            'text': 'Follow me',
                            'confidence': 0.98,
                            'bbox': {'x1': 150, 'y1': 350, 'x2': 250, 'y2': 380},
                            'category': 'call_to_action'
                        }
                    ]
                }
            ]
        }
    
    @pytest.fixture
    def mock_yolo_data(self):
        """Mock YOLO tracking data"""
        return {
            'objectAnnotations': [
                {
                    'trackId': 'object_1',
                    'entity': {'entityId': 'person', 'description': 'person'},
                    'confidence': 0.95,
                    'frames': [
                        {'frame': 0, 'confidence': 0.95},
                        {'frame': 30, 'confidence': 0.94},
                        {'frame': 1530, 'confidence': 0.93}
                    ]
                },
                {
                    'trackId': 'object_2',
                    'entity': {'entityId': 'product', 'description': 'product'},
                    'confidence': 0.90,
                    'frames': [
                        {'frame': 45, 'confidence': 0.90},
                        {'frame': 1650, 'confidence': 0.91}
                    ]
                }
            ]
        }
    
    @pytest.fixture
    def mock_mediapipe_data(self):
        """Mock MediaPipe analysis data"""
        return {
            'frame_analyses': [
                {
                    'frame': 0,
                    'faces': [{'expression': 'neutral'}],
                    'action_recognition': {'primary_action': 'talking', 'action_confidence': 0.8}
                },
                {
                    'frame': 2,
                    'faces': [{'expression': 'happy'}],
                    'action_recognition': {'primary_action': 'pointing', 'action_confidence': 0.9}
                },
                {
                    'frame': 102,
                    'faces': [{'expression': 'surprise'}],
                    'hands': [{'hand_type': 'right'}]
                }
            ]
        }
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = TemporalMarkerPipeline('test_video_123')
        assert pipeline.video_id == 'test_video_123'
        assert pipeline.base_dir == Path('.')
        assert pipeline.video_metadata is None
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_video_metadata_from_file(self, mock_file, mock_exists, mock_video_metadata):
        """Test getting video metadata from metadata.json"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(mock_video_metadata)
        
        pipeline = TemporalMarkerPipeline('test_video')
        metadata = pipeline._get_video_metadata()
        
        assert metadata == mock_video_metadata
        mock_exists.assert_called()
    
    @patch('pathlib.Path.exists')
    def test_get_video_metadata_fallback(self, mock_exists):
        """Test fallback when no metadata available"""
        mock_exists.return_value = False
        
        pipeline = TemporalMarkerPipeline('test_video')
        metadata = pipeline._get_video_metadata()
        
        assert metadata is None
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_extract_ocr_markers(self, mock_file, mock_exists, mock_ocr_data, mock_video_metadata):
        """Test OCR marker extraction"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(mock_ocr_data)
        
        pipeline = TemporalMarkerPipeline('test_video')
        pipeline.video_metadata = mock_video_metadata
        
        markers = pipeline._extract_ocr_markers()
        
        assert markers is not None
        assert 'first_5_seconds' in markers
        assert 'cta_window' in markers
        assert len(markers['first_5_seconds']['text_moments']) > 0
        assert len(markers['cta_window']['cta_appearances']) > 0
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_extract_yolo_markers(self, mock_file, mock_exists, mock_yolo_data, mock_video_metadata):
        """Test YOLO marker extraction"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(mock_yolo_data)
        
        pipeline = TemporalMarkerPipeline('test_video')
        pipeline.video_metadata = mock_video_metadata
        
        markers = pipeline._extract_yolo_markers()
        
        assert markers is not None
        assert 'first_5_seconds' in markers
        assert 'cta_window' in markers
        assert len(markers['first_5_seconds']['object_appearances']) > 0
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_extract_mediapipe_markers(self, mock_file, mock_exists, mock_mediapipe_data, mock_video_metadata):
        """Test MediaPipe marker extraction"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(mock_mediapipe_data)
        
        pipeline = TemporalMarkerPipeline('test_video')
        pipeline.video_metadata = mock_video_metadata
        
        markers = pipeline._extract_mediapipe_markers()
        
        assert markers is not None
        assert 'first_5_seconds' in markers
        assert 'cta_window' in markers
        assert len(markers['first_5_seconds']['emotion_sequence']) == 5
    
    def test_convert_object_annotations(self, mock_yolo_data):
        """Test conversion of object annotations to tracking format"""
        pipeline = TemporalMarkerPipeline('test_video')
        tracking_data = pipeline._convert_object_annotations(mock_yolo_data['objectAnnotations'])
        
        assert 'tracks' in tracking_data
        assert len(tracking_data['tracks']) == 2
        assert tracking_data['tracks'][0]['track_id'] == '1'
        assert tracking_data['tracks'][0]['detections'][0]['class'] == 'person'
    
    def test_convert_frame_analyses_to_timeline(self, mock_mediapipe_data):
        """Test conversion of frame analyses to timeline format"""
        pipeline = TemporalMarkerPipeline('test_video')
        timeline_data = pipeline._convert_frame_analyses_to_timeline(mock_mediapipe_data['frame_analyses'])
        
        assert 'timeline' in timeline_data
        assert 'expressions' in timeline_data['timeline']
        assert 'gestures' in timeline_data['timeline']
        assert len(timeline_data['timeline']['expressions']) == 3
        assert len(timeline_data['timeline']['gestures']) == 2  # pointing + open_hand
    
    @patch.object(TemporalMarkerPipeline, '_get_video_metadata')
    @patch.object(TemporalMarkerPipeline, '_extract_ocr_markers')
    @patch.object(TemporalMarkerPipeline, '_extract_yolo_markers')
    @patch.object(TemporalMarkerPipeline, '_extract_mediapipe_markers')
    def test_extract_all_markers(self, mock_mp, mock_yolo, mock_ocr, mock_metadata, mock_video_metadata):
        """Test full marker extraction pipeline"""
        # Setup mocks
        mock_metadata.return_value = mock_video_metadata
        mock_ocr.return_value = {
            'first_5_seconds': {
                'text_moments': [{'time': 0.0, 'text': 'Hello'}],
                'density_progression': [1, 0, 0, 0, 0]
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'cta_appearances': []
            }
        }
        mock_yolo.return_value = {
            'first_5_seconds': {
                'object_appearances': [{'time': 0.0, 'objects': ['person']}],
                'density_progression': [1, 1, 0, 0, 0]
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'object_focus': []
            }
        }
        mock_mp.return_value = {
            'first_5_seconds': {
                'emotion_sequence': ['neutral', 'happy', 'neutral', 'neutral', 'neutral'],
                'gesture_moments': []
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'gesture_sync': []
            }
        }
        
        pipeline = TemporalMarkerPipeline('test_video')
        markers = pipeline.extract_all_markers()
        
        assert 'first_5_seconds' in markers
        assert 'cta_window' in markers
        assert 'metadata' in markers
        assert markers['metadata']['video_id'] == 'test_video'
        assert markers['metadata']['video_duration'] == 60.0
        
        # Check integration worked
        assert len(markers['first_5_seconds']['text_moments']) > 0
        assert len(markers['first_5_seconds']['object_appearances']) > 0
        assert markers['first_5_seconds']['emotion_sequence'] == ['neutral', 'happy', 'neutral', 'neutral', 'neutral']
    
    @patch('pathlib.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_markers(self, mock_file, mock_mkdir):
        """Test saving markers to file"""
        pipeline = TemporalMarkerPipeline('test_video')
        markers = {'test': 'data'}
        
        output_path = pipeline.save_markers(markers)
        
        assert 'test_video_temporal_markers.json' in output_path
        mock_file.assert_called()
        mock_file().write.assert_called()
    
    def test_get_marker_summary(self, mock_video_metadata):
        """Test marker summary generation"""
        pipeline = TemporalMarkerPipeline('test_video')
        pipeline.video_metadata = mock_video_metadata
        
        markers = {
            'first_5_seconds': {
                'text_moments': [{'time': 0.0, 'text': 'Hi'}, {'time': 1.0, 'text': 'There'}],
                'density_progression': [2, 1, 0, 0, 0],
                'emotion_sequence': ['happy', 'happy', 'neutral', 'neutral', 'neutral'],
                'gesture_moments': [{'time': 1.5, 'gesture': 'wave'}],
                'object_appearances': [{'time': 0.0, 'objects': ['person']}]
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'cta_appearances': [{'time': 55.0, 'text': 'Follow'}],
                'gesture_sync': [],
                'object_focus': []
            }
        }
        
        summary = pipeline.get_marker_summary(markers)
        
        assert summary['video_id'] == 'test_video'
        assert summary['duration'] == 60.0
        assert summary['first_5_seconds']['text_moments'] == 2
        assert summary['first_5_seconds']['density_avg'] == 0.6  # (2+1+0+0+0)/5
        assert summary['first_5_seconds']['gesture_count'] == 1
        assert summary['cta_window']['cta_count'] == 1
        assert summary['size_kb'] > 0
    
    @patch.object(TemporalMarkerPipeline, 'extract_all_markers')
    def test_convenience_function(self, mock_extract):
        """Test the convenience function"""
        mock_extract.return_value = {'test': 'markers'}
        
        result = extract_temporal_markers('test_video')
        
        assert result == {'test': 'markers'}
        mock_extract.assert_called_once()
    
    def test_no_analyzers_available(self, mock_video_metadata):
        """Test handling when no analyzer data is available"""
        pipeline = TemporalMarkerPipeline('test_video')
        pipeline.video_metadata = mock_video_metadata
        
        # Mock all extractors to return None
        with patch.object(pipeline, '_extract_ocr_markers', return_value=None), \
             patch.object(pipeline, '_extract_yolo_markers', return_value=None), \
             patch.object(pipeline, '_extract_mediapipe_markers', return_value=None):
            
            markers = pipeline.extract_all_markers()
            
            # Should still return valid structure
            assert 'first_5_seconds' in markers
            assert 'cta_window' in markers
            assert 'metadata' in markers
            
            # But with minimal data
            assert markers['first_5_seconds']['density_progression'] == [0, 0, 0, 0, 0]
            assert markers['first_5_seconds']['emotion_sequence'] == ['neutral'] * 5