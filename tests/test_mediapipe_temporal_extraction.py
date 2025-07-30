"""
Test MediaPipe Temporal Marker Extraction
Verifies that the MediaPipe analyzer correctly extracts temporal markers
"""

import pytest
from python.temporal_marker_extractors import MediaPipeTemporalExtractor
from python.temporal_marker_safety import TemporalMarkerSafety


class TestMediaPipeTemporalExtraction:
    """Test suite for MediaPipe temporal marker extraction"""
    
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
    def sample_analysis_data(self):
        """Sample MediaPipe analysis data"""
        return {
            'timeline': {
                'expressions': [
                    {'frame': 0, 'expression': 'neutral'},      # 0s (extracted frame 0)
                    {'frame': 2, 'expression': 'happy'},        # 1s (extracted frame 2)
                    {'frame': 4, 'expression': 'happy'},        # 2s (extracted frame 4)
                    {'frame': 6, 'expression': 'surprise'},     # 3s (extracted frame 6)
                    {'frame': 8, 'expression': 'joy'},          # 4s (extracted frame 8)
                    {'frame': 10, 'expression': 'neutral'},     # 5s (extracted frame 10)
                    {'frame': 102, 'expression': 'happy'},      # 51s (CTA window - frame 102 at 2fps = 51s)
                    {'frame': 110, 'expression': 'surprise'},   # 55s (frame 110 at 2fps = 55s)
                ],
                'gestures': [
                    {'frame': 3, 'gesture': 'pointing', 'confidence': 0.95, 'target': 'product'},     # 1.5s
                    {'frame': 6, 'gesture': 'wave', 'confidence': 0.90},                            # 3s
                    {'frame': 9, 'gesture': 'thumbs_up', 'confidence': 0.92},                       # 4.5s
                    {'frame': 105, 'gesture': 'pointing', 'confidence': 0.94},                      # 52.5s (CTA)
                    {'frame': 115, 'gesture': 'clapping', 'confidence': 0.96},                      # 57.5s (CTA)
                ]
            }
        }
    
    def test_basic_extraction(self, video_metadata, sample_analysis_data):
        """Test basic temporal marker extraction"""
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(sample_analysis_data)
        
        # Check structure
        assert 'first_5_seconds' in markers
        assert 'cta_window' in markers
        
        # Check first 5 seconds
        first_5 = markers['first_5_seconds']
        assert 'emotion_sequence' in first_5
        assert 'gesture_moments' in first_5
        
        # Check emotion sequence
        assert len(first_5['emotion_sequence']) == 5
        assert first_5['emotion_sequence'] == ['neutral', 'happy', 'happy', 'surprise', 'happy']
        
        # Check gesture moments
        assert len(first_5['gesture_moments']) == 3
        assert all(g['time'] < 5.0 for g in first_5['gesture_moments'])
        
        # Check CTA window
        cta = markers['cta_window']
        assert cta['time_range'] == '51.0-60.0s'
        assert len(cta['gesture_sync']) == 2
    
    def test_emotion_standardization(self, video_metadata):
        """Test emotion vocabulary standardization"""
        analysis_data = {
            'timeline': {
                'expressions': [
                    {'frame': 0, 'expression': 'happiness'},    # Should map to 'happy'
                    {'frame': 2, 'expression': 'joy'},          # Should map to 'happy'
                    {'frame': 4, 'expression': 'shocked'},      # Should map to 'surprise'
                    {'frame': 6, 'expression': 'calm'},         # Should map to 'neutral'
                    {'frame': 8, 'expression': 'confused'},     # Should map to 'unknown'
                ],
                'gestures': []
            }
        }
        
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(analysis_data)
        
        emotions = markers['first_5_seconds']['emotion_sequence']
        assert emotions[0] == 'happy'    # happiness -> happy
        assert emotions[1] == 'happy'    # joy -> happy
        assert emotions[2] == 'surprise' # shocked -> surprise
        assert emotions[3] == 'neutral'  # calm -> neutral
        assert emotions[4] == 'unknown'  # confused -> unknown
    
    def test_gesture_standardization(self, video_metadata):
        """Test gesture vocabulary standardization"""
        analysis_data = {
            'timeline': {
                'expressions': [],
                'gestures': [
                    {'frame': 2, 'gesture': 'pointing_up', 'confidence': 0.95},      # -> pointing
                    {'frame': 4, 'gesture': 'thumbs_up', 'confidence': 0.90},        # -> approval
                    {'frame': 6, 'gesture': 'waving', 'confidence': 0.92},           # -> wave
                    {'frame': 8, 'gesture': 'random_gesture', 'confidence': 0.88},   # -> unknown
                ]
            }
        }
        
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(analysis_data)
        
        gestures = markers['first_5_seconds']['gesture_moments']
        assert gestures[0]['gesture'] == 'pointing'
        assert gestures[1]['gesture'] == 'approval'
        assert gestures[2]['gesture'] == 'wave'
        assert gestures[3]['gesture'] == 'unknown'
    
    def test_gesture_target_preservation(self, video_metadata):
        """Test that gesture targets are preserved"""
        analysis_data = {
            'timeline': {
                'expressions': [],
                'gestures': [
                    {'frame': 2, 'gesture': 'pointing', 'confidence': 0.95, 'target': 'product'},
                    {'frame': 4, 'gesture': 'pointing', 'confidence': 0.90, 'target': 'text'},
                ]
            }
        }
        
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(analysis_data)
        
        gestures = markers['first_5_seconds']['gesture_moments']
        assert gestures[0]['target'] == 'product'
        assert gestures[1]['target'] == 'text'
    
    def test_cta_gesture_filtering(self, video_metadata):
        """Test that only CTA-relevant gestures appear in CTA window"""
        analysis_data = {
            'timeline': {
                'expressions': [],
                'gestures': [
                    {'frame': 102, 'gesture': 'pointing', 'confidence': 0.95},     # Should appear
                    {'frame': 104, 'gesture': 'approval', 'confidence': 0.90},     # Should appear
                    {'frame': 106, 'gesture': 'wave', 'confidence': 0.92},         # Should appear
                    {'frame': 108, 'gesture': 'unknown', 'confidence': 0.88},      # Should NOT appear
                    {'frame': 110, 'gesture': 'clapping', 'confidence': 0.94},     # Should appear
                ]
            }
        }
        
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(analysis_data)
        
        cta_gestures = markers['cta_window']['gesture_sync']
        cta_gesture_types = [g['gesture'] for g in cta_gestures]
        
        # Should have 4 gestures (excluding 'unknown')
        assert len(cta_gestures) == 4
        assert 'unknown' not in cta_gesture_types
        
        # All non-unknown gestures should be included
        expected_gestures = {'pointing', 'approval', 'wave', 'clap'}
        assert set(cta_gesture_types) == expected_gestures
    
    def test_empty_timeline(self, video_metadata):
        """Test handling of empty timeline data"""
        analysis_data = {
            'timeline': {
                'expressions': [],
                'gestures': []
            }
        }
        
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(analysis_data)
        
        # Should have default neutral emotions
        assert markers['first_5_seconds']['emotion_sequence'] == ['neutral'] * 5
        assert markers['first_5_seconds']['gesture_moments'] == []
        assert markers['cta_window']['gesture_sync'] == []
    
    def test_gesture_limits(self, video_metadata):
        """Test that gesture limits are enforced"""
        # Create many gestures
        gestures = []
        for i in range(20):
            gestures.append({
                'frame': i,  # All in first 10 seconds (at 2fps extraction)
                'gesture': 'pointing',
                'confidence': 0.9
            })
        
        analysis_data = {
            'timeline': {
                'expressions': [],
                'gestures': gestures
            }
        }
        
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(analysis_data)
        
        # Should be limited to 8 gestures in first 5 seconds
        assert len(markers['first_5_seconds']['gesture_moments']) <= 8
        
        # CTA gestures should be limited to 5
        assert len(markers['cta_window']['gesture_sync']) <= 5
    
    def test_cta_alignment_flag(self, video_metadata):
        """Test that CTA gestures have alignment flag"""
        analysis_data = {
            'timeline': {
                'expressions': [],
                'gestures': [
                    {'frame': 102, 'gesture': 'pointing', 'confidence': 0.95},
                    {'frame': 110, 'gesture': 'approval', 'confidence': 0.90},
                ]
            }
        }
        
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(analysis_data)
        
        cta_gestures = markers['cta_window']['gesture_sync']
        assert all(g['aligns_with_cta'] is True for g in cta_gestures)
    
    def test_sparse_emotions(self, video_metadata):
        """Test handling of sparse emotion data"""
        analysis_data = {
            'timeline': {
                'expressions': [
                    {'frame': 0, 'expression': 'happy'},    # 0s
                    {'frame': 6, 'expression': 'surprise'}, # 3s
                    # Missing data for seconds 1, 2, 4
                ],
                'gestures': []
            }
        }
        
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(analysis_data)
        
        emotions = markers['first_5_seconds']['emotion_sequence']
        # Should fill missing with previous or neutral
        assert emotions[0] == 'happy'
        assert emotions[1] == 'neutral'  # No data for second 1
        assert emotions[2] == 'neutral'  # No data for second 2
        assert emotions[3] == 'surprise'
        assert emotions[4] == 'neutral'  # No data for second 4
    
    def test_confidence_preservation(self, video_metadata):
        """Test that confidence scores are preserved"""
        analysis_data = {
            'timeline': {
                'expressions': [],
                'gestures': [
                    {'frame': 2, 'gesture': 'pointing', 'confidence': 0.99},
                    {'frame': 4, 'gesture': 'wave', 'confidence': 0.85},
                ]
            }
        }
        
        extractor = MediaPipeTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(analysis_data)
        
        gestures = markers['first_5_seconds']['gesture_moments']
        assert gestures[0]['confidence'] == 0.99
        assert gestures[1]['confidence'] == 0.85