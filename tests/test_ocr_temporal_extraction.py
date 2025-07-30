"""
Test OCR Temporal Marker Extraction
Verifies that the OCR analyzer correctly extracts temporal markers
"""

import pytest
import json
from python.temporal_marker_extractors import OCRTemporalExtractor


class TestOCRTemporalExtraction:
    """Test suite for OCR temporal marker extraction"""
    
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
    def sample_frame_results(self):
        """Sample OCR frame analysis results"""
        return [
            {
                'frame': 'frame_0000_t0.00.jpg',
                'text_elements': [
                    {
                        'text': 'WAIT FOR IT',
                        'confidence': 0.99,
                        'bbox': {'x1': 100, 'y1': 200, 'x2': 300, 'y2': 250},
                        'category': 'overlay_text'
                    }
                ]
            },
            {
                'frame': 'frame_0003_t1.50.jpg',
                'text_elements': [
                    {
                        'text': 'This is amazing',
                        'confidence': 0.95,
                        'bbox': {'x1': 150, 'y1': 400, 'x2': 350, 'y2': 430},
                        'category': 'overlay_text'
                    },
                    {
                        'text': 'Follow for more',
                        'confidence': 0.98,
                        'bbox': {'x1': 100, 'y1': 500, 'x2': 250, 'y2': 530},
                        'category': 'call_to_action'
                    }
                ]
            },
            {
                'frame': 'frame_0006_t3.00.jpg',
                'text_elements': [
                    {
                        'text': '#viral #fyp',
                        'confidence': 0.92,
                        'bbox': {'x1': 50, 'y1': 600, 'x2': 150, 'y2': 620},
                        'category': 'hashtag'
                    }
                ]
            },
            {
                'frame': 'frame_0010_t5.00.jpg',
                'text_elements': [
                    {
                        'text': 'Watch till the end',
                        'confidence': 0.97,
                        'bbox': {'x1': 200, 'y1': 300, 'x2': 400, 'y2': 340},
                        'category': 'overlay_text'
                    }
                ]
            },
            # CTA window frame (85% of 60s = 51s)
            {
                'frame': 'frame_0102_t51.00.jpg',
                'text_elements': [
                    {
                        'text': 'Link in bio',
                        'confidence': 0.99,
                        'bbox': {'x1': 150, 'y1': 350, 'x2': 250, 'y2': 380},
                        'category': 'call_to_action'
                    }
                ]
            },
            {
                'frame': 'frame_0110_t55.00.jpg',
                'text_elements': [
                    {
                        'text': 'Follow for Part 2',
                        'confidence': 0.98,
                        'bbox': {'x1': 100, 'y1': 300, 'x2': 300, 'y2': 350},
                        'category': 'call_to_action'
                    }
                ]
            }
        ]
    
    def test_basic_extraction(self, video_metadata, sample_frame_results):
        """Test basic temporal marker extraction"""
        extractor = OCRTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(sample_frame_results)
        
        # Check structure
        assert 'first_5_seconds' in markers
        assert 'cta_window' in markers
        
        # Check first 5 seconds
        first_5 = markers['first_5_seconds']
        assert 'text_moments' in first_5
        assert 'density_progression' in first_5
        
        # Should have 4 text moments in first 5 seconds
        assert len(first_5['text_moments']) == 4
        
        # Check text moments are sorted by time
        times = [tm['time'] for tm in first_5['text_moments']]
        assert times == sorted(times)
        
        # Check density progression
        assert len(first_5['density_progression']) == 5
        assert first_5['density_progression'][0] == 1  # One text at 0s
        assert first_5['density_progression'][1] == 2  # Two texts at 1.5s
        assert first_5['density_progression'][3] == 1  # One text at 3s
        
        # Check CTA window
        cta = markers['cta_window']
        assert cta['time_range'] == '51.0-60.0s'
        assert len(cta['cta_appearances']) == 2
    
    def test_text_truncation(self, video_metadata):
        """Test that long text is properly truncated"""
        long_text = "This is a very long text that exceeds the maximum allowed length and should be truncated properly with ellipsis"
        
        frame_results = [{
            'frame': 'frame_0000_t0.00.jpg',
            'text_elements': [{
                'text': long_text,
                'confidence': 0.99,
                'bbox': {'x1': 100, 'y1': 200, 'x2': 300, 'y2': 250},
                'category': 'overlay_text'
            }]
        }]
        
        extractor = OCRTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(frame_results)
        
        text_moment = markers['first_5_seconds']['text_moments'][0]
        assert len(text_moment['text']) == 50  # Max length including ellipsis
        assert text_moment['text'].endswith('...')
    
    def test_cta_detection_in_first_5(self, video_metadata):
        """Test CTA detection in first 5 seconds"""
        frame_results = [{
            'frame': 'frame_0003_t1.50.jpg',
            'text_elements': [{
                'text': 'Follow me now',
                'confidence': 0.98,
                'bbox': {'x1': 100, 'y1': 200, 'x2': 250, 'y2': 230},
                'category': 'call_to_action'
            }]
        }]
        
        extractor = OCRTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(frame_results)
        
        text_moment = markers['first_5_seconds']['text_moments'][0]
        assert text_moment['is_cta'] is True
        assert text_moment['time'] == 1.5
    
    def test_text_size_classification(self, video_metadata):
        """Test text size classification from bounding box"""
        frame_results = [
            {
                'frame': 'frame_0000_t0.00.jpg',
                'text_elements': [
                    {
                        'text': 'LARGE TEXT',
                        'confidence': 0.99,
                        'bbox': {'x1': 0, 'y1': 0, 'x2': 200, 'y2': 100},  # Area: 20,000
                        'category': 'overlay_text'
                    }
                ]
            },
            {
                'frame': 'frame_0002_t1.00.jpg',
                'text_elements': [
                    {
                        'text': 'Medium text',
                        'confidence': 0.95,
                        'bbox': {'x1': 10, 'y1': 10, 'x2': 60, 'y2': 50},  # Area: 2,000
                        'category': 'overlay_text'
                    }
                ]
            },
            {
                'frame': 'frame_0004_t2.00.jpg',
                'text_elements': [
                    {
                        'text': 'small',
                        'confidence': 0.90,
                        'bbox': {'x1': 0, 'y1': 0, 'x2': 30, 'y2': 20},  # Area: 600
                        'category': 'overlay_text'
                    }
                ]
            }
        ]
        
        extractor = OCRTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(frame_results)
        
        text_moments = markers['first_5_seconds']['text_moments']
        assert text_moments[0]['size'] == 'L'
        assert text_moments[1]['size'] == 'M'
        assert text_moments[2]['size'] == 'S'
    
    def test_empty_frames(self, video_metadata):
        """Test handling of frames with no text"""
        frame_results = [
            {
                'frame': 'frame_0000_t0.00.jpg',
                'text_elements': []
            },
            {
                'frame': 'frame_0002_t1.00.jpg',
                'text_elements': [
                    {
                        'text': 'Hello',
                        'confidence': 0.95,
                        'bbox': {'x1': 100, 'y1': 200, 'x2': 150, 'y2': 220},
                        'category': 'overlay_text'
                    }
                ]
            }
        ]
        
        extractor = OCRTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(frame_results)
        
        # Should only have one text moment
        assert len(markers['first_5_seconds']['text_moments']) == 1
        assert markers['first_5_seconds']['text_moments'][0]['time'] == 1.0
    
    def test_invalid_timestamps(self, video_metadata):
        """Test handling of invalid timestamps"""
        frame_results = [
            {
                'frame': 'invalid_frame_name.jpg',  # Invalid format
                'text_elements': [
                    {
                        'text': 'This should be ignored',
                        'confidence': 0.95,
                        'bbox': {'x1': 100, 'y1': 200, 'x2': 200, 'y2': 220},
                        'category': 'overlay_text'
                    }
                ]
            },
            {
                'frame': 'frame_0002_t1.00.jpg',  # Valid
                'text_elements': [
                    {
                        'text': 'This is valid',
                        'confidence': 0.95,
                        'bbox': {'x1': 100, 'y1': 200, 'x2': 200, 'y2': 220},
                        'category': 'overlay_text'
                    }
                ]
            }
        ]
        
        extractor = OCRTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(frame_results)
        
        # Should only process valid timestamp
        assert len(markers['first_5_seconds']['text_moments']) == 1
        assert markers['first_5_seconds']['text_moments'][0]['text'] == 'This is valid'
    
    def test_size_limits_applied(self, video_metadata):
        """Test that size limits are properly applied"""
        # Create many text elements
        frame_results = []
        for i in range(20):  # Create 20 text elements in first 5 seconds
            frame_results.append({
                'frame': f'frame_{i:04d}_t{i*0.25:.2f}.jpg',
                'text_elements': [
                    {
                        'text': f'Text element {i}',
                        'confidence': 0.95,
                        'bbox': {'x1': 100, 'y1': 200, 'x2': 200, 'y2': 220},
                        'category': 'overlay_text'
                    }
                ]
            })
        
        extractor = OCRTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(frame_results)
        
        # Should be limited to MAX_TEXT_EVENTS_FIRST_5S (10)
        assert len(markers['first_5_seconds']['text_moments']) <= 10
    
    def test_position_determination(self, video_metadata):
        """Test text position determination"""
        frame_results = [
            {
                'frame': 'frame_0000_t0.00.jpg',
                'text_elements': [
                    {
                        'text': 'Top text',
                        'confidence': 0.99,
                        'bbox': {'x1': 100, 'y1': 50, 'x2': 200, 'y2': 100},  # Top
                        'category': 'overlay_text'
                    },
                    {
                        'text': 'Center text',
                        'confidence': 0.99,
                        'bbox': {'x1': 100, 'y1': 400, 'x2': 200, 'y2': 450},  # Center
                        'category': 'overlay_text'
                    },
                    {
                        'text': 'Bottom text',
                        'confidence': 0.99,
                        'bbox': {'x1': 100, 'y1': 700, 'x2': 200, 'y2': 750},  # Bottom
                        'category': 'overlay_text'
                    }
                ]
            }
        ]
        
        extractor = OCRTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(frame_results)
        
        text_moments = markers['first_5_seconds']['text_moments']
        assert text_moments[0]['position'] == 'top'
        assert text_moments[1]['position'] == 'center'
        assert text_moments[2]['position'] == 'bottom'
    
    def test_cta_window_calculation(self, video_metadata):
        """Test CTA window is correctly calculated"""
        # Test with different video durations
        test_cases = [
            (30.0, '25.5-30.0s'),   # 85% of 30s = 25.5s
            (60.0, '51.0-60.0s'),   # 85% of 60s = 51s
            (120.0, '102.0-120.0s'), # 85% of 120s = 102s
        ]
        
        for duration, expected_range in test_cases:
            metadata = video_metadata.copy()
            metadata['duration'] = duration
            
            extractor = OCRTemporalExtractor(metadata)
            markers = extractor.extract_temporal_markers([])
            
            assert markers['cta_window']['time_range'] == expected_range