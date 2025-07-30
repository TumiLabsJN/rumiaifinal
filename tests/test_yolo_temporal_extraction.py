"""
Test YOLO Temporal Marker Extraction
Verifies that the YOLO analyzer correctly extracts temporal markers
"""

import pytest
from python.temporal_marker_extractors import YOLOTemporalExtractor


class TestYOLOTemporalExtraction:
    """Test suite for YOLO temporal marker extraction"""
    
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
    def sample_tracking_data(self):
        """Sample YOLO tracking data"""
        return {
            'tracks': [
                {
                    'track_id': 1,
                    'detections': [
                        {'frame': 0, 'class': 'person', 'confidence': 0.99},
                        {'frame': 30, 'class': 'person', 'confidence': 0.98},  # 1 second
                        {'frame': 60, 'class': 'person', 'confidence': 0.97},  # 2 seconds
                        {'frame': 90, 'class': 'person', 'confidence': 0.96},  # 3 seconds
                        {'frame': 120, 'class': 'person', 'confidence': 0.95}, # 4 seconds
                        {'frame': 150, 'class': 'person', 'confidence': 0.94}, # 5 seconds
                        {'frame': 1530, 'class': 'person', 'confidence': 0.93}, # 51 seconds (CTA window)
                        {'frame': 1650, 'class': 'person', 'confidence': 0.92}, # 55 seconds
                    ]
                },
                {
                    'track_id': 2,
                    'detections': [
                        {'frame': 45, 'class': 'product', 'confidence': 0.90},  # 1.5 seconds
                        {'frame': 75, 'class': 'product', 'confidence': 0.91},  # 2.5 seconds
                        {'frame': 105, 'class': 'product', 'confidence': 0.92}, # 3.5 seconds
                        {'frame': 1680, 'class': 'product', 'confidence': 0.93}, # 56 seconds (CTA)
                    ]
                },
                {
                    'track_id': 3,
                    'detections': [
                        {'frame': 15, 'class': 'cat', 'confidence': 0.85},  # 0.5 seconds
                        {'frame': 45, 'class': 'cat', 'confidence': 0.86},  # 1.5 seconds
                    ]
                }
            ]
        }
    
    def test_basic_extraction(self, video_metadata, sample_tracking_data):
        """Test basic temporal marker extraction"""
        extractor = YOLOTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(sample_tracking_data)
        
        # Check structure
        assert 'first_5_seconds' in markers
        assert 'cta_window' in markers
        
        # Check first 5 seconds
        first_5 = markers['first_5_seconds']
        assert 'object_appearances' in first_5
        assert 'density_progression' in first_5
        
        # Check object appearances are sorted by time
        appearances = first_5['object_appearances']
        times = [app['time'] for app in appearances]
        assert times == sorted(times)
        
        # Check density progression
        assert len(first_5['density_progression']) == 5
        # At 0s: person + cat = 2 objects
        # At 1s: person = 1 object  
        # At 1.5s: product + cat = 2 objects (grouped into second 1)
        # At 2s: person + product = 2 objects
        # At 3s: person + product = 2 objects
        # At 4s: person = 1 object
        
        # Check CTA window
        cta = markers['cta_window']
        assert cta['time_range'] == '51.0-60.0s'
        assert len(cta['object_focus']) > 0
        
        # Should only have person and product in CTA (key objects)
        cta_objects = [obj['object'] for obj in cta['object_focus']]
        assert all(obj in ['person', 'product'] for obj in cta_objects)
    
    def test_object_grouping_by_time(self, video_metadata):
        """Test that objects at same timestamp are grouped"""
        tracking_data = {
            'tracks': [
                {
                    'track_id': 1,
                    'detections': [
                        {'frame': 30, 'class': 'person', 'confidence': 0.99},  # 1.0s
                    ]
                },
                {
                    'track_id': 2,
                    'detections': [
                        {'frame': 30, 'class': 'dog', 'confidence': 0.95},  # 1.0s
                    ]
                },
                {
                    'track_id': 3,
                    'detections': [
                        {'frame': 30, 'class': 'ball', 'confidence': 0.90},  # 1.0s
                    ]
                }
            ]
        }
        
        extractor = YOLOTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(tracking_data)
        
        appearances = markers['first_5_seconds']['object_appearances']
        # Should have one appearance with all 3 objects
        assert len(appearances) == 1
        assert appearances[0]['time'] == 1.0
        assert set(appearances[0]['objects']) == {'person', 'dog', 'ball'}
        assert len(appearances[0]['confidence']) == 3
    
    def test_cta_window_filtering(self, video_metadata):
        """Test that only key objects appear in CTA window"""
        tracking_data = {
            'tracks': [
                {
                    'track_id': 1,
                    'detections': [
                        {'frame': 1530, 'class': 'person', 'confidence': 0.99},    # 51s
                        {'frame': 1560, 'class': 'hand', 'confidence': 0.95},      # 52s
                        {'frame': 1590, 'class': 'product', 'confidence': 0.98},   # 53s
                        {'frame': 1620, 'class': 'cat', 'confidence': 0.90},       # 54s (should be filtered)
                        {'frame': 1650, 'class': 'chair', 'confidence': 0.85},     # 55s (should be filtered)
                    ]
                }
            ]
        }
        
        extractor = YOLOTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(tracking_data)
        
        cta_focus = markers['cta_window']['object_focus']
        cta_objects = [obj['object'] for obj in cta_focus]
        
        # Only person, hand, product should appear
        assert 'person' in cta_objects
        assert 'hand' in cta_objects
        assert 'product' in cta_objects
        assert 'cat' not in cta_objects
        assert 'chair' not in cta_objects
    
    def test_empty_tracking_data(self, video_metadata):
        """Test handling of empty tracking data"""
        tracking_data = {'tracks': []}
        
        extractor = YOLOTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(tracking_data)
        
        assert markers['first_5_seconds']['object_appearances'] == []
        assert markers['first_5_seconds']['density_progression'] == [0, 0, 0, 0, 0]
        assert markers['cta_window']['object_focus'] == []
    
    def test_density_calculation(self, video_metadata):
        """Test density progression calculation"""
        tracking_data = {
            'tracks': [
                {
                    'track_id': 1,
                    'detections': [
                        {'frame': 0, 'class': 'person', 'confidence': 0.99},    # 0s
                        {'frame': 15, 'class': 'person', 'confidence': 0.98},   # 0.5s
                        {'frame': 30, 'class': 'person', 'confidence': 0.97},   # 1s
                        {'frame': 60, 'class': 'person', 'confidence': 0.96},   # 2s
                    ]
                },
                {
                    'track_id': 2,
                    'detections': [
                        {'frame': 0, 'class': 'dog', 'confidence': 0.95},       # 0s
                        {'frame': 30, 'class': 'dog', 'confidence': 0.94},      # 1s
                        {'frame': 90, 'class': 'dog', 'confidence': 0.93},      # 3s
                    ]
                },
                {
                    'track_id': 3,
                    'detections': [
                        {'frame': 120, 'class': 'ball', 'confidence': 0.90},    # 4s
                    ]
                }
            ]
        }
        
        extractor = YOLOTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(tracking_data)
        
        density = markers['first_5_seconds']['density_progression']
        # Second 0: person + dog at frame 0, person at frame 15 = 3 total events
        # Second 1: person + dog = 2
        # Second 2: person = 1
        # Second 3: dog = 1
        # Second 4: ball = 1
        assert density[0] == 3  # Two objects at 0s, one at 0.5s
        assert density[1] == 2  # person + dog at 1s
        assert density[2] == 1  # person at 2s
        assert density[3] == 1  # dog at 3s
        assert density[4] == 1  # ball at 4s
    
    def test_object_limit_enforcement(self, video_metadata):
        """Test that object appearance limits are enforced"""
        # Create many object appearances
        tracks = []
        for i in range(20):
            tracks.append({
                'track_id': i,
                'detections': [
                    {'frame': i * 3, 'class': f'object_{i}', 'confidence': 0.9}  # Spread across first 2 seconds
                ]
            })
        
        tracking_data = {'tracks': tracks}
        
        extractor = YOLOTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(tracking_data)
        
        # Should be limited to 10 appearances
        assert len(markers['first_5_seconds']['object_appearances']) <= 10
        
        # CTA focus should be limited to 5
        assert len(markers['cta_window']['object_focus']) <= 5
    
    def test_confidence_preservation(self, video_metadata):
        """Test that confidence scores are preserved"""
        tracking_data = {
            'tracks': [
                {
                    'track_id': 1,
                    'detections': [
                        {'frame': 30, 'class': 'person', 'confidence': 0.99},
                        {'frame': 30, 'class': 'person', 'confidence': 0.95},  # Duplicate should not appear
                    ]
                },
                {
                    'track_id': 2,
                    'detections': [
                        {'frame': 30, 'class': 'dog', 'confidence': 0.87},
                    ]
                }
            ]
        }
        
        extractor = YOLOTemporalExtractor(video_metadata)
        markers = extractor.extract_temporal_markers(tracking_data)
        
        appearance = markers['first_5_seconds']['object_appearances'][0]
        assert 0.99 in appearance['confidence']
        assert 0.87 in appearance['confidence']
        assert len(appearance['objects']) == 2  # person and dog
        assert len(appearance['confidence']) == 2
    
    def test_cta_time_range_calculation(self, video_metadata):
        """Test CTA window calculation for different durations"""
        test_cases = [
            (30.0, '25.5-30.0s'),
            (60.0, '51.0-60.0s'),
            (120.0, '102.0-120.0s'),
        ]
        
        for duration, expected_range in test_cases:
            metadata = video_metadata.copy()
            metadata['duration'] = duration
            
            extractor = YOLOTemporalExtractor(metadata)
            markers = extractor.extract_temporal_markers({'tracks': []})
            
            assert markers['cta_window']['time_range'] == expected_range