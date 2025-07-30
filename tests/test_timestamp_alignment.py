"""
Critical timestamp alignment tests
Ensures events appear at the same time across all analyzers
This is the most critical test - without proper alignment, patterns are meaningless
"""

import pytest
from python.timestamp_normalizer import TimestampNormalizer


class TestTimestampAlignment:
    """Test timestamp alignment across different analyzers"""
    
    @pytest.fixture
    def video_metadata(self):
        """Standard test video metadata - 30fps, 60 second video"""
        return {
            'fps': 30.0,
            'extraction_fps': 2.0,  # Frames extracted at 2 FPS
            'frame_count': 1800,    # 60s * 30fps
            'duration': 60.0
        }
    
    @pytest.fixture
    def normalizer(self, video_metadata):
        return TimestampNormalizer(video_metadata)
    
    def test_critical_event_alignment(self, normalizer):
        """
        CRITICAL TEST: Same event must appear at same time across analyzers
        Simulates a text appearing at exactly 3.0 seconds
        """
        # Event at 3.0 seconds
        expected_time = 3.0
        
        # OCR sees it in frame filename
        ocr_filename = "frame_0006_t3.00.jpg"  # 6th extracted frame at 2fps = 3s
        ocr_time = normalizer.normalize_to_seconds(ocr_filename, 'frame_filename')
        
        # YOLO sees it as frame 90 (3s * 30fps)
        yolo_frame = 90
        yolo_time = normalizer.normalize_to_seconds(yolo_frame, 'frame_index')
        
        # MediaPipe sees it as extracted frame 6 (3s * 2fps)
        mediapipe_frame = 6
        mediapipe_time = normalizer.normalize_to_seconds(mediapipe_frame, 'extracted_frame_index')
        
        # UnifiedTimeline sees it as range
        timeline_range = "3-4s"
        timeline_time = normalizer.normalize_to_seconds(timeline_range, 'timeline_string')
        
        # All should be exactly 3.0 seconds
        assert ocr_time == 3.0, f"OCR time {ocr_time} != 3.0"
        assert yolo_time == 3.0, f"YOLO time {yolo_time} != 3.0"
        assert mediapipe_time == 3.0, f"MediaPipe time {mediapipe_time} != 3.0"
        assert timeline_time == 3.0, f"Timeline time {timeline_time} != 3.0"
        
        # All should be within 0.1s of each other (allowing for float precision)
        times = [ocr_time, yolo_time, mediapipe_time, timeline_time]
        for i, time1 in enumerate(times):
            for j, time2 in enumerate(times):
                assert abs(time1 - time2) < 0.1, \
                    f"Times not aligned: {time1} vs {time2} (diff: {abs(time1 - time2)})"
    
    def test_multiple_event_alignment(self, normalizer):
        """Test alignment of multiple events throughout video"""
        test_events = [
            # (seconds, ocr_filename, yolo_frame, mediapipe_extracted_frame, timeline_string)
            (0.0, "frame_0000_t0.00.jpg", 0, 0, "0-1s"),
            (1.5, "frame_0003_t1.50.jpg", 45, 3, "1-2s"),
            (5.0, "frame_0010_t5.00.jpg", 150, 10, "5-6s"),
            (10.0, "frame_0020_t10.00.jpg", 300, 20, "10-11s"),
            (30.0, "frame_0060_t30.00.jpg", 900, 60, "30-31s"),
            (59.5, "frame_0119_t59.50.jpg", 1785, 119, "59-60s")
        ]
        
        for expected, ocr_file, yolo_frame, mp_frame, timeline in test_events:
            ocr_time = normalizer.normalize_to_seconds(ocr_file, 'frame_filename')
            yolo_time = normalizer.normalize_to_seconds(yolo_frame, 'frame_index')
            mp_time = normalizer.normalize_to_seconds(mp_frame, 'extracted_frame_index')
            tl_time = normalizer.normalize_to_seconds(timeline, 'timeline_string')
            
            # OCR, YOLO, and MediaPipe should be exact
            assert abs(ocr_time - expected) < 0.01, \
                f"OCR time {ocr_time} != expected {expected}"
            assert abs(yolo_time - expected) < 0.01, \
                f"YOLO time {yolo_time} != expected {expected}"
            assert abs(mp_time - expected) < 0.01, \
                f"MediaPipe time {mp_time} != expected {expected}"
            
            # Timeline is less precise (only gives start of range)
            assert abs(tl_time - int(expected)) < 0.01, \
                f"Timeline time {tl_time} not aligned with {int(expected)}"
    
    def test_extraction_fps_mismatch(self):
        """Test alignment when extraction FPS doesn't match expected"""
        # Video extracted at 5 FPS instead of expected 2 FPS
        metadata_5fps = {
            'fps': 30.0,
            'extraction_fps': 5.0,  # Different extraction rate
            'frame_count': 1800,
            'duration': 60.0
        }
        normalizer_5fps = TimestampNormalizer(metadata_5fps)
        
        # Frame 15 at 5fps extraction = 3.0 seconds
        time_5fps = normalizer_5fps.normalize_to_seconds(15, 'extracted_frame_index')
        assert time_5fps == 3.0
        
        # Should still align with other formats
        yolo_time = normalizer_5fps.normalize_to_seconds(90, 'frame_index')  # 90/30 = 3.0
        assert abs(time_5fps - yolo_time) < 0.01
    
    def test_different_video_fps_alignment(self):
        """Test alignment with different source video FPS"""
        # 24 FPS video (film)
        metadata_24fps = {
            'fps': 24.0,
            'extraction_fps': 2.0,
            'frame_count': 1440,  # 60s * 24fps
            'duration': 60.0
        }
        normalizer_24 = TimestampNormalizer(metadata_24fps)
        
        # 60 FPS video (high frame rate)
        metadata_60fps = {
            'fps': 60.0,
            'extraction_fps': 2.0,
            'frame_count': 3600,  # 60s * 60fps
            'duration': 60.0
        }
        normalizer_60 = TimestampNormalizer(metadata_60fps)
        
        # Same extracted frame should give same time
        extracted_frame = 6  # 3 seconds at 2fps
        time_24 = normalizer_24.normalize_to_seconds(extracted_frame, 'extracted_frame_index')
        time_60 = normalizer_60.normalize_to_seconds(extracted_frame, 'extracted_frame_index')
        
        assert time_24 == 3.0
        assert time_60 == 3.0
        
        # But original frame numbers differ
        frame_24 = normalizer_24.normalize_to_seconds(72, 'frame_index')  # 72/24 = 3.0
        frame_60 = normalizer_60.normalize_to_seconds(180, 'frame_index')  # 180/60 = 3.0
        
        assert frame_24 == 3.0
        assert frame_60 == 3.0
    
    def test_floating_point_precision(self, normalizer):
        """Test that floating point errors don't break alignment"""
        # Common problematic values
        test_times = [
            1/3,    # 0.333...
            1/6,    # 0.166...
            1/30,   # One frame at 30fps
            10/3,   # 3.333...
        ]
        
        for expected_time in test_times:
            # Convert to frame number and back
            frame_num = int(expected_time * 30)
            reconstructed_time = normalizer.normalize_to_seconds(frame_num, 'frame_index')
            
            # Should be very close (within one frame duration)
            assert abs(reconstructed_time - expected_time) < (1/30), \
                f"Floating point error too large: {reconstructed_time} vs {expected_time}"
    
    def test_boundary_conditions(self, normalizer):
        """Test alignment at video boundaries"""
        # First frame
        assert normalizer.normalize_to_seconds("frame_0000_t0.00.jpg", 'frame_filename') == 0.0
        assert normalizer.normalize_to_seconds(0, 'frame_index') == 0.0
        assert normalizer.normalize_to_seconds(0, 'extracted_frame_index') == 0.0
        assert normalizer.normalize_to_seconds("0-1s", 'timeline_string') == 0.0
        
        # Last frame (60 seconds)
        assert normalizer.normalize_to_seconds("frame_0120_t60.00.jpg", 'frame_filename') == 60.0
        assert normalizer.normalize_to_seconds(1800, 'frame_index') == 60.0
        assert normalizer.normalize_to_seconds(120, 'extracted_frame_index') == 60.0
        assert normalizer.normalize_to_seconds("60-61s", 'timeline_string') == 60.0
    
    def test_cross_analyzer_data_structure(self, normalizer):
        """Test alignment when processing actual analyzer outputs"""
        # Simulate actual data structures from different analyzers
        
        # OCR output
        ocr_output = {
            'frame_details': [
                {'frame': 'frame_0006_t3.00.jpg', 'text_elements': [{'text': 'HELLO'}]},
                {'frame': 'frame_0010_t5.00.jpg', 'text_elements': [{'text': 'WORLD'}]}
            ]
        }
        
        # YOLO output
        yolo_output = {
            'tracks': [
                {'detections': [
                    {'frame': 90, 'class': 'person'},  # Frame 90 = 3s
                    {'frame': 150, 'class': 'person'}  # Frame 150 = 5s
                ]}
            ]
        }
        
        # MediaPipe output (using extracted frames)
        mediapipe_output = {
            'timeline': {
                'expressions': [
                    {'frame': 6, 'expression': 'happy'},   # Frame 6 at 2fps = 3s
                    {'frame': 10, 'expression': 'surprise'} # Frame 10 at 2fps = 5s
                ]
            }
        }
        
        # Extract and normalize timestamps
        ocr_times = []
        for frame_detail in ocr_output['frame_details']:
            time = normalizer.normalize_to_seconds(frame_detail['frame'], 'frame_filename')
            ocr_times.append(time)
        
        yolo_times = []
        for track in yolo_output['tracks']:
            for detection in track['detections']:
                time = normalizer.normalize_to_seconds(detection['frame'], 'frame_index')
                yolo_times.append(time)
        
        mp_times = []
        for expression in mediapipe_output['timeline']['expressions']:
            time = normalizer.normalize_to_seconds(expression['frame'], 'extracted_frame_index')
            mp_times.append(time)
        
        # All should have events at 3.0 and 5.0 seconds
        assert ocr_times == [3.0, 5.0]
        assert yolo_times == [3.0, 5.0]
        assert mp_times == [3.0, 5.0]
    
    def test_error_handling_maintains_alignment(self, normalizer):
        """Test that errors don't break alignment for valid data"""
        # Mix of valid and invalid data
        test_data = [
            ("frame_0006_t3.00.jpg", 'frame_filename', 3.0),
            ("invalid_filename.jpg", 'frame_filename', None),
            (90, 'frame_index', 3.0),
            ("not_a_number", 'frame_index', None),
            (6, 'extracted_frame_index', 3.0),
        ]
        
        results = []
        for value, source_type, expected in test_data:
            result = normalizer.normalize_to_seconds(value, source_type)
            results.append(result)
            
            if expected is not None:
                assert result == expected, f"Expected {expected}, got {result}"
            else:
                assert result is None, f"Expected None for invalid input, got {result}"
        
        # Valid results should still align
        valid_results = [r for r in results if r is not None]
        assert all(r == 3.0 for r in valid_results)