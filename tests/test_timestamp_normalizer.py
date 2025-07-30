"""
Comprehensive tests for TimestampNormalizer
Ensures all timestamp formats are correctly normalized
"""

import pytest
import time
from python.timestamp_normalizer import TimestampNormalizer, create_from_video_path


class TestTimestampNormalizer:
    """Test suite for timestamp normalization"""
    
    @pytest.fixture
    def video_metadata(self):
        """Standard test video metadata"""
        return {
            'fps': 30.0,
            'extraction_fps': 2.0,
            'frame_count': 1800,  # 60 seconds at 30fps
            'duration': 60.0
        }
    
    @pytest.fixture
    def normalizer(self, video_metadata):
        """Create normalizer with test metadata"""
        return TimestampNormalizer(video_metadata)
    
    def test_initialization(self, video_metadata):
        """Test normalizer initialization"""
        normalizer = TimestampNormalizer(video_metadata)
        assert normalizer.fps == 30.0
        assert normalizer.extraction_fps == 2.0
        assert normalizer.frame_count == 1800
        assert normalizer.duration == 60.0
    
    def test_invalid_metadata(self):
        """Test initialization with invalid metadata"""
        # Invalid FPS
        with pytest.raises(ValueError, match="Invalid FPS"):
            TimestampNormalizer({'fps': 0})
        
        # Invalid extraction FPS
        with pytest.raises(ValueError, match="Invalid extraction FPS"):
            TimestampNormalizer({'extraction_fps': -1})
    
    def test_frame_filename_parsing(self, normalizer):
        """Test parsing frame filenames"""
        # Standard format
        assert normalizer.normalize_to_seconds('frame_0015_t0.50.jpg', 'frame_filename') == 0.5
        assert normalizer.normalize_to_seconds('frame_0150_t5.00.jpg', 'frame_filename') == 5.0
        assert normalizer.normalize_to_seconds('frame_0001_t0.03.jpg', 'frame_filename') == 0.03
        
        # With path
        assert normalizer.normalize_to_seconds('/path/to/frame_0030_t1.00.jpg', 'frame_filename') == 1.0
        
        # Decimal timestamps
        assert normalizer.normalize_to_seconds('frame_0045_t1.50.jpg', 'frame_filename') == 1.5
        assert normalizer.normalize_to_seconds('frame_0123_t4.10.jpg', 'frame_filename') == 4.1
        
        # Fallback to frame number (extracted frames)
        assert normalizer.normalize_to_seconds('frame_0010.jpg', 'frame_filename') == 5.0  # 10th frame at 2fps = 5s
        
        # Malformed filenames
        assert normalizer.normalize_to_seconds('not_a_frame.jpg', 'frame_filename') is None
        assert normalizer.normalize_to_seconds('frame_abc_t1.0.jpg', 'frame_filename') == 1.0
    
    def test_frame_index_conversion(self, normalizer):
        """Test frame index to seconds conversion"""
        # At 30 FPS
        assert normalizer.normalize_to_seconds(0, 'frame_index') == 0.0
        assert normalizer.normalize_to_seconds(30, 'frame_index') == 1.0
        assert normalizer.normalize_to_seconds(15, 'frame_index') == 0.5
        assert normalizer.normalize_to_seconds(450, 'frame_index') == 15.0
        assert normalizer.normalize_to_seconds(1800, 'frame_index') == 60.0
        
        # String input
        assert normalizer.normalize_to_seconds('30', 'frame_index') == 1.0
    
    def test_extracted_frame_conversion(self, normalizer):
        """Test extracted frame index conversion"""
        # At 2 FPS extraction rate
        assert normalizer.normalize_to_seconds(0, 'extracted_frame_index') == 0.0
        assert normalizer.normalize_to_seconds(2, 'extracted_frame_index') == 1.0
        assert normalizer.normalize_to_seconds(10, 'extracted_frame_index') == 5.0
        assert normalizer.normalize_to_seconds(120, 'extracted_frame_index') == 60.0
        
        # String input
        assert normalizer.normalize_to_seconds('10', 'extracted_frame_index') == 5.0
    
    def test_timeline_string_parsing(self, normalizer):
        """Test timeline string format parsing"""
        # Standard format
        assert normalizer.normalize_to_seconds('0-1s', 'timeline_string') == 0.0
        assert normalizer.normalize_to_seconds('5-6s', 'timeline_string') == 5.0
        assert normalizer.normalize_to_seconds('15-16s', 'timeline_string') == 15.0
        
        # Without 's' suffix
        assert normalizer.normalize_to_seconds('10-11', 'timeline_string') == 10.0
        
        # Decimal values
        assert normalizer.normalize_to_seconds('1.5-2.5s', 'timeline_string') == 1.5
        assert normalizer.normalize_to_seconds('10.25-10.75s', 'timeline_string') == 10.25
        
        # Malformed strings
        assert normalizer.normalize_to_seconds('invalid', 'timeline_string') is None
        assert normalizer.normalize_to_seconds('', 'timeline_string') is None
    
    def test_float_seconds_passthrough(self, normalizer):
        """Test float seconds passthrough"""
        assert normalizer.normalize_to_seconds(0.0, 'float_seconds') == 0.0
        assert normalizer.normalize_to_seconds(1.5, 'float_seconds') == 1.5
        assert normalizer.normalize_to_seconds(30.0, 'float_seconds') == 30.0
        assert normalizer.normalize_to_seconds('5.5', 'float_seconds') == 5.5
    
    def test_unknown_source_type(self, normalizer):
        """Test handling of unknown source types"""
        with pytest.raises(ValueError, match="Unknown source type"):
            normalizer.normalize_to_seconds(1.0, 'unknown_type')
    
    def test_timestamp_validation(self, normalizer):
        """Test timestamp validation"""
        # Valid timestamps
        assert normalizer.validate_timestamp(0.0) is True
        assert normalizer.validate_timestamp(30.0) is True
        assert normalizer.validate_timestamp(60.0) is True
        
        # Small overshoot allowed (rounding errors)
        assert normalizer.validate_timestamp(60.05) is True
        
        # Invalid timestamps
        assert normalizer.validate_timestamp(-1.0) is False
        assert normalizer.validate_timestamp(61.0) is False
        assert normalizer.validate_timestamp(None) is False
        
        # Small negative allowed (rounding errors)
        assert normalizer.validate_timestamp(-0.05) is True
        assert normalizer.validate_timestamp(-0.2) is False
    
    def test_timeline_range_formatting(self, normalizer):
        """Test timeline range formatting"""
        assert normalizer.get_timeline_range(0.0, 1.0) == "0.0-1.0s"
        assert normalizer.get_timeline_range(5.5, 6.5) == "5.5-6.5s"
        assert normalizer.get_timeline_range(10.123, 11.456) == "10.1-11.5s"
    
    def test_batch_normalize(self, normalizer):
        """Test batch normalization"""
        # Frame filenames
        filenames = [
            'frame_0000_t0.00.jpg',
            'frame_0030_t1.00.jpg',
            'frame_0060_t2.00.jpg'
        ]
        results = normalizer.batch_normalize(filenames, 'frame_filename')
        assert results == [0.0, 1.0, 2.0]
        
        # Frame indices
        indices = [0, 15, 30, 45, 60]
        results = normalizer.batch_normalize(indices, 'frame_index')
        assert results == [0.0, 0.5, 1.0, 1.5, 2.0]
        
        # Mixed valid/invalid
        mixed = ['0-1s', 'invalid', '5-6s', None, '10-11s']
        results = normalizer.batch_normalize(mixed, 'timeline_string')
        assert results == [0.0, None, 5.0, None, 10.0]
    
    def test_performance(self, normalizer):
        """Test performance requirement: 1000 timestamps in <100ms"""
        # Generate 1000 test timestamps
        timestamps = [f'frame_{i:04d}_t{i/30:.2f}.jpg' for i in range(1000)]
        
        start_time = time.time()
        results = normalizer.batch_normalize(timestamps, 'frame_filename')
        elapsed = time.time() - start_time
        
        assert len(results) == 1000
        assert elapsed < 0.1, f"Normalization took {elapsed:.3f}s, should be <0.1s"
    
    def test_edge_cases(self, normalizer):
        """Test edge cases and boundary conditions"""
        # Empty/None inputs
        assert normalizer.normalize_to_seconds('', 'frame_filename') is None
        assert normalizer.normalize_to_seconds(None, 'float_seconds') is None
        
        # Very large frame numbers
        assert normalizer.normalize_to_seconds(1_000_000, 'frame_index') == 1_000_000 / 30.0
        
        # Negative frame numbers (should work mathematically)
        assert normalizer.normalize_to_seconds(-30, 'frame_index') == -1.0
        
        # Zero/negative FPS protection
        zero_fps_normalizer = TimestampNormalizer({'fps': 0.001, 'extraction_fps': 0.001})
        assert zero_fps_normalizer.normalize_to_seconds(1, 'frame_index') == 1000.0
    
    def test_different_fps_values(self):
        """Test with various FPS values"""
        # 24 FPS (film)
        normalizer_24 = TimestampNormalizer({'fps': 24.0, 'extraction_fps': 2.0})
        assert normalizer_24.normalize_to_seconds(24, 'frame_index') == 1.0
        assert normalizer_24.normalize_to_seconds(12, 'frame_index') == 0.5
        
        # 60 FPS (high frame rate)
        normalizer_60 = TimestampNormalizer({'fps': 60.0, 'extraction_fps': 5.0})
        assert normalizer_60.normalize_to_seconds(60, 'frame_index') == 1.0
        assert normalizer_60.normalize_to_seconds(30, 'frame_index') == 0.5
        
        # 29.97 FPS (NTSC)
        normalizer_ntsc = TimestampNormalizer({'fps': 29.97, 'extraction_fps': 2.0})
        result = normalizer_ntsc.normalize_to_seconds(30, 'frame_index')
        assert abs(result - 1.001) < 0.001  # Allow small floating point error
    
    def test_consistency_across_formats(self, normalizer):
        """Test that same timestamp gives same result across formats"""
        # 1.5 seconds should be consistent
        assert normalizer.normalize_to_seconds('frame_0045_t1.50.jpg', 'frame_filename') == 1.5
        assert normalizer.normalize_to_seconds(45, 'frame_index') == 1.5
        assert normalizer.normalize_to_seconds(3, 'extracted_frame_index') == 1.5  # 3 frames at 2fps
        assert normalizer.normalize_to_seconds('1.5-2.5s', 'timeline_string') == 1.5
        assert normalizer.normalize_to_seconds(1.5, 'float_seconds') == 1.5


class TestCreateFromVideoPath:
    """Test video metadata extraction"""
    
    def test_create_from_invalid_path(self):
        """Test creation with invalid video path"""
        normalizer = create_from_video_path('/invalid/path/video.mp4')
        assert normalizer is None
    
    # Note: Actual video file tests would require test video files
    # Skipping for now as they depend on cv2 and actual video files