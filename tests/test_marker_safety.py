"""
Tests for Temporal Marker Safety Controls
Ensures size limits and content sanitization work correctly
"""

import pytest
import json
from python.temporal_marker_safety import TemporalMarkerSafety


class TestTemporalMarkerSafety:
    """Test suite for temporal marker safety controls"""
    
    def test_text_truncation(self):
        """Test text truncation with various inputs"""
        safety = TemporalMarkerSafety
        
        # Normal text
        assert safety.truncate_text("Hello world") == "Hello world"
        
        # Long text
        long_text = "This is a very long text that exceeds the maximum allowed length and should be truncated"
        result = safety.truncate_text(long_text)
        assert len(result) == 50
        assert result.endswith("...")
        assert result == "This is a very long text that exceeds the maxi..."
        
        # Exactly at limit
        exact_text = "a" * 50
        assert safety.truncate_text(exact_text) == exact_text
        
        # Just over limit
        over_text = "a" * 51
        result = safety.truncate_text(over_text)
        assert len(result) == 50
        assert result == "a" * 47 + "..."
        
        # Empty/None inputs
        assert safety.truncate_text("") == ""
        assert safety.truncate_text(None) == ""
        assert safety.truncate_text("   ") == ""
        
        # Whitespace handling
        assert safety.truncate_text("  Hello   World  ") == "Hello World"
        assert safety.truncate_text("Line1\n\nLine2\t\tLine3") == "Line1 Line2 Line3"
        
        # Non-string inputs
        assert safety.truncate_text(123) == "123"
        assert safety.truncate_text(True) == "True"
        assert safety.truncate_text(['list']) == "['list']"
    
    def test_gesture_standardization(self):
        """Test gesture vocabulary standardization"""
        safety = TemporalMarkerSafety
        
        # Pointing variations
        assert safety.standardize_gesture("pointing_up") == "pointing"
        assert safety.standardize_gesture("pointing_down") == "pointing"
        assert safety.standardize_gesture("finger_point") == "pointing"
        assert safety.standardize_gesture("POINTING_UP") == "pointing"
        assert safety.standardize_gesture("  point  ") == "pointing"
        
        # Wave variations
        assert safety.standardize_gesture("wave") == "wave"
        assert safety.standardize_gesture("hand_wave") == "wave"
        assert safety.standardize_gesture("waving") == "wave"
        
        # Approval gestures
        assert safety.standardize_gesture("thumbs_up") == "approval"
        assert safety.standardize_gesture("thumb_up") == "approval"
        assert safety.standardize_gesture("ok_sign") == "approval"
        
        # Unknown/invalid
        assert safety.standardize_gesture("random_gesture") == "unknown"
        assert safety.standardize_gesture("") == "unknown"
        assert safety.standardize_gesture(None) == "unknown"
        assert safety.standardize_gesture(123) == "unknown"
    
    def test_emotion_standardization(self):
        """Test emotion vocabulary standardization"""
        safety = TemporalMarkerSafety
        
        # Happy variations
        assert safety.standardize_emotion("happy") == "happy"
        assert safety.standardize_emotion("happiness") == "happy"
        assert safety.standardize_emotion("joy") == "happy"
        assert safety.standardize_emotion("SMILE") == "happy"
        
        # Surprise variations
        assert safety.standardize_emotion("surprise") == "surprise"
        assert safety.standardize_emotion("shocked") == "surprise"
        assert safety.standardize_emotion("amazed") == "surprise"
        
        # Neutral
        assert safety.standardize_emotion("neutral") == "neutral"
        assert safety.standardize_emotion("calm") == "neutral"
        
        # Unknown/invalid
        assert safety.standardize_emotion("confused") == "unknown"
        assert safety.standardize_emotion("") == "unknown"
        assert safety.standardize_emotion(None) == "unknown"
    
    def test_text_size_classification(self):
        """Test text size classification based on bounding box"""
        safety = TemporalMarkerSafety
        
        # Large text
        large_bbox = {'x1': 0, 'y1': 0, 'x2': 200, 'y2': 100}  # Area: 20,000
        assert safety.classify_text_size(large_bbox) == "L"
        
        # Medium text
        medium_bbox = {'x1': 10, 'y1': 10, 'x2': 60, 'y2': 50}  # Area: 2,000
        assert safety.classify_text_size(medium_bbox) == "M"
        
        # Small text
        small_bbox = {'x1': 0, 'y1': 0, 'x2': 30, 'y2': 20}  # Area: 600
        assert safety.classify_text_size(small_bbox) == "S"
        
        # Edge cases
        assert safety.classify_text_size(None) == "M"
        assert safety.classify_text_size({}) == "M"
        assert safety.classify_text_size({'x1': 0}) == "M"  # Missing fields
        assert safety.classify_text_size("not a dict") == "M"
    
    def test_size_reduction_text_events(self):
        """Test reduction of text events"""
        markers = {
            'first_5_seconds': {
                'text_moments': [
                    {'time': i * 0.5, 'text': f'Text {i}'} 
                    for i in range(20)  # 20 text events
                ]
            }
        }
        
        reduced = TemporalMarkerSafety._reduce_text_events(markers)
        assert len(reduced['first_5_seconds']['text_moments']) == TemporalMarkerSafety.MAX_TEXT_EVENTS_FIRST_5S
        assert reduced['first_5_seconds']['text_moments'][0]['text'] == 'Text 0'
        assert reduced['first_5_seconds']['text_moments'][9]['text'] == 'Text 9'
    
    def test_size_reduction_progressive(self):
        """Test progressive size reduction"""
        # Create oversized markers
        large_text = "Very long text that will contribute to size " * 10
        markers = {
            'first_5_seconds': {
                'text_moments': [
                    {'time': i * 0.1, 'text': large_text, 'confidence': 0.99, 'position': 'center'} 
                    for i in range(15)
                ],
                'gesture_moments': [
                    {'time': i * 0.2, 'gesture': 'pointing', 'confidence': 0.95, 'target': 'product'} 
                    for i in range(10)
                ],
                'emotion_sequence': ['happy'] * 5,
                'density_progression': [3, 5, 4, 7, 6]
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'cta_appearances': [
                    {'time': 52 + i, 'text': f'CTA {i}', 'type': 'overlay', 'confidence': 0.9} 
                    for i in range(10)
                ],
                'gesture_sync': [
                    {'time': 52 + i, 'gesture': 'pointing', 'aligns_with_cta': True} 
                    for i in range(5)
                ],
                'ui_emphasis': [
                    {'time': 55 + i, 'element': 'follow_button', 'action': 'highlight'} 
                    for i in range(3)
                ]
            }
        }
        
        # Check initial size
        initial_size_kb = len(json.dumps(markers)) / 1024
        assert initial_size_kb > TemporalMarkerSafety.MAX_MARKER_SIZE_KB
        
        # Apply reduction
        reduced = TemporalMarkerSafety.check_and_reduce_size(markers, target_kb=5.0)  # Very small target
        final_size_kb = len(json.dumps(reduced)) / 1024
        
        # Verify reductions were applied
        assert len(reduced['first_5_seconds']['text_moments']) <= 10
        assert len(reduced['cta_window']['cta_appearances']) <= 8
        
        # Verify optional fields were removed
        if reduced['first_5_seconds']['text_moments']:
            assert 'confidence' not in reduced['first_5_seconds']['text_moments'][0]
            assert 'position' not in reduced['first_5_seconds']['text_moments'][0]
        
        # Size should be significantly reduced
        assert final_size_kb < initial_size_kb
    
    def test_remove_optional_fields(self):
        """Test removal of optional fields"""
        markers = {
            'first_5_seconds': {
                'text_moments': [
                    {
                        'time': 1.0,
                        'text': 'Hello',
                        'confidence': 0.99,  # Optional
                        'position': 'center',  # Optional
                        'bbox': {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 50}  # Optional
                    }
                ],
                'gesture_moments': [
                    {
                        'time': 2.0,
                        'gesture': 'pointing',
                        'intensity': 0.8,  # Optional
                        'target': 'product'  # Optional
                    }
                ]
            }
        }
        
        reduced = TemporalMarkerSafety._remove_optional_fields(markers)
        
        # Check required fields remain
        assert reduced['first_5_seconds']['text_moments'][0]['time'] == 1.0
        assert reduced['first_5_seconds']['text_moments'][0]['text'] == 'Hello'
        
        # Check optional fields removed
        assert 'confidence' not in reduced['first_5_seconds']['text_moments'][0]
        assert 'position' not in reduced['first_5_seconds']['text_moments'][0]
        assert 'bbox' not in reduced['first_5_seconds']['text_moments'][0]
        assert 'intensity' not in reduced['first_5_seconds']['gesture_moments'][0]
        assert 'target' not in reduced['first_5_seconds']['gesture_moments'][0]
    
    def test_aggressive_reduction(self):
        """Test aggressive reduction as last resort"""
        markers = {
            'first_5_seconds': {
                'text_moments': [{'time': i * 0.5, 'text': f'Text {i}'} for i in range(10)],
                'gesture_moments': [{'time': i * 0.5, 'gesture': 'wave'} for i in range(8)],
                'object_appearances': [{'time': i, 'objects': ['person']} for i in range(10)]
            },
            'cta_window': {
                'cta_appearances': [{'time': 52 + i, 'text': f'CTA {i}'} for i in range(8)],
                'gesture_sync': [{'time': 52 + i, 'gesture': 'pointing'} for i in range(5)],
                'ui_emphasis': [{'time': 55 + i, 'element': 'button'} for i in range(3)]
            }
        }
        
        reduced = TemporalMarkerSafety._aggressive_reduction(markers)
        
        # Check aggressive limits
        assert len(reduced['first_5_seconds']['text_moments']) <= 5
        assert len(reduced['first_5_seconds']['gesture_moments']) <= 3
        assert len(reduced['first_5_seconds']['object_appearances']) <= 5
        assert len(reduced['cta_window']['cta_appearances']) <= 3
        
        # Check non-critical data removed
        assert 'gesture_sync' not in reduced['cta_window']
        assert 'ui_emphasis' not in reduced['cta_window']
    
    def test_validate_markers(self):
        """Test marker validation"""
        # Valid markers
        valid_markers = {
            'first_5_seconds': {
                'text_moments': [
                    {'time': 1.0, 'text': 'Hello'},
                    {'time': 3.5, 'text': 'World'}
                ],
                'emotion_sequence': ['neutral', 'happy', 'happy', 'surprise', 'joy']
            },
            'cta_window': {
                'time_range': '51.0-60.0s',
                'cta_appearances': [
                    {'time': 52.0, 'text': 'Follow me'}
                ]
            }
        }
        
        errors = TemporalMarkerSafety.validate_markers(valid_markers)
        assert len(errors) == 0
        
        # Invalid markers - wrong types
        invalid_markers = {
            'first_5_seconds': {
                'text_moments': [
                    {'time': 'not a number', 'text': 'Hello'},  # Invalid time
                    {'time': 6.0, 'text': 'Too late'}  # Time > 5s
                ],
                'emotion_sequence': 'not a list'  # Should be list
            }
        }
        
        errors = TemporalMarkerSafety.validate_markers(invalid_markers)
        assert len(errors) > 0
        assert any('valid time' in error for error in errors)
        assert any('exceeds 5 seconds' in error for error in errors)
        assert any('must be a list' in error for error in errors)
        
        # Missing required fields
        incomplete_markers = {
            'cta_window': {
                # Missing time_range
                'cta_appearances': [
                    {'text': 'No time field'}  # Missing time
                ]
            }
        }
        
        errors = TemporalMarkerSafety.validate_markers(incomplete_markers)
        assert any('missing time_range' in error for error in errors)
        assert any('missing valid time' in error for error in errors)
    
    def test_sanitize_for_json(self):
        """Test JSON sanitization"""
        safety = TemporalMarkerSafety
        
        # Basic types
        assert safety.sanitize_for_json("string") == "string"
        assert safety.sanitize_for_json(123) == 123
        assert safety.sanitize_for_json(123.45) == 123.45
        assert safety.sanitize_for_json(True) == True
        assert safety.sanitize_for_json(None) is None
        
        # Collections
        assert safety.sanitize_for_json([1, 2, 3]) == [1, 2, 3]
        assert safety.sanitize_for_json({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}
        
        # Nested structures
        nested = {
            'list': [1, 'two', {'three': 3}],
            'dict': {'nested': {'value': 123}}
        }
        assert safety.sanitize_for_json(nested) == nested
        
        # Numpy types (if available)
        try:
            import numpy as np
            assert safety.sanitize_for_json(np.int32(42)) == 42
            assert safety.sanitize_for_json(np.float64(3.14)) == 3.14
            assert safety.sanitize_for_json(np.array([1, 2, 3])) == [1, 2, 3]
        except ImportError:
            pass  # Numpy not required for tests
        
        # Non-serializable objects
        class CustomObject:
            def __str__(self):
                return "custom"
        
        assert safety.sanitize_for_json(CustomObject()) == "custom"
    
    def test_size_limit_constants(self):
        """Test that size limit constants are reasonable"""
        assert TemporalMarkerSafety.MAX_TEXT_LENGTH == 50
        assert TemporalMarkerSafety.MAX_TEXT_EVENTS_FIRST_5S == 10
        assert TemporalMarkerSafety.MAX_GESTURE_EVENTS_FIRST_5S == 8
        assert TemporalMarkerSafety.MAX_CTA_EVENTS == 8
        assert TemporalMarkerSafety.MAX_MARKER_SIZE_KB == 50
        assert TemporalMarkerSafety.HARD_PAYLOAD_LIMIT_KB == 180
        
        # Ensure hard limit leaves buffer
        assert TemporalMarkerSafety.HARD_PAYLOAD_LIMIT_KB < 200  # API limit is 200KB