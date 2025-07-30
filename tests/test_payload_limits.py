"""
Payload size limit tests
Ensures temporal markers don't exceed API limits even with worst-case data
"""

import pytest
import json
from python.temporal_marker_safety import TemporalMarkerSafety


class TestPayloadLimits:
    """Test payload size limits and reduction strategies"""
    
    def create_text_bomb_timeline(self, num_texts=50, text_length=200):
        """Create a timeline with excessive text to test size limits"""
        long_text = "This is a very long text overlay that might appear in a recipe or tutorial video. " * (text_length // 80)
        
        timeline = {
            'textOverlayTimeline': {
                f'{i}-{i+1}s': {
                    'texts': [long_text],
                    'confidence': 0.99,
                    'bbox': {'x1': 0, 'y1': 0, 'x2': 1000, 'y2': 500}
                } for i in range(num_texts)
            },
            'gestureTimeline': {
                f'{i}-{i+1}s': {
                    'gestures': ['pointing', 'waving', 'thumbs_up'],
                    'confidence_scores': [0.9, 0.85, 0.92]
                } for i in range(30)
            },
            'objectTimeline': {
                f'{i}-{i+1}s': {
                    'objects': {
                        'person': 1,
                        'product': 2,
                        'background_items': ['table', 'chair', 'lamp', 'window']
                    }
                } for i in range(60)
            }
        }
        
        return timeline
    
    def create_realistic_markers(self):
        """Create realistic temporal markers for testing"""
        return {
            'first_5_seconds': {
                'density_progression': [3, 5, 4, 7, 6],
                'text_moments': [
                    {'time': 0.5, 'text': 'WAIT FOR IT', 'size': 'L', 'position': 'center', 'confidence': 0.99},
                    {'time': 1.2, 'text': 'You won\'t believe this', 'size': 'M', 'position': 'bottom', 'confidence': 0.95},
                    {'time': 2.0, 'text': 'Amazing transformation', 'size': 'L', 'position': 'center', 'confidence': 0.98},
                    {'time': 3.5, 'text': 'Watch closely', 'size': 'M', 'position': 'top', 'confidence': 0.97},
                    {'time': 4.8, 'text': 'Here it comes...', 'size': 'S', 'position': 'bottom', 'confidence': 0.96}
                ],
                'emotion_sequence': ['neutral', 'neutral', 'surprise', 'joy', 'excitement'],
                'gesture_moments': [
                    {'time': 1.5, 'gesture': 'pointing', 'target': 'product', 'confidence': 0.92},
                    {'time': 3.2, 'gesture': 'hands_up', 'intensity': 0.8, 'confidence': 0.88},
                    {'time': 4.5, 'gesture': 'clapping', 'intensity': 0.9, 'confidence': 0.94}
                ],
                'object_appearances': [
                    {'time': 0.0, 'objects': ['person'], 'confidence': [0.99]},
                    {'time': 1.0, 'objects': ['person', 'product'], 'confidence': [0.98, 0.95]},
                    {'time': 2.5, 'objects': ['person', 'product', 'table'], 'confidence': [0.97, 0.96, 0.90]},
                    {'time': 4.0, 'objects': ['product'], 'confidence': [0.99]}
                ]
            },
            'cta_window': {
                'time_range': '55.0-60.0s',
                'cta_appearances': [
                    {'time': 55.2, 'text': 'Follow for more tips', 'type': 'text_overlay', 'size': 'L', 'confidence': 0.98},
                    {'time': 56.5, 'text': 'Link in bio', 'type': 'caption', 'size': 'M', 'confidence': 0.96},
                    {'time': 58.0, 'text': 'Part 2 tomorrow!', 'type': 'text_overlay', 'size': 'L', 'confidence': 0.99},
                    {'time': 59.5, 'text': 'Don\'t miss out!', 'type': 'text_overlay', 'size': 'M', 'confidence': 0.97}
                ],
                'gesture_sync': [
                    {'time': 55.3, 'gesture': 'pointing_down', 'aligns_with_cta': True, 'confidence': 0.91},
                    {'time': 58.1, 'gesture': 'tapping', 'target': 'follow_button', 'confidence': 0.89}
                ],
                'ui_emphasis': [
                    {'time': 56.5, 'element': 'follow_button', 'action': 'highlight', 'duration': 1.5},
                    {'time': 59.0, 'element': 'profile_link', 'action': 'pulse', 'duration': 1.0}
                ]
            }
        }
    
    def test_realistic_payload_size(self):
        """Test that realistic markers stay within limits"""
        markers = self.create_realistic_markers()
        
        # Check size
        json_str = json.dumps(markers)
        size_kb = len(json_str) / 1024
        
        # Should be well within limits
        assert size_kb < TemporalMarkerSafety.MAX_MARKER_SIZE_KB, \
            f"Realistic markers {size_kb:.1f}KB exceeds limit"
        
        # Validate structure
        errors = TemporalMarkerSafety.validate_markers(markers)
        assert len(errors) == 0, f"Validation errors: {errors}"
    
    def test_text_heavy_video_reduction(self):
        """Test size reduction for text-heavy videos (like recipes)"""
        # Create markers with lots of text
        markers = {
            'first_5_seconds': {
                'text_moments': []
            }
        }
        
        # Add 30 text overlays in first 5 seconds (recipe ingredients)
        ingredients = [
            "2 cups all-purpose flour, sifted",
            "1 cup granulated sugar",
            "3 large eggs, room temperature",
            "1/2 cup unsalted butter, melted",
            "2 teaspoons pure vanilla extract",
            "1 teaspoon baking powder",
            "1/2 teaspoon salt",
            "1 cup whole milk, warmed",
            "1/4 cup cocoa powder",
            "1/2 cup chocolate chips"
        ] * 3  # Repeat to get 30 items
        
        for i, ingredient in enumerate(ingredients[:30]):
            markers['first_5_seconds']['text_moments'].append({
                'time': i * 0.15,  # Every 0.15 seconds
                'text': ingredient,
                'size': 'M',
                'position': 'center',
                'confidence': 0.95
            })
        
        # Check initial size
        initial_size = len(json.dumps(markers)) / 1024
        
        # Apply safety limits
        reduced = TemporalMarkerSafety.check_and_reduce_size(markers)
        final_size = len(json.dumps(reduced)) / 1024
        
        # Verify reduction
        assert len(reduced['first_5_seconds']['text_moments']) <= TemporalMarkerSafety.MAX_TEXT_EVENTS_FIRST_5S
        assert final_size < initial_size
        assert final_size < TemporalMarkerSafety.MAX_MARKER_SIZE_KB
        
        # Check text truncation
        for text_moment in reduced['first_5_seconds']['text_moments']:
            assert len(text_moment['text']) <= TemporalMarkerSafety.MAX_TEXT_LENGTH
    
    def test_extreme_payload_bomb(self):
        """Test handling of extreme payload that could crash the system"""
        # Create massive markers
        markers = {
            'first_5_seconds': {
                'text_moments': [],
                'gesture_moments': [],
                'object_appearances': [],
                'density_progression': list(range(1000)),  # Huge array
                'emotion_sequence': ['happy'] * 1000,
                'extra_data': {f'key_{i}': f'value_{i}' * 100 for i in range(100)}
            },
            'cta_window': {
                'time_range': '50-60s',
                'cta_appearances': [],
                'extra_arrays': [list(range(1000)) for _ in range(10)]
            }
        }
        
        # Add maximum text with maximum length
        for i in range(100):
            markers['first_5_seconds']['text_moments'].append({
                'time': i * 0.05,
                'text': 'X' * 500,  # Very long text
                'metadata': {f'meta_{j}': 'data' * 50 for j in range(20)}
            })
        
        # This should be massive
        initial_size = len(json.dumps(markers)) / 1024
        assert initial_size > 100  # Should be over 100KB
        
        # Apply reduction
        reduced = TemporalMarkerSafety.check_and_reduce_size(markers)
        final_size = len(json.dumps(reduced)) / 1024
        
        # Should be drastically reduced
        assert final_size < TemporalMarkerSafety.MAX_MARKER_SIZE_KB
        assert final_size < initial_size / 2  # At least 50% reduction
        
        # Check aggressive reduction was applied
        assert len(reduced['first_5_seconds']['text_moments']) <= 5  # Aggressive limit
    
    def test_progressive_reduction_steps(self):
        """Test that reduction happens progressively, not all at once"""
        # Create markers just over the limit
        markers = {
            'first_5_seconds': {
                'text_moments': [
                    {
                        'time': i * 0.3,
                        'text': f'Text overlay number {i}',
                        'size': 'M',
                        'position': 'center',
                        'confidence': 0.95 + i * 0.001,
                        'bbox': {'x1': i, 'y1': i, 'x2': 100+i, 'y2': 50+i}
                    } for i in range(12)  # Slightly over limit of 10
                ],
                'gesture_moments': [
                    {
                        'time': i * 0.5,
                        'gesture': 'pointing',
                        'confidence': 0.9,
                        'target': 'product',
                        'intensity': 0.8
                    } for i in range(10)  # Over limit of 8
                ]
            }
        }
        
        # Set a specific target that requires progressive reduction
        reduced = TemporalMarkerSafety.check_and_reduce_size(markers, target_kb=2.0)
        
        # Check that limits were applied
        assert len(reduced['first_5_seconds']['text_moments']) <= 10
        assert len(reduced['first_5_seconds']['gesture_moments']) <= 8
        
        # Check that optional fields were removed
        if reduced['first_5_seconds']['text_moments']:
            first_text = reduced['first_5_seconds']['text_moments'][0]
            assert 'confidence' not in first_text
            assert 'bbox' not in first_text
            assert 'position' not in first_text  # Optional fields removed
            assert 'time' in first_text  # Required field remains
            assert 'text' in first_text  # Required field remains
    
    def test_size_validation_with_context(self):
        """Test size validation including surrounding context"""
        # Create a complete context as it would be sent to Claude
        context = {
            'video_info': {
                'duration': 60.0,
                'fps': 30.0,
                'resolution': '1920x1080',
                'video_id': 'test_123'
            },
            'computed_metrics': {
                'avg_density': 4.5,
                'total_text_events': 45,
                'emotion_distribution': {'happy': 0.6, 'neutral': 0.3, 'surprise': 0.1}
            },
            'timelines': {
                # Compressed timeline (existing approach)
                'textOverlayTimeline': {f'{i}-{i+1}s': {'text': f'Text {i}'} for i in range(20)},
                'gestureTimeline': {f'{i}-{i+1}s': {'gesture': 'pointing'} for i in range(15)}
            },
            'temporal_markers': self.create_realistic_markers()
        }
        
        # Check total context size
        total_size_kb = len(json.dumps(context)) / 1024
        
        # Should be well under API limit with buffer
        assert total_size_kb < TemporalMarkerSafety.HARD_PAYLOAD_LIMIT_KB, \
            f"Total context {total_size_kb:.1f}KB exceeds hard limit"
        
        # Ensure markers are a reasonable portion of total
        markers_size_kb = len(json.dumps(context['temporal_markers'])) / 1024
        markers_percentage = (markers_size_kb / total_size_kb) * 100
        
        assert markers_percentage < 50, \
            f"Markers take up {markers_percentage:.1f}% of payload, should be <50%"
    
    def test_cta_window_size_limits(self):
        """Test CTA window specific size limits"""
        markers = {
            'cta_window': {
                'time_range': '50-60s',
                'cta_appearances': [],
                'gesture_sync': [],
                'ui_emphasis': []
            }
        }
        
        # Add many CTAs
        cta_phrases = [
            "Follow for more",
            "Link in bio",
            "Check out my profile",
            "Part 2 coming tomorrow",
            "Don't forget to subscribe",
            "Turn on notifications",
            "Share with friends",
            "Save this for later",
            "DM for questions",
            "Comment your thoughts"
        ]
        
        for i, phrase in enumerate(cta_phrases * 2):  # 20 CTAs
            markers['cta_window']['cta_appearances'].append({
                'time': 50 + i * 0.5,
                'text': phrase + " - Special offer just for you!",
                'type': 'overlay',
                'confidence': 0.95
            })
        
        # Apply reduction
        reduced = TemporalMarkerSafety.check_and_reduce_size(markers)
        
        # Check CTA limit applied
        assert len(reduced['cta_window']['cta_appearances']) <= TemporalMarkerSafety.MAX_CTA_EVENTS
        
        # Check text truncation
        for cta in reduced['cta_window']['cta_appearances']:
            assert len(cta['text']) <= TemporalMarkerSafety.MAX_TEXT_LENGTH
    
    def test_memory_efficiency(self):
        """Test that size reduction doesn't create memory leaks"""
        import sys
        
        # Create large markers
        large_markers = {
            'first_5_seconds': {
                'text_moments': [
                    {'time': i * 0.1, 'text': 'X' * 200, 'extra': 'Y' * 1000}
                    for i in range(50)
                ]
            }
        }
        
        # Get initial size
        initial_json = json.dumps(large_markers)
        initial_refs = sys.getrefcount(large_markers)
        
        # Apply reduction multiple times
        for _ in range(10):
            reduced = TemporalMarkerSafety.check_and_reduce_size(large_markers)
            
        # Original should be unchanged
        assert json.dumps(large_markers) == initial_json
        
        # Reference count shouldn't have grown significantly
        final_refs = sys.getrefcount(large_markers)
        assert final_refs - initial_refs < 5  # Allow small increase
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in text"""
        markers = {
            'first_5_seconds': {
                'text_moments': [
                    {'time': 0.5, 'text': '🔥 HOT DEAL 🔥'},
                    {'time': 1.0, 'text': '50% OFF 💰💰💰'},
                    {'time': 1.5, 'text': 'こんにちは'},  # Japanese
                    {'time': 2.0, 'text': '🎉🎊🎈🎁🎀'},  # Many emojis
                    {'time': 2.5, 'text': 'Price: $99.99 → $49.99'},
                    {'time': 3.0, 'text': '"Special" Offer\'s Here!'},  # Quotes
                ]
            }
        }
        
        # Should handle without errors
        json_str = json.dumps(markers)
        size_kb = len(json_str.encode('utf-8')) / 1024  # UTF-8 encoding
        
        # Apply safety checks
        validated = TemporalMarkerSafety.check_and_reduce_size(markers)
        
        # Should serialize properly
        json.dumps(validated)  # Should not raise exception
        
        # Text should be preserved (unless truncated for length)
        assert any('🔥' in moment['text'] for moment in validated['first_5_seconds']['text_moments'])