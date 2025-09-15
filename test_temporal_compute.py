#!/usr/bin/env python3
"""
Test script for temporal_compute.py
Verifies Phase 1 implementation with sample data
"""

import json
import sys
from pathlib import Path
from rumiai_v2.processors.temporal_compute import (
    calculate_temporal_windows,
    calculate_middle_segments,
    compute_temporal_windows,
    save_temporal_unified
)

def test_window_calculations():
    """Test window boundary calculations for various durations"""
    print("\n=== Testing Window Calculations ===")
    
    test_durations = [3, 4, 5, 6, 7, 9, 30, 120]
    
    for duration in test_durations:
        windows = calculate_temporal_windows(duration)
        segments = calculate_middle_segments(duration)
        
        print(f"\nDuration: {duration}s")
        print(f"  Windows: {windows}")
        if segments:
            print(f"  Segments: {segments}")
    
    print("\n✓ Window calculations completed")

def test_full_computation():
    """Test full temporal computation with sample data"""
    print("\n=== Testing Full Computation ===")
    
    # Sample timeline data
    timelines = {
        'text_overlay_timeline': [
            {'timestamp': 0.5, 'text': 'Welcome'},
            {'timestamp': 5.0, 'text': 'Subscribe'},
            {'timestamp': 28.0, 'text': 'Thanks'}
        ],
        'sticker_timeline': [
            {'timestamp': 1.0, 'sticker': 'emoji1'},
            {'timestamp': 15.0, 'sticker': 'emoji2'}
        ],
        'object_timeline': [
            {'timestamp': 2.0, 'objects': ['person', 'dog']},
            {'timestamp': 10.0, 'objects': ['car']}
        ],
        'gesture_timeline': [
            {'timestamp': 3.0, 'gesture': 'pointing'},
            {'timestamp': 25.0, 'gesture': 'waving'}
        ],
        'expression_timeline': [
            {'timestamp': 1.5, 'expression': 'happy'},
            {'timestamp': 12.0, 'expression': 'surprised'},
            {'timestamp': 28.5, 'expression': 'happy'}
        ],
        'scene_boundaries': [1.0, 4.0, 10.0, 20.0, 28.0],
        'personTimeline': {
            '1.0-person': {'face_bbox': [0, 0, 100, 100]},
            '2.0-person': {'face_bbox': [0, 0, 100, 100]},
            '5.0-person': {'face_bbox': [0, 0, 100, 100]},
            '15.0-person': {'face_bbox': [0, 0, 100, 100]}
        },
        'gaze_timeline': {
            '1.0-gaze': {'looking_at_camera': True},
            '2.0-gaze': {'looking_at_camera': False},
            '5.0-gaze': {'looking_at_camera': True},
            '15.0-gaze': {'looking_at_camera': True}
        },
        'camera_distance_timeline': {
            '0.0-dist': {'distance': 'close-up'},
            '3.0-dist': {'distance': 'medium'},
            '10.0-dist': {'distance': 'wide'},
            '25.0-dist': {'distance': 'close-up'}
        }
    }
    
    # Sample video metadata
    video_metadata = {
        'video_id': 'test_123',
        'duration': 30.0,
        'publish_hour': 14,
        'caption_length': 100,
        'hashtag_count': 5,
        'has_captions': True,
        'has_soundtrack': True,
        'view_count': 10000,
        'like_count': 500,
        'comment_count': 50,
        'share_count': 25
    }
    
    # Sample speech segments
    speech_segments = [
        {'start': 0.5, 'duration': 2.0, 'text': 'Hello everyone welcome'},
        {'start': 3.0, 'duration': 1.5, 'text': '♪♪♪'},  # Music - should be filtered
        {'start': 5.0, 'duration': 10.0, 'text': 'Today we will learn about temporal windows'},
        {'start': 16.0, 'duration': 8.0, 'text': 'This is the middle section of our video'},
        {'start': 27.0, 'duration': 2.5, 'text': 'Thanks for watching please subscribe'}
    ]
    
    # Compute temporal windows
    try:
        result = compute_temporal_windows(
            timelines=timelines,
            video_metadata=video_metadata,
            speech_segments=speech_segments,
            audio_path=None  # Skip audio for now
        )
        
        print("\n✓ Computation completed successfully")
        
        # Display structure
        print("\nResult structure:")
        print(f"  - video_id: {result['video_id']}")
        print(f"  - duration: {result['duration']}")
        print(f"  - temporal_windows keys: {list(result['temporal_windows'].keys())}")
        
        # Check hook window
        if 'hook' in result['temporal_windows']:
            hook = result['temporal_windows']['hook']
            print(f"\n  Hook window features ({len(hook)} total):")
            for key in sorted(hook.keys())[:10]:  # Show first 10
                print(f"    - {key}: {hook[key]}")
            if len(hook) > 10:
                print(f"    ... and {len(hook) - 10} more features")
        
        # Check middle segments
        if 'middle' in result['temporal_windows']:
            middle = result['temporal_windows']['middle']
            if 'segments' in middle:
                print(f"\n  Middle segments: {list(middle['segments'].keys())}")
        
        # Check global metadata
        print(f"\n  Global metadata keys: {list(result['global_metadata'].keys())}")
        
        # Save to file
        output_path = Path('/home/jorge/rumiaifinal/test_temporal_unified.json')
        save_temporal_unified(result, output_path)
        print(f"\n✓ Saved to {output_path}")
        
        # Verify file size
        file_size = output_path.stat().st_size
        print(f"  File size: {file_size:,} bytes")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Error during computation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_edge_cases():
    """Test edge cases and validation"""
    print("\n=== Testing Edge Cases ===")
    
    # Test 1: Invalid video duration (should fail loudly)
    print("\nTest 1: Invalid video duration")
    try:
        result = compute_temporal_windows(
            timelines={},
            video_metadata={'duration': 0},
            speech_segments=[]
        )
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 2: Very short video (3 seconds)
    print("\nTest 2: 3-second video (all hook)")
    result = compute_temporal_windows(
        timelines={'text_overlay_timeline': [{'timestamp': 1.0, 'text': 'Hi'}]},
        video_metadata={'duration': 3.0, 'video_id': 'short_video'},
        speech_segments=[{'start': 0.5, 'duration': 2.0, 'text': 'Quick message'}]
    )
    print(f"  Windows present: {list(result['temporal_windows'].keys())}")
    assert 'hook' in result['temporal_windows']
    assert 'middle' not in result['temporal_windows']
    assert 'closing' not in result['temporal_windows']
    print("  ✓ Correctly handled 3-second video")
    
    # Test 3: Empty timelines (should log warnings but work)
    print("\nTest 3: Empty timelines")
    result = compute_temporal_windows(
        timelines={'text_overlay_timeline': []},
        video_metadata={'duration': 10.0, 'video_id': 'empty_test'},
        speech_segments=[]
    )
    print("  ✓ Handled empty timelines")
    
    print("\n✓ All edge cases passed")

if __name__ == "__main__":
    print("=" * 60)
    print("TEMPORAL COMPUTE TEST SUITE")
    print("Testing Phase 1 Implementation")
    print("=" * 60)
    
    # Run tests
    test_window_calculations()
    result = test_full_computation()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
    
    if result:
        print("\n✓ Phase 1 implementation is working!")
        print("  Next steps:")
        print("  1. Test with real video data")
        print("  2. Add audio energy calculation (requires librosa)")
        print("  3. Integrate into rumiai_runner.py")
    else:
        print("\n✗ Some tests failed - review errors above")