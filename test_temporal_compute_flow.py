#!/usr/bin/env python3
"""
Test script for temporal_compute.py
Tests the complete flow with mock data simulating real ML service outputs
"""

import json
import sys
import traceback
from pathlib import Path
from rumiai_v2.processors.temporal_compute import compute_temporal_windows

def create_test_analysis_dict(duration=15.5):
    """Create a test analysis_dict with all required ML services and timeline data"""
    return {
        'ml_data': {
            'ocr': {
                'textAnnotations': [
                    {'timestamp': 1.5, 'text': 'Hello World'},
                    {'timestamp': 2.3, 'text': 'Subscribe!'},
                    {'timestamp': 4.0, 'text': 'Like this video'},
                    {'timestamp': 8.5, 'text': 'Comment below'},
                    {'timestamp': 14.0, 'text': 'See you next time'}
                ],
                'stickers': [
                    {'timestamp': 0.5, 'type': 'emoji', 'value': '❤️'},
                    {'timestamp': 5.0, 'type': 'emoji', 'value': '👍'},
                    {'timestamp': 13.5, 'type': 'emoji', 'value': '🔥'}
                ]
            },
            'whisper': {
                'segments': [
                    {'start': 0.0, 'end': 2.5, 'text': 'Hey everyone welcome back'},
                    {'start': 2.5, 'end': 5.0, 'text': 'Today we are going to learn'},
                    {'start': 5.0, 'end': 8.0, 'text': 'Something really amazing'},
                    {'start': 8.0, 'end': 11.0, 'text': 'Pay close attention to this'},
                    {'start': 11.0, 'end': 14.0, 'text': 'Hope you enjoyed this video'},
                    {'start': 14.0, 'end': 15.5, 'text': 'See you in the next one'}
                ]
            },
            'yolo': {
                'objectAnnotations': [
                    {'timestamp': 0.5, 'label': 'person', 'confidence': 0.95},
                    {'timestamp': 1.0, 'label': 'person', 'confidence': 0.92},
                    {'timestamp': 3.5, 'label': 'laptop', 'confidence': 0.88},
                    {'timestamp': 7.0, 'label': 'person', 'confidence': 0.90},
                    {'timestamp': 10.0, 'label': 'book', 'confidence': 0.75},
                    {'timestamp': 14.5, 'label': 'person', 'confidence': 0.93}
                ]
            },
            'mediapipe': {
                'poses': [
                    {'timestamp': 0.5, 'landmarks': []},
                    {'timestamp': 1.0, 'landmarks': []},
                    {'timestamp': 5.0, 'landmarks': []},
                    {'timestamp': 10.0, 'landmarks': []},
                    {'timestamp': 15.0, 'landmarks': []}
                ],
                'faces': [
                    {'timestamp': 0.5, 'bbox': {'width': 0.3, 'height': 0.4}},
                    {'timestamp': 1.0, 'bbox': {'width': 0.28, 'height': 0.38}},
                    {'timestamp': 5.0, 'bbox': {'width': 0.15, 'height': 0.2}},
                    {'timestamp': 10.0, 'bbox': {'width': 0.05, 'height': 0.07}},
                    {'timestamp': 15.0, 'bbox': {'width': 0.35, 'height': 0.45}}
                ]
            },
            'audio_energy': {
                'rms_frames': [0.1, 0.15, 0.2, 0.18, 0.22, 0.25, 0.3, 0.28, 
                              0.35, 0.4, 0.38, 0.42, 0.45, 0.5, 0.48, 0.52,
                              0.55, 0.6, 0.58, 0.62, 0.65, 0.7, 0.68, 0.72,
                              0.75, 0.8, 0.78, 0.82, 0.85, 0.9] * 15,  # Simulate 15 seconds at 30fps
                'frames_per_second': 30
            }
        },
        'timeline': {
            'entries': [
                # Emotions from FEAT
                {'entry_type': 'emotion', 'start': 0.5, 'data': {'emotion': 'happy', 'confidence': 0.8}},
                {'entry_type': 'emotion', 'start': 2.0, 'data': {'emotion': 'neutral', 'confidence': 0.7}},
                {'entry_type': 'emotion', 'start': 5.0, 'data': {'emotion': 'surprised', 'confidence': 0.85}},
                {'entry_type': 'emotion', 'start': 8.0, 'data': {'emotion': 'happy', 'confidence': 0.9}},
                {'entry_type': 'emotion', 'start': 12.0, 'data': {'emotion': 'sad', 'confidence': 0.6}},
                {'entry_type': 'emotion', 'start': 14.5, 'data': {'emotion': 'happy', 'confidence': 0.88}},
                
                # Scene changes from PySceneDetect
                {'entry_type': 'scene_change', 'start': 0.0, 'data': {'scene_number': 1}},
                {'entry_type': 'scene_change', 'start': 3.0, 'data': {'scene_number': 2}},
                {'entry_type': 'scene_change', 'start': 8.5, 'data': {'scene_number': 3}},
                {'entry_type': 'scene_change', 'start': 13.0, 'data': {'scene_number': 4}},
                
                # Gestures from MediaPipe
                {'entry_type': 'gesture', 'start': 1.0, 'data': {'gesture': 'pointing', 'confidence': 0.75}},
                {'entry_type': 'gesture', 'start': 4.5, 'data': {'gesture': 'waving', 'confidence': 0.82}},
                {'entry_type': 'gesture', 'start': 9.0, 'data': {'gesture': 'thumbs_up', 'confidence': 0.9}},
                {'entry_type': 'gesture', 'start': 14.0, 'data': {'gesture': 'waving', 'confidence': 0.88}},
                
                # Gaze from MediaPipe
                {'entry_type': 'gaze', 'start': 0.5, 'data': {'looking_at_camera': True, 'confidence': 0.95}},
                {'entry_type': 'gaze', 'start': 3.0, 'data': {'looking_at_camera': False, 'confidence': 0.8}},
                {'entry_type': 'gaze', 'start': 6.0, 'data': {'looking_at_camera': True, 'confidence': 0.92}},
                {'entry_type': 'gaze', 'start': 10.0, 'data': {'looking_at_camera': True, 'confidence': 0.88}},
                {'entry_type': 'gaze', 'start': 14.5, 'data': {'looking_at_camera': True, 'confidence': 0.9}},
                
                # Camera distance from face bbox analysis
                {'entry_type': 'camera_distance', 'start': 0.5, 'data': {'distance': 'close', 'confidence': 0.9}},
                {'entry_type': 'camera_distance', 'start': 1.0, 'data': {'distance': 'close', 'confidence': 0.88}},
                {'entry_type': 'camera_distance', 'start': 5.0, 'data': {'distance': 'medium', 'confidence': 0.85}},
                {'entry_type': 'camera_distance', 'start': 10.0, 'data': {'distance': 'wide', 'confidence': 0.7}},
                {'entry_type': 'camera_distance', 'start': 15.0, 'data': {'distance': 'close', 'confidence': 0.92}}
            ]
        },
        'metadata': {
            'video_id': 'test_video_123',
            'caption_length': 150,
            'hashtag_count': 5,
            'diggCount': 1250,
            'playCount': 45000,
            'collectCount': 320,
            'shareCount': 89,
            'commentCount': 67,
            'createTime': 1698765432,
            'author': {'uniqueId': 'test_creator'},
            'desc': 'Test video description #test #temporal #windows'
        },
        'duration': duration
    }

def test_temporal_compute():
    """Test the temporal compute function with various video durations"""
    
    print("=" * 60)
    print("TESTING TEMPORAL COMPUTE MODULE")
    print("=" * 60)
    
    test_cases = [
        ("Very short video (2s)", 2.0),
        ("Short video (5s)", 5.0),
        ("Medium video (15.5s)", 15.5),
        ("Longer video (30s)", 30.0),
        ("Long video (60s)", 60.0)
    ]
    
    for test_name, duration in test_cases:
        print(f"\n{test_name} - Duration: {duration}s")
        print("-" * 40)
        
        try:
            # Create test data
            analysis_dict = create_test_analysis_dict(duration)
            
            # Run temporal compute
            result = compute_temporal_windows(analysis_dict)
            
            # Display results
            print(f"✅ Success! Video ID: {result['video_id']}")
            print(f"   Duration: {result['duration']}s")
            
            # Check hook window
            if result['temporal_windows']['hook']:
                hook = result['temporal_windows']['hook']
                print(f"   Hook Window: {hook['start']:.1f}-{hook['end']:.1f}s")
                print(f"     - Elements: {hook['element_count']}")
                print(f"     - Speech coverage: {hook['speech_coverage']:.2%}")
                print(f"     - Energy: {hook['energy_level']:.3f}")
                print(f"     - Burst: {hook['burst_pattern']}")
                
                # Check framing distribution
                framing = {k: v for k, v in hook.items() if k.endswith('_ratio')}
                if any('close' in k for k in framing.keys()):
                    print(f"     - Framing: close={hook.get('close_ratio', 0):.1%}, "
                          f"medium={hook.get('medium_ratio', 0):.1%}, "
                          f"wide={hook.get('wide_ratio', 0):.1%}")
            
            # Check middle segments
            segments = result['temporal_windows']['middle_segments']
            if segments:
                print(f"   Middle Segments: {len(segments)}")
                for seg in segments:
                    print(f"     - {seg['segment_name']}: {seg['start']:.1f}-{seg['end']:.1f}s "
                          f"(elements={seg['element_count']})")
            
            # Check closing window
            if result['temporal_windows']['closing']:
                closing = result['temporal_windows']['closing']
                print(f"   Closing Window: {closing['start']:.1f}-{closing['end']:.1f}s")
                print(f"     - Elements: {closing['element_count']}")
                print(f"     - Speech coverage: {closing['speech_coverage']:.2%}")
                print(f"     - Energy: {closing['energy_level']:.3f}")
                
            # Save output for inspection
            output_file = f"test_output_{duration}s.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"   💾 Saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

def test_error_cases():
    """Test error handling and validation"""
    print("\n" + "=" * 60)
    print("TESTING ERROR CASES")
    print("=" * 60)
    
    # Test missing ML service
    print("\n1. Testing missing ML service...")
    try:
        bad_dict = create_test_analysis_dict()
        del bad_dict['ml_data']['ocr']  # Remove required service
        result = compute_temporal_windows(bad_dict)
        print("❌ Should have raised an error!")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    # Test invalid duration
    print("\n2. Testing invalid duration...")
    try:
        bad_dict = create_test_analysis_dict()
        bad_dict['duration'] = -1.0  # Invalid duration
        result = compute_temporal_windows(bad_dict)
        print("❌ Should have raised an error!")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    # Test missing timeline
    print("\n3. Testing missing timeline...")
    try:
        bad_dict = create_test_analysis_dict()
        del bad_dict['timeline']  # Remove timeline
        result = compute_temporal_windows(bad_dict)
        print("❌ Should have raised an error!")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Run tests
    test_temporal_compute()
    test_error_cases()
    
    print("\n✨ All tests completed! Check the generated JSON files for details.")
    print("Files created: test_output_2.0s.json, test_output_5.0s.json, etc.")