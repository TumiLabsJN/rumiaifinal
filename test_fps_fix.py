#!/usr/bin/env python3
"""
Test script to verify FPS context manager fixes scene pacing timestamps
"""

import json
from fps_utils import FPSContextManager

def test_fps_conversion():
    """Test the FPS context manager with our problematic video"""
    
    video_id = "7367449043070356782"
    manager = FPSContextManager(video_id)
    
    print("=== Testing FPS Context Manager ===")
    print(f"Video: {video_id}")
    print(f"Original FPS: {manager.registry['video_specs']['original_fps']}")
    print(f"Duration: {manager.registry['video_specs']['duration']}s")
    print("")
    
    # Test problematic timestamps
    test_cases = [
        ("543-544s", "scene_detection"),
        ("584-585s", "scene_detection"),
        ("891-892s", "scene_detection"),
        ("1-2s", "yolo_tracking"),
        ("30-31s", "ocr_analysis")
    ]
    
    print("=== Timestamp Conversions ===")
    for timestamp, context in test_cases:
        seconds = manager.timestamp_to_seconds(timestamp, context)
        print(f"{timestamp} ({context}) → {seconds:.2f} seconds")
    
    print("\n=== Display Timestamp Generation ===")
    
    # Test scene data with different formats
    scene_data_with_time = {
        "frame": 584,
        "startTime": 19.5,
        "endTime": 29.733333,
        "type": "shot_change"
    }
    
    scene_data_without_time = {
        "frame": 584,
        "type": "shot_change"
    }
    
    # Test with time data available
    display1 = manager.get_display_timestamp(scene_data_with_time, 307, 'scene_detection')
    print(f"With time data: {display1}")
    
    # Test without time data (should calculate from frames)
    display2 = manager.get_display_timestamp(scene_data_without_time, 307, 'scene_detection')
    print(f"Without time data: {display2}")
    
    # Test the problematic case
    print("\n=== Fixing the 584-891s Problem ===")
    print("Original broken output: 584-891s")
    print(f"Fixed output: {display2}")
    print(f"Explanation: Frame 584 at 30fps = {584/30:.1f}s, duration 307 frames = {307/30:.1f}s")


if __name__ == "__main__":
    test_fps_conversion()