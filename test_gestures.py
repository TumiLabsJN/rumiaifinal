#!/usr/bin/env python3
"""Test gesture detection implementation"""
import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from rumiai_v2.api.gesture_recognizer_service import GestureRecognizerService
import cv2
import numpy as np

def test_gesture_recognizer():
    """Test the gesture recognizer service"""
    print("Testing Gesture Recognizer Service...")
    
    # Initialize service
    service = GestureRecognizerService()
    
    # Create test frame (blank for now)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test recognition
    gestures = service.recognize_frame(test_frame, timestamp_ms=0)
    
    print(f"✓ Service initialized")
    print(f"✓ Recognition completed: {len(gestures)} gestures detected")
    
    return True

def test_with_video(video_path):
    """Test with actual video"""
    print(f"\nTesting with video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    service = GestureRecognizerService()
    
    frame_count = 0
    gesture_count = 0
    
    while cap.isOpened() and frame_count < 100:  # Test first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:  # Sample every 10th frame
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gestures = service.recognize_frame(frame_rgb, timestamp_ms=frame_count * 33)
            
            if gestures:
                gesture_count += len(gestures)
                for g in gestures:
                    print(f"  Frame {frame_count}: {g['type']} (confidence: {g['confidence']:.2f})")
        
        frame_count += 1
    
    cap.release()
    print(f"✓ Processed {frame_count} frames, detected {gesture_count} gestures")
    
    return gesture_count > 0

if __name__ == "__main__":
    # Test basic functionality
    if test_gesture_recognizer():
        print("\n✅ Basic test passed")
    
    # Test with video if provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if Path(video_path).exists():
            if test_with_video(video_path):
                print("\n✅ Video test passed - gestures detected!")
            else:
                print("\n⚠️ No gestures detected in video")
        else:
            print(f"\n❌ Video not found: {video_path}")