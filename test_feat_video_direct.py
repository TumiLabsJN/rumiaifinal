#!/usr/bin/env python3
"""
Test FEAT's detect_video method to see if it's faster than frame-by-frame
"""

import sys
from pathlib import Path
import time

# Apply compatibility patches
sys.path.insert(0, str(Path(__file__).parent))
from scipy_compat import ensure_feat_compatibility
ensure_feat_compatibility()

from feat import Detector

print("=" * 60)
print("FEAT detect_video Method Test")
print("=" * 60)

# Initialize detector
detector = Detector(
    face_model='retinaface',
    landmark_model='mobilefacenet',
    au_model='xgb',
    emotion_model='resmasknet',
    device='cuda'
)

# Get a test video path
test_video_path = "/home/jorge/rumiaifinal/test_videos/video_02_highenergy_cuts.mp4"
if not Path(test_video_path).exists():
    print(f"Test video not found at {test_video_path}")
    print("Please provide a valid video path")
    sys.exit(1)

print(f"\nTest video: {test_video_path}")

# Test detect_video with different parameters
print("\n1. Testing detect_video with skip_frames parameter:")

# Try different skip_frames values
skip_values = [1, 10, 30, 60]  # Process every Nth frame

for skip in skip_values:
    print(f"\n   skip_frames={skip} (process every {skip} frames):")
    start = time.time()
    try:
        result = detector.detect_video(
            test_video_path,
            skip_frames=skip
        )
        elapsed = time.time() - start
        print(f"   ✅ Success - Time: {elapsed:.2f}s")
        print(f"   Result shape: {result.shape}")
        print(f"   Frames processed: {len(result)}")
        print(f"   FPS: {len(result)/elapsed:.2f} frames/sec")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

print("\n2. Checking detect_video signature:")
import inspect
sig = inspect.signature(detector.detect_video)
print(f"   Parameters: {list(sig.parameters.keys())}")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)