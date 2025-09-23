#!/usr/bin/env python3
"""
Test if FEAT can accept numpy arrays directly instead of file paths
This could eliminate significant I/O overhead
"""

import numpy as np
import cv2
import sys
from pathlib import Path
import time

# Apply compatibility patches
sys.path.insert(0, str(Path(__file__).parent))
from scipy_compat import ensure_feat_compatibility
ensure_feat_compatibility()

from feat import Detector

print("=" * 60)
print("FEAT Numpy Array Input Test")
print("=" * 60)

# Initialize detector
detector = Detector(
    face_model='retinaface',
    landmark_model='mobilefacenet',
    au_model='xgb',
    emotion_model='resmasknet',
    device='cuda'
)

# Create a dummy image (white background with a face-like pattern)
# In real use, this would be a video frame
dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

print("\n1. Testing different input methods:")

# Method 1: File path (current approach)
print("\n   Method 1: File path")
import tempfile
temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
cv2.imwrite(temp_file.name, dummy_frame)
temp_file.close()

start = time.time()
try:
    result1 = detector.detect_image(temp_file.name)
    print(f"   ✅ File path method works - Time: {time.time() - start:.3f}s")
    print(f"   Result shape: {result1.shape}")
except Exception as e:
    print(f"   ❌ File path method failed: {e}")

import os
os.unlink(temp_file.name)

# Method 2: Numpy array directly
print("\n   Method 2: Numpy array directly")
start = time.time()
try:
    result2 = detector.detect_image(dummy_frame)
    print(f"   ✅ Numpy array method works - Time: {time.time() - start:.3f}s")
    print(f"   Result shape: {result2.shape}")
except Exception as e:
    print(f"   ❌ Numpy array method failed: {e}")

# Method 3: List of numpy arrays (batch)
print("\n   Method 3: List of numpy arrays (batch)")
batch_frames = [dummy_frame, dummy_frame, dummy_frame]
start = time.time()
try:
    result3 = detector.detect_image(batch_frames)
    print(f"   ✅ Batch numpy array method works - Time: {time.time() - start:.3f}s")
    print(f"   Result shape: {result3.shape}")
except Exception as e:
    print(f"   ❌ Batch numpy array method failed: {e}")

# Check detector methods
print("\n2. Available detector methods:")
methods = [m for m in dir(detector) if not m.startswith('_')]
for method in sorted(methods):
    if 'detect' in method.lower():
        print(f"   - {method}")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)