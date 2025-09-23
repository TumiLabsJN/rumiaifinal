#!/usr/bin/env python3
"""
Test script to diagnose FEAT GPU usage issues
"""

import torch
import sys
import os
from pathlib import Path

print("=" * 60)
print("FEAT GPU Diagnostic Test")
print("=" * 60)

# 1. Check PyTorch CUDA availability
print("\n1. PyTorch CUDA Check:")
print(f"   - torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - CUDA device count: {torch.cuda.device_count()}")
    print(f"   - Current CUDA device: {torch.cuda.current_device()}")
    print(f"   - CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"   - CUDA version: {torch.version.cuda}")
else:
    print("   ❌ CUDA is NOT available")

# 2. Check environment variables
print("\n2. Environment Variables:")
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print(f"   - CUDA_VISIBLE_DEVICES: {cuda_visible}")

# 3. Try to initialize FEAT
print("\n3. FEAT Initialization Test:")
try:
    # Apply compatibility patches
    sys.path.insert(0, str(Path(__file__).parent))
    from scipy_compat import ensure_feat_compatibility
    ensure_feat_compatibility()
    print("   ✅ Compatibility patches applied")
except ImportError:
    print("   ⚠️ Compatibility patches not found")

try:
    from feat import Detector

    # Try GPU initialization
    if torch.cuda.is_available():
        print("\n   Attempting GPU initialization...")
        detector_gpu = Detector(
            face_model='retinaface',
            landmark_model='mobilefacenet',
            au_model='xgb',
            emotion_model='resmasknet',
            device='cuda'
        )
        print(f"   ✅ GPU detector initialized on device: {detector_gpu.device}")
        print(f"   - Batch size (GPU): {detector_gpu.batch_size if hasattr(detector_gpu, 'batch_size') else 'Not accessible'}")

    # Try CPU initialization for comparison
    print("\n   Attempting CPU initialization...")
    detector_cpu = Detector(
        face_model='retinaface',
        landmark_model='mobilefacenet',
        au_model='xgb',
        emotion_model='resmasknet',
        device='cpu'
    )
    print(f"   ✅ CPU detector initialized on device: {detector_cpu.device}")
    print(f"   - Batch size (CPU): {detector_cpu.batch_size if hasattr(detector_cpu, 'batch_size') else 'Not accessible'}")

except Exception as e:
    print(f"   ❌ FEAT initialization failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Check actual device being used by emotion service
print("\n4. Emotion Service Device Check:")
try:
    from rumiai_v2.ml_services.emotion_detection_service import get_emotion_detector
    detector_service = get_emotion_detector()
    print(f"   - Service device: {detector_service.device}")
    print(f"   - Service detector device: {detector_service.detector.device}")
except Exception as e:
    print(f"   ❌ Failed to check service: {e}")

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)