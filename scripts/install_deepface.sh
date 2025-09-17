#!/bin/bash
# Quick installation script for DeepFace gender detection service
# Run from project root: bash scripts/install_deepface.sh

echo "================================================"
echo "Installing DeepFace Gender Detection Service"
echo "================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Python version: $python_version"

if [[ ! "$python_version" == "3.12" ]]; then
    echo "Warning: Python 3.12 recommended, you have $python_version"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 1: Installing DeepFace and dependencies..."
echo "------------------------------------------------"
pip install deepface>=0.0.92
pip install tensorflow>=2.16.1,<2.17
pip install tf-keras>=2.16.0

echo ""
echo "Step 2: Pre-downloading DeepFace models..."
echo "------------------------------------------------"
python3 -c "
from deepface import DeepFace
import numpy as np
print('Downloading gender detection models...')
try:
    test_img = np.zeros((224,224,3), dtype=np.uint8)
    result = DeepFace.analyze(test_img, actions=['gender'], enforce_detection=False, silent=True)
    print('✓ Models downloaded successfully')
except Exception as e:
    print(f'✗ Model download failed: {e}')
    exit(1)
"

echo ""
echo "Step 3: Testing DeepFace service..."
echo "------------------------------------------------"
python3 -c "
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from rumiai_v2.ml_services.deepface_gender_service import DeepFaceGenderService, DeepFaceConfig

    # Test configuration
    config = DeepFaceConfig(timeout=10, detector_backend='opencv')
    print(f'✓ Configuration created: timeout={config.timeout}s, backend={config.detector_backend}')

    # Test service initialization
    service = DeepFaceGenderService(config=config)
    print('✓ Service initialized successfully')

    # Test imports
    import cv2
    import asyncio
    print('✓ All dependencies imported successfully')

    print('')
    print('DeepFace Gender Detection Service installed successfully!')
    print('')
    print('To use the service:')
    print('  1. Set environment variables (optional):')
    print('     export DEEPFACE_TIMEOUT=10')
    print('     export DEEPFACE_DETECTOR=opencv')
    print('     export DEEPFACE_USE_GPU=false')
    print('  2. Run video analysis with deepface_gender enabled')

except ImportError as e:
    print(f'✗ Import failed: {e}')
    print('Make sure you are in the project root directory')
    exit(1)
except Exception as e:
    print(f'✗ Service test failed: {e}')
    exit(1)
"

echo ""
echo "Step 4: Creating test directories..."
echo "------------------------------------------------"
mkdir -p gender_detection_outputs
echo "✓ Created gender_detection_outputs/"

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Test with a video: python3 scripts/test_deepface_video.py <video_path>"
echo "2. Run unit tests: pytest tests/test_deepface_service.py -v"
echo "3. Process videos with RumiAI runner including 'deepface_gender' in selected analyses"