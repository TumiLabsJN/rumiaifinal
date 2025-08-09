#!/bin/bash
# Install ML dependencies for RumiAI Python-only processing
# For Python 3.12.3 system-wide installation

echo "==========================================="
echo "RumiAI ML Dependencies Installation"
echo "Python version: $(python3 --version)"
echo "Installing to: $(python3 -m site --user-site)"
echo "==========================================="

# Check if running as user jorge
if [ "$USER" != "jorge" ]; then
    echo "Warning: Not running as user jorge. Packages will install for $USER"
fi

# Upgrade pip first (important for Python 3.12 compatibility)
echo "1. Upgrading pip..."
python3 -m pip install --upgrade pip --break-system-packages

# Install PyTorch first (required by many other packages)
echo "2. Installing PyTorch..."
# For CPU-only (no CUDA)
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# Install YOLO (ultralytics)
echo "3. Installing YOLO (ultralytics)..."
python3 -m pip install ultralytics --break-system-packages

# Install MediaPipe
echo "4. Installing MediaPipe..."
python3 -m pip install mediapipe --break-system-packages

# Install EasyOCR
echo "5. Installing EasyOCR..."
python3 -m pip install easyocr --break-system-packages

# Install SceneDetect
echo "6. Installing SceneDetect..."
python3 -m pip install scenedetect[opencv] --break-system-packages

# Verify installations
echo ""
echo "==========================================="
echo "Verifying installations..."
echo "==========================================="

python3 -c "
import sys
print(f'Python: {sys.version}')
print()

packages = {
    'torch': 'PyTorch',
    'ultralytics': 'YOLO',
    'mediapipe': 'MediaPipe',
    'easyocr': 'EasyOCR',
    'cv2': 'OpenCV',
    'scenedetect': 'SceneDetect'
}

failed = []
for module, name in packages.items():
    try:
        if module == 'cv2':
            import cv2
            version = cv2.__version__
        elif module == 'torch':
            import torch
            version = torch.__version__
        elif module == 'ultralytics':
            import ultralytics
            version = ultralytics.__version__
        elif module == 'mediapipe':
            import mediapipe
            version = mediapipe.__version__ if hasattr(mediapipe, '__version__') else 'installed'
        elif module == 'easyocr':
            import easyocr
            version = 'installed'
        elif module == 'scenedetect':
            import scenedetect
            version = scenedetect.__version__ if hasattr(scenedetect, '__version__') else 'installed'
        print(f'✅ {name:15} {version}')
    except ImportError as e:
        print(f'❌ {name:15} FAILED: {e}')
        failed.append(name)

print()
if failed:
    print(f'⚠️  Failed to install: {', '.join(failed)}')
    print('   Try installing them individually with debugging:')
    for pkg in failed:
        print(f'   python3 -m pip install {pkg.lower()} --verbose')
    sys.exit(1)
else:
    print('🎉 All ML dependencies installed successfully!')
    print('   RumiAI Python-only processing is ready to use.')
"

echo ""
echo "==========================================="
echo "Testing YOLO model download..."
echo "==========================================="

# Test YOLO model download
python3 -c "
from ultralytics import YOLO
import os

model_path = '/home/jorge/rumiaifinal/yolov8n.pt'
if os.path.exists(model_path):
    print(f'✅ YOLO model already exists: {model_path}')
else:
    print('📥 Downloading YOLO model...')
    model = YOLO('yolov8n.pt')
    print(f'✅ YOLO model ready')
"

echo ""
echo "Installation complete!"
echo "You can now run: python3 scripts/rumiai_runner.py"