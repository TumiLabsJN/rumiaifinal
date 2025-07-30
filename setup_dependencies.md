# Setup Dependencies for rumiai_runner.py

This guide provides step-by-step instructions to set up all dependencies required to run `rumiai_runner.py` in a fresh environment.

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.8-3.11 (recommended) or 3.12 (requires package adjustments)
- **RAM**: Minimum 4GB, 8GB+ recommended
- **Storage**: 10GB+ for ML models and outputs
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA support for 10-30x speedup)

> **Note**: Python 3.12 requires different package versions than those in requirements.txt. See Python 3.12 section below.

### Required System Packages

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Python and pip
sudo apt install python3 python3-pip python3-venv

# Video processing dependencies
sudo apt install ffmpeg

# OpenCV dependencies
sudo apt install libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Additional libraries for GUI-less servers
sudo apt install libgl1-mesa-glx libglib2.0-0 libgomp1

# Git (for installing CLIP from GitHub)
sudo apt install git
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 ffmpeg git
```

### Optional: CUDA Setup (for GPU acceleration)
If you have an NVIDIA GPU:
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA Toolkit 11.8 (compatible with PyTorch 2.1.0)
# Follow instructions at: https://developer.nvidia.com/cuda-11-8-0-download-archive
```

## Pre-Installation: Check GPU Availability

Before starting installation, check if you have an NVIDIA GPU:

```bash
# Check for NVIDIA GPU
nvidia-smi

# If the command shows GPU information, you have GPU support
# If it says "command not found", you'll use CPU-only versions
```

> **CRITICAL**: The presence of a GPU significantly affects which PyTorch version to install. GPU acceleration provides 10-30x speedup for ML tasks!

## Installation Steps

> **Important**: If you encounter package version conflicts during installation, especially with Python 3.12, it's recommended to install packages in groups rather than all at once. This helps identify and resolve conflicts more easily.

### 1. Clone/Copy Repository
```bash
# If cloning
git clone <repository-url> rumiaifinal
cd rumiaifinal

# Or if copying
cp -r /path/to/rumiaifinal /new/location/rumiaifinal
cd /new/location/rumiaifinal
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Python Dependencies

#### Option A: Install from requirements.txt (Python 3.8-3.11)
```bash
# Upgrade pip first
pip install --upgrade pip

# Install main requirements
pip install -r requirements.txt

# For exact reproducibility
pip install -r requirements_exact.txt
```

#### Option B: Install for Python 3.12
If you have Python 3.12, you'll need to install packages with compatible versions:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install basic dependencies
pip install requests==2.31.0 python-dotenv==1.0.0 anthropic==0.18.1 \
    opencv-python-headless==4.8.1.78 moviepy==1.0.3 ffmpeg-python==0.2.0

# Install more dependencies
pip install aiohttp==3.9.1 psutil==5.9.6 pandas==2.1.3 scipy scikit-learn \
    colorama==0.4.6

# Install PyTorch - IMPORTANT: Check for GPU first!
# Check if you have NVIDIA GPU:
nvidia-smi  # If this shows GPU info, use CUDA version below

# For systems WITH NVIDIA GPU (RECOMMENDED):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For systems WITHOUT GPU (CPU only):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install ML models
pip install openai-whisper==20231117 transformers==4.35.2

# Install remaining ML packages (latest compatible versions)
pip install ultralytics mediapipe easyocr scenedetect deep-sort-realtime

# Install CLIP from GitHub
pip install git+https://github.com/openai/CLIP.git
```

> **Note**: The requirements_py312.txt file may still have version conflicts. The above commands install compatible versions.

#### Option C: Manual Installation (if requirements.txt is missing)
```bash
# Core dependencies
pip install anthropic==0.7.8
pip install aiohttp==3.9.1
pip install requests==2.31.0
pip install python-dotenv==1.0.0
pip install psutil==5.9.6
pip install numpy==1.24.3

# ML models and video processing
pip install openai-whisper==20231117
pip install ultralytics==8.0.200
pip install mediapipe==0.10.8
pip install easyocr==1.7.0
pip install opencv-python-headless==4.8.1.78
pip install moviepy==1.0.3
pip install ffmpeg-python==0.2.0
pip install scenedetect[opencv]==0.6.2
pip install deep-sort-realtime==1.3.2

# PyTorch - IMPORTANT: Choose the right version for your system
# Check if you have NVIDIA GPU first:
nvidia-smi  # If this works, use CUDA version

# For CUDA 11.8 (RECOMMENDED if you have NVIDIA GPU):
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU only (use ONLY if no GPU available):
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Additional ML dependencies
pip install transformers==4.35.2
pip install pandas==2.1.3
pip install scipy==1.11.4
pip install scikit-learn==1.3.2
pip install pillow==10.1.0
pip install tqdm==4.66.1
pip install colorama==0.4.6

# Install CLIP from GitHub
pip install git+https://github.com/openai/CLIP.git
```

### 4. Download ML Models

Some models need to be downloaded on first use. To pre-download:

```python
# Create a script to download models
python3 << 'EOF'
# Download Whisper model
import whisper
model = whisper.load_model("base")  # or "small", "medium", "large"

# Download YOLO model
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically

# EasyOCR models download on first use
import easyocr
reader = easyocr.Reader(['en'])  # Downloads English model

print("Models downloaded successfully!")
EOF
```

### 5. Set Up Environment Variables

Create a `.env` file in the project root:
```bash
# Copy example env file
cp .env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

Add the following required variables:
```env
# Required API Keys (IMPORTANT: Use these exact variable names)
CLAUDE_API_KEY=your_anthropic_api_key_here
APIFY_API_TOKEN=your_apify_token_here

# Feature Flags (Required for ML precompute mode)
USE_ML_PRECOMPUTE=true
USE_CLAUDE_SONNET=true
OUTPUT_FORMAT_VERSION=v2

# Optional: Performance Settings
MAX_VIDEO_DURATION=180
DEFAULT_FPS=2
PROMPT_DELAY=10
LOG_LEVEL=INFO

# Optional: GPU Usage
RUMIAI_USE_GPU=false  # Set to true if you have GPU

# Note: The .env.example file may have different variable names. 
# Make sure to use CLAUDE_API_KEY (not ANTHROPIC_API_KEY) and 
# APIFY_API_TOKEN (not APIFY_TOKEN)
```

### 6. Create Required Directories

The script creates these automatically, but you can pre-create them:
```bash
mkdir -p temp
mkdir -p insights
mkdir -p unified_analysis
mkdir -p temporal_markers
mkdir -p object_detection_outputs
mkdir -p speech_transcriptions
mkdir -p human_analysis_outputs
mkdir -p creative_analysis_outputs
mkdir -p scene_detection_outputs
mkdir -p ml_outputs
mkdir -p fps_registry
```

### 7. Node.js Dependencies (Optional - for Node.js integration)

If you need the Node.js integration layer:

```bash
# Install Node.js (v18+ recommended)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Node.js dependencies
npm install
```

## Verification Steps

### 1. Verify Python Installation
```bash
# Check Python version
python --version  # Should be 3.8+

# Check virtual environment is activated
which python  # Should point to venv/bin/python
```

### 2. Verify Dependencies
```bash
# Check key packages
pip list | grep -E "anthropic|whisper|ultralytics|mediapipe"

# Test imports
python -c "import rumiai_v2; print('rumiai_v2 package OK')"
python -c "import torch; print(f'PyTorch OK, GPU: {torch.cuda.is_available()}')"
python -c "import whisper; print('Whisper OK')"
```

### 3. Test FFmpeg
```bash
ffmpeg -version  # Should show version info
```

### 4. Test rumiai_runner.py
```bash
# Show help
./venv/bin/python scripts/rumiai_runner.py --help

# Test with a video URL (requires valid API keys in .env)
./venv/bin/python scripts/rumiai_runner.py "https://www.tiktok.com/@user/video/123"
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'rumiai_v2'**
   - Ensure you're in the project root directory
   - Check that rumiai_v2 directory exists

2. **CUDA/GPU errors**
   - Install CPU version of PyTorch if no GPU
   - Check CUDA version compatibility
   - Set `RUMIAI_USE_GPU=false` in .env

3. **FFmpeg not found**
   - Ensure ffmpeg is in PATH
   - On Ubuntu: `sudo apt install ffmpeg`
   - On macOS: `brew install ffmpeg`

4. **Memory errors**
   - Increase system swap space
   - Reduce batch sizes in processing
   - Use smaller ML models (e.g., whisper "base" instead of "large")

5. **Missing whisper_transcribe.py**
   - This is a known issue in MLServices
   - The functionality exists in other scripts but isn't properly integrated

6. **Python 3.12 Package Version Conflicts**
   - If you get errors like "No matching distribution found for mediapipe==0.10.8"
   - Use the Python 3.12 installation instructions (Option B) instead of requirements.txt
   - Common version adjustments needed:
     - mediapipe: Use latest version instead of 0.10.8
     - torch: Use 2.3.0+ instead of 2.1.0
     - ultralytics: Use 8.3.0+ instead of 8.0.200

7. **Environment Variable Issues**
   - Make sure to use the exact variable names shown above
   - The .env.example file may have incorrect names
   - Required: CLAUDE_API_KEY (not ANTHROPIC_API_KEY)
   - Required: APIFY_API_TOKEN (not APIFY_TOKEN)

### Performance Optimization

1. **Use GPU acceleration** - Provides 10-30x speedup for ML tasks
2. **Adjust FPS settings** - Lower FPS for faster processing
3. **Use smaller models** - Trade accuracy for speed when needed
4. **Monitor memory usage** - Script includes automatic memory management

## Quick Start Script

Create a `setup_rumiai.sh` script:
```bash
#!/bin/bash
set -e

echo "Setting up RumiAI v2 environment..."

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create directories
mkdir -p temp insights unified_analysis temporal_markers
mkdir -p object_detection_outputs speech_transcriptions
mkdir -p human_analysis_outputs creative_analysis_outputs
mkdir -p scene_detection_outputs ml_outputs fps_registry

# Copy env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Please edit .env file with your API keys"
fi

echo "Setup complete! Activate environment with: source venv/bin/activate"
```

Make it executable: `chmod +x setup_rumiai.sh`

## Minimal Setup (Quick Test)

For a minimal setup to test basic functionality:
```bash
# Create venv and install minimal deps
python3 -m venv venv
source venv/bin/activate
pip install anthropic aiohttp requests python-dotenv psutil numpy

# Create minimal .env
echo "CLAUDE_API_KEY=your_key_here" > .env
echo "APIFY_API_TOKEN=your_token_here" >> .env

# Test import
python -c "from rumiai_v2 import *; print('Basic setup OK')"
```

Note: This minimal setup won't include ML capabilities.