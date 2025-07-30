# RumiAI Final - Clean ML Video Analysis Pipeline

This is a cleaned version of RumiAI v2 containing only the essential files needed to run `rumiai_runner.py`.

## What's Included

### Core Components
- **scripts/rumiai_runner.py** - Main entry point for video analysis
- **rumiai_v2/** - Complete Python package with all processors, APIs, and utilities
- **prompt_templates/** - Claude prompt templates for different analysis types
- **local_analysis/** - ML implementation scripts
- **requirements.txt** - All Python dependencies

### ML Scripts
- **mediapipe_human_detector.py** - Human detection using MediaPipe
- **detect_tiktok_creative_elements.py** - Comprehensive ML analysis

### Configuration
- **config/temporal_markers.json** - Temporal marker settings
- **.env.example** - Environment variable template
- **setup.sh** - Automated setup script
- **Dockerfile** - Container configuration

## Quick Start

1. **Install dependencies**:
   ```bash
   # Follow setup_dependencies.md for detailed instructions
   ./setup.sh
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run analysis**:
   ```bash
   source venv/bin/activate
   python scripts/rumiai_runner.py "https://www.tiktok.com/@user/video/123"
   ```

## Documentation

- **ML_PRECOMPUTE_DEPENDENCY_MAP.md** - Complete dependency documentation
- **setup_dependencies.md** - Detailed setup instructions

## Directory Structure

```
rumiaifinal/
├── scripts/                 # Entry points
│   └── rumiai_runner.py
├── rumiai_v2/              # Core Python package
├── prompt_templates/       # Claude prompts
├── local_analysis/         # ML implementations
├── config/                 # Configuration files
└── [output directories]    # Auto-created on first run
```

## Features

- TikTok video scraping via Apify
- Multiple ML analyses (YOLO, Whisper, MediaPipe, OCR)
- Claude AI integration for content analysis
- Temporal marker generation
- Unified timeline building
- Memory-efficient processing

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB recommended)
- FFmpeg
- Valid API keys for Claude and Apify

See `setup_dependencies.md` for complete setup instructions.