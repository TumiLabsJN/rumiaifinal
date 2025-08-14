#!/bin/bash
# Test PersonFramingV2 implementation with E2E test

# Set Python-only processing environment variables
export USE_PYTHON_ONLY_PROCESSING=true
export USE_ML_PRECOMPUTE=true
export PRECOMPUTE_CREATIVE_DENSITY=true
export PRECOMPUTE_EMOTIONAL_JOURNEY=true
export PRECOMPUTE_PERSON_FRAMING=true
export PRECOMPUTE_SCENE_PACING=true
export PRECOMPUTE_SPEECH_ANALYSIS=true
export PRECOMPUTE_VISUAL_OVERLAY=true
export PRECOMPUTE_METADATA=true

# Run test on first available video
VIDEO_PATH="temp/7521850433560775991.mp4"

echo "=================================================================="
echo "PersonFramingV2 E2E Test"
echo "=================================================================="
echo "Video: $VIDEO_PATH"
echo "Environment: Python-only processing enabled"
echo ""

# Run the E2E test
python3 test_python_only_e2e.py "$VIDEO_PATH"