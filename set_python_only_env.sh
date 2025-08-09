#\!/bin/bash
# Set environment variables for Python-only E2E testing

echo "Setting Python-only environment variables..."

# Core flags
export USE_PYTHON_ONLY_PROCESSING=true
export USE_ML_PRECOMPUTE=true

# Individual analysis flags
export PRECOMPUTE_CREATIVE_DENSITY=true
export PRECOMPUTE_EMOTIONAL_JOURNEY=true
export PRECOMPUTE_PERSON_FRAMING=true
export PRECOMPUTE_SCENE_PACING=true
export PRECOMPUTE_SPEECH_ANALYSIS=true
export PRECOMPUTE_VISUAL_OVERLAY=true
export PRECOMPUTE_METADATA=true

echo "✅ Environment configured for Python-only processing"
echo ""
echo "Current settings:"
echo "USE_PYTHON_ONLY_PROCESSING=$USE_PYTHON_ONLY_PROCESSING"
echo "USE_ML_PRECOMPUTE=$USE_ML_PRECOMPUTE"
echo ""
echo "To run the E2E test:"
echo "python test_python_only_e2e.py <video_path>"
