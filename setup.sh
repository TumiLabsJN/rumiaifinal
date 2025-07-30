#!/bin/bash

echo "🚀 Setting up RumiAI TikTok Flow..."

# Create necessary directories
echo "📁 Creating output directories..."
mkdir -p insights
mkdir -p unified_analysis
mkdir -p temp
mkdir -p temp/video-analysis
mkdir -p frame_outputs
mkdir -p object_detection_outputs
mkdir -p creative_analysis_outputs
mkdir -p human_analysis_outputs
mkdir -p enhanced_human_analysis_outputs
mkdir -p scene_detection_outputs
mkdir -p comprehensive_analysis_outputs
mkdir -p audio_analysis_outputs
mkdir -p speech_transcriptions
mkdir -p local_analysis_outputs
mkdir -p local_analysis_outputs/scene_detection_outputs

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate and install Python dependencies
echo "📦 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install Node dependencies
echo "📦 Installing Node dependencies..."
npm install

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and add your API keys"
echo "2. Run: node test_rumiai_complete_flow.js <tiktok-url>"