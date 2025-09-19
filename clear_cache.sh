#!/bin/bash
# Cache clearing script for cold start testing
# As specified in PHASE_1_SERVICE_STRUCTURE.md

echo "=== Clearing all caches for cold start test ==="

# 1. Clear frame caches
echo "Clearing frame caches..."
rm -rf /tmp/rumiai_frames_*
rm -rf /tmp/tmp*frame*.jpg

# 2. Clear video downloads (optional - comment out to keep test videos)
# echo "Clearing video downloads..."
# rm -rf /home/jorge/rumiaifinal/temp/*.mp4

# 3. Clear test outputs
echo "Clearing test outputs..."
rm -rf /tmp/vision_test_*
rm -rf /home/jorge/rumiaifinal/insights/test_*

# 4. Clear Python cache
echo "Clearing Python cache..."
find /home/jorge/rumiaifinal -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 5. Clear service output directories
echo "Clearing service outputs..."
rm -rf /home/jorge/rumiaifinal/yolo_outputs/*
rm -rf /home/jorge/rumiaifinal/whisper_outputs/*
rm -rf /home/jorge/rumiaifinal/mediapipe_outputs/*
rm -rf /home/jorge/rumiaifinal/ocr_outputs/*
rm -rf /home/jorge/rumiaifinal/scene_detection_outputs/*
rm -rf /home/jorge/rumiaifinal/audio_energy_outputs/*
rm -rf /home/jorge/rumiaifinal/emotion_detection_outputs/*
rm -rf /home/jorge/rumiaifinal/gender_detection_outputs/*

# 6. Kill all Python processes (optional - uncomment if needed)
# echo "Killing Python processes..."
# pkill -f python3

# 7. Wait for system to settle
echo "Waiting for system to settle..."
sleep 2

echo "=== Cache clearing complete ==="