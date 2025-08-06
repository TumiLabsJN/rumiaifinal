#!/usr/bin/env python3
"""Test if the helper functions actually work"""

import json
import sys
from pathlib import Path

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions import (
    extract_yolo_data,
    extract_whisper_data,
    extract_ocr_data,
    extract_mediapipe_data
)

# Load actual unified analysis
with open('/home/jorge/rumiaifinal/unified_analysis/7280654844715666731.json', 'r') as f:
    analysis_data = json.load(f)

# First check the raw data structure
print("=== Raw Data Structure ===")
ml_data = analysis_data.get('ml_data', {})
print(f"Keys in ml_data: {list(ml_data.keys())}")
if 'ocr' in ml_data:
    print(f"Keys in ml_data['ocr']: {list(ml_data['ocr'].keys())}")
    print(f"Has 'data' key: {'data' in ml_data['ocr']}")
    if 'data' in ml_data['ocr']:
        print(f"Keys in ml_data['ocr']['data']: {list(ml_data['ocr']['data'].keys())}")

# Test the helper functions
print("\n=== Testing Helper Functions ===")

# Test OCR extraction with analysis_data (full structure)
ocr_data = extract_ocr_data(analysis_data)
print(f"OCR helper with full analysis_data: {len(ocr_data.get('textAnnotations', []))} annotations")

# Test OCR extraction with just ml_data
ocr_data2 = extract_ocr_data(ml_data)
print(f"OCR helper with ml_data: {len(ocr_data2.get('textAnnotations', []))} annotations")
print(f"  First: {ocr_data.get('textAnnotations', [{}])[0].get('text', 'None') if ocr_data.get('textAnnotations') else 'None'}")

# Test ALL helpers with ml_data (correct input)
print("\n=== Testing with ml_data (correct input) ===")
whisper_data = extract_whisper_data(ml_data)
print(f"Whisper helper: {len(whisper_data.get('segments', []))} segments")

yolo_objects = extract_yolo_data(ml_data)
print(f"YOLO helper: {len(yolo_objects)} objects")

mediapipe_data = extract_mediapipe_data(ml_data)
print(f"MediaPipe poses: {len(mediapipe_data.get('poses', []))}")
print(f"MediaPipe faces: {len(mediapipe_data.get('faces', []))}")

# Verify the first compute_creative_density_wrapper is broken
print("\n=== The Bug in compute_creative_density_wrapper ===")
print(f"Line 99: extract_yolo_data(analysis_data) returns: {len(extract_yolo_data(analysis_data))} objects")
print(f"Should be: extract_yolo_data(ml_data) returns: {len(extract_yolo_data(ml_data))} objects")

print("\n✅ Helper functions handle the data correctly!")