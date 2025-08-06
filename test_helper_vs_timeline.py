#!/usr/bin/env python3
"""Compare helper output vs timeline format needed"""

import json
import sys
from pathlib import Path

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions import (
    extract_yolo_data,
    extract_ocr_data,
    extract_whisper_data
)

# Load actual unified analysis
with open('/home/jorge/rumiaifinal/unified_analysis/7280654844715666731.json', 'r') as f:
    analysis_data = json.load(f)
    ml_data = analysis_data.get('ml_data', {})

print("=== HELPER FUNCTIONS RETURN ===\n")

# Test OCR helper
ocr_result = extract_ocr_data(ml_data)
print("1. OCR Helper returns:")
print(f"   Type: {type(ocr_result)}")
print(f"   Keys: {list(ocr_result.keys())}")
print(f"   Structure: {{'textAnnotations': [...], 'stickers': [...]}}")
print(f"   textAnnotations is a: {type(ocr_result.get('textAnnotations', []))}")
print(f"   Count: {len(ocr_result.get('textAnnotations', []))} annotations")

# Test YOLO helper  
yolo_result = extract_yolo_data(ml_data)
print("\n2. YOLO Helper returns:")
print(f"   Type: {type(yolo_result)}")
print(f"   Structure: List of objects directly")
print(f"   Count: {len(yolo_result)} objects")
if yolo_result:
    print(f"   First item keys: {list(yolo_result[0].keys())}")

# Test Whisper helper
whisper_result = extract_whisper_data(ml_data)
print("\n3. Whisper Helper returns:")
print(f"   Type: {type(whisper_result)}")
print(f"   Keys: {list(whisper_result.keys())}")
print(f"   Structure: {{'text': '...', 'segments': [...]}}")
print(f"   segments is a: {type(whisper_result.get('segments', []))}")
print(f"   Count: {len(whisper_result.get('segments', []))} segments")

print("\n=== COMPUTE FUNCTIONS EXPECT ===\n")
print("Timeline format:")
print("""{
    'textOverlayTimeline': {
        '0-1s': {'text': '...', 'position': '...'},  # Dict with timestamp keys
        '1-2s': {...}
    },
    'objectTimeline': {
        '0-1s': [{'class': '...', 'confidence': 0.8}],  # List per timestamp
        '1-2s': [...]
    },
    'speechTimeline': {
        '0-3s': {'text': '...', 'start_time': 0, 'end_time': 3}
    }
}""")

print("\n=== ANALYSIS ===\n")
print("Helper functions return:")
print("- OCR: Raw object with 'textAnnotations' array")
print("- YOLO: Raw array of objects") 
print("- Whisper: Raw object with 'segments' array")
print("")
print("Compute functions expect:")
print("- Timeline dictionaries with timestamp keys ('0-1s', '1-2s', etc.)")
print("- Each timestamp maps to formatted data")
print("")
print("CONCLUSION: Helpers return RAW ML DATA, not TIMELINE FORMAT")
print("Need transformation: ML Data → Timeline Format")
print("")
print("The helpers ARE working correctly - they extract the data.")
print("But they DON'T transform it to timeline format.")
print("That's what _extract_timelines_from_analysis is supposed to do!")