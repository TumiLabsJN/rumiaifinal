#!/usr/bin/env python3
"""Test if we can use helpers + transformation to fix the issue"""

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

print("=== PROPOSED SOLUTION ===\n")
print("Use helpers for EXTRACTION + add TRANSFORMATION:\n")

# Step 1: Use helpers to extract data (handles multiple formats)
ocr_data = extract_ocr_data(ml_data)
yolo_objects = extract_yolo_data(ml_data)
whisper_data = extract_whisper_data(ml_data)

print(f"Step 1 - Extraction via helpers:")
print(f"  OCR: {len(ocr_data.get('textAnnotations', []))} annotations extracted ✓")
print(f"  YOLO: {len(yolo_objects)} objects extracted ✓")
print(f"  Whisper: {len(whisper_data.get('segments', []))} segments extracted ✓")

# Step 2: Transform to timeline format
print(f"\nStep 2 - Transform to timeline format:")

# Transform OCR to timeline
textOverlayTimeline = {}
for annotation in ocr_data.get('textAnnotations', []):
    timestamp = annotation.get('timestamp', 0)
    start = int(timestamp)
    end = start + 1
    timestamp_key = f"{start}-{end}s"
    
    # Derive position from bbox
    bbox = annotation.get('bbox', [0, 0, 0, 0])
    y_pos = bbox[1] if len(bbox) > 1 else 0
    position = 'bottom' if y_pos > 350 else 'center'
    
    textOverlayTimeline[timestamp_key] = {
        'text': annotation.get('text', ''),
        'position': position,
        'confidence': annotation.get('confidence', 0.9)
    }

print(f"  OCR: {len(textOverlayTimeline)} timeline entries created ✓")

# Transform YOLO to timeline
objectTimeline = {}
for obj in yolo_objects:
    timestamp = obj.get('timestamp', 0)
    start = int(timestamp)
    end = start + 1
    timestamp_key = f"{start}-{end}s"
    
    if timestamp_key not in objectTimeline:
        objectTimeline[timestamp_key] = []
    
    objectTimeline[timestamp_key].append({
        'class': obj.get('className', 'unknown'),
        'confidence': obj.get('confidence', 0.5)
    })

print(f"  YOLO: {len(objectTimeline)} timeline entries created ✓")

# Transform Whisper to timeline
speechTimeline = {}
for segment in whisper_data.get('segments', []):
    start = int(segment.get('start', 0))
    end = int(segment.get('end', start + 1))
    timestamp_key = f"{start}-{end}s"
    
    speechTimeline[timestamp_key] = {
        'text': segment.get('text', ''),
        'confidence': 0.9
    }

print(f"  Whisper: {len(speechTimeline)} timeline entries created ✓")

print("\n=== BENEFITS OF THIS APPROACH ===\n")
print("1. Helpers handle MULTIPLE formats (old/new/nested)")
print("2. Clear separation: Extraction vs Transformation")
print("3. Reusable helpers across the codebase")
print("4. Defensive programming built into helpers")
print("5. No hardcoded assumptions about data structure")

print("\n=== THE RIGHT FIX ===\n")
print("Modify _extract_timelines_from_analysis to:")
print("1. Use helpers for extraction (handles formats)")
print("2. Keep existing transformation logic")
print("3. Remove hardcoded paths and keys")
print("4. Add validation logging")