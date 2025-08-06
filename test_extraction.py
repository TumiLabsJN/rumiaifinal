#!/usr/bin/env python3
"""Test the timeline extraction to see what's actually happening"""

import json
import sys
from pathlib import Path

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions import _extract_timelines_from_analysis

# Load actual unified analysis
with open('/home/jorge/rumiaifinal/unified_analysis/7280654844715666731.json', 'r') as f:
    analysis_data = json.load(f)

# Run the extraction
timelines = _extract_timelines_from_analysis(analysis_data)

# Check what we got
print("=== Extraction Results ===")
print(f"textOverlayTimeline entries: {len(timelines.get('textOverlayTimeline', {}))}")
print(f"speechTimeline entries: {len(timelines.get('speechTimeline', {}))}")
print(f"objectTimeline entries: {len(timelines.get('objectTimeline', {}))}")
print(f"sceneChangeTimeline entries: {len(timelines.get('sceneChangeTimeline', {}))}")

# Show first few entries of each
print("\n=== Sample textOverlayTimeline ===")
text_timeline = timelines.get('textOverlayTimeline', {})
for i, (k, v) in enumerate(list(text_timeline.items())[:3]):
    print(f"  {k}: {v}")
if len(text_timeline) == 0:
    print("  (empty)")

print("\n=== Sample speechTimeline ===")
speech_timeline = timelines.get('speechTimeline', {})
for i, (k, v) in enumerate(list(speech_timeline.items())[:3]):
    print(f"  {k}: {v['text'][:50]}...")
if len(speech_timeline) == 0:
    print("  (empty)")

print("\n=== Sample objectTimeline ===")
object_timeline = timelines.get('objectTimeline', {})
for i, (k, v) in enumerate(list(object_timeline.items())[:3]):
    print(f"  {k}: {len(v)} objects")
if len(object_timeline) == 0:
    print("  (empty)")

# Check the raw OCR data
print("\n=== Raw OCR Data Check ===")
ocr_data = analysis_data.get('ml_data', {}).get('ocr', {})
print(f"OCR keys: {list(ocr_data.keys())}")
print(f"textAnnotations count: {len(ocr_data.get('textAnnotations', []))}")
print(f"First annotation: {ocr_data.get('textAnnotations', [{}])[0] if ocr_data.get('textAnnotations') else 'None'}")

# Check the raw Whisper data
print("\n=== Raw Whisper Data Check ===")
whisper_data = analysis_data.get('ml_data', {}).get('whisper', {})
print(f"Whisper keys: {list(whisper_data.keys())}")
print(f"Segments count: {len(whisper_data.get('segments', []))}")

# Check the raw YOLO data
print("\n=== Raw YOLO Data Check ===")
yolo_data = analysis_data.get('ml_data', {}).get('yolo', {})
print(f"YOLO keys: {list(yolo_data.keys())}")
print(f"objectAnnotations count: {len(yolo_data.get('objectAnnotations', []))}")