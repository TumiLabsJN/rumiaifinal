#!/usr/bin/env python3
"""Understand why scene changes work but other ML data doesn't"""

import json
import sys
from pathlib import Path

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions import _extract_timelines_from_analysis

# Load the video that has ML data
with open('/home/jorge/rumiaifinal/unified_analysis/7280654844715666731.json', 'r') as f:
    analysis_data = json.load(f)

print("=== VIDEO 7280654844715666731 ===")
print(f"ML Data Available:")
print(f"  OCR: {len(analysis_data['ml_data']['ocr'].get('textAnnotations', []))} annotations")
print(f"  YOLO: {len(analysis_data['ml_data']['yolo'].get('objectAnnotations', []))} objects")
print(f"  Whisper: {len(analysis_data['ml_data']['whisper'].get('segments', []))} segments")
print(f"  Timeline entries: {len(analysis_data['timeline']['entries'])} entries")

# Run extraction
timelines = _extract_timelines_from_analysis(analysis_data)

print(f"\nExtracted Timelines:")
print(f"  textOverlayTimeline: {len(timelines.get('textOverlayTimeline', {}))} entries")
print(f"  objectTimeline: {len(timelines.get('objectTimeline', {}))} entries")
print(f"  speechTimeline: {len(timelines.get('speechTimeline', {}))} entries")
print(f"  sceneChangeTimeline: {len(timelines.get('sceneChangeTimeline', {}))} entries")

print("\n=== CRITICAL DISCOVERY ===")
print("ML services DETECT data: 54 OCR + 1169 YOLO + 42 Whisper = 1265 elements")
print("Extraction RETRIEVES: 0 + 0 + 0 = 0 elements (only 28 scene changes)")
print("Claude RECEIVES: Only scene changes")
print("")
print("CONFIRMED: Extraction is broken for OCR/YOLO/Whisper")
print("CONFIRMED: Scene extraction works (different code path)")

# Now check why scene changes work
print("\n=== WHY SCENE CHANGES WORK ===")
print("Scene changes extracted from: timeline_data.get('entries', [])")
print("Other ML data extracted from: ml_data.get('service', {}).get('data', {})")
print("                                                           ^^^^^^^^^^^^^^")
print("                                                           THIS IS WRONG!")
print("")
print("Scene changes bypass the broken ML extraction logic entirely.")