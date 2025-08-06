#!/usr/bin/env python3
"""Check if ML services are actually detecting things"""

import json
import os

video_ids = [
    '7454575786134195489',
    '7280654844715666731',
    '7389683775929699616'
]

print("=== CHECKING IF ML SERVICES DETECT DATA ===\n")
print("Video ID                | OCR Text | YOLO Objects | Whisper Segments | Scenes")
print("-" * 80)

for vid in video_ids:
    unified_file = f'/home/jorge/rumiaifinal/unified_analysis/{vid}.json'
    
    if os.path.exists(unified_file):
        with open(unified_file, 'r') as f:
            data = json.load(f)
            ml_data = data.get('ml_data', {})
            
            # Count detections
            ocr_count = len(ml_data.get('ocr', {}).get('textAnnotations', []))
            yolo_count = len(ml_data.get('yolo', {}).get('objectAnnotations', []))
            whisper_count = len(ml_data.get('whisper', {}).get('segments', []))
            
            # Count timeline entries for scenes
            timeline = data.get('timeline', {})
            scene_entries = len([e for e in timeline.get('entries', []) if e.get('entry_type') == 'scene_change'])
            
            print(f"{vid:<20} | {ocr_count:>8} | {yolo_count:>12} | {whisper_count:>16} | {scene_entries:>6}")
    else:
        print(f"{vid:<20} | File not found")

print("\n=== CRITICAL FINDING ===")
print("If ML services detect data (non-zero counts above) but Claude receives 0,")
print("then extraction IS broken as claimed.")
print("\nBut wait... let me check what precomputed metrics are sent to Claude...")