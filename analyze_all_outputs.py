#!/usr/bin/env python3
"""Analyze all Claude outputs to see what data is actually getting through"""

import json
import glob

# Find all creative density outputs
files = glob.glob('/home/jorge/rumiaifinal/insights/*/creative_density/*complete*.json')

print(f"=== ANALYZING {len(files)} CREATIVE DENSITY OUTPUTS ===\n")

results = []
for file in files[:10]:  # Analyze first 10
    with open(file, 'r') as f:
        data = json.load(f)
        if 'response' in data and data['response']:
            response = json.loads(data['response'])
            metrics = response.get('densityCoreMetrics', {})
            
            video_id = file.split('/')[-3]
            
            result = {
                'video_id': video_id,
                'total_elements': metrics.get('totalElements', 0),
                'scene_changes': metrics.get('sceneChangeCount', 0),
                'text': metrics.get('elementCounts', {}).get('text', 0),
                'objects': metrics.get('elementCounts', {}).get('object', 0),
                'confidence': metrics.get('confidence', 0)
            }
            results.append(result)

# Print results
print("Video ID                | Total | Scenes | Text | Objects | Confidence")
print("-" * 75)
for r in results:
    print(f"{r['video_id']:<20} | {r['total_elements']:>5} | {r['scene_changes']:>6} | {r['text']:>4} | {r['objects']:>7} | {r['confidence']:>10.2f}")

# Analyze patterns
print(f"\n=== PATTERN ANALYSIS ===\n")

# Count how many have non-zero elements
has_scenes = sum(1 for r in results if r['scene_changes'] > 0)
has_text = sum(1 for r in results if r['text'] > 0)
has_objects = sum(1 for r in results if r['objects'] > 0)

print(f"Videos with scene changes: {has_scenes}/{len(results)} ({has_scenes*100//len(results)}%)")
print(f"Videos with text detected: {has_text}/{len(results)} ({has_text*100//len(results)}%)")
print(f"Videos with objects detected: {has_objects}/{len(results)} ({has_objects*100//len(results)}%)")

# Check if total equals scenes only
scenes_only = sum(1 for r in results if r['total_elements'] == r['scene_changes'] and r['total_elements'] > 0)
print(f"\nVideos where total_elements = scene_changes only: {scenes_only}/{len(results)}")

# Average confidence
avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
print(f"\nAverage confidence: {avg_confidence:.2f}")

print("\n=== CONCLUSION ===")
if has_text == 0 and has_objects == 0:
    print("❌ NO videos have text or object data reaching Claude")
    print("✅ Scene changes ARE reaching Claude")
    print("→ Confirms: OCR/YOLO extraction is broken, scene extraction works")
else:
    print("⚠️ SOME data is getting through - need to investigate which videos")