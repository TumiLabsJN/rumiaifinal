#!/usr/bin/env python3
"""Test PersonFramingV2 with actual video data."""

import json
import sys
from pathlib import Path

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions_full import (
    compute_person_framing_metrics,
    calculate_temporal_framing_simple
)

# Load the unified analysis
video_id = "7274651255392210219"
unified_path = f"unified_analysis/{video_id}.json"

with open(unified_path) as f:
    data = json.load(f)

# Extract timelines
from rumiai_v2.processors.precompute_functions import extract_mediapipe_data

# Get MediaPipe data
mediapipe_data = data.get('ml_data', {}).get('mediapipe', {})
extracted = extract_mediapipe_data(mediapipe_data)

print(f"MediaPipe data keys: {list(extracted.keys())}")
print(f"Poses: {len(extracted.get('poses', []))}")
print(f"Faces: {len(extracted.get('faces', []))}")

# Extract timelines using the precompute function module
from rumiai_v2.processors.precompute_functions import _extract_timelines_from_analysis
timelines = _extract_timelines_from_analysis(data)

print(f"\nTimeline keys: {list(timelines.keys())}")
print(f"personTimeline entries: {len(timelines.get('personTimeline', {}))}")

# Check if personTimeline has face_bbox data
person_timeline = timelines.get('personTimeline', {})
sample_entries = list(person_timeline.items())[:3]
for key, val in sample_entries:
    print(f"\nSample {key}:")
    print(f"  Keys: {list(val.keys())}")
    if 'face_bbox' in val:
        print(f"  face_bbox: {val['face_bbox']}")

# Test calculate_temporal_framing_simple
if person_timeline:
    print("\n" + "="*50)
    print("Testing calculate_temporal_framing_simple")
    print("="*50)
    
    duration = data.get('video_metadata', {}).get('duration', 58)
    framing_timeline = calculate_temporal_framing_simple(person_timeline, duration)
    
    print(f"Framing timeline entries: {len(framing_timeline)}")
    
    # Show first 5 entries
    for i, (key, val) in enumerate(list(framing_timeline.items())[:5]):
        print(f"{key}: shot_type={val['shot_type']}, face_size={val['face_size']:.2f}, confidence={val['confidence']:.2f}")
    
    # Count shot types
    shot_counts = {}
    for entry in framing_timeline.values():
        shot_type = entry['shot_type']
        shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
    
    print(f"\nShot type distribution: {shot_counts}")
    
    # Check if PersonFramingV2 would add the fields
    print(f"\n✅ PersonFramingV2 would add:")
    print(f"   - framing_timeline: {len(framing_timeline)} entries")
    print(f"   - framing_progression: Would be calculated")
    print(f"   - framing_changes: Would be calculated")
else:
    print("\n❌ No personTimeline data found!")