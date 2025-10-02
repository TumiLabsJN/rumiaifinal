#!/usr/bin/env python3
"""Test temporal compute for video 595997271203511 with OCR fixes."""

import json
import sys
from rumiai_v2.processors.temporal_compute import compute_temporal_windows

# Load unified analysis
with open('unified_analysis/595997271203511.json', 'r') as f:
    data = json.load(f)

# Extract what we need
video_id = data['video_id']
duration = data['duration']
timeline = data.get('timeline', {})

print(f"Processing video {video_id} (duration={duration}s)")
print(f"Timeline has {len(timeline)} entries")

# Compute temporal windows
result = compute_temporal_windows(data)

# Print all results
print(f"\n=== HOOK ({result['temporal_windows']['hook']['start']}s - {result['temporal_windows']['hook']['end']}s) ===")
print(f"overlay_unique_count: {result['temporal_windows']['hook']['overlay_unique_count']}")
print(f"has_captions: {result['temporal_windows']['hook'].get('has_captions', False)}")

for i, segment in enumerate(result['temporal_windows']['middle_segments'], 1):
    segment_name = segment.get('segment_name', f'segment_{i}')
    print(f"\n=== {segment_name.upper()} ({segment['start']}s - {segment['end']}s) ===")
    print(f"overlay_unique_count: {segment['overlay_unique_count']}")
    print(f"has_captions: {segment.get('has_captions', False)}")

# Save full output
output_path = f'insights/{video_id}_temporal_windows_test.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nFull output saved to {output_path}")
