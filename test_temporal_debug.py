#!/usr/bin/env python3
"""Test temporal_compute on segment_3 to see debug output"""
import json
import sys
sys.path.insert(0, '/home/jorge/rumiaifinal')

from rumiai_v2.processors.temporal_compute import compute_temporal_windows

# Load unified analysis
with open('unified_analysis/7480428850522950920.json') as f:
    data = json.load(f)

# Run temporal compute
result = compute_temporal_windows(data)

# Check segment_3 result
for seg in result['temporal_windows']['middle_segments']:
    if seg.get('segment_name') == 'segment_3':
        print(f"\n=== FINAL OUTPUT FOR SEGMENT_3 ===")
        print(f"overlay_unique_count: {seg['overlay_unique_count']}")
        print(f"has_captions: {seg['has_captions']}")
        print()
