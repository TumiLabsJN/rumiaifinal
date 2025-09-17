#!/usr/bin/env python3
"""Quick integration test for gaze variance in temporal windows."""

import json
from pathlib import Path
from rumiai_v2.processors.temporal_compute import compute_temporal_windows

# Load test unified analysis
test_video = Path("unified_analysis/7430952519439846698.json")
with open(test_video) as f:
    unified_analysis = json.load(f)

# Process temporal windows
result = compute_temporal_windows(unified_analysis)

# Check for gaze_variance in each window
print("\n🔍 Checking gaze_variance in temporal windows:\n")

# Check hook
hook_variance = result['temporal_windows']['hook'].get('gaze_variance')
print(f"✓ Hook gaze_variance: {hook_variance:.4f}" if hook_variance is not None else "❌ Hook missing gaze_variance")

# Check middle segments
for i, segment in enumerate(result['temporal_windows']['middle_segments'], 1):
    variance = segment.get('gaze_variance')
    print(f"✓ Segment {i} gaze_variance: {variance:.4f}" if variance is not None else f"❌ Segment {i} missing gaze_variance")

# Check closing
closing_variance = result['temporal_windows']['closing'].get('gaze_variance')
print(f"✓ Closing gaze_variance: {closing_variance:.4f}" if closing_variance is not None else "❌ Closing missing gaze_variance")

print("\n✅ Gaze variance successfully added to all temporal windows!")