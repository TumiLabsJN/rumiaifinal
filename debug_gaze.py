#!/usr/bin/env python3
"""Debug gaze variance calculation."""

import json
from pathlib import Path
from rumiai_v2.processors.temporal_compute import calculate_gaze_variance

# Load test unified analysis
test_video = Path("unified_analysis/7430952519439846698.json")
with open(test_video) as f:
    unified_analysis = json.load(f)

# Get timeline entries
timeline_entries = unified_analysis.get('timeline', {}).get('entries', [])
print(f"Total timeline entries: {len(timeline_entries)}")

# Count gaze entries
gaze_entries = [e for e in timeline_entries if e.get('entry_type') == 'gaze']
print(f"Total gaze entries: {len(gaze_entries)}")

# Show first few gaze entries
print("\nFirst 3 gaze entries:")
for entry in gaze_entries[:3]:
    print(f"  Time: {entry.get('start')}, Eye contact: {entry.get('data', {}).get('eye_contact')}")

# Test hook window (0-3s)
print("\n🔍 Testing hook window (0-3s):")
hook_gaze = [e for e in gaze_entries if 0 <= e.get('start', 0) <= 3]
print(f"  Gaze entries in hook: {len(hook_gaze)}")

if hook_gaze:
    eye_contacts = [e.get('data', {}).get('eye_contact', 0) for e in hook_gaze]
    print(f"  Eye contact values: {eye_contacts[:5]}...")

    # Calculate variance manually
    if len(eye_contacts) > 1:
        import statistics
        variance = statistics.variance(eye_contacts)
        print(f"  Manual variance: {variance:.6f}")

# Test with calculate_gaze_variance function
variance = calculate_gaze_variance(timeline_entries, 0, 3)
print(f"  Function variance: {variance:.6f}")