#!/usr/bin/env python3
"""Test full integration with gaze variance."""

import json
from pathlib import Path

# Load test unified analysis
test_video = Path("unified_analysis/7430952519439846698.json")
with open(test_video) as f:
    unified_analysis = json.load(f)

# Import and run temporal compute
from rumiai_v2.processors.temporal_compute import compute_temporal_windows

print("\n🚀 Processing temporal windows with gaze variance...\n")
result = compute_temporal_windows(unified_analysis)

# Check gaze variance in each window
print("📊 Gaze Variance Results:\n")

# Hook
hook = result['temporal_windows']['hook']
print(f"Hook (0-3s):")
print(f"  gaze_variance: {hook.get('gaze_variance', 'MISSING'):.6f}")

# Middle segments
for i, segment in enumerate(result['temporal_windows']['middle_segments'], 1):
    print(f"\nSegment {i} ({segment['start']:.1f}-{segment['end']:.1f}s):")
    print(f"  gaze_variance: {segment.get('gaze_variance', 'MISSING'):.6f}")

# Closing
closing = result['temporal_windows']['closing']
print(f"\nClosing ({closing['start']}-{closing['end']}s):")
print(f"  gaze_variance: {closing.get('gaze_variance', 'MISSING'):.6f}")

# Summary
all_variances = [hook.get('gaze_variance', 0)]
all_variances.extend([s.get('gaze_variance', 0) for s in result['temporal_windows']['middle_segments']])
all_variances.append(closing.get('gaze_variance', 0))

print(f"\n📈 Summary:")
print(f"  Average variance: {sum(all_variances)/len(all_variances):.6f}")
print(f"  Min variance: {min(all_variances):.6f}")
print(f"  Max variance: {max(all_variances):.6f}")

if all(v > 0 for v in all_variances):
    print("\n✅ SUCCESS: All windows have non-zero gaze variance!")
else:
    zeros = sum(1 for v in all_variances if v == 0)
    print(f"\n⚠️  {zeros} windows have zero variance (might lack sufficient gaze data)")