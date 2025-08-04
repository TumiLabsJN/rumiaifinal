#!/usr/bin/env python3
"""Debug the scene pacing compute function."""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions_full import parse_timestamp_to_seconds

def test_parse_timestamp():
    """Test the timestamp parsing function."""
    test_timestamps = ['1s', '2s', '10s', '1.5s', '60s', '1-2s', '0s']
    
    print("Testing parse_timestamp_to_seconds:")
    for ts in test_timestamps:
        result = parse_timestamp_to_seconds(ts)
        print(f"  '{ts}' -> {result}")

def debug_scene_timeline():
    """Debug what's happening in the compute function."""
    # Create a simple scene timeline
    scene_timeline = {
        '0s': {'type': 'scene_change', 'scene_index': 1},
        '5s': {'type': 'scene_change', 'scene_index': 2},
        '10s': {'type': 'scene_change', 'scene_index': 3},
        '15s': {'type': 'scene_change', 'scene_index': 4}
    }
    
    print("\n\nDebugging scene timeline processing:")
    print(f"Scene timeline: {scene_timeline}")
    
    # Simulate what compute_scene_pacing_metrics does
    timestamps = sorted(scene_timeline.keys(), key=lambda x: parse_timestamp_to_seconds(x) or 0)
    print(f"\nSorted timestamps: {timestamps}")
    
    scene_changes = []
    shot_durations = []
    video_duration = 20.0
    
    for i in range(len(timestamps)):
        current_time = parse_timestamp_to_seconds(timestamps[i])
        print(f"\nProcessing timestamp {i}: '{timestamps[i]}' -> {current_time}s")
        
        if current_time is None:
            print("  Skipping: could not parse timestamp")
            continue
            
        # Get next timestamp or use video duration
        if i < len(timestamps) - 1:
            next_time = parse_timestamp_to_seconds(timestamps[i + 1])
            if next_time is not None:
                duration = next_time - current_time
                shot_durations.append(duration)
                print(f"  Shot duration: {duration}s (until next cut)")
        else:
            # Last shot duration
            duration = video_duration - current_time
            if duration > 0:
                shot_durations.append(duration)
                print(f"  Last shot duration: {duration}s (until video end)")
        
        # Track scene change data
        scene_data = scene_timeline[timestamps[i]]
        print(f"  Scene data: {scene_data}")
        
        # Check if it's a scene change
        if scene_data.get('type') in ['scene_change', 'shot_change'] or scene_data.get('scene_change'):
            scene_changes.append({'timestamp': timestamps[i], 'time': current_time})
            print("  ✓ Added to scene_changes")
        else:
            print("  ✗ Not recognized as scene change")
    
    print(f"\nFinal results:")
    print(f"  Scene changes detected: {len(scene_changes)}")
    print(f"  Shot durations: {shot_durations}")
    print(f"  Total shots: {len(scene_changes) + 1}")

if __name__ == "__main__":
    test_parse_timestamp()
    debug_scene_timeline()