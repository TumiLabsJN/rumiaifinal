#!/usr/bin/env python3
"""Test the scene pacing fix."""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions import compute_scene_pacing_wrapper, _extract_timelines_from_analysis
from rumiai_v2.core.models import UnifiedAnalysis

def test_scene_pacing():
    # Load the unified analysis
    with open('unified_analysis/7320300319580146987.json', 'r') as f:
        analysis_data = json.load(f)
    
    print("Testing scene pacing extraction...")
    print(f"Video duration: {analysis_data['timeline']['duration']}s")
    
    # Count scene changes in timeline
    scene_change_count = 0
    for entry in analysis_data['timeline']['entries']:
        if entry['entry_type'] == 'scene_change':
            scene_change_count += 1
    
    print(f"Scene changes in timeline: {scene_change_count}")
    
    # Debug: Extract timelines and check
    print("\nDebugging timeline extraction...")
    timelines = _extract_timelines_from_analysis(analysis_data)
    print(f"SceneChangeTimeline entries: {len(timelines.get('sceneChangeTimeline', {}))}")
    if timelines.get('sceneChangeTimeline'):
        print("First few scene changes:")
        for i, (timestamp, data) in enumerate(list(timelines['sceneChangeTimeline'].items())[:5]):
            print(f"  {timestamp}: {data}")
    
    # Import and test the actual compute function  
    try:
        from rumiai_v2.processors.precompute_functions_full import compute_scene_pacing_metrics
        print("\nTesting direct compute_scene_pacing_metrics...")
        scene_timeline = timelines.get('sceneChangeTimeline', {})
        result_direct = compute_scene_pacing_metrics(
            scene_timeline=scene_timeline,
            video_duration=60.0,
            object_timeline={},
            camera_distance_timeline={},
            video_id='7320300319580146987'
        )
        print(f"Direct result - total_shots: {result_direct.get('total_shots', 'N/A')}")
    except Exception as e:
        print(f"Could not test direct function: {e}")
    
    # Test the wrapper
    try:
        result = compute_scene_pacing_wrapper(analysis_data)
        print(f"\nCompute function result:")
        print(f"- Total shots: {result.get('total_shots', 'N/A')}")
        print(f"- Shots per minute: {result.get('shots_per_minute', 'N/A')}")
        print(f"- Average shot duration: {result.get('avg_shot_duration', 'N/A')}s")
        print(f"- Shortest shot: {result.get('shortest_shot', 'N/A')}s")
        print(f"- Longest shot: {result.get('longest_shot', 'N/A')}s")
        
        if result.get('total_shots', 0) == 1:
            print("\n❌ ISSUE: Still seeing only 1 shot/scene!")
            print("\nDebugging info:")
            print(f"- Pacing classification: {result.get('pacing_classification', 'N/A')}")
            print(f"- Data completeness: {result.get('data_completeness', 'N/A')}")
            print(f"- Scene detection rate: {result.get('scene_detection_rate', 'N/A')}")
        else:
            print(f"\n✅ SUCCESS: Correctly detected {result.get('total_shots', 0)} shots/scenes!")
            
    except Exception as e:
        print(f"\n❌ Error running compute function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scene_pacing()