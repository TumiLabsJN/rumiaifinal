#!/usr/bin/env python3
"""
Detailed debug of scene pacing computation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions import _extract_timelines_from_analysis
from rumiai_v2.processors.precompute_functions_full import compute_scene_pacing_metrics
import json

# Load the actual analysis
analysis_path = Path('unified_analysis/7460652578183974166.json')
if analysis_path.exists():
    with open(analysis_path, 'r') as f:
        analysis_dict = json.load(f)
    
    # Extract timelines
    timelines = _extract_timelines_from_analysis(analysis_dict)
    
    print("=== SCENE PACING DEBUG ===")
    scene_timeline = timelines.get('sceneChangeTimeline', {})
    video_duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    print(f"scene_timeline: {scene_timeline}")
    print(f"video_duration: {video_duration}")
    print(f"scene_timeline keys: {list(scene_timeline.keys())}")
    
    # Test the actual function call
    try:
        result = compute_scene_pacing_metrics(
            scene_timeline=scene_timeline,
            video_duration=video_duration,
            object_timeline={},
            camera_distance_timeline={},
            video_id=analysis_dict.get('video_id')
        )
        
        print(f"Result: {result}")
        core_metrics = result.get('CoreMetrics', {})
        print(f"CoreMetrics: {core_metrics}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Analysis file not found")