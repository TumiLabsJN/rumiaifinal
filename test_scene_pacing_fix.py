#!/usr/bin/env python3
"""
Test scene pacing computation with fixed timeline data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Test the scene pacing computation
from rumiai_v2.processors.precompute_functions import compute_scene_pacing_wrapper
import json

# Load the actual analysis
analysis_path = Path('unified_analysis/7460652578183974166.json')
if analysis_path.exists():
    with open(analysis_path, 'r') as f:
        analysis_dict = json.load(f)
    
    try:
        result = compute_scene_pacing_wrapper(analysis_dict)
        print("✅ Scene pacing computation SUCCESS!")
        
        print(f"Full result: {result}")
        
        # Check if we now have scene data
        core_metrics = result.get('CoreMetrics', {})
        scene_core_metrics = result.get('scenePacingCoreMetrics', {})
        
        total_scenes = core_metrics.get('totalScenes', 0) or scene_core_metrics.get('totalScenes', 0)
        avg_duration = core_metrics.get('averageSceneDuration', 0) or scene_core_metrics.get('averageSceneDuration', 0)
        
        print(f"Total scenes detected: {total_scenes}")
        print(f"Average scene duration: {avg_duration:.2f}s")
        
        if total_scenes > 0:
            print("🎉 Scene pacing now has real data!")
        else:
            print("❌ Scene pacing still shows 0 scenes")
            
    except Exception as e:
        print(f"❌ Scene pacing failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Analysis file not found")