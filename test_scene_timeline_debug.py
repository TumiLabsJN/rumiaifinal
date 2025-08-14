#!/usr/bin/env python3
"""
Debug scene timeline extraction issue
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.core.models.analysis import UnifiedAnalysis

# Load the actual analysis for video 7460652578183974166
analysis_path = Path('unified_analysis/7460652578183974166.json')
if analysis_path.exists():
    import json
    with open(analysis_path, 'r') as f:
        analysis_dict = json.load(f)
    
    # Extract timelines
    from rumiai_v2.processors.precompute_functions import _extract_timelines_from_analysis
    
    timelines = _extract_timelines_from_analysis(analysis_dict)
    
    print("=== TIMELINE EXTRACTION DEBUG ===")
    print(f"sceneChangeTimeline keys: {list(timelines.get('sceneChangeTimeline', {}).keys())}")
    print(f"sceneChangeTimeline content: {timelines.get('sceneChangeTimeline', {})}")
    print(f"sceneTimeline keys: {list(timelines.get('sceneTimeline', {}).keys())}")
    print(f"sceneTimeline content: {timelines.get('sceneTimeline', {})}")
    
    # Check what scene entries exist in the timeline
    timeline_entries = analysis_dict.get('timeline', {}).get('entries', [])
    scene_entries = [e for e in timeline_entries if e.get('entry_type') in ['scene', 'scene_change']]
    
    print(f"\n=== TIMELINE ENTRIES DEBUG ===")
    print(f"Total timeline entries: {len(timeline_entries)}")
    print(f"Scene-related entries: {len(scene_entries)}")
    for i, entry in enumerate(scene_entries[:5]):  # Show first 5
        print(f"Entry {i+1}: {entry}")
    
else:
    print(f"Analysis file not found: {analysis_path}")