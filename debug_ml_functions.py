#!/usr/bin/env python3
"""
Debug script to test ML functions and understand which ones actually execute successfully
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions import (
    compute_creative_density_wrapper,
    compute_emotional_wrapper,
    compute_speech_wrapper,
    compute_visual_overlay_wrapper,
    compute_metadata_wrapper,
    compute_person_framing_wrapper, 
    compute_scene_pacing_wrapper
)

# Load the actual analysis data for video 7515849242703973662
analysis_file = Path("unified_analysis/7515849242703973662.json")

if analysis_file.exists():
    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)
    print(f"✅ Loaded analysis data for video 7515849242703973662")
    print(f"   - Video duration: {analysis_data.get('duration', 'N/A')}s")
    print(f"   - objectTimeline entries: {len(analysis_data.get('objectTimeline', {}))}")
    print(f"   - ML data available: {analysis_data.get('ml_data', {}).keys()}")
else:
    print("❌ Analysis file not found")
    sys.exit(1)

# Test each function
functions_to_test = [
    ("creative_density", compute_creative_density_wrapper),
    ("emotional_journey", compute_emotional_wrapper),
    ("speech_analysis", compute_speech_wrapper),
    ("visual_overlay_analysis", compute_visual_overlay_wrapper),
    ("metadata_analysis", compute_metadata_wrapper),
    ("person_framing", compute_person_framing_wrapper), 
    ("scene_pacing", compute_scene_pacing_wrapper)
]

print("\n" + "="*60)
print("TESTING ML FUNCTIONS")
print("="*60)

results = {}
for func_name, func in functions_to_test:
    print(f"\n🔍 Testing {func_name}...")
    try:
        result = func(analysis_data)
        print(f"   ✅ SUCCESS - Function executed without errors")
        print(f"   📊 Returned {len(result)} metrics")
        results[func_name] = "SUCCESS"
        
        # Show a few sample keys from result
        sample_keys = list(result.keys())[:5]
        print(f"   📝 Sample metrics: {sample_keys}")
        
    except Exception as e:
        print(f"   ❌ ERROR - {str(e)}")
        print(f"   🔍 Error type: {type(e).__name__}")
        
        # Get more detailed traceback for debugging
        import traceback
        tb_str = traceback.format_exc()
        # Find the relevant line numbers
        relevant_lines = [line for line in tb_str.split('\n') if 'precompute_functions' in line]
        if relevant_lines:
            print(f"   📍 Location: {relevant_lines[-1].strip()}")
        
        results[func_name] = f"ERROR: {str(e)}"

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for func_name, status in results.items():
    status_emoji = "✅" if status == "SUCCESS" else "❌"
    print(f"{status_emoji} {func_name}: {status}")

# Check what functions should be working based on 4/7 success rate
successful_count = sum(1 for status in results.values() if status == "SUCCESS")
print(f"\n📈 Success rate: {successful_count}/7 functions tested")

if successful_count == 4:
    print("🎯 Matches expected 4/7 success rate!")
elif successful_count == 7:
    print("🎉 All functions are working!")
elif successful_count == 0:
    print("🚨 All functions are broken!")
else:
    print(f"⚠️ Mixed results - {successful_count} work, {7-successful_count} fail")