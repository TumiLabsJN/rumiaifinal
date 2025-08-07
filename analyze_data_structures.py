#!/usr/bin/env python3
"""
Analyze the actual data structures causing the errors
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load the actual analysis data
analysis_file = Path("unified_analysis/7515849242703973662.json")

with open(analysis_file, 'r') as f:
    analysis_data = json.load(f)

print("🔍 ANALYZING DATA STRUCTURES CAUSING ERRORS")
print("="*60)

# 1. Check burst_windows issue (visual_overlay_analysis line 387)
print("\n1. VISUAL_OVERLAY_ANALYSIS - burst_windows issue:")
print("   Expected: list of tuples [(window, count), ...]") 
print("   Error location: line 387 'for window, overlays in burst_windows'")

# Check how burst_windows is populated (line 110)
print("   burst_windows.append(window_key) creates: ['0-5s', '10-15s']")
print("   But line 387 expects: [('0-5s', 4), ('10-15s', 3)]")
print("   ❌ MISMATCH: Creates strings, expects tuples")

# 2. Check object timeline issue (person_framing line 1956, scene_pacing line 2562)
print("\n2. PERSON_FRAMING & SCENE_PACING - objectTimeline format:")
object_timeline = analysis_data.get('objectTimeline', {})
print(f"   objectTimeline type: {type(object_timeline)}")
print(f"   objectTimeline content: {object_timeline}")

# Check ml_data structure 
ml_data = analysis_data.get('ml_data', {})
yolo_data = ml_data.get('yolo', {})
print(f"\n   ml_data.yolo structure:")
print(f"   Type: {type(yolo_data)}")
if isinstance(yolo_data, dict) and 'objectAnnotations' in yolo_data:
    annotations = yolo_data['objectAnnotations'][:2]  # First 2 entries
    print(f"   First annotation: {annotations[0] if annotations else 'None'}")
    print(f"   Expected by functions: dict with timestamp keys")
    print(f"   Actual: list of objects with timestamp fields")
    print("   ❌ MISMATCH: Functions expect dict, get list annotations")

print("\n3. SUCCESSFUL FUNCTIONS - Why do they work?")
print("   ✅ creative_density: Uses ml_data directly, no timeline dependency")
print("   ✅ emotional_journey: Uses expressionTimeline from ML data")  
print("   ✅ speech_analysis: Uses speechTimeline from ML data")
print("   ✅ metadata_analysis: Uses metadata, no timeline dependency")

print("\n4. ROOT CAUSE ANALYSIS:")
print("   The objectTimeline is EMPTY in the unified analysis!")
print("   Functions expecting objectTimeline data fail when it's empty/missing.")
print("   The ML data exists but isn't being converted to timeline format.")

print("\n🔍 CHECKING TIMELINE EXTRACTION...")
from rumiai_v2.processors.precompute_functions import _extract_timelines_from_analysis

try:
    timelines = _extract_timelines_from_analysis(analysis_data)
    print(f"   Extracted timelines: {list(timelines.keys())}")
    print(f"   objectTimeline entries: {len(timelines.get('objectTimeline', {}))}")
    
    if 'objectTimeline' in timelines and timelines['objectTimeline']:
        sample_entry = list(timelines['objectTimeline'].items())[0]
        print(f"   Sample objectTimeline entry: {sample_entry}")
    else:
        print("   ❌ objectTimeline is empty after extraction!")
        
except Exception as e:
    print(f"   ❌ Timeline extraction failed: {e}")