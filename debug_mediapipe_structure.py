#!/usr/bin/env python3
"""Debug the MediaPipe data structure to understand why faces aren't being extracted."""

import json
import sys
from pathlib import Path

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_mediapipe_structure():
    """Debug the MediaPipe data structure"""
    video_id = "7274651255392210219"
    unified_path = f"unified_analysis/{video_id}.json"
    
    print(f"Loading data from: {unified_path}")
    with open(unified_path) as f:
        data = json.load(f)
    
    # Check the ml_data structure
    ml_data = data.get('ml_data', {})
    print(f"ml_data keys: {list(ml_data.keys())}")
    
    # Check mediapipe structure
    mediapipe_data = ml_data.get('mediapipe', {})
    print(f"mediapipe keys: {list(mediapipe_data.keys())}")
    
    # Check if there's a 'data' nested key
    if 'data' in mediapipe_data:
        nested_data = mediapipe_data['data']
        print(f"mediapipe.data keys: {list(nested_data.keys())}")
        faces = nested_data.get('faces', [])
        print(f"mediapipe.data.faces count: {len(faces)}")
    
    # Check direct access
    faces_direct = mediapipe_data.get('faces', [])
    print(f"mediapipe.faces (direct) count: {len(faces_direct)}")
    
    # Show a sample face if found
    if faces_direct:
        print(f"Sample face (direct): {faces_direct[0]}")
    elif 'data' in mediapipe_data and mediapipe_data['data'].get('faces'):
        print(f"Sample face (nested): {mediapipe_data['data']['faces'][0]}")
    
    print("\nNow testing extract_mediapipe_data function:")
    
    # Test the extract function
    from rumiai_v2.processors.precompute_functions import extract_mediapipe_data
    extracted = extract_mediapipe_data(ml_data)
    print(f"Extracted faces: {len(extracted.get('faces', []))}")
    print(f"Extracted poses: {len(extracted.get('poses', []))}")
    
    if extracted.get('faces'):
        print(f"First extracted face: {extracted['faces'][0]}")

if __name__ == "__main__":
    debug_mediapipe_structure()