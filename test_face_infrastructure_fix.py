#!/usr/bin/env python3
"""Test the face infrastructure fix (Phase 1 of personframingfix.md)."""

import json
import sys
from pathlib import Path

def test_face_data_sources():
    """Verify face data comes from timeline, not direct ML extraction."""

    # Load a test video with faces
    test_file = Path('unified_analysis/7430952519439846698.json')
    if not test_file.exists():
        print(f"❌ Test file {test_file} not found")
        return False

    with open(test_file) as f:
        data = json.load(f)

    # Count faces from both sources
    timeline_entries = data.get('timeline', {}).get('entries', [])
    timeline_faces = [e for e in timeline_entries if e.get('entry_type') == 'face']
    ml_faces = data.get('ml_data', {}).get('mediapipe', {}).get('faces', [])

    print(f"Timeline has: {len(timeline_faces)} face entries")
    print(f"ML data has: {len(ml_faces)} faces")

    # Import and test the extraction
    from rumiai_v2.processors.temporal_compute import extract_timelines_for_temporal

    timelines = extract_timelines_for_temporal(data)
    face_timeline = timelines.get('face_timeline', [])

    print(f"\nExtracted face_timeline has: {len(face_timeline)} faces")

    # Verify structure (should only have timestamp and bbox)
    if face_timeline:
        first_face = face_timeline[0]
        fields = list(first_face.keys())
        print(f"Face structure: {fields}")

        # Check we removed unused fields
        if 'confidence' in fields:
            print("❌ ERROR: confidence field still present (should be removed)")
            return False
        if 'frame_number' in fields:
            print("❌ ERROR: frame_number field present (should not be from ML)")
            return False
        if 'count' in fields:
            print("❌ ERROR: count field present (should not be from ML)")
            return False

        # Check we have required fields
        if 'timestamp' not in fields:
            print("❌ ERROR: timestamp field missing")
            return False
        if 'bbox' not in fields:
            print("❌ ERROR: bbox field missing")
            return False

        print("✓ Face structure correct: only timestamp and bbox")

    # Test fail-fast validation with mock data
    print("\n=== Testing fail-fast validation ===")

    # Create test case where timeline_builder missed faces
    bad_data = {
        'timeline': {'entries': []},  # No face entries!
        'ml_data': {
            'mediapipe': {
                'faces': [{'timestamp': 0, 'bbox': {'x': 0.3, 'y': 0.3, 'width': 0.2, 'height': 0.3}}]
            }
        }
    }

    try:
        timelines = extract_timelines_for_temporal(bad_data)
        print("❌ ERROR: Should have raised ValueError for missing faces")
        return False
    except ValueError as e:
        if "Timeline builder missing" in str(e):
            print(f"✓ Fail-fast validation working: {e}")
        else:
            print(f"❌ Wrong error: {e}")
            return False

    print("\n✅ All Phase 1 tests passed!")
    return True

if __name__ == "__main__":
    success = test_face_data_sources()
    sys.exit(0 if success else 1)