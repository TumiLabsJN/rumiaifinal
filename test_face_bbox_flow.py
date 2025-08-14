#!/usr/bin/env python3
"""Test the complete face bbox data flow from MediaPipe to personTimeline to framing metrics."""

import json
import sys
from pathlib import Path

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions import (
    extract_mediapipe_data,
    _extract_timelines_from_analysis
)
from rumiai_v2.processors.precompute_functions_full import (
    compute_person_framing_metrics,
    calculate_temporal_framing_simple
)

def test_face_bbox_data_flow():
    """Test the complete face bbox data flow"""
    print("="*60)
    print("TESTING FACE BBOX DATA FLOW")
    print("="*60)
    
    # Load the unified analysis
    video_id = "7274651255392210219"
    unified_path = f"unified_analysis/{video_id}.json"
    
    print(f"Loading data from: {unified_path}")
    with open(unified_path) as f:
        data = json.load(f)
    
    # Step 1: Extract raw MediaPipe data
    print("\nStep 1: Extract raw MediaPipe data")
    print("-" * 40)
    mediapipe_data = data.get('ml_data', {}).get('mediapipe', {})
    extracted = extract_mediapipe_data(mediapipe_data)
    
    print(f"Raw MediaPipe faces: {len(extracted.get('faces', []))}")
    
    # Show first few faces
    faces = extracted.get('faces', [])
    if faces:
        for i, face in enumerate(faces[:3]):
            print(f"  Face {i}: timestamp={face.get('timestamp')}, confidence={face.get('confidence', 0):.3f}")
            bbox = face.get('bbox', {})
            print(f"    bbox: x={bbox.get('x', 0):.3f}, y={bbox.get('y', 0):.3f}, w={bbox.get('width', 0):.3f}, h={bbox.get('height', 0):.3f}")
            face_area = bbox.get('width', 0) * bbox.get('height', 0) * 100
            print(f"    face_area: {face_area:.2f}%")
    
    # Step 2: Extract timelines (including personTimeline)
    print("\nStep 2: Build personTimeline")
    print("-" * 40)
    timelines = _extract_timelines_from_analysis(data)
    
    person_timeline = timelines.get('personTimeline', {})
    print(f"personTimeline entries: {len(person_timeline)}")
    
    # Count entries with face_bbox
    entries_with_bbox = 0
    total_face_area = 0
    for timestamp, person_data in person_timeline.items():
        if person_data.get('face_bbox'):
            entries_with_bbox += 1
            bbox = person_data['face_bbox']
            face_area = bbox.get('width', 0) * bbox.get('height', 0) * 100
            total_face_area += face_area
    
    print(f"Entries with face_bbox: {entries_with_bbox}")
    avg_face_area = total_face_area / entries_with_bbox if entries_with_bbox > 0 else 0
    print(f"Average face area: {avg_face_area:.2f}%")
    
    # Step 3: Test calculate_temporal_framing_simple
    print("\nStep 3: Calculate temporal framing")
    print("-" * 40)
    duration = data.get('video_metadata', {}).get('duration', 58)
    framing_timeline = calculate_temporal_framing_simple(person_timeline, duration)
    
    print(f"Framing timeline entries: {len(framing_timeline)}")
    
    # Count face_size > 0
    valid_face_sizes = 0
    for entry in framing_timeline.values():
        if entry['face_size'] > 0:
            valid_face_sizes += 1
    
    print(f"Entries with face_size > 0: {valid_face_sizes}")
    
    # Step 4: Test compute_person_framing_metrics
    print("\nStep 4: Compute person framing metrics")
    print("-" * 40)
    
    # Create dummy timelines for the other parameters
    expression_timeline = {}
    object_timeline = {}
    camera_distance_timeline = {}
    enhanced_human_data = {}
    gaze_timeline = {}
    
    metrics = compute_person_framing_metrics(
        expression_timeline=expression_timeline,
        object_timeline=object_timeline, 
        camera_distance_timeline=camera_distance_timeline,
        person_timeline=person_timeline,
        enhanced_human_data=enhanced_human_data,
        duration=duration,
        gaze_timeline=gaze_timeline
    )
    
    # Check if face sizes were computed
    print(f"Person framing metrics computed successfully")
    print(f"Face screen time ratio: {metrics.get('face_screen_time_ratio', 0):.3f}")
    print(f"Person screen time ratio: {metrics.get('person_screen_time_ratio', 0):.3f}")
    
    # Check average camera distance calculation
    avg_camera_distance = metrics.get('average_camera_distance', 'unknown')
    dominant_shot_type = metrics.get('dominant_shot_type', 'unknown')
    print(f"Average camera distance: {avg_camera_distance}")
    print(f"Dominant shot type: {dominant_shot_type}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ MediaPipe faces extracted: {len(faces)}")
    print(f"✅ personTimeline entries: {len(person_timeline)}")
    print(f"✅ Entries with face_bbox: {entries_with_bbox}")
    print(f"✅ Framing timeline entries: {len(framing_timeline)}")
    print(f"✅ Valid face sizes: {valid_face_sizes}")
    print(f"✅ Person framing metrics computed: {bool(metrics)}")
    
    if entries_with_bbox == 0:
        print("❌ NO FACE BBOX DATA FOUND!")
        return False
    elif valid_face_sizes == 0:
        print("❌ NO VALID FACE SIZES CALCULATED!")
        return False
    else:
        print("🎉 FACE BBOX DATA FLOW IS WORKING!")
        return True

if __name__ == "__main__":
    success = test_face_bbox_data_flow()
    sys.exit(0 if success else 1)