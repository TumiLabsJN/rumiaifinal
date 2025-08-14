#!/usr/bin/env python3
"""
Comprehensive analysis of face bbox data flow in personTimeline

This script traces the complete data flow from MediaPipe faces to personTimeline
to verify that bbox data is correctly processed at every step.
"""

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

def analyze_face_bbox_flow():
    """Complete analysis of face bbox data flow"""
    print("="*80)
    print("FACE BBOX DATA FLOW ANALYSIS")
    print("="*80)
    
    # Load data
    video_id = "7274651255392210219"
    unified_path = f"unified_analysis/{video_id}.json"
    
    with open(unified_path) as f:
        data = json.load(f)
    
    duration = data.get('video_metadata', {}).get('duration', 58)
    
    print(f"Video: {video_id}")
    print(f"Duration: {duration} seconds")
    print()
    
    # ANALYSIS 1: MediaPipe Raw Data
    print("ANALYSIS 1: MediaPipe Raw Data")
    print("-" * 50)
    
    ml_data = data.get('ml_data', {})
    mediapipe_data = ml_data.get('mediapipe', {})
    raw_faces = mediapipe_data.get('faces', [])
    
    print(f"Raw MediaPipe faces: {len(raw_faces)}")
    
    if raw_faces:
        # Analyze face data quality
        valid_faces = 0
        total_face_area = 0
        confidence_sum = 0
        
        for face in raw_faces:
            bbox = face.get('bbox', {})
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            confidence = face.get('confidence', 0)
            
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                if 0 <= width <= 1 and 0 <= height <= 1:
                    valid_faces += 1
                    face_area = width * height * 100
                    total_face_area += face_area
                    confidence_sum += confidence
        
        avg_face_area = total_face_area / valid_faces if valid_faces > 0 else 0
        avg_confidence = confidence_sum / valid_faces if valid_faces > 0 else 0
        
        print(f"Valid faces: {valid_faces}/{len(raw_faces)}")
        print(f"Average face area: {avg_face_area:.2f}%")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Show time range
        timestamps = [f.get('timestamp', 0) for f in raw_faces]
        print(f"Time range: {min(timestamps):.2f}s - {max(timestamps):.2f}s")
    
    # ANALYSIS 2: Extracted MediaPipe Data
    print(f"\nANALYSIS 2: extract_mediapipe_data() Function")
    print("-" * 50)
    
    extracted = extract_mediapipe_data(ml_data)
    extracted_faces = extracted.get('faces', [])
    
    print(f"Extracted faces: {len(extracted_faces)}")
    print(f"Extraction success: {len(extracted_faces) == len(raw_faces)}")
    
    # ANALYSIS 3: Timeline Building 
    print(f"\nANALYSIS 3: _extract_timelines_from_analysis() Function")
    print("-" * 50)
    
    timelines = _extract_timelines_from_analysis(data)
    person_timeline = timelines.get('personTimeline', {})
    
    print(f"personTimeline entries: {len(person_timeline)}")
    
    # Analyze personTimeline quality
    entries_with_bbox = 0
    entries_with_confidence = 0
    timeline_face_areas = []
    
    for timestamp, person_data in person_timeline.items():
        if person_data.get('face_bbox'):
            entries_with_bbox += 1
            bbox = person_data['face_bbox']
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            if width > 0 and height > 0:
                face_area = width * height * 100
                timeline_face_areas.append(face_area)
        
        if person_data.get('face_confidence'):
            entries_with_confidence += 1
    
    print(f"Entries with face_bbox: {entries_with_bbox}")
    print(f"Entries with face_confidence: {entries_with_confidence}")
    
    if timeline_face_areas:
        avg_timeline_area = sum(timeline_face_areas) / len(timeline_face_areas)
        print(f"Average face area in timeline: {avg_timeline_area:.2f}%")
    
    # ANALYSIS 4: Temporal Framing Calculation
    print(f"\nANALYSIS 4: calculate_temporal_framing_simple() Function")
    print("-" * 50)
    
    framing_timeline = calculate_temporal_framing_simple(person_timeline, duration)
    
    print(f"Framing timeline entries: {len(framing_timeline)}")
    
    # Analyze framing quality
    shot_counts = {}
    face_size_stats = []
    
    for entry in framing_timeline.values():
        shot_type = entry['shot_type']
        face_size = entry['face_size']
        
        shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
        if face_size > 0:
            face_size_stats.append(face_size)
    
    print(f"Shot type distribution: {shot_counts}")
    print(f"Valid face sizes: {len(face_size_stats)}")
    
    if face_size_stats:
        print(f"Face size range: {min(face_size_stats):.2f}% - {max(face_size_stats):.2f}%")
        print(f"Average face size: {sum(face_size_stats)/len(face_size_stats):.2f}%")
    
    # ANALYSIS 5: Person Framing Metrics
    print(f"\nANALYSIS 5: compute_person_framing_metrics() Function")
    print("-" * 50)
    
    # Create minimal timelines for testing
    metrics = compute_person_framing_metrics(
        expression_timeline={},
        object_timeline={}, 
        camera_distance_timeline={},
        person_timeline=person_timeline,
        enhanced_human_data={},
        duration=duration,
        gaze_timeline={}
    )
    
    print(f"Metrics computed successfully: {bool(metrics)}")
    
    relevant_metrics = [
        'face_screen_time_ratio',
        'person_screen_time_ratio', 
        'average_camera_distance',
        'dominant_shot_type'
    ]
    
    for metric in relevant_metrics:
        value = metrics.get(metric, 'N/A')
        print(f"{metric}: {value}")
    
    # FINAL ANALYSIS
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS")
    print("="*80)
    
    data_flow_success = all([
        len(raw_faces) > 0,
        len(extracted_faces) == len(raw_faces),
        entries_with_bbox > 0,
        len(framing_timeline) > 0,
        len(face_size_stats) > 0,
        bool(metrics)
    ])
    
    print(f"🔍 MediaPipe Detections: {len(raw_faces)} faces")
    print(f"📊 Timeline Integration: {entries_with_bbox} entries with bbox")
    print(f"🎯 Framing Calculation: {len(face_size_stats)} valid face sizes") 
    print(f"📈 Metrics Generation: {'Success' if metrics else 'Failed'}")
    print(f"\n{'🎉 FACE BBOX DATA FLOW: WORKING CORRECTLY!' if data_flow_success else '❌ FACE BBOX DATA FLOW: ISSUES DETECTED!'}")
    
    return data_flow_success

if __name__ == "__main__":
    success = analyze_face_bbox_flow()
    sys.exit(0 if success else 1)