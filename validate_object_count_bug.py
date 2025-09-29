#!/usr/bin/env python3
"""
Validate that the object_count bug exists in current implementation
"""
import json
import sys

def extract_instance_id(track_id):
    """Extract instance ID from YOLO trackId"""
    if not track_id or '_' not in track_id:
        return None
    parts = track_id.split('_')
    if len(parts) == 3 and parts[0] == 'obj':
        return parts[2]
    return None

def validate_object_count_bug(unified_json_path, temporal_json_path):
    """Check if object_count is incorrectly including persons"""

    # Load unified analysis (has YOLO raw data)
    with open(unified_json_path) as f:
        unified_data = json.load(f)

    # Load temporal windows output
    with open(temporal_json_path) as f:
        temporal_data = json.load(f)

    # Get YOLO detections
    yolo_objects = unified_data.get('ml_data', {}).get('yolo', {}).get('objectAnnotations', [])

    # Separate person and non-person detections
    person_detections = [o for o in yolo_objects if o['className'] == 'person']
    other_detections = [o for o in yolo_objects if o['className'] != 'person']

    # Count unique instances
    unique_person_ids = set()
    unique_other_ids = set()
    all_unique_ids = set()

    for obj in yolo_objects:
        instance_id = extract_instance_id(obj.get('trackId', ''))
        if instance_id:
            all_unique_ids.add(instance_id)
            if obj['className'] == 'person':
                unique_person_ids.add(instance_id)
            else:
                unique_other_ids.add(instance_id)

    # Get temporal window counts
    hook = temporal_data['temporal_windows']['hook']
    reported_object_count = hook['object_count']
    reported_person_count = hook['person_count']

    print("=" * 60)
    print("OBJECT COUNT BUG VALIDATION")
    print("=" * 60)
    print("\n📊 YOLO Detection Analysis:")
    print(f"  Total unique objects detected: {len(all_unique_ids)}")
    print(f"  Unique persons: {len(unique_person_ids)}")
    print(f"  Unique non-person objects: {len(unique_other_ids)}")

    print("\n📈 Temporal Window Counts (Hook):")
    print(f"  Reported object_count: {reported_object_count}")
    print(f"  Reported person_count: {reported_person_count}")

    print("\n🔍 Bug Detection:")

    # Check if bug exists
    if len(unique_other_ids) == 0 and reported_object_count > 0:
        print("  ❌ BUG CONFIRMED: object_count is non-zero when only persons exist!")
        print(f"     Expected object_count: 0 (no non-person objects)")
        print(f"     Got object_count: {reported_object_count}")
        print("\n  🐛 Double-counting detected: Persons are being counted in object_count")
    elif reported_object_count == len(all_unique_ids):
        print("  ❌ BUG CONFIRMED: object_count includes all objects (including persons)")
        print(f"     This causes double-counting when persons are present")
        print(f"     object_count should be: {len(unique_other_ids)} (non-person objects only)")
    elif reported_object_count == len(unique_other_ids):
        print("  ✅ NO BUG: object_count correctly excludes persons")
    else:
        print("  ⚠️ UNEXPECTED: object_count doesn't match any expected pattern")
        print(f"     Investigate manually")

    print("\n📋 Class Distribution:")
    class_counts = {}
    for obj in yolo_objects:
        class_name = obj['className']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} detections")

    return len(unique_other_ids) == 0 and reported_object_count > 0

if __name__ == "__main__":
    unified_path = "/home/jorge/rumiaifinal/unified_analysis/493430654043793.json"
    temporal_path = "/home/jorge/rumiaifinal/insights/493430654043793_temporal_windows_updated.json"

    bug_exists = validate_object_count_bug(unified_path, temporal_path)

    print("\n" + "=" * 60)
    if bug_exists:
        print("RESULT: Bug exists - fix needed!")
        sys.exit(1)
    else:
        print("RESULT: No bug detected")
        sys.exit(0)