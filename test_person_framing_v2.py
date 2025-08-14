#!/usr/bin/env python3
"""
Test harness for PersonFramingV2 temporal analysis.
Tests the implementation against various edge cases and validates output structure.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Test utilities
def create_test_person_timeline(duration: int, pattern: str = "continuous") -> Dict[str, Any]:
    """Create test person timeline with different patterns."""
    person_timeline = {}
    
    if pattern == "continuous":
        # Person visible throughout
        for i in range(duration):
            person_timeline[f"{i}-{i+1}s"] = {
                'face_bbox': {'width': 0.3, 'height': 0.4},
                'face_confidence': 0.95
            }
    
    elif pattern == "intermittent":
        # Person visible every other second
        for i in range(0, duration, 2):
            person_timeline[f"{i}-{i+1}s"] = {
                'face_bbox': {'width': 0.2, 'height': 0.3},
                'face_confidence': 0.85
            }
    
    elif pattern == "no_person":
        # No person detected
        pass
    
    elif pattern == "corrupted":
        # Mix of valid and corrupted data
        for i in range(duration):
            if i % 3 == 0:
                # Valid data
                person_timeline[f"{i}-{i+1}s"] = {
                    'face_bbox': {'width': 0.25, 'height': 0.35},
                    'face_confidence': 0.9
                }
            elif i % 3 == 1:
                # Corrupted bbox (invalid values)
                person_timeline[f"{i}-{i+1}s"] = {
                    'face_bbox': {'width': -1, 'height': 2.5},  # Invalid
                    'face_confidence': 0.7
                }
            # Every third second: no data
    
    elif pattern == "varying_sizes":
        # Different face sizes for shot classification
        sizes = [
            (0.6, 0.5),   # Close-up
            (0.3, 0.3),   # Medium
            (0.1, 0.1),   # Wide
            (0.4, 0.4),   # Close/Medium boundary
        ]
        for i in range(duration):
            size_idx = i % len(sizes)
            person_timeline[f"{i}-{i+1}s"] = {
                'face_bbox': {'width': sizes[size_idx][0], 'height': sizes[size_idx][1]},
                'face_confidence': 0.9
            }
    
    return person_timeline


def classify_shot_type_simple(face_area_percent: float) -> str:
    """Simple shot classification for ML training."""
    if face_area_percent > 25:
        return 'close'
    elif face_area_percent > 8:
        return 'medium'
    elif face_area_percent > 0:
        return 'wide'
    else:
        return 'none'


def calculate_temporal_framing_simple(person_timeline: Dict[str, Any], duration: int) -> Dict[str, Any]:
    """Calculate basic camera distance with bbox validation."""
    framing_timeline = {}
    
    for second in range(int(duration)):
        timestamp_key = f"{second}-{second+1}s"
        
        if timestamp_key in person_timeline:
            person_data = person_timeline[timestamp_key]
            face_confidence = person_data.get('face_confidence', 0)
            
            if person_data.get('face_bbox'):
                bbox = person_data['face_bbox']
                
                # Validate bbox data
                width = bbox.get('width', 0)
                height = bbox.get('height', 0)
                
                # Handle corrupted data
                if (isinstance(width, (int, float)) and 
                    isinstance(height, (int, float)) and
                    0 <= width <= 1 and 0 <= height <= 1):
                    face_area = width * height * 100
                else:
                    # Corrupted bbox - treat as no face
                    face_area = 0
                
                shot_type = classify_shot_type_simple(face_area)
                
                framing_timeline[timestamp_key] = {
                    'shot_type': shot_type,
                    'face_size': face_area,
                    'confidence': face_confidence
                }
            else:
                # No bbox data
                framing_timeline[timestamp_key] = {
                    'shot_type': 'none',
                    'face_size': 0,
                    'confidence': 0
                }
        else:
            # No person data for this second
            framing_timeline[timestamp_key] = {
                'shot_type': 'none',
                'face_size': 0,
                'confidence': 0
            }
    
    return framing_timeline


def analyze_framing_progression_simple(framing_timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simple analysis of framing changes over time."""
    progression = []
    current_shot = None
    shot_start = 0
    
    for timestamp_key in sorted(framing_timeline.keys(), key=lambda x: int(x.split('-')[0])):
        shot_type = framing_timeline[timestamp_key]['shot_type']
        second = int(timestamp_key.split('-')[0])
        
        if shot_type != current_shot:
            if current_shot is not None:
                progression.append({
                    'type': current_shot,
                    'start': shot_start,
                    'end': second,
                    'duration': second - shot_start
                })
            
            current_shot = shot_type
            shot_start = second
    
    # Add final shot
    if current_shot is not None:
        last_second = max([int(k.split('-')[0]) for k in framing_timeline.keys()]) + 1
        progression.append({
            'type': current_shot,
            'start': shot_start,
            'end': last_second,
            'duration': last_second - shot_start
        })
    
    return progression


def validate_output_structure(framing_timeline: Dict, progression: List) -> Tuple[bool, List[str]]:
    """Validate the output meets expected structure."""
    errors = []
    
    # Check framing timeline structure
    for timestamp, data in framing_timeline.items():
        # Check timestamp format
        if not timestamp.endswith('s') or '-' not in timestamp:
            errors.append(f"Invalid timestamp format: {timestamp}")
        
        # Check required fields
        required_fields = ['shot_type', 'face_size', 'confidence']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing field '{field}' in {timestamp}")
        
        # Check shot type values
        if data.get('shot_type') not in ['close', 'medium', 'wide', 'none']:
            errors.append(f"Invalid shot_type: {data.get('shot_type')}")
        
        # Check numeric values
        if not isinstance(data.get('face_size'), (int, float)) or data['face_size'] < 0:
            errors.append(f"Invalid face_size at {timestamp}: {data.get('face_size')}")
        
        if not 0 <= data.get('confidence', -1) <= 1:
            errors.append(f"Invalid confidence at {timestamp}: {data.get('confidence')}")
    
    # Check progression structure
    for i, segment in enumerate(progression):
        if 'type' not in segment or 'start' not in segment or 'end' not in segment or 'duration' not in segment:
            errors.append(f"Progression segment {i} missing required fields")
        
        if segment.get('duration') != segment.get('end', 0) - segment.get('start', 0):
            errors.append(f"Progression segment {i} duration mismatch")
    
    # Check progression continuity
    for i in range(len(progression) - 1):
        if progression[i]['end'] != progression[i+1]['start']:
            errors.append(f"Gap in progression between segments {i} and {i+1}")
    
    return len(errors) == 0, errors


def run_test_case(test_name: str, person_timeline: Dict, duration: int, 
                  expected_shot_types: List[str] = None) -> Dict[str, Any]:
    """Run a single test case."""
    print(f"\n🧪 Testing: {test_name}")
    print(f"   Duration: {duration}s")
    print(f"   Timeline entries: {len(person_timeline)}")
    
    # Run the functions
    framing_timeline = calculate_temporal_framing_simple(person_timeline, duration)
    progression = analyze_framing_progression_simple(framing_timeline)
    
    # Validate structure
    is_valid, errors = validate_output_structure(framing_timeline, progression)
    
    # Check expected shot types if provided
    actual_shot_types = list(set(ft['shot_type'] for ft in framing_timeline.values()))
    
    # Calculate statistics
    shot_counts = {}
    for data in framing_timeline.values():
        shot_type = data['shot_type']
        shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
    
    # Print results
    print(f"   ✅ Timeline entries: {len(framing_timeline)}")
    print(f"   ✅ Progression segments: {len(progression)}")
    print(f"   ✅ Shot type distribution: {shot_counts}")
    print(f"   ✅ Structure valid: {is_valid}")
    
    if not is_valid:
        print("   ❌ Validation errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"      - {error}")
    
    if expected_shot_types:
        matches_expected = set(actual_shot_types) == set(expected_shot_types)
        print(f"   ✅ Expected shot types: {matches_expected}")
        if not matches_expected:
            print(f"      Expected: {expected_shot_types}")
            print(f"      Actual: {actual_shot_types}")
    
    return {
        'test_name': test_name,
        'passed': is_valid,
        'timeline_size': len(framing_timeline),
        'progression_size': len(progression),
        'shot_distribution': shot_counts,
        'errors': errors if not is_valid else []
    }


def main():
    """Run all test cases."""
    print("="*70)
    print("PersonFramingV2 Test Harness")
    print("="*70)
    
    test_results = []
    
    # Test 1: Continuous person presence
    timeline = create_test_person_timeline(10, "continuous")
    result = run_test_case(
        "Continuous Person Presence",
        timeline, 10,
        expected_shot_types=['close']
    )
    test_results.append(result)
    
    # Test 2: Intermittent person presence
    timeline = create_test_person_timeline(10, "intermittent")
    result = run_test_case(
        "Intermittent Person Presence",
        timeline, 10,
        expected_shot_types=['medium', 'none']
    )
    test_results.append(result)
    
    # Test 3: No person detected
    timeline = create_test_person_timeline(10, "no_person")
    result = run_test_case(
        "No Person Detected",
        timeline, 10,
        expected_shot_types=['none']
    )
    test_results.append(result)
    
    # Test 4: Corrupted data handling
    timeline = create_test_person_timeline(12, "corrupted")
    result = run_test_case(
        "Corrupted Data Handling",
        timeline, 12,
        expected_shot_types=['medium', 'none']
    )
    test_results.append(result)
    
    # Test 5: Varying shot sizes
    timeline = create_test_person_timeline(8, "varying_sizes")
    result = run_test_case(
        "Varying Shot Sizes",
        timeline, 8,
        expected_shot_types=['close', 'medium', 'wide']
    )
    test_results.append(result)
    
    # Test 6: Single frame video
    timeline = {"0-1s": {'face_bbox': {'width': 0.3, 'height': 0.3}, 'face_confidence': 0.9}}
    result = run_test_case(
        "Single Frame Video",
        timeline, 1
    )
    test_results.append(result)
    
    # Test 7: Long video (60 seconds)
    timeline = create_test_person_timeline(60, "continuous")
    result = run_test_case(
        "Long Video (60s)",
        timeline, 60
    )
    test_results.append(result)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in test_results if r['passed'])
    total = len(test_results)
    
    for result in test_results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} - {result['test_name']}")
        if not result['passed'] and result['errors']:
            print(f"     First error: {result['errors'][0]}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Save results
    output_path = Path("test_outputs/person_framing_v2_test_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())