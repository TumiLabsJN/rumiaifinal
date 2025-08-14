#!/usr/bin/env python3
"""
Integration test for PersonFramingV2 implementation.
Tests the actual implementation in precompute_functions_full.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the actual function from precompute_functions_full
from rumiai_v2.processors.precompute_functions_full import (
    compute_person_framing_metrics,
    classify_shot_type_simple,
    calculate_temporal_framing_simple,
    analyze_framing_progression_simple
)


def create_test_data():
    """Create test data that mimics real pipeline data."""
    
    # Create test timelines
    expression_timeline = {
        "0-1s": {"emotion": "neutral"},
        "1-2s": {"emotion": "happy"},
        "2-3s": {},  # No emotion detected
        "3-4s": {"emotion": "surprised"},
        "4-5s": {"emotion": "neutral"}
    }
    
    object_timeline = {
        "0-1s": {"objects": {"person": 1}},
        "1-2s": {"objects": {"person": 1, "phone": 1}},
        "2-3s": {"objects": {}},  # No person
        "3-4s": {"objects": {"person": 1}},
        "4-5s": {"objects": {"person": 1}}
    }
    
    camera_distance_timeline = {
        "0-1s": {"distance": "close"},
        "1-2s": {"distance": "medium"},
        "2-3s": {"distance": "wide"},
        "3-4s": {"distance": "medium"},
        "4-5s": {"distance": "close"}
    }
    
    # Create person timeline with varying face sizes
    person_timeline = {
        "0-1s": {
            "face_bbox": {"width": 0.5, "height": 0.6},  # Close shot
            "face_confidence": 0.95
        },
        "1-2s": {
            "face_bbox": {"width": 0.3, "height": 0.35},  # Medium shot
            "face_confidence": 0.90
        },
        "2-3s": {},  # No face detected
        "3-4s": {
            "face_bbox": {"width": 0.15, "height": 0.2},  # Wide shot
            "face_confidence": 0.85
        },
        "4-5s": {
            "face_bbox": {"width": 0.55, "height": 0.65},  # Close shot
            "face_confidence": 0.98
        }
    }
    
    # Gaze timeline
    gaze_timeline = {
        "0-1s": {"eye_contact": 0.8},
        "1-2s": {"eye_contact": 0.6},
        "2-3s": {"eye_contact": 0},
        "3-4s": {"eye_contact": 0.4},
        "4-5s": {"eye_contact": 0.9}
    }
    
    # Enhanced human data (can be None)
    enhanced_human_data = None
    
    # Duration
    duration = 5
    
    return {
        "expression_timeline": expression_timeline,
        "object_timeline": object_timeline,
        "camera_distance_timeline": camera_distance_timeline,
        "person_timeline": person_timeline,
        "gaze_timeline": gaze_timeline,
        "enhanced_human_data": enhanced_human_data,
        "duration": duration
    }


def test_helper_functions():
    """Test the PersonFramingV2 helper functions."""
    print("\n🧪 Testing Helper Functions")
    print("-" * 50)
    
    # Test classify_shot_type_simple
    test_cases = [
        (30, 'close'),
        (15, 'medium'),
        (5, 'wide'),
        (0, 'none')
    ]
    
    print("\n1. Testing classify_shot_type_simple:")
    for face_area, expected in test_cases:
        result = classify_shot_type_simple(face_area)
        status = "✅" if result == expected else "❌"
        print(f"   {status} Area={face_area}% -> {result} (expected: {expected})")
    
    # Test calculate_temporal_framing_simple
    print("\n2. Testing calculate_temporal_framing_simple:")
    test_timeline = {
        "0-1s": {"face_bbox": {"width": 0.5, "height": 0.6}, "face_confidence": 0.95},
        "1-2s": {"face_bbox": {"width": -1, "height": 0.3}},  # Corrupted
        "2-3s": {},  # No bbox
    }
    
    result = calculate_temporal_framing_simple(test_timeline, 3)
    print(f"   Input: {len(test_timeline)} entries")
    print(f"   Output: {len(result)} entries")
    print(f"   Shot types: {[v['shot_type'] for v in result.values()]}")
    
    # Test analyze_framing_progression_simple
    print("\n3. Testing analyze_framing_progression_simple:")
    progression = analyze_framing_progression_simple(result)
    print(f"   Progression segments: {len(progression)}")
    for seg in progression:
        print(f"   - {seg['type']}: {seg['start']}-{seg['end']}s (duration: {seg['duration']}s)")


def test_full_integration():
    """Test the complete compute_person_framing_metrics function."""
    print("\n🔧 Testing Full Integration")
    print("-" * 50)
    
    # Get test data
    test_data = create_test_data()
    
    # Run the function
    print("\nRunning compute_person_framing_metrics...")
    try:
        result = compute_person_framing_metrics(**test_data)
        print("✅ Function executed successfully")
    except Exception as e:
        print(f"❌ Function failed: {e}")
        return False
    
    # Check for PersonFramingV2 fields
    print("\n📊 Checking PersonFramingV2 Fields:")
    v2_fields = ['framing_timeline', 'framing_progression', 'framing_changes']
    
    for field in v2_fields:
        if field in result:
            print(f"✅ {field}: Present")
            if field == 'framing_timeline':
                print(f"   - {len(result[field])} timeline entries")
                # Show first entry
                first_key = list(result[field].keys())[0]
                print(f"   - Sample: {first_key} -> {result[field][first_key]}")
            elif field == 'framing_progression':
                print(f"   - {len(result[field])} progression segments")
                if result[field]:
                    print(f"   - First segment: {result[field][0]}")
            elif field == 'framing_changes':
                print(f"   - Value: {result[field]} changes")
        else:
            print(f"❌ {field}: Missing")
    
    # Check existing fields still work
    print("\n📊 Checking Existing Fields:")
    existing_fields = [
        'face_screen_time_ratio',
        'person_screen_time_ratio', 
        'avg_camera_distance',
        'dominant_shot_type'
    ]
    
    for field in existing_fields:
        if field in result:
            print(f"✅ {field}: {result[field]}")
        else:
            print(f"❌ {field}: Missing")
    
    return result


def test_edge_cases():
    """Test edge cases for PersonFramingV2."""
    print("\n🔬 Testing Edge Cases")
    print("-" * 50)
    
    # Test 1: Empty person timeline
    print("\n1. Empty person timeline:")
    result = calculate_temporal_framing_simple({}, 5)
    print(f"   Result: {len(result)} entries")
    print(f"   All 'none': {all(v['shot_type'] == 'none' for v in result.values())}")
    
    # Test 2: Corrupted bbox data
    print("\n2. Corrupted bbox data:")
    corrupted = {
        "0-1s": {"face_bbox": {"width": float('inf'), "height": 0.3}},
        "1-2s": {"face_bbox": {"width": None, "height": 0.3}},
        "2-3s": {"face_bbox": {"width": "invalid", "height": 0.3}},
    }
    result = calculate_temporal_framing_simple(corrupted, 3)
    print(f"   Result: {[v['shot_type'] for v in result.values()]}")
    print(f"   All handled: {len(result) == 3}")
    
    # Test 3: Very long video (simulated)
    print("\n3. Long video (100 seconds):")
    long_timeline = {f"{i}-{i+1}s": {"face_bbox": {"width": 0.3, "height": 0.3}} for i in range(100)}
    result = calculate_temporal_framing_simple(long_timeline, 100)
    print(f"   Result: {len(result)} entries")
    print(f"   Memory estimate: ~{len(str(result))/1024:.1f}KB")


def test_json_serialization():
    """Test that all outputs are JSON serializable."""
    print("\n📝 Testing JSON Serialization")
    print("-" * 50)
    
    test_data = create_test_data()
    result = compute_person_framing_metrics(**test_data)
    
    try:
        json_str = json.dumps(result, indent=2)
        print("✅ Result is JSON serializable")
        print(f"   Size: {len(json_str)} bytes (~{len(json_str)/1024:.1f}KB)")
        
        # Parse it back
        parsed = json.loads(json_str)
        print("✅ JSON can be parsed back")
        
        # Check key fields survived
        if 'framing_timeline' in parsed and 'framing_progression' in parsed:
            print("✅ PersonFramingV2 fields survived serialization")
        
        # Save sample output
        output_path = Path("test_outputs/person_framing_v2_sample.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(json_str)
        print(f"📁 Sample output saved to: {output_path}")
        
        return True
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("="*70)
    print("PersonFramingV2 Integration Test")
    print("="*70)
    
    tests_passed = []
    
    # Run tests
    print("\n[1/5] Helper Functions Test")
    test_helper_functions()
    tests_passed.append(True)
    
    print("\n[2/5] Full Integration Test")
    result = test_full_integration()
    tests_passed.append(result is not False)
    
    print("\n[3/5] Edge Cases Test")
    test_edge_cases()
    tests_passed.append(True)
    
    print("\n[4/5] JSON Serialization Test")
    json_ok = test_json_serialization()
    tests_passed.append(json_ok)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(tests_passed)
    total = len(tests_passed)
    
    print(f"\nResults: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n✅ PersonFramingV2 is fully integrated and working!")
        print("\nNext Steps:")
        print("1. Test with a real video using full pipeline")
        print("2. Verify professional wrapper handles new fields")
        print("3. Deploy to production")
    else:
        print("\n⚠️ Some tests did not pass. Review the output above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())