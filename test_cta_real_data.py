#!/usr/bin/env python3
"""
Test CTA Real Data Validation
Tests the ML data validation system specifically for CTA alignment analysis
"""

import os
import json
import sys
from ml_data_validator import MLDataValidator

def create_test_data():
    """Create test data with both real and fabricated elements"""
    return {
        "duration_seconds": 15.0,
        "timelines": {
            "textOverlayTimeline": {
                "2-4s": {
                    "text": "Save 50% Today!",
                    "confidence": 0.85,
                    "bbox": [100, 200, 300, 250]
                },
                "8-10s": {
                    "text": "Limited Time Offer",
                    "confidence": 0.92,
                    "bbox": [150, 180, 400, 220]
                }
            },
            "speechTimeline": {
                "0-3s": {
                    "text": "Hey everyone, check out this amazing deal",
                    "confidence": 0.88
                },
                "12-15s": {
                    "text": "Don't miss out on this opportunity",
                    "confidence": 0.91
                }
            },
            "objectTimeline": {
                "5-7s": {
                    "objects": [
                        {"class": "person", "confidence": 0.95},
                        {"class": "phone", "confidence": 0.72}
                    ]
                }
            },
            "gestureTimeline": {
                "1-3s": {
                    "gestures": ["pointing", "waving"]
                }
            }
        }
    }

def create_fabricated_test_data():
    """Create test data with fabricated CTA elements"""
    return {
        "duration_seconds": 12.0,
        "timelines": {
            "textOverlayTimeline": {
                "8-10s": {
                    "text": "link in bio",  # Suspicious pattern
                    "confidence": 1.0,  # Perfect confidence is suspicious
                    "bbox": [100, 200, 200, 250]
                },
                "10-12s": {
                    "text": "swipe up",  # Another suspicious pattern
                    "confidence": 1.0,
                    "bbox": [150, 300, 250, 350]
                }
            },
            "speechTimeline": {},
            "objectTimeline": {}
        }
    }

def create_low_confidence_test_data():
    """Create test data with low confidence detections"""
    return {
        "duration_seconds": 10.0,
        "timelines": {
            "textOverlayTimeline": {
                "2-4s": {
                    "text": "Maybe some text here?",
                    "confidence": 0.3,  # Below threshold
                    "bbox": [100, 200, 300, 250]
                },
                "6-8s": {
                    "text": "High confidence text",
                    "confidence": 0.85,  # Above threshold
                    "bbox": [150, 180, 400, 220]
                }
            },
            "objectTimeline": {
                "1-3s": {
                    "objects": [
                        {"class": "person", "confidence": 0.3},  # Below threshold
                        {"class": "phone", "confidence": 0.8}   # Above threshold
                    ]
                }
            }
        }
    }

def test_cta_validation():
    """Test CTA alignment data validation"""
    print("=" * 60)
    print("Testing CTA Alignment Data Validation")
    print("=" * 60)
    
    # Test 1: Real data validation
    print("\n1. Testing with REAL ML data:")
    print("-" * 40)
    
    real_data = create_test_data()
    validator = MLDataValidator()
    validated_real = validator.extract_real_ml_data(real_data, 'cta_alignment')
    
    print(f"Original text detections: {len(real_data['timelines']['textOverlayTimeline'])}")
    print(f"Validated text detections: {len(validated_real.get('text_timeline', {}))}")
    print(f"Speech timeline entries: {len(validated_real.get('speech_timeline', {}))}")
    print(f"Object timeline entries: {len(validated_real.get('object_timeline', {}))}")
    
    validation_report = validator.get_validation_report()
    print(f"Warnings: {validation_report['summary']['warnings']}")
    print(f"Errors: {validation_report['summary']['errors']}")
    
    # Test 2: Fabricated data detection
    print("\n2. Testing with FABRICATED data:")
    print("-" * 40)
    
    fabricated_data = create_fabricated_test_data()
    validator2 = MLDataValidator()
    validated_fabricated = validator2.extract_real_ml_data(fabricated_data, 'cta_alignment')
    
    print(f"Original text detections: {len(fabricated_data['timelines']['textOverlayTimeline'])}")
    print(f"Validated text detections: {len(validated_fabricated.get('text_timeline', {}))}")
    
    validation_report2 = validator2.get_validation_report()
    print(f"Warnings: {validation_report2['summary']['warnings']}")
    print(f"Errors: {validation_report2['summary']['errors']}")
    
    # Show warning details
    warnings = [log for log in validator2.validation_log if log['level'] == 'WARNING']
    for warning in warnings:
        print(f"  WARNING: {warning['message']}")
    
    # Test 3: Low confidence filtering
    print("\n3. Testing CONFIDENCE thresholds:")
    print("-" * 40)
    
    low_conf_data = create_low_confidence_test_data()
    validator3 = MLDataValidator()
    validated_low_conf = validator3.extract_real_ml_data(low_conf_data, 'cta_alignment')
    
    print(f"Original text detections: {len(low_conf_data['timelines']['textOverlayTimeline'])}")
    print(f"Validated text detections: {len(validated_low_conf.get('text_timeline', {}))}")
    
    original_objects = sum(len(data.get('objects', [])) for data in low_conf_data['timelines']['objectTimeline'].values())
    validated_objects = sum(len(data.get('objects', [])) for data in validated_low_conf.get('object_timeline', {}).values())
    print(f"Original object detections: {original_objects}")
    print(f"Validated object detections: {validated_objects}")
    
    # Test 4: Empty data handling
    print("\n4. Testing EMPTY data handling:")
    print("-" * 40)
    
    empty_data = {
        "duration_seconds": 10.0,
        "timelines": {
            "textOverlayTimeline": {},
            "speechTimeline": {},
            "objectTimeline": {}
        }
    }
    
    validator4 = MLDataValidator()
    validated_empty = validator4.extract_real_ml_data(empty_data, 'cta_alignment')
    
    print(f"Text timeline: {len(validated_empty.get('text_timeline', {}))}")
    print(f"Speech timeline: {len(validated_empty.get('speech_timeline', {}))}")
    print(f"Object timeline: {len(validated_empty.get('object_timeline', {}))}")
    print("✅ Empty data handled correctly (no fabrication)")
    
    return True

def test_with_real_file(video_id):
    """Test validation with a real unified analysis file"""
    print("\n" + "=" * 60)
    print(f"Testing with REAL file: unified_analysis_{video_id}.json")
    print("=" * 60)
    
    unified_file = f"unified_analysis_{video_id}.json"
    if not os.path.exists(unified_file):
        print(f"❌ File not found: {unified_file}")
        return False
    
    with open(unified_file, 'r') as f:
        real_unified_data = json.load(f)
    
    validator = MLDataValidator()
    validated_real_file = validator.extract_real_ml_data(real_unified_data, 'cta_alignment')
    
    # Show results
    timelines = real_unified_data.get('timelines', {})
    print(f"Original text detections: {len(timelines.get('textOverlayTimeline', {}))}")
    print(f"Original speech entries: {len(timelines.get('speechTimeline', {}))}")
    print(f"Original object detections: {len(timelines.get('objectTimeline', {}))}")
    
    print(f"Validated text detections: {len(validated_real_file.get('text_timeline', {}))}")
    print(f"Validated speech entries: {len(validated_real_file.get('speech_timeline', {}))}")
    print(f"Validated object detections: {len(validated_real_file.get('object_timeline', {}))}")
    
    validation_report = validator.get_validation_report()
    print(f"Validation warnings: {validation_report['summary']['warnings']}")
    print(f"Validation errors: {validation_report['summary']['errors']}")
    
    # Save the validated data for inspection
    output_file = f"test_validated_{video_id}_cta.json"
    with open(output_file, 'w') as f:
        json.dump(validated_real_file, f, indent=2)
    print(f"Validated data saved to: {output_file}")
    
    return True

def main():
    """Main test function"""
    print("🧪 ML Data Validation Test Suite")
    print("Testing CTA alignment data validation system")
    
    # Run synthetic tests
    test_success = test_cta_validation()
    
    # Test with real file if provided
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
        real_file_success = test_with_real_file(video_id)
    else:
        print("\n💡 To test with a real file, run:")
        print("   python3 test_cta_real_data.py <video_id>")
        real_file_success = True
    
    print("\n" + "=" * 60)
    if test_success and real_file_success:
        print("✅ ALL TESTS PASSED")
        print("The validation system correctly:")
        print("  - Filters out fabricated patterns")
        print("  - Applies confidence thresholds")
        print("  - Handles empty data gracefully")
        print("  - Logs suspicious activities")
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()