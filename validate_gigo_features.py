#!/usr/bin/env python3
"""
Validation script for MLFeaturesGIGO.md implementation
Tests that removed features are absent and new features are present with correct ranges
"""

import json
import sys

def validate_temporal_window(window_data, window_name="window"):
    """Validate a single temporal window has correct features"""

    # Features that should NOT exist
    removed_features = [
        'close_ratio', 'medium_ratio', 'wide_ratio', 'none_ratio',
        'element_count', 'avg_density', 'changes_per_second',
        'has_greeting', 'has_question', 'has_instruction', 'has_speech_cta',
        'burst_pattern', 'joy_ratio', 'sadness_ratio', 'anger_ratio',
        'fear_ratio', 'disgust_ratio', 'surprise_ratio', 'neutral_ratio'
    ]

    errors = []

    # Check removed features are absent
    for feature in removed_features:
        if feature in window_data:
            errors.append(f"❌ Found removed feature '{feature}' in {window_name}")

    # Validate new features exist and have correct ranges
    if 'dominant_emotion_id' in window_data:
        val = window_data['dominant_emotion_id']
        if not (1 <= val <= 8):
            errors.append(f"❌ dominant_emotion_id={val} out of range [1-8] in {window_name}")
    else:
        errors.append(f"❌ Missing required feature 'dominant_emotion_id' in {window_name}")

    if 'emotional_valence' in window_data:
        val = window_data['emotional_valence']
        if not (-1.0 <= val <= 1.0):
            errors.append(f"❌ emotional_valence={val} out of range [-1.0, 1.0] in {window_name}")
    else:
        errors.append(f"❌ Missing required feature 'emotional_valence' in {window_name}")

    if 'emotion_consistency' in window_data:
        val = window_data['emotion_consistency']
        if not (0.0 <= val <= 1.0):
            errors.append(f"❌ emotion_consistency={val} out of range [0.0, 1.0] in {window_name}")
    else:
        errors.append(f"❌ Missing required feature 'emotion_consistency' in {window_name}")

    # Check that kept features are still present
    kept_features = [
        'average_face_size', 'expression_count', 'object_count',
        'gesture_count', 'scene_count', 'speech_coverage', 'word_count',
        'energy_level', 'energy_variance', 'energy_max'
    ]

    for feature in kept_features:
        if feature not in window_data:
            errors.append(f"⚠️  Missing kept feature '{feature}' in {window_name}")

    return errors

def validate_json_file(filepath):
    """Validate a JSON file output from temporal_compute"""

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

    if 'temporal_windows' not in data:
        print("❌ Missing 'temporal_windows' in output")
        return False

    temporal_windows = data['temporal_windows']
    all_errors = []

    # Validate hook window
    if 'hook' in temporal_windows:
        errors = validate_temporal_window(temporal_windows['hook'], 'hook')
        all_errors.extend(errors)
        if not errors:
            print("✅ Hook window validation passed")

    # Validate closing window
    if 'closing' in temporal_windows:
        errors = validate_temporal_window(temporal_windows['closing'], 'closing')
        all_errors.extend(errors)
        if not errors:
            print("✅ Closing window validation passed")

    # Validate middle segments
    if 'middle' in temporal_windows and isinstance(temporal_windows['middle'], list):
        for i, segment in enumerate(temporal_windows['middle']):
            errors = validate_temporal_window(segment, f'middle[{i}]')
            all_errors.extend(errors)
        if not errors:
            print(f"✅ All {len(temporal_windows['middle'])} middle segments validation passed")

    # Print all errors
    if all_errors:
        print("\n🔴 Validation Errors Found:")
        for error in all_errors:
            print(f"  {error}")
        return False
    else:
        print("\n🎉 All validations passed! MLFeaturesGIGO.md successfully implemented.")
        return True

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python validate_gigo_features.py <json_output_file>")
        print("Example: python validate_gigo_features.py output.json")
        sys.exit(1)

    filepath = sys.argv[1]
    success = validate_json_file(filepath)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()