#!/usr/bin/env python3
"""Count ALL features including nested structures and arrays"""

import json
from pathlib import Path

def count_all_leaf_features(data, path="", feature_list=None):
    """
    Count ALL leaf features (including array elements as individual features)
    and track their paths
    """
    if feature_list is None:
        feature_list = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                count_all_leaf_features(value, new_path, feature_list)
            elif isinstance(value, list):
                # For lists, we need to decide how to count
                if len(value) > 0:
                    if isinstance(value[0], dict):
                        # List of complex objects - count each object's features
                        for i, item in enumerate(value):
                            count_all_leaf_features(item, f"{new_path}[{i}]", feature_list)
                    else:
                        # List of simple values - could be time series data
                        # Count as one feature (the array itself) or each element
                        # For ML, typically the whole array is one feature
                        feature_list.append((new_path, f"array[{len(value)}]"))
            else:
                # Leaf value (string, number, boolean)
                feature_list.append((new_path, type(value).__name__))
    
    return feature_list

def analyze_json_features():
    output_dir = Path("test_outputs")
    
    files = [
        "7280654844715666731_emotional_journey.json",
        "7280654844715666731_person_framing.json", 
        "7280654844715666731_scene_pacing.json",
        "7280654844715666731_speech_analysis.json",
        "7280654844715666731_visual_overlay_analysis.json",
        "7280654844715666731_metadata_analysis.json"
    ]
    
    all_features = {}
    total_scalar_features = 0
    total_array_features = 0
    total_nested_features = 0
    
    print("=" * 80)
    print("COMPREHENSIVE FEATURE COUNT FROM JSON OUTPUTS")
    print("=" * 80)
    
    for filename in files:
        filepath = output_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            
            analysis_type = filename.replace("7280654844715666731_", "").replace(".json", "")
            features = count_all_leaf_features(data)
            all_features[analysis_type] = features
            
            # Categorize features
            scalar_count = len([f for f in features if 'array' not in f[1]])
            array_count = len([f for f in features if 'array' in f[1]])
            nested_count = len([f for f in features if '[' in f[0] and ']' in f[0]])
            
            print(f"\n{analysis_type.upper()}:")
            print(f"  Scalar features: {scalar_count}")
            print(f"  Array features: {array_count}")
            print(f"  Nested object features: {nested_count}")
            print(f"  TOTAL: {len(features)} features")
            
            # Show sample features
            print(f"  Sample features:")
            for feat, feat_type in features[:5]:
                print(f"    - {feat}: {feat_type}")
            
            total_scalar_features += scalar_count
            total_array_features += array_count
            total_nested_features += nested_count
    
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL 6 ANALYSIS TYPES:")
    print("=" * 80)
    print(f"Scalar features (numbers, strings): {total_scalar_features}")
    print(f"Array features (time series, lists): {total_array_features}")
    print(f"Nested features (objects in arrays): {total_nested_features}")
    print(f"\nTOTAL UNIQUE FEATURES: {total_scalar_features + total_array_features + total_nested_features}")
    
    # Calculate with creative_density estimate
    creative_density_estimate = 45  # Based on other analyses
    print(f"\nWith creative_density (estimated): {total_scalar_features + total_array_features + total_nested_features + creative_density_estimate}")
    
    print("\n" + "=" * 80)
    print("FEATURE TYPES FOR ML TRAINING:")
    print("=" * 80)
    print("1. DIRECT FEATURES (can use as-is): ~{}".format(total_scalar_features))
    print("   - Metrics, scores, counts, ratios")
    print("   - Example: speechDensity=3.63, emotionalDiversity=0.85")
    print("\n2. TIME SERIES FEATURES (need aggregation): ~{}".format(total_array_features))
    print("   - Progressions, temporal patterns")
    print("   - Example: emotionProgression over time")
    print("\n3. STRUCTURED FEATURES (need flattening): ~{}".format(total_nested_features))
    print("   - Complex objects with multiple properties")
    print("   - Example: keyMoments with timestamp+intensity")
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    actual_total = total_scalar_features + total_array_features + total_nested_features
    with_creative = actual_total + creative_density_estimate
    
    print(f"Current JSON outputs provide: {actual_total} features")
    print(f"With creative_density fixed: ~{with_creative} features")
    print(f"\nThese {with_creative} features are the HIGH-LEVEL SEMANTIC FEATURES")
    print("derived from the raw ML data for training recommendation models.")

if __name__ == "__main__":
    analyze_json_features()