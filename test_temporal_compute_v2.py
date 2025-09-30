#!/usr/bin/env python3
"""
Test script for temporal compute using production unified_analysis.
This script is a TRUE MIRROR of the production pipeline.

Production Mirror Architecture:
1. Load the unified_analysis file that production created
2. Call compute_temporal_windows() with the exact same data
3. Compare results with production output

This ensures the test uses identical inputs and processing as production.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Import ONLY the production compute function
from rumiai_v2.processors.temporal_compute import compute_temporal_windows


def load_unified_analysis(video_id: str) -> Dict[str, Any]:
    """
    Load the unified_analysis file that production created.
    This ensures we use the EXACT same data as production.
    """
    unified_path = Path(f"unified_analysis/{video_id}.json")
    
    if not unified_path.exists():
        print(f"❌ Error: {unified_path} not found!")
        print("Run production pipeline first: python3 scripts/rumiai_runner.py")
        sys.exit(1)
    
    print(f"📂 Loading production unified_analysis from {unified_path}")
    with open(unified_path) as f:
        unified_dict = json.load(f)
    
    # Display what we loaded
    timeline_entries = unified_dict.get('timeline', {}).get('entries', [])
    ml_data_keys = list(unified_dict.get('ml_data', {}).keys())
    
    print(f"  ✓ Loaded unified analysis:")
    print(f"    - Video ID: {unified_dict.get('video_id')}")
    print(f"    - Duration: {unified_dict.get('duration')} seconds")
    print(f"    - Timeline entries: {len(timeline_entries)}")
    print(f"    - ML data sources: {ml_data_keys}")
    
    return unified_dict


def validate_temporal_windows(result: Dict[str, Any]) -> None:
    """
    Validate that all expected features are present in temporal windows.
    """
    print("\n" + "="*60)
    print("📊 FEATURE VALIDATION REPORT")
    print("="*60)
    
    # Expected features based on our implementation
    expected_features = {
        'P0 Core': ['unique_text_count', 'max_simultaneous_texts', 'text_appearance_count',
                    'text_coverage', 'avg_text_lifespan', 'text_change_count',
                    'sticker_count', 'object_count', 
                    'gesture_count', 'expression_count', 'scene_count',
                    'element_count'],
        # 'P0 Density': ['max_density', 'min_density', 'avg_density'],  # REMOVED - see RemoveDensity.md
        'P0 Speech': ['speech_coverage', 'word_count'],
        'P0 Emotions': ['joy_ratio', 'sadness_ratio', 'anger_ratio', 
                       'fear_ratio', 'disgust_ratio', 'surprise_ratio', 
                       'neutral_ratio'],
        'P0 Framing': ['close_ratio', 'medium_ratio', 'wide_ratio', 'none_ratio'],
        'P0 Audio': ['energy_level', 'energy_variance', 'energy_max', 'burst_pattern'],
        'P1 Scene': ['shortest_scene', 'longest_scene'],
        'P2 Variance': ['scene_duration_variance']
    }
    
    # Check hook window
    hook = result['temporal_windows']['hook']
    print("\n🎬 Hook Window (0-3s):")
    
    for category, features in expected_features.items():
        print(f"\n{category}:")
        for feature in features:
            if feature in hook:
                value = hook[feature]
                if isinstance(value, float):
                    print(f"  ✓ {feature}: {value:.4f}")
                else:
                    print(f"  ✓ {feature}: {value}")
            else:
                print(f"  ✗ {feature}: MISSING")
    
    # Summary statistics
    all_features = [f for features in expected_features.values() for f in features]
    present = sum(1 for f in all_features if f in hook)
    total = len(all_features)
    
    print("\n" + "="*60)
    print(f"📈 SUMMARY: {present}/{total} features present ({present*100//total}%)")
    print("="*60)
    
    # Check for any unexpected features
    expected_set = set(all_features)
    actual_set = set(hook.keys()) - {'start', 'end', 'duration'}
    unexpected = actual_set - expected_set
    if unexpected:
        print(f"\n⚠️  Unexpected features found: {unexpected}")


def compare_with_production(test_result: Dict[str, Any], prod_path: Path) -> None:
    """
    Compare test output with production output.
    """
    print(f"\n🔍 Comparing with production output...")
    
    with open(prod_path) as f:
        prod_result = json.load(f)
    
    # Compare hook window as sample
    test_hook = test_result['temporal_windows']['hook']
    prod_hook = prod_result['temporal_windows']['hook']
    
    # Key metrics to compare
    key_metrics = ['unique_text_count', 'text_coverage', 'sticker_count', 'object_count', 
                   'element_count', 'speech_coverage', 'joy_ratio',
                   'shortest_scene', 'scene_duration_variance']
    
    discrepancies = []
    matches = []
    
    for metric in key_metrics:
        test_val = test_hook.get(metric)
        prod_val = prod_hook.get(metric)
        
        # Handle float comparison with tolerance
        if isinstance(test_val, float) and isinstance(prod_val, float):
            if abs(test_val - prod_val) < 0.0001:
                matches.append(f"  ✓ {metric}: {test_val:.4f}")
            else:
                discrepancies.append(f"  ✗ {metric}: test={test_val:.4f}, prod={prod_val:.4f}")
        elif test_val == prod_val:
            matches.append(f"  ✓ {metric}: {test_val}")
        else:
            discrepancies.append(f"  ✗ {metric}: test={test_val}, prod={prod_val}")
    
    # Display results
    if matches:
        print("\nMatching metrics:")
        for m in matches:
            print(m)
    
    if discrepancies:
        print("\n⚠️  Discrepancies found:")
        for d in discrepancies:
            print(d)
    else:
        print("\n✅ All key metrics match production!")


def main():
    """
    Main test flow that mirrors production pipeline exactly.
    This is a TRUE MIRROR - same data, same function, should get same output.
    """
    # Get video ID from command line or use default
    video_id = sys.argv[1] if len(sys.argv) > 1 else "7430952519439846698"
    
    print("="*60)
    print("🚀 TEMPORAL COMPUTE TEST - True Production Mirror")
    print("="*60)
    print(f"Video ID: {video_id}")
    print()
    
    # Step 1: Load the EXACT unified_analysis that production created
    unified_dict = load_unified_analysis(video_id)
    
    # Step 2: Call compute_temporal_windows EXACTLY like production
    # This matches rumiai_runner.py line 286: compute_temporal_windows(unified_analysis.to_dict())
    print(f"\n⚙️  Computing temporal windows with production function...")
    
    try:
        result = compute_temporal_windows(unified_dict)
        print(f"  ✓ Temporal windows computed successfully")
    except Exception as e:
        print(f"  ✗ Error computing temporal windows: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Validate the output has all expected features
    validate_temporal_windows(result)
    
    # Step 4: Save test results
    output_path = Path(f"test_outputs/{video_id}_temporal_windows_test.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Test results saved to: {output_path}")
    
    # Step 5: Compare with production output if it exists
    prod_path = Path(f"insights/{video_id}_temporal_windows_updated.json")
    if prod_path.exists():
        compare_with_production(result, prod_path)
    else:
        print(f"\n⚠️  No production output found at {prod_path}")
        print("Run production first: python3 scripts/rumiai_runner.py")
    
    print("\n" + "="*60)
    print("✅ Test complete!")
    print("\nThis test is a TRUE MIRROR of production:")
    print("  1. Uses exact same unified_analysis data")
    print("  2. Calls exact same compute_temporal_windows function")
    print("  3. Should produce identical output")
    print("="*60)


if __name__ == "__main__":
    main()