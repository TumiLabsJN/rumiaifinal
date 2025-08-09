#!/usr/bin/env python3
"""Count all features in the 6-block output files"""

import json
from pathlib import Path

def count_features(data, prefix=""):
    """Recursively count all leaf features in nested dict"""
    count = 0
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                count += count_features(value, f"{prefix}.{key}" if prefix else key)
            else:
                # This is a leaf feature
                count += 1
                # print(f"  {prefix}.{key}" if prefix else key)  # Uncomment to see feature names
    elif isinstance(data, list):
        # For lists, count each item
        if len(data) > 0 and not isinstance(data[0], (dict, list)):
            count += 1  # Simple list counts as 1 feature
    return count

output_dir = Path("test_outputs")
total_features = 0

files = [
    "7280654844715666731_emotional_journey.json",
    "7280654844715666731_person_framing.json", 
    "7280654844715666731_scene_pacing.json",
    "7280654844715666731_speech_analysis.json",
    "7280654844715666731_visual_overlay_analysis.json",
    "7280654844715666731_metadata_analysis.json"
]

print("Features per analysis type:")
print("-" * 40)

for filename in files:
    filepath = output_dir / filename
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
        
        # Count features in each block
        block_counts = {}
        for block_name, block_data in data.items():
            block_counts[block_name] = count_features(block_data)
        
        analysis_type = filename.replace("7280654844715666731_", "").replace(".json", "")
        file_total = sum(block_counts.values())
        total_features += file_total
        
        print(f"\n{analysis_type}: {file_total} features")
        for block, count in block_counts.items():
            print(f"  {block}: {count}")

print("\n" + "=" * 40)
print(f"TOTAL FEATURES ACROSS ALL ANALYSES: {total_features}")
print("=" * 40)

print("\nComparison with ML_FEATURES_DOCUMENTATION_V2.md:")
print(f"- Documentation claims: ~300 features")
print(f"- Actual output: {total_features} features")
print(f"- Ratio: {total_features/300*100:.1f}%")

# Note: creative_density is missing due to the bug, which would add ~40-50 more features
print(f"\nWith creative_density fixed: ~{total_features + 45} features")
print(f"That would be: {(total_features + 45)/300*100:.1f}% of documented features")