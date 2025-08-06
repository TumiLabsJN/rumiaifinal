#!/usr/bin/env python3
"""Test what analysis.to_dict() contains"""

import json
from pathlib import Path

# Load the unified analysis file
with open('/home/jorge/rumiaifinal/unified_analysis/7280654844715666731.json', 'r') as f:
    analysis_dict = json.load(f)

print("=== Top-level keys in analysis.to_dict() ===")
print(list(analysis_dict.keys()))

print("\n=== Keys in 'ml_data' ===")
if 'ml_data' in analysis_dict:
    print(list(analysis_dict['ml_data'].keys()))
    
print("\n=== Keys in 'timeline' ===")
if 'timeline' in analysis_dict:
    print(list(analysis_dict['timeline'].keys()))

# Check if the helpers would work with this structure
print("\n=== Would helpers work? ===")
print(f"Helper expects: ml_data['ocr']")
print(f"Available: analysis_dict['ml_data']['ocr'] = {bool(analysis_dict.get('ml_data', {}).get('ocr'))}")

# The helpers need ml_data, not analysis_dict
print("\n=== What wrappers should pass to helpers ===")
print("Current (WRONG): extract_ocr_data(analysis_dict)")  
print("Should be: extract_ocr_data(analysis_dict['ml_data'])")