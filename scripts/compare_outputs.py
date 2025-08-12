#!/usr/bin/env python3
"""
Compare production and test output JSON files for structural differences
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

def get_json_keys_recursive(obj, prefix=""):
    """Get all keys in a JSON object recursively with their paths"""
    keys = set()
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key)
            if isinstance(value, dict):
                keys.update(get_json_keys_recursive(value, full_key))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Sample first item in list
                keys.update(get_json_keys_recursive(value[0], f"{full_key}[0]"))
    
    return keys

def compare_json_structure(prod_file: Path, test_file: Path) -> Dict:
    """Compare structure of two JSON files"""
    
    with open(prod_file) as f:
        prod_data = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)
    
    prod_keys = get_json_keys_recursive(prod_data)
    test_keys = get_json_keys_recursive(test_data)
    
    return {
        'prod_only': prod_keys - test_keys,
        'test_only': test_keys - prod_keys,
        'common': prod_keys & test_keys,
        'prod_data': prod_data,
        'test_data': test_data
    }

def check_field_naming(prod_data: dict, test_data: dict, file_type: str, analysis_type: str) -> List[str]:
    """Check if field naming conventions match expectations"""
    issues = []
    
    if file_type == 'ml':
        # ML files should have generic names (CoreMetrics, Dynamics, etc.)
        expected_generic = ['CoreMetrics', 'Dynamics', 'Interactions', 'KeyEvents', 'Patterns', 'Quality']
        
        # Check production
        if 'parsed_response' in prod_data:
            prod_fields = set(prod_data['parsed_response'].keys())
        else:
            prod_fields = set(prod_data.keys())
            
        # Check test
        test_fields = set(test_data.keys())
        
        # Check for prefixed names in ML files (they shouldn't exist)
        prefix_map = {
            'creative_density': 'density',
            'emotional_journey': 'emotional',
            'person_framing': 'framing',
            'scene_pacing': 'pacing',
            'speech_analysis': 'speech',
            'visual_overlay_analysis': 'overlay',
            'metadata_analysis': 'metadata'
        }
        
        prefix = prefix_map.get(analysis_type, analysis_type)
        
        for field in test_fields:
            if field.startswith(prefix) and field != analysis_type:
                issues.append(f"ML file has prefixed field '{field}' - should be generic")
                
    elif file_type == 'result':
        # Result files should have analysis-specific names
        prefix_map = {
            'creative_density': 'density',
            'emotional_journey': 'emotional',
            'person_framing': 'framing',
            'scene_pacing': 'pacing',
            'speech_analysis': 'speech',
            'visual_overlay_analysis': 'overlay',
            'metadata_analysis': 'metadata'
        }
        
        prefix = prefix_map.get(analysis_type, analysis_type)
        
        # Check if result has proper prefixed names
        test_fields = set(test_data.keys())
        
        has_prefixed = any(field.startswith(prefix) for field in test_fields)
        has_generic = any(field in ['CoreMetrics', 'Dynamics', 'Interactions', 'KeyEvents', 'Patterns', 'Quality'] 
                         for field in test_fields)
        
        if has_generic and not has_prefixed:
            issues.append(f"Result file has generic field names - should have '{prefix}' prefix")
    
    return issues

def main():
    prod_base = Path("/home/jorge/rumiaifinal/insights/7518107389090876727")
    test_base = Path("/home/jorge/rumiaifinal/insights/TestVideoTester12.08pt2")
    
    analysis_types = [
        'creative_density', 'emotional_journey', 'metadata_analysis',
        'person_framing', 'scene_pacing', 'speech_analysis',
        'temporal_markers', 'visual_overlay_analysis'
    ]
    
    all_issues = []
    
    for analysis in analysis_types:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {analysis}")
        print('='*60)
        
        for file_type in ['complete', 'ml', 'result']:
            # Find the files
            prod_files = list(prod_base.glob(f"{analysis}/{analysis}_{file_type}_*.json"))
            test_files = list(test_base.glob(f"{analysis}/{analysis}_{file_type}_*.json"))
            
            if not prod_files or not test_files:
                issue = f"Missing {file_type} file - Prod: {len(prod_files)}, Test: {len(test_files)}"
                all_issues.append(f"{analysis}/{file_type}: {issue}")
                print(f"\n❌ {file_type}: {issue}")
                continue
                
            prod_file = prod_files[0]
            test_file = test_files[0]
            
            print(f"\n{file_type.upper()} FILE:")
            print(f"  Prod: {prod_file.name}")
            print(f"  Test: {test_file.name}")
            
            # Compare structure
            comparison = compare_json_structure(prod_file, test_file)
            
            # Check field naming conventions
            naming_issues = check_field_naming(
                comparison['prod_data'], 
                comparison['test_data'],
                file_type,
                analysis
            )
            
            if comparison['prod_only']:
                print(f"  ❌ Fields only in PRODUCTION:")
                for key in sorted(comparison['prod_only'])[:5]:  # Show first 5
                    print(f"     - {key}")
                    all_issues.append(f"{analysis}/{file_type}: Missing in test: {key}")
                if len(comparison['prod_only']) > 5:
                    print(f"     ... and {len(comparison['prod_only']) - 5} more")
                    
            if comparison['test_only']:
                print(f"  ❌ Fields only in TEST:")
                for key in sorted(comparison['test_only'])[:5]:
                    print(f"     - {key}")
                    all_issues.append(f"{analysis}/{file_type}: Extra in test: {key}")
                if len(comparison['test_only']) > 5:
                    print(f"     ... and {len(comparison['test_only']) - 5} more")
                    
            if naming_issues:
                print(f"  ❌ Field naming issues:")
                for issue in naming_issues:
                    print(f"     - {issue}")
                    all_issues.append(f"{analysis}/{file_type}: {issue}")
                    
            if not comparison['prod_only'] and not comparison['test_only'] and not naming_issues:
                print(f"  ✅ Structure matches perfectly")
    
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL ISSUES")
    print('='*60)
    
    if all_issues:
        print(f"\nFound {len(all_issues)} total issues:\n")
        for issue in all_issues[:20]:  # Show first 20 issues
            print(f"  • {issue}")
        if len(all_issues) > 20:
            print(f"\n  ... and {len(all_issues) - 20} more issues")
    else:
        print("\n✅ No structural differences found! Test and production outputs match.")
    
    return 0 if not all_issues else 1

if __name__ == "__main__":
    sys.exit(main())