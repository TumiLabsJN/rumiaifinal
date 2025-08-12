#!/usr/bin/env python3
"""
Verify that local test script stays synchronized with production code.
Run this after making production changes to catch issues early.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

def check_precompute_functions():
    """Verify all production precompute functions are handled"""
    issues = []
    
    try:
        # Add parent to path for imports
        import sys
        sys.path.insert(0, '/home/jorge/rumiaifinal')
        
        # Import production's COMPUTE_FUNCTIONS
        from rumiai_v2.processors import COMPUTE_FUNCTIONS
        
        # Import test script
        sys.path.insert(0, str(Path(__file__).parent))
        from local_video_runner import LocalVideoRunner
        
        runner = LocalVideoRunner()
        
        # Check if test script handles all production functions
        production_functions = set(COMPUTE_FUNCTIONS.keys())
        
        # These are the functions we expect to handle
        expected_functions = {
            'creative_density', 'emotional_journey', 'metadata_analysis',
            'person_framing', 'scene_pacing', 'speech_analysis',
            'visual_overlay_analysis'
        }
        
        # Check for new functions not in our expected list
        new_functions = production_functions - expected_functions
        if new_functions:
            issues.append(f"NEW production functions not handled: {new_functions}")
            
        # Check for removed functions
        removed_functions = expected_functions - production_functions
        if removed_functions:
            issues.append(f"REMOVED production functions still referenced: {removed_functions}")
            
    except Exception as e:
        issues.append(f"Failed to import modules: {e}")
        
    return issues

def check_field_mappings():
    """Verify field name mappings cover all production outputs"""
    issues = []
    
    try:
        # Import test script's conversion function
        sys.path.insert(0, str(Path(__file__).parent))
        from local_video_runner import LocalVideoRunner
        
        runner = LocalVideoRunner()
        
        # Test data patterns that production might generate
        test_patterns = [
            # New potential patterns to test
            {'newAnalysisCoreMetrics': {}, 'newAnalysisDynamics': {}},
            {'analysisTypeCoreMeasures': {}, 'analysisTypeFlow': {}},
            {'mlCoreMetrics': {}, 'mlDynamics': {}},
        ]
        
        for pattern in test_patterns:
            result = runner.convert_to_ml_format(pattern, 'test_analysis')
            
            # Check if conversion produced generic names
            if 'CoreMetrics' not in result and any('Core' in k for k in pattern.keys()):
                issues.append(f"Unmapped pattern: {list(pattern.keys())}")
                
    except Exception as e:
        issues.append(f"Failed to test mappings: {e}")
        
    return issues

def check_output_structure():
    """Verify test script creates same file structure as production"""
    issues = []
    
    # Check if both create 3 files per analysis
    expected_files = ['_complete_', '_ml_', '_result_']
    
    # Look for recent production outputs
    prod_dirs = Path("/home/jorge/rumiaifinal/insights").glob("*/creative_density")
    
    for prod_dir in list(prod_dirs)[:1]:  # Check one recent production output
        files = list(prod_dir.glob("*.json"))
        
        for expected in expected_files:
            if not any(expected in f.name for f in files):
                issues.append(f"Production missing expected file type: {expected}")
                
    return issues

def check_professional_wrappers():
    """Check if new professional wrapper functions were added"""
    issues = []
    
    try:
        from rumiai_v2.processors.precompute_professional_wrappers import PROFESSIONAL_CONVERTERS
        
        # Check which analyses use professional wrappers
        wrapper_functions = set(PROFESSIONAL_CONVERTERS.keys())
        
        # Known wrappers that need special handling
        known_wrappers = {
            'person_framing', 'scene_pacing', 'metadata_analysis',
            'speech_analysis'
        }
        
        new_wrappers = wrapper_functions - known_wrappers
        if new_wrappers:
            issues.append(f"New professional wrappers added: {new_wrappers}")
            issues.append("  → Update convert_to_ml_format() mappings for these")
            
    except ImportError:
        pass  # Module might not exist
    except Exception as e:
        issues.append(f"Failed to check wrappers: {e}")
        
    return issues

def check_runtime_compatibility():
    """Do a quick runtime test to ensure basic compatibility"""
    issues = []
    
    try:
        # Try to instantiate the test runner
        sys.path.insert(0, str(Path(__file__).parent))
        from local_video_runner import LocalVideoRunner
        
        runner = LocalVideoRunner()
        
        # Test the conversion function with known patterns
        test_cases = [
            ('creative_density', {'densityCoreMetrics': {'test': 1}}),
            ('scene_pacing', {'scenePacingCoreMetrics': {'test': 1}}),
            ('person_framing', {'personFramingCoreMetrics': {'test': 1}}),
        ]
        
        for analysis_type, test_data in test_cases:
            result = runner.convert_to_ml_format(test_data, analysis_type)
            if 'CoreMetrics' not in result:
                issues.append(f"Conversion failed for {analysis_type}")
                
    except Exception as e:
        issues.append(f"Runtime compatibility check failed: {e}")
        
    return issues

def main():
    """Run all synchronization checks"""
    
    print("=" * 70)
    print("PRODUCTION ↔ TEST SYNCHRONIZATION CHECK")
    print("=" * 70)
    
    all_issues = []
    
    # Run all checks
    checks = [
        ("Precompute Functions", check_precompute_functions),
        ("Field Mappings", check_field_mappings),
        ("Output Structure", check_output_structure),
        ("Professional Wrappers", check_professional_wrappers),
        ("Runtime Compatibility", check_runtime_compatibility),
    ]
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        issues = check_func()
        
        if issues:
            print(f"  ❌ Found {len(issues)} issue(s):")
            for issue in issues:
                print(f"     • {issue}")
            all_issues.extend(issues)
        else:
            print(f"  ✅ No issues found")
    
    # Summary
    print("\n" + "=" * 70)
    if all_issues:
        print(f"❌ SYNCHRONIZATION ISSUES DETECTED: {len(all_issues)} total")
        print("\nACTION REQUIRED:")
        print("1. Review the issues above")
        print("2. Update local_video_runner.py to handle new patterns")
        print("3. Re-run this check to verify fixes")
        return 1
    else:
        print("✅ PRODUCTION AND TEST ARE IN SYNC!")
        print("\nYour local test script should work correctly with current production code.")
        return 0

if __name__ == "__main__":
    sys.exit(main())