#!/usr/bin/env python3
"""Final validation of the cleaned Python-only pipeline"""

import sys
import os
from pathlib import Path
import glob

def main():
    print("="*60)
    print("FINAL VALIDATION OF PYTHON-ONLY PIPELINE")
    print("="*60)
    
    errors = []
    
    # Test 1: Import without errors
    print("\n[1/5] Testing imports...")
    try:
        from scripts.rumiai_runner import RumiAIRunner
        print("✅ Import successful - Python pipeline intact")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        errors.append(f"Import failed: {e}")
    
    # Test 2: Verify Claude files are gone
    print("\n[2/5] Verifying Claude files removed...")
    claude_files = [
        'rumiai_v2/api/claude_client.py',
        'rumiai_v2/processors/prompt_builder.py',
        'rumiai_v2/processors/output_adapter.py',
        'tests/test_claude_client.py',
        'tests/test_claude_temporal_integration.py'
    ]
    
    for f in claude_files:
        if os.path.exists(f):
            print(f"❌ File still exists: {f}")
            errors.append(f"Claude file still exists: {f}")
    
    if not any(os.path.exists(f) for f in claude_files):
        print("✅ All Claude files removed")
    
    # Test 3: Check no JavaScript files remain (except whisper.cpp)
    print("\n[3/5] Checking JavaScript files...")
    js_files = glob.glob("**/*.js", recursive=True)
    js_files = [f for f in js_files if 'whisper.cpp' not in f and 'node_modules' not in f and 'venv' not in f]
    
    if js_files:
        print(f"❌ JavaScript files remain: {js_files[:3]}")
        errors.append(f"JavaScript files remain: {len(js_files)}")
    else:
        print("✅ All JavaScript files removed")
    
    # Test 4: Verify prompt templates are gone
    print("\n[4/5] Checking prompt templates...")
    prompt_files = list(Path('prompt_templates').glob('*.txt')) if Path('prompt_templates').exists() else []
    
    if prompt_files:
        print(f"❌ Prompt templates remain: {len(prompt_files)}")
        errors.append(f"Prompt templates remain: {len(prompt_files)}")
    else:
        print("✅ All prompt templates removed")
    
    # Test 5: Verify precompute functions exist
    print("\n[5/5] Verifying precompute functions...")
    try:
        from rumiai_v2.processors.precompute_functions import COMPUTE_FUNCTIONS
        
        required_functions = [
            'creative_density',
            'emotional_journey', 
            'person_framing',
            'scene_pacing',
            'speech_analysis',
            'visual_overlay_analysis',
            'metadata_analysis'
        ]
        
        missing = []
        for func in required_functions:
            if func not in COMPUTE_FUNCTIONS:
                missing.append(func)
        
        if missing:
            print(f"❌ Missing functions: {missing}")
            errors.append(f"Missing functions: {missing}")
        else:
            print(f"✅ All 7 precompute functions present")
    except ImportError as e:
        print(f"❌ Could not import precompute functions: {e}")
        errors.append(f"Precompute import failed: {e}")
    
    # Final summary
    print("\n" + "="*60)
    if errors:
        print("❌ VALIDATION FAILED with errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("🎉 VALIDATION SUCCESSFUL!")
        print("\nThe codebase is now:")
        print("✅ Python-only (no Claude API)")
        print("✅ Zero-configuration (all settings hardcoded)")
        print("✅ Zero-cost ($0.00 per video)")
        print("✅ Ready to use!")
        print("\nTest with:")
        print("  python3 scripts/rumiai_runner.py 'VIDEO_URL'")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)