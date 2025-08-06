#!/usr/bin/env python3
"""
Test script for fail-fast implementation
Tests all validators and error handling
"""

import json
import sys
from pathlib import Path

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.contracts.validators import MLServiceValidators
from rumiai_v2.contracts.claude_output_validators import ClaudeOutputValidators
from rumiai_v2.core.error_handler import RumiAIErrorHandler
from rumiai_v2.core.session_context import SessionContextLoader

def test_ml_validators():
    """Test ML service validators with actual data"""
    print("Testing ML Service Validators...")
    
    # Test OCR with old polygon format (should fail)
    ocr_old = {
        'textAnnotations': [{
            'text': 'test',
            'confidence': 0.9,
            'bbox': [[100, 100], [200, 100], [200, 150], [100, 150]]  # Polygon format
        }]
    }
    
    is_valid, msg = MLServiceValidators.validate('ocr', ocr_old)
    print(f"  OCR (old polygon format): {is_valid} - {msg}")
    assert not is_valid, "Should reject polygon format"
    
    # Test OCR with new flat format (should pass)
    ocr_new = {
        'textAnnotations': [{
            'text': 'test',
            'confidence': 0.9,
            'bbox': [100, 100, 100, 50]  # Flat format [x, y, w, h]
        }]
    }
    
    is_valid, msg = MLServiceValidators.validate('ocr', ocr_new)
    print(f"  OCR (new flat format): {is_valid} - {msg}")
    assert is_valid, "Should accept flat format"
    
    # Test empty OCR (should pass)
    ocr_empty = {'textAnnotations': []}
    is_valid, msg = MLServiceValidators.validate('ocr', ocr_empty)
    print(f"  OCR (empty): {is_valid} - {msg}")
    assert is_valid, "Should accept empty annotations"
    
    print("✅ ML validators working correctly\n")

def test_claude_validators():
    """Test Claude output validators"""
    print("Testing Claude Output Validators...")
    
    # Test with wrong block names (should fail)
    wrong_blocks = {
        'densityCoreMetrics': {},  # Wrong - prefixed
        'densityDynamics': {},
        'densityInteractions': {},
        'densityKeyEvents': {},
        'densityPatterns': {},
        'densityQuality': {}
    }
    
    is_valid, msg = ClaudeOutputValidators.validate_claude_response('creative_density', wrong_blocks)
    print(f"  Prefixed blocks: {is_valid} - {msg[:50]}...")
    assert not is_valid, "Should reject prefixed block names"
    
    # Test with correct block names (should pass)
    correct_blocks = {
        'CoreMetrics': {'avgDensity': 2.5, 'maxDensity': 5, 'totalElements': 25, 'confidence': 0.85},
        'Dynamics': {'densityCurve': [], 'volatility': 0.3, 'accelerationPattern': 'steady', 'confidence': 0.8},
        'Interactions': {'multiModalPeaks': [], 'elementCooccurrence': {}, 'confidence': 0.75},
        'KeyEvents': {'peakMoments': [], 'deadZones': [], 'confidence': 0.9},
        'Patterns': {'densityPattern': 'moderate', 'temporalFlow': 'linear', 'confidence': 0.82},
        'Quality': {'dataCompleteness': 0.95, 'detectionReliability': {}, 'overallConfidence': 0.87}
    }
    
    is_valid, msg = ClaudeOutputValidators.validate_claude_response('creative_density', correct_blocks)
    print(f"  Correct blocks: {is_valid} - {msg}")
    assert is_valid, "Should accept correct block structure"
    
    # Test Quality block with wrong confidence field (should fail)
    bad_quality = correct_blocks.copy()
    bad_quality['Quality'] = {'confidence': 0.9}  # Wrong - should be overallConfidence
    
    is_valid, msg = ClaudeOutputValidators.validate_claude_response('creative_density', bad_quality)
    print(f"  Wrong Quality confidence: {is_valid} - {msg[:50]}...")
    assert not is_valid, "Should reject Quality block without overallConfidence"
    
    print("✅ Claude validators working correctly\n")

def test_error_handler():
    """Test error handler initialization"""
    print("Testing Error Handler...")
    
    # Initialize error handler (will create directories)
    handler = RumiAIErrorHandler()
    
    # Check directories were created
    assert handler.log_dir.exists(), f"Log dir not created: {handler.log_dir}"
    assert handler.debug_dir.exists(), f"Debug dir not created: {handler.debug_dir}"
    
    print(f"  Log directory: {handler.log_dir}")
    print(f"  Debug directory: {handler.debug_dir}")
    print("✅ Error handler initialized correctly\n")

def test_session_context():
    """Test session context loader"""
    print("Testing Session Context Loader...")
    
    # Initialize context loader
    loader = SessionContextLoader()
    
    # Get context (should work even with no history)
    context = loader.get_recent_context()
    
    print(f"  Recent errors: {len(context.get('recent_errors', []))}")
    print(f"  Known issues: {context.get('known_issues', {})}")
    print(f"  Last success: {context.get('last_successful_run', 'None')}")
    
    # Generate summary report
    report = loader.get_summary_report()
    print("\nSummary Report Preview:")
    print(report[:200] + "...")
    
    print("✅ Session context loader working correctly\n")

def test_ocr_fix():
    """Test that OCR bbox fix was applied"""
    print("Testing OCR bbox fix in ml_services_unified.py...")
    
    ml_services_file = Path("/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py")
    
    if ml_services_file.exists():
        content = ml_services_file.read_text()
        
        # Check for the fix
        if "Convert polygon bbox to flat format" in content:
            print("  ✅ OCR fix comment found")
        else:
            print("  ⚠️ OCR fix comment not found")
            
        if "max(xs)-min(xs), max(ys)-min(ys)" in content:
            print("  ✅ OCR conversion logic found")
        else:
            print("  ⚠️ OCR conversion logic not found")
    else:
        print("  ❌ ml_services_unified.py not found")
    
    print()

def main():
    """Run all tests"""
    print("=" * 60)
    print("FAIL-FAST IMPLEMENTATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_ocr_fix()
        test_ml_validators()
        test_claude_validators()
        test_error_handler()
        test_session_context()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED - FAIL-FAST IMPLEMENTATION COMPLETE")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()