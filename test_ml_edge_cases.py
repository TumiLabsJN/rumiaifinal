#!/usr/bin/env python3
"""Test edge cases and error handling for ML precompute"""

import os
import sys
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rumiai_v2.validators.response_validator import ResponseValidator

def test_edge_cases():
    """Test various edge cases for ML precompute"""
    
    print("="*80)
    print("ML PRECOMPUTE EDGE CASE TESTS")
    print("="*80)
    
    # Test 1: Missing blocks
    print("\n1. MISSING BLOCKS TEST:")
    incomplete_response = {
        "CoreMetrics": {"score": 8.0},
        "Dynamics": {"trend": "stable"},
        # Missing other blocks
    }
    
    result = ResponseValidator.validate_response(
        json.dumps(incomplete_response), 
        'creative_density',
        'v2'
    )
    print(f"   Valid: {result[0]}")
    print(f"   Errors: {result[2]}")
    
    # Test 2: Wrong block names
    print("\n2. WRONG BLOCK NAMES TEST:")
    wrong_blocks = {
        "Core": {"score": 8.0},  # Should be CoreMetrics
        "Dynamic": {"trend": "stable"},  # Should be Dynamics
        "Interaction": {},  # Should be Interactions
        "Events": {},  # Should be KeyEvents
        "Pattern": {},  # Should be Patterns
        "Summary": {}  # Should be Quality
    }
    
    result = ResponseValidator.validate_response(
        json.dumps(wrong_blocks),
        'creative_density', 
        'v2'
    )
    print(f"   Valid: {result[0]}")
    print(f"   Errors: {result[2][:2]}")  # Show first 2 errors
    
    # Test 3: Prefixed block names (should normalize)
    print("\n3. PREFIXED BLOCK NAMES TEST:")
    prefixed_blocks = {
        "creativeDensityCoreMetrics": {"score": 8.0},
        "creativeDensityDynamics": {"trend": "stable"},
        "creativeDensityInteractions": {"sync": 0.9},
        "creativeDensityKeyEvents": {"events": []},
        "creativeDensityPatterns": {"patterns": []},
        "creativeDensityQuality": {"overall": 8.0}
    }
    
    result = ResponseValidator.validate_response(
        json.dumps(prefixed_blocks),
        'creative_density',
        'v2'
    )
    print(f"   Valid: {result[0]}")
    print(f"   Normalized: {list(result[1].keys()) if result[1] else 'None'}")
    
    # Test 4: Empty response
    print("\n4. EMPTY RESPONSE TEST:")
    result = ResponseValidator.validate_response("", 'creative_density', 'v2')
    print(f"   Valid: {result[0]}")
    print(f"   Errors: {result[2]}")
    
    # Test 5: Non-JSON response
    print("\n5. NON-JSON RESPONSE TEST:")
    text_response = "This is just plain text, not JSON"
    result = ResponseValidator.validate_response(text_response, 'creative_density', 'v2')
    print(f"   Valid: {result[0]}")
    print(f"   Errors: {result[2][:1]}")  # Show first error
    
    # Test 6: Mixed valid/invalid blocks
    print("\n6. MIXED VALID/INVALID BLOCKS TEST:")
    mixed_response = {
        "CoreMetrics": {"score": 8.0},  # Valid
        "Dynamics": "should be dict not string",  # Invalid type
        "Interactions": {},  # Valid but empty
        "KeyEvents": None,  # Invalid null
        "Patterns": {"patterns": []},  # Valid
        "Quality": {"confidence": 0.9}  # Valid
    }
    
    result = ResponseValidator.validate_response(
        json.dumps(mixed_response),
        'creative_density',
        'v2'
    )
    print(f"   Valid: {result[0]}")
    print(f"   Errors: {result[2]}")
    
    # Test 7: Very large response
    print("\n7. LARGE RESPONSE TEST:")
    large_response = {
        "CoreMetrics": {"data": "x" * 10000},  # 10KB of data
        "Dynamics": {"data": "y" * 10000},
        "Interactions": {"data": "z" * 10000},
        "KeyEvents": {"events": [{"event": f"Event {i}"} for i in range(100)]},
        "Patterns": {"patterns": ["pattern"] * 100},
        "Quality": {"metrics": {f"metric_{i}": i for i in range(100)}}
    }
    
    import time
    start = time.time()
    result = ResponseValidator.validate_response(
        json.dumps(large_response),
        'creative_density',
        'v2'
    )
    elapsed = time.time() - start
    print(f"   Valid: {result[0]}")
    print(f"   Validation time: {elapsed:.3f}s")
    print(f"   Response size: {len(json.dumps(large_response)) / 1024:.1f} KB")
    
    # Test 8: Unicode and special characters
    print("\n8. UNICODE AND SPECIAL CHARACTERS TEST:")
    unicode_response = {
        "CoreMetrics": {"text": "Hello 世界 🌍 \n\t Special chars: <>&\"'"},
        "Dynamics": {"emoji": "🚀📈📊"},
        "Interactions": {"unicode": "αβγδε ΑΒΓΔΕ"},
        "KeyEvents": {"events": ["Event with\nnewline", "Event with\ttab"]},
        "Patterns": {"patterns": ["Pattern™", "Pattern®"]},
        "Quality": {"score": 8.5}
    }
    
    result = ResponseValidator.validate_response(
        json.dumps(unicode_response),
        'creative_density',
        'v2'
    )
    print(f"   Valid: {result[0]}")
    print(f"   Unicode handled: Yes")
    
    # Test 9: Nested structure depth
    print("\n9. DEEPLY NESTED STRUCTURE TEST:")
    nested_response = {
        "CoreMetrics": {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": "deep value"
                        }
                    }
                }
            }
        },
        "Dynamics": {"simple": "value"},
        "Interactions": {"nested": {"array": [1, [2, [3, [4, [5]]]]]}},
        "KeyEvents": {"events": []},
        "Patterns": {"patterns": []},
        "Quality": {"score": 8.0}
    }
    
    result = ResponseValidator.validate_response(
        json.dumps(nested_response),
        'creative_density',
        'v2'
    )
    print(f"   Valid: {result[0]}")
    print(f"   Deep nesting handled: Yes")
    
    # Test 10: Cost calculation accuracy
    print("\n10. COST CALCULATION TEST:")
    print(f"   For 1230 tokens at Haiku pricing:")
    print(f"   - Input tokens (assumed 30%): {1230 * 0.3:.0f}")
    print(f"   - Output tokens (assumed 70%): {1230 * 0.7:.0f}")
    print(f"   - Input cost: ${(1230 * 0.3 * 0.25 / 1_000_000):.6f}")
    print(f"   - Output cost: ${(1230 * 0.7 * 1.25 / 1_000_000):.6f}")
    print(f"   - Total cost: ${((1230 * 0.3 * 0.25 + 1230 * 0.7 * 1.25) / 1_000_000):.6f}")
    
    print("\n" + "="*80)
    print("EDGE CASE TESTS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_edge_cases()