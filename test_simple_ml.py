#!/usr/bin/env python3
"""Simple test to verify ML precompute functionality"""

import os
import sys
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the API to avoid real calls
os.environ['CLAUDE_API_KEY'] = 'test-key'

def test_6_block_structure():
    """Test that we can create proper 6-block structure"""
    
    print("="*80)
    print("TESTING 6-BLOCK STRUCTURE")
    print("="*80)
    
    # Import after setting env
    from rumiai_v2.validators.response_validator import ResponseValidator
    
    # Test 1: Check expected blocks
    print("\n1. EXPECTED BLOCKS:")
    print(f"   Expected blocks: {ResponseValidator.EXPECTED_BLOCKS}")
    
    # Test 2: Create sample 6-block response
    sample_response = {
        "CoreMetrics": {
            "density_score": 7.8,
            "peak_density": 9.2,
            "consistency": 0.85
        },
        "Dynamics": {
            "trend": "increasing",
            "volatility": 0.3,
            "momentum": 0.7
        },
        "Interactions": {
            "visual_audio_sync": 0.9,
            "element_cohesion": 0.8
        },
        "KeyEvents": {
            "events": [
                {"timestamp": 12.5, "description": "Peak creative moment"},
                {"timestamp": 45.2, "description": "Transition point"}
            ]
        },
        "Patterns": {
            "recurring_elements": ["color shifts", "tempo changes"],
            "pattern_strength": 0.75
        },
        "Summary": {
            "overall_rating": 8.0,
            "key_insights": ["High creative density", "Strong visual narrative"]
        }
    }
    
    print("\n2. SAMPLE 6-BLOCK RESPONSE:")
    print(json.dumps(sample_response, indent=2))
    
    # Test 3: Validate the response
    print("\n3. VALIDATION TEST:")
    validation_result = ResponseValidator.validate_response(json.dumps(sample_response))
    print(f"   Valid: {validation_result['valid']}")
    if validation_result['errors']:
        print(f"   Errors: {validation_result['errors']}")
    print(f"   Blocks found: {list(validation_result['data'].keys())}")
    
    # Test 4: Test block normalization
    print("\n4. BLOCK NORMALIZATION TEST:")
    prefixed_response = {
        "creativeDensityCoreMetrics": {"score": 8.0},
        "creativeDensityDynamics": {"trend": "stable"},
        "creativeDensityInteractions": {"sync": 0.9},
        "creativeDensityKeyEvents": {"events": []},
        "creativeDensityPatterns": {"patterns": []},
        "creativeDensitySummary": {"overall": "good"}
    }
    
    normalized = ResponseValidator._normalize_block_names(prefixed_response)
    print(f"   Original keys: {list(prefixed_response.keys())[:3]}...")
    print(f"   Normalized keys: {list(normalized.keys())}")
    
    # Test 5: Test with mock Claude response
    print("\n5. MOCK CLAUDE RESPONSE TEST:")
    mock_claude_response = """
    Let me analyze this video with a 6-block structure:

    CoreMetrics:
    {
        "analysis_score": 8.5,
        "confidence": 0.92,
        "processing_quality": "high"
    }

    Dynamics:
    {
        "change_rate": 0.6,
        "stability": 0.8,
        "evolution": "progressive"
    }

    Interactions:
    {
        "element_correlation": 0.85,
        "sync_quality": 0.9
    }

    KeyEvents:
    {
        "significant_moments": [
            {"time": 10.0, "event": "Introduction"},
            {"time": 30.0, "event": "Main point"}
        ]
    }

    Patterns:
    {
        "identified_patterns": ["repetition", "crescendo"],
        "pattern_confidence": 0.8
    }

    Summary:
    {
        "overall_assessment": "Well-structured content",
        "rating": 8.5
    }
    """
    
    validation = ResponseValidator.validate_response(mock_claude_response)
    print(f"   Valid: {validation['valid']}")
    print(f"   Extracted blocks: {list(validation['data'].keys()) if validation['data'] else 'None'}")
    
    return True

if __name__ == "__main__":
    test_6_block_structure()