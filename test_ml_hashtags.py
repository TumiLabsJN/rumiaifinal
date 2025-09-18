#!/usr/bin/env python3
"""Test ML-compatible hashtag features."""

import json
from pathlib import Path
from rumiai_v2.processors.temporal_compute import extract_hashtag_metrics, compute_temporal_windows

def test_ml_compatibility():
    """Verify only ML-compatible features are returned."""

    # Test with mixed hashtags
    metadata = {
        'hashtags': [
            {'name': 'fyp'},
            {'name': 'viral'},
            {'name': 'fitness'},
            {'name': 'gym'},
            {'name': 'workout'}
        ]
    }

    result = extract_hashtag_metrics(metadata)

    print("✅ ML-Compatible Hashtag Features:")
    print(json.dumps(result, indent=2))

    # Verify all values are numeric
    assert isinstance(result['hashtag_count'], int)
    assert isinstance(result['generic_hashtag_count'], int)
    assert isinstance(result['specific_hashtag_count'], int)
    assert isinstance(result['generic_ratio'], float)

    # Verify no text fields
    assert 'hashtag_strategy' not in result
    assert 'specific_hashtags' not in result

    print("\n✅ All features are ML-compatible (numeric only)")

    # Test edge cases
    print("\n🧪 Edge Cases:")

    # Empty hashtags
    empty_result = extract_hashtag_metrics({'hashtags': []})
    print(f"  Empty: {empty_result}")
    assert empty_result['hashtag_count'] == 0
    assert empty_result['generic_ratio'] == 0.0

    # All generic
    all_generic = extract_hashtag_metrics({
        'hashtags': [{'name': 'fyp'}, {'name': 'viral'}, {'name': 'trending'}]
    })
    print(f"  All generic: {all_generic}")
    assert all_generic['generic_ratio'] == 1.0

    # All specific
    all_specific = extract_hashtag_metrics({
        'hashtags': [{'name': 'cortisollevels'}, {'name': 'hormonehealth'}]
    })
    print(f"  All specific: {all_specific}")
    assert all_specific['generic_ratio'] == 0.0

    print("\n✅ Edge cases handled correctly")

def test_full_pipeline():
    """Test in full temporal compute pipeline."""

    print("\n📊 Full Pipeline Test:")

    # Load real video
    test_file = Path("unified_analysis/7430952519439846698.json")
    with open(test_file) as f:
        unified_analysis = json.load(f)

    # Run temporal compute
    result = compute_temporal_windows(unified_analysis)

    # Get hashtag analysis
    hashtag_data = result.get('metadata', {}).get('hashtag_analysis', {})

    print("\nHashtag Analysis Output:")
    print(json.dumps(hashtag_data, indent=2))

    # Verify ML compatibility
    if hashtag_data:
        # Check all values are numeric
        for key, value in hashtag_data.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric: {type(value)}"

        # Ensure no non-ML fields
        assert 'specific_hashtags' not in hashtag_data
        assert 'hashtag_strategy' not in hashtag_data

        print("\n✅ Output is fully ML-compatible")

        # Show what ML model would receive
        print("\n🤖 ML Feature Vector:")
        print(f"  [hashtag_count={hashtag_data['hashtag_count']},")
        print(f"   generic_count={hashtag_data['generic_hashtag_count']},")
        print(f"   specific_count={hashtag_data['specific_hashtag_count']},")
        print(f"   generic_ratio={hashtag_data['generic_ratio']}]")

if __name__ == "__main__":
    test_ml_compatibility()
    test_full_pipeline()

    print("\n" + "="*50)
    print("🎉 ML-COMPATIBLE HASHTAG FEATURES CONFIRMED!")
    print("="*50)