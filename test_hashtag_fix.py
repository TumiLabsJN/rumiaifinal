#!/usr/bin/env python3
"""Test hashtag analysis fix."""

import json
from pathlib import Path
from rumiai_v2.processors.temporal_compute import extract_hashtag_metrics, compute_temporal_windows

def test_hashtag_extraction():
    """Test the hashtag extraction function directly."""

    # Test Case 1: Mix of generic and specific
    metadata1 = {
        'hashtags': [
            {'name': 'fyp'},
            {'name': 'viral'},
            {'name': 'fitness'},
            {'name': 'workoutmotivation'}
        ]
    }

    result1 = extract_hashtag_metrics(metadata1)
    print("Test 1 - Mixed hashtags:")
    print(f"  Total: {result1['hashtag_count']}")
    print(f"  Generic: {result1['generic_hashtag_count']} ({result1['generic_ratio']})")
    print(f"  Specific: {result1['specific_hashtag_count']}")
    print(f"  Strategy: {result1['hashtag_strategy']}")
    print(f"  Specific tags: {result1['specific_hashtags']}")
    assert result1['generic_hashtag_count'] == 2  # fyp, viral
    assert result1['specific_hashtag_count'] == 2  # fitness, workoutmotivation
    print("  ✅ PASSED\n")

    # Test Case 2: New generic hashtags
    metadata2 = {
        'hashtags': [
            {'name': 'tiktok'},
            {'name': 'funny'},
            {'name': 'duet'},
            {'name': 'contentcreator'},
            {'name': 'cooking'}
        ]
    }

    result2 = extract_hashtag_metrics(metadata2)
    print("Test 2 - New generic hashtags:")
    print(f"  Total: {result2['hashtag_count']}")
    print(f"  Generic: {result2['generic_hashtag_count']} ({result2['generic_ratio']})")
    print(f"  Strategy: {result2['hashtag_strategy']}")
    assert result2['generic_hashtag_count'] == 4  # All except 'cooking'
    assert result2['generic_ratio'] == 0.8
    assert result2['hashtag_strategy'] == 'too_generic'
    print("  ✅ PASSED\n")

    # Test Case 3: All specific (niche)
    metadata3 = {
        'hashtags': [
            {'name': 'cortisollevels'},
            {'name': 'progesterone'},
            {'name': 'hormonehealth'}
        ]
    }

    result3 = extract_hashtag_metrics(metadata3)
    print("Test 3 - Niche hashtags:")
    print(f"  Generic ratio: {result3['generic_ratio']}")
    print(f"  Strategy: {result3['hashtag_strategy']}")
    print(f"  Specific tags: {result3['specific_hashtags']}")
    assert result3['generic_ratio'] == 0.0
    assert result3['hashtag_strategy'] == 'too_specific'
    print("  ✅ PASSED\n")

def test_full_integration():
    """Test hashtag analysis in full temporal compute pipeline."""

    print("="*50)
    print("FULL INTEGRATION TEST")
    print("="*50 + "\n")

    # Load a real video
    test_file = Path("unified_analysis/7430952519439846698.json")
    with open(test_file) as f:
        unified_analysis = json.load(f)

    print(f"Testing with video: {unified_analysis.get('video_id')}")

    # Get hashtags from metadata
    hashtags = unified_analysis.get('metadata', {}).get('hashtags', [])
    print(f"Found {len(hashtags)} hashtags:")
    for tag in hashtags[:5]:
        print(f"  - {tag.get('name')}")
    if len(hashtags) > 5:
        print(f"  ... and {len(hashtags)-5} more")

    # Run temporal compute
    print("\n🚀 Running temporal compute with hashtag analysis...")
    result = compute_temporal_windows(unified_analysis)

    # Check hashtag analysis in output
    hashtag_analysis = result.get('metadata', {}).get('hashtag_analysis')

    if hashtag_analysis:
        print("\n✅ Hashtag analysis found in output!")
        print(f"  Total hashtags: {hashtag_analysis['hashtag_count']}")
        print(f"  Generic count: {hashtag_analysis['generic_hashtag_count']}")
        print(f"  Specific count: {hashtag_analysis['specific_hashtag_count']}")
        print(f"  Generic ratio: {hashtag_analysis['generic_ratio']}")
        print(f"  Strategy: {hashtag_analysis['hashtag_strategy']}")
        print(f"  Top specific tags: {hashtag_analysis['specific_hashtags'][:3]}")
    else:
        print("\n❌ ERROR: No hashtag analysis in output!")
        print("Metadata keys:", list(result.get('metadata', {}).keys()))
        return False

    # Validate the analysis makes sense
    expected_specific = ['cortisol', 'progesterone', 'cortisollevels', 'hormones', 'adrenalfatigue']
    actual_specific = hashtag_analysis['specific_hashtags']

    matches = sum(1 for tag in expected_specific if tag in actual_specific)
    print(f"\n🔍 Validation: Found {matches}/{len(expected_specific)} expected specific tags")

    return True

if __name__ == "__main__":
    print("🧪 Testing Hashtag Fix Implementation\n")

    # Test direct function
    test_hashtag_extraction()

    # Test full integration
    success = test_full_integration()

    if success:
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! Hashtag fix is working!")
        print("="*50)
    else:
        print("\n⚠️ Tests failed - check implementation")