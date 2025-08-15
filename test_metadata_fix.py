#!/usr/bin/env python3
"""Test the metadata analysis fixes"""

import json
import sys
sys.path.append('/home/jorge/rumiaifinal')

from rumiai_v2.processors.precompute_functions_full import compute_metadata_analysis_metrics

def test_emoji_detection():
    """Test that 🌺 emoji is now detected"""
    print("Testing emoji detection fix...")
    
    test_data = {
        'metadata_summary': {
            'description': 'Depuff with me 🌺',
            'views': 3200000,
            'likes': 346000,
            'comments': 1234,
            'shares': 567
        },
        'static_metadata': {
            'text': 'Depuff with me 🌺',
            'hashtags': [],
            'createTime': '2024-01-01T12:00:00Z',
            'author': {'username': 'testuser', 'verified': False}
        }
    }
    
    result = compute_metadata_analysis_metrics(
        static_metadata=test_data['static_metadata'],
        metadata_summary=test_data['metadata_summary'],
        video_duration=58.0
    )
    
    # Check critical fixes
    assert result['metadataCoreMetrics']['emojiCount'] == 1, f"Expected 1 emoji, got {result['metadataCoreMetrics']['emojiCount']}"
    assert result['metadataCoreMetrics']['viewCount'] == 3200000, f"Expected 3200000 views, got {result['metadataCoreMetrics']['viewCount']}"
    assert result['metadataCoreMetrics']['engagementRate'] > 0, f"Expected engagement rate > 0, got {result['metadataCoreMetrics']['engagementRate']}"
    
    print("✅ Emoji detection test passed")
    return result

def test_hashtag_strategy():
    """Test hashtag generic/niche breakdown"""
    print("\nTesting hashtag strategy fix...")
    
    test_data = {
        'metadata_summary': {
            'description': 'Test video',
            'views': 10000,
            'likes': 1000,
            'comments': 50,
            'shares': 20
        },
        'static_metadata': {
            'text': 'Test video',
            'hashtags': [
                {'name': 'kidney'},
                {'name': 'healthytea'},
                {'name': 'fyp'},
                {'name': 'tiktokshop'},
                {'name': 'foryou'},
                {'name': 'tea'}
            ],
            'createTime': '2024-01-01T12:00:00Z',
            'author': {}
        }
    }
    
    result = compute_metadata_analysis_metrics(
        static_metadata=test_data['static_metadata'],
        metadata_summary=test_data['metadata_summary'],
        video_duration=30.0
    )
    
    # Check hashtag breakdown exists
    assert 'hashtagBreakdown' in result['metadataPatterns'], "hashtagBreakdown missing from output"
    
    breakdown = result['metadataPatterns']['hashtagBreakdown']
    assert breakdown['generic'] == 2, f"Expected 2 generic hashtags, got {breakdown['generic']}"
    assert breakdown['niche'] == 4, f"Expected 4 niche hashtags, got {breakdown['niche']}"
    assert breakdown['genericRatio'] == 0.33, f"Expected 0.33 ratio, got {breakdown['genericRatio']}"
    
    print("✅ Hashtag strategy test passed")
    return result

def test_cta_detection():
    """Test ML-ready CTA detection"""
    print("\nTesting CTA detection...")
    
    test_data = {
        'metadata_summary': {
            'description': 'Follow me for more content! Drop a like if you enjoyed. Limited time offer!',
            'views': 5000,
            'likes': 500,
            'comments': 25,
            'shares': 10
        },
        'static_metadata': {
            'text': 'Follow me for more content! Drop a like if you enjoyed. Limited time offer!',
            'hashtags': [],
            'createTime': '2024-01-01T12:00:00Z',
            'author': {}
        }
    }
    
    result = compute_metadata_analysis_metrics(
        static_metadata=test_data['static_metadata'],
        metadata_summary=test_data['metadata_summary'],
        video_duration=15.0
    )
    
    # Check CTA features
    assert 'ctaFeatures' in result['metadataPatterns'], "ctaFeatures missing from output"
    
    cta = result['metadataPatterns']['ctaFeatures']
    assert cta['hasCTA'] == 1, "Should detect CTA"
    assert cta['ctaFollow'] == 1, "Should detect follow CTA"
    assert cta['ctaLike'] == 1, "Should detect like CTA"
    assert cta['ctaUrgency'] == 1, "Should detect urgency"
    assert cta['ctaCount'] == 3, f"Expected 3 CTAs, got {cta['ctaCount']}"
    
    print("✅ CTA detection test passed")
    return result

def test_output_structure():
    """Test that output has correct 6-block structure"""
    print("\nTesting output structure...")
    
    test_data = {
        'metadata_summary': {
            'description': 'Simple test',
            'views': 1000,
            'likes': 100,
            'comments': 10,
            'shares': 5
        },
        'static_metadata': {
            'text': 'Simple test',
            'hashtags': [],
            'createTime': '2024-01-01T12:00:00Z',
            'author': {}
        }
    }
    
    result = compute_metadata_analysis_metrics(
        static_metadata=test_data['static_metadata'],
        metadata_summary=test_data['metadata_summary'],
        video_duration=10.0
    )
    
    # Check 6-block structure
    required_blocks = [
        'metadataCoreMetrics',
        'metadataDynamics',
        'metadataInteractions',
        'metadataKeyEvents',
        'metadataPatterns',
        'metadataQuality'
    ]
    
    for block in required_blocks:
        assert block in result, f"Missing required block: {block}"
    
    # Check removed fields are gone
    assert 'sentimentCategory' not in str(result), "sentimentCategory should be removed"
    assert 'urgencyLevel' not in str(result), "urgencyLevel should be removed"
    assert 'viralPotential' not in str(result), "viralPotential should be removed"
    assert 'captionStyle' not in str(result), "captionStyle should be removed"
    
    print("✅ Output structure test passed")
    return result

if __name__ == "__main__":
    print("=" * 50)
    print("METADATA ANALYSIS FIX TESTS")
    print("=" * 50)
    
    try:
        r1 = test_emoji_detection()
        r2 = test_hashtag_strategy()
        r3 = test_cta_detection()
        r4 = test_output_structure()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✅")
        print("=" * 50)
        
        print("\nSample output (emoji test):")
        print(json.dumps(r1, indent=2))
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)