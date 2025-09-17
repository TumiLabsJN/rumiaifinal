#!/usr/bin/env python3
"""
Test script for speech content indicators in temporal_compute.py.
Tests the calculate_speech_content_indicators() function.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.temporal_compute import calculate_speech_content_indicators


def test_empty_window():
    """Test that empty windows return all zeros."""
    print("\n🧪 Testing empty window...")

    # No speech segments
    result = calculate_speech_content_indicators([], 0, 3, 3)
    assert result == {
        'has_greeting': 0,
        'has_question': 0,
        'has_instruction': 0,
        'has_speech_cta': 0
    }, f"Empty window should return all zeros, got: {result}"
    print("  ✓ Empty segments returns all zeros")

    # Speech segments outside window
    segments = [{'start': 5, 'end': 8, 'text': 'Hello there'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result == {
        'has_greeting': 0,
        'has_question': 0,
        'has_instruction': 0,
        'has_speech_cta': 0
    }, f"Segments outside window should return all zeros, got: {result}"
    print("  ✓ Segments outside window returns all zeros")


def test_greeting_detection():
    """Test greeting detection."""
    print("\n🧪 Testing greeting detection...")

    # Basic greeting
    segments = [{'start': 0, 'end': 3, 'text': 'Hey everyone welcome back'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_greeting'] == 1, "Should detect 'hey'"
    print("  ✓ Detects 'hey'")

    # Hello greeting
    segments = [{'start': 0, 'end': 3, 'text': 'Hello there my friends'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_greeting'] == 1, "Should detect 'hello'"
    print("  ✓ Detects 'hello'")

    # Hi with space (avoid 'high')
    segments = [{'start': 0, 'end': 3, 'text': 'Hi guys welcome'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_greeting'] == 1, "Should detect 'hi '"
    print("  ✓ Detects 'hi ' (with space)")

    # Should NOT detect 'high' as greeting
    segments = [{'start': 0, 'end': 3, 'text': 'High quality content'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_greeting'] == 0, "Should NOT detect 'high' as greeting"
    print("  ✓ Does not detect 'high' as greeting")

    # Greeting not at start (after 50 chars) should not be detected
    segments = [{'start': 0, 'end': 3, 'text': 'This is a very long introduction that goes on and on and then says hello'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_greeting'] == 0, "Should not detect greeting after 50 chars"
    print("  ✓ Ignores greetings after first 50 chars")


def test_question_detection():
    """Test question detection."""
    print("\n🧪 Testing question detection...")

    # Question mark
    segments = [{'start': 0, 'end': 3, 'text': 'Did you know this?'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_question'] == 1, "Should detect '?'"
    print("  ✓ Detects question mark")

    # Question words
    segments = [{'start': 0, 'end': 3, 'text': 'How do we fix this'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_question'] == 1, "Should detect 'how '"
    print("  ✓ Detects 'how '")

    segments = [{'start': 0, 'end': 3, 'text': 'What is the best way'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_question'] == 1, "Should detect 'what '"
    print("  ✓ Detects 'what '")

    # No question
    segments = [{'start': 0, 'end': 3, 'text': 'This is a statement'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_question'] == 0, "Should not detect question"
    print("  ✓ No false positives for statements")


def test_instruction_detection():
    """Test instruction detection."""
    print("\n🧪 Testing instruction detection...")

    # First/then/next
    segments = [{'start': 0, 'end': 3, 'text': 'First open the settings'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_instruction'] == 1, "Should detect 'first '"
    print("  ✓ Detects 'first '")

    segments = [{'start': 0, 'end': 3, 'text': 'Then click on preferences'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_instruction'] == 1, "Should detect 'then '"
    print("  ✓ Detects 'then '")

    # Make sure / don't forget
    segments = [{'start': 0, 'end': 3, 'text': "Make sure you save the file"}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_instruction'] == 1, "Should detect 'make sure'"
    print("  ✓ Detects 'make sure'")

    segments = [{'start': 0, 'end': 3, 'text': "Don't forget to subscribe"}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_instruction'] == 1, "Should detect 'don't forget'"
    print("  ✓ Detects 'don't forget'")


def test_cta_detection():
    """Test call-to-action detection."""
    print("\n🧪 Testing CTA detection...")

    # Subscribe/follow/like
    segments = [{'start': 0, 'end': 3, 'text': 'Please subscribe to my channel'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_speech_cta'] == 1, "Should detect 'subscribe'"
    print("  ✓ Detects 'subscribe'")

    segments = [{'start': 0, 'end': 3, 'text': 'Follow me for more'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_speech_cta'] == 1, "Should detect 'follow'"
    print("  ✓ Detects 'follow'")

    segments = [{'start': 0, 'end': 3, 'text': 'Like this video if helpful'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_speech_cta'] == 1, "Should detect 'like'"
    print("  ✓ Detects 'like'")

    # Link in bio
    segments = [{'start': 0, 'end': 3, 'text': 'Link in bio for details'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_speech_cta'] == 1, "Should detect 'link in bio'"
    print("  ✓ Detects 'link in bio'")

    # No CTA
    segments = [{'start': 0, 'end': 3, 'text': 'This is just information'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_speech_cta'] == 0, "Should not detect CTA"
    print("  ✓ No false positives for regular content")


def test_case_insensitive():
    """Test that detection is case insensitive."""
    print("\n🧪 Testing case insensitivity...")

    # Uppercase greeting
    segments = [{'start': 0, 'end': 3, 'text': 'HELLO EVERYONE'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_greeting'] == 1, "Should detect uppercase HELLO"
    print("  ✓ Detects uppercase HELLO")

    # Mixed case CTA
    segments = [{'start': 0, 'end': 3, 'text': 'Please SUBSCRIBE and Follow'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_speech_cta'] == 1, "Should detect mixed case SUBSCRIBE"
    print("  ✓ Detects mixed case SUBSCRIBE")


def test_multiple_indicators():
    """Test that multiple indicators can be detected in same window."""
    print("\n🧪 Testing multiple indicators...")

    segments = [{'start': 0, 'end': 3, 'text': 'Hey guys! How do you like this? First step: subscribe!'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)

    assert result['has_greeting'] == 1, "Should detect greeting"
    assert result['has_question'] == 1, "Should detect question"
    assert result['has_instruction'] == 1, "Should detect instruction (step)"
    assert result['has_speech_cta'] == 1, "Should detect CTA"

    print("  ✓ Can detect all 4 indicators in same window")


def test_partial_window_overlap():
    """Test handling of segments that partially overlap the window."""
    print("\n🧪 Testing partial window overlap...")

    # Segment starts before window
    segments = [{'start': -1, 'end': 2, 'text': 'Hello there everyone'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_greeting'] == 1, "Should detect greeting from partial segment"
    print("  ✓ Handles segment starting before window")

    # Segment ends after window
    segments = [{'start': 2, 'end': 5, 'text': 'Please subscribe now'}]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_speech_cta'] == 1, "Should detect CTA from partial segment"
    print("  ✓ Handles segment ending after window")

    # Multiple segments, some overlapping
    segments = [
        {'start': -1, 'end': 1, 'text': 'Hey there'},
        {'start': 1, 'end': 2, 'text': 'How are you'},
        {'start': 2.5, 'end': 4, 'text': 'Follow me'}
    ]
    result = calculate_speech_content_indicators(segments, 0, 3, 3)
    assert result['has_greeting'] == 1, "Should detect greeting"
    assert result['has_question'] == 1, "Should detect question"
    assert result['has_speech_cta'] == 1, "Should detect CTA"
    print("  ✓ Handles multiple overlapping segments")


def run_all_tests():
    """Run all test functions."""
    print("="*60)
    print("🚀 SPEECH CONTENT INDICATORS TEST SUITE")
    print("="*60)

    test_empty_window()
    test_greeting_detection()
    test_question_detection()
    test_instruction_detection()
    test_cta_detection()
    test_case_insensitive()
    test_multiple_indicators()
    test_partial_window_overlap()

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()