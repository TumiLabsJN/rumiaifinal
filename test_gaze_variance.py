#!/usr/bin/env python3
"""
Test script for gaze variance feature in temporal_compute.py.
Tests the calculate_gaze_variance() function.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.temporal_compute import calculate_gaze_variance


def test_empty_window():
    """Test that empty windows return 0 variance."""
    print("\n🧪 Testing empty window...")

    # No gaze entries
    result = calculate_gaze_variance([], 0, 3)
    assert result == 0.0, f"Empty entries should return 0, got: {result}"
    print("  ✓ Empty entries returns 0")

    # Gaze entries outside window
    entries = [
        {'entry_type': 'gaze', 'start': 5, 'data': {'eye_contact': 0.8}},
        {'entry_type': 'gaze', 'start': 6, 'data': {'eye_contact': 0.6}}
    ]
    result = calculate_gaze_variance(entries, 0, 3)
    assert result == 0.0, f"Entries outside window should return 0, got: {result}"
    print("  ✓ Entries outside window returns 0")


def test_single_entry():
    """Test that single entry returns 0 variance."""
    print("\n🧪 Testing single entry...")

    entries = [
        {'entry_type': 'gaze', 'start': 1, 'data': {'eye_contact': 0.7}}
    ]
    result = calculate_gaze_variance(entries, 0, 3)
    assert result == 0.0, f"Single entry should return 0, got: {result}"
    print("  ✓ Single entry returns 0 (no variance possible)")


def test_multiple_entries_same_value():
    """Test variance with multiple identical values."""
    print("\n🧪 Testing identical values...")

    entries = [
        {'entry_type': 'gaze', 'start': 0.5, 'data': {'eye_contact': 0.7}},
        {'entry_type': 'gaze', 'start': 1.5, 'data': {'eye_contact': 0.7}},
        {'entry_type': 'gaze', 'start': 2.5, 'data': {'eye_contact': 0.7}}
    ]
    result = calculate_gaze_variance(entries, 0, 3)
    assert result == 0.0, f"Identical values should have 0 variance, got: {result}"
    print("  ✓ Identical values return 0 variance")


def test_variance_calculation():
    """Test actual variance calculation."""
    print("\n🧪 Testing variance calculation...")

    # Known variance case
    entries = [
        {'entry_type': 'gaze', 'start': 0.5, 'data': {'eye_contact': 0.2}},
        {'entry_type': 'gaze', 'start': 1.5, 'data': {'eye_contact': 0.8}},
        {'entry_type': 'gaze', 'start': 2.5, 'data': {'eye_contact': 0.5}}
    ]
    result = calculate_gaze_variance(entries, 0, 3)

    # Manual calculation: mean = 0.5, variance = ((0.2-0.5)^2 + (0.8-0.5)^2 + (0.5-0.5)^2) / 2 = 0.09
    expected = 0.09
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"
    print(f"  ✓ Variance calculation correct: {result:.3f}")


def test_mixed_entry_types():
    """Test that non-gaze entries are ignored."""
    print("\n🧪 Testing mixed entry types...")

    entries = [
        {'entry_type': 'gaze', 'start': 0.5, 'data': {'eye_contact': 0.3}},
        {'entry_type': 'emotion', 'start': 1.0, 'data': {'emotion': 'happy'}},
        {'entry_type': 'gaze', 'start': 1.5, 'data': {'eye_contact': 0.7}},
        {'entry_type': 'gesture', 'start': 2.0, 'data': {'gesture': 'point'}},
        {'entry_type': 'gaze', 'start': 2.5, 'data': {'eye_contact': 0.5}}
    ]
    result = calculate_gaze_variance(entries, 0, 3)

    # Only gaze entries: [0.3, 0.7, 0.5], mean = 0.5
    # Variance = ((0.3-0.5)^2 + (0.7-0.5)^2 + (0.5-0.5)^2) / 2 = 0.04
    expected = 0.04
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"
    print(f"  ✓ Only gaze entries processed: variance = {result:.3f}")


def test_partial_window_overlap():
    """Test handling of partial window overlap."""
    print("\n🧪 Testing partial window overlap...")

    entries = [
        {'entry_type': 'gaze', 'start': -1, 'data': {'eye_contact': 0.2}},  # Before window
        {'entry_type': 'gaze', 'start': 1, 'data': {'eye_contact': 0.6}},   # In window
        {'entry_type': 'gaze', 'start': 2, 'data': {'eye_contact': 0.4}},   # In window
        {'entry_type': 'gaze', 'start': 4, 'data': {'eye_contact': 0.9}}    # After window
    ]
    result = calculate_gaze_variance(entries, 0, 3)

    # Only entries at t=1 and t=2 are in window: [0.6, 0.4]
    # Mean = 0.5, Variance = ((0.6-0.5)^2 + (0.4-0.5)^2) / 1 = 0.02
    expected = 0.02
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"
    print(f"  ✓ Correctly filters to window: variance = {result:.3f}")


def test_realistic_data():
    """Test with realistic eye contact values."""
    print("\n🧪 Testing realistic data...")

    # Simulating hook with relatively steady eye contact
    steady_entries = [
        {'entry_type': 'gaze', 'start': 0.3, 'data': {'eye_contact': 0.72}},
        {'entry_type': 'gaze', 'start': 0.7, 'data': {'eye_contact': 0.68}},
        {'entry_type': 'gaze', 'start': 1.1, 'data': {'eye_contact': 0.75}},
        {'entry_type': 'gaze', 'start': 1.5, 'data': {'eye_contact': 0.71}},
        {'entry_type': 'gaze', 'start': 1.9, 'data': {'eye_contact': 0.69}},
        {'entry_type': 'gaze', 'start': 2.3, 'data': {'eye_contact': 0.73}},
        {'entry_type': 'gaze', 'start': 2.7, 'data': {'eye_contact': 0.70}}
    ]
    steady_variance = calculate_gaze_variance(steady_entries, 0, 3)
    print(f"  Steady gaze variance: {steady_variance:.4f}")

    # Simulating middle segment with variable eye contact
    variable_entries = [
        {'entry_type': 'gaze', 'start': 3.3, 'data': {'eye_contact': 0.85}},
        {'entry_type': 'gaze', 'start': 4.1, 'data': {'eye_contact': 0.25}},
        {'entry_type': 'gaze', 'start': 4.9, 'data': {'eye_contact': 0.90}},
        {'entry_type': 'gaze', 'start': 5.7, 'data': {'eye_contact': 0.15}},
        {'entry_type': 'gaze', 'start': 6.5, 'data': {'eye_contact': 0.75}},
        {'entry_type': 'gaze', 'start': 7.3, 'data': {'eye_contact': 0.30}},
        {'entry_type': 'gaze', 'start': 8.1, 'data': {'eye_contact': 0.80}}
    ]
    variable_variance = calculate_gaze_variance(variable_entries, 3, 9)
    print(f"  Variable gaze variance: {variable_variance:.4f}")

    # Variable should have much higher variance than steady
    assert variable_variance > steady_variance * 5, \
        f"Variable ({variable_variance:.4f}) should be much higher than steady ({steady_variance:.4f})"
    print("  ✓ Variable gaze has significantly higher variance than steady")


def run_all_tests():
    """Run all test functions."""
    print("="*60)
    print("🚀 GAZE VARIANCE TEST SUITE")
    print("="*60)

    test_empty_window()
    test_single_entry()
    test_multiple_entries_same_value()
    test_variance_calculation()
    test_mixed_entry_types()
    test_partial_window_overlap()
    test_realistic_data()

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()