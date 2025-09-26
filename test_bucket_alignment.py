#!/usr/bin/env python3
"""
Test script to validate BucketsPlan.md alignment changes.
Tests critical boundary conditions for the 6s → 9s change.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from rumiai_v2.processors.temporal_compute import (
    calculate_temporal_windows,
    calculate_middle_segments,
    BUCKET_THRESHOLDS
)

def test_boundary_conditions():
    """Test all critical boundary conditions after alignment changes."""

    test_cases = [
        # Duration, Expected Middle Segments
        (3.0, None),   # Hook only
        (6.0, None),   # Changed: was {}, now None
        (6.5, None),   # Changed: had segments, now None
        (8.9, None),   # Still no middle
        (9.0, None),   # Critical boundary - no middle
        (9.1, 3),      # Should have 3 segments
        (12.0, 3),     # 3 segments
        (18.0, 3),     # Boundary - still 3 segments
        (18.1, 4),     # Should have 4 segments
        (33.0, 4),     # Boundary - still 4 segments
        (33.1, 5),     # Should have 5 segments
        (75.0, 5),     # Still 5 segments
        (120.0, 5),    # Capped at 5 segments
    ]

    print("=" * 60)
    print("BUCKET ALIGNMENT TEST - Critical Boundaries")
    print("=" * 60)

    failures = []

    for duration, expected_segments in test_cases:
        # Test window calculation
        windows = calculate_temporal_windows(duration)

        # Test segment calculation
        segments = calculate_middle_segments(duration)

        # Count actual segments
        if segments is None:
            actual_count = None
        elif isinstance(segments, dict):
            actual_count = len(segments)
        else:
            actual_count = 0

        # Check if matches expectation
        success = actual_count == expected_segments
        status = "✅" if success else "❌"

        print(f"{status} {duration:5.1f}s video: ", end="")

        if actual_count is None:
            print(f"No middle segments (as expected)")
        else:
            print(f"{actual_count} segments", end="")
            if not success:
                print(f" (expected {expected_segments})", end="")
                failures.append((duration, expected_segments, actual_count))
            print()

        # Show window structure for key boundaries
        if duration in [6.0, 9.0, 9.1, 18.0, 18.1]:
            print(f"       Windows: Hook={windows['hook']}, "
                  f"Middle={windows['middle']}, "
                  f"Closing={windows['closing']}")
            if segments:
                for name, seg in segments.items():
                    print(f"       {name}: {seg['start']:.2f}-{seg['end']:.2f}s")

    print("\n" + "=" * 60)
    if failures:
        print(f"❌ TEST FAILED: {len(failures)} mismatches")
        for dur, exp, act in failures:
            print(f"   {dur}s: expected {exp}, got {act}")
    else:
        print("✅ ALL TESTS PASSED - Alignment matches BucketsPlan.md")
    print("=" * 60)

    # Print summary of changes
    print("\n📝 SUMMARY OF CHANGES:")
    print("1. Boundary moved from 6s to 9s")
    print("2. Videos 6-9s now have NO middle segments (was empty dict)")
    print("3. Segment count based on TOTAL duration (not middle duration)")
    print("4. New bucket thresholds:")
    for key, value in BUCKET_THRESHOLDS.items():
        print(f"   - {key}: {value}s")

    return len(failures) == 0

if __name__ == "__main__":
    success = test_boundary_conditions()
    sys.exit(0 if success else 1)