#!/usr/bin/env python3
"""Test the average_face_size feature (Phase 2 of personframingfix.md)."""

import json
from pathlib import Path
from rumiai_v2.processors.temporal_compute import compute_temporal_windows

def test_average_face_size_feature():
    """Test that average_face_size appears in all temporal windows."""

    # Load test video
    test_file = Path('unified_analysis/7430952519439846698.json')
    if not test_file.exists():
        print(f"❌ Test file {test_file} not found")
        return False

    with open(test_file) as f:
        data = json.load(f)

    # Process with temporal compute
    result = compute_temporal_windows(data)

    print("=== Testing average_face_size Feature ===\n")

    all_passed = True

    # Check hook window
    hook = result['temporal_windows'].get('hook')
    if hook:
        avg_size = hook.get('average_face_size')
        close_ratio = hook.get('close_ratio', 0)
        print(f"Hook Window (0-3s):")
        print(f"  average_face_size: {avg_size}")
        print(f"  close_ratio: {close_ratio}")

        if avg_size is None:
            print("  ❌ MISSING average_face_size")
            all_passed = False
        else:
            # Verify it's in reasonable range (0-1, typically 0.05-0.40)
            if 0 <= avg_size <= 1:
                print(f"  ✓ Value in valid range: {avg_size:.4f}")
            else:
                print(f"  ❌ Value out of range: {avg_size}")
                all_passed = False

            # Check correlation with framing ratios
            if avg_size > 0.25 and close_ratio < 0.5:
                print(f"  ⚠️  Large face but low close_ratio - may need calibration")
            elif avg_size < 0.08 and close_ratio > 0.5:
                print(f"  ⚠️  Small face but high close_ratio - may need calibration")

    # Check middle segments
    middle_segments = result['temporal_windows'].get('middle_segments', [])
    print(f"\nMiddle Segments: {len(middle_segments)} segments")

    for i, segment in enumerate(middle_segments):
        avg_size = segment.get('average_face_size')
        close_ratio = segment.get('close_ratio', 0)
        print(f"\n  Segment {i+1} ({segment.get('start')}-{segment.get('end')}s):")
        print(f"    average_face_size: {avg_size}")
        print(f"    close_ratio: {close_ratio}")

        if avg_size is None:
            print("    ❌ MISSING average_face_size")
            all_passed = False
        elif 0 <= avg_size <= 1:
            print(f"    ✓ Valid: {avg_size:.4f}")
        else:
            print(f"    ❌ Out of range: {avg_size}")
            all_passed = False

    # Check closing window
    closing = result['temporal_windows'].get('closing')
    if closing:
        avg_size = closing.get('average_face_size')
        close_ratio = closing.get('close_ratio', 0)
        print(f"\nClosing Window ({closing.get('start')}-{closing.get('end')}s):")
        print(f"  average_face_size: {avg_size}")
        print(f"  close_ratio: {close_ratio}")

        if avg_size is None:
            print("  ❌ MISSING average_face_size")
            all_passed = False
        elif 0 <= avg_size <= 1:
            print(f"  ✓ Valid: {avg_size:.4f}")
        else:
            print(f"  ❌ Out of range: {avg_size}")
            all_passed = False

    # Test correlation patterns
    print("\n=== ML Pattern Analysis ===")

    # Collect all average_face_sizes
    sizes = []
    if hook and hook.get('average_face_size') is not None:
        sizes.append(('hook', hook.get('average_face_size')))

    for i, seg in enumerate(middle_segments):
        if seg.get('average_face_size') is not None:
            sizes.append((f'middle_{i+1}', seg.get('average_face_size')))

    if closing and closing.get('average_face_size') is not None:
        sizes.append(('closing', closing.get('average_face_size')))

    if len(sizes) >= 2:
        # Check for patterns
        print(f"Face size progression: {' → '.join([f'{name}:{size:.3f}' for name, size in sizes])}")

        # Detect if it's increasing (zoom in), decreasing (zoom out), or stable
        first_size = sizes[0][1]
        last_size = sizes[-1][1]

        if last_size > first_size * 1.2:
            print("Pattern: ZOOM IN (face getting larger)")
        elif last_size < first_size * 0.8:
            print("Pattern: ZOOM OUT (face getting smaller)")
        else:
            print("Pattern: STABLE (consistent framing)")

    if all_passed:
        print("\n✅ All tests passed! average_face_size feature working correctly.")
    else:
        print("\n❌ Some tests failed. Check implementation.")

    return all_passed

if __name__ == "__main__":
    import sys
    success = test_average_face_size_feature()
    sys.exit(0 if success else 1)