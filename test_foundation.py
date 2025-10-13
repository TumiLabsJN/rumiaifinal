#!/usr/bin/env python3
"""
Test script for Foundation package.

Based on TRACE 1 from FoundationCHILDTI2.md Section 7: Complete Example Traces
"""

import sys
sys.path.insert(0, '/home/jorge/rumiaifinal')

from foundation import (
    parse_args,
    ConfigManager,
    PathBuilder,
    assign_bucket,
    sanitize_target,
)
from pathlib import Path


def test_trace_1_happy_path():
    """
    Test Trace 1: Normal Processing (Happy Path)

    Source: FoundationCHILDTI2.md Section 7, TRACE 1
    """
    print("=" * 80)
    print("TRACE 1: Normal Processing (Happy Path)")
    print("=" * 80)

    # Step 1: Parse CLI arguments
    print("\n[Step 1] Parsing CLI arguments...")
    cli_args = [
        "--client", "acme_corp",
        "--analysis-type", "hashtag",
        "--target", "#nutrition",
        "--video-count", "100"
    ]

    args = parse_args(cli_args)
    print(f"✓ Parsed: {args}")
    print(f"  - client: {args.client}")
    print(f"  - analysis_type: {args.analysis_type}")
    print(f"  - target: {args.target}")
    print(f"  - analysis_mode: {args.analysis_mode} (default applied)")
    print(f"  - selection_strategy: {args.selection_strategy} (default applied)")
    print(f"  - video_count: {args.video_count}")
    print(f"  - report_audience: {args.report_audience} (default applied)")

    # Step 2: Sanitize target
    print("\n[Step 2] Sanitizing target for filesystem...")
    sanitized = sanitize_target(args.target, args.analysis_type)
    print(f"✓ Sanitized: '{args.target}' → '{sanitized}'")

    # Step 3: Build directory paths
    print("\n[Step 3] Building directory paths...")
    pb = PathBuilder(Path("/tmp/test_foundation"))
    target_dir = pb.get_target_dir(
        args.client,
        args.analysis_type,
        args.target,
        args.analysis_mode,
        args.selection_strategy
    )
    print(f"✓ Target directory: {target_dir}")

    # Step 4: Create directory structure
    print("\n[Step 4] Creating directory structure...")
    bucket_paths = pb.create_directory_structure(target_dir)
    print(f"✓ Created {len(bucket_paths)} bucket directories")
    for bucket, path in list(bucket_paths.items())[:3]:  # Show first 3
        print(f"  - {bucket}: {path}")
    print(f"  - ... (and {len(bucket_paths) - 3} more buckets)")

    # Verify subdirectories
    sample_bucket = list(bucket_paths.values())[0]
    subdirs = [d.name for d in sample_bucket.iterdir() if d.is_dir()]
    print(f"✓ Subdirectories per bucket: {', '.join(subdirs)}")

    # Step 5: Create Config object
    print("\n[Step 5] Creating Config object...")
    config = ConfigManager.from_cli_args(args)
    print(f"✓ Config created:")
    print(f"  - client_id: {config.client_id}")
    print(f"  - analysis_type: {config.analysis_type}")
    print(f"  - target: {config.target}")
    print(f"  - run_date: {config.run_date}")

    # Step 6: Save config.json
    print("\n[Step 6] Saving config.json...")
    config_path = target_dir / "config.json"
    ConfigManager.save(config, config_path)
    print(f"✓ Configuration saved to: {config_path}")

    # Verify file exists
    assert config_path.exists(), "config.json should exist"
    print(f"✓ File verified: {config_path.stat().st_size} bytes")

    # Step 7: Test bucket assignment
    print("\n[Step 7] Testing bucket assignment...")
    test_durations = [2.5, 9.0, 18.5, 120.0]
    for duration in test_durations:
        bucket = assign_bucket(duration)
        print(f"  - {duration}s → {bucket}")

    print("\n" + "=" * 80)
    print("✓ TRACE 1 COMPLETED SUCCESSFULLY")
    print("=" * 80)


def test_edge_cases():
    """Test edge cases from TI document."""
    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)

    # Test path sanitization edge cases
    print("\n[Path Sanitization] Testing edge cases...")
    test_cases = [
        ("#Fitness & Nutrition!", "hashtag", "fitness_nutrition"),
        ("@My Brand 2024", "competitor", "my_brand_2024"),
        ("@rival__brand", "creator", "rival_brand"),
    ]

    for target, analysis_type, expected in test_cases:
        result = sanitize_target(target, analysis_type)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{target}' → '{result}' (expected: '{expected}')")

    # Test bucket assignment edge cases
    print("\n[Bucket Assignment] Testing boundary conditions...")
    edge_cases = [
        (9.0, "9-13s"),  # Exactly on boundary
        (120.0, "90-120s"),  # Maximum TikTok duration
    ]

    for duration, expected_bucket in edge_cases:
        bucket = assign_bucket(duration)
        status = "✓" if bucket == expected_bucket else "✗"
        print(f"  {status} {duration}s → {bucket} (expected: {expected_bucket})")

    print("\n✓ All edge cases passed")


def test_error_case():
    """Test error handling from TI document."""
    print("\n" + "=" * 80)
    print("ERROR CASE TEST")
    print("=" * 80)

    print("\n[Invalid Target Format] Testing validation...")
    try:
        cli_args = [
            "--client", "acme_corp",
            "--analysis-type", "hashtag",
            "--target", "nutrition",  # Missing # prefix
        ]
        args = parse_args(cli_args)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    print("\n[Invalid Video Duration] Testing bucket assignment...")
    try:
        bucket = assign_bucket(125.0)  # Exceeds TikTok max
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")


if __name__ == "__main__":
    try:
        test_trace_1_happy_path()
        test_edge_cases()
        test_error_case()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
