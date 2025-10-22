#!/usr/bin/env python3
"""
Test Stage 7 Integration in rumiai_ml_batch.py

This script tests the Stage 7 helper functions and validates the integration
without actually calling the LLM API.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.bucket_definitions import BUCKET_WINDOWS


def setup_test_bucket(bucket_path: Path, bucket_name: str):
    """Create mock Stage 6 outputs for testing."""
    ml_analysis_dir = bucket_path / "ml_analysis"
    ml_analysis_dir.mkdir(parents=True, exist_ok=True)

    windows = BUCKET_WINDOWS[bucket_name]

    # Create video-level RF JSON
    with open(ml_analysis_dir / "rf_video_analysis.json", 'w') as f:
        json.dump({
            "bucket": bucket_name,
            "feature_importance": [
                {"feature": f"feature_{i}", "importance": 0.1} for i in range(10)
            ]
        }, f)

    # Create window-level RF and K-Means JSONs
    for window in windows:
        # RF analysis
        with open(ml_analysis_dir / f"{window}_rf_analysis.json", 'w') as f:
            json.dump({
                "window_type": window,
                "bucket": bucket_name,
                "feature_importance": [
                    {
                        "feature": "eye_contact_rate",
                        "importance": 0.35,
                        "rank": 1,
                        "top_performer_avg": 0.88,
                        "bottom_performer_avg": 0.45,
                        "gap": 0.43,
                        "distribution": {
                            "top_performers": {
                                "high_percentage": 0.75,
                                "low_percentage": 0.15
                            }
                        }
                    }
                ] * 10
            }, f)

        # K-Means analysis
        with open(ml_analysis_dir / f"{window}_kmeans_analysis.json", 'w') as f:
            json.dump({
                "window_type": window,
                "bucket": bucket_name,
                "n_clusters": 3,
                "total_videos": 100,
                "clusters": [
                    {
                        "cluster_id": i,
                        "size": 33,
                        "video_ids": [f"video_{i}_{j}" for j in range(33)],
                        "centroid": {
                            "eye_contact_rate": 0.85 if i == 0 else 0.45,
                            "word_count": 120 if i == 0 else 80
                        },
                        "high_contrast_features": [
                            {"feature": "eye_contact_rate", "centroid_value": 0.85}
                        ]
                    } for i in range(3)
                ]
            }, f)

    print(f"✓ Created {1 + len(windows) * 2} mock Stage 6 JSONs for {bucket_name}")


def create_stage7_outputs(bucket_path: Path, bucket_name: str):
    """Create mock Stage 7 outputs for testing skip-if-complete logic."""
    llm_output_dir = bucket_path / "ml_analysis/llm"
    llm_output_dir.mkdir(parents=True, exist_ok=True)

    windows = BUCKET_WINDOWS[bucket_name]

    # Create Phase 1 window analyses
    for window in windows:
        with open(llm_output_dir / f"{window}_analysis.json", 'w') as f:
            json.dump({
                "window_type": window,
                "bucket": bucket_name,
                "clusters": [
                    {"cluster_id": i, "name": f"Cluster {i}"} for i in range(3)
                ]
            }, f)

    # Create Phase 2 synthesis
    with open(llm_output_dir / "synthesis.json", 'w') as f:
        json.dump({
            "scenario": "A",
            "bucket": bucket_name,
            "winning_formulas": ["Formula 1", "Formula 2"]
        }, f)

    # Create complete analysis
    with open(llm_output_dir / "complete_analysis.json", 'w') as f:
        json.dump({
            "bucket": bucket_name,
            "phase1_window_analyses": {},
            "phase2_synthesis": {}
        }, f)

    print(f"✓ Created {len(windows) + 2} mock Stage 7 outputs for {bucket_name}")


def test_validate_stage7_prerequisites():
    """Test Stage 7 prerequisite validation."""
    print("\n" + "="*80)
    print("TEST 1: validate_stage7_prerequisites()")
    print("="*80)

    from rumiai_ml_batch import validate_stage7_prerequisites

    with tempfile.TemporaryDirectory() as tmpdir:
        bucket_path = Path(tmpdir) / "bucket_18-33s"
        bucket_path.mkdir()

        # Test 1a: Missing prerequisites (should raise FileNotFoundError)
        print("\nTest 1a: Missing prerequisites...")
        try:
            validate_stage7_prerequisites(str(bucket_path), "18-33s")
            print("✗ FAIL: Should have raised FileNotFoundError")
            return False
        except FileNotFoundError as e:
            print(f"✓ PASS: Correctly raised FileNotFoundError")
            print(f"  Message: {str(e)[:100]}...")

        # Test 1b: All prerequisites present (should pass)
        print("\nTest 1b: All prerequisites present...")
        setup_test_bucket(bucket_path, "18-33s")
        try:
            validate_stage7_prerequisites(str(bucket_path), "18-33s")
            print("✓ PASS: Validation passed with all prerequisites present")
        except Exception as e:
            print(f"✗ FAIL: Unexpected error - {e}")
            return False

        # Test 1c: Invalid bucket name (should raise ValueError)
        print("\nTest 1c: Invalid bucket name...")
        try:
            validate_stage7_prerequisites(str(bucket_path), "invalid-bucket")
            print("✗ FAIL: Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"✓ PASS: Correctly raised ValueError for invalid bucket")

    return True


def test_validate_stage7_outputs():
    """Test Stage 7 output validation."""
    print("\n" + "="*80)
    print("TEST 2: validate_stage7_outputs()")
    print("="*80)

    from rumiai_ml_batch import validate_stage7_outputs

    with tempfile.TemporaryDirectory() as tmpdir:
        bucket_path = Path(tmpdir) / "bucket_18-33s"
        bucket_path.mkdir()

        # Test 2a: Missing outputs (should raise AssertionError)
        print("\nTest 2a: Missing outputs...")
        try:
            validate_stage7_outputs(str(bucket_path), "18-33s")
            print("✗ FAIL: Should have raised AssertionError")
            return False
        except AssertionError as e:
            print(f"✓ PASS: Correctly raised AssertionError")
            print(f"  Message: {str(e)[:100]}...")

        # Test 2b: All outputs present (should pass)
        print("\nTest 2b: All outputs present...")
        create_stage7_outputs(bucket_path, "18-33s")
        try:
            validate_stage7_outputs(str(bucket_path), "18-33s")
            print("✓ PASS: Validation passed with all outputs present")
        except Exception as e:
            print(f"✗ FAIL: Unexpected error - {e}")
            return False

        # Test 2c: Malformed Phase 1 output (wrong cluster count)
        print("\nTest 2c: Malformed Phase 1 output...")
        llm_output_dir = bucket_path / "ml_analysis/llm"
        with open(llm_output_dir / "hook_analysis.json", 'w') as f:
            json.dump({
                "window_type": "hook",
                "bucket": "18-33s",
                "clusters": [{"cluster_id": 0}, {"cluster_id": 1}]  # Only 2 clusters (should be 3)
            }, f)

        try:
            validate_stage7_outputs(str(bucket_path), "18-33s")
            print("✗ FAIL: Should have raised AssertionError for wrong cluster count")
            return False
        except AssertionError as e:
            print(f"✓ PASS: Correctly detected wrong cluster count")
            print(f"  Message: {str(e)[:100]}...")

    return True


def test_error_handling():
    """Test Stage 7 error handling functions."""
    print("\n" + "="*80)
    print("TEST 3: handle_stage7_error() and cleanup_stage7_partial_outputs()")
    print("="*80)

    from rumiai_ml_batch import handle_stage7_error, cleanup_stage7_partial_outputs

    with tempfile.TemporaryDirectory() as tmpdir:
        bucket_path = Path(tmpdir) / "bucket_18-33s"
        bucket_path.mkdir()

        # Test 3a: LLM validation error handling
        print("\nTest 3a: LLM validation error...")
        error = ValueError("Missing 'clusters' key in response")
        try:
            handle_stage7_error(error, str(bucket_path))
            print("✓ PASS: Error handler executed without crashing")
        except Exception as e:
            print(f"✗ FAIL: Error handler crashed - {e}")
            return False

        # Test 3b: API authentication error handling
        print("\nTest 3b: API authentication error...")
        error = RuntimeError("401 Unauthorized")
        try:
            handle_stage7_error(error, str(bucket_path))
            print("✓ PASS: Error handler executed without crashing")
        except Exception as e:
            print(f"✗ FAIL: Error handler crashed - {e}")
            return False

        # Test 3c: Cleanup partial outputs
        print("\nTest 3c: Cleanup partial outputs...")
        create_stage7_outputs(bucket_path, "18-33s")

        # Count files before cleanup
        llm_output_dir = bucket_path / "ml_analysis/llm"
        files_before = len(list(llm_output_dir.glob("*.json")))
        print(f"  Files before cleanup: {files_before}")

        cleanup_stage7_partial_outputs(str(bucket_path))

        # Count files after cleanup
        files_after = len(list(llm_output_dir.glob("*.json")))
        print(f"  Files after cleanup: {files_after}")

        if files_after == 0:
            print("✓ PASS: All partial outputs cleaned up")
        else:
            print(f"✗ FAIL: {files_after} files remain after cleanup")
            return False

    return True


def test_integration_workflow():
    """Test the complete Stage 7 integration workflow (without LLM API call)."""
    print("\n" + "="*80)
    print("TEST 4: Complete Integration Workflow")
    print("="*80)

    from rumiai_ml_batch import (
        validate_stage7_prerequisites,
        validate_stage7_outputs
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        bucket_path = Path(tmpdir) / "bucket_18-33s"
        bucket_path.mkdir()

        print("\nStep 1: Setup Stage 6 outputs...")
        setup_test_bucket(bucket_path, "18-33s")

        print("\nStep 2: Validate prerequisites...")
        try:
            validate_stage7_prerequisites(str(bucket_path), "18-33s")
            print("✓ Prerequisites validated")
        except Exception as e:
            print(f"✗ Prerequisite validation failed: {e}")
            return False

        print("\nStep 3: Simulate Stage 7 execution (create outputs)...")
        create_stage7_outputs(bucket_path, "18-33s")

        print("\nStep 4: Validate outputs...")
        try:
            validate_stage7_outputs(str(bucket_path), "18-33s")
            print("✓ Outputs validated")
        except Exception as e:
            print(f"✗ Output validation failed: {e}")
            return False

        print("\nStep 5: Test skip-if-complete logic...")
        complete_file = bucket_path / "ml_analysis/llm/complete_analysis.json"
        if complete_file.exists():
            print("✓ complete_analysis.json exists - skip logic would trigger")
        else:
            print("✗ complete_analysis.json missing")
            return False

    return True


def main():
    """Run all Stage 7 integration tests."""
    print("="*80)
    print("STAGE 7 INTEGRATION TEST SUITE")
    print("="*80)
    print("Testing Stage 7 helper functions and integration logic...")
    print("Note: This does NOT call the actual LLM API")
    print()

    tests = [
        ("Prerequisite Validation", test_validate_stage7_prerequisites),
        ("Output Validation", test_validate_stage7_outputs),
        ("Error Handling", test_error_handling),
        ("Complete Workflow", test_integration_workflow),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ TEST CRASHED: {test_name} - {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✅ All Stage 7 integration tests passed!")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
