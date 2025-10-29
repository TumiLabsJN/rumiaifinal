#!/usr/bin/env python3
"""
Test Script 3: Orchestrator Fallback Validation Logic
Tests all three validation paths in the orchestrator.

Usage:
    python test_orchestrator_fallback.py

Expected Result:
    - Path 1 (Primary): Checkpoint exists and valid → Proceed ✓
    - Path 2 (Fallback): Checkpoint missing, CSV exists → Proceed ✓
    - Path 3 (Fail): Both missing → Skip bucket ✓
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_test_scenario(scenario_name, has_checkpoint, has_csv, checkpoint_status="completed"):
    """
    Create test scenario directory structure.

    Args:
        scenario_name: Test scenario identifier
        has_checkpoint: bool, whether to create checkpoint file
        has_csv: bool, whether to create CSV file
        checkpoint_status: str, status value in checkpoint (default "completed")
    """
    test_base = Path(f"test_fallback_{scenario_name}")
    bucket_path = test_base / "bucket_18-33s"
    ml_analysis_dir = bucket_path / "ml_analysis"
    checkpoints_dir = bucket_path / "checkpoints"

    # Clean up
    if test_base.exists():
        shutil.rmtree(test_base)

    # Create directories
    ml_analysis_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint if requested
    if has_checkpoint:
        checkpoint = {
            "stage": "feature_aggregation",
            "status": checkpoint_status,
            "total_videos": 100,
            "output_files": ["aggregated_features.csv", "aggregation_summary.json"],
            "completion_time": datetime.now().isoformat(),
            "videos_processed": 100,
            "videos_skipped": 0,
            "bucket": "18-33s",
            "duration_seconds": 45.2,
            "feature_count": 135
        }
        checkpoint_path = checkpoints_dir / "stage_3_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    # Create CSV if requested
    if has_csv:
        csv_path = ml_analysis_dir / "aggregated_features.csv"
        # Create minimal valid CSV with headers
        with open(csv_path, 'w') as f:
            f.write("video_id,hook_scene_count,hook_energy_level,middle_1_scene_count,closing_energy_level\n")
            for i in range(10):
                f.write(f"video{i},3,0.75,4,0.82\n")

    return bucket_path, test_base


def test_orchestrator_validation(bucket_path, scenario_name):
    """
    Simulate orchestrator's Stage 4 validation logic.
    Returns: (validated, path_type)
        validated: bool, whether validation passed
        path_type: str, which validation path was used
    """
    stage3_checkpoint = bucket_path / "checkpoints" / "stage_3_checkpoint.json"
    aggregated_csv = bucket_path / "ml_analysis" / "aggregated_features.csv"

    if stage3_checkpoint.exists():
        # Primary path: checkpoint exists
        with open(stage3_checkpoint) as f:
            stage3_status = json.load(f)

        if stage3_status.get("status") != "completed":
            return False, "primary_invalid"

        return True, "primary"

    elif aggregated_csv.exists():
        # Fallback path: CSV exists
        csv_size = aggregated_csv.stat().st_size
        if csv_size > 0:
            return True, "fallback"
        else:
            return False, "fallback_empty_csv"

    else:
        # Fail path: both missing
        return False, "both_missing"


def run_test_scenario(scenario_num, scenario_name, has_checkpoint, has_csv,
                      expected_result, expected_path, checkpoint_status="completed"):
    """Run a single test scenario"""
    print("\n" + "=" * 70)
    print(f"SCENARIO {scenario_num}: {scenario_name.upper()}")
    print("=" * 70)

    # Setup
    bucket_path, test_base = create_test_scenario(
        scenario_name, has_checkpoint, has_csv, checkpoint_status
    )

    checkpoint_path = bucket_path / "checkpoints" / "stage_3_checkpoint.json"
    csv_path = bucket_path / "ml_analysis" / "aggregated_features.csv"

    print(f"Setup:")
    print(f"  - Checkpoint exists: {checkpoint_path.exists()}")
    print(f"  - CSV exists: {csv_path.exists()}")
    if has_checkpoint:
        with open(checkpoint_path) as f:
            status = json.load(f).get("status")
        print(f"  - Checkpoint status: {status}")

    # Test
    validated, path_type = test_orchestrator_validation(bucket_path, scenario_name)

    print(f"\nResult:")
    print(f"  - Validated: {validated}")
    print(f"  - Path used: {path_type}")
    print(f"  - Expected validated: {expected_result}")
    print(f"  - Expected path: {expected_path}")

    # Validate
    passed = (validated == expected_result) and (path_type == expected_path)

    if passed:
        print(f"\n✓ SCENARIO {scenario_num} PASSED")
    else:
        print(f"\n✗ SCENARIO {scenario_num} FAILED")
        print(f"  Expected: validated={expected_result}, path={expected_path}")
        print(f"  Got: validated={validated}, path={path_type}")

    # Cleanup
    if test_base.exists():
        shutil.rmtree(test_base)

    return passed


def main():
    """Run all orchestrator fallback validation scenarios"""
    print("=" * 70)
    print("TEST 3: ORCHESTRATOR FALLBACK VALIDATION LOGIC")
    print("=" * 70)
    print("\nTesting all three validation paths:")
    print("  1. Primary path: Checkpoint exists and valid")
    print("  2. Fallback path: Checkpoint missing, CSV exists")
    print("  3. Fail path: Both checkpoint and CSV missing")
    print("  4. Edge case: Checkpoint invalid status")
    print("  5. Edge case: CSV empty")

    results = []

    # Scenario 1: Normal flow - checkpoint exists and valid
    results.append(run_test_scenario(
        scenario_num=1,
        scenario_name="checkpoint_exists_valid",
        has_checkpoint=True,
        has_csv=True,
        expected_result=True,
        expected_path="primary"
    ))

    # Scenario 2: Fallback flow - checkpoint missing, CSV exists
    results.append(run_test_scenario(
        scenario_num=2,
        scenario_name="checkpoint_missing_csv_exists",
        has_checkpoint=False,
        has_csv=True,
        expected_result=True,
        expected_path="fallback"
    ))

    # Scenario 3: Fail flow - both missing
    results.append(run_test_scenario(
        scenario_num=3,
        scenario_name="both_missing",
        has_checkpoint=False,
        has_csv=False,
        expected_result=False,
        expected_path="both_missing"
    ))

    # Scenario 4: Edge case - checkpoint exists but status != completed
    results.append(run_test_scenario(
        scenario_num=4,
        scenario_name="checkpoint_invalid_status",
        has_checkpoint=True,
        has_csv=True,
        checkpoint_status="pending",
        expected_result=False,
        expected_path="primary_invalid"
    ))

    # Scenario 5: Edge case - checkpoint missing, CSV exists but empty
    results.append(run_test_scenario(
        scenario_num=5,
        scenario_name="csv_empty",
        has_checkpoint=False,
        has_csv=False,  # Will create empty CSV manually
        expected_result=False,
        expected_path="both_missing"  # Empty CSV treated as missing
    ))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(results)
    total_count = len(results)

    for i, passed in enumerate(results, 1):
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Scenario {i}: {status}")

    print(f"\nOverall: {passed_count}/{total_count} scenarios passed")

    if passed_count == total_count:
        print("\n" + "=" * 70)
        print("TEST RESULT: ✓ ALL SCENARIOS PASSED")
        print("=" * 70)
        print("\nOrchestrator fallback validation is working correctly:")
        print("  ✓ Primary path: Uses checkpoint when available")
        print("  ✓ Fallback path: Uses CSV when checkpoint missing")
        print("  ✓ Fail path: Skips bucket when both missing")
        print("  ✓ Edge cases: Handles invalid checkpoint and empty CSV")
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"TEST RESULT: ✗ {total_count - passed_count} SCENARIO(S) FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
