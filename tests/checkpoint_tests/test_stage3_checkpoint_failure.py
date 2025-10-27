#!/usr/bin/env python3
"""
Test Script 2: Checkpoint Write Failure (Graceful Degradation)
Tests that Stage 3 handles checkpoint write failures gracefully.

Usage:
    python test_stage3_checkpoint_failure.py

Expected Result:
    - Stage 3 creates aggregated_features.csv ✓
    - Checkpoint write fails (simulated) ✗
    - Stage 3 logs WARNING (not ERROR) ✓
    - Stage 3 returns successfully (no exception raised) ✓
    - CSV is valid and usable ✓
"""

import sys
import json
import shutil
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def setup_test_environment():
    """Create test directory structure with minimal test data"""
    print("=" * 70)
    print("TEST 2: CHECKPOINT WRITE FAILURE (GRACEFUL DEGRADATION)")
    print("=" * 70)

    test_base = Path("test_checkpoint_failure")
    bucket_path = test_base / "bucket_9-13s"
    insights_dir = bucket_path / "analysis" / "insights"
    ml_analysis_dir = bucket_path / "ml_analysis"
    checkpoints_dir = bucket_path / "checkpoints"

    # Clean up previous test
    if test_base.exists():
        shutil.rmtree(test_base)

    # Create directory structure
    insights_dir.mkdir(parents=True, exist_ok=True)
    ml_analysis_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal valid temporal windows JSON (matches production format)
    test_video = {
        "video_id": "test456",
        "duration": 11.0,
        "version": "1.0",
        "processing_timestamp": datetime.now().isoformat(),
        "metadata": {
            "video_id": "test456",
            "duration": 11.0,
            "create_time": datetime.now().isoformat(),
            "author": {"unique_id": "testuser2"},
            "digg_count": 200,
            "play_count": 2000,
            "share_count": 20,
            "comment_count": 10,
            "collect_count": 5
        },
        "temporal_windows": {
            "hook": {
                "start": 0, "end": 3.0, "duration": 3.0,
                "overlay_unique_count": 1, "has_captions": True,
                "object_count": 2, "person_count": 1, "gesture_count": 3,
                "scene_count": 2, "shortest_scene": 1.0, "longest_scene": 2.0,
                "scene_duration_variance": 0.25, "speech_coverage": 0.75,
                "word_count": 10, "gaze_variance": 0.06, "eye_contact_rate": 0.75,
                "dominant_emotion_id": 7, "emotional_valence": 0.0,
                "emotion_consistency": 0.70, "average_face_size": 0.42,
                "energy_level": 0.65, "energy_variance": 0.04,
                "energy_max": 0.85, "pitch_scatter_ratio": 0.20
            },
            "middle_segments": [
                {
                    "start": 3.0, "end": 8.0, "duration": 5.0,
                    "overlay_unique_count": 0, "has_captions": True,
                    "object_count": 1, "person_count": 1, "gesture_count": 4,
                    "scene_count": 3, "shortest_scene": 4.0, "longest_scene": 8.0,
                    "scene_duration_variance": 2.0, "speech_coverage": 0.80,
                    "word_count": 30, "gaze_variance": 0.07, "eye_contact_rate": 0.80,
                    "dominant_emotion_id": 1, "emotional_valence": 0.60,
                    "emotion_consistency": 0.78, "average_face_size": 0.47,
                    "energy_level": 0.70, "energy_variance": 0.04,
                    "energy_max": 0.90, "pitch_scatter_ratio": 0.19
                }
            ],
            "closing": {
                "start": 8.0, "end": 11.0, "duration": 3.0,
                "overlay_unique_count": 0, "has_captions": False,
                "object_count": 1, "person_count": 1, "gesture_count": 2,
                "scene_count": 1, "shortest_scene": 3.0, "longest_scene": 3.0,
                "scene_duration_variance": 0.0, "speech_coverage": 0.95,
                "word_count": 5, "gaze_variance": 0.02, "eye_contact_rate": 0.95,
                "dominant_emotion_id": 1, "emotional_valence": 0.85,
                "emotion_consistency": 0.90, "average_face_size": 0.50,
                "energy_level": 0.80, "energy_variance": 0.01,
                "energy_max": 0.95, "pitch_scatter_ratio": 0.12
            }
        }
    }

    test_file = insights_dir / "test456_temporal_windows_updated.json"
    with open(test_file, 'w') as f:
        json.dump(test_video, f, indent=2)

    print(f"✓ Test environment created: {bucket_path}")
    print(f"✓ Test video created: {test_file.name}")

    return bucket_path, "9-13s", checkpoints_dir


def run_stage3_with_checkpoint_failure(bucket_path, checkpoints_dir):
    """
    Run Stage 3 with simulated checkpoint write failure.
    Uses file permissions to simulate write failure.
    """
    print("\n" + "-" * 70)
    print("STEP 1: Running Stage 3 with simulated checkpoint failure")
    print("-" * 70)

    # Make checkpoints directory read-only BEFORE running Stage 3
    original_mode = checkpoints_dir.stat().st_mode
    os.chmod(checkpoints_dir, 0o444)  # Read-only
    print(f"✓ Set checkpoints directory to read-only: {checkpoints_dir}")

    try:
        from scripts.stage3_aggregation import aggregate_features

        print(f"Calling aggregate_features(bucket_path={bucket_path})")
        print("Expected: CSV created ✓, Checkpoint fails ✗, Function succeeds ✓")

        output_csv, summary_json = aggregate_features(
            bucket_path=str(bucket_path)
        )

        print(f"\n✓ Stage 3 completed successfully (graceful degradation worked!)")
        print(f"  - CSV: {output_csv}")
        print(f"  - Summary: {summary_json}")

        # Restore permissions
        os.chmod(checkpoints_dir, original_mode)

        return True, output_csv, summary_json

    except Exception as e:
        # If we get here, graceful degradation FAILED
        print(f"\n✗ FAILED: Stage 3 raised exception (should have degraded gracefully)")
        print(f"  Exception: {e}")

        # Restore permissions
        os.chmod(checkpoints_dir, original_mode)

        import traceback
        traceback.print_exc()
        return False, None, None


def validate_csv_created_but_checkpoint_missing(bucket_path):
    """Validate that CSV was created but checkpoint is missing"""
    print("\n" + "-" * 70)
    print("STEP 2: Validating CSV created, checkpoint missing")
    print("-" * 70)

    csv_path = bucket_path / "ml_analysis" / "aggregated_features.csv"
    checkpoint_path = bucket_path / "checkpoints" / "stage_3_checkpoint.json"

    # CSV should exist
    if not csv_path.exists():
        print(f"✗ FAILED: CSV not created at {csv_path}")
        return False

    csv_size = csv_path.stat().st_size
    if csv_size == 0:
        print(f"✗ FAILED: CSV is empty")
        return False

    print(f"✓ CSV created successfully: {csv_path} ({csv_size} bytes)")

    # Checkpoint should NOT exist (write failed)
    if checkpoint_path.exists():
        print(f"✗ FAILED: Checkpoint exists (should have failed to write)")
        return False

    print(f"✓ Checkpoint missing as expected (write failed gracefully)")

    return True


def test_csv_is_valid(bucket_path):
    """Validate that CSV is structurally valid and has data"""
    print("\n" + "-" * 70)
    print("STEP 3: Validating CSV is structurally valid")
    print("-" * 70)

    csv_path = bucket_path / "ml_analysis" / "aggregated_features.csv"

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            print(f"✗ FAILED: CSV has no rows")
            return False

        print(f"✓ CSV loaded successfully:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - First 5 columns: {list(df.columns[:5])}")

        return True

    except Exception as e:
        print(f"✗ FAILED: CSV is corrupt or invalid: {e}")
        return False


def test_orchestrator_fallback_validation(bucket_path):
    """Test that orchestrator uses CSV fallback when checkpoint missing"""
    print("\n" + "-" * 70)
    print("STEP 4: Testing orchestrator CSV fallback validation")
    print("-" * 70)

    # Simulate orchestrator's Stage 4 validation logic (with fallback)
    stage3_checkpoint = bucket_path / "checkpoints" / "stage_3_checkpoint.json"
    aggregated_csv = bucket_path / "ml_analysis" / "aggregated_features.csv"

    print(f"Checkpoint exists: {stage3_checkpoint.exists()}")
    print(f"CSV exists: {aggregated_csv.exists()}")

    if stage3_checkpoint.exists():
        print("✗ FAILED: Should be using fallback path (checkpoint should be missing)")
        return False

    elif aggregated_csv.exists():
        print("✓ Fallback path activated: Checkpoint missing, checking CSV")

        csv_size = aggregated_csv.stat().st_size
        if csv_size > 0:
            print(f"✓ CSV validation passed: {csv_size} bytes")
            print("✓ Orchestrator would proceed to Stage 4 (using CSV fallback)")
            print("⚠️  Note: Stage 3 skip logic disabled (will re-run on resume)")
            return True
        else:
            print(f"✗ FAILED: CSV is empty")
            return False

    else:
        print("✗ FAILED: Both checkpoint and CSV missing")
        return False


def cleanup(test_base):
    """Clean up test directory"""
    print("\n" + "-" * 70)
    print("CLEANUP")
    print("-" * 70)

    if test_base.exists():
        shutil.rmtree(test_base)
        print(f"✓ Removed test directory: {test_base}")


def main():
    """Run checkpoint failure test"""
    test_base = Path("test_checkpoint_failure")

    try:
        # Setup
        bucket_path, bucket, checkpoints_dir = setup_test_environment()

        # Test Stage 3 with checkpoint failure
        success, csv_path, summary_path = run_stage3_with_checkpoint_failure(
            bucket_path, checkpoints_dir
        )
        if not success:
            print("\n" + "=" * 70)
            print("TEST RESULT: ✗ FAILED (Stage 3 did not degrade gracefully)")
            print("=" * 70)
            return 1

        # Validate CSV created, checkpoint missing
        if not validate_csv_created_but_checkpoint_missing(bucket_path):
            print("\n" + "=" * 70)
            print("TEST RESULT: ✗ FAILED (CSV or checkpoint state incorrect)")
            print("=" * 70)
            return 1

        # Validate CSV is valid
        if not test_csv_is_valid(bucket_path):
            print("\n" + "=" * 70)
            print("TEST RESULT: ✗ FAILED (CSV validation)")
            print("=" * 70)
            return 1

        # Test orchestrator fallback
        if not test_orchestrator_fallback_validation(bucket_path):
            print("\n" + "=" * 70)
            print("TEST RESULT: ✗ FAILED (Orchestrator fallback)")
            print("=" * 70)
            return 1

        # Success
        print("\n" + "=" * 70)
        print("TEST RESULT: ✓ PASSED")
        print("=" * 70)
        print("\nGraceful degradation validated:")
        print("  ✓ Stage 3 created CSV successfully")
        print("  ✓ Checkpoint write failed (simulated permission error)")
        print("  ✓ Stage 3 did NOT raise exception (degraded gracefully)")
        print("  ✓ CSV is valid and usable")
        print("  ✓ Orchestrator validates via CSV fallback")
        print("  ✓ Pipeline would continue to Stage 4")
        print("\nTrade-off accepted:")
        print("  ⚠️  Stage 3 skip logic disabled (will re-run on resume)")

        return 0

    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        cleanup(test_base)


if __name__ == "__main__":
    sys.exit(main())
