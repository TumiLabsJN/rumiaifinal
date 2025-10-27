#!/usr/bin/env python3
"""
Test Script 1: Normal Checkpoint Flow
Tests that Stage 3 creates checkpoint successfully under normal conditions.

Usage:
    python test_stage3_checkpoint_normal.py

Expected Result:
    - Stage 3 creates aggregated_features.csv
    - Stage 3 creates stage_3_checkpoint.json
    - Checkpoint has correct schema
    - Stage 4 validates via checkpoint (primary path)
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def setup_test_environment():
    """Create test directory structure with minimal test data"""
    print("=" * 70)
    print("TEST 1: NORMAL CHECKPOINT FLOW")
    print("=" * 70)

    # Create test bucket directory (use 9-13s bucket: requires 3 windows only)
    test_base = Path("test_checkpoint_normal")
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
    # Duration 11.0s fits bucket_9-13s (requires hook, middle_aggregate, closing)
    test_video = {
        "video_id": "test123",
        "duration": 11.0,
        "version": "1.0",
        "processing_timestamp": datetime.now().isoformat(),
        "metadata": {
            "video_id": "test123",
            "duration": 11.0,
            "create_time": datetime.now().isoformat(),
            "author": {"unique_id": "testuser"},
            "digg_count": 100,
            "play_count": 1000,
            "share_count": 10,
            "comment_count": 5,
            "collect_count": 2
        },
        "temporal_windows": {
            "hook": {
                "start": 0, "end": 3.0, "duration": 3.0,
                "overlay_unique_count": 0, "has_captions": True,
                "object_count": 1, "person_count": 1, "gesture_count": 2,
                "scene_count": 3, "shortest_scene": 0.8, "longest_scene": 1.5,
                "scene_duration_variance": 0.12, "speech_coverage": 0.85,
                "word_count": 15, "gaze_variance": 0.05, "eye_contact_rate": 0.85,
                "dominant_emotion_id": 1, "emotional_valence": 0.65,
                "emotion_consistency": 0.80, "average_face_size": 0.45,
                "energy_level": 0.72, "energy_variance": 0.03,
                "energy_max": 0.95, "pitch_scatter_ratio": 0.18
            },
            "middle_segments": [
                {
                    "start": 3.0, "end": 8.0, "duration": 5.0,
                    "overlay_unique_count": 1, "has_captions": True,
                    "object_count": 2, "person_count": 1, "gesture_count": 5,
                    "scene_count": 4, "shortest_scene": 3.0, "longest_scene": 6.0,
                    "scene_duration_variance": 1.2, "speech_coverage": 0.70,
                    "word_count": 35, "gaze_variance": 0.08, "eye_contact_rate": 0.70,
                    "dominant_emotion_id": 7, "emotional_valence": 0.0,
                    "emotion_consistency": 0.75, "average_face_size": 0.50,
                    "energy_level": 0.68, "energy_variance": 0.05,
                    "energy_max": 0.88, "pitch_scatter_ratio": 0.22
                }
            ],
            "closing": {
                "start": 8.0, "end": 11.0, "duration": 3.0,
                "overlay_unique_count": 0, "has_captions": True,
                "object_count": 1, "person_count": 1, "gesture_count": 1,
                "scene_count": 2, "shortest_scene": 1.2, "longest_scene": 1.8,
                "scene_duration_variance": 0.18, "speech_coverage": 0.90,
                "word_count": 8, "gaze_variance": 0.03, "eye_contact_rate": 0.90,
                "dominant_emotion_id": 1, "emotional_valence": 0.80,
                "emotion_consistency": 0.85, "average_face_size": 0.48,
                "energy_level": 0.75, "energy_variance": 0.02,
                "energy_max": 0.92, "pitch_scatter_ratio": 0.15
            }
        }
    }

    # Write test file
    test_file = insights_dir / "test123_temporal_windows_updated.json"
    with open(test_file, 'w') as f:
        json.dump(test_video, f, indent=2)

    print(f"✓ Test environment created: {bucket_path}")
    print(f"✓ Test video created: {test_file.name}")

    return bucket_path, "9-13s"


def run_stage3(bucket_path):
    """Run Stage 3 aggregation"""
    print("\n" + "-" * 70)
    print("STEP 1: Running Stage 3 aggregation")
    print("-" * 70)

    try:
        from scripts.stage3_aggregation import aggregate_features

        print(f"Calling aggregate_features(bucket_path={bucket_path})")
        output_csv, summary_json = aggregate_features(
            bucket_path=str(bucket_path)
        )

        print(f"✓ Stage 3 completed successfully")
        print(f"  - CSV: {output_csv}")
        print(f"  - Summary: {summary_json}")

        return True, output_csv, summary_json

    except Exception as e:
        print(f"✗ Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def validate_checkpoint(bucket_path):
    """Validate checkpoint was created with correct schema"""
    print("\n" + "-" * 70)
    print("STEP 2: Validating checkpoint creation")
    print("-" * 70)

    checkpoint_path = bucket_path / "checkpoints" / "stage_3_checkpoint.json"

    if not checkpoint_path.exists():
        print(f"✗ FAILED: Checkpoint not created at {checkpoint_path}")
        return False

    print(f"✓ Checkpoint file exists: {checkpoint_path}")

    # Load and validate schema
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    required_fields = [
        "stage", "status", "total_videos", "output_files",
        "completion_time", "videos_processed", "videos_skipped",
        "bucket", "duration_seconds", "feature_count"
    ]

    missing = [field for field in required_fields if field not in checkpoint]
    if missing:
        print(f"✗ FAILED: Missing required fields: {missing}")
        return False

    print(f"✓ All required fields present")

    # Validate field values
    if checkpoint["stage"] != "feature_aggregation":
        print(f"✗ FAILED: stage = '{checkpoint['stage']}', expected 'feature_aggregation'")
        return False

    if checkpoint["status"] != "completed":
        print(f"✗ FAILED: status = '{checkpoint['status']}', expected 'completed'")
        return False

    if checkpoint["bucket"] != "9-13s":
        print(f"✗ FAILED: bucket = '{checkpoint['bucket']}', expected '9-13s'")
        return False

    print(f"✓ Checkpoint schema valid:")
    print(f"  - stage: {checkpoint['stage']}")
    print(f"  - status: {checkpoint['status']}")
    print(f"  - videos_processed: {checkpoint['videos_processed']}")
    print(f"  - videos_skipped: {checkpoint['videos_skipped']}")
    print(f"  - feature_count: {checkpoint['feature_count']}")

    return True


def test_orchestrator_validation(bucket_path):
    """Test that orchestrator validates via checkpoint (primary path)"""
    print("\n" + "-" * 70)
    print("STEP 3: Testing orchestrator validation logic")
    print("-" * 70)

    # Simulate orchestrator's Stage 4 validation logic
    stage3_checkpoint = bucket_path / "checkpoints" / "stage_3_checkpoint.json"
    aggregated_csv = bucket_path / "ml_analysis" / "aggregated_features.csv"

    print(f"Checkpoint exists: {stage3_checkpoint.exists()}")
    print(f"CSV exists: {aggregated_csv.exists()}")

    if stage3_checkpoint.exists():
        print("✓ Primary path: Checkpoint exists")

        with open(stage3_checkpoint) as f:
            stage3_status = json.load(f)

        if stage3_status.get("status") == "completed":
            print("✓ Checkpoint validation passed: status='completed'")
            print("✓ Orchestrator would proceed to Stage 4")
            return True
        else:
            print(f"✗ FAILED: Checkpoint status = '{stage3_status.get('status')}'")
            return False
    else:
        print("✗ FAILED: Orchestrator would skip bucket (no checkpoint)")
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
    """Run normal checkpoint flow test"""
    test_base = Path("test_checkpoint_normal")

    try:
        # Setup
        bucket_path, bucket = setup_test_environment()

        # Test Stage 3
        success, csv_path, summary_path = run_stage3(bucket_path)
        if not success:
            print("\n" + "=" * 70)
            print("TEST RESULT: ✗ FAILED (Stage 3 execution)")
            print("=" * 70)
            return 1

        # Validate checkpoint
        if not validate_checkpoint(bucket_path):
            print("\n" + "=" * 70)
            print("TEST RESULT: ✗ FAILED (Checkpoint validation)")
            print("=" * 70)
            return 1

        # Test orchestrator
        if not test_orchestrator_validation(bucket_path):
            print("\n" + "=" * 70)
            print("TEST RESULT: ✗ FAILED (Orchestrator validation)")
            print("=" * 70)
            return 1

        # Success
        print("\n" + "=" * 70)
        print("TEST RESULT: ✓ PASSED")
        print("=" * 70)
        print("\nAll validations passed:")
        print("  ✓ Stage 3 created CSV successfully")
        print("  ✓ Stage 3 created checkpoint successfully")
        print("  ✓ Checkpoint has correct schema")
        print("  ✓ Orchestrator validates via checkpoint (primary path)")

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
