"""
Stage 2 Unit Tests: Test individual Stage 2 components

Tests checkpoint management, utilities, and validation logic.

Run with: python3 ml_pipeline/tests/run_stage2_unit_tests.py
"""

import sys
import os
import json
import shutil
import tempfile

# Add parent directory to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from ml_pipeline.stage2_processing import checkpoint, utils, validation
from ml_pipeline.stage2_processing.exceptions import CheckpointCorruptionError, ValidationError


class TestRunner:
    """Simple test runner without pytest dependency"""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failed_tests = []

    def run_test(self, test_name, test_func):
        """Run a single test function"""
        self.tests_run += 1
        try:
            test_func()
            self.tests_passed += 1
            print(f"✓ {test_name}")
        except AssertionError as e:
            self.tests_failed += 1
            self.failed_tests.append((test_name, str(e)))
            print(f"✗ {test_name}: {e}")
        except Exception as e:
            self.tests_failed += 1
            self.failed_tests.append((test_name, f"Exception: {e}"))
            print(f"✗ {test_name}: Exception: {e}")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_failed}")

        if self.failed_tests:
            print("\nFailed tests:")
            for test_name, error in self.failed_tests:
                print(f"  - {test_name}: {error}")

        print("="*70)

        return self.tests_failed == 0


# ============================================================================
# Checkpoint Module Tests
# ============================================================================

def test_initialize_checkpoint_new():
    """Test checkpoint initialization when no checkpoint exists"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set DATA_ROOT to temp directory to avoid permission issues
        old_data_root = os.environ.get('DATA_ROOT')
        os.environ['DATA_ROOT'] = tmpdir

        try:
            config = {
                "client_id": "test_client",
                "analysis_type": "hashtag",
                "target": "fitness",
                "analysis_mode": "top",
                "selection_strategy": "contrastive",
                "download_dir": f"{tmpdir}/videos",
                "timeout": 300
            }

            video_list = [
                {"id": "vid1", "duration": 25},
                {"id": "vid2", "duration": 30}
            ]

            checkpoint_data, remaining_videos = checkpoint.initialize_checkpoint(
                bucket_name="18-33s",
                video_list=video_list,
                config=config
            )

            # Check checkpoint structure
            assert "bucket" in checkpoint_data
            assert "config" in checkpoint_data
            assert "completed_video_ids" in checkpoint_data
            assert "failed_video_ids" in checkpoint_data
            assert "last_checkpoint" in checkpoint_data

            # Check values
            assert checkpoint_data["bucket"] == "18-33s"
            assert checkpoint_data["completed_video_ids"] == []
            assert checkpoint_data["failed_video_ids"] == []
            assert len(remaining_videos) == 2

        finally:
            # Restore original DATA_ROOT
            if old_data_root is not None:
                os.environ['DATA_ROOT'] = old_data_root
            elif 'DATA_ROOT' in os.environ:
                del os.environ['DATA_ROOT']


def test_initialize_checkpoint_resume():
    """Test checkpoint auto-resume when checkpoint exists"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set DATA_ROOT to temp directory
        old_data_root = os.environ.get('DATA_ROOT')
        os.environ['DATA_ROOT'] = tmpdir

        try:
            bucket_name = "18-33s"

            config_dict = {
                "client_id": "test_client",
                "analysis_type": "hashtag",
                "target": "fitness",
                "analysis_mode": "top",
                "selection_strategy": "contrastive",
                "timeout": 300
            }

            # Construct checkpoint path using get_bucket_path logic
            checkpoint_dir = f"{tmpdir}/clients/test_client/hashtags/fitness/top_contrastive/buckets/bucket_{bucket_name}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            existing_checkpoint = {
                "bucket": bucket_name,
                "config": config_dict,
                "completed_video_ids": ["vid1"],
                "failed_video_ids": [],
                "total_videos": 2,
                "completed": 1,
                "failed": 0,
                "remaining": 1,
                "last_checkpoint": "2025-01-01T00:00:00"
            }

            checkpoint_path = f"{checkpoint_dir}/stage_2_checkpoint.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(existing_checkpoint, f)

            # Initialize with same config
            config = {
                "client_id": "test_client",
                "analysis_type": "hashtag",
                "target": "fitness",
                "analysis_mode": "top",
                "selection_strategy": "contrastive",
                "download_dir": f"{tmpdir}/videos",
                "timeout": 300
            }

            video_list = [
                {"id": "vid1", "duration": 25},
                {"id": "vid2", "duration": 30}
            ]

            checkpoint_data, remaining_videos = checkpoint.initialize_checkpoint(
                bucket_name=bucket_name,
                video_list=video_list,
                config=config
            )

            # Should resume with vid1 already completed
            assert "vid1" in checkpoint_data["completed_video_ids"]
            assert len(remaining_videos) == 1
            assert remaining_videos[0]["id"] == "vid2"

        finally:
            # Restore original DATA_ROOT
            if old_data_root is not None:
                os.environ['DATA_ROOT'] = old_data_root
            elif 'DATA_ROOT' in os.environ:
                del os.environ['DATA_ROOT']


def test_save_checkpoint_creates_backup():
    """Test that save_checkpoint_with_backup creates backup file"""

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = f"{tmpdir}/stage_2_checkpoint.json"
        backup_path = f"{tmpdir}/stage_2_checkpoint.backup.json"

        # Create initial checkpoint
        checkpoint_data = {
            "bucket_name": "18-33s",
            "completed_video_ids": ["vid1"]
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        # Update checkpoint
        checkpoint_data["completed_video_ids"].append("vid2")

        checkpoint.save_checkpoint_with_backup(checkpoint_path, checkpoint_data)

        # Verify backup was created
        assert os.path.exists(backup_path)

        # Verify backup contains old data
        with open(backup_path) as f:
            backup_data = json.load(f)

        assert backup_data["completed_video_ids"] == ["vid1"]

        # Verify main checkpoint has new data
        with open(checkpoint_path) as f:
            new_data = json.load(f)

        assert new_data["completed_video_ids"] == ["vid1", "vid2"]


def test_load_checkpoint_recovery_from_backup():
    """Test checkpoint recovery from backup when main is corrupted"""

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = f"{tmpdir}/stage_2_checkpoint.json"
        backup_path = f"{tmpdir}/stage_2_checkpoint.backup.json"

        # Create valid backup
        backup_data = {
            "bucket_name": "18-33s",
            "completed_video_ids": ["vid1"]
        }

        with open(backup_path, 'w') as f:
            json.dump(backup_data, f)

        # Create corrupted main checkpoint
        with open(checkpoint_path, 'w') as f:
            f.write("{ invalid json }")

        # Should recover from backup
        recovered_data = checkpoint.load_checkpoint_with_recovery(checkpoint_path)

        assert recovered_data["bucket_name"] == "18-33s"
        assert recovered_data["completed_video_ids"] == ["vid1"]


def test_load_checkpoint_both_corrupted():
    """Test checkpoint fails when both main and backup are corrupted"""

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = f"{tmpdir}/stage_2_checkpoint.json"
        backup_path = f"{tmpdir}/stage_2_checkpoint.backup.json"

        # Create corrupted main checkpoint
        with open(checkpoint_path, 'w') as f:
            f.write("{ invalid json }")

        # Create corrupted backup
        with open(backup_path, 'w') as f:
            f.write("{ also invalid }")

        # Should raise CheckpointCorruptionError
        try:
            checkpoint.load_checkpoint_with_recovery(checkpoint_path)
            assert False, "Should have raised CheckpointCorruptionError"
        except CheckpointCorruptionError:
            pass  # Expected


# ============================================================================
# Utils Module Tests
# ============================================================================

def test_save_and_load_json():
    """Test atomic JSON save and load"""

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test.json"

        test_data = {
            "key1": "value1",
            "key2": 123,
            "key3": [1, 2, 3]
        }

        # Save
        utils.save_json(filepath, test_data)

        # Load
        loaded_data = utils.load_json(filepath)

        assert loaded_data == test_data


def test_get_bucket_path():
    """Test bucket path construction"""

    config = {
        "client_id": "test_client",
        "analysis_type": "hashtag",
        "target": "fitness",
        "analysis_mode": "top",
        "selection_strategy": "contrastive"
    }

    bucket_path = utils.get_bucket_path(config, "18-33s")

    # Should construct path from config
    assert "test_client" in bucket_path
    assert "fitness" in bucket_path
    assert "bucket_18-33s" in bucket_path


# ============================================================================
# Validation Module Tests
# ============================================================================

def test_validate_temporal_windows_valid():
    """Test validation passes for valid temporal_windows"""

    # Create dict with 60+ features for hook and closing
    features = {f"feature_{i}": i for i in range(65)}

    valid_data = {
        "video_id": "test_123",
        "duration": 25.5,
        "temporal_windows": {
            "hook": features.copy(),
            "middle_segments": [features.copy()],
            "closing": features.copy()
        },
        "metadata": {
            "source": "test",
            "version": "1.0"
        },
        "processing_timestamp": "2025-01-01T00:00:00"
    }

    # Should not raise exception
    validation.validate_temporal_windows_schema(valid_data)


def test_validate_temporal_windows_missing_metadata():
    """Test validation fails when metadata missing"""

    features = {f"feature_{i}": i for i in range(65)}

    invalid_data = {
        "video_id": "test_123",
        "duration": 25.5,
        "temporal_windows": {
            "hook": features.copy(),
            "middle_segments": [features.copy()],
            "closing": features.copy()
        },
        "processing_timestamp": "2025-01-01T00:00:00"
        # metadata missing
    }

    try:
        validation.validate_temporal_windows_schema(invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "metadata" in str(e)


def test_validate_temporal_windows_missing_timestamp():
    """Test validation fails when processing_timestamp missing"""

    features = {f"feature_{i}": i for i in range(65)}

    invalid_data = {
        "video_id": "test_123",
        "duration": 25.5,
        "temporal_windows": {
            "hook": features.copy(),
            "middle_segments": [features.copy()],
            "closing": features.copy()
        },
        "metadata": {"source": "test"}
        # processing_timestamp missing
    }

    try:
        validation.validate_temporal_windows_schema(invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "processing_timestamp" in str(e)


def test_validate_temporal_windows_missing_section():
    """Test validation fails when temporal_windows section missing"""

    features = {f"feature_{i}": i for i in range(65)}

    invalid_data = {
        "video_id": "test_123",
        "duration": 25.5,
        "temporal_windows": {
            "hook": features.copy(),
            "middle_segments": [features.copy()]
            # "closing" missing
        },
        "metadata": {"source": "test"},
        "processing_timestamp": "2025-01-01T00:00:00"
    }

    try:
        validation.validate_temporal_windows_schema(invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "closing" in str(e) or "temporal_windows" in str(e)


def test_validate_temporal_windows_invalid_type():
    """Test validation fails when temporal_windows is not a dict"""

    invalid_data = {
        "video_id": "test_123",
        "duration": 25.5,
        "temporal_windows": [],  # Should be dict, not list
        "metadata": {"source": "test"},
        "processing_timestamp": "2025-01-01T00:00:00"
    }

    try:
        validation.validate_temporal_windows_schema(invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "dict" in str(e) or "temporal_windows" in str(e)


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all Stage 2 unit tests"""

    print("="*70)
    print("Running Stage 2 Unit Tests")
    print("="*70)
    print()

    runner = TestRunner()

    # Checkpoint tests
    print("Checkpoint Module Tests:")
    runner.run_test("test_initialize_checkpoint_new", test_initialize_checkpoint_new)
    runner.run_test("test_initialize_checkpoint_resume", test_initialize_checkpoint_resume)
    runner.run_test("test_save_checkpoint_creates_backup", test_save_checkpoint_creates_backup)
    runner.run_test("test_load_checkpoint_recovery_from_backup", test_load_checkpoint_recovery_from_backup)
    runner.run_test("test_load_checkpoint_both_corrupted", test_load_checkpoint_both_corrupted)

    print()

    # Utils tests
    print("Utils Module Tests:")
    runner.run_test("test_save_and_load_json", test_save_and_load_json)
    runner.run_test("test_get_bucket_path", test_get_bucket_path)

    print()

    # Validation tests
    print("Validation Module Tests:")
    runner.run_test("test_validate_temporal_windows_valid", test_validate_temporal_windows_valid)
    runner.run_test("test_validate_temporal_windows_missing_metadata", test_validate_temporal_windows_missing_metadata)
    runner.run_test("test_validate_temporal_windows_missing_timestamp", test_validate_temporal_windows_missing_timestamp)
    runner.run_test("test_validate_temporal_windows_missing_section", test_validate_temporal_windows_missing_section)
    runner.run_test("test_validate_temporal_windows_invalid_type", test_validate_temporal_windows_invalid_type)

    print()

    # Print summary
    success = runner.print_summary()

    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
