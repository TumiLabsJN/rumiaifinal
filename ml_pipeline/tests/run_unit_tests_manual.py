"""
Manual test runner for Stage 2.5 unit tests (no pytest required)

Run with: python3 ml_pipeline/tests/run_unit_tests_manual.py
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from ml_pipeline.stage2_5_organize.file_organizer import (
    load_winning_buckets,
    validate_checkpoint,
    detect_duplicates_across_buckets
)


class TestRunner:
    """Simple test runner"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, test_name, test_func):
        """Run a single test"""
        try:
            test_func()
            print(f"✓ {test_name}")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
            self.failed += 1
            self.errors.append((test_name, str(e)))
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
            self.failed += 1
            self.errors.append((test_name, f"ERROR - {str(e)}"))

    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Summary: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


# Test functions
def test_load_valid_winner_analysis():
    """Test loading valid winner_analysis.json"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        analysis_base = Path(tmp_dir) / "analysis"
        analysis_base.mkdir()

        winner_analysis = {
            "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
            "top_100_distribution": {"18-33s": 45, "33-60s": 30, "13-18s": 20},
            "winner_coverage": 95.0
        }

        with open(analysis_base / "winner_analysis.json", 'w') as f:
            json.dump(winner_analysis, f)

        buckets = load_winning_buckets(str(analysis_base))

        assert buckets == ["18-33s", "33-60s", "13-18s"], f"Expected ['18-33s', '33-60s', '13-18s'], got {buckets}"
        assert len(buckets) == 3, f"Expected 3 buckets, got {len(buckets)}"


def test_load_missing_file():
    """Test FileNotFoundError when winner_analysis.json missing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        analysis_base = Path(tmp_dir) / "analysis"
        # Note: directory doesn't exist

        try:
            load_winning_buckets(str(analysis_base))
            raise AssertionError("Expected FileNotFoundError to be raised")
        except FileNotFoundError as e:
            assert "winner_analysis.json not found" in str(e)


def test_load_missing_top_3_buckets_field():
    """Test ValueError when top_3_buckets field missing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        analysis_base = Path(tmp_dir) / "analysis"
        analysis_base.mkdir()

        winner_analysis = {"top_100_distribution": {"18-33s": 45}}

        with open(analysis_base / "winner_analysis.json", 'w') as f:
            json.dump(winner_analysis, f)

        try:
            load_winning_buckets(str(analysis_base))
            raise AssertionError("Expected ValueError to be raised")
        except ValueError as e:
            assert "missing 'top_3_buckets' field" in str(e)


def test_load_empty_buckets_list():
    """Test ValueError when top_3_buckets is empty"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        analysis_base = Path(tmp_dir) / "analysis"
        analysis_base.mkdir()

        winner_analysis = {"top_3_buckets": []}

        with open(analysis_base / "winner_analysis.json", 'w') as f:
            json.dump(winner_analysis, f)

        try:
            load_winning_buckets(str(analysis_base))
            raise AssertionError("Expected ValueError to be raised")
        except ValueError as e:
            assert "empty" in str(e).lower()


def test_validate_completed_checkpoint():
    """Test validation passes for completed checkpoint"""
    checkpoint = {
        "stage": "video_processing",
        "bucket": "18-33s",
        "completed_video_ids": ["123", "456", "789"],
        "status": "completed",
        "total_videos": 10
    }

    video_ids = validate_checkpoint(checkpoint, "18-33s")

    assert video_ids == ["123", "456", "789"]
    assert len(video_ids) == 3


def test_validate_missing_required_fields():
    """Test ValueError when required fields missing"""
    checkpoint = {"stage": "video_processing"}

    try:
        validate_checkpoint(checkpoint, "18-33s")
        raise AssertionError("Expected ValueError to be raised")
    except ValueError as e:
        assert "invalid schema" in str(e)
        assert "missing" in str(e)


def test_validate_zero_completions():
    """Test returns empty list for zero completed videos"""
    checkpoint = {
        "stage": "video_processing",
        "bucket": "18-33s",
        "completed_video_ids": [],
        "status": "completed",
        "total_videos": 10
    }

    video_ids = validate_checkpoint(checkpoint, "18-33s")

    assert video_ids == []


def test_no_duplicates():
    """Test validation passes when no duplicates"""
    files = [
        {"video_id": "123", "bucket": "18-33s"},
        {"video_id": "456", "bucket": "33-60s"},
        {"video_id": "789", "bucket": "13-18s"}
    ]

    # Should not raise
    detect_duplicates_across_buckets(files)


def test_duplicate_detected():
    """Test ValueError raised when duplicate video_id detected"""
    files = [
        {"video_id": "123", "bucket": "18-33s"},
        {"video_id": "456", "bucket": "33-60s"},
        {"video_id": "123", "bucket": "13-18s"}  # Duplicate!
    ]

    try:
        detect_duplicates_across_buckets(files)
        raise AssertionError("Expected ValueError to be raised")
    except ValueError as e:
        error_msg = str(e)
        assert "123" in error_msg
        assert "appears in multiple buckets" in error_msg


def test_empty_file_list():
    """Test validation passes for empty file list"""
    files = []
    # Should not raise
    detect_duplicates_across_buckets(files)


# Main test runner
if __name__ == "__main__":
    print("Running Stage 2.5 Unit Tests\n")

    runner = TestRunner()

    # Load winning buckets tests
    print("Testing load_winning_buckets:")
    runner.run_test("test_load_valid_winner_analysis", test_load_valid_winner_analysis)
    runner.run_test("test_load_missing_file", test_load_missing_file)
    runner.run_test("test_load_missing_top_3_buckets_field", test_load_missing_top_3_buckets_field)
    runner.run_test("test_load_empty_buckets_list", test_load_empty_buckets_list)

    # Validate checkpoint tests
    print("\nTesting validate_checkpoint:")
    runner.run_test("test_validate_completed_checkpoint", test_validate_completed_checkpoint)
    runner.run_test("test_validate_missing_required_fields", test_validate_missing_required_fields)
    runner.run_test("test_validate_zero_completions", test_validate_zero_completions)

    # Detect duplicates tests
    print("\nTesting detect_duplicates_across_buckets:")
    runner.run_test("test_no_duplicates", test_no_duplicates)
    runner.run_test("test_duplicate_detected", test_duplicate_detected)
    runner.run_test("test_empty_file_list", test_empty_file_list)

    # Print summary
    success = runner.summary()
    sys.exit(0 if success else 1)
