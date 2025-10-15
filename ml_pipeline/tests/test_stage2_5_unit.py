"""
Unit tests for Stage 2.5: File Organization

Tests individual functions with mock data, no external dependencies.

Source: Phase 1 testing recommendations
"""

import pytest
import json
import os
from pathlib import Path

from ml_pipeline.stage2_5_organize.file_organizer import (
    load_winning_buckets,
    validate_checkpoint,
    detect_duplicates_across_buckets
)


class TestLoadWinningBuckets:
    """Test load_winning_buckets function"""

    def test_load_valid_winner_analysis(self, tmp_path):
        """Test loading valid winner_analysis.json"""
        # Setup
        analysis_base = tmp_path / "analysis"
        analysis_base.mkdir()

        winner_analysis = {
            "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
            "top_100_distribution": {"18-33s": 45, "33-60s": 30, "13-18s": 20},
            "winner_coverage": 95.0
        }

        with open(analysis_base / "winner_analysis.json", 'w') as f:
            json.dump(winner_analysis, f)

        # Test
        buckets = load_winning_buckets(str(analysis_base))

        # Verify
        assert buckets == ["18-33s", "33-60s", "13-18s"]
        assert len(buckets) == 3

    def test_load_missing_file(self, tmp_path):
        """Test FileNotFoundError when winner_analysis.json missing"""
        analysis_base = tmp_path / "analysis"
        # Note: directory doesn't exist

        with pytest.raises(FileNotFoundError) as excinfo:
            load_winning_buckets(str(analysis_base))

        assert "winner_analysis.json not found" in str(excinfo.value)
        assert "Stage 1.3" in str(excinfo.value)

    def test_load_missing_top_3_buckets_field(self, tmp_path):
        """Test ValueError when top_3_buckets field missing"""
        # Setup
        analysis_base = tmp_path / "analysis"
        analysis_base.mkdir()

        winner_analysis = {
            "top_100_distribution": {"18-33s": 45},
            # Missing 'top_3_buckets' field
        }

        with open(analysis_base / "winner_analysis.json", 'w') as f:
            json.dump(winner_analysis, f)

        # Test
        with pytest.raises(ValueError) as excinfo:
            load_winning_buckets(str(analysis_base))

        assert "missing 'top_3_buckets' field" in str(excinfo.value)

    def test_load_empty_buckets_list(self, tmp_path):
        """Test ValueError when top_3_buckets is empty"""
        # Setup
        analysis_base = tmp_path / "analysis"
        analysis_base.mkdir()

        winner_analysis = {
            "top_3_buckets": [],  # Empty list
        }

        with open(analysis_base / "winner_analysis.json", 'w') as f:
            json.dump(winner_analysis, f)

        # Test
        with pytest.raises(ValueError) as excinfo:
            load_winning_buckets(str(analysis_base))

        assert "empty" in str(excinfo.value).lower()

    def test_load_invalid_type(self, tmp_path):
        """Test TypeError when top_3_buckets is not a list"""
        # Setup
        analysis_base = tmp_path / "analysis"
        analysis_base.mkdir()

        winner_analysis = {
            "top_3_buckets": "18-33s",  # String instead of list
        }

        with open(analysis_base / "winner_analysis.json", 'w') as f:
            json.dump(winner_analysis, f)

        # Test
        with pytest.raises(TypeError) as excinfo:
            load_winning_buckets(str(analysis_base))

        assert "must be list" in str(excinfo.value)


class TestValidateCheckpoint:
    """Test validate_checkpoint function"""

    def test_validate_completed_checkpoint(self):
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

    def test_validate_missing_required_fields(self):
        """Test ValueError when required fields missing"""
        checkpoint = {
            "stage": "video_processing",
            # Missing 'bucket', 'completed_video_ids', 'status', 'total_videos'
        }

        with pytest.raises(ValueError) as excinfo:
            validate_checkpoint(checkpoint, "18-33s")

        assert "invalid schema" in str(excinfo.value)
        assert "missing" in str(excinfo.value)

    def test_validate_partial_checkpoint(self):
        """Test warning logged for status='in_progress' checkpoint"""
        checkpoint = {
            "stage": "video_processing",
            "bucket": "18-33s",
            "completed_video_ids": ["123", "456"],
            "status": "in_progress",  # Not completed
            "total_videos": 10
        }

        # Should not raise, but log warning
        video_ids = validate_checkpoint(checkpoint, "18-33s")

        assert video_ids == ["123", "456"]

    def test_validate_zero_completions(self):
        """Test returns empty list for zero completed videos"""
        checkpoint = {
            "stage": "video_processing",
            "bucket": "18-33s",
            "completed_video_ids": [],  # Empty
            "status": "completed",
            "total_videos": 10
        }

        video_ids = validate_checkpoint(checkpoint, "18-33s")

        assert video_ids == []

    def test_validate_invalid_type_completed_video_ids(self):
        """Test ValueError when completed_video_ids is not a list"""
        checkpoint = {
            "stage": "video_processing",
            "bucket": "18-33s",
            "completed_video_ids": "123",  # String instead of list
            "status": "completed",
            "total_videos": 10
        }

        with pytest.raises(ValueError) as excinfo:
            validate_checkpoint(checkpoint, "18-33s")

        assert "must be list" in str(excinfo.value)


class TestDetectDuplicates:
    """Test detect_duplicates_across_buckets function"""

    def test_no_duplicates(self):
        """Test validation passes when no duplicates"""
        files = [
            {"video_id": "123", "bucket": "18-33s"},
            {"video_id": "456", "bucket": "33-60s"},
            {"video_id": "789", "bucket": "13-18s"}
        ]

        # Should not raise
        detect_duplicates_across_buckets(files)

    def test_duplicate_detected(self):
        """Test ValueError raised when duplicate video_id detected"""
        files = [
            {"video_id": "123", "bucket": "18-33s"},
            {"video_id": "456", "bucket": "33-60s"},
            {"video_id": "123", "bucket": "13-18s"}  # Duplicate!
        ]

        with pytest.raises(ValueError) as excinfo:
            detect_duplicates_across_buckets(files)

        error_msg = str(excinfo.value)
        assert "123" in error_msg
        assert "appears in multiple buckets" in error_msg
        assert "18-33s" in error_msg
        assert "13-18s" in error_msg

    def test_empty_file_list(self):
        """Test validation passes for empty file list"""
        files = []

        # Should not raise
        detect_duplicates_across_buckets(files)

    def test_single_file(self):
        """Test validation passes for single file"""
        files = [
            {"video_id": "123", "bucket": "18-33s"}
        ]

        # Should not raise
        detect_duplicates_across_buckets(files)
