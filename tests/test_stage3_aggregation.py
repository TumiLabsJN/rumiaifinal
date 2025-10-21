#!/usr/bin/env python3
"""
Unit tests for Stage 3: Feature Aggregation

Test Coverage:
1. test_bucket_33_60s() - Separate middle segments (5 windows)
2. test_bucket_9_13s_aggregation() - Middle aggregation (3 windows)
3. test_malformed_json() - Error handling

Source: FeatureAggregationTI.md Section 5.1
"""

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.stage3_aggregation import aggregate_features


class TestBucket3360s:
    """Test bucket with separate middle segments (33-60s duration)."""

    def test_bucket_33_60s(self, tmp_path):
        """Test 50s video with 5 separate middle segments (7 windows total)."""
        # Setup: Create bucket directory structure
        bucket_path = tmp_path / "bucket_33-60s"
        insights_dir = bucket_path / "analysis" / "insights"
        insights_dir.mkdir(parents=True)

        # Copy test data from insights/
        test_video = "238506412723073_temporal_windows_updated.json"
        source_file = Path("/home/jorge/rumiaifinal/insights") / test_video

        if not source_file.exists():
            pytest.skip(f"Test data file not found: {source_file}")

        shutil.copy(source_file, insights_dir / test_video)

        # Execute
        csv_path, summary_path = aggregate_features(str(bucket_path))

        # Verify CSV structure
        df = pd.read_csv(csv_path)
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        assert len(df.columns) == 150, f"Expected 150 columns, got {len(df.columns)}"

        # Check column naming convention
        assert 'video_id' in df.columns
        assert 'create_time' in df.columns
        assert 'gender' in df.columns
        assert 'hook_scene_count' in df.columns
        assert 'hook_word_count' in df.columns
        assert 'hook_energy_level' in df.columns

        # Check middle segments (separate, not aggregated)
        assert 'middle_1_scene_count' in df.columns
        assert 'middle_2_scene_count' in df.columns
        assert 'middle_3_scene_count' in df.columns
        assert 'middle_4_scene_count' in df.columns
        assert 'middle_5_scene_count' in df.columns

        # Verify middle_aggregate does NOT exist (33-60s keeps separate segments)
        assert 'middle_aggregate_scene_count' not in df.columns

        # Check closing window
        assert 'closing_scene_count' in df.columns
        assert 'closing_energy_level' in df.columns

        # Verify summary JSON
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary['videos_processed'] == 1
        assert summary['videos_skipped'] == 0
        assert summary['output_csv']['rows'] == 1
        assert summary['output_csv']['columns'] == 150


class TestBucket913sAggregation:
    """Test bucket with middle segment aggregation (9-13s duration)."""

    def test_bucket_9_13s_aggregation(self, tmp_path):
        """Test 10s video with middle_aggregate (3 windows total: hook + middle_aggregate + closing)."""
        # Setup: Create bucket directory structure
        bucket_path = tmp_path / "bucket_9-13s"
        insights_dir = bucket_path / "analysis" / "insights"
        insights_dir.mkdir(parents=True)

        # Copy test data from insights/
        test_video = "7099027230512139526_temporal_windows_updated.json"
        source_file = Path("/home/jorge/rumiaifinal/insights") / test_video

        if not source_file.exists():
            pytest.skip(f"Test data file not found: {source_file}")

        shutil.copy(source_file, insights_dir / test_video)

        # Execute
        csv_path, summary_path = aggregate_features(str(bucket_path))

        # Verify CSV structure
        df = pd.read_csv(csv_path)
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        assert len(df.columns) == 66, f"Expected 66 columns, got {len(df.columns)}"

        # Check aggregation columns exist
        assert 'middle_aggregate_scene_count' in df.columns
        assert 'middle_aggregate_word_count' in df.columns
        assert 'middle_aggregate_energy_level' in df.columns

        # Verify separate middle segments do NOT exist (aggregated bucket)
        assert 'middle_1_scene_count' not in df.columns
        assert 'middle_2_scene_count' not in df.columns
        assert 'middle_3_scene_count' not in df.columns

        # Verify aggregation strategies applied correctly
        # SUM features: scene_count should be sum of all middle segments
        # (Cannot verify exact value without reading raw JSON, but column should exist)
        assert df['middle_aggregate_scene_count'].notna().all()

        # Verify summary JSON
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary['videos_processed'] == 1
        assert summary['videos_skipped'] == 0
        assert summary['output_csv']['rows'] == 1
        assert summary['output_csv']['columns'] == 66


class TestErrorHandling:
    """Test graceful error handling."""

    def test_malformed_json(self, tmp_path):
        """Test graceful handling of malformed JSON (skip bad video, process good video)."""
        # Setup: Create bucket with 1 good + 1 bad JSON
        bucket_path = tmp_path / "bucket_33-60s"
        insights_dir = bucket_path / "analysis" / "insights"
        insights_dir.mkdir(parents=True)

        # Copy good video
        good_video = "238506412723073_temporal_windows_updated.json"
        source_file = Path("/home/jorge/rumiaifinal/insights") / good_video

        if not source_file.exists():
            pytest.skip(f"Test data file not found: {source_file}")

        shutil.copy(source_file, insights_dir / good_video)

        # Create malformed JSON file
        bad_json_path = insights_dir / "bad_video_temporal_windows_updated.json"
        with open(bad_json_path, 'w') as f:
            f.write("{invalid json content without closing brace")

        # Execute
        csv_path, summary_path = aggregate_features(str(bucket_path))

        # Verify only good video processed
        df = pd.read_csv(csv_path)
        assert len(df) == 1, f"Expected 1 row (good video only), got {len(df)}"

        # Verify summary reports skipped video
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary['videos_processed'] == 1
        assert summary['videos_skipped'] == 1
        assert summary['skipped_reasons']['malformed_json'] == 1


class TestZeroValidVideos:
    """Test failure when all videos fail validation."""

    def test_all_videos_fail(self, tmp_path):
        """Test that Stage 3 raises ValueError when zero valid videos processed."""
        # Setup: Create bucket with only malformed JSON
        bucket_path = tmp_path / "bucket_33-60s"
        insights_dir = bucket_path / "analysis" / "insights"
        insights_dir.mkdir(parents=True)

        # Create only malformed JSON file
        bad_json_path = insights_dir / "bad_video_temporal_windows_updated.json"
        with open(bad_json_path, 'w') as f:
            f.write("{invalid json")

        # Execute and expect ValueError
        with pytest.raises(ValueError, match="No valid videos processed"):
            aggregate_features(str(bucket_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
