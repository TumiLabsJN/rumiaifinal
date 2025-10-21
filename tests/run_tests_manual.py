#!/usr/bin/env python3
"""
Manual test runner for Stage 3 Feature Aggregation
Runs tests without pytest dependency

Source: FeatureAggregationTI.md Section 5.1
"""

import json
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd

from scripts.stage3_aggregation import aggregate_features


def test_bucket_33_60s():
    """Test 50s video with 5 separate middle segments (7 windows total)."""
    print("\n" + "="*80)
    print("TEST 1: Bucket 33-60s (Separate Middle Segments)")
    print("="*80)

    start_time = time.time()

    try:
        # Setup: Create bucket directory structure
        with tempfile.TemporaryDirectory() as tmp_dir:
            bucket_path = Path(tmp_dir) / "bucket_33-60s"
            insights_dir = bucket_path / "analysis" / "insights"
            insights_dir.mkdir(parents=True)

            # Copy test data from insights/
            test_video = "238506412723073_temporal_windows_updated.json"
            source_file = Path("/home/jorge/rumiaifinal/insights") / test_video

            if not source_file.exists():
                print(f"❌ SKIPPED: Test data file not found: {source_file}")
                return False

            shutil.copy(source_file, insights_dir / test_video)

            # Execute
            csv_path, summary_path = aggregate_features(str(bucket_path))

            # Verify CSV structure
            df = pd.read_csv(csv_path)

            # Assertions
            assert len(df) == 1, f"Expected 1 row, got {len(df)}"
            assert len(df.columns) == 150, f"Expected 150 columns, got {len(df.columns)}"

            # Check column naming convention
            required_cols = [
                'video_id', 'create_time', 'gender',
                'hook_scene_count', 'hook_word_count', 'hook_energy_level',
                'middle_1_scene_count', 'middle_2_scene_count', 'middle_3_scene_count',
                'middle_4_scene_count', 'middle_5_scene_count',
                'closing_scene_count', 'closing_energy_level'
            ]

            for col in required_cols:
                assert col in df.columns, f"Missing required column: {col}"

            # Verify middle_aggregate does NOT exist (33-60s keeps separate segments)
            assert 'middle_aggregate_scene_count' not in df.columns, \
                "middle_aggregate should not exist for bucket 33-60s"

            # Verify summary JSON
            with open(summary_path) as f:
                summary = json.load(f)

            assert summary['videos_processed'] == 1
            assert summary['videos_skipped'] == 0
            assert summary['output_csv']['rows'] == 1
            assert summary['output_csv']['columns'] == 150

            duration = time.time() - start_time
            print(f"✅ PASS - Duration: {duration:.2f}s")
            print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
            print(f"   Sample columns: {list(df.columns[:5])}")
            return True

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ FAIL - Duration: {duration:.2f}s")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False


def test_bucket_9_13s_aggregation():
    """Test 10s video with middle_aggregate (3 windows total)."""
    print("\n" + "="*80)
    print("TEST 2: Bucket 9-13s (Middle Aggregation)")
    print("="*80)

    start_time = time.time()

    try:
        # Setup: Create bucket directory structure
        with tempfile.TemporaryDirectory() as tmp_dir:
            bucket_path = Path(tmp_dir) / "bucket_9-13s"
            insights_dir = bucket_path / "analysis" / "insights"
            insights_dir.mkdir(parents=True)

            # Copy test data from insights/
            test_video = "7099027230512139526_temporal_windows_updated.json"
            source_file = Path("/home/jorge/rumiaifinal/insights") / test_video

            if not source_file.exists():
                print(f"❌ SKIPPED: Test data file not found: {source_file}")
                return False

            shutil.copy(source_file, insights_dir / test_video)

            # Execute
            csv_path, summary_path = aggregate_features(str(bucket_path))

            # Verify CSV structure
            df = pd.read_csv(csv_path)

            # Assertions
            assert len(df) == 1, f"Expected 1 row, got {len(df)}"
            assert len(df.columns) == 66, f"Expected 66 columns, got {len(df.columns)}"

            # Check aggregation columns exist
            aggregation_cols = [
                'middle_aggregate_scene_count',
                'middle_aggregate_word_count',
                'middle_aggregate_energy_level'
            ]

            for col in aggregation_cols:
                assert col in df.columns, f"Missing aggregation column: {col}"

            # Verify separate middle segments do NOT exist (aggregated bucket)
            forbidden_cols = ['middle_1_scene_count', 'middle_2_scene_count', 'middle_3_scene_count']
            for col in forbidden_cols:
                assert col not in df.columns, f"Unexpected column {col} (should be aggregated)"

            # Verify aggregation strategies applied correctly
            assert df['middle_aggregate_scene_count'].notna().all(), \
                "middle_aggregate_scene_count should not be null"

            # Verify summary JSON
            with open(summary_path) as f:
                summary = json.load(f)

            assert summary['videos_processed'] == 1
            assert summary['videos_skipped'] == 0
            assert summary['output_csv']['rows'] == 1
            assert summary['output_csv']['columns'] == 66

            duration = time.time() - start_time
            print(f"✅ PASS - Duration: {duration:.2f}s")
            print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
            print(f"   Aggregated columns verified: {aggregation_cols[:2]}")
            return True

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ FAIL - Duration: {duration:.2f}s")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False


def test_malformed_json():
    """Test graceful handling of malformed JSON."""
    print("\n" + "="*80)
    print("TEST 3: Error Handling (Malformed JSON)")
    print("="*80)

    start_time = time.time()

    try:
        # Setup: Create bucket with 1 good + 1 bad JSON
        with tempfile.TemporaryDirectory() as tmp_dir:
            bucket_path = Path(tmp_dir) / "bucket_33-60s"
            insights_dir = bucket_path / "analysis" / "insights"
            insights_dir.mkdir(parents=True)

            # Copy good video
            good_video = "238506412723073_temporal_windows_updated.json"
            source_file = Path("/home/jorge/rumiaifinal/insights") / good_video

            if not source_file.exists():
                print(f"❌ SKIPPED: Test data file not found: {source_file}")
                return False

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

            duration = time.time() - start_time
            print(f"✅ PASS - Duration: {duration:.2f}s")
            print(f"   Processed: {summary['videos_processed']}, Skipped: {summary['videos_skipped']}")
            print(f"   Skip reason: malformed_json={summary['skipped_reasons']['malformed_json']}")
            return True

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ FAIL - Duration: {duration:.2f}s")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("STAGE 3 FEATURE AGGREGATION - UNIT TESTS")
    print("="*80)

    tests = [
        ("test_bucket_33_60s", test_bucket_33_60s),
        ("test_bucket_9_13s_aggregation", test_bucket_9_13s_aggregation),
        ("test_malformed_json", test_malformed_json),
    ]

    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed

    print(f"\nTotal: {len(results)} tests")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    print("\nDetailed Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
