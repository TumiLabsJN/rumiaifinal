"""
Phase 2 Integration Test: End-to-end Stage 2.5 with synthetic data

Tests complete file organization flow with 3 synthetic files.

Run with: python3 ml_pipeline/tests/run_integration_test_phase2.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from ml_pipeline.stage2_5_organize import stage_2_5_file_organization_main
from ml_pipeline.stage2_5_organize import file_organizer


def run_phase2_integration_test():
    """Run Phase 2 integration test with synthetic data"""

    print("="*70)
    print("Phase 2 Integration Test: Stage 2.5 File Organization")
    print("="*70)

    # IMPORTANT: Temporarily override SOURCE_DIR constant for test
    original_source_dir = file_organizer.SOURCE_DIR
    file_organizer.SOURCE_DIR = "/tmp/rumiai_test/insights/"

    try:
        print("\n1. Verifying test fixtures exist...")
        analysis_base = "/tmp/rumiai_test/analysis"

        # Check winner_analysis.json exists
        if not os.path.exists(f"{analysis_base}/winner_analysis.json"):
            print("   ✗ winner_analysis.json missing!")
            return False

        # Check source files exist
        source_dir = "/tmp/rumiai_test/insights/"
        source_files = [
            "test_video_001_temporal_windows_updated.json",
            "test_video_002_temporal_windows_updated.json",
            "test_video_003_temporal_windows_updated.json"
        ]

        for f in source_files:
            if not os.path.exists(f"{source_dir}{f}"):
                print(f"   ✗ Source file missing: {f}")
                return False

        print("   ✓ All test fixtures exist")

        print("\n2. Running Stage 2.5 file organization...")
        summary = stage_2_5_file_organization_main(analysis_base)

        print(f"\n3. Checking results:")
        print(f"   - Moved: {summary['moved_count']} files")
        print(f"   - Skipped: {summary['skipped_already_organized']} files")
        print(f"   - Missing: {summary['missing_count']} files")
        print(f"   - Total processed: {summary['total_processed']}")
        print(f"   - Winning buckets: {summary['winning_buckets']}")

        # Verify expected results
        errors = []

        if summary['moved_count'] != 3:
            errors.append(f"Expected moved_count=3, got {summary['moved_count']}")

        if summary['skipped_already_organized'] != 0:
            errors.append(f"Expected skipped=0, got {summary['skipped_already_organized']}")

        if summary['missing_count'] != 0:
            errors.append(f"Expected missing=0, got {summary['missing_count']}")

        if summary['winning_buckets'] != ["18-33s"]:
            errors.append(f"Expected winning_buckets=['18-33s'], got {summary['winning_buckets']}")

        if errors:
            print("\n   ✗ Summary validation failed:")
            for error in errors:
                print(f"      - {error}")
            return False

        print("   ✓ Summary validation passed")

        print("\n4. Verifying files moved to bucket directory...")
        target_dir = f"{analysis_base}/buckets/bucket_18-33s/analysis/insights/"

        if not os.path.exists(target_dir):
            print(f"   ✗ Target directory not created: {target_dir}")
            return False

        target_files = []
        for f in source_files:
            target_path = f"{target_dir}{f}"
            if os.path.exists(target_path):
                target_files.append(f)
                print(f"   ✓ Found: {f}")
            else:
                print(f"   ✗ Missing: {f}")
                errors.append(f"Target file missing: {f}")

        if len(target_files) != 3:
            print(f"   ✗ Expected 3 files in target directory, found {len(target_files)}")
            return False

        print(f"   ✓ All 3 files moved to bucket directory")

        print("\n5. Verifying source directory empty...")
        remaining_files = os.listdir(source_dir)
        if len(remaining_files) > 0:
            print(f"   ✗ Source directory not empty: {remaining_files}")
            return False

        print("   ✓ Source directory empty")

        print("\n" + "="*70)
        print("✅ Phase 2 Integration Test: PASSED")
        print("="*70)
        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Restore original SOURCE_DIR
        file_organizer.SOURCE_DIR = original_source_dir


if __name__ == "__main__":
    success = run_phase2_integration_test()
    sys.exit(0 if success else 1)
