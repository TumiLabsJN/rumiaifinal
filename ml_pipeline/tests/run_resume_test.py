"""
Resume Behavior Test: Verify detection-based resume works

Tests that Stage 2.5 automatically skips already-organized files.

Run with: python3 ml_pipeline/tests/run_resume_test.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from ml_pipeline.stage2_5_organize import stage_2_5_file_organization_main
from ml_pipeline.stage2_5_organize import file_organizer


def run_resume_test():
    """Test detection-based resume behavior"""

    print("="*70)
    print("Resume Behavior Test: Detection-Based Resume")
    print("="*70)

    # IMPORTANT: Temporarily override SOURCE_DIR constant for test
    original_source_dir = file_organizer.SOURCE_DIR
    file_organizer.SOURCE_DIR = "/tmp/rumiai_test/insights/"

    try:
        analysis_base = "/tmp/rumiai_test/analysis"

        print("\n1. Running Stage 2.5 SECOND time (files already organized)...")
        print("   Expected: All files should be skipped (already organized)\n")

        summary = stage_2_5_file_organization_main(analysis_base)

        print(f"\n2. Checking resume results:")
        print(f"   - Moved: {summary['moved_count']} files (expected: 0)")
        print(f"   - Skipped: {summary['skipped_already_organized']} files (expected: 3)")
        print(f"   - Missing: {summary['missing_count']} files (expected: 0)")
        print(f"   - Total processed: {summary['total_processed']}")

        # Verify expected results
        errors = []

        if summary['moved_count'] != 0:
            errors.append(f"Expected moved_count=0 (no new files), got {summary['moved_count']}")

        if summary['skipped_already_organized'] != 3:
            errors.append(f"Expected skipped=3 (all files already organized), got {summary['skipped_already_organized']}")

        if summary['missing_count'] != 0:
            errors.append(f"Expected missing=0, got {summary['missing_count']}")

        if errors:
            print("\n   ✗ Resume behavior validation failed:")
            for error in errors:
                print(f"      - {error}")
            return False

        print("   ✓ Resume behavior validation passed")
        print("   ✓ All 3 files detected as already organized")
        print("   ✓ No files were moved (idempotent operation)")

        print("\n3. Verifying files still in bucket directory...")
        target_dir = f"{analysis_base}/buckets/bucket_18-33s/analysis/insights/"
        target_files = [f for f in os.listdir(target_dir) if f.endswith('.json')]

        if len(target_files) != 3:
            print(f"   ✗ Expected 3 files in target directory, found {len(target_files)}")
            return False

        print(f"   ✓ All 3 files still in bucket directory")

        print("\n4. Verifying source directory still empty...")
        source_dir = "/tmp/rumiai_test/insights/"
        remaining_files = os.listdir(source_dir)

        if len(remaining_files) > 0:
            print(f"   ✗ Source directory not empty: {remaining_files}")
            return False

        print("   ✓ Source directory still empty")

        print("\n" + "="*70)
        print("✅ Resume Behavior Test: PASSED")
        print("="*70)
        print("\nConclusion: Detection-based resume works correctly!")
        print("- Stage 2.5 is idempotent (safe to re-run)")
        print("- No checkpoint needed for resume")
        print("- Files organized once stay organized")
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
    success = run_resume_test()
    sys.exit(0 if success else 1)
