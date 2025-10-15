"""
Phase 3 Integration Test: End-to-end Stage 2.5 with REAL video data

Tests complete file organization flow with 1 real temporal_windows file
from actual RumiAI processing.

Video ID: 7384423133157100843 (61s video → 33-60s bucket)

Run with: python3 ml_pipeline/tests/run_phase3_real_video_test.py
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from ml_pipeline.stage2_5_organize import stage_2_5_file_organization_main
from ml_pipeline.stage2_5_organize import file_organizer


def run_phase3_real_video_test():
    """Run Phase 3 integration test with real video data"""

    print("="*70)
    print("Phase 3 Integration Test: Stage 2.5 with REAL Video Data")
    print("="*70)

    # Video details
    video_id = "7384423133157100843"
    bucket = "33-60s"

    # IMPORTANT: Use REAL insights directory (not /tmp)
    original_source_dir = file_organizer.SOURCE_DIR
    # SOURCE_DIR is already set to /home/jorge/rumiaifinal/insights/ - perfect!

    try:
        print(f"\n1. Verifying real temporal_windows file exists...")
        print(f"   Video ID: {video_id}")
        print(f"   Expected bucket: {bucket} (61s video)")

        source_file = f"{file_organizer.SOURCE_DIR}{video_id}_temporal_windows_updated.json"

        if not os.path.exists(source_file):
            print(f"   ✗ Real temporal_windows file not found: {source_file}")
            return False

        # Load and verify it's valid JSON with expected structure
        with open(source_file) as f:
            data = json.load(f)

        if data.get('video_id') != video_id:
            print(f"   ✗ Video ID mismatch in file: {data.get('video_id')}")
            return False

        if 'temporal_windows' not in data:
            print(f"   ✗ Missing temporal_windows in file")
            return False

        print(f"   ✓ Real temporal_windows file exists and is valid")
        print(f"   ✓ Duration: {data.get('duration')}s")
        print(f"   ✓ Has temporal_windows structure")

        print("\n2. Verifying test fixtures exist...")
        analysis_base = "/tmp/real_test/analysis"

        if not os.path.exists(f"{analysis_base}/winner_analysis.json"):
            print("   ✗ winner_analysis.json missing!")
            return False

        checkpoint_path = f"{analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json"
        if not os.path.exists(checkpoint_path):
            print(f"   ✗ stage_2_checkpoint.json missing!")
            return False

        print("   ✓ All test fixtures exist")

        print("\n3. Running Stage 2.5 with REAL temporal_windows file...")
        summary = stage_2_5_file_organization_main(analysis_base)

        print(f"\n4. Checking results:")
        print(f"   - Moved: {summary['moved_count']} file (expected: 1)")
        print(f"   - Skipped: {summary['skipped_already_organized']} files (expected: 0)")
        print(f"   - Missing: {summary['missing_count']} files (expected: 0)")
        print(f"   - Total processed: {summary['total_processed']}")
        print(f"   - Winning buckets: {summary['winning_buckets']}")

        # Verify expected results
        errors = []

        if summary['moved_count'] != 1:
            errors.append(f"Expected moved_count=1, got {summary['moved_count']}")

        if summary['skipped_already_organized'] != 0:
            errors.append(f"Expected skipped=0, got {summary['skipped_already_organized']}")

        if summary['missing_count'] != 0:
            errors.append(f"Expected missing=0, got {summary['missing_count']}")

        if summary['winning_buckets'] != ["33-60s"]:
            errors.append(f"Expected winning_buckets=['33-60s'], got {summary['winning_buckets']}")

        if errors:
            print("\n   ✗ Summary validation failed:")
            for error in errors:
                print(f"      - {error}")
            return False

        print("   ✓ Summary validation passed")

        print("\n5. Verifying REAL file moved to bucket directory...")
        target_file = f"{analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json"

        if not os.path.exists(target_file):
            print(f"   ✗ Target file not found: {target_file}")
            return False

        print(f"   ✓ Found: {video_id}_temporal_windows_updated.json in bucket_{bucket}/")

        # Verify file content is valid
        with open(target_file) as f:
            organized_data = json.load(f)

        if organized_data.get('video_id') != video_id:
            print(f"   ✗ Video ID mismatch in organized file")
            return False

        print(f"   ✓ Organized file is valid JSON with correct video_id")

        print("\n6. Verifying source file moved (not copied)...")
        if os.path.exists(source_file):
            print(f"   ✗ Source file still exists (should be moved, not copied)")
            return False

        print("   ✓ Source file removed (atomic move confirmed)")

        print("\n" + "="*70)
        print("✅ Phase 3 Integration Test: PASSED")
        print("="*70)
        print(f"\nConclusion: Stage 2.5 successfully organized REAL RumiAI output!")
        print(f"- Real temporal_windows file (61s video, video_id={video_id})")
        print(f"- Organized into correct bucket (33-60s)")
        print(f"- File moved atomically (not copied)")
        print(f"- Ready for Stage 3 (Feature Aggregation)")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Restore original SOURCE_DIR (though it shouldn't have changed)
        file_organizer.SOURCE_DIR = original_source_dir


if __name__ == "__main__":
    success = run_phase3_real_video_test()
    sys.exit(0 if success else 1)
