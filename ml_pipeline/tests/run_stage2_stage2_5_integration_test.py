"""
Stage 2 & 2.5 Integration Test

Tests Stage 2 (Video Processing) and Stage 2.5 (File Organization) using
existing Stage 1 output from test_vitamin cluster.

This test:
1. Loads existing winner_analysis.json and selected_videos.json
2. Runs Stage 2 video processing for each winning bucket
3. Runs Stage 2.5 file organization
4. Validates the complete pipeline output

Run with: python3 ml_pipeline/tests/run_stage2_stage2_5_integration_test.py
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from ml_pipeline.stage2_processing import stage_2_video_processing_main
from ml_pipeline.stage2_5_organize import stage_2_5_file_organization_main


def run_stage2_stage2_5_test():
    """Run Stage 2 & 2.5 integration test using existing Stage 1 output"""

    # Set DATA_ROOT to correct location
    os.environ['DATA_ROOT'] = '/home/jorge/rumiaifinal/data'

    print("="*80)
    print("STAGE 2 & 2.5 INTEGRATION TEST")
    print("="*80)
    print()

    # Test data location (from previous Stage 1 run)
    analysis_base = Path("/home/jorge/rumiaifinal/data/clients/testy_client/hashtags/test_vitamin/top_contrastive")

    try:
        # ===== VERIFY STAGE 1 OUTPUT EXISTS =====
        print("Step 1: Verifying Stage 1 output exists...")

        winner_analysis_path = analysis_base / "winner_analysis.json"
        config_path = analysis_base / "config.json"

        if not winner_analysis_path.exists():
            print(f"✗ winner_analysis.json not found at: {winner_analysis_path}")
            print("  Run Stage 1 first to generate test data")
            return False

        if not config_path.exists():
            print(f"✗ config.json not found at: {config_path}")
            return False

        print(f"✓ Found winner_analysis.json")
        print(f"✓ Found config.json")

        # Load config and winner analysis
        with open(config_path) as f:
            config = json.load(f)

        with open(winner_analysis_path) as f:
            winner_analysis = json.load(f)

        winning_buckets = winner_analysis['top_3_buckets']
        print(f"✓ Winning buckets: {', '.join(winning_buckets)}")

        # Verify selected_videos.json exists for each bucket
        for bucket_name in winning_buckets:
            bucket_path = analysis_base / f"buckets/bucket_{bucket_name}/selected_videos.json"
            if not bucket_path.exists():
                print(f"✗ Missing selected_videos.json for {bucket_name}")
                return False

            with open(bucket_path) as f:
                bucket_data = json.load(f)
            videos = bucket_data['videos'] if 'videos' in bucket_data else bucket_data
            print(f"✓ Bucket {bucket_name}: {len(videos)} videos selected")

        # ===== STAGE 2: VIDEO PROCESSING =====
        print("\n" + "="*80)
        print("STAGE 2: VIDEO PROCESSING")
        print("="*80)
        print()

        stage2_summaries = {}
        for bucket_name in winning_buckets:
            print(f"\nProcessing bucket: {bucket_name}")
            print("-" * 40)

            # Load selected videos
            bucket_videos_path = analysis_base / f"buckets/bucket_{bucket_name}/selected_videos.json"
            with open(bucket_videos_path) as f:
                bucket_data = json.load(f)

            # Extract video list from wrapper structure
            video_list = bucket_data['videos'] if 'videos' in bucket_data else bucket_data

            print(f"Videos to process: {len(video_list)}")

            # Run Stage 2
            try:
                summary = stage_2_video_processing_main(
                    config=config,
                    video_list=video_list,
                    bucket_name=bucket_name,
                    enable_pause_support=False  # Disable for automated test
                )

                stage2_summaries[bucket_name] = summary
                print(f"\n✓ Bucket {bucket_name} complete:")
                print(f"  Total: {summary['total']}")
                print(f"  Completed: {summary['completed']}")
                print(f"  Failed: {summary['failed']}")
                print(f"  Status: {summary['status']}")

            except Exception as e:
                print(f"\n✗ Bucket {bucket_name} failed: {e}")
                import traceback
                traceback.print_exc()
                # Continue with other buckets
                continue

        # Calculate Stage 2 totals
        total_videos = sum(s['total'] for s in stage2_summaries.values())
        completed_videos = sum(s['completed'] for s in stage2_summaries.values())
        failed_videos = sum(s['failed'] for s in stage2_summaries.values())

        print("\n" + "="*80)
        print(f"STAGE 2 SUMMARY: {completed_videos}/{total_videos} videos processed")
        print(f"  Successful: {completed_videos}")
        print(f"  Failed: {failed_videos}")
        print("="*80)

        # ===== STAGE 2.5: FILE ORGANIZATION =====
        print("\n" + "="*80)
        print("STAGE 2.5: FILE ORGANIZATION")
        print("="*80)
        print()

        print("Organizing temporal_windows files into bucket directories...")

        organization_summary = stage_2_5_file_organization_main(
            analysis_base=str(analysis_base)
        )

        print(f"\n✓ Stage 2.5 complete:")
        print(f"  Moved: {organization_summary['moved_count']} files")
        print(f"  Skipped (already organized): {organization_summary['skipped_already_organized']} files")
        print(f"  Missing: {organization_summary['missing_count']} files")
        print(f"  Total processed: {organization_summary['total_processed']}")

        # ===== VERIFY OUTPUTS =====
        print("\n" + "="*80)
        print("VERIFYING OUTPUTS")
        print("="*80)
        print()

        print("Checking organized files in bucket directories...")
        for bucket_name in winning_buckets:
            insights_dir = analysis_base / f"buckets/bucket_{bucket_name}/analysis/insights"

            if insights_dir.exists():
                files = list(insights_dir.glob("*_temporal_windows_updated.json"))
                print(f"✓ Bucket {bucket_name}: {len(files)} temporal_windows files")
            else:
                print(f"✗ Bucket {bucket_name}: insights directory not found")

        # ===== FINAL RESULTS =====
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"✅ Stage 2: {completed_videos}/{total_videos} videos processed")
        print(f"✅ Stage 2.5: {organization_summary['moved_count']} files organized")
        print()
        print(f"Output location: {analysis_base}")
        print("="*80)

        # Determine test success
        if completed_videos == 0:
            print("\n⚠️  WARNING: No videos successfully processed")
            print("   This may be expected if videos failed to download/process")
            return False

        if organization_summary['moved_count'] != completed_videos:
            print(f"\n⚠️  WARNING: Mismatch between processed ({completed_videos}) and organized ({organization_summary['moved_count']}) videos")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_stage2_stage2_5_test()
    sys.exit(0 if success else 1)
