#!/usr/bin/env python3
"""
Test Stage 2.5 Enhancement: selection_manifest.json creation

Tests the new create_selection_manifest() function using existing test data.
This mimics what the full pipeline does in Step 7 of Stage 2.5.

Usage:
    python test_stage2_5_enhancement.py
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline.stage2_5_organize.file_organizer import (
    load_winning_buckets,
    create_selection_manifest
)

def main():
    print("="*80)
    print("TESTING STAGE 2.5 ENHANCEMENT: selection_manifest.json Creation")
    print("="*80)
    print()

    # Use existing test data path
    analysis_base = "/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive"

    print(f"Test data location: {analysis_base}")
    print()

    # Step 1: Load winning buckets (production code)
    print("Step 1: Loading winning buckets from winner_analysis.json...")
    try:
        winning_buckets = load_winning_buckets(analysis_base)
        print(f"✓ Loaded winning buckets: {winning_buckets}")
    except Exception as e:
        print(f"✗ Failed to load winning buckets: {e}")
        return 1

    print()

    # Step 2: Create selection manifest (NEW production code)
    print("Step 2: Creating selection_manifest.json...")
    try:
        create_selection_manifest(analysis_base, winning_buckets)
        print(f"✓ selection_manifest.json created")
    except Exception as e:
        print(f"✗ Failed to create manifest: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()

    # Step 3: Verify the manifest was created and has correct structure
    print("Step 3: Verifying selection_manifest.json...")
    manifest_path = f"{analysis_base}/selection_manifest.json"

    if not os.path.exists(manifest_path):
        print(f"✗ Manifest file not created at: {manifest_path}")
        return 1

    print(f"✓ Manifest file exists: {manifest_path}")

    # Load and validate structure
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check required fields
    required_fields = ['hashtag', 'selected_buckets', 'videos_by_bucket']
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        print(f"✗ Manifest missing required fields: {missing}")
        return 1

    print(f"✓ Manifest has all required fields: {required_fields}")

    # Validate hashtag
    print(f"  - hashtag: {manifest['hashtag']}")

    # Validate buckets
    print(f"  - selected_buckets: {manifest['selected_buckets']}")

    # Validate videos_by_bucket structure
    print(f"  - videos_by_bucket:")
    total_top = 0
    total_bottom = 0
    for bucket, videos in manifest['videos_by_bucket'].items():
        top_count = len(videos['top_performers'])
        bottom_count = len(videos['bottom_performers'])
        total_top += top_count
        total_bottom += bottom_count
        print(f"    - {bucket}: {top_count} top, {bottom_count} bottom")

    print()
    print(f"✓ Total videos in manifest: {total_top} top + {total_bottom} bottom = {total_top + total_bottom}")

    # Step 4: Verify Stage 2.6 can read the manifest
    print()
    print("Step 4: Verifying Stage 2.6 compatibility...")

    # Check that all video IDs in manifest have transcripts
    missing_transcripts = []
    for bucket, videos in manifest['videos_by_bucket'].items():
        all_videos = videos['top_performers'] + videos['bottom_performers']
        for video_id in all_videos[:3]:  # Check first 3 per bucket as sample
            transcript_path = f"/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json"
            if not os.path.exists(transcript_path):
                missing_transcripts.append(video_id)

    if missing_transcripts:
        print(f"⚠️  Warning: {len(missing_transcripts)} sample videos missing transcripts")
    else:
        print(f"✓ Sample videos have transcripts available")

    print()
    print("="*80)
    print("TEST COMPLETE: Stage 2.5 Enhancement")
    print("="*80)
    print()
    print(f"✅ selection_manifest.json created successfully")
    print(f"✅ Manifest structure validated")
    print(f"✅ Ready for Stage 2.6 pattern discovery")
    print()
    print(f"Manifest location: {manifest_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
