#!/usr/bin/env python3
"""
Fix Test 3 Checkpoints with Correct Video IDs

Purpose: Update Test 3's corrupted checkpoints to reflect the actual videos
         that now exist in each bucket after copying from Test 4.

The checkpoint should match the actual files in the file system.
"""

import os
import json
from datetime import datetime
from pathlib import Path

# Test 3 base path
TEST3_BASE = Path("/home/jorge/rumiaifinal/data/clients/rollo_test3/hashtags/wellness_test3/top_contrastive")

# Buckets to fix
BUCKETS = ["3-9s", "60-90s", "18-33s"]


def get_actual_video_ids(bucket_name):
    """
    Get list of video IDs that actually exist in the bucket's videos directory.

    Args:
        bucket_name: str, e.g., "3-9s"

    Returns:
        list of video IDs (strings)
    """
    videos_dir = TEST3_BASE / "buckets" / f"bucket_{bucket_name}" / "videos"

    if not videos_dir.exists():
        print(f"  ❌ Videos directory not found: {videos_dir}")
        return []

    video_files = [f.stem for f in videos_dir.glob("*.mp4")]
    return sorted(video_files)


def get_actual_insights_ids(bucket_name):
    """
    Get list of video IDs that have insights files.

    Args:
        bucket_name: str, e.g., "3-9s"

    Returns:
        list of video IDs (strings)
    """
    insights_dir = TEST3_BASE / "buckets" / f"bucket_{bucket_name}" / "analysis" / "insights"

    if not insights_dir.exists():
        print(f"  ❌ Insights directory not found: {insights_dir}")
        return []

    insight_files = [f.stem.replace("_temporal_windows_updated", "")
                     for f in insights_dir.glob("*_temporal_windows_updated.json")]
    return sorted(insight_files)


def fix_checkpoint(bucket_name):
    """
    Fix checkpoint for a specific bucket.

    Args:
        bucket_name: str, e.g., "3-9s"
    """
    print(f"\n{'='*70}")
    print(f"Fixing checkpoint for bucket: {bucket_name}")
    print('='*70)

    checkpoint_path = TEST3_BASE / "buckets" / f"bucket_{bucket_name}" / "checkpoints" / "stage_2_checkpoint.json"

    # Get actual video IDs from file system
    actual_video_ids = get_actual_video_ids(bucket_name)
    actual_insights_ids = get_actual_insights_ids(bucket_name)

    print(f"\n📁 File System Status:")
    print(f"  Videos in directory: {len(actual_video_ids)}")
    print(f"  Insights in directory: {len(actual_insights_ids)}")

    # Verify videos and insights match
    if set(actual_video_ids) != set(actual_insights_ids):
        print(f"\n  ⚠️  WARNING: Videos and insights don't match!")
        missing_insights = set(actual_video_ids) - set(actual_insights_ids)
        extra_insights = set(actual_insights_ids) - set(actual_video_ids)
        if missing_insights:
            print(f"  Missing insights for: {list(missing_insights)[:3]}...")
        if extra_insights:
            print(f"  Extra insights for: {list(extra_insights)[:3]}...")

    # Load current checkpoint
    if not checkpoint_path.exists():
        print(f"\n  ❌ Checkpoint not found: {checkpoint_path}")
        return False

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    print(f"\n📄 Current Checkpoint:")
    print(f"  Claimed completed: {checkpoint.get('completed', 0)}")
    print(f"  Claimed total: {checkpoint.get('total_videos', 0)}")
    print(f"  Status: {checkpoint.get('status', 'unknown')}")

    # Show first 3 IDs from checkpoint vs actual
    checkpoint_ids = checkpoint.get('completed_video_ids', [])
    print(f"\n🔍 ID Comparison (first 3):")
    print(f"  Checkpoint IDs: {checkpoint_ids[:3]}")
    print(f"  Actual IDs:     {actual_video_ids[:3]}")

    if set(checkpoint_ids) == set(actual_video_ids):
        print(f"\n  ✅ Checkpoint already correct! No changes needed.")
        return True

    # Create backup
    backup_path = checkpoint_path.with_suffix('.json.backup_before_fix')
    with open(backup_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"\n💾 Backup saved: {backup_path.name}")

    # Update checkpoint with correct IDs
    checkpoint['completed_video_ids'] = actual_video_ids
    checkpoint['completed'] = len(actual_video_ids)
    checkpoint['total_videos'] = len(actual_video_ids)
    checkpoint['remaining'] = 0
    checkpoint['failed_video_ids'] = []
    checkpoint['failed'] = 0
    checkpoint['status'] = 'completed'
    checkpoint['last_checkpoint'] = datetime.utcnow().isoformat()

    # Add fix metadata
    if 'fix_metadata' not in checkpoint:
        checkpoint['fix_metadata'] = {}
    checkpoint['fix_metadata']['fixed_at'] = datetime.utcnow().isoformat()
    checkpoint['fix_metadata']['reason'] = 'Checkpoint corruption - replaced with actual file system video IDs'
    checkpoint['fix_metadata']['previous_count'] = len(checkpoint_ids)
    checkpoint['fix_metadata']['new_count'] = len(actual_video_ids)

    # Save updated checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\n✅ Updated Checkpoint:")
    print(f"  Completed: {checkpoint['completed']}")
    print(f"  Total: {checkpoint['total_videos']}")
    print(f"  Video IDs: {len(checkpoint['completed_video_ids'])} entries")
    print(f"  Status: {checkpoint['status']}")

    return True


def main():
    """Fix all bucket checkpoints."""
    print("="*70)
    print("Test 3 Checkpoint Fix Script")
    print("="*70)
    print(f"\nTest 3 Base: {TEST3_BASE}")
    print(f"Buckets to fix: {', '.join(BUCKETS)}")

    results = {}
    for bucket in BUCKETS:
        try:
            success = fix_checkpoint(bucket)
            results[bucket] = "✅ Success" if success else "❌ Failed"
        except Exception as e:
            print(f"\n❌ Error fixing {bucket}: {e}")
            results[bucket] = f"❌ Error: {e}"

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    for bucket, result in results.items():
        print(f"  {bucket}: {result}")

    all_success = all("✅" in r for r in results.values())
    if all_success:
        print(f"\n✅✅✅ All checkpoints fixed successfully!")
        return 0
    else:
        print(f"\n⚠️  Some checkpoints failed to fix. Check output above.")
        return 1


if __name__ == "__main__":
    exit(main())
