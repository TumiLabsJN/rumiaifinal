#!/usr/bin/env python3
"""
One-time data patch: Add is_top_performer tags to existing selected_videos.json files.

This allows Stage 4 to run correctly with current data without re-running Stage 1.

Usage:
    python3 patch_selected_videos_tags.py --client test_final --target test_vitamin
"""

import json
import argparse
from pathlib import Path

def patch_bucket(bucket_path: Path):
    """Add is_top_performer tags to selected_videos.json in a bucket."""
    
    selected_videos_path = bucket_path / "selected_videos.json"
    
    if not selected_videos_path.exists():
        print(f"  ⚠️  Skipping {bucket_path.name}: selected_videos.json not found")
        return False
    
    # Load existing file
    with open(selected_videos_path, 'r') as f:
        data = json.load(f)
    
    # Check if already tagged
    if data['videos'] and 'is_top_performer' in data['videos'][0]:
        print(f"  ✅ {bucket_path.name}: Already tagged (skipping)")
        return False
    
    # Extract counts from metadata
    top_count = data.get('top_count', 0)
    bottom_count = data.get('bottom_count', 0)
    total_videos = len(data['videos'])
    
    print(f"  🔧 {bucket_path.name}: Tagging {total_videos} videos ({top_count} top, {bottom_count} bottom)")
    
    # Tag videos based on position (already sorted by engagement from Apify)
    # Top performers: first N videos
    for i in range(min(top_count, total_videos)):
        data['videos'][i]['is_top_performer'] = True
    
    # Bottom performers: remaining videos
    for i in range(top_count, total_videos):
        data['videos'][i]['is_top_performer'] = False
    
    # Backup original file
    backup_path = bucket_path / "selected_videos.json.backup"
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"     📦 Backup saved: {backup_path.name}")
    
    # Write patched file
    with open(selected_videos_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"     ✅ Patched: {selected_videos_path.name}")
    
    # Verify
    tagged_top = sum(1 for v in data['videos'] if v.get('is_top_performer') == True)
    tagged_bottom = sum(1 for v in data['videos'] if v.get('is_top_performer') == False)
    print(f"     ✓ Verification: {tagged_top} top, {tagged_bottom} bottom")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch selected_videos.json with is_top_performer tags")
    parser.add_argument('--client', required=True, help='Client ID (e.g., test_final)')
    parser.add_argument('--target', required=True, help='Target name (e.g., test_vitamin)')
    parser.add_argument('--mode-strategy', default='top_contrastive', help='Mode and strategy (default: top_contrastive)')
    args = parser.parse_args()
    
    # Construct path
    base_path = Path(f"/home/jorge/rumiaifinal/data/clients/{args.client}/hashtags/{args.target}/{args.mode_strategy}/buckets")
    
    if not base_path.exists():
        print(f"❌ ERROR: Path not found: {base_path}")
        return 1
    
    print("=" * 70)
    print("DATA PATCH: Adding is_top_performer tags to selected_videos.json")
    print("=" * 70)
    print(f"\nTarget: {base_path}")
    print(f"\nScanning buckets...")
    
    # Find all bucket directories
    buckets = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('bucket_')])
    
    if not buckets:
        print(f"\n❌ No bucket directories found in {base_path}")
        return 1
    
    print(f"Found {len(buckets)} buckets\n")
    
    # Patch each bucket
    patched_count = 0
    for bucket_path in buckets:
        if patch_bucket(bucket_path):
            patched_count += 1
    
    print("\n" + "=" * 70)
    print(f"✅ COMPLETE: Patched {patched_count}/{len(buckets)} buckets")
    print("=" * 70)
    
    if patched_count > 0:
        print("\n📋 Next steps:")
        print("  1. Re-run Stage 4 to use the new tags")
        print("  2. Stage 4 will create proper 80/20 split in rf_transformed.csv")
        print("  3. Stage 5 Random Forest training will work with both classes")
        print("\n⚠️  Note: This is a one-time workaround. Future runs should use")
        print("   the fixed Stage 1 code that tags videos automatically.")
    
    return 0


if __name__ == '__main__':
    exit(main())
