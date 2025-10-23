#!/usr/bin/env python3
"""
Quick test script to verify scaler creation in Stage 4.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from rumiai_v2.processors.feature_transformation import run_stage4_transformation

if __name__ == "__main__":
    bucket_path = "data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s"

    config = {
        'strategy': 'contrastive',
        'video_count': 50
    }

    print(f"Running Stage 4 on {bucket_path}")
    print(f"Config: {config}")

    # Run transformation
    success, output_files, duration = run_stage4_transformation(bucket_path, config)

    print("\n" + "="*80)
    print("TRANSFORMATION COMPLETE")
    print("="*80)

    # Check for scaler files
    ml_analysis_dir = os.path.join(bucket_path, 'ml_analysis')
    scaler_files = [f for f in os.listdir(ml_analysis_dir) if f.endswith('_scalers.pkl')]

    print(f"\nScaler files created: {len(scaler_files)}")
    for f in sorted(scaler_files):
        file_path = os.path.join(ml_analysis_dir, f)
        file_size_kb = os.path.getsize(file_path) / 1024
        print(f"  ✓ {f} ({file_size_kb:.1f} KB)")

    if len(scaler_files) == 0:
        print("  ❌ NO SCALER FILES CREATED!")
        sys.exit(1)
    else:
        print(f"\n✅ SUCCESS: {len(scaler_files)} scaler files created")
        sys.exit(0)
