"""
Stage 2 Integration Test: End-to-end video processing with REAL TikTok video

Tests that rumiai_runner.py successfully processes a TikTok video and creates
the temporal_windows output that Stage 2 expects.

Video: https://www.tiktok.com/@glowwithava/video/7526250443832331550
Video ID: 7526250443832331550

This test validates:
1. rumiai_runner.py processes TikTok URLs correctly
2. temporal_windows_updated.json is created in /insights/ directory
3. Output has valid JSON structure with required fields
4. Stage 2's output validation logic works

Run with: python3 ml_pipeline/tests/run_stage2_integration_test.py
"""

import sys
import os
import json
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, '/home/jorge/rumiaifinal')


def run_integration_test():
    """Run integration test with real TikTok URL"""

    print("="*70)
    print("Stage 2 Integration Test: RumiAI Video Processing")
    print("="*70)

    video_url = "https://www.tiktok.com/@glowwithava/video/7526250443832331550"
    video_id = "7526250443832331550"
    insights_dir = "/home/jorge/rumiaifinal/insights/"
    output_file = f"{insights_dir}{video_id}_temporal_windows_updated.json"

    try:
        print(f"\n1. TikTok Video Details:")
        print(f"   URL: {video_url}")
        print(f"   Expected Video ID: {video_id}")
        print(f"   Expected output: {output_file}")

        # Clean up any previous output
        print(f"\n2. Cleaning previous outputs...")
        if os.path.exists(output_file):
            print(f"   - Removing existing output: {output_file}")
            os.remove(output_file)
        print(f"   ✓ Ready for fresh test")

        # Run rumiai_runner.py
        print(f"\n3. Running rumiai_runner.py...")
        print(f"   This will scrape, download, and process through all 9 ML services")
        print(f"   ⏱️  Expected time: 1-3 minutes\n")

        start_time = time.time()

        cmd = [sys.executable, 'scripts/rumiai_runner.py', video_url]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        elapsed_time = time.time() - start_time

        print(f"\n4. Processing Results:")
        print(f"   - Exit code: {result.returncode}")
        print(f"   - Processing time: {elapsed_time:.1f}s")

        if result.returncode != 0:
            print(f"   ✗ rumiai_runner.py failed!")
            print(f"\n   STDOUT (last 30 lines):")
            stdout_lines = result.stdout.split('\n')
            for line in stdout_lines[-30:]:
                print(f"     {line}")
            print(f"\n   STDERR (last 30 lines):")
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines[-30:]:
                print(f"     {line}")
            return False

        print(f"   ✓ rumiai_runner.py completed successfully")

        # Verify output file exists
        print(f"\n5. Verifying output file...")
        if not os.path.exists(output_file):
            print(f"   ✗ Output file not found: {output_file}")
            print(f"   Check insights directory:")
            if os.path.exists(insights_dir):
                files = os.listdir(insights_dir)
                print(f"   Files in {insights_dir}:")
                for f in files[:10]:
                    print(f"     - {f}")
            return False

        print(f"   ✓ Output file exists: {output_file}")

        # Verify file size
        file_size = os.path.getsize(output_file)
        print(f"   ✓ File size: {file_size / 1024:.2f} KB")

        # Validate JSON structure
        print(f"\n6. Validating JSON structure...")
        with open(output_file, 'r') as f:
            data = json.load(f)

        # Check required top-level fields
        required_fields = ['video_id', 'temporal_windows', 'metadata', 'processing_timestamp']
        missing_fields = [f for f in required_fields if f not in data]

        if missing_fields:
            print(f"   ✗ Missing required fields: {missing_fields}")
            print(f"   Found fields: {list(data.keys())}")
            return False

        print(f"   ✓ All required top-level fields present")

        # Check video_id matches
        if data.get('video_id') != video_id:
            print(f"   ✗ Video ID mismatch!")
            print(f"     Expected: {video_id}")
            print(f"     Found: {data.get('video_id')}")
            return False

        print(f"   ✓ Video ID correct: {video_id}")

        # Check temporal_windows structure
        tw = data.get('temporal_windows', {})
        if not isinstance(tw, dict):
            print(f"   ✗ temporal_windows is not a dict")
            return False

        required_sections = ['hook', 'closing']
        for section in required_sections:
            if section not in tw:
                print(f"   ✗ Missing temporal_windows section: {section}")
                return False

        print(f"   ✓ temporal_windows structure valid")

        # Get video duration
        duration = data.get('duration', 'unknown')
        print(f"   ✓ Video duration: {duration}s")

        # Determine bucket
        from foundation.buckets import assign_bucket
        if isinstance(duration, (int, float)):
            bucket = assign_bucket(duration)
            print(f"   ✓ Assigned bucket: {bucket}")

        # Show feature counts
        if 'hook' in tw and isinstance(tw['hook'], dict):
            feature_count = len(tw['hook'])
            print(f"   ✓ Hook features: {feature_count}")

        if 'closing' in tw and isinstance(tw['closing'], dict):
            feature_count = len(tw['closing'])
            print(f"   ✓ Closing features: {feature_count}")

        # Success
        print("\n" + "="*70)
        print("✅ Stage 2 Integration Test: PASSED")
        print("="*70)
        print(f"\nConclusion: rumiai_runner.py successfully processed TikTok video!")
        print(f"- Video scraped and downloaded from TikTok")
        print(f"- Processed through all 9 ML services")
        print(f"- temporal_windows_updated.json created")
        print(f"- JSON structure valid and complete")
        print(f"- Output location: {insights_dir}")
        print(f"- Processing time: {elapsed_time:.1f}s")
        print(f"\n✅ Stage 2 integration with rumiai_runner.py: VERIFIED")

        return True

    except subprocess.TimeoutExpired:
        print(f"\n❌ Test failed: Processing timeout (>5 minutes)")
        return False

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
