#!/usr/bin/env python3
"""Test parallel execution mode."""

import os
import sys
import asyncio
import time
from pathlib import Path

sys.path.insert(0, '/home/jorge/rumiaifinal')

async def test_parallel():
    """Test in parallel mode."""

    video_path = Path('/home/jorge/rumiaifinal/temp/6923023955813092613.mp4')

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    print(f"Testing with 120s video: {video_path}")
    print(f"File size: {video_path.stat().st_size / (1024*1024):.1f} MB")

    print("\n" + "="*60)
    print("TESTING PARALLEL MODE (Default)")
    print("="*60)

    os.environ['SEQUENTIAL_TEST'] = 'false'
    os.environ['METRICS_DISPLAY'] = 'summary'

    from rumiai_v2.processors.video_analyzer import VideoAnalyzer
    from rumiai_v2.api.ml_services import MLServices

    ml_services = MLServices()
    analyzer = VideoAnalyzer(ml_services)

    video_id = video_path.stem
    video_metadata = {
        'filename': video_path.name,
        'path': str(video_path),
        'size_mb': video_path.stat().st_size / (1024*1024)
    }

    print("Starting analysis...")
    start_time = time.time()

    try:
        results = await analyzer.analyze_video(
            video_id=video_id,
            video_path=video_path
        )

        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f}s")

        print("\nServices completed:")
        for service_name, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"  {status} {service_name}: {result.processing_time:.2f}s")
            if not result.success and result.error:
                print(f"      Error: {result.error[:100]}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nCheck instrumentation_metrics/ for detailed reports")

if __name__ == "__main__":
    asyncio.run(test_parallel())