#!/usr/bin/env python3
"""Comprehensive vision services performance test for different video durations"""

import asyncio
import time
import json
from pathlib import Path
import tracemalloc
import sys
from rumiai_v2.api.ml_services_unified import UnifiedMLServices
from rumiai_v2.processors.unified_frame_manager import UnifiedFrameManager

async def test_single_video(video_path: Path, video_name: str):
    """Test all vision services on a single video"""

    video_id = f"test_{video_name}"
    output_dir = Path(f"/tmp/vision_test_{video_name}")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize services
    ml_services = UnifiedMLServices()
    frame_manager = UnifiedFrameManager()

    results = {
        'video': video_name,
        'video_path': str(video_path),
        'services': {}
    }

    # Extract frames once
    print(f"\n{'='*60}")
    print(f"Testing: {video_name}")
    print('='*60)

    print("Extracting frames...")
    start = time.time()

    try:
        frame_data = await frame_manager.extract_frames(video_path, video_id)
        frames = frame_data['frames']
        metadata = frame_data['metadata']

        frame_time = time.time() - start

        print(f"✓ Frame extraction: {frame_time:.2f}s")
        print(f"  - Total frames: {len(frames)}")
        print(f"  - Video duration: {metadata.duration:.2f}s")
        print(f"  - FPS: {metadata.fps:.2f}")

        results['duration'] = metadata.duration
        results['total_frames'] = len(frames)
        results['extraction_time'] = frame_time

        # Test each vision service
        services = {
            'yolo': ml_services._run_yolo_on_frames,
            'mediapipe': ml_services._run_mediapipe_on_frames,
            'ocr': ml_services._run_ocr_on_frames
        }

        for service_name, service_func in services.items():
            print(f"\nTesting {service_name.upper()}...")

            # Get service-specific frames
            service_frames = frame_manager.get_frames_for_service(frames, service_name)
            print(f"  - Frames for {service_name}: {len(service_frames)}")

            # Measure performance
            tracemalloc.start()
            start = time.time()

            try:
                result = await service_func(frames, video_id, output_dir)
                elapsed = time.time() - start
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                print(f"  ✓ Processing time: {elapsed:.2f}s")
                print(f"  ✓ Peak memory: {peak/1024/1024:.2f}MB")
                print(f"  ✓ Speed: {metadata.duration/elapsed:.2f}x realtime")

                # Store results
                results['services'][service_name] = {
                    'frames_processed': len(service_frames),
                    'processing_time': elapsed,
                    'peak_memory_mb': peak/1024/1024,
                    'realtime_factor': metadata.duration/elapsed
                }

                # Add service-specific metrics
                if isinstance(result, dict) and 'metadata' in result:
                    results['services'][service_name]['metadata'] = result['metadata']

            except Exception as e:
                print(f"  ✗ Error: {e}")
                tracemalloc.stop()
                results['services'][service_name] = {'error': str(e)}

    except Exception as e:
        print(f"✗ Failed to extract frames: {e}")
        results['error'] = str(e)

    return results

async def main():
    """Test multiple videos of different durations"""

    test_videos = [
        (Path("/home/jorge/rumiaifinal/temp/7015376025727143174.mp4"), "16s_video"),
        (Path("/home/jorge/rumiaifinal/temp/7515687288257465630.mp4"), "44s_video"),
        (Path("/home/jorge/rumiaifinal/temp/7274651255392210219.mp4"), "59s_video")
    ]

    all_results = []

    for video_path, video_name in test_videos:
        if video_path.exists():
            results = await test_single_video(video_path, video_name)
            all_results.append(results)
        else:
            print(f"⚠️ Video not found: {video_path}")

    # Save comprehensive results
    output_file = Path("/home/jorge/rumiaifinal/vision_performance_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    # Print summary table
    print(f"\n{'Video':<15} {'Duration':<10} {'Service':<12} {'Time(s)':<10} {'Frames':<10} {'Speed':<10}")
    print("-"*77)

    for result in all_results:
        if 'error' not in result:
            duration = f"{result['duration']:.1f}s"
            for service, data in result['services'].items():
                if 'error' not in data:
                    time_str = f"{data['processing_time']:.2f}"
                    frames = data['frames_processed']
                    speed = f"{data['realtime_factor']:.2f}x"
                    print(f"{result['video']:<15} {duration:<10} {service:<12} {time_str:<10} {frames:<10} {speed:<10}")

    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())