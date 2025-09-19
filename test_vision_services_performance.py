#!/usr/bin/env python3
"""Test vision services performance for documentation"""

import asyncio
import time
import json
from pathlib import Path
import tracemalloc
from rumiai_v2.api.ml_services_unified import UnifiedMLServices
from rumiai_v2.processors.unified_frame_manager import UnifiedFrameManager

async def test_vision_services():
    """Test individual vision service performance"""

    # Use an actual video from temp
    video_path = Path("/home/jorge/rumiaifinal/temp/7015376025727143174.mp4")
    if not video_path.exists():
        print("Test video not found")
        return

    video_id = "test_video"
    output_dir = Path("/tmp/vision_test")
    output_dir.mkdir(exist_ok=True)

    # Initialize services
    ml_services = UnifiedMLServices()
    frame_manager = UnifiedFrameManager()

    print("\n=== Vision Services Performance Test ===\n")

    # Extract frames once
    print("Extracting frames...")
    start = time.time()
    tracemalloc.start()

    frame_data = await frame_manager.extract_frames(video_path, video_id)
    frames = frame_data['frames']
    metadata = frame_data['metadata']

    frame_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Frame extraction: {frame_time:.2f}s")
    print(f"Total frames: {len(frames)}")
    print(f"Video duration: {metadata.duration:.2f}s")
    print(f"Peak memory: {peak/1024/1024:.2f}MB")

    # Test each vision service
    services = ['yolo', 'mediapipe', 'ocr', 'scene']

    for service in services:
        print(f"\n--- Testing {service.upper()} ---")

        # Get service-specific frames
        service_frames = frame_manager.get_frames_for_service(frames, service)
        print(f"Frames for {service}: {len(service_frames)}")

        # Measure performance
        tracemalloc.start()
        start = time.time()

        try:
            if service == 'yolo':
                result = await ml_services._run_yolo_on_frames(frames, video_id, output_dir)
            elif service == 'mediapipe':
                result = await ml_services._run_mediapipe_on_frames(frames, video_id, output_dir)
            elif service == 'ocr':
                result = await ml_services._run_ocr_on_frames(frames, video_id, output_dir)
            elif service == 'scene':
                result = await ml_services._run_scene_detection(video_path, video_id, output_dir)

            elapsed = time.time() - start
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"Processing time: {elapsed:.2f}s")
            print(f"Peak memory: {peak/1024/1024:.2f}MB")

            # Check output
            if isinstance(result, dict):
                if 'metadata' in result:
                    print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
                elif service == 'scene' and 'scene_changes' in result:
                    print(f"Scenes detected: {len(result['scene_changes'])}")

        except Exception as e:
            print(f"Error testing {service}: {e}")
            tracemalloc.stop()

if __name__ == "__main__":
    asyncio.run(test_vision_services())