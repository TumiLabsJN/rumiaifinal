#!/usr/bin/env python3
"""
Benchmark script to measure I/O performance improvement with RAM disk
"""

import tempfile
import time
import cv2
import numpy as np
import os
from pathlib import Path

def benchmark_io(directory=None, num_frames=60):
    """Benchmark writing and reading frames to a directory"""

    # Create dummy frames (640x480 RGB)
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
              for _ in range(num_frames)]

    temp_files = []

    # Benchmark WRITE
    write_start = time.time()
    for i, frame in enumerate(frames):
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f'_frame_{i}.jpg',
            delete=False,
            dir=directory
        )
        cv2.imwrite(temp_file.name, frame)
        temp_files.append(temp_file.name)
        temp_file.close()
    write_time = time.time() - write_start

    # Benchmark READ
    read_start = time.time()
    for temp_file in temp_files:
        _ = cv2.imread(temp_file)
    read_time = time.time() - read_start

    # Cleanup
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass

    total_time = write_time + read_time

    return {
        'write_time': write_time,
        'read_time': read_time,
        'total_time': total_time,
        'write_fps': num_frames / write_time,
        'read_fps': num_frames / read_time
    }

print("=" * 60)
print("FEAT I/O Performance Benchmark")
print("=" * 60)
print(f"\nTesting with 60 frames (typical for 120s video)...\n")

# Test 1: Regular disk (default temp directory)
print("1. Regular Disk (SSD/HDD):")
regular_results = benchmark_io(directory=None)
print(f"   Write: {regular_results['write_time']:.3f}s ({regular_results['write_fps']:.1f} FPS)")
print(f"   Read:  {regular_results['read_time']:.3f}s ({regular_results['read_fps']:.1f} FPS)")
print(f"   Total: {regular_results['total_time']:.3f}s")

# Test 2: /dev/shm (if available)
if os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK):
    print("\n2. RAM Disk (/dev/shm):")
    shm_dir = Path('/dev/shm/feat_test')
    shm_dir.mkdir(exist_ok=True)

    ram_results = benchmark_io(directory=str(shm_dir))
    print(f"   Write: {ram_results['write_time']:.3f}s ({ram_results['write_fps']:.1f} FPS)")
    print(f"   Read:  {ram_results['read_time']:.3f}s ({ram_results['read_fps']:.1f} FPS)")
    print(f"   Total: {ram_results['total_time']:.3f}s")

    # Calculate improvement
    speedup = regular_results['total_time'] / ram_results['total_time']
    time_saved = regular_results['total_time'] - ram_results['total_time']

    print(f"\n📊 Performance Improvement:")
    print(f"   Speedup: {speedup:.2f}x faster")
    print(f"   Time saved: {time_saved:.3f}s per 60 frames")
    print(f"   Percentage: {(1 - ram_results['total_time']/regular_results['total_time'])*100:.1f}% reduction")

    # Cleanup
    import shutil
    shutil.rmtree(shm_dir, ignore_errors=True)
else:
    print("\n⚠️  /dev/shm not available on this system")

print("\n" + "=" * 60)