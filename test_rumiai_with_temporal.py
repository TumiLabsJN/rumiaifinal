#!/usr/bin/env python3
"""
Test script to verify temporal_windows integration with rumiai_runner.py
"""
import json
import sys
from pathlib import Path
import subprocess
import time

def test_temporal_integration():
    """Test the temporal_windows integration with a real video"""
    
    print("=" * 60)
    print("TESTING TEMPORAL WINDOWS INTEGRATION")
    print("=" * 60)
    
    # Check if we have any test videos available
    test_videos = [
        "temp/7015376025727143174.mp4",
        "temp/7155207990998584582.mp4",
        "data/raw/7409647328226839850.mp4",
        "data/raw/test_video.mp4",
        "test_data/sample_video.mp4"
    ]
    
    video_path = None
    for path in test_videos:
        if Path(path).exists():
            video_path = path
            print(f"✅ Found test video: {video_path}")
            break
    
    if not video_path:
        print("❌ No test video found. Please provide a test video.")
        print("   Expected locations:")
        for path in test_videos:
            print(f"   - {path}")
        return False
    
    # Extract video ID from path
    video_id = Path(video_path).stem
    print(f"📹 Video ID: {video_id}")
    
    # Run rumiai_runner with the test video
    print("\n" + "=" * 60)
    print("RUNNING RUMIAI WITH TEMPORAL COMPUTE")
    print("=" * 60)
    
    cmd = [
        "python3", "scripts/rumiai_runner.py",
        "--video-path", video_path,
        "--use-python-only"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ RumiAI failed with code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    print(f"✅ RumiAI completed in {duration:.1f}s")
    
    # Check for temporal_windows output
    print("\n" + "=" * 60)
    print("CHECKING TEMPORAL WINDOWS OUTPUT")
    print("=" * 60)
    
    # Check different possible output locations
    output_paths = [
        f"data/insights/{video_id}_temporal_windows.json",
        f"data/processed/{video_id}/temporal_windows.json",
        f"output/{video_id}_temporal_windows.json"
    ]
    
    temporal_output = None
    for path in output_paths:
        if Path(path).exists():
            temporal_output = path
            print(f"✅ Found temporal windows output: {path}")
            break
    
    if not temporal_output:
        print("❌ No temporal_windows.json output found")
        print("   Checked locations:")
        for path in output_paths:
            print(f"   - {path}")
        
        # List actual insight files
        insights_dir = Path("data/insights")
        if insights_dir.exists():
            print(f"\n   Files in {insights_dir}:")
            for file in insights_dir.glob("*.json"):
                print(f"   - {file.name}")
        return False
    
    # Load and analyze the output
    with open(temporal_output, 'r') as f:
        temporal_data = json.load(f)
    
    print("\n" + "-" * 60)
    print("TEMPORAL WINDOWS STRUCTURE:")
    print("-" * 60)
    
    # Check required fields
    required_fields = ['video_id', 'duration', 'temporal_windows', 'metadata', 'processing_timestamp', 'version']
    for field in required_fields:
        if field in temporal_data:
            print(f"✅ {field}: present")
        else:
            print(f"❌ {field}: MISSING")
    
    # Analyze temporal windows
    if 'temporal_windows' in temporal_data:
        tw = temporal_data['temporal_windows']
        print("\n" + "-" * 60)
        print("TEMPORAL WINDOWS BREAKDOWN:")
        print("-" * 60)
        
        if 'hook' in tw and tw['hook']:
            hook = tw['hook']
            print(f"\n🎬 HOOK: {hook.get('start', 0):.1f}s - {hook.get('end', 0):.1f}s")
            print(f"   Elements: {hook.get('element_count', 0)}")
            print(f"   Speech coverage: {hook.get('speech_coverage', 0)*100:.1f}%")
            print(f"   Energy: {hook.get('energy_level', 0):.3f}")
            print(f"   Burst pattern: {hook.get('burst_pattern', 'unknown')}")
        
        if 'middle_segments' in tw:
            segments = tw['middle_segments']
            print(f"\n📊 MIDDLE SEGMENTS: {len(segments)}")
            for i, seg in enumerate(segments, 1):
                print(f"   Segment {i}: {seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s")
                print(f"     Elements: {seg.get('element_count', 0)}")
                print(f"     Density: {seg.get('avg_density', 0):.2f}")
        
        if 'closing' in tw and tw['closing']:
            closing = tw['closing']
            print(f"\n🎭 CLOSING: {closing.get('start', 0):.1f}s - {closing.get('end', 0):.1f}s")
            print(f"   Elements: {closing.get('element_count', 0)}")
            print(f"   Speech coverage: {closing.get('speech_coverage', 0)*100:.1f}%")
            print(f"   Energy: {closing.get('energy_level', 0):.3f}")
    
    # Check metadata
    if 'metadata' in temporal_data:
        meta = temporal_data['metadata']
        print("\n" + "-" * 60)
        print("VIDEO METADATA:")
        print("-" * 60)
        print(f"Duration: {meta.get('duration', 0):.1f}s")
        print(f"Play count: {meta.get('play_count', 0):,}")
        print(f"Digg count: {meta.get('digg_count', 0):,}")
        print(f"Author: {meta.get('author', 'unknown')}")
    
    print("\n" + "=" * 60)
    print("✨ TEMPORAL WINDOWS INTEGRATION TEST COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_temporal_integration()
    sys.exit(0 if success else 1)