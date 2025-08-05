#!/usr/bin/env python3
"""
Test suite for unified ML pipeline
Validates frame extraction, ML processing, and output formats
"""

import asyncio
import argparse
import json
import time
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_unified_pipeline(video_path: Path, output_dir: Path, video_id: str = None):
    """Test the complete unified ML pipeline"""
    
    # Import the unified services
    from rumiai_v2.api.ml_services import MLServices
    from rumiai_v2.processors.unified_frame_manager import get_frame_manager
    
    print("\n" + "="*60)
    print("UNIFIED ML PIPELINE TEST")
    print("="*60)
    
    # Initialize services
    ml_services = MLServices()
    frame_manager = get_frame_manager()
    
    # Generate video ID if not provided
    if not video_id:
        video_id = video_path.stem if video_path.stem.isdigit() else f"test_{int(time.time())}"
    
    # Track performance metrics
    start_time = time.time()
    
    print(f"\nTest video: {video_path}")
    print(f"Video ID: {video_id}")
    print(f"Output directory: {output_dir}")
    
    # Test 1: Frame extraction
    print("\n" + "-"*40)
    print("TEST 1: Frame Extraction")
    print("-"*40)
    
    frame_data = await frame_manager.extract_frames(video_path, video_id)
    
    if frame_data.get('success'):
        print(f"✅ Frames extracted: {len(frame_data['frames'])}")
        if frame_data.get('metadata'):
            print(f"   Video FPS: {frame_data['metadata'].fps:.2f}")
            print(f"   Duration: {frame_data['metadata'].duration:.2f}s")
        print(f"   Cache hit: {frame_data.get('cache_hit', False)}")
        print(f"   Degraded: {frame_data.get('degraded', False)}")
    else:
        print(f"❌ Frame extraction failed: {frame_data.get('error')}")
        return False
    
    # Test 2: Frame caching
    print("\n" + "-"*40)
    print("TEST 2: Frame Caching")
    print("-"*40)
    
    frame_data2 = await frame_manager.extract_frames(video_path, video_id)
    print(f"✅ Cache working: {frame_data2.get('cache_hit', False)}")
    
    # Test 3: Individual ML services
    print("\n" + "-"*40)
    print("TEST 3: Individual ML Services")
    print("-"*40)
    
    # Test that individual services run ONLY their service
    print("\nTesting individual service methods...")
    
    # Test YOLO only
    yolo_start = time.time()
    yolo_result = await ml_services.run_yolo_detection(video_path, output_dir)
    yolo_time = time.time() - yolo_start
    print(f"YOLO: {len(yolo_result.get('objectAnnotations', []))} objects in {yolo_time:.2f}s")
    
    # Test MediaPipe only
    mp_start = time.time()
    mp_result = await ml_services.run_mediapipe_analysis(video_path, output_dir)
    mp_time = time.time() - mp_start
    print(f"MediaPipe: {len(mp_result.get('poses', []))} poses in {mp_time:.2f}s")
    
    # Test OCR only
    ocr_start = time.time()
    ocr_result = await ml_services.run_ocr_analysis(video_path, output_dir)
    ocr_time = time.time() - ocr_start
    print(f"OCR: {len(ocr_result.get('textAnnotations', []))} texts in {ocr_time:.2f}s")
    
    # Test Whisper only
    whisper_start = time.time()
    whisper_result = await ml_services.run_whisper_transcription(video_path, output_dir)
    whisper_time = time.time() - whisper_start
    print(f"Whisper: {len(whisper_result.get('text', ''))} chars in {whisper_time:.2f}s")
    
    # Test 4: All ML services at once
    print("\n" + "-"*40)
    print("TEST 4: All ML Services (Parallel)")
    print("-"*40)
    
    ml_start = time.time()
    all_results = await ml_services.run_all_ml_services(video_path, output_dir)
    ml_duration = time.time() - ml_start
    
    print(f"\n✅ All ML services completed in {ml_duration:.2f}s")
    
    # Check service results
    services_status = []
    
    for service, key in [('YOLO', 'yolo'), ('MediaPipe', 'mediapipe'), 
                         ('OCR', 'ocr'), ('Whisper', 'whisper')]:
        data = all_results.get(key, {})
        processed = data.get('metadata', {}).get('processed', False)
        
        if service == 'YOLO':
            count = len(data.get('objectAnnotations', []))
            detail = f"{count} objects"
        elif service == 'MediaPipe':
            count = len(data.get('poses', []))
            detail = f"{count} poses"
        elif service == 'OCR':
            count = len(data.get('textAnnotations', []))
            detail = f"{count} texts"
        else:  # Whisper
            count = len(data.get('text', ''))
            detail = f"{count} chars"
            
        services_status.append((service, processed, detail))
        status = "✅" if processed else "❌"
        print(f"  {status} {service}: {detail}")
        
    # Test 5: Output files
    print("\n" + "-"*40)
    print("TEST 5: Output Files")
    print("-"*40)
    
    expected_files = [
        f"{video_id}_yolo_detections.json",
        f"{video_id}_human_analysis.json",
        f"{video_id}_creative_analysis.json",
        f"{video_id}_whisper.json"
    ]
    
    for filename in expected_files:
        file_path = output_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size / 1024  # KB
            print(f"  ✅ {filename} ({size:.1f} KB)")
        else:
            print(f"  ❌ {filename} (missing)")
            
    # Performance summary
    print("\n" + "-"*40)
    print("PERFORMANCE SUMMARY")
    print("-"*40)
    
    total_duration = time.time() - start_time
    
    print(f"Total time: {total_duration:.2f}s")
    
    # Cleanup
    print("\n" + "-"*40)
    print("CLEANUP")
    print("-"*40)
    
    await ml_services.cleanup()
    await frame_manager.cleanup()
    
    print("✅ Resources cleaned up")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # Return success based on all services processing
    all_processed = all(status[1] for status in services_status)
    return all_processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Unified ML Pipeline')
    parser.add_argument('video', type=Path, help='Path to test video file')
    parser.add_argument('--output', type=Path, default=Path('/tmp/ml_test'),
                       help='Output directory (default: /tmp/ml_test)')
    parser.add_argument('--video-id', type=str, 
                       help='Video ID for caching (auto-generated if not provided)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate video exists
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Run test
    success = asyncio.run(test_unified_pipeline(args.video, args.output, args.video_id))
    exit(0 if success else 1)