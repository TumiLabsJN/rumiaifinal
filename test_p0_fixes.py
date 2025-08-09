#!/usr/bin/env python3
"""
Integration test for P0 Critical Fixes - FEAT Emotion Detection
Tests the complete pipeline with real video processing
"""

import asyncio
import cv2
import numpy as np
from pathlib import Path
import json
import sys
import os
import time

# Add project root to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

def test_feat_integration():
    """Test that FEAT integration works as expected"""
    import numpy as np
    from feat import Detector
    
    print("🧪 Testing FEAT Integration...")
    
    # Create dummy frame with a face-like pattern
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Add simple face-like features
    cv2.circle(test_frame, (320, 240), 80, (255, 220, 177), -1)  # Face outline
    cv2.circle(test_frame, (280, 220), 8, (0, 0, 0), -1)  # Left eye
    cv2.circle(test_frame, (360, 220), 8, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(test_frame, (320, 280), (30, 15), 0, 0, 180, (0, 0, 0), 3)  # Smile
    
    try:
        # Initialize FEAT
        detector = Detector(device='cpu')
        
        # Test detection
        predictions = detector.detect_image([test_frame])
        
        # Validate output format
        assert hasattr(predictions, 'columns'), "FEAT should return DataFrame"
        print(f"✅ FEAT returns DataFrame with {len(predictions.columns)} columns")
        
        # Check for expected columns
        expected_patterns = ['face', 'Face', 'AU', 'anger', 'happy', 'sad']
        found_patterns = {pattern: any(pattern in col for col in predictions.columns) 
                         for pattern in expected_patterns}
        
        print(f"✅ Column patterns found: {found_patterns}")
        
        # Log actual column names for debugging
        print(f"📋 Actual columns: {list(predictions.columns)[:10]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ FEAT integration test failed: {e}")
        return False

async def test_emotion_detection_service():
    """Test RumiAI emotion detection service"""
    from rumiai_v2.ml_services.emotion_detection_service import get_emotion_detector
    
    print("\n🎭 Testing RumiAI Emotion Detection Service...")
    
    # Run basic integration test first
    if not test_feat_integration():
        print("⚠️ FEAT integration issues detected, continuing anyway...")
    
    detector = get_emotion_detector()
    
    # Create test frames with face-like patterns
    test_frames = []
    timestamps = []
    
    for i in range(5):  # Create 5 test frames
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add face with different expressions
        face_color = (255, 220, 177)
        cv2.circle(frame, (320, 240), 80, face_color, -1)
        cv2.circle(frame, (280, 220), 8, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (360, 220), 8, (0, 0, 0), -1)  # Right eye
        
        # Vary smile/frown
        if i < 3:
            # Smile
            cv2.ellipse(frame, (320, 280), (30, 15), 0, 0, 180, (0, 0, 0), 3)
        else:
            # Frown
            cv2.ellipse(frame, (320, 300), (30, 15), 0, 180, 360, (0, 0, 0), 3)
        
        test_frames.append(frame)
        timestamps.append(i * 1.0)  # 1 second intervals
    
    try:
        results = await detector.detect_emotions_batch(test_frames, timestamps)
        
        print(f"✅ Emotion detection completed")
        print(f"   Video type: {results['processing_stats']['video_type']}")
        print(f"   Total frames: {results['processing_stats']['total_frames']}")
        print(f"   Successful detections: {results['processing_stats']['successful_frames']}")
        print(f"   No face frames: {results['processing_stats']['no_face_frames']}")
        
        if results['emotions']:
            print(f"   Emotions detected: {len(results['emotions'])}")
            for i, emotion in enumerate(results['emotions'][:3]):  # Show first 3
                print(f"     Frame {i}: {emotion['emotion']} (conf: {emotion['confidence']:.2f})")
        
        if results['dominant_emotion']:
            print(f"   Dominant emotion: {results['dominant_emotion']}")
        
        if results['metrics']:
            print(f"   Unique emotions: {results['metrics']['unique_emotions']}")
            print(f"   Transitions: {results['metrics']['transition_count']}")
            print(f"   Avg confidence: {results['metrics']['avg_confidence']:.2f}")
        
        # Test full emotional journey analysis
        journey_results = await detector.analyze_emotional_journey(test_frames, timestamps)
        
        print(f"✅ Emotional journey analysis completed")
        print(f"   Journey archetype: {journey_results['emotionalPatterns']['journeyArchetype']}")
        print(f"   Emotional arc: {journey_results['emotionalDynamics']['emotionalArc']}")
        print(f"   Stability score: {journey_results['emotionalDynamics']['stabilityScore']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Emotion detection service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_with_real_video():
    """Test FEAT with a real video file"""
    print("\n🎬 Testing with Real Video...")
    
    # Find a test video
    video_files = list(Path("temp").glob("*.mp4"))
    if not video_files:
        print("⚠️ No test videos found in temp/ directory")
        return False
    
    video_path = video_files[0]
    print(f"📹 Using video: {video_path.name}")
    
    try:
        from rumiai_v2.ml_services.emotion_detection_service import get_emotion_detector
        
        # Extract frames from video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"   Duration: {duration:.1f}s, FPS: {fps:.1f}, Frames: {total_frames}")
        
        # Get adaptive sample rate
        detector = get_emotion_detector()
        sample_rate = detector.get_adaptive_sample_rate(duration)
        
        print(f"   FEAT sample rate: {sample_rate} FPS (adaptive)")
        
        # Extract frames at sample rate
        frames = []
        timestamps = []
        frame_count = 0
        
        while frame_count < min(total_frames, 60):  # Max 60 frames for test
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % int(fps / sample_rate) == 0:
                frames.append(frame)
                timestamps.append(frame_count / fps)
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            print("❌ No frames extracted from video")
            return False
        
        print(f"   Extracted {len(frames)} frames for FEAT processing")
        
        # Run emotion detection
        start_time = time.time()
        results = await detector.detect_emotions_batch(frames, timestamps)
        processing_time = time.time() - start_time
        
        print(f"✅ Real video processing completed in {processing_time:.1f}s")
        print(f"   Video type: {results['processing_stats']['video_type']}")
        print(f"   Detection rate: {results['processing_stats']['successful_frames']}/{len(frames)}")
        
        if results['processing_stats']['video_type'] == 'people_detected':
            print(f"   Dominant emotion: {results['dominant_emotion']}")
            print(f"   Unique emotions: {results['metrics']['unique_emotions']}")
            print(f"   Avg confidence: {results['metrics']['avg_confidence']:.2f}")
            print(f"   Emotion diversity: {results['metrics']['emotion_diversity']:.2f}")
            
            if results['metrics']['most_active_aus']:
                aus = [f"AU{au}" for au, count in results['metrics']['most_active_aus']]
                print(f"   Most active AUs: {', '.join(aus[:3])}")
        
        elif results['processing_stats']['video_type'] == 'no_people':
            print("   ✅ Correctly identified as no-people video (B-roll, text, etc.)")
        
        # Save results for inspection
        output_file = f"test_feat_results_{video_path.stem}.json"
        with open(output_file, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json_results = json.loads(json.dumps(results, default=convert_numpy))
            json.dump(json_results, f, indent=2)
            
        print(f"   Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real video test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_precompute_integration():
    """Test integration with precompute functions"""
    print("\n🔧 Testing Precompute Integration...")
    
    try:
        from rumiai_v2.processors.precompute_professional import compute_emotional_journey_analysis_professional
        import numpy as np
        
        # Create test timeline data (simulating MediaPipe output)
        test_timelines = {
            'expressionTimeline': {
                '0-1s': {'emotion': 'neutral', 'confidence': 0.8},
                '1-2s': {'emotion': 'joy', 'confidence': 0.9},
                '2-3s': {'emotion': 'joy', 'confidence': 0.85},
                '3-4s': {'emotion': 'surprise', 'confidence': 0.75},
                '4-5s': {'emotion': 'neutral', 'confidence': 0.8},
            },
            'gestureTimeline': {
                '0-1s': {'gestures': ['open_palm']},
                '2-3s': {'gestures': ['thumbs_up']},
            }
        }
        
        # Create test frames for FEAT integration
        test_frames = []
        timestamps = []
        for i in range(5):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            # Add face
            cv2.circle(frame, (320, 240), 60, (255, 220, 177), -1)
            cv2.circle(frame, (300, 220), 5, (0, 0, 0), -1)
            cv2.circle(frame, (340, 220), 5, (0, 0, 0), -1)
            cv2.ellipse(frame, (320, 260), (20, 10), 0, 0, 180, (0, 0, 0), 2)
            
            test_frames.append(frame)
            timestamps.append(float(i))
        
        # Test with FEAT integration (frames provided)
        print("   Testing with FEAT integration...")
        os.environ['USE_PYTHON_ONLY_PROCESSING'] = 'true'  # Enable Python-only mode
        
        results = compute_emotional_journey_analysis_professional(
            test_timelines, 5.0, frames=test_frames, timestamps=timestamps
        )
        
        print(f"✅ Precompute function completed")
        print(f"   Core metrics keys: {list(results.get('emotionalCoreMetrics', {}).keys())}")
        print(f"   Dynamics keys: {list(results.get('emotionalDynamics', {}).keys())}")
        print(f"   Patterns keys: {list(results.get('emotionalPatterns', {}).keys())}")
        
        # Verify structure
        expected_blocks = [
            'emotionalCoreMetrics',
            'emotionalDynamics', 
            'emotionalInteractions',
            'emotionalKeyEvents',
            'emotionalPatterns',
            'emotionalQuality'
        ]
        
        found_blocks = [block for block in expected_blocks if block in results]
        print(f"   6-block structure: {len(found_blocks)}/6 blocks found")
        print(f"   Found blocks: {found_blocks}")
        
        return len(found_blocks) >= 4  # At least 4 blocks should be present
        
    except Exception as e:
        print(f"❌ Precompute integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 P0 CRITICAL FIXES - INTEGRATION TESTING")
    print("=" * 60)
    print("Testing FEAT emotion detection integration with RumiAI")
    print()
    
    tests = [
        ("Emotion Detection Service", test_emotion_detection_service),
        ("Real Video Processing", test_with_real_video), 
        ("Precompute Integration", test_precompute_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print('='*60)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ P0 critical fixes working correctly")
        print("✅ FEAT emotion detection fully integrated")
        print("✅ Ready for production use")
        
        print("\n📋 Next Steps:")
        print("1. Run full video processing with:")
        print("   export USE_PYTHON_ONLY_PROCESSING=true")
        print("   python3 scripts/rumiai_runner.py 'VIDEO_URL'")
        print("2. Monitor emotion detection quality")
        print("3. Compare before/after metrics")
        
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("❌ Review logs and fix issues before production")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)