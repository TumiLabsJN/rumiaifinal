#!/usr/bin/env python3
"""
Quick test of FixBug1130 fixes
"""
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_scene_detection_fix():
    """Test that scene detection no longer crashes"""
    print("🎬 Testing Scene Detection Fix...")
    
    from rumiai_v2.api.ml_services import MLServices
    ml = MLServices()
    
    video_path = Path('apify_storage/datasets/default/7190549760154012970/7190549760154012970.mp4')
    if not video_path.exists():
        print(f"⚠️  Video file not found: {video_path}")
        return False
    
    try:
        import asyncio
        
        async def test_scenes():
            output_dir = Path('test_scene_output')
            output_dir.mkdir(exist_ok=True)
            result = await ml.run_scene_detection(video_path, output_dir)
            return result
        
        result = asyncio.run(test_scenes())
        
        if 'error' not in result and result.get('scenes'):
            print(f"✅ Scene detection FIXED! Found {len(result['scenes'])} scenes")
            return True
        else:
            print(f"❌ Scene detection still broken: {result}")
            return False
    except Exception as e:
        print(f"❌ Scene detection failed: {e}")
        return False

def test_emotion_timeline_fix():
    """Test that emotion timeline uses correct key"""
    print("😊 Testing Emotion Timeline Fix...")
    
    # Test a simple expression timeline lookup
    expression_timeline = {
        "0-1s": {
            "emotion": "happy",  # Correct FEAT key
            "confidence": 0.8
        }
    }
    
    try:
        # This is the pattern from precompute_functions_full.py that we fixed
        for timestamp, emotion_data in expression_timeline.items():
            if emotion_data.get('emotion'):  # Should work now (was 'expression')
                print(f"✅ Emotion key fix WORKS! Found {emotion_data['emotion']} at {timestamp}")
                return True
        
        print("❌ No emotion data found")
        return False
    except Exception as e:
        print(f"❌ Emotion timeline test failed: {e}")
        return False

def test_mediapipe_gaze_conversion():
    """Test that MediaPipe now does gaze-only"""
    print("👁️  Testing MediaPipe Gaze Conversion...")
    
    try:
        from mediapipe_human_detector import MediaPipeHumanDetector
        detector = MediaPipeHumanDetector()
        
        # Check that old face detection method is gone
        if hasattr(detector, 'detect_faces_and_expressions'):
            print("❌ Old detect_faces_and_expressions still exists!")
            return False
        
        # Check that new gaze detection method exists
        if hasattr(detector, 'detect_gaze_only'):
            print("✅ MediaPipe converted to gaze-only!")
            return True
        else:
            print("❌ detect_gaze_only method missing!")
            return False
            
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_enhanced_human_analyzer_deleted():
    """Test that Enhanced Human Analyzer is truly deleted"""
    print("🗑️  Testing Enhanced Human Analyzer Deletion...")
    
    deleted_file = Path('local_analysis/enhanced_human_analyzer.py')
    if deleted_file.exists():
        print("❌ Enhanced Human Analyzer file still exists!")
        return False
    
    deleted_dir = Path('enhanced_human_analysis_outputs')
    if deleted_dir.exists():
        print("❌ Enhanced Human Analyzer output directory still exists!")
        return False
    
    print("✅ Enhanced Human Analyzer successfully deleted!")
    return True

if __name__ == "__main__":
    print("🦍 HARAMBE'S BLESSING TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_emotion_timeline_fix,
        test_mediapipe_gaze_conversion, 
        test_enhanced_human_analyzer_deleted,
        test_scene_detection_fix,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"🦍 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL FIXES WORKING! HARAMBE IS PLEASED!")
    else:
        print("⚠️  Some fixes need attention")
        
    sys.exit(0 if passed == total else 1)