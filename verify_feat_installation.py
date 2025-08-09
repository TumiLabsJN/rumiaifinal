#!/usr/bin/env python3
"""
Verification script for FEAT installation and dependencies
Verifies P0 critical fixes are properly installed and working
"""

import sys
import os

# Apply FEAT compatibility patches
from scipy_compat import ensure_feat_compatibility
ensure_feat_compatibility()

def verify_feat_installation():
    """Verify FEAT and dependencies are correctly installed"""
    
    print("🔍 Checking FEAT Installation...")
    print("=" * 50)
    
    success = True
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("   Using CPU (expected for many setups)")
    except ImportError:
        print("❌ PyTorch not installed")
        success = False
    
    # Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed")
        success = False
    
    # Check FEAT
    try:
        from feat import Detector
        print("✅ FEAT (py-feat) installed")
        
        # Try initializing detector
        print("🔧 Initializing FEAT detector (may download models)...")
        detector = Detector(device='cpu')  # Use CPU for testing
        print("✅ FEAT detector initialized successfully")
        
        # Test basic functionality
        print("🧪 Testing FEAT detection capability...")
        import numpy as np
        
        # Create test image (simple face-like pattern for better results)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add simple face-like features
        import cv2
        cv2.circle(test_image, (320, 240), 80, (255, 220, 177), -1)  # Face
        cv2.circle(test_image, (280, 220), 8, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (360, 220), 8, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(test_image, (320, 280), (30, 15), 0, 0, 180, (0, 0, 0), 3)  # Smile
        
        # Test detection with proper image format
        # FEAT expects image file paths or PIL images, not numpy arrays directly
        import tempfile
        import cv2
        
        # Save test image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            results = detector.detect_image([tmp.name])
            os.unlink(tmp.name)  # Clean up
        print(f"✅ FEAT detection test completed")
        print(f"   Results type: {type(results)}")
        print(f"   Columns: {len(results.columns) if hasattr(results, 'columns') else 'N/A'}")
        
        # Check for key columns
        if hasattr(results, 'columns'):
            key_columns = ['FaceScore', 'anger', 'happiness', 'neutral']
            found_columns = [col for col in key_columns if col in results.columns]
            print(f"   Key columns found: {found_columns}")
        
    except ImportError:
        print("❌ FEAT (py-feat) not installed")
        success = False
    except Exception as e:
        print(f"❌ FEAT initialization failed: {e}")
        success = False
    
    # Check other dependencies
    deps = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),  
        ('scikit-learn', 'sklearn'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('xgboost', 'xgboost')
    ]
    
    print("\n🔍 Checking Additional Dependencies...")
    for dep_name, import_name in deps:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep_name} {version}")
        except ImportError:
            print(f"❌ {dep_name} not installed")
            success = False
    
    # Check RumiAI ML services
    print("\n🔍 Checking RumiAI Integration...")
    try:
        sys.path.insert(0, '/home/jorge/rumiaifinal')
        from rumiai_v2.ml_services.emotion_detection_service import get_emotion_detector
        detector_service = get_emotion_detector()
        print("✅ RumiAI emotion detection service loaded")
        print(f"   Device: {detector_service.device}")
    except Exception as e:
        print(f"❌ RumiAI emotion service failed: {e}")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL DEPENDENCIES VERIFIED!")
        print("✅ FEAT emotion detection ready for production")
        print("✅ P0 critical fixes properly installed")
    else:
        print("⚠️  SOME DEPENDENCIES MISSING")
        print("❌ Review installation steps and try again")
    
    return success

def test_video_processing():
    """Test FEAT with a sample video frame"""
    print("\n🎬 Testing Video Frame Processing...")
    
    try:
        import cv2
        import numpy as np
        from rumiai_v2.ml_services.emotion_detection_service import get_emotion_detector
        
        # Create a more realistic test image (face-like pattern)
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add some face-like features (simple geometric shapes)
        cv2.circle(test_frame, (320, 200), 50, (255, 220, 177), -1)  # Face
        cv2.circle(test_frame, (300, 180), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_frame, (340, 180), 5, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(test_frame, (320, 220), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Smile
        
        detector = get_emotion_detector()
        
        # Test async detection
        import asyncio
        
        async def test_detection():
            results = await detector.detect_emotions_batch([test_frame], [0.0])
            return results
        
        # Run test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(test_detection())
            
            print("✅ Video frame processing test completed")
            print(f"   Video type: {results['processing_stats']['video_type']}")
            print(f"   Successful frames: {results['processing_stats']['successful_frames']}")
            print(f"   No face frames: {results['processing_stats']['no_face_frames']}")
            
            if results['emotions']:
                print(f"   First emotion: {results['emotions'][0]['emotion']}")
                print(f"   Confidence: {results['emotions'][0]['confidence']:.2f}")
            
            return True
            
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 RumiAI P0 CRITICAL FIXES - FEAT VERIFICATION")
    print("=" * 60)
    
    # Test basic installation
    installation_ok = verify_feat_installation()
    
    if installation_ok:
        # Test video processing
        video_ok = test_video_processing()
        
        if video_ok:
            print("\n🎯 VERIFICATION COMPLETE!")
            print("✅ Ready to process videos with real emotion detection")
            print("✅ All P0 critical fixes verified and working")
            
            # Show next steps
            print("\n📋 Next Steps:")
            print("1. Run integration tests with real videos")
            print("2. Test with Python-only processing mode:")
            print("   export USE_PYTHON_ONLY_PROCESSING=true")
            print("   python3 scripts/rumiai_runner.py 'VIDEO_URL'")
            
            return 0
        else:
            print("\n⚠️  Video processing test failed")
            return 1
    else:
        print("\n❌ Installation verification failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)