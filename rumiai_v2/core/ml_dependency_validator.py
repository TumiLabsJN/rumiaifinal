"""
ML Dependency Validator - Fail-Fast Architecture
Ensures all ML dependencies are available before processing starts
"""

import sys
import importlib
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class CriticalDependencyError(Exception):
    """Raised when critical ML dependencies are missing"""
    pass

class MLDependencyValidator:
    """Validates all ML dependencies at startup - FAIL FAST"""
    
    REQUIRED_PACKAGES = {
        'torch': 'PyTorch',
        'ultralytics': 'YOLO',
        'mediapipe': 'MediaPipe',
        'easyocr': 'EasyOCR',
        'cv2': 'OpenCV',
        'scenedetect': 'SceneDetect'
    }
    
    REQUIRED_FILES = {
        '/home/jorge/rumiaifinal/yolov8n.pt': 'YOLO model',
        '/home/jorge/rumiaifinal/whisper.cpp/models/ggml-base.bin': 'Whisper model'
    }
    
    @classmethod
    def validate_all(cls) -> None:
        """
        Validate ALL dependencies upfront - no graceful degradation
        Raises CriticalDependencyError if any dependency is missing
        """
        missing_packages = []
        failed_imports = []
        missing_files = []
        
        # Check Python packages
        for package, name in cls.REQUIRED_PACKAGES.items():
            try:
                module = importlib.import_module(package)
                logger.debug(f"✅ {name} package found")
            except ImportError as e:
                missing_packages.append(f"{name} ({package})")
                failed_imports.append(str(e))
                logger.error(f"❌ {name} package missing: {e}")
        
        # Check required model files
        import os
        for filepath, name in cls.REQUIRED_FILES.items():
            if not os.path.exists(filepath):
                missing_files.append(f"{name}: {filepath}")
                logger.error(f"❌ {name} not found at {filepath}")
            else:
                logger.debug(f"✅ {name} found")
        
        # Build error message if anything is missing
        if missing_packages or missing_files:
            error_msg = "CRITICAL: ML Dependencies Failed\n"
            error_msg += "=" * 50 + "\n"
            
            if missing_packages:
                error_msg += "\n❌ Missing Python packages:\n"
                for pkg in missing_packages:
                    error_msg += f"   - {pkg}\n"
                error_msg += "\nTo fix, run:\n"
                error_msg += "   ./install_ml_dependencies.sh\n"
                error_msg += "Or:\n"
                error_msg += "   pip install ultralytics mediapipe easyocr torch torchvision --break-system-packages\n"
            
            if missing_files:
                error_msg += "\n❌ Missing model files:\n"
                for file in missing_files:
                    error_msg += f"   - {file}\n"
            
            error_msg += "\n" + "=" * 50
            error_msg += "\n⚠️  The ML pipeline CANNOT run without these dependencies."
            error_msg += "\n⚠️  Fix the issues above and try again."
            
            raise CriticalDependencyError(error_msg)
        
        logger.info("✅ All ML dependencies validated successfully")
    
    @classmethod
    def validate_ml_models_loaded(cls, models: Dict) -> None:
        """
        Validate that ML models are actually loaded, not None
        Called after model loading attempts
        """
        failed_models = []
        
        required_models = ['yolo', 'mediapipe', 'ocr', 'whisper']
        
        for model_name in required_models:
            if model_name not in models or models[model_name] is None:
                failed_models.append(model_name)
                logger.error(f"❌ {model_name} model failed to load")
        
        if failed_models:
            error_msg = f"CRITICAL: Failed to load ML models: {', '.join(failed_models)}\n"
            error_msg += "Check the logs for specific errors.\n"
            error_msg += "The pipeline cannot continue without these models."
            raise CriticalDependencyError(error_msg)
        
        logger.info("✅ All ML models loaded successfully")
    
    @classmethod
    def validate_ml_result(cls, result: Dict, service: str) -> None:
        """
        Validate ML service output structure and processing status
        Raises exception if result is invalid
        """
        from typing import Any
        
        if not result:
            raise ValueError(f"{service} returned empty result")
        
        # Check for required structure based on service
        if service == 'yolo':
            if 'objectAnnotations' not in result:
                raise ValueError(f"YOLO result missing objectAnnotations field")
            if result.get('metadata', {}).get('processed') is False:
                raise ValueError(f"YOLO failed to process video")
        
        elif service == 'mediapipe':
            if 'poses' not in result or 'faces' not in result:
                raise ValueError(f"MediaPipe result missing required fields")
            if result.get('metadata', {}).get('processed') is False:
                raise ValueError(f"MediaPipe failed to process video")
        
        elif service == 'ocr':
            if 'textAnnotations' not in result:
                raise ValueError(f"OCR result missing textAnnotations field")
            if result.get('metadata', {}).get('processed') is False:
                raise ValueError(f"OCR failed to process video")
        
        elif service == 'whisper':
            if 'segments' not in result:
                raise ValueError(f"Whisper result missing segments field")
        
        elif service == 'scene_detection':
            if 'scenes' not in result:
                raise ValueError(f"Scene detection result missing scenes field")
        
        # If we get here, validation passed
        logger.debug(f"✅ {service} result validation passed")