"""
MediaPipe Gesture Recognition Service
Handles gesture detection for all video analysis flows
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class GestureRecognizerService:
    """Singleton gesture recognizer to avoid multiple model loads"""
    _instance = None
    _recognizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._recognizer is None:
            self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Initialize MediaPipe gesture recognizer"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            
            # Model path
            model_path = Path(__file__).parent.parent.parent / "models" / "gesture_recognizer.task"
            
            if not model_path.exists():
                logger.error(f"Gesture model not found at {model_path}")
                logger.info("Download from: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task")
                self._recognizer = None
                return
            
            # Create recognizer
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.GestureRecognizerOptions(
                base_options=base_options,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                num_hands=2
            )
            
            self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)
            logger.info("Gesture recognizer initialized successfully")
            
        except ImportError as e:
            logger.error(f"MediaPipe import failed: {e}")
            self._recognizer = None
        except Exception as e:
            logger.error(f"Failed to initialize gesture recognizer: {e}")
            self._recognizer = None
    
    def recognize_frame(self, frame: np.ndarray, timestamp_ms: int = 0) -> List[Dict[str, Any]]:
        """
        Recognize gestures in a single frame
        
        Args:
            frame: RGB frame as numpy array
            timestamp_ms: Frame timestamp in milliseconds
            
        Returns:
            List of detected gestures with confidence scores
        """
        if self._recognizer is None:
            return []
        
        try:
            import mediapipe as mp
            
            # Convert frame to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Perform recognition
            result = self._recognizer.recognize(mp_image)
            
            # Process results
            gestures = []
            if result.gestures:
                for hand_idx, gesture_list in enumerate(result.gestures):
                    if gesture_list:
                        top_gesture = gesture_list[0]
                        if top_gesture.score > 0.5:  # Confidence threshold
                            gestures.append({
                                'type': self._map_gesture_name(top_gesture.category_name),
                                'confidence': float(top_gesture.score),
                                'hand': 'left' if hand_idx == 0 else 'right',
                                'timestamp_ms': timestamp_ms
                            })
            
            return gestures
            
        except Exception as e:
            logger.warning(f"Gesture recognition failed for frame: {e}")
            return []
    
    def _map_gesture_name(self, mediapipe_name: str) -> str:
        """Map MediaPipe gesture names to RumiAI conventions"""
        mapping = {
            'Thumb_Up': 'thumbs_up',
            'Thumb_Down': 'thumbs_down',
            'Victory': 'victory',
            'Pointing_Up': 'pointing',
            'Open_Palm': 'open_palm',
            'Closed_Fist': 'closed_fist',
            'ILoveYou': 'love',
            'None': 'none'
        }
        return mapping.get(mediapipe_name, mediapipe_name.lower())
    
    def cleanup(self):
        """Release resources"""
        if self._recognizer:
            self._recognizer.close()
            self._recognizer = None
            logger.info("Gesture recognizer cleaned up")