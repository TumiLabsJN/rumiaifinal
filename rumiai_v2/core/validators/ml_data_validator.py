"""
ML data validators for RumiAI v2.

CRITICAL: These validators must handle multiple format variants found in production.
They normalize data rather than just validating to ensure pipeline continues.
"""
from typing import Dict, Any, List, Optional
from ..exceptions import ValidationError, handle_error
import logging

logger = logging.getLogger(__name__)


class MLDataValidator:
    """Validate and normalize ML model outputs."""
    
    @staticmethod
    def validate_yolo_data(data: Dict[str, Any], video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate and normalize YOLO object detection data.
        
        CRITICAL: YOLO output format varies by version.
        Must handle v5, v8, and custom formats.
        
        Expected structure (after normalization):
        {
            'objectAnnotations': [
                {
                    'class': str,
                    'frames': [
                        {
                            'timestamp': float/str,
                            'confidence': float,
                            'bbox': [x, y, w, h]
                        }
                    ]
                }
            ],
            'metadata': {...}
        }
        """
        try:
            # Handle multiple format variants
            if 'objectAnnotations' not in data:
                # Try alternative keys
                if 'detections' in data:
                    logger.warning(f"Converting 'detections' to 'objectAnnotations' for video {video_id}")
                    data['objectAnnotations'] = data['detections']
                elif 'results' in data:
                    logger.warning(f"Converting 'results' to 'objectAnnotations' for video {video_id}")
                    data['objectAnnotations'] = data['results']
                elif 'predictions' in data:
                    logger.warning(f"Converting 'predictions' to 'objectAnnotations' for video {video_id}")
                    data['objectAnnotations'] = data['predictions']
                else:
                    # Return empty but valid structure
                    logger.warning(f"No YOLO detections found for video {video_id}, using empty structure")
                    data['objectAnnotations'] = []
            
            # Ensure it's a list
            if not isinstance(data['objectAnnotations'], list):
                logger.warning(f"objectAnnotations is not a list for video {video_id}, wrapping in list")
                data['objectAnnotations'] = [data['objectAnnotations']]
            
            # Validate/normalize each annotation
            for i, annotation in enumerate(data['objectAnnotations']):
                if not isinstance(annotation, dict):
                    logger.warning(f"Skipping non-dict annotation at index {i}")
                    continue
                
                # Ensure required fields
                if 'class' not in annotation:
                    annotation['class'] = 'unknown'
                
                if 'frames' not in annotation:
                    # Try alternative keys
                    if 'detections' in annotation:
                        annotation['frames'] = annotation['detections']
                    elif 'instances' in annotation:
                        annotation['frames'] = annotation['instances']
                    else:
                        annotation['frames'] = []
                
                # Ensure frames is a list
                if not isinstance(annotation['frames'], list):
                    annotation['frames'] = [annotation['frames']]
            
            # Add metadata if missing
            if 'metadata' not in data:
                data['metadata'] = {'validated': True, 'version': 'unknown'}
            
            return data
            
        except Exception as e:
            return handle_error(
                ValidationError('yolo_data', data, 'valid YOLO structure', video_id),
                logger,
                default_return={'objectAnnotations': [], 'metadata': {'error': str(e)}}
            )
    
    @staticmethod
    def validate_whisper_data(data: Dict[str, Any], video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate and normalize Whisper transcription data.
        
        Expected structure:
        {
            'segments': [
                {
                    'start': float,
                    'end': float,
                    'text': str,
                    'confidence': float (optional)
                }
            ],
            'text': str,  # Full transcript
            'language': str
        }
        """
        try:
            # Ensure required fields
            if 'segments' not in data:
                logger.warning(f"No segments in Whisper data for video {video_id}")
                data['segments'] = []
            
            if not isinstance(data['segments'], list):
                data['segments'] = []
            
            # Validate each segment
            valid_segments = []
            for segment in data['segments']:
                if not isinstance(segment, dict):
                    continue
                
                # Ensure required segment fields
                if all(key in segment for key in ['start', 'end', 'text']):
                    # Convert timestamps to float
                    try:
                        segment['start'] = float(segment['start'])
                        segment['end'] = float(segment['end'])
                        valid_segments.append(segment)
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping segment with invalid timestamps: {segment}")
            
            data['segments'] = valid_segments
            
            # Generate full text if missing
            if 'text' not in data or not data['text']:
                data['text'] = ' '.join(seg['text'] for seg in valid_segments)
            
            # Set default language if missing
            if 'language' not in data:
                data['language'] = 'unknown'
            
            return data
            
        except Exception as e:
            return handle_error(
                ValidationError('whisper_data', data, 'valid Whisper structure', video_id),
                logger,
                default_return={'segments': [], 'text': '', 'language': 'unknown', 'error': str(e)}
            )
    
    @staticmethod
    def validate_mediapipe_data(data: Dict[str, Any], video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate and normalize MediaPipe human analysis data.
        
        Expected structure varies, but typically includes:
        {
            'poses': [...],
            'hands': [...],
            'face': [...],
            'presence_percentage': float,
            'frames_with_people': int
        }
        """
        try:
            # Ensure it's a dictionary
            if not isinstance(data, dict):
                logger.warning(f"MediaPipe data is not a dict for video {video_id}")
                data = {}
            
            # Add default values for missing fields
            if 'presence_percentage' not in data:
                data['presence_percentage'] = 0.0
            
            if 'frames_with_people' not in data:
                data['frames_with_people'] = 0
            
            # Ensure numeric values
            try:
                data['presence_percentage'] = float(data['presence_percentage'])
                data['frames_with_people'] = int(data['frames_with_people'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid numeric values in MediaPipe data for video {video_id}")
                data['presence_percentage'] = 0.0
                data['frames_with_people'] = 0
            
            return data
            
        except Exception as e:
            return handle_error(
                ValidationError('mediapipe_data', data, 'valid MediaPipe structure', video_id),
                logger,
                default_return={'presence_percentage': 0.0, 'frames_with_people': 0, 'error': str(e)}
            )
    
    @staticmethod
    def validate_ocr_data(data: Dict[str, Any], video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate and normalize OCR text detection data.
        
        Expected structure:
        {
            'textAnnotations': [...],
            'stickers': [...],
            'metadata': {...}
        }
        """
        try:
            # Handle different key names
            if 'textAnnotations' not in data:
                if 'text_annotations' in data:
                    data['textAnnotations'] = data['text_annotations']
                elif 'texts' in data:
                    data['textAnnotations'] = data['texts']
                else:
                    data['textAnnotations'] = []
            
            # Ensure lists
            if not isinstance(data['textAnnotations'], list):
                data['textAnnotations'] = []
            
            if 'stickers' not in data:
                data['stickers'] = []
            elif not isinstance(data['stickers'], list):
                data['stickers'] = []
            
            return data
            
        except Exception as e:
            return handle_error(
                ValidationError('ocr_data', data, 'valid OCR structure', video_id),
                logger,
                default_return={'textAnnotations': [], 'stickers': [], 'error': str(e)}
            )
    
    @staticmethod
    def validate_scene_detection_data(data: Dict[str, Any], video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate and normalize scene detection data.
        
        Expected structure:
        {
            'scenes': [
                {
                    'start_time': float,
                    'end_time': float,
                    'duration': float
                }
            ],
            'scene_changes': [float, ...],  # List of timestamps
            'total_scenes': int
        }
        """
        try:
            # Handle different formats
            if 'scenes' not in data:
                if 'cuts' in data:
                    data['scenes'] = data['cuts']
                elif 'shots' in data:
                    data['scenes'] = data['shots']
                else:
                    data['scenes'] = []
            
            # Ensure it's a list
            if not isinstance(data['scenes'], list):
                data['scenes'] = []
            
            # Extract scene change timestamps if not present
            if 'scene_changes' not in data:
                data['scene_changes'] = []
                for scene in data['scenes']:
                    if isinstance(scene, dict) and 'start_time' in scene:
                        try:
                            data['scene_changes'].append(float(scene['start_time']))
                        except (ValueError, TypeError):
                            pass
            
            # Add total count
            if 'total_scenes' not in data:
                data['total_scenes'] = len(data['scenes'])
            
            return data
            
        except Exception as e:
            return handle_error(
                ValidationError('scene_detection_data', data, 'valid scene detection structure', video_id),
                logger,
                default_return={'scenes': [], 'scene_changes': [], 'total_scenes': 0, 'error': str(e)}
            )
    
    @staticmethod
    def validate_emotion_data(data: Dict[str, Any], video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate FEAT emotion detection data with fail-fast approach.
        Raises exception on any data quality issues rather than normalizing.
        
        Expected structure:
        {
            'emotions': [
                {
                    'timestamp': float,
                    'emotion': str,
                    'confidence': float,
                    'all_scores': {...},
                    'action_units': [...],
                    'au_intensities': {...}
                }
            ]
        }
        """
        # FAIL if no emotions key
        if 'emotions' not in data:
            raise ValueError(f"FATAL: FEAT data missing 'emotions' key for video {video_id}")
        
        if not isinstance(data['emotions'], list):
            raise ValueError(f"FATAL: FEAT 'emotions' is not a list for video {video_id}")
        
        valid_emotions = {'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'}
        
        for i, emotion_entry in enumerate(data['emotions']):
            if not isinstance(emotion_entry, dict):
                raise ValueError(f"FATAL: FEAT emotion entry {i} is not a dict for video {video_id}")
            
            # Required fields - FAIL if missing
            if 'timestamp' not in emotion_entry:
                raise ValueError(f"FATAL: FEAT emotion entry {i} missing timestamp for video {video_id}")
            
            if 'emotion' not in emotion_entry:
                raise ValueError(f"FATAL: FEAT emotion entry {i} missing emotion for video {video_id}")
            
            # Validate timestamp
            timestamp = emotion_entry['timestamp']
            if not isinstance(timestamp, (int, float)) or timestamp < 0:
                raise ValueError(f"FATAL: Invalid timestamp {timestamp} at entry {i} for video {video_id}")
            
            # Validate emotion - FAIL if unknown
            emotion = emotion_entry['emotion']
            if emotion not in valid_emotions:
                raise ValueError(f"FATAL: Unknown emotion '{emotion}' at {timestamp}s for video {video_id}. "
                               f"Valid emotions: {valid_emotions}")
            
            # Validate confidence - FAIL if out of range
            if 'confidence' in emotion_entry:
                confidence = emotion_entry['confidence']
                if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"FATAL: Invalid confidence {confidence} at {timestamp}s for video {video_id}")
            
            # Validate action_units if present - FAIL if invalid
            if 'action_units' in emotion_entry:
                aus = emotion_entry['action_units']
                if not isinstance(aus, list):
                    raise ValueError(f"FATAL: action_units is not a list at {timestamp}s for video {video_id}")
                for au in aus:
                    if not isinstance(au, int) or not 1 <= au <= 45:
                        raise ValueError(f"FATAL: Invalid AU {au} at {timestamp}s for video {video_id}")
            
            # Validate au_intensities if present
            if 'au_intensities' in emotion_entry:
                intensities = emotion_entry['au_intensities']
                if not isinstance(intensities, dict):
                    raise ValueError(f"FATAL: au_intensities is not a dict at {timestamp}s for video {video_id}")
                for au, intensity in intensities.items():
                    if not isinstance(intensity, (int, float)) or not 0.0 <= intensity <= 1.0:
                        raise ValueError(f"FATAL: Invalid AU intensity {intensity} for AU {au} at {timestamp}s")
        
        # Return data AS-IS if all validation passes
        return data