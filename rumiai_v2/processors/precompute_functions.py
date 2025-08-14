"""
Precompute functions for ML analysis.

This module provides compute functions that generate smart metrics
instead of sending full timelines to Claude API.
"""

import math
import numpy as np
import statistics
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import re
import logging

logger = logging.getLogger(__name__)

try:
    from .service_contracts import validate_compute_contract, validate_output_contract, ServiceContractViolation
except ImportError:
    # Fallback if service_contracts not available yet
    class ServiceContractViolation(ValueError):
        pass
    def validate_compute_contract(timelines, duration):
        pass
    def validate_output_contract(result, function_name):
        pass

logger = logging.getLogger(__name__)

# Format extraction helpers for compatibility with both old and new formats

def extract_yolo_data(ml_data):
    """Extract YOLO data from either old or new format"""
    yolo_data = ml_data.get('yolo', {})
    
    # Handle new format
    if 'objectAnnotations' in yolo_data:
        return yolo_data['objectAnnotations']
    
    # Legacy format fallback
    if 'detections' in yolo_data:
        return yolo_data['detections']
    
    # Check nested data structure
    if 'data' in yolo_data and 'objectAnnotations' in yolo_data['data']:
        return yolo_data['data']['objectAnnotations']
    
    return []

def extract_mediapipe_data(ml_data):
    """Extract MediaPipe data from either format"""
    mp_data = ml_data.get('mediapipe', {})
    
    # Direct access
    if 'poses' in mp_data:
        return {
            'poses': mp_data.get('poses', []),
            'faces': mp_data.get('faces', []),
            'hands': mp_data.get('hands', []),
            'gaze': mp_data.get('gaze', []),  # ADD: Extract gaze data
            'gestures': mp_data.get('gestures', []),
            'presence_percentage': mp_data.get('presence_percentage', 0),
            'frames_with_people': mp_data.get('frames_with_people', 0)
        }
    
    # Nested access
    if 'data' in mp_data:
        data = mp_data['data']
        return {
            'poses': data.get('poses', []),
            'faces': data.get('faces', []),
            'hands': data.get('hands', []),
            'gaze': data.get('gaze', []),  # ADD: Extract gaze data
            'gestures': data.get('gestures', []),
            'presence_percentage': data.get('presence_percentage', 0),
            'frames_with_people': data.get('frames_with_people', 0)
        }
    
    return {
        'poses': [], 'faces': [], 'hands': [], 'gaze': [], 'gestures': [],  # ADD: Include gaze
        'presence_percentage': 0, 'frames_with_people': 0
    }

def extract_whisper_data(ml_data):
    """Extract Whisper data from either format"""
    whisper_data = ml_data.get('whisper', {})
    
    if 'segments' in whisper_data:
        return whisper_data
    
    if 'data' in whisper_data and 'segments' in whisper_data['data']:
        return whisper_data['data']
    
    return {'text': '', 'segments': [], 'language': 'unknown'}

def extract_ocr_data(ml_data):
    """Extract OCR data from either format"""
    ocr_data = ml_data.get('ocr', {})
    
    if 'textAnnotations' in ocr_data:
        return ocr_data
    
    if 'data' in ocr_data and 'textAnnotations' in ocr_data['data']:
        return ocr_data['data']
    
    return {'textAnnotations': [], 'stickers': []}

# Helper functions above handle format extraction defensively


def parse_timestamp_to_seconds(timestamp: str) -> Optional[int]:
    """Convert timestamp like '0-1s' to start second"""
    try:
        return int(timestamp.split('-')[0])
    except:
        return None


def is_timestamp_in_second(timestamp: str, second: int) -> bool:
    """Check if a timestamp range overlaps with a given second"""
    try:
        parts = timestamp.split('-')
        if len(parts) == 2:
            start = float(parts[0])
            end = float(parts[1].replace('s', ''))
            return start <= second < end
        return False
    except:
        return False


def mean(values: List[float]) -> float:
    """Calculate mean of a list"""
    return sum(values) / len(values) if values else 0


def stdev(values: List[float]) -> float:
    """Calculate standard deviation of a list"""
    if len(values) < 2:
        return 0
    return statistics.stdev(values)


# Import the actual functions
try:
    # Import new creative density implementation
    from .precompute_creative_density import compute_creative_density_analysis
    logger.info("Successfully imported new creative_density implementation")
except ImportError as e:
    logger.error(f"Failed to import creative_density: {e}")
    # Fallback to placeholder
    def compute_creative_density_analysis(*args, **kwargs):
        logger.warning("Using placeholder for compute_creative_density_analysis")
        return {}

# Import professional functions first, fallback to basic implementations
try:
    from .precompute_professional import (
        compute_visual_overlay_analysis_professional,
        compute_emotional_journey_analysis_professional
    )
    logger.info("Successfully imported professional precompute functions!")
    
    # Create wrapper functions to match expected signatures
    def compute_visual_overlay_metrics(text_overlay_timeline, sticker_timeline, gesture_timeline, 
                                      speech_timeline, object_timeline, video_duration):
        # Convert individual timelines to combined dict format
        timelines = {
            'textTimeline': text_overlay_timeline or {},
            'stickerTimeline': sticker_timeline or {},
            'gestureTimeline': gesture_timeline or {},
            'speechTimeline': speech_timeline or {},
            'objectTimeline': object_timeline or {}
        }
        return compute_visual_overlay_analysis_professional(timelines, video_duration)
    
    def compute_emotional_metrics(expression_timeline, speech_timeline, gesture_timeline, duration, **kwargs):
        # Convert individual timelines to combined dict format  
        timelines = {
            'expressionTimeline': expression_timeline or {},
            'speechTimeline': speech_timeline or {},
            'gestureTimeline': gesture_timeline or {},
            'poseTimeline': kwargs.get('pose_timeline', {})
        }
        return compute_emotional_journey_analysis_professional(timelines, duration)
        
except ImportError as e:
    logger.error(f"Failed to import professional functions: {e}")
    # Fallback to basic functions from the full file
    try:
        from .precompute_functions_full import (
            compute_visual_overlay_metrics,
            compute_emotional_metrics,
            compute_metadata_analysis_metrics,
            compute_person_framing_metrics,
            compute_scene_pacing_metrics,
            compute_speech_analysis_metrics
        )
        logger.info("Using basic precompute functions as fallback")
    except ImportError as e2:
        logger.error(f"Failed to import basic precompute functions: {e2}")
        # Define placeholder functions if both imports fail
        def compute_visual_overlay_metrics(*args, **kwargs):
            logger.warning("Using placeholder for compute_visual_overlay_metrics")
            return {}
        
        def compute_emotional_metrics(*args, **kwargs):
            logger.warning("Using placeholder for compute_emotional_metrics")
            return {}

# Import remaining basic functions (not yet upgraded to professional)
try:
    from .precompute_functions_full import (
        compute_metadata_analysis_metrics,
        compute_person_framing_metrics,
        compute_scene_pacing_metrics,
        compute_speech_analysis_metrics
    )
    logger.info("Successfully imported remaining basic precompute functions")
except ImportError as e:
    logger.error(f"Failed to import remaining functions: {e}")
    
    def compute_metadata_analysis_metrics(*args, **kwargs):
        logger.warning("Using placeholder for compute_metadata_analysis_metrics")
        return {}
    
    def compute_person_framing_metrics(expression_timeline, object_timeline, camera_distance_timeline,
                                      person_timeline, enhanced_human_data, duration, gaze_timeline=None):
        """Compute person framing metrics using MediaPipe face data from person_timeline"""
        
        # Initialize metrics
        face_sizes = []
        total_frames = int(duration) if duration else 0
        face_frames = 0
        eye_contact_frames = 0
        max_faces_in_frame = 0  # Track maximum faces detected in any frame
        
        # Extract MediaPipe face data from person_timeline (populated by wrapper)
        face_seconds = set()  # Track unique seconds with faces
        
        for timestamp_key, person_data in person_timeline.items():
            # Check if this entry has MediaPipe face data
            if person_data.get('face_bbox') and person_data.get('detected'):
                second = int(timestamp_key.split('-')[0])  # Extract second from "0-1s" format
                face_seconds.add(second)
                
                # Track face count for multi-person detection
                face_count = person_data.get('face_count', 1)
                max_faces_in_frame = max(max_faces_in_frame, face_count)
                
                # Extract face bbox for size calculation  
                bbox = person_data['face_bbox']
                if bbox and isinstance(bbox, dict):
                    # MediaPipe bbox is in relative coordinates (0-1), convert to percentage
                    face_area = bbox.get('width', 0) * bbox.get('height', 0) * 100
                    face_sizes.append(face_area)
        
        # Face frames = unique seconds with MediaPipe face detections
        face_frames = len(face_seconds)
        
        # Calculate face visibility and average face size
        face_visibility_rate = face_frames / total_frames if total_frames > 0 else 0
        avg_face_size = sum(face_sizes) / len(face_sizes) if face_sizes else 0
        
        # Extract eye contact data from gaze timeline
        if gaze_timeline:
            for timestamp_key, gaze_data in gaze_timeline.items():
                if gaze_data.get('eye_contact', 0) > 0.5:  # Threshold for eye contact
                    eye_contact_frames += 1
        
        eye_contact_rate = eye_contact_frames / total_frames if total_frames > 0 else 0
        
        # Calculate framing consistency (simplified)
        framing_consistency = 0.8 if face_frames > 0 else 0  # Default for now
        
        # Determine dominant framing based on face size
        if avg_face_size > 30:
            dominant_framing = 'close-up'
        elif avg_face_size > 15:
            dominant_framing = 'medium'
        else:
            dominant_framing = 'wide'
        
        # Calculate subject count dynamically from MediaPipe data
        subject_count = max(1, max_faces_in_frame)  # At least 1, up to max detected
        primary_subject = 'multiple' if subject_count > 1 else 'single'
        
        # Calculate gaze steadiness based on eye contact consistency
        if eye_contact_rate > 0.7:
            gaze_steadiness = 'high'
        elif eye_contact_rate > 0.4:
            gaze_steadiness = 'medium'
        elif eye_contact_rate > 0:
            gaze_steadiness = 'low'
        else:
            gaze_steadiness = 'unknown'
        
        # Return the expected fields for professional wrapper
        return {
            'avg_face_size': avg_face_size,
            'face_visibility_rate': face_visibility_rate,
            'eye_contact_rate': eye_contact_rate,
            'framing_consistency': framing_consistency,
            'primary_subject': primary_subject,  # Calculated from data
            'subject_count': subject_count,  # Calculated from data
            'dominant_framing': dominant_framing,
            'gaze_steadiness': gaze_steadiness,  # Calculated from eye contact
            'framing_progression': [],  # Default for now
            'distance_variation': 0,  # Default for now
            'framing_transitions': 0,  # Default for now
            'movement_pattern': 'static',  # Default for now
            'stability_score': framing_consistency
        }
    
    def compute_scene_pacing_metrics(*args, **kwargs):
        logger.warning("Using placeholder for compute_scene_pacing_metrics")
        return {}
    
    def compute_speech_analysis_metrics(*args, **kwargs):
        logger.warning("Using placeholder for compute_speech_analysis_metrics")
        return {}


# Helper functions to extract data from unified analysis
def _extract_timelines_from_analysis(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract timeline data from unified analysis with validation"""
    timeline_data = analysis_dict.get('timeline', {})
    ml_data = analysis_dict.get('ml_data', {})
    
    # Track extraction metrics
    video_id = analysis_dict.get('video_id', 'unknown')
    logger.info(f"Starting timeline extraction for video {video_id}")
    
    # Build timelines dictionary expected by compute functions
    timelines = {
        'textOverlayTimeline': {},
        'stickerTimeline': {},
        'speechTimeline': {},
        'objectTimeline': {},
        'gestureTimeline': {},
        'expressionTimeline': {},
        'gazeTimeline': {},  # NEW: For eye contact tracking
        'sceneTimeline': {},
        'sceneChangeTimeline': {},  # Add this for scene pacing
        'personTimeline': {},
        'cameraDistanceTimeline': {}
    }
    
    # Use helper for robust OCR extraction (handles multiple formats)
    ocr_data = extract_ocr_data(ml_data)
    
    # Transform text annotations to timeline format
    for annotation in ocr_data.get('textAnnotations', []):
        timestamp = annotation.get('timestamp', 0)
        start = int(timestamp)
        end = start + 1
        timestamp_key = f"{start}-{end}s"
        
        # Calculate position from bbox
        bbox = annotation.get('bbox', [0, 0, 0, 0])
        y_pos = bbox[1] if len(bbox) > 1 else 0
        
        if y_pos > 400:  # Bottom third (subtitles)
            position = 'bottom'
        elif y_pos < 200:  # Top third
            position = 'top'
        else:
            position = 'center'
        
        timelines['textOverlayTimeline'][timestamp_key] = {
            'text': annotation.get('text', ''),
            'position': position,
            'size': 'medium',
            'confidence': annotation.get('confidence', 0.9)
        }
    
    # Handle stickers
    for sticker in ocr_data.get('stickers', []):
        timestamp = sticker.get('timestamp', 0)
        timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
        timelines['stickerTimeline'][timestamp_key] = sticker
    
    # Use helper for robust Whisper extraction
    whisper_data = extract_whisper_data(ml_data)
    
    # Transform segments to timeline format
    for segment in whisper_data.get('segments', []):
        start = int(segment.get('start', 0))
        end = int(segment.get('end', start + 1))
        timestamp = f"{start}-{end}s"
        timelines['speechTimeline'][timestamp] = {
            'text': segment.get('text', ''),
            'confidence': segment.get('confidence', 0.9),
            'start_time': segment.get('start', 0),
            'end_time': segment.get('end', 0)
        }
    
    # Use helper for robust YOLO extraction
    yolo_objects = extract_yolo_data(ml_data)
    
    # Transform objects to timeline format
    for obj in yolo_objects:
        timestamp = obj.get('timestamp', 0)
        start = int(timestamp)
        end = start + 1
        timestamp_key = f"{start}-{end}s"
        
        if timestamp_key not in timelines['objectTimeline']:
            timelines['objectTimeline'][timestamp_key] = {
                'objects': {},
                'total_objects': 0,
                'confidence_details': []
            }
        
        # Count objects by class
        obj_class = obj.get('className', 'unknown')
        entry = timelines['objectTimeline'][timestamp_key]
        
        if obj_class not in entry['objects']:
            entry['objects'][obj_class] = 0
        
        entry['objects'][obj_class] += 1
        entry['total_objects'] += 1
        
        # Preserve full details for reference
        entry['confidence_details'].append({
            'class': obj_class,
            'confidence': obj.get('confidence', 0.5),
            'trackId': obj.get('trackId', ''),
            'bbox': obj.get('bbox', [])
        })
    
    # Extract scene changes from timeline entries
    timeline_entries = timeline_data.get('entries', [])
    scenes = []
    scene_changes = []
    current_scene_start = 0
    
    for entry in timeline_entries:
        if entry.get('entry_type') == 'scene_change':
            # Extract seconds from start time (handles both string and numeric formats)
            start_value = entry.get('start', 0)
            if isinstance(start_value, str) and start_value.endswith('s'):
                try:
                    start_seconds = float(start_value[:-1])
                except ValueError:
                    start_seconds = 0
            elif isinstance(start_value, (int, float)):
                start_seconds = float(start_value)
            else:
                start_seconds = 0
            
            # Add to scene changes list
            scene_changes.append(start_seconds)
            
            scenes.append({
                'start': current_scene_start,
                'end': start_seconds
            })
            current_scene_start = start_seconds
    
    # Add final scene
    duration = timeline_data.get('duration', 0)
    if current_scene_start < duration:
        scenes.append({
            'start': current_scene_start,
            'end': duration
        })
    
    # Convert scenes to timeline format
    for i, scene in enumerate(scenes):
        start = scene['start']
        end = scene['end']
        # Keep integer format for display
        timestamp = f"{int(start)}-{int(end)}s"
        timelines['sceneTimeline'][timestamp] = {
            'scene_number': i + 1,
            'duration': end - start
        }
    
    # Also create sceneChangeTimeline expected by compute_scene_pacing_metrics
    # Use the format expected by parse_timestamp_to_seconds: "X-Ys"
    # Group scene changes by integer second to avoid overwriting
    scene_changes_by_second = {}
    for i, change_time in enumerate(scene_changes):
        second = int(change_time)
        if second not in scene_changes_by_second:
            scene_changes_by_second[second] = []
        scene_changes_by_second[second].append({
            'index': i + 1,
            'actual_time': change_time
        })
    
    # Create timeline entries, handling multiple changes per second
    for second, changes in scene_changes_by_second.items():
        if len(changes) == 1:
            # Single change in this second - ensure end > start for validation
            timestamp = f"{second}-{second + 1}s"
            timelines['sceneChangeTimeline'][timestamp] = {
                'type': 'scene_change',
                'scene_index': changes[0]['index'],
                'actual_time': changes[0]['actual_time']
            }
        else:
            # Multiple changes in same second - use proper fractional seconds
            # FIX for bug 1601M: Use actual decimal math instead of string concatenation
            for j, change in enumerate(changes):
                # Use actual fractional seconds (supports up to 1000 changes per second)
                fraction = j / 1000.0
                start_time = second + fraction
                end_time = second + (j + 1) / 1000.0
                
                # Format with consistent precision to ensure start < end
                timestamp = f"{start_time:.3f}-{end_time:.3f}s"
                # Examples: "0.000-0.001s", "0.009-0.010s", "0.999-1.000s"
                
                timelines['sceneChangeTimeline'][timestamp] = {
                    'type': 'scene_change',
                    'scene_index': change['index'],
                    'actual_time': change['actual_time'],
                    'multiple_in_second': True
                }
    
    # Use helper for robust MediaPipe extraction
    mediapipe_data = extract_mediapipe_data(ml_data)
    
    # Transform poses to timeline
    for pose in mediapipe_data.get('poses', []):
        timestamp = pose.get('timestamp', 0)
        timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
        timelines['personTimeline'][timestamp_key] = {
            'detected': True,
            'pose_confidence': pose.get('confidence', 0.8),
            'face_bbox': None,  # Will be filled if face detected at same time
            'face_confidence': None
        }
    
    # Merge face data into personTimeline (Step 2B from GazeFix.md)
    for face in mediapipe_data.get('faces', []):
        timestamp = face.get('timestamp', 0)
        timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
        
        # Create or update personTimeline entry with face data
        if timestamp_key in timelines['personTimeline']:
            # Person already detected via pose, add face data
            timelines['personTimeline'][timestamp_key]['face_bbox'] = face.get('bbox')
            timelines['personTimeline'][timestamp_key]['face_confidence'] = face.get('confidence')
            timelines['personTimeline'][timestamp_key]['face_count'] = face.get('count', 1)  # Add face count
        else:
            # Face detected but no pose, create entry
            timelines['personTimeline'][timestamp_key] = {
                'detected': True,
                'pose_confidence': None,
                'face_bbox': face.get('bbox'),
                'face_confidence': face.get('confidence', 0),
                'face_count': face.get('count', 1)  # Add face count
            }
    
    # Transform gestures to timeline
    for gesture in mediapipe_data.get('gestures', []):
        timestamp = gesture.get('timestamp', 0)
        timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
        timelines['gestureTimeline'][timestamp_key] = gesture
    
    # Extract gaze data from MediaPipe (new from FaceMesh iris detection)
    for gaze in mediapipe_data.get('gaze', []):
        timestamp = gaze.get('timestamp', 0)
        timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
        timelines['gazeTimeline'][timestamp_key] = gaze
    
    # Extract FEAT emotion entries from timeline instead of MediaPipe faces
    for entry in timeline_entries:
        if entry.get('entry_type') == 'emotion':
            # Extract timestamp range
            start = entry.get('start', 0)
            # Handle both string format ("1.5s") and numeric format
            if isinstance(start, str) and start.endswith('s'):
                start = float(start[:-1])
            elif hasattr(start, 'seconds'):
                start = start.seconds
            else:
                start = float(start)
            
            end = entry.get('end', start + 1)
            if isinstance(end, str) and end.endswith('s'):
                end = float(end[:-1])
            elif hasattr(end, 'seconds'):
                end = end.seconds
            else:
                end = float(end)
            
            timestamp_key = f"{int(start)}-{int(end)}s"
            
            # Add FEAT data to expressionTimeline
            timelines['expressionTimeline'][timestamp_key] = entry.get('data', {})
    
    # Extract gaze entries from timeline for eye contact tracking
    for entry in timeline_entries:
        if entry.get('entry_type') == 'gaze':
            # Extract timestamp range
            start = entry.get('start', 0)
            if isinstance(start, str) and start.endswith('s'):
                start = float(start[:-1])
            elif hasattr(start, 'seconds'):
                start = start.seconds
            else:
                start = float(start)
            
            timestamp_key = f"{int(start)}-{int(start)+1}s"
            
            # Add gaze data to gazeTimeline
            timelines['gazeTimeline'][timestamp_key] = entry.get('data', {})
    
    # Log extraction results for validation
    extraction_summary = {
        'video_id': video_id,
        'text_overlays': len(timelines['textOverlayTimeline']),
        'stickers': len(timelines['stickerTimeline']),
        'speech_segments': len(timelines['speechTimeline']),
        'object_timestamps': len(timelines['objectTimeline']),
        'scene_changes': len(timelines['sceneChangeTimeline']),
        'poses': len(timelines['personTimeline']),
        'gestures': len(timelines['gestureTimeline']),
        'expressions': len(timelines['expressionTimeline']),
        'gaze_tracking': len(timelines['gazeTimeline'])
    }
    
    # Check for potential extraction failures
    ocr_available = len(ml_data.get('ocr', {}).get('textAnnotations', []))
    yolo_available = len(ml_data.get('yolo', {}).get('objectAnnotations', []))
    whisper_available = len(ml_data.get('whisper', {}).get('segments', []))
    
    if ocr_available > 0 and extraction_summary['text_overlays'] == 0:
        logger.warning(f"OCR extraction may have failed: {ocr_available} annotations available but 0 extracted")
    
    if yolo_available > 0 and extraction_summary['object_timestamps'] == 0:
        logger.warning(f"YOLO extraction may have failed: {yolo_available} objects available but 0 in timeline")
    
    if whisper_available > 0 and extraction_summary['speech_segments'] == 0:
        logger.warning(f"Whisper extraction may have failed: {whisper_available} segments available but 0 extracted")
    
    logger.info(f"Timeline extraction complete: {extraction_summary}")
    
    return timelines


def _extract_metadata_summary(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata summary from analysis"""
    metadata = analysis_dict.get('metadata', {})
    
    return {
        'title': metadata.get('title', metadata.get('description', '')[:50]),  # Use first 50 chars of description as title
        'description': metadata.get('description', ''),
        'views': metadata.get('views', 0),
        'likes': metadata.get('likes', 0),
        'comments': metadata.get('comments', 0),
        'shares': metadata.get('shares', 0),
        'hashtags': [tag.get('name', '') for tag in metadata.get('hashtags', [])],
        'mentions': metadata.get('mentions', []),
        'music': metadata.get('music', {}).get('musicName', ''),
        'author': metadata.get('author', {}).get('nickName', metadata.get('username', ''))
    }


# Wrapper functions for each compute function
def compute_creative_density_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for creative density computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    duration = analysis_dict.get('timeline', {}).get('duration', 0)
    return compute_creative_density_analysis(timelines, duration)


def compute_emotional_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for emotional metrics computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    expression_timeline = timelines.get('expressionTimeline', {})  # Get the correct timeline
    speech_timeline = timelines.get('speechTimeline', {})
    gesture_timeline = timelines.get('gestureTimeline', {})
    duration = analysis_dict.get('timeline', {}).get('duration', 0)
    return compute_emotional_metrics(expression_timeline, speech_timeline, gesture_timeline, duration)


def compute_speech_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for speech analysis computation"""
    ml_data = analysis_dict.get('ml_data', {})
    
    # Use helpers for robust extraction
    whisper_data = extract_whisper_data(ml_data)
    transcript = whisper_data.get('text', '')
    speech_segments = whisper_data.get('segments', [])
    
    timelines = _extract_timelines_from_analysis(analysis_dict)
    expression_timeline = timelines.get('expressionTimeline', {})
    gesture_timeline = timelines.get('gestureTimeline', {})
    
    # Use helper for MediaPipe data extraction
    mediapipe_data = extract_mediapipe_data(ml_data)
    human_analysis_data = {
        'faces': mediapipe_data.get('faces', []),
        'poses': mediapipe_data.get('poses', []),
        'gestures': mediapipe_data.get('gestures', [])
    }
    
    # Extract audio energy data if available
    audio_energy_data = ml_data.get('audio_energy', {})
    energy_level_windows = audio_energy_data.get('energy_level_windows', {})
    energy_variance = audio_energy_data.get('energy_variance', 0)
    climax_timestamp = audio_energy_data.get('climax_timestamp', 0)
    burst_pattern = audio_energy_data.get('burst_pattern', 'none')
    
    video_duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    speech_timeline = timelines.get('speechTimeline', {})
    
    # Get basic metrics first
    basic_result = compute_speech_analysis_metrics(
        speech_timeline, transcript, speech_segments, expression_timeline, 
        gesture_timeline, human_analysis_data, video_duration,
        energy_level_windows, energy_variance, climax_timestamp, burst_pattern
    )
    
    # Convert to professional 6-block format
    from .precompute_professional_wrappers import ensure_professional_format
    return ensure_professional_format(basic_result, 'speech_analysis')


def compute_visual_overlay_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for visual overlay computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    
    text_overlay_timeline = timelines.get('textOverlayTimeline', {})
    sticker_timeline = timelines.get('stickerTimeline', {})
    gesture_timeline = timelines.get('gestureTimeline', {})
    speech_timeline = timelines.get('speechTimeline', {})
    object_timeline = timelines.get('objectTimeline', {})
    video_duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    return compute_visual_overlay_metrics(
        text_overlay_timeline, sticker_timeline, gesture_timeline,
        speech_timeline, object_timeline, video_duration
    )


def compute_metadata_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for metadata analysis computation"""
    static_metadata = analysis_dict.get('metadata', {})  # Get actual metadata
    metadata_summary = _extract_metadata_summary(analysis_dict)
    video_duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    # Get basic metrics first
    basic_result = compute_metadata_analysis_metrics(static_metadata, metadata_summary, video_duration)
    
    # Convert to professional 6-block format
    from .precompute_professional_wrappers import ensure_professional_format
    return ensure_professional_format(basic_result, 'metadata_analysis')


def compute_person_framing_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for person framing computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    
    expression_timeline = timelines.get('expressionTimeline', {})
    object_timeline = timelines.get('objectTimeline', {})
    camera_distance_timeline = timelines.get('cameraDistanceTimeline', {})
    person_timeline = timelines.get('personTimeline', {})
    
    # Enhanced human data is no longer used - always empty
    enhanced_human_data = {}
    
    duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    # Debug logging to check which function is being used
    logger.info(f"compute_person_framing_metrics module: {compute_person_framing_metrics.__module__}")
    logger.info(f"personTimeline entries: {len(person_timeline)}")
    logger.info(f"gazeTimeline entries: {len(timelines.get('gazeTimeline', {}))}")
    logger.info(f"expressionTimeline entries: {len(expression_timeline)}")
    
    # Get basic metrics first
    basic_result = compute_person_framing_metrics(
        expression_timeline, object_timeline, camera_distance_timeline,
        person_timeline, enhanced_human_data, duration,
        gaze_timeline=timelines.get('gazeTimeline', {})
    )
    
    # Debug logging to check the result
    logger.info(f"compute_person_framing_metrics result: eye_contact_rate={basic_result.get('eye_contact_rate', 0)}, avg_face_size={basic_result.get('avg_face_size', 0)}")
    
    # Convert to professional 6-block format
    from .precompute_professional_wrappers import ensure_professional_format
    return ensure_professional_format(basic_result, 'person_framing')


def compute_scene_pacing_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for scene pacing computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    video_duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    # Extract the specific timelines needed
    scene_timeline = timelines.get('sceneChangeTimeline', {})
    object_timeline = timelines.get('objectTimeline', {})
    camera_distance_timeline = timelines.get('cameraDistanceTimeline', {})
    video_id = analysis_dict.get('video_id', None)
    
    # Get basic metrics first
    basic_result = compute_scene_pacing_metrics(
        scene_timeline=scene_timeline,
        video_duration=video_duration,
        object_timeline=object_timeline,
        camera_distance_timeline=camera_distance_timeline,
        video_id=video_id
    )
    
    # Convert to professional 6-block format
    from .precompute_professional_wrappers import ensure_professional_format
    return ensure_professional_format(basic_result, 'scene_pacing')


# Create a mapping of compute function names to their wrapper implementations
COMPUTE_FUNCTIONS = {
    'creative_density': compute_creative_density_wrapper,
    'emotional_journey': compute_emotional_wrapper,
    'person_framing': compute_person_framing_wrapper,
    'scene_pacing': compute_scene_pacing_wrapper,
    'speech_analysis': compute_speech_wrapper,
    'visual_overlay_analysis': compute_visual_overlay_wrapper,
    'metadata_analysis': compute_metadata_wrapper
}


def get_compute_function(name: str):
    """Get a compute function by name"""
    return COMPUTE_FUNCTIONS.get(name)