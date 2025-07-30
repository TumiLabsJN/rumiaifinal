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


# Import the actual functions from the full file
try:
    from .precompute_functions_full import (
        compute_visual_overlay_metrics,
        compute_creative_density_analysis,
        compute_emotional_metrics,
        compute_metadata_analysis_metrics,
        compute_person_framing_metrics,
        compute_scene_pacing_metrics,
        compute_speech_analysis_metrics
    )
    logger.info("Successfully imported precompute functions")
except ImportError as e:
    logger.error(f"Failed to import precompute functions: {e}")
    # Define placeholder functions if import fails
    def compute_visual_overlay_metrics(*args, **kwargs):
        logger.warning("Using placeholder for compute_visual_overlay_metrics")
        return {}
    
    def compute_creative_density_analysis(*args, **kwargs):
        logger.warning("Using placeholder for compute_creative_density_analysis")
        return {}
    
    def compute_emotional_metrics(*args, **kwargs):
        logger.warning("Using placeholder for compute_emotional_metrics")
        return {}
    
    def compute_metadata_analysis_metrics(*args, **kwargs):
        logger.warning("Using placeholder for compute_metadata_analysis_metrics")
        return {}
    
    def compute_person_framing_metrics(*args, **kwargs):
        logger.warning("Using placeholder for compute_person_framing_metrics")
        return {}
    
    def compute_scene_pacing_metrics(*args, **kwargs):
        logger.warning("Using placeholder for compute_scene_pacing_metrics")
        return {}
    
    def compute_speech_analysis_metrics(*args, **kwargs):
        logger.warning("Using placeholder for compute_speech_analysis_metrics")
        return {}


# Helper functions to extract data from unified analysis
def _extract_timelines_from_analysis(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract timeline data from unified analysis"""
    timeline_data = analysis_dict.get('timeline', {})
    ml_data = analysis_dict.get('ml_data', {})
    
    # Build timelines dictionary expected by compute functions
    timelines = {
        'textOverlayTimeline': {},
        'stickerTimeline': {},
        'speechTimeline': {},
        'objectTimeline': {},
        'gestureTimeline': {},
        'expressionTimeline': {},
        'sceneTimeline': {},
        'personTimeline': {},
        'cameraDistanceTimeline': {}
    }
    
    # Extract text overlays from OCR data
    ocr_data = ml_data.get('ocr', {}).get('data', {})
    if 'text_overlays' in ocr_data:
        for overlay in ocr_data['text_overlays']:
            timestamp = overlay.get('timestamp', '0-1s')
            timelines['textOverlayTimeline'][timestamp] = {
                'text': overlay.get('text', ''),
                'position': overlay.get('position', 'center'),
                'size': overlay.get('size', 'medium')
            }
    
    # Extract stickers
    if 'stickers' in ocr_data:
        for sticker in ocr_data['stickers']:
            timestamp = sticker.get('timestamp', '0-1s')
            timelines['stickerTimeline'][timestamp] = sticker
    
    # Extract speech from whisper data
    whisper_data = ml_data.get('whisper', {}).get('data', {})
    if 'segments' in whisper_data:
        for segment in whisper_data['segments']:
            start = int(segment.get('start', 0))
            end = int(segment.get('end', start + 1))
            timestamp = f"{start}-{end}s"
            timelines['speechTimeline'][timestamp] = {
                'text': segment.get('text', ''),
                'confidence': segment.get('confidence', 0.9)
            }
    
    # Extract objects from YOLO data
    yolo_data = ml_data.get('yolo', {}).get('data', {})
    if 'detections' in yolo_data:
        for detection in yolo_data['detections']:
            timestamp = detection.get('timestamp', '0-1s')
            if timestamp not in timelines['objectTimeline']:
                timelines['objectTimeline'][timestamp] = []
            timelines['objectTimeline'][timestamp].append({
                'class': detection.get('class', 'unknown'),
                'confidence': detection.get('confidence', 0.5)
            })
    
    # Extract scene changes from timeline entries
    timeline_entries = timeline_data.get('entries', [])
    scenes = []
    current_scene_start = 0
    
    for entry in timeline_entries:
        if entry.get('entry_type') == 'scene_change':
            # Extract seconds from start time string (e.g., "3s" -> 3)
            start_str = entry.get('start', '0s')
            if isinstance(start_str, str) and start_str.endswith('s'):
                start_seconds = int(start_str[:-1])
            else:
                start_seconds = 0
            
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
        start = int(scene['start'])
        end = int(scene['end'])
        timestamp = f"{start}-{end}s"
        timelines['sceneTimeline'][timestamp] = {
            'scene_number': i + 1,
            'duration': end - start
        }
    
    # Extract MediaPipe data
    mediapipe_data = ml_data.get('mediapipe', {}).get('data', {})
    if 'poses' in mediapipe_data:
        for pose in mediapipe_data['poses']:
            timestamp = pose.get('timestamp', '0-1s')
            timelines['personTimeline'][timestamp] = pose
    
    if 'gestures' in mediapipe_data:
        for gesture in mediapipe_data['gestures']:
            timestamp = gesture.get('timestamp', '0-1s')
            timelines['gestureTimeline'][timestamp] = gesture
    
    if 'faces' in mediapipe_data:
        for face in mediapipe_data['faces']:
            timestamp = face.get('timestamp', '0-1s')
            timelines['expressionTimeline'][timestamp] = face
    
    return timelines


def _extract_metadata_summary(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata summary from analysis"""
    metadata = analysis_dict.get('metadata', {})
    
    return {
        'title': metadata.get('text', ''),
        'description': metadata.get('text', ''),
        'views': metadata.get('playCount', 0),
        'likes': metadata.get('diggCount', 0),
        'comments': metadata.get('commentCount', 0),
        'shares': metadata.get('shareCount', 0),
        'hashtags': [tag.get('name', '') for tag in metadata.get('hashtags', [])],
        'mentions': metadata.get('mentions', []),
        'music': metadata.get('musicMeta', {}).get('musicName', ''),
        'author': metadata.get('authorMeta', {}).get('nickName', '')
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
    speech_timeline = timelines.get('speechTimeline', {})
    gesture_timeline = timelines.get('gestureTimeline', {})
    duration = analysis_dict.get('timeline', {}).get('duration', 0)
    return compute_emotional_metrics(timelines, speech_timeline, gesture_timeline, duration)


def compute_speech_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for speech analysis computation"""
    whisper_data = analysis_dict.get('ml_data', {}).get('whisper', {}).get('data', {})
    transcript = whisper_data.get('text', '')
    speech_segments = whisper_data.get('segments', [])
    
    timelines = _extract_timelines_from_analysis(analysis_dict)
    expression_timeline = timelines.get('expressionTimeline', {})
    gesture_timeline = timelines.get('gestureTimeline', {})
    
    # Extract human analysis data
    mediapipe_data = analysis_dict.get('ml_data', {}).get('mediapipe', {}).get('data', {})
    human_analysis_data = {
        'faces': mediapipe_data.get('faces', []),
        'poses': mediapipe_data.get('poses', []),
        'gestures': mediapipe_data.get('gestures', [])
    }
    
    video_duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    speech_timeline = timelines.get('speechTimeline', {})
    
    return compute_speech_analysis_metrics(
        speech_timeline, transcript, speech_segments, expression_timeline, 
        gesture_timeline, human_analysis_data, video_duration
    )


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
    timelines = _extract_timelines_from_analysis(analysis_dict)
    metadata_summary = _extract_metadata_summary(analysis_dict)
    video_duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    return compute_metadata_analysis_metrics(timelines, metadata_summary, video_duration)


def compute_person_framing_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for person framing computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    
    expression_timeline = timelines.get('expressionTimeline', {})
    object_timeline = timelines.get('objectTimeline', {})
    camera_distance_timeline = timelines.get('cameraDistanceTimeline', {})
    person_timeline = timelines.get('personTimeline', {})
    
    # Extract enhanced human data
    mediapipe_data = analysis_dict.get('ml_data', {}).get('mediapipe', {}).get('data', {})
    enhanced_human_data = {
        'faces': mediapipe_data.get('faces', []),
        'poses': mediapipe_data.get('poses', [])
    }
    
    duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    return compute_person_framing_metrics(
        expression_timeline, object_timeline, camera_distance_timeline,
        person_timeline, enhanced_human_data, duration
    )


def compute_scene_pacing_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for scene pacing computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    video_duration = analysis_dict.get('timeline', {}).get('duration', 0)
    
    return compute_scene_pacing_metrics(timelines, video_duration)


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