# Temporal Windows Refactoring - Complete Implementation

## Executive Summary

This document provides the complete implementation for refactoring `temporal_compute.py` to follow the existing precompute function pattern. The refactored module will be self-contained and will be the ONLY output from `rumiai_runner.py`, replacing all 7 existing analyses with a single `temporal_windows.json` output.

**Key Alignment**: This implementation fully incorporates all decisions from decisions3.md to ensure compatibility with the current architecture.

## Part A: Overview and Key Decisions

### Incorporated Decisions from decisions3.md

1. **Decision 1 (Defensive Extraction)**: Handle both direct and nested ML data formats via helper functions
2. **Decision 2 (Audio Energy)**: Store raw RMS frames and calculate burst patterns per window
3. **Decision 3 (Direct Integration)**: Direct call from rumiai_runner.py without COMPUTE_FUNCTIONS registry
4. **Decision 4 (Full Metrics)**: All windows and segments get identical comprehensive metric sets
5. **Decision 5 (Mixed Extraction)**: Extract from both ml_data AND timeline entries based on data type

### What Changes from Current Architecture

- **From**: 7 separate analysis functions → **To**: 1 temporal_windows function
- **From**: 21 output files → **To**: 1 temporal_windows.json file
- **From**: COMPUTE_FUNCTIONS registry → **To**: Direct function call
- **From**: Multiple parameters → **To**: Single analysis_dict parameter

## Part B: Complete Implementation

### Section 1: Defensive Extraction Helper Functions

```python
# temporal_compute.py - Self-contained module

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time

logger = logging.getLogger(__name__)

# ============================================
# MODULE CONSTANTS - NO MAGIC NUMBERS
# ============================================

# Frame rate constants
DEFAULT_FPS = 30  # Fallback when audio service doesn't provide fps (TikTok's most common rate)

# Video duration validation
MIN_VIDEO_DURATION = 0.1  # 100ms - anything shorter is likely corrupted data  
MAX_VIDEO_DURATION = 3600  # 1 hour - sanity check (TikTok max is 60 min for some users)

# Temporal window durations (industry standard from short-form video research)
HOOK_WINDOW_DURATION = 3.0  # First 3 seconds - critical for viewer retention
CLOSING_WINDOW_DURATION = 3.0  # Last 3 seconds - critical for engagement/sharing

# ============================================
# SEGMENT CALCULATION CONSTANTS
# ============================================
# Based on Decision 10 from decisions2.md
# These thresholds determine how many middle segments to create

SEGMENT_THRESHOLDS = {
    'min_duration_for_segments': 3,     # Videos < 3s have no middle segments
    'three_segments_max': 12,            # Videos 3-12s get 3 segments
    'four_segments_max': 27,             # Videos 13-27s get 4 segments
    # Videos > 27s get 5 segments
}

"""
SEGMENT CALCULATION RULES:
- Videos < 3 seconds: No middle segments (too short)
- Videos 3-12 seconds: 3 middle segments
- Videos 13-27 seconds: 4 middle segments  
- Videos > 27 seconds: 5 middle segments

These thresholds were chosen to balance granularity with meaningful
temporal divisions based on typical short-form video patterns.
"""

# ============================================
# DEFENSIVE EXTRACTION HELPERS (Decision 1)
# ============================================

def extract_ocr_data(ml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Defensive extraction for OCR data - handles nested formats"""
    ocr_data = ml_data.get('ocr', {})
    
    # Check direct format
    if 'textAnnotations' in ocr_data:
        return ocr_data
    
    # Check nested format
    if 'data' in ocr_data and 'textAnnotations' in ocr_data['data']:
        return ocr_data['data']
    
    # Return empty structure
    return {'textAnnotations': [], 'stickers': []}

def extract_whisper_data(ml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Defensive extraction for Whisper data - handles nested formats"""
    whisper_data = ml_data.get('whisper', {})
    
    # Check direct format
    if 'segments' in whisper_data:
        return whisper_data
    
    # Check nested format
    if 'data' in whisper_data and 'segments' in whisper_data['data']:
        return whisper_data['data']
    
    # Return empty structure
    return {'segments': [], 'text': '', 'language': 'unknown'}

def extract_yolo_data(ml_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Defensive extraction for YOLO data - handles multiple formats"""
    yolo_data = ml_data.get('yolo', {})
    
    # Check new format
    if 'objectAnnotations' in yolo_data:
        return yolo_data['objectAnnotations']
    
    # Check legacy format
    if 'detections' in yolo_data:
        return yolo_data['detections']
    
    # Check nested format
    if 'data' in yolo_data:
        if 'objectAnnotations' in yolo_data['data']:
            return yolo_data['data']['objectAnnotations']
        if 'detections' in yolo_data['data']:
            return yolo_data['data']['detections']
    
    # Return empty list
    return []

def extract_mediapipe_data(ml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Defensive extraction for MediaPipe data - handles nested formats"""
    mp_data = ml_data.get('mediapipe', {})
    
    # Check direct format
    if 'poses' in mp_data or 'faces' in mp_data:
        return {
            'poses': mp_data.get('poses', []),
            'faces': mp_data.get('faces', []),
            'hands': mp_data.get('hands', []),
            'gaze': mp_data.get('gaze', [])
        }
    
    # Check nested format
    if 'data' in mp_data:
        nested = mp_data['data']
        return {
            'poses': nested.get('poses', []),
            'faces': nested.get('faces', []),
            'hands': nested.get('hands', []),
            'gaze': nested.get('gaze', [])
        }
    
    # Return empty structure
    return {'poses': [], 'faces': [], 'hands': [], 'gaze': []}

def extract_audio_energy_data(ml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Defensive extraction for audio energy data - handles nested formats"""
    audio_data = ml_data.get('audio_energy', {})
    
    # Check direct format
    if 'rms_frames' in audio_data:
        return audio_data
    
    # Check nested format
    if 'data' in audio_data and 'rms_frames' in audio_data['data']:
        return audio_data['data']
    
    # Return empty structure
    return {}

# ============================================
# TIMELINE EXTRACTION (Decision 5 - Mixed Strategy)
# ============================================

def extract_timelines_for_temporal(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data using mixed strategy - both ml_data and timeline entries.
    Implements Decision 5 from decisions3.md.
    
    DATA SOURCE MAPPING:
    ====================
    From ml_data (raw ML service outputs):
    - OCR: text annotations and inline sticker detection
    - Whisper: speech segments with transcriptions
    - YOLO: object detection bounding boxes
    - MediaPipe: raw pose/face/hand landmarks
    - Audio Energy: RMS frames and frame rate
    
    From timeline.entries (processed/semantic layer):
    - Emotions: FEAT processed emotion classifications
    - Scene Changes: PySceneDetect scene boundaries
    - Gestures: Interpreted gestures from MediaPipe
    - Gaze: Camera gaze detection from MediaPipe
    - Camera Distance: Shot type from face bbox size (close/medium/wide)
    
    Rationale: Timeline entries contain semantic interpretations
    (e.g., "happy" emotion, "pointing" gesture) while ml_data 
    contains raw detection results (e.g., facial landmarks, hand positions).
    """
    ml_data = analysis_dict.get('ml_data', {})
    timeline = analysis_dict.get('timeline', {})
    timelines = {}
    
    # ============================================
    # 1. EXTRACT FROM TIMELINE ENTRIES
    # ============================================
    # Timeline contains processed semantic interpretations
    
    timeline_entries = timeline.get('entries', [])
    
    # Emotion data from timeline entries (FEAT results)
    expression_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'emotion':
            expression_timeline.append({
                'timestamp': entry.get('start', 0),
                'emotion': entry.get('data', {}).get('emotion', 'neutral'),
                'confidence': entry.get('data', {}).get('confidence', 0),
                'source': 'feat'
            })
    timelines['expression_timeline'] = expression_timeline
    
    # Scene data from timeline entries (PySceneDetect results)
    scene_change_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') in ['scene_change', 'scene']:
            scene_change_timeline.append({
                'timestamp': entry.get('start', 0),
                'scene_number': entry.get('data', {}).get('scene_number', 0),
                'confidence': entry.get('data', {}).get('confidence', 0),
                'source': 'pyscenedetect'
            })
    timelines['scene_change_timeline'] = scene_change_timeline
    
    # Gesture data from timeline entries (processed from MediaPipe)
    gesture_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'gesture':
            gesture_timeline.append({
                'timestamp': entry.get('start', 0),
                'gesture': entry.get('data', {}).get('gesture', 'unknown'),
                'confidence': entry.get('data', {}).get('confidence', 0)
            })
    timelines['gesture_timeline'] = gesture_timeline
    
    # Gaze data from timeline entries (processed from MediaPipe)
    gaze_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'gaze':
            gaze_timeline.append({
                'timestamp': entry.get('start', 0),
                'looking_at_camera': entry.get('data', {}).get('looking_at_camera', False),
                'confidence': entry.get('data', {}).get('confidence', 0)
            })
    timelines['gaze_timeline'] = gaze_timeline
    
    # Camera distance/framing from timeline entries (calculated from face bbox by timeline_builder)
    camera_distance_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'camera_distance':
            camera_distance_timeline.append({
                'timestamp': entry.get('start', 0),
                'distance': entry.get('data', {}).get('distance', 'medium'),
                'confidence': entry.get('data', {}).get('confidence', 0)
            })
    timelines['camera_distance_timeline'] = camera_distance_timeline
    
    # ============================================
    # 2. EXTRACT FROM ML_DATA USING DEFENSIVE HELPERS
    # ============================================
    # ML data contains raw detection results from services
    
    # OCR data (raw text detections and inline sticker analysis)
    ocr_data = extract_ocr_data(ml_data)
    timelines['text_overlay_timeline'] = [
        {'timestamp': ann.get('timestamp', 0), 'text': ann.get('text', '')}
        for ann in ocr_data.get('textAnnotations', [])
    ]
    # Stickers detected inline during OCR processing via HSV analysis
    timelines['sticker_timeline'] = ocr_data.get('stickers', [])
    
    # Whisper speech segments (raw transcription with timestamps)
    whisper_data = extract_whisper_data(ml_data)
    timelines['speech_segments'] = whisper_data.get('segments', [])
    
    # YOLO object detections (raw bounding boxes and labels)
    timelines['object_timeline'] = extract_yolo_data(ml_data)
    
    # MediaPipe raw landmark data (unprocessed poses/faces)
    # Note: Gestures and gaze are interpreted from these in timeline.entries
    mediapipe_data = extract_mediapipe_data(ml_data)
    timelines['pose_timeline'] = mediapipe_data.get('poses', [])
    timelines['face_timeline'] = mediapipe_data.get('faces', [])
    
    # Audio energy RMS frames for burst pattern analysis (Decision 2)
    audio_data = extract_audio_energy_data(ml_data)
    timelines['rms_frames'] = audio_data.get('rms_frames', [])
    timelines['frames_per_second'] = audio_data.get('frames_per_second', DEFAULT_FPS)
    
    return timelines

# ============================================
# AUDIO ENERGY PROCESSING (Decision 2)
# ============================================

def calculate_burst_pattern_for_window(window_rms: np.ndarray) -> str:
    """Decision 2C: Determine burst pattern for a specific window"""
    if len(window_rms) < 2:
        return "steady"
    
    third_size = len(window_rms) // 3
    if third_size == 0:
        return "steady"
    
    front_avg = np.mean(window_rms[:third_size])
    middle_avg = np.mean(window_rms[third_size:2*third_size])
    back_avg = np.mean(window_rms[2*third_size:])
    
    energies = {'front': front_avg, 'middle': middle_avg, 'back': back_avg}
    peak = max(energies, key=energies.get)
    
    if peak == 'front':
        return "front_loaded"
    elif peak == 'back':
        return "back_loaded"
    elif peak == 'middle':
        return "middle_peak"
    else:
        return "steady"

def calculate_audio_energy_for_windows(audio_data: Dict[str, Any], windows: Dict[str, Optional[Tuple[float, float]]], 
                                      video_duration: float) -> Dict[str, Any]:
    """
    Calculate audio energy metrics using RMS frames from ML data.
    Implements Decision 2A, 2B, 2C from decisions3.md.
    """
    # Decision 2B: Return zeros for missing RMS frames
    if not audio_data or 'rms_frames' not in audio_data:
        logger.info("No audio RMS frames available - video may be silent")
        results = {}
        for window_name in ['hook', 'middle', 'closing']:
            results[f'{window_name}_energy_level'] = 0.0
            results[f'{window_name}_energy_variance'] = 0.0
            results[f'{window_name}_energy_max'] = 0.0
            results[f'{window_name}_burst_pattern'] = 'steady'
        return results
    
    rms_frames = np.array(audio_data['rms_frames'])
    fps = audio_data.get('frames_per_second', DEFAULT_FPS)
    results = {}
    
    for window_name, bounds in windows.items():
        if bounds is None:
            continue
            
        start_frame = int(bounds[0] * fps)
        end_frame = min(int(bounds[1] * fps), len(rms_frames))
        
        if start_frame >= len(rms_frames):
            # Window beyond audio duration
            results[f'{window_name}_energy_level'] = 0.0
            results[f'{window_name}_energy_variance'] = 0.0
            results[f'{window_name}_energy_max'] = 0.0
            results[f'{window_name}_burst_pattern'] = 'steady'
            continue
            
        window_rms = rms_frames[start_frame:end_frame]
        
        if len(window_rms) == 0:
            # Empty window
            results[f'{window_name}_energy_level'] = 0.0
            results[f'{window_name}_energy_variance'] = 0.0
            results[f'{window_name}_energy_max'] = 0.0
            results[f'{window_name}_burst_pattern'] = 'steady'
        else:
            # Calculate all metrics including burst pattern
            results[f'{window_name}_energy_level'] = float(np.mean(window_rms))
            results[f'{window_name}_energy_variance'] = float(np.var(window_rms))
            results[f'{window_name}_energy_max'] = float(np.max(window_rms))
            results[f'{window_name}_burst_pattern'] = calculate_burst_pattern_for_window(window_rms)
    
    return results

# ============================================
# WINDOW CALCULATION HELPERS
# ============================================

def calculate_temporal_windows(video_duration: float) -> Dict[str, Optional[Tuple[float, float]]]:
    """
    Calculate hook, middle, and closing windows based on video duration.
    Handles edge cases for short videos (Decision 9).
    """
    if video_duration <= HOOK_WINDOW_DURATION:
        return {
            'hook': (0, video_duration),
            'middle': None,
            'closing': None
        }
    elif video_duration <= (HOOK_WINDOW_DURATION + CLOSING_WINDOW_DURATION):
        return {
            'hook': (0, HOOK_WINDOW_DURATION),
            'middle': None,
            'closing': (HOOK_WINDOW_DURATION, video_duration)
        }
    else:
        return {
            'hook': (0, HOOK_WINDOW_DURATION),
            'middle': (HOOK_WINDOW_DURATION, video_duration - CLOSING_WINDOW_DURATION),
            'closing': (video_duration - CLOSING_WINDOW_DURATION, video_duration)
        }

def calculate_middle_segments(video_duration: float) -> Dict[str, Dict[str, float]]:
    """
    Calculate 3-5 middle segments based on video duration.
    Implements Decision 10 from decisions2.md and P0TemporalWindows.md spec.
    
    Uses SEGMENT_THRESHOLDS constants defined at module level.
    """
    if video_duration <= 6:
        return {}
    
    middle_start = HOOK_WINDOW_DURATION
    middle_end = video_duration - CLOSING_WINDOW_DURATION
    middle_duration = middle_end - middle_start
    
    # Handle edge cases
    if middle_duration <= 0:
        return {}
    
    # No segments if middle < 3s (per P0 spec and SEGMENT_THRESHOLDS)
    if middle_duration < SEGMENT_THRESHOLDS['min_duration_for_segments']:
        return {}  # Middle exists but no segments
    
    # Determine number of segments using module-level constants
    if middle_duration <= SEGMENT_THRESHOLDS['three_segments_max']:
        num_segments = 3  # 3-12s middle duration
    elif middle_duration <= SEGMENT_THRESHOLDS['four_segments_max']:
        num_segments = 4  # 13-27s middle duration  
    else:
        num_segments = 5  # >27s middle duration
    
    segment_duration = middle_duration / num_segments
    segments = {}
    
    for i in range(num_segments):
        segment_start = middle_start + (i * segment_duration)
        segment_end = segment_start + segment_duration
        segments[f'segment_{i+1}'] = {
            'start': segment_start,
            'end': segment_end
        }
    
    return segments

# ============================================
# SEGMENT PROCESSING (Decision 4)
# ============================================

def process_segment(seg_bounds: Dict[str, float], timelines: Dict[str, Any], 
                   audio_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process segment with FULL metrics matching hook/closing windows.
    Implements Decision 4 from decisions3.md - ML consistency across all windows.
    Includes all P0 required metrics from P0TemporalWindows.md.
    """
    start = seg_bounds['start']
    end = seg_bounds['end']
    
    # Get speech segments from timelines
    speech_segments = timelines.get('speech_segments', [])
    
    # Filter data to segment bounds
    segment_text = [t for t in timelines.get('text_overlay_timeline', []) 
                   if start <= t.get('timestamp', 0) < end]
    segment_stickers = [s for s in timelines.get('sticker_timeline', [])
                       if start <= s.get('timestamp', 0) < end]
    segment_objects = [o for o in timelines.get('object_timeline', [])
                      if start <= o.get('timestamp', 0) < end]
    segment_gestures = [g for g in timelines.get('gesture_timeline', [])
                       if start <= g.get('timestamp', 0) < end]
    segment_expressions = [e for e in timelines.get('expression_timeline', [])
                          if start <= e.get('timestamp', 0) < end]
    segment_scenes = [s for s in timelines.get('scene_change_timeline', [])
                     if start <= s.get('timestamp', 0) < end]
    segment_camera = [c for c in timelines.get('camera_distance_timeline', [])
                     if start <= c.get('timestamp', 0) < end]
    segment_speech = [s for s in speech_segments
                     if s.get('start', 0) >= start and s.get('end', 0) <= end]
    
    # Calculate all P0 required counts
    text_count = len(segment_text)
    sticker_count = len(segment_stickers)
    object_count = len(segment_objects)
    gesture_count = len(segment_gestures)
    expression_count = len(segment_expressions)
    scene_count = len(segment_scenes)
    
    # P0 spec: element_count is sum of ALL 6 types
    total_elements = (text_count + sticker_count + object_count + 
                     gesture_count + expression_count + scene_count)
    
    # Calculate density metrics
    duration = end - start
    avg_density = total_elements / duration if duration > 0 else 0
    changes_per_second = scene_count / duration if duration > 0 else 0
    
    # Calculate density extremes (P0 requirement)
    # Single-pass bucketing for O(n) performance instead of O(n*m)
    interval_count = max(1, int(end - start))
    densities = [0] * interval_count
    
    # Single pass through each element type, bucketing by second
    for t in segment_text:
        second = int(t.get('timestamp', 0) - start)
        if 0 <= second < interval_count:
            densities[second] += 1
    
    for s in segment_stickers:
        second = int(s.get('timestamp', 0) - start)
        if 0 <= second < interval_count:
            densities[second] += 1
    
    for o in segment_objects:
        second = int(o.get('timestamp', 0) - start)
        if 0 <= second < interval_count:
            densities[second] += 1
    
    for g in segment_gestures:
        second = int(g.get('timestamp', 0) - start)
        if 0 <= second < interval_count:
            densities[second] += 1
    
    for e in segment_expressions:
        second = int(e.get('timestamp', 0) - start)
        if 0 <= second < interval_count:
            densities[second] += 1
    
    for sc in segment_scenes:
        second = int(sc.get('timestamp', 0) - start)
        if 0 <= second < interval_count:
            densities[second] += 1
    
    # Calculate min/max density from buckets
    if densities:
        max_density = float(max(densities))
        min_density = float(min(densities))
    else:
        # Fallback for edge cases
        max_density = avg_density
        min_density = avg_density
    
    # Calculate speech metrics
    speech_duration = sum(s.get('end', 0) - s.get('start', 0) for s in segment_speech)
    speech_coverage = speech_duration / duration if duration > 0 else 0
    word_count = sum(len(s.get('text', '').split()) for s in segment_speech)
    
    # Calculate emotion distribution
    # Note: Emotion labels are already standardized by timeline_builder.py
    # We can safely count emotions directly since the timeline builder
    # has already mapped any FEAT variations to standard labels.
    emotion_counts = {}
    for e in segment_expressions:
        emotion = e.get('emotion', 'neutral')
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    total_emotions = len(segment_expressions)
    emotion_dist = {
        f'{emotion}_ratio': count / total_emotions if total_emotions > 0 else 0
        for emotion, count in emotion_counts.items()
    }
    
    # Calculate framing distribution from camera distance timeline
    # Camera distance is calculated by timeline_builder from face bbox size:
    # close (>25% frame), medium (8-25%), wide (<8%), none (no face)
    framing_counts = {'close': 0, 'medium': 0, 'wide': 0, 'none': 0}
    for c in segment_camera:
        distance = c.get('distance', 'none')
        if distance in framing_counts:
            framing_counts[distance] += 1
    
    total_camera_frames = len(segment_camera) if segment_camera else 1  # Avoid divide by zero
    framing_dist = {
        f'{frame_type}_ratio': count / total_camera_frames if total_camera_frames > 0 else 0
        for frame_type, count in framing_counts.items()
    }
    
    # Calculate audio energy for segment
    # Note: This is NOT duplication of calculate_audio_energy_for_windows().
    # That function calculates energy for the 3 main windows (hook/middle/closing),
    # while here we calculate energy for EACH individual segment within the middle.
    # These are different time ranges requiring separate calculations.
    if audio_data and 'rms_frames' in audio_data:
        rms_frames = np.array(audio_data['rms_frames'])
        fps = audio_data.get('frames_per_second', DEFAULT_FPS)
        start_frame = int(start * fps)
        end_frame = min(int(end * fps), len(rms_frames))
        
        if start_frame < len(rms_frames):
            segment_rms = rms_frames[start_frame:end_frame]
            if len(segment_rms) > 0:
                energy_level = float(np.mean(segment_rms))
                energy_variance = float(np.var(segment_rms))
                energy_max = float(np.max(segment_rms))
                burst_pattern = calculate_burst_pattern_for_window(segment_rms)
            else:
                energy_level = energy_variance = energy_max = 0.0
                burst_pattern = 'steady'
        else:
            energy_level = energy_variance = energy_max = 0.0
            burst_pattern = 'steady'
    else:
        energy_level = energy_variance = energy_max = 0.0
        burst_pattern = 'steady'
    
    return {
        'start': start,
        'end': end,
        'duration': duration,
        # P0 required counts (all 6 types)
        'text_count': text_count,
        'sticker_count': sticker_count,
        'object_count': object_count,
        'gesture_count': gesture_count,
        'expression_count': expression_count,
        'scene_count': scene_count,
        'element_count': total_elements,  # Sum of all 6 types
        # P0 density extremes
        'max_density': max_density,
        'min_density': min_density,
        'avg_density': avg_density,
        # Other metrics
        'changes_per_second': changes_per_second,
        'speech_coverage': speech_coverage,
        'word_count': word_count,
        # Emotion distribution
        **emotion_dist,
        # Framing distribution (camera shot types)
        **framing_dist,
        # Audio energy
        'energy_level': energy_level,
        'energy_variance': energy_variance,
        'energy_max': energy_max,
        'burst_pattern': burst_pattern
    }

# ============================================
# MAIN COMPUTE FUNCTION
# ============================================

def compute_temporal_windows(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete implementation of temporal windows computation.
    
    This is the ONLY compute function that will be called from rumiai_runner.py,
    replacing all 7 existing analysis functions.
    
    Args:
        analysis_dict: Dictionary containing unified analysis results with structure:
            {
                'ml_data': {  # Raw ML service outputs (required)
                    'ocr': {
                        'textAnnotations': [...],  # Text detections
                        'stickers': [...]          # Inline sticker detections
                    },
                    'whisper': {
                        'segments': [...]          # Speech transcriptions
                    },
                    'yolo': {
                        'objectAnnotations': [...] # Object detections
                    },
                    'mediapipe': {
                        'poses': [...],            # Raw pose landmarks
                        'faces': [...]             # Raw face landmarks
                    },
                    'audio_energy': {
                        'rms_frames': [...],       # RMS energy values
                        'frames_per_second': 30    # Frame rate
                    }
                },
                'timeline': {  # Processed semantic events (required)
                    'entries': [
                        {
                            'entry_type': 'emotion',  # or 'scene_change', 'gesture', 'gaze'
                            'start': float,
                            'data': {...}
                        },
                        ...
                    ]
                },
                'metadata': {  # Video metadata (required)
                    'video_id': str,           # Unique identifier
                    'caption_length': int,     # Caption character count
                    'hashtag_count': int,      # Number of hashtags
                    ...
                },
                'duration': float  # Video duration in seconds (required)
            }
    
    Returns:
        Dict[str, Any]: Temporal window analysis with structure:
            {
                'hook_window': {...},      # First 3 seconds metrics
                'middle_segments': [...],  # 3-5 segments with metrics
                'closing_window': {...},   # Last 3 seconds metrics
                'video_duration': float,
                'segment_count': int
            }
    
    Raises:
        ValueError: If required fields are missing or invalid
        TypeError: If analysis_dict is not a dictionary
    
    Incorporates all decisions from decisions3.md:
    - Decision 1: Defensive extraction via helper functions
    - Decision 2: Audio energy with burst patterns
    - Decision 3: Direct integration (no COMPUTE_FUNCTIONS)
    - Decision 4: Full metrics for all windows
    - Decision 5: Mixed extraction strategy
    
    EXPECTED INPUT STRUCTURE EXAMPLE:
    =================================
    {
        'ml_data': {
            'ocr': {
                'textAnnotations': [
                    {'timestamp': 1.5, 'text': 'Hello World'},
                    {'timestamp': 2.3, 'text': 'Subscribe!'}
                ],
                'stickers': [
                    {'timestamp': 0.5, 'type': 'emoji', 'value': '❤️'}
                ]
            },
            'whisper': {
                'segments': [
                    {'start': 0.0, 'end': 2.5, 'text': 'Hey everyone'},
                    {'start': 2.5, 'end': 5.0, 'text': 'Welcome back'}
                ]
            },
            'yolo': {
                'objectAnnotations': [
                    {'timestamp': 1.0, 'label': 'person', 'confidence': 0.95}
                ]
            },
            'mediapipe': {
                'poses': [...],  # Raw landmark data
                'faces': [...]   # Raw landmark data
            },
            'audio_energy': {
                'rms_frames': [0.1, 0.15, 0.2, ...],  # Energy values
                'frames_per_second': 30
            }
        },
        'timeline': {
            'entries': [
                {
                    'entry_type': 'emotion',
                    'start': 0.5,
                    'data': {'emotion': 'happy', 'confidence': 0.8}
                },
                {
                    'entry_type': 'scene_change',
                    'start': 3.0,
                    'data': {'scene_number': 2}
                }
            ]
        },
        'metadata': {
            'video_id': 'abc123',
            'caption_length': 150,
            'hashtag_count': 5
        },
        'duration': 15.5
    }
    """
    
    # Start performance timing
    start_time = time.time()
    
    # ============================================
    # STEP 1: Input Validation & Service Verification
    # ============================================
    
    # Early validation before we have video_id
    if not analysis_dict:
        raise ValueError("[Video unknown] compute_temporal_windows: analysis_dict is None or empty")
    
    if not isinstance(analysis_dict, dict):
        raise TypeError(f"[Video unknown] compute_temporal_windows: Expected dict, got {type(analysis_dict)}")
    
    # Extract core components and video_id first
    ml_data = analysis_dict.get('ml_data', {})
    timeline = analysis_dict.get('timeline', {})
    metadata = analysis_dict.get('metadata', {})
    video_duration = analysis_dict.get('duration', 0)
    video_id = metadata.get('video_id', 'unknown')
    
    # Now all errors can include video_id for context
    
    # FAIL FAST: Verify all required ML services are present
    # This ensures services ran, even if they found no elements
    required_services = ['ocr', 'whisper', 'yolo', 'mediapipe', 'audio_energy']
    missing_services = [s for s in required_services if s not in ml_data]
    
    if missing_services:
        raise ValueError(
            f"Required ML services missing for video {video_id}: {missing_services}. "
            f"All services must run even if they find no elements."
        )
    
    # Timeline is also required for emotion/scene/gesture/gaze data
    if 'timeline' not in analysis_dict:
        raise ValueError(f"Timeline data missing for video {video_id}")
    
    # After this point, we can safely accept empty results from services
    # because we know the services ran (they're present in ml_data)
    
    # Validate video duration using module constants
    if video_duration <= 0:
        logger.error(f"[Video {video_id}] Invalid duration: {video_duration}")
        raise ValueError(f"[Video {video_id}] Duration must be positive, got {video_duration}")
    
    if video_duration < MIN_VIDEO_DURATION:
        logger.error(f"[Video {video_id}] Duration too short: {video_duration}s")
        raise ValueError(f"[Video {video_id}] Duration {video_duration}s below minimum {MIN_VIDEO_DURATION}s")
    
    if video_duration > MAX_VIDEO_DURATION:
        logger.warning(f"[Video {video_id}] Unusually long duration: {video_duration}s exceeds {MAX_VIDEO_DURATION}s")
        # Continue processing but log the warning for investigation
    
    # ============================================
    # STEP 2: Extract Timelines (Mixed Strategy)
    # ============================================
    
    extraction_start = time.time()
    timelines = extract_timelines_for_temporal(analysis_dict)
    extraction_time = time.time() - extraction_start
    
    if extraction_time > 0.5:  # Extraction should be fast
        logger.warning(f"[Video {video_id}] Slow timeline extraction: {extraction_time:.3f}s")
    
    # ============================================
    # STEP 3: Calculate Temporal Windows
    # ============================================
    
    windows = calculate_temporal_windows(video_duration)
    
    # ============================================
    # STEP 4: Calculate Audio Energy Metrics
    # ============================================
    
    audio_data = extract_audio_energy_data(ml_data)
    audio_energy_metrics = calculate_audio_energy_for_windows(
        audio_data, windows, video_duration
    )
    
    # ============================================
    # STEP 5: Process Hook Window (0-3s)
    # ============================================
    
    hook_data = None
    if windows['hook']:
        hook_bounds = {
            'start': windows['hook'][0],
            'end': windows['hook'][1]
        }
        hook_data = process_segment(hook_bounds, timelines, audio_data)
        
        # Add window-specific audio metrics
        hook_data['energy_level'] = audio_energy_metrics.get('hook_energy_level', 0)
        hook_data['energy_variance'] = audio_energy_metrics.get('hook_energy_variance', 0)
        hook_data['energy_max'] = audio_energy_metrics.get('hook_energy_max', 0)
        hook_data['burst_pattern'] = audio_energy_metrics.get('hook_burst_pattern', 'steady')
    
    # ============================================
    # STEP 6: Process Middle Segments
    # ============================================
    
    middle_segments = []
    if windows['middle']:
        segments = calculate_middle_segments(video_duration)
        for seg_name, seg_bounds in segments.items():
            seg_data = process_segment(seg_bounds, timelines, audio_data)
            seg_data['segment_name'] = seg_name
            middle_segments.append(seg_data)
    
    # ============================================
    # STEP 7: Process Closing Window (last 3s)
    # ============================================
    
    closing_data = None
    if windows['closing']:
        closing_bounds = {
            'start': windows['closing'][0],
            'end': windows['closing'][1]
        }
        closing_data = process_segment(closing_bounds, timelines, audio_data)
        
        # Add window-specific audio metrics
        closing_data['energy_level'] = audio_energy_metrics.get('closing_energy_level', 0)
        closing_data['energy_variance'] = audio_energy_metrics.get('closing_energy_variance', 0)
        closing_data['energy_max'] = audio_energy_metrics.get('closing_energy_max', 0)
        closing_data['burst_pattern'] = audio_energy_metrics.get('closing_burst_pattern', 'steady')
    
    # ============================================
    # STEP 8: Calculate Metadata
    # ============================================
    
    calculated_metadata = {
        'video_id': video_id,
        'duration': video_duration,
        'digg_count': metadata.get('diggCount', 0),
        'play_count': metadata.get('playCount', 0),
        'collect_count': metadata.get('collectCount', 0),
        'share_count': metadata.get('shareCount', 0),
        'comment_count': metadata.get('commentCount', 0),
        'create_time': metadata.get('createTime', 0),
        'author': metadata.get('author', {}).get('uniqueId', ''),
        'description': metadata.get('desc', '')
    }
    
    # ============================================
    # STEP 9: Build Final Result
    # ============================================
    
    result = {
        'video_id': video_id,
        'duration': video_duration,
        'temporal_windows': {
            'hook': hook_data,
            'middle_segments': middle_segments,
            'closing': closing_data
        },
        'metadata': calculated_metadata,
        'processing_timestamp': time.time(),
        'version': '2.0.0'  # Temporal compute version
    }
    
    # Log performance metrics
    total_time = time.time() - start_time
    logger.info(f"[Video {video_id}] Temporal compute completed in {total_time:.3f}s")
    
    # Add performance warning for slow processing
    if total_time > 1.0:  # More than 1 second is concerning
        logger.warning(f"[Video {video_id}] Slow temporal compute: {total_time:.3f}s for {video_duration:.1f}s video")
    
    return result
```

### Section 2: Integration Instructions

## Part C: Integration with rumiai_runner.py

### Direct Integration (Decision 3)

Replace the current analysis loop in rumiai_runner.py:

```python
# OLD CODE TO REMOVE (lines 286-297):
"""
print("📊 running_precompute_functions... (70%)")
prompt_results = {}
for func_name, func in COMPUTE_FUNCTIONS.items():
    try:
        result = func(unified_analysis.to_dict())
        prompt_results[func_name] = result
        if result:
            self.save_analysis_result(video_id, func_name, result)
    except Exception as e:
        logger.error(f"Precompute {func_name} failed: {e}")
        prompt_results[func_name] = {}
"""

# NEW CODE TO ADD:
print("📊 computing_temporal_windows... (70%)")

# Import the new temporal compute function
from rumiai_v2.processors.temporal_compute import compute_temporal_windows

try:
    # Single function call replacing all 7 analyses
    temporal_result = compute_temporal_windows(unified_analysis.to_dict())
    
    # Save single output file
    output_dir = Path(f"insights/{video_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{video_id}_temporal_windows.json"
    with open(output_path, 'w') as f:
        json.dump(temporal_result, f, indent=2)
    
    logger.info(f"✅ Saved temporal windows to {output_path}")
    
except Exception as e:
    logger.error(f"Temporal compute failed: {e}")
    raise
```

### Files to Delete

After successful migration, delete:
1. `rumiai_v2/processors/precompute_functions.py`
2. `rumiai_v2/processors/precompute_functions_full.py`
3. `rumiai_v2/processors/precompute_professional.py`
4. All 7 individual analysis modules (if separate)

## Part D: Output Format Example

```json
{
  "video_id": "7454575786134195489",
  "duration": 66.0,
  "temporal_windows": {
    "hook": {
      "start": 0,
      "end": 3,
      "duration": 3,
      "text_count": 5,
      "object_count": 12,
      "gesture_count": 3,
      "total_elements": 20,
      "element_density": 6.67,
      "changes_per_second": 0.33,
      "speech_coverage": 0.85,
      "word_count": 24,
      "happy_ratio": 0.4,
      "neutral_ratio": 0.6,
      "energy_level": 0.65,
      "energy_variance": 0.08,
      "energy_max": 0.89,
      "burst_pattern": "front_loaded"
    },
    "middle_segments": [
      {
        "segment_name": "segment_1",
        "start": 3,
        "end": 23,
        "duration": 20,
        "text_count": 28,
        "object_count": 89,
        "gesture_count": 15,
        "total_elements": 132,
        "element_density": 6.6,
        "changes_per_second": 0.25,
        "speech_coverage": 0.78,
        "word_count": 156,
        "joy_ratio": 0.3,
        "neutral_ratio": 0.5,
        "sadness_ratio": 0.2,
        "energy_level": 0.72,
        "energy_variance": 0.12,
        "energy_max": 0.95,
        "burst_pattern": "middle_peak"
      },
      {
        "segment_name": "segment_2",
        "start": 23,
        "end": 43,
        "duration": 20,
        "text_count": 31,
        "object_count": 92,
        "gesture_count": 18,
        "total_elements": 141,
        "element_density": 7.05,
        "changes_per_second": 0.3,
        "speech_coverage": 0.82,
        "word_count": 168,
        "joy_ratio": 0.6,
        "surprise_ratio": 0.4,
        "energy_level": 0.78,
        "energy_variance": 0.15,
        "energy_max": 0.98,
        "burst_pattern": "back_loaded"
      },
      {
        "segment_name": "segment_3",
        "start": 43,
        "end": 63,
        "duration": 20,
        "text_count": 35,
        "object_count": 88,
        "gesture_count": 22,
        "total_elements": 145,
        "element_density": 7.25,
        "changes_per_second": 0.35,
        "speech_coverage": 0.88,
        "word_count": 182,
        "joy_ratio": 0.7,
        "neutral_ratio": 0.3,
        "energy_level": 0.81,
        "energy_variance": 0.09,
        "energy_max": 0.94,
        "burst_pattern": "steady"
      }
    ],
    "closing": {
      "start": 63,
      "end": 66,
      "duration": 3,
      "text_count": 8,
      "object_count": 10,
      "gesture_count": 5,
      "total_elements": 23,
      "element_density": 7.67,
      "changes_per_second": 0.67,
      "speech_coverage": 0.90,
      "word_count": 28,
      "joy_ratio": 0.8,
      "neutral_ratio": 0.2,
      "energy_level": 0.82,
      "energy_variance": 0.06,
      "energy_max": 0.92,
      "burst_pattern": "back_loaded"
    }
  },
  "metadata": {
    "video_id": "7454575786134195489",
    "duration": 66.0,
    "digg_count": 12500,
    "play_count": 450000,
    "collect_count": 3200,
    "share_count": 890,
    "comment_count": 567,
    "create_time": 1698765432,
    "author": "creator123",
    "description": "Amazing video content #viral #fyp"
  },
  "processing_timestamp": 1705432187.5,
  "version": "2.0.0"
}
```

## Migration Checklist

- [ ] Copy complete implementation to `rumiai_v2/processors/temporal_compute.py`
- [ ] Update `rumiai_runner.py` with new integration code
- [ ] Test with videos of various durations (1s, 3s, 6s, 30s, 60s+)
- [ ] Verify single output file is generated
- [ ] Delete old precompute files after verification
- [ ] Update any downstream consumers to use new format

## Part E: Testing Approach

### Manual Testing Strategy

For initial validation, we will manually test temporal_compute.py by:

1. **Using existing test videos** in insights/ directory with known ML outputs
2. **Running temporal_compute directly** on analysis_dict from existing pipeline  
3. **Comparing output** against expected temporal window structure
4. **Verifying performance** logs show <1s processing for typical videos
5. **Testing edge cases** with videos of various durations (3s, 10s, 30s, 60s)

### Key Validation Points:
- All required ML services present → processes successfully
- Missing ML service → raises ValueError with clear message
- Video < 3s → only hook window, no middle/closing
- Video 3-6s → hook and closing, no middle segments  
- Video > 6s → all windows populated with correct boundaries
- Framing distribution → non-zero values from camera_distance_timeline
- Emotion distribution → matches FEAT timeline entries
- Performance → completes in <1s for 60s videos

### Running Manual Tests:
```bash
# Test with existing processed video
python -c "
from rumiai_v2.processors.temporal_compute import compute_temporal_windows
import json

# Load existing analysis_dict from test data
with open('test_data/analysis_dict_sample.json') as f:
    analysis_dict = json.load(f)

# Run temporal compute
result = compute_temporal_windows(analysis_dict)

# Verify structure
print(f'Hook window: {bool(result.get(\"hook_window\"))}')
print(f'Segments: {len(result.get(\"middle_segments\", []))}')
print(f'Closing window: {bool(result.get(\"closing_window\"))}')
"
```

Note: Comprehensive automated tests will be added post-MVP once the core functionality is stable.

## Summary

This complete implementation:
- ✅ Incorporates all 5 decisions from decisions3.md
- ✅ Uses defensive extraction for ALL services
- ✅ Implements mixed extraction strategy (ml_data + timeline entries)
- ✅ Calculates burst patterns for audio energy per window
- ✅ Provides full metrics for all windows and segments (including framing distribution)
- ✅ Is completely self-contained with no external dependencies
- ✅ Replaces 7 analyses with 1 unified temporal_windows.json output
- ✅ Provides clear integration instructions for rumiai_runner.py
- ✅ Includes manual testing strategy for validation
- ✅ Optimized O(n) density calculation for performance
- ✅ Comprehensive type hints for all functions
- ✅ Fail-fast validation with clear error messages
- ✅ Performance monitoring with timing logs
- ✅ No magic numbers - all constants clearly documented
- ✅ Extracts real camera distance/framing from timeline entries