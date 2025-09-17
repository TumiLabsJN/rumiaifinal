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
    
    # Check direct format - return full data if present
    if 'rms_frames' in audio_data or 'energy_level_windows' in audio_data:
        return audio_data
    
    # Check nested format
    if 'data' in audio_data:
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

    # Store raw timeline entries for use in process_segment
    timelines['timeline'] = {'entries': timeline_entries}

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
    
    # Face data from timeline entries (MediaPipe face detections)
    face_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'face':
            face_timeline.append({
                'timestamp': entry.get('start', 0),
                'bbox': entry.get('data', {}).get('bbox', {}),
                'confidence': entry.get('data', {}).get('confidence', 0)
            })
    timelines['face_timeline'] = face_timeline
    
    # Text overlay data from timeline entries (OCR-detected text)
    text_entry_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'text':
            text_entry_timeline.append({
                'timestamp': entry.get('start', 0),
                'data': {'text': entry.get('data', {}).get('text', '')},
                'source': 'timeline'
            })
    
    # Sticker data from timeline entries (OCR-detected stickers)
    sticker_entry_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'sticker':
            sticker_entry_timeline.append({
                'timestamp': entry.get('start', 0),
                'sticker_id': entry.get('data', {}).get('sticker_id', ''),
                'source': 'timeline'
            })
    
    # Camera distance/framing from timeline entries (calculated from face bbox by timeline_builder)
    # NOTE: Currently not being created by timeline_builder, so will be empty
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
    
    # OCR text and stickers - merge timeline entries with any ml_data
    # Timeline entries are preferred (already processed with timestamps)
    # But keep ml_data extraction as fallback for older data
    timelines['text_overlay_timeline'] = text_entry_timeline  # From timeline entries above
    timelines['sticker_timeline'] = sticker_entry_timeline    # From timeline entries above
    
    # Fallback: If no timeline entries, try ml_data
    if not timelines['text_overlay_timeline']:
        ocr_data = extract_ocr_data(ml_data)
        timelines['text_overlay_timeline'] = [
            {'timestamp': ann.get('timestamp', 0), 'text': ann.get('text', '')}
            for ann in ocr_data.get('textAnnotations', [])
        ]
    if not timelines['sticker_timeline']:
        ocr_data = extract_ocr_data(ml_data) if 'ocr_data' not in locals() else ocr_data
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
# TEXT OVERLAY PROCESSING
# ============================================

def process_text_overlays(text_timeline: List[Dict], start: float, end: float, 
                         duration: float, speech_segments: List[Dict] = None) -> Dict[str, Any]:
    """
    Process text overlay detections to compute meaningful metrics.
    Phase 1: Separates marketing overlays from speech captions using temporal patterns + speech matching.
    
    Design decisions:
    1. Temporal clustering: Group texts within 1.0s of each other
    2. Pattern detection: Rate-based (>0.5/sec = captions, <0.2/sec = overlays)
    3. Pattern-weighted: Trust temporal pattern over speech matching
    4. Breaking change: Removes unique_text_count (was fundamentally wrong)
    """
    import numpy as np
    import re
    
    # Classification thresholds (based on typical behavior, to be validated)
    CAPTION_CHANGE_RATE = 0.5  # Captions typically change >0.5 times/sec with speech
    OVERLAY_CHANGE_RATE = 0.2  # Marketing overlays typically change <0.2 times/sec
    CLUSTER_GAP_THRESHOLD = 1.0  # Texts >1.0s apart are considered separate events
    PERSIST_BUFFER = 0.5  # Seconds to assume text persists after last detection
    
    def normalize_text(text: str) -> str:
        """Normalize text for grouping similar OCR detections."""
        # Convert to lowercase
        text = text.lower()
        # Remove emojis and special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def calculate_speech_overlap(text: str, timestamp: float, speech_segments: List[Dict]) -> float:
        """Calculate % overlap between text and speech at given timestamp."""
        if not speech_segments:
            return 0.0
        
        # Find speech segments overlapping this timestamp
        for segment in speech_segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', seg_start + 1)
            if seg_start <= timestamp <= seg_end:
                # Get segment text
                segment_text = segment.get('text', '').lower()
                if not segment_text:
                    continue
                
                # Normalize both texts
                text_normalized = normalize_text(text)
                segment_normalized = normalize_text(segment_text)
                
                # Calculate word overlap
                text_words = set(text_normalized.split())
                segment_words = set(segment_normalized.split())
                
                if not text_words:
                    return 0.0
                
                overlap_words = text_words.intersection(segment_words)
                overlap_ratio = len(overlap_words) / len(text_words)
                return overlap_ratio
        
        return 0.0
    
    # Note: Removed cluster_texts_temporally and analyze_cluster_pattern
    # Solution 5 uses speech-first approach, not temporal clustering
    
    # Handle edge cases
    if not text_timeline or speech_segments is None:
        speech_segments = []
    
    # Step 1: Filter texts in window
    window_texts = []
    for entry in text_timeline:
        timestamp = entry.get('timestamp', 0)
        if start <= timestamp < end:
            # Text is nested in 'data' field
            text_content = entry.get('data', {}).get('text', '')
            if text_content and normalize_text(text_content):  # Skip empty after normalization
                window_texts.append(entry)
    
    # Early return if no texts in this window
    if not window_texts:
        return {
            'overlay_unique_count': 0,
            'overlay_coverage': 0.0,
            'overlay_persistence': 0.0,
            'has_captions': False
        }
    
    # Step 2: Speech-first classification on individual texts
    # Calculate speech overlap for EVERY text first
    for entry in window_texts:
        text_content = entry.get('data', {}).get('text', '')
        timestamp = entry.get('timestamp', 0)
        entry['speech_overlap'] = calculate_speech_overlap(text_content, timestamp, speech_segments)
        entry['normalized_text'] = normalize_text(text_content)
    
    # Step 3: Group by confidence levels
    # Define thresholds for speech-first classification
    HIGH_SPEECH_THRESHOLD = 0.7  # >70% = definitely caption
    LOW_SPEECH_THRESHOLD = 0.3   # <30% = definitely overlay
    
    high_confidence_captions = []  
    high_confidence_overlays = []  
    uncertain_texts = []            
    
    for entry in window_texts:
        overlap = entry['speech_overlap']
        if overlap > HIGH_SPEECH_THRESHOLD:
            high_confidence_captions.append(entry)
        elif overlap < LOW_SPEECH_THRESHOLD:
            high_confidence_overlays.append(entry)
        else:
            uncertain_texts.append(entry)
    
    # Step 4: For uncertain texts, use persistence as tiebreaker
    # Group uncertain texts by content to check persistence
    uncertain_by_content = {}
    for entry in uncertain_texts:
        text_key = entry['normalized_text']
        if text_key not in uncertain_by_content:
            uncertain_by_content[text_key] = []
        uncertain_by_content[text_key].append(entry)
    
    # Classify uncertain texts based on persistence
    for text_key, entries in uncertain_by_content.items():
        if len(entries) >= 2:
            # Text appears multiple times
            timestamps = [e.get('timestamp', 0) for e in entries]
            time_span = max(timestamps) - min(timestamps)
            
            if time_span > 2.0:
                # Persists across time = likely overlay
                high_confidence_overlays.extend(entries)
            else:
                # Multiple appearances but close together = likely caption repetition
                high_confidence_captions.extend(entries)
        else:
            # Single appearance of uncertain text
            # Use 0.5 threshold as final fallback
            entry = entries[0]
            if entry['speech_overlap'] > 0.5:
                high_confidence_captions.append(entry)
            else:
                high_confidence_overlays.append(entry)
    
    # Step 5: Final classification
    overlay_texts = high_confidence_overlays
    caption_texts = high_confidence_captions
    
    # Step 4: Calculate metrics for overlays and captions separately
    overlay_groups = {}
    caption_groups = {}
    
    for entry in overlay_texts:
        text_content = entry.get('data', {}).get('text', '')
        normalized = normalize_text(text_content)
        if normalized not in overlay_groups:
            overlay_groups[normalized] = []
        overlay_groups[normalized].append(entry.get('timestamp', 0))
    
    for entry in caption_texts:
        text_content = entry.get('data', {}).get('text', '')
        normalized = normalize_text(text_content)
        if normalized not in caption_groups:
            caption_groups[normalized] = []
        caption_groups[normalized].append(entry.get('timestamp', 0))
    
    # Combine for processing metrics
    all_texts = overlay_texts + caption_texts
    text_groups = {**overlay_groups, **caption_groups}
    
    # Calculate counts
    overlay_unique_count = len(overlay_groups)
    caption_unique_count = len(caption_groups)
    
    # Handle empty case (shouldn't happen after early return, but be safe)
    if len(text_groups) == 0:
        return {
            'overlay_unique_count': 0,
            'overlay_coverage': 0.0,
            'overlay_persistence': 0.0,
            'has_captions': False
        }
    
    # Calculate lifespan of each unique text (accounting for gaps)
    text_lifespans = {}
    text_appearances = {}  # Track separate appearances
    
    for text, timestamps in text_groups.items():
        timestamps.sort()
        appearances = []
        current_appearance = [timestamps[0]]
        
        for i in range(1, len(timestamps)):
            # Check if gap is too large
            if timestamps[i] - timestamps[i-1] > CLUSTER_GAP_THRESHOLD:
                # End current appearance, start new one
                appearances.append(current_appearance)
                current_appearance = [timestamps[i]]
            else:
                current_appearance.append(timestamps[i])
        appearances.append(current_appearance)
        
        text_appearances[text] = appearances
        
        # Calculate total lifespan (sum of all appearances)
        total_lifespan = 0
        for appearance in appearances:
            first = appearance[0]
            last = appearance[-1]
            # Add buffer for each appearance, but don't exceed segment
            lifespan = min(last + PERSIST_BUFFER, end) - first
            total_lifespan += lifespan
        
        text_lifespans[text] = total_lifespan
    
    # Note: Removed event processing for max_simultaneous_texts and text_coverage
    # These metrics were redundant with other features
    
    # Calculate overlay-specific metrics
    overlay_coverage = 0.0
    overlay_persistence = 0.0
    if overlay_groups:
        overlay_lifespans = []
        for text, timestamps in overlay_groups.items():
            if text in text_lifespans:
                overlay_lifespans.append(text_lifespans[text])
        overlay_persistence = sum(overlay_lifespans) / len(overlay_lifespans) if overlay_lifespans else 0.0
        
        # Calculate overlay coverage
        overlay_events = []
        for text, timestamps in overlay_groups.items():
            sorted_ts = sorted(timestamps)
            for i in range(len(sorted_ts)):
                if i == 0 or sorted_ts[i] - sorted_ts[i-1] > CLUSTER_GAP_THRESHOLD:
                    overlay_events.append((sorted_ts[i], 'appear'))
                if i == len(sorted_ts) - 1 or (i < len(sorted_ts) - 1 and sorted_ts[i+1] - sorted_ts[i] > CLUSTER_GAP_THRESHOLD):
                    overlay_events.append((min(sorted_ts[i] + PERSIST_BUFFER, end), 'disappear'))
        
        overlay_events.sort()
        overlay_active = 0
        overlay_time = 0.0
        prev_time = start
        for event_time, event_type in overlay_events:
            if overlay_active > 0:
                overlay_time += (event_time - prev_time)
            if event_type == 'appear':
                overlay_active += 1
            else:
                overlay_active = max(0, overlay_active - 1)
            prev_time = event_time
        overlay_coverage = overlay_time / duration if duration > 0 else 0.0
    
    # Simplified caption metric - just binary presence
    has_captions = len(caption_groups) > 0
    
    return {
        # Overlay metrics (marketing text only)
        'overlay_unique_count': overlay_unique_count,
        'overlay_coverage': float(overlay_coverage),
        'overlay_persistence': float(overlay_persistence),
        
        # Caption presence (simplified to binary)
        'has_captions': has_captions
        
        # REMOVED: unique_text_count (was fundamentally wrong)
        # REMOVED: caption metrics (redundant with speech_coverage)
        # REMOVED: max_simultaneous_texts, text_coverage (redundant/not actionable)
        # REMOVED: text_appearance_count, avg_text_lifespan, text_change_count (over-engineered)
    }

# ============================================
# SEGMENT PROCESSING (Decision 4)
# ============================================

def calculate_speech_metrics_for_window(speech_segments, start, end, duration):
    """
    Calculate speech coverage and word count for a temporal window.
    Uses proportional calculation for segments that partially overlap.

    This is a module-level function for testability and potential reuse.

    Args:
        speech_segments: List of speech segments from Whisper
        start: Window start time in seconds
        end: Window end time in seconds
        duration: Window duration (end - start)

    Returns:
        tuple: (speech_coverage, word_count)

    Raises:
        ValueError: If segment data is corrupted (invalid timestamps, zero duration)
                   or if duration parameter doesn't match end - start
    """
    # Validate duration parameter
    expected_duration = end - start
    if abs(duration - expected_duration) > 0.001:  # Allow tiny floating point differences
        raise ValueError(f"Duration {duration} doesn't match end-start {expected_duration}")

    # No speech is valid - return zeros
    if not speech_segments:
        return 0.0, 0

    total_speech_duration = 0.0
    total_word_count = 0.0

    for segment in speech_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)
        seg_text = segment.get('text', '')

        # Validate segment data - fail fast on corruption
        if seg_start > seg_end:
            raise ValueError(f"Corrupted segment: start {seg_start} > end {seg_end}")
        if seg_start == seg_end:
            raise ValueError(f"Zero-duration segment at {seg_start}s - indicates corrupted data")
        if seg_start < 0 or seg_end < 0:
            raise ValueError(f"Invalid negative timestamps: {seg_start} to {seg_end}")

        # Handle None text gracefully (empty speech segment)
        if seg_text is None:
            seg_text = ''

        # Check if segment overlaps with window at all
        if seg_start < end and seg_end > start:
            # Calculate the overlap duration
            overlap_start = max(seg_start, start)
            overlap_end = min(seg_end, end)
            overlap_duration = overlap_end - overlap_start

            # Calculate what proportion of the segment is in our window
            segment_duration = seg_end - seg_start
            # segment_duration is guaranteed > 0 due to validation above
            proportion_in_window = overlap_duration / segment_duration

            # Add the overlap duration to total
            total_speech_duration += overlap_duration

            # Count proportional words
            segment_words = len(seg_text.split())
            words_in_window = segment_words * proportion_in_window
            total_word_count += words_in_window

    # Calculate coverage as percentage of window
    raw_coverage = total_speech_duration / duration if duration > 0 else 0

    # Cap at 100% but log if exceeded (indicates overlapping segments)
    if raw_coverage > 1.0:
        logger.debug(f"Speech coverage {raw_coverage:.1%} exceeds 100% - overlapping segments detected")
        # DEBUG level chosen because:
        # 1. Function handles gracefully by capping at 100%
        # 2. Overlapping segments can be normal (interviews, multiple speakers)
        # 3. Doesn't clutter production logs with non-critical warnings

    speech_coverage = min(1.0, raw_coverage)

    # Round word count to nearest integer
    word_count = int(round(total_word_count))

    return speech_coverage, word_count

def calculate_speech_content_indicators(speech_segments, start, end, duration):
    """
    Calculate speech content indicators for a temporal window.

    This function follows the same pattern as calculate_speech_metrics_for_window()
    and other calculate_* functions in temporal_compute.py.

    Args:
        speech_segments: List of speech segments from Whisper
        start: Window start time in seconds
        end: Window end time in seconds
        duration: Window duration (end - start)

    Returns:
        dict: Speech content indicators (has_greeting, has_question,
              has_instruction, has_speech_cta)
    """
    # Collect all text in window
    window_text = ""

    for segment in speech_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)

        # Skip segments outside window
        if seg_end <= start or seg_start >= end:
            continue

        seg_text = segment.get('text', '')
        if seg_text:
            window_text += " " + seg_text

    # Prepare for analysis
    text_lower = window_text.lower().strip()

    # If no text in window, return all zeros
    if not text_lower:
        return {
            'has_greeting': 0,
            'has_question': 0,
            'has_instruction': 0,
            'has_speech_cta': 0
        }

    # Content indicators (binary features)
    greetings = ['hey', 'hello', 'hi ', 'welcome', "what's up", 'good morning',
                 'good afternoon', 'good evening']
    # Check greetings in first 50 chars of window (more likely at start)
    has_greeting = 1 if any(g in text_lower[:50] for g in greetings) else 0

    questions = ['how ', 'what ', 'why ', 'when ', 'where ', 'can you', 'do you',
                 'are you', 'is it', 'have you']
    has_question = 1 if ('?' in window_text or any(q in text_lower for q in questions)) else 0

    instructions = ['first ', 'then ', 'next ', 'step ', 'make sure', "don't forget",
                   'remember to', 'be sure to', 'start by', 'follow']
    has_instruction = 1 if any(i in text_lower for i in instructions) else 0

    # Call-to-action patterns
    cta_patterns = ['subscribe', 'follow', 'like', 'comment', 'share', 'click',
                   'link in bio', 'check out', 'swipe up', 'tap the', 'hit the']
    has_speech_cta = 1 if any(cta in text_lower for cta in cta_patterns) else 0

    return {
        'has_greeting': has_greeting,
        'has_question': has_question,
        'has_instruction': has_instruction,
        'has_speech_cta': has_speech_cta
    }

def calculate_max_unique_persons(segment_objects, confidence_threshold=0.5):
    """
    Calculate the maximum number of unique persons visible at any point in the segment.

    This counts unique persons by finding the maximum number of unique trackIds
    at any single timestamp, after filtering by confidence.

    Args:
        segment_objects: List of object detections from YOLO
        confidence_threshold: Minimum confidence to consider a detection valid

    Returns:
        int: Maximum number of unique persons visible at any point
    """
    # Filter for person detections with sufficient confidence
    person_detections = [
        obj for obj in segment_objects
        if (obj.get('className') == 'person' or obj.get('label') == 'person')
        and obj.get('confidence', 0) >= confidence_threshold
    ]

    if not person_detections:
        return 0

    # Group by timestamp
    from collections import defaultdict
    detections_by_timestamp = defaultdict(list)

    for detection in person_detections:
        timestamp = detection.get('timestamp', 0)
        track_id = detection.get('trackId', f"unknown_{id(detection)}")
        detections_by_timestamp[timestamp].append(track_id)

    # Find maximum unique persons at any timestamp
    max_persons = 0
    for timestamp, track_ids in detections_by_timestamp.items():
        unique_persons = len(set(track_ids))
        max_persons = max(max_persons, unique_persons)

    return max_persons

def calculate_gaze_variance(timeline_entries: List[Dict], start: float, end: float) -> float:
    """
    Calculate variance in eye contact scores within a temporal window.

    Args:
        timeline_entries: List of timeline entries containing gaze data
        start: Window start time in seconds
        end: Window end time in seconds

    Returns:
        Variance of eye contact scores (0.0 if insufficient data)
    """
    import statistics

    eye_contact_scores = []

    # Collect eye contact scores from gaze entries in the window
    for entry in timeline_entries:
        if entry.get('entry_type') == 'gaze':
            entry_start = entry.get('start', 0)
            if start <= entry_start <= end:
                eye_contact = entry.get('data', {}).get('eye_contact', 0)
                if eye_contact is not None:  # Valid measurement
                    eye_contact_scores.append(eye_contact)

    # Calculate variance if we have enough data points
    if len(eye_contact_scores) > 1:
        return statistics.variance(eye_contact_scores)

    # Return 0 if insufficient data for variance calculation
    return 0.0

def process_segment(seg_bounds: Dict[str, float], timelines: Dict[str, Any],
                   audio_data: Dict[str, Any], video_duration: float) -> Dict[str, Any]:
    """
    Process segment with FULL metrics matching hook/closing windows.
    Implements Decision 4 from decisions3.md - ML consistency across all windows.
    Includes all P0 required metrics from P0TemporalWindows.md.
    """
    start = seg_bounds['start']
    end = seg_bounds['end']
    duration = end - start
    
    # Get speech segments from timelines
    speech_segments = timelines.get('speech_segments', [])
    
    # Process text overlays with advanced metrics (Phase 1: with speech segments)
    text_metrics = process_text_overlays(
        timelines.get('text_overlay_timeline', []),
        start, end, duration,
        speech_segments  # Pass speech segments for overlay vs caption classification
    )
    
    # Filter other data to segment bounds
    # segment_stickers removed - see StickersProblem.md
    # segment_stickers = [s for s in timelines.get('sticker_timeline', [])
    #                    if start <= s.get('timestamp', 0) < end]
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
    # Speech segments handled by calculate_speech_metrics_for_window function
    
    # Calculate all P0 required counts
    # sticker_count removed - see StickersProblem.md for why
    object_count = len(segment_objects)
    # MVP: Count unique persons (maximum at any timestamp)
    person_count = calculate_max_unique_persons(segment_objects)
    gesture_count = len(segment_gestures)
    expression_count = len(segment_expressions)
    scene_count = len(segment_scenes)
    
    # P0 spec: element_count is sum of ALL 6 types
    # Use overlay + caption count instead of removed unique_text_count
    total_unique_texts = text_metrics.get('overlay_unique_count', 0) + text_metrics.get('caption_unique_count', 0)
    total_elements = (total_unique_texts + 
                     object_count + gesture_count + expression_count + scene_count)
    
    # Calculate scene durations for this segment (P0 requirement)
    scene_durations = []
    # Get all scene changes in entire video to calculate durations properly
    all_scenes = timelines.get('scene_change_timeline', [])
    if all_scenes:
        sorted_all_scenes = sorted(all_scenes, key=lambda x: x.get('timestamp', 0))
        # Find scenes that overlap with this segment
        for i, scene in enumerate(sorted_all_scenes):
            scene_start = scene.get('timestamp', 0)
            # Find next scene or use video end
            if i < len(sorted_all_scenes) - 1:
                scene_end = sorted_all_scenes[i + 1].get('timestamp', 0)
            else:
                scene_end = video_duration  # Use video duration as end of last scene
            
            # Check if this scene overlaps with our segment
            if scene_end > start and scene_start < end:
                # Calculate the portion of the scene within our segment
                overlap_start = max(scene_start, start)
                overlap_end = min(scene_end, end)
                scene_duration = overlap_end - overlap_start
                if scene_duration > 0:
                    scene_durations.append(scene_duration)
    
    # Calculate density metrics
    avg_density = total_elements / duration if duration > 0 else 0
    changes_per_second = scene_count / duration if duration > 0 else 0
    
    # Calculate density extremes (P0 requirement)
    # Single-pass bucketing for O(n) performance instead of O(n*m)
    interval_count = max(1, int(end - start))
    densities = [0] * interval_count
    
    # Single pass through each element type, bucketing by second
    # For text, we need to track unique texts per second
    text_timeline = timelines.get('text_overlay_timeline', [])
    for t in text_timeline:
        timestamp = t.get('timestamp', 0)
        if start <= timestamp < end:
            second = int(timestamp - start)
            if 0 <= second < interval_count:
                densities[second] += 1
    
    # Stickers removed from density calculation - see StickersProblem.md
    # for s in segment_stickers:
    #     second = int(s.get('timestamp', 0) - start)
    #     if 0 <= second < interval_count:
    #         densities[second] += 1
    
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
    
    # Calculate speech metrics using proportional approach
    speech_coverage, word_count = calculate_speech_metrics_for_window(
        speech_segments, start, end, duration
    )

    # Calculate speech content indicators (NEW)
    speech_content_indicators = calculate_speech_content_indicators(
        speech_segments, start, end, duration
    )

    # Calculate gaze variance (NEW)
    gaze_variance = calculate_gaze_variance(
        timelines.get('timeline', {}).get('entries', []), start, end
    )

    # Calculate emotion distribution
    # Note: Emotion labels are already standardized by timeline_builder.py
    # We can safely count emotions directly since the timeline builder
    # has already mapped any FEAT variations to standard labels.
    
    # Initialize all 7 emotions to ensure consistent features for ML
    all_emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']
    emotion_counts = {emotion: 0 for emotion in all_emotions}
    
    for e in segment_expressions:
        emotion = e.get('emotion', 'neutral')
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
    
    total_emotions = len(segment_expressions)
    emotion_dist = {
        f'{emotion}_ratio': count / total_emotions if total_emotions > 0 else 0
        for emotion, count in emotion_counts.items()
    }
    
    # Calculate framing distribution from face bbox sizes
    # Since camera_distance entries don't exist, calculate from face entries
    # close (>25% frame), medium (8-25%), wide (<8%), none (no face)
    framing_counts = {'close': 0, 'medium': 0, 'wide': 0, 'none': 0}
    
    # Get face entries for this segment
    segment_faces = [f for f in timelines.get('face_timeline', [])
                    if start <= f.get('timestamp', 0) < end]
    
    for face in segment_faces:
        bbox = face.get('bbox', {})
        if bbox:
            # Calculate face area as percentage of frame
            face_area = bbox.get('width', 0) * bbox.get('height', 0) * 100
            
            if face_area > 25:
                framing_counts['close'] += 1
            elif face_area > 8:
                framing_counts['medium'] += 1
            elif face_area > 0:
                framing_counts['wide'] += 1
            else:
                framing_counts['none'] += 1
        else:
            framing_counts['none'] += 1
    
    # If no faces in segment, count as 'none'
    if not segment_faces:
        framing_counts['none'] = 1
    
    total_camera_frames = sum(framing_counts.values()) if sum(framing_counts.values()) > 0 else 1
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
    
    # Calculate scene duration metrics (P0 requirement)
    if scene_durations:
        shortest_scene = float(min(scene_durations))
        longest_scene = float(max(scene_durations))
        scene_duration_variance = float(np.var(scene_durations)) if len(scene_durations) > 1 else 0.0
    else:
        # No scenes in this segment
        shortest_scene = 0.0
        longest_scene = 0.0
        scene_duration_variance = 0.0
    
    return {
        'start': start,
        'end': end,
        'duration': duration,
        # Text overlay metrics (replacing text_count)
        **text_metrics,  # Unpacks all 6 text metrics
        # 'sticker_count': sticker_count,  # Removed - see StickersProblem.md
        'object_count': object_count,
        'person_count': person_count,  # MVP: Person-specific count
        'gesture_count': gesture_count,
        'expression_count': expression_count,
        'scene_count': scene_count,
        'element_count': total_elements,  # Sum of all 6 types
        # P0 density extremes
        'max_density': max_density,
        'min_density': min_density,
        'avg_density': avg_density,
        # P1 scene duration metrics
        'shortest_scene': shortest_scene,
        'longest_scene': longest_scene,
        # P2 scene variance
        'scene_duration_variance': scene_duration_variance,
        # Other metrics
        'changes_per_second': changes_per_second,
        'speech_coverage': speech_coverage,
        'word_count': word_count,
        **speech_content_indicators,  # Unpack speech content indicators (NEW)
        'gaze_variance': gaze_variance,  # Gaze variance (NEW)
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
    # video_id is at top level, not in metadata
    video_id = analysis_dict.get('video_id', metadata.get('id', 'unknown'))
    
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
        hook_data = process_segment(hook_bounds, timelines, audio_data, video_duration)
        
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
            seg_data = process_segment(seg_bounds, timelines, audio_data, video_duration)
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
        closing_data = process_segment(closing_bounds, timelines, audio_data, video_duration)
        
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
        'digg_count': metadata.get('likes', 0),  # Field name changed
        'play_count': metadata.get('views', 0),  # Field name changed
        'collect_count': metadata.get('saves', 0),  # Field name changed
        'share_count': metadata.get('shares', 0),
        'comment_count': metadata.get('comments', 0),
        'create_time': metadata.get('createTime', metadata.get('createTimeISO', '')),
        'author': metadata.get('author', {}).get('uniqueId', metadata.get('author', {}).get('name', '')),
        'description': metadata.get('description', '')
    }

    # Add gender detection data if available
    gender_data = ml_data.get('deepface_gender', {})
    if gender_data:
        calculated_metadata['gender_detection'] = {
            'gender': gender_data.get('gender'),
            'confidence': gender_data.get('confidence', 0.0),
            'method': gender_data.get('method', 'deepface')
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
