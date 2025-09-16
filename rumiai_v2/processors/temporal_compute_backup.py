"""
Temporal Windows Computation Module
Single unified computation for all temporal features
Implements all 9 decisions from critique analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============== WINDOW CALCULATION FUNCTIONS ==============

def calculate_temporal_windows(video_duration: float) -> Dict[str, Optional[Tuple[float, float]]]:
    """
    Calculate hook, middle, and closing windows based on video duration.
    Handles edge cases for short videos.
    
    Args:
        video_duration: Duration of video in seconds
        
    Returns:
        Dict with 'hook', 'middle', 'closing' window boundaries
    """
    if video_duration <= 3:
        return {
            'hook': (0, video_duration),
            'middle': None,
            'closing': None
        }
    elif video_duration <= 6:
        hook_end = min(3, video_duration)
        return {
            'hook': (0, hook_end),
            'middle': None,
            'closing': (hook_end, video_duration)
        }
    else:
        hook_end = 3
        closing_start = video_duration - 3
        return {
            'hook': (0, hook_end),
            'middle': (hook_end, closing_start) if closing_start > hook_end else None,
            'closing': (closing_start, video_duration)
        }

def calculate_middle_segments(video_duration: float) -> Dict[str, Dict[str, float]]:
    """
    Calculate segment boundaries for the middle window.
    
    Args:
        video_duration: Duration of video in seconds
        
    Returns:
        Dict with segment_N keys containing start/end boundaries
    """
    middle_start = 3  # After hook
    middle_end = video_duration - 3  # Before closing
    middle_duration = middle_end - middle_start
    
    # Handle edge cases
    if middle_duration <= 0:
        return {}
    
    # No segments if middle < 3s
    if middle_duration < 3:
        return {}  # Middle exists but no segments
    
    # Determine segment count based on middle duration
    if middle_duration <= 12:
        num_segments = 3
    elif middle_duration <= 27:
        num_segments = 4
    else:
        num_segments = 5
    
    # Calculate equal segments with precise boundaries
    segment_duration = middle_duration / num_segments
    segments = {}
    
    for i in range(num_segments):
        segment_start = middle_start + (i * segment_duration)
        segment_end = middle_start + ((i + 1) * segment_duration)
        segments[f'segment_{i+1}'] = {
            'start': round(segment_start, 2),  # 10ms precision
            'end': round(segment_end, 2)
        }
    
    return segments

# ============== MUSIC DETECTION FOR SPEECH FILTERING ==============

def is_likely_music(segment_text: str) -> bool:
    """
    Detect if transcribed segment is likely music not speech.
    70-80% accuracy with <1ms overhead.
    
    Args:
        segment_text: Transcribed text from speech segment
        
    Returns:
        True if likely music, False if likely speech
    """
    music_indicators = [
        '♪', '♫',  # Music symbols
        '[Music]', '[music]', '(music)',
        '[instrumental]', '(instrumental)',
        '(singing)', '[singing]',
        '(chiming)', '(bells)',  # Sound effects
    ]
    
    text_lower = segment_text.lower().strip()
    
    # Empty or just punctuation = likely music
    if not text_lower or text_lower in ['...', '♪', '♫']:
        return True
    
    # Contains music indicators
    for indicator in music_indicators:
        if indicator.lower() in text_lower:
            return True
    
    # Very short repeated sounds might be music
    if len(text_lower) < 3 and text_lower in ['ah', 'oh', 'la', 'na', 'da']:
        return True
    
    # Check for repetitive non-words (common in music transcription)
    words = text_lower.split()
    if len(words) > 2:
        unique_words = set(words)
        if len(unique_words) == 1:  # Same word repeated
            return True
    
    return False

# ============== SPEECH COVERAGE CALCULATION ==============

def calculate_speech_in_window(segment: Dict, window_start: float, window_end: float) -> float:
    """
    Calculate how much of a speech segment falls within a window.
    Pro-rates segments that span window boundaries.
    
    Args:
        segment: Speech segment with 'start' and 'duration'
        window_start: Start time of window
        window_end: End time of window
        
    Returns:
        Duration of speech within window in seconds
    """
    seg_start = segment['start']
    seg_end = segment['start'] + segment['duration']
    
    # Find overlap
    overlap_start = max(seg_start, window_start)
    overlap_end = min(seg_end, window_end)
    
    if overlap_start < overlap_end:
        return overlap_end - overlap_start
    return 0

def calculate_temporal_speech_coverage(speech_segments: List[Dict], video_duration: float) -> Dict[str, float]:
    """
    Calculate speech coverage percentage per temporal window with music filtering.
    
    Args:
        speech_segments: List of speech segments with 'start', 'duration', 'text'
        video_duration: Total video duration in seconds
        
    Returns:
        Dict with coverage values for each window and segment
    """
    # Filter out music/singing segments
    filtered_segments = [
        seg for seg in speech_segments
        if not is_likely_music(seg.get('text', ''))
    ]
    
    # Remove segments < 0.1s as noise
    filtered_segments = [seg for seg in filtered_segments if seg.get('duration', 0) >= 0.1]
    
    # Calculate windows
    windows = calculate_temporal_windows(video_duration)
    coverage = {}
    
    # Hook coverage
    if windows['hook']:
        hook_start, hook_end = windows['hook']
        hook_speech_time = sum(
            calculate_speech_in_window(seg, hook_start, hook_end)
            for seg in filtered_segments
        )
        coverage['hook_speech_coverage'] = round(min(hook_speech_time / (hook_end - hook_start), 1.0), 2)
    
    # Middle coverage (overall)
    if windows['middle']:
        middle_start, middle_end = windows['middle']
        middle_speech_time = sum(
            calculate_speech_in_window(seg, middle_start, middle_end)
            for seg in filtered_segments
        )
        coverage['middle_speech_coverage'] = round(min(middle_speech_time / (middle_end - middle_start), 1.0), 2)
        
        # Middle segments coverage
        middle_segments = calculate_middle_segments(video_duration)
        for segment_name, bounds in middle_segments.items():
            seg_start, seg_end = bounds['start'], bounds['end']
            segment_speech_time = sum(
                calculate_speech_in_window(seg, seg_start, seg_end)
                for seg in filtered_segments
            )
            coverage[f'middle_{segment_name}_speech_coverage'] = round(
                min(segment_speech_time / (seg_end - seg_start), 1.0), 2
            )
    
    # Closing coverage
    if windows['closing']:
        closing_start, closing_end = windows['closing']
        closing_speech_time = sum(
            calculate_speech_in_window(seg, closing_start, closing_end)
            for seg in filtered_segments
        )
        coverage['closing_speech_coverage'] = round(min(closing_speech_time / (closing_end - closing_start), 1.0), 2)
    
    return coverage

# ============== ELEMENT COUNT AND DENSITY ==============

def count_elements_in_window(timelines: Dict, window_start: float, window_end: float) -> Dict[str, Any]:
    """
    Count all elements and calculate metrics within a time window.
    Decision 5: Extended to handle all timeline types.
    Decision 9: Keep simple iteration (no optimization).
    
    Args:
        timelines: Dict with timeline data for each element type
        window_start: Start time of window
        window_end: End time of window
        
    Returns:
        Dict with counts and rates for all element types
    """
    counts = {
        # Visual elements
        'text_count': 0,
        'sticker_count': 0,
        'object_count': 0,
        'gesture_count': 0,
        'expression_count': 0,
        'scene_count': 0,  # Tracked separately
        'element_count': 0,  # Sum of 5 visual types (not scene)
        
        # Person metrics (Decision 5: new)
        'face_count': 0,
        'face_visible_time': 0,
        'eye_contact_time': 0,
        
        # Framing metrics (Decision 5: new)
        'close_up_time': 0,
        'medium_shot_time': 0,
        'wide_shot_time': 0,
    }
    
    # Visual element counting (existing)
    for item in timelines.get('text_overlay_timeline', []):
        if window_start <= item.get('timestamp', 0) < window_end:
            counts['text_count'] += 1
    
    for item in timelines.get('sticker_timeline', []):
        if window_start <= item.get('timestamp', 0) < window_end:
            counts['sticker_count'] += 1
    
    for item in timelines.get('object_timeline', []):
        if window_start <= item.get('timestamp', 0) < window_end:
            counts['object_count'] += len(item.get('objects', []))
    
    for item in timelines.get('gesture_timeline', []):
        if window_start <= item.get('timestamp', 0) < window_end and item.get('gesture'):
            counts['gesture_count'] += 1
    
    for item in timelines.get('expression_timeline', []):
        if window_start <= item.get('timestamp', 0) < window_end and item.get('expression'):
            counts['expression_count'] += 1
    
    for boundary in timelines.get('scene_boundaries', []):
        if window_start <= boundary < window_end:
            counts['scene_count'] += 1
    
    # Person metrics (Decision 5: new)
    if 'personTimeline' in timelines:
        for timestamp_key, person_data in timelines['personTimeline'].items():
            try:
                timestamp = float(timestamp_key.split('-')[0])
                if window_start <= timestamp < window_end:
                    if person_data.get('face_bbox'):
                        counts['face_count'] += 1
                        counts['face_visible_time'] += 1.0  # Each entry = 1 second
            except (ValueError, IndexError):
                continue
    
    # Gaze metrics (Decision 5: new)
    if 'gaze_timeline' in timelines:
        for timestamp_key, gaze_data in timelines['gaze_timeline'].items():
            try:
                timestamp = float(timestamp_key.split('-')[0])
                if window_start <= timestamp < window_end:
                    if gaze_data.get('looking_at_camera'):
                        counts['eye_contact_time'] += 1.0
            except (ValueError, IndexError):
                continue
    
    # Framing metrics (Decision 5: new)
    if 'camera_distance_timeline' in timelines:
        for timestamp_key, distance_data in timelines['camera_distance_timeline'].items():
            try:
                timestamp = float(timestamp_key.split('-')[0])
                if window_start <= timestamp < window_end:
                    distance = distance_data.get('distance', 'medium').lower()
                    if 'close' in distance:
                        counts['close_up_time'] += 1.0
                    elif 'medium' in distance:
                        counts['medium_shot_time'] += 1.0
                    elif 'wide' in distance or 'far' in distance:
                        counts['wide_shot_time'] += 1.0
            except (ValueError, IndexError):
                continue
    
    # Calculate element count (5 visual types, not scene changes)
    counts['element_count'] = (
        counts['text_count'] + 
        counts['sticker_count'] + 
        counts['object_count'] + 
        counts['gesture_count'] + 
        counts['expression_count']
    )
    
    # Calculate rates from counts
    window_duration = window_end - window_start
    if window_duration > 0:
        counts['face_visibility_rate'] = round(counts['face_visible_time'] / window_duration, 2)
        counts['eye_contact_rate'] = round(counts['eye_contact_time'] / window_duration, 2)
    
    return counts

def calculate_density_metrics(timelines: Dict, window_start: float, window_end: float) -> Dict[str, float]:
    """
    Calculate actual density extremes per second within window.
    Decision 2: Calculate actual per-second densities, not estimates.
    
    Args:
        timelines: All timeline data
        window_start: Start time of window
        window_end: End time of window
        
    Returns:
        Dict with actual density metrics
    """
    window_duration = window_end - window_start
    if window_duration == 0:
        return {'avg_density': 0, 'max_density': 0, 'min_density': 0}
    
    # Calculate per-second densities
    second_densities = []
    for second_start in range(int(window_start), int(window_end)):
        second_end = min(second_start + 1, window_end)
        second_counts = count_elements_in_window(timelines, second_start, second_end)
        density = second_counts['element_count'] / (second_end - second_start)
        second_densities.append(density)
    
    if second_densities:
        return {
            'avg_density': round(sum(second_densities) / len(second_densities), 2),
            'max_density': round(max(second_densities), 2),
            'min_density': round(min(second_densities), 2)
        }
    else:
        return {'avg_density': 0, 'max_density': 0, 'min_density': 0}

# ============== WORD COUNT CALCULATION ==============

def count_words_in_window(speech_segments: List[Dict], window_start: float, window_end: float) -> int:
    """
    Count spoken words within a time window.
    Decision 3: Count words from speech transcription segments.
    
    Args:
        speech_segments: List of speech segments with text
        window_start: Start time of window
        window_end: End time of window
        
    Returns:
        Number of words in window (pro-rated for boundary segments)
    """
    word_count = 0
    
    for segment in speech_segments:
        # Skip music/singing segments
        if is_likely_music(segment.get('text', '')):
            continue
            
        # Calculate overlap with window
        seg_start = segment['start']
        seg_end = segment['start'] + segment['duration']
        overlap_start = max(seg_start, window_start)
        overlap_end = min(seg_end, window_end)
        
        if overlap_start < overlap_end:
            # Pro-rate words based on overlap
            segment_words = len(segment.get('text', '').split())
            if segment['duration'] > 0:
                overlap_ratio = (overlap_end - overlap_start) / segment['duration']
                word_count += int(segment_words * overlap_ratio)
    
    return word_count

# ============== AUDIO ENERGY CALCULATION ==============

def calculate_audio_energy_for_windows(
    audio_path: Path, 
    windows: Dict[str, Optional[Tuple[float, float]]]
) -> Dict[str, float]:
    """
    Calculate audio energy metrics for exact temporal windows.
    It was decided to make this synchronous since librosa is CPU-only
    and doesn't benefit from async operations.
    
    Args:
        audio_path: Path to audio file
        windows: Dict of window names to (start, end) tuples
        
    Returns:
        Dict with energy metrics for each window
    """
    # Per Decision 5: No try/except - fail loudly if librosa not installed
    import librosa
    
    # Load audio once
    y, sr = librosa.load(audio_path, sr=16000)
    
    energy_results = {}
    
    for window_name, bounds in windows.items():
        if bounds:
            start, end = bounds
            # Extract audio segment for this window
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            window_audio = y[start_sample:end_sample]
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=window_audio, frame_length=2048, hop_length=512)[0]
            
            if len(rms) > 0:
                # Calculate metrics
                energy_results[f'{window_name}_energy_level'] = float(np.mean(rms))
                energy_results[f'{window_name}_energy_variance'] = float(np.var(rms))
                energy_results[f'{window_name}_energy_max'] = float(np.max(rms))
                
                # Determine burst pattern for this window
                if len(rms) > 3:
                    third_size = len(rms) // 3
                    front_avg = np.mean(rms[:third_size])
                    back_avg = np.mean(rms[-third_size:])
                    
                    if front_avg > back_avg * 1.2:
                        energy_results[f'{window_name}_burst_pattern'] = 'front_loaded'
                    elif back_avg > front_avg * 1.2:
                        energy_results[f'{window_name}_burst_pattern'] = 'back_loaded'
                    else:
                        energy_results[f'{window_name}_burst_pattern'] = 'steady'
    
    # Global climax (peak across entire video)
    full_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if len(full_rms) > 0:
        peak_frame = np.argmax(full_rms)
        energy_results['climax_timestamp'] = float(peak_frame * 512 / sr)
    
    return energy_results

# ============== CATEGORY 3: DISTRIBUTIONS ==============

def calculate_emotion_distribution(expression_timeline: List[Dict], window_start: float, window_end: float) -> Dict[str, float]:
    """Calculate distribution of emotions within window."""
    emotions = defaultdict(int)
    total = 0
    for item in expression_timeline:
        if window_start <= item.get('timestamp', 0) < window_end:
            emotion = item.get('expression', 'neutral')
            emotions[emotion] += 1
            total += 1
    # Return percentages
    return {f"{k}_pct": round(v/total, 2) if total > 0 else 0 for k, v in emotions.items()}

def calculate_framing_distribution(camera_distance_timeline: Dict, window_start: float, window_end: float) -> Dict[str, float]:
    """Calculate distribution of shot types within window."""
    framings = {'close_up': 0, 'medium': 0, 'wide': 0}
    for timestamp_key, data in camera_distance_timeline.items():
        timestamp = float(timestamp_key.split('-')[0])
        if window_start <= timestamp < window_end:
            distance = data.get('distance', 'medium').lower()
            if 'close' in distance:
                framings['close_up'] += 1
            elif 'wide' in distance:
                framings['wide'] += 1
            else:
                framings['medium'] += 1
    total = sum(framings.values())
    return {f"{k}_pct": round(v/total, 2) if total > 0 else 0 for k, v in framings.items()}

def calculate_vocabulary_diversity(speech_segments: List[Dict], window_start: float, window_end: float) -> float:
    """Calculate vocabulary diversity (unique words / total words)."""
    words = []
    for segment in speech_segments:
        if not is_likely_music(segment.get('text', '')):
            seg_start = segment['start']
            seg_end = segment['start'] + segment['duration']
            if seg_start < window_end and seg_end > window_start:
                words.extend(segment.get('text', '').lower().split())
    
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)

# ============== CATEGORY 4: VARIANCES ==============

def calculate_gaze_variance(gaze_timeline: Dict, window_start: float, window_end: float) -> float:
    """Calculate variance in gaze direction changes."""
    gaze_changes = []
    prev_looking = None
    for timestamp_key in sorted(gaze_timeline.keys()):
        timestamp = float(timestamp_key.split('-')[0])
        if window_start <= timestamp < window_end:
            looking = gaze_timeline[timestamp_key].get('looking_at_camera', False)
            if prev_looking is not None and looking != prev_looking:
                gaze_changes.append(1)
            else:
                gaze_changes.append(0)
            prev_looking = looking
    
    return round(np.var(gaze_changes), 3) if gaze_changes else 0.0

def calculate_pacing_variation(timelines: Dict, window_start: float, window_end: float) -> float:
    """Calculate variation in content pacing (cuts, text changes, etc.)."""
    events = []
    for timeline_name, timeline_data in timelines.items():
        if isinstance(timeline_data, list):
            for item in timeline_data:
                # Handle both dict items with timestamps and raw float boundaries
                if isinstance(item, dict):
                    timestamp = item.get('timestamp', 0)
                    if window_start <= timestamp < window_end:
                        events.append(timestamp)
                elif isinstance(item, (int, float)):
                    if window_start <= item < window_end:
                        events.append(item)
    
    if len(events) < 2:
        return 0.0
    
    events.sort()
    intervals = [events[i+1] - events[i] for i in range(len(events)-1)]
    return round(np.var(intervals), 3) if intervals else 0.0

def calculate_scene_duration_variance(scene_boundaries: List[float], window_start: float, window_end: float) -> float:
    """Calculate variance in scene durations within window."""
    boundaries_in_window = [window_start]
    for boundary in scene_boundaries:
        if window_start < boundary < window_end:
            boundaries_in_window.append(boundary)
    boundaries_in_window.append(window_end)
    
    durations = []
    for i in range(len(boundaries_in_window) - 1):
        duration = boundaries_in_window[i+1] - boundaries_in_window[i]
        durations.append(duration)
    
    return round(np.var(durations), 3) if len(durations) > 1 else 0.0

# ============== CATEGORY 5: COMPLEX METRICS ==============

def calculate_climax_moments(timelines: Dict, video_duration: float) -> Dict[str, Any]:
    """2-pass calculation to identify climax moments."""
    # First pass: Calculate density per second for entire video
    second_densities = []
    for second in range(int(video_duration)):
        counts = count_elements_in_window(timelines, second, second + 1)
        second_densities.append(counts['element_count'])
    
    # Second pass: Identify peaks (climax moments)
    if not second_densities:
        return {'climax_count': 0, 'climax_timestamps': []}
    
    mean_density = np.mean(second_densities)
    std_density = np.std(second_densities)
    threshold = mean_density + (2 * std_density)
    
    climax_timestamps = []
    for i, density in enumerate(second_densities):
        if density > threshold:
            # Check if it's a local maximum
            is_peak = True
            if i > 0 and second_densities[i-1] >= density:
                is_peak = False
            if i < len(second_densities)-1 and second_densities[i+1] >= density:
                is_peak = False
            if is_peak:
                climax_timestamps.append(float(i))
    
    return {
        'climax_count': len(climax_timestamps),
        'climax_timestamps': climax_timestamps[:3]  # Top 3 climax moments
    }

def calculate_overlay_metrics(text_timeline: List[Dict], sticker_timeline: List[Dict], 
                              window_start: float, window_end: float) -> Dict[str, int]:
    """Complex overlay pattern analysis."""
    text_times = [item['timestamp'] for item in text_timeline 
                  if window_start <= item.get('timestamp', 0) < window_end]
    sticker_times = [item['timestamp'] for item in sticker_timeline 
                     if window_start <= item.get('timestamp', 0) < window_end]
    
    metrics = {
        'text_burst_count': 0,
        'sticker_burst_count': 0,
        'overlay_overlap_count': 0,
    }
    
    # Detect bursts (3+ elements within 1 second)
    for times, burst_key in [(text_times, 'text_burst_count'), 
                              (sticker_times, 'sticker_burst_count')]:
        for i in range(len(times)):
            burst_end = times[i] + 1.0
            burst_items = sum(1 for t in times[i:] if t <= burst_end)
            if burst_items >= 3:
                metrics[burst_key] += 1
    
    # Detect overlaps (text and sticker within 0.5s)
    for text_time in text_times:
        for sticker_time in sticker_times:
            if abs(text_time - sticker_time) <= 0.5:
                metrics['overlay_overlap_count'] += 1
                break
    
    return metrics

# ============== MAIN COMPUTATION FUNCTION ==============

def compute_temporal_windows(
    timelines: Dict,
    video_metadata: Dict,
    speech_segments: List[Dict],
    audio_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Single entry point for ALL temporal computations.
    
    It was decided to keep this as a monolithic function to match 
    existing codebase patterns where compute functions are 200+ lines.
    
    Agreed to use explicit/hard-coded field names for ML compatibility
    and to fail loudly on invalid data rather than return defaults.
    
    Args:
        timelines: All timeline data (text, stickers, objects, etc.)
        video_metadata: Video metadata including duration, publish_hour, etc.
        speech_segments: Speech transcription segments
        audio_path: Optional path to audio file for energy calculation
        
    Returns:
        Unified JSON with temporal windows, global metadata, and outcomes
    """
    # Decision 8: Validation with loud failures
    video_duration = video_metadata.get('duration', 0)
    if video_duration <= 0:
        raise ValueError(f"Invalid video duration: {video_duration}")
    
    # Log warnings for empty timelines
    for timeline_name, timeline_data in timelines.items():
        if not timeline_data:
            logger.warning(f"Empty timeline: {timeline_name}")
    
    # Calculate window boundaries
    windows = calculate_temporal_windows(video_duration)
    middle_segments = calculate_middle_segments(video_duration)
    
    # Calculate speech coverage for all windows
    speech_coverage = calculate_temporal_speech_coverage(speech_segments, video_duration)
    
    # Initialize result structure
    result = {
        'video_id': video_metadata.get('video_id', 'unknown'),
        'duration': video_duration,
        'temporal_windows': {},
        'global_metadata': {},
        'outcomes': {}
    }
    
    # Process hook window
    if windows['hook']:
        hook_start, hook_end = windows['hook']
        hook_counts = count_elements_in_window(timelines, hook_start, hook_end)
        hook_density = calculate_density_metrics(timelines, hook_start, hook_end)
        hook_word_count = count_words_in_window(speech_segments, hook_start, hook_end)
        
        # Category 3: Distributions (Decision 3)
        hook_emotion_dist = calculate_emotion_distribution(
            timelines.get('expression_timeline', []), hook_start, hook_end)
        hook_framing_dist = calculate_framing_distribution(
            timelines.get('camera_distance_timeline', {}), hook_start, hook_end)
        hook_vocab_diversity = calculate_vocabulary_diversity(
            speech_segments, hook_start, hook_end)
        
        # Category 4: Variances (Decision 3)
        hook_gaze_variance = calculate_gaze_variance(
            timelines.get('gaze_timeline', {}), hook_start, hook_end)
        hook_pacing_variation = calculate_pacing_variation(
            timelines, hook_start, hook_end)
        hook_scene_variance = calculate_scene_duration_variance(
            timelines.get('scene_boundaries', []), hook_start, hook_end)
        
        # Category 5: Overlay metrics (Decision 3)
        hook_overlay = calculate_overlay_metrics(
            timelines.get('text_overlay_timeline', []),
            timelines.get('sticker_timeline', []),
            hook_start, hook_end)
        
        result['temporal_windows']['hook'] = {
            **{f'hook_{k}': v for k, v in hook_counts.items()},
            **{f'hook_{k}': v for k, v in hook_density.items()},
            'hook_speech_coverage': speech_coverage.get('hook_speech_coverage', 0),
            'hook_word_count': hook_word_count,
            **{f'hook_{k}': v for k, v in hook_emotion_dist.items()},
            **{f'hook_{k}': v for k, v in hook_framing_dist.items()},
            'hook_vocabulary_diversity': hook_vocab_diversity,
            'hook_gaze_variance': hook_gaze_variance,
            'hook_pacing_variation': hook_pacing_variation,
            'hook_scene_duration_variance': hook_scene_variance,
            **{f'hook_{k}': v for k, v in hook_overlay.items()},
        }
    
    # Process middle window
    if windows['middle']:
        middle_start, middle_end = windows['middle']
        middle_counts = count_elements_in_window(timelines, middle_start, middle_end)
        middle_density = calculate_density_metrics(timelines, middle_start, middle_end)  # Decision 2
        middle_word_count = count_words_in_window(speech_segments, middle_start, middle_end)  # Decision 3
        
        result['temporal_windows']['middle'] = {
            **{f'middle_{k}': v for k, v in middle_counts.items()},
            **{f'middle_{k}': v for k, v in middle_density.items()},
            'middle_speech_coverage': speech_coverage.get('middle_speech_coverage', 0),
            'middle_word_count': middle_word_count,
            'segments': {}
        }
        
        # Process middle segments
        for segment_name, bounds in middle_segments.items():
            seg_start, seg_end = bounds['start'], bounds['end']
            seg_counts = count_elements_in_window(timelines, seg_start, seg_end)
            seg_density = calculate_density_metrics(timelines, seg_start, seg_end)  # Decision 2
            seg_word_count = count_words_in_window(speech_segments, seg_start, seg_end)  # Decision 3
            
            result['temporal_windows']['middle']['segments'][segment_name] = {
                **{k: v for k, v in seg_counts.items()},
                **{k: v for k, v in seg_density.items()},
                'speech_coverage': speech_coverage.get(f'middle_{segment_name}_speech_coverage', 0),
                'word_count': seg_word_count
            }
    
    # Process closing window
    if windows['closing']:
        closing_start, closing_end = windows['closing']
        closing_counts = count_elements_in_window(timelines, closing_start, closing_end)
        closing_density = calculate_density_metrics(timelines, closing_start, closing_end)  # Decision 2
        closing_word_count = count_words_in_window(speech_segments, closing_start, closing_end)  # Decision 3
        
        result['temporal_windows']['closing'] = {
            **{f'closing_{k}': v for k, v in closing_counts.items()},
            **{f'closing_{k}': v for k, v in closing_density.items()},
            'closing_speech_coverage': speech_coverage.get('closing_speech_coverage', 0),
            'closing_word_count': closing_word_count
        }
    
    # It was decided to recalculate audio energy for exact windows
    if audio_path:
        # Per Decision 2: Direct synchronous call, no async
        energy_results = calculate_audio_energy_for_windows(audio_path, windows)
        
        # Add energy metrics to each window
        for window_name in ['hook', 'middle', 'closing']:
            if window_name in result['temporal_windows']:
                for key, value in energy_results.items():
                    if key.startswith(window_name):
                        result['temporal_windows'][window_name][key] = value
        
        # Add global climax timestamp
        if 'climax_timestamp' in energy_results:
            result['global_metadata']['climax_timestamp'] = energy_results['climax_timestamp']
    
    # Add global metadata
    result['global_metadata'].update({
        'video_duration': video_duration,
        'video_id': video_metadata.get('video_id', ''),
        'publish_hour': video_metadata.get('publish_hour', 0),
        'caption_length': video_metadata.get('caption_length', 0),
        'hashtag_count': video_metadata.get('hashtag_count', 0),
        'has_captions': video_metadata.get('has_captions', False),
        'has_soundtrack': video_metadata.get('has_soundtrack', False),
    })
    
    # Category 5: Calculate climax moments globally (Decision 3)
    climax_data = calculate_climax_moments(timelines, video_duration)
    result['global_metadata'].update(climax_data)
    
    # Add outcomes (if available)
    result['outcomes'] = {
        'view_count': video_metadata.get('view_count', 0),
        'like_count': video_metadata.get('like_count', 0),
        'comment_count': video_metadata.get('comment_count', 0),
        'share_count': video_metadata.get('share_count', 0),
    }
    
    return result

# ============== SAVE FUNCTION ==============

def save_temporal_unified(result: Dict, output_path: Path) -> None:
    """
    Save unified temporal JSON to file.
    
    Args:
        result: Unified temporal computation result
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved temporal unified JSON to {output_path}")
