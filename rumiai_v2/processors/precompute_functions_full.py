"""
Precompute functions for ML-ready analysis - Full implementation.
These functions compute comprehensive metrics from timeline data.
"""

import os
import sys
import json
import time
import statistics
import re
import logging
from datetime import datetime
from statistics import variance

logger = logging.getLogger(__name__)

def parse_timestamp_to_seconds(timestamp):
    """Convert timestamp like '0-1s' to start second"""
    try:
        return int(timestamp.split('-')[0])
    except:
        return None
def is_timestamp_in_second(timestamp, second):
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
def mean(values):
    """Calculate mean of a list"""
    return sum(values) / len(values) if values else 0
def stdev(values):
    """Calculate standard deviation of a list"""
    if len(values) < 2:
        return 0
    return statistics.stdev(values)
def compute_visual_overlay_metrics(text_overlay_timeline, sticker_timeline, gesture_timeline, 
                                  speech_timeline, object_timeline, video_duration):
    """Compute comprehensive visual overlay metrics for ML-ready analysis"""
    import re
    from collections import defaultdict, Counter
    
    # Initialize tracking
    seconds = int(video_duration) + 1
    
    # Core metrics
    total_text_overlays = len(text_overlay_timeline)
    unique_texts = set()
    text_appearances = []
    
    # Process text overlays
    for timestamp, data in text_overlay_timeline.items():
        text = data.get('text', '')
        if text:
            unique_texts.add(text.lower().strip())
            try:
                start_sec = float(timestamp.split('-')[0])
                text_appearances.append((start_sec, text))
            except:
                pass
    
    # 1. Core Metrics
    avg_texts_per_second = total_text_overlays / video_duration if video_duration > 0 else 0
    unique_text_count = len(unique_texts)
    time_to_first_text = min([t[0] for t in text_appearances]) if text_appearances else video_duration
    
    # Calculate average display duration
    display_durations = []
    for timestamp, data in text_overlay_timeline.items():
        try:
            parts = timestamp.split('-')
            if len(parts) == 2:
                start = float(parts[0])
                end = float(parts[1].replace('s', ''))
                duration = end - start
                display_durations.append(duration)
        except:
            pass
    avg_text_display_duration = mean(display_durations)
    
    # 2. Overlay Rhythm & Density
    appearance_intervals = []
    sorted_appearances = sorted(text_appearances, key=lambda x: x[0])
    for i in range(1, len(sorted_appearances)):
        interval = sorted_appearances[i][0] - sorted_appearances[i-1][0]
        appearance_intervals.append(interval)
    
    # Burst windows and clutter timeline
    burst_windows = []
    clutter_timeline = {}
    for start in range(0, seconds, 5):
        end = min(start + 5, seconds)
        window_key = f"{start}-{end}s"
        
        text_count = sum(1 for t in text_appearances if start <= t[0] < end)
        sticker_count = sum(1 for ts, _ in sticker_timeline.items() 
                           if start <= parse_timestamp_to_seconds(ts) < end)
        total_count = text_count + sticker_count
        
        clutter_timeline[window_key] = {
            'text': text_count,
            'sticker': sticker_count,
            'total': total_count
        }
        
        if total_count >= 3:
            burst_windows.append((window_key, total_count))  # Store as tuple with count
    
    # Calculate breathing room ratio
    seconds_with_overlays = set()
    for timestamp in text_overlay_timeline:
        start_sec = parse_timestamp_to_seconds(timestamp)
        if start_sec is not None:
            seconds_with_overlays.add(int(start_sec))
    breathing_room_ratio = (seconds - len(seconds_with_overlays)) / seconds if seconds > 0 else 0
    
    # Average simultaneous texts
    simultaneous_counts = []
    for sec in range(seconds):
        count = sum(1 for ts in text_overlay_timeline 
                   if is_timestamp_in_second(ts, sec))
        if count > 0:
            simultaneous_counts.append(count)
    avg_simultaneous_texts = mean(simultaneous_counts)
    
    # 3. Readability Components (defaults)
    readability_components = {
        'avg_contrast_ratio': 0.75,
        'avg_text_size_normalized': 0.6,
        'center_screen_percentage': 0.7,
        'occlusion_events': 0
    }
    
    # Count occlusion events
    for i, (time1, _) in enumerate(sorted_appearances):
        for time2, _ in sorted_appearances[i+1:]:
            if abs(time1 - time2) < 1.0:
                readability_components['occlusion_events'] += 1
    
    # 4. Text Position Distribution (defaults)
    text_position_distribution = {
        'top_third': 0.3,
        'middle_third': 0.5,
        'bottom_third': 0.2
    }
    
    # 5. Text Hierarchy Metrics
    text_sizes = []
    for _, text in text_appearances:
        size_score = len(text) * (2 if text.isupper() else 1)
        text_sizes.append(size_score)
    
    text_size_variance = stdev(text_sizes) if len(text_sizes) > 1 else 0
    
    dominant_text_changes = 0
    if text_sizes:
        last_dominant = text_sizes[0]
        mean_size = mean(text_sizes)
        for size in text_sizes[1:]:
            if abs(size - last_dominant) > mean_size * 0.5:
                dominant_text_changes += 1
                last_dominant = size
    
    # 6. CTA Reinforcement Matrix
    cta_keywords = ['buy', 'shop', 'click', 'link', 'follow', 'subscribe', 'comment', 
                    'share', 'order', 'get', 'save', 'discount', 'sale', 'limited']
    
    cta_reinforcement_matrix = {
        'text_only': 0,
        'text_gesture': 0,
        'text_sticker': 0,
        'all_three': 0
    }
    
    for timestamp, data in text_overlay_timeline.items():
        text = data.get('text', '').lower()
        if any(keyword in text for keyword in cta_keywords):
            has_gesture = timestamp in gesture_timeline
            has_sticker = timestamp in sticker_timeline
            
            if has_gesture and has_sticker:
                cta_reinforcement_matrix['all_three'] += 1
            elif has_gesture:
                cta_reinforcement_matrix['text_gesture'] += 1
            elif has_sticker:
                cta_reinforcement_matrix['text_sticker'] += 1
            else:
                cta_reinforcement_matrix['text_only'] += 1
    
    # 7. Semantic Clustering
    text_semantic_groups = {
        'product_mentions': 0,
        'urgency_phrases': 0,
        'social_proof': 0,
        'questions': 0,
        'other_text': 0
    }
    
    product_patterns = ['product', 'item', 'brand', 'quality', 'feature', 'benefit']
    urgency_patterns = ['now', 'today', 'limited', 'last', 'hurry', 'quick', 'fast']
    social_patterns = ['everyone', 'viral', 'trending', 'popular', 'love', 'favorite']
    
    for _, text in text_appearances:
        text_lower = text.lower()
        classified = False
        
        if any(pattern in text_lower for pattern in product_patterns):
            text_semantic_groups['product_mentions'] += 1
            classified = True
        elif any(pattern in text_lower for pattern in urgency_patterns):
            text_semantic_groups['urgency_phrases'] += 1
            classified = True
        elif any(pattern in text_lower for pattern in social_patterns):
            text_semantic_groups['social_proof'] += 1
            classified = True
        elif '?' in text:
            text_semantic_groups['questions'] += 1
            classified = True
        
        if not classified:
            text_semantic_groups['other_text'] += 1
    
    # 8. Cross-Modal Alignment - Speech
    text_speech_alignment = {
        'text_matches_speech': 0,
        'text_precedes_speech': 0,
        'text_follows_speech': 0,
        'text_contradicts_speech': 0
    }
    
    speech_texts = []
    for timestamp, data in speech_timeline.items():
        if 'text' in data:
            speech_texts.append((parse_timestamp_to_seconds(timestamp), data['text'].lower()))
    
    for text_time, text in text_appearances:
        text_lower = text.lower()
        alignment_found = False
        
        for speech_time, speech_text in speech_texts:
            time_diff = text_time - speech_time
            
            if any(word in speech_text for word in text_lower.split()) or \
               any(word in text_lower for word in speech_text.split()):
                if abs(time_diff) < 1.0:
                    text_speech_alignment['text_matches_speech'] += 1
                elif time_diff < -1.0:
                    text_speech_alignment['text_precedes_speech'] += 1
                elif time_diff > 1.0:
                    text_speech_alignment['text_follows_speech'] += 1
                alignment_found = True
                break
        
        if not alignment_found and speech_texts:
            text_speech_alignment['text_contradicts_speech'] += 1
    
    # 9. Cross-Modal Alignment - Gesture
    text_gesture_coordination = {
        'aligned': 0,
        'misaligned': 0,
        'neutral': 0
    }
    
    for timestamp in text_overlay_timeline:
        text_sec = parse_timestamp_to_seconds(timestamp)
        if text_sec is not None:
            gesture_found = False
            for gesture_ts in gesture_timeline:
                gesture_sec = parse_timestamp_to_seconds(gesture_ts)
                if gesture_sec is not None and abs(text_sec - gesture_sec) < 1.0:
                    gesture_data = gesture_timeline[gesture_ts]
                    if any(g in str(gesture_data).lower() for g in ['point', 'tap', 'swipe']):
                        text_gesture_coordination['aligned'] += 1
                    else:
                        text_gesture_coordination['neutral'] += 1
                    gesture_found = True
                    break
            
            if not gesture_found:
                text_gesture_coordination['misaligned'] += 1
    
    # 10. Additional Core Metrics
    total_sticker_count = len(sticker_timeline)
    total_text_elements = total_text_overlays  # Clarify naming
    
    # 11. Text Categories Analysis
    text_categories = {
        'headline': 0,
        'subtitle': 0,
        'cta': 0,
        'caption': 0,
        'numbers': 0,
        'hashtag': 0
    }
    
    cta_keywords = ['buy', 'shop', 'click', 'link', 'follow', 'subscribe', 'comment', 
                    'share', 'order', 'get', 'save', 'discount', 'sale', 'limited']
    
    for _, text in text_appearances:
        text_lower = text.lower()
        
        # Categorize text
        if len(text) > 30 or text.isupper():
            text_categories['headline'] += 1
        elif any(keyword in text_lower for keyword in cta_keywords):
            text_categories['cta'] += 1
        elif text.startswith('#'):
            text_categories['hashtag'] += 1
        elif any(char.isdigit() for char in text):
            text_categories['numbers'] += 1
        elif len(text) > 15 and len(text) <= 30:
            text_categories['subtitle'] += 1
        else:
            text_categories['caption'] += 1
    
    # 12. Key Alignment Moments
    key_alignment_moments = []
    
    # Find text-gesture sync moments
    for timestamp, data in text_overlay_timeline.items():
        text_sec = parse_timestamp_to_seconds(timestamp)
        if text_sec is not None:
            text = data.get('text', '')
            
            # Check for gesture alignment
            for gesture_ts in gesture_timeline:
                gesture_sec = parse_timestamp_to_seconds(gesture_ts)
                if gesture_sec is not None and abs(text_sec - gesture_sec) < 0.5:
                    gesture_data = gesture_timeline[gesture_ts]
                    if 'point' in str(gesture_data).lower() and any(kw in text.lower() for kw in cta_keywords):
                        key_alignment_moments.append({
                            'timestamp': round(text_sec, 1),
                            'type': 'text_gesture_sync',
                            'elements': ['CTA text', 'pointing gesture']
                        })
                        break
    
    # Limit to top 5 key moments
    key_alignment_moments = key_alignment_moments[:5]
    
    # 13. Calculate Overall Sync Score
    total_alignments = sum(text_gesture_coordination.values())
    overall_sync_score = text_gesture_coordination['aligned'] / total_alignments if total_alignments > 0 else 0
    
    # 14. ML Tags Generation
    ml_tags = []
    
    if avg_texts_per_second > 1.0:
        ml_tags.append('text_heavy')
    if cta_reinforcement_matrix['text_gesture'] > 0 or cta_reinforcement_matrix['all_three'] > 0:
        ml_tags.append('cta_focused')
    if sum(cta_reinforcement_matrix.values()) >= 3:
        ml_tags.append('multi_reinforced')
    if text_size_variance > 50:
        ml_tags.append('position_varied')
    if avg_simultaneous_texts > 2:
        ml_tags.append('overlay_dense')
    if breathing_room_ratio < 0.3:
        ml_tags.append('continuous_overlay')
    
    # 15. Overlay Strategy Classification
    if avg_texts_per_second < 0.3:
        overlay_strategy = 'minimal'
    elif avg_texts_per_second < 0.8:
        overlay_strategy = 'moderate'
    elif avg_texts_per_second > 0.8:
        overlay_strategy = 'heavy'
    
    if text_size_variance > 50:
        overlay_strategy = 'dynamic'
    
    # 16. Pattern Flags
    pattern_flags = {
        'hasTextBursts': len(burst_windows) > 0,
        'hasCTAReinforcement': sum(cta_reinforcement_matrix.values()) > 0,
        'isCluttered': avg_simultaneous_texts > 3,
        'isMobileOptimized': readability_components['avg_text_size_normalized'] > 0.03 and 
                            readability_components['center_screen_percentage'] > 0.7
    }
    
    # 17. Additional metrics for simplified output
    # Peak overlay moments (highest density windows)
    peak_overlay_moments = []
    for window, overlays in burst_windows:
        peak_overlay_moments.append({
            'timestamp': window,
            'overlay_count': overlays,
            'types': ['text', 'sticker']  # Simplified
        })
    
    # Multimodal peaks (when multiple elements align)
    multimodal_peaks = []
    for moment in key_alignment_moments[:5]:  # Top 5
        multimodal_peaks.append({
            'timestamp': moment['timestamp'],
            'modality_count': len(moment['elements']),
            'alignment': moment.get('score', 0.8)
        })
    
    # Sticker type breakdown
    sticker_types = {'emoji': 0, 'animated': 0, 'static': 0}
    for _, data in sticker_timeline.items():
        sticker_type = data.get('sticker_type', 'static')
        if sticker_type in sticker_types:
            sticker_types[sticker_type] += 1
    
    # Temporal density calculations
    early_density = sum(1 for t, _ in text_appearances if t < video_duration/3) / (video_duration/3) if video_duration > 0 else 0
    late_density = sum(1 for t, _ in text_appearances if t > 2*video_duration/3) / (video_duration/3) if video_duration > 0 else 0
    
    # Updated pattern flags for our simplified metrics
    pattern_flags = {
        'highDensity': avg_texts_per_second > 1.0,
        'textHeavy': total_text_overlays > total_sticker_count * 2,
        'stickerHeavy': total_sticker_count > total_text_overlays * 2,
        'burstPattern': len(burst_windows) > 2,
        'consistentRhythm': stdev(appearance_intervals) < mean(appearance_intervals) * 0.3 if appearance_intervals else False,
        'ctaFocused': text_categories.get('cta', 0) / total_text_overlays > 0.3 if total_text_overlays > 0 else False,
        'urgencyDriven': text_categories.get('urgency', 0) / total_text_overlays > 0.2 if total_text_overlays > 0 else False
    }
    
    # 18. Data Quality Metrics
    expected_text_count = int(video_duration * 0.5)  # Expect at least 0.5 texts per second on average
    text_detection_rate = min(total_text_overlays / expected_text_count, 1.0) if expected_text_count > 0 else 1.0
    
    expected_sticker_count = int(video_duration * 0.1)  # Expect at least 0.1 stickers per second
    sticker_detection_rate = min(total_sticker_count / expected_sticker_count, 1.0) if expected_sticker_count > 0 else 1.0
    
    data_completeness = (text_detection_rate + sticker_detection_rate) / 2
    
    # Calculate overall confidence
    confidence_factors = [
        text_detection_rate,
        sticker_detection_rate,
        1.0 if total_text_overlays > 0 else 0.0,
        1.0 if len(text_appearances) == total_text_overlays else 0.8
    ]
    overall_confidence = mean(confidence_factors)
    
    return {
        # Core metrics
        'total_text_overlays': total_text_overlays,
        'unique_text_count': unique_text_count,
        'avg_texts_per_second': round(avg_texts_per_second, 3),
        'time_to_first_text': round(time_to_first_text, 2),
        'avg_text_display_duration': round(avg_text_display_duration, 2),
        'text_repetition_ratio': round((total_text_overlays - unique_text_count) / total_text_overlays, 2) if total_text_overlays > 0 else 0,
        
        # Rhythm & Density
        'appearance_intervals': [round(i, 2) for i in appearance_intervals[:10]],
        'burst_windows': burst_windows,
        'clutter_timeline': clutter_timeline,
        'peak_overlay_moments': peak_overlay_moments,
        'overlay_free_duration': round(breathing_room_ratio * video_duration, 2),
        'rhythm_consistency': round(1 - (stdev(appearance_intervals) / mean(appearance_intervals) if appearance_intervals and mean(appearance_intervals) > 0 else 0), 2),
        
        # Semantic content - map from existing categories to expected ones
        'text_categories': {
            'product_mention': text_semantic_groups.get('product_mentions', 0),
            'price_discount': text_categories.get('numbers', 0),  # Numbers often indicate prices
            'urgency': text_semantic_groups.get('urgency_phrases', 0),
            'social_proof': text_semantic_groups.get('social_proof', 0),
            'cta': text_categories.get('cta', 0),
            'feature_benefit': text_semantic_groups.get('other_text', 0)  # Approximation
        },
        'primary_message_type': max(text_semantic_groups, key=text_semantic_groups.get) if text_semantic_groups else 'none',
        'cta_density': round(text_categories.get('cta', 0) / video_duration * 10, 2) if video_duration > 0 else 0,
        'urgency_score': round(text_semantic_groups.get('urgency_phrases', 0) / total_text_overlays, 2) if total_text_overlays > 0 else 0,
        
        # Cross-modal alignment
        'text_speech_alignment': text_speech_alignment,
        'text_gesture_coordination': text_gesture_coordination,
        'reinforced_moments': key_alignment_moments,
        'cta_reinforcement_matrix': cta_reinforcement_matrix,
        'multimodal_peaks': multimodal_peaks,
        
        # Sticker integration
        'total_stickers': total_sticker_count,
        'sticker_types': sticker_types,
        'text_sticker_ratio': round(total_text_overlays / total_sticker_count, 2) if total_sticker_count > 0 else float('inf'),
        'combined_overlay_density': round((total_text_overlays + total_sticker_count) / video_duration, 2) if video_duration > 0 else 0,
        
        # Temporal patterns
        'front_loaded_ratio': round(sum(1 for t, _ in text_appearances if t < video_duration/3) / len(text_appearances), 2) if text_appearances else 0,
        'climax_timing': round(float(burst_windows[0][0].split('-')[0]) if burst_windows and isinstance(burst_windows[0][0], str) else video_duration/2, 2),
        'overlay_acceleration': round((late_density - early_density) / early_density, 2) if early_density > 0 else 0,
        'intro_overlay_count': sum(1 for t, _ in text_appearances if t < 5),
        'outro_overlay_count': sum(1 for t, _ in text_appearances if t > video_duration - 5),
        
        # Quality metrics
        'data_completeness': round(data_completeness, 2),
        'detection_confidence': round(overall_confidence, 2),
        
        # ML features
        'overlay_pattern_flags': pattern_flags,
        'ml_tags': ml_tags
    }
def compute_creative_density_analysis(timelines, duration):
    """Compute smart creative density analysis instead of sending full timelines
    
    Args:
        timelines: Dictionary containing all timeline data (textOverlayTimeline, stickerTimeline, etc.)
        duration: Video duration in seconds
        
    Returns:
        dict: Comprehensive density analysis metrics
    """
    # Initialize density tracking
    seconds = int(duration) + 1
    density_per_second = [0] * seconds
    element_types_per_second = [{'text': 0, 'sticker': 0, 'effect': 0, 'transition': 0, 'object': 0, 'gesture': 0, 'expression': 0, 'scene_change': 0} for _ in range(seconds)]
    
    # Calculate density for each second
    for timeline_type, timeline_data in [
        ('text', timelines.get('textOverlayTimeline', {})),
        ('sticker', timelines.get('stickerTimeline', {})),
        ('effect', timelines.get('effectTimeline', {})),
        ('transition', timelines.get('transitionTimeline', {})),
        ('scene_change', timelines.get('sceneChangeTimeline', {})),
        ('object', timelines.get('objectTimeline', {})),
        ('gesture', timelines.get('gestureTimeline', {})),
        ('expression', timelines.get('expressionTimeline', {}))
    ]:
        for timestamp, data in timeline_data.items():
            # Parse timestamp
            try:
                # Handle negative timestamps by taking absolute value
                start_str = timestamp.split('-')[0]
                second = abs(int(start_str))
                
                # Ensure second is within video duration
                if second < seconds:
                    # Count elements based on type
                    if timeline_type == 'text':
                        count = len(data.get('texts', [])) if 'texts' in data else 1
                    elif timeline_type == 'sticker':
                        count = len(data.get('stickers', [])) if 'stickers' in data else 1
                    elif timeline_type == 'effect':
                        count = len(data.get('effects', [])) if 'effects' in data else 1
                    elif timeline_type == 'transition':
                        count = 1  # Each transition counts as 1
                    elif timeline_type == 'object':
                        count = data.get('total_objects', 1)
                    elif timeline_type == 'gesture':
                        count = len(data.get('gestures', [])) if 'gestures' in data else 1
                    elif timeline_type == 'expression':
                        count = 1  # Each expression counts as 1
                    elif timeline_type == 'scene_change':
                        count = 1  # Each scene change counts as 1
                    else:
                        count = 0
                    
                    density_per_second[second] += count
                    element_types_per_second[second][timeline_type] += count
            except:
                continue
    
    # Calculate statistics
    total_elements = sum(density_per_second)
    avg_density = mean(density_per_second)
    max_density = max(density_per_second) if density_per_second else 0
    min_density = min(density_per_second) if density_per_second else 0
    std_deviation = stdev(density_per_second) if len(density_per_second) > 1 else 0
    
    # Find peak moments (top 5-10 peaks)
    peak_moments = []
    density_with_index = [(d, i) for i, d in enumerate(density_per_second)]
    density_with_index.sort(reverse=True)
    
    # Calculate surprise scores for peaks
    num_peaks = min(10, len([d for d, _ in density_with_index if d > avg_density]))
    for density, second in density_with_index[:num_peaks]:
        if density > 0:
            # Calculate surprise score based on how much it exceeds local average
            surrounding_window = density_per_second[max(0, second-3):min(seconds, second+4)]
            local_avg = mean(surrounding_window) if surrounding_window else 0
            
            if std_deviation > 0:
                surprise_score = (density - local_avg) / std_deviation
            else:
                surprise_score = 0
            
            peak_moments.append({
                'timestamp': f"{second}-{second+1}s",
                'second': second,
                'total_elements': density,
                'surprise_score': round(surprise_score, 1),
                'breakdown': element_types_per_second[second]
            })
    
    # Sort peaks by timestamp
    peak_moments.sort(key=lambda x: x['second'])
    
    # Identify patterns
    patterns = []
    
    # Strong opening hook
    if seconds > 3:
        opening_density = mean(density_per_second[:3])
        rest_density = mean(density_per_second[3:]) if len(density_per_second) > 3 else 0
        if rest_density > 0 and opening_density > rest_density * 1.5:
            patterns.append('strong_opening_hook')
    
    # Crescendo pattern
    if seconds > 10:
        first_third = mean(density_per_second[:seconds//3])
        last_third = mean(density_per_second[-seconds//3:])
        if first_third > 0 and last_third > first_third * 1.5:
            patterns.append('crescendo_pattern')
    
    # Front loaded
    if seconds > 10:
        first_third = mean(density_per_second[:seconds//3])
        last_third = mean(density_per_second[-seconds//3:])
        if last_third > 0 and first_third > last_third * 1.5:
            patterns.append('front_loaded')
    
    # Consistent density
    if std_deviation < avg_density * 0.3:
        patterns.append('consistent_density')
    
    # Dual peak structure
    significant_peaks = [p for p in peak_moments if p['total_elements'] > avg_density * 1.5]
    if len(significant_peaks) >= 2:
        # Check if peaks are sufficiently separated
        peak_times = [p['second'] for p in significant_peaks]
        for i in range(len(peak_times)-1):
            if peak_times[i+1] - peak_times[i] >= 8:
                patterns.append('dual_peak_structure')
                break
    
    # Multi peak rhythm
    if len(significant_peaks) >= 3:
        patterns.append('multi_peak_rhythm')
    
    # Build density curve (include ALL seconds for complete data)
    density_curve = []
    for i in range(seconds):
        # Find primary element type at this second
        elements = element_types_per_second[i]
        primary_element = max(elements.items(), key=lambda x: x[1])[0] if any(elements.values()) else 'none'
        
        density_curve.append({
            'second': i,
            'density': density_per_second[i],
            'primary_element': primary_element
        })
    
    # Calculate element distribution
    element_distribution = {
        'text': sum(e['text'] for e in element_types_per_second),
        'sticker': sum(e['sticker'] for e in element_types_per_second),
        'effect': sum(e.get('effect', 0) for e in element_types_per_second),
        'transition': sum(e['transition'] for e in element_types_per_second),
        'object': sum(e['object'] for e in element_types_per_second),
        'gesture': sum(e.get('gesture', 0) for e in element_types_per_second),
        'expression': sum(e.get('expression', 0) for e in element_types_per_second)
    }
    
    # Add scene changes if available
    scene_changes = len(timelines.get('sceneChangeTimeline', {}))
    
    # Calculate additional metrics expected by prompt
    empty_seconds = len([d for d in density_per_second if d == 0])
    timeline_coverage = (seconds - empty_seconds) / seconds if seconds > 0 else 0
    
    # Normalize volatility to 0-1 range
    density_volatility = std_deviation / avg_density if avg_density > 0 else 0
    density_volatility = min(1.0, density_volatility)  # Cap at 1.0
    
    # Calculate density score (0-10)
    if avg_density < 0.5:
        creative_density_score = avg_density * 6  # 0-3 range for minimal
    elif avg_density < 1.5:
        creative_density_score = 3 + (avg_density - 0.5) * 3  # 3-6 range for medium
    else:
        creative_density_score = 6 + min((avg_density - 1.5) * 2, 4)  # 6-10 range for heavy
    
    # Determine primary density pattern
    if 'front_loaded' in patterns:
        density_pattern = 'front_loaded'
    elif 'crescendo_pattern' in patterns:
        density_pattern = 'back_loaded'
    elif 'consistent_density' in patterns:
        density_pattern = 'even'
    else:
        density_pattern = 'variable'
    
    # Generate ML tags
    creative_ml_tags = []
    if 'strong_opening_hook' in patterns:
        creative_ml_tags.append('hook_heavy')
    if element_distribution['text'] > total_elements * 0.4:
        creative_ml_tags.append('text_driven')
    if element_distribution['effect'] > total_elements * 0.3:
        creative_ml_tags.append('effect_rich')
    if len(significant_peaks) >= 3:
        creative_ml_tags.append('multi_peak')
    if empty_seconds > seconds * 0.3:
        creative_ml_tags.append('sparse_density')
    if density_volatility > 0.6:
        creative_ml_tags.append('dynamic_pacing')
    
    # Build peak_density_moments in expected format
    peak_density_moments = []
    for peak in peak_moments[:5]:  # Top 5 peaks
        peak_density_moments.append({
            'timestamp': peak['timestamp'],
            'element_count': peak['total_elements'],
            'surprise_score': peak['surprise_score']
        })
    
    # Convert patterns list to density_pattern_flags boolean object
    density_pattern_flags = {
        'strong_opening_hook': 'strong_opening_hook' in patterns,
        'crescendo_pattern': 'crescendo_pattern' in patterns,
        'front_loaded': 'front_loaded' in patterns,
        'consistent_density': 'consistent_density' in patterns,
        'dual_peak_structure': 'dual_peak_structure' in patterns,
        'multi_peak_rhythm': 'multi_peak_rhythm' in patterns
    }
    
    # Calculate timeline_frame_counts
    timeline_frame_counts = {
        'effect_count': len(timelines.get('effectTimeline', {})),
        'transition_count': len(timelines.get('transitionTimeline', {})),
        'object_detection_frames': len(timelines.get('objectTimeline', {})),
        'text_overlay_count': len(timelines.get('textOverlayTimeline', {})),
        'sticker_count': len(timelines.get('stickerTimeline', {})),
        'gesture_count': len(timelines.get('gestureTimeline', {})),
        'expression_count': len(timelines.get('expressionTimeline', {}))
    }
    
    # Compute multi-modal peaks
    multi_modal_peaks = []
    for i in range(seconds):
        active_types = [t for t, count in element_types_per_second[i].items() if count > 0]
        if len(active_types) >= 3:
            # Determine sync type based on element combination
            sync_type = 'reinforcing'
            if 'text' in active_types and 'gesture' in active_types:
                sync_type = 'reinforcing'
            elif 'effect' in active_types and 'transition' in active_types:
                sync_type = 'complementary'
            elif len(set(active_types)) == len(active_types):
                sync_type = 'redundant'
            
            multi_modal_peaks.append({
                'timestamp': f"{i}-{i+1}s",
                'elements': active_types,
                'syncType': sync_type
            })
    
    # Compute element co-occurrence
    element_cooccurrence = {}
    for i in range(seconds):
        types = element_types_per_second[i]
        # Check common pairings - focusing on creative elements
        if types['text'] > 0 and types['effect'] > 0:
            element_cooccurrence['text_effect'] = element_cooccurrence.get('text_effect', 0) + 1
        if types['text'] > 0 and types['transition'] > 0:
            element_cooccurrence['text_transition'] = element_cooccurrence.get('text_transition', 0) + 1
        if types['text'] > 0 and types['sticker'] > 0:
            element_cooccurrence['text_sticker'] = element_cooccurrence.get('text_sticker', 0) + 1
        if types['effect'] > 0 and types['transition'] > 0:
            element_cooccurrence['effect_transition'] = element_cooccurrence.get('effect_transition', 0) + 1
        if types['effect'] > 0 and types['sticker'] > 0:
            element_cooccurrence['effect_sticker'] = element_cooccurrence.get('effect_sticker', 0) + 1
        if types['sticker'] > 0 and types['transition'] > 0:
            element_cooccurrence['sticker_transition'] = element_cooccurrence.get('sticker_transition', 0) + 1
        # Scene change combinations
        if types['effect'] > 0 and types['scene_change'] > 0:
            element_cooccurrence['effect_sceneChange'] = element_cooccurrence.get('effect_sceneChange', 0) + 1
        if types['text'] > 0 and types['scene_change'] > 0:
            element_cooccurrence['text_sceneChange'] = element_cooccurrence.get('text_sceneChange', 0) + 1
        if types['transition'] > 0 and types['scene_change'] > 0:
            element_cooccurrence['transition_sceneChange'] = element_cooccurrence.get('transition_sceneChange', 0) + 1
        # Keep gesture/expression for emotional_journey compatibility
        if types['text'] > 0 and types['gesture'] > 0:
            element_cooccurrence['text_gesture'] = element_cooccurrence.get('text_gesture', 0) + 1
        if types['text'] > 0 and types['expression'] > 0:
            element_cooccurrence['text_expression'] = element_cooccurrence.get('text_expression', 0) + 1
        if types['gesture'] > 0 and types['expression'] > 0:
            element_cooccurrence['gesture_expression'] = element_cooccurrence.get('gesture_expression', 0) + 1
    
    # Find dominant combination
    dominant_combination = max(element_cooccurrence.items(), key=lambda x: x[1])[0] if element_cooccurrence else 'none'
    
    # Convert empty seconds to dead zones
    dead_zones = []
    if empty_seconds > 0:
        in_dead_zone = False
        zone_start = None
        
        for i in range(seconds):
            if density_per_second[i] == 0:
                if not in_dead_zone:
                    in_dead_zone = True
                    zone_start = i
            else:
                if in_dead_zone:
                    dead_zones.append({
                        'start': zone_start,
                        'end': i,
                        'duration': i - zone_start
                    })
                    in_dead_zone = False
        
        # Handle case where video ends in dead zone
        if in_dead_zone:
            dead_zones.append({
                'start': zone_start,
                'end': seconds - 1,
                'duration': seconds - 1 - zone_start
            })
    
    # Detect density shifts
    density_shifts = []
    for i in range(1, seconds - 1):
        prev_avg = mean(density_per_second[max(0, i-3):i])
        next_avg = mean(density_per_second[i:min(seconds, i+3)])
        
        if prev_avg > avg_density * 1.5 and next_avg < avg_density * 0.5:
            density_shifts.append({
                'timestamp': i,
                'from': 'high',
                'to': 'low'
            })
        elif prev_avg < avg_density * 0.5 and next_avg > avg_density * 1.5:
            density_shifts.append({
                'timestamp': i,
                'from': 'low',
                'to': 'high'
            })
    
    # Determine classification and patterns
    if avg_density < 0.5:
        density_classification = 'minimal'
    elif avg_density < 1.5:
        density_classification = 'medium'
    else:
        density_classification = 'heavy'
    
    # Pacing style
    if 'consistent_density' in patterns:
        pacing_style = 'consistent'
    elif density_volatility > 0.7:
        pacing_style = 'erratic'
    else:
        pacing_style = 'dynamic'
    
    # Cognitive load
    if max_density > 5 and density_volatility > 0.5:
        cognitive_load_category = 'high'
    elif max_density < 3 and density_volatility < 0.3:
        cognitive_load_category = 'low'
    else:
        cognitive_load_category = 'medium'
    
    # Detection reliability (estimated based on element types)
    detection_reliability = {
        'text': 0.95,
        'effect': 0.90,
        'transition': 0.85,
        'sticker': 0.92,
        'object': 0.88,
        'gesture': 0.87,
        'expression': 0.82
    }
    
    # Overall confidence
    data_completeness = timeline_coverage
    overall_confidence = round(mean([data_completeness, timeline_coverage, 0.9]), 2)
    
    return {
        'density_analysis': {
            # Core metrics expected by prompt
            'average_density': round(avg_density, 2),
            'max_density': max_density,
            'min_density': min_density,
            'std_deviation': round(std_deviation, 1),
            'total_creative_elements': total_elements,
            'element_distribution': element_distribution,
            'scene_changes': scene_changes,
            'timeline_coverage': round(timeline_coverage, 2),
            
            # Dynamics
            'density_curve': density_curve,  # Already structured array
            'density_volatility': round(density_volatility, 2),
            'acceleration_pattern': density_pattern,  # Maps to accelerationPattern
            'density_progression': 'increasing' if patterns and 'crescendo_pattern' in patterns else 'stable',
            'empty_seconds': [i for i in range(seconds) if density_per_second[i] == 0],  # List format
            
            # Interactions
            'multi_modal_peaks': multi_modal_peaks,
            'element_cooccurrence': element_cooccurrence,
            'dominant_combination': dominant_combination,
            
            # Key events
            'peak_density_moments': peak_moments,  # Already correct structure
            'dead_zones': dead_zones,
            'density_shifts': density_shifts,
            
            # Patterns
            'structural_patterns': density_pattern_flags,  # Already boolean object
            'density_classification': density_classification,
            'pacing_style': pacing_style,
            'cognitive_load_category': cognitive_load_category,
            'ml_tags': creative_ml_tags,
            
            # Quality
            'data_completeness': round(data_completeness, 2),
            'detection_reliability': detection_reliability,
            'overall_confidence': overall_confidence,
            
            # Additional fields for compatibility
            'creative_density_score': round(creative_density_score, 1),
            'elements_per_second': round(avg_density, 2),
            'density_pattern': density_pattern,
            'density_pattern_flags': density_pattern_flags,
            'density_volatility': round(density_volatility, 2),
            'creative_ml_tags': creative_ml_tags,
            'timeline_frame_counts': timeline_frame_counts,
            'duration_seconds': int(duration),
            'patterns_identified': patterns,
            'peak_moments': peak_moments
        }
    }
def compute_emotional_metrics(expression_timeline, speech_timeline, gesture_timeline, duration, 
                            sample_interval=1, intensity_threshold=0.6):
    """Compute emotional metrics for ML-ready analysis
    
    Args:
        expression_timeline: Timeline of facial expressions
        speech_timeline: Timeline of speech segments
        gesture_timeline: Timeline of gestures
        duration: Video duration in seconds
        sample_interval: Seconds between emotion samples (default: 1)
        intensity_threshold: Threshold for emotional peaks (default: 0.6)
    """
    # Define standardized emotion labels and valence mapping
    EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
    EMOTION_VALENCE = {
        'joy': 0.8, 'happy': 0.8, 'excited': 0.9,
        'neutral': 0.0, 'calm': 0.1,
        'sadness': -0.6, 'sad': -0.6,
        'anger': -0.8, 'angry': -0.8,
        'fear': -0.7, 'worried': -0.5,
        'surprise': 0.3, 'surprised': 0.3,
        'disgust': -0.9,
        'contemplative': -0.1, 'thoughtful': -0.1
    }
    
    # Initialize tracking
    seconds = int(duration) + 1
    emotion_sequence = []
    emotion_valence = []
    emotion_timestamps = []
    
    # Sample emotions at specified interval
    for i in range(0, seconds, sample_interval):
        timestamp = f"{i}-{min(i+sample_interval, seconds)}s"
        
        # Get emotions in this window
        emotions_in_window = []
        for ts, data in expression_timeline.items():
            ts_second = parse_timestamp_to_seconds(ts)
            if ts_second is not None and i <= ts_second < i + sample_interval:
                if 'expression' in data:
                    emotions_in_window.append(data['expression'])
        
        # Determine dominant emotion
        if emotions_in_window:
            # Most common emotion in window
            from collections import Counter
            emotion_counts = Counter(emotions_in_window)
            dominant = emotion_counts.most_common(1)[0][0]
            
            # Map to standardized emotion
            if dominant in ['happy', 'excited']:
                dominant_std = 'joy'
            elif dominant in ['sad']:
                dominant_std = 'sadness'
            elif dominant in ['contemplative', 'thoughtful']:
                dominant_std = 'neutral'
            else:
                dominant_std = dominant if dominant in EMOTION_LABELS else 'neutral'
            
            emotion_sequence.append(dominant_std)
            emotion_valence.append(EMOTION_VALENCE.get(dominant, 0.0))
            emotion_timestamps.append(timestamp)
        else:
            # No emotion detected - assume neutral
            emotion_sequence.append('neutral')
            emotion_valence.append(0.0)
            emotion_timestamps.append(timestamp)
    
    # Calculate emotion variability
    emotion_variability = stdev(emotion_valence) if len(emotion_valence) > 1 else 0
    
    # Find emotional peaks (top 5 by absolute intensity)
    emotional_peaks = []
    for i, (ts, emotion, valence) in enumerate(zip(emotion_timestamps, emotion_sequence, emotion_valence)):
        if abs(valence) > intensity_threshold:
            emotional_peaks.append({
                'timestamp': ts,
                'emotion': emotion,
                'intensity': abs(valence)
            })
    
    # Sort peaks by intensity
    emotional_peaks.sort(key=lambda x: x['intensity'], reverse=True)
    emotional_peaks = emotional_peaks[:5]  # Top 5 peaks
    
    # Determine dominant emotion
    if emotion_sequence:
        from collections import Counter
        emotion_counter = Counter(emotion_sequence)
        dominant_emotion = emotion_counter.most_common(1)[0][0]
    else:
        dominant_emotion = 'neutral'
    
    # Calculate emotional trajectory
    if len(emotion_valence) >= 3:
        start_val = mean(emotion_valence[:len(emotion_valence)//3])
        end_val = mean(emotion_valence[-len(emotion_valence)//3:])
        
        if end_val - start_val > 0.3:
            emotional_trajectory = 'ascending'
        elif start_val - end_val > 0.3:
            emotional_trajectory = 'descending'
        else:
            # Check for U-shaped
            middle_val = mean(emotion_valence[len(emotion_valence)//3:-len(emotion_valence)//3])
            if start_val > middle_val + 0.2 and end_val > middle_val + 0.2:
                emotional_trajectory = 'u-shaped'
            else:
                emotional_trajectory = 'flat'
    else:
        emotional_trajectory = 'flat'
    
    # Calculate emotion-gesture alignment
    alignment_count = 0
    total_checks = 0
    
    for ts, data in expression_timeline.items():
        if ts in gesture_timeline and 'expression' in data:
            emotion = data['expression']
            gestures = gesture_timeline[ts].get('gestures', [])
            
            # Check for alignment patterns
            if emotion in ['happy', 'excited'] and any(g in ['thumbs_up', 'victory', 'pointing'] for g in gestures):
                alignment_count += 1
            elif emotion in ['sad', 'thoughtful'] and any(g in ['closed_fist', 'open_palm'] for g in gestures):
                alignment_count += 1
            elif emotion == 'surprised' and any(g in ['pointing', 'open_palm'] for g in gestures):
                alignment_count += 1
            
            total_checks += 1
    
    emotion_gesture_alignment = alignment_count / total_checks if total_checks > 0 else 0
    
    # Calculate emotion change rate
    emotion_changes = []
    for i in range(1, len(emotion_valence)):
        change = abs(emotion_valence[i] - emotion_valence[i-1])
        emotion_changes.append(change)
    
    emotion_change_rate = mean(emotion_changes) if emotion_changes else 0
    
    # Calculate additional metrics
    emotional_consistency = 1 - emotion_change_rate if emotion_change_rate <= 1 else 0
    has_high_emotion_peak = any(abs(v) > 0.8 for v in emotion_valence)
    peak_intensity_count = len([v for v in emotion_valence if abs(v) > intensity_threshold])
    
    # Calculate emotion diversity
    unique_emotions = len(set(emotion_sequence))
    emotion_diversity = unique_emotions / len(EMOTION_LABELS) if EMOTION_LABELS else 0
    
    # Calculate positive/negative ratios
    positive_count = len([v for v in emotion_valence if v > 0.1])
    negative_count = len([v for v in emotion_valence if v < -0.1])
    total_samples = len(emotion_valence)
    
    positive_ratio = positive_count / total_samples if total_samples > 0 else 0
    negative_ratio = negative_count / total_samples if total_samples > 0 else 0
    
    # Build emotion valence curve
    emotion_valence_curve = []
    for ts, valence, emotion in zip(emotion_timestamps, emotion_valence, emotion_sequence):
        emotion_valence_curve.append({
            'timestamp': ts,
            'valence': round(valence, 2),
            'emotion': emotion
        })
    
    # Calculate emotion transition matrix
    emotion_transition_matrix = {}
    for i in range(1, len(emotion_sequence)):
        transition = f"{emotion_sequence[i-1]}_to_{emotion_sequence[i]}"
        emotion_transition_matrix[transition] = emotion_transition_matrix.get(transition, 0) + 1
    
    # Normalize transition counts to probabilities
    total_transitions = sum(emotion_transition_matrix.values())
    if total_transitions > 0:
        for key in emotion_transition_matrix:
            emotion_transition_matrix[key] = round(emotion_transition_matrix[key] / total_transitions, 2)
    
    # Calculate valence momentum
    valence_momentum = []
    for i in range(1, len(emotion_valence)):
        momentum = emotion_valence[i] - emotion_valence[i-1]
        valence_momentum.append(momentum)
    
    if valence_momentum:
        positive_momentum = [m for m in valence_momentum if m > 0]
        negative_momentum = [m for m in valence_momentum if m < 0]
        
        valence_momentum_stats = {
            'average_momentum': round(mean(valence_momentum), 3),
            'max_positive_momentum': round(max(positive_momentum), 2) if positive_momentum else 0,
            'max_negative_momentum': round(min(negative_momentum), 2) if negative_momentum else 0,
            'momentum_changes': valence_momentum
        }
    else:
        valence_momentum_stats = {
            'average_momentum': 0,
            'max_positive_momentum': 0,
            'max_negative_momentum': 0,
            'momentum_changes': []
        }
    
    # Calculate peak rhythm metrics
    peak_spacings = []
    peak_times = [p['timestamp'] for p in emotional_peaks]
    for i in range(1, len(peak_times)):
        # Extract seconds from timestamps
        time1 = parse_timestamp_to_seconds(peak_times[i-1])
        time2 = parse_timestamp_to_seconds(peak_times[i])
        if time1 is not None and time2 is not None:
            spacing = time2 - time1
            peak_spacings.append(spacing)
    
    if peak_spacings:
        peak_spacing_mean = mean(peak_spacings)
        peak_spacing_variance = variance(peak_spacings) if len(peak_spacings) > 1 else 0
        
        # Calculate regularity score (inverse of coefficient of variation)
        if peak_spacing_mean > 0:
            cv = (peak_spacing_variance ** 0.5) / peak_spacing_mean
            regularity_score = 1 / (1 + cv)  # Higher score = more regular
        else:
            regularity_score = 0
        
        peak_rhythm = {
            'peak_spacing_mean': round(peak_spacing_mean, 1),
            'peak_spacing_variance': round(peak_spacing_variance, 2),
            'regularity_score': round(regularity_score, 2),
            'peak_count': len(emotional_peaks)
        }
    else:
        peak_rhythm = {
            'peak_spacing_mean': 0,
            'peak_spacing_variance': 0,
            'regularity_score': 0,
            'peak_count': len(emotional_peaks)
        }
    
    # Calculate neutral ratio (missing metric)
    neutral_ratio = 1.0 - positive_ratio - negative_ratio
    neutral_ratio = max(0.0, min(1.0, neutral_ratio))  # Clamp to 0-1
    
    # Build emotion-gesture sync array for detailed timeline
    emotion_gesture_sync = []
    for timestamp in emotion_timestamps:
        if timestamp in expression_timeline and timestamp in gesture_timeline:
            emotion = expression_timeline[timestamp].get('expression', 'neutral')
            gestures = gesture_timeline[timestamp].get('gestures', [])
            
            # Calculate alignment score for this timestamp
            alignment = 1.0 if gestures else 0.0
            if emotion in ['happy', 'excited'] and any(g in ['thumbs_up', 'victory'] for g in gestures):
                alignment = 1.0
            elif emotion in ['sad', 'angry'] and gestures:
                alignment = 0.3  # Misaligned
                
            emotion_gesture_sync.append({
                'timestamp': timestamp,
                'emotion': emotion,
                'gesturePresent': len(gestures) > 0,
                'alignmentScore': round(alignment, 2)
            })
    
    # Build emotion-speech alignment with conflict detection
    emotion_speech_alignment = {
        'overallAlignment': 0.0,
        'conflictsDetected': False,
        'misalignmentCount': 0,
        'alignmentEvents': []
    }
    
    if speech_timeline:
        alignment_events = []
        conflicts = 0
        
        for timestamp in emotion_timestamps:
            if timestamp in expression_timeline and timestamp in speech_timeline:
                visual_emotion = expression_timeline[timestamp].get('expression', 'neutral')
                speech_text = speech_timeline[timestamp].get('text', '')
                
                # Simple sentiment from speech
                speech_sentiment = 'neutral'
                if speech_text:
                    speech_lower = speech_text.lower()
                    if any(word in speech_lower for word in ['love', 'great', 'amazing', 'happy']):
                        speech_sentiment = 'positive'
                    elif any(word in speech_lower for word in ['hate', 'bad', 'terrible', 'sad']):
                        speech_sentiment = 'negative'
                
                # Check congruence
                congruent = True
                if visual_emotion in ['happy', 'excited'] and speech_sentiment == 'negative':
                    congruent = False
                    conflicts += 1
                elif visual_emotion in ['sad', 'angry'] and speech_sentiment == 'positive':
                    congruent = False
                    conflicts += 1
                
                alignment_events.append({
                    'timestamp': timestamp,
                    'visualEmotion': visual_emotion,
                    'speechSentiment': speech_sentiment,
                    'congruent': congruent
                })
        
        emotion_speech_alignment['alignmentEvents'] = alignment_events[:10]  # Limit to 10 events
        emotion_speech_alignment['misalignmentCount'] = conflicts
        emotion_speech_alignment['conflictsDetected'] = conflicts > 0
        emotion_speech_alignment['overallAlignment'] = 1.0 - (conflicts / max(len(alignment_events), 1))
    
    # Cross-modal consistency (average of gesture and speech alignment)
    cross_modal_consistency = (emotion_gesture_alignment + emotion_speech_alignment['overallAlignment']) / 2
    
    # Detect emotion transitions
    emotion_transitions = []
    for i in range(1, len(emotion_sequence)):
        if emotion_sequence[i] != emotion_sequence[i-1]:
            from_emotion = emotion_sequence[i-1]
            to_emotion = emotion_sequence[i]
            transition_key = f"{from_emotion}_to_{to_emotion}"
            
            emotion_transitions.append({
                'timestamp': emotion_timestamps[i],
                'fromEmotion': from_emotion,
                'toEmotion': to_emotion,
                'transitionProbability': emotion_transition_matrix.get(transition_key, 0.0)
            })
    
    # Detect dead zones (low emotional intensity periods)
    dead_zones = []
    zone_start = None
    
    for i, (timestamp, valence) in enumerate(zip(emotion_timestamps, emotion_valence)):
        if abs(valence) < 0.2:  # Low emotion threshold
            if zone_start is None:
                zone_start = i
        else:
            if zone_start is not None:
                # End of dead zone
                if i - zone_start >= 2:  # At least 2 samples
                    dead_zones.append({
                        'start': emotion_timestamps[zone_start],
                        'end': emotion_timestamps[i-1],
                        'averageValence': round(mean(emotion_valence[zone_start:i]), 2)
                    })
                zone_start = None
    
    # Check for dead zone at end
    if zone_start is not None and len(emotion_timestamps) - zone_start >= 2:
        dead_zones.append({
            'start': emotion_timestamps[zone_start],
            'end': emotion_timestamps[-1],
            'averageValence': round(mean(emotion_valence[zone_start:]), 2)
        })
    
    # Classify emotional archetype
    emotional_archetype = 'flat'
    archetype_flags = {
        'hasEmotionalRollercoaster': False,
        'hasFlatArc': False,
        'isPositiveDominant': False,
        'isNegativeDominant': False,
        'isEmotionallyDiverse': False
    }
    
    if emotion_variability > 0.6 and emotion_change_rate > 0.5:
        emotional_archetype = 'rollercoaster'
        archetype_flags['hasEmotionalRollercoaster'] = True
    elif emotional_trajectory == 'flat' and emotion_variability < 0.3:
        emotional_archetype = 'monotonous'
        archetype_flags['hasFlatArc'] = True
    elif positive_ratio > 0.7:
        emotional_archetype = 'steady_positive'
        archetype_flags['isPositiveDominant'] = True
    elif negative_ratio > 0.6:
        emotional_archetype = 'steady_negative'
        archetype_flags['isNegativeDominant'] = True
    elif emotion_diversity > 0.5:
        emotional_archetype = 'volatile'
        archetype_flags['isEmotionallyDiverse'] = True
    
    archetype_flags['isPositiveDominant'] = positive_ratio > 0.6 and negative_ratio < 0.3
    archetype_flags['isNegativeDominant'] = negative_ratio > 0.6 and positive_ratio < 0.3
    archetype_flags['isEmotionallyDiverse'] = emotion_diversity > 0.5
    
    # Generate ML tags
    emotional_ml_tags = []
    if emotion_variability > 0.6:
        emotional_ml_tags.append('high_variability')
    if positive_ratio > 0.7:
        emotional_ml_tags.append('positive_bias')
    if emotional_trajectory in ['ascending', 'u-shaped']:
        emotional_ml_tags.append('uplifting_arc')
    if len(emotional_peaks) >= 3:
        emotional_ml_tags.append('multi_peak')
    if emotion_gesture_alignment > 0.7:
        emotional_ml_tags.append('authentic_expression')
    if emotion_speech_alignment['conflictsDetected']:
        emotional_ml_tags.append('mixed_signals')
    
    # Calculate quality metrics
    data_completeness = len(emotion_sequence) / max(duration / sample_interval, 1)
    expression_detection_rate = len(expression_timeline) / max(duration, 1)
    timeline_coverage = min(len(emotion_timestamps) * sample_interval / duration, 1.0) if duration > 0 else 0
    
    return {
        # Core metrics
        'emotion_variability': round(emotion_variability, 2),
        'emotion_sequence': emotion_sequence,
        'emotion_valence': [round(v, 2) for v in emotion_valence],
        'emotion_timestamps': emotion_timestamps,
        'emotional_peaks': emotional_peaks,
        'dominant_emotion': dominant_emotion,
        'emotional_trajectory': emotional_trajectory,
        'emotion_gesture_alignment': round(emotion_gesture_alignment, 2),
        'emotion_change_rate': round(emotion_change_rate, 2),
        'emotional_consistency': round(emotional_consistency, 2),
        'has_high_emotion_peak': has_high_emotion_peak,
        'peak_intensity_count': peak_intensity_count,
        'emotion_diversity': round(emotion_diversity, 2),
        'positive_ratio': round(positive_ratio, 2),
        'negative_ratio': round(negative_ratio, 2),
        'neutral_ratio': round(neutral_ratio, 2),  # NEW
        'emotion_valence_curve': emotion_valence_curve,
        'emotion_transition_matrix': emotion_transition_matrix,
        'valence_momentum': valence_momentum_stats,
        'peak_rhythm': peak_rhythm,
        
        # Detailed interactions (NEW)
        'emotion_gesture_sync': emotion_gesture_sync,
        'emotion_speech_alignment': emotion_speech_alignment,
        'cross_modal_consistency': round(cross_modal_consistency, 2),
        
        # Key events (NEW)
        'emotion_transitions': emotion_transitions,
        'dead_zones': dead_zones,
        
        # Pattern classification (NEW)
        'emotional_archetype': emotional_archetype,
        'archetype_flags': archetype_flags,
        'emotional_ml_tags': emotional_ml_tags,
        
        # Quality metrics (NEW)
        'data_completeness': round(data_completeness, 2),
        'expression_detection_rate': round(expression_detection_rate, 2),
        'timeline_coverage': round(timeline_coverage, 2),
        
        'analysis_parameters': {
            'sample_interval': sample_interval,
            'intensity_threshold': intensity_threshold
        }
    }
def compute_metadata_analysis_metrics(static_metadata, metadata_summary, video_duration):
    """Compute comprehensive metadata metrics for ML analysis
    
    Args:
        static_metadata: Raw metadata from API
        metadata_summary: Processed metadata summary
        video_duration: Video duration in seconds
        
    Returns:
        dict: Structured metadata metrics for ML
    """
    import re
    from datetime import datetime
    
    # Extract core data
    caption_text = static_metadata.get('captionText', '')
    hashtags = static_metadata.get('hashtags', [])
    stats = static_metadata.get('stats', {})
    create_time_str = static_metadata.get('createTime', '')
    author = static_metadata.get('author', {})
    
    # Parse timestamp
    try:
        create_time = datetime.fromisoformat(create_time_str.replace('Z', '+00:00'))
        publish_hour = create_time.hour
        publish_day_of_week = create_time.weekday()  # 0=Monday, 6=Sunday
    except:
        publish_hour = 0
        publish_day_of_week = 0
    
    # Caption analysis
    words = caption_text.split()
    word_count = len(words)
    
    # Emoji detection
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    emojis = emoji_pattern.findall(caption_text)
    emoji_count = len(emojis)
    
    # Mention detection (@username)
    mention_pattern = re.compile(r'@\w+')
    mentions = mention_pattern.findall(caption_text)
    mention_count = len(mentions)
    
    # Link detection
    link_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    links = link_pattern.findall(caption_text)
    link_present = len(links) > 0
    
    # Hashtag analysis
    hashtag_names = [h.get('name', '') for h in hashtags]
    hashtag_count = len(hashtag_names)
    
    # Classify hashtags
    generic_hashtags = ['fyp', 'foryou', 'foryoupage', 'viral', 'trending', 'explore']
    niche_count = 0
    generic_count = 0
    
    for tag in hashtag_names:
        tag_lower = tag.lower()
        if tag_lower in generic_hashtags:
            generic_count += 1
        else:
            niche_count += 1
    
    # Readability approximation (simple version)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    sentence_endings = caption_text.count('.') + caption_text.count('!') + caption_text.count('?')
    sentence_count = max(sentence_endings, 1)
    avg_sentence_length = word_count / sentence_count
    
    # Simple readability score (0-1, higher is easier)
    readability_score = min(1.0, max(0.0, 1.0 - (avg_word_length - 4) * 0.1 - (avg_sentence_length - 15) * 0.02))
    
    # Sentiment analysis (basic)
    positive_words = ['love', 'amazing', 'great', 'awesome', 'best', 'perfect', 'beautiful', 'excellent', 'happy', 'good']
    negative_words = ['hate', 'bad', 'worst', 'terrible', 'awful', 'horrible', 'ugly', 'sad', 'angry', 'poor']
    
    caption_lower = caption_text.lower()
    positive_count = sum(1 for word in positive_words if word in caption_lower)
    negative_count = sum(1 for word in negative_words if word in caption_lower)
    
    sentiment_polarity = (positive_count - negative_count) / max(word_count, 1)
    if sentiment_polarity > 0.1:
        sentiment_category = 'positive'
    elif sentiment_polarity < -0.1:
        sentiment_category = 'negative'
    else:
        sentiment_category = 'neutral'
    
    # Urgency detection
    urgency_high = ['now', 'today', 'last chance', 'ends soon', 'limited time', 'hurry']
    urgency_medium = ['limited', "don't miss", 'quick', 'fast', 'soon']
    
    has_high_urgency = any(phrase in caption_lower for phrase in urgency_high)
    has_medium_urgency = any(phrase in caption_lower for phrase in urgency_medium)
    
    if has_high_urgency:
        urgency_level = 'high'
    elif has_medium_urgency:
        urgency_level = 'medium'
    else:
        urgency_level = 'none'
    
    # Hook detection
    hook_patterns = [
        'wait for it', 'watch till end', 'you won\'t believe', 'this is how',
        'the secret to', 'why you should', 'how to', 'what happens when'
    ]
    
    hooks = []
    for pattern in hook_patterns:
        if pattern in caption_lower:
            position = caption_lower.find(pattern)
            relative_position = position / len(caption_text) if caption_text else 0
            if relative_position < 0.2:
                position_label = 'start'
            elif relative_position > 0.8:
                position_label = 'end'
            else:
                position_label = 'middle'
            
            hooks.append({
                'text': pattern,
                'position': position_label,
                'type': 'curiosity' if 'believe' in pattern or 'wait' in pattern else 'promise',
                'strength': 0.8 if position_label == 'start' else 0.5
            })
    
    # CTA detection
    cta_patterns = {
        'follow': ['follow for more', 'follow me', 'hit follow'],
        'like': ['drop a like', 'hit like', 'like if', 'double tap'],
        'comment': ['comment below', 'let me know', 'tell me', 'drop a comment'],
        'share': ['share this', 'tag someone', 'send this to']
    }
    
    ctas = []
    for cta_type, patterns in cta_patterns.items():
        for pattern in patterns:
            if pattern in caption_lower:
                ctas.append({
                    'text': pattern,
                    'type': cta_type,
                    'explicitness': 'direct',
                    'urgency': urgency_level
                })
    
    # Linguistic markers
    question_count = caption_text.count('?')
    exclamation_count = caption_text.count('!')
    caps_words = len([w for w in words if w.isupper() and len(w) > 1])
    personal_pronouns = ['i', 'me', 'my', 'you', 'your', 'we', 'our']
    pronoun_count = sum(1 for word in words if word.lower() in personal_pronouns)
    
    # Caption style classification
    if word_count < 10:
        caption_style = 'minimal'
    elif question_count > 0 and caption_text.strip().endswith('?'):
        caption_style = 'question'
    elif sentence_count > 3:
        caption_style = 'storytelling'
    elif any(str(i) in caption_text for i in range(1, 10)):
        caption_style = 'list'
    else:
        caption_style = 'direct'
    
    # Hashtag strategy
    if hashtag_count == 0:
        hashtag_strategy = 'none'
    elif hashtag_count < 3:
        hashtag_strategy = 'minimal'
    elif hashtag_count <= 7:
        hashtag_strategy = 'moderate'
    elif hashtag_count <= 15:
        hashtag_strategy = 'heavy'
    else:
        hashtag_strategy = 'spam'
    
    # Engagement calculations
    view_count = stats.get('views', 0)
    like_count = stats.get('likes', 0)
    comment_count = stats.get('comments', 0)
    share_count = stats.get('shares', 0)
    
    engagement_rate = ((like_count + comment_count + share_count) / view_count * 100) if view_count > 0 else 0
    likes_to_views = like_count / view_count if view_count > 0 else 0
    comments_to_views = comment_count / view_count if view_count > 0 else 0
    shares_to_views = share_count / view_count if view_count > 0 else 0
    
    # Viral potential score (0-1)
    viral_score = 0
    if engagement_rate > 15:
        viral_score += 0.3
    elif engagement_rate > 10:
        viral_score += 0.2
    elif engagement_rate > 5:
        viral_score += 0.1
    
    if hooks:
        viral_score += 0.2
    if hashtag_strategy == 'moderate':
        viral_score += 0.1
    if caption_style in ['question', 'storytelling']:
        viral_score += 0.1
    if urgency_level != 'none':
        viral_score += 0.1
    if sentiment_category == 'positive':
        viral_score += 0.1
    if emoji_count > 0 and emoji_count < 5:
        viral_score += 0.1
    
    viral_score = min(1.0, viral_score)
    
    # Caption and hashtag quality
    if word_count > 10 and readability_score > 0.7:
        caption_quality = 'high'
    elif word_count > 5 and readability_score > 0.5:
        caption_quality = 'medium'
    elif word_count > 0:
        caption_quality = 'low'
    else:
        caption_quality = 'empty'
    
    if generic_count > hashtag_count * 0.5:
        hashtag_quality = 'spammy'
    elif hashtag_count == 0:
        hashtag_quality = 'none'
    elif generic_count <= 2:
        hashtag_quality = 'relevant'
    else:
        hashtag_quality = 'mixed'
    
    # Structure the output
    return {
        # Core metrics
        'caption_length': len(caption_text),
        'word_count': word_count,
        'hashtag_count': hashtag_count,
        'emoji_count': emoji_count,
        'emoji_list': emojis,
        'mention_count': mention_count,
        'mention_list': mentions,
        'link_present': link_present,
        'video_duration': video_duration,
        'publish_hour': publish_hour,
        'publish_day_of_week': publish_day_of_week,
        'view_count': view_count,
        'like_count': like_count,
        'comment_count': comment_count,
        'share_count': share_count,
        'engagement_rate': round(engagement_rate, 2),
        
        # Dynamics
        'hashtag_strategy': hashtag_strategy,
        'caption_style': caption_style,
        'emoji_density': round(emoji_count / word_count if word_count > 0 else 0, 2),
        'mention_density': round(mention_count / word_count if word_count > 0 else 0, 2),
        'readability_score': round(readability_score, 2),
        'sentiment_polarity': round(sentiment_polarity, 2),
        'sentiment_category': sentiment_category,
        'urgency_level': urgency_level,
        'viral_potential_score': round(viral_score, 2),
        
        # Interactions
        'hashtag_counts': {
            'niche_count': niche_count,
            'generic_count': generic_count
        },
        'engagement_alignment': {
            'likes_to_views_ratio': round(likes_to_views, 4),
            'comments_to_views_ratio': round(comments_to_views, 4),
            'shares_to_views_ratio': round(shares_to_views, 4),
            'above_average_engagement': engagement_rate > 5.0
        },
        'creator_context': {
            'username': author.get('username', 'unknown'),
            'verified': author.get('verified', False)
        },
        
        # Key events
        'hashtags_detailed': [
            {
                'tag': f"#{name}",
                'position': i + 1,
                'type': 'generic' if name.lower() in generic_hashtags else 'niche',
                'estimated_reach': 'high' if name.lower() in generic_hashtags else 'medium'
            }
            for i, name in enumerate(hashtag_names[:10])  # Limit to 10
        ],
        'emojis_detailed': [
            {
                'emoji': emoji,
                'count': emojis.count(emoji),
                'sentiment': 'positive',  # Simplified
                'emphasis': True if emojis.count(emoji) > 1 else False
            }
            for emoji in set(emojis[:5])  # Unique emojis, limit to 5
        ],
        'hooks': hooks,
        'call_to_actions': ctas,
        'linguistic_markers': {
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'caps_lock_words': caps_words,
            'personal_pronoun_count': pronoun_count
        },
        'hashtag_patterns': {
            'lead_with_generic': hashtag_names[0].lower() in generic_hashtags if hashtag_names else False,
            'all_caps': any(tag.isupper() for tag in hashtag_names)
        },
        
        # Quality
        'caption_quality': caption_quality,
        'hashtag_quality': hashtag_quality,
        
        # Caption text for validation
        'caption_text': caption_text,
        'hashtag_list': hashtag_names
    }


def check_intro_person_present(expression_timeline, object_timeline):
    """Check if person is present in the first 3 seconds"""
    for i in range(3):
        timestamp = f"{i}-{i+1}s"
        # Check expression timeline
        if timestamp in expression_timeline and expression_timeline[timestamp].get('expression'):
            return True
        # Check object timeline
        if timestamp in object_timeline:
            objects = object_timeline[timestamp].get('objects', {})
            if 'person' in objects and objects['person'] > 0:
                return True
    return False


def classify_transition_type(from_shot, to_shot):
    """Classify the type of shot transition"""
    if from_shot == to_shot:
        return "cut"  # Same shot type, just a cut
    
    shot_values = {'close': 3, 'medium': 2, 'far': 1, 'unknown': 0}
    from_val = shot_values.get(from_shot, 0)
    to_val = shot_values.get(to_shot, 0)
    
    if to_val > from_val:
        return "zoom_in"
    elif to_val < from_val:
        return "zoom_out"
    else:
        return "cut"


def compute_detection_coverage(expression_timeline, object_timeline, duration):
    """Compute detection success rates"""
    total_seconds = int(duration) + 1
    face_detected = 0
    person_detected = 0
    
    for i in range(total_seconds):
        timestamp = f"{i}-{i+1}s"
        # Count face detections
        if timestamp in expression_timeline and expression_timeline[timestamp].get('expression'):
            face_detected += 1
        # Count person detections
        if timestamp in object_timeline:
            objects = object_timeline[timestamp].get('objects', {})
            if 'person' in objects and objects['person'] > 0:
                person_detected += 1
    
    return {
        'faceDetectionRate': round(face_detected / total_seconds, 2) if total_seconds > 0 else 0,
        'personDetectionRate': round(person_detected / total_seconds, 2) if total_seconds > 0 else 0,
        'gazeDataAvailable': False,  # Will be updated based on enhanced_human_data
        'actionDataAvailable': False,  # Will be updated based on enhanced_human_data
        'backgroundDataAvailable': False  # Will be updated based on enhanced_human_data
    }


def compute_framing_archetype(face_ratio, person_ratio, action_diversity, framing_volatility):
    """Compute framing archetype based on rules"""
    if face_ratio > 0.7 and framing_volatility < 0.3:
        return "talking_head"
    elif action_diversity and action_diversity > 5 and framing_volatility > 0.5:
        return "dynamic_demo"
    elif person_ratio < 0.3:
        return "product_focused"
    elif 0.3 <= person_ratio <= 0.7:
        return "balanced_presence"
    else:
        return "unknown"


def analyze_temporal_evolution(expression_timeline, object_timeline, camera_distance_timeline, duration):
    """Analyze how framing evolves over time"""
    # Divide video into thirds
    third_duration = duration / 3
    
    evolution = {
        'first_third': {'face_presence': 0, 'person_presence': 0, 'camera_distance': {}},
        'middle_third': {'face_presence': 0, 'person_presence': 0, 'camera_distance': {}},
        'final_third': {'face_presence': 0, 'person_presence': 0, 'camera_distance': {}}
    }
    
    # Count presence in each third
    for timestamp in expression_timeline:
        second = parse_timestamp_to_seconds(timestamp)
        if second is None:
            continue
            
        if expression_timeline[timestamp].get('expression'):
            if second < third_duration:
                evolution['first_third']['face_presence'] += 1
            elif second < 2 * third_duration:
                evolution['middle_third']['face_presence'] += 1
            else:
                evolution['final_third']['face_presence'] += 1
    
    for timestamp in object_timeline:
        second = parse_timestamp_to_seconds(timestamp)
        if second is None:
            continue
            
        objects = object_timeline[timestamp].get('objects', {})
        if 'person' in objects and objects['person'] > 0:
            if second < third_duration:
                evolution['first_third']['person_presence'] += 1
            elif second < 2 * third_duration:
                evolution['middle_third']['person_presence'] += 1
            else:
                evolution['final_third']['person_presence'] += 1
    
    # Analyze camera distance evolution
    for timestamp in camera_distance_timeline:
        second = parse_timestamp_to_seconds(timestamp)
        if second is None:
            continue
            
        distance = camera_distance_timeline[timestamp].get('distance', 'medium').lower()
        
        if second < third_duration:
            third = 'first_third'
        elif second < 2 * third_duration:
            third = 'middle_third'
        else:
            third = 'final_third'
        
        if distance not in evolution[third]['camera_distance']:
            evolution[third]['camera_distance'][distance] = 0
        evolution[third]['camera_distance'][distance] += 1
    
    # Calculate dominant camera distance for each third
    for third in evolution:
        distances = evolution[third]['camera_distance']
        if distances:
            evolution[third]['dominant_distance'] = max(distances, key=distances.get)
        else:
            evolution[third]['dominant_distance'] = 'unknown'
    
    return evolution


def infer_video_intent(face_ratio, person_ratio, intro_shot_type, action_recognition, shot_distribution):
    """Infer the primary intent of the video based on framing patterns"""
    # Default intent
    intent = 'general_content'
    
    # High face presence suggests personal/educational content
    if face_ratio > 0.6:
        if shot_distribution.get('close', 0) > 0.5:
            intent = 'personal_message'
        else:
            intent = 'educational'
    
    # Low person presence suggests product/scene focused
    elif person_ratio < 0.3:
        intent = 'product_showcase'
    
    # Action-heavy content
    elif action_recognition and action_recognition.get('action_diversity', 0) > 3:
        primary_actions = action_recognition.get('primary_actions', [])
        if 'dancing' in primary_actions:
            intent = 'entertainment_dance'
        elif 'cooking' in primary_actions or 'eating' in primary_actions:
            intent = 'tutorial_cooking'
        else:
            intent = 'tutorial_demonstration'
    
    # Intro shot type can override
    if intro_shot_type == 'close' and face_ratio > 0.5:
        intent = 'personal_message'
    elif intro_shot_type == 'far' and person_ratio > 0.5:
        intent = 'performance'
    
    return intent


def calculate_intent_alignment_risk(intent, face_ratio, volatility, shot_distribution, pattern_tags):
    """Calculate risk score for intent-framing misalignment"""
    risk_score = 0.0
    
    # Personal message should have stable close shots
    if intent == 'personal_message':
        if volatility > 0.5:
            risk_score += 0.3
        if shot_distribution.get('close', 0) < 0.5:
            risk_score += 0.3
        if 'stable_framing' not in pattern_tags:
            risk_score += 0.2
    
    # Product showcase should have varied shots
    elif intent == 'product_showcase':
        if face_ratio > 0.5:
            risk_score += 0.3
        if volatility < 0.3:
            risk_score += 0.2
    
    # Tutorial should be clear and stable
    elif 'tutorial' in intent:
        if volatility > 0.7:
            risk_score += 0.4
        if shot_distribution.get('far', 0) > 0.5:
            risk_score += 0.2
    
    # Entertainment should be dynamic
    elif 'entertainment' in intent:
        if volatility < 0.3:
            risk_score += 0.3
        if 'dynamic_framing' not in pattern_tags:
            risk_score += 0.2
    
    return min(1.0, risk_score)


def compute_person_framing_metrics(expression_timeline, object_timeline, camera_distance_timeline,
                                  person_timeline, enhanced_human_data, duration):
    """Compute person framing metrics for ML-ready analysis
    
    Args:
        expression_timeline: Timeline of facial expressions
        object_timeline: Timeline of detected objects
        camera_distance_timeline: Timeline of camera distances
        person_timeline: Timeline of person detections (currently unused)
        enhanced_human_data: Enhanced human analysis data from metadata
        duration: Video duration in seconds
        
    Returns:
        dict: Comprehensive person framing metrics
    """
    # Initialize tracking
    seconds = int(duration) + 1
    face_frames = 0
    person_frames = 0
    
    # Count face presence from expression timeline
    for timestamp in expression_timeline:
        if expression_timeline[timestamp].get('expression'):
            face_frames += 1
    
    # Count person presence from object timeline
    for timestamp, data in object_timeline.items():
        objects = data.get('objects', {})
        if 'person' in objects and objects['person'] > 0:
            person_frames += 1
    
    # Calculate basic ratios
    total_frames = len(expression_timeline) if expression_timeline else seconds
    face_screen_time_ratio = face_frames / total_frames if total_frames > 0 else 0
    person_screen_time_ratio = person_frames / total_frames if total_frames > 0 else 0
    
    # Analyze camera distances
    distance_counts = {'close': 0, 'medium': 0, 'far': 0}
    distance_changes = 0
    last_distance = None
    intro_shot_type = 'unknown'
    
    for timestamp in sorted(camera_distance_timeline.keys()):
        distance_data = camera_distance_timeline[timestamp]
        distance = distance_data.get('distance', 'medium').lower()
        
        # Count distances
        if distance in distance_counts:
            distance_counts[distance] += 1
        
        # Track changes
        if last_distance and last_distance != distance:
            distance_changes += 1
        last_distance = distance
        
        # Capture intro shot type (first 3 seconds)
        second = parse_timestamp_to_seconds(timestamp)
        if second is not None and second < 3 and intro_shot_type == 'unknown':
            intro_shot_type = distance
    
    # Calculate distance metrics
    total_distance_frames = sum(distance_counts.values())
    if total_distance_frames > 0:
        shot_type_distribution = {
            k: round(v / total_distance_frames, 2) 
            for k, v in distance_counts.items()
        }
        dominant_shot_type = max(distance_counts, key=distance_counts.get)
        avg_camera_distance = dominant_shot_type
    else:
        shot_type_distribution = {'close': 0, 'medium': 0, 'far': 0}
        dominant_shot_type = 'unknown'
        avg_camera_distance = 'unknown'
        
        # Try to infer intro shot type from face presence in first 3 seconds
        if intro_shot_type == 'unknown':
            face_in_intro = False
            for i in range(min(3, seconds)):
                timestamp = f"{i}-{i+1}s"
                if timestamp in expression_timeline and expression_timeline[timestamp].get('expression'):
                    face_in_intro = True
                    break
            intro_shot_type = 'face_focused' if face_in_intro else 'no_person'
    
    # Calculate framing volatility
    framing_volatility = distance_changes / total_frames if total_frames > 1 else 0

    # Build distance evolution for shot transition detection
    distance_evolution = []
    for timestamp in sorted(camera_distance_timeline.keys()):
        second = parse_timestamp_to_seconds(timestamp)
        if second is not None:
            distance_data = camera_distance_timeline[timestamp]
            distance_value = {'close': 0.2, 'medium': 0.5, 'far': 0.8}.get(
                distance_data.get('distance', 'medium').lower(), 0.5
            )
            distance_evolution.append({
                'time': second,
                'distance': distance_value
            })

    # Detect shot transitions from scene changes
    shot_transitions = []
    # Note: scene_change_timeline is not available in this function
    # We can detect transitions from camera distance changes instead
    for i in range(1, len(distance_evolution)):
        if abs(distance_evolution[i]['distance'] - distance_evolution[i-1]['distance']) > 0.3:
            shot_transitions.append({
                'timestamp': f"{distance_evolution[i]['time']}s",
                'time': distance_evolution[i]['time'],
                'shot_number': i,
                'type': 'distance_change'
            })
    
    # If we have scene changes but no camera distance changes, update volatility
    if not distance_changes and len(shot_transitions) > 0:
        # Use scene changes as a proxy for framing changes
        framing_volatility = len(shot_transitions) / duration if duration > 0 else 0
        
    # Track subject absence
    absence_segments = []
    current_absence_start = None
    subject_absence_count = 0
    
    for i in range(seconds):
        timestamp = f"{i}-{i+1}s"
        
        # Check if person is present in this second
        person_present = False
        
        # Check expression timeline
        if timestamp in expression_timeline and expression_timeline[timestamp].get('expression'):
            person_present = True
        
        # Check object timeline
        if timestamp in object_timeline:
            objects = object_timeline[timestamp].get('objects', {})
            if 'person' in objects and objects['person'] > 0:
                person_present = True
        
        # Track absence segments
        if not person_present:
            subject_absence_count += 1
            if current_absence_start is None:
                current_absence_start = i
        else:
            if current_absence_start is not None:
                absence_segments.append({
                    'start': current_absence_start,
                    'end': i,
                    'duration': i - current_absence_start
                })
                current_absence_start = None
    
    # Handle final absence segment
    if current_absence_start is not None:
        absence_segments.append({
            'start': current_absence_start,
            'end': seconds,
            'duration': seconds - current_absence_start
        })
    
    # Calculate longest absence
    longest_absence_duration = max([s['duration'] for s in absence_segments]) if absence_segments else 0
    
    # Generate pattern tags
    person_framing_pattern_tags = []
    
    # Creator presence patterns
    if face_screen_time_ratio > 0.7:
        person_framing_pattern_tags.append('strong_creator_presence')
    elif face_screen_time_ratio < 0.3:
        person_framing_pattern_tags.append('minimal_creator_presence')
    
    # Absence patterns
    if len(absence_segments) > 5:
        person_framing_pattern_tags.append('cutaway_heavy')
    elif len(absence_segments) == 0:
        person_framing_pattern_tags.append('continuous_presence')
    
    # Framing stability patterns
    if framing_volatility < 0.2:
        person_framing_pattern_tags.append('stable_framing')
    elif framing_volatility > 0.6:
        person_framing_pattern_tags.append('dynamic_framing')
    
    # Style patterns
    if face_screen_time_ratio > 0.6 and distance_counts.get('close', 0) > distance_counts.get('far', 0):
        person_framing_pattern_tags.append('talking_head_style')
    elif person_screen_time_ratio > 0.7 and distance_counts.get('far', 0) > distance_counts.get('close', 0):
        person_framing_pattern_tags.append('full_body_content')
    
    # Extract data from enhanced_human_data if available
    gaze_analysis = None
    action_recognition = None
    background_analysis = None
    
    if enhanced_human_data:
        # Use enhanced data if available
        if 'face_screen_time_ratio' in enhanced_human_data:
            face_screen_time_ratio = enhanced_human_data['face_screen_time_ratio']
        if 'person_screen_time_ratio' in enhanced_human_data:
            person_screen_time_ratio = enhanced_human_data['person_screen_time_ratio']
        if 'avg_camera_distance' in enhanced_human_data:
            avg_camera_distance = enhanced_human_data['avg_camera_distance']
        if 'framing_volatility' in enhanced_human_data:
            framing_volatility = enhanced_human_data['framing_volatility']
        if 'dominant_shot_type' in enhanced_human_data:
            dominant_shot_type = enhanced_human_data['dominant_shot_type']
        if 'intro_shot_type' in enhanced_human_data:
            intro_shot_type = enhanced_human_data['intro_shot_type']
        if 'subject_absence_count' in enhanced_human_data:
            subject_absence_count = enhanced_human_data['subject_absence_count']
        
        # Extract gaze analysis
        if 'gaze_patterns' in enhanced_human_data:
            gaze_data = enhanced_human_data['gaze_patterns']
            gaze_analysis = {
                'eye_contact_ratio': gaze_data.get('eye_contact_ratio', 0),
                'primary_gaze_direction': 'camera' if isinstance(gaze_data.get('primary_gaze_direction'), dict) else gaze_data.get('primary_gaze_direction', 'unknown')
            }
        
        # Extract action recognition
        if 'primary_actions' in enhanced_human_data:
            actions = enhanced_human_data['primary_actions']
            if isinstance(actions, dict):
                # Extract top actions
                top_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:3]
                primary_actions = [action[0] for action in top_actions]
                action_diversity = len([a for a in actions.values() if a > 0])
            else:
                primary_actions = actions if isinstance(actions, list) else []
                action_diversity = len(primary_actions)
            
            action_recognition = {
                'primary_actions': primary_actions,
                'action_diversity': action_diversity
            }
        
        # Extract background analysis
        if 'scene_analysis' in enhanced_human_data:
            scene_data = enhanced_human_data['scene_analysis']
            if 'background_changes' in scene_data:
                bg_data = scene_data['background_changes']
                background_analysis = {
                    'background_stability': bg_data.get('background_stability', 'unknown'),
                    'background_changes': bg_data.get('background_changes', []),
                    'change_frequency': bg_data.get('change_frequency', 0),
                    'avg_change_magnitude': bg_data.get('avg_change_magnitude', 0)
                }
    
    # Build final metrics
    metrics = {
        'face_screen_time_ratio': round(face_screen_time_ratio, 2),
        'person_screen_time_ratio': round(person_screen_time_ratio, 2),
        'avg_camera_distance': avg_camera_distance,
        'framing_volatility': round(framing_volatility, 2),
        'dominant_shot_type': dominant_shot_type,
        'intro_shot_type': intro_shot_type,
        'subject_absence_count': subject_absence_count,
        'person_framing_pattern_tags': person_framing_pattern_tags,
        'shot_type_distribution': shot_type_distribution,
        'longest_absence_duration': longest_absence_duration,
        'shot_transitions': shot_transitions,
        'absence_segments': absence_segments
    }
    
    # Add optional enhanced metrics
    if gaze_analysis:
        metrics['gaze_analysis'] = gaze_analysis
    if action_recognition:
        metrics['action_recognition'] = action_recognition
    if background_analysis:
        metrics['background_analysis'] = background_analysis
    
    # Compute video intent based on metrics
    video_intent = infer_video_intent(
        face_screen_time_ratio, person_screen_time_ratio,
        intro_shot_type, action_recognition, shot_type_distribution
    )
    metrics['video_intent'] = video_intent
    
    # Add gaze steadiness proxy (using eye contact ratio)
    if gaze_analysis and 'eye_contact_ratio' in gaze_analysis:
        # High eye contact suggests steady gaze
        eye_contact = gaze_analysis['eye_contact_ratio']
        if eye_contact > 0.7:
            metrics['gaze_steadiness'] = 'high'
        elif eye_contact > 0.4:
            metrics['gaze_steadiness'] = 'medium'
        else:
            metrics['gaze_steadiness'] = 'low'
    else:
        metrics['gaze_steadiness'] = 'unknown'
    
    # === NEW COMPUTATIONS ===
    
    # 1. Check intro person present
    intro_person_present = check_intro_person_present(expression_timeline, object_timeline)
    
    # 2. Classify transition types
    enhanced_transitions = []
    for transition in shot_transitions:
        enhanced = transition.copy()
        enhanced['transitionType'] = classify_transition_type(
            transition.get('fromShot', 'unknown'),
            transition.get('toShot', 'unknown')
        )
        enhanced_transitions.append(enhanced)
    metrics['shot_transitions'] = enhanced_transitions
    
    # 3. Filter prolonged absences (>5 seconds)
    prolonged_absences = [
        segment for segment in absence_segments 
        if segment['duration'] > 5
    ]
    
    # 4. Compute framing archetype
    action_diversity = action_recognition.get('action_diversity', 0) if action_recognition else 0
    framing_archetype = compute_framing_archetype(
        face_screen_time_ratio, person_screen_time_ratio,
        action_diversity, framing_volatility
    )
    
    # 5. Generate additional ML tags
    ml_tags = []
    if face_screen_time_ratio > 0.6:
        ml_tags.append("high_face_time")
    if framing_volatility < 0.2:
        ml_tags.append("low_volatility")
    if intro_shot_type == 'close':
        ml_tags.append("close_shot_intro")
    if person_screen_time_ratio > 0.8:
        ml_tags.append("person_dominant")
    if len(prolonged_absences) > 0:
        ml_tags.append("has_prolonged_absences")
    if action_diversity > 3:
        ml_tags.append("high_action_diversity")
    
    # 6. Compute detection coverage
    detection_coverage = compute_detection_coverage(expression_timeline, object_timeline, duration)
    # Update availability flags based on enhanced data
    detection_coverage['gazeDataAvailable'] = bool(gaze_analysis)
    detection_coverage['actionDataAvailable'] = bool(action_recognition)
    detection_coverage['backgroundDataAvailable'] = bool(background_analysis)
    
    # 7. Compute data completeness
    data_sources = [
        expression_timeline,
        object_timeline,
        camera_distance_timeline,
        gaze_analysis,
        action_recognition,
        background_analysis
    ]
    available_sources = sum(1 for source in data_sources if source)
    data_completeness = round(available_sources / len(data_sources), 2)
    
    # 8. Compute confidence scores
    # Core metrics confidence based on detection rates
    core_confidence = round(mean([
        detection_coverage['faceDetectionRate'],
        detection_coverage['personDetectionRate'],
        0.9  # Base confidence for timeline data
    ]), 2)
    
    # Dynamics confidence based on transitions and evolution
    dynamics_confidence = round(0.85 if len(shot_transitions) > 0 else 0.7, 2)
    
    # Interactions confidence based on available enhanced data
    interactions_confidence = round(mean([
        1.0 if gaze_analysis else 0.5,
        1.0 if action_recognition else 0.5,
        1.0 if background_analysis else 0.5
    ]), 2)
    
    # Overall confidence
    overall_confidence = round(mean([
        core_confidence,
        dynamics_confidence,
        interactions_confidence,
        data_completeness
    ]), 2)
    
    # Add all new metrics
    metrics.update({
        'intro_person_present': intro_person_present,
        'prolonged_absences': prolonged_absences,
        'framing_archetype': framing_archetype,
        'ml_tags': ml_tags,
        'detection_coverage': detection_coverage,
        'data_completeness': data_completeness,
        'confidence_scores': {
            'core': core_confidence,
            'dynamics': dynamics_confidence,
            'interactions': interactions_confidence,
            'overall': overall_confidence
        }
    })
    
    return metrics


def compute_scene_pacing_metrics(scene_timeline, video_duration, object_timeline=None, camera_distance_timeline=None, video_id=None):
    """Compute scene pacing metrics for ML-ready analysis
    
    Args:
        scene_timeline: Timeline of scene changes
        video_duration: Video duration in seconds
        object_timeline: Timeline of detected objects (optional)
        camera_distance_timeline: Timeline of camera distances (optional)
        video_id: Video ID for FPS context lookup (optional)
        
    Returns:
        dict: Comprehensive scene pacing metrics
    """
    # Initialize metrics
    shot_durations = []
    shot_transitions = []
    scene_changes = []
    
    # Import FPS context manager if video_id provided
    fps_manager = None
    if video_id:
        try:
            from fps_utils import FPSContextManager
            fps_manager = FPSContextManager(video_id)
        except ImportError:
            print("⚠️ FPS context manager not available, using legacy behavior")
    
    # Convert scene timeline to sorted list
    timestamps = sorted(scene_timeline.keys(), key=lambda x: parse_timestamp_to_seconds(x) or 0)
    
    # Calculate shot durations
    for i in range(len(timestamps)):
        current_time = parse_timestamp_to_seconds(timestamps[i])
        if current_time is None:
            continue
            
        # Get next timestamp or use video duration
        if i < len(timestamps) - 1:
            next_time = parse_timestamp_to_seconds(timestamps[i + 1])
            if next_time is not None:
                duration = next_time - current_time
                shot_durations.append(duration)
        else:
            # Last shot duration
            duration = video_duration - current_time
            if duration > 0:
                shot_durations.append(duration)
        
        # Track scene change data
        scene_data = scene_timeline[timestamps[i]]
        # Handle both formats: type='shot_change' or scene_change=True
        if scene_data.get('type') in ['scene_change', 'shot_change'] or scene_data.get('scene_change'):
            scene_changes.append({
                'timestamp': timestamps[i],
                'time': current_time,
                'confidence': scene_data.get('confidence', 0.5)
            })
    
    # Basic metrics
    total_shots = len(scene_changes) + 1  # +1 for initial shot
    avg_shot_duration = mean(shot_durations) if shot_durations else video_duration
    min_shot_duration = min(shot_durations) if shot_durations else video_duration
    max_shot_duration = max(shot_durations) if shot_durations else video_duration
    
    # Cut frequency (cuts per minute)
    cut_frequency = (len(scene_changes) / video_duration) * 60 if video_duration > 0 else 0
    
    # Pacing classification
    if cut_frequency < 10:
        pacing_classification = "slow"
    elif cut_frequency < 20:
        pacing_classification = "moderate"
    elif cut_frequency < 40:
        pacing_classification = "fast"
    else:
        pacing_classification = "very_fast"
    
    # Shot distribution
    short_shots = sum(1 for d in shot_durations if d < 2)
    medium_shots = sum(1 for d in shot_durations if 2 <= d < 5)
    long_shots = sum(1 for d in shot_durations if d >= 5)
    
    # Rhythm consistency and variance (std deviation of shot durations)
    if len(shot_durations) > 1:
        rhythm_variability = stdev(shot_durations)
        shot_duration_variance = statistics.variance(shot_durations)
        rhythm_consistency = 1 / (1 + rhythm_variability)  # Higher = more consistent
        # Classify rhythm consistency
        if rhythm_consistency > 0.7:
            rhythm_consistency_class = "consistent"
        elif rhythm_consistency > 0.4:
            rhythm_consistency_class = "varied"
        else:
            rhythm_consistency_class = "erratic"
    else:
        rhythm_variability = 0
        shot_duration_variance = 0
        rhythm_consistency = 1
        rhythm_consistency_class = "consistent"
    
    # Acceleration patterns
    acceleration_phases = []
    if len(shot_durations) >= 3:
        for i in range(len(shot_durations) - 2):
            # Check if shots are getting progressively shorter
            if shot_durations[i] > shot_durations[i+1] > shot_durations[i+2]:
                acceleration_phases.append({
                    'start_shot': i,
                    'acceleration_rate': (shot_durations[i] - shot_durations[i+2]) / 2
                })
    
    # Scene complexity changes
    complexity_changes = 0
    if object_timeline:
        for i in range(1, len(timestamps)):
            prev_objects = len(object_timeline.get(timestamps[i-1], {}).get('objects', {}))
            curr_objects = len(object_timeline.get(timestamps[i], {}).get('objects', {}))
            if abs(curr_objects - prev_objects) > 2:
                complexity_changes += 1
    
    # Montage detection (rapid cuts in sequence)
    montage_segments = []
    rapid_cut_threshold = 1.5  # seconds
    consecutive_rapid = 0
    montage_start = None
    
    for i, duration in enumerate(shot_durations):
        if duration < rapid_cut_threshold:
            if consecutive_rapid == 0:
                montage_start = i
            consecutive_rapid += 1
        else:
            if consecutive_rapid >= 3:  # At least 3 rapid cuts
                montage_segments.append({
                    'start': montage_start,
                    'end': i - 1,
                    'shots': consecutive_rapid
                })
            consecutive_rapid = 0
    
    # Check last segment
    if consecutive_rapid >= 3:
        montage_segments.append({
            'start': montage_start,
            'end': len(shot_durations) - 1,
            'shots': consecutive_rapid
        })
    
    # Intro/outro pacing
    intro_cuts = sum(1 for sc in scene_changes if sc['time'] < 3)
    outro_cuts = sum(1 for sc in scene_changes if sc['time'] > video_duration - 3)
    
    # Peak pacing moments (where cuts accelerate)
    peak_moments = []
    window_size = 3
    for i in range(len(shot_durations) - window_size):
        window_avg = mean(shot_durations[i:i+window_size])
        if window_avg < avg_shot_duration * 0.5:  # 50% faster than average
            peak_moments.append({
                'shot_index': i,
                'avg_duration': window_avg
            })
    
    # Energy curve (inverse of shot duration over time)
    energy_curve = []
    cumulative_time = 0
    for i, duration in enumerate(shot_durations):
        energy = 1 / duration if duration > 0 else 1
        energy_curve.append({
            'time': cumulative_time,
            'energy': round(energy, 2)
        })
        cumulative_time += duration
    
    # Camera movement correlation and shot type changes
    camera_movement_cuts = 0
    shot_type_changes = 0
    last_distance = None
    if camera_distance_timeline:
        for sc in scene_changes:
            # Check if camera distance changed around cut
            time = sc['time']
            for timestamp, data in camera_distance_timeline.items():
                if abs(parse_timestamp_to_seconds(timestamp) - time) < 0.5:
                    camera_movement_cuts += 1
                    break
        # Count shot type changes
        for timestamp in sorted(camera_distance_timeline.keys()):
            distance = camera_distance_timeline[timestamp].get('distance')
            if last_distance and distance != last_distance:
                shot_type_changes += 1
            last_distance = distance
    
    # Pacing curve - shot density per 10-second window
    pacing_curve = {}
    window_size = 10
    for i in range(0, int(video_duration), window_size):
        window_start = i
        window_end = min(i + window_size, int(video_duration))
        window_key = f"{window_start}-{window_end}s"
        
        # Count cuts in this window
        cuts_in_window = sum(1 for sc in scene_changes 
                           if window_start <= sc['time'] < window_end)
        pacing_curve[window_key] = cuts_in_window
    
    # Acceleration score - compare first half to second half
    mid_point = video_duration / 2
    first_half_cuts = sum(1 for sc in scene_changes if sc['time'] < mid_point)
    second_half_cuts = sum(1 for sc in scene_changes if sc['time'] >= mid_point)
    
    if first_half_cuts > 0:
        acceleration_score = (second_half_cuts - first_half_cuts) / first_half_cuts
    else:
        acceleration_score = 0 if second_half_cuts == 0 else 1
    
    # Cut density zones - find windows with high cut frequency
    if pacing_curve:
        max_cuts = max(pacing_curve.values())
        threshold = max_cuts * 0.8  # 80% of peak
        cut_density_zones = [window for window, cuts in pacing_curve.items() 
                            if cuts >= threshold and max_cuts > 0]
    else:
        cut_density_zones = []
    
    # Visual load per scene
    visual_load_per_scene = 0
    if object_timeline and total_shots > 0:
        total_objects = 0
        for timestamp, data in object_timeline.items():
            objects = data.get('objects', {})
            total_objects += sum(objects.values())
        visual_load_per_scene = round(total_objects / total_shots, 2)
    
    # Pattern tags (pacing_tags)
    pattern_tags = []
    
    # Quick cuts and pacing tags
    if pacing_classification in ["fast", "very_fast"] or avg_shot_duration < 4:
        pattern_tags.append("quick_cuts")
    
    # Acceleration/deceleration
    if acceleration_score > 0.3:
        pattern_tags.append("accelerating_pace")
    elif acceleration_score < -0.3:
        pattern_tags.append("decelerating_pace")
    
    # Rhythm
    if rhythm_consistency_class == "consistent":
        pattern_tags.append("rhythmic_editing")
    
    # Montage sections
    if len(cut_density_zones) > 0:
        pattern_tags.append("has_montage_sections")
    
    # MTV style
    if avg_shot_duration < 1.5:
        pattern_tags.append("mtv_style")
    
    # Experimental pacing
    if shot_duration_variance > avg_shot_duration:
        pattern_tags.append("experimental_pacing")
    
    # Build structured ML features for missing components
    
    # Acceleration trend classification
    if acceleration_score > 0.3:
        acceleration_trend = "accelerating"
    elif acceleration_score < -0.3:
        acceleration_trend = "decelerating"
    else:
        acceleration_trend = "stable"
    
    # Tempo balance (cuts in first vs second half)
    mid_point = video_duration / 2
    # Convert timestamps to seconds for comparison
    scene_times = [parse_timestamp_to_seconds(ts) for ts in timestamps if parse_timestamp_to_seconds(ts) is not None]
    early_cuts = sum(1 for ts in scene_times if ts < mid_point)
    late_cuts = total_shots - early_cuts
    balance_index = early_cuts / total_shots if total_shots > 0 else 0.5
    
    tempo_balance = {
        'earlyCuts': early_cuts,
        'lateCuts': late_cuts,
        'balanceIndex': round(balance_index, 2)
    }
    
    # Object density pacing
    avg_objects_per_shot = visual_load_per_scene  # Already computed
    if avg_objects_per_shot < 2.0:
        complexity_level = "low"
    elif avg_objects_per_shot < 5.0:
        complexity_level = "moderate"
    else:
        complexity_level = "high"
    
    object_density_pacing = {
        'avgObjectsPerShot': round(avg_objects_per_shot, 2),
        'complexityLevel': complexity_level
    }
    
    # Shot camera alignment
    camera_changes_per_shot = shot_type_changes / total_shots if total_shots > 0 else 0
    # Alignment score: higher when camera changes align with cuts
    alignment_score = min(1.0, camera_changes_per_shot) if camera_distance_timeline else 0.0
    
    shot_camera_alignment = {
        'cameraChangesPerShot': round(camera_changes_per_shot, 2),
        'alignmentScore': round(alignment_score, 2)
    }
    
    # Peak cut moments - extract from existing data
    peak_cut_moments = []
    for zone in cut_density_zones[:3]:  # Top 3 zones
        window_start = int(zone.split('-')[0])
        window_cuts = sum(1 for ts in scene_times 
                         if window_start <= ts < window_start + 10)
        
        peak_cut_moments.append({
            'window': zone,
            'cutCount': window_cuts,
            'intensity': round(window_cuts / 10 * 2, 2)  # Normalize to intensity
        })
    
    # Longest shot moment
    longest_idx = shot_durations.index(max_shot_duration) if shot_durations else 0
    
    # Use FPS-aware timestamp generation if available
    if fps_manager and longest_idx < len(timestamps):
        timestamp_key = timestamps[longest_idx]
        scene_data = scene_timeline.get(timestamp_key, {})
        longest_timestamp = fps_manager.get_display_timestamp(scene_data, max_shot_duration, context='scene_detection')
    elif longest_idx < len(scene_times):
        # Legacy behavior - but with sanity check
        if scene_times[longest_idx] < video_duration:
            # Looks like seconds
            longest_timestamp = f"{int(scene_times[longest_idx])}-{int(scene_times[longest_idx] + max_shot_duration)}s"
        else:
            # Looks like frames - estimate position
            position_pct = longest_idx / len(scene_times) if len(scene_times) > 0 else 0.5
            estimated_start = int(video_duration * position_pct)
            estimated_end = min(int(estimated_start + 2), int(video_duration))  # Assume 2 second shot
            longest_timestamp = f"{estimated_start}-{estimated_end}s"
    else:
        longest_timestamp = "0-0s"
    
    if scene_times and longest_idx < len(scene_times):
        position_ratio = scene_times[longest_idx] / video_duration
        if position_ratio < 0.2:
            longest_position = "intro"
        elif position_ratio > 0.8:
            longest_position = "outro"
        else:
            longest_position = "middle"
    else:
        longest_position = "unknown"
    
    longest_shot_moment = {
        'timestamp': longest_timestamp,
        'duration': round(max_shot_duration, 2),
        'position': longest_position
    }
    
    # Shortest shot moment
    shortest_idx = shot_durations.index(min_shot_duration) if shot_durations else 0
    
    # Use FPS-aware timestamp generation if available
    if fps_manager and shortest_idx < len(timestamps):
        timestamp_key = timestamps[shortest_idx]
        scene_data = scene_timeline.get(timestamp_key, {})
        shortest_timestamp = fps_manager.get_display_timestamp(scene_data, min_shot_duration, context='scene_detection')
    elif shortest_idx < len(scene_times):
        # Legacy behavior - but with sanity check
        if scene_times[shortest_idx] < video_duration:
            # Looks like seconds
            shortest_timestamp = f"{int(scene_times[shortest_idx])}-{int(scene_times[shortest_idx] + min_shot_duration)}s"
        else:
            # Looks like frames - estimate position
            position_pct = shortest_idx / len(scene_times) if len(scene_times) > 0 else 0.5
            estimated_start = int(video_duration * position_pct)
            estimated_end = min(int(estimated_start + 1), int(video_duration))  # Assume 1 second shot
            shortest_timestamp = f"{estimated_start}-{estimated_end}s"
    else:
        shortest_timestamp = "0-0s"
    
    if scene_times and shortest_idx < len(scene_times):
        position_ratio = scene_times[shortest_idx] / video_duration
        if position_ratio < 0.2:
            shortest_position = "intro"
        elif position_ratio > 0.8:
            shortest_position = "outro"
        else:
            shortest_position = "middle"
    else:
        shortest_position = "unknown"
    
    shortest_shot_moment = {
        'timestamp': shortest_timestamp,
        'duration': round(min_shot_duration, 2),
        'position': shortest_position
    }
    
    # Montage segments - build from cut density zones
    montage_segments_detailed = []
    for zone in cut_density_zones:
        try:
            start = int(zone.split('-')[0])
            end = int(zone.split('-')[1].replace('s', ''))
            
            # Calculate average shot duration in this segment
            segment_shots = [d for i, d in enumerate(shot_durations) 
                           if i < len(scene_times) and start <= scene_times[i] < end]
            avg_duration = mean(segment_shots) if segment_shots else 1.0
            
            montage_segments_detailed.append({
                'start': f"{start}s",
                'end': f"{end}s",
                'avgShotDuration': round(avg_duration, 2)
            })
        except:
            continue
    
    # Editing style classification
    if "mtv_style" in pattern_tags:
        editing_style = "mtv_style"
    elif rhythm_consistency_class == "consistent" and avg_shot_duration > 3:
        editing_style = "classic_narrative"
    elif rhythm_consistency_class == "consistent":
        editing_style = "rhythmic"
    else:
        editing_style = "experimental"
    
    # Classified patterns
    classified_patterns = {
        'hasRapidPacing': pacing_classification in ["rapid", "fast"],
        'isRhythmic': rhythm_consistency_class == "consistent",
        'isExperimental': shot_duration_variance > avg_shot_duration,
        'hasMontageSections': len(cut_density_zones) > 0
    }
    
    # ML tags (enhanced from pattern_tags)
    ml_tags = pattern_tags.copy()
    if pacing_classification in ["rapid", "fast"]:
        ml_tags.append("fast_paced")
    if rhythm_consistency_class == "varied":
        ml_tags.append("variable_rhythm")
    if acceleration_trend == "accelerating":
        ml_tags.append("building_momentum")
    
    # Quality metrics
    data_completeness = len(scene_times) / max(video_duration / 2, 1)  # Expect ~1 cut per 2s
    data_completeness = min(1.0, data_completeness)
    
    scene_detection_rate = 1.0 if scene_timeline else 0.0  # Simple binary for now
    camera_data_available = bool(camera_distance_timeline)
    object_data_available = bool(object_timeline)
    cut_detection_confidence = 0.9 if total_shots > 0 else 0.0
    
    overall_quality_confidence = mean([
        data_completeness,
        scene_detection_rate,
        cut_detection_confidence
    ])
    
    return {
        # Core metrics
        'total_shots': total_shots,
        'avg_shot_duration': round(avg_shot_duration, 2),
        'shots_per_minute': round(cut_frequency, 2),
        'shortest_shot': round(min_shot_duration, 2),
        'longest_shot': round(max_shot_duration, 2),
        'shot_duration_variance': round(shot_duration_variance, 2),
        'visual_load_per_scene': visual_load_per_scene,
        'shot_type_changes': shot_type_changes,
        
        # Dynamics
        'pacing_classification': pacing_classification,
        'rhythm_consistency': rhythm_consistency_class,
        'acceleration_score': round(acceleration_score, 2),
        'acceleration_trend': acceleration_trend,  # NEW
        'intro_pacing': intro_cuts,
        'outro_pacing': outro_cuts,
        'tempo_balance': tempo_balance,  # NEW
        
        # Interactions
        'pacing_curve': pacing_curve,
        'cut_density_zones': cut_density_zones,
        'object_density_pacing': object_density_pacing,  # NEW
        'shot_camera_alignment': shot_camera_alignment,  # NEW
        
        # Key Events (NEW)
        'peak_cut_moments': peak_cut_moments,
        'longest_shot_moment': longest_shot_moment,
        'shortest_shot_moment': shortest_shot_moment,
        'montage_segments_detailed': montage_segments_detailed,
        
        # Patterns
        'pacing_tags': pattern_tags,
        'editing_style': editing_style,  # NEW
        'classified_patterns': classified_patterns,  # NEW
        'ml_tags': ml_tags,  # NEW
        
        # Quality (NEW)
        'data_completeness': round(data_completeness, 2),
        'scene_detection_rate': round(scene_detection_rate, 2),
        'camera_data_available': camera_data_available,
        'object_data_available': object_data_available,
        'cut_detection_confidence': round(cut_detection_confidence, 2),
        'overall_quality_confidence': round(overall_quality_confidence, 2),
        
        # Additional metrics for backward compatibility
        'cut_frequency': round(cut_frequency, 2),
        'min_shot_duration': round(min_shot_duration, 2),
        'max_shot_duration': round(max_shot_duration, 2),
        'rhythm_consistency_score': round(rhythm_consistency, 2),
        'shot_distribution': {
            'short': short_shots,
            'medium': medium_shots,
            'long': long_shots
        },
        'rhythm_variability': round(rhythm_variability, 2),
        'acceleration_phases': len(acceleration_phases),
        'complexity_changes': complexity_changes,
        'montage_segments': len(montage_segments),
        'peak_pacing_moments': len(peak_moments),
        'energy_curve': energy_curve[:10],
        'camera_movement_cuts': camera_movement_cuts,
        'scene_pacing_pattern_tags': pattern_tags
    }
def compute_speech_analysis_metrics(speech_timeline, transcript, speech_segments, 
                                   expression_timeline, gesture_timeline, 
                                   human_analysis_data, video_duration,
                                   energy_level_windows=None, energy_variance=0, 
                                   climax_timestamp=0, burst_pattern='none'):
    """Compute comprehensive speech analysis metrics for ML-ready analysis
    
    Args:
        speech_timeline: Timeline of detected words with timestamps
        transcript: Full transcript text
        speech_segments: List of speech segments with timing
        expression_timeline: Timeline of facial expressions
        gesture_timeline: Timeline of gestures
        human_analysis_data: Enhanced human analysis data
        video_duration: Video duration in seconds
        
    Returns:
        dict: Comprehensive speech analysis metrics
    """
    import re
    from collections import defaultdict, Counter
    
    # Initialize metrics
    metrics = {}
    
    # 1. Basic Speech Metrics
    words = transcript.split() if transcript else []
    word_count = len(words)
    unique_words = len(set(word.lower() for word in words))
    vocabulary_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Calculate speech coverage from segments
    speech_time = 0
    if speech_segments:
        for segment in speech_segments:
            speech_time += segment.get('end', 0) - segment.get('start', 0)
    
    speech_coverage = speech_time / video_duration if video_duration > 0 else 0
    speech_density = word_count / video_duration if video_duration > 0 else 0
    speech_rate_wpm = (word_count / speech_time * 60) if speech_time > 0 else 0
    
    # Find first and last word timestamps
    first_word_timestamp = float('inf')
    last_word_timestamp = 0
    
    for timestamp, data in speech_timeline.items():
        try:
            time = parse_timestamp_to_seconds(timestamp)
            if time is not None:
                first_word_timestamp = min(first_word_timestamp, time)
                last_word_timestamp = max(last_word_timestamp, time)
        except:
            pass
    
    if first_word_timestamp == float('inf'):
        first_word_timestamp = 0
    
    metrics['total_words'] = word_count
    metrics['speech_density'] = round(speech_density, 2)
    metrics['speech_coverage'] = round(speech_coverage, 2)
    metrics['words_per_minute'] = round(speech_rate_wpm, 1)
    metrics['unique_words'] = unique_words
    metrics['vocabulary_diversity'] = round(vocabulary_diversity, 2)
    metrics['silence_ratio'] = round(1 - speech_coverage, 2)
    
    # Calculate segment-based metrics
    if speech_segments:
        segment_durations = [seg.get('end', 0) - seg.get('start', 0) for seg in speech_segments]
        segment_word_counts = [len(seg.get('text', '').split()) for seg in speech_segments]
        
        metrics['avg_segment_duration'] = round(sum(segment_durations) / len(segment_durations), 2) if segment_durations else 0
        metrics['longest_segment'] = round(max(segment_durations), 2) if segment_durations else 0
        metrics['shortest_segment'] = round(min(segment_durations), 2) if segment_durations else 0
        metrics['avg_words_per_segment'] = round(sum(segment_word_counts) / len(segment_word_counts), 1) if segment_word_counts else 0
    else:
        metrics['avg_segment_duration'] = 0
        metrics['longest_segment'] = 0
        metrics['shortest_segment'] = 0
        metrics['avg_words_per_segment'] = 0
    
    # 2. Speech Rhythm & Pacing
    wpm_by_segment = {}
    window_size = 1  # 1-second windows for better granularity
    
    for i in range(0, int(video_duration), window_size):
        window_start = i
        window_end = min(i + window_size, int(video_duration))
        window_key = f"{window_start}-{window_end}s"
        
        # Count words in this window
        window_words = 0
        window_time = 0
        
        for segment in speech_segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check overlap with window
            if seg_start < window_end and seg_end > window_start:
                overlap_start = max(seg_start, window_start)
                overlap_end = min(seg_end, window_end)
                overlap_duration = overlap_end - overlap_start
                
                # Estimate words in overlap
                segment_words = len(segment.get('text', '').split())
                segment_duration = seg_end - seg_start
                if segment_duration > 0:
                    overlap_words = int(segment_words * (overlap_duration / segment_duration))
                    window_words += overlap_words
                    window_time += overlap_duration
        
        if window_time > 0:
            wpm = (window_words / window_time) * 60
            wpm_by_segment[window_key] = round(wpm, 1)
        else:
            wpm_by_segment[window_key] = 0
    
    # Calculate acceleration score
    first_half_words = 0
    second_half_words = 0
    mid_point = video_duration / 2
    
    for segment in speech_segments:
        seg_words = len(segment.get('text', '').split())
        if segment.get('start', 0) < mid_point:
            first_half_words += seg_words
        else:
            second_half_words += seg_words
    
    if first_half_words > 0:
        speech_acceleration_score = (second_half_words - first_half_words) / first_half_words
    else:
        speech_acceleration_score = 0 if second_half_words == 0 else 1
    
    # Determine rhythm type
    wpm_values = list(wpm_by_segment.values())
    if wpm_values:
        wpm_variance = statistics.variance(wpm_values) if len(wpm_values) > 1 else 0
        avg_wpm = mean(wpm_values)
        
        if wpm_variance < avg_wpm * 0.2:
            speech_rhythm_type = "flowing"
        elif speech_acceleration_score > 0.3:
            speech_rhythm_type = "building"
        elif max(wpm_values) > avg_wpm * 1.5:
            speech_rhythm_type = "staccato"
        else:
            speech_rhythm_type = "erratic"
        
        rhythm_consistency_score = 1 / (1 + wpm_variance / avg_wpm) if avg_wpm > 0 else 0
    else:
        speech_rhythm_type = "none"
        rhythm_consistency_score = 0
        wpm_variance = 0
    
    # Front-load ratio
    front_words = 0
    front_limit = video_duration * 0.2  # First 20%
    
    for segment in speech_segments:
        if segment.get('start', 0) < front_limit:
            front_words += len(segment.get('text', '').split())
    
    speech_front_load_ratio = front_words / word_count if word_count > 0 else 0
    
    metrics['wpm_by_segment'] = wpm_by_segment
    metrics['speech_acceleration_score'] = round(speech_acceleration_score, 2)
    metrics['speech_rhythm_type'] = speech_rhythm_type
    metrics['rhythm_consistency_score'] = round(rhythm_consistency_score, 2)
    metrics['speech_front_load_ratio'] = round(speech_front_load_ratio, 2)
    
    # 3. Pause & Gap Analysis
    gaps = []
    if speech_segments:
        sorted_segments = sorted(speech_segments, key=lambda x: x.get('start', 0))
        
        for i in range(1, len(sorted_segments)):
            gap_start = sorted_segments[i-1].get('end', 0)
            gap_end = sorted_segments[i].get('start', 0)
            gap_duration = gap_end - gap_start
            
            if gap_duration > 1:  # Pauses > 1 second
                gap_type = "dramatic" if gap_duration > 2 else "strategic"
                if gap_duration > 3:
                    gap_type = "awkward"
                
                gaps.append({
                    'start': round(gap_start, 2),
                    'duration': round(gap_duration, 2),
                    'type': gap_type
                })
    
    total_pause_time = sum(gap['duration'] for gap in gaps)
    pause_count = len(gaps)
    longest_pause_duration = max((gap['duration'] for gap in gaps), default=0)
    strategic_pauses = sum(1 for gap in gaps if gap['type'] == 'strategic')
    awkward_pauses = sum(1 for gap in gaps if gap['type'] == 'awkward')
    
    metrics['pause_analysis'] = {
        'gaps': gaps,
        'total_pause_time': round(total_pause_time, 2),
        'pause_count': pause_count,
        'longest_pause_duration': round(longest_pause_duration, 2),
        'strategic_pauses': strategic_pauses,
        'awkward_pauses': awkward_pauses
    }
    
    # 4. Hook Detection
    hook_phrases = []
    hook_patterns = [
        (r'\b(did you know|guess what|wait for it|watch this|check this out)\b', 'question'),
        (r'\b(amazing|incredible|mind-blowing|shocking|unbelievable)\b', 'hyperbole'),
        (r'\b(you won\'t believe|you need to see|this will change)\b', 'promise'),
        (r'^(hey|hi|hello|what\'s up)', 'greeting'),
        (r'\b(secret|trick|hack|tip)\b', 'value_prop')
    ]
    
    # Analyze transcript for hooks
    lower_transcript = transcript.lower() if transcript else ""
    
    for pattern, hook_type in hook_patterns:
        matches = re.finditer(pattern, lower_transcript, re.IGNORECASE)
        for match in matches:
            # Find approximate timestamp
            word_position = len(lower_transcript[:match.start()].split())
            estimated_time = (word_position / word_count * speech_time) if word_count > 0 else 0
            
            hook_phrases.append({
                'text': match.group(),
                'timestamp': round(estimated_time, 2),
                'type': hook_type,
                'confidence': 0.8
            })
    
    # Calculate hook density per 10s
    hook_density_per_10s = defaultdict(int)
    for hook in hook_phrases:
        window = int(hook['timestamp'] / 10) * 10
        window_key = f"{window}-{window+10}s"
        hook_density_per_10s[window_key] += 1
    
    # Opening hook strength (based on first 3 seconds)
    opening_hooks = sum(1 for hook in hook_phrases if hook['timestamp'] < 3)
    opening_hook_strength = min(opening_hooks / 2, 1)  # Normalize to 0-1
    
    metrics['hook_density_per_10s'] = dict(hook_density_per_10s)
    metrics['hook_phrases'] = hook_phrases[:10]  # Top 10 hooks
    metrics['opening_hook_strength'] = round(opening_hook_strength, 2)
    
    # 5. CTA Analysis
    cta_phrases = []
    cta_patterns = [
        (r'\b(follow|like|subscribe|comment|share|hit the)\b', 'engagement'),
        (r'\b(link in bio|check out|visit|go to)\b', 'traffic'),
        (r'\b(buy|purchase|get yours|order now)\b', 'conversion'),
        (r'\b(save this|bookmark|remember)\b', 'save'),
        (r'\b(tag someone|send this|share with)\b', 'viral')
    ]
    
    for pattern, cta_category in cta_patterns:
        matches = re.finditer(pattern, lower_transcript, re.IGNORECASE)
        for match in matches:
            word_position = len(lower_transcript[:match.start()].split())
            estimated_time = (word_position / word_count * speech_time) if word_count > 0 else 0
            
            # Determine urgency based on context
            urgency = "low"
            if any(word in match.group().lower() for word in ['now', 'today', 'quick']):
                urgency = "high"
            elif estimated_time > speech_time * 0.7:  # In last 30%
                urgency = "medium"
            
            cta_phrases.append({
                'text': match.group(),
                'timestamp': round(estimated_time, 2),
                'urgency': urgency,
                'category': cta_category
            })
    
    # CTA density and clustering
    cta_density_per_10s = defaultdict(int)
    for cta in cta_phrases:
        window = int(cta['timestamp'] / 10) * 10
        window_key = f"{window}-{window+10}s"
        cta_density_per_10s[window_key] += 1
    
    # Find CTA clusters
    cta_clustering = []
    if cta_phrases:
        sorted_ctas = sorted(cta_phrases, key=lambda x: x['timestamp'])
        cluster_start = sorted_ctas[0]['timestamp']
        cluster_count = 1
        
        for i in range(1, len(sorted_ctas)):
            if sorted_ctas[i]['timestamp'] - sorted_ctas[i-1]['timestamp'] < 5:
                cluster_count += 1
            else:
                if cluster_count >= 2:
                    cta_clustering.append({
                        'start': round(cluster_start, 2),
                        'end': round(sorted_ctas[i-1]['timestamp'], 2),
                        'count': cluster_count
                    })
                cluster_start = sorted_ctas[i]['timestamp']
                cluster_count = 1
        
        # Check last cluster
        if cluster_count >= 2:
            cta_clustering.append({
                'start': round(cluster_start, 2),
                'end': round(sorted_ctas[-1]['timestamp'], 2),
                'count': cluster_count
            })
    
    metrics['cta_density_per_10s'] = dict(cta_density_per_10s)
    metrics['cta_phrases'] = cta_phrases[:10]  # Top 10 CTAs
    metrics['cta_clustering'] = cta_clustering
    
    # 6. Speech Quality
    clarity_score_by_window = {}
    overall_confidences = []
    
    # Calculate clarity from speech segments
    for i in range(0, int(video_duration), 5):
        window_key = f"{i}-{i+5}s"
        window_confidences = []
        
        for segment in speech_segments:
            if i <= segment.get('start', 0) < i + 5:
                confidence = segment.get('confidence', 0.5)
                window_confidences.append(confidence)
                overall_confidences.append(confidence)
        
        if window_confidences:
            clarity_score_by_window[window_key] = round(mean(window_confidences), 2)
        else:
            clarity_score_by_window[window_key] = 0
    
    overall_clarity_score = mean(overall_confidences) if overall_confidences else 0
    
    # Filler word detection
    filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'literally', 'actually']
    filler_count = 0
    for filler in filler_words:
        filler_count += len(re.findall(r'\b' + filler + r'\b', lower_transcript, re.IGNORECASE))
    
    filler_word_ratio = filler_count / word_count if word_count > 0 else 0
    
    # Find mumbling segments (low confidence)
    mumbling_segments = []
    for segment in speech_segments:
        if segment.get('confidence', 1) < 0.6:
            mumbling_segments.append({
                'start': round(segment.get('start', 0), 2),
                'end': round(segment.get('end', 0), 2),
                'confidence': round(segment.get('confidence', 0), 2)
            })
    
    metrics['clarity_score_by_window'] = clarity_score_by_window
    metrics['overall_clarity_score'] = round(overall_clarity_score, 2)
    metrics['filler_word_ratio'] = round(filler_word_ratio, 3)
    metrics['mumbling_segments'] = mumbling_segments
    metrics['background_noise_ratio'] = 0.1  # Placeholder - would need audio analysis
    
    # 7. Engagement Patterns
    direct_address_count = len(re.findall(r'\b(you|your|you\'re|you\'ll|you\'ve)\b', lower_transcript, re.IGNORECASE))
    inclusive_words = len(re.findall(r'\b(we|us|our|let\'s)\b', lower_transcript, re.IGNORECASE))
    inclusive_language_ratio = inclusive_words / word_count if word_count > 0 else 0
    
    # Find repetition patterns
    words = lower_transcript.split()
    phrase_counter = Counter()
    
    # Count 2-3 word phrases
    for i in range(len(words) - 2):
        phrase_2 = ' '.join(words[i:i+2])
        phrase_3 = ' '.join(words[i:i+3])
        
        # Skip common phrases
        if not any(common in phrase_2 for common in ['the', 'and', 'for', 'you', 'this']):
            phrase_counter[phrase_2] += 1
        if not any(common in phrase_3 for common in ['the', 'and', 'for']):
            phrase_counter[phrase_3] += 1
    
    # Find repeated phrases
    repetition_phrases = []
    for phrase, count in phrase_counter.most_common(5):
        if count >= 2:
            repetition_phrases.append({
                'text': phrase,
                'count': count,
                'timestamps': []  # Would need more detailed analysis
            })
    
    # Count questions
    question_count = len(re.findall(r'\?|[.!]\s*(?:did|what|why|how|when|where|who|which)', lower_transcript, re.IGNORECASE))
    
    metrics['direct_address_count'] = direct_address_count
    metrics['inclusive_language_ratio'] = round(inclusive_language_ratio, 2)
    metrics['repetition_patterns'] = {
        'phrases': repetition_phrases,
        'emphasis_words': []  # Would need emphasis detection
    }
    metrics['question_count'] = question_count
    
    # 8. Speech Bursts & Energy
    speech_bursts = []
    burst_threshold = speech_rate_wpm * 1.3  # 30% faster than average
    
    for segment in speech_segments:
        seg_duration = segment.get('end', 0) - segment.get('start', 0)
        seg_words = len(segment.get('text', '').split())
        
        if seg_duration > 0:
            seg_wpm = (seg_words / seg_duration) * 60
            
            if seg_wpm > burst_threshold and seg_duration > 2:
                burst_type = "rapid" if seg_wpm > burst_threshold * 1.5 else "energetic"
                speech_bursts.append({
                    'start': round(segment.get('start', 0), 2),
                    'end': round(segment.get('end', 0), 2),
                    'words': seg_words,
                    'wpm': round(seg_wpm, 1),
                    'type': burst_type
                })
    
    # Determine burst pattern
    if speech_bursts:
        avg_burst_time = mean([burst['start'] for burst in speech_bursts])
        if avg_burst_time < video_duration * 0.3:
            burst_pattern = "front_loaded"
        elif avg_burst_time > video_duration * 0.7:
            burst_pattern = "climax"
        else:
            burst_pattern = "distributed"
    else:
        burst_pattern = "none"
    
    # Integrate audio energy analysis if available
    if energy_level_windows:
        # Use real audio energy data from AudioEnergyService
        logger.info("Using real audio energy data for burst pattern analysis")
        
        # Override computed burst pattern with actual audio analysis
        if burst_pattern and burst_pattern != 'none':
            # Use the burst pattern from AudioEnergyService
            pass  # Keep the passed burst_pattern
        
        # Add energy metrics to output
        energy_metrics = {
            'energy_level_windows': energy_level_windows,
            'energy_variance': float(energy_variance),
            'climax_timestamp': float(climax_timestamp),
            'burst_pattern': burst_pattern,
            'has_audio_energy': True
        }
    else:
        # Fallback to computed burst pattern (original logic)
        logger.info("No audio energy data available - using computed burst pattern")
        energy_metrics = {
            'energy_level_windows': {},
            'energy_variance': 0.0,
            'climax_timestamp': video_duration / 2,  # Default to middle
            'burst_pattern': burst_pattern,
            'has_audio_energy': False
        }
    
    metrics['speech_bursts'] = speech_bursts
    metrics['burst_pattern'] = burst_pattern
    
    # 9. Speech-to-Visual Sync Features
    gesture_sync_ratio = 0
    face_on_screen_during_speech = 0
    
    # Calculate gesture sync
    if gesture_timeline and speech_segments:
        speech_with_gesture = 0
        
        for segment in speech_segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if any gestures occur during this speech segment
            for timestamp, gesture_data in gesture_timeline.items():
                gesture_time = parse_timestamp_to_seconds(timestamp)
                if gesture_time and seg_start <= gesture_time <= seg_end:
                    speech_with_gesture += 1
                    break
        
        gesture_sync_ratio = speech_with_gesture / len(speech_segments) if speech_segments else 0
    
    # Calculate face visibility during speech
    if expression_timeline and speech_time > 0:
        face_time_during_speech = 0
        
        for segment in speech_segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check expressions during this segment
            for timestamp, expr_data in expression_timeline.items():
                expr_time = parse_timestamp_to_seconds(timestamp)
                if expr_time and seg_start <= expr_time <= seg_end:
                    face_time_during_speech += 1
        
        face_on_screen_during_speech = face_time_during_speech / speech_time
    
    # Find gesture emphasis moments
    gesture_emphasis_moments = []
    
    # This would require more detailed word-level timing
    # For now, we'll identify potential moments
    
    # Visual absence analysis
    off_camera_speech_segments = []
    for segment in speech_segments:
        # Check if face is visible during this segment
        seg_has_face = False
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)
        
        for timestamp in expression_timeline:
            expr_time = parse_timestamp_to_seconds(timestamp)
            if expr_time and seg_start <= expr_time <= seg_end:
                seg_has_face = True
                break
        
        if not seg_has_face:
            off_camera_speech_segments.append({
                'start': round(seg_start, 2),
                'end': round(seg_end, 2),
                'speech_content': segment.get('text', '')[:50] + '...'
            })
    
    speech_only_ratio = len(off_camera_speech_segments) / len(speech_segments) if speech_segments else 0
    
    # Count expression variety during speech
    expressions_during_speech = set()
    for segment in speech_segments:
        for timestamp, expr_data in expression_timeline.items():
            expr_time = parse_timestamp_to_seconds(timestamp)
            if expr_time and segment.get('start', 0) <= expr_time <= segment.get('end', 0):
                expressions_during_speech.add(expr_data.get('expression', 'unknown'))
    
    expression_variety_during_speech = len(expressions_during_speech) / 10  # Normalize to 0-1
    
    metrics['speech_visual_alignment'] = {
        'gesture_emphasis_moments': gesture_emphasis_moments,
        'expression_peaks_during_speech': [],  # Would need peak detection
        'lip_sync_quality': 0.8,  # Placeholder
        'body_language_congruence': round(gesture_sync_ratio, 2)
    }
    
    metrics['gesture_sync_ratio'] = round(gesture_sync_ratio, 2)
    metrics['face_on_screen_during_speech'] = round(face_on_screen_during_speech, 2)
    metrics['off_camera_speech_segments'] = off_camera_speech_segments
    metrics['speech_only_ratio'] = round(speech_only_ratio, 2)
    metrics['visual_punctuation_count'] = len(gesture_emphasis_moments)
    metrics['expression_variety_during_speech'] = round(expression_variety_during_speech, 2)
    
    # 10. Pattern Tags
    speech_pattern_tags = []
    
    if opening_hook_strength > 0.5:
        speech_pattern_tags.append("strong_opening")
    if speech_rate_wpm > 150:
        speech_pattern_tags.append("rapid_delivery")
    if len(gaps) > 3 and any(gap['type'] == 'dramatic' for gap in gaps):
        speech_pattern_tags.append("has_dramatic_pauses")
    if repetition_phrases:
        speech_pattern_tags.append("repetitive_emphasis")
    if direct_address_count > word_count * 0.05:
        speech_pattern_tags.append("direct_address_heavy")
    if cta_clustering:
        speech_pattern_tags.append("cta_clustered")
    if burst_pattern in ["climax", "building"]:
        speech_pattern_tags.append("energy_building")
    if overall_clarity_score > 0.85:
        speech_pattern_tags.append("clear_articulation")
    if gesture_sync_ratio > 0.6:
        speech_pattern_tags.append("high_gesture_sync")
    if face_on_screen_during_speech > 0.8:
        speech_pattern_tags.append("face_focused_delivery")
    
    metrics['speech_pattern_tags'] = speech_pattern_tags
    
    # 11. Conversation Analysis
    speech_type = "monologue"  # Default, would need speaker diarization
    speaker_changes = 0
    dominant_speaker_ratio = 1.0
    
    metrics['speech_type'] = speech_type
    metrics['speaker_changes'] = speaker_changes
    metrics['dominant_speaker_ratio'] = dominant_speaker_ratio
    
    # 12. Summary Scores
    hook_effectiveness_score = min((len(hook_phrases) * 0.2 + opening_hook_strength), 1)
    cta_effectiveness_score = min((len(cta_phrases) * 0.1 + len(cta_clustering) * 0.3), 1)
    delivery_confidence_score = overall_clarity_score
    
    # Calculate authenticity based on patterns
    authenticity_factors = [
        filler_word_ratio < 0.05,  # Not too polished
        filler_word_ratio > 0.01,  # Not too scripted
        len(repetition_phrases) < 3,  # Not too repetitive
        len(mumbling_segments) < 2  # Clear but natural
    ]
    authenticity_score = sum(authenticity_factors) / len(authenticity_factors)
    
    # Overall engagement score
    engagement_factors = [
        hook_effectiveness_score,
        direct_address_count > 0,
        question_count > 0,
        gesture_sync_ratio > 0.5,
        speech_rate_wpm > 120,
        burst_pattern != "none"
    ]
    verbal_engagement_score = sum(1 for f in engagement_factors if f) / len(engagement_factors)
    
    # Visual-verbal harmony
    visual_verbal_harmony_score = (gesture_sync_ratio + face_on_screen_during_speech) / 2
    
    metrics['hook_effectiveness_score'] = round(hook_effectiveness_score, 2)
    metrics['cta_effectiveness_score'] = round(cta_effectiveness_score, 2)
    metrics['delivery_confidence_score'] = round(delivery_confidence_score, 2)
    metrics['authenticity_score'] = round(authenticity_score, 2)
    metrics['verbal_engagement_score'] = round(verbal_engagement_score, 2)
    metrics['visual_verbal_harmony_score'] = round(visual_verbal_harmony_score, 2)
    
    # Add energy metrics to final output
    metrics.update(energy_metrics)
    
    return metrics
def compress_timeline_aggressively(timeline, max_entries=50, remove_unknown=True):
    """Aggressively compress timeline to reduce API payload size
    
    Args:
        timeline: Timeline dictionary to compress
        max_entries: Maximum number of entries to keep
        remove_unknown: Whether to skip entries with only 'unknown' objects
    
    Returns:
        Compressed timeline
    """
    if not timeline or not isinstance(timeline, dict):
        return {}
    
    # If timeline is small enough, return as is
    if len(timeline) <= max_entries:
        return timeline
    
    entries = list(timeline.items())
    total_entries = len(entries)
    
    # First, identify and preserve important entries (non-unknown objects)
    important_entries = []
    unknown_entries = []
    
    for timestamp, data in entries:
        if isinstance(data, dict):
            objects = data.get('objects', {})
            if isinstance(objects, dict) and list(objects.keys()) == ['unknown']:
                unknown_entries.append((timestamp, data))
            else:
                important_entries.append((timestamp, data))
        else:
            important_entries.append((timestamp, data))
    
    # Start with all important entries
    compressed = dict(important_entries)
    
    # If we still have room, add some unknown entries
    remaining_slots = max_entries - len(compressed)
    if remaining_slots > 0 and unknown_entries and not remove_unknown:
        # Sample unknown entries evenly
        sample_rate = max(1, len(unknown_entries) // remaining_slots)
        for i in range(0, len(unknown_entries), sample_rate):
            if len(compressed) >= max_entries:
                break
            compressed[unknown_entries[i][0]] = unknown_entries[i][1]
    
    # If still too large, sample the important entries
    if len(compressed) > max_entries:
        compressed_list = list(compressed.items())
        sample_rate = max(1, len(compressed_list) // max_entries)
        compressed = {}
        
        # Always keep first and last
        for i in range(0, len(compressed_list), sample_rate):
            compressed[compressed_list[i][0]] = compressed_list[i][1]
        
        # Ensure first and last are included
        if compressed_list:
            compressed[compressed_list[0][0]] = compressed_list[0][1]
            compressed[compressed_list[-1][0]] = compressed_list[-1][1]
    
    return compressed

def clean_timeline_for_api(timeline, timeline_type):
    """Clean timeline data to reduce payload size while preserving essential information"""
    if not timeline or not isinstance(timeline, dict):
        return {}
    
    # First apply aggressive compression if timeline is too large
    # For longer videos (>60s), we may need more aggressive compression
    if len(timeline) > 100:
        timeline = compress_timeline_aggressively(timeline, max_entries=50)
    elif len(timeline) > 60:
        # For medium-length videos, use gentler compression
        timeline = compress_timeline_aggressively(timeline, max_entries=80)
    
    cleaned = {}
    
    for timestamp, data in timeline.items():
        if timeline_type == 'object':
            # For object timeline, handle both dict and list formats
            objects = data.get('objects', {})
            
            # Handle case where objects is a dict (e.g., {'person': 2, 'bottle': 1})
            if isinstance(objects, dict):
                cleaned[timestamp] = {
                    'objects': objects,
                    'total_objects': sum(objects.values()),
                    'class_names': list(objects.keys())
                }
            else:
                # Handle case where objects is a list
                cleaned_objects = []
                for obj in objects:
                    if isinstance(obj, dict):
                        # Keep only essential object fields
                        cleaned_obj = {
                            'class': obj.get('class', ''),
                            'track_id': obj.get('track_id', ''),
                            'bbox': obj.get('bbox', [])
                        }
                        # Include confidence only if it exists
                        if 'confidence' in obj:
                            cleaned_obj['confidence'] = round(obj.get('confidence', 0), 2)
                        cleaned_objects.append(cleaned_obj)
                
                cleaned[timestamp] = {
                    'objects': cleaned_objects,
                    'total_objects': data.get('total_objects', len(cleaned_objects)),
                    'class_names': list(set(obj.get('class', '') for obj in cleaned_objects if isinstance(obj, dict)))
                }
            
        elif timeline_type == 'expression':
            # For expression timeline, remove verbose descriptions
            cleaned[timestamp] = {
                'expression': data.get('expression', ''),
                'confidence': round(data.get('confidence', 0), 2) if 'confidence' in data else None
            }
            
        elif timeline_type == 'camera_distance':
            # Keep camera distance simple
            cleaned[timestamp] = {
                'distance': data.get('distance', ''),
                'shot_type': data.get('shot_type', '')
            }
            
        else:
            # For other timelines, keep as is but remove any 'description' fields
            cleaned_data = data.copy() if isinstance(data, dict) else data
            if isinstance(cleaned_data, dict) and 'description' in cleaned_data:
                del cleaned_data['description']
            cleaned[timestamp] = cleaned_data
    
    return cleaned

def validate_timeline_data(timeline, timeline_name):
    """Validate timeline data and return cleaned version"""
    if timeline is None:
        return {}
    
    if not isinstance(timeline, dict):
        print(f"⚠️ Warning: {timeline_name} is not a dict (got {type(timeline).__name__}), returning empty")
        return {}
    
    # Check for corrupted data
    if len(str(timeline)) > 5_000_000:  # > 5MB single timeline
        print(f"⚠️ Warning: {timeline_name} is extremely large ({len(str(timeline)):,} chars), may cause issues")
    
    return timeline

def detect_language_simple(text):
    """Simple language detection for ML features using character patterns"""
    if not text:
        return 'unknown'
    
    # Check for CJK characters
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'  # Chinese
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return 'ja'  # Japanese
    if re.search(r'[\uac00-\ud7af]', text):
        return 'ko'  # Korean
    
    # Check for Arabic
    if re.search(r'[\u0600-\u06ff]', text):
        return 'ar'
    
    # Check for Cyrillic
    if re.search(r'[\u0400-\u04ff]', text):
        return 'ru'
    
    # Spanish indicators
    if re.search(r'\b(el|la|los|las|un|una|que|es|en|de|por|para)\b', text.lower()):
        return 'es'
    
    # Default to English for Latin scripts
    return 'en'

def compute_ml_readability(text):
    """Simple readability score for ML features (0-100)"""
    if not text:
        return 100.0
    
    words = text.split()
    if not words:
        return 100.0
    
    # Average word length
    avg_word_length = sum(len(w) for w in words) / len(words)
    
    # Sentence count (approximate)
    sentences = max(len(re.split(r'[.!?]+', text.strip())), 1)
    words_per_sentence = len(words) / sentences
    
    # Simple formula: shorter words and sentences = higher readability
    # Approximates Flesch Reading Ease
    score = 100 - (avg_word_length * 10) - (words_per_sentence * 2)
    
    return max(0.0, min(100.0, score))

def classify_sentiment_simple(text):
    """Simple sentiment classification for ML training"""
    if not text:
        return 'neutral'
    
    text_lower = text.lower()
    
    # Positive indicators
    positive_words = ['love', 'amazing', 'great', 'awesome', 'excellent', 'best', 'perfect', 
                     'beautiful', 'fantastic', 'wonderful', 'happy', 'good', 'like', 'enjoy']
    positive_emojis = ['😊', '😍', '❤️', '🔥', '💯', '👍', '✨', '🎉', '💖', '😄']
    
    # Negative indicators
    negative_words = ['hate', 'terrible', 'awful', 'worst', 'bad', 'horrible', 'disgusting', 
                     'ugly', 'nasty', 'poor', 'wrong', 'fail', 'don\'t', 'not']
    negative_emojis = ['😞', '😢', '😡', '👎', '💔', '😠', '😤', '😭']
    
    # Count indicators
    pos_count = sum(1 for word in positive_words if word in text_lower)
    pos_count += sum(1 for emoji in positive_emojis if emoji in text)
    
    neg_count = sum(1 for word in negative_words if word in text_lower)
    neg_count += sum(1 for emoji in negative_emojis if emoji in text)
    
    # Classify
    if pos_count > neg_count * 1.5:
        return 'positive'
    elif neg_count > pos_count * 1.5:
        return 'negative'
    else:
        return 'neutral'

def extract_top_terms(text, n=5):
    """Extract top N meaningful terms for ML features"""
    if not text:
        return []
    
    # Extract words (3+ characters)
    words = re.findall(r'\b\w{3,}\b', text.lower())
    
    # Common stopwords to exclude
    stopwords = {
        'the', 'and', 'for', 'this', 'that', 'with', 'from', 'have', 'has',
        'was', 'were', 'been', 'being', 'are', 'our', 'your', 'their',
        'what', 'when', 'where', 'which', 'who', 'why', 'how', 'can', 'will'
    }
    
    # Filter meaningful words
    meaningful = [w for w in words if w not in stopwords and not w.isdigit()]
    
    # Count frequencies
    word_freq = {}
    for word in meaningful:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return top N terms
    sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, freq in sorted_terms[:n]]

def extract_entity_patterns(text):
    """Extract entity patterns for ML features"""
    entities = {
        'has_brand_mention': False,
        'has_price_mention': False,
        'has_cta': False,
        'has_url': False,
        'has_question': False,
        'has_user_mention': False
    }
    
    if not text:
        return entities
    
    text_lower = text.lower()
    
    # Brand mentions (@ mentions or hashtags with 'brand', 'official')
    if re.search(r'@\w+|#\w*(brand|official|store)\w*', text):
        entities['has_brand_mention'] = True
    
    # Price mentions
    if re.search(r'\$\d+|\d+%|price|cost|free|sale|discount|deal', text_lower):
        entities['has_price_mention'] = True
    
    # Call to action
    if re.search(r'\b(link|bio|click|buy|shop|order|get|visit|check|follow|comment|share|like)\b', text_lower):
        entities['has_cta'] = True
    
    # URL detection
    if re.search(r'https?://|www\.|\.com|\.net|\.org', text_lower):
        entities['has_url'] = True
    
    # Question detection
    if '?' in text or re.search(r'\b(what|when|where|which|who|why|how|can|will|would|should)\b', text_lower):
        entities['has_question'] = True
    
    # User mentions
    if re.search(r'@\w+', text):
        entities['has_user_mention'] = True
    
    return entities

def extract_real_ml_data(unified_data, prompt_name, video_id=None):
    """Extract only real ML detection data for a specific prompt"""

    context_data = {
        'video_id': video_id
    }
    timelines = unified_data.get('timelines', {})

    if prompt_name == 'visual_overlay_analysis':
        # Compute comprehensive visual overlay metrics
        visual_overlay_metrics = compute_visual_overlay_metrics(
            timelines.get('textOverlayTimeline', {}),
            timelines.get('stickerTimeline', {}),
            timelines.get('gestureTimeline', {}),
            timelines.get('speechTimeline', {}),
            timelines.get('objectTimeline', {}),
            unified_data.get('duration_seconds', 30)
        )

        context_data['visual_overlay_metrics'] = visual_overlay_metrics
        
        # Include raw timeline data as expected by prompt
        context_data['textOverlayTimeline'] = timelines.get('textOverlayTimeline', {})
        context_data['stickerTimeline'] = timelines.get('stickerTimeline', {})
        context_data['gestureTimeline'] = timelines.get('gestureTimeline', {})
        context_data['speechTimeline'] = timelines.get('speechTimeline', {})
        context_data['objectTimeline'] = timelines.get('objectTimeline', {})
    elif prompt_name == 'creative_density':
        # Compute density metrics instead of sending full timelines
        density_metrics = compute_creative_density_analysis(
            timelines,
            unified_data.get('duration_seconds', 30)
        )
        # Flatten the density_analysis dict to root level as expected by prompt
        density_data = density_metrics['density_analysis']
        context_data.update({
            # Core metrics
            'average_density': density_data.get('average_density'),
            'max_density': density_data.get('max_density'),
            'min_density': density_data.get('min_density'),
            'std_deviation': density_data.get('std_deviation'),
            'total_creative_elements': density_data.get('total_creative_elements'),
            'element_distribution': density_data.get('element_distribution'),
            'scene_changes': density_data.get('scene_changes'),
            'timeline_coverage': density_data.get('timeline_coverage'),
            
            # Dynamics
            'density_curve': density_data.get('density_curve'),
            'density_volatility': density_data.get('density_volatility'),
            'acceleration_pattern': density_data.get('acceleration_pattern'),
            'density_progression': density_data.get('density_progression'),
            'empty_seconds': density_data.get('empty_seconds'),
            
            # Interactions
            'multi_modal_peaks': density_data.get('multi_modal_peaks'),
            'element_cooccurrence': density_data.get('element_cooccurrence'),
            'dominant_combination': density_data.get('dominant_combination'),
            
            # Key events
            'peak_density_moments': density_data.get('peak_density_moments'),
            'dead_zones': density_data.get('dead_zones'),
            'density_shifts': density_data.get('density_shifts'),
            
            # Patterns
            'structural_patterns': density_data.get('structural_patterns'),
            'density_classification': density_data.get('density_classification'),
            'pacing_style': density_data.get('pacing_style'),
            'cognitive_load_category': density_data.get('cognitive_load_category'),
            'ml_tags': density_data.get('ml_tags'),
            
            # Quality
            'data_completeness': density_data.get('data_completeness'),
            'detection_reliability': density_data.get('detection_reliability'),
            'overall_confidence': density_data.get('overall_confidence'),
            
            # Legacy fields for compatibility
            'density_pattern_flags': density_data.get('density_pattern_flags'),
            'timeline_frame_counts': density_data.get('timeline_frame_counts')
        })

    elif prompt_name == 'emotional_journey':
        # Compute emotional metrics
        emotional_metrics = compute_emotional_metrics(
            expression_timeline=timelines.get('expressionTimeline', {}),
            speech_timeline=timelines.get('speechTimeline', {}),
            gesture_timeline=timelines.get('gestureTimeline', {}),
            duration=unified_data.get('duration_seconds', 30)
        )
        context_data['emotional_metrics'] = emotional_metrics

        # Also include raw timelines for validation
        context_data['expression_timeline'] = timelines.get('expressionTimeline', {})
        context_data['gesture_timeline'] = timelines.get('gestureTimeline', {})
        context_data['speech_timeline'] = timelines.get('speechTimeline', {})
        context_data['camera_distance_timeline'] = timelines.get('cameraDistanceTimeline', {})
        context_data['transcript'] = unified_data.get('metadata_summary', {}).get('transcript', '')

    elif prompt_name == 'person_framing':
        # Get enhanced human analysis data if available
        enhanced_human_data = unified_data.get('metadata_summary', {}).get('enhancedHumanAnalysis', {})
        
        # Compute person framing metrics
        person_framing_metrics = compute_person_framing_metrics(
            expression_timeline=timelines.get('expressionTimeline', {}),
            object_timeline=timelines.get('objectTimeline', {}),
            camera_distance_timeline=timelines.get('cameraDistanceTimeline', {}),
            person_timeline=timelines.get('personTimeline', {}),
            enhanced_human_data=enhanced_human_data,
            duration=unified_data.get('duration_seconds', 30)
        )
        
        # Add metrics to context data
        context_data['person_framing_metrics'] = person_framing_metrics

        # Add raw timeline data for validation as expected by prompt
        # Validate and clean timelines before adding
        camera_timeline = validate_timeline_data(timelines.get('cameraDistanceTimeline', {}), 'camera_distance_timeline')
        object_timeline = validate_timeline_data(timelines.get('objectTimeline', {}), 'object_timeline')
        expression_timeline = validate_timeline_data(timelines.get('expressionTimeline', {}), 'expression_timeline')
        person_timeline = validate_timeline_data(timelines.get('personTimeline', {}), 'person_timeline')
        
        # Clean timelines to reduce payload size
        context_data['camera_distance_timeline'] = clean_timeline_for_api(camera_timeline, 'camera_distance')
        context_data['object_timeline'] = clean_timeline_for_api(object_timeline, 'object')
        context_data['expression_timeline'] = clean_timeline_for_api(expression_timeline, 'expression')
        context_data['person_timeline'] = clean_timeline_for_api(person_timeline, 'person')
        
        # Include timeline summary for validation
        context_data['timeline_summary'] = {
            'total_frames': len(timelines.get('expressionTimeline', {})),
            'gesture_count': len(timelines.get('gestureTimeline', {})),
            'expression_count': len(timelines.get('expressionTimeline', {})),
            'object_detection_frames': len(timelines.get('objectTimeline', {})),
            'text_detection_frames': len(timelines.get('textOverlayTimeline', {})),
            'speech_segments': len(timelines.get('speechTimeline', {})),
            'scene_changes': len(timelines.get('sceneChangeTimeline', {}))
        }

        # Include insights if available
        context_data['insights'] = unified_data.get('insights', {})
        
        # Final payload size check
        payload_size = len(json.dumps(context_data))
        if payload_size > 200000:  # 200KB limit
            print(f"⚠️ Warning: person_framing payload is very large ({payload_size:,} bytes), applying additional compression")
            # Apply more aggressive compression
            context_data['object_timeline'] = compress_timeline_aggressively(context_data['object_timeline'], max_entries=30, remove_unknown=True)
            context_data['expression_timeline'] = compress_timeline_aggressively(context_data['expression_timeline'], max_entries=30)
            context_data['camera_distance_timeline'] = compress_timeline_aggressively(context_data['camera_distance_timeline'], max_entries=30)
            context_data['person_timeline'] = compress_timeline_aggressively(context_data['person_timeline'], max_entries=30)
            
            # Check again after compression
            new_size = len(json.dumps(context_data))
            print(f"✅ Compressed to {new_size:,} bytes ({new_size/1024:.1f} KB)")
            
            # If still too large, use extreme compression
            if new_size > 180000:  # Getting close to limit
                print(f"⚠️ Still large, applying extreme compression")
                context_data['object_timeline'] = compress_timeline_aggressively(context_data['object_timeline'], max_entries=20, remove_unknown=True)
                context_data['expression_timeline'] = compress_timeline_aggressively(context_data['expression_timeline'], max_entries=20)
                context_data['camera_distance_timeline'] = compress_timeline_aggressively(context_data['camera_distance_timeline'], max_entries=20)
                context_data['person_timeline'] = compress_timeline_aggressively(context_data['person_timeline'], max_entries=20)
    elif prompt_name == 'scene_pacing':
        # Compute scene pacing metrics
        scene_pacing_metrics = compute_scene_pacing_metrics(
            scene_timeline=timelines.get('sceneChangeTimeline', {}),
            video_duration=unified_data.get('duration_seconds', 30),
            object_timeline=timelines.get('objectTimeline', {}),
            camera_distance_timeline=timelines.get('cameraDistanceTimeline', {}),
            video_id=video_id  # Pass video_id for FPS context
        )

        # Add metrics to context data
        context_data['scene_pacing_metrics'] = scene_pacing_metrics
        
        # Include raw timeline for validation - cleaned to reduce payload
        scene_timeline = validate_timeline_data(timelines.get('sceneChangeTimeline', {}), 'scene_timeline')
        object_timeline = validate_timeline_data(timelines.get('objectTimeline', {}), 'object_timeline')
        camera_timeline = validate_timeline_data(timelines.get('cameraDistanceTimeline', {}), 'camera_distance_timeline')
        
        # Clean timelines to reduce payload size
        context_data['scene_timeline'] = clean_timeline_for_api(scene_timeline, 'scene')
        context_data['object_timeline'] = clean_timeline_for_api(object_timeline, 'object')
        context_data['camera_distance_timeline'] = clean_timeline_for_api(camera_timeline, 'camera_distance')
        
        # Final payload size check
        payload_size = len(json.dumps(context_data))
        if payload_size > 200000:  # 200KB limit
            print(f"⚠️ Warning: scene_pacing payload is very large ({payload_size:,} bytes), applying additional compression")
            # Apply more aggressive compression
            context_data['object_timeline'] = compress_timeline_aggressively(context_data['object_timeline'], max_entries=30, remove_unknown=True)
            context_data['scene_timeline'] = compress_timeline_aggressively(context_data['scene_timeline'], max_entries=40)
            context_data['camera_distance_timeline'] = compress_timeline_aggressively(context_data['camera_distance_timeline'], max_entries=30)
            
            # Check again after compression
            new_size = len(json.dumps(context_data))
            print(f"✅ Compressed to {new_size:,} bytes ({new_size/1024:.1f} KB)")
            
            # If still too large, use extreme compression
            if new_size > 180000:  # Getting close to limit
                print(f"⚠️ Still large, applying extreme compression")
                context_data['object_timeline'] = compress_timeline_aggressively(context_data['object_timeline'], max_entries=20, remove_unknown=True)
                context_data['scene_timeline'] = compress_timeline_aggressively(context_data['scene_timeline'], max_entries=25)
                context_data['camera_distance_timeline'] = compress_timeline_aggressively(context_data['camera_distance_timeline'], max_entries=20)

    elif prompt_name == 'speech_analysis':
        # Get metadata summary for transcript and speech segments
        metadata_summary = unified_data.get('metadata_summary', {})
        # Get enhanced human analysis data if available
        enhanced_human_data = metadata_summary.get('enhancedHumanAnalysis', {})
        # Compute speech analysis metrics
        speech_analysis_metrics = compute_speech_analysis_metrics(
            speech_timeline=timelines.get('speechTimeline', {}),
            transcript=metadata_summary.get('transcript', ''),
            speech_segments=metadata_summary.get('speechSegments', []),
            expression_timeline=timelines.get('expressionTimeline', {}),
            gesture_timeline=timelines.get('gestureTimeline', {}),
            human_analysis_data=enhanced_human_data,
            video_duration=unified_data.get('duration_seconds', 30)
        )

        # Add metrics to context data
        context_data['speech_analysis_metrics'] = speech_analysis_metrics
        
        # Include raw data for validation - only what prompt expects
        context_data['speech_timeline'] = timelines.get('speechTimeline', {})
        context_data['transcript'] = metadata_summary.get('transcript', '')
        context_data['speech_segments'] = metadata_summary.get('speechSegments', [])
        context_data['expression_timeline'] = timelines.get('expressionTimeline', {})
        context_data['gesture_timeline'] = timelines.get('gestureTimeline', {})
        
    elif prompt_name == 'metadata_analysis':
        # Get static metadata and metadata summary
        static_metadata = unified_data.get('static_metadata', {})
        metadata_summary = unified_data.get('metadata_summary', {})
        
        # Compute comprehensive metadata metrics
        metadata_metrics = compute_metadata_analysis_metrics(
            static_metadata=static_metadata,
            metadata_summary=metadata_summary,
            video_duration=unified_data.get('duration_seconds', 30)
        )
        
        # Add all computed metrics to context
        context_data['metadata_metrics'] = metadata_metrics
        
        # Don't include raw data - it's too large and already processed into metrics
        # Only include minimal validation data
        context_data['static_metadata'] = {
            'videoId': static_metadata.get('videoId', ''),
            'duration': static_metadata.get('duration', 0),
            'createTime': static_metadata.get('createTime', ''),
            'captionText': static_metadata.get('captionText', '')[:500],  # First 500 chars only
            'hashtags': static_metadata.get('hashtags', [])[:10]  # First 10 hashtags
        }
        
        # Include minimal author/stats for validation
        context_data['videoStats'] = static_metadata.get('stats', {})
        context_data['authorStats'] = static_metadata.get('author', {})
        
        # Get caption text and stats from static_metadata
        caption_text = static_metadata.get('captionText', '') or static_metadata.get('text', '') or ''
        stats = static_metadata.get('stats', {})
        
        # Count emojis (simplified - counts common emoji ranges)
        import re
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U00002600-\U000027BF"  # Miscellaneous Symbols
            u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
            "]+", flags=re.UNICODE)
        
        emojis = emoji_pattern.findall(caption_text)
        emoji_list = []
        for emoji_str in emojis:
            emoji_list.extend(list(emoji_str))
        
        context_data['emoji_count'] = len(emoji_list)
        context_data['emoji_list'] = list(set(emoji_list))  # Unique emojis
        
        # Count mentions
        mentions = re.findall(r'@\w+', caption_text)
        context_data['mention_count'] = len(mentions)
        context_data['mention_list'] = mentions
        
        # Check for links
        context_data['link_present'] = bool(re.search(r'https?://\S+|www\.\S+', caption_text))
        
        # Language detection
        context_data['language_code'] = detect_language_simple(caption_text)
        
        # Video metadata
        context_data['video_duration'] = unified_data.get('duration_seconds', 0)
        context_data['publish_time'] = static_metadata.get('createTime', '')
        
        # Engagement stats
        context_data['view_count'] = stats.get('viewCount', 0) or stats.get('playCount', 0) or 0
        context_data['like_count'] = stats.get('likeCount', 0) or stats.get('diggCount', 0) or 0
        context_data['comment_count'] = stats.get('commentCount', 0) or 0
        context_data['share_count'] = stats.get('shareCount', 0) or 0
        
        # Calculate engagement rate
        views = context_data['view_count']
        if views > 0:
            engagement = (context_data['like_count'] + context_data['comment_count'] + context_data['share_count']) / views
            context_data['engagement_rate'] = round(engagement * 100, 2)
        else:
            context_data['engagement_rate'] = 0.0
        
        # Creator information - check multiple possible locations
        author = static_metadata.get('author', {}) or static_metadata.get('authorMeta', {})
        context_data['creator_username'] = (
            author.get('uniqueId', '') or 
            author.get('nickname', '') or 
            author.get('username', '') or 
            ''
        )
        context_data['creator_follower_count'] = (
            author.get('followers', 0) or 
            author.get('followerCount', 0) or 
            0
        )
        
        # Music information - comprehensive extraction
        music_info = static_metadata.get('music', {}) or static_metadata.get('musicMeta', {})
        context_data['music_id'] = (
            music_info.get('id', '') or 
            music_info.get('musicId', '') or 
            static_metadata.get('musicId', '') or 
            ''
        )
        
        # Effect IDs - might be in different locations
        context_data['effect_ids'] = (
            static_metadata.get('effectIds', []) or 
            static_metadata.get('effects', []) or 
            static_metadata.get('effectStickers', []) or 
            []
        )
        
        # NLP analysis using our helper functions
        context_data['readability_score'] = compute_ml_readability(caption_text)
        
        # Sentiment analysis with ML-friendly structure
        sentiment = classify_sentiment_simple(caption_text)
        context_data['sentiment_analysis'] = {
            'category': sentiment,
            'confidence': 0.7  # Fixed confidence for simple analysis
        }
        
        # Keyword extraction
        context_data['keyword_extraction'] = extract_top_terms(caption_text, n=5)
        
        # Entity recognition patterns
        context_data['entity_recognition'] = extract_entity_patterns(caption_text)

    else:
        # For other prompts, include relevant timeline data
        context_data['timelines'] = timelines
        context_data['metadata_summary'] = unified_data.get('metadata_summary', {})

    return context_data
def update_progress(video_id, prompt_name, status, message=""):
    """Update progress file for real-time monitoring"""
    progress_file = f'insights/{video_id}/progress.json'
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)

    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {'prompts': {}, 'start_time': datetime.now().isoformat()}
        
        progress['prompts'][prompt_name] = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        progress['last_update'] = datetime.now().isoformat()
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to update progress file: {e}")
def run_single_prompt(video_id, prompt_name):
    """Run a single prompt for a video"""
    print(f"\n🎬 Running {prompt_name} for video {video_id}")
    update_progress(video_id, prompt_name, 'started')

    # Check if unified analysis exists
    unified_path = f'unified_analysis/{video_id}.json'
    if not os.path.exists(unified_path):
        print(f"❌ Error: {unified_path} not found!")
        # Try to list what files are in the directory
        import glob
        unified_files = glob.glob('unified_analysis/*.json')
        if unified_files:
            print(f"   Available files in unified_analysis/: {', '.join([os.path.basename(f) for f in unified_files[:5]])}")
        return False

    # Load unified analysis
    try:
        with open(unified_path, 'r') as f:
            unified_data = json.load(f)
        print(f"✅ Loaded unified analysis: {len(str(unified_data))} characters")
        update_progress(video_id, prompt_name, 'processing', 'Loaded unified analysis')
    except Exception as e:
        print(f"❌ Error loading unified analysis: {str(e)}")
        update_progress(video_id, prompt_name, 'failed', f'Error loading unified analysis: {str(e)}')
        return False

    # Extract ML data
    context_data = extract_real_ml_data(unified_data, prompt_name, video_id)

    # Pre-flight check for large data
    timelines = unified_data.get('timelines', {})
    total_timeline_entries = sum(len(t) for t in timelines.values() if isinstance(t, dict))
    
    if total_timeline_entries > 1000:
        print(f"\n⚠️  Large video analysis detected:")
        print(f"📊 Total timeline entries: {total_timeline_entries:,}")
        print(f"   Breakdown:")
        for name, timeline in timelines.items():
            if isinstance(timeline, dict) and len(timeline) > 50:
                print(f"   - {name}: {len(timeline)} entries")
        
        # Estimate processing time
        if prompt_name == 'person_framing':
            object_entries = len(timelines.get('objectTimeline', {}))
            estimated_time = 90 + (object_entries // 100) * 10
            print(f"⏱️  Estimated processing time: {estimated_time}s ({estimated_time/60:.1f} minutes)")
        
        print("")  # Empty line for readability

    # Load prompt template
    prompt_template_path = f'prompt_templates/{prompt_name}.txt'
    if not os.path.exists(prompt_template_path):
        print(f"❌ Error: Prompt template {prompt_template_path} not found!")
        return False

    with open(prompt_template_path, 'r') as f:
        prompt_text = f.read()

    # Add mode to context data
    context_data['mode'] = 'labeling'
    
    # Log payload size
    payload_size = len(json.dumps(context_data))
    print(f"📏 Payload size: {payload_size:,} characters")
    if payload_size > 1_000_000:
        print(f"   ⚠️  Large payload warning: This may take longer to process")
    
    # Run the prompt with enhanced error handling
    try:
        result = runner.run_claude_prompt(
            video_id=video_id,
            prompt_name=prompt_name,
            prompt_text=prompt_text,
            context_data=context_data
        )

        if result and result.get('success'):
            print(f"✅ {prompt_name} completed successfully!")
            update_progress(video_id, prompt_name, 'completed', 'Success')
            return True
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
            print(f"❌ {prompt_name} failed!")
            print(f"Error: {error_msg}")
            
            # Log additional error details if available
            if result and 'traceback' in result:
                print(f"\n📋 Detailed Error Traceback:", file=sys.stderr)
                print(result['traceback'], file=sys.stderr)
            
            update_progress(video_id, prompt_name, 'failed', error_msg)
            return False
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"❌ {prompt_name} crashed with exception!")
        print(f"Error: {error_msg}")
        print(f"\n📋 Exception Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        update_progress(video_id, prompt_name, 'failed', error_msg)
        return False
