# temporal_compute.py - Self-contained module

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
from collections import defaultdict
from difflib import SequenceMatcher
import sys
from pathlib import Path

# Add project root to path to import foundation.buckets
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from foundation.buckets import assign_bucket

logger = logging.getLogger(__name__)

# Word count metrics tracking (for validation of word-level timestamp fix)
_word_count_metrics = defaultdict(lambda: {'word_based': 0, 'fallback': 0})

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

# Updated to match BucketsPlan.md - thresholds now based on TOTAL video duration
BUCKET_THRESHOLDS = {
    'no_middle_max': 9,           # 0-9s: No middle segments
    'three_segments_max': 18,     # 9-18s: 3 middle segments
    'four_segments_max': 33,      # 18-33s: 4 middle segments
    'five_segments_max': 75,      # 33-75s: 5 middle segments
    # >75s: 5 segments (capped)
}

# Keep old thresholds for backward compatibility reference
SEGMENT_THRESHOLDS = {
    'min_duration_for_segments': 3,     # DEPRECATED: Use BUCKET_THRESHOLDS
    'three_segments_max': 12,            # DEPRECATED: Use BUCKET_THRESHOLDS
    'four_segments_max': 27,             # DEPRECATED: Use BUCKET_THRESHOLDS
}

"""
BUCKET CALCULATION RULES (Per BucketsPlan.md):
- Videos 0-9 seconds: No middle segments
- Videos 9-18 seconds: 3 middle segments
- Videos 18-33 seconds: 4 middle segments
- Videos 33-75 seconds: 5 middle segments
- Videos >75 seconds: 5 middle segments (capped)

These thresholds align with natural content boundaries for TikTok videos.
"""

# ============================================
# FUZZY TEXT DEDUPLICATION (OCRFix2.md)
# ============================================

# Configuration for dual thresholds
CHAR_SIMILARITY_THRESHOLD = 0.85   # For OCR typos like "yu" vs "you"
TOKEN_SIMILARITY_THRESHOLD = 0.75  # For partial text like "things" vs "3 things"

def calculate_similarities(text1: str, text2: str) -> Tuple[float, float]:
    """
    Calculate both character and token similarity separately.
    Returns: (character_similarity, token_similarity)
    """
    # Character-based similarity (catches typos like "yu" vs "you")
    char_sim = SequenceMatcher(None, text1, text2).ratio()

    # Token-based similarity (catches partial text like "things" vs "3 things")
    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 or not words2:
        token_sim = 0.0
    else:
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        token_sim = len(intersection) / len(union)

    return char_sim, token_sim

def should_merge_texts(text1: str, text2: str) -> bool:
    """
    Determine if two texts should be merged using dual thresholds.

    Design Decision: Use separate thresholds for different error types:
    - Character threshold (0.85): For OCR typos like "yu" vs "you"
    - Token threshold (0.75): For partial text like "things" vs "3 things"

    Merge if EITHER threshold is met.
    """
    char_sim, token_sim = calculate_similarities(text1, text2)

    return (char_sim >= CHAR_SIMILARITY_THRESHOLD) or (token_sim >= TOKEN_SIMILARITY_THRESHOLD)

def deduplicate_with_fuzzy_matching(texts: List[str]) -> List[str]:
    """
    Deduplicate texts using O(n²) fuzzy matching with dual thresholds.
    Performance: ~50ms for 200 texts (acceptable overhead).

    Design Decisions:
    - Dual thresholds: 0.85 character, 0.75 token similarity
    - Keep longest version when duplicates found (more complete OCR)
    - O(n²) comparison for thoroughness (performance impact negligible)
    """
    unique_texts = []

    for text in texts:
        found_match = False

        # Compare against all existing unique texts (O(n²) but thorough)
        for i, unique_text in enumerate(unique_texts):
            if should_merge_texts(text, unique_text):
                found_match = True
                # Keep the longer version (assumed more complete)
                if len(text) > len(unique_text):
                    unique_texts[i] = text
                break

        if not found_match:
            unique_texts.append(text)

    return unique_texts

def normalize_text_global(text: str) -> str:
    """Normalize text for grouping similar OCR detections."""
    import re
    original = text.strip()
    # Convert to lowercase
    text = text.lower()

    # Common emoji to text mappings
    emoji_mappings = {
        '🔥': '[fire]',
        '❤️': '[heart]', '❤': '[heart]',
        '😊': '[smile]', '😀': '[smile]', '😃': '[smile]',
        '👍': '[thumbsup]', '👍🏻': '[thumbsup]',
        '💯': '[100]',
        '✨': '[sparkles]',
        '🎉': '[party]',
        '💪': '[strong]',
        '🙏': '[pray]',
        '😂': '[laugh]',
        '😍': '[love]',
        '🤔': '[think]',
        '👀': '[eyes]',
        '🎯': '[target]',
        '🚀': '[rocket]',
        '💰': '[money]',
        '🏆': '[trophy]',
        '⭐': '[star]',
        '✅': '[check]',
        '❌': '[x]',
        '⚡': '[lightning]',
        '🌟': '[star]',
    }

    # Replace known emojis with text markers
    for emoji, marker in emoji_mappings.items():
        text = text.replace(emoji, marker)

    # Remove remaining emojis and special characters, keep alphanumeric, spaces, and brackets
    text = re.sub(r'[^a-z0-9\s\[\]]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())

    # If normalization resulted in empty string but original had content,
    # it was an unmapped emoji/symbol - create unique identifier
    if not text and original:
        import hashlib
        text = f"[emoji_{hashlib.md5(original.encode()).hexdigest()[:6]}]"

    return text

def spatial_cluster_overlays_with_bbox(overlay_entries: List[Dict]) -> List[Dict]:
    """
    Group texts by spatial proximity using bbox coordinates.
    Merge fragments at same vertical position (y) and close horizontal position (x).

    This is the NEW spatial clustering that uses bbox coordinates instead of position metadata.
    Applied ONLY to overlay_entries (texts classified as overlays by position).

    Returns: List[Dict] with keys 'text', 'timestamp', 'y_pos' to preserve info for temporal clustering
    """
    if not overlay_entries:
        return []

    # Group by time bucket (100ms buckets for same-timestamp grouping)
    time_buckets = {}
    for entry in overlay_entries:
        timestamp = entry.get('timestamp', entry.get('start', 0))
        if hasattr(timestamp, 'seconds'):
            timestamp = timestamp.seconds

        bucket = round(timestamp / 0.1) * 0.1  # 100ms buckets
        if bucket not in time_buckets:
            time_buckets[bucket] = []
        time_buckets[bucket].append(entry)

    merged_results = []

    for bucket, bucket_entries in time_buckets.items():
        # Sort by x-position (left to right)
        bucket_entries.sort(key=lambda e: e.get('data', {}).get('bbox', [0])[0] if len(e.get('data', {}).get('bbox', [])) >= 4 else 0)

        # Group by y-position proximity (±50 pixels)
        y_groups = []
        for entry in bucket_entries:
            bbox = entry.get('data', {}).get('bbox', [])
            if len(bbox) >= 4:
                y_pos = bbox[1]  # y-coordinate

                # Find existing group with similar y-position
                found_group = False
                for group in y_groups:
                    group_y = group['y_avg']
                    if abs(y_pos - group_y) < 50:  # Within 50 pixels
                        old_count = len(group['entries'])
                        group['entries'].append(entry)
                        # Update running average: (old_avg * old_count + new_value) / new_count
                        group['y_avg'] = (group_y * old_count + y_pos) / (old_count + 1)
                        found_group = True
                        break

                if not found_group:
                    y_groups.append({
                        'y_avg': y_pos,
                        'entries': [entry]
                    })
            else:
                # No bbox data - treat as single text
                y_groups.append({
                    'y_avg': 0,
                    'entries': [entry]
                })

        # Merge texts within each y-group and preserve timestamp + y_pos
        for group in y_groups:
            texts = [e.get('data', {}).get('text', '') for e in group['entries']]
            merged_text = ' '.join(texts) if len(texts) > 1 else texts[0]

            # Use timestamp from first entry in group
            first_entry = group['entries'][0]
            timestamp = first_entry.get('timestamp', first_entry.get('start', 0))
            if hasattr(timestamp, 'seconds'):
                timestamp = timestamp.seconds

            merged_results.append({
                'text': merged_text,
                'timestamp': timestamp,
                'y_pos': group['y_avg'],
                'data': {'text': merged_text}
            })

    return merged_results

def temporal_cluster_overlays(overlay_entries: List[Dict]) -> List[str]:
    """
    Enhanced temporal clustering with spatial proximity clustering (FixOCR6).
    Four-step approach: temporal grouping → spatial grouping → within-bucket dedup → cross-bucket clustering.
    """
    if not overlay_entries:
        return []

    # Step 1: Group by temporal proximity (0.5s buckets - optimal for OCR detection patterns)
    TEMPORAL_BUCKET_SIZE = 0.5  # Balanced: captures multi-line text without over-merging
    time_buckets = {}

    for entry in overlay_entries:
        time_bucket = round(entry['timestamp'] / TEMPORAL_BUCKET_SIZE) * TEMPORAL_BUCKET_SIZE
        if time_bucket not in time_buckets:
            time_buckets[time_bucket] = []
        time_buckets[time_bucket].append(entry)

    # Step 2: Apply spatial clustering within each temporal bucket
    bucket_results = []
    for bucket_entries in time_buckets.values():
        if bucket_entries:
            spatially_clustered = spatial_cluster_within_bucket(bucket_entries)
            bucket_results.extend(spatially_clustered)

    # Step 3: Cross-bucket clustering for multi-line text spanning multiple buckets
    if bucket_results:
        return aggressive_fuzzy_matching(bucket_results)

    return []

def spatial_cluster_within_bucket(bucket_entries: List[Dict]) -> List[str]:
    """
    Group texts by spatial proximity (y-position) within temporal bucket.
    Now uses y_pos from spatial clustering for accurate grouping.
    """
    if not bucket_entries:
        return []

    # Group by y-position proximity (±50 pixels)
    y_groups = []
    for entry in bucket_entries:
        y_pos = entry.get('y_pos', 0)
        text = entry.get('text', '')

        # Find existing group with similar y-position
        found_group = False
        for group in y_groups:
            if abs(y_pos - group['y_avg']) < 50:  # Within 50 pixels
                group['texts'].append(text)
                # Update running average
                old_count = len(group['texts']) - 1
                group['y_avg'] = (group['y_avg'] * old_count + y_pos) / len(group['texts'])
                found_group = True
                break

        if not found_group:
            y_groups.append({
                'y_avg': y_pos,
                'texts': [text]
            })

    # Merge texts within each y-group into single overlay
    # If texts are at same y-position in same time bucket, they're the same overlay
    merged_texts = []
    for group in y_groups:
        if len(group['texts']) == 1:
            merged_texts.append(group['texts'][0])
        else:
            # Multiple texts at same y-position = OCR variations of same overlay
            # Keep the longest one (usually most complete)
            longest = max(group['texts'], key=len)
            merged_texts.append(longest)

    return merged_texts

def merge_spatial_group(texts: List[str]) -> List[str]:
    """
    Attempt to merge texts in same spatial location into coherent phrases.
    """
    if len(texts) <= 1:
        return texts

    # Try all combinations to form valid phrases
    merged = []
    used_indices = set()

    for i, text1 in enumerate(texts):
        if i in used_indices:
            continue

        best_combination = [text1]
        best_indices = {i}

        # Look for complementary texts
        for j, text2 in enumerate(texts):
            if j in used_indices or j == i:
                continue

            if can_form_phrase(best_combination + [text2]):
                best_combination.append(text2)
                best_indices.add(j)

        if len(best_combination) > 1:
            # Merge into single phrase
            merged_phrase = combine_into_phrase(best_combination)
            merged.append(merged_phrase)
        else:
            merged.append(text1)

        used_indices.update(best_indices)

    return merged

def can_form_phrase(texts: List[str]) -> bool:
    """
    Determine if texts can logically form a coherent phrase.
    """
    if len(texts) < 2:
        return False

    # Normalize texts
    normalized = [normalize_text_global(text) for text in texts]

    # Check common phrase patterns
    combined_variations = generate_phrase_combinations(normalized)

    for combination in combined_variations:
        if is_valid_phrase(combination):
            return True

    return False

def is_valid_phrase(phrase: str) -> bool:
    """
    Check if combined text forms a valid phrase.
    More conservative approach - only exact matches and strict patterns.
    """
    # Known complete phrases (exact matches only)
    fitness_phrases = {
        "daily movement", "daily exercise", "daily workout",
        "more variety of plant foods", "more variety plant foods",
        "vitamin d", "gut health", "better health", "wellness hacks",
        "what i eat in a day", "eat in a day"
    }

    phrase_lower = phrase.lower().strip()

    # Normalize common OCR variations for exact matching
    normalized_phrase = phrase_lower.replace("@f", "of").replace("varfety", "variety")

    # Direct exact match only
    if normalized_phrase in fitness_phrases:
        return True

    # Conservative adjective-noun pattern (only 2 words)
    words = normalized_phrase.split()
    if len(words) == 2:
        if has_adjective_noun_pattern(words):
            return True

    return False

def has_adjective_noun_pattern(words: List[str]) -> bool:
    """
    Check if two words form common adjective-noun pattern.
    """
    common_adjectives = {"daily", "more", "better", "healthy", "fresh", "natural"}
    common_nouns = {"movement", "exercise", "variety", "foods", "health", "workout"}

    if len(words) == 2:
        word1, word2 = words
        return (word1 in common_adjectives and word2 in common_nouns) or \
               (word2 in common_adjectives and word1 in common_nouns)
    return False

def generate_phrase_combinations(texts: List[str]) -> List[str]:
    """
    Generate different orderings of texts to test for valid phrases.
    """
    if len(texts) <= 1:
        return texts

    # Sort by length (shorter words often come first)
    sorted_texts = sorted(texts, key=len)

    combinations = [
        " ".join(sorted_texts),
        " ".join(reversed(sorted_texts)),
        " ".join(sorted(sorted_texts, key=lambda x: x.lower()))
    ]

    return combinations

def combine_into_phrase(texts: List[str]) -> str:
    """
    Intelligently combine texts into single phrase.
    """
    # Try different orderings
    combinations = generate_phrase_combinations(texts)

    # Return the most phrase-like combination
    for combo in combinations:
        if is_valid_phrase(combo):
            return combo.strip()

    # Fallback: alphabetical order
    return " ".join(sorted(texts)).strip()

def aggressive_fuzzy_matching(texts: List[str]) -> List[str]:
    """
    More aggressive fuzzy matching for temporal clustering context.
    Uses lower thresholds to group OCR fragments that belong to same overlay.
    """
    if not texts:
        return []

    unique_texts = []

    for text in texts:
        if not text or not text.strip():
            continue

        found_match = False

        # More aggressive thresholds for temporal clustering
        for i, unique_text in enumerate(unique_texts):
            if should_merge_texts_aggressive(text, unique_text):
                found_match = True
                # Keep the longer version (assumed more complete)
                if len(text) > len(unique_text):
                    unique_texts[i] = text
                break

        if not found_match:
            unique_texts.append(text)

    return unique_texts

def should_merge_texts_aggressive(text1: str, text2: str) -> bool:
    """
    Aggressive text similarity for temporal clustering.
    Uses lower thresholds to group fragments of same overlay.
    """
    import difflib

    if not text1 or not text2:
        return False

    # Convert to lowercase for comparison
    t1, t2 = text1.lower().strip(), text2.lower().strip()

    if t1 == t2:
        return True

    # More aggressive thresholds
    AGGRESSIVE_CHAR_THRESHOLD = 0.7   # Was 0.85
    AGGRESSIVE_TOKEN_THRESHOLD = 0.6  # Was 0.75

    # Character-level similarity (handles OCR errors)
    char_sim = difflib.SequenceMatcher(None, t1, t2).ratio()

    # Token-level similarity (handles partial matches)
    tokens1 = set(t1.split())
    tokens2 = set(t2.split())
    if tokens1 and tokens2:
        token_sim = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    else:
        token_sim = 0.0

    # Additional checks for fragments
    # If one text is contained in another (handles "day" vs "what eat in a day")
    if len(t1) >= 3 and len(t2) >= 3:
        if t1 in t2 or t2 in t1:
            return True

    # Check if they share significant common words
    if tokens1 and tokens2:
        common_words = tokens1 & tokens2
        if common_words and len(common_words) >= min(len(tokens1), len(tokens2)) * 0.5:
            return True

    return (char_sim >= AGGRESSIVE_CHAR_THRESHOLD) or (token_sim >= AGGRESSIVE_TOKEN_THRESHOLD)

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
# HASHTAG EXTRACTION AND ANALYSIS
# ============================================

def extract_hashtag_metrics(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and analyze hashtag strategy from video metadata.

    Args:
        metadata: Video metadata containing hashtags

    Returns:
        Dictionary with hashtag metrics
    """
    hashtags = metadata.get('hashtags', [])

    # Define generic hashtags (expanded list)
    generic_hashtags = [
        # Discovery-focused (original 6)
        'fyp', 'foryou', 'foryoupage', 'viral', 'trending', 'explore',

        # Platform identity (new)
        'tiktok',           # Platform name itself
        'tiktokviral',      # Viral aspiration

        # Creator community (new)
        'tiktokcreator',    # Creator-focused discovery
        'contentcreator',   # Generic creator tag

        # Engagement bait (new)
        'funny',            # General entertainment
        'duet',             # Collaboration/reaction content

        # Trending variations (new)
        'trendingvideo',    # Variation of trending
        'tiktokchallenge'   # Challenge participation
    ]

    # Count hashtags
    total_count = len(hashtags)
    generic_count = 0
    specific_hashtags = []

    for tag in hashtags:
        # Extract tag text (handles both string and dict formats)
        if isinstance(tag, dict):
            tag_text = tag.get('name', '').lower().strip('#')
        else:
            tag_text = str(tag).lower().strip('#')

        if tag_text in generic_hashtags:
            generic_count += 1
        else:
            specific_hashtags.append(tag_text)

    # Calculate metrics
    generic_ratio = generic_count / total_count if total_count > 0 else 0

    # Return only ML-compatible numeric features
    return {
        'hashtag_count': total_count,
        'generic_hashtag_count': generic_count,
        'specific_hashtag_count': total_count - generic_count,
        'generic_ratio': round(generic_ratio, 3)  # Keep 3 decimals for ML precision
    }

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
        if entry.get('entry_type') == 'scene_change':
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
    # Simplified: Only extract fields that are actually used (timestamp and bbox)
    face_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'face':
            face_data = entry.get('data', {})
            bbox = face_data.get('bbox')
            if bbox:  # Only add if bbox exists
                face_timeline.append({
                    'timestamp': entry.get('start', 0),
                    'bbox': bbox
                    # Excluded unused fields: confidence, count, frame_number
                })
    timelines['face_timeline'] = face_timeline
    
    # Text overlay data from timeline entries (OCR-detected text)
    text_entry_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'text':
            # Use start time (when text appears) for caption alignment with speech
            start_time = entry.get('start', 0)
            end_time = entry.get('end', start_time)
            text_entry_timeline.append({
                'timestamp': start_time,  # Changed from midpoint to start_time
                'start': start_time,
                'end': end_time,
                'data': entry.get('data', {}),  # Preserve ALL data fields (text, position, size, style)
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

    # DEBUG: Log what we extracted
    logger.warning(f"[DEBUG] Extracted {len(timelines['speech_segments'])} speech segments from Whisper data")
    if timelines['speech_segments']:
        logger.warning(f"[DEBUG] First segment: {timelines['speech_segments'][0]}")
    
    # YOLO object detections from timeline entries (processed by timeline_builder)
    object_timeline = []
    for entry in timeline_entries:
        if entry.get('entry_type') == 'object':
            object_timeline.append({
                'timestamp': entry.get('start', 0),
                'className': entry.get('data', {}).get('class', 'unknown'),  # Use className for compatibility
                'confidence': entry.get('data', {}).get('confidence', 0),
                'bbox': entry.get('data', {}).get('bbox', []),
                'trackId': entry.get('data', {}).get('track_id', None),  # Use trackId for compatibility
                'tracked': entry.get('data', {}).get('tracked', True)  # Preserve tracking status
            })
    timelines['object_timeline'] = object_timeline
    
    # Face data validation - ensure timeline_builder processed faces correctly
    # We use timeline entries as single source of truth (MediaPipe → timeline_builder → timeline entries)
    ml_mediapipe = ml_data.get('mediapipe', {})
    ml_faces = ml_mediapipe.get('faces', [])

    # Fail-fast validation to catch timeline_builder bugs
    if ml_faces and not timelines.get('face_timeline', []):
        raise ValueError(f"Data integrity error: Timeline builder missing {len(ml_faces)} faces from MediaPipe. "
                        f"This indicates a bug in timeline_builder that must be fixed.")
    elif not ml_faces and not timelines.get('face_timeline', []):
        logger.debug("No faces detected in video - both sources agree")

    # Note: face_timeline is already populated from timeline entries above (lines 302-310)
    # Removed redundant MediaPipe extraction that was overwriting timeline data
    
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
    if video_duration < HOOK_WINDOW_DURATION:
        return {
            'hook': (0, video_duration),
            'middle': None,
            'closing': None
        }
    elif video_duration < 9:  # Use < for consistent boundary handling (9 <= x goes to next bucket)
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
    Updated to match BucketsPlan.md: no middle segments for videos ≤9s.

    Uses SEGMENT_THRESHOLDS constants defined at module level.
    """
    if video_duration < 9:  # Use < for consistent boundary handling (9 <= x has middle segments)
        return None  # Return None instead of {} for consistency
    
    middle_start = HOOK_WINDOW_DURATION
    middle_end = video_duration - CLOSING_WINDOW_DURATION
    middle_duration = middle_end - middle_start

    # Handle edge cases
    if middle_duration <= 0:
        return None

    # Determine number of segments based on TOTAL video duration (BucketsPlan.md)
    # Use < for consistent boundary handling (upper bound exclusive)
    # This ensures videos at exact boundaries (18s, 33s, 75s) go into the higher bucket
    if video_duration < BUCKET_THRESHOLDS['three_segments_max']:
        num_segments = 3  # 9-17.999s videos (9 <= x < 18)
    elif video_duration < BUCKET_THRESHOLDS['four_segments_max']:
        num_segments = 4  # 18-32.999s videos (18 <= x < 33)
    elif video_duration < BUCKET_THRESHOLDS['five_segments_max']:
        num_segments = 5  # 33-74.999s videos (33 <= x < 75)
    else:
        num_segments = 5  # >=75s videos (capped at 5)
    
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

def classify_text_hybrid(text_entry: Dict, speech_segments: List[Dict], video_height: int = 1080) -> tuple:
    """
    Classify single text entry using hybrid position + adaptive fuzzy approach.

    Returns:
        (classification, reason) where classification is 'caption' or 'overlay'
    """
    # Extract bbox
    bbox = text_entry.get('data', {}).get('bbox', [])

    if len(bbox) < 4:
        # Fallback: No bbox data, use speech overlap only
        text_content = text_entry.get('data', {}).get('text', '')
        timestamp = text_entry.get('start', text_entry.get('timestamp', 0))
        if hasattr(timestamp, 'seconds'):
            timestamp = timestamp.seconds

        # Use standard fuzzy matching
        overlap = calculate_speech_overlap_standard(text_content, timestamp, speech_segments)

        if overlap > 0.7:
            return 'caption', 'fallback_high_overlap'
        else:
            return 'overlay', 'fallback_low_overlap'

    # Calculate vertical position
    y = bbox[1]
    height = bbox[3]
    y_center = y + (height / 2)
    y_percent = y_center / video_height

    # Calculate speech overlap with adaptive fuzzy
    text_content = text_entry.get('data', {}).get('text', '')
    timestamp = text_entry.get('start', text_entry.get('timestamp', 0))
    if hasattr(timestamp, 'seconds'):
        timestamp = timestamp.seconds

    speech_overlap = calculate_speech_overlap_adaptive(text_content, timestamp, speech_segments, y_percent)

    # Position-based classification
    if y_percent > 0.6:
        # BOTTOM REGION: Caption Zone (position is primary signal)
        # Text at y>60% is almost always captions, regardless of speech overlap
        return 'caption', f'bottom_zone_y{y_percent:.2f}_overlap{speech_overlap:.2f}'

    elif y_percent < 0.4:
        # TOP REGION: Overlay Zone (very strict threshold)
        if speech_overlap > 0.9:
            return 'caption', f'top_misplaced_y{y_percent:.2f}_overlap{speech_overlap:.2f}'
        else:
            return 'overlay', f'top_zone_y{y_percent:.2f}_overlap{speech_overlap:.2f}'

    else:
        # MIDDLE REGION: Uncertain Zone (moderate threshold)
        if speech_overlap > 0.6:
            return 'caption', f'middle_high_y{y_percent:.2f}_overlap{speech_overlap:.2f}'
        else:
            return 'overlay', f'middle_low_y{y_percent:.2f}_overlap{speech_overlap:.2f}'

def calculate_speech_overlap_standard(text: str, timestamp: float, speech_segments: List[Dict]) -> float:
    """Standard speech overlap calculation (for fallback)."""
    if not speech_segments:
        return 0.0

    grace_period = 0.5

    for segment in speech_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', seg_start + 1)

        if seg_start <= timestamp <= seg_end + grace_period:
            segment_text = segment.get('text', '').lower()
            if not segment_text:
                continue

            text_normalized = text.lower()
            from difflib import SequenceMatcher
            char_similarity = SequenceMatcher(None, text_normalized, segment_text).ratio()

            text_words = set(text_normalized.split())
            segment_words = set(segment_text.split())

            if text_words:
                common_words = text_words & segment_words
                word_overlap = len(common_words) / len(text_words)
            else:
                word_overlap = 0.0

            return max(char_similarity, word_overlap)

    return 0.0

def calculate_speech_overlap_adaptive(text: str, timestamp: float, speech_segments: List[Dict], y_percent: float) -> float:
    """Calculate speech overlap with position-adaptive parameters."""
    if not speech_segments:
        return 0.0

    # Unified grace period for all zones
    grace_period = 0.5

    # Find matching segments
    for segment in speech_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', seg_start + 1)

        if seg_start <= timestamp <= seg_end + grace_period:
            segment_text = segment.get('text', '').lower()
            if not segment_text:
                continue

            text_normalized = text.lower()

            from difflib import SequenceMatcher
            char_similarity = SequenceMatcher(None, text_normalized, segment_text).ratio()

            text_words = set(text_normalized.split())
            segment_words = set(segment_text.split())

            if text_words:
                common_words = text_words & segment_words
                word_overlap = len(common_words) / len(text_words)
            else:
                word_overlap = 0.0

            # Use word overlap as primary, char similarity as backup
            return max(char_similarity, word_overlap)

    return 0.0

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
        original = text.strip()

        # Apply OCR error fixes FIRST (before normalization)
        text = fix_ocr_errors(original)

        # Convert to lowercase
        text = text.lower()

        # Common emoji to text mappings
        emoji_mappings = {
            '🔥': '[fire]',
            '❤️': '[heart]', '❤': '[heart]',
            '😊': '[smile]', '😀': '[smile]', '😃': '[smile]',
            '👍': '[thumbsup]', '👍🏻': '[thumbsup]',
            '💯': '[100]',
            '✨': '[sparkles]',
            '🎉': '[party]',
            '💪': '[strong]',
            '🙏': '[pray]',
            '😂': '[laugh]',
            '😍': '[love]',
            '🤔': '[think]',
            '👀': '[eyes]',
            '🎯': '[target]',
            '🚀': '[rocket]',
            '💰': '[money]',
            '🏆': '[trophy]',
            '⭐': '[star]',
            '✅': '[check]',
            '❌': '[x]',
            '⚡': '[lightning]',
            '🌟': '[star]',
        }

        # Replace known emojis with text markers
        for emoji, marker in emoji_mappings.items():
            text = text.replace(emoji, marker)

        # Remove remaining emojis and special characters, keep alphanumeric, spaces, and brackets
        text = re.sub(r'[^a-z0-9\s\[\]]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())

        # If normalization resulted in empty string but original had content,
        # it was an unmapped emoji/symbol - create unique identifier
        if not text and original:
            import hashlib
            text = f"[emoji_{hashlib.md5(original.encode()).hexdigest()[:6]}]"

        return text

    def fix_ocr_errors(text: str) -> str:
        """Fix common OCR errors that break fuzzy matching."""
        if not text:
            return text

        # STAGE 1: Exact known OCR error fixes (dictionary)
        text_lower = text.lower()
        ocr_fixes = {
            'gvm': 'gym',        # v→y character substitution
            'sayoh': 'say oh',   # Word merging (missing space)
        }
        if text_lower in ocr_fixes:
            return ocr_fixes[text_lower]

        # STAGE 2: Pattern-based fixes (existing)
        # Split merged words (e.g., "formeI" → "forme I")
        text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)
        # Fix pipe symbols (e.g., "modeland|" → "modeland ")
        text = re.sub(r'(\w)\|', r'\1 ', text)
        return text

    def calculate_speech_overlap(text: str, timestamp: float, speech_segments: List[Dict]) -> float:
        """Calculate % overlap between text and speech at given timestamp."""
        if not speech_segments:
            return 0.0

        # Grace period for captions appearing slightly after speech ends
        CAPTION_GRACE_PERIOD = 0.5  # 500ms tolerance

        # Find speech segments overlapping this timestamp
        for segment in speech_segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', seg_start + 1)
            # Allow captions to appear up to 0.5s after speech ends
            if seg_start <= timestamp <= seg_end + CAPTION_GRACE_PERIOD:

                # TIMING STRICTNESS: Prevent false positives from thematic overlays
                time_alignment = min(abs(timestamp - seg_start), abs(timestamp - seg_end))
                if time_alignment > 2.0:  # Text >2s from speech boundaries
                    return 0.0  # Too far from speech timing - likely thematic overlay

                # Get segment text
                segment_text = segment.get('text', '').lower()
                if not segment_text:
                    continue

                # Apply OCR error correction before normalization
                text_fixed = fix_ocr_errors(text)
                segment_text_fixed = fix_ocr_errors(segment_text)

                # Normalize both texts
                text_normalized = normalize_text(text_fixed)
                segment_normalized = normalize_text(segment_text_fixed)
                
                # Calculate overlap using fuzzy matching to handle OCR errors
                if not text_normalized:
                    return 0.0

                # Use fuzzy similarity to handle OCR variations
                from difflib import SequenceMatcher
                char_similarity = SequenceMatcher(None, text_normalized, segment_normalized).ratio()

                # Also calculate word overlap for additional confidence
                text_words = set(text_normalized.split())
                segment_words = set(segment_normalized.split())

                if text_words and segment_words:
                    overlap_words = text_words.intersection(segment_words)
                    word_overlap_ratio = len(overlap_words) / len(text_words)
                else:
                    word_overlap_ratio = 0.0

                # Use the higher of character similarity or word overlap
                # This handles both OCR errors and partial text captures
                overlap_ratio = max(char_similarity, word_overlap_ratio)

                # STAGE 3: Edit distance fallback for unknown OCR errors
                # Only for single words ≤4 letters that didn't match well
                if len(text_normalized.split()) == 1:
                    single_word = text_normalized
                    if len(single_word) <= 4:
                        try:
                            from Levenshtein import distance
                            for speech_word in segment_normalized.split():
                                if len(speech_word) <= 4:
                                    edit_dist = distance(single_word, speech_word)
                                    if edit_dist == 1:
                                        # 1 character difference - boost overlap to 0.8
                                        overlap_ratio = max(overlap_ratio, 0.8)
                                        break
                                    elif edit_dist == 2 and len(single_word) >= 4:
                                        # 2 character difference on 4-letter word - boost to 0.6
                                        overlap_ratio = max(overlap_ratio, 0.6)
                                        break
                        except ImportError:
                            # python-Levenshtein not installed - skip edit distance
                            pass

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
            normalized = normalize_text(text_content)

            # Debug logging for overlay investigation
            if start == 11.0:  # Closing window
                logger.debug(f"[OVERLAY DEBUG] Closing window text at {timestamp}: '{text_content}' -> normalized: '{normalized}'")

            if text_content and normalized:  # Skip empty after normalization
                window_texts.append(entry)
            elif text_content and not normalized:
                # Text exists but normalizes to empty (e.g., emojis)
                if start == 11.0:
                    logger.debug(f"[OVERLAY DEBUG] Skipped emoji/special text at {timestamp}: '{text_content}'")
    
    # Early return if no texts in this window
    if not window_texts:
        return {
            'overlay_unique_count': 0,
            'has_captions': False
        }
    
    # Step 2: HYBRID CLASSIFICATION - Position + Adaptive Fuzzy
    caption_entries = []
    overlay_entries = []

    for entry in window_texts:
        classification, reason = classify_text_hybrid(entry, speech_segments, video_height=1080)

        if classification == 'caption':
            caption_entries.append(entry)
        else:
            overlay_entries.append(entry)
    
    # Step 3: OVERLAY COUNTING PIPELINE (ONLY for overlays - y<40%)
    # Multi-stage pipeline: spatial → temporal → aggressive fuzzy
    if overlay_entries:
        # 3a. Spatial clustering: Merge fragments using bbox proximity
        # Returns List[Dict] with 'text', 'timestamp', 'y_pos'
        overlay_spatial = spatial_cluster_overlays_with_bbox(overlay_entries)

        # 3b. Temporal clustering: Group same overlay over time (0.5s buckets)
        # Now receives proper timestamps from spatial clustering
        overlay_temporal = temporal_cluster_overlays(overlay_spatial)

        # 3c. Aggressive fuzzy matching: Handle OCR errors (0.7/0.6 thresholds)
        overlay_unique_texts = aggressive_fuzzy_matching(overlay_temporal)
    else:
        overlay_unique_texts = []

    # Step 4: CAPTION DEDUPLICATION (simple fuzzy only)
    # Captions (y>60%) don't need spatial/temporal clustering
    if caption_entries:
        caption_texts = [e.get('data', {}).get('text', '') for e in caption_entries]
        caption_unique_texts = deduplicate_with_fuzzy_matching(caption_texts)
    else:
        caption_unique_texts = []

    # Step 5: CROSS-CATEGORY DEDUPLICATION (safety net)
    # Remove overlays that also appear as captions (favor captions)
    overlay_unique_texts_filtered = []
    for overlay_text in overlay_unique_texts:
        # Check if this overlay text also appears in captions
        is_duplicate = False
        for caption_text in caption_unique_texts:
            # Use fuzzy matching to check if they're the same
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, overlay_text, caption_text).ratio()
            if similarity > 0.85:  # Very similar - consider it a duplicate
                is_duplicate = True
                break
        if not is_duplicate:
            overlay_unique_texts_filtered.append(overlay_text)

    overlay_unique_texts = overlay_unique_texts_filtered

    # Step 4: Calculate counts using deduplicated texts
    overlay_unique_count = len(overlay_unique_texts)
    caption_unique_count = len(caption_unique_texts)


    # Simplified caption metric - just binary presence
    has_captions = len(caption_unique_texts) > 0
    
    return {
        # Overlay metrics (marketing text only)
        'overlay_unique_count': overlay_unique_count,

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

def calculate_speech_metrics_for_window(speech_segments, start, end, duration, video_id=None):
    """
    Calculate speech coverage and word count for a temporal window.
    Uses word-level timestamps when available, falls back to proportional calculation.

    This is a module-level function for testability and potential reuse.

    Args:
        speech_segments: List of speech segments from Whisper
        start: Window start time in seconds
        end: Window end time in seconds
        duration: Window duration (end - start)
        video_id: Optional video ID for metrics tracking

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

    def is_non_speech_segment(text):
        """Returns True if text is non-speech (music tags or sung lyrics)"""
        if not text:
            return False

        text_stripped = text.strip()

        # Check for bracketed non-speech tags
        NON_SPEECH_TAGS = [
            '[Music]', '[MUSIC]', '[music]',
            '[Applause]', '[APPLAUSE]', '[applause]',
            '[Laughter]', '[LAUGHTER]', '[laughter]',
            '[BLANK_AUDIO]', '[Silence]', '[silence]',
            '[Background noise]', '[Inaudible]', '[inaudible]'
        ]

        if text_stripped in NON_SPEECH_TAGS:
            return True

        # Check for sung lyrics (musical note symbols)
        if '♪' in text or '♫' in text or '🎵' in text or '🎶' in text:
            return True

        # Check if text is ONLY non-speech tags with whitespace
        # e.g., "[Music] [Music] [Music]"
        cleaned_text = text
        for pattern in NON_SPEECH_TAGS:
            cleaned_text = cleaned_text.replace(pattern, '')

        return not cleaned_text.strip()  # True if nothing left after removing tags

    # Filter out non-speech segments
    actual_speech_segments = []
    filtered_count = 0

    for segment in speech_segments:
        text = segment.get('text', '')

        if not is_non_speech_segment(text):
            actual_speech_segments.append(segment)
        else:
            filtered_count += 1
            logger.warning(f"[SPEECH_FIX] Filtering non-speech: '{text[:50]}...'")

    # Log filtering results
    if filtered_count > 0:
        logger.warning(f"[SPEECH_FIX] Video {video_id}: Filtered {filtered_count}/{len(speech_segments)} non-speech segments")

    # Use filtered segments for calculation
    speech_segments = actual_speech_segments

    # If no actual speech segments remain, return zeros
    if not speech_segments:
        logger.warning(f"[SPEECH_FIX] Video {video_id}: No actual speech found after filtering")
        return 0.0, 0

    # DEBUG: Log what we're calculating (after filtering)
    logger.warning(f"[DEBUG] calculate_speech_metrics: {len(speech_segments)} actual speech segments for window {start}-{end}s")

    total_speech_duration = 0.0
    total_word_count = 0

    for segment in speech_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)
        seg_text = segment.get('text', '')
        words = segment.get('words', [])

        # DEBUG: Log segment details
        logger.warning(f"[DEBUG] Speech segment: start={seg_start}, end={seg_end}, words={len(words)}, text='{seg_text[:50]}...'")


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

            # Add the overlap duration to total
            total_speech_duration += overlap_duration

            # Count actual words that fall in this window (using word-level timestamps)
            if words:  # Use word-level timestamps if available
                if video_id:
                    _word_count_metrics[video_id]['word_based'] += 1

                for word in words:
                    word_start = word.get('start', 0)
                    word_end = word.get('end', word_start)
                    # Count word if its MIDPOINT falls within this window
                    word_midpoint = (word_start + word_end) / 2.0
                    if start <= word_midpoint < end:
                        total_word_count += 1
            else:
                # Fallback to proportional estimation if words array is empty
                if video_id:
                    _word_count_metrics[video_id]['fallback'] += 1
                    logger.warning(
                        f"Video {video_id}: No word timestamps for segment {seg_start:.1f}s-{seg_end:.1f}s, "
                        f"using proportional estimation"
                    )

                segment_duration = seg_end - seg_start
                # segment_duration is guaranteed > 0 due to validation above
                proportion_in_window = overlap_duration / segment_duration

                # Count proportional words
                segment_words = len(seg_text.split())
                words_in_window = segment_words * proportion_in_window
                total_word_count += int(round(words_in_window))

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

    # total_word_count is already integer (from word-based counting or int(round()) from fallback)
    return speech_coverage, total_word_count

def log_word_count_metrics(video_id: str):
    """
    Log word count metrics for validation. Call at end of temporal processing.

    Args:
        video_id: Video ID to log metrics for
    """
    if video_id in _word_count_metrics:
        metrics = _word_count_metrics[video_id]
        total = metrics['word_based'] + metrics['fallback']
        success_rate = (metrics['word_based'] / total * 100) if total > 0 else 0

        logger.info(
            f"Video {video_id}: Word-based: {metrics['word_based']}/{total} segments, "
            f"Fallback: {metrics['fallback']}/{total}, Success rate: {success_rate:.1f}%"
        )

        # Clean up to prevent memory leak
        del _word_count_metrics[video_id]

def calculate_pitch_metrics(audio_data: Dict[str, Any],
                           ml_data: Dict[str, Any],
                           start: float,
                           end: float) -> float:
    """
    Calculate pitch scatter ratio for a temporal window.

    Note: avg_pitch_normalized removed due to harmonic detection error - see SpeechPitchFix.md
    avg_pitch_hz also removed - only used internally, never stored

    Returns:
        pitch_scatter_ratio as float, or None if insufficient data
    """
    import numpy as np

    # Check audio data exists
    if not audio_data or 'pitch_frames' not in audio_data:
        return None

    # Extract window frames
    pitch_fps = audio_data.get('pitch_fps', 43.07)
    pitch_frames = audio_data['pitch_frames']

    start_frame = int(start * pitch_fps)
    end_frame = int(end * pitch_fps)

    # Bounds check
    if end_frame > len(pitch_frames):
        end_frame = len(pitch_frames)

    # Extract window and get voiced frames
    window_pitches = pitch_frames[start_frame:end_frame]
    voiced_pitches = [p for p in window_pitches if p > 80]

    # Minimum frames for calculation
    MIN_FRAMES_AVG = 10
    MIN_FRAMES_RANGE = 30

    # Early return if insufficient voiced content
    if len(voiced_pitches) < MIN_FRAMES_AVG:
        return None  # No pitch metrics available

    # Calculate average pitch (internal use only)
    avg_pitch_hz = float(np.mean(voiced_pitches))

    # Safety check (shouldn't happen with voiced pitches)
    if avg_pitch_hz == 0:
        return None

    # Calculate pitch scatter ratio (needs more frames for accuracy)
    if len(voiced_pitches) >= MIN_FRAMES_RANGE:
        pitch_scatter_ratio = (max(voiced_pitches) - min(voiced_pitches)) / avg_pitch_hz
        pitch_scatter_ratio = min(pitch_scatter_ratio, 1.0)  # Cap at 1.0
        return round(float(pitch_scatter_ratio), 4)
    else:
        # Not enough frames for reliable scatter measurement
        return None

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

def extract_instance_id(track_id: str) -> Optional[str]:
    """
    Extract instance ID from YOLO trackId with strict validation.

    Args:
        track_id: YOLO tracking ID (e.g., "obj_10_39")

    Returns:
        Instance ID string or None if invalid format
    """
    if not track_id or '_' not in track_id:
        return None

    parts = track_id.split('_')

    # Strict validation: must be exactly 3 parts
    if len(parts) != 3 or parts[0] != 'obj':
        return None

    # Validate numeric parts
    try:
        int(parts[1])  # Frame number must be valid
        int(parts[2])  # Instance ID must be valid
        return parts[2]
    except ValueError:
        return None

# DEPRECATED: Replaced by instance-based counting in the main computation
# def calculate_max_unique_persons(segment_objects, confidence_threshold=0.5):
#     """
#     Calculate the maximum number of unique persons visible at any point in the segment.
#
#     This counts unique persons by finding the maximum number of unique trackIds
#     at any single timestamp, after filtering by confidence.
#
#     Args:
#         segment_objects: List of object detections from YOLO
#         confidence_threshold: Minimum confidence to consider a detection valid
#
#     Returns:
#         int: Maximum number of unique persons visible at any point
#     """
#     # Filter for person detections with sufficient confidence
#     person_detections = [
#         obj for obj in segment_objects
#         if (obj.get('className') == 'person' or obj.get('label') == 'person')
#         and obj.get('confidence', 0) >= confidence_threshold
#     ]
#
#     if not person_detections:
#         return 0
#
#     # Group by timestamp
#     from collections import defaultdict
#     detections_by_timestamp = defaultdict(list)
#
#     for detection in person_detections:
#         timestamp = detection.get('timestamp', 0)
#         track_id = detection.get('trackId', f"unknown_{id(detection)}")
#         detections_by_timestamp[timestamp].append(track_id)
#
#     # Find maximum unique persons at any timestamp
#     max_persons = 0
#     for timestamp, track_ids in detections_by_timestamp.items():
#         unique_persons = len(set(track_ids))
#         max_persons = max(max_persons, unique_persons)
#
#     return max_persons

def bbox_overlap_ratio(bbox1, bbox2):
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0

    x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
    x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

    # Calculate intersection
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)

    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0  # No overlap

    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def spatial_dedup_people(detections):
    """Remove overlapping detections that represent the same person."""
    if not detections:
        return []

    # Sort by confidence (keep highest confidence detections)
    detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)

    unique_people = []
    for detection in detections:
        bbox = detection.get('bbox', [])
        if not bbox:
            continue

        # Check if this detection overlaps significantly with existing people
        is_duplicate = False
        for existing in unique_people:
            existing_bbox = existing.get('bbox', [])
            if bbox_overlap_ratio(bbox, existing_bbox) > 0.5:  # 50% overlap threshold
                is_duplicate = True
                break

        if not is_duplicate:
            unique_people.append(detection)

    return unique_people

def calculate_eye_contact_rate(timeline_entries: List[Dict], start: float, end: float) -> float:
    """
    Calculate the rate of eye contact within a temporal window.

    Args:
        timeline_entries: List of timeline entries containing gaze data
        start: Window start time in seconds
        end: Window end time in seconds

    Returns:
        Proportion of frames with eye contact (0.0 to 1.0)
    """
    eye_contact_scores = []

    # Collect eye contact scores from gaze entries in the window
    for entry in timeline_entries:
        if entry.get('entry_type') == 'gaze':
            entry_start = entry.get('start', 0)
            if start <= entry_start < end:
                eye_contact = entry.get('data', {}).get('eye_contact', 0)
                if eye_contact is not None:  # Valid measurement
                    eye_contact_scores.append(eye_contact)

    # Calculate mean if we have data
    if eye_contact_scores:
        return sum(eye_contact_scores) / len(eye_contact_scores)

    # Return 0 if no eye contact data available
    return 0.0

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
            if start <= entry_start < end:
                eye_contact = entry.get('data', {}).get('eye_contact', 0)
                if eye_contact is not None:  # Valid measurement
                    eye_contact_scores.append(eye_contact)

    # Calculate variance if we have enough data points
    if len(eye_contact_scores) > 1:
        return statistics.variance(eye_contact_scores)

    # Return 0 if insufficient data for variance calculation
    return 0.0

def process_segment(seg_bounds: Dict[str, float], timelines: Dict[str, Any],
                   audio_data: Dict[str, Any], ml_data: Dict[str, Any],
                   video_duration: float, video_id: Optional[str] = None) -> Dict[str, Any]:
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

    # DEBUG: Log what speech segments we have
    if speech_segments:
        logger.warning(f"[DEBUG] Found {len(speech_segments)} speech segments in segment {start}-{end}s")
        for seg in speech_segments[:3]:  # Log first 3
            logger.warning(f"[DEBUG] Speech segment: {seg}")

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
    # segment_scenes removed - was counting scene changes, not scenes (fixed below)
    segment_camera = [c for c in timelines.get('camera_distance_timeline', [])
                     if start <= c.get('timestamp', 0) < end]
    # Speech segments handled by calculate_speech_metrics_for_window function
    
    # Calculate all P0 required counts
    # sticker_count removed - see StickersProblem.md for why

    # Calculate person count using overlapping temporal windows (PersonFix2.md)
    # Calculate object count by unique class names across segment
    # Uses 0.2s windows with 0.1s stride to handle person track fragmentation

    WINDOW_SIZE = 0.2  # Matches 5 FPS sampling rate
    STRIDE = 0.1       # 50% overlap between windows

    # Initialize counters
    max_persons = 0
    unique_object_classes = set()

    # Process objects first: count unique class names (not instances)
    # E.g., 5 apples = 1 class, 1 apple + 1 book + 1 cup = 3 classes
    for obj in segment_objects:
        if obj.get('className') != 'person':
            timestamp = obj.get('timestamp', 0)
            if start <= timestamp < end:
                tracked = obj.get('tracked', False)  # Default False: only count successfully tracked objects
                if tracked:
                    unique_object_classes.add(obj.get('className'))

    # Enhanced person counting with dominant track logic to handle ByteTrack fragmentation
    track_counts = {}
    for obj in segment_objects:
        if obj.get('className') == 'person':
            timestamp = obj.get('timestamp', 0)
            if start <= timestamp < end:
                tracked = obj.get('tracked')
                if tracked is None:
                    logger.warning(f"Missing 'tracked' flag at {timestamp}s, defaulting to False")
                    tracked = False  # Default False: only count successfully tracked persons

                # Only count tracked detections (fallbacks have tracked=False)
                if tracked:
                    track_id = obj.get('trackId')
                    if track_id:
                        track_counts[track_id] = track_counts.get(track_id, 0) + 1

    # Calculate person count with dominant track logic
    if not track_counts:
        person_count = 0
    else:
        total_detections = sum(track_counts.values())
        max_track_count = max(track_counts.values())

        # If one track dominates with >95% of detections, it's the same person with tracking fragmentation
        if max_track_count / total_detections > 0.95:
            person_count = 1
        else:
            # Multiple balanced tracks = multiple people or uncertain case
            person_count = len(track_counts)
    object_count = len(unique_object_classes)

    # Deduplicate gestures - group consecutive same gestures within 0.8s
    unique_gestures = []
    if segment_gestures:
        # Sort by timestamp to process in order
        sorted_gestures = sorted(segment_gestures, key=lambda g: g.get('timestamp', 0))

        last_gesture = None
        for gesture in sorted_gestures:
            # Check if this is a continuation of the previous gesture
            if last_gesture and \
               gesture.get('type') == last_gesture.get('type') and \
               gesture.get('hand') == last_gesture.get('hand') and \
               (gesture.get('timestamp', 0) - last_gesture.get('timestamp', 0)) <= 2.0:
                # Same gesture continuing, don't count as new
                continue
            else:
                # New unique gesture
                unique_gestures.append(gesture)
                last_gesture = gesture

    gesture_count = len(unique_gestures)
    # expression_count removed - was always constant based on FEAT sampling rate

    # Fix scene counting: Count scenes that overlap with segment, not scene changes within segment
    scene_count = 0
    all_scenes = timelines.get('scene_change_timeline', [])
    if all_scenes:
        sorted_all_scenes = sorted(all_scenes, key=lambda x: x.get('timestamp', 0))
        for i, scene in enumerate(sorted_all_scenes):
            scene_start = scene.get('timestamp', 0)
            # Find next scene or use video end
            if i < len(sorted_all_scenes) - 1:
                scene_end = sorted_all_scenes[i + 1].get('timestamp', 0)
            else:
                scene_end = video_duration  # Use video duration as end of last scene

            # Check if this scene overlaps with our segment
            if scene_end > start and scene_start < end:
                scene_count += 1
    else:
        scene_count = 1  # Default: entire video is one scene if no scene changes detected
    
    # element_count removed per MLFeaturesGIGO.md - pure derivative
    # ML can compute sum if needed from raw components
    
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
    
    # Density metrics removed per MLFeaturesGIGO.md
    # avg_density is element_count/duration (double derivative)
    # changes_per_second is scene_count/duration (simple division)
    
    # Density metrics removed - see RemoveDensity.md
    # max_density/min_density measured sampling frequency (our processing artifact),
    # not scene complexity. Entity counts (person_count, object_count, etc.)
    # already provide meaningful scene complexity metrics without the noise.
    
    # Calculate speech metrics using word-level timestamps
    speech_coverage, word_count = calculate_speech_metrics_for_window(
        speech_segments, start, end, duration, video_id=video_id
    )

    # Speech content indicators removed per MLFeaturesGIGO.md
    # Arbitrary keyword matching creates false signals

    # Calculate gaze variance (NEW)
    gaze_variance = calculate_gaze_variance(
        timelines.get('timeline', {}).get('entries', []), start, end
    )

    # Calculate eye contact rate (NEW - Temporal Eye Contact Metrics)
    eye_contact_rate = calculate_eye_contact_rate(
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
    # Emotion ratios removed per MLFeaturesGIGO.md - perfect multicollinearity
    # Replaced with new non-collinear features below

    # Get face entries for this segment (needed early for emotion fallback logic)
    segment_faces = [f for f in timelines.get('face_timeline', [])
                    if start <= f.get('timestamp', 0) < end]

    # Calculate new emotion features (MLFeaturesGIGO.md)
    emotion_encoding = {
        'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4,
        'disgust': 5, 'surprise': 6, 'neutral': 7, 'no_person': 8
    }

    if total_emotions > 0:
        # Feature 1: Dominant emotion (with deterministic tie handling)
        max_count = max(emotion_counts.values())
        dominant_emotion = None
        # First emotion in this order wins ties
        for emotion in ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']:
            if emotion_counts.get(emotion, 0) == max_count:
                dominant_emotion = emotion
                break
        dominant_emotion_id = emotion_encoding[dominant_emotion]

        # Feature 2: Emotional valence (-1 to +1)
        # Note: Surprise excluded as it's ambiguous (can be positive or negative)
        positive_count = emotion_counts.get('joy', 0)
        negative_count = (emotion_counts.get('sadness', 0) +
                         emotion_counts.get('anger', 0) +
                         emotion_counts.get('fear', 0) +
                         emotion_counts.get('disgust', 0))
        # Neutral and surprise don't affect valence
        emotional_valence = (positive_count - negative_count) / total_emotions

        # Feature 3: Emotion consistency (how unified)
        max_emotion_count = max(emotion_counts.values())
        emotion_consistency = max_emotion_count / total_emotions
    else:
        # No emotions detected - distinguish neutral vs no_person
        if segment_faces:
            dominant_emotion_id = 7  # neutral (face detected but no emotions classified)
        else:
            dominant_emotion_id = 8  # no_person (no face detected)
        emotional_valence = 0.0
        emotion_consistency = 0.0

    # Calculate framing distribution from face bbox sizes
    # Since camera_distance entries don't exist, calculate from face entries
    # close (>25% frame), medium (8-25%), wide (<8%), none (no face)
    framing_counts = {'close': 0, 'medium': 0, 'wide': 0, 'none': 0}

    # Collect face areas for average calculation (average_face_size feature)
    face_areas = []

    for face in segment_faces:
        bbox = face.get('bbox', {})
        if bbox:
            # Calculate face area as percentage of frame
            face_area = bbox.get('width', 0) * bbox.get('height', 0) * 100

            # Store for averaging (new feature)
            face_areas.append(face_area)

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

    # Framing ratios removed per MLFeaturesGIGO.md - multicollinearity issue
    # Only keeping average_face_size as continuous metric

    # Calculate average face size (new feature from personframingfix.md)
    # Note: face_areas are already in percentage (0-100), convert to 0-1 range
    average_face_size = (sum(face_areas) / len(face_areas) / 100.0) if face_areas else 0.0
    
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
                # burst_pattern removed per MLFeaturesGIGO.md
            else:
                energy_level = energy_variance = energy_max = 0.0
        else:
            energy_level = energy_variance = energy_max = 0.0
    else:
        energy_level = energy_variance = energy_max = 0.0

    # Calculate pitch metrics if available
    pitch_scatter_ratio = calculate_pitch_metrics(audio_data, ml_data, start, end)

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
        'scene_count': scene_count,
        # element_count removed per MLFeaturesGIGO.md - pure derivative
        # max_density/min_density removed per RemoveDensity.md - sampling frequency artifact
        # P1 scene duration metrics
        'shortest_scene': shortest_scene,
        'longest_scene': longest_scene,
        # P2 scene variance
        'scene_duration_variance': scene_duration_variance,
        # Other metrics
        # changes_per_second removed - scene_count/duration
        'speech_coverage': speech_coverage,
        'word_count': word_count,
        # semantic speech features removed per MLFeaturesGIGO.md - arbitrary keywords
        'gaze_variance': gaze_variance,  # Gaze variance (NEW)
        'eye_contact_rate': eye_contact_rate,  # Eye contact rate (NEW - Temporal Eye Contact Metrics)
        # New emotion features (replace removed ratios)
        'dominant_emotion_id': dominant_emotion_id,
        'emotional_valence': round(emotional_valence, 4),
        'emotion_consistency': round(emotion_consistency, 4),
        # Framing ratios removed - multicollinearity
        'average_face_size': round(average_face_size, 4),  # New ML feature: continuous face size metric
        # Audio energy
        'energy_level': energy_level,
        'energy_variance': energy_variance,
        'energy_max': energy_max,
        # burst_pattern removed - arbitrary categorization
        # Pitch metrics
        'pitch_scatter_ratio': pitch_scatter_ratio if pitch_scatter_ratio is not None else 0.0  # 0.0 means "no voiced content detected"
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
        hook_data = process_segment(hook_bounds, timelines, audio_data, ml_data, video_duration, video_id=video_id)
        
        # Add window-specific audio metrics
        hook_data['energy_level'] = audio_energy_metrics.get('hook_energy_level', 0)
        hook_data['energy_variance'] = audio_energy_metrics.get('hook_energy_variance', 0)
        hook_data['energy_max'] = audio_energy_metrics.get('hook_energy_max', 0)
        # burst_pattern removed per MLFeaturesGIGO.md
    
    # ============================================
    # STEP 6: Process Middle Segments
    # ============================================

    if windows['middle'] is None:
        # No middle window for videos ≤9s (per BucketsPlan.md)
        middle_segments = None
    elif windows['middle']:
        segments = calculate_middle_segments(video_duration)
        if segments is None:
            middle_segments = None
        else:
            middle_segments = []
            for seg_name, seg_bounds in segments.items():
                seg_data = process_segment(seg_bounds, timelines, audio_data, ml_data, video_duration, video_id=video_id)
                seg_data['segment_name'] = seg_name
                middle_segments.append(seg_data)
    else:
        middle_segments = []
    
    # ============================================
    # STEP 7: Process Closing Window (last 3s)
    # ============================================
    
    closing_data = None
    if windows['closing']:
        closing_bounds = {
            'start': windows['closing'][0],
            'end': windows['closing'][1]
        }
        closing_data = process_segment(closing_bounds, timelines, audio_data, ml_data, video_duration, video_id=video_id)
        
        # Add window-specific audio metrics
        closing_data['energy_level'] = audio_energy_metrics.get('closing_energy_level', 0)
        closing_data['energy_variance'] = audio_energy_metrics.get('closing_energy_variance', 0)
        closing_data['energy_max'] = audio_energy_metrics.get('closing_energy_max', 0)
        # burst_pattern removed per MLFeaturesGIGO.md
    
    # ============================================
    # STEP 8: Calculate Metadata
    # ============================================

    # Extract hashtag metrics
    hashtag_metrics = extract_hashtag_metrics(metadata)

    calculated_metadata = {
        'video_id': video_id,
        'duration': video_duration,
        'url': metadata.get('url', ''),  # TikTok video URL
        'digg_count': metadata.get('likes', 0),  # Field name changed
        'play_count': metadata.get('views', 0),  # Field name changed
        'collect_count': metadata.get('saves', 0),  # Field name changed
        'share_count': metadata.get('shares', 0),
        'comment_count': metadata.get('comments', 0),
        'create_time': metadata.get('createTime', metadata.get('createTimeISO', '')),
        'author': metadata.get('author', {}).get('uniqueId', metadata.get('author', {}).get('name', '')),
        'description': metadata.get('description', ''),
        'hashtags': metadata.get('hashtags', [])  # Preserve raw hashtags array for downstream analysis
    }

    # Add gender detection data if available
    gender_data = ml_data.get('deepface_gender', {})
    if gender_data:
        # Normalize problematic gender values for CSV compatibility
        # null: No faces detected by DeepFace
        # multiple_people: Multiple people in video, gender ambiguous
        raw_gender = gender_data.get('gender')
        if raw_gender is None or raw_gender == 'multiple_people':
            normalized_gender = 'Unknown'
        else:
            normalized_gender = raw_gender

        calculated_metadata['gender_detection'] = {
            'gender': normalized_gender,
            'confidence': gender_data.get('confidence', 0.0),
            'method': gender_data.get('method', 'deepface')
        }

    # Add hashtag analysis
    if hashtag_metrics:
        calculated_metadata['hashtag_analysis'] = hashtag_metrics

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

    # Log word count metrics (SpeechCountFix.md validation)
    log_word_count_metrics(video_id)

    return result
