"""
Semantic Interpretations for RumiAI Features

This module defines semantic ranges for all 26 base features used in Stage 7 analysis.
Each feature has defined ranges that convert numeric values to creator-friendly labels.

Usage:
    from config.semantic_interpretations import SEMANTIC_INTERPRETATIONS, interpret_value

    label, desc = interpret_value('average_face_size', 0.058)
    # Returns: ('wide shot', 'face occupies <6% of frame')

Maintenance:
    - When Stage 6 adds new features, add corresponding semantic ranges here
    - Ranges are data-driven (based on production quartile analysis)
    - Validate ranges periodically as data distribution changes
"""

SEMANTIC_INTERPRETATIONS = {
    # ========================================
    # CATEGORY 1: VISUAL COMPOSITION (4 features)
    # ========================================

    'average_face_size': {
        'metric_type': 'ratio',
        'direction': 'higher_is_closer',
        'unit': 'proportion of frame occupied by face',
        'data_range': (0.031, 0.456),  # Updated from visual calibration (2025-10-29)
        'ranges': [
            (0.0, 0.04, 'wide shot', 'full body visible with environment'),
            (0.04, 0.09, 'medium shot', 'upper body (chest/waist up)'),
            (0.09, 0.15, 'close-up', 'head and shoulders prominent'),
            (0.15, 0.30, 'tight close-up', 'head fills most of frame'),
            (0.30, 1.0, 'extreme close-up', 'face dominates entire frame')
        ],
        'notes': 'RECALIBRATED using 6 real TikTok images. Original thresholds (0.06, 0.10, 0.20) adjusted to (0.04, 0.09, 0.15, 0.30) based on visual validation. Added "tight close-up" category for better granularity between close-up and extreme close-up.'
    },

    'person_count': {
        'metric_type': 'count',
        'direction': 'neutral',
        'unit': 'number of people visible in frame',
        'data_range': (1.0, 5.0),  # Typical range (outliers exist for crowd scenes)
        'ranges': [
            (0, 1.5, 'solo', 'single person on screen'),
            (1.5, 2.5, 'duo', 'two people visible'),
            (2.5, 5.0, 'small group', '3-5 people visible'),
            (5.0, 100, 'large group', 'more than 5 people')
        ],
        'notes': 'Count-based metric from YOLO detection. Averages can be non-integer due to frame-level counting. Top performers avg 3.6, suggesting intimate or duo content performs better.'
    },

    'object_count': {
        'metric_type': 'count',
        'direction': 'neutral',
        'unit': 'number of detected objects/props',
        'data_range': (2.28, 7.68),  # From production thresholds
        'ranges': [
            (0, 3.0, 'minimal objects', 'very few objects/props visible (between 0-3)'),
            (3.0, 6.0, 'moderate objects', 'balanced visual elements (between 3-6)'),
            (6.0, 10.0, 'many objects', 'rich visual environment (between 6-10)'),
            (10.0, 100, 'cluttered', 'visually dense/busy composition (10+)')
        ],
        'notes': 'YOLO object detection counts. Production shows 2-8 objects typical. Neither high nor low appears definitively better (top/bottom similar).'
    },

    'overlay_unique_count': {
        'metric_type': 'count',
        'direction': 'neutral',
        'unit': 'number of unique text overlay elements',
        'data_range': (0, 8),  # From production data
        'ranges': [
            (0, 0.5, 'no text', 'no text overlays present'),
            (0.5, 3, 'minimal text', '1-3 text elements'),
            (3, 5, 'moderate text', '3-5 text elements'),
            (5, 8, 'heavy text', '5+ text elements')
        ],
        'notes': 'OCR-detected unique text overlays. Top performers avg 2.8-2.9, bottom avg 3.6-5.1. Suggests minimal text may perform better.'
    },

    # ========================================
    # CATEGORY 2: ENERGY/PERFORMANCE (4 features) - IN PROGRESS
    # ========================================

    'energy_max': {
        'metric_type': 'continuous',
        'direction': 'higher_is_more',
        'unit': 'peak audio amplitude (RMS)',
        'data_range': (0.054, 0.225),
        'ranges': [
            (0.0, 0.12, 'subdued peak', 'Low Volume Peak'),
            (0.12, 0.20, 'soft peak', 'Mid Volume Peak'),
            (0.20, 0.40, 'moderate peak', 'Loud Peak'),
            (0.40, 1.0, 'loud peak', 'Very Loud Peak')
        ],
        'notes': 'Measures the single loudest moment (peak amplitude) in a temporal window. Captures overall loudness regardless of source (voice, music, sound effects, ambient noise, or combination). Production range: 0.054-0.225.'
    },

    'energy_level': {
        'metric_type': 'continuous',
        'direction': 'higher_is_more',
        'unit': 'average audio amplitude (RMS)',
        'data_range': (0.023, 0.195),
        'ranges': [
            (0.0, 0.05, 'very quiet', 'Very Low Average Volume'),
            (0.05, 0.12, 'quiet', 'Low Average Volume'),
            (0.12, 0.30, 'moderate', 'Mid Average Volume'),
            (0.30, 0.40, 'loud', 'High Average Volume'),
            (0.40, 1.0, 'very loud', 'Very High Average Volume')
        ],
        'notes': 'Measures average loudness across entire temporal window. Captures sustained audio energy regardless of source. Production range: 0.023-0.195.'
    },

    'energy_variance': {
        'metric_type': 'variance',
        'direction': 'neutral',
        'unit': 'variance in audio amplitude',
        'data_range': (0.00035, 0.00362),
        'ranges': [
            (0.0, 0.001, 'very consistent', 'minimal volume variation'),
            (0.001, 0.0025, 'moderate variation', 'some dynamic range'),
            (0.0025, 0.004, 'varied', 'dynamic audio changes'),
            (0.004, 1, 'highly varied', 'significant volume shifts')
        ],
        'notes': 'Measures how much audio volume fluctuates over time. Low variance = consistent delivery, high variance = dynamic volume changes. Production range: 0.00035-0.00362. Direction is neutral as optimal variance depends on content style.'
    },

    'emotional_valence': {
        'metric_type': 'continuous',
        'direction': 'higher_is_more',
        'unit': 'emotion score (-1 to +1 scale)',
        'data_range': (-0.45, 0.08),
        'ranges': [
            (-1.0, -0.3, 'very negative', 'predominantly sad/neutral expressions'),
            (-0.3, -0.1, 'negative', 'somewhat negative/neutral emotions'),
            (-0.1, 0.1, 'neutral', 'balanced emotional tone'),
            (0.1, 0.3, 'positive', 'happy/joyful expressions'),
            (0.3, 1.0, 'very positive', 'extremely happy/excited')
        ],
        'notes': 'Measures overall emotional tone from facial expressions on -1 to +1 scale (FEAT analysis). Production range: -0.45 to 0.08. Top performers avg 0.08 (neutral) vs bottom avg -0.45 (very negative).'
    },

    # ========================================
    # CATEGORY 3: AUDIO/SPEECH (4 features) - IN PROGRESS
    # ========================================

    'pitch_scatter_ratio': {
        'metric_type': 'ratio',
        'direction': 'neutral',
        'unit': 'pitch variation measure',
        'data_range': (0.594, 0.913),
        'ranges': [
            (0.0, 0.40, 'monotone', 'very consistent pitch'),
            (0.40, 0.60, 'steady tone', 'slight pitch variation'),
            (0.60, 0.80, 'varied tone', 'moderate pitch changes'),
            (0.80, 1.0, 'very expressive', 'high pitch variation')
        ],
        'notes': 'Measures voice pitch variation from Whisper audio analysis. Production range: 0.594-0.913. Direction is neutral as optimal pitch variation is context-dependent.'
    },

    'word_count': {
        'metric_type': 'count',
        'direction': 'neutral',
        'unit': 'number of spoken words',
        'data_range': (0, 44),
        'ranges': [
            (0, 1, 'no speech', 'silent or music-only'),
            (1, 5, 'few words', 'minimal dialogue'),
            (5, 20, 'some words', 'light talking'),
            (20, 45, 'many words', 'substantial dialogue'),
            (45, 100, 'very many words', 'heavy dialogue')
        ],
        'notes': 'Raw word count from Whisper transcription. Production range: 0-44 words per window. Note: Interpretation varies by window duration - same count represents different speaking rates in different buckets.'
    },

    # TODO: speech_coverage, word_density_std

    # ========================================
    # REMAINING CATEGORIES (15 features) - TODO
    # ========================================
    # CATEGORY 4: Eye Contact/Gaze (3 features)
    #   - eye_contact_rate, eye_contact_consistency, gaze_variance
    # CATEGORY 5: Scene/Pacing (4 features)
    #   - scene_count, scene_duration_variance, longest_scene, shortest_scene
    # CATEGORY 6: Movement/Temporal/Metadata (7 features)
    #   - gesture_count, energy_progression_slope, middle_to_closing_energy,
    #     middle_to_closing_delta, hour, day_of_week, dominant_emotion_id
}

# NOTE: 4 of 26 features defined (Visual Composition complete)
# See FeatureThresholdLogic.md for methodology and process
# See LLMOutputFix.md for finalized decisions


def interpret_value(feature: str, value: float) -> tuple[str, str]:
    """
    Convert numeric value to semantic label and description.

    Args:
        feature: Base feature name (without window prefix, e.g., 'average_face_size')
        value: Numeric value to interpret

    Returns:
        tuple[str, str]: (label, description)
        Example: ('wide shot', 'face occupies <6% of frame')

    Examples:
        >>> interpret_value('average_face_size', 0.058)
        ('wide shot', 'face occupies <6% of frame')

        >>> interpret_value('person_count', 3.6)
        ('small group', '3-5 people visible')

        >>> interpret_value('overlay_unique_count', 2.9)
        ('moderate text', '3-4 text elements')
    """
    if feature not in SEMANTIC_INTERPRETATIONS:
        # Feature not yet defined, return placeholder
        return ('unknown', f'value: {value:.2f}')

    interp = SEMANTIC_INTERPRETATIONS[feature]

    # Find matching range
    for min_val, max_val, label, description in interp['ranges']:
        if min_val <= value < max_val:
            return (label, description)

    # Edge case: value at max boundary (handle inclusive upper bound)
    if value >= interp['ranges'][-1][1]:
        return (interp['ranges'][-1][2], interp['ranges'][-1][3])

    # Fallback: value outside all ranges (shouldn't happen with proper range definitions)
    return ('out_of_range', f'value: {value:.2f}')


# Export for easy importing
__all__ = ['SEMANTIC_INTERPRETATIONS', 'interpret_value']
