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
        'data_range': (0.034, 0.142),  # From production analysis (2025-10-29)
        'ranges': [
            (0.0, 0.06, 'wide shot', 'face occupies <6% of frame'),
            (0.06, 0.10, 'medium shot', 'face occupies 6-10% of frame'),
            (0.10, 0.20, 'close-up', 'face occupies 10-20% of frame'),
            (0.20, 1.0, 'extreme close-up', 'face occupies >20% of frame')
        ],
        'notes': 'Based on standard cinematography shot classifications. Production data shows most values 0.05-0.13. Higher values = closer to camera.'
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
            (0, 3.0, 'minimal objects', 'very few objects/props visible'),
            (3.0, 6.0, 'moderate objects', 'balanced visual elements'),
            (6.0, 10.0, 'many objects', 'rich visual environment'),
            (10.0, 100, 'cluttered', 'visually dense/busy composition')
        ],
        'notes': 'YOLO object detection counts. Production shows 2-8 objects typical. Neither high nor low appears definitively better (top/bottom similar).'
    },

    'overlay_unique_count': {
        'metric_type': 'count',
        'direction': 'neutral',
        'unit': 'number of unique text overlay elements',
        'data_range': (1.0, 5.08),  # From production data
        'ranges': [
            (0, 0.5, 'no text', 'no text overlays present'),
            (0.5, 2.5, 'minimal text', '1-2 text elements'),
            (2.5, 4.5, 'moderate text', '3-4 text elements'),
            (4.5, 20, 'heavy text', '5+ text elements')
        ],
        'notes': 'OCR-detected unique text overlays. Top performers avg 2.8-2.9, bottom avg 3.6-5.1. Suggests minimal text may perform better.'
    },

    # ========================================
    # REMAINING CATEGORIES (22 features) - TODO
    # ========================================
    # CATEGORY 2: Energy/Performance (4 features)
    #   - energy_max, energy_level, energy_variance, emotional_valence
    # CATEGORY 3: Audio/Speech (4 features)
    #   - pitch_scatter_ratio, word_count, speech_coverage, word_density_std
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
