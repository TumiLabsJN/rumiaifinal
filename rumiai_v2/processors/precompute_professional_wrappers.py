"""
Professional wrapper functions to convert basic metrics to 6-block format
Ensures all Python-only outputs maintain professional structure for ML training
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def convert_to_speech_professional(basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert basic speech metrics to professional 6-block format"""
    
    # Extract basic metrics with defaults
    total_words = basic_metrics.get('total_words', 0)
    speech_density = basic_metrics.get('speech_density', 0)
    words_per_minute = basic_metrics.get('words_per_minute', 0)
    vocabulary_diversity = basic_metrics.get('vocabulary_diversity', 0)
    silence_ratio = basic_metrics.get('silence_ratio', 1.0)
    
    return {
        "speechCoreMetrics": {
            "totalWords": total_words,
            "speechDensity": speech_density,
            "wordsPerMinute": words_per_minute,
            "vocabularyDiversity": vocabulary_diversity,
            "silenceRatio": silence_ratio,
            "speechCoverage": basic_metrics.get('speech_coverage', 0),
            "avgSegmentDuration": basic_metrics.get('avg_segment_duration', 0),
            "uniqueWords": basic_metrics.get('unique_words', 0),
            "confidence": 0.85
        },
        "speechDynamics": {
            "wpmProgression": list(basic_metrics.get('wpm_by_segment', {}).values())[:10],
            "silencePeriods": basic_metrics.get('silence_periods', []),
            "speechBursts": basic_metrics.get('speech_bursts', []),  # Fixed field name
            "burstPattern": basic_metrics.get('burst_pattern', 'none'),  # Added burst pattern
            "energyVariance": basic_metrics.get('energy_variance', 0),  # Added energy variance
            "hasAudioEnergy": basic_metrics.get('has_audio_energy', False),  # Added flag
            "pacingVariation": basic_metrics.get('pacing_std', 0),
            "temporalAlignment": basic_metrics.get('speech_visual_alignment', 0)
        },
        "speechInteractions": {
            "speechGestureSync": basic_metrics.get('speech_gesture_alignment', 0),
            "speechEmotionAlignment": basic_metrics.get('speech_emotion_alignment', 0),
            "speechTextOverlap": basic_metrics.get('speech_text_overlap', 0),
            "multiModalCoherence": basic_metrics.get('multimodal_coherence', 0)
        },
        "speechKeyEvents": {
            "longestSegment": {
                "duration": basic_metrics.get('longest_segment', 0),
                "timestamp": basic_metrics.get('longest_segment_time', "0-5s"),
                "words": basic_metrics.get('longest_segment_words', 0)
            },
            "climaxMoment": {
                "timestamp": basic_metrics.get('climax_timestamp', 0),
                "intensity": basic_metrics.get('climax_intensity', 0)
            },
            "silentMoments": basic_metrics.get('significant_silences', [])
        },
        "speechPatterns": {
            "repetitionRate": basic_metrics.get('repetition_rate', 0),
            "emphasisTechniques": basic_metrics.get('emphasis_techniques', []),
            "narrativeStyle": basic_metrics.get('narrative_style', "conversational"),
            "verbalHooks": basic_metrics.get('verbal_hooks', [])
        },
        "speechQuality": {
            "clarity": basic_metrics.get('clarity_score', 0.8),
            "engagement": basic_metrics.get('engagement_score', 0.7),
            "emotionalRange": basic_metrics.get('emotional_range', 0.5),
            "deliveryStyle": basic_metrics.get('delivery_style', "natural"),
            "overallScore": basic_metrics.get('overall_quality', 0.75)
        }
    }


def convert_to_metadata_professional(basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert basic metadata metrics to professional 6-block format"""
    
    return {
        "metadataCoreMetrics": {
            "captionLength": basic_metrics.get('caption_length', 0),
            "wordCount": basic_metrics.get('word_count', 0),
            "hashtagCount": basic_metrics.get('hashtag_count', 0),
            "emojiCount": basic_metrics.get('emoji_count', 0),
            "mentionCount": basic_metrics.get('mention_count', 0),
            "engagementRate": basic_metrics.get('engagement_rate', 0),
            "viewCount": basic_metrics.get('view_count', 0),
            "videoDuration": basic_metrics.get('video_duration', 0)
        },
        "metadataDynamics": {
            "hashtagStrategy": basic_metrics.get('hashtag_strategy', "none"),
            "captionStyle": basic_metrics.get('caption_style', "minimal"),
            "emojiDensity": basic_metrics.get('emoji_density', 0),
            "mentionDensity": basic_metrics.get('mention_density', 0),
            "publishHour": basic_metrics.get('publish_hour', 0),
            "publishDayOfWeek": basic_metrics.get('publish_day_of_week', 0)
        },
        "metadataInteractions": {
            "likeCount": basic_metrics.get('like_count', 0),
            "commentCount": basic_metrics.get('comment_count', 0),
            "shareCount": basic_metrics.get('share_count', 0),
            "saveCount": basic_metrics.get('save_count', 0),
            "engagementVelocity": basic_metrics.get('engagement_velocity', 0)
        },
        "metadataKeyEvents": {
            "topHashtags": basic_metrics.get('hashtag_list', [])[:5],
            "keyMentions": basic_metrics.get('mention_list', [])[:3],
            "primaryEmojis": basic_metrics.get('emoji_list', [])[:5],
            "callToAction": basic_metrics.get('cta_present', False)
        },
        "metadataPatterns": {
            "sentimentCategory": basic_metrics.get('sentiment_category', "neutral"),
            "urgencyLevel": basic_metrics.get('urgency_level', "none"),
            "contentCategory": basic_metrics.get('content_category', "general"),
            "viralPotential": basic_metrics.get('viral_potential_score', 0)
        },
        "metadataQuality": {
            "readabilityScore": basic_metrics.get('readability_score', 0.8),
            "sentimentPolarity": basic_metrics.get('sentiment_polarity', 0),
            "hashtagRelevance": basic_metrics.get('hashtag_relevance', 0.5),
            "captionEffectiveness": basic_metrics.get('caption_effectiveness', 0.6),
            "overallScore": basic_metrics.get('metadata_quality_score', 0.7)
        }
    }


def convert_to_person_framing_professional(basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert basic person framing metrics to professional 6-block format"""
    
    return {
        "personFramingCoreMetrics": {
            "primarySubject": basic_metrics.get('primary_subject', "single"),
            "averageFaceSize": basic_metrics.get('avg_face_size', 0),
            "faceVisibilityRate": basic_metrics.get('face_visibility_rate', 0),
            "framingConsistency": basic_metrics.get('framing_consistency', 0),
            "subjectCount": basic_metrics.get('subject_count', 1),
            "dominantFraming": basic_metrics.get('dominant_framing', "medium"),
            "eyeContactRate": basic_metrics.get('eye_contact_rate', 0)
        },
        "personFramingDynamics": {
            "framingProgression": basic_metrics.get('framing_progression', []),
            "distanceVariation": basic_metrics.get('distance_variation', 0),
            "framingTransitions": basic_metrics.get('framing_transitions', 0),
            "movementPattern": basic_metrics.get('movement_pattern', "static"),
            "stabilityScore": basic_metrics.get('stability_score', 0.8)
        },
        "personFramingInteractions": {
            "multiPersonDynamics": basic_metrics.get('multi_person_dynamics', {}),
            "speakerFraming": basic_metrics.get('speaker_framing', {}),
            "interactionZones": basic_metrics.get('interaction_zones', []),
            "socialDistance": basic_metrics.get('social_distance', "personal")
        },
        "personFramingKeyEvents": {
            "closeUpMoments": basic_metrics.get('closeup_moments', []),
            "groupShots": basic_metrics.get('group_shots', []),
            "framingChanges": basic_metrics.get('framing_changes', []),
            "keySubjectMoments": basic_metrics.get('key_subject_moments', [])
        },
        "personFramingPatterns": {
            "framingDistribution": basic_metrics.get('framing_distribution', {
                "extreme_close": 0,
                "close": 0.2,
                "medium": 0.6,
                "full": 0.2
            }),
            "compositionRule": basic_metrics.get('composition_rule', "center"),
            "cinematicStyle": basic_metrics.get('cinematic_style', "casual"),
            "framingTechnique": basic_metrics.get('framing_technique', "standard")
        },
        "personFramingQuality": {
            "compositionScore": basic_metrics.get('composition_score', 0.7),
            "framingAppropriate": basic_metrics.get('framing_appropriate', 0.8),
            "visualEngagement": basic_metrics.get('visual_engagement', 0.75),
            "professionalLevel": basic_metrics.get('professional_level', 0.6),
            "overallScore": basic_metrics.get('framing_quality', 0.72)
        }
    }


def convert_to_scene_pacing_professional(basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert basic scene pacing metrics to professional 6-block format"""
    
    return {
        "scenePacingCoreMetrics": {
            "totalScenes": basic_metrics.get('total_scenes', 0),
            "averageSceneDuration": basic_metrics.get('avg_scene_duration', 0),
            "scenesPerMinute": basic_metrics.get('scenes_per_minute', 0),
            "pacingScore": basic_metrics.get('pacing_score', 0),
            "rhythmConsistency": basic_metrics.get('rhythm_consistency', 0),
            "transitionSmoothing": basic_metrics.get('transition_smoothness', 0.8)
        },
        "scenePacingDynamics": {
            "pacingProgression": basic_metrics.get('pacing_progression', []),
            "sceneRhythm": basic_metrics.get('scene_rhythm', "regular"),
            "temporalFlow": basic_metrics.get('temporal_flow', "linear"),
            "accelerationPoints": basic_metrics.get('acceleration_points', []),
            "decelerationPoints": basic_metrics.get('deceleration_points', [])
        },
        "scenePacingInteractions": {
            "audioVisualSync": basic_metrics.get('audio_visual_sync', 0.8),
            "beatMatching": basic_metrics.get('beat_matching', 0),
            "emotionalPacing": basic_metrics.get('emotional_pacing_alignment', 0.7),
            "narrativeFlow": basic_metrics.get('narrative_flow', 0.75)
        },
        "scenePacingKeyEvents": {
            "longestScene": {
                "duration": basic_metrics.get('longest_scene_duration', 0),
                "timestamp": basic_metrics.get('longest_scene_time', "0-5s")
            },
            "shortestScene": {
                "duration": basic_metrics.get('shortest_scene_duration', 0),
                "timestamp": basic_metrics.get('shortest_scene_time', "0-1s")
            },
            "pacingShifts": basic_metrics.get('pacing_shifts', []),
            "climaxTiming": basic_metrics.get('climax_timing', 0.7)
        },
        "scenePacingPatterns": {
            "editingStyle": basic_metrics.get('editing_style', "standard"),
            "transitionTypes": basic_metrics.get('transition_types', ["cut"]),
            "pacingPattern": basic_metrics.get('pacing_pattern', "steady"),
            "rhythmStructure": basic_metrics.get('rhythm_structure', "4/4")
        },
        "scenePacingQuality": {
            "flowScore": basic_metrics.get('flow_score', 0.75),
            "engagementPacing": basic_metrics.get('engagement_pacing', 0.7),
            "viewerRetention": basic_metrics.get('viewer_retention_estimate', 0.65),
            "editingQuality": basic_metrics.get('editing_quality', 0.7),
            "overallScore": basic_metrics.get('pacing_quality', 0.71)
        }
    }


# Wrapper function to automatically convert based on analysis type
def ensure_professional_format(result: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
    """
    Ensure the result is in professional 6-block format
    If it's already in 6-block format, return as-is
    Otherwise, convert from basic metrics
    """
    
    # Check if already in professional format (has CoreMetrics)
    if any(key.endswith('CoreMetrics') for key in result.keys()):
        return result
    
    # Convert based on analysis type
    converters = {
        'speech_analysis': convert_to_speech_professional,
        'metadata_analysis': convert_to_metadata_professional,
        'person_framing': convert_to_person_framing_professional,
        'scene_pacing': convert_to_scene_pacing_professional
    }
    
    converter = converters.get(analysis_type)
    if converter:
        logger.info(f"Converting {analysis_type} to professional 6-block format")
        return converter(result)
    
    logger.warning(f"No converter found for {analysis_type}, returning as-is")
    return result