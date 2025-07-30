"""
Output adapter for converting between 6-block ML format and legacy format.
Ensures backwards compatibility during transition.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OutputAdapter:
    """Converts between new 6-block format and legacy format for backwards compatibility"""
    
    @staticmethod
    def convert_6block_to_legacy(six_block_data: Dict[str, Any], prompt_type: str) -> Dict[str, Any]:
        """
        Convert new 6-block format to legacy format.
        
        Args:
            six_block_data: Dictionary containing 6-block structured output
            prompt_type: Type of prompt (e.g., 'creative_density', 'emotional_journey')
            
        Returns:
            Dictionary in legacy format
        """
        if not six_block_data:
            return {}
        
        try:
            if prompt_type == 'creative_density':
                return OutputAdapter._convert_creative_density(six_block_data)
            elif prompt_type == 'emotional_journey':
                return OutputAdapter._convert_emotional_journey(six_block_data)
            elif prompt_type == 'person_framing':
                return OutputAdapter._convert_person_framing(six_block_data)
            elif prompt_type == 'scene_pacing':
                return OutputAdapter._convert_scene_pacing(six_block_data)
            elif prompt_type == 'speech_analysis':
                return OutputAdapter._convert_speech_analysis(six_block_data)
            elif prompt_type == 'visual_overlay_analysis':
                return OutputAdapter._convert_visual_overlay(six_block_data)
            elif prompt_type == 'metadata_analysis':
                return OutputAdapter._convert_metadata_analysis(six_block_data)
            else:
                logger.warning(f"Unknown prompt type for conversion: {prompt_type}")
                return six_block_data
                
        except Exception as e:
            logger.error(f"Error converting {prompt_type} to legacy format: {e}")
            return six_block_data
    
    @staticmethod
    def _convert_creative_density(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert creative density 6-block to legacy format"""
        core = data.get('densityCoreMetrics', {})
        dynamics = data.get('densityDynamics', {})
        patterns = data.get('densityPatterns', {})
        key_events = data.get('densityKeyEvents', {})
        
        return {
            'creative_density_score': core.get('avgDensity', 0) * 2,  # Scale to 0-10
            'elements_per_second': core.get('elementsPerSecond', 0),
            'density_pattern': dynamics.get('accelerationPattern', 'unknown'),
            'peak_moments': key_events.get('peakMoments', []),
            'visual_complexity': patterns.get('densityClassification', 'medium'),
            'ml_tags': patterns.get('mlTags', []),
            'confidence': core.get('confidence', 0.0)
        }
    
    @staticmethod
    def _convert_emotional_journey(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert emotional journey 6-block to legacy format"""
        core = data.get('emotionalCoreMetrics', {})
        dynamics = data.get('emotionalDynamics', {})
        patterns = data.get('emotionalPatterns', {})
        
        return {
            'dominant_emotion': core.get('dominantEmotion', 'neutral'),
            'emotional_arc': dynamics.get('emotionalArc', 'stable'),
            'emotion_changes': core.get('emotionTransitions', 0),
            'engagement_pattern': patterns.get('viewerJourneyMap', 'steady'),
            'emotional_peaks': dynamics.get('peakEmotionMoments', []),
            'confidence': core.get('confidence', 0.0)
        }
    
    @staticmethod
    def _convert_person_framing(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert person framing 6-block to legacy format"""
        core = data.get('personFramingCoreMetrics', {})
        dynamics = data.get('personFramingDynamics', {})
        patterns = data.get('personFramingPatterns', {})
        
        return {
            'person_presence_rate': core.get('personPresenceRate', 0),
            'average_person_count': core.get('avgPersonCount', 0),
            'dominant_framing': core.get('dominantFraming', 'medium'),
            'framing_quality': core.get('overallFramingQuality', 0),
            'production_value': patterns.get('productionValue', 'medium'),
            'confidence': core.get('confidence', 0.0)
        }
    
    @staticmethod
    def _convert_scene_pacing(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert scene pacing 6-block to legacy format"""
        core = data.get('scenePacingCoreMetrics', {})
        dynamics = data.get('scenePacingDynamics', {})
        patterns = data.get('scenePacingPatterns', {})
        
        return {
            'total_scenes': core.get('totalScenes', 0),
            'average_scene_duration': core.get('avgSceneDuration', 0),
            'pacing_style': patterns.get('pacingStyle', 'moderate'),
            'rhythm_consistency': core.get('pacingConsistency', 0),
            'editing_style': patterns.get('editingRhythm', 'standard'),
            'confidence': core.get('confidence', 0.0)
        }
    
    @staticmethod
    def _convert_speech_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert speech analysis 6-block to legacy format"""
        core = data.get('speechCoreMetrics', {})
        dynamics = data.get('speechDynamics', {})
        patterns = data.get('speechPatterns', {})
        key_events = data.get('speechKeyEvents', {})
        
        return {
            'total_speech_duration': core.get('speechDuration', 0),
            'words_per_minute': core.get('wordsPerMinute', 0),
            'speech_clarity': core.get('speechClarityScore', 0),
            'delivery_style': patterns.get('deliveryStyle', 'conversational'),
            'key_phrases': key_events.get('keyPhrases', []),
            'pause_pattern': dynamics.get('pausePattern', 'natural'),
            'confidence': core.get('confidence', 0.0)
        }
    
    @staticmethod
    def _convert_visual_overlay(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert visual overlay 6-block to legacy format"""
        core = data.get('overlaysCoreMetrics', {})
        dynamics = data.get('overlaysDynamics', {})
        patterns = data.get('overlaysPatterns', {})
        key_events = data.get('overlaysKeyEvents', {})
        
        return {
            'total_text_overlays': core.get('totalTextOverlays', 0),
            'overlay_density': core.get('overlayDensity', 0),
            'text_strategy': patterns.get('overlayStrategy', 'moderate'),
            'visual_complexity': core.get('visualComplexityScore', 0),
            'key_text_moments': key_events.get('keyTextMoments', []),
            'production_quality': patterns.get('productionQuality', 'medium'),
            'confidence': core.get('confidence', 0.0)
        }
    
    @staticmethod
    def _convert_metadata_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata analysis 6-block to legacy format"""
        core = data.get('metadataCoreMetrics', {})
        dynamics = data.get('metadataDynamics', {})
        key_events = data.get('metadataKeyEvents', {})
        
        return {
            'caption_length': core.get('captionLength', 0),
            'hashtag_count': core.get('hashtagCount', 0),
            'engagement_rate': core.get('engagementRate', 0),
            'viral_potential': dynamics.get('viralPotentialScore', 0),
            'caption_sentiment': dynamics.get('sentimentCategory', 'neutral'),
            'hashtags': key_events.get('hashtags', []),
            'hooks': key_events.get('hooks', []),
            'confidence': core.get('confidence', 0.0)
        }
    
    @staticmethod
    def merge_all_analyses(analyses: Dict[str, Any], output_format: str = 'v1') -> Dict[str, Any]:
        """
        Merge all ML analyses into final output format.
        
        Args:
            analyses: Dictionary with keys as prompt types and values as 6-block data
            output_format: 'v1' for legacy, 'v2' for 6-block
            
        Returns:
            Merged analysis in requested format
        """
        if output_format == 'v2':
            # Return as-is for v2 format
            return analyses
        
        # Convert to legacy format
        legacy_output = {}
        for prompt_type, data in analyses.items():
            if isinstance(data, dict) and not data.get('error'):
                legacy_data = OutputAdapter.convert_6block_to_legacy(data, prompt_type)
                legacy_output[prompt_type] = legacy_data
            else:
                # Pass through errors or invalid data
                legacy_output[prompt_type] = data
        
        return legacy_output