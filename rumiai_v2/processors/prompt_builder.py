"""
Prompt builder for RumiAI v2.

This module builds Claude prompts from extracted ML data.
"""
from typing import Dict, Any, Optional
import json
import logging

from ..core.models import PromptType, PromptContext

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build Claude prompts with context data."""
    
    def __init__(self, prompt_templates: Dict[str, str]):
        """
        Initialize with prompt templates.
        
        Args:
            prompt_templates: Dictionary mapping prompt type to template text
        """
        self.templates = prompt_templates
    
    def build_prompt(self, context: PromptContext) -> str:
        """
        Build complete prompt from template and context.
        
        Returns formatted prompt ready for Claude API.
        """
        # Get template for this prompt type
        template = self.templates.get(context.prompt_type.value)
        if not template:
            raise ValueError(f"No template found for prompt type: {context.prompt_type.value}")
        
        # Build context sections based on prompt type
        context_sections = []
        
        # Video metadata section
        context_sections.append(self._build_metadata_section(context))
        
        # ML data section
        if context.ml_data:
            context_sections.append(self._build_ml_data_section(context))
        
        # Timeline sections
        if context.timelines:
            context_sections.append(self._build_timeline_section(context))
        
        # Temporal markers section
        if context.temporal_markers:
            context_sections.append(self._build_temporal_markers_section(context))
        
        # Combine all sections
        full_context = "\n\n".join(context_sections)
        
        # Format the complete prompt
        formatted_prompt = f"{template}\n\nCONTEXT DATA:\n{full_context}"
        
        # Log prompt size
        prompt_size = len(formatted_prompt)
        logger.info(f"Built prompt for {context.prompt_type.value}: {prompt_size} characters")
        
        return formatted_prompt
    
    def _build_metadata_section(self, context: PromptContext) -> str:
        """Build video metadata section."""
        metadata = context.metadata
        
        # Format key metrics
        lines = [
            "VIDEO METADATA:",
            f"- Duration: {context.duration:.1f} seconds",
            f"- Views: {metadata.get('views', 0):,}",
            f"- Likes: {metadata.get('likes', 0):,}",
            f"- Engagement Rate: {metadata.get('engagementRate', 0):.2f}%"
        ]
        
        # Add hashtags if present
        if 'hashtags' in metadata and metadata['hashtags']:
            hashtag_names = [h.get('name', '') for h in metadata['hashtags']]
            lines.append(f"- Hashtags: {', '.join(hashtag_names)}")
        
        # Add description preview
        if 'description' in metadata:
            desc_preview = metadata['description'][:200]
            if len(metadata['description']) > 200:
                desc_preview += "..."
            lines.append(f"- Description: {desc_preview}")
        
        return "\n".join(lines)
    
    def _build_ml_data_section(self, context: PromptContext) -> str:
        """Build ML analysis data section."""
        lines = ["ML ANALYSIS DATA:"]
        
        # Format ML data based on prompt type
        if context.prompt_type == PromptType.CREATIVE_DENSITY:
            lines.extend([
                f"- Total Text Elements: {context.ml_data.get('total_text_elements', 0)}",
                f"- Total Stickers: {context.ml_data.get('total_stickers', 0)}",
                f"- Unique Text Positions: {context.ml_data.get('unique_text_positions', 0)}"
            ])
        
        elif context.prompt_type == PromptType.SPEECH_ANALYSIS:
            lines.extend([
                f"- Total Speech Segments: {context.ml_data.get('total_segments', 0)}",
                f"- Speech Duration: {context.ml_data.get('total_speech_time', 0):.1f}s",
                f"- Speech Coverage: {context.ml_data.get('speech_percentage', 0):.1f}%",
                f"- Words Per Minute: {context.ml_data.get('words_per_minute', 0):.1f}",
                f"- Language: {context.ml_data.get('language', 'unknown')}"
            ])
            
            # Add top words
            top_words = context.ml_data.get('top_words', {})
            if top_words:
                word_list = [f"{word}({count})" for word, count in list(top_words.items())[:10]]
                lines.append(f"- Top Words: {', '.join(word_list)}")
        
        elif context.prompt_type == PromptType.PERSON_FRAMING:
            lines.extend([
                f"- Person Present: {context.ml_data.get('presence_percentage', 0):.1f}% of video",
                f"- Face Screen Time: {context.ml_data.get('face_screen_time', 0):.1f}%",
                f"- Eye Contact Ratio: {context.ml_data.get('eye_contact_ratio', 0):.2f}"
            ])
            
            # Add primary actions
            actions = context.ml_data.get('primary_actions', [])
            if actions:
                lines.append(f"- Primary Actions: {', '.join(actions)}")
        
        elif context.prompt_type == PromptType.SCENE_PACING:
            lines.extend([
                f"- Total Scenes: {context.ml_data.get('total_scenes', 0)}",
                f"- Scene Changes: {context.ml_data.get('scene_changes', 0)}",
                f"- Average Scene Duration: {context.ml_data.get('average_scene_duration', 0):.1f}s",
                f"- Scenes Per Minute: {context.ml_data.get('scenes_per_minute', 0):.1f}"
            ])
        
        # Add generic data for other prompt types
        else:
            for key, value in context.ml_data.items():
                if isinstance(value, (int, float)):
                    lines.append(f"- {key}: {value}")
                elif isinstance(value, list) and value and isinstance(value[0], str):
                    lines.append(f"- {key}: {', '.join(value[:5])}")  # First 5 items
        
        return "\n".join(lines)
    
    def _build_timeline_section(self, context: PromptContext) -> str:
        """Build timeline data section."""
        lines = ["TIMELINE DATA:"]
        
        # Add different timelines based on prompt type
        if context.prompt_type == PromptType.CREATIVE_DENSITY:
            # Text timeline
            text_timeline = context.timelines.get('text_timeline', {})
            if text_timeline:
                lines.append("\nText Timeline:")
                for timestamp, texts in list(text_timeline.items())[:20]:  # Limit to 20
                    text_str = "; ".join(t['text'] for t in texts) if isinstance(texts, list) else str(texts)
                    lines.append(f"  {timestamp}: {text_str[:100]}")
            
            # Density buckets
            density = context.timelines.get('density_buckets', [])
            if density:
                lines.append(f"\nDensity Pattern: {density[:20]}")  # First 20 buckets
        
        elif context.prompt_type == PromptType.EMOTIONAL_JOURNEY:
            # Emotional timeline
            emotional = context.timelines.get('emotional_timeline', {})
            if emotional:
                lines.append("\nEmotional Timeline (sampled):")
                for timestamp, data in list(emotional.items())[:15]:  # Sample 15 points
                    events = data.get('events', [])
                    lines.append(f"  {timestamp}: {', '.join(events) if events else 'quiet'}")
            
            # Scene changes
            scene_changes = context.timelines.get('scene_changes', [])
            if scene_changes:
                lines.append(f"\nScene Changes: {scene_changes[:20]}")  # First 20
        
        elif context.prompt_type == PromptType.SPEECH_ANALYSIS:
            # Speech segments
            segments = context.timelines.get('speech_segments', [])
            if segments:
                lines.append("\nSpeech Segments:")
                for seg in segments[:10]:  # First 10 segments
                    lines.append(f"  {seg['start']:.1f}-{seg['end']:.1f}s: \"{seg['text'][:80]}...\"")
            
            # Speech gaps
            gaps = context.timelines.get('speech_gaps', [])
            if gaps:
                lines.append("\nNotable Speech Gaps:")
                for gap in gaps[:5]:  # Top 5 gaps
                    lines.append(f"  {gap['start']:.1f}-{gap['end']:.1f}s ({gap['duration']:.1f}s)")
        
        elif context.prompt_type == PromptType.VISUAL_OVERLAY:
            # Visual timeline (carefully limited)
            visual = context.timelines.get('visual_timeline', {})
            if visual:
                lines.append("\nVisual Elements Timeline (sampled):")
                sample_points = list(visual.items())[:30:3]  # Every 3rd point, max 10
                for timestamp, elements in sample_points:
                    text_count = len(elements.get('text', []))
                    object_count = len(elements.get('objects', []))
                    lines.append(f"  {timestamp}: {text_count} text, {object_count} objects")
        
        return "\n".join(lines)
    
    def _build_temporal_markers_section(self, context: PromptContext) -> str:
        """Build temporal markers section."""
        if not context.temporal_markers:
            return ""
        
        lines = ["TEMPORAL PATTERN DATA:"]
        
        # First 5 seconds analysis
        first_5 = context.temporal_markers.get('first_5_seconds', {})
        if first_5:
            lines.append("\nFirst 5 Seconds:")
            lines.append(f"- Density: {first_5.get('density_progression', [])}")
            lines.append(f"- Emotions: {first_5.get('emotion_sequence', [])}")
            
            text_moments = first_5.get('text_moments', [])
            if text_moments:
                lines.append(f"- Text Moments: {len(text_moments)}")
        
        # CTA window
        cta = context.temporal_markers.get('cta_window', {})
        if cta:
            lines.append(f"\nCTA Window ({cta.get('time_range', 'unknown')}):")
            
            cta_appearances = cta.get('cta_appearances', [])
            if cta_appearances:
                lines.append(f"- CTA Keywords Found: {len(cta_appearances)}")
                for appearance in cta_appearances[:3]:  # First 3
                    lines.append(f"  - {appearance['timestamp']}: \"{appearance['keyword']}\"")
        
        # Peak moments
        peaks = context.temporal_markers.get('peak_moments', [])
        if peaks:
            lines.append("\nPeak Engagement Moments:")
            for peak in peaks[:3]:  # Top 3
                lines.append(f"- {peak['timestamp']:.1f}s: intensity={peak['intensity']:.1f}")
        
        # Engagement curve
        curve = context.temporal_markers.get('engagement_curve', [])
        if curve:
            lines.append(f"\nEngagement Curve: {curve[:20]}")  # First 20 points
        
        return "\n".join(lines)