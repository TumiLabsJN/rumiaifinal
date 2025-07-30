"""
Prompt manager for loading and formatting ML analysis prompts.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt templates for ML analysis"""
    
    def __init__(self, template_dir: str = "prompt_templates"):
        """
        Initialize prompt manager.
        
        Args:
            template_dir: Directory containing prompt template files
        """
        self.template_dir = Path(template_dir)
        self.templates = {}
        self._load_all_templates()
    
    def _load_all_templates(self):
        """Load all prompt templates from disk"""
        prompt_types = [
            'creative_density',
            'emotional_journey', 
            'person_framing',
            'scene_pacing',
            'speech_analysis',
            'visual_overlay_analysis',
            'metadata_analysis'
        ]
        
        for prompt_type in prompt_types:
            template_file = self.template_dir / f"{prompt_type}_v2.txt"
            
            if template_file.exists():
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        self.templates[prompt_type] = f.read()
                    logger.info(f"Loaded template for {prompt_type}")
                except Exception as e:
                    logger.error(f"Failed to load template {template_file}: {e}")
                    self.templates[prompt_type] = self._get_fallback_template(prompt_type)
            else:
                logger.warning(f"Template file not found: {template_file}")
                self.templates[prompt_type] = self._get_fallback_template(prompt_type)
    
    def get_prompt(self, prompt_type: str) -> str:
        """
        Get the raw prompt template for a specific type.
        
        Args:
            prompt_type: Type of prompt to retrieve
            
        Returns:
            Raw prompt template string
        """
        return self.templates.get(prompt_type, self._get_fallback_template(prompt_type))
    
    def format_prompt(self, prompt_type: str, context: Dict[str, Any]) -> str:
        """
        Format a prompt template with context data.
        
        Args:
            prompt_type: Type of prompt to format
            context: Dictionary containing precomputed metrics and other data
            
        Returns:
            Formatted prompt ready for Claude API
        """
        template = self.get_prompt(prompt_type)
        
        # Build the context section
        context_parts = []
        
        # Add precomputed metrics if available
        if 'precomputed_metrics' in context:
            context_parts.append("## Precomputed Metrics:")
            context_parts.append(json.dumps(context['precomputed_metrics'], indent=2))
            context_parts.append("")
        
        # Add video duration if available
        if 'video_duration' in context:
            context_parts.append(f"## Video Duration: {context['video_duration']} seconds")
            context_parts.append("")
        
        # Add any additional context
        for key, value in context.items():
            if key not in ['precomputed_metrics', 'video_duration']:
                context_parts.append(f"## {key.replace('_', ' ').title()}:")
                if isinstance(value, (dict, list)):
                    context_parts.append(json.dumps(value, indent=2))
                else:
                    context_parts.append(str(value))
                context_parts.append("")
        
        # Combine template with context
        full_prompt = template + "\n\n"
        if context_parts:
            full_prompt += "### Context Data:\n\n"
            full_prompt += "\n".join(context_parts)
        
        return full_prompt
    
    def _get_fallback_template(self, prompt_type: str) -> str:
        """Get a fallback template if the file is missing"""
        fallback_templates = {
            'creative_density': """Analyze the creative density and visual complexity of this video.
Focus on element distribution, pacing patterns, and cognitive load.
Output structured data in 6 blocks: CoreMetrics, Dynamics, Interactions, KeyEvents, Patterns, Quality.""",
            
            'emotional_journey': """Analyze the emotional journey and narrative arc of this video.
Focus on emotion progression, transitions, and multimodal coherence.
Output structured data in 6 blocks: CoreMetrics, Dynamics, Interactions, KeyEvents, Patterns, Quality.""",
            
            'person_framing': """Analyze the person framing and human presence in this video.
Focus on positioning, visibility, and engagement techniques.
Output structured data in 6 blocks: CoreMetrics, Dynamics, Interactions, KeyEvents, Patterns, Quality.""",
            
            'scene_pacing': """Analyze the scene pacing and visual rhythm of this video.
Focus on cut frequency, duration patterns, and editing style.
Output structured data in 6 blocks: CoreMetrics, Dynamics, Interactions, KeyEvents, Patterns, Quality.""",
            
            'speech_analysis': """Analyze the speech patterns and verbal content of this video.
Focus on pacing, clarity, and speech-gesture synchronization.
Output structured data in 6 blocks: CoreMetrics, Dynamics, Interactions, KeyEvents, Patterns, Quality.""",
            
            'visual_overlay': """Analyze the visual overlay strategy and text placement in this video.
Focus on timing, density, and cross-modal alignment.
Output structured data in 6 blocks: CoreMetrics, Dynamics, Interactions, KeyEvents, Patterns, Quality.""",
            
            'metadata_analysis': """Analyze how the video's metadata aligns with its content.
Focus on caption effectiveness, hashtag relevance, and engagement patterns.
Output structured data in 6 blocks: CoreMetrics, Dynamics, Interactions, KeyEvents, Patterns, Quality."""
        }
        
        return fallback_templates.get(prompt_type, "Analyze this video and provide structured insights.")
    
    def validate_prompt_size(self, prompt: str) -> tuple[bool, int]:
        """
        Validate that a prompt isn't too large.
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            Tuple of (is_valid, size_in_kb)
        """
        size_bytes = len(prompt.encode('utf-8'))
        size_kb = size_bytes / 1024
        
        # Warn if over 200KB (matching JS implementation)
        if size_kb > 200:
            logger.warning(f"Large prompt size: {size_kb:.1f}KB")
        
        # Consider invalid if over 1MB
        is_valid = size_kb < 1024
        
        return is_valid, int(size_kb)