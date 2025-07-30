"""
Configuration settings for RumiAI v2.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class Settings:
    """
    Central configuration for RumiAI v2.
    
    Loads from environment variables and config files.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config")
        
        # API Keys
        self.claude_api_key = os.getenv('CLAUDE_API_KEY', '')
        self.apify_token = os.getenv('APIFY_API_TOKEN', '')
        
        # Model settings
        default_model = 'claude-3-haiku-20240307'
        if os.getenv('USE_CLAUDE_SONNET', 'false').lower() == 'true':
            default_model = 'claude-3-5-sonnet-20241022'
        self.claude_model = os.getenv('CLAUDE_MODEL', default_model)
        
        # Paths
        self.output_dir = Path(os.getenv('RUMIAI_OUTPUT_DIR', 'outputs'))
        self.temp_dir = Path(os.getenv('RUMIAI_TEMP_DIR', 'temp'))
        self.unified_dir = Path(os.getenv('RUMIAI_UNIFIED_DIR', 'unified_analysis'))
        self.insights_dir = Path(os.getenv('RUMIAI_INSIGHTS_DIR', 'insights'))
        self.temporal_dir = Path(os.getenv('RUMIAI_TEMPORAL_DIR', 'temporal_markers'))
        
        # Processing settings
        self.max_video_duration = int(os.getenv('RUMIAI_MAX_VIDEO_DURATION', '300'))  # 5 minutes
        self.frame_sample_rate = float(os.getenv('RUMIAI_FRAME_SAMPLE_RATE', '1.0'))  # 1 fps
        self.prompt_delay = int(os.getenv('RUMIAI_PROMPT_DELAY', '10'))  # seconds between prompts
        
        # Prompt timeouts (seconds)
        self.prompt_timeouts = {
            'creative_density': 60,
            'emotional_journey': 90,
            'speech_analysis': 90,
            'visual_overlay_analysis': 120,  # Larger timeout for problematic prompt
            'metadata_analysis': 60,
            'person_framing': 60,
            'scene_pacing': 60
        }
        
        # Feature flags
        self.temporal_markers_enabled = os.getenv('RUMIAI_TEMPORAL_MARKERS', 'true').lower() == 'true'
        self.strict_mode = os.getenv('RUMIAI_STRICT_MODE', 'false').lower() == 'true'
        self.cleanup_video = os.getenv('RUMIAI_CLEANUP_VIDEO', 'false').lower() == 'true'
        
        # ML Enhancement Feature Flags
        self.use_ml_precompute = os.getenv('USE_ML_PRECOMPUTE', 'false').lower() == 'true'
        self.use_claude_sonnet = os.getenv('USE_CLAUDE_SONNET', 'false').lower() == 'true'
        self.output_format_version = os.getenv('OUTPUT_FORMAT_VERSION', 'v1')  # v1 (legacy) or v2 (6-block)
        
        # Precompute settings - control which prompts use precompute
        self.precompute_enabled_prompts = {
            'creative_density': os.getenv('PRECOMPUTE_CREATIVE_DENSITY', 'false').lower() == 'true',
            'emotional_journey': os.getenv('PRECOMPUTE_EMOTIONAL_JOURNEY', 'false').lower() == 'true',
            'person_framing': os.getenv('PRECOMPUTE_PERSON_FRAMING', 'false').lower() == 'true',
            'scene_pacing': os.getenv('PRECOMPUTE_SCENE_PACING', 'false').lower() == 'true',
            'speech_analysis': os.getenv('PRECOMPUTE_SPEECH_ANALYSIS', 'false').lower() == 'true',
            'visual_overlay_analysis': os.getenv('PRECOMPUTE_VISUAL_OVERLAY', 'false').lower() == 'true',
            'metadata_analysis': os.getenv('PRECOMPUTE_METADATA', 'false').lower() == 'true'
        }
        
        # Cost control settings
        self.claude_cost_threshold = float(os.getenv('CLAUDE_COST_THRESHOLD', '0.10'))  # Alert if single prompt > $0.10
        self.enable_cost_monitoring = os.getenv('ENABLE_COST_MONITORING', 'true').lower() == 'true'
        
        # Load prompt templates
        self._prompt_templates = self._load_prompt_templates()
        
        # Validate configuration
        self._validate_config()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from files or defaults."""
        templates = {}
        
        # Try to load from config directory
        prompts_file = self.config_dir / "prompts.json"
        if prompts_file.exists():
            try:
                with open(prompts_file, 'r') as f:
                    templates = json.load(f)
                logger.info(f"Loaded {len(templates)} prompt templates from {prompts_file}")
            except Exception as e:
                logger.error(f"Failed to load prompt templates: {e}")
        
        # Add default templates for any missing
        defaults = {
            'creative_density': """Analyze the creative density and visual complexity of this TikTok video.
Focus on:
1. Text overlay frequency and positioning
2. Visual effects and transitions
3. Information density over time
4. Creative element patterns

Provide insights on how the creative elements contribute to viewer engagement.""",
            
            'emotional_journey': """Analyze the emotional journey and narrative arc of this TikTok video.
Focus on:
1. Emotional progression throughout the video
2. Key emotional peaks and valleys
3. How visuals, speech, and music create emotional impact
4. Viewer emotional engagement patterns

Provide insights on the emotional storytelling techniques used.""",
            
            'speech_analysis': """Analyze the speech patterns and verbal content of this TikTok video.
Focus on:
1. Speaking pace and rhythm
2. Key topics and themes
3. Verbal hooks and memorable phrases
4. Speech-to-action synchronization

Provide insights on how speech contributes to the video's effectiveness.""",
            
            'visual_overlay_analysis': """Analyze the visual overlay strategy and text placement in this TikTok video.
Focus on:
1. Text timing and duration
2. Visual hierarchy and readability
3. Text-to-action coordination
4. Information delivery patterns

Provide insights on the visual communication strategy.""",
            
            'metadata_analysis': """Analyze how the video's metadata (caption, hashtags) aligns with its content.
Focus on:
1. Hashtag relevance to content
2. Caption effectiveness
3. SEO optimization
4. Discoverability factors

Provide insights on metadata optimization opportunities.""",
            
            'person_framing': """Analyze the person framing and human presence in this TikTok video.
Focus on:
1. Screen time and positioning
2. Eye contact and engagement
3. Body language and gestures
4. Person-to-content balance

Provide insights on how human presence affects viewer connection.""",
            
            'scene_pacing': """Analyze the scene pacing and visual rhythm of this TikTok video.
Focus on:
1. Cut frequency and timing
2. Scene duration patterns
3. Visual flow and transitions
4. Pacing impact on retention

Provide insights on the video's editing rhythm and viewer attention management."""
        }
        
        # Merge with defaults
        for prompt_type, template in defaults.items():
            if prompt_type not in templates:
                templates[prompt_type] = template
        
        return templates
    
    def get_prompt_template(self, prompt_type: str) -> str:
        """Get prompt template for a specific type."""
        return self._prompt_templates.get(prompt_type, "Analyze this video.")
    
    def _validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Check required API keys
        if not self.claude_api_key:
            errors.append("CLAUDE_API_KEY environment variable not set")
        
        if not self.apify_token:
            errors.append("APIFY_API_TOKEN environment variable not set")
        
        # Check paths are writable
        for path_name, path in [
            ('output_dir', self.output_dir),
            ('temp_dir', self.temp_dir),
            ('unified_dir', self.unified_dir),
            ('insights_dir', self.insights_dir),
            ('temporal_dir', self.temporal_dir)
        ]:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create {path_name} at {path}: {e}")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            
            if self.strict_mode:
                raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'claude_model': self.claude_model,
            'output_dir': str(self.output_dir),
            'temp_dir': str(self.temp_dir),
            'max_video_duration': self.max_video_duration,
            'frame_sample_rate': self.frame_sample_rate,
            'prompt_delay': self.prompt_delay,
            'temporal_markers_enabled': self.temporal_markers_enabled,
            'strict_mode': self.strict_mode,
            'cleanup_video': self.cleanup_video,
            'use_ml_precompute': self.use_ml_precompute,
            'use_claude_sonnet': self.use_claude_sonnet,
            'output_format_version': self.output_format_version,
            'precompute_enabled_prompts': self.precompute_enabled_prompts,
            'enable_cost_monitoring': self.enable_cost_monitoring
        }