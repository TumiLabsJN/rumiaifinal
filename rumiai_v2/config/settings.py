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
        self.apify_token = os.getenv('APIFY_API_TOKEN', '')
        
        # Model settings
        default_model = 'claude-3-haiku-20240307'
            default_model = 'claude-3-5-sonnet-20241022'
        
        # Paths
        self.output_dir = Path(os.getenv('RUMIAI_OUTPUT_DIR', 'outputs'))
        self.temp_dir = Path(os.getenv('RUMIAI_TEMP_DIR', 'temp'))
        self.unified_dir = Path(os.getenv('RUMIAI_UNIFIED_DIR', 'unified_analysis'))
        self.insights_dir = Path(os.getenv('RUMIAI_INSIGHTS_DIR', 'insights'))
        self.temporal_dir = Path(os.getenv('RUMIAI_TEMPORAL_DIR', 'temporal_markers'))
        
        # Processing settings
        self.max_video_duration = int(os.getenv('RUMIAI_MAX_VIDEO_DURATION', '300'))  # 5 minutes
        self.frame_sample_rate = float(os.getenv('RUMIAI_FRAME_SAMPLE_RATE', '1.0'))  # 1 fps
        
        # Prompt timeouts (seconds)
            'creative_density': 60,
            'emotional_journey': 90,
            'speech_analysis': 90,
            'visual_overlay_analysis': 120,  # Larger timeout for problematic prompt
            'metadata_analysis': 60,
            'person_framing': 60,
            'scene_pacing': 60
        }
        
        # Feature flags
        self.temporal_markers_enabled = True  # HARDCODED
        self.strict_mode = os.getenv('RUMIAI_STRICT_MODE', 'false').lower() == 'true'
        self.cleanup_video = os.getenv('RUMIAI_CLEANUP_VIDEO', 'false').lower() == 'true'
        self.use_python_only_processing = True  # HARDCODED
        
        # ML Enhancement Feature Flags
        self.use_ml_precompute = True  # HARDCODED
        self.output_format_version = "v2"  # HARDCODED
        
        # Precompute settings - control which prompts use precompute
        self.precompute_enabled_prompts = {
            'creative_density': True,  # HARDCODED
            'emotional_journey': True,  # HARDCODED
            'person_framing': True,  # HARDCODED
            'scene_pacing': True,  # HARDCODED
            'speech_analysis': True,  # HARDCODED
            'visual_overlay_analysis': True,  # HARDCODED
            'metadata_analysis': True  # HARDCODED
        }
        
        # Cost control settings
        self.enable_cost_monitoring = True  # HARDCODED
        
                return self._prompt_templates.get(prompt_type, "Analyze this video.")
    
    def _validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Check required API keys
        
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
            'output_dir': str(self.output_dir),
            'temp_dir': str(self.temp_dir),
            'max_video_duration': self.max_video_duration,
            'frame_sample_rate': self.frame_sample_rate,
            'temporal_markers_enabled': self.temporal_markers_enabled,
            'strict_mode': self.strict_mode,
            'cleanup_video': self.cleanup_video,
            'use_ml_precompute': self.use_ml_precompute,
            'use_claude_sonnet': self.use_claude_sonnet,
            'output_format_version': self.output_format_version,
            'precompute_enabled_prompts': self.precompute_enabled_prompts,
            'enable_cost_monitoring': self.enable_cost_monitoring
        }