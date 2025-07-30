"""
Prompt-related models for RumiAI v2.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime


class PromptType(Enum):
    """Enumeration of all Claude prompt types."""
    CREATIVE_DENSITY = "creative_density"
    EMOTIONAL_JOURNEY = "emotional_journey"
    SPEECH_ANALYSIS = "speech_analysis"
    VISUAL_OVERLAY = "visual_overlay_analysis"
    METADATA_ANALYSIS = "metadata_analysis"
    PERSON_FRAMING = "person_framing"
    SCENE_PACING = "scene_pacing"


@dataclass
class PromptContext:
    """Context data for a Claude prompt."""
    video_id: str
    prompt_type: PromptType
    duration: float
    metadata: Dict[str, Any]
    ml_data: Dict[str, Any] = field(default_factory=dict)
    timelines: Dict[str, Any] = field(default_factory=dict)
    temporal_markers: Optional[Dict[str, Any]] = None
    
    def get_size_bytes(self) -> int:
        """Estimate size of context data."""
        import json
        # Create a serializable version of the data
        data = {
            'video_id': self.video_id,
            'prompt_type': self.prompt_type.value,  # Use .value for enum
            'duration': self.duration,
            'metadata': self.metadata,
            'ml_data': self.ml_data,
            'timelines': self.timelines,
            'temporal_markers': self.temporal_markers
        }
        return len(json.dumps(data))


@dataclass
class PromptResult:
    """Result from a Claude prompt."""
    prompt_type: PromptType
    success: bool
    response: str = ""
    error: Optional[str] = None
    processing_time: float = 0.0
    tokens_used: int = 0
    estimated_cost: float = 0.0
    retry_attempts: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'prompt_type': self.prompt_type.value,
            'success': self.success,
            'response': self.response,
            'error': self.error,
            'processing_time': self.processing_time,
            'tokens_used': self.tokens_used,
            'estimated_cost': self.estimated_cost,
            'retry_attempts': self.retry_attempts,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PromptBatch:
    """Batch of prompts to process for a video."""
    video_id: str
    prompts: List[PromptType]
    results: Dict[str, PromptResult] = field(default_factory=dict)
    
    def add_result(self, result: PromptResult) -> None:
        """Add a prompt result."""
        self.results[result.prompt_type.value] = result
    
    def is_complete(self) -> bool:
        """Check if all prompts have been processed."""
        return len(self.results) == len(self.prompts)
    
    def get_success_rate(self) -> float:
        """Get percentage of successful prompts."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results.values() if r.success)
        return successful / len(self.results)
    
    def get_total_cost(self) -> float:
        """Get total estimated cost of all prompts."""
        return sum(r.estimated_cost for r in self.results.values())
    
    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return sum(r.tokens_used for r in self.results.values())