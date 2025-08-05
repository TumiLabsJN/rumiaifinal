"""
Analysis models for RumiAI v2.

This module provides data structures for ML analysis results and unified analysis.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from .timeline import Timeline
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class MLAnalysisResult:
    """Results from a single ML model."""
    model_name: str  # "yolo", "whisper", "mediapipe", "ocr", "scene_detection"
    model_version: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0
    output_path: Optional[str] = None  # Path to saved results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'processing_time': self.processing_time,
            'output_path': self.output_path
        }


@dataclass
class UnifiedAnalysis:
    """
    Complete analysis results for a video.
    
    This is the central data structure that flows through the entire pipeline.
    """
    video_id: str
    video_metadata: Dict[str, Any]
    timeline: Timeline
    ml_results: Dict[str, MLAnalysisResult] = field(default_factory=dict)
    temporal_markers: Optional[Dict[str, Any]] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_ml_result(self, result: MLAnalysisResult) -> None:
        """Add ML analysis result."""
        self.ml_results[result.model_name] = result
        logger.info(f"Added {result.model_name} result (success={result.success})")
    
    def get_ml_data(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get data from specific ML model.
        
        Returns None if model hasn't run or failed.
        """
        if model_name in self.ml_results and self.ml_results[model_name].success:
            return self.ml_results[model_name].data
        return None
    
    def is_complete(self) -> bool:
        """Check if all required analyses are complete."""
        required_models = ['yolo', 'whisper', 'mediapipe', 'ocr', 'scene_detection']
        return all(
            model in self.ml_results and self.ml_results[model].success
            for model in required_models
        )
    
    def get_completion_status(self) -> Dict[str, bool]:
        """Get detailed completion status for each model."""
        required_models = ['yolo', 'whisper', 'mediapipe', 'ocr', 'scene_detection']
        return {
            model: (model in self.ml_results and self.ml_results[model].success)
            for model in required_models
        }
    
    def get_errors(self) -> Dict[str, str]:
        """Get all errors from failed analyses."""
        errors = {}
        for model_name, result in self.ml_results.items():
            if not result.success and result.error:
                errors[model_name] = result.error
        return errors
    
    def to_dict(self, legacy_mode: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Uses clean ml_data structure, no legacy fields by default.
        """
        # Build the unified analysis structure expected by prompts
        result = {
            'video_id': self.video_id,
            'metadata': self.video_metadata,
            'duration': self.timeline.duration,
            'pipeline_status': {
                'yolo': self.get_ml_data('yolo') is not None,
                'mediapipe': self.get_ml_data('mediapipe') is not None,
                'ocr': self.get_ml_data('ocr') is not None,
                'scene': self.get_ml_data('scene_detection') is not None,
                'whisper': self.get_ml_data('whisper') is not None,
                'temporal': self.temporal_markers is not None
            },
            'processing_metadata': self.processing_metadata
        }
        
        # Add temporal markers if present
        if self.temporal_markers:
            result['temporalMarkers'] = self.temporal_markers
        
        # Add timeline in various formats for backward compatibility
        result['timeline'] = self.timeline.to_dict(legacy_mode)
        
        # Add timeline sections for specific entry types
        result['speechTimeline'] = self.timeline.to_prompt_format('speech')
        result['textTimeline'] = self.timeline.to_prompt_format('text')
        result['objectTimeline'] = self.timeline.to_prompt_format('object')
        
        # Add ml_data field that precompute functions expect
        # This provides a clean, consistent interface for ML data access
        required_models = ['yolo', 'whisper', 'mediapipe', 'ocr', 'scene_detection']
        
        # Log any unexpected services (validation)
        for service in self.ml_results:
            if service not in required_models:
                logger.warning(f"Unexpected ML service in results: {service}")
        
        # Create ml_data with all expected services
        result['ml_data'] = {}
        for service in required_models:
            if service in self.ml_results and self.ml_results[service].success:
                result['ml_data'][service] = self.ml_results[service].data
            else:
                result['ml_data'][service] = {}
        
        return result
    
    def save_to_file(self, file_path: str, legacy_mode: bool = False) -> None:
        """Save unified analysis to JSON file."""
        import tempfile
        import shutil
        from pathlib import Path
        
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first (atomic write)
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=Path(file_path).parent,
            delete=False,
            suffix='.tmp'
        ) as tmp:
            json.dump(self.to_dict(legacy_mode), tmp, indent=2)
            tmp_path = tmp.name
        
        # Atomic move
        shutil.move(tmp_path, file_path)
        logger.info(f"Saved unified analysis to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'UnifiedAnalysis':
        """Load unified analysis from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct from saved data
        # This is a simplified version - full implementation would
        # properly reconstruct all nested objects
        video_id = data.get('video_id', '')
        video_metadata = data.get('metadata', {})
        duration = data.get('duration', 0)
        
        timeline = Timeline(video_id=video_id, duration=duration)
        analysis = cls(
            video_id=video_id,
            video_metadata=video_metadata,
            timeline=timeline
        )
        
        # Add ML results if present
        if 'objectDetection' in data:
            analysis.add_ml_result(MLAnalysisResult(
                model_name='yolo',
                model_version='v8',
                success=True,
                data=data['objectDetection']
            ))
        
        # Add other ML results similarly...
        
        if 'temporalMarkers' in data:
            analysis.temporal_markers = data['temporalMarkers']
        
        return analysis
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (f"UnifiedAnalysis(video_id={self.video_id}, "
                f"complete={self.is_complete()}, "
                f"models={list(self.ml_results.keys())})")