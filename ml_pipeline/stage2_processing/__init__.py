"""
Stage 2: Video Processing

Processes videos through RumiAI pipeline with checkpoint/resume capability.

Source: VideoProcessingTI.md
"""

from ml_pipeline.stage2_processing.main import stage_2_video_processing_main
from ml_pipeline.stage2_processing.exceptions import (
    DownloadError,
    ProcessingError,
    ValidationError,
    CheckpointCorruptionError
)

__all__ = [
    "stage_2_video_processing_main",
    "DownloadError",
    "ProcessingError",
    "ValidationError",
    "CheckpointCorruptionError"
]
