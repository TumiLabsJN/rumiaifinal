"""
Pydantic schemas for validation.

Source: FoundationCHILD.md Section 5: Configuration Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


class Config(BaseModel):
    """
    Configuration schema for RumiAI ML Pipeline.

    Source: FoundationCHILD.md Section 5.1: config.json Schema
    """
    client_id: str = Field(..., pattern=r"^[a-zA-Z0-9_]+$")
    analysis_type: str = Field(..., pattern=r"^(hashtag|competitor|creator)$")
    target: str
    analysis_mode: str = Field(..., pattern=r"^(top|recent)$")
    selection_strategy: str = Field(..., pattern=r"^(contrastive|top)$")
    video_count: int = Field(..., ge=10, le=500)
    date_filter: str = Field(..., pattern=r"^last_\d+_days$")
    report_type: str = Field(..., pattern=r"^(single|comparison)$")
    report_audience: str = Field(..., pattern=r"^(client|internal|creator)$")
    auto_confirm: bool
    run_date: str  # ISO 8601 format

    @field_validator("target")
    @classmethod
    def validate_target_format(cls, v: str, info) -> str:
        """
        Validate target format based on analysis_type.

        Updated to support cluster names for hashtag analysis.
        Source: HashtagVolumeV2_TI.md DECISION 1
        """
        import re
        analysis_type = info.data.get("analysis_type")
        if analysis_type == "hashtag":
            # Allow both cluster names and single hashtags (deprecated)
            # Cluster names: alphanumeric + underscore (e.g., "nutrition", "fitness_tips")
            # Single hashtags: start with # (e.g., "#nutrition") - deprecated in video_discovery.py
            if v.startswith("#"):
                # Single hashtag format (deprecated but valid for schema)
                if len(v) < 2:
                    raise ValueError("Hashtag target must have at least 2 characters")
            else:
                # Cluster name format - validate alphanumeric + underscore
                if not re.match(r"^[a-zA-Z0-9_]+$", v):
                    raise ValueError("Cluster name must be alphanumeric + underscore only (e.g., 'nutrition', 'fitness_tips')")
        elif analysis_type in ["competitor", "creator"]:
            if not v.startswith("@") or len(v) < 2:
                raise ValueError(f"{analysis_type.capitalize()} target must start with @ and have at least 2 characters")
        return v

    @field_validator("date_filter")
    @classmethod
    def validate_date_filter_range(cls, v: str) -> str:
        """Validate date filter days are in range 1-365."""
        days = int(v.split("_")[1])
        if not (1 <= days <= 365):
            raise ValueError(f"Date filter days must be between 1 and 365, got {days}")
        return v

    model_config = {"frozen": True}  # Immutable after creation


class ApifyVideoMetadata(BaseModel):
    """
    Apify video metadata schema.

    Source: FoundationCHILD.md Section 5.2: Apify Video Metadata Schema

    Note: This is a subset of fields. Apify returns 30+ fields.
    These 8 are required for Stage 1 processing.
    """
    id: str = Field(..., description="Unique video identifier")
    createTime: int = Field(..., description="Unix timestamp in UTC")
    duration: int = Field(..., description="Video length in seconds")
    playCount: int = Field(..., description="View count")
    shareCount: int = Field(..., description="Share count")
    commentCount: int = Field(..., description="Comment count")
    likeCount: int = Field(..., description="Like count")
    webVideoUrl: str = Field(..., description="TikTok web URL")

    model_config = {"extra": "allow"}  # Allow additional fields from Apify


class CheckpointFailedVideo(BaseModel):
    """Nested schema for failed video records."""
    video_id: str
    error: str
    timestamp: Optional[str] = None  # ISO timestamp
    stage: Optional[str] = None  # Substage that failed (e.g., "FEAT", "Whisper")


class CheckpointSchema(BaseModel):
    """
    Checkpoint schema for resumable processing.

    Source: FoundationCHILD.md Section 5.3: Checkpoint Schema
    """
    stage: str = Field(..., description="Stage name", examples=["video_processing"])
    bucket: str = Field(..., description="Bucket name", examples=["18-33s"])
    total_videos: int = Field(..., description="Total videos to process")
    completed: int = Field(..., description="Successfully processed count")
    failed: int = Field(..., description="Failed with errors count")
    remaining: int = Field(..., description="Not yet processed count")
    last_checkpoint: str = Field(..., description="ISO timestamp")
    completed_video_ids: List[str] = Field(default_factory=list)
    failed_video_ids: List[CheckpointFailedVideo] = Field(default_factory=list)
