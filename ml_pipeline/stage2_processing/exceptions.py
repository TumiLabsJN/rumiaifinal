"""
Custom exceptions for Stage 2: Video Processing

Source: VideoProcessingTI.md Section 6: Error Handling
"""


class DownloadError(Exception):
    """
    Raised when video download fails after max retry attempts.

    Captures error context (video_id, attempts, original error).

    Source: VideoProcessingTI.md Section 6
    """
    def __init__(self, video_id: str, attempts: int, original_error: Exception):
        self.video_id = video_id
        self.attempts = attempts
        self.original_error = original_error
        super().__init__(
            f"Failed to download video {video_id} after {attempts} attempts: {original_error}"
        )


class ProcessingError(Exception):
    """
    Raised when RumiAI pipeline fails.

    Captures error context (video_id, stage, message).

    Source: VideoProcessingTI.md Section 6
    """
    def __init__(self, video_id: str, stage: str, message: str):
        self.video_id = video_id
        self.stage = stage
        self.message = message
        super().__init__(f"RumiAI processing failed for {video_id} at stage {stage}: {message}")


class ValidationError(Exception):
    """
    Raised when output schema validation fails.

    Captures error context (video_id, field, expected, actual).

    Source: VideoProcessingTI.md Section 6
    """
    def __init__(self, video_id: str, field: str, expected: str, actual: str):
        self.video_id = video_id
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Schema validation failed for {video_id}: "
            f"field '{field}' expected {expected}, got {actual}"
        )


class CheckpointCorruptionError(Exception):
    """
    Raised when both checkpoint and backup are corrupted.

    Provides recovery instructions.

    Source: VideoProcessingTI.md Section 6
    """
    def __init__(self, checkpoint_path: str, backup_path: str, original_error: Exception):
        self.checkpoint_path = checkpoint_path
        self.backup_path = backup_path
        self.original_error = original_error

        recovery_msg = (
            f"Checkpoint and backup both corrupted.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Backup: {backup_path}\n"
            f"Original error: {original_error}\n\n"
            f"Recovery options:\n"
            f"  1. Use --force flag to discard checkpoint and restart\n"
            f"  2. Manually inspect checkpoint files for partial recovery\n"
            f"  3. Contact support if data recovery is critical"
        )
        super().__init__(recovery_msg)
