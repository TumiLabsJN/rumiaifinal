"""
Custom exceptions for RumiAI v2.

CRITICAL: These exceptions must provide clear error messages for debugging
without breaking the automated pipeline.
"""
from typing import Any, Optional, Dict
import os


# Check if we're in strict mode (fail fast) or production mode (graceful degradation)
STRICT_MODE = os.getenv('RUMIAI_STRICT_MODE', 'false').lower() == 'true'


class RumiAIError(Exception):
    """Base exception for all RumiAI errors."""
    
    def __init__(self, message: str, video_id: Optional[str] = None):
        self.video_id = video_id
        if video_id:
            message = f"[Video {video_id}] {message}"
        super().__init__(message)


class ValidationError(RumiAIError):
    """Data validation error."""
    
    def __init__(self, field: str, value: Any, expected: str, video_id: Optional[str] = None):
        self.field = field
        self.value = value
        self.expected = expected
        message = f"Invalid {field}: got {type(value).__name__}={repr(value)}, expected {expected}"
        super().__init__(message, video_id)


class TimelineError(RumiAIError):
    """Timeline-related error."""
    
    def __init__(self, message: str, video_id: Optional[str] = None, details: Optional[Dict] = None):
        self.details = details or {}
        if details:
            message = f"{message} | Details: {details}"
        super().__init__(message, video_id)


class MLAnalysisError(RumiAIError):
    """ML analysis error."""
    
    def __init__(self, model: str, reason: str, video_id: Optional[str] = None):
        self.model = model
        self.reason = reason
        message = f"ML analysis failed for {model}: {reason}"
        super().__init__(message, video_id)


class PromptError(RumiAIError):
    """Prompt processing error."""
    
    def __init__(self, prompt_type: str, reason: str, video_id: Optional[str] = None):
        self.prompt_type = prompt_type
        self.reason = reason
        message = f"Prompt '{prompt_type}' failed: {reason}"
        super().__init__(message, video_id)


class APIError(RumiAIError):
    """External API error."""
    
    def __init__(self, service: str, status_code: int, message: str, video_id: Optional[str] = None):
        self.service = service
        self.status_code = status_code
        error_message = f"{service} API error ({status_code}): {message}"
        super().__init__(error_message, video_id)


class FileSystemError(RumiAIError):
    """File system operation error."""
    
    def __init__(self, operation: str, path: str, reason: str, video_id: Optional[str] = None):
        self.operation = operation
        self.path = path
        self.reason = reason
        message = f"File {operation} failed for '{path}': {reason}"
        super().__init__(message, video_id)


class ConfigurationError(RumiAIError):
    """Configuration error."""
    
    def __init__(self, config_key: str, reason: str):
        self.config_key = config_key
        self.reason = reason
        message = f"Configuration error for '{config_key}': {reason}"
        super().__init__(message)


def handle_error(error: Exception, logger, default_return: Any = None) -> Any:
    """
    Central error handling logic.
    
    In STRICT_MODE: Re-raise the error (fail fast)
    In Production: Log and return default value (graceful degradation)
    """
    if STRICT_MODE:
        raise error
    else:
        logger.error(f"{type(error).__name__}: {str(error)}", exc_info=True)
        return default_return