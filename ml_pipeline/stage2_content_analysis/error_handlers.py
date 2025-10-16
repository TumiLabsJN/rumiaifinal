"""
Error Handling Utilities for Content Analysis Stage (2.6 & 2.7)

Source: ContentAnalysisCHILDTI.md Section 6
"""

import time
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


# ===== ERROR HANDLER: MISSING INPUT FILE =====
# Error ID: E1
# Source: ContentAnalysisCHILD.md Section 6.2 row 1

def handle_missing_input_file(file_path: str, stage_name: str):
    """
    Handle missing input file error.

    Source: ContentAnalysisCHILDTI.md Section 6.2

    Args:
        file_path: Path to missing file
        stage_name: Name of stage that should have created the file

    Raises:
        FileNotFoundError: Always (fail-fast strategy)
    """
    raise FileNotFoundError(
        f"❌ Required input not found: {file_path}\n"
        f"This file should have been created by {stage_name}.\n"
        f"Action: Verify {stage_name} completed successfully."
    )


# ===== ERROR HANDLER: API TIMEOUT WITH RETRY =====
# Error IDs: E5, E6
# Source: ContentAnalysisCHILD.md Section 6.2 rows 5, 6

def handle_api_timeout_with_retry(
    api_call_func: Callable[[], Any],
    context: str,
    max_retries: int = 3,
    backoff_delays: list = None
) -> Any:
    """
    Handle API timeout with exponential backoff retry.

    Source: ContentAnalysisCHILDTI.md Section 6.2

    Args:
        api_call_func: Function to call (must raise TimeoutError on failure)
        context: Description for logging (e.g., "Discovery", "Video 123 classification")
        max_retries: Number of retry attempts (default: 3)
        backoff_delays: Delay in seconds between retries (default: [1, 2, 4])

    Returns:
        Result from api_call_func

    Raises:
        TimeoutError: After all retries exhausted
    """
    if backoff_delays is None:
        backoff_delays = [1, 2, 4]

    for attempt in range(max_retries):
        try:
            return api_call_func()
        except TimeoutError as e:
            if attempt < max_retries - 1:
                delay = backoff_delays[attempt]
                logger.warning(
                    f"⏰ {context} timeout. Retry {attempt + 1}/{max_retries} in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"❌ {context} failed after {max_retries} retries.\n"
                    f"Action: Check status.anthropic.com and retry manually."
                )
                raise

    raise RuntimeError("Unreachable: retry loop exited unexpectedly")


# ===== ERROR HANDLER: INVALID JSON RESPONSE =====
# Error ID: E7
# Source: ContentAnalysisCHILD.md Section 6.2 row 7

def handle_invalid_json_response(
    response_text: str,
    context: str,
    max_retries: int = 3
):
    """
    Handle invalid JSON response from LLM.

    Source: ContentAnalysisCHILDTI.md Section 6.2

    Note: This is called WITHIN a retry loop, not a standalone handler.

    Args:
        response_text: Raw response from LLM
        context: Description of what was being processed
        max_retries: Number of retries configured

    Raises:
        ValueError: Always (to trigger retry logic in caller)
    """
    # Log raw response for debugging
    logger.error(
        f"⚠️ Invalid JSON from LLM ({context}).\n"
        f"Raw response (first 500 chars): {response_text[:500]}\n"
        f"Action: Check prompt formatting, report to Anthropic if recurring."
    )

    # Re-raise to trigger retry logic in caller
    raise ValueError(f"LLM returned invalid JSON for {context}")


# ===== ERROR HANDLER: GRACEFUL SKIP =====
# Error ID: E11
# Source: ContentAnalysisCHILD.md Section 6.2 row 11

def handle_graceful_skip(video_id: str, reason: str, error_type: str = "warning"):
    """
    Handle non-fatal errors by skipping video and logging.

    Source: ContentAnalysisCHILDTI.md Section 6.2

    Args:
        video_id: Video identifier
        reason: Why video is being skipped
        error_type: "warning" or "info" (determines log level)
    """
    if error_type == "warning":
        logger.warning(f"⚠️  Skipping video {video_id}: {reason}")
    else:
        logger.info(f"ℹ️  Skipping video {video_id}: {reason}")


# ===== ERROR HANDLER: API ERROR WITH RATE LIMITING =====
# Helper for handling Anthropic API rate limit errors

def handle_api_rate_limit(
    api_call_func: Callable[[], Any],
    context: str,
    max_retries: int = 5,
    initial_backoff: float = 1.0
) -> Any:
    """
    Handle API rate limit errors with exponential backoff.

    Source: Best practice for Anthropic API rate limits

    Args:
        api_call_func: Function to call
        context: Description for logging
        max_retries: Number of retry attempts (default: 5)
        initial_backoff: Initial backoff delay in seconds (default: 1.0)

    Returns:
        Result from api_call_func

    Raises:
        Exception: Original exception after all retries exhausted
    """
    backoff = initial_backoff

    for attempt in range(max_retries):
        try:
            return api_call_func()
        except Exception as e:
            # Check if it's a rate limit error
            error_str = str(e).lower()
            if 'rate' in error_str or '429' in error_str:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠️ Rate limit hit for {context}. "
                        f"Retry {attempt + 1}/{max_retries} in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"❌ Rate limit error for {context} after {max_retries} retries."
                    )
                    raise
            else:
                # Not a rate limit error, raise immediately
                raise

    raise RuntimeError("Unreachable: retry loop exited unexpectedly")
