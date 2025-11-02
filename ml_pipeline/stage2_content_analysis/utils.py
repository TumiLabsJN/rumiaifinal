"""
Shared Utilities for Content Analysis Stage (2.6 & 2.7)

Source: ContentAnalysisCHILDTI.md Section 8.2
"""

import json
import os
import re
import logging
from typing import Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_llm_json(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling markdown code fences.

    LLM APIs (Claude, GPT, etc.) often return JSON wrapped in markdown code fences.
    This utility strips those fences before parsing to prevent JSONDecodeError.

    Handles:
    - ```json { ... } ```  (common Claude format)
    - ``` { ... } ```       (generic markdown)
    - { ... }               (raw JSON - pass through)

    Args:
        response_text: Raw text from LLM API response

    Returns:
        dict: Parsed JSON object

    Raises:
        json.JSONDecodeError: If text is not valid JSON after cleaning

    Examples:
        >>> response = '```json\\n{"key": "value"}\\n```'
        >>> data = parse_llm_json(response)
        >>> print(data)
        {'key': 'value'}

        >>> response = '{"key": "value"}'
        >>> data = parse_llm_json(response)
        >>> print(data)
        {'key': 'value'}
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r'^```(?:json)?\s*', '', text)
        # Remove closing fence
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

    # Parse and return JSON
    return json.loads(text)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with error handling.

    Source: ContentAnalysisCHILDTI.md Section 8.2 (utils.py responsibilities)

    Args:
        file_path: Path to JSON file

    Returns:
        dict: Loaded JSON data

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file contains invalid JSON

    Example:
        >>> data = load_json("/data/clients/acme/hashtags/nutrition/selection_manifest.json")
        >>> print(data['hashtag'])
        'nutrition'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


def save_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """
    Save JSON file with atomic writes.

    Source: ContentAnalysisCHILDTI.md Section 8.2 (utils.py responsibilities)

    Uses atomic write pattern: write to temporary file, then rename to ensure
    no partial writes if process crashes mid-save.

    Args:
        file_path: Path to save JSON file
        data: Dictionary to save as JSON
        indent: JSON indentation (default: 2 spaces)

    Raises:
        OSError: If file cannot be written (permissions, disk full, etc.)

    Example:
        >>> taxonomy = {"hashtag": "nutrition", "patterns": []}
        >>> save_json("/data/clients/acme/hashtags/nutrition/taxonomy.json", taxonomy)
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Atomic write pattern: write to temp file, then rename
    temp_path = f"{file_path}.tmp"

    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        # Atomic rename (replaces existing file if present)
        os.replace(temp_path, file_path)
        logger.debug(f"Saved JSON to {file_path}")

    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for content analysis modules.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting content analysis")
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Format: [TIMESTAMP] [LEVEL] [MODULE] Message
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def extract_transcript_ending(text: str, max_words: int = 10) -> str:
    """
    Extract last N words from transcript for closing strategy analysis.

    HARDENED against edge cases:
    - Ellipsis handling (strips trailing punctuation)
    - Empty/None transcripts (returns "")
    - No punctuation (splits by whitespace only)
    - Multiple whitespace/newlines (normalizes with split())

    Args:
        text: Full transcript text from Whisper
        max_words: Number of words to extract from end (default: 10)

    Returns:
        str: Last N words, or full text if shorter, or empty string if no text

    Examples:
        >>> extract_transcript_ending("Text here...")
        "Text here"

        >>> extract_transcript_ending("Word " * 5)
        "Word Word Word Word Word"

        >>> extract_transcript_ending("no punctuation here")
        "no punctuation here"

        >>> extract_transcript_ending("text  \n\n  more")
        "text more"
    """
    # DEFENSE 1: Handle empty/None input
    if not text or not text.strip():
        return ""

    # DEFENSE 4: Normalize whitespace (handles multiple spaces, newlines, tabs)
    # .split() with no args splits on ANY whitespace and removes empty strings
    text = text.strip()
    words = text.split()  # Handles "text  \n\n  more" → ["text", "more"]

    if not words:
        return ""

    # DEFENSE 2: Handle transcripts shorter than max_words
    if len(words) <= max_words:
        # Return all words joined (already normalized)
        return " ".join(words)

    # DEFENSE 3: Extract last N words (no regex needed - whitespace already handled)
    ending_words = words[-max_words:]
    ending_text = " ".join(ending_words)

    # DEFENSE 1 (Ellipsis): Strip trailing punctuation (., !, ?, ...)
    # This handles "... text here..." → "text here"
    ending_text = ending_text.rstrip('.!?…')  # Note: … is single char ellipsis

    return ending_text.strip()


def extract_transcript_opening(text: str, max_words: int = 10) -> str:
    """
    Extract first N words from transcript for hook strategy analysis.

    COMPANION FUNCTION to extract_transcript_ending for symmetry.
    Uses same hardened approach (whitespace normalization, no regex).

    Args:
        text: Full transcript text from Whisper
        max_words: Number of words to extract from beginning (default: 10)

    Returns:
        str: First N words, or full text if shorter, or empty string if no text

    Examples:
        >>> extract_transcript_opening("This is why you need to try this. It changed my life.", max_words=10)
        "This is why you need to try this"

        >>> extract_transcript_opening("Short text", max_words=10)
        "Short text"
    """
    # Handle empty or None input
    if not text or not text.strip():
        return ""

    # Normalize whitespace (handles multiple spaces, newlines, tabs)
    text = text.strip()
    words = text.split()  # Split on any whitespace

    if not words:
        return ""

    # Handle transcripts shorter than max_words
    if len(words) <= max_words:
        return " ".join(words)

    # Extract first N words
    opening_words = words[:max_words]
    return " ".join(opening_words)
