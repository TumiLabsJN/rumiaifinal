"""
Utility functions for Stage 2: Video Processing

Source: VideoProcessingTI.md Section 2.3.1 (Utility Functions)
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_bucket_path(config: dict, bucket_name: str) -> str:
    """
    Construct full bucket directory path from config and bucket name.

    Naming convention:
    - bucket_name: Duration range only (e.g., "18-33s")
    - bucket_path: Full directory path (e.g., "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/")

    All functions use bucket_name as parameter and construct full paths using this helper.

    Args:
        config: dict, loaded from config.json (FoundationCHILD Section 5.1)
        bucket_name: str, duration range (e.g., "18-33s")

    Returns:
        str: Full bucket directory path with trailing slash

    Source: VideoProcessingTI.md Section 4 (Helper Function)
    """
    data_root = os.getenv('DATA_ROOT', '/data')

    # Construct analysis_base path
    analysis_base = (
        f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/"
        f"{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"
    )

    return f"{analysis_base}buckets/bucket_{bucket_name}/"


def save_json(filepath: str, data: Dict[str, Any]):
    """
    Save dictionary to JSON file.

    Requirements:
    - Create parent directories if they don't exist (os.makedirs)
    - Use UTF-8 encoding
    - Pretty-print with indent=2 for readability
    - Handle write errors (raise IOError on failure)
    - Atomic write preferred (write to temp file, then rename)

    Args:
        filepath: str, path to JSON file
        data: dict, data to serialize

    Raises:
        IOError: if write fails

    Source: VideoProcessingTI.md Section 2.3.1 (Utility Functions)
    """
    filepath = Path(filepath)

    # Create parent directories
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file first, then rename
    temp_path = filepath.with_suffix(filepath.suffix + '.tmp')

    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_path.replace(filepath)

    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to write JSON to {filepath}: {e}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file to dictionary.

    Requirements:
    - Use UTF-8 encoding
    - Handle missing file (raise FileNotFoundError with clear message)
    - Handle invalid JSON (raise json.JSONDecodeError, preserve original error)
    - Return parsed dict

    Args:
        filepath: str, path to JSON file

    Returns:
        dict: parsed JSON data

    Raises:
        FileNotFoundError: if file doesn't exist
        json.JSONDecodeError: if JSON is malformed

    Source: VideoProcessingTI.md Section 2.3.1 (Utility Functions)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {filepath}: {e}")
        raise  # Re-raise original error with traceback


# Configuration constants
MIN_VIDEO_SIZE = 100_000  # 100KB minimum for valid video files
