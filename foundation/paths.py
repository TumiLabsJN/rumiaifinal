"""
Path generation and sanitization utilities.

Source: FoundationCHILD.md Section 2.1: Directory Structure, Section 2.2.1: Path Sanitization Rules
"""

from pathlib import Path
import re
from typing import Dict


class PathBuilder:
    """
    Builds and manages file system paths for client data.

    Source: FoundationCHILD.md Section 2.1: Directory Structure
    """

    def __init__(self, base_path: Path = None):
        """
        Initialize PathBuilder.

        Args:
            base_path: Base directory for all client data
                      If None, uses DATA_ROOT env var or defaults to /data
        """
        import os
        if base_path is None:
            data_root = os.getenv("DATA_ROOT", "/data")
            base_path = Path(data_root)
        self.base_path = base_path

    def get_client_base(self, client_id: str) -> Path:
        """
        Get base directory for client.

        Args:
            client_id: Sanitized client identifier

        Returns:
            Path: /data/clients/{client_id}/
        """
        return self.base_path / "clients" / client_id

    def get_target_dir(
        self,
        client_id: str,
        analysis_type: str,
        target: str,
        analysis_mode: str,
        selection_strategy: str
    ) -> Path:
        """
        Get target directory path.

        Args:
            client_id: Client identifier
            analysis_type: "hashtag", "competitor", or "creator"
            target: Target with prefix (#nutrition, @brand)
            analysis_mode: "top" or "recent"
            selection_strategy: "contrastive" or "top"

        Returns:
            Path: /data/clients/{client_id}/{analysis_type}s/{sanitized_target}/{mode}_{strategy}/

        Example:
            >>> pb = PathBuilder()
            >>> pb.get_target_dir("acme", "hashtag", "#nutrition", "top", "contrastive")
            PosixPath('/data/clients/acme/hashtags/nutrition/top_contrastive')
        """
        sanitized_target = sanitize_target(target, analysis_type)

        # Pluralize analysis_type for directory name
        analysis_type_plural = f"{analysis_type}s"

        # Mode + Strategy directory name
        mode_strategy = f"{analysis_mode}_{selection_strategy}"

        return (
            self.base_path
            / "clients"
            / client_id
            / analysis_type_plural
            / sanitized_target
            / mode_strategy
        )

    def get_bucket_dir(self, target_dir: Path, bucket: str) -> Path:
        """
        Get bucket directory within target.

        Args:
            target_dir: Target base directory
            bucket: Bucket name (e.g., "18-33s")

        Returns:
            Path: {target_dir}/buckets/bucket_{bucket}/

        Example:
            >>> target_dir = Path("/data/clients/acme/hashtags/nutrition/top_contrastive")
            >>> pb = PathBuilder()
            >>> pb.get_bucket_dir(target_dir, "18-33s")
            PosixPath('/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s')
        """
        return target_dir / "buckets" / f"bucket_{bucket}"

    def create_directory_structure(self, target_dir: Path) -> Dict[str, Path]:
        """
        Create directory structure for target.

        Creates:
        - config.json location (target_dir)
        - buckets/ subdirectory
        - All 8 bucket subdirectories with videos/, analysis/, ml_analysis/, models/, reports/, checkpoints/, logs/

        Args:
            target_dir: Target base directory

        Returns:
            Dict mapping bucket names to their paths

        Source: FoundationCHILD.md Section 2.1: Directory Structure
        """
        from foundation.constants import BUCKET_DEFINITIONS

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create buckets/ parent directory
        buckets_dir = target_dir / "buckets"
        buckets_dir.mkdir(exist_ok=True)

        # Create each bucket subdirectory
        bucket_paths = {}
        for bucket in BUCKET_DEFINITIONS.keys():
            bucket_dir = self.get_bucket_dir(target_dir, bucket)
            bucket_dir.mkdir(exist_ok=True)

            # Create subdirectories within each bucket
            (bucket_dir / "videos").mkdir(exist_ok=True)
            (bucket_dir / "analysis").mkdir(exist_ok=True)
            (bucket_dir / "ml_analysis").mkdir(exist_ok=True)
            (bucket_dir / "models").mkdir(exist_ok=True)
            (bucket_dir / "reports").mkdir(exist_ok=True)
            (bucket_dir / "checkpoints").mkdir(exist_ok=True)
            (bucket_dir / "logs").mkdir(exist_ok=True)

            bucket_paths[bucket] = bucket_dir

        return bucket_paths


def sanitize_target(target: str, analysis_type: str) -> str:
    """
    Sanitize target for filesystem path usage.

    Rules from FoundationCHILD.md Section 2.2.1: Path Sanitization Rules:
    1. Remove prefix (# for hashtag, @ for competitor/creator)
    2. Convert to lowercase
    3. Replace spaces with underscores
    4. Remove special characters (keep only alphanumeric, underscore, hyphen)
    5. Collapse multiple underscores to single underscore
    6. Strip leading/trailing underscores

    Args:
        target: Original target with prefix
        analysis_type: "hashtag", "competitor", or "creator"

    Returns:
        Sanitized target string

    Examples:
        >>> sanitize_target("#Fitness & Nutrition!", "hashtag")
        'fitness_nutrition'
        >>> sanitize_target("@My Brand 2024", "competitor")
        'my_brand_2024'
        >>> sanitize_target("#nutrition", "hashtag")
        'nutrition'
    """
    # Step 1: Remove prefix
    if analysis_type == "hashtag" and target.startswith("#"):
        sanitized = target[1:]
    elif analysis_type in ["competitor", "creator"] and target.startswith("@"):
        sanitized = target[1:]
    else:
        sanitized = target

    # Step 2: Lowercase
    sanitized = sanitized.lower()

    # Step 3: Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")

    # Step 4: Remove special characters (keep alphanumeric, underscore, hyphen)
    sanitized = re.sub(r'[^a-z0-9_-]', '', sanitized)

    # Step 5: Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Step 6: Strip leading/trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized


def sanitize_client_id(client_id: str) -> str:
    """
    Sanitize client ID (follows same rules as target, but no prefix removal).

    Args:
        client_id: Original client ID

    Returns:
        Sanitized client ID

    Example:
        >>> sanitize_client_id("Acme Corp!")
        'acme_corp'
    """
    # Follow same sanitization as target, but no prefix removal step
    sanitized = client_id.lower()
    sanitized = sanitized.replace(" ", "_")
    sanitized = re.sub(r'[^a-z0-9_-]', '', sanitized)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized
