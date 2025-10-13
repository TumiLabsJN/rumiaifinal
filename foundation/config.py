"""
Configuration management for RumiAI ML Pipeline.

Source: FoundationCHILD.md Section 5.1: config.json Schema
"""

from pathlib import Path
from datetime import datetime
import json
from foundation.schemas import Config


class ConfigManager:
    """Manages configuration loading, saving, and creation."""

    @staticmethod
    def from_cli_args(args) -> Config:
        """
        Create Config from CLI arguments.

        Args:
            args: Validated CLI arguments (CLIArgs dataclass)

        Returns:
            Config object with current run_date
        """
        return Config(
            client_id=args.client,
            analysis_type=args.analysis_type,
            target=args.target,
            analysis_mode=args.analysis_mode,
            selection_strategy=args.selection_strategy,
            video_count=args.video_count,
            date_filter=args.date_filter,
            report_type=args.report_type,
            report_audience=args.report_audience,
            auto_confirm=args.auto_confirm,
            run_date=datetime.utcnow().isoformat() + "Z"
        )

    @staticmethod
    def load(path: Path) -> Config:
        """
        Load config from JSON file.

        Args:
            path: Path to config.json

        Returns:
            Validated Config object

        Raises:
            FileNotFoundError: If config.json doesn't exist
            ValueError: If config.json is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        try:
            return Config(**data)
        except Exception as e:
            raise ValueError(f"Invalid config.json: {e}")

    @staticmethod
    def save(config: Config, path: Path) -> None:
        """
        Save config to JSON file.

        Args:
            config: Config object to save
            path: Path to config.json

        Creates parent directories if they don't exist.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
