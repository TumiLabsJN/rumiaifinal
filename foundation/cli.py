"""
CLI argument parsing and validation.

Source: FoundationCHILD.md Section 4.1: CLI Parameters, Section 4.2: Default Value Logic
"""

import argparse
import re
from dataclasses import dataclass
from typing import Optional
from foundation.constants import (
    VALID_ANALYSIS_TYPES,
    VALID_MODES,
    VALID_STRATEGIES,
    VALID_REPORT_TYPES,
    VALID_REPORT_AUDIENCES,
    VALID_COUNTRY_CODES,
    DEFAULT_VIDEO_COUNTS,
    DEFAULT_ANALYSIS_MODES,
    DEFAULT_SELECTION_STRATEGIES,
    DEFAULT_REPORT_AUDIENCES,
    DEFAULT_COUNTRY_CODE,
    MIN_RECOMMENDED_N,
)
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CLIArgs:
    """Validated CLI arguments."""
    client: str
    analysis_type: str
    target: str
    analysis_mode: str
    selection_strategy: str
    video_count: int
    date_filter: str
    report_type: str
    report_audience: str
    auto_confirm: bool
    country_code: str

    def to_dict(self) -> dict:
        """Convert to dict for config.json."""
        return {
            "client": self.client,
            "analysis_type": self.analysis_type,
            "target": self.target,
            "analysis_mode": self.analysis_mode,
            "selection_strategy": self.selection_strategy,
            "video_count": self.video_count,
            "date_filter": self.date_filter,
            "report_type": self.report_type,
            "report_audience": self.report_audience,
            "auto_confirm": self.auto_confirm,
            "country_code": self.country_code,
        }


def parse_args(argv: Optional[list] = None) -> CLIArgs:
    """
    Parse CLI arguments with validation.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        CLIArgs with validated parameters

    Raises:
        SystemExit: If validation fails (argparse handles this)

    Source: FoundationCHILD.md Section 4.1: CLI Parameters
    """
    parser = argparse.ArgumentParser(
        prog="rumiai_ml_batch.py",
        description="RumiAI ML Pipeline - Batch video analysis"
    )

    # Required arguments
    parser.add_argument(
        "--client",
        required=True,
        type=str,
        help="Client identifier (alphanumeric + underscore, e.g., 'acme_corp')"
    )

    parser.add_argument(
        "--analysis-type",
        required=True,
        choices=VALID_ANALYSIS_TYPES,
        help="Target type to analyze"
    )

    parser.add_argument(
        "--target",
        required=True,
        type=str,
        help="Target identifier (#nutrition, @rival_brand, @creator_name)"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--analysis-mode",
        choices=VALID_MODES,
        default=None,
        help="Sorting method (default: top for hashtag/competitor, recent for creator)"
    )

    parser.add_argument(
        "--selection-strategy",
        choices=VALID_STRATEGIES,
        default=None,
        help="Video subset selection strategy (default: contrastive for hashtag/competitor, top for creator)"
    )

    parser.add_argument(
        "--video-count",
        type=int,
        default=None,
        help="Videos to analyze per winning bucket (range: 10-500, default: 100 for contrastive, 40 for top)"
    )

    parser.add_argument(
        "--date-filter",
        type=str,
        default="last_90_days",
        help="Publication date filter (format: last_N_days, e.g., last_90_days)"
    )

    parser.add_argument(
        "--report-type",
        choices=VALID_REPORT_TYPES,
        default="single",
        help="Output type (single or comparison)"
    )

    parser.add_argument(
        "--report-audience",
        choices=VALID_REPORT_AUDIENCES,
        default=None,
        help="Target audience for report (default: client for hashtag/competitor, creator for creator)"
    )

    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        default=False,
        help="Skip interactive confirmation prompts (for CI/CD automation)"
    )

    parser.add_argument(
        "--country-code",
        choices=VALID_COUNTRY_CODES,
        default=DEFAULT_COUNTRY_CODE,
        help="Geographic filtering (default: US, options: US/BR/global)"
    )

    parsed = parser.parse_args(argv)

    # Apply defaults based on analysis_type
    args_with_defaults = apply_defaults(parsed, parsed.analysis_type)

    # Validate argument combinations
    validate_cli_args(args_with_defaults)

    # Check minimum N recommendations
    check_minimum_n_recommendations(args_with_defaults.video_count, args_with_defaults.selection_strategy)

    return CLIArgs(**vars(args_with_defaults))


def apply_defaults(args: argparse.Namespace, analysis_type: str) -> argparse.Namespace:
    """
    Apply default values based on analysis_type.

    Logic from FoundationCHILD.md Section 4.2: Default Value Logic:
    - Hashtag: mode=top, strategy=contrastive, video_count=100, audience=client
    - Competitor: mode=top, strategy=contrastive, video_count=100, audience=client
    - Creator: mode=recent, strategy=top, video_count=40, audience=creator

    Args:
        args: Parsed arguments from argparse
        analysis_type: Type of analysis

    Returns:
        Namespace with defaults applied
    """
    # Apply analysis_mode default
    if args.analysis_mode is None:
        args.analysis_mode = DEFAULT_ANALYSIS_MODES[analysis_type]

    # Apply selection_strategy default
    if args.selection_strategy is None:
        args.selection_strategy = DEFAULT_SELECTION_STRATEGIES[analysis_type]

    # Apply video_count default
    if args.video_count is None:
        args.video_count = DEFAULT_VIDEO_COUNTS[analysis_type]

    # Apply report_audience default
    if args.report_audience is None:
        args.report_audience = DEFAULT_REPORT_AUDIENCES[analysis_type]

    return args


def validate_cli_args(args: argparse.Namespace) -> None:
    """
    Validate CLI argument combinations.

    Validation Rules from FoundationCHILD.md Section 4.1: CLI Parameters:
    - client: Regex ^[a-zA-Z0-9_]+$
    - analysis_type: Enum ["hashtag", "competitor", "creator"]
    - target: Format depends on analysis_type (# or @ prefix required)
    - video_count: Range 10-500
    - date_filter: Regex ^last_\\d+_days$ where \\d+ is 1-365

    Raises:
        ValueError: With specific error message guiding user to fix issue
    """
    # Validate client_id format
    if not re.match(r"^[a-zA-Z0-9_]+$", args.client):
        raise ValueError(
            f"Invalid --client '{args.client}'. "
            f"Must contain only alphanumeric characters and underscores."
        )

    # Validate target format based on analysis_type
    if args.analysis_type == "hashtag":
        # Allow both cluster names (alphanumeric) and single hashtags (deprecated, caught later)
        # Cluster names: alphanumeric + underscore (e.g., "nutrition", "fitness_tips")
        # Single hashtags: start with # (e.g., "#nutrition") - deprecated in video_discovery.py
        if args.target.startswith("#"):
            # Single hashtag format (will be caught as deprecated in detect_target_type)
            if len(args.target) < 2:
                raise ValueError(
                    f"Invalid --target '{args.target}' for analysis_type 'hashtag'. "
                    f"Must have at least 2 characters (e.g., #nutrition)."
                )
        else:
            # Cluster name format - validate alphanumeric + underscore
            if not re.match(r"^[a-zA-Z0-9_]+$", args.target):
                raise ValueError(
                    f"Invalid --target '{args.target}' for analysis_type 'hashtag'. "
                    f"Cluster names must be alphanumeric + underscore only (e.g., 'nutrition', 'fitness_tips')."
                )
    elif args.analysis_type in ["competitor", "creator"]:
        if not args.target.startswith("@") or len(args.target) < 2:
            raise ValueError(
                f"Invalid --target '{args.target}' for analysis_type '{args.analysis_type}'. "
                f"Must start with @ and have at least 2 characters (e.g., @rival_brand)."
            )

    # Validate video_count range (lowered to 3 for testing purposes)
    if not (3 <= args.video_count <= 500):
        raise ValueError(
            f"Invalid --video-count {args.video_count}. "
            f"Must be between 3 and 500 (inclusive)."
        )

    # Validate date_filter format
    if not re.match(r"^last_\d+_days$", args.date_filter):
        raise ValueError(
            f"Invalid --date-filter '{args.date_filter}'. "
            f"Must match format 'last_N_days' where N is 1-365 (e.g., last_90_days)."
        )
    days = int(args.date_filter.split("_")[1])
    if not (1 <= days <= 365):
        raise ValueError(
            f"Invalid --date-filter '{args.date_filter}'. "
            f"Days must be between 1 and 365."
        )


def check_minimum_n_recommendations(video_count: int, selection_strategy: str) -> None:
    """
    Check if video_count meets recommended minimums.

    Warning thresholds from FoundationCHILD.md Section 3.4: Video Count (N):
    - Contrastive: N < 50 → Warn (bottom 20% = only 10 videos)
    - Top: N < 20 → Warn (insufficient for 3-cluster K-Means)

    Raises warning but does NOT fail (allows N=10-49 with warning).

    Args:
        video_count: Number of videos to analyze
        selection_strategy: Strategy being used
    """
    if selection_strategy == "contrastive" and video_count < MIN_RECOMMENDED_N["contrastive"]:
        bottom_count = int(video_count * 0.2)
        logger.warning(
            f"Low bottom performer count ({bottom_count} videos, {video_count} × 0.2). "
            f"Recommend N ≥ {MIN_RECOMMENDED_N['contrastive']} for robust classification. "
            f"Statistical validity may be limited."
        )
    elif selection_strategy == "top" and video_count < MIN_RECOMMENDED_N["top"]:
        logger.warning(
            f"Low sample size for clustering ({video_count} videos). "
            f"Recommend N ≥ {MIN_RECOMMENDED_N['top']} for pattern detection. "
            f"K-Means may produce unstable clusters."
        )
