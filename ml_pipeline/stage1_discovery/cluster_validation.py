"""
Cluster Configuration Validation Module

Validates hashtag cluster configuration schemas and constraints.

Source: HashtagVolumeV2_TI.md Section 4.1
"""

import re
import logging
from datetime import datetime

from .constants import (
    MIN_VARIANT_HASHTAGS,
    MAX_VARIANT_HASHTAGS,
    MIN_RUNS_PER_HASHTAG,
    MAX_RUNS_PER_HASHTAG,
    MIN_DELAY_BETWEEN_RUNS_MS,
    MAX_DELAY_BETWEEN_RUNS_MS,
    MIN_RESULTS_PER_PAGE,
    MAX_RESULTS_PER_PAGE
)

logger = logging.getLogger(__name__)


def validate_cluster_config(config: dict, config_path: str):
    """
    Validate cluster configuration schema and constraints.

    Source: HashtagVolumeV2_TI.md Section 4.1

    Args:
        config: dict, loaded cluster configuration
        config_path: str, path to config file (for error messages)

    Raises:
        ValueError: if validation fails (with specific error message)

    Returns:
        None (raises ValueError on failure)
    """
    # ===== REQUIRED FIELDS =====
    required_fields = [
        "cluster_id",
        "description",
        "primary_hashtag",
        "variant_hashtags",
        "scrape_config"
    ]

    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"Missing required field '{field}' in {config_path}\n"
                f"Schema: ClusterConfigSchema (see HashtagVolumeV2_TI.md Section 2.2)"
            )

    # ===== CLUSTER_ID VALIDATION =====
    cluster_id = config['cluster_id']

    if not re.match(r'^[a-zA-Z0-9_]+$', cluster_id):
        raise ValueError(
            f"Invalid cluster_id: '{cluster_id}'\n"
            f"Must be alphanumeric + underscore only (regex: ^[a-zA-Z0-9_]+$)\n"
            f"Example: 'nutrition', 'fitness_tips'"
        )

    if len(cluster_id) < 1:
        raise ValueError(
            f"Invalid cluster_id: '{cluster_id}'\n"
            f"Must be at least 1 character"
        )

    # ===== DESCRIPTION VALIDATION =====
    description = config['description']

    if not isinstance(description, str) or len(description) < 1:
        raise ValueError(
            f"Invalid description: must be non-empty string"
        )

    if len(description) > 500:
        raise ValueError(
            f"Invalid description: exceeds 500 characters ({len(description)} chars)\n"
            f"Keep descriptions concise."
        )

    # ===== PRIMARY_HASHTAG VALIDATION =====
    primary = config['primary_hashtag']

    if not isinstance(primary, str):
        raise ValueError(f"Invalid primary_hashtag: must be string, got {type(primary)}")

    if not primary.startswith('#'):
        raise ValueError(
            f"Invalid primary_hashtag: '{primary}'\n"
            f"Must start with # (e.g., '#nutrition')"
        )

    if len(primary) < 2:
        raise ValueError(
            f"Invalid primary_hashtag: '{primary}'\n"
            f"Must be at least 2 characters (# + 1 char)"
        )

    if not re.match(r'^#[a-zA-Z0-9_]+$', primary):
        raise ValueError(
            f"Invalid primary_hashtag: '{primary}'\n"
            f"Must be alphanumeric + underscore only (regex: ^#[a-zA-Z0-9_]+$)\n"
            f"Example: '#nutrition', '#fitness_tips'"
        )

    # ===== VARIANT_HASHTAGS VALIDATION =====
    variants = config['variant_hashtags']

    if not isinstance(variants, list):
        raise ValueError(
            f"Invalid variant_hashtags: must be array, got {type(variants)}"
        )

    if len(variants) < MIN_VARIANT_HASHTAGS or len(variants) > MAX_VARIANT_HASHTAGS:
        raise ValueError(
            f"Invalid variant_hashtags: must have {MIN_VARIANT_HASHTAGS}-{MAX_VARIANT_HASHTAGS} variants, got {len(variants)}\n"
            f"Current variants: {variants}"
        )

    # Validate each variant
    for i, variant in enumerate(variants):
        if not isinstance(variant, str):
            raise ValueError(
                f"Invalid variant_hashtags[{i}]: must be string, got {type(variant)}"
            )

        if not variant.startswith('#'):
            raise ValueError(
                f"Invalid variant_hashtags[{i}]: '{variant}'\n"
                f"Must start with # (e.g., '#nutritionist')"
            )

        if len(variant) < 2:
            raise ValueError(
                f"Invalid variant_hashtags[{i}]: '{variant}'\n"
                f"Must be at least 2 characters (# + 1 char)"
            )

        if not re.match(r'^#[a-zA-Z0-9_]+$', variant):
            raise ValueError(
                f"Invalid variant_hashtags[{i}]: '{variant}'\n"
                f"Must be alphanumeric + underscore only (regex: ^#[a-zA-Z0-9_]+$)"
            )

    # Check for duplicates (case-insensitive)
    all_hashtags = [primary.lower()] + [v.lower() for v in variants]
    if len(all_hashtags) != len(set(all_hashtags)):
        duplicates = [h for h in all_hashtags if all_hashtags.count(h) > 1]
        raise ValueError(
            f"Duplicate hashtags found (case-insensitive): {list(set(duplicates))}\n"
            f"Each hashtag must be unique within cluster"
        )

    # ===== SCRAPE_CONFIG VALIDATION =====
    if 'scrape_config' not in config:
        raise ValueError(
            f"Missing required field 'scrape_config' in {config_path}"
        )

    scrape_config = config['scrape_config']

    # Required scrape_config fields
    required_scrape_fields = [
        "runs_per_hashtag",
        "delay_between_runs_ms",
        "results_per_page"
    ]

    for field in required_scrape_fields:
        if field not in scrape_config:
            raise ValueError(
                f"Missing required field 'scrape_config.{field}' in {config_path}"
            )

    # Validate runs_per_hashtag
    runs = scrape_config['runs_per_hashtag']
    if not isinstance(runs, int):
        raise ValueError(
            f"Invalid scrape_config.runs_per_hashtag: must be integer, got {type(runs)}"
        )

    if runs < MIN_RUNS_PER_HASHTAG or runs > MAX_RUNS_PER_HASHTAG:
        raise ValueError(
            f"Invalid scrape_config.runs_per_hashtag: must be {MIN_RUNS_PER_HASHTAG}-{MAX_RUNS_PER_HASHTAG}, got {runs}"
        )

    # Validate delay_between_runs_ms
    delay = scrape_config['delay_between_runs_ms']
    if not isinstance(delay, int):
        raise ValueError(
            f"Invalid scrape_config.delay_between_runs_ms: must be integer, got {type(delay)}"
        )

    if delay < MIN_DELAY_BETWEEN_RUNS_MS or delay > MAX_DELAY_BETWEEN_RUNS_MS:
        raise ValueError(
            f"Invalid scrape_config.delay_between_runs_ms: must be {MIN_DELAY_BETWEEN_RUNS_MS}-{MAX_DELAY_BETWEEN_RUNS_MS} ms, got {delay}\n"
            f"Acceptable range: {MIN_DELAY_BETWEEN_RUNS_MS/60000:.1f}-{MAX_DELAY_BETWEEN_RUNS_MS/60000:.1f} minutes"
        )

    # Validate results_per_page
    results = scrape_config['results_per_page']
    if not isinstance(results, int):
        raise ValueError(
            f"Invalid scrape_config.results_per_page: must be integer, got {type(results)}"
        )

    if results < MIN_RESULTS_PER_PAGE or results > MAX_RESULTS_PER_PAGE:
        raise ValueError(
            f"Invalid scrape_config.results_per_page: must be {MIN_RESULTS_PER_PAGE}-{MAX_RESULTS_PER_PAGE}, got {results}"
        )

    # ===== METADATA VALIDATION (OPTIONAL) =====
    if 'metadata' in config:
        metadata = config['metadata']

        # Validate created_date if present
        if 'created_date' in metadata:
            created_date = metadata['created_date']
            try:
                datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(
                    f"Invalid metadata.created_date: '{created_date}'\n"
                    f"Must be ISO 8601 format (e.g., '2025-10-09T10:30:00Z')"
                )

    # All validations passed
    logger.debug(f"✅ Cluster config validation passed: {cluster_id}")
