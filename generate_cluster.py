#!/usr/bin/env python3
"""
Interactive Cluster Configuration Generator

Creates hashtag cluster configurations for multi-hashtag scraping strategy.

Source: HashtagVolumeV2_TI.md DECISION 4
Usage: python generate_cluster.py
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline.stage1_discovery.constants import (
    MIN_VARIANT_HASHTAGS,
    MAX_VARIANT_HASHTAGS,
    MIN_RUNS_PER_HASHTAG,
    MAX_RUNS_PER_HASHTAG,
    MIN_DELAY_BETWEEN_RUNS_MS,
    MAX_DELAY_BETWEEN_RUNS_MS,
    MIN_RESULTS_PER_PAGE,
    MAX_RESULTS_PER_PAGE,
    DEFAULT_RUNS_PER_HASHTAG,
    DEFAULT_DELAY_BETWEEN_RUNS_MS,
    DEFAULT_RESULTS_PER_PAGE
)
from ml_pipeline.stage1_discovery.cluster_validation import validate_cluster_config


def print_header():
    """Print welcome header."""
    print()
    print("="*80)
    print("HASHTAG CLUSTER CONFIGURATION GENERATOR")
    print("="*80)
    print()
    print("This tool helps you create a hashtag cluster configuration for multi-hashtag")
    print("scraping. Cluster scraping provides 2-3x more unique videos with rich")
    print("analytics for optimization.")
    print()
    print("="*80)
    print()


def prompt_cluster_id() -> str:
    """Prompt for cluster ID with validation."""
    print("Step 1: Cluster Identifier")
    print("-" * 80)
    print("Enter a unique identifier for this cluster (alphanumeric + underscore only)")
    print("Examples: nutrition, fitness_tips, wellness")
    print()

    while True:
        cluster_id = input("Cluster ID: ").strip()

        if not cluster_id:
            print("❌ Cluster ID cannot be empty. Please try again.\n")
            continue

        if not re.match(r'^[a-zA-Z0-9_]+$', cluster_id):
            print("❌ Invalid format. Use only letters, numbers, and underscores.\n")
            continue

        # Check if cluster already exists
        config_path = Path(__file__).parent / "config" / "hashtag_clusters" / f"{cluster_id}.json"
        if config_path.exists():
            print(f"❌ Cluster '{cluster_id}' already exists at: {config_path}")
            overwrite = input("Overwrite existing cluster? (yes/no): ").strip().lower()
            if overwrite != "yes":
                print("Please choose a different cluster ID.\n")
                continue

        print(f"✅ Cluster ID: {cluster_id}\n")
        return cluster_id


def prompt_description() -> str:
    """Prompt for cluster description."""
    print("Step 2: Description")
    print("-" * 80)
    print("Enter a brief description of this cluster (1-500 characters)")
    print("Example: Nutrition niche - narrow semantic cluster")
    print()

    while True:
        description = input("Description: ").strip()

        if not description:
            print("❌ Description cannot be empty. Please try again.\n")
            continue

        if len(description) > 500:
            print(f"❌ Description too long ({len(description)} chars). Maximum 500 characters.\n")
            continue

        print(f"✅ Description: {description}\n")
        return description


def prompt_primary_hashtag() -> str:
    """Prompt for primary hashtag with validation."""
    print("Step 3: Primary Hashtag")
    print("-" * 80)
    print("Enter the main hashtag for this cluster (include # prefix)")
    print("Example: #nutrition")
    print()

    while True:
        hashtag = input("Primary hashtag: ").strip()

        if not hashtag:
            print("❌ Hashtag cannot be empty. Please try again.\n")
            continue

        if not hashtag.startswith('#'):
            print("❌ Hashtag must start with #. Please try again.\n")
            continue

        if len(hashtag) < 2:
            print("❌ Hashtag must be at least 2 characters (# + 1 char). Please try again.\n")
            continue

        if not re.match(r'^#[a-zA-Z0-9_]+$', hashtag):
            print("❌ Invalid format. Use only letters, numbers, and underscores after #.\n")
            continue

        print(f"✅ Primary hashtag: {hashtag}\n")
        return hashtag


def prompt_variant_hashtags(primary_hashtag: str) -> list:
    """Prompt for variant hashtags."""
    print("Step 4: Variant Hashtags")
    print("-" * 80)
    print(f"Enter {MIN_VARIANT_HASHTAGS}-{MAX_VARIANT_HASHTAGS} variant hashtags (semantically related to {primary_hashtag})")
    print("Examples: #nutritionist, #nutritiontips, #nutritioncoach")
    print()
    print("Enter one hashtag at a time. Type 'done' when finished.")
    print()

    variants = []
    all_hashtags = {primary_hashtag.lower()}

    while len(variants) < MAX_VARIANT_HASHTAGS:
        prompt = f"Variant #{len(variants)+1} (or 'done'): "
        variant = input(prompt).strip()

        if variant.lower() == 'done':
            if len(variants) < MIN_VARIANT_HASHTAGS:
                print(f"❌ Need at least {MIN_VARIANT_HASHTAGS} variant(s). Currently have {len(variants)}.\n")
                continue
            break

        if not variant:
            print("❌ Hashtag cannot be empty. Type 'done' to finish.\n")
            continue

        if not variant.startswith('#'):
            print("❌ Hashtag must start with #. Please try again.\n")
            continue

        if len(variant) < 2:
            print("❌ Hashtag must be at least 2 characters (# + 1 char). Please try again.\n")
            continue

        if not re.match(r'^#[a-zA-Z0-9_]+$', variant):
            print("❌ Invalid format. Use only letters, numbers, and underscores after #.\n")
            continue

        if variant.lower() in all_hashtags:
            print(f"❌ Duplicate hashtag (case-insensitive). Already added.\n")
            continue

        variants.append(variant)
        all_hashtags.add(variant.lower())
        print(f"✅ Added: {variant} ({len(variants)}/{MAX_VARIANT_HASHTAGS})\n")

    print(f"✅ Total variant hashtags: {len(variants)}\n")
    return variants


def prompt_scrape_config() -> dict:
    """Prompt for scrape configuration."""
    print("Step 5: Scrape Configuration")
    print("-" * 80)
    print("Configure scraping parameters (or press Enter for defaults)")
    print()

    # Runs per hashtag
    print(f"Runs per hashtag ({MIN_RUNS_PER_HASHTAG}-{MAX_RUNS_PER_HASHTAG}, default: {DEFAULT_RUNS_PER_HASHTAG})")
    print("How many times to scrape each hashtag?")
    while True:
        runs_input = input(f"Runs [{DEFAULT_RUNS_PER_HASHTAG}]: ").strip()
        if not runs_input:
            runs_per_hashtag = DEFAULT_RUNS_PER_HASHTAG
            break

        try:
            runs_per_hashtag = int(runs_input)
            if MIN_RUNS_PER_HASHTAG <= runs_per_hashtag <= MAX_RUNS_PER_HASHTAG:
                break
            print(f"❌ Must be between {MIN_RUNS_PER_HASHTAG} and {MAX_RUNS_PER_HASHTAG}.\n")
        except ValueError:
            print("❌ Must be an integer.\n")

    print(f"✅ Runs per hashtag: {runs_per_hashtag}\n")

    # Delay between runs
    print(f"Delay between runs in minutes ({MIN_DELAY_BETWEEN_RUNS_MS/60000:.0f}-{MAX_DELAY_BETWEEN_RUNS_MS/60000:.0f}, default: {DEFAULT_DELAY_BETWEEN_RUNS_MS/60000:.0f})")
    print("How long to wait between scrapes?")
    while True:
        delay_input = input(f"Delay (minutes) [{DEFAULT_DELAY_BETWEEN_RUNS_MS/60000:.0f}]: ").strip()
        if not delay_input:
            delay_ms = DEFAULT_DELAY_BETWEEN_RUNS_MS
            break

        try:
            delay_minutes = float(delay_input)
            delay_ms = int(delay_minutes * 60000)
            if MIN_DELAY_BETWEEN_RUNS_MS <= delay_ms <= MAX_DELAY_BETWEEN_RUNS_MS:
                break
            print(f"❌ Must be between {MIN_DELAY_BETWEEN_RUNS_MS/60000:.0f} and {MAX_DELAY_BETWEEN_RUNS_MS/60000:.0f} minutes.\n")
        except ValueError:
            print("❌ Must be a number.\n")

    print(f"✅ Delay: {delay_ms/60000:.1f} minutes ({delay_ms}ms)\n")

    # Results per page
    print(f"Results per scrape ({MIN_RESULTS_PER_PAGE}-{MAX_RESULTS_PER_PAGE}, default: {DEFAULT_RESULTS_PER_PAGE})")
    print("How many videos to scrape per run?")
    while True:
        results_input = input(f"Results [{DEFAULT_RESULTS_PER_PAGE}]: ").strip()
        if not results_input:
            results_per_page = DEFAULT_RESULTS_PER_PAGE
            break

        try:
            results_per_page = int(results_input)
            if MIN_RESULTS_PER_PAGE <= results_per_page <= MAX_RESULTS_PER_PAGE:
                break
            print(f"❌ Must be between {MIN_RESULTS_PER_PAGE} and {MAX_RESULTS_PER_PAGE}.\n")
        except ValueError:
            print("❌ Must be an integer.\n")

    print(f"✅ Results per scrape: {results_per_page}\n")

    return {
        "runs_per_hashtag": runs_per_hashtag,
        "delay_between_runs_ms": delay_ms,
        "results_per_page": results_per_page
    }


def save_cluster_config(cluster_id: str, config: dict):
    """Save cluster configuration to JSON file."""
    config_dir = Path(__file__).parent / "config" / "hashtag_clusters"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"{cluster_id}.json"

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Saved cluster configuration: {config_path}\n")
    return config_path


def print_summary(cluster_id: str, config: dict, config_path: Path):
    """Print configuration summary and next steps."""
    print("="*80)
    print("CLUSTER CONFIGURATION SUMMARY")
    print("="*80)
    print()
    print(f"Cluster ID: {config['cluster_id']}")
    print(f"Description: {config['description']}")
    print(f"Primary Hashtag: {config['primary_hashtag']}")
    print(f"Variant Hashtags ({len(config['variant_hashtags'])}): {', '.join(config['variant_hashtags'])}")
    print()
    print(f"Scrape Configuration:")
    print(f"  - Runs per hashtag: {config['scrape_config']['runs_per_hashtag']}")
    print(f"  - Delay between runs: {config['scrape_config']['delay_between_runs_ms']/60000:.1f} minutes")
    print(f"  - Results per scrape: {config['scrape_config']['results_per_page']}")
    print()

    total_hashtags = len(config['variant_hashtags']) + 1
    total_scrapes = total_hashtags * config['scrape_config']['runs_per_hashtag']
    estimated_videos = total_scrapes * config['scrape_config']['results_per_page']

    print(f"Expected Scraping:")
    print(f"  - Total hashtags: {total_hashtags}")
    print(f"  - Total scrapes: {total_scrapes}")
    print(f"  - Estimated videos (before dedup): {estimated_videos:,}")
    print()
    print("="*80)
    print()
    print("NEXT STEPS:")
    print("="*80)
    print()
    print("1. Review the configuration file:")
    print(f"   cat {config_path}")
    print()
    print("2. Run the ML pipeline with your new cluster:")
    print(f"   python rumiai_ml_batch.py --client YOUR_CLIENT --target {cluster_id} --analysis-type hashtag")
    print()
    print("3. View cluster analytics after scraping:")
    print(f"   cat data/clients/YOUR_CLIENT/hashtag/{cluster_id}/cluster_analytics.json")
    print()
    print("="*80)
    print()


def main():
    """Main entry point for cluster generator."""
    try:
        print_header()

        # Step 1: Cluster ID
        cluster_id = prompt_cluster_id()

        # Step 2: Description
        description = prompt_description()

        # Step 3: Primary hashtag
        primary_hashtag = prompt_primary_hashtag()

        # Step 4: Variant hashtags
        variant_hashtags = prompt_variant_hashtags(primary_hashtag)

        # Step 5: Scrape configuration
        scrape_config = prompt_scrape_config()

        # Assemble configuration
        config = {
            "cluster_id": cluster_id,
            "description": description,
            "primary_hashtag": primary_hashtag,
            "variant_hashtags": variant_hashtags,
            "scrape_config": scrape_config,
            "metadata": {
                "created_date": datetime.now(timezone.utc).isoformat(),
                "created_by": os.getenv("USER", "unknown"),
                "notes": "Generated with generate_cluster.py"
            }
        }

        # Validate configuration
        print("Validating configuration...")
        validate_cluster_config(config, f"{cluster_id}.json")
        print("✅ Validation passed\n")

        # Save configuration
        config_path = save_cluster_config(cluster_id, config)

        # Print summary
        print_summary(cluster_id, config, config_path)

        return 0

    except KeyboardInterrupt:
        print("\n\n❌ Configuration cancelled by user (Ctrl+C)")
        return 130

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
