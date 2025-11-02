#!/usr/bin/env python3
"""
CLI Entry Point for Stage 2.7: Video Classification

Source: ContentAnalysisCHILDTI.md Section 6

Usage:
    python run_stage_2_7.py --client acme_corp --hashtag nutrition
    python run_stage_2_7.py --client acme_corp --hashtag nutrition --parallel
    python run_stage_2_7.py --client acme_corp --hashtag nutrition --parallel --max-workers 10
    python run_stage_2_7.py --client acme_corp --hashtag nutrition --no-checkpoint
"""

import argparse
import logging
import sys
import os

# Add ml_pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml_pipeline'))

from stage2_content_analysis.classification import run_classification_stage
from stage2_content_analysis.taxonomy_validation import validate_curated_taxonomy
from foundation.paths import PathBuilder, sanitize_target


def setup_logging(verbose=False):
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Stage 2.7: Video Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (sequential mode)
  python run_stage_2_7.py --client acme_corp --hashtag nutrition

  # Parallel mode (5 workers, ~5x faster)
  python run_stage_2_7.py --client acme_corp --hashtag nutrition --parallel

  # Parallel mode with custom worker count
  python run_stage_2_7.py --client acme_corp --hashtag nutrition --parallel --max-workers 10

  # Resume from checkpoint after interruption
  python run_stage_2_7.py --client acme_corp --hashtag nutrition

  # Disable checkpointing
  python run_stage_2_7.py --client acme_corp --hashtag nutrition --no-checkpoint

Environment Variables:
  ANTHROPIC_API_KEY: Required - API key for Claude
  RUMIAI_ROOT: Optional - Base directory (default: /home/jorge/rumiaifinal)
  ENABLE_PARALLEL_CLASSIFICATION: Optional - Enable parallel mode (true/false, default: false)
  MAX_CLASSIFICATION_WORKERS: Optional - Worker count for parallel mode (default: 5)
        """
    )

    parser.add_argument(
        '--client',
        required=True,
        help='Client identifier (e.g., acme_corp)'
    )

    parser.add_argument(
        '--hashtag',
        required=True,
        help='Hashtag name (with or without # prefix)'
    )

    parser.add_argument(
        '--analysis-type',
        required=True,
        choices=['hashtag', 'competitor', 'creator'],
        help='Analysis type (hashtag, competitor, or creator)'
    )

    parser.add_argument(
        '--analysis-mode',
        default='top',
        choices=['top', 'recent'],
        help='Analysis mode (default: top)'
    )

    parser.add_argument(
        '--selection-strategy',
        default='contrastive',
        choices=['contrastive', 'top'],
        help='Selection strategy (default: contrastive)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel classification mode (5x faster, ~5 workers)'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Number of workers for parallel mode (default: 5, max recommended: 10)'
    )

    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Disable checkpoint/resume (not recommended for large batches)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate taxonomy, do not run classification'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    # Parse args
    args = parse_args()

    # Setup
    setup_logging(verbose=args.verbose)

    # Validate environment
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("❌ ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("   Set it with: export ANTHROPIC_API_KEY=your_api_key")
        sys.exit(1)

    # Sanitize target (remove prefix based on analysis type)
    target_sanitized = sanitize_target(args.hashtag, args.analysis_type)

    # Step 1: Validate taxonomy
    print("\n" + "=" * 80)
    print("STEP 1: VALIDATING TAXONOMY")
    print("=" * 80)

    # Build taxonomy path using PathBuilder
    path_builder = PathBuilder()
    target_dir = path_builder.get_target_dir(
        client_id=args.client,
        analysis_type=args.analysis_type,
        target=args.hashtag,
        analysis_mode=args.analysis_mode,
        selection_strategy=args.selection_strategy
    )
    taxonomy_path = str(target_dir / "content_taxonomies" / f"{target_sanitized}_taxonomy.json")

    try:
        validate_curated_taxonomy(taxonomy_path)
        print("\n✅ Taxonomy validation passed!")

        if args.validate_only:
            print("\n--validate-only flag set, skipping classification")
            sys.exit(0)

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\n   Make sure you have:")
        print("   1. Run Stage 2.6 (python run_stage_2_6.py)")
        print("   2. Manually curated the raw discovery file")
        print(f"   3. Saved it as: {taxonomy_path}")
        sys.exit(1)

    except ValueError as e:
        print(f"❌ VALIDATION ERROR: {e}")
        print(f"\n   Fix the errors in: {taxonomy_path}")
        print("   Then re-run this command")
        sys.exit(1)

    # Step 2: Run classification
    print("\n" + "=" * 80)
    print("STEP 2: CLASSIFYING VIDEOS")
    print("=" * 80)

    # Show configuration
    mode = "PARALLEL" if args.parallel else "SEQUENTIAL"
    checkpoint_status = "DISABLED" if args.no_checkpoint else "ENABLED"
    print(f"\nConfiguration:")
    print(f"  - Mode: {mode}")
    if args.parallel:
        print(f"  - Workers: {args.max_workers}")
    print(f"  - Checkpoints: {checkpoint_status}")
    print()

    try:
        summary = run_classification_stage(
            client_id=args.client,
            hashtag=args.hashtag,
            analysis_type=args.analysis_type,
            analysis_mode=args.analysis_mode,
            selection_strategy=args.selection_strategy,
            parallel=args.parallel,
            max_workers=args.max_workers,
            checkpoint_enabled=not args.no_checkpoint
        )

        print("\n" + "=" * 80)
        print("✅ STAGE 2.7 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nResults:")
        print(f"  - Total videos: {summary['total']}")
        print(f"  - Successfully classified: {summary['completed']}")
        print(f"  - Failed: {summary['failed']}")
        print(f"  - Mode: {summary['mode']}")
        print(f"  - Duration: {summary['duration_seconds']:.2f}s")

        if summary['failed'] > 0:
            print(f"\n⚠️  {summary['failed']} videos failed classification:")
            for video_id in summary['failed_ids'][:10]:  # Show first 10
                print(f"     - {video_id}")
            if len(summary['failed_ids']) > 10:
                print(f"     ... and {len(summary['failed_ids']) - 10} more")

        print(f"\n📝 Next step: Run Stage 7 (LLM Report Generation)")

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\n   Make sure Stage 2.5 (Bucket Selection) has completed successfully.")
        sys.exit(1)

    except ValueError as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Classification interrupted by user (Ctrl+C)")
        if not args.no_checkpoint:
            print("   ✅ Progress saved to checkpoint - you can resume by re-running this command")
        sys.exit(130)

    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
