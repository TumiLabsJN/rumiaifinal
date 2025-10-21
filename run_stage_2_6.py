#!/usr/bin/env python3
"""
CLI Entry Point for Stage 2.6: Content Pattern Discovery

Source: ContentAnalysisCHILDTI.md Section 6

Usage:
    python run_stage_2_6.py --client acme_corp --hashtag nutrition
    python run_stage_2_6.py --client acme_corp --hashtag nutrition --sample-size 50
"""

import argparse
import logging
import sys
import os

# Add ml_pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml_pipeline'))

from stage2_content_analysis.discovery import run_discovery_stage


def setup_logging():
    """Configure logging for CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Stage 2.6: Content Pattern Discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_stage_2_6.py --client acme_corp --hashtag nutrition

  # Custom sample size
  python run_stage_2_6.py --client acme_corp --hashtag nutrition --sample-size 100

  # Different analysis mode
  python run_stage_2_6.py --client acme_corp --hashtag nutrition --analysis-mode recent

Environment Variables:
  ANTHROPIC_API_KEY: Required - API key for Claude
  RUMIAI_ROOT: Optional - Base directory (default: /home/jorge/rumiaifinal)
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
        '--sample-size',
        type=int,
        default=50,
        help='Number of transcripts to sample (default: 50)'
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    # Setup
    setup_logging()
    args = parse_args()

    # Validate environment
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("❌ ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("   Set it with: export ANTHROPIC_API_KEY=your_api_key")
        sys.exit(1)

    # Clean hashtag (remove # if present)
    hashtag = args.hashtag.lstrip('#')

    # Run Stage 2.6
    try:
        result = run_discovery_stage(
            client_id=args.client,
            hashtag=hashtag,
            analysis_mode=args.analysis_mode,
            selection_strategy=args.selection_strategy,
            sample_size=args.sample_size
        )

        print("\n✅ Stage 2.6 completed successfully!")
        print(f"\nDiscovered patterns:")
        print(f"  - Content categories: {len(result['discovered_patterns']['content_categories'])}")
        print(f"  - Hook strategies: {len(result['discovered_patterns']['hook_strategies'])}")
        print(f"  - Pain points: {len(result['discovered_patterns']['audience_pain_points'])}")
        print(f"  - Keywords: {len(result['discovered_patterns']['trending_keywords'])}")
        print(f"  - Engagement drivers: {len(result['discovered_patterns']['engagement_drivers'])}")
        print(f"  - Content tactics: {len(result['discovered_patterns']['content_tactics'])}")

        print(f"\n📝 Next step: Manually curate the raw discovery file and save as {hashtag}_taxonomy.json")
        print(f"   Then run: python run_stage_2_7.py --client {args.client} --hashtag {hashtag}")

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\n   Make sure Stage 2.5 (Bucket Selection) has completed successfully.")
        sys.exit(1)

    except ValueError as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
