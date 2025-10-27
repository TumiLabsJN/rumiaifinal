#!/usr/bin/env python3
"""
RumiAI ML Batch Processing Pipeline - Main Entry Point

Orchestrates the complete ML pipeline from video discovery to report generation.

Usage:
    python rumiai_ml_batch.py --client acme_corp --target "#nutrition"

For full help:
    python rumiai_ml_batch.py --help

Source: VideoDiscoveryCHILDTI.md + FoundationCHILD.md
"""

import sys
import os
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
def load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

load_env()

from foundation.cli import parse_args
from foundation.config import ConfigManager
from foundation.paths import PathBuilder, sanitize_client_id
from ml_pipeline.stage1_discovery import VideoDiscovery
from ml_pipeline.stage2_processing import stage_2_video_processing_main
from ml_pipeline.stage2_5_organize import stage_2_5_file_organization_main
from ml_pipeline.stage2_content_analysis.discovery import run_discovery_stage
from ml_pipeline.stage2_content_analysis.classification import run_classification_stage
from ml_pipeline.stage2_content_analysis.taxonomy_validation import validate_curated_taxonomy
# Stage 3: Feature Aggregation
from scripts.stage3_aggregation import aggregate_features
# Stage 3.4: Review CSV Generation
from ml_pipeline.stage3_aggregation.review_csv_generator import generate_review_csv_for_bucket
# Stage 4: Feature Transformation
from rumiai_v2.processors.feature_transformation import run_stage4_transformation  # Source: FeatureTransformationTI.md Section 8
# Stage 5: ML Model Training
from rumiai_v2.processors.model_training import (
    run_stage5_training,
    StageInputError,
    InsufficientDataError,
    ModelTrainingError,
    ValidationError
)  # Source: MLModelTrainingCHILDTI.md Section 11.4, Section 6.1 (exception classes)
# Stage 6: ML Analysis Generation
from ml_pipeline.stage6_analysis.ml_analysis_generation import generate_ml_analysis_jsons
# Stage 7: LLM Analysis - Hybrid Two-Phase Approach
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_llm_analysis_main
from config.bucket_definitions import BUCKET_WINDOWS
from pathlib import Path
import json


def setup_logging(client_id: str, target: str):
    """
    Configure logging for the ML pipeline.

    Logs to both console and file.
    """
    # Create logs directory (use local path if /data not accessible)
    data_root = os.getenv("DATA_ROOT", str(Path(__file__).parent / "data"))
    log_dir = Path(data_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_target = target.replace('#', '').replace('@', '')
    log_file = log_dir / f"rumiai_ml_{client_id}_{sanitized_target}_{timestamp}.log"

    # Configure logging
    # force=True ensures reconfiguration even if logging was initialized by imports
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Python 3.8+: Reconfigure even if already initialized
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")

    return logger


def check_taxonomy_exists(analysis_base: Path, hashtag: str) -> bool:
    """Check if curated taxonomy exists for hashtag."""
    taxonomy_path = analysis_base / f"content_taxonomies/{hashtag}_taxonomy.json"
    return taxonomy_path.exists()


def load_content_analysis_state(analysis_base: Path) -> dict:
    """
    Load content analysis state from .content_analysis_state.json.

    Returns minimal state tracking taxonomy lifecycle only.
    Separate from Stage 2 video processing checkpoints.
    """
    state_path = analysis_base / ".content_analysis_state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {
        "taxonomy_discovered": False,
        "taxonomy_curated": False,
        "taxonomy_path": None,
        "discovery_date": None,
        "classification_date": None,
        "last_updated": None
    }


def save_content_analysis_state(analysis_base: Path, state: dict) -> None:
    """
    Save content analysis state to .content_analysis_state.json.

    Atomic write pattern: temp file + replace for crash safety.
    """
    state['last_updated'] = datetime.utcnow().isoformat() + "Z"

    state_path = analysis_base / ".content_analysis_state.json"
    temp_path = state_path.with_suffix('.json.tmp')

    # Write to temp file first
    with open(temp_path, 'w') as f:
        json.dump(state, f, indent=2)

    # Atomic rename (crash-safe)
    temp_path.replace(state_path)


def display_curation_instructions(raw_discovery_path: Path, taxonomy_path: Path) -> None:
    """Display manual curation instructions to user."""
    print("\n" + "="*80)
    print("📋 MANUAL CURATION INSTRUCTIONS")
    print("="*80)
    print(f"\n1. Open the raw discovery file:")
    print(f"   {raw_discovery_path}")
    print(f"\n2. Review and curate the discovered patterns:")
    print(f"   - Remove patterns with <10% frequency")
    print(f"   - Merge similar categories (e.g., 'recipe' + 'cooking_tutorial' → 'recipe_tutorial')")
    print(f"   - Ensure names are snake_case (lowercase, underscores only)")
    print(f"   - Add clear definitions (minimum 10 characters)")
    print(f"   - Remove duplicates")
    print(f"\n3. Save the curated taxonomy as:")
    print(f"   {taxonomy_path}")
    print(f"\n4. Validate your taxonomy (optional but recommended):")
    print(f"   python run_stage_2_7.py --client <CLIENT> --hashtag <HASHTAG> --validate-only")
    print(f"\n5. Common mistakes to avoid:")
    print(f"   ❌ Names with spaces or capitals (use: 'recipe_tutorial' NOT 'Recipe Tutorial')")
    print(f"   ❌ Empty arrays [] (must have at least 1 item per category)")
    print(f"   ❌ Definitions too short (minimum 10 chars)")
    print(f"   ❌ Duplicate names or items")


def validate_stage_6_prerequisites(bucket_path: str) -> None:
    """
    Validate Stage 6 input dependencies exist.
    Source: MLAnalysisGenerationCHILDTI.md Section 3.1 (INPUT FILES)

    Raises:
        FileNotFoundError: If required upstream output missing
    """
    # Get bucket name from path
    bucket_name = Path(bucket_path).name.replace('bucket_', '')

    windows = BUCKET_WINDOWS[bucket_name]

    bucket_path_obj = Path(bucket_path)

    # Required files from TI Section 2.1 (Stage6Input)
    required_files = []

    # Stage 3 output (1 file)
    required_files.append(bucket_path_obj / "ml_analysis" / "aggregated_features.csv")

    # Stage 4 outputs (1 + 2×N files)
    required_files.append(bucket_path_obj / "ml_analysis" / "rf_transformed.csv")
    for window in windows:
        required_files.append(bucket_path_obj / "ml_analysis" / f"{window}_rf_transformed.csv")
        required_files.append(bucket_path_obj / "ml_analysis" / f"{window}_km_transformed.csv")

    # Stage 5 outputs (2 + 4×N files)
    required_files.append(bucket_path_obj / "models" / f"rf_video_{bucket_name}.pkl")
    required_files.append(bucket_path_obj / "models" / "model_metrics.json")
    for window in windows:
        required_files.append(bucket_path_obj / "models" / f"rf_{window}_{bucket_name}.pkl")
        required_files.append(bucket_path_obj / "models" / f"{window}_kmeans_{bucket_name}.pkl")
        required_files.append(bucket_path_obj / "models" / f"{window}_scalers_{bucket_name}.pkl")
        required_files.append(bucket_path_obj / "models" / f"{window}_X_data_{bucket_name}.pkl")

    missing = [str(f) for f in required_files if not f.exists()]

    if missing:
        raise FileNotFoundError(
            f"Stage 6 prerequisites missing ({len(missing)} files):\n" +
            "\n".join(f"  - {f}" for f in missing[:5]) +  # Show first 5
            (f"\n  ... and {len(missing)-5} more" if len(missing) > 5 else "") +
            f"\n\nAction: Ensure Stages 3, 4, and 5 completed successfully"
        )


def validate_stage_6_outputs(bucket_path: str) -> None:
    """
    Validate Stage 6 outputs created correctly.
    Source: MLAnalysisGenerationCHILDTI.md Section 5 (validate_stage_output)

    Raises:
        AssertionError: If output validation fails
    """
    # Get bucket name and windows
    bucket_name = Path(bucket_path).name.replace('bucket_', '')
    windows = BUCKET_WINDOWS[bucket_name]

    bucket_path_obj = Path(bucket_path)
    ml_analysis_dir = bucket_path_obj / "ml_analysis"

    # Validate ml_analysis directory exists
    assert ml_analysis_dir.exists(), \
        f"Stage 6 output directory missing: {ml_analysis_dir}"

    # Validate video-level RF JSON
    video_rf_json = ml_analysis_dir / "rf_video_analysis.json"
    assert video_rf_json.exists(), \
        f"Stage 6 video RF JSON missing: {video_rf_json}"

    # Validate video RF JSON structure
    with open(video_rf_json) as f:
        video_rf_data = json.load(f)

    assert "bucket" in video_rf_data, "Video RF JSON missing 'bucket' field"
    assert "feature_importance" in video_rf_data, "Video RF JSON missing 'feature_importance' field"
    assert isinstance(video_rf_data["feature_importance"], list), \
        "Video RF 'feature_importance' must be a list"
    assert len(video_rf_data["feature_importance"]) <= 10, \
        f"Video RF has {len(video_rf_data['feature_importance'])} features (max 10)"

    # Validate window-level JSONs (2 per window)
    for window in windows:
        # Validate window RF JSON
        window_rf_json = ml_analysis_dir / f"{window}_rf_analysis.json"
        assert window_rf_json.exists(), \
            f"Stage 6 window RF JSON missing: {window_rf_json}"

        with open(window_rf_json) as f:
            window_rf_data = json.load(f)

        assert window_rf_data.get("window_type") == window, \
            f"Window RF JSON has wrong window_type: {window_rf_data.get('window_type')} (expected {window})"

        # Validate K-Means JSON
        window_km_json = ml_analysis_dir / f"{window}_kmeans_analysis.json"
        assert window_km_json.exists(), \
            f"Stage 6 K-Means JSON missing: {window_km_json}"

        with open(window_km_json) as f:
            window_km_data = json.load(f)

        assert "clusters" in window_km_data, f"K-Means JSON missing 'clusters' field"
        assert len(window_km_data["clusters"]) == 3, \
            f"K-Means has {len(window_km_data['clusters'])} clusters (expected 3)"

    # All validations passed
    print(f"  ✓ Output validation passed: {1 + len(windows)*2} JSON files")


def validate_stage7_prerequisites(bucket_path: str, bucket: str) -> None:
    """
    Validate Stage 7 input dependencies exist.
    Source: LLMAnalysisCHILDTI.md Section 12.2 (Upstream TI Requirements)

    Stage 7 requires Stage 6 outputs:
    - rf_video_analysis.json (video-level RF features)
    - {window}_rf_analysis.json (per window RF features)
    - {window}_kmeans_analysis.json (per window K-Means clusters)

    Raises:
        FileNotFoundError: If required upstream output missing
    """
    ml_analysis_dir = os.path.join(bucket_path, "ml_analysis")

    # BUCKET_WINDOWS already imported at top
    window_types = BUCKET_WINDOWS.get(bucket, [])

    if not window_types:
        raise ValueError(f"Invalid bucket: {bucket}")

    required_files = [
        # Video-level RF (cross-window features)
        os.path.join(ml_analysis_dir, "rf_video_analysis.json"),
    ]

    # Window-level files (RF + K-Means for each window)
    for window in window_types:
        required_files.append(os.path.join(ml_analysis_dir, f"{window}_rf_analysis.json"))
        required_files.append(os.path.join(ml_analysis_dir, f"{window}_kmeans_analysis.json"))

    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        raise FileNotFoundError(
            f"Stage 7 prerequisites missing ({len(missing)} files):\n" +
            "\n".join(f"  - {os.path.basename(f)}" for f in missing) +
            f"\n\nAction: Ensure Stage 6 (ML Analysis Generation) completed successfully for bucket {bucket}"
        )

    logger = logging.getLogger(__name__)
    logger.info(f"✓ Stage 7 prerequisites validated for bucket {bucket} ({len(required_files)} files)")


def validate_stage7_outputs(bucket_path: str, bucket: str) -> None:
    """
    Validate Stage 7 outputs created correctly.
    Source: LLMAnalysisCHILDTI.md Section 3 (Stage Contract - StageOutput)

    Validates:
    - Phase 1 window analyses (hook, middle_X, closing)
    - Phase 2 synthesis (cross-window insights)
    - Complete analysis (combined Phase 1 + Phase 2)

    Raises:
        AssertionError: If output validation fails
        FileNotFoundError: If required output missing
    """
    llm_output_dir = os.path.join(bucket_path, "ml_analysis/llm")

    # BUCKET_WINDOWS already imported at top
    window_types = BUCKET_WINDOWS.get(bucket, [])

    if not window_types:
        raise ValueError(f"Invalid bucket: {bucket}")

    # Validate Phase 1 outputs (one per window)
    for window in window_types:
        window_file = os.path.join(llm_output_dir, f"{window}_analysis.json")

        assert os.path.exists(window_file), \
            f"Stage 7 Phase 1 output missing: {window}_analysis.json"

        # Validate JSON structure
        with open(window_file, 'r') as f:
            data = json.load(f)
            assert 'window_type' in data, f"{window}_analysis.json missing 'window_type' field"
            assert 'clusters' in data, f"{window}_analysis.json missing 'clusters' field"
            assert len(data['clusters']) == 3, f"{window}_analysis.json must have exactly 3 clusters"

    # Validate Phase 2 synthesis output
    synthesis_file = os.path.join(llm_output_dir, "synthesis.json")
    assert os.path.exists(synthesis_file), \
        "Stage 7 Phase 2 output missing: synthesis.json"

    with open(synthesis_file, 'r') as f:
        synthesis = json.load(f)
        assert 'winning_formulas' in synthesis, "synthesis.json missing 'winning_formulas'"
        assert 'scenario' in synthesis, "synthesis.json missing 'scenario' field"

    # Validate complete analysis (combined output)
    complete_file = os.path.join(llm_output_dir, "complete_analysis.json")
    assert os.path.exists(complete_file), \
        "Stage 7 complete analysis output missing: complete_analysis.json"

    with open(complete_file, 'r') as f:
        complete = json.load(f)
        assert 'phase1_window_analyses' in complete
        assert 'phase2_synthesis' in complete
        assert 'bucket' in complete
        assert complete['bucket'] == bucket

    logger = logging.getLogger(__name__)
    logger.info(f"✓ Stage 7 outputs validated for bucket {bucket} (Phase 1: {len(window_types)} windows, Phase 2: synthesis, Complete: 1 file)")


def handle_stage7_error(error: Exception, bucket_path: str) -> None:
    """
    Handle Stage 7 errors.
    Source: LLMAnalysisCHILDTI.md Section 6 (Error Handling)

    Stage 7 has 3 main error categories:
    - LLMValidationError: Invalid LLM response (wrong schema, missing fields)
    - Phase1ExecutionError: Phase 1 window analysis failure
    - InsufficientDataError: <3 cluster paths meet 10% threshold (fallback to feature-based reports)

    Note: Stage 7 raises exceptions with error codes in message strings.
    This error matching pattern is specific to current Stage 7 implementation.
    """
    logger = logging.getLogger(__name__)

    # LLM Validation Errors (malformed API responses)
    if "Missing 'clusters' key" in str(error) or "Expected 3 clusters" in str(error):
        logger.error("Stage 7 Error: LLM response validation failed")
        logger.error("Issue: Claude API returned malformed response (wrong schema)")
        logger.error("Action: Check LLM prompt construction and API response parsing")
        logger.error("Retry Policy: Automatic retry with exponential backoff [0s, 2s, 4s]")

    # Phase 1 Execution Errors (window analysis failures)
    elif "Phase1ExecutionError" in str(type(error).__name__):
        logger.error("Stage 7 Error: Phase 1 window analysis failed")
        logger.error(f"Details: {error}")
        logger.error("Action: Check .phase1_status.json for partial progress, resume from checkpoint")

    # Insufficient Data Errors (fallback scenario)
    elif "InsufficientDataError" in str(error):
        logger.error("Stage 7 Error: Insufficient cluster paths (<3 paths meet 10% threshold)")
        logger.error("Issue: Not enough common viral patterns detected")
        logger.error("Action: System will automatically use feature-based fallback reports")
        logger.info("Note: This is not a failure - fallback strategy will generate insights from RF features")

    # API Errors (authentication, rate limits, timeouts)
    elif "401" in str(error):
        logger.error("Stage 7 Error: Claude API authentication failed")
        logger.error("Issue: ANTHROPIC_API_KEY invalid or missing")
        logger.error("Action: Verify ANTHROPIC_API_KEY in .env file")
        logger.error("Retry Policy: NO RETRY (non-retryable error)")

    elif "429" in str(error) or "503" in str(error):
        logger.error(f"Stage 7 Error: Claude API {error}")
        logger.error("Issue: Rate limit exceeded or service unavailable")
        logger.error("Retry Policy: Automatic retry with exponential backoff [0s, 2s, 4s]")

    else:
        logger.error(f"Stage 7 Error: Unexpected error - {error}")
        logger.error("Action: Check logs for full traceback")

    # Cleanup partial outputs (only if catastrophic failure)
    if not isinstance(error, (FileNotFoundError, ValueError)):
        cleanup_stage7_partial_outputs(bucket_path)


def cleanup_stage7_partial_outputs(bucket_path: str) -> None:
    """
    Remove partial outputs from failed Stage 7 execution.
    Source: LLMAnalysisCHILDTI.md Section 3 StageOutput

    Note: Stage 7 has checkpoint/resume capability (.phase1_status.json).
    Only cleanup if catastrophic failure (not for recoverable errors).
    """
    logger = logging.getLogger(__name__)
    llm_output_dir = os.path.join(bucket_path, "ml_analysis/llm")

    if not os.path.exists(llm_output_dir):
        return

    # List of output files to clean up (all Phase 1 + Phase 2 outputs)
    partial_files = [
        "synthesis.json",
        "complete_analysis.json",
        ".phase1_status.json"
    ]

    # Add window-specific files (variable count based on bucket)
    try:
        for filename in os.listdir(llm_output_dir):
            if filename.endswith("_analysis.json") and filename != "complete_analysis.json":
                partial_files.append(filename)
    except Exception:
        pass

    for filename in partial_files:
        file_path = os.path.join(llm_output_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up partial output: {filename}")
            except Exception as e:
                logger.warning(f"Failed to clean up {filename}: {e}")


def main():
    """
    Main entry point for RumiAI ML Batch Pipeline.

    Pipeline Stages:
    - Stage 0: Foundation (CLI parsing, config, directory setup)
    - Stage 1: Video Discovery & Selection
    - Stage 2: Video Processing (RumiAI feature extraction)
    - Stage 2.5: File Organization
    - Stage 2.6: Content Analysis - Pattern Discovery
    - Stage 2.7: Content Analysis - Video Classification
    - Stage 3: Feature Aggregation
    - Stage 4: Feature Transformation
    - Stage 5: ML Model Training
    - Stage 6: ML Analysis Generation
    - Stage 7: LLM Report Generation (TODO)

    Exit Codes:
    - 0: Success (pipeline completed fully)
    - 1: Error (validation failure, missing inputs, etc.)
    - 2: Paused for manual curation (Stage 2.6 complete, awaiting user)
    - 4: I/O failure (Stage 4: disk full, permission denied)
    - 8: Timeout (Stage 4: processing exceeded limits)
    - 99: Unexpected error (Stage 4)
    - 130: Interrupted by user (Ctrl+C)
    """
    try:
        # ===== STAGE 0: FOUNDATION =====
        print("="*80)
        print("RumiAI ML BATCH PIPELINE")
        print("="*80)
        print()

        # Stage 0.1: Parse CLI arguments
        print("Stage 0: Foundation - Parsing CLI arguments...")
        cli_args = parse_args()
        print(f"✓ Parsed CLI arguments")

        # Stage 0.2: Validate environment
        print("Stage 0: Foundation - Validating environment...")
        apify_api_key = os.getenv("APIFY_API_KEY")
        if not apify_api_key:
            print("✗ ERROR: APIFY_API_KEY environment variable not set")
            print("  Obtain API key from: https://console.apify.com/account/integrations")
            print("  Set with: export APIFY_API_KEY='your_key_here'")
            return 1
        print(f"✓ Environment validated")

        # Stage 0.3: Setup logging
        logger = setup_logging(cli_args.client, cli_args.target)
        logger.info("="*80)
        logger.info("RumiAI ML BATCH PIPELINE STARTED")
        logger.info("="*80)

        # Stage 0.4: Create Config object and paths
        print("Stage 0: Foundation - Creating configuration and paths...")
        config = ConfigManager.from_cli_args(cli_args)

        path_builder = PathBuilder()
        analysis_base = path_builder.get_target_dir(
            client_id=sanitize_client_id(cli_args.client),
            analysis_type=cli_args.analysis_type,
            target=cli_args.target,
            analysis_mode=cli_args.analysis_mode,
            selection_strategy=cli_args.selection_strategy
        )
        analysis_base.mkdir(parents=True, exist_ok=True)

        print(f"✓ Created directory structure: {analysis_base}")
        logger.info(f"Created directory structure: {analysis_base}")

        # Stage 0.5: Save configuration
        print("Stage 0: Foundation - Saving configuration...")
        config_path = analysis_base / "config.json"
        ConfigManager.save(config, config_path)
        print(f"✓ Saved configuration: {config_path}")
        logger.info(f"Saved configuration: {config_path}")

        print()

        # ===== STAGE 1: VIDEO DISCOVERY & SELECTION =====
        logger.info("Starting Stage 1: Video Discovery & Selection")

        # Define checkpoint path (consistent with Stages 4-7)
        stage1_checkpoint_path = analysis_base / "checkpoints" / "stage_1_checkpoint.json"

        # Check if Stage 1 already complete
        if stage1_checkpoint_path.exists():
            logger.info(f"Stage 1 checkpoint found: {stage1_checkpoint_path}")

            try:
                # Load checkpoint
                with open(stage1_checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)

                # Validate checkpoint schema (required fields)
                required_fields = ["winning_buckets", "output_files", "completion_timestamp"]
                missing_fields = [field for field in required_fields if field not in checkpoint]

                if missing_fields:
                    # Corrupt checkpoint: missing required fields
                    logger.warning(f"Checkpoint corrupt (missing fields: {missing_fields}). Deleting and re-running Stage 1.")
                    stage1_checkpoint_path.unlink()
                    raise ValueError("Checkpoint validation failed")

                # Verify all output files exist
                output_files = checkpoint["output_files"]
                missing_files = [f for f in output_files if not Path(f).exists()]

                if missing_files:
                    # Incomplete outputs: some files missing
                    logger.warning(f"Stage 1 outputs incomplete ({len(missing_files)} files missing). Re-running Stage 1.")
                    logger.debug(f"Missing files: {missing_files}")
                    stage1_checkpoint_path.unlink()
                    raise ValueError("Output files missing")

                # Checkpoint valid - skip Stage 1
                winning_buckets = checkpoint["winning_buckets"]

                logger.info("Stage 1 already complete (checkpoint valid)")
                print("\n" + "="*80)
                print("STAGE 1: VIDEO DISCOVERY & SELECTION")
                print("="*80)
                print(f"\n✓ Stage 1: Video Discovery - SKIPPED (already complete)")
                print(f"  Winning buckets: {', '.join(winning_buckets)}\n")

                # Skip to Stage 2 (winning_buckets loaded from checkpoint)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Checkpoint invalid (JSON error, validation error, or missing key)
                logger.warning(f"Checkpoint invalid ({type(e).__name__}: {e}). Deleting and re-running Stage 1.")
                if stage1_checkpoint_path.exists():
                    stage1_checkpoint_path.unlink()
                # Fall through to run Stage 1

        # If checkpoint doesn't exist OR was deleted above, run Stage 1
        if not stage1_checkpoint_path.exists():
            logger.info("Running Stage 1: Video Discovery & Selection")
            print("\n" + "="*80)
            print("STAGE 1: VIDEO DISCOVERY & SELECTION")
            print("="*80)

            # Run full Stage 1
            video_discovery = VideoDiscovery(
                config=config.model_dump(),  # Convert pydantic model to dict
                apify_api_key=apify_api_key,
                path_builder=path_builder
            )

            exit_code = video_discovery.run()

            if exit_code != 0:
                logger.error(f"Stage 1 failed with exit code {exit_code}")
                return exit_code

            logger.info("Stage 1 completed successfully")

            # Load winning buckets from newly created files
            winner_analysis_path = analysis_base / "winner_analysis.json"
            with open(winner_analysis_path, 'r') as f:
                winner_analysis = json.load(f)
            winning_buckets = winner_analysis['top_3_buckets']

            # ===== CREATE CHECKPOINT =====
            logger.info("Creating Stage 1 checkpoint")

            # Build output file list
            output_files = [
                str(analysis_base / "winner_analysis.json")
            ]

            # Add selected_videos.json for each winning bucket
            for bucket in winning_buckets:
                selected_videos_path = analysis_base / f"buckets/bucket_{bucket}/selected_videos.json"
                output_files.append(str(selected_videos_path))

            # Add cluster_analytics.json if cluster mode
            # Cluster mode detection: hashtag analysis with target NOT starting with #
            is_cluster_mode = (
                config.analysis_type == "hashtag" and
                not config.target.startswith("#")
            )

            if is_cluster_mode:
                cluster_analytics_path = analysis_base / "cluster_analytics.json"
                if cluster_analytics_path.exists():
                    output_files.append(str(cluster_analytics_path))
                    logger.info("Cluster mode detected: Added cluster_analytics.json to checkpoint")

            # Create checkpoint data
            checkpoint_data = {
                "stage": "stage_1_video_discovery",
                "completion_timestamp": datetime.now(timezone.utc).isoformat(),
                "winning_buckets": winning_buckets,
                "output_files": output_files,
                "analysis_mode": config.analysis_mode,
                "analysis_type": config.analysis_type,
                "target": config.target,
                "video_count": config.video_count,
            }

            # Ensure checkpoints directory exists
            checkpoint_dir = analysis_base / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Write checkpoint
            try:
                with open(stage1_checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                logger.info(f"Stage 1 checkpoint created: {stage1_checkpoint_path}")
            except (IOError, PermissionError) as e:
                # Checkpoint creation failed (non-fatal warning)
                logger.warning(f"Failed to create Stage 1 checkpoint: {e}")
                logger.warning("Pipeline will re-run Stage 1 on next resume")

            print(f"\n✓ Stage 1: Video Discovery - COMPLETE")
            print(f"  Winning buckets: {', '.join(winning_buckets)}\n")

        # ===== STAGE 2: VIDEO PROCESSING =====
        logger.info("Starting Stage 2: Video Processing")
        print("\n" + "="*80)
        print("STAGE 2: VIDEO PROCESSING")
        print("="*80)

        # Load winner_analysis.json to get winning buckets
        winner_analysis_path = analysis_base / "winner_analysis.json"
        if not winner_analysis_path.exists():
            logger.error("winner_analysis.json not found - Stage 1 may have failed")
            return 1

        with open(winner_analysis_path) as f:
            winner_analysis = json.load(f)

        winning_buckets = winner_analysis['top_3_buckets']
        logger.info(f"Processing {len(winning_buckets)} winning buckets: {winning_buckets}")
        print(f"\nWinning buckets: {', '.join(winning_buckets)}")

        # Process each winning bucket
        stage2_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 2 for bucket: {bucket_name}")
            print(f"\n--- Processing bucket: {bucket_name} ---")

            # Load selected videos for this bucket
            bucket_videos_path = analysis_base / f"buckets/bucket_{bucket_name}/selected_videos.json"
            if not bucket_videos_path.exists():
                logger.warning(f"No selected videos found for bucket {bucket_name}, skipping")
                print(f"⚠️  No selected videos found for {bucket_name}, skipping")
                continue

            with open(bucket_videos_path) as f:
                video_data = json.load(f)
                video_list = video_data['videos']  # Extract just the videos array

            print(f"Processing {len(video_list)} videos for bucket {bucket_name}...")

            # Run Stage 2 video processing for this bucket
            try:
                summary = stage_2_video_processing_main(
                    config=config.model_dump(),
                    video_list=video_list,
                    bucket_name=bucket_name,
                    enable_pause_support=True  # Allow Ctrl+C graceful pause
                )

                stage2_summaries[bucket_name] = summary
                logger.info(f"Bucket {bucket_name} complete: {summary['completed']}/{summary['total']} videos processed")
                print(f"✓ Bucket {bucket_name}: {summary['completed']}/{summary['total']} videos processed")
                if summary['failed'] > 0:
                    print(f"  ⚠️  {summary['failed']} videos failed")

            except Exception as e:
                logger.error(f"Stage 2 failed for bucket {bucket_name}: {e}", exc_info=True)
                print(f"✗ Bucket {bucket_name} failed: {e}")
                # Continue with other buckets (skip-on-fail policy)
                continue

        logger.info("Stage 2 completed for all buckets")
        print("\n✓ Stage 2: Video Processing - COMPLETE")

        # Log Stage 2 summary
        total_videos = sum(s['total'] for s in stage2_summaries.values())
        completed_videos = sum(s['completed'] for s in stage2_summaries.values())
        failed_videos = sum(s['failed'] for s in stage2_summaries.values())
        logger.info(f"Stage 2 Summary: {completed_videos}/{total_videos} videos processed, {failed_videos} failed")
        print(f"Summary: {completed_videos}/{total_videos} videos processed, {failed_videos} failed")

        # ===== STAGE 2.5: FILE ORGANIZATION =====
        logger.info("Starting Stage 2.5: File Organization")
        print("\n" + "="*80)
        print("STAGE 2.5: FILE ORGANIZATION")
        print("="*80)

        try:
            # Stage 2.5 organizes all temporal_windows files from flat directory
            # into bucket-specific directories
            organization_summary = stage_2_5_file_organization_main(
                analysis_base=str(analysis_base)
            )

            logger.info("Stage 2.5 completed successfully")
            logger.info(f"Moved: {organization_summary['moved_count']} files")
            logger.info(f"Skipped (already organized): {organization_summary['skipped_already_organized']} files")
            logger.info(f"Missing: {organization_summary['missing_count']} files")

            print(f"\n✓ Stage 2.5: File Organization - COMPLETE")
            print(f"  Organized {organization_summary['moved_count']} temporal_windows files")
            if organization_summary['skipped_already_organized'] > 0:
                print(f"  Skipped {organization_summary['skipped_already_organized']} files (already organized)")
            if organization_summary['missing_count'] > 0:
                print(f"  ⚠️  {organization_summary['missing_count']} files missing")

        except Exception as e:
            logger.error(f"Stage 2.5 failed: {e}", exc_info=True)
            print(f"\n✗ Stage 2.5 failed: {e}")
            return 1

        # ===== STAGE 2.6 & 2.7: CONTENT ANALYSIS =====
        logger.info("Starting Stage 2.6/2.7: Content Analysis")
        print("\n" + "="*80)
        print("STAGE 2.6/2.7: CONTENT ANALYSIS")
        print("="*80)

        # Sanitize target name (remove # and @ prefixes)
        from foundation.paths import sanitize_target
        target_sanitized = sanitize_target(cli_args.target, cli_args.analysis_type)

        # Construct taxonomy path
        taxonomy_dir = analysis_base / "content_taxonomies"
        taxonomy_dir.mkdir(parents=True, exist_ok=True)
        taxonomy_path = taxonomy_dir / f"{target_sanitized}_taxonomy.json"

        # Load or initialize content analysis state
        ca_state = load_content_analysis_state(analysis_base)

        # === STAGE 2.6: PATTERN DISCOVERY ===
        if not taxonomy_path.exists():
            # Taxonomy doesn't exist - run discovery
            logger.info("Taxonomy not found - running Stage 2.6: Pattern Discovery")
            print("\n--- Stage 2.6: Pattern Discovery (One-Time Setup) ---")
            print(f"Discovering content patterns from sample transcripts...")

            try:
                # Run discovery
                raw_taxonomy = run_discovery_stage(
                    client_id=sanitize_client_id(cli_args.client),
                    hashtag=cli_args.target,  # Pass original target with prefix
                    analysis_type=cli_args.analysis_type,
                    analysis_mode=cli_args.analysis_mode,
                    selection_strategy=cli_args.selection_strategy,
                    sample_size=50  # Default from TI spec
                )

                # Update content analysis state
                ca_state['taxonomy_discovered'] = True
                ca_state['taxonomy_curated'] = False
                ca_state['taxonomy_path'] = f"content_taxonomies/{target_sanitized}_taxonomy.json"
                ca_state['discovery_date'] = datetime.utcnow().isoformat() + "Z"
                save_content_analysis_state(analysis_base, ca_state)

                # Display curation instructions
                raw_discovery_path = taxonomy_dir / f"{target_sanitized}_raw_discovery.json"
                display_curation_instructions(raw_discovery_path, taxonomy_path)

                logger.info("Stage 2.6 complete - awaiting manual curation")

                # EXIT WITH CODE 2 - Paused for manual step (C3 Resolution)
                print("\n" + "="*80)
                print("✅ Stage 2.6 Discovery Complete!")
                print("="*80)
                print("\n📋 NEXT STEP: Manual curation required (estimated time: 1-3 hours depending on complexity)")
                print(f"\nAfter curation, re-run this command to continue:")
                print(f"  python rumiai_ml_batch.py --client {cli_args.client} --target {cli_args.target}")
                print()

                logger.info("="*80)
                logger.info("PIPELINE PAUSED FOR MANUAL CURATION (Exit Code 2)")
                logger.info("="*80)

                return 2  # Exit code 2 = paused for manual step

            except FileNotFoundError as e:
                logger.error(f"Stage 2.6 failed - missing required input: {e}")
                print(f"\n✗ Stage 2.6 failed: {e}")
                print("   Ensure Stage 2.5 completed successfully (selection_manifest.json must exist)")
                return 1

            except Exception as e:
                logger.error(f"Stage 2.6 failed: {e}", exc_info=True)
                print(f"\n✗ Stage 2.6 failed: {e}")
                return 1

        else:
            # Taxonomy exists - skip discovery
            logger.info("Taxonomy found - skipping Stage 2.6 (already complete)")
            print("\n✓ Stage 2.6: Pattern Discovery - SKIPPED (taxonomy exists)")
            ca_state['taxonomy_discovered'] = True
            ca_state['taxonomy_curated'] = True  # Assumed true if taxonomy exists

        # === STAGE 2.7: VIDEO CLASSIFICATION ===
        logger.info("Starting Stage 2.7: Video Classification")
        print("\n--- Stage 2.7: Video Classification ---")

        try:
            # Step 1: Validate taxonomy
            print("Validating taxonomy...")
            validate_curated_taxonomy(str(taxonomy_path))
            print("✓ Taxonomy validation passed")
            logger.info("Taxonomy validation passed")

            # Step 2: Determine classification mode
            # Use environment variable or default to sequential (M4 Resolution)
            enable_parallel = os.environ.get('ENABLE_PARALLEL_CLASSIFICATION', 'false').lower() == 'true'
            max_workers = int(os.environ.get('MAX_CLASSIFICATION_WORKERS', '5'))

            mode_str = f"parallel ({max_workers} workers)" if enable_parallel else "sequential"
            print(f"Classification mode: {mode_str}")

            # Step 3: Run classification
            print(f"Classifying videos across {len(winning_buckets)} buckets...")

            summary = run_classification_stage(
                client_id=sanitize_client_id(cli_args.client),
                hashtag=cli_args.target,  # Pass original target with prefix
                analysis_type=cli_args.analysis_type,
                analysis_mode=cli_args.analysis_mode,
                selection_strategy=cli_args.selection_strategy,
                parallel=enable_parallel,
                max_workers=max_workers,
                checkpoint_enabled=True  # Enable checkpoint/resume
            )

            # Step 4: Update content analysis state
            ca_state['classification_date'] = datetime.utcnow().isoformat() + "Z"
            save_content_analysis_state(analysis_base, ca_state)

            # Step 5: Log results
            logger.info(f"Stage 2.7 complete: {summary['completed']}/{summary['total']} videos classified in {summary['duration_seconds']:.2f}s")
            print(f"\n✓ Stage 2.7: Classified {summary['completed']}/{summary['total']} videos in {summary['duration_seconds']:.2f}s")

            if summary['failed'] > 0:
                print(f"  ⚠️  {summary['failed']} videos failed classification")
                logger.warning(f"{summary['failed']} videos failed classification: {summary['failed_ids']}")

        except FileNotFoundError as e:
            logger.error(f"Stage 2.7 failed - taxonomy not found: {e}")
            print(f"\n✗ Stage 2.7 failed: Taxonomy not found")
            print(f"   Expected: {taxonomy_path}")
            print("\n   This should not happen - please report this bug")
            return 1

        except ValueError as e:
            # Taxonomy validation failed
            logger.error(f"Stage 2.7 failed - invalid taxonomy: {e}")
            print(f"\n✗ Stage 2.7 failed: Invalid taxonomy")
            print(f"\n   Validation error: {e}")
            print(f"\n   Fix the errors in: {taxonomy_path}")
            print(f"   Then re-run this command")
            return 1

        except KeyboardInterrupt:
            print("\n\n⚠️  Classification interrupted (Ctrl+C)")
            print(f"   Progress saved to checkpoint - re-run to resume")
            logger.warning("Stage 2.7 interrupted by user - checkpoint saved")
            return 130

        except Exception as e:
            logger.error(f"Stage 2.7 failed: {e}", exc_info=True)
            print(f"\n✗ Stage 2.7 failed: {e}")
            return 1

        print("\n✓ Stage 2.6/2.7: Content Analysis - COMPLETE")

        # ===== STAGE 3: FEATURE AGGREGATION =====
        logger.info("Starting Stage 3: Feature Aggregation")
        print("\n" + "="*80)
        print("STAGE 3: FEATURE AGGREGATION")
        print("="*80)

        # Process each winning bucket
        stage3_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 3 for bucket: {bucket_name}")
            print(f"\n--- Aggregating features for bucket: {bucket_name} ---")

            bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

            # Check if Stage 3 already complete for this bucket
            checkpoint_path = bucket_path / "checkpoints" / "stage_3_checkpoint.json"
            if checkpoint_path.exists():
                logger.info(f"Bucket {bucket_name}: Stage 3 already complete (checkpoint exists)")
                print(f"✓ Bucket {bucket_name}: Aggregation already complete (skipping)")

                try:
                    # Load checkpoint for summary reporting
                    with open(checkpoint_path) as f:
                        checkpoint = json.load(f)

                    # Validate checkpoint has required fields
                    if checkpoint.get("status") != "completed":
                        logger.warning(f"Bucket {bucket_name}: Stage 3 checkpoint status != completed. Re-running.")
                        checkpoint_path.unlink()  # Delete invalid checkpoint
                    else:
                        stage3_summaries[bucket_name] = {
                            "videos_processed": checkpoint.get("videos_processed", 0),
                            "videos_skipped": checkpoint.get("videos_skipped", 0),
                            "output_csv": {
                                "columns": checkpoint.get("feature_count", "unknown")
                            },
                            "duration_seconds": checkpoint.get("duration_seconds", 0)
                        }
                        continue  # Skip to next bucket

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Bucket {bucket_name}: Invalid checkpoint ({e}). Re-running Stage 3.")
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    # Fall through to run Stage 3

            try:
                # Validate bucket structure exists (from Stage 2.5)
                insights_dir = bucket_path / "analysis" / "insights"
                if not insights_dir.exists():
                    logger.error(
                        f"Bucket {bucket_name}: Insights directory missing ({insights_dir}). "
                        "Stage 2.5 may have failed."
                    )
                    print(f"✗ Bucket {bucket_name}: Missing insights directory (skipping)")
                    continue

                json_count = len(list(insights_dir.glob("*_temporal_windows_updated.json")))
                if json_count == 0:
                    logger.warning(
                        f"Bucket {bucket_name}: No temporal_windows_updated.json files found. "
                        "Skipping."
                    )
                    print(f"⚠️  Bucket {bucket_name}: No JSON files (skipping)")
                    continue

                print(f"Found {json_count} temporal_windows files")

                # Execute Stage 3 aggregation
                # Source: FeatureAggregationTI.md Section 4.2
                csv_path, summary_path = aggregate_features(str(bucket_path))

                # Load summary for reporting
                with open(summary_path) as f:
                    summary = json.load(f)

                # ===== STAGE 3.4: REVIEW CSV GENERATION =====
                # Generate review CSV for manual outlier investigation
                # Source: ReviewCSVGenerationCHILD.md Section 2.3
                try:
                    generate_review_csv_for_bucket(bucket_path)

                    logger.info(f"Bucket {bucket_name}: Generated video_review.csv")
                    print(f"  ✓ Review CSV: validation/video_review.csv")

                except ValueError as e:
                    # All videos missing url (non-fatal - Stage 2 modification not deployed?)
                    # Source: ReviewCSVGenerationCHILD.md Section 6.2 Error 2
                    logger.warning(f"Bucket {bucket_name}: {e}")
                    print(f"  ⚠️  Review CSV not generated (all videos missing url)")
                    # Continue pipeline - review CSV is optional

                except (IOError, OSError) as e:
                    # I/O failure (disk full, permission denied) - non-fatal for review CSV
                    # Source: ReviewCSVGenerationCHILD.md Section 6.2 Error 3
                    logger.error(f"Bucket {bucket_name}: Failed to generate review CSV: {e}")
                    print(f"  ⚠️  Review CSV generation failed: {e}")
                    # Continue pipeline - review CSV is optional

                except Exception as e:
                    # Unexpected error - log but don't fail pipeline
                    logger.error(
                        f"Bucket {bucket_name}: Unexpected error in Stage 3.4: {e}",
                        exc_info=True
                    )
                    print(f"  ⚠️  Review CSV generation failed (unexpected error)")
                    # Continue pipeline - review CSV is optional

                stage3_summaries[bucket_name] = summary

                logger.info(
                    f"Bucket {bucket_name} complete: "
                    f"{summary['videos_processed']}/{summary['input_files_found']} videos aggregated"
                )
                print(
                    f"✓ Bucket {bucket_name}: {summary['videos_processed']} videos → "
                    f"{summary['output_csv']['columns']} features"
                )

                # Warn if videos were skipped
                if summary['videos_skipped'] > 0:
                    print(f"  ⚠️  {summary['videos_skipped']} videos skipped")
                    if summary['skipped_reasons']:
                        print(f"     Reasons: {summary['skipped_reasons']}")

            except ValueError as e:
                # Pre-flight validation failed (invalid inputs, missing files)
                # Source: FeatureAggregationTI.md Section 7.1
                logger.error(f"Stage 3 validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name} validation failed: {e}")
                print("   This indicates Stage 2.5 incomplete or corrupted data")
                return 1  # Exit code 1 = validation failure

            except AssertionError as e:
                # Output validation failed (schema mismatch, column count error)
                # Source: FeatureAggregationTI.md Section 7.1
                logger.error(f"Stage 3 output validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name} output validation failed: {e}")
                print("   This indicates a bug in feature extraction logic")
                return 3  # Exit code 3 = output validation failure

            except (IOError, OSError) as e:
                # I/O failure (disk full, permission denied)
                # Source: FeatureAggregationTI.md Appendix B
                logger.error(f"Stage 3 I/O error for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name} I/O error: {e}")
                print("   Check disk space and permissions")
                return 4  # Exit code 4 = I/O failure

            except Exception as e:
                # Unexpected error
                logger.error(
                    f"Stage 3 unexpected error for bucket {bucket_name}: {e}",
                    exc_info=True
                )
                print(f"✗ Bucket {bucket_name} unexpected error: {e}")
                return 99  # Exit code 99 = unexpected error

        logger.info("Stage 3 completed for all buckets")
        print("\n✓ Stage 3: Feature Aggregation - COMPLETE")

        # Log Stage 3 summary
        if stage3_summaries:
            total_aggregated = sum(s['videos_processed'] for s in stage3_summaries.values())
            total_skipped = sum(s['videos_skipped'] for s in stage3_summaries.values())
            logger.info(
                f"Stage 3 Summary: {total_aggregated} videos aggregated, "
                f"{total_skipped} skipped across {len(stage3_summaries)} buckets"
            )
            print(
                f"Summary: {total_aggregated} videos aggregated across "
                f"{len(stage3_summaries)} buckets"
            )
        else:
            logger.warning("Stage 3 completed but no buckets were processed")
            print("⚠️  No buckets processed in Stage 3 (all skipped)")

        # ===== STAGE 4: FEATURE TRANSFORMATION =====
        logger.info("Starting Stage 4: Feature Transformation")
        print("\n" + "="*80)
        print("STAGE 4: FEATURE TRANSFORMATION")
        print("="*80)

        # Process each winning bucket
        stage4_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 4 for bucket: {bucket_name}")
            print(f"\n--- Transforming features for bucket: {bucket_name} ---")

            bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

            try:
                # Validate Stage 3 completed successfully
                # Checkpoint = performance optimization, CSV = source of truth
                # Fallback to CSV validation if checkpoint missing
                stage3_checkpoint = bucket_path / "checkpoints" / "stage_3_checkpoint.json"
                aggregated_csv = bucket_path / "ml_analysis" / "aggregated_features.csv"

                if stage3_checkpoint.exists():
                    # Primary path: checkpoint exists
                    with open(stage3_checkpoint) as f:
                        stage3_status = json.load(f)

                    if stage3_status.get("status") != "completed":
                        logger.error(f"Bucket {bucket_name}: Stage 3 status={stage3_status.get('status')}")
                        print(f"✗ Bucket {bucket_name}: Stage 3 incomplete (skipping)")
                        continue

                    logger.info(f"Bucket {bucket_name}: Stage 3 validated via checkpoint")

                elif aggregated_csv.exists():
                    # Fallback path: checkpoint missing but CSV exists
                    # This happens if checkpoint write failed but Stage 3 outputs are valid
                    logger.warning(
                        f"Bucket {bucket_name}: Stage 3 checkpoint missing, "
                        f"validating via CSV existence as fallback"
                    )

                    # Basic validation: CSV exists and is non-empty
                    csv_size = aggregated_csv.stat().st_size
                    if csv_size > 0:
                        logger.info(
                            f"Bucket {bucket_name}: Stage 3 CSV validated "
                            f"({csv_size} bytes) - proceeding with Stage 4"
                        )
                        print(f"⚠️  Bucket {bucket_name}: Using Stage 3 CSV (checkpoint missing, skip logic disabled)")
                    else:
                        logger.error(f"Bucket {bucket_name}: Stage 3 CSV is empty")
                        print(f"✗ Bucket {bucket_name}: Stage 3 CSV invalid (skipping)")
                        continue

                else:
                    # Both missing: Stage 3 truly failed or didn't run
                    logger.error(f"Bucket {bucket_name}: Stage 3 checkpoint AND CSV missing")
                    print(f"✗ Bucket {bucket_name}: Stage 3 not complete (skipping)")
                    continue

                # Check if Stage 4 already complete for this bucket
                checkpoint_path = bucket_path / "checkpoints" / "stage_4_checkpoint.json"
                if checkpoint_path.exists():
                    logger.info(f"Bucket {bucket_name}: Stage 4 already complete (checkpoint exists)")
                    print(f"✓ Bucket {bucket_name}: Transformation already complete (skipping)")

                    # Load checkpoint to get output file list
                    with open(checkpoint_path) as f:
                        checkpoint = json.load(f)

                    stage4_summaries[bucket_name] = {
                        "output_files": checkpoint["output_files"],
                        "elapsed_time": 0.0  # Skipped, no time
                    }
                    continue

                # Validate prerequisites (aggregated_features.csv exists)
                aggregated_csv = bucket_path / "ml_analysis" / "aggregated_features.csv"
                if not aggregated_csv.exists():
                    logger.error(
                        f"Bucket {bucket_name}: aggregated_features.csv missing ({aggregated_csv}). "
                        "Stage 3 may have failed."
                    )
                    print(f"✗ Bucket {bucket_name}: Missing aggregated CSV (skipping)")
                    continue

                # Load config for this bucket
                bucket_config = {
                    "strategy": config.selection_strategy,
                    "video_count": config.video_count
                }

                # Execute Stage 4 transformation (from TI Section 8: ENTRY_FUNCTION)
                # Source: FeatureTransformationTI.md Appendix A
                # Note: success is always True when function returns (errors raise exceptions)
                success, output_files, elapsed_time = run_stage4_transformation(
                    bucket_path=str(bucket_path),  # TI Section 2 StageInput param 1
                    config=bucket_config            # TI Section 2 StageInput param 2
                )

                stage4_summaries[bucket_name] = {
                    "output_files": output_files,
                    "elapsed_time": elapsed_time
                }

                logger.info(
                    f"Bucket {bucket_name} complete: "
                    f"{len(output_files)} files generated in {elapsed_time:.1f}s"
                )
                print(
                    f"✓ Bucket {bucket_name}: {len(output_files)} transformation files → "
                    f"{elapsed_time:.1f}s"
                )

            except ValueError as e:
                # Input validation failed (invalid schema, NaN values, out-of-range)
                # Source: FeatureTransformationTI.md Section 6 (Error Case 6)
                # Strategy: Skip bucket, continue pipeline (bucket-specific data error)
                logger.error(f"Stage 4 validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Data validation failed (skipping)")
                print("   This indicates Stage 3 produced invalid data for this bucket")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except AssertionError as e:
                # Output validation failed (wrong schema, column count mismatch)
                # Source: FeatureTransformationTI.md Section 6 (Error Case 3)
                # Strategy: Skip bucket, continue pipeline (likely bucket-specific issue)
                logger.error(f"Stage 4 output validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Output validation failed (skipping)")
                print("   This indicates a transformation issue for this bucket")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except FileNotFoundError as e:
                # Missing upstream file (Stage 3 checkpoint or aggregated CSV)
                # Source: FeatureTransformationTI.md Section 6 (Error Case 1)
                # Strategy: Skip bucket, continue pipeline (bucket-specific missing data)
                logger.error(f"Stage 4 prerequisite missing for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Prerequisites missing (skipping)")
                print("   Ensure Stage 3 completed successfully for this bucket")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except (IOError, OSError) as e:
                # I/O failure (disk full, permission denied)
                # Source: FeatureTransformationTI.md Section 6 (Error Case 4)
                # Strategy: Exit pipeline (system-wide issue affects all buckets)
                logger.error(f"Stage 4 I/O error for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: I/O error (exiting pipeline)")
                print("   Check disk space and permissions")
                print("   This is a system-wide issue - stopping pipeline")
                return 4  # Exit code 4 = I/O failure

            except TimeoutError as e:
                # Processing exceeded 5-minute timeout
                # Source: FeatureTransformationTI.md Section 6 (Error Case 8)
                # Strategy: Exit pipeline (system overload or pathological bucket)
                logger.error(f"Stage 4 timeout for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Timeout (exiting pipeline)")
                print("   Reduce --video-count or check system load")
                print("   This may indicate system issues - stopping pipeline")
                return 8  # Exit code 8 = timeout

            except Exception as e:
                # Unexpected error
                logger.error(
                    f"Stage 4 unexpected error for bucket {bucket_name}: {e}",
                    exc_info=True
                )
                print(f"✗ Bucket {bucket_name} unexpected error: {e}")
                return 99  # Exit code 99 = unexpected error

        logger.info("Stage 4 completed for all buckets")
        print("\n✓ Stage 4: Feature Transformation - COMPLETE")

        # Log Stage 4 summary
        if stage4_summaries:
            total_files = sum(len(s['output_files']) for s in stage4_summaries.values())
            avg_time = sum(s['elapsed_time'] for s in stage4_summaries.values()) / len(stage4_summaries)
            logger.info(
                f"Stage 4 Summary: {total_files} transformation files generated "
                f"across {len(stage4_summaries)} buckets (avg {avg_time:.1f}s per bucket)"
            )
            print(
                f"Summary: {total_files} transformation files across "
                f"{len(stage4_summaries)} buckets"
            )
        else:
            logger.warning("Stage 4 completed but no buckets were processed")
            print("⚠️  No buckets processed in Stage 4 (all skipped)")

        # ===== STAGE 5: ML MODEL TRAINING =====
        logger.info("Starting Stage 5: ML Model Training")
        print("\n" + "="*80)
        print("STAGE 5: ML MODEL TRAINING")
        print("="*80)

        # Note: Hyperparameters are loaded inside run_stage5_training() via load_model_config()
        # No orchestrator-level configuration needed (TI Section 11.4)

        # Process each winning bucket
        stage5_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 5 for bucket: {bucket_name}")
            print(f"\n--- Training models for bucket: {bucket_name} ---")

            bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

            try:
                # Validate Stage 4 completed successfully
                stage4_checkpoint = bucket_path / "checkpoints" / "stage_4_checkpoint.json"
                if not stage4_checkpoint.exists():
                    logger.error(f"Bucket {bucket_name}: Stage 4 checkpoint missing")
                    print(f"✗ Bucket {bucket_name}: Stage 4 not complete (skipping)")
                    continue

                with open(stage4_checkpoint) as f:
                    stage4_status = json.load(f)

                if stage4_status.get("status") != "completed":
                    logger.error(f"Bucket {bucket_name}: Stage 4 status={stage4_status.get('status')}")
                    print(f"✗ Bucket {bucket_name}: Stage 4 incomplete (skipping)")
                    continue

                # Check if Stage 5 already complete for this bucket
                checkpoint_path = bucket_path / "checkpoints" / "stage_5_checkpoint.json"
                if checkpoint_path.exists():
                    logger.info(f"Bucket {bucket_name}: Stage 5 already complete (checkpoint exists)")
                    print(f"✓ Bucket {bucket_name}: Training already complete (skipping)")

                    # Load checkpoint to get model count
                    with open(checkpoint_path) as f:
                        checkpoint = json.load(f)

                    stage5_summaries[bucket_name] = {
                        "models_trained": checkpoint["models_trained"],
                        "elapsed_time": 0.0  # Skipped, no time
                    }
                    continue

                # Prepare config for Stage 5 entry point
                # Source: MLModelTrainingCHILDTI.md Section 11.4 (actual implementation)
                bucket_config = {
                    "bucket": bucket_name,
                    "strategy": config.selection_strategy,
                    "video_count": config.video_count
                }

                # Execute Stage 5 training (from TI Section 11.4: ACTUAL IMPLEMENTATION)
                # Source: MLModelTrainingCHILDTI.md Section 11.4 lines 2945-2947
                # Entry point: run_stage5_training() returns (success, output_files, elapsed_time)
                success, output_files, elapsed_time = run_stage5_training(
                    bucket_path=str(bucket_path),
                    config=bucket_config,
                    selection_strategy=config.selection_strategy
                )

                # Count models trained from output_files list
                models_trained = len([f for f in output_files if f.endswith('.pkl')])

                # CREATE CHECKPOINT
                # Source: Stage 4 pattern (rumiai_ml_batch.py lines 725-738)
                # Note: output_files already provided by run_stage5_training()
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                checkpoint_data = {
                    "stage": "stage_5_ml_model_training",
                    "bucket": bucket_name,
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "models_trained": models_trained,
                    "output_files": output_files  # From run_stage5_training() return value
                }

                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                stage5_summaries[bucket_name] = {
                    "models_trained": models_trained,
                    "elapsed_time": elapsed_time
                }

                logger.info(
                    f"Bucket {bucket_name} complete: "
                    f"{models_trained} models trained in {elapsed_time:.1f}s"
                )
                print(
                    f"✓ Bucket {bucket_name}: {models_trained} models trained → "
                    f"{elapsed_time:.1f}s"
                )

            except StageInputError as e:
                # Missing upstream file (Stage 4 CSVs or checkpoint)
                # Source: MLModelTrainingCHILDTI.md Section 6.1, 6.2 Scenario 1
                # Strategy: Skip bucket, continue pipeline (bucket-specific missing data)
                logger.error(f"Stage 5 prerequisite missing for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Prerequisites missing (skipping)")
                print("   Ensure Stage 4 completed successfully for this bucket")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except InsufficientDataError as e:
                # Insufficient videos or invalid data
                # Source: MLModelTrainingCHILDTI.md Section 6.1, 6.2 Scenario 2
                # Strategy: Skip bucket, continue pipeline (bucket-specific data error)
                logger.error(f"Stage 5 validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Data validation failed (skipping)")
                print("   This indicates insufficient videos or invalid labels")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except ModelTrainingError as e:
                # Model training failed (sklearn errors)
                # Source: MLModelTrainingCHILDTI.md Section 6.1, 6.2 Scenario 3
                # Strategy: Skip bucket, continue pipeline (bucket-specific training failure)
                logger.error(f"Stage 5 training failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Model training failed (skipping)")
                print("   This may indicate data quality issues for this bucket")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except ValidationError as e:
                # Output validation failed (model metrics, feature overlap)
                # Source: MLModelTrainingCHILDTI.md Section 6.1 (custom exception)
                # Strategy: Skip bucket, continue pipeline (bucket-specific validation failure)
                logger.error(f"Stage 5 output validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Output validation failed (skipping)")
                print("   This indicates low-quality models for this bucket")
                print("   Other buckets will continue processing")
                continue  # Skip this bucket, process remaining buckets

            except (IOError, OSError) as e:
                # I/O failure (disk full, permission denied)
                # Source: MLModelTrainingCHILDTI.md Section 6 (system-wide failure)
                # Strategy: Exit pipeline (system-wide issue affects all buckets)
                logger.error(f"Stage 5 I/O error for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: I/O error (exiting pipeline)")
                print("   Check disk space and permissions")
                print("   This is a system-wide issue - stopping pipeline")
                return 4  # Exit code 4 = I/O failure

            except Exception as e:
                # Unexpected error
                # Source: MLModelTrainingCHILDTI.md Section 6 (catch-all)
                logger.error(
                    f"Stage 5 unexpected error for bucket {bucket_name}: {e}",
                    exc_info=True
                )
                print(f"✗ Bucket {bucket_name} unexpected error: {e}")
                return 99  # Exit code 99 = unexpected error

        logger.info("Stage 5 completed for all buckets")
        print("\n✓ Stage 5: ML Model Training - COMPLETE")

        # Log Stage 5 summary
        if stage5_summaries:
            total_models = sum(s['models_trained'] for s in stage5_summaries.values())
            avg_time = sum(s['elapsed_time'] for s in stage5_summaries.values()) / len(stage5_summaries)
            logger.info(
                f"Stage 5 Summary: {total_models} models trained across "
                f"{len(stage5_summaries)} buckets (avg {avg_time:.1f}s per bucket)"
            )
        else:
            logger.warning("Stage 5 completed but no buckets were processed")
            print("⚠️  No buckets processed in Stage 5 (all skipped)")

        # ===== STAGE 6: ML ANALYSIS GENERATION =====
        logger.info("Starting Stage 6: ML Analysis Generation")
        print("\n" + "="*80)
        print("STAGE 6: ML ANALYSIS GENERATION")
        print("="*80)

        # Process each winning bucket
        stage6_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 6 for bucket: {bucket_name}")
            print(f"\n--- Generating ML analysis for bucket: {bucket_name} ---")

            bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

            try:
                # Validate Stage 5 completed successfully
                stage5_checkpoint = bucket_path / "checkpoints" / "stage_5_checkpoint.json"
                if not stage5_checkpoint.exists():
                    logger.error(f"Bucket {bucket_name}: Stage 5 checkpoint missing")
                    print(f"✗ Bucket {bucket_name}: Stage 5 not complete (skipping)")
                    continue

                with open(stage5_checkpoint) as f:
                    stage5_status = json.load(f)

                if stage5_status.get("status") != "completed":
                    logger.error(f"Bucket {bucket_name}: Stage 5 status={stage5_status.get('status')}")
                    print(f"✗ Bucket {bucket_name}: Stage 5 incomplete (skipping)")
                    continue

                # Check if Stage 6 already complete for this bucket
                checkpoint_path = bucket_path / "checkpoints" / "stage_6_checkpoint.json"
                if checkpoint_path.exists():
                    logger.info(f"Bucket {bucket_name}: Stage 6 already complete (checkpoint exists)")
                    print(f"✓ Bucket {bucket_name}: Analysis already complete (skipping)")

                    # Load checkpoint to get output count
                    with open(checkpoint_path) as f:
                        checkpoint = json.load(f)

                    stage6_summaries[bucket_name] = {
                        "json_files_generated": len(checkpoint["output_files"]),
                        "elapsed_time": 0.0  # Skipped, no time
                    }
                    continue

                # Validate prerequisites
                validate_stage_6_prerequisites(str(bucket_path))

                # Get windows configuration for this bucket
                # Source: MLAnalysisGenerationCHILDTI.md Section 2.1 (bucket parameter)
                windows = BUCKET_WINDOWS[bucket_name]

                # Execute Stage 6 analysis generation
                # Source: MLAnalysisGenerationCHILDTI.md Section 8.2 (entry function)
                # Entry point: generate_ml_analysis_jsons(bucket_path, bucket, windows)
                # Returns: exit_code (0 = success, non-zero = failure)
                import time
                start_time = time.time()

                exit_code = generate_ml_analysis_jsons(
                    bucket_path=str(bucket_path),  # TI Section 2.1 StageInput param 1
                    bucket=bucket_name,            # TI Section 2.1 StageInput param 2
                    windows=windows                # TI Section 2.1 StageInput param 3
                )

                elapsed_time = time.time() - start_time

                if exit_code != 0:
                    raise ValueError(f"Stage 6 failed with exit code {exit_code}")

                # Validate outputs (includes JSON count validation)
                validate_stage_6_outputs(str(bucket_path))

                # Count generated JSON files for checkpoint
                # Expected: 1 video RF + N window RF + N window K-Means = 1 + (2 × N)
                # Note: Only count Stage 6 JSONs (rf_video, *_rf_analysis, *_kmeans_analysis)
                # Exclude other JSONs like aggregation_summary.json, transformation_summary.json
                ml_analysis_dir = bucket_path / "ml_analysis"

                # Collect Stage 6-specific JSON files
                stage6_json_files = []
                stage6_json_files.append(ml_analysis_dir / "rf_video_analysis.json")
                for window in windows:
                    stage6_json_files.append(ml_analysis_dir / f"{window}_rf_analysis.json")
                    stage6_json_files.append(ml_analysis_dir / f"{window}_kmeans_analysis.json")

                # Filter to only existing files (for checkpoint output_files list)
                json_files = [f for f in stage6_json_files if f.exists()]
                json_count = len(json_files)

                # Create checkpoint
                # Source: rumiai_ml_batch.py Stage 5 pattern (lines 894-909)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                checkpoint_data = {
                    "stage": "stage_6_ml_analysis_generation",
                    "bucket": bucket_name,
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "json_files_generated": json_count,
                    "output_files": [str(f.relative_to(bucket_path)) for f in json_files]
                }

                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                stage6_summaries[bucket_name] = {
                    "json_files_generated": json_count,
                    "elapsed_time": elapsed_time
                }

                logger.info(
                    f"Bucket {bucket_name} complete: "
                    f"{json_count} JSON files generated in {elapsed_time:.1f}s"
                )
                print(
                    f"✓ Bucket {bucket_name}: {json_count} analysis JSONs → "
                    f"{elapsed_time:.1f}s"
                )

            except FileNotFoundError as e:
                # Missing upstream file (Stage 5 models or Stage 4 CSVs)
                # Source: MLAnalysisGenerationCHILDTI.md Section 6 (Error Case 1)
                # Strategy: Skip bucket, continue pipeline
                logger.error(f"Stage 6 prerequisite missing for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Prerequisites missing (skipping)")
                print("   Ensure Stage 5 completed successfully for this bucket")
                print("   Other buckets will continue processing")
                continue

            except ValueError as e:
                # Input validation failed (corrupted PKL, invalid data)
                # Source: MLAnalysisGenerationCHILDTI.md Section 6 (Error Case 2)
                # Strategy: Skip bucket, continue pipeline
                logger.error(f"Stage 6 validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Data validation failed (skipping)")
                print("   This indicates Stage 5 produced invalid models")
                print("   Other buckets will continue processing")
                continue

            except AssertionError as e:
                # Output validation failed (JSON count mismatch, schema error)
                # Source: MLAnalysisGenerationCHILDTI.md Section 6 (Error Case 3)
                # Strategy: Skip bucket, continue pipeline
                logger.error(f"Stage 6 output validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Output validation failed (skipping)")
                print("   This indicates incorrect JSON generation")
                print("   Other buckets will continue processing")
                continue

            except (IOError, OSError) as e:
                # I/O failure (disk full, permission denied)
                # Source: MLAnalysisGenerationCHILDTI.md Section 6 (Error Case 4)
                # Strategy: Exit pipeline (system-wide issue)
                logger.error(f"Stage 6 I/O error for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: I/O error (exiting pipeline)")
                print("   Check disk space and permissions")
                print("   This is a system-wide issue - stopping pipeline")
                return 4  # Exit code 4 = I/O failure

            except Exception as e:
                # Unexpected error
                logger.error(
                    f"Stage 6 unexpected error for bucket {bucket_name}: {e}",
                    exc_info=True
                )
                print(f"✗ Bucket {bucket_name} unexpected error: {e}")
                return 99  # Exit code 99 = unexpected error

        logger.info("Stage 6 completed for all buckets")
        print("\n✓ Stage 6: ML Analysis Generation - COMPLETE")

        # Log Stage 6 summary
        if stage6_summaries:
            total_jsons = sum(s['json_files_generated'] for s in stage6_summaries.values())
            avg_time = sum(s['elapsed_time'] for s in stage6_summaries.values()) / len(stage6_summaries)
            logger.info(
                f"Stage 6 Summary: {total_jsons} JSON files generated "
                f"across {len(stage6_summaries)} buckets (avg {avg_time:.1f}s per bucket)"
            )
            print(
                f"Summary: {total_jsons} analysis JSONs across "
                f"{len(stage6_summaries)} buckets"
            )
        else:
            logger.warning("Stage 6 completed but no buckets were processed")
            print("⚠️  No buckets processed in Stage 6 (all skipped)")

        # ===== STAGE 7: LLM ANALYSIS - HYBRID TWO-PHASE APPROACH =====
        logger.info("Starting Stage 7: LLM Analysis")
        print("\n" + "="*80)
        print("STAGE 7: LLM ANALYSIS - HYBRID TWO-PHASE APPROACH")
        print("="*80)

        # Validate API key availability
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY environment variable not set")
            print("\n✗ ERROR: ANTHROPIC_API_KEY environment variable not set")
            print("  Obtain API key from: https://console.anthropic.com/settings/keys")
            print("  Set with: export ANTHROPIC_API_KEY='your_key_here'")
            return 1

        # Source: Pattern from Stage 3-6 (analysis_base and winning_buckets already defined)
        # Process each winning bucket
        stage7_summaries = {}
        for bucket_name in winning_buckets:
            logger.info(f"Starting Stage 7 for bucket: {bucket_name}")
            print(f"\n--- LLM Analysis for bucket: {bucket_name} ---")

            bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

            try:
                # Check if Stage 7 outputs already exist
                llm_output_dir = bucket_path / "ml_analysis/llm"
                complete_analysis_file = llm_output_dir / "complete_analysis.json"

                if complete_analysis_file.exists():
                    logger.info(f"Bucket {bucket_name}: Stage 7 already complete (complete_analysis.json found)")
                    print(f"✓ Bucket {bucket_name}: LLM analysis already complete (skipping)")

                    # Count existing output files for summary
                    json_count = 0
                    if llm_output_dir.exists():
                        json_count = len([f for f in llm_output_dir.glob("*.json") if f.name != ".phase1_status.json"])

                    stage7_summaries[bucket_name] = {
                        "json_files_generated": json_count,
                        "elapsed_time": 0.0  # Skipped, no time
                    }
                    continue

                # Validate prerequisites
                validate_stage7_prerequisites(str(bucket_path), bucket_name)

                # Execute Stage 7 LLM Analysis
                # Source: stage7_llm_analysis.py main() function at line 536
                # Parameters: bucket_path (str), bucket (str), hashtag (str | None)
                import time
                start_time = time.time()

                stage7_llm_analysis_main(
                    bucket_path=str(bucket_path),  # Convert Path to str for Stage 7 API
                    bucket=bucket_name,             # e.g., "18-33s"
                    hashtag=cli_args.target         # From CLI args (e.g., "#nutrition")
                )

                elapsed_time = time.time() - start_time

                # Validate outputs
                validate_stage7_outputs(str(bucket_path), bucket_name)

                # Count generated JSON files
                json_count = 0
                if llm_output_dir.exists():
                    json_count = len([f for f in llm_output_dir.glob("*.json") if f.name != ".phase1_status.json"])

                stage7_summaries[bucket_name] = {
                    "json_files_generated": json_count,
                    "elapsed_time": elapsed_time
                }

                logger.info(
                    f"Bucket {bucket_name} complete: "
                    f"{json_count} JSON files generated in {elapsed_time:.1f}s"
                )
                print(
                    f"✓ Bucket {bucket_name}: {json_count} LLM analysis JSONs → "
                    f"{elapsed_time:.1f}s"
                )

            except FileNotFoundError as e:
                # Missing upstream file (Stage 6 JSONs or checkpoint)
                # Source: LLMAnalysisCHILDTI.md Section 6 (Pre-Flight Validation Errors)
                # Strategy: Skip bucket, continue pipeline
                logger.error(f"Stage 7 prerequisite missing for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Prerequisites missing (skipping)")
                print("   Ensure Stage 6 completed successfully for this bucket")
                print("   Other buckets will continue processing")
                continue

            except ValueError as e:
                # Input validation failed (invalid JSON, schema error)
                # Source: LLMAnalysisCHILDTI.md Section 6 (Phase 1 Execution Failures)
                # Strategy: Skip bucket, continue pipeline
                logger.error(f"Stage 7 validation failed for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: Data validation failed (skipping)")
                print("   This indicates Stage 6 produced invalid data")
                print("   Other buckets will continue processing")
                continue

            except RuntimeError as e:
                # API error (401, 403, rate limit exceeded)
                # Source: LLMAnalysisCHILDTI.md Section 6.2 (Non-Retryable API Errors)
                # Strategy: Exit pipeline (system-wide issue)
                logger.error(f"Stage 7 API error for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: API error (exiting pipeline)")
                print("   Check ANTHROPIC_API_KEY and account status")
                print("   This is a system-wide issue - stopping pipeline")
                return 4  # Exit code 4 = API authentication failure

            except (IOError, OSError) as e:
                # I/O failure (disk full, permission denied)
                # Source: LLMAnalysisCHILDTI.md Section 6 (Phase 2 Synthesis Failures)
                # Strategy: Exit pipeline (system-wide issue)
                logger.error(f"Stage 7 I/O error for bucket {bucket_name}: {e}")
                print(f"✗ Bucket {bucket_name}: I/O error (exiting pipeline)")
                print("   Check disk space and permissions")
                print("   This is a system-wide issue - stopping pipeline")
                return 4  # Exit code 4 = I/O failure

            except Exception as e:
                # Unexpected error
                logger.error(
                    f"Stage 7 unexpected error for bucket {bucket_name}: {e}",
                    exc_info=True
                )
                print(f"✗ Bucket {bucket_name} unexpected error: {e}")
                return 99  # Exit code 99 = unexpected error

        logger.info("Stage 7 completed for all buckets")
        print("\n✓ Stage 7: LLM Analysis - COMPLETE")

        # Log Stage 7 summary
        if stage7_summaries:
            total_jsons = sum(s['json_files_generated'] for s in stage7_summaries.values())
            avg_time = sum(s['elapsed_time'] for s in stage7_summaries.values()) / len(stage7_summaries)
            logger.info(
                f"Stage 7 Summary: {total_jsons} JSON files generated "
                f"across {len(stage7_summaries)} buckets (avg {avg_time:.1f}s per bucket)"
            )
            print(
                f"Summary: {total_jsons} LLM analysis JSONs across "
                f"{len(stage7_summaries)} buckets"
            )
        else:
            logger.warning("Stage 7 completed but no buckets were processed")
            print("⚠️  No buckets processed in Stage 7 (all skipped)")

        # ===== FINAL STATUS =====
        print("\n" + "="*80)
        print("PIPELINE STATUS")
        print("="*80)
        print("✓ Stage 0: Foundation - COMPLETE")
        print("✓ Stage 1: Video Discovery & Selection - COMPLETE")
        print("✓ Stage 2: Video Processing - COMPLETE")
        print("✓ Stage 2.5: File Organization - COMPLETE")
        print("✓ Stage 2.6/2.7: Content Analysis - COMPLETE")
        print("✓ Stage 3: Feature Aggregation - COMPLETE")
        print("✓ Stage 3.4: Review CSV Generation - COMPLETE")
        print("✓ Stage 4: Feature Transformation - COMPLETE")
        print("✓ Stage 5: ML Model Training - COMPLETE")
        print("✓ Stage 6: ML Analysis Generation - COMPLETE")
        print("✓ Stage 7: LLM Analysis - COMPLETE")
        print("="*80)
        print()
        print(f"✅ All stages complete!")
        print(f"   Processed {completed_videos} videos across {len(winning_buckets)} buckets")
        print(f"   Classified {summary['completed']} videos with content taxonomy")
        if stage3_summaries:
            print(f"   Aggregated {total_aggregated} videos into {len(stage3_summaries)} bucket CSVs")
        if stage4_summaries:
            print(f"   Transformed {total_files} ML-ready files for {len(stage4_summaries)} buckets")
        if stage5_summaries:
            print(f"   Trained {total_models} ML models across {len(stage5_summaries)} buckets")
        if stage6_summaries:
            print(f"   Generated {total_jsons} analysis JSONs across {len(stage6_summaries)} buckets")
        if stage7_summaries:
            total_llm_jsons = sum(s['json_files_generated'] for s in stage7_summaries.values())
            print(f"   Generated {total_llm_jsons} LLM analysis JSONs across {len(stage7_summaries)} buckets")
        print(f"   Output location: {analysis_base}")
        print()

        logger.info("="*80)
        logger.info("PIPELINE EXECUTION COMPLETE (Stages 0-7)")
        logger.info("="*80)

        return 0

    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user (Ctrl+C)")
        return 130

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        if 'logger' in locals():
            logger.error(f"Pipeline failed: {e}", exc_info=True)
        else:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
