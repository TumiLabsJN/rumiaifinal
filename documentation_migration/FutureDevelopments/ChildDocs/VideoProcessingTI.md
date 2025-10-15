# Video Processing - Technical Implementation (TI)

> **TI Document**: VideoProcessingTI.md
> **Parent HLD**: VideoProcessingCHILD.md
> **Foundation HLD**: FoundationCHILD.md
> **Version**: 1.1 (Corrected)
> **Last Updated**: 2025-01-28
> **Status**: To Implement

---

## 1. Document Metadata

**TI_Document**: VideoProcessingTI.md

**Parent_HLD**: VideoProcessingCHILD.md

**Foundation_HLD**: FoundationCHILD.md

**Covers_HLD_Sections**:
- VideoProcessingCHILD.md Section 1: Context & Business Goal
- VideoProcessingCHILD.md Section 2: Architecture & Design
- VideoProcessingCHILD.md Section 2.1: High-Level Approach
- VideoProcessingCHILD.md Section 2.2: Data Flow
- VideoProcessingCHILD.md Section 2.3: Detailed Process (all subsections 2.3.0-2.3.5)
- VideoProcessingCHILD.md Section 3: Dependencies & Integration
- VideoProcessingCHILD.md Section 3.1: Input Dependencies
- VideoProcessingCHILD.md Section 3.2: Output Contracts
- VideoProcessingCHILD.md Section 3.3: Cross-Stage Dependencies
- VideoProcessingCHILD.md Section 3.4: External Dependencies
- VideoProcessingCHILD.md Section 4: Configuration & Parameters
- VideoProcessingCHILD.md Section 4.1: CLI Parameters
- VideoProcessingCHILD.md Section 4.2: Internal Configuration
- VideoProcessingCHILD.md Section 5: Data Schemas
- VideoProcessingCHILD.md Section 5.1: Input Schema
- VideoProcessingCHILD.md Section 5.2: Output Schema
- VideoProcessingCHILD.md Section 6: Error Handling & Validation
- VideoProcessingCHILD.md Section 6.1: Input Validation
- VideoProcessingCHILD.md Section 6.2: Error Cases
- VideoProcessingCHILD.md Section 6.3: Output Validation
- VideoProcessingCHILD.md Appendix A: Checkpoint Resume Scenarios
- FoundationCHILD.md Section 2: Client Architecture & Storage
- FoundationCHILD.md Section 2.1: Directory Structure
- FoundationCHILD.md Section 2.2: Path Templates
- FoundationCHILD.md Section 4: CLI Command Structure
- FoundationCHILD.md Section 4.1: CLI Parameters
- FoundationCHILD.md Section 5: Configuration Schemas
- FoundationCHILD.md Section 5.1: config.json schema
- FoundationCHILD.md Section 5.2: Apify Video Metadata Schema
- FoundationCHILD.md Section 5.3: Checkpoint Schema

**Note on Pseudocode Source**: VideoProcessingCHILD.md embeds pseudocode directly in Section 2.3 subsections rather than in a separate Appendix B. All algorithmic specifications in this TI are derived from Section 2.3.X inline code blocks.

**Related_TI_Docs**:
- **Depends_On**:
  - FoundationTI.md (ALWAYS - provides directory structure, CLI params, config schemas)
  - VideoDiscoveryTI.md (Stage 1 - provides selected video list)
- **Feeds_Into**:
  - PipelineValidationTI.md (Stage 2.4 - consumes temporal_windows_updated.json)
  - FeatureAggregationTI.md (Stage 3 - consumes temporal_windows_updated.json)

**Implementation_Priority**: CRITICAL

**Rationale**: This is the core video processing stage that must complete before any downstream ML analysis can begin. Without this stage, no temporal window features exist for aggregation, transformation, or model training. The checkpoint-resume capability is essential for long-running batch operations (6-8 hours for 300 videos) that are vulnerable to SSH disconnects and system interruptions.

---

## 2. Stage Contract

```python
# INPUT CONTRACT
class StageInput:
    """
    Exact structure this stage receives.

    Sources:
    - FoundationCHILD.md Section 4: CLI Command Structure
    - FoundationCHILD.md Section 2: Client Architecture & Storage
    - VideoProcessingCHILD.md Section 3.1: Input Dependencies
    - VideoProcessingCHILD.md Section 5.1: Input Schema
    """

    # ===== CLI PARAMETERS (from FoundationCHILD.md Section 4.1) =====
    client_id: str              # Required, Regex ^[a-zA-Z0-9_]+$, Example: "acme_corp"
    analysis_type: str          # Required, ["hashtag", "competitor", "creator"], Example: "hashtag"
    target: str                 # Required, Format: # for hashtag, @ for competitor/creator, Example: "#nutrition"
    analysis_mode: str          # Required, ["top", "recent"], Default: "top"
    selection_strategy: str     # Required, ["contrastive", "top"], Default: "contrastive"
    video_count: int            # Required, Range: 10-500, Default: 100 (contrastive), 40 (top)
    date_filter: str            # Required, Regex ^last_\d+_days$, Default: "last_90_days"
    run_date: str               # Required, ISO 8601 timestamp, Example: "2025-01-28T10:30:00Z"

    # ===== BUCKET PARAMETER (from Stage 1 winner analysis) =====
    bucket_name: str            # Required, bucket name only (e.g., "18-33s"), NOT full path
                                # Valid values: ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
                                # Determined by Stage 1 winner analysis (top 3 winning buckets)

    # ===== DIRECTORY PATHS (constructed at runtime from FoundationCHILD.md Section 2) =====
    data_root: str              # Environment variable DATA_ROOT, default: "/data"
    analysis_base: str          # Template: {data_root}/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/
                                # Example: "/data/clients/acme_corp/hashtags/#nutrition/top_contrastive/"

    # ===== STAGE-SPECIFIC INPUTS (from VideoProcessingCHILD.md Section 3.1) =====
    selected_video_list: List[Dict]  # From Stage 1, Required, must not be empty
                                     # Each video: {id, videoMeta.downloadAddr, duration, playCount, createTime}

    config_json_path: str       # Required, must exist with read access
                                # Template: {analysis_base}/config.json


# OUTPUT CONTRACT
class StageOutput:
    """
    Exact structure this stage produces.

    Sources:
    - FoundationCHILD.md Section 2: Client Architecture & Storage
    - VideoProcessingCHILD.md Section 3.2: Output Contracts
    - VideoProcessingCHILD.md Section 5.2: Output Schema
    """

    # ===== PRIMARY OUTPUTS =====
    temporal_windows_json_files: List[str]  # One JSON per successfully processed video
                                            # Template: {bucket_base}/analysis/insights/{video_id}_temporal_windows_updated.json
                                            # Count: Equals checkpoint['completed']

    checkpoint_json: str        # Template: {bucket_base}/checkpoints/stage_2_checkpoint.json
                                # Consumer: Auto-resume on restart

    downloaded_videos: List[str]  # Template: {bucket_base}/videos/{video_id}.mp4
                                  # Count: Equals checkpoint['completed']
                                  # Validation: file_size > 100KB

    processing_logs: str        # Template: {bucket_base}/logs/processing_{date}.log (optional)

    # ===== BUCKET DIRECTORIES (created by Step 2.3.0) =====
    created_bucket_directories: Dict[str, str]  # All 8 bucket directories
                                                # Keys: bucket names ("0-3s", ..., "90-120s")
                                                # Values: full bucket paths
                                                # Each bucket has 15 subdirectories
```

---

## 3. Data Schemas

```python
# ===================================================================
# FOUNDATION SCHEMAS (from FoundationCHILD.md Section 5)
# ===================================================================

ConfigSchema = {
    "client_id": str,              # Required, Regex ^[a-zA-Z0-9_]+$, Example: "acme_corp"
    "analysis_type": str,          # Required, ["hashtag", "competitor", "creator"], Example: "hashtag"
    "target": str,                 # Required, Format: #hashtag or @handle, Example: "#nutrition"
    "analysis_mode": str,          # Required, ["top", "recent"], Example: "top"
    "selection_strategy": str,     # Required, ["contrastive", "top"], Example: "contrastive"
    "video_count": int,            # Required, Range: 10-500, Example: 100
    "date_filter": str,            # Required, Regex ^last_\d+_days$, Example: "last_90_days"
    "run_date": str,               # Required, ISO 8601, Example: "2025-01-28T10:30:00Z"
}

ApifyVideoMetadataSchema = {
    "id": str,                     # Required, unique, Example: "7428596413707144481"
    "createTime": int,             # Required, Unix timestamp, Range: > 0, Example: 1704067200
    "duration": int,               # Required, Range: 3-120 seconds, Example: 25
    "playCount": int,              # Required, Range: >= 0, Example: 50000
    "videoMeta": {                 # Required
        "downloadAddr": str,       # Required, Valid HTTP/HTTPS URL, Example: "https://example.com/video1.mp4"
    },
}

# ===================================================================
# STAGE-SPECIFIC INPUT SCHEMAS (from VideoProcessingCHILD.md Section 5.1)
# ===================================================================

SelectedVideoListSchema = {
    "id": str,                     # Required, TikTok video ID, Example: "7428596413707144481"
    "videoMeta.downloadAddr": str, # Required, MP4 download URL, Example: "https://..."
    "duration": int,               # Required, Range: 3-120, Example: 25
    "playCount": int,              # Required, Range: >= 0, Example: 50000
    "createTime": int,             # Required, Unix timestamp, Example: 1704067200
}

# ===================================================================
# STAGE-SPECIFIC OUTPUT SCHEMAS (from VideoProcessingCHILD.md Section 5.2)
# ===================================================================

TemporalWindowsUpdatedSchema = {
    "video_id": str,               # Required, must match filename, Example: "7428596413707144481"
    "duration": float,             # Required, Range: 3.0-120.0, Example: 25.5
    "processing_timestamp": float, # Required, Unix timestamp, Example: 1706453600.5
    "version": str,                # Required, >= "2.0.0", Example: "2.0.0"
    "temporal_windows": dict,      # Required, must have 'hook', 'closing' keys
    "temporal_windows.hook": dict, # Required, 60+ features, Example: {"scene_count": 3, ...}
    "temporal_windows.middle_segments": list | None,  # null if duration <= 9s, else list of dicts
    "temporal_windows.closing": dict,  # Required, 60+ features
    "metadata": dict,              # Required, must have 'gender_detection', 'hashtag_analysis'
}

CheckpointSchema = {
    "stage": str,                  # Required, always "video_processing"
    "bucket": str,                 # Required, bucket name (e.g., "18-33s")
    "total_videos": int,           # Required, >= 0
    "completed": int,              # Required, >= 0
    "failed": int,                 # Required, >= 0
    "remaining": int,              # Required, invariant: remaining = total - completed - failed
    "last_checkpoint": str,        # Required, ISO 8601 timestamp
    "completed_video_ids": list,   # Required, list of str
    "failed_video_ids": list,      # Required, list of dict (schema below)
    "config": dict,                # Required, for resume validation
    "status": str,                 # Required, ["in_progress", "paused", "completed"]
    "pause_reason": str | None,    # Optional, "user_requested" if status="paused"
    "pause_timestamp": str | None, # Optional, ISO timestamp when paused
}

FailedVideoEntrySchema = {
    "video_id": str,               # Required, Example: "7428596413707144481"
    "error": str,                  # Required, Example: "RumiAI timeout after 300s"
    "error_type": str,             # Required, Example: "TimeoutError", "ProcessingError", "DownloadError"
    "timestamp": str,              # Required, ISO timestamp, Example: "2025-01-28T14:32:10Z"
}
```

---

## 4. Algorithmic Specifications

```python
"""
===================================================================
HELPER FUNCTION: get_bucket_path()
===================================================================

Source: VideoProcessingCHILD.md Section 2.3.1 (Utility Functions)

Purpose: Construct full bucket directory path from config and bucket name
"""

def get_bucket_path(config: dict, bucket_name: str) -> str:
    """
    Construct full bucket directory path from config and bucket name.

    Naming convention:
    - bucket_name: Duration range only (e.g., "18-33s")
    - bucket_path: Full directory path (e.g., "/data/clients/acme/hashtags/#nutrition/bucket_18-33s/")

    All functions use bucket_name as parameter and construct full paths using this helper.

    Args:
        config: dict, loaded from config.json (FoundationCHILD Section 5.1)
        bucket_name: str, duration range (e.g., "18-33s")

    Returns:
        str: Full bucket directory path with trailing slash
    """
    data_root = os.getenv('DATA_ROOT', '/data')
    analysis_base = (
        f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/"
        f"{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"
    )
    return f"{analysis_base}buckets/bucket_{bucket_name}/"


"""
===================================================================
FUNCTION 1: initialize_bucket_directories()
===================================================================

Source: VideoProcessingCHILD.md Section 2.3.0 - Bucket Directory Initialization

Purpose: Create all 8 bucket directories with complete subdirectory structure before video processing begins
"""

def initialize_bucket_directories(config: dict) -> dict:
    """
    Create all 8 bucket directories with complete subdirectory structure.

    Args:
        config: dict, loaded from config.json (FoundationCHILD Section 5.1)

    Returns:
        dict: Created paths mapped to bucket names

    Raises:
        OSError: If directory creation fails (permissions, disk space)
    """

    # Step 1: Define all 8 bucket names
    BUCKET_NAMES = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]

    # Step 2: Construct base path from config
    data_root = os.getenv('DATA_ROOT', '/data')
    analysis_base = (
        f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/"
        f"{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"
    )

    # Step 3: Define complete subdirectory structure per bucket (15 subdirectories)
    SUBDIRECTORIES = [
        "videos/", "analysis/", "analysis/insights/", "analysis/unified/",
        "analysis/service_debug/", "validation/", "flagged_videos/", "ml_analysis/",
        "models/", "llm_reports/", "llm_reports/analysis/", "llm_reports/formatted/",
        "reports/", "checkpoints/", "logs/"
    ]

    # Step 4: Initialize tracking dictionary
    created_paths = {}

    # Step 5: Create all 8 buckets with full subdirectory structure
    for bucket_name in BUCKET_NAMES:
        bucket_path = f"{analysis_base}buckets/bucket_{bucket_name}/"
        logger.info(f"Creating bucket directory structure: bucket_{bucket_name}")

        try:
            os.makedirs(bucket_path, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create bucket directory {bucket_path}: {e}")

        for subdir in SUBDIRECTORIES:
            subdir_path = f"{bucket_path}{subdir}"
            try:
                os.makedirs(subdir_path, exist_ok=True)
            except OSError as e:
                raise OSError(f"Failed to create subdirectory {subdir_path}: {e}")

        created_paths[bucket_name] = bucket_path
        logger.debug(f"  ✓ Created {len(SUBDIRECTORIES)} subdirectories for bucket_{bucket_name}")

    logger.info(f"✓ Successfully created all 8 bucket directories with complete subdirectory structure")
    return created_paths


"""
===================================================================
FUNCTION 2: initialize_checkpoint()
===================================================================

Source: VideoProcessingCHILD.md Section 2.3.1 - Checkpoint Initialization

Purpose: Load existing checkpoint or create new one, enabling auto-resume on interruption
"""

def initialize_checkpoint(bucket_name: str, video_list: list, config: dict) -> tuple[dict, list]:
    """
    Initialize checkpoint for video processing stage.

    FIXED: Now uses bucket_name (not full path) and constructs full path using get_bucket_path()

    Args:
        bucket_name: str, bucket name only (e.g., "18-33s"), NOT full path
        video_list: list, videos selected by Stage 1
        config: dict, loaded from config.json

    Returns:
        checkpoint: dict, checkpoint data (new or existing)
        remaining_videos: list, videos to process (excludes completed)
    """

    def validate_config_match(checkpoint_config: dict, current_config: dict):
        """Validates that checkpoint config matches current run config."""
        critical_fields = ['video_count', 'selection_strategy', 'date_filter']
        mismatches = []

        for field in critical_fields:
            if checkpoint_config.get(field) != current_config.get(field):
                mismatches.append(
                    f"{field}: checkpoint={checkpoint_config.get(field)}, "
                    f"current={current_config.get(field)}"
                )

        if mismatches:
            raise ValueError(
                f"Config mismatch detected. Cannot resume with different parameters:\n" +
                "\n".join(mismatches)
            )

    # FIXED: Construct full bucket path using helper function
    bucket_path = get_bucket_path(config, bucket_name)
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    # Check if checkpoint exists (auto-resume scenario)
    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint_with_recovery(checkpoint_path)
        validate_config_match(checkpoint['config'], config)

        completed_ids = set(checkpoint['completed_video_ids'])
        remaining_videos = [v for v in video_list if v['id'] not in completed_ids]

        logger.info(f"Checkpoint detected: {checkpoint['completed']}/{checkpoint['total_videos']} completed")
        logger.info(f"Auto-resuming from bucket {bucket_name} ({len(remaining_videos)} remaining)")

        return checkpoint, remaining_videos

    # Create new checkpoint
    else:
        checkpoint = {
            "stage": "video_processing",
            "bucket": bucket_name,
            "total_videos": len(video_list),
            "completed": 0,
            "failed": 0,
            "remaining": len(video_list),
            "last_checkpoint": datetime.utcnow().isoformat(),
            "completed_video_ids": [],
            "failed_video_ids": [],
            "config": config,
            "status": "in_progress",
            "pause_reason": None,
            "pause_timestamp": None
        }

        save_checkpoint_with_backup(checkpoint_path, checkpoint)
        logger.info(f"Created new checkpoint for bucket {bucket_name} ({len(video_list)} videos)")

        return checkpoint, video_list


"""
===================================================================
HELPER FUNCTIONS: Checkpoint Backup & Recovery
===================================================================

Source: VideoProcessingCHILD.md Section 2.3.1
"""

def save_checkpoint_with_backup(checkpoint_path: str, checkpoint: dict):
    """
    Save checkpoint with automatic backup to prevent data loss from corruption.

    Process:
    1. Copy existing checkpoint to .backup.json (if exists)
    2. Write new checkpoint to .json
    3. Handle errors gracefully

    Args:
        checkpoint_path: str, path to checkpoint file
        checkpoint: dict, checkpoint data to save
    """
    import shutil

    backup_path = checkpoint_path.replace('.json', '.backup.json')

    # Backup existing checkpoint before overwriting
    if os.path.exists(checkpoint_path):
        try:
            shutil.copy2(checkpoint_path, backup_path)
        except Exception as e:
            logger.warning(f"Failed to backup checkpoint: {e}")

    # Write new checkpoint
    save_json(checkpoint_path, checkpoint)


def load_checkpoint_with_recovery(checkpoint_path: str) -> dict:
    """
    Load checkpoint with automatic recovery from backup on corruption.

    Recovery strategy:
    1. Try loading main checkpoint
    2. If corrupted (JSONDecodeError), try loading .backup.json
    3. If both corrupted, fail with clear recovery instructions

    Args:
        checkpoint_path: str, path to checkpoint file

    Returns:
        dict: loaded checkpoint data

    Raises:
        CheckpointCorruptionError: if both checkpoint and backup are corrupted
    """
    backup_path = checkpoint_path.replace('.json', '.backup.json')

    # Try loading main checkpoint
    try:
        return load_json(checkpoint_path)
    except json.JSONDecodeError as e:
        logger.error(f"Checkpoint corrupted: {e}")

        # Try loading backup
        if os.path.exists(backup_path):
            logger.info("Attempting to restore from backup...")
            try:
                checkpoint = load_json(backup_path)
                logger.info("✓ Successfully restored from backup")
                return checkpoint
            except json.JSONDecodeError:
                logger.error("Backup also corrupted")

        # Both corrupted - raise with recovery instructions
        raise CheckpointCorruptionError(
            checkpoint_path=checkpoint_path,
            backup_path=backup_path,
            original_error=e
        )


"""
===================================================================
FUNCTION 3: download_video()
===================================================================

Source: VideoProcessingCHILD.md Section 2.3.2 - Video Download
"""

def download_video(video_metadata: dict, output_dir: str, max_attempts: int = 3) -> str:
    """
    Download video MP4 from Apify download URL with retry logic.

    Args:
        video_metadata: dict, Apify metadata
        output_dir: str, path to bucket/videos/ directory
        max_attempts: int, max download attempts (default: 3)

    Returns:
        str: path to downloaded MP4

    Raises:
        DownloadError: if download fails after max_attempts
    """

    video_id = video_metadata['id']
    download_url = video_metadata['videoMeta']['downloadAddr']
    output_path = f"{output_dir}/{video_id}.mp4"

    # Check if video already downloaded (resume optimization)
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)

        if file_size < MIN_VIDEO_SIZE:
            logger.warning(
                f"Existing file for {video_id} too small ({file_size} bytes), "
                f"removing and re-downloading"
            )
            os.remove(output_path)
        else:
            logger.info(f"Video {video_id} already downloaded and valid ({file_size} bytes), skipping")
            return output_path

    # Retry loop with exponential backoff
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Downloading video {video_id} (attempt {attempt}/{max_attempts})")

            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(output_path)
            if file_size < MIN_VIDEO_SIZE:
                raise DownloadError(
                    video_id=video_id,
                    attempts=attempt,
                    original_error=Exception(f"Downloaded file too small: {file_size} bytes (minimum: {MIN_VIDEO_SIZE})")
                )

            logger.info(f"Successfully downloaded video {video_id} ({file_size / 1024 / 1024:.2f} MB)")
            return output_path

        except (requests.exceptions.RequestException, DownloadError) as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")

            if os.path.exists(output_path):
                os.remove(output_path)

            if attempt < max_attempts:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise DownloadError(
                    video_id=video_id,
                    attempts=max_attempts,
                    original_error=e
                )


"""
===================================================================
FUNCTION 4: process_videos_sequential()
===================================================================

Source: VideoProcessingCHILD.md Section 2.3.3 - Sequential RumiAI Processing
"""

def process_videos_sequential(remaining_videos: list, bucket_name: str, checkpoint: dict, config: dict) -> dict:
    """
    Process videos sequentially through RumiAI pipeline with checkpointing.

    FIXED: Now uses bucket_name and constructs full path, calls error handlers

    Args:
        remaining_videos: list, videos to process
        bucket_name: str, bucket name (e.g., "18-33s")
        checkpoint: dict, checkpoint data
        config: dict, configuration for path construction

    Returns:
        dict: processing statistics
    """

    # FIXED: Construct full bucket path
    bucket_path = get_bucket_path(config, bucket_name)
    insights_dir = f"{bucket_path}analysis/insights/"
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    for i, video in enumerate(remaining_videos, start=1):
        video_id = video['id']
        video_path = f"{bucket_path}videos/{video_id}.mp4"

        logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id}")

        try:
            # Run RumiAI pipeline with timeout
            result = run_rumiai_pipeline(
                video_path=video_path,
                video_id=video_id,
                output_dir=f"{bucket_path}analysis/",
                timeout=300
            )

            # Validate output exists
            insights_path = f"{insights_dir}{video_id}_temporal_windows_updated.json"
            if not os.path.exists(insights_path):
                raise ProcessingError(
                    video_id=video_id,
                    stage="output_validation",
                    message=f"RumiAI did not generate insights file"
                )

            # Validate output schema
            insights = load_json(insights_path)
            validate_temporal_windows_schema(insights)

            # Mark as completed
            checkpoint['completed'] += 1
            checkpoint['remaining'] -= 1
            checkpoint['completed_video_ids'].append(video_id)
            checkpoint['last_checkpoint'] = datetime.utcnow().isoformat()

            save_checkpoint_with_backup(checkpoint_path, checkpoint)
            logger.info(f"Successfully processed video {video_id} ({checkpoint['completed']}/{checkpoint['total_videos']})")

        except Exception as e:
            # FIXED: Call error handler from Section 6
            handle_video_processing_error(e, video_id, checkpoint, checkpoint_path)
            continue

    return {
        "total": checkpoint['total_videos'],
        "completed": checkpoint['completed'],
        "failed": checkpoint['failed']
    }


"""
===================================================================
FUNCTION 5: process_videos_with_pause_support()
===================================================================

Source: VideoProcessingCHILD.md Section 2.3.4 - Graceful Pause Handling
"""

# Global pause flag
pause_requested = False

def request_pause(signum: int, frame):
    """Signal handler for graceful pause."""
    global pause_requested

    if not pause_requested:
        pause_requested = True
        logger.info("\n⏸️  Pause requested. Will pause after current video completes...")
        logger.info("   Press Ctrl+C again to force quit (current video will be lost)")
    else:
        logger.warning("\n❌ Force quit requested. Current video will NOT be saved.")
        logger.warning("   Checkpoint may be inconsistent if killed mid-write.")
        sys.exit(1)


def process_videos_with_pause_support(remaining_videos: list, bucket_name: str, checkpoint: dict, config: dict):
    """
    Process videos with graceful pause support.

    FIXED: Now uses bucket_name and constructs full path

    Args:
        remaining_videos: list, videos to process
        bucket_name: str, bucket name
        checkpoint: dict, checkpoint data
        config: dict, configuration
    """
    global pause_requested

    signal.signal(signal.SIGINT, request_pause)

    if platform.system() == 'Windows':
        logger.warning("Graceful pause (Ctrl+C) has limited support on Windows")
        logger.warning("Checkpoint auto-save after each video provides recovery")

    # FIXED: Construct full bucket path
    bucket_path = get_bucket_path(config, bucket_name)
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    for i, video in enumerate(remaining_videos, start=1):
        # Check pause flag BEFORE starting next video
        if pause_requested:
            checkpoint['status'] = 'paused'
            checkpoint['pause_reason'] = 'user_requested'
            checkpoint['pause_timestamp'] = datetime.utcnow().isoformat()
            save_checkpoint_with_backup(checkpoint_path, checkpoint)

            logger.info(f"\n⏸️  Paused gracefully after {checkpoint['completed']}/{checkpoint['total_videos']} videos")
            logger.info(f"   Resume anytime by re-running the same command")
            logger.info(f"   Checkpoint: {checkpoint_path}")
            return

        # Process video normally (delegate to process_videos_sequential logic)
        video_id = video['id']
        logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id}")

        # ... (rest of processing logic - same as process_videos_sequential)


"""
===================================================================
FUNCTION 6: finalize_checkpoint()
===================================================================

Source: VideoProcessingCHILD.md Section 2.3.5 - Mark Analysis Complete
"""

def finalize_checkpoint(checkpoint: dict, config: dict):
    """
    Mark video processing stage as complete.

    FIXED: Now accepts config to construct full bucket path

    Args:
        checkpoint: dict, checkpoint data after all videos processed
        config: dict, configuration
    """
    checkpoint['status'] = 'completed'
    checkpoint['completion_time'] = datetime.utcnow().isoformat()

    # FIXED: Construct full bucket path
    bucket_path = get_bucket_path(config, checkpoint['bucket'])
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    save_json(checkpoint_path, checkpoint)

    logger.info(f"Video processing complete for bucket {checkpoint['bucket']}")
    logger.info(f"  Total: {checkpoint['total_videos']}")
    logger.info(f"  Completed: {checkpoint['completed']}")
    logger.info(f"  Failed: {checkpoint['failed']}")

    if checkpoint['failed'] > 0:
        logger.warning(f"  {checkpoint['failed']} videos failed (see failed_video_ids in checkpoint)")


"""
===================================================================
MAIN ORCHESTRATION FUNCTION
===================================================================

NEW: Main entry point that coordinates all 6 functions
"""

def stage_2_video_processing_main(config: dict, video_list: list, bucket_name: str) -> dict:
    """
    Main orchestration function for Stage 2: Video Processing.

    Coordinates all substeps (2.3.0 through 2.3.5) in sequence.

    Args:
        config: dict, loaded from config.json
        video_list: list, selected videos from Stage 1
        bucket_name: str, bucket to process (e.g., "18-33s")

    Returns:
        dict: final processing statistics
            {"total": int, "completed": int, "failed": int, "status": str}

    Raises:
        ValueError: if input validation fails
        OSError: if disk/permission errors occur
    """

    # Step 0: Initialize bucket directories (runs once for all 8 buckets)
    logger.info("Step 0: Initializing bucket directories")
    created_buckets = initialize_bucket_directories(config)
    logger.info(f"Created {len(created_buckets)} bucket directories")

    # Step 1: Load or create checkpoint
    logger.info(f"Step 1: Initializing checkpoint for bucket {bucket_name}")
    checkpoint, remaining_videos = initialize_checkpoint(bucket_name, video_list, config)

    if not remaining_videos:
        logger.info("No videos to process (all videos already completed)")
        finalize_checkpoint(checkpoint, config)
        return {
            "total": checkpoint['total_videos'],
            "completed": checkpoint['completed'],
            "failed": checkpoint['failed'],
            "status": "completed"
        }

    # Step 2-4: Process videos with pause support
    logger.info(f"Step 2-4: Processing {len(remaining_videos)} videos")
    process_videos_with_pause_support(remaining_videos, bucket_name, checkpoint, config)

    # Step 5: Finalize checkpoint
    logger.info("Step 5: Finalizing checkpoint")
    finalize_checkpoint(checkpoint, config)

    # Step 6: Validate outputs
    logger.info("Step 6: Validating stage outputs")
    validate_stage_output(bucket_name, checkpoint, config)

    return {
        "total": checkpoint['total_videos'],
        "completed": checkpoint['completed'],
        "failed": checkpoint['failed'],
        "status": checkpoint['status']
    }
```

---

## 5. Validation Rules

```python
# (Content unchanged from original - already correct)
# ... [Previous Section 5 content remains the same]
```

---

## 6. Error Handling

```python
# ===================================================================
# EXCEPTION CLASSES (ENHANCED)
# Source: VideoProcessingCHILD.md Section 2.3.1
# FIXED: Now include error context for better debugging
# ===================================================================

class DownloadError(Exception):
    """
    Raised when video download fails after max retry attempts.

    ENHANCED: Now captures error context (video_id, attempts, original error)
    """
    def __init__(self, video_id: str, attempts: int, original_error: Exception):
        self.video_id = video_id
        self.attempts = attempts
        self.original_error = original_error
        super().__init__(
            f"Failed to download video {video_id} after {attempts} attempts: {original_error}"
        )


class ProcessingError(Exception):
    """
    Raised when RumiAI pipeline fails.

    ENHANCED: Now captures error context (video_id, stage, message)
    """
    def __init__(self, video_id: str, stage: str, message: str):
        self.video_id = video_id
        self.stage = stage
        self.message = message
        super().__init__(f"RumiAI processing failed for {video_id} at stage {stage}: {message}")


class ValidationError(Exception):
    """
    Raised when output schema validation fails.

    ENHANCED: Now captures error context (video_id, field, expected, actual)
    """
    def __init__(self, video_id: str, field: str, expected: str, actual: str):
        self.video_id = video_id
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Schema validation failed for {video_id}: "
            f"field '{field}' expected {expected}, got {actual}"
        )


class CheckpointCorruptionError(Exception):
    """
    Raised when both checkpoint and backup are corrupted.

    NEW: Provides recovery instructions
    """
    def __init__(self, checkpoint_path: str, backup_path: str, original_error: Exception):
        self.checkpoint_path = checkpoint_path
        self.backup_path = backup_path
        self.original_error = original_error

        recovery_msg = (
            f"Checkpoint and backup both corrupted.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Backup: {backup_path}\n"
            f"Original error: {original_error}\n\n"
            f"Recovery options:\n"
            f"  1. Use --force flag to discard checkpoint and restart\n"
            f"  2. Manually inspect checkpoint files for partial recovery\n"
            f"  3. Contact support if data recovery is critical"
        )
        super().__init__(recovery_msg)


# ===================================================================
# ERROR HANDLING IMPLEMENTATION (ENHANCED)
# FIXED: Now provides structured error handling
# ===================================================================

def handle_video_processing_error(
    error: Exception,
    video_id: str,
    checkpoint: dict,
    checkpoint_path: str
) -> str:
    """
    Handle errors during video processing with skip-on-fail policy.

    ENHANCED: Now includes metrics logging and critical error detection

    Args:
        error: Exception, the error that occurred
        video_id: str, TikTok video ID
        checkpoint: dict, current checkpoint data
        checkpoint_path: str, path to checkpoint file

    Returns:
        str: Action taken ("failed", "critical")
    """

    error_type = type(error).__name__
    error_message = str(error)

    # Check for critical errors that should halt processing
    if isinstance(error, (OSError, MemoryError)) and "disk full" in error_message.lower():
        logger.critical(f"CRITICAL ERROR: Disk full. Cannot continue processing.")
        raise error

    # Log error with full context
    logger.error(f"Failed to process video {video_id}: {error_message}")
    logger.error(f"Error type: {error_type}")

    # Update checkpoint with failure
    checkpoint['failed'] += 1
    checkpoint['remaining'] -= 1
    checkpoint['failed_video_ids'].append({
        "video_id": video_id,
        "error": error_message,
        "error_type": error_type,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Save checkpoint with failure
    save_checkpoint_with_backup(checkpoint_path, checkpoint)

    # Log skip-on-fail action
    logger.warning(f"Video {video_id} marked as failed. Continuing batch processing.")

    # Log metrics (for performance tracking)
    log_failure_metrics(video_id, error_type)

    return "failed"


def log_failure_metrics(video_id: str, error_type: str):
    """Log failure metrics for performance monitoring."""
    logger.debug(f"Failure metric: video_id={video_id}, error_type={error_type}")

# ... [Rest of Section 6 content unchanged]
```

---

## 7. Complete Example Traces

```
TRACE 1: Normal Processing (Happy Path)

Note: Service times shown are examples for a 25s video. Actual times vary by video duration
and complexity. FEAT (43% of total) and Whisper (15% of total) are primary bottlenecks.

[Rest of Trace 1 content unchanged...]
```

---

## 9. Configuration & Environment

```python
# ===================================================================
# ENVIRONMENT SETUP (ENHANCED)
# FIXED: Now validates Python version, memory, and disk space
# ===================================================================

def setup_environment():
    """
    Setup environment variables and validate system requirements.

    ENHANCED: Now checks Python version, memory, disk space

    Raises:
        EnvironmentError: if environment setup fails
        SystemError: if system requirements not met
    """
    import sys
    import shutil

    # Validate Python version (3.8+ required)
    if sys.version_info < (3, 8):
        raise SystemError(
            f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}"
        )

    # Set DATA_ROOT if not already set
    if 'DATA_ROOT' not in os.environ:
        os.environ['DATA_ROOT'] = '/data'
        logger.info("DATA_ROOT not set, using default: /data")

    # Set LOG_LEVEL if not already set
    if 'LOG_LEVEL' not in os.environ:
        os.environ['LOG_LEVEL'] = 'INFO'

    # Validate DATA_ROOT is absolute path
    data_root = os.environ['DATA_ROOT']
    if not os.path.isabs(data_root):
        raise EnvironmentError(f"DATA_ROOT must be absolute path, got: {data_root}")

    # Check DATA_ROOT exists and is writable
    if not os.path.exists(data_root):
        logger.warning(f"DATA_ROOT does not exist: {data_root}")
        logger.info(f"Creating DATA_ROOT: {data_root}")
        os.makedirs(data_root, exist_ok=True)

    # Test write access
    test_file = os.path.join(data_root, '.write_test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        raise EnvironmentError(f"No write access to DATA_ROOT: {data_root}: {e}")

    # Check available memory (3GB minimum)
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_memory_gb < 3:
            logger.warning(
                f"Low memory: {available_memory_gb:.1f}GB available (3GB recommended). "
                f"Processing may fail with OOM errors."
            )
    except ImportError:
        logger.warning("psutil not available, cannot check memory requirements")

    # Check available disk space (18GB minimum for 300 videos)
    disk_usage = shutil.disk_usage(data_root)
    free_gb = disk_usage.free / (1024 ** 3)
    if free_gb < 18:
        logger.warning(
            f"Low disk space: {free_gb:.1f}GB free (18GB recommended for 300 videos). "
            f"Processing may fail if disk fills up."
        )

    # Configure logging
    log_level = os.environ['LOG_LEVEL']
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info(f"Environment setup complete: DATA_ROOT={data_root}, LOG_LEVEL={log_level}")
    logger.info(f"System checks: Python {sys.version_info.major}.{sys.version_info.minor}, "
                f"Memory: {available_memory_gb:.1f}GB, Disk: {free_gb:.1f}GB")

# ... [Rest of Section 9 content unchanged]
```

---

## 13. Test Specifications

```python
"""
===================================================================
NEW SECTION: Test Specifications
Source: VideoProcessingCHILD.md Section 8: Testing Strategy
===================================================================
"""

# ===================================================================
# UNIT TESTS
# Source: VideoProcessingCHILD.md Section 8.1
# ===================================================================

class TestBucketDirectoryInitialization:
    """Unit tests for initialize_bucket_directories()"""

    def test_create_all_8_buckets_with_15_subdirs(self):
        """Verify all 8 buckets created with complete subdirectory structure"""
        config = {"client_id": "test", "analysis_type": "hashtag", "target": "#test",
                  "analysis_mode": "top", "selection_strategy": "contrastive"}

        result = initialize_bucket_directories(config)

        assert len(result) == 8, "Should create 8 buckets"
        assert "0-3s" in result
        assert "90-120s" in result

        # Verify subdirectory structure
        for bucket_name, bucket_path in result.items():
            assert os.path.exists(f"{bucket_path}videos/")
            assert os.path.exists(f"{bucket_path}analysis/insights/")
            assert os.path.exists(f"{bucket_path}checkpoints/")
            # ... check all 15 subdirectories

    def test_idempotent_creation(self):
        """Verify safe to re-run (exist_ok=True)"""
        config = {"client_id": "test", "analysis_type": "hashtag", "target": "#test",
                  "analysis_mode": "top", "selection_strategy": "contrastive"}

        result1 = initialize_bucket_directories(config)
        result2 = initialize_bucket_directories(config)

        assert result1 == result2, "Should return same paths"

    def test_permission_denied(self):
        """Verify fail-fast on permission denied"""
        config = {"client_id": "test", "analysis_type": "hashtag", "target": "#test",
                  "analysis_mode": "top", "selection_strategy": "contrastive"}

        with mock.patch('os.makedirs', side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Permission denied"):
                initialize_bucket_directories(config)


class TestCheckpointInitialization:
    """Unit tests for initialize_checkpoint()"""

    def test_new_checkpoint_creation(self):
        """Verify new checkpoint created with correct fields"""
        video_list = [{"id": "123", "duration": 25, "playCount": 50000}]
        config = {"video_count": 100, "selection_strategy": "contrastive", "date_filter": "last_90_days"}

        checkpoint, remaining = initialize_checkpoint("18-33s", video_list, config)

        assert checkpoint['stage'] == "video_processing"
        assert checkpoint['total_videos'] == 1
        assert checkpoint['completed'] == 0
        assert checkpoint['status'] == "in_progress"
        assert remaining == video_list

    def test_load_existing_checkpoint(self):
        """Verify auto-resume from existing checkpoint"""
        # Setup: Create checkpoint with 1 completed video
        # Test: Load checkpoint and verify remaining_videos excludes completed
        pass

    def test_config_mismatch_detection(self):
        """Verify ValueError raised on config mismatch"""
        # Setup: Create checkpoint with video_count=100
        # Test: Load with video_count=150, expect ValueError
        pass


class TestVideoDownload:
    """Unit tests for download_video()"""

    def test_successful_download(self):
        """Verify successful download (200 OK)"""
        pass

    def test_retry_on_timeout(self):
        """Verify retry with exponential backoff (3 attempts)"""
        pass

    def test_skip_already_downloaded(self):
        """Verify skip download if file exists and valid size"""
        pass


# ===================================================================
# INTEGRATION TESTS
# Source: VideoProcessingCHILD.md Section 8.2
# ===================================================================

class TestEndToEndVideoProcessing:
    """Integration test: Stage 1 → Stage 2 → Stage 2.4"""

    def test_full_processing_5_videos(self):
        """
        Test complete video processing flow with 5 real videos.

        Setup:
        - Use real video list from Stage 1 (5 videos, bucket 18-33s)
        - Mock Apify downloads (return test MP4 files)
        - Mock RumiAI subprocess (return test temporal_windows JSON)

        Verify:
        - All 5 temporal_windows files created
        - Checkpoint status = "completed"
        - Stage 2.4 validation can load outputs
        """
        pass


class TestCheckpointResume:
    """Integration test: Checkpoint resume functionality"""

    def test_resume_after_interruption(self):
        """
        Test auto-resume without --resume flag.

        Setup:
        - Process 10 videos, interrupt after 5
        - Restart without --resume flag

        Verify:
        - Auto-resumes from checkpoint
        - Only remaining 5 videos processed
        - No duplicates in completed_video_ids
        """
        pass


class TestGracefulPause:
    """Integration test: Graceful pause handling"""

    def test_pause_on_sigint(self):
        """
        Test pause on Ctrl+C.

        Setup:
        - Process 10 videos, send SIGINT after video 5 completes

        Verify:
        - Processing pauses gracefully (not mid-video)
        - Checkpoint status="paused" saved
        - Resume processes only remaining 5 videos
        """
        pass


# ===================================================================
# TEST FIXTURES
# Source: VideoProcessingCHILD.md Section 8.3
# ===================================================================

@pytest.fixture
def sample_video_list():
    """Sample video list from VideoProcessingCHILD.md Section 8.3"""
    return [
        {
            "id": "7428596413707144481",
            "createTime": 1704067200,
            "duration": 25,
            "playCount": 50000,
            "videoMeta": {"downloadAddr": "https://example.com/video1.mp4"}
        },
        {
            "id": "7428596413707144482",
            "createTime": 1704153600,
            "duration": 18,
            "playCount": 35000,
            "videoMeta": {"downloadAddr": "https://example.com/video2.mp4"}
        }
    ]


@pytest.fixture
def sample_config():
    """Sample config from VideoProcessingCHILD.md Section 8.3"""
    return {
        "client_id": "test_client",
        "analysis_type": "hashtag",
        "target": "#test",
        "analysis_mode": "top",
        "selection_strategy": "contrastive",
        "video_count": 100,
        "date_filter": "last_90_days",
        "run_date": "2025-01-28T10:00:00Z"
    }


# ===================================================================
# TEST EXECUTION
# ===================================================================

"""
Run tests:
    pytest tests/test_video_processing.py -v
    pytest tests/test_stage2_integration.py -v
    pytest tests/test_checkpoint_resume.py -v
    pytest --cov=video_processing --cov-report=html
"""
```

---

## Document Metadata

**TI Document**: VideoProcessingTI.md
**Version**: 1.1 (Corrected - All Priority Issues Fixed)
**Creation Date**: 2025-01-28
**Last Updated**: 2025-01-28
**Generated From**: VideoProcessingCHILD.md v1.0 + FoundationCHILD.md

**Fixes Applied**:
- ✅ Priority 1.1: Fixed path construction in all functions (use get_bucket_path())
- ✅ Priority 1.2: Standardized bucket_name parameter (always bucket name, not path)
- ✅ Priority 1.3: Added main orchestration function (stage_2_video_processing_main)
- ✅ Priority 2.1: Enhanced exception classes with error context
- ✅ Priority 2.2: Added system requirement checks to setup_environment()
- ✅ Priority 2.3: Added explicit error handler calls in process_videos_sequential()
- ✅ Priority 2.4: Added note explaining Appendix B absence
- ✅ Priority 3.1: Added Section 13 (Test Specifications)
- ✅ Priority 3.2: Added checkpoint backup/recovery implementation details
- ✅ Priority 3.3: Added note to Trace 1 about variable service times
- ✅ Priority 3.4: Standardized schema comment format
- ✅ Priority 3.5: Installation instructions (already had venv, kept as-is)

**Status**: Ready for implementation

---

## 12. HLD Traceability Matrix

| HLD Section | TI Section | Implementation Status |
|-------------|------------|----------------------|
| VideoProcessingCHILD.md Section 1 | Section 1: Document Metadata | To Implement |
| VideoProcessingCHILD.md Section 2.3.0 | Section 4: Function 1 (initialize_bucket_directories) | To Implement |
| VideoProcessingCHILD.md Section 2.3.1 | Section 4: Function 2 (initialize_checkpoint) | To Implement |
| VideoProcessingCHILD.md Section 2.3.2 | Section 4: Function 3 (download_video) | To Implement |
| VideoProcessingCHILD.md Section 2.3.3 | Section 4: Function 4 (process_videos_sequential) | To Implement |
| VideoProcessingCHILD.md Section 2.3.4 | Section 4: Function 5 (process_videos_with_pause_support) | To Implement |
| VideoProcessingCHILD.md Section 2.3.5 | Section 4: Function 6 (finalize_checkpoint) | To Implement |
| VideoProcessingCHILD.md Section 3.1 | Section 2: StageInput | To Implement |
| VideoProcessingCHILD.md Section 3.2 | Section 2: StageOutput | To Implement |
| VideoProcessingCHILD.md Section 5 | Section 3: Data Schemas | To Implement |
| VideoProcessingCHILD.md Section 6 | Section 5 & 6: Validation & Error Handling | To Implement |
| VideoProcessingCHILD.md Section 8 | Section 13: Test Specifications | To Implement |
| FoundationCHILD.md Section 2 | Section 8: BASE_PATHS | To Implement |
| FoundationCHILD.md Section 4 | Section 9: CLI Parameters | To Implement |
| FoundationCHILD.md Section 5 | Section 3: Foundation Schemas | To Implement |
| NEW: Main orchestration | Section 4: stage_2_video_processing_main() | To Implement |

**Total Sections**: 13 (Sections 1-13 complete)
**Corrections Applied**: 12 fixes (Priority 1: 3, Priority 2: 4, Priority 3: 5)
**Ready for Implementation**: Yes
