# Video Processing - High-Level Design

> **Parent**: MLPlanningv2.md - Section "Stage 2: Video Processing (RumiAI Pipeline)"
> **Version**: 1.0
> **Last Updated**: 2025-01-28
> **Status**: Draft

---

## 1. Context & Business Goal

<!-- PURPOSE: Provide business context and justification. TI generator needs to understand WHY this feature exists. -->

### 1.1 What Problem Does This Solve?

Long-running batch video analyses (6-8 hours for 300+ videos) must reliably process TikTok videos through RumiAI's ML pipeline despite interruptions (SSH disconnects, system crashes, manual stops). Without checkpoint-resume capability and sequential processing with failure tracking, interruptions waste hours of compute time and require re-processing hundreds of completed videos. This stage implements the core video processing workflow with automatic checkpointing, enabling reliable batch analysis for Tumi Labs' RippleOS consultancy.

### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This stage depends on FoundationCHILD.md for:
- Client directory structure (Section 2: Client Architecture & Storage)
- CLI parameter definitions (Section 4: CLI Command Structure)
- Config.json schema (Section 5.1: Configuration Schemas)
- Apify video metadata schema (Section 5.2: Apify Video Metadata Schema)
- Checkpoint schema (Section 5.3: Checkpoint Schema)

```
Stage 0: Configuration (see FoundationCHILD.md)
   ↓ CLI parameters + Directory structure + Config.json
Stage 1: Video Discovery & Selection
   ↓ Output: Selected video list (per bucket), video metadata from Apify
Stage 2: Video Processing (THIS FEATURE)
   ├── 2.1: Video Download
   ├── 2.2: Sequential RumiAI Processing
   ├── 2.3: Checkpoint System
   └── 2.4: Pipeline Validation
   ↓ Output: temporal_windows_updated.json (N files per bucket)
Stage 3: Feature Aggregation
```

### 1.3 Success Criteria

- [ ] Process up to 300 videos per bucket sequentially without data loss
- [ ] Checkpoint after each video completes (recovery on interruption)
- [ ] Auto-resume from last completed video when restarted (no --resume flag needed)
- [ ] Skip-on-fail policy (log failed videos, continue batch)
- [ ] Download videos via Apify with retry logic (max 3 download attempts per video)
- [ ] Generate `temporal_windows_updated.json` per video (60+ features per window)
- [ ] Processing completes within reasonable time for batch operations (see Section 7.2 for measured performance)

---

## 2. Architecture & Design

<!-- PURPOSE: Core technical design. This is the PRIMARY section TI generator reads. -->

### 2.1 High-Level Approach

This stage processes selected videos sequentially through RumiAI's existing analysis pipeline (`rumiai_runner.py`) with automatic checkpointing for failure recovery. Videos are downloaded via Apify's download URLs, processed one-at-a-time to generate temporal window features, and checkpointed after each successful completion. The checkpoint system enables automatic resume on interruption without user intervention, while failed videos are logged and skipped (not retried) to prevent batch stalls.

### 2.2 Data Flow

```
Input: Selected video list from Stage 1 (List[video_metadata])
       Video metadata from Apify (id, download_url, duration, engagement)
       Config.json from FoundationCHILD.md Section 5.1
   ↓
Process Step 0: Initialize Bucket Directories (NEW - Section 2.3.0)
       Create all 8 bucket directories with complete subdirectory structure
       Idempotent (safe to re-run on resume)
   ↓
Process Step 1: Load/Create Checkpoint (Section 2.3.1)
       Check if checkpoint exists for this bucket
       If exists → Load completed_video_ids, auto-resume
       If not exists → Create new checkpoint
   ↓
Process Step 2: Filter Completed Videos (Section 2.3.2)
       Remove already-processed videos from selected list
       Remaining videos = videos to process
   ↓
Process Step 3: Sequential Video Download (Section 2.3.3)
       For each video in remaining_videos:
         - Download MP4 via Apify download URL
         - Save to bucket/videos/{video_id}.mp4
         - Retry up to 3 times on failure
   ↓
Process Step 4: Sequential RumiAI Processing (Section 2.3.4)
       For each video in remaining_videos:
         - Run rumiai_runner.py (9 ML services)
         - Generate temporal_windows_updated.json
         - Save to bucket/analysis/insights/
         - Update checkpoint (mark completed)
   ↓
Process Step 5: Mark Analysis Complete (Section 2.3.5)
       Update checkpoint status to "completed"
       Log final statistics (completed, failed, duration)
   ↓
Output: temporal_windows_updated.json (N files)
        Location: bucket_{duration}/analysis/insights/{video_id}_temporal_windows_updated.json
        Schema: 60+ features per temporal window (hook, middle segments, closing)
        Checkpoint: bucket_{duration}/checkpoints/stage_2_checkpoint.json
```

### 2.3 Detailed Process

#### Step 2.3.0: Bucket Directory Initialization

**Purpose**: Create all 8 bucket directories with complete subdirectory structure before video processing begins

**When This Runs**:
- Immediately after Stage 1 completes (before any video downloads)
- Only runs once per analysis (idempotent - safe to re-run)
- Runs before checkpoint initialization to ensure infrastructure exists

**Which Buckets to Create**:

All 8 potential bucket directories are created upfront, even though only the top 3 winning buckets (identified by Stage 1.3) will be populated with videos:

```
bucket_0-3s/     # Created ✅ | Usually empty (ultra-short videos rarely win)
bucket_3-9s/     # Created ✅ | Usually empty (short hooks rarely win)
bucket_9-13s/    # Created ✅ | Sometimes populated (depends on winner distribution)
bucket_13-18s/   # Created ✅ | Often populated (TikTok sweet spot ~15s)
bucket_18-33s/   # Created ✅ | Often populated (medium-form content, high winner rate)
bucket_33-60s/   # Created ✅ | Often populated (long-form content, high winner rate)
bucket_60-90s/   # Created ✅ | Sometimes populated (extended content)
bucket_90-120s/  # Created ✅ | Usually empty (maximum TikTok length, rare winners)
```

**Rationale for Creating All 8**:
- **Infrastructure consistency**: Predictable directory structure regardless of which buckets contain videos
- **Stage 1 compatibility**: `winner_analysis.json` may reference buckets with 0 videos selected
- **Future-proofing**: Different targets have different winning bucket distributions
- **Minimal overhead**: Empty directories consume negligible disk space (~4KB × 8 = ~32KB total)
- **Simplifies downstream stages**: Stages 3-7 can iterate over all buckets without existence checks

**Complete Subdirectory Structure Per Bucket** (from FoundationCHILD.md Section 2.1):

```
bucket_{duration}/
├── videos/              # Raw MP4 files downloaded from Apify
├── analysis/
│   ├── insights/        # temporal_windows_updated.json (1 per video)
│   ├── unified/         # Intermediate timeline+ml_data from RumiAI
│   └── service_debug/   # ML service debug outputs (YOLO, FEAT, etc.)
├── validation/          # Pipeline validation outputs (Stage 2.4)
│   ├── rolling_stats.json
│   └── validation_summary.json
├── flagged_videos/      # Investigation packages for outliers
│   └── {video_id}/
│       ├── video.mp4
│       ├── temporal_windows_updated.json
│       ├── unified_analysis.json
│       ├── service_debug/
│       └── validation_report.json
├── ml_analysis/         # ML pipeline outputs (Stage 3+)
│   ├── aggregated_features.csv
│   ├── rf_transformed.csv
│   ├── km_transformed.csv
│   ├── random_forest_analysis.json
│   └── kmeans_analysis.json
├── models/              # Trained ML models (Stage 5)
│   ├── random_forest_v1.pkl
│   ├── kmeans_v1.pkl
│   ├── scalers.pkl
│   └── model_metrics.json
├── llm_reports/         # LLM-generated reports (Stage 7)
│   ├── analysis/        # LLM Call 1 (insight extraction)
│   │   ├── call_1_rf_prompt.txt
│   │   ├── call_1_rf_raw_response.json
│   │   ├── call_1_kmeans_prompt.txt
│   │   ├── call_1_kmeans_raw_response.json
│   │   └── insights.json
│   └── formatted/       # LLM Call 2 (report generation)
│       ├── call_2_prompt.txt
│       ├── call_2_raw_response.json
│       ├── rf_feature_importance.md
│       ├── strategy_1_the_educator.md
│       ├── strategy_2_visual_storyteller.md
│       ├── strategy_3_personal_journey.md
│       └── bucket_summary.md
├── reports/             # Final PDFs (Stage 7)
│   ├── rf_feature_importance.pdf
│   ├── strategy_1_the_educator.pdf
│   ├── strategy_2_visual_storyteller.pdf
│   ├── strategy_3_personal_journey.pdf
│   └── bucket_summary.pdf
├── checkpoints/         # Processing state checkpoints
│   └── stage_{X}_checkpoint.json
└── logs/                # Processing logs
    └── processing_{date}.log
```

**Logic**:
```python
def initialize_bucket_directories(config):
    """
    Create all 8 bucket directories with complete subdirectory structure.

    This function creates directories for ALL 8 potential buckets, even though
    only the top 3 winning buckets will be populated with videos. This ensures
    consistent infrastructure and simplifies downstream stages.

    Args:
        config: dict, loaded from config.json (FoundationCHILD Section 5.1)

    Returns:
        dict: Created paths mapped to bucket names

    Raises:
        OSError: If directory creation fails (permissions, disk space)
    """
    # Define all 8 bucket names (from FoundationCHILD Section 6)
    BUCKET_NAMES = [
        "0-3s",
        "3-9s",
        "9-13s",
        "13-18s",
        "18-33s",
        "33-60s",
        "60-90s",
        "90-120s"
    ]

    # Base path template (from FoundationCHILD Section 2.2)
    data_root = os.getenv('DATA_ROOT', '/data')
    analysis_base = f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"

    # Subdirectory structure per bucket (from FoundationCHILD Section 2.1, lines 113-164)
    SUBDIRECTORIES = [
        "videos/",
        "analysis/",
        "analysis/insights/",
        "analysis/unified/",
        "analysis/service_debug/",
        "validation/",
        "flagged_videos/",
        "ml_analysis/",
        "models/",
        "llm_reports/",
        "llm_reports/analysis/",
        "llm_reports/formatted/",
        "reports/",
        "checkpoints/",
        "logs/"
    ]

    created_paths = {}

    # Create all 8 buckets with full subdirectory structure
    for bucket_name in BUCKET_NAMES:
        bucket_path = f"{analysis_base}buckets/bucket_{bucket_name}/"

        logger.info(f"Creating bucket directory structure: bucket_{bucket_name}")

        # Create bucket root
        try:
            os.makedirs(bucket_path, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create bucket directory {bucket_path}: {e}")

        # Create all subdirectories for this bucket
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


def validate_bucket_structure(bucket_path, bucket_name):
    """
    Validate that bucket directory has complete subdirectory structure.

    Used for verification after initialization or when resuming from checkpoint.

    Args:
        bucket_path: str, full path to bucket directory
        bucket_name: str, bucket name (e.g., "18-33s")

    Raises:
        ValueError: If any required subdirectories are missing
    """
    REQUIRED_SUBDIRS = [
        "videos/",
        "analysis/insights/",
        "analysis/unified/",
        "analysis/service_debug/",
        "checkpoints/"
    ]

    missing = []
    for subdir in REQUIRED_SUBDIRS:
        subdir_path = f"{bucket_path}{subdir}"
        if not os.path.exists(subdir_path):
            missing.append(subdir)

    if missing:
        raise ValueError(
            f"Bucket {bucket_name} missing required subdirectories: {missing}. "
            f"Run bucket initialization before processing."
        )
```

**Example Output** (typical hashtag analysis):
```
Stage 1.3: Winner Analysis Complete
  Top 3 buckets identified:
    1. bucket_18-33s (45% of winners)
    2. bucket_33-60s (30% of winners)
    3. bucket_13-18s (20% of winners)

Stage 2.0: Initializing Bucket Directories
  Creating bucket directory structure: bucket_0-3s
    ✓ Created 15 subdirectories for bucket_0-3s
  Creating bucket directory structure: bucket_3-9s
    ✓ Created 15 subdirectories for bucket_3-9s
  Creating bucket directory structure: bucket_9-13s
    ✓ Created 15 subdirectories for bucket_9-13s
  Creating bucket directory structure: bucket_13-18s
    ✓ Created 15 subdirectories for bucket_13-18s
  Creating bucket directory structure: bucket_18-33s
    ✓ Created 15 subdirectories for bucket_18-33s
  Creating bucket directory structure: bucket_33-60s
    ✓ Created 15 subdirectories for bucket_33-60s
  Creating bucket directory structure: bucket_60-90s
    ✓ Created 15 subdirectories for bucket_60-90s
  Creating bucket directory structure: bucket_90-120s
    ✓ Created 15 subdirectories for bucket_90-120s
  ✓ Successfully created all 8 bucket directories

Stage 2.1-2.3: Processing Videos (Winning Buckets Only)
  → Processing bucket_18-33s (100 videos)
  → Processing bucket_33-60s (100 videos)
  → Processing bucket_13-18s (100 videos)
  ✓ Skipping bucket_0-3s (0 videos selected - not a winning bucket)
  ✓ Skipping bucket_3-9s (0 videos selected - not a winning bucket)
  ✓ Skipping bucket_9-13s (0 videos selected - not a winning bucket)
  ✓ Skipping bucket_60-90s (0 videos selected - not a winning bucket)
  ✓ Skipping bucket_90-120s (0 videos selected - not a winning bucket)
```

**Edge Cases**:

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Directories already exist | Skip creation (idempotent via `exist_ok=True`) | Safe to re-run after interruption |
| Insufficient disk space | Fail-fast with OSError | Cannot proceed without storage |
| Missing parent directories | Auto-create parents via `os.makedirs()` | Graceful handling of fresh installs |
| Permission denied | Fail-fast with OSError | User must fix permissions before proceeding |
| Partial directory structure exists | Complete missing subdirectories | Resume-friendly (creates only what's missing) |

**Validation on Resume**:

When resuming from checkpoint, validate bucket structure before processing:

```python
def resume_video_processing(checkpoint, config):
    """Resume video processing from checkpoint."""
    bucket_name = checkpoint['bucket']
    bucket_path = get_bucket_path(config, bucket_name)

    # Validate bucket structure before resuming
    try:
        validate_bucket_structure(bucket_path, bucket_name)
    except ValueError as e:
        logger.error(f"Bucket structure validation failed: {e}")
        logger.info("Re-initializing bucket directories...")
        initialize_bucket_directories(config)

    # Continue with normal resume logic
    remaining_videos = load_remaining_videos(checkpoint)
    # ...
```

---

#### Step 2.3.1: Checkpoint Initialization

**Purpose**: Load existing checkpoint or create new one, enabling auto-resume on interruption

**Logic**:
```python
# Exception Classes
class DownloadError(Exception):
    """Raised when video download fails after max retry attempts"""
    pass

class ProcessingError(Exception):
    """Raised when RumiAI pipeline fails"""
    pass

class ValidationError(Exception):
    """Raised when output schema validation fails"""
    pass


# Utility Functions
def get_bucket_path(config, bucket_name):
    """
    Construct full bucket directory path from config and bucket name.

    Naming convention:
    - bucket_name: Duration range only (e.g., "18-33s")
    - bucket_path: Full directory path (e.g., "/data/clients/acme/hashtags/#nutrition/bucket_18-33s/")

    All functions in this document use bucket_name as parameter and construct
    full paths using this helper when needed.

    Args:
        config: dict, loaded from config.json (FoundationCHILD Section 5.1)
        bucket_name: str, duration range (e.g., "18-33s")

    Returns:
        str: Full bucket directory path with trailing slash

    Example:
        config = {"client_id": "acme", "analysis_type": "hashtag", "target": "#nutrition"}
        bucket_name = "18-33s"
        returns: "/data/clients/acme/hashtags/#nutrition/bucket_18-33s/"
    """
    data_root = os.getenv('DATA_ROOT', '/data')
    return f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/{config['target']}/bucket_{bucket_name}/"


def save_json(filepath, data):
    """
    Save dictionary to JSON file.

    Requirements for TI implementation:
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
    """
    pass  # TI: Implement based on requirements above


def load_json(filepath):
    """
    Load JSON file to dictionary.

    Requirements for TI implementation:
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
    """
    pass  # TI: Implement based on requirements above


def save_checkpoint_with_backup(checkpoint_path, checkpoint):
    """
    Save checkpoint with automatic backup to prevent data loss from corruption.

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


def load_checkpoint_with_recovery(checkpoint_path):
    """
    Load checkpoint with automatic recovery from backup on corruption.

    Args:
        checkpoint_path: str, path to checkpoint file

    Returns:
        checkpoint: dict, loaded checkpoint data

    Raises:
        ValueError: if both checkpoint and backup are corrupted
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

        # Both corrupted - fail with clear message
        raise ValueError(
            f"Checkpoint and backup both corrupted. Use --force to restart from beginning.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Backup: {backup_path}"
        )


def initialize_checkpoint(bucket, video_list, config):
    """
    Initialize checkpoint for video processing stage.

    Checkpoint granularity: Video-level (saves after each video completes).
    - If interrupted during video processing, that video's progress is lost.
    - Resume starts from next video (last completed + 1).
    - Max work lost per interruption: 1 video (60-140s).

    Naming convention: 'bucket' parameter is the bucket name (duration range),
    not full path. Use get_bucket_path() to construct full paths when needed.

    Args:
        bucket: str, bucket name only (e.g., "18-33s"), not full path
        video_list: list, videos selected by Stage 1
        config: dict, loaded from config.json (FoundationCHILD Section 5.1)

    Returns:
        checkpoint: dict, checkpoint data (new or existing)
        remaining_videos: list, videos to process (excludes completed)
    """

    def validate_config_match(checkpoint_config: dict, current_config: dict):
        """
        Validates that checkpoint config matches current run config.
        Prevents resuming with different parameters that would corrupt analysis.

        Raises:
            ValueError: If critical parameters don't match
        """
        critical_fields = ['video_count', 'selection_strategy', 'date_filter']
        mismatches = []

        for field in critical_fields:
            if checkpoint_config.get(field) != current_config.get(field):
                mismatches.append(f"{field}: checkpoint={checkpoint_config.get(field)}, current={current_config.get(field)}")

        if mismatches:
            raise ValueError(f"Config mismatch detected. Cannot resume with different parameters:\n" + "\n".join(mismatches))

    checkpoint_path = f"{bucket}/checkpoints/stage_2_checkpoint.json"

    # Load existing checkpoint (auto-resume with corruption recovery)
    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint_with_recovery(checkpoint_path)

        # Validate config matches (prevent resume with different parameters)
        validate_config_match(checkpoint['config'], config)

        # Filter out completed videos
        completed_ids = set(checkpoint['completed_video_ids'])
        remaining_videos = [v for v in video_list if v['id'] not in completed_ids]

        logger.info(f"Checkpoint detected: {checkpoint['completed']}/{checkpoint['total_videos']} completed")
        logger.info(f"Auto-resuming from bucket {bucket} ({len(remaining_videos)} remaining)")

        return checkpoint, remaining_videos

    # Create new checkpoint
    else:
        checkpoint = {
            "stage": "video_processing",
            "bucket": bucket,
            "total_videos": len(video_list),
            "completed": 0,
            "failed": 0,
            "remaining": len(video_list),  # Invariant: remaining = total - completed - failed
            "last_checkpoint": datetime.utcnow().isoformat(),
            "completed_video_ids": [],
            "failed_video_ids": [],
            "config": config,  # Store config for validation on resume
            "status": "in_progress",  # State machine: in_progress -> paused/completed
            "pause_reason": None,
            "pause_timestamp": None
        }

        save_checkpoint_with_backup(checkpoint_path, checkpoint)
        logger.info(f"Created new checkpoint for bucket {bucket} ({len(video_list)} videos)")

        return checkpoint, video_list
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Checkpoint exists with status "completed" | Auto-resume (remaining_videos will be empty, skips processing) | All videos already processed, batch completes immediately |
| Config mismatch on resume | Raise ValueError with field details (via validate_config_match) | Prevent resume with different video_count or strategy |
| Corrupted checkpoint JSON | Auto-restore from .backup.json file (via load_checkpoint_with_recovery) | Automatic recovery from checkpoint corruption |
| No checkpoint exists | Create new checkpoint with status="in_progress" | Fresh start, initialize tracking |

#### Step 2.3.2: Video Download

**Purpose**: Download videos from Apify download URLs with retry logic

**Logic**:
```python
def download_video(video_metadata, output_dir, max_attempts=3):
    """
    Download video MP4 from Apify download URL.

    Args:
        video_metadata: dict, Apify metadata (FoundationCHILD Section 5.2)
        output_dir: str, path to bucket/videos/ directory
        max_attempts: int, max download attempts (default: 3)

    Returns:
        video_path: str, path to downloaded MP4

    Raises:
        DownloadError: if download fails after max_attempts
    """
    video_id = video_metadata['id']
    download_url = video_metadata['videoMeta']['downloadAddr']
    output_path = f"{output_dir}/{video_id}.mp4"

    # Validate existing file if present (resume scenario)
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)

        if file_size < MIN_VIDEO_SIZE:
            # Corrupt cached file (e.g., interrupted download) - re-download
            logger.warning(
                f"Existing file for {video_id} too small ({file_size} bytes), "
                f"removing and re-downloading"
            )
            os.remove(output_path)
            # Fall through to download logic below
        else:
            # Valid cached file - skip download
            logger.info(f"Video {video_id} already downloaded and valid ({file_size} bytes), skipping")
            return output_path

    # Retry loop
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Downloading video {video_id} (attempt {attempt}/{max_attempts})")

            # Download via requests with streaming
            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()

            # Write to file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Validate file size (catches error pages, truncated headers, empty files)
            file_size = os.path.getsize(output_path)
            if file_size < MIN_VIDEO_SIZE:  # < 100KB suggests corrupt download
                raise DownloadError(
                    f"Downloaded file too small: {file_size} bytes "
                    f"(minimum: {MIN_VIDEO_SIZE} bytes / {MIN_VIDEO_SIZE / 1024:.0f} KB)"
                )

            logger.info(f"Successfully downloaded video {video_id} ({file_size / 1024 / 1024:.2f} MB)")
            return output_path

        except (requests.exceptions.RequestException, DownloadError) as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")

            # Clean up partial download
            if os.path.exists(output_path):
                os.remove(output_path)

            # Retry with exponential backoff
            if attempt < max_attempts:
                sleep_time = 2 ** attempt  # 2s, 4s, 8s
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                # Max attempts exceeded
                raise DownloadError(f"Failed to download video {video_id} after {max_attempts} attempts: {e}")
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Video already downloaded (valid size) | Skip download, return existing path | Resume optimization |
| Video already downloaded (< 100KB) | Remove file, re-download | Corrupt cached file from interrupted download |
| Download URL expired (404) | Mark as failed, continue batch | TikTok deleted video, unrecoverable |
| Network timeout | Retry with exponential backoff | Transient network issues |
| File too small (< 100KB) after download | Retry from scratch (max 3 attempts) | Catches error pages, truncated headers, incomplete transfers |
| Disk full | Fail-fast with clear error | Cannot proceed, user must free space |

#### Step 2.3.3: Sequential RumiAI Processing

**Purpose**: Process each video through RumiAI pipeline to generate temporal window features

**Logic**:
```python
def process_videos_sequential(remaining_videos, bucket, checkpoint):
    """
    Process videos sequentially through RumiAI pipeline with checkpointing.

    Args:
        remaining_videos: list, videos to process (excludes completed)
        bucket: str, bucket name (e.g., "18-33s")
        checkpoint: dict, checkpoint data from Step 2.3.1

    Returns:
        results: dict, processing statistics
    """
    insights_dir = f"{bucket}/analysis/insights/"
    checkpoint_path = f"{bucket}/checkpoints/stage_2_checkpoint.json"

    # Process each video
    for i, video in enumerate(remaining_videos, start=1):
        video_id = video['id']
        video_path = f"{bucket}/videos/{video_id}.mp4"

        logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id}")

        try:
            # Run RumiAI pipeline with timeout enforcement
            # 300s wrapper timeout enforces maximum processing time per video
            result = run_rumiai_pipeline(
                video_path=video_path,
                video_id=video_id,
                output_dir=f"{bucket}/analysis/",
                timeout=300  # 5-minute max processing time per video
            )

            # Validate output exists
            insights_path = f"{insights_dir}/{video_id}_temporal_windows_updated.json"
            if not os.path.exists(insights_path):
                raise ProcessingError(f"RumiAI did not generate insights for {video_id}")

            # Validate output schema (basic check)
            insights = load_json(insights_path)
            validate_temporal_windows_schema(insights)

            # Mark as completed in checkpoint
            checkpoint['completed'] += 1
            checkpoint['remaining'] -= 1
            checkpoint['completed_video_ids'].append(video_id)
            checkpoint['last_checkpoint'] = datetime.utcnow().isoformat()

            # Save checkpoint after each video (failure recovery with backup)
            save_checkpoint_with_backup(checkpoint_path, checkpoint)

            logger.info(f"Successfully processed video {video_id} ({checkpoint['completed']}/{checkpoint['total_videos']})")

        except Exception as e:
            # Log failure, continue batch (skip-on-fail policy)
            logger.error(f"Failed to process video {video_id}: {e}")

            checkpoint['failed'] += 1
            checkpoint['remaining'] -= 1
            checkpoint['failed_video_ids'].append({
                "video_id": video_id,
                "error": str(e),
                "error_type": type(e).__name__,  # Exception type for debugging
                "timestamp": datetime.utcnow().isoformat()
            })

            # Save checkpoint with failure (with backup)
            save_checkpoint_with_backup(checkpoint_path, checkpoint)

            # Continue to next video (skip-on-fail)
            continue

    # Return final statistics
    return {
        "total": checkpoint['total_videos'],
        "completed": checkpoint['completed'],
        "failed": checkpoint['failed']
    }


def run_rumiai_pipeline(video_path, video_id, output_dir, timeout=300):
    """
    Run existing RumiAI pipeline (rumiai_runner.py wrapper) with timeout enforcement.

    Success validation strategy:
    1. Subprocess exits with code 0 (check=True raises CalledProcessError if exit != 0)
    2. Insights file exists at expected path
    3. Insights file passes schema validation (handled by caller at Line 464)

    IMPORTANT: rumiai_runner.py does NOT output JSON to stdout. Contract is file-based.
    Stdout/stderr are logged for debugging only (warnings, progress, errors).

    Args:
        video_path: str, path to downloaded MP4
        video_id: str, TikTok video ID
        output_dir: str, base output directory for analysis
        timeout: int, maximum processing time in seconds (default: 300s)

    Returns:
        dict: {"status": "success", "video_id": video_id, "insights_path": path}

    Raises:
        ProcessingError: if RumiAI pipeline fails or insights file missing
        TimeoutError: if processing exceeds timeout
    """
    import subprocess
    import sys

    insights_path = f"{output_dir}/insights/{video_id}_temporal_windows_updated.json"

    cmd = [
        sys.executable,  # python3
        'rumiai_runner.py',
        video_path  # rumiai_runner.py accepts URL or path as positional arg
    ]

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,  # Kill process if exceeds 300s
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError if exit code != 0
        )

        # Log stdout/stderr for debugging (don't parse - no JSON contract)
        if result.stdout:
            logger.debug(f"RumiAI stdout: {result.stdout[:500]}")
        if result.stderr:
            logger.warning(f"RumiAI stderr: {result.stderr[:500]}")

        # Validate insights file exists (ground truth for success)
        if not os.path.exists(insights_path):
            raise ProcessingError(
                f"RumiAI exited successfully (code 0) but no insights file at {insights_path}"
            )

        return {
            "status": "success",
            "video_id": video_id,
            "insights_path": insights_path
        }

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"RumiAI processing exceeded {timeout}s timeout for video {video_id}")
    except subprocess.CalledProcessError as e:
        raise ProcessingError(
            f"RumiAI pipeline failed for {video_id} (exit code {e.returncode}). "
            f"Stderr: {e.stderr[:200] if e.stderr else 'none'}"
        )


def validate_temporal_windows_schema(insights):
    """
    Validate temporal_windows_updated.json structure and completeness.

    Requirements spec for TI implementation. Validates that RumiAI output
    matches expected schema without duplicating feature definitions from
    SystemArchitecturev2.md (lines 395-460).

    Args:
        insights: dict, loaded temporal_windows JSON

    Raises:
        ValidationError: if schema invalid
    """
    # 1. Check required top-level keys
    required_keys = ['temporal_windows', 'metadata', 'processing_timestamp']
    missing = [k for k in required_keys if k not in insights]
    if missing:
        raise ValidationError(f"Missing top-level keys: {missing}")

    # 2. Validate temporal_windows structure
    windows = insights['temporal_windows']
    if not isinstance(windows, dict):
        raise ValidationError(f"temporal_windows must be dict, got {type(windows).__name__}")

    # 3. Check required window sections exist and are dicts
    required_sections = ['hook', 'closing']
    for section in required_sections:
        if section not in windows:
            raise ValidationError(f"Missing required window section: {section}")
        if not isinstance(windows[section], dict):
            raise ValidationError(f"Window section '{section}' must be dict, got {type(windows[section]).__name__}")

        # 4. Check feature count (expect 60+ features per window)
        # Feature names/types defined in SystemArchitecturev2.md - validate count only
        if len(windows[section]) < 50:
            raise ValidationError(f"Window section '{section}' has only {len(windows[section])} features (expected 60+)")

    # 5. Validate middle_segments logic (null for short videos ≤9s, list otherwise)
    video_duration = insights['metadata'].get('duration', 0)
    middle_segments = windows.get('middle_segments')

    if video_duration <= 9:
        # Short videos: middle_segments should be null
        if middle_segments is not None:
            raise ValidationError(f"Video duration {video_duration}s ≤ 9s but middle_segments is not null")
    else:
        # Longer videos: middle_segments should be list of dicts
        if not isinstance(middle_segments, list):
            raise ValidationError(f"Video duration {video_duration}s > 9s but middle_segments is {type(middle_segments).__name__}, expected list")

        # Each middle segment should be a dict with features
        for i, segment in enumerate(middle_segments):
            if not isinstance(segment, dict):
                raise ValidationError(f"Middle segment {i} must be dict, got {type(segment).__name__}")
            if len(segment) < 50:
                raise ValidationError(f"Middle segment {i} has only {len(segment)} features (expected 60+)")

    # 6. Validate metadata structure
    metadata = insights['metadata']
    if not isinstance(metadata, dict):
        raise ValidationError(f"metadata must be dict, got {type(metadata).__name__}")

    required_metadata = ['video_id', 'duration', 'analysis_timestamp']
    missing_meta = [k for k in required_metadata if k not in metadata]
    if missing_meta:
        raise ValidationError(f"Missing metadata fields: {missing_meta}")

    # 7. Validate timestamp format
    timestamp = insights['processing_timestamp']
    if not isinstance(timestamp, str):
        raise ValidationError(f"processing_timestamp must be string, got {type(timestamp).__name__}")

    # Note: Detailed feature name/type validation deferred to TI implementation
    # See SystemArchitecturev2.md (lines 395-460) for complete feature specifications
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| RumiAI subprocess exits with error (exit != 0) | Mark as failed, continue batch | Skip-on-fail prevents batch stall |
| RumiAI subprocess timeout (> 300s) | Mark as failed, continue batch | Wrapper timeout prevents indefinite hangs |
| RumiAI exits 0 but no insights file | Mark as failed, continue batch | File is ground truth for success |
| Out of memory during processing | Mark as failed, continue batch | Service-specific issue, other videos may succeed |
| Invalid temporal_windows schema | Mark as failed, continue batch | RumiAI bug, needs investigation |
| SSH disconnect mid-video | Resume from last checkpoint | Checkpoint saved after each video |
| User Ctrl+C mid-video | Graceful pause after current video | See Step 2.3.4 for pause handling |

#### Step 2.3.4: Graceful Pause Handling

**Purpose**: Allow user to manually pause processing gracefully (finish current video, then pause)

**Logic**:
```python
import signal
import platform

# Global pause flag (signal-safe for single-threaded processing)
pause_requested = False

def request_pause(signum, frame):
    """
    Signal handler for graceful pause.

    Thread-safety note: Global flag is safe for single-threaded sequential
    video processing. Flag is only checked between videos, not during processing.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global pause_requested

    if not pause_requested:
        pause_requested = True
        logger.info("\n⏸️  Pause requested. Will pause after current video completes...")
        logger.info("   Press Ctrl+C again to force quit (current video will be lost)")
    else:
        # Second Ctrl+C = force quit
        logger.warning("\n❌ Force quit requested. Current video will NOT be saved.")
        logger.warning("   Checkpoint may be inconsistent if killed mid-write.")
        sys.exit(1)


def process_videos_with_pause_support(remaining_videos, bucket, checkpoint):
    """
    Process videos with graceful pause support.

    Integrates with Step 2.3.3 processing loop.

    Args:
        remaining_videos: list, videos to process
        bucket: str, bucket name
        checkpoint: dict, checkpoint data
    """
    global pause_requested

    # Register signal handler (cross-platform SIGINT support)
    signal.signal(signal.SIGINT, request_pause)   # Ctrl+C

    # Platform compatibility note
    if platform.system() == 'Windows':
        logger.warning("Graceful pause (Ctrl+C) has limited support on Windows")
        logger.warning("Checkpoint auto-save after each video provides recovery")

    checkpoint_path = f"{bucket}/checkpoints/stage_2_checkpoint.json"

    # Process each video (from Step 2.3.3)
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

            return  # Exit gracefully

        # Process video normally (Step 2.3.3 logic)
        video_id = video['id']
        logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id}")

        # ... (rest of Step 2.3.3 processing logic)
```

**User Experience**:
```bash
# Terminal output during pause:
Processing video 47/100: 7428596413707144481
  ✓ Downloaded (25.3 MB)
  ✓ RumiAI processing...

# User presses Ctrl+C (first time)
^C
⏸️  Pause requested. Will pause after current video completes...
   Press Ctrl+C again to force quit (current video will be lost)

# Video 47 continues processing...
  ✓ RumiAI complete (82.4s)
  ✓ Checkpoint updated (47/100)

⏸️  Paused gracefully after 47/100 videos
   Resume anytime by re-running the same command
   Checkpoint: bucket_18-33s/checkpoints/stage_2_checkpoint.json
```

**Alternative Pause Method (Unix)**:
```bash
# In another terminal, send pause signal:
ps aux | grep rumiai_ml_batch  # Get process ID
kill -USR1 <process_id>        # Send graceful pause signal
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Ctrl+C during video processing | Finish current video, then pause | Avoid wasting processing time |
| Ctrl+C twice (force quit) | Immediate exit, current video lost | User wants immediate stop |
| Pause requested on last video | Complete last video, mark as completed | No pause needed |
| Resume from paused checkpoint | Auto-resume from next video | Same as interruption resume |

#### Step 2.3.5: Mark Analysis Complete

**Purpose**: Finalize checkpoint status and log statistics

**Logic**:
```python
def finalize_checkpoint(checkpoint):
    """
    Mark video processing stage as complete.

    Args:
        checkpoint: dict, checkpoint data after all videos processed
    """
    checkpoint['status'] = 'completed'
    checkpoint['completion_time'] = datetime.utcnow().isoformat()

    save_json(f"{checkpoint['bucket']}/checkpoints/stage_2_checkpoint.json", checkpoint)

    logger.info(f"Video processing complete for bucket {checkpoint['bucket']}")
    logger.info(f"  Total: {checkpoint['total_videos']}")
    logger.info(f"  Completed: {checkpoint['completed']}")
    logger.info(f"  Failed: {checkpoint['failed']}")

    if checkpoint['failed'] > 0:
        logger.warning(f"  {checkpoint['failed']} videos failed (see failed_video_ids in checkpoint)")
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| All videos failed | Mark as completed with warning | Stage completed (albeit with failures) |
| Partial failures | Mark as completed, log failures | Batch processing should continue to next stage |

---

## 3. Dependencies & Integration

<!-- PURPOSE: Explicit contracts with other stages. TI generator uses this for imports and validation. -->

### 3.1 Input Dependencies

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **Foundation setup** | FoundationCHILD.md (FoundationTI.md implementation) | config.json + analysis_base directory | client_id, analysis_type, target, analysis_mode, selection_strategy, video_count | Fail-fast if config.json missing or analysis_base doesn't exist |
| Selected video list | Stage 1 output | List[dict] | `id`, `videoMeta.downloadAddr`, `duration`, `engagement` | Fail-fast if empty list |
| Config.json | FoundationCHILD.md Section 5.1 | JSON | `video_count`, `selection_strategy`, `date_filter`, `run_date` | Fail-fast with error message |
| Bucket directories | **Created by Step 2.3.0** (this stage) | Directory structure | All 8 buckets with 15 subdirectories each | Auto-created if missing, fail-fast on permission/disk errors |
| RumiAI pipeline | Existing rumiai_runner.py | Python module | VideoAnalyzer class | Fail-fast if rumiai_runner.py not available |

### 3.2 Output Contracts

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| temporal_windows_updated.json | JSON (1 per video) | 60+ features per temporal window | Stage 2.4 (Validation), Stage 3 (Aggregation) | Assert 'temporal_windows' key exists |
| Checkpoint | JSON | FoundationCHILD Section 5.3 schema | Auto-resume on restart | Assert all required fields present |
| Downloaded videos | MP4 files | Video format | Stage 2.4 (Validation - investigation packages) | Assert file size > 1KB |
| Processing logs | Text file | Log entries (see example below) | Debugging, auditing | None (optional) |

**Example Log Format**:
```
2025-01-28 14:32:10 - INFO - ✓ Created new checkpoint for bucket 18-33s (100 videos)
2025-01-28 14:32:11 - INFO - Processing video 1/100: 7428596413707144481
2025-01-28 14:32:15 - INFO - Downloading video 7428596413707144481 (attempt 1/3)
2025-01-28 14:32:42 - INFO - ✓ Video downloaded (27s, 48.3MB)
2025-01-28 14:33:55 - INFO - ✓ RumiAI processing complete (73s)
2025-01-28 14:33:56 - INFO - Successfully processed video 7428596413707144481 (1/100)
2025-01-28 14:33:56 - ERROR - Failed to process video 7428596413707145632: FEAT timeout after 300s
2025-01-28 14:33:56 - INFO - Checkpoint updated (1 completed, 1 failed, 98 remaining)
```

### 3.3 Cross-Stage Dependencies

**This feature depends on**:
- **Stage 0 (Foundation)**: Must set up directories and config.json
- **Stage 1 (Video Selection)**: Must complete successfully (video list finalized)

**This feature is required by**:
- **Stage 2.4 (Pipeline Validation)**: Expects temporal_windows_updated.json for each video
- **Stage 3 (Feature Aggregation)**: Expects N temporal_windows_updated.json files

**Failure Impact**:
- If this stage fails: Stages 2.4+ cannot run (no temporal windows data)
- Checkpoint: Resume from last completed video without re-running Stage 1

### 3.4 External Dependencies

**Python Libraries**:
```python
import os  # File system operations
import json  # JSON I/O
import time  # Sleep for retry backoff
import logging  # Logging
import requests  # Video download via HTTP
import subprocess  # RumiAI subprocess execution with timeout
import signal  # Signal handling for graceful pause
import platform  # Platform detection for Windows compatibility
from datetime import datetime  # Timestamps
```

**File System**:
- Read access: `/data/clients/{client_id}/` (config.json, selected videos)
- Write access:
  - `{bucket}/videos/` (downloaded MP4s)
  - `{bucket}/analysis/insights/` (temporal_windows JSONs)
  - `{bucket}/checkpoints/` (stage checkpoints)
  - `{bucket}/logs/` (processing logs)

**Environment Variables**:
- `DATA_ROOT`: Root directory for client data (default: `/data`)
- `LOG_LEVEL`: Logging verbosity (default: `INFO`)

**External Services**:
- **Apify download URLs** (HTTP GET for video download)
  - Expected response: `200 OK` with video/mp4 content
  - Handle `404 Not Found`: Video deleted from TikTok (skip video, mark as failed)
  - Request timeout: 60s per download
  - Response format: Streaming binary (MP4 file)
  - No authentication required (pre-signed URLs from Apify metadata)

---

## 4. Configuration & Parameters

<!-- PURPOSE: All tunable values. TI generator uses this for config parsing and defaults. -->

### 4.1 CLI Parameters (if applicable)

**Note**: CLI parameters are defined in FoundationCHILD.md Section 4. This stage reads from config.json created by Foundation.

| Parameter | Source | Type | Impact | Used In |
|-----------|--------|------|--------|---------|
| `video_count` | config.json | int | Determines total videos to process | Checkpoint initialization (Section 2.3.1) |
| `selection_strategy` | config.json | str | No direct impact (used by Stage 1) | Config validation on resume |
| `bucket` | Runtime | str | Determines output directories | All file paths |

### 4.2 Internal Configuration

```python
# Video download configuration
MAX_DOWNLOAD_ATTEMPTS = 3          # Max download attempts per video (3 total attempts)
DOWNLOAD_TIMEOUT = 60             # Seconds per download request
DOWNLOAD_CHUNK_SIZE = 8192        # Bytes per chunk (streaming download)
MIN_VIDEO_SIZE = 100_000          # Bytes (100KB minimum for valid video files)
                                  # Rationale: Catches error pages (< 10KB) and truncated
                                  #            MP4 headers (< 50KB) while allowing highly
                                  #            compressed short videos. RumiAI services
                                  #            fail-fast on any remaining corrupt files.

# Retry backoff configuration
RETRY_BASE_DELAY = 2              # Base delay for exponential backoff (seconds)
# Actual delay = RETRY_BASE_DELAY ** attempt (2s, 4s, 8s)

# RumiAI processing configuration
RUMIAI_SEQUENTIAL = True          # Always sequential for batch processing
RUMIAI_TIMEOUT = 300              # Max processing time per video (seconds)

# Checkpoint configuration
CHECKPOINT_WRITE_FREQUENCY = 1    # Save checkpoint after every N videos (1 = every video)

# File paths (relative to bucket directory - from FoundationCHILD Section 2.2)
VIDEOS_DIR = "videos/"
INSIGHTS_DIR = "analysis/insights/"
UNIFIED_DIR = "analysis/unified/"
SERVICE_DEBUG_DIR = "analysis/service_debug/"
CHECKPOINT_DIR = "checkpoints/"
LOGS_DIR = "logs/"

CHECKPOINT_FILENAME = "stage_2_checkpoint.json"
```

---

## 5. Data Schemas

<!-- PURPOSE: Exact data structures. TI generator uses this for validation and type hints. -->

### 5.1 Input Schema

**File**: config.json (from FoundationCHILD.md Section 5.1)

Referenced fields in this stage:
| Column | Type | Required | Description | Used In |
|--------|------|----------|-------------|---------|
| `client_id` | str | Yes | Client identifier | Directory path construction |
| `analysis_type` | str | Yes | Target type | Directory path construction |
| `target` | str | Yes | Target identifier | Directory path construction |
| `analysis_mode` | str | Yes | Sorting method | Config validation on resume |
| `selection_strategy` | str | Yes | Selection strategy | Config validation on resume |
| `video_count` | int | Yes | Videos per bucket | Checkpoint total_videos |
| `date_filter` | str | Yes | Date filter | Config validation on resume |
| `run_date` | str | Yes | Analysis start timestamp | Logging |

**File**: Selected video list (from Stage 1)

Based on Apify metadata schema (FoundationCHILD.md Section 5.2):
| Column | Type | Required | Description | Used In |
|--------|------|----------|-------------|---------|
| `id` | str | Yes | TikTok video ID | Filename, checkpoint tracking |
| `videoMeta.downloadAddr` | str | Yes | MP4 download URL | Video download (Section 2.3.2) |
| `duration` | int | Yes | Video duration (seconds) | Validation |
| `playCount` | int | Yes | View count | Logging |
| `createTime` | int | Yes | Unix timestamp | Logging |

### 5.2 Output Schema

**File**: `{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json`

Schema defined by existing RumiAI pipeline (SystemArchitecturev2.md lines 395-460):
| Key | Type | Description | Validation |
|-----|------|-------------|------------|
| `video_id` | str | TikTok video ID | Must match filename |
| `duration` | float | Video duration (seconds) | Must be 3.0-120.0 |
| `processing_timestamp` | float | Unix timestamp | Must be present |
| `version` | str | RumiAI version | Must be "2.0.0" or higher |
| `temporal_windows` | dict | Features per window | Must have 'hook', 'closing' keys |
| `temporal_windows.hook` | dict | Hook features (0-3s) | 60+ features |
| `temporal_windows.middle_segments` | list or null | Middle segment features | null if duration <= 9s, else list |
| `temporal_windows.closing` | dict | Closing features (last 3s) | 60+ features |
| `metadata` | dict | Video-level metadata | Must have 'gender_detection', 'hashtag_analysis' |

**File**: `{bucket}/checkpoints/stage_2_checkpoint.json`

Schema from FoundationCHILD.md Section 5.3 (extended with pause support):
| Key | Type | Description | Validation |
|-----|------|-------------|------------|
| `stage` | str | Always "video_processing" | Must match |
| `bucket` | str | Bucket name (e.g., "18-33s") | Must match bucket being processed |
| `total_videos` | int | Total videos to process | Must equal len(video_list) |
| `completed` | int | Successfully processed | Must be >= 0 |
| `failed` | int | Failed with errors | Must be >= 0 |
| `remaining` | int | Videos not yet attempted (neither completed nor failed) | Must equal total_videos - completed - failed |
| `last_checkpoint` | str | ISO timestamp | Must be valid ISO 8601 |
| `completed_video_ids` | list | List of completed video IDs | Must be list of str |
| `failed_video_ids` | list | List of failed videos with error details | Must be list of dict (schema below) |
| `config` | dict | Config from config.json | For resume validation |
| `status` | str | "in_progress", "paused", or "completed" | Set by Step 2.3.4 or 2.3.5 |
| `pause_reason` | str | Optional, "user_requested" if paused | Only present if status="paused" |
| `pause_timestamp` | str | Optional, ISO timestamp when paused | Only present if status="paused" |

**`failed_video_ids` Entry Schema**:
| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `video_id` | str | TikTok video ID | "7428596413707144481" |
| `error` | str | Error message | "RumiAI timeout after 300s" |
| `error_type` | str | Exception class name | "TimeoutError", "ProcessingError", "DownloadError" |
| `timestamp` | str | ISO timestamp when error occurred | "2025-01-28T14:32:10Z" |

---

## 6. Error Handling & Validation

<!-- PURPOSE: All error scenarios. TI generator uses this for try/catch blocks and assertions. -->

### 6.1 Input Validation

```python
def validate_inputs(video_list, config, bucket):
    """
    Validate inputs before starting video processing.

    Args:
        video_list: list, videos from Stage 1
        config: dict, loaded from config.json
        bucket: str, bucket name

    Raises:
        ValueError: if validation fails with specific error message
    """
    # 1. Check video list not empty
    if not video_list:
        raise ValueError("Video list is empty. Did Stage 1 complete successfully?")

    # 2. Validate config.json has required fields
    required_config_fields = ['client_id', 'analysis_type', 'target', 'video_count', 'selection_strategy', 'analysis_mode', 'date_filter', 'run_date']
    missing = [f for f in required_config_fields if f not in config]
    if missing:
        raise ValueError(f"Config.json missing required fields: {missing}")

    # 3. Check analysis_base directory exists (parent of buckets/)
    data_root = os.getenv('DATA_ROOT', '/data')
    analysis_base = f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"
    if not os.path.exists(analysis_base):
        raise ValueError(f"Analysis base directory does not exist: {analysis_base}. Did Foundation setup run?")

    # 4. Validate each video has required metadata
    for video in video_list:
        if 'id' not in video:
            raise ValueError(f"Video missing 'id' field: {video}")
        if 'videoMeta' not in video or 'downloadAddr' not in video['videoMeta']:
            raise ValueError(f"Video {video['id']} missing download URL")

    # 5. Check write permissions to analysis_base
    test_file = f"{analysis_base}/test_write.tmp"
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise ValueError(f"No write permission to {analysis_base}: {e}")
```

### 6.2 Error Cases

| Error | Detection | Handling | User Message | Exit Code |
|-------|-----------|----------|--------------|-----------|
| Empty video list | `len(video_list) == 0` | Fail-fast | `"No videos to process. Did Stage 1 complete successfully?"` | 1 |
| Missing config.json | `not os.path.exists(config_path)` | Fail-fast | `"Config.json not found. Did Foundation setup run?"` | 2 |
| Analysis base directory missing | `not os.path.exists(analysis_base)` | Fail-fast | `"Analysis base directory does not exist. Did Foundation setup run?"` | 3 |
| Config mismatch on resume | Config validation | Fail-fast | `"Config mismatch: checkpoint video_count={X}, provided video_count={Y}"` | 4 |
| Cached file too small | `file_size < MIN_VIDEO_SIZE` on resume | Remove file, re-download (self-healing) | `"Existing file too small ({size} bytes), removing and re-downloading"` | 0 (auto-retry) |
| Downloaded file too small | `file_size < MIN_VIDEO_SIZE` (100KB) | Retry (max 3), then skip-on-fail | `"Downloaded file too small: {size} bytes (minimum: 100KB). Skipping after 3 attempts."` | 0 (skip-on-fail) |
| Download failure (max retries) | Retry loop exhausted | Log as failed, continue batch | `"Failed to download video {id} after 3 attempts. Skipping."` | 0 (skip-on-fail) |
| RumiAI subprocess failure | `subprocess.CalledProcessError` (exit != 0) | Log as failed, continue batch | `"RumiAI failed for {id} (exit code {code}): {stderr[:200]}"` | 0 (skip-on-fail) |
| RumiAI no insights file | `not os.path.exists(insights_path)` after exit 0 | Log as failed, continue batch | `"RumiAI exit 0 but no insights file at {path}"` | 0 (skip-on-fail) |
| RumiAI processing timeout | `subprocess.TimeoutExpired` | Log as failed, continue batch | `"RumiAI timeout after 300s for video {id}. Skipping."` | 0 (skip-on-fail) |
| Invalid temporal_windows schema | Schema validation | Log as failed, continue batch | `"Video {id} generated invalid output. Skipping."` | 0 (skip-on-fail) |
| Disk full | OSError | Fail-fast | `"Disk full. Free space and resume."` | 5 |
| Out of memory | MemoryError | Log as failed, continue batch | `"Video {id} caused OOM. Skipping (consider increasing memory)."` | 0 (skip-on-fail) |
| Graceful pause (user requested) | Ctrl+C signal | Pause gracefully, save checkpoint | `"⏸️  Paused after {N} videos. Resume anytime."` | 0 |

### 6.3 Output Validation

```python
def validate_stage_outputs(bucket, checkpoint):
    """
    Validate stage outputs after all videos processed.

    Args:
        bucket: str, bucket name
        checkpoint: dict, final checkpoint data

    Raises:
        AssertionError: if output validation fails
    """
    insights_dir = f"{bucket}/analysis/insights/"

    # 1. Check insights directory exists
    assert os.path.exists(insights_dir), f"Insights directory not created: {insights_dir}"

    # 2. Check number of insights files matches completed count
    insights_files = [f for f in os.listdir(insights_dir) if f.endswith('_temporal_windows_updated.json')]
    assert len(insights_files) == checkpoint['completed'], \
        f"Mismatch: {len(insights_files)} insights files, {checkpoint['completed']} completed in checkpoint"

    # 3. Check each completed video has insights file
    for video_id in checkpoint['completed_video_ids']:
        insights_path = f"{insights_dir}/{video_id}_temporal_windows_updated.json"
        assert os.path.exists(insights_path), f"Missing insights for completed video {video_id}"

    # 4. Check checkpoint saved
    checkpoint_path = f"{bucket}/checkpoints/stage_2_checkpoint.json"
    assert os.path.exists(checkpoint_path), "Checkpoint file not saved"

    # 5. Log summary
    logger.info(f"Stage 2 validation passed for bucket {bucket}")
    logger.info(f"  Insights files: {len(insights_files)}")
    logger.info(f"  Completed videos: {checkpoint['completed']}")
    logger.info(f"  Failed videos: {checkpoint['failed']}")
```

---

## 7. Performance & Scalability

<!-- PURPOSE: Performance targets and bottlenecks. TI generator uses this for optimization. -->

### 7.1 Performance Targets

- **Throughput**: Process 300 videos (bucket 18-33s) in < 8 hours
- **Per-video processing**: < 90 seconds per 60-second video (sequential mode)
- **Memory**: Peak usage < 3GB (RumiAI services)
- **Disk I/O**: < 5s per checkpoint write (small JSON)
- **Download speed**: < 30s per 50MB video

### 7.2 Measured Performance

**From production measurements (Jan 2025)**:

| Metric | 60s Video | 120s Video | Notes |
|--------|-----------|------------|-------|
| Total RumiAI processing | 83.96s | 177.52s | Sequential mode (9 services) |
| FEAT emotion detection | 73.96s | ~120s | Bottleneck (43% of total) |
| Whisper transcription | 26.14s | ~45s | Second bottleneck (15%) |
| Other services | ~25s | ~50s | YOLO, MediaPipe, OCR, etc. |
| Video download | 10-30s | 20-50s | Depends on network |
| Checkpoint write | < 1s | < 1s | Small JSON write |

**Total time per video**: ~110-140s (60s video), ~200-230s (120s video)

**Batch estimates**:
- 100 videos (60s avg): ~3.5 hours
- 300 videos (60s avg): ~10-11 hours
- Actual: Varies by video duration distribution

### 7.3 Bottlenecks & Mitigations

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| FEAT processing | 43% of time | CPU-intensive facial analysis | Enable GPU acceleration (future) | High |
| Network downloads | 10-30s per video | Large video files (50MB avg) | Parallel downloads (future), CDN caching | Medium |
| Sequential processing | No parallelism | One video at a time | Acceptable for reliability (checkpoint after each) | Low |
| Checkpoint writes | Minimal (<1s) | JSON I/O | Batch writes every N videos (optimization for future) | Low |

### 7.4 Scalability Limits

- **Max videos per batch**: 500 (checkpoint file size: ~500KB)
- **Max video duration**: 120s (RumiAI supports up to 120s)
- **Min video duration**: 3s (RumiAI requires >= 3s)
- **Disk space**: ~50MB per video × 300 = 15GB per batch
- **Memory**: 3GB peak (RumiAI services, no leak detected)

---

## 8. Testing Strategy

<!-- PURPOSE: Test plan. TI generator uses this to create test suite. -->

### 8.1 Unit Tests

- [ ] **Test bucket directory initialization**
  - Create all 8 buckets with 15 subdirectories each
  - Idempotent creation (safe to re-run, uses exist_ok=True)
  - Validate bucket structure (check required subdirectories exist)
  - Handle missing parent directories (auto-create via os.makedirs)
  - Handle permission denied (fail-fast with OSError)
  - Handle disk full (fail-fast with OSError)

- [ ] **Test checkpoint initialization**
  - New checkpoint creation (no existing checkpoint)
  - Load existing checkpoint (auto-resume)
  - Config mismatch detection (raises ValueError)
  - Corrupted checkpoint handling (suggest backup restore)

- [ ] **Test video download**
  - Successful download (200 OK)
  - Retry on timeout (3 attempts)
  - Retry on network error (3 attempts)
  - Fail after max retries (raises DownloadError)
  - Skip already downloaded videos (resume scenario)

- [ ] **Test RumiAI processing**
  - Successful processing (generates temporal_windows)
  - Processing timeout (mark as failed)
  - Service error (mark as failed, continue batch)
  - Invalid output schema (mark as failed)

- [ ] **Test checkpoint updates**
  - Mark video as completed (increment completed, update IDs)
  - Mark video as failed (increment failed, append to failed_video_ids)
  - Save after each video (checkpoint file updated)

- [ ] **Test edge cases**
  - Empty video list (raises ValueError)
  - All videos already completed (resume scenario, skip all)
  - Partial completion (resume from middle)
  - All videos failed (mark stage as completed with failures)

- [ ] **Test graceful pause signal handling**
  - First Ctrl+C sets pause flag, continues current video
  - Second Ctrl+C exits immediately (force quit)
  - Pause flag checked before next video starts
  - Checkpoint status set to "paused" correctly

### 8.2 Integration Tests

- [ ] **End-to-end: Stage 1 → Stage 2 → Stage 2.4**
  - Use real video list from Stage 1 (5 videos, bucket 18-33s)
  - Run full video processing
  - Validate temporal_windows files exist (5 files)
  - Verify Stage 2.4 validation can load outputs

- [ ] **Checkpoint resume**
  - Process 10 videos, interrupt after 5
  - Restart without --resume flag (auto-resume)
  - Verify only remaining 5 videos processed
  - Verify no duplicates in completed_video_ids

- [ ] **Failure handling**
  - Simulate download failure (network timeout)
  - Verify video marked as failed
  - Verify batch continues to next video

- [ ] **Config validation**
  - Process with video_count=100
  - Restart with video_count=50
  - Verify config mismatch error raised

- [ ] **Graceful pause integration test**
  - Process 10 videos, send SIGINT after video 5 completes
  - Verify processing pauses gracefully (not mid-video)
  - Verify checkpoint status="paused" saved
  - Resume and verify only remaining 5 videos processed

- [ ] **Error handling integration test**
  - Inject disk full error during bucket creation (should fail-fast with OSError)
  - Inject permission denied during bucket creation (should fail-fast with OSError)
  - Inject config mismatch on resume (different video_count, should exit code 4)
  - Inject missing config.json (should exit code 2)
  - Inject missing analysis_base directory (should exit code 3)
  - Inject download timeout (should retry 3x, then skip video)
  - Inject RumiAI timeout (should mark video as failed, continue batch)
  - Verify checkpoint tracks all failures correctly

### 8.3 Test Data

**File**: `tests/fixtures/sample_video_list.json`

```json
[
  {
    "id": "7428596413707144481",
    "createTime": 1704067200,
    "duration": 25,
    "playCount": 50000,
    "videoMeta": {
      "downloadAddr": "https://example.com/video1.mp4"
    }
  },
  {
    "id": "7428596413707144482",
    "createTime": 1704153600,
    "duration": 18,
    "playCount": 35000,
    "videoMeta": {
      "downloadAddr": "https://example.com/video2.mp4"
    }
  }
]
```

**File**: `tests/fixtures/sample_config.json`

```json
{
  "client_id": "test_client",
  "analysis_type": "hashtag",
  "target": "#test",
  "analysis_mode": "top",
  "selection_strategy": "contrastive",
  "video_count": 100,
  "date_filter": "last_90_days",
  "run_date": "2025-01-28T10:00:00Z"
}
```

### 8.4 Test Execution

```bash
# Run unit tests
pytest tests/test_video_processing.py -v

# Run integration tests
pytest tests/test_stage2_integration.py -v

# Run checkpoint resume test
pytest tests/test_checkpoint_resume.py -v

# Run with coverage
pytest --cov=video_processing --cov-report=html
```

---

## 9. Future Enhancements

<!-- PURPOSE: Planned improvements. TI generator ignores this section (not for current implementation). -->

### 9.1 Planned Improvements

- **Phase 2: Parallel video downloads**
  - Current: Sequential download (one at a time)
  - Future: Download next video while processing current video
  - Impact: 10-30s saved per video

- **Phase 3: Batch checkpoint writes**
  - Current: Write checkpoint after every video
  - Future: Write every 5 videos (configurable)
  - Impact: Reduce disk I/O by 80%
  - Tradeoff: Lose up to 5 videos of progress on interruption

- **Phase 4: GPU-accelerated FEAT**
  - Current: CPU-only FEAT processing (73s per 60s video)
  - Future: GPU acceleration (estimated 10-15s)
  - Impact: 60s saved per video, 5x speedup

- **Phase 5: Retry failed videos**
  - Current: Skip-on-fail (no retries)
  - Future: `--retry-failed` flag to reprocess failed videos
  - Impact: Recover from transient failures

### 9.2 Known Limitations

- **Sequential processing only**: No parallel processing (by design for reliability)
- **No video re-download on corruption**: If downloaded video is corrupt, manual cleanup required
- **No automatic retry on OOM**: Out-of-memory errors require manual intervention
- **Fixed checkpoint location**: Cannot customize checkpoint directory

---

## 10. References & Related Docs

<!-- PURPOSE: Links to other documentation. TI generator uses this for additional context if needed. -->

### 10.1 Parent Document

- **MLPlanningv2.md - Section "Stage 2: Video Processing (RumiAI Pipeline)"**
  - High-level stage overview
  - Stage position in pipeline
  - Input/output contracts

### 10.2 Foundation Document

- **FoundationCHILD.md** (shared across all stages)
  - Section 1: System Goals & Success Criteria
  - Section 2: Client Architecture & Storage (directory paths used in this stage)
  - Section 4: CLI Command Structure (complete command syntax)
  - Section 5.1: Configuration Schemas (config.json)
  - Section 5.2: Apify Video Metadata Schema
  - Section 5.3: Checkpoint Schema

**Key Sections Referenced in This Stage**:
- Section 2.1: Directory Structure (provides base paths for video storage, insights, checkpoints)
- Section 2.2: Path Templates (used for constructing file paths)
- Section 5.1: config.json schema (read at stage start)
- Section 5.2: Apify metadata (used for video download)
- Section 5.3: Checkpoint schema (used for auto-resume)

### 10.3 Related Child Docs

- **VideoDiscoveryCHILD.md** (Stage 1)
  - Produces selected video list (input to this stage)
  - Defines video metadata format

- **PipelineValidationCHILD.md** (Stage 2.4)
  - Consumes temporal_windows_updated.json (output from this stage)
  - Validates feature outliers

- **FeatureAggregationCHILD.md** (Stage 3)
  - Consumes temporal_windows_updated.json (output from this stage)
  - Aggregates features to video-level

### 10.4 External References

- **RumiAI SystemArchitecturev2.md**: RumiAI processing pipeline architecture (lines 1-534)
- **RumiAI temporal_windows schema**: SystemArchitecturev2.md (lines 395-460)
- **Apify Documentation**:
  - clockworks/tiktok-hashtag-scraper
  - clockworks/tiktok-scraper

### 10.5 Code References

- **Existing RumiAI code**: `/home/jorge/rumiaifinal/rumiai_runner.py` (VideoAnalyzer class)
- **Existing services**: `/home/jorge/rumiaifinal/services/` (YOLO, Whisper, FEAT, etc.)
- **Utility functions**: `/home/jorge/rumiaifinal/utils/` (file_handler, logger, validators)

---

## Appendix A: Checkpoint Resume Scenarios

<!-- PURPOSE: Concrete examples of checkpoint resume behavior -->

### A.1 Fresh Start (No Checkpoint)

**Command**:
```bash
python rumiai_ml_batch.py --client "acme" --analysis-type hashtag --target "#nutrition"
```

**Output**:
```
✓ Created new checkpoint for bucket 18-33s (100 videos)
→ Processing video 1/100 (bucket_18-33s: 1/100)...
→ Processing video 2/100 (bucket_18-33s: 2/100)...
...
```

### A.2 Auto-Resume (Checkpoint Exists)

**Scenario**: Process interrupted at video 45/100

**Command** (same as original):
```bash
python rumiai_ml_batch.py --client "acme" --analysis-type hashtag --target "#nutrition"
```

**Output**:
```
✓ Checkpoint detected: 45/100 videos completed (45%)
  Last updated: 2 hours ago
  Failed videos: 2 (logged, will not retry)

→ Auto-resuming from bucket_18-33s (45/100 completed)
→ Skipping 45 already-processed videos
→ Processing remaining 55 videos...

→ Processing video 46/100 (bucket_18-33s: 46/100)...
```

### A.3 Config Mismatch (Error)

**Scenario**: User changes video_count when resuming

**Original Command**:
```bash
python rumiai_ml_batch.py --client "acme" --analysis-type hashtag --target "#nutrition" --video-count 100
```

**Resume Command** (different video_count):
```bash
python rumiai_ml_batch.py --client "acme" --analysis-type hashtag --target "#nutrition" --video-count 150
```

**Output**:
```
✗ Checkpoint config mismatch:
  - Checkpoint video_count: 100
  - Provided video_count: 150

Options:
  1. Match original config (remove --video-count 150)
  2. Use --force to discard checkpoint and restart

Error: Config mismatch prevents resume. Exit code 4.
```

### A.4 Graceful Pause (User Requested)

**Scenario**: User pauses processing gracefully via Ctrl+C

**Command**:
```bash
python rumiai_ml_batch.py --client "acme" --analysis-type hashtag --target "#nutrition"
```

**Output** (during processing):
```
→ Processing video 47/100 (bucket_18-33s: 47/100)...
  ✓ Downloaded: 7428596413707144481 (25.3 MB)
  ✓ RumiAI processing...

# User presses Ctrl+C
^C
⏸️  Pause requested. Will pause after current video completes...
   Press Ctrl+C again to force quit (current video will be lost)

# Video 47 continues processing...
  ✓ YOLO object detection (5.2s)
  ✓ Whisper transcription (26.1s)
  ✓ FEAT emotion detection (73.8s)
  ✓ All services complete
  ✓ Checkpoint updated (47/100)

⏸️  Paused gracefully after 47/100 videos
   Resume anytime by re-running the same command
   Checkpoint: bucket_18-33s/checkpoints/stage_2_checkpoint.json
```

**Resume Command** (same as original):
```bash
python rumiai_ml_batch.py --client "acme" --analysis-type hashtag --target "#nutrition"
```

**Resume Output**:
```
✓ Checkpoint detected: 47/100 videos completed (47%)
  Status: paused (user requested)
  Last updated: 15 minutes ago
  Failed videos: 2 (logged, will not retry)

→ Auto-resuming from bucket_18-33s (47/100 completed)
→ Skipping 47 already-processed videos
→ Processing remaining 53 videos...

→ Processing video 48/100 (bucket_18-33s: 48/100)...
```

**Checkpoint JSON** (paused state):
```json
{
  "stage": "video_processing",
  "bucket": "18-33s",
  "total_videos": 100,
  "completed": 47,
  "failed": 2,
  "remaining": 53,
  "last_checkpoint": "2025-01-28T16:45:30Z",
  "status": "paused",
  "pause_reason": "user_requested",
  "pause_timestamp": "2025-01-28T16:45:30Z",
  "completed_video_ids": ["123", "456", ..., "789"],
  "failed_video_ids": [
    {
      "video_id": "321",
      "error": "RumiAI processing exceeded 300s timeout",
      "error_type": "TimeoutError",
      "timestamp": "2025-01-28T15:20:15Z"
    }
  ]
}
```

---

## Document Metadata

**Creation Date**: 2025-01-28
**Last Modified**: 2025-01-28
**Authors**: RumiAI Team
**Reviewers**: [Pending]
**Approved By**: [Pending]
**Next Review Date**: [Pending]

---

