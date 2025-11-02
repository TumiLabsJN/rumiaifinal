# STAGE_2_IMPL.md - Video Processing (ML Services)

**Version**: 1.0.0
**Last Updated**: 2025-11-02
**Purpose**: Implementation guide for Stage 2: Video Processing
**Target Audience**: LLM agents debugging, modifying, or extending Stage 2

**Related**: [PRODUCTION_FLOW.md Stage 2 Contract](PRODUCTION_FLOW.md#stage-2-video-processing-ml-services)

---

## Quick Reference

### Entry Points

**Main Entry**: `stage_2_video_processing_main()` at [`ml_pipeline/stage2_processing/main.py:28-136`](ml_pipeline/stage2_processing/main.py#L28-L136)

**Orchestrator Call**: [`rumiai_ml_batch.py:751-819`](rumiai_ml_batch.py#L751-L819)
```python
# Lines 791-796
summary = stage_2_video_processing_main(
    config=config.model_dump(),
    video_list=video_list,
    bucket_name=bucket_name,
    enable_pause_support=True  # Allow Ctrl+C graceful pause
)
```

### Key Characteristics

- **Duration**: ~60-80 seconds per 60-second video
- **Bottleneck**: FEAT emotion detection (43% of processing time)
- **Processing Mode**: Sequential (one video at a time per bucket)
- **Error Strategy**: Skip-on-fail (individual video failures don't stop bucket)
- **Checkpoint**: Automatic save after each video, supports resume
- **Pause Support**: Ctrl+C graceful pause between videos
- **Output Location**: **HARDCODED** `/home/jorge/rumiaifinal/insights/`

### Module Structure

```
ml_pipeline/stage2_processing/          (1,434 total lines)
├── main.py                   (184)  # Main orchestrator
├── video_processor.py        (267)  # Core processing + subprocess call
├── checkpoint.py             (208)  # Checkpoint management
├── pause_handler.py          (161)  # Ctrl+C graceful handling
├── bucket_init.py            (167)  # Directory structure creation
├── video_download.py         (120)  # Video download with retry
├── utils.py                  (134)  # Helper functions (path, JSON I/O)
├── validation.py              (88)  # Output schema validation
└── exceptions.py              (82)  # Custom exceptions

External Dependency:
└── scripts/rumiai_runner.py  (508)  # ML pipeline subprocess
    └── Output: {video_id}_temporal_windows_updated.json
```

---

## Table of Contents

1. [Overview](#overview)
2. [Input Contract](#input-contract)
3. [Output Contract](#output-contract)
4. [Core Functions](#core-functions)
5. [Checkpoint Strategy](#checkpoint-strategy)
6. [Error Handling](#error-handling)
7. [Subprocess Contract](#subprocess-contract-rumiai_runnerpy)
8. [Debugging Guide](#debugging-guide)
9. [Modification Guide](#modification-guide)
10. [Related Documentation](#related-documentation)

---

## Overview

**Stage 2** processes videos through the RumiAI ML pipeline to extract 350+ features per temporal window. It orchestrates video download, ML processing, checkpoint management, and error handling with graceful pause support.

### Processing Flow (Per Bucket)

**Step 0**: Ensure bucket directory exists (defensive programming)
**Step 1**: Initialize checkpoint (load existing or create new)
**Step 2**: Pre-download videos (optional, if download URLs available)
**Step 3-4**: Process videos sequentially with pause support
**Step 5**: Finalize checkpoint (mark as completed)
**Step 6**: Validate outputs (check file existence, schema)

### Sub-Steps Per Video

For each video in the bucket:
1. Check if already completed (checkpoint resume)
2. Download video if needed (or use TikTok URL directly)
3. **Call rumiai_runner.py subprocess** (ML pipeline)
4. Validate output exists and passes schema validation
5. Update checkpoint (completed or failed)
6. Check for pause request (Ctrl+C)

### Critical Design Decisions

**Hardcoded Output Path**: `/home/jorge/rumiaifinal/insights/`
- Stage 2 outputs to flat directory (no bucket awareness)
- Stage 2.5 organizes files into bucket directories later
- Rationale: rumiai_runner.py is a legacy script with hardcoded paths

**Hybrid Mode (URL vs Local File)**:
- Always prefer TikTok URL over local MP4 file
- Rationale: Avoids rumiai_runner.py URL validation issues
- Fallback: Use local file if URL unavailable

**Skip-on-Fail Policy**:
- Individual video failures don't stop bucket processing
- Failed videos tracked in checkpoint
- Rationale: Maximize completed videos per bucket

---

## Input Contract

### Prerequisites

**Stage 1 Outputs** (Required):
- `winner_analysis.json` - Top 3 winning buckets
- `buckets/bucket_{name}/selected_videos.json` - Video metadata per bucket

### Required Inputs

**1. Config** (from Stage 0):
```python
{
  "client_id": "acme_corp",
  "analysis_type": "hashtag",
  "target": "nutrition",
  "analysis_mode": "top",
  "selection_strategy": "contrastive",
  "video_count": 100,
  "date_filter": "last_90_days"
}
```

**2. video_list** (from Stage 1 selected_videos.json):
```python
[
  {
    "id": "7123456789012345678",
    "webVideoUrl": "https://www.tiktok.com/@user/video/7123456789012345678",
    "duration": 25,
    "playCount": 1500000,
    "videoMeta": {
      "downloadAddr": "https://...",  # Optional (API changed Oct 2025)
      "duration": 25
    },
    "mediaUrls": ["https://..."],  # Alternative download URL
    ...
  }
]
```

**3. bucket_name** (from Stage 1 winner_analysis.json):
- Format: `"18-33s"` (duration bucket name only, NOT full path)
- Examples: `"18-33s"`, `"33-60s"`, `"13-18s"`

**4. enable_pause_support** (optional):
- Default: `true`
- Set to `false` to disable Ctrl+C handling

### Input Validation

**Config Validation** (checkpoint.py:122-138):
```python
critical_fields = ['video_count', 'selection_strategy', 'date_filter']

# On resume, validate config matches checkpoint
if checkpoint_config.get(field) != current_config.get(field):
    raise ValueError(
        f"Config mismatch detected. Cannot resume with different parameters:\n"
        f"{field}: checkpoint={checkpoint_config.get(field)}, "
        f"current={current_config.get(field)}"
    )
```

**Bucket Name Validation** (bucket_init.py:66-84):
```python
VALID_BUCKETS = ["0-3s", "3-9s", "9-13s", "13-18s",
                 "18-33s", "33-60s", "60-90s", "90-120s"]

if bucket_name not in VALID_BUCKETS:
    raise ValueError(
        f"Invalid bucket name: '{bucket_name}'\n"
        f"Expected one of: {VALID_BUCKETS}"
    )
```

---

## Output Contract

### Output Files

**1. temporal_windows_updated.json** (per video)

**Path**: `/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json`

**Schema** (from validation.py:17-89):
```json
{
  "video_id": "7123456789012345678",
  "duration": 25,
  "temporal_windows": {
    "hook": {
      "visual_scene_count": 3,
      "audio_energy_mean": 0.45,
      "emotion_joy_max": 0.78,
      "text_overlay_count": 2,
      ...  // 60+ features per window
    },
    "middle_segments": [
      {
        "visual_scene_count": 2,
        "audio_energy_mean": 0.52,
        ...  // 60+ features
      },
      {
        "visual_scene_count": 1,
        ...  // 60+ features
      }
    ],
    "closing": {
      "visual_scene_count": 2,
      "audio_energy_mean": 0.38,
      ...  // 60+ features
    }
  },
  "metadata": {
    "video_id": "7123456789012345678",
    "duration": 25,
    "processing_timestamp": "2025-11-02T10:30:00Z"
  },
  "processing_timestamp": "2025-11-02T10:30:00Z"
}
```

**Temporal Window Structure**:
- **hook**: First 3 seconds (0-3s) - 60+ features
- **middle_segments**: Variable segments based on duration
  - `null` for videos < 9s
  - List of dicts for videos ≥ 9s (3-5 segments)
- **closing**: Last 3 seconds - 60+ features

**Validation Rules** (validation.py:36-77):
1. Required top-level keys: `temporal_windows`, `metadata`, `processing_timestamp`
2. `temporal_windows` must be dict
3. Required sections: `hook`, `closing` (always present)
4. Each section must be dict with ≥50 features (warns if < 50)
5. `middle_segments`: null for short videos (<9s), list otherwise

**2. stage_2_checkpoint.json** (per bucket)

**Path**: `{bucket_path}/checkpoints/stage_2_checkpoint.json`

**Schema** (checkpoint.py:159-173):
```json
{
  "stage": "video_processing",
  "bucket": "18-33s",
  "total_videos": 100,
  "completed": 45,
  "failed": 2,
  "remaining": 53,
  "last_checkpoint": "2025-11-02T10:30:00.123456",
  "completed_video_ids": [
    "7123456789012345678",
    "7123456789012345679"
  ],
  "failed_video_ids": [
    {
      "video_id": "7123456789012345680",
      "error": "RumiAI pipeline failed (exit code 1)",
      "error_type": "ProcessingError",
      "timestamp": "2025-11-02T10:29:00.123456"
    }
  ],
  "config": {
    "client_id": "acme_corp",
    "analysis_type": "hashtag",
    "target": "nutrition",
    "video_count": 100,
    "selection_strategy": "contrastive",
    "date_filter": "last_90_days"
  },
  "status": "in_progress",
  "pause_reason": null,
  "pause_timestamp": null
}
```

**Status Values**:
- `"in_progress"` - Processing videos
- `"paused"` - User pressed Ctrl+C
- `"completed"` - All videos processed

**3. stage_2_checkpoint.backup.json** (automatic backup)

**Path**: `{bucket_path}/checkpoints/stage_2_checkpoint.backup.json`

**Purpose**: Automatic backup created before each checkpoint write
- Enables recovery from checkpoint corruption
- Same schema as main checkpoint

---

## Core Functions

### 1. stage_2_video_processing_main() - Orchestrator

**Location**: `ml_pipeline/stage2_processing/main.py:28-136`

**Purpose**: Main orchestration function for Stage 2

**Function Signature**:
```python
def stage_2_video_processing_main(
    config: dict,
    video_list: List[dict],
    bucket_name: str,
    enable_pause_support: bool = True
) -> Dict[str, any]:
```

**Returns**:
```python
{
  "total": 100,
  "completed": 98,
  "failed": 2,
  "status": "completed"
}
```

**Implementation**:
```python
# Step 0: Ensure bucket directory exists (defensive)
bucket_path = ensure_bucket_exists(config, bucket_name)

# Step 1: Load or create checkpoint
checkpoint, remaining_videos = initialize_checkpoint(bucket_name, video_list, config)

if not remaining_videos:
    # All videos already completed (resume scenario)
    finalize_checkpoint(checkpoint, config)
    return {
        "total": checkpoint['total_videos'],
        "completed": checkpoint['completed'],
        "failed": checkpoint['failed'],
        "status": "completed"
    }

# Step 2: Pre-download videos (optional)
bucket_path = get_bucket_path(config, bucket_name)
videos_dir = f"{bucket_path}videos/"

downloadable_count = 0
for video in remaining_videos:
    video_id = video['id']

    # Check if video has download URL (multiple API formats)
    has_download_url = False
    video_meta = video.get('videoMeta', {})

    if video_meta and 'downloadAddr' in video_meta:
        has_download_url = True
    elif 'mediaUrls' in video and video.get('mediaUrls'):
        has_download_url = True

    if has_download_url:
        try:
            download_video(video, videos_dir)
            downloadable_count += 1
        except Exception as e:
            logger.warning(f"Pre-download failed for {video_id}: {e}")
            # Will use webVideoUrl during processing instead

logger.info(f"Pre-downloaded {downloadable_count}/{len(remaining_videos)} videos")

# Step 3-4: Process videos
if enable_pause_support:
    process_videos_with_pause_support(remaining_videos, bucket_name, checkpoint, config)
else:
    process_videos_sequential(remaining_videos, bucket_name, checkpoint, config)

# Step 5: Finalize checkpoint
finalize_checkpoint(checkpoint, config)

# Step 6: Validate outputs
validate_stage_output(bucket_name, checkpoint, config)

return {
    "total": checkpoint['total_videos'],
    "completed": checkpoint['completed'],
    "failed": checkpoint['failed'],
    "status": checkpoint['status']
}
```

---

### 2. run_rumiai_pipeline() - Subprocess Call

**Location**: `ml_pipeline/stage2_processing/video_processor.py:30-127`

**Purpose**: Run RumiAI ML pipeline via subprocess with timeout

**Function Signature**:
```python
def run_rumiai_pipeline(
    video_path: str,
    video_id: str,
    output_dir: str,
    timeout: int = 300
) -> Dict[str, Any]:
```

**CRITICAL: Subprocess Contract**
```python
cmd = [
    sys.executable,  # python3
    'scripts/rumiai_runner.py',
    video_path  # TikTok URL or local MP4 path
]

result = subprocess.run(
    cmd,
    timeout=300,  # 5 minutes max
    capture_output=True,
    text=True,
    check=True  # Raises CalledProcessError if exit != 0
)
```

**Success Validation Strategy**:
1. Subprocess exits with code 0 (`check=True` raises exception if != 0)
2. Insights file exists at expected path
3. Insights file passes schema validation (handled by caller)

**Important Notes**:
- rumiai_runner.py does NOT output JSON to stdout (file-based contract)
- Stdout/stderr are logged for debugging only
- Output location is **HARDCODED**: `/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json`

**Implementation**:
```python
# FIXED: rumiai_runner.py outputs to hardcoded flat directory
insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"

cmd = [
    sys.executable,
    'scripts/rumiai_runner.py',
    video_path  # Pass URL directly (hybrid mode prefers URLs)
]

try:
    result = subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=True
    )

    # Log stdout/stderr for debugging (no JSON parsing)
    if result.stdout:
        logger.debug(f"RumiAI stdout: {result.stdout[:500]}")
    if result.stderr:
        logger.warning(f"RumiAI stderr: {result.stderr[:500]}")

    # Validate insights file exists (ground truth for success)
    if not os.path.exists(insights_path):
        raise ProcessingError(
            video_id=video_id,
            stage="output_validation",
            message=f"RumiAI exited successfully (code 0) but no insights file at {insights_path}"
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
        video_id=video_id,
        stage="rumiai_pipeline",
        message=f"RumiAI pipeline failed (exit code {e.returncode}). "
               f"Stderr: {e.stderr[:200] if e.stderr else 'none'}"
    )
```

**Hybrid Mode (URL vs Local File)**:
```python
# Always prefer URL over local file (avoid URL validation issues)
if 'webVideoUrl' in video and video['webVideoUrl']:
    video_path = video['webVideoUrl']
    logger.info(f"Processing video {video_id} (TikTok URL)")
elif os.path.exists(local_video_path):
    video_path = local_video_path
    logger.warning(f"Processing video {video_id} (local file - no URL available)")
else:
    raise ValueError(f"Video {video_id} missing: no local file and no webVideoUrl")
```

---

### 3. initialize_checkpoint() - Checkpoint Initialization

**Location**: `ml_pipeline/stage2_processing/checkpoint.py:100-178`

**Purpose**: Load existing checkpoint or create new one with resume support

**Function Signature**:
```python
def initialize_checkpoint(
    bucket_name: str,
    video_list: list,
    config: dict
) -> Tuple[dict, list]:
```

**Returns**:
- `checkpoint`: dict - Checkpoint data (new or existing)
- `remaining_videos`: list - Videos to process (excludes completed)

**Implementation**:
```python
def validate_config_match(checkpoint_config: dict, current_config: dict):
    """Validate checkpoint config matches current run."""
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

# Construct bucket path
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
```

**Checkpoint Recovery** (checkpoint.py:54-97):
```python
def load_checkpoint_with_recovery(checkpoint_path: str) -> dict:
    """Load checkpoint with automatic recovery from backup on corruption."""
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
```

---

### 4. process_videos_with_pause_support() - Graceful Pause

**Location**: `ml_pipeline/stage2_processing/pause_handler.py:51-161`

**Purpose**: Process videos with Ctrl+C graceful pause between videos

**Signal Handler** (pause_handler.py:25-48):
```python
pause_requested = False  # Global flag (signal-safe for single-threaded)

def request_pause(signum: int, frame):
    """Signal handler for graceful pause."""
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
```

**Implementation**:
```python
def process_videos_with_pause_support(
    remaining_videos: list,
    bucket_name: str,
    checkpoint: dict,
    config: dict
):
    global pause_requested

    # Register signal handler (cross-platform SIGINT support)
    signal.signal(signal.SIGINT, request_pause)

    bucket_path = get_bucket_path(config, bucket_name)
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    # Process each video
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

        # Get video path (hybrid mode: prefer URL)
        video_id = video['id']
        local_video_path = f"{bucket_path}videos/{video_id}.mp4"

        if os.path.exists(local_video_path):
            video_path = local_video_path
        elif 'webVideoUrl' in video:
            video_path = video['webVideoUrl']
        else:
            handle_video_processing_error(
                ValueError(f"Video {video_id} missing"),
                video_id, checkpoint, checkpoint_path
            )
            continue

        # Process video through RumiAI pipeline
        try:
            result = run_rumiai_pipeline(
                video_path=video_path,
                video_id=video_id,
                output_dir=f"{bucket_path}analysis/",
                timeout=300
            )

            # Validate output exists
            insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
            if not os.path.exists(insights_path):
                raise ProcessingError(
                    video_id=video_id,
                    stage="output_validation",
                    message=f"RumiAI did not generate insights file at {insights_path}"
                )

            # Validate schema
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
            handle_video_processing_error(e, video_id, checkpoint, checkpoint_path)
            continue

    logger.info(f"All videos processed successfully")
```

---

### 5. handle_video_processing_error() - Error Handler

**Location**: `ml_pipeline/stage2_processing/video_processor.py:217-267`

**Purpose**: Handle errors with skip-on-fail policy

**Implementation**:
```python
def handle_video_processing_error(
    error: Exception,
    video_id: str,
    checkpoint: dict,
    checkpoint_path: str
) -> str:
    """
    Handle errors during video processing with skip-on-fail policy.

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

    return "failed"
```

---

### 6. download_video() - Video Download with Retry

**Location**: `ml_pipeline/stage2_processing/video_download.py:21-120`

**Purpose**: Download video from Apify download URL with exponential backoff

**Implementation**:
```python
def download_video(
    video_metadata: Dict[str, Any],
    output_dir: str,
    max_attempts: int = 3
) -> str:
    """Download video MP4 with retry logic."""

    video_id = video_metadata['id']

    # Get download URL - try multiple API formats (API changed Oct 2025)
    download_url = None

    # Option 1: Old API format (videoMeta.downloadAddr)
    if 'videoMeta' in video_metadata and 'downloadAddr' in video_metadata.get('videoMeta', {}):
        download_url = video_metadata['videoMeta']['downloadAddr']

    # Option 2: mediaUrls array
    if not download_url and 'mediaUrls' in video_metadata and video_metadata.get('mediaUrls'):
        download_url = video_metadata['mediaUrls'][0]

    if not download_url:
        raise DownloadError(
            video_id=video_id,
            attempts=0,
            original_error=KeyError(f"No download URL found for video {video_id}")
        )

    output_path = f"{output_dir}/{video_id}.mp4"

    # Check if already downloaded (resume optimization)
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size >= MIN_VIDEO_SIZE:  # 100KB minimum
            logger.info(f"Video {video_id} already downloaded ({file_size} bytes), skipping")
            return output_path
        else:
            os.remove(output_path)  # Remove corrupt file

    # Retry loop with exponential backoff
    for attempt in range(1, max_attempts + 1):
        try:
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
                    original_error=Exception(f"Downloaded file too small: {file_size} bytes")
                )

            logger.info(f"Successfully downloaded video {video_id} ({file_size / 1024 / 1024:.2f} MB)")
            return output_path

        except (requests.exceptions.RequestException, DownloadError) as e:
            if os.path.exists(output_path):
                os.remove(output_path)

            if attempt < max_attempts:
                sleep_time = 2 ** attempt  # 2s, 4s, 8s
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise DownloadError(video_id=video_id, attempts=max_attempts, original_error=e)
```

---

### 7. validate_temporal_windows_schema() - Output Validation

**Location**: `ml_pipeline/stage2_processing/validation.py:17-89`

**Purpose**: Validate temporal_windows_updated.json structure

**Implementation**:
```python
def validate_temporal_windows_schema(insights: Dict[str, Any]):
    """Validate temporal_windows_updated.json structure and completeness."""

    video_id = insights.get('video_id', 'unknown')

    # 1. Check required top-level keys
    required_keys = ['temporal_windows', 'metadata', 'processing_timestamp']
    missing = [k for k in required_keys if k not in insights]
    if missing:
        raise ValidationError(video_id, 'top_level_keys', str(required_keys), f"missing: {missing}")

    # 2. Validate temporal_windows structure
    windows = insights['temporal_windows']
    if not isinstance(windows, dict):
        raise ValidationError(video_id, 'temporal_windows', 'dict', type(windows).__name__)

    # 3. Check required window sections
    required_sections = ['hook', 'closing']
    for section in required_sections:
        if section not in windows:
            raise ValidationError(video_id, f'temporal_windows.{section}', 'present', 'missing')
        if not isinstance(windows[section], dict):
            raise ValidationError(video_id, f'temporal_windows.{section}', 'dict', type(windows[section]).__name__)

        # 4. Check feature count (expect 60+ features per window)
        if len(windows[section]) < 50:
            logger.warning(f"Window section '{section}' has only {len(windows[section])} features (expected 60+)")

    # 5. Validate middle_segments logic
    video_duration = insights.get('duration', 0)
    middle_segments = windows.get('middle_segments')

    if video_duration < 9:
        # Short videos: middle_segments should be null
        if middle_segments is not None:
            raise ValidationError(video_id, 'middle_segments', 'null', f'not null (duration={video_duration}s)')
    else:
        # Longer videos: middle_segments should be list of dicts
        if not isinstance(middle_segments, list):
            raise ValidationError(video_id, 'middle_segments', 'list', type(middle_segments).__name__)

        for i, segment in enumerate(middle_segments):
            if not isinstance(segment, dict):
                raise ValidationError(video_id, f'middle_segments[{i}]', 'dict', type(segment).__name__)
            if len(segment) < 50:
                logger.warning(f"Middle segment {i} has only {len(segment)} features (expected 60+)")

    # 6. Validate metadata structure
    metadata = insights['metadata']
    if not isinstance(metadata, dict):
        raise ValidationError(video_id, 'metadata', 'dict', type(metadata).__name__)

    logger.debug(f"Schema validation passed for video {video_id}")
```

---

## Checkpoint Strategy

### Checkpoint Location

**Path**: `{bucket_path}/checkpoints/stage_2_checkpoint.json`

**Backup Path**: `{bucket_path}/checkpoints/stage_2_checkpoint.backup.json`

### Automatic Backup Strategy

**Location**: `checkpoint.py:26-51`

```python
def save_checkpoint_with_backup(checkpoint_path: str, checkpoint: dict):
    """Save checkpoint with automatic backup."""
    backup_path = checkpoint_path.replace('.json', '.backup.json')

    # Backup existing checkpoint before overwriting
    if os.path.exists(checkpoint_path):
        try:
            shutil.copy2(checkpoint_path, backup_path)
        except Exception as e:
            logger.warning(f"Failed to backup checkpoint: {e}")

    # Write new checkpoint (atomic write in utils.py)
    save_json(checkpoint_path, checkpoint)
```

### Atomic Write Implementation

**Location**: `utils.py:56-95`

```python
def save_json(filepath: str, data: Dict[str, Any]):
    """Save dictionary to JSON with atomic write."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file first, then rename
    temp_path = filepath.with_suffix(filepath.suffix + '.tmp')

    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_path.replace(filepath)

    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to write JSON to {filepath}: {e}")
```

### Resume Behavior

**Automatic Resume on Re-run**:

1. **Check checkpoint exists** (checkpoint.py:145-155)
2. **Validate config matches** (checkpoint.py:122-138)
   - Critical fields: `video_count`, `selection_strategy`, `date_filter`
   - Raises `ValueError` if mismatch
3. **Filter to remaining videos** (checkpoint.py:149-150)
   ```python
   completed_ids = set(checkpoint['completed_video_ids'])
   remaining_videos = [v for v in video_list if v['id'] not in completed_ids]
   ```
4. **Log resume info** (checkpoint.py:152-153)
5. **Continue processing** from where it left off

**Pause and Resume Flow**:

```
User runs pipeline
  ↓
Processing video 1... ✓ (checkpoint saved)
Processing video 2... ✓ (checkpoint saved)
Processing video 3... [User presses Ctrl+C]
  ↓
Pause handler detects signal
  ↓
Current video completes
  ↓
Checkpoint saved with status="paused"
  ↓
Pipeline exits gracefully
  ↓
[User re-runs same command]
  ↓
initialize_checkpoint() detects existing checkpoint
  ↓
Validates config matches
  ↓
Filters to remaining videos (4-100)
  ↓
Resumes processing from video 4
```

---

## Error Handling

### Error Matrix

| Error Type | Exception | Location | Strategy | Recovery |
|------------|-----------|----------|----------|----------|
| **Download Failed** | `DownloadError` | video_download.py:116 | Skip video | Use webVideoUrl in processing |
| **Subprocess Timeout** | `TimeoutError` | video_processor.py:119 | Skip video | Logged in failed_video_ids |
| **Subprocess Failed** | `CalledProcessError` | video_processor.py:121 | Skip video | Check stderr for details |
| **Processing Error** | `ProcessingError` | video_processor.py:186 | Skip video | Logged in failed_video_ids |
| **Validation Failed** | `ValidationError` | validation.py:39+ | Skip video | Check schema mismatch |
| **Checkpoint Corrupt** | `CheckpointCorruptionError` | checkpoint.py:93 | Restore from backup | If backup corrupt, manual recovery |
| **Disk Full** | `OSError` | video_processor.py:243 | **Exit pipeline** | Free disk space, resume |
| **Memory Error** | `MemoryError` | video_processor.py:243 | **Exit pipeline** | Increase memory, resume |
| **Config Mismatch** | `ValueError` | checkpoint.py:135 | **Exit pipeline** | Use same config or delete checkpoint |

### Custom Exceptions

**Location**: `ml_pipeline/stage2_processing/exceptions.py`

**1. DownloadError** (exceptions.py:8-22):
```python
class DownloadError(Exception):
    """Raised when video download fails after max retry attempts."""
    def __init__(self, video_id: str, attempts: int, original_error: Exception):
        self.video_id = video_id
        self.attempts = attempts
        self.original_error = original_error
        super().__init__(
            f"Failed to download video {video_id} after {attempts} attempts: {original_error}"
        )
```

**2. ProcessingError** (exceptions.py:25-37):
```python
class ProcessingError(Exception):
    """Raised when RumiAI pipeline fails."""
    def __init__(self, video_id: str, stage: str, message: str):
        self.video_id = video_id
        self.stage = stage
        self.message = message
        super().__init__(f"RumiAI processing failed for {video_id} at stage {stage}: {message}")
```

**3. ValidationError** (exceptions.py:40-56):
```python
class ValidationError(Exception):
    """Raised when output schema validation fails."""
    def __init__(self, video_id: str, field: str, expected: str, actual: str):
        self.video_id = video_id
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Schema validation failed for {video_id}: "
            f"field '{field}' expected {expected}, got {actual}"
        )
```

**4. CheckpointCorruptionError** (exceptions.py:59-82):
```python
class CheckpointCorruptionError(Exception):
    """Raised when both checkpoint and backup are corrupted."""
    def __init__(self, checkpoint_path: str, backup_path: str, original_error: Exception):
        recovery_msg = (
            f"Checkpoint and backup both corrupted.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Backup: {backup_path}\n"
            f"Original error: {original_error}\n\n"
            f"Recovery options:\n"
            f"  1. Delete checkpoint and restart Stage 2 (loses progress)\n"
            f"  2. Manually inspect checkpoint files for partial recovery\n"
            f"  3. Contact support if data recovery is critical"
        )
        super().__init__(recovery_msg)
```

### Skip-on-Fail Policy

**Implementation** (video_processor.py:217-267):

```python
def handle_video_processing_error(error, video_id, checkpoint, checkpoint_path):
    """Skip-on-fail: Mark video as failed, continue with next."""

    # Check for critical errors (should NOT skip)
    if isinstance(error, (OSError, MemoryError)) and "disk full" in str(error).lower():
        logger.critical(f"CRITICAL ERROR: Disk full. Cannot continue.")
        raise error  # Exit pipeline

    # Non-critical error: Skip video
    logger.error(f"Failed to process video {video_id}: {error}")

    checkpoint['failed'] += 1
    checkpoint['remaining'] -= 1
    checkpoint['failed_video_ids'].append({
        "video_id": video_id,
        "error": str(error),
        "error_type": type(error).__name__,
        "timestamp": datetime.utcnow().isoformat()
    })

    save_checkpoint_with_backup(checkpoint_path, checkpoint)
    logger.warning(f"Video {video_id} marked as failed. Continuing batch processing.")

    return "failed"
```

**Result**: Pipeline processes as many videos as possible, tracks failures in checkpoint

---

## Subprocess Contract (rumiai_runner.py)

### Input Contract

**Subprocess Command** (video_processor.py:64-68):
```bash
python3 scripts/rumiai_runner.py <video_url_or_path>
```

**Arguments**:
- Positional arg 1: TikTok URL (preferred) OR local MP4 path
- Must be valid URL starting with `http://` or `https://`
- Local paths are supported but may fail URL validation

**Example**:
```bash
python3 scripts/rumiai_runner.py https://www.tiktok.com/@user/video/7123456789
```

### Output Contract

**Output Location** (HARDCODED):
```
/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json
```

**Exit Codes** (rumiai_runner.py:454-459):
- `0` - Success (insights file generated)
- `1` - General failure
- `2` - Invalid arguments
- `3` - API failure
- `4` - ML processing failure

**Success Criteria**:
1. Exit code 0
2. Insights file exists at hardcoded path
3. Insights file passes schema validation

**Failure Detection**:
```python
# Exit code != 0
except subprocess.CalledProcessError as e:
    raise ProcessingError(f"RumiAI pipeline failed (exit code {e.returncode})")

# Exit code 0 but no output file
if not os.path.exists(insights_path):
    raise ProcessingError(f"RumiAI exited successfully but no insights file at {insights_path}")
```

### Processing Steps (High-Level)

**rumiai_runner.py executes** (see rumiai_runner.py documentation for details):

1. **Scrape metadata** from TikTok (if URL provided)
2. **Download video** to temp directory
3. **Run 9 ML services** (sequential):
   - YOLO - Object detection
   - Whisper - Speech transcription
   - MediaPipe - Pose/gesture detection
   - OCR - Text detection
   - Scene Detection - Visual scene analysis
   - Audio Energy - Audio level analysis
   - **FEAT - Emotion detection (BOTTLENECK: 43% of time)**
   - DeepFace - Gender classification
   - MediaPipe Face - Facial landmarks
4. **Build timeline** - Merge service outputs
5. **Compute temporal windows** - Extract 350+ features
6. **Save output** - Write temporal_windows_updated.json

**Typical Duration**: ~60-80 seconds per 60-second video

### Timeout Handling

**Timeout**: 300 seconds (5 minutes)

**Implementation** (video_processor.py:71-76):
```python
result = subprocess.run(
    cmd,
    timeout=300,  # Kill process if exceeds 5 minutes
    capture_output=True,
    text=True,
    check=True
)
```

**On Timeout**:
```python
except subprocess.TimeoutExpired:
    raise TimeoutError(f"RumiAI processing exceeded 300s timeout for video {video_id}")
```

**Result**: Video marked as failed, pipeline continues

---

## Debugging Guide

### Common Issues

#### Issue 1: "RumiAI exited successfully (code 0) but no insights file"

**Symptom**:
```
ProcessingError: RumiAI exited successfully (code 0) but no insights file at /home/jorge/rumiaifinal/insights/7123456789_temporal_windows_updated.json
```

**Cause**: rumiai_runner.py completed but failed to write output file

**Debug**:
```bash
# Check if insights directory exists
ls -la /home/jorge/rumiaifinal/insights/

# Check rumiai_runner.py logs
grep "7123456789" logs/rumiai_v2.log

# Run rumiai_runner.py manually to see full output
python3 scripts/rumiai_runner.py "https://www.tiktok.com/@user/video/7123456789"
```

**Fix**:
1. Check disk space: `df -h`
2. Check permissions: `ls -ld /home/jorge/rumiaifinal/insights/`
3. Check rumiai_runner.py logs for ML service failures

**Location**: `video_processor.py:105-110`

---

#### Issue 2: "Checkpoint and backup both corrupted"

**Symptom**:
```
CheckpointCorruptionError: Checkpoint and backup both corrupted.
Checkpoint: /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/checkpoints/stage_2_checkpoint.json
Backup: /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/checkpoints/stage_2_checkpoint.backup.json

Recovery options:
  1. Delete checkpoint and restart Stage 2 (loses progress)
  2. Manually inspect checkpoint files for partial recovery
  3. Contact support if data recovery is critical
```

**Cause**: Both checkpoint files have invalid JSON (incomplete writes, disk issues)

**Debug**:
```bash
# Check checkpoint file
cat {bucket_path}/checkpoints/stage_2_checkpoint.json

# Check backup
cat {bucket_path}/checkpoints/stage_2_checkpoint.backup.json

# Validate JSON
jq empty {bucket_path}/checkpoints/stage_2_checkpoint.json
```

**Fix Options**:

**Option 1: Restart from scratch (loses progress)**
```bash
rm {bucket_path}/checkpoints/stage_2_checkpoint.json
rm {bucket_path}/checkpoints/stage_2_checkpoint.backup.json
# Re-run pipeline
```

**Option 2: Manual recovery**
```bash
# Try to fix JSON manually (remove trailing commas, fix brackets)
vi {bucket_path}/checkpoints/stage_2_checkpoint.json

# Validate fixed JSON
jq empty {bucket_path}/checkpoints/stage_2_checkpoint.json
```

**Option 3: Partial recovery**
```bash
# Extract completed video IDs from logs
grep "Successfully processed video" logs/*.log | awk '{print $NF}' > completed_ids.txt

# Manually create minimal checkpoint
cat > {bucket_path}/checkpoints/stage_2_checkpoint.json <<EOF
{
  "stage": "video_processing",
  "bucket": "18-33s",
  "total_videos": 100,
  "completed": 45,
  "completed_video_ids": [...]
  "status": "in_progress"
}
EOF
```

**Location**: `checkpoint.py:93-97`

---

#### Issue 3: "Config mismatch detected. Cannot resume with different parameters"

**Symptom**:
```
ValueError: Config mismatch detected. Cannot resume with different parameters:
video_count: checkpoint=100, current=40
```

**Cause**: Trying to resume with different CLI parameters than original run

**Debug**:
```bash
# Check current CLI args
echo "Current: --video-count 40"

# Check checkpoint config
cat {bucket_path}/checkpoints/stage_2_checkpoint.json | jq '.config.video_count'
# Output: 100
```

**Fix**:

**Option 1: Use same parameters**
```bash
# Use original parameters from checkpoint
python rumiai_ml_batch.py --client acme --target nutrition --video-count 100
```

**Option 2: Delete checkpoint and restart**
```bash
# If you want to use new parameters, delete checkpoint
rm {bucket_path}/checkpoints/stage_2_checkpoint.json
rm {bucket_path}/checkpoints/stage_2_checkpoint.backup.json

# Re-run with new parameters
python rumiai_ml_batch.py --client acme --target nutrition --video-count 40
```

**Location**: `checkpoint.py:122-138`

---

#### Issue 4: Processing hangs (no progress)

**Symptom**: Pipeline appears stuck, no log output for >10 minutes

**Cause**: Subprocess (rumiai_runner.py) hung on ML service

**Debug**:
```bash
# Check if rumiai_runner.py subprocess is running
ps aux | grep rumiai_runner.py

# Check which ML service is running (may be stuck)
ps aux | grep -E "python|yolo|whisper|feat"

# Check resource usage
top -p <rumiai_runner_pid>

# Check logs for last activity
tail -f logs/rumiai_v2.log
```

**Fix**:

**Option 1: Wait for timeout (300s)**
- Subprocess will auto-kill after 5 minutes
- Video will be marked as failed

**Option 2: Manual kill**
```bash
# Find subprocess PID
ps aux | grep rumiai_runner.py

# Kill subprocess
kill -9 <pid>

# Pipeline will catch subprocess.TimeoutExpired and continue
```

**Option 3: Reduce timeout** (if videos consistently timeout):
```python
# File: ml_pipeline/stage2_processing/video_processor.py:180
result = run_rumiai_pipeline(
    video_path=video_path,
    video_id=video_id,
    output_dir=f"{bucket_path}analysis/",
    timeout=180  # Reduce from 300s to 180s
)
```

**Location**: `video_processor.py:71`

---

#### Issue 5: Validation failed - "middle_segments expected null"

**Symptom**:
```
ValidationError: Schema validation failed for 7123456789: field 'middle_segments' expected null, got list (duration=5s)
```

**Cause**: Short video (<9s) has middle_segments populated (should be null)

**Debug**:
```bash
# Check video duration
cat /home/jorge/rumiaifinal/insights/7123456789_temporal_windows_updated.json | jq '.duration'
# Output: 5

# Check middle_segments value
cat /home/jorge/rumiaifinal/insights/7123456789_temporal_windows_updated.json | jq '.temporal_windows.middle_segments'
# Output: [{"feature": "value"}]  (should be null)
```

**Fix**: This is a rumiai_runner.py bug - see rumiai_runner.py documentation

**Workaround**: Skip validation for this video (manual intervention)

**Location**: `validation.py:62-65`

---

### Debug Commands

**Check Stage 2 progress**:
```bash
# View checkpoint status
cat {bucket_path}/checkpoints/stage_2_checkpoint.json | jq '{
  completed: .completed,
  failed: .failed,
  remaining: .remaining,
  status: .status
}'

# Count completed videos
ls /home/jorge/rumiaifinal/insights/*_temporal_windows_updated.json | wc -l

# Check last checkpoint time
cat {bucket_path}/checkpoints/stage_2_checkpoint.json | jq '.last_checkpoint'
```

**Check failed videos**:
```bash
# List all failed videos
cat {bucket_path}/checkpoints/stage_2_checkpoint.json | jq '.failed_video_ids[] | {
  video_id: .video_id,
  error_type: .error_type,
  error: .error
}'

# Count failures by error type
cat {bucket_path}/checkpoints/stage_2_checkpoint.json | jq '.failed_video_ids | group_by(.error_type) | map({error_type: .[0].error_type, count: length})'
```

**Monitor processing in real-time**:
```bash
# Watch checkpoint updates
watch -n 5 'cat {bucket_path}/checkpoints/stage_2_checkpoint.json | jq ".completed"'

# Tail logs
tail -f logs/rumiai_ml_batch.log | grep "Stage 2"
```

**Validate outputs**:
```bash
# Check if all completed videos have insights files
for video_id in $(cat {bucket_path}/checkpoints/stage_2_checkpoint.json | jq -r '.completed_video_ids[]'); do
  if [ ! -f "/home/jorge/rumiaifinal/insights/${video_id}_temporal_windows_updated.json" ]; then
    echo "MISSING: $video_id"
  fi
done

# Validate JSON structure
for file in /home/jorge/rumiaifinal/insights/*_temporal_windows_updated.json; do
  jq empty "$file" 2>/dev/null || echo "INVALID JSON: $file"
done
```

---

## Modification Guide

### Scenario 1: Change Subprocess Timeout from 300s to 600s

**Requirement**: Increase timeout for long videos or slow ML services

**Files to Modify**:
1. `ml_pipeline/stage2_processing/video_processor.py`
2. `ml_pipeline/stage2_processing/pause_handler.py`

**Steps**:

**Step 1**: Update timeout in video_processor.py
```python
# File: ml_pipeline/stage2_processing/video_processor.py:30
def run_rumiai_pipeline(
    video_path: str,
    video_id: str,
    output_dir: str,
    timeout: int = 600  # Changed from 300 to 600
) -> Dict[str, Any]:
```

**Step 2**: Update timeout in process_videos_sequential
```python
# File: ml_pipeline/stage2_processing/video_processor.py:180
result = run_rumiai_pipeline(
    video_path=video_path,
    video_id=video_id,
    output_dir=f"{bucket_path}analysis/",
    timeout=600  # Changed from 300 to 600
)
```

**Step 3**: Update timeout in pause_handler.py
```python
# File: ml_pipeline/stage2_processing/pause_handler.py:130
result = run_rumiai_pipeline(
    video_path=video_path,
    video_id=video_id,
    output_dir=f"{bucket_path}analysis/",
    timeout=600  # Changed from 300 to 600
)
```

**Test**:
```bash
# Run Stage 2 and check logs for timeout value
python rumiai_ml_batch.py --client test --target nutrition

# Check logs
grep "timeout" logs/rumiai_ml_batch.log
```

---

### Scenario 2: Add Retry Logic for Failed Videos

**Requirement**: Automatically retry failed videos once before marking as failed

**Files to Modify**:
1. `ml_pipeline/stage2_processing/video_processor.py`

**Steps**:

**Step 1**: Add retry parameter to handle_video_processing_error
```python
# File: ml_pipeline/stage2_processing/video_processor.py:217
def handle_video_processing_error(
    error: Exception,
    video_id: str,
    checkpoint: dict,
    checkpoint_path: str,
    retry_count: int = 0,  # NEW: Track retry attempts
    max_retries: int = 1   # NEW: Max 1 retry
) -> str:
```

**Step 2**: Implement retry logic
```python
# File: ml_pipeline/stage2_processing/video_processor.py:240-267
def handle_video_processing_error(error, video_id, checkpoint, checkpoint_path, retry_count=0, max_retries=1):
    """Handle errors with optional retry."""

    # Check for critical errors (no retry)
    if isinstance(error, (OSError, MemoryError)) and "disk full" in str(error).lower():
        logger.critical(f"CRITICAL ERROR: Disk full. Cannot continue.")
        raise error

    # Retry logic
    if retry_count < max_retries:
        logger.warning(f"Video {video_id} failed (attempt {retry_count + 1}), retrying...")
        time.sleep(5)  # Wait 5s before retry
        return "retry"  # Signal to retry

    # Max retries exceeded - mark as failed
    logger.error(f"Failed to process video {video_id} after {max_retries + 1} attempts: {error}")

    checkpoint['failed'] += 1
    checkpoint['remaining'] -= 1
    checkpoint['failed_video_ids'].append({
        "video_id": video_id,
        "error": str(error),
        "error_type": type(error).__name__,
        "retry_count": retry_count,
        "timestamp": datetime.utcnow().isoformat()
    })

    save_checkpoint_with_backup(checkpoint_path, checkpoint)
    return "failed"
```

**Step 3**: Update caller to handle retry
```python
# File: ml_pipeline/stage2_processing/video_processor.py:205-208
except Exception as e:
    action = handle_video_processing_error(e, video_id, checkpoint, checkpoint_path, retry_count=0)
    if action == "retry":
        # Retry once (could wrap in loop for multiple retries)
        try:
            # Run processing again
            result = run_rumiai_pipeline(...)
            # ... success handling ...
        except Exception as retry_error:
            handle_video_processing_error(retry_error, video_id, checkpoint, checkpoint_path, retry_count=1)
    continue
```

**Test**:
```bash
# Run with a video known to fail
# Check logs for retry attempt
grep "retrying" logs/rumiai_ml_batch.log
```

---

### Scenario 3: Change Output Directory from Hardcoded to Bucket-Specific

**Requirement**: Make rumiai_runner.py output to bucket directories instead of flat `/insights/`

**Current Behavior**: Hardcoded `/home/jorge/rumiaifinal/insights/` (Stage 2.5 moves files later)

**Files to Modify**:
1. `ml_pipeline/stage2_processing/video_processor.py`
2. `scripts/rumiai_runner.py` (pass output dir as argument)

**Note**: This is a MAJOR change affecting rumiai_runner.py contract

**Steps**:

**Step 1**: Add --output-dir argument to rumiai_runner.py
```python
# File: scripts/rumiai_runner.py:466
parser.add_argument('--output-dir', help='Output directory for insights', default='/home/jorge/rumiaifinal/insights/')
```

**Step 2**: Update subprocess call to pass output dir
```python
# File: ml_pipeline/stage2_processing/video_processor.py:64-68
cmd = [
    sys.executable,
    'scripts/rumiai_runner.py',
    video_path,
    '--output-dir', output_dir  # NEW: Pass bucket-specific output dir
]
```

**Step 3**: Update output path construction
```python
# File: ml_pipeline/stage2_processing/video_processor.py:62
# OLD: insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
# NEW: insights_path = f"{output_dir}/{video_id}_temporal_windows_updated.json"
```

**Step 4**: Update rumiai_runner.py to use --output-dir
```python
# File: scripts/rumiai_runner.py (multiple locations)
# Replace hardcoded RUMIAI_OUTPUT_DIR with args.output_dir
```

**Impact**: Stage 2.5 (File Organization) becomes unnecessary

**Test**:
```bash
# Run Stage 2
python rumiai_ml_batch.py --client test --target nutrition

# Check outputs in bucket directory
ls {bucket_path}/analysis/insights/*_temporal_windows_updated.json

# Verify Stage 2.5 can be skipped
```

---

### Scenario 4: Disable Pre-Download (Always Use URLs)

**Requirement**: Skip video pre-download, always pass URLs to rumiai_runner.py

**Files to Modify**:
1. `ml_pipeline/stage2_processing/main.py`

**Steps**:

**Step 1**: Comment out pre-download logic
```python
# File: ml_pipeline/stage2_processing/main.py:77-109

# Step 2: Download videos if downloadAddr available (optional pre-download)
logger.info(f"Step 2: Checking for pre-downloadable videos")
bucket_path = get_bucket_path(config, bucket_name)
videos_dir = f"{bucket_path}videos/"

# DISABLED: Pre-download logic
"""
downloadable_count = 0
for i, video in enumerate(remaining_videos, start=1):
    ...
    download_video(video, videos_dir)
    ...
"""
downloadable_count = 0  # Always 0 (disabled)

logger.info(f"Pre-download disabled, using URLs for all videos")
```

**Step 2**: Ensure hybrid mode always uses URL
```python
# File: ml_pipeline/stage2_processing/video_processor.py:160-172
# This already prefers URLs, no change needed:
if 'webVideoUrl' in video and video['webVideoUrl']:
    video_path = video['webVideoUrl']
    logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id} (TikTok URL)")
elif os.path.exists(local_video_path):
    video_path = local_video_path
    logger.warning(f"Processing video {i}/{len(remaining_videos)}: {video_id} (local file)")
```

**Impact**: Faster startup, relies on rumiai_runner.py to download

**Test**:
```bash
# Run Stage 2
python rumiai_ml_batch.py --client test --target nutrition

# Check logs - should show "Pre-download disabled"
grep "Pre-download" logs/rumiai_ml_batch.log
```

---

### Scenario 5: Add Progress Bar for Video Processing

**Requirement**: Show visual progress bar during video processing

**Files to Modify**:
1. `ml_pipeline/stage2_processing/pause_handler.py`
2. Add dependency: `tqdm`

**Steps**:

**Step 1**: Install tqdm
```bash
pip install tqdm
```

**Step 2**: Import tqdm
```python
# File: ml_pipeline/stage2_processing/pause_handler.py:1
from tqdm import tqdm
```

**Step 3**: Wrap video loop with progress bar
```python
# File: ml_pipeline/stage2_processing/pause_handler.py:85
# OLD: for i, video in enumerate(remaining_videos, start=1):
# NEW:
with tqdm(total=len(remaining_videos), desc=f"Processing {bucket_name}", unit="video") as pbar:
    for i, video in enumerate(remaining_videos, start=1):
        # ... existing code ...

        # Update progress bar on success
        if successfully_processed:
            pbar.update(1)
            pbar.set_postfix({
                'completed': checkpoint['completed'],
                'failed': checkpoint['failed']
            })
```

**Test**:
```bash
# Run Stage 2 - should see progress bar
python rumiai_ml_batch.py --client test --target nutrition

# Output:
# Processing 18-33s: 45%|████████      | 45/100 [15:30<17:45, 19.4s/video] completed=45, failed=2
```

---

## Related Documentation

### Pipeline Documentation
- **[PRODUCTION_FLOW.md](PRODUCTION_FLOW.md)**: Complete pipeline overview (Stages 0-7)
- **[PRODUCTION_FLOW.md - Stage 2 Contract](PRODUCTION_FLOW.md#stage-2-video-processing-ml-services)**: Stage 2 inputs/outputs/dependencies

### Upstream/Downstream Stages
- **[STAGE_1_IMPL.md](STAGE_1_IMPL.md)**: Video Discovery & Selection (provides selected_videos.json)
- **Stage 2.5**: File Organization (moves temporal_windows files to bucket directories)

### External Dependencies
- **scripts/rumiai_runner.py**: ML pipeline subprocess (see rumiai_runner.py documentation)
- **9 ML Services**: YOLO, Whisper, MediaPipe, OCR, Scene Detection, Audio Energy, FEAT, DeepFace, MediaPipe Face

### Foundation Documentation
- **[foundation/paths.py](foundation/paths.py)**: Path building (get_bucket_path)
- **[foundation/buckets.py](foundation/buckets.py)**: Bucket definitions (VALID_BUCKETS)

### Stage 2 Source Files
- **[ml_pipeline/stage2_processing/main.py](ml_pipeline/stage2_processing/main.py)**: Main orchestrator (184 lines)
- **[ml_pipeline/stage2_processing/video_processor.py](ml_pipeline/stage2_processing/video_processor.py)**: Core processing (267 lines)
- **[ml_pipeline/stage2_processing/checkpoint.py](ml_pipeline/stage2_processing/checkpoint.py)**: Checkpoint management (208 lines)
- **[ml_pipeline/stage2_processing/pause_handler.py](ml_pipeline/stage2_processing/pause_handler.py)**: Ctrl+C handling (161 lines)
- **[ml_pipeline/stage2_processing/bucket_init.py](ml_pipeline/stage2_processing/bucket_init.py)**: Directory creation (167 lines)
- **[ml_pipeline/stage2_processing/video_download.py](ml_pipeline/stage2_processing/video_download.py)**: Video download (120 lines)
- **[ml_pipeline/stage2_processing/utils.py](ml_pipeline/stage2_processing/utils.py)**: Helper functions (134 lines)
- **[ml_pipeline/stage2_processing/validation.py](ml_pipeline/stage2_processing/validation.py)**: Schema validation (88 lines)
- **[ml_pipeline/stage2_processing/exceptions.py](ml_pipeline/stage2_processing/exceptions.py)**: Custom exceptions (82 lines)

---

## Document Metadata

**Generated**: 2025-11-02
**Source**: 100% systematic code reading (1,434 production lines across 9 modules + 508 lines rumiai_runner.py)
**Verification**: All line numbers, schemas, and code snippets from actual source code
**Coverage**: Complete Stage 2 processing wrapper (orchestration, checkpoint, error handling, subprocess contract)

**Last Validated**: 2025-11-02
**Subprocess**: scripts/rumiai_runner.py - Outputs to hardcoded `/home/jorge/rumiaifinal/insights/`
