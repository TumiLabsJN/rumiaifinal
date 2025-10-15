# File Organization (Stage 2.5) - High-Level Design

> **Parent**: MLPlanningv2.md - Stage 2.5
> **Version**: 1.0
> **Last Updated**: 2025-01-13
> **Status**: Draft

---

## 1. Context & Business Goal

<!-- PURPOSE: Provide business context and justification. TI generator needs to understand WHY this feature exists. -->

### 1.1 What Problem Does This Solve?

Stage 2 (rumiai_runner.py) processes videos sequentially, saving all temporal_windows_updated.json files to a flat `/insights/` directory with no bucket awareness. This creates a organizational problem: Stage 3 (Feature Aggregation) requires bucket-organized inputs to process videos efficiently within their duration-specific groups. Stage 2.5 solves this by performing a one-time batch file organization operation, moving files from the flat structure into bucket-specific directories based on video duration, enabling downstream stages to process videos in properly organized duration buckets.

### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This stage depends on FoundationCHILD.md for:
- Client directory structure (Section 2: Client Architecture & Storage)
- CLI parameter definitions (Section 4: CLI Command Structure)
- Config.json schema (Section 5.1: Configuration Schemas)
- Checkpoint schema (Section 5.3: Checkpoint Schema)
- Bucket definitions (Section 6: Bucket Definitions)

```
Stage 1: Video Discovery & Winner Analysis
   ↓ Output: winner_analysis.json (top_3_buckets), selected_videos.json (per bucket)
Stage 2: Video Processing (RumiAI Pipeline)
   ↓ Output: temporal_windows_updated.json (N files in flat /insights/ directory)
   ↓ Output: stage_2_checkpoint.json (per bucket, with completed_video_ids)
Stage 2.5: File Organization [THIS COMPONENT]
   ↓ Output: temporal_windows_updated.json (organized into bucket-specific directories)
Stage 3: Feature Aggregation
```

### 1.3 Success Criteria

- [ ] Organize 300 temporal_windows JSON files in < 5 seconds
- [ ] Zero data loss during file moves (atomic operations within filesystem)
- [ ] Idempotent operation (safe to re-run without side effects)
- [ ] Process only winning buckets identified by Stage 1
- [ ] Skip videos from non-winning buckets with clear warning logs
- [ ] Fail-fast on checkpoint corruption or missing dependencies
- [ ] Support automatic resume detection after interruption

---

## 2. Architecture & Design

<!-- PURPOSE: Core technical design. This is the PRIMARY section TI generator reads. -->

### 2.1 High-Level Approach

Stage 2.5 is a checkpoint-driven batch file organization operation. It reads winner_analysis.json from Stage 1 to determine which 3 winning buckets to process, then loads Stage 2 checkpoints for those buckets to get the list of successfully completed video IDs. For each video, it moves the temporal_windows JSON file from the flat `/insights/` directory to the appropriate bucket-specific directory based on duration assignment. The operation uses detection-based resume (checking filesystem state) rather than maintaining its own checkpoint, making it idempotent and safe to re-run. Files are moved (not copied) to avoid storage duplication, with graceful handling of missing files and automatic directory creation.

### 2.2 Data Flow

```
Input 1: winner_analysis.json (from Stage 1)
        Location: {analysis_base}/winner_analysis.json
        Schema: {"top_3_buckets": ["18-33s", "33-60s", "13-18s"]}
   ↓
Input 2: stage_2_checkpoint.json (per winning bucket)
        Location: {analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json
        Schema: {"completed_video_ids": ["123", "456", "789"]}
   ↓
Input 3: temporal_windows_updated.json files (flat directory)
        Location: /home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json
        Schema: 60+ features per temporal window
   ↓
Process Step 1: Load winning buckets from winner_analysis.json
   ↓
Process Step 2: For each winning bucket, load checkpoint to get completed_video_ids
   ↓
Process Step 3: Build file list (video_id → source path, target path, bucket)
   ↓
Process Step 4: Validate no duplicate video_ids across buckets
   ↓
Process Step 5: For each file, check if already organized (target exists, source doesn't)
   ↓
Process Step 6: Move file from source to target (create directories if needed)
   ↓
Output: temporal_windows_updated.json files organized by bucket
        Location: {analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json
        Result: Empty /insights/ directory, files distributed to winning buckets
```

### 2.3 Detailed Process

#### Step 2.3.1: Load Winning Buckets

**Purpose**: Determine which 3 buckets were selected by Stage 1 winner analysis

**Logic**:
```python
def load_winning_buckets(analysis_base):
    """
    Load winning buckets from winner_analysis.json.

    Args:
        analysis_base: str, path to analysis directory
                      Example: /data/clients/acme/hashtags/nutrition/top_contrastive/

    Returns:
        list: winning bucket names (e.g., ["18-33s", "33-60s", "13-18s"])

    Raises:
        FileNotFoundError: if winner_analysis.json doesn't exist
        ValueError: if file is corrupted or has invalid schema

    Source: QA Q5, Q10
    """
    winner_analysis_path = f"{analysis_base}/winner_analysis.json"

    # Validate file exists
    if not os.path.exists(winner_analysis_path):
        raise FileNotFoundError(
            f"winner_analysis.json not found at:\n"
            f"  {winner_analysis_path}\n\n"
            f"This file is created by Stage 1.3 (Winner Analysis).\n"
            f"Stage 2.5 requires this file to know which buckets to organize.\n\n"
            f"Solutions:\n"
            f"  1. Complete Stage 1 (Video Discovery & Winner Analysis)\n"
            f"  2. Check if Stage 1 completed successfully\n"
            f"  3. Verify analysis_base path is correct: {analysis_base}"
        )

    # Load and validate
    with open(winner_analysis_path) as f:
        winner_analysis = json.load(f)

    if 'top_3_buckets' not in winner_analysis:
        raise ValueError(f"winner_analysis.json missing 'top_3_buckets' field")

    if not isinstance(winner_analysis['top_3_buckets'], list):
        raise TypeError(f"'top_3_buckets' must be list, got {type(winner_analysis['top_3_buckets'])}")

    if len(winner_analysis['top_3_buckets']) == 0:
        raise ValueError("'top_3_buckets' is empty - no winning buckets identified")

    logger.info(f"Loaded winning buckets: {winner_analysis['top_3_buckets']}")
    return winner_analysis['top_3_buckets']
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| File doesn't exist | Fail-fast with FileNotFoundError | Stage 1 didn't complete |
| JSON corrupted | Fail-fast with ValueError | Cannot proceed without valid data |
| Empty top_3_buckets | Fail-fast with ValueError | No buckets to organize |

#### Step 2.3.2: Build File List from Checkpoints

**Purpose**: Determine exactly which files to organize by reading Stage 2 checkpoints for completed video IDs

**Logic**:
```python
def build_file_list(analysis_base, winning_buckets):
    """
    Build list of files to organize from Stage 2 checkpoints.

    Args:
        analysis_base: str, path to analysis directory
        winning_buckets: list, bucket names from winner_analysis.json

    Returns:
        list: file info dicts with keys: video_id, bucket, source_path, target_path

    Source: QA Q8, Q9
    """
    files_to_process = []

    for bucket in winning_buckets:
        # Load checkpoint for this bucket
        checkpoint_path = f"{analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json"

        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint missing for bucket {bucket}: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found for bucket {bucket}. Did Stage 2 complete?")

        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Validate checkpoint schema
        video_ids = validate_checkpoint(checkpoint, bucket)

        # Skip if no completed videos
        if len(video_ids) == 0:
            logger.info(f"Bucket {bucket} has 0 completed videos. Skipping.")
            continue

        # Build file info for each video
        for video_id in video_ids:
            source_path = f"/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json"
            target_path = f"{analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json"

            files_to_process.append({
                'video_id': video_id,
                'bucket': bucket,
                'source_path': source_path,
                'target_path': target_path
            })

    logger.info(f"Built file list: {len(files_to_process)} files across {len(winning_buckets)} buckets")
    return files_to_process


def validate_checkpoint(checkpoint, bucket):
    """
    Validate checkpoint schema and extract completed_video_ids.

    Args:
        checkpoint: dict, loaded from stage_2_checkpoint.json
        bucket: str, bucket name for error messages

    Returns:
        list: completed_video_ids to process

    Raises:
        ValueError: if checkpoint schema is invalid

    Source: QA Q9
    """
    # Strict schema validation
    required_fields = ['stage', 'bucket', 'completed_video_ids', 'status', 'total_videos']
    missing = [f for f in required_fields if f not in checkpoint]
    if missing:
        raise ValueError(
            f"Checkpoint for {bucket} has invalid schema (missing {missing}). "
            f"Re-run Stage 2 to regenerate checkpoint."
        )

    # Validate field types
    if not isinstance(checkpoint['completed_video_ids'], list):
        raise ValueError(f"Checkpoint for {bucket}: 'completed_video_ids' must be list")

    # Allow partial completion
    if checkpoint['status'] != 'completed':
        logger.warning(
            f"Checkpoint for {bucket} status is '{checkpoint['status']}' (not 'completed'). "
            f"Processing {len(checkpoint['completed_video_ids'])}/{checkpoint['total_videos']} videos."
        )

    # Handle zero completions gracefully
    if len(checkpoint['completed_video_ids']) == 0:
        logger.info(f"Bucket {bucket} has 0 completed videos. Skipping bucket.")
        return []

    return checkpoint['completed_video_ids']
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Checkpoint missing | Fail-fast with FileNotFoundError | Stage 2 didn't complete for this bucket |
| Checkpoint status="paused" | Warning log, process partial | Allow organizing partial results |
| Zero completed videos | Info log, skip bucket | No files to organize for this bucket |
| Invalid schema | Fail-fast with ValueError | Cannot trust corrupted checkpoint |

#### Step 2.3.3: Detect Duplicate Video IDs

**Purpose**: Ensure each video appears in exactly one bucket (data integrity check)

**Logic**:
```python
def detect_duplicates_across_buckets(files_to_process):
    """
    Detect if same video_id appears in multiple buckets.

    Args:
        files_to_process: list of dict with keys: video_id, bucket, source_path

    Raises:
        ValueError: if duplicate video_id detected

    Source: QA Q11
    """
    video_id_to_buckets = {}

    for file_info in files_to_process:
        video_id = file_info['video_id']
        bucket = file_info['bucket']

        if video_id in video_id_to_buckets:
            previous_bucket = video_id_to_buckets[video_id]
            raise ValueError(
                f"Video ID '{video_id}' appears in multiple buckets:\n"
                f"  - Bucket: {previous_bucket}\n"
                f"  - Bucket: {bucket}\n\n"
                f"This indicates checkpoint corruption or Stage 2 bug.\n"
                f"Each video should belong to exactly one bucket based on duration.\n\n"
                f"Solutions:\n"
                f"  1. Re-run Stage 2 to regenerate checkpoints\n"
                f"  2. Manually inspect checkpoints and remove duplicate entries"
            )

        video_id_to_buckets[video_id] = bucket

    logger.info(f"Validation passed: {len(video_id_to_buckets)} unique videos")
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Duplicate detected | Fail-fast with detailed error | Data corruption - must investigate |
| All videos unique | Continue processing | Normal operation |

#### Step 2.3.4: Organize Files with Detection-Based Resume

**Purpose**: Move files from flat /insights/ directory to bucket directories with automatic resume detection

**Logic**:
```python
def organize_files_with_detection(files_to_process):
    """
    Organize files with automatic resume detection (no checkpoint needed).

    Args:
        files_to_process: list of dict with keys: video_id, bucket, source_path, target_path

    Returns:
        dict: Summary statistics (moved_count, skipped_count, missing_count)

    Source: QA Q6, Q12
    """
    moved_count = 0
    skipped_already_organized = 0
    missing_count = 0

    for file_info in files_to_process:
        source = file_info['source_path']
        target = file_info['target_path']
        video_id = file_info['video_id']
        bucket = file_info['bucket']

        source_exists = os.path.exists(source)
        target_exists = os.path.exists(target)

        # Case 1: Already moved in previous run
        if target_exists and not source_exists:
            logger.debug(f"Already organized: {video_id} → {bucket}")
            skipped_already_organized += 1
            continue

        # Case 2: Missing entirely
        if not source_exists and not target_exists:
            logger.warning(
                f"Missing source and target for video {video_id}. "
                f"Stage 2 checkpoint indicated completion, but file doesn't exist."
            )
            missing_count += 1
            continue

        # Case 3: Source exists (move it)
        try:
            # Ensure target directory exists
            target_dir = os.path.dirname(target)
            os.makedirs(target_dir, exist_ok=True)

            # Move file (atomic within same filesystem)
            shutil.move(source, target)
            moved_count += 1

            logger.info(f"Moved: {video_id} → {bucket} ({moved_count}/{len(files_to_process)})")

        except Exception as e:
            logger.error(f"Failed to move {video_id}: {e}")
            # Continue processing other files (non-fatal)
            continue

    # Summary
    total_processed = moved_count + skipped_already_organized + missing_count
    logger.info(
        f"\nOrganization complete:\n"
        f"  Total files:  {len(files_to_process)}\n"
        f"  Moved:        {moved_count}\n"
        f"  Already done: {skipped_already_organized}\n"
        f"  Missing:      {missing_count}\n"
        f"  Processed:    {total_processed}/{len(files_to_process)}"
    )

    if missing_count > 0:
        logger.warning(f"{missing_count} files missing despite checkpoint indicating completion.")

    return {
        'moved_count': moved_count,
        'skipped_already_organized': skipped_already_organized,
        'missing_count': missing_count,
        'total_processed': total_processed
    }
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Target exists, source missing | Skip with debug log | Already organized in previous run |
| Both exist | Re-move (overwrites target) | Handles interrupted moves |
| Neither exists | Warning log, skip | Checkpoint/file mismatch |
| Move failure (permissions/disk) | Error log, continue | Non-fatal, process other files |

---

## 3. Dependencies & Integration

<!-- PURPOSE: Explicit contracts with other stages. TI generator uses this for imports and validation. -->

### 3.1 Input Dependencies

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **Foundation setup** | FoundationCHILD.md (implemented in Foundation TI) | Directory structure + config.json | analysis_base path, bucket definitions | Fail-fast if analysis_base doesn't exist |
| winner_analysis.json | Stage 1.3 (Winner Analysis) | JSON | `top_3_buckets` (list of strings) | Fail-fast with FileNotFoundError if missing (QA Q10) |
| stage_2_checkpoint.json | Stage 2 (Video Processing) per bucket | JSON | `stage`, `bucket`, `completed_video_ids`, `status`, `total_videos` | Fail-fast with ValueError if invalid schema (QA Q9) |
| temporal_windows_updated.json | Stage 2 output (flat directory) | JSON (1 per video) | `metadata.duration` field | Warning if file missing, skip gracefully (QA Q12) |
| Bucket directories | Stage 2 (created during initialization) | Directory structure | `{analysis_base}/buckets/bucket_{bucket}/` exists | Create subdirectories if missing (QA Q7) |

### 3.2 Output Contracts

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| Organized temporal_windows files | JSON (1 per video) | Same as input (no modification) | Stage 3 (Feature Aggregation) | Assert file exists at target location |
| Empty /insights/ directory | Directory state | All processed files removed | Indicates completion | Optional cleanup check |
| Organization summary log | Log output | moved_count, skipped_count, missing_count | Monitoring, debugging | None (informational) |

### 3.3 Cross-Stage Dependencies

**This feature depends on**:
- **Stage 1.3 (Winner Analysis)**: Must complete successfully to produce winner_analysis.json
- **Stage 2 (Video Processing)**: Must complete for winning buckets to produce temporal_windows files and checkpoints

**This feature is required by**:
- **Stage 3 (Feature Aggregation)**: Expects temporal_windows files organized by bucket in specific directories

**Failure Impact**:
- If this stage fails: Stage 3 cannot run (files not in expected bucket directories)
- Resume: Idempotent - safe to re-run Stage 2.5, will detect already-organized files

### 3.4 External Dependencies

**Python Libraries**:
```python
import os  # Standard library
import json  # Standard library
import shutil  # Standard library
import logging  # Standard library
```

**File System**:
- Read access: `/home/jorge/rumiaifinal/insights/` (source directory)
- Read access: `{analysis_base}/winner_analysis.json`
- Read access: `{analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json`
- Write access: `{analysis_base}/buckets/bucket_{bucket}/analysis/insights/` (target directories)

**Environment Variables**:
- None (paths are determined from analysis_base parameter)

**External Services**: None (pure file system operations)

---

## 4. Configuration & Parameters

<!-- PURPOSE: All tunable values. TI generator uses this for config parsing and defaults. -->

### 4.1 CLI Parameters

**CLI Invocation**:
```bash
# Stage 2.5 is invoked after Stage 2 completes, passing analysis_base directory
python3 ml_pipeline/stage2_5_organize.py \
  --analysis-base /data/clients/acme/hashtags/nutrition/top_contrastive/
```

**Parameters** (loaded from config.json at analysis_base):

| Parameter | Type | Source | Example | Usage |
|-----------|------|--------|---------|-------|
| `client_id` | str | config.json | "acme" | Construct analysis_base path |
| `analysis_type` | str | config.json | "hashtag" | Construct analysis_base path (pluralized) |
| `target` | str | config.json | "nutrition" | Construct analysis_base path (sanitized) |
| `analysis_mode` | str | config.json | "top" | Construct analysis_base path |
| `selection_strategy` | str | config.json | "contrastive" | Construct analysis_base path |

**Example analysis_base construction**:
```python
analysis_base = f"/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/"
# Result: /data/clients/acme/hashtags/nutrition/top_contrastive/
```

### 4.2 Internal Configuration

```python
# Source directory (flat structure with mixed durations)
SOURCE_DIR = "/home/jorge/rumiaifinal/insights/"

# Bucket names (from FoundationCHILD.md Section 6)
ALL_BUCKETS = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]

# File pattern matching
SOURCE_FILE_PATTERN = "{video_id}_temporal_windows_updated.json"
TARGET_SUBDIR = "analysis/insights/"

# Logging configuration
LOG_LEVEL = "INFO"  # DEBUG for troubleshooting

# Performance tuning
BATCH_SIZE = None  # Process all files in single pass (no batching)
```

---

## 5. Data Schemas

<!-- PURPOSE: Exact data structures. TI generator uses this for validation and type hints. -->

### 5.1 Input Schema

**File 1**: `{analysis_base}/winner_analysis.json`

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `top_3_buckets` | list[str] | Yes | Winning bucket names from Stage 1 | `["18-33s", "33-60s", "13-18s"]` |
| `top_100_distribution` | dict | No | Bucket distribution (informational) | `{"18-33s": 45, "33-60s": 30}` |
| `winner_coverage` | float | No | Percentage of winners in top 3 buckets | `95.0` |

**File 2**: `{analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json`

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `stage` | str | Yes | Stage name | `"video_processing"` |
| `bucket` | str | Yes | Bucket name | `"18-33s"` |
| `completed_video_ids` | list[str] | Yes | Successfully processed video IDs | `["123", "456", "789"]` |
| `status` | str | Yes | Checkpoint status | `"completed"`, `"in_progress"`, `"paused"` |
| `total_videos` | int | Yes | Total videos for this bucket | `100` |
| `failed` | int | No | Count of failed videos | `2` |

**File 3**: `/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json`

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `metadata.duration` | float | Yes | Video duration (seconds) | `25.3` |
| `temporal_windows` | dict | Yes | Feature data (not validated by Stage 2.5) | `{"hook": {...}, "middle_segments": [...]}` |

**Total Input Files**:
- 1 × winner_analysis.json
- 3 × stage_2_checkpoint.json (one per winning bucket)
- N × temporal_windows_updated.json (N = sum of completed_video_ids across winning buckets)

### 5.2 Output Schema

**File**: `{analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json`

**Schema**: Identical to input (files are moved, not modified)

**Output Location Pattern**:
```
/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json
```

**Example**:
```
/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144481_temporal_windows_updated.json
```

---

## 6. Error Handling & Validation

<!-- PURPOSE: All error scenarios. TI generator uses this for try/catch blocks and assertions. -->

### 6.1 Input Validation

```python
def validate_inputs(analysis_base, winning_buckets):
    """
    Validate inputs before starting file organization.

    Args:
        analysis_base: str, path to analysis directory
        winning_buckets: list, bucket names from winner_analysis.json

    Raises:
        ValueError: if validation fails

    Source: QA Q5, Q9, Q10
    """
    # 1. Check analysis_base exists
    if not os.path.exists(analysis_base):
        raise ValueError(
            f"Analysis base directory does not exist: {analysis_base}. "
            f"Did Foundation setup run?"
        )

    # 2. Check winning_buckets is valid
    if not winning_buckets or len(winning_buckets) == 0:
        raise ValueError("No winning buckets provided. Check winner_analysis.json.")

    # Validate bucket names are in expected set
    for bucket in winning_buckets:
        if bucket not in ALL_BUCKETS:
            raise ValueError(f"Invalid bucket name: {bucket}. Expected one of {ALL_BUCKETS}")

    # 3. Check source directory exists
    if not os.path.exists(SOURCE_DIR):
        raise ValueError(
            f"Source directory does not exist: {SOURCE_DIR}. "
            f"Did Stage 2 complete?"
        )

    # 4. Check write permissions to analysis_base
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
| Missing winner_analysis.json | `not os.path.exists(path)` | Fail-fast | `"winner_analysis.json not found at {path}. Did Stage 1 complete?"` | 1 |
| Invalid winner_analysis schema | JSON validation | Fail-fast | `"winner_analysis.json missing 'top_3_buckets' field. Re-run Stage 1."` | 2 |
| Missing checkpoint file | `not os.path.exists(checkpoint_path)` | Fail-fast | `"Checkpoint not found for bucket {bucket}. Did Stage 2 complete?"` | 3 |
| Invalid checkpoint schema | Schema validation | Fail-fast | `"Checkpoint for {bucket} has invalid schema (missing {fields}). Re-run Stage 2."` | 4 |
| Duplicate video_id across buckets | Duplicate detection | Fail-fast | `"Video ID '{id}' appears in multiple buckets. Checkpoint corruption detected."` | 5 |
| Missing source file | `not os.path.exists(source)` | Warning log, skip | `"Missing source for video {id}. Skipping."` | 0 (continue) |
| Move failure (permissions) | Exception during move | Error log, continue | `"Failed to move {id}: {error}. Continuing with other files."` | 0 (continue) |
| Move failure (disk full) | Exception during move | Error log, continue | `"Failed to move {id}: disk full. Free space and re-run."` | 0 (continue) |
| No files to organize | Empty file list | Warning log, exit gracefully | `"No files to organize. All checkpoints have 0 completed videos."` | 0 (success) |

### 6.3 Output Validation

```python
def validate_output(analysis_base, winning_buckets, moved_count):
    """
    Validate output after file organization completes.

    Args:
        analysis_base: str, path to analysis directory
        winning_buckets: list, bucket names processed
        moved_count: int, number of files moved

    Source: QA Q12
    """
    # 1. Check files exist in target locations
    total_organized = 0
    for bucket in winning_buckets:
        target_dir = f"{analysis_base}/buckets/bucket_{bucket}/analysis/insights/"
        if os.path.exists(target_dir):
            organized_count = len([f for f in os.listdir(target_dir) if f.endswith('.json')])
            total_organized += organized_count
            logger.info(f"Bucket {bucket}: {organized_count} files organized")

    # 2. Log summary
    logger.info(f"Total files organized: {total_organized}")
    logger.info(f"Files moved this run: {moved_count}")

    # 3. Warn if mismatch (some files may have been organized in previous run)
    if total_organized < moved_count:
        logger.warning(
            f"Organized count ({total_organized}) less than moved count ({moved_count}). "
            f"Some files may be missing."
        )
```

---

## 7. Performance & Scalability

<!-- PURPOSE: Performance targets and bottlenecks. TI generator uses this for optimization. -->

### 7.1 Performance Targets

- **Throughput**: Organize 300 files in < 5 seconds
- **Memory**: Peak usage < 100 MB (minimal data in memory, streaming approach)
- **Disk I/O**: ~2-3 seconds per 300 files (move operations are fast)
- **CPU**: < 5% average utilization (I/O bound, not CPU bound)

### 7.2 Measured Performance

**Estimated Performance** (based on filesystem operations):

| Metric | N=100 files | N=300 files | Notes |
|--------|-------------|-------------|-------|
| Load winner_analysis.json | < 0.1s | < 0.1s | Small JSON file |
| Load checkpoints (3 buckets) | < 0.3s | < 0.3s | 3 small JSON files |
| File existence checks | 0.5s | 1.5s | Fast filesystem checks |
| File moves | 1-2s | 3-4s | Atomic renames within filesystem |
| Total time | 2-3s | 5-6s | Linear scaling |

### 7.3 Bottlenecks & Mitigations

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| File moves across filesystems | 10x slower | Copy + delete instead of rename | Ensure /insights/ and /data/ on same filesystem | High |
| Sequential file processing | Linear scaling | One file at a time | Acceptable for this operation (simple, fast) | Low |
| Checkpoint loading | < 1s total | 3 JSON file reads | Minimal impact, no optimization needed | Low |

### 7.4 Scalability Limits

- **Max files per run**: 1000 (estimated 10-15 seconds)
- **Max buckets**: 8 (all defined buckets, typically organize 3)
- **Min files per run**: 0 (graceful handling, warning log)
- **Disk space**: Negligible (files moved, not duplicated)

---

## 8. Testing Strategy

<!-- PURPOSE: Test plan. TI generator uses this to create test suite. -->

### 8.1 Unit Tests

- [ ] **Test load_winning_buckets**
  - Valid winner_analysis.json (returns list of 3 buckets)
  - Missing file (raises FileNotFoundError)
  - Invalid schema (raises ValueError)
  - Empty top_3_buckets (raises ValueError)

- [ ] **Test validate_checkpoint**
  - Valid checkpoint (returns completed_video_ids)
  - Missing required fields (raises ValueError)
  - Invalid field types (raises ValueError)
  - Status="paused" (warning log, returns video_ids)
  - Zero completed videos (returns empty list)

- [ ] **Test detect_duplicates_across_buckets**
  - No duplicates (passes without error)
  - Duplicate video_id (raises ValueError with details)

- [ ] **Test organize_files_with_detection**
  - Fresh run (moves all files)
  - Resume after interruption (skips already-organized files)
  - Missing source file (warning log, skips)
  - Move failure (error log, continues with other files)

### 8.2 Integration Tests

- [ ] **End-to-end: Stage 2 → Stage 2.5 → Stage 3**
  - Use real stage_2_checkpoint.json (5 videos, bucket 18-33s)
  - Run file organization
  - Verify files exist in bucket directories
  - Verify Stage 3 can read organized files

- [ ] **Resume after interruption**
  - Organize 10 files, stop after 5
  - Re-run Stage 2.5
  - Verify only remaining 5 files are moved
  - Verify no duplicates created

- [ ] **Error handling**
  - Missing winner_analysis.json (fails with clear message)
  - Missing checkpoint (fails with clear message)
  - Checkpoint corruption (fails with clear message)

### 8.3 Test Data

**File**: `tests/fixtures/winner_analysis_sample.json`

```json
{
  "top_100_distribution": {"18-33s": 45, "33-60s": 30, "13-18s": 20},
  "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
  "winner_coverage": 95.0,
  "scrape_timestamp": "2025-01-28T10:30:00Z"
}
```

**File**: `tests/fixtures/stage_2_checkpoint_18-33s_sample.json`

```json
{
  "stage": "video_processing",
  "bucket": "18-33s",
  "total_videos": 5,
  "completed": 5,
  "failed": 0,
  "remaining": 0,
  "status": "completed",
  "completed_video_ids": ["7428596413707144481", "7428596413707144482", "7428596413707144483"],
  "failed_video_ids": []
}
```

### 8.4 Test Execution

```bash
# Run unit tests
pytest tests/test_file_organization.py -v

# Run integration tests
pytest tests/test_stage2_5_integration.py -v

# Run with coverage
pytest --cov=file_organization --cov-report=html

# Test resume behavior
pytest tests/test_file_organization.py::test_resume_after_interruption -v
```

---

## 9. Future Enhancements

<!-- PURPOSE: Planned improvements. TI generator ignores this section (not for current implementation). -->

### 9.1 Planned Improvements

- **Phase 2: Parallel file moves**
  - Current: Sequential file moves (one at a time)
  - Future: Move files in parallel using ThreadPoolExecutor
  - Impact: 2-3x speedup for large file counts (N > 500)
  - Tradeoff: Increased complexity, potential race conditions

- **Phase 3: Verification mode**
  - Current: No post-move verification
  - Future: Add `--verify` flag to check file integrity after moves
  - Impact: Catch filesystem corruption or incomplete moves
  - Tradeoff: Additional runtime overhead

### 9.2 Known Limitations

- **No rollback on partial failure**: If process fails mid-way, some files organized, some not (manual cleanup required)
- **Single-threaded**: No parallel processing (acceptable for current scale)
- **No file integrity checks**: Assumes filesystem operations are reliable
- **No progress bar**: User sees individual file logs but no overall progress indicator

---

## 10. References & Related Docs

<!-- PURPOSE: Links to other documentation. TI generator uses this for additional context if needed. -->

### 10.1 Parent Document

- **MLPlanningv2.md - Section 2.5 "File Organization (Bucket Assignment)"**
  - High-level stage overview (lines 1035-1175)
  - Stage position in pipeline
  - Input/output contracts

### 10.2 Mother Document Foundation

- **FoundationCHILD.md** (shared across all stages)
  - Section 2: Client Architecture & Storage - Provides directory path templates used in this stage
  - Section 5.1: Configuration Schemas - config.json schema for extracting analysis parameters
  - Section 5.3: Checkpoint Schema - stage_2_checkpoint.json schema for loading completed video IDs
  - Section 6: Bucket Definitions - 8 bucket duration ranges and assignment logic

**Key Sections Referenced in This Stage**:
- Section 2.2: Path Templates - Used for constructing analysis_base and target paths
- Section 5.3: Checkpoint Schema - Used for validating Stage 2 checkpoints
- Section 6.1: Bucket Assignment Logic - Used for determining bucket from duration (if needed)

### 10.3 Related Child Docs

- **VideoDiscoveryCHILD.md** (Stage 1)
  - Produces winner_analysis.json (input to this stage)
  - Defines WinnerAnalysisSchema (Section 5.3)

- **VideoProcessingCHILD.md** (Stage 2)
  - Produces temporal_windows_updated.json files (input to this stage)
  - Produces stage_2_checkpoint.json (input to this stage)
  - Creates bucket directories that this stage organizes into

- **FeatureAggregationCHILD.md** (Stage 3)
  - Consumes organized temporal_windows files (output from this stage)
  - Expects files in bucket-specific directories

### 10.4 External References

- **Python shutil documentation**: https://docs.python.org/3/library/shutil.html#shutil.move
- **Python os documentation**: https://docs.python.org/3/library/os.html

---

## Appendix A: Decision Log

<!-- PURPOSE: Document key design decisions with rationale. -->

**Decision 1**: Use checkpoint-driven file discovery (read completed_video_ids from Stage 2 checkpoints)
- **Context**: Need to determine which files to organize from flat /insights/ directory
- **Alternatives Considered**:
  - Option A: Glob pattern matching (process all *_temporal_windows_updated.json files)
  - Option B: Read from Stage 2 checkpoints (selected)
  - Option C: Read duration from each JSON file
- **Rationale**: Checkpoint-driven approach guarantees we only process files that Stage 2 successfully completed, automatically excludes failed videos, and provides clear lineage (checkpoint → video_id → file)
- **Trade-offs**: Requires reading 3 checkpoint files (minimal overhead), but provides exact list of files to process
- **Date**: 2025-01-13
- **Source**: QA Q8

**Decision 2**: Use MOVE operation (not COPY) for file organization
- **Context**: Need to decide whether to move or copy files from /insights/ to bucket directories
- **Alternatives Considered**:
  - Option A: MOVE (delete from source) (selected)
  - Option B: COPY (keep in source)
  - Option C: MOVE with checkpoint/rollback
- **Rationale**: Stage 2.5 is a one-time organizational step, files are already processed, no need to preserve in /insights/, prevents storage bloat, aligns with original design intent
- **Trade-offs**: Cannot re-run Stage 2.5 without re-running Stage 2 (intentional - maintains pipeline consistency)
- **Date**: 2025-01-13
- **Source**: QA Q6

**Decision 3**: Use detection-based resume (no checkpoint for Stage 2.5)
- **Context**: Need to support resume after interruption (power loss, Ctrl+C, crash)
- **Alternatives Considered**:
  - Option A: No resume support (re-run from scratch)
  - Option B: Checkpoint-based resume (write checkpoint after each file)
  - Option C: Detection-based resume (check filesystem state) (selected)
- **Rationale**: Detection-based resume makes Stage 2.5 idempotent (safe to re-run), no checkpoint file to manage, simple logic (check if target exists), Stage 2.5 is fast enough that re-running isn't expensive
- **Trade-offs**: No progress tracking during run, but simplicity outweighs this limitation
- **Date**: 2025-01-13
- **Source**: QA Q12

**Decision 4**: Fail-fast on duplicate video_ids across buckets
- **Context**: Need to handle scenario where same video_id appears in multiple bucket checkpoints
- **Alternatives Considered**:
  - Option A: Fail-fast on duplicate detection (selected)
  - Option B: Use first occurrence, skip duplicates silently
  - Option C: Resolve by reading duration from file
- **Rationale**: Duplicates indicate serious data corruption or Stage 2 bug, should never happen in normal operation (each video has one duration → one bucket), fail-fast forces investigation of root cause
- **Trade-offs**: No automatic recovery, but enforces critical data integrity constraint
- **Date**: 2025-01-13
- **Source**: QA Q11

**Decision 5**: Skip non-winning bucket videos with warning (not fail-fast)
- **Context**: Need to handle videos from non-winning buckets that shouldn't have been processed
- **Alternatives Considered**:
  - Option A: Skip with warning (selected)
  - Option B: Fail-fast on non-winning bucket detection
  - Option C: Process all buckets regardless
- **Rationale**: Graceful degradation allows pipeline to continue, indicates potential upstream issue but allows recovery, leaves file in original location for troubleshooting
- **Trade-offs**: Non-fatal validation (potential for accumulating skipped files), but prevents halting entire pipeline
- **Date**: 2025-01-13
- **Source**: QA Q4

---

## Appendix B: Example Data

### B.1 Sample winner_analysis.json

**File**: `{analysis_base}/winner_analysis.json`

```json
{
  "top_100_distribution": {
    "18-33s": 45,
    "33-60s": 30,
    "13-18s": 20,
    "60-90s": 3,
    "9-13s": 2
  },
  "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
  "winner_coverage": 95.0,
  "scrape_timestamp": "2025-01-28T10:30:00Z",
  "analysis_date": "2025-01-28T10:32:15Z"
}
```

### B.2 Sample stage_2_checkpoint.json

**File**: `{analysis_base}/buckets/bucket_18-33s/checkpoints/stage_2_checkpoint.json`

```json
{
  "stage": "video_processing",
  "bucket": "18-33s",
  "total_videos": 100,
  "completed": 98,
  "failed": 2,
  "remaining": 0,
  "status": "completed",
  "last_checkpoint": "2025-01-28T14:32:15Z",
  "completed_video_ids": [
    "7428596413707144481",
    "7428596413707144482",
    "7428596413707144483"
  ],
  "failed_video_ids": [
    {
      "video_id": "321",
      "error": "FEAT timeout after 120s",
      "error_type": "TimeoutError",
      "timestamp": "2025-01-28T12:15:00Z"
    }
  ]
}
```

### B.3 Sample File Organization Flow

**Before Stage 2.5**:
```
/home/jorge/rumiaifinal/insights/
├── 7428596413707144481_temporal_windows_updated.json  (duration: 25.3s → bucket 18-33s)
├── 7428596413707144482_temporal_windows_updated.json  (duration: 45.1s → bucket 33-60s)
├── 7428596413707144483_temporal_windows_updated.json  (duration: 15.8s → bucket 13-18s)
└── ... (N files, mixed durations)
```

**After Stage 2.5**:
```
/home/jorge/rumiaifinal/insights/
└── (empty - all files moved)

/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/
├── bucket_18-33s/analysis/insights/
│   └── 7428596413707144481_temporal_windows_updated.json
├── bucket_33-60s/analysis/insights/
│   └── 7428596413707144482_temporal_windows_updated.json
└── bucket_13-18s/analysis/insights/
    └── 7428596413707144483_temporal_windows_updated.json
```

---

## Document Metadata

**Creation Date**: 2025-01-13
**Last Modified**: 2025-01-13
**Authors**: RumiAI Team
**Reviewers**: [Pending]
**Approved By**: [Pending]
**Next Review Date**: [Pending]

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-13 | RumiAI Team | Initial draft from Phase 2 QA and Phase 3 generation |
