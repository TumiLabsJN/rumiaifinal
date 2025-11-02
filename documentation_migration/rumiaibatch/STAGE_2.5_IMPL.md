# STAGE_2.5_IMPL.md - File Organization

**Version**: 1.0.0
**Last Updated**: 2025-11-02
**Purpose**: Implementation guide for Stage 2.5: File Organization
**Target Audience**: LLM agents debugging, modifying, or extending Stage 2.5

**Related**: [PRODUCTION_FLOW.md Stage 2.5 Contract](PRODUCTION_FLOW.md#stage-25-file-organization)

---

## Quick Reference

### Entry Points

**Main Entry**: `stage_2_5_file_organization_main()` at [`ml_pipeline/stage2_5_organize/main.py:24-103`](ml_pipeline/stage2_5_organize/main.py#L24-L103)

**Orchestrator Call**: [`rumiai_ml_batch.py:820-853`](rumiai_ml_batch.py#L820-L853)
```python
# Lines 833-835
organization_summary = stage_2_5_file_organization_main(
    analysis_base=str(analysis_base)
)
```

### Key Characteristics

- **Duration**: 1-5 seconds (file moves only, no processing)
- **Processing Mode**: Sequential file moves with detection-based resume
- **No Checkpoint File**: Uses file existence detection for resume
- **Error Strategy**: Skip-on-fail for individual files, exit on validation errors
- **Critical Output**: `selection_manifest.json` (required by Stages 2.6, 2.7, 8)
- **Files Per Video**: 3 (temporal_windows, video MP4, unified_analysis)

### Module Structure

```
ml_pipeline/stage2_5_organize/          (658 total lines)
├── main.py                   (103)  # Main orchestrator
├── file_organizer.py         (452)  # Core file organization logic
├── validation.py              (92)  # Input/output validation
└── __init__.py                (11)  # Exports
```

---

## Table of Contents

1. [Overview](#overview)
2. [Input Contract](#input-contract)
3. [Output Contract](#output-contract)
4. [Core Functions](#core-functions)
5. [Data Flow](#data-flow)
6. [Resume Strategy](#resume-strategy-detection-based)
7. [Error Handling](#error-handling)
8. [Debugging Guide](#debugging-guide)
9. [Modification Guide](#modification-guide)
10. [Related Documentation](#related-documentation)

---

## Overview

**Stage 2.5** organizes files from Stage 2's hardcoded flat directories into bucket-specific directories and creates `selection_manifest.json` for downstream stages.

### Purpose

**Primary Goal**: Move 3 file types per video from global flat directories to bucket-specific directories

**Secondary Goal**: Create `selection_manifest.json` with top/bottom performer split for Stages 2.6, 2.7, and 8

### Why Stage 2.5 Exists

**Problem**: Stage 2 (rumiai_runner.py) outputs to **HARDCODED** flat directories:
- `/home/jorge/rumiaifinal/insights/` - temporal_windows_updated.json
- `/home/jorge/rumiaifinal/temp/` - Video MP4 files
- `/home/jorge/rumiaifinal/unified_analysis/` - Unified analysis JSON

**Solution**: Stage 2.5 organizes these files into bucket-specific directories for downstream stages

### Processing Flow (7 Steps)

**Step 1**: Load winning buckets from `winner_analysis.json`
**Step 2**: Validate inputs (directories exist, permissions)
**Step 3**: Build file list from Stage 2 checkpoints (3 files per completed video)
**Step 4**: Detect duplicate video IDs across buckets (validation)
**Step 5**: Organize files with detection-based resume
**Step 6**: Validate outputs (verify files in target locations)
**Step 7**: Create `selection_manifest.json` for Stage 2.6, 2.7, 8

---

## Input Contract

### Prerequisites

**Stage 1 Outputs** (Required):
- `winner_analysis.json` - Top 3 winning buckets

**Stage 2 Outputs** (Required):
- `buckets/bucket_{name}/checkpoints/stage_2_checkpoint.json` - Per bucket
- `buckets/bucket_{name}/selected_videos.json` - Per bucket
- `/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json` - Per video
- `/home/jorge/rumiaifinal/temp/{video_id}.mp4` - Per video (optional)
- `/home/jorge/rumiaifinal/unified_analysis/{video_id}.json` - Per video (optional)

### Required Inputs

**1. analysis_base** (from Foundation):
```python
analysis_base = "/data/clients/acme_corp/hashtags/nutrition/top_contrastive/"
```

**Format**: Absolute path to analysis directory (string, not Path object)

**Example**:
```python
stage_2_5_file_organization_main(
    analysis_base="/data/clients/acme_corp/hashtags/nutrition/top_contrastive/"
)
```

### Input Validation

**Location**: `validation.py:18-61`

**Validation Rules**:
1. `analysis_base` directory must exist
2. `winning_buckets` must be non-empty list
3. All bucket names must be in `ALL_BUCKETS` (8 valid buckets)
4. Source directory `/home/jorge/rumiaifinal/insights/` must exist
5. Must have write permissions to `analysis_base`

**Implementation** (validation.py:31-60):
```python
# Validation 1: Check analysis_base exists
if not os.path.exists(analysis_base):
    raise ValueError(f"Analysis base directory does not exist: {analysis_base}")

# Validation 2: Check winning_buckets is valid
if not winning_buckets or len(winning_buckets) == 0:
    raise ValueError("No winning buckets provided. Check winner_analysis.json.")

# Validation 3: Validate bucket names
for bucket in winning_buckets:
    if bucket not in ALL_BUCKETS:
        raise ValueError(f"Invalid bucket name: {bucket}")

# Validation 4: Check source directory exists
SOURCE_DIR = "/home/jorge/rumiaifinal/insights/"
if not os.path.exists(SOURCE_DIR):
    raise ValueError(f"Source directory does not exist: {SOURCE_DIR}")

# Validation 5: Check write permissions
test_file = f"{analysis_base}/test_write.tmp"
try:
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
except Exception as e:
    raise ValueError(f"No write permission to {analysis_base}: {e}")
```

---

## Output Contract

### Output Files

**1. Organized Files** (3 per video)

**Target Locations**:
```
{analysis_base}/buckets/bucket_{name}/
├── analysis/insights/{video_id}_temporal_windows_updated.json
├── analysis/unified/{video_id}.json
└── videos/{video_id}.mp4
```

**Example** (100 videos in bucket "18-33s"):
```
/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/
├── analysis/insights/
│   ├── 7123456789_temporal_windows_updated.json
│   ├── 7123456790_temporal_windows_updated.json
│   └── ... (100 files)
├── analysis/unified/
│   ├── 7123456789.json
│   ├── 7123456790.json
│   └── ... (100 files)
└── videos/
    ├── 7123456789.mp4
    ├── 7123456790.mp4
    └── ... (100 files)
```

**2. selection_manifest.json** ⚠️ **CRITICAL OUTPUT**

**Path**: `{analysis_base}/selection_manifest.json`

**Schema** (file_organizer.py:353-363):
```json
{
  "hashtag": "nutrition",
  "selected_buckets": ["18-33s", "33-60s", "13-18s"],
  "videos_by_bucket": {
    "18-33s": {
      "top_performers": [
        "7123456789012345678",
        "7123456789012345679",
        "..."
      ],
      "bottom_performers": [
        "7123456789012345680",
        "7123456789012345681",
        "..."
      ]
    },
    "33-60s": {
      "top_performers": [...],
      "bottom_performers": [...]
    },
    "13-18s": {
      "top_performers": [...],
      "bottom_performers": [...]
    }
  }
}
```

**Purpose**:
- **Stage 2.6 (Content Discovery)**: Needs video list for taxonomy generation
- **Stage 2.7 (Content Classification)**: Needs top/bottom split for classification
- **Stage 8 (Report Generation)**: Needs top/bottom split for contrastive analysis

**Top/Bottom Split Logic** (file_organizer.py:413-419):
```python
# Preserve top/bottom distinction from Stage 1 selection
all_video_ids = [v['id'] for v in selected['videos']]
top_count = selected['top_count']  # From selected_videos.json

top_intended = all_video_ids[:top_count]      # First 80 videos (80%)
bottom_intended = all_video_ids[top_count:]   # Last 20 videos (20%)

# Filter by completion status (from checkpoint)
completed_ids = set(checkpoint['completed_video_ids'])
top_completed = [vid for vid in top_intended if vid in completed_ids]
bottom_completed = [vid for vid in bottom_intended if vid in completed_ids]
```

**3. Organization Summary** (Return Value)

**Schema**:
```python
{
  "moved_count": 240,                 # Files actually moved this run
  "skipped_already_organized": 60,    # Already moved in previous run
  "missing_count": 0,                 # Missing files (warns if > 0)
  "total_processed": 300,             # Sum of above
  "winning_buckets": ["18-33s", "33-60s", "13-18s"]
}
```

**Example Output**:
```python
{
  "moved_count": 240,           # 80 videos × 3 files = 240
  "skipped_already_organized": 60,  # 20 videos × 3 files = 60 (resume scenario)
  "missing_count": 0,
  "total_processed": 300,       # 100 videos × 3 files = 300
  "winning_buckets": ["18-33s", "33-60s", "13-18s"]
}
```

---

## Core Functions

### 1. stage_2_5_file_organization_main() - Orchestrator

**Location**: `ml_pipeline/stage2_5_organize/main.py:24-103`

**Purpose**: Main orchestration function for Stage 2.5

**Function Signature**:
```python
def stage_2_5_file_organization_main(analysis_base: str) -> Dict[str, any]:
```

**Returns**:
```python
{
  "moved_count": int,
  "skipped_already_organized": int,
  "missing_count": int,
  "total_processed": int,
  "winning_buckets": list[str]
}
```

**Implementation**:
```python
# Step 1: Load winning buckets from winner_analysis.json
winning_buckets = load_winning_buckets(analysis_base)

# Step 2: Validate inputs before processing
validate_inputs(analysis_base, winning_buckets)

# Step 3: Build file list from Stage 2 checkpoints
files_to_process = build_file_list(analysis_base, winning_buckets)

# Handle empty file list gracefully
if len(files_to_process) == 0:
    logger.warning("No files to organize. All checkpoints have 0 completed videos.")
    return {
        "moved_count": 0,
        "skipped_already_organized": 0,
        "missing_count": 0,
        "total_processed": 0,
        "winning_buckets": winning_buckets
    }

# Step 4: Detect duplicate video IDs across buckets
detect_duplicates_across_buckets(files_to_process)

# Step 5: Organize files with detection-based resume
summary = organize_files_with_detection(files_to_process)

# Step 6: Validate outputs
validate_output(analysis_base, winning_buckets, summary['moved_count'])

# Step 7: Create selection manifest for Stage 2.6
create_selection_manifest(analysis_base, winning_buckets)

# Step 8: Return summary with winning buckets
summary['winning_buckets'] = winning_buckets
return summary
```

---

### 2. load_winning_buckets() - Load Winner Analysis

**Location**: `ml_pipeline/stage2_5_organize/file_organizer.py:28-81`

**Purpose**: Load winning bucket names from Stage 1 output

**Implementation**:
```python
def load_winning_buckets(analysis_base: str) -> List[str]:
    """Load winning buckets from winner_analysis.json."""

    # Step 1: Construct path
    winner_analysis_path = f"{analysis_base}/winner_analysis.json"

    # Step 2: Validate file exists
    if not os.path.exists(winner_analysis_path):
        raise FileNotFoundError(
            f"winner_analysis.json not found at:\n"
            f"  {winner_analysis_path}\n\n"
            f"This file is created by Stage 1.3 (Winner Analysis).\n"
            f"Solutions:\n"
            f"  1. Complete Stage 1 (Video Discovery & Winner Analysis)\n"
            f"  2. Check if Stage 1 completed successfully\n"
            f"  3. Verify analysis_base path is correct: {analysis_base}"
        )

    # Step 3: Load JSON file
    with open(winner_analysis_path) as f:
        winner_analysis = json.load(f)

    # Step 4: Validate schema - check 'top_3_buckets' field exists
    if 'top_3_buckets' not in winner_analysis:
        raise ValueError(f"winner_analysis.json missing 'top_3_buckets' field")

    # Step 5: Validate 'top_3_buckets' is a list
    if not isinstance(winner_analysis['top_3_buckets'], list):
        raise TypeError(
            f"'top_3_buckets' must be list, got {type(winner_analysis['top_3_buckets'])}"
        )

    # Step 6: Validate list is not empty
    if len(winner_analysis['top_3_buckets']) == 0:
        raise ValueError("'top_3_buckets' is empty - no winning buckets identified")

    # Step 7: Log success and return
    logger.info(f"Loaded winning buckets: {winner_analysis['top_3_buckets']}")
    return winner_analysis['top_3_buckets']
```

---

### 3. build_file_list() - Build File List from Checkpoints

**Location**: `ml_pipeline/stage2_5_organize/file_organizer.py:130-200`

**Purpose**: Build list of files to organize from Stage 2 checkpoints

**Returns**:
```python
[
  {
    'video_id': '7123456789',
    'bucket': '18-33s',
    'file_type': 'temporal_windows',
    'source_path': '/home/jorge/rumiaifinal/insights/7123456789_temporal_windows_updated.json',
    'target_path': '{analysis_base}/buckets/bucket_18-33s/analysis/insights/7123456789_temporal_windows_updated.json'
  },
  {
    'video_id': '7123456789',
    'bucket': '18-33s',
    'file_type': 'video',
    'source_path': '/home/jorge/rumiaifinal/temp/7123456789.mp4',
    'target_path': '{analysis_base}/buckets/bucket_18-33s/videos/7123456789.mp4'
  },
  {
    'video_id': '7123456789',
    'bucket': '18-33s',
    'file_type': 'unified_analysis',
    'source_path': '/home/jorge/rumiaifinal/unified_analysis/7123456789.json',
    'target_path': '{analysis_base}/buckets/bucket_18-33s/analysis/unified/7123456789.json'
  }
]
```

**Implementation**:
```python
def build_file_list(analysis_base: str, winning_buckets: List[str]) -> List[Dict[str, str]]:
    """Build list of files to organize from Stage 2 checkpoints."""

    files_to_process = []

    # Step 1: Iterate through each winning bucket
    for bucket in winning_buckets:
        # Step 2: Construct checkpoint path
        checkpoint_path = f"{analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json"

        # Step 3: Validate checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found for bucket {bucket}. Did Stage 2 complete?"
            )

        # Step 4: Load checkpoint JSON
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Step 5: Validate checkpoint and extract video IDs
        video_ids = validate_checkpoint(checkpoint, bucket)

        # Step 6: Skip if no completed videos
        if len(video_ids) == 0:
            logger.info(f"Bucket {bucket} has 0 completed videos. Skipping.")
            continue

        # Step 7: Build file info for each video (3 file types)
        for video_id in video_ids:
            # 7a. Temporal windows JSON (Stage 2 output)
            files_to_process.append({
                'video_id': video_id,
                'bucket': bucket,
                'file_type': 'temporal_windows',
                'source_path': f"{SOURCE_DIRS['temporal_windows']}{video_id}_temporal_windows_updated.json",
                'target_path': f"{analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json"
            })

            # 7b. Video file (MP4)
            files_to_process.append({
                'video_id': video_id,
                'bucket': bucket,
                'file_type': 'video',
                'source_path': f"{SOURCE_DIRS['videos']}{video_id}.mp4",
                'target_path': f"{analysis_base}/buckets/bucket_{bucket}/videos/{video_id}.mp4"
            })

            # 7c. Unified analysis JSON
            files_to_process.append({
                'video_id': video_id,
                'bucket': bucket,
                'file_type': 'unified_analysis',
                'source_path': f"{SOURCE_DIRS['unified_analysis']}{video_id}.json",
                'target_path': f"{analysis_base}/buckets/bucket_{bucket}/analysis/unified/{video_id}.json"
            })

    # Step 8: Log summary
    logger.info(f"Built file list: {len(files_to_process)} files across {len(winning_buckets)} buckets")
    return files_to_process
```

**Helper Function: validate_checkpoint()** (file_organizer.py:84-127):
```python
def validate_checkpoint(checkpoint: Dict[str, Any], bucket: str) -> List[str]:
    """Validate checkpoint schema and extract completed_video_ids."""

    # Step 1: Strict schema validation
    required_fields = ['stage', 'bucket', 'completed_video_ids', 'status', 'total_videos']
    missing = [f for f in required_fields if f not in checkpoint]

    if missing:
        raise ValueError(
            f"Checkpoint for {bucket} has invalid schema (missing {missing}). "
            f"Re-run Stage 2 to regenerate checkpoint."
        )

    # Step 2: Validate field types
    if not isinstance(checkpoint['completed_video_ids'], list):
        raise ValueError(f"Checkpoint for {bucket}: 'completed_video_ids' must be list")

    # Step 3: Allow partial completion - warn if status != "completed"
    if checkpoint['status'] != 'completed':
        logger.warning(
            f"Checkpoint for {bucket} status is '{checkpoint['status']}' (not 'completed'). "
            f"Processing {len(checkpoint['completed_video_ids'])}/{checkpoint['total_videos']} videos."
        )

    # Step 4: Handle zero completions gracefully
    if len(checkpoint['completed_video_ids']) == 0:
        logger.info(f"Bucket {bucket} has 0 completed videos. Skipping bucket.")
        return []

    # Step 5: Return completed video IDs
    return checkpoint['completed_video_ids']
```

---

### 4. detect_duplicates_across_buckets() - Duplicate Detection

**Location**: `ml_pipeline/stage2_5_organize/file_organizer.py:203-250`

**Purpose**: Validate that same video_id doesn't appear in multiple buckets

**Why Important**: Each video should only be in ONE bucket based on its duration

**Implementation**:
```python
def detect_duplicates_across_buckets(files_to_process: List[Dict[str, str]]) -> None:
    """Detect if same video_id appears in multiple buckets."""

    # Step 1: Initialize tracking dictionary - track (video_id, file_type) → bucket
    video_file_to_bucket = {}

    # Step 2: Iterate through all files
    for file_info in files_to_process:
        video_id = file_info['video_id']
        bucket = file_info['bucket']
        file_type = file_info.get('file_type', 'unknown')

        # Create composite key (video_id, file_type)
        composite_key = (video_id, file_type)

        # Step 3: Check if this (video_id, file_type) already seen
        if composite_key in video_file_to_bucket:
            previous_bucket = video_file_to_bucket[composite_key]

            # Step 4: Duplicate detected - raise error
            raise ValueError(
                f"Video ID '{video_id}' (file_type: {file_type}) appears in multiple buckets:\n"
                f"  - Bucket: {previous_bucket}\n"
                f"  - Bucket: {bucket}\n\n"
                f"This indicates checkpoint corruption or Stage 2 bug.\n"
                f"Each video should belong to exactly one bucket based on duration.\n\n"
                f"Solutions:\n"
                f"  1. Re-run Stage 2 to regenerate checkpoints\n"
                f"  2. Manually inspect checkpoints and remove duplicate entries"
            )

        # Step 5: Record mapping
        video_file_to_bucket[composite_key] = bucket

    # Step 6: Log validation success
    unique_videos = len(set(vid for vid, _ in video_file_to_bucket.keys()))
    logger.info(f"Validation passed: {unique_videos} unique videos × 3 file types = {len(video_file_to_bucket)} files")
```

---

### 5. organize_files_with_detection() - File Organization with Resume

**Location**: `ml_pipeline/stage2_5_organize/file_organizer.py:252-338`

**Purpose**: Organize files with automatic resume detection (no checkpoint needed)

**Resume Strategy**: Detection-based (see [Resume Strategy](#resume-strategy-detection-based) section)

**Implementation**:
```python
def organize_files_with_detection(files_to_process: List[Dict[str, str]]) -> Dict[str, int]:
    """Organize files with automatic resume detection."""

    # Step 1: Initialize counters
    moved_count = 0
    skipped_already_organized = 0
    missing_count = 0

    # Step 2: Iterate through each file
    for file_info in files_to_process:
        source = file_info['source_path']
        target = file_info['target_path']
        video_id = file_info['video_id']
        bucket = file_info['bucket']
        file_type = file_info.get('file_type', 'unknown')

        # Step 3: Check file existence states
        source_exists = os.path.exists(source)
        target_exists = os.path.exists(target)

        # Step 4: Case 1 - Already moved in previous run
        if target_exists and not source_exists:
            logger.debug(f"Already organized: {video_id} ({file_type}) → {bucket}")
            skipped_already_organized += 1
            continue

        # Step 5: Case 2 - Missing entirely
        if not source_exists and not target_exists:
            logger.warning(
                f"Missing source and target for video {video_id} ({file_type}). "
                f"Stage 2 checkpoint indicated completion, but file doesn't exist."
            )
            missing_count += 1
            continue

        # Step 6: Case 3 - Source exists (move it)
        try:
            # 6a: Ensure target directory exists
            target_dir = os.path.dirname(target)
            os.makedirs(target_dir, exist_ok=True)

            # 6b: Move file (atomic within same filesystem)
            shutil.move(source, target)
            moved_count += 1

            # 6c: Log success
            logger.info(f"Moved: {video_id} ({file_type}) → {bucket} ({moved_count}/{len(files_to_process)})")

        except Exception as e:
            # 6d: Log error but continue processing other files
            logger.error(f"Failed to move {video_id} ({file_type}): {e}")
            continue

    # Step 7: Calculate summary
    total_processed = moved_count + skipped_already_organized + missing_count

    # Step 8: Log summary
    logger.info(
        f"\nOrganization complete:\n"
        f"  Total files:  {len(files_to_process)}\n"
        f"  Moved:        {moved_count}\n"
        f"  Already done: {skipped_already_organized}\n"
        f"  Missing:      {missing_count}\n"
        f"  Processed:    {total_processed}/{len(files_to_process)}"
    )

    # Step 9: Warn if missing files
    if missing_count > 0:
        logger.warning(f"{missing_count} files missing despite checkpoint indicating completion.")

    # Step 10: Return summary
    return {
        'moved_count': moved_count,
        'skipped_already_organized': skipped_already_organized,
        'missing_count': missing_count,
        'total_processed': total_processed
    }
```

---

### 6. create_selection_manifest() - Create Manifest for Downstream Stages

**Location**: `ml_pipeline/stage2_5_organize/file_organizer.py:341-452`

**Purpose**: Create `selection_manifest.json` for Stages 2.6, 2.7, and 8

**Implementation**:
```python
def create_selection_manifest(analysis_base: str, winning_buckets: List[str]) -> None:
    """Create selection_manifest.json for Stage 2.6 content analysis."""

    # Step 1: Load hashtag from config.json
    config_path = f"{analysis_base}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found at {config_path}. "
            f"Stage 0 must complete before Stage 2.5."
        )

    with open(config_path) as f:
        config = json.load(f)

    hashtag = config['target'].lstrip('#').lstrip('@')  # Handle both hashtag and handle

    # Step 2: Build videos_by_bucket structure
    videos_by_bucket = {}

    for bucket_name in winning_buckets:
        # 2a: Load selected_videos.json (intended selection from Stage 1)
        selected_videos_path = f"{analysis_base}/buckets/bucket_{bucket_name}/selected_videos.json"
        if not os.path.exists(selected_videos_path):
            logger.warning(f"selected_videos.json not found for bucket {bucket_name}, skipping")
            continue

        with open(selected_videos_path) as f:
            selected = json.load(f)

        # 2b: Load checkpoint (actual completions from Stage 2)
        checkpoint_path = f"{analysis_base}/buckets/bucket_{bucket_name}/checkpoints/stage_2_checkpoint.json"
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found for bucket {bucket_name}, skipping")
            continue

        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        completed_ids = set(checkpoint['completed_video_ids'])

        # 2c: Extract video IDs and split by top/bottom
        all_video_ids = [v['id'] for v in selected['videos']]
        top_count = selected['top_count']  # From Stage 1 selection (e.g., 80)

        # Preserve top/bottom distinction based on original selection order
        top_intended = all_video_ids[:top_count]      # First 80 videos
        bottom_intended = all_video_ids[top_count:]   # Last 20 videos

        # Filter by completion status
        top_completed = [vid for vid in top_intended if vid in completed_ids]
        bottom_completed = [vid for vid in bottom_intended if vid in completed_ids]

        videos_by_bucket[bucket_name] = {
            'top_performers': top_completed,
            'bottom_performers': bottom_completed
        }

        logger.info(
            f"Bucket {bucket_name}: "
            f"top {len(top_completed)}/{len(top_intended)}, "
            f"bottom {len(bottom_completed)}/{len(bottom_intended)} completed"
        )

    # Step 3: Create selection manifest
    selection_manifest = {
        'hashtag': hashtag,
        'selected_buckets': winning_buckets,
        'videos_by_bucket': videos_by_bucket
    }

    # Step 4: Save manifest
    manifest_path = f"{analysis_base}/selection_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(selection_manifest, f, indent=2)

    logger.info(f"✓ Created selection_manifest.json: {manifest_path}")

    # Step 5: Log summary statistics
    total_top = sum(len(v['top_performers']) for v in videos_by_bucket.values())
    total_bottom = sum(len(v['bottom_performers']) for v in videos_by_bucket.values())
    logger.info(
        f"Selection manifest contains {total_top} top performers + "
        f"{total_bottom} bottom performers across {len(videos_by_bucket)} buckets"
    )
```

---

## Data Flow

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: analysis_base path                                           │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Load Winning Buckets                                        │
│ ─────────────────────────────────────────────────────────────────  │
│ Read: {analysis_base}/winner_analysis.json (from Stage 1)          │
│ Extract: top_3_buckets = ["18-33s", "33-60s", "13-18s"]           │
│                                                                     │
│ Output: winning_buckets list                                       │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Validate Inputs                                            │
│ ─────────────────────────────────────────────────────────────────  │
│ Check:                                                              │
│   • analysis_base directory exists                                 │
│   • winning_buckets not empty                                      │
│   • All bucket names valid                                         │
│   • Source directory exists (/home/jorge/rumiaifinal/insights/)    │
│   • Write permissions to analysis_base                             │
│                                                                     │
│ Output: Validation passed or raises ValueError                     │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Build File List                                            │
│ ─────────────────────────────────────────────────────────────────  │
│ For each winning bucket:                                           │
│   1. Read stage_2_checkpoint.json                                  │
│   2. Validate checkpoint schema                                    │
│   3. Extract completed_video_ids                                   │
│   4. For each video ID, add 3 files:                               │
│      a. temporal_windows_updated.json                              │
│         Source: /home/jorge/rumiaifinal/insights/{id}_*.json       │
│         Target: {analysis_base}/buckets/bucket_{name}/analysis/    │
│                 insights/{id}_temporal_windows_updated.json        │
│      b. Video MP4                                                  │
│         Source: /home/jorge/rumiaifinal/temp/{id}.mp4              │
│         Target: {analysis_base}/buckets/bucket_{name}/videos/      │
│                 {id}.mp4                                            │
│      c. Unified analysis JSON                                      │
│         Source: /home/jorge/rumiaifinal/unified_analysis/{id}.json │
│         Target: {analysis_base}/buckets/bucket_{name}/analysis/    │
│                 unified/{id}.json                                   │
│                                                                     │
│ Output: files_to_process list (3 files per video)                  │
│         Example: 100 videos × 3 files = 300 files                  │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Detect Duplicates Across Buckets                           │
│ ─────────────────────────────────────────────────────────────────  │
│ For each file in files_to_process:                                 │
│   1. Create composite key: (video_id, file_type)                   │
│   2. Check if already seen in different bucket                     │
│   3. If duplicate found: RAISE ValueError                          │
│      (indicates checkpoint corruption or Stage 2 bug)              │
│                                                                     │
│ Output: Validation passed or raises ValueError                     │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Organize Files with Detection-Based Resume                 │
│ ─────────────────────────────────────────────────────────────────  │
│ For each file in files_to_process:                                 │
│   Check states:                                                    │
│   • source_exists = os.path.exists(source)                         │
│   • target_exists = os.path.exists(target)                         │
│                                                                     │
│   Case 1: target_exists AND NOT source_exists                      │
│     → Already moved in previous run                                │
│     → skipped_already_organized += 1                               │
│     → Continue to next file                                        │
│                                                                     │
│   Case 2: NOT source_exists AND NOT target_exists                  │
│     → Missing entirely (Stage 2 checkpoint says complete but       │
│        file doesn't exist)                                         │
│     → missing_count += 1                                           │
│     → Log warning, continue to next file                           │
│                                                                     │
│   Case 3: source_exists (target may or may not exist)              │
│     → Create target directory (os.makedirs)                        │
│     → Move file (shutil.move)                                      │
│     → moved_count += 1                                             │
│     → Log success, continue to next file                           │
│                                                                     │
│ Output: Summary dict {moved, skipped, missing counts}              │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Validate Output                                            │
│ ─────────────────────────────────────────────────────────────────  │
│ For each winning bucket:                                           │
│   1. Check target directory exists:                                │
│      {analysis_base}/buckets/bucket_{name}/analysis/insights/      │
│   2. Count organized files (*.json files)                          │
│   3. Log per-bucket counts                                         │
│                                                                     │
│ Compare:                                                            │
│   • total_organized (files in target dirs)                         │
│   • moved_count (files moved this run)                             │
│   • Warn if mismatch                                               │
│                                                                     │
│ Output: Validation logged                                          │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7: Create selection_manifest.json                             │
│ ─────────────────────────────────────────────────────────────────  │
│ For each winning bucket:                                           │
│   1. Load config.json → Extract hashtag                            │
│   2. Load selected_videos.json → Get intended selection            │
│      (all_video_ids, top_count, bottom_count)                      │
│   3. Load stage_2_checkpoint.json → Get completed videos           │
│      (completed_video_ids)                                         │
│   4. Split by top/bottom:                                          │
│      top_intended = all_video_ids[:top_count]                      │
│      bottom_intended = all_video_ids[top_count:]                   │
│   5. Filter by completion:                                         │
│      top_completed = [v for v in top_intended if v in completed]   │
│      bottom_completed = [v for v in bottom_intended if v in        │
│                          completed]                                 │
│   6. Store in videos_by_bucket dict                                │
│                                                                     │
│ Create manifest:                                                   │
│   {                                                                 │
│     "hashtag": "nutrition",                                        │
│     "selected_buckets": ["18-33s", "33-60s", "13-18s"],           │
│     "videos_by_bucket": {                                          │
│       "18-33s": {                                                  │
│         "top_performers": [...],                                   │
│         "bottom_performers": [...]                                 │
│       }                                                             │
│     }                                                               │
│   }                                                                 │
│                                                                     │
│ Save: {analysis_base}/selection_manifest.json                      │
│                                                                     │
│ Output: selection_manifest.json created                            │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT:                                                             │
│   • Files organized into bucket directories                         │
│   • selection_manifest.json created (CRITICAL for Stages 2.6, 2.7, 8)│
│   • Organization summary returned                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Resume Strategy (Detection-Based)

### No Checkpoint File

**Unlike Stage 2**, Stage 2.5 does NOT use a checkpoint file. Instead, it uses **file existence detection** for automatic resume.

### Detection Logic

**Location**: `file_organizer.py:278-313`

```python
source_exists = os.path.exists(source)
target_exists = os.path.exists(target)

# Case 1: Already moved
if target_exists and not source_exists:
    skipped_already_organized += 1
    continue

# Case 2: Missing entirely
if not source_exists and not target_exists:
    missing_count += 1
    logger.warning(f"Missing: {video_id}")
    continue

# Case 3: Source exists - move it
shutil.move(source, target)
moved_count += 1
```

### Resume Scenarios

**Scenario 1: Fresh Run (First Time)**

```
State:
  Source: EXISTS (/home/jorge/rumiaifinal/insights/7123456789_*.json)
  Target: NOT EXISTS ({bucket}/analysis/insights/7123456789_*.json)

Action:
  → Move file from source to target
  → moved_count += 1

Result:
  Source: NOT EXISTS (moved away)
  Target: EXISTS
```

**Scenario 2: Resume After Interruption**

```
State (files 1-50 already moved, 51-100 remaining):

File 1:
  Source: NOT EXISTS (already moved)
  Target: EXISTS
  → Action: Skip (skipped_already_organized += 1)

File 51:
  Source: EXISTS (not moved yet)
  Target: NOT EXISTS
  → Action: Move (moved_count += 1)
```

**Scenario 3: Missing File (Stage 2 bug)**

```
State:
  Source: NOT EXISTS (never created by Stage 2)
  Target: NOT EXISTS

Action:
  → missing_count += 1
  → Log warning
  → Continue to next file

Note: This indicates Stage 2 checkpoint said "completed" but file doesn't exist
```

### Advantages of Detection-Based Resume

✅ **No checkpoint corruption risk** - No checkpoint file to corrupt
✅ **Automatic resume** - Just re-run, automatically skips completed files
✅ **Idempotent** - Safe to run multiple times
✅ **Simple** - No checkpoint validation logic needed

### Disadvantages

⚠️ **Cannot track progress** - No way to see "50/300 files moved" mid-run
⚠️ **Cannot detect partial moves** - If process killed mid-shutil.move, file may be in inconsistent state

---

## Error Handling

### Error Matrix

| Error Type | Exception | Location | Strategy | Recovery |
|------------|-----------|----------|----------|----------|
| **winner_analysis.json missing** | FileNotFoundError | file_organizer.py:49 | Exit pipeline | Complete Stage 1 |
| **Invalid bucket name** | ValueError | validation.py:45 | Exit pipeline | Check winner_analysis.json |
| **Checkpoint missing** | FileNotFoundError | file_organizer.py:153 | Exit pipeline | Complete Stage 2 |
| **Checkpoint corrupt** | ValueError | file_organizer.py:105 | Exit pipeline | Re-run Stage 2 |
| **Duplicate video across buckets** | ValueError | file_organizer.py:233 | Exit pipeline | Re-run Stage 2 or fix checkpoints |
| **Source directory missing** | ValueError | validation.py:49 | Exit pipeline | Check Stage 2 completed |
| **No write permission** | ValueError | validation.py:60 | Exit pipeline | Fix permissions |
| **config.json missing** | FileNotFoundError | file_organizer.py:376 | Exit pipeline | Complete Stage 0 |
| **File move failure** | Exception | file_organizer.py:310 | Skip file, continue | Check permissions, disk space |
| **Missing file** | None | file_organizer.py:289 | Warn, continue | Investigate Stage 2 |

### Skip-on-Fail vs Exit-Pipeline

**Skip-on-Fail** (individual file move errors):
- File move failure → Log error, continue with other files
- Missing file → Log warning, increment missing_count, continue

**Exit-Pipeline** (validation errors):
- winner_analysis.json missing → Exit immediately
- Checkpoint missing/corrupt → Exit immediately
- Duplicate detection → Exit immediately
- Permission errors → Exit immediately

### Missing Files Handling

**Detection** (file_organizer.py:288-294):
```python
if not source_exists and not target_exists:
    logger.warning(
        f"Missing source and target for video {video_id} ({file_type}). "
        f"Stage 2 checkpoint indicated completion, but file doesn't exist."
    )
    missing_count += 1
    continue
```

**Behavior**:
- Does NOT fail the pipeline
- Increments `missing_count`
- Logs warning
- Continues processing other files

**Result**:
- Summary includes `"missing_count": N`
- Orchestrator logs warning if `missing_count > 0`
- User can investigate why Stage 2 checkpoint says "completed" but file missing

---

## Debugging Guide

### Common Issues

#### Issue 1: "winner_analysis.json not found"

**Symptom**:
```
FileNotFoundError: winner_analysis.json not found at:
  /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json

This file is created by Stage 1.3 (Winner Analysis).
```

**Cause**: Stage 1 didn't complete or ran with different config

**Debug**:
```bash
# Check if Stage 1 completed
ls -la {analysis_base}/winner_analysis.json

# Check if running with correct analysis_base
echo $analysis_base
```

**Fix**:
```bash
# Run Stage 1 to create winner_analysis.json
python rumiai_ml_batch.py --client acme --target nutrition --analysis-type hashtag
```

**Location**: `file_organizer.py:49`

---

#### Issue 2: "Checkpoint not found for bucket"

**Symptom**:
```
FileNotFoundError: Checkpoint not found for bucket 18-33s. Did Stage 2 complete?
```

**Cause**: Stage 2 didn't complete for this bucket

**Debug**:
```bash
# Check if checkpoint exists
ls -la {analysis_base}/buckets/bucket_18-33s/checkpoints/stage_2_checkpoint.json

# Check if Stage 2 ran for this bucket
grep "bucket_18-33s" logs/rumiai_ml_batch.log
```

**Fix**:
```bash
# Re-run Stage 2 (will auto-resume if partially complete)
python rumiai_ml_batch.py --client acme --target nutrition
```

**Location**: `file_organizer.py:153`

---

#### Issue 3: "Video ID appears in multiple buckets"

**Symptom**:
```
ValueError: Video ID '7123456789' (file_type: temporal_windows) appears in multiple buckets:
  - Bucket: 18-33s
  - Bucket: 33-60s

This indicates checkpoint corruption or Stage 2 bug.
```

**Cause**: Same video ID in multiple bucket checkpoints (should be impossible)

**Debug**:
```bash
# Check which checkpoints contain this video ID
for bucket in 18-33s 33-60s 13-18s; do
  echo "Bucket: $bucket"
  cat {analysis_base}/buckets/bucket_$bucket/checkpoints/stage_2_checkpoint.json | \
    jq '.completed_video_ids | map(select(. == "7123456789"))'
done
```

**Fix**:

**Option 1: Re-run Stage 2** (recommended):
```bash
# Delete all Stage 2 checkpoints
rm {analysis_base}/buckets/bucket_*/checkpoints/stage_2_checkpoint.json

# Re-run Stage 2
python rumiai_ml_batch.py --client acme --target nutrition
```

**Option 2: Manual fix**:
```bash
# Determine correct bucket based on video duration
cat /home/jorge/rumiaifinal/insights/7123456789_temporal_windows_updated.json | jq '.duration'
# Output: 25 (should be in 18-33s bucket)

# Remove from incorrect bucket checkpoint
vi {analysis_base}/buckets/bucket_33-60s/checkpoints/stage_2_checkpoint.json
# Manually remove video ID from completed_video_ids array
```

**Location**: `file_organizer.py:233`

---

#### Issue 4: Missing files despite checkpoint completion

**Symptom**:
```
Organization complete:
  Total files:  300
  Moved:        270
  Already done: 0
  Missing:      30

⚠️  30 files missing despite checkpoint indicating completion.
```

**Cause**: Stage 2 checkpoint says video completed but output file doesn't exist

**Debug**:
```bash
# Find which videos are missing
# Check checkpoint for completed IDs
cat {analysis_base}/buckets/bucket_18-33s/checkpoints/stage_2_checkpoint.json | jq '.completed_video_ids[]'

# Check which files exist in source
ls /home/jorge/rumiaifinal/insights/ | grep temporal_windows_updated.json

# Find missing IDs
comm -23 \
  <(cat {analysis_base}/buckets/bucket_18-33s/checkpoints/stage_2_checkpoint.json | jq -r '.completed_video_ids[]' | sort) \
  <(ls /home/jorge/rumiaifinal/insights/ | sed 's/_temporal_windows_updated.json//' | sort)
```

**Fix**:

**Option 1: Re-process missing videos in Stage 2**:
```bash
# Edit checkpoint to remove missing video IDs from completed_video_ids
vi {analysis_base}/buckets/bucket_18-33s/checkpoints/stage_2_checkpoint.json

# Re-run Stage 2 (will process missing videos only)
python rumiai_ml_batch.py --client acme --target nutrition
```

**Option 2: Accept missing files**:
```bash
# If only a few files missing and acceptable loss, continue pipeline
# selection_manifest.json will only include actually completed videos
```

**Location**: `file_organizer.py:329`

---

#### Issue 5: "No write permission to analysis_base"

**Symptom**:
```
ValueError: No write permission to /data/clients/acme/hashtags/nutrition/top_contrastive: [Errno 13] Permission denied
```

**Cause**: Insufficient write permissions to analysis_base directory

**Debug**:
```bash
# Check permissions
ls -ld {analysis_base}

# Check ownership
stat {analysis_base}

# Try to create test file
touch {analysis_base}/test_write.tmp
```

**Fix**:
```bash
# Fix permissions
chmod -R u+w {analysis_base}

# Or change ownership if needed
sudo chown -R $USER:$USER {analysis_base}
```

**Location**: `validation.py:60`

---

### Debug Commands

**Check Stage 2.5 inputs**:
```bash
# Verify winner_analysis.json exists
cat {analysis_base}/winner_analysis.json | jq '.top_3_buckets'

# Check all checkpoints exist
for bucket in $(cat {analysis_base}/winner_analysis.json | jq -r '.top_3_buckets[]'); do
  ls -lh {analysis_base}/buckets/bucket_$bucket/checkpoints/stage_2_checkpoint.json
done

# Count completed videos per bucket
for bucket in $(cat {analysis_base}/winner_analysis.json | jq -r '.top_3_buckets[]'); do
  count=$(cat {analysis_base}/buckets/bucket_$bucket/checkpoints/stage_2_checkpoint.json | jq '.completed | tonumber')
  echo "Bucket $bucket: $count completed"
done
```

**Check source files**:
```bash
# Count temporal_windows files in source
ls /home/jorge/rumiaifinal/insights/*_temporal_windows_updated.json | wc -l

# Count video files in source
ls /home/jorge/rumiaifinal/temp/*.mp4 | wc -l

# Count unified_analysis files in source
ls /home/jorge/rumiaifinal/unified_analysis/*.json | wc -l
```

**Verify file organization**:
```bash
# Count organized files per bucket
for bucket in $(cat {analysis_base}/winner_analysis.json | jq -r '.top_3_buckets[]'); do
  echo "Bucket: $bucket"
  echo "  temporal_windows: $(ls {analysis_base}/buckets/bucket_$bucket/analysis/insights/*.json 2>/dev/null | wc -l)"
  echo "  videos: $(ls {analysis_base}/buckets/bucket_$bucket/videos/*.mp4 2>/dev/null | wc -l)"
  echo "  unified: $(ls {analysis_base}/buckets/bucket_$bucket/analysis/unified/*.json 2>/dev/null | wc -l)"
done
```

**Check selection_manifest.json**:
```bash
# Verify manifest exists
ls -lh {analysis_base}/selection_manifest.json

# View manifest structure
cat {analysis_base}/selection_manifest.json | jq '{
  hashtag: .hashtag,
  buckets: .selected_buckets,
  summary: (.videos_by_bucket | to_entries | map({
    bucket: .key,
    top: (.value.top_performers | length),
    bottom: (.value.bottom_performers | length)
  }))
}'
```

---

## Modification Guide

### Scenario 1: Add Support for 4th File Type (e.g., service_debug JSON)

**Requirement**: Organize service_debug JSON files in addition to existing 3 file types

**Files to Modify**:
1. `file_organizer.py` - Add 4th file type to build_file_list()

**Steps**:

**Step 1**: Update SOURCE_DIRS constant
```python
# File: ml_pipeline/stage2_5_organize/file_organizer.py:18-23
SOURCE_DIRS = {
    'temporal_windows': '/home/jorge/rumiaifinal/insights/',
    'videos': '/home/jorge/rumiaifinal/temp/',
    'unified_analysis': '/home/jorge/rumiaifinal/unified_analysis/',
    'service_debug': '/home/jorge/rumiaifinal/service_debug/'  # NEW
}
```

**Step 2**: Add 4th file type in build_file_list()
```python
# File: ml_pipeline/stage2_5_organize/file_organizer.py:196-197
# After line 196, add:

# 7d. Service debug JSON (NEW)
files_to_process.append({
    'video_id': video_id,
    'bucket': bucket,
    'file_type': 'service_debug',
    'source_path': f"{SOURCE_DIRS['service_debug']}{video_id}_service_debug.json",
    'target_path': f"{analysis_base}/buckets/bucket_{bucket}/analysis/service_debug/{video_id}_service_debug.json"
})
```

**Impact**: Now organizes 4 files per video instead of 3

**Test**:
```bash
# Run Stage 2.5
python rumiai_ml_batch.py --client test --target nutrition

# Verify 4 file types organized
ls {analysis_base}/buckets/bucket_18-33s/analysis/service_debug/
```

---

### Scenario 2: Change Source Directory from Hardcoded to Config-Based

**Requirement**: Make source directories configurable instead of hardcoded

**Files to Modify**:
1. `file_organizer.py` - Accept source_dirs parameter
2. `main.py` - Pass source_dirs from config

**Steps**:

**Step 1**: Update build_file_list() signature
```python
# File: ml_pipeline/stage2_5_organize/file_organizer.py:130
# OLD: def build_file_list(analysis_base: str, winning_buckets: List[str]) -> List[Dict[str, str]]:
# NEW:
def build_file_list(
    analysis_base: str,
    winning_buckets: List[str],
    source_dirs: Dict[str, str] = None
) -> List[Dict[str, str]]:
    """Build list of files to organize."""

    # Use provided source_dirs or fall back to default
    if source_dirs is None:
        source_dirs = SOURCE_DIRS  # Use hardcoded default

    # Rest of function remains same, uses source_dirs variable
    files_to_process.append({
        'source_path': f"{source_dirs['temporal_windows']}{video_id}_temporal_windows_updated.json",
        ...
    })
```

**Step 2**: Update main.py to accept and pass source_dirs
```python
# File: ml_pipeline/stage2_5_organize/main.py:24
def stage_2_5_file_organization_main(
    analysis_base: str,
    source_dirs: Dict[str, str] = None  # NEW: Optional source directories
) -> Dict[str, any]:

    # Pass source_dirs to build_file_list
    files_to_process = build_file_list(analysis_base, winning_buckets, source_dirs)
```

**Step 3**: Update orchestrator to pass custom source_dirs
```python
# File: rumiai_ml_batch.py:833
organization_summary = stage_2_5_file_organization_main(
    analysis_base=str(analysis_base),
    source_dirs={
        'temporal_windows': '/custom/path/insights/',
        'videos': '/custom/path/temp/',
        'unified_analysis': '/custom/path/unified/'
    }
)
```

**Test**:
```bash
# Test with custom source directories
# Files should be organized from custom paths
```

---

### Scenario 3: Skip Missing Files Silently (Don't Log Warnings)

**Requirement**: Don't log warnings for missing files (reduce log noise)

**Files to Modify**:
1. `file_organizer.py`

**Steps**:

**Step 1**: Change logging level or remove warning
```python
# File: ml_pipeline/stage2_5_organize/file_organizer.py:289-293

# OLD:
if not source_exists and not target_exists:
    logger.warning(
        f"Missing source and target for video {video_id} ({file_type}). "
        f"Stage 2 checkpoint indicated completion, but file doesn't exist."
    )
    missing_count += 1
    continue

# NEW: Use debug level instead of warning
if not source_exists and not target_exists:
    logger.debug(f"Missing: {video_id} ({file_type})")
    missing_count += 1
    continue
```

**Impact**: Missing files still counted in `missing_count` but no warning logs

**Test**:
```bash
# Run Stage 2.5
# Check logs - should not see warnings for missing files
grep "Missing" logs/rumiai_ml_batch.log
```

---

### Scenario 4: Add Retry Logic for Failed File Moves

**Requirement**: Retry failed file moves 3 times before giving up

**Files to Modify**:
1. `file_organizer.py`

**Steps**:

**Step 1**: Add retry logic to organize_files_with_detection()
```python
# File: ml_pipeline/stage2_5_organize/file_organizer.py:296-313

# OLD: Single try/except
try:
    os.makedirs(target_dir, exist_ok=True)
    shutil.move(source, target)
    moved_count += 1
except Exception as e:
    logger.error(f"Failed to move {video_id} ({file_type}): {e}")
    continue

# NEW: Retry loop
import time

max_retries = 3
for attempt in range(max_retries):
    try:
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(source, target)
        moved_count += 1
        logger.info(f"Moved: {video_id} ({file_type}) → {bucket}")
        break  # Success, exit retry loop

    except Exception as e:
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            logger.warning(f"Move failed (attempt {attempt + 1}/{max_retries}): {video_id} ({file_type}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
        else:
            # Final attempt failed
            logger.error(f"Failed to move {video_id} ({file_type}) after {max_retries} attempts: {e}")
            continue
```

**Test**:
```bash
# Simulate move failure (remove write permission temporarily)
chmod -w {analysis_base}/buckets/bucket_18-33s/analysis/insights/

# Run Stage 2.5
# Should see retry attempts in logs

# Restore permission
chmod +w {analysis_base}/buckets/bucket_18-33s/analysis/insights/
```

---

### Scenario 5: Add Dry-Run Mode (Preview Without Moving)

**Requirement**: Add --dry-run flag to preview what would be moved without actually moving

**Files to Modify**:
1. `file_organizer.py` - Add dry_run parameter
2. `main.py` - Pass dry_run parameter
3. `rumiai_ml_batch.py` - Add CLI flag

**Steps**:

**Step 1**: Update organize_files_with_detection() to support dry-run
```python
# File: ml_pipeline/stage2_5_organize/file_organizer.py:252
def organize_files_with_detection(
    files_to_process: List[Dict[str, str]],
    dry_run: bool = False  # NEW parameter
) -> Dict[str, int]:
    """Organize files with optional dry-run mode."""

    # ... existing code ...

    # Step 6: Case 3 - Source exists (move it)
    if dry_run:
        # Dry-run mode: Log what WOULD be moved
        logger.info(f"[DRY-RUN] Would move: {video_id} ({file_type}) → {bucket}")
        moved_count += 1
    else:
        # Normal mode: Actually move
        try:
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(source, target)
            moved_count += 1
            logger.info(f"Moved: {video_id} ({file_type}) → {bucket}")
        except Exception as e:
            logger.error(f"Failed to move {video_id} ({file_type}): {e}")
            continue
```

**Step 2**: Update main.py
```python
# File: ml_pipeline/stage2_5_organize/main.py:24
def stage_2_5_file_organization_main(
    analysis_base: str,
    dry_run: bool = False  # NEW parameter
) -> Dict[str, any]:

    # ... existing steps ...

    # Step 5: Organize files
    summary = organize_files_with_detection(files_to_process, dry_run=dry_run)

    if dry_run:
        logger.info("[DRY-RUN] No files were actually moved")
```

**Step 3**: Update orchestrator
```python
# File: rumiai_ml_batch.py:833
organization_summary = stage_2_5_file_organization_main(
    analysis_base=str(analysis_base),
    dry_run=args.dry_run  # Pass from CLI args
)
```

**Test**:
```bash
# Add --dry-run flag to CLI
python rumiai_ml_batch.py --client test --target nutrition --dry-run

# Should see "[DRY-RUN] Would move: ..." logs
# No files actually moved
```

---

## Related Documentation

### Pipeline Documentation
- **[PRODUCTION_FLOW.md](PRODUCTION_FLOW.md)**: Complete pipeline overview (Stages 0-7)
- **[PRODUCTION_FLOW.md - Stage 2.5 Contract](PRODUCTION_FLOW.md#stage-25-file-organization)**: Stage 2.5 inputs/outputs/dependencies

### Upstream/Downstream Stages
- **[STAGE_1_IMPL.md](STAGE_1_IMPL.md)**: Video Discovery (provides winner_analysis.json)
- **[STAGE_2_IMPL.md](STAGE_2_IMPL.md)**: Video Processing (provides checkpoints, temporal_windows files)
- **Stage 2.6**: Content Discovery (consumes selection_manifest.json)
- **Stage 2.7**: Content Classification (consumes selection_manifest.json)
- **Stage 8**: Report Generation (consumes selection_manifest.json)

### Stage 2.5 Source Files
- **[ml_pipeline/stage2_5_organize/main.py](ml_pipeline/stage2_5_organize/main.py)**: Main orchestrator (103 lines)
- **[ml_pipeline/stage2_5_organize/file_organizer.py](ml_pipeline/stage2_5_organize/file_organizer.py)**: Core file organization (452 lines)
- **[ml_pipeline/stage2_5_organize/validation.py](ml_pipeline/stage2_5_organize/validation.py)**: Input/output validation (92 lines)
- **[ml_pipeline/stage2_5_organize/__init__.py](ml_pipeline/stage2_5_organize/__init__.py)**: Exports (11 lines)

---

## Document Metadata

**Generated**: 2025-11-02
**Source**: 100% systematic code reading (658 production lines across 4 modules)
**Verification**: All line numbers, schemas, and code snippets from actual source code
**Coverage**: Complete Stage 2.5 implementation (file organization + manifest creation)

**Last Validated**: 2025-11-02
**Hardcoded Paths**: /home/jorge/rumiaifinal/insights/, temp/, unified_analysis/
