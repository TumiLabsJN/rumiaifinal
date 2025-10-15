# File Organization (Stage 2.5) - Technical Implementation

> **TI Document**: FileOrganizationCHILDTI.md
> **Parent HLD**: FileOrganizationCHILD.md
> **Foundation HLD**: FoundationCHILD.md (Shared across all stages)
> **Version**: 1.0
> **Last Updated**: 2025-01-13
> **Status**: To Implement

---

## 1. Document Metadata

**TI_Document**: FileOrganizationCHILDTI.md

**Parent_HLD**: FileOrganizationCHILD.md

**Foundation_HLD**: FoundationCHILD.md (Shared cross-cutting information)

**Covers_HLD_Sections**:
  - FileOrganizationCHILD.md Section 1: Context & Business Goal
  - FileOrganizationCHILD.md Section 2: Architecture & Design
  - FileOrganizationCHILD.md Section 2.1: High-Level Approach
  - FileOrganizationCHILD.md Section 2.2: Data Flow
  - FileOrganizationCHILD.md Section 2.3: Detailed Process
  - FileOrganizationCHILD.md Section 3: Dependencies & Integration
  - FileOrganizationCHILD.md Section 3.1: Input Dependencies
  - FileOrganizationCHILD.md Section 3.2: Output Contracts
  - FileOrganizationCHILD.md Section 3.3: Cross-Stage Dependencies
  - FileOrganizationCHILD.md Section 3.4: External Dependencies
  - FileOrganizationCHILD.md Section 4: Configuration & Parameters
  - FileOrganizationCHILD.md Section 5: Data Schemas
  - FileOrganizationCHILD.md Section 6: Error Handling & Validation
  - FoundationCHILD.md Section 2: Client Architecture & Storage
  - FoundationCHILD.md Section 2.1: Directory Structure
  - FoundationCHILD.md Section 2.2: Path Templates
  - FoundationCHILD.md Section 4: CLI Command Structure
  - FoundationCHILD.md Section 4.1: CLI Parameters
  - FoundationCHILD.md Section 5: Configuration Schemas
  - FoundationCHILD.md Section 5.1: config.json Schema
  - FoundationCHILD.md Section 5.3: Checkpoint Schema
  - FoundationCHILD.md Section 6: Bucket Definitions

**Related_TI_Docs**:
  - Depends_On:
    - FoundationTI.md (ALWAYS - provides directory structure, CLI params, config schemas)
    - VideoDiscoveryTI.md (Stage 1.3 - produces winner_analysis.json)
    - VideoProcessingTI.md (Stage 2 - produces temporal_windows_updated.json files and stage_2_checkpoint.json)
  - Feeds_Into:
    - FeatureAggregationTI.md (Stage 3 - consumes organized temporal_windows files)

**Implementation_Priority**: CRITICAL (Blocking - Stage 3 cannot run without organized files in bucket directories)

---

## 2. Stage Contract

### INPUT CONTRACT

```python
class StageInput:
    """
    Exact structure this stage receives.
    Sources: FoundationCHILD.md Sections 2 & 4, FileOrganizationCHILD Sections 3.1 & 5.1
    """
    # From FoundationCHILD.md Section 4 (CLI parameters)
    client_id: str  # CLI parameter --client, Required
    analysis_type: str  # CLI parameter --analysis-type, Required, ["hashtag", "competitor", "creator"]
    target: str  # CLI parameter --target, Required, Example: "#nutrition"
    analysis_mode: str  # CLI parameter --analysis-mode, Default: "top"
    selection_strategy: str  # CLI parameter --selection-strategy, Default: "contrastive"
    video_count: int  # CLI parameter --video-count, Default: 100

    # From FoundationCHILD.md Section 2 (directory paths)
    # Base path construction: /data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/
    analysis_base: str  # Path to analysis directory
                        # Example: /data/clients/acme/hashtags/nutrition/top_contrastive/

    # From FileOrganizationCHILD Section 3.1 (stage-specific inputs)
    winner_analysis_json_path: str  # Path to winner_analysis.json, Required, must exist
                                    # Location: {analysis_base}/winner_analysis.json
                                    # Schema: {"top_3_buckets": list[str]}

    stage_2_checkpoint_paths: dict[str, str]  # Per-bucket checkpoint paths, Required, must exist
                                              # Key: bucket name (e.g., "18-33s")
                                              # Value: path to stage_2_checkpoint.json
                                              # Location: {analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json

    source_insights_dir: str  # Flat directory with temporal_windows_updated.json files, Required
                              # Location: /home/jorge/rumiaifinal/insights/
                              # Contains: {video_id}_temporal_windows_updated.json files

    # Note: temporal_windows_files list is built during processing (not a direct input)
    # It is derived from stage_2_checkpoint.json completed_video_ids in build_file_list()
    # Source files location: {source_insights_dir}/{video_id}_temporal_windows_updated.json
```

### OUTPUT CONTRACT

```python
class StageOutput:
    """
    Exact structure this stage produces.
    Sources: FoundationCHILD.md Section 2, FileOrganizationCHILD Sections 3.2 & 5.2
    """
    # Construct using base paths from FoundationCHILD.md Section 2
    organized_files_by_bucket: dict[str, list[str]]
        # Key: bucket name (e.g., "18-33s")
        # Value: list of organized file paths
        # Location: {analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json

    empty_source_dir: bool  # True if /insights/ directory is empty after organization
                            # Indicates successful completion

    organization_summary: dict  # Summary statistics
        # Schema:
        # {
        #   "moved_count": int,
        #   "skipped_already_organized": int,
        #   "missing_count": int,
        #   "total_processed": int
        # }
```

---

## 3. Data Schemas

### FOUNDATION SCHEMAS

```python
# From FoundationCHILD.md Section 5.1: config.json Schema
ConfigSchema = {
    "client_id": str,              # Required, alphanumeric + underscore, Example: "acme_corp"
    "analysis_type": str,          # Required, ["hashtag", "competitor", "creator"], Example: "hashtag"
    "target": str,                 # Required, format depends on analysis_type, Example: "#nutrition"
    "analysis_mode": str,          # Required, ["top", "recent"], Example: "top"
    "selection_strategy": str,     # Required, ["contrastive", "top"], Example: "contrastive"
    "video_count": int,            # Required, Range: 10-500, Example: 100
    "date_filter": str,            # Required, "last_N_days", Example: "last_90_days"
    "country_code": str,           # Required, ["US", "BR", "global"], Example: "US"
    "report_type": str,            # Required, ["single", "comparison"], Example: "single"
    "report_audience": str,        # Required, ["client", "internal", "creator"], Example: "client"
    "auto_confirm": bool,          # Required, skip interactive prompts, Example: false
    "run_date": str,               # Required, ISO 8601 format, Example: "2025-01-28T10:30:00Z"
}

# From FoundationCHILD.md Section 5.3: Checkpoint Schema
CheckpointSchema = {
    "stage": str,                  # Required, Stage name, Example: "video_processing"
    "bucket": str,                 # Required, Bucket name, Example: "18-33s"
    "total_videos": int,           # Required, Total videos to process, Example: 100
    "completed": int,              # Required, Successfully processed, Example: 45
    "failed": int,                 # Required, Failed with errors, Example: 2
    "remaining": int,              # Required, Not yet processed, Example: 53
    "status": str,                 # Required, ["completed", "in_progress", "paused"], Example: "completed"
    "last_checkpoint": str,        # Required, ISO timestamp, Example: "2025-01-28T14:32:15Z"
    "completed_video_ids": list[str],   # Required, List of processed video IDs
    "failed_video_ids": list[dict],     # Required, List of failure records
        # Nested schema for failed_video_ids items:
        # {
        #   "video_id": str,        # Required, Video ID that failed
        #   "error": str,           # Required, Error message/reason
        #   "timestamp": str,       # Optional, ISO timestamp of failure
        #   "stage": str            # Optional, Substage that failed (e.g., "FEAT", "Whisper")
        # }
}
```

### STAGE-SPECIFIC INPUT SCHEMAS

```python
# File 1: winner_analysis.json
# Source: FileOrganizationCHILD.md Section 5.1
# Location: {analysis_base}/winner_analysis.json

WinnerAnalysisSchema = {
    "top_3_buckets": list[str],    # Required, Winning bucket names from Stage 1
                                   # Example: ["18-33s", "33-60s", "13-18s"]
                                   # Description: List of 3 winning bucket names identified by Stage 1 winner analysis

    "top_100_distribution": dict,  # Optional, Bucket distribution (informational)
                                   # Example: {"18-33s": 45, "33-60s": 30}
                                   # Description: Distribution of top 100 videos across buckets

    "winner_coverage": float,      # Optional, Percentage of winners in top 3 buckets
                                   # Range: 0-100
                                   # Example: 95.0
                                   # Description: Percentage of top performers covered by winning buckets

    "scrape_timestamp": str,       # Optional, ISO 8601 timestamp
                                   # Example: "2025-01-28T10:30:00Z"
                                   # Description: When Stage 1 analysis was completed
}

# File 2: stage_2_checkpoint.json (per winning bucket)
# Source: FileOrganizationCHILD.md Section 5.1
# Location: {analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json

Stage2CheckpointSchema = CheckpointSchema  # Inherits from FoundationCHILD.md Section 5.3
    # Required fields specific to Stage 2:
    # - stage: must equal "video_processing"
    # - bucket: must match bucket directory name
    # - completed_video_ids: list of successfully processed video IDs to organize

# File 3: temporal_windows_updated.json (flat directory)
# Source: FileOrganizationCHILD.md Section 5.1
# Location: /home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json

TemporalWindowsSchema = {
    "metadata.duration": float,    # Required, Video duration (seconds)
                                   # Range: 3-120
                                   # Example: 25.3
                                   # Description: Video length used for bucket assignment

    "temporal_windows": dict,      # Required, Feature data (not validated by Stage 2.5)
                                   # Example: {"hook": {...}, "middle_segments": [...]}
                                   # Description: Temporal window features (passed through without modification)
}
```

### STAGE-SPECIFIC OUTPUT SCHEMAS

```python
# Output: Organized temporal_windows_updated.json files
# Source: FileOrganizationCHILD.md Section 5.2
# Location: {analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json

OrganizedFileSchema = TemporalWindowsSchema  # Schema identical to input (files are moved, not modified)
    # No modifications to file content during organization
    # Only file location changes from flat /insights/ directory to bucket-specific directories

# Organization Summary (in-memory only, not persisted to file)
# Source: FileOrganizationCHILD.md Section 2.3.4

OrganizationSummarySchema = {
    "moved_count": int,                   # Required, Number of files moved this run
                                          # Range: >= 0
                                          # Example: 150

    "skipped_already_organized": int,     # Required, Number of files already organized in previous run
                                          # Range: >= 0
                                          # Example: 50

    "missing_count": int,                 # Required, Number of files missing despite checkpoint indicating completion
                                          # Range: >= 0
                                          # Example: 2

    "total_processed": int,               # Required, Total files processed (moved + skipped + missing)
                                          # Range: >= 0
                                          # Example: 202
}
```

**Field Count Verification (Cross-Referenced with Child HLD):**
- WinnerAnalysisSchema: 4 fields ✓ (Child Section 5.1 Table has 4 rows: top_3_buckets, top_100_distribution, winner_coverage, scrape_timestamp)
- Stage2CheckpointSchema: Inherits from CheckpointSchema (8 base fields) ✓ (Foundation Section 5.3)
- TemporalWindowsSchema: 2 fields ✓ (Child Section 5.1 Table has 2 rows: metadata.duration, temporal_windows)
- OrganizationSummarySchema: 4 fields ✓ (Child Section 2.3.4 defines 4 return values: moved_count, skipped_already_organized, missing_count, total_processed)

---

## 4. Algorithmic Specifications

### Function 1: load_winning_buckets

**Source**: FileOrganizationCHILD.md Section 2.3.1 - Load Winning Buckets

**Purpose**: Determine which 3 buckets were selected by Stage 1 winner analysis

**Algorithm (Pseudocode)**:
```python
def load_winning_buckets(analysis_base: str) -> list[str]:
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

    Source: FileOrganizationCHILD.md Section 2.3.1, QA Q5, Q10
    """
    # Step 1: Construct winner_analysis.json path
    winner_analysis_path = f"{analysis_base}/winner_analysis.json"

    # Step 2: Validate file exists
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

**Edge Cases (Exhaustive List)**:
- **Case 1**: File doesn't exist → Fail-fast with FileNotFoundError (Rationale: Stage 1 didn't complete)
- **Case 2**: JSON corrupted → Fail-fast with ValueError (Rationale: Cannot proceed without valid data)
- **Case 3**: Empty top_3_buckets → Fail-fast with ValueError (Rationale: No buckets to organize)

**Validation Rules**:
- Assert winner_analysis_path exists before reading
- Assert 'top_3_buckets' field present in loaded JSON
- Assert 'top_3_buckets' is list type
- Assert 'top_3_buckets' is non-empty list

**Error Conditions**:
- Missing winner_analysis.json (Error Code: 1)
- Invalid winner_analysis schema (Error Code: 2)

**Example Input**:
```json
{
  "top_100_distribution": {"18-33s": 45, "33-60s": 30, "13-18s": 20},
  "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
  "winner_coverage": 95.0,
  "scrape_timestamp": "2025-01-28T10:30:00Z"
}
```

**Example Output**:
```python
["18-33s", "33-60s", "13-18s"]
```

**Example Trace (Step-by-Step)**:
```
Input: analysis_base = "/data/clients/acme/hashtags/nutrition/top_contrastive/"
Step 1: Construct path → "/data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json"
Step 2: Check file exists → True
Step 3: Load JSON → winner_analysis dict loaded
Step 4: Validate 'top_3_buckets' exists → True
Step 5: Validate type is list → True
Step 6: Validate non-empty → True (3 buckets)
Step 7: Log and return → ["18-33s", "33-60s", "13-18s"]
Output: ["18-33s", "33-60s", "13-18s"]
```

---

### Function 2: build_file_list

**Source**: FileOrganizationCHILD.md Section 2.3.2 - Build File List from Checkpoints

**Purpose**: Determine exactly which files to organize by reading Stage 2 checkpoints for completed video IDs

**Algorithm (Pseudocode)**:
```python
def build_file_list(analysis_base: str, winning_buckets: list[str]) -> list[dict]:
    """
    Build list of files to organize from Stage 2 checkpoints.

    Args:
        analysis_base: str, path to analysis directory
        winning_buckets: list, bucket names from winner_analysis.json

    Returns:
        list: file info dicts with keys: video_id, bucket, source_path, target_path

    Source: FileOrganizationCHILD.md Section 2.3.2, QA Q8, Q9
    """
    files_to_process = []

    # Step 1: Iterate through each winning bucket
    for bucket in winning_buckets:
        # Step 2: Construct checkpoint path for this bucket
        checkpoint_path = f"{analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json"

        # Step 3: Validate checkpoint file exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint missing for bucket {bucket}: {checkpoint_path}")
            raise FileNotFoundError(
                f"Checkpoint not found for bucket {bucket}. Did Stage 2 complete?"
            )

        # Step 4: Load checkpoint JSON
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Step 5: Validate checkpoint schema and extract video IDs
        video_ids = validate_checkpoint(checkpoint, bucket)

        # Step 6: Skip if no completed videos
        if len(video_ids) == 0:
            logger.info(f"Bucket {bucket} has 0 completed videos. Skipping.")
            continue

        # Step 7: Build file info for each video
        for video_id in video_ids:
            # Construct source path (flat directory)
            source_path = f"/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json"

            # Construct target path (bucket-specific directory)
            target_path = f"{analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json"

            # Add to file list
            files_to_process.append({
                'video_id': video_id,
                'bucket': bucket,
                'source_path': source_path,
                'target_path': target_path
            })

    # Step 8: Log summary
    logger.info(f"Built file list: {len(files_to_process)} files across {len(winning_buckets)} buckets")
    return files_to_process


def validate_checkpoint(checkpoint: dict, bucket: str) -> list[str]:
    """
    Validate checkpoint schema and extract completed_video_ids.

    Args:
        checkpoint: dict, loaded from stage_2_checkpoint.json
        bucket: str, bucket name for error messages

    Returns:
        list: completed_video_ids to process

    Raises:
        ValueError: if checkpoint schema is invalid

    Source: FileOrganizationCHILD.md Section 2.3.2, QA Q9
    """
    # Step 1: Strict schema validation - check required fields
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

**Edge Cases (Exhaustive List)**:
- **Case 1**: Checkpoint missing → Fail-fast with FileNotFoundError (Rationale: Stage 2 didn't complete for this bucket)
- **Case 2**: Checkpoint status="paused" → Warning log, process partial (Rationale: Allow organizing partial results)
- **Case 3**: Zero completed videos → Info log, skip bucket (Rationale: No files to organize for this bucket)
- **Case 4**: Invalid schema → Fail-fast with ValueError (Rationale: Cannot trust corrupted checkpoint)

**Validation Rules**:
- Assert checkpoint file exists for each winning bucket
- Assert checkpoint has all required fields: ['stage', 'bucket', 'completed_video_ids', 'status', 'total_videos']
- Assert 'completed_video_ids' is list type
- Warn if status != "completed" but allow processing

**Error Conditions**:
- Missing checkpoint file (Error Code: 3)
- Invalid checkpoint schema (Error Code: 4)

**Example Input**:
```python
analysis_base = "/data/clients/acme/hashtags/nutrition/top_contrastive/"
winning_buckets = ["18-33s", "33-60s", "13-18s"]

# Checkpoint file content for bucket 18-33s:
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

**Example Output**:
```python
[
    {
        'video_id': '7428596413707144481',
        'bucket': '18-33s',
        'source_path': '/home/jorge/rumiaifinal/insights/7428596413707144481_temporal_windows_updated.json',
        'target_path': '/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144481_temporal_windows_updated.json'
    },
    {
        'video_id': '7428596413707144482',
        'bucket': '18-33s',
        'source_path': '/home/jorge/rumiaifinal/insights/7428596413707144482_temporal_windows_updated.json',
        'target_path': '/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144482_temporal_windows_updated.json'
    },
    # ... more files for buckets 33-60s and 13-18s
]
```

---

### Function 3: detect_duplicates_across_buckets

**Source**: FileOrganizationCHILD.md Section 2.3.3 - Detect Duplicate Video IDs

**Purpose**: Ensure each video appears in exactly one bucket (data integrity check)

**Algorithm (Pseudocode)**:
```python
def detect_duplicates_across_buckets(files_to_process: list[dict]) -> None:
    """
    Detect if same video_id appears in multiple buckets.

    Args:
        files_to_process: list of dict with keys: video_id, bucket, source_path

    Raises:
        ValueError: if duplicate video_id detected

    Source: FileOrganizationCHILD.md Section 2.3.3, QA Q11
    """
    # Step 1: Initialize tracking dictionary
    video_id_to_buckets = {}

    # Step 2: Iterate through all files to process
    for file_info in files_to_process:
        video_id = file_info['video_id']
        bucket = file_info['bucket']

        # Step 3: Check if video_id already seen
        if video_id in video_id_to_buckets:
            previous_bucket = video_id_to_buckets[video_id]

            # Step 4: Duplicate detected - raise error with detailed message
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

        # Step 5: Record video_id → bucket mapping
        video_id_to_buckets[video_id] = bucket

    # Step 6: Log validation success
    logger.info(f"Validation passed: {len(video_id_to_buckets)} unique videos")
```

**Edge Cases (Exhaustive List)**:
- **Case 1**: Duplicate detected → Fail-fast with detailed error (Rationale: Data corruption - must investigate)
- **Case 2**: All videos unique → Continue processing (Rationale: Normal operation)

**Validation Rules**:
- Assert each video_id appears in exactly one bucket
- Fail-fast if duplicate detected (critical data integrity issue)

**Error Conditions**:
- Duplicate video_id across buckets (Error Code: 5)

**Example Input**:
```python
files_to_process = [
    {'video_id': '123', 'bucket': '18-33s', 'source_path': '/insights/123_temporal_windows_updated.json'},
    {'video_id': '456', 'bucket': '18-33s', 'source_path': '/insights/456_temporal_windows_updated.json'},
    {'video_id': '789', 'bucket': '33-60s', 'source_path': '/insights/789_temporal_windows_updated.json'},
]
```

**Example Output**:
```python
# No return value - function validates and logs success
# If validation passes: logger.info("Validation passed: 3 unique videos")
# If validation fails: raises ValueError
```

---

### Function 4: organize_files_with_detection

**Source**: FileOrganizationCHILD.md Section 2.3.4 - Organize Files with Detection-Based Resume

**Purpose**: Move files from flat /insights/ directory to bucket directories with automatic resume detection

**Algorithm (Pseudocode)**:
```python
def organize_files_with_detection(files_to_process: list[dict]) -> dict:
    """
    Organize files with automatic resume detection (no checkpoint needed).

    Args:
        files_to_process: list of dict with keys: video_id, bucket, source_path, target_path

    Returns:
        dict: Summary statistics (moved_count, skipped_count, missing_count)

    Source: FileOrganizationCHILD.md Section 2.3.4, QA Q6, Q12
    """
    # Step 1: Initialize counters
    moved_count = 0
    skipped_already_organized = 0
    missing_count = 0

    # Step 2: Iterate through each file to process
    for file_info in files_to_process:
        source = file_info['source_path']
        target = file_info['target_path']
        video_id = file_info['video_id']
        bucket = file_info['bucket']

        # Step 3: Check file existence states
        source_exists = os.path.exists(source)
        target_exists = os.path.exists(target)

        # Step 4: Case 1 - Already moved in previous run
        if target_exists and not source_exists:
            logger.debug(f"Already organized: {video_id} → {bucket}")
            skipped_already_organized += 1
            continue

        # Step 5: Case 2 - Missing entirely
        if not source_exists and not target_exists:
            logger.warning(
                f"Missing source and target for video {video_id}. "
                f"Stage 2 checkpoint indicated completion, but file doesn't exist."
            )
            missing_count += 1
            continue

        # Step 6: Case 3 - Source exists (move it)
        try:
            # Step 6a: Ensure target directory exists
            target_dir = os.path.dirname(target)
            os.makedirs(target_dir, exist_ok=True)

            # Step 6b: Move file (atomic within same filesystem)
            shutil.move(source, target)
            moved_count += 1

            # Step 6c: Log success
            logger.info(f"Moved: {video_id} → {bucket} ({moved_count}/{len(files_to_process)})")

        except Exception as e:
            # Step 6d: Log error but continue processing other files
            logger.error(f"Failed to move {video_id}: {e}")
            # Non-fatal error - continue with other files
            continue

    # Step 7: Calculate summary statistics
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

    # Step 9: Warn if missing files detected
    if missing_count > 0:
        logger.warning(f"{missing_count} files missing despite checkpoint indicating completion.")

    # Step 10: Return summary dictionary
    return {
        'moved_count': moved_count,
        'skipped_already_organized': skipped_already_organized,
        'missing_count': missing_count,
        'total_processed': total_processed
    }
```

**Edge Cases (Exhaustive List)**:
- **Case 1**: Target exists, source missing → Skip with debug log (Rationale: Already organized in previous run)
- **Case 2**: Both exist → Re-move (overwrites target) (Rationale: Handles interrupted moves)
- **Case 3**: Neither exists → Warning log, skip (Rationale: Checkpoint/file mismatch)
- **Case 4**: Move failure (permissions/disk) → Error log, continue (Rationale: Non-fatal, process other files)

**Validation Rules**:
- Check source and target existence before attempting move
- Create target directories if they don't exist (os.makedirs with exist_ok=True)
- Handle exceptions during file move gracefully (log and continue)

**Error Conditions**:
- Missing source file (Warning, non-fatal)
- Move failure (Error log, continue with other files)

**Example Input**:
```python
files_to_process = [
    {
        'video_id': '7428596413707144481',
        'bucket': '18-33s',
        'source_path': '/home/jorge/rumiaifinal/insights/7428596413707144481_temporal_windows_updated.json',
        'target_path': '/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144481_temporal_windows_updated.json'
    },
    # ... more files
]
```

**Example Output**:
```python
{
    'moved_count': 150,
    'skipped_already_organized': 50,
    'missing_count': 2,
    'total_processed': 202
}
```

---

## 5. Validation Rules

### INPUT VALIDATION

```python
# Source: FileOrganizationCHILD.md Section 6.1: Input Validation

def validate_inputs(analysis_base: str, winning_buckets: list[str]) -> None:
    """
    Validate inputs before starting file organization.

    Args:
        analysis_base: str, path to analysis directory
        winning_buckets: list, bucket names from winner_analysis.json

    Raises:
        ValueError: if validation fails

    Source: FileOrganizationCHILD.md Section 6.1, QA Q5, Q9, Q10
    """
    # Validation 1: Check analysis_base exists
    assert os.path.exists(analysis_base), \
        f"Analysis base directory does not exist: {analysis_base}. Did Foundation setup run?"

    # Validation 2: Check winning_buckets is valid
    assert winning_buckets and len(winning_buckets) > 0, \
        "No winning buckets provided. Check winner_analysis.json."

    # Validation 3: Validate bucket names are in expected set
    ALL_BUCKETS = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
    for bucket in winning_buckets:
        assert bucket in ALL_BUCKETS, \
            f"Invalid bucket name: {bucket}. Expected one of {ALL_BUCKETS}"

    # Validation 4: Check source directory exists
    SOURCE_DIR = "/home/jorge/rumiaifinal/insights/"
    assert os.path.exists(SOURCE_DIR), \
        f"Source directory does not exist: {SOURCE_DIR}. Did Stage 2 complete?"

    # Validation 5: Check write permissions to analysis_base
    test_file = f"{analysis_base}/test_write.tmp"
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise ValueError(f"No write permission to {analysis_base}: {e}")
```

### BUSINESS LOGIC VALIDATION

```python
# Source: FileOrganizationCHILD.md Section 2.3.X Edge Cases tables

def validate_business_rules(checkpoint: dict, bucket: str) -> None:
    """
    Validate business rules during processing.
    Source: FileOrganizationCHILD.md Section 2.3.2 Edge Cases
    """
    # Edge Case: Allow partial completion (status != "completed")
    if checkpoint['status'] != 'completed':
        logger.warning(
            f"Checkpoint for {bucket} status is '{checkpoint['status']}' (not 'completed'). "
            f"Processing partial results."
        )

    # Edge Case: Handle zero completions gracefully
    if len(checkpoint['completed_video_ids']) == 0:
        logger.info(f"Bucket {bucket} has 0 completed videos. Skipping bucket.")
```

### OUTPUT VALIDATION

```python
# Source: FileOrganizationCHILD.md Section 6.3: Output Validation

def validate_output(analysis_base: str, winning_buckets: list[str], moved_count: int) -> None:
    """
    Validate output after file organization completes.

    Args:
        analysis_base: str, path to analysis directory
        winning_buckets: list, bucket names processed
        moved_count: int, number of files moved

    Source: FileOrganizationCHILD.md Section 6.3, QA Q12
    """
    # Validation 1: Check files exist in target locations
    total_organized = 0
    for bucket in winning_buckets:
        target_dir = f"{analysis_base}/buckets/bucket_{bucket}/analysis/insights/"
        if os.path.exists(target_dir):
            organized_count = len([f for f in os.listdir(target_dir) if f.endswith('.json')])
            total_organized += organized_count
            logger.info(f"Bucket {bucket}: {organized_count} files organized")

    # Validation 2: Log summary
    logger.info(f"Total files organized: {total_organized}")
    logger.info(f"Files moved this run: {moved_count}")

    # Validation 3: Warn if mismatch (some files may have been organized in previous run)
    if total_organized < moved_count:
        logger.warning(
            f"Organized count ({total_organized}) less than moved count ({moved_count}). "
            f"Some files may be missing."
        )
```

---

## 6. Error Handling

```python
# Source: FileOrganizationCHILD.md Section 6.2: Error Cases table

ERROR_CONDITIONS = {
    "missing_winner_analysis_json": {
        "condition": "not os.path.exists(winner_analysis_path)",
        "error_type": "FileNotFoundError",
        "action": "Fail-fast (raise exception, exit with code 1)",
        "retry_policy": "No retry",
        "user_message": "winner_analysis.json not found at {path}. Did Stage 1 complete?"
    },

    "invalid_winner_analysis_schema": {
        "condition": "'top_3_buckets' not in winner_analysis or not isinstance(winner_analysis['top_3_buckets'], list)",
        "error_type": "ValueError",
        "action": "Fail-fast (raise exception, exit with code 2)",
        "retry_policy": "No retry",
        "user_message": "winner_analysis.json missing 'top_3_buckets' field. Re-run Stage 1."
    },

    "missing_checkpoint_file": {
        "condition": "not os.path.exists(checkpoint_path)",
        "error_type": "FileNotFoundError",
        "action": "Fail-fast (raise exception, exit with code 3)",
        "retry_policy": "No retry",
        "user_message": "Checkpoint not found for bucket {bucket}. Did Stage 2 complete?"
    },

    "invalid_checkpoint_schema": {
        "condition": "missing_fields = [f for f in required_fields if f not in checkpoint]; bool(missing_fields)",
        "error_type": "ValueError",
        "action": "Fail-fast (raise exception, exit with code 4)",
        "retry_policy": "No retry",
        "user_message": "Checkpoint for {bucket} has invalid schema (missing {fields}). Re-run Stage 2."
    },

    "duplicate_video_id_across_buckets": {
        "condition": "video_id in video_id_to_buckets",
        "error_type": "ValueError",
        "action": "Fail-fast (raise exception, exit with code 5)",
        "retry_policy": "No retry",
        "user_message": "Video ID '{id}' appears in multiple buckets. Checkpoint corruption detected."
    },

    "missing_source_file": {
        "condition": "not os.path.exists(source) and not os.path.exists(target)",
        "error_type": "Warning",
        "action": "Warning log, skip file (exit code 0, continue processing)",
        "retry_policy": "No retry",
        "user_message": "Missing source for video {id}. Skipping."
    },

    "move_failure_permissions": {
        "condition": "Exception during shutil.move() with PermissionError",
        "error_type": "PermissionError",
        "action": "Error log, continue with other files (exit code 0)",
        "retry_policy": "No retry",
        "user_message": "Failed to move {id}: {error}. Continuing with other files."
    },

    "move_failure_disk_full": {
        "condition": "Exception during shutil.move() with OSError",
        "error_type": "OSError",
        "action": "Error log, continue with other files (exit code 0)",
        "retry_policy": "No retry",
        "user_message": "Failed to move {id}: disk full. Free space and re-run."
    },

    "no_files_to_organize": {
        "condition": "len(files_to_process) == 0",
        "error_type": "Warning",
        "action": "Warning log, exit gracefully (exit code 0)",
        "retry_policy": "No retry",
        "user_message": "No files to organize. All checkpoints have 0 completed videos."
    },
}
```

---

## 7. Complete Example Traces

### TRACE 1: Normal Processing (Happy Path)

**Source**: FileOrganizationCHILD.md Appendix B.1: Sample winner_analysis.json → Appendix B.2: Sample stage_2_checkpoint.json, Section 2.2: Data Flow

**Input**:

```json
// File: /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json
{
  "top_100_distribution": {"18-33s": 45, "33-60s": 30, "13-18s": 20},
  "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
  "winner_coverage": 95.0,
  "scrape_timestamp": "2025-01-28T10:30:00Z"
}

// File: /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/checkpoints/stage_2_checkpoint.json
{
  "stage": "video_processing",
  "bucket": "18-33s",
  "total_videos": 100,
  "completed": 98,
  "failed": 2,
  "remaining": 0,
  "status": "completed",
  "completed_video_ids": ["7428596413707144481", "7428596413707144482", "7428596413707144483"],
  "failed_video_ids": [
    {"video_id": "321", "error": "FEAT timeout after 120s"}
  ]
}

// Source files in flat directory:
/home/jorge/rumiaifinal/insights/7428596413707144481_temporal_windows_updated.json
/home/jorge/rumiaifinal/insights/7428596413707144482_temporal_windows_updated.json
/home/jorge/rumiaifinal/insights/7428596413707144483_temporal_windows_updated.json
```

**Processing Steps**:

```
Step 1: Load winner_analysis.json → Loaded winning buckets: ["18-33s", "33-60s", "13-18s"]
        Intermediate: winning_buckets = ["18-33s", "33-60s", "13-18s"]

Step 2: Build file list from Stage 2 checkpoints → Loaded checkpoint for bucket 18-33s
        Intermediate: checkpoint loaded with 3 completed video IDs

Step 3: Validate checkpoint schema → Validation passed
        Intermediate: All required fields present, status="completed"

Step 4: Extract completed_video_ids → Extracted 3 video IDs from checkpoint
        Intermediate: video_ids = ["7428596413707144481", "7428596413707144482", "7428596413707144483"]

Step 5: Build file info for each video → Created file info dicts
        Intermediate: files_to_process has 3 entries for bucket 18-33s

Step 6: Repeat steps 2-5 for buckets 33-60s and 13-18s → Built complete file list
        Intermediate: files_to_process = 7 files across 3 buckets
        (3 files from 18-33s + 2 files from 33-60s + 2 files from 13-18s)

Step 7: Detect duplicate video IDs → Validation passed: 7 unique videos
        Intermediate: No duplicates detected

Step 8: Organize files with detection → Processing file 1/7
        Intermediate: source exists, target doesn't exist → move file

Step 9: Create target directory → Created /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/
        Intermediate: target_dir created successfully

Step 10: Move file → Moved video 7428596413707144481 to bucket 18-33s (1/7)
         Intermediate: moved_count = 1

Step 11: Repeat steps 8-10 for remaining 6 files → Organization complete
         Intermediate: moved_count = 7, skipped_already_organized = 0, missing_count = 0
```

**Output**:

```python
# Organized files by bucket:
{
    "18-33s": [
        "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144481_temporal_windows_updated.json",
        "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144482_temporal_windows_updated.json",
        "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144483_temporal_windows_updated.json"
    ],
    "33-60s": [
        "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_33-60s/analysis/insights/7428596413707144484_temporal_windows_updated.json",
        "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_33-60s/analysis/insights/7428596413707144485_temporal_windows_updated.json"
    ],
    "13-18s": [
        "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_13-18s/analysis/insights/7428596413707144486_temporal_windows_updated.json",
        "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_13-18s/analysis/insights/7428596413707144487_temporal_windows_updated.json"
    ]
}

# Organization summary:
{
    "moved_count": 7,
    "skipped_already_organized": 0,
    "missing_count": 0,
    "total_processed": 7
}

# Empty source directory:
/home/jorge/rumiaifinal/insights/ → (empty - all 7 files moved)
```

**Files Created**:
- /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144481_temporal_windows_updated.json
- /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144482_temporal_windows_updated.json
- /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144483_temporal_windows_updated.json
- /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_33-60s/analysis/insights/7428596413707144484_temporal_windows_updated.json
- /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_33-60s/analysis/insights/7428596413707144485_temporal_windows_updated.json
- /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_13-18s/analysis/insights/7428596413707144486_temporal_windows_updated.json
- /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_13-18s/analysis/insights/7428596413707144487_temporal_windows_updated.json

**Logs**:
```
INFO: Loading winning buckets from /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json
INFO: Loaded winning buckets: ['18-33s', '33-60s', '13-18s']
INFO: Building file list from Stage 2 checkpoints
INFO: Built file list: 7 files across 3 buckets
INFO: Validating for duplicate video IDs across buckets
INFO: Validation passed: 7 unique videos
INFO: Starting file organization
INFO: Moved: 7428596413707144481 → 18-33s (1/7)
INFO: Moved: 7428596413707144482 → 18-33s (2/7)
INFO: Moved: 7428596413707144483 → 18-33s (3/7)
INFO: Moved: 7428596413707144484 → 33-60s (4/7)
INFO: Moved: 7428596413707144485 → 33-60s (5/7)
INFO: Moved: 7428596413707144486 → 13-18s (6/7)
INFO: Moved: 7428596413707144487 → 13-18s (7/7)
INFO: Organization complete:
  Total files:  7
  Moved:        7
  Already done: 0
  Missing:      0
  Processed:    7/7
INFO: Bucket 18-33s: 3 files organized
INFO: Bucket 33-60s: 2 files organized
INFO: Bucket 13-18s: 2 files organized
INFO: Total files organized: 7
INFO: Files moved this run: 7
```

---

### TRACE 2: Edge Case - Resume After Interruption (Already Organized Files)

**Source**: FileOrganizationCHILD.md Section 2.3.4 Edge Cases table

**Input**:

```json
// Same winner_analysis.json and checkpoints as Trace 1
// But 50 files were already organized in a previous run that was interrupted

// Files already organized (target exists, source doesn't):
/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144481_temporal_windows_updated.json (exists)
/home/jorge/rumiaifinal/insights/7428596413707144481_temporal_windows_updated.json (doesn't exist)

// Files still needing organization (source exists, target doesn't):
/home/jorge/rumiaifinal/insights/7428596413707144482_temporal_windows_updated.json (exists)
/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144482_temporal_windows_updated.json (doesn't exist)
```

**Processing Steps**:

```
Step 1: Load winner_analysis.json → Loaded winning buckets: ["18-33s", "33-60s", "13-18s"]
        Intermediate: winning_buckets = ["18-33s", "33-60s", "13-18s"]

Step 2: Build file list from checkpoints → Built file list: 7 files across 3 buckets
        Intermediate: files_to_process = 7 files

Step 3: Detect duplicates → Validation passed: 7 unique videos
        Intermediate: No duplicates detected

Step 4: Organize file 1 - video 7428596413707144481 → Check file existence
        Intermediate: source_exists = False, target_exists = True

Step 5: Detect already organized (target exists, source missing) → Skip with debug log
        Intermediate: skipped_already_organized = 1
        Edge Case Handling: "Already organized in previous run" → Skip file

Step 6: Organize file 2 - video 7428596413707144482 → Check file existence
        Intermediate: source_exists = True, target_exists = False

Step 7: Move file 2 → Moved video 7428596413707144482 to bucket 18-33s (1/7)
        Intermediate: moved_count = 1

Step 8: Repeat for remaining 5 files → Organization complete
        Intermediate: moved_count = 4, skipped_already_organized = 3, missing_count = 0
        (Scenario: 3 files already organized from previous run, 4 files moved this run)
```

**Output**:

```python
# Organization summary:
{
    "moved_count": 4,
    "skipped_already_organized": 3,
    "missing_count": 0,
    "total_processed": 7
}
```

**Logs**:
```
INFO: Loading winning buckets from /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json
INFO: Loaded winning buckets: ['18-33s', '33-60s', '13-18s']
INFO: Built file list: 7 files across 3 buckets
INFO: Validation passed: 7 unique videos
INFO: Starting file organization
DEBUG: Already organized: 7428596413707144481 → 18-33s
INFO: Moved: 7428596413707144482 → 18-33s (1/7)
DEBUG: Already organized: 7428596413707144483 → 18-33s
DEBUG: Already organized: 7428596413707144484 → 33-60s
INFO: Moved: 7428596413707144485 → 33-60s (2/7)
INFO: Moved: 7428596413707144486 → 13-18s (3/7)
INFO: Moved: 7428596413707144487 → 13-18s (4/7)
INFO: Organization complete:
  Total files:  7
  Moved:        4
  Already done: 3
  Missing:      0
  Processed:    7/7
INFO: Total files organized: 7
INFO: Files moved this run: 4
```

---

### TRACE 3: Error Case - Missing winner_analysis.json

**Source**: FileOrganizationCHILD.md Section 6.2: Error Cases table

**Input**:

```python
# winner_analysis.json is missing at expected location
analysis_base = "/data/clients/acme/hashtags/nutrition/top_contrastive/"
winner_analysis_path = "/data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json"
# File does not exist
```

**Processing Steps**:

```
Step 1: Load winner_analysis.json → Construct path
        Intermediate: winner_analysis_path = "/data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json"

Step 2: Validate file exists → os.path.exists(winner_analysis_path) returns False
        Intermediate: File not found

Step 3: Raise FileNotFoundError → PROCESS TERMINATED
        Error Detection: "not os.path.exists(winner_analysis_path)"
        Error Handling: "Fail-fast with error message"
```

**Output**:

```python
# No output - process terminated before any file operations
```

**Error**:

```
FileNotFoundError: winner_analysis.json not found at:
  /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json

This file is created by Stage 1.3 (Winner Analysis).
Stage 2.5 requires this file to know which buckets to organize.

Solutions:
  1. Complete Stage 1 (Video Discovery & Winner Analysis)
  2. Check if Stage 1 completed successfully
  3. Verify analysis_base path is correct: /data/clients/acme/hashtags/nutrition/top_contrastive/
```

**Exit Code**: 1

**Logs**:
```
INFO: Starting File Organization (Stage 2.5)
INFO: Loading winning buckets from /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json
ERROR: winner_analysis.json not found at: /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json
ERROR: This file is created by Stage 1.3 (Winner Analysis). Stage 2.5 requires this file to know which buckets to organize.
ERROR: Process terminated with exit code 1
```

---

## 8. File Structure & Integration

### MODULE LOCATION
```python
FILE_PATH = "/rumiai_v2/ml_pipeline/stage2_5_organize.py"
# Location rationale: Part of ML pipeline stages, between Stage 2 and Stage 3
```

### IMPORTS
```python
# From FileOrganizationCHILD.md Section 3.4: External Dependencies - Python Libraries

import os        # Standard library - file system operations
import json      # Standard library - JSON file reading/writing
import shutil    # Standard library - file move operations
import logging   # Standard library - logging
```

### ENTRY POINT
```python
ENTRY_FUNCTION = "organize_files"
# Main orchestration function that executes all 4 steps:
# 1. load_winning_buckets()
# 2. build_file_list()
# 3. detect_duplicates_across_buckets()
# 4. organize_files_with_detection()
```

### INTEGRATION POINTS
```python
CALLS_TO_EXTERNAL_SYSTEMS = {}
# No external systems - pure file system operations
```

### BASE DIRECTORY STRUCTURE
```python
# From FoundationCHILD.md Section 2: Client Architecture & Storage

BASE_PATHS = {
    "client_base": "/data/clients/{client_id}/",
    "analysis_type_base": "{client_base}/{analysis_type}s/",  # Note: plural form
    "target_base": "{analysis_type_base}/{target}/",
    "analysis_base": "{target_base}/{mode}_{strategy}/",
    "bucket_base": "{analysis_base}/buckets/bucket_{bucket}/",

    # Standard subdirectories per bucket
    "videos": "{bucket_base}/videos/",
    "analysis": "{bucket_base}/analysis/",
    "insights": "{bucket_base}/analysis/insights/",
    "unified": "{bucket_base}/analysis/unified/",
    "service_debug": "{bucket_base}/analysis/service_debug/",
    "checkpoints": "{bucket_base}/checkpoints/",
    "logs": "{bucket_base}/logs/",
}

# Example path construction:
# analysis_base = "/data/clients/acme/hashtags/nutrition/top_contrastive/"
# insights_path = "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/"
```

### STAGE INPUT PATHS
```python
# From FileOrganizationCHILD.md Section 3.1: Input Dependencies

INPUT_PATHS = {
    "winner_analysis_json": "{analysis_base}/winner_analysis.json",
    "stage_2_checkpoint": "{analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json",
    "source_insights_dir": "/home/jorge/rumiaifinal/insights/",
    "temporal_windows_file_pattern": "/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json",
}
```

### STAGE OUTPUT PATHS
```python
# From FileOrganizationCHILD.md Section 3.2: Output Contracts
# Construct using BASE_PATHS + stage-specific subdirectories

OUTPUT_PATHS = {
    "organized_insights": "{bucket_base}/analysis/insights/{video_id}_temporal_windows_updated.json",
    # Example: /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144481_temporal_windows_updated.json

    # Note: No checkpoint for Stage 2.5 (uses detection-based resume)
    # Logs are optional (if needed for debugging)
    "logs": "{bucket_base}/logs/stage_2_5_organize_{date}.log",
}
```

---

## 9. Configuration & Environment

### ENVIRONMENT VARIABLES REQUIRED
```python
# From FileOrganizationCHILD.md Section 3.4: External Dependencies - Environment Variables
# Note: Stage 2.5 has no environment variables (paths determined from analysis_base parameter)

ENV_VARS = {}
# No environment variables required for this stage
```

### CONFIGURATION OBJECT
```python
# From FileOrganizationCHILD.md Section 4.1: CLI Parameters
# Note: Stage 2.5 is invoked programmatically, not via direct CLI

CONFIG_SCHEMA = {
    "cli_params": {
        # Stage 2.5 receives analysis_base as input parameter (not CLI parameter)
        # Base CLI parameters are defined in FoundationCHILD.md Section 4
        # and are used to construct analysis_base path
        "client_id": None,           # Required, passed from Foundation
        "analysis_type": None,       # Required, passed from Foundation
        "target": None,              # Required, passed from Foundation
        "analysis_mode": "top",      # Default from Foundation
        "selection_strategy": "contrastive",  # Default from Foundation
    },

    # From FileOrganizationCHILD.md Section 4.2: Internal Configuration
    "internal_constants": {
        "SOURCE_DIR": "/home/jorge/rumiaifinal/insights/",
        "ALL_BUCKETS": ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"],
        "SOURCE_FILE_PATTERN": "{video_id}_temporal_windows_updated.json",
        "TARGET_SUBDIR": "analysis/insights/",
        "LOG_LEVEL": "INFO",  # DEBUG for troubleshooting
        "BATCH_SIZE": None,   # Process all files in single pass (no batching)
    },
}
```

### CONSTANTS
```python
# From FileOrganizationCHILD.md Section 4.2: Internal Configuration

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

## 10. Logging Specifications

### LOG LEVELS & MESSAGES
```python
# From FileOrganizationCHILD.md Section 2.3.X pseudocode (log statements extracted)

LOG_MESSAGES = {
    # Stage start/end
    # Source: Inferred from standard stage execution pattern
    "stage_start": ("INFO", "Starting File Organization (Stage 2.5)"),
    "stage_complete": ("INFO", "File Organization (Stage 2.5) complete"),

    # Function 1: load_winning_buckets
    # Source: Child Section 2.3.1, lines 144, 293-303
    "loading_winner_analysis": ("INFO", "Loading winning buckets from {path}"),
    "loaded_winner_buckets": ("INFO", "Loaded winning buckets: {buckets}"),
    "winner_analysis_missing": ("ERROR", "winner_analysis.json not found at: {path}"),

    # Function 2: build_file_list
    # Source: Child Section 2.3.2, lines 180-207, 244-248
    "building_file_list": ("INFO", "Building file list from Stage 2 checkpoints"),
    "checkpoint_missing": ("ERROR", "Checkpoint missing for bucket {bucket}: {path}"),
    "bucket_zero_completions": ("INFO", "Bucket {bucket} has 0 completed videos. Skipping."),
    "checkpoint_partial": ("WARNING", "Checkpoint for {bucket} status is '{status}' (not 'completed'). Processing {completed}/{total} videos."),
    "file_list_built": ("INFO", "Built file list: {count} files across {bucket_count} buckets"),

    # Function 3: detect_duplicates_across_buckets
    # Source: Child Section 2.3.3, lines 301
    "validating_duplicates": ("INFO", "Validating for duplicate video IDs across buckets"),
    "validation_passed": ("INFO", "Validation passed: {count} unique videos"),

    # Function 4: organize_files_with_detection
    # Source: Child Section 2.3.4, lines 343-385
    "organizing_files": ("INFO", "Starting file organization"),
    "file_already_organized": ("DEBUG", "Already organized: {video_id} → {bucket}"),
    "file_missing": ("WARNING", "Missing source and target for video {video_id}. Stage 2 checkpoint indicated completion, but file doesn't exist."),
    "file_moved": ("INFO", "Moved: {video_id} → {bucket} ({moved_count}/{total})"),
    "file_move_failed": ("ERROR", "Failed to move {video_id}: {error}"),

    # Summary
    # Source: Child Section 2.3.4, lines 374-385
    "organization_summary": ("INFO", """
Organization complete:
  Total files:  {total}
  Moved:        {moved}
  Already done: {skipped}
  Missing:      {missing}
  Processed:    {processed}/{total}
"""),
    "missing_files_warning": ("WARNING", "{count} files missing despite checkpoint indicating completion."),

    # Output validation
    # Source: Child Section 6.3, lines 653-664
    "bucket_organized_count": ("INFO", "Bucket {bucket}: {count} files organized"),
    "total_organized": ("INFO", "Total files organized: {count}"),
    "files_moved_this_run": ("INFO", "Files moved this run: {count}"),
    "organized_count_mismatch": ("WARNING", "Organized count ({organized}) less than moved count ({moved}). Some files may be missing."),
}
```

### METRICS TO TRACK
```python
# Inferred from FileOrganizationCHILD.md Section 7.1: Performance Targets

METRICS = {
    "files_processed": "Total files organized (for throughput calculation)",
    "processing_time_seconds": "Total processing time in seconds",
    "moved_count": "Number of files moved this run",
    "skipped_already_organized": "Number of files skipped (already organized in previous run)",
    "missing_count": "Number of files missing despite checkpoint indicating completion",
    "files_per_bucket": "Number of files organized per bucket (dict[str, int])",
    "file_move_time_seconds": "Time spent moving files (cumulative)",
    "checkpoint_load_time_seconds": "Time spent loading and validating checkpoints",
    "duplicate_detection_time_seconds": "Time spent detecting duplicate video IDs",
}
```

---

## 11. Dependencies & Prerequisites

### EXTERNAL DEPENDENCIES
```python
# From FileOrganizationCHILD.md Section 3.4: External Dependencies - Python Libraries

EXTERNAL_DEPS = {
    "os": {
        "version": "Standard library (built-in)",
        "purpose": "File system operations (path checks, directory creation)",
        "pip_install": "N/A (built-in)"
    },
    "json": {
        "version": "Standard library (built-in)",
        "purpose": "JSON file loading for winner_analysis.json and checkpoints",
        "pip_install": "N/A (built-in)"
    },
    "shutil": {
        "version": "Standard library (built-in)",
        "purpose": "File move operations (shutil.move for atomic file moves)",
        "pip_install": "N/A (built-in)"
    },
    "logging": {
        "version": "Standard library (built-in)",
        "purpose": "Logging operations for stage execution tracking",
        "pip_install": "N/A (built-in)"
    },
}
```

### UPSTREAM TI REQUIREMENTS
```python
# From FileOrganizationCHILD.md Section 3.3: Cross-Stage Dependencies

UPSTREAM_OUTPUTS_REQUIRED = {
    "FoundationTI": [
        # Directory structure must be created
        "/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/",
        "/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/buckets/bucket_{bucket}/",
    ],

    "VideoDiscoveryTI": [
        # Stage 1.3 output: winner_analysis.json
        "/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/winner_analysis.json",
    ],

    "VideoProcessingTI": [
        # Stage 2 outputs: temporal_windows_updated.json files (flat directory)
        "/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json",

        # Stage 2 outputs: stage_2_checkpoint.json (per winning bucket)
        "/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json",
    ],
}
```

### SYSTEM PREREQUISITES
```python
# From FileOrganizationCHILD.md Section 7.1: Performance Targets and Section 7.4: Scalability Limits

SYSTEM_REQUIREMENTS = {
    "disk_space": "Negligible (files moved, not duplicated). Ensure /insights/ and /data/ on same filesystem for atomic moves.",
    "memory": "< 100 MB (minimal data in memory, streaming approach)",
    "api_keys": [],  # None required for this stage
    "network": "Not required (pure computational stage, no external API calls)",
    "filesystem": "/insights/ and /data/ must be on same filesystem (for atomic move operations, 10x faster than copy+delete)",
}
```

---

## 12. HLD Traceability Matrix

| HLD Section | TI Section | Implementation Status |
|-------------|------------|----------------------|
| FileOrganizationCHILD.md Section 1: Context & Business Goal | Section 1: Document Metadata | To Implement |
| FileOrganizationCHILD.md Section 2.1: High-Level Approach | Section 4: Algorithmic Specifications (intro) | To Implement |
| FileOrganizationCHILD.md Section 2.2: Data Flow | Section 7: Complete Example Traces (Trace 1) | To Implement |
| FileOrganizationCHILD.md Section 2.3.1: Load Winning Buckets | Section 4: Function 1: load_winning_buckets() | To Implement |
| FileOrganizationCHILD.md Section 2.3.2: Build File List from Checkpoints | Section 4: Function 2: build_file_list() | To Implement |
| FileOrganizationCHILD.md Section 2.3.3: Detect Duplicate Video IDs | Section 4: Function 3: detect_duplicates_across_buckets() | To Implement |
| FileOrganizationCHILD.md Section 2.3.4: Organize Files with Detection-Based Resume | Section 4: Function 4: organize_files_with_detection() | To Implement |
| FileOrganizationCHILD.md Section 3.1: Input Dependencies | Section 2: StageInput contract | To Implement |
| FileOrganizationCHILD.md Section 3.2: Output Contracts | Section 2: StageOutput contract | To Implement |
| FileOrganizationCHILD.md Section 3.3: Cross-Stage Dependencies | Section 11: Upstream TI Requirements | To Implement |
| FileOrganizationCHILD.md Section 3.4: External Dependencies | Section 11: External Dependencies | To Implement |
| FileOrganizationCHILD.md Section 4.1: CLI Parameters | Section 9: Configuration & Environment | To Implement |
| FileOrganizationCHILD.md Section 4.2: Internal Configuration | Section 9: Constants | To Implement |
| FileOrganizationCHILD.md Section 5.1: Input Schema | Section 3: Data Schemas (input) | To Implement |
| FileOrganizationCHILD.md Section 5.2: Output Schema | Section 3: Data Schemas (output) | To Implement |
| FileOrganizationCHILD.md Section 6.1: Input Validation | Section 5: Validation Rules (input) | To Implement |
| FileOrganizationCHILD.md Section 6.2: Error Cases | Section 6: Error Handling | To Implement |
| FileOrganizationCHILD.md Section 6.3: Output Validation | Section 5: Validation Rules (output) | To Implement |
| FileOrganizationCHILD.md Section 7: Performance & Scalability | Section 10: Metrics | To Implement |
| FileOrganizationCHILD.md Appendix B: Example Data | Section 7: Complete Example Traces | To Implement |
| FoundationCHILD.md Section 2: Client Architecture & Storage | Section 8: File Structure & Integration (BASE_PATHS) | To Implement |
| FoundationCHILD.md Section 2.1: Directory Structure | Section 8: File Structure & Integration (BASE_PATHS) | To Implement |
| FoundationCHILD.md Section 2.2: Path Templates | Section 8: File Structure & Integration (OUTPUT_PATHS) | To Implement |
| FoundationCHILD.md Section 4: CLI Command Structure | Section 2: Stage Contract (StageInput CLI params) | To Implement |
| FoundationCHILD.md Section 4.1: CLI Parameters | Section 9: Configuration & Environment (cli_params) | To Implement |
| FoundationCHILD.md Section 5: Configuration Schemas | Section 3: Data Schemas (Foundation schemas) | To Implement |
| FoundationCHILD.md Section 5.1: config.json Schema | Section 3: Data Schemas (ConfigSchema) | To Implement |
| FoundationCHILD.md Section 5.3: Checkpoint Schema | Section 3: Data Schemas (CheckpointSchema) | To Implement |
| FoundationCHILD.md Section 6: Bucket Definitions | Section 9: Constants (ALL_BUCKETS) | To Implement |

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
| 1.0 | 2025-01-13 | RumiAI Team | Initial TI generation from FileOrganizationCHILD.md and FoundationCHILD.md |
