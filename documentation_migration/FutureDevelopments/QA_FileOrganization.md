# Clarification Q&A: File Organization (Stage 2.5)

> **Mother Doc**: MLPlanningv2.md Section 2.5 "File Organization (Bucket Assignment)"
> **Phase 1**: Critique_FileOrganization.md
> **Date**: 2025-01-09
> **Status**: IN PROGRESS

## Questions by Category

### Input/Output Contracts

#### Q1: [CRITICAL] Input Source Directory & File Pattern
**Question**: What is the exact absolute path to the source directory where Stage 2 outputs temporal_windows_updated.json files? Is it Option A (`/home/jorge/rumiaifinal/insights/`), Option B (analysis-specific path), or something else? Also confirm exact filename pattern.

**Answer**: Option A - `/home/jorge/rumiaifinal/insights/` (single global directory for all videos)

**Filename Pattern**: Confirmed as `{video_id}_temporal_windows_updated.json` where `{video_id}` is the TikTok video ID (e.g., `7428596413707144481_temporal_windows_updated.json`)

**For HLD Section**: 5.1 (Input Schema), 3.1 (Input Dependencies)

**Notes**: This confirms Stage 2 (rumiai_runner.py) currently saves all outputs to a single flat directory regardless of client/hashtag/bucket. Stage 2.5 will read from this global location and organize into bucket-specific directories.

#### Q2: [CRITICAL] Output Directory Path Construction
**Question**: What is the exact absolute path where Stage 2.5 should move files? Should it use analysis-scoped path?

**Answer**: Use analysis-scoped path (full path construction):
```
/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json
```

**Example**:
```
/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596413707144481_temporal_windows_updated.json
```

**Path Parameters**:
- `client_id`: From CLI `--client` parameter (e.g., "acme")
- `analysis_type`: From CLI `--analysis-type` parameter (hashtag/competitor/creator) - **use plural form** (hashtags/competitors/creators)
- `target`: From CLI `--target` parameter, **sanitized** (remove # or @) (e.g., "nutrition")
- `mode`: From CLI `--analysis-mode` parameter (top/recent)
- `strategy`: From CLI `--selection-strategy` parameter (contrastive/top)
- `bucket`: Determined by duration via `assign_bucket()` function (e.g., "18-33s")
- `video_id`: From JSON filename (e.g., "7428596413707144481")

**Parameter Source Decision**: Read from `config.json` (recommended by Claude, user said "not sure" but approved analysis-scoped path)
- Location: `/data/clients/{client_id}/{analysis_type}s/{target}/{mode}_{strategy}/config.json`
- Schema: FoundationCHILD.md Section 5.1 (ConfigSchema)

**For HLD Section**: 5.2 (Output Schema), 3.2 (Output Contracts), 2.3 (Detailed Process)

**Notes**:
- Matches FoundationCHILD.md Section 2.2 path templates (line 212: `"insights": "{bucket_base}/analysis/insights/"`)
- Supports multi-analysis coexistence (different mode/strategy combinations can run side-by-side)
- Must sanitize target (remove # or @ prefix) per FoundationCHILD.md Section 2.2.1

#### Q3: [CRITICAL] Stage 2.5 Execution Scope
**Question**: Does Stage 2.5 run once per analysis run (organizing ALL buckets), or once per bucket?

**Answer**: Once per analysis run - processes all videos from `/insights/` and distributes to multiple bucket directories

**Implications**:
- Single script invocation processes entire `/insights/` directory
- Outputs to multiple bucket directories in one pass (e.g., bucket_18-33s, bucket_33-60s, bucket_13-18s)
- Atomic operation: Either all files organized or none (transaction-like behavior recommended)

**For HLD Section**: 2.1 (High-Level Approach), 2.2 (Data Flow), 4.1 (CLI Parameters)

**Notes**:
- More efficient than per-bucket execution (single directory scan)
- Requires tracking which buckets are "active" (selected by Stage 1 winner analysis)
- Must handle videos from non-winning buckets gracefully (see Q4)

#### Q4: [CRITICAL] Non-Winning Bucket Video Handling
**Question**: What happens if Stage 2.5 encounters a video with duration that doesn't match any of the 3 selected winning buckets?

**Example Scenario**: Winner analysis selected buckets [18-33s, 33-60s, 13-18s], but a 9.5s video (bucket 9-13s) exists in `/insights/`

**Answer**: **Option A - Skip it with warning (it shouldn't have been processed by Stage 2)**

**Handling**:
- Validate video duration against expected winning buckets
- If video belongs to non-winning bucket → **SKIP with warning**
- Warning message: "Skipping video {video_id} (duration {duration}s, bucket {bucket}). Not in winning buckets: {winning_buckets}. File remains in /insights/ directory."
- Continue processing remaining videos (non-fatal)

**Rationale**:
- Graceful degradation (doesn't halt entire pipeline)
- Indicates potential upstream issue but allows recovery
- Leaves file in original location for manual inspection
- Logs warning for debugging/auditing

**For HLD Section**: 6.2 (Error Cases), 6.1 (Input Validation), 2.3 (Detailed Process)

**Notes**:
- Non-fatal validation warning
- Should rarely happen if Stage 1 → Stage 2 handoff is correct
- Skipped videos remain in `/insights/` for troubleshooting
- Summary report should include count of skipped videos

#### Q5: [CRITICAL] Winning Buckets Discovery
**Question**: How does Stage 2.5 discover which buckets are the "winning buckets" (to validate against non-winning bucket videos per Q4)?

**Answer**: **Option A - Read from `winner_analysis.json` (from Stage 1)**

**Implementation**:
- **File Location**: `{analysis_base}/winner_analysis.json`
- **Field**: `top_3_buckets` (list of bucket names)
- **Example**:
  ```json
  {
    "top_100_distribution": {"18-33s": 45, "33-60s": 30, "13-18s": 20},
    "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
    "winner_coverage": 95.0,
    "scrape_timestamp": "2025-01-28T10:30:00Z",
    "analysis_date": "2025-01-28T10:32:15Z"
  }
  ```

**Loading Logic**:
```python
# Construct path to winner_analysis.json
winner_analysis_path = f"{analysis_base}/winner_analysis.json"

# Load winning buckets
with open(winner_analysis_path) as f:
    winner_analysis = json.load(f)

winning_buckets = set(winner_analysis['top_3_buckets'])  # Convert to set for O(1) lookup
# Example: {"18-33s", "33-60s", "13-18s"}
```

**Validation Usage**:
```python
# For each video file in /insights/
video_bucket = assign_bucket(video_duration)

if video_bucket not in winning_buckets:
    logger.warning(f"Skipping {video_id}: bucket {video_bucket} not in winning buckets {winning_buckets}")
    skipped_count += 1
    continue  # Skip this video
```

**For HLD Section**: 3.1 (Input Dependencies), 2.3 (Detailed Process), 6.1 (Input Validation)

**Notes**:
- Consistent with config.json reading pattern (Stage 1 artifacts as input)
- Explicit source of truth (no inference from filesystem)
- Schema defined in VideoDiscoveryCHILD.md Section 5.3 (WinnerAnalysisSchema)
- Must handle file not found gracefully (fail-fast with clear error message)

#### Q6: [CRITICAL] File Operation Type (Move vs Copy)
**Question**: Should Stage 2.5 **move** (delete from source) or **copy** (keep in source) the temporal_windows JSON files from `/insights/` to bucket directories?

**Answer**: **Option A - MOVE (delete from `/insights/` after organizing)**

**Implementation**:
```python
import shutil

# Move file from source to destination
shutil.move(source_path, target_path)
# Result: File exists ONLY in bucket directory, removed from /insights/
```

**Rationale**:
- Stage 2.5 is a one-time organizational step per analysis run
- Files are already processed by Stage 2 (no need to preserve in `/insights/`)
- Prevents storage bloat (no duplicate JSON files)
- Clear signal that file has been processed and organized
- Aligns with original design intent ("move files from insights to buckets" - MLPlanningv2.md line 828)
- User confirmed: "We will not be testing different bucket strategies"

**Operation Characteristics**:
- **Atomic**: `shutil.move()` is atomic within same filesystem
- **Idempotent**: Cannot re-run without re-running Stage 2 (intentional)
- **Failure Handling**: If move fails mid-process, some files remain in `/insights/`, some in buckets (partial state possible)

**For HLD Section**: 2.3 (Detailed Process), 6.2 (Error Cases), 7.1 (Performance Baselines)

**Notes**:
- Use `shutil.move()` not `os.rename()` (handles cross-filesystem moves)
- Log each move operation for auditability
- Track moved count in summary report
- If re-organization needed, must re-run Stage 2 first (full pipeline consistency)

#### Q7: [CRITICAL] Directory Creation & Permissions
**Question**: Who is responsible for creating the bucket directory structure that Stage 2.5 moves files into?

**Answer**: **Stage 2 (Video Processing) creates all bucket directories**

**Clarification from User**: "Stage 2 will create the buckets"

**Implication for Stage 2.5**:
- Stage 2.5 can **assume bucket base directories already exist** (e.g., `bucket_18-33s/`)
- Stage 2.5 should **validate that the target subdirectory exists** before moving files
- Stage 2.5 should **create only the specific output subdirectory if missing**: `{bucket_base}/analysis/insights/`

**Implementation**:
```python
# Construct target directory path
target_dir = f"{bucket_base}/analysis/insights/"

# Ensure output directory exists (Stage 2 created bucket_base, but insights/ might not exist)
os.makedirs(target_dir, exist_ok=True)

# Move file
shutil.move(source_path, target_path)
```

**Validation**:
- If `bucket_base` doesn't exist → This indicates Stage 2 didn't run properly → Stage 2.5 should fail-fast with clear error
- If `analysis/insights/` doesn't exist → Stage 2.5 creates it (normal operation)

**For HLD Section**: 2.3 (Detailed Process), 3.1 (Input Dependencies), 6.1 (Input Validation)

**Notes**:
- Stage 2 validation (VideoProcessingCHILD.md line 1082) confirms bucket directories must exist before Stage 2 runs
- This means Stage 2 creates them on-demand during video download/processing
- Stage 2.5 inherits existing structure, only ensures specific subdirectories exist

#### Q8: [CRITICAL] Source Input File Pattern Matching
**Question**: How should Stage 2.5 identify which files to process from the `/insights/` directory?

**Answer**: **Option B - Read from Stage 2 checkpoint (winning buckets only)**

**Implementation**:
```python
# Load winning buckets from winner_analysis.json
winner_analysis = load_json(f"{analysis_base}/winner_analysis.json")
winning_buckets = winner_analysis['top_3_buckets']  # e.g., ["18-33s", "33-60s", "13-18s"]

files_to_process = []

# For each winning bucket, read checkpoint to get completed_video_ids
for bucket in winning_buckets:
    checkpoint_path = f"{analysis_base}/buckets/bucket_{bucket}/checkpoints/stage_2_checkpoint.json"

    # Load checkpoint
    checkpoint = load_json(checkpoint_path)
    video_ids = checkpoint['completed_video_ids']

    # Build list of files to process for this bucket
    for video_id in video_ids:
        source_file = f"/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json"

        if os.path.exists(source_file):
            files_to_process.append({
                'source_path': source_file,
                'video_id': video_id,
                'bucket': bucket
            })
        else:
            logger.warning(f"Missing file for completed video {video_id}: {source_file}")

# Result: Exact list of files to move, filtered to winning buckets only
```

**Rationale**:
- ✅ Processes exactly the videos Stage 2 completed successfully
- ✅ Automatically filters to winning buckets (reads from `top_3_buckets`)
- ✅ Handles partial Stage 2 completion gracefully (uses `completed_video_ids`, skips `failed_video_ids`)
- ✅ No risk of processing stale files from previous analysis runs
- ✅ Aligns with checkpoint-driven architecture
- ✅ Clear audit trail (knows which checkpoint authorized each file)

**Edge Cases**:
- **Missing checkpoint file**: Fail-fast with clear error (Stage 2 didn't complete for this bucket)
- **Missing source file**: Log warning, skip file, continue processing (file may have been manually deleted)
- **Checkpoint has 0 completed videos**: Skip bucket with info log (all videos failed in Stage 2)

**For HLD Section**: 2.3 (Detailed Process), 3.1 (Input Dependencies), 5.1 (Input Schema)

**Notes**:
- This approach guarantees Stage 2.5 only processes files that Stage 2 successfully completed
- Failed videos (in `checkpoint['failed_video_ids']`) are automatically excluded
- Provides clear lineage: checkpoint → video_id → source file → target location

#### Q9: [HIGH] Stage 2 Checkpoint Schema Dependency
**Question**: What should Stage 2.5 do if it encounters an unexpected checkpoint schema version or missing required fields?

**Answer**: **Option C - Hybrid (validate schema, allow partial completion)**

**Implementation**:
```python
def validate_checkpoint(checkpoint, bucket):
    """
    Validate checkpoint schema strictly, but allow partial completion.

    Args:
        checkpoint: dict, loaded from stage_2_checkpoint.json
        bucket: str, bucket name for error messages

    Returns:
        list: completed_video_ids to process (empty list if no completions)

    Raises:
        ValueError: if checkpoint schema is invalid or corrupted
    """

    # Strict schema validation (catches corruption/version mismatches)
    required_fields = ['stage', 'bucket', 'completed_video_ids', 'status', 'total_videos']
    missing = [f for f in required_fields if f not in checkpoint]
    if missing:
        raise ValueError(
            f"Checkpoint for {bucket} has invalid schema (missing {missing}). "
            f"This may indicate checkpoint corruption or version mismatch. "
            f"Re-run Stage 2 to regenerate checkpoint."
        )

    # Validate field types
    if not isinstance(checkpoint['completed_video_ids'], list):
        raise ValueError(f"Checkpoint for {bucket}: 'completed_video_ids' must be list, got {type(checkpoint['completed_video_ids'])}")

    if not isinstance(checkpoint['total_videos'], int):
        raise ValueError(f"Checkpoint for {bucket}: 'total_videos' must be int, got {type(checkpoint['total_videos'])}")

    # Allow partial completion (in_progress, paused, completed)
    valid_statuses = ['in_progress', 'paused', 'completed']
    if checkpoint['status'] not in valid_statuses:
        raise ValueError(
            f"Checkpoint for {bucket} has invalid status: '{checkpoint['status']}'. "
            f"Valid statuses: {valid_statuses}"
        )

    # Log warning for non-completed status (informational, not blocking)
    if checkpoint['status'] != 'completed':
        logger.warning(
            f"Checkpoint for bucket {bucket} status is '{checkpoint['status']}' (not 'completed'). "
            f"Processing {len(checkpoint['completed_video_ids'])}/{checkpoint['total_videos']} completed videos. "
            f"Failed: {checkpoint.get('failed', 0)}"
        )

    # Handle zero completions gracefully (skip bucket, not an error)
    if len(checkpoint['completed_video_ids']) == 0:
        logger.info(
            f"Bucket {bucket} has 0 completed videos "
            f"({checkpoint.get('failed', 0)} failed, {checkpoint.get('remaining', 0)} remaining). "
            f"Skipping bucket."
        )
        return []

    # Validation passed
    logger.info(
        f"Checkpoint for bucket {bucket}: {len(checkpoint['completed_video_ids'])} videos to organize "
        f"(status: {checkpoint['status']})"
    )

    return checkpoint['completed_video_ids']
```

**Rationale**:
- ✅ **Strict schema validation**: Catches checkpoint corruption, version mismatches, invalid types
- ✅ **Flexible for partial completion**: Allows Stage 2.5 to run even if Stage 2 was paused or interrupted
- ✅ **Clear user feedback**: Logging explains what's happening (partial vs full completion)
- ✅ **Aligns with checkpoint architecture**: Partial completion is a valid state (not an error)
- ✅ **Graceful degradation**: Zero completions → skip bucket (informational), not fail-fast

**Edge Cases Handled**:
| Scenario | Handling | User Experience |
|----------|----------|-----------------|
| Checkpoint missing required fields | Fail-fast with ValueError | Clear error: "Invalid schema, re-run Stage 2" |
| Checkpoint `status: "in_progress"` | Warning log, process partial results | "Processing 45/100 videos, status: in_progress" |
| Checkpoint `status: "paused"` | Warning log, process partial results | "Processing 72/100 videos, status: paused" |
| Checkpoint `status: "completed"` | Info log, process all results | "Processing 98/100 videos, status: completed" |
| Zero completed videos | Info log, skip bucket gracefully | "Bucket has 0 completed videos. Skipping." |
| Invalid field types | Fail-fast with TypeError | "completed_video_ids must be list, got dict" |

**For HLD Section**: 3.1 (Input Dependencies), 6.1 (Input Validation), 6.2 (Error Cases)

**Notes**:
- This approach balances data integrity (strict schema) with operational flexibility (partial completion)
- Users can run Stage 2.5 even if Stage 2 failed partway through (processes what exists)
- Clear distinction between schema errors (fail-fast) and operational states (graceful handling)

#### Q10: [HIGH] Missing winner_analysis.json Handling
**Question**: What should Stage 2.5 do if `winner_analysis.json` is missing or corrupted?

**Answer**: **Option C - Fail-fast with helpful suggestion**

**Implementation**:
```python
def load_winning_buckets(analysis_base):
    """
    Load winning buckets from winner_analysis.json with comprehensive error handling.

    Args:
        analysis_base: str, path to analysis directory (e.g., /data/clients/acme/hashtags/nutrition/top_contrastive/)

    Returns:
        list: winning bucket names (e.g., ["18-33s", "33-60s", "13-18s"])

    Raises:
        FileNotFoundError: if winner_analysis.json doesn't exist
        ValueError: if file is corrupted or has invalid schema
        TypeError: if top_3_buckets has wrong type
    """
    winner_analysis_path = f"{analysis_base}/winner_analysis.json"

    # Check file exists
    if not os.path.exists(winner_analysis_path):
        raise FileNotFoundError(
            f"winner_analysis.json not found at:\n"
            f"  {winner_analysis_path}\n\n"
            f"This file is created by Stage 1.3 (Winner Analysis).\n"
            f"Stage 2.5 requires this file to know which buckets to organize.\n\n"
            f"Solutions:\n"
            f"  1. Complete Stage 1 (Video Discovery & Winner Analysis)\n"
            f"  2. Check if Stage 1 completed successfully (look for 'Stage 1.3: Winner Analysis Complete' in logs)\n"
            f"  3. Verify analysis_base path is correct: {analysis_base}"
        )

    # Load with error handling
    try:
        winner_analysis = load_json(winner_analysis_path)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"winner_analysis.json is corrupted (invalid JSON):\n"
            f"  File: {winner_analysis_path}\n"
            f"  Error: {e}\n\n"
            f"Solution: Re-run Stage 1 to regenerate winner_analysis.json"
        )

    # Validate schema - top_3_buckets field exists
    if 'top_3_buckets' not in winner_analysis:
        raise ValueError(
            f"winner_analysis.json missing 'top_3_buckets' field.\n"
            f"  File: {winner_analysis_path}\n"
            f"  Available fields: {list(winner_analysis.keys())}\n\n"
            f"Solution: Re-run Stage 1 to regenerate winner_analysis.json with correct schema"
        )

    # Validate type
    if not isinstance(winner_analysis['top_3_buckets'], list):
        raise TypeError(
            f"'top_3_buckets' must be list, got {type(winner_analysis['top_3_buckets']).__name__}.\n"
            f"  File: {winner_analysis_path}\n\n"
            f"Solution: Re-run Stage 1 to regenerate winner_analysis.json"
        )

    # Validate not empty
    if len(winner_analysis['top_3_buckets']) == 0:
        raise ValueError(
            f"'top_3_buckets' is empty - no winning buckets identified.\n"
            f"  File: {winner_analysis_path}\n\n"
            f"This may indicate:\n"
            f"  1. Stage 1 found no videos matching selection criteria\n"
            f"  2. Winner distribution analysis failed\n"
            f"  3. All videos fell into bottom-performing buckets\n\n"
            f"Solutions:\n"
            f"  1. Check Stage 1 logs for video discovery results\n"
            f"  2. Verify date_filter and selection_strategy parameters\n"
            f"  3. Check winner_analysis.json for 'top_100_distribution' field to see full bucket breakdown"
        )

    # Validation passed
    logger.info(f"Loaded winning buckets: {winner_analysis['top_3_buckets']}")
    return winner_analysis['top_3_buckets']
```

**Rationale**:
- ✅ **Fail-fast with clear errors**: Prevents organizing files into wrong buckets
- ✅ **Helpful error messages**: Guides users to root cause and solution
- ✅ **Strict dependency**: Ensures Stage 1 completed properly before Stage 2.5 runs
- ✅ **Prevents data corruption**: Won't organize files without knowing winning buckets
- ✅ **Comprehensive validation**: Checks existence, JSON validity, schema, type, and content

**Error Message Examples**:

**Scenario 1: File Missing**
```
FileNotFoundError: winner_analysis.json not found at:
  /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json

This file is created by Stage 1.3 (Winner Analysis).
Stage 2.5 requires this file to know which buckets to organize.

Solutions:
  1. Complete Stage 1 (Video Discovery & Winner Analysis)
  2. Check if Stage 1 completed successfully (look for 'Stage 1.3: Winner Analysis Complete' in logs)
  3. Verify analysis_base path is correct: /data/clients/acme/hashtags/nutrition/top_contrastive/
```

**Scenario 2: Empty Buckets**
```
ValueError: 'top_3_buckets' is empty - no winning buckets identified.
  File: /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json

This may indicate:
  1. Stage 1 found no videos matching selection criteria
  2. Winner distribution analysis failed
  3. All videos fell into bottom-performing buckets

Solutions:
  1. Check Stage 1 logs for video discovery results
  2. Verify date_filter and selection_strategy parameters
  3. Check winner_analysis.json for 'top_100_distribution' field to see full bucket breakdown
```

**Edge Cases Handled**:
| Scenario | Handling | Exit Code |
|----------|----------|-----------|
| File doesn't exist | Fail-fast with FileNotFoundError | 1 |
| JSON corrupted | Fail-fast with ValueError | 2 |
| Missing `top_3_buckets` field | Fail-fast with ValueError | 3 |
| `top_3_buckets` wrong type | Fail-fast with TypeError | 4 |
| `top_3_buckets` empty list | Fail-fast with ValueError | 5 |

**For HLD Section**: 3.1 (Input Dependencies), 6.1 (Input Validation), 6.2 (Error Cases)

**Notes**:
- This approach prevents silent failures where Stage 2.5 might organize files incorrectly
- Error messages provide actionable next steps for users
- Strict validation ensures upstream stages completed successfully

### Dependencies & Integration

#### Q11: [HIGH] Duplicate video_id Handling
**Question**: What should Stage 2.5 do if it encounters the same `video_id` in multiple bucket checkpoints?

**Answer**: **Option A - Fail-fast on duplicate detection**

**Implementation**:
```python
def detect_duplicates_across_buckets(files_to_process):
    """
    Detect if same video_id appears in multiple buckets.

    In normal operation, each video has exactly one duration and therefore belongs
    to exactly one bucket. If a video_id appears in multiple buckets, this indicates
    checkpoint corruption or a Stage 2 bug.

    Args:
        files_to_process: list of dict, each with keys: 'video_id', 'bucket', 'source_path'

    Raises:
        ValueError: if duplicate video_id detected across buckets
    """
    video_id_to_buckets = {}

    for file_info in files_to_process:
        video_id = file_info['video_id']
        bucket = file_info['bucket']

        if video_id in video_id_to_buckets:
            # Duplicate detected - fail-fast
            previous_bucket = video_id_to_buckets[video_id]
            raise ValueError(
                f"Video ID '{video_id}' appears in multiple buckets:\n"
                f"  - Bucket: {previous_bucket}\n"
                f"  - Bucket: {bucket}\n\n"
                f"This indicates checkpoint corruption or Stage 2 bug.\n"
                f"Each video should belong to exactly one bucket based on its duration.\n\n"
                f"Diagnosis:\n"
                f"  1. Check {previous_bucket}/checkpoints/stage_2_checkpoint.json\n"
                f"  2. Check {bucket}/checkpoints/stage_2_checkpoint.json\n"
                f"  3. Verify video {video_id} duration in source file\n\n"
                f"Solutions:\n"
                f"  1. Re-run Stage 2 to regenerate checkpoints\n"
                f"  2. Manually inspect checkpoints and remove duplicate entries\n"
                f"  3. Report bug if issue persists"
            )

        video_id_to_buckets[video_id] = bucket

    # No duplicates detected
    logger.info(f"Validation passed: {len(video_id_to_buckets)} unique videos across {len(set(f['bucket'] for f in files_to_process))} buckets")
```

**Rationale**:
- ✅ **Catches data corruption early**: Prevents organizing same file to multiple locations
- ✅ **Fail-fast with clear diagnosis**: Error message explains what went wrong and how to fix
- ✅ **Simple logic**: Easy to understand and maintain
- ✅ **Enforces invariant**: Each video belongs to exactly one bucket (by duration)
- ✅ **Forces root cause investigation**: Duplicates should never happen in normal operation

**Why This Should Never Happen**:
```
Video duration → assign_bucket() → Single bucket assignment

Example:
  - Video ID "123" has duration = 25.3s
  - assign_bucket(25.3) → "18-33s" (deterministic)
  - Video "123" should ONLY appear in bucket_18-33s checkpoint
  - If also appears in bucket_33-60s checkpoint → corruption or bug
```

**Error Message Example**:
```
ValueError: Video ID '7428596413707144481' appears in multiple buckets:
  - Bucket: 18-33s
  - Bucket: 33-60s

This indicates checkpoint corruption or Stage 2 bug.
Each video should belong to exactly one bucket based on its duration.

Diagnosis:
  1. Check 18-33s/checkpoints/stage_2_checkpoint.json
  2. Check 33-60s/checkpoints/stage_2_checkpoint.json
  3. Verify video 7428596413707144481 duration in source file

Solutions:
  1. Re-run Stage 2 to regenerate checkpoints
  2. Manually inspect checkpoints and remove duplicate entries
  3. Report bug if issue persists
```

**When This Might Occur**:
| Scenario | Root Cause | Solution |
|----------|-----------|----------|
| Manual checkpoint editing | User manually edited checkpoint files | Re-run Stage 2 |
| Checkpoint corruption | Disk I/O error during checkpoint write | Re-run Stage 2 |
| Stage 2 bug | Bug in `assign_bucket()` or checkpoint logic | Report bug, re-run Stage 2 |
| Race condition | Parallel Stage 2 runs (shouldn't happen - sequential design) | Ensure only one Stage 2 instance runs |

**For HLD Section**: 6.1 (Input Validation), 6.2 (Error Cases), 2.3 (Detailed Process)

**Notes**:
- This validation should run AFTER loading all checkpoints, BEFORE processing files
- Catches corruption before any file operations occur
- Provides clear diagnostic information for troubleshooting
- Enforces critical data integrity constraint

#### Q12: [HIGH] Operation Atomicity & Partial Failure
**Question**: How should Stage 2.5 handle partial completion and support resume if interrupted mid-process?

**Answer**: **Option C - Detection-based resume (no checkpoint)**

**Implementation**:
```python
def organize_files_with_detection(files_to_process):
    """
    Organize files with automatic resume detection (no checkpoint needed).

    Detection strategy:
    1. Check if source file exists in /insights/
    2. Check if target file exists in bucket directory
    3. If target exists but source doesn't → already moved (skip)
    4. If both exist → source takes precedence (move again, may overwrite)
    5. If neither exists → warn and skip (Stage 2 may have failed)

    This approach makes Stage 2.5 idempotent - safe to re-run multiple times.

    Args:
        files_to_process: list of dict with keys: 'video_id', 'bucket', 'source_path', 'target_path'

    Returns:
        dict: Summary statistics (moved_count, skipped_count, missing_count)
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

        # Case 1: Already moved in previous run (target exists, source doesn't)
        if target_exists and not source_exists:
            logger.debug(f"Already organized: {video_id} → {bucket}")
            skipped_already_organized += 1
            continue

        # Case 2: Missing entirely (neither source nor target exists)
        if not source_exists and not target_exists:
            logger.warning(
                f"Missing source and target for video {video_id}. "
                f"Stage 2 checkpoint indicated completion, but file doesn't exist. "
                f"Source: {source}"
            )
            missing_count += 1
            continue

        # Case 3: Source exists (move it, regardless of target existence)
        # If target exists, this overwrites (handles interrupted moves)
        try:
            # Ensure target directory exists
            target_dir = os.path.dirname(target)
            os.makedirs(target_dir, exist_ok=True)

            # Move file (atomic operation within same filesystem)
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
        logger.warning(
            f"{missing_count} files missing despite checkpoint indicating completion. "
            f"Check Stage 2 logs."
        )

    return {
        'moved_count': moved_count,
        'skipped_already_organized': skipped_already_organized,
        'missing_count': missing_count,
        'total_processed': total_processed
    }
```

**Rationale**:
- ✅ **No checkpoint management**: Simplifies implementation (no additional files to manage)
- ✅ **Idempotent**: Safe to re-run multiple times without side effects
- ✅ **Automatic resume detection**: Detects previous progress via filesystem state
- ✅ **Fast recovery**: Interruptions only lose current file being moved (at most 1 file)
- ✅ **Simple logic**: Easy to understand and maintain
- ✅ **No checkpoint I/O overhead**: No writes after every file
- ✅ **Aligns with Stage 2.5 nature**: One-time organization step, fast enough to re-run

**Resume Behavior Examples**:

**Scenario 1: Fresh Run**
```
Total files: 300
  - All 300 files in /insights/
  - No files in bucket directories
Result: Move all 300 files
```

**Scenario 2: Interrupted After 150 Files**
```
Total files: 300
  - 150 files already in bucket directories (from previous run)
  - 150 files still in /insights/
Result: Skip 150 already organized, move remaining 150
```

**Scenario 3: Complete Re-Run**
```
Total files: 300
  - All 300 files already in bucket directories
  - No files in /insights/
Result: Skip all 300 (already organized)
```

**Scenario 4: Source File Missing**
```
File: video_123
  - Source: /insights/video_123_temporal_windows_updated.json (doesn't exist)
  - Target: bucket_18-33s/analysis/insights/video_123_temporal_windows_updated.json (doesn't exist)
Result: Warning logged, skip file, continue processing
```

**Atomicity Characteristics**:
| Operation | Atomic? | Failure Mode | Recovery |
|-----------|---------|--------------|----------|
| `shutil.move()` within same filesystem | ✅ Yes (rename syscall) | Power loss → file in source OR target, never lost | Re-run Stage 2.5 |
| `shutil.move()` across filesystems | ❌ No (copy + delete) | Power loss → file may exist in both locations | Re-run Stage 2.5 (overwrites target) |
| Directory creation (`os.makedirs`) | ✅ Yes | Power loss → directory exists or doesn't | Idempotent with `exist_ok=True` |

**Edge Cases Handled**:
| Scenario | Detection | Handling |
|----------|-----------|----------|
| Already organized (target exists, source missing) | `target_exists and not source_exists` | Skip with debug log |
| Partial move (both exist) | `source_exists and target_exists` | Re-move (overwrites target) |
| Missing entirely (neither exists) | `not source_exists and not target_exists` | Warning log, skip |
| Move failure (permissions, disk full) | Exception caught | Error log, continue processing others |

**For HLD Section**: 2.3 (Detailed Process), 6.2 (Error Cases), 7.1 (Performance)

**Notes**:
- This approach makes Stage 2.5 a pure function of filesystem state (no hidden checkpoint state)
- Stage 2.5 is fast enough (~1-2 seconds for 300 files) that re-running isn't expensive
- Detection-based resume is simpler than checkpoint management for this use case
- Users can safely re-run Stage 2.5 after interruptions without manual intervention

### Edge Cases & Validation

### Performance & Scale

### Error Handling

### Testing

## Completeness Check

### Questions Answered: 12 Total

#### [CRITICAL] Questions (8/8 Complete)
- ✅ Q1: Input Source Directory & File Pattern
- ✅ Q2: Output Directory Path Construction
- ✅ Q3: Stage 2.5 Execution Scope
- ✅ Q4: Non-Winning Bucket Video Handling
- ✅ Q5: Winning Buckets Discovery
- ✅ Q6: File Operation Type (Move vs Copy)
- ✅ Q7: Directory Creation & Permissions
- ✅ Q8: Source Input File Pattern Matching

#### [HIGH] Questions (4/4 Complete)
- ✅ Q9: Stage 2 Checkpoint Schema Dependency
- ✅ Q10: Missing winner_analysis.json Handling
- ✅ Q11: Duplicate video_id Handling
- ✅ Q12: Operation Atomicity & Partial Failure

### Coverage Assessment

#### Input/Output Contracts: ✅ COMPLETE
- Q1: Input source path and filename pattern
- Q2: Output path construction (full analysis-scoped path)
- Q8: File discovery mechanism (checkpoint-driven)

#### Dependencies & Integration: ✅ COMPLETE
- Q5: winner_analysis.json dependency
- Q7: Directory creation responsibilities (Stage 2)
- Q9: Checkpoint schema validation
- Q10: winner_analysis.json error handling

#### Edge Cases & Validation: ✅ COMPLETE
- Q4: Non-winning bucket handling (skip with warning)
- Q11: Duplicate video_id detection (fail-fast)
- Q12: Partial failure recovery (detection-based resume)

#### Error Handling: ✅ COMPLETE
- Q9: Checkpoint corruption handling
- Q10: Missing winner_analysis.json handling
- Q11: Duplicate detection with diagnostic messages
- Q12: Move failure handling (continue processing)

#### Performance & Scale: ⚠️ PARTIAL
- Q3: Execution scope (once per analysis, all buckets)
- Q6: Move operation (no duplicate storage)
- Q12: Performance notes (~1-2s for 300 files)
- ❓ Missing: Disk I/O optimization, parallel processing considerations

#### Testing: ❌ NOT COVERED
- No specific testing questions asked
- Will be addressed in HLD Section 8 (Testing Strategy)

### Critical Decisions Summary

| Decision Point | Answer | Rationale |
|----------------|--------|-----------|
| **Input location** | `/home/jorge/rumiaifinal/insights/` | Single flat directory confirmed |
| **Output structure** | Analysis-scoped path | Supports multi-analysis coexistence |
| **Execution model** | Once per analysis, all buckets | Batch operation, efficient |
| **File operation** | MOVE (delete source) | No duplicate storage, single source of truth |
| **Directory creation** | Stage 2 creates, Stage 2.5 validates | Clear responsibility, Q7 confirmed |
| **File discovery** | Checkpoint-driven (Option B) | Exact match with Stage 2 completions |
| **Resume strategy** | Detection-based (Option C) | Idempotent, no checkpoint overhead |
| **Error philosophy** | Fail-fast for corruption, graceful for missing files | Data integrity priority |

### Ready for Phase 3?

✅ **YES** - All critical questions answered

**Readiness Checklist**:
- ✅ Input contracts defined (Q1, Q8)
- ✅ Output contracts defined (Q2, Q7)
- ✅ Dependencies identified (Q5, Q9, Q10)
- ✅ Error handling strategy defined (Q9, Q10, Q11, Q12)
- ✅ Edge cases covered (Q4, Q11, Q12)
- ✅ Integration points clear (Q3, Q6, Q7)

**Phase 3 Notes**:
- Performance section may need expansion during HLD writing
- Testing strategy will be developed in HLD Section 8
- All critical architectural decisions documented

## Proceed to Phase 3

**Status**: ✅ READY

**Next Steps**:
1. Review QA document for any gaps
2. Create HLD document following ChildTemplate.md structure
3. Reference QA answers in corresponding HLD sections
4. Expand on performance and testing in HLD

**HLD Sections to Populate from QA**:
- Section 2.3 (Detailed Process): Q3, Q6, Q8, Q12
- Section 3.1 (Input Dependencies): Q1, Q5, Q8, Q9, Q10
- Section 3.2 (Output Contracts): Q2, Q7
- Section 5.1 (Input Schema): Q1, Q5, Q9
- Section 5.2 (Output Schema): Q2
- Section 6.1 (Input Validation): Q9, Q10, Q11
- Section 6.2 (Error Cases): Q4, Q9, Q10, Q11, Q12
- Section 7 (Performance): Q3, Q12
- Section 8 (Testing): New content based on QA insights
