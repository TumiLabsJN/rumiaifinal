# E2E Test 3 Issues & Fixes Documentation

**Date Created:** 2025-10-29
**Test:** Test 3 (180-Minute Scrape Delay + 100 Videos)
**Client:** rollo_test3
**Status:** Issues identified, fixes pending implementation

---

## Executive Summary

Test 3 experienced checkpoint corruption causing 60 out of 126 videos to fail processing despite the checkpoint claiming 100% completion. This document identifies the root causes and proposes comprehensive fixes.

### Key Metrics:
- **Total videos selected:** 126 (40 + 39 + 47 across 3 buckets)
- **Actually processed:** 66 (3 + 16 + 47)
- **Missing:** 60 videos (37 + 23 + 0)
- **Checkpoint claimed:** 126/126 completed ❌ FALSE
- **Test 4 has all missing videos:** ✅ YES (100% overlap)

---

## ISSUE 1: Missing Video Outputs - Recovery via Test 4

### Problem Statement

Test 3 failed to process 60 videos across 2 buckets, yet the checkpoint incorrectly reported 100% completion. The pipeline proceeded to Stage 2.5+ based on false completion data.

### Missing Videos Breakdown

| Bucket | Selected | Processed | Missing | Test 4 Has |
|--------|----------|-----------|---------|------------|
| **3-9s** | 40 | 3 (7.5%) | 37 | ✅ 37/37 |
| **60-90s** | 39 | 16 (41%) | 23 | ✅ 23/23 |
| **18-33s** | 47 | 47 (100%) | 0 | N/A |
| **TOTAL** | **126** | **66 (52%)** | **60** | **✅ 60/60** |

### Root Cause Analysis

**Why did Test 4 have Test 3's videos?**

Both tests scraped from the same hashtag cluster (`wellness` in Test 2 vs `wellness_test3` in Test 3) within overlapping timeframes. The scraper returned overlapping video sets:

- Test 3 scraped: 933 unique videos (after dedup)
- Test 4 scraped: Similar pool with different `video_count` parameter
- **Overlap:** Test 3 selected 40 videos for bucket 3-9s, Test 4 selected 59 videos (37 in common)

### Verification Evidence

```bash
# Verified Test 4 has exact video IDs Test 3 needs:
Bucket 3-9s overlap: 37/37 videos (100%)
Bucket 60-90s overlap: 23/23 videos (100%)
Total overlap: 60/60 videos (100%)

# Spot check verification:
Video 7468418161742810398 (missing from Test 3):
  ✅ Test 4 has video file: bucket_3-9s/videos/7468418161742810398.mp4
  ✅ Test 4 has insights: bucket_3-9s/analysis/insights/7468418161742810398_temporal_windows_updated.json
  ✅ Test 4 has unified: bucket_3-9s/analysis/unified/7468418161742810398.json
```

---

### Proposed Solutions

#### **Option A: Copy Processed Outputs from Test 4 → Test 3** ⭐ RECOMMENDED

Copy all ML-processed outputs from Test 4 to Test 3 for the 60 missing videos.

**Advantages:**
- ✅ Saves ~60 hours of reprocessing time
- ✅ Reuses valid, already-processed ML outputs
- ✅ Same video IDs = safe to copy
- ✅ Test 4 outputs are fresh (processed after Test 3 failure)

**Disadvantages:**
- ⚠️ Requires careful file copying (manual process)
- ⚠️ Needs checkpoint correction after copy

**Files to Copy per Video:**

```bash
# For each of the 60 missing videos:
Source: /data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive/buckets/bucket_X/
Target: /data/clients/rollo_test3/hashtags/wellness_test3/top_contrastive/buckets/bucket_X/

Files:
1. videos/{video_id}.mp4
2. analysis/insights/{video_id}_temporal_windows_updated.json
3. analysis/unified/{video_id}.json
```

**Implementation Steps:**

1. **Generate copy script from missing video lists:**
   ```bash
   # Lists already generated:
   /tmp/test3_missing_3_9s.txt (37 videos)
   /tmp/test3_missing_60_90s.txt (23 videos)

   # Create bash script to copy files
   ```

2. **Verify Test 4 files exist before copying**
3. **Copy files to Test 3 bucket directories**
4. **Verify copied file integrity (checksums)**
5. **Update Test 3 checkpoint** (see ISSUE 3 for checkpoint strategy)

**Risk Assessment:** LOW
- Videos are identical (same IDs from same scrape pool)
- Test 4 processed after Test 3, so outputs are newer
- No processing differences between tests (same ML pipeline)

---

#### **Option B: Re-download from TikTok and Reprocess**

Delete checkpoint and rerun Stage 2 to re-download all 60 videos from TikTok.

**Advantages:**
- ✅ Simplest approach (no manual copying)
- ✅ Ensures fresh data from source
- ✅ Checkpoint will be accurate

**Disadvantages:**
- ❌ Takes ~60 hours to reprocess 60 videos
- ❌ Videos may be deleted from TikTok since original scrape
- ❌ Redundant work (videos already processed in Test 4)

**When to Use:**
- If Option A fails validation
- If Test 4 outputs are suspect
- If we need to verify TikTok videos still exist

---

#### **Option C: Hybrid - Copy Videos, Reprocess ML**

Copy only `.mp4` files from Test 4, then reprocess through ML pipeline.

**Advantages:**
- ✅ Avoids TikTok re-download (faster than Option B)
- ✅ Fresh ML processing ensures consistency
- ✅ Cleaner checkpoint (will auto-regenerate)

**Disadvantages:**
- ❌ Still takes ~50 hours for ML processing
- ❌ Redundant ML work

**When to Use:**
- If we don't trust Test 4's ML outputs
- If ML pipeline has changed since Test 4

---

### Recommendation

**Use Option A (Copy from Test 4)** for the following reasons:

1. **Time Critical:** Saves 60 hours
2. **Data Integrity:** Test 4 outputs verified identical to expected schema
3. **Low Risk:** Same videos, same ML pipeline, verified overlap
4. **Practical:** Test 3 is waiting for manual taxonomy curation anyway

**Fallback:** If Option A reveals issues, fall back to Option C (copy videos, reprocess ML).

---

## ISSUE 2: Checkpoint Corruption & Validation Failures

### Point 1: Checkpoint Corruption Root Cause

#### The Mystery

Test 3's checkpoint for bucket_3-9s was populated with **40 video IDs from Test 4**, not Test 3's actual videos.

**Evidence:**
```bash
# Test 3 checkpoint claims (WRONG):
completed_video_ids: [7473258825550892334, 7478440226415709471, ...] (40 IDs)

# Test 3 actually processed (CORRECT):
videos/: [7474696049303522591, 7489929709598379306, 7560005211549469982] (3 files)

# Where those checkpoint IDs exist:
rollo_test4/hashtags/wellness_test4/.../bucket_3-9s/videos/ ✅ FOUND
```

#### Hypotheses for Investigation

##### **Hypothesis 1: Concurrent Test Execution**

**Theory:** Test 3 and Test 4 ran simultaneously, causing checkpoint file race condition.

**Evidence to Investigate:**
- [ ] Check Test 3 and Test 4 run timestamps from logs
- [ ] Compare `last_checkpoint` timestamps in checkpoint files
- [ ] Look for overlapping execution windows
- [ ] Check if same terminal/process ran both tests

**Investigation Steps:**
```bash
# Check Test 3 execution timeline
grep "Starting Stage 2 for bucket: 3-9s" data/logs/*Rollo_Test3*.log

# Check Test 4 execution timeline
grep "Starting Stage 2 for bucket: 3-9s" data/logs/*Rollo_Test4*.log

# Compare checkpoint timestamps
cat data/clients/rollo_test3/.../bucket_3-9s/checkpoints/stage_2_checkpoint.json | jq '.last_checkpoint'
cat data/clients/rollo_test4/.../bucket_3-9s/checkpoints/stage_2_checkpoint.json | jq '.last_checkpoint'
```

**If Confirmed:** Add mutex/lock mechanism to prevent concurrent checkpoint writes.

---

##### **Hypothesis 2: Checkpoint Path Resolution Bug**

**Theory:** Checkpoint saving logic constructed wrong path, writing Test 3 checkpoint to Test 4 location (or vice versa).

**Evidence to Investigate:**
- [ ] Review `get_bucket_path()` function in `stage2_processing/utils.py`
- [ ] Check if `config['client_id']` was correct during Test 3 execution
- [ ] Look for hardcoded paths in checkpoint code
- [ ] Check if environment variables affected path construction

**Investigation Steps:**
```python
# Review checkpoint path construction
# File: ml_pipeline/stage2_processing/checkpoint.py:141
bucket_path = get_bucket_path(config, bucket_name)
checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

# Questions:
# - Was config['client_id'] = "rollo_test3" or "rollo_test4"?
# - Did get_bucket_path() return correct path?
# - Were there any symlinks or path aliases?
```

**Code to Review:**
- `ml_pipeline/stage2_processing/utils.py::get_bucket_path()`
- `ml_pipeline/stage2_processing/checkpoint.py::save_checkpoint_with_backup()`
- `rumiai_ml_batch.py` - config initialization

**If Confirmed:** Add path validation before checkpoint save, log full paths.

---

##### **Hypothesis 3: Copy-Paste Checkpoint from Test 4**

**Theory:** Manual intervention - someone copied Test 4's checkpoint to Test 3 for debugging.

**Evidence to Investigate:**
- [ ] Check file modification timestamps (Test 3 vs Test 4 checkpoints)
- [ ] Review checkpoint backup files (`.backup.json`)
- [ ] Check git history for manual checkpoint edits
- [ ] Ask team if manual checkpoint copying occurred

**Investigation Steps:**
```bash
# Compare file timestamps
stat data/clients/rollo_test3/.../checkpoints/stage_2_checkpoint.json
stat data/clients/rollo_test4/.../checkpoints/stage_2_checkpoint.json

# Check backup timestamps
stat data/clients/rollo_test3/.../checkpoints/stage_2_checkpoint.backup.json
```

**If Confirmed:** Add documentation warning against manual checkpoint editing.

---

##### **Hypothesis 4: Checkpoint Resume Logic Error**

**Theory:** Checkpoint resume detected existing Test 4 checkpoint when initializing Test 3.

**Evidence to Investigate:**
- [ ] Check if `DATA_ROOT` or path environment variables were incorrect
- [ ] Review `initialize_checkpoint()` logic for path validation
- [ ] Check if checkpoint files were in unexpected locations
- [ ] Look for symlinks between Test 3 and Test 4 directories

**Investigation Steps:**
```python
# Review checkpoint initialization logic
# File: ml_pipeline/stage2_processing/checkpoint.py:144-155
if os.path.exists(checkpoint_path):
    checkpoint = load_checkpoint_with_recovery(checkpoint_path)
    # Did this load Test 4's checkpoint by mistake?
```

**If Confirmed:** Add checkpoint ownership validation (verify `config['client_id']` matches checkpoint).

---

##### **Hypothesis 5: Stage 2.5 File Organization Bug**

**Theory:** Stage 2.5 moved Test 4's checkpoint files into Test 3's directory.

**Evidence to Investigate:**
- [ ] Check Stage 2.5 logs for unexpected file moves
- [ ] Review `file_organizer.py` for global file search logic
- [ ] Check if Stage 2.5 ran across multiple test directories
- [ ] Look for warnings about "180 files missing" (already found in logs)

**Investigation Steps:**
```bash
# Stage 2.5 already logged the issue:
# "180 files missing despite checkpoint indicating completion."

# Review file organization logic
# File: ml_pipeline/stage2_5_organize/file_organizer.py:18-22
SOURCE_DIRS = {
    'temporal_windows': '/home/jorge/rumiaifinal/insights/',
    'videos': '/home/jorge/rumiaifinal/temp/',
    'unified_analysis': '/home/jorge/rumiaifinal/unified_analysis/'
}
# These are GLOBAL paths - could organize files from any test!
```

**If Confirmed:** This is likely a symptom, not the cause. But Stage 2.5 should validate checkpoint ownership.

---

#### Investigation Priority

**High Priority:**
1. Hypothesis 2 (Path Resolution Bug) - Most likely, easiest to fix
2. Hypothesis 1 (Concurrent Execution) - High impact if true

**Medium Priority:**
3. Hypothesis 4 (Resume Logic Error) - Possible configuration issue
4. Hypothesis 5 (Stage 2.5 Bug) - Already has warnings, needs improvement

**Low Priority:**
5. Hypothesis 3 (Manual Intervention) - Unlikely, but check timestamps

---

### Point 2: Stage 2 Validation Must Be Improved

#### Current Validation Gaps

**What Stage 2 Currently Validates:**
```python
# File: ml_pipeline/stage2_processing/main.py:139-184
def validate_stage_output(bucket_name: str, checkpoint: dict, config: dict):
    # 1. Check insights directory exists
    assert os.path.exists(insights_dir)

    # 2. Check each completed video has insights file at GLOBAL location
    for video_id in checkpoint['completed_video_ids']:
        insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
        if not os.path.exists(insights_path):
            missing_files.append(video_id)

    assert len(missing_files) == 0

    # 3. Check checkpoint saved
    assert os.path.exists(checkpoint_path)
```

**Problems:**
1. ❌ Only checks if files exist in `/home/jorge/rumiaifinal/insights/` (GLOBAL directory)
2. ❌ Doesn't verify files belong to THIS test (could be from Test 4!)
3. ❌ Doesn't compare `checkpoint['completed']` count vs actual file count
4. ❌ Doesn't validate bucket-specific output directories
5. ❌ Validation happens BEFORE Stage 2.5 moves files to bucket directories

---

#### Proposed Validation Improvements

##### **Enhancement 1: Count-Based Validation**

**Requirement:** Verify that the number of completed videos in checkpoint matches actual processed files.

```python
def validate_stage_output(bucket_name: str, checkpoint: dict, config: dict):
    """Enhanced validation with count verification."""

    # Current validation (file existence check)
    # ...existing code...

    # NEW: Count-based validation
    expected_count = checkpoint['completed']
    actual_files = len([f for f in os.listdir(insights_dir)
                       if f.endswith('_temporal_windows_updated.json')])

    if actual_files < expected_count:
        raise ValidationError(
            f"Checkpoint claims {expected_count} videos completed, "
            f"but only {actual_files} insights files found. "
            f"Missing: {expected_count - actual_files} files."
        )
```

**Advantages:**
- ✅ Catches checkpoint vs reality mismatches
- ✅ Simple to implement
- ✅ Works with current GLOBAL directory structure

**Limitations:**
- ⚠️ Still doesn't verify files belong to THIS test
- ⚠️ May have false positives if multiple tests run concurrently

---

##### **Enhancement 2: Bucket-Specific Validation (Post Stage 2.5)**

**Requirement:** After Stage 2.5 organizes files, verify each bucket has correct number of insights.

```python
def validate_bucket_completion(bucket_name: str, bucket_path: str, expected_count: int):
    """
    Validate bucket has correct number of processed videos AFTER Stage 2.5.

    This catches cases where Stage 2 validation passed but files weren't organized.
    """

    # Check bucket-specific insights directory
    insights_path = f"{bucket_path}analysis/insights/"

    if not os.path.exists(insights_path):
        raise ValidationError(f"Insights directory missing for bucket {bucket_name}")

    # Count actual insights files
    insights_files = [f for f in os.listdir(insights_path)
                     if f.endswith('_temporal_windows_updated.json')]
    actual_count = len(insights_files)

    if actual_count != expected_count:
        raise ValidationError(
            f"Bucket {bucket_name}: Expected {expected_count} insights files, "
            f"found {actual_count}. Missing: {expected_count - actual_count}"
        )

    logger.info(f"✓ Bucket {bucket_name} validation passed: {actual_count}/{expected_count} insights")
```

**Where to Add:**
- **Option A:** End of Stage 2.5 (file organization)
- **Option B:** Start of Stage 2.6 (pattern discovery)

**Recommended:** Add to Stage 2.5 validation (`ml_pipeline/stage2_5_organize/validation.py`)

---

##### **Enhancement 3: Checkpoint Ownership Validation**

**Requirement:** Verify checkpoint belongs to the current test run.

```python
def validate_checkpoint_ownership(checkpoint: dict, config: dict):
    """
    Verify checkpoint matches current test configuration.

    Prevents loading checkpoints from different tests.
    """

    # Check client_id matches
    checkpoint_client = checkpoint.get('config', {}).get('client_id')
    current_client = config.get('client_id')

    if checkpoint_client != current_client:
        raise CheckpointOwnershipError(
            f"Checkpoint belongs to different client: "
            f"checkpoint='{checkpoint_client}', current='{current_client}'. "
            f"This indicates checkpoint corruption or wrong directory."
        )

    # Check target matches (hashtag cluster name)
    checkpoint_target = checkpoint.get('config', {}).get('target')
    current_target = config.get('target')

    if checkpoint_target != current_target:
        raise CheckpointOwnershipError(
            f"Checkpoint belongs to different target: "
            f"checkpoint='{checkpoint_target}', current='{current_target}'"
        )

    # Log checkpoint provenance for debugging
    logger.info(f"✓ Checkpoint ownership validated: {checkpoint_client}/{checkpoint_target}")
```

**Where to Add:** `ml_pipeline/stage2_processing/checkpoint.py::initialize_checkpoint()` (line 146, after loading)

---

##### **Enhancement 4: Video ID Provenance Check**

**Requirement:** Verify that video IDs in checkpoint match video IDs in `selected_videos.json`.

```python
def validate_checkpoint_video_ids(checkpoint: dict, selected_videos: list):
    """
    Verify checkpoint video IDs are subset of selected videos.

    Catches cases where checkpoint has IDs from different test.
    """

    selected_ids = set(v['id'] for v in selected_videos)
    checkpoint_ids = set(checkpoint['completed_video_ids'])

    # Check for IDs in checkpoint that weren't selected
    unexpected_ids = checkpoint_ids - selected_ids

    if unexpected_ids:
        raise CheckpointIntegrityError(
            f"Checkpoint contains {len(unexpected_ids)} video IDs that were NOT selected: "
            f"{list(unexpected_ids)[:5]}... "
            f"This indicates checkpoint corruption or cross-test contamination."
        )

    logger.info(f"✓ Checkpoint video IDs validated: {len(checkpoint_ids)} IDs match selection")
```

**Where to Add:** `ml_pipeline/stage2_processing/checkpoint.py::initialize_checkpoint()` (line 147, after loading)

---

#### Validation Implementation Summary

| Enhancement | Priority | Location | Catches |
|-------------|----------|----------|---------|
| **1. Count-Based** | HIGH | `stage2_processing/main.py::validate_stage_output()` | Checkpoint vs file count mismatch |
| **2. Bucket-Specific** | HIGH | `stage2_5_organize/validation.py` | Post-organization file verification |
| **3. Ownership** | CRITICAL | `stage2_processing/checkpoint.py::initialize_checkpoint()` | Cross-test checkpoint contamination |
| **4. Provenance** | CRITICAL | `stage2_processing/checkpoint.py::initialize_checkpoint()` | Video ID mismatches |

**Implementation Order:**
1. Enhancement 3 (Ownership) - Prevents loading wrong checkpoints
2. Enhancement 4 (Provenance) - Catches ID mismatches immediately
3. Enhancement 1 (Count) - Validates at Stage 2 completion
4. Enhancement 2 (Bucket) - Final validation after organization

---

## ISSUE 3: Checkpoint Resume Behavior & Data Loss Risk

### Current Checkpoint Behavior

#### How Checkpoint Resume Works

**File:** `ml_pipeline/stage2_processing/checkpoint.py:100-178`

```python
def initialize_checkpoint(bucket_name: str, video_list: list, config: dict):
    """
    Initialize or resume checkpoint.

    AUTO-RESUME LOGIC:
    1. Check if checkpoint exists
    2. If exists: Load checkpoint, validate config match
    3. Extract completed_video_ids from checkpoint
    4. Filter video_list to get remaining videos
    5. Return (checkpoint, remaining_videos)

    If checkpoint does NOT exist:
    1. Create new checkpoint with all videos as "remaining"
    2. Return (new_checkpoint, all_videos)
    """

    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    if os.path.exists(checkpoint_path):
        # AUTO-RESUME from existing checkpoint
        checkpoint = load_checkpoint_with_recovery(checkpoint_path)
        completed_ids = set(checkpoint['completed_video_ids'])
        remaining_videos = [v for v in video_list if v['id'] not in completed_ids]

        return checkpoint, remaining_videos  # Only process remaining
    else:
        # Create NEW checkpoint
        checkpoint = {
            "completed": 0,
            "completed_video_ids": [],
            # ...
        }
        return checkpoint, video_list  # Process all videos
```

---

### The Data Loss Question

#### Scenario: Delete Corrupted Checkpoint

**User's Concern:**
> "If we delete corrupted checkpoint, does this mean we will reprocess ALL Videos? We would lose work."

**Answer:** YES - But only if we delete the checkpoint AND don't preserve the processed files.

---

### Current Risk: Deleting Checkpoint = Losing Progress

**What Happens if We Delete `stage_2_checkpoint.json`:**

```bash
# Current state:
bucket_3-9s/
├── checkpoints/
│   └── stage_2_checkpoint.json (CORRUPTED - claims 40/40 completed)
├── videos/
│   ├── 7474696049303522591.mp4 (3 videos actually processed)
│   ├── 7489929709598379306.mp4
│   └── 7560005211549469982.mp4
└── analysis/insights/
    ├── 7474696049303522591_temporal_windows_updated.json
    ├── 7489929709598379306_temporal_windows_updated.json
    └── 7560005211549469982_temporal_windows_updated.json

# If we delete checkpoint:
rm bucket_3-9s/checkpoints/stage_2_checkpoint.json

# Next run:
python rumiai_ml_batch.py ...
# → initialize_checkpoint() finds NO checkpoint
# → Creates NEW checkpoint with completed: 0, completed_video_ids: []
# → Returns remaining_videos = ALL 40 videos
# → Will attempt to reprocess ALL 40 videos (including the 3 already done)
```

**Result:** ❌ The 3 already-processed videos would be reprocessed (duplicating work).

---

### Why Current Resume Logic Can't Handle This

**Problem:** Resume logic only looks at checkpoint, not actual file system.

```python
# Current logic (checkpoint.py:149-150):
completed_ids = set(checkpoint['completed_video_ids'])
remaining_videos = [v for v in video_list if v['id'] not in completed_ids]

# If checkpoint is corrupted with WRONG IDs:
# - checkpoint['completed_video_ids'] = [Test 4 IDs] (WRONG)
# - remaining_videos = ALL Test 3 videos (WRONG - ignores 3 already processed)
```

**Current logic assumes:** Checkpoint is always correct.

**Reality:** Checkpoint can be corrupted.

---

### Proposed Solutions

#### **Option A: Detection-Based Resume** ⭐ RECOMMENDED

**Concept:** Resume based on ACTUAL files in file system, not just checkpoint.

```python
def initialize_checkpoint_with_detection(bucket_name: str, video_list: list, config: dict):
    """
    Enhanced checkpoint initialization with file-system detection.

    DETECTION LOGIC:
    1. Check if checkpoint exists
    2. Check what files actually exist in bucket directories
    3. Cross-reference checkpoint vs actual files
    4. Resume based on ACTUAL processed files (not just checkpoint)
    """

    bucket_path = get_bucket_path(config, bucket_name)
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    # Step 1: Detect actually processed videos from file system
    insights_dir = f"{bucket_path}analysis/insights/"
    actually_processed = set()

    if os.path.exists(insights_dir):
        for filename in os.listdir(insights_dir):
            if filename.endswith('_temporal_windows_updated.json'):
                video_id = filename.replace('_temporal_windows_updated.json', '')
                actually_processed.add(video_id)

    logger.info(f"Detected {len(actually_processed)} actually processed videos in file system")

    # Step 2: Load or create checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint_with_recovery(checkpoint_path)
        checkpoint_claimed = set(checkpoint['completed_video_ids'])

        # Step 3: Validate checkpoint against reality
        if checkpoint_claimed != actually_processed:
            discrepancy_count = len(checkpoint_claimed.symmetric_difference(actually_processed))
            logger.warning(
                f"⚠️  Checkpoint discrepancy detected: "
                f"{len(checkpoint_claimed)} claimed vs {len(actually_processed)} actual. "
                f"Discrepancy: {discrepancy_count} videos."
            )

            # DECISION: Trust file system over corrupted checkpoint
            logger.info("Using file system detection to determine resume point")
            checkpoint['completed_video_ids'] = list(actually_processed)
            checkpoint['completed'] = len(actually_processed)
            checkpoint['failed_video_ids'] = []  # Reset failed (unknown from detection)
            checkpoint['remaining'] = len(video_list) - len(actually_processed)

            # Save corrected checkpoint
            save_checkpoint_with_backup(checkpoint_path, checkpoint)
            logger.info("✓ Checkpoint corrected based on file system detection")
    else:
        # No checkpoint - but files may exist from previous run
        if actually_processed:
            logger.info(f"No checkpoint found, but {len(actually_processed)} processed files detected")
            logger.info("Creating checkpoint from file system detection")

        checkpoint = {
            "stage": "video_processing",
            "bucket": bucket_name,
            "total_videos": len(video_list),
            "completed": len(actually_processed),
            "failed": 0,
            "remaining": len(video_list) - len(actually_processed),
            "completed_video_ids": list(actually_processed),
            "failed_video_ids": [],
            "config": config,
            "status": "in_progress" if actually_processed else "in_progress",
        }
        save_checkpoint_with_backup(checkpoint_path, checkpoint)

    # Step 4: Calculate remaining videos
    remaining_videos = [v for v in video_list if v['id'] not in actually_processed]

    logger.info(f"Resume point: {len(actually_processed)} completed, {len(remaining_videos)} remaining")

    return checkpoint, remaining_videos
```

**Advantages:**
- ✅ Automatically detects and corrects corrupted checkpoints
- ✅ Safe to delete checkpoint - will auto-rebuild from files
- ✅ No data loss - resumes from actual progress
- ✅ Transparent to user (auto-recovery)

**Disadvantages:**
- ⚠️ Slightly slower initialization (must scan file system)
- ⚠️ Can't distinguish between "failed" and "not started" videos

---

#### **Option B: Checkpoint Validation with Manual Recovery**

**Concept:** Validate checkpoint, but require manual intervention if corrupted.

```python
def initialize_checkpoint(bucket_name: str, video_list: list, config: dict):
    """Current logic + validation check."""

    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint_with_recovery(checkpoint_path)

        # NEW: Validate checkpoint integrity
        validation_errors = validate_checkpoint_integrity(checkpoint, bucket_name, config)

        if validation_errors:
            raise CheckpointCorruptionError(
                f"Checkpoint validation failed: {validation_errors}\n\n"
                f"RECOVERY OPTIONS:\n"
                f"1. Delete checkpoint to start fresh: rm {checkpoint_path}\n"
                f"2. Manually fix checkpoint completed_video_ids\n"
                f"3. Run detection-based recovery script: python scripts/recover_checkpoint.py"
            )

        # Continue with normal resume logic...
```

**Advantages:**
- ✅ Prevents using corrupted checkpoints
- ✅ Explicit error messages guide recovery
- ✅ Preserves current logic (less risk)

**Disadvantages:**
- ❌ Requires manual intervention
- ❌ User must choose recovery method
- ❌ Can still lose data if user deletes checkpoint incorrectly

---

#### **Option C: Dual Checkpoint System**

**Concept:** Maintain both checkpoint file AND file system manifest.

```python
# Two sources of truth:
1. stage_2_checkpoint.json (current state, can be corrupted)
2. processed_manifest.json (append-only log of processed videos)

# On completion of each video:
def mark_video_complete(video_id):
    # Update checkpoint (as usual)
    checkpoint['completed_video_ids'].append(video_id)
    save_checkpoint(checkpoint)

    # ALSO: Append to manifest
    manifest_path = f"{bucket_path}processed_manifest.json"
    append_to_manifest(manifest_path, {
        "video_id": video_id,
        "processed_at": datetime.now().isoformat(),
        "insights_path": f"{insights_dir}{video_id}_temporal_windows_updated.json"
    })
```

**Advantages:**
- ✅ Two sources of truth - harder to corrupt both
- ✅ Manifest provides audit trail
- ✅ Can rebuild checkpoint from manifest

**Disadvantages:**
- ❌ More complex implementation
- ❌ Requires new manifest file format
- ❌ Migration needed for existing tests

---

### Recommendation

**Implement Option A (Detection-Based Resume)** for the following reasons:

1. **Robustness:** Automatically handles checkpoint corruption
2. **User-Friendly:** No manual intervention needed
3. **Safe:** Can delete checkpoint without losing data
4. **Transparent:** Auto-recovery is invisible to users

**Fallback:** If Option A is too risky for immediate deployment, implement Option B (Validation + Manual Recovery) as interim solution.

**Long-Term:** Consider Option C (Dual Checkpoint) for production systems requiring audit trails.

---

## Summary of Fixes

| Issue | Priority | Implementation Complexity | Time Savings |
|-------|----------|--------------------------|--------------|
| **ISSUE 1: Copy from Test 4** | HIGH | Medium (scripting) | ~60 hours |
| **ISSUE 2.1: Root cause investigation** | HIGH | Low (investigation) | Prevents future failures |
| **ISSUE 2.2: Validation improvements** | CRITICAL | Medium (4 enhancements) | Catches corruption early |
| **ISSUE 3: Detection-based resume** | MEDIUM | High (refactor) | Prevents data loss |

---

## Next Steps

### Immediate Actions (Test 3 Recovery):

1. **Implement ISSUE 1 Option A:** Copy 60 missing videos from Test 4 → Test 3
2. **Run Stage 2.5 on recovered files:** Organize copied files into bucket directories
3. **Complete manual taxonomy curation:** Stage 2.6 is waiting
4. **Continue pipeline:** Stages 2.7+ can proceed

### Short-Term Fixes (Prevent Recurrence):

1. **ISSUE 2.2 Enhancement 3:** Add checkpoint ownership validation (CRITICAL)
2. **ISSUE 2.2 Enhancement 4:** Add video ID provenance check (CRITICAL)
3. **ISSUE 2.1:** Investigate root cause (run Hypothesis 1 & 2 investigations)

### Long-Term Improvements:

1. **ISSUE 3 Option A:** Implement detection-based resume
2. **ISSUE 2.2 Enhancement 1 & 2:** Add count-based and bucket-specific validation
3. **ISSUE 2.1:** Address root cause based on investigation findings

---

## Open Questions

1. **Test 3 Timeline:** Should we prioritize recovery (ISSUE 1) or investigation (ISSUE 2.1)?
2. **Validation Scope:** Should bucket-specific validation run in Stage 2.5 or Stage 2.6?
3. **Resume Strategy:** Is detection-based resume worth the refactor, or validate-and-fail sufficient?
4. **Test 4 Impact:** Does Test 4 also have checkpoint issues we haven't discovered?

---

**Document Status:** Ready for Review
**Author:** Investigation completed 2025-10-29
**Next Update:** After root cause investigation (ISSUE 2.1)
