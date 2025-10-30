# Stage 2 & 2.5 Architecture Bug: Shared Output Directory Issue

> **Document**: Stage2and2.5Bugfix.md
> **Version**: 1.0
> **Date**: 2025-01-30
> **Status**: Bug Analysis & Fix Proposal
> **Related**: E2EFixesv2.md (ISSUE 2), SystemArchitecturev2.md

---

## Executive Summary

**Bug Discovered**: Test 3's checkpoint contained 40 video IDs from Test 4, causing 60 videos to be skipped during processing.

**Root Cause**: Stage 2 writes outputs to **shared global directories** (`/insights/`, `/temp/`) and updates checkpoints BEFORE files are moved to test-specific bucket directories (done later by Stage 2.5). This creates a "vulnerable window" where multiple tests can access and incorrectly claim ownership of the same files.

**Impact**:
- Test 3 processed only 66/126 videos (52%)
- 60 videos marked as "completed" but never actually processed
- Checkpoint corruption blocked downstream ML pipeline stages
- ~60 hours of lost processing time

**Fix**: Move files to test-specific bucket directories IMMEDIATELY after processing (before updating checkpoint), eliminating the vulnerable window.

---

## Table of Contents

1. [The Bug: What Happened](#1-the-bug-what-happened)
2. [Discovery Process](#2-discovery-process)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Current Architecture (Broken)](#4-current-architecture-broken)
5. [Why the Bug Occurs](#5-why-the-bug-occurs)
6. [Proposed Fix](#6-proposed-fix)
7. [Implementation Plan](#7-implementation-plan)
8. [Testing Strategy](#8-testing-strategy)
9. [Alternative Solutions](#9-alternative-solutions)
10. [Lessons Learned](#10-lessons-learned)

---

## 1. The Bug: What Happened

### 1.1 The Incident

**Test 3 Checkpoint Corruption Event:**

```bash
# Test 3 Expected Behavior:
- Process 126 videos across 8 buckets
- Checkpoint tracks progress per bucket
- Each video processed through RumiAI pipeline

# Test 3 Actual Behavior:
- Only processed 66/126 videos (52%)
- Checkpoint claimed 126/126 completed (false!)
- Missing: 60 videos (37 from bucket 3-9s, 23 from bucket 60-90s)

# Evidence:
Test 3 checkpoint contained 40 video IDs from Test 4, not Test 3!
Test 3 actually processed: 3 videos
Test 3 checkpoint claimed: 40 videos completed
```

### 1.2 Impact

**Immediate Impact:**
- 60 videos missing temporal_windows_updated.json outputs
- Stage 3 (Feature Aggregation) blocked - can't aggregate missing features
- Manual intervention required to copy files from Test 4

**Time Impact:**
- ~60 hours of processing time lost
- Manual taxonomy curation delayed
- ML training pipeline blocked

**Data Integrity:**
- Checkpoint falsely claimed completion
- Validation passed despite missing files
- Silent failure - no error raised until Stage 3

---

## 2. Discovery Process

### 2.1 Initial Hypothesis (INCORRECT)

**From E2EFixesv2.md:**

**Hypothesis 1: Concurrent Execution**
- Theory: Test 3 and Test 4 ran simultaneously, causing checkpoint file race condition
- Status: ❌ **NO EVIDENCE** - only listed as hypothesis to investigate
- Conclusion: Concurrent execution NOT required for bug to occur

**Hypothesis 2: Path Resolution Bug**
- Theory: Checkpoint saving constructed wrong path, writing Test 3 checkpoint to Test 4 location
- Status: ❌ **DISPROVEN** - Path construction verified correct (uses centralized PathBuilder)
- Evidence: Checkpoints saved to different paths:
  - Test 3: `/data/clients/rollo_test3/.../bucket_3-9s/checkpoints/stage_2_checkpoint.json`
  - Test 4: `/data/clients/rollo_test4/.../bucket_3-9s/checkpoints/stage_2_checkpoint.json`

### 2.2 Key Insights Leading to Discovery

**Insight 1 (User):**
> "Test 3 and 4 were scraping the same hashtags, with different configurations - both tests having same videos is logical"

**Insight 2 (User):**
> "I believe rumiai_runner.py does NOT download videos twice... Does this have something to do?"

**Insight 3 (User):**
> "At what stage is the checkpoint made? After we move videos to their buckets?"

**Insight 4 (User):**
> "Shouldn't the checkpoint be AFTER files are moved?"

**These insights led to the discovery of the shared output directory architecture flaw.**

### 2.3 The Breakthrough

**Code Analysis Revealed:**

1. **rumiai_runner.py writes to HARDCODED global directories:**
   ```python
   # rumiai_v2/config/settings.py:33
   self.insights_dir = Path(os.getenv('RUMIAI_INSIGHTS_DIR', 'insights'))

   # Default: /home/jorge/rumiaifinal/insights/
   ```

2. **Stage 2 validates outputs in GLOBAL directory:**
   ```python
   # ml_pipeline/stage2_processing/video_processor.py:26
   RUMIAI_OUTPUT_DIR = "/home/jorge/rumiaifinal/insights/"

   # Line 184: Validation checks global directory
   insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
   ```

3. **Checkpoint updated BEFORE files move to bucket directories:**
   ```python
   # video_processor.py:196-202
   checkpoint['completed'] += 1
   checkpoint['completed_video_ids'].append(video_id)
   save_checkpoint_with_backup(checkpoint_path, checkpoint)
   # ⚠️ File is STILL in /insights/ (global shared directory!)
   ```

4. **Stage 2.5 moves files LATER as separate batch operation:**
   ```python
   # file_organizer.py:176-177
   'source_path': f"{SOURCE_DIRS['temporal_windows']}{video_id}_temporal_windows_updated.json"
   'target_path': f"{analysis_base}/buckets/bucket_{bucket}/analysis/insights/{video_id}_temporal_windows_updated.json"
   ```

---

## 3. Root Cause Analysis

### 3.1 The ACTUAL Root Cause

**Primary Cause: Architectural Design Flaw**

Stage 2 and Stage 2.5 are **separated** in a way that creates a "vulnerable window":

1. Stage 2 processes videos → writes to **GLOBAL** `/insights/` directory
2. Stage 2 updates checkpoint → marks video as "completed"
3. Files remain in GLOBAL directory (shared by all tests)
4. Stage 2.5 moves files → runs LATER as batch operation

**The Vulnerable Window:**
- Between Stage 2 checkpoint update and Stage 2.5 file movement
- Files sit in GLOBAL `/insights/` directory (accessible to all tests)
- Multiple tests can find the same files and claim ownership

### 3.2 Why Concurrent Execution is NOT Required

**The bug occurs with SEQUENTIAL execution:**

```
Timeline (Sequential):

Day 1: Test 4 completes Stage 2
├─ Test 4 processes 59 videos
├─ Writes to: /insights/ (GLOBAL)
├─ Checkpoint: /test4/.../checkpoints/stage_2_checkpoint.json
│   completed_video_ids: [video_042, video_043, ..., video_100]
└─ Files remain in /insights/ (waiting for Stage 2.5)

Day 2: Test 3 starts Stage 2 (BEFORE Test 4's Stage 2.5 runs)
├─ Test 3 tries to process video_042
├─ rumiai_runner.py checks: /insights/video_042_temporal_windows_updated.json
├─ File EXISTS! (Test 4 created it yesterday)
├─ Validation: File exists ✅ (PASSES)
├─ Test 3 checkpoint: completed_video_ids.append(video_042)
└─ BUG: Test 3 marks video_042 as "completed" without processing it
```

**Key Point:** Even with sequential execution, if Test 4's Stage 2.5 hasn't run yet, its files remain in `/insights/` where Test 3 can find them.

### 3.3 Contributing Factors

**Secondary Factors:**

1. **Checkpoint Timing Bug:**
   - Checkpoint updated BEFORE files move to test-specific directories
   - Should update AFTER files are isolated

2. **Validation Location Bug:**
   - Stage 2 validates files in GLOBAL `/insights/` directory
   - Should validate in test-specific bucket directory

3. **Missing Validation (from E2EFixesv2.md):**
   - **Enhancement 3:** No checkpoint ownership validation (client_id, target)
   - **Enhancement 4:** No video ID provenance check (vs selected_videos.json)
   - **Enhancement 1:** No count-based validation (checkpoint count vs actual files)

---

## 4. Current Architecture (Broken)

### 4.1 Stage 2: Video Processing

**Per-Video Processing Loop:**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. rumiai_runner.py processes video                         │
│    - Downloads video to: /temp/{video_id}.mp4               │
│    - Writes output to: /insights/{video_id}_temporal...json │
│    - Writes unified to: /unified_analysis/{video_id}.json   │
│    ⚠️ ALL IN GLOBAL FLAT DIRECTORIES (shared by all tests) │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. video_processor.py validates output exists               │
│    - Checks: /insights/{video_id}_temporal...json EXISTS?   │
│    - ⚠️ Validates in GLOBAL directory (not test-specific!)  │
│    - Validates schema                                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ⭐ CHECKPOINT UPDATED (video_processor.py:196-202)       │
│    Location: /data/clients/{client}/{target}/...            │
│              /buckets/bucket_{bucket}/checkpoints/           │
│              stage_2_checkpoint.json                         │
│                                                              │
│    Updates:                                                  │
│    - checkpoint['completed'] += 1                           │
│    - checkpoint['remaining'] -= 1                           │
│    - checkpoint['completed_video_ids'].append(video_id)     │
│    - save_checkpoint_with_backup(checkpoint_path, ...)      │
│                                                              │
│    ⚠️ FILES ARE STILL IN GLOBAL DIRECTORIES!                │
│    Videos NOT moved to bucket directories yet               │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Stage 2.5: File Organization (Separate Batch Operation)

**Runs AFTER all Stage 2 processing completes:**

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 2.5: File Organization (file_organizer.py)            │
│                                                              │
│ 1. READS checkpoints (doesn't modify them!)                 │
│    - Loads: bucket_{bucket}/checkpoints/stage_2_checkpoint  │
│    - Extracts: checkpoint['completed_video_ids']            │
│    - Uses video IDs to know WHAT files to move              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. MOVES files from GLOBAL → BUCKET directories             │
│    Source → Target:                                          │
│    /insights/{vid}_temporal...json →                        │
│      /buckets/bucket_{bucket}/analysis/insights/{vid}...    │
│                                                              │
│    /temp/{vid}.mp4 →                                        │
│      /buckets/bucket_{bucket}/videos/{vid}.mp4              │
│                                                              │
│    /unified_analysis/{vid}.json →                           │
│      /buckets/bucket_{bucket}/analysis/unified/{vid}.json   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Stage 2.5 does NOT update checkpoints                    │
│    - Checkpoints remain unchanged                           │
│    - No Stage 2.5 checkpoint file created                   │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 The Vulnerable Window

```
Timeline showing the vulnerable window:

Test 4 Stage 2 Completes
│
├─ Checkpoint: /test4/.../checkpoints/stage_2_checkpoint.json
│   Status: "completed"
│   completed_video_ids: [video_042, video_043, ..., video_100]
│
├─ Files Location: /insights/ (GLOBAL SHARED)
│   /insights/video_042_temporal_windows_updated.json ✅
│   /insights/video_043_temporal_windows_updated.json ✅
│
│   ⚠️ VULNERABLE WINDOW OPENS ⚠️
│   Files accessible to ALL tests!
│
│   Test 3 Stage 2 Starts (can access Test 4's files)
│   ├─ Tries video_042
│   ├─ Finds: /insights/video_042_...json (Test 4's file!)
│   ├─ Validation: File exists ✅
│   └─ Marks video_042 as "completed" ❌ BUG!
│
│   ⚠️ VULNERABLE WINDOW REMAINS OPEN ⚠️
│
├─ Test 4 Stage 2.5 Runs (LATER)
│   Moves: /insights/video_042... → /test4/buckets/.../insights/
│
└─ VULNERABLE WINDOW CLOSES (files now isolated)
    But damage already done - Test 3 checkpoint corrupted!
```

---

## 5. Why the Bug Occurs

### 5.1 Failure Scenario: Sequential Execution

**No concurrent execution needed! Bug occurs with sequential runs:**

```
┌─────────────────────────────────────────────────────────────┐
│ Day 1: Test 4 runs (Stage 1 + Stage 2)                      │
│ - Selects 59 videos for bucket 3-9s                         │
│ - Processes all 59 videos successfully                       │
│ - Writes to: /insights/ (GLOBAL)                            │
│ - Checkpoint: /test4/.../checkpoints/...                    │
│   completed_video_ids: [video_042, ..., video_100]          │
│ - Does NOT run Stage 2.5 yet (waiting for manual trigger)   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Day 2: Test 3 runs (Stage 1 + Stage 2)                      │
│ - Selects 40 videos for bucket 3-9s                         │
│ - 37 videos overlap with Test 4 (same hashtags!)            │
│ - Creates new checkpoint: /test3/.../checkpoints/...        │
│   completed_video_ids: [] (empty)                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Test 3 processes video_001 (unique to Test 3)               │
│ - rumiai_runner.py: /insights/video_001_...json ✅          │
│ - Validation: File exists ✅                                 │
│ - Checkpoint: completed_video_ids = [video_001]             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Test 3 processes video_042 (OVERLAPS with Test 4)           │
│                                                              │
│ Scenario A: rumiai_runner.py has skip logic                │
│ - Checks: /insights/video_042_...json exists? YES          │
│ - SKIPS processing (assumes already done)                   │
│ - video_processor.py validation: File exists ✅             │
│ - Checkpoint: completed_video_ids.append(video_042)         │
│                                                              │
│ Scenario B: rumiai_runner.py always overwrites             │
│ - Checks: /insights/video_042_...json exists? YES          │
│ - OVERWRITES with Test 3's version                         │
│ - video_processor.py validation: File exists ✅             │
│ - Checkpoint: completed_video_ids.append(video_042)         │
│                                                              │
│ EITHER WAY: Test 3 marks video_042 as "completed"          │
│ But video_042 is Test 4's video, not Test 3's!             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Why Validation Doesn't Catch It

**Stage 2 validation checks GLOBAL directory:**

```python
# ml_pipeline/stage2_processing/video_processor.py:184-190

def process_videos_sequential(...):
    # After rumiai_runner.py completes
    insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
    # RUMIAI_OUTPUT_DIR = "/home/jorge/rumiaifinal/insights/" (GLOBAL!)

    if not os.path.exists(insights_path):
        raise ProcessingError(...)  # Doesn't raise - file exists!

    # Validation passes ✅ (false positive)
```

**Why it's a false positive:**
- File exists in `/insights/` ✅
- But it's Test 4's file, not Test 3's file!
- Validation doesn't check file ownership
- Validation doesn't check if file belongs to THIS test

### 5.3 Why Count Validation Doesn't Exist

**Missing validation from E2EFixesv2.md Enhancement 1:**

```python
# Should check (but doesn't):
actual_count = len([f for f in os.listdir(insights_dir)
                   if f.endswith('_temporal_windows_updated.json')])
expected_count = checkpoint['completed']

if actual_count != expected_count:
    raise AssertionError(f"Expected {expected_count}, found {actual_count}")
```

**Why this would catch the bug:**
- Test 3 checkpoint claims 40 completed
- But only 3 files belong to Test 3
- Count mismatch would be detected

---

## 6. Proposed Fix

### 6.1 Solution Overview

**Fix: Move files to test-specific bucket directories IMMEDIATELY after processing**

**Key Changes:**
1. Move files from global `/insights/` to bucket-specific directories IMMEDIATELY
2. Update checkpoint AFTER file is moved (not before!)
3. Validate files in bucket-specific directories (not global)
4. Eliminate Stage 2.5 (no longer needed - files already in correct locations)

### 6.2 Architecture: BEFORE vs AFTER

#### BEFORE (Broken):

```
Stage 2:
  process_video()
    → rumiai_runner.py writes to /insights/ (GLOBAL)
    → validate in /insights/ (GLOBAL) ✅
    → update checkpoint ⚠️ (FILES STILL IN GLOBAL DIR)

Stage 2.5: (runs LATER)
  organize_files()
    → move /insights/ → /buckets/.../insights/
```

#### AFTER (Fixed):

```
Stage 2:
  process_video()
    → rumiai_runner.py writes to /insights/ (GLOBAL - can't change)
    → IMMEDIATELY move /insights/ → /buckets/.../insights/ ✅
    → validate in /buckets/.../insights/ (TEST-SPECIFIC) ✅
    → update checkpoint ✅ (FILES NOW IN TEST-SPECIFIC DIR)

Stage 2.5: ELIMINATED (no longer needed)
```

### 6.3 Code Changes

#### Change 1: Modify `run_rumiai_pipeline()` to move files immediately

**File:** `ml_pipeline/stage2_processing/video_processor.py`

**BEFORE:**

```python
def run_rumiai_pipeline(video_path: str, video_id: str, output_dir: str, timeout: int = 300):
    """Run RumiAI pipeline with timeout enforcement."""

    # rumiai_runner.py writes to HARDCODED /insights/
    insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"

    # Run subprocess
    subprocess.run([...], check=True)

    # Validate output exists (in GLOBAL directory)
    if not os.path.exists(insights_path):
        raise ProcessingError(...)

    return {"status": "success", "insights_path": insights_path}
```

**AFTER:**

```python
def run_rumiai_pipeline(
    video_path: str,
    video_id: str,
    bucket_path: str,  # NEW: bucket-specific path
    timeout: int = 300
):
    """
    Run RumiAI pipeline with immediate file isolation.

    Changes:
    - Moves files to bucket-specific directories IMMEDIATELY after processing
    - Validates files in bucket-specific directories (not global)
    - Eliminates vulnerable window where files remain in shared directory
    """

    # Step 1: rumiai_runner.py writes to HARDCODED /insights/ (can't change external script)
    temp_insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
    temp_video_path = f"{RUMIAI_TEMP_DIR}{video_id}.mp4"
    temp_unified_path = f"{RUMIAI_UNIFIED_DIR}{video_id}.json"

    # Step 2: Run subprocess (writes to global directories)
    subprocess.run([...], check=True)

    # Step 3: IMMEDIATELY move files to test-specific bucket directories
    target_insights_path = f"{bucket_path}analysis/insights/{video_id}_temporal_windows_updated.json"
    target_video_path = f"{bucket_path}videos/{video_id}.mp4"
    target_unified_path = f"{bucket_path}analysis/unified/{video_id}.json"

    # Ensure target directories exist
    os.makedirs(f"{bucket_path}analysis/insights/", exist_ok=True)
    os.makedirs(f"{bucket_path}videos/", exist_ok=True)
    os.makedirs(f"{bucket_path}analysis/unified/", exist_ok=True)

    # Move files atomically (within same filesystem)
    try:
        shutil.move(temp_insights_path, target_insights_path)
        logger.info(f"Moved insights: {video_id} → {bucket_path}analysis/insights/")

        if os.path.exists(temp_video_path):
            shutil.move(temp_video_path, target_video_path)
            logger.info(f"Moved video: {video_id} → {bucket_path}videos/")

        if os.path.exists(temp_unified_path):
            shutil.move(temp_unified_path, target_unified_path)
            logger.info(f"Moved unified: {video_id} → {bucket_path}analysis/unified/")

    except Exception as e:
        logger.error(f"Failed to move files for {video_id}: {e}")
        raise ProcessingError(
            video_id=video_id,
            stage="file_isolation",
            message=f"Failed to move files to bucket directory: {e}"
        )

    # Step 4: Validate output exists in BUCKET-SPECIFIC directory (not global)
    if not os.path.exists(target_insights_path):
        raise ProcessingError(
            video_id=video_id,
            stage="output_validation",
            message=f"Insights file missing after move: {target_insights_path}"
        )

    return {
        "status": "success",
        "insights_path": target_insights_path,
        "video_path": target_video_path if os.path.exists(target_video_path) else None,
        "unified_path": target_unified_path if os.path.exists(target_unified_path) else None
    }
```

#### Change 2: Update `process_videos_sequential()` to pass bucket_path

**File:** `ml_pipeline/stage2_processing/video_processor.py`

**BEFORE:**

```python
def process_videos_sequential(remaining_videos, bucket_name, checkpoint, config):
    bucket_path = get_bucket_path(config, bucket_name)

    for video in remaining_videos:
        # Run RumiAI pipeline
        result = run_rumiai_pipeline(
            video_path=video_path,
            video_id=video_id,
            output_dir=f"{bucket_path}analysis/",  # Not actually used by rumiai_runner
            timeout=300
        )

        # Validate output exists at GLOBAL directory (WRONG!)
        insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
        if not os.path.exists(insights_path):
            raise ProcessingError(...)

        # Update checkpoint
        checkpoint['completed'] += 1
        checkpoint['completed_video_ids'].append(video_id)
        save_checkpoint_with_backup(checkpoint_path, checkpoint)
```

**AFTER:**

```python
def process_videos_sequential(remaining_videos, bucket_name, checkpoint, config):
    bucket_path = get_bucket_path(config, bucket_name)
    checkpoint_path = f"{bucket_path}checkpoints/stage_2_checkpoint.json"

    for video in remaining_videos:
        # Run RumiAI pipeline (now moves files immediately)
        result = run_rumiai_pipeline(
            video_path=video_path,
            video_id=video_id,
            bucket_path=bucket_path,  # Pass bucket_path for immediate file isolation
            timeout=300
        )

        # Validate output exists at BUCKET-SPECIFIC directory (CORRECT!)
        insights_path = result['insights_path']  # Already in bucket directory
        if not os.path.exists(insights_path):
            raise ProcessingError(...)

        # Validate schema
        insights = load_json(insights_path)
        validate_temporal_windows_schema(insights)

        # Update checkpoint (AFTER files are isolated in bucket directory)
        checkpoint['completed'] += 1
        checkpoint['remaining'] -= 1
        checkpoint['completed_video_ids'].append(video_id)
        checkpoint['last_checkpoint'] = datetime.utcnow().isoformat()

        save_checkpoint_with_backup(checkpoint_path, checkpoint)
        logger.info(f"Successfully processed video {video_id} ({checkpoint['completed']}/{checkpoint['total_videos']})")
```

#### Change 3: Update validation to check bucket-specific directories

**File:** `ml_pipeline/stage2_processing/main.py`

**BEFORE:**

```python
def validate_stage_output(bucket_name: str, checkpoint: dict, config: dict):
    """Validate stage outputs after all videos processed."""

    # WRONG: Validates in GLOBAL directory
    insights_dir = RUMIAI_OUTPUT_DIR  # /insights/ (global shared)

    # Check each completed video has insights file
    missing_files = []
    for video_id in checkpoint['completed_video_ids']:
        insights_path = f"{insights_dir}{video_id}_temporal_windows_updated.json"
        if not os.path.exists(insights_path):
            missing_files.append(video_id)

    assert len(missing_files) == 0, f"Missing insights files: {missing_files}"
```

**AFTER:**

```python
def validate_stage_output(bucket_name: str, checkpoint: dict, config: dict):
    """
    Validate stage outputs after all videos processed.

    FIXED: Validates in BUCKET-SPECIFIC directory (test-isolated).
    """
    bucket_path = get_bucket_path(config, bucket_name)
    insights_dir = f"{bucket_path}analysis/insights/"

    # Enhancement 1: Count-based validation
    actual_files = [f for f in os.listdir(insights_dir)
                   if f.endswith('_temporal_windows_updated.json')]
    actual_count = len(actual_files)
    expected_count = checkpoint['completed']

    if actual_count != expected_count:
        raise AssertionError(
            f"Stage 2 completion mismatch: Checkpoint claims {expected_count} completed, "
            f"but {actual_count} files exist in {insights_dir}. "
            f"Missing: {expected_count - actual_count} files"
        )

    # Check each completed video has insights file
    missing_files = []
    for video_id in checkpoint['completed_video_ids']:
        insights_path = f"{insights_dir}{video_id}_temporal_windows_updated.json"
        if not os.path.exists(insights_path):
            missing_files.append(video_id)

    assert len(missing_files) == 0, \
        f"Missing insights files for {len(missing_files)} completed videos: {missing_files}"

    logger.info(f"Stage 2 validation passed for bucket {bucket_name}")
    logger.info(f"  Completed videos: {checkpoint['completed']}")
    logger.info(f"  Files location: {insights_dir} (test-specific)")
```

#### Change 4: Eliminate Stage 2.5 (no longer needed)

**Stage 2.5 is NO LONGER NEEDED** because files are already in bucket directories after Stage 2.

**Migration Path:**
1. Keep Stage 2.5 code for backward compatibility (in case old runs need organization)
2. Add check: "Skip Stage 2.5 if files already in bucket directories"
3. Eventually deprecate and remove Stage 2.5

**File:** `ml_pipeline/stage2_5_organize/file_organizer.py`

```python
def stage_2_5_main(analysis_base: str):
    """
    Stage 2.5: File Organization (DEPRECATED after immediate file isolation fix).

    This stage is now a NO-OP if Stage 2 has been updated to move files immediately.
    Kept for backward compatibility with older runs that still have files in global directories.
    """

    # Check if files need organization
    needs_organization = check_if_organization_needed(analysis_base)

    if not needs_organization:
        logger.info("Stage 2.5 skipped: Files already organized in bucket directories")
        logger.info("This is expected if Stage 2 has been updated with immediate file isolation fix")
        return {"status": "skipped", "reason": "files_already_organized"}

    logger.warning("Stage 2.5 running: Files found in global directories (old Stage 2 behavior)")
    logger.warning("Consider re-running Stage 2 with updated version for better isolation")

    # Run original Stage 2.5 logic for backward compatibility
    # ... (existing code)
```

---

## 7. Implementation Plan

### 7.1 Implementation Phases

#### Phase 1: Fix Stage 2 (Immediate File Isolation) - CRITICAL

**Priority:** P0 (Critical - Prevents future checkpoint corruption)

**Tasks:**
1. ✅ Modify `run_rumiai_pipeline()` to move files immediately
2. ✅ Update `process_videos_sequential()` to pass bucket_path
3. ✅ Update validation to check bucket-specific directories
4. ✅ Add Enhancement 1 (count-based validation)
5. ✅ Add comprehensive logging for file movements

**Files to Modify:**
- `ml_pipeline/stage2_processing/video_processor.py`
- `ml_pipeline/stage2_processing/main.py`

**Testing:**
- Unit tests: File movement logic
- Integration tests: Full Stage 2 run with multiple videos
- Regression tests: Ensure no existing functionality broken

**Success Criteria:**
- Files moved to bucket directories immediately after processing
- Checkpoint updated AFTER files are moved
- Validation checks bucket-specific directories
- No files remain in global directories after Stage 2

#### Phase 2: Add Missing Validations - HIGH

**Priority:** P1 (High - Catches corruption if it occurs)

**Tasks:**
1. ✅ Add Enhancement 3 (checkpoint ownership validation)
2. ✅ Add Enhancement 4 (video ID provenance check)
3. ✅ Add Enhancement 1 (count-based validation) - already done in Phase 1
4. ✅ Add Enhancement 2 (bucket-specific validation post-Stage 2.5)

**Files to Modify:**
- `ml_pipeline/stage2_processing/checkpoint.py`
- `ml_pipeline/stage2_processing/main.py`

**Testing:**
- Test checkpoint ownership validation catches wrong client_id
- Test provenance check catches video IDs not in selected_videos.json
- Test count validation catches mismatches

**Success Criteria:**
- Checkpoint corruption detected immediately
- Clear error messages guide recovery
- All validation enhancements pass tests

#### Phase 3: Update Stage 2.5 for Backward Compatibility - MEDIUM

**Priority:** P2 (Medium - Supports older runs)

**Tasks:**
1. ✅ Add check: Skip Stage 2.5 if files already organized
2. ✅ Keep Stage 2.5 functional for older runs
3. ✅ Add deprecation warnings
4. ✅ Update documentation

**Files to Modify:**
- `ml_pipeline/stage2_5_organize/file_organizer.py`
- Documentation

**Testing:**
- Test Stage 2.5 skips correctly for new runs
- Test Stage 2.5 still works for old runs (backward compatibility)

**Success Criteria:**
- Stage 2.5 skips for new runs (files already organized)
- Stage 2.5 works for old runs (files in global directories)
- Clear logging explains skip behavior

#### Phase 4: Update Documentation - MEDIUM

**Priority:** P2 (Medium - Prevents future confusion)

**Tasks:**
1. ✅ Update SystemArchitecturev2.md
2. ✅ Update VideoProcessingTI.md
3. ✅ Update FileOrganizationCHILDTI.md (mark as deprecated)
4. ✅ Create migration guide for existing runs

**Success Criteria:**
- Documentation reflects new architecture
- Migration guide available for existing runs
- Deprecation notices in place

### 7.2 Rollout Strategy

#### Option A: Immediate Rollout (Recommended)

**For NEW tests:**
- Use fixed Stage 2 (immediate file isolation)
- Skip Stage 2.5 (no longer needed)

**For EXISTING tests (Test 3, Test 4):**
- Already completed Stage 2 with old code
- Run Stage 2.5 to organize files (backward compatibility)
- Then continue with downstream stages

#### Option B: Gradual Rollout

**Week 1:** Deploy Phase 1 (fix Stage 2) to staging
**Week 2:** Test with sample runs, validate no regressions
**Week 3:** Deploy to production, monitor
**Week 4:** Deploy Phase 2 (validations), monitor

### 7.3 Testing Checklist

**Before Deployment:**

- [ ] Unit tests pass for file movement logic
- [ ] Integration tests pass for full Stage 2 run
- [ ] Concurrent test simulation passes (Test 3 + Test 4 scenario)
- [ ] Sequential test simulation passes (Test 4 → Test 3 scenario)
- [ ] Validation catches checkpoint corruption
- [ ] Backward compatibility verified (old runs with Stage 2.5)
- [ ] Performance impact measured (file moves add latency?)
- [ ] Disk space impact measured (files in bucket dirs vs global)

**After Deployment:**

- [ ] Monitor first 3 production runs
- [ ] Verify no files remain in `/insights/` after Stage 2
- [ ] Verify checkpoint counts match actual file counts
- [ ] Verify downstream stages (Stage 3+) receive correct inputs
- [ ] No regression in processing time
- [ ] No increase in error rates

---

## 8. Testing Strategy

### 8.1 Test Scenarios

#### Test Case 1: Sequential Execution (Test 4 → Test 3)

**Scenario:** Reproduce the original bug scenario

**Setup:**
1. Run Test 4 (Stage 1 + Stage 2) with OLD code
2. Do NOT run Test 4 Stage 2.5 yet
3. Run Test 3 (Stage 1 + Stage 2) with OLD code

**Expected Result (OLD CODE - BUG):**
- Test 3 checkpoint contains Test 4's video IDs ❌
- Test 3 processes only 3 videos, but checkpoint claims 40 ❌

**Expected Result (NEW CODE - FIXED):**
- Test 3 checkpoint contains ONLY Test 3's video IDs ✅
- Test 3 processes all selected videos ✅
- No files remain in `/insights/` after Stage 2 ✅

#### Test Case 2: Overlapping Video Sets

**Scenario:** Test 3 and Test 4 have 37 common videos

**Setup:**
1. Test 3 selected_videos.json: [video_001, video_002, ..., video_042, ...]
2. Test 4 selected_videos.json: [video_042, video_043, ..., video_100]
3. Run Test 4 Stage 2 (new code)
4. Run Test 3 Stage 2 (new code)

**Expected Result:**
- Test 4 files: `/test4/buckets/bucket_3-9s/analysis/insights/video_042_...json`
- Test 3 files: `/test3/buckets/bucket_3-9s/analysis/insights/video_042_...json`
- DIFFERENT LOCATIONS - No collision! ✅
- Test 3 processes video_042 independently ✅

#### Test Case 3: Concurrent Execution

**Scenario:** Test 3 and Test 4 run simultaneously

**Setup:**
1. Start Test 3 Stage 2 in background
2. Start Test 4 Stage 2 in background
3. Both process overlapping video sets

**Expected Result:**
- No checkpoint corruption ✅
- Files isolated in test-specific directories ✅
- Both tests complete successfully ✅

#### Test Case 4: Checkpoint Ownership Validation

**Scenario:** Accidentally load wrong checkpoint

**Setup:**
1. Copy Test 4's checkpoint to Test 3's checkpoint location
2. Try to resume Test 3

**Expected Result:**
- Enhancement 3 validation catches ownership mismatch ✅
- Error: "Checkpoint belongs to different test (client_id mismatch)" ✅

#### Test Case 5: Video ID Provenance Check

**Scenario:** Checkpoint contains video IDs not in selected_videos.json

**Setup:**
1. Manually corrupt checkpoint: Add video_999 to completed_video_ids
2. video_999 NOT in Test 3's selected_videos.json
3. Try to resume Test 3

**Expected Result:**
- Enhancement 4 validation catches provenance mismatch ✅
- Error: "Checkpoint contains video IDs not in selected_videos.json" ✅

#### Test Case 6: Count Validation

**Scenario:** Checkpoint count doesn't match actual file count

**Setup:**
1. Checkpoint claims 40 completed
2. Only 3 files exist in bucket directory
3. Run validation

**Expected Result:**
- Enhancement 1 validation catches count mismatch ✅
- Error: "Expected 40 files, found 3" ✅

### 8.2 Performance Testing

**Metrics to Measure:**

| Metric | Old Code | New Code | Change |
|--------|----------|----------|--------|
| Per-video processing time | ~60-80s | ? | Measure |
| File move overhead | 0s (deferred to Stage 2.5) | ~0.1-0.5s | Estimate |
| Stage 2 total time (100 videos) | ~2 hours | ? | Measure |
| Disk I/O during Stage 2 | Low (writes only) | Medium (write + move) | Monitor |
| Stage 2.5 time | ~10-20 minutes | 0s (skipped) | Net savings! |

**Expected Impact:**
- Minor increase in Stage 2 time (file moves add ~0.1-0.5s per video)
- Major decrease in overall time (Stage 2.5 eliminated)
- Net savings: ~10-20 minutes for 100-video batch

### 8.3 Regression Testing

**Ensure No Breaking Changes:**

- [ ] Stage 2 completes successfully with new code
- [ ] Checkpoint format unchanged (backward compatible)
- [ ] Downstream stages (3-7) receive correct inputs
- [ ] Error handling still works (failed videos skipped)
- [ ] Graceful pause (Ctrl+C) still works
- [ ] Resume from checkpoint still works

---

## 9. Alternative Solutions

### 9.1 Alternative 1: Test-Specific Subdirectories in Global Dirs

**Approach:**

Instead of moving files to bucket directories immediately, create test-specific subdirectories in global directories:

```python
# Instead of:
/insights/{video_id}_temporal_windows_updated.json

# Use:
/insights/{client_id}_{target}/{video_id}_temporal_windows_updated.json

# Example:
/insights/rollo_test3_wellness_test3/video_042_...json
/insights/rollo_test4_wellness_test4/video_042_...json
```

**Pros:**
- ✅ Tests isolated in global directories
- ✅ Less code change (rumiai_runner.py just needs output dir parameter)
- ✅ Can keep Stage 2.5 (moves from test-specific global → bucket-specific)

**Cons:**
- ⚠️ Still has "vulnerable window" (but isolated per test)
- ⚠️ Extra indirection (global test dirs → bucket dirs)
- ⚠️ Doesn't eliminate Stage 2.5 (still needed)
- ⚠️ Requires modifying rumiai_runner.py (external script)

**Verdict:** ❌ Not recommended (more complex, doesn't eliminate root issue)

### 9.2 Alternative 2: File Locking Mechanism

**Approach:**

Add file locks to prevent multiple tests from accessing same files:

```python
import fcntl

def process_video_with_lock(video_id):
    lock_file = f"/tmp/rumiai_{video_id}.lock"

    with open(lock_file, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # Exclusive lock

        # Process video
        run_rumiai_pipeline(...)

        # Lock released automatically on exit
```

**Pros:**
- ✅ Prevents concurrent access to same video
- ✅ No architecture change needed

**Cons:**
- ⚠️ Doesn't fix the root issue (shared global directories)
- ⚠️ Adds complexity (lock management, deadlock detection)
- ⚠️ Doesn't help with sequential execution (Test 4 → Test 3)
- ⚠️ Requires cross-process coordination

**Verdict:** ❌ Not recommended (doesn't address root cause)

### 9.3 Alternative 3: Make rumiai_runner.py Test-Aware

**Approach:**

Modify rumiai_runner.py to accept test-specific output directories:

```python
# rumiai_runner.py
def __init__(self, output_base_dir: str = None):
    if output_base_dir:
        self.insights_dir = Path(output_base_dir) / 'insights'
        self.temp_dir = Path(output_base_dir) / 'temp'
    else:
        # Default (backward compatibility)
        self.insights_dir = Path('insights')
        self.temp_dir = Path('temp')
```

**Pros:**
- ✅ Test-specific outputs from the start
- ✅ No file moves needed

**Cons:**
- ⚠️ Requires modifying rumiai_runner.py (external script, may break other uses)
- ⚠️ All calling code must pass output_base_dir
- ⚠️ Backward compatibility concerns

**Verdict:** ⚠️ Possible, but requires more coordination (external script changes)

### 9.4 Comparison Table

| Solution | Complexity | Effectiveness | Backward Compatible | Eliminates Stage 2.5 |
|----------|-----------|---------------|-------------------|---------------------|
| **Proposed Fix** (Move files immediately) | Medium | ✅ Complete | ✅ Yes | ✅ Yes |
| Alt 1 (Test subdirs in global) | Medium | ⚠️ Partial | ✅ Yes | ❌ No |
| Alt 2 (File locking) | High | ❌ Minimal | ✅ Yes | ❌ No |
| Alt 3 (Test-aware rumiai_runner) | Low | ✅ Complete | ⚠️ Requires changes | ✅ Yes |

**Recommendation:** Proposed Fix (Move files immediately) is the best solution.

---

## 10. Lessons Learned

### 10.1 Architecture Lessons

**1. Avoid Shared Global State in Concurrent Systems**

**Problem:** All tests writing to same global directories created race conditions.

**Lesson:** Isolate resources per execution unit (test, user, session) from the start.

**Best Practice:**
```python
# BAD: Global shared directory
output_dir = "/shared/outputs/"

# GOOD: Test-specific directory
output_dir = f"/shared/outputs/{test_id}/"
```

**2. Update Checkpoints AFTER State Changes Are Complete**

**Problem:** Checkpoint updated before files moved to final location.

**Lesson:** Checkpoints should reflect COMMITTED state, not PENDING state.

**Best Practice:**
```python
# BAD: Update checkpoint before state change commits
checkpoint['completed'] += 1
save_checkpoint(...)  # Files still in temp location!
move_files(...)

# GOOD: Update checkpoint after state change commits
move_files(...)  # Files now in final location
checkpoint['completed'] += 1
save_checkpoint(...)  # Checkpoint reflects committed state
```

**3. Separate Batch Operations Create Vulnerable Windows**

**Problem:** Stage 2 and Stage 2.5 separated → vulnerable window where state is inconsistent.

**Lesson:** Keep related operations atomic or immediate.

**Best Practice:**
```python
# BAD: Deferred batch operation
stage_2_process_video()  # Writes to temp
stage_2_5_organize_files()  # Moves to final (runs LATER)

# GOOD: Immediate operation
stage_2_process_video()
  ├─ Write to temp
  └─ IMMEDIATELY move to final (atomic)
```

### 10.2 Testing Lessons

**1. Test with Overlapping Data Sets**

**Problem:** Bug only appeared with overlapping video sets (same hashtags, different configs).

**Lesson:** Test scenarios must include realistic data overlaps.

**Test Cases to Add:**
- Multiple tests with same input data
- Multiple tests with overlapping input data
- Sequential execution with shared resources
- Concurrent execution with shared resources

**2. Test Sequential AND Concurrent Execution**

**Problem:** Assumed concurrent execution was the issue, but bug occurs with sequential too.

**Lesson:** Test both execution modes separately.

**Test Matrix:**
- Sequential: Test 4 completes → Test 3 starts
- Concurrent: Test 4 and Test 3 run simultaneously
- Interleaved: Test 4 Stage 2 → Test 3 Stage 2 → Test 4 Stage 2.5

**3. Validate Assumptions with Evidence**

**Problem:** E2EFixesv2.md listed "Concurrent Execution" as hypothesis without evidence.

**Lesson:** Distinguish between hypotheses and confirmed facts.

**Document Structure:**
```markdown
## Root Cause Analysis

### Confirmed Facts (Evidence-Based)
- ✅ Test 3 checkpoint contains 40 video IDs from Test 4
- ✅ Test 3 only processed 3 videos
- ✅ Files found in shared /insights/ directory

### Hypotheses (To Be Investigated)
- ❓ Hypothesis 1: Concurrent execution
  - Evidence needed: Timestamps, logs
- ❓ Hypothesis 2: Path resolution bug
  - Evidence needed: Path construction verification
```

### 10.3 Documentation Lessons

**1. Document Discovery Process, Not Just Solutions**

**Problem:** E2EFixesv2.md listed solutions without explaining how the bug was discovered.

**Lesson:** Document the journey, not just the destination.

**This Document Includes:**
- User insights that led to discovery
- Code analysis that revealed root cause
- Alternative hypotheses that were investigated
- Evidence that confirmed/rejected each hypothesis

**2. Separate Bug Analysis from Fix Proposals**

**Problem:** E2EFixesv2.md mixed bug analysis with enhancement proposals.

**Lesson:** Separate concerns for clarity.

**Document Structure:**
- Section 1-5: What happened (bug analysis)
- Section 6-9: How to fix it (proposals)
- Section 10: Why it happened (lessons)

**3. Include Timeline Diagrams**

**Problem:** Text descriptions of timing issues are hard to follow.

**Lesson:** Visual timelines clarify sequential/concurrent scenarios.

**This Document Includes:**
- Vulnerable window timeline
- Sequential execution scenario
- Before/after architecture diagrams

### 10.4 Collaboration Lessons

**1. User Questions Led to Discovery**

**Key Questions from User:**
> "Test 3 and 4 were scraping the same hashtags... both tests having same videos is logical"

> "I believe rumiai_runner.py does NOT download videos twice..."

> "At what stage is the checkpoint made?"

> "Shouldn't the checkpoint be AFTER files are moved?"

**Lesson:** User domain knowledge is critical for discovering architectural issues.

**2. Question Assumptions**

**Assumption (WRONG):** Concurrent execution caused the bug.

**Reality:** Sequential execution with shared resources caused the bug.

**Lesson:** Challenge initial hypotheses with evidence.

**3. Trace Complete Data Flow**

**Discovery Method:** Traced files from creation → validation → checkpoint update → file movement

**Lesson:** Follow data through entire pipeline to find gaps.

**Questions to Ask:**
- Where is data created?
- Where is data validated?
- When is state updated?
- When is data moved to final location?
- Who else can access data at each stage?

---

## Appendix A: File Locations Reference

### Current File Paths (OLD - Broken)

**Global Directories (Shared by ALL tests):**
```
/home/jorge/rumiaifinal/
├── insights/                              # GLOBAL (all tests write here)
│   ├── video_001_temporal_windows_updated.json
│   ├── video_042_temporal_windows_updated.json  # Which test owns this?
│   └── ...
├── temp/                                  # GLOBAL (all tests write here)
│   ├── video_001.mp4
│   ├── video_042.mp4
│   └── ...
└── unified_analysis/                      # GLOBAL (all tests write here)
    ├── video_001.json
    ├── video_042.json
    └── ...
```

**Test-Specific Directories (Checkpoints only):**
```
/home/jorge/rumiaifinal/data/clients/
├── rollo_test3/
│   └── hashtags/
│       └── wellness_test3/
│           └── top_contrastive/
│               ├── config.json
│               └── buckets/
│                   ├── bucket_3-9s/
│                   │   ├── checkpoints/
│                   │   │   └── stage_2_checkpoint.json  # Test 3 checkpoint
│                   │   ├── videos/                      # EMPTY (files in /temp/)
│                   │   └── analysis/
│                   │       └── insights/                # EMPTY (files in /insights/)
│                   └── ...
└── rollo_test4/
    └── hashtags/
        └── wellness_test4/
            └── top_contrastive/
                └── buckets/
                    ├── bucket_3-9s/
                    │   ├── checkpoints/
                    │   │   └── stage_2_checkpoint.json  # Test 4 checkpoint
                    │   ├── videos/                      # EMPTY
                    │   └── analysis/
                    │       └── insights/                # EMPTY
                    └── ...
```

### Fixed File Paths (NEW - After Fix)

**Global Directories (Temporary only - cleaned immediately):**
```
/home/jorge/rumiaifinal/
├── insights/                              # TEMPORARY (files moved immediately)
│   └── (empty - files moved immediately to bucket dirs)
├── temp/                                  # TEMPORARY
│   └── (empty - files moved immediately to bucket dirs)
└── unified_analysis/                      # TEMPORARY
    └── (empty - files moved immediately to bucket dirs)
```

**Test-Specific Directories (Files AND checkpoints):**
```
/home/jorge/rumiaifinal/data/clients/
├── rollo_test3/
│   └── hashtags/
│       └── wellness_test3/
│           └── top_contrastive/
│               ├── config.json
│               └── buckets/
│                   ├── bucket_3-9s/
│                   │   ├── checkpoints/
│                   │   │   └── stage_2_checkpoint.json  # Test 3 checkpoint
│                   │   ├── videos/                      # Test 3 videos HERE
│                   │   │   ├── video_001.mp4
│                   │   │   ├── video_002.mp4
│                   │   │   └── ...
│                   │   └── analysis/
│                   │       ├── insights/                # Test 3 insights HERE
│                   │       │   ├── video_001_temporal_windows_updated.json
│                   │       │   ├── video_002_temporal_windows_updated.json
│                   │       │   └── ...
│                   │       └── unified/                 # Test 3 unified HERE
│                   │           ├── video_001.json
│                   │           ├── video_002.json
│                   │           └── ...
│                   └── ...
└── rollo_test4/
    └── hashtags/
        └── wellness_test4/
            └── top_contrastive/
                └── buckets/
                    ├── bucket_3-9s/
                    │   ├── checkpoints/
                    │   │   └── stage_2_checkpoint.json  # Test 4 checkpoint
                    │   ├── videos/                      # Test 4 videos HERE (isolated!)
                    │   │   ├── video_042.mp4
                    │   │   ├── video_043.mp4
                    │   │   └── ...
                    │   └── analysis/
                    │       ├── insights/                # Test 4 insights HERE (isolated!)
                    │       │   ├── video_042_temporal_windows_updated.json
                    │       │   ├── video_043_temporal_windows_updated.json
                    │       │   └── ...
                    │       └── unified/
                    │           └── ...
                    └── ...
```

**Key Difference:**
- OLD: Files in GLOBAL `/insights/` and `/temp/` (shared by all tests)
- NEW: Files in TEST-SPECIFIC `/test3/buckets/.../insights/` and `/test4/buckets/.../insights/` (isolated)

---

## Appendix B: Code References

### Files to Modify

**Stage 2 Core Files:**
- `ml_pipeline/stage2_processing/video_processor.py` (main changes)
  - Line 30: `run_rumiai_pipeline()` - add immediate file isolation
  - Line 129: `process_videos_sequential()` - pass bucket_path, update validation
- `ml_pipeline/stage2_processing/main.py`
  - Line 139: `validate_stage_output()` - validate bucket-specific directories
- `ml_pipeline/stage2_processing/checkpoint.py`
  - Line 122: `validate_config_match()` - add ownership validation (Enhancement 3)
  - Line 145: After loading checkpoint - add provenance check (Enhancement 4)

**Stage 2.5 Files:**
- `ml_pipeline/stage2_5_organize/file_organizer.py`
  - Add skip logic if files already organized
  - Keep backward compatibility for old runs

**Configuration:**
- `rumiai_v2/config/settings.py` (no changes - rumiai_runner.py uses hardcoded dirs)

### External Dependencies

**rumiai_runner.py:**
- Location: `scripts/rumiai_runner.py`
- Current behavior: Writes to hardcoded `/insights/`, `/temp/`, `/unified_analysis/`
- Change needed: None (Stage 2 moves files immediately after rumiai_runner completes)

---

## Appendix C: Validation Enhancements (From E2EFixesv2.md)

### Enhancement 1: Count-Based Validation

**Location:** `ml_pipeline/stage2_processing/main.py::validate_stage_output()`

**Purpose:** Detect mismatches between checkpoint count and actual file count

**Implementation:**
```python
actual_count = len([f for f in os.listdir(insights_dir)
                   if f.endswith('_temporal_windows_updated.json')])
expected_count = checkpoint['completed']

if actual_count != expected_count:
    raise AssertionError(
        f"Expected {expected_count} files, found {actual_count}. "
        f"Missing: {expected_count - actual_count}"
    )
```

### Enhancement 2: Bucket-Specific Validation (Post Stage 2.5)

**Location:** `ml_pipeline/stage2_5_organize/validation.py` (new file)

**Purpose:** Validate files exist in bucket directories AFTER Stage 2.5 moves them

**Status:** NOT NEEDED if Stage 2 moves files immediately (files already in bucket dirs)

### Enhancement 3: Checkpoint Ownership Validation

**Location:** `ml_pipeline/stage2_processing/checkpoint.py::initialize_checkpoint()`

**Purpose:** Verify checkpoint belongs to THIS test (client_id, target match)

**Implementation:**
```python
def validate_checkpoint_ownership(checkpoint_config: dict, current_config: dict):
    ownership_fields = ['client_id', 'target']

    for field in ownership_fields:
        if checkpoint_config.get(field) != current_config.get(field):
            raise ValueError(
                f"Checkpoint ownership mismatch: {field}={checkpoint_config[field]} "
                f"(checkpoint) vs {current_config[field]} (current). "
                f"This checkpoint belongs to a different analysis."
            )

# Call after loading checkpoint (line 147)
validate_checkpoint_ownership(checkpoint['config'], config)
```

### Enhancement 4: Video ID Provenance Check

**Location:** `ml_pipeline/stage2_processing/checkpoint.py::initialize_checkpoint()`

**Purpose:** Ensure checkpoint video IDs are subset of selected_videos.json

**Implementation:**
```python
def validate_video_provenance(checkpoint: dict, video_list: list):
    selected_ids = set(v['id'] for v in video_list)
    checkpoint_ids = set(checkpoint['completed_video_ids'])

    unexpected_ids = checkpoint_ids - selected_ids

    if unexpected_ids:
        raise ValueError(
            f"Checkpoint contains {len(unexpected_ids)} video IDs NOT in selected_videos.json: "
            f"{list(unexpected_ids)[:10]}... "
            f"This indicates checkpoint corruption or wrong analysis."
        )

# Call after loading checkpoint (line 150)
validate_video_provenance(checkpoint, video_list)
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-30 | Claude (with User) | Initial documentation of bug discovery and fix proposal |

---

**END OF DOCUMENT**
