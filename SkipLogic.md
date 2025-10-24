# Skip Logic & Checkpoint System Analysis

## 📋 Document Overview

**Purpose:** Document the checkpoint and skip logic behavior across all pipeline stages (0-7)

**Status:** INCOMPLETE - Critical bugs identified in Stages 1 and 3

**Impact:** ~55 minutes + $0.80 wasted on every pipeline resume

**Last Updated:** 2025-10-22

---

## 🎯 What is Skip Logic?

**Skip Logic** is the mechanism that allows the pipeline to **resume from where it left off** without re-executing already-completed stages.

### Why It's Critical

When the pipeline pauses (e.g., Stage 2.6 manual curation) and you re-run the command:

**WITH proper skip logic:**
```bash
python rumiai_ml_batch.py ...
# Checks each stage → All done → Continues to new work → 11 seconds ✅
```

**WITHOUT proper skip logic:**
```bash
python rumiai_ml_batch.py ...
# Re-scrapes videos → Re-processes → Re-aggregates → 55 minutes + $0.80 ❌
```

### Two Types of Skip Logic

#### 1. **External Skip Logic (BEST)** ✅
Checks BEFORE calling the stage function:

```python
if checkpoint_exists():
    print("✓ Stage already complete (skipping)")
    continue  # Never calls stage function
else:
    run_stage()  # Only runs if needed
```

**Advantages:**
- Fast (just file check)
- Clean separation
- Clear logging

#### 2. **Internal Skip Logic (OK)** 🟡
Checks INSIDE the stage function:

```python
def run_stage():
    checkpoint, remaining = load_checkpoint()
    if len(remaining) == 0:
        return early  # Skips processing but function was called

    # Process remaining items
```

**Disadvantages:**
- Still calls function (overhead)
- Mixed concerns
- Less obvious in logs

#### 3. **No Skip Logic (BUG)** ❌
Always runs regardless of previous completion:

```python
def run_stage():
    # No checks
    scrape_all_videos()  # Always re-scrapes
    overwrite_results()  # Wastes time and money
```

**This is a bug that must be fixed.**

---

## 📊 Current State: Stage-by-Stage Analysis

### Stage 0: Foundation (Directory Setup & Config)

**Skip Logic Type:** None (but idempotent)

**Code Location:** `rumiai_ml_batch.py:544-564`

**Current Behavior:**
```python
# Creates directory structure
analysis_base.mkdir(parents=True, exist_ok=True)

# Saves configuration
config_path = analysis_base / "config.json"
ConfigManager.save(config, config_path)
```

**Issue:** Always runs, but operations are idempotent:
- `mkdir(exist_ok=True)` - Safe if directory exists
- Overwrites `config.json` - Harmless (same content)

**Impact:** ⚠️ **Low** - Takes ~1 second, no data corruption

**Status:** 🟢 **ACCEPTABLE** - Not a critical bug

---

### Stage 1: Video Discovery & Selection

**Skip Logic Type:** ❌ **NONE** (CRITICAL BUG)

**Code Location:** `rumiai_ml_batch.py:568-583`

**Current Behavior:**
```python
video_discovery = VideoDiscovery(
    config=config.model_dump(),
    apify_api_key=apify_api_key,
    path_builder=path_builder
)

exit_code = video_discovery.run()  # ❌ ALWAYS RUNS FULLY
```

**What Happens:**
1. ❌ Re-scrapes all hashtags from Apify (8 scrapes for wellness cluster)
2. ❌ Re-filters by date
3. ❌ Re-buckets by duration
4. ❌ Re-selects winners
5. ❌ Overwrites `winner_analysis.json`
6. ❌ Overwrites `cluster_analytics.json`

**Impact:** 🔴 **CRITICAL**
- **Time Wasted:** ~45 minutes
- **Cost Wasted:** ~$0.80 in Apify credits
- **Frequency:** Every resume after Stage 2.6 pause

**Output Files Created:**
- `cluster_analytics.json`
- `winner_analysis.json`
- `buckets/bucket_{duration}/selected_videos.json` (per bucket)

**Fix Required:** ✅ **YES - HIGH PRIORITY**

**Proposed Fix:**
```python
# Check if Stage 1 already complete
winner_analysis_path = analysis_base / "winner_analysis.json"

if winner_analysis_path.exists():
    logger.info("Stage 1 already complete (winner_analysis.json exists)")
    print("\n✓ Stage 1: Video Discovery - SKIPPED (already complete)")

    # Load winning buckets from existing file
    with open(winner_analysis_path) as f:
        winner_analysis = json.load(f)
    winning_buckets = winner_analysis['top_3_buckets']
else:
    # Run full Stage 1
    logger.info("Starting Stage 1: Video Discovery & Selection")
    video_discovery = VideoDiscovery(...)
    exit_code = video_discovery.run()

    if exit_code != 0:
        logger.error(f"Stage 1 failed with exit code {exit_code}")
        return exit_code

    # Load winning buckets from newly created file
    with open(winner_analysis_path) as f:
        winner_analysis = json.load(f)
    winning_buckets = winner_analysis['top_3_buckets']
```

---

### Stage 2: Video Processing

**Skip Logic Type:** 🟡 **Internal Checkpoint**

**Code Location:** `rumiai_ml_batch.py:604-643`

**Current Behavior:**
```python
for bucket_name in winning_buckets:
    # ALWAYS calls stage function
    summary = stage_2_video_processing_main(
        config=config.model_dump(),
        video_list=video_list,
        bucket_name=bucket_name,
        enable_pause_support=True
    )

    # INSIDE stage_2_video_processing_main():
    checkpoint, remaining_videos = initialize_checkpoint(bucket_name, video_list, config)
    if len(remaining_videos) == 0:
        # All videos already processed
        finalize_checkpoint(checkpoint, config)
        return {
            'total': checkpoint['total_videos'],
            'completed': checkpoint['completed'],
            'failed': checkpoint['failed']
        }
```

**What Happens:**
1. ✅ Calls stage function for each bucket
2. ✅ Internal checkpoint check: loads `.stage2_checkpoint.json`
3. ✅ Filters `remaining_videos = [v for v in video_list if v not in completed]`
4. ✅ If `remaining_videos == []`, returns immediately
5. ✅ Only processes videos not in checkpoint

**Impact:** 🟢 **LOW**
- **Time Wasted:** ~5 seconds (checkpoint load + validation)
- **Cost Wasted:** $0
- **Data Safety:** ✅ No re-processing, no overwriting

**Checkpoint File:** `buckets/bucket_{duration}/.stage2_checkpoint.json`

**Status:** 🟢 **ACCEPTABLE** - Works correctly, could be optimized

**Potential Improvement:** Convert to external skip logic:
```python
checkpoint_path = bucket_path / ".stage2_checkpoint.json"
if checkpoint_path.exists():
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)
    if checkpoint['completed'] == checkpoint['total_videos']:
        print(f"✓ Bucket {bucket_name}: Processing already complete (skipping)")
        continue
```

---

### Stage 2.5: File Organization

**Skip Logic Type:** 🟡 **Internal Skip**

**Code Location:** `rumiai_ml_batch.py:654-682`

**Current Behavior:**
```python
organization_summary = stage_2_5_file_organization_main(
    analysis_base=str(analysis_base)
)

# INSIDE stage_2_5_file_organization_main():
# Checks if each file is already in organized location
if organized_file_path.exists():
    skipped_already_organized += 1
    continue
else:
    shutil.move(flat_file, organized_file_path)
    moved_count += 1
```

**What Happens:**
1. ✅ Calls stage function
2. ✅ Scans flat `insights/` directory for temporal_windows files
3. ✅ For each file, checks if already in `buckets/bucket_{duration}/insights/`
4. ✅ Skips files already organized
5. ✅ Moves only unorganized files

**Impact:** 🟢 **LOW**
- **Time Wasted:** ~1 second (file system scan)
- **Cost Wasted:** $0
- **Data Safety:** ✅ No file duplication, safe move

**Status:** 🟢 **ACCEPTABLE** - Works correctly by design

---

### Stage 2.6: Pattern Discovery (Content Analysis)

**Skip Logic Type:** ✅ **External Check (PERFECT)**

**Code Location:** `rumiai_ml_batch.py:703-764`

**Current Behavior:**
```python
taxonomy_path = taxonomy_dir / f"{target_sanitized}_taxonomy.json"

if not taxonomy_path.exists():
    # Taxonomy doesn't exist - run discovery
    logger.info("Taxonomy not found - running Stage 2.6: Pattern Discovery")

    raw_taxonomy = run_discovery_stage(...)

    # PAUSE FOR MANUAL CURATION
    display_curation_instructions(raw_discovery_path, taxonomy_path)
    return 2  # Exit code 2 = paused for manual step
else:
    # Taxonomy exists - skip discovery
    logger.info("Taxonomy found - skipping Stage 2.6 (already complete)")
    print("\n✓ Stage 2.6: Pattern Discovery - SKIPPED (taxonomy exists)")
```

**What Happens:**
1. ✅ Checks if `{hashtag}_taxonomy.json` exists BEFORE calling stage
2. ✅ If exists, skips entirely (0 seconds)
3. ✅ If not exists, runs discovery, then pauses (exit code 2)

**Impact:** 🟢 **PERFECT**
- **Time Wasted:** 0 seconds
- **Cost Wasted:** $0

**Checkpoint File:** `content_taxonomies/{hashtag}_taxonomy.json`

**Status:** ✅ **EXCELLENT** - This is the correct pattern

---

### Stage 2.7: Video Classification

**Skip Logic Type:** 🟡 **Internal Checkpoint**

**Code Location:** `rumiai_ml_batch.py:766-809`

**Current Behavior:**
```python
summary = run_classification_stage(
    client_id=sanitize_client_id(cli_args.client),
    hashtag=cli_args.target,
    analysis_type=cli_args.analysis_type,
    analysis_mode=cli_args.analysis_mode,
    selection_strategy=cli_args.selection_strategy,
    parallel=enable_parallel,
    max_workers=max_workers,
    checkpoint_enabled=True  # Enable checkpoint/resume
)

# INSIDE run_classification_stage():
checkpoint, remaining_videos = load_checkpoint(...)
if len(remaining_videos) == 0:
    return {
        'completed': checkpoint['completed'],
        'total': checkpoint['total'],
        'failed': checkpoint['failed']
    }
```

**What Happens:**
1. ✅ Calls stage function
2. ✅ Internal checkpoint check: loads `.classification_checkpoint.json`
3. ✅ Filters remaining videos not yet classified
4. ✅ If all classified, returns immediately
5. ✅ Only processes unclassified videos

**Impact:** 🟢 **LOW**
- **Time Wasted:** ~5 seconds (checkpoint load + validation)
- **Cost Wasted:** $0

**Checkpoint File:** `buckets/bucket_{duration}/.classification_checkpoint.json`

**Status:** 🟢 **ACCEPTABLE** - Works correctly, could be optimized

---

### Stage 3: Feature Aggregation

**Skip Logic Type:** ❌ **NONE** (MEDIUM BUG)

**Code Location:** `rumiai_ml_batch.py:850-983`

**Current Behavior:**
```python
for bucket_name in winning_buckets:
    bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"

    # Validation checks (directory exists, files exist)
    insights_dir = bucket_path / "analysis" / "insights"
    if not insights_dir.exists():
        logger.error("Insights directory missing")
        continue

    # ❌ NO CHECKPOINT CHECK
    # ALWAYS runs aggregation
    csv_path, summary_path = aggregate_features(str(bucket_path))
```

**What Happens:**
1. ✅ Validates prerequisites (insights directory, JSON files)
2. ❌ NO check for existing `aggregated_features.csv`
3. ❌ Re-reads all temporal_windows JSON files
4. ❌ Re-aggregates features into CSV
5. ❌ Overwrites `aggregated_features.csv`

**Impact:** 🟡 **MEDIUM**
- **Time Wasted:** ~10 minutes (reads 240 JSON files, aggregates)
- **Cost Wasted:** $0
- **Data Integrity:** ⚠️ Overwrites existing CSV (deterministic, but wasteful)

**Output Files Created:**
- `buckets/bucket_{duration}/ml_analysis/aggregated_features.csv`
- `buckets/bucket_{duration}/checkpoints/stage_3_checkpoint.json`

**Fix Required:** ✅ **YES - MEDIUM PRIORITY**

**Proposed Fix:**
```python
for bucket_name in winning_buckets:
    bucket_path = analysis_base / f"buckets/bucket_{bucket_name}"
    checkpoint_path = bucket_path / "checkpoints" / "stage_3_checkpoint.json"

    # Check if Stage 3 already complete
    if checkpoint_path.exists():
        logger.info(f"Bucket {bucket_name}: Stage 3 already complete (checkpoint exists)")
        print(f"✓ Bucket {bucket_name}: Aggregation already complete (skipping)")

        # Load checkpoint for summary
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        stage3_summaries[bucket_name] = checkpoint
        continue

    # Otherwise run Stage 3
    csv_path, summary_path = aggregate_features(str(bucket_path))
    ...
```

**Note:** Stage 3 DOES create a checkpoint file (`stage_3_checkpoint.json`), but it's only used by Stage 4 for prerequisite validation, NOT for Stage 3 skip logic.

---

### Stage 4: Feature Transformation

**Skip Logic Type:** ✅ **External Check (PERFECT)**

**Code Location:** `rumiai_ml_batch.py:1015-1029`

**Current Behavior:**
```python
# Validate Stage 3 completed successfully
stage3_checkpoint = bucket_path / "checkpoints" / "stage_3_checkpoint.json"
if not stage3_checkpoint.exists():
    logger.error(f"Bucket {bucket_name}: Stage 3 checkpoint missing")
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
```

**What Happens:**
1. ✅ Validates prerequisite: Stage 3 checkpoint exists
2. ✅ Checks if `stage_4_checkpoint.json` exists BEFORE calling stage
3. ✅ If exists, skips entirely (0 seconds)
4. ✅ Only runs if checkpoint missing

**Impact:** 🟢 **PERFECT**
- **Time Wasted:** 0 seconds
- **Cost Wasted:** $0

**Checkpoint File:** `buckets/bucket_{duration}/checkpoints/stage_4_checkpoint.json`

**Status:** ✅ **EXCELLENT** - This is the correct pattern

---

### Stage 5: ML Model Training

**Skip Logic Type:** ✅ **External Check (PERFECT)**

**Code Location:** `rumiai_ml_batch.py:1180-1194`

**Current Behavior:**
```python
# Validate Stage 4 completed successfully
stage4_checkpoint = bucket_path / "checkpoints" / "stage_4_checkpoint.json"
if not stage4_checkpoint.exists():
    logger.error(f"Bucket {bucket_name}: Stage 4 checkpoint missing")
    print(f"✗ Bucket {bucket_name}: Stage 4 not complete (skipping)")
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
```

**What Happens:**
1. ✅ Validates prerequisite: Stage 4 checkpoint exists
2. ✅ Checks if `stage_5_checkpoint.json` exists BEFORE calling stage
3. ✅ If exists, skips entirely (0 seconds)
4. ✅ Only runs if checkpoint missing

**Impact:** 🟢 **PERFECT**
- **Time Wasted:** 0 seconds
- **Cost Wasted:** $0

**Checkpoint File:** `buckets/bucket_{duration}/checkpoints/stage_5_checkpoint.json`

**Status:** ✅ **EXCELLENT** - Consistent with Stage 4 pattern

---

### Stage 6: ML Analysis Generation

**Skip Logic Type:** ✅ **External Check (PERFECT)**

**Code Location:** `rumiai_ml_batch.py:1352-1366`

**Current Behavior:**
```python
# Validate Stage 5 completed successfully
stage5_checkpoint = bucket_path / "checkpoints" / "stage_5_checkpoint.json"
if not stage5_checkpoint.exists():
    logger.error(f"Bucket {bucket_name}: Stage 5 checkpoint missing")
    print(f"✗ Bucket {bucket_name}: Stage 5 not complete (skipping)")
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
```

**What Happens:**
1. ✅ Validates prerequisite: Stage 5 checkpoint exists
2. ✅ Checks if `stage_6_checkpoint.json` exists BEFORE calling stage
3. ✅ If exists, skips entirely (0 seconds)
4. ✅ Only runs if checkpoint missing

**Impact:** 🟢 **PERFECT**
- **Time Wasted:** 0 seconds
- **Cost Wasted:** $0

**Checkpoint File:** `buckets/bucket_{duration}/checkpoints/stage_6_checkpoint.json`

**Status:** ✅ **EXCELLENT** - Consistent pattern continues

---

### Stage 7: LLM Report Generation

**Skip Logic Type:** ✅ **External Check (PERFECT)**

**Code Location:** `rumiai_ml_batch.py:1536-1553`

**Current Behavior:**
```python
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
```

**What Happens:**
1. ✅ Checks if `complete_analysis.json` exists BEFORE calling stage
2. ✅ If exists, skips entirely (0 seconds)
3. ✅ Only runs if file missing

**Impact:** 🟢 **PERFECT**
- **Time Wasted:** 0 seconds
- **Cost Wasted:** $0

**Checkpoint File:** `buckets/bucket_{duration}/ml_analysis/llm/complete_analysis.json`

**Status:** ✅ **EXCELLENT** - Final stage, consistent pattern

---

## 🐛 Bug Summary

### Bug #1: Stage 1 Missing Skip Logic (CRITICAL)

**Severity:** 🔴 **CRITICAL**

**Location:** `rumiai_ml_batch.py:568-583`

**Issue:** Video Discovery stage ALWAYS re-scrapes all videos from Apify, even when `winner_analysis.json` already exists from a previous run.

**Impact:**
- **Time Wasted:** ~45 minutes per resume
- **Cost Wasted:** ~$0.80 in Apify credits per resume
- **Trigger:** Every pipeline resume after Stage 2.6 manual curation pause

**Example Scenario:**
```
10:00 AM - Run pipeline
10:45 AM - Stage 1 complete (scrapes 8 times, $0.80 spent)
12:45 PM - Stage 2.6 pauses for manual curation
01:00 PM - User curates taxonomy
01:00 PM - Re-run pipeline
01:00 PM - ❌ Stage 1 re-scrapes AGAIN (another 45 min, another $0.80)
```

**Root Cause:**
```python
# Current code has NO check
video_discovery = VideoDiscovery(...)
exit_code = video_discovery.run()  # Always runs
```

**Expected Behavior:**
```python
# Should check for existing output first
if (analysis_base / "winner_analysis.json").exists():
    print("✓ Stage 1 already complete (skipping)")
else:
    video_discovery.run()
```

**Status:** ✅ **FIXED (2025-10-24)** - Checkpoint pattern implemented

**Implementation Notes:**
- Pattern: External checkpoint (consistent with Stages 4-7 A+)
- Checkpoint tracks: 4+ output files (winner_analysis.json, 3× selected_videos.json, cluster_analytics.json if cluster mode)
- Recovery: Automatic from corrupt checkpoints (delete + re-run)
- Logging: Simple skip messages (clean logs)
- Metrics: Tracked in logs only, not checkpoint (focus on skip logic)
- Testing: Test suite documents 5 key scenarios

---

### Bug #2: Stage 3 Missing Skip Logic (MEDIUM)

**Severity:** 🟡 **MEDIUM**

**Location:** `rumiai_ml_batch.py:850-983`

**Issue:** Feature Aggregation stage ALWAYS re-aggregates features from JSON files, even when `aggregated_features.csv` and `stage_3_checkpoint.json` already exist.

**Impact:**
- **Time Wasted:** ~10 minutes per resume
- **Cost Wasted:** $0 (local processing, no API calls)
- **Trigger:** Every pipeline resume after Stage 2.6 manual curation pause

**Example Scenario:**
```
10:00 AM - Run pipeline
03:00 PM - Stage 3 complete (aggregates 240 videos → CSV)
03:05 PM - Stage 2.6 pauses for manual curation
04:00 PM - User curates taxonomy
04:00 PM - Re-run pipeline
04:55 PM - ❌ Stage 3 re-aggregates AGAIN (another 10 min)
```

**Root Cause:**
```python
# Current code has NO checkpoint check
for bucket_name in winning_buckets:
    # Validates prerequisites exist
    # But does NOT check if Stage 3 already complete
    csv_path, summary_path = aggregate_features(str(bucket_path))  # Always runs
```

**Expected Behavior:**
```python
checkpoint_path = bucket_path / "checkpoints" / "stage_3_checkpoint.json"
if checkpoint_path.exists():
    print("✓ Stage 3 already complete (skipping)")
    continue
else:
    aggregate_features(str(bucket_path))
```

**Fix Priority:** 🟡 **MEDIUM** - Should implement for efficiency

**Workaround:** Accept the 10-minute overhead on resume

---

## 📊 Skip Logic Report Card

| Stage | Skip Logic | Grade | Time on Resume | Notes |
|-------|-----------|-------|----------------|-------|
| **Stage 0** | None (idempotent) | B | 1 sec | Harmless but could check config.json |
| **Stage 1** | External check (checkpoint) | **A+** | **0 sec** | ✅ **Fixed (2025-10-24)** - Checkpoint pattern |
| **Stage 2** | Internal checkpoint | B+ | 5 sec | Works but could be external |
| **Stage 2.5** | Internal skip | B+ | 1 sec | Works by design |
| **Stage 2.6** | External check | A+ | 0 sec | ✅ Perfect implementation |
| **Stage 2.7** | Internal checkpoint | B+ | 5 sec | Works but could be external |
| **Stage 3** | ❌ **NONE** | F | ❌ **10 min** | **MEDIUM BUG** - Re-aggregates all features |
| **Stage 4** | External check | A+ | 0 sec | ✅ Perfect implementation |
| **Stage 5** | External check | A+ | 0 sec | ✅ Perfect implementation |
| **Stage 6** | External check | A+ | 0 sec | ✅ Perfect implementation |
| **Stage 7** | External check | A+ | 0 sec | ✅ Perfect implementation |

**Overall Grade:** B+ (Stages 1, 4-7 excellent, Stage 3 needs fix)

**Total Resume Overhead (Before Stage 1 Fix):** ~56 seconds + ~55 minutes waste

**Total Resume Overhead (After Stage 1 Fix):** ~16 seconds + ~10 minutes (Stage 3 only)

**Total Resume Overhead (After All Fixes):** ~13 seconds (just validation checks)

---

## 💰 Cost Analysis

### Current Situation (With Bugs)

**Per Resume After Stage 2.6 Pause:**

| Stage | Time | Cost | Avoidable? |
|-------|------|------|------------|
| Stage 0 | 1 sec | $0 | No (idempotent) |
| Stage 1 | ✅ 0 sec | ✅ $0 | No (✅ **Fixed 2025-10-24**) |
| Stage 2 | 5 sec | $0 | No (works correctly) |
| Stage 2.5 | 1 sec | $0 | No (works correctly) |
| Stage 2.6 | 0 sec | $0 | No (works correctly) |
| Stage 2.7 | 5 sec | $0 | No (works correctly) |
| Stage 3 | ❌ 10 min | $0 | ✅ **YES** (Bug #2) |
| Stage 4 | 0 sec | $0 | No (works correctly) |
| Stage 5 | 0 sec | $0 | No (works correctly) |
| Stage 6 | 0 sec | $0 | No (works correctly) |
| Stage 7 | 0 sec | $0 | No (works correctly) |
| **TOTAL** | **~56 min** | **$0.80** | **55 min avoidable** |

---

### Fixed Situation (Bugs Resolved)

**Per Resume After Stage 2.6 Pause:**

| Stage | Time | Cost | Notes |
|-------|------|------|-------|
| Stage 0 | 1 sec | $0 | Idempotent operations |
| Stage 1 | ✅ 0 sec | ✅ $0 | **Checks checkpoint + all output files** |
| Stage 2 | 5 sec | $0 | Checkpoint validation |
| Stage 2.5 | 1 sec | $0 | File scan |
| Stage 2.6 | 0 sec | $0 | Taxonomy exists check |
| Stage 2.7 | 5 sec | $0 | Checkpoint validation |
| Stage 3 | ✅ 0 sec | ✅ $0 | **Checks stage_3_checkpoint.json** |
| Stage 4-7 | 0 sec | $0 | Already have skip logic |
| **TOTAL** | **~13 sec** | **$0** | **All validation, no waste** |

**Savings per resume:** 55 minutes + $0.80

**Annual savings (10 resumes):** 9 hours + $8

---

## 🔧 Fixes Required

### Fix #1: Add Stage 1 Skip Logic (CRITICAL)

**Priority:** 🔴 **HIGH**

**Effort:** ~30 minutes

**Files Modified:** `rumiai_ml_batch.py`

**Location:** Lines 568-583

**Implementation:**

```python
# ===== STAGE 1: VIDEO DISCOVERY & SELECTION =====
logger.info("Starting Stage 1: Video Discovery & Selection")

# Check if Stage 1 already complete
winner_analysis_path = analysis_base / "winner_analysis.json"

if winner_analysis_path.exists():
    logger.info("Stage 1 already complete (winner_analysis.json exists)")
    print("\n" + "="*80)
    print("STAGE 1: VIDEO DISCOVERY & SELECTION")
    print("="*80)
    print("\n✓ Stage 1: Video Discovery - SKIPPED (already complete)")
    print(f"  Found existing: {winner_analysis_path}")

    # Load winning buckets from existing file
    with open(winner_analysis_path) as f:
        winner_analysis = json.load(f)
    winning_buckets = winner_analysis['top_3_buckets']

    logger.info(f"Loaded {len(winning_buckets)} winning buckets: {winning_buckets}")
    print(f"  Winning buckets: {', '.join(winning_buckets)}\n")
else:
    # Run full Stage 1
    video_discovery = VideoDiscovery(
        config=config.model_dump(),
        apify_api_key=apify_api_key,
        path_builder=path_builder
    )

    exit_code = video_discovery.run()

    if exit_code != 0:
        logger.error(f"Stage 1 failed with exit code {exit_code}")
        return exit_code

    logger.info("Stage 1 completed successfully")

    # Load winning buckets from newly created file
    with open(winner_analysis_path) as f:
        winner_analysis = json.load(f)
    winning_buckets = winner_analysis['top_3_buckets']
```

**Testing:**
```bash
# Test 1: Fresh run (no winner_analysis.json)
rm -f data/clients/test/hashtag/wellness/winner_analysis.json
python rumiai_ml_batch.py ...
# Expected: Stage 1 runs fully

# Test 2: Resume (winner_analysis.json exists)
python rumiai_ml_batch.py ...
# Expected: Stage 1 skipped instantly
```

**Rollback Plan:** If issues occur, revert to original code (always runs Stage 1)

---

### Fix #2: Add Stage 3 Skip Logic (MEDIUM)

**Priority:** 🟡 **MEDIUM**

**Effort:** ~20 minutes

**Files Modified:** `rumiai_ml_batch.py`

**Location:** Lines 850-983

**Implementation:**

```python
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
    checkpoint_path = bucket_path / "checkpoints" / "stage_3_checkpoint.json"

    # Check if Stage 3 already complete for this bucket
    if checkpoint_path.exists():
        logger.info(f"Bucket {bucket_name}: Stage 3 already complete (checkpoint exists)")
        print(f"✓ Bucket {bucket_name}: Aggregation already complete (skipping)")

        # Load checkpoint for summary reporting
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        stage3_summaries[bucket_name] = checkpoint
        continue

    # Otherwise run Stage 3 aggregation
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

        # ... rest of Stage 3 logic
        csv_path, summary_path = aggregate_features(str(bucket_path))

        # Load summary for reporting
        with open(summary_path) as f:
            summary = json.load(f)

        stage3_summaries[bucket_name] = summary

        logger.info(
            f"Bucket {bucket_name} complete: "
            f"{summary['videos_processed']}/{summary['input_files_found']} videos aggregated"
        )
        print(
            f"✓ Bucket {bucket_name}: {summary['videos_processed']} videos → "
            f"{summary['output_csv']['columns']} features"
        )

    except Exception as e:
        # Error handling...
        logger.error(f"Stage 3 failed for bucket {bucket_name}: {e}")
        print(f"✗ Bucket {bucket_name} failed: {e}")
        continue

logger.info("Stage 3 completed for all buckets")
print("\n✓ Stage 3: Feature Aggregation - COMPLETE")
```

**Testing:**
```bash
# Test 1: Fresh run (no checkpoint)
rm -f data/clients/test/hashtag/wellness/top_contrastive/buckets/*/checkpoints/stage_3_checkpoint.json
python rumiai_ml_batch.py ...
# Expected: Stage 3 runs fully

# Test 2: Resume (checkpoint exists)
python rumiai_ml_batch.py ...
# Expected: Stage 3 skipped for buckets with checkpoint
```

**Rollback Plan:** Revert to original code (always runs Stage 3)

---

## 📝 Implementation Checklist

### Phase 1: Critical Fix (Stage 1)
- [ ] Add skip logic check for `winner_analysis.json`
- [ ] Test fresh run (Stage 1 executes)
- [ ] Test resume (Stage 1 skips)
- [ ] Verify winning_buckets loaded correctly on skip
- [ ] Update E2ETest.md with new expected behavior
- [ ] Commit changes

### Phase 2: Efficiency Fix (Stage 3)
- [ ] Add skip logic check for `stage_3_checkpoint.json`
- [ ] Test fresh run (Stage 3 executes)
- [ ] Test resume (Stage 3 skips)
- [ ] Verify stage3_summaries populated correctly on skip
- [ ] Update E2ETest.md with new expected behavior
- [ ] Commit changes

### Phase 3: Optional Optimizations
- [ ] Convert Stage 2 internal checkpoint to external check
- [ ] Convert Stage 2.7 internal checkpoint to external check
- [ ] Add Stage 0 skip logic (check config.json exists)
- [ ] Performance benchmark before/after fixes

---

## 🧪 Test Scenarios

### Scenario 1: Fresh Pipeline Run
**Expected:** All stages execute normally, no skip logic triggers

```bash
# Clean test environment
rm -rf data/clients/test_skip_logic

# Run full pipeline
python rumiai_ml_batch.py \
  --client test_skip_logic \
  --target wellness \
  --analysis-type hashtag

# Expected:
# Stage 0: Runs (creates directories)
# Stage 1: Runs (scrapes videos)
# Stage 2: Runs (processes videos)
# Stage 3: Runs (aggregates features)
# Stages 4-7: Run normally
```

---

### Scenario 2: Resume After Stage 2.6 Pause (CURRENT BUG)
**Expected (Current):** Stages 1 and 3 re-run wastefully

```bash
# First run - pauses at Stage 2.6
python rumiai_ml_batch.py ...
# → Pauses for manual curation

# Curate taxonomy
vi data/clients/test_skip_logic/hashtag/wellness/top_contrastive/content_taxonomies/wellness_taxonomy.json

# Resume pipeline
python rumiai_ml_batch.py ...

# Current behavior:
# Stage 0: Runs (1 sec)
# Stage 1: ❌ RE-RUNS FULLY (45 min + $0.80) ← BUG
# Stage 2: Checks checkpoint, skips (5 sec)
# Stage 2.5: Skips organized files (1 sec)
# Stage 2.6: Skips (taxonomy exists)
# Stage 2.7: Runs classification
# Stage 3: ❌ RE-RUNS FULLY (10 min) ← BUG
# Stages 4-7: Skip correctly (checkpoints exist)
```

---

### Scenario 3: Resume After Stage 2.6 Pause (FIXED)
**Expected (After Fix):** All stages skip correctly

```bash
# After implementing Fix #1 and Fix #2

# Resume pipeline
python rumiai_ml_batch.py ...

# Fixed behavior:
# Stage 0: Runs (1 sec) - idempotent
# Stage 1: ✅ SKIPS (winner_analysis.json exists) - 0 sec
# Stage 2: Checks checkpoint, skips (5 sec)
# Stage 2.5: Skips organized files (1 sec)
# Stage 2.6: Skips (taxonomy exists) - 0 sec
# Stage 2.7: Runs classification (only unclassified videos)
# Stage 3: ✅ SKIPS (stage_3_checkpoint.json exists) - 0 sec
# Stages 4-7: Skip correctly (checkpoints exist) - 0 sec
```

**Total time:** ~13 seconds (all validation, no work)

---

### Scenario 4: Partial Stage 2 Completion
**Expected:** Stage 2 checkpoint resumes from last processed video

```bash
# Run pipeline
python rumiai_ml_batch.py ...

# Interrupt during Stage 2 (Ctrl+C)
# → Stage 2 checkpoint saved

# Resume
python rumiai_ml_batch.py ...

# Expected:
# Stage 1: ✅ Skips (winner_analysis.json exists)
# Stage 2: Loads checkpoint, processes ONLY remaining videos
# Stage 3+: Continue normally
```

---

## 📚 Related Documentation

- **E2ETest_Wellness_Rollo.md** - End-to-end test specification
- **CLUSTER_QUICK_START.md** - Cluster setup guide
- **SystemArchitecturev2.md** - System architecture overview

---

## 🔄 Changelog

### Version 1.0 (2025-10-22)
- Initial analysis of skip logic across all stages
- Identified Bug #1: Stage 1 missing skip logic (CRITICAL)
- Identified Bug #2: Stage 3 missing skip logic (MEDIUM)
- Documented current behavior and proposed fixes
- Created implementation checklist

---

## 📞 Support

**Bug Reports:** If you encounter skip logic issues, check:

1. **Stage checkpoint files exist:**
   ```bash
   find data/clients/*/hashtag/*/top_contrastive -name "*checkpoint.json"
   ```

2. **Winner analysis file exists:**
   ```bash
   ls data/clients/*/hashtag/*/winner_analysis.json
   ```

3. **Taxonomy file exists:**
   ```bash
   ls data/clients/*/hashtag/*/top_contrastive/content_taxonomies/*_taxonomy.json
   ```

4. **Logs show skip logic:**
   ```bash
   grep -i "already complete\|skipping" data/logs/rumiai_ml_*.log
   ```

---

**Document Status:** 🔴 **INCOMPLETE** - Critical bugs documented, fixes required before production use

**Next Steps:** Implement Fix #1 (Stage 1 skip logic) as HIGH PRIORITY
