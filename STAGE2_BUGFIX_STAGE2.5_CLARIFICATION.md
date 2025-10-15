# Stage 2 Bug Fix & Stage 2.5 Clarification

**Date**: 2025-10-13
**Issue**: Stage 2 validation was checking wrong output path
**Status**: ✅ Fixed

---

## The Bug

### What Was Wrong

**Stage 2 Implementation Assumption** (INCORRECT):
```python
# BUG: Assumed rumiai_runner.py would output to bucket-specific directories
insights_path = f"{bucket_path}analysis/insights/{video_id}_temporal_windows_updated.json"
```

**Reality** (from FileOrganizationCHILD.md line 16):
> "Stage 2 (rumiai_runner.py) processes videos sequentially, saving all temporal_windows_updated.json files to a **flat /insights/ directory with no bucket awareness**"

Your existing `rumiai_runner.py` outputs to:
```
/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json
```

This is a **hardcoded path** in the original RumiAI script, not configurable.

---

## The Fix

### Files Modified

1. **ml_pipeline/stage2_processing/video_processor.py**
   - Added `RUMIAI_OUTPUT_DIR = "/home/jorge/rumiaifinal/insights/"` constant
   - Fixed `run_rumiai_pipeline()` to validate at correct path (line 56)
   - Fixed `process_videos_sequential()` to validate at correct path (line 146)

2. **ml_pipeline/stage2_processing/main.py**
   - Imported `RUMIAI_OUTPUT_DIR` constant
   - Fixed `validate_stage_output()` to check hardcoded directory (line 136)
   - Added clarifying comments about Stage 2.5's role

### Changes Made

```python
# BEFORE (WRONG):
insights_path = f"{output_dir}/insights/{video_id}_temporal_windows_updated.json"

# AFTER (CORRECT):
RUMIAI_OUTPUT_DIR = "/home/jorge/rumiaifinal/insights/"
insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
```

---

## Why Stage 2.5 Is Still Necessary

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Video Processing (BATCH WRAPPER)                   │
│ ─────────────────────────────────────────────────────────── │
│ For each video:                                             │
│   1. Download video → /data/.../bucket_18-33s/videos/       │
│   2. Call: python3 scripts/rumiai_runner.py video.mp4       │
│   3. rumiai_runner.py outputs to HARDCODED FLAT directory:  │
│      /home/jorge/rumiaifinal/insights/VIDEO_ID.json         │
│   4. Validate file exists at flat location ✓                │
│   5. Checkpoint progress                                    │
│                                                             │
│ Result: 300 files in FLAT directory, MIXED buckets         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2.5: File Organization (BUCKET ORGANIZER)             │
│ ─────────────────────────────────────────────────────────── │
│ 1. Read winner_analysis.json (which 3 buckets won)          │
│ 2. Read stage_2_checkpoint.json per bucket                  │
│ 3. Move files from flat to bucket-specific directories:     │
│                                                             │
│    FROM: /home/jorge/rumiaifinal/insights/VIDEO_ID.json    │
│    TO:   /data/.../bucket_18-33s/analysis/insights/...     │
│                                                             │
│ Result: Files organized by bucket, flat directory empty    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Feature Aggregation                                │
│ ─────────────────────────────────────────────────────────── │
│ Reads: /data/.../bucket_18-33s/analysis/insights/*.json    │
│ Expects: Files organized by bucket ← Needs Stage 2.5!      │
└─────────────────────────────────────────────────────────────┘
```

---

## Why rumiai_runner.py Outputs to Flat Directory

### Design Context

Your original `scripts/rumiai_runner.py` was designed for:
- **Single video processing** (not batch)
- **Development/testing workflow** (quick iteration)
- **Simple output management** (one flat directory)

It was NOT designed with:
- Bucket awareness
- Multi-tenant client structure
- Batch processing coordination

### Why We Don't Modify rumiai_runner.py

**Pros of keeping it unchanged**:
- ✅ Proven stable in production
- ✅ Simple single-video interface
- ✅ No risk of breaking existing workflows
- ✅ Separation of concerns (processing vs organization)

**Stage 2.5 approach**:
- ✅ Non-invasive (wraps existing script)
- ✅ Adds batch coordination layer
- ✅ Organizes outputs after processing
- ✅ Idempotent (safe to re-run)

---

## Stage 2.5 Implementation (Still TODO)

### What Needs to Be Built

Following **FileOrganizationCHILD.md** specification:

```python
# Stage 2.5 main function
def stage_2_5_file_organization_main(analysis_base: str):
    """
    Organize temporal_windows files from flat directory into bucket directories.

    Process:
    1. Load winner_analysis.json (top 3 buckets)
    2. Load stage_2_checkpoint.json per winning bucket
    3. Build file list (video_id → source/target paths)
    4. Validate no duplicates across buckets
    5. Move files with detection-based resume

    Source: FileOrganizationCHILD.md Section 2.3
    """

    # Load winning buckets
    winning_buckets = load_winning_buckets(analysis_base)
    # Result: ["18-33s", "33-60s", "13-18s"]

    # Build file list from checkpoints
    files_to_process = build_file_list(analysis_base, winning_buckets)
    # Result: [{video_id, bucket, source_path, target_path}, ...]

    # Validate no duplicates
    detect_duplicates_across_buckets(files_to_process)

    # Move files with auto-resume detection
    stats = organize_files_with_detection(files_to_process)
    # Result: {moved_count, skipped_already_organized, missing_count}

    return stats
```

### Key Features

- **Checkpoint-driven**: Reads Stage 2 checkpoints to know which files to organize
- **Detection-based resume**: Checks filesystem state, no need for Stage 2.5 checkpoint
- **Idempotent**: Safe to re-run (skips already-organized files)
- **Fast**: 300 files in ~5 seconds (file moves are quick)

### When to Run

```bash
# After Stage 2 completes for all winning buckets:
python -m ml_pipeline.stage2_5_organize \
  --analysis-base /data/clients/acme/hashtags/nutrition/top_contrastive/
```

---

## Testing Impact

### Before Fix
```
✗ Stage 2 validation FAILED
AssertionError: Missing insights for completed video 7428596413707144481
  Expected: /data/.../bucket_18-33s/analysis/insights/7428596413707144481.json
  Actual:   /home/jorge/rumiaifinal/insights/7428596413707144481.json
```

### After Fix
```
✓ Stage 2 validation PASSED
  Completed videos: 100
  Failed videos: 2
  Insights location: /home/jorge/rumiaifinal/insights/ (flat structure)
  Note: Stage 2.5 will organize files into bucket directories
```

---

## Summary

| Aspect | Stage 2 | Stage 2.5 |
|--------|---------|-----------|
| **Purpose** | Process videos through RumiAI | Organize outputs by bucket |
| **Input** | Video URLs (from Stage 1) | temporal_windows files (flat) |
| **Output** | Flat directory of JSON files | Bucket-organized JSON files |
| **Duration** | 8-10 hours (300 videos) | ~5 seconds (300 file moves) |
| **Resume** | Checkpoint per video | Detection-based (filesystem) |
| **Status** | ✅ Implemented & Fixed | ⏳ TODO (needs implementation) |

**Next Step**: Implement Stage 2.5 from FileOrganizationCHILD.md specification.
