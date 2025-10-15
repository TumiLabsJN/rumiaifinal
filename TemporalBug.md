# Temporal Windows Computation Bug - Silent Failures

---

## 🚨 FOR FRESH CLI INSTANCE

**If you're a new CLI session being told to read this document:**

This document describes a bug discovered during Stage 2 end-to-end testing where **3 videos (out of 114) silently failed** to produce output files despite `rumiai_runner.py` exiting with code 0 (success). The bug causes temporal windows computation failures to be caught and hidden, resulting in "no insights file" errors.

**Your task**: Investigate the actual exception being raised by `compute_temporal_windows()` for these 3 specific videos and determine the fix.

---

## 📋 Table of Contents

1. [Bug Summary](#bug-summary)
2. [How the Bug Was Discovered](#how-the-bug-was-discovered)
3. [Investigation Process](#investigation-process)
4. [Root Cause Analysis](#root-cause-analysis)
5. [Next Steps](#next-steps)
6. [Reference Data](#reference-data)

---

## Bug Summary

### The Problem

During Stage 2 video processing, 3 videos failed with this error:
```
RumiAI processing failed for {video_id} at stage output_validation:
RumiAI exited successfully (code 0) but no insights file at
/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json
```

### Key Characteristics

- **Exit Code**: 0 (success) ✅
- **stdout**: Normal processing messages ✅
- **stderr**: Error logged but buried ⚠️
- **Output File**: Missing ❌
- **Processing Time**: ~100 seconds (normal duration)

### Impact

- **Success Rate**: 111/114 videos (97.4%)
- **Failure Rate**: 3/114 videos (2.6%)
- **Pattern**: All failures in bucket `13-18s`, all low-engagement videos

---

## How the Bug Was Discovered

### Context: Stage 2 End-to-End Test

**Date**: 2025-10-14
**Test**: Full pipeline test of 150 videos across 3 winning buckets (18-33s, 13-18s, 60-90s)
**Command**: `python3 run_stage2_only.py`

### Initial Confusion: "117 Failures"

The final Stage 2 summary reported:
```
Stage 2 Summary: 111/24 videos processed, 117 failed
```

This seemed wrong because:
- We had 111 output files in `/insights/`
- Only 3 actual error messages in logs (not 117)

### Discovery: Stale Checkpoint Data

Investigation revealed the checkpoint files contained **stale failure data from a previous run**:

1. **First Run** (15:21:07): Used `pause_handler.py`, all 150 videos failed with `KeyError: 'downloadAddr'`
2. **Second Run** (15:37:15): Used `video_processor.py` with hybrid approach, 111 videos succeeded
3. **Bug**: Checkpoints weren't cleared between runs, so same videos appeared in BOTH `completed_video_ids` AND `failed_video_ids`

**Actual failures**: Only 3 videos, all with "no insights file" error

---

## Investigation Process

### Step 1: Identify the Real Failures

Filtered out the stale `'downloadAddr'` errors:
```bash
grep "Failed to process video" logs/stage2_only_*.log | grep -v "downloadAddr"
```

**Result**: Found exactly 3 failures with identical error pattern:
```
Failed to process video 7529376976470101304: RumiAI exited successfully (code 0) but no insights file
Failed to process video 7553533122717961490: RumiAI exited successfully (code 0) but no insights file
Failed to process video 7560992348897840396: RumiAI exited successfully (code 0) but no insights file
```

### Step 2: Analyze Video Metadata

Checked the 3 failed videos in `selected_videos.json`:

| Video ID | Duration | Views | Likes | Bucket |
|----------|----------|-------|-------|--------|
| 7529376976470101304 | 13s | 11,900 | 64 | 13-18s |
| 7553533122717961490 | 15s | 576 | 7 | 13-18s |
| 7560992348897840396 | 15s | 307 | 2 | 13-18s |

**Pattern observed**:
- All in bucket `13-18s` (26/50 succeeded in this bucket = 52% success rate)
- 2 of 3 are bottom 20% videos (very low engagement)
- Other buckets: `18-33s` (100% success), `60-90s` (70% success)

### Step 3: Examine Processing Logs

Extracted processing timeline for first failed video:
```
17:55:23 - Processing video 19/29: 7529376976470101304 (TikTok URL)
17:55:28 - rumiai_v2 - INFO - Processing video URL: https://www.tiktok.com/@seputaralatkesehatan/video/7529376976470101304
17:55:28 - rumiai_v2 - INFO - ✅ All ML dependencies validated successfully
17:55:28 - rumiai_v2 - INFO - ✅ GPU available: NVIDIA GeForce RTX 4060 Laptop GPU
17:55:28 - rumiai_v2 - INFO - 🚀 Starting processing for: https://...
[~100 seconds of processing]
17:57:08 - Failed to process video 7529376976470101304: RumiAI exited successfully (code 0) but no insights file
```

**Observations**:
- ML dependencies validated ✅
- GPU available ✅
- Processing started ✅
- Normal processing duration (~100s) ✅
- No obvious errors in visible logs ❓
- Exit code 0 but no output file ❌

### Step 4: Search for Output Files

Checked for any related files:
```bash
find /home/jorge/rumiaifinal -name "*7529376976470101304*"
```
**Result**: No files found - video processing left no trace

### Step 5: Trace Code Path

Located the critical code in `scripts/rumiai_runner.py:283-296`:

```python
# Step 6: Compute temporal windows
print("📊 computing_temporal_windows... (70%)")
prompt_results = {}

# Use temporal_compute with unified dict
try:
    temporal_windows = compute_temporal_windows(unified_analysis.to_dict())
    prompt_results['temporal_windows'] = temporal_windows

    # Save temporal windows as a single JSON file
    if temporal_windows:  # ← BUG: Empty dict evaluates to False!
        temporal_path = self.insights_handler.get_path(f"{video_id}_temporal_windows_updated.json")
        with open(temporal_path, 'w') as f:
            json.dump(temporal_windows, f, indent=2)
        logger.info(f"✅ Temporal windows saved to {temporal_path}")
except Exception as e:
    logger.error(f"Temporal windows computation failed: {e}")  # ← Logged to stderr
    prompt_results['temporal_windows'] = {}  # ← Continues with empty dict!

# Step 7: Generate final report (continues normally)
print("📊 generating_report... (95%)")
report = self._generate_report(unified_analysis, prompt_results)
```

**Next code**: After this try/except, processing continues to Step 7 and beyond, eventually returning `{"success": True}` with exit code 0.

---

## Root Cause Analysis

### The Silent Failure Mechanism

#### What Happens When `compute_temporal_windows()` Raises an Exception:

1. **Exception raised** in `temporal_compute.py:compute_temporal_windows()`
2. **Exception caught** on line 294 in `rumiai_runner.py`
3. **Error logged** to stderr: `"Temporal windows computation failed: {e}"`
4. **Empty dict assigned**: `prompt_results['temporal_windows'] = {}`
5. **File write skipped**: `if temporal_windows:` evaluates to `False` for `{}`
6. **Processing continues** to Step 7 (report generation)
7. **Returns success**: `{"success": True}` with exit code 0
8. **Stage 2 validation fails**: No file at expected path

### The Actual Bug

**Location**: `scripts/rumiai_runner.py:288-296`

**Problem**: Exception handling swallows the error and continues processing instead of failing fast.

```python
# CURRENT CODE (WRONG - Silent Failure)
try:
    temporal_windows = compute_temporal_windows(unified_analysis.to_dict())
    prompt_results['temporal_windows'] = temporal_windows

    if temporal_windows:  # Empty dict = False, file never written
        temporal_path = self.insights_handler.get_path(f"{video_id}_temporal_windows_updated.json")
        with open(temporal_path, 'w') as f:
            json.dump(temporal_windows, f, indent=2)
        logger.info(f"✅ Temporal windows saved to {temporal_path}")
except Exception as e:
    logger.error(f"Temporal windows computation failed: {e}")
    prompt_results['temporal_windows'] = {}  # Continues execution!

# EXPECTED BEHAVIOR (Fail Fast)
try:
    temporal_windows = compute_temporal_windows(unified_analysis.to_dict())

    # Validate result before continuing
    if not temporal_windows:
        raise ValueError("compute_temporal_windows returned empty result")

    # Save file
    temporal_path = self.insights_handler.get_path(f"{video_id}_temporal_windows_updated.json")
    with open(temporal_path, 'w') as f:
        json.dump(temporal_windows, f, indent=2)

    prompt_results['temporal_windows'] = temporal_windows
except Exception as e:
    logger.error(f"Temporal windows computation failed: {e}")
    raise  # Re-raise to propagate error and exit with code 1
```

### Why Exit Code is 0

The top-level exception handler in `rumiai_runner.py:330-347` is never reached because:
1. Exception is caught at the temporal computation level
2. Processing continues past the exception
3. Function returns normally with `{"success": True}`
4. No exception propagates to `main()`
5. Exit code defaults to 0

### Why No Output File

The file write is inside the `if temporal_windows:` conditional:
- When exception occurs, `temporal_windows = {}` (empty dict)
- Python evaluates `if {}:` as `False`
- File write code never executes
- No output file created

---

## Next Steps

### Priority 1: Find the Actual Exception ⚠️ CRITICAL

**Goal**: Determine what exception `compute_temporal_windows()` raised for these 3 videos.

**Option A: Search stderr logs (FASTEST)**
```bash
# Extract full stderr output for one of the failed videos
sed -n '/Processing video 19\/29: 7529376976470101304/,/Processing video 20\/29/p' \
  /home/jorge/rumiaifinal/data/logs/stage2_only_20251014_153715.log | \
  grep -A100 "RumiAI stderr:"
```

**Option B: Check unified_analysis files**
```bash
# See if unified_analysis was created (ML services succeeded)
ls -la /home/jorge/rumiaifinal/unified_analysis/ | grep "7529376976470101304"

# If exists, manually run temporal compute on it
python3 -c "
import json
from rumiai_v2.processors.temporal_compute import compute_temporal_windows

with open('/home/jorge/rumiaifinal/unified_analysis/7529376976470101304.json') as f:
    analysis = json.load(f)

try:
    result = compute_temporal_windows(analysis)
    print('SUCCESS:', result)
except Exception as e:
    print('ERROR:', type(e).__name__, str(e))
    import traceback
    traceback.print_exc()
"
```

**Option C: Re-run one failed video with full debug logging**
```bash
# Test with the first failed video
export LOG_LEVEL=DEBUG
python3 scripts/rumiai_runner.py "https://www.tiktok.com/@seputaralatkesehatan/video/7529376976470101304" 2>&1 | tee debug_7529376976470101304.log

# Check if it reproduces the error
ls -la /home/jorge/rumiaifinal/insights/ | grep "7529376976470101304"
```

### Priority 2: Investigate Why Only 13-18s Bucket Affected

**Goal**: Understand why failures are concentrated in the 13-18s bucket.

**Investigation Questions**:
1. Does the 13-18s bucket use a different temporal window structure?
   - Check: `temporal_compute.py` bucket thresholds
   - Bucket 13-18s should have **3 middle segments** (same as 9-13s)
   - Compare with 18-33s (4 segments) - had 100% success

2. Are low-engagement videos missing certain ML features?
   - Check unified_analysis for successful vs failed videos
   - Compare ML service outputs (YOLO, Whisper, MediaPipe, etc.)
   - Hypothesis: Bottom 20% videos might have no speech/text/faces

3. Is there a validation error in `compute_temporal_windows()`?
   - Check `temporal_compute.py:2382` for validation logic
   - Look for `ValueError` or `TypeError` raises
   - Check if certain ML data is assumed present but missing

### Priority 3: Implement Proper Error Handling

**Goal**: Make temporal computation failures fail fast instead of silently continuing.

**Required Changes** in `scripts/rumiai_runner.py:283-296`:

```python
# Step 6: Compute temporal windows
print("📊 computing_temporal_windows... (70%)")
prompt_results = {}

try:
    temporal_windows = compute_temporal_windows(unified_analysis.to_dict())

    # VALIDATION: Ensure non-empty result
    if not temporal_windows or not isinstance(temporal_windows, dict):
        raise ValueError(
            f"compute_temporal_windows returned invalid result: {type(temporal_windows)}"
        )

    # VALIDATION: Ensure required keys exist
    required_keys = ['hook', 'middle_segments', 'closing']
    missing = [k for k in required_keys if k not in temporal_windows]
    if missing:
        raise ValueError(
            f"compute_temporal_windows missing required keys: {missing}"
        )

    # Save file BEFORE assigning to prompt_results
    temporal_path = self.insights_handler.get_path(f"{video_id}_temporal_windows_updated.json")
    with open(temporal_path, 'w') as f:
        json.dump(temporal_windows, f, indent=2)
    logger.info(f"✅ Temporal windows saved to {temporal_path}")

    # Only assign after successful save
    prompt_results['temporal_windows'] = temporal_windows

except Exception as e:
    logger.error(f"Temporal windows computation failed: {e}")
    # FAIL FAST: Re-raise instead of continuing
    raise  # This will propagate to top-level handler and exit with code 1
```

**Expected Behavior After Fix**:
- Videos that fail temporal computation will exit with code 1
- Stage 2 processor will correctly mark them as failed
- No silent failures with exit code 0
- Errors will be visible in logs and debugging will be straightforward

---

## Reference Data

### Test Environment

- **Test Date**: 2025-10-14
- **Test Duration**: 15:37:15 - 19:55:57 (4 hours 18 minutes)
- **Total Videos**: 114 attempted (150 selected, 36 skipped due to stale checkpoints)
- **Success**: 111 videos (97.4%)
- **Failed**: 3 videos (2.6%)

### Bucket Performance

| Bucket | Videos Attempted | Succeeded | Failed | Success Rate |
|--------|------------------|-----------|---------|--------------|
| 18-33s | 50 | 50 | 0 | 100% ✅ |
| 13-18s | 29 | 26 | 3 | 89.7% ⚠️ |
| 60-90s | 35 | 35 | 0 | 100% ✅ |

**Note**: Bucket 13-18s had 21 videos skipped due to stale checkpoint data from first run.

### Failed Video Details

```json
{
  "failed_videos": [
    {
      "video_id": "7529376976470101304",
      "url": "https://www.tiktok.com/@seputaralatkesehatan/video/7529376976470101304",
      "duration": 13,
      "bucket": "13-18s",
      "engagement": {
        "playCount": 11900,
        "diggCount": 64,
        "shareCount": null
      },
      "processing_time": "~100 seconds",
      "error_time": "17:57:08",
      "error": "RumiAI exited successfully (code 0) but no insights file"
    },
    {
      "video_id": "7553533122717961490",
      "url": "https://www.tiktok.com/@unknown/video/7553533122717961490",
      "duration": 15,
      "bucket": "13-18s",
      "engagement": {
        "playCount": 576,
        "diggCount": 7,
        "shareCount": null
      },
      "processing_time": "~130 seconds",
      "error_time": "17:59:19",
      "error": "RumiAI exited successfully (code 0) but no insights file",
      "note": "Bottom 20% video"
    },
    {
      "video_id": "7560992348897840396",
      "url": "https://www.tiktok.com/@unknown/video/7560992348897840396",
      "duration": 15,
      "bucket": "13-18s",
      "engagement": {
        "playCount": 307,
        "diggCount": 2,
        "shareCount": null
      },
      "processing_time": "~105 seconds",
      "error_time": "18:01:05",
      "error": "RumiAI exited successfully (code 0) but no insights file",
      "note": "Bottom 20% video"
    }
  ]
}
```

### File Locations

- **Logs**: `/home/jorge/rumiaifinal/data/logs/stage2_only_20251014_153715.log`
- **Checkpoints**: `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_13-18s/checkpoints/stage_2_checkpoint.json`
- **Selected Videos**: `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_13-18s/selected_videos.json`
- **Output Directory**: `/home/jorge/rumiaifinal/insights/` (expected location, files missing)
- **Unified Analysis**: `/home/jorge/rumiaifinal/unified_analysis/` (check if exists)

### Code Locations

- **Bug Location**: `scripts/rumiai_runner.py:283-296` (exception handler)
- **Computation**: `rumiai_v2/processors/temporal_compute.py:2382` (compute_temporal_windows function)
- **Validation**: `rumiai_v2/processors/temporal_compute.py:2516-2540` (input validation)

---

## Quick Start for Fresh CLI

**To investigate this bug, execute these commands in order:**

```bash
# 1. Check if unified_analysis files exist
ls -la /home/jorge/rumiaifinal/unified_analysis/ | grep -E "7529376976470101304|7553533122717961490|7560992348897840396"

# 2. If files exist, manually test temporal compute
python3 -c "
import json
from rumiai_v2.processors.temporal_compute import compute_temporal_windows

video_id = '7529376976470101304'
try:
    with open(f'/home/jorge/rumiaifinal/unified_analysis/{video_id}.json') as f:
        analysis = json.load(f)
    result = compute_temporal_windows(analysis)
    print(f'SUCCESS: {len(result)} keys')
except FileNotFoundError:
    print(f'ERROR: unified_analysis/{video_id}.json not found')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
"

# 3. Search for the actual exception in logs
grep -B5 -A20 "Temporal windows computation failed" /home/jorge/rumiaifinal/data/logs/stage2_only_20251014_153715.log

# 4. If still unclear, re-run one failed video with debug logging
export LOG_LEVEL=DEBUG
python3 scripts/rumiai_runner.py "https://www.tiktok.com/@seputaralatkesehatan/video/7529376976470101304" 2>&1 | tee temporal_bug_debug.log
```

**Report back with**:
1. The actual exception type and message from `compute_temporal_windows()`
2. Whether unified_analysis files exist for the failed videos
3. Any patterns in the ML data that might explain the failures

---

## Success Criteria

This bug will be considered **resolved** when:

1. ✅ The actual exception cause is identified
2. ✅ Error handling is fixed to fail fast (no silent failures)
3. ✅ All 3 failed videos can be successfully re-processed OR
4. ✅ Known limitations are documented (if videos are inherently unprocessable)
5. ✅ Tests confirm exit code 1 for temporal computation failures (not 0)

---

**Last Updated**: 2025-10-14
**Status**: Investigation Required
**Priority**: High (affects 2.6% of videos)
