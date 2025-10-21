# Bug Report: RumiAI Temporal Windows Schema Mismatch

**Date Discovered**: 2025-10-21
**Severity**: HIGH
**Component**: RumiAI Pipeline (rumiai_runner.py, temporal_compute)
**Status**: ACTIVE - Blocking Stage 2 video processing

---

## Executive Summary

RumiAI pipeline fails with schema validation error when processing videos via TikTok URLs. The `compute_temporal_windows` function produces a nested schema structure that doesn't match the expected flat structure, causing all videos to fail processing with exit code 1.

**Impact**: 100% failure rate on test_supplement videos (29/29 videos failed)

---

## Bug Description

### The Error

**Error 1 - Schema Validation Failure**:
```python
ValueError: compute_temporal_windows missing required keys: ['hook', 'middle_segments', 'closing'].
Got keys: ['video_id', 'duration', 'temporal_windows', 'metadata', 'processing_timestamp', 'version']
```

**Error 2 - UnboundLocalError (Consequence of Error 1)**:
```python
UnboundLocalError: cannot access local variable 'temporal_path' where it is not associated with a value
```

**Location**: `scripts/rumiai_runner.py` line 299 and line 344

---

## Root Cause Analysis

### Schema Mismatch

**Expected Schema** (what validation checks for):
```python
{
  "hook": {
    "start_time": 0.0,
    "end_time": 3.0,
    "features": {...}
  },
  "middle_segments": [...],
  "closing": {
    "start_time": ...,
    "end_time": ...,
    "features": {...}
  }
}
```

**Actual Schema** (what temporal_compute produces):
```python
{
  "video_id": "7533600925676539158",
  "duration": 67,
  "temporal_windows": {          # ← Data is nested here!
    "hook": {...},
    "middle_segments": [...],
    "closing": {...}
  },
  "metadata": {...},
  "processing_timestamp": "2025-10-21T11:48:35.123Z",
  "version": "2.0"
}
```

**The Issue**: The validation in `rumiai_runner.py` expects `hook`, `middle_segments`, and `closing` at the **top level**, but `temporal_compute` returns them **nested inside `temporal_windows`**.

---

## Reproduction Steps

### Minimal Reproduction

```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
python scripts/rumiai_runner.py "https://www.tiktok.com/@drinkpoppi/video/7533600925676539158"
```

**Expected**: Video processes successfully, generates temporal_windows_updated.json
**Actual**: Fails with schema validation error (exit code 1)

### Full Context Reproduction

```bash
# Run full ML pipeline test
python rumiai_ml_batch.py \
  --client test_production \
  --analysis-type hashtag \
  --target test_supplement \
  --video-count 15 \
  --auto-confirm
```

**Result**: All videos fail at Stage 2 processing with same error

---

## Technical Details

### Error Stack Trace

```python
File "/home/jorge/rumiaifinal/scripts/rumiai_runner.py", line 299, in process_video_url
    raise ValueError(
ValueError: compute_temporal_windows missing required keys: ['hook', 'middle_segments', 'closing'].
Got keys: ['video_id', 'duration', 'temporal_windows', 'metadata', 'processing_timestamp', 'version']

# Then, because temporal_path is never set due to the error:
File "/home/jorge/rumiaifinal/scripts/rumiai_runner.py", line 344, in process_video_url
    'temporal': str(temporal_path),
                    ^^^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'temporal_path' where it is not associated with a value
```

### Code Analysis

**Location 1**: `scripts/rumiai_runner.py` (circa line 295-305)

The validation code expects:
```python
# Validation check (pseudo-code)
required_keys = ['hook', 'middle_segments', 'closing']
if not all(key in temporal_windows for key in required_keys):
    raise ValueError(f"compute_temporal_windows missing required keys: {required_keys}")
```

**Location 2**: `rumiai_v2/processors/temporal_compute.py` (needs investigation)

The temporal_compute module returns:
```python
{
  "video_id": video_id,
  "duration": duration,
  "temporal_windows": {
    "hook": {...},
    "middle_segments": [...],
    "closing": {...}
  },
  "metadata": {...},
  ...
}
```

**Mismatch**: One expects flat structure, the other produces nested structure.

---

## Additional Context

### Related Observations

1. **Empty ML Results**: Test videos produced:
   ```
   Timeline built with 0 entries
   Entry types: {}
   Whisper: Added 0 segments
   OCR: Added 0 text overlays
   MediaPipe: Added 0 poses, 0 faces
   Scene Detection: Added 0 scene changes
   Emotion detection: Skipped
   ```

   This may indicate:
   - Videos are silent (no speech)
   - No text on screen
   - No faces/people visible
   - Low visual variance

2. **Processing Otherwise Succeeded**:
   - ✅ Video successfully scraped from TikTok
   - ✅ Video downloaded via Apify
   - ✅ ML services ran (YOLO, Whisper, OCR, MediaPipe, etc.)
   - ✅ Timeline built (albeit empty)
   - ❌ Temporal windows validation failed

3. **Not Related to Stage 2 Fix**:
   - This issue occurs AFTER videos are successfully passed to RumiAI
   - Stage 2 fix (subtitleLinks removal) is working correctly
   - Videos reach RumiAI via webVideoUrl as intended

---

## Impact Assessment

### Affected Components

- ✅ **Stage 1 (Video Discovery)**: Working correctly
- ✅ **Stage 2 (Download/URL handling)**: Working correctly (our fix validated)
- ❌ **RumiAI Pipeline**: BLOCKED - Schema validation failure
- ❌ **Stage 2 (Video Processing)**: BLOCKED - Cannot complete due to RumiAI failure
- ❌ **Stages 3-5**: BLOCKED - Depend on Stage 2 completion

### Test Results

**test_supplement (2025-10-21)**:
- Videos selected: 29
- Videos processed: 0 ✅ (0%)
- Videos failed: 29 ❌ (100%)
- Failure reason: RumiAI schema validation error

**Previous test_final (2025-10-14)**:
- Videos processed: 111 ✅
- Note: This test used drinkpoppi data, may have had different schema or code version

---

## Proposed Solutions

### Option 1: Fix rumiai_runner.py Validation (Quick Fix)

**Change**: Update schema validation to handle nested structure

**File**: `scripts/rumiai_runner.py` (circa line 295-305)

**Before**:
```python
required_keys = ['hook', 'middle_segments', 'closing']
if not all(key in temporal_windows for key in required_keys):
    raise ValueError(...)
```

**After**:
```python
# Handle nested temporal_windows structure
if 'temporal_windows' in temporal_windows:
    # New schema: data is nested
    actual_windows = temporal_windows['temporal_windows']
else:
    # Old schema: data is flat
    actual_windows = temporal_windows

required_keys = ['hook', 'middle_segments', 'closing']
if not all(key in actual_windows for key in required_keys):
    raise ValueError(...)

# Continue processing with actual_windows instead of temporal_windows
```

**Pros**:
- Quick fix, backward compatible
- Handles both old and new schema formats

**Cons**:
- Band-aid solution, doesn't address root cause
- Unclear which schema is "correct"

---

### Option 2: Fix temporal_compute Output (Proper Fix)

**Change**: Update `temporal_compute` to return flat structure

**File**: `rumiai_v2/processors/temporal_compute.py` (needs code review)

**Current Output**:
```python
return {
  "video_id": video_id,
  "duration": duration,
  "temporal_windows": {
    "hook": {...},
    "middle_segments": [...],
    "closing": {...}
  },
  ...
}
```

**Proposed Output**:
```python
return {
  "hook": {...},
  "middle_segments": [...],
  "closing": {...},
  "metadata": {
    "video_id": video_id,
    "duration": duration,
    ...
  }
}
```

**Pros**:
- Matches expected schema
- Cleaner separation of data vs metadata

**Cons**:
- May break other code expecting nested structure
- Requires thorough testing

---

### Option 3: Standardize on New Schema (Architectural Fix)

**Change**: Update ALL code to use new nested schema format

**Files to Update**:
1. `scripts/rumiai_runner.py` - Accept nested structure
2. `ml_pipeline/stage2_processing/video_processor.py` - Update validation
3. Documentation - Update schema specs

**Pros**:
- Clear separation of data vs metadata
- More extensible for future changes
- Standardized across codebase

**Cons**:
- Larger scope, more testing required
- Need to update documentation

---

## Recommended Solution

**Short-term**: **Option 1** (Quick Fix)
- Implement backward-compatible validation
- Unblocks testing immediately
- Minimal risk

**Long-term**: **Option 3** (Architectural Fix)
- Standardize on nested schema
- Update all validation code
- Document canonical schema in TI

---

## Testing Plan

### Unit Tests

```python
def test_temporal_windows_nested_schema():
    """Test that validation handles nested temporal_windows schema"""
    nested_schema = {
        "video_id": "123",
        "temporal_windows": {
            "hook": {...},
            "middle_segments": [...],
            "closing": {...}
        }
    }
    # Should pass validation
    validate_temporal_windows(nested_schema)

def test_temporal_windows_flat_schema():
    """Test that validation handles flat schema (backward compat)"""
    flat_schema = {
        "hook": {...},
        "middle_segments": [...],
        "closing": {...}
    }
    # Should also pass validation
    validate_temporal_windows(flat_schema)
```

### Integration Tests

1. **Test with actual TikTok URLs**:
   ```bash
   python scripts/rumiai_runner.py "https://www.tiktok.com/@user/video/123"
   ```

2. **Test with drinkpoppi data** (known working):
   ```bash
   # Use existing successful test data to ensure no regressions
   python rumiai_ml_batch.py --client test_competitor --analysis-type competitor --target drinkpoppi
   ```

3. **Test with test_supplement** (currently failing):
   ```bash
   # Should pass after fix
   python rumiai_ml_batch.py --client test_production --analysis-type hashtag --target test_supplement
   ```

---

## Timeline Estimate

**Option 1 (Quick Fix)**:
- Investigation: 30 minutes (✅ DONE)
- Implementation: 1 hour
- Testing: 1 hour
- Total: **~2 hours**

**Option 3 (Architectural Fix)**:
- Schema design: 1 hour
- Implementation: 3-4 hours
- Testing: 2-3 hours
- Documentation: 1-2 hours
- Total: **~7-10 hours**

---

## Workaround (None Available)

Currently, there is **no workaround** for this issue. Videos cannot be processed through RumiAI until the schema mismatch is resolved.

**Alternative**: Use existing drinkpoppi data which was processed before this schema change occurred.

---

## Related Issues

- **Stage2Fix.md**: Our Stage 2 fix (subtitleLinks removal) is WORKING. This is a separate issue.
- **VideoProcessingTI.md**: May need schema updates to document new temporal_windows structure

---

## Test Evidence

### Logs

**Main log**: `test_supplement_20251021_111346.log`

**Sample Error Entry**:
```
2025-10-21 11:30:45,159 - ml_pipeline.stage2_processing.video_processor - ERROR -
Failed to process video 7533600925676539158:
RumiAI processing failed for 7533600925676539158 at stage rumiai_pipeline:
RumiAI pipeline failed (exit code 1).
Stderr: 2025-10-21 11:30:00 - rumiai_v2.core.ml_dependency_validator - INFO - ✅ All ML dependencies validated successfully
```

### Direct RumiAI Test

```bash
$ python scripts/rumiai_runner.py "https://www.tiktok.com/@drinkpoppi/video/7533600925676539158"

2025-10-21 11:48:35 - rumiai_v2 - ERROR - Temporal windows computation failed:
compute_temporal_windows missing required keys: ['hook', 'middle_segments', 'closing'].
Got keys: ['video_id', 'duration', 'temporal_windows', 'metadata', 'processing_timestamp', 'version']

ValueError: compute_temporal_windows missing required keys...
```

### Affected Videos (Sample)

All test_supplement videos failed with same error:
- 7533600925676539158
- 7530069183997005078
- 7504815285950532894
- 7555957488936439047
- 7552956451547778326
- ... (29 total)

---

## Next Steps

1. **Immediate**: Document this bug (✅ DONE - Bug2.md)
2. **Short-term**: Implement Option 1 (quick fix) to unblock testing
3. **Long-term**: Plan Option 3 (architectural fix) for proper schema standardization
4. **Validation**: Commit Stage 2 fix (it's working correctly, separate from this issue)

---

## References

- **Discovery Session**: 2025-10-21 session with Claude Code
- **Test Run**: test_supplement_20251021_111346.log
- **Related Docs**:
  - Stage2Fix.md (our fix is working!)
  - VideoProcessingTI.md
  - VideoProcessingCHILD.md

---

## Document Metadata

**Created**: 2025-10-21
**Author**: Jorge (with Claude Code assistance)
**Status**: ACTIVE BUG
**Priority**: HIGH (blocking all video processing)
**Component**: RumiAI Pipeline
**Affects**: Stage 2 video processing
**Fix Target**: Within 1-2 days
