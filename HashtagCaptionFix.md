# Hashtag & Caption Data Fix - Stage 2.6.5

## Problem Discovered

**Date**: 2025-10-31
**Context**: Implementing Stage 8 (Report Extraction) - `extract_client_data.py`

### Issue 1: Missing Caption Analysis Data
Stage 2.7 output (`*_content.json`) shows:
```json
"caption_analysis": {
  "hook_type": "statement",
  "cta_type": "none",
  "hashtag_count": 0  // Always 0
}
```

Expected fields missing:
- `caption_length` - Not present
- `emoji_usage` - Not present
- `hashtag_count` - Always 0 (incorrect)

### Issue 2: Root Cause - Missing Input Files
Stage 2.7 expects three input directories:
```
/speech_transcriptions/{video_id}_whisper.json  ✅ EXISTS
/video_captions/{video_id}_caption.json         ❌ MISSING
/video_hashtags/{video_id}_hashtags.json        ❌ MISSING
```

Code at `ml_pipeline/stage2_content_analysis/classification.py:495-521`:
```python
caption_path = f"{RUMIAI_ROOT}/video_captions/{video_id}_caption.json"
hashtags_path = f"{RUMIAI_ROOT}/video_hashtags/{video_id}_hashtags.json"

try:
    caption_data = load_json(caption_path)
    caption = caption_data.get('text', '')
except FileNotFoundError:
    caption = ''  # Falls back to empty string

try:
    hashtags_data = load_json(hashtags_path)
    hashtags = hashtags_data.get('hashtags', [])
except FileNotFoundError:
    hashtags = []  # Falls back to empty array
```

Result: Stage 2.7 LLM receives empty captions and empty hashtags for all videos.

### Issue 3: Source Data Exists But Not Extracted
Caption and hashtag data exists in `selected_videos.json`:
```json
{
  "videos": [
    {
      "id": "7528856221252635926",
      "text": "easy peanut butter bites #healthy #peanutbutter...",
      "hashtags": [
        {"name": "healthy"},
        {"name": "peanutbutter"},
        {"name": "healthysnacks"}
      ]
    }
  ]
}
```

But never extracted into individual per-video files.

---

## Solution: Add Stage 2.6.5 - Caption/Hashtag Extraction

### Architecture Decision
**Location**: Stage 2 (Content Analysis)
**Timing**: After Stage 2.6 (Discovery), Before Stage 2.7 (Classification)

```
Stage 2 Flow:
├── Stage 2.1-2.5: ML feature extraction
├── Stage 2.6: Pattern discovery (LLM)
├── Stage 2.6.5: Extract captions/hashtags (NEW)
└── Stage 2.7: Classification (LLM)
```

### Implementation

**File**: `/ml_pipeline/stage2_content_analysis/prepare_inputs.py`

**Function Signature**:
```python
def extract_captions_and_hashtags(
    client_id: str,
    target: str,
    analysis_type: str,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive"
) -> Dict[str, int]:
    """
    Extract captions and hashtags from selected_videos.json into target-scoped files.

    Uses PathBuilder to construct paths dynamically based on analysis_type.
    Works for hashtag, competitor, and creator analysis.

    Args:
        client_id: Client identifier (e.g., "acme_corp")
        target: Target with prefix (#nutrition, @brand, @creator)
        analysis_type: "hashtag", "competitor", or "creator"
        analysis_mode: "top" or "recent" (default: "top")
        selection_strategy: "contrastive" or "top" (default: "contrastive")

    Returns:
        dict: Summary with keys:
            - extracted_count: Number of videos extracted
            - skipped_count: Number of videos skipped (missing data)

    Creates:
        {target_dir}/video_captions/{video_id}_caption.json
        {target_dir}/video_hashtags/{video_id}_hashtags.json

    Source: HashtagCaptionFix.md - Stage 2.6.5 implementation
    """
```

**Input Source** (using PathBuilder):
```python
from foundation.paths import PathBuilder

path_builder = PathBuilder()
target_dir = path_builder.get_target_dir(
    client_id=client_id,
    analysis_type=analysis_type,  # Dynamic: hashtag/competitor/creator
    target=target,
    analysis_mode=analysis_mode,
    selection_strategy=selection_strategy
)

# Example paths:
# Hashtag: /data/clients/acme/hashtags/nutrition/top_contrastive/
# Competitor: /data/clients/acme/competitors/gnclivewell/top_contrastive/
# Creator: /data/clients/acme/creators/wellnesscreator/top_contrastive/

# Input: {target_dir}/buckets/bucket_{name}/selected_videos.json
# Output: {target_dir}/video_captions/{video_id}_caption.json
#         {target_dir}/video_hashtags/{video_id}_hashtags.json
```

**Output Files** (target-scoped):
```json
// {target_dir}/video_captions/7528856221252635926_caption.json
{
  "video_id": "7528856221252635926",
  "text": "easy peanut butter bites #healthy #peanutbutter #healthysnacks...",
  "text_language": "en"
}

// {target_dir}/video_hashtags/7528856221252635926_hashtags.json
{
  "video_id": "7528856221252635926",
  "hashtags": [
    {"name": "healthy"},
    {"name": "peanutbutter"},
    {"name": "healthysnacks"}
  ]
}
```

**Directory Structure**:
```
/data/clients/{client}/{analysis_type}s/{target}/{mode}_{strategy}/
├── winner_analysis.json (input)
├── video_captions/ (NEW - output)
│   ├── 7528856221252635926_caption.json
│   └── 7428394729374927394_caption.json
├── video_hashtags/ (NEW - output)
│   ├── 7528856221252635926_hashtags.json
│   └── 7428394729374927394_hashtags.json
└── buckets/
    └── bucket_18-33s/
        └── selected_videos.json (input)
```

**Logic**:
1. Use PathBuilder to get target_dir (supports hashtag/competitor/creator)
2. Load `{target_dir}/winner_analysis.json` → get winning bucket names
3. Create output directories: `{target_dir}/video_captions/`, `{target_dir}/video_hashtags/`
4. For each winning bucket:
   - Load `{target_dir}/buckets/bucket_{name}/selected_videos.json`
   - For each video in `videos` array:
     - Extract `text` field → save to `{target_dir}/video_captions/{video_id}_caption.json`
     - Extract `hashtags` array → save to `{target_dir}/video_hashtags/{video_id}_hashtags.json`
5. Handle missing/empty fields gracefully:
   - Missing `text` → save `{"video_id": "...", "text": "", "text_language": "unknown"}`
   - Missing `hashtags` → save `{"video_id": "...", "hashtags": []}`
   - Missing video in selected_videos.json → log warning, skip
6. Return summary: `{"extracted_count": 285, "skipped_count": 15}`

### Integration Points

#### **Integration Point 1: Main Batch Runner**

**File**: `rumiai_ml_batch.py`

Add after Stage 2.6 Discovery, before Stage 2.7 Classification (around line 940):
```python
# ===== STAGE 2.6.5: EXTRACT CAPTIONS & HASHTAGS =====
logger.info("Starting Stage 2.6.5: Caption/Hashtag Extraction")
print("\n" + "="*80)
print("STAGE 2.6.5: CAPTION/HASHTAG EXTRACTION")
print("="*80)

from ml_pipeline.stage2_content_analysis.prepare_inputs import extract_captions_and_hashtags

try:
    summary = extract_captions_and_hashtags(
        client_id=sanitize_client_id(cli_args.client),
        target=cli_args.target,
        analysis_type=cli_args.analysis_type,
        analysis_mode=cli_args.analysis_mode,
        selection_strategy=cli_args.selection_strategy
    )

    logger.info(f"Stage 2.6.5 complete: {summary['extracted_count']} videos extracted")
    print(f"\n✓ Stage 2.6.5: Caption/Hashtag Extraction - COMPLETE")
    print(f"  Extracted: {summary['extracted_count']} videos")
    if summary['skipped_count'] > 0:
        print(f"  ⚠️  Skipped: {summary['skipped_count']} videos (missing data)")

except FileNotFoundError as e:
    logger.error(f"Stage 2.6.5 failed: {e}")
    print(f"\n✗ Stage 2.6.5 failed: {e}")
    return 1

except Exception as e:
    logger.error(f"Stage 2.6.5 failed: {e}", exc_info=True)
    print(f"\n✗ Stage 2.6.5 failed: {e}")
    return 1

# === STAGE 2.7: VIDEO CLASSIFICATION ===
# (continues with existing Stage 2.7 code...)
```

#### **Integration Point 2: Standalone Classification Script**

**File**: `run_stage_2_7.py`

Add before `run_classification_stage()` call (around line 195):
```python
# Step 1.5: Extract captions and hashtags (NEW)
print("\n" + "=" * 80)
print("STEP 1.5: EXTRACTING CAPTIONS & HASHTAGS")
print("=" * 80)

from stage2_content_analysis.prepare_inputs import extract_captions_and_hashtags

try:
    summary = extract_captions_and_hashtags(
        client_id=args.client,
        target=args.hashtag,
        analysis_type=args.analysis_type,
        analysis_mode=args.analysis_mode,
        selection_strategy=args.selection_strategy
    )

    print(f"\n✅ Extraction complete: {summary['extracted_count']} videos")
    if summary['skipped_count'] > 0:
        print(f"   ⚠️  Skipped {summary['skipped_count']} videos (missing data)")

except Exception as e:
    print(f"❌ ERROR: Failed to extract captions/hashtags: {e}")
    sys.exit(1)

# Step 2: Run classification (will now have caption/hashtag data)
# (continues with existing run_classification_stage() call...)
```

#### **Integration Point 3: Classification Module Path Reading**

**File**: `ml_pipeline/stage2_content_analysis/classification.py`

**Current code** (lines 493-523) uses global RUMIAI_ROOT paths:
```python
# CURRENT (GLOBAL PATHS - WRONG):
def load_video_data(video_id: str):
    RUMIAI_ROOT = os.environ.get('RUMIAI_ROOT', '/home/jorge/rumiaifinal')
    transcript_path = f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json"
    caption_path = f"{RUMIAI_ROOT}/video_captions/{video_id}_caption.json"
    hashtags_path = f"{RUMIAI_ROOT}/video_hashtags/{video_id}_hashtags.json"
    # ...
```

**UPDATE TO** (target-scoped paths):
```python
# NEW (TARGET-SCOPED PATHS):
def load_video_data(
    video_id: str,
    target_dir: Path  # NEW PARAMETER
) -> Tuple[Dict[str, Any], str, List[Dict[str, str]]]:
    """
    Load transcript, caption, and hashtags for a video.

    Args:
        video_id: Video identifier
        target_dir: Target directory (from PathBuilder)

    Returns:
        tuple: (transcript_dict, caption_str, hashtags_list)
    """
    RUMIAI_ROOT = os.environ.get('RUMIAI_ROOT', '/home/jorge/rumiaifinal')

    # Transcript still in global location (from Stage 2)
    transcript_path = f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json"

    # Caption and hashtags now in target-scoped location (from Stage 2.6.5)
    caption_path = target_dir / "video_captions" / f"{video_id}_caption.json"
    hashtags_path = target_dir / "video_hashtags" / f"{video_id}_hashtags.json"

    # Load transcript
    try:
        transcript_data = load_json(transcript_path)
        transcript = {'text': transcript_data.get('text', ''), 'available': True}
    except FileNotFoundError:
        transcript = {'text': '', 'available': False}
        logger.warning(f"No transcript for {video_id}")

    # Load caption
    try:
        caption_data = load_json(str(caption_path))
        caption = caption_data.get('text', '')
    except FileNotFoundError:
        caption = ''
        logger.warning(f"No caption for {video_id} at {caption_path}")

    # Load hashtags
    try:
        hashtags_data = load_json(str(hashtags_path))
        hashtags = hashtags_data.get('hashtags', [])
    except FileNotFoundError:
        hashtags = []
        logger.warning(f"No hashtags for {video_id} at {hashtags_path}")

    return transcript, caption, hashtags
```

**Update callers** to pass `target_dir`:
```python
# classify_single_video_with_save() around line 561
# OLD:
transcript, caption, hashtags = load_video_data(video_id)

# NEW (needs target_dir from run_classification_stage):
transcript, caption, hashtags = load_video_data(video_id, target_dir)
```

**Update `run_classification_stage()`** to construct and pass `target_dir`:
```python
# Around line 993-1015 in run_classification_stage()
from foundation.paths import PathBuilder

path_builder = PathBuilder()
target_dir = path_builder.get_target_dir(
    client_id=client_id,
    analysis_type=analysis_type,
    target=hashtag,
    analysis_mode=analysis_mode,
    selection_strategy=selection_strategy
)

# Pass target_dir to classify_single_video_with_save() calls
# ...
```

---

## Expected Impact

### Stage 2.7 Output Will Now Include:
```json
{
  "caption_analysis": {
    "hook_type": "statement",
    "cta_type": "link_in_bio",
    "hashtag_count": 8  // Calculated correctly
  }
}
```

### Stage 8 Reports Will Show:
```
CAPTION_HOOK_TYPE: Statement
CAPTION_HOOK_TYPE_PCT: 68
TOP_CTA_1: Link In Bio
TOP_CTA_1_PCT: 32
TOP_CTA_2: Follow
TOP_CTA_2_PCT: 15
TOP_CTA_3: Comment
TOP_CTA_3_PCT: 8
NO_CTA_PCT: 45
NO_HASHTAGS_PCT: 12
OPTIMAL_HASHTAG_COUNT: 7
```

---

## Notes

### Why Not Calculate in Stage 8?
- Stage 2.7 LLM already analyzes caption content (hook_type, cta_type)
- Having LLM see actual caption text improves classification quality
- Maintains separation of concerns: Stage 2 = content analysis, Stage 8 = reporting

### Why Target-Scoped Instead of Global?
**Global directory issues**:
- ❌ Name collisions: Same video_id across multiple targets
- ❌ Not scalable: All clients/targets mixed together
- ❌ Cleanup problems: Can't delete data for one target

**Target-scoped benefits**:
- ✅ No collisions: Each target has its own directory
- ✅ Scalable: Clean separation by client/target
- ✅ Easy cleanup: Delete target_dir removes all data
- ✅ Consistent with PathBuilder architecture

### Error Handling
**Missing files**:
- `winner_analysis.json` missing → Raise FileNotFoundError (Stage 2.5 incomplete)
- `selected_videos.json` missing for bucket → Log warning, skip bucket
- Video in manifest but not in selected_videos.json → Log warning, skip video

**Missing/null fields**:
- `text` field null/missing → Save empty caption: `{"video_id": "...", "text": "", "text_language": "unknown"}`
- `hashtags` field null/missing → Save empty array: `{"video_id": "...", "hashtags": []}`
- `text_language` field missing → Default to "unknown"

**Exceptions**:
- JSON decode errors → Log error, skip video, continue
- File write errors → Raise immediately (disk full, permissions)

### Idempotency
- Extraction is idempotent (can run multiple times safely)
- Overwrites existing files if they exist
- No checkpointing needed (fast operation)
- Re-running after Stage 2.1 adds new videos will update all files

### Performance
- ~300 videos × 2 files = 600 small JSON writes
- Estimated time: < 1 second
- Negligible overhead before Stage 2.7 LLM calls
- Target-scoped paths avoid global directory bottlenecks

---

## Implementation Summary

### Files to Create

**1. `/ml_pipeline/stage2_content_analysis/prepare_inputs.py`** (NEW)
- Main function: `extract_captions_and_hashtags()`
- Uses PathBuilder for path construction
- Extracts from `selected_videos.json` → individual caption/hashtag files
- Returns summary dict: `{"extracted_count": int, "skipped_count": int}`

### Files to Modify

**2. `/ml_pipeline/stage2_content_analysis/classification.py`**
- Update `load_video_data()` signature: Add `target_dir: Path` parameter
- Update caption/hashtag path reading to use target-scoped paths
- Update `classify_single_video_with_save()` to pass `target_dir`
- Update `run_classification_stage()` to construct `target_dir` using PathBuilder

**3. `rumiai_ml_batch.py`**
- Add Stage 2.6.5 section after Stage 2.6, before Stage 2.7 (line ~940)
- Import and call `extract_captions_and_hashtags()`
- Add error handling (FileNotFoundError, general exceptions)

**4. `run_stage_2_7.py`**
- Add Step 1.5 section before `run_classification_stage()` call (line ~195)
- Import and call `extract_captions_and_hashtags()`
- Add error handling

### Architecture Alignment

**Consistent with PathBuilder migration**:
- ✅ All paths constructed using PathBuilder
- ✅ No hardcoded "hashtags" directory
- ✅ Supports hashtag, competitor, and creator analysis types
- ✅ Target-scoped directories (no global flat directories)

**Fixes the same bug we corrected in `2.5PathbuilderUpdate.md`**:
- ✅ Includes `analysis_type` parameter
- ✅ Uses dynamic path resolution
- ✅ No construct_path() legacy code

### Testing Checklist

After implementation, verify:

1. **Hashtag analysis** - Caption/hashtag extraction works
   ```bash
   python rumiai_ml_batch.py --client test --target "#nutrition" --analysis-type hashtag
   ```

2. **Competitor analysis** - Paths resolve correctly
   ```bash
   python rumiai_ml_batch.py --client test --target "@brand" --analysis-type competitor
   ```

3. **Files created** in correct location:
   ```
   /data/clients/test/hashtags/nutrition/top_contrastive/video_captions/
   /data/clients/test/competitors/brand/top_contrastive/video_captions/
   ```

4. **Stage 2.7 classification** receives caption/hashtag data:
   - Check classification output has populated `caption_analysis` fields
   - Verify `hashtag_count` is calculated correctly (M10 fix)

5. **Stage 8 reports** show caption statistics:
   - CAPTION_HOOK_TYPE_PCT shows distribution
   - TOP_CTA metrics populated
   - OPTIMAL_HASHTAG_COUNT calculated from real data
