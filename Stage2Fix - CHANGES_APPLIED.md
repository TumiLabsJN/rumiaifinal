# Stage 2 Bug Fix - Changes Applied

**Date**: 2025-10-21
**Branch**: (create: `fix/stage2-subtitlelinks-bug`)
**Issue**: Stage 2 video processing failing due to subtitleLinks downloading caption files instead of videos

---

## Changes Applied

### Change 1: ml_pipeline/stage2_processing/video_download.py

**Lines Modified**: 44-51, 61

**What Changed**:
- ✅ Removed `subtitleLinks` fallback logic (lines 44-51)
- ✅ Updated error message to remove "subtitleLinks" from checked fields (line 61)
- ✅ Renumbered options: Option 2→1 (downloadAddr), Option 3→2 (mediaUrls)

**Before**:
```python
# Option 1: New API format (subtitleLinks contains video MP4) - try first
if 'videoMeta' in video_metadata and 'subtitleLinks' in video_metadata.get('videoMeta', {}):
    subtitle_links = video_metadata['videoMeta']['subtitleLinks']
    if subtitle_links and len(subtitle_links) > 0:
        download_url = subtitle_links[0].get('downloadLink') or subtitle_links[0].get('tiktokLink')

# Option 2: Old API format (videoMeta.downloadAddr)
# Option 3: mediaUrls array
```

**After**:
```python
# Option 1: Old API format (videoMeta.downloadAddr) - backwards compatibility
if 'videoMeta' in video_metadata and 'downloadAddr' in video_metadata.get('videoMeta', {}):
    download_url = video_metadata['videoMeta']['downloadAddr']

# Option 2: mediaUrls array (if populated)
if not download_url and 'mediaUrls' in video_metadata and video_metadata.get('mediaUrls'):
    download_url = video_metadata['mediaUrls'][0]
```

---

### Change 2: ml_pipeline/stage2_processing/main.py

**Lines Modified**: 90-92

**What Changed**:
- ✅ Removed `subtitleLinks` check from pre-download validation
- ✅ Updated to only check `downloadAddr` and `mediaUrls`

**Before**:
```python
# Check new API format (subtitleLinks)
if video_meta and 'subtitleLinks' in video_meta and video_meta.get('subtitleLinks'):
    has_download_url = True
# Check old API format (downloadAddr)
elif video_meta and 'downloadAddr' in video_meta:
    has_download_url = True
```

**After**:
```python
# Check old API format (downloadAddr)
if video_meta and 'downloadAddr' in video_meta:
    has_download_url = True
# Check mediaUrls
elif 'mediaUrls' in video and video.get('mediaUrls'):
    has_download_url = True
```

---

### Change 3: ml_pipeline/stage2_processing/pause_handler.py

**Lines Modified**: 9 (new import), 84-85 (removed imports), 101-161 (replaced logic)

**What Changed**:
- ✅ Added `import os` for path checking
- ✅ Removed unused imports (process_videos_sequential, download_video)
- ✅ Replaced download-only logic with hybrid approach (local file OR webVideoUrl)
- ✅ Added full RumiAI processing pipeline (matching video_processor.py)

**Before**:
```python
try:
    # Download video
    video_path = download_video(video, videos_dir)
except Exception as e:
    logger.error(f"Failed to download video {video_id}: {e}")
    handle_video_processing_error(e, video_id, checkpoint, checkpoint_path)
    continue
```

**After**:
```python
# Hybrid approach: Use local file if exists, otherwise use TikTok URL
if os.path.exists(local_video_path):
    video_path = local_video_path
    logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id} (local file)")
elif 'webVideoUrl' in video:
    video_path = video['webVideoUrl']
    logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id} (TikTok URL)")
else:
    logger.error(f"Video {video_id} not found locally and no webVideoUrl available")
    handle_video_processing_error(...)
    continue

# Process video through RumiAI pipeline
try:
    result = run_rumiai_pipeline(video_path=video_path, video_id=video_id, ...)
    # Validate output exists
    insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
    # Validate schema
    validate_temporal_windows_schema(insights)
    # Mark as completed
    checkpoint['completed'] += 1
    # ... (full processing logic)
except Exception as e:
    handle_video_processing_error(e, video_id, checkpoint, checkpoint_path)
    continue
```

---

## Validation

**Syntax Check**: ✅ All files passed `python3 -m py_compile`
- ✅ video_download.py
- ✅ main.py  
- ✅ pause_handler.py

**Backups Created**:
- ✅ video_download.py.backup (4.9K)
- ✅ main.py.backup (7.4K)
- ✅ pause_handler.py.backup (4.2K)

---

## Expected Behavior After Fix

1. **Pre-download Phase** (main.py lines 80-114):
   - Will NOT attempt to download from subtitleLinks
   - Only tries downloadAddr or mediaUrls (both likely null/empty)
   - Most videos will skip pre-download
   - Logs: "No videos pre-downloaded. Will use webVideoUrl for all videos during processing."

2. **Processing Phase** (pause_handler.py lines 101-161):
   - Checks for local video file first (likely won't exist)
   - Falls back to webVideoUrl (always present in API responses)
   - Passes TikTok URL to rumiai_runner.py
   - RumiAI scrapes video via ApifyClient internally
   - Video processed successfully

3. **Success Indicators**:
   - ✅ No "Downloaded file too small" errors
   - ✅ No "subtitleLinks" references in logs
   - ✅ Logs show "Processing video X/Y: VIDEO_ID (TikTok URL)"
   - ✅ All temporal_windows_updated.json files created
   - ✅ Checkpoint shows 0 failures

---

## Next Steps

### Immediate Testing
```bash
# Clean previous test data
rm -rf data/clients/test_production/hashtags/test_supplement/

# Run test
python rumiai_ml_batch.py \
  --client test_production \
  --analysis-type hashtag \
  --target test_supplement \
  --analysis-mode top \
  --selection-strategy contrastive \
  --video-count 15 \
  --date-filter last_180_days \
  --auto-confirm
```

### Expected Test Results
- Stage 1: ✅ ~30 videos selected across 3 buckets
- Stage 2: ✅ All videos processed via webVideoUrl → rumiai_runner.py → Apify
- Stage 3-5: ✅ Complete successfully
- Final checkpoint: `"completed": 30, "failed": 0`

### If Test Passes
```bash
# Commit changes
git add ml_pipeline/stage2_processing/video_download.py
git add ml_pipeline/stage2_processing/main.py
git add ml_pipeline/stage2_processing/pause_handler.py
git commit -m "fix(stage2): Remove subtitleLinks logic, add hybrid approach to pause_handler

- Remove broken subtitleLinks fallback (downloads caption files, not videos)
- Add hybrid approach to pause_handler.py (local file OR webVideoUrl)
- Match video_processor.py logic for consistency
- Fixes test_supplement failures (0/30 videos → 30/30 videos)

Resolves: Stage 2 video processing bug (subtitleLinks caption file downloads)
See: Stage2Fix.md for comprehensive analysis"
```

---

## Architectural Notes

**What This Fix Does**:
- ✅ Restores separation of concerns (download_video only handles direct downloads)
- ✅ Makes pause_handler consistent with video_processor (both use hybrid approach)
- ✅ Aligns implementation with TI intent (pause_handler should use "same logic" as video_processor)

**What This Fix Doesn't Do** (future work):
- ⚠️ Doesn't eliminate code duplication (pause_handler vs video_processor)
- ⚠️ Doesn't update TI documentation (downloadAddr still marked as "Required")
- ⚠️ Doesn't document hybrid approach in TI spec

**Follow-up Tasks** (See Stage2Fix.md):
- Priority 2: Refactor to create shared `_process_single_video()` helper
- Priority 3: Update VideoProcessingTI.md and VideoProcessingCHILD.md
- Priority 4: Optional enhancements (parallel downloads, retry-failed, etc.)

---

## References

- **Bug Analysis**: Stage2Fix.md
- **TI Specification**: VideoProcessingTI.md (lines 507-579, 662-727)
- **CHILD Specification**: VideoProcessingCHILD.md (Section 2.3.2, 2.3.3, 2.3.4)
- **Discovery Log**: Stage2Fix.md "Discovery & Validation" section
