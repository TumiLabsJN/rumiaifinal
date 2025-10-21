# Stage 2 Video Download Bug Fix

**Date**: 2025-10-21
**Status**: Bug Identified - Fix Documented
**Affected Component**: `ml_pipeline/stage2_processing/video_download.py`
**Git Commit with Bug**: e73bd4a (Oct 15, 2025)

---

## Executive Summary

Stage 2 video processing is failing because `video_download.py` downloads subtitle caption files (2-6KB) instead of video MP4 files, causing all videos to fail the minimum file size validation (100KB).

**Root Cause**: Incorrect fallback logic added to handle Apify API changes. The code tries to download from `subtitleLinks` which contains multilingual caption files, not video files. Additionally, `pause_handler.py` lacks the hybrid fallback logic that exists in `video_processor.py`, causing it to mark videos as failed instead of trying the `webVideoUrl` alternative.

**Solution**:
1. Remove the broken `subtitleLinks` logic from `video_download.py` and `main.py`
2. Update `pause_handler.py` to use the same hybrid approach as `video_processor.py` (check for local file, fallback to webVideoUrl)

**Affected Files**: 3 files need changes
- `ml_pipeline/stage2_processing/video_download.py`
- `ml_pipeline/stage2_processing/main.py`
- `ml_pipeline/stage2_processing/pause_handler.py`

---

## Timeline of Events

| Date | Event | Impact |
|------|-------|--------|
| **Oct 14, 2025** | test_final test runs successfully | 111 videos processed, all temporal_windows files created |
| **Oct 15, 2025** | Commit e73bd4a adds `subtitleLinks` fallback | Intended to handle Apify API change (downloadAddr missing) |
| **Oct 20, 2025** | test_supplement test fails completely | 0/30 videos processed, all fail with "file too small" error |
| **Oct 21, 2025** | Root cause identified | subtitleLinks contains captions, not videos |

---

## The Bug

### What Went Wrong

**File**: `ml_pipeline/stage2_processing/video_download.py`
**Lines**: 44-51

```python
# WRONG: Tries subtitleLinks first!
if 'videoMeta' in video_metadata and 'subtitleLinks' in video_metadata.get('videoMeta', {}):
    subtitle_links = video_metadata['videoMeta']['subtitleLinks']
    if subtitle_links and len(subtitle_links) > 0:
        download_url = subtitle_links[0].get('downloadLink') or subtitle_links[0].get('tiktokLink')
        if download_url:
            logger.debug(f"Video {video_id}: Using subtitleLinks (new API format)")
```

### Why It's Wrong

1. **`subtitleLinks` contains subtitle/caption files, NOT video files**
   - These are multilingual caption tracks (ceb-PH, ukr-UA, kaz-KZ, etc.)
   - File sizes: 2-6KB (captions) vs 5-20MB (videos)
   - MIME type in URL parameter says `video_mp4` but actual file is caption data

2. **Example from actual API response**:
   ```json
   "subtitleLinks": [{
     "LanguageID": "ceb-PH",
     "LanguageCodeName": "ceb",
     "downloadLink": "https://.../?mime_type=video_mp4",  // MISLEADING!
     "tiktokLink": "https://.../?mime_type=video_mp4"      // MISLEADING!
   }]
   ```

3. **Actual file size verification**:
   ```bash
   $ curl -sI "https://v16m-webapp.tiktokcdn-us.com/.../mime_type=video_mp4" | grep content-length
   content-length: 1879  # Only 1.8KB - this is a subtitle file!
   ```

### Current Test Results

**test_supplement (Oct 20, 2025) - ALL FAILED**:
- Stage 1: ✅ 30 videos selected across 3 buckets
- Stage 2: ❌ 0/30 videos processed
- Error: `Downloaded file too small: 4449 bytes (minimum: 100000)`
- All downloads were subtitle files instead of videos

---

## Intended Architecture (per VideoProcessingTI.md)

### The Correct Design

Stage 2 video processing has **two separate responsibilities**:

#### Function 3: `download_video()` (TI lines 507-579)
- **Purpose**: Download video MP4 from **direct download URL** ONLY
- **Expected Input**: `video_metadata['videoMeta']['downloadAddr']`
- **Scope**: HTTP download with retry logic
- **Failure Mode**: Raise `DownloadError` if no direct URL or download fails

#### Function 4: `process_videos_sequential()` (TI lines 590-657)
- **Purpose**: Process videos through RumiAI pipeline
- **Expected Input**: Path to downloaded MP4 file
- **Assumption**: Video already downloaded by Function 3

### The Hybrid Architecture (Actual Implementation)

The actual code in `video_processor.py` implements a **hybrid approach** not in the TI:

```python
# video_processor.py:138-151
if os.path.exists(local_video_path):
    video_path = local_video_path
    logger.info(f"Processing {video_id} (local file)")
elif 'webVideoUrl' in video:
    video_path = video['webVideoUrl']
    logger.info(f"Processing {video_id} (TikTok URL)")
else:
    logger.error(f"Video {video_id} missing: no local file and no webVideoUrl")
```

**This hybrid approach is the CORRECT architectural solution!** It handles two scenarios:

1. **Scenario A**: Direct download URL available → download MP4 → process local file
2. **Scenario B**: No direct URL → use `webVideoUrl` → let `rumiai_runner.py` scrape+download

---

## How test_final Actually Worked (Oct 14)

### Evidence from test_final

**selected_videos.json structure**:
```json
{
  "id": "7528143196287733006",
  "webVideoUrl": "https://www.tiktok.com/@pamelapedrozaa/video/7528143196287733006",
  "mediaUrls": [],
  "videoMeta": {
    "height": 1024,
    "width": 576,
    "duration": 60
    // NO downloadAddr field!
  }
}
```

**Videos successfully processed**:
```bash
$ ls -lh test_final/buckets/bucket_60-90s/videos/*.mp4 | head -3
-rw-r--r-- 1 jorge jorge 6.0M Oct 14 18:16 7528143196287733006.mp4
-rw-r--r-- 1 jorge jorge 5.5M Oct 14 19:23 7532514471483280662.mp4
-rw-r--r-- 1 jorge jorge  13M Oct 14 19:06 7533062661525769486.mp4
```

**Temporal windows files created**:
```bash
$ ls test_final/buckets/bucket_60-90s/analysis/insights/*.json | wc -l
35  # All videos successfully processed!
```

### How It Worked Without downloadAddr

1. **Stage 1**: Scraped videos with `shouldDownloadVideos: False` (no pre-download)
2. **Stage 2**: `download_video()` tried to access `downloadAddr` → **raised KeyError**
3. **video_processor.py**: Caught error, used `webVideoUrl` fallback
4. **rumiai_runner.py**: Called `ApifyClient.scrape_video(webVideoUrl)` with `shouldDownloadVideos: True`
5. **Apify**: Downloaded video and returned it with metadata
6. **Result**: Videos processed successfully via URL scraping

**The existing architecture already handled missing `downloadAddr` correctly!**

---

## The Architectural Deviation

### What We Did Wrong

| Layer | Intended Responsibility | What We Did (WRONG) | Should Have Done |
|-------|------------------------|---------------------|------------------|
| `download_video()` | Download from direct URL ONLY | Added `subtitleLinks` fallback | Remove fallback, fail gracefully |
| `video_processor.py` | Orchestrate download OR use URL | Gets bypassed by broken fallback | Let it handle the error |
| `rumiai_runner.py` | Process video (file or URL) | Never gets the URL | Use it as intended |

### The Mistake

We violated **separation of concerns** by trying to make `download_video()` handle API changes. The function should:
- ✅ Download from direct URL if available
- ✅ Raise error if URL missing
- ❌ NOT try alternative APIs (that's the orchestrator's job)

The hybrid fallback in `video_processor.py` was the **correct architectural solution**. We broke it by adding logic at the wrong layer.

---

## Deep Dive: Why pause_handler.py is the Real Problem

### Discovery Process

Initially, we identified that `subtitleLinks` was downloading caption files. However, deeper investigation revealed:

1. **Two Processing Paths Exist**:
   - `video_processor.py`: Has hybrid logic (local file → webVideoUrl fallback) ✅
   - `pause_handler.py`: Missing hybrid logic, always tries to download first ❌

2. **Test Execution Path**:
   ```bash
   $ grep "pause\|Step 3-4:" test_supplement_20251021_071900.log
   Step 3-4: Processing 11 videos
   Processing video 1/11: 7551116787392220446  # pause_handler.py!
   Failed to download video 7551116787392220446: Downloaded file too small: 1879 bytes
   ```

3. **The Missing Hybrid Logic**:
   - `video_processor.py` (lines 137-150): ✅ Has hybrid approach
     ```python
     if os.path.exists(local_video_path):
         video_path = local_video_path  # Use local file
     elif 'webVideoUrl' in video:
         video_path = video['webVideoUrl']  # Fallback to URL!
     ```

   - `pause_handler.py` (lines 107-115): ❌ No hybrid logic
     ```python
     try:
         video_path = download_video(video, videos_dir)  # Always download!
     except Exception as e:
         handle_video_processing_error(e, ...)  # Just marks as FAILED!
         continue
     ```

4. **Why It Fails**:
   - `pause_handler` calls `download_video()` unconditionally
   - `download_video()` tries `subtitleLinks` → gets caption file
   - File size check fails → raises DownloadError
   - Exception caught → `handle_video_processing_error()` marks as **failed**
   - **Never tries webVideoUrl alternative!**

### The Architectural Inconsistency

`pause_handler.py` and `video_processor.py` should use the SAME logic, but they don't:

| Component | Local File Check | webVideoUrl Fallback | Behavior on Download Fail |
|-----------|------------------|---------------------|---------------------------|
| `video_processor.py` | ✅ Yes (line 138) | ✅ Yes (line 142) | Uses webVideoUrl |
| `pause_handler.py` | ❌ No | ❌ No | Marks as failed |

**This inconsistency is why the test failed.**

---

## The Complete Fix

### THREE Files Need Changes

#### 1. File: `ml_pipeline/stage2_processing/video_download.py`

#### Remove Lines 44-51 (subtitleLinks logic)

```python
# DELETE THIS ENTIRE BLOCK:
if 'videoMeta' in video_metadata and 'subtitleLinks' in video_metadata.get('videoMeta', {}):
    subtitle_links = video_metadata['videoMeta']['subtitleLinks']
    if subtitle_links and len(subtitle_links) > 0:
        download_url = subtitle_links[0].get('downloadAddr') or subtitle_links[0].get('tiktokLink')
        if download_url:
            logger.debug(f"Video {video_id}: Using subtitleLinks (new API format)")
```

#### Keep Lines 53-61 (downloadAddr and mediaUrls - backwards compatibility)

```python
# KEEP: Try downloadAddr (backwards compatibility for old API)
if 'videoMeta' in video_metadata and 'downloadAddr' in video_metadata.get('videoMeta', {}):
    download_url = video_metadata['videoMeta']['downloadAddr']
    logger.debug(f"Video {video_id}: Using downloadAddr (old API format)")

# KEEP: Try mediaUrls as fallback
if not download_url and 'mediaUrls' in video_metadata and video_metadata.get('mediaUrls'):
    download_url = video_metadata['mediaUrls'][0]
    logger.debug(f"Video {video_id}: Using mediaUrls")
```

#### Update Error Message (Lines 62-72)

```python
if not download_url:
    available_fields = list(video_metadata.get('videoMeta', {}).keys()) if 'videoMeta' in video_metadata else []
    raise DownloadError(
        video_id=video_id,
        attempts=0,
        original_error=KeyError(
            f"No download URL found for video {video_id}. "
            f"Checked: downloadAddr, mediaUrls. "
            f"Available videoMeta fields: {available_fields}"
        )
    )
```

### Why This Fix Works

1. **Removes broken logic**: No more subtitle file downloads
2. **Preserves backwards compatibility**: Still tries `downloadAddr` and `mediaUrls` if present
3. **Trusts the architecture**: Lets `video_processor.py` handle the fallback to `webVideoUrl`
4. **Follows TI spec**: Function does ONLY what it's designed to do (direct downloads)

---

## Data Flow After Fix

### Scenario 1: Old API (has downloadAddr)

```
Stage 1 Scraper (shouldDownloadVideos: False)
  ↓
selected_videos.json: {downloadAddr: "https://cdn.../video.mp4"}
  ↓
Stage 2: download_video()
  → Finds downloadAddr
  → Downloads MP4 directly (6MB)
  → Returns local path
  ↓
video_processor.py: Uses local file
  ↓
rumiai_runner.py: Processes local MP4
  ↓
✅ SUCCESS
```

### Scenario 2: New API (no downloadAddr)

```
Stage 1 Scraper (shouldDownloadVideos: False)
  ↓
selected_videos.json: {webVideoUrl: "https://tiktok.com/@user/video/123"}
  ↓
Stage 2: download_video()
  → No downloadAddr found
  → No mediaUrls found
  → Raises DownloadError ⚠️
  ↓
video_processor.py: Catches error
  → Checks for webVideoUrl ✓
  → Uses webVideoUrl instead of local path
  ↓
rumiai_runner.py: Receives TikTok URL
  → Calls ApifyClient.scrape_video(url, shouldDownloadVideos=True)
  → Apify scrapes video page
  → Apify downloads video
  → Returns video with metadata
  → Processes video
  ↓
✅ SUCCESS
```

---

## Verification Steps

### After Applying Fix

1. **Clean previous test data**:
   ```bash
   rm -rf data/clients/test_production/hashtags/test_supplement/
   ```

2. **Re-run production test**:
   ```bash
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

3. **Expected results**:
   - Stage 1: ✅ ~30 videos selected
   - Stage 2: ✅ All videos processed via webVideoUrl → rumiai_runner.py → Apify
   - Stage 3-5: ✅ Complete successfully
   - Checkpoint: `"completed": 30, "failed": 0`

4. **Validate video files**:
   ```bash
   # Should see proper video files (5-20MB each)
   ls -lh data/clients/test_production/hashtags/test_supplement/top_contrastive/buckets/*/videos/*.mp4

   # Should NOT see tiny files (< 100KB)
   find data/clients/test_production -name "*.mp4" -size -100k
   ```

5. **Validate temporal_windows files**:
   ```bash
   # Should have one per video
   find data/clients/test_production -name "*_temporal_windows_updated.json" | wc -l
   ```

---

## Code Changes

### Summary of Changes

| File | Lines Changed | Type | Description |
|------|---------------|------|-------------|
| `video_download.py` | 44-51, 68 | Delete + Update | Remove subtitleLinks logic, update error message |
| `main.py` | 90-92 | Delete | Remove subtitleLinks from pre-download check |
| `pause_handler.py` | 107-115 | Replace | Add hybrid logic (match video_processor.py) |

---

### Change 1: ml_pipeline/stage2_processing/video_download.py

**Remove lines 44-51** (subtitleLinks download logic):

```diff
@@ -41,19 +41,9 @@ def download_video(video_metadata: Dict[str, Any], output_dir: str, max_attempt
     # Get download URL - try multiple API formats (API changed Oct 2025)
     download_url = None

-    # Option 1: New API format (subtitleLinks contains video MP4) - try first
-    if 'videoMeta' in video_metadata and 'subtitleLinks' in video_metadata.get('videoMeta', {}):
-        subtitle_links = video_metadata['videoMeta']['subtitleLinks']
-        if subtitle_links and len(subtitle_links) > 0:
-            download_url = subtitle_links[0].get('downloadLink') or subtitle_links[0].get('tiktokLink')
-            if download_url:
-                logger.debug(f"Video {video_id}: Using subtitleLinks (new API format)")
-
-    # Option 2: Old API format (videoMeta.downloadAddr) - backwards compatibility
+    # Option 1: Old API format (videoMeta.downloadAddr) - backwards compatibility
     if not download_url and 'videoMeta' in video_metadata and 'downloadAddr' in video_metadata.get('videoMeta', {}):
         download_url = video_metadata['videoMeta']['downloadAddr']
         logger.debug(f"Video {video_id}: Using downloadAddr (old API format)")

-    # Option 3: mediaUrls array (if populated)
+    # Option 2: mediaUrls array (if populated)
     if not download_url and 'mediaUrls' in video_metadata and video_metadata.get('mediaUrls'):
         download_url = video_metadata['mediaUrls'][0]
@@ -66,7 +56,7 @@ def download_video(video_metadata: Dict[str, Any], output_dir: str, max_attempt
             original_error=KeyError(
                 f"No download URL found for video {video_id}. "
-                f"Checked: downloadAddr, subtitleLinks, mediaUrls. "
+                f"Checked: downloadAddr, mediaUrls. "
                 f"Available videoMeta fields: {available_fields}"
             )
         )
```

---

### Change 2: ml_pipeline/stage2_processing/main.py

**Remove lines 90-92** (subtitleLinks pre-download check):

```diff
@@ -86,12 +86,6 @@ def stage_2_video_processing_main(
         # Check if video has download URL (try multiple API formats - API changed Oct 2025)
         has_download_url = False
         video_meta = video.get('videoMeta', {})

-        # Check new API format (subtitleLinks)
-        if video_meta and 'subtitleLinks' in video_meta and video_meta.get('subtitleLinks'):
-            has_download_url = True
-        # Check old API format (downloadAddr)
-        elif video_meta and 'downloadAddr' in video_meta:
+        # Check old API format (downloadAddr)
+        if video_meta and 'downloadAddr' in video_meta:
             has_download_url = True
         # Check mediaUrls
         elif 'mediaUrls' in video and video.get('mediaUrls'):
```

---

### Change 3: ml_pipeline/stage2_processing/pause_handler.py

**Replace lines 107-115** with hybrid logic (match video_processor.py pattern):

```diff
@@ -104,14 +104,24 @@ def process_videos_with_pause_support(
         videos_dir = f"{bucket_path}videos/"

         logger.info(f"Processing video {i}/{len(remaining_videos)}: {video_id}")

-        try:
-            # Download video
-            video_path = download_video(video, videos_dir)
-        except Exception as e:
-            logger.error(f"Failed to download video {video_id}: {e}")
-            # Handle download error using the error handler
-            from ml_pipeline.stage2_processing.video_processor import handle_video_processing_error
-            handle_video_processing_error(e, video_id, checkpoint, checkpoint_path)
-            continue
+        # FIXED: Use hybrid approach (same as video_processor.py)
+        local_video_path = f"{bucket_path}videos/{video_id}.mp4"
+
+        if os.path.exists(local_video_path):
+            video_path = local_video_path
+            logger.debug(f"Using local file for video {video_id}")
+        elif 'webVideoUrl' in video:
+            video_path = video['webVideoUrl']
+            logger.debug(f"Using webVideoUrl for video {video_id}")
+        else:
+            logger.error(f"Video {video_id} not found locally and no webVideoUrl available")
+            from ml_pipeline.stage2_processing.video_processor import handle_video_processing_error
+            handle_video_processing_error(
+                ValueError(f"Video {video_id} missing: no local file and no webVideoUrl"),
+                video_id, checkpoint, checkpoint_path
+            )
+            continue
+
+        # Continue with RumiAI processing (video_path is now either local file or TikTok URL)
+        # ... rest of processing logic follows ...
```

**Note**: After this change, `pause_handler.py` will need the full processing logic from `video_processor.py` lines 152-189 (RumiAI pipeline call + validation). The current incomplete implementation just has the download part.

---

## Testing Checklist

### Pre-Implementation
- [ ] Backup current files (video_download.py, main.py, pause_handler.py)
- [ ] Kill any running test processes
- [ ] Review all three code changes carefully

### Implementation
- [ ] **Change 1**: Remove `subtitleLinks` logic from `video_download.py` (lines 44-51)
- [ ] **Change 1**: Update error message in `video_download.py` (line 68)
- [ ] **Change 1**: Update comment numbering (Option 2→1, Option 3→2)
- [ ] **Change 2**: Remove `subtitleLinks` check from `main.py` (lines 90-92)
- [ ] **Change 2**: Update `elif` to `if` in `main.py` (line 94)
- [ ] **Change 3**: Replace download logic in `pause_handler.py` with hybrid approach
- [ ] **Change 3**: Copy full RumiAI processing logic from `video_processor.py` to `pause_handler.py`

### Post-Implementation Validation
- [ ] Syntax check: `python -m py_compile ml_pipeline/stage2_processing/video_download.py`
- [ ] Syntax check: `python -m py_compile ml_pipeline/stage2_processing/main.py`
- [ ] Syntax check: `python -m py_compile ml_pipeline/stage2_processing/pause_handler.py`
- [ ] Clean test data: `rm -rf data/clients/test_production/hashtags/test_supplement/`
- [ ] Remove old logs: `rm test_supplement_*.log`

### Test Execution
- [ ] Run production test with `--auto-confirm`
- [ ] Monitor log for "Using webVideoUrl" messages (should appear!)
- [ ] Verify Stage 1 completes (videos selected)
- [ ] Verify Stage 2 does NOT show "Downloaded file too small" errors
- [ ] Verify Stage 2 shows successful video processing via webVideoUrl
- [ ] Check video files are NOT downloaded locally (webVideoUrl mode doesn't pre-download)
- [ ] Verify all temporal_windows files created in `insights/` directory
- [ ] Verify Stages 3-5 complete successfully
- [ ] Verify final checkpoint shows 0 failures

### Success Criteria
- [ ] No "subtitleLinks" references in logs
- [ ] No "Downloaded file too small" errors
- [ ] All videos processed via webVideoUrl → rumiai_runner.py → Apify scraping
- [ ] All 30 videos have temporal_windows_updated.json files
- [ ] Checkpoint shows `"completed": 30, "failed": 0`

---

## References

- **TI Specification**: `VideoProcessingTI.md` (Section 4, Function 3, lines 507-579)
- **Bug Commit**: e73bd4a (Oct 15, 2025) - "Pre Stage 2.6 implementation"
- **Working Test**: test_final (Oct 14, 2025) - 111 videos processed successfully
- **Failed Test**: test_supplement (Oct 20, 2025) - 0/30 videos processed
- **Architecture Docs**:
  - `VideoProcessingCHILD.md`
  - `FoundationCHILD.md`
  - `MLROADMAP.md`

---

## Lessons Learned

1. **Respect separation of concerns**: Each function should do ONE thing well
2. **Trust the architecture**: Fallback mechanisms may already exist at the right layer
3. **Verify assumptions**: `subtitleLinks` seemed like video links but weren't
4. **Test before deploy**: The fix was never validated with actual API responses
5. **Follow the spec**: VideoProcessingTI.md was clear about Function 3's scope
6. **Check ALL code paths**: `pause_handler.py` and `video_processor.py` should have the same logic
7. **Deep investigation is critical**: Initial fix seemed simple (remove subtitleLinks) but pause_handler was the real issue
8. **Trace actual execution**: Checking logs revealed which code path was actually being used
9. **Don't assume consistency**: Two functions doing similar things doesn't mean they use the same approach
10. **API field names are misleading**: `mime_type=video_mp4` in URL doesn't mean the file IS a video

### Why This Bug Was Hard to Find

1. **Multiple layers of abstraction**: The bug spanned 3 files and 2 different processing paths
2. **Misleading API response**: `subtitleLinks` with `mime_type=video_mp4` looked like video URLs
3. **Exception handling masked the issue**: Errors were caught but handled incorrectly (marked as failed instead of trying fallback)
4. **Inconsistent implementations**: `pause_handler` and `video_processor` should have been identical but weren't
5. **Pre-download optimization complicated the flow**: The attempt to pre-download videos added another layer where bugs could hide

### Prevention for Future

1. **Unified processing logic**: Create a single function that both `pause_handler` and `video_processor` call
2. **Integration tests**: Test with actual Apify API responses, not mocked data
3. **Validate assumptions**: When API changes, download a sample file and check its actual size/type
4. **Document code paths**: Make it clear when pause_handler vs video_processor is used
5. **Add logging**: More debug logs showing which code path is taken and why

---

## Discovery & Validation (2025-10-21)

**Status**: ✅ Comprehensive discovery complete, findings validated against TI specification

### Verification Completed

1. ✅ **Read VideoProcessingTI.md** - Canonical architecture specification
2. ✅ **Read actual implementation files** - video_download.py, video_processor.py, pause_handler.py, main.py
3. ✅ **Examined actual API responses** - Verified from drinkpoppi/selected_videos.json
4. ✅ **Traced execution path** - Confirmed from test_supplement_20251021_071900.log
5. ✅ **Validated rumiai_runner.py contract** - Confirmed TikTok URL support

---

## Critical Findings

### Finding #1: TI Specification is Outdated ⚠️

**VideoProcessingTI.md (lines 163-171)**:
```python
ApifyVideoMetadataSchema = {
    "videoMeta": {
        "downloadAddr": str,  # Required, Valid HTTP/HTTPS URL
    },
}
```

**Reality (verified from actual data)**:
```json
{
  "videoMeta": {
    "downloadAddr": null  // ← MISSING/NULL in current Apify API
  }
}
```

**Impact**: TI assumes `downloadAddr` is always present, but Apify API changed (Oct 2025) and now returns null.

---

### Finding #2: subtitleLinks Verified as Caption Files

**Actual API response** (drinkpoppi/bucket_18-33s/selected_videos.json):
```json
{
  "videoMeta": {
    "subtitleLinks": [
      {
        "LanguageID": null,
        "downloadLink": "https://v16m-webapp.tiktokcdn-us.com/..."
      }
    ]
  }
}
```

**Verified via HTTP HEAD request**:
```
content-length: 555 bytes    ← Caption file (NOT a 5-20MB video!)
content-type: video/mp4      ← MISLEADING (actual content is subtitle data)
```

**Confirmed**: Stage2Fix analysis was correct - `subtitleLinks` contains subtitle/caption files (555-3577 bytes), not video MP4s.

---

### Finding #3: Execution Path Confirmed

**From test_supplement_20251021_071900.log**:
```
Step 3-4: Processing 11 videos
ml_pipeline.stage2_processing.pause_handler - INFO - Processing video 1/11: 7551116787392220446
ml_pipeline.stage2_processing.pause_handler - ERROR - Failed to download video: Downloaded file too small: 1879 bytes
```

**Confirmed**:
- ✅ `pause_handler.py` is the default execution path (not `video_processor.py`)
- ✅ All videos failed with "Downloaded file too small" (1208-3577 bytes)
- ✅ No `webVideoUrl` fallback was attempted (pause_handler lacks hybrid logic)

---

### Finding #4: Two Implementations, One Spec

**TI Specification** (VideoProcessingTI.md lines 662-727):
- Only defines ONE function: `process_videos_with_pause_support()`
- Line 726 comment: "rest of processing logic - same as process_videos_sequential"

**Actual Implementation**:
- TWO separate files: `video_processor.py` and `pause_handler.py`
- DIFFERENT logic: video_processor has hybrid approach, pause_handler doesn't

**Comparison**:

| Feature | video_processor.py (lines 138-151) | pause_handler.py (lines 107-115) |
|---------|-----------------------------------|----------------------------------|
| Local file check | ✅ `if os.path.exists(local_video_path)` | ❌ No |
| webVideoUrl fallback | ✅ `elif 'webVideoUrl' in video` | ❌ No |
| Error handling | ✅ Tries fallback before marking failed | ❌ Marks as failed immediately |
| Used by default? | ❌ No (requires `enable_pause_support=False`) | ✅ Yes (default) |
| Matches TI spec? | ❌ No (hybrid not in TI) | ✅ Yes (but TI is outdated) |

**Root cause**: Code duplication without consistency enforcement.

---

### Finding #5: RumiAI URL Support Confirmed

**rumiai_runner.py** (line 28, 80, 503):
```python
from rumiai_v2.api import ApifyClient  # Line 28
self.apify = ApifyClient(self.settings.apify_token)  # Line 80
logger.error("Please provide a complete TikTok URL starting with http://")  # Line 503
```

✅ **Confirmed**: RumiAI can accept TikTok URLs and will scrape them via Apify internally with `shouldDownloadVideos: True`

**This validates the hybrid approach**: When no local file exists, passing `webVideoUrl` to RumiAI works correctly.

---

### Finding #6: Control Flow Decision Point

**main.py (lines 119-124)**:
```python
if enable_pause_support:
    # Use pause handler (checks for Ctrl+C between videos)
    process_videos_with_pause_support(remaining_videos, bucket_name, checkpoint, config)
else:
    # Use direct processing (no pause handling)
    stats = process_videos_sequential(remaining_videos, bucket_name, checkpoint, config)
```

**Default**: `enable_pause_support=True` (pause_handler is used by default)

**Why this matters**: The working hybrid logic in `video_processor.py` is NOT used by default!

---

## Architectural Assessment

### The Proposed Fix is CORRECT ✅

**Stage2Fix Changes 1-3 are architecturally sound**:

1. ✅ **Remove subtitleLinks from video_download.py** - Restores separation of concerns (download function should only handle direct downloads)
2. ✅ **Remove subtitleLinks from main.py** - Prevents pre-download of caption files
3. ✅ **Add hybrid logic to pause_handler.py** - Makes it consistent with video_processor.py

**Validates against TI**:
- TI says `download_video()` should raise `DownloadError` on failure (line 520) ✅
- TI says pause_handler should use "same logic as process_videos_sequential" (line 726) ✅
- Fix makes actual implementation match TI intent ✅

### But There Are Architectural Gaps ⚠️

**Gap #1: TI Doesn't Specify Hybrid Approach**

The TI (VideoProcessingTI.md) does NOT document:
- `webVideoUrl` fallback mechanism
- Hybrid approach (local file OR URL)
- How to handle missing `downloadAddr`

**The hybrid approach in `video_processor.py` is an undocumented enhancement**, not a TI-specified feature.

**Gap #2: Code Duplication**

Two files (`video_processor.py` and `pause_handler.py`) implement the same video processing logic with different approaches. This violates DRY principle and led to the inconsistency.

**Gap #3: Outdated Schema**

TI schema marks `downloadAddr` as "Required" but it's now optional/null in Apify API.

---

## Next Steps

### Immediate (Fix the Bug) - PRIORITY 1

**Goal**: Unblock test_supplement by fixing the immediate bug

**Tasks**:
- [ ] **Change 1**: Remove `subtitleLinks` logic from `video_download.py` (lines 44-51, 69)
- [ ] **Change 2**: Remove `subtitleLinks` check from `main.py` (lines 90-92)
- [ ] **Change 3**: Add hybrid logic to `pause_handler.py` (replace lines 107-115)
- [ ] **Syntax validation**: Run `python -m py_compile` on all 3 files
- [ ] **Test execution**: Run test_supplement with `--auto-confirm`
- [ ] **Verify success**: All 30 videos processed via webVideoUrl, 0 failures

**Expected outcome**: test_supplement passes, all videos processed successfully via webVideoUrl fallback.

**Timeline**: ~1-2 hours (changes + testing)

---

### Short-Term (Refactor for Consistency) - PRIORITY 2

**Goal**: Eliminate code duplication and architectural inconsistency

**Tasks**:
- [ ] **Create shared helper function** `_process_single_video()`:
  ```python
  def _process_single_video(video: dict, bucket_path: str, checkpoint: dict, checkpoint_path: str) -> None:
      """
      Shared processing logic for both pause_handler and video_processor.

      Implements hybrid approach: local file OR webVideoUrl.
      """
      video_id = video['id']
      local_video_path = f"{bucket_path}videos/{video_id}.mp4"

      # Hybrid approach: Use local file if exists, otherwise use TikTok URL
      if os.path.exists(local_video_path):
          video_path = local_video_path
          logger.info(f"Using local file for video {video_id}")
      elif 'webVideoUrl' in video:
          video_path = video['webVideoUrl']
          logger.info(f"Using webVideoUrl for video {video_id}")
      else:
          raise ValueError(f"Video {video_id} missing: no local file and no webVideoUrl")

      # Run RumiAI pipeline
      result = run_rumiai_pipeline(video_path=video_path, video_id=video_id, ...)

      # Validate output
      insights_path = f"{RUMIAI_OUTPUT_DIR}{video_id}_temporal_windows_updated.json"
      if not os.path.exists(insights_path):
          raise ProcessingError(...)

      # Mark as completed
      checkpoint['completed'] += 1
      checkpoint['completed_video_ids'].append(video_id)
      save_checkpoint_with_backup(checkpoint_path, checkpoint)
  ```

- [ ] **Update pause_handler.py** to call `_process_single_video()`
- [ ] **Update video_processor.py** to call `_process_single_video()`
- [ ] **Add unit tests** for `_process_single_video()` covering both code paths
- [ ] **Integration test**: Verify both pause_handler and video_processor work identically

**Expected outcome**: Single source of truth for video processing logic, no code duplication.

**Timeline**: ~2-3 hours (refactoring + tests)

---

### Long-Term (Update Documentation) - PRIORITY 3

**Goal**: Update TI and CHILD docs to reflect Apify API reality

**Tasks**:

#### A. Update VideoProcessingTI.md
- [ ] **Line 169**: Change `"downloadAddr": str, # Required` → `"downloadAddr": str | None, # Optional (may be null)`
- [ ] **Lines 507-579**: Update `download_video()` function spec to document behavior when `downloadAddr` is missing
- [ ] **Add new section**: "Hybrid Processing Approach" documenting local file + webVideoUrl fallback
- [ ] **Add new schema field**: `"webVideoUrl": str, # Required, TikTok video page URL`
- [ ] **Update Section 2**: Add `webVideoUrl` to StageInput contract
- [ ] **Update error handling**: Document that `DownloadError` is expected and should be handled by orchestrator

#### B. Update VideoProcessingCHILD.md
- [ ] **Section 2.3.2**: Add note about Apify API change (Oct 2025)
- [ ] **Section 5.1**: Update ApifyVideoMetadataSchema (`downloadAddr` optional, add `webVideoUrl`)
- [ ] **Section 2.3.3**: Document hybrid approach (local file OR webVideoUrl)
- [ ] **Section 6.2**: Update error cases to reflect hybrid fallback

#### C. Create Migration Guide
- [ ] **Document**: `docs/ApifyAPIChanges_Oct2025.md`
  - What changed in Apify API
  - Why `downloadAddr` is now null
  - How hybrid approach handles this
  - Backward compatibility notes

**Expected outcome**: TI and docs accurately reflect current architecture and API reality.

**Timeline**: ~3-4 hours (documentation updates)

---

### Optional (Future Enhancements) - PRIORITY 4

**Goal**: Additional improvements for robustness

**Tasks**:
- [ ] **Pre-download optimization**: Implement parallel downloads while processing (save 10-30s per video)
- [ ] **Retry failed videos**: Add `--retry-failed` flag to reprocess failed videos from checkpoint
- [ ] **API field validation**: Add runtime check to warn if Apify API structure changes again
- [ ] **Metrics logging**: Track how often local files vs webVideoUrl is used
- [ ] **Integration test suite**: Test with real Apify API responses (not mocked)

**Timeline**: ~4-6 hours (enhancements + tests)

---

## Implementation Checklist

### Pre-Implementation
- [x] ✅ Discovery complete (TI read, code reviewed, data verified)
- [x] ✅ Architectural assessment complete (fix validated against TI)
- [ ] Backup current files (video_download.py, main.py, pause_handler.py)
- [ ] Create git branch: `fix/stage2-subtitlelinks-bug`
- [ ] Review all three code changes carefully

### Priority 1: Immediate Fix
- [ ] Apply Change 1 (video_download.py)
- [ ] Apply Change 2 (main.py)
- [ ] Apply Change 3 (pause_handler.py)
- [ ] Syntax validation (all 3 files)
- [ ] Clean test data: `rm -rf data/clients/test_production/hashtags/test_supplement/`
- [ ] Run test: `python rumiai_ml_batch.py --client test_production --analysis-type hashtag --target test_supplement --video-count 15 --auto-confirm`
- [ ] Verify: 30 videos processed, 0 failures
- [ ] Commit changes: `git commit -m "fix(stage2): Remove subtitleLinks logic, add hybrid approach to pause_handler"`

### Priority 2: Refactoring
- [ ] Create `_process_single_video()` helper function
- [ ] Update pause_handler.py to use helper
- [ ] Update video_processor.py to use helper
- [ ] Write unit tests
- [ ] Run integration tests
- [ ] Commit changes: `git commit -m "refactor(stage2): Extract shared video processing logic"`

### Priority 3: Documentation
- [ ] Update VideoProcessingTI.md
- [ ] Update VideoProcessingCHILD.md
- [ ] Create ApifyAPIChanges_Oct2025.md
- [ ] Commit changes: `git commit -m "docs(stage2): Update TI and CHILD docs for Apify API changes"`

### Priority 4: Optional Enhancements
- [ ] Implement parallel downloads
- [ ] Add --retry-failed flag
- [ ] Add API validation checks
- [ ] Commit changes: `git commit -m "feat(stage2): Add optional enhancements"`

---

## Success Criteria

### Immediate Fix Success Criteria
✅ test_supplement completes with 0 failures
✅ All videos processed via webVideoUrl (no local downloads)
✅ All temporal_windows_updated.json files created
✅ Checkpoint shows `"completed": 30, "failed": 0`
✅ No "Downloaded file too small" errors in logs
✅ No "subtitleLinks" references in logs

### Refactoring Success Criteria
✅ No code duplication between pause_handler and video_processor
✅ Single source of truth for video processing logic
✅ Unit tests pass for shared helper function
✅ Integration tests pass for both code paths

### Documentation Success Criteria
✅ TI accurately reflects Apify API (downloadAddr optional)
✅ Hybrid approach documented in TI
✅ CHILD docs updated with API change notes
✅ Migration guide explains Oct 2025 API changes

---

## Risk Assessment

### Immediate Fix Risks
- **Risk**: Fix doesn't work, videos still fail
  **Mitigation**: Comprehensive discovery validates fix; test with actual data before deploying

- **Risk**: webVideoUrl missing in some videos
  **Mitigation**: Logs show webVideoUrl is always present in actual API responses

- **Risk**: RumiAI doesn't accept URLs
  **Mitigation**: rumiai_runner.py code confirms URL support (line 503)

### Refactoring Risks
- **Risk**: Shared function breaks existing behavior
  **Mitigation**: Unit tests + integration tests + verify with test_supplement

- **Risk**: Performance regression
  **Mitigation**: Shared function has same logic, no performance change expected

### Documentation Risks
- **Risk**: TI changes conflict with future API changes
  **Mitigation**: Document as current state (Oct 2025), add note about potential future changes

---

## Questions Answered

### Q1: What does the TI say `download_video()` should do when `downloadAddr` is missing?
**A**: TI assumes `downloadAddr` is always present (line 524: direct access). TI is outdated; needs update to reflect API reality.

### Q2: Is the hybrid approach (local file → webVideoUrl) in the TI spec?
**A**: No. The hybrid approach in `video_processor.py` is an undocumented implementation enhancement, not a TI-specified feature.

### Q3: Why do `pause_handler.py` and `video_processor.py` exist as separate paths?
**A**: TI only specifies one function (`process_videos_with_pause_support`). The split into two files appears to be an implementation decision, but they should have identical processing logic per TI line 726.

### Q4: Does the TI specify that `download_video()` should raise an error and let the orchestrator handle fallback?
**A**: Yes. TI line 520 says function "Raises: DownloadError if download fails after max_attempts". The orchestrator (pause_handler/video_processor) should handle the error.

### Q5: Is the proposed fix (remove subtitleLinks + add hybrid to pause_handler) architecturally correct per TI?
**A**: Yes. It makes actual implementation match TI intent (pause_handler should use "same logic" as video_processor). However, the hybrid approach itself is not in the TI and should be added to the spec.

---

## Conclusion

**The Stage2Fix analysis was fundamentally correct**, but comprehensive discovery revealed:

1. ✅ **subtitleLinks contains caption files** - Verified (555-3577 bytes)
2. ✅ **pause_handler lacks hybrid logic** - Confirmed (lines 107-115)
3. ✅ **This is the default code path** - Verified from logs
4. ✅ **Proposed fix is architecturally sound** - Validated against TI

**Additional findings**:
- ⚠️ TI specification is outdated (assumes `downloadAddr` always present)
- ⚠️ Code duplication between pause_handler and video_processor
- ⚠️ Hybrid approach is undocumented enhancement

**Recommended approach**:
1. **Apply immediate fix** (Priority 1) - Unblock test_supplement
2. **Refactor for consistency** (Priority 2) - Eliminate code duplication
3. **Update documentation** (Priority 3) - Bring TI in line with reality