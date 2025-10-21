# Bug 2.2: Incomplete Fix - subtitleLinks Still in RumiAI Core

**Date**: 2025-10-21
**Severity**: CRITICAL
**Status**: DISCOVERED - Original fix incomplete

---

## Executive Summary

The Stage 2 fix applied to `ml_pipeline/stage2_processing/` was **incomplete**. The same subtitleLinks bug exists in `rumiai_v2/core/models/video.py`, causing RumiAI to download caption files when processing TikTok URLs.

**Result**: 26 out of 27 videos still downloaded as 1-6KB caption files instead of actual videos.

---

## Discovery Timeline

### Initial Fix (2025-10-21 10:30)
Applied changes to 3 files:
1. ✅ `ml_pipeline/stage2_processing/video_download.py` - Removed subtitleLinks
2. ✅ `ml_pipeline/stage2_processing/main.py` - Removed subtitleLinks check
3. ✅ `ml_pipeline/stage2_processing/pause_handler.py` - Added hybrid logic

### Test Run (2025-10-21 10:49)
- Stage 1: ✅ Completed successfully
- Stage 2: ❌ All videos failed processing
- Logs showed: "Processing video X/Y: VIDEO_ID (TikTok URL)" ✅
- No "Downloaded file too small" errors in Stage 2 ✅

**Initial Assessment**: Fix appeared to be working (incorrect!)

### User Discovery (2025-10-21 11:57)
User checked `/temp` directory and found:
- 26 videos: 1.1KB - 6.4KB (caption files) ❌
- 1 video: 4.3MB (actual video) ✅

**Critical Question**: "Are you sure stage 2 fix is proven to work? Why did only 1 video download?"

---

## Root Cause Analysis

### The Complete Bug Chain

```
Stage 2 Processing Flow:
1. pause_handler.py checks for local file
   ✅ No local file found (expected)

2. pause_handler.py uses webVideoUrl fallback
   ✅ Passes TikTok URL to rumiai_runner.py (CORRECT)

3. rumiai_runner.py receives TikTok URL
   ✅ Calls ApifyClient.scrape_video(url, shouldDownloadVideos=True)

4. Apify scrapes the TikTok page
   ✅ Returns video metadata with subtitleLinks

5. rumiai_v2/core/models/video.py processes metadata
   ❌ STILL TRIES SUBTITLELINKS (lines 52-56) - BUG NOT FIXED!

6. Downloads 1-6KB caption file instead of video
   ❌ Video processing fails (no actual video to analyze)
```

---

## Evidence from temp/ Directory

**Files created during test run (Oct 21 11:30-11:51)**:

```bash
$ ls -lh temp/*.mp4 | grep "Oct 21 11:" | awk '{print $5, $9}'

1.1K temp/7504035881909538078.mp4    # Caption file
1.2K temp/7546701607405112598.mp4    # Caption file
1.8K temp/7555957488936439047.mp4    # Caption file
2.1K temp/7512125189765778731.mp4    # Caption file
2.1K temp/7530069183997005078.mp4    # Caption file
2.2K temp/7518016071878692118.mp4    # Caption file
2.4K temp/7497429703259589894.mp4    # Caption file
2.4K temp/7542079160676519189.mp4    # Caption file
2.4K temp/7543744690110156087.mp4    # Caption file
2.5K temp/7533600925676539158.mp4    # Caption file
2.5K temp/7548578672924478733.mp4    # Caption file
2.9K temp/7509655146327084334.mp4    # Caption file
3.0K temp/7552956451547778326.mp4    # Caption file
3.2K temp/7563442903415147831.mp4    # Caption file
3.3K temp/7500428104788184362.mp4    # Caption file
3.3K temp/7520652862678895927.mp4    # Caption file
3.4K temp/7560820561568451862.mp4    # Caption file
3.8K temp/7535565770068217119.mp4    # Caption file
4.0K temp/7558921764877749517.mp4    # Caption file
4.1K temp/7504775279043202335.mp4    # Caption file
4.1K temp/7562237575491046669.mp4    # Caption file
4.1K temp/7563350298714983693.mp4    # Caption file
4.2K temp/7528810258563747103.mp4    # Caption file
4.6K temp/7535964586088860958.mp4    # Caption file
4.8K temp/7505540770628177182.mp4    # Caption file
6.4K temp/7504815285950532894.mp4    # Caption file
4.3M temp/7550741061493083410.mp4    # ACTUAL VIDEO ✓
```

**Result**: 26/27 videos (96%) are still caption files!

---

## The Missing Fix Location

**File**: `rumiai_v2/core/models/video.py`
**Lines**: 52-56
**Problem**: Same subtitleLinks logic as video_download.py

```python
# NEW: Try subtitleLinks (new API format Oct 2025)
elif data.get('videoMeta', {}).get('subtitleLinks'):
    subtitle_links = data['videoMeta']['subtitleLinks']
    if subtitle_links and len(subtitle_links) > 0:
        download_url = subtitle_links[0].get('downloadLink', subtitle_links[0].get('tiktokLink', ''))
```

**Current Priority Order** (rumiai_v2/core/models/video.py lines 44-62):
1. `videoUrl` (if present)
2. `downloadAddr` (if present)
3. **`subtitleLinks`** ❌ BUG - Downloads caption files
4. `mediaUrls` (if present)
5. `downloadUrl` (fallback)

---

## Why the Bug Was Missed

1. **Scope Assumption**: Assumed bug was only in `ml_pipeline/stage2_processing/`
2. **Log Misinterpretation**: Saw "TikTok URL" in logs and thought fix was working
3. **No File Size Validation**: Didn't check actual downloaded file sizes in temp/
4. **Apify Quota Limit**: Processing failures masked the real issue (blamed on API quota)

---

## Impact Assessment

### What Worked ✅
- Stage 2 pause_handler hybrid logic (local file → webVideoUrl)
- TikTok URLs passed to rumiai_runner.py correctly
- No "Downloaded file too small" errors in ml_pipeline logs
- Stage 1 video selection completed successfully

### What Failed ❌
- rumiai_runner.py still downloads caption files via subtitleLinks
- 96% of videos (26/27) downloaded as 1-6KB caption files
- Video processing fails (can't analyze caption files as videos)
- All videos marked as failed in Stage 2

### Actual vs Expected Behavior

**Expected** (after fix):
```
rumiai_runner.py receives TikTok URL
→ Apify scrapes metadata
→ rumiai_v2 tries: videoUrl → downloadAddr → mediaUrls → downloadUrl
→ Downloads 5-20MB video MP4
→ Processes successfully
```

**Actual** (current):
```
rumiai_runner.py receives TikTok URL
→ Apify scrapes metadata
→ rumiai_v2 tries: videoUrl → downloadAddr → subtitleLinks ❌
→ Downloads 1-6KB caption file
→ Processing fails (not a video)
```

---

## Required Fix

### Change 4: rumiai_v2/core/models/video.py

**File**: `rumiai_v2/core/models/video.py`
**Lines to Remove**: 52-56

**Before**:
```python
# Get download URL - try multiple possible field names (API changed Oct 2025)
download_url = ''
# First try videoUrl (as shown in JS implementation)
if data.get('videoUrl'):
    download_url = data.get('videoUrl')
# Then try downloadAddr (old API)
elif data.get('downloadAddr'):
    download_url = data.get('downloadAddr')
# NEW: Try subtitleLinks (new API format Oct 2025)
elif data.get('videoMeta', {}).get('subtitleLinks'):
    subtitle_links = data['videoMeta']['subtitleLinks']
    if subtitle_links and len(subtitle_links) > 0:
        download_url = subtitle_links[0].get('downloadLink', subtitle_links[0].get('tiktokLink', ''))
# Then try mediaUrls array
elif data.get('mediaUrls') and len(data.get('mediaUrls', [])) > 0:
    download_url = data.get('mediaUrls')[0]
# Finally fallback to downloadUrl
else:
    download_url = data.get('downloadUrl', '')
```

**After**:
```python
# Get download URL - try multiple possible field names (API changed Oct 2025)
download_url = ''
# First try videoUrl (as shown in JS implementation)
if data.get('videoUrl'):
    download_url = data.get('videoUrl')
# Then try downloadAddr (old API)
elif data.get('downloadAddr'):
    download_url = data.get('downloadAddr')
# Then try mediaUrls array
elif data.get('mediaUrls') and len(data.get('mediaUrls', [])) > 0:
    download_url = data.get('mediaUrls')[0]
# Finally fallback to downloadUrl
else:
    download_url = data.get('downloadUrl', '')
```

**Lines to Remove**: 52-56 (5 lines total)

---

## Test Plan for Complete Fix

### 1. Apply Change 4
```bash
# Backup first
cp rumiai_v2/core/models/video.py rumiai_v2/core/models/video.py.backup

# Apply fix (remove lines 52-56)
# (manual edit or script)

# Validate syntax
python3 -m py_compile rumiai_v2/core/models/video.py
```

### 2. Clean temp/ Directory
```bash
# Remove old caption files
find temp/ -name "*.mp4" -size -100k -mtime -1 -delete

# Verify cleanup
ls -lh temp/*.mp4 | grep "Oct 21"
```

### 3. Re-run Test (Small Scale)
```bash
# Test with 5 videos only
python rumiai_ml_batch.py \
  --client test_production \
  --analysis-type hashtag \
  --target test_supplement \
  --video-count 5 \
  --auto-confirm
```

### 4. Validate Downloaded Files
```bash
# Should see 5 videos, each >1MB
ls -lh temp/*.mp4 | tail -5

# None should be <100KB
find temp/ -name "*.mp4" -size -100k -mtime -1 | wc -l  # Should be 0
```

---

## Success Criteria (Complete Fix)

### File Size Validation
- ✅ All downloaded videos > 1MB (typical: 5-20MB)
- ✅ ZERO files < 100KB in temp/
- ✅ No caption files (1-6KB)

### Log Validation
- ✅ No "Downloaded file too small" errors
- ✅ No subtitleLinks references in error messages
- ✅ Videos process successfully through RumiAI

### Processing Validation
- ✅ temporal_windows_updated.json files created
- ✅ Checkpoint shows `"failed": 0`
- ✅ All videos marked as completed

---

## Lessons Learned

1. **Search Scope**: When fixing a bug, search the ENTIRE codebase for the pattern
2. **Integration Points**: Consider all code paths, not just the entry point
3. **Validation**: Check actual file sizes, not just log messages
4. **Test Data**: Examine actual downloaded files before declaring success
5. **Grep Thoroughly**: `grep -r "subtitleLinks" .` would have found both locations

---

## Related Files

- **Stage2Fix.md** - Original bug analysis (incomplete scope)
- **CHANGES_APPLIED.md** - Changes 1-3 (incomplete fix)
- **Bug2.2.md** - This document (complete analysis)

---

## Next Steps

1. ✅ Document findings in Bug2.2.md (this file)
2. [ ] Apply Change 4 to rumiai_v2/core/models/video.py
3. [ ] Run syntax validation
4. [ ] Clean temp/ directory
5. [ ] Re-test with small scale (5 videos)
6. [ ] Validate file sizes in temp/
7. [ ] Update CHANGES_APPLIED.md with Change 4
8. [ ] Update Stage2Fix.md with complete fix
9. [ ] Run full production test when Apify quota resets

---

## File Locations

**Files Modified (Changes 1-3)** ✅:
- `ml_pipeline/stage2_processing/video_download.py`
- `ml_pipeline/stage2_processing/main.py`
- `ml_pipeline/stage2_processing/pause_handler.py`

**File Requiring Change 4** ❌:
- `rumiai_v2/core/models/video.py` (lines 52-56)

**Backup Files**:
- `ml_pipeline/stage2_processing/*.backup` (3 files)
- `rumiai_v2/core/models/video.py.backup` (to be created)

---

## Conclusion

The original Stage 2 fix was **architecturally correct but incomplete**. The subtitleLinks bug exists in TWO locations:

1. ✅ **Fixed**: `ml_pipeline/stage2_processing/video_download.py`
2. ❌ **Not Fixed**: `rumiai_v2/core/models/video.py`

**The bug persists in the RumiAI core module**, causing 96% of videos to download as caption files.

**Action Required**: Apply Change 4 to `rumiai_v2/core/models/video.py` to complete the fix.
