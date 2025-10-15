# Stage 2 Bug Fix & Test Resumption Guide

---

## 🚨 FOR FRESH CLI INSTANCE

**If you're a new CLI session being told to read this document:**

This document describes an **active bug** that was found during end-to-end testing. A previous session completed Stage 0-1 successfully (scraped and analyzed 150 videos), but Stage 2 failed with a data format error.

**What you need to do:**
1. Read the "Bug Found" section below to understand the issue
2. Read the "Proposed Fix" section - it's a simple 2-line change
3. Follow the "How to Resume Testing" section to fix and continue **without re-scraping**

**Important**: The 150 videos are already scraped and ready. You just need to fix the bug and resume Stage 2.

**Answers to Common Questions:**
- **Which fix approach?** Use the **simple approach** (just extract `video_data['videos']`) - we don't need backwards compatibility
- **Quick test first?** **Yes** - Run `test_stage2_fix.py` validation script first to verify the fix works
- **How to resume?** **Option C** - Run the full pipeline. Stage 1 will complete in ~1 second (just loads existing data)
- **Testing scope?** Apply fix → Run quick test → Run full pipeline (Stage 2 will take 3-4 hours for 150 videos)

---

## ⚡ Quick Action Plan (Do This)

**Follow these steps in order:**

1. **Apply the fix** (see "Proposed Fix" section below)
   - Edit `rumiai_ml_batch.py` lines 199-200
   - Use the simple approach (extract `video_data['videos']`)

2. **Create and run the validation test** (see "Quick Test" section below)
   - Create `test_stage2_fix.py`
   - Run it to verify the fix works
   - Should output: "✓ Loaded 50 videos" for each bucket

3. **Resume the full pipeline**
   ```bash
   export DATA_ROOT=/home/jorge/rumiaifinal/data
   python3 rumiai_ml_batch.py \
     --client test_final \
     --target test_vitamin \
     --analysis-type hashtag \
     --video-count 50 \
     --auto-confirm
   ```
   - Stage 1 will complete in ~1 second (loads existing data)
   - Stage 2 will process 150 videos (~3-4 hours)
   - Stage 2.5 will organize files

4. **Verify outputs** (after completion)
   - Check `/data/clients/test_final/.../buckets/bucket_*/analysis/insights/`
   - Should have 150 `*_temporal_windows_updated.json` files

---

**Date**: 2025-10-14
**Context**: Full end-to-end pipeline test (STEP 7 from StageTests.md)
**Status**: Stage 0-1 ✅ COMPLETE | Stage 2 ❌ FAILED (bug found) | Needs fix + resume

---

## 🐛 Bug Found: Stage 2 Data Format Error

### Issue Description
Stage 2 failed for all 3 winning buckets with:
```
TypeError: string indices must be integers, not 'str'
File: ml_pipeline/stage2_processing/main.py:82
Line: video_id = video['id']
```

### Root Cause
**Location**: `rumiai_ml_batch.py:199-200`

The orchestrator loads the entire `selected_videos.json` structure instead of extracting just the `videos` array:

```python
# CURRENT CODE (WRONG):
with open(bucket_videos_path) as f:
    video_list = json.load(f)  # Loads entire dict

# This passes the wrong structure to Stage 2
summary = stage_2_video_processing_main(
    config=config.model_dump(),
    video_list=video_list,  # ❌ This is a dict, not a list!
    bucket_name=bucket_name
)
```

**Problem**: `selected_videos.json` has this structure:
```json
{
  "bucket": "18-33s",
  "strategy": "contrastive",
  "video_count": 50,
  "selected_count": 50,
  "videos": [
    {"id": "7544734155570105656", "webVideoUrl": "...", ...},
    {"id": "7554856008615529783", "webVideoUrl": "...", ...}
  ]
}
```

Stage 2 expects a **list of video objects**, but receives the **entire dict**.

---

## 🔧 Proposed Fix

### Change Required in `rumiai_ml_batch.py`

**Line 199-200** should be changed from:
```python
with open(bucket_videos_path) as f:
    video_list = json.load(f)
```

**To (RECOMMENDED - Simple Approach):**
```python
with open(bucket_videos_path) as f:
    video_data = json.load(f)
    video_list = video_data['videos']  # Extract just the videos array
```

This is the approach you should use. The defensive approach below is not needed.

---

## 📊 Current Test Status

### ✅ Completed Successfully:
- **Stage 0**: Foundation (config, paths) - COMPLETE
- **Stage 1**: Video Discovery & Selection - COMPLETE
  - Scraped: 1,826 videos (8 scrapes)
  - Deduplicated: 697 unique videos
  - Date filtered: 255 videos (last 90 days)
  - **Winning buckets identified**: `18-33s`, `13-18s`, `60-90s`
  - **Videos selected**: 50 per bucket × 3 buckets = **150 videos ready**

### ❌ Failed (needs fix):
- **Stage 2**: Video Processing - FAILED (data format bug)
  - 0/150 videos processed
  - All 3 buckets failed immediately with TypeError

### ⏸️ Not Run Yet:
- **Stage 2.5**: File Organization - Skipped (no videos processed)
- **Stage 3+**: Future stages

---

## 🔄 How to Resume Testing (DO NOT RE-SCRAPE)

### Why We Don't Need to Re-Scrape:
1. ✅ Stage 1 completed successfully
2. ✅ All data files are intact:
   - `/data/clients/test_final/hashtags/test_vitamin/top_contrastive/winner_analysis.json`
   - `/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_*/selected_videos.json`
3. ✅ 150 videos are selected and ready for processing

### Resumption Steps:

#### 1. Apply the Fix
```bash
cd /home/jorge/rumiaifinal

# Edit rumiai_ml_batch.py
# Change lines 199-200 as described in "Proposed Fix" section above
```

#### 2. Run Stage 2 Only (Skip Stage 1)

**Option A: Modify orchestrator to skip Stage 1**
Add a CLI flag `--skip-stage-1` or comment out Stage 1 execution in `rumiai_ml_batch.py`

**Option B: Create a Stage 2-only test script**
```bash
# Create a test script that loads existing Stage 1 data
# and runs only Stage 2 + 2.5
```

**Option C: Run full pipeline (will be fast)**
```bash
export DATA_ROOT=/home/jorge/rumiaifinal/data

python3 rumiai_ml_batch.py \
  --client test_final \
  --target test_vitamin \
  --analysis-type hashtag \
  --video-count 50 \
  --auto-confirm
```

**Note**: Stage 1 will detect existing data and complete quickly (no re-scraping), then proceed to Stage 2.

#### 3. Expected Results After Fix:
```
✓ Stage 0: Foundation - COMPLETE (quick, just validates paths)
✓ Stage 1: Video Discovery & Selection - COMPLETE (loads existing data, ~1 second)
⏳ Stage 2: Video Processing - IN PROGRESS
   - Processing 150 videos (50 per bucket)
   - Using hybrid approach (webVideoUrl)
   - Expected time: ~3-4 hours (150 videos × 90s each)
✓ Stage 2.5: File Organization - COMPLETE
```

#### 4. Verify Output Location:
After successful completion, files will be at:
```
/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/
├── bucket_18-33s/analysis/insights/    (50 temporal_windows files)
├── bucket_13-18s/analysis/insights/    (50 temporal_windows files)
└── bucket_60-90s/analysis/insights/    (50 temporal_windows files)
```

---

## 🧪 Quick Test (Before Full Run)

To validate the fix works without processing 150 videos:

### Create a minimal test:
```python
# test_stage2_fix.py
import json
from pathlib import Path

# Load one bucket's selected videos
bucket_path = Path("/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s/selected_videos.json")

with open(bucket_path) as f:
    video_data = json.load(f)
    video_list = video_data['videos']  # THE FIX

print(f"✓ Loaded {len(video_list)} videos")
print(f"✓ First video ID: {video_list[0]['id']}")
print(f"✓ First video has webVideoUrl: {'webVideoUrl' in video_list[0]}")

# Test that Stage 2 can iterate
for video in video_list[:2]:  # Just first 2
    video_id = video['id']  # Should NOT error
    web_url = video.get('webVideoUrl')
    print(f"  - Video {video_id}: {web_url}")
```

**Run:**
```bash
python3 test_stage2_fix.py
```

**Expected Output:**
```
✓ Loaded 50 videos
✓ First video ID: 7544734155570105656
✓ First video has webVideoUrl: True
  - Video 7544734155570105656: https://www.tiktok.com/@_safiraxo2/video/7544734155570105656
  - Video 7554856008615529783: https://www.tiktok.com/@chiisana_koneko/video/7554856008615529783
```

If this works, the fix is correct and you can proceed with the full Stage 2 run.

---

## 📝 Test Data Summary

### Cluster Configuration:
- **Cluster ID**: `test_vitamin`
- **Hashtags**: `#vitamin`, `#vitamins`, `#dailyvitamins`, `#vitamintok`
- **Scraping**: 8 scrapes (4 hashtags × 2 runs) with 2-minute delays

### Stage 1 Results:
| Metric | Value |
|--------|-------|
| Total scraped | 1,826 videos |
| After deduplication | 697 unique (61.8% overlap) |
| After date filter (90 days) | 255 videos |
| Winning buckets | 3 buckets |
| Videos per bucket | 50 (40 top + 10 bottom) |
| **Total for Stage 2** | **150 videos** |

### Winning Buckets:
1. **18-33s** - 50 videos selected
2. **13-18s** - 50 videos selected
3. **60-90s** - 50 videos selected

### Files Ready for Processing:
```
/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/
├── bucket_18-33s/selected_videos.json    (50 videos with webVideoUrl)
├── bucket_13-18s/selected_videos.json    (50 videos with webVideoUrl)
└── bucket_60-90s/selected_videos.json    (50 videos with webVideoUrl)
```

---

## 🎯 Next Steps for Fresh CLI Instance

1. **Read this document** to understand the bug
2. **Apply the fix** to `rumiai_ml_batch.py:199-200`
3. **Run the quick test** (`test_stage2_fix.py`) to validate the fix
4. **Resume the pipeline** - Stage 1 will load existing data, Stage 2 will process 150 videos
5. **Verify outputs** in `/data/clients/test_final/.../buckets/bucket_*/analysis/insights/`

---

## 📚 Related Documentation

- **Full test guide**: `/home/jorge/rumiaifinal/StageTests.md` (STEP 7)
- **Logs**: `/home/jorge/rumiaifinal/data/logs/rumiai_ml_test_final_test_vitamin_20251014_132200.log`
- **Quick Reference**: `/home/jorge/rumiaifinal/QUICK_REFERENCE.md`
- **System Architecture**: `/home/jorge/rumiaifinal/SystemArchitecturev2.md`
