# Stage 2 & 2.5 Integration Testing Guide

## Current State

**✅ What's Complete:**
- Stage 0: Foundation (CLI parsing, config, paths)
- Stage 1: Video Discovery & Selection (scraping, bucket analysis)
- Stage 2: Video Processing (implemented and unit tested - 12/12 tests passing)
- Stage 2.5: File Organization (implemented and tested)
- **Integration:** Stage 2 & 2.5 are integrated into `rumiai_ml_batch.py`

**❌ What Needs Testing:**
- End-to-end pipeline (Stage 0 → Stage 1 → Stage 2 → Stage 2.5)
- Stage 2 & 2.5 with real scraped video data
- Data format compatibility between stages

**⚠️ Known Issue:**
Videos scraped by Stage 1 are missing `videoMeta.downloadAddr` field, causing Stage 2 download failures.

---

## Interactive Testing Process

Follow these steps in order. Each step explains **what to do** and **why we're doing it**.

---

### STEP 1: Verify Current Integration Status

**What to do:**
```bash
cd /home/jorge/rumiaifinal
grep -A5 "STAGE 2: VIDEO PROCESSING" rumiai_ml_batch.py
```

**Why:**
Check that Stage 2 and Stage 2.5 are integrated into the main orchestrator.

**Expected Result:**
You should see code that:
- Loads `winner_analysis.json`
- Loops through winning buckets
- Calls `stage_2_video_processing_main()`
- Calls `stage_2_5_file_organization_main()`

**Decision Point:**
- ✅ If integration code exists → **Proceed to STEP 2**
- ❌ If missing → Run: `git status` to check if file was modified

---

### STEP 2: Inspect Scraped Video Data Format

**What to do:**
```bash
# Check if we have scraped data from previous test
ls -la data/clients/testy_client/hashtags/test_vitamin/top_contrastive/buckets/

# Inspect a video from selected_videos.json
head -100 data/clients/testy_client/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s/selected_videos.json
```

**Why:**
We need to verify the video data structure. Stage 2 expects videos to have:
- `id` field
- `videoMeta.downloadAddr` field for downloading
- `videoMeta.duration` field for validation

The previous test showed `downloadAddr` is missing, which causes all video downloads to fail.

**Expected Result:**
You should see JSON with video objects containing fields like:
- `id`
- `authorMeta`
- `videoMeta` (check if it has `downloadAddr`)

**Decision Point:**
- ✅ If `videoMeta.downloadAddr` exists → **Proceed to STEP 4** (skip Step 3)
- ❌ If `videoMeta.downloadAddr` missing → **Proceed to STEP 3**

---

### STEP 3: Investigate Why downloadAddr is Missing

**What to do:**
```bash
# Check the Apify scraper code to see what fields are requested
grep -n "shouldDownload" ml_pipeline/stage1_discovery/apify_scraper.py

# Check if there's a scraper configuration issue
cat config/hashtag_clusters/test_vitamin.json
```

**Why:**
Apify scraper has options for what data to include. We may have disabled video downloads to save costs, but we still need the download URLs (not the actual videos yet).

The scraper parameters include:
- `shouldDownloadVideos: false` (we don't want to download videos during scraping)
- `shouldDownloadCovers: false` (we don't need cover images)

But we NEED the `downloadAddr` URLs so Stage 2 can download videos later.

**Expected Result:**
Find configuration that shows `shouldDownloadVideos: false` but still gets URLs.

**Decision Point:**
- If scraper is configured correctly but URLs missing → **Apify API issue**
- If scraper needs adjustment → **Fix scraper, re-run Stage 1**

---

### STEP 4: Run Stage 2 & 2.5 Test with Valid Video

**What to do:**
```bash
# We have a known working TikTok URL from previous test
# Let's test Stage 2 directly with that video

python3 ml_pipeline/tests/run_stage2_integration_test.py
```

**Why:**
This test uses a real TikTok URL and validates:
1. Stage 2 can scrape + download + process a single video
2. `rumiai_runner.py` works end-to-end
3. Output file (`temporal_windows_updated.json`) is created in `/insights/`
4. Output has valid JSON structure

This test bypasses the scraping issue and validates Stage 2 works independently.

**Expected Result:**
```
✅ Stage 2 Integration Test: PASSED
- Video scraped and downloaded from TikTok
- Processed through all 9 ML services
- temporal_windows_updated.json created
- Processing time: ~2-3 minutes
```

**Decision Point:**
- ✅ If test passes → **Stage 2 works!** → **Proceed to STEP 5**
- ❌ If test fails → Debug the specific failure, check logs

---

### STEP 5: Create Test with Pre-Downloaded Videos

**What to do:**
Since scraped videos don't have download URLs, create a test that:
1. Uses videos already on disk (we have one: `7384423133157100843.mp4`)
2. Manually creates a video list with file paths instead of URLs
3. Runs Stage 2 processing
4. Runs Stage 2.5 organization

**Command:**
```bash
# This test needs to be created - ask the LLM:
# "Create a Stage 2 test that uses local video files instead of download URLs"
```

**Why:**
This tests the full Stage 2 → Stage 2.5 flow while bypassing the download step. It validates:
- Video processing works
- Checkpoint/resume works
- Stage 2.5 can organize the output files
- Integration between stages works

**Expected Result:**
- Videos processed through rumiai_runner.py ✓
- temporal_windows files created ✓
- Stage 2.5 organizes files into bucket directories ✓

**Decision Point:**
- ✅ If works → **Integration is solid, just need to fix data source**
- ❌ If fails → **Debug integration issues**

---

### STEP 6: Fix the Root Cause - Apify Data Format

**What to do:**
Two options:

**Option A: Fix Apify scraping to include downloadAddr**
```bash
# Investigate Apify actor settings
# Check if there's a parameter to include download URLs without downloading files
# Update ml_pipeline/stage1_discovery/apify_scraper.py if needed
```

**Option B: Modify Stage 2 to work without pre-scraped data**
```bash
# Instead of using downloadAddr from scraping:
# 1. Stage 2 re-scrapes just the selected videos to get fresh URLs
# 2. Uses Apify single-video scraper
# 3. Gets downloadAddr at processing time
```

**Why:**
Need to bridge the gap between Stage 1 (scraping) and Stage 2 (processing). The current issue is that Stage 1 scrapes metadata but doesn't include video download URLs.

**Expected Result:**
Stage 2 can successfully download videos for processing.

**Decision Point:**
- **Option A preferred** if Apify supports it (cheaper, faster)
- **Option B fallback** if Apify doesn't provide URLs without downloading

---

### STEP 7: Run Full End-to-End Pipeline Test

**What to do:**
```bash
# With fixed data format, run complete pipeline
python3 rumiai_ml_batch.py \
  --client test_final \
  --target test_vitamin \
  --analysis-type hashtag \
  --video-count 10 \
  --auto-confirm
```

**Why:**
This is the ultimate integration test. It validates:
- Stage 0: Foundation setup ✓
- Stage 1: Scraping & bucket analysis ✓
- Stage 2: Video download & processing ✓
- Stage 2.5: File organization ✓
- Data flows correctly between all stages ✓

**Expected Result:**
```
================================================================================
PIPELINE STATUS
================================================================================
✓ Stage 0: Foundation - COMPLETE
✓ Stage 1: Video Discovery & Selection - COMPLETE
✓ Stage 2: Video Processing - COMPLETE
✓ Stage 2.5: File Organization - COMPLETE
⧗ Stage 3: Feature Aggregation - TODO
...
================================================================================

✅ Stages 0-2.5 complete!
   Processed 30 videos across 3 buckets
   Output location: /home/jorge/rumiaifinal/data/clients/test_final/...
```

**Decision Point:**
- ✅ If complete → **Pipeline works end-to-end! 🎉**
- ❌ If fails → Debug specific stage failure

---

### STEP 8: Verify Final Output Structure

**What to do:**
```bash
# Check that files are organized correctly
tree -L 5 data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/

# Verify temporal_windows files in bucket directories
ls data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s/analysis/insights/

# Validate a temporal_windows file structure
python3 -m json.tool data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s/analysis/insights/*_temporal_windows_updated.json | head -50
```

**Why:**
Verify the output matches the expected structure for Stage 3 (Feature Aggregation). The directory structure should be:
```
buckets/
  bucket_18-33s/
    analysis/
      insights/
        video1_temporal_windows_updated.json
        video2_temporal_windows_updated.json
        ...
    checkpoints/
      stage_2_checkpoint.json
    videos/
      video1.mp4
      video2.mp4
```

**Expected Result:**
- ✅ Bucket directories exist for each winning bucket
- ✅ temporal_windows files in `bucket_*/analysis/insights/`
- ✅ Checkpoint files show completed status
- ✅ JSON files have valid structure (video_id, temporal_windows, metadata, processing_timestamp)

**Decision Point:**
- ✅ If structure correct → **Ready for Stage 3 implementation!**
- ❌ If structure wrong → **Fix Stage 2.5 organization logic**

---

## Summary of Testing Goals

| Test | Purpose | Validates |
|------|---------|-----------|
| STEP 1 | Integration check | Code is integrated into orchestrator |
| STEP 2-3 | Data format investigation | Understand why downloadAddr is missing |
| STEP 4 | Stage 2 isolated test | Stage 2 works independently |
| STEP 5 | Local file test | Stage 2 + 2.5 integration works |
| STEP 6 | Root cause fix | Data pipeline is complete |
| STEP 7 | Full pipeline test | End-to-end validation |
| STEP 8 | Output verification | Ready for Stage 3 |

---

## Common Issues & Solutions

### Issue: "downloadAddr missing" error

**Cause:** Apify scraper not configured to include download URLs

**Solution:**
1. Check if `mediaUrls` field exists in scraped data (alternative to `videoMeta.downloadAddr`)
2. Update Stage 2 to check multiple possible URL fields
3. Or re-scrape with correct Apify settings

---

### Issue: "Permission denied: /data"

**Cause:** Stage 2 trying to use default DATA_ROOT=/data

**Solution:**
```bash
export DATA_ROOT=/home/jorge/rumiaifinal/data
# Or set in code: os.environ['DATA_ROOT'] = '/home/jorge/rumiaifinal/data'
```

---

### Issue: Videos failing to process through rumiai_runner.py

**Cause:** Video URL expired, or TikTok blocking access

**Solution:**
- Fresh scrape gets fresh URLs
- Use Apify proxy (already configured with proxyCountryCode: 'US')
- Check if video is still available on TikTok

---

### Issue: Stage 2.5 reports "missing files"

**Cause:** Stage 2 processing failed for those videos

**Solution:**
- Check Stage 2 checkpoint: `cat data/.../bucket_*/checkpoints/stage_2_checkpoint.json`
- Look at `failed_video_ids` list
- Stage 2.5 only organizes successful videos (expected behavior)

---

## Next Steps After Testing

Once all tests pass:

1. **Document any data format changes** needed
2. **Update Stage 1 scraper** if needed to include required fields
3. **Begin Stage 3 implementation** (Feature Aggregation)
4. **Consider optimization:**
   - Batch video processing
   - Parallel processing across buckets
   - Caching to avoid re-processing

---

## Quick Reference Commands

```bash
# Run full pipeline
python3 rumiai_ml_batch.py --client test --target test_vitamin --analysis-type hashtag --video-count 10 --auto-confirm

# Run Stage 2 test only
python3 ml_pipeline/tests/run_stage2_integration_test.py

# Run Stage 2 + 2.5 test (uses existing Stage 1 output)
python3 ml_pipeline/tests/run_stage2_stage2_5_integration_test.py

# Check pipeline logs
tail -100 data/logs/rumiai_ml_*.log

# Verify output structure
tree -L 5 data/clients/*/hashtags/*/top_contrastive/buckets/
```

---

## Getting Help

**Prompt the LLM with:**
- "I'm on STEP X, here's what happened: [paste output]"
- "The test failed with error: [paste error]"
- "I need help debugging [specific issue]"
- "What should I do if [scenario]?"

The LLM can help debug, create new tests, or adjust the integration code as needed.
