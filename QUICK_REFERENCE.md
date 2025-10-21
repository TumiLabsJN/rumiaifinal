# Stage 2 Fix - Quick Reference Card

**Test Started**: 2025-10-21 10:49:02
**Log File**: test_supplement_20251021_104901.log
**Status Check**: Automatic check running (10 min timer)

---

## One-Line Status Checks

```bash
# Overall progress
/tmp/monitor_test.sh

# Is test still running?
ps aux | grep rumiai_ml_batch.py | grep -v grep

# What stage are we on?
grep -E "Stage [0-9]|Step [0-9]" test_supplement_20251021_104901.log | tail -5

# Has Stage 2 started? (The critical fix test)
grep -q "Step 3-4: Processing" test_supplement_20251021_104901.log && echo "✅ Stage 2 STARTED" || echo "⏳ Not yet (still in Stage 1)"

# Are videos using webVideoUrl? (Success indicator)
grep -c "TikTok URL" test_supplement_20251021_104901.log

# Any "file too small" errors? (Failure indicator)
grep -c "Downloaded file too small" test_supplement_20251021_104901.log
```

---

## Critical Success Checks (Run When Stage 2 Completes)

```bash
# 1. How many videos succeeded?
grep -c "Successfully processed video" test_supplement_20251021_104901.log

# 2. How many temporal_windows files created?
find data/clients/test_production/hashtags/test_supplement -name "*_temporal_windows_updated.json" 2>/dev/null | wc -l

# 3. Check final checkpoint status
find data/clients/test_production/hashtags/test_supplement -name "stage_2_checkpoint.json" -exec cat {} \; | grep -E "completed|failed" | head -5

# 4. Were there ANY failures?
find data/clients/test_production/hashtags/test_supplement -name "stage_2_checkpoint.json" -exec cat {} \; | grep '"failed": 0' && echo "✅ ZERO FAILURES" || echo "⚠️ Some failures detected"
```

---

## Expected Timeline

| Time | Stage | Activity |
|------|-------|----------|
| 10:49 | Start | Test launched |
| 10:50-10:58 | Stage 1 | Scraping 4 hashtag runs |
| ~11:00 | Stage 2 Start | **Fix gets tested here!** |
| 11:00-12:30 | Stage 2 | Processing 30 videos via webVideoUrl |
| 12:30-12:45 | Stages 3-5 | ML analysis & reports |
| ~12:45 | Complete | Final validation |

**Check back around 11:00 to see Stage 2 start!**

---

## What the Fix Changes

**BEFORE (Buggy)**:
```
→ Tries to download from subtitleLinks
→ Gets 555-byte caption file
→ Error: "Downloaded file too small"
→ Video marked as FAILED
```

**AFTER (Fixed)**:
```
→ No local file found
→ Uses webVideoUrl fallback
→ Passes TikTok URL to RumiAI
→ RumiAI scrapes via Apify
→ Video processed successfully ✅
```

---

## If Test Fails

```bash
# 1. Stop the running test
pkill -f rumiai_ml_batch.py

# 2. Check what went wrong
grep -i "error\|failed" test_supplement_20251021_104901.log | tail -20

# 3. Restore backups
cp ml_pipeline/stage2_processing/video_download.py.backup ml_pipeline/stage2_processing/video_download.py
cp ml_pipeline/stage2_processing/main.py.backup ml_pipeline/stage2_processing/main.py
cp ml_pipeline/stage2_processing/pause_handler.py.backup ml_pipeline/stage2_processing/pause_handler.py

# 4. Clean test data
rm -rf data/clients/test_production/hashtags/test_supplement/
```

---

## Files to Review

- **Stage2Fix.md** - Complete analysis and findings
- **CHANGES_APPLIED.md** - What was changed
- **TEST_IN_PROGRESS.md** - Monitoring guide
- **/tmp/stage2_fix_test_report.md** - Non-production test results

---

## Key Log Patterns to Search For

```bash
# Fix working (should see these)
grep "TikTok URL" test_supplement_20251021_104901.log
grep "Successfully processed video" test_supplement_20251021_104901.log

# Fix NOT working (should NOT see these)  
grep "Downloaded file too small" test_supplement_20251021_104901.log
grep "Checked: downloadAddr, subtitleLinks" test_supplement_20251021_104901.log
```

---

**Next automatic check in 10 minutes at ~11:00!**
