# Stage 1 Skip Logic - Test Validation Checklist

**Implementation Date**: 2025-10-24
**Bug Fixed**: Bug #1 - Stage 1 Missing Skip Logic (CRITICAL)
**File Modified**: `rumiai_ml_batch.py` (lines 19, 568-704)
**Documentation Updated**: `SkipLogic.md` (6 sections)

---

## 🧪 Test Scenarios

### **Test 1: Fresh Run (No Checkpoint)**

**Goal**: Verify Stage 1 runs fully when no checkpoint exists

**Setup**:
```bash
# Remove checkpoint if exists
rm -f /home/jorge/rumiaifinal/data/clients/test_client/hashtags/test_target/top_contrastive/checkpoints/stage_1_checkpoint.json

# Remove Stage 1 outputs
rm -f /home/jorge/rumiaifinal/data/clients/test_client/hashtags/test_target/top_contrastive/winner_analysis.json
rm -rf /home/jorge/rumiaifinal/data/clients/test_client/hashtags/test_target/top_contrastive/buckets/
```

**Execute**:
```bash
cd /home/jorge/rumiaifinal
python rumiai_ml_batch.py --client test_client --target "#test_target" --video-count 10
```

**Expected Output**:
```
STAGE 1: VIDEO DISCOVERY & SELECTION
================================================================================

[Stage 1 processing logs...]

✓ Stage 1: Video Discovery - COMPLETE
  Winning buckets: 18-33s, 33-60s, 13-18s
```

**Verify**:
- [ ] Stage 1 runs fully (not skipped)
- [ ] Checkpoint created: `checkpoints/stage_1_checkpoint.json`
- [ ] Output files created:
  - [ ] `winner_analysis.json`
  - [ ] `buckets/bucket_18-33s/selected_videos.json`
  - [ ] `buckets/bucket_33-60s/selected_videos.json`
  - [ ] `buckets/bucket_13-18s/selected_videos.json`
- [ ] Checkpoint contains correct fields:
  - [ ] `stage`: "stage_1_video_discovery"
  - [ ] `completion_timestamp`: ISO 8601 format
  - [ ] `winning_buckets`: [list of 3 buckets]
  - [ ] `output_files`: [list of 4+ files]
- [ ] Log shows: "Stage 1 checkpoint created: {path}"

---

### **Test 2: Resume with Valid Checkpoint**

**Goal**: Verify Stage 1 skips when checkpoint exists and outputs are valid

**Setup**:
```bash
# Ensure Test 1 completed successfully (checkpoint + all files exist)
ls /home/jorge/rumiaifinal/data/clients/test_client/hashtags/test_target/top_contrastive/checkpoints/stage_1_checkpoint.json
ls /home/jorge/rumiaifinal/data/clients/test_client/hashtags/test_target/top_contrastive/winner_analysis.json
```

**Execute**:
```bash
cd /home/jorge/rumiaifinal
python rumiai_ml_batch.py --client test_client --target "#test_target" --video-count 10
```

**Expected Output**:
```
STAGE 1: VIDEO DISCOVERY & SELECTION
================================================================================

✓ Stage 1: Video Discovery - SKIPPED (already complete)
  Winning buckets: 18-33s, 33-60s, 13-18s
```

**Verify**:
- [ ] Stage 1 skipped (~0 seconds, not 45 minutes)
- [ ] Log shows: "Stage 1 checkpoint found: {path}"
- [ ] Log shows: "Stage 1 already complete (checkpoint valid)"
- [ ] No Apify API calls made
- [ ] winning_buckets loaded from checkpoint
- [ ] Stage 2 proceeds with correct buckets

---

### **Test 3: Corrupt Checkpoint (Invalid JSON)**

**Goal**: Verify automatic recovery from corrupt checkpoint

**Setup**:
```bash
# Corrupt checkpoint file
echo "invalid json" > /home/jorge/rumiaifinal/data/clients/test_client/hashtags/test_target/top_contrastive/checkpoints/stage_1_checkpoint.json
```

**Execute**:
```bash
cd /home/jorge/rumiaifinal
python rumiai_ml_batch.py --client test_client --target "#test_target" --video-count 10
```

**Expected Output**:
```
STAGE 1: VIDEO DISCOVERY & SELECTION
================================================================================

[Warning: Checkpoint invalid (JSONDecodeError: ...). Deleting and re-running Stage 1.]

[Stage 1 processing logs...]

✓ Stage 1: Video Discovery - COMPLETE
  Winning buckets: ...
```

**Verify**:
- [ ] Log shows: "Checkpoint invalid (JSONDecodeError: ...)"
- [ ] Corrupt checkpoint deleted automatically
- [ ] Stage 1 re-runs fully
- [ ] New valid checkpoint created
- [ ] All output files created

---

### **Test 4: Missing Output File (Incomplete Stage 1)**

**Goal**: Verify detection of incomplete outputs and automatic re-run

**Setup**:
```bash
# Delete one output file but keep checkpoint
rm -f /home/jorge/rumiaifinal/data/clients/test_client/hashtags/test_target/top_contrastive/buckets/bucket_13-18s/selected_videos.json
```

**Execute**:
```bash
cd /home/jorge/rumiaifinal
python rumiai_ml_batch.py --client test_client --target "#test_target" --video-count 10
```

**Expected Output**:
```
STAGE 1: VIDEO DISCOVERY & SELECTION
================================================================================

[Warning: Stage 1 outputs incomplete (1 files missing). Re-running Stage 1.]

[Stage 1 processing logs...]

✓ Stage 1: Video Discovery - COMPLETE
  Winning buckets: ...
```

**Verify**:
- [ ] Log shows: "Stage 1 outputs incomplete (1 files missing)"
- [ ] Checkpoint deleted automatically
- [ ] Stage 1 re-runs fully
- [ ] Missing file recreated
- [ ] New checkpoint created

---

### **Test 5: Cluster Mode (cluster_analytics.json included)**

**Goal**: Verify cluster_analytics.json tracked in checkpoint for cluster mode

**Setup**:
```bash
# Use cluster target (no # prefix)
rm -rf /home/jorge/rumiaifinal/data/clients/test_client/hashtags/wellness/
```

**Execute**:
```bash
cd /home/jorge/rumiaifinal
python rumiai_ml_batch.py --client test_client --target wellness --analysis-type hashtag --video-count 10
```

**Expected Output**:
```
[Stage 1 processing logs...]

[Info: Cluster mode detected: Added cluster_analytics.json to checkpoint]

✓ Stage 1: Video Discovery - COMPLETE
  Winning buckets: ...
```

**Verify**:
- [ ] Log shows: "Cluster mode detected: Added cluster_analytics.json to checkpoint"
- [ ] Checkpoint `output_files` includes `cluster_analytics.json`
- [ ] cluster_analytics.json file exists
- [ ] Resume run skips Stage 1 correctly (all files including cluster_analytics verified)

---

### **Test 6: Missing Required Fields in Checkpoint**

**Goal**: Verify detection of incomplete checkpoint schema

**Setup**:
```bash
# Create checkpoint with missing field
cat > /home/jorge/rumiaifinal/data/clients/test_client/hashtags/test_target/top_contrastive/checkpoints/stage_1_checkpoint.json <<EOF
{
  "stage": "stage_1_video_discovery",
  "winning_buckets": ["18-33s", "33-60s", "13-18s"]
}
EOF
```

**Expected Output**:
```
STAGE 1: VIDEO DISCOVERY & SELECTION
================================================================================

[Warning: Checkpoint corrupt (missing fields: ['output_files', 'completion_timestamp']). Deleting and re-running Stage 1.]

[Stage 1 processing logs...]
```

**Verify**:
- [ ] Log shows: "Checkpoint corrupt (missing fields: ...)"
- [ ] Checkpoint deleted
- [ ] Stage 1 re-runs
- [ ] Complete checkpoint created

---

## 📊 Performance Validation

### **Baseline Metrics** (from Bug #1 analysis)

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| **Resume Time (Stage 1)** | 45 minutes | ~0.05 seconds | **99.998%** |
| **Resume Cost (Stage 1)** | $0.80 | $0 | **$0.80 saved** |
| **Apify Calls on Resume** | 8 scrapes | 0 scrapes | **100% reduction** |
| **Skip Overhead** | N/A | ~50ms | Negligible |

### **Performance Test**

**Measure skip overhead**:
```bash
# Test 1: Fresh run (baseline)
time python rumiai_ml_batch.py --client test_client --target "#test_target" --video-count 10

# Test 2: Resume (skip)
time python rumiai_ml_batch.py --client test_client --target "#test_target" --video-count 10
```

**Verify**:
- [ ] Resume completes Stage 1 in < 1 second
- [ ] No Apify API calls logged
- [ ] winning_buckets correctly loaded

---

## 🔍 Code Review Checklist

### **rumiai_ml_batch.py**

- [ ] Line 19: `from datetime import datetime, timezone` imported
- [ ] Line 572: Checkpoint path defined: `analysis_base / "checkpoints" / "stage_1_checkpoint.json"`
- [ ] Line 575-621: Skip logic checks:
  - [ ] Checkpoint exists check
  - [ ] JSON load + schema validation
  - [ ] Required fields validation
  - [ ] Output files existence check
  - [ ] Try/except handles JSONDecodeError, ValueError, KeyError
  - [ ] Corrupt checkpoint deleted + re-run triggered
- [ ] Line 624-704: Stage 1 execution when checkpoint missing:
  - [ ] VideoDiscovery called
  - [ ] winner_analysis.json loaded
  - [ ] winning_buckets extracted
  - [ ] Checkpoint created with all fields
  - [ ] Cluster mode detection + cluster_analytics.json inclusion
  - [ ] Checkpoint write with try/except (non-fatal)
- [ ] winning_buckets variable populated in both skip and run paths

### **SkipLogic.md**

- [ ] Line 715: Stage 1 grade updated to A+
- [ ] Line 726: Overall grade updated to B+
- [ ] Line 728-732: Resume overhead updated (3 rows)
- [ ] Line 745: Cost table Stage 1 row updated
- [ ] Line 766: Fixed situation table Stage 1 note updated
- [ ] Line 656-664: Bug #1 status updated to FIXED with implementation notes

---

## ✅ Sign-Off Criteria

**All tests must pass before marking Bug #1 as RESOLVED**:

- [ ] **Test 1** (Fresh run): PASS
- [ ] **Test 2** (Resume): PASS
- [ ] **Test 3** (Corrupt checkpoint): PASS
- [ ] **Test 4** (Missing output): PASS
- [ ] **Test 5** (Cluster mode): PASS
- [ ] **Test 6** (Missing checkpoint fields): PASS
- [ ] **Performance**: Resume < 1 second
- [ ] **Code Review**: All checkboxes checked

**Additional Validation**:
- [ ] No regression in Stage 2-7 execution
- [ ] Pipeline completes end-to-end successfully
- [ ] Logs are clear and actionable

---

## 📝 Test Results Log

**Date**: _____________________
**Tester**: _____________________
**Environment**: _____________________

| Test | Status | Duration | Notes |
|------|--------|----------|-------|
| Test 1 | ⬜ PASS / ❌ FAIL |  |  |
| Test 2 | ⬜ PASS / ❌ FAIL |  |  |
| Test 3 | ⬜ PASS / ❌ FAIL |  |  |
| Test 4 | ⬜ PASS / ❌ FAIL |  |  |
| Test 5 | ⬜ PASS / ❌ FAIL |  |  |
| Test 6 | ⬜ PASS / ❌ FAIL |  |  |

**Overall Status**: ⬜ **READY FOR PRODUCTION** / ❌ **NEEDS FIXES**

---

**End of Test Checklist**
