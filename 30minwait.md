# 30-Minute Delay Test: Non-Determinism Validation

**Status**: ⏳ Waiting for 30-minute delay to complete
**Date Created**: 2025-10-09
**Purpose**: Test if 30-minute delay between Apify scrapes reduces video overlap
**Cost**: $2.80 (2 scrapes × $1.40 each)

---

## 🎯 Context: Why This Test Exists

### Problem Statement
US geographic filtering reduces hashtag video volume by 57% (596 → 253 videos), making it difficult to achieve sufficient sample sizes for ML contrastive analysis (need 50-100+ videos per duration bucket).

### Initial Hypothesis (Test 1)
Multiple sequential scrapes might return different videos, allowing us to multiply volume (similar to profile scraping which showed 0% overlap).

### Test Results So Far

| Test | Delay | Hashtag | Run 1 | Run 2 | Overlap % | Gain | Unique Total | Conclusion |
|------|-------|---------|-------|-------|-----------|------|--------------|------------|
| **Test 1** | 5 seconds | #supplement | 193 | 228 | **96.9%** | 41 (21.2%) | 234 | ❌ Not viable |
| **Test 1.A** | 2 minutes | #supplement | 213 | 208 | **86.9%** | 23 (10.8%) | 236 | ❌ Not viable |
| **Test 1.B** | **30 minutes** | #supplement | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **⏳ In Progress** |

### Key Insight from Previous Tests
- Profile scraping: 0% overlap (completely non-deterministic)
- Hashtag scraping (5s delay): 96.9% overlap
- Hashtag scraping (2min delay): 86.9% overlap (10% improvement, but still too high)

### New Hypothesis (Test 1.B)
TikTok's "For You" algorithm may refresh significantly over longer time periods (30 minutes). If overlap drops below 50%, multiple scrapes with 30-minute delays become viable for volume multiplication.

---

## 📊 Test 1.B: First Scrape (COMPLETED)

### Run Details
- **Run ID**: `6h2zNstUqmXNZ8daq`
- **Run URL**: `https://console.apify.com/view/runs/6h2zNstUqmXNZ8daq`
- **Videos Scraped**: `210`
- **Recent (90d)**: `49 (23.3%)`
- **Timestamp**: `2025-10-09 14:08:53 UTC (2:08 PM)`

### Command Used
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
python /tmp/test_hashtag_no_date.py
```

### Configuration
```python
# From /tmp/test_hashtag_no_date.py
{
    'resultsPerPage': 800,
    'shouldDownloadCovers': False,
    'shouldDownloadVideos': False,
    'shouldDownloadSubtitles': False,
    'shouldDownloadSlideshowImages': False,
    'hashtags': ['#supplement'],
    'proxyCountryCode': 'US'
    # No date filters (date filtering doesn't work for hashtags)
}
```

---

## ⏱️ WAIT 30 MINUTES

**Wait Start Time**: `2025-10-09 14:08:53 UTC (2:08 PM)`
**Next Scrape Time**: `2025-10-09 14:38:53 UTC (2:38 PM)`

### During the Wait
- ✅ Run Test Suite 2 (Date Distribution Analysis) - Free, 5 min
- ✅ Run Test Suite 3 (Language Distribution Analysis) - Free, 5 min
- ✅ Update documentation
- ✅ Other tasks

---

## 📊 Test 1.B: Second Scrape (PENDING - Run After 30 Minutes)

### When to Run
After 30 minutes have elapsed from the first scrape completion.

### Command to Run
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
python /tmp/test_hashtag_no_date.py
```

### Expected Output
- **Run ID**: `TBD_RUN6_ID` ← **Update this after second scrape!**
- **Run URL**: `https://console.apify.com/view/runs/TBD_RUN6_ID`
- **Videos Scraped**: `TBD`
- **Recent (90d)**: `TBD`
- **Timestamp**: `TBD`

---

## 📈 Analysis: Overlap Calculation (Run After Second Scrape)

### Step 1: Update Overlap Analysis Script

Create `/tmp/analyze_overlap_30min.py` with the following content:

```python
#!/usr/bin/env python3
"""Analyze overlap between two Apify scrape runs (30-minute delay test)."""
import os
import sys
from pathlib import Path

# Load .env
env_file = Path('/home/jorge/rumiaifinal/.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

from apify_client import ApifyClient

# Get API key
apify_api_key = os.getenv('APIFY_API_KEY')
if not apify_api_key:
    print("ERROR: APIFY_API_KEY not found")
    sys.exit(1)

# ⚠️ IMPORTANT: UPDATE THESE RUN IDS AFTER BOTH SCRAPES COMPLETE
run5_id = '6h2zNstUqmXNZ8daq'  # ✅ UPDATED - First scrape (210 videos)
run6_id = 'TBD_RUN6_ID'  # ← UPDATE THIS with second scrape Run ID

print('='*80)
print('TEST 1.B: 30-MINUTE DELAY VALIDATION')
print('='*80)
print(f'Run 5 (first):  {run5_id}')
print(f'Run 6 (30min later): {run6_id}')
print()

# Initialize client
client = ApifyClient(apify_api_key)

# Fetch datasets
print('Fetching Run 5 dataset...')
run5_info = client.run(run5_id).get()
run5_dataset_id = run5_info['defaultDatasetId']
run5_videos = client.dataset(run5_dataset_id).list_items().items

print('Fetching Run 6 dataset...')
run6_info = client.run(run6_id).get()
run6_dataset_id = run6_info['defaultDatasetId']
run6_videos = client.dataset(run6_dataset_id).list_items().items

print()
print(f'Run 5 videos: {len(run5_videos)}')
print(f'Run 6 videos: {len(run6_videos)}')
print()

# Extract video IDs
run5_ids = set(v['id'] for v in run5_videos if 'id' in v)
run6_ids = set(v['id'] for v in run6_videos if 'id' in v)

# Calculate overlap
overlap = run5_ids & run6_ids  # Intersection
union = run5_ids | run6_ids     # Union
run5_only = run5_ids - run6_ids
run6_only = run6_ids - run5_ids

# Calculate percentages
overlap_count = len(overlap)
overlap_pct_run5 = (overlap_count / len(run5_ids)) * 100 if run5_ids else 0
overlap_pct_run6 = (overlap_count / len(run6_ids)) * 100 if run6_ids else 0
unique_total = len(union)

print('='*80)
print('OVERLAP ANALYSIS (30-MINUTE DELAY)')
print('='*80)
print(f'Videos in both runs: {overlap_count}')
print(f'  Overlap (% of Run 5): {overlap_pct_run5:.1f}%')
print(f'  Overlap (% of Run 6): {overlap_pct_run6:.1f}%')
print()
print(f'Unique to Run 5: {len(run5_only)} videos')
print(f'Unique to Run 6: {len(run6_only)} videos')
print()
print(f'Total unique videos: {unique_total}')
print(f'Gain from 2nd scrape: {len(run6_only)} videos ({len(run6_only)/len(run5_ids)*100:.1f}% increase)')
print()

# Compare to all previous tests
print('='*80)
print('COMPARISON: ALL DELAY TESTS')
print('='*80)
print('Test 1 (5-second delay):')
print('  Run 1: 193 videos')
print('  Run 2: 228 videos')
print('  Overlap: 96.9%')
print('  Unique total: 234 videos')
print('  Gain: 41 videos (21.2%)')
print()
print('Test 1.A (2-minute delay):')
print('  Run 3: 213 videos')
print('  Run 4: 208 videos')
print('  Overlap: 86.9%')
print('  Unique total: 236 videos')
print('  Gain: 23 videos (10.8%)')
print()
print('Test 1.B (30-minute delay):')
print(f'  Run 5: {len(run5_videos)} videos')
print(f'  Run 6: {len(run6_videos)} videos')
print(f'  Overlap: {overlap_pct_run5:.1f}%')
print(f'  Unique total: {unique_total} videos')
print(f'  Gain: {len(run6_only)} videos ({len(run6_only)/len(run5_ids)*100:.1f}%)')
print()

# Calculate improvement trend
test1_overlap = 96.9
test1a_overlap = 86.9
improvement_5s_to_2min = test1_overlap - test1a_overlap
improvement_2min_to_30min = test1a_overlap - overlap_pct_run5

print('='*80)
print('TREND ANALYSIS')
print('='*80)
print(f'Improvement from 5s → 2min delay: {improvement_5s_to_2min:.1f}% reduction in overlap')
print(f'Improvement from 2min → 30min delay: {improvement_2min_to_30min:.1f}% reduction in overlap')
print()

# Decision criteria
print('='*80)
print('DECISION CRITERIA')
print('='*80)
if overlap_pct_run5 < 20:
    print('✅ RESULT: < 20% overlap → 30-minute delay WORKS!')
    print('✅ RECOMMENDATION: Use 30-minute delays between scrapes (Option A/C viable)')
    print()
    print('Expected outcomes with 30-min delays:')
    print(f'  3 runs: ~{unique_total * 1.8:.0f} videos (estimated)')
    print(f'  5 runs: ~{unique_total * 2.5:.0f} videos (estimated)')
    print()
    print('⚠️  TRADE-OFF: Time cost is significant (2.5 hours for 3 runs)')
elif overlap_pct_run5 < 50:
    print('⚠️  RESULT: 20-50% overlap → Meaningful improvement, borderline viable')
    print('⚠️  RECOMMENDATION: 30-minute delays help significantly')
    print()
    print('Expected outcomes with 30-min delays:')
    print(f'  3 runs: ~{unique_total * 1.5:.0f} videos (estimated)')
    print(f'  5 runs: ~{unique_total * 2:.0f} videos (estimated)')
    print()
    print('⚠️  TRADE-OFF: Time cost vs volume gain needs evaluation')
elif overlap_pct_run5 < 70:
    print('⚠️  RESULT: 50-70% overlap → Some improvement, but diminishing returns')
    print('❌ RECOMMENDATION: 30-minute delays help but not enough for practical use')
    print('   Consider Option D (Language Filter) or Option E (Adaptive Strategy)')
else:
    print('❌ RESULT: > 70% overlap → Minimal improvement from longer delay')
    print('❌ RECOMMENDATION: Time delay strategy NOT VIABLE')
    print('   Apify/TikTok returns similar video pool regardless of delay duration')
    print('   Recommend Option D (Language Filter) or Option E (Adaptive Strategy)')

print()
print('='*80)
print('TEST 1.B COMPLETE')
print('='*80)
```

### Step 2: Run Overlap Analysis

```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate

# ⚠️ IMPORTANT: First update the run IDs in the script above!
python /tmp/analyze_overlap_30min.py
```

---

## 🎯 Success Criteria

| Overlap % | Outcome | Next Action |
|-----------|---------|-------------|
| **< 20%** | ✅ **SUCCESS** - Non-determinism confirmed | Implement Option C (Hybrid) with 30-min delays |
| **20-50%** | ⚠️ **PARTIAL** - Meaningful improvement | Evaluate time/cost trade-off |
| **50-70%** | ⚠️ **MARGINAL** - Some improvement | Consider Option D (Language Filter) |
| **> 70%** | ❌ **FAILURE** - Minimal improvement | Rule out time delay strategy entirely |

---

## 📝 Related Documents

- **HashtagVolumeStrategy.md**: Main strategy document with all options
- **ScraperLimitations.md**: Original scraper limitation analysis
- **Test Scripts**:
  - `/tmp/test_hashtag_no_date.py` - Scraping script (US filter, no date filter)
  - `/tmp/analyze_overlap.py` - Test 1 analysis (5-second delay)
  - `/tmp/analyze_overlap_2min.py` - Test 1.A analysis (2-minute delay)
  - `/tmp/analyze_overlap_30min.py` - Test 1.B analysis (30-minute delay)

---

## 📋 Baseline Test Data (For Reference)

### Test 1: 5-Second Delay
- **Run 1 ID**: GYImoRDaLGVRYwjUP
- **Run 2 ID**: UJeE95ICb9z1OXH5b
- **Run 1 URL**: https://console.apify.com/view/runs/GYImoRDaLGVRYwjUP
- **Run 2 URL**: https://console.apify.com/view/runs/UJeE95ICb9z1OXH5b
- **Result**: 193 videos, 228 videos, 96.9% overlap

### Test 1.A: 2-Minute Delay
- **Run 3 ID**: 750ocm0oU7BgIk02I
- **Run 4 ID**: ugbYNrcI5mSrYy4qe
- **Run 3 URL**: https://console.apify.com/view/runs/750ocm0oU7BgIk02I
- **Run 4 URL**: https://console.apify.com/view/runs/ugbYNrcI5mSrYy4qe
- **Result**: 213 videos, 208 videos, 86.9% overlap

### Test 1.B: 30-Minute Delay (CURRENT TEST)
- **Run 5 ID**: `6h2zNstUqmXNZ8daq` ✅
- **Run 6 ID**: `TBD_RUN6_ID` ← **UPDATE THIS!**
- **Run 5 URL**: https://console.apify.com/view/runs/6h2zNstUqmXNZ8daq
- **Run 6 URL**: `TBD`
- **Result**: 210 videos (Run 5), awaiting Run 6

---

## 🚀 Quick Start Instructions (For Fresh CLI Instance)

If you're a new CLI instance or lost context, follow these steps:

### Step 1: Understand the Context
Read the "Context: Why This Test Exists" section above.

### Step 2: Check Current Status
Look at the "Test 1.B: First Scrape" section. If Run IDs are marked `TBD`, you need to run the first scrape.

### Step 3A: If First Scrape NOT Done
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
python /tmp/test_hashtag_no_date.py
```

After completion:
1. Update "Test 1.B: First Scrape" section with Run ID and video count
2. Update "Wait Start Time"
3. Wait 30 minutes

### Step 3B: If 30 Minutes Have Passed
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
python /tmp/test_hashtag_no_date.py
```

After completion:
1. Update "Test 1.B: Second Scrape" section with Run ID and video count
2. Proceed to Step 4

### Step 4: Run Overlap Analysis
```bash
# First, update the run IDs in /tmp/analyze_overlap_30min.py
# Then run:
cd /home/jorge/rumiaifinal
source venv/bin/activate
python /tmp/analyze_overlap_30min.py
```

### Step 5: Document Results
Update the "Baseline Test Data" section with Test 1.B results.

---

## 💰 Cost Tracking

| Test | Scrapes | Cost | Total Spent |
|------|---------|------|-------------|
| Test 1 (5s delay) | 2 | $2.80 | $2.80 |
| Test 1.A (2min delay) | 2 | $2.80 | $5.60 |
| Test 1.B (30min delay) | 2 | $2.80 | **$8.40** |

---

## 🔄 Version History

| Date | Change | Author |
|------|--------|--------|
| 2025-10-09 | Initial document created | Claude |
| TBD | First scrape completed, Run ID added | TBD |
| TBD | Second scrape completed, overlap analysis done | TBD |
