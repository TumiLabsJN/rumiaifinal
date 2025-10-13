# Time-Based Hashtag Volume Test (TimeBasedHVT)

**Test ID**: TimeBasedHVT
**Created**: 2025-10-10
**Status**: In Progress
**Parent Document**: HashtagVolumeStrategy.md

---

## Executive Summary

**Hypothesis**: TikTok's algorithm returns different video sets based on time of day when scraping occurs.

**Test Design**: Run identical Apify scrapes at 4 different times throughout the day (São Paulo timezone) to measure time-sensitivity of results.

**Expected Outcome**: 10-30% variation in returned videos if algorithm is time-sensitive.

**Business Value**: If time-based variation exists, strategic scheduling could optimize video volume and quality for RumiAI ML training.

---

## Test Configuration

### Fixed Parameters (Constant Across All Runs)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Apify Actor** | `clockworks/tiktok-scraper` | Standard TikTok hashtag scraper |
| **Hashtag** | `#supplement` | Consistent with previous HashtagVolumeStrategy tests |
| **Country Code** | `US` | Geographic filter via proxy routing |
| **Results Requested** | 800 videos | Maximum practical limit for volume testing |
| **Download Options** | All disabled | Metadata-only scraping (faster, cheaper) |

### Variable Parameter (Changes Across Runs)

| Parameter | Values | Purpose |
|-----------|--------|---------|
| **Scrape Time** | 08:00, 11:00, 14:00, 18:30 (São Paulo, Brazil time) | Test time-of-day algorithm sensitivity |

**Timezone Note**: All times are **São Paulo, Brazil (BRT/BRST, UTC-3)**. Convert to other timezones as needed.

---

## Test Schedule

| Run | São Paulo Time | UTC Time | EST Time | Status | Run ID | Videos Scraped |
|-----|----------------|----------|----------|--------|--------|----------------|
| 1 | 08:00 | 11:00 | 06:00 | ⏳ Pending | - | - |
| 2 | 11:00 | 14:00 | 09:00 | ⏳ Pending | - | - |
| 3 | 14:00 | 17:00 | 12:00 | ⏳ Pending | - | - |
| 4 | 18:30 | 21:30 | 16:30 | ⏳ Pending | - | - |

**Schedule Status**: Not started

---

## Execution Instructions

### Prerequisites

1. **Apify Account**: Active account with API token set as `APIFY_API_TOKEN` environment variable in `/home/jorge/rumiaifinal/.env`
2. **Python Environment**: Python 3.8+ with `apify-client` installed
3. **Working Directory**: `/tmp/` for test scripts and results
4. **Internet Connection**: Required for Apify API calls

---

### Step 1: Run Apify Scrape

**Create test script** (`/tmp/timebased_hvt_run.py`):

```python
#!/usr/bin/env python3
"""
Time-Based Hashtag Volume Test - Single Run
Scrapes TikTok #supplement hashtag at scheduled time
"""

from apify_client import ApifyClient
import json
import os
from datetime import datetime
from pathlib import Path

# Load .env manually
env_file = Path('/home/jorge/rumiaifinal/.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

# Get API key
apify_api_key = os.getenv('APIFY_API_KEY')
if not apify_api_key:
    print("ERROR: APIFY_API_KEY not found in .env")
    exit(1)

# Initialize Apify client
client = ApifyClient(apify_api_key)

# Configuration (FIXED - DO NOT MODIFY)
HASHTAG = "#supplement"
COUNTRY_CODE = "US"
RESULTS_REQUESTED = 800

# Run scraper
print(f"[{datetime.now().isoformat()}] Starting Apify scrape...")
print(f"  Hashtag: {HASHTAG}")
print(f"  Country: {COUNTRY_CODE}")
print(f"  Requested: {RESULTS_REQUESTED} videos")

run = client.actor("clockworks/tiktok-scraper").call(
    run_input={
        'resultsPerPage': RESULTS_REQUESTED,
        'shouldDownloadCovers': False,
        'shouldDownloadVideos': False,
        'shouldDownloadSubtitles': False,
        'shouldDownloadSlideshowImages': False,
        'hashtags': [HASHTAG],
        'proxyCountryCode': COUNTRY_CODE
    }
)

# Fetch results
print(f"[{datetime.now().isoformat()}] Scrape completed. Fetching results...")
items = list(client.dataset(run['defaultDatasetId']).iterate_items())

# Save results
run_id = run['id']
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"/tmp/timebased_hvt_{timestamp}.json"

with open(output_file, 'w') as f:
    json.dump({
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'hashtag': HASHTAG,
            'country_code': COUNTRY_CODE,
            'results_requested': RESULTS_REQUESTED
        },
        'results': {
            'videos_scraped': len(items),
            'videos': items
        }
    }, f, indent=2)

print(f"\n{'='*60}")
print(f"RUN COMPLETE")
print(f"{'='*60}")
print(f"Run ID: {run_id}")
print(f"Run URL: https://console.apify.com/view/runs/{run_id}")
print(f"Videos Scraped: {len(items)}")
print(f"Output File: {output_file}")
print(f"{'='*60}\n")
```

**Execute at scheduled time**:
```bash
python3 /tmp/timebased_hvt_run.py
```

**Expected output**:
```
[2025-10-10T08:00:15] Starting Apify scrape...
  Hashtag: #supplement
  Country: US
  Requested: 800 videos
[2025-10-10T08:03:42] Scrape completed. Fetching results...

============================================================
RUN COMPLETE
============================================================
Run ID: AbCdEfGhIjKlMnOp
Run URL: https://console.apify.com/view/runs/AbCdEfGhIjKlMnOp
Videos Scraped: 253
Output File: /tmp/timebased_hvt_20251010_080342.json
============================================================
```

---

### Step 2: Analyze Results

**Create analysis script** (`/tmp/analyze_timebased_hvt.py`):

```python
#!/usr/bin/env python3
"""
Analyze Time-Based HVT results
Extracts key metrics: video count, video IDs, engagement stats, bucket distribution
"""

import json
import sys
from datetime import datetime

if len(sys.argv) < 2:
    print("Usage: python3 analyze_timebased_hvt.py <result_file.json>")
    sys.exit(1)

result_file = sys.argv[1]

# Load results
with open(result_file, 'r') as f:
    data = json.load(f)

videos = data['results']['videos']
run_id = data['run_id']
timestamp = data['timestamp']

# Extract video IDs and engagement metrics
video_ids = [v['id'] for v in videos]
views = [v.get('playCount', 0) for v in videos]
likes = [v.get('likeCount', 0) for v in videos]
shares = [v.get('shareCount', 0) for v in videos]

# Calculate statistics
avg_views = sum(views) / len(views) if views else 0
avg_likes = sum(likes) / len(likes) if likes else 0
avg_shares = sum(shares) / len(shares) if shares else 0

# Bucket distribution
buckets = {
    '0-3s': (0, 3),
    '3-9s': (3, 9),
    '9-13s': (9, 13),
    '13-18s': (13, 18),
    '18-33s': (18, 33),
    '33-60s': (33, 60),
    '60-90s': (60, 90),
    '90-120s': (90, 120)
}

bucket_dist = {name: [] for name in buckets.keys()}
for video in videos:
    duration = video.get('videoMeta', {}).get('duration', 0)
    for bucket_name, (min_dur, max_dur) in buckets.items():
        if min_dur <= duration < max_dur:
            bucket_dist[bucket_name].append(video['id'])
            break

# Sort buckets by count
sorted_buckets = sorted(bucket_dist.items(), key=lambda x: len(x[1]), reverse=True)

print(f"\n{'='*60}")
print(f"TIME-BASED HVT ANALYSIS")
print(f"{'='*60}")
print(f"Run ID: {run_id}")
print(f"Timestamp: {timestamp}")
print(f"Videos Scraped: {len(videos)}")
print(f"\nEngagement Averages:")
print(f"  Views:  {avg_views:,.0f}")
print(f"  Likes:  {avg_likes:,.0f}")
print(f"  Shares: {avg_shares:,.0f}")
print(f"\nTop 3 Buckets:")
for i, (bucket_name, video_list) in enumerate(sorted_buckets[:3], 1):
    print(f"  {i}. {bucket_name}: {len(video_list)} videos")
print(f"\nVideo ID Range:")
print(f"  First: {video_ids[0]}")
print(f"  Last:  {video_ids[-1]}")
print(f"{'='*60}\n")

# Save video IDs for overlap analysis
ids_file = result_file.replace('.json', '_ids.txt')
with open(ids_file, 'w') as f:
    f.write('\n'.join(video_ids))

print(f"Video IDs saved to: {ids_file}")
```

**Execute**:
```bash
python3 /tmp/analyze_timebased_hvt.py /tmp/timebased_hvt_20251010_080342.json
```

---

### Step 3: Document Results in TimeBasedHVT.md

After each run:

1. **Update Test Schedule table** with:
   - Status: ✅ Complete
   - Run ID from Apify output
   - Video count from analysis

2. **Fill in corresponding "Run N" results section** (see Results Template below)

3. **Save the JSON and IDs files** - DO NOT DELETE until all 4 runs complete

---

### Step 4: Run Overlap Analysis (After All 4 Runs Complete)

**Create overlap analysis script** (`/tmp/overlap_timebased_hvt.py`):

```python
#!/usr/bin/env python3
"""
Calculate video overlap across all Time-Based HVT runs
Determines how many videos are shared vs unique across time slots
"""

import sys

if len(sys.argv) < 5:
    print("Usage: python3 overlap_timebased_hvt.py <run1_ids.txt> <run2_ids.txt> <run3_ids.txt> <run4_ids.txt>")
    sys.exit(1)

# Load video ID sets
sets = []
labels = ['Run 1 (08:00)', 'Run 2 (11:00)', 'Run 3 (14:00)', 'Run 4 (18:30)']

for i, file in enumerate(sys.argv[1:5]):
    with open(file, 'r') as f:
        video_ids = set(line.strip() for line in f if line.strip())
        sets.append(video_ids)
        print(f"{labels[i]}: {len(video_ids)} videos")

# Calculate overlaps
all_videos = sets[0].union(*sets[1:])
shared_all = sets[0].intersection(*sets[1:])

print(f"\n{'='*60}")
print(f"OVERLAP ANALYSIS")
print(f"{'='*60}")
print(f"Total Unique Videos: {len(all_videos)}")
print(f"Shared Across All 4 Runs: {len(shared_all)} ({len(shared_all)/len(sets[0])*100:.1f}%)")

# Pairwise overlaps
print(f"\nPairwise Overlaps:")
for i in range(len(sets)):
    for j in range(i+1, len(sets)):
        overlap = len(sets[i].intersection(sets[j]))
        pct = overlap / len(sets[i]) * 100
        print(f"  {labels[i]} vs {labels[j]}: {overlap} videos ({pct:.1f}%)")

# Calculate average overlap
all_overlaps = []
for i in range(len(sets)):
    for j in range(i+1, len(sets)):
        overlap = len(sets[i].intersection(sets[j]))
        pct = overlap / len(sets[i]) * 100
        all_overlaps.append(pct)

avg_overlap = sum(all_overlaps) / len(all_overlaps)
print(f"\nAverage Pairwise Overlap: {avg_overlap:.1f}%")

# Viability assessment
print(f"\n{'='*60}")
print(f"VIABILITY ASSESSMENT")
print(f"{'='*60}")
if avg_overlap < 20:
    print("✅ VIABLE: Low overlap (<20%) - Time-based scraping provides significant variation")
    print(f"   Expected unique videos from 4 runs: ~{len(all_videos)} videos")
elif avg_overlap < 50:
    print("⚠️  MARGINAL: Moderate overlap (20-50%) - Time-based scraping provides some variation")
    print(f"   Expected unique videos from 4 runs: ~{len(all_videos)} videos")
else:
    print("❌ NOT VIABLE: High overlap (>50%) - Time-based scraping does NOT provide meaningful variation")
    print(f"   Expected unique videos from 4 runs: only ~{len(all_videos)} videos")

print(f"{'='*60}\n")
```

**Execute after all 4 runs**:
```bash
python3 /tmp/overlap_timebased_hvt.py \
  /tmp/timebased_hvt_20251010_080342_ids.txt \
  /tmp/timebased_hvt_20251010_110215_ids.txt \
  /tmp/timebased_hvt_20251010_140428_ids.txt \
  /tmp/timebased_hvt_20251010_183552_ids.txt
```

---

## Results Template

### Run 1: 08:00 São Paulo Time

**Status**: ⏳ Pending

**Details**:
- Run ID: -
- Run URL: -
- Timestamp: -
- Videos Scraped: -
- Top 3 Buckets: -
- Engagement Averages: -

---

### Run 2: 11:00 São Paulo Time

**Status**: ⏳ Pending

**Details**:
- Run ID: -
- Run URL: -
- Timestamp: -
- Videos Scraped: -
- Top 3 Buckets: -
- Engagement Averages: -

---

### Run 3: 14:00 São Paulo Time

**Status**: ⏳ Pending

**Details**:
- Run ID: -
- Run URL: -
- Timestamp: -
- Videos Scraped: -
- Top 3 Buckets: -
- Engagement Averages: -

---

### Run 4: 18:30 São Paulo Time

**Status**: ⏳ Pending

**Details**:
- Run ID: -
- Run URL: -
- Timestamp: -
- Videos Scraped: -
- Top 3 Buckets: -
- Engagement Averages: -

---

## Overlap Analysis

**Status**: ⏳ Awaiting all 4 runs to complete

**Results**:
- Total Unique Videos: -
- Shared Across All 4 Runs: - (-%)
- Average Pairwise Overlap: -
- Pairwise Details: -

---

## Conclusions

**Status**: ⏳ Analysis pending

**Findings**: To be documented after all runs complete

**Recommendation**: To be determined based on overlap analysis

**Viability for HashtagVolumeStrategy.md**:
- ✅ **VIABLE** if avg overlap < 20%
- ⚠️ **MARGINAL** if avg overlap 20-50%
- ❌ **NOT VIABLE** if avg overlap > 50%

---

## Next Steps for Subsequent Runs

**When you see: "revise TimeBasedHVT.md it's time for another run"**

### Do this:

1. **Check Test Schedule table** - Find first row with ⏳ Pending status
2. **Note the scheduled time** (e.g., "Run 2: 11:00 São Paulo")
3. **Execute scrape**: `python3 /tmp/timebased_hvt_run.py`
4. **Execute analysis**: `python3 /tmp/analyze_timebased_hvt.py /tmp/timebased_hvt_<timestamp>.json`
5. **Update Test Schedule table**:
   - Change status to ✅ Complete
   - Add Run ID
   - Add Videos Scraped count
6. **Update corresponding "Run N" section** with:
   - Run ID
   - Run URL
   - Timestamp
   - Videos Scraped
   - Top 3 Buckets (from analysis output)
   - Engagement Averages (from analysis output)
7. **If this was Run 4**:
   - Execute overlap analysis: `python3 /tmp/overlap_timebased_hvt.py <ids1> <ids2> <ids3> <ids4>`
   - Document findings in "Overlap Analysis" section
   - Document conclusions
   - Update HashtagVolumeStrategy.md with test results
   - Add as new option if viable
8. **Save and commit changes**

### Files to Preserve

**Keep these files until all 4 runs complete and overlap analysis is done**:
- `/tmp/timebased_hvt_*.json` - All run results (4 files)
- `/tmp/timebased_hvt_*_ids.txt` - All video ID lists (4 files)
- `/tmp/timebased_hvt_run.py` - Test execution script
- `/tmp/analyze_timebased_hvt.py` - Analysis script
- `/tmp/overlap_timebased_hvt.py` - Overlap analysis script

---

## Comparison to Previous Tests

### Baseline Reference (HashtagVolumeStrategy.md)

| Test | Hashtag | Country | Videos | Overlap | Status |
|------|---------|---------|--------|---------|--------|
| Test 1 (5s delay) | #supplement | US | 193-228 | 96.9% | ❌ Not viable |
| Test 1.A (2min delay) | #supplement | US | 208-213 | 86.9% | ❌ Not viable |
| Test 1.B (30min delay) | #supplement | US | 206-210 | 82.4% | ❌ Not viable |
| **TimeBasedHVT** | **#supplement** | **US** | **TBD** | **TBD** | **⏳ Testing** |

**Key Difference**: Previous tests used short delays (5s-30min). TimeBasedHVT uses longer intervals (3-10.5 hours) to test if TikTok's algorithm refreshes content at different times of day.

**Hypothesis**: Longer time intervals may reduce overlap if TikTok surfaces different "trending" content during peak vs off-peak hours.

---

## Appendix: Cost & Time Estimates

**Per-Run Cost**: $1-1.50 (Apify hashtag scrape)
**Total Test Cost**: $4-6 for 4 runs
**Per-Run Duration**: ~3-5 minutes
**Total Test Duration**: 10.5 hours (08:00 to 18:30 São Paulo time)
**Spread**: Full business day coverage (early morning to evening)

---

## Related Documents

- **HashtagVolumeStrategy.md**: Parent strategy document with all previous test results
- **30minwait.md**: Test 1.B protocol (30-minute delay test)
- **VideoDiscoveryCHILD.md**: Stage 1 design document
- **ScraperLimitations.md**: Apify scraper limitations analysis

---

## Revision History

| Date | Run Completed | Changes Made | Updated By |
|------|---------------|--------------|------------|
| 2025-10-10 | Initial | Document created | Claude |

