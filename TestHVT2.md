# Profile-Based Hashtag Volume Test (TestHVT2)

**Test ID**: TestHVT2
**Created**: 2025-10-10
**Status**: Not Started
**Parent Document**: HashtagVolumeStrategy.md

---

## Executive Summary

**Hypothesis**: Profile scraping provides 0% overlap (proven in ScraperLimitations.md). Scraping top US creators in the supplement niche will yield higher unique video volume than hashtag scraping.

**Test Design**: Identify 10-20 top US supplement creators and scrape their profiles (50 videos each) to achieve 500-1000+ unique videos.

**Expected Outcome**: 0% overlap between creators, guaranteed unique videos, US-specific content through manual creator selection.

**Business Value**: Solve the hashtag volume problem by pivoting from hashtag scraping to profile-based scraping, leveraging proven 0% overlap characteristic.

---

## Background: Why This Test Matters

### Proven Foundation (ScraperLimitations.md Issue 3)

From previous testing:
- **Profile scraping has 0% overlap** between runs
- **Hashtag scraping has 82-97% overlap** (Test Suite 1: 5s, 2min, 30min delays)
- **Profile scraping is non-deterministic** - each scrape returns different videos from the same profile

### Hashtag Volume Problem

From HashtagVolumeStrategy.md:
- US filter reduces hashtag volume by **57%** (596 → 253 videos)
- Multiple hashtag scrapes provide **minimal gain** due to high overlap (74.9% duplicates across 6 scrapes)
- Popular hashtags (#fyp) provide **34% FEWER videos** than niche hashtags with US filter
- **NO clear winner** identified across all tested hashtag strategies

### The Pivot Strategy

**Key Insight**: If profile scraping has 0% overlap, scraping multiple US creator profiles should yield guaranteed unique videos without the overlap problem that plagues hashtag scraping.

---

## Test Configuration

### Core Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Apify Actor** | `clockworks/tiktok-scraper` | Standard TikTok profile scraper |
| **Target Type** | Profile (creator handles) | Proven 0% overlap between profiles |
| **Niche** | Supplement creators | Matches HashtagVolumeStrategy.md test niche |
| **Geographic Focus** | US creators | Manual US creator selection ensures US content quality |
| **Videos Per Creator** | 50 | Balance between volume and cost |
| **Creator Count** | 10-20 creators | Target 500-1000+ total unique videos |
| **Download Options** | All disabled | Metadata-only scraping (faster, cheaper) |

### Variable Parameters

| Parameter | Options | Purpose |
|-----------|---------|---------|
| **Creator List** | US supplement creators | Manual selection of proven US-based creators |
| **Hashtag Filter** | `#supplement` in video hashtags (optional) | Further refinement if needed |
| **Date Filter** | last_90_days (optional) | Client-side filtering for recency |

---

## Creator Selection Criteria

### How to Identify Top US Supplement Creators

**Method 1: Manual Research (Recommended)**
1. Search TikTok for `#supplement` hashtag
2. Identify creators with:
   - 100K+ followers
   - US-based (check bio, content language, location tags)
   - Consistent supplement content (not one-off posts)
   - High engagement (50K+ views per video)
3. Verify creator is active (posted in last 30 days)

**Method 2: Automated Discovery (Future Enhancement)**
1. Run hashtag scrape for `#supplement`
2. Extract author handles from top-performing videos
3. Rank by frequency and engagement
4. Manually verify US location

### Minimum Creator Requirements

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| Followers | 100K+ | Ensures established creator with content library |
| US-based | Verified in bio/content | Guarantees US content quality |
| Supplement niche | 80%+ supplement content | Ensures niche relevance |
| Recent activity | Posted in last 30 days | Active creators with fresh content |
| Avg views | 50K+ per video | High-quality, engaging content |

---

## Test Execution Plan

### Phase 1: Creator Discovery (Manual)

**Goal**: Identify 10-20 US supplement creators

**Steps**:
1. Search TikTok for `#supplement`, `#supplements`, `#supplementstack`
2. Open creator profiles for top-performing videos
3. Document creator handle, follower count, engagement, US verification
4. Aim for 15-20 creators (target: 750-1000 videos at 50/creator)

**Deliverable**: Creator list with handles and metrics

**Estimated Time**: 1-2 hours (manual research)

**Creator List Template**:
```
@creator1 | 250K followers | US-based ✅ | Avg 100K views | Active
@creator2 | 180K followers | US-based ✅ | Avg 75K views | Active
...
```

---

### Phase 2: Profile Scraping (Automated)

**Goal**: Scrape 50 videos from each creator

**Create test script** (`/tmp/profile_based_hvt_run.py`):

```python
#!/usr/bin/env python3
"""
Profile-Based Hashtag Volume Test
Scrapes multiple US supplement creator profiles to achieve high unique video volume
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

# Configuration
CREATORS = [
    "@creator1",
    "@creator2",
    "@creator3",
    # Add 10-20 creators here
]

VIDEOS_PER_CREATOR = 50
COUNTRY_CODE = "US"  # Optional: may help with content localization

# Results storage
all_videos = []
creator_results = {}

print(f"{'='*60}")
print(f"PROFILE-BASED HVT TEST")
print(f"{'='*60}")
print(f"Total Creators: {len(CREATORS)}")
print(f"Videos Per Creator: {VIDEOS_PER_CREATOR}")
print(f"Expected Total: ~{len(CREATORS) * VIDEOS_PER_CREATOR} videos")
print(f"{'='*60}\n")

# Scrape each creator
for i, creator in enumerate(CREATORS, 1):
    print(f"[{i}/{len(CREATORS)}] Scraping {creator}...")

    try:
        run = client.actor("clockworks/tiktok-scraper").call(
            run_input={
                'profiles': [creator],
                'resultsPerPage': VIDEOS_PER_CREATOR,
                'shouldDownloadCovers': False,
                'shouldDownloadVideos': False,
                'shouldDownloadSubtitles': False,
                'shouldDownloadSlideshowImages': False,
                'proxyCountryCode': COUNTRY_CODE
            }
        )

        # Fetch results
        items = list(client.dataset(run['defaultDatasetId']).iterate_items())

        creator_results[creator] = {
            'run_id': run['id'],
            'videos_scraped': len(items),
            'run_url': f"https://console.apify.com/view/runs/{run['id']}"
        }

        all_videos.extend(items)

        print(f"  ✅ {creator}: {len(items)} videos scraped")
        print(f"  Run ID: {run['id']}")

    except Exception as e:
        print(f"  ❌ {creator}: Failed - {str(e)}")
        creator_results[creator] = {
            'error': str(e),
            'videos_scraped': 0
        }

# Summary
print(f"\n{'='*60}")
print(f"SCRAPING COMPLETE")
print(f"{'='*60}")
print(f"Total Videos Scraped: {len(all_videos)}")
print(f"Successful Creators: {sum(1 for r in creator_results.values() if 'error' not in r)}/{len(CREATORS)}")
print(f"Failed Creators: {sum(1 for r in creator_results.values() if 'error' in r)}")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"/tmp/profile_based_hvt_{timestamp}.json"

with open(output_file, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'config': {
            'creators': CREATORS,
            'videos_per_creator': VIDEOS_PER_CREATOR,
            'country_code': COUNTRY_CODE
        },
        'creator_results': creator_results,
        'results': {
            'total_videos': len(all_videos),
            'videos': all_videos
        }
    }, f, indent=2)

print(f"\nResults saved to: {output_file}")
print(f"{'='*60}\n")
```

**Execute**:
```bash
# Edit /tmp/profile_based_hvt_run.py to add creator handles
python3 /tmp/profile_based_hvt_run.py
```

**Expected Runtime**: ~1-2 minutes per creator × 10-20 creators = 10-40 minutes total

**Expected Cost**: $1-1.50 per creator × 10-20 creators = $10-30 total

---

### Phase 3: Analysis

**Create analysis script** (`/tmp/analyze_profile_based_hvt.py`):

```python
#!/usr/bin/env python3
"""
Analyze Profile-Based HVT results
Validates 0% overlap hypothesis and assesses bucket distribution
"""

import json
import sys
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python3 analyze_profile_based_hvt.py <result_file.json>")
    sys.exit(1)

result_file = sys.argv[1]

# Load results
with open(result_file, 'r') as f:
    data = json.load(f)

videos = data['results']['videos']
creator_results = data['creator_results']
creators = data['config']['creators']

# Analyze video IDs for uniqueness
video_ids = [v['id'] for v in videos]
unique_ids = set(video_ids)
duplicate_count = len(video_ids) - len(unique_ids)

print(f"\n{'='*60}")
print(f"PROFILE-BASED HVT ANALYSIS")
print(f"{'='*60}")
print(f"Total Creators: {len(creators)}")
print(f"Successful Scrapes: {sum(1 for r in creator_results.values() if 'error' not in r)}")
print(f"Failed Scrapes: {sum(1 for r in creator_results.values() if 'error' in r)}")
print(f"\nVideo Statistics:")
print(f"  Total Videos Scraped: {len(videos)}")
print(f"  Unique Videos: {len(unique_ids)}")
print(f"  Duplicates: {duplicate_count} ({duplicate_count/len(videos)*100:.1f}%)")

# Per-creator breakdown
print(f"\nPer-Creator Results:")
for creator in creators:
    result = creator_results.get(creator, {})
    if 'error' in result:
        print(f"  {creator}: ❌ FAILED - {result['error']}")
    else:
        print(f"  {creator}: ✅ {result['videos_scraped']} videos")
        print(f"    Run URL: {result['run_url']}")

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
print(f"BUCKET DISTRIBUTION")
print(f"{'='*60}")
print(f"{'Bucket':<15} {'Videos':<10} {'Percentage':<12} {'Status'}")
print(f"{'-'*60}")

for bucket_name, video_list in sorted_buckets:
    count = len(video_list)
    pct = count / len(videos) * 100 if videos else 0

    # Status indicator
    if count >= 100:
        status = "✅ EXCELLENT"
    elif count >= 50:
        status = "⚠️  ADEQUATE"
    else:
        status = "❌ INSUFFICIENT"

    print(f"{bucket_name:<15} {count:<10} {pct:>5.1f}%       {status}")

print(f"\nTop 3 Winning Buckets:")
for i, (bucket_name, video_list) in enumerate(sorted_buckets[:3], 1):
    count = len(video_list)
    print(f"  {i}. {bucket_name}: {count} videos")

# Contrastive analysis readiness
insufficient_buckets = sum(1 for _, videos in sorted_buckets[:3] if len(videos) < 50)
if insufficient_buckets == 0:
    print(f"\n✅ CONTRASTIVE ANALYSIS READY: All top 3 buckets have 50+ videos")
else:
    print(f"\n❌ NOT READY: {insufficient_buckets} bucket(s) in top 3 have < 50 videos")

# Hashtag analysis (optional)
print(f"\n{'='*60}")
print(f"HASHTAG ANALYSIS")
print(f"{'='*60}")

supplement_hashtags = ['#supplement', '#supplements', '#supplementstack']
hashtag_matches = 0

for video in videos:
    video_hashtags = [tag.lower() for tag in video.get('hashtags', [])]
    if any(ht in video_hashtags for ht in supplement_hashtags):
        hashtag_matches += 1

print(f"Videos with supplement hashtags: {hashtag_matches} ({hashtag_matches/len(videos)*100:.1f}%)")

# Save video IDs for comparison
ids_file = result_file.replace('.json', '_ids.txt')
with open(ids_file, 'w') as f:
    f.write('\n'.join(unique_ids))

print(f"\nVideo IDs saved to: {ids_file}")
print(f"{'='*60}\n")
```

**Execute**:
```bash
python3 /tmp/analyze_profile_based_hvt.py /tmp/profile_based_hvt_20251010_120000.json
```

---

## Success Criteria

### Volume Metrics

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Total Unique Videos | 500-1000+ | 300+ |
| Videos Per Creator (Avg) | 50 | 30 |
| Duplicate Rate | 0% | < 5% |
| Top 3 Buckets (each) | 100+ videos | 50+ videos |

### Quality Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| US Content | 100% | Manual creator selection (US-based creators) |
| Supplement Niche | 80%+ | Check `#supplement` in video hashtags |
| Recent Content | 70%+ within 90 days | Date filter analysis |
| High Engagement | 50K+ avg views | Engagement metrics |

### Cost Efficiency

| Comparison | Profile-Based (TestHVT2) | Hashtag-Based (Baseline) |
|------------|-------------------------|-------------------------|
| Unique Videos | 500-1000+ (estimated) | 253 (proven) |
| Cost | $10-30 (10-20 creators) | $1.40 (single scrape) |
| Time | 10-40 minutes | 1.5 minutes |
| Overlap Rate | 0% (proven) | 82-97% (proven) |
| Cost per 100 unique videos | $1-3 | $5.53 ($1.40 × 100/253) |

**Key Insight**: While total cost is higher, cost per unique video is **50-80% LOWER** than hashtag scraping due to 0% overlap.

---

## Expected Outcomes

### Best Case Scenario

- 20 creators × 50 videos = **1,000 unique videos**
- 0% duplicate rate (proven for profile scraping)
- All top 3 buckets exceed 100 videos (contrastive ready)
- 100% US content quality (manual creator selection)
- 80%+ supplement-relevant content

**Result**: ✅✅ **CLEAR WINNER** - Solves hashtag volume problem definitively

### Realistic Scenario

- 15 creators × 45 videos avg = **675 unique videos**
- <5% duplicate rate (some creators may share collabs)
- Top 3 buckets: 150+, 100+, 80+ videos (contrastive ready)
- 100% US content quality
- 70%+ supplement-relevant content

**Result**: ✅ **VIABLE STRATEGY** - Significant improvement over hashtag scraping

### Worst Case Scenario

- 10 creators × 40 videos avg = **400 unique videos**
- 10% duplicate rate (cross-creator collaborations)
- Top 3 buckets: 80+, 60+, 50+ videos (barely contrastive ready)
- 90% US content quality (some creators may travel)
- 60%+ supplement-relevant content

**Result**: ⚠️ **MARGINAL** - Improvement over hashtag but not decisive

---

## Risk Analysis

### High Risks

**Risk 1: Creator Identification Overhead**
- **Impact**: High (blocks test execution)
- **Likelihood**: Medium
- **Mitigation**: Start with 10 creators minimum, expand if successful

**Risk 2: Profile Scraping Cost**
- **Impact**: Medium (cost overrun)
- **Likelihood**: Low (Apify pricing is transparent)
- **Mitigation**: Start with 5 creators as pilot, assess cost before scaling

### Medium Risks

**Risk 3: Creators Post Non-Supplement Content**
- **Impact**: Medium (dilutes niche relevance)
- **Likelihood**: Medium (fitness creators often diversify)
- **Mitigation**: Apply hashtag filter (`#supplement` in video tags)

**Risk 4: Bucket Distribution Skew**
- **Impact**: Medium (some buckets may still be insufficient)
- **Likelihood**: Low (larger sample size reduces skew)
- **Mitigation**: Target 750+ videos to ensure all buckets have sufficient volume

### Low Risks

**Risk 5: Duplicate Videos Across Creators**
- **Impact**: Low (reduces unique count)
- **Likelihood**: Low (profile scraping proven 0% overlap)
- **Mitigation**: Accept as minor issue, still far better than hashtag scraping

---

## Comparison to Other Options (HashtagVolumeStrategy.md)

| Strategy | Unique Videos | Cost | Time | Quality | Overlap Rate | Status |
|----------|---------------|------|------|---------|--------------|--------|
| **Option A** (Multiple Runs) | 316 (6 scrapes) | $8.40 | ~40 min | High ✅ | 82-97% | ❌ Not Viable |
| **Option B** (Extend Date) | 253 + 35-88 | $1.40 | 1.5 min | Medium ⚠️ | N/A | ✅ Adjunct only |
| **Option C** (Hybrid A+B) | 178 at 270d | $8.40 | ~40 min | Medium ⚠️ | 74.9% | ❌ Not Viable |
| **Option D** (Global + English) | 455 (~137 US-quality) | $1.40 | 1.5 min | Low ❌ | N/A | ⚠️ Questionable |
| **Option F** (Broader Hashtags) | 166 (#fyp) | $1.40 | 1.5 min | High ✅ | N/A | ❌ Not Viable |
| **TimeBasedHVT** | TBD | $4-6 | 10.5 hrs | High ✅ | TBD | ⏳ Testing |
| **TestHVT2** (Profile-Based) | **500-1000** | **$10-30** | **10-40 min** | **High ✅** | **0% (proven)** | **⏳ Proposed** |

**Key Advantage**: TestHVT2 is the **ONLY strategy** that leverages proven 0% overlap characteristic, guaranteeing unique videos at scale.

---

## Implementation Roadmap

### Step 1: Pilot Test (Immediate)

**Goal**: Validate approach with 5 creators

**Actions**:
1. Manually identify 5 top US supplement creators
2. Run profile scrapes (5 × 50 videos = 250 expected)
3. Analyze results (volume, duplicates, buckets, cost)
4. Calculate cost per unique video

**Duration**: 2-3 hours (1 hr research + 10 min scraping + analysis)
**Cost**: $5-7.50
**Decision Point**: If pilot yields 225+ unique videos with 0% duplicates → Proceed to full test

---

### Step 2: Full Test (If Pilot Succeeds)

**Goal**: Scale to 15-20 creators for 750-1000+ videos

**Actions**:
1. Identify additional 10-15 US supplement creators
2. Run profile scrapes (15-20 × 50 videos)
3. Comprehensive analysis (buckets, engagement, hashtags)
4. Document results in TestHVT2.md

**Duration**: 3-4 hours (2 hrs research + 30-40 min scraping + analysis)
**Cost**: $15-30
**Decision Point**: If full test yields 650+ unique videos → Recommend as primary strategy in HashtagVolumeStrategy.md

---

### Step 3: Integration (If Full Test Succeeds)

**Goal**: Integrate into ml_pipeline as alternative to hashtag scraping

**Actions**:
1. Create `creator_lists.json` for different niches
2. Update VideoDiscoveryCHILD.md to include profile-based discovery
3. Add CLI parameter: `--discovery-method {hashtag|profile}`
4. Update HashtagVolumeStrategy.md Decision Matrix

**Duration**: 1 day implementation
**Impact**: Solves hashtag volume problem permanently

---

## Next Steps

### Immediate Actions

1. ⏳ **Manual creator identification** (1-2 hours)
   - Search TikTok for `#supplement`, `#supplements`
   - Document 15-20 US-based creators with 100K+ followers
   - Verify active status and supplement niche focus

2. ⏳ **Create test scripts** (30 minutes)
   - `/tmp/profile_based_hvt_run.py` - Execution script
   - `/tmp/analyze_profile_based_hvt.py` - Analysis script

3. ⏳ **Run pilot test** (5 creators)
   - Execute scrapes
   - Analyze results
   - Calculate cost per unique video
   - Go/No-Go decision for full test

### Future Actions (If Pilot Succeeds)

4. ⏳ **Run full test** (15-20 creators)
5. ⏳ **Document results** in TestHVT2.md
6. ⏳ **Update HashtagVolumeStrategy.md** with new option
7. ⏳ **Recommend integration** into ml_pipeline

---

## Creator List Template

### US Supplement Creators (To Be Populated)

| # | Handle | Followers | US-Based | Avg Views | Supplement % | Status | Notes |
|---|--------|-----------|----------|-----------|--------------|--------|-------|
| 1 | @creator1 | - | ☐ | - | - | ⏳ | - |
| 2 | @creator2 | - | ☐ | - | - | ⏳ | - |
| 3 | @creator3 | - | ☐ | - | - | ⏳ | - |
| 4 | @creator4 | - | ☐ | - | - | ⏳ | - |
| 5 | @creator5 | - | ☐ | - | - | ⏳ | - |
| 6 | @creator6 | - | ☐ | - | - | ⏳ | - |
| 7 | @creator7 | - | ☐ | - | - | ⏳ | - |
| 8 | @creator8 | - | ☐ | - | - | ⏳ | - |
| 9 | @creator9 | - | ☐ | - | - | ⏳ | - |
| 10 | @creator10 | - | ☐ | - | - | ⏳ | - |
| 11 | @creator11 | - | ☐ | - | - | ⏳ | - |
| 12 | @creator12 | - | ☐ | - | - | ⏳ | - |
| 13 | @creator13 | - | ☐ | - | - | ⏳ | - |
| 14 | @creator14 | - | ☐ | - | - | ⏳ | - |
| 15 | @creator15 | - | ☐ | - | - | ⏳ | - |
| 16 | @creator16 | - | ☐ | - | - | ⏳ | - |
| 17 | @creator17 | - | ☐ | - | - | ⏳ | - |
| 18 | @creator18 | - | ☐ | - | - | ⏳ | - |
| 19 | @creator19 | - | ☐ | - | - | ⏳ | - |
| 20 | @creator20 | - | ☐ | - | - | ⏳ | - |

---

## Test Results

### Pilot Test (5 Creators)

**Status**: ⏳ Not Started

**Results**:
- Total Videos Scraped: -
- Unique Videos: -
- Duplicate Rate: -
- Top 3 Buckets: -
- Cost: -
- Cost per 100 Unique Videos: -

**Decision**: -

---

### Full Test (15-20 Creators)

**Status**: ⏳ Not Started

**Results**:
- Total Videos Scraped: -
- Unique Videos: -
- Duplicate Rate: -
- Top 3 Buckets: -
- Cost: -
- Cost per 100 Unique Videos: -

**Decision**: -

---

## Related Documents

- **HashtagVolumeStrategy.md**: Parent strategy document with all hashtag-based test results
- **ScraperLimitations.md**: Original documentation of 0% overlap for profile scraping (Issue 3)
- **VideoDiscoveryCHILD.md**: Stage 1 design document (may need updates if this strategy succeeds)
- **TimeBasedHVT.md**: Alternative test exploring time-based hashtag scraping

---

## Revision History

| Date | Change | Author |
|------|--------|--------|
| 2025-10-10 | Initial document created | Claude |

