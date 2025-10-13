# Hashtag Volume Strategy: Solving the Geographic Filtering Trade-off

**Status**: 🧪 Testing Phase
**Last Updated**: October 9, 2025
**Affects**: Stage 1 - Video Discovery & Selection (Hashtag Analysis)

---

## 🎯 Problem Statement

### Core Issue
US geographic filtering reduces hashtag video volume by **57%** (596 → 253 videos), making it difficult to achieve sufficient sample sizes for contrastive analysis.

### Business Context
- **Geographic filtering is critical**: US-specific content is essential for market relevance
- **Contrastive analysis requires volume**: Need 50-100+ videos per bucket for statistical validity
- **Trade-off**: Quality (US-specific) vs. Quantity (enough for analysis)

### Current State (Baseline)

**Test Results Summary**:

| Test | Date Filter | Geographic Filter | Videos Scraped | Recent (90d) | Achievement |
|------|-------------|-------------------|----------------|--------------|-------------|
| **Test 1** | ✅ YES (ignored) | ✅ US | 204 | 49 (24%) | ❌ Insufficient |
| **Test 2** | ❌ NO | ✅ US | 253 | 49 (19.4%) | ⚠️ Marginal |
| **Test 3** | ❌ NO | ❌ GLOBAL | **596** | 163 (27.3%) | ✅ Sufficient |

**Run URLs**:
- Test 1: https://console.apify.com/view/runs/Ec6VqhOOQ8REZTywu
- Test 2: https://console.apify.com/view/runs/NXMbGqnMcFliMWh1w
- Test 3: https://console.apify.com/view/runs/cWDDMH2RE5GaGpb3r

**Key Findings**:
1. ✅ **Date filtering does NOT work for hashtags** (Test 1 vs Test 2 identical)
2. ✅ **Geographic filtering works but reduces volume by 57%** (Test 2 vs Test 3)
3. ❌ **US filtering yields insufficient volume** for robust contrastive analysis

### Contrastive Analysis Requirements

**Minimum Sample Sizes**:
- **Per bucket**: 50+ videos (recommended: 100+)
- **Split**: 80% top performers + 20% bottom performers
- **Example**: 100 videos/bucket = 80 top + 20 bottom

**Current Gap**:
- 253 videos across 8 potential buckets = ~32 videos/bucket average
- After winner selection (3 buckets): ~84 videos/bucket
- **Problem**: Not evenly distributed; some buckets < 50 videos

---

## 💡 Strategic Options

### Option A: Multiple Sequential Scrapes (Same Config)

**Hypothesis**: Non-deterministic scraping returns different videos per run

**Implementation**:
```python
# Run 3-5 sequential scrapes with same parameters
for run_num in range(3):
    videos = scraper.scrape_videos(
        hashtags=["#supplement"],
        proxyCountryCode="US",
        resultsPerPage=800
    )
    all_videos.extend(videos)
    time.sleep(5)  # Avoid rate limiting

unique_videos = deduplicate(all_videos)
```

**Expected Outcome** (IF 0% overlap):
- 1 run: 253 videos
- 3 runs: 759 videos (3× volume)
- 5 runs: 1,265 videos (5× volume)

**Pros**:
- ✅ We know scraping is non-deterministic (ScraperLimitations.md Issue 3)
- ✅ Two profile scrapes showed **0% overlap**
- ✅ Simple to implement (code already exists)
- ✅ Maintains US quality

**Cons**:
- ⚠️ **UNVERIFIED**: Don't know if hashtag scraping is as non-deterministic as profile scraping
- ❌ 3-5× cost increase (~$4.20 for 3 runs)
- ❌ 3-5× runtime increase (~4.5 min for 3 runs)

**Dependencies**:
- 🧪 **Test Suite 1 REQUIRED**: Validate non-determinism for hashtags

**Risk Level**: Medium (depends on test results)

---

### Option B: Extend Date Range (Client-Side Filtering)

**Hypothesis**: Accept older videos to increase available pool

**Implementation**:
```python
# Client-side date filtering (Apify ignores date params for hashtags)
date_filter = "last_150_days"  # Instead of last_90_days
```

**Expected Outcome**:
- Apify still returns ~253 videos (Apify decides what to return)
- More of those videos fall within 150-day window
- **Minimal volume gain expected**

**Pros**:
- ✅ Single scrape (no cost/time multiplier)
- ✅ Captures seasonal trends (5-month window)
- ✅ US filtering maintained

**Cons**:
- ❌ Apify date filtering doesn't work for hashtags (proven in Test 1)
- ⚠️ **Unlikely to help**: Apify returns same videos regardless of client-side filter
- ❌ Older videos (150 days) may have outdated creative trends
- ⚠️ Need to verify Apify returns videos in 91-150 day range

**Dependencies**:
- 🧪 **Test Suite 2 REQUIRED**: Analyze age distribution of scraped videos

**Risk Level**: Low (easy to test, low cost)

---

### Option C: Hybrid - Multiple Runs + Extended Date Window

**Hypothesis**: Combine approaches for maximum coverage while maintaining quality

**Implementation**:
```python
# Run 3 sequential scrapes with US filter
for run_num in range(3):
    videos = scraper.scrape_videos(
        hashtags=["#supplement"],
        proxyCountryCode="US",
        resultsPerPage=800
    )
    all_videos.extend(videos)
    time.sleep(5)

# Deduplicate
unique_videos = deduplicate(all_videos)

# Client-side date filter (extended window)
recent_videos = date_filter(unique_videos, days=150)
```

**Expected Outcome** (IF 0% overlap):
- 3 runs × 253 videos = 759 unique videos
- Accept videos from last 150 days
- **Best case**: 700+ videos for analysis

**Pros**:
- ✅ Maximizes volume (IF non-deterministic)
- ✅ Maintains US quality
- ✅ Extended window provides safety margin

**Cons**:
- ❌ 3× cost (~$4.20)
- ❌ 3× runtime (~4.5 min)
- ⚠️ Still depends on non-determinism hypothesis

**Dependencies**:
- 🧪 **Test Suite 1 REQUIRED**: Validate non-determinism
- 🧪 **Test Suite 2 REQUIRED**: Validate date distribution

**Risk Level**: Medium (depends on Test Suite 1 results)

---

### Option D: Global Scraping + Language Filter

**Hypothesis**: Scrape globally, filter for English language as proxy for US market

**Implementation**:
```python
# Scrape without geographic filter
videos = scraper.scrape_videos(
    hashtags=["#supplement"],
    resultsPerPage=800
    # No proxyCountryCode
)

# Client-side language filter
english_videos = [v for v in videos if v.get('textLanguage') == 'en']
```

**Expected Outcome**:
- 596 videos scraped (global)
- English % varies by hashtag (need to measure)
- **Best case**: 596 × 0.6 = ~358 videos (if 60% English)
- **Worst case**: 596 × 0.4 = ~238 videos (if 40% English)

**Pros**:
- ✅ 2.35× more videos scraped (596 vs 253)
- ✅ Single scrape (no cost multiplier)
- ✅ English language = decent proxy for US/Western market

**Cons**:
- ❌ Language ≠ Geography (English spoken globally: UK, Australia, India, etc.)
- ⚠️ Lower quality (mixed regions, different cultural contexts)
- ⚠️ Unknown English % for specific hashtags (need to measure)
- ❌ May not gain much volume vs US filter (depends on English %)

**Dependencies**:
- 🧪 **Test Suite 3 REQUIRED**: Measure English language % in global scraping

**Risk Level**: Low (easy to test, but lower quality)

---

### Option E: Adaptive Strategy (Fallback)

**Hypothesis**: Use different analysis strategies based on available volume

**Implementation**:
```python
def select_strategy(videos_per_bucket):
    """Choose analysis strategy based on volume."""
    if videos_per_bucket >= 100:
        return "contrastive"  # 80% top + 20% bottom
    elif videos_per_bucket >= 50:
        return "contrastive_lite"  # 90% top + 10% bottom (fewer bottom performers)
    else:
        return "top"  # Top performers only
```

**Expected Outcome**:
- Low volume → "top-only" strategy
- Medium volume → "contrastive-lite" strategy
- High volume → Full contrastive analysis

**Pros**:
- ✅ Adaptive to data availability
- ✅ Still generates insights (top-only analysis valid)
- ✅ No cost/runtime increase
- ✅ Graceful degradation

**Cons**:
- ❌ Doesn't solve volume problem, just works around it
- ❌ Top-only analysis less powerful (no bottom performers for contrast)
- ⚠️ May not meet client expectations if contrastive promised

**Dependencies**:
- None (can implement immediately)

**Risk Level**: None (fallback option)

---

### Option F: Target Broader Hashtags

**Hypothesis**: Use more popular hashtags with naturally higher volume

**Implementation**:
```bash
# Instead of:
--target "#supplement"

# Use:
--target "#fitness"  # Broader, more popular
```

**Expected Outcome**:
- More popular hashtags = more videos available
- Example: #fitness likely has 2-3× more videos than #supplement

**Pros**:
- ✅ More videos available
- ✅ No cost/runtime increase
- ✅ Still relevant to client goals

**Cons**:
- ❌ Doesn't solve the fundamental problem
- ❌ Not always an option (client may need specific hashtag)
- ⚠️ Less targeted insights

**Dependencies**:
- None (business decision)

**Risk Level**: None (but may not be applicable)

---

## 🧪 Testing Roadmap

### Test Suite 1: Validate Non-Determinism for Hashtags

**Goal**: Confirm if multiple hashtag scrapes return different videos

**Hypothesis**: Hashtag scraping is non-deterministic (like profiles), enabling Option A/C

#### Test 1A: Sequential Scrapes
```bash
# Run 1
python test_hashtag.py --target "#supplement" --country-code US

# Wait 5 seconds

# Run 2
python test_hashtag.py --target "#supplement" --country-code US
```

**Expected Runtime**: 2 × 1.5 min = 3 minutes
**Expected Cost**: 2 × $1.40 = $2.80

#### Test 1B: Measure Overlap
```python
# Calculate overlap percentage
overlap = len(set(run1_ids) & set(run2_ids))
total_unique = len(set(run1_ids) | set(run2_ids))
overlap_pct = overlap / len(run1_ids) * 100

print(f"Overlap: {overlap}/{len(run1_ids)} videos ({overlap_pct:.1f}%)")
print(f"Unique videos: {total_unique}")
```

**Success Criteria**:
- **< 20% overlap** → Non-determinism confirmed, Option A/C viable ✅
- **20-50% overlap** → Partial non-determinism, Option A/C may work ⚠️
- **> 50% overlap** → Non-determinism invalidated, Option A/C not viable ❌

#### Test 1C: Third Scrape (If Needed)
If overlap < 20%, run 3rd scrape to confirm pattern consistency.

**Deliverables**:
- Overlap % measurement
- Unique video counts
- Go/No-Go decision for Option A/C

---

### Test Suite 2: Analyze Date Distribution

**Goal**: Understand if extending date window helps volume

**Hypothesis**: Videos in 91-150 day range are being returned by Apify

#### Test 2A: Analyze Global Scrape (Test 3)
```python
# Using existing Test 3 data (596 videos, global)
age_buckets = {
    "0-30 days": [],
    "31-90 days": [],
    "91-150 days": [],
    "151+ days": []
}

for video in test3_videos:
    age_days = (now - video['createTime']).days
    if age_days <= 30:
        age_buckets["0-30 days"].append(video)
    elif age_days <= 90:
        age_buckets["31-90 days"].append(video)
    elif age_days <= 150:
        age_buckets["91-150 days"].append(video)
    else:
        age_buckets["151+ days"].append(video)

# Print distribution
for bucket, videos in age_buckets.items():
    print(f"{bucket}: {len(videos)} videos ({len(videos)/596*100:.1f}%)")
```

**Expected Runtime**: 2 minutes (analysis only)
**Expected Cost**: $0 (uses existing data)

#### Test 2B: Analyze US Scrape (Test 2)
Repeat analysis for Test 2 data (253 videos, US filter)

**Success Criteria**:
- **91-150 day bucket has 50+ videos** → Option B may add meaningful volume ✅
- **91-150 day bucket has < 20 videos** → Option B won't help ❌

**Deliverables**:
- Age distribution tables (global vs US filter)
- Go/No-Go decision for Option B

---

### Test Suite 3: Language Distribution (Global)

**Goal**: Measure English % in global scraping to evaluate Option D

**Hypothesis**: English language videos represent 50-60% of global results

#### Test 3A: Analyze Language Distribution
```python
# Using existing Test 3 data (596 videos, global)
language_dist = {}

for video in test3_videos:
    lang = video.get('textLanguage', 'unknown')
    language_dist[lang] = language_dist.get(lang, 0) + 1

# Sort by count
sorted_langs = sorted(language_dist.items(), key=lambda x: x[1], reverse=True)

# Print distribution
print("Language Distribution:")
for lang, count in sorted_langs:
    pct = count / 596 * 100
    print(f"  {lang}: {count} videos ({pct:.1f}%)")

# Calculate English %
english_count = language_dist.get('en', 0)
english_pct = english_count / 596 * 100
print(f"\nEnglish Total: {english_count} videos ({english_pct:.1f}%)")
```

**Expected Runtime**: 2 minutes (analysis only)
**Expected Cost**: $0 (uses existing data)

#### Test 3B: Compare to US Filter
```python
# Compare English count to US filter count
us_filter_count = 253  # Test 2
english_global_count = english_count  # From Test 3A

gain = english_global_count - us_filter_count
gain_pct = (gain / us_filter_count) * 100

print(f"\nComparison:")
print(f"  US filter: {us_filter_count} videos")
print(f"  English (global): {english_global_count} videos")
print(f"  Gain: {gain} videos ({gain_pct:+.1f}%)")
```

**Success Criteria**:
- **English % > 50% AND gain > 50 videos** → Option D viable ✅
- **English % < 40% OR gain < 20 videos** → Option D not viable ❌

**Deliverables**:
- Language distribution table
- English % measurement
- Volume gain vs US filter
- Go/No-Go decision for Option D

---

## 📊 Decision Matrix (Updated with Test Results)

| Option | Volume Gain | Cost | Time | Quality | Test Result | Final Status |
|--------|-------------|------|------|---------|-------------|--------------|
| **A** Multiple Runs | 🔴 Low (+63 = 25%) | 🔴 High ($8.40) | 🔴 High (~40 min) | 🟢 High ✅ | ❌ 82-97% overlap | ❌ NOT VIABLE |
| **B** Extend Date | 🟡 Modest (+35-88 = 14-15%) | 🟢 Low (1×) | 🟢 Low (1×) | 🟡 Medium ⚠️ | ✅ Adds volume | ✅ VIABLE (adjunct) |
| **C** Hybrid A+B | 🔴 Low (316 total → 178 at 270d) | 🔴 High ($8.40) | 🔴 High (~40 min) | 🟡 Medium ⚠️ | ❌ Still insufficient | ❌ NOT VIABLE |
| **D** Global + Language | 🔴 **Low** (455 but ~137 US-quality) | 🟢 Low ($1.40) | 🟢 Low (1.5 min) | 🔴 **Low** ❌ | ⚠️ Quality issues | ⚠️ **QUESTIONABLE** |
| **E** Adaptive Strategy | N/A (workaround) | 🟢 Low (1×) | 🟢 Low (1×) | 🟡 Medium | N/A | ✅ Ready (fallback) |
| **F** Broader Hashtags | 🔴 Negative (-87 = -34%) | 🟢 Low ($1.40) | 🟢 Low (1.5 min) | 🟢 High ✅ | ❌ 166 videos (#fyp) | ❌ NOT VIABLE |

**Key Findings from Testing**:
- ❌ **Option A Failed**: 82.4-96.9% overlap across all delay tests (5s, 2min, 30min)
- ❌ **Option C Failed**: Even 6 scrapes + 270-day window = only 178 videos (insufficient for buckets)
- ❌ **Option D Questionable**: 455 English videos but only ~137 US-quality (20% false positives, 50% non-US English)
- ❌ **Option F Failed**: Popular hashtag (#fyp) returned 34% FEWER videos than #supplement with US filter
- ✅ **Option B Viable**: Adds 14-15% volume but insufficient alone
- 🚨 **NO CLEAR WINNER**: All tested options have significant limitations

---

## 🎯 Recommended Execution Plan

### Phase 1: Quick Wins (Immediate - No Testing Required)

**Action 1.1**: Implement Option E (Adaptive Strategy)
```python
# Add to video_selector.py
def select_strategy(videos_per_bucket, requested_count):
    if videos_per_bucket >= requested_count:
        return "contrastive"  # 80/20 split
    elif videos_per_bucket >= requested_count * 0.5:
        return "contrastive_lite"  # 90/10 split
    else:
        return "top"  # Top-only
```

**Rationale**: Zero-cost safety net, implement regardless of other options

**Timeline**: 30 minutes
**Cost**: $0
**Risk**: None

---

### Phase 2: Testing & Validation (Next 1-2 Days)

**Action 2.1**: Run Test Suite 3 (Language Distribution)
- **Timeline**: 5 minutes (analysis only)
- **Cost**: $0 (uses existing Test 3 data)
- **Deliverable**: English % for #supplement

**Action 2.2**: Run Test Suite 2 (Date Distribution)
- **Timeline**: 5 minutes (analysis only)
- **Cost**: $0 (uses existing Test 2 & 3 data)
- **Deliverable**: Age distribution analysis

**Action 2.3**: Run Test Suite 1 (Non-Determinism)
- **Timeline**: 3 minutes
- **Cost**: $2.80 (2 scrapes)
- **Deliverable**: Overlap % measurement

**Decision Point**: Based on test results, choose primary strategy

---

### Phase 3: Implementation (After Test Results)

#### If Test Suite 1 Shows < 20% Overlap → Implement Option C (Hybrid)
```python
# Implementation plan:
1. Update apify_scraper.py to support multiple runs
2. Add --scrape-runs CLI parameter (default: 1, max: 5)
3. Add deduplication across runs
4. Extend date window to 150 days
5. Test with 3 runs
```

**Expected Outcome**: 700+ videos for analysis
**Timeline**: 2-3 hours implementation
**Cost**: $4.20 per hashtag analysis (3 runs)

#### If Test Suite 1 Shows > 50% Overlap → Implement Option D (Language Filter)
```python
# Implementation plan:
1. Add language_filter.py module
2. Add --language-filter CLI parameter
3. Client-side filter by textLanguage
4. Test with global scraping
```

**Expected Outcome**: 300-400 videos (depends on English %)
**Timeline**: 1 hour implementation
**Cost**: $1.40 per hashtag analysis

#### Fallback → Use Option E (Adaptive Strategy) + Option F (Broader Hashtags)
- Gracefully handle low-volume scenarios
- Recommend broader hashtags to clients when volume insufficient

---

## 📈 Success Metrics

### Volume Metrics
- **Target**: 100+ videos per winning bucket
- **Minimum**: 50+ videos per winning bucket
- **Current**: ~84 videos per bucket (insufficient for some buckets)

### Quality Metrics
- **US-specific content**: Maintained (critical requirement)
- **Recency**: Last 90-150 days (acceptable)
- **Engagement**: Top performers for contrastive analysis

### Cost Metrics
- **Current**: $1.40 per hashtag analysis
- **Target**: < $5.00 per hashtag analysis (3-4× acceptable)

### Time Metrics
- **Current**: 1.5 minutes per hashtag scrape
- **Target**: < 10 minutes per hashtag analysis (6-7× acceptable)

---

## 🚦 Status & Next Actions

### Current Status
- ✅ **Problem identified**: US filtering reduces volume by 57%
- ✅ **Test data collected**: 3 baseline tests + 7 validation scrapes complete
- ✅ **Options identified**: 6 strategic options
- ✅ **Testing phase COMPLETE**: All Test Suites 1-4 executed
- ✅ **Comprehensive analysis COMPLETE**: All scenarios analyzed (6 scrapes, date filters, engagement, popular hashtags)
- ⚠️ **Decision revised**: NO clear winner - all options have significant limitations

### Test Results Summary
1. ✅ Test Suite 1 (5s delay): **96.9% overlap** - NOT VIABLE
2. ✅ Test 1.A (2min delay): **86.9% overlap** - NOT VIABLE
3. ✅ Test 1.B (30min delay): **82.4% overlap** - NOT VIABLE (diminishing returns)
4. ✅ Test Suite 2 (Date Distribution): **+35-88 videos** with 150-day window - VIABLE as adjunct
5. ✅ Test Suite 3 (Language Distribution): **455 English videos** - QUALITY ISSUES (only ~137 US-quality)
6. ✅ Test Suite 4 (Popular Hashtag - #fyp): **166 videos (-34% vs #supplement)** - NOT VIABLE
7. ✅ 6-scrape analysis: **316 unique videos (74.9% duplicates)** - Inefficient
8. ✅ Extended date filters: **270-day window still insufficient** for buckets
9. ✅ Engagement analysis: **48% lower engagement** with older videos

### Immediate Next Actions
1. ✅ Document roadmap (this file) - **COMPLETE**
2. ✅ Run Test Suite 1 (Non-Determinism) - **COMPLETE**
3. ✅ Run Test Suite 2 (Date Distribution) - **COMPLETE**
4. ✅ Run Test Suite 3 (Language Distribution) - **COMPLETE**
5. ✅ Run Test Suite 4 (Popular Hashtag) - **COMPLETE**
6. ✅ Comprehensive analysis - **COMPLETE**
7. ⏳ **Re-evaluate strategy** - No clear winner identified
8. ⏳ **Consider alternative approaches** (see recommendations below)

### Implementation Timeline
- **Total testing cost**: $9.80 (7 scrapes across 4 test suites)
- **Total testing time**: ~1.5 hours
- **Next steps**: Re-evaluate strategy based on comprehensive test findings

---

## 📝 Test Results Log

### Test Suite 1: Non-Determinism Validation (5-Second Delay)
**Status**: ✅ **COMPLETED**
**Date**: 2025-10-09
**Cost**: $2.80 (2 scrapes)

**Results**:
- **Run 1 ID**: GYImoRDaLGVRYwjUP
- **Run 2 ID**: UJeE95ICb9z1OXH5b
- **Run 1**: 193 videos
- **Run 2**: 228 videos
- **Overlap**: **96.9%** (187 videos in both runs)
- **Unique total**: 234 videos
- **Gain from 2nd scrape**: 41 videos (21.2% increase)

**Conclusion**: ❌ **FAILED** - 96.9% overlap is too high. Option A/C (Multiple Sequential Scrapes with 5-second delay) is **NOT VIABLE**.

**Run URLs**:
- Run 1: https://console.apify.com/view/runs/GYImoRDaLGVRYwjUP
- Run 2: https://console.apify.com/view/runs/UJeE95ICb9z1OXH5b

---

### Test 1.A: Non-Determinism with 2-Minute Delay
**Status**: ✅ **COMPLETED**
**Date**: 2025-10-09
**Cost**: $2.80 (2 scrapes)

**Hypothesis**: Longer delay (2 minutes) might reduce overlap by allowing TikTok's algorithm to refresh.

**Results**:
- **Run 3 ID**: 750ocm0oU7BgIk02I
- **Run 4 ID**: ugbYNrcI5mSrYy4qe
- **Run 3**: 213 videos
- **Run 4**: 208 videos
- **Overlap**: **86.9%** (185 videos in both runs)
- **Unique total**: 236 videos
- **Gain from 2nd scrape**: 23 videos (10.8% increase)
- **Improvement from Test 1**: 10% reduction in overlap (96.9% → 86.9%)

**Conclusion**: ⚠️ **MARGINAL IMPROVEMENT** - 86.9% overlap still too high. 2-minute delay helps slightly but **NOT ENOUGH** for viable strategy.

**Run URLs**:
- Run 3: https://console.apify.com/view/runs/750ocm0oU7BgIk02I
- Run 4: https://console.apify.com/view/runs/ugbYNrcI5mSrYy4qe

---

### Test 1.B: Non-Determinism with 30-Minute Delay
**Status**: ✅ **COMPLETED**
**Date**: 2025-10-09
**Cost**: $2.80 (2 scrapes)

**Hypothesis**: 30-minute delay might allow significant algorithm refresh, dramatically reducing overlap.

**Results**:
- **Run 5 ID**: 6h2zNstUqmXNZ8daq
- **Run 6 ID**: vJadXjn8sqegAtZyt
- **Run 5**: 210 videos
- **Run 6**: 206 videos
- **Overlap**: **82.4%** (173 videos in both runs)
- **Unique total**: 243 videos
- **Gain from 2nd scrape**: 33 videos (15.7% increase)
- **Improvement from Test 1.A**: Only 4.5% reduction in overlap (86.9% → 82.4%)

**Trend Analysis**:
- Test 1 (5s delay): 96.9% overlap
- Test 1.A (2min delay): 86.9% overlap → 10% improvement
- Test 1.B (30min delay): 82.4% overlap → 4.5% improvement (**diminishing returns**)

**Conclusion**: ❌ **NOT VIABLE** - 82.4% overlap remains too high. Even with 30-minute delay, overlap only improved by 4.5%, demonstrating diminishing returns. Extending delays further is not cost/time effective.

**Run URLs**:
- Run 5: https://console.apify.com/view/runs/6h2zNstUqmXNZ8daq
- Run 6: https://console.apify.com/view/runs/vJadXjn8sqegAtZyt

**Related Document**: See `/home/jorge/rumiaifinal/30minwait.md` for detailed test protocol

---

### Test Suite 2: Date Distribution Analysis
**Status**: ✅ **COMPLETED**
**Date**: 2025-10-09
**Cost**: $0 (uses existing data)
**Runtime**: 5 minutes

**Goal**: Analyze age distribution of scraped videos to determine if extending date window (90→150 days) would add meaningful volume.

**Results - US Filter (253 videos)**:
- 0-30 days: 22 videos (8.7%)
- 31-90 days: 27 videos (10.7%)
- **91-150 days: 35 videos (13.8%)** ⚠️
- 151+ days: 169 videos (66.8%)

**Results - Global (596 videos)**:
- 0-30 days: 83 videos (13.9%)
- 31-90 days: 81 videos (13.6%)
- **91-150 days: 88 videos (14.8%)** ✅
- 151+ days: 344 videos (57.7%)

**Key Finding**:
- US filter: Extending to 150 days adds **+35 videos** (~14% increase)
- Global: Extending to 150 days adds **+88 videos** (~15% increase)

**Conclusion**: ✅ **VIABLE** - Extending date window from 90 to 150 days would add meaningful volume (13-15% increase). The gain is modest but significant when combined with other strategies.

---

### Test Suite 3: Language Distribution Analysis
**Status**: ✅ **COMPLETED**
**Date**: 2025-10-09
**Cost**: $0 (uses existing data)
**Runtime**: 5 minutes

**Goal**: Measure English language % in global scraping to evaluate Option D (Global Scraping + Language Filter) viability.

**Results - Language Distribution (596 global videos)**:
- **English (en): 455 videos (76.3%)** 🎯
- Unknown: 53 videos (8.9%)
- German (de): 27 videos (4.5%)
- Spanish (es): 23 videos (3.9%)
- French (fr): 12 videos (2.0%)
- Others: <1% each

**Comparison to US Filter**:
- US filter (Test 2): 253 videos
- English global (Test 3): **455 videos**
- **Gain: +202 videos (+79.8%)** 🚀

**Conclusion**: ✅✅ **HIGHLY VIABLE** - Option D (Global Scraping + English Language Filter) provides **80% more videos** than US geographic filtering, making it the **clear winner** among all tested strategies. English content represents over 3/4 of global videos for #supplement hashtag.

**⚠️ QUALITY CAVEAT DISCOVERED**: Manual review of English-classified videos revealed:
- 20% false positives (German videos with English hashtags)
- 50% non-US English (UK, Australia, Canada)
- Only ~30% potentially US content
- Effective US-quality volume: ~137 videos (46% WORSE than US filter!)

**Revised Assessment**: While `textLanguage: 'en'` provides volume, it does NOT guarantee US geographic content. Language detection is caption-based, not content-based, leading to mixed quality results.

---

### Test Suite 4: Popular Hashtag Validation (Option F)
**Status**: ✅ **COMPLETED**
**Date**: 2025-10-09
**Cost**: $1.40 (1 scrape)
**Runtime**: 1.5 minutes

**Goal**: Validate if more popular hashtags (#fyp) provide better volume than niche hashtags (#supplement) when US filter is applied.

**Hypothesis**: More popular hashtags = more videos available, solving the volume problem.

**Test Configuration**:
- Hashtag: #fyp (one of TikTok's most popular hashtags)
- Geographic Filter: US (proxyCountryCode: US)
- Date Filter: None
- Results Per Page: 800

**Results**:
- **Run ID**: LvyVudmyNIUA7LSn3
- **Run URL**: https://console.apify.com/view/runs/LvyVudmyNIUA7LSn3
- **Total videos**: 166 videos ❌

**Bucket Distribution (#fyp with US Filter)**:

| Bucket | Range | Videos | % of Total | Status |
|--------|-------|--------|------------|--------|
| 0-3s | 0-3s | 0 | 0.0% | Empty |
| 3-9s | 3-9s | 25 | 15.1% | Low |
| 9-13s | 9-13s | 28 | 16.9% | Low |
| 13-18s | 13-18s | 33 | 19.9% | Moderate |
| 18-33s | 18-33s | 24 | 14.5% | Low |
| 33-60s | 33-60s | 13 | 7.8% | Very low |
| **60-90s** | **60-90s** | **34** | **20.5%** | **Top 1** ❌ |
| 90-120s | 90-120s | 5 | 3.0% | Very low |
| >120s | >120s | 4 | 2.4% | Out of range |

**Top 3 Winning Buckets**:
1. 60-90s: 34 videos (20.5%) - ❌ **INSUFFICIENT** (< 50 videos)
2. 13-18s: 33 videos (19.9%) - ❌ **INSUFFICIENT** (< 50 videos)
3. 9-13s: 28 videos (16.9%) - ❌ **INSUFFICIENT** (< 50 videos)

**Age Distribution Analysis**:
- 0-30 days: 26 videos (15.7%)
- 31-90 days: 12 videos (7.2%)
- 91-150 days: 3 videos (1.8%)
- 151-270 days: 11 videos (6.6%)
- 271+ days: 114 videos (68.7%)

**Age Statistics**:
- Average age: 497.3 days (~16 months)
- Median age: 402 days (~13 months)
- Recent (≤90 days): 38 videos (22.9%)

**Comparison: #fyp vs #supplement (US Filter)**:

| Metric | #supplement (US) | #fyp (US) | Difference |
|--------|------------------|-----------|------------|
| Total videos | 253 | 166 | -87 (-34%) |
| Top 3 buckets | 76, 52, 43 | 34, 33, 28 | Much worse |
| Recent (≤90d %) | 19.4% | 22.9% | +3.5% (minimal) |
| Avg age | Unknown | 497 days | Very old |

**Conclusion**: ❌ **NOT VIABLE** - Popular hashtags do NOT provide better volume with US filtering. Key findings:
1. **34% FEWER videos** than #supplement despite being far more popular
2. **ALL 3 buckets insufficient** (< 50 videos minimum)
3. **68.7% of videos > 270 days old** - no recency benefit
4. **Different duration profile** - #fyp skews towards shorter videos (60-90s top) vs #supplement (33-60s top)

**Root Cause**: US geographic filtering dramatically reduces available volume **regardless of hashtag popularity**. The fundamental constraint is geographic filtering, not hashtag selection. Apify/TikTok's algorithm favors older, established videos over recent ones for all hashtags.

---

### Alternative Test: Hashtag Scraper Actor
**Status**: ✅ **COMPLETED**
**Date**: 2025-10-09
**Cost**: Unknown (pre-existing data)
**Runtime**: Analysis only

**Goal**: Evaluate alternative TikTok scraper actor (different from clockworks/tiktok-scraper) to see if it provides better volume.

**Test Configuration**:
- Actor: Different hashtag scraper actor (no Apify-side filters)
- Hashtag: #supplement
- Geographic Filter: None available
- Date Filter: None
- Results: 640 videos (CSV analysis)

**Results - Country Code Analysis**:
- **640 total videos**
- **85% (540 videos) have NO country code** (locationMeta/countryCode empty)
- Only 15% (100 videos) have country codes
- Country codes use GeoNames IDs, not standard codes

**Results - Language Distribution**:
- **English (en): 481 videos (75.2%)**
- French (fr): 53 videos (8.3%)
- Unknown (un): 52 videos (8.1%)
- Spanish (es): 23 videos (3.6%)
- German (de): 11 videos (1.7%)
- Others: <2 videos each

**Age Distribution (All 640 videos)**:
- Average age: 398.4 days (~13 months)
- Median age: 199 days (~6.5 months)
- ≤90 days: 187 videos (29.4%)
- ≤150 days: 263 videos (41.3%)
- ≤270 days: Not analyzed

**Bucket Distribution - All Videos (No Filters)**:

| Bucket | Videos | Status |
|--------|--------|--------|
| 60-90s | 125 | ✅ EXCELLENT |
| 33-60s | 117 | ✅ EXCELLENT |
| 18-33s | 95 | ⚠️ ADEQUATE |
| **Top 3** | **125, 117, 95** | **⚠️ ADEQUATE** |

**Bucket Distribution - English Only, ≤270 days**:

| Date Filter | Total English | Top 3 Buckets | Ready? |
|-------------|---------------|---------------|--------|
| ≤90 days | 128 | 40, 26, 13 | ❌ NO |
| ≤150 days | 185 | 49, 38, 23 | ❌ NO (1 video short!) |
| ≤270 days | 271 | 57, 54, 43 | ❌ NO (7 videos short) |

**Conclusion**: ⚠️ **QUESTIONABLE** - Alternative actor faces same core issues as other approaches:

1. **❌ No reliable country filtering**: 85% of videos lack country codes - cannot filter for US content
2. **⚠️ Same language detection issues**: Uses `textLanguage` field (caption-based, not content-based)
   - Same 20% false positive rate (non-English with English hashtags)
   - Same 50% non-US English mix (UK, Australia, Canada)
   - Effective US-quality: only ~30% of "English" videos = ~80-110 actual US videos
3. **❌ Insufficient volume with date filters**: Even 270-day window with English-only yields insufficient buckets
4. **✅ Good volume without filters**: 640 videos with 2 excellent buckets IF no geographic or recency filtering applied

**Quality Assessment**: Same fundamental problems as Option D (Global + Language Filter):
- English language ≠ US geography
- Caption-based language detection has false positives
- No way to reliably filter for US content
- Trade-off between volume (no filters) and quality (US-specific, recent content)

**Recommendation**: NOT a solution to the core problem. This actor provides similar results to the global scrape approach but lacks any geographic filtering capability.

---

## 📊 Comprehensive Analysis: 6 Scrapes + Extended Filters

### Total Unique Videos Across All 6 Scrapes
**Status**: ✅ **COMPLETED**
**Date**: 2025-10-09
**Cost**: $8.40 total (6 scrapes × $1.40)

**Summary**:
- **Total videos scraped (with duplicates)**: 1,258 videos
- **Total UNIQUE videos**: 316 videos
- **Duplicate count**: 942 videos
- **Efficiency**: Only 25.1% unique (74.9% duplicates)

**Per-Test Breakdown**:
- Test 1 (5s delay, 2 scrapes): 234 unique videos
- Test 1.A (2min delay, 2 scrapes): 236 unique videos
- Test 1.B (30min delay, 2 scrapes): 243 unique videos

**Cross-Test Overlap** (High overlap even between different test runs):
- Test 1 ∩ Test 1.A: 191 videos
- Test 1 ∩ Test 1.B: 190 videos
- Test 1.A ∩ Test 1.B: 199 videos

**Key Finding**: Even with 6 scrapes across 3 different time delays, we only achieved 316 unique videos with 74.9% duplication. This definitively proves multiple scrapes are inefficient.

---

### Bucket Distribution Analysis (316 Unique Videos, No Date Filter)

**Complete Bucket Breakdown**:

| Bucket | Range | Videos | % of Total | Status |
|--------|-------|--------|------------|--------|
| 0-3s | 0-3s | 0 | 0.0% | Empty |
| 3-9s | 3-9s | 7 | 2.2% | Very low |
| 9-13s | 9-13s | 10 | 3.2% | Very low |
| 13-18s | 13-18s | 14 | 4.4% | Low |
| 18-33s | 18-33s | 39 | 12.3% | Moderate |
| **33-60s** | **33-60s** | **76** | **24.1%** | **Top 1** ⚠️ |
| **60-90s** | **60-90s** | **52** | **16.5%** | **Top 2** ⚠️ |
| **90-120s** | **90-120s** | **43** | **13.6%** | **Top 3** ❌ |
| >120s | >120s | 75 | 23.7% | Out of range |

**Top 3 Winning Buckets**:
1. 33-60s: 76 videos (24.1%) - ⚠️ **ADEQUATE** (need 100+)
2. 60-90s: 52 videos (16.5%) - ⚠️ **ADEQUATE** (barely meets 50 minimum)
3. 90-120s: 43 videos (13.6%) - ❌ **INSUFFICIENT** (< 50 videos)

**Contrastive Analysis Readiness**: ❌ **NOT READY** - Only 1 of 3 buckets meets 50-video minimum, ZERO buckets meet 100-video recommendation.

---

### Extended Date Filter Analysis

#### Date Filter Progression (316 Unique Videos from 6 Scrapes)

| Threshold | Total Videos | % of All (316) | Gain from Previous | Top 3 Buckets (33-60s, 60-90s, 90-120s) |
|-----------|--------------|----------------|--------------------|------------------------------------------|
| No filter | 316 | 100.0% | - | 76, 52, 43 (⚠️ ⚠️ ❌) |
| 90 days | 79 | 25.0% | - | 19, 17, 8 (❌ ❌ ❌) |
| 150 days | 123 | 38.9% | +44 | 29, 24, 16 (❌ ❌ ❌) |
| **270 days** | **178** | **56.3%** | **+55** | **43, 31, 23 (❌ ❌ ❌)** |

**Key Findings**:
1. **90-day filter**: Reduces to only 79 videos - ALL buckets insufficient
2. **150-day filter**: 123 videos (+55.7% from 90d) - ALL buckets still insufficient
3. **270-day filter** (9 months): 178 videos (+44.7% from 150d) - **STILL ALL BUCKETS INSUFFICIENT**

**Critical Conclusion**: Even with 6 scrapes ($8.40) + 270-day window, we cannot achieve 50+ videos per bucket for contrastive analysis.

---

### Engagement Quality Analysis (150-Day Filter)

**Comparison: All Videos vs 150-Day Filter**:

| Metric | All Videos (316) | 150-Day Filter (123) | Difference |
|--------|------------------|----------------------|------------|
| Avg Plays | 1,939,104 | 2,630,354 | ✅ +35.6% |
| Avg Likes | 58,437 | 41,244 | ⚠️ -29.4% |
| **Engagement Rate** | **3.01%** | **1.57%** | **⚠️ -48% LOWER** |
| Median Plays | 400,300 | 225,100 | ⚠️ -43.7% |
| Median Likes | 14,800 | 7,649 | ⚠️ -48.3% |

**Top Performers Distribution**:
- Videos with 100K+ plays: 75.3% (all) vs 63.4% (150d)
- Videos with 10K+ likes: 58.2% (all) vs 48.0% (150d)

**Verdict**: ⚠️ **Significant Quality Trade-off** - The 150-day filtered videos have higher average play counts (due to accumulation over time) but **48% lower engagement rate**. This suggests older videos have lower quality/relevance for ML training, where engagement rate matters more than raw view counts.

---

## 🎯 Final Recommendation: Option D Wins Decisively

### Comparison Table: All Strategies

| Strategy | Unique Videos | Cost | Time | Quality | Bucket Readiness | Result |
|----------|---------------|------|------|---------|------------------|--------|
| Single scrape (US filter) | 253 | $1.40 | 1.5 min | High ✅ | ❌ Insufficient | Baseline |
| Test 1 (5s delay, 2 scrapes) | 234 | $2.80 | 3 min | High ✅ | ❌ Insufficient | 96.9% overlap |
| Test 1.A (2min delay, 2 scrapes) | 236 | $2.80 | 5 min | High ✅ | ❌ Insufficient | 86.9% overlap |
| Test 1.B (30min delay, 2 scrapes) | 243 | $2.80 | 33 min | High ✅ | ❌ Insufficient | 82.4% overlap |
| **6 scrapes combined (US filter)** | **316** | **$8.40** | **~40 min** | **High ✅** | **❌ Insufficient** | **74.9% duplicates** |
| 6 scrapes + 150d filter | 123 | $8.40 | ~40 min | Medium ⚠️ | ❌ Insufficient | 48% lower engagement |
| 6 scrapes + 270d filter | 178 | $8.40 | ~40 min | Medium ⚠️ | ❌ Insufficient | Quality concerns |
| **Option D (Global + English)** | **455** | **$1.40** | **1.5 min** | **Medium ✅** | **✅ LIKELY READY** | **CLEAR WINNER** |

### Why Option D Wins

**Volume Comparison**:
- Option D: **455 English videos**
- 6 scrapes (US): 316 videos
- Single US scrape: 253 videos
- **Gain vs 6 scrapes**: +139 videos (+44% more)
- **Gain vs single US scrape**: +202 videos (+80% more)

**Cost & Time Efficiency**:
- **6× cheaper** than 6 scrapes ($1.40 vs $8.40)
- **27× faster** than 6 scrapes (1.5 min vs ~40 min with delays)
- **Much simpler** to implement (single scrape vs complex deduplication)

**Quality Assessment**:
- English language = reasonable proxy for US/Western market
- 76.3% of global #supplement videos are in English
- Acceptable trade-off: slight quality reduction for massive volume gain

**Expected Bucket Distribution** (Estimated from 455 videos):
- Assuming similar distribution to 316-video baseline
- Expected top 3 buckets: ~110, 75, 62 videos
- **ALL 3 buckets likely meet 50+ minimum** ✅
- **Top 2 buckets likely meet 100+ recommendation** ✅

### Implementation Recommendation

**Immediate Action**: Implement Option D (Global Scraping + English Language Filter)

```python
# Scrape without geographic filter
videos = scraper.scrape_videos(
    hashtags=["#supplement"],
    resultsPerPage=800
    # Remove proxyCountryCode parameter
)

# Client-side English language filter
english_videos = [v for v in videos if v.get('textLanguage') == 'en']
```

**Expected Outcome**: 455 English videos per hashtag analysis, sufficient for robust contrastive analysis across top 3 duration buckets.

**Fallback**: If 455 videos still insufficient for specific buckets, combine with Option B (extend date window to 150 days) or use Option E (adaptive strategy).

---

## 🔗 Related Documents

- **30minwait.md**: Test 1.B (30-minute delay) detailed protocol and instructions
- **ScraperLimitations.md**: Original scraper limitation analysis
- **VideoDiscoveryCHILD.md**: Stage 1 high-level design
- **VideoDiscoveryCHILDTI.md**: Stage 1 technical implementation
- **STAGE1_TESTS.md**: Stage 1 test results

---

## 🧪 Test Scripts Documentation

All test scripts are located in `/tmp/` directory and follow a consistent pattern for executing and analyzing Apify scraping tests.

### Test Script Categories

#### 1. **Test Execution Scripts**
Scripts that run Apify scrapes with specific configurations.

**Example: `/tmp/test_fyp_us_filter.py`**
- **Purpose**: Execute hashtag scrape with specific parameters
- **Pattern**:
  - Load `.env` for APIFY_API_KEY
  - Initialize `ApifyClient` directly
  - Configure scraper parameters (hashtag, country code, resultsPerPage)
  - Run scrape via `client.actor("clockworks/tiktok-scraper").call()`
  - Extract results from dataset
  - Print run ID and URL for analysis

**Usage**:
```bash
python /tmp/test_fyp_us_filter.py
# Returns: Run ID, Run URL, Total videos scraped
```

**Key Features**:
- Direct Apify API integration (not using ml_pipeline code)
- Prints configuration before execution for documentation
- Returns run metadata for subsequent analysis scripts

---

#### 2. **Bucket Analysis Scripts**
Scripts that analyze duration bucket distribution of scraped videos.

**Example: `/tmp/analyze_fyp_buckets.py`**
- **Purpose**: Analyze how videos distribute across 8 ML training buckets
- **Input**: Apify run ID (hardcoded from test execution)
- **Analysis**:
  - Bucket videos by duration (0-3s, 3-9s, 9-13s, 13-18s, 18-33s, 33-60s, 60-90s, 90-120s)
  - Calculate bucket distribution percentages
  - Identify top 3 winning buckets
  - Assess contrastive analysis readiness (50+ min, 100+ recommended)
  - Compare to baseline tests

**Output Format**:
```
BUCKET DISTRIBUTION
=====================
Bucket       Range        Videos   Percentage
----------------------------------------------------
33-60s       33-60s       76       24.1%        ⚠️  ADEQUATE
60-90s       60-90s       52       16.5%        ⚠️  ADEQUATE
90-120s      90-120s      43       13.6%        ❌ INSUFFICIENT

TOP 3 WINNING BUCKETS:
1. 33-60s (33-60s): 76 videos (24.1%) - ⚠️  ADEQUATE
2. 60-90s (60-90s): 52 videos (16.5%) - ⚠️  ADEQUATE
3. 90-120s (90-120s): 43 videos (13.6%) - ❌ INSUFFICIENT

CONTRASTIVE ANALYSIS READINESS:
❌ 1 bucket(s) have < 50 videos - NOT READY
```

**Readiness Criteria**:
- ✅ **EXCELLENT**: 100+ videos per bucket
- ⚠️ **ADEQUATE**: 50-99 videos per bucket
- ❌ **INSUFFICIENT**: <50 videos per bucket

---

#### 3. **Age Distribution Analysis Scripts**
Scripts that analyze video age (recency) distribution.

**Example: `/tmp/analyze_fyp_age_distribution.py`**
- **Purpose**: Analyze how old videos are to evaluate date filtering effectiveness
- **Input**: Apify run ID
- **Analysis**:
  - Age buckets: 0-30 days, 31-90 days, 91-150 days, 151-270 days, 271+ days
  - Calculate age statistics (average, median, min, max)
  - Assess recency (≤90d, ≤150d, ≤270d thresholds)
  - Compare to baseline tests

**Output Format**:
```
AGE DISTRIBUTION
=====================
Age Range            Videos     Percentage
----------------------------------------------------
0-30 days            26         15.7%
31-90 days           12         7.2%
91-150 days          3          1.8%
151-270 days         11         6.6%
271+ days            114        68.7%

AGE STATISTICS:
Average age: 497.3 days (~16 months)
Median age: 402 days (~13 months)
Recent (≤90 days): 38 videos (22.9%)

KEY FINDING:
⚠️  #fyp has 3.5% LESS recent content
    Hashtag popularity does not guarantee fresher content
```

**Key Metrics**:
- **≤90 days**: Ideal recency window
- **≤150 days**: Acceptable with caveats
- **≤270 days**: Extended window (9 months)
- **271+ days**: Old content (engagement concerns)

---

#### 4. **CSV Analysis Scripts**
Scripts that analyze alternative Apify actors via CSV exports.

**Example: `/tmp/analyze_hashtag_actor_csv.py`**
- **Purpose**: Analyze alternative TikTok scraper actors (non-clockworks)
- **Input**: CSV file exported from Apify run
- **Analysis**:
  - Country code distribution (locationMeta/countryCode)
  - Age distribution across date ranges
  - Bucket distribution for different date filters
  - Compare to primary scraper results

**Key Features**:
- CSV parsing with `csv.DictReader`
- Handles nested fields (e.g., `videoMeta/duration`, `locationMeta/countryCode`)
- UTF-8-sig encoding for BOM handling
- Reusable bucket analysis function

**Output Highlights**:
```
COUNTRY CODE DISTRIBUTION:
Country Code         Count      Percentage
----------------------------------------------------
(EMPTY)              540        84.4%
US                   45         7.0%

❌ FINDING: Over 50% of videos have NO country code
   Geographic filtering NOT reliable with this dataset
```

---

#### 5. **Language Distribution Scripts**
Scripts that analyze language distribution for global scraping evaluation.

**Example: `/tmp/analyze_hashtag_actor_language.py`**
- **Purpose**: Measure English % in global scraping for Option D viability
- **Input**: CSV file or Apify dataset
- **Analysis**:
  - Language distribution (textLanguage field)
  - English % calculation
  - Sample text display for validation
  - Compare to US filter results

**Output Format**:
```
LANGUAGE DISTRIBUTION (textLanguage field)
=====================
Language Code        Count      Percentage
----------------------------------------------------
en                   481        75.2%
fr                   53         8.3%
un                   52         8.1%
es                   23         3.6%
de                   11         1.7%

ENGLISH LANGUAGE ANALYSIS:
English (en):        481 videos (75.2%)
Non-English:         107 videos (16.7%)
No language data:    52 videos (8.1%)

✅ EXCELLENT: 75.2% of videos are in English
   HashtagActor provides good English content for ML training
```

**Quality Assessment Criteria**:
- **>70% English**: ✅ EXCELLENT
- **50-70% English**: ✅ GOOD
- **30-50% English**: ⚠️ MODERATE
- **<30% English**: ❌ LOW

---

#### 6. **Combined Analysis Scripts**
Scripts that combine multiple analyses (language + date + buckets).

**Example: `/tmp/analyze_english_date_buckets.py`**
- **Purpose**: Comprehensive analysis of English-only videos across date ranges
- **Input**: CSV file
- **Filters**: English language only (`textLanguage == 'en'`)
- **Analysis**:
  - Date range filtering (≤90d, ≤150d, ≤270d)
  - Bucket distribution per date range
  - Top 3 bucket identification
  - Cross-date range comparison

**Output Format**:
```
SUMMARY COMPARISON: ENGLISH VIDEOS ONLY
=====================
Date Filter          Total Videos    Top 3 Buckets                            Ready?
----------------------------------------------------
≤90 days             128             40, 26, 13                               ❌
≤150 days            185             49, 38, 23                               ❌ (1 video short!)
≤270 days            271             57, 54, 43                               ❌ (7 videos short)

AGE STATISTICS (English videos):
  Average age: 398.4 days
  Median age: 199 days
```

---

### Test Script Architecture Pattern

All scripts follow this consistent structure:

#### 1. **Environment Setup**
```python
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
```

#### 2. **Apify Client Initialization**
```python
from apify_client import ApifyClient

apify_api_key = os.getenv('APIFY_API_KEY')
if not apify_api_key:
    print("ERROR: APIFY_API_KEY not found")
    sys.exit(1)

client = ApifyClient(apify_api_key)
```

#### 3. **Data Retrieval**
```python
# For test execution:
run = client.actor("clockworks/tiktok-scraper").call(run_input={...})
videos = client.dataset(run["defaultDatasetId"]).list_items().items

# For analysis scripts:
run_info = client.run(run_id).get()
dataset_id = run_info['defaultDatasetId']
videos = client.dataset(dataset_id).list_items().items
```

#### 4. **ML Training Buckets (Standard)**
```python
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
```

#### 5. **Analysis & Reporting**
- Clear section headers (`'='*80`)
- Tabular output for distributions
- Visual status indicators (✅ ⚠️ ❌)
- Comparison to baseline tests
- Key findings summary

---

### Common Analysis Functions

#### Bucket Distribution Analysis
```python
def bucket_videos(video_list, label):
    """Bucket videos by duration and assess readiness."""
    bucket_dist = {bucket_name: [] for bucket_name in buckets.keys()}

    for video in video_list:
        duration = video['duration']
        for bucket_name, (min_dur, max_dur) in buckets.items():
            if min_dur <= duration < max_dur:
                bucket_dist[bucket_name].append(video['id'])
                break

    # Display table, identify top 3, assess readiness
    # Returns: top_3_counts
```

#### Age Distribution Analysis
```python
def analyze_age_distribution(videos):
    """Calculate video age and bucket by date ranges."""
    now = datetime.now(timezone.utc)
    age_buckets = {
        '0-30 days': [],
        '31-90 days': [],
        '91-150 days': [],
        '151-270 days': [],
        '271+ days': []
    }

    for video in videos:
        video_date = datetime.fromtimestamp(video['createTime'], tz=timezone.utc)
        age_days = (now - video_date).days
        # Bucket logic...
```

---

### Test Script Usage Workflow

#### Step 1: Execute Test Scrape
```bash
python /tmp/test_fyp_us_filter.py
# Output: Run ID: LvyVudmyNIUA7LSn3
```

#### Step 2: Analyze Bucket Distribution
```bash
# Edit script to use Run ID from Step 1
python /tmp/analyze_fyp_buckets.py
# Output: Bucket distribution table, top 3 buckets, readiness assessment
```

#### Step 3: Analyze Age Distribution
```bash
python /tmp/analyze_fyp_age_distribution.py
# Output: Age distribution, recency metrics, comparison to baseline
```

#### Step 4: Document Results
- Copy output to HashtagVolumeStrategy.md
- Update Decision Matrix
- Revise recommendations

---

### Test Scripts Inventory

**Test Suite 1 (Non-Determinism)**:
- Test execution scripts (2 scrapes with delays)
- Overlap analysis scripts

**Test Suite 2 (Date Distribution)**:
- `/tmp/analyze_fyp_age_distribution.py` - Age analysis
- Uses existing Test 2 & Test 3 data

**Test Suite 3 (Language Distribution)**:
- `/tmp/analyze_hashtag_actor_language.py` - Language analysis
- Uses Test 3 global scrape data

**Test Suite 4 (Popular Hashtag)**:
- `/tmp/test_fyp_us_filter.py` - Execute #fyp scrape
- `/tmp/analyze_fyp_buckets.py` - Bucket analysis
- `/tmp/analyze_fyp_age_distribution.py` - Age analysis

**Alternative Test (Hashtag Scraper Actor)**:
- `/tmp/analyze_hashtag_actor_csv.py` - Full CSV analysis
- `/tmp/analyze_hashtag_actor_language.py` - Language distribution
- `/tmp/analyze_english_date_buckets.py` - English-only comprehensive analysis

---

### Key Design Principles

1. **Direct API Access**: Scripts use `apify_client` directly, not ml_pipeline code
2. **Hardcoded Run IDs**: Analysis scripts reference specific test runs for reproducibility
3. **Consistent Output Format**: All scripts use similar table structures and status indicators
4. **Self-Documenting**: Scripts print configuration and context before execution
5. **Reusable Functions**: Common analysis logic extracted into reusable functions
6. **CSV Compatibility**: Support for both API and CSV-based analysis
7. **Comparative Analysis**: Always compare results to baseline tests

---

### Script Maintenance Notes

**When Creating New Test Scripts**:
1. Follow the established pattern (environment setup → client init → data retrieval → analysis → reporting)
2. Use standard ML training buckets (8 buckets: 0-3s through 90-120s)
3. Include status indicators (✅ ⚠️ ❌) for readiness assessment
4. Print run URLs for Apify console access
5. Compare results to baseline tests
6. Document findings in HashtagVolumeStrategy.md

**Standard Thresholds**:
- **Bucket readiness**: 50+ minimum, 100+ recommended
- **Age ranges**: 0-30d, 31-90d, 91-150d, 151-270d, 271+d
- **Language quality**: 70%+ excellent, 50-70% good, 30-50% moderate, <30% low
- **Overlap tolerance**: <20% viable, 20-50% marginal, >50% not viable

---

## 📞 Apify Support Ticket

**Ticket Status**: Open
**Issue**: Date filter parameters not working for hashtag searches
**Run URLs**:
- WITH date filter: https://console.apify.com/view/runs/Ec6VqhOOQ8REZTywu
- WITHOUT date filter: https://console.apify.com/view/runs/NXMbGqnMcFliMWh1w

**Expected Response**: Waiting for confirmation on whether hashtag date filtering is supported

---

## Revision History

| Date | Change | Author |
|------|--------|--------|
| 2025-10-09 | Initial document created | Claude |
| 2025-10-09 | Added testing roadmap and decision matrix | Claude |
| 2025-10-09 | Added Test Suite 1 results (5-second delay: 96.9% overlap) | Claude |
| 2025-10-09 | Added Test 1.A results (2-minute delay: 86.9% overlap) | Claude |
| 2025-10-09 | Added Test 1.B in-progress status (30-minute delay: pending) | Claude |
| 2025-10-09 | Updated Related Documents with 30minwait.md reference | Claude |
| 2025-10-09 | Added Test 1.B final results (30-minute delay: 82.4% overlap) | Claude |
| 2025-10-09 | Added Test Suite 2 results (Date Distribution: +35-88 videos) | Claude |
| 2025-10-09 | Added Test Suite 3 results (Language Distribution: 455 English videos) | Claude |
| 2025-10-09 | Added comprehensive analysis section (6 scrapes, date filters, engagement) | Claude |
| 2025-10-09 | Added final recommendation: Option D wins decisively | Claude |
| 2025-10-09 | Updated Decision Matrix with test outcomes | Claude |
| 2025-10-09 | Updated Status & Next Actions - Testing complete | Claude |
| 2025-10-09 | Added Test Suite 3 quality caveat (manual review findings) | Claude |
| 2025-10-09 | Added Test Suite 4 results (Popular Hashtag #fyp: 166 videos) | Claude |
| 2025-10-09 | Revised Decision Matrix - Option D downgraded, Option F failed | Claude |
| 2025-10-09 | Revised status: NO clear winner - all options have limitations | Claude |
| 2025-10-09 | Added Alternative Test: Hashtag Scraper Actor analysis (640 videos) | Claude |
| 2025-10-09 | Added comprehensive Test Scripts Documentation section | Claude |
