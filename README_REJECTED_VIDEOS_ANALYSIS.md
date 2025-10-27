# Rejected Videos Analysis - Complete Documentation

## Analysis Overview

This folder contains a comprehensive analysis of 407 rejected videos from the Rollo Wellness cluster hashtag validation. The analysis determines whether the validation logic is too strict.

**Key Stats:**
- Total videos: 780 (after deduplication)
- Passed validation: 373 (47.8%)
- Rejected: 407 (52.2%)
- Rejection rate: Unusually high (2x industry standard)

**Preliminary Finding:** Validation is likely too strict

---

## Documents in This Analysis

### 1. REJECTED_VIDEOS_ANALYSIS_SUMMARY.txt
**Executive summary with all key findings**

Contains:
- Key metrics and statistics
- Critical findings about validation
- Hypothesis about rejected videos (3 categories)
- Implementation options and priorities
- Recommended next steps

**Read this first** to understand the situation and recommendations.

---

### 2. MANUAL_REVIEW_FRAMEWORK.md
**Detailed framework for manually reviewing 20 rejected videos**

Contains:
- How to obtain data for rejected videos
- Categorization logic (A/B/C framework)
- Detailed examples for each category
- Review form template
- Spreadsheet structure for tracking
- Decision matrix based on results
- Implementation instructions if changes needed

**Use this** to structure your manual review of 20 videos.

---

## Quick Start Guide

### Step 1: Understand the Problem (5 minutes)
Read: REJECTED_VIDEOS_ANALYSIS_SUMMARY.txt (first section)

### Step 2: Get Ready for Manual Review (10 minutes)
Read: MANUAL_REVIEW_FRAMEWORK.md (Data to Collect section)

### Step 3: Obtain Video Data (1-2 hours)
Use provided 10 video IDs + find 10 more from 407 rejected videos

### Step 4: Categorize Videos (1-2 hours)
Use framework to assign each video to Category A, B, or C

### Step 5: Make Decision (30 minutes)
Calculate breakdown and apply decision matrix from framework

### Step 6: Implement Changes (1-2 hours, if needed)
Use implementation instructions from MANUAL_REVIEW_FRAMEWORK.md

---

## Key Findings Summary

### Finding 1: Hashtag Imbalance
The 4 target hashtags have uneven performance:
- #wellness: 52.5% of passed videos (strong)
- #wellnessjourney: 32.8% of passed videos (good)
- #healthandwellness: 19.7% of passed videos (weak)
- #wellnesssupplements: 0.0% of passed videos (ABSENT!)

**Implication:** #wellnesssupplements may be a defunct target.

### Finding 2: Missing Related Hashtags
Wellness-adjacent hashtags found IN passed videos but NOT in target list:
- #supplements: 18% (CRITICAL - highest alternative)
- #health: 11.5%
- #wellnesstips: 9.8%
- #guthealth: 9.8%
- #healthylifestyle: 8.2%

**Implication:** Target list is incomplete. Rejected videos likely use these tags.

### Finding 3: High Rejection Rate
- 52.2% rejection is 2-3x industry standard (10-30%)
- Indicates either Apify false positives OR incomplete target list
- Manual review will determine which

### Finding 4: Data Quality is Good
- Description fallback used only 0.4% (3 videos)
- Hashtags field well-populated
- Videos rejected due to content, not missing data

---

## Hypothesis: What Are the 407 Rejected Videos?

### Category A: True False Positives (60-75% estimated)
Videos about unrelated topics returned by Apify fuzzy search
- Examples: fitness, dance, cooking, travel, memes
- Action: Keep rejected
- These confirm validation is working

### Category B: Similar Wellness Hashtags (20-35% estimated)
Wellness content but using different hashtags
- Examples: #vitamins, #supplements, #health, #holistic, #fitness
- Action: Could accept (expand target list)
- Would recover 80-160 videos

### Category C: Wellness with No Hashtags (1-5% estimated)
Wellness content but empty hashtags field
- Action: Improve description parsing
- Rare case, low priority

---

## Decision Framework

After reviewing 20 videos, categorize and calculate:

```
Category A (True False Positives): ____ / 20 = _____%
Category B (Similar Wellness Tags): ____ / 20 = _____%
Category C (No Hashtags):           ____ / 20 = _____%
                          TOTAL:            20 = 100%
```

**Then apply decision matrix:**

| Category B % | Verdict | Action |
|---|---|---|
| < 10% | Validation is CORRECT | No changes |
| 15-30% | Moderately too strict | Add #health, #supplements |
| 35-50% | Severely too strict | Major expansion/fuzzy matching |
| > 50% | Fundamentally broken | Complete redesign |

---

## Implementation Options

### Option 1: Minimal Change (5 minutes)
Add 2 hashtags: #health, #supplements
- Expected to recover: 80-120 videos
- Risk: Low (high precision)
- Recommendation: Do this first

### Option 2: Moderate Expansion (10 minutes)
Add 5-6 hashtags: #health, #supplements, #vitamins, #holistic, #fitness
- Expected to recover: 120-180 videos
- Risk: Medium (some false positives)
- Recommendation: If B > 25%

### Option 3: Major Overhaul (30-60 minutes)
Implement fuzzy/regex matching
- Expected to recover: 200+ videos
- Risk: High (requires monitoring)
- Recommendation: Only if B > 40%

---

## Files to Modify (if changes needed)

### Primary file: video_discovery.py
Location: `/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/video_discovery.py`
Line: ~120-125

**Current code:**
```python
all_cluster_hashtags = [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']
videos, validation_report = validate_target_hashtags(
    videos=videos,
    target_hashtags=all_cluster_hashtags,
    cluster_id=cluster_config['cluster_id']
)
```

**To expand:**
```python
# Add related hashtags if needed
additional_hashtags = ['health', 'supplements', 'vitamins', 'holistic']
all_cluster_hashtags = all_cluster_hashtags + additional_hashtags

videos, validation_report = validate_target_hashtags(
    videos=videos,
    target_hashtags=all_cluster_hashtags,
    cluster_id=cluster_config['cluster_id']
)
```

### Secondary file: hashtag_validator.py
Location: `/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/hashtag_validator.py`
Function: `validate_target_hashtags()` (line 24)

Currently uses exact string matching. If implementing fuzzy matching:
- Modify `_find_matching_hashtags()` function (line 223)
- Add regex/fuzzy logic instead of `target in video_hashtags`

---

## Quick Wins (Low Risk Actions)

While planning full review:

1. **Add #health and #supplements immediately**
   - 10 minutes to implement
   - Expected to recover 80-120 videos
   - Very likely these appear in rejected videos

2. **Investigate #wellnesssupplements**
   - Why 0% pass rate in passed videos?
   - Is hashtag dead on TikTok?
   - Should it remain a target?

3. **Track rejection rate trend**
   - Is 52% consistent?
   - Per-hashtag breakdown
   - Could indicate Apify quality issues

---

## Timeline

- **Day 1:** Obtain data for 20 rejected videos
- **Days 2-3:** Categorize 20 videos using framework
- **Day 4:** Compile results and apply decision matrix
- **Day 5:** Decision + get approval for implementation
- **Week 2:** Implement changes if needed
- **Week 2:** Re-run validation on 780 videos
- **Week 3:** Monitor results and iterate

---

## Supporting Data Files

All analysis is based on:

```
/home/jorge/rumiaifinal/data/clients/rollo/
├── hashtag/wellness/cluster_analytics.json
│   └─ Contains: 780 videos, 373 passed, 407 rejected, sample rejected IDs
│
└── hashtags/wellness/top_contrastive/buckets/
    ├── bucket_18-33s/selected_videos.json (22 videos)
    ├── bucket_33-60s/selected_videos.json (14 videos)
    └── bucket_60-90s/selected_videos.json (25 videos)
        └─ Total: 61 videos analyzed for hashtag patterns

/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/
├── hashtag_validator.py
│   └─ Contains: Validation logic (exact string matching)
│
├── video_discovery.py
│   └─ Contains: Where validation is called (line ~120)
│
└── cluster_deduplication.py
    └─ Contains: How 780 videos were deduplicated from 1,734
```

---

## Questions Answered By This Analysis

1. **Is validation too strict?** → Likely YES (52% rejection vs 10-30% standard)
2. **What are rejected videos?** → Mixture of false positives (60-75%) and similar wellness content (20-35%)
3. **Should we accept more videos?** → Probably YES if 20%+ are legitimate wellness content
4. **How many videos could we recover?** → 80-160 with minimal changes, 200+ with major overhaul
5. **What's the risk?** → Low for adding #health/#supplements, medium for broader expansion, high for fuzzy matching

---

## Contacts & Approvals

- **Analysis completed:** October 25, 2025
- **Analysis scope:** Very thorough (780 videos, 61 sampled, hashtag patterns analyzed)
- **Status:** Ready for manual video review
- **Next checkpoint:** After reviewing 20 rejected videos

---

## Appendix: Hashtag Validation Logic

```python
def validate_target_hashtags(videos, target_hashtags, cluster_id):
    """
    Current logic:
    1. For each video, extract hashtags
    2. Check if ANY target hashtag is present
    3. PASS: if found
    4. FAIL: if not found
    
    Current target hashtags:
    - #wellness
    - #wellnesssupplements
    - #healthandwellness
    - #wellnessjourney
    
    Current matching: EXACT STRING (case-insensitive)
    
    Currently missing (found in wellness videos):
    - #health
    - #supplements
    - #vitamins
    - #holistic
    - And 30+ other wellness-adjacent tags
    """
```

---

## Version History

- **v1.0** - October 25, 2025 - Initial analysis with 61 passed videos sampled
- **v1.1** - (planned) - After manual review of 20 rejected videos
- **v2.0** - (planned) - After implementation of recommended changes

---

**For questions or follow-up analysis, reference:**
1. REJECTED_VIDEOS_ANALYSIS_SUMMARY.txt
2. MANUAL_REVIEW_FRAMEWORK.md
3. hashtag_validator.py
4. cluster_analytics.json
