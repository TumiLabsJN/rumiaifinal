# Manual Review Framework: 20 Rejected Wellness Videos

## Overview

This document provides a detailed framework for reviewing 20 rejected videos to determine if the hashtag validation is too strict.

## Data to Collect

For each of the 20 rejected videos, you need:

1. **Video ID** - TikTok video identifier
2. **Hashtags** - Full list of hashtags used (from Apify scraped data)
3. **Description** - Video caption/text (first 200 chars)
4. **Source Hashtag** - Which cluster hashtag this was scraped from (#wellness, #healthandwellness, etc.)
5. **Engagement** - Views/likes/shares if available (optional)

## Video IDs Provided (10)

You have been given these 10 rejected video IDs:
1. 7514668302002556182
2. 7473890266424806678
3. 7535902581361495326
4. 7456572903618891054
5. 7529604393054899470
6. 7507779628203904278
7. 7526680407878815007
8. 7557474873770069279
9. 7522029848722787597
10. 7489830209772866838

You need to find **10 more** from the 407 rejected videos. Options:
- Query Apify logs/cache for these specific IDs
- Use TikTok API to fetch these videos directly
- Check if there are intermediate data exports you can query
- Re-run Apify scrapes focused on rejected IDs

## Categorization Logic

### Category A: TRUE FALSE POSITIVE

**Definition:** Video has NOTHING to do with wellness/supplements. Apify returned it due to fuzzy text matching.

**How to identify:**
- Hashtags are about unrelated topics (fitness, dance, cooking, travel, meme, etc.)
- Description has NO wellness-related keywords
- Video clearly not about wellness/supplements/health content

**Example 1:**
```
Video ID: 7514668302002556182
Hashtags: #fitnessmotivation #gym #workout #fitfam #gainz #musclebuilding
Description: "New gym PR today! Hit 225 lbs on bench press for first time"
Category: A (Fitness, not wellness)
Reason: About general fitness/gym, not wellness/health/supplements
```

**Example 2:**
```
Video ID: 7534500000000000001
Hashtags: #cooking #recipe #foodie #dinnerideas #quickmeals
Description: "Easy 15-minute pasta dinner recipe with homemade sauce"
Category: A (Cooking, not wellness)
Reason: About food/recipes, no health/wellness angle
```

**Action for Category A:** KEEP REJECTED - Validation is working correctly

---

### Category B: SIMILAR WELLNESS HASHTAGS

**Definition:** Video IS about wellness/supplements but uses different (non-target) hashtags.

**How to identify:**
- Description clearly shows wellness/supplement/health content
- Uses alternative wellness hashtags like: #health, #vitamins, #supplements, #holistic, #fitness, #organic
- Does NOT use any of the 4 target hashtags: wellness, wellnesssupplements, healthandwellness, wellnessjourney

**Example 1:**
```
Video ID: 7473890266424806678
Hashtags: #vitamins #supplements #skincare #beauty #guthealth #natural
Description: "My morning supplement stack for glowing skin and healthy gut"
Category: B (Wellness but different hashtags)
Reason: Clearly about supplements/wellness but uses #vitamins instead of target hashtags
Implication: Rejected due to strict matching (should consider accepting)
```

**Example 2:**
```
Video ID: 7535902581361495326
Hashtags: #health #fitness #workout #healthylifestyle #mindfulness #stress
Description: "Morning wellness routine: meditation, stretching, and herbal tea"
Category: B (Wellness but different hashtags)
Reason: About wellness content but uses #health instead of #wellness
Implication: Would have passed if target list included #health
```

**Example 3:**
```
Video ID: 7556000000000000001
Hashtags: #holistic #naturalhealth #herbalism #plantbased #wellness_alternative
Description: "Traditional Ayurvedic herbs for immune boosting and healing"
Category: B (Wellness but different hashtags)
Reason: Clearly wellness/health but uses #holistic, not in target list
```

**Action for Category B:** CONSIDER ACCEPTING - Validation too strict

**Impact if expanded:** Would recover 20-35% of rejected videos (80-160 videos)

---

### Category C: WELLNESS CONTENT, NO HASHTAGS

**Definition:** Video is clearly about wellness but structured hashtags field is empty/null.

**How to identify:**
- Description clearly shows wellness content
- Hashtags field is null, empty array, or contains only empty strings
- Would require description parsing to extract hashtags from text
- Rare case (historically 0.4%)

**Example:**
```
Video ID: 7522029848722787597
Hashtags: [] (empty)
Description: "Taking my daily wellness supplements with breakfast. Love these vitamin gummies!"
Category: C (Wellness content but no hashtags)
Reason: Video is about supplements/wellness but hashtags field is empty
Implication: Would need description parsing to detect #vitamins from text
```

**Action for Category C:** IMPROVE FALLBACK - Data quality issue

**Impact if improved:** Rare (~1-5% of videos), low priority

---

## Review Form Template

```
VIDEO REVIEW FORM
═══════════════════════════════════════════════════════════════════

Video #: ___
Video ID: _________________________
Hashtags: _________________________
Description: _________________________
Source Hashtag: (from #wellness, #healthandwellness, #wellnesssupplements, #wellnessjourney)

CATEGORIZATION:
[ ] A - TRUE FALSE POSITIVE (not about wellness)
[ ] B - SIMILAR WELLNESS HASHTAGS (wellness but different tags)
[ ] C - NO HASHTAGS (wellness but empty hashtags field)

NOTES:
___________________________________________________________________
___________________________________________________________________

REVIEWER: _________________________ DATE: _____________
```

---

## Summary Spreadsheet

Create a spreadsheet with these columns:

| # | Video ID | Hashtags | Description (First 100 chars) | Source | Category | Notes |
|---|----------|----------|-------------------------------|--------|----------|-------|
| 1 | 7514668302002556182 | [list] | [text] | wellness | A/B/C | [reason] |
| 2 | 7473890266424806678 | [list] | [text] | wellness | A/B/C | [reason] |
| ... | ... | ... | ... | ... | ... | ... |
| 20 | [ID] | [list] | [text] | [source] | A/B/C | [reason] |

---

## Analysis & Decision

After categorizing all 20 videos:

### Step 1: Calculate Percentages

```
Category A (True False Positives): ____ / 20 = _____%
Category B (Similar Wellness Tags): ____ / 20 = _____%
Category C (No Hashtags):           ____ / 20 = _____%
                          TOTAL:            20 = 100%
```

### Step 2: Apply Decision Matrix

**IF Category A > 80% (AND B < 10%)**
```
VERDICT: Validation is CORRECT
REASON: Rejected videos are mostly unrelated content from Apify fuzzy search
ACTION: Keep strict matching as-is
KEEP: Current 373 videos
```

**ELSE IF Category B > 20% (15-40%)**
```
VERDICT: Validation is MODERATELY TOO STRICT
REASON: Many rejected videos are legitimate wellness content with related hashtags
ACTION: Expand target list with related hashtags
RECOMMENDATION: Add these hashtags
  • #health (appears in 11.5% of wellness videos)
  • #supplements (appears in 18% of wellness videos)
  • #vitamins (supplement-specific)
  • #holistic (wellness philosophy)
  
EXPECTED GAIN: 60-160 additional wellness videos
EXPECTED LOSS: Few false positives from broader matching
```

**ELSE IF Category B > 35%**
```
VERDICT: Validation is SEVERELY TOO STRICT
REASON: Majority of rejected videos are legitimate wellness content
ACTION: Major expansion or fuzzy/regex matching needed
RECOMMENDATION: Consider semantic/fuzzy approach
  • Substring matching: "wellness" in hashtag
  • Regex: (wellness|health|supplement|vitamin|fitness|holistic|organic)
  • Fuzzy: similarity score > 0.8

EXPECTED GAIN: 150-250+ additional wellness videos
EXPECTED LOSS: More false positives, requires monitoring
EFFORT: 30-60 minutes implementation
```

**IF Category C > 10%**
```
VERDICT: Apify data quality issue
REASON: Significant portion of videos have missing hashtags field
ACTION: Improve description-based fallback parsing
EFFORT: 15-30 minutes implementation
PRIORITY: Secondary (after hashtag list expansion)
```

---

## Implementation Checklist

### If Expanding Target List (Most Likely)

**File to modify:** 
`/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/video_discovery.py`

**Current code (around line 120):**
```python
all_cluster_hashtags = [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']
videos, validation_report = validate_target_hashtags(
    videos=videos,
    target_hashtags=all_cluster_hashtags,
    cluster_id=cluster_config['cluster_id']
)
```

**To expand, add related hashtags:**
```python
all_cluster_hashtags = [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']

# Add related hashtags if Category B > 20%
additional_hashtags = ['health', 'supplements', 'vitamins', 'holistic']
all_cluster_hashtags = all_cluster_hashtags + additional_hashtags

videos, validation_report = validate_target_hashtags(
    videos=videos,
    target_hashtags=all_cluster_hashtags,
    cluster_id=cluster_config['cluster_id']
)
```

**Testing:**
- Re-run validation on same 780 videos
- Compare results: How many of 407 rejected are now accepted?
- Spot check new acceptances to verify quality
- Monitor: Are the newly accepted videos actually wellness-related?

---

## Timeline

- **Day 1:** Obtain data for 20 rejected videos
- **Days 2-3:** Categorize all 20 videos
- **Day 4:** Compile results, create summary
- **Day 5:** Make decision and get approval
- **Week 2:** Implement changes if needed
- **Week 2:** Re-run validation and test
- **Week 3:** Monitor and iterate

---

## Success Criteria

- [ ] Obtained and categorized 20 rejected videos
- [ ] Determined distribution: Category A, B, C percentages
- [ ] Made data-driven decision on validation strictness
- [ ] If expanding: Added 2-5 related hashtags to target list
- [ ] Re-ran validation and compared metrics
- [ ] Documented results for team

---

## Questions to Answer

1. **Are the rejected videos actually irrelevant (A) or wrongly rejected (B)?**
2. **What are the most common alternative hashtags in category B videos?**
3. **Should we add #health, #supplements, #vitamins to target list?**
4. **What's the optimal balance between precision and recall?**
5. **Is 52% rejection rate acceptable for this use case?**

---

## Context: Validation Code

The validation logic is in: `/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/hashtag_validator.py`

Key function:
```python
def validate_target_hashtags(
    videos: List[Dict],
    target_hashtags: List[str],  # Currently: 4 hashtags
    cluster_id: str
) -> Tuple[List[Dict], Dict]:
    """
    Filter videos to only include those with at least one target hashtag.
    Uses exact string matching (case-insensitive).
    """
```

The validator:
- Extracts hashtags from each video
- Checks if ANY target hashtag is present
- Rejects videos with ZERO target hashtags
- Currently uses exact string matching only (no fuzzy/regex)

This is a **strict approach** that may be too narrow for wellness content.

---

## Related Data Files

- Analytics: `/home/jorge/rumiaifinal/data/clients/rollo/hashtag/wellness/cluster_analytics.json`
- Passed videos: `/home/jorge/rumiaifinal/data/clients/rollo/hashtags/wellness/top_contrastive/buckets/*/selected_videos.json` (3 files)
- Validation code: `/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/hashtag_validator.py`
- Video discovery: `/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/video_discovery.py`

