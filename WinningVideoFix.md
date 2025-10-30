# Winning Video Selection Fix

**Issue ID:** Stage1-WinnerAnalysis-001
**Date Discovered:** 2025-10-30
**Status:** Ready for Implementation
**Severity:** Medium (Data Quality Issue)
**Affects:** Stage 1 (Video Discovery & Selection)

---

## Table of Contents

1. [Problem Summary](#problem-summary)
2. [Discovery Process](#discovery-process)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Impact Assessment](#impact-assessment)
5. [Solution Design](#solution-design)
6. [Implementation Plan](#implementation-plan)
7. [Testing Strategy](#testing-strategy)
8. [Appendix](#appendix)

---

## Problem Summary

### Issue Description

The `winner_analysis.json` file generates a `top_100_distribution` field that contains **only 92 videos** instead of the expected 100 videos.

**Example from Test 5 (wellnesspt2_test5):**

```json
{
  "top_100_distribution": {
    "60-90s": 25,
    "18-33s": 12,
    "33-60s": 12,
    "13-18s": 11,
    "3-9s": 10,
    "9-13s": 8,
    "90-120s": 8,
    "0-3s": 6
  },
  "winner_coverage": 53.2608695652174,
  "top_3_buckets": ["60-90s", "18-33s", "33-60s"]
}
```

**Sum:** 25 + 12 + 12 + 11 + 10 + 8 + 8 + 6 = **92 videos** ❌ (expected: 100)

### Business Impact

- **Misleading file naming:** File claims "top_100" but contains 92
- **Incorrect winner_coverage calculation:** Uses 92 as denominator instead of 100
- **Statistical validity concerns:** Analysis based on 92 samples instead of intended 100
- **Inconsistent results:** Different datasets may have different shortfalls

---

## Discovery Process

### Initial Observation

User noticed `winner_analysis.json` looked "wrong" while reviewing Test 5 output files.

### Investigation Steps

**Step 1: Verified the Data**
```bash
# Check actual video count
jq '.top_100_distribution | to_entries | map(.value) | add' winner_analysis.json
# Output: 92
```

**Step 2: Traced Data Flow**

Identified 5 key stages where videos could be filtered:

1. **Scraping** (`apify_scraper.py`)
   - Input: N/A
   - Output: 10,620 videos (Test 5)
   - Deduplication: → 3,593 unique
   - **Sorting:** By `playCount` DESC ✅

2. **Hashtag Validation** (`hashtag_validator.py`)
   - Input: 3,593 sorted videos
   - Removed: 1,625 (45.2% - missing cluster hashtags)
   - Output: 1,968 validated videos
   - **Order preserved:** Still sorted by engagement

3. **Date Filtering** (`date_filter.py`)
   - Input: 1,968 validated videos
   - Removed: 0 (all within 270 days)
   - Output: 1,968 filtered videos
   - **Order preserved**

4. **Winner Analysis - Top 100 Selection** (`winner_analyzer.py`)
   - Input: 1,968 filtered videos
   - Code: `return videos[:TOP_PERFORMERS_FOR_ANALYSIS]`
   - Output: 100 videos (positions 1-100 of filtered list)
   - **Problem identified:** Takes first 100 AFTER 45% removed ⚠️

5. **Bucketing** (`winner_analyzer.py:_bucket_top_performers`)
   - Input: 100 selected videos
   - Skipped: 8 videos (duration=None OR duration>120s)
   - Output: 92 videos in distribution
   - **Second problem:** Videos with invalid durations skipped ⚠️

**Step 3: Code Analysis**

Analyzed the complete Stage 1 pipeline:

- `video_discovery.py` (orchestrator)
- `winner_analyzer.py` (selection logic)
- `apify_scraper.py` (sorting)
- `hashtag_validator.py` (filtering)
- `date_filter.py` (filtering)

**Step 4: Dependency Check**

Verified no downstream impacts:
- `rumiai_ml_batch.py` only uses `winner_analysis['top_3_buckets']`
- No other files depend on the exact count in `top_100_distribution`

---

## Root Cause Analysis

### Primary Root Cause

**File:** `ml_pipeline/stage1_discovery/winner_analyzer.py`
**Method:** `_select_top_performers()` (lines 97-114)
**Issue:** Returns first 100 videos without validating duration

**Problematic code:**
```python
def _select_top_performers(self, videos: List[Dict]) -> List[Dict]:
    if len(videos) < TOP_PERFORMERS_FOR_ANALYSIS:
        return videos
    else:
        logger.info(f"Analyzing top {TOP_PERFORMERS_FOR_ANALYSIS} performers")
        return videos[:TOP_PERFORMERS_FOR_ANALYSIS]  # ❌ Naive slice
```

**What happens:**
1. Takes first 100 videos from validated list
2. Passes to `_bucket_top_performers()`
3. Bucketing encounters 8 videos with invalid durations
4. Skips those 8 videos
5. Result: Only 92 videos in distribution

### Contributing Factors

**Factor 1: Hashtag Validation Removal**
- 45.2% of scraped videos removed (Test 5: 1,625 out of 3,593)
- Creates gaps in the "top 100 by engagement" list
- Position 100 in validated list ≠ Position 100 in original scraped list

**Factor 2: Invalid Duration Handling**
- Some videos lack `duration` field (metadata quality issue)
- Some videos have `duration > 120s` (outside supported buckets)
- Bucketing skips these videos silently

**Factor 3: No Pre-Validation**
- Selection happens BEFORE duration validation
- No guarantee that selected videos will pass bucketing

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ CURRENT FLOW (Broken)                                       │
└─────────────────────────────────────────────────────────────┘

3,593 scraped (sorted by engagement)
    ↓
[Hashtag Validation: Remove 1,625]
    ↓
1,968 validated (still sorted, but gaps created)
    ↓
[Select [:100]] ← Takes positions 1-100 of filtered list
    ↓              NOT positions 1-100 of original list!
100 selected
    ↓
[Bucketing: Skip 8 invalid durations]
    ↓
92 in distribution ❌


┌─────────────────────────────────────────────────────────────┐
│ DESIRED FLOW (Fixed)                                         │
└─────────────────────────────────────────────────────────────┘

3,593 scraped (sorted by engagement)
    ↓
[Hashtag Validation: Remove 1,625]
    ↓
1,968 validated (still sorted)
    ↓
[Iterate: Collect 100 with valid durations]
    ↓
100 selected (all have valid durations)
    ↓
[Bucketing: No skips]
    ↓
100 in distribution ✅
```

---

## Impact Assessment

### Affected Components

**Direct Impact:**
- ✅ `winner_analysis.json` → Contains 92 instead of 100
- ✅ `winner_coverage` calculation → Uses wrong denominator
- ✅ `top_100_distribution` → Misleading name

**Indirect Impact:**
- ⚠️ Statistical analysis → Based on 92 samples not 100
- ⚠️ Bucket selection → Percentages slightly off
- ⚠️ Report generation → May reference "top 100" inaccurately

**NOT Affected:**
- ✅ `top_3_buckets` → Correct (based on distribution percentages)
- ✅ Stage 2 processing → Uses `selected_videos.json` (100 videos per bucket)
- ✅ ML training → Not dependent on winner_analysis.json

### Severity Assessment

**Classification:** Medium Priority

**Rationale:**
- Does not break pipeline execution
- Does not affect video processing (Stage 2+)
- Does not affect final ML models
- DOES affect data quality and transparency
- DOES create misleading analytics

**Recommendation:** Fix in next pipeline run, not urgent hotfix

---

## Solution Design

### Selected Approach: Option A

**Ensure exactly 100 videos with valid durations are selected**

### Design Rationale

**Why Option A?**
1. ✅ **Cleanest solution:** Pre-validates durations before selection
2. ✅ **Guaranteed 100 videos:** No more shortfalls
3. ✅ **Minimal code changes:** Single method modification
4. ✅ **No breaking changes:** Output schema unchanged
5. ✅ **Performance:** Negligible overhead (< 0.1s)

**Rejected Alternatives:**

**Option B: Over-select with buffer**
```python
# Select 120 videos, trim to 100 after bucketing
return videos[:TOP_PERFORMERS_FOR_ANALYSIS + 20]
```
- ❌ Arbitrary buffer size (what if 20 isn't enough?)
- ❌ Still requires post-processing
- ❌ Less transparent

**Option C: Fix in post-processing**
```python
# If < 100 after bucketing, select more videos
```
- ❌ Complex retry logic
- ❌ Multiple passes through data
- ❌ Harder to reason about

### Solution Architecture

**Core Change:** Modify `_select_top_performers()` to validate durations DURING selection

**Pseudocode:**
```python
def _select_top_performers(videos):
    selected = []
    skipped = 0

    for video in videos:
        if video.duration is None:
            skipped += 1
            continue

        try:
            bucket = assign_bucket(video.duration)  # Validates 0-120s
            selected.append(video)

            if len(selected) >= 100:
                break
        except ValueError:
            skipped += 1  # duration > 120s
            continue

    return selected  # Guaranteed 100 videos (if available)
```

**Key Features:**
- ✅ Iterates through sorted validated videos
- ✅ Checks duration validity inline
- ✅ Stops at exactly 100 valid videos
- ✅ Handles edge case: < 100 valid videos available

---

## Implementation Plan

### Files to Modify

#### **1. PRIMARY: winner_analyzer.py**

**File:** `/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/winner_analyzer.py`
**Lines:** 97-114
**Method:** `_select_top_performers()`

**Current code:**
```python
def _select_top_performers(self, videos: List[Dict]) -> List[Dict]:
    """
    Select top performers for analysis.

    Normal mode: top 100 if ≥100 available
    Degraded mode: all videos if < 100 available
    """
    if len(videos) < TOP_PERFORMERS_FOR_ANALYSIS:
        # Degraded mode
        logger.warning(
            f"Small dataset ({len(videos)} videos). Analyzing all available. "
            f"Statistical validity may be limited. Recommended: ≥100 videos."
        )
        return videos
    else:
        # Normal mode
        logger.info(f"Analyzing top {TOP_PERFORMERS_FOR_ANALYSIS} performers")
        return videos[:TOP_PERFORMERS_FOR_ANALYSIS]
```

**New code:**
```python
def _select_top_performers(self, videos: List[Dict]) -> List[Dict]:
    """
    Select top performers ensuring exactly TOP_PERFORMERS_FOR_ANALYSIS (100)
    videos with valid durations.

    Iterates through sorted videos (by engagement DESC) and collects
    videos with valid durations (not None, within 0-120s range) until
    reaching target count or exhausting available videos.

    Normal mode: Collect 100 videos with valid durations
    Degraded mode: Return all available if < 100 valid videos exist

    Args:
        videos: Filtered videos sorted by engagement DESC

    Returns:
        List of top performer videos (up to TOP_PERFORMERS_FOR_ANALYSIS)
        All returned videos have valid durations for bucketing
    """
    if len(videos) < TOP_PERFORMERS_FOR_ANALYSIS:
        # Degraded mode: Not enough videos total
        logger.warning(
            f"Small dataset ({len(videos)} videos). Analyzing all available. "
            f"Statistical validity may be limited. Recommended: ≥100 videos."
        )
        return videos

    # Normal mode: Select exactly 100 videos with valid durations
    selected = []
    skipped_invalid_duration = 0

    for video in videos:
        duration = video.get("duration")

        # Check 1: Duration exists
        if duration is None:
            skipped_invalid_duration += 1
            continue

        # Check 2: Duration within valid range (0-120s)
        try:
            bucket = assign_bucket(duration)  # Raises ValueError if > 120s
            selected.append(video)

            # Stop when we have exactly 100
            if len(selected) >= TOP_PERFORMERS_FOR_ANALYSIS:
                break

        except ValueError:
            # Duration > 120s (invalid for our bucket system)
            skipped_invalid_duration += 1
            continue

    # Log results
    logger.info(f"Analyzing top {TOP_PERFORMERS_FOR_ANALYSIS} performers")

    if skipped_invalid_duration > 0:
        logger.info(
            f"Skipped {skipped_invalid_duration} videos with invalid/missing duration "
            f"during top performer selection"
        )

    # Handle edge case: Not enough valid videos
    if len(selected) < TOP_PERFORMERS_FOR_ANALYSIS:
        logger.warning(
            f"Only {len(selected)} videos with valid durations available "
            f"(target: {TOP_PERFORMERS_FOR_ANALYSIS}). Analyzing {len(selected)} videos. "
            f"Statistical validity may be limited."
        )

    return selected
```

**Changes summary:**
- ✅ Added duration validation loop
- ✅ Uses existing `assign_bucket()` function for validation
- ✅ Improved logging (reports skipped videos)
- ✅ Handles edge case (< 100 valid videos)
- ✅ Updated docstring

---

#### **2. OPTIONAL: video_discovery.py**

**File:** `/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/video_discovery.py`
**Lines:** 13, 271-273
**Purpose:** Fix `winner_coverage` calculation

**Add import (line 13):**
```python
from .constants import TOP_PERFORMERS_FOR_ANALYSIS
```

**Fix calculation (lines 271-273):**

**Current:**
```python
"winner_coverage": sum(
    winner_distribution.get(b, 0) for b in winning_buckets
) / sum(winner_distribution.values()) * 100,
```

**Fixed:**
```python
"winner_coverage": sum(
    winner_distribution.get(b, 0) for b in winning_buckets
) / TOP_PERFORMERS_FOR_ANALYSIS * 100,
```

**Why?**
- Current uses `sum(winner_distribution.values())` = 92 as denominator
- Should use `TOP_PERFORMERS_FOR_ANALYSIS` = 100 as denominator
- Coverage should be "X out of 100", not "X out of 92"

---

### Implementation Steps

1. **Backup current code**
   ```bash
   cp ml_pipeline/stage1_discovery/winner_analyzer.py \
      ml_pipeline/stage1_discovery/winner_analyzer.py.backup
   ```

2. **Apply Change 1: winner_analyzer.py**
   - Replace `_select_top_performers()` method (lines 97-114)
   - No import changes needed (`assign_bucket` already imported)

3. **Apply Change 2: video_discovery.py** (optional)
   - Add import: `from .constants import TOP_PERFORMERS_FOR_ANALYSIS`
   - Fix winner_coverage calculation (lines 271-273)

4. **Verify syntax**
   ```bash
   python3 -m py_compile ml_pipeline/stage1_discovery/winner_analyzer.py
   python3 -m py_compile ml_pipeline/stage1_discovery/video_discovery.py
   ```

5. **Run linter** (optional)
   ```bash
   flake8 ml_pipeline/stage1_discovery/winner_analyzer.py
   ```

---

## Testing Strategy

### Unit Tests (Recommended)

**Test Case 1: Normal operation - all videos valid**
```python
def test_select_top_performers_all_valid():
    videos = [{"id": str(i), "duration": 30} for i in range(150)]
    analyzer = WinnerAnalyzer()
    result = analyzer._select_top_performers(videos)

    assert len(result) == 100  # Exactly 100
    assert result[0]["id"] == "0"  # First video selected
    assert result[99]["id"] == "99"  # 100th video selected
```

**Test Case 2: Some videos have invalid durations**
```python
def test_select_top_performers_with_invalid():
    videos = [
        {"id": "0", "duration": 30},   # Valid
        {"id": "1", "duration": None},  # Invalid - None
        {"id": "2", "duration": 150},   # Invalid - > 120s
        {"id": "3", "duration": 60},    # Valid
        # ... 100+ more videos
    ]
    analyzer = WinnerAnalyzer()
    result = analyzer._select_top_performers(videos)

    assert len(result) == 100  # Exactly 100
    assert all(v.get("duration") is not None for v in result)  # All have duration
    assert all(v.get("duration") <= 120 for v in result)  # All within range
```

**Test Case 3: Not enough valid videos**
```python
def test_select_top_performers_insufficient():
    videos = [{"id": str(i), "duration": 30} for i in range(50)]
    analyzer = WinnerAnalyzer()
    result = analyzer._select_top_performers(videos)

    assert len(result) == 50  # Returns all available
    # Should log warning about insufficient videos
```

### Integration Test

**Test with Test 5 data** (or new test run):

1. **Setup:**
   ```bash
   # Clear Stage 1 output
   rm -f data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/winner_analysis.json
   rm -f data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/checkpoints/stage_1_checkpoint.json
   ```

2. **Run Stage 1 only:**
   ```bash
   python rumiai_ml_batch.py --client Rollo_Test5 --target wellnesspt2_test5 \
     --analysis-type hashtag --selection-strategy contrastive \
     --video-count 100 --date-filter last_270_days
   # Abort after Stage 1 complete
   ```

3. **Verify output:**
   ```bash
   # Check video count in distribution
   jq '.top_100_distribution | to_entries | map(.value) | add' \
     data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/winner_analysis.json
   # Expected: 100

   # Check winner_coverage uses correct denominator
   jq '.winner_coverage' \
     data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/winner_analysis.json
   # Expected: ~53.0 (not 53.26)
   ```

4. **Check logs:**
   ```bash
   grep "Skipped.*videos with invalid" logs/rumiai_ml_*.log
   # Should see: "Skipped 8 videos with invalid/missing duration during top performer selection"
   ```

### Acceptance Criteria

✅ **Must Have:**
- [ ] `sum(top_100_distribution.values()) == 100`
- [ ] `winner_coverage` uses denominator of 100
- [ ] Log shows "Skipped N videos" message (if applicable)
- [ ] No regression in downstream stages (Stage 2-7)

✅ **Nice to Have:**
- [ ] Unit tests pass
- [ ] No linter warnings
- [ ] Performance < 0.1s overhead

---

## Appendix

### A. Test 5 Detailed Metrics

**Cluster:** wellnesspt2_test5 (9 hashtags)
**Scrape Date:** 2025-10-29
**Total Scrapes:** 36 (4 runs × 9 hashtags)

**Stage 1 Flow:**
```
10,620 videos scraped
  → 3,593 unique (deduplication)
  → 1,968 validated (hashtag validation: -1,625, -45.2%)
  → 1,968 date filtered (date filter: -0)
  → 100 selected (winner analysis)
  → 92 bucketed (bucketing: -8 invalid durations)
```

**Winning Buckets:**
1. bucket_60-90s: 25/92 = 27.2%
2. bucket_18-33s: 12/92 = 13.0%
3. bucket_33-60s: 12/92 = 13.0%

**Winner Coverage:** 49/92 = 53.26%

---

### B. Code References

**Primary files:**
- `ml_pipeline/stage1_discovery/winner_analyzer.py:97-114` (fix location)
- `ml_pipeline/stage1_discovery/video_discovery.py:271-273` (optional fix)

**Related files:**
- `ml_pipeline/stage1_discovery/apify_scraper.py:326` (sorting by engagement)
- `ml_pipeline/stage1_discovery/hashtag_validator.py:24-143` (validation)
- `ml_pipeline/stage1_discovery/constants.py:56` (TOP_PERFORMERS_FOR_ANALYSIS)
- `foundation/buckets.py` (assign_bucket function)

---

### C. Edge Cases Matrix

| Scenario | Input | Expected Output | Handled? |
|----------|-------|----------------|----------|
| All videos valid | 150 videos, all duration 30s | First 100 selected | ✅ Yes |
| 8 invalid in first 100 | 150 videos, 8 with duration=None | Skip 8, select next 8 → 100 total | ✅ Yes |
| 50 invalid scattered | 150 videos, 50 with duration>120s | Skip 50, select 100 valid | ✅ Yes |
| Only 80 valid total | 150 videos, only 80 valid | Return 80, log warning | ✅ Yes |
| Exactly 100 valid | 100 videos, all valid | Return all 100 | ✅ Yes |
| < 100 videos total | 50 videos | Degraded mode, return 50 | ✅ Yes |

---

### D. Performance Analysis

**Current Implementation:**
- Time: O(1) - Simple slice `videos[:100]`
- Space: O(1) - No additional memory

**New Implementation:**
- Time: O(n) where n = videos scanned (typically 100-150)
- Space: O(100) - Stores up to 100 selected videos
- Overhead: ~50 extra duration checks in worst case
- Impact: < 0.1s (negligible)

**Justification:**
- Stage 1 runs once per pipeline (infrequent)
- Correctness > micro-optimization
- Overhead negligible compared to Apify scraping (minutes)

---

### E. Rollback Plan

If issues discovered after deployment:

1. **Immediate rollback:**
   ```bash
   mv ml_pipeline/stage1_discovery/winner_analyzer.py.backup \
      ml_pipeline/stage1_discovery/winner_analyzer.py
   ```

2. **Re-run affected pipelines:**
   ```bash
   # Clear Stage 1 checkpoint
   rm data/clients/*/hashtags/*/top_contrastive/checkpoints/stage_1_checkpoint.json

   # Re-run pipeline
   python rumiai_ml_batch.py --client <CLIENT> --target <TARGET> ...
   ```

3. **Risk:** Low - Changes isolated to single method

---

### F. Related Issues

**Future Enhancements:**

1. **Issue:** Why do 8 videos have invalid durations?
   - Root cause: Apify data quality issue OR TikTok videos > 120s
   - Resolution: Filter during scraping OR expand bucket definitions

2. **Issue:** 45.2% hashtag validation removal rate
   - Root cause: Apify fuzzy search returns false positives
   - Resolution: Improve Apify query specificity OR refine hashtag list

3. **Issue:** File named "top_100_distribution" but contains variable count
   - Resolution: Rename to "top_performers_distribution" OR enforce 100 guarantee (this fix)

---

### G. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-30 | Claude | Initial documentation - problem discovery and fix design |

---

### H. Sign-Off

**Prepared by:** Claude (AI Assistant)
**Reviewed by:** _Pending_
**Approved by:** _Pending_
**Implementation Date:** _Pending_

---

**End of Document**
