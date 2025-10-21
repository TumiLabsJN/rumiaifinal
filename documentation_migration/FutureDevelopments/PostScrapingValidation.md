# Post-Scraping Validation: Hashtag Filtering Fix

> **Issue**: Apify TikTok scraper returns false positives through text-based search
> **Impact**: ~40-60% of scraped videos may lack target hashtags
> **Solution**: Post-scraping hashtag validation filter
> **Date**: 2025-01-17
> **Status**: APPROVED - All 6 critique issues resolved (ready for implementation)
>
> **Design Decisions**:
> - ✅ **Issue #1 RESOLVED**: Validate post-deduplication against ALL cluster hashtags (see Section 11 - Decision #1)
> - ✅ **Issue #2 RESOLVED**: Use strict validation, remove fail-safe mode (see Section 11 - Decision #3)
> - ✅ **Issue #4 RESOLVED**: Add description fallback for empty hashtags field (see Section 11 - Decision #2)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Proposed Solution](#3-proposed-solution)
4. [Implementation Details](#4-implementation-details)
5. [Integration Points](#5-integration-points)
6. [Testing & Validation](#6-testing--validation)
7. [Metrics & Monitoring](#7-metrics--monitoring)
8. [Rollout Plan](#8-rollout-plan)

---

## 1. Problem Statement

### 1.1 Issue Discovery

**Test Case**: Vitamin Supplement Analysis
- **Target Hashtags**: `#vitamin`, `#vitamins`, `#dailyvitamins`, `#vitamintok`
- **Scrape Configuration**: 4 hashtags × 2 runs = 8 scrapes @ 800 videos/scrape
- **Expected**: Videos with target hashtags only
- **Actual**: Mix of vitamin supplements + "VITAMINA TREND" (TikTok animation meme)

**Example False Positive**:
```json
{
  "video_id": "7549981732183887135",
  "description": "VITAMINA TREND || it's so off timing 😭 | Creds to @❓ for the pose idea.",
  "hashtags": [
    {"name": "foryoupage"},
    {"name": "animationmeme"},
    {"name": "jujutsukaisen"},
    {"name": "sugurugeto"},
    {"name": "saturogojo"}
  ]
}
```

**Problem**: Video has ZERO target hashtags but was returned by Apify when searching for `#vitamin`.

### 1.2 Impact Assessment

**False Positive Rate**: Estimated 20-60% depending on query ambiguity
- **Low ambiguity** (e.g., `#vitamind3k2`): ~10-20% false positives
- **Medium ambiguity** (e.g., `#vitamin`): ~30-50% false positives (VITAMINA trend pollution)
- **High ambiguity** (e.g., `#love`, `#health`): ~50-80% false positives

**Downstream Effects**:
1. **Stage 1 Selection**: Incorrect bucket distribution (false positives dilute engagement metrics)
2. **Stage 2 Processing**: Wasted compute resources processing irrelevant videos
3. **Stage 3-7 Analysis**: Polluted ML training data (pattern noise)
4. **Business Impact**: Incorrect insights (recommendations based on anime trends, not supplements)

---

## 2. Root Cause Analysis

### 2.1 Apify Scraper Behavior

**Current Request**:
```python
input_params = {
    "hashtags": ["#vitamin"],  # User expects strict hashtag match
    "resultsPerPage": 800
}
```

**Actual Apify Behavior**: **Fuzzy text search** across ALL fields
- ✅ Searches description text
- ✅ Searches caption text
- ✅ Searches hashtags
- ✅ Matches partial strings ("vitamin" matches "vitamina")
- ❌ Does NOT enforce strict hashtag-only matching

**Evidence**:
1. Apify returned video with description "VITAMINA TREND"
2. Video hashtags: `#foryoupage`, `#animationmeme`, `#jujutsukaisen` (0/4 target hashtags)
3. Apify matched "VITAMINA" (Portuguese/Spanish word) to query "#vitamin"

### 2.2 Why This Happens

**TikTok Search API Design**:
- TikTok's search API prioritizes **engagement + relevance** over strict matching
- "Relevance" includes text similarity across ALL fields, not just hashtags
- Apify scraper wraps TikTok's API and inherits this behavior

**Business Reason** (TikTok's perspective):
- Users searching "vitamin" likely want content ABOUT vitamins, not just tagged with #vitamin
- Text-based search increases result volume and engagement

**Our Requirement** (RumiAI's perspective):
- We need **viral pattern analysis** for specific hashtag communities
- Cross-contamination from unrelated communities (e.g., anime) invalidates insights

---

## 3. Proposed Solution

### 3.1 Solution Overview

**Add post-scraping hashtag validation filter** to Stage 1 (Video Discovery).

**Filter Logic**:
1. Apify scrapes all hashtags for cluster (returns raw videos, may include false positives)
2. Deduplicate videos with provenance tracking
3. **NEW**: Validate each unique video's `hashtags[]` array contains at least one cluster target hashtag
4. Remove videos that fail validation
5. Log removal metrics for monitoring

**Filter Placement**: After deduplication, before bucketing (**DECISION: Issue #1 Resolved**)

**Rationale for Post-Deduplication Validation**:
- **Architectural clarity**: Clean separation of concerns (scrape → dedupe → validate → bucket)
- **Efficiency**: Validates ~1,400 unique videos once instead of ~6,400 videos across 8 scrapes
- **Simplicity**: Scraping functions stay focused (no cluster context needed)
- **Cluster-level validation**: Validates against ALL cluster hashtags, not per-scrape hashtag
- **No over-filtering**: Videos with variant hashtags (e.g., `#vitamins`) pass validation even if found during `#vitamin` scrape

```
ApifyScraper.scrape_videos() × 8 scrapes
    ↓
✅ Apify returns raw videos (6,400 videos)
    ↓
deduplicate_with_provenance()
    ↓
✅ Unique videos (~1,400 videos)
    ↓
🆕 FILTER: validate_hashtags(ALL cluster hashtags)  ← INSERT HERE
    ↓
✅ Validated videos (~900 videos)
    ↓
bucket_by_duration()
```

### 3.2 Design Principles

1. **Strict Validation**: Videos must have verifiable hashtags (structured field or description text) matching cluster targets
   - See Decision #3 (Section 11) for rationale

2. **Description Fallback**: If structured hashtags field is empty, parse description text as safety net for API glitches
   - See Decision #2 (Section 11) for implementation details
   - See Section 4.1 `_extract_hashtags()` for code

3. **Flexible Matching**: Support exact match + case-insensitive + prefix removal
   - See Section 4.1 `_normalize_hashtag()` for normalization logic

4. **Multi-Hashtag Support**: Accept videos with ANY of the target hashtags (OR logic)
   - Videos pass if they have primary OR variant hashtags
   - See Decision #1 (Section 11) for cluster-level validation rationale

5. **Provenance Tracking**: Log which target hashtag(s) matched for analytics
   - See Section 4.3 `cluster_validation.total_cluster_hashtags_found` for metrics

6. **Metrics**: Track filter effectiveness, false positive removal rate, and description fallback usage
   - See Section 4.3 for complete analytics structure
   - See Decision #4 (Section 11) for dual-layer metrics design
   - See Section 7.2 for alert thresholds

---

## 4. Implementation Details

### 4.1 New Module: `hashtag_validator.py`

**Location**: `/ml_pipeline/stage1_discovery/hashtag_validator.py`

```python
"""
Post-Scraping Hashtag Validation

Filters Apify results to ensure videos contain target hashtags.
Addresses fuzzy text search false positives.

Source: PostScrapingValidation.md
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def validate_target_hashtags(
    videos: List[Dict],
    target_hashtags: List[str],
    cluster_id: str
) -> Tuple[List[Dict], Dict]:
    """
    Filter videos to only include those with at least one target hashtag.

    Apify may return false positives via fuzzy text search (e.g., "VITAMINA TREND"
    for query "#vitamin"). This filter enforces strict hashtag matching.

    Args:
        videos: List of video metadata from Apify (ApifyVideoMetadataSchema)
        target_hashtags: List of target hashtags (e.g., ["#vitamin", "#vitamins"])
                        Can include or exclude # prefix
        cluster_id: Cluster ID for logging (e.g., "test_vitamin")

    Returns:
        tuple:
            - filtered_videos: List of videos that passed validation
            - validation_report: Dict with metrics
                {
                    "total_input": int,
                    "passed": int,
                    "removed": int,
                    "removal_rate_pct": float,
                    "removed_video_ids": List[str]
                }

    Example:
        >>> videos = [
        ...     {"id": "123", "hashtags": [{"name": "vitamin"}]},      # ✅ Pass
        ...     {"id": "456", "hashtags": [{"name": "foryoupage"}]}    # ❌ Fail
        ... ]
        >>> filtered, report = validate_target_hashtags(videos, ["#vitamin"], "test")
        >>> len(filtered)
        1
        >>> report["removal_rate_pct"]
        50.0
    """
    # Normalize target hashtags (lowercase, remove # prefix)
    normalized_targets = [_normalize_hashtag(h) for h in target_hashtags]

    logger.info(f"[{cluster_id}] Starting hashtag validation")
    logger.info(f"  Input videos: {len(videos)}")
    logger.info(f"  Target hashtags (normalized): {normalized_targets}")

    filtered_videos = []
    removed_videos = []
    fallback_count = 0  # Track description fallback usage (Decision #2)

    for video in videos:
        video_id = video.get('id', 'unknown')

        # Extract hashtags from video metadata (with description fallback)
        video_hashtags, used_fallback = _extract_hashtags(video)

        if used_fallback:
            fallback_count += 1
            logger.debug(f"  ⚠️  {video_id}: Used description fallback for hashtag extraction")

        # Check if ANY target hashtag is present (OR logic)
        matched_hashtags = _find_matching_hashtags(video_hashtags, normalized_targets)

        if matched_hashtags:
            # PASS: Video has at least one target hashtag
            # Add matched_target_hashtags field for provenance tracking
            video['matched_target_hashtags'] = matched_hashtags
            filtered_videos.append(video)
            logger.debug(f"  ✅ PASS: {video_id} (matched: {matched_hashtags})")
        else:
            # FAIL: Video has zero target hashtags (false positive)
            removed_videos.append(video_id)
            logger.debug(
                f"  ❌ FAIL: {video_id} (has: {video_hashtags[:3]}, "
                f"expected: {normalized_targets})"
            )

    # Generate validation report
    total_input = len(videos)
    passed = len(filtered_videos)
    removed = len(removed_videos)
    removal_rate = (removed / total_input * 100) if total_input > 0 else 0.0
    fallback_rate = (fallback_count / total_input * 100) if total_input > 0 else 0.0

    validation_report = {
        "total_input": total_input,
        "passed": passed,
        "removed": removed,
        "removal_rate_pct": round(removal_rate, 1),
        "removed_video_ids": removed_videos[:10],  # First 10 for debugging
        "description_fallback_count": fallback_count,
        "description_fallback_rate_pct": round(fallback_rate, 1)
    }

    # Log summary
    logger.info(f"[{cluster_id}] Hashtag validation complete:")
    logger.info(f"  ✅ Passed: {passed}/{total_input} ({100-removal_rate:.1f}%)")
    logger.info(f"  ❌ Removed: {removed}/{total_input} ({removal_rate:.1f}%)")

    if fallback_count > 0:
        logger.info(f"  ⚠️  Description fallback used: {fallback_count}/{total_input} ({fallback_rate:.1f}%)")

    if removal_rate > 50:
        logger.warning(
            f"  ⚠️  HIGH REMOVAL RATE: {removal_rate:.1f}% of videos removed. "
            f"This may indicate Apify returning many false positives for cluster '{cluster_id}'. "
            f"Consider refining target hashtags or using more specific queries."
        )

    if fallback_rate > 10:
        logger.warning(
            f"  ⚠️  HIGH FALLBACK USAGE: {fallback_rate:.1f}% of videos used description parsing. "
            f"This may indicate Apify data quality issues (empty/missing hashtags fields). "
            f"Normal rate should be < 5%."
        )

    return filtered_videos, validation_report


def _normalize_hashtag(hashtag: str) -> str:
    """
    Normalize hashtag for comparison.

    - Remove # prefix if present
    - Convert to lowercase
    - Strip whitespace

    Examples:
        "#Vitamin" → "vitamin"
        "vitamins " → "vitamins"
        "DailyVitamins" → "dailyvitamins"
    """
    return hashtag.lstrip('#').strip().lower()


def _extract_hashtags(video: Dict) -> Tuple[List[str], bool]:
    """
    Extract and normalize hashtags from video metadata with description fallback.

    Two-layer extraction strategy (Decision #2):
    1. Try structured hashtags field first (accurate, preferred)
    2. Fallback to description text parsing (safety net for API glitches)

    Handles multiple Apify response formats:
    - Format 1: hashtags: [{"name": "vitamin", "id": "123"}, ...]
    - Format 2: hashtags: ["vitamin", "health", ...]
    - Format 3: hashtags: null/missing → triggers description fallback
    - Format 4: hashtags: [{"name": ""}, ...] → empty strings skipped, triggers fallback if all empty

    Args:
        video: Video metadata dict from Apify

    Returns:
        Tuple of (hashtags_list, used_fallback_boolean)
        - hashtags_list: List of normalized hashtag strings (lowercase, no # prefix)
        - used_fallback_boolean: True if description parsing was used, False if structured field worked
    """
    hashtags_raw = video.get('hashtags', [])

    # Try structured field first
    if hashtags_raw:
        normalized = []

        for h in hashtags_raw:
            if isinstance(h, dict):
                # Format 1: {"name": "vitamin"}
                name = h.get('name', '')
            elif isinstance(h, str):
                # Format 2: "vitamin"
                name = h
            else:
                # Unknown format, skip
                continue

            # Normalize and add (skip empty strings)
            if name and name.strip():
                normalized.append(_normalize_hashtag(name))

        # If we found valid hashtags, return them
        if normalized:
            return normalized, False  # Success with structured data

    # Fallback: Parse description text for hashtags
    # Handles edge cases: empty hashtags field, null field, or all-empty-string entries
    description = video.get('text', '') or video.get('description', '')
    if description:
        import re
        # Extract hashtags from description text (e.g., "#vitamin #health")
        hashtags_from_desc = re.findall(r'#(\w+)', description.lower())
        if hashtags_from_desc:
            normalized_from_desc = [_normalize_hashtag(h) for h in hashtags_from_desc]
            return normalized_from_desc, True  # Fallback used

    # No hashtags found anywhere (structured field or description)
    return [], False


def _find_matching_hashtags(
    video_hashtags: List[str],
    target_hashtags: List[str]
) -> List[str]:
    """
    Find which target hashtags are present in video's hashtags.

    Uses case-insensitive exact matching.

    Args:
        video_hashtags: List of normalized hashtags from video
        target_hashtags: List of normalized target hashtags

    Returns:
        List of target hashtags that matched (empty if no matches)

    Example:
        >>> _find_matching_hashtags(["vitamin", "health"], ["vitamin", "vitamins"])
        ["vitamin"]
        >>> _find_matching_hashtags(["foryoupage"], ["vitamin", "vitamins"])
        []
    """
    matched = []

    for target in target_hashtags:
        if target in video_hashtags:
            matched.append(target)

    return matched
```

### 4.2 Updated `cluster_scraper.py`

**Location**: `/ml_pipeline/stage1_discovery/cluster_scraper.py`

**Changes**:
1. Import new validator
2. Call validator AFTER deduplication at cluster orchestration level
3. Pass ALL cluster hashtags to validator
4. Log validation metrics

```python
from .hashtag_validator import validate_target_hashtags

def scrape_cluster_videos(
    cluster_config: dict,
    apify_scraper: ApifyScraper,
    ...
) -> List[Dict]:
    """
    Orchestrate multi-hashtag scraping with post-deduplication validation.
    """
    logger.info(f"Starting cluster scraping for: {cluster_config['cluster_id']}")

    all_hashtags = [cluster_config['primary_hashtag']] + cluster_config['variant_hashtags']
    all_videos = []

    # STEP 1: Scrape all hashtags (no validation yet)
    for hashtag in all_hashtags:
        for run_num in range(1, cluster_config['scrapes_per_hashtag'] + 1):
            logger.info(f"Scraping {hashtag} (run {run_num})")

            videos = _scrape_with_retry(
                apify_scraper=apify_scraper,
                hashtag=hashtag,
                run_num=run_num,
                ...
            )

            all_videos.extend(videos)

    logger.info(f"Total videos scraped: {len(all_videos)}")

    # STEP 2: Deduplicate (with provenance tracking)
    unique_videos = deduplicate_with_provenance(all_videos)
    logger.info(f"Unique videos after deduplication: {len(unique_videos)}")

    # STEP 3: 🆕 Validate against ALL cluster hashtags (ONCE)
    all_cluster_hashtags = [
        cluster_config['primary_hashtag']
    ] + cluster_config['variant_hashtags']

    validated_videos, validation_report = validate_target_hashtags(
        videos=unique_videos,
        target_hashtags=all_cluster_hashtags,  # All cluster hashtags
        cluster_id=cluster_config['cluster_id']
    )

    logger.info(f"Videos passing validation: {len(validated_videos)}")
    logger.info(
        f"False positives removed: {validation_report['removed']} "
        f"({validation_report['removal_rate_pct']}%)"
    )

    # STEP 4: Continue with bucketing
    return validated_videos
```

**Key Points**:
- ✅ `_scrape_with_retry()` stays simple (just scrapes, no validation)
- ✅ Validation happens once on deduplicated data
- ✅ ALL cluster hashtags passed to validator (no over-filtering)
- ✅ Clear pipeline: scrape → dedupe → validate → bucket

### 4.3 Cluster Configuration Update

**Add validation reporting to cluster analytics**:

**Location**: `cluster_analytics.json` (single source of truth for all cluster metrics)

```python
# cluster_analytics.py - generate_cluster_analytics()

analytics = {
    # ... existing fields (scrape_summary, deduplication_summary, bucket_distribution) ...

    "hashtag_validation": {
        # Overall validation metrics
        "total_scraped": 6400,                    # Total videos from all scrapes
        "total_unique_after_dedup": 1400,         # After deduplication
        "total_validated": 900,                   # After hashtag validation
        "false_positives_removed": 500,           # Videos with NO cluster hashtags
        "false_positive_rate_pct": 35.7,          # (500/1400) * 100

        # Cluster-level validation breakdown
        "cluster_validation": {
            "videos_with_primary_hashtag": 400,           # Has primary hashtag (e.g., #vitamin)
            "videos_with_variant_only": 500,              # Has variant hashtags only (e.g., #vitamins, #dailyvitamins)
            "videos_with_multiple_cluster_hashtags": 100, # Has 2+ cluster hashtags
            "total_cluster_hashtags_found": {
                "#vitamin": 500,         # 500 videos have this hashtag
                "#vitamins": 600,        # 600 videos have this hashtag
                "#dailyvitamins": 200,
                "#vitamintok": 150
            }
        },

        # Per-hashtag scrape quality (Apify search effectiveness)
        "scrape_quality_by_hashtag": {
            "#vitamin": {
                "total_scraped": 1600,                      # Apify returned this many across all runs
                "had_any_cluster_hashtag": 1040,            # Of those, this many have ANY cluster hashtag
                "apify_precision_pct": 65.0,                # 1040/1600 = 65% precision
                "common_false_positive_hashtags": [
                    {"hashtag": "#vitamina", "count": 280},        # VITAMINA megatrend pollution
                    {"hashtag": "#animationmeme", "count": 150},
                    {"hashtag": "#foryoupage", "count": 130}
                ]
            },
            "#vitamins": {
                "total_scraped": 1600,
                "had_any_cluster_hashtag": 1200,
                "apify_precision_pct": 75.0,
                "common_false_positive_hashtags": []        # Clean results
            },
            "#dailyvitamins": {
                "total_scraped": 1600,
                "had_any_cluster_hashtag": 1312,
                "apify_precision_pct": 82.0,
                "common_false_positive_hashtags": []
            },
            "#vitamintok": {
                "total_scraped": 1600,
                "had_any_cluster_hashtag": 880,
                "apify_precision_pct": 55.0,
                "common_false_positive_hashtags": [
                    {"hashtag": "#vitamina", "count": 200}
                ]
            }
        },

        # Quality alerts (informational, non-blocking)
        "scrape_quality_alerts": {
            "warning_threshold_pct": 60,
            "critical_threshold_pct": 40,

            "warnings": [
                {
                    "hashtag": "#vitamintok",
                    "precision_pct": 55.0,
                    "severity": "WARNING",
                    "message": "Low scrape quality (55%). Consider monitoring or refining query.",
                    "false_positive_examples": ["#vitamina", "#animationmeme"]
                }
            ],

            "critical_alerts": [
                {
                    "hashtag": "#vitamin",
                    "precision_pct": 35.0,
                    "severity": "CRITICAL",
                    "message": "Very low scrape quality (35%). High contamination from megatrends. Consider removing from cluster.",
                    "action_required": true,
                    "false_positive_examples": ["#vitamina", "#animationmeme", "#foryoupage"]
                }
            ],

            "overall_assessment": {
                "status": "WARNING",                  # "HEALTHY", "WARNING", or "CRITICAL"
                "contamination_estimate_pct": 45.0,   # % of dataset from critical-precision hashtags
                "recommendation": "Remove #vitamin from cluster config and re-scrape for better data quality."
            }
        },

        # Description fallback usage metrics
        "description_fallback": {
            "videos_using_fallback": 15,              # Videos where hashtags extracted from description
            "fallback_usage_rate_pct": 1.07,          # 15/1400 = 1.07% (should be < 5%)
            "alert": null                             # Alert if > 10% (indicates Apify data quality issue)
        }
    }
}
```

**Key Features**:

1. **Cluster-Level Validation**: Answers "How many valid videos do we have?" and "What's the composition?"
2. **Scrape Quality Metrics**: Enables optimization (identifies which hashtags return low-quality results)
3. **Alert System**: Non-blocking warnings/critical alerts for poor-performing hashtags
4. **Megatrend Detection**: `common_false_positive_hashtags` identifies viral trends polluting results
5. **Actionable Recommendations**: Clear guidance on which hashtags to remove

**Alert Behavior** (Decision: Option A - Non-Blocking):
- Alerts are **informational only** (processing continues regardless)
- CRITICAL alerts logged prominently for user review
- User reviews `cluster_analytics.json` and decides whether to re-run with different hashtags
- Future enhancement: `--strict-validation` flag for blocking behavior (opt-in)

---

## 5. Integration Points

### 5.1 Stage 1: Video Discovery

**File**: `/ml_pipeline/stage1_discovery/cluster_scraper.py`

**Integration Point**: `scrape_cluster_videos()` function (cluster orchestration level)

**Before**:
```python
def scrape_cluster_videos(...):
    # Scrape all hashtags
    all_videos = []
    for hashtag in all_hashtags:
        videos = _scrape_with_retry(...)
        all_videos.extend(videos)

    # Deduplicate
    unique_videos = deduplicate_with_provenance(all_videos)

    # Bucket by duration
    return bucket_by_duration(unique_videos)  # May contain false positives
```

**After**:
```python
def scrape_cluster_videos(...):
    # Scrape all hashtags
    all_videos = []
    for hashtag in all_hashtags:
        videos = _scrape_with_retry(...)
        all_videos.extend(videos)

    # Deduplicate
    unique_videos = deduplicate_with_provenance(all_videos)

    # 🆕 Validate against ALL cluster hashtags
    all_cluster_hashtags = [
        cluster_config['primary_hashtag']
    ] + cluster_config['variant_hashtags']

    validated_videos, validation_report = validate_target_hashtags(
        videos=unique_videos,
        target_hashtags=all_cluster_hashtags,
        cluster_id=cluster_config['cluster_id']
    )

    # Bucket by duration
    return bucket_by_duration(validated_videos)  # Only validated videos
```

**Key Change**: Validation inserted AFTER deduplication, validating against ALL cluster hashtags (not per-scrape)

### 5.2 Analytics & Logging

**Add validation metrics to**:
1. `cluster_analytics.json` - per-hashtag false positive rates
2. Pipeline logs - validation summaries
3. `winner_analysis.json` - note if high removal rate detected

**Example log output**:
```
[2025-01-17 10:30:45] INFO: [#vitamin_run1] Starting hashtag validation
[2025-01-17 10:30:45] INFO:   Input videos: 800
[2025-01-17 10:30:45] INFO:   Target hashtags (normalized): ['vitamin']
[2025-01-17 10:30:47] INFO: [#vitamin_run1] Hashtag validation complete:
[2025-01-17 10:30:47] INFO:   ✅ Passed: 520/800 (65.0%)
[2025-01-17 10:30:47] INFO:   ❌ Removed: 280/800 (35.0%)
[2025-01-17 10:30:47] WARNING: Removed 280 false positives (35.0%) from #vitamin run 1
```

---

## 6. Testing & Validation

### 6.1 Unit Tests

**File**: `/ml_pipeline/tests/test_hashtag_validator.py`

```python
import pytest
from ml_pipeline.stage1_discovery.hashtag_validator import (
    validate_target_hashtags,
    _normalize_hashtag,
    _extract_hashtags,
    _find_matching_hashtags
)

def test_normalize_hashtag():
    """Test hashtag normalization."""
    assert _normalize_hashtag("#Vitamin") == "vitamin"
    assert _normalize_hashtag("vitamins ") == "vitamins"
    assert _normalize_hashtag("DailyVitamins") == "dailyvitamins"
    assert _normalize_hashtag("#") == ""

def test_extract_hashtags_dict_format():
    """Test hashtag extraction from dict format."""
    video = {
        "hashtags": [
            {"name": "vitamin", "id": "123"},
            {"name": "Health", "id": "456"}
        ]
    }
    result = _extract_hashtags(video)
    assert result == ["vitamin", "health"]

def test_extract_hashtags_string_format():
    """Test hashtag extraction from string array format."""
    video = {"hashtags": ["vitamin", "Health"]}
    result = _extract_hashtags(video)
    assert result == ["vitamin", "health"]

def test_extract_hashtags_missing():
    """Test hashtag extraction when field missing."""
    video = {"id": "123"}
    result = _extract_hashtags(video)
    assert result == []

def test_find_matching_hashtags():
    """Test matching logic."""
    video_hashtags = ["vitamin", "health", "wellness"]
    target_hashtags = ["vitamin", "vitamins"]

    matched = _find_matching_hashtags(video_hashtags, target_hashtags)
    assert matched == ["vitamin"]

def test_find_matching_hashtags_no_match():
    """Test no matches."""
    video_hashtags = ["foryoupage", "animationmeme"]
    target_hashtags = ["vitamin", "vitamins"]

    matched = _find_matching_hashtags(video_hashtags, target_hashtags)
    assert matched == []

def test_validate_target_hashtags():
    """Test end-to-end validation."""
    videos = [
        {
            "id": "123",
            "hashtags": [{"name": "vitamin"}]
        },  # ✅ Pass
        {
            "id": "456",
            "hashtags": [{"name": "foryoupage"}]
        },  # ❌ Fail
        {
            "id": "789",
            "hashtags": [{"name": "vitamins"}, {"name": "health"}]
        }  # ✅ Pass
    ]

    filtered, report = validate_target_hashtags(
        videos,
        target_hashtags=["#vitamin", "#vitamins"],
        cluster_id="test"
    )

    assert len(filtered) == 2
    assert report["total_input"] == 3
    assert report["passed"] == 2
    assert report["removed"] == 1
    assert report["removal_rate_pct"] == 33.3
    assert filtered[0]["matched_target_hashtags"] == ["vitamin"]
    assert filtered[1]["matched_target_hashtags"] == ["vitamins"]

def test_validate_target_hashtags_empty_input():
    """Test validation with empty input."""
    filtered, report = validate_target_hashtags([], ["#vitamin"], "test")

    assert len(filtered) == 0
    assert report["total_input"] == 0
    assert report["removal_rate_pct"] == 0.0

def test_validate_target_hashtags_all_pass():
    """Test validation where all videos pass."""
    videos = [
        {"id": "123", "hashtags": [{"name": "vitamin"}]},
        {"id": "456", "hashtags": [{"name": "vitamins"}]}
    ]

    filtered, report = validate_target_hashtags(
        videos,
        target_hashtags=["vitamin", "vitamins"],
        cluster_id="test"
    )

    assert len(filtered) == 2
    assert report["removed"] == 0
    assert report["removal_rate_pct"] == 0.0

def test_validate_target_hashtags_all_fail():
    """Test validation where all videos fail."""
    videos = [
        {"id": "123", "hashtags": [{"name": "foryoupage"}]},
        {"id": "456", "hashtags": [{"name": "animationmeme"}]}
    ]

    filtered, report = validate_target_hashtags(
        videos,
        target_hashtags=["vitamin"],
        cluster_id="test"
    )

    assert len(filtered) == 0
    assert report["removed"] == 2
    assert report["removal_rate_pct"] == 100.0
```

### 6.2 Integration Test

**Test Scenario**: Re-run Vitamin Supplement analysis with validation enabled

**Expected Results**:
- **Before**: 1,939 scraped → 1,400 unique (28% overlap)
- **After**: ~900 scraped (35% removed) → ~600 unique (33% overlap)
- **Quality**: All videos should have at least one of: `#vitamin`, `#vitamins`, `#dailyvitamins`, `#vitamintok`

**Verification Script**:
```python
# scripts/verify_hashtag_validation.py
import json
from pathlib import Path

def verify_bucket_hashtags(bucket_path: Path, target_hashtags: list):
    """Verify all videos in bucket have target hashtag."""
    selected_videos_path = bucket_path / "selected_videos.json"

    with open(selected_videos_path) as f:
        data = json.load(f)

    videos = data['videos']
    normalized_targets = [h.lstrip('#').lower() for h in target_hashtags]

    violations = []

    for video in videos:
        hashtags = [h.get('name', '').lower() for h in video.get('hashtags', [])]
        has_target = any(h in normalized_targets for h in hashtags)

        if not has_target:
            violations.append({
                "video_id": video['id'],
                "description": video.get('text', '')[:100],
                "hashtags": hashtags[:5]
            })

    print(f"Validation Results:")
    print(f"  Total videos: {len(videos)}")
    print(f"  Violations: {len(violations)}")
    print(f"  Validation rate: {(1 - len(violations)/len(videos))*100:.1f}%")

    if violations:
        print(f"\n❌ VIOLATIONS FOUND:")
        for v in violations[:5]:
            print(f"  - {v['video_id']}: {v['description']}...")
            print(f"    Hashtags: {v['hashtags']}")
    else:
        print(f"\n✅ ALL VIDEOS VALID")

    return len(violations) == 0

# Usage:
bucket_path = Path("/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s")
assert verify_bucket_hashtags(bucket_path, ["#vitamin", "#vitamins", "#dailyvitamins", "#vitamintok"])
```

---

## 7. Metrics & Monitoring

### 7.1 Key Metrics

**See Section 4.3 for complete analytics structure and Decision #4 for alert behavior.**

The validation system tracks metrics in `cluster_analytics.json` using a **dual-layer approach**:

#### 1. Cluster-Level Validation Metrics (Outcome Quality)
- `total_scraped`: Total videos from all scrapes (e.g., 6,400)
- `total_unique_after_dedup`: Unique videos after deduplication (e.g., 1,400)
- `total_validated`: Videos passing validation (e.g., 900)
- `false_positives_removed`: Videos with ZERO cluster hashtags (e.g., 500)
- `cluster_validation`: Breakdown by primary vs variant hashtags

**Purpose**: Answers "How many valid videos do we have?" and "What's the dataset composition?"

#### 2. Scrape Quality Metrics (Search Effectiveness)
- `scrape_quality_by_hashtag`: Apify precision % per hashtag
- `common_false_positive_hashtags`: Megatrend detection (e.g., "#vitamina")
- `scrape_quality_alerts`: Warning/critical alerts for low-quality hashtags

**Purpose**: Enables optimization - identifies which hashtags return poor results

#### 3. Description Fallback Metrics (Data Quality Monitoring)
- `description_fallback_count`: Videos using description parsing
- `description_fallback_rate_pct`: Percentage using fallback (should be < 5%)
- Alert if > 10% (indicates Apify data quality issues)

**Full Example**: See Section 4.3 (lines 458-553) for complete JSON structure

### 7.2 Alert Thresholds

**Scrape Quality Precision Thresholds** (defined in Decision #4):

#### Warning Threshold: Apify Precision < 60%
- **Meaning**: 40%+ of Apify results are false positives for this hashtag
- **Action**: Monitor hashtag, consider refining query or using more specific variant
- **Example**: `#vitamintok` at 55% precision → Warning logged in analytics
- **Severity**: 🟡 MEDIUM - Not critical but indicates quality issues

#### Critical Threshold: Apify Precision < 40%
- **Meaning**: 60%+ of Apify results are false positives for this hashtag
- **Action**: Strong recommendation to remove hashtag from cluster configuration
- **Example**: `#vitamin` at 35% precision → Critical alert with recommendation
- **Severity**: 🔴 HIGH - Majority of results are noise, wasting compute resources

#### Alert Behavior (Decision #4 - Option A: Non-Blocking)
- **Processing continues** regardless of alert severity
- CRITICAL alerts logged prominently in both logs and `cluster_analytics.json`
- User reviews analytics and decides whether to re-run with different hashtags
- **Rationale**: First-run can't predict bad hashtags until after scraping; blocking creates poor UX

#### Description Fallback Alert Threshold
- **Warning**: Fallback usage rate > 10%
- **Meaning**: Indicates Apify data quality issues (many empty/missing hashtags fields)
- **Normal Rate**: Should be < 5%
- **Action**: Investigate Apify scraper configuration or API issues

**Note**: These thresholds measure **scrape quality** (Apify search effectiveness), not cluster-level validation quality.

---

## 8. Rollout Plan

### 8.1 Phase 1: Implementation (Week 1)

**Tasks**:
1. Create `hashtag_validator.py` module
2. Write unit tests (target: 100% coverage)
3. Integrate into `cluster_scraper.py`
4. Update cluster analytics schema

**Deliverables**:
- `hashtag_validator.py` (fully tested)
- `test_hashtag_validator.py` (20+ test cases)
- Updated `cluster_scraper.py`
- Updated `cluster_analytics.py`

### 8.2 Phase 2: Testing (Week 2)

**Tasks**:
1. Re-run Vitamin Supplement test case
2. Run verification script
3. Compare before/after metrics
4. Validate business impact (content quality improvement)

**Success Criteria**:
- ✅ 0% hashtag violations in selected videos
- ✅ Validation metrics logged correctly
- ✅ No performance degradation (< 1s added per scrape)
- ✅ False positive removal rate 20-60% (confirms Apify issue)

### 8.3 Phase 3: Production (Week 3)

**Tasks**:
1. Deploy to production pipeline
2. Monitor first 5 production runs
3. Tune alert thresholds based on real data
4. Document learnings in this file

**Monitoring**:
- Track validation metrics for all clusters
- Identify hashtags with consistently high removal rates
- Build allowlist of "high-quality" hashtags (low removal rate)

---

## 9. Future Enhancements

### 9.1 Smart Filtering

**Fuzzy Matching** (for typos/variants):
```python
# Accept "vitamin" for query "#vitamins" (singular/plural)
# Accept "vitaminD3" for query "#vitamind3"

from difflib import SequenceMatcher

def fuzzy_match_hashtag(video_tag: str, target_tag: str, threshold: float = 0.85) -> bool:
    ratio = SequenceMatcher(None, video_tag, target_tag).ratio()
    return ratio >= threshold
```

### 9.2 Description Fallback ✅ IMPLEMENTED (Decision #2)

**Status**: Moved from future enhancement to **core requirement** via Decision #2.

**Rationale** (from Decision #2):
- Protects against rare Apify API glitches (empty/null hashtags field)
- Two-layer extraction strategy: structured field (preferred) → description text (fallback)
- Minimal overhead (~1-2ms per video)
- Handles edge cases: empty string entries, data quality issues

**Implementation**: See Section 4.1 `_extract_hashtags()` function (lines 301-360) for full implementation with description fallback.

**Metrics Tracking**:
- `description_fallback_count`: Videos using description parsing
- `description_fallback_rate_pct`: Percentage using fallback (should be < 5%)
- Alert if > 10% (indicates Apify data quality issues)

**Example**:
```python
# Implemented in Section 4.1
video_hashtags, used_fallback = _extract_hashtags(video)

if used_fallback:
    # Video used description parsing (hashtags field was empty/null)
    logger.debug(f"Used description fallback for {video_id}")
```

**See Also**: Decision #2 (Section 11) for complete design rationale

### 9.3 Whitelist/Blacklist

**Cluster-specific filters**:
```yaml
# cluster_config.yaml
cluster_id: "vitamin_supplement"
target_hashtags:
  - "#vitamin"
  - "#vitamins"

validation_rules:
  blacklist_hashtags:
    - "#vitamina"  # Known false positive (animation trend)
    - "#animationmeme"
  whitelist_keywords:
    - "supplement"
    - "nutrition"
    - "health"
```

---

## 10. Appendices

### 10.1 Example Validation Report

```json
{
  "cluster_id": "test_vitamin",
  "validation_timestamp": "2025-01-17T15:30:45Z",
  "target_hashtags": ["#vitamin", "#vitamins", "#dailyvitamins", "#vitamintok"],

  "scraping_summary": {
    "total_scrapes": 8,
    "total_scraped": 6400,
    "scrape_breakdown": [
      {"hashtag": "#vitamin", "run": 1, "scraped": 800, "validated": 520, "removed": 280},
      {"hashtag": "#vitamin", "run": 2, "scraped": 800, "validated": 540, "removed": 260}
    ]
  },

  "validation_summary": {
    "total_validated": 4200,
    "total_removed": 2200,
    "removal_rate_pct": 34.4,
    "pass_rate_pct": 65.6
  },

  "quality_indicators": {
    "high_removal_rate_detected": true,
    "problematic_hashtags": ["#vitamin"],
    "recommendation": "Consider using more specific hashtags (e.g., #vitamind3, #multivitamin)"
  },

  "sample_removed_videos": [
    {
      "video_id": "7549981732183887135",
      "description": "VITAMINA TREND || it's so off timing 😭",
      "actual_hashtags": ["foryoupage", "animationmeme", "jujutsukaisen"],
      "reason": "No target hashtags present"
    }
  ]
}
```

### 10.2 References

- **Apify TikTok Profile Scraper**: https://apify.com/clockworks/tiktok-profile-scraper
- **VideoDiscoveryCHILDTI.md**: Section 4.1 (Apify scraping implementation)
- **HashtagVolumeV2_TI.md**: Section 3.2 (Cluster scraping orchestration)
- **cluster_scraper.py**: Line 217 (`_scrape_with_retry()`)

---

## 11. Design Decisions Log

### Decision #1: Validation Placement (Issue #1) ✅ RESOLVED

**Date**: 2025-01-20
**Status**: APPROVED

**Problem**:
Original proposal validated per-scrape against single hashtag, causing:
- 30-50% over-filtering of videos with variant hashtags
- Inconsistent results (same video kept/removed depending on scrape order)
- Architectural complexity (cluster_config passed through scraping layer)

**Alternatives Considered**:
1. **Validate per-scrape against current hashtag only** (original proposal)
   - ❌ Over-filters valid videos with variant hashtags
   - ❌ Inconsistent results

2. **Validate per-scrape against ALL cluster hashtags**
   - ❌ Mixed concerns (scraping + validation)
   - ❌ Requires cluster_config in scraping layer
   - ❌ Validates duplicates multiple times

3. **Validate post-deduplication against ALL cluster hashtags** ✅ CHOSEN
   - ✅ Clean separation of concerns
   - ✅ More efficient (validates unique videos once)
   - ✅ Simpler architecture
   - ✅ No over-filtering

**Decision**: Alternative 3 - Post-deduplication validation

**Rationale**:
- **Architectural soundness**: Clean pipeline (scrape → dedupe → validate → bucket)
- **Efficiency**: Validates ~1,400 unique videos once vs ~6,400 total
- **Simplicity**: Scraping functions stay focused, no cluster context needed
- **Correct business logic**: Videos with ANY cluster hashtag pass validation

**Implementation Impact**:
- Validation happens in `scrape_cluster_videos()` orchestration function
- `_scrape_with_retry()` remains simple (no validation logic)
- ALL cluster hashtags passed to validator
- No per-scrape validation overhead

**Next Steps**: Proceed to Issue #4 (Description Fallback)

---

### Decision #2: Description Fallback for Empty Hashtags (Issue #4) ✅ RESOLVED

**Date**: 2025-01-20
**Status**: APPROVED

**Problem**:
Should validator handle edge case when Apify's `hashtags` field is empty/null?

**Discovery Findings**:
```bash
# Stage 1 data (Apify response in selected_videos.json)
{
  "id": "7549981732183887135",
  "description": "VITAMINA TREND #foryoupage #animationmeme",
  "hashtags": [
    {"name": ""},           # Empty string entry (data quality issue)
    {"name": "foryoupage"},
    {"name": "animationmeme"}
  ]
}
```

**Key Insights**:
- ✅ Apify DOES provide hashtags in Stage 1 data (validated via actual files)
- ✅ Hashtags persist through stages 1-3 (Apify → ContentAnalysis-Captions → unified_analysis)
- ❌ Hashtags dropped in Stage 4 (temporal_compute.py bug - separate issue)
- ⚠️ Empty string entries observed in first position (data quality quirk)

**Scenarios**:
1. **Apify API glitch**: `hashtags: []` or `hashtags: null` (rare but possible)
2. **Video legitimately has no hashtags**: Creator didn't add any
3. **Apify data quality**: Empty string entries mixed with real hashtags

**Alternatives Considered**:
1. **Trust Apify hashtags field only** (strict)
   - ✅ Simple implementation
   - ✅ Fast (no regex parsing)
   - ❌ False negatives if API glitches
   - ❌ Misses hashtags if Apify field empty but description has them

2. **Add description fallback as safety net** ✅ CHOSEN
   - ✅ Robust against Apify API glitches
   - ✅ Catches hashtags in description when field is empty
   - ✅ Two-layer extraction (structured → description)
   - ⚠️ Slightly more complex (~10 lines of code)
   - ⚠️ Regex parsing adds minimal overhead (~1-2ms per video)

3. **Hybrid with rate threshold** (complex)
   - Use fallback only if >5% of videos have empty hashtags
   - ❌ Too complex for edge case handling
   - ❌ Harder to test and maintain

**Decision**: Alternative 2 - Description fallback as safety net

**Rationale**:
- **Robustness**: Protects against rare Apify API glitches
- **No false negatives**: Videos with hashtags in description won't be incorrectly removed
- **Minimal cost**: Regex parsing is fast (~1-2ms per video), negligible for 1,400 videos
- **Priority order**: Structured field preferred (accurate), description as fallback (safety net)
- **Data quality**: Handles empty string entries gracefully

**Implementation**:
```python
def _extract_hashtags(video: Dict) -> Tuple[List[str], bool]:
    """
    Extract hashtags with description fallback.

    Returns: (hashtags_list, used_fallback_boolean)
    """
    hashtags_raw = video.get('hashtags', [])

    # Try structured field first
    if hashtags_raw:
        normalized = []
        for h in hashtags_raw:
            name = h.get('name', '') if isinstance(h, dict) else str(h)
            if name and name.strip():  # Skip empty strings
                normalized.append(_normalize_hashtag(name))

        if normalized:
            return normalized, False  # Success with structured data

    # Fallback: Parse description text
    description = video.get('text', '') or video.get('description', '')
    if description:
        import re
        found = re.findall(r'#(\w+)', description.lower())
        return [_normalize_hashtag(h) for h in found], True  # Fallback used

    # No hashtags found anywhere
    return [], False
```

**Metrics to Track**:
- `description_fallback_count`: How many videos used description parsing
- `description_fallback_rate_pct`: Percentage using fallback (should be < 5%)
- Alert if > 10% use fallback (indicates Apify data quality issue)

**Next Steps**: Move Section 9.2 (Description Fallback) from "Future Enhancements" to core implementation in Section 4.1

---

### Decision #3: Fail-Safe Mode vs Strict Validation (Issue #2) ✅ RESOLVED

**Date**: 2025-01-20
**Status**: APPROVED

**Problem**:
Original design principle stated "Fail-Safe: If hashtags field missing/malformed, keep video" but implementation actually removes videos without hashtags. Documentation/code mismatch.

**Alternatives Considered**:
1. **Remove fail-safe from design principles (update docs to match code)** ✅ CHOSEN
   - ✅ Documentation matches implementation
   - ✅ Clear expectations (strict validation)
   - ✅ Ensures data quality
   - ✅ Simpler (no conditional logic)

2. **Implement true fail-safe mode (update code to match docs)**
   - ❌ Allows unverifiable videos into pipeline
   - ❌ False positives leak through
   - ❌ Requires manual review
   - ❌ Adds complexity

3. **Hybrid - conditional fail-safe based on missing rate**
   - ❌ Too complex
   - ❌ Inconsistent behavior
   - ❌ Harder to test

**Decision**: Alternative 1 - Strict validation (remove fail-safe principle)

**Rationale**:
- **Description fallback makes fail-safe unnecessary**: Two-layer extraction (structured → description) already handles API glitches
- **Strict filtering aligns with goals**: Need verifiable, high-quality data for ML training
- **Simpler implementation**: Clear rule (no hashtags = removed), easy to debug
- **Monitoring catches issues**: `description_fallback_rate` metric alerts if API has problems

**Updated Design Principles (Section 3.2)**:
1. **Strict Validation**: Videos must have verifiable hashtags (structured or description)
2. **Description Fallback**: Safety net for empty structured field
3. **Flexible Matching**: Case-insensitive, prefix removal
4. **Multi-Hashtag Support**: OR logic across cluster hashtags
5. **Provenance Tracking**: Log matched hashtags
6. **Metrics**: Track effectiveness and fallback usage

**Implementation Impact**:
- No code changes (already implements strict validation)
- Documentation updated to remove "fail-safe" language
- Section 3.2 rewritten to reflect actual behavior

**Next Steps**: Proceed to Issue #3 (Analytics Structure)

---

### Decision #4: Analytics Structure - Cluster-Level vs Per-Hashtag Metrics (Issue #3) ✅ RESOLVED

**Date**: 2025-01-20
**Status**: APPROVED

**Problem**:
Original analytics design tracked validation metrics per-hashtag, which created misleading data:
- Videos with variant hashtags appeared as "removed" during primary hashtag scrape
- Per-hashtag removal rates didn't reflect true false positive detection
- Couldn't answer key questions: "How many false positives?" vs "Which hashtags perform poorly?"

**Critique Issue #3 Summary**:
- **Video-level provenance**: ✅ Already fixed by Decision #1 (all matched cluster hashtags tracked)
- **Analytics structure**: ❌ Still using misleading per-hashtag metrics

**Alternatives Considered**:

1. **Replace per-hashtag with cluster-level only**
   - ✅ Clear cluster validation metrics
   - ❌ Loses scrape quality data (can't identify bad Apify queries)
   - ❌ Can't optimize future scraping

2. **Keep both - Separate cluster-level + scrape quality metrics** ✅ CHOSEN
   - ✅ Cluster-level metrics answer: "Is our validated dataset high quality?"
   - ✅ Scrape quality metrics answer: "Which hashtags should we use in future?"
   - ✅ Enables iterative improvement (identify and drop low-precision hashtags)
   - ⚠️ More complex structure (two metric categories)

3. **Hybrid - Cluster-level + simplified precision scores**
   - ✅ Simpler than Alternative 2
   - ❌ Loses debugging data (common false positive patterns)
   - ❌ Can't identify megatrends polluting results

**Decision**: Alternative 2 - Dual-layer analytics (cluster-level + scrape quality)

**Rationale**:
- **Business Value**: Cluster metrics measure outcome, scrape metrics enable optimization
- **Megatrend Detection**: `common_false_positive_hashtags` identifies viral trends like "#vitamina"
- **Production-Ready**: Enables data-driven hashtag selection for future scrapes
- **Clear Separation**: Validation quality (cluster) vs search effectiveness (per-hashtag)

**Implementation Details**:

1. **Metrics Location**: `cluster_analytics.json` (single source of truth)
   - All cluster metrics in one file (scrape, dedup, validation, buckets, selection)
   - Simpler debugging and monitoring

2. **Alert Thresholds**:
   - Warning: < 60% precision (indicates quality issues)
   - Critical: < 40% precision (strong recommendation to remove hashtag)

3. **Alert Behavior**: Option A - Non-Blocking (Informational)
   - Processing continues regardless of alert severity
   - CRITICAL alerts logged prominently for user review
   - User decides whether to re-run with different hashtags
   - Rationale: First-run can't predict bad hashtags until after scraping

4. **Provenance Tracking**: By hashtag (not individual run)
   - Simpler implementation
   - Sufficient granularity for optimization decisions

**Analytics Structure** (See Section 4.3 for full implementation):
```python
"hashtag_validation": {
    # Cluster-level validation (outcome metrics)
    "cluster_validation": {
        "videos_with_primary_hashtag": 400,
        "videos_with_variant_only": 500,
        "videos_with_multiple_cluster_hashtags": 100
    },

    # Scrape quality (search effectiveness per hashtag)
    "scrape_quality_by_hashtag": {
        "#vitamin": {
            "apify_precision_pct": 65.0,
            "common_false_positive_hashtags": [
                {"hashtag": "#vitamina", "count": 280}  # Megatrend detection
            ]
        }
    },

    # Quality alerts (non-blocking)
    "scrape_quality_alerts": {
        "warnings": [...],
        "critical_alerts": [...],
        "overall_assessment": {
            "recommendation": "Remove #vitamin and re-scrape for better quality"
        }
    }
}
```

**Implementation Impact**:
- Section 4.3 updated with complete analytics structure
- Cluster-level metrics answer validation quality questions
- Scrape quality metrics enable hashtag optimization
- Alert system provides actionable recommendations (non-blocking)

**Resolution Summary**:
- ✅ Issue #3A (Video provenance): Fixed by Decision #1
- ✅ Issue #3B (Analytics structure): Fixed by this decision (dual-layer metrics)
- ✅ All 6 critique issues now resolved

---

**Document Version**: 1.4
**Last Updated**: 2025-01-20
**Author**: Claude (RumiAI Assistant)
**Reviewed By**: [Pending]
**Status**: APPROVED - All 6 critique issues resolved (Issues #1, #2, #3, #4, #5, #6)
