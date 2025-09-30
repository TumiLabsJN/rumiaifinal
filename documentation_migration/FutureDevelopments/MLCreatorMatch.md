# Creator Match Analysis System

## Overview

### Business Problem
Hiring affiliate creators without understanding their natural content style leads to:
- **High coaching overhead**: Forcing creators to change their rhythm/duration preferences
- **Poor performance**: Creators struggling to produce content outside their comfort zone
- **Wasted resources**: Paying for content that doesn't align with proven viral patterns

### Solution
Analyze a creator's most recent 40 videos to understand their **natural production style**, then cross-check against the client's hashtag/competitor viral patterns to generate a **compatibility score** and **hiring recommendation**.

### Stakeholder Value
- **Tumi Labs**: Data-driven hiring decisions reduce risk and improve affiliate program ROI
- **Brands**: Higher performing affiliate creators with less coaching overhead
- **Creators**: Better matches mean they produce content in their natural style (higher success rate)

---

## Analysis Logic

### 1. Data Collection

**Input**: Creator's TikTok handle (e.g., `@potential_affiliate`)

**Collection Strategy**: Scrape most recent 40 videos via Apify

**Why Recent 40?**
- **Current style**: Shows what creator is actively producing NOW (not what worked 2 years ago)
- **Natural output**: Reveals genuine content rhythm, not just viral outliers
- **Realistic sample**: Most creators have 40+ videos, avoids data scarcity
- **Predictive**: "What will this creator make for us?" vs "What did they do well once?"

**Apify Integration**:
```python
# Use existing Apify TikTok Profile Scraper
# Parameters:
#   - profilesUrls: ["https://www.tiktok.com/@creator_handle"]
#   - resultsPerPage: 40
#   - shouldDownloadVideos: true
#   - shouldDownloadCovers: false
```

---

### 2. Duration Distribution Analysis

**Process**:
1. Run RumiAI analysis on all 40 videos → Get `temporal_windows_updated.json` for each
2. Extract `duration` metadata from each video
3. Bucket videos by duration (8 buckets: 0-3s, 3-9s, 9-13s, 13-18s, 18-33s, 33-60s, 60-90s, 90-120s)
4. Calculate percentage distribution

**Output Example**:
```json
{
  "analysis_type": "recent_40_videos",
  "total_videos_analyzed": 40,
  "date_range": {
    "oldest_video": "2024-12-15",
    "newest_video": "2025-01-28"
  },
  "duration_distribution": {
    "bucket_13-18s": {
      "video_count": 22,
      "percentage": 0.55,
      "avg_engagement": 45000,
      "video_ids": ["7428596413707144481", "7429..."],
      "interpretation": "Primary content style"
    },
    "bucket_18-33s": {
      "video_count": 12,
      "percentage": 0.30,
      "avg_engagement": 38000,
      "interpretation": "Secondary content style"
    },
    "bucket_60-90s": {
      "video_count": 6,
      "percentage": 0.15,
      "avg_engagement": 52000,
      "interpretation": "Occasional long-form (performs well but rare)"
    }
  },
  "dominant_bucket": "13-18s",
  "content_consistency_score": 0.85,  // High = clear preferred duration
  "natural_style_summary": "Creator primarily produces 13-18s content (55%), with secondary 18-33s style (30%). Rarely creates long-form."
}
```

---

### 3. Feature Aggregation (Per Bucket)

For each bucket where creator has videos, calculate average features:

```json
{
  "bucket_13-18s": {
    "video_count": 22,
    "avg_features": {
      // Visual features
      "close_ratio": 0.68,
      "element_count": 24.5,
      "scene_count": 3.2,

      // Audio features
      "speech_coverage": 0.82,
      "word_count": 42,
      "energy_level": 0.65,

      // Behavioral features
      "joy_ratio": 0.35,
      "eye_contact_rate": 0.71,

      // ... all 60+ features averaged
    },
    "feature_variance": {
      "joy_ratio_std": 0.12,  // Low variance = consistent style
      "energy_level_std": 0.18
    }
  }
}
```

---

### 4. Compatibility Scoring

**Formula**:
```
Compatibility Score = (Distribution Match × 0.6) + (Feature Alignment × 0.4)
```

#### 4a. Distribution Match Score

Measures how well creator's natural production aligns with client's viral content distribution.

```python
def calculate_distribution_match(creator_dist, hashtag_viral_dist):
    """
    creator_dist: % of recent 40 videos per bucket
    hashtag_viral_dist: % of viral hashtag content per bucket (from top 40 analysis)

    Returns: 0-1 score (1 = perfect alignment)
    """
    overlap = 0
    for bucket in all_8_buckets:
        creator_pct = creator_dist.get(bucket, 0)
        viral_pct = hashtag_viral_dist.get(bucket, 0)
        overlap += min(creator_pct, viral_pct)  # Intersection of distributions

    return overlap
```

**Example**:
```
Creator Distribution:
  - 13-18s: 55%
  - 18-33s: 30%
  - 60-90s: 15%

Hashtag Viral Distribution (#nutrition):
  - 13-18s: 45%
  - 18-33s: 35%
  - 60-90s: 10%
  - Others: 10%

Distribution Match = min(0.55, 0.45) + min(0.30, 0.35) + min(0.15, 0.10)
                   = 0.45 + 0.30 + 0.10
                   = 0.85 (STRONG MATCH)
```

#### 4b. Feature Alignment Score

For buckets where BOTH creator produces AND hashtag succeeds, compare average features.

```python
def calculate_feature_alignment(creator_features, hashtag_viral_features):
    """
    Compare creator's avg features vs viral content avg features
    Only in overlapping buckets

    Returns: 0-1 score (1 = perfect feature match)
    """
    alignment_scores = []

    for bucket in overlapping_buckets:
        bucket_score = 0
        feature_count = 0

        for feature in important_features:
            creator_val = creator_features[bucket][feature]
            viral_val = hashtag_viral_features[bucket][feature]

            # Normalized difference (0 = identical, 1 = completely different)
            diff = abs(creator_val - viral_val) / max(creator_val, viral_val, 0.01)
            similarity = 1 - min(diff, 1)

            bucket_score += similarity
            feature_count += 1

        alignment_scores.append(bucket_score / feature_count)

    return sum(alignment_scores) / len(alignment_scores)
```

**Example**:
```
Bucket 13-18s:
  Creator joy_ratio: 0.35
  Hashtag viral joy_ratio: 0.62
  Difference: 0.27 / 0.62 = 0.44 (44% different)
  Similarity: 1 - 0.44 = 0.56

  Creator energy_level: 0.68
  Hashtag viral energy_level: 0.72
  Difference: 0.04 / 0.72 = 0.06 (6% different)
  Similarity: 1 - 0.06 = 0.94

  [Average across all features] = 0.72 (Feature Alignment for this bucket)
```

#### 4c. Final Compatibility Score

```json
{
  "compatibility_analysis": {
    "distribution_match": 0.85,
    "feature_alignment": 0.72,
    "final_score": 0.80,  // (0.85 × 0.6) + (0.72 × 0.4) = 0.80
    "interpretation": "Strong fit - creator naturally produces at viral durations with moderately aligned style"
  }
}
```

---

### 5. Hiring Recommendation Tiers

Based on final compatibility score:

| Score Range | Tier | Recommendation | Reasoning |
|-------------|------|----------------|-----------|
| 0.80 - 1.00 | **Tier 1: Immediate Hire** | Highly recommended | Creator naturally makes what works, minimal coaching needed |
| 0.65 - 0.79 | **Tier 2: Coach & Hire** | Recommended with coaching | Creator makes right durations, needs style refinement |
| 0.50 - 0.64 | **Tier 3: Risky** | Proceed with caution | Significant style/duration mismatch, high coaching overhead |
| 0.00 - 0.49 | **Tier 4: Pass** | Not recommended | Fundamental misalignment, better candidates available |

**Output Example**:
```json
{
  "hiring_recommendation": {
    "tier": 1,
    "tier_name": "Immediate Hire",
    "confidence": "HIGH",
    "summary": "Creator's natural style strongly aligns with #nutrition viral patterns. 55% of their content is 13-18s (client's primary success zone). Feature alignment is strong with minor coaching needed on joy_ratio in hook.",
    "strengths": [
      "Produces 85% of content in client's top 2 viral duration buckets",
      "Energy level and speech coverage match viral patterns closely",
      "Consistent style (low feature variance)"
    ],
    "coaching_needs": [
      "Increase joy_ratio in hook from 0.35 to 0.60+ (client viral avg)",
      "Add more text overlays in first 3 seconds"
    ],
    "risk_factors": []
  }
}
```

---

## Architecture

### Directory Structure

```
/data/clients/{client_id}/creators/{creator_handle}/
    ├── videos/                          # Most recent 40 videos (raw MP4s)
    │   └── {video_id}.mp4
    │
    ├── analysis/                        # RumiAI outputs for all 40 videos
    │   ├── insights/                    # temporal_windows JSON (1 per video)
    │   │   └── {video_id}_temporal_windows_updated.json
    │   ├── unified/                     # Intermediate timeline+ml_data (debugging)
    │   └── service_debug/               # emotion_detection, audio_energy outputs
    │
    ├── distribution_analysis/           # Duration distribution insights
    │   ├── duration_distribution.json   # Main output (see Section 2)
    │   │
    │   └── bucket_breakdown/            # Per-bucket feature aggregation
    │       ├── bucket_13-18s/
    │       │   ├── video_list.json      # Which of 40 videos fall here
    │       │   └── avg_features.json    # Average features for creator in this bucket
    │       ├── bucket_18-33s/
    │       └── ... (only buckets where creator has content)
    │
    ├── compatibility_analysis/          # Cross-check vs client patterns
    │   ├── vs_hashtag_{hashtag_name}/
    │   │   ├── distribution_match.json      # Distribution overlap score
    │   │   ├── feature_alignment.json       # Feature-level comparison
    │   │   ├── compatibility_score.json     # Final score + breakdown
    │   │   └── hiring_recommendation.json   # Tier + reasoning
    │   │
    │   └── vs_competitor_{handle}/
    │       └── [same structure]
    │
    ├── creator_summary/                 # Final reports
    │   ├── style_profile.pdf            # "This creator excels at 13-18s, uses moderate energy"
    │   ├── compatibility_report.pdf     # "Strong fit with #nutrition (0.80 score)"
    │   └── creator_metrics.json         # Aggregated stats
    │
    ├── checkpoints/                     # Processing state
    │   └── analysis_state.json
    │
    └── logs/                            # Processing logs
        └── creator_analysis.log
```

---

## Implementation Checklist

### Phase 1: Data Collection & Analysis
- [ ] Extend Apify client to support "recent 40 videos" mode for profile scraping
- [ ] Build batch processor to run RumiAI analysis on 40 videos sequentially
- [ ] Implement checkpoint/resume for creator analysis (40 videos × 80s each = 53 min total)
- [ ] Create duration distribution calculator
- [ ] Build per-bucket feature aggregator

### Phase 2: Compatibility Scoring
- [ ] Implement distribution match algorithm
- [ ] Implement feature alignment algorithm (weighted by feature importance)
- [ ] Build compatibility score calculator (0.6/0.4 weighted formula)
- [ ] Create hiring recommendation tier logic

### Phase 3: Cross-Check Integration
- [ ] Build comparator: creator vs hashtag analysis
- [ ] Build comparator: creator vs competitor analysis
- [ ] Handle edge cases (no overlap buckets, <40 videos available)

### Phase 4: Reporting
- [ ] Design PDF template for style_profile.pdf
- [ ] Design PDF template for compatibility_report.pdf
- [ ] Implement JSON schema for compatibility_scores.json
- [ ] Build report generator using Claude API for narrative generation

### Phase 5: Testing & Validation
- [ ] Unit tests for distribution match algorithm
- [ ] Unit tests for feature alignment algorithm
- [ ] Integration test: full creator analysis pipeline
- [ ] Edge case testing (creator with <40 videos, single-bucket creator, etc.)
- [ ] Validate against real creator data

### Phase 6: Deployment
- [ ] Create CLI command: `python rumiai_creator_match.py --client={id} --creator={handle} --compare-to=hashtag:{name}`
- [ ] Add creator analysis to main MLPlanning pipeline
- [ ] Document API for integration with Tumi Labs' internal tools

---

## Example Outputs

### Scenario: Strong Fit

**Creator**: @fitness_jane
**Client**: #nutrition brand
**Analysis**: Recent 40 videos

```json
{
  "creator_handle": "@fitness_jane",
  "analysis_date": "2025-01-28",
  "videos_analyzed": 40,

  "duration_distribution": {
    "13-18s": 0.55,
    "18-33s": 0.30,
    "60-90s": 0.15
  },

  "compatibility_vs_hashtag_nutrition": {
    "distribution_match": 0.85,
    "feature_alignment": 0.72,
    "final_score": 0.80,
    "tier": 1,
    "recommendation": "Immediate Hire"
  },

  "insights": {
    "strengths": [
      "85% of content in client's top 2 viral buckets",
      "Energy level matches viral patterns (0.68 vs 0.72)",
      "Strong speech coverage (0.82 vs 0.78)"
    ],
    "coaching_needs": [
      "Increase joy_ratio in hook (+0.27 difference)",
      "Add text overlays earlier (element_count timing)"
    ]
  }
}
```

### Scenario: Poor Fit

**Creator**: @storyteller_mike
**Client**: #nutrition brand
**Analysis**: Recent 40 videos

```json
{
  "creator_handle": "@storyteller_mike",
  "analysis_date": "2025-01-28",
  "videos_analyzed": 40,

  "duration_distribution": {
    "60-90s": 0.70,
    "90-120s": 0.20,
    "18-33s": 0.10
  },

  "compatibility_vs_hashtag_nutrition": {
    "distribution_match": 0.20,
    "feature_alignment": 0.65,
    "final_score": 0.38,
    "tier": 4,
    "recommendation": "Pass"
  },

  "insights": {
    "strengths": [
      "Features align well in 18-33s bucket (limited sample)"
    ],
    "risk_factors": [
      "90% of content is 60-120s (client needs 13-33s)",
      "Fundamental content rhythm mismatch",
      "Would require complete restructuring of creator's style"
    ],
    "recommendation_detail": "Creator excels at long-form storytelling (60-120s), but #nutrition viral content is 75% in 13-33s range. Better candidates available who naturally produce short-form."
  }
}
```

---

## Edge Cases & Handling

### 1. Creator Has <40 Videos
**Solution**: Analyze all available videos (minimum 20 required)
- Add `confidence_penalty` based on sample size
- Flag in report: "Analysis based on limited sample (N=25)"

### 2. Creator Changed Style Recently
**Detection**: High variance in duration distribution over time
**Handling**:
- Weight recent 20 videos more heavily (70/30 split)
- Flag: "Creator style appears to be evolving"

### 3. No Overlap with Client Success Buckets
**Example**: Creator makes 60-90s, client's viral content is all 13-18s
**Handling**:
- Distribution match = 0
- Feature alignment = N/A
- Auto-assign Tier 4 (Pass)

### 4. Sparse Bucket Distribution
**Example**: Creator has 1 video in bucket, 38 in another, 1 in third
**Handling**:
- `content_consistency_score` = HIGH (clear preference)
- Only calculate feature alignment for bucket with ≥5 videos

### 5. Multiple Hashtag Comparisons
**Scenario**: Client wants to compare creator against #nutrition AND #fitness
**Handling**:
- Run compatibility analysis for each hashtag separately
- Generate combined report showing best-fit hashtag
- Recommend creator for specific client campaign based on alignment

---

## Future Enhancements

- **Trend Analysis**: Track how creator's style evolves over time (quarterly re-analysis)
- **A/B Testing**: Compare predicted compatibility vs actual performance post-hire
- **Collaborative Filtering**: "Creators similar to @fitness_jane also work well with this client"
- **Real-time Monitoring**: Alert when creator's recent content drifts from original profile