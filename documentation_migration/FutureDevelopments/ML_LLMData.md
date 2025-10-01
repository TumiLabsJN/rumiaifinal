# ML LLM Data Strategy

## Overview

This document outlines the high-level data formatting strategies for sending analyzed video data to Claude API for generating insights and comparison reports. The approach differs based on analysis type (single vs comparison) and data volume.

For technical implementation details, JSON schemas, and code examples, see [ML_LLMDataTI.md](ML_LLMDataTI.md).

---

## Context

**Raw Data per Bucket:**
- Each duration bucket (e.g., 18-33s) contains 50-70 videos
- Each video JSON: ~10KB with ~35 features
- Features include temporal window metrics (hook, middle, closing) from 9 ML services

**ML Analysis per Bucket:**
- **2 separate analyses** run on each bucket:
  1. **Random Forest:** Feature importance, predictive patterns for engagement
  2. **K-Means Clustering:** Creative archetypes, cluster performance insights
- Each analysis produces its own JSON output that must be sent to LLM

**LLM Constraints:**
- Claude API token limit: ~200K tokens (≈800KB text)
- Hallucination risk increases with very large context windows
- Trade-off: data richness vs processing reliability

---

## Hashtag Flow

### Single Hashtag Analysis

**Data Volume:**
- 1 bucket = 60 videos × 35 features = ~30KB per JSON
- **2 JSONs per bucket** (Random Forest + K-Means) = ~60KB per bucket
- 8 buckets × 60KB = **~480KB total**

**Options:**

#### Option 1: Full Raw Data

**Approach:**
- Send complete video-level data with all 35 features
- Random Forest JSON includes feature importance rankings
- K-Means JSON includes cluster assignments and centroids
- Separate JSON per analysis type (2 JSONs per bucket)

**Size:** ~30KB per JSON × 2 = **~60KB per bucket**

**Pros:**
- LLM sees all patterns and outliers from both RF and K-Means
- Can identify specific high-performing videos and which cluster they belong to
- Feature importance + cluster membership = nuanced strategic insights

**Cons:**
- Larger payload (~480KB total)
- More tokens consumed

#### Option 2: Aggregated Statistics

**Approach:**
- Combine RF feature importance + K-Means cluster insights
- Replace video-level data with statistical summaries per feature
- Include: mean, std, min, max, median, quartiles, distribution shape
- Single combined JSON per bucket

**Size:** ~6-8KB per bucket × 8 buckets = **~60KB total**

**Pros:**
- Compact, efficient (~60KB vs 480KB)
- Combines RF feature importance + K-Means clusters in one view
- Captures range and distribution shape
- Still preserves outlier information (min/max)

**Cons:**
- Loses individual video patterns
- Cannot identify specific examples

**Verdict:** Both options feasible. Full raw data (~480KB) still well within 800KB limit and provides richer insights.

---

### Multiple Hashtag Comparison

**Data Volume:**
- 3 hashtags × 60 videos × 35 features = ~90KB per JSON
- **2 JSONs per bucket** (RF + K-Means) = ~180KB per bucket
- 8 buckets × 180KB = **~1.44MB total**
- **Exceeds 800KB token limit** - aggregation required

**Options:**

#### Option 1: Full Raw Data

**Approach:**
- Send complete video-level data for all hashtags
- Separate RF and K-Means JSONs per bucket
- Each hashtag includes all videos with 35 features

**Size:** ~90KB per JSON × 2 = **~180KB per bucket**, ~1.44MB for all 8 buckets

**Pros:**
- Complete pattern visibility across hashtags
- LLM can identify cross-hashtag trends in both RF and clustering

**Cons:**
- **Exceeds 800KB token limit** (~1.44MB)
- **High hallucination risk** with very large contexts
- Processing will be slow and unreliable
- Would require sequential bucket processing (16 separate LLM calls)

**Limit:** **Not feasible** for 3+ hashtags with full raw data.

#### Option 2: Aggregated Statistics

**Approach:**
- Combine RF + K-Means insights into single JSON per bucket
- Replace video-level data with statistical summaries
- Per-hashtag feature distributions and cluster insights
- Cross-hashtag comparison in compact format

**Size:** ~20-25KB per bucket, **~200KB for all 8 buckets**

**Pros:**
- Safe for **5-10+ hashtag comparisons**
- Reliable LLM processing (well under 800KB limit)
- Combines RF + K-Means insights in single JSON per bucket
- Focuses on strategic patterns and cluster comparisons
- Preserves distribution insights (min/max/quartiles)

**Cons:**
- Loses granular video-level insights
- Cannot identify specific examples across hashtags

**Verdict:** **Aggregated statistics required** for 3+ hashtags. Full raw data exceeds token limits and risks hallucination.

---

## Recommendations

### Single Hashtag Analysis
- **Use Full Raw Data** (Option 1)
- **Rationale:**
  - Data volume manageable (~480KB for 2 JSONs × 8 buckets)
  - Well under 800KB token limit
  - Raw data provides richer insights for strategic recommendations
  - LLM can identify specific high-performing videos and patterns
- **LLM Calls:** 16 total (2 per bucket: RF + K-Means)

### Multiple Hashtag Comparison
- **MUST use Aggregated Statistics** (Option 2)
- **Rationale:**
  - Full raw data would be ~1.44MB (exceeds 800KB limit)
  - Aggregated stats reduce to ~200KB (safe margin)
  - Preserves strategic insights without granular video data
  - Combines RF feature importance + K-Means clustering in one JSON per bucket
- **LLM Calls:** 8 total (1 combined JSON per bucket)
- **Scalability:** Can handle 5-10+ hashtags with aggregated approach

### Implementation Strategy
- **Single hashtag:** Send 2 separate JSONs per bucket (RF + K-Means)
- **Multi-hashtag:** Combine RF + K-Means insights into 1 aggregated JSON per bucket
- Start with full raw data for single hashtag MVP
- Monitor Claude API performance and adjust if needed
- Consider hybrid for future: aggregated stats + top 5 example videos per hashtag

---

## Data Components

### Random Forest Insights
- **Feature Importance:** Ranked list of features with importance scores (sum = 1.0)
- **Top Performer Patterns:** Text description of what distinguishes high-engagement videos
- **Video-Level Data:** Complete feature vectors for all videos (Option 1 only)

### K-Means Clustering Insights
- **Cluster Distribution:** Count and average engagement per cluster
- **Cluster Definitions:** Defining features that characterize each creative archetype
- **Centroids:** Feature values at cluster centers (Option 1 only)
- **Video Assignments:** Cluster ID per video (Option 1 only)

### Statistical Measures (Aggregated Option)
- **Central Tendency:** Mean, median
- **Spread:** Standard deviation, quartiles
- **Range:** Min, max values
- **Distribution Shape:** Normal, bimodal, skewed, uniform

---

## Future Considerations

- **Prompt optimization:** Test different prompt structures to handle larger payloads efficiently
- **Chunking strategy:** If needed, process buckets sequentially rather than all at once
- **Model upgrades:** Future Claude models may handle larger contexts more reliably
- **Dynamic selection:** Algorithm to choose raw vs aggregated based on hashtag count and video volume
- **Hybrid approach:** Combine aggregated stats with selective example videos for best of both worlds
