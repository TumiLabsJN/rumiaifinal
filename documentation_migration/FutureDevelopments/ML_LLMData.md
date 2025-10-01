# ML LLM Data Strategy

## Overview

This document outlines the data formatting strategies for sending analyzed video data to Claude API for generating insights and comparison reports. The approach differs based on analysis type (single vs comparison) and data volume.

---

## Context

**Raw Data per Bucket:**
- Each duration bucket (e.g., 18-33s) contains 50-70 videos
- Each video JSON: ~10KB with ~35 features
- Features include temporal window metrics (hook, middle, closing) from 9 ML services

**LLM Constraints:**
- Claude API token limit: ~200K tokens (≈800KB text)
- Hallucination risk increases with very large context windows
- Trade-off: data richness vs processing reliability

---

## Hashtag Flow

### Single Hashtag Analysis

**Data Volume:**
- 1 bucket = 60 videos × 35 features = ~30KB
- 8 buckets × 30KB = ~240KB total

**Options:**

#### Option 1: Full Raw Data
```json
{
  "bucket": "18-33s",
  "hashtag": "#fitnesstips",
  "video_count": 60,
  "videos": {
    "video_1": {
      "engagement_score": 245000,
      "hook_face_count": 1.8,
      "hook_emotion_joy": 0.65,
      "middle_scene_changes": 2.3,
      "closing_text_density": 0.42
      // ... 35 features total
    },
    "video_2": { ... },
    // ... 60 videos
  }
}
```

**Size:** ~30KB per bucket

**Pros:**
- LLM sees all patterns and outliers
- Can identify specific high-performing videos
- Nuanced insights about distribution shapes

**Cons:**
- Larger payload
- More tokens consumed

#### Option 2: Aggregated Statistics
```json
{
  "bucket": "18-33s",
  "hashtag": "#fitnesstips",
  "video_count": 60,
  "features": {
    "hook_face_count": {
      "mean": 1.8,
      "std": 0.4,
      "min": 0.2,
      "max": 5.0,
      "median": 1.5,
      "quartiles": [1.0, 1.5, 2.3],
      "distribution": "bimodal: 0-1 (30%), 2-3 (50%), 4-5 (20%)"
    },
    "hook_emotion_joy": {
      "mean": 0.65,
      "min": 0.1,
      "max": 0.95,
      "median": 0.68,
      "quartiles": [0.45, 0.68, 0.82]
    }
    // ... 35 features with stats
  },
  "cluster_insights": {
    "cluster_0": {
      "count": 18,
      "avg_engagement": 350000,
      "defining_features": ["high face count", "low text density"]
    },
    "cluster_1": {
      "count": 25,
      "avg_engagement": 180000,
      "defining_features": ["single face", "high emotion joy"]
    }
  }
}
```

**Size:** ~3-4KB per bucket

**Pros:**
- Compact, efficient
- Captures range and distribution shape
- Still preserves outlier information (min/max)

**Cons:**
- Loses individual video patterns
- Cannot identify specific examples

**Verdict:** Both options totally feasible for single hashtag analysis (~240KB well within limits).

---

### Multiple Hashtag Comparison

**Data Volume:**
- 3 hashtags × 60 videos × 35 features = ~90KB per bucket
- 8 buckets × 90KB = ~720KB total

**Options:**

#### Option 1: Full Raw Data
```json
{
  "bucket": "18-33s",
  "comparison": [
    {
      "hashtag": "#fitnesstips",
      "video_count": 62,
      "videos": { /* 62 videos with 35 features each */ }
    },
    {
      "hashtag": "#workoutmotivation",
      "video_count": 58,
      "videos": { /* 58 videos with 35 features each */ }
    },
    {
      "hashtag": "#gymlife",
      "video_count": 65,
      "videos": { /* 65 videos with 35 features each */ }
    }
  ]
}
```

**Size:** ~90KB per bucket, ~720KB for all 8 buckets

**Pros:**
- Complete pattern visibility across hashtags
- LLM can identify cross-hashtag trends

**Cons:**
- Approaching 200K token limit (~800KB)
- **Hallucination risk increases** with very large contexts
- Processing may be slower and less reliable

**Limit:** Feasible for **2-3 hashtags**, but risky beyond that.

#### Option 2: Aggregated Statistics
```json
{
  "bucket": "18-33s",
  "comparison": [
    {
      "hashtag": "#fitnesstips",
      "video_count": 62,
      "features": {
        "hook_face_count": {
          "mean": 1.8,
          "min": 0.2,
          "max": 5.0,
          "median": 1.5,
          "quartiles": [1.0, 1.5, 2.3]
        }
        // ... 35 features with stats
      },
      "cluster_insights": { ... }
    },
    {
      "hashtag": "#workoutmotivation",
      "video_count": 58,
      "features": { ... },
      "cluster_insights": { ... }
    },
    {
      "hashtag": "#gymlife",
      "video_count": 65,
      "features": { ... },
      "cluster_insights": { ... }
    }
  ]
}
```

**Size:** ~12-15KB per bucket, ~120KB for all 8 buckets

**Pros:**
- Safe for **5-10+ hashtag comparisons**
- Reliable LLM processing
- Focuses on strategic patterns, not individual videos

**Cons:**
- Loses granular video-level insights
- Cannot identify specific examples across hashtags

**Verdict:** Both options feasible, but **aggregated stats recommended** for 4+ hashtags to avoid hallucination risk.

---

## Recommendations

### Single Hashtag Analysis
- **Use Full Raw Data** (Option 1)
- Rationale: Data volume is manageable (~240KB), and raw data provides richer insights for strategic recommendations

### Multiple Hashtag Comparison
- **2-3 hashtags:** Full Raw Data acceptable (~720KB, near limit)
- **4+ hashtags:** Use Aggregated Statistics to stay well below token limits and maintain reliability

### Implementation Strategy
- Start with Full Raw Data approach for MVP
- Monitor Claude API performance and hallucination rates
- Switch to Aggregated Statistics if reliability issues emerge
- Consider hybrid approach: send aggregated stats + top 5-10 example videos per hashtag

---

## Future Considerations

- **Prompt optimization:** Test different prompt structures to handle larger payloads efficiently
- **Chunking strategy:** If needed, process buckets sequentially rather than all at once
- **Model upgrades:** Future Claude models may handle larger contexts more reliably
- **Dynamic selection:** Algorithm to choose raw vs aggregated based on hashtag count and video volume
