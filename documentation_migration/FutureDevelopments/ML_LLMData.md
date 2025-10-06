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
- 1 bucket = N videos × 35 features (N from --video-count, default 100 for contrastive)
- **2 JSONs per bucket** (Random Forest + K-Means) = variable per bucket
- Typical: 3 qualified buckets processed (adaptive bucket processing)
- Estimated total: **3 buckets × variable KB** (depends on N)

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

**Implementation Decision:** Two-call approach (Analysis → Formatting)

**Rationale:**
- Better quality through separation of concerns
- Easier iteration (change formatting without re-analysis)
- Intermediate insights valuable for debugging and reuse
- Only moderate cost increase (24 vs 8 calls per hashtag)
- Full raw data approach for analysis phase (~480KB well within limits)

**See**: [MLPlanning.md](./MLPlanning.md) "ML Analysis Pipeline" for detailed implementation

---

### Multiple Hashtag Comparison

**Data Volume:**
- 3 hashtags × N videos × 35 features (N from --video-count)
- **2 JSONs per bucket** (RF + K-Means) = variable per bucket
- Typical: 3 qualified buckets processed per hashtag (adaptive bucket processing)
- Estimated total: **3 hashtags × 3 buckets × variable KB** (depends on N)
- May exceed 800KB token limit with large N - aggregation may be required

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

## File Architecture

This section documents where ML analysis outputs are stored and how they feed into LLM report generation.

### Input Files (Generated by ML Training)

**Location**: `bucket_{duration}/ml_analysis/`

**Files**:
- `aggregated_features.csv` - Raw aggregated features (N videos × ~35 columns, N from --video-count)
- `rf_transformed.csv` - RF-ready features (N videos × ~39 columns)
- `km_transformed.csv` - KMeans-ready features (N videos × ~40 columns)
- **`random_forest_analysis.json`** - Input to LLM (~30KB)
- **`kmeans_analysis.json`** - Input to LLM (~30KB)

**JSON Schemas**: See detailed structures in "Stage 4: LLM Input Generation" in [MLPlanning.md](./MLPlanning.md)

---

### Output Files (Generated by LLM)

**Location**: `bucket_{duration}/llm_reports/`

**Files**:
- `rf_insights_prompt.txt` - Prompt sent to Claude API (RF analysis)
- `rf_insights_raw.json` - Raw LLM response (RF analysis)
- `rf_insights_report.md` - Formatted RF insights report
- `kmeans_patterns_prompt.txt` - Prompt sent to Claude API (KMeans analysis)
- `kmeans_patterns_raw.json` - Raw LLM response (KMeans analysis)
- `kmeans_patterns_report.md` - Formatted KMeans patterns report

---

### Complete Directory Structure

```
bucket_18-33s/
├── analysis/insights/
│   └── *_temporal_windows_updated.json       # 60 raw feature JSONs
│
├── ml_analysis/                              # ML pipeline outputs
│   ├── aggregated_features.csv
│   ├── rf_transformed.csv
│   ├── km_transformed.csv
│   ├── random_forest_analysis.json           # → Input to LLM Call 1
│   └── kmeans_analysis.json                  # → Input to LLM Call 1
│
├── models/                                    # Trained models
│   ├── random_forest_v1.pkl
│   ├── kmeans_v1.pkl
│   ├── scalers.pkl
│   └── model_metrics.json
│
├── llm_reports/                              # LLM outputs
│   ├── analysis/                             # LLM Call 1 outputs (insight extraction)
│   │   ├── call_1_rf_prompt.txt
│   │   ├── call_1_rf_raw_response.json
│   │   ├── call_1_kmeans_prompt.txt
│   │   ├── call_1_kmeans_raw_response.json
│   │   └── insights.json                     # → Input to LLM Call 2
│   │
│   └── formatted/                            # LLM Call 2 outputs (report generation)
│       ├── call_2_prompt.txt
│       ├── call_2_raw_response.json
│       ├── rf_feature_importance.md
│       ├── strategy_1_the_educator.md
│       ├── strategy_2_visual_storyteller.md
│       ├── strategy_3_personal_journey.md
│       └── bucket_summary.md
│
└── reports/                                   # Final PDFs (no LLM - converted from markdown)
    ├── rf_feature_importance.pdf
    ├── strategy_1_the_educator.pdf
    ├── strategy_2_visual_storyteller.pdf
    ├── strategy_3_personal_journey.pdf
    └── bucket_summary.pdf
```

---

### Data Flow

```
temporal_windows_updated.json (60 files)
    ↓ aggregation & transformation
random_forest_analysis.json + kmeans_analysis.json
    ↓ LLM Call 1: Analysis (extract insights)
insights.json (structured ML insights)
    ↓ LLM Call 2: Formatting (generate reports)
5 markdown reports
    ↓ PDF generation (no LLM)
5 creative strategy reports (PDFs)
```

---

### Integration

**Feature Transforms**: See [FeatureTransformation.md](./FeatureTransformation.md) for complete transformation specifications

**File Locations**: See [MLPlanning.md](./MLPlanning.md) "ML Production Pipeline Architecture" for full directory structure

**Pipeline Stages**: See [MLPlanning.md](./MLPlanning.md) "ML Analysis Pipeline" for end-to-end process

**Scaler Details**: See [Kmeans.md](./Kmeans.md) for K-Means scaler fitting

---

## Future Considerations

- **Prompt optimization:** Test different prompt structures to handle larger payloads efficiently
- **Chunking strategy:** If needed, process buckets sequentially rather than all at once
- **Model upgrades:** Future Claude models may handle larger contexts more reliably
- **Dynamic selection:** Algorithm to choose raw vs aggregated based on hashtag count and video volume
- **Hybrid approach:** Combine aggregated stats with selective example videos for best of both worlds
