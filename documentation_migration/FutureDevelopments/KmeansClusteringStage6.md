# ML Model Architecture: Dual Random Forest + Window-Level K-Means

> **Parent Document**: MLPlanningv2.md Stage 6 "ML Analysis Generation"
> **Date**: 2025-01-10
> **Status**: APPROVED - Dual RF + Window-Level K-Means Architecture

---

## Document Overview

This document defines the complete ML model architecture for Stage 6, covering:
1. **Dual Random Forest Architecture** (8 video-level + 41 window-level models)
2. **Window-Level K-Means Architecture** (41 models)
3. **Total: 90 ML models** (49 RF + 41 K-Means)

**Key Architectural Decision**: Both cross-window patterns AND within-window patterns are crucial for understanding viral TikTok videos. Dual RF ensures complete pattern coverage with no blind spots.

---

# Part 1: Dual Random Forest Architecture

## Overview

To capture **BOTH** cross-window patterns AND within-window patterns, Stage 6 trains **two types of Random Forest models**:

1. **Video-Level RF** (8 models) - Detects cross-window interactions
2. **Window-Level RF** (41 models) - Validates within-window features

**Total RF Models: 49** (8 video-level + 41 window-level)

---

## Rationale: Why Dual RF?

### The Problem

**Cross-window patterns AND within-window patterns are BOTH crucial** for understanding viral TikTok videos.

**Examples of Cross-Window Patterns:**
- Sequential patterns: "Energy builds from hook → middle → closing"
- Consistency patterns: "Hook topic matches middle topic"
- Contrast effects: "Closing energy is impactful when middle is calm"
- Weak link detection: "One bad window kills virality"

**Examples of Within-Window Patterns:**
- Which features define a "strong hook" vs "weak hook"
- Which features define a "strong middle" vs "weak middle"
- Which features define a "strong closing" vs "weak closing"

**The Solution**: Train BOTH types of RF models to capture all patterns.

---

## RF Type 1: Video-Level Random Forest (Cross-Window Patterns)

### Architecture

- **Models**: 8 total (1 per bucket)
- **Input**: All windows concatenated (129-150 features depending on bucket)
- **Purpose**: Detect cross-window interactions and temporal progressions
- **Output**: Feature importance across entire video journey

### What It Captures

1. **Sequential Patterns**
   - Example: "Energy builds from hook → middle → closing predicts virality"
   - Feature: `hook_to_middle_energy_delta` (energy change from hook to middle average)
   - RF importance: 0.12 (top performers: +0.15, bottom performers: -0.08)

2. **Consistency Patterns**
   - Example: "Hook topic matches middle topic increases viral rate by 35%"
   - Feature: `eye_contact_consistency` (std deviation across all windows)
   - RF importance: 0.08 (top performers: 0.12 std, bottom performers: 0.35 std)

3. **Contrast Effects**
   - Example: "Large energy gap between middle avg and closing peak predicts virality"
   - Feature: `middle_to_closing_contrast` (energy gap)
   - RF importance: 0.10 (top performers: 0.28 gap, bottom performers: 0.05 gap)

4. **Weak Link Detection**
   - Example: "Videos with strong hooks and middles but weak closings still fail"
   - Feature: `closing_energy_max` with negative coefficient when other windows are strong
   - RF importance: 0.15

### Training Input

**Bucket 18-33s Example:**
```python
# Load RF-transformed data (all windows concatenated)
X = pd.read_csv('ml_analysis/rf_transformed.csv')  # Shape: (100 videos, 190 features)

# Features include:
# - hook_eye_contact_rate, hook_word_count, ... (30 hook features)
# - middle_1_eye_contact_rate, middle_1_word_count, ... (30 middle_1 features)
# - middle_2_*, middle_3_*, middle_4_* (30 features each)
# - closing_* (30 closing features)
# - hour, day_of_week, is_weekend, is_business_hours (4 temporal features)
# - gender_male, gender_female (2 categorical features)

y = X['is_top_performer']  # Binary labels (top 80% vs bottom 20%)
X = X.drop(['is_top_performer'], axis=1)

# Train video-level RF
rf_video = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_video.fit(X, y)

# Save model
joblib.dump(rf_video, 'models/rf_video_18-33s.pkl')
```

### Output: Video-Level RF Analysis JSON

**File**: `ml_analysis/rf_video_analysis.json`

```json
{
  "model_type": "video_level_rf",
  "bucket": "18-33s",
  "total_videos": 100,
  "input_features": 190,
  "model_performance": {
    "accuracy": 0.87,
    "precision": 0.89,
    "recall": 0.84
  },
  "feature_importance": [
    {
      "feature": "hook_eye_contact_rate",
      "importance": 0.22,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "rank": 1,
      "pattern_type": "single_window"
    },
    {
      "feature": "middle_3_word_count",
      "importance": 0.18,
      "top_performer_avg": 52,
      "bottom_performer_avg": 28,
      "gap": 24,
      "rank": 2,
      "pattern_type": "single_window"
    },
    {
      "feature": "closing_energy_max",
      "importance": 0.15,
      "top_performer_avg": 0.92,
      "bottom_performer_avg": 0.57,
      "gap": 0.35,
      "rank": 3,
      "pattern_type": "single_window"
    },
    {
      "feature": "hook_to_middle_energy_delta",
      "importance": 0.12,
      "interpretation": "Energy change from hook to middle average",
      "top_performer_avg": 0.15,
      "bottom_performer_avg": -0.08,
      "gap": 0.23,
      "rank": 4,
      "pattern_type": "cross_window"
    },
    {
      "feature": "middle_to_closing_contrast",
      "importance": 0.10,
      "interpretation": "Energy gap between middle avg and closing peak",
      "top_performer_avg": 0.28,
      "bottom_performer_avg": 0.05,
      "gap": 0.23,
      "rank": 5,
      "pattern_type": "cross_window"
    },
    {
      "feature": "eye_contact_consistency",
      "importance": 0.08,
      "interpretation": "Std deviation of eye contact across all windows",
      "top_performer_avg": 0.12,
      "bottom_performer_avg": 0.35,
      "gap": 0.23,
      "rank": 6,
      "pattern_type": "cross_window"
    }
    // ... top 15-20 features
  ],
  "cross_window_insights": [
    "Energy progression (hook → middle → closing) has 0.12 importance",
    "Eye contact consistency across windows has 0.08 importance",
    "Closing contrast effect (vs middle) has 0.10 importance"
  ]
}
```

**Key Difference from Window-Level RF**: Includes **cross-window features** like `hook_to_middle_energy_delta`, `middle_to_closing_contrast` that don't exist in window-level analysis.

---

## RF Type 2: Window-Level Random Forest (Within-Window Patterns)

### Architecture

- **Models**: 41 total (1 per window per bucket, same count as K-Means)
- **Input**: 21 features per window (separate models for each window type)
- **Purpose**: Validate which features matter within each specific window type
- **Output**: Feature importance per window (perfectly aligns with K-Means clusters)

### What It Captures

- Which features define a "strong hook" vs "weak hook"
- Which features define a "strong middle" vs "weak middle"
- Which features define a "strong closing" vs "weak closing"
- Direct validation for K-Means cluster defining features

### Model Count Per Bucket

Same structure as K-Means (window-level granularity):

| Bucket | Windows | Window-Level RF Models |
|--------|---------|------------------------|
| 0-3s | 1 (hook only) | 1 |
| 3-9s | 2 (hook, closing) | 2 |
| 9-13s | 3 (hook, middle_aggregate, closing) | 3 |
| 13-18s | 3 (hook, middle_aggregate, closing) | 3 |
| 18-33s | 6 (hook, middle_1-4, closing) | 6 |
| 33-60s | 7 (hook, middle_1-5, closing) | 7 |
| 60-90s | 7 | 7 |
| 90-120s | 7 | 7 |
| **Total** | **41 windows across 8 buckets** | **41 models** |

### Training Input Per Window

**Hook RF Training (Bucket 18-33s):**
```python
# Load window-specific transformed data
X = pd.read_csv('ml_analysis/hook_rf_transformed.csv')  # Shape: (100 videos, 21 features)

# Features: All base temporal features for hooks only
# - scene_count, eye_contact_rate, word_count, speech_coverage
# - energy_level, gesture_count, emotional_valence, emotion_consistency
# - average_face_size, overlay_unique_count, has_captions
# - shortest_scene, longest_scene, scene_duration_variance
# - object_count, person_count, energy_variance, energy_max
# - pitch_scatter_ratio, gaze_variance, dominant_emotion_id

y = X['is_top_performer']  # Binary labels
X = X.drop(['is_top_performer'], axis=1)

# Train window-level RF
rf_hook = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_hook.fit(X, y)

# Save model
joblib.dump(rf_hook, 'models/rf_hook_18-33s.pkl')
```

**Repeat for all window types**: middle_1, middle_2, middle_3, middle_4, closing

### Output: Window-Level RF Analysis JSON

**File**: `ml_analysis/hook_rf_analysis.json`

```json
{
  "model_type": "window_level_rf",
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "input_features": 21,
  "model_performance": {
    "accuracy": 0.82,
    "precision": 0.85,
    "recall": 0.78
  },
  "feature_importance": [
    {
      "feature": "eye_contact_rate",
      "importance": 0.35,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "rank": 1
    },
    {
      "feature": "energy_level",
      "importance": 0.22,
      "top_performer_avg": 0.82,
      "bottom_performer_avg": 0.54,
      "gap": 0.28,
      "rank": 2
    },
    {
      "feature": "word_count",
      "importance": 0.18,
      "top_performer_avg": 52,
      "bottom_performer_avg": 28,
      "gap": 24,
      "rank": 3
    },
    {
      "feature": "emotional_valence",
      "importance": 0.12,
      "top_performer_avg": 0.65,
      "bottom_performer_avg": 0.42,
      "gap": 0.23,
      "rank": 4
    },
    {
      "feature": "gesture_count",
      "importance": 0.08,
      "top_performer_avg": 6.2,
      "bottom_performer_avg": 3.1,
      "gap": 3.1,
      "rank": 5
    }
    // ... top 10-15 features by importance
  ]
}
```

**Multiple files per bucket**: hook_rf_analysis.json, middle_1_rf_analysis.json, middle_2_rf_analysis.json, etc.

---

## Dual RF Summary

### Total Model Count

| RF Type | Models Per Bucket | Total Models (8 buckets) |
|---------|-------------------|--------------------------|
| **Video-Level** | 1 | 8 |
| **Window-Level** | Varies (1-7 depending on bucket) | 41 |
| **Total RF** | — | **49 models** |

### File Outputs Per Bucket (Example: 18-33s)

```
bucket_18-33s/
├── ml_analysis/
│   ├── rf_video_analysis.json            # Video-level RF (cross-window patterns)
│   ├── hook_rf_analysis.json             # Window-level RF (hook patterns)
│   ├── middle_1_rf_analysis.json         # Window-level RF
│   ├── middle_2_rf_analysis.json         # Window-level RF
│   ├── middle_3_rf_analysis.json         # Window-level RF
│   ├── middle_4_rf_analysis.json         # Window-level RF
│   └── closing_rf_analysis.json          # Window-level RF
└── models/
    ├── rf_video_18-33s.pkl               # Video-level RF model
    ├── rf_hook_18-33s.pkl                # Window-level RF model
    ├── rf_middle_1_18-33s.pkl            # Window-level RF model
    ├── rf_middle_2_18-33s.pkl            # Window-level RF model
    ├── rf_middle_3_18-33s.pkl            # Window-level RF model
    ├── rf_middle_4_18-33s.pkl            # Window-level RF model
    └── rf_closing_18-33s.pkl             # Window-level RF model
```

### Why Dual RF?

| Benefit | Video-Level RF | Window-Level RF |
|---------|----------------|-----------------|
| **Cross-window patterns** | ✅ Captures | ❌ Misses |
| **Within-window validation** | ❌ Mixed signal | ✅ Perfect alignment |
| **K-Means validation** | ⚠️ Indirect | ✅ Direct (same granularity) |
| **Temporal progressions** | ✅ Quantified | ❌ Not visible |
| **Feature importance clarity** | ⚠️ Fragmented | ✅ Per-window clarity |

**Conclusion**: Both types are necessary for complete pattern coverage.

---

# Part 2: Window-Level K-Means Architecture

## Overview

K-Means clustering operates at the **window level**, not the video level. Each temporal window type (Hook, Middle_1, Middle_2, etc.) gets its own separate clustering analysis.

**Key Principle**: Compare apples to apples - Hook features against Hook features, Middle_1 against Middle_1, etc.

---

## Rationale: Why Window-Level K-Means?

**1. Clean Feature Comparison**
- Hooks should be compared to other hooks (first 3 seconds vs first 3 seconds)
- Middles should be compared to other middles (sustained content patterns)
- Closings should be compared to other closings (CTA strategies)
- Avoids mixing temporal contexts in the same distance calculation

**2. Smaller Feature Space**
- Window-level: 21 features per sample (base temporal features)
- Video-level alternative: 129-150 features (all windows concatenated)
- Smaller feature space → better clustering quality, less curse of dimensionality

**3. Business Value Alignment**
- Creators need actionable insights per video section:
  - "What makes viral hooks in 18-33s videos?" → Hook clustering results
  - "What middle segment patterns work?" → Middle clustering results
  - "What CTA strategies perform best?" → Closing clustering results
- Window-level insights are directly actionable

**4. Analysis Context**
- All videos in a bucket are **top performers** (viral videos only)
- Goal: Discover **different strategies** that all lead to success
- Window-level clustering reveals: "3 distinct hook strategies that all work"

---

## Architecture

### K-Means Models Per Bucket

For each duration bucket (e.g., 18-33s), train **separate K-Means models per window type**.

**Bucket 18-33s Example (6 windows):**
- `hook_kmeans.pkl` - Clusters 100 hooks (21 features each)
- `middle_1_kmeans.pkl` - Clusters 100 middle_1 segments (21 features each)
- `middle_2_kmeans.pkl` - Clusters 100 middle_2 segments (21 features each)
- `middle_3_kmeans.pkl` - Clusters 100 middle_3 segments (21 features each)
- `middle_4_kmeans.pkl` - Clusters 100 middle_4 segments (21 features each)
- `closing_kmeans.pkl` - Clusters 100 closings (21 features each)

**Total Models Per Bucket**: 6 K-Means models (one per window type)

**Total Models Per Analysis**:
- 8 duration buckets × 6-7 window models per bucket = **~48-56 K-Means models**
- (Bucket 0-9s has 2 windows, bucket 9-18s has 5 windows, bucket 18-33s+ has 6-7 windows)

---

## Training Data Structure

### Input: Aggregated Features (Stage 3 Output)

From `ml_analysis/aggregated_features.csv`:

```csv
video_id,hook_scene_count,hook_eye_contact_rate,hook_word_count,...,middle_1_scene_count,middle_1_eye_contact_rate,...,closing_scene_count,...
video_001,2,0.87,14,...,3,0.45,...,1,...
video_002,4,0.28,48,...,2,0.62,...,3,...
video_003,1,0.65,28,...,4,0.55,...,2,...
...
```

### Transformation (Stage 4): Separate Per Window

**Hook Transformation** → `ml_analysis/hook_km_transformed.csv`:
```csv
video_id,scene_count,eye_contact_rate,word_count,energy_level,...  (21 features)
video_001,2,0.87,14,0.45,...
video_002,4,0.28,48,0.62,...
video_003,1,0.65,28,0.51,...
```

**Middle_1 Transformation** → `ml_analysis/middle_1_km_transformed.csv`:
```csv
video_id,scene_count,eye_contact_rate,word_count,energy_level,...  (21 features)
video_001,3,0.45,15,0.55,...
video_002,2,0.62,50,0.48,...
video_003,4,0.55,30,0.60,...
```

**Closing Transformation** → `ml_analysis/closing_km_transformed.csv`:
```csv
video_id,scene_count,eye_contact_rate,word_count,energy_level,...  (21 features)
video_001,1,0.82,8,0.85,...
video_002,3,0.75,12,0.90,...
video_003,2,0.55,10,0.80,...
```

### Training Input Per Window

**Hook K-Means Training:**
- **Samples**: 100 rows (one per video's hook)
- **Features**: 21 (base temporal features after K-Means transformation)
- **No target variable** (unsupervised clustering)

**Result**: 3 hook clusters
- Cluster 0: 35 videos (e.g., "High eye contact, low word count hooks")
- Cluster 1: 42 videos (e.g., "Text overlay heavy hooks")
- Cluster 2: 23 videos (e.g., "Action-driven hooks")

---

## Centroid Structure

### Centroid Dimensions: 21 Features (Not 150!)

Each centroid represents the **average window** for that cluster:

```python
# After training hook K-Means
hook_kmeans = KMeans(n_clusters=3)
hook_kmeans.fit(hook_transformed_data)  # Shape: (100 videos, 21 features)

centroids = hook_kmeans.cluster_centers_  # Shape: (3 clusters, 21 features)

# Cluster 0 centroid (21 numbers):
centroids[0] = [
    2.5,   # scene_count average
    0.87,  # eye_contact_rate average
    14.2,  # word_count average
    0.45,  # speech_coverage average
    0.55,  # energy_level average
    ...    # 16 more feature averages (21 total)
]
```

### Defining Features (Simplified)

With only 21 features per centroid, **all features can be sent to the LLM** without overwhelming context.

**No need for complex pre-processing** - the LLM receives:
- 3 centroids × 21 features = 63 numbers per window type
- LLM can easily identify which features define each cluster

---

## Stage 6 Output: K-Means Analysis JSON

### File Structure

**Per Window Type**: `ml_analysis/{window}_kmeans_analysis.json`

Example: `ml_analysis/hook_kmeans_analysis.json`

```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "centroid": {
        "scene_count": 2.5,
        "eye_contact_rate": 0.87,
        "word_count": 14.2,
        "speech_coverage": 0.45,
        "energy_level": 0.55,
        "gesture_count": 3.2,
        "emotional_valence": 0.6,
        "emotion_consistency": 0.75,
        "average_face_size": 0.42,
        "overlay_unique_count": 1.8,
        "has_captions": 0.8,
        "shortest_scene": 0.8,
        "longest_scene": 2.1,
        "scene_duration_variance": 0.3,
        "object_count": 2.1,
        "person_count": 1.0,
        "energy_variance": 0.15,
        "energy_max": 0.65,
        "pitch_scatter_ratio": 0.35,
        "gaze_variance": 0.12,
        "dominant_emotion_id": 1
      },
      "videos": [
        {
          "video_id": "video_001",
          "distance_to_centroid": 0.12
        },
        {
          "video_id": "video_015",
          "distance_to_centroid": 0.18
        }
        // ... 33 more videos
      ]
    },
    {
      "cluster_id": 1,
      "size": 42,
      "centroid": {
        "scene_count": 4.1,
        "eye_contact_rate": 0.28,
        "word_count": 48.5,
        // ... 18 more features
      },
      "videos": [...]
    },
    {
      "cluster_id": 2,
      "size": 23,
      "centroid": {
        "scene_count": 1.9,
        "eye_contact_rate": 0.65,
        "word_count": 28.3,
        // ... 18 more features
      },
      "videos": [...]
    }
  ]
}
```

### Multiple Window Files Per Bucket

For bucket 18-33s, Stage 6 generates **6 K-Means analysis files**:
- `hook_kmeans_analysis.json`
- `middle_1_kmeans_analysis.json`
- `middle_2_kmeans_analysis.json`
- `middle_3_kmeans_analysis.json`
- `middle_4_kmeans_analysis.json`
- `closing_kmeans_analysis.json`

---

## Stage 7 LLM Integration

### LLM Receives Per-Window Insights

**Input to LLM** (per window type):
- 3 clusters with 21-dimensional centroids (63 total numbers)
- Video assignments per cluster
- Metadata: window type, bucket, total videos

**LLM Prompt Structure**:
```
You are analyzing HOOK segments from viral 18-33s videos in the #nutrition hashtag.

You have 3 clusters of hooks from 100 viral videos:

Cluster 0 (35 videos):
- scene_count: 2.5
- eye_contact_rate: 0.87
- word_count: 14.2
- energy_level: 0.55
- ... (all 21 features)

Cluster 1 (42 videos):
- scene_count: 4.1
- eye_contact_rate: 0.28
- word_count: 48.5
- ... (all 21 features)

Cluster 2 (23 videos):
- scene_count: 1.9
- eye_contact_rate: 0.65
- word_count: 28.3
- ... (all 21 features)

Task: Identify what makes each cluster distinct and generate a creative strategy name for each hook pattern.
```

**LLM Output** (per window):
- Cluster interpretations (e.g., "Cluster 0: The Direct Eye Contact Hook")
- Defining features per cluster (e.g., "High eye_contact_rate, low word_count")
- Actionable recommendations (e.g., "Start with direct-to-camera address in first 2 seconds")

### Combining Window Insights into Full Report

**Stage 7 generates creative reports by combining window-level insights**:

Example Creative Strategy: "The Educator's Formula"
- **Hook Strategy**: Cluster 0 pattern (high eye contact, low word count)
- **Middle Strategy**: Cluster 1 pattern (high word count, consistent emotion)
- **Closing Strategy**: Cluster 2 pattern (high energy, speech CTA)

The LLM can mix-and-match window strategies to create **hybrid formulas** based on successful video combinations.

---

## Advantages Over Video-Level Clustering

| Aspect | Window-Level | Video-Level (Alternative) |
|--------|--------------|---------------------------|
| **Feature Dimensions** | 21 per window | 129-150 (all windows concatenated) |
| **Centroid Interpretability** | Simple (21 features) | Complex (150 features) |
| **LLM Context Size** | 63 numbers (3 clusters × 21) | 450 numbers (3 clusters × 150) |
| **Business Insights** | Per-section strategies | Holistic patterns only |
| **Curse of Dimensionality** | Minimal (21D space) | High risk (150D space) |
| **Actionability** | Direct ("Use this hook type") | Abstract ("Use this overall structure") |
| **Models Per Bucket** | 6-7 (one per window) | 1 (all windows together) |

---

## Implementation Notes

### Stage 4 (Feature Transformation) Changes

**Previous assumption**: One transformation per video (all windows concatenated)

**NEW requirement**: Separate transformations per window type

**Code Structure**:
```python
# For each window type in bucket
for window_type in ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']:
    # Extract window-specific features from aggregated_features.csv
    window_features = extract_window_features(aggregated_df, window_type)

    # Apply K-Means transformations (log+scale, cyclical, etc.)
    window_km_transformed = apply_km_transforms(window_features)

    # Save per-window transformed data
    window_km_transformed.to_csv(f'ml_analysis/{window_type}_km_transformed.csv')
```

### Stage 5 (K-Means Training) Changes

**Previous assumption**: One K-Means model per bucket

**NEW requirement**: 6-7 K-Means models per bucket (one per window)

**Code Structure**:
```python
# For each window type in bucket
for window_type in ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']:
    # Load window-specific transformed data
    X = pd.read_csv(f'ml_analysis/{window_type}_km_transformed.csv')

    # Train K-Means (3 clusters)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Save model
    joblib.dump(kmeans, f'models/{window_type}_kmeans.pkl')
```

### Stage 6 (Analysis Generation) Changes

**Previous assumption**: One kmeans_analysis.json per bucket

**NEW requirement**: 6-7 kmeans_analysis.json files per bucket (one per window)

**Code Structure**:
```python
# For each window type in bucket
for window_type in ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']:
    # Load model and data
    kmeans = joblib.load(f'models/{window_type}_kmeans.pkl')
    X = pd.read_csv(f'ml_analysis/{window_type}_km_transformed.csv')

    # Generate cluster assignments
    labels = kmeans.predict(X)
    centroids = kmeans.cluster_centers_

    # Build analysis JSON
    analysis = {
        'window_type': window_type,
        'bucket': bucket_name,
        'total_videos': len(X),
        'clusters': []
    }

    for cluster_id in range(3):
        cluster_videos = X[labels == cluster_id]
        analysis['clusters'].append({
            'cluster_id': cluster_id,
            'size': len(cluster_videos),
            'centroid': dict(zip(feature_names, centroids[cluster_id])),
            'videos': [{'video_id': vid, 'distance_to_centroid': dist}
                      for vid, dist in cluster_videos.items()]
        })

    # Save per-window analysis
    with open(f'ml_analysis/{window_type}_kmeans_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
```

---

## Validation Questions

### Q1: Number of Clusters Per Window
**Current**: 3 clusters per window type

**Alternative**: Dynamic (2-5 clusters based on elbow method or silhouette score)

**Decision**: Start with fixed 3 clusters for consistency, can optimize later

### Q2: Cross-Window Patterns
**Limitation**: Window-level clustering loses cross-window temporal patterns (e.g., "energy builds from hook to closing")

**Mitigation**: Stage 7 LLM can identify cross-window patterns by analyzing video assignments across windows (e.g., "Videos in Hook Cluster 0 tend to also appear in Closing Cluster 2")

**Future Enhancement**: Could add a separate "temporal arc analysis" that looks at cluster transitions

### Q3: Middle Segment Aggregation
**Current**: Each middle segment gets its own clustering (middle_1, middle_2, middle_3, middle_4)

**Alternative**: Aggregate all middle segments into one "middle" clustering

**Decision**: Keep separate for now - different middle segments may have different roles (e.g., middle_1 often continues the hook, middle_4 transitions to closing)

---

## Complete File Architecture (Dual RF + K-Means)

```
bucket_18-33s/
├── ml_analysis/
│   ├── aggregated_features.csv              # Stage 3 output (100 videos, all windows)
│   │
│   ├── rf_transformed.csv                   # Stage 4 output (100 videos, 190 features - video-level)
│   │
│   ├── hook_rf_transformed.csv              # Stage 4 output (100 samples, 21 features - window-level)
│   ├── middle_1_rf_transformed.csv          # Stage 4 output (100 samples, 21 features)
│   ├── middle_2_rf_transformed.csv          # Stage 4 output (100 samples, 21 features)
│   ├── middle_3_rf_transformed.csv          # Stage 4 output (100 samples, 21 features)
│   ├── middle_4_rf_transformed.csv          # Stage 4 output (100 samples, 21 features)
│   ├── closing_rf_transformed.csv           # Stage 4 output (100 samples, 21 features)
│   │
│   ├── hook_km_transformed.csv              # Stage 4 output (100 samples, 21 features - K-Means)
│   ├── middle_1_km_transformed.csv          # Stage 4 output (100 samples, 21 features)
│   ├── middle_2_km_transformed.csv          # Stage 4 output (100 samples, 21 features)
│   ├── middle_3_km_transformed.csv          # Stage 4 output (100 samples, 21 features)
│   ├── middle_4_km_transformed.csv          # Stage 4 output (100 samples, 21 features)
│   ├── closing_km_transformed.csv           # Stage 4 output (100 samples, 21 features)
│   │
│   ├── rf_video_analysis.json               # Stage 6 output (video-level RF, cross-window patterns)
│   │
│   ├── hook_rf_analysis.json                # Stage 6 output (window-level RF, within-window patterns)
│   ├── middle_1_rf_analysis.json            # Stage 6 output
│   ├── middle_2_rf_analysis.json            # Stage 6 output
│   ├── middle_3_rf_analysis.json            # Stage 6 output
│   ├── middle_4_rf_analysis.json            # Stage 6 output
│   ├── closing_rf_analysis.json             # Stage 6 output
│   │
│   ├── hook_kmeans_analysis.json            # Stage 6 output (3 clusters, 21D centroids)
│   ├── middle_1_kmeans_analysis.json        # Stage 6 output
│   ├── middle_2_kmeans_analysis.json        # Stage 6 output
│   ├── middle_3_kmeans_analysis.json        # Stage 6 output
│   ├── middle_4_kmeans_analysis.json        # Stage 6 output
│   └── closing_kmeans_analysis.json         # Stage 6 output
│
└── models/
    ├── rf_video_18-33s.pkl                  # Stage 5 output (video-level RF model)
    │
    ├── rf_hook_18-33s.pkl                   # Stage 5 output (window-level RF models)
    ├── rf_middle_1_18-33s.pkl               # Stage 5 output
    ├── rf_middle_2_18-33s.pkl               # Stage 5 output
    ├── rf_middle_3_18-33s.pkl               # Stage 5 output
    ├── rf_middle_4_18-33s.pkl               # Stage 5 output
    ├── rf_closing_18-33s.pkl                # Stage 5 output
    │
    ├── hook_kmeans.pkl                      # Stage 5 output (K-Means models)
    ├── middle_1_kmeans.pkl                  # Stage 5 output
    ├── middle_2_kmeans.pkl                  # Stage 5 output
    ├── middle_3_kmeans.pkl                  # Stage 5 output
    ├── middle_4_kmeans.pkl                  # Stage 5 output
    ├── closing_kmeans.pkl                   # Stage 5 output
    │
    ├── hook_scalers.pkl                     # Stage 5 output (K-Means MinMaxScalers)
    ├── middle_1_scalers.pkl                 # Stage 5 output
    ├── middle_2_scalers.pkl                 # Stage 5 output
    ├── middle_3_scalers.pkl                 # Stage 5 output
    ├── middle_4_scalers.pkl                 # Stage 5 output
    └── closing_scalers.pkl                  # Stage 5 output
```

---

---

# Part 3: Complete Model Summary

## Total Model Count Across All Buckets

| Model Type | Architecture | Total Models |
|------------|--------------|--------------|
| **Random Forest (Video-Level)** | 1 per bucket | 8 |
| **Random Forest (Window-Level)** | 1 per window per bucket | 41 |
| **K-Means (Window-Level)** | 1 per window per bucket | 41 |
| **TOTAL** | — | **90 models** |

## Why 90 Models?

**The Trade-off**:
- 90 models is 16% more than K-Means alone (41 models)
- But provides **complete pattern coverage** with no blind spots:
  - Cross-window patterns (video-level RF)
  - Within-window patterns (window-level RF)
  - Creative strategies (window-level K-Means)

**The Value**:
- Phase 1 LLM recommendations are RF-validated (high-importance features prioritized)
- Phase 2 LLM formulas are RF-validated (cross-window patterns confirmed)
- Zero blind spots in viral video analysis

---

## Stage 6 JSON Output Summary

For each bucket (e.g., 18-33s), Stage 6 generates:

| JSON File | Content | Size | LLM Consumer |
|-----------|---------|------|--------------|
| `rf_video_analysis.json` | Video-level RF cross-window patterns | ~30KB | Stage 7 Phase 2 |
| `hook_rf_analysis.json` | Window-level RF feature importance | ~5KB | Stage 7 Phase 1 |
| `middle_1_rf_analysis.json` | Window-level RF feature importance | ~5KB | Stage 7 Phase 1 |
| `middle_2_rf_analysis.json` | Window-level RF feature importance | ~5KB | Stage 7 Phase 1 |
| `middle_3_rf_analysis.json` | Window-level RF feature importance | ~5KB | Stage 7 Phase 1 |
| `middle_4_rf_analysis.json` | Window-level RF feature importance | ~5KB | Stage 7 Phase 1 |
| `closing_rf_analysis.json` | Window-level RF feature importance | ~5KB | Stage 7 Phase 1 |
| `hook_kmeans_analysis.json` | 3 clusters, 21D centroids | ~5KB | Stage 7 Phase 1 |
| `middle_1_kmeans_analysis.json` | 3 clusters, 21D centroids | ~5KB | Stage 7 Phase 1 |
| `middle_2_kmeans_analysis.json` | 3 clusters, 21D centroids | ~5KB | Stage 7 Phase 1 |
| `middle_3_kmeans_analysis.json` | 3 clusters, 21D centroids | ~5KB | Stage 7 Phase 1 |
| `middle_4_kmeans_analysis.json` | 3 clusters, 21D centroids | ~5KB | Stage 7 Phase 1 |
| `closing_kmeans_analysis.json` | 3 clusters, 21D centroids | ~5KB | Stage 7 Phase 1 |

**Total per bucket**: 13 JSON files, ~95KB total

---

## Implementation Impacts

### Stage 4 (Feature Transformation) Changes

**NEW requirement**: Three transformation pipelines

1. **Video-level RF transformation** → `rf_transformed.csv` (190 features)
2. **Window-level RF transformation** → `{window}_rf_transformed.csv` (21 features × 6 windows)
3. **Window-level K-Means transformation** → `{window}_km_transformed.csv` (21 features × 6 windows)

### Stage 5 (Model Training) Changes

**NEW requirement**: Train 90 models per complete analysis (8 buckets)

1. **8 video-level RF models** (1 per bucket)
2. **41 window-level RF models** (varies by bucket)
3. **41 K-Means models** (varies by bucket)

### Stage 6 (Analysis Generation) Changes

**NEW requirement**: Generate 13 JSON files per bucket

1. **1 video-level RF JSON** (cross-window patterns)
2. **6 window-level RF JSONs** (within-window validation)
3. **6 K-Means JSONs** (cluster centroids)

---

## Next Steps

1. ✅ **Dual RF Architecture Decision**: APPROVED
2. ✅ **Window-Level K-Means**: APPROVED
3. **Update**: Revise MLPlanningv2.md Stage 4-6 to reflect dual RF + window-level K-Means architecture
4. **Update**: Revise FeatureTransformation.md to document three transformation pipelines
5. **Update**: Revise Critique_FeatureTransformation.md to mark issues as RESOLVED
6. **Create**: LLMAnalysis7.md now references this document for Stage 6 outputs (no duplication)

---

**Status**: APPROVED - Ready for Stage 4-6 implementation and parent document updates
