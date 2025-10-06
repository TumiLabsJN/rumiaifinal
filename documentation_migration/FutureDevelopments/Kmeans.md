# K-Means Clustering for RumiAI ML Pipeline

**Purpose**: Document K-Means preprocessing, scaler fitting, and clustering strategy for temporal video analysis

**Parent Documents**:
- [FeatureTransformation.md](./FeatureTransformation.md) - Transformation specifications
- [MLPlanningv2.md](./MLPlanningv2.md) - Overall ML pipeline architecture (stage-based)

**Last Updated**: 2025-10-06

---

## Overview

K-Means clustering is used to discover distinct creative patterns within TikTok videos. Unlike Random Forest (which predicts top vs bottom performers), K-Means **groups similar videos together** to identify repeatable strategies.

**Use Cases:**
- **Hashtag Analysis**: Cluster top performers to find 3-5 distinct viral strategies
- **Competitor Analysis**: Discover competitor's dominant content patterns
- **Creator Analysis**: Understand creator's natural style distribution

---

## Why K-Means Requires Scaling

K-Means uses **Euclidean distance** to measure similarity between videos:

```
distance = √[(feature1_A - feature1_B)² + (feature2_A - feature2_B)² + ...]
```

**Problem Without Scaling:**

Video A: `scene_count: 15`, `speech_coverage: 0.8`
Video B: `scene_count: 5`, `speech_coverage: 0.6`

Distance calculation:
```
√[(15-5)² + (0.8-0.6)²] = √[100 + 0.04] = √100.04 ≈ 10
```

**Issue:** `scene_count` difference (100) dominates over `speech_coverage` difference (0.04). The algorithm ignores speech_coverage entirely!

**Solution: Scale all features to [0-1]:**

Video A (scaled): `scene_count: 0.83`, `speech_coverage: 0.8`
Video B (scaled): `scene_count: 0.25`, `speech_coverage: 0.6`

Distance calculation:
```
√[(0.83-0.25)² + (0.8-0.6)²] = √[0.34 + 0.04] = √0.38 ≈ 0.62
```

**Now both features contribute equally to distance calculation!**

---

## Temporal Aggregation Structure

Before scaling, we must convert temporal windows into a flat feature vector.

### Raw Temporal Windows (From RumiAI Output)

One video produces variable-length temporal data:

```json
{
  "duration": 20,
  "temporal_windows": {
    "hook": {
      "scene_count": 3,
      "word_count": 12,
      "gesture_count": 2,
      "energy_level": 0.7
    },
    "middle_segments": [
      {"scene_count": 2, "word_count": 8, "gesture_count": 1, "energy_level": 0.6},
      {"scene_count": 4, "word_count": 15, "gesture_count": 3, "energy_level": 0.8},
      {"scene_count": 3, "word_count": 10, "gesture_count": 2, "energy_level": 0.7},
      {"scene_count": 5, "word_count": 18, "gesture_count": 4, "energy_level": 0.9}
    ],
    "closing": {
      "scene_count": 1,
      "word_count": 5,
      "gesture_count": 0,
      "energy_level": 0.5
    }
  }
}
```

**Problem:** Can't feed arrays to K-Means (needs fixed-length vectors).

### Aggregated Feature Vector (ML Training Input)

**Aggregation Strategy:**
- **Hook features**: Use directly (always 1 hook window)
- **Middle features**: Average across all middle segments (handles variable count)
- **Closing features**: Use directly (always 1 closing window)

**Result (fixed structure):**

```python
{
  # Hook features (3s window)
  "hook_scene_count": 3,
  "hook_word_count": 12,
  "hook_gesture_count": 2,
  "hook_energy_level": 0.7,

  # Middle features (averaged across 4 segments)
  "middle_avg_scene_count": (2+4+3+5)/4 = 3.5,
  "middle_avg_word_count": (8+15+10+18)/4 = 12.75,
  "middle_avg_gesture_count": (1+3+2+4)/4 = 2.5,
  "middle_avg_energy_level": (0.6+0.8+0.7+0.9)/4 = 0.75,

  # Closing features (3s window)
  "closing_scene_count": 1,
  "closing_word_count": 5,
  "closing_gesture_count": 0,
  "closing_energy_level": 0.5,

  # Global features
  "duration": 20
}
```

**This creates 13 separate features from one video's temporal windows.**

---

## Scaler Fitting: Why Each Position Needs Its Own Scaler

### The Core Insight

After aggregation, `scene_count` becomes **3 different features**:
1. `hook_scene_count` (scenes in first 3 seconds)
2. `middle_avg_scene_count` (average scenes per middle segment)
3. `closing_scene_count` (scenes in last 3 seconds)

**These have fundamentally different distributions** → need separate scalers.

---

### Example: 18-33s Bucket Training Data (N Videos from --video-count)

**Training Matrix (simplified to 3 features):**

| Video | hook_scene_count | middle_avg_scene_count | closing_scene_count |
|-------|------------------|------------------------|---------------------|
| 1     | 3                | 3.5                    | 1                   |
| 2     | 2                | 5.0                    | 2                   |
| 3     | 4                | 2.75                   | 1                   |
| 4     | 3                | 6.25                   | 2                   |
| 5     | 5                | 4.0                    | 1                   |
| ...   | ...              | ...                    | ...                 |
| N     | 3                | 3.25                   | 2                   |

**Note:** Each column is a separate feature with different values. N is configurable (default 100 for contrastive strategy).

---

### Scaler Fitting Process

**Step 1: Fit Scaler to Column 1 (hook_scene_count)**

Extract all N values from column 1:
```python
hook_scene_count_data = [3, 2, 4, 3, 5, 2, 3, 4, 3, ..., 3]  # N values

scaler_hook = MinMaxScaler()
scaler_hook.fit(hook_scene_count_data.reshape(-1, 1))

# Scaler learns from THIS column:
# min_value = 2  (lowest hook scene count across N videos)
# max_value = 5  (highest hook scene count across N videos)
```

**Step 2: Fit Scaler to Column 2 (middle_avg_scene_count)**

Extract all N values from column 2:
```python
middle_avg_scene_count_data = [3.5, 5.0, 2.75, 6.25, 4.0, ..., 3.25]  # N values

scaler_middle = MinMaxScaler()
scaler_middle.fit(middle_avg_scene_count_data.reshape(-1, 1))

# Scaler learns from THIS column:
# min_value = 2.5   (lowest middle avg)
# max_value = 6.5   (highest middle avg)
```

**Step 3: Fit Scaler to Column 3 (closing_scene_count)**

Extract all N values from column 3:
```python
closing_scene_count_data = [1, 2, 1, 2, 1, 1, 2, 1, ..., 2]  # N values

scaler_closing = MinMaxScaler()
scaler_closing.fit(closing_scene_count_data.reshape(-1, 1))

# Scaler learns from THIS column:
# min_value = 1  (lowest closing scene count)
# max_value = 2  (highest closing scene count)
```

---

### Why Different Ranges?

**Hook (3s window):**
- Physical constraint: Only 3 seconds to fit scenes
- Typical range: 1-5 scene cuts
- Scaler learns: min=2, max=5

**Middle (averaged across ~12-27s total):**
- More accumulated time → higher counts possible
- Averaging smooths values → decimal results
- Typical range: 2.5-6.5 scene cuts (averaged)
- Scaler learns: min=2.5, max=6.5

**Closing (3s window):**
- Often static shots for CTA (fewer cuts)
- Typical range: 1-2 scene cuts
- Scaler learns: min=1, max=2

**These are different contexts → different distributions → need separate scalers!**

---

### Scaling Transformation

After fitting, transform each column with its own scaler.

**Video 1 (before scaling):**
```python
[hook: 3, middle_avg: 3.5, closing: 1]
```

**Apply scalers (formula: (value - min) / (max - min)):**

```python
# hook_scene_count: 3
scaled_hook = (3 - 2) / (5 - 2) = 1/3 = 0.33

# middle_avg_scene_count: 3.5
scaled_middle = (3.5 - 2.5) / (6.5 - 2.5) = 1.0/4.0 = 0.25

# closing_scene_count: 1
scaled_closing = (1 - 1) / (2 - 1) = 0/1 = 0.0

# Result:
video_1_scaled = [0.33, 0.25, 0.0]
```

**Video 4 (before scaling):**
```python
[hook: 3, middle_avg: 6.25, closing: 2]
```

**Apply scalers:**
```python
scaled_hook = (3 - 2) / (5 - 2) = 0.33      # Same hook value as Video 1
scaled_middle = (6.25 - 2.5) / (6.5 - 2.5) = 0.94  # High middle value
scaled_closing = (2 - 1) / (2 - 1) = 1.0    # Max closing value

# Result:
video_4_scaled = [0.33, 0.94, 1.0]
```

**Key Insight:** Same raw value (3) in different positions gets different scaled values (0.33 in hook, but 3 would be 0.12 in middle_avg) because the scales are different!

---

## Per-Bucket Scaler Architecture

### Why Scalers Are Bucket-Specific

**9-13s bucket training data:**
- Middle segments: 1.0-2.33s each
- `middle_avg_scene_count` range: 1.5-3.0 (short segments = fewer scenes)
- Scaler learns: min=1.5, max=3.0

**18-33s bucket training data:**
- Middle segments: 3.0-6.75s each
- `middle_avg_scene_count` range: 2.5-6.5 (longer segments = more scenes)
- Scaler learns: min=2.5, max=6.5

**Same feature name, different bucket → different scaler!**

If we used one global scaler:
- 9-13s videos would all scale to 0.0-0.3 (low end)
- 18-33s videos would scale to 0.4-1.0 (high end)
- K-Means would cluster by duration, not by pattern (defeats the purpose)

**Solution: Fit scalers per bucket** → normalize relative to what's "high" or "low" within that bucket's context.

---

### Total Scalers Per Bucket

**Example: 18-33s bucket**

Numerical features requiring scaling (after aggregation):

**Hook features (~10 features):**
- hook_scene_count
- hook_word_count
- hook_gesture_count
- hook_energy_level
- hook_energy_variance
- hook_overlay_unique_count
- hook_object_count
- hook_person_count
- hook_gaze_variance
- ... (all count/variance features)

**Middle features (~10 features):**
- middle_avg_scene_count
- middle_avg_word_count
- middle_avg_gesture_count
- middle_avg_energy_level
- middle_avg_energy_variance
- middle_avg_overlay_unique_count
- middle_avg_object_count
- middle_avg_person_count
- middle_avg_gaze_variance
- ...

**Closing features (~10 features):**
- closing_scene_count
- closing_word_count
- closing_gesture_count
- closing_energy_level
- closing_energy_variance
- closing_overlay_unique_count
- closing_object_count
- closing_person_count
- closing_gaze_variance
- ...

**Global features (~2-3 features):**
- duration
- emotional_valence (after shift)
- ...

**Total: ~25-30 scalers per bucket**

---

### Storage Structure

**Per-bucket scaler files:**

```
/data/clients/acme/hashtags/nutrition/bucket_18-33s/models/scalers/
├── hook_scene_count_scaler.pkl
├── hook_word_count_scaler.pkl
├── hook_gesture_count_scaler.pkl
├── middle_avg_scene_count_scaler.pkl
├── middle_avg_word_count_scaler.pkl
├── middle_avg_gesture_count_scaler.pkl
├── closing_scene_count_scaler.pkl
├── closing_word_count_scaler.pkl
├── closing_gesture_count_scaler.pkl
├── duration_scaler.pkl
└── ... (~25-30 .pkl files)
```

**Total system scalers (with adaptive bucket processing):**
- **Theoretical capacity**: 8 buckets × ~25 scalers = ~200 files
- **Actual per analysis**: 3 qualified buckets × ~25 scalers = **~75 scaler files per analysis**
- Only top 3 buckets with highest winner concentration are processed (adaptive processing)

---

## Full K-Means Pipeline

### Step-by-Step Walkthrough

**Input:** N videos in 18-33s bucket (from --video-count parameter, default 100)
- **Contrastive strategy**: 80% top + 20% bottom performers (e.g., 80+20 if N=100)
- **Top strategy**: N top performers only (e.g., 40 if N=40)

---

#### 1. Temporal Aggregation

For each of N videos, convert temporal windows → flat feature vector.

**Video 1:**
```python
temporal_windows → [hook_scene_count: 3, hook_word_count: 12, ...,
                     middle_avg_scene_count: 3.5, middle_avg_word_count: 12.75, ...,
                     closing_scene_count: 1, closing_word_count: 5, ...,
                     duration: 20]
```

**Repeat for all N videos → N × 40 feature matrix (before scaling)**

---

#### 2. Apply Feature Transformations (from FeatureTransformation.md)

**Direct features (already [0-1]):**
- hook_energy_level, middle_avg_energy_level, closing_energy_level → use as-is
- hook_eye_contact_rate, closing_eye_contact_rate → use as-is

**Log + scale features (count/variance features):**
- hook_scene_count → log1p(3) = 1.39
- middle_avg_scene_count → log1p(3.5) = 1.50
- closing_scene_count → log1p(1) = 0.69
- (Continue for all count features)

**One-hot features:**
- dominant_emotion_id: 4 → [0, 0, 0, 1, 0, 0, 0] (7 binary columns)

**Cyclical features:**
- create_time → [hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]

**Result after transformations:** N × 40 matrix (numerical features only)

---

#### 3. Fit Scalers (One Per Column)

**For each of the ~25-30 numerical columns:**

```python
from sklearn.preprocessing import MinMaxScaler

# Column 1: hook_scene_count (after log transform)
hook_scene_count_log = [1.39, 1.10, 1.61, 1.39, 1.79, ...]  # N values
scaler_hook_scene = MinMaxScaler()
scaler_hook_scene.fit(hook_scene_count_log.reshape(-1, 1))
joblib.dump(scaler_hook_scene, 'scalers/hook_scene_count_scaler.pkl')

# Column 2: middle_avg_scene_count (after log transform)
middle_avg_scene_count_log = [1.50, 1.79, 1.32, 1.95, ...]  # N values
scaler_middle_scene = MinMaxScaler()
scaler_middle_scene.fit(middle_avg_scene_count_log.reshape(-1, 1))
joblib.dump(scaler_middle_scene, 'scalers/middle_avg_scene_count_scaler.pkl')

# ... repeat for all 25-30 numerical features
```

---

#### 4. Transform (Scale All Columns)

**Load fitted scalers and transform each column:**

```python
# Transform hook_scene_count column
hook_scene_count_scaled = scaler_hook_scene.transform(hook_scene_count_log)

# Transform middle_avg_scene_count column
middle_avg_scene_count_scaled = scaler_middle_scene.transform(middle_avg_scene_count_log)

# ... repeat for all columns

# Combine back into matrix
X_scaled = np.hstack([
    hook_scene_count_scaled,
    middle_avg_scene_count_scaled,
    # ... all other scaled columns
])

# Result: N × 40 matrix, all values in [0-1] range
```

---

#### 5. Train K-Means

```python
from sklearn.cluster import KMeans

# Typically 3-5 clusters for hashtag analysis
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Save model
joblib.dump(kmeans, 'bucket_18-33s_kmeans.pkl')

# Get cluster assignments
cluster_labels = kmeans.labels_  # [0, 2, 1, 0, 3, 2, ...]
```

---

#### 6. Analyze Clusters

**Cluster assignments:**
- Cluster 0: Videos 1, 4, 8, 15, ... (12 videos)
- Cluster 1: Videos 3, 9, 12, 20, ... (15 videos)
- Cluster 2: Videos 2, 6, 10, 14, ... (18 videos)
- Cluster 3: Videos 5, 7, 11, 19, ... (15 videos)

**Inverse transform to interpret patterns:**

```python
# Get cluster 0 centroid (in scaled space)
cluster_0_centroid_scaled = kmeans.cluster_centers_[0]

# Inverse transform each feature to original scale
hook_scene_centroid = scaler_hook_scene.inverse_transform(
    cluster_0_centroid_scaled[0].reshape(1, -1)
)
# Result: 1.45 (log space) → exp(1.45) - 1 ≈ 3.3 scenes

# Repeat for all features → get interpretable pattern
```

**Cluster 0 pattern interpretation:**
- Hook: 3-4 scene cuts, 10-12 words, high eye contact
- Middle: 5-6 scene cuts avg, 20-25 words avg, product visible
- Closing: 1-2 scene cuts, CTA present, static shot

**Generate report:** "The Question Hook Strategy for 18-33s videos"

---

## Analysis-Type Variations

### Hashtag Analysis (Contrastive Clustering)

**Training data (per qualified bucket):**
- N samples (from --video-count parameter, default 100)
- **Contrastive split**: 80% top + 20% bottom performers (e.g., 80+20 if N=100)
- K-Means clusters ALL N videos together

**Adaptive Bucket Processing:**
- Analyzes top 100 performers to identify winner distribution
- Processes only top 3 buckets with highest winner concentration
- Example: If 18-33s, 33-60s, 13-18s contain 95% of winners → process only these 3
- Skips buckets with few winners (e.g., 9-13s with 5% of winners)

**Post-clustering analysis:**
- Check which clusters contain mostly top performers
- Example: Cluster 0 = 68 top, 4 bottom (94.4% top performers)
- Cluster 0 becomes "Strategy A - High Performer Pattern"

**Report generation:**
- Extract Cluster 0 centroid → interpret features → create actionable report
- "Videos in this cluster average 3-4 hook scenes, 20-25 middle words, ..."

---

### Competitor Analysis (Pattern Discovery)

**Training data:**
- All 150 competitor videos (no top/bottom split)
- No labels needed

**Clustering:**
- K-Means discovers competitor's natural strategies
- Example: 3 clusters found
  - Cluster A: Product demos (60 videos, 40%)
  - Cluster B: UGC testimonials (50 videos, 33%)
  - Cluster C: Educational content (40 videos, 27%)

**Report generation:**
- "Competitor uses 3 distinct content strategies..."
- Breakdown of each strategy's characteristics

---

### Creator Analysis (Style Profiling)

**Different aggregation:**
- Don't cluster individual videos
- Average features per bucket for the creator
- Example: Creator has 5 videos in 18-33s bucket
  - Average all 5 videos' hook_scene_count → 3.2
  - Average all 5 videos' middle_avg_word_count → 22.5
  - Result: One vector representing "creator's typical 18-33s style"

**Compatibility scoring:**
- Load client's top-performing cluster centroids (from hashtag analysis)
- Calculate cosine similarity between creator's avg vector and cluster centroids
- Score: 0.85 → "Creator's style matches Cluster 2 pattern (Question Hook Strategy)"

---

## Common Pitfalls

### Pitfall 1: Forgetting to Scale

```python
# ❌ BAD: Train K-Means on unscaled data
kmeans.fit(X_unscaled)
# Result: scene_count dominates, other features ignored
```

```python
# ✅ GOOD: Scale first, then train
X_scaled = scale_all_features(X_unscaled, scalers)
kmeans.fit(X_scaled)
```

---

### Pitfall 2: Using Wrong Bucket's Scaler

```python
# ❌ BAD: Load 9-13s scaler for 18-33s video
scaler = load('bucket_9-13s/scalers/hook_scene_count_scaler.pkl')
scaled = scaler.transform(video_18s_data)
# Result: Wrong normalization, nonsensical clusters
```

```python
# ✅ GOOD: Use matching bucket scaler
scaler = load('bucket_18-33s/scalers/hook_scene_count_scaler.pkl')
scaled = scaler.transform(video_18s_data)
```

---

### Pitfall 3: Mixing Temporal Windows Directly

```python
# ❌ BAD: Feed temporal windows array to K-Means
middle_segments = [
    {"scene_count": 2},
    {"scene_count": 4},
    {"scene_count": 3},
    {"scene_count": 5}
]
kmeans.fit(middle_segments)  # Error! Variable-length arrays
```

```python
# ✅ GOOD: Aggregate first
middle_avg_scene_count = (2 + 4 + 3 + 5) / 4  # 3.5
feature_vector = [hook_scene_count, middle_avg_scene_count, closing_scene_count]
kmeans.fit([feature_vector])
```

---

### Pitfall 4: Not Saving Scalers

```python
# ❌ BAD: Fit scalers but don't save
scaler.fit(training_data)
# Train model, but forget to save scalers
# Later: Can't scale new videos for prediction!
```

```python
# ✅ GOOD: Save all scalers
for feature_name, scaler in scalers.items():
    joblib.dump(scaler, f'scalers/{feature_name}_scaler.pkl')
# Later: Load scalers to scale new videos consistently
```

---

## Implementation Checklist

**Before Training:**
- [ ] Temporal windows aggregated (hook + avg_middle + closing)
- [ ] Feature transformations applied (log, one-hot, cyclical)
- [ ] All features are numerical (no strings)
- [ ] Feature matrix is 2D (samples × features)

**During Training:**
- [ ] Fit one scaler per numerical feature column
- [ ] Save all fitted scalers to disk
- [ ] Transform data using fitted scalers
- [ ] All scaled values in [0-1] or similar range
- [ ] Train K-Means on scaled data

**After Training:**
- [ ] Save K-Means model to disk
- [ ] Save cluster centroids (scaled and unscaled)
- [ ] Document cluster interpretations
- [ ] Generate creative reports per cluster

**For Inference (new videos):**
- [ ] Load correct bucket's scalers
- [ ] Apply same transformations as training
- [ ] Scale using loaded scalers (don't refit!)
- [ ] Predict cluster assignment
- [ ] Inverse transform centroid for interpretation

---

## Implementation Reference

**Related Documents:**
- [FeatureTransformation.md](./FeatureTransformation.md) - Transform specifications
- [MLPlanningv2.md](./MLPlanningv2.md) - Overall pipeline (stage-based architecture)
- [MLPlanning.md](./MLPlanning.md) - Legacy planning document (comprehensive details)
- [feature_transforms.json](./feature_transforms.json) - Machine-readable config
- [QUICK_REFERENCE.md](../../QUICK_REFERENCE.md) - Temporal window structure

**Python Libraries:**
- `sklearn.preprocessing.MinMaxScaler` - Scaler fitting/transformation
- `sklearn.cluster.KMeans` - Clustering algorithm
- `joblib` - Model/scaler serialization
- `numpy` - Matrix operations
