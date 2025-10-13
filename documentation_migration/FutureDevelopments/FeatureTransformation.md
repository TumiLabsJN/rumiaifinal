# Feature Transformation Specification

**Purpose**: Define how each feature is transformed for Random Forest and K-Means ML models

**Parent Document**: See [TotalFeatures.md](./TotalFeatures.md) for complete feature definitions, sources, and dependencies

**Last Updated**: 2025-10-04

---

## Model Training Architecture

**Per-Bucket Training Strategy:**
- **8 duration buckets** (0-3s, 3-9s, 9-13s, 13-18s, 18-33s, 33-60s, 60-90s, 90-120s)
- **Separate models per bucket**: 8 RF models + 8 KMeans models = **16 models total per analysis**
- **Same transformation logic across all buckets** (this document applies to all)
- **Bucket-specific fitted scalers** (each bucket learns its own min/max ranges for K-Means)

**Why Per-Bucket Models?**

**Variance Reduction:** Segment durations must be comparable for meaningful patterns
- 9-13s bucket: Middle segments are 1.0-2.33s each (2.33x variance)
- 13-18s bucket: Middle segments are 2.33-4.0s each (1.72x variance)
- If combined: 1.0-4.0s segments (4x variance) → incomparable contexts

**Example of the problem:**
- 9s video: 1s segment with `scene_count: 1` (constrained by time)
- 18s video: 4s segment with `scene_count: 4` (4x more time)
- Can't compare these in one model (different physical constraints)

**Per-bucket models ensure:**
- ✅ Comparable temporal contexts (similar segment durations)
- ✅ Reliable pattern discovery (apples-to-apples comparisons)
- ✅ Accurate scalers (fitted to bucket-specific data ranges)

---

## Transform Strategy Overview

**Random Forest (RF):**
- Tree-based algorithm, scale-invariant
- Handles mixed data types naturally
- Minimal preprocessing needed
- Output: ~39 features per sample

**K-Means (KM):**
- Distance-based algorithm, scale-sensitive
- Requires all numerical features with similar scales
- Heavy preprocessing required
- Output: ~40 features per sample

**Scaler Fitting (K-Means only):**
- Each bucket fits its own MinMaxScaler to training data
- Example: `scene_count` in 9-13s bucket learns min=1, max=5
- Example: `scene_count` in 18-33s bucket learns min=3, max=15
- Same feature, different ranges → bucket-specific scaling

---

## Feature Transformation Table

| Feature Name | Description | Data Type | Data Range | RF Transform | RF Output | KM Transform | KM Output | Complexity | Notes |
|--------------|-------------|-----------|------------|--------------|-----------|--------------|-----------|------------|-------|
| average_face_size | Mean face prominence in frame | Float | [0-1] | Direct | 1 feature | Scale [0-1] | 1 feature | Low | Already normalized |
| overlay_unique_count | Count of unique text overlays | Integer | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Skewed distribution |
| has_captions | Speech-synchronized captions present | Boolean | True/False | One-hot | 2 features | Label encode | 1 feature | Low | Binary categorical |
| scene_count | Number of scene changes | Integer | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Count feature |
| shortest_scene | Duration of shortest scene | Float | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Can have extreme outliers |
| longest_scene | Duration of longest scene | Float | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Can have extreme outliers |
| scene_duration_variance | Variance in scene durations | Float | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Right-skewed variance |
| object_count | Non-person objects detected | Integer | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Count feature |
| person_count | Maximum persons visible simultaneously | Integer | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Count feature |
| speech_coverage | Speech density (% of video with speech) | Float | [0-1] | Direct | 1 feature | Scale [0-1] | 1 feature | Low | Already normalized |
| word_count | Total words spoken | Integer | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Count feature |
| energy_level | Mean audio intensity | Float | [0-1] | Direct | 1 feature | Scale [0-1] | 1 feature | Low | Already normalized |
| energy_variance | Audio intensity variance | Float | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Right-skewed variance |
| energy_max | Peak audio intensity | Float | [0-1] | Direct | 1 feature | Scale [0-1] | 1 feature | Low | Already normalized |
| pitch_scatter_ratio | Pitch instability measure | Float | [0-1] | Direct | 1 feature | Scale [0-1] | 1 feature | Low | Already normalized |
| gesture_count | Hand movement count | Integer | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Count feature |
| gaze_variance | Gaze stability variance | Float | [0-∞] | Direct | 1 feature | Log + scale | 1 feature | Low | Right-skewed variance |
| eye_contact_rate | Eye contact percentage | Float | [0-1] | Direct | 1 feature | Scale [0-1] | 1 feature | Low | Already normalized |
| create_time | Video publish timestamp | String | ISO 8601 | Extract-date | 5 features | Cyclical | 6 features | Medium | Extract: hour, day, month, weekend, business_hours (RF) / sin/cos pairs (KM) |
| gender_detection | Detected gender classification | Object | Nested | Extract + one-hot | 2-3 features | Extract + label encode | 1 feature | Low | Extract gender_label only |
| dominant_emotion_id | Most frequent emotion | Categorical | 1-7 | One-hot | 7 features | One-hot | 7 features | Low | 7 categories: joy, sadness, anger, fear, disgust, surprise, neutral |
| emotional_valence | Positive vs negative tone | Float | [-1, 1] | Direct | 1 feature | Shift + scale | 1 feature | Low | Shift [-1,1] → [0,1] for KM |
| emotion_consistency | Emotional focus consistency | Float | [0, 1] | Direct | 1 feature | Scale [0-1] | 1 feature | Low | Already normalized |

---

## Temporal Features to ML Training Input

**Key Concept:** Features are calculated **per temporal window**, then aggregated to video level for ML training.

### Window Structure by Duration:

Videos are processed through temporal windows (Hook + Middle Segments + Closing):

| Duration | Window Structure | Total Windows |
|----------|------------------|---------------|
| 0-9s | Hook + Closing (no middle) | 2 windows |
| 9-18s | Hook + 3 Middle + Closing | 5 windows |
| 18-33s | Hook + 4 Middle + Closing | 6 windows |
| 33-75s+ | Hook + 5 Middle + Closing | 7 windows |

### From Temporal Windows to Training Samples:

**Step 1: Temporal Compute** (upstream)
- Each window gets its own feature values
- Example 18s video (6 windows):
  - Hook: `scene_count: 2`, `word_count: 10`
  - Middle_1: `scene_count: 3`, `word_count: 15`
  - Middle_2: `scene_count: 2`, `word_count: 12`
  - Middle_3: `scene_count: 4`, `word_count: 18`
  - Middle_4: `scene_count: 3`, `word_count: 14`
  - Closing: `scene_count: 1`, `word_count: 8`

**Step 2: Full Temporal Granularity (Stage 3: Feature Aggregation)**
- **Hook features**: Use hook window values directly
  - `hook_scene_count = 2`, `hook_word_count = 10`
- **Middle features**: Preserve ALL middle segments separately (NO AVERAGING)
  - `middle_1_scene_count = 3`, `middle_1_word_count = 15`
  - `middle_2_scene_count = 2`, `middle_2_word_count = 12`
  - `middle_3_scene_count = 4`, `middle_3_word_count = 18`
  - `middle_4_scene_count = 3`, `middle_4_word_count = 14`
- **Closing features**: Use closing window values directly
  - `closing_scene_count = 1`, `closing_word_count = 8`
- **Metadata features**: Video-level (not per-window)
  - `duration`, `create_time`, `gender_detection`

**Step 3: Bucket-Specific Feature Vector**
- One training sample = one video's full temporal features
- **Feature count varies by bucket** (bucket-specific models handle this):
  - Bucket 0-9s: 21 features × 2 windows + 3 metadata = **45 features**
  - Bucket 9-18s: 21 features × 5 windows + 3 metadata = **108 features**
  - Bucket 18-33s: 21 features × 6 windows + 3 metadata = **129 features**
  - Bucket 33-75s+: 21 features × 7 windows + 3 metadata = **150 features**
- Feature structure: `[hook_*, middle_1_*, middle_2_*, ..., middle_N_*, closing_*, metadata]`
- Example (18-33s bucket): `[hook_scene_count, middle_1_scene_count, middle_2_scene_count, middle_3_scene_count, middle_4_scene_count, closing_scene_count, duration, ...]`

**Why Full Granularity (No Averaging):**
- **Preserves temporal evolution**: Emotional arcs (neutral → happy → sad), pacing changes, narrative structure
- **Bucket-specific models**: Each bucket trains separate RF and KMeans models, so different feature counts are acceptable
- **No information loss**: Middle segments capture sustained patterns WITHOUT averaging
- **Example**: A video that builds suspense (low energy → high energy) would lose this pattern if middle segments were averaged

**Architecture Note:**
- Each bucket has IDENTICAL window structures (all 18-33s videos have 6 windows)
- This eliminates ragged arrays WITHIN a bucket
- Different feature counts ACROSS buckets are handled by per-bucket models

---

## Analysis-Specific Workflows

Different analysis types use the same feature transformations but different training strategies.

### 1. Hashtag Analysis (Contrastive Learning)

**Goal:** Discover what makes videos go viral for a hashtag

**Data Collection:**
- Scrape 800 videos for #nutrition (engagement sorted)
- Client-side bucket by duration → identify top 3 active buckets
- Per qualified bucket: select top 80% and bottom 20% (N from --video-count, default 100)
- Each qualified bucket gets N videos (e.g., N=100 → 80 top + 20 bottom)

**Target Variable:** `is_top_performer` (boolean)
- 1 = Top 80% performers (high engagement, e.g., top 80 if N=100)
- 0 = Bottom 20% performers (lower engagement within 800-video sample, e.g., bottom 20 if N=100)

**Training:**
1. Apply feature transformations (this document)
2. Add `is_top_performer` label
3. Train RF (binary classification): predicts if video will be top performer
4. Train KMeans (clustering): discovers pattern clusters in top performers

**Output:**
- RF feature importance → "Which features predict viral success?"
- KMeans clusters → "What distinct strategies work?"
- Creative reports per bucket (e.g., "The Question Hook Formula for 18-33s videos")

---

### 2. Competitor Analysis (Pattern Discovery)

**Goal:** Understand what a competitor does successfully

**Data Collection:**
- Scrape 150 recent videos from @rival_brand
- No top/bottom split (analyze ALL videos)

**Target Variable:** None (no classification)

**Training:**
1. Apply feature transformations (this document)
2. No labels added
3. RF not used (or used for feature importance only)
4. Train KMeans (clustering): discovers competitor's dominant patterns

**Output:**
- KMeans clusters → "Competitor uses 3 distinct content strategies"
- Cluster analysis → "Strategy A: Product demos (40% of videos), Strategy B: UGC testimonials (35%), Strategy C: Educational content (25%)"
- Competitor intelligence report

---

### 3. Creator Analysis (Compatibility Scoring)

**Goal:** Assess if a creator's natural style matches client's viral patterns

**Data Collection:**
- Scrape 40 recent videos from @potential_creator
- Analyze creator's natural content distribution

**Feature Aggregation (Different from Hashtag/Competitor):**
1. Apply feature transformations per bucket
2. **Average features per bucket** (not per video)
   - If creator has 5 videos in 18-33s bucket → average their features
   - Result: One feature vector representing "creator's typical 18-33s style"
3. Compare against client's hashtag patterns per bucket

**Compatibility Scoring:**
- Load client's top-performing patterns (from hashtag analysis)
- Calculate cosine similarity between creator's avg features and client's viral features
- Score per bucket: 0.0 (no match) to 1.0 (perfect match)

**Output:**
- Compatibility scores per bucket
- Hiring recommendation (Tier 1-4)
- Style profile report

---

## Target Variable (Hashtag Analysis Only)

**For Hashtag Analysis:**

Random Forest is trained as a **binary classifier** to predict video performance.

**Target Variable:** `is_top_performer` (boolean)
- **Value 1**: Video is in top 40 performers (by engagement)
- **Value 0**: Video is in bottom 20 performers (by engagement)

**How It's Determined:**
1. Scrape 300+ videos for hashtag
2. Sort by engagement rate (likes + shares + comments) / views
3. Select top 40 → label = 1
4. Select bottom 20 → label = 0
5. Middle performers excluded (clear contrast needed)

**Training Matrix Structure:**
```
Per qualified bucket (e.g., 18-33s bucket):
- N samples (80% top + 20% bottom, N from --video-count, default 100)
- ~39 features (after transformation)
- 1 target column (is_top_performer)

Example row:
[hook_scene_count: 3, middle_avg_word_count: 25, ..., is_top_performer: 1]
```

**Model Output:**
- RF predicts: "Given these features, is this video likely to be a top performer?" (probability 0-1)
- Feature importance: "scene_count in hook is 15% important, word_count in middle is 22% important, ..."

**For Competitor/Creator Analysis:**
- No target variable needed
- RF either not used, or used for unsupervised feature importance only
- KMeans used for pattern discovery

---

## Transform Definitions

### Random Forest Transforms

**Direct**
- Use raw values without transformation
- RF is scale-invariant (doesn't care about feature ranges)
- Simplest preprocessing

**One-hot**
- Convert categorical to binary columns
- Example: has_captions → [no_captions, has_captions] (2 columns)
- Example: dominant_emotion_id → 7 binary columns

**Extract-date**
- Parse timestamp into temporal features
- Extracted features: hour_of_day, day_of_week, month, is_weekend, is_business_hours
- Creates 5 new features from 1 timestamp

**Extract + one-hot**
- Extract field from nested object, then one-hot encode
- Example: gender_detection → extract gender_label → one-hot (2-3 categories)

**ContentAnalysis**
- Feature analyzed in separate ContentAnalysis system
- Not fed into RF model
- Preserves semantic richness

**Exclude**
- Feature not used in RF model
- Reason: system metadata or non-predictive identifiers

---

### K-Means Transforms

**Scale [0-1]**
- Min-max normalization: (x - min) / (max - min)
- For features already in [0-1] range (identity function)

**Log + scale**
- Apply log1p transform: log(1 + x) (handles zeros)
- Then min-max scale to [0-1]
- Compresses skewed distributions (viral outliers)

**Cyclical**
- Sin/cos encoding for temporal features
- Preserves circular nature (hour 23 close to hour 0)
- Formula: sin(2π × value / period), cos(2π × value / period)
- Example: create_time → 6 features (hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos)

**One-hot**
- Convert categorical to binary columns (same as RF)

**Label encode**
- Convert categorical to integers (0, 1, 2, ...)
- Then optionally scale to [0-1]

**Shift + scale**
- Shift negative ranges to positive
- Example: emotional_valence [-1,1] → (x + 1) / 2 → [0,1]

**Extract + label encode**
- Extract field from nested object, then label encode
- Example: gender_detection → extract gender_label → label encode → scale

**ContentAnalysis**
- Feature analyzed in separate ContentAnalysis system (same as RF)

**Exclude**
- Feature not used in KM model (same as RF)

---

## Feature Count Summary

**Total Raw Features**: 37 (from temporal_windows JSON output)

**After Transformation (per video sample):**

**Random Forest:**
- Direct: 26 features
- One-hot: 9 features (2 from has_captions + 7 from dominant_emotion_id)
- Extract-date: 5 features (from create_time)
- Extract + one-hot: 2-3 features (from gender_detection)
- ContentAnalysis: 2 features (description, hashtag_analysis - analyzed separately)
- Exclude: 4 features (video_id, processing_timestamp, version, author)
- **Total per sample: ~39 features**

**K-Means:**
- Scale [0-1]: 8 features (already normalized)
- Log + scale: 16 features (counts, variances)
- One-hot: 7 features (dominant_emotion_id)
- Cyclical: 6 features (from create_time)
- Shift + scale: 1 feature (emotional_valence)
- Label encode: 1 feature (has_captions)
- Extract + label encode: 1 feature (from gender_detection)
- ContentAnalysis: 2 features (description, hashtag_analysis - analyzed separately)
- Exclude: 4 features (video_id, processing_timestamp, version, author)
- **Total per sample: ~40 features**

---

## Per-Bucket Model Architecture

**Total Models per Analysis:**
- 8 buckets × 2 model types = **16 models per client/hashtag/competitor**

**Training Data per Bucket (Hashtag Analysis):**
- N samples (top 80% + bottom 20%, N from --video-count, default 100)
- ~39 features (RF) or ~40 features (KM)
- 1 target variable (`is_top_performer` for RF only)

**Example Training Matrix (18-33s bucket, Hashtag Analysis):**
```
Shape: (N samples, 39 features + 1 target)
Example with N=100:

Samples:
- 80 top performers (label=1)
- 20 bottom performers (label=0)

Features (example subset):
- hook_scene_count: [2, 3, 1, 4, ...]
- middle_avg_word_count: [25, 18, 30, 22, ...]
- closing_eye_contact_rate: [0.8, 0.6, 0.9, 0.7, ...]
- duration: [20, 28, 22, 31, ...]
- ...
- is_top_performer: [1, 1, 1, 1, ..., 0, 0, 0, ...]
```

**Model Outputs per Bucket:**
- `bucket_18-33s_random_forest.pkl` (trained on 60 samples, 39 features)
- `bucket_18-33s_kmeans.pkl` (trained on 60 samples, 40 features)
- `bucket_18-33s_scene_count_scaler.pkl` (fitted MinMaxScaler for scene_count)
- `bucket_18-33s_word_count_scaler.pkl` (fitted MinMaxScaler for word_count)
- ... (one scaler per numerical feature for KM)

**Why 16 Models?**
- Each bucket has unique segment duration ranges
- Scalers must be fitted to bucket-specific data ranges
- Patterns differ by duration (9s viral ≠ 60s viral)
- Ensures comparable temporal contexts within each model

---

## Complexity Breakdown

**Simple (30 features / 81%)**:
- Already normalized [0-1]: 7 features
- Simple counts/durations: 18 features
- Simple categorical: 2 features (has_captions, dominant_emotion_id)
- Excluded: 4 features

**Medium Complexity (1 feature / 3%)**:
- create_time (temporal feature extraction)

**ContentAnalysis (2 features / 5%)**:
- description, hashtag_analysis

**Total complex transformation: Only 1 feature (create_time)**

**Additional Complexity (Per-Bucket):**
- Scaler fitting (K-Means): Each bucket fits ~20 scalers (one per numerical feature)
- Model training: 16 models total per analysis (8 buckets × 2 model types)
- Feature aggregation: Temporal windows → video-level features (hook + avg_middle + closing)

---

## Implementation Reference

### Pipeline Integration

**Output Locations** (see [MLPlanning.md](./MLPlanning.md) for complete file architecture):

```
bucket_18-33s/
├── analysis/insights/
│   └── *_temporal_windows_updated.json       # Input: N raw temporal window files (N from --video-count)
│
└── ml_analysis/
    ├── aggregated_features.csv               # Aggregated temporal windows (N videos)
    ├── rf_transformed.csv                    # RF-ready features (~39 features)
    ├── km_transformed.csv                    # KMeans-ready features (~40 features)
    ├── random_forest_analysis.json           # Input to LLM (~30KB)
    └── kmeans_analysis.json                  # Input to LLM (~30KB)
```

**Pipeline Stages**:

1. **Input**: `analysis/insights/*_temporal_windows_updated.json`
   - 60 JSON files per bucket
   - Features per temporal window (hook, middle segments, closing)

2. **Aggregation**: → `ml_analysis/aggregated_features.csv`
   - Hook features: Use directly
   - Middle features: Average across segments
   - Closing features: Use directly
   - See "Temporal Features to ML Training Input" section above

3. **RF Transformation**: → `ml_analysis/rf_transformed.csv`
   - Direct use of numerical features
   - One-hot encoding for categoricals
   - Extract temporal features from create_time

4. **KM Transformation**: → `ml_analysis/km_transformed.csv`
   - Log + scale for right-skewed features (see [KMValidation.md](./KMValidation.md))
   - Scale [0-1] for normalized features
   - Cyclical encoding for create_time
   - Scaler fitting per bucket (see [Kmeans.md](./Kmeans.md))

5. **ML Training**: → `models/*.pkl` + `ml_analysis/*_analysis.json`
   - Train Random Forest and K-Means models
   - Generate analysis JSONs for LLM input
   - See [ML_LLMData.md](./ML_LLMData.md) for JSON schemas

**Next Steps**: LLM report generation (see [MLPlanning.md](./MLPlanning.md) "ML Analysis Pipeline" section)

---

### Cross-References

**Machine-readable config**: See [feature_transforms.json](./feature_transforms.json)

**Feature definitions**: See [TotalFeatures.md](./TotalFeatures.md)

**Content analysis system**: See [ContentAnalysis.md](../../ContentAnalysis.md)

**ML Pipeline**: See [MLPlanning.md](./MLPlanning.md) for complete training workflow and file architecture

**Temporal Window Structure**: See [QUICK_REFERENCE.md](../../QUICK_REFERENCE.md) for window calculation logic

**Validation**: See [KMValidation.md](./KMValidation.md) for post-collection feature validation

**Scaler Details**: See [Kmeans.md](./Kmeans.md) for K-Means scaler fitting process
