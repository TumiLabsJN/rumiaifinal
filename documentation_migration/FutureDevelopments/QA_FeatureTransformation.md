# Clarification Q&A: Feature Transformation

> **Mother Doc**: MLPlanningv2.md Section "Stage 4: Feature Transformation" (Lines 1360-1586)
> **Phase 1**: Critique_FeatureTransformation.md (APPROVED - Triple Pipeline Architecture)
> **Date**: 2025-10-13
> **Status**: COMPLETE

## Questions by Category

### Input/Output Contracts

#### Q1: [CRITICAL] Exact Input Schema - aggregated_features.csv Complete Base Features

**Question**: MLPlanningv2.md Section 4 (Line 1364) states the input is `ml_analysis/aggregated_features.csv` with "bucket-specific feature count: ~65-215 features." What are the COMPLETE 21 base features per window, their data types, and valid ranges?

**Answer**: From FeatureTransformation.md (lines 62-86), there are exactly **21 base features per temporal window**:

| # | Feature Name | Data Type | Valid Range | Notes |
|---|--------------|-----------|-------------|-------|
| 1 | average_face_size | Float | [0-1] | Already normalized |
| 2 | overlay_unique_count | Integer | [0-∞] | Count feature |
| 3 | has_captions | Boolean | True/False | Binary categorical |
| 4 | scene_count | Integer | [0-∞] | Count feature |
| 5 | shortest_scene | Float | [0-∞] | Duration in seconds |
| 6 | longest_scene | Float | [0-∞] | Duration in seconds |
| 7 | scene_duration_variance | Float | [0-∞] | Right-skewed variance |
| 8 | object_count | Integer | [0-∞] | Non-person objects |
| 9 | person_count | Integer | [0-∞] | Max simultaneous persons |
| 10 | speech_coverage | Float | [0-1] | % of segment with speech |
| 11 | word_count | Integer | [0-∞] | Total words spoken |
| 12 | energy_level | Float | [0-1] | Mean audio intensity |
| 13 | energy_variance | Float | [0-∞] | Audio intensity variance |
| 14 | energy_max | Float | [0-1] | Peak audio intensity |
| 15 | pitch_scatter_ratio | Float | [0-1] | Pitch instability |
| 16 | gesture_count | Integer | [0-∞] | Hand movement count |
| 17 | gaze_variance | Float | [0-∞] | Gaze stability variance |
| 18 | eye_contact_rate | Float | [0-1] | Eye contact percentage |
| 19 | dominant_emotion_id | Categorical | 1-7 | joy, sadness, anger, fear, disgust, surprise, neutral |
| 20 | emotional_valence | Float | [-1, 1] | Positive vs negative tone |
| 21 | emotion_consistency | Float | [0, 1] | Emotional focus consistency |

**Metadata features (video-level, NOT per-window)**:
- `create_time` (String, ISO 8601 timestamp)
- `gender_detection` (Object, nested with gender_label field)
- `duration` (Float, video duration in seconds)

**Bucket 18-33s CSV structure**:
- 21 base features × 6 windows (hook + middle_1-4 + closing) = 126 temporal columns
- Plus 3 metadata columns = **129 total columns**

**Column naming convention**:
- Temporal: `{window}_{feature}` (e.g., `hook_scene_count`, `middle_1_word_count`, `closing_eye_contact_rate`)
- Metadata: `create_time`, `gender_detection`, `duration`

**For HLD Sections**:
- 5.1 (Input Schema) - Complete column table ✅
- 6.1 (Input Validation) - Valid ranges for validation logic ✅
- 2.3 (Detailed Process) - Which features get which transformations ✅

#### Q2a: [CRITICAL] Video-Level RF Output Schema - Exact Column Count Calculation

**Question**: MLPlanningv2.md Section 4.1 (Line 1417) states Video-Level RF output `rf_transformed.csv` has "~190 features for bucket 18-33s." Where are the ~57 features beyond the 133 I calculated? Are cross-window derived features created in Stage 4?

**Answer**: Based on the Feature Transformation Table provided, Video-Level RF does NOT create cross-window derived features. The ~190 count comes from transforming the 21 base features per window:

**Per-window RF transformation** (from table):
- 18 features with "Direct" transform → 18 features (average_face_size, overlay_unique_count, scene_count, shortest_scene, longest_scene, scene_duration_variance, object_count, person_count, speech_coverage, word_count, energy_level, energy_variance, energy_max, pitch_scatter_ratio, gesture_count, gaze_variance, eye_contact_rate, emotional_valence, emotion_consistency)
- has_captions "One-hot" → 2 features (no_captions, has_captions)
- dominant_emotion_id "One-hot" → 7 features (joy, sadness, anger, fear, disgust, surprise, neutral)

**Total per window: 18 + 2 + 7 = 27 features**

Wait, let me recount including emotional_valence and emotion_consistency:
- 17 "Direct" features (numbers 1, 2, 4-18 from list)
- has_captions (one-hot) = 2 features
- dominant_emotion_id (one-hot) = 7 features
- emotional_valence (Direct) = 1 feature
- emotion_consistency (Direct) = 1 feature

**Corrected: 17 + 2 + 7 + 1 + 1 = 28 features per window**

**For bucket 18-33s (6 windows)**:
- 28 features × 6 windows = 168 temporal features

**Plus metadata (video-level)**:
- create_time "Extract-date" → 5 features (hour, day_of_week, month, is_weekend, is_business_hours)
- gender_detection "Extract + one-hot" → 2-3 features (gender_male, gender_female, possibly gender_unknown)
- duration (Direct) → 1 feature

**Plus target variable** (contrastive strategy only):
- is_top_performer → 1 feature

**Total: 168 + 5 + 3 + 1 + 1 = 178 features**

The ~190 in MLPlanningv2.md appears to be an approximation. Actual count is ~178-180 features for bucket 18-33s.

**Key Finding**: Stage 4 Video-Level RF does NOT derive cross-window features (no energy_progression, consistency_ratios, etc.). It only applies transformation rules from the table (Direct, One-hot, Extract-date).

**For HLD Sections**:
- 5.2 (Output Schema) - Exact count: ~178-180 features for bucket 18-33s ✅
- 2.3.1 (Video-Level RF Process) - No cross-window feature derivation in Stage 4 ✅

**Notes**: Cross-window features mentioned in Critique Q13 are NOT part of Stage 4. Video-Level RF uses raw temporal features as-is after basic transformation (Direct/One-hot).

#### Q2b: [CRITICAL] Window-Level RF Output Schema - Raw Features, No Transformation

**Question**: Does Window-Level RF apply transformations (one-hot, etc.) to the 21 base features, or use them raw? What are the exact column names?

**Answer**: By comparing MLPlanningv2.md Section 4.2 (Window-Level RF) vs Section 4.3 (Window-Level K-Means), the difference is clear:

**Window-Level RF** (Lines 1443-1455):
```python
window_features = df[[f'{window_type}_{feat}' for feat in BASE_FEATURES]]
window_features.columns = BASE_FEATURES  # Remove prefix
window_features['is_top_performer'] = (...)
window_features.to_csv(...)  # NO transformation applied before save
```
**Output**: 21 base features + 1 target = **22 columns** (Line 1455)

**Window-Level K-Means** (Lines 1479-1518):
```python
window_features = df[[f'{window_type}_{feat}' for feat in BASE_FEATURES]]
window_features.columns = BASE_FEATURES  # Remove prefix
df_km_window = window_features.copy()

# EXPLICIT transformation logic (30+ lines)
for feature in count_features:
    df_km_window[f'{feature}_log'] = np.log1p(...)
    df_km_window[f'{feature}_scaled'] = (...)
    df_km_window.drop(columns=[feature], inplace=True)  # Drop originals

df_km_window.to_csv(...)  # Save AFTER transformation
```
**Output**: ~21-40 transformed features (Line 1520)

**Confirmed findings**:

1. **Column names**: YES, window prefix is removed
   - `hook_rf_transformed.csv` has: `scene_count`, `eye_contact_rate`, `word_count`, etc. (NOT `hook_scene_count`)
   - All 6 window files have identical column names (just different data rows)

2. **Transformation applied**: NO transformation for Window-Level RF
   - Uses RAW 21 base features directly from `aggregated_features.csv`
   - NO one-hot encoding (has_captions stays Boolean, dominant_emotion_id stays 1-7)
   - NO log transforms, NO scaling
   - This is different from Video-Level RF which applies one-hot to has_captions and dominant_emotion_id

3. **Column count**: Exactly 22 columns per file
   - 21 raw base features + 1 target variable (`is_top_performer`)

4. **Metadata exclusion**: Confirmed NO metadata
   - NO `create_time`, NO `gender_detection`, NO `duration`
   - Only the 21 base features + target

**Why no transformation for Window-Level RF?**
- Random Forest is scale-invariant (doesn't need normalization)
- Boolean and categorical features can be used directly by tree-based models
- Simplifies the transformation pipeline

**For HLD Sections**:
- 5.2 (Output Schema) - 22 columns: 21 raw base features + is_top_performer ✅
- 2.3.2 (Window-Level RF Process) - Extract → Remove prefix → Add target → Save (no transformation) ✅

#### Q2c: [CRITICAL] Window-Level K-Means Output Schema - INCOMPLETE CODE IN MLPlanningv2.md

**Question**: What are the exact transformed columns for Window-Level K-Means output? MLPlanningv2.md Section 4.3 only shows transformations for 13 out of 21 features.

**Answer**: By comparing MLPlanningv2.md Section 4.3 with FeatureTransformation.md table, **the pseudocode in MLPlanningv2.md is INCOMPLETE**. It's missing transformations for 8 features:

**Features covered in MLPlanningv2.md Section 4.3** (13 features):
1. **5 count features** (Log + scale): scene_count, word_count, gesture_count, object_count, person_count
2. **8 rate features** (Scale [0-1]): eye_contact_rate, speech_coverage, emotional_valence, emotion_consistency, energy_level, energy_variance, pitch_scatter_ratio, gaze_variance

**Features MISSING from MLPlanningv2.md Section 4.3** (8 features):

**Missing Log + scale transformations** (4 features):
- overlay_unique_count (Integer [0-∞], skewed distribution)
- shortest_scene (Float [0-∞], extreme outliers)
- longest_scene (Float [0-∞], extreme outliers)
- scene_duration_variance (Float [0-∞], right-skewed variance)

**Missing Scale [0-1] transformations** (2 features):
- average_face_size (Float [0-1], already normalized)
- energy_max (Float [0-1], already normalized)

**Missing categorical transformations** (2 features):
- has_captions (Boolean) → Label encode → 1 feature
- dominant_emotion_id (Categorical 1-7) → One-hot → 7 features

**CONFLICT RESOLVED - energy_variance and gaze_variance**:
- MLPlanningv2.md Line 1504: Incorrectly lists these as "rate features" for Scale [0-1] only (ERROR in pseudocode)
- FeatureTransformation.md Lines 76, 80: Correctly specifies "Log + scale" (they're variances, right-skewed, [0-∞] range)
- **User Decision**: Use Log + scale (Option A) - Variances need log transform to compress right-skewed distributions

**Rationale for Log + scale**:
- Data range [0-∞] (unbounded, not normalized)
- Explicitly labeled "Right-skewed variance"
- Consistent with scene_duration_variance treatment
- Prevents extreme variance outliers from dominating K-Means distance calculations

**Complete Window-Level K-Means Output Schema** (39 columns total):

**Log + scale** (11 features → 22 output columns):
1. scene_count → scene_count_log, scene_count_scaled
2. word_count → word_count_log, word_count_scaled
3. gesture_count → gesture_count_log, gesture_count_scaled
4. object_count → object_count_log, object_count_scaled
5. person_count → person_count_log, person_count_scaled
6. overlay_unique_count → overlay_unique_count_log, overlay_unique_count_scaled
7. shortest_scene → shortest_scene_log, shortest_scene_scaled
8. longest_scene → longest_scene_log, longest_scene_scaled
9. scene_duration_variance → scene_duration_variance_log, scene_duration_variance_scaled
10. energy_variance → energy_variance_log, energy_variance_scaled
11. gaze_variance → gaze_variance_log, gaze_variance_scaled

**Scale [0-1]** (7 features → 7 output columns):
1. average_face_size → average_face_size_scaled
2. speech_coverage → speech_coverage_scaled
3. energy_level → energy_level_scaled
4. energy_max → energy_max_scaled
5. pitch_scatter_ratio → pitch_scatter_ratio_scaled
6. eye_contact_rate → eye_contact_rate_scaled
7. emotion_consistency → emotion_consistency_scaled

**Shift + scale** (1 feature → 1 output column):
1. emotional_valence → emotional_valence_scaled (shifts [-1,1] to [0,1])

**Label encode** (1 feature → 1 output column):
1. has_captions → has_captions_encoded (0 or 1)

**One-hot** (1 feature → 7 output columns):
1. dominant_emotion_id → joy, sadness, anger, fear, disgust, surprise, neutral

**Total: 22 + 7 + 1 + 1 + 7 = 39 columns**

**For HLD Sections**:
- 5.2 (Output Schema) - Complete 39-column schema documented ✅
- 2.3.3 (Window-Level K-Means Process) - Must document COMPLETE transformation logic (not partial MLPlanningv2.md code) ✅
- 10.1 (Known Issues) - Note that MLPlanningv2.md Section 4.3 has incomplete transformation code ✅

### Dependencies & Integration

#### Q3: [CRITICAL] File Paths and Directory Structure

**Question**: What are the COMPLETE file paths for Stage 4 input and outputs? MLPlanningv2.md shows `bucket_18-33s/ml_analysis/` but I need the full path from `/data/clients/...`

**Answer**: User confirmed option 1 - files are under `ml_analysis/` subdirectory:

**Input file path**:
```
/data/clients/{client_id}/hashtags/{cluster_id}/{mode}_{strategy}/buckets/bucket_18-33s/ml_analysis/aggregated_features.csv
```

**Output file paths** (13 files for bucket 18-33s, all under same `ml_analysis/` directory):
```
/data/clients/{client_id}/hashtags/{cluster_id}/{mode}_{strategy}/buckets/bucket_18-33s/ml_analysis/
├── rf_transformed.csv                   # Video-level RF (1 file)
├── hook_rf_transformed.csv              # Window-level RF (6 files)
├── middle_1_rf_transformed.csv
├── middle_2_rf_transformed.csv
├── middle_3_rf_transformed.csv
├── middle_4_rf_transformed.csv
├── closing_rf_transformed.csv
├── hook_km_transformed.csv              # Window-level K-Means (6 files)
├── middle_1_km_transformed.csv
├── middle_2_km_transformed.csv
├── middle_3_km_transformed.csv
├── middle_4_km_transformed.csv
└── closing_km_transformed.csv
```

**Path variables**:
- `{client_id}`: Client identifier (e.g., "nike", "cocacola")
- `{cluster_id}`: Hashtag cluster identifier (e.g., "fitness_2024-01-15")
- `{mode}_{strategy}`: Analysis mode + strategy (e.g., "top_contrastive", "recent_top")
- `bucket_18-33s`: Duration bucket (one of 8 buckets: 0-3s, 3-9s, 9-13s, 13-18s, 18-33s, 33-60s, 60-90s, 90-120s)

**Path template source**: References Foundation path templates from MLPlanningv2.md Part 1 (Lines 116-236)

**For HLD Sections**:
- 3.1 (Input Dependencies) - Input path documented ✅
- 3.2 (Output Contracts) - All 13 output paths documented ✅
- 3.4 (External Dependencies) - Foundation path templates referenced ✅

[Questions will be filled iteratively]

### Edge Cases & Validation

#### Q4: [CRITICAL] Input Validation - Missing or Invalid Data

**Question**: How should Stage 4 handle data quality issues in `aggregated_features.csv`? (Missing values, missing columns, wrong data types, out-of-range values)

**Answer**: User confirmed **fail-fast approach with clear error messages** for all data quality issues:

**Scenario 1: Missing values (NaN)** → **Fail-fast (Option A)**
- Error message: `"Invalid input: 5 rows contain NaN values in hook_scene_count. Check Stage 3 aggregation logic."`
- Rationale: NaN indicates upstream bug, imputation creates synthetic data, dropping rows reduces sample size

**Scenario 2: Missing columns** → **Fail-fast (Option A)**
- Error message: `"Required column missing: hook_eye_contact_rate. Expected 126 temporal columns (21 features × 6 windows), found 120."`
- Rationale: All 21 base features are contractually expected from Stage 3, missing columns break model consistency

**Scenario 3: Wrong data type** → **Accept with semantic validation (Option C)**
- Accept: `scene_count` as float (3.0) vs int (3) doesn't affect transformations
- Validate: Check `scene_count >= 0` and `scene_count < 10000` (sanity bounds)
- Rationale: Python/pandas handles numeric type coercion naturally, but still validate semantics

**Scenario 4: Out-of-range values** → **Fail-fast (Option A)**
- Error message: `"Out of range: hook_eye_contact_rate has value 1.5, expected [0.0-1.0]. Check Stage 2 eye contact calculation."`
- Rationale: Out-of-range values indicate data corruption or upstream bugs, clipping hides problems

**Exception - Row count flexibility**:
- If N < expected (e.g., 95 instead of 100):
  - **Warning** if N >= 50: `"Warning: Expected 100 videos, found 95. Proceeding with reduced sample size."`
  - **Fail** if N < 50: `"Insufficient data: 45 videos found, minimum 50 required for ML training."`
- Allows some upstream failures without halting entire pipeline

**Validation logic summary**:
1. Check column count matches expected (bucket-specific)
2. Check all required columns exist
3. Check for NaN values (fail if any found)
4. Check normalized features [0-1] are in range
5. Check count features [0-∞] are non-negative with sanity bounds (<10000)
6. Check minimum row count (N >= 50)

**For HLD Sections**:
- 6.1 (Input Validation) - Pre-transformation validation step with fail-fast rules ✅
- 6.2 (Error Cases) - Specific error messages for each validation failure ✅
- 2.3 (Detailed Process) - Add validation step before transformation ✅

[Questions will be filled iteratively]

### Performance & Scale

#### Q5: [CRITICAL] Performance Targets and Bottlenecks

**Question**: What are the acceptable performance targets for Stage 4? (Processing time, memory constraints, timeout threshold, expected bottlenecks)

**Answer**: User confirmed performance targets based on pandas operations on N=100 videos:

**1. Target Processing Time: <30 seconds**
- Expected actual time: 5-15 seconds (pandas operations on 100 rows × 200 columns)
- Target provides 2-3× buffer for slower machines or HDD I/O
- Warning if >1 minute
- Fail if >5 minutes (timeout)

**2. Memory Constraints: <500 MB peak**
- Expected actual memory: <50 MB (100 rows × 180 columns × 8 bytes + pandas overhead ≈ 20-30 MB)
- Target provides 10× buffer
- Warning if >1 GB (indicates memory leak or data duplication)
- Fail if >2 GB (prevent OOM crashes)

**3. Timeout Threshold: 5 minutes**
- 10× buffer over 30-second target
- Catches hangs (disk I/O issues, infinite loops) without false positives
- Rationale: If Stage 4 takes >5 minutes for 100 rows, critical issue exists

**4. Expected Bottlenecks (priority order)**:
1. **Disk I/O** (writing 13 CSV files)
   - HDD: 5-10 seconds, SSD: <1 second
   - Most significant performance variability
   - Mitigation: Use buffered writes

2. **One-hot encoding** (Video-Level RF, dominant_emotion_id)
   - Creates 7 columns from 1 categorical
   - Expected: 1-2 seconds for 100 rows
   - Mitigation: None needed (pandas optimized)

3. **Window-Level K-Means transformations** (6 iterations)
   - 66 log operations + 108 scale operations
   - Expected: 2-4 seconds total
   - Mitigation: None needed (numpy vectorization fast)

4. **Validation** (pre-transformation checks)
   - Expected: <1 second
   - Not a bottleneck

**Performance Logging Requirements**:
- Log actual time: `"Stage 4 completed in 8.3 seconds (target: <30s)"`
- Log peak memory: `"Peak memory: 42 MB (target: <500 MB)"`
- Log per-operation time: `"Video-Level RF: 2.1s, Window-Level RF: 1.3s, Window-Level K-Means: 3.8s, I/O: 1.1s"`

**Summary Table**:

| Metric | Target | Warning | Fail | Rationale |
|--------|--------|---------|------|-----------|
| Processing Time | <30 seconds | >1 minute | >5 minutes | 2-3× expected with buffer |
| Peak Memory | <500 MB | >1 GB | >2 GB | 10× actual need with buffer |
| Expected Bottleneck | Disk I/O (HDD) | - | - | Writing 13 CSV files |

**For HLD Sections**:
- 7.1 (Performance Targets) - Concrete time/memory targets documented ✅
- 7.3 (Bottlenecks) - Expected slow operations identified ✅
- 6.2 (Error Cases) - Timeout handling specified ✅

[Questions will be filled iteratively]

### Error Handling

#### Q6: [CRITICAL] Cross-Stage Dependencies and Execution Context

**Question**: How does Stage 4 integrate with the pipeline? (Execution trigger, bucket context, multi-bucket execution, completion detection)

**Answer**: Based on deep documentation analysis, the ML pipeline uses **single-command orchestration with file-based contracts**:

**1. Execution Trigger: Sequential orchestration via single CLI command**
- **Single entry point**: All stages invoked through one command:
  ```bash
  python rumiai_ml_batch.py \
    --client "nike" \
    --analysis-type hashtag \
    --target "nutrition" \
    --analysis-mode top \
    --selection-strategy contrastive \
    --video-count 100
  ```
- **Internal orchestration**: `rumiai_ml_batch.py` handles all stage progression (Stage 1 → 2 → 3 → 4 → 5 → 7)
- **No individual stage scripts**: No commands like `python stage4_transform.py`
- Source: MLPlanningv2.md lines 278-289

**2. Bucket Context: File-based discovery from directory structure**
- **No CLI --bucket flag**: Buckets discovered from file system structure
- **Stage 1 creates buckets**: All 8 bucket directories created upfront: `bucket_0-3s/`, `bucket_3-9s/`, ..., `bucket_90-120s/`
- **selected_videos.json contract**: Each bucket has this file if it contains videos:
  ```json
  // Location: /data/clients/{client}/hashtags/{cluster}/{mode}_{strategy}/buckets/bucket_18-33s/selected_videos.json
  {
    "bucket": "18-33s",
    "videos": [...],  // 100 videos
    "selection_metadata": {
      "strategy": "contrastive",
      "top_count": 80,
      "bottom_count": 20
    }
  }
  ```
- **Path construction**: Stages read config and construct paths dynamically:
  ```python
  bucket_path = f"/data/clients/{config['client_id']}/hashtags/{config['target']}/{config['mode']}_{config['strategy']}/buckets/bucket_{bucket_name}/"
  ```
- Source: VideoProcessingCHILD.md lines 406-430

**3. Multi-Bucket Execution: Sequential (one bucket at a time)**
- **Top 3 buckets processed**: Stage 1 identifies winning buckets (e.g., 18-33s, 33-60s, 13-18s)
- **Sequential flow**:
  ```
  Process bucket_18-33s (Stages 2-7) → checkpoint after each stage
     ↓
  Process bucket_33-60s (Stages 2-7) → checkpoint after each stage
     ↓
  Process bucket_13-18s (Stages 2-7) → checkpoint after each stage
  ```
- **Why sequential**: Checkpoint-based resume, fail-fast architecture, resource constraints (CPU/GPU intensive)
- **Explicitly stated**: "Processing: Sequential (one-by-one) with resumption capability" (MLPlanningv2.md line 107)

**4. Stage Completion Detection: File existence checks + checkpoint status validation**
- **Stage 3 → Stage 4 contract**:
  - Stage 3 creates: `aggregated_features.csv` in `bucket_{duration}/ml_analysis/`
  - Stage 4 validates: File exists + checkpoint status == "completed"

- **Checkpoint-based completion**:
  ```json
  // bucket_18-33s/checkpoints/stage_3_checkpoint.json
  {
    "stage": "feature_aggregation",
    "status": "completed",  // ← Completion signal
    "total_videos": 100,
    "aggregated_csv_path": "ml_analysis/aggregated_features.csv",
    "completion_time": "2025-01-28T16:45:30Z"
  }
  ```
- **No .complete marker files**: Checkpoint JSON with status field serves as completion marker
- **Validation before Stage 4**: Orchestrator checks `status == "completed"` and file existence

**Stage 4 Execution Flow**:
```
1. Orchestrator detects Stage 3 completed (checkpoint status + file exists)
2. Orchestrator calls Stage 4 function with bucket_path parameter
3. Stage 4 validates input (aggregated_features.csv exists, correct schema)
4. Stage 4 processes transformations (13 files)
5. Stage 4 writes checkpoint: {"stage": "feature_transformation", "status": "completed"}
6. Orchestrator proceeds to Stage 5
```

**Failure Handling**:
- **Fail-fast for infrastructure**: Missing `aggregated_features.csv` → exit code 2, log error, stop pipeline
- **Skip-on-fail for data quality**: If Stage 4 validation finds corrupted data → log error, skip bucket, continue with next bucket
- **Checkpoint enables resume**: If Stage 4 fails mid-transformation, can resume from checkpoint

**For HLD Sections**:
- 3.3 (Cross-Stage Dependencies) - File-based contracts with Stage 3/5 documented ✅
- 4.1 (Tech Stack) - No CLI interface (called by orchestrator internally) ✅
- 2.2 (Data Flow) - Sequential bucket processing with checkpoint gates ✅
- 6.2 (Error Cases) - Checkpoint-based error recovery ✅

[Questions will be filled iteratively]

### Testing

#### Q7: [HIGH] Testing Strategy - Test Scenarios and Data

**Question**: What test data and scenarios should be used to validate Stage 4 works correctly?

**Answer**: User confirmed **two-layer testing strategy** (Unit + Integration):

**Layer 1: Unit Tests with Synthetic Data** (Fast, Isolated)

**Purpose**: Test individual transformation functions in isolation

**Test Data**: Small synthetic CSV fixtures created in test suite
- `test_bucket_18-33s_minimal.csv` (10 videos, 129 columns)
- `test_bucket_9-13s_minimal.csv` (10 videos, 66 columns with middle_aggregate)
- `test_bucket_3-9s_minimal.csv` (10 videos, 45 columns, hook + closing only)
- Location: `/tests/fixtures/stage4/`

**Test Coverage**:
1. **Individual transformations**:
   - Log + scale: Verify `scene_count: 3` → `scene_count_log: 1.386`, `scene_count_scaled: 0.5`
   - One-hot: Verify `dominant_emotion_id` produces 7 columns (joy, sadness, anger, fear, disgust, surprise, neutral)
   - Shift + scale: Verify `emotional_valence: -0.5` → `emotional_valence_scaled: 0.25` (shifts [-1,1] to [0,1])

2. **Window extraction**:
   - Bucket 18-33s: Extract 6 windows (hook, middle_1-4, closing)
   - Bucket 9-13s: Extract 3 windows (hook, middle_aggregate, closing)
   - Bucket 3-9s: Extract 2 windows (hook, closing)

3. **Schema validation**:
   - Video-Level RF: ~178-180 columns
   - Window-Level RF: 22 columns (21 base + target)
   - Window-Level K-Means: 39 columns

**Expected Runtime**: <1 second total

**Layer 2: Integration Tests with Real Stage 3 Output** (Realistic)

**Purpose**: Validate end-to-end transformation with real data patterns

**Test Data**: Capture real `aggregated_features.csv` from test analysis run
- **Creation process**:
  1. Run small test analysis: `python rumiai_ml_batch.py --client test --target fitness --video-count 50`
  2. After Stage 3 completes, copy output CSV to test fixtures
  3. Store as: `/tests/fixtures/stage4/real_bucket_18-33s_stage3_output.csv` (50 videos, ~20 KB)
  4. Commit to git (version control real test data)

**Test Coverage**:
1. **End-to-end processing**:
   - Load real Stage 3 output
   - Run Stage 4 transformation
   - Verify all 13 files created successfully

2. **Output schema validation**:
   - Video-Level RF: 175-185 columns (allow range for flexibility)
   - Window-Level RF: Exactly 22 columns per file
   - Window-Level K-Means: Exactly 39 columns per file
   - Required columns exist (hook_scene_count, is_top_performer, joy, etc.)

3. **Value range validation**:
   - All `_scaled` columns in K-Means are [0-1]
   - All one-hot encoded columns are {0, 1}
   - All log-transformed values are non-negative

4. **Row preservation**:
   - Input: 50 rows → Output: 50 rows in all files
   - No rows dropped during transformation

**Expected Runtime**: <5 seconds

**Test File Structure**:
```
tests/
├── unit/
│   ├── test_video_level_rf.py
│   ├── test_window_level_rf.py
│   ├── test_window_level_kmeans.py
│   └── test_validation.py
├── integration/
│   └── test_stage4_full_pipeline.py
└── fixtures/
    └── stage4/
        ├── test_bucket_18-33s_minimal.csv          # 10 videos, synthetic
        ├── test_bucket_9-13s_minimal.csv           # 10 videos, with middle_aggregate
        ├── test_bucket_3-9s_minimal.csv            # 10 videos, hook + closing only
        └── real_bucket_18-33s_stage3_output.csv    # 50 videos, real data
```

**Total Test Runtime**: ~6 seconds (unit + integration)

**Not Included** (deferred):
- Edge case tests (minimum/maximum bucket sizes, all 8 bucket types)
- Golden reference tests (regression detection)
- Performance tests (timeout validation)

**For HLD Sections**:
- 8.1 (Unit Tests) - Synthetic data with known transformations ✅
- 8.2 (Integration Tests) - Real Stage 3 output validation ✅
- 8.3 (Test Data) - Fixture creation and storage strategy ✅

## Completeness Check

Can write these HLD sections without TODOs or gaps?

### Section 2 (Architecture & Design)
- **2.1: High-level approach** - ✅ YES - Triple pipeline architecture documented (Video-Level RF + Window-Level RF + Window-Level K-Means)
- **2.2: Data flow** - ✅ YES - Sequential bucket processing with checkpoint gates (Q6)
- **2.3: Detailed process** - ✅ YES - Complete transformation logic for all 3 pipelines (Q2a, Q2b, Q2c), including validation step (Q4)

### Section 3 (Dependencies & Integration)
- **3.1: Input dependencies** - ✅ YES - Input path documented (Q3), Stage 3 contract defined (Q6)
- **3.2: Output contracts** - ✅ YES - All 13 output file paths documented (Q3), exact schemas (Q2a, Q2b, Q2c)
- **3.3: Cross-stage dependencies** - ✅ YES - File-based contracts with Stage 3/5, checkpoint-based completion detection (Q6)
- **3.4: External dependencies** - ✅ YES - Foundation path templates, FeatureTransformation.md table (Q1, Q2c, Q3)

### Section 5 (Data Schemas)
- **5.1: Input schema** - ✅ YES - Complete 21 base features with types and ranges documented (Q1)
- **5.2: Output schema** - ✅ YES - Video-Level RF (~178 cols), Window-Level RF (22 cols), Window-Level K-Means (39 cols) all documented (Q2a, Q2b, Q2c)

### Section 6 (Error Handling)
- **6.1: Input validation** - ✅ YES - Fail-fast rules for NaN, missing columns, out-of-range values (Q4)
- **6.2: Error cases** - ✅ YES - Specific error messages, timeout handling (5 min), checkpoint-based recovery (Q4, Q5, Q6)
- **6.3: Output validation** - ✅ YES - Schema checks, row count validation, range validation (Q4)

### Section 7 (Performance)
- **7.1: Performance targets** - ✅ YES - <30 seconds target, <500 MB memory, 5-minute timeout (Q5)
- **7.3: Bottlenecks** - ✅ YES - Disk I/O (primary), one-hot encoding (secondary), K-Means transforms (Q5)

### Section 8 (Testing Strategy)
- **8.1-8.3: Test cases** - ✅ YES - Unit tests (synthetic), Integration tests (real Stage 3 output), test fixtures documented (Q7)

## Proceed to Phase 3

**Ready for HLD Generation**: YES

**All critical information gathered**:
- ✅ Complete 21 base features with transformation rules
- ✅ Exact output schemas for all 3 transformation types (13 files per bucket)
- ✅ Complete file paths with Foundation path templates
- ✅ Fail-fast validation strategy with specific error messages
- ✅ Performance targets with concrete thresholds
- ✅ Pipeline integration via checkpoint-based orchestration
- ✅ Testing strategy with fixtures

**Key Findings**:
1. MLPlanningv2.md Section 4.3 has **incomplete transformation code** - missing 8 features (documented in Q2c)
2. energy_variance and gaze_variance should use **Log + scale** (not Scale [0-1] as shown in pseudocode)
3. Triple pipeline approved in Phase 1 - delivers 100% value with 1-week implementation cost
4. Stage 4 is called by orchestrator internally - no standalone CLI interface

**Status**: COMPLETE - Ready for Phase 3 (Child HLD Generation)
