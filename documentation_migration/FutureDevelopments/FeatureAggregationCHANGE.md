# Feature Aggregation - Middle Segment Aggregation Change

> **Parent Document**: FeatureAggregationCHILD.md
> **Related**: Question 2 from K-Means window-level clustering analysis
> **Date**: 2025-01-10
> **Status**: PROPOSED CHANGE - Not Yet Implemented

---

## 1. Change Summary

### 1.1 What Is Changing

Modify Stage 3 Feature Aggregation to **aggregate middle segments** for buckets 9-13s and 13-18s, instead of keeping them separate.

**Reason**: Middle segments in these buckets are too short (1-4 seconds) to produce reliable measurements for certain features (scene_count, scene_duration_variance, speech_coverage, word_count). Aggregating them creates a single longer window (4.5-9.3 seconds) where all 21 features become reliable.

### 1.2 Impact Summary

| Aspect | Before Change | After Change | Impact |
|--------|---------------|--------------|--------|
| **Bucket 9-13s** | 108 features (5 windows) | 66 features (3 windows) | -42 features |
| **Bucket 13-18s** | 108 features (5 windows) | 66 features (3 windows) | -42 features |
| **Other buckets** | No change | No change | 0 |
| **Feature reliability** | 13/21 reliable in short middles | 21/21 reliable in aggregated middle | +8 features |
| **Temporal granularity** | Separate middle_1, middle_2, middle_3 | Single middle_aggregate | Lost progression for 9-18s videos |

### 1.3 Affected Stages

| Stage | Impact | Changes Required |
|-------|--------|------------------|
| **Stage 3 (Feature Aggregation)** | ✅ Major | Modify extract_features() logic + update configs |
| **Stage 4 (Feature Transformation)** | ✅ Minor | Column names change (middle_aggregate_* instead of middle_1_*, middle_2_*, middle_3_*) but transformation logic unchanged |
| **Stage 5 (ML Training)** | ✅ Minor | Models for 9-13s and 13-18s now have 66 features instead of 108 |
| **Stage 6 (ML Analysis)** | ✅ Minor | JSON outputs reference middle_aggregate features |
| **Stage 7 (LLM Reports)** | ✅ Minor | Reports say "Middle segment (aggregated)" instead of "Middle segments 1-3" |

---

## 2. Technical Changes

### 2.1 File: FeatureAggregationCHILD.md

#### Change 1: Update Section 2.3.2 - Feature Extraction Logic

**Location**: Lines 142-227 (extract_features function)

**Current Code** (lines 197-211):
```python
# Middle features (0-5 segments depending on bucket)
middle_segments = windows.get('middle_segments')
if middle_segments is None or len(middle_segments) == 0:
    # For buckets 0-3s, 3-9s: middle_segments is null (expected)
    # For longer buckets: this is an error (Source: QA Q5)
    if bucket not in ['0-3s', '3-9s']:
        raise ValueError(
            f"Video {video_id}: null or empty middle_segments "
            f"(bucket {bucket} requires middle segments)"
        )
else:
    # Extract features from each middle segment
    for i, segment in enumerate(middle_segments, start=1):
        for feature in BASE_FEATURES:
            video_features[f'middle_{i}_{feature}'] = segment.get(feature)
```

**New Code**:
```python
# Middle features (0-5 segments depending on bucket)
middle_segments = windows.get('middle_segments')

if middle_segments is None or len(middle_segments) == 0:
    # For buckets 0-3s, 3-9s: middle_segments is null (expected)
    if bucket not in ['0-3s', '3-9s']:
        raise ValueError(
            f"Video {video_id}: null or empty middle_segments "
            f"(bucket {bucket} requires middle segments)"
        )
else:
    # **NEW: Aggregate middle segments for short-window buckets**
    if bucket in AGGREGATE_MIDDLE_BUCKETS:
        # Aggregate all middle segments into single "middle_aggregate"
        # Reason: Short windows (1-4s) produce unreliable scene/speech features
        # Aggregation creates longer window (4.5-9.3s) for reliable measurements

        import numpy as np

        for feature in BASE_FEATURES:
            # Collect feature values from all middle segments
            feature_values = [
                seg.get(feature)
                for seg in middle_segments
                if seg.get(feature) is not None
            ]

            # Average them (skip if all values are None)
            if len(feature_values) > 0:
                video_features[f'middle_aggregate_{feature}'] = np.mean(feature_values)
            else:
                video_features[f'middle_aggregate_{feature}'] = None

        logger.debug(
            f"Video {video_id}: Aggregated {len(middle_segments)} middle segments "
            f"into middle_aggregate (bucket {bucket} has short windows)"
        )

    else:
        # **ORIGINAL: Keep separate middle segments for longer buckets**
        # Buckets 18-33s, 33-60s, 60-90s, 90-120s have longer windows (3-22s)
        # All features reliable at this duration
        for i, segment in enumerate(middle_segments, start=1):
            for feature in BASE_FEATURES:
                video_features[f'middle_{i}_{feature}'] = segment.get(feature)
```

**Why this change**:
- Preserves all feature values (averages instead of discards)
- Clearly documents why aggregation happens (short window reliability issue)
- Uses numpy.mean() for clean averaging (handles mixed int/float types)
- Logs aggregation for debugging (can verify in logs which videos were aggregated)
- Maintains backward compatibility for buckets 18-33s+

---

#### Change 2: Update Section 4.2 - Internal Configuration

**Location**: Lines 472-522 (Configuration constants)

**Current Code** (lines 500-522):
```python
# Bucket configurations (window counts) - Source: MLPlanningv2.md Stage 3
BUCKET_MIDDLE_SEGMENTS = {
    '0-3s': 0,
    '3-9s': 0,
    '9-13s': 3,
    '13-18s': 3,
    '18-33s': 4,
    '33-60s': 5,
    '60-90s': 5,
    '90-120s': 5
}

# Expected feature counts (for validation)
EXPECTED_FEATURE_COUNTS = {
    '0-3s': 45,   # 21 × 2 windows + 3 metadata
    '3-9s': 45,
    '9-13s': 108,  # 21 × 5 windows + 3 metadata
    '13-18s': 108,
    '18-33s': 129, # 21 × 6 windows + 3 metadata
    '33-60s': 150, # 21 × 7 windows + 3 metadata
    '60-90s': 150,
    '90-120s': 150
}
```

**New Code**:
```python
# Bucket configurations (window counts) - Source: MLPlanningv2.md Stage 3
BUCKET_MIDDLE_SEGMENTS = {
    '0-3s': 0,
    '3-9s': 0,
    '9-13s': 3,
    '13-18s': 3,
    '18-33s': 4,
    '33-60s': 5,
    '60-90s': 5,
    '90-120s': 5
}

# **NEW: Buckets that aggregate middle segments (short windows)**
# These buckets have middle windows of 1-4s, which produce unreliable measurements
# for scene_count, scene_duration_variance, speech_coverage, word_count, etc.
# Aggregation creates 4.5-9.3s windows where all 21 features are reliable.
AGGREGATE_MIDDLE_BUCKETS = ['9-13s', '13-18s']

# Expected feature counts (for validation) - UPDATED with middle aggregation
EXPECTED_FEATURE_COUNTS = {
    '0-3s': 45,   # 21 × 2 windows + 3 metadata
    '3-9s': 45,
    '9-13s': 66,  # 21 × 3 windows (hook + middle_aggregate + closing) + 3 metadata [CHANGED from 108]
    '13-18s': 66, # 21 × 3 windows (hook + middle_aggregate + closing) + 3 metadata [CHANGED from 108]
    '18-33s': 129, # 21 × 6 windows + 3 metadata
    '33-60s': 150, # 21 × 7 windows + 3 metadata
    '60-90s': 150,
    '90-120s': 150
}
```

**Why this change**:
- Adds explicit `AGGREGATE_MIDDLE_BUCKETS` constant (single source of truth)
- Updates expected feature counts to match new structure (66 instead of 108)
- Documents the reliability reasoning in comments
- Makes it easy to add/remove buckets from aggregation list in future

---

#### Change 3: Update Section 3.2 - Output Contracts

**Location**: Lines 399-408

**Current Text**:
```
**Column Count by Bucket**:
- 0-3s, 3-9s: 45 features (21 × 2 windows + 3 metadata)
- 9-13s, 13-18s: 108 features (21 × 5 windows + 3 metadata)
- 18-33s: 129 features (21 × 6 windows + 3 metadata)
- 33-60s, 60-90s, 90-120s: 150 features (21 × 7 windows + 3 metadata)
```

**New Text**:
```
**Column Count by Bucket** (UPDATED - Middle Aggregation):
- 0-3s, 3-9s: 45 features (21 × 2 windows + 3 metadata)
- 9-13s, 13-18s: 66 features (21 × 3 windows + 3 metadata) [CHANGED - middle segments aggregated]
  - Structure: hook_* (21) + middle_aggregate_* (21) + closing_* (21) + metadata (3)
  - Reason: Short middle windows (1-4s) aggregated for feature reliability
- 18-33s: 129 features (21 × 6 windows + 3 metadata)
- 33-60s, 60-90s, 90-120s: 150 features (21 × 7 windows + 3 metadata)
```

**Why this change**:
- Updates documentation to match implementation
- Explains the structural difference (middle_aggregate vs middle_1/2/3)
- Provides reasoning for future maintainers

---

#### Change 4: Update Section 5.2 - Output Schema

**Location**: Lines 607-611

**Current Text**:
```
**Total Columns by Bucket**:
- 0-3s, 3-9s: 45 columns (2 windows × 21 features + 5 metadata)
- 9-13s, 13-18s: 108 columns (5 windows × 21 features + 3 metadata)
- 18-33s: 129 columns (6 windows × 21 features + 3 metadata)
- 33-60s, 60-90s, 90-120s: 150 columns (7 windows × 21 features + 3 metadata)
```

**New Text**:
```
**Total Columns by Bucket** (UPDATED - Middle Aggregation):
- 0-3s, 3-9s: 45 columns (2 windows × 21 features + 5 metadata)
- 9-13s, 13-18s: 66 columns (3 windows × 21 features + 3 metadata) [CHANGED]
  - Column naming: hook_*, middle_aggregate_*, closing_*, metadata
  - Note: middle_1_*, middle_2_*, middle_3_* replaced by single middle_aggregate_*
- 18-33s: 129 columns (6 windows × 21 features + 3 metadata)
- 33-60s, 60-90s, 90-120s: 150 columns (7 windows × 21 features + 3 metadata)
```

**Why this change**:
- Documents the column naming change
- Helps Stage 4 developers understand what to expect

---

#### Change 5: Add Section to Decision Log (Appendix A)

**Location**: After line 1045 (end of Appendix A)

**New Entry**:
```markdown
**Decision 7**: Aggregate middle segments for buckets 9-13s and 13-18s
- **Context**: Middle windows in buckets 9-13s (1-2.3s each) and 13-18s (2.3-4s each) are too short to produce reliable measurements for scene features (need 3+ seconds for multiple scene cuts) and speech features (need 2+ seconds for meaningful word counts). This affects 8/21 base features (38%).
- **Alternatives Considered**:
  - Option A: Keep separate middle segments, accept unreliable features - Rejected (pollutes ML training data)
  - Option B: Aggregate middle segments into single "middle_aggregate" window - **CHOSEN**
  - Option C: Remove unreliable features for these buckets only - Rejected (creates inconsistent feature sets across buckets)
  - Option D: Do nothing, rely on K-Means to handle noise - Rejected (high-dimensional noise degrades cluster quality)
- **Rationale**:
  - Feature reliability: 13/21 reliable in 1-2.3s windows → 21/21 reliable in 4.5-9.3s aggregated window
  - Bucket-specific models already handle different feature counts (45, 66, 129, 150)
  - Temporal granularity loss acceptable (middle progressions unreliable anyway in short segments)
  - Simpler than feature filtering (maintains consistent 21-feature schema across all windows)
- **Trade-offs**:
  - Lose middle segment progression for 9-18s videos (e.g., can't detect "word_count increases middle_1 → middle_3")
  - But: This progression data was unreliable due to short windows (noise, not signal)
  - Feature count reduced from 108 → 66 for these buckets (42 fewer features)
  - But: Fewer high-quality features better than more low-quality features for ML
- **Impact**:
  - Stage 3: +15 lines of code (aggregation logic)
  - Stage 4-7: Column name changes only (middle_aggregate_* instead of middle_1_*, middle_2_*, middle_3_*)
  - K-Means clustering: Better quality (21 reliable features instead of 13 reliable + 8 noisy)
  - Downstream stages: No logic changes (just different column names in DataFrame)
- **Date**: 2025-01-10 (Source: Question 2 analysis of temporal window reliability)
```

**Why this addition**:
- Documents the reasoning for future maintainers
- Captures the alternatives considered (prevents re-litigating the decision)
- Explains the trade-offs explicitly

---

#### Change 6: Update Section 1.1 - Problem Statement

**Location**: Lines 10-14

**Current Text**:
```
Machine learning algorithms require fixed-size feature vectors, but raw temporal window data has variable-length structures (2-7 windows depending on video duration). This component eliminates the ragged array problem by organizing videos into duration buckets where all videos share identical temporal window structures. Each bucket processes videos with consistent window counts, enabling direct CSV aggregation while preserving full temporal granularity and narrative pacing patterns critical for creative analysis.
```

**New Text**:
```
Machine learning algorithms require fixed-size feature vectors, but raw temporal window data has variable-length structures (2-7 windows depending on video duration). This component eliminates the ragged array problem by organizing videos into duration buckets where all videos share identical temporal window structures. Each bucket processes videos with consistent window counts, enabling direct CSV aggregation while preserving full temporal granularity and narrative pacing patterns critical for creative analysis.

**Middle Segment Aggregation**: For buckets 9-13s and 13-18s, middle segments are aggregated into a single "middle_aggregate" window instead of kept separate (middle_1, middle_2, middle_3). This ensures all 21 base features are reliably measured, as individual middle windows in these buckets are too short (1-4s) for features like scene_count, speech_coverage, and scene_duration_variance to produce stable values. Buckets 18-33s and longer preserve separate middle segments as their windows (3-22.8s each) are long enough for reliable feature extraction.
```

**Why this change**:
- Adds high-level explanation of aggregation to the problem statement
- Sets context early in the document
- Helps readers understand why output schemas differ for short vs long buckets

---

### 2.2 Validation After Changes

**Tests to Add/Update**:

1. **Unit Test: Aggregation Logic**
   ```python
   def test_middle_aggregation_9_13s():
       """Test that bucket 9-13s aggregates middle segments correctly."""
       # Setup
       video_data = {
           'temporal_windows': {
               'hook': {'scene_count': 1, 'word_count': 5},
               'middle_segments': [
                   {'scene_count': 2, 'word_count': 10},
                   {'scene_count': 3, 'word_count': 15},
                   {'scene_count': 1, 'word_count': 12}
               ],
               'closing': {'scene_count': 1, 'word_count': 4}
           },
           'metadata': {'duration': 11.0, 'create_time': '2025-01-01'}
       }

       # Extract features
       features = extract_features(video_data, bucket='9-13s')

       # Assertions
       assert 'middle_aggregate_scene_count' in features
       assert features['middle_aggregate_scene_count'] == 2.0  # (2+3+1)/3
       assert features['middle_aggregate_word_count'] == 12.33  # (10+15+12)/3

       # Should NOT have separate middle segments
       assert 'middle_1_scene_count' not in features
       assert 'middle_2_scene_count' not in features
       assert 'middle_3_scene_count' not in features
   ```

2. **Integration Test: Column Count Validation**
   ```python
   def test_column_count_9_13s():
       """Test that 9-13s bucket produces exactly 66 columns."""
       # Process bucket with aggregation
       df = process_bucket('bucket_9-13s')

       # Assertion
       assert len(df.columns) == 66, f"Expected 66 columns, got {len(df.columns)}"

       # Verify column names
       assert 'middle_aggregate_scene_count' in df.columns
       assert 'middle_1_scene_count' not in df.columns
   ```

3. **Integration Test: Stage 4 Compatibility**
   ```python
   def test_stage4_can_transform_aggregated_csv():
       """Test that Stage 4 can transform aggregated middle features."""
       # Run Stage 3 with aggregation
       stage3_output = run_stage3('bucket_9-13s')

       # Run Stage 4 transformation
       rf_transformed, km_transformed = run_stage4(stage3_output)

       # Assertions
       assert rf_transformed is not None
       assert km_transformed is not None
       assert 'middle_aggregate_scene_count' in rf_transformed.columns
   ```

---

## 3. Downstream Impact Analysis

### 3.1 Stage 4: Feature Transformation

**Impact**: Minor (column names change, logic unchanged)

**Changes Required**:
- None (transformation logic applies to any column name)
- Automatic: `middle_aggregate_scene_count` gets log+scaled just like `middle_1_scene_count` would

**Verification**:
- Stage 4 tests should pass without modification
- May need to update test fixtures if they hardcode column names

---

### 3.2 Stage 5: ML Model Training

**Impact**: Minor (model input dimensions change)

**Changes Required**:
- None (models automatically adapt to input feature count)
- Bucket 9-13s model: 108 features → 66 features
- Bucket 13-18s model: 108 features → 66 features

**Benefits**:
- Better model quality (fewer noisy features)
- Faster training (66 features instead of 108)
- Lower memory usage

---

### 3.3 Stage 6: ML Analysis Generation

**Impact**: Minor (JSON field names change)

**Changes Required**:
- Feature importance rankings will reference `middle_aggregate_*` instead of `middle_1_*`, etc.
- No code changes needed (reads from DataFrame columns automatically)

**Example Output Change**:
```json
// Before
{
  "feature": "middle_1_scene_count",
  "importance": 0.15
}

// After
{
  "feature": "middle_aggregate_scene_count",
  "importance": 0.15
}
```

---

### 3.4 Stage 7: LLM Report Generation

**Impact**: Minor (prompt language adjustment)

**Changes Required**:
- Update LLM prompts to say "Middle segment (aggregated)" instead of "Middle segments 1-3"
- LLM will receive features like `middle_aggregate_eye_contact_rate`

**Example Prompt Change**:
```
// Before
### Middle Segment 1 (3-6s)
- scene_count: 2.5
- word_count: 18

### Middle Segment 2 (6-9s)
- scene_count: 3.2
- word_count: 22

### Middle Segment 3 (9-12s)
- scene_count: 1.8
- word_count: 15

// After
### Middle Segment (aggregated 3-12s)
- scene_count: 2.5 (average across 3 segments)
- word_count: 18.3 (average across 3 segments)
```

---

## 4. Implementation Plan

### 4.1 Step-by-Step Implementation

**Step 1: Update FeatureAggregationCHILD.md** ✓ (This document)
- Document all changes
- Update all affected sections
- Add decision log entry

**Step 2: Implement Code Changes in stage3_aggregation.py**
1. Add `AGGREGATE_MIDDLE_BUCKETS` constant
2. Update `EXPECTED_FEATURE_COUNTS` dictionary
3. Modify `extract_features()` function with aggregation logic
4. Add debug logging for aggregation events

**Step 3: Update Unit Tests**
1. Add `test_middle_aggregation_9_13s()`
2. Add `test_middle_aggregation_13_18s()`
3. Update `test_column_count_validation()` with new expected counts
4. Update test fixtures if they hardcode column names

**Step 4: Run Integration Tests**
1. Process test bucket 9-13s through full pipeline (Stage 3 → 4 → 5 → 6 → 7)
2. Verify no errors
3. Verify ML models train successfully
4. Verify LLM reports generate correctly

**Step 5: Update Downstream Stage Documentation** (if needed)
1. Stage 4: Note that middle_aggregate_* columns exist for buckets 9-13s, 13-18s
2. Stage 7: Update prompt templates to handle aggregated middle segments

**Step 6: Production Deployment**
1. Run on real 9-13s and 13-18s videos
2. Compare cluster quality before/after change
3. Validate feature distributions in aggregated middle windows

---

### 4.2 Rollback Plan

If aggregation causes issues:

1. **Revert code changes**:
   - Remove `AGGREGATE_MIDDLE_BUCKETS` constant
   - Restore original `EXPECTED_FEATURE_COUNTS` values (108 for 9-13s, 13-18s)
   - Restore original `extract_features()` logic (no aggregation)

2. **Re-run Stage 3** for affected buckets

3. **Investigate root cause**:
   - Were averaged features causing ML issues?
   - Were column names breaking downstream stages?
   - Was there a bug in aggregation logic?

**Risk**: Low (change is well-isolated to Stage 3, minimal downstream impact)

---

## 5. Testing Checklist

- [ ] Unit test: Aggregation logic correctly averages features
- [ ] Unit test: Aggregation only applies to buckets 9-13s and 13-18s
- [ ] Unit test: Column count validation passes (66 for aggregated buckets)
- [ ] Unit test: Null values handled correctly in aggregation
- [ ] Integration test: Stage 3 → Stage 4 pipeline works with aggregated columns
- [ ] Integration test: Stage 3 → Stage 5 → ML training succeeds
- [ ] Integration test: Stage 3 → Stage 7 → LLM reports generate successfully
- [ ] Regression test: Buckets 18-33s+ unchanged (still separate middle segments)
- [ ] Performance test: Aggregation doesn't slow down Stage 3 significantly
- [ ] Data quality test: Aggregated features have reasonable distributions (no anomalies)

---

## 6. Expected Benefits

### 6.1 Feature Reliability Improvement

| Bucket | Before Aggregation | After Aggregation |
|--------|-------------------|-------------------|
| **9-13s** | 13/21 features reliable (62%) | 21/21 features reliable (100%) |
| **13-18s** | 18/21 features reliable (86%) | 21/21 features reliable (100%) |

**Unreliable features fixed**:
- `scene_count` (needs 3s for multiple scenes)
- `shortest_scene`, `longest_scene` (needs 3s for detection)
- `scene_duration_variance` (needs 4s for variance calculation)
- `speech_coverage`, `word_count` (needs 2s for meaningful counts)
- `gesture_count` (needs 1.5s for gestures)
- `gaze_variance`, `energy_variance` (need 2s for variance)

---

### 6.2 ML Model Quality Improvement

**K-Means Clustering**:
- Before: 108 features (13 reliable + 8 noisy from each of 3 middle segments)
- After: 66 features (all 21 reliable)
- **Expected result**: Better cluster separation, more interpretable centroids

**Random Forest Classification**:
- Before: 108 features with noisy scene/speech features
- After: 66 features, all reliable
- **Expected result**: Cleaner feature importance rankings, better predictions

---

### 6.3 Reduced Feature Count

| Bucket | Before | After | Reduction |
|--------|--------|-------|-----------|
| 9-13s | 108 | 66 | -39% |
| 13-18s | 108 | 66 | -39% |

**Benefits**:
- Faster ML training (less features to process)
- Lower memory usage
- Simpler models (less overfitting risk)
- Easier interpretation (fewer features to explain)

---

## 7. References

### 7.1 Related Documents

- **Question 2 Analysis**: Analysis of temporal window reliability that identified this issue
- **FeatureAggregationCHILD.md**: Parent document being modified
- **KmeansClusteringStage6.md**: Benefits from improved feature quality
- **FeatureTransformation.md**: Lists all 21 base features and their reliability requirements

### 7.2 Decision Context

This change was proposed based on analysis showing:
- Middle windows in bucket 9-13s range from 1-2.3s (too short)
- Middle windows in bucket 13-18s range from 2.3-4s (marginal)
- Scene features need 3+ seconds for reliable detection
- Speech features need 2+ seconds for meaningful word counts
- Variance features need 1.5-2+ seconds for stable calculations

**Conclusion**: Aggregating 3 short middle segments (total 4.5-9.3s) produces more reliable measurements than keeping them separate.

---

## Document Metadata

**Creation Date**: 2025-01-10
**Author**: Technical Architect
**Status**: PROPOSED CHANGE - Awaiting Approval
**Related Question**: Question 2 (K-Means temporal window reliability)
**Parent Document**: FeatureAggregationCHILD.md
**Estimated Implementation Time**: 4-6 hours (2h code + 2h tests + 2h validation)

---

## Approval Status

- [ ] Technical Review: _______________ (Date: ______)
- [ ] Business Review: _______________ (Date: ______)
- [ ] Implementation Approved: YES / NO
- [ ] Deployment Date: ______________________
