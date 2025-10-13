# New Features Business Scenario

> **Purpose**: Document the operational workflow and maintenance burden when adding new RumiAI features to the ML pipeline
> **Related Docs**: MLPlanningv2.md Stage 3 (Aggregation), Stage 4 (Transformation)
> **Status**: DRAFT - Brainstorm for maintenance cost analysis

---

## Scenario: Adding a New RumiAI Feature to ML Pipeline

### Business Context

As RumiAI evolves, new ML services may be added or existing services may output additional features. For example:
- MediaPipe adds "gesture_count" (hand gesture detection)
- YOLO adds "product_visibility_ratio" (product placement detection)
- Whisper adds "speaking_pace_wpm" (words per minute)

Each new feature must be integrated into the ML training pipeline to be used for pattern detection and creative strategy recommendations.

---

## Current Workflow: Adding New Feature

### Step 1: Modify Stage 3 - Feature Aggregation

**File**: `processors/feature_aggregation.py` (hypothetical TI implementation)

**Task**: Add new feature to BASE_FEATURES list and extraction logic

**Example**:
```python
# Before
BASE_FEATURES = [
    'scene_count', 'eye_contact_rate', 'word_count', 'speech_coverage',
    'energy_level', 'joy_ratio', 'surprise_ratio', 'anger_ratio',
    'close_ratio', 'medium_ratio', 'wide_ratio', 'element_count',
    # ... ~18 more features
]

# After (adding gesture_count from MediaPipe)
BASE_FEATURES = [
    'scene_count', 'eye_contact_rate', 'word_count', 'speech_coverage',
    'energy_level', 'joy_ratio', 'surprise_ratio', 'anger_ratio',
    'close_ratio', 'medium_ratio', 'wide_ratio', 'element_count',
    'gesture_count',  # ← NEW FEATURE
    # ... ~18 more features
]

# Extraction logic (automatically applied to all windows)
for feature in BASE_FEATURES:
    video_features[f'hook_{feature}'] = windows['hook'][feature]
    # ... repeat for middle segments and closing
```

**Effort**: Low (5-10 minutes) - Simple list addition if RumiAI already outputs the feature

---

### Step 2: Analyze Feature Distribution (Critical Decision Point)

**Task**: Examine feature distribution across sample videos to determine transformation needs

**Questions to Answer**:
1. **Is this a count feature?** (e.g., scene_count, word_count, gesture_count)
   - Check: Does it have unbounded integers (0, 1, 5, 15, 50, 200)?
   - Check: Is distribution right-skewed (most values low, few values very high)?
   - **If YES**: Needs log transformation for K-Means

2. **Is this a rate/ratio feature?** (e.g., eye_contact_rate, speech_coverage, joy_ratio)
   - Check: Is it already bounded [0, 1] or [0, 100]?
   - Check: Is distribution relatively uniform or normal?
   - **If YES**: Needs direct MinMax scaling for K-Means

3. **Is this a categorical feature?** (e.g., gender, scene_type)
   - Check: Are values discrete categories (male/female, indoor/outdoor)?
   - **If YES**: Needs one-hot encoding for both RF and K-Means

4. **Is this a temporal feature?** (e.g., create_time, publish_hour)
   - Check: Does it represent cyclical time data?
   - **If YES**: Needs cyclical encoding (sin/cos) for K-Means

**Analysis Method**:
```python
# Generate distribution report
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ml_analysis/aggregated_features.csv")

# Check distribution for new feature across all windows
for window in ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']:
    feature_col = f'{window}_gesture_count'
    if feature_col in df.columns:
        print(f"\n{feature_col} statistics:")
        print(df[feature_col].describe())

        # Visualize distribution
        df[feature_col].hist(bins=50)
        plt.title(f'{feature_col} Distribution')
        plt.savefig(f'analysis/{feature_col}_distribution.png')
```

**Effort**: Medium (30-60 minutes) - Requires statistical analysis and domain knowledge

---

### Step 3: Update Random Forest Transformation (Stage 4.1)

**File**: `processors/rf_transformation.py` (hypothetical TI implementation)

**Task**: Add transformation logic for Random Forest

**Scenario A: Count Feature (gesture_count)**
```python
# Random Forest handles raw counts well (scale-invariant)
# NO transformation needed - feature auto-included from BASE_FEATURES

# If categorical encoding needed:
if 'gesture_type' in df_rf.columns:
    df_rf = pd.get_dummies(df_rf, columns=['gesture_type'], prefix='gesture')
```

**Scenario B: Categorical Feature (scene_type)**
```python
# One-hot encoding required
if 'scene_type' in df_rf.columns:
    df_rf = pd.get_dummies(df_rf, columns=['scene_type'], prefix='scene')
```

**Scenario C: Temporal Feature (publish_hour)**
```python
# Cyclical features for Random Forest (optional but helpful)
df_rf['publish_hour_sin'] = np.sin(2 * np.pi * df_rf['publish_hour'] / 24)
df_rf['publish_hour_cos'] = np.cos(2 * np.pi * df_rf['publish_hour'] / 24)
```

**Effort**: Low-Medium (15-30 minutes) - Depends on feature type

---

### Step 4: Update K-Means Transformation (Stage 4.2)

**File**: `processors/kmeans_transformation.py` (hypothetical TI implementation)

**Task**: Add transformation logic for K-Means (MUST scale for distance calculations)

**Scenario A: Count Feature (gesture_count)**
```python
# Add to count_features list (applies across ALL windows)
count_features = [
    'hook_scene_count', 'hook_word_count', 'hook_element_count', 'hook_gesture_count',  # ← ADD
    'middle_1_scene_count', 'middle_1_word_count', 'middle_1_element_count', 'middle_1_gesture_count',  # ← ADD
    'middle_2_scene_count', 'middle_2_word_count', 'middle_2_element_count', 'middle_2_gesture_count',  # ← ADD
    # ... repeat for all windows (7 windows = 7 additions)
]

# Log transformation logic already handles all features in count_features list
for feature in count_features:
    if feature in df_km.columns:
        df_km[f'{feature}_log'] = np.log1p(df_km[feature])
        df_km[f'{feature}_scaled'] = (
            (df_km[f'{feature}_log'] - df_km[f'{feature}_log'].min()) /
            (df_km[f'{feature}_log'].max() - df_km[f'{feature}_log'].min())
        )
        df_km.drop(columns=[feature], inplace=True)
```

**Scenario B: Rate/Ratio Feature (gesture_rate)**
```python
# Add to rate_features list (applies across ALL windows)
rate_features = [
    'hook_eye_contact_rate', 'hook_speech_coverage', 'hook_joy_ratio', 'hook_gesture_rate',  # ← ADD
    'middle_1_eye_contact_rate', 'middle_1_speech_coverage', 'middle_1_joy_ratio', 'middle_1_gesture_rate',  # ← ADD
    # ... repeat for all windows (7 windows = 7 additions)
]

# MinMax scaling logic already handles all features in rate_features list
for feature in rate_features:
    if feature in df_km.columns:
        df_km[f'{feature}_scaled'] = (
            (df_km[feature] - df_km[feature].min()) /
            (df_km[feature].max() - df_km[feature].min())
        )
        df_km.drop(columns=[feature], inplace=True)
```

**Scenario C: Categorical Feature (gesture_type)**
```python
# One-hot encoding (same as RF)
if 'gesture_type' in df_km.columns:
    df_km = pd.get_dummies(df_km, columns=['gesture_type'], prefix='gesture')
```

**Effort**: Medium (30-45 minutes) - Must update feature lists across all windows, risk of missing windows

---

### Step 5: Test Both Pipelines

**File**: `tests/test_feature_transformation.py` (hypothetical)

**Task**: Validate new feature transforms correctly in both pipelines

**Test Cases**:
1. **Feature Presence Test**: Verify new feature appears in output CSVs
   ```python
   def test_gesture_count_in_rf_output():
       df = pd.read_csv("ml_analysis/rf_transformed.csv")
       # For count features, RF keeps original (no transformation)
       assert 'hook_gesture_count' in df.columns
       assert 'middle_1_gesture_count' in df.columns

   def test_gesture_count_in_kmeans_output():
       df = pd.read_csv("ml_analysis/km_transformed.csv")
       # K-Means transforms to log+scaled
       assert 'hook_gesture_count_scaled' in df.columns
       assert 'middle_1_gesture_count_scaled' in df.columns
       # Original should be dropped
       assert 'hook_gesture_count' not in df.columns
   ```

2. **Transformation Correctness Test**: Verify scaling/encoding applied correctly
   ```python
   def test_gesture_count_kmeans_scaling():
       df = pd.read_csv("ml_analysis/km_transformed.csv")
       # Scaled features should be in [0, 1] range
       assert df['hook_gesture_count_scaled'].min() >= 0
       assert df['hook_gesture_count_scaled'].max() <= 1
   ```

3. **No Regression Test**: Verify existing features still work
   ```python
   def test_existing_features_unchanged():
       df_rf = pd.read_csv("ml_analysis/rf_transformed.csv")
       df_km = pd.read_csv("ml_analysis/km_transformed.csv")

       # Verify old features still present
       assert 'hook_scene_count' in df_rf.columns
       assert 'hook_scene_count_scaled' in df_km.columns
   ```

**Effort**: Medium-High (45-90 minutes) - Must test both pipelines, validate correctness, check for regressions

---

### Step 6: Update Documentation

**Files to Update**:
1. **FeatureTransformation.md** (Stage 4 HLD)
   - Add new feature to BASE_FEATURES list
   - Document transformation decision (why log vs scale vs one-hot)
   - Update feature count tables

2. **TotalFeatures.md** (Feature catalog)
   - Add feature definition
   - Document source service (e.g., MediaPipe)
   - Explain business meaning

3. **MLPlanningv2.md** (if feature counts change)
   - Update feature count tables in Stage 3.2 and Stage 4

**Effort**: Low-Medium (20-40 minutes) - Documentation updates

---

## Total Maintenance Burden Per New Feature

| Step | Effort | Complexity | Risk |
|------|--------|------------|------|
| 1. Stage 3 Aggregation | 5-10 min | Low | Low (auto-applied to all windows) |
| 2. Analyze Distribution | 30-60 min | Medium | Medium (wrong decision = poor clustering) |
| 3. RF Transformation | 15-30 min | Low-Medium | Low (RF is forgiving) |
| 4. K-Means Transformation | 30-45 min | Medium | **HIGH** (easy to miss windows, breaks clustering) |
| 5. Test Both Pipelines | 45-90 min | Medium-High | **HIGH** (must catch regressions) |
| 6. Update Documentation | 20-40 min | Low | Low (but skipping hurts future devs) |
| **TOTAL** | **2.5-4.5 hours** | **Medium-High** | **HIGH** |

---

## Key Maintenance Concerns

### 1. Dual Pipeline Synchronization Risk

**Issue**: RF and K-Means transformations must stay synchronized. Easy to:
- Update RF logic but forget K-Means
- Add feature to count_features list but miss one window
- Change scaling approach in one pipeline but not the other

**Impact**: Silent bugs where one model uses different features than the other, making cross-model insights inconsistent.

### 2. Window Explosion Complexity

**Issue**: For K-Means, each new base feature requires N additions to feature lists (N = number of windows = 2-7 depending on bucket).

**Example**: Adding `gesture_count` requires:
- Bucket 0-3s, 3-9s: 2 additions (hook, closing)
- Bucket 9-13s, 13-18s: 5 additions (hook, 3 middle, closing)
- Bucket 18-33s: 6 additions (hook, 4 middle, closing)
- Bucket 33-60s+: 7 additions (hook, 5 middle, closing)

**Impact**: Easy to miss windows, leading to inconsistent feature sets across buckets.

### 3. Statistical Analysis Burden

**Issue**: Must analyze feature distribution before deciding transformation approach. Wrong decision = poor model performance.

**Example**: Treating a ratio feature as a count (applying log transform) will distort already-normalized values.

**Impact**: Requires ML expertise for each new feature, can't be delegated to junior developers.

### 4. Testing Multiplication

**Issue**: Each new feature doubles testing requirements (RF + K-Means pipelines).

**Impact**: Test suite grows linearly with feature count, slowing CI/CD.

---

## Proposed Simplifications (For Discussion)

### Option A: Unified Transformation Pipeline

**Approach**: Use K-Means transformations for both models
- Apply log+scale for counts, direct scale for rates (as K-Means requires)
- Feed transformed features to BOTH Random Forest and K-Means
- Random Forest handles scaled features fine (slight performance drop acceptable)

**Pros**:
- Single transformation pipeline (50% less code)
- Single set of tests
- Single feature list update per new feature
- Easier to maintain

**Cons**:
- Random Forest may perform 5-10% worse with scaled features (hypothesis - needs testing)
- Loses RF's native handling of raw count distributions

### Option B: Auto-Detection Transformation

**Approach**: Write logic to auto-detect feature type and apply transformations
```python
def detect_feature_type(df, feature):
    """Auto-detect if feature is count, rate, or categorical"""
    if df[feature].dtype == 'object':
        return 'categorical'
    elif df[feature].max() <= 1.0 and df[feature].min() >= 0:
        return 'rate'
    elif df[feature].dtype == 'int64' and df[feature].skew() > 1.0:
        return 'count'
    else:
        return 'unknown'
```

**Pros**:
- No manual decision for each feature
- Scales better as feature count grows
- Reduces human error

**Cons**:
- Auto-detection may misclassify edge cases
- Less explicit (harder to debug why feature was transformed certain way)
- Requires validation that auto-detection works correctly

### Option C: Feature Metadata Catalog

**Approach**: Maintain a feature metadata file that drives transformations
```yaml
# features.yaml
gesture_count:
  type: count
  source: mediapipe
  description: "Number of hand gestures detected"
  transformation:
    rf: raw
    kmeans: log_scale
  applies_to: [hook, middle_1, middle_2, middle_3, middle_4, closing]

gesture_rate:
  type: rate
  source: mediapipe
  description: "Percentage of frames with gestures"
  transformation:
    rf: raw
    kmeans: minmax_scale
  applies_to: [hook, middle_1, middle_2, middle_3, middle_4, closing]
```

**Pros**:
- Single source of truth for all features
- Easy to review feature transformations at a glance
- Transformation logic reads from metadata (DRY principle)
- Forces documentation of new features

**Cons**:
- Adds indirection (must look at YAML to understand transformations)
- YAML must stay synchronized with code
- Requires YAML parsing in transformation code

---

## Recommendations

1. **Immediate**: Document current maintenance workflow (this document)
2. **Phase 1**: Implement comprehensive feature transformation tests to catch regressions
3. **Phase 2**: Evaluate Option A (Unified Transformation) with A/B test:
   - Train RF on both raw and scaled features
   - Compare feature importance rankings and classification accuracy
   - If <10% performance drop, adopt unified approach
4. **Phase 3**: If maintaining dual pipelines, consider Option C (Feature Metadata Catalog) for long-term maintainability

---

## Related Documents

- MLPlanningv2.md Stage 4 (Feature Transformation design)
- FeatureTransformation.md (Child HLD with transformation specifications)
- Critique_FeatureTransformation.md (Business critique identifying maintenance burden)

---

**Status**: DRAFT - Pending Q&A validation in Phase 1 Business Critique
**Next Step**: Use this document to answer Q3 about maintenance cost trade-offs
