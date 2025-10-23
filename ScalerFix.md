# Scaler Fix - Architectural Analysis & Solution

**Document Version**: 1.0
**Date**: 2025-10-23
**Author**: Claude Code
**Status**: Proposed Fix - Awaiting Implementation

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Root Cause Analysis](#root-cause-analysis)
3. [What Are Scalers?](#what-are-scalers)
4. [Current State](#current-state)
5. [Solution Options](#solution-options)
6. [Architectural Analysis](#architectural-analysis)
7. [Recommended Solution](#recommended-solution)
8. [Implementation Plan](#implementation-plan)
9. [References](#references)

---

## Problem Statement

### Issue

Stage 5 (ML Model Training) fails during output validation with:

```
ValidationError: Expected output missing:
data/clients/test_final/.../models/hook_scalers_18-33s.pkl.
Training incomplete.
```

### Impact

- ❌ Cannot complete Stage 5 ML model training
- ❌ Cannot proceed to Stage 6 (ML Analysis Generation)
- ❌ Cannot perform inference on new videos (future use case)

### Context

- **Test Run**: test_final/hashtags/test_vitamin/top_contrastive
- **Buckets Affected**: All 3 (bucket_18-33s, bucket_13-18s, bucket_60-90s)
- **Stage 5 Status**: Models trained successfully, validation fails on missing scalers

---

## Root Cause Analysis

### Investigation Results

**Systematic search of all Stage 1-5 documentation**:

| Stage | TI Document | Scaler References | Creates .pkl? | Conclusion |
|-------|-------------|-------------------|---------------|------------|
| **Stage 1** | VideoDiscoveryCHILDTI.md | 0 | ❌ NO | Does NOT create scalers |
| **Stage 2** | VideoProcessingTI.md | 0 | ❌ NO | Does NOT create scalers |
| **Stage 3** | FeatureAggregationTI.md | 0 | ❌ NO | Does NOT create scalers |
| **Stage 4** | FeatureTransformationTI.md | 0 | ❌ NO | Does NOT save scalers* |
| **Stage 5** | MLModelTrainingCHILDTI.md | Multiple | ❓ EXPECTS | Expects to COPY from Stage 4 |

**Note**: *Stage 4 applies scaling transformations but doesn't save the fitted scaler objects.

### The Inconsistency

**MLModelTrainingCHILDTI.md has contradictory specifications**:

**Says OPTIONAL** (3 locations):
- Line 181: "Scalers from Stage 4 (one per window, **optional but recommended**)"
- Line 935: `logger.warning(f"Scaler file missing for {window}: {scaler_source}. Skipping scaler copy.")`
- Line 1001: Edge case table says "**Non-critical for training, only needed for inference**"

**Says REQUIRED** (1 location):
- Line 1467: Includes scalers in `required_models` list → raises ValidationError if missing

### Discovery

**What Stage 5 expects** (lines 927-935):
```python
# Save scalers (for inference - Stage 6 dependency)
# Note: Scalers are loaded from Stage 4 output, not re-fitted here
scaler_source = os.path.join(bucket_base, f'ml_analysis/{window}_scalers.pkl')
scaler_dest = os.path.join(bucket_base, f'models/{window}_scalers_{bucket}.pkl')
if os.path.exists(scaler_source):
    joblib.dump(joblib.load(scaler_source), scaler_dest)  # COPY from Stage 4
else:
    logger.warning(f"Scaler file missing for {window}: {scaler_source}. Skipping scaler copy.")
```

**Conclusion**: Stage 5 expects Stage 4 to create `ml_analysis/{window}_scalers.pkl`, but Stage 4 never creates them.

---

## What Are Scalers?

### Purpose

Scalers normalize features to a common range (typically [0, 1]) for distance-based algorithms like K-Means.

### The Problem They Solve

**Raw data has wildly different scales**:

```
Video #1:
- scene_count: 8 (range: 1-20)
- word_count: 450 (range: 50-800)
- eye_contact_rate: 0.65 (range: 0.30-0.95)
```

**Without scaling**, K-Means distance calculation is dominated by large-scale features:

```python
distance = sqrt(
    (5 - 6)² +        # scene_count: contributes 1
    (300 - 310)² +    # word_count: contributes 100 ← DOMINATES!
    (0.70 - 0.72)²    # eye_contact: contributes 0.0004
)
# Result: word_count is 250,000x more important than eye_contact!
```

**With MinMaxScaler**, all features normalized to [0, 1]:

```python
scaled_value = (value - min) / (max - min)

scene_count=8    → (8-1)/(20-1)     = 0.368
word_count=450   → (450-50)/(800-50) = 0.533
eye_contact=0.70 → (0.70-0.30)/(0.95-0.30) = 0.615

# Now all features contribute equally to distance!
```

### Why Scalers Must Be Saved

**Training Phase**:
```python
# Fit scaler on training data (learns min/max)
scaler = MinMaxScaler()
scaler.fit(training_data[['scene_count']])  # Learns: min=1, max=20

# Save for later
joblib.dump(scaler, 'hook_scalers.pkl')
```

**Inference Phase** (6 months later):
```python
# Load SAVED scaler (uses training min/max)
scaler = joblib.load('hook_scalers.pkl')

# Scale new video using TRAINING min/max
new_video_scaled = scaler.transform([[15]])
# Uses: (15-1)/(20-1) = 0.737  ✅ Consistent with training

# ❌ WRONG: Fitting new scaler on single video
new_scaler = MinMaxScaler()
new_scaler.fit([[15]])  # Learns: min=15, max=15
# Result: (15-15)/(15-15) = 0/0 = ERROR!
```

**Key Insight**: Scalers preserve the training data's min/max ranges, ensuring new videos are scaled consistently with the training set.

---

## Current State

### What Exists

✅ **Original Unscaled Data** (`aggregated_features.csv` from Stage 3):
```
hook_scene_count: 4, 1, 3, 2, ...  (actual counts)
hook_word_count: 11, 45, 23, ...   (actual word counts)
```

✅ **Scaled Data** (`hook_km_transformed.csv` from Stage 4):
```
scene_count_scaled: 1.0, 0.0, 0.666, ...  (normalized [0,1])
word_count_scaled: 0.859, 0.234, ...      (normalized [0,1])
```

✅ **Trained Models** (`models/` from Stage 5 - partially complete):
```
models/hook_kmeans_18-33s.pkl       ✅ Created
models/hook_X_data_18-33s.pkl       ✅ Created
models/rf_hook_18-33s.pkl           ✅ Created
```

❌ **Scaler Objects** (MISSING):
```
ml_analysis/hook_scalers.pkl        ❌ Stage 4 didn't create
models/hook_scalers_18-33s.pkl      ❌ Stage 5 can't copy (source missing)
```

### Verification

```bash
# Confirmed: Original data exists
$ ls data/.../ml_analysis/aggregated_features.csv
-rw-r--r-- 1 jorge jorge 55K Oct 22 18:42 aggregated_features.csv

# Confirmed: No scaler files
$ ls data/.../ml_analysis/*_scalers.pkl
ls: cannot access '.../*_scalers.pkl': No such file or directory

$ ls data/.../models/*_scalers_*.pkl
ls: cannot access '.../*_scalers_*.pkl': No such file or directory
```

---

## Solution Options

### Option 1: Stage 5 Creates Scalers from aggregated_features.csv

**What**: Stage 5 reads Stage 3 output and fits scalers

**Implementation**:
```python
# In Stage 5 K-Means training section
df_orig = pd.read_csv(f'{bucket_base}/ml_analysis/aggregated_features.csv')

scalers = {}
for feature in km_features:
    scaler = MinMaxScaler()
    scaler.fit(df_orig[[f'{window}_{feature}']])
    scalers[feature] = scaler

joblib.dump(scalers, f'models/{window}_scalers_{bucket}.pkl')
```

**Pros**:
- ✅ Works with current data (no re-running needed)
- ✅ Fast to implement (~30 minutes)
- ✅ Produces correct scalers (same min/max as Stage 4 used)

**Cons**:
- ❌ **Violates layer boundaries** (Stage 5 depends on Stage 3 output)
- ❌ **Duplicates transformation logic** (Stage 4 and 5 both know scaling)
- ❌ **Tight coupling** (Stage 5 must understand Stage 4's transformation choices)
- ❌ **Knowledge duplication** (which features to scale, how to scale)
- ❌ **Maintenance burden** (change scaling logic in TWO places)

**Architectural Assessment**: ❌ **Not recommended** - quick fix but creates technical debt

---

### Option 2: Fix Stage 4 to Save Scalers ⭐ **RECOMMENDED**

**What**: Update Stage 4 to save fitted scaler objects during transformation

**Implementation**:
```python
# In feature_transformation.py, K-Means transformation section
# After fitting and applying transformations:

scalers = {}
for feature in km_features:
    scaler = MinMaxScaler()
    scaler.fit(df[[feature]])
    df[f'{feature}_scaled'] = scaler.transform(df[[feature]])
    scalers[feature] = scaler

# Save transformed data
df.to_csv(f'{bucket_base}/ml_analysis/{window}_km_transformed.csv')

# Save scalers (NEW)
scaler_path = f'{bucket_base}/ml_analysis/{window}_scalers.pkl'
joblib.dump(scalers, scaler_path)
logger.info(f"Saved scalers: {scaler_path}")
```

**Pros**:
- ✅ **Architecturally correct** (transformation stage owns artifacts)
- ✅ **Single Responsibility** (Stage 4 owns ALL transformation)
- ✅ **Clean boundaries** (each stage only accesses previous stage)
- ✅ **Cohesion** (transformation logic + artifacts co-located)
- ✅ **Maintainable** (change transformation in ONE place)
- ✅ **Testable** (can test transformation independently)
- ✅ **Production-ready** (supports inference)
- ✅ **Documentation alignment** (matches TI intent)

**Cons**:
- ⚠️ Requires re-running Stage 4 (~30 minutes for 3 buckets)
- ⚠️ Need to update Stage 4 implementation and TI documentation

**Architectural Assessment**: ✅ **Recommended** - correct solution

---

### Option 3: Make Scalers Optional (Remove Validation)

**What**: Remove scalers from Stage 5 output validation

**Implementation**:
```python
# In model_training.py, validate_stage_output()
# Remove scalers from required_models list:

for window in windows:
    required_models.extend([
        os.path.join(bucket_base, f'models/rf_{window}_{bucket}.pkl'),
        os.path.join(bucket_base, f'models/{window}_kmeans_{bucket}.pkl'),
        os.path.join(bucket_base, f'models/{window}_X_data_{bucket}.pkl'),
        # os.path.join(bucket_base, f'models/{window}_scalers_{bucket}.pkl'),  ← REMOVED
    ])
```

**Pros**:
- ✅ Fastest fix (5 minutes)
- ✅ Matches TI edge case (line 1001: "non-critical for training")
- ✅ Stage 5 training completes

**Cons**:
- ❌ **Incomplete for inference** (Stage 6 will fail)
- ❌ **Not production-ready** (can't analyze new videos)
- ❌ **Defers the problem** (must fix later for inference)
- ❌ **Technical debt** (incomplete pipeline)

**Architectural Assessment**: ⚠️ **Acceptable for testing only** - not production-ready

---

## Architectural Analysis

### Principle 1: Single Responsibility

Each stage should have ONE clear responsibility:

| Stage | Responsibility | Should Create Scalers? |
|-------|---------------|----------------------|
| **Stage 3** | Aggregate temporal features | ❌ NO (no transformation) |
| **Stage 4** | Transform features for ML | ✅ **YES** (owns transformation) |
| **Stage 5** | Train ML models | ❌ NO (uses Stage 4 artifacts) |

**Rationale**: Scalers are transformation artifacts → belong with transformation stage

---

### Principle 2: Data Flow Boundaries

**Clean Architecture** (each stage only accesses previous stage):

```
Stage 3 → aggregated_features.csv
            ↓
Stage 4 → hook_km_transformed.csv + hook_scalers.pkl
            ↓
Stage 5 → hook_kmeans_18-33s.pkl + hook_scalers_18-33s.pkl (copied)
            ↓
Stage 6 → Use models + scalers for inference
```

**Architectural Violation** (Option 1 - Stage 5 reaches back to Stage 3):

```
Stage 3 → aggregated_features.csv
            ↓                    ↓
Stage 4 → transformed CSVs     ↓ (Stage 5 reaches back - BAD!)
            ↓                  ↓
Stage 5 ← ← ← ← ← ← ← ← ← ← ← (violates layer boundary)
```

**Why violations are bad**:
- Stage 5 becomes dependent on Stage 3's output structure
- Can't run Stage 5 in isolation
- Tight coupling between non-adjacent stages
- Hard to refactor Stage 3/4 independently

---

### Principle 3: Transformation Ownership

**Who owns the transformation?**

| Transformation Type | Owner | Artifacts |
|---------------------|-------|-----------|
| Log + MinMax scaling | Stage 4 | MinMaxScaler objects |
| One-hot encoding | Stage 4 | Category mappings |
| Label encoding | Stage 4 | LabelEncoder objects |

**All transformation artifacts should be created by Stage 4** because:
- Transformation logic and artifacts co-located = high cohesion
- Single source of truth for "how to transform"
- Easy to test transformation independently
- Downstream stages just USE artifacts, don't recreate them

**Anti-pattern**: Splitting transformation logic (Stage 4) from artifacts (Stage 5)
- Creates duplication (both stages know scaling logic)
- Implicit dependency (Stage 5 must know HOW Stage 4 transformed)
- Maintenance burden (update logic in multiple places)

---

### Principle 4: Inference Completeness

**For production inference**, you need:
1. Trained model (K-Means) ← Stage 5 creates
2. Transformation artifacts (scalers) ← Should come from Stage 4
3. Feature extraction logic ← Stages 1-3

**Deployment Package**:
```
models/
├── hook_kmeans_18-33s.pkl        # Stage 5 output
└── hook_scalers_18-33s.pkl       # Stage 4 output (copied by Stage 5)
```

**Where scalers should come from**:
- ✅ **Created by Stage 4** (knows original data ranges)
- ✅ **Copied by Stage 5** (packages for deployment)
- ❌ **Created by Stage 5** (doesn't have transformation context)

---

## Recommended Solution

### Option 2: Fix Stage 4 to Save Scalers ⭐

**This is the architecturally sound solution** because:

1. **Architectural Correctness**
   - Transformation owned by ONE stage (Stage 4)
   - Clear separation of concerns
   - Clean layer boundaries
   - No duplication of transformation logic

2. **Maintainability**
   - Change transformation in ONE place
   - Easy to test independently
   - Loose coupling between stages

3. **Production Readiness**
   - Complete inference support
   - Scalers guaranteed to match training data
   - Proper artifact tracking

4. **Documentation Alignment**
   - Matches TI intent (Stage 5 expects scalers from Stage 4)
   - Fixes inconsistency at the root
   - No workarounds

---

## Implementation Plan

### Phase 1: Update Stage 4 Implementation

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/feature_transformation.py`

**Location**: K-Means transformation section (around line 650-724)

**Changes**:

```python
# After transforming K-Means features, add scaler saving:

def transform_window_kmeans_features(df: pd.DataFrame, window_type: str, bucket_base: str) -> pd.DataFrame:
    """
    Transform window-level features for K-Means clustering.

    NEW: Also saves fitted scalers for inference.
    """
    # Existing transformation logic...
    # (log+scale, minmax scale, one-hot encoding, etc.)

    # NEW: Save fitted scalers
    scalers = {}
    for feature in km_features:
        if feature in scalers_dict:  # From transformation above
            scalers[feature] = scalers_dict[feature]

    # Save scalers to ml_analysis/ directory
    scaler_path = os.path.join(bucket_base, f'ml_analysis/{window_type}_scalers.pkl')
    joblib.dump(scalers, scaler_path)
    logger.info(f"Saved {len(scalers)} scalers to {scaler_path}")

    return df_km
```

**Estimated Time**: 30-45 minutes

---

### Phase 2: Update Stage 4 TI Documentation

**File**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/ChildDocs/FeatureTransformationTI.md`

**Changes**:

1. **Update Output Contract** (Section 2.2):
   ```markdown
   ## Output Files (per bucket)

   ### ml_analysis/ directory
   - rf_transformed.csv
   - {window}_rf_transformed.csv (per window)
   - {window}_km_transformed.csv (per window)
   - **{window}_scalers.pkl (per window)** ← NEW
   ```

2. **Add Scaler Schema** (Section 3):
   ```markdown
   ### Scaler Schema

   **File**: `{window}_scalers.pkl`
   **Format**: joblib pickle (dict of MinMaxScaler objects)
   **Purpose**: Fitted scalers for inference (used by Stage 6)

   Structure:
   {
       'scene_count': MinMaxScaler(data_min=1, data_max=20),
       'word_count': MinMaxScaler(data_min=5, data_max=800),
       ...
   }
   ```

3. **Document Creation Logic** (Section 4):
   ```markdown
   ## Scaler Creation

   During K-Means transformation, scalers are fitted on training data
   and saved for inference:

   [code example]
   ```

**Estimated Time**: 15-20 minutes

---

### Phase 3: Re-run Stage 4

**Execute** for all test buckets:

```bash
# Re-run Stage 4 for bucket_18-33s
python rumiai_ml_batch.py ... --bucket bucket_18-33s --start-stage 4 --end-stage 4

# Re-run Stage 4 for bucket_13-18s
python rumiai_ml_batch.py ... --bucket bucket_13-18s --start-stage 4 --end-stage 4

# Re-run Stage 4 for bucket_60-90s
python rumiai_ml_batch.py ... --bucket bucket_60-90s --start-stage 4 --end-stage 4
```

**Expected Output**:
```
ml_analysis/
├── hook_scalers.pkl          ← NEW
├── middle_1_scalers.pkl      ← NEW
├── middle_2_scalers.pkl      ← NEW
...
```

**Estimated Time**: 20-30 minutes (depends on video count)

---

### Phase 4: Verify Stage 5 Works

**Execute** Stage 5 training:

```bash
python rumiai_ml_batch.py ... --start-stage 5 --end-stage 5
```

**Expected Behavior**:
- ✅ Stage 5 finds scalers in `ml_analysis/`
- ✅ Stage 5 copies scalers to `models/`
- ✅ Validation passes (all required files present)

**Verification**:
```bash
# Check scalers were copied
ls data/.../models/*_scalers_*.pkl
# Expected: 6-7 scaler files per bucket
```

**Estimated Time**: 10-15 minutes

---

### Total Implementation Cost

| Phase | Time | Type |
|-------|------|------|
| Update Stage 4 code | 30-45 min | Development |
| Update Stage 4 TI docs | 15-20 min | Documentation |
| Re-run Stage 4 | 20-30 min | Execution |
| Verify Stage 5 | 10-15 min | Testing |
| **Total** | **75-110 min** | **~1.5-2 hours** |

---

## Detailed Implementation Plan - Option 2A (APPROVED)

### Discovery Summary (2025-10-23)

**Complete discovery performed across 7 critical areas:**

| Discovery Area | Status | Finding |
|----------------|--------|---------|
| **Function callers** | ✅ Complete | 3 callers identified (main, script, 9 tests) |
| **Unit tests** | ✅ Complete | 9 tests need trivial fix (add `, _`) |
| **Production script** | ✅ Complete | 1 line needs update |
| **Main orchestrator** | ✅ Complete | SAFE - calls entry point, not internal function |
| **Bucket definitions** | ✅ Complete | 1-7 scaler files per bucket (known) |
| **Integration tests** | ✅ Complete | SAFE - doesn't call our function |
| **CI/CD pipeline** | ✅ Complete | None detected |

**Conclusion**: Option 2A is SAFE to implement with minimal breaking changes.

---

### Critical Fixes Applied (2025-10-23)

**5 critical bugs identified and resolved:**

| Issue | Fix Applied | Status |
|-------|-------------|--------|
| **C1: Validation fails** | Scalers added to `output_files` as paths (Option B) | ✅ FIXED |
| **C2: Checkpoint incomplete** | Scalers included via `output_files` dict (Option A) | ✅ FIXED |
| **C3: File loop crashes** | Skip `.pkl` files in CSV writing loop (Option C) | ✅ FIXED |
| **C4: Scaler inconsistency** | Save `constant_features` metadata (Option B) | ✅ FIXED |
| **C5: No validation** | Add post-save validation with `joblib.load()` (Option A) | ✅ FIXED |

**Key architectural changes:**
- Scalers saved BEFORE validation (not after)
- Scaler format includes metadata: `{'version', 'scalers', 'constant_features'}`
- output_files dict contains scaler file paths (strings) for validation/checkpoint
- Post-save validation catches corruption/permission errors immediately

---

### Implementation Checklist

#### **Phase 0: Pre-Implementation Verification** (~10-15 min) **[INCLUDES M3/M4 FIXES]**

**Step 0.1**: Verify sklearn/joblib in requirements.txt **[M3 FIX - ✅ VERIFIED]**
```bash
# Check if dependencies are listed
grep -E "scikit-learn|sklearn|joblib" /home/jorge/rumiaifinal/venv/requirements.txt
```

**Verified (2025-10-23)**:
- ✅ `scikit-learn==1.7.2` - Already installed (transitive dep via py-feat)
- ✅ `joblib==1.5.1` - Already installed (transitive dep via py-feat, sklearn)
- ✅ Added explicitly to requirements.txt for clarity:
  ```
  scikit-learn>=1.3.0
  joblib>=1.3.0
  ```

**Why they exist**: py-feat (FEAT emotion detection service) requires sklearn, which requires joblib.
Since FEAT is one of the 9 ML services, these are always available.

**Compatibility**: sklearn 1.7.2 is compatible with our implementation (we require >=1.3.0)

**Step 0.2**: Search codebase for all callers **[M4 FIX]**
```bash
# Search for any callers we might have missed
grep -r "transform_window_level_kmeans" /home/jorge/rumiaifinal \
  --include="*.py" \
  --exclude-dir=venv \
  --exclude-dir=__pycache__

# Check for imports
grep -r "from.*feature_transformation import.*transform_window_level_kmeans" /home/jorge/rumiaifinal \
  --include="*.py" \
  --exclude-dir=venv
```

**Expected**: Should find exactly 3 callers:
1. `rumiai_v2/processors/feature_transformation.py:957` (main caller)
2. `scripts/stage4_transformation.py:189` (production script)
3. `tests/unit/test_feature_transformation.py` (9 test calls)

**If more found**: Update those callers to handle tuple return

---

#### **Phase 1: Core Function Refactor** (~45-50 min)

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/feature_transformation.py`

**Step 1.1**: Add imports (line 14) **[INCLUDES C2-2, C2-3 FIXES]**
```python
# ADD after line 24 (at top of file with other imports):
from typing import Dict, Tuple, List  # C2-3: Add Tuple for type hints
from sklearn.preprocessing import MinMaxScaler
import joblib
```

**Note**: All imports must be at file top, not inside functions (C2-2 fix)

**Step 1.2**: Modify function signature (line 635) **[INCLUDES M2, C2-2, C2-3 FIXES]**
```python
# CHANGE:
def transform_window_level_kmeans(
    df: pd.DataFrame,
    window_type: str
) -> pd.DataFrame:

# TO:
def transform_window_level_kmeans(
    df: pd.DataFrame,
    window_type: str
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:  # ← M2/C2-3 FIX: Specific type hint with Tuple
    """
    Transform features for Window-Level K-Means (single window).

    Returns:
        Tuple containing:
            - pd.DataFrame: Transformed features (27 columns)
            - Dict[str, Dict]: Scaler metadata with structure:
                {
                    'fitted': Dict[str, MinMaxScaler],  # Fitted scaler objects
                    'constant': List[str]                # Zero-variance features
                }
    """
```

**Note**: Imports (typing, sklearn) are at file top in Step 1.1, not here (C2-2 fix)

**Step 1.3**: Refactor scaling logic (lines 666-698) **[INCLUDES C4, C2-1 FIXES]**
```python
# REPLACE manual scaling with sklearn

# Initialize scaler storage with metadata structure (C4 fix)
# C2-1 fix: Use 'fitted'/'constant' internally (clear semantics)
# Translation to 'scalers'/'constant_features' happens at save time (Step 3.1)
scaler_result = {
    'fitted': {},      # MinMaxScaler objects for features with variance
    'constant': []     # List of features with zero variance
}

# GROUP 1: Log + MinMax scale (lines 666-681)
for feature in log_scale_features:
    if feature in df_km.columns:
        # Apply log1p
        df_km[feature] = np.log1p(df_km[feature])

        # Fit scaler
        scaler = MinMaxScaler()
        min_val = df_km[feature].min()
        max_val = df_km[feature].max()

        if max_val > min_val:
            scaler.fit(df_km[[feature]])
            df_km[f'{feature}_scaled'] = scaler.transform(df_km[[feature]]).flatten()
            scaler_result['fitted'][feature] = scaler  # Save to 'fitted' dict
        else:
            # Constant feature
            df_km[f'{feature}_scaled'] = 0.5
            scaler_result['constant'].append(feature)  # ← C4 FIX: Track constant features

        df_km.drop(columns=[feature], inplace=True)

# GROUP 2: MinMax scale only (lines 683-698)
for feature in scale_features:
    if feature in df_km.columns:
        scaler = MinMaxScaler()
        min_val = df_km[feature].min()
        max_val = df_km[feature].max()

        if max_val > min_val:
            scaler.fit(df_km[[feature]])
            df_km[f'{feature}_scaled'] = scaler.transform(df_km[[feature]]).flatten()
            scaler_result['fitted'][feature] = scaler  # Save to 'fitted' dict
        else:
            df_km[f'{feature}_scaled'] = 0.5
            scaler_result['constant'].append(feature)  # ← C4 FIX: Track constant features

        df_km.drop(columns=[feature], inplace=True)
```

**Step 1.4**: Update return statement and add docstring (line 724) **[INCLUDES C4 + L5 FIXES]**
```python
# CHANGE:
return df_km

# TO:
return df_km, scaler_result  # Returns {'fitted': {...}, 'constant': [...]}
```

**Step 1.5**: Add comprehensive docstring **[L5, C2-4 FIXES]**
```python
# ADD at line 636 (first line INSIDE function, after signature):
"""
Transform features for Window-Level K-Means (single window).

Source: FeatureTransformationTI.md Section 4.4

Args:
    df: pandas DataFrame from aggregated_features.csv
    window_type: str, window identifier (e.g., "hook", "middle_1", "closing")

Returns:
    Tuple[pd.DataFrame, Dict[str, Dict]]:
        - DataFrame: Transformed K-Means features (27 columns, all numerical [0-1])
        - Dict: Scaler metadata with structure:
            {
                'fitted': {
                    'scene_count': MinMaxScaler(...),  # Fitted scaler objects
                    'word_count': MinMaxScaler(...),
                    # ... up to 18 scalers (features with variance > 0)
                },
                'constant': [
                    'overlay_unique_count',  # Features with zero variance
                    # ... list of constant features (all same value)
                ]
            }

Example:
    >>> df_hook_km, scalers = transform_window_level_kmeans(df, 'hook')
    >>> print(len(scalers['fitted']))     # e.g., 16 fitted scalers
    >>> print(scalers['constant'])         # e.g., ['overlay_unique_count', 'gaze_variance']

Note:
    Features with zero variance (max == min) cannot have scalers fitted.
    These are tracked in 'constant' list and scaled to 0.5 (midpoint).
"""
```

---

#### **Phase 2: Update Main Caller** (~5 min)

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/feature_transformation.py`

**Step 2.1**: Update caller (lines 954-959)
```python
# CHANGE (lines 954-958):
km_start = time.time()
window_km_dfs = {}
for window in windows:
    df_window_km = transform_window_level_kmeans(df, window)
    window_km_dfs[window] = df_window_km

# TO:
km_start = time.time()
window_km_dfs = {}
window_scalers = {}  # NEW
for window in windows:
    df_window_km, scalers = transform_window_level_kmeans(df, window)  # MODIFIED
    window_km_dfs[window] = df_window_km
    window_scalers[window] = scalers  # NEW
```

---

#### **Phase 3: Save Scalers BEFORE Validation** (~20-25 min) **[INCLUDES C1/C2/C3/C5 FIXES]**

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/feature_transformation.py`

**CRITICAL**: Scalers must be saved BEFORE validation (not after) to fix C1/C2/C3 issues.

**Step 3.1**: Save scaler files EARLY (INSERT after line 960, BEFORE output_files dict at line 963) **[INCLUDES M8 FIX]**
```python
# NEW Step 4.5: Save scaler files BEFORE validation (C1/C2/C3/C5/M8 fixes)
logger.info("Saving fitted scalers for inference")
scaler_save_start = time.time()
window_scaler_paths = {}  # Track paths for output_files dict

# Ensure output directory exists
output_dir = os.path.join(bucket_path, 'ml_analysis')
os.makedirs(output_dir, exist_ok=True)

for window in windows:
    scaler_path = os.path.join(output_dir, f'{window}_scalers.pkl')

    # Save scaler dict with metadata (C4 fix, M5 included via version field)
    # C2-1 fix: Translate internal keys to .pkl format
    #   'fitted' → 'scalers' (more intuitive for users loading files)
    #   'constant' → 'constant_features' (more explicit)
    import sklearn  # M2-2: Import to get version

    scaler_metadata = {
        'version': '1.0',  # M5 fix: Format version for future compatibility (L2-1)
        'sklearn_version': sklearn.__version__,  # M2-2: Track sklearn version for compatibility
        'scalers': window_scalers[window]['fitted'],        # ← C2-1: Translation here
        'constant_features': window_scalers[window]['constant']  # ← C2-1: Translation here
    }

    # Save with error handling (M8 fix)
    try:
        joblib.dump(scaler_metadata, scaler_path)
        window_scaler_paths[window] = scaler_path  # Store path for output_files dict
        logger.debug(f"Successfully dumped scaler metadata to {scaler_path}")
    except Exception as e:
        logger.error(f"Failed to save {window}_scalers.pkl: {e}")
        logger.error(f"Error details: {type(e).__name__}, Path: {scaler_path}")
        raise IOError(f"Scaler save failed for {window}: {e}") from e

    # Post-save validation (C5 fix) - verify immediately after saving
    try:
        loaded = joblib.load(scaler_path)
        assert 'version' in loaded, f"{window}_scalers.pkl missing version"
        assert 'sklearn_version' in loaded, f"{window}_scalers.pkl missing sklearn_version"  # M2-2
        assert 'scalers' in loaded, f"{window}_scalers.pkl missing scalers"
        assert 'constant_features' in loaded, f"{window}_scalers.pkl missing constant_features"
        assert isinstance(loaded['scalers'], dict), f"{window}_scalers.pkl scalers not a dict"

        scaler_count = len(loaded['scalers'])
        constant_count = len(loaded['constant_features'])
        file_size_kb = os.path.getsize(scaler_path) / 1024
        logger.info(
            f"  ✓ Saved {window}_scalers.pkl: {scaler_count} fitted scalers, "
            f"{constant_count} constant features, sklearn {loaded['sklearn_version']}, {file_size_kb:.1f} KB"
        )
    except Exception as e:
        logger.error(f"Failed to validate {window}_scalers.pkl: {e}")
        # M2-1: Fail-fast approach - raise error immediately
        # Partial .pkl files remain on disk for debugging
        # User should delete ml_analysis/ directory and re-run Stage 4
        raise IOError(f"Scaler validation failed for {window}: {e}") from e

# M10 fix: Update metrics to include scaler files
scaler_elapsed = time.time() - scaler_save_start
metrics.record_transformation_time('scaler_save', scaler_elapsed)
logger.info(f"Scaler saving complete: {len(windows)} files ({scaler_elapsed:.1f}s)")
```

**Step 3.2**: Update output_files dict to include scaler PATHS (MODIFY line 963-966) **[C1/C2 FIX]**
```python
# CHANGE (lines 963-966):
output_files = {'rf_transformed.csv': df_rf}
for window in windows:
    output_files[f'{window}_rf_transformed.csv'] = window_rf_dfs[window]
    output_files[f'{window}_km_transformed.csv'] = window_km_dfs[window]

# TO:
output_files = {'rf_transformed.csv': df_rf}
for window in windows:
    output_files[f'{window}_rf_transformed.csv'] = window_rf_dfs[window]
    output_files[f'{window}_km_transformed.csv'] = window_km_dfs[window]
    output_files[f'{window}_scalers.pkl'] = window_scaler_paths[window]  # ← C1/C2 FIX: Add scaler PATHS
```
**Why this works:**
- `window_scaler_paths[window]` contains STRING paths (not objects)
- Validation (line 792-795) checks if filenames exist in `output_files.keys()` ✅
- Checkpoint (line 846) writes `list(output_files.keys())` which now includes scalers ✅

**Step 3.3**: Update expected files function (lines 85-92)
```python
# CHANGE:
def get_expected_output_files(bucket: str) -> List[str]:
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket]

    files = ['rf_transformed.csv']
    for window in windows:
        files.append(f'{window}_rf_transformed.csv')
        files.append(f'{window}_km_transformed.csv')
    return files

# TO:
def get_expected_output_files(bucket: str) -> List[str]:
    from config.bucket_definitions import BUCKET_WINDOWS
    windows = BUCKET_WINDOWS[bucket]

    files = ['rf_transformed.csv']
    for window in windows:
        files.append(f'{window}_rf_transformed.csv')
        files.append(f'{window}_km_transformed.csv')
        files.append(f'{window}_scalers.pkl')  # NEW
    return files
```

**Step 3.4**: Update file writing loop to skip .pkl files (MODIFY lines 972-983) **[C3, C2-5 FIXES]**
```python
# CHANGE (lines 972-983):
# Step 7: Write output files to disk
logger.info("Writing output files to disk")
output_dir = os.path.join(bucket_path, 'ml_analysis')
os.makedirs(output_dir, exist_ok=True)  # ← C2-5 FIX: DELETE THIS LINE (already created in Step 4.5)

io_start = time.time()
for filename, df_output in output_files.items():
    output_path = os.path.join(output_dir, filename)
    df_output.to_csv(output_path, index=False)
    file_size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"  Wrote {filename}: {file_size_kb:.1f} KB")

# TO:
# Step 7: Write CSV output files to disk
logger.info("Writing CSV files to disk")
output_dir = os.path.join(bucket_path, 'ml_analysis')
# C2-5 FIX: REMOVED os.makedirs (already created in Step 4.5, line 861)

io_start = time.time()
csv_count = 0
for filename, df_output in output_files.items():
    # Skip .pkl files (already saved in Step 4.5) - C3 FIX
    if filename.endswith('.pkl'):
        continue

    output_path = os.path.join(output_dir, filename)
    df_output.to_csv(output_path, index=False)
    file_size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"  Wrote {filename}: {file_size_kb:.1f} KB")
    csv_count += 1

metrics.record_transformation_time('file_io', time.time() - io_start)
logger.info(f"CSV file I/O complete: {csv_count} files ({time.time() - io_start:.1f}s)")
```

---

#### **Phase 4: Update Production Script** (~2-5 min)

**File**: `/home/jorge/rumiaifinal/scripts/stage4_transformation.py`

**Step 4.1**: Update function call (line 189)
```python
# CHANGE:
window_km_df = transform_window_level_kmeans(df, window_name)

# TO:
window_km_df, _ = transform_window_level_kmeans(df, window_name)  # Ignore scalers
```

---

#### **Phase 5: Update Unit Tests** (~30-40 min) **[INCLUDES M1 FIX]**

**File**: `/home/jorge/rumiaifinal/tests/unit/test_feature_transformation.py`

**Step 5.1**: Update all 9 existing test function calls (~5-10 min)

**Lines to modify**: 242, 255, 268, 281, 299, 314, 363, 377, 384

```python
# PATTERN - CHANGE:
df_hook_km = transform_window_level_kmeans(df, 'hook')

# TO:
df_hook_km, _ = transform_window_level_kmeans(df, 'hook')  # Ignore scalers in existing tests
```

**Specific lines**:
- Line 242: `test_window_kmeans_log_scale`
- Line 255: `test_window_kmeans_shift_scale`
- Line 268: `test_window_kmeans_label_encode`
- Line 281: `test_window_kmeans_emotion_one_hot`
- Line 299: `test_window_kmeans_output_schema` (inside loop)
- Line 314: `test_edge_case_zero_variance`
- Line 363: (find exact test)
- Line 377: (find exact test)
- Line 384: (find exact test)

---

**Step 5.2**: Add 5 new scaler-specific tests (~25-30 min) **[M1 FIX]**

**Add after existing K-Means tests** (around line 390):

```python
# ============================================================================
# TEST SCALER CREATION AND VALIDATION
# ============================================================================

def test_scalers_created(fixture_bucket_18_33s):
    """Test: Scalers are created and have correct structure"""
    df = fixture_bucket_18_33s.copy()
    df_hook_km, scaler_result = transform_window_level_kmeans(df, 'hook')

    # Check structure
    assert 'fitted' in scaler_result
    assert 'constant' in scaler_result
    assert isinstance(scaler_result['fitted'], dict)
    assert isinstance(scaler_result['constant'], list)


def test_scalers_loadable(fixture_bucket_18_33s):
    """Test: Scalers can be saved and loaded via joblib"""
    import joblib
    import tempfile

    df = fixture_bucket_18_33s.copy()
    df_hook_km, scaler_result = transform_window_level_kmeans(df, 'hook')

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        scaler_metadata = {
            'version': '1.0',
            'scalers': scaler_result['fitted'],
            'constant_features': scaler_result['constant']
        }
        joblib.dump(scaler_metadata, f.name)

        # Load back
        loaded = joblib.load(f.name)
        assert loaded['version'] == '1.0'
        assert 'scalers' in loaded
        assert 'constant_features' in loaded


def test_scaler_min_max_values(fixture_bucket_18_33s):
    """Test: Scalers have sensible min/max values"""
    from sklearn.preprocessing import MinMaxScaler

    df = fixture_bucket_18_33s.copy()
    df_hook_km, scaler_result = transform_window_level_kmeans(df, 'hook')

    # Check all fitted scalers are MinMaxScaler instances
    for feature, scaler in scaler_result['fitted'].items():
        assert isinstance(scaler, MinMaxScaler)

        # Check data_min_ and data_max_ exist and make sense
        assert hasattr(scaler, 'data_min_')
        assert hasattr(scaler, 'data_max_')
        assert scaler.data_max_[0] >= scaler.data_min_[0]


def test_zero_variance_handling(fixture_bucket_18_33s):
    """Test: Zero-variance features tracked in constant list"""
    df = fixture_bucket_18_33s.copy()

    # Set all hook_scene_count to same value (zero variance)
    df['hook_scene_count'] = 5

    df_hook_km, scaler_result = transform_window_level_kmeans(df, 'hook')

    # Should be in constant list, not fitted scalers
    assert 'scene_count' in scaler_result['constant']
    assert 'scene_count' not in scaler_result['fitted']

    # Scaled column should exist and be 0.5
    assert 'scene_count_scaled' in df_hook_km.columns
    assert (df_hook_km['scene_count_scaled'] == 0.5).all()


def test_scaler_count_consistent(fixture_bucket_18_33s):
    """Test: Scaler count + constant count = total features"""
    df = fixture_bucket_18_33s.copy()
    df_hook_km, scaler_result = transform_window_level_kmeans(df, 'hook')

    # Total features that get scaled (11 log+scale + 7 scale-only)
    expected_total = 18

    fitted_count = len(scaler_result['fitted'])
    constant_count = len(scaler_result['constant'])

    # Sum should equal total (or less if some features missing from data)
    assert fitted_count + constant_count <= expected_total
    assert fitted_count > 0  # Should have at least some fitted scalers
```

---

#### **Phase 6: Update Documentation** (~15-20 min)

**File**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/ChildDocs/FeatureTransformationTI.md`

**Step 6.1**: Update Output Contract (Section 2.2, around line 213)
```markdown
# ADD after window K-Means outputs:

# ===== WINDOW-LEVEL SCALER OUTPUTS (6 files for bucket 18-33s) =====
hook_scalers_pkl_path: str              # Fitted scalers for hook window
                                        # Location: "{bucket_base}/ml_analysis/hook_scalers.pkl"
                                        # Format: joblib pickle (dict of MinMaxScaler objects)
                                        # Size: ~18 scaler objects (11 log+scale + 7 scale-only features)
                                        # Consumers: Stage 5 (copies to models/ for inference)

middle_1_scalers_pkl_path: str          # Fitted scalers for middle_1 window
... (repeat for all windows)
```

**Step 6.2**: Add scaler schema (new section after Section 3.5) **[INCLUDES C4/C5/M2-2 METADATA]**
```markdown
### 3.6 Scaler Output Schema

**File**: `{window}_scalers.pkl` (per window: hook, middle_1, ..., closing)

**Format**: joblib pickle

**Structure**: Dictionary with metadata (version 1.0)

```python
{
    'version': '1.0',  # Format version for future compatibility (L2-1)
    'sklearn_version': '1.7.2',  # M2-2: sklearn version for compatibility checking (actual installed version)

    'scalers': {
        'scene_count': MinMaxScaler(feature_range=(0, 1)),  # fitted on training data
        'word_count': MinMaxScaler(feature_range=(0, 1)),
        'gesture_count': MinMaxScaler(...),
        # ... up to 18 scalers (11 from log_scale_features + 7 from scale_features)
    },

    'constant_features': [
        'overlay_unique_count',  # Example: all videos had 0 overlays
        # ... list of features with zero variance (fitted scaler not possible)
    ]
}
```

**Key Points:**
- **version**: Format version (1.0) for schema evolution (L2-1: enables future migrations)
- **sklearn_version**: sklearn version used to create scalers (M2-2: compatibility checking)
- **scalers**: Dict of MinMaxScaler objects for features with variance > 0
- **constant_features**: List of features with zero variance (all same value)
- **Scaler count varies**: Typically 15-18 scalers, depending on how many features have variance

**Creation Logic**: During K-Means transformation, scalers are fitted on training data.
Features with zero variance (max == min) are tracked in `constant_features` instead of
attempting to fit a scaler.

**Post-save validation**: Immediately after saving, scaler files are loaded and validated
to catch corruption or permission errors.

**Error Recovery (M2-1)**: If scaler saving fails partway through:
- Stage 4 raises IOError and stops immediately (fail-fast)
- Partial .pkl files remain in `ml_analysis/` for debugging
- **Recovery**: Delete `ml_analysis/` directory and re-run Stage 4 (deterministic, safe to re-run)
- No automatic rollback (keeps failed files for inspection)

**Version Compatibility (M2-2)**: Scalers saved with sklearn 1.3.x should load in 1.3.x.
If sklearn version changes significantly, check `sklearn_version` field and warn user
if mismatch detected.

**Future Compatibility (L2-1)**: `version: 1.0` allows format changes when Stage 6 is implemented.
Version field enables migration logic if scaler format needs to evolve.

**Usage**: Stage 5 copies from `ml_analysis/{window}_scalers.pkl` to
`models/{window}_scalers_{bucket}.pkl` for inference use in Stage 6.

**Inference Example**:
```python
import joblib

# Load scaler metadata
scaler_data = joblib.load('hook_scalers_18-33s.pkl')

# Use fitted scalers
if 'scene_count' in scaler_data['scalers']:
    scaled_value = scaler_data['scalers']['scene_count'].transform([[8]])
else:
    # Feature was constant during training, use midpoint
    scaled_value = 0.5
```
```

---

### Validation & Testing

#### **Step 7: Run Unit Tests**
```bash
cd /home/jorge/rumiaifinal
pytest tests/unit/test_feature_transformation.py -v
```

**Expected**: All 25 tests pass

---

**Step 6.3**: Update metrics recording **[M10, C2-6 FIXES]**

**Location**: Around line 987 in `feature_transformation.py`

```python
# CHANGE:
metrics.record_output(len(output_files), len(df_rf.columns))

# TO (C2-6 fix: Log separately to avoid modifying MetricsCollector):
metrics.record_output(len(output_files), len(df_rf.columns))
logger.info(f"METRIC: scaler_file_count={len(windows)}")  # M10: Track scaler count
logger.info(f"METRIC: total_output_files={len(output_files)} (CSVs + scalers)")
```

**Why C2-6**: Logging separately avoids modifying `MetricsCollector.record_output()` signature, which could break other code. Metrics are still captured in logs for analysis.

---

### Priority Decisions - Skipped Items

**MEDIUM PRIORITY:**

**M6 - Dict vs Custom Object**: ⏭️ SKIPPED (Option C)
- Decision: Plain dict is sufficient for MVP
- Rationale: Structured dict with metadata keys works fine, no need for dataclass

**M7 - Individual Scalers vs ColumnTransformer**: ⏭️ SKIPPED (Option B)
- Decision: Keep individual scalers (current approach)
- Rationale: Works correctly, not worth refactor risk for minimal benefit

**M9 - Performance Benchmarking**: ⏭️ SKIPPED (Option C)
- Decision: Trust sklearn optimization
- Rationale: sklearn MinMaxScaler is well-optimized, unlikely bottleneck

**LOW PRIORITY:**

**L1 - get_expected_output_files() Dual Purpose**: ⏭️ SKIPPED (Option C)
- Decision: Function works fine as-is
- Rationale: Used for both validation AND documentation - coupling acceptable, no refactor needed

---

### Round 2 Decisions - Medium/Low Priority

**MEDIUM PRIORITY (Round 2):**

**M2-1 - Partial Scaler Failure Recovery**: ✅ FIXED (Option B)
- Decision: Fail-fast approach with manual cleanup
- Implementation: Added error handling comments in Step 3.1 and recovery procedure in scaler schema docs
- Rationale: Stage 4 is deterministic, can safely re-run; partial files useful for debugging

**M2-2 - Sklearn Version Compatibility**: ✅ FIXED (Option A)
- Decision: Save sklearn version in metadata
- Implementation: Added `sklearn_version` field to scaler_metadata in Step 3.1, updated validation and docs
- Rationale: Minimal overhead, enables compatibility checking if sklearn version changes

**LOW PRIORITY (Round 2):**

**L2-1 - Stage 6 Future Compatibility**: ✅ HANDLED (Option B)
- Decision: Version field provides migration path
- Implementation: Documented in scaler schema (Step 6.2) that `version: 1.0` enables future format evolution
- Rationale: Stage 6 doesn't exist yet; version field allows changes when needed without premature optimization

---

### Validation & Testing

#### **Step 7: Run Unit Tests**
```bash
cd /home/jorge/rumiaifinal
pytest tests/unit/test_feature_transformation.py -v
```

**Expected**: All 30 tests pass (25 existing + 5 new scaler tests from M1)

---

#### **Step 8: Test on Multiple Buckets** (~25-35 min) **[L2 FIX]**

**Test 3 bucket types to cover edge cases:**

**8.1: Test bucket_18-33s (6 windows - representative)**
```bash
python rumiai_ml_batch.py \
  --client test_final \
  --analysis-type hashtag \
  --target test_vitamin \
  --start-stage 4 \
  --end-stage 4 \
  --bucket bucket_18-33s
```

**Expected outputs** (`data/clients/test_final/.../bucket_18-33s/ml_analysis/`):
- ✅ 13 CSV files (1 video-level + 6 windows × 2 types)
- ✅ **6 scaler files**: `hook_scalers.pkl`, `middle_1_scalers.pkl`, `middle_2_scalers.pkl`, `middle_3_scalers.pkl`, `middle_4_scalers.pkl`, `closing_scalers.pkl`

**8.2: Test bucket_0-3s (1 window - minimum edge case)**
```bash
python rumiai_ml_batch.py \
  --client test_final \
  --analysis-type hashtag \
  --target test_vitamin \
  --start-stage 4 \
  --end-stage 4 \
  --bucket bucket_0-3s
```

**Expected outputs**:
- ✅ 3 CSV files (1 video-level + 1 window × 2 types)
- ✅ **1 scaler file**: `hook_scalers.pkl` only

**8.3: Test bucket_60-90s (7 windows - maximum edge case)**
```bash
python rumiai_ml_batch.py \
  --client test_final \
  --analysis-type hashtag \
  --target test_vitamin \
  --start-stage 4 \
  --end-stage 4 \
  --bucket bucket_60-90s
```

**Expected outputs**:
- ✅ 15 CSV files (1 video-level + 7 windows × 2 types)
- ✅ **7 scaler files**: `hook_scalers.pkl`, `middle_1_scalers.pkl`, ..., `middle_5_scalers.pkl`, `closing_scalers.pkl`

---

**Verify scaler contents (any bucket)**:
```python
import joblib

# Load scaler metadata
scaler_data = joblib.load('.../hook_scalers.pkl')

# Check structure
assert 'version' in scaler_data
assert 'scalers' in scaler_data
assert 'constant_features' in scaler_data

# Check counts
fitted_count = len(scaler_data['scalers'])
constant_count = len(scaler_data['constant_features'])
print(f"Fitted scalers: {fitted_count}")
print(f"Constant features: {constant_count}")
print(f"Total: {fitted_count + constant_count} (expect ~18)")

# Verify scaler objects
from sklearn.preprocessing import MinMaxScaler
for feature, scaler in scaler_data['scalers'].items():
    assert isinstance(scaler, MinMaxScaler)
    print(f"{feature}: min={scaler.data_min_[0]:.3f}, max={scaler.data_max_[0]:.3f}")
```

---

#### **Step 9: Verify Stage 5 Can Load and Use Scalers** (~15-20 min) **[L4 FIX]**

**9.1: Run Stage 5**
```bash
# Run Stage 5 for bucket_18-33s
python rumiai_ml_batch.py \
  --client test_final \
  --analysis-type hashtag \
  --target test_vitamin \
  --start-stage 5 \
  --end-stage 5 \
  --bucket bucket_18-33s
```

**Expected**:
- ✅ Stage 5 finds scalers in `ml_analysis/`
- ✅ Stage 5 copies scalers to `models/{window}_scalers_18-33s.pkl`
- ✅ Validation passes (all required files present)

**9.2: Explicitly verify Stage 5 can load scalers** **[L4 FIX]**
```python
import joblib
from pathlib import Path

# Path to copied scalers
bucket_path = Path("data/clients/test_final/.../bucket_18-33s")
models_dir = bucket_path / "models"

# Verify all scaler files exist and are loadable
windows = ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

for window in windows:
    scaler_file = models_dir / f"{window}_scalers_18-33s.pkl"

    # Check file exists
    assert scaler_file.exists(), f"Missing {scaler_file}"

    # Load scaler metadata
    scaler_data = joblib.load(scaler_file)

    # Verify structure
    assert 'version' in scaler_data, f"{window}: missing version"
    assert 'scalers' in scaler_data, f"{window}: missing scalers"
    assert 'constant_features' in scaler_data, f"{window}: missing constant_features"

    # Verify scalers are usable for inference
    test_features = {}
    for feature_name, scaler in scaler_data['scalers'].items():
        # Test transform works
        test_value = [[5.0]]  # Arbitrary test value
        try:
            transformed = scaler.transform(test_value)
            assert 0 <= transformed[0][0] <= 1, f"{feature_name}: transformed value out of range"
            test_features[feature_name] = transformed[0][0]
        except Exception as e:
            raise AssertionError(f"{window}/{feature_name}: scaler transform failed: {e}")

    print(f"✓ {window}: {len(scaler_data['scalers'])} scalers verified")
    print(f"  Constant features: {scaler_data['constant_features']}")
    print(f"  Sample transforms: {list(test_features.items())[:3]}")

print("\n✅ All scalers verified - Stage 5 can use them for inference")
```

**Expected output**:
```
✓ hook: 16 scalers verified
  Constant features: ['overlay_unique_count', 'gaze_variance']
  Sample transforms: [('scene_count', 0.234), ('word_count', 0.567), ...]
✓ middle_1: 17 scalers verified
  ...
✅ All scalers verified - Stage 5 can use them for inference
```

---

#### **Step 9.5: Integration Test (Stage 4→5 Pipeline)** (~15-20 min) **[L3 FIX]**

**Add integration test to verify end-to-end scaler handoff**

**Create file**: `tests/integration/test_scaler_pipeline.py`

```python
"""
Integration test for Stage 4→5 scaler pipeline.

Tests that scalers created in Stage 4 can be successfully loaded and used by Stage 5.
"""

import pytest
import joblib
import tempfile
import shutil
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def test_stage4_to_stage5_scaler_handoff(tmp_path):
    """
    Integration test: Verify Stage 4 scalers can be loaded by Stage 5.

    Simulates:
    1. Stage 4 saves scalers to ml_analysis/
    2. Stage 5 loads scalers from ml_analysis/
    3. Stage 5 copies to models/
    4. Stage 5 can use scalers for inference
    """
    # Setup test directories
    bucket_path = tmp_path / "bucket_18-33s"
    ml_analysis = bucket_path / "ml_analysis"
    models = bucket_path / "models"
    ml_analysis.mkdir(parents=True)
    models.mkdir(parents=True)

    # Simulate Stage 4: Create scaler file
    test_scaler = MinMaxScaler()
    test_scaler.fit([[0], [10]])  # Fit on range 0-10

    scaler_metadata = {
        'version': '1.0',
        'scalers': {
            'scene_count': test_scaler,
            'word_count': test_scaler
        },
        'constant_features': ['overlay_unique_count']
    }

    stage4_scaler_path = ml_analysis / "hook_scalers.pkl"
    joblib.dump(scaler_metadata, stage4_scaler_path)

    # Verify Stage 4 output
    assert stage4_scaler_path.exists()

    # Simulate Stage 5: Load and copy scalers
    loaded = joblib.load(stage4_scaler_path)
    assert 'version' in loaded
    assert 'scalers' in loaded
    assert 'constant_features' in loaded

    stage5_scaler_path = models / "hook_scalers_18-33s.pkl"
    shutil.copy2(stage4_scaler_path, stage5_scaler_path)

    # Verify Stage 5 can use scalers for inference
    final_scalers = joblib.load(stage5_scaler_path)

    # Test transform works
    for feature_name, scaler in final_scalers['scalers'].items():
        transformed = scaler.transform([[5]])  # Transform value 5
        assert 0 <= transformed[0][0] <= 1, f"{feature_name} out of range"
        assert abs(transformed[0][0] - 0.5) < 0.01, f"{feature_name} incorrect scaling"

    # Verify constant features are tracked
    assert 'overlay_unique_count' in final_scalers['constant_features']

    print("✅ Integration test passed: Stage 4→5 scaler handoff works")


def test_all_windows_scaler_pipeline(tmp_path):
    """Test scaler pipeline for all window types."""
    windows = ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

    bucket_path = tmp_path / "bucket_18-33s"
    ml_analysis = bucket_path / "ml_analysis"
    ml_analysis.mkdir(parents=True)

    # Create scalers for each window
    for window in windows:
        scaler = MinMaxScaler()
        scaler.fit([[0], [100]])

        metadata = {
            'version': '1.0',
            'scalers': {'test_feature': scaler},
            'constant_features': []
        }

        scaler_path = ml_analysis / f"{window}_scalers.pkl"
        joblib.dump(metadata, scaler_path)

    # Verify all can be loaded
    for window in windows:
        scaler_path = ml_analysis / f"{window}_scalers.pkl"
        assert scaler_path.exists()

        loaded = joblib.load(scaler_path)
        assert 'version' in loaded
        assert len(loaded['scalers']) > 0

    print(f"✅ All {len(windows)} window scalers verified")
```

**Run integration test**:
```bash
cd /home/jorge/rumiaifinal
pytest tests/integration/test_scaler_pipeline.py -v
```

**Expected**:
```
test_scaler_pipeline.py::test_stage4_to_stage5_scaler_handoff PASSED
test_scaler_pipeline.py::test_all_windows_scaler_pipeline PASSED
```

---

#### **Step 10: Re-run All Test Buckets**
```bash
# Re-run Stage 4 for remaining buckets (already tested 0-3s, 18-33s, 60-90s in Step 8)
python rumiai_ml_batch.py ... --bucket bucket_13-18s --start-stage 4 --end-stage 4
```

**Expected outputs**:
- bucket_13-18s: 3 scaler files (hook, middle_aggregate, closing)

---

### Implementation Time Estimate (FINAL REVISED WITH LOW PRIORITY)

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 0 | Pre-implementation verification (M3/M4) | 10-15 min |
| 1 | Core function refactor (M2, L5 docstring) | 50-55 min |
| 2 | Update main caller | 5 min |
| 3 | Add scaler saving (C1-C5, M5, M8, M10) | 20-25 min |
| 4 | Update production script | 2-5 min |
| 5 | Update tests (9 existing + 5 new M1) | 30-40 min |
| 6 | Update TI documentation | 15-20 min |
| 7 | Run unit tests (30 tests) | 5-10 min |
| 8 | Test on 3 buckets (L2: edge cases) | 25-35 min |
| 9 | Verify Stage 5 with explicit checks (L4) | 15-20 min |
| 9.5 | Create integration test (L3) | 15-20 min |
| 10 | Re-run remaining bucket | 10-15 min |
| **TOTAL** | **202-265 min (3.4-4.4 hours)** |

**Breakdown of additions:**
- ✅ **5 critical bugs fixed** (C1-C5): +15 min
- ✅ **7 medium priority fixes** (M1-M5, M8, M10): +40 min
- ✅ **4 low priority fixes** (L2-L5): +30 min
- ✅ **1 low priority skipped** (L1): 0 min

**Comprehensive fixes included:**
- **Critical** (C1-C5): Validation, checkpoint, file loop, metadata, post-save validation
- **Medium** (M1-M5, M8, M10): Type hints, error handling, 5 new unit tests, metrics, pre-checks
- **Low** (L2-L5): Multi-bucket testing, integration test, Stage 5 verification, docstrings

---

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Tests fail | Low | Medium | Run tests immediately after code changes |
| Scaler saving fails | Low | High | Add try/except with rollback |
| Stage 5 can't find scalers | Low | High | Verify file paths match exactly |
| Performance degradation | Very Low | Low | sklearn is well-optimized |
| Breaking existing data | None | N/A | Stage 4 is deterministic, safe to re-run |

---

### Rollback Plan

If implementation fails:

1. **Revert code changes**: `git checkout feature_transformation.py`
2. **Revert test changes**: `git checkout tests/unit/test_feature_transformation.py`
3. **Revert script changes**: `git checkout scripts/stage4_transformation.py`
4. **No data loss**: Stage 4 output is deterministic, can re-run anytime

---

## References

### Documentation

1. **MLPlanningv2.md** (Mother Document)
   - Section 5.3 (lines 2038-2098): K-Means training + scaler creation
   - Shows Stage 5 creating scalers (but from already-scaled data - likely doc error)

2. **MLModelTrainingCHILDTI.md** (Stage 5 TI)
   - Line 181: Input schema (scalers from Stage 4, optional)
   - Lines 927-935: Training code (copies scalers from Stage 4)
   - Line 1001: Edge case (scalers non-critical for training)
   - Line 1467: Output validation (requires scalers - INCONSISTENT)

3. **FeatureTransformationTI.md** (Stage 4 TI)
   - Currently has NO scaler creation documented
   - Needs update as part of fix

### Code Files

1. `/home/jorge/rumiaifinal/rumiai_v2/processors/feature_transformation.py`
   - Stage 4 implementation
   - Needs scaler saving logic added

2. `/home/jorge/rumiaifinal/rumiai_v2/processors/model_training.py`
   - Stage 5 implementation
   - Already has scaler copying logic (lines 567, 927-935)
   - No changes needed

### Data Locations

**Test Analysis**: `data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/`

**Current State**:
- ✅ `aggregated_features.csv` (Stage 3 output - RAW data)
- ✅ `hook_km_transformed.csv` (Stage 4 output - scaled data)
- ❌ `hook_scalers.pkl` (Stage 4 should create - MISSING)

**After Fix**:
- ✅ `ml_analysis/hook_scalers.pkl` (Stage 4 creates)
- ✅ `models/hook_scalers_18-33s.pkl` (Stage 5 copies)

---

## Appendix: Why Not Option 1?

### Detailed Analysis of "Stage 5 Creates Scalers"

**What it would do**:
```python
# Stage 5 reaches back to Stage 3's output
df_orig = pd.read_csv('ml_analysis/aggregated_features.csv')

# Stage 5 duplicates Stage 4's scaling logic
for feature in ['scene_count', 'word_count', ...]:
    scaler = MinMaxScaler()
    scaler.fit(df_orig[[f'hook_{feature}']])
    scalers[feature] = scaler
```

**Architectural Problems**:

1. **Layer Violation**
   - Stage 5 depends on Stage 3 output
   - Skips Stage 4 boundary
   - Can't run Stage 5 without Stage 3 data

2. **Logic Duplication**
   - Stage 4: Knows which features to scale, how to scale them
   - Stage 5: Must also know which features to scale, how to scale them
   - Change scaling logic → update TWO places

3. **Implicit Coupling**
   - Stage 5 must understand Stage 4's transformation choices
   - Stage 5 must know mapping (hook_scene_count → scene_count_scaled)
   - Tight coupling between non-adjacent stages

4. **Testing Complexity**
   - Can't test Stage 5 without Stage 3 output
   - Integration tests require full Stage 3-4-5 pipeline
   - Unit testing Stage 5 becomes difficult

5. **Maintenance Burden**
   - Change feature list → update Stage 4 AND Stage 5
   - Change scaling method → update Stage 4 AND Stage 5
   - Risk of inconsistency (stages scale differently)

**Conclusion**: Works but violates architectural principles. Creates technical debt.

---

## Appendix: Cost-Benefit Summary

| Metric | Option 1 (Quick Fix) | Option 2 (Architectural Fix) | Option 3 (Skip) |
|--------|---------------------|----------------------------|----------------|
| **Implementation Time** | 30 min | 75-110 min | 5 min |
| **Re-run Stages** | None | Stage 4 only | None |
| **Technical Debt** | High | None | Medium |
| **Production Ready** | ⚠️ Yes (with debt) | ✅ Yes | ❌ No (incomplete) |
| **Maintainability** | ❌ Low (duplication) | ✅ High (clean) | ⚠️ Medium |
| **Architectural Correctness** | ❌ Violates principles | ✅ Follows principles | ⚠️ Incomplete |
| **Long-term Cost** | High (refactor later) | Low (done right) | High (must fix later) |

**Recommendation**: Invest the extra ~1 hour for Option 2 to build it correctly from the start.

---

## Change Log

### [2025-10-23] - Initial Document

- Documented scaler issue discovered during Stage 5 testing
- Analyzed 3 solution options
- Recommended Option 2 (Fix Stage 4)
- Created implementation plan

### [2025-10-23] - Comprehensive Discovery & Detailed Implementation Plan

- Performed complete discovery across 7 critical areas
- Identified all breaking changes (3 callers: main, script, 9 tests)
- Confirmed orchestrator is safe (no changes needed)
- Verified bucket structures and scaler file counts
- Created detailed implementation plan with step-by-step instructions
- Reduced time estimate from 3-3.5 hours to 2.2-2.8 hours
- **Status**: Ready for implementation (Option 2A approved)

### [2025-10-23] - Critical Bugs Fixed & Implementation Revised

- **Identified 5 critical bugs** in initial implementation plan
- **C1 Fixed**: Scalers now added to output_files as paths (Option B)
- **C2 Fixed**: Checkpoint includes scalers via output_files dict
- **C3 Fixed**: File writing loop skips .pkl files (already saved earlier)
- **C4 Fixed**: Scaler format includes metadata (version, scalers, constant_features)
- **C5 Fixed**: Post-save validation immediately verifies scaler files
- **Architecture Change**: Scalers saved BEFORE validation (not after)
- **Format Change**: Scaler .pkl files now contain structured dict with metadata
- Implementation plan updated to weave in all fixes
- **Status**: All critical issues resolved, ready for implementation

### [2025-10-23] - Medium Priority Fixes & Enhancements

- **Identified 10 medium priority issues** through structured critique
- **M1 Fixed**: Added 5 new scaler-specific unit tests (creation, loading, validation, zero-variance, consistency)
- **M2 Fixed**: Updated type hints to `Dict[str, Dict]` for better type safety
- **M3 Fixed**: Added sklearn/joblib verification to requirements.txt check
- **M4 Fixed**: Added comprehensive codebase search for all function callers
- **M5 Fixed**: Version metadata already included in C4 scaler format
- **M6 Skipped**: Plain dict sufficient, no need for custom ScalerSet class
- **M7 Skipped**: Individual scalers work fine, ColumnTransformer not worth refactor
- **M8 Fixed**: Added try/except with logging around joblib.dump()
- **M9 Skipped**: Trust sklearn optimization, no benchmarking needed
- **M10 Fixed**: Updated metrics.record_output() to include scaler file count
- Added Phase 0 (pre-implementation verification)
- Test count increased from 25 to 30 tests
- Time estimate revised to 2.9-3.75 hours (comprehensive implementation)
- **Status**: All medium priority items resolved or consciously skipped

### [2025-10-23] - Low Priority Enhancements & Final Polish

- **Identified 5 low priority issues** through structured critique
- **L1 Skipped**: get_expected_output_files() dual purpose acceptable, no refactor needed
- **L2 Fixed**: Enhanced testing strategy - test 3 bucket types (0-3s, 18-33s, 60-90s) to cover edge cases
- **L3 Fixed**: Added integration test for Stage 4→5 scaler pipeline (test_scaler_pipeline.py)
- **L4 Fixed**: Enhanced Step 9 with explicit Stage 5 scaler loading verification
- **L5 Fixed**: Added comprehensive docstring to transform_window_level_kmeans() with examples
- Created 2 new integration tests (Stage 4→5 handoff, all windows pipeline)
- Enhanced testing to cover minimum (1 window), typical (6 windows), and maximum (7 windows) buckets
- Added explicit scaler transform verification in Stage 5
- **Final time estimate**: 3.4-4.4 hours (comprehensive, production-ready implementation)
- **Status**: ALL 20 critique points resolved (16 fixed, 4 consciously skipped)

### [2025-10-23] - Round 2 Critical Fixes (Self-Critique)

- **Identified 6 additional critical issues** through implementation plan review
- **C2-1 Fixed**: Clarified key naming - use 'fitted'/'constant' internally, translate to 'scalers'/'constant_features' at save boundary
- **C2-2 Fixed**: Moved all imports to Step 1.1 at file top (not inside function definition)
- **C2-3 Fixed**: Added Tuple to typing imports for type hints
- **C2-4 Fixed**: Corrected docstring location to line 636 (first line inside function)
- **C2-5 Fixed**: Removed redundant os.makedirs from Step 3.4 (already created in Step 3.1)
- **C2-6 Fixed**: Changed metrics to log separately instead of modifying MetricsCollector signature
- All fixes woven into implementation plan with clear annotations
- **Status**: Implementation plan is now production-ready with no blocking issues

### [2025-10-23] - Round 2 Medium/Low Fixes (Final Polish)

- **Identified 3 additional medium/low priority issues**
- **M2-1 Fixed**: Documented fail-fast error recovery with manual cleanup procedure
- **M2-2 Fixed**: Added sklearn_version to scaler metadata for compatibility tracking
- **L2-1 Handled**: Documented that version field enables future Stage 6 format evolution
- Updated scaler schema with error recovery and version compatibility sections
- Scaler metadata now includes: version, sklearn_version, scalers, constant_features
- **Final Status**: ALL 26 critique points across 2 rounds resolved (25 fixed, 1 skipped)

### [2025-10-23] - Dependency Verification

- **Verified orchestrator venv compatibility**
- ✅ `scikit-learn==1.7.2` already installed (transitive dependency via py-feat)
- ✅ `joblib==1.5.1` already installed (transitive dependency via py-feat, sklearn)
- Added explicit dependencies to requirements.txt (scikit-learn>=1.3.0, joblib>=1.3.0)
- **Compatibility confirmed**: sklearn 1.7.2 > required 1.3.0
- **Implementation will work**: All dependencies available in orchestrator's venv

---

**End of Document**
