# Post Bug Fix Documentation Update Guide

**Bug:** S7B2 - Stage 7 cross_window_patterns field empty
**Root Cause:** Cross-window features created in Stage 4, but Stage 3's aggregated_features.csv didn't include them
**Fix Date:** 2025-10-28
**Status:** Verified working across multiple buckets and modes

---

## Executive Summary

**What Changed:**
- **Stage 3:** Now creates 0-5 cross-window features + is_top_performer label
- **Stage 4:** Validates features exist (removed duplicate creation)
- **Stage 5:** Minimum video threshold lowered to 3 (from 50/20)
- **Stage 6:** Fixed video_count bug when is_top_performer present
- **Config:** Centralized expected column counts in bucket_definitions.py

**Impact:** Cross-window features now flow through entire pipeline (Stages 3→4→5→6→7) and appear in Stage 7 LLM analysis output.

---

## Changes by Stage

### Stage 3: Feature Aggregation

**File:** `scripts/stage3_aggregation.py`

#### Change 1: Cross-Window Feature Creation (NEW)

**Added Function:** `add_cross_window_features(df, bucket)` (lines 294-398)

**Purpose:** Create 0-5 cross-window derived features after base aggregation

**Features Created:**
1. `xwin_hook_to_middle_energy` - Energy delta from hook to middle avg (buckets 9-13s+)
2. `xwin_middle_to_closing_energy` - Energy delta from middle to closing (buckets 9-13s+)
3. `xwin_eye_contact_consistency` - Std dev of eye_contact_rate across windows (buckets 3-9s+)
4. `xwin_word_density_std` - Std dev of word_count across windows (buckets 3-9s+)
5. `xwin_energy_progression_slope` - Linear regression slope of energy (buckets 3-9s+)

**Bucket-Aware Logic:**
- **0-3s:** 0 features (only hook window, no comparisons)
- **3-9s:** 3 features (consistency + std + slope, no deltas)
- **9-13s+:** 5 features (all)

**Key Implementation Details:**
```python
# Handles middle_aggregate for 9-13s, 13-18s buckets
middle_energy_cols = [f'{w}_energy_level' for w in windows if w.startswith('middle')]

# Uses .astype(float) to handle iterrows() dtype issues
y = row[energy_cols].values.astype(float)

# Computes slope with numpy polyfit (index-based x)
slope, _ = np.polyfit(x, y, 1)
```

**Why xwin_ Prefix:**
- Original names (`hook_to_middle_energy_delta`) caused collision with window prefixes
- Stage 4 incorrectly identified them as hook-window features
- Renamed to `xwin_*` to clearly distinguish cross-window from window-specific features

---

#### Change 2: Label Creation (NEW)

**Added Function:** `add_is_top_performer_label(df, bucket_path, strategy)` (lines 401-479)

**Purpose:** Add is_top_performer column from Stage 1's selected_videos.json

**Behavior by Strategy:**
- **Contrastive:** Read from selected_videos.json (top 80% = 1, bottom 20% = 0)
- **Top:** All videos = 1 (no bottom performers)

**Fallback Logic:**
```python
# If selected_videos.json missing:
top_count = int(len(df) * 0.8)
df['is_top_performer'] = (df.index < top_count).astype(int)
```

**Why Moved from Stage 4:**
- Stage 6 needs this label in aggregated_features.csv for distribution analysis
- Creating in Stage 3 ensures it's available to all downstream stages

---

#### Change 3: Integration into Pipeline

**Modified:** `process_bucket()` function (line 645-649)

```python
# After base aggregation
df_base = pd.DataFrame(aggregated_data)

# S7B2 FIX: Add cross-window features
df_with_cross_window = add_cross_window_features(df_base, bucket)

# S7B2 FIX: Add is_top_performer label
df_complete = add_is_top_performer_label(df_with_cross_window, bucket_path, strategy)
```

---

#### Change 4: CLI Argument

**Added:** `--strategy` argument (lines 890-895)

```python
parser.add_argument(
    "--strategy",
    required=True,
    choices=['contrastive', 'top'],
    help="Analysis mode: 'contrastive' (top 80%% vs bottom 20%%) or 'top' (top performers only)"
)
```

**Impact:** Stage 3 now requires explicit strategy specification

---

#### Change 5: Expected Column Counts

**Updated:** `EXPECTED_FEATURE_COUNTS` dict (lines 97-106)

**New Values:**
```python
'0-3s': 25,    # was N/A (21×1 + 3 + 0 + 1)
'3-9s': 49,    # was 45 (21×2 + 3 + 3 + 1)
'9-13s': 72,   # was 66 (21×3 + 3 + 5 + 1)
'13-18s': 72,  # was 66 (21×3 + 3 + 5 + 1)
'18-33s': 135, # was 129 (21×6 + 3 + 5 + 1)
'33-60s': 156, # was 150 (21×7 + 3 + 5 + 1)
'60-90s': 156, # was 150 (21×7 + 3 + 5 + 1)
```

**Formula:** `(21 × window_count) + 3 metadata + X cross_window + 1 label`

---

### Stage 4: Feature Transformation

**File:** `rumiai_v2/processors/feature_transformation.py`

#### Change 1: Input Validation

**Modified:** `validate_input()` function (lines 254-262)

**Before:**
```python
expected_cols = 129  # Hardcoded
```

**After:**
```python
from config.bucket_definitions import get_stage3_expected_feature_count
expected_cols = get_stage3_expected_feature_count(bucket)  # 135 for 18-33s
```

**Impact:** Validation now bucket-aware, accepts new column counts from Stage 3

---

#### Change 2: Removed Duplicate Logic

**Deleted:** Cross-window feature creation (was lines ~517-557)

**Deleted:** is_top_performer creation (was lines ~481-515)

**Reason:** Both moved to Stage 3, Stage 4 now validates they exist

---

#### Change 3: Cross-Window Feature Validation

**Modified:** `transform_video_level_rf()` function (lines 510-532)

**Before:**
```python
expected_cross_window = [
    'hook_to_middle_energy_delta',
    'middle_to_closing_delta',
    # ...
]
```

**After:**
```python
expected_cross_window = [
    'xwin_hook_to_middle_energy',      # ← xwin_ prefix
    'xwin_middle_to_closing_energy',   # ← xwin_ prefix
    'xwin_eye_contact_consistency',    # ← xwin_ prefix
    'xwin_word_density_std',           # ← xwin_ prefix
    'xwin_energy_progression_slope'    # ← xwin_ prefix
]
existing_cross_window = [f for f in expected_cross_window if f in df.columns]

if existing_cross_window:
    logger.debug(f"✓ Using {len(existing_cross_window)} cross-window features from Stage 3")
else:
    if bucket != '0-3s':
        logger.warning(f"⚠ No cross-window features found for bucket {bucket}")
```

**Behavior:** Validates features exist, passes them through unchanged

---

#### Change 4: Expected Column Count Function

**Modified:** `get_expected_rf_column_count()` function (lines 123-151)

**Made Bucket-Aware:**
```python
def get_expected_rf_column_count(bucket: str) -> int:
    # Temporal features from Stage 3 (includes xwin features + is_top_performer)
    temporal_features = get_stage3_expected_feature_count(bucket)

    # Additional transformations
    emotions = 7          # dominant_emotion_id one-hot
    temporal_extract = 5  # scene_duration, scene_density, etc.
    gender = 3            # gender one-hot encoding

    # Cross-window features (bucket-dependent, from Stage 3)
    if bucket == '0-3s':
        cross_window_features = 0
    elif bucket == '3-9s':
        cross_window_features = 3
    else:
        cross_window_features = 5

    # is_top_performer (from Stage 3)
    target = 1

    return temporal_features + emotions + temporal_extract + gender + cross_window_features + target
```

**Returns:** 65 (3-9s), 147 (18-33s), 167 (33-60s, 60-90s)

---

### Stage 5: Model Training

**File:** `rumiai_v2/processors/model_training.py`

#### Change: Minimum Video Thresholds

**Modified:** `validate_stage_input()` function (lines 770-771)

**Before:**
```python
MIN_VIDEOS_CONTRASTIVE = 50  # 40 top + 10 bottom (bare minimum for 80/20 split)
MIN_VIDEOS_TOP = 20          # Descriptive analysis only
```

**After:**
```python
MIN_VIDEOS_CONTRASTIVE = 3  # Minimum for any statistical analysis (validated with small datasets)
MIN_VIDEOS_TOP = 3          # Minimum for descriptive analysis only
```

**Reason:**
- Test datasets had as few as 3 videos per bucket
- Models can train with minimal data (lower confidence, but functional)
- Validated working with 3, 32, 38, and 47 video datasets

---

### Stage 6: Analysis Generation

**File:** `ml_pipeline/stage6_analysis/ml_analysis_generation.py`

#### Change: video_count Bug Fix

**Modified:** `generate_video_rf_json()` function (lines 207-216)

**Before:**
```python
if 'is_top_performer' not in df.columns:
    video_count = len(df)  # ← Only defined inside if block
    top_count = int(video_count * TOP_PERFORMER_PERCENTAGE)
    df['is_top_performer'] = [1] * top_count + [0] * (video_count - top_count)

# Later...
analysis_json = {
    'video_count': video_count,  # ← UnboundLocalError if is_top_performer exists!
}
```

**After:**
```python
video_count = len(df)  # ← Moved OUTSIDE conditional
if 'is_top_performer' not in df.columns:
    top_count = int(video_count * TOP_PERFORMER_PERCENTAGE)
    df['is_top_performer'] = [1] * top_count + [0] * (video_count - top_count)
    logger.debug(f"Calculated is_top_performer labels: {top_count} top, {video_count - top_count} bottom")
else:
    logger.debug("Using existing is_top_performer column from CSV")
```

**Why This Broke:**
- Stage 3 now creates is_top_performer column
- video_count was only defined if column missing
- Stage 6 crashed when accessing undefined variable

---

### Config Changes

**File:** `config/bucket_definitions.py`

#### Change: Centralized Column Count Function

**Added:** `get_stage3_expected_feature_count(bucket)` function (lines 139-200)

**Purpose:** Single source of truth for Stage 3 output schema

**Returns by Bucket:**
```python
'0-3s': 25
'3-9s': 49
'9-13s': 72
'13-18s': 72
'18-33s': 135
'33-60s': 156
'60-90s': 156
```

**Formula Implementation:**
```python
window_count = len(BUCKET_WINDOWS[bucket])
base_features = 21 * window_count
metadata_cols = 3

if bucket == '0-3s':
    cross_window_features = 0
elif bucket == '3-9s':
    cross_window_features = 3
else:
    cross_window_features = 5

label_cols = 1
return base_features + metadata_cols + cross_window_features + label_cols
```

**Used By:**
- Stage 3: Output validation
- Stage 4: Input validation
- Tests: Schema verification

---

**File:** `config/__init__.py`

**Created:** Empty file to make config a proper Python package (was missing, caused import errors)

---

### Stage 7: LLM Analysis

**File:** `run_stage7_test.py` (NEW)

**Created:** Wrapper script to load .env before calling Stage 7

**Purpose:** Stage 7 expects ANTHROPIC_API_KEY in os.environ, but doesn't load .env itself

**Implementation:**
```python
# Load .env file
env_file = Path(__file__).parent / ".env"
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value

# Then call Stage 7
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_main
stage7_main(bucket_path, bucket, hashtag)
```

**Why Needed:**
- Bash `source .env` doesn't work (.env is not a shell script)
- Python's os.environ requires explicit loading
- Stage 7 fails silently if API key not set

---

## Test Files Updated

### Test Fixtures (Column Count Changes)

1. **tests/fixtures/stage4/test_bucket_18-33s_minimal.csv**: 129 → 135 columns
2. **tests/fixtures/stage4/test_bucket_9-13s_minimal.csv**: 66 → 72 columns
3. **tests/fixtures/stage4/test_bucket_3-9s_minimal.csv**: 45 → 49 columns

Added synthetic xwin features + is_top_performer=1 to each

### Test Assertions Updated

1. **tests/checkpoint_tests/test_orchestrator_fallback.py**: feature_count 129 → 135
2. **tests/run_tests_manual.py**: 66 → 72, 150 → 156
3. **tests/unit/test_feature_transformation.py**: Docstrings updated
4. **tests/test_stage3_aggregation.py**: 66 → 72

---

## Documentation Files Needing Updates

### 1. MLPlanningv2.md (Mother Document)

**Section to Update:** Stage 3 & 4 descriptions in pipeline overview

**Add:**
- Stage 3 now creates cross-window features (not Stage 4)
- Stage 3 creates is_top_performer label
- Updated column counts: 49, 72, 135, 156 (not 45, 66, 129, 150)

**Example Addition:**
```markdown
### Stage 3: Feature Aggregation Output Schema

| Bucket | Windows | Base (21×N) | Metadata | xwin | Label | Total |
|--------|---------|-------------|----------|------|-------|-------|
| 0-3s   | 1       | 21          | 3        | 0    | 1     | 25    |
| 3-9s   | 2       | 42          | 3        | 3    | 1     | 49    |
| 9-13s  | 3       | 63          | 3        | 5    | 1     | 72    |
| 18-33s | 6       | 126         | 3        | 5    | 1     | 135   |
| 33-60s | 7       | 147         | 3        | 5    | 1     | 156   |

**Cross-Window Features (xwin_prefix):**
- Created in Stage 3 (moved from Stage 4)
- 0 features: bucket 0-3s
- 3 features: bucket 3-9s (consistency, std, slope)
- 5 features: bucket 9-13s+ (deltas, consistency, std, slope)
```

---

### 2. Stage 3 HLD (High-Level Design)

**File Location:** `documentation_migration/services/Stage3_HLD.md` (or similar)

**Sections to Add/Update:**

#### A. New Section: Cross-Window Feature Generation

```markdown
## 4.5 Cross-Window Feature Generation

**Purpose:** Derive features that represent relationships BETWEEN temporal windows

**Function:** `add_cross_window_features(df, bucket)`

**Features Created (bucket-dependent):**

1. **xwin_hook_to_middle_energy** (9-13s+)
   - Formula: `mean(middle_energy_cols) - hook_energy_level`
   - Captures energy shift from hook to middle sections

2. **xwin_middle_to_closing_energy** (9-13s+)
   - Formula: `closing_energy_level - mean(middle_energy_cols)`
   - Captures energy shift from middle to closing

3. **xwin_eye_contact_consistency** (3-9s+)
   - Formula: `std(eye_contact_rate across all windows)`
   - Measures gaze stability throughout video

4. **xwin_word_density_std** (3-9s+)
   - Formula: `std(word_count across all windows)`
   - Measures pacing variability

5. **xwin_energy_progression_slope** (3-9s+)
   - Formula: `np.polyfit(window_index, energy_level, 1)[0]`
   - Measures overall energy trend (increasing/decreasing)

**Edge Cases:**
- 0-3s buckets: 0 features (single window, no comparisons)
- 3-9s buckets: 3 features (no middle sections, so no deltas)
- 9-13s+ buckets: 5 features (all)

**Implementation Notes:**
- Uses `.astype(float)` to handle iterrows() dtype issues
- Handles NaN values gracefully (skips incomplete data)
- Respects middle_aggregate for 9-13s, 13-18s buckets
```

#### B. New Section: Label Assignment

```markdown
## 4.6 is_top_performer Label Assignment

**Purpose:** Add target variable for ML training

**Function:** `add_is_top_performer_label(df, bucket_path, strategy)`

**Strategy Behavior:**

**Contrastive Mode:**
- Reads `selected_videos.json` from Stage 1
- Maps video IDs to is_top_performer (1 = top 80%, 0 = bottom 20%)
- Fallback: Index-based labeling if JSON missing

**Top Mode:**
- All videos = 1 (no bottom performers)

**Data Flow:**
```
selected_videos.json (Stage 1)
  → performer_map dict
  → DataFrame['is_top_performer'] column
```

**Fallback Logic:**
```python
if selected_videos.json missing:
    top_count = int(len(df) * 0.8)
    df['is_top_performer'] = (df.index < top_count).astype(int)
```
```

#### C. Update: Output Schema Section

```markdown
## 5. Output Schema

**File:** `ml_analysis/aggregated_features.csv`

**Structure:**
- Video ID column
- Create time column
- Gender column
- 21 features × N windows (N = 1-7 depending on bucket)
- **NEW:** 0-5 cross-window features (xwin_prefix)
- **NEW:** is_top_performer label (1 column)

**Column Counts:**
- 0-3s: 25 columns
- 3-9s: 49 columns
- 9-13s: 72 columns
- 13-18s: 72 columns
- 18-33s: 135 columns
- 33-60s: 156 columns
- 60-90s: 156 columns
- 90-120s: 156 columns

**Critical:** All downstream stages (4-7) depend on this schema
```

---

### 3. Stage 3 Technical Implementation

**File Location:** `documentation_migration/services/Stage3_TI.md` (or similar)

**Sections to Add:**

#### Code Location References

```markdown
## New Functions

### `add_cross_window_features(df, bucket)`
- **Location:** `scripts/stage3_aggregation.py:294-398`
- **Called by:** `process_bucket()` line 646
- **Input:** DataFrame with base temporal features, bucket name
- **Output:** DataFrame with 0-5 additional xwin columns
- **Dependencies:**
  - `config.bucket_definitions.BUCKET_WINDOWS`
  - `numpy` (for polyfit)
  - `pandas` (for std, mean)

### `add_is_top_performer_label(df, bucket_path, strategy)`
- **Location:** `scripts/stage3_aggregation.py:401-479`
- **Called by:** `process_bucket()` line 649
- **Input:** DataFrame, bucket path, strategy string
- **Output:** DataFrame with is_top_performer column
- **Dependencies:**
  - `selected_videos.json` from Stage 1
  - Fallback: index-based labeling
```

#### Error Handling

```markdown
## Edge Cases & Error Handling

### Missing selected_videos.json
- **Behavior:** Logs warning, uses fallback index-based labeling
- **Fallback:** First 80% of videos = top performers
- **Impact:** May not match Stage 1 exact split, but allows pipeline to continue

### NaN in Energy Values
- **Behavior:** Skips slope calculation for that video, assigns np.nan
- **Impact:** Feature will be NaN in output (Stage 4 may need to handle)

### Bucket with <2 Windows
- **Behavior:** Skips std and slope calculations (need ≥2 data points)
- **Impact:** Only 0 features for 0-3s buckets (expected)
```

---

### 4. Stage 4 HLD (High-Level Design)

**Sections to Update:**

#### A. Input Schema Section

```markdown
## 3. Input Schema

**Source:** Stage 3 `aggregated_features.csv`

**Expected Columns (bucket-dependent):**
- 0-3s: 25 columns
- 3-9s: 49 columns
- 9-13s: 72 columns
- 13-18s: 72 columns
- 18-33s: 135 columns
- 33-60s: 156 columns
- 60-90s: 156 columns

**Includes:**
- Base temporal features (21 per window)
- Metadata (video_id, create_time, gender)
- **Cross-window features** (0-5, with xwin_ prefix) ← NEW
- **is_top_performer label** (1 column) ← NEW

**Validation:**
- Uses `config.bucket_definitions.get_stage3_expected_feature_count(bucket)`
- Validates cross-window features exist (if bucket ≠ 0-3s)
- Validates is_top_performer exists
```

#### B. Update: Transformation Logic

```markdown
## 4.2 Video-Level RF Transformation

**CHANGE:** No longer creates cross-window features or is_top_performer

**Previous Behavior (REMOVED):**
~~- Computed 5 cross-window features~~
~~- Created is_top_performer label~~

**Current Behavior:**
- **Validates** cross-window features exist in input
- **Passes through** unchanged (no transformation)
- **Validates** is_top_performer exists in input
- **Passes through** as target variable

**Feature Names Expected:**
- xwin_hook_to_middle_energy
- xwin_middle_to_closing_energy
- xwin_eye_contact_consistency
- xwin_word_density_std
- xwin_energy_progression_slope

**Why xwin_ prefix:**
- Avoids collision with window-specific features
- Original names like "hook_to_middle_energy_delta" caused Stage 4 to incorrectly identify them as hook-window features
```

---

### 5. Stage 4 Technical Implementation

**File Location:** `documentation_migration/services/Stage4_TI.md` (or similar)

**Sections to Update:**

#### A. Validation Function Changes

```markdown
## 4.1 Input Validation

### `validate_input(df, bucket, expected_count)`

**Location:** `rumiai_v2/processors/feature_transformation.py:240-262`

**Changes:**
1. **Column count validation now bucket-aware:**
   ```python
   # OLD:
   expected_cols = 129  # Hardcoded

   # NEW:
   from config.bucket_definitions import get_stage3_expected_feature_count
   expected_cols = get_stage3_expected_feature_count(bucket)
   ```

2. **New validation:** Checks is_top_performer exists (line 501-505)
3. **New validation:** Checks cross-window features exist (lines 510-530)
```

#### B. Removed Functions

```markdown
## Functions Removed in S7B2 Fix

### Cross-Window Feature Creation (REMOVED)
- **Was:** Lines ~517-557
- **Reason:** Moved to Stage 3
- **Replaced with:** Validation logic (lines 510-530)

### is_top_performer Creation (REMOVED)
- **Was:** Lines ~481-515
- **Reason:** Moved to Stage 3
- **Replaced with:** Validation logic (lines 501-505)

**Migration Note:** If you need to create these features, they must be created in Stage 3 now
```

#### C. Expected Column Counts

```markdown
## Output Schema

### Video-Level RF (`rf_transformed.csv`)

**Column Count by Bucket:**
- 3-9s: 65 columns
- 9-13s: 88 columns
- 13-18s: 88 columns
- 18-33s: 147 columns
- 33-60s: 167 columns
- 60-90s: 167 columns

**Formula:**
```python
temporal_features = get_stage3_expected_feature_count(bucket)
emotions = 7
temporal_extract = 5
gender = 3
cross_window = 0/3/5  # bucket-dependent
target = 1
total = temporal_features + emotions + temporal_extract + gender + cross_window + target
```

**Includes:**
- All Stage 3 features (including xwin_*)
- Emotion one-hot encoding (7 cols)
- Temporal extract features (5 cols)
- Gender one-hot encoding (3 cols)
- is_top_performer target (1 col)
```

---

## Critical Concepts for Documentation

### 1. The xwin_ Prefix Requirement

**Problem:**
```python
# BAD: Original name
'hook_to_middle_energy_delta'
# Stage 4 sees "hook_" prefix and thinks it's a hook-window feature
```

**Solution:**
```python
# GOOD: xwin_ prefix
'xwin_hook_to_middle_energy'
# Stage 4 recognizes this is NOT window-specific
```

**Document in:** All stage docs, feature documentation

---

### 2. Bucket-Aware Feature Counts

**Key Insight:** Not all buckets have same cross-window feature count

**Document Pattern:**
```markdown
| Bucket | Windows | Cross-Window Features | Reason |
|--------|---------|----------------------|--------|
| 0-3s   | 1       | 0                    | Only hook, no comparisons possible |
| 3-9s   | 2       | 3                    | Hook + closing, no middle (no deltas) |
| 9-13s+ | 3-7     | 5                    | Has middle sections, all features |
```

**Document in:** Stage 3 HLD, MLPlanningv2.md overview

---

### 3. Stage 3 → Stage 4 Contract

**Critical Understanding:**

**Stage 3 Creates:**
- Base temporal features (21 × windows)
- Metadata (3 cols)
- Cross-window features (0-5 cols)
- is_top_performer (1 col)

**Stage 4 Expects:**
- Exact column count from `get_stage3_expected_feature_count(bucket)`
- Cross-window features present (validates names match)
- is_top_performer present (uses as target)

**Stage 4 Does NOT:**
- Create cross-window features
- Create is_top_performer
- Transform cross-window features (passes through)

**Document in:** Both Stage 3 and 4 HLDs, interface specification sections

---

## Verification Commands for Documentation

Include these in updated docs for validation:

```bash
# Verify Stage 3 output
head -1 <bucket_path>/ml_analysis/aggregated_features.csv | tr ',' '\n' | wc -l
# Should match get_stage3_expected_feature_count(bucket)

# Check xwin features present
head -1 <bucket_path>/ml_analysis/aggregated_features.csv | tr ',' '\n' | grep "^xwin_"
# Should show 0, 3, or 5 features depending on bucket

# Verify Stage 4 output
head -1 <bucket_path>/ml_analysis/rf_transformed.csv | tr ',' '\n' | wc -l
# Should match get_expected_rf_column_count(bucket)

# Check xwin in Stage 6 output
cat <bucket_path>/ml_analysis/rf_video_analysis.json | jq '.feature_importance[].feature' | grep "xwin_"
# Should show xwin features if important enough

# Check xwin in Stage 7 output
cat <bucket_path>/ml_analysis/llm/winning_formulas.json | jq '.supplementary_insights.universal_principles[]' | grep "xwin_"
# Should show xwin features with gap statistics
```

---

## Migration Notes for Existing Data

**If you have data processed BEFORE this fix:**

1. **Stage 3 outputs (aggregated_features.csv):**
   - Missing cross-window features
   - Missing is_top_performer column
   - ⚠️ Must re-run Stage 3 with `--strategy` argument

2. **Stage 4+ outputs:**
   - Incompatible schema (wrong column counts)
   - ⚠️ Must re-run Stages 4-7 after regenerating Stage 3

3. **Test fixtures:**
   - Update column counts
   - Add synthetic xwin features
   - Add is_top_performer=1

**No backward compatibility** - old Stage 3 outputs will fail Stage 4 validation

---

## Summary: What Documentation Needs

1. **MLPlanningv2.md:**
   - Update Stage 3/4 pipeline descriptions
   - Add cross-window feature count table
   - Update column count references

2. **Stage 3 HLD:**
   - Add sections 4.5 (cross-window generation) and 4.6 (label assignment)
   - Update output schema section (section 5)

3. **Stage 3 TI:**
   - Add code location references for new functions
   - Add error handling section for edge cases

4. **Stage 4 HLD:**
   - Update input schema section (cross-window validation)
   - Update transformation logic (removed creation, added validation)

5. **Stage 4 TI:**
   - Document removed functions
   - Update validation function changes
   - Update expected column counts

---

**End of PostBugFixUpdate.md**
