# Stage 7 LLM Output Improvements - Implementation Guide

**Document Status**: Current Implementation Strategy (2025-10-30)
**Supersedes**: LLMOutputFix.md (archived for historical reference)

---

## 📖 Quick Navigation

- [Executive Summary](#executive-summary)
- [Issues Identified](#issues-identified)
- [✅ Current Implementation Strategy](#-current-implementation-strategy-phase-1-upstream-fix)
- [Implementation Steps](#implementation-steps)
- [Testing & Validation](#testing--validation)

---

## Executive Summary

### Problems
Stage 7 LLM outputs contain three issues that reduce creator-friendliness:
1. **Issue #1**: Raw technical feature names in `supplementary_insights` (e.g., "hook_energy_max: 0.13")
2. **Issue #2**: Granular middle segments in `structure` field (middle_1, middle_2, middle_3 instead of single "middle")
3. **Issue #3**: Numeric feature values in `step_by_step_template` (e.g., "eye contact (0.77)")

### Solution
**Phase 1 Upstream Fix**: Add semantic interpretation at Phase 1 prompt construction, so semantic labels flow through the entire pipeline.

### Impact
- ✅ Fixes Issue #1 + Issue #3 with ONE implementation (upstream preprocessing)
- ✅ Issue #2 already fixed via Phase 2 prompt updates (2025-10-30)
- ✅ 100% deterministic (Python interprets, not LLM)
- ✅ ~2-3 hours implementation time

### Status
- ✅ **Issue #2**: FIXED (condensed prompt rules added to `stage7_prompts.py`)
- 🟡 **Issue #1 + #3**: Ready to implement (semantic_interpretations.py complete, just needs integration)

---

## Issues Identified

### Issue #1: Supplementary Insights - Raw Technical Data

**Location**: `winning_formulas.json` → `supplementary_insights.universal_principles`

**Problem**: Outputs raw ML feature names that creators can't understand.

**Current output**:
```json
"universal_principles": [
  "hook_gesture_count: 0.00 in top vs 0.80 in bottom (gap: 0.80)",
  "middle_2_gesture_count: 0.00 in top vs 0.90 in bottom (gap: 0.90)",
  "middle_3_emotional_valence: -0.18 in top vs 0.30 in bottom (gap: 0.49)"
]
```

**Issues**:
- ❌ `hook_gesture_count` - Technical jargon
- ❌ `0.00 in top vs 0.80 in bottom` - What does 0.00 mean?
- ❌ `gap: 0.80` - Is this good or bad?

**Expected output** (after fix):
```json
"universal_principles": [
  "Minimal hand gestures in opening: 72% of top performers use still framing vs 15% of bottom",
  "Consistent minimal gestures throughout: Found in 68% of winning videos",
  "Neutral to positive emotional tone: Top performers average neutral (0.0) vs bottom negative (-0.18)"
]
```

**Why it matters**: Creators need actionable insights in plain English, not normalized ML values.

#### Root Cause: Current Data Flow

Understanding WHY Issue #1 occurs helps debug the Phase 1 upstream fix:

```
┌──────────────────────────────────────────────────────────────┐
│ Stage 6: RF Video-Level Analysis                            │
│ Outputs: rf_video_analysis.json                             │
│   feature_importance: [                                      │
│     {                                                        │
│       "feature": "hook_gesture_count",                       │
│       "top_performer_avg": 0.00,                             │
│       "bottom_performer_avg": 0.80,                          │
│       "gap": 0.80                                            │
│     }                                                        │
│   ]                                                          │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 7 Phase 2: Python Preprocessing                       │
│ Function: generate_universal_principles()                   │
│ Location: ml_pipeline/stage7_llm_analysis/                  │
│           stage7_preprocessing.py (lines 444-620)           │
│                                                              │
│ for feature in rf_video_data['feature_importance']:         │
│     principle = f"{feature['feature']}: "                   │
│     principle += f"{feature['top_performer_avg']} in top"   │
│     principle += f" vs {feature['bottom_performer_avg']}"   │
│     universal_principles.append(principle)                  │
│                                                              │
│ Returns: List[str] with RAW feature names and values        │
│   ["hook_gesture_count: 0.00 in top vs 0.80 in bottom..."] │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 7 Phase 2: LLM Prompt Construction                    │
│ Function: build_phase2_prompt()                             │
│ Location: ml_pipeline/stage7_llm_analysis/                  │
│           stage7_prompts.py (lines 841-843)                 │
│                                                              │
│ prompt += f'''                                               │
│   "supplementary_insights": {{                              │
│     "universal_principles": {json.dumps(universal_principles)}│
│   }}                                                         │
│ '''                                                          │
│                                                              │
│ LLM receives PRE-FILLED JSON with raw values!               │
│ LLM just COPIES this into output (no transformation)        │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ Output: winning_formulas.json                               │
│ "supplementary_insights": {                                 │
│   "universal_principles": [                                 │
│     "hook_gesture_count: 0.00 in top vs 0.80 in bottom"    │
│   ]                                                         │
│ }                                                           │
│                                                             │
│ ❌ Raw feature names and normalized values preserved        │
└──────────────────────────────────────────────────────────────┘
```

**Key Issue**: Python generates raw strings → LLM receives them as pre-filled JSON → LLM copies without interpretation

#### ⚠️ CRITICAL VERIFICATION: Does Phase 1 Fix Actually Solve Issue #1?

**Question**: If we fix Phase 1 to use semantic labels, will those labels reach `generate_universal_principles()` which generates Issue #1's raw output?

**Answer**: **NO** - Phase 1 and Issue #1 are in DIFFERENT data flows! Additional fix needed.

**Detailed Trace**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1 LLM Output (window analyses)                            │
│ ✅ Uses semantic labels after our fix                           │
│                                                                  │
│ {                                                                │
│   "hook": {                                                      │
│     "clusters": [{                                               │
│       "defining_features": [                                     │
│         "rapid cuts with frequent eye contact"  ← Semantic!     │
│       ]                                                          │
│     }]                                                           │
│   }                                                              │
│ }                                                                │
└────────────────────┬─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Uses Phase 1 output for creative_reports               │
│ ✅ Inherits semantic labels                                     │
│                                                                  │
│ step_by_step_template: [                                        │
│   "Hook (0-3s): Use rapid cuts with frequent eye contact"       │
│ ]                                                                │
│                                                                  │
│ ✅ Issue #3 FIXED (semantic labels in step_by_step)             │
└──────────────────────────────────────────────────────────────────┘

BUT...

┌─────────────────────────────────────────────────────────────────┐
│ Stage 6: RF Video-Level Analysis (SEPARATE source)              │
│ ❌ Still outputs RAW feature names and values                   │
│                                                                  │
│ rf_video_analysis.json:                                         │
│   feature_importance: [                                          │
│     {"feature": "hook_gesture_count", "top_avg": 0.00, ...}     │
│   ]                                                              │
└────────────────────┬─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ generate_universal_principles() - stage7_preprocessing.py       │
│ ❌ Reads DIRECTLY from Stage 6 RF data (not Phase 1 output)    │
│                                                                  │
│ for feat in rf_video_data['feature_importance']:                │
│     principle = f"{feat['feature']}: {feat['top_avg']}..."      │
│                                                                  │
│ ❌ Issue #1 NOT FIXED by Phase 1 semantic labels!               │
└──────────────────────────────────────────────────────────────────┘
```

**Reality Check**: Phase 1 semantic labels DON'T reach `generate_universal_principles()` because:
1. Phase 1 analyzes **window-level K-Means clusters**
2. Issue #1 uses **video-level RF feature importance** (different source!)
3. `generate_universal_principles()` reads directly from Stage 6 RF output, not Phase 1

**The Fix That Actually Works**:

Option A: Update `generate_universal_principles()` to use semantic interpretation
```python
# In stage7_preprocessing.py (lines 444-620)
from config.semantic_interpretations import interpret_value, extract_base_feature

def generate_universal_principles(rf_video_data: dict, top_n: int = 7) -> List[str]:
    principles = []

    for feature in rf_video_data['feature_importance'][:top_n]:
        feature_name = feature['feature']
        top_avg = feature['top_performer_avg']
        bottom_avg = feature['bottom_performer_avg']
        gap = feature['gap']

        # Extract base feature and get semantic interpretation
        base_feature = extract_base_feature(feature_name)
        label_top, _ = interpret_value(base_feature, top_avg)
        label_bottom, _ = interpret_value(base_feature, bottom_avg)

        # Extract window context
        window = feature_name.split('_')[0] if '_' in feature_name else ''
        window_text = {'hook': 'in opening', 'closing': 'in closing',
                       'xwin': 'throughout'}.get(window, '')

        # Format with semantic labels
        if label_top != 'unknown' and label_bottom != 'unknown':
            principle = f"{label_top.capitalize()} {window_text}: "
            principle += f"Top performers use {label_top} vs bottom use {label_bottom}"
        else:
            # Fallback to current format if no semantic interpretation
            principle = f"{feature_name}: {top_avg:.2f} in top vs {bottom_avg:.2f} in bottom (gap: {gap:.2f})"

        principles.append(principle)

    return principles
```

**Updated Implementation Required**:
1. ✅ Phase 1 fix (Step 2) - Fixes Issue #3
2. ✅ Additional fix to `generate_universal_principles()` - Fixes Issue #1

**Status**: Issue #1 requires ADDITIONAL implementation beyond Phase 1 fix

---

### Issue #2: Granular Middle Structure ✅ FIXED

**Location**: `winning_formulas.json` → `creative_reports[].structure`

**Problem**: Outputs granular middle segments (middle_1, middle_2, etc.) that confuse creators.

**Current output**:
```json
"structure": {
  "hook": "Verbal-Heavy Direct Hook (Cluster 2)",
  "middle_1": "Verbal Engagement Hook (Cluster 0)",
  "middle_2": "Dynamic Gesture-Rich Approach (Cluster 1)",
  "middle_3": "Speech-Driven Approach (Cluster 2)",
  "middle_4": "Silent Emotional Hook (Cluster 0)",
  "closing": "Static Single-Scene Closer (Cluster 0)"
}
```

**Issues**:
- ❌ 4 separate middle segments (confusing!)
- ❌ Creators ask: "Why 4 middle sections? What's the difference?"
- ❌ Inconsistent with 3-part structure (hook/middle/closing)

**Expected output** (after fix):
```json
"structure": {
  "hook": "Verbal-Heavy Direct Hook (Cluster 2)",
  "middle": "Transitions from verbal engagement through gesture-rich dynamics to speech-driven climax, ending in silent emotional connection",
  "closing": "Static Single-Scene Closer (Cluster 0)"
}
```

**Status**: ✅ **FIXED** (2025-10-30) via condensed prompt rules in `stage7_prompts.py` lines 794-870

---

### Issue #3: Step-by-Step Template with Numeric Values

**Location**: `winning_formulas.json` → `creative_reports[].step_by_step_template`

**Problem**: Outputs normalized ML feature values that creators can't interpret.

**Current output**:
```json
"step_by_step_template": [
  "Hook (0-3s): High word count (0.85), strong eye contact (0.58), minimal gestures",
  "Middle_1 (3-8s): Peak verbal engagement (0.87 words), maintain eye contact (0.72)",
  "Middle_2 (8-13s): Continue high verbal (0.88), minimal gestures (0.06)",
  "Closing (23-26s): Maintain silence, single scene (0.00 variance), consistent energy"
]
```

**Issues**:
- ❌ Numeric values: `(0.85)`, `(0.58)`, `(0.06)` - meaningless to creators
- ❌ Granular middle segments: Middle_1, Middle_2, etc.
- ⚠️ Timings: `(0-3s)`, `(3-8s)` - THESE ARE HELPFUL (should keep!)

**Expected output** (after fix):
```json
"step_by_step_template": [
  "Hook (0-3s): Use substantial dialogue with moderate eye contact and minimal gestures",
  "Middle (3-23s): Maintain high verbal engagement with consistent eye contact, then transition to complete silence for emotional processing",
  "Closing (23-26s): Use single sustained scene with consistent energy for emotional resolution"
]
```

**Why timings should stay**: Window timings `(0-3s)` provide helpful context for WHEN to apply tactics. Feature values `(0.85)` are confusing measurements that should be removed.

---

## ✅ Current Implementation Strategy: Phase 1 Upstream Fix

### The Insight

**Fix at the source (Phase 1) instead of downstream (Phase 2)**

```
┌─────────────────────────────────────────────────────────────┐
│ OLD APPROACH (Downstream - Phase 2 Fix)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Phase 1 LLM sees: "scene_count: 0.58"                      │
│       ↓ (LLM guesses interpretation)                        │
│ Phase 1 outputs: "rapid cuts (0.58)"  ← Still has numbers! │
│       ↓                                                      │
│ Phase 2: Must remove numbers from Phase 1 output           │
│       ↓                                                      │
│ Result: Issue #3 fixed, but Issue #1 needs separate fix    │
│                                                              │
│ ❌ 2 separate fixes needed                                  │
│ ❌ Phase 1 LLM still guesses                                │
│ ❌ ~6+ hours implementation                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ NEW APPROACH (Upstream - Phase 1 Fix) ✅                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Python preprocessing: 0.58 → "rapid cuts (5-8 changes)"    │
│       ↓                                                      │
│ Phase 1 LLM sees: "rapid cuts (5-8 changes)" ← No numbers! │
│       ↓                                                      │
│ Phase 1 outputs: "Hook (0-3s): Use rapid cuts"             │
│       ↓                                                      │
│ Phase 2: Inherits semantic labels from Phase 1             │
│       ↓                                                      │
│ Result: Issue #3 fixed!                                     │
│                                                              │
│ ⚠️  Issue #1 needs ADDITIONAL fix (see verification above)  │
│ ✅ 100% deterministic (Python interprets)                   │
│ ✅ ~3-4 hours implementation (2 fixes needed)               │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Flow

```
┌──────────────────────────────────────────────────────────────┐
│ Stage 6: Cluster Analysis                                    │
│ Outputs: Cluster centroids with normalized values           │
│   {scene_count_scaled: 0.58, eye_contact_rate_scaled: 0.77} │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ ✨ NEW: Python Preprocessing (stage7_prompts.py line ~424)  │
│ Import: semantic_interpretations.py                         │
│                                                              │
│ FOR each cluster feature:                                   │
│   1. Extract base feature: scene_count_scaled → scene_count │
│   2. Interpret: interpret_value('scene_count', 0.58)        │
│   3. Returns: ('rapid cuts', '5-8 scene changes')           │
│                                                              │
│ Outputs: Semantic labels replace raw numbers                │
│   "rapid cuts (5-8 scene changes)"                          │
│   "frequent eye contact (77% camera gaze)"                  │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 1 LLM (receives semantic labels, NOT raw numbers)     │
│                                                              │
│ Cluster 1: High Performers                                  │
│   1. rapid cuts - 5-8 scene changes                         │
│   2. frequent eye contact - 77% camera gaze                 │
│   3. moderate volume - Mid average volume                   │
│                                                              │
│ LLM Task: Synthesize into creative strategy                 │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 1 Output (window analyses)                            │
│                                                              │
│ defining_features: [                                         │
│   "rapid cuts with frequent eye contact",                   │
│   "moderate volume for balanced delivery"                   │
│ ]                                                            │
│                                                              │
│ ✅ Issue #3 PREVENTED (no raw numbers entered system)      │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 2 LLM (receives Phase 1 output + RF data)             │
│                                                              │
│ Semantic labels from Phase 1 flow through naturally         │
│ Combines with RF importance rankings                        │
│                                                              │
│ Outputs:                                                     │
│   step_by_step_template:                                    │
│     "Hook (0-3s): Use rapid cuts with frequent eye contact" │
│                                                              │
│   supplementary_insights:                                   │
│     "Scene pacing: 72% of top use rapid cuts vs 28% slow"  │
│                                                              │
│ ✅ Issue #1 FIXED (inherited semantic labels)               │
│ ✅ Issue #3 FIXED (never had raw numbers to begin with)     │
└──────────────────────────────────────────────────────────────┘
```

### Why This Works

**Single Source of Truth**: `semantic_interpretations.py` (26 features, all ranges defined)

```python
SEMANTIC_INTERPRETATIONS = {
    'scene_count': {
        'ranges': [
            (0, 2, 'minimal cuts', 'Up to 2 scene changes'),
            (2, 5, 'moderate cuts', '2 to 5 scene changes'),
            (5, 8, 'frequent cuts', '5 to 8 scene changes'),
            (8, 15, 'rapid cuts', '8-15 scene changes'),
            (15, 100, 'very rapid cuts', 'Over 15 scene changes')
        ]
    },

    'eye_contact_rate': {
        'ranges': [
            (0.0, 0.2, 'minimal eye contact', 'rarely looks at camera'),
            (0.2, 0.5, 'occasional eye contact', 'some camera engagement'),
            (0.5, 0.7, 'moderate eye contact', 'balanced camera presence'),
            (0.7, 0.9, 'frequent eye contact', 'mostly looks at camera'),
            (0.9, 1.0, 'constant eye contact', 'continuous camera gaze')
        ]
    },

    # ... 24 more features (all complete)
}

def interpret_value(feature: str, value: float) -> tuple[str, str]:
    """Convert 0.58 → ('rapid cuts', '8-15 scene changes')"""
    # Returns (label, description)
```

**Status**: ✅ All 26 features complete (see `config/semantic_interpretations.py`)

---

## Implementation Steps

### Prerequisites ✅ Already Complete

1. ✅ All 26 semantic range definitions exist in `config/semantic_interpretations.py`
2. ✅ `interpret_value()` function implemented and tested
3. ✅ Issue #2 fixed via Phase 2 prompt updates (lines 794-870 in `stage7_prompts.py`)

### Step 1: Add `extract_base_feature()` Helper

**File**: `config/semantic_interpretations.py`
**Location**: After `interpret_value()` function (around line 460)
**Time**: 30 minutes

**Purpose**: Remove window prefixes from feature names
- `hook_scene_count` → `scene_count`
- `middle_2_eye_contact_rate` → `eye_contact_rate`
- `xwin_energy_progression_slope` → `energy_progression_slope`

**Implementation**:

```python
def extract_base_feature(feature: str) -> str:
    """
    Extract base feature name by removing window prefix.

    Args:
        feature: Full feature name (e.g., 'hook_energy_level', 'middle_2_scene_count')

    Returns:
        str: Base feature name (e.g., 'energy_level', 'scene_count')

    Examples:
        >>> extract_base_feature('hook_energy_level')
        'energy_level'

        >>> extract_base_feature('middle_2_scene_count')
        'scene_count'

        >>> extract_base_feature('xwin_eye_contact_consistency')
        'eye_contact_consistency'

        >>> extract_base_feature('energy_level')
        'energy_level'
    """
    # Remove cross-window prefix
    if feature.startswith('xwin_'):
        return feature[5:]  # Remove 'xwin_'

    # Remove temporal window prefixes
    prefixes = ['hook_', 'closing_']
    for prefix in prefixes:
        if feature.startswith(prefix):
            return feature[len(prefix):]

    # Remove middle segment prefixes (middle_1_, middle_2_, etc.)
    import re
    middle_pattern = r'^middle_\d+_'
    match = re.match(middle_pattern, feature)
    if match:
        return feature[len(match.group()):]

    # Special case: middle_aggregate_
    if feature.startswith('middle_aggregate_'):
        return feature[17:]  # Remove 'middle_aggregate_'

    # No prefix found - return as-is
    return feature


# Update exports
__all__ = ['SEMANTIC_INTERPRETATIONS', 'interpret_value', 'extract_base_feature']
```

---

### Step 2: Update Phase 1 Prompt Construction

**File**: `ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
**Function**: `build_phase1_prompt()`
**Location**: Around line 424 (cluster feature formatting loop)
**Time**: 1 hour

**Current code**:
```python
for j, enriched_feat in enumerate(cluster_data['enriched_features'][:12], 1):
    prompt += f"  {j}. {enriched_feat['formatted']}\n"
    # Outputs: "scene_count_scaled: 0.58 (RF rank #3, importance 0.35)"
```

#### Verify Data Structure

Before modifying the code, verify `enriched_feat` structure to prevent KeyError:

**Structure Definition** (`ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py` lines 80-110):

```python
def enrich_high_contrast_features(...) -> List[dict]:
    """
    Returns:
        List[dict]: Enriched features with structure:
            [
                {
                    'feature': 'scene_count_scaled',      # Feature name with _scaled suffix
                    'centroid_value': 0.58,               # Normalized value from K-Means centroid
                    'rf_rank': 3,                          # RF importance rank (1-based)
                    'rf_importance': 0.35,                 # RF importance score
                    'gap': 0.43,                           # Top vs bottom performer gap
                    'formatted': 'scene_count: 0.58...'   # Current formatted string (will replace)
                },
                ...
            ]
    """
```

**Expected Keys**:
- `feature`: Full feature name (e.g., `'hook_scene_count_scaled'`)
- `centroid_value`: Normalized value [0, 1] from cluster centroid
- `rf_rank`: Integer rank from RF analysis (may be `None` if not in top features)
- `rf_importance`: Float importance score (may be `0.0` if not in top features)
- `formatted`: Current string format (we'll replace this)

**Debugging Tips**:

If you encounter `KeyError`:
1. **Check actual structure**:
   ```python
   # Add at line 424 temporarily
   print(f"DEBUG enriched_feat keys: {enriched_feat.keys()}")
   print(f"DEBUG enriched_feat sample: {enriched_feat}")
   ```

2. **Verify function returns expected format**:
   ```bash
   grep -A 30 "def enrich_high_contrast_features" ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py
   ```

3. **Check if key names changed**: If structure doesn't match, adjust `.get()` calls to actual key names

**Updated code**:
```python
# Add imports at top of file (around line 17)
import logging
from config.semantic_interpretations import interpret_value, extract_base_feature

logger = logging.getLogger("rumiai.stage7_prompts")

# Update formatting loop (around line 424)
for j, enriched_feat in enumerate(cluster_data['enriched_features'][:12], 1):
    feature_name = enriched_feat.get('feature', '')

    # Get normalized value from enriched feature
    # enriched_feat structure: {'feature': 'hook_scene_count_scaled', 'centroid_value': 0.58, ...}
    normalized_val = enriched_feat.get('centroid_value', 0.0)

    # Get RF metadata
    rf_rank = enriched_feat.get('rf_rank', 'N/A')
    rf_importance = enriched_feat.get('rf_importance', 0.0)

    # Extract base feature (remove window prefix + _scaled suffix)
    base_feature = extract_base_feature(feature_name)
    base_feature = base_feature.replace('_scaled', '')

    # Get semantic interpretation
    label, description = interpret_value(base_feature, normalized_val)

    # Fallback for unknown features (graceful degradation)
    if label == 'unknown':
        # Feature not in semantic_interpretations.py - use old format
        logger.warning(f"No semantic interpretation for {base_feature}, using raw value")
        prompt += f"  {j}. {feature_name}: {normalized_val:.2f}"
        if rf_rank != 'N/A':
            prompt += f" (RF rank #{rf_rank}, importance {rf_importance:.2f})"
        prompt += "\n"
    else:
        # Format with semantic labels (NO raw numbers)
        prompt += f"  {j}. {label} - {description}"
        if rf_rank != 'N/A':
            prompt += f" (RF rank #{rf_rank}, importance {rf_importance:.2f})"
        prompt += "\n"

    # Example output: "rapid cuts - 5-8 scene changes (RF rank #3, importance 0.35)"
```

**Impact**: Phase 1 LLM now receives semantic labels instead of raw normalized values!

**Example transformation**:
```
BEFORE:
  1. scene_count_scaled: 0.58 (RF rank #3, importance 0.35)
  2. eye_contact_rate_scaled: 0.77 (RF rank #1, importance 0.42)
  3. energy_level_scaled: 0.12 (RF rank #5, importance 0.18)

AFTER:
  1. rapid cuts - 5-8 scene changes (RF rank #3, importance 0.35)
  2. frequent eye contact - mostly looks at camera (RF rank #1, importance 0.42)
  3. quiet - Low average volume (RF rank #5, importance 0.18)
```

---

### Step 2B: Update `generate_universal_principles()` (Issue #1 Fix)

**File**: `ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py`
**Function**: `generate_universal_principles()`
**Location**: Around lines 444-620
**Time**: 1 hour

**Why this is needed**: As shown in Issue #1 verification above, Phase 1 semantic labels don't reach `generate_universal_principles()` because it reads directly from Stage 6 RF data (different source than Phase 1 K-Means clusters).

**Current code** (lines ~550-580):
```python
def generate_universal_principles(rf_video_data: dict, top_n: int = 7) -> List[str]:
    """Generate universal principles from RF video-level analysis."""
    principles = []

    for feature in rf_video_data['feature_importance'][:top_n]:
        feature_name = feature['feature']
        top_avg = feature['top_performer_avg']
        bottom_avg = feature['bottom_performer_avg']
        gap = feature['gap']

        # Current: Raw format
        principle = f"{feature_name}: {top_avg:.2f} in top vs {bottom_avg:.2f} in bottom (gap: {gap:.2f})"
        principles.append(principle)

    return principles
```

**Updated code**:
```python
# Add imports at top of file (around line 17)
from config.semantic_interpretations import interpret_value, extract_base_feature
import logging

logger = logging.getLogger("rumiai.stage7_preprocessing")

def generate_universal_principles(rf_video_data: dict, top_n: int = 7) -> List[str]:
    """Generate universal principles from RF video-level analysis with semantic labels."""
    principles = []

    for feature in rf_video_data['feature_importance'][:top_n]:
        feature_name = feature['feature']
        top_avg = feature['top_performer_avg']
        bottom_avg = feature['bottom_performer_avg']
        gap = feature['gap']

        # Extract base feature and get semantic interpretations
        base_feature = extract_base_feature(feature_name)
        label_top, desc_top = interpret_value(base_feature, top_avg)
        label_bottom, desc_bottom = interpret_value(base_feature, bottom_avg)

        # Extract window context (hook_, middle_X_, closing_, xwin_)
        window_prefix = feature_name.split('_')[0] if '_' in feature_name else ''
        window_text = {
            'hook': 'in opening',
            'closing': 'in closing',
            'xwin': 'throughout video'
        }.get(window_prefix, '')

        # Handle middle_N_ prefixes
        if 'middle' in feature_name:
            window_text = 'in middle'

        # Fallback if semantic interpretation not available
        if label_top == 'unknown' or label_bottom == 'unknown':
            logger.warning(f"No semantic interpretation for {base_feature}, using raw format")
            principle = f"{feature_name}: {top_avg:.2f} in top vs {bottom_avg:.2f} in bottom (gap: {gap:.2f})"
        else:
            # Format with semantic labels
            principle = f"{label_top.capitalize()} {window_text}: "
            principle += f"Top performers use {label_top} vs bottom use {label_bottom} "
            principle += f"(gap: {gap:.2f})"

        principles.append(principle)

    return principles
```

**Example transformation**:
```
BEFORE:
  "hook_gesture_count: 0.00 in top vs 0.80 in bottom (gap: 0.80)"

AFTER:
  "No gestures in opening: Top performers use no gestures vs bottom use minimal gestures (gap: 0.80)"
```

**Impact**: Issue #1 FIXED - supplementary_insights now uses creator-friendly semantic labels!

---

### Step 3: Verify Phase 2 Prompt (Issue #2 Already Fixed)

**File**: `ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
**Location**: Lines 794-870
**Status**: ✅ Already implemented (2025-10-30)

**What was added**:
- Structure Field Guidelines: Synthesize middle_1, middle_2, etc. into single "middle" field
- Step-by-Step Template Guidelines: Keep timings (0-3s), remove feature numbers (0.58)

**Verify it contains**:
```python
## Output Format Rules

### 1. Structure Field (Issue #2 Fix)
**Rule**: Output single "middle" field (NOT middle_1, middle_2, etc.)

### 2. Step-by-Step Template (Issue #3 Fix)
**Rules**:
1. Synthesize all middle segments into single "Middle:" entry
2. ✅ KEEP window timings: (0-3s), (3-18s), (18-21s)
3. ❌ REMOVE feature value numbers: (0.77), (0.58), (0.25)
```

**Note**: With Phase 1 upstream fix, feature numbers won't even reach Phase 2, so removal is automatic!

---

### Step 4: Test with Real Data

**Time**: 30 minutes

**Test Case 1: Bucket 18-33s (Issue #2 & #3)**

```bash
cd /home/jorge/rumiaifinal

# Run Stage 7 on problematic bucket
python rumiai_ml_batch.py \
  --client rollo_test4 \
  --analysis-type hashtag \
  --target wellness_test4 \
  --analysis-mode top \
  --selection-strategy contrastive \
  --bucket-filter 18-33s \
  --stage 7

# Automated validation (use these commands to verify output)
OUTPUT_FILE="data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive/buckets/bucket_18-33s/ml_analysis/llm/winning_formulas.json"

# 1. Check for granular middle structure (Issue #2)
echo "Checking for granular middle segments..."
if grep -q '"middle_[0-9]"' "$OUTPUT_FILE"; then
    echo "❌ FAIL: Found middle_1, middle_2, etc. (Issue #2 not fixed)"
else
    echo "✅ PASS: Single 'middle' field only"
fi

# 2. Check for feature numbers in step_by_step_template (Issue #3)
echo "Checking for feature numbers..."
if grep -E '"step_by_step_template"' -A 10 "$OUTPUT_FILE" | grep -E '\([0-9]\.[0-9]+\)' | grep -v '([0-9]+-[0-9]+s)'; then
    echo "❌ FAIL: Found feature numbers like (0.77), (0.58)"
else
    echo "✅ PASS: No feature numbers found"
fi

# 3. Verify window timings are preserved (Issue #3)
echo "Checking window timings..."
if grep -E '"step_by_step_template"' -A 10 "$OUTPUT_FILE" | grep -E '\([0-9]+-[0-9]+s\)' > /dev/null; then
    echo "✅ PASS: Window timings like (0-3s) preserved"
else
    echo "⚠️  WARNING: No window timings found (may be correct for very short videos)"
fi

# 4. Check for raw feature names in supplementary_insights (Issue #1)
echo "Checking for raw feature names..."
if grep -E '"universal_principles"' -A 10 "$OUTPUT_FILE" | grep -E '(hook|middle|closing)_[a-z_]+:' > /dev/null; then
    echo "❌ FAIL: Found raw feature names like 'hook_gesture_count:'"
else
    echo "✅ PASS: No raw feature names in supplementary_insights"
fi

echo ""
echo "Manual verification:"
cat "$OUTPUT_FILE" | jq '.creative_reports[0].structure, .creative_reports[0].step_by_step_template, .supplementary_insights.universal_principles[0:3]'
```

**Test Case 2: Bucket 3-9s (Special Case - No Middle)**

```bash
# Test short video handling
python rumiai_ml_batch.py \
  --client rollo_test3 \
  --analysis-type hashtag \
  --target wellness_test3 \
  --analysis-mode top \
  --selection-strategy contrastive \
  --bucket-filter 3-9s \
  --stage 7

# Check output
cat data/clients/rollo_test3/hashtags/wellness_test3/top_contrastive/buckets/bucket_3-9s/ml_analysis/llm/winning_formulas.json

# Verify:
# ✅ structure has NO "middle" key (only hook + closing)
# ✅ step_by_step_template has 2 items (Hook + Closing only)
# ✅ No numbers in step_by_step_template
```

**Expected Results**:

```json
// Bucket 18-33s
{
  "structure": {
    "hook": "Verbal-Heavy Direct Hook",
    "middle": "Transitions from verbal engagement through dynamic gestures to silent emotional connection",
    "closing": "Static Single-Scene Closer"
  },
  "step_by_step_template": [
    "Hook (0-3s): Use substantial dialogue with moderate eye contact and minimal gestures",
    "Middle (3-23s): Transition from high verbal engagement to complete silence for emotional processing",
    "Closing (23-26s): Use single sustained scene with consistent energy"
  ]
}

// Bucket 3-9s
{
  "structure": {
    "hook": "Fast-Cut Energy Hook",
    "closing": "Balanced Multi-Element Closer"
  },
  "step_by_step_template": [
    "Hook (0-3s): Use rapid scene cuts for high visual stimulation",
    "Closing (6-9s): Transition to longer scenes with dynamic gaze for sustained message"
  ]
}
```

---

## Testing & Validation

### Manual Validation Checklist

After running Stage 7, manually inspect `winning_formulas.json`:

**Issue #1 (supplementary_insights)**:
- [ ] `universal_principles` uses semantic labels ("rapid cuts" not "0.58")
- [ ] No raw feature names (no "hook_gesture_count")
- [ ] Percentages are creator-friendly ("72% of top performers")

**Issue #2 (structure)**:
- [ ] `structure` has single "middle" field (NOT middle_1, middle_2, etc.)
- [ ] Short videos (3-9s) omit "middle" key entirely
- [ ] Middle description synthesizes overall progression

**Issue #3 (step_by_step_template)**:
- [ ] NO feature numbers: (0.58), (0.77), (0.25), etc.
- [ ] HAS window timings: (0-3s), (3-18s), (18-21s)
- [ ] Long videos have 3 items: Hook, Middle, Closing
- [ ] Short videos (3-9s) have 2 items: Hook, Closing

### Automated Validation (Optional)

Create validation script:

```python
# scripts/validate_winning_formulas.py

import json
import re
from pathlib import Path

def validate_output(json_path: str) -> dict:
    """Validate winning_formulas.json against Issue #1, #2, #3 fixes."""

    with open(json_path) as f:
        data = json.load(f)

    issues = []

    # Issue #1: Check supplementary_insights
    principles = data.get('supplementary_insights', {}).get('universal_principles', [])
    for principle in principles:
        if re.search(r'_\d+_', principle):  # Check for middle_1_, middle_2_, etc.
            issues.append(f"Issue #1: Raw feature name in principle: {principle}")
        if re.search(r': \d+\.\d+ in top', principle):  # Check for raw values
            issues.append(f"Issue #1: Raw value in principle: {principle}")

    # Issue #2 & #3: Check creative_reports
    for report in data.get('creative_reports', []):
        # Issue #2: Check structure
        structure = report.get('structure', {})
        if structure and any(k.startswith('middle_') for k in structure.keys()):
            issues.append(f"Issue #2: Granular middle in report {report['report_id']}")

        # Issue #3: Check step_by_step_template
        template = report.get('step_by_step_template', [])
        for step in template:
            # Check for feature numbers (but allow window timings)
            if re.search(r'\(\d+\.\d+\)', step):  # (0.58), (0.77)
                issues.append(f"Issue #3: Feature number in step: {step}")
            if re.search(r'Middle_\d+', step):  # Middle_1, Middle_2
                issues.append(f"Issue #3: Granular middle in step: {step}")

    return {
        'valid': len(issues) == 0,
        'issues': issues
    }

# Usage
result = validate_output('data/.../winning_formulas.json')
if result['valid']:
    print("✅ All validations passed!")
else:
    print("❌ Issues found:")
    for issue in result['issues']:
        print(f"  - {issue}")
```

---

## Summary

### What Gets Fixed

| Issue | Problem | Solution | Status |
|-------|---------|----------|--------|
| **#1** | Raw feature names in supplementary_insights | Update `generate_universal_principles()` (Step 2B) | 🟡 Ready to implement |
| **#2** | Granular middle_1, middle_2 in structure | Phase 2 prompt synthesis rules | ✅ Implemented (2025-10-30) |
| **#3** | Feature numbers (0.58) in step_by_step | Phase 1 prompt semantic preprocessing (Step 2) | 🟡 Ready to implement |

**Note**: Issue #1 and #3 require SEPARATE fixes (different data sources - see Issue #1 verification section)

### Implementation Effort

| Task | Time | Files Modified |
|------|------|----------------|
| **Step 1**: Add `extract_base_feature()` helper | 30 min | 1 file (semantic_interpretations.py) |
| **Step 2**: Update Phase 1 prompt construction | 1 hour | 1 file (stage7_prompts.py) |
| **Step 2B**: Update `generate_universal_principles()` | 1 hour | 1 file (stage7_preprocessing.py) |
| **Step 4**: Test with real data | 30 min | 0 files (validation) |
| **Total** | **3 hours** | **3 files** |

### Key Benefits

1. ✅ **Fixes all 3 issues** (Issue #1, #2, #3)
2. ✅ **100% deterministic** (Python interprets, not LLM guessing)
3. ✅ **Single source of truth** (semantic_interpretations.py)
4. ✅ **Consistent everywhere** (same semantic dictionary for all features)
5. ✅ **Easy to maintain** (update ranges in one place)
6. ✅ **Already complete infrastructure** (all 26 features defined)
7. ✅ **Graceful fallback** (handles unknown features without breaking)

### Next Actions

1. Implement `extract_base_feature()` in `config/semantic_interpretations.py`
2. Update Phase 1 prompt construction in `ml_pipeline/stage7_llm_analysis/stage7_prompts.py` (Issue #3 fix)
3. Update `generate_universal_principles()` in `ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py` (Issue #1 fix)
4. Test on bucket_18-33s and bucket_3-9s with automated validation
5. Validate all three issues are fixed
6. Run on full dataset

---

## Appendix: Example Transformations

### Before (Current System)

**Phase 1 Input**:
```
Cluster 0 centroids:
  scene_count_scaled: 0.58
  eye_contact_rate_scaled: 0.77
  energy_level_scaled: 0.12
```

**Phase 1 Output**:
```json
{
  "defining_features": [
    "scene_count: 0.58 (RF rank #3)",
    "eye_contact_rate: 0.77 (RF rank #1)"
  ]
}
```

**Phase 2 Output**:
```json
{
  "structure": {
    "hook": "...",
    "middle_1": "...",
    "middle_2": "...",
    "closing": "..."
  },
  "step_by_step_template": [
    "Hook (0-3s): Rapid cuts (0.58), strong eye contact (0.77)",
    "Middle_1 (3-8s): Continue pattern (0.60)...",
    "Closing: ..."
  ],
  "supplementary_insights": {
    "universal_principles": [
      "hook_scene_count: 0.58 in top vs 0.25 in bottom (gap: 0.33)"
    ]
  }
}
```

### After (With Phase 1 Upstream Fix)

**Phase 1 Input** (After Python preprocessing):
```
Cluster 0 features:
  rapid cuts - 5-8 scene changes (RF rank #3)
  frequent eye contact - mostly looks at camera (RF rank #1)
  quiet - Low average volume (RF rank #5)
```

**Phase 1 Output**:
```json
{
  "defining_features": [
    "rapid cuts with frequent eye contact for immediate engagement",
    "quiet volume for intimate, personal delivery"
  ]
}
```

**Phase 2 Output**:
```json
{
  "structure": {
    "hook": "Fast-Trust Hook",
    "middle": "Builds from rapid visual stimulation to sustained emotional connection",
    "closing": "Intimate Personal Closer"
  },
  "step_by_step_template": [
    "Hook (0-3s): Use rapid scene cuts with frequent eye contact for immediate engagement",
    "Middle (3-23s): Maintain visual dynamism while building personal connection through consistent eye contact",
    "Closing (23-26s): Transition to quiet, intimate delivery with sustained single scene"
  ],
  "supplementary_insights": {
    "universal_principles": [
      "Scene pacing in opening: 72% of top performers use rapid cuts vs 28% use moderate pacing",
      "Eye contact throughout: Top performers maintain frequent camera gaze (avg 77%) vs bottom occasional (avg 45%)"
    ]
  }
}
```

**Key Differences**:
- ✅ No raw numbers (0.58, 0.77)
- ✅ Single "middle" field (not middle_1, middle_2)
- ✅ Creator-friendly language throughout
- ✅ Semantic labels ("rapid cuts", "frequent eye contact")
- ✅ Window timings preserved (0-3s) for context

---

**Document Version**: 2.0 (2025-10-30)
**Status**: Current implementation guide
**Previous Version**: LLMOutputFix.md (archived)
