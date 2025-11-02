# ML Bug Fixes - Implementation Guide

**Date**: 2025-11-02
**Status**: Ready for Implementation
**Companion Document**: `MLBugFixes.md` (analysis and diagnosis)

This document provides **exact code changes** to fix all critical data quality issues in the ML pipeline.

---

## Table of Contents

1. [Bug #1: Add Distribution Data to Window-Level RF](#bug-1-add-distribution-data-to-window-level-rf)
2. [Bug #2: Fix Number Formatting in Prompt Builder](#bug-2-fix-number-formatting-in-prompt-builder)
3. [Bug #3: Fix RF Alignment Tolerance](#bug-3-fix-rf-alignment-tolerance)
4. [Bug #4: Fix Universal Principles Formatting](#bug-4-fix-universal-principles-formatting)
5. [Bug #5: Improve Semantic Interpretation Robustness](#bug-5-improve-semantic-interpretation-robustness)
6. [Testing & Validation](#testing--validation)

---

## Bug #1: Add Distribution Data to Window-Level RF

### **Priority**: 🔴 P0 CRITICAL

### **Problem**
Window-level RF analysis JSON files are missing the `distribution` field, causing bimodal detection to return `(0% high, 0% low)` for all features.

### **Root Cause**
`ml_pipeline/stage6_analysis/ml_analysis_generation.py:412-430` calculates only `top_performer_avg`, `bottom_performer_avg`, and `gap`, but does NOT calculate distribution percentiles like video-level RF does.

### **Fix Location**
File: `ml_pipeline/stage6_analysis/ml_analysis_generation.py`
Function: `generate_window_rf_json()` (lines 345-447)

### **Code Change**

Replace lines 412-430 with this extended version:

```python
    # ===== Step 4: Compute Distribution Stats =====
    rf_csv_path = os.path.join(bucket_path, f'ml_analysis/{window}_rf_transformed.csv')
    df = pd.read_csv(rf_csv_path)
    logger.debug(f"Loading {window}_rf_transformed.csv for distribution analysis ({len(df)} rows × {len(df.columns)} columns)")

    # Determine top/bottom performers
    if 'is_top_performer' not in df.columns:
        logger.warning(f"{window}_rf_transformed.csv missing is_top_performer column, calculating fallback")
        video_count = len(df)
        top_count = int(video_count * TOP_PERFORMER_PERCENTAGE)
        df['is_top_performer'] = [1] * top_count + [0] * (video_count - top_count)
    else:
        logger.debug(f"Using existing is_top_performer column from {window}_rf_transformed.csv")

    # Compute distribution stats for each top feature
    logger.debug(f"Computing distribution stats for top {len(top_features)} features...")

    for feature_data in top_features:
        feature_name = feature_data['feature']

        if feature_name not in df.columns:
            # Feature not in CSV (shouldn't happen for window-level)
            feature_data['top_performer_avg'] = None
            feature_data['bottom_performer_avg'] = None
            feature_data['gap'] = None
            feature_data['distribution'] = None
            logger.warning(f"Feature {feature_name} not in {window}_rf_transformed.csv - skipping")
            continue

        # Split by performance tier
        top_performers = df[df['is_top_performer'] == 1][feature_name]
        bottom_performers = df[df['is_top_performer'] == 0][feature_name]

        # Compute averages
        top_avg = float(top_performers.mean())
        bottom_avg = float(bottom_performers.mean())
        gap = abs(top_avg - bottom_avg)

        # Compute percentile thresholds (66th, 33rd) - SAME AS VIDEO-LEVEL RF
        high_threshold = float(top_performers.quantile(HIGH_PERCENTILE))
        low_threshold = float(top_performers.quantile(LOW_PERCENTILE))

        # Compute percentage distributions for TOP performers
        top_high_pct = (top_performers >= high_threshold).sum() / len(top_performers)
        top_med_pct = ((top_performers >= low_threshold) & (top_performers < high_threshold)).sum() / len(top_performers)
        top_low_pct = (top_performers < low_threshold).sum() / len(top_performers)

        # Compute percentage distributions for BOTTOM performers
        bottom_high_pct = (bottom_performers >= high_threshold).sum() / len(bottom_performers)
        bottom_med_pct = ((bottom_performers >= low_threshold) & (bottom_performers < high_threshold)).sum() / len(bottom_performers)
        bottom_low_pct = (bottom_performers < low_threshold).sum() / len(bottom_performers)

        # Add to feature data (CRITICAL: Include distribution field)
        feature_data['top_performer_avg'] = top_avg
        feature_data['bottom_performer_avg'] = bottom_avg
        feature_data['gap'] = gap
        feature_data['distribution'] = {
            'thresholds': {
                'high': high_threshold,
                'low': low_threshold
            },
            'top_performers': {
                'high_percentage': float(top_high_pct),
                'medium_percentage': float(top_med_pct),
                'low_percentage': float(top_low_pct)
            },
            'bottom_performers': {
                'high_percentage': float(bottom_high_pct),
                'medium_percentage': float(bottom_med_pct),
                'low_percentage': float(bottom_low_pct)
            }
        }

        logger.debug(
            f"Feature {feature_data['rank']}/{len(top_features)}: {feature_name} "
            f"(top_avg={top_avg:.4f}, bottom_avg={bottom_avg:.4f}, gap={gap:.4f}, "
            f"bimodal: {top_high_pct:.0%} high / {top_low_pct:.0%} low)"
        )
```

### **What This Does**
1. Calculates 66th and 33rd percentile thresholds from top performer distribution
2. Computes percentage of top performers with high/medium/low values
3. Computes percentage of bottom performers with high/medium/low values
4. Adds `distribution` field to feature JSON (matching video-level RF structure)

### **Expected Output After Fix**

Before (BROKEN):
```json
{
  "feature": "energy_variance",
  "importance": 0.09,
  "rank": 2,
  "top_performer_avg": 0.0020836716241754826,
  "bottom_performer_avg": 0.002377395944208026,
  "gap": 0.0002937243200325434
}
```

After (FIXED):
```json
{
  "feature": "energy_variance",
  "importance": 0.09,
  "rank": 2,
  "top_performer_avg": 0.0021,
  "bottom_performer_avg": 0.0024,
  "gap": 0.0003,
  "distribution": {
    "thresholds": {
      "high": 0.0028,
      "low": 0.0015
    },
    "top_performers": {
      "high_percentage": 0.28,
      "medium_percentage": 0.37,
      "low_percentage": 0.35
    },
    "bottom_performers": {
      "high_percentage": 0.42,
      "medium_percentage": 0.35,
      "low_percentage": 0.23
    }
  }
}
```

### **Validation**
After rerunning Stage 6 with this fix:
```bash
python3 -c "
import json
data = json.load(open('bucket_60-90s/ml_analysis/hook_rf_analysis.json'))
print('Has distribution:', 'distribution' in data['feature_importance'][0])
print('High %:', data['feature_importance'][0]['distribution']['top_performers']['high_percentage'])
print('Low %:', data['feature_importance'][0]['distribution']['top_performers']['low_percentage'])
"
```

Expected output:
```
Has distribution: True
High %: 0.28
Low %: 0.35
```

---

## Bug #2: Fix Number Formatting in Prompt Builder

### **Priority**: 🔴 P0 CRITICAL

### **Problem**
Using `.2f` formatting rounds very small values like `0.0021` to `0.00`, making them appear meaningless to the LLM.

### **Root Cause**
`ml_pipeline/stage7_llm_analysis/stage7_prompts.py:418-421` uses fixed 2-decimal formatting regardless of value magnitude.

### **Fix Location**
File: `ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
Lines: 418-421

### **Code Change**

**Step 1**: Add helper function at top of file (after imports, around line 30):

```python
def format_rf_value(value: float) -> str:
    """
    Format RF values with adaptive precision based on magnitude.

    Ensures small values (like 0.002) are displayed with enough precision
    to be meaningful, while large values don't show unnecessary decimals.

    Args:
        value: Numeric value to format

    Returns:
        Formatted string with appropriate precision

    Examples:
        >>> format_rf_value(0.0021)
        '0.0021'
        >>> format_rf_value(0.087)
        '0.087'
        >>> format_rf_value(2.456)
        '2.46'
    """
    abs_val = abs(value)

    if abs_val < 0.01:
        # Very small values: show 4 decimals (e.g., 0.0021)
        return f"{value:.4f}"
    elif abs_val < 0.1:
        # Small values: show 3 decimals (e.g., 0.087)
        return f"{value:.3f}"
    elif abs_val < 10:
        # Medium values: show 2 decimals (e.g., 2.46)
        return f"{value:.2f}"
    else:
        # Large values: show 1 decimal (e.g., 14.3)
        return f"{value:.1f}"
```

**Step 2**: Replace lines 418-421 with adaptive formatting:

```python
# OLD (WRONG):
prompt += f"   Top: avg {feature['top_performer_avg']:.2f} "
prompt += f"({bimodal['high_percentage']:.0%} high, {bimodal['low_percentage']:.0%} low) | "
prompt += f"Bottom: avg {feature['bottom_performer_avg']:.2f} | "
prompt += f"Gap: {feature['gap']:.2f} | Pattern: {pattern_label}\n"

# NEW (FIXED):
prompt += f"   Top: avg {format_rf_value(feature['top_performer_avg'])} "
prompt += f"({bimodal['high_percentage']:.0%} high, {bimodal['low_percentage']:.0%} low) | "
prompt += f"Bottom: avg {format_rf_value(feature['bottom_performer_avg'])} | "
prompt += f"Gap: {format_rf_value(feature['gap'])} | Pattern: {pattern_label}\n"
```

### **Expected Output After Fix**

Before (BROKEN):
```
2. energy_variance - RF Importance: 0.10 (rank #2)
   Top: avg 0.00 (28% high, 35% low) | Bottom: avg 0.00 | Gap: 0.00 | Pattern: UNIMODAL
```

After (FIXED):
```
2. energy_variance - RF Importance: 0.0907 (rank #2)
   Top: avg 0.0021 (28% high, 35% low) | Bottom: avg 0.0024 | Gap: 0.0003 | Pattern: UNIMODAL
```

### **Additional Formatting Fixes**

Also apply adaptive formatting to importance display (line 417):

```python
# OLD:
prompt += f"{i}. {feature['feature']} - RF Importance: {feature['importance']:.2f} (rank #{i})\n"

# NEW:
prompt += f"{i}. {feature['feature']} - RF Importance: {format_rf_value(feature['importance'])} (rank #{i})\n"
```

### **Validation**
After applying fix, regenerate Phase 1 prompts and verify:
```bash
# Check that small values show proper precision
grep "energy_variance" /tmp/phase1_contrastive_prompt.txt
# Should show: "Top: avg 0.0021 ... Gap: 0.0003"
# NOT: "Top: avg 0.00 ... Gap: 0.00"
```

---

## Bug #3: Fix RF Alignment Tolerance

### **Priority**: 🟡 P1 HIGH

### **Problem**
RF alignment returns 0 matches for all clusters because no features meet the 0.15 importance threshold. In the 60-90s bucket, highest importance is 0.127.

### **Root Cause**
`ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py:226-230` uses hardcoded `tolerance=0.15`, which is too strict for buckets with low RF model performance.

### **Evidence**
```python
# From actual data:
RF features with importance >= 0.15: Count = 0
Highest importance = 0.127 (pitch_scatter_ratio)
```

### **Fix Location**
File: `ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py`
Function: `compute_rf_alignment()` (line 191)

### **Code Change**

**Option A: Lower the threshold (Quick Fix)**

Change line 212 from:
```python
def compute_rf_alignment(cluster_features: List[str],
                        rf_features: List[dict],
                        tolerance: float = 0.15) -> dict:
```

To:
```python
def compute_rf_alignment(cluster_features: List[str],
                        rf_features: List[dict],
                        tolerance: float = 0.10) -> dict:
```

**Option B: Adaptive threshold based on data (Better Fix)**

Replace lines 226-230 with:

```python
    # Adaptive tolerance: use provided tolerance OR top 5 features (whichever is more lenient)
    # This ensures we always have SOME RF features to match against
    if not rf_features:
        return {
            'alignment_score': 0.0,
            'matched_features': [],
            'top_rf_features': [],
            'alignment_ratio': '0/0',
            'insight': "No RF features available"
        }

    # Get top 5 features by importance
    sorted_rf = sorted(rf_features, key=lambda x: x.get('importance', 0), reverse=True)[:5]
    min_importance_top5 = sorted_rf[-1].get('importance', 0) if sorted_rf else 0

    # Use the more lenient threshold: either provided tolerance OR top-5 minimum
    effective_tolerance = min(tolerance, min_importance_top5)

    logger.debug(f"RF alignment: requested tolerance={tolerance:.3f}, top-5 min={min_importance_top5:.3f}, using={effective_tolerance:.3f}")

    # Filter RF features by effective tolerance
    top_rf_features = [
        rf['feature'] for rf in rf_features
        if rf.get('importance', 0) >= effective_tolerance
    ]
```

### **Rationale**
- **Option A** works for this specific bucket but might fail for others
- **Option B** adapts to data quality - always includes at least top 5 RF features
- Bucket with poor RF model (importance all <0.15) still gets meaningful alignment checks

### **Expected Output After Fix**

Before (BROKEN):
```
RF Alignment (features matching top performer patterns):
  ❌ No features align with RF top patterns (creative novelty - not a bug!)
```

After (FIXED with Option B):
```
RF Alignment (features matching top performer patterns):
  ✅ pitch_scatter_ratio (RF rank #1, importance 0.127)
  ✅ energy_variance (RF rank #2, importance 0.091)
  Alignment: 2 of 5 top RF features present in this cluster
```

### **Validation**
After fix, check that at least ONE cluster shows aligned features:
```bash
python3 -c "
import json
from ml_pipeline.stage7_llm_analysis.stage7_prompts import build_phase1_prompt
# ... load data and generate prompt ...
# Check prompt contains '✅' markers in RF Alignment section
"
```

---

## Bug #4: Fix Universal Principles Formatting

### **Priority**: 🟡 P1 HIGH

### **Problem**
Universal principles show identical strategies for top vs bottom performers:
```
1. Moderate hold in middle: Top performers use moderate hold vs bottom use moderate hold
```

### **Root Cause**
Two possible causes:
1. Number formatting (Bug #2) rounds different raw values to same semantic label
2. `generate_universal_principles()` uses too coarse-grained semantic interpretation

### **Fix Location**
File: `ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py`
Function: `generate_universal_principles()` (around line 500)

### **Investigation Needed**
Before applying fix, check the root cause:

```python
# Load video-level RF
with open('bucket_60-90s/ml_analysis/rf_video_analysis.json') as f:
    rf_video = json.load(f)

# Check top features
for feat in rf_video['feature_importance'][:7]:
    print(f"{feat['feature']}:")
    print(f"  Top: {feat['top_performer_avg']:.6f}")
    print(f"  Bottom: {feat['bottom_performer_avg']:.6f}")
    print(f"  Gap: {feat['gap']:.6f}")
```

### **Code Change**

If gaps are truly tiny (< 0.01), modify `generate_universal_principles()` to filter them out:

```python
def generate_universal_principles(rf_video_data: dict, top_n: int = 7) -> List[str]:
    """
    Extract universal principles from video-level RF analysis.

    Returns insights about features that predict success across all cluster paths.
    """
    if not rf_video_data or 'feature_importance' not in rf_video_data:
        return ["Universal principles not available (TOP mode - RF not trained)"]

    principles = []
    feature_importance = rf_video_data['feature_importance'][:top_n]

    for feature_data in feature_importance:
        feature_name = feature_data['feature']
        top_avg = feature_data.get('top_performer_avg')
        bottom_avg = feature_data.get('bottom_performer_avg')
        gap = feature_data.get('gap', 0)

        # CRITICAL FIX: Skip features with meaningless gaps
        if gap < 0.01:
            logger.debug(f"Skipping universal principle for {feature_name}: gap too small ({gap:.4f})")
            continue

        # Extract base feature name (remove window prefixes)
        base_feature = extract_base_feature(feature_name)

        # Interpret values using semantic ranges
        top_label, top_desc = interpret_value(base_feature, top_avg)
        bottom_label, bottom_desc = interpret_value(base_feature, bottom_avg)

        # ADDITIONAL FIX: Skip if semantic labels are identical
        if top_label == bottom_label:
            logger.debug(f"Skipping universal principle for {feature_name}: same label ({top_label})")
            continue

        # Format principle with gap magnitude
        principle = (
            f"{base_feature} contrast: "
            f"Top performers use {top_label} vs bottom use {bottom_label} "
            f"(gap: {format_rf_value(gap)})"
        )
        principles.append(principle)

    # If no meaningful principles found, return fallback
    if not principles:
        return ["No universal principles with meaningful contrast found (top/bottom performers very similar)"]

    return principles[:top_n]
```

### **Expected Output After Fix**

Before (BROKEN):
```
1. Moderate hold in middle: Top performers use moderate hold vs bottom use moderate hold
2. Very consistent in opening: Top performers use very consistent vs bottom use very consistent
```

After (FIXED):
```
1. pitch_scatter_ratio contrast: Top performers use higher variation vs bottom use lower variation (gap: 0.117)
2. overlay_unique_count contrast: Top performers use minimal text vs bottom use moderate text (gap: 0.575)
```

---

## Bug #5: Improve Semantic Interpretation Robustness

### **Priority**: 🟡 P2 MEDIUM

### **Problem**
Some features display as `"out_of_range - value: -0.00"` in the prompt, indicating values outside defined ranges.

### **Root Cause**
Either:
1. Denormalization producing unexpected values (negative where shouldn't be)
2. Semantic interpretation ranges missing edge cases
3. Normalized values (0-1) being passed to interpret_value expecting raw values

### **Fix Location**
File: `config/semantic_interpretations.py`
Function: `interpret_value()` (line 421)

### **Code Change**

Make `interpret_value()` more robust to edge cases:

```python
def interpret_value(feature: str, value: float) -> tuple[str, str]:
    """
    Convert numeric value to semantic label and description.

    Args:
        feature: Base feature name (without window prefix, e.g., 'average_face_size')
        value: Numeric value to interpret

    Returns:
        tuple[str, str]: (label, description)
    """
    if feature not in SEMANTIC_INTERPRETATIONS:
        # Feature not yet defined, return placeholder
        return ('unknown feature', f'{feature}={value:.3f}')

    interp = SEMANTIC_INTERPRETATIONS[feature]

    # Handle special values
    if np.isnan(value):
        return ('no data', 'value not available')

    # Handle normalized values (0-1 range) - common mistake
    if 0 <= value <= 1 and interp.get('data_range', (0, 1))[1] > 1:
        logger.warning(
            f"Feature '{feature}' received normalized value {value:.3f} "
            f"but expects raw range {interp['data_range']}. "
            f"May need denormalization."
        )

    # Find matching range
    for min_val, max_val, label, description in interp['ranges']:
        if min_val <= value < max_val:
            return (label, description)

    # Edge case: value at max boundary (handle inclusive upper bound)
    last_range = interp['ranges'][-1]
    if value >= last_range[1]:
        return (last_range[2], last_range[3])

    # Fallback: value outside all ranges
    data_range = interp.get('data_range', 'unknown')
    return (
        'out_of_range',
        f"value {value:.3f} outside expected range {data_range}"
    )
```

### **Additional Investigation**

Check if denormalization is working correctly:

```python
# In stage7_prompts.py around line 445
centroid_normalized = cluster_data['centroid']
centroid_raw = denormalize_centroid(centroid_normalized, scalers)

# Add validation:
for feat, raw_val in centroid_raw.items():
    norm_val = centroid_normalized.get(feat, None)
    if norm_val is not None and not (0 <= norm_val <= 1):
        logger.warning(f"Feature {feat}: normalized value {norm_val} not in [0,1] range")
    if raw_val < 0 and feat not in ['emotional_valence', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']:
        logger.warning(f"Feature {feat}: denormalized to negative value {raw_val}")
```

---

## Testing & Validation

### **Test Plan**

After implementing fixes #1 and #2, run Stage 6 and Stage 7:

```bash
# Step 1: Regenerate Stage 6 JSONs with Bug #1 fix
cd /home/jorge/rumiaifinal
python3 ml_pipeline/stage6_analysis/ml_analysis_generation.py \
  --client rollo_test5 \
  --bucket 60-90s \
  --analysis-base data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive

# Step 2: Validate window RF has distribution field
python3 << 'EOF'
import json
rf_path = 'data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/buckets/bucket_60-90s/ml_analysis/hook_rf_analysis.json'
with open(rf_path) as f:
    data = json.load(f)

feat = data['feature_importance'][0]
assert 'distribution' in feat, "Missing distribution field!"
assert 'top_performers' in feat['distribution'], "Missing top_performers!"
assert feat['distribution']['top_performers']['high_percentage'] > 0, "high_percentage is 0!"

print("✓ Bug #1 FIXED: Window RF has distribution data")
print(f"  Feature: {feat['feature']}")
print(f"  High %: {feat['distribution']['top_performers']['high_percentage']:.1%}")
print(f"  Low %: {feat['distribution']['top_performers']['low_percentage']:.1%}")
EOF

# Step 3: Regenerate Stage 7 Phase 1 prompts with Bug #2 fix
python3 ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py \
  --client rollo_test5 \
  --hashtag wellnesspt2_test5 \
  --mode contrastive \
  --bucket 60-90s

# Step 4: Check prompt formatting
grep "energy_variance" data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/buckets/bucket_60-90s/ml_analysis/llm/.debug/hook_prompt.txt
# Should show: "Top: avg 0.0021 ... Gap: 0.0003"
# NOT: "Top: avg 0.00 ... Gap: 0.00"

# Step 5: Verify LLM returns all 3 clusters
python3 << 'EOF'
import json
hook_analysis = 'data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/buckets/bucket_60-90s/ml_analysis/llm/hook_analysis.json'
with open(hook_analysis) as f:
    data = json.load(f)

assert len(data['clusters']) == 3, f"Expected 3 clusters, got {len(data['clusters'])}!"
print("✓ Bug RESOLVED: LLM returned all 3 clusters")
for cluster in data['clusters']:
    print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} videos - {cluster['name']}")
EOF
```

### **Success Criteria**

- [ ] Window-level RF JSONs have `distribution` field
- [ ] Bimodal detection shows non-zero percentages (e.g., 28% high, 35% low)
- [ ] Prompt displays small values correctly (0.0021 not 0.00)
- [ ] LLM returns exactly 3 clusters for ALL windows
- [ ] RF alignment shows at least 1 match for at least 1 cluster
- [ ] Universal principles show different strategies for top vs bottom

---

## Rollback Plan

If fixes cause Stage 6 or Stage 7 to fail:

1. **Revert Stage 6 changes**:
   ```bash
   cd /home/jorge/rumiaifinal/ml_pipeline/stage6_analysis
   git diff ml_analysis_generation.py  # Review changes
   git checkout ml_analysis_generation.py  # Revert if needed
   ```

2. **Revert Stage 7 changes**:
   ```bash
   cd /home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis
   git checkout stage7_prompts.py stage7_preprocessing.py
   ```

3. **Re-run from last known good state**:
   - Stage 5 outputs are unaffected
   - Re-run Stage 6 with original code
   - Re-run Stage 7 with original code

---

## Implementation Order

1. **First**: Fix Bug #1 (Stage 6 distribution) - CRITICAL, unblocks bimodal detection
2. **Second**: Fix Bug #2 (formatting) - CRITICAL, makes data readable
3. **Third**: Test end-to-end - Verify LLM returns 3 clusters
4. **Fourth**: Fix Bug #3 (RF alignment) - HIGH, improves prompt quality
5. **Fifth**: Fix Bug #4 (universal principles) - MEDIUM, improves Phase 2
6. **Last**: Fix Bug #5 (semantic ranges) - LOW, minor robustness improvement

---

## Timeline Estimate

- **Bug #1 implementation**: 30 minutes (code + test)
- **Bug #2 implementation**: 15 minutes (code + test)
- **Full validation run**: 60 minutes (re-run Stage 6 + Stage 7 for all windows)
- **Bug #3 implementation**: 20 minutes
- **Bug #4 implementation**: 30 minutes (requires investigation)
- **Bug #5 implementation**: 20 minutes

**Total**: ~3 hours for P0 fixes + validation

---

## Notes

- All fixes are **backward compatible** - won't break existing TOP mode
- Fixes improve **data quality**, not prompts - prompts were fine
- After fixes, consider re-running **all production buckets** to regenerate with proper data
- Monitor LLM token costs - better data quality may increase prompt size slightly

---

**Last Updated**: 2025-11-02
**Implementation Status**: Ready for review and deployment
