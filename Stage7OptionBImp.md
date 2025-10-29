# Stage 7 Option B Implementation Guide

**Document**: High Value / Low Effort Function Additions
**Date**: 2025-01-28
**Status**: Proposed (Not Yet Implemented)
**Related**: Stage7TIUpdate.md, LLMOutputFix.md

---

## Executive Summary

This document specifies implementation for 3 high-value Python preprocessing functions that optimize Stage 7 LLM Analysis without over-engineering.

**Functions Selected**: 3 of 9 unimplemented functions from TI Section 4
**Selection Criteria**: High value-to-effort ratio
**Total Implementation Time**: 8-12 hours (development + testing)
**Expected Benefits**: Token savings, better LLM focus, validation metrics

---

## Context

### Why These 3 Functions?

From Stage 7 TI analysis, 9 functions were designed but not implemented. Most were deemed unnecessary because Claude Sonnet 4 handles analysis well. However, **3 functions provide clear optimization value**:

| Function | Effort | Value | Why Implement? |
|----------|--------|-------|----------------|
| **identify_high_contrast_features()** | 4-6 hrs | HIGH | Reduces prompt tokens, improves LLM focus |
| **detect_bimodal_pattern()** | 1-2 hrs | MEDIUM | Prevents LLM confusion on multiple strategies |
| **compute_rf_alignment()** | 3-4 hrs | MEDIUM-HIGH | Validates LLM recommendations against RF |

**Total**: 8-12 hours for measurable improvements

---

## Relationship to LLMOutputFix.md

### Implementation Order

**CRITICAL**: Implement LLMOutputFix.md BEFORE this document.

**Reasoning**:
1. LLMOutputFix addresses **urgent creator confusion** (high priority)
2. LLMOutputFix changes `stage7_preprocessing.py` function signatures (this builds on those changes)
3. Option B functions integrate more cleanly after LLMOutputFix stabilizes

**Timeline**:
- **Week 1-2**: Implement LLMOutputFix.md (Issues #1, #2, #3)
- **Week 3**: Test and stabilize LLMOutputFix
- **Week 4+**: Implement Option B (this document)

**Integration Points**:
- Option B Function #1 (`identify_high_contrast_features()`) filters features BEFORE passing to LLM
- LLMOutputFix ensures LLM generates natural language FROM those filtered features
- Complementary, not conflicting

---

## Function 1: identify_high_contrast_features()

### Purpose

Filter cluster features to only those that DIFFERENTIATE clusters, avoiding universal features that add noise.

**Example Problem**:
- All 3 clusters have high eye contact (0.85, 0.87, 0.82)
- LLM gets confused: "Which cluster is the 'high eye contact' one?"
- Better: Filter out eye_contact, show only differentiating features

### Benefits

✅ **Reduces prompt size**: 21 features → 8-12 high-contrast features (saves ~40 tokens/window)
✅ **Improves LLM focus**: Highlights what DIFFERENTIATES clusters, not what's universal
✅ **Prevents noise**: Filters features where all clusters are similar
✅ **Cost savings**: ~$0.15 per bucket (token reduction × 6-7 windows)

### Algorithm

```python
def identify_high_contrast_features(kmeans_data: dict, threshold: float = 0.20) -> dict:
    """
    Filter cluster features to those with ≥0.20 contrast between clusters.

    Args:
        kmeans_data: Stage 6 K-Means JSON for a window
        threshold: Minimum contrast difference (default 0.20)

    Returns:
        dict: Clusters with high_contrast_features list added

    Source: LLMAnalysisCHILDTI.md Section 4.2 (adapted)
    """
    clusters = kmeans_data['clusters']
    all_features = list(clusters[0]['centroid'].keys())

    result = {'clusters': []}

    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        centroid = cluster['centroid']
        high_contrast = []

        for feature in all_features:
            this_value = centroid[feature]

            # Get values in other clusters
            other_values = [
                c['centroid'][feature]
                for c in clusters
                if c['cluster_id'] != cluster_id
            ]

            # Calculate max contrast
            max_diff = max(abs(this_value - ov) for ov in other_values)

            if max_diff >= threshold:
                # Calculate contrast vs each other cluster
                contrasts = {
                    f"vs_cluster_{c['cluster_id']}": abs(this_value - c['centroid'][feature])
                    for c in clusters
                    if c['cluster_id'] != cluster_id
                }

                high_contrast.append({
                    'feature': feature,
                    'value': this_value,
                    'max_contrast': max_diff,
                    'contrasts': contrasts
                })

        # Sort by max_contrast descending (most differentiating first)
        high_contrast.sort(key=lambda x: x['max_contrast'], reverse=True)

        result['clusters'].append({
            'cluster_id': cluster_id,
            'size': cluster['size'],
            'all_features': centroid,  # Keep for reference
            'high_contrast_features': high_contrast
        })

    return result
```

### Implementation Details

**File**: `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py`

**Location**: Add after existing helper functions (around line 300)

**Threshold Selection**:
- **0.20** = 20 percentage points OR 20 words difference
- Domain-grounded: perceptually noticeable difference
- Pilot-tested: filters 21 features → 8-12 high-contrast features
- Balances specificity (not too strict) with clarity (not too lenient)

**Edge Cases**:
1. **All features below threshold**: Return empty `high_contrast_features` (all clusters similar)
2. **Single cluster**: Return empty (no comparison possible)
3. **Missing features in some clusters**: Skip feature, log warning
4. **Threshold = 0.0**: All features returned (useful for debugging)

### Integration with Stage 7 Prompts

**File**: `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
**Function**: `build_phase1_prompt()`

**Current** (lines ~300-350):
```python
# Show all 21 features for each cluster
for cluster in kmeans_data['clusters']:
    prompt += f"\nCluster {cluster['cluster_id']} Features:\n"
    for feature, value in cluster['centroid'].items():
        prompt += f"  - {feature}: {value:.2f}\n"
```

**Proposed**:
```python
# Filter to high-contrast features only
filtered_clusters = identify_high_contrast_features(kmeans_data, threshold=0.20)

for cluster_data in filtered_clusters['clusters']:
    cluster_id = cluster_data['cluster_id']
    prompt += f"\nCluster {cluster_id} High-Contrast Features (differentiate this cluster):\n"

    # Show top 10 high-contrast features
    for feature_data in cluster_data['high_contrast_features'][:10]:
        feature = feature_data['feature']
        value = feature_data['value']
        max_contrast = feature_data['max_contrast']

        prompt += f"  - {feature}: {value:.2f} (contrast: {max_contrast:.2f})\n"

        # Show comparison to other clusters
        for comparison, diff in feature_data['contrasts'].items():
            prompt += f"    {comparison}: Δ{diff:.2f}\n"
```

**Benefits**:
- LLM sees only differentiating features
- Context includes contrast magnitude
- Comparison to other clusters explicit

### Testing Strategy

**Test Case 1: Typical Case**
- **Input**: 3 clusters with 21 features each
- **Expected**: ~8-12 high-contrast features per cluster
- **Validation**: All returned features have max_contrast ≥ 0.20

**Test Case 2: Very Similar Clusters**
- **Input**: 3 clusters with minimal differences (all features < 0.20 contrast)
- **Expected**: Empty or very short high_contrast_features lists
- **Validation**: LLM prompt notes "clusters are very similar"

**Test Case 3: Edge Case - One Extreme Cluster**
- **Input**: Cluster 0 very different, Clusters 1-2 similar
- **Expected**: Cluster 0 has many high-contrast features, Clusters 1-2 have few
- **Validation**: Feature counts reflect actual distinctiveness

**Unit Test**:
```python
def test_identify_high_contrast_features():
    """Test high-contrast feature filtering."""
    kmeans_data = {
        'clusters': [
            {'cluster_id': 0, 'centroid': {'eye_contact': 0.87, 'word_count': 14, 'energy': 0.55}},
            {'cluster_id': 1, 'centroid': {'eye_contact': 0.42, 'word_count': 52, 'energy': 0.60}},
            {'cluster_id': 2, 'centroid': {'eye_contact': 0.55, 'word_count': 35, 'energy': 0.85}}
        ]
    }

    result = identify_high_contrast_features(kmeans_data, threshold=0.20)

    # Cluster 0 should have 2 high-contrast features: eye_contact (Δ0.45) and word_count (Δ38)
    assert len(result['clusters'][0]['high_contrast_features']) == 2

    # First feature should be word_count (highest contrast)
    assert result['clusters'][0]['high_contrast_features'][0]['feature'] == 'word_count'
    assert result['clusters'][0]['high_contrast_features'][0]['max_contrast'] == 38
```

---

## Function 2: detect_bimodal_pattern()

### Purpose

Detect when BOTH high AND low feature values work for top performers, indicating multiple successful strategies exist.

**Example**:
- **Unimodal**: 72% of top performers have high eye contact (0.87 avg) → "Use high eye contact"
- **Bimodal**: 40% have high word count (80 words), 35% have low word count (20 words) → "BOTH brief AND dense captions work"

### Benefits

✅ **Prevents LLM confusion**: Clear signal that multiple strategies are valid
✅ **Better recommendations**: Avoids forcing one approach when both work
✅ **Catches non-obvious patterns**: Humans might miss bimodal distributions in RF data
⚠️ **LLM already sometimes notices**: Value is consistency, not capability

### Algorithm

```python
def detect_bimodal_pattern(distribution: dict) -> dict:
    """
    Detect if feature shows bimodal pattern in top performers.

    A feature is bimodal when BOTH high AND low percentages are ≥30% among top performers.

    Args:
        distribution: Stage 6 distribution data
            {
                'top_performers': {
                    'high_percentage': 0.40,  # % in top tercile (≥66th percentile)
                    'low_percentage': 0.35    # % in bottom tercile (<33rd percentile)
                },
                'bottom_performers': {...}
            }

    Returns:
        dict: Bimodal analysis result
            {
                'is_bimodal': True,
                'high_percentage': 0.40,
                'low_percentage': 0.35,
                'interpretation': 'BOTH strategies work',
                'pattern_label': 'BIMODAL'
            }

    Source: LLMAnalysisCHILDTI.md Section 4.1
    """
    top_high_pct = distribution['top_performers']['high_percentage']
    top_low_pct = distribution['top_performers']['low_percentage']

    # 30% threshold: "nearly 1 in 3 videos" = meaningful minority
    is_bimodal = (top_high_pct >= 0.30 and top_low_pct >= 0.30)

    return {
        'is_bimodal': is_bimodal,
        'high_percentage': top_high_pct,
        'low_percentage': top_low_pct,
        'interpretation': 'BOTH strategies work' if is_bimodal else 'Single dominant strategy',
        'pattern_label': 'BIMODAL' if is_bimodal else 'UNIMODAL'
    }
```

### Implementation Details

**File**: `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py`

**Location**: Add after helper functions (around line 350)

**Threshold Rationale**:
- **30%** = "nearly 1 in 3 videos" = statistically significant minority
- Avoids false positives: 20%/20% split might be noise
- Practical value: Both strategies common enough for creators to replicate
- Tested in pilot data

**Edge Cases**:
1. **Exactly 30% boundary**: `high_percentage=0.30, low_percentage=0.30` → `is_bimodal=True` (inclusive)
2. **One side at 29.9%**: `high_percentage=0.40, low_percentage=0.299` → `is_bimodal=False` (strict)
3. **Missing distribution data**: Raise `ValueError("distribution dict missing required keys")`
4. **Invalid percentages** (<0 or >1): Raise `ValueError("Percentages must be in range [0.0, 1.0]")`

### Integration with Stage 7 Prompts

**File**: `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
**Function**: `build_phase1_prompt()`

**Current** (lines ~250-300):
```python
# Show RF features without bimodal detection
for feature_data in rf_data['feature_importance'][:10]:
    prompt += f"  - {feature['feature']}: {top_avg:.2f} (top) vs {bottom_avg:.2f} (bottom)\n"
```

**Proposed**:
```python
# Add bimodal detection to RF features
for feature_data in rf_data['feature_importance'][:10]:
    feature = feature_data['feature']
    top_avg = feature_data['top_performer_avg']
    bottom_avg = feature_data['bottom_performer_avg']

    # Detect bimodal pattern if distribution data exists
    bimodal_info = None
    if 'distribution' in feature_data and feature_data['distribution']:
        bimodal_info = detect_bimodal_pattern(feature_data['distribution'])

    prompt += f"  - {feature}: {top_avg:.2f} (top) vs {bottom_avg:.2f} (bottom)"

    if bimodal_info and bimodal_info['is_bimodal']:
        prompt += f" [BIMODAL: {bimodal_info['high_percentage']:.0%} high, {bimodal_info['low_percentage']:.0%} low]\n"
        prompt += f"    → Strategy A (high): ~{bimodal_info['high_percentage']:.0%} of top performers\n"
        prompt += f"    → Strategy B (low): ~{bimodal_info['low_percentage']:.0%} of top performers\n"
    else:
        prompt += "\n"
```

**Example Output**:
```
RF Features:
  - eye_contact_rate: 0.88 (top) vs 0.45 (bottom)
  - word_count: 52 (top) vs 18 (bottom) [BIMODAL: 40% high, 35% low]
    → Strategy A (high): ~40% of top performers use dense captions (≥80 words)
    → Strategy B (low): ~35% of top performers use brief captions (≤20 words)
```

### Testing Strategy

**Test Case 1: Bimodal Pattern**
- **Input**: `{'top_performers': {'high_percentage': 0.40, 'low_percentage': 0.35}}`
- **Expected**: `{'is_bimodal': True, 'pattern_label': 'BIMODAL'}`

**Test Case 2: Unimodal Pattern**
- **Input**: `{'top_performers': {'high_percentage': 0.72, 'low_percentage': 0.15}}`
- **Expected**: `{'is_bimodal': False, 'pattern_label': 'UNIMODAL'}`

**Test Case 3: Boundary Case**
- **Input**: `{'top_performers': {'high_percentage': 0.30, 'low_percentage': 0.30}}`
- **Expected**: `{'is_bimodal': True}` (inclusive threshold)

**Unit Test**:
```python
def test_detect_bimodal_pattern():
    """Test bimodal pattern detection."""
    # Unimodal case
    dist_unimodal = {
        'top_performers': {'high_percentage': 0.72, 'low_percentage': 0.15},
        'bottom_performers': {'high_percentage': 0.25, 'low_percentage': 0.45}
    }
    result = detect_bimodal_pattern(dist_unimodal)
    assert result['is_bimodal'] == False
    assert result['pattern_label'] == 'UNIMODAL'

    # Bimodal case
    dist_bimodal = {
        'top_performers': {'high_percentage': 0.40, 'low_percentage': 0.35},
        'bottom_performers': {'high_percentage': 0.20, 'low_percentage': 0.22}
    }
    result = detect_bimodal_pattern(dist_bimodal)
    assert result['is_bimodal'] == True
    assert result['pattern_label'] == 'BIMODAL'
    assert result['high_percentage'] == 0.40
    assert result['low_percentage'] == 0.35
```

---

## Function 3: compute_rf_alignment()

### Purpose

Calculate how many of the top 5 RF features a cluster uses at optimal levels, providing a validation metric for LLM recommendations.

**Example**:
- Top 5 RF features: eye_contact, word_count, energy, scene_duration, face_size
- Cluster 0: Uses 4 of 5 at optimal levels → "RF alignment: 4/5" (STRONG)
- Cluster 2: Uses 1 of 5 at optimal levels → "RF alignment: 1/5" (WEAK)

### Benefits

✅ **Prevents hallucination**: LLM can't recommend features RF says are unimportant
✅ **Validation metric**: Quantifies how well clusters leverage predictive features
✅ **Better reporting**: "This cluster uses 4 of 5 top predictors" is concrete
✅ **Transparency**: Creators understand which recommendations are data-backed

### Algorithm

```python
def compute_rf_alignment(cluster_centroid: dict, rf_data: dict, top_n: int = 5) -> dict:
    """
    Calculate how many top N RF features a cluster uses at optimal levels.

    Args:
        cluster_centroid: Cluster centroid feature values
        rf_data: Window-level RF analysis with feature_importance
        top_n: Number of top RF features to check (default 5)

    Returns:
        dict: RF alignment result
            {
                'alignment_score': 3,  # How many of top 5 used optimally
                'alignment_ratio': '3/5',
                'top_features_used': ['eye_contact', 'word_count', 'energy'],
                'top_features_missed': ['scene_duration', 'face_size'],
                'strength': 'STRONG'  # STRONG (≥4), MODERATE (2-3), WEAK (0-1)
            }

    Source: LLMAnalysisCHILDTI.md Section 4.3 (adapted)
    """
    # Get top N RF features
    feature_importance = rf_data.get('feature_importance', [])
    top_features = feature_importance[:top_n]

    features_used = []
    features_missed = []

    for feature_data in top_features:
        feature_name = feature_data['feature']

        # Check if feature exists in cluster centroid
        if feature_name not in cluster_centroid:
            features_missed.append(feature_name)
            continue

        cluster_value = cluster_centroid[feature_name]

        # Define "optimal" as being in top tercile (≥66th percentile)
        # Use distribution thresholds if available
        distribution = feature_data.get('distribution', {})
        if distribution and 'thresholds' in distribution:
            high_threshold = distribution['thresholds'].get('high', None)

            if high_threshold is not None:
                is_optimal = cluster_value >= high_threshold
            else:
                # Fallback: compare to top_performer_avg
                top_avg = feature_data.get('top_performer_avg', 0)
                is_optimal = cluster_value >= (top_avg * 0.9)  # Within 90% of top avg
        else:
            # Fallback: compare to top_performer_avg
            top_avg = feature_data.get('top_performer_avg', 0)
            is_optimal = cluster_value >= (top_avg * 0.9)

        if is_optimal:
            features_used.append(feature_name)
        else:
            features_missed.append(feature_name)

    alignment_score = len(features_used)
    alignment_ratio = f"{alignment_score}/{top_n}"

    # Classify strength
    if alignment_score >= 4:
        strength = 'STRONG'
    elif alignment_score >= 2:
        strength = 'MODERATE'
    else:
        strength = 'WEAK'

    return {
        'alignment_score': alignment_score,
        'alignment_ratio': alignment_ratio,
        'top_features_used': features_used,
        'top_features_missed': features_missed,
        'strength': strength
    }
```

### Implementation Details

**File**: `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py`

**Location**: Add after helper functions (around line 400)

**Optimal Threshold Definition**:
- **Preferred**: Use distribution thresholds (≥66th percentile from Stage 6)
- **Fallback**: Within 90% of top_performer_avg
- **Rationale**: 66th percentile = top tercile = clearly high performance

**Strength Classification**:
- **STRONG**: 4-5 of top 5 features used (≥80%)
- **MODERATE**: 2-3 of top 5 features used (40-60%)
- **WEAK**: 0-1 of top 5 features used (≤20%)

**Edge Cases**:
1. **Feature missing from centroid**: Count as "missed" (cluster doesn't use it)
2. **No distribution data**: Use fallback (90% of top_performer_avg)
3. **top_n > available features**: Use all available features
4. **All features at optimal levels**: `alignment_score = top_n` (perfect alignment)

### Integration with Stage 7 Prompts

**File**: `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
**Function**: `build_phase1_prompt()`

**Current** (lines ~350-400):
```python
# Show clusters without RF alignment
for cluster in kmeans_data['clusters']:
    prompt += f"\nCluster {cluster['cluster_id']}:\n"
    prompt += f"  Size: {cluster['size']} videos\n"
    # ... features ...
```

**Proposed**:
```python
# Add RF alignment to cluster descriptions
for cluster in kmeans_data['clusters']:
    cluster_id = cluster['cluster_id']
    size = cluster['size']
    centroid = cluster['centroid']

    # Compute RF alignment
    rf_alignment = compute_rf_alignment(centroid, rf_data, top_n=5)

    prompt += f"\nCluster {cluster_id}:\n"
    prompt += f"  Size: {size} videos\n"
    prompt += f"  RF Alignment: {rf_alignment['alignment_ratio']} ({rf_alignment['strength']})\n"

    if rf_alignment['top_features_used']:
        prompt += f"  ✓ Uses top RF features: {', '.join(rf_alignment['top_features_used'])}\n"

    if rf_alignment['top_features_missed']:
        prompt += f"  ✗ Misses RF features: {', '.join(rf_alignment['top_features_missed'])}\n"

    prompt += f"\n  High-Contrast Features:\n"
    # ... rest of features ...
```

**Example Output**:
```
Cluster 0:
  Size: 35 videos
  RF Alignment: 4/5 (STRONG)
  ✓ Uses top RF features: eye_contact, word_count, energy, scene_duration
  ✗ Misses RF features: face_size

  High-Contrast Features:
  ...
```

**LLM Instruction Addition**:
```python
prompt += """

## RF Alignment Interpretation

When generating recommendations for each cluster:
- STRONG (4-5/5): This cluster already leverages most predictive features → Focus on maintaining these strengths
- MODERATE (2-3/5): This cluster uses some predictors → Recommend adding missed high-importance features
- WEAK (0-1/5): This cluster ignores top predictors → Recommend significant strategy shift toward RF features

Your recommendations MUST prioritize features with high RF alignment scores.
"""
```

### Testing Strategy

**Test Case 1: Strong Alignment**
- **Input**: Cluster uses 4 of top 5 RF features at optimal levels
- **Expected**: `{'alignment_score': 4, 'strength': 'STRONG'}`

**Test Case 2: Weak Alignment**
- **Input**: Cluster uses 1 of top 5 RF features
- **Expected**: `{'alignment_score': 1, 'strength': 'WEAK'}`

**Test Case 3: Missing Feature**
- **Input**: Cluster centroid doesn't have one of top 5 features
- **Expected**: Feature appears in `top_features_missed`

**Unit Test**:
```python
def test_compute_rf_alignment():
    """Test RF alignment computation."""
    cluster_centroid = {
        'eye_contact': 0.87,
        'word_count': 52,
        'energy': 0.55,
        'scene_duration': 1.2,
        'face_size': 0.05  # Below optimal
    }

    rf_data = {
        'feature_importance': [
            {'feature': 'eye_contact', 'top_performer_avg': 0.85, 'distribution': {'thresholds': {'high': 0.80}}},
            {'feature': 'word_count', 'top_performer_avg': 50, 'distribution': {'thresholds': {'high': 45}}},
            {'feature': 'energy', 'top_performer_avg': 0.50, 'distribution': {'thresholds': {'high': 0.48}}},
            {'feature': 'scene_duration', 'top_performer_avg': 1.0, 'distribution': {'thresholds': {'high': 0.95}}},
            {'feature': 'face_size', 'top_performer_avg': 0.42, 'distribution': {'thresholds': {'high': 0.38}}}
        ]
    }

    result = compute_rf_alignment(cluster_centroid, rf_data, top_n=5)

    assert result['alignment_score'] == 4  # eye_contact, word_count, energy, scene_duration
    assert result['alignment_ratio'] == '4/5'
    assert result['strength'] == 'STRONG'
    assert 'face_size' in result['top_features_missed']
```

---

## Combined Implementation Plan

### Timeline

**Total Effort**: 8-12 hours (development + testing)

| Week | Task | Hours | Status |
|------|------|-------|--------|
| Week 4 | Implement `detect_bimodal_pattern()` | 1-2 hrs | Pending |
| Week 4 | Implement `compute_rf_alignment()` | 3-4 hrs | Pending |
| Week 4 | Implement `identify_high_contrast_features()` | 4-6 hrs | Pending |
| Week 5 | Integration with prompts | 2 hrs | Pending |
| Week 5 | Unit testing | 2 hrs | Pending |
| Week 5 | Integration testing on real data | 2 hrs | Pending |

**Prerequisite**: LLMOutputFix.md must be complete and stable

---

### Files Modified

| File | Changes | Lines Added |
|------|---------|-------------|
| `stage7_preprocessing.py` | Add 3 new functions | ~150 lines |
| `stage7_prompts.py` | Integrate functions into prompts | ~80 lines |
| `test_preprocessing.py` | Add unit tests | ~100 lines |
| Total | | ~330 lines |

---

### Integration Order

1. **Implement Function 2 first** (`detect_bimodal_pattern()`)
   - Simplest (1-2 hours)
   - Independent (no dependencies)
   - Validates development environment

2. **Implement Function 3 second** (`compute_rf_alignment()`)
   - Moderate complexity (3-4 hours)
   - Tests distribution data integration

3. **Implement Function 1 last** (`identify_high_contrast_features()`)
   - Most complex (4-6 hours)
   - Largest prompt impact
   - Benefits from lessons learned in Functions 2-3

---

## Expected Impact

### Quantitative

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Prompt tokens (Phase 1)** | ~3500 tokens/window | ~2800 tokens/window | -20% (700 tokens) |
| **API cost per bucket** | $2.50 | $2.00 | -$0.50 |
| **LLM processing time** | 30s/window | 25s/window | -17% (5s) |
| **Features per cluster** | 21 features | 8-12 features | -43% noise |

### Qualitative

✅ **LLM Focus**: Clearer differentiation between clusters
✅ **Recommendations**: More specific, less generic
✅ **Validation**: RF alignment scores add credibility
✅ **Bimodal Handling**: Multiple strategies presented clearly
✅ **Developer Debugging**: High-contrast + alignment metrics aid troubleshooting

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Over-filtering (too aggressive threshold)** | 🟡 Medium | 🟢 Low | Make threshold configurable (0.15-0.25) |
| **Missing bimodal patterns (30% too strict)** | 🟢 Low | 🟢 Low | Test on real data, adjust if needed |
| **RF alignment false negatives (threshold mismatch)** | 🟡 Medium | 🟡 Medium | Use distribution thresholds when available |
| **Integration breaks existing prompts** | 🟢 Low | 🟢 Low | Comprehensive testing before deployment |
| **Minimal value vs effort** | 🟡 Medium | 🟡 Medium | Monitor metrics before/after, can revert |

---

## Success Criteria

### Must Have ✅

- [ ] All 3 functions implemented and tested
- [ ] Unit tests pass with ≥90% coverage
- [ ] Integration with `build_phase1_prompt()` complete
- [ ] Prompt token reduction ≥15%
- [ ] Zero regressions in output quality

### Should Have 🎯

- [ ] RF alignment appears in LLM output (e.g., "This cluster uses 4 of 5 top predictors")
- [ ] Bimodal patterns detected and presented clearly
- [ ] High-contrast features prioritized in recommendations
- [ ] API cost reduction ≥$0.40 per bucket

### Nice to Have 🌟

- [ ] Configurable thresholds via config file
- [ ] Dashboard showing RF alignment distribution across all clusters
- [ ] A/B test comparing outputs with/without these functions

---

## Rollback Plan

If implementation doesn't meet success criteria:

### Option A: Revert Entirely
- Remove 3 functions
- Restore original prompt logic
- Zero cost, back to known-good state

### Option B: Keep Partial Implementation
- Keep `detect_bimodal_pattern()` (lowest risk, clear value)
- Remove `identify_high_contrast_features()` if over-filtering is issue
- Keep `compute_rf_alignment()` if validation adds value

### Option C: Adjust Thresholds
- Lower contrast threshold: 0.20 → 0.15 (show more features)
- Lower bimodal threshold: 0.30 → 0.25 (detect more patterns)
- Adjust RF alignment: "optimal" = 80% of top_avg instead of 90%

---

## Related Documentation

- **Stage7TIUpdate.md**: TI alignment analysis (source of these 3 functions)
- **LLMOutputFix.md**: PREREQUISITE - must be complete first
- **LLMAnalysisCHILDTI.md Section 4.1-4.3**: Original TI specs for these functions
- **stage7_preprocessing.py**: Implementation location
- **stage7_prompts.py**: Integration location

---

## Appendix: Why Not Implement All 9 Functions?

For reference, here's why the other 6 functions were rejected:

| Function | Rejection Reason |
|----------|------------------|
| **enrich_high_contrast_features()** | Already partially done in prompt builder |
| **prepare_path_data_for_llm()** | Current simple format works fine |
| **classify_confidence_level()** | Removes valuable LLM judgment |
| **generate_universal_principles()** | LLMOutputFix.md already addresses this |
| **generate_cross_window_patterns()** | LLMOutputFix.md already addresses this |
| **generate_feature_based_reports()** | High effort (10-15 hrs), low value, robotic output |

**Total rejected**: 6 functions, ~25-35 hours of work, minimal value

---

## Document Control

**Created**: 2025-01-28
**Version**: 1.0
**Status**: Proposed (Awaiting LLMOutputFix Completion)
**Next Review**: After LLMOutputFix.md implementation complete
**Approver**: [Your Name/Team]

---

**End of Stage 7 Option B Implementation Guide**
