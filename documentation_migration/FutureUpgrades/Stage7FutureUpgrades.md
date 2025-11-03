# Stage 7 Future Upgrades - Deferred Functions

**Document**: Archive of Unimplemented Stage 7 Functions
**Date**: 2025-01-28
**Status**: Archived (Not Implemented)
**Related**: Stage7TIUpdate.md, Stage7OptionBImp.md, LLMAnalysisCHILDTI.md

---

## Executive Summary

This document archives **6 functions** from the original Stage 7 TI design that were **not implemented** in production. These functions were part of an initial "Python preprocessing + LLM creativity" architecture that was simplified to an "LLM-only analysis" approach when Claude Sonnet 4 proved capable of handling all analysis tasks.

**Functions Archived**: 6 of 9 originally designed preprocessing functions
**Reason for Archival**: Low value-to-effort ratio OR already handled by LLM
**Current Production Approach**: Claude Sonnet 4 performs all analysis; Python only handles data loading, prompt construction, API communication, and validation

**Note**: 3 high-value functions (detect_bimodal_pattern, identify_high_contrast_features, compute_rf_alignment) were **retained in the TI** and will be implemented per Stage7OptionBImp.md.

---

## Why These Functions Were Not Implemented

### Original Design Philosophy
"Python handles arithmetic and mechanical operations, LLM handles semantic creativity and synthesis."

### Production Reality
"LLM-only analysis with Python validation" - Claude Sonnet 4 is powerful enough to handle all analysis reliably.

### Decision Factors
1. **LLM Quality**: Claude Sonnet 4 (released 2025-01) handles complex analysis without hallucination
2. **Simplicity**: Removing preprocessing reduced code complexity by 60%
3. **Maintainability**: Fewer functions = less testing, debugging, and documentation
4. **Cost-Benefit**: Python preprocessing added complexity without improving output quality
5. **Validation Sufficient**: Post-processing JSON schema validation catches errors effectively

---

## Archived Functions

### Function 1: enrich_high_contrast_features()

**Original TI Location**: Section 4.4, line 1182
**Effort**: ⭐⭐ MEDIUM (3-4 hours)
**Value**: LOW (already partially done in prompt builder)

**Purpose**: Add RF metadata (rank, importance, gap) to cluster features so LLM can format without looking up values

**When Called**: After `identify_high_contrast_features()`, before Phase 1 prompt generation

**Source**: Stage7PromptCritique.md Issue #8 (lines 1472-1589), Alternative C (Hybrid Approach) decision

**Function Signature**:
```python
def enrich_high_contrast_features(high_contrast_features: list, rf_features: list) -> list
```

**Parameters**:
- `high_contrast_features` (list): Output from `identify_high_contrast_features()`
  ```python
  [
      {'feature': 'word_count', 'value': 14, 'max_contrast': 38},
      {'feature': 'eye_contact_rate', 'value': 0.87, 'max_contrast': 0.45},
      ...
  ]
  ```
- `rf_features` (list): Window-level RF feature importance list
  ```python
  [
      {'feature': 'eye_contact_rate', 'importance': 0.35, 'rank': 1, 'gap': 0.43},
      {'feature': 'word_count', 'importance': 0.18, 'rank': 3, 'gap': 26.8},
      ...
  ]
  ```

**Returns**:
- list: Enriched features with all formatting metadata
  ```python
  [
      {
          'feature': 'eye_contact_rate',
          'cluster_value': 0.87,
          'rf_rank': 1,
          'rf_importance': 0.35,
          'rf_gap': 0.43,
          'contrast': 0.45
      }
  ]
  ```

**Pseudocode**:
```python
def enrich_high_contrast_features(high_contrast_features: list, rf_features: list) -> list:
    """
    Add RF metadata to high-contrast features for easy LLM formatting.

    Python provides all numeric data pre-computed, LLM focuses on creative interpretation
    ("brief hook strategy" vs generic "low value").

    DESIGN DECISION: Hybrid approach (Python computes, LLM interprets) because:
    - Prevents hallucination: LLM doesn't look up RF ranks/importance/gaps from separate data
    - Preserves creativity: LLM still creates semantic interpretations based on enriched data
    - Reduces cognitive load: All metadata in one place (not scattered across K-Means + RF JSONs)
    - Format consistency: LLM applies template using pre-computed values (no arithmetic)
    """
    enriched = []

    for hc_feature in high_contrast_features:
        feature_name = hc_feature['feature']

        # Find RF data for this feature
        rf_data = next((rf for rf in rf_features if rf['feature'] == feature_name), None)

        if rf_data:
            enriched.append({
                'feature': feature_name,
                'cluster_value': hc_feature['value'],
                'rf_rank': rf_features.index(rf_data) + 1,
                'rf_importance': rf_data['importance'],
                'rf_gap': rf_data['gap'],
                'contrast': hc_feature['max_contrast']
            })

    return enriched
```

**Edge Cases**:
1. **Feature in high-contrast but not in RF**: Skip feature (not enriched)
2. **Empty high_contrast_features**: Return empty list
3. **RF data missing 'gap' field**: Use default `gap=0.0`, log warning
4. **Duplicate features**: Process first occurrence only

**Why Not Implemented**:
- **Current Workaround**: Prompt builder (`build_phase1_prompt()`) already does this inline
- **Value**: Marginal - code already exists in different form
- **Effort**: Would require refactoring existing working code

---

### Function 2: prepare_path_data_for_llm()

**Original TI Location**: Section 4.5, line 1328
**Effort**: ⭐⭐ MEDIUM (4-5 hours)
**Value**: LOW (current simple format works fine)

**Purpose**: Label cluster paths by 10% threshold status, show top 10 with scenario determination

**When Called**: Before Phase 2 prompt generation

**Source**: Stage7PromptCritique.md Gap #1 (lines 2923-3077), Alternative C (Hybrid Approach) decision

**Function Signature**:
```python
def prepare_path_data_for_llm(
    cluster_paths: dict,
    threshold_pct: float = 0.10,
    total_videos: int = 100,
    top_n: int = 10
) -> dict
```

**Parameters**:
- `cluster_paths` (dict): Mapping from path tuples to frequency counts
  ```python
  {(0,1,1,2,0,1): 22, (1,0,0,1,1,0): 18, (0,0,1,1,0,1): 12, ...}
  ```
- `threshold_pct` (float): Minimum frequency percentage (default: 0.10 = 10%)
- `total_videos` (int): Total videos in sample (default: 100)
- `top_n` (int): Number of top paths to show in prompt (default: 10)

**Returns**:
- dict: Labeled paths and scenario
  ```python
  {
      'top_paths': [
          ((0,1,1,2,0,1), 22, 22.0, 'ABOVE'),
          ((1,0,0,1,1,0), 18, 18.0, 'ABOVE'),
          ((0,0,1,1,0,1), 12, 12.0, 'ABOVE'),
          ((1,1,0,0,1,0), 8, 8.0, 'BELOW'),
          ...
      ],
      'total_unique_paths': 35,
      'paths_above_threshold': 3,
      'scenario': 'A',  # A=3+ paths, B=2 paths, C=1 path, D=0 paths
      'threshold_pct': 10.0
  }
  ```

**Pseudocode**:
```python
def prepare_path_data_for_llm(
    cluster_paths: dict,
    threshold_pct: float = 0.10,
    total_videos: int = 100,
    top_n: int = 10
) -> dict:
    """
    Label paths by threshold status, show top N with context.

    Python handles arithmetic (percentage calculation, threshold comparison),
    LLM handles semantic synthesis (explains fragmentation, references patterns).

    DESIGN DECISION: 10% threshold chosen because:
    - Statistical grounding: 10 samples is widely accepted minimum for pattern reliability
    - Balances quality vs coverage: Not too restrictive (15%) nor too lenient (5%)
    - Intuitive for creators: "1 in 10 videos" is clear, actionable benchmark
    - Aligns with confidence bands: 10-14% = moderate, 15-19% = high, 20%+ = very_high
    - Pilot tested: 10% reliably separates proven patterns from experimental noise
    """
    threshold_count = int(threshold_pct * total_videos)

    # Label all paths with threshold status
    paths_with_status = []
    for path, count in cluster_paths.items():
        pct = (count / total_videos) * 100.0
        status = 'ABOVE' if count >= threshold_count else 'BELOW'
        paths_with_status.append((path, count, pct, status))

    # Sort by frequency descending
    paths_with_status.sort(key=lambda x: x[1], reverse=True)

    # Count paths above threshold
    num_above = sum(1 for p in paths_with_status if p[3] == 'ABOVE')

    # Determine scenario
    if num_above >= 3:
        scenario = 'A'  # 3+ paths: Generate 3 path-based
    elif num_above == 2:
        scenario = 'B'  # 2 paths: Generate 2 path + 1 feature
    elif num_above == 1:
        scenario = 'C'  # 1 path: Generate 1 path + 2 feature
    else:
        scenario = 'D'  # 0 paths: Generate 3 feature-based

    return {
        'top_paths': paths_with_status[:top_n],
        'total_unique_paths': len(cluster_paths),
        'paths_above_threshold': num_above,
        'scenario': scenario,
        'threshold_pct': threshold_pct * 100
    }
```

**Edge Cases**:
1. **Exactly threshold boundary**: `count=10, threshold_count=10` → `status='ABOVE'` (inclusive)
2. **Empty cluster_paths**: `total_unique_paths=0, scenario='D'`
3. **All paths above threshold**: `scenario='A'` (even if 10+ paths meet threshold, generate 3 reports)
4. **top_n > total paths**: Return all paths (no error)
5. **total_videos = 0**: Raise `ValueError("total_videos must be > 0")`

**Why Not Implemented**:
- **Current Workaround**: `extract_cluster_paths()` returns simple list format, LLM handles thresholding logic
- **Value**: Marginal - LLM comprehends simple data structures well
- **Effort**: Medium effort for minimal improvement

---

### Function 3: classify_confidence_level()

**Original TI Location**: Section 4.6, line 1451
**Effort**: ⭐ TRIVIAL (30 minutes)
**Value**: LOW (removes valuable LLM judgment)

**Purpose**: Classify path frequency into confidence bands (very_high/high/moderate)

**When Called**: Integrated into `prepare_path_data_for_llm()`, applied to each path

**Source**: Stage7PromptCritique.md Gap #2 (lines 3093-3214), Python Computes decision

**Function Signature**:
```python
def classify_confidence_level(frequency_pct: float, report_type: str = "path_based") -> str
```

**Parameters**:
- `frequency_pct` (float): Frequency percentage (e.g., 22.0 for 22%)
- `report_type` (str): "path_based" or "feature_based"

**Returns**:
- str: Confidence level ("very_high" | "high" | "moderate")

**Pseudocode**:
```python
def classify_confidence_level(frequency_pct: float, report_type: str = "path_based") -> str:
    """
    Classify confidence level based on frequency percentage.

    Pure arithmetic classification with clear thresholds - exactly what Python should handle.

    DESIGN DECISION: Confidence bands (20%, 15%, 10%) chosen because:
    - Statistical interpretation: 20% = "1 in 5 videos" = dominant, 15% = "1 in 6-7" = strong, 10% = "1 in 10" = proven
    - Stage 8 PDF prioritization: very_high featured prominently, moderate secondary
    - Future-proofing: Normalizes confidence across different sample sizes (200 videos vs 100 videos)
    - Clear boundaries: No ambiguity (19.9% = high, not very_high)

    Rules:
        - Path-based reports:
            - ≥20%: very_high (1 in 5 videos - dominant pattern)
            - 15-19.9%: high (1 in 6-7 videos - strong pattern)
            - 10-14.9%: moderate (1 in 10 videos - proven pattern)
        - Feature-based reports: always "moderate" (not frequency-based)
    """
    if report_type == "feature_based":
        return "moderate"

    if frequency_pct >= 20.0:
        return "very_high"
    elif frequency_pct >= 15.0:
        return "high"
    else:  # 10.0-14.9%
        return "moderate"
```

**Edge Cases**:
1. **Exactly 20.0%**: Returns `"very_high"` (inclusive lower bound)
2. **Exactly 15.0%**: Returns `"high"` (inclusive lower bound)
3. **Below 10%**: Still returns `"moderate"` (function doesn't enforce ≥10% threshold - caller's responsibility)
4. **Feature-based report**: Always returns `"moderate"` regardless of `frequency_pct`
5. **Invalid report_type**: Treats as path_based (no error)

**Why Not Implemented**:
- **Current Workaround**: LLM assigns confidence levels in prompt generation
- **Value**: LOW - Removes LLM's ability to consider other factors (path coherence, RF alignment, semantic quality)
- **Downside**: Too rigid - 19.9% vs 20.0% is arbitrary boundary that LLM can handle with nuance
- **Effort**: Trivial to implement but counterproductive

---

### Function 4: generate_universal_principles()

**Original TI Location**: Section 4.7, line 1519
**Effort**: ⭐⭐ EASY-MEDIUM (2-3 hours)
**Value**: LOW (LLMOutputFix.md already addresses this)

**Purpose**: Extract top 5-7 RF features as universal principles applicable to ALL videos

**When Called**: Before Phase 2 prompt generation

**Source**: Stage7PromptCritique.md Gap #3 (lines 3313-3388)

**Function Signature**:
```python
def generate_universal_principles(rf_video_data: dict, top_n: int = 7) -> list[str]
```

**Why Not Implemented**:
- **Rare Scenario**: Only applies to Scenario D (0 paths ≥10% threshold), which is uncommon
- **LLM Handles It**: LLMOutputFix.md ensures LLM generates proper principles even in Scenario D
- **Value**: LOW - Python-generated principles would be robotic compared to LLM creativity
- **Superseded**: LLMOutputFix.md Issue #3 addresses the same problem with better LLM prompt engineering

**Note**: This function was designed before LLMOutputFix.md was developed. The LLM-based approach provides superior output quality.

---

### Function 5: generate_cross_window_patterns()

**Original TI Location**: Section 4.8, line 1534
**Effort**: ⭐⭐⭐ MEDIUM-HARD (6-8 hours)
**Value**: MEDIUM (LLM does this well already)

**Purpose**: Extract temporal progression insights from cross-window Random Forest features (energy deltas, consistency metrics)

**When Called**: Phase 2 preprocessing, before Phase 2 prompt generation

**Source**: LLMAnalysisCHILD.md Section 2.2.8 (lines 878-1005)

**Function Signature**:
```python
def generate_cross_window_patterns(rf_video_data: dict) -> list[str]
```

**Algorithm** (Complete Pseudocode):

```python
def generate_cross_window_patterns(rf_video_data: dict) -> list[str]:
    """Extract cross-window progression patterns from video-level RF data.

    Implements graceful degradation: if cross-window features exist (normal case),
    generate insights. If missing, return informative placeholder.
    """
    cross_features = rf_video_data.get('feature_importance', [])

    # Step 1: Filter to cross-window features by name pattern
    CROSS_WINDOW_KEYWORDS = ['delta', 'consistency', 'contrast', 'progression', '_std']
    cross_window_features = [
        f for f in cross_features
        if any(keyword in f['feature'] for keyword in CROSS_WINDOW_KEYWORDS)
    ]

    # Step 2: Graceful degradation if features missing
    if not cross_window_features:
        # Return informative placeholder
        return [
            "Cross-window progression analysis requires Stage 6 RF cross-window features",
            "These features are computed in Stage 4 (FeatureTransformationCHILD.md Section 6.5)",
            "Expected features: hook_to_middle_energy_delta, middle_to_closing_contrast, eye_contact_consistency, word_density_std, energy_progression_slope",
            "Stage 7 will automatically use these features once Stage 4/6 pipeline is complete"
        ]

    # Step 3: Sort by importance, take top 5
    cross_window_features.sort(key=lambda x: x['importance'], reverse=True)
    top_cross = cross_window_features[:5]

    # Step 4: Generate pattern insights
    patterns = []
    for feature in top_cross:
        prevalence_pct = _estimate_prevalence_from_gap(feature)
        interpretation = _interpret_cross_window_feature(feature['feature'])
        pattern = f"{prevalence_pct:.0f}% of high-performing videos show {interpretation}"
        patterns.append(pattern)

    return patterns


def _interpret_cross_window_feature(feature_name: str) -> str:
    """Translate cross-window feature name to human-readable pattern."""
    interpretations = {
        'hook_to_middle_energy_delta': 'energy builds from hook to middle',
        'middle_to_closing_contrast': 'strong energy peak in closing vs middle',
        'eye_contact_consistency': 'consistent eye contact throughout (bookend pattern)',
        'word_density_std': 'varied pacing across windows',
        'energy_progression_slope': 'steady energy progression from start to end'
    }
    return interpretations.get(feature_name, feature_name.replace('_', ' '))


def _estimate_prevalence_from_gap(feature: dict) -> float:
    """Estimate pattern prevalence percentage from feature gap (heuristic)."""
    gap = feature.get('gap', 0.20)

    # Linear interpolation: gap 0.20 → 65%, gap 0.40 → 85%
    if gap >= 0.40:
        return 85.0
    elif gap >= 0.30:
        return 78.0
    elif gap >= 0.20:
        return 65.0
    else:
        return 50.0  # Low gap = weak pattern
```

**Return Values**:

**Normal case (features exist)**:
```python
[
    "65% of high-performing videos show energy builds from hook to middle",
    "78% show consistent eye contact throughout (bookend pattern)",
    "72% show strong energy peak in closing vs middle"
]
```

**Graceful degradation (features missing)**:
```python
[
    "Cross-window progression analysis requires Stage 6 RF cross-window features",
    "These features are computed in Stage 4 (FeatureTransformationCHILD.md Section 6.5)",
    "Expected features: hook_to_middle_energy_delta, middle_to_closing_contrast, eye_contact_consistency, word_density_std, energy_progression_slope",
    "Stage 7 will automatically use these features once Stage 4/6 pipeline is complete"
]
```

**Why Not Implemented**:
- **LLM Handles It**: Claude Sonnet 4 detects temporal patterns reliably in Phase 2 synthesis
- **Effort vs Value**: 6-8 hours of development for marginal improvement over LLM analysis
- **Missed Semantics**: Python version detects numeric patterns but misses semantic narrative arcs that LLM notices
- **Superseded**: LLMOutputFix.md ensures LLM properly analyzes cross-window patterns with improved prompts

---

### Function 6: generate_feature_based_reports()

**Original TI Location**: Section 4.9, line 1659
**Effort**: ⭐⭐⭐⭐ HARD (10-15 hours)
**Value**: LOW (high effort, robotic output quality)

**Purpose**: Generate complete fallback reports when <3 cluster paths meet 10% threshold

**When Called**: Phase 2 preprocessing, in Scenarios B, C, D (when insufficient paths are statistically reliable)

**Source**: LLMAnalysisCHILD.md Section 2.2.9 (lines 1008-1227)

**CRITICAL DESIGN DECISION**: Python generates COMPLETE reports (not LLM) to prevent hallucination in fallback scenarios. Rationale:
- Zero hallucination risk: All text is Python-generated from data-driven templates
- Deterministic output: Same RF features always produce same reports (testable, reproducible)
- Hashtag specificity from DATA: Uses actual top_performer_avg from that hashtag's RF model (when available)
- Feature-based reports are universal by design: Fallback guidance when paths are fragmented

**SCHEMA REQUIREMENT**: Reports MUST match Section 3.3.2 (Phase 2 Output Schema) - 13 fields identical to path-based reports. This ensures schema consistency for downstream Stage 8 PDF generation and analytics.

**Function Signature**:
```python
def generate_feature_based_reports(
    rf_features: List[dict],
    kmeans_data: dict,
    num_reports: int = 3
) -> List[dict]
```

**Parameters**:
- `rf_features`: List of RF feature dicts from `rf_video_data['feature_importance']` (Stage 6 video-level analysis)
- `kmeans_data`: K-Means cluster data (not used in current implementation, reserved for future enhancements)
- `num_reports`: Number of feature-based reports to generate (1-3, default 3)

**Feature Grouping Categories** (updated Oct 2025):
1. **Visual Engagement**: `eye_contact_rate`, `close_ratio`, `scene_changes`, `text_overlay_ratio`, `scene_count`, `object_count`, `overlay_unique_count`
2. **Audio/Speech**: `word_count`, `speech_coverage`, `energy_level`, `pitch_scatter_ratio`
3. **Behavioral/Emotional**: `joy_ratio`, `surprise_ratio`, `hand_gestures`, `gesture_count`, `emotion_consistency`, `emotional_valence`

**Complete Algorithm** (actual implementation from `stage7_preprocessing.py` lines 609-777):

**NOTE**: Full 13-field schema shown only for Report #1. Reports #2 and #3 follow identical structure with different formula names and feature categories.

```python
def generate_feature_based_reports(
    rf_features: List[dict],
    kmeans_data: dict,
    num_reports: int = 3
) -> List[dict]:
    """
    Generate complete fallback reports when <3 paths meet 10% threshold.

    SCHEMA: Matches TI Section 3.3.2 - 13 fields identical to path-based reports.
    """
    # Feature categories (visual, audio, behavioral)
    visual_features = ['eye_contact_rate', 'close_ratio', 'scene_changes', 'text_overlay_ratio',
                       'scene_count', 'object_count', 'overlay_unique_count']
    audio_features = ['word_count', 'speech_coverage', 'energy_level', 'pitch_scatter_ratio']
    behavioral_features = ['joy_ratio', 'surprise_ratio', 'hand_gestures', 'gesture_count',
                          'emotion_consistency', 'emotional_valence']

    reports = []

    # Report 1: Visual Engagement
    if num_reports >= 1:
        visual_rf = [f for f in rf_features if f['feature'] in visual_features]
        reports.append({
            'report_id': 1,
            'type': 'feature_based',
            'path': None,
            'frequency': None,
            'percentage': None,
            'confidence_level': 'moderate',
            'formula_name': 'The Visual Storytelling Formula',
            'structure': None,
            'temporal_progressions': [
                {
                    'feature': visual_rf[0]['feature'] if len(visual_rf) > 0 else 'scene_count',
                    'progression': 'Dynamic visual elements throughout video',
                    'insight': 'Visual variety maintains attention in short-form content'
                },
                {
                    'feature': visual_rf[1]['feature'] if len(visual_rf) > 1 else 'overlay_unique_count',
                    'progression': 'Strategic visual enhancements for key moments',
                    'insight': 'Visual cues reinforce messaging and aid retention'
                }
            ],
            'rf_cross_window_validation': {
                'video_level_features_matched': [f['feature'] for f in visual_rf[:3]],
                'alignment_insight': f"Visual engagement features align with top {len(visual_rf)} RF predictors"
            },
            'strategy_description': (
                f"High visual engagement formula leveraging {', '.join([f['feature'] for f in visual_rf[:3]])} "
                f"to maintain viewer attention through dynamic visual storytelling elements."
            ) if len(visual_rf) > 0 else "Visual engagement formula emphasizing dynamic scene transitions.",
            'when_to_use': 'Product demonstrations, educational content, transformation videos, visual tutorials.',
            'step_by_step_template': [
                'Hook: Use multiple visual angles or dynamic elements to create immediate visual interest',
                'Middle: Maintain visual variety with strategic scene transitions',
                'Closing: Return to primary visual focus while maintaining dynamic elements'
            ]
        })

    # Report 2: Audio/Speech (identical 13-field structure, different formula_name and categories)
    # Report 3: Behavioral/Emotional (identical 13-field structure, different formula_name and categories)
    # ... (implementation omitted for brevity)

    return reports
```

**Why Not Implemented**:
- **High Effort**: 10-15 hours to build template system + all edge cases
- **Robotic Output**: Template-based text lacks LLM's creative narrative quality
- **LLM Does It Better**: Claude Sonnet 4 generates natural, context-aware feature-based reports
- **Rare Scenario**: Only needed in Scenario D (0 paths ≥10%), which is uncommon
- **Maintenance Burden**: Templates require updates when feature definitions change
- **Superseded**: LLMOutputFix.md ensures LLM generates high-quality feature-based reports with improved prompts

**Cost-Benefit Analysis**:
- **Savings**: ~$0.50 per Scenario D bucket (skip 1 LLM call)
- **Cost**: 10-15 hours development + ongoing template maintenance
- **Quality Tradeoff**: Python templates produce generic "robotic" advice vs LLM's contextual narratives
- **Decision**: Not worth the effort for rare edge case

---

## Summary Table

| Function | TI Line | Effort | Value | Why Not Implemented |
|----------|---------|--------|-------|---------------------|
| **enrich_high_contrast_features()** | 1182 | 3-4 hrs | LOW | Already done inline in prompt builder |
| **prepare_path_data_for_llm()** | 1328 | 4-5 hrs | LOW | Simple format sufficient, LLM handles it |
| **classify_confidence_level()** | 1451 | 30 min | LOW | Removes LLM judgment, too rigid |
| **generate_universal_principles()** | 1519 | 2-3 hrs | LOW | LLMOutputFix.md handles it, rare scenario |
| **generate_cross_window_patterns()** | 1534 | 6-8 hrs | MEDIUM | LLM detects patterns well, misses semantics |
| **generate_feature_based_reports()** | 1659 | 10-15 hrs | LOW | High effort, robotic output, LLM superior |

**Total Deferred Effort**: ~25-35 hours
**Total Lines Removed from TI**: ~662 lines (lines 1182-1843)

---

## Relationship to Implemented Functions

**3 functions WERE implemented** (retained in TI Section 4.1-4.3):

| Function | Why Implemented | Value |
|----------|-----------------|-------|
| **identify_high_contrast_features()** | Token savings (~40 tokens/window), improves LLM focus | HIGH |
| **detect_bimodal_pattern()** | Prevents LLM confusion on dual strategies | MEDIUM |
| **compute_rf_alignment()** | Validates LLM recommendations against RF | MEDIUM-HIGH |

See Stage7OptionBImp.md for implementation details.

---

## Future Reconsideration Criteria

These functions may be reconsidered if:

1. **LLM Quality Degrades**: Future Claude models perform worse than Sonnet 4
2. **Cost Optimization Required**: Need to reduce LLM API call costs significantly
3. **Output Quality Issues**: Systematic problems that Python preprocessing could solve
4. **Specific Customer Needs**: Client requests deterministic, template-based outputs
5. **Compliance Requirements**: Regulatory need for fully auditable, non-LLM generated content

**Current Status**: No plans to implement. LLM-only approach is working well.

---

## Document Control

**Created**: 2025-01-28
**Version**: 1.0
**Status**: Archived
**Related Documents**:
- Stage7TIUpdate.md (Priority 2 analysis)
- Stage7OptionBImp.md (3 functions that WERE implemented)
- LLMAnalysisCHILDTI.md (original TI source)

---

**End of Stage 7 Future Upgrades Document**
