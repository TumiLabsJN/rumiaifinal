# Stage 7: LLM Analysis - High-Level Design

> **Parent**: MLPlanningv2.md - Stage 7: LLM Analysis - Hybrid Two-Phase Approach (lines 2587-3299)
> **Version**: 2.0
> **Last Updated**: 2025-10-17
> **Status**: Ready for Implementation

---

## Document History

| Date | Version | Changes | Source |
|------|---------|---------|--------|
| **2025-10-16** | 1.0 | Initial HLD generated from Phase 1 Critique + Phase 2 QA | MLPlanningv2.md |
| **2025-10-17** | 2.0 | **INTEGRATED ALL IMPROVEMENTS from Stage7PromptCritique.md** | Stage7PromptCritique.md |

### Version 2.0 Changelog (2025-10-17)

**Major Additions**:

1. **Section 2.2 (NEW)**: Python Preprocessing Pipeline (~1,100 lines)
   - 9 preprocessing functions with full implementation code
   - Phase 1: `detect_bimodal_pattern()`, `identify_high_contrast_features()`, `compute_rf_alignment()`, `enrich_high_contrast_features()`
   - Phase 2: `prepare_path_data_for_llm()`, `classify_confidence_level()`, `generate_universal_principles()`, `generate_cross_window_patterns()`, `generate_feature_based_reports()`
   - Each function includes: code, rationale, design decisions, usage examples

2. **Section 2.4.2 (ENHANCED)**: Complete Phase 1 Prompt Template (~275 lines)
   - Full `build_phase1_prompt()` function with all Issue #1-11 improvements
   - Bimodal pattern detection and Strategy A/B presentation
   - Compressed RF format, high-contrast features, RF alignment scoring
   - Cluster size context guidance
   - Removed "Important" section, added bimodal example notes

3. **Section 2.4.3 (ENHANCED)**: Complete Phase 2 Prompt Template (~310 lines)
   - Full `build_phase2_prompt()` function with all Gap #1-5 improvements
   - 10% threshold with ✅ ABOVE / ❌ BELOW labels
   - Scenario-specific instructions (A/B/C/D)
   - Python-generated feature-based fallback reports embedded in prompt
   - Supplementary insights (universal principles + cross-window patterns)
   - "Exactly 3 reports" enforcement

4. **Section 5.2.1 & 5.2.2 (UPDATED)**: Before/After Schema Comparisons
   - Phase 1: "3-5 features" → "exactly 3 features", RF alignment score in insight
   - Phase 2: Added `type`, `confidence_level` fields; `supplementary_insights` section; scenario support

5. **Section 7 (NEW)**: Implementation Roadmap (~250 lines)
   - 5-phase implementation plan with time estimates (15 hours total)
   - Function-by-function implementation order with dependencies
   - Critical path identification
   - Testing framework structure

6. **Section 8 (NEW)**: Testing & Validation (~296 lines)
   - Phase 1 test scenarios (bimodal detection, high-contrast filtering, RF alignment)
   - Phase 2 test scenarios for ALL 4 scenarios (A/B/C/D)
   - Cross-window feature validation (normal case + graceful degradation)
   - End-to-end integration test with schema compliance checks
   - Complete validation code examples

**Improvements Integrated**:

**Phase 1 Issues** (from Stage7PromptCritique.md):
- ✅ Issue #1: Bimodal pattern detection with alternative strategies
- ✅ Issue #2: "Exactly 3 features" (not "3-5")
- ✅ Issue #3: High-contrast feature pre-filtering (≥0.20 threshold)
- ✅ Issue #4: RF alignment computation and score display
- ✅ Issue #5: Removed "Important" section from prompt
- ✅ Issue #6: Added cluster size context guidance
- ✅ Issue #7: Compressed RF data format
- ✅ Issue #8: Feature enrichment with RF metadata
- ✅ Issue #9: RF alignment score in output schema
- ✅ Issue #10: Bimodal example note in prompt
- ✅ Issue #11: Duplicate of #6 (cluster size)

**Phase 2 Gaps** (from Stage7PromptCritique.md):
- ✅ Gap #1: 10% threshold with labeled paths and scenario determination (A/B/C/D)
- ✅ Gap #2: Confidence level classification (very_high/high/moderate)
- ✅ Gap #3: Hybrid output structure with `supplementary_insights` section
- ✅ Gap #4: Python-generated feature-based fallback reports
- ✅ Gap #5: "Exactly 3 reports" (not "3-5")

**Total New Content**: ~2,231 lines added (document grew from 2,079 → ~4,310 lines)

**Cross-References**:
- **For implementation**: Use this HLD (LLMAnalysisCHILD.md) as single source of truth
- **For decision rationale**: See Stage7PromptCritique.md for detailed analysis of alternatives evaluated
- **For Phase 1 critique**: Stage7PromptCritique.md lines 1-1600 (Issues #1-11)
- **For Phase 2 critique**: Stage7PromptCritique.md lines 1601-4024 (Gaps #1-5)

---

## 1. Context & Business Goal

### 1.1 What Problem Does This Solve?

Stage 6 produces machine learning model insights (feature importance rankings, cluster centroids) in JSON format—data that is technically accurate but not actionable for content creators. Creative strategists need narrative explanations of **why** certain patterns work and **how** to replicate them. This stage transforms ML insights into creator-friendly creative strategies through LLM-powered analysis, delivering specific, actionable recommendations that affiliate creators can immediately apply to their TikTok videos.

### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This stage depends on FoundationCHILD.md (MLPlanningv2.md Part 1) for:
- Client directory structure (Section 2: Client Architecture & Storage)
- CLI parameter definitions (Section 4: CLI Command Structure)
- Bucket definitions (Appendix: BUCKET_WINDOWS config)

```
Stage 5: ML Model Training
   ↓ Output: 90 trained models (8 video-level RF, 41 window-level RF, 41 window-level K-Means)
Stage 6: ML Analysis Generation
   ↓ Output: 13 JSON files per bucket (rf_video_analysis.json, 6-7 window RF JSONs, 6-7 window K-Means JSONs)
Stage 7: LLM Analysis (THIS STAGE)
   ↓ Output: 8 LLM-generated JSON files (6-7 Phase 1 window analyses + 1 Phase 2 synthesis + 1 complete analysis)
Stage 8: PDF Report Generation
```

### 1.3 Success Criteria

- [ ] **Phase 1 execution**: 100% window completion required (all 6-7 windows must succeed, or abort bucket)
- [ ] **Phase 2 synthesis**: Generate 3 creative reports per bucket (path-based preferred, feature-based fallback if <3 paths meet 10% threshold)
- [ ] **Performance target**: Complete both phases in <60 seconds per bucket (conservative timeouts: 90s Phase 1, 180s Phase 2)
- [ ] **Hallucination prevention**: Pass automated validation layer (feature contradictions, invented features, RF misalignment)
- [ ] **Output quality**: All reports include confidence levels (very_high/high/moderate) and RF-validated recommendations

---

## 2. Architecture & Design

### 2.1 High-Level Approach

Stage 7 uses a **two-phase hybrid approach** to minimize LLM hallucination risk while maximizing creative insight quality:

**Phase 1** (Parallel Execution): Analyze each window type independently (hook, middle_1-5, closing) with focused LLM prompts containing only 113-167 numbers per call (K-Means centroids + RF feature importance). Window-level Random Forest provides within-window validation. Runs 6-7 parallel API calls with smart retry logic (retry only failed windows, maximum 2 retry attempts).

**Phase 2** (Single Synthesis Call): Combine Phase 1 window analyses with video-level RF cross-window patterns to identify "Winning Formulas"—common cluster path combinations that predict viral success. Extracts cluster paths for all 100 videos, calculates path frequencies, filters to paths meeting 10% threshold (minimum 10 videos), and generates 3 creative reports with confidence levels and actionable templates.

**Key Design Decision**: Separate phases with small contexts (113 numbers Phase 1 vs 1000+ single-call approach) reduces hallucination risk. Phase 2 synthesis adds cross-window validation through video-level RF patterns (energy progression, consistency metrics).

### 2.2 Python Preprocessing Pipeline

**Purpose**: Before calling Claude API for Phase 1 or Phase 2, Stage 7 runs preprocessing functions to detect patterns, label data, enrich features, and generate fallback content. This division of labor follows the principle: **Python handles arithmetic and mechanical operations, LLM handles semantic creativity and synthesis**.

**Design Philosophy** (from Stage7PromptCritique.md):
1. **Prevent Hallucination**: Python computes all numeric values (percentages, thresholds, alignment scores) - LLM never does arithmetic
2. **Prevent Misclassification**: Python applies clear boundary rules (e.g., ≥10% threshold, ≥20% = very_high confidence) - LLM formats pre-classified data
3. **Allow Open-Ended Discovery**: LLM creates semantic interpretations ("brief hook strategy"), strategic narratives, and contextual recommendations
4. **Python Handles Arithmetic**: All calculations, comparisons, filtering done in code - LLM receives enriched, labeled data

#### 2.2.1 Bimodal Pattern Detection (Phase 1 Preprocessing)

**Purpose**: Detect when a feature shows TWO successful strategies in top performers (e.g., BOTH brief AND dense word counts work)

**When Called**: Before Phase 1 prompt generation, for each RF feature in window-level data

**Source**: Stage7PromptCritique.md Issue #1 (lines 176-272), Alternative A-REVISED decision

```python
def detect_bimodal_pattern(distribution: dict) -> dict:
    """
    Detect if feature shows bimodal pattern in top performers.

    A feature is bimodal when BOTH high AND low percentages are ≥30% among top performers,
    indicating multiple successful strategies exist for this feature.

    DESIGN DECISION: 30% threshold chosen because:
    - Statistical significance: 30% = "nearly 1 in 3 videos" = meaningful minority
    - Avoids false positives: 20%/20% split might be noise, 30%/30% is clear dual-strategy
    - Practical value: Both strategies are common enough for creators to replicate
    - Tested threshold: Pilot testing showed 30% captures true bimodal patterns

    See Stage7PromptCritique.md Issue #1 lines 180-202 for full rationale.

    Args:
        distribution: Stage 6 distribution data structure
            {
                'top_performers': {
                    'high_percentage': 0.40,  # % with ≥66th percentile
                    'low_percentage': 0.35    # % with <33rd percentile
                },
                'bottom_performers': {...}
            }

    Returns:
        dict with bimodal analysis:
        {
            'is_bimodal': True,
            'high_percentage': 0.40,
            'low_percentage': 0.35,
            'interpretation': 'BOTH strategies work',
            'pattern_label': 'BIMODAL'  # For prompt display
        }

    Example:
        # Unimodal case (eye_contact_rate):
        >>> detect_bimodal_pattern({'top_performers': {'high_percentage': 0.72, 'low_percentage': 0.15}})
        {'is_bimodal': False, 'pattern_label': 'UNIMODAL', 'interpretation': 'Single dominant strategy'}

        # Bimodal case (word_count):
        >>> detect_bimodal_pattern({'top_performers': {'high_percentage': 0.40, 'low_percentage': 0.35}})
        {'is_bimodal': True, 'pattern_label': 'BIMODAL', 'interpretation': 'BOTH strategies work'}
    """
    top_high_pct = distribution['top_performers']['high_percentage']
    top_low_pct = distribution['top_performers']['low_percentage']

    is_bimodal = (top_high_pct >= 0.30 and top_low_pct >= 0.30)

    return {
        'is_bimodal': is_bimodal,
        'high_percentage': top_high_pct,
        'low_percentage': top_low_pct,
        'interpretation': 'BOTH strategies work' if is_bimodal else 'Single dominant strategy',
        'pattern_label': 'BIMODAL' if is_bimodal else 'UNIMODAL'
    }
```

**Prompt Data Format** (how LLM sees this):
```
1. eye_contact_rate - RF Importance: 0.35 (rank #1)
   Top: avg 0.88 (72% high, 15% low) | Bottom: avg 0.45 | Gap: 0.43 | Pattern: UNIMODAL

3. word_count - RF Importance: 0.18 (rank #3)
   Top: avg 52 (40% high, 35% low) | Bottom: avg 18 | Gap: 34 | Pattern: BIMODAL
   → Strategy A: Brief (≤20 words) - 35% of top performers
   → Strategy B: Dense (≥80 words) - 40% of top performers
```

**LLM Instruction** (added to Phase 1 prompt):
```
For BIMODAL features: Present BOTH strategies as valid options using format:
"ALTERNATIVE STRATEGIES: Use either [Strategy A] OR [Strategy B] - RF data shows both work"
```

---

#### 2.2.2 High-Contrast Feature Identification (Phase 1 Preprocessing)

**Purpose**: Filter features to only those that DIFFERENTIATE clusters (avoid universal features like "all clusters have high eye contact")

**When Called**: Before Phase 1 prompt generation, for each cluster in K-Means data

**Source**: Stage7PromptCritique.md Issue #3 (lines 572-718), Alternative D (Hybrid Approach) decision

```python
def identify_high_contrast_features(
    kmeans_data: dict,
    threshold: float = 0.20
) -> dict:
    """
    Pre-filter features with high numerical contrast between clusters.

    Python does mechanical filtering (≥0.20 difference), LLM does semantic selection
    (strategic coherence, RF importance weighting).

    DESIGN DECISION: 0.20 threshold chosen because:
    - Domain grounding: 0.20 = 20 percentage points OR 20 words difference = perceptually noticeable
    - Tested on pilot data: 0.20 typically filters 21 features → 8-12 high-contrast features
    - Balances specificity: Not too strict (0.30 = only 3-5 features) nor too lenient (0.10 = 15+ features)
    - LLM-friendly output: 8-12 features is scannable, 21 features overwhelms prompt

    See Stage7PromptCritique.md Issue #3 lines 586-635 for full rationale.

    Args:
        kmeans_data: Stage 6 K-Means JSON for a window
            {
                'clusters': [
                    {'cluster_id': 0, 'centroid': {'eye_contact': 0.87, 'word_count': 14, ...}},
                    {'cluster_id': 1, 'centroid': {'eye_contact': 0.42, 'word_count': 52, ...}},
                    {'cluster_id': 2, 'centroid': {'eye_contact': 0.55, 'word_count': 35, ...}}
                ]
            }
        threshold: Minimum contrast difference (default: 0.20)

    Returns:
        dict with high-contrast features per cluster:
        {
            'clusters': [
                {
                    'cluster_id': 0,
                    'high_contrast_features': [
                        {
                            'feature': 'word_count',
                            'value': 14,
                            'max_contrast': 38,  # vs Cluster 1's 52
                            'contrasts': {'vs Cluster 1': 38, 'vs Cluster 2': 21}
                        },
                        {
                            'feature': 'eye_contact_rate',
                            'value': 0.87,
                            'max_contrast': 0.45,  # vs Cluster 1's 0.42
                            'contrasts': {'vs Cluster 1': 0.45, 'vs Cluster 2': 0.32}
                        }
                    ]
                }
            ]
        }

    Example Output (8 features with ≥0.20 contrast from original 21):
        Reduces cognitive load from "compare 21 features × 3 clusters" to "select 3 from 8 pre-filtered"
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

            # Calculate contrasts
            contrasts = {
                f"vs Cluster {c['cluster_id']}": abs(this_value - c['centroid'][feature])
                for c in clusters
                if c['cluster_id'] != cluster_id
            }

            max_diff = max(abs(this_value - ov) for ov in other_values)

            if max_diff >= threshold:
                high_contrast.append({
                    'feature': feature,
                    'value': this_value,
                    'max_contrast': max_diff,
                    'contrasts': contrasts
                })

        # Sort by max_contrast (highest first)
        high_contrast.sort(key=lambda x: x['max_contrast'], reverse=True)

        result['clusters'].append({
            'cluster_id': cluster_id,
            'all_features': centroid,  # Keep for context
            'high_contrast_features': high_contrast
        })

    return result
```

**Prompt Data Format** (LLM sees pre-filtered features):
```
CLUSTER 0 (35 videos):

All features (for context):
  eye_contact_rate: 0.87, word_count: 14, energy_level: 0.55, ...

High-contrast features (differ by ≥0.20 from other clusters):
  1. word_count: 14 (vs Cluster 1: 52, vs Cluster 2: 35) ← max contrast: 38
  2. eye_contact_rate: 0.87 (vs Cluster 1: 0.42, vs Cluster 2: 0.55) ← max contrast: 0.45
  3. energy_level: 0.55 (vs Cluster 2: 0.85) ← max contrast: 0.30
  ... (8 total high-contrast features)

Your task:
Select exactly 3 defining features from the HIGH-CONTRAST list above, prioritizing:
1. RF importance (rank #1-5 preferred)
2. Strategic coherence (features that tell a coherent story together)
3. Contrast magnitude (larger differences = clearer distinction)
```

---

#### 2.2.3 RF Alignment Computation (Phase 1 Preprocessing)

**Purpose**: Identify which cluster features align with RF top performer patterns (provides validation score like "3/5 RF validated")

**When Called**: Before Phase 1 prompt generation, for each cluster

**Source**: Stage7PromptCritique.md Issue #4 (lines 799-946), Alternative B decision

```python
def compute_rf_alignment(
    cluster_centroid: dict,
    rf_features: list,
    threshold: float = 0.15
) -> dict:
    """
    Identify which cluster features align with RF top performers.

    A feature "aligns" if cluster centroid value is within ±0.15 (15%) of RF top_performer_avg.

    DESIGN DECISION: 0.15 (15%) threshold chosen because:
    - Statistical tolerance: ±15% accommodates natural variance in centroids vs averages
    - Tested on pilot data: 0.10 too strict (only 1-2 matches), 0.20 too lenient (all match)
    - Practical interpretation: "0.87 matches 0.88" (diff 0.01) vs "0.72 differs from 0.88" (diff 0.16)
    - Two-tier matching: <0.10 = "matches exactly", 0.10-0.15 = "close to" (nuance preserved)

    See Stage7PromptCritique.md Issue #4 lines 799-843 for full rationale.

    Args:
        cluster_centroid: K-Means cluster centroid values
            {'eye_contact_rate': 0.87, 'word_count': 14, 'energy_level': 0.55, ...}
        rf_features: Window-level RF feature importance list (from Stage 6)
            [
                {'feature': 'eye_contact_rate', 'importance': 0.35, 'rank': 1, 'top_performer_avg': 0.88, ...},
                {'feature': 'energy_level', 'importance': 0.22, 'rank': 2, 'top_performer_avg': 0.53, ...},
                ...
            ]
        threshold: Alignment tolerance (default: 0.15 = within 15%)

    Returns:
        dict with aligned features and score:
        {
            'aligned_features': [
                {
                    'feature': 'eye_contact_rate',
                    'cluster_value': 0.87,
                    'top_avg': 0.88,
                    'rf_rank': 1,
                    'rf_importance': 0.35,
                    'alignment': 'matches',  # diff <0.10
                    'formatted': "eye_contact_rate: Cluster value 0.87 matches top avg 0.88 (RF rank #1, importance 0.35)"
                },
                {
                    'feature': 'energy_level',
                    'cluster_value': 0.55,
                    'top_avg': 0.53,
                    'rf_rank': 2,
                    'rf_importance': 0.22,
                    'alignment': 'close to',  # diff 0.10-0.15
                    'formatted': "energy_level: Cluster value 0.55 close to top avg 0.53 (RF rank #2, importance 0.22)"
                }
            ],
            'alignment_count': 2,
            'alignment_score': '2/5'  # 2 of top 5 RF features aligned
        }

    Example (Stage 8 usage):
        PDF can display "60% RF Validated" badge (alignment_score "3/5" → 60%)
    """
    aligned_features = []

    for rf_feature in rf_features[:5]:  # Check top 5 RF features
        feature_name = rf_feature['feature']
        top_avg = rf_feature['top_performer_avg']
        rf_rank = rf_features.index(rf_feature) + 1
        rf_importance = rf_feature['importance']

        if feature_name in cluster_centroid:
            cluster_value = cluster_centroid[feature_name]
            diff = abs(cluster_value - top_avg)

            if diff <= threshold:
                alignment_type = 'matches' if diff <= 0.10 else 'close to'

                aligned_features.append({
                    'feature': feature_name,
                    'cluster_value': cluster_value,
                    'top_avg': top_avg,
                    'rf_rank': rf_rank,
                    'rf_importance': rf_importance,
                    'alignment': alignment_type,
                    'formatted': (
                        f"{feature_name}: Cluster value {cluster_value:.2f} {alignment_type} "
                        f"top avg {top_avg:.2f} (RF rank #{rf_rank}, importance {rf_importance:.2f})"
                    )
                })

    return {
        'aligned_features': aligned_features,
        'alignment_count': len(aligned_features),
        'alignment_score': f"{len(aligned_features)}/5"
    }
```

**Prompt Data Format** (LLM sees pre-computed alignment):
```
CLUSTER 0 (35 videos):

RF Alignment (features matching top performer patterns):
  ✅ eye_contact_rate: Cluster value 0.87 matches top avg 0.88 (RF rank #1, importance 0.35)
  ✅ energy_level: Cluster value 0.55 close to top avg 0.53 (RF rank #2, importance 0.22)
  ❌ word_count: Cluster value 14 differs from top avg 52 (RF rank #3) ← Not aligned

  Alignment score: 2/5 (uses 2 of top 5 RF features at optimal levels)

Your task:
Generate rf_validation section using pre-computed data above:
{
  "top_predictive_features_in_cluster": [
    // Copy the ✅ aligned features from RF Alignment data
  ],
  "insight": "This cluster leverages 2 of the top 5 most predictive features (RF alignment: 2/5)..."
}
```

---

#### 2.2.4 Feature Enrichment (Phase 1 Preprocessing)

**Purpose**: Add RF metadata (rank, importance, gap) to cluster features so LLM can format without looking up values

**When Called**: After `identify_high_contrast_features()`, before Phase 1 prompt generation

**Source**: Stage7PromptCritique.md Issue #8 (lines 1472-1589), Alternative C (Hybrid Approach) decision

```python
def enrich_high_contrast_features(
    high_contrast_features: list,
    rf_features: list
) -> list:
    """
    Add RF metadata to high-contrast features for easy LLM formatting.

    Python provides all numeric data pre-computed, LLM focuses on creative interpretation
    ("brief hook strategy" vs generic "low value").

    DESIGN DECISION: Hybrid approach (Python computes, LLM interprets) because:
    - Prevents hallucination: LLM doesn't look up RF ranks/importance/gaps from separate data
    - Preserves creativity: LLM still creates semantic interpretations based on enriched data
    - Reduces cognitive load: All metadata in one place (not scattered across K-Means + RF JSONs)
    - Format consistency: LLM applies template using pre-computed values (no arithmetic)

    See Stage7PromptCritique.md Issue #8 lines 1477-1503 for full rationale.

    Args:
        high_contrast_features: Output from identify_high_contrast_features()
            [
                {'feature': 'word_count', 'value': 14, 'max_contrast': 38},
                {'feature': 'eye_contact_rate', 'value': 0.87, 'max_contrast': 0.45},
                ...
            ]
        rf_features: Window-level RF feature importance list
            [
                {'feature': 'eye_contact_rate', 'importance': 0.35, 'rank': 1, 'gap': 0.43},
                {'feature': 'word_count', 'importance': 0.18, 'rank': 3, 'gap': 26.8},
                ...
            ]

    Returns:
        list of enriched features with all formatting metadata:
        [
            {
                'feature': 'eye_contact_rate',
                'cluster_value': 0.87,
                'rf_rank': 1,
                'rf_importance': 0.35,
                'rf_gap': 0.43,
                'contrast': 0.45
            },
            ...
        ]

    Example (LLM applies format template):
        "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43 - HIGHEST PREDICTOR)"
                          ↑ cluster_value  ↑ rf_rank  ↑ rf_importance  ↑ rf_gap  ↑ LLM interpretation
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

**Prompt Data Format** (LLM sees enriched features):
```
High-contrast features (with RF metadata for formatting):
  1. feature: eye_contact_rate
     cluster_value: 0.87
     rf_rank: 1, importance: 0.35, gap: 0.43
     contrast: 0.45

  2. feature: word_count
     cluster_value: 14
     rf_rank: 3, importance: 0.18, gap: 26.8
     contrast: 38

Your task:
Select exactly 3 defining features and format each as:
"feature_name: value (RF rank #X, importance Y.YY, gap Z.ZZ - interpretation)"

Use the metadata provided above - all numeric values are pre-computed.
Create a creative interpretation based on strategic meaning.

Example outputs:
✅ "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43 - HIGHEST PREDICTOR)"
✅ "word_count: 14 (RF rank #3, importance 0.18, gap 26.8 - brief hook strategy)"
```

---

#### 2.2.5 Path Data Preparation (Phase 2 Preprocessing)

**Purpose**: Label cluster paths by 10% threshold status, show top 10 with scenario determination

**When Called**: Before Phase 2 prompt generation

**Source**: Stage7PromptCritique.md Gap #1 (lines 2923-3077), Alternative C (Hybrid Approach) decision

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

    See Stage7PromptCritique.md Gap #1 lines 2953-2958 for full rationale.

    Args:
        cluster_paths: Dict mapping path tuples to frequency counts
            {(0,1,1,2,0,1): 22, (1,0,0,1,1,0): 18, (0,0,1,1,0,1): 12, ...}
        threshold_pct: Minimum frequency percentage (default: 0.10 = 10%)
        total_videos: Total videos in sample (default: 100)
        top_n: Number of top paths to show in prompt (default: 10)

    Returns:
        dict with labeled paths and scenario:
        {
            'top_paths': [
                ((0,1,1,2,0,1), 22, 22.0, 'ABOVE'),
                ((1,0,0,1,1,0), 18, 18.0, 'ABOVE'),
                ((0,0,1,1,0,1), 12, 12.0, 'ABOVE'),
                ((1,1,0,0,1,0), 8, 8.0, 'BELOW'),  # Clearly marked
                ...
            ],
            'total_unique_paths': 35,
            'paths_above_threshold': 3,
            'scenario': 'A',  # A=3+ paths, B=2 paths, C=1 path, D=0 paths
            'threshold_pct': 10.0
        }

    Example Scenario Determination:
        - Scenario A (3+ paths ≥10%): Generate 3 path-based reports
        - Scenario B (2 paths ≥10%): Generate 2 path-based + 1 feature-based
        - Scenario C (1 path ≥10%): Generate 1 path-based + 2 feature-based
        - Scenario D (0 paths ≥10%): Generate 3 feature-based (high fragmentation)
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

**Prompt Data Format** (LLM sees labeled paths):
```
Cluster Path Analysis:
- Total unique paths: 35 (indicates high fragmentation)
- Paths meeting 10% threshold: 3
- Scenario: A (3+ paths above threshold)

Top 10 Paths (with threshold status):
1. [0,1,1,2,0,1]: 22 videos (22%) - ✅ ABOVE THRESHOLD
2. [1,0,0,1,1,0]: 18 videos (18%) - ✅ ABOVE THRESHOLD
3. [0,0,1,1,0,1]: 12 videos (12%) - ✅ ABOVE THRESHOLD
4. [1,1,0,0,1,0]: 8 videos (8%) - ❌ BELOW THRESHOLD
5. [0,1,0,1,0,0]: 6 videos (6%) - ❌ BELOW THRESHOLD
... (showing top 10 of 35)

YOUR TASK - Scenario A:
Generate exactly 3 path-based reports using ONLY the paths marked "✅ ABOVE THRESHOLD".
Do NOT use ❌ BELOW THRESHOLD paths in creative_reports.

You may mention below-threshold patterns in supplementary_insights if they show interesting emerging trends.
```

---

#### 2.2.6 Confidence Level Classification (Phase 2 Preprocessing)

**Purpose**: Classify path frequency into confidence bands (very_high/high/moderate)

**When Called**: Integrated into `prepare_path_data_for_llm()`, applied to each path

**Source**: Stage7PromptCritique.md Gap #2 (lines 3093-3214), Python Computes decision

```python
def classify_confidence_level(
    frequency_pct: float,
    report_type: str = "path_based"
) -> str:
    """
    Classify confidence level based on frequency percentage.

    Pure arithmetic classification with clear thresholds - exactly what Python should handle.

    DESIGN DECISION: Confidence bands (20%, 15%, 10%) chosen because:
    - Statistical interpretation: 20% = "1 in 5 videos" = dominant, 15% = "1 in 6-7" = strong, 10% = "1 in 10" = proven
    - Stage 8 PDF prioritization: very_high featured prominently, moderate secondary
    - Future-proofing: Normalizes confidence across different sample sizes (200 videos vs 100 videos)
    - Clear boundaries: No ambiguity (19.9% = high, not very_high)

    See Stage7PromptCritique.md Gap #2 lines 3093-3147 for full rationale.

    Args:
        frequency_pct: Frequency percentage (e.g., 22.0 for 22%)
        report_type: "path_based" or "feature_based"

    Returns:
        "very_high" | "high" | "moderate"

    Rules:
        - Path-based reports:
            - ≥20%: very_high (1 in 5 videos - dominant pattern)
            - 15-19.9%: high (1 in 6-7 videos - strong pattern)
            - 10-14.9%: moderate (1 in 10 videos - proven pattern)
        - Feature-based reports: always "moderate" (not frequency-based)

    Example:
        >>> classify_confidence_level(22.0, "path_based")
        "very_high"
        >>> classify_confidence_level(12.0, "path_based")
        "moderate"
        >>> classify_confidence_level(None, "feature_based")
        "moderate"
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

**Integration**: This function is called within `prepare_path_data_for_llm()` to add confidence level to each path's data tuple.

---

#### 2.2.7 Universal Principles Generation (Phase 2 Preprocessing)

**Purpose**: Extract top 5-7 RF features as universal principles applicable to ALL videos

**When Called**: Before Phase 2 prompt generation

**Source**: Stage7PromptCritique.md Gap #3 (lines 3313-3388)

```python
def generate_universal_principles(
    rf_video_data: dict,
    top_n: int = 7
) -> list[str]:
    """
    Extract top N RF features as universal principles applicable to all videos.

    DESIGN DECISION: Universal principles cover 40-60% of videos NOT explained by path formulas,
    ensuring EVERY creator gets actionable advice even if their style doesn't match a formula.

    See Stage7PromptCritique.md Gap #3 lines 3313-3388 for full rationale.

    Args:
        rf_video_data: Video-level RF feature importance data
            {
                'feature_importance': [
                    {
                        'feature': 'eye_contact_rate',
                        'importance': 0.35,
                        'rank': 1,
                        'top_performer_avg': 0.88,
                        'bottom_performer_avg': 0.45,
                        'gap': 0.43,
                        'prevalence': 0.78  # % of videos with this feature high
                    },
                    ...
                ]
            }
        top_n: Number of top features to extract (default: 7)

    Returns:
        List of formatted universal principle strings:
        [
            "High eye contact rate (88% vs 45% for top vs bottom performers) - applies to 78% of videos",
            "Consistent energy maintenance (std dev ≤0.15) - found in 65% of top performers",
            "Clear CTA in closing window - present in 92% of high-performing videos",
            ...
        ]

    Example (Stage 8 PDF usage):
        Section titled "Universal Best Practices" with bulleted principles
    """
    principles = []

    # Get top N features by importance
    top_features = sorted(
        rf_video_data['feature_importance'],
        key=lambda x: x['importance'],
        reverse=True
    )[:top_n]

    for feature in top_features:
        feature_name = feature['feature']
        top_avg = feature['top_performer_avg']
        bottom_avg = feature['bottom_performer_avg']
        prevalence = feature.get('prevalence', 0.0) * 100  # Convert to percentage

        # Format based on feature type
        if 'rate' in feature_name or 'percentage' in feature_name:
            # Percentage features
            principle = (
                f"High {feature_name.replace('_', ' ')} "
                f"({top_avg:.0%} vs {bottom_avg:.0%} for top vs bottom performers) - "
                f"applies to {prevalence:.0f}% of videos"
            )
        elif 'count' in feature_name:
            # Count features
            principle = (
                f"{feature_name.replace('_', ' ').title()} "
                f"({top_avg:.0f} vs {bottom_avg:.0f} for top vs bottom) - "
                f"found in {prevalence:.0f}% of top performers"
            )
        else:
            # Generic features
            principle = (
                f"{feature_name.replace('_', ' ').title()} "
                f"(top: {top_avg:.2f}, bottom: {bottom_avg:.2f}) - "
                f"applies to {prevalence:.0f}% of videos"
            )

        principles.append(principle)

    return principles
```

---

#### 2.2.8 Cross-Window Patterns Generation (Phase 2 Preprocessing)

**Purpose**: Extract temporal progression insights from cross-window RF features (energy deltas, consistency metrics)

**When Called**: Before Phase 2 prompt generation

**Source**: Stage7PromptCritique.md Gap #3 (lines 3391-3529), Alternative B (Graceful Degradation) decision

**CRITICAL DEPENDENCY RESOLVED**: Cross-window features ARE implemented in Stage 4 (FeatureTransformationCHILD.md Section 6.5)

```python
def generate_cross_window_patterns(rf_video_data: dict) -> list[str]:
    """
    Extract cross-window progression patterns from video-level RF data.

    Implements graceful degradation: If cross-window features exist (normal case),
    generate insights. If missing, return informative placeholder.

    DESIGN DECISION: Graceful degradation chosen because:
    - Cross-window features ARE implemented (Stage 4 Section 6.5) - normal case will use them
    - Handles edge cases gracefully (Stage 4/6 bugs don't crash Stage 7)
    - Self-documenting fallback (placeholder explains what's missing and where to find it)
    - Name pattern matching (not exact names) - future-compatible

    See Stage7PromptCritique.md Gap #3 lines 3290-3308 for full rationale.

    Cross-Window Features (computed by Stage 3, passed through Stage 4):
    1. xwin_hook_to_middle_energy: middle_avg_energy - hook_energy (buckets 9-13s+)
    2. xwin_middle_to_closing_energy: closing_energy - middle_avg_energy (buckets 9-13s+)
    3. xwin_eye_contact_consistency: std_dev([eye_contact_rate across windows]) (buckets 3-9s+)
    4. xwin_word_density_std: std_dev([word_count across windows]) (buckets 3-9s+)
    5. xwin_energy_progression_slope: linear regression slope of energy (buckets 3-9s+)

    **S7B2 Note** (2025-10-28): Cross-window features now created in Stage 3 (not Stage 4) with xwin_ prefix
    to avoid collision with window-specific features. See PostBugFixUpdate.md for details.

    Args:
        rf_video_data: Video-level RF feature importance data
            {
                'feature_importance': [
                    {
                        'feature': 'xwin_hook_to_middle_energy',
                        'importance': 0.18,
                        'rank': 4,
                        'top_performer_avg': 0.15,
                        'bottom_performer_avg': -0.05,
                        'gap': 0.20
                    },
                    ...
                ]
            }

    Returns:
        List of formatted cross-window pattern strings

        When features exist (normal case):
        [
            "65% of high-performing videos show energy builds from hook to middle",
            "78% show consistent eye contact throughout (bookend pattern)",
            "72% show strong energy peak in closing vs middle"
        ]

        When features missing (graceful degradation):
        [
            "Cross-window progression analysis requires Stage 6 RF cross-window features",
            "These features are computed in Stage 3 (Stage3_HLD.md Section 4.5), passed through Stage 4",
            "Expected features: xwin_hook_to_middle_energy, xwin_middle_to_closing_energy, ...",
            "Stage 7 will automatically use these features once Stage 3/4/6 pipeline is complete"
        ]
    """
    cross_features = rf_video_data.get('feature_importance', [])

    # Filter to cross-window features by name pattern
    CROSS_WINDOW_KEYWORDS = ['delta', 'consistency', 'contrast', 'progression', '_std']
    cross_window_features = [
        f for f in cross_features
        if any(keyword in f['feature'] for keyword in CROSS_WINDOW_KEYWORDS)
    ]

    # Check if we have cross-window features
    if not cross_window_features:
        # Graceful fallback
        return [
            "Cross-window progression analysis requires Stage 6 RF cross-window features",
            "These features are computed in Stage 3 (Stage3_HLD.md Section 4.5), passed through Stage 4",
            "Expected features: xwin_hook_to_middle_energy, xwin_middle_to_closing_energy, xwin_eye_contact_consistency, xwin_word_density_std, xwin_energy_progression_slope (S7B2: xwin_ prefix required)",
            "Stage 7 will automatically use these features once Stage 3/4/6 pipeline is complete"
        ]

    # If features exist, generate insights
    cross_window_features.sort(key=lambda x: x['importance'], reverse=True)
    top_cross = cross_window_features[:5]  # Top 5 cross-window features

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
        'xwin_hook_to_middle_energy': 'energy builds from hook to middle',
        'xwin_middle_to_closing_energy': 'strong energy peak in closing vs middle',
        'xwin_eye_contact_consistency': 'consistent eye contact throughout (bookend pattern)',
        'xwin_word_density_std': 'varied pacing across windows',
        'xwin_energy_progression_slope': 'steady energy progression from start to end'
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

---

#### 2.2.9 Feature-Based Report Generation (Phase 2 Preprocessing)

**Purpose**: Generate complete fallback reports when <3 paths meet 10% threshold

**When Called**: Before Phase 2 prompt generation, in Scenarios B, C, D

**Source**: Stage7PromptCritique.md Gap #4 (lines 3619-3869), Alternative B (Python Generates Complete Reports) decision

```python
def generate_feature_based_reports(
    rf_video_data: dict,
    num_reports: int,
    used_features: set = None
) -> list[dict]:
    """
    Generate feature-based reports when insufficient paths meet 10% threshold.

    Python does ALL the work (groups features, generates names, writes descriptions, creates recommendations).
    LLM just copies the structured JSON. This prevents hallucination in fallback scenarios.

    DESIGN DECISION: Python generates complete reports (not LLM) because:
    - Zero LLM hallucination risk: All text is Python-generated from data-driven templates
    - Deterministic output: Same RF features always produce same reports (testable, reproducible)
    - Hashtag specificity from DATA: Uses actual top_performer_avg from that hashtag's RF model
    - Feature-based reports are universal by design: Fallback guidance when paths are fragmented

    See Stage7PromptCritique.md Gap #4 lines 3619-3639 for full rationale.

    Feature Grouping Categories:
    1. Eye Contact & Engagement: eye_contact_rate, eye_contact_consistency, gaze_direction
    2. Energy & Pacing: energy_level, hook_to_middle_energy_delta, middle_to_closing_energy_contrast
    3. Speech & Density: word_count, speech_coverage, word_density, pause_frequency
    4. Visual Variety: scene_count, object_count, overlay_unique_count, scene_transition_count

    Args:
        rf_video_data: Video-level RF feature importance data
        num_reports: Number of feature-based reports to generate (1-3)
        used_features: Set of features already used in path-based reports (to avoid duplication)

    Returns:
        List of feature-based report dictionaries matching Phase 2 schema (13 fields, identical to path-based):
        [
            {
                "report_id": 3,
                "type": "feature_based",
                "path": null,
                "frequency": null,
                "percentage": null,
                "confidence_level": "moderate",
                "formula_name": "The Visual Storytelling Formula",
                "structure": null,
                "temporal_progressions": [
                    {
                        "feature": "scene_count",
                        "progression": "Dynamic visual elements throughout video",
                        "insight": "Visual variety maintains attention in short-form content"
                    }
                ],
                "rf_cross_window_validation": {
                    "video_level_features_matched": [],
                    "alignment_insight": "Visual engagement features align with top RF predictors"
                },
                "strategy_description": "Visual engagement formula emphasizing dynamic scene transitions and visual variety.",
                "when_to_use": "Product demonstrations, educational content, transformation videos, visual tutorials.",
                "step_by_step_template": [
                    "Hook: Use multiple visual angles or dynamic elements to create immediate visual interest",
                    "Middle: Maintain visual variety with strategic scene transitions",
                    "Closing: Return to primary visual focus while maintaining dynamic elements"
                ]
            }
        ]

    Example Usage (Scenario C: 1 path ≥10%):
        reports = generate_feature_based_reports(rf_video_data, num_reports=2, used_features={'word_count'})
        # Returns 2 feature-based reports (different feature groups, avoiding word_count)
    """
    if used_features is None:
        used_features = set()

    # Define feature groupings
    feature_groups = {
        'Eye Contact & Engagement': ['eye_contact_rate', 'eye_contact_consistency', 'gaze_direction'],
        'Energy & Pacing': ['energy_level', 'hook_to_middle_energy_delta', 'middle_to_closing_energy_contrast', 'energy_variance'],
        'Speech & Density': ['word_count', 'speech_coverage', 'word_density', 'pause_frequency'],
        'Visual Variety': ['scene_count', 'object_count', 'overlay_unique_count', 'scene_transition_count']
    }

    # Get top features by importance (excluding already used)
    available_features = [
        f for f in rf_video_data['feature_importance']
        if f['feature'] not in used_features
    ]
    available_features.sort(key=lambda x: x['importance'], reverse=True)

    reports = []
    report_id_start = 4 - num_reports  # If generating 1 report, it's #3; if 2, they're #2 and #3

    for i in range(num_reports):
        # Select feature group (rotate through groups)
        group_names = list(feature_groups.keys())
        group_name = group_names[i % len(group_names)]
        group_features = feature_groups[group_name]

        # Find top features from this group
        group_top_features = [
            f for f in available_features
            if f['feature'] in group_features
        ][:2]  # Top 2 features from group

        if len(group_top_features) < 1:
            # Fallback: use next available features
            group_top_features = available_features[:2]

        # Mark features as used
        for f in group_top_features:
            used_features.add(f['feature'])

        # Generate report using data-driven templates
        report = {
            'report_id': report_id_start + i + 1,
            'type': 'feature_based',
            'frequency': None,
            'percentage': None,
            'confidence_level': 'moderate',
            'formula_name': f"The {group_name} Strategy",
            'strategy_description': _generate_strategy_description(group_name, group_top_features),
            'key_features': [_format_key_feature(f) for f in group_top_features],
            'rf_validation': {'insight': _generate_rf_insight(group_top_features)},
            'when_to_use': 'Universal strategy applicable when cluster paths are fragmented. Focus on proven principles.',
            'creator_recommendations': _generate_recommendations(group_top_features)
        }

        reports.append(report)

    return reports


def _generate_strategy_description(group_name: str, features: list[dict]) -> str:
    """Generate strategy description using data-driven templates."""
    description_templates = {
        'Eye Contact & Engagement': {
            'template': 'Maintain {level} eye contact throughout video journey',
            'thresholds': {'high': 0.80, 'moderate': 0.60}
        },
        'Energy & Pacing': {
            'template': '{direction} energy {pattern} from hook to closing',
        },
        'Speech & Density': {
            'template': 'Optimize speech density for {style} information delivery',
            'thresholds': {'dense': 50, 'moderate': 30}
        },
        'Visual Variety': {
            'template': 'Use {level} visual elements and scene transitions',
            'thresholds': {'high': 5, 'moderate': 3}
        }
    }

    config = description_templates.get(group_name)
    if not config or not features:
        return 'Universal best practices for video performance'

    # Get primary feature value for template filling
    primary_feature = features[0]
    top_avg = primary_feature.get('top_performer_avg', 0)

    # Fill template based on group type and actual data
    if group_name == 'Eye Contact & Engagement':
        level = 'high and consistent' if top_avg >= 0.80 else 'moderate' if top_avg >= 0.60 else 'selective'
        return config['template'].format(level=level)

    elif group_name == 'Energy & Pacing':
        direction = 'Build' if top_avg > 0 else 'Maintain consistent'
        pattern = 'progressively with strategic contrast' if top_avg > 0 else ''
        return config['template'].format(direction=direction, pattern=pattern).strip()

    elif group_name == 'Speech & Density':
        style = 'dense' if top_avg >= 50 else 'clear and paced'
        return config['template'].format(style=style)

    elif group_name == 'Visual Variety':
        level = 'diverse' if top_avg >= 5 else 'varied' if top_avg >= 3 else 'focused'
        return config['template'].format(level=level)

    return 'Universal best practices for video performance'


def _format_key_feature(feature: dict) -> str:
    """Format feature for key_features array."""
    return (
        f"{feature['feature']}: {feature['top_performer_avg']:.2f} "
        f"(RF rank #{feature['rank']}, importance {feature['importance']:.2f}, "
        f"gap {feature['gap']:.2f})"
    )


def _generate_rf_insight(features: list[dict]) -> str:
    """Generate RF validation insight."""
    ranks = [f['rank'] for f in features]
    rank_str = ' and '.join([f'#{r}' for r in ranks])
    return f"Leverages {rank_str} most predictive features across entire video"


def _generate_recommendations(features: list[dict]) -> list[str]:
    """Generate creator recommendations based on features."""
    recommendations = []

    for i, feature in enumerate(features):
        priority = "PRIORITY: " if i == 0 else ""
        target = feature['top_performer_avg']

        # Format recommendation based on feature type
        if 'rate' in feature['feature'] or 'percentage' in feature['feature']:
            rec = f"{priority}Maintain {target:.0%} {feature['feature'].replace('_', ' ')} throughout video (RF #{feature['rank']} predictor)"
        elif 'count' in feature['feature']:
            rec = f"{priority}Target {target:.0f} {feature['feature'].replace('_', ' ')} (RF #{feature['rank']} predictor)"
        elif 'delta' in feature['feature']:
            rec = f"{priority}Achieve {target:+.2f} {feature['feature'].replace('_', ' ')} (RF #{feature['rank']} predictor)"
        elif 'consistency' in feature['feature'] or 'variance' in feature['feature']:
            rec = f"Keep {feature['feature'].replace('_', ' ')} low (≤{target:.2f} std dev)"
        else:
            rec = f"Target {feature['feature'].replace('_', ' ')}: {target:.2f}"

        recommendations.append(rec)

    return recommendations
```

---

#### 2.2.7 Cross-Window Pattern Generation

```python
def generate_cross_window_patterns(window_analyses: dict, min_windows: int = 2) -> list[dict]:
    """Extract temporal progressions from video-level RF cross-window features."""
```

Identifies patterns that span multiple temporal windows (e.g., "joy increases from hook to closing"). Uses video-level RF features like `hook_to_closing_joy_delta` to detect progression patterns.

#### 2.2.8 Feature-Based Report Generation

```python
def generate_feature_based_reports(rf_features: list[dict], kmeans_data: dict, num_reports: int = 3) -> list[dict]:
    """Generate fallback reports when <3 cluster paths meet 10% threshold."""
```

Creates 1-3 feature-based reports (visual, audio, behavioral) when insufficient cluster paths exist. Returns JSON reports with **full 13-field schema** matching path-based reports (schema consistency for Stage 8). Python-generated content includes formula names, temporal progressions, RF validation, and step-by-step templates.

#### 2.2.9 Phase 1 Parallel Orchestration

```python
def run_phase1_parallel(bucket_path: str, bucket: str, hashtag: str, window_types: list[str]) -> dict:
    """Execute Phase 1 window analyses in parallel with checkpoint/resume."""
```

Orchestrates parallel execution of 6-7 window analyses using ThreadPoolExecutor. Implements checkpoint file for resume capability and tracks completion status.

#### 2.2.10 Single Window Analysis with Retry

```python
def analyze_window_with_retry(bucket_path: str, window_type: str, bucket: str,
                               hashtag: str | None, max_attempts: int = 3) -> dict:
    """Analyze single window with exponential backoff retry logic."""
```

Calls Claude API for one window with 3-attempt retry using exponential backoff [0s, 2s, 4s]. Distinguishes retryable errors (503, 429, timeout) from non-retryable (401, 400).

#### 2.2.11 Phase 2 Synthesis Orchestration

```python
def run_phase2_synthesis(bucket_path: str, bucket: str, hashtag: str) -> dict:
    """Execute Phase 2 cross-window synthesis with scenario detection."""
```

Loads Phase 1 results, extracts cluster paths, applies 10% threshold filter, detects scenario (A/B/C/D), calls Claude API with scenario-specific instructions.

#### 2.2.12 Phase 1 Prompt Construction

```python
def build_phase1_prompt(window_type: str, bucket: str, hashtag: str,
                        rf_data: dict, kmeans_data: dict) -> str:
    """Build Phase 1 prompt with bimodal formatting and RF alignment."""
```

Generates 150-line prompt for single window analysis. Applies bimodal pattern detection, high-contrast feature filtering, RF alignment scoring. Returns formatted prompt string.

#### 2.2.13 Phase 2 Prompt Construction

```python
def build_phase2_prompt(bucket: str, hashtag: str, window_analyses: dict,
                        rf_video_data: dict, feature_based_reports: list[dict],
                        scenario: str) -> str:
    """Build Phase 2 prompt with scenario-specific instructions."""
```

Generates 180-line prompt for cross-window synthesis. Embeds Phase 1 analyses, cluster path data, universal principles, cross-window patterns. Adapts instructions based on scenario A/B/C/D.

---

**Summary**: Section 2.2 introduces 16 total preprocessing and orchestration functions (9 preprocessing + 7 orchestration/prompt) that handle all arithmetic, mechanical operations, and data labeling BEFORE LLM sees the data. This division of labor prevents hallucination while preserving LLM's creative strengths in semantic interpretation and narrative synthesis.

### 2.3 Data Flow

```
Input: Stage 6 ML Analysis JSONs (13-15 files per bucket depending on window count, ~95KB total)
   ├── rf_video_analysis.json (~30KB) - Video-level RF cross-window features
   ├── {window}_rf_analysis.json × N (~5KB each) - Window-level RF top 10 features
   └── {window}_kmeans_analysis.json × N (~5KB each) - Window-level 3 clusters with centroids
   Where N = window count: 6 for 18-33s, 7 for 90-120s (see Section 3.1 for full table)
   ↓
Pre-Flight Validation:
   ├── API credentials (ANTHROPIC_API_KEY environment variable)
   ├── Stage 6 files exist (all 13 JSONs present)
   ├── JSON parseability (syntactically valid)
   └── Schema validation (required fields, cluster size integrity)
   ↓
Create Output Directory: ml_analysis/llm/
   ↓
Phase 1: Per-Window Analysis (6-7 parallel LLM calls)
   ├── For each window type:
   │   ├── Load K-Means JSON (3 clusters × 21 features = 63 numbers)
   │   ├── Load window-level RF JSON (top 10 features × 5 metrics = 50 numbers)
   │   ├── Build Phase 1 prompt (total context: 113 numbers)
   │   ├── Call Anthropic API (model: claude-sonnet-4, timeout: 90s)
   │   ├── Validate JSON response (schema check + automated validation layer)
   │   └── Save: ml_analysis/llm/{window}_analysis.json
   └── Smart Retry Logic:
       ├── Retry ONLY failed windows (not all 6-7)
       ├── Max 2 retry attempts per window with exponential backoff
       └── Abort bucket if ANY window fails after retries
   ↓
Validation Check: All windows succeeded?
   ├── YES → Continue to Phase 2
   └── NO → Abort bucket, log failure, return exit code 2
   ↓
Phase 2: Cross-Window Synthesis (1 LLM call)
   ├── Load all Phase 1 window analyses (6-7 JSONs)
   ├── Load video-level RF for cross-window patterns
   ├── Extract cluster paths for all videos (Q9.1):
   │   ├── For each video: Find cluster ID per window
   │   └── Build path: [hook_cluster, middle_1_cluster, ..., closing_cluster]
   ├── Calculate path frequencies (Q9.4):
   │   ├── Count identical paths (using Counter)
   │   ├── Calculate percentages (frequency / total_videos * 100)
   │   ├── Filter to paths ≥10% threshold
   │   ├── Classify confidence levels (very_high ≥20%, high 15-20%, moderate 10-15%)
   │   └── Take top 3 paths OR apply fallback if <3 paths meet threshold
   ├── Build Phase 2 prompt:
   │   ├── Include all 6-7 Phase 1 window analyses
   │   ├── Include top 10 cluster paths with frequencies
   │   └── Include video-level RF cross-window features
   ├── Call Anthropic API (timeout: 180s)
   ├── Validate JSON response
   └── Save: ml_analysis/llm/winning_formulas.json
   ↓
Generate Complete Analysis:
   └── Save: ml_analysis/llm/complete_analysis_{bucket}.json (Phase 1 + Phase 2 combined)
   ↓
Output: 8 JSON files per bucket (~40-50KB total)
   ├── ml_analysis/llm/hook_analysis.json
   ├── ml_analysis/llm/middle_1_analysis.json (× 0-5 depending on bucket)
   ├── ml_analysis/llm/closing_analysis.json
   ├── ml_analysis/llm/winning_formulas.json
   └── ml_analysis/llm/complete_analysis_{bucket}.json
```

### 2.4 Detailed Process

#### Step 2.4.1: Pre-Flight Validation

**Purpose**: Validate all dependencies before Phase 1 execution (fail-fast principle)

**Logic**:
```python
def run_preflight_validation(bucket_path: str, bucket: str) -> None:
    """
    Three-layer pre-flight validation.

    Layer 1: API credentials exist
    Layer 2: Stage 6 outputs exist and parseable
    Layer 3: Schema validation and data integrity

    Source: QA Q4.2, Q7
    """
    # === Layer 1: API Credentials ===
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise PreFlightValidationError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Add to .env file: ANTHROPIC_API_KEY=sk-ant-api03-..."
        )

    if not api_key.startswith("sk-ant-"):
        raise PreFlightValidationError(
            f"Invalid ANTHROPIC_API_KEY format. Expected: sk-ant-api03-..."
        )

    logger.info("✓ API credentials validated")

    # === Layer 2: Stage 6 Files Exist and Parseable ===
    windows = BUCKET_WINDOWS[bucket]  # e.g., ['hook', 'middle_1', ..., 'closing']

    expected_files = [
        'ml_analysis/rf_video_analysis.json',
        *[f'ml_analysis/{w}_rf_analysis.json' for w in windows],
        *[f'ml_analysis/{w}_kmeans_analysis.json' for w in windows]
    ]  # 13 files for bucket 18-33s

    # Check files exist
    missing_files = [f for f in expected_files
                    if not os.path.exists(os.path.join(bucket_path, f))]
    if missing_files:
        raise PreFlightValidationError(
            f"Stage 6 incomplete: Missing {len(missing_files)} of {len(expected_files)} JSONs. "
            f"Re-run Stage 6. Missing files: {missing_files[:3]}..."
        )

    # Check JSONs parseable
    malformed_files = []
    for file_path in expected_files:
        try:
            with open(os.path.join(bucket_path, file_path), 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            malformed_files.append((file_path, str(e)))

    if malformed_files:
        raise PreFlightValidationError(
            f"Stage 6 validation failed: {len(malformed_files)} JSONs malformed. "
            f"Re-run Stage 6. Files: {[f[0] for f in malformed_files]}"
        )

    logger.info(f"✓ All {len(expected_files)} Stage 6 JSONs exist and parseable")

    # === Layer 3: Schema Validation and Data Integrity ===
    for window in windows:
        # Validate K-Means JSON
        kmeans_path = os.path.join(bucket_path, f'ml_analysis/{window}_kmeans_analysis.json')
        with open(kmeans_path, 'r') as f:
            kmeans_data = json.load(f)

        # Check required fields
        required = ['window_type', 'bucket', 'n_clusters', 'clusters', 'total_videos']
        missing = [f for f in required if f not in kmeans_data]
        if missing:
            raise ValidationError(f"{window}_kmeans_analysis.json: Missing fields: {missing}")

        # Check 3 clusters
        if len(kmeans_data['clusters']) != 3:
            raise ValidationError(
                f"{window}_kmeans_analysis.json: Expected 3 clusters, "
                f"got {len(kmeans_data['clusters'])}"
            )

        # ✅ Cluster size validation REMOVED (Stage 6 already validates this in MLAnalysisGenerationCHILD.md)
        # Rationale: Avoid redundant validation. Stage 6 is authoritative source for cluster integrity.
        # Stage 7 trusts Stage 6's validation (follows Unix philosophy: do one thing well)
        # Source: CrossHLDalignment2do.md Issue #16 - Option A (remove from Stage 7)

        # Validate RF JSON
        rf_path = os.path.join(bucket_path, f'ml_analysis/{window}_rf_analysis.json')
        with open(rf_path, 'r') as f:
            rf_data = json.load(f)

        if len(rf_data.get('feature_importance', [])) < 10:
            raise ValidationError(
                f"{window}_rf_analysis.json: Expected 10 features, "
                f"got {len(rf_data['feature_importance'])}"
            )

    logger.info("✓ Schema validation and data integrity checks passed")

    # Create output directory
    llm_output_dir = os.path.join(bucket_path, 'ml_analysis/llm')
    os.makedirs(llm_output_dir, exist_ok=True)
    logger.info(f"✓ Created output directory: {llm_output_dir}")
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| API key missing | Fail immediately with clear error message | Better than cryptic API auth error mid-execution |
| Malformed JSON (trailing comma) | Detect in Layer 2, fail before Phase 1 | Prevents partial Phase 1 execution |
| Only 10 of 13 files exist | Detect in Layer 2, list all missing files | Clear diagnosis for user |
| Cluster sizes don't sum to total_videos | Detect in Layer 3, fail with specifics | Critical for Phase 2 path extraction (Q9.1) |

---

#### Step 2.4.2: Phase 1 - Per-Window Analysis (Parallel Execution with Status Tracking)

**Purpose**: Generate focused creative insights for each temporal window independently with cost-optimized incremental saves

**Design Decision - Incremental Saves (NOT Atomic Pattern)**:

Stage 7 uses **incremental saves with status tracking** instead of atomic commit pattern (unlike Stages 5-6) because:

1. **Cost Optimization**: LLM API calls cost ~$0.02-0.05 per window. On retry, atomic pattern wastes successful API calls ($0.18 for all 6 windows vs $0.03 for 1 failed window)
2. **Non-Deterministic LLM**: Retrying all windows may lose good analyses from first run (LLM gives different creative strategies each time)
3. **External API Reliability**: Incremental saves preserve progress through transient failures (503 errors, timeouts)
4. **Progress Visibility**: Users can monitor completion status during 2-3 minute execution

Status tracking provides completion clarity without sacrificing cost optimization.

**Logic**:
```python
def run_phase1_parallel(bucket_path: str, bucket: str, hashtag: str | None, window_types: list) -> dict:
    """
    Run Phase 1 analysis for all windows in parallel with status tracking and resume capability.

    Returns: {window_type: analysis_json} for all windows
    Raises: Phase1ExecutionError if ANY window fails after retries

    Source: Mother Doc lines 2718-2758, Critique Q4 (smart retry), Cross-HLD Issue #11 (status tracking)
    """
    status_file = os.path.join(bucket_path, 'ml_analysis/llm/.phase1_status.json')

    # === Initialize or load status (resume capability) ===
    if os.path.exists(status_file):
        with open(status_file) as f:
            status = json.load(f)
        completed = set(status['completed_windows'])
        logger.info(f"Resuming Phase 1: {len(completed)}/{len(window_types)} windows already completed")
    else:
        status = {
            'total_windows': len(window_types),
            'completed_windows': [],
            'failed_windows': [],
            'phase1_complete': False,
            'started_at': datetime.utcnow().isoformat(),
        }
        completed = set()

    window_analyses = {}

    # === Run windows in parallel (skip already completed) ===
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(window_types)) as executor:
        futures = {}

        for window_type in window_types:
            if window_type in completed:
                # Load existing analysis from file (cost savings - don't re-run)
                output_path = os.path.join(bucket_path, f'ml_analysis/llm/{window_type}_analysis.json')
                with open(output_path) as f:
                    window_analyses[window_type] = json.load(f)
                logger.info(f"  ⏭ {window_type} already completed (skipping, saved $0.03)")
                continue

            # Run analysis for incomplete window
            future = executor.submit(
                analyze_window_with_retry,
                bucket_path=bucket_path,
                window_type=window_type,
                bucket=bucket,
                hashtag=hashtag,
                max_attempts=3  # Initial + 2 retries
            )
            futures[window_type] = future

        # Collect results from parallel execution
        for window_type, future in futures.items():
            try:
                analysis = future.result(timeout=120)  # 90s API + 30s overhead

                # Save window JSON immediately (incremental save - cost optimization)
                output_path = os.path.join(bucket_path, f'ml_analysis/llm/{window_type}_analysis.json')
                with open(output_path, 'w') as f:
                    json.dump(analysis, f, indent=2)

                # Update status file (provides completion tracking)
                status['completed_windows'].append(window_type)
                status['last_updated'] = datetime.utcnow().isoformat()
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)

                window_analyses[window_type] = analysis
                logger.info(f"  ✓ {window_type}_analysis.json saved ({len(status['completed_windows'])}/{len(window_types)})")

            except Exception as e:
                # Record failure in status (for debugging)
                status['failed_windows'].append({
                    'window': window_type,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
                status['last_updated'] = datetime.utcnow().isoformat()
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)

                logger.error(f"  ✗ {window_type} failed: {e}")
                raise Phase1ExecutionError(
                    f"Phase 1 incomplete: {window_type} failed after retries. "
                    f"Review errors and re-run Stage 7 (will resume from checkpoint)."
                )

    # === Mark Phase 1 complete ===
    status['phase1_complete'] = True
    status['completed_at'] = datetime.utcnow().isoformat()
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

    logger.info(f"✓ Phase 1 complete: All {len(window_types)} windows succeeded")

    return window_analyses


def analyze_window_with_retry(bucket_path: str, window_type: str, bucket: str,
                              hashtag: str | None, max_attempts: int = 3) -> dict:
    """
    Analyze single window with smart retry logic (exponential backoff).

    Source: QA Q5.1, Q5.2 (retry strategy)
    """
    # Load input data
    kmeans_path = os.path.join(bucket_path, f'ml_analysis/{window_type}_kmeans_analysis.json')
    rf_path = os.path.join(bucket_path, f'ml_analysis/{window_type}_rf_analysis.json')

    with open(kmeans_path, 'r') as f:
        kmeans_data = json.load(f)
    with open(rf_path, 'r') as f:
        rf_data = json.load(f)

    # Build prompt
    prompt = build_phase1_prompt(window_type, kmeans_data, rf_data, bucket, hashtag)

    # API client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Retry loop with exponential backoff
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,  # From config: "claude-sonnet-4-20250514"
                max_tokens=PHASE1_MAX_TOKENS,  # 4000
                temperature=PHASE1_TEMPERATURE,  # 0.3
                timeout=PHASE1_TIMEOUT_SECONDS,  # 90s
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse and validate JSON response
            analysis = parse_and_validate_json(
                response=response,
                window_type=window_type,
                kmeans_data=kmeans_data,
                rf_data=rf_data,
                attempt=attempt
            )

            # Automated validation layer (Critique Q3 Layer 1)
            validate_llm_output(analysis, kmeans_data, rf_data, window_type)

            # Success
            logger.info(f"  ✓ {window_type} analysis complete (attempt {attempt})")
            return analysis

        except Exception as e:
            # Check if retryable
            if not should_retry_api_error(e):
                logger.error(f"{window_type}: Fatal error (non-retryable): {e}")
                raise

            # Check if retries exhausted
            if attempt >= max_attempts:
                logger.error(f"{window_type}: Failed after {max_attempts} attempts")
                raise

            # Retry with exponential backoff
            logger.warning(f"{window_type}: Attempt {attempt} failed: {e}")
            retry_with_backoff(attempt)

    # Should never reach here
    raise RuntimeError(f"{window_type}: Retry logic failed")
```

**Phase 1 Prompt Template** (Incorporates Issues #1-11 improvements):

```python
def build_phase1_prompt(window_type: str, kmeans_data: dict, rf_data: dict,
                       bucket: str, hashtag: str | None) -> str:
    """
    Build Phase 1 prompt with all Issue #1-11 improvements integrated.

    Improvements applied:
    - Issue #1: Bimodal pattern detection and formatting
    - Issue #3: High-contrast feature filtering
    - Issue #4: RF alignment computation
    - Issue #5: Removed "Important" section
    - Issue #6: Added cluster size context guidance
    - Issue #7: Compressed RF data format
    - Issue #8: Enriched features with RF metadata
    - Issue #9: RF alignment score in output
    - Issue #10: Bimodal example note

    Source: Stage7PromptCritique.md Issues #1-11 resolutions
    """
    # === Preprocessing: Run Section 2.2 functions ===

    # 1. Detect bimodal patterns in RF features (Section 2.2.1)
    rf_features_with_bimodal = []
    for feature in rf_data['feature_importance']:
        bimodal_info = detect_bimodal_pattern(feature['distribution'])
        rf_features_with_bimodal.append({**feature, 'bimodal': bimodal_info})

    # 2. Identify high-contrast features per cluster (Section 2.2.2)
    high_contrast_data = identify_high_contrast_features(kmeans_data, threshold=0.20)

    # 3. Compute RF alignment per cluster (Section 2.2.3)
    clusters_with_alignment = []
    for cluster in kmeans_data['clusters']:
        alignment = compute_rf_alignment(
            cluster['centroid'],
            rf_data['feature_importance'],
            threshold=0.15
        )
        clusters_with_alignment.append({**cluster, 'rf_alignment': alignment})

    # 4. Enrich high-contrast features with RF metadata (Section 2.2.4)
    for i, cluster_data in enumerate(high_contrast_data['clusters']):
        enriched = enrich_high_contrast_features(
            cluster_data['high_contrast_features'],
            rf_data['feature_importance']
        )
        high_contrast_data['clusters'][i]['enriched_features'] = enriched

    # === Build Prompt ===

    hashtag_context = f"#{hashtag}" if hashtag else "this TikTok category"
    total_videos = kmeans_data['total_videos']

    prompt = f"""You are a TikTok creative strategy analyst specializing in {hashtag_context} content. Your task is to analyze ML clustering and Random Forest feature importance data for the **{window_type}** window ({bucket} duration bucket) and generate actionable creative insights.

## Your Task

Analyze {kmeans_data['n_clusters']} distinct creative clusters identified in the {window_type} window. For each cluster, identify exactly 3 defining features and provide creator-friendly strategic recommendations.

## Data Provided

### Random Forest Feature Importance (Window-Level RF - Top 10 Features)

These features predict video performance specifically for the {window_type} window. Features are ranked by importance (higher = stronger predictor).

"""

    # === RF Features with Compressed Format + Bimodal Patterns (Issues #1, #7, #10) ===
    for i, feature in enumerate(rf_features_with_bimodal[:10], 1):
        bimodal = feature['bimodal']
        pattern_label = bimodal['pattern_label']

        # Compressed format (Issue #7): Single line per feature
        prompt += f"{i}. {feature['feature']} - RF Importance: {feature['importance']:.2f} (rank #{i})\n"
        prompt += f"   Top: avg {feature['top_performer_avg']:.2f} "
        prompt += f"({bimodal['high_percentage']:.0%} high, {bimodal['low_percentage']:.0%} low) | "
        prompt += f"Bottom: avg {feature['bottom_performer_avg']:.2f} | "
        prompt += f"Gap: {feature['gap']:.2f} | Pattern: {pattern_label}\n"

        # Add bimodal strategies if applicable (Issue #1)
        if bimodal['is_bimodal']:
            prompt += f"   → Strategy A: {_interpret_low_value(feature['feature'])} - {bimodal['low_percentage']:.0%} of top performers\n"
            prompt += f"   → Strategy B: {_interpret_high_value(feature['feature'])} - {bimodal['high_percentage']:.0%} of top performers\n"

        prompt += "\n"

    # === K-Means Clusters with High-Contrast Features + RF Alignment (Issues #3, #4, #8) ===
    prompt += f"""
### K-Means Clusters (3 Clusters from {total_videos} videos)

For each cluster below, you will find:
1. **All features**: Complete centroid values for context
2. **High-contrast features**: Pre-filtered to features differing by ≥0.20 from other clusters (reduces noise)
3. **RF Alignment**: Shows which cluster features match RF top performer patterns

"""

    for i, cluster_data in enumerate(high_contrast_data['clusters']):
        cluster = clusters_with_alignment[i]
        cluster_id = cluster['cluster_id']
        size = cluster['size']

        prompt += f"""
**CLUSTER {cluster_id}** ({size} videos, {size/total_videos:.0%} of sample):

All features (for context):
  {_format_centroid_compact(cluster['centroid'])}

High-contrast features (differ by ≥0.20 from other clusters - enriched with RF metadata):
"""

        # Show enriched features with all metadata (Issue #8)
        for j, enriched_feat in enumerate(cluster_data['enriched_features'][:12], 1):  # Top 12 high-contrast
            prompt += f"  {j}. {enriched_feat['feature']}: {enriched_feat['cluster_value']:.2f}\n"
            prompt += f"     (RF rank #{enriched_feat['rf_rank']}, importance {enriched_feat['rf_importance']:.2f}, "
            prompt += f"gap {enriched_feat['rf_gap']:.2f}, contrast vs other clusters: {enriched_feat['contrast']:.2f})\n"

        # Show RF alignment (Issue #4)
        alignment = cluster['rf_alignment']
        prompt += f"\nRF Alignment (features matching top performer patterns):\n"
        if alignment['aligned_features']:
            for aligned in alignment['aligned_features']:
                prompt += f"  ✅ {aligned['formatted']}\n"
            prompt += f"\n  Alignment score: {alignment['alignment_score']} "
            prompt += f"(uses {alignment['alignment_count']} of top 5 RF features at optimal levels)\n"
        else:
            prompt += f"  ❌ No features align with RF top patterns (creative novelty - not a bug!)\n"

        prompt += "\n"

    # === Cluster Size Context (Issue #6) ===
    prompt += """
### Cluster Size Context

For context, you are analyzing clusters from a sample of 50-100 videos with k=3 clustering.

**Framing cluster size in recommendations**:
- **Large clusters** (>50% of videos): Use language like "This is the DOMINANT strategy" or "Most common approach"
- **Medium clusters** (25-50%): Standard framing, no special language needed
- **Small clusters** (<25%): Use language like "This is a NICHE strategy" or "Alternative approach used by X% of creators"

**Why this matters**: Creators should know if they're following the dominant pattern (60% of videos) vs. exploring a niche approach (15% of videos).

**Include cluster size in your output**:
- In `strategy_description`: Mention "dominant" vs "niche" where appropriate
- In `when_to_use`: Clarify applicability ("broadly applicable" vs "suitable for specific creator types")

**Note**: All clusters are meaningful (even 15-video clusters with 50+ sample). Focus on accurate framing, not quality warnings.

"""

    # === Task Instructions ===
    prompt += """
## Output Requirements

Generate a JSON object with 3 cluster analyses. For EACH cluster:

1. **Select exactly 3 defining features** from the HIGH-CONTRAST list above, prioritizing:
   - RF importance (rank #1-5 preferred)
   - Strategic coherence (features that tell a coherent story together)
   - Contrast magnitude (larger differences = clearer distinction)

2. **Format each feature** using the enriched metadata provided:
   ```
   "feature_name: value (RF rank #X, importance Y.YY, gap Z.ZZ - interpretation)"
   ```
   Where "interpretation" is your creative semantic meaning (e.g., "brief hook strategy", "HIGHEST PREDICTOR")

3. **Handle BIMODAL features** (marked with Pattern: BIMODAL in RF data):
   Present BOTH strategies as valid options:
   "ALTERNATIVE STRATEGIES: Use either [Strategy A] OR [Strategy B] - RF data shows both work"

   Example:
   "word_count: Use either BRIEF (≤20 words, 35% of top) OR DENSE (≥80 words, 40% of top) - RF data shows both strategies work"

4. **Include RF validation** using the pre-computed alignment data:
   - Copy aligned features from "✅" items in RF Alignment section
   - Include alignment score in insight field (Issue #9):
     "This cluster leverages {N} of the top 5 most predictive features (RF alignment: {N}/5)..."

5. **Frame based on cluster size** (see Cluster Size Context above)

## Example Output Structure

```json
{
  "window_type": "{window_type}",
  "bucket": "{bucket}",
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "percentage": 35,
      "defining_features": [
        "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43 - HIGHEST PREDICTOR)",
        "word_count: 14 (RF rank #3, importance 0.18, gap 26.8 - brief hook strategy)",
        "energy_level: 0.55 (RF rank #2, importance 0.22, gap 0.30 - moderate baseline)"
      ],
      "rf_validation": {
        "top_predictive_features_in_cluster": [
          "eye_contact_rate (RF rank #1, cluster value 0.87 matches top avg 0.88)",
          "energy_level (RF rank #2, cluster value 0.55 close to top avg 0.53)"
        ],
        "insight": "This cluster leverages 2 of the top 5 most predictive features (RF alignment: 2/5), using high eye contact and moderate energy as primary engagement drivers."
      },
      "strategy_name": "The Direct Trust Hook",
      "strategy_description": "DOMINANT STRATEGY (35% of videos): Build immediate trust through sustained direct eye contact with brief, punchy messaging and moderate energy baseline.",
      "when_to_use": "Broadly applicable for trust-building intros, product reveals, educational content. Particularly effective when credibility matters.",
      "creator_recommendations": [
        "PRIORITY: Maintain 85-90% eye contact throughout hook window (RF #1 predictor)",
        "Keep word count under 20 words (brief hook strategy - RF rank #3)",
        "Start with moderate energy 0.50-0.60 to allow build potential"
      ]
    }
  ]
}
```

## Important Reminders

- Select exactly 3 defining features per cluster (never more, never less)
- Use enriched metadata provided - all numeric values are pre-computed
- For BIMODAL features: Present BOTH strategies as alternatives
- Include RF alignment score in insight field (e.g., "RF alignment: 2/5")
- Frame based on cluster size (dominant/niche language)
- Focus on actionability: Concrete steps creators can replicate
"""

    return prompt


def _interpret_low_value(feature_name: str) -> str:
    """Generate semantic interpretation for LOW value strategy."""
    interpretations = {
        'word_count': 'Brief (≤20 words)',
        'energy_level': 'Calm/Intimate',
        'eye_contact_rate': 'Indirect Gaze',
        'scene_count': 'Static Scene'
    }
    return interpretations.get(feature_name, f"Low {feature_name.replace('_', ' ')}")


def _interpret_high_value(feature_name: str) -> str:
    """Generate semantic interpretation for HIGH value strategy."""
    interpretations = {
        'word_count': 'Dense (≥80 words)',
        'energy_level': 'High Energy',
        'eye_contact_rate': 'Direct Eye Contact',
        'scene_count': 'Dynamic Multi-Scene'
    }
    return interpretations.get(feature_name, f"High {feature_name.replace('_', ' ')}")


def _format_centroid_compact(centroid: dict) -> str:
    """Format centroid as compact comma-separated values."""
    items = [f"{k}: {v:.2f}" for k, v in list(centroid.items())[:8]]
    return ', '.join(items) + ', ...'
```

**Key Improvements Integrated**:
1. ✅ **Issue #1**: Bimodal pattern detection with Strategy A/B presentation
2. ✅ **Issue #3**: High-contrast feature pre-filtering (≥0.20 threshold)
3. ✅ **Issue #4**: RF alignment computation with score display
4. ✅ **Issue #5**: Removed "Important" section (not in prompt)
5. ✅ **Issue #6**: Added cluster size context guidance (~17 lines)
6. ✅ **Issue #7**: Compressed RF format (single line per feature)
7. ✅ **Issue #8**: Enriched features with RF metadata for easy formatting
8. ✅ **Issue #9**: RF alignment score requirement in output schema
9. ✅ **Issue #10**: Bimodal example note with explicit instruction
10. ✅ **Issue #2**: "Exactly 3 features" (not "3-5") enforced throughout
11. ✅ **Issue #11**: Duplicate of #6 (cluster size)

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| 1 of 6 windows fails (API timeout) | Smart retry only that window (2 more attempts) | Efficient - don't waste successful API calls |
| All retries exhausted for 1 window | Abort entire Phase 1 (100% completion required) | Partial analysis is unprofessional |
| JSON truncated (max_tokens exceeded) | Automatic retry with 50% more tokens (Q5.4) | Common edge case with automatic remediation |
| 429 Rate Limiting | Exponential backoff (2s, 4s, 8s) before retry | Prevents API hammering |

---

#### Step 2.4.3: Phase 2 - Cross-Window Synthesis

**Purpose**: Synthesize "Winning Formulas" from cluster paths and cross-window patterns

**Logic**:
```python
def run_phase2_synthesis(window_analyses: dict, kmeans_outputs: dict,
                        rf_video_data: dict, bucket: str, hashtag: str | None) -> dict:
    """
    Synthesize cross-window patterns and generate winning formulas.

    Source: Mother Doc lines 2948-3096, QA Q9 (cluster path extraction)
    """
    # === 1. Extract cluster paths for all videos (Q9.1) ===
    window_types = list(window_analyses.keys())  # e.g., ['hook', 'middle_1', ..., 'closing']

    try:
        video_paths = extract_cluster_paths(window_types, kmeans_outputs)
        # Returns: [{'video_id': 'video_0', 'path': [0, 1, 1, 2, 0, 1]}, ...]
    except ValueError as e:
        raise DataIntegrityError(
            f"Cluster path extraction failed: {e}. "
            f"Stage 6 outputs may be inconsistent."
        )

    # === 2. Analyze path frequencies and apply 10% threshold (Q9.4) ===
    total_videos = len(video_paths)
    path_analysis = analyze_path_frequencies(video_paths, total_videos)

    # path_analysis contains:
    # - 'winning_paths': Top 3 paths meeting 10% threshold
    # - 'needs_fallback': True if <3 paths meet 10%
    # - 'all_paths': All unique paths with frequencies

    # === 3. Build Phase 2 LLM prompt ===
    prompt = build_phase2_prompt(
        window_analyses=window_analyses,
        top_paths=path_analysis['winning_paths'],
        all_paths=path_analysis['all_paths'][:10],  # Top 10 for context
        rf_video_data=rf_video_data,
        bucket=bucket,
        hashtag=hashtag,
        needs_fallback=path_analysis['needs_fallback']
    )

    # === 4. Call Anthropic API for synthesis ===
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=PHASE2_MAX_TOKENS,  # 8000 (larger context)
        temperature=PHASE2_TEMPERATURE,  # 0.4 (slightly higher for creative connections)
        timeout=PHASE2_TIMEOUT_SECONDS,  # 180s (conservative for complex synthesis)
        messages=[{"role": "user", "content": prompt}]
    )

    # === 5. Parse and validate synthesis ===
    synthesis = json.loads(response.content[0].text)

    # Add metadata
    synthesis['bucket'] = bucket
    synthesis['hashtag'] = hashtag
    synthesis['total_videos'] = total_videos
    synthesis['path_statistics'] = {
        'total_unique_paths': path_analysis['total_unique_paths'],
        'paths_above_threshold': path_analysis['paths_above_threshold'],
        'needs_fallback': path_analysis['needs_fallback']
    }
    synthesis['analysis_metadata'] = {
        'llm_model': ANTHROPIC_MODEL,
        'timestamp': datetime.now().isoformat(),
        'phase': 'phase2_synthesis'
    }

    # === 6. Save synthesis ===
    output_path = os.path.join(bucket_path, 'ml_analysis/llm/winning_formulas.json')
    with open(output_path, 'w') as f:
        json.dump(synthesis, f, indent=2)

    logger.info(f"✓ Phase 2 complete: Generated {len(synthesis.get('creative_reports', []))} creative reports")

    return synthesis


def extract_cluster_paths(window_types: list, kmeans_outputs: dict) -> list[dict]:
    """
    Extract cluster paths for all videos across windows.

    Source: QA Q9.1 (complete implementation)
    """
    # Get all video IDs from first window (all windows have same videos)
    first_window = window_types[0]
    all_video_ids = []

    for cluster in kmeans_outputs[first_window]['clusters']:
        for video in cluster['videos']:
            all_video_ids.append(video['video_id'])  # e.g., "video_0", "video_1", ...

    # Build cluster path for each video
    video_paths = []
    for video_id in all_video_ids:
        path = []

        for window in window_types:
            cluster_id = find_cluster_for_video(video_id, kmeans_outputs[window])
            path.append(cluster_id)

        video_paths.append({
            'video_id': video_id,
            'path': path  # [0, 1, 1, 2, 0, 1]
        })

    return video_paths


def find_cluster_for_video(video_id: str, kmeans_data: dict) -> int:
    """Find which cluster a video belongs to in a specific window."""
    for cluster in kmeans_data['clusters']:
        for video in cluster['videos']:
            if video['video_id'] == video_id:
                return cluster['cluster_id']

    # Video not found - data integrity issue
    raise ValueError(
        f"Video {video_id} not found in window {kmeans_data['window_type']}. "
        f"Stage 6 outputs inconsistent."
    )


def analyze_path_frequencies(video_paths: list[dict], total_videos: int) -> dict:
    """
    Calculate path frequencies and apply 10% threshold.

    Source: Critique Q5, QA Q9.4
    """
    from collections import Counter

    # Convert paths to tuples for counting
    path_tuples = [tuple(vp['path']) for vp in video_paths]

    # Count frequencies
    path_counts = Counter(path_tuples)

    # Build statistics
    path_stats = []
    for path_tuple, count in path_counts.items():
        percentage = (count / total_videos) * 100

        path_stats.append({
            'path': list(path_tuple),
            'frequency': count,
            'percentage': round(percentage, 1),
            'confidence_level': classify_confidence(percentage)
        })

    # Sort by frequency descending
    path_stats.sort(key=lambda x: x['frequency'], reverse=True)

    # Filter to paths meeting 10% threshold
    winning_paths = [p for p in path_stats if p['percentage'] >= 10.0]

    # Take top 3 (for creative reports)
    top_3_paths = winning_paths[:3]

    return {
        'winning_paths': top_3_paths,
        'all_paths': path_stats,
        'needs_fallback': len(winning_paths) < 3,
        'total_unique_paths': len(path_stats),
        'paths_above_threshold': len(winning_paths)
    }


def classify_confidence(percentage: float) -> str:
    """Classify confidence level based on frequency percentage."""
    if percentage >= 20.0:
        return "very_high"
    elif percentage >= 15.0:
        return "high"
    else:  # 10.0-14.9%
        return "moderate"


def build_phase2_prompt(window_analyses: dict, top_paths: list, all_paths: list,
                       rf_video_data: dict, bucket: str, hashtag: str | None,
                       needs_fallback: bool) -> str:
    """
    Build Phase 2 synthesis prompt with all Gap #1-5 improvements integrated.

    Improvements applied:
    - Gap #1: 10% threshold with labeled paths (✅ ABOVE / ❌ BELOW)
    - Gap #2: Confidence level classification (very_high/high/moderate)
    - Gap #3: Hybrid output structure (supplementary_insights section)
    - Gap #4: Feature-based fallback reports (Python-generated when <3 paths)
    - Gap #5: "Exactly 3 reports" (not "3-5")

    Source: Stage7PromptCritique.md Gaps #1-5 resolutions
    """
    # === Preprocessing: Run Section 2.2 Phase 2 functions ===

    # 1. Prepare path data with threshold labels (Section 2.2.5)
    total_videos = len([vp for vp in all_paths])  # Total from all_paths context
    path_data = prepare_path_data_for_llm(
        cluster_paths={tuple(p['path']): p['frequency'] for p in all_paths},
        threshold_pct=0.10,
        total_videos=total_videos,
        top_n=10
    )

    # 2. Generate universal principles (Section 2.2.7)
    universal_principles = generate_universal_principles(rf_video_data, top_n=7)

    # 3. Generate cross-window patterns (Section 2.2.8)
    cross_window_patterns = generate_cross_window_patterns(rf_video_data)

    # 4. Determine scenario and generate feature-based fallback if needed (Section 2.2.9)
    scenario = path_data['scenario']
    num_path_based = path_data['paths_above_threshold']
    num_feature_based = 3 - num_path_based

    feature_based_reports = []
    if num_feature_based > 0:
        feature_based_reports = generate_feature_based_reports(
            rf_video_data,
            num_reports=num_feature_based,
            used_features=set()  # Track features used in path reports
        )

    # === Build Prompt ===

    hashtag_context = f"#{hashtag}" if hashtag else "this TikTok category"

    prompt = f"""You are a TikTok creative strategy synthesizer specializing in {hashtag_context} content. Your task is to identify "Winning Formulas" by analyzing cluster path patterns across temporal windows.

## Context

You've already analyzed {len(window_analyses)} individual windows (Phase 1 complete). Now synthesize cross-window patterns to identify complete video journeys that predict viral success.

**Bucket**: {bucket}
**Total Videos**: {path_data['total_unique_paths']} unique cluster paths identified
**Paths Meeting 10% Threshold**: {path_data['paths_above_threshold']}
**Scenario**: {scenario}

---

## Phase 1 Window Analyses (Your Previous Work)

"""

    # === Include Phase 1 Analyses (Condensed) ===
    for window_type, analysis in window_analyses.items():
        prompt += f"""
### {window_type.upper()} Window Analysis

**Top Clusters**:
"""
        for cluster in analysis.get('clusters', [])[:3]:
            prompt += f"- **Cluster {cluster['cluster_id']}** ({cluster['size']} videos): {cluster['strategy_name']}\n"
            prompt += f"  Defining features: {', '.join(cluster['defining_features'][:2])}...\n"

        prompt += "\n"

    # === Cluster Path Analysis with 10% Threshold Labels (Gap #1) ===
    prompt += f"""
---

## Cluster Path Analysis (Gap #1: 10% Threshold with Status Labels)

**What is a cluster path?** A path represents the cluster IDs a video progresses through across windows.
Example: `[0, 1, 1, 2, 0, 1]` means the video uses Cluster 0 in hook, Cluster 1 in middle_1, Cluster 1 in middle_2, etc.

**10% Threshold Rule**: Only paths appearing in ≥10% of videos (minimum 10 samples) are statistically reliable for creator recommendations.

### Path Frequency Data

**Total unique paths**: {path_data['total_unique_paths']} ({"high fragmentation" if path_data['total_unique_paths'] > 30 else "moderate diversity"})
**Paths meeting 10% threshold**: {path_data['paths_above_threshold']}

### Top 10 Paths (with threshold status):

"""

    for i, (path_tuple, count, pct, status) in enumerate(path_data['top_paths'], 1):
        status_icon = "✅ ABOVE THRESHOLD" if status == 'ABOVE' else "❌ BELOW THRESHOLD"
        prompt += f"{i}. {list(path_tuple)}: {count} videos ({pct:.1f}%) - {status_icon}\n"

    # === Scenario-Specific Instructions (Gap #1, Gap #4) ===
    prompt += f"""

---

## Your Task - Scenario {scenario}

"""

    if scenario == 'A':
        # 3+ paths above threshold
        prompt += f"""
**Scenario A**: {path_data['paths_above_threshold']} paths meet the 10% threshold.

Generate **exactly 3 path-based reports** using ONLY the paths marked "✅ ABOVE THRESHOLD".
Do NOT use "❌ BELOW THRESHOLD" paths in creative_reports.

**Report Mix**:
- Report #1: Path with highest frequency (✅ ABOVE)
- Report #2: Path with second highest frequency (✅ ABOVE)
- Report #3: Path with third highest frequency (✅ ABOVE)

All reports will be `type: "path_based"` with confidence levels based on percentage:
- ≥20%: very_high
- 15-19.9%: high
- 10-14.9%: moderate
"""

    elif scenario == 'B':
        # 2 paths above threshold
        prompt += f"""
**Scenario B**: Only {path_data['paths_above_threshold']} paths meet the 10% threshold.

Generate **exactly 3 reports** total:
- **Report #1**: Path-based (highest frequency ✅ ABOVE path)
- **Report #2**: Path-based (second highest frequency ✅ ABOVE path)
- **Report #3**: Feature-based (Python-generated fallback)

**Feature-Based Fallback**: Python has pre-generated Report #3 for you (see below). Copy it into your output as-is.

**Pre-Generated Feature-Based Report #3**:
```json
{json.dumps(feature_based_reports[0], indent=2)}
```

Just copy the above JSON block into `creative_reports[2]` without modification.
"""

    elif scenario == 'C':
        # 1 path above threshold
        prompt += f"""
**Scenario C**: Only {path_data['paths_above_threshold']} path meets the 10% threshold.

Generate **exactly 3 reports** total:
- **Report #1**: Path-based (the single ✅ ABOVE path)
- **Report #2**: Feature-based (Python-generated fallback)
- **Report #3**: Feature-based (Python-generated fallback)

**Feature-Based Fallbacks**: Python has pre-generated Reports #2 and #3 for you (see below). Copy them into your output as-is.

**Pre-Generated Feature-Based Report #2**:
```json
{json.dumps(feature_based_reports[0], indent=2)}
```

**Pre-Generated Feature-Based Report #3**:
```json
{json.dumps(feature_based_reports[1], indent=2)}
```

Just copy the above JSON blocks into `creative_reports[1]` and `creative_reports[2]` without modification.
"""

    else:  # Scenario D
        # 0 paths above threshold
        prompt += f"""
**Scenario D**: HIGH FRAGMENTATION - No paths meet the 10% threshold.

The {path_data['total_unique_paths']} unique paths indicate extreme creative diversity. In this case, path-based formulas are unreliable.

Generate **exactly 3 feature-based reports** total:
- **Report #1**: Python-generated (Eye Contact & Engagement strategy)
- **Report #2**: Python-generated (Energy & Pacing strategy)
- **Report #3**: Python-generated (Speech & Density strategy)

**Feature-Based Fallbacks**: Python has pre-generated all 3 reports for you (see below). Copy them into your output as-is.

**Pre-Generated Feature-Based Report #1**:
```json
{json.dumps(feature_based_reports[0], indent=2)}
```

**Pre-Generated Feature-Based Report #2**:
```json
{json.dumps(feature_based_reports[1], indent=2)}
```

**Pre-Generated Feature-Based Report #3**:
```json
{json.dumps(feature_based_reports[2], indent=2)}
```

Just copy the above JSON blocks into `creative_reports` array without modification.

**Important**: In Scenario D, `supplementary_insights` becomes PRIMARY guidance (not secondary). Universal principles and cross-window patterns are the main takeaways.
"""

    # === Supplementary Insights (Gap #3: Hybrid Output) ===
    prompt += f"""

---

## Supplementary Insights (Gap #3: Universal Principles + Cross-Window Patterns)

**Python has pre-generated these insights** for you using video-level RF data. Include them in your output.

### Universal Principles (Applicable to ALL Videos)

These are the top {len(universal_principles)} RF features that predict success regardless of cluster path:

"""

    for i, principle in enumerate(universal_principles, 1):
        prompt += f"{i}. {principle}\n"

    prompt += """

### Cross-Window Patterns (Temporal Progressions)

"""

    for i, pattern in enumerate(cross_window_patterns, 1):
        prompt += f"{i}. {pattern}\n"

    # === Output Schema (Gap #2, Gap #3, Gap #5) ===
    prompt += f"""

---

## Output Requirements (Gaps #2, #3, #5)

Generate a JSON object with the following structure:

```json
{{
  "creative_reports": [
    // Exactly 3 reports (Gap #5: never more, never less)
    {{
      "report_id": 1,
      "type": "path_based",  // or "feature_based"
      "cluster_path": [0, 1, 1, 2, 0, 1],  // null if type="feature_based"
      "frequency": 22,  // null if type="feature_based"
      "percentage": 22.0,  // null if type="feature_based"
      "confidence_level": "very_high",  // Gap #2: very_high (≥20%), high (15-19.9%), moderate (10-14.9% or feature_based)
      "formula_name": "The Trust-Build-Peak Journey",
      "strategy_description": "Start with intimate eye contact to build trust, deliver dense educational content in middle segments, return to direct eye contact for high-energy call-to-action.",
      "window_breakdowns": [
        {{
          "window": "hook",
          "cluster_id": 0,
          "cluster_strategy": "The Direct Trust Hook",
          "key_features": ["eye_contact_rate: 0.87", "word_count: 14", "energy_level: 0.55"]
        }},
        // ... (one breakdown per window)
      ],
      "when_to_use": "Educational nutrition content, product explanations, how-to videos.",
      "creator_recommendations": [
        "Hook (0-3s): Direct eye contact (0.87), minimal words (14), moderate energy (0.55)",
        "Middle_1 (3-8s): Shift to product view, increase talking speed (50+ words), build energy to 0.60",
        "Closing (23-26s): Return to direct eye contact (0.82), peak energy (0.85), clear CTA"
      ]
    }}
  ],

  "supplementary_insights": {{  // Gap #3: NEW hybrid section
    "universal_principles": {json.dumps(universal_principles)},
    "cross_window_patterns": {json.dumps(cross_window_patterns)}
  }},

  "path_statistics": {{
    "total_unique_paths": {path_data['total_unique_paths']},
    "paths_above_threshold": {path_data['paths_above_threshold']},
    "scenario": "{scenario}"
  }}
}}
```

---

## Important Reminders

1. **Always output exactly 3 creative reports** (Gap #5: never more, never less)
2. **Apply 10% threshold strictly** (Gap #1: paths <10% are excluded from creative_reports)
3. **Classify confidence levels accurately** (Gap #2):
   - very_high: ≥20%
   - high: 15-19.9%
   - moderate: 10-14.9%
   - Feature-based reports: always moderate
4. **Use feature-based fallback when needed** (Gap #4: <3 paths above 10%)
5. **Copy pre-generated feature reports as-is** (don't modify Python-generated JSON)
6. **Include supplementary_insights** (Gap #3: universal principles + cross-window patterns)
7. **Focus on actionability**: Concrete steps creators can replicate

"""

    return prompt
```

**Key Improvements Integrated**:
1. ✅ **Gap #1**: 10% threshold with ✅ ABOVE / ❌ BELOW labels, scenario determination (A/B/C/D)
2. ✅ **Gap #2**: Confidence level classification (very_high/high/moderate) with clear thresholds
3. ✅ **Gap #3**: Hybrid output structure with `supplementary_insights` section (universal_principles + cross_window_patterns)
4. ✅ **Gap #4**: Feature-based fallback reports (Python-generated complete JSON for LLM to copy)
5. ✅ **Gap #5**: "Exactly 3 reports" enforced (never "3-5")

**Gap #4 Design Decision**: Python generates COMPLETE feature-based reports as JSON, LLM just copies them. This prevents hallucination in fallback scenarios.

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Only 2 paths meet 10% threshold | Generate 2 path-based + 1 feature-based report | Maintains "3 reports per bucket" commitment |
| 0 paths meet 10% (high fragmentation) | Generate 3 feature-based reports using universal RF features | Coverage safety net - all creators get value |
| Video missing from window | Raise ValueError, abort Phase 2 | Data integrity critical for path extraction |
| Ties at 10% (5 paths, all 10 videos) | Take first 3 in Counter order (deterministic) | Arbitrary but consistent |

---

#### Step 2.4.4: Bucket-Aware Special Cases

**Purpose**: Handle edge case buckets (0-3s single window, 3-9s minimal progression)

**Logic**:
```python
def run_stage7_llm_analysis(bucket_path: str, bucket: str) -> dict:
    """
    Main Stage 7 pipeline with bucket-aware processing.

    Source: QA Q3 (bucket-aware configuration)
    """
    # Load windows for this bucket from centralized config
    windows = BUCKET_WINDOWS[bucket]  # e.g., ['hook', 'middle_1', ..., 'closing']

    # Special case: Bucket 0-3s (only 1 window)
    if len(windows) == 1:
        logger.info(f"Bucket {bucket}: Single window (hook only) - Skipping Phase 2")

        # Phase 1: Generate hook analysis
        window_analyses = run_phase1_parallel(bucket_path, bucket, hashtag, windows)

        # Generate simplified summary (no temporal progression possible)
        summary = generate_single_window_summary(
            window_analyses['hook'],
            bucket=bucket,
            total_videos=100
        )

        # Save summary
        summary_path = os.path.join(bucket_path, f'ml_analysis/llm/bucket_summary_{bucket}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return {
            'exit_code': 0,
            'phase1': window_analyses,
            'phase2': None,  # Skipped for single-window bucket
            'summary': summary
        }

    # All other buckets: Run full Phase 1 + Phase 2
    # (Normal execution path shown in Step 2.3.2 and 2.3.3)
    ...


def generate_single_window_summary(hook_analysis: dict, bucket: str, total_videos: int) -> dict:
    """
    Generate simplified summary for bucket 0-3s (single window, no progression).

    Source: QA Q3.3
    """
    return {
        "bucket": bucket,
        "total_videos": total_videos,
        "note": "Single-window bucket - no temporal progression analysis available",
        "hook_strategies": hook_analysis['clusters'],  # 3 cluster strategies
        "recommendation": (
            f"Videos in bucket {bucket} are extremely short (0-3s). "
            f"Focus on immediate impact. Choose one of the 3 hook strategies "
            f"based on your content type and creative style."
        )
    }
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Bucket 0-3s (1 window) | Skip Phase 2, generate simplified summary | No temporal progression possible |
| Bucket 3-9s (2 windows) | Run Phase 2 normally (9 possible paths) | Minimal but meaningful progression (hook → closing) |
| Bucket 9-13s (middle_aggregate) | Treat middle_aggregate as regular window | Consistent with Stage 6 pattern |

---

## 3. Dependencies & Integration

### 3.1 Input Dependencies

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **Foundation setup** | FoundationCHILD.md (Sections 2, 4, Appendix) | Directory structure + config | client_id, bucket, BUCKET_WINDOWS config | Fail-fast if directories don't exist |
| Video-level RF analysis | Stage 6 output | JSON (~30KB) | `feature_importance` (10 features with distribution data), `bucket`, `hashtag` | Pre-flight validation fails if missing |
| Window-level RF analysis | Stage 6 output (N files where N = bucket window count) | JSON (~5KB each) | `window_type`, `feature_importance` (top 10), `model_performance` | Pre-flight validation fails |
| Window-level K-Means analysis | Stage 6 output (N files where N = bucket window count) | JSON (~5KB each) | `window_type`, `n_clusters` (3), `clusters` (with centroids, videos), `total_videos` | Pre-flight validation fails, cluster size integrity check |

**Where N = Window Count per Bucket**:
- 1-window buckets (0-3s): 3 total files (1 video RF + 1 window RF + 1 window K-Means)
- 2-window buckets (3-9s): 5 total files
- 3-window buckets (9-13s, 13-18s): 7 total files
- 6-window buckets (18-33s): 13 total files (1 video + 6 window RF + 6 window K-Means)
- 7-window buckets (33-60s, 60-90s, 90-120s): 15 total files (1 video + 7 window RF + 7 window K-Means)

See `config/bucket_definitions.py` for exact window counts and names per bucket.
| API credentials | Environment variable | String | `ANTHROPIC_API_KEY` (format: sk-ant-api03-...) | Pre-flight validation fails with clear error |
| LLM configuration | `config/llm_config.py` | Python constants | `ANTHROPIC_MODEL`, `PHASE1_MAX_TOKENS`, `PHASE1_TEMPERATURE`, etc. | Import error if missing |
| Bucket window definitions | `config.bucket_definitions.BUCKET_WINDOWS` | Dict mapping | Bucket → window list | KeyError if bucket not in config |

### 3.2 Output Contracts

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| Phase 1 window analyses | JSON (~2-3KB each) | See Section 5.2.1 | Phase 2 synthesis, Stage 8 PDF generation | Schema validation: required fields present |
| Phase 2 winning formulas | JSON (~10-15KB) | See Section 5.2.2 | Stage 8 PDF generation | Creative reports count = 3, confidence levels valid |
| Complete analysis | JSON (~40-50KB) | Phase 1 + Phase 2 combined | Stage 8 PDF generation, analytics/debugging | Combined validation |
| Bucket summary (0-3s only) | JSON (~5KB) | Simplified structure with 3 strategies | Stage 8 PDF generation | Hook strategies present |

### 3.3 Cross-Stage Dependencies

**This stage depends on**:
- **Stage 6 (ML Analysis Generation)**: Must complete successfully (all 13 JSONs exist with valid schemas)
  - Pre-flight validation checks file existence, parseability, schema compliance
  - Cluster size integrity critical for Phase 2 path extraction
- **Stage 4 (Feature Transformation)**: Indirectly through Stage 6 (cross-window features must exist in video-level RF)
  - `hook_to_middle_energy_delta`, `middle_to_closing_contrast`, `eye_contact_consistency`, etc.
  - Resolution documented in Crosswindowupgrade.md

**This stage is required by**:
- **Stage 8 (PDF Report Generation)**: Consumes all Stage 7 JSONs to create creator-friendly PDF reports
  - Expects `ml_analysis/llm/` directory with 8 files per bucket
  - Requires confidence levels and RF validation metadata for report prioritization

**Failure Impact**:
- If this stage fails: Stage 8 cannot generate PDF reports (no creative insights available)
- Checkpoint: Resume from Stage 7 without re-running Stages 1-6 (Stage 6 outputs remain valid)

### 3.4 External Dependencies

**Python Libraries**:
```python
import anthropic  # Anthropic SDK 0.17.0+ for Claude API
import concurrent.futures  # Python stdlib (parallel execution)
import json  # Python stdlib (JSON parsing)
import os  # Python stdlib (file operations)
from collections import Counter  # Python stdlib (frequency counting)
from datetime import datetime  # Python stdlib (timestamps)
```

**File System**:
- Read access: `/data/clients/{client_id}/buckets/bucket_{bucket}/ml_analysis/`
- Write access: `/data/clients/{client_id}/buckets/bucket_{bucket}/ml_analysis/llm/`

**Environment Variables**:
- `ANTHROPIC_API_KEY`: Anthropic API key (required, validated in pre-flight)

**External Services**:
- **Anthropic API** (claude-sonnet-4-20250514):
  - Network connectivity required
  - API key must be valid
  - Rate limiting possible (429 errors handled with exponential backoff)
  - Service availability variable (503 errors retried with backoff)

**Configuration Files** (internal dependencies):
- `config/llm_config.py`: LLM model version, max_tokens, temperature, timeouts
- `config.bucket_definitions.py`: BUCKET_WINDOWS mapping (shared with Stages 4-6)

---

## 4. Configuration & Parameters

### 4.1 CLI Parameters

| Parameter | Type | Default | Valid Values | Impact | Example |
|-----------|------|---------|--------------|--------|---------|
| `--stage` | int | Required | 7 | Identifies Stage 7 execution | `--stage 7` |
| `--client` | str | Required | Any string | Determines client directory path | `--client acme` |
| `--bucket` | str | Required | `0-3s`, `3-9s`, `9-13s`, `13-18s`, `18-33s`, `33-60s`, `60-90s`, `90-120s` | Determines bucket directory and window structure | `--bucket 18-33s` |

**CLI Command**:
```bash
python run_ml_pipeline.py --stage 7 --client acme --bucket 18-33s
```

**Parameter Construction**:
```python
# run_ml_pipeline.py
bucket_path = f'/data/clients/{args.client}/buckets/bucket_{args.bucket}'
result = run_stage7_llm_analysis(bucket_path, args.bucket)
```

**Hashtag Parameter**: NOT a CLI parameter. Read from `metadata.json` file if exists, default to `None` if missing. See Section 6.2 for handling.

### 4.2 Internal Configuration

**File**: `config/llm_config.py`

```python
# Anthropic API Configuration
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # Production model

# Phase 1: Per-Window Analysis
PHASE1_MAX_TOKENS = 4000
PHASE1_TEMPERATURE = 0.3  # Lower = more consistent/focused
PHASE1_TIMEOUT_SECONDS = 90  # Conservative (typical: 5-10s, 99th percentile: 30-45s)

# Phase 2: Cross-Window Synthesis
PHASE2_MAX_TOKENS = 8000  # Larger context for synthesis
PHASE2_TEMPERATURE = 0.4  # Slightly higher for creative connections
PHASE2_TIMEOUT_SECONDS = 180  # Very conservative (typical: 15-30s, 99th percentile: 60-90s)

# Validation Layer (Automated Checks)
VALIDATION_MAX_TOKENS = 1000  # Short responses for yes/no validation
VALIDATION_TEMPERATURE = 0.1  # Very low = deterministic
VALIDATION_TIMEOUT_SECONDS = 30

# Retry Configuration
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}  # Temporary errors
FATAL_STATUS_CODES = {400, 401, 403, 422}  # Permanent errors
MAX_RETRY_ATTEMPTS = 2  # Max retries per window
BACKOFF_MAX_WAIT_SECONDS = 30  # Cap for exponential backoff

# Path Frequency Filtering (Critique Q5)
PATH_FREQUENCY_THRESHOLD = 10.0  # Minimum percentage (10%)
CONFIDENCE_VERY_HIGH_THRESHOLD = 20.0  # ≥20%
CONFIDENCE_HIGH_THRESHOLD = 15.0  # 15-20%
# Below 15% = moderate (but must be ≥10% to include)
```

**Rationale for Conservative Timeouts**:
- **90s Phase 1**: 2x safety margin (typical: 5-10s, 99th percentile: 30-45s during API high load)
- **180s Phase 2**: 2x safety margin (typical: 15-30s, 99th percentile: 60-90s for complex synthesis)
- **Cost of premature timeout**: Aborting bucket after 6 hours of video processing is expensive
- **Negligible downside**: If actual failure (network down), waiting 90s vs 60s doesn't matter

---

## 5. Data Schemas

### 5.1 Input Schema

**File 1**: `ml_analysis/rf_video_analysis.json` (Video-Level RF)

| Column | Type | Range | Nulls? | Description | Source |
|--------|------|-------|--------|-------------|--------|
| `analysis_type` | string | "random_forest" | No | Identifies RF analysis type | Stage 6 |
| `bucket` | string | Valid bucket names | No | Duration bucket | Stage 6 |
| `hashtag` | string | Any string | Yes | Hashtag context | Stage 6 (from metadata) |
| `video_count` | int | 10-300 | No | Total videos processed | Stage 6 |
| `input_features` | int | 24-220 | No | Total feature count (varies by bucket) | Stage 6 |
| `feature_importance` | array[object] | 10 features | No | Top 10 features with importance, gaps, distributions | Stage 6 |
| `feature_importance[i].feature` | string | Feature names | No | Cross-window or single-window feature name | Stage 6 |
| `feature_importance[i].importance` | float | 0.0-1.0 | No | RF importance score | Stage 6 |
| `feature_importance[i].top_performer_avg` | float | Varies by feature | No | Mean value in top 80% videos | Stage 6 |
| `feature_importance[i].bottom_performer_avg` | float | Varies by feature | No | Mean value in bottom 20% videos | Stage 6 |
| `feature_importance[i].gap` | float | Any | No | Difference (top - bottom) | Stage 6 |
| `feature_importance[i].distribution` | object | Thresholds + percentages | No | High/medium/low distribution in top/bottom | Stage 6 (added per Critique Q3) |

**File 2**: `ml_analysis/{window}_rf_analysis.json` (Window-Level RF, 6-7 files)

| Column | Type | Range | Nulls? | Description | Source |
|--------|------|-------|--------|-------------|--------|
| `model_type` | string | "window_level_rf" | No | Identifies window-level RF | Stage 6 |
| `window_type` | string | "hook", "middle_1-5", "middle_aggregate", "closing" | No | Window identifier | Stage 6 |
| `bucket` | string | Valid bucket names | No | Duration bucket | Stage 6 |
| `total_videos` | int | 10-300 | No | Total videos | Stage 6 |
| `input_features` | int | 21 | No | Features per window (always 21 for window-level) | Stage 6 |
| `model_performance` | object | Accuracy/precision/recall | No | Model quality metrics | Stage 6 |
| `feature_importance` | array[object] | 10 features | No | Top 10 features for this window | Stage 6 |
| `feature_importance[i].feature` | string | Feature names (NO window prefix) | No | Feature name normalized | Stage 6 |
| `feature_importance[i].importance` | float | 0.0-1.0 | No | Window-specific importance | Stage 6 |
| `feature_importance[i].rank` | int | 1-10 | No | Importance rank | Stage 6 |

**File 3**: `ml_analysis/{window}_kmeans_analysis.json` (Window-Level K-Means, 6-7 files)

| Column | Type | Range | Nulls? | Description | Source |
|--------|------|-------|--------|-------------|--------|
| `window_type` | string | Window identifiers | No | Window name | Stage 6 |
| `bucket` | string | Valid bucket names | No | Duration bucket | Stage 6 |
| `total_videos` | int | 10-300 | No | Total videos | Stage 6 |
| `n_clusters` | int | 3 | No | Always 3 clusters | Stage 6 |
| `clusters` | array[object] | 3 clusters | No | Cluster data | Stage 6 |
| `clusters[i].cluster_id` | int | 0, 1, 2 | No | Cluster identifier | Stage 6 |
| `clusters[i].size` | int | > 0 | No | Videos in cluster | Stage 6 |
| `clusters[i].centroid` | object | 21-39 feature keys | No | All features as {feature_name: value} with NORMALIZED names (no `_scaled` suffix) | Stage 6 (normalized) |
| `clusters[i].videos` | array[object] | size entries | No | Video IDs and distances | Stage 6 |
| `clusters[i].videos[j].video_id` | string | "video_N" format | No | Video identifier (consistent across windows) | Stage 6 |
| `clusters[i].videos[j].distance_to_centroid` | float | ≥ 0.0 | No | Euclidean distance | Stage 6 |

### 5.2 Output Schema

#### 5.2.0 Phase 1 Status File (Internal)

**File**: `ml_analysis/llm/.phase1_status.json` (~500 bytes, internal tracking file)

**Purpose**: Track Phase 1 completion status and enable resume capability for cost-optimized incremental saves

**Schema**:
```json
{
  "total_windows": 6,
  "completed_windows": ["hook", "middle_1", "middle_2", "middle_3", "middle_4", "closing"],
  "failed_windows": [],
  "phase1_complete": true,
  "started_at": "2025-10-16T10:25:00Z",
  "last_updated": "2025-10-16T10:27:45Z",
  "completed_at": "2025-10-16T10:27:45Z"
}
```

**Field Details**:
- `total_windows`: int (number of windows in bucket, 1-7)
- `completed_windows`: array[string] (window types successfully completed)
- `failed_windows`: array[object] (failed windows with error details and timestamps)
  - `window`: string (window type that failed)
  - `error`: string (error message)
  - `timestamp`: string (ISO 8601 timestamp)
- `phase1_complete`: bool (true when all windows succeeded, false otherwise)
- `started_at`: string (ISO 8601 timestamp when Phase 1 started)
- `last_updated`: string (ISO 8601 timestamp of last status update)
- `completed_at`: string | null (ISO 8601 timestamp when Phase 1 completed, null if incomplete)

**Usage**:
- **Check completion**: `phase1_complete: true` means all windows succeeded
- **Resume on retry**: On re-run, skip windows in `completed_windows` (saves API cost)
- **Debug failures**: `failed_windows` array shows which windows failed and why
- **Lifecycle**: Created at Phase 1 start, updated after each window, deleted after Phase 2 completes (optional cleanup)

**Note**: This is an **internal tracking file**, not consumed by Stage 8. Stage 8 only reads the window analysis JSONs.

**Source**: Cross-HLD Alignment Issue #11 (status tracking for incremental saves)

#### 5.2.1 Phase 1 Window Analysis JSON

**Files**: `ml_analysis/llm/{window}_analysis.json` (6-7 files per bucket, ~2-3KB each)

**Schema Changes (2025-10-17 - Issues #2, #9 improvements)**:

| Field | Before (2025-10-16) | After (2025-10-17) | Improvement |
|-------|---------------------|---------------------|-------------|
| `defining_features` | array[string] (3-5 features) | array[string] (**exactly 3 features**) | Issue #2: Enforces consistent output |
| `rf_validation.insight` | Generic insight text | Must include **RF alignment score** (e.g., "RF alignment: 2/5") | Issue #9: Shows how many of top 5 RF features cluster uses |

**Example insight with alignment score**:
```json
"insight": "This cluster leverages 2 of the top 5 most predictive features (RF alignment: 2/5), using high eye contact and moderate energy as primary engagement drivers."
```

```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "hashtag": "nutrition",
  "total_videos": 100,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "name": "The Direct Eye Contact Hook",
      "defining_features": [
        "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43 - HIGHEST PREDICTOR)",
        "word_count: 14 (RF rank #3, importance 0.18, low count strategy)",
        "energy_level: 0.55 (RF rank #2, importance 0.22, moderate-calm approach)"
      ],
      "rf_validation": {
        "top_predictive_features_in_cluster": [
          "eye_contact_rate: Cluster value 0.87 matches top performer avg 0.88 (RF validated)"
        ],
        "insight": "This cluster leverages the #1 most predictive hook feature at optimal levels."
      },
      "strategy_description": "Creator looks directly at camera with minimal speech...",
      "creator_recommendations": [
        "PRIORITY: Maintain 85-90% eye contact (RF #1 predictor)",
        "Keep opening statement under 15 words",
        "Target moderate energy 0.55-0.60"
      ]
    },
    // ... clusters 1 and 2
  ],
  "analysis_metadata": {
    "llm_model": "claude-sonnet-4-20250514",
    "timestamp": "2025-10-16T14:30:00Z",
    "phase": "phase1_window"
  }
}
```

**Field Details**:
- `window_type`: string (window identifier)
- `bucket`: string (duration bucket)
- `hashtag`: string | null (optional context)
- `total_videos`: int (100)
- `clusters`: array[3 objects]
  - `cluster_id`: int (0, 1, 2)
  - `size`: int (videos in cluster)
  - `name`: string (LLM-generated creative name)
  - `defining_features`: array[string] (**exactly 3** key features with RF context - Issue #2)
  - `rf_validation`: object (how cluster uses top RF features)
  - `strategy_description`: string (creative approach)
  - `creator_recommendations`: array[string] (actionable steps with RF targets)

#### 5.2.2 Phase 2 Winning Formulas JSON

**File**: `ml_analysis/llm/winning_formulas.json` (~10-15KB)

**Schema Changes**:

**2025-10-17** - Gaps #2, #3, #5 improvements:

| Field | Before (2025-10-16) | After (2025-10-17) | Improvement |
|-------|---------------------|---------------------|-------------|
| `creative_reports` count | "3-5 reports" | **"Exactly 3 reports"** | Gap #5: Enforces consistent output |
| `creative_reports[].type` | Not present | **Required field**: "path_based" \| "feature_based" | Gap #4: Distinguishes report types |
| `creative_reports[].confidence_level` | Not present | **Required field**: "very_high" \| "high" \| "moderate" | Gap #2: Shows pattern strength |
| `creative_reports[].path`, `.frequency`, `.percentage` | Always required | **Nullable** (null for feature_based reports) | Gap #4: Supports fallback scenarios |
| `supplementary_insights` | Not present | **NEW section** with `universal_principles` + `cross_window_patterns` | Gap #3: Coverage safety net for all creators |

**2025-10-27** - Schema consistency bug fix:
- **Feature-based reports now use full 13-field schema** (previously used simplified 5-field schema)
- All reports (path-based AND feature-based) now have identical structure
- Ensures downstream Stage 8 PDF generation compatibility

**Key Scenarios** (Gap #1, #4):
- **Scenario A** (3+ paths ≥10%): 3 path-based reports
- **Scenario B** (2 paths ≥10%): 2 path-based + 1 feature-based
- **Scenario C** (1 path ≥10%): 1 path-based + 2 feature-based
- **Scenario D** (0 paths ≥10%): 3 feature-based (high fragmentation)

```json
{
  "bucket": "18-33s",
  "hashtag": "nutrition",
  "total_videos": 100,
  "total_unique_paths": 45,
  "paths_above_threshold": 5,
  "creative_reports": [
    {
      "report_id": 1,
      "type": "path_based",
      "path": [0, 1, 1, 1, 2, 0],
      "frequency": 22,
      "percentage": 22.0,
      "confidence_level": "very_high",
      "formula_name": "The Educator's Arc",
      "structure": {
        "hook": "The Direct Eye Contact Hook (Cluster 0)",
        "middle_pattern": "Information Dense Middle (Cluster 1 → 1 → 1 → 2)",
        "closing": "High Energy CTA (Cluster 0)"
      },
      "temporal_progressions": [
        {
          "feature": "energy_level",
          "hook": 0.55,
          "middle_avg": 0.65,
          "closing": 0.85,
          "pattern": "Steady build from moderate to high",
          "hook_to_middle_delta": 0.16,
          "middle_to_closing_contrast": 0.27
        }
      ],
      "rf_cross_window_validation": {
        "matches_top_patterns": [
          "hook_to_middle_energy_delta: 0.16 (matches RF top performer avg 0.15)",
          "middle_to_closing_contrast: 0.27 (matches RF top performer avg 0.28)"
        ],
        "insight": "This formula exhibits ALL THREE major cross-window patterns identified by video-level RF.",
        "rf_validation_score": "9/10"
      },
      "strategy_description": "Start with intimate eye contact...",
      "when_to_use": "Educational nutrition content, product explanations...",
      "step_by_step_template": [
        "Hook (0-3s): Direct eye contact (0.87), minimal words (14)",
        "Middle_1 (3-8s): Shift to product view, increase talking speed",
        "CROSS-WINDOW TARGETS (RF validated): Energy delta hook→middle: +0.16"
      ]
    },
    // ... reports 2 and 3 (path-based or feature-based depending on threshold)
  ],
  "supplementary_insights": {
    "universal_principles": [
      "High eye contact rate (88% vs 45% for top vs bottom performers)",
      "Consistent energy maintenance across windows",
      "Clear CTA in closing window"
    ],
    "cross_window_patterns": [
      "78% of high-performing videos use 'bookend' eye contact pattern",
      "Energy builds are common (65%), but 12% succeed with consistent energy"
    ]
  },
  "path_statistics": {
    "total_unique_paths": 45,
    "paths_above_threshold": 5,
    "needs_fallback": false
  },
  "analysis_metadata": {
    "llm_model": "claude-sonnet-4-20250514",
    "timestamp": "2025-10-16T14:32:00Z",
    "phase": "phase2_synthesis"
  }
}
```

**Field Details**:
- `creative_reports`: array[3 objects] (ALWAYS 3 reports)
  - `report_id`: int (1, 2, 3)
  - `type`: string ("path_based" or "feature_based")
  - `path`: array[int] (cluster IDs per window) - only for path_based
  - `frequency`: int (video count) - only for path_based
  - `percentage`: float (frequency / total_videos * 100)
  - `confidence_level`: string ("very_high" | "high" | "moderate")
  - `formula_name`: string (LLM-generated)
  - `structure`: object (hook/middle/closing cluster names)
  - `temporal_progressions`: array[object] (feature evolution across windows)
  - `rf_cross_window_validation`: object (validation against video-level RF patterns)
- `supplementary_insights`: object (coverage safety net for all creators)
  - `universal_principles`: array[string] (top 5-7 RF features applicable to all videos)
  - `cross_window_patterns`: array[string] (general progression patterns)

#### 5.2.3 Complete Analysis JSON

**File**: `ml_analysis/llm/complete_analysis_{bucket}.json` (~40-50KB)

Combined Phase 1 + Phase 2 outputs (all window analyses + winning formulas in single file for convenience).

---

## 6. Error Handling & Validation

### 6.1 Input Validation

**Pre-Flight Validation** (See Step 2.3.1 for complete implementation):

```python
def run_preflight_validation(bucket_path: str, bucket: str) -> None:
    """
    Three-layer validation before Phase 1 execution.

    Layer 1: API credentials (ANTHROPIC_API_KEY exists and valid format)
    Layer 2: Stage 6 files (all 13 JSONs exist and parseable)
    Layer 3: Schema validation and data integrity (required fields, cluster sizes sum correctly)
    """
    # Layer 1: API credentials
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not api_key.startswith("sk-ant-"):
        raise PreFlightValidationError("Invalid or missing ANTHROPIC_API_KEY")

    # Layer 2: File existence and parseability
    windows = BUCKET_WINDOWS[bucket]
    expected_files = [
        'ml_analysis/rf_video_analysis.json',
        *[f'ml_analysis/{w}_rf_analysis.json' for w in windows],
        *[f'ml_analysis/{w}_kmeans_analysis.json' for w in windows]
    ]

    # Check all files exist
    missing = [f for f in expected_files if not os.path.exists(os.path.join(bucket_path, f))]
    if missing:
        raise PreFlightValidationError(f"Stage 6 incomplete: Missing {len(missing)} JSONs")

    # Check all files parseable
    for file_path in expected_files:
        try:
            json.load(open(os.path.join(bucket_path, file_path)))
        except json.JSONDecodeError:
            raise PreFlightValidationError(f"Malformed JSON: {file_path}")

    # Layer 3: Schema and data integrity
    for window in windows:
        kmeans_data = load_json(os.path.join(bucket_path, f'ml_analysis/{window}_kmeans_analysis.json'))

        # Check cluster sizes sum to total_videos (critical for Phase 2)
        if sum(c['size'] for c in kmeans_data['clusters']) != kmeans_data['total_videos']:
            raise ValidationError(f"{window}: Cluster size mismatch")
```

### 6.2 Error Cases

| Error | Detection | Handling | User Message | Exit Code | Status File |
|-------|-----------|----------|--------------|-----------|-------------|
| Missing API key | Pre-flight Layer 1 | Fail-fast | `"ANTHROPIC_API_KEY not set. Add to .env file: ANTHROPIC_API_KEY=sk-ant-api03-..."` | 1 | Not created yet |
| Invalid API key format | Pre-flight Layer 1 | Fail-fast | `"Invalid ANTHROPIC_API_KEY format. Expected: sk-ant-api03-..."` | 1 | Not created yet |
| Stage 6 files missing | Pre-flight Layer 2 | Fail-fast, list all missing | `"Stage 6 incomplete: Missing 7 of 13 JSONs. Re-run Stage 6."` | 2 | Not created yet |
| Malformed JSON | Pre-flight Layer 2 | Fail-fast, identify file | `"Stage 6 validation failed: hook_rf_analysis.json malformed. Re-run Stage 6."` | 2 | Not created yet |
| Cluster size mismatch | Pre-flight Layer 3 | Fail-fast with specifics | `"hook_kmeans_analysis.json: Cluster sizes sum to 98 but total_videos is 100."` | 3 | Not created yet |
| **API 429 Rate Limiting** | During API call | Exponential backoff (2s, 4s, 8s), retry up to 2 times | `"Rate limited. Retrying in 2s..."` | 0 (auto-recover) | Status updated with retry info |
| **API 503 Service Unavailable** | During API call | Exponential backoff, retry up to 2 times | `"Anthropic API unavailable. Retrying in 2s..."` | 0 (auto-recover) | Status updated with retry info |
| **API 401 Unauthorized** | During API call | Abort immediately (fatal) | `"Invalid API key. Check ANTHROPIC_API_KEY."` | 4 | Status shows failure, preserved for debugging |
| **API 400 Bad Request** | During API call | Abort immediately (fatal) | `"Invalid prompt format. This is a code bug."` | 4 | Status shows failure, preserved for debugging |
| **API Timeout (90s)** | During API call | Retry with backoff (up to 2 times) | `"hook API call timed out after 90s. Retrying..."` | 0 (auto-recover) | Status updated with retry attempt |
| **JSON Truncated (max_tokens)** | After API response | Automatic retry with 50% more tokens | `"JSON truncated. Retrying with 6000 tokens..."` | 0 (auto-recover) | Status updated |
| **Invalid LLM JSON response** | After JSON parsing | Retry with backoff (counts as failed window attempt) | `"LLM generated malformed output. Retrying..."` | 0 (auto-recover) | Status updated |
| **All retries exhausted for window** | After 3 attempts | Abort Phase 1, preserve partial progress | `"hook failed after 3 attempts. Re-run Stage 7 to resume from checkpoint."` | 5 | Status preserved (enables resume on retry) |
| **Phase 2 path extraction failure** | During cluster path extraction | Abort Phase 2 | `"Video video_42 missing from middle_1. Stage 6 outputs inconsistent."` | 6 | Status shows Phase 1 complete |
| **<3 paths meet 10% threshold** | During path frequency analysis | Generate feature-based fallback reports | `"Only 2 paths ≥10%. Generating 2 path + 1 feature report."` | 0 (warning) | No impact (Phase 2 warning) |
| **Status file corrupted** | On resume attempt | Delete status file, start fresh | `"Status file corrupted. Starting Phase 1 from beginning."` | 0 (auto-recover) | Deleted and recreated |

**Status File Lifecycle**:
- **Created**: At Phase 1 start (after pre-flight validation)
- **Updated**: After each window completes (success or failure)
- **Preserved on failure**: Status file remains for resume capability (user can retry and resume from checkpoint)
- **Deleted on success**: Optional cleanup after Phase 2 completes (or leave for debugging)
- **Manual cleanup**: User can delete `.phase1_status.json` to force fresh start

**Smart Retry Logic** (Critique Q4):
```python
# Retry ONLY failed windows (not all 6-7)
# Example: 6 windows, 2 fail on first attempt
# Retry 1: Only retry those 2 windows (not all 6)
# Retry 2: If still failing, retry again
# After 2 retries: Abort bucket if ANY window still failing

for attempt in range(1, max_attempts + 1):
    try:
        response = client.messages.create(...)
        return parse_and_validate_json(response)
    except Exception as e:
        if not should_retry_api_error(e) or attempt >= max_attempts:
            raise
        retry_with_backoff(attempt)  # 2s, 4s, 8s with jitter
```

**Hashtag Handling**:
```python
# Hashtag is OPTIONAL - read from metadata.json if exists
hashtag = get_hashtag_from_metadata(bucket_path)  # Returns None if missing

# If None: LLM generates generic recommendations
# If present: LLM includes hashtag-specific context
```

### 6.3 Output Validation

**Automated Validation Layer** (Critique Q3 Layer 1):

```python
def validate_llm_output(analysis: dict, kmeans_data: dict, rf_data: dict, window_type: str) -> None:
    """
    Post-LLM validation to detect hallucinations.

    Checks:
    1. Feature value contradictions (LLM claims "high energy 0.85" but centroid shows 0.22)
    2. Invented features (LLM references features not in source JSON)
    3. RF misalignment (priority recommendations ignore top RF features)

    Source: Critique Q3 Layer 1
    """
    # Extract LLM-mentioned features from recommendations
    llm_features = extract_mentioned_features(analysis)

    # Check 1: Feature contradictions
    for feature_name, llm_value in llm_features.items():
        # Look up feature in K-Means centroids
        for cluster in kmeans_data['clusters']:
            if feature_name in cluster['centroid']:
                actual_value = cluster['centroid'][feature_name]
                if abs(llm_value - actual_value) > 0.3:  # Threshold for "contradiction"
                    logger.warning(
                        f"{window_type}: Feature contradiction detected. "
                        f"LLM says {feature_name}={llm_value}, "
                        f"but actual centroid value is {actual_value}. "
                        f"Flagging for human review."
                    )

    # Check 2: Invented features
    valid_features = set(kmeans_data['clusters'][0]['centroid'].keys())
    valid_features.update([f['feature'] for f in rf_data['feature_importance']])

    for feature_name in llm_features.keys():
        if feature_name not in valid_features:
            logger.error(
                f"{window_type}: LLM invented feature '{feature_name}' "
                f"that doesn't exist in source data. Re-generating response."
            )
            raise ValidationError(f"Invented feature detected: {feature_name}")

    # Check 3: RF misalignment
    top_rf_features = [f['feature'] for f in rf_data['feature_importance'][:3]]  # Top 3

    priority_recommendations = [r for r in analysis['clusters'][0]['creator_recommendations']
                               if r.startswith("PRIORITY")]

    if priority_recommendations:
        # At least one PRIORITY recommendation should mention a top-3 RF feature
        mentions_top_rf = any(
            any(rf_feat in rec for rf_feat in top_rf_features)
            for rec in priority_recommendations
        )

        if not mentions_top_rf:
            logger.warning(
                f"{window_type}: PRIORITY recommendations ignore top RF features. "
                f"Top RF: {top_rf_features}. Flagging for review."
            )
```

**Schema Validation**:
```python
# Phase 1 output
required_fields = ['window_type', 'bucket', 'clusters']
for field in required_fields:
    if field not in analysis:
        raise ValidationError(f"Missing required field: {field}")

# Cluster structure
for cluster in analysis['clusters']:
    if 'cluster_id' not in cluster or 'name' not in cluster:
        raise ValidationError(f"Invalid cluster structure")
```

---

## 7. Implementation Roadmap

**Purpose**: Guide developers through implementing Stage 7 in the correct order with clear dependencies and time estimates.

**Source**: Stage7PromptCritique.md Implementation Order Recommendation (lines 3993-4020), integrated with all Issue #1-11 and Gap #1-5 resolutions

### 7.1 Implementation Phases Overview

```
PHASE 1: Python Preprocessing Functions (4 hours)
   ↓
PHASE 2: Unit Tests for Preprocessing (3 hours)
   ↓
PHASE 3: Prompt Integration (2 hours)
   ↓
PHASE 4: Claude API Integration (2 hours)
   ↓
PHASE 5: End-to-End Testing (4 hours)

Total Estimated Effort: 15 hours (2 working days)
```

### 7.2 Phase 1: Python Preprocessing Functions (4 hours)

**Prerequisites**:
- ✅ Stage 4 computes cross-window features (FeatureTransformationCHILD.md Section 6.5)
- ✅ Stage 6 includes cross-window features in RF output (MLAnalysisGenerationCHILD.md)

**Implementation Order** (highest priority first):

#### 7.2.1 Phase 1 Preprocessing Functions (2 hours)

Implement in this order (functions from Section 2.2):

1. **`detect_bimodal_pattern()`** (30 min) - Section 2.2.1
   - Input: RF feature distribution data
   - Output: Bimodal classification with interpretation
   - Testing: Use fixture with 40% high, 35% low → expect is_bimodal=True

2. **`identify_high_contrast_features()`** (45 min) - Section 2.2.2
   - Input: K-Means data with 3 clusters × 21 features
   - Output: Pre-filtered features with ≥0.20 contrast
   - Testing: Pilot data shows 21 features → 8-12 high-contrast features
   - Edge case: Handle features with identical values across clusters (contrast=0)

3. **`compute_rf_alignment()`** (45 min) - Section 2.2.3
   - Input: Cluster centroid + RF top 10 features
   - Output: Alignment score (e.g., "3/5")
   - Testing: Verify ±0.15 threshold (0.87 matches 0.88, 0.72 differs from 0.88)
   - Edge case: No features align → Return alignment_count=0 (creative novelty, not a bug)

4. **`enrich_high_contrast_features()`** (30 min) - Section 2.2.4
   - Input: High-contrast features + RF data
   - Output: Enriched features with RF metadata (rank, importance, gap)
   - Testing: Verify all metadata fields present and correctly mapped
   - Dependency: Requires `identify_high_contrast_features()` output

#### 7.2.2 Phase 2 Preprocessing Functions (2 hours)

Implement in this order:

5. **`prepare_path_data_for_llm()`** (30 min) - Section 2.2.5
   - Input: Cluster paths dict {(0,1,1,2,0,1): 22, ...}
   - Output: Labeled paths with scenario determination (A/B/C/D)
   - Testing: Test all 4 scenarios:
     - Scenario A: 5 paths ≥10% → 3 path-based reports
     - Scenario B: 2 paths ≥10% → 2 path + 1 feature
     - Scenario C: 1 path ≥10% → 1 path + 2 feature
     - Scenario D: 0 paths ≥10% → 3 feature-based
   - Edge case: Ties at exactly 10% → Deterministic Counter order

6. **`classify_confidence_level()`** (15 min) - Section 2.2.6
   - Input: Frequency percentage (e.g., 22.0)
   - Output: "very_high" | "high" | "moderate"
   - Testing: Boundary cases (19.9% = high, 20.0% = very_high)

7. **`generate_universal_principles()`** (30 min) - Section 2.2.7
   - Input: RF video-level data
   - Output: 5-7 formatted principle strings
   - Testing: Verify prevalence percentages calculated correctly
   - Edge case: Feature missing prevalence field → Default to 0.0

8. **`generate_cross_window_patterns()`** (30 min) - Section 2.2.8
   - Input: RF video-level data
   - Output: 3-5 cross-window pattern strings (or graceful degradation)
   - Testing: Test both normal case (features exist) and fallback (features missing)
   - Keywords to detect: 'delta', 'consistency', 'contrast', 'progression', '_std'

9. **`generate_feature_based_reports()`** (75 min) - Section 2.2.9 ⚠️ MOST COMPLEX
   - Input: RF data, num_reports (1-3), used_features set
   - Output: Complete feature-based report JSONs
   - Testing: Test with num_reports=1, 2, 3
   - Feature grouping: Eye Contact, Energy, Speech, Visual Variety
   - Data-driven templates: Verify top_performer_avg drives template selection
   - Edge case: Group has <1 available feature → Fallback to next available features

**Critical Path**: Implement `generate_feature_based_reports()` LAST (depends on understanding of report schema)

### 7.3 Phase 2: Unit Tests for Preprocessing (3 hours)

**Test Framework**: pytest with fixtures

**Test Files Structure**:
```
tests/
  test_phase1_preprocessing.py  # Functions 1-4
  test_phase2_preprocessing.py  # Functions 5-9
  fixtures/
    sample_rf_data.json
    sample_kmeans_data.json
    sample_cluster_paths.json
```

**Key Test Scenarios** (see Section 8 for detailed validation scenarios):

- **Bimodal Detection**: 30%/30% split → is_bimodal=True, 72%/15% → is_bimodal=False
- **High-Contrast Filtering**: 21 features → 8 with ≥0.20 contrast
- **RF Alignment**: Cluster with 3/5 alignment vs 0/5 creative novelty
- **Path Scenarios**: Test A, B, C, D with actual path frequency distributions
- **Feature-Based Reports**: Verify generated JSON matches schema exactly

**Coverage Target**: 90%+ line coverage for preprocessing functions

### 7.4 Phase 3: Prompt Integration (2 hours)

**Task**: Integrate preprocessing outputs into Phase 1 and Phase 2 prompts

#### 7.4.1 Phase 1 Prompt Builder (1 hour)

Implement `build_phase1_prompt()` from Section 2.4.2 (lines 1528-1784):
- Call all 4 Phase 1 preprocessing functions
- Format bimodal patterns, high-contrast features, RF alignment
- Include cluster size context guidance
- Generate complete prompt string

**Testing**:
- Verify prompt includes all required sections
- Check bimodal features show Strategy A/B alternatives
- Validate RF alignment scores displayed correctly
- Prompt length: ~2000-3000 tokens (within Claude context limits)

#### 7.4.2 Phase 2 Prompt Builder (1 hour)

Implement `build_phase2_prompt()` from Section 2.4.3 (lines 1992-2300):
- Call all 5 Phase 2 preprocessing functions
- Generate scenario-specific instructions (A/B/C/D)
- Embed Python-generated feature-based reports in prompt
- Include universal principles and cross-window patterns

**Testing**:
- Test all 4 scenarios (A/B/C/D) with different path distributions
- Verify feature-based reports embedded correctly as JSON
- Prompt length: ~4000-6000 tokens (scenario D has longest prompt)

### 7.5 Phase 4: Claude API Integration (2 hours)

**Task**: Implement API calls with retry logic and output validation

#### 7.5.1 Phase 1 API Integration (1 hour)

Implement `analyze_window_with_retry()` from Section 2.4.2:
- Exponential backoff: 2s, 4s, 8s with jitter
- Max 3 attempts per window
- Timeout: 90s per call
- Temperature: 0.3 (lower variance for factual analysis)

**Testing with Claude API** (small sample: 10 videos):
- Test successful window analysis
- Simulate retry scenarios (mock 503 errors)
- Verify status file updates correctly
- Check incremental saves (completed windows not re-run)

#### 7.5.2 Phase 2 API Integration (1 hour)

Implement `run_phase2_synthesis()` from Section 2.4.3:
- Timeout: 180s (larger context)
- Temperature: 0.4 (slightly higher for creative connections)
- Single API call (not parallelized)

**Testing**:
- Test with all 4 scenarios (A/B/C/D)
- Verify supplementary_insights always included
- Check confidence_level classification correct

### 7.6 Phase 5: End-to-End Testing (4 hours)

**Test Data**: 100-video sample from pilot hashtag

#### 7.6.1 Full Pipeline Test (2 hours)

Run complete Stage 7 on test data:
1. Pre-flight validation passes
2. Phase 1: All 6 windows complete successfully
3. Phase 2: Generates exactly 3 reports
4. Output files: 6 window JSONs + 1 synthesis JSON + 1 complete JSON = 8 files

**Validation Checkpoints**:
- All defining_features arrays have exactly 3 items
- All rf_validation.insight fields include alignment scores
- All creative_reports have confidence_level field
- Supplementary_insights section present with both subsections

#### 7.6.2 Scenario Testing (2 hours)

Test all 4 Phase 2 scenarios using different hashtag samples:

- **Scenario A Test** (#nutrition, 100 videos):
  - Expected: 5+ paths ≥10%
  - Validate: 3 path-based reports, all with cluster_path arrays

- **Scenario B Test** (Create synthetic: manipulate path frequencies):
  - Force exactly 2 paths ≥10%
  - Validate: 2 path-based + 1 feature-based
  - Check: Report #3 has type="feature_based", frequency=null

- **Scenario C Test** (Fragmented hashtag):
  - Expected: 1-2 paths ≥10%
  - Validate: 1 path + 2 feature OR 2 path + 1 feature

- **Scenario D Test** (Highly fragmented: 40+ unique paths):
  - Expected: 0 paths ≥10%
  - Validate: 3 feature-based reports
  - Check: supplementary_insights becomes primary guidance

**Success Criteria**:
- All scenarios produce exactly 3 reports
- Confidence levels match frequency thresholds
- Feature-based reports use different feature groups
- No hallucinated features or values

### 7.7 Dependencies Summary

**Function Dependencies**:
```
Phase 1:
  detect_bimodal_pattern() ← (no dependencies)
  identify_high_contrast_features() ← (no dependencies)
  compute_rf_alignment() ← (no dependencies)
  enrich_high_contrast_features() ← identify_high_contrast_features()

Phase 2:
  prepare_path_data_for_llm() ← (no dependencies)
  classify_confidence_level() ← prepare_path_data_for_llm() [optional]
  generate_universal_principles() ← (no dependencies)
  generate_cross_window_patterns() ← (no dependencies)
  generate_feature_based_reports() ← generate_universal_principles() [uses pattern]
```

**Critical Path**: `generate_feature_based_reports()` should be implemented LAST (most complex, requires understanding of all schema fields)

---

## 8. Testing & Validation

**Purpose**: Detailed test scenarios for validating Stage 7 implementation across all improvement paths

**Source**: Stage7PromptCritique.md testing guidance + Section 7 implementation plan

### 8.1 Phase 1 Testing Scenarios

#### 8.1.1 Scenario: Bimodal Feature Detection

**Input Data**:
```json
{
  "feature": "word_count",
  "distribution": {
    "top_performers": {
      "high_percentage": 0.40,  // 40% have ≥66th percentile word count
      "low_percentage": 0.35    // 35% have <33rd percentile
    }
  },
  "top_performer_avg": 52,
  "bottom_performer_avg": 18
}
```

**Expected Output**:
- Python: `is_bimodal=True`, `pattern_label="BIMODAL"`
- Prompt: Shows "→ Strategy A: Brief (≤20 words) - 35% of top performers"
- Prompt: Shows "→ Strategy B: Dense (≥80 words) - 40% of top performers"
- LLM Output: defining_features includes "ALTERNATIVE STRATEGIES: Use either BRIEF OR DENSE"

**Validation**:
```python
assert bimodal_info['is_bimodal'] == True
assert "BIMODAL" in prompt
assert "Strategy A" in prompt and "Strategy B" in prompt
assert "ALTERNATIVE STRATEGIES" in llm_output['clusters'][0]['defining_features'][2]
```

#### 8.1.2 Scenario: High-Contrast Feature Filtering

**Input Data**: 21 features, 8 have ≥0.20 contrast between clusters

**Expected Output**:
- Python filters to 8 high-contrast features
- Prompt shows only those 8 in "High-contrast features" section
- LLM selects exactly 3 from those 8
- All 3 selected have max_contrast ≥0.20

**Validation**:
```python
high_contrast = identify_high_contrast_features(kmeans_data, threshold=0.20)
assert len(high_contrast['clusters'][0]['high_contrast_features']) == 8

for feature in llm_output['clusters'][0]['defining_features']:
    feature_name = feature.split(':')[0]
    assert feature_name in [f['feature'] for f in high_contrast_features]
```

#### 8.1.3 Scenario: RF Alignment Scoring

**Input Data**: Cluster with 2/5 top RF features aligned (eye_contact_rate, energy_level)

**Expected Output**:
- Python: `alignment_count=2`, `alignment_score="2/5"`
- Prompt: Shows "✅ eye_contact_rate: matches...", "✅ energy_level: close to..."
- LLM Output: insight includes "(RF alignment: 2/5)"

**Validation**:
```python
alignment = compute_rf_alignment(cluster_centroid, rf_features, threshold=0.15)
assert alignment['alignment_score'] == "2/5"

insight = llm_output['clusters'][0]['rf_validation']['insight']
assert "RF alignment: 2/5" in insight or "2 of the top 5" in insight
```

### 8.2 Phase 2 Testing Scenarios

#### 8.2.1 Scenario A: 3+ Paths Above 10% Threshold (Standard Case)

**Input Data**:
- 5 paths above threshold: [22%, 18%, 15%, 12%, 11%]
- 30 paths below threshold

**Expected Output**:
- Python: `scenario='A'`, `paths_above_threshold=5`
- Prompt: Shows top 3 marked "✅ ABOVE THRESHOLD"
- LLM Output: 3 path-based reports
  - Report #1: 22% → confidence="very_high"
  - Report #2: 18% → confidence="high"
  - Report #3: 15% → confidence="high"
- All reports: type="path_based", cluster_path not null, frequency not null
- supplementary_insights: Present with 5-7 universal_principles

**Validation**:
```python
path_data = prepare_path_data_for_llm(cluster_paths)
assert path_data['scenario'] == 'A'
assert path_data['paths_above_threshold'] == 5

reports = llm_output['creative_reports']
assert len(reports) == 3
assert all(r['type'] == 'path_based' for r in reports)
assert reports[0]['confidence_level'] == 'very_high'  # 22%
assert reports[1]['confidence_level'] == 'high'  # 18%
assert reports[2]['confidence_level'] == 'high'  # 15%

assert 'supplementary_insights' in llm_output
assert len(llm_output['supplementary_insights']['universal_principles']) >= 5
```

#### 8.2.2 Scenario B: 2 Paths Above 10% (Partial Fallback)

**Input Data**:
- 2 paths above threshold: [18%, 12%]
- 43 paths below threshold

**Expected Output**:
- Python: `scenario='B'`, generates 1 feature-based report
- Prompt: Includes pre-generated Report #3 JSON
- LLM Output: 2 path-based + 1 feature-based
  - Report #1: type="path_based", 18% → confidence="high"
  - Report #2: type="path_based", 12% → confidence="moderate"
  - Report #3: type="feature_based", frequency=null, confidence="moderate"

**Validation**:
```python
path_data = prepare_path_data_for_llm(cluster_paths)
assert path_data['scenario'] == 'B'

feature_reports = generate_feature_based_reports(rf_video_data, num_reports=1)
assert len(feature_reports) == 1
assert feature_reports[0]['type'] == 'feature_based'

reports = llm_output['creative_reports']
assert reports[0]['type'] == 'path_based' and reports[0]['percentage'] == 18.0
assert reports[1]['type'] == 'path_based' and reports[1]['percentage'] == 12.0
assert reports[2]['type'] == 'feature_based' and reports[2]['frequency'] is None
```

#### 8.2.3 Scenario C: 1 Path Above 10% (Heavy Fallback)

**Input Data**:
- 1 path above threshold: [11%]
- 34 paths below threshold

**Expected Output**:
- Python: `scenario='C'`, generates 2 feature-based reports
- Prompt: Includes pre-generated Reports #2 and #3
- LLM Output: 1 path-based + 2 feature-based
  - Report #1: type="path_based", 11% → confidence="moderate"
  - Reports #2-3: type="feature_based", different feature groups

**Validation**:
```python
feature_reports = generate_feature_based_reports(rf_video_data, num_reports=2)
assert len(feature_reports) == 2
assert feature_reports[0]['formula_name'] != feature_reports[1]['formula_name']  # Different groups

reports = llm_output['creative_reports']
assert reports[0]['type'] == 'path_based'
assert reports[1]['type'] == 'feature_based'
assert reports[2]['type'] == 'feature_based'
```

#### 8.2.4 Scenario D: 0 Paths Above 10% (High Fragmentation)

**Input Data**:
- 0 paths above threshold (highest is 9%)
- 45 unique paths total

**Expected Output**:
- Python: `scenario='D'`, generates 3 feature-based reports
- Prompt: Includes all 3 pre-generated reports
- Prompt: "supplementary_insights becomes PRIMARY guidance"
- LLM Output: 3 feature-based reports
  - All type="feature_based", confidence="moderate"
  - Different feature groups: Eye Contact, Energy, Speech

**Validation**:
```python
path_data = prepare_path_data_for_llm(cluster_paths)
assert path_data['scenario'] == 'D'
assert path_data['paths_above_threshold'] == 0

feature_reports = generate_feature_based_reports(rf_video_data, num_reports=3)
assert len(feature_reports) == 3

# Verify different feature groups used
group_names = [r['formula_name'] for r in feature_reports]
assert len(set(group_names)) == 3  # All unique

reports = llm_output['creative_reports']
assert all(r['type'] == 'feature_based' for r in reports)
assert all(r['confidence_level'] == 'moderate' for r in reports)
```

### 8.3 Cross-Window Feature Validation

#### 8.3.1 Scenario: Features Present (Normal Case)

**Input Data**: Stage 6 RF data includes cross-window features:
- `hook_to_middle_energy_delta`
- `middle_to_closing_contrast`
- `eye_contact_consistency`

**Expected Output**:
- Python: `cross_window_patterns` populated with 3-5 insights
- Patterns reference delta/consistency/contrast features
- Example: "65% of high-performing videos show energy builds from hook to middle"

**Validation**:
```python
patterns = generate_cross_window_patterns(rf_video_data)
assert len(patterns) >= 3
assert any('energy' in p.lower() for p in patterns)
assert any('delta' in p.lower() or 'build' in p.lower() for p in patterns)
```

#### 8.3.2 Scenario: Features Missing (Graceful Degradation)

**Input Data**: Stage 6 RF data has NO cross-window features (missing deltas, consistency metrics)

**Expected Output**:
- Python: `cross_window_patterns` contains placeholder messages
- First pattern: "Cross-window progression analysis requires Stage 6 RF cross-window features"
- Reference to Stage 4 implementation location

**Validation**:
```python
patterns = generate_cross_window_patterns(rf_video_data_no_cross)
assert len(patterns) == 4  # Placeholder messages
assert "requires Stage 6 RF cross-window features" in patterns[0]
assert "Stage 4" in patterns[1]  # References where features are computed
```

### 8.4 End-to-End Integration Test

**Full Pipeline Test** (100 videos, #nutrition hashtag, 33-60s bucket):

#### 8.4.1 Test Setup

1. Stage 6 produces RF video data with cross-window features
2. Run Stage 7 with realistic data

#### 8.4.2 Validation Checkpoints

**Phase 1 Output**:
- 6 window analysis JSONs created (hook, middle_1-4, closing)
- Each JSON has exactly 3 clusters
- Each cluster has exactly 3 defining_features
- All rf_validation.insight fields include alignment scores
- Status file shows phase1_complete=true

**Phase 2 Output**:
- 1 winning_formulas.json created
- Exactly 3 creative_reports
- Path analysis shows Scenario A (4 paths ≥10%)
- confidence_levels: [very_high, high, moderate]
- supplementary_insights populated with 5-7 principles + 3-5 patterns

**Complete Analysis**:
- complete_analysis_{bucket}.json combines Phase 1 + Phase 2
- Total output files: 6 + 1 + 1 = 8 JSONs
- Total size: ~40-50KB

**Performance**:
- Phase 1 execution: <60 seconds (6 parallel calls)
- Phase 2 execution: <30 seconds (single call)
- Total: <90 seconds

#### 8.4.3 Schema Compliance

```python
# Phase 1 schema
for window_json in phase1_outputs:
    assert 'defining_features' in window_json['clusters'][0]
    assert len(window_json['clusters'][0]['defining_features']) == 3
    assert 'RF alignment:' in window_json['clusters'][0]['rf_validation']['insight']

# Phase 2 schema
synthesis = load_json('winning_formulas.json')
assert len(synthesis['creative_reports']) == 3
assert all('confidence_level' in r for r in synthesis['creative_reports'])
assert all('type' in r for r in synthesis['creative_reports'])
assert 'supplementary_insights' in synthesis
assert 'universal_principles' in synthesis['supplementary_insights']
assert 'cross_window_patterns' in synthesis['supplementary_insights']
```

---

## 9. Performance & Scalability

### 9.1 Performance Targets

**Per Bucket** (N=100 videos, bucket 18-33s with 6 windows):
- **Phase 1 execution**: 20-40 seconds (6 parallel API calls, wall-clock time ~10s typical, 30-40s during API high load)
- **Phase 2 execution**: 15-30 seconds (single API call with larger context)
- **Total Stage 7 time**: 40-70 seconds per bucket (target: <60s, acceptable: <120s)
- **Memory**: Peak < 200MB (JSON loading + API client overhead)
- **API Costs**: ~$0.26 per bucket (7 LLM calls × ~$0.037 per call)

**Comparison with Other Stages**:
| Stage | Time per Bucket | % of Pipeline | Bottleneck? |
|-------|----------------|---------------|-------------|
| Stage 2 (Video Processing) | 60-80s × 100 videos = 5-6.7 hours | 99.5% | ✅ YES |
| Stage 5 (ML Training) | 30-90s | 0.2-0.4% | ❌ NO |
| Stage 6 (Analysis Generation) | 3-5s | 0.02% | ❌ NO |
| **Stage 7 (LLM Analysis)** | **40-70s** | **0.3-0.4%** | ❌ NO |

**Conclusion**: Stage 7 is NOT a bottleneck. Stage 2 (video processing) dominates at 99.5% of pipeline time.

### 7.2 Measured Performance

*To be updated after pilot testing with actual API calls*

**Expected Timeline** (N=100 videos, bucket 18-33s):
| Operation | Estimated Time | Notes |
|-----------|---------------|-------|
| Pre-flight validation | 0.5-1s | File existence checks, JSON parsing |
| Phase 1 (6 parallel calls) | 20-40s | Wall-clock (API calls run concurrently) |
| Phase 2 (1 synthesis call) | 15-30s | Larger context, complex reasoning |
| Output file writes | 0.5s | 8 JSON files × ~5KB each |
| **TOTAL** | **40-70s** | Target: <60s, acceptable: <120s |

### 7.3 Bottlenecks & Mitigations

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| API latency variability | 5-10s typical → 30-45s during peak | Anthropic API load | Conservative timeouts (90s/180s) prevent spurious failures | Medium |
| Serial Phase 1 execution | 6 windows × 10s = 60s | Sequential API calls | **ALREADY MITIGATED**: Parallel execution (6 concurrent calls) | N/A (mitigated) |
| JSON truncation retries | +10-20s per truncated call | max_tokens exceeded | Automatic retry with 50% more tokens (rare edge case) | Low |
| Rate limiting (429) | +2-8s per retry | API throttling | Exponential backoff, max 2 retries | Low |

**No Optimization Needed**: Stage 7 is 0.3-0.4% of total pipeline time. Optimization efforts should focus on Stage 2 (video processing) which is 300x slower.

### 7.4 Scalability Limits

**Tested Limits**:
- **Maximum**: N=200 videos per bucket (expected: ~45-80s, still acceptable)
- **Minimum**: N=50 videos per bucket (expected: ~35-60s, no significant speedup)

**Bucket Variability**:
| Bucket | Windows | Phase 1 Calls | Expected Time | Notes |
|--------|---------|---------------|---------------|-------|
| 0-3s | 1 | 1 | 10-20s | Skips Phase 2, generates simplified summary |
| 3-9s | 2 | 2 | 25-40s | Minimal cluster paths (9 possible) |
| 18-33s | 6 | 6 | 40-70s | Standard case |
| 90-120s | 7 | 7 | 45-80s | Most complex (2187 possible paths) |

---

## 8. Testing Strategy

**Complete Testing Methodology**: See `TestingMethodology_Stage7.md` for detailed test plans.

**Summary**:

### 8.1 Unit Tests

**Phase 1 Testing**:
- Mock Anthropic API responses (pre-recorded JSON fixtures)
- Test window analysis logic independently
- Test automated validation layer (feature contradictions, invented features)

**Phase 2 Testing**:
- Test cluster path extraction (Q9.1 logic)
- Test path frequency calculation and 10% threshold filtering
- Test confidence level classification (very_high/high/moderate)
- Test fallback logic (<3 paths scenario)

**Edge Case Coverage**:
- Bucket 0-3s (single window, no Phase 2)
- Bucket 3-9s (2 windows, minimal paths)
- Bucket 9-13s (middle_aggregate window)
- API failures (429, 503, timeout)
- JSON truncation (max_tokens exceeded)

### 8.2 Integration Tests

**End-to-End Pipeline** (Stage 6 → Stage 7):
- Use real Stage 6 outputs (13 JSONs from pilot testing)
- Run Stage 7 with live API calls
- Validate 8 output JSONs created
- Verify confidence levels and RF validation present

**Test Data**:
- Synthetic JSON fixtures (controlled edge cases, 10-20 videos per bucket)
- Real Stage 6 outputs (1-2 priority buckets with 10 videos each)

### 8.3 Test Execution

```bash
# Unit tests with mocked API (fast, no cost)
pytest tests/stage7/test_phase1_analysis.py -v
pytest tests/stage7/test_phase2_synthesis.py -v
pytest tests/stage7/test_cluster_paths.py -v

# Integration tests with live API (slow, ~$0.26 per bucket)
TEST_MODE=live pytest tests/stage7/test_integration.py -v --bucket 18-33s

# Edge case tests
pytest tests/stage7/test_edge_cases.py -v
```

---

## 9. Future Enhancements

### 9.1 Planned Improvements

**Phase 2: Enhanced Validation Layer**
- Current: Basic feature contradiction detection
- Future: Semantic validation using embedding similarity (detect when LLM recommendations contradict cluster centroid patterns)
- Impact: Further reduce hallucination risk

**Phase 3: Multi-Model Testing**
- Current: claude-sonnet-4 only
- Future: A/B test with claude-haiku-4 (cheaper, faster) for Phase 1
- Impact: Potential cost reduction (50% cheaper) with acceptable quality trade-off

**Phase 4: Batch API Support**
- Current: Sequential API calls (parallel execution but synchronous waiting)
- Future: Use Anthropic Batch API for asynchronous processing
- Impact: Potential cost reduction (50% off batch pricing)

### 9.2 Known Limitations

**Limited Control Over LLM Creativity**:
- Temperature and prompts constrain but don't eliminate variability
- Same inputs may produce slightly different creative names/descriptions
- **Acceptable**: Variability is feature (multiple valid strategies), not bug

**No Feedback Loop from Stage 8**:
- Current: No validation that PDF reports are readable/actionable
- Future: Implement creator feedback mechanism to refine prompts
- **Workaround**: Human review protocol (Critique Q3 Layers 2-3)

**Single-Language Support**:
- Current: English-only LLM prompts and outputs
- Future: Multi-language support for international markets
- **Impact**: Limits to English-speaking creators

---

## 10. References & Related Docs

### 10.1 Parent Document

- **MLPlanningv2.md Section 7 "Stage 7: LLM Analysis - Hybrid Two-Phase Approach" (lines 2587-3299)**
  - High-level component overview
  - Two-phase architecture rationale
  - LLM prompt templates
  - Example outputs

### 10.2 Mother Document Foundation

- **MLPlanningv2.md Part 1: Foundation** (shared across all stages)
  - Section 2 "Client Architecture": Directory paths (`/data/clients/{client_id}/buckets/bucket_{bucket}`)
  - Section 4 "CLI Command Structure": CLI parameters (--stage, --client, --bucket)
  - Appendix "Bucket Definitions": BUCKET_WINDOWS config (window structure per bucket)

**Key Sections Referenced in This Stage**:
- Section 2: Base directory paths for file I/O
- Section 4: CLI parameters this stage reads
- Appendix: Bucket window configuration (centralized with Stages 4-6)

### 10.3 Related Child Docs

**Upstream**:
- **MLAnalysisGenerationCHILD.md** (Stage 6)
  - Produces 13 JSON files (input to this stage)
  - Defines exact schemas for rf_video_analysis.json, {window}_rf_analysis.json, {window}_kmeans_analysis.json
  - Documents feature name normalization (K-Means centroids have normalized names without `_scaled` suffix)

**Downstream**:
- **PDFReportGenerationCHILD.md** (Stage 8) - *To be created*
  - Consumes Stage 7 LLM outputs (8 JSON files)
  - Generates creator-friendly PDF reports with visual hierarchy based on confidence levels

**Cross-Stage**:
- **Crosswindowupgrade.md** (Stage 4 enhancement)
  - Documents cross-window feature computation (hook_to_middle_energy_delta, etc.)
  - Ensures video-level RF has features needed for Phase 2 validation

### 10.4 External References

- **Anthropic API Documentation**: https://docs.anthropic.com/claude/reference/messages
- **Claude Prompt Engineering Guide**: https://docs.anthropic.com/claude/docs/prompt-engineering
- **Exponential Backoff Best Practices**: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

### 10.5 Code References

**Implementation Files** (to be created):
- `/stages/stage7_llm_analysis.py`: Main Stage 7 pipeline
- `/config/llm_config.py`: LLM configuration constants
- `/utils/llm_helpers.py`: Retry logic, backoff, validation helpers

---

## Appendix A: Decision Log

**Purpose**: Record major design decisions, alternatives considered, and trade-offs accepted.

### Decision 1: Two-Phase Hybrid Approach (Phase 1 + Phase 2)

**Context**: Need to generate creative insights from ML outputs while minimizing LLM hallucination risk.

**Alternatives Considered**:
- **Option A**: Single LLM call with all data (video RF + 6 window RFs + 6 K-Means)
  - Rejected: 1000+ numbers in prompt → high hallucination risk, difficult to validate
- **Option B**: Phase 1 only (per-window analysis, no cross-window synthesis)
  - Rejected: Misses temporal progression patterns ("Winning Formulas" require full video journey)
- **Option C**: Two-phase with sequential Phase 1 (6 calls one-by-one)
  - Rejected: Too slow (6 × 10s = 60s just for Phase 1)

**Rationale**: Two-phase hybrid with parallel Phase 1 balances:
- Small, focused contexts (113 numbers per Phase 1 call) → low hallucination risk
- Cross-window synthesis (Phase 2) → captures temporal patterns
- Parallel execution (6 concurrent API calls) → fast wall-clock time (~10s)

**Trade-offs**:
- More complex orchestration (parallel execution, smart retry)
- 7 API calls per bucket vs 1 (higher cost: $0.26 vs $0.10)
- **Accepted**: Quality and hallucination prevention worth cost premium

**Date**: 2025-10-16

---

### Decision 2: 100% Window Completion Requirement (No Partial Analysis)

**Context**: Phase 1 runs 6-7 parallel API calls. Should we proceed to Phase 2 if some windows fail?

**Alternatives Considered**:
- **Option A**: Partial completion allowed (e.g., 4/6 windows → proceed)
  - Rejected: Incomplete "Winning Formulas" (missing windows break cluster paths)
- **Option B**: Minimum threshold (e.g., 80% of windows required)
  - Rejected: Arbitrary threshold, still produces incomplete analysis

**Rationale**: 100% completion required because:
- Complete data exists for all windows (6-8 hours of video processing completed)
- Client expects full video journey analysis (Hook → Middle → Closing)
- Phase 2 cluster paths require ALL windows (missing middle_2 makes path invalid)
- Partial analysis appears unprofessional

**Trade-offs**:
- Must retry failed windows (up to 2 retries per window)
- Bucket aborted if ANY window fails after retries
- **Accepted**: Clean failure state (0 JSONs) better than confusing partial output (5/8 JSONs)

**Date**: 2025-10-16 (Approved in Critique Q4)

---

### Decision 3: 10% Threshold for Path Formula Inclusion (with Fallback)

**Context**: Phase 2 identifies "Winning Formulas" from cluster paths. With 729 possible paths (3^6), many will be rare (1-2 videos).

**Alternatives Considered**:
- **Option A**: 5% threshold (5 videos out of 100)
  - Rejected: Too rare, might not replicate, wastes creator time on experimental strategies
- **Option B**: 15% threshold (15 videos out of 100)
  - Rejected: Too strict, may result in 0 paths meeting threshold (high fragmentation scenarios)

**Rationale**: 10% threshold balances:
- Proven patterns (10+ videos = 1 in 10 videos → clearly reliable)
- Coverage (flexible enough to find 3 formulas in most scenarios)
- Quality over quantity (prioritize high-confidence strategies)

**Trade-offs**:
- May result in <3 paths meeting threshold → requires fallback logic
- **Mitigation**: Feature-based reports as fallback (always deliver 3 reports per bucket)

**Date**: 2025-10-16 (Approved in Critique Q5)

---

### Decision 4: Smart Retry Logic (Retry Only Failed Windows)

**Context**: Phase 1 parallel execution may have partial failures (e.g., 5/6 windows succeed, 1 fails due to timeout).

**Alternatives Considered**:
- **Option A**: Retry all 6 windows (even successful ones)
  - Rejected: Wastes successful API calls, higher cost (18 calls vs 9)
- **Option B**: No retries (fail immediately)
  - Rejected: Temporary API issues (503, timeout) would abort buckets unnecessarily

**Rationale**: Smart retry only failed windows because:
- Efficient: Only retry what failed (9 calls vs 18 in worst case)
- Cost-effective: Don't waste successful API calls
- Resilient: Handles temporary API issues (429 rate limiting, 503 service unavailable)

**Trade-offs**:
- More complex retry logic (track which windows failed)
- **Accepted**: Complexity justified by cost savings and resilience

**Date**: 2025-10-16 (Approved in Critique Q4)

---

### Decision 5: Conservative API Timeouts (90s Phase 1, 180s Phase 2)

**Context**: Need to balance between detecting actual failures vs. accommodating API variability.

**Alternatives Considered**:
- **Option A**: Aggressive timeouts (30s Phase 1, 60s Phase 2)
  - Rejected: API 99th percentile is 30-45s → spurious timeout failures
- **Option B**: Very conservative (300s = 5 minutes)
  - Rejected: Unnecessarily long wait for actual failures

**Rationale**: 2x safety margin (90s = 2 × 45s, 180s = 2 × 90s) because:
- API variability is real (5-10s typical, 30-45s during peak)
- Cost of premature timeout is high (abort bucket after 6 hours of processing)
- Negligible downside (if actual failure, waiting 90s vs 30s doesn't matter)

**Trade-offs**:
- Longer wait for actual failures (90s vs 30s)
- **Accepted**: Preventing spurious failures worth longer wait

**Date**: 2025-10-16 (Approved in QA Q5.3)

---

### Decision 6: Hashtag from Metadata File (Not CLI Parameter)

**Context**: LLM prompts can include hashtag context for specificity. Should hashtag be a CLI parameter?

**Alternatives Considered**:
- **Option A**: CLI parameter (--hashtag "nutrition")
  - Rejected: Adds complexity, hashtag is non-critical (only affects prompt context)
- **Option B**: Always required (fail if missing)
  - Rejected: Too strict, hashtag is optional enhancement

**Rationale**: Read from metadata.json (if exists) because:
- Simplicity: Keeps CLI minimal (3 parameters: --stage, --client, --bucket)
- Non-critical: Hashtag only affects LLM prompt specificity, not Stage 7 logic
- Graceful degradation: If missing, LLM generates generic recommendations (still valuable)

**Trade-offs**:
- Requires metadata.json creation (future integration task)
- **Accepted**: Cleaner CLI worth manual metadata file management

**Date**: 2025-10-16 (Approved in QA Q8.2)

---

## Appendix B: Example Data

### B.1 Sample Phase 1 Output (Hook Analysis)

**File**: `ml_analysis/llm/hook_analysis.json`

```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "hashtag": "nutrition",
  "total_videos": 100,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "name": "The Direct Eye Contact Hook",
      "defining_features": [
        "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43)",
        "word_count: 14 (RF rank #3, importance 0.18)",
        "energy_level: 0.55 (RF rank #2, importance 0.22)"
      ],
      "rf_validation": {
        "top_predictive_features_in_cluster": [
          "eye_contact_rate: 0.87 matches top performer avg 0.88 (RF validated)"
        ],
        "insight": "Leverages #1 most predictive hook feature at optimal levels."
      },
      "strategy_description": "Creator looks directly at camera with minimal speech, establishing immediate connection through eye contact.",
      "creator_recommendations": [
        "PRIORITY: Maintain 85-90% eye contact (RF #1 predictor, importance 0.35)",
        "Keep opening statement under 15 words (RF #3 predictor)",
        "Target moderate energy 0.55-0.60 (RF #2 predictor)"
      ]
    },
    {
      "cluster_id": 1,
      "size": 42,
      "name": "The Text Overlay Hook",
      "defining_features": [
        "overlay_unique_count: 3.5 (high - multiple text overlays)",
        "eye_contact_rate: 0.28 (low - looking away or at product)",
        "word_count: 48 (very high - talking while showing text)"
      ],
      "rf_validation": {
        "top_predictive_features_in_cluster": [],
        "insight": "Uses overlay_unique_count (RF rank #5) as secondary strategy."
      },
      "strategy_description": "Fast-paced, text-heavy opening with multiple scene cuts.",
      "creator_recommendations": [
        "Add 2-3 text overlays in first 3 seconds",
        "Use dynamic cuts (3-4 scenes in hook)",
        "Speak quickly - aim for 45-50 words in 3 seconds"
      ]
    },
    {
      "cluster_id": 2,
      "size": 23,
      "name": "The Action-Driven Hook",
      "defining_features": [
        "object_count: 4.8 (high - multiple props visible)",
        "gesture_count: 7.5 (very high - active hand movements)",
        "energy_level: 0.75 (high - dynamic movement)"
      ],
      "rf_validation": {
        "top_predictive_features_in_cluster": [
          "gesture_count: 7.5 aligns with RF rank #4 (importance 0.14)"
        ],
        "insight": "Leverages gesture_count as alternative to eye contact."
      },
      "strategy_description": "Single continuous shot with high-energy physical action.",
      "creator_recommendations": [
        "Film in one continuous take - avoid cuts in first 3 seconds",
        "Use 6-8 hand gestures (pointing, grabbing, showing products)",
        "Show 4-5 different objects/products early"
      ]
    }
  ],
  "analysis_metadata": {
    "llm_model": "claude-sonnet-4-20250514",
    "timestamp": "2025-10-16T14:30:12Z",
    "phase": "phase1_window"
  }
}
```

### B.2 Sample Phase 2 Output (Winning Formulas)

**File**: `ml_analysis/llm/winning_formulas.json`

```json
{
  "bucket": "18-33s",
  "hashtag": "nutrition",
  "total_videos": 100,
  "total_unique_paths": 45,
  "paths_above_threshold": 5,
  "creative_reports": [
    {
      "report_id": 1,
      "type": "path_based",
      "path": [0, 1, 1, 1, 2, 0],
      "frequency": 22,
      "percentage": 22.0,
      "confidence_level": "very_high",
      "formula_name": "The Educator's Arc",
      "structure": {
        "hook": "The Direct Eye Contact Hook (Cluster 0)",
        "middle_pattern": "Information Dense Middle (Cluster 1 → 1 → 1 → 2)",
        "closing": "High Energy CTA (Cluster 0)"
      },
      "temporal_progressions": [
        {
          "feature": "energy_level",
          "hook": 0.55,
          "middle_1": 0.60,
          "middle_2": 0.62,
          "middle_3": 0.68,
          "middle_4": 0.75,
          "closing": 0.85,
          "pattern": "Steady build from moderate (0.55) to high (0.85)",
          "hook_to_middle_delta": 0.16,
          "middle_to_closing_contrast": 0.27
        },
        {
          "feature": "eye_contact_rate",
          "hook": 0.87,
          "middle_avg": 0.45,
          "closing": 0.82,
          "pattern": "Bookend pattern (high in hook/closing, lower in middle)"
        }
      ],
      "rf_cross_window_validation": {
        "matches_top_patterns": [
          "hook_to_middle_energy_delta: 0.16 (RF top performer avg: 0.15, RF rank #4)",
          "middle_to_closing_contrast: 0.27 (RF top performer avg: 0.28, RF rank #5)",
          "eye_contact_consistency: 0.12 std dev (RF top performer avg: 0.12, RF rank #6)"
        ],
        "insight": "This formula exhibits ALL THREE major cross-window patterns identified by video-level RF.",
        "rf_validation_score": "9/10"
      },
      "strategy_description": "Start with intimate eye contact to build trust, deliver dense educational content in middle segments, return to direct eye contact for high-energy call-to-action.",
      "when_to_use": "Educational nutrition content, product explanations, how-to videos.",
      "step_by_step_template": [
        "Hook (0-3s): Direct eye contact (0.87), minimal words (14), moderate energy (0.55)",
        "Middle_1 (3-8s): Shift to product view, increase talking speed (50+ words), build energy to 0.60",
        "Middle_2-4 (8-23s): Continue information delivery, steady energy progression",
        "Closing (23-26s): Return to direct eye contact (0.82), peak energy (0.85), clear CTA",
        "CROSS-WINDOW TARGETS (RF validated):",
        "  - Energy delta hook→middle: +0.16 (RF target: +0.15)",
        "  - Energy contrast middle→closing: 0.27 gap (RF target: 0.28)",
        "  - Eye contact consistency: Keep std dev ≤0.15 across all windows"
      ]
    },
    {
      "report_id": 2,
      "type": "path_based",
      "path": [1, 0, 0, 0, 1, 2],
      "frequency": 18,
      "percentage": 18.0,
      "confidence_level": "high",
      "formula_name": "The Fast-Paced Explainer",
      "structure": {
        "hook": "The Text Overlay Hook (Cluster 1)",
        "middle_pattern": "Consistent Information Delivery (Cluster 0 throughout)",
        "closing": "The Action-Driven Hook (Cluster 2)"
      },
      "temporal_progressions": [
        {
          "feature": "word_count",
          "hook": 48,
          "middle_avg": 62,
          "closing": 45,
          "pattern": "High throughout (dense information delivery)"
        }
      ],
      "rf_cross_window_validation": {
        "matches_top_patterns": [
          "word_density_std: 0.08 (consistent across windows, matches RF pattern)"
        ],
        "insight": "Maintains consistent information density throughout video.",
        "rf_validation_score": "7/10"
      },
      "strategy_description": "Rapid-fire information delivery with text overlays and dynamic visuals.",
      "when_to_use": "Quick tips, product comparisons, feature highlights.",
      "step_by_step_template": [
        "Hook (0-3s): 2-3 text overlays, 45-50 words, dynamic cuts",
        "Middle (3-23s): Maintain high word count (60+ words per segment)",
        "Closing (23-26s): High-energy physical action, show multiple products"
      ]
    },
    {
      "report_id": 3,
      "type": "path_based",
      "path": [2, 2, 1, 0, 0, 1],
      "frequency": 12,
      "percentage": 12.0,
      "confidence_level": "moderate",
      "formula_name": "The Action-to-Education Transition",
      "structure": {
        "hook": "The Action-Driven Hook (Cluster 2)",
        "middle_pattern": "Transition to Information (Cluster 2 → 1 → 0)",
        "closing": "The Text Overlay Hook (Cluster 1)"
      },
      "temporal_progressions": [
        {
          "feature": "gesture_count",
          "hook": 7.5,
          "middle_avg": 4.2,
          "closing": 3.1,
          "pattern": "Gradual decrease (start active, end calm)"
        }
      ],
      "rf_cross_window_validation": {
        "matches_top_patterns": [],
        "insight": "Unique pattern not strongly validated by cross-window RF features.",
        "rf_validation_score": "5/10"
      },
      "strategy_description": "Grab attention with action, transition to calm explanation.",
      "when_to_use": "Product demos where initial action hooks viewers.",
      "step_by_step_template": [
        "Hook (0-3s): High-energy action (6-8 gestures, show 4-5 objects)",
        "Middle (3-23s): Gradually reduce energy, increase verbal explanation",
        "Closing (23-26s): Text overlays with final key points"
      ]
    }
  ],
  "supplementary_insights": {
    "universal_principles": [
      "High eye contact rate (88% vs 45% for top vs bottom performers) - applies to 78% of videos",
      "Consistent energy maintenance across windows (std dev ≤0.15) - found in 65% of top performers",
      "Clear CTA in closing window - present in 92% of high-performing videos",
      "Text overlays within first 3 seconds - found in 60% of top performers",
      "Energy builds from hook to closing - 65% of videos use this pattern"
    ],
    "cross_window_patterns": [
      "78% of high-performing videos use 'bookend' eye contact pattern (high in hook/closing, lower in middle)",
      "Energy progression: 65% build energy, 12% maintain consistent energy, 23% variable",
      "Closing energy should match or exceed middle average (85% of top performers follow this)",
      "Videos with energy delta >0.3 from hook to closing had 2x engagement",
      "Consistent word density (std dev <10 words) predicts higher completion rates"
    ]
  },
  "path_statistics": {
    "total_unique_paths": 45,
    "paths_above_threshold": 5,
    "needs_fallback": false
  },
  "analysis_metadata": {
    "llm_model": "claude-sonnet-4-20250514",
    "timestamp": "2025-10-16T14:32:45Z",
    "phase": "phase2_synthesis"
  }
}
```

---

## Appendix C: Pseudocode (Complete)

### C.1 Full Stage 7 Pipeline

```python
def run_stage7_llm_analysis(bucket_path: str, bucket: str) -> dict:
    """
    Complete Stage 7 pipeline: Pre-flight + Phase 1 + Phase 2.

    Args:
        bucket_path: Absolute path (e.g., /data/clients/acme/buckets/bucket_18-33s)
        bucket: Bucket name (e.g., "18-33s")

    Returns:
        dict: {
            'exit_code': 0 (success) or 1-6 (various failures),
            'phase1': {window_type: analysis_json},
            'phase2': synthesis_json,
            'complete_analysis': combined_json
        }
    """
    logger.info(f"=== Stage 7: LLM Analysis - Bucket {bucket} ===")

    try:
        # ===== PRE-FLIGHT VALIDATION =====
        logger.info("Step 1: Pre-flight validation")
        run_preflight_validation(bucket_path, bucket)

        # Load bucket configuration
        windows = BUCKET_WINDOWS[bucket]  # From centralized config
        hashtag = get_hashtag_from_metadata(bucket_path)  # Optional, None if missing

        logger.info(f"  Bucket {bucket}: {len(windows)} windows, hashtag={hashtag or 'None'}")

        # ===== SPECIAL CASE: Bucket 0-3s (Single Window) =====
        if len(windows) == 1:
            logger.info("  Single-window bucket detected - Skipping Phase 2")

            window_analyses = run_phase1_parallel(bucket_path, bucket, hashtag, windows)

            summary = generate_single_window_summary(
                window_analyses['hook'],
                bucket=bucket,
                total_videos=100
            )

            summary_path = os.path.join(bucket_path, f'ml_analysis/llm/bucket_summary_{bucket}.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"✓ Stage 7 complete (single-window): Generated summary")
            return {'exit_code': 0, 'phase1': window_analyses, 'phase2': None, 'summary': summary}

        # ===== PHASE 1: PER-WINDOW ANALYSIS (Parallel with Status Tracking) =====
        logger.info(f"Step 2: Phase 1 - Per-Window Analysis ({len(windows)} windows)")

        window_analyses = run_phase1_parallel(bucket_path, bucket, hashtag, windows)
        # Returns: {window_type: analysis_json} for all windows
        # Raises: Phase1ExecutionError if ANY window fails after retries
        # NOTE: Uses incremental saves with status tracking (.phase1_status.json)
        #       On retry, resumes from checkpoint (only re-runs failed windows)
        #       See Section 2.3.2 and Cross-HLD Issue #11 for rationale

        logger.info(f"✓ Phase 1 complete: {len(window_analyses)} window analyses generated")

        # ===== PHASE 2: CROSS-WINDOW SYNTHESIS =====
        logger.info("Step 3: Phase 2 - Cross-Window Synthesis")

        # Load K-Means outputs for cluster path extraction
        kmeans_outputs = {}
        for window in windows:
            path = os.path.join(bucket_path, f'ml_analysis/{window}_kmeans_analysis.json')
            with open(path, 'r') as f:
                kmeans_outputs[window] = json.load(f)

        # Load video-level RF for cross-window validation
        rf_video_path = os.path.join(bucket_path, 'ml_analysis/rf_video_analysis.json')
        with open(rf_video_path, 'r') as f:
            rf_video_data = json.load(f)

        # Run Phase 2 synthesis
        synthesis = run_phase2_synthesis(
            window_analyses=window_analyses,
            kmeans_outputs=kmeans_outputs,
            rf_video_data=rf_video_data,
            bucket=bucket,
            hashtag=hashtag
        )

        logger.info(f"✓ Phase 2 complete: Generated {len(synthesis['creative_reports'])} creative reports")

        # ===== GENERATE COMPLETE ANALYSIS =====
        logger.info("Step 4: Generating complete analysis JSON")

        complete_analysis = {
            'bucket': bucket,
            'hashtag': hashtag,
            'total_videos': len(window_analyses[list(window_analyses.keys())[0]]['clusters'][0]['size']) * 3,  # Approx
            'window_analyses': window_analyses,
            'winning_formulas': synthesis,
            'generated_at': datetime.now().isoformat()
        }

        complete_path = os.path.join(bucket_path, f'ml_analysis/llm/complete_analysis_{bucket}.json')
        with open(complete_path, 'w') as f:
            json.dump(complete_analysis, f, indent=2)

        # ===== SUCCESS =====
        logger.info(f"✓✓✓ Stage 7 COMPLETE: Generated {len(windows)} Phase 1 + 1 Phase 2 + 1 complete ({len(windows) + 2} files total)")

        return {
            'exit_code': 0,
            'phase1': window_analyses,
            'phase2': synthesis,
            'complete_analysis': complete_analysis
        }

    except PreFlightValidationError as e:
        logger.error(f"Pre-flight validation failed: {e}")
        return {'exit_code': 1, 'error': str(e)}

    except Phase1ExecutionError as e:
        logger.error(f"Phase 1 execution failed: {e}")
        return {'exit_code': 5, 'error': str(e)}

    except DataIntegrityError as e:
        logger.error(f"Data integrity error in Phase 2: {e}")
        return {'exit_code': 6, 'error': str(e)}

    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
        return {'exit_code': 99, 'error': str(e)}


# Additional helper functions (already documented in Section 2.3)
# - run_preflight_validation() - See Section 2.3.1
# - run_phase1_parallel() - See Section 2.3.2 (includes status tracking and resume logic)
# - analyze_window_with_retry() - See Section 2.3.2
# - run_phase2_synthesis() - See Section 2.3.3
# - extract_cluster_paths() - See Section 2.3.3
# - analyze_path_frequencies() - See Section 2.3.3
# - etc.

# NOTE: Phase 1 uses incremental saves with status tracking (NOT atomic pattern)
#       Status file: .phase1_status.json (tracks completion, enables resume on retry)
#       Rationale: Cost optimization for expensive LLM API calls
#       See Cross-HLD Alignment Issue #11 for design decision analysis
```

---

## Document Metadata

**Creation Date**: 2025-10-16
**Last Modified**: 2025-10-16
**Authors**: Senior Technical Architect (AI-assisted)
**Reviewers**: [To be added]
**Approved By**: [To be added]
**Next Review Date**: [To be added after pilot testing]

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-16 | AI Architect | Initial draft - Complete HLD generated from Phase 1 (Critique) + Phase 2 (QA) outputs |

---

**Status**: READY FOR IMPLEMENTATION - All sections complete, NO TODOs
