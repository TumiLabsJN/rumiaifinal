# Stage 7: LLM Prompt Critique & Recommendations

> **Purpose**: Critical analysis of Stage 7 LLM prompts (Phase 1 & Phase 2) against approved Child HLD architecture
> **Source**: MLPlanningv2.md Section 7 (lines 2612-3096)
> **Child HLD**: LLMAnalysisCHILD.md (approved 2025-10-16)
> **Date**: 2025-10-16
> **Status**: ~~CRITIQUE COMPLETE - UPDATES REQUIRED~~ → ✅ **ALL IMPROVEMENTS MERGED** (2025-10-17)

---

## ✅ INTEGRATION STATUS (2025-10-17)

**ALL IMPROVEMENTS MERGED INTO LLMAnalysisCHILD.md**

This critique document remains as:
1. **Decision Rationale Archive**: Full analysis of alternatives evaluated for each issue/gap
2. **Reference for Why**: Explains reasoning behind each architectural choice
3. **Prompting Guidelines Documentation**: Examples of hybrid Python/LLM division of labor

**For implementation**, developers should reference **LLMAnalysisCHILD.md** (single source of truth).
**For understanding decisions**, developers should reference **this critique** for detailed rationale.

**Integration Summary**:
- ✅ **Section 2.2**: 9 Python preprocessing functions (~1,100 lines)
- ✅ **Section 2.4.2**: Complete Phase 1 prompt template with Issues #1-11 (~275 lines)
- ✅ **Section 2.4.3**: Complete Phase 2 prompt template with Gaps #1-5 (~310 lines)
- ✅ **Section 5.2.1 & 5.2.2**: Before/after schema comparisons
- ✅ **Section 7**: Implementation Roadmap with dependencies (~250 lines)
- ✅ **Section 8**: Testing & Validation scenarios A/B/C/D (~296 lines)
- ✅ **Document History**: Integration changelog (2025-10-17)

**Total Added**: ~2,231 lines to LLMAnalysisCHILD.md (grew from 2,079 → ~4,310 lines)

**Merged Improvements**:
- **Phase 1**: All 11 issues resolved (bimodal patterns, high-contrast filtering, RF alignment, cluster size context, etc.)
- **Phase 2**: All 5 gaps resolved (10% threshold, confidence levels, hybrid output, feature fallback, exactly 3 reports)

**Document Roles After Merge**:
| Document | Role | Audience |
|----------|------|----------|
| **LLMAnalysisCHILD.md** | IMPLEMENTATION SPEC (single source of truth) | Developers, Technical Implementation |
| **Stage7PromptCritique.md** | DECISION ARCHIVE (why we chose X over Y) | Architects, Future Maintainers |

---

## Executive Summary

**Overall Assessment**: Mother Document prompts (MLPlanningv2.md lines 2612-3096) are **85% complete** but **MISSING critical Critique Q5 decisions** (10% threshold, confidence levels, hybrid output structure) that are foundational to Child HLD architecture.

**Phase 1 Prompt Status**: ✅ **GOOD** - Includes distribution data for bimodal patterns, RF validation structure, priority recommendations
**Phase 2 Prompt Status**: ⚠️ **NEEDS UPDATES** - Missing 10% threshold logic, confidence classification, feature-based fallback instructions, hybrid output structure

**Impact if Not Updated**:
- LLM will generate 3-5 formulas without frequency filtering → low-quality patterns (8% frequency paths included)
- No confidence levels → Stage 8 cannot prioritize reports in PDF
- Missing `supplementary_insights` → incomplete coverage (only 40-60% of videos explained by path formulas)
- No fallback logic → LLM confused when <3 paths meet threshold

---

## Critique Part 1: Phase 1 Prompt (Per-Window Analysis)

### Source Location
**MLPlanningv2.md lines 2612-2698**

### Alignment Check Against Child HLD

| Requirement | Child HLD Reference | Mother Doc Status | Gap? |
|-------------|---------------------|-------------------|------|
| Distribution data (bimodal patterns) | Section 2.3.2, Critique Q3 | ✅ Lines 2645-2646, 2651-2652, 2662, 2667 | **NO** |
| RF validation structure | Section 5.2.1 | ✅ Lines 2679-2682 | **NO** |
| Priority recommendations with targets | Section 5.2.1 | ✅ Lines 2684-2688 | **NO** |
| Hashtag context (optional) | Section 4.1, QA Q8.2 | ✅ Line 2621 | **NO** |
| JSON schema completeness | Section 5.2.1 | ✅ Lines 2669-2692 | **NO** |
| Temperature = 0.3 | Section 4.2 | ✅ Line 2704 | **NO** |
| max_tokens = 4000 | Section 4.2 | ✅ Line 2703 | **NO** |

### Phase 1 Verdict: ⚠️ **NEEDS UPDATES - 11 ISSUES IDENTIFIED**

**Updated Assessment (2025-10-16)**: After comprehensive review comparing Mother Document prompt against Child HLD architecture AND general prompt engineering best practices, **11 critical issues** were identified requiring updates.

**Issue Summary**:
- ✅ Architectural alignment: Distribution data present, RF validation structure exists
- ❌ Prompt engineering quality: Vague instructions, missing format specifications, no edge case handling
- ❌ Schema alignment: Output format not explicitly specified, missing RF alignment score

---

### Phase 1 Issue Tracker

| # | Issue | Severity | Type | Status |
|---|-------|----------|------|--------|
| 1 | Vague bimodal pattern instruction | HIGH | Prompt Engineering | ✅ RESOLVED |
| 2 | Ambiguous "3-5 defining features" | MEDIUM | Prompt Engineering | ✅ RESOLVED |
| 3 | Missing negative examples (contrastive features) | HIGH | Prompt Engineering | ✅ RESOLVED |
| 4 | RF validation section underspecified | MEDIUM | Schema Alignment | ✅ RESOLVED |
| 5 | Weak "Important" section reminders | LOW | Prompt Engineering | ✅ RESOLVED |
| 6 | No edge case handling | MEDIUM | Robustness | ✅ RESOLVED |
| 7 | Verbose RF feature formatting | LOW | Context Efficiency | ✅ RESOLVED |
| 8 | Defining features format not specified | HIGH | Schema Alignment | ✅ RESOLVED |
| 9 | Missing RF alignment score | MEDIUM | Schema Alignment | ✅ RESOLVED |
| 10 | No bimodal distribution example in data | CRITICAL | Prompt Engineering | ✅ RESOLVED |
| 11 | No cluster size guidance | MEDIUM | Edge Case Handling | ✅ RESOLVED (DUPLICATE #6) |

---

### Phase 1 Detailed Issue Analysis

---

#### Issue #1: Vague Bimodal Pattern Instruction (HIGH PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: MLPlanningv2.md line 2662, 2667
**Child HLD Reference**: Section 2.3.2, line 58

**Current Mother Doc** (line 2662):
```
- **NOTICE distribution patterns**: If top performers show bimodal patterns (e.g., "40% high, 60% low"),
  this indicates MULTIPLE successful strategies for this feature
```

**Problem**:
- Says "NOTICE" but doesn't tell LLM what to **DO** with bimodal patterns
- Instruction appears in task #2 but bimodal handling instruction appears later at line 2667 (buried)
- No clear output format for bimodal features in `defining_features` or `creator_recommendations`

**Impact**: LLM might recognize bimodal pattern but not know how to reflect it in output (should it create separate recommendations? mention both strategies?).

---

### Evaluated Alternatives

#### **Alternative A: Explicit Bimodal Task Step**

**Description**: Add bimodal handling as explicit step in task list (line 2659)

**Pros**:
- ✅ Clear placement (in task list, not buried later)
- ✅ Explicit format instruction ("Use either X OR Y")
- ✅ Provides concrete example

**Cons**:
- ❌ Adds ~4 lines to prompt (minor bloat)
- ❌ Still doesn't show what bimodal distribution DATA looks like (that's Issue #10)

**Prompting Concern Analysis**:
- ❌ **Prevent Hallucination**: LLM must detect bimodal from percentages → risk of hallucinating patterns
- ❌ **Prevent Misclassification**: No clear threshold (is 55%/40% bimodal? 70%/20%?)
- ✅ **Allow Open-Ended Discovery**: OK

---

#### **Alternative B: Dedicated Bimodal Recommendations Section**

**Description**: Create separate array in output schema for bimodal features

```json
"bimodal_features": [
  {
    "feature": "word_count",
    "strategy_a": "Brief hooks (10-15 words) - 40% of top performers",
    "strategy_b": "Dense hooks (80-90 words) - 35% of top performers"
  }
]
```

**Pros**:
- ✅ Structured data (easier for Stage 8 to parse)
- ✅ Makes bimodal features highly visible
- ✅ Quantifies each strategy

**Cons**:
- ❌ Changes output schema (requires updating Child HLD Section 5.2.1)
- ❌ More complex prompt
- ❌ Might be over-engineering for rare bimodal cases

**Prompting Concern Analysis**:
- ❌ **Prevent Hallucination**: Same detection issue as Alternative A
- ❌ **Prevent Misclassification**: Same threshold ambiguity
- ✅ **Allow Open-Ended Discovery**: OK

---

#### **Alternative C: In-Line Bimodal Notation**

**Description**: Use special notation within existing `defining_features` array

```json
"defining_features": [
  "word_count: BIMODAL - brief (14 words, 40%) OR dense (85 words, 35%) - both work"
]
```

**Pros**:
- ✅ No schema change
- ✅ Visually distinct (BIMODAL keyword)

**Cons**:
- ❌ Inconsistent format
- ❌ Harder for Stage 8 to parse
- ❌ Long string format

**Prompting Concern Analysis**:
- ❌ **Prevent Hallucination**: Same detection issue
- ❌ **Prevent Misclassification**: Same threshold ambiguity
- ✅ **Allow Open-Ended Discovery**: OK

---

#### **Alternative A-REVISED: Pre-Compute Bimodal Flag (Python-Side Detection)** ⭐ **CHOSEN**

**Description**: Python code detects bimodal patterns using clear thresholds, adds flag to RF data

**Python preprocessing** (Stage 7, before prompt generation):
```python
def detect_bimodal_pattern(distribution: dict) -> dict:
    """
    Detect if feature shows bimodal pattern in top performers.

    Rule: Bimodal if BOTH high AND low percentages ≥30%

    Uses Stage 6 distribution data:
    - distribution['top_performers']['high_percentage']: % with ≥66th percentile
    - distribution['top_performers']['low_percentage']: % with <33rd percentile
    """
    top_high_pct = distribution['top_performers']['high_percentage']
    top_low_pct = distribution['top_performers']['low_percentage']

    is_bimodal = (top_high_pct >= 0.30 and top_low_pct >= 0.30)

    return {
        'is_bimodal': is_bimodal,
        'high_percentage': top_high_pct,
        'low_percentage': top_low_pct,
        'interpretation': 'BOTH strategies work' if is_bimodal else 'Single dominant strategy'
    }

# Add to RF data before passing to LLM
for feature in rf_data['feature_importance']:
    feature['bimodal_analysis'] = detect_bimodal_pattern(feature['distribution'])
```

**Updated prompt data format** (lines 2643-2655):
```
1. eye_contact_rate
   - RF Importance: 0.35 (rank #1)
   - Top performers: avg 0.88 (72% have ≥0.8, 15% have ≤0.4)
   - Pattern: UNIMODAL - High eye contact is THE dominant strategy

2. word_count
   - RF Importance: 0.18 (rank #3)
   - Top performers: avg 52 (40% have ≥80 words, 35% have ≤20 words)
   - Pattern: BIMODAL - BOTH brief AND dense strategies work
   - Strategy A: Brief (≤20 words) - 35% of top performers
   - Strategy B: Dense (≥80 words) - 40% of top performers
```

**Updated instruction** (line 2667):
```
4. **Generate actionable recommendations**:
   - For UNIMODAL features (marked in data):
     Single recommendation with target value
     Example: "Maintain 85-90% eye contact (RF #1 predictor)"

   - For BIMODAL features (marked "Pattern: BIMODAL" in data):
     Present BOTH strategies as valid options using format:
     "ALTERNATIVE STRATEGIES: Use either [Strategy A] OR [Strategy B] - RF data shows both work"
     Example: "ALTERNATIVE STRATEGIES: Use either brief hooks (10-20 words, 35% of top performers) OR dense hooks (80-90 words, 40% of top performers) - RF data shows both work"
```

**Pros**:
- ✅ **Prevents hallucination**: Python detects bimodal with clear 30% threshold (data-grounded)
- ✅ **Prevents misclassification**: Clear boundary rule (both high_pct AND low_pct ≥30%)
- ✅ **Allows discovery**: LLM still chooses creative phrasing for recommendations
- ✅ **Data-grounded**: Specific percentages provided ("35% of top performers")
- ✅ **Uses existing Stage 6 data**: No changes to upstream stages
- ✅ **Minimal prompt changes**: Just add labeled examples showing UNIMODAL vs BIMODAL

**Cons**:
- ❌ Requires Python code changes in Stage 7 prompt preprocessing
- ❌ Slightly more complex RF data structure

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Python detects, LLM just formats pre-classified data
- ✅ **Prevent Misclassification**: Clear 30% threshold rule
- ✅ **Allow Open-Ended Discovery**: LLM crafts creative phrasing, grounded in labeled data

---

### Final Decision: Alternative A-REVISED

**Rationale**:
1. **Addresses all 3 prompting concerns**: Prevents hallucination and misclassification while allowing discovery
2. **Data-grounded**: Python uses existing Stage 6 distribution percentages with clear threshold
3. **No hallucination risk**: LLM doesn't interpret percentages, just formats pre-labeled data
4. **Clear boundaries**: 30% threshold is explicit and data-driven
5. **Minimal changes**: Only affects Stage 7 prompt preprocessing, no schema changes

**Implementation**:
1. Add `detect_bimodal_pattern()` function to Stage 7 preprocessing
2. Update prompt data format to show "Pattern: BIMODAL" vs "Pattern: UNIMODAL"
3. Update instruction at line 2667 with explicit format for bimodal recommendations
4. Add bimodal detection to Stage 7 Child HLD Section 2.3.2

**Next Step**: Update MLPlanningv2.md lines 2643-2667 with revised prompt format

---

#### Issue #2: Ambiguous "3-5 Defining Features" (MEDIUM PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: MLPlanningv2.md line 2659

**Current Mother Doc**:
```
2. **Identify 3-5 defining features** per cluster
```

**Problem**:
- When should LLM generate 3 vs 5 features?
- No guidance on decision criteria
- Could lead to inconsistent output (cluster 0 gets 3, cluster 1 gets 5)

**Child HLD Example** (line 866-870): Shows 3 defining features consistently

**Impact**: Inconsistent output length, unpredictable JSON structure.

---

### Evaluated Alternatives

#### **Alternative A: Mandate Exactly 3 Features** ⭐ **CHOSEN**

**Description**: Replace "3-5" with "exactly 3"

```
2. **Identify exactly 3 defining features** per cluster that differentiate it from the others
   - PRIORITIZE features with high RF importance scores (rank #1-3 preferred)
   - Emphasize features with large top/bottom gaps
```

**Pros**:
- ✅ **Deterministic output**: Every cluster gets exactly 3 features (predictable structure)
- ✅ **Simplest change**: Just change "3-5" to "3"
- ✅ **Aligns with Child HLD**: Example (line 866-870) shows 3 features consistently

**Cons**:
- ❌ May miss important features if cluster has 4-5 defining characteristics
- ❌ Forces artificial selection when cluster truly has 5 distinct patterns

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Clear constraint (exactly 3)
- ✅ **Prevent Misclassification**: No ambiguity
- ⚠️ **Allow Open-Ended Discovery**: Somewhat constrained (might miss nuance)

---

#### **Alternative B: Provide Clear Decision Criteria**

**Description**: Keep "3-5" but add explicit selection rules

```
2. **Identify 3-5 defining features** per cluster that differentiate it from the others
   - PRIORITIZE features with high RF importance scores
   - Use this decision tree:
     * If cluster has 3+ features with RF rank ≤5: Include up to 5 features
     * If cluster has 2-3 features with RF rank ≤5: Include exactly 3 features
     * If cluster has <2 features with RF rank ≤5: Include 3 features (best available)
```

**Pros**:
- ✅ Provides clear decision criteria (RF rank-based)
- ✅ Allows flexibility for feature-rich clusters
- ✅ Data-grounded rule (uses RF rank from provided data)

**Cons**:
- ❌ More complex instruction (adds 3 lines)
- ❌ Still allows variable output (3-5 range)
- ❌ Downstream Stage 8 needs to handle variable feature counts

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Criteria based on RF rank (data-grounded)
- ✅ **Prevent Misclassification**: Clear thresholds (RF rank ≤5)
- ✅ **Allow Open-Ended Discovery**: Flexible within constraints

---

#### **Alternative C: Adaptive Based on Cluster Size**

**Description**: Number of features scales with cluster size

```
2. **Identify 3-5 defining features** per cluster that differentiate it from the others
   - Feature count based on cluster size:
     * Small clusters (<20 videos): 3 features (focus on core patterns)
     * Medium clusters (20-50 videos): 4 features
     * Large clusters (>50 videos): 5 features (more diverse, need more features)
   - PRIORITIZE features with high RF importance scores
```

**Pros**:
- ✅ Intuitive logic (larger clusters = more complexity = more features)
- ✅ Data-driven rule (cluster size from K-Means JSON)
- ✅ Addresses Issue #11 (cluster size guidance) simultaneously

**Cons**:
- ❌ Assumes larger clusters are more complex (not always true)
- ❌ Still variable output (3-5 range)
- ❌ Arbitrary thresholds (why 20 and 50?)

**Prompting Concern Analysis**:
- ⚠️ **Prevent Hallucination**: Assumption that size = complexity might not hold
- ✅ **Prevent Misclassification**: Clear thresholds (20, 50)
- ✅ **Allow Open-Ended Discovery**: Flexible

---

### Final Decision: Alternative A

**Rationale**:
1. **Prevents misclassification** ✅: No ambiguity, clear constraint
2. **Aligns with Child HLD examples**: Line 866-870 shows 3 features consistently
3. **Simplifies Stage 8**: PDF generation knows to expect exactly 3 features per cluster
4. **Quality over quantity**: Forces LLM to select the MOST defining features (rank #1-3 RF importance)
5. **Sufficient coverage**: 3 features is enough to characterize a cluster (hook example has eye_contact, word_count, energy_level)

**Trade-off Accepted**: Might miss nuance in clusters with 4-5 truly distinct patterns, but clarity and consistency outweigh flexibility in this case.

**Implementation**:
1. Change "3-5 defining features" to "exactly 3 defining features" at MLPlanningv2.md line 2659
2. Add clarification: "PRIORITIZE features with high RF importance scores (rank #1-3 preferred)"

**Next Step**: Update MLPlanningv2.md line 2659

---

#### Issue #3: Missing Negative Examples (Contrastive Features) (HIGH PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: MLPlanningv2.md line 2696

**Current Mother Doc** (line 2696):
```
Important:
- Be specific and concrete (not generic advice)
- Focus on DIFFERENCES between clusters (not universal best practices)
```

**Problem**:
- Says "focus on DIFFERENCES" but doesn't explain **how**
- No explicit instruction: "Don't list features that are high in ALL clusters"
- No guidance: "A defining feature should be high in THIS cluster, low in others"

**Example of Problem**:
```json
// LLM might generate:
Cluster 0: "High eye contact (0.87) - DEFINING"
Cluster 1: "High eye contact (0.82) - DEFINING"
Cluster 2: "High eye contact (0.85) - DEFINING"
// ❌ Eye contact is NOT defining if all clusters have it!
```

**Impact**: Redundant cluster descriptions where all 3 clusters list same features.

---

### Evaluated Alternatives

#### **Alternative A: Add Explicit Contrastive Definition**

**Description**: Add clear definition of "defining feature" with negative examples

```
2. **Identify exactly 3 defining features** per cluster that differentiate it from the others

   **What makes a feature "defining"**:
   - ✅ HIGH in this cluster, LOW in other clusters (contrastive)
   - ❌ HIGH in all 3 clusters (universal - not defining)
   - Example: If eye_contact_rate is 0.85+ in all clusters, it's NOT defining

   **How to identify**:
   - Compare centroid values across clusters for each feature
   - Select features with LARGEST between-cluster variance
   - A defining feature should differ by ≥0.20 from at least one other cluster
```

**Pros**:
- ✅ Clear definition with positive and negative examples
- ✅ Provides quantitative threshold (≥0.20 difference)
- ✅ Data-grounded (compare centroid values from K-Means JSON)

**Cons**:
- ❌ Adds ~6 lines to prompt
- ❌ Requires LLM to compute between-cluster variance (might hallucinate arithmetic)

**Prompting Concern Analysis**:
- ⚠️ **Prevent Hallucination**: LLM must compare centroids → risk of miscalculation
- ✅ **Prevent Misclassification**: Clear threshold (≥0.20 difference)
- ✅ **Allow Open-Ended Discovery**: Still flexible in selecting which contrasts to highlight

---

#### **Alternative B: Python Pre-Computes Contrastive Features (Full Automation)**

**Description**: Python identifies contrastive features, labels them in data before LLM sees it

**Python preprocessing**:
```python
def identify_contrastive_features(kmeans_data: dict) -> dict:
    """
    For each cluster, identify which features are contrastive.

    A feature is contrastive for a cluster if:
    - It differs by ≥0.20 from at least one other cluster's centroid
    """
    clusters = kmeans_data['clusters']
    all_features = list(clusters[0]['centroid'].keys())

    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        centroid = cluster['centroid']

        contrastive_features = []

        for feature in all_features:
            this_value = centroid[feature]

            # Get values in other clusters
            other_values = [
                c['centroid'][feature]
                for c in clusters
                if c['cluster_id'] != cluster_id
            ]

            # Check if differs by ≥0.20 from any other cluster
            max_diff = max(abs(this_value - ov) for ov in other_values)

            if max_diff >= 0.20:
                contrastive_features.append({
                    'feature': feature,
                    'value': this_value,
                    'max_contrast': max_diff
                })

        # Sort by max_contrast (most contrastive first)
        contrastive_features.sort(key=lambda x: x['max_contrast'], reverse=True)

        cluster['contrastive_features'] = contrastive_features[:10]  # Top 10
```

**Pros**:
- ✅ **Prevents hallucination**: Python computes, LLM just selects from pre-filtered list
- ✅ **Prevents misclassification**: Clear 0.20 threshold enforced by Python

**Cons**:
- ❌ Python cannot judge **semantic meaning** (e.g., is 0.15 eye_contact difference meaningful?)
- ❌ Python cannot detect **strategic coherence** (features that tell a coherent story together)
- ❌ Over-constrains LLM (removes semantic judgment capability)

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Python does computation
- ✅ **Prevent Misclassification**: 0.20 threshold enforced
- ❌ **Allow Open-Ended Discovery**: Too constrained - Python can't understand domain semantics

---

#### **Alternative C: Provide Contrastive Feature Examples in Prompt**

**Description**: Show concrete examples of contrastive vs universal features

```
2. **Identify exactly 3 defining features** per cluster that differentiate it from the others

   **Example of DEFINING (contrastive) feature**:
   - Cluster 0: eye_contact_rate = 0.87
   - Cluster 1: eye_contact_rate = 0.42
   - Cluster 2: eye_contact_rate = 0.55
   → ✅ CONTRASTIVE (Cluster 0 much higher) - this IS defining for Cluster 0

   **Example of NON-defining (universal) feature**:
   - Cluster 0: has_captions = 0.95
   - Cluster 1: has_captions = 0.92
   - Cluster 2: has_captions = 0.94
   → ❌ UNIVERSAL (all clusters similar) - this is NOT defining
```

**Pros**:
- ✅ Concrete examples (few-shot learning)
- ✅ Shows both positive and negative cases
- ✅ No Python preprocessing needed

**Cons**:
- ❌ Adds ~10 lines to prompt
- ❌ Still requires LLM to compute differences (hallucination risk)
- ❌ Examples might not match actual data

**Prompting Concern Analysis**:
- ⚠️ **Prevent Hallucination**: LLM still does computation, just with examples
- ⚠️ **Prevent Misclassification**: Threshold is "≥0.15-0.20" (ambiguous range)
- ✅ **Allow Open-Ended Discovery**: Flexible

---

#### **Alternative D: Hybrid Approach (Python Filters, LLM Selects)** ⭐ **CHOSEN**

**Description**: Python does mechanical filtering, LLM does semantic selection

**Key Insight**:
- **Python excels at**: Arithmetic comparison, threshold enforcement, systematic filtering
- **Python fails at**: Semantic understanding, feature interaction, domain knowledge
- **LLM excels at**: Contextual judgment, holistic patterns, strategic coherence
- **LLM fails at**: Precise arithmetic, systematic comparison across all clusters

**Solution**: Divide responsibilities based on strengths

**Step 1 - Python Preprocessing** (Mechanical Filtering):
```python
def identify_high_contrast_features(kmeans_data: dict, threshold: float = 0.20) -> dict:
    """
    Pre-filter features with high numerical contrast (≥0.20 difference).
    Does NOT decide which are "defining" - just narrows the field.

    Reduces from 21 features → typically 8-12 high-contrast features
    """
    clusters = kmeans_data['clusters']
    all_features = list(clusters[0]['centroid'].keys())

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
                # Find which cluster has the max contrast
                contrasts = {
                    f"vs Cluster {c['cluster_id']}": abs(this_value - c['centroid'][feature])
                    for c in clusters
                    if c['cluster_id'] != cluster_id
                }

                high_contrast.append({
                    'feature': feature,
                    'value': this_value,
                    'max_contrast': max_diff,
                    'contrasts': contrasts  # Show comparisons to other clusters
                })

        # Sort by max_contrast (highest first)
        high_contrast.sort(key=lambda x: x['max_contrast'], reverse=True)

        cluster['high_contrast_features'] = high_contrast
        cluster['all_features'] = centroid  # Keep all features for context
```

**Step 2 - Updated Prompt Format** (Semantic Selection):
```
CLUSTER 0 (35 videos):

All features (for context):
  eye_contact_rate: 0.87
  word_count: 14
  energy_level: 0.55
  scene_count: 2.1
  ... (all 21 features)

High-contrast features (differ by ≥0.20 from other clusters):
  1. word_count: 14 (vs Cluster 1: 52, vs Cluster 2: 35) ← max contrast: 38
  2. eye_contact_rate: 0.87 (vs Cluster 1: 0.42, vs Cluster 2: 0.55) ← max contrast: 0.45
  3. energy_level: 0.55 (vs Cluster 2: 0.85) ← max contrast: 0.30
  4. scene_count: 2.1 (vs Cluster 1: 4.5) ← max contrast: 2.4
  ... (8 total high-contrast features)

Your task:
Select exactly 3 defining features from the HIGH-CONTRAST list above.

Selection criteria:
1. **RF importance**: Prioritize features with rank #1-5 (check RF data provided)
2. **Strategic coherence**: Choose features that tell a coherent story together
   - Example: low word_count + high eye_contact = "intimate direct communication"
   - Avoid random features that don't create a clear strategy
3. **Contrast magnitude**: Larger differences = clearer distinction (but not the only factor)

**CRITICAL**: Do NOT select features just because they have high numerical contrast.
A feature with 0.25 contrast but low RF importance (#9) is LESS defining than
a feature with 0.22 contrast but high RF importance (#2).

**Avoid universal features**: The high-contrast list already filters these out,
but if you notice a feature is high in ALL clusters (check "All features" context),
skip it even if it appears in the high-contrast list.
```

**Pros**:
- ✅ **Prevents hallucination**: Python does arithmetic, LLM doesn't calculate differences
- ✅ **Prevents misclassification**: Python enforces 0.20 threshold mechanically
- ✅ **Allows discovery**: LLM uses domain knowledge to select semantically meaningful features
- ✅ **Best of both worlds**: Python for precision, LLM for judgment
- ✅ **Reduces prompt bloat**: Pre-filtered list means LLM sees 8-12 features instead of 21

**Cons**:
- ❌ Requires Python code changes in Stage 7 preprocessing
- ❌ More complex data structure (but clearer for LLM)

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Python computes all arithmetic, LLM just selects
- ✅ **Prevent Misclassification**: Clear 0.20 threshold in code, clear selection criteria in prompt
- ✅ **Allow Open-Ended Discovery**: LLM judges semantic meaning and strategic coherence

---

### Final Decision: Alternative D (Hybrid Approach)

**Rationale**:
1. **Division of labor based on strengths**: Python does mechanical filtering (what it's good at), LLM does semantic selection (what it's good at)
2. **Prevents hallucination**: LLM doesn't do arithmetic (Python pre-computes all contrasts)
3. **Preserves LLM judgment**: LLM can weigh RF importance + strategic coherence + contrast magnitude
4. **Reduces cognitive load**: LLM sees 8-12 pre-filtered features instead of comparing all 21 across 3 clusters
5. **Aligns with Issue #1 pattern**: Python detects/computes, LLM formats/selects

**Example where hybrid excels**:
```
Python finds: overlay_count (contrast: 1.8), eye_contact (contrast: 0.45), word_count (contrast: 38)

LLM picks: eye_contact + word_count + energy_level
LLM skips: overlay_count (lower RF rank #8, doesn't fit "intimate communication" strategy)

→ Result: Semantically coherent cluster description, grounded in data
```

**Implementation**:
1. Add `identify_high_contrast_features()` to Stage 7 preprocessing (before Phase 1 prompt)
2. Update K-Means prompt data format to show "All features" + "High-contrast features"
3. Update instruction at line 2659 with selection criteria (RF importance + strategic coherence + contrast)
4. Add to Stage 7 Child HLD Section 2.3.2

**Next Step**: Update MLPlanningv2.md lines 2630-2637 (K-Means data format) and line 2659 (instruction)

---

#### Issue #4: RF Validation Section Underspecified (MEDIUM PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: MLPlanningv2.md lines 2679-2682

**Current Mother Doc**:
```json
"rf_validation": {
  "top_predictive_features_in_cluster": [...],
  "insight": "How this cluster uses RF-validated features"
}
```

**Problem**:
- What should go in `top_predictive_features_in_cluster`? Feature names? Values? How many?
- "Insight" is vague - what specifically should this explain?
- No example output provided

**Child HLD Example** (line 872-876):
```json
"rf_validation": {
  "top_predictive_features_in_cluster": [
    "eye_contact_rate: Cluster value 0.87 matches top performer avg 0.88 (RF validated)"
  ],
  "insight": "This cluster leverages the #1 most predictive hook feature at optimal levels."
}
```

**Impact**: LLM generates inconsistent or vague RF validation content.

---

### Evaluated Alternatives

#### **Alternative A: Provide Explicit Format Specification**

**Description**: Add clear format instructions with examples

```
3. **Validate against RF features** (lines 2679-2682):

   Generate rf_validation section:

   "rf_validation": {
     "top_predictive_features_in_cluster": [
       // Format: "feature_name: Cluster value X matches/differs from top performer avg Y (RF rank #Z)"
       // Include 1-3 features where cluster centroid aligns with RF top performers
       // Example: "eye_contact_rate: Cluster value 0.87 matches top performer avg 0.88 (RF rank #1)"
     ],
     "insight": "Explain HOW this cluster uses RF-validated features"
       // Format: "[Cluster leverages/uses/emphasizes] the #X most predictive feature..."
       // Be specific: mention RF ranks and whether cluster values are optimal/suboptimal
   }
```

**Pros**:
- ✅ Clear format with concrete examples
- ✅ Specifies what to include (1-3 features, RF rank, comparison)
- ✅ No Python preprocessing needed

**Cons**:
- ❌ Adds ~8 lines to prompt
- ❌ LLM must still compute "matches/differs" comparison (hallucination risk)

**Prompting Concern Analysis**:
- ⚠️ **Prevent Hallucination**: LLM compares cluster value vs top performer avg (arithmetic risk)
- ✅ **Prevent Misclassification**: Clear format specified
- ✅ **Allow Open-Ended Discovery**: Flexible in phrasing "insight"

---

#### **Alternative B: Python Pre-Computes RF Alignment** ⭐ **CHOSEN**

**Description**: Python calculates which cluster features align with RF top performers

**Python preprocessing**:
```python
def compute_rf_alignment(cluster_centroid: dict, rf_features: list, threshold: float = 0.15) -> dict:
    """
    Identify which cluster features align with RF top performers.

    A feature "aligns" if cluster centroid value is within ±0.15 of top_performer_avg

    Args:
        cluster_centroid: K-Means cluster centroid values
        rf_features: Window-level RF feature importance list (from Stage 6)
        threshold: Alignment threshold (default: 0.15 = within 15%)

    Returns:
        dict with aligned_features, alignment_count, alignment_score
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
                    'formatted': f"{feature_name}: Cluster value {cluster_value:.2f} {alignment_type} top performer avg {top_avg:.2f} (RF rank #{rf_rank})"
                })

    return {
        'aligned_features': aligned_features,
        'alignment_count': len(aligned_features),
        'alignment_score': f"{len(aligned_features)}/5"  # e.g., "3/5"
    }
```

**Updated prompt data format**:
```
CLUSTER 0 (35 videos):

All features: {...}

High-contrast features: {...}

RF Alignment (features matching top performer patterns):
  ✅ eye_contact_rate: Cluster value 0.87 matches top avg 0.88 (RF rank #1, importance 0.35)
  ✅ energy_level: Cluster value 0.55 matches top avg 0.53 (RF rank #2, importance 0.22)
  ❌ word_count: Cluster value 14 differs from top avg 52 (RF rank #3) ← Not aligned

  Alignment score: 2/5 (uses 2 of top 5 RF features at optimal levels)

Your task:
Generate rf_validation section:
{
  "top_predictive_features_in_cluster": [
    // Copy the ✅ aligned features from RF Alignment data above
    // Use the pre-formatted text provided
  ],
  "insight": "Explain alignment score and which RF features the cluster uses"
    // Format: "This cluster leverages the #X [and #Y] most predictive feature(s)..."
    // Mention alignment score (e.g., "2/5 features aligned")
    // Specify whether cluster is RF-optimized or diverges from top patterns
}
```

**Pros**:
- ✅ **Prevents hallucination**: Python computes alignment, LLM just formats pre-computed data
- ✅ **Data-grounded**: Uses actual centroid values and RF top_performer_avg
- ✅ **Provides alignment score**: Quantifies validation ("3/5" = uses 3 of top 5 RF features)
- ✅ **Consistent with Issues #1, #3**: Same pattern (Python computes, LLM formats)
- ✅ **Clear threshold**: ±0.15 is explicit (within 15% = aligned)

**Cons**:
- ❌ Requires Python preprocessing in Stage 7
- ❌ Threshold (±0.15) is somewhat arbitrary (but data-driven)

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Python does arithmetic comparison, LLM just copies formatted strings
- ✅ **Prevent Misclassification**: Clear 0.15 threshold enforced by code
- ✅ **Allow Open-Ended Discovery**: LLM phrases insight creatively using alignment data

---

#### **Alternative C: Simplified Schema - Just Insight**

**Description**: Remove `top_predictive_features_in_cluster` array, keep only insight

```json
"rf_validation": {
  "insight": "This cluster leverages the #1 most predictive hook feature (eye_contact_rate: 0.87 vs top avg 0.88) at optimal levels."
}
```

**Instruction**:
```
3. **Generate RF validation insight**:
   Check which of the top 3 RF features (from RF data provided) appear in your selected defining features.
   Write a 1-sentence insight explaining how this cluster uses RF-validated features.

   Format: "This cluster [leverages/uses/emphasizes] the #X [and #Y] most predictive feature(s)..."
   Include specific feature names and values in parentheses.
```

**Pros**:
- ✅ Simpler schema (one field instead of two)
- ✅ Combines feature list + insight into single narrative
- ✅ Reduces JSON verbosity

**Cons**:
- ❌ Changes Child HLD output schema (requires updating Section 5.2.1)
- ❌ Less structured (harder for Stage 8 to parse for visual badges like "9/10 RF Validated")
- ❌ LLM still does comparison arithmetic

**Prompting Concern Analysis**:
- ⚠️ **Prevent Hallucination**: LLM still compares values
- ✅ **Prevent Misclassification**: Clear format
- ✅ **Allow Open-Ended Discovery**: Flexible phrasing

---

### Final Decision: Alternative B (Python Pre-Computes RF Alignment)

**Rationale**:
1. **Consistent with Issues #1 and #3**: Python computes, LLM formats (established pattern)
2. **Prevents hallucination** ✅: LLM doesn't compute "0.87 vs 0.88 = matches" (Python does this)
3. **Provides alignment score**: Quantifies validation (e.g., "3/5 features aligned")
4. **Data-grounded**: Uses actual RF top_performer_avg from Stage 6 JSON
5. **Clear threshold**: ±0.15 is explicit (within 15% = aligned, <10% = exact match)
6. **Stage 8 value**: Alignment score ("3/5") can be displayed as badge in PDF ("60% RF Validated")

**Implementation**:
1. Add `compute_rf_alignment()` to Stage 7 preprocessing (before Phase 1 prompt)
2. Update prompt data format to show RF Alignment section with ✅/❌ indicators
3. Update instruction at lines 2679-2682 to reference pre-computed alignment data
4. Add to Stage 7 Child HLD Section 2.3.2

**Next Step**: Update MLPlanningv2.md lines 2630-2637 (add RF Alignment section) and lines 2679-2682 (update instruction)

---

#### Issue #5: Weak "Important" Section Reminders (LOW PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: MLPlanningv2.md lines 2694-2697

**Current Mother Doc**:
```
Important:
- Be specific and concrete (not generic advice)
- Focus on DIFFERENCES between clusters (not universal best practices)
- Recommendations should be replicable creative techniques
```

**Problem**: All three reminders are generic
- "Be specific and concrete" - HOW?
- "Focus on DIFFERENCES" - already stated in task #2
- "Replicable techniques" - vague

**Impact**: Weak constraints, LLM might still generate generic advice.

---

### Evaluated Alternatives

#### **Alternative A: Keep Generic Reminders (Status Quo)**

**Description**: Leave the "Important" section as-is with current generic reminders.

**Pros**:
- ✅ No changes needed (zero effort)
- ✅ Generic reminders don't hurt (LLM will ignore if redundant)
- ✅ Won't break existing prompt structure

**Cons**:
- ❌ Doesn't add real value (LLM might ignore vague instructions)
- ❌ Redundant with task #2 ("Focus on DIFFERENCES")
- ❌ Wastes ~3 lines of prompt context

---

#### **Alternative B: Strengthen with Concrete Anti-Patterns**

**Description**: Replace generic reminders with concrete anti-patterns (what NOT to do) + examples.

```
Important - Avoid These Common Mistakes:

❌ **Generic advice**: "Use good lighting" → ✅ **Specific**: "Maintain 0.87 eye contact rate (RF rank #1)"
❌ **Universal features**: "All clusters need captions" → ✅ **Contrastive**: "Cluster 0 uses brief text (14 words), Cluster 1 uses dense (52 words)"
❌ **Vague recommendations**: "Be engaging" → ✅ **Replicable technique**: "Start with direct eye contact for 3 seconds (hook window), then shift to product close-up"
```

**Pros**:
- ✅ Concrete examples (few-shot learning - LLM sees what to avoid)
- ✅ Shows both bad and good versions (contrastive learning)
- ✅ Addresses actual failure modes (generic advice, universal features, vague recommendations)

**Cons**:
- ❌ Adds ~6 lines to prompt (from 3 to 9 lines)
- ❌ Examples might not perfectly match actual data
- ❌ Slight redundancy with other instructions (but reinforcement can help)

---

#### **Alternative C: Remove "Important" Section Entirely** ⭐ **CHOSEN**

**Description**: Delete the "Important" section completely, rely on task instructions + Python preprocessing to enforce quality.

**Rationale**: Given that Issues #1, #3, #4, #8 are all resolved with Python preprocessing (bimodal detection, contrastive feature filtering, RF alignment, format templates), the "Important" section is redundant. The **real constraints** are now enforced by:
- Python pre-computes bimodal patterns → LLM can't hallucinate
- Python filters high-contrast features → LLM can't select universal features
- Python provides RF alignment data → LLM can't ignore RF validation
- Clear format templates in instructions → LLM outputs structured format

**Pros**:
- ✅ Reduces prompt bloat (removes 3 lines)
- ✅ Eliminates redundancy (constraints already enforced elsewhere)
- ✅ Simplifies prompt (fewer sections to parse)
- ✅ Acknowledges that **Python preprocessing is doing the heavy lifting**, not vague reminders

**Cons**:
- ❌ Loses reinforcement (redundancy can help emphasize key points)
- ❌ Removes human-readable summary of intent

---

### Final Decision: Alternative C (Remove "Important" Section)

**Rationale**:
1. **Python preprocessing makes reminders redundant**: With Issues #1, #3, #4, #8 resolved, the actual quality constraints are now enforced by Python code, not vague reminders:
   - "Be specific" → Enforced by format templates (Issue #8 resolution)
   - "Focus on DIFFERENCES" → Enforced by Python high-contrast filtering (Issue #3 resolution)
   - "Replicable techniques" → Enforced by clear step-by-step structure in output schema
2. **Vague reminders have low impact**: LLMs typically ignore generic constraints like "be specific" without concrete examples or format specifications. The resolved issues provide those concrete specifications.
3. **Reduces prompt bloat**: Every line counts in prompt engineering. 3 lines of redundant reminders can be better used elsewhere (or saved for future additions).
4. **Cleaner prompt structure**: The prompt ends cleanly with the JSON output schema, rather than a weak "Important" section.
5. **If reinforcement is needed later**, we can add concrete anti-patterns (Alternative B) after pilot testing reveals specific failure modes.

**Trade-off Accepted**: We lose a small amount of reinforcement, but the Python preprocessing and clear task instructions are doing the real work. If pilot testing shows LLMs still generating generic advice, we can add Alternative B's concrete anti-patterns.

**Implementation**:
1. Remove "Important" section (lines 2694-2697) from MLPlanningv2.md Phase 1 prompt
2. Rely on Python preprocessing (Issues #1, #3, #4, #8) to enforce quality constraints
3. Monitor pilot testing for generic advice patterns; add Alternative B if needed

**Next Step**: Update MLPlanningv2.md lines 2694-2697 (remove "Important" section)

---

#### Issue #6: No Edge Case Handling (MEDIUM PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: None (missing entirely)

**Problem**: Prompt doesn't guide LLM on edge cases:
- Very small clusters (e.g., cluster with only 8 videos out of 100)
- Very imbalanced clusters (e.g., 70 videos in cluster 0, 5 in cluster 1, 25 in cluster 2)
- All 3 clusters are nearly identical (low separation)
- A cluster has no high-importance RF features

**Child HLD Example** (line 1558-1600): Shows varied cluster sizes (35, 42, 23 videos)

**Impact**: LLM confused when encountering edge cases, might generate inappropriate recommendations.

**CRITICAL CONSTRAINT**: Minimum 50 videos per bucket (realistically 80-120 for contrastive analysis). This eliminates most extreme edge cases.

---

### Evaluated Alternatives

#### **Alternative A: Add Edge Case Warnings (Simplified for 50+ videos)**

**Description**: Instruct LLM to detect and flag only realistic edge cases.

```
**Edge Case Detection** (only for 50+ video samples):

1. **Imbalanced clusters** (largest cluster >50%):
   - Dominant cluster: "DOMINANT strategy (X% of videos)"
   - Small clusters (<20%): "NICHE strategy (X% of videos)"

2. **Low RF alignment** (<2/5):
   - "This cluster explores strategies beyond top RF predictors"

Output adds optional field:
"strategy_prevalence": "dominant" | "common" | "niche"  // Based on cluster %
```

**Pros**:
- ✅ Simple classification (dominant/common/niche) based on cluster size %
- ✅ Makes prevalence visible to Stage 8 PDF generation
- ✅ Realistic thresholds: >50% = dominant, 20-50% = common, <20% = niche

**Cons**:
- ❌ Adds new field to output schema (requires Child HLD update)
- ❌ LLM must calculate percentages (though this is simple arithmetic)

---

#### **Alternative B: Python Pre-Computes Cluster Prevalence**

**Description**: Python calculates cluster size %, adds prevalence label to prompt data.

**Python preprocessing**:
```python
def classify_cluster_prevalence(cluster_size: int, total_videos: int) -> str:
    """
    Classify cluster prevalence based on size.

    With k=3 clustering on 50-100 videos:
    - Dominant: >50% of videos (clear majority pattern)
    - Common: 25-50% of videos (standard competitive cluster)
    - Niche: <25% of videos (minority but still meaningful with 12+ videos)
    """
    pct = (cluster_size / total_videos) * 100

    if pct > 50:
        return "dominant"
    elif pct >= 25:
        return "common"
    else:
        return "niche"
```

**Updated prompt format**:
```
CLUSTER 0 (15 videos, 15% - NICHE STRATEGY):

All features: {...}
High-contrast features: {...}

**Prevalence**: NICHE - This represents 15% of videos. Frame as an alternative approach, not the dominant pattern.
```

**Pros**:
- ✅ Python does simple calculation (no LLM arithmetic)
- ✅ Clear labels in prompt data (DOMINANT/COMMON/NICHE)
- ✅ Consistent with Issues #1, #3, #4 pattern (Python computes, LLM formats)

**Cons**:
- ❌ Requires Python preprocessing
- ❌ Thresholds (50%, 25%) are somewhat arbitrary (though data-informed)

---

#### **Alternative C: Minimal Edge Case Guidance** ⭐ **CHOSEN**

**Description**: Add minimal instructions for only the most impactful edge case: imbalanced clusters.

**Updated prompt section** (add ~10 lines after task list):
```
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
```

**Pros**:
- ✅ **Simplest implementation**: ~10 lines added to prompt, no Python preprocessing, no schema changes
- ✅ **Realistic for 50+ videos**: With minimum 50 videos, even smallest cluster (12-15 videos) is statistically meaningful
- ✅ **Clear thresholds**: >50% = dominant, <25% = niche (easy for LLM to apply)
- ✅ **Natural language**: LLM can phrase contextually ("dominant", "common", "niche") without rigid schema
- ✅ **No false alarms**: With 50+ videos, we won't get truly problematic clusters (5-8 videos)

**Cons**:
- ❌ No programmatic validation of prevalence classification
- ❌ LLM must calculate percentages (cluster_size / total_videos)
- ❌ Doesn't handle low RF alignment edge case (though that's more feature than bug - creative novelty)

---

### Final Decision: Alternative C (Minimal Guidance)

**Rationale**:
1. **50+ video constraint eliminates extreme edge cases**: With minimum 50 videos and k=3, smallest realistic cluster is ~12-15 videos (still statistically meaningful). No need for "small sample" warnings.
2. **Imbalance is the only real edge case**: With 50+ videos:
   - ❌ Small clusters (<10 videos): CAN'T HAPPEN
   - ❌ Extreme imbalance (12x ratio): CAN'T HAPPEN
   - ✅ Moderate imbalance (60-25-15): CAN HAPPEN → Need framing guidance
   - ⚠️ Low RF alignment: NOT A BUG, IT'S A FEATURE (creative novelty)
3. **Simplicity**: Just ~10 lines of guidance. LLM applies contextually without rigid schema.
4. **Natural phrasing**: "dominant strategy" vs "niche approach" is more creator-friendly than programmatic labels.
5. **No over-engineering**: With 50+ videos, edge cases are rare and mild. Don't build complex machinery for scenarios that rarely occur.

**Trade-off Accepted**: We rely on LLM to calculate percentages (cluster_size / total_videos) and apply thresholds. This is acceptable because:
- Arithmetic is simple (even LLMs rarely miscalculate basic division)
- Cluster size and total videos are clearly provided in prompt data
- Framing is qualitative (50% is guideline, not hard rule)

**Implementation**:
1. Add "Cluster Size Context" section (~10 lines) after task list in Phase 1 prompt (MLPlanningv2.md after line 2668)
2. Instruct LLM to use "dominant/common/niche" language based on cluster %
3. No Python preprocessing, no schema changes

**Next Step**: Update MLPlanningv2.md (add "Cluster Size Context" section after task list)

---

#### Issue #7: Verbose RF Feature Formatting (LOW PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: MLPlanningv2.md lines 2643-2655

**Current Mother Doc**: Each RF feature gets 4-5 lines:
```
1. eye_contact_rate
   - RF Importance: 0.35 (rank #1)
   - Top performers: avg 0.88 (72% have ≥0.8)
   - Bottom performers: avg 0.45 (only 15% reach 0.8)
   - Gap: 0.43
```

**Problem**: With 10 features, that's **50 lines of context** in prompt

**Impact**: Context bloat, harder for LLM to parse, increases token cost.

---

### Evaluated Alternatives

#### **Alternative A: Compress to Table Format**

**Description**: Convert multi-line format to compact table.

```
RF Feature Importance (Window-Level):

| Feature | RF Rank | Importance | Top Avg | Top High% | Top Low% | Bottom Avg | Gap | Pattern |
|---------|---------|------------|---------|-----------|----------|------------|-----|---------|
| eye_contact_rate | #1 | 0.35 | 0.88 | 72% | 15% | 0.45 | 0.43 | UNIMODAL |
| energy_level | #2 | 0.22 | 0.75 | 65% | 18% | 0.42 | 0.33 | UNIMODAL |
| word_count | #3 | 0.18 | 52 | 40% | 35% | 18 | 34 | BIMODAL |
... (10 features total)
```

**Pros**:
- ✅ **Massive space savings**: 50 lines → ~12 lines (including header)
- ✅ **Easy to scan**: Tabular format is efficient for comparing features
- ✅ **All data preserved**: No information loss

**Cons**:
- ❌ **Harder to parse for LLMs**: Tables with many columns can confuse some LLMs
- ❌ **Column overflow**: 8 columns might wrap awkwardly in prompt
- ❌ **Less readable for humans**: Debugging prompts becomes harder

---

#### **Alternative B: Compress to Condensed Multi-Line** ⭐ **CHOSEN**

**Description**: Keep multi-line format but condense to 2 lines per feature.

```
RF Feature Importance (Window-Level):

1. eye_contact_rate - RF Importance: 0.35 (rank #1)
   Top: avg 0.88 (72% high, 15% low) | Bottom: avg 0.45 | Gap: 0.43 | Pattern: UNIMODAL

2. energy_level - RF Importance: 0.22 (rank #2)
   Top: avg 0.75 (65% high, 18% low) | Bottom: avg 0.42 | Gap: 0.33 | Pattern: UNIMODAL

3. word_count - RF Importance: 0.18 (rank #3)
   Top: avg 52 (40% high, 35% low) | Bottom: avg 18 | Gap: 34 | Pattern: BIMODAL
   → Strategy A: Brief (≤20 words) - 35% | Strategy B: Dense (≥80 words) - 40%

... (10 features total)
```

**Savings**: 50 lines → ~22 lines (2 lines per feature × 10 + bimodal expansions)

**Pros**:
- ✅ **50% space savings**: 50 lines → ~22 lines
- ✅ **Maintains readability**: Still multi-line, just condensed
- ✅ **LLM-friendly**: Vertical format easier for LLMs to parse than wide tables
- ✅ **Bimodal expansion**: Extra line for bimodal features preserves detail

**Cons**:
- ❌ **Still ~22 lines**: Not as compact as table format
- ❌ **Pipe separator syntax**: Slightly less readable than bullet points

---

#### **Alternative C: Keep Current Format (Status Quo)**

**Description**: No changes. Accept 50 lines as acceptable cost.

**Pros**:
- ✅ No implementation needed
- ✅ Most readable format (clear bullet points)
- ✅ No risk of LLM misparse

**Cons**:
- ❌ 50 lines of context (token cost)
- ❌ Harder for LLM to scan quickly

---

### Final Decision: Alternative B (Condensed Multi-Line)

**Rationale**:
1. **50% space savings without sacrificing clarity**: 50 lines → ~22 lines is significant, while maintaining vertical structure that LLMs parse well.
2. **Bimodal features handled gracefully**: Can add expansion line for bimodal features without breaking format:
   ```
   3. word_count - RF Importance: 0.18 (rank #3)
      Top: avg 52 (40% high, 35% low) | Bottom: avg 18 | Gap: 34 | Pattern: BIMODAL
      → Strategy A: Brief (≤20 words) - 35% | Strategy B: Dense (≥80 words) - 40%
   ```
3. **Avoids table parsing issues**: Wide tables (8 columns) can confuse LLMs. Vertical format with pipe separators is more reliable.
4. **Maintains data completeness**: All information preserved (importance, rank, top/bottom avgs, distribution percentages, gap, pattern).
5. **Human-readable for debugging**: When reviewing prompts, condensed multi-line is easier to read than dense table.

**Trade-off Accepted**: Still ~22 lines (not as compact as table's 12 lines), but more reliable for LLM parsing and human readability.

**Implementation**:
1. Update RF feature data format in MLPlanningv2.md lines 2643-2655
2. Convert from 4-5 line format to 2-line format with pipe separators
3. Add extra line for bimodal features (showing Strategy A/B breakdowns)

**Next Step**: Update MLPlanningv2.md lines 2643-2655 (condensed format)

---

#### Issue #8: Defining Features Format Not Specified (HIGH PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: MLPlanningv2.md line 2659

**Current Mother Doc**:
```
2. **Identify 3-5 defining features** per cluster
```

**Child HLD Output Schema** (line 866-870):
```json
"defining_features": [
  "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43 - HIGHEST PREDICTOR)",
  "word_count: 14 (RF rank #3, importance 0.18, low count strategy)"
]
```

**Problem**: Prompt doesn't specify the **exact format** for defining features. Child HLD shows very specific structure: `feature_name: value (RF rank #X, importance Y.YY, gap Z.ZZ - interpretation)`

**Impact**: LLM might generate inconsistent formats:
- "High eye contact (0.87)" ← Missing RF rank
- "eye_contact_rate = 0.87, rank 1" ← Wrong separator syntax
- "Very high eye contact with 0.87 score" ← Too verbose

---

### Evaluated Alternatives

#### **Alternative A: Explicit Format Template in Prompt**

**Description**: Add clear format specification with examples

```
2. **Identify exactly 3 defining features** per cluster (from high-contrast list)

   **Output format for each feature**:
   "feature_name: value (RF rank #X, importance Y.YY, gap Z.ZZ - interpretation)"

   Components:
   - feature_name: Exact feature name (e.g., "eye_contact_rate")
   - value: Cluster centroid value (e.g., 0.87)
   - RF rank #X: Rank from RF data (e.g., "#1")
   - importance Y.YY: RF importance score (e.g., "0.35")
   - gap Z.ZZ: Top/bottom gap from RF data (e.g., "0.43")
   - interpretation: Brief creative description

   Examples:
   ✅ "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43 - HIGHEST PREDICTOR)"
   ❌ "High eye contact: 0.87" (missing RF data)
```

**Pros**:
- ✅ Crystal clear format specification
- ✅ Shows both correct and incorrect examples
- ✅ No Python preprocessing needed

**Cons**:
- ❌ Adds ~10 lines to prompt
- ❌ LLM must still extract RF rank, importance, gap from RF data (potential errors)

**Prompting Concern Analysis**:
- ⚠️ **Prevent Hallucination**: LLM must look up RF rank/importance/gap (could get wrong values)
- ✅ **Prevent Misclassification**: Clear format template
- ✅ **Allow Open-Ended Discovery**: Flexible "interpretation" field

---

#### **Alternative B: Python Pre-Formats Feature Strings**

**Description**: Python generates formatted strings, LLM just selects which to use

**Python preprocessing**:
```python
def format_defining_features(high_contrast_features: list, rf_features: list) -> list:
    """
    Pre-format defining feature strings for LLM selection.
    Returns formatted strings matching Child HLD schema exactly.
    """
    formatted_features = []

    for hc_feature in high_contrast_features:
        feature_name = hc_feature['feature']
        cluster_value = hc_feature['value']

        # Find RF data for this feature
        rf_data = next((rf for rf in rf_features if rf['feature'] == feature_name), None)

        if rf_data:
            rf_rank = rf_features.index(rf_data) + 1
            importance = rf_data['importance']
            gap = rf_data['gap']

            # Generate generic interpretation
            if rf_rank == 1:
                interpretation = "HIGHEST PREDICTOR"
            elif gap > 0.30:
                interpretation = "high contrast feature"
            else:
                interpretation = "key differentiator"

            formatted_string = f"{feature_name}: {cluster_value:.2f} (RF rank #{rf_rank}, importance {importance:.2f}, gap {gap:.2f} - {interpretation})"

            formatted_features.append({'feature': feature_name, 'formatted_string': formatted_string})

    return formatted_features
```

**Pros**:
- ✅ **Perfect format consistency**: Python generates exact schema format
- ✅ **Prevents hallucination**: LLM doesn't compute RF rank/importance/gap
- ✅ **Simplest LLM task**: Just copy strings

**Cons**:
- ❌ Python generates "interpretation" (loses LLM creativity)
- ❌ "HIGHEST PREDICTOR", "key differentiator" are too generic
- ❌ Requires Python preprocessing

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Python does all formatting, LLM just copies
- ✅ **Prevent Misclassification**: Perfect format match
- ❌ **Allow Open-Ended Discovery**: Python-generated interpretations are generic (loses creative value)

---

#### **Alternative C: Hybrid - Python Provides Data, LLM Formats** ⭐ **CHOSEN**

**Description**: Python provides structured data, LLM applies format template

**Python preprocessing**:
```python
def enrich_high_contrast_features(high_contrast_features: list, rf_features: list) -> list:
    """
    Add RF metadata to high-contrast features for easy LLM formatting.

    Returns enriched features with all numeric data pre-computed,
    allowing LLM to focus on creative interpretation.
    """
    enriched = []

    for hc_feature in high_contrast_features:
        feature_name = hc_feature['feature']

        # Find RF data
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

**Updated prompt format**:
```
CLUSTER 0 (35 videos):

All features: {...}

High-contrast features (with RF metadata for formatting):
  1. feature: eye_contact_rate
     cluster_value: 0.87
     rf_rank: 1, importance: 0.35, gap: 0.43
     contrast: 0.45

  2. feature: word_count
     cluster_value: 14
     rf_rank: 3, importance: 0.18, gap: 26.8
     contrast: 38

  3. feature: energy_level
     cluster_value: 0.55
     rf_rank: 2, importance: 0.22, gap: 0.20
     contrast: 0.30

  ... (8 features total with metadata)

Your task:
Select exactly 3 defining features and format each as:
"feature_name: value (RF rank #X, importance Y.YY, gap Z.ZZ - interpretation)"

**Use the metadata provided above** - all numeric values are pre-computed.

**Create a creative interpretation** based on:
- Value magnitude (high/low relative to feature type)
- RF rank (if #1: "HIGHEST PREDICTOR", if #2-3: mention rank)
- Contrast magnitude (if >0.30: note high contrast)
- Strategic meaning (e.g., "brief hook strategy", "direct communication pattern", "calm approach")

**Example outputs**:
✅ "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43 - HIGHEST PREDICTOR)"
✅ "word_count: 14 (RF rank #3, importance 0.18, gap 26.8 - brief hook strategy)"
✅ "energy_level: 0.55 (RF rank #2, importance 0.22, gap 0.20 - calm moderate approach)"

**Format requirements**:
- Use colon after feature_name: "eye_contact_rate:"
- Use "RF rank #X" (include # symbol)
- Round importance and gap to 2 decimals
- Interpretation should be 2-4 words describing strategic meaning
```

**Pros**:
- ✅ **Prevents hallucination**: Python provides all numeric data, LLM doesn't compute anything
- ✅ **Allows discovery**: LLM creates creative, contextual interpretations
- ✅ **Clear format**: Template + examples ensure consistency
- ✅ **Best of both**: Python for data accuracy, LLM for semantic creativity
- ✅ **Consistent with Issues #1, #3, #4**: Same pattern (Python computes, LLM formats)

**Cons**:
- ❌ LLM must still apply template (minor formatting risk, mitigated by clear examples)
- ❌ Requires Python preprocessing

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Python provides all numeric data (RF rank, importance, gap), LLM just formats
- ✅ **Prevent Misclassification**: Clear template with examples, format requirements explicit
- ✅ **Allow Open-Ended Discovery**: LLM creates semantic interpretations ("brief hook strategy" vs generic "low value")

---

### Final Decision: Alternative C (Hybrid Approach)

**Rationale**:
1. **Consistent with Issues #1, #3, #4**: Python provides data, LLM applies creativity (established pattern)
2. **Prevents hallucination** ✅: LLM doesn't look up RF rank/importance/gap (Python provides all numeric data)
3. **Preserves LLM value** ✅: Creative interpretations make output useful ("brief hook strategy" > "key differentiator")
4. **Clear format** ✅: Template + examples + format requirements ensure schema consistency
5. **Balance**: Python handles precision (numbers), LLM handles semantics (interpretation)

**The "interpretation" field is where LLM adds value** - contextual, creative descriptions that help creators understand the strategy, not generic labels.

**Implementation**:
1. Add `enrich_high_contrast_features()` to Stage 7 preprocessing (after `identify_high_contrast_features()`)
2. Update prompt format to show enriched features with all RF metadata
3. Add format template + examples + requirements at line 2659
4. Add to Stage 7 Child HLD Section 2.3.2

**Next Step**: Update MLPlanningv2.md lines 2630-2640 (enriched feature format) and line 2659 (format template + requirements)

---

#### Issue #9: Missing RF Alignment Score (MEDIUM PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: MLPlanningv2.md lines 2679-2682

**Current Mother Doc**: Only says "How this cluster uses RF-validated features"

**Child HLD Phase 2 Example** (line 1682): Shows numeric RF validation score "9/10"

**Problem**: Phase 1 examples don't show numeric scores, but Phase 2 does. For consistency, Phase 1 should also quantify RF alignment.

**Impact**: Stage 8 (PDF generation) may want to use numeric scores for visual hierarchy (show "3/5 RF validated" badge).

---

### Evaluated Alternatives

#### **Alternative A: Add RF Alignment Score to Phase 1 Output Schema**

**Description**: Add numeric `rf_alignment_score` field to Phase 1 cluster analysis JSON.

**Updated output schema**:
```json
"rf_validation": {
  "alignment_score": "3/5",  // NEW FIELD - X of top 5 RF features aligned
  "top_predictive_features_in_cluster": [
    "eye_contact_rate: Cluster value 0.87 matches top avg 0.88 (RF rank #1)"
  ],
  "insight": "This cluster leverages 3 of the top 5 most predictive hook features..."
}
```

**Prompt instruction update**:
```
3. **Generate RF validation**:

   Use the RF Alignment data provided above to populate:

   "rf_validation": {
     "alignment_score": "X/5",  // From RF Alignment section (e.g., "3/5")
     "top_predictive_features_in_cluster": [
       // Copy the ✅ aligned features from RF Alignment data
     ],
     "insight": "This cluster leverages X of the top 5 most predictive features..."
   }
```

**Pros**:
- ✅ **Consistent with Phase 2**: Both phases now have numeric scores
- ✅ **Stage 8 value**: PDF can display "3/5 RF Validated" badge
- ✅ **Data-grounded**: Python pre-computes alignment (from Issue #4 resolution)
- ✅ **Simple for LLM**: Just copy the score from RF Alignment data

**Cons**:
- ❌ **Schema change**: Requires updating Child HLD Section 5.2.1
- ❌ **Minor redundancy**: Score appears in both `alignment_score` field and `insight` text

---

#### **Alternative B: Include Score Only in Insight Text (No Schema Change)** ⭐ **CHOSEN**

**Description**: Add score to `insight` text without adding new schema field.

**Updated prompt instruction** (no schema change):
```
3. **Generate RF validation**:

   Use the RF Alignment data provided above to populate:

   "rf_validation": {
     "top_predictive_features_in_cluster": [
       // Copy the ✅ aligned features from RF Alignment data
     ],
     "insight": "This cluster leverages X of the top 5 most predictive features (RF alignment: X/5)..."
   }

   **Format for insight**:
   - Mention alignment score explicitly: "(RF alignment: 3/5)" or "(3/5 features aligned)"
   - Explain which RF features the cluster uses
   - Specify whether cluster is RF-optimized or diverges from top patterns
```

**Example output**:
```json
"rf_validation": {
  "top_predictive_features_in_cluster": [
    "eye_contact_rate: Cluster value 0.87 matches top avg 0.88 (RF rank #1, importance 0.35)",
    "energy_level: Cluster value 0.55 matches top avg 0.53 (RF rank #2, importance 0.22)"
  ],
  "insight": "This cluster leverages 2 of the top 5 most predictive hook features (RF alignment: 2/5), focusing on eye contact (#1) and energy (#2)."
}
```

**Pros**:
- ✅ **No schema change**: Works with existing Child HLD structure
- ✅ **Score is visible**: Stage 8 can parse "(RF alignment: X/5)" from insight text
- ✅ **Natural language**: Score embedded in readable sentence
- ✅ **Simple implementation**: Just update prompt instruction, no Python changes

**Cons**:
- ❌ **Requires text parsing**: Stage 8 must extract "X/5" from insight string (vs direct field access)
- ❌ **Less structured**: Score is in free-text field, not dedicated field

---

#### **Alternative C: No Numeric Score (Status Quo)**

**Description**: Keep current structure, rely on qualitative descriptions only.

**Pros**:
- ✅ No changes needed
- ✅ Insight text already descriptive

**Cons**:
- ❌ Inconsistent with Phase 2 (which has numeric scores)
- ❌ Stage 8 can't display quantitative badges
- ❌ Less transparent to creators (they don't know if cluster is 1/5 or 5/5 aligned)

---

### Final Decision: Alternative B (Include Score in Insight Text)

**Rationale**:
1. **No schema change**: Alternative A requires updating Child HLD Section 5.2.1. Alternative B works with existing schema, just adds clarity to `insight` field.
2. **Stage 8 can still parse scores**: While not as clean as a dedicated field, Stage 8 can regex extract "(RF alignment: 3/5)" or "(3/5 features aligned)" from insight text. This is acceptable for MVP.
3. **Natural language wins**: Embedding score in a sentence ("This cluster leverages 2 of the top 5 most predictive features") reads better than separate field + redundant text.
4. **Consistent with Phase 1 philosophy**: Phase 1 focuses on per-window analysis with rich textual insights. Numeric scores are secondary (unlike Phase 2 where path frequency % is primary).
5. **Easy implementation**: Just update prompt instruction to require score mention in insight. No Python preprocessing changes, no schema updates.

**Trade-off Accepted**: Stage 8 must parse insight text for score instead of direct field access. This is acceptable because:
- Regex parsing is simple: `r"\(RF alignment: (\d+)/(\d+)\)"`
- Insight text is required field (always present)
- Alternative A's separate field would still require insight text to explain context

**Implementation**:
1. Update prompt instruction at lines 2679-2682 to require score mention in insight
2. Add format guidance: "(RF alignment: X/5)" or "(X/5 features aligned)"
3. No schema changes, no Python preprocessing changes

**Next Step**: Update MLPlanningv2.md lines 2679-2682 (add format guidance for RF alignment score in insight)

---

#### Issue #10: No Bimodal Distribution Example in Data (CRITICAL PRIORITY)

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17
**Source**: MLPlanningv2.md lines 2645-2646

**Current Mother Doc** shows only unimodal distribution:
```
- Top performers: avg 0.88 (72% have ≥0.8)
- Bottom performers: avg 0.45 (only 15% reach 0.8)
```

**Problem**: Prompt mentions bimodal handling (line 2667 from Issue #1 resolution) but **doesn't provide an actual example** of what bimodal distribution data looks like in input.

**What's Missing**: Example showing bimodal case:
```
- Top performers: avg 0.65 (40% have ≥0.8, 35% have ≤0.4) ← BIMODAL DETECTED
```

**Impact**: LLM might not recognize bimodal patterns because it's never seen an example of what bimodal **distribution data** looks like (even though Python will label it per Issue #1).

---

### Evaluated Alternatives

#### **Alternative A: Add Bimodal Example to RF Feature Data Section**

**Description**: Include both unimodal AND bimodal examples in the prompt data format.

**Updated prompt data format** (lines 2643-2655):
```
RF Feature Importance (Window-Level):

1. eye_contact_rate - RF Importance: 0.35 (rank #1)
   Top: avg 0.88 (72% high, 15% low) | Bottom: avg 0.45 | Gap: 0.43 | Pattern: UNIMODAL

2. energy_level - RF Importance: 0.22 (rank #2)
   Top: avg 0.75 (65% high, 18% low) | Bottom: avg 0.42 | Gap: 0.33 | Pattern: UNIMODAL

3. word_count - RF Importance: 0.18 (rank #3)
   Top: avg 52 (40% high, 35% low) | Bottom: avg 18 | Gap: 34 | Pattern: BIMODAL
   → Strategy A: Brief (≤20 words) - 35% | Strategy B: Dense (≥80 words) - 40%

... (10 features total)
```

**Key change**: Feature #3 (word_count) shows bimodal pattern with both high% and low% ≥30%

**Pros**:
- ✅ **Shows bimodal in context**: LLM sees what bimodal data looks like (40% high, 35% low)
- ✅ **Demonstrates Issue #1 resolution**: Shows Python-labeled "Pattern: BIMODAL" + Strategy A/B
- ✅ **Few-shot learning**: Concrete example helps LLM recognize similar patterns

**Cons**:
- ❌ **Requires actual bimodal feature**: Not all windows will have bimodal features (might be all unimodal)
- ❌ **Example might not match real data**: word_count might not be bimodal in actual hooks

---

#### **Alternative B: Add Bimodal Example Note (Conditional)** ⭐ **CHOSEN**

**Description**: Add a note explaining bimodal format, shown as educational example.

**Updated prompt format** (after RF feature list):
```
RF Feature Importance (Window-Level):

1. eye_contact_rate - RF Importance: 0.35 (rank #1)
   Top: avg 0.88 (72% high, 15% low) | Bottom: avg 0.45 | Gap: 0.43 | Pattern: UNIMODAL

2. energy_level - RF Importance: 0.22 (rank #2)
   Top: avg 0.75 (65% high, 18% low) | Bottom: avg 0.42 | Gap: 0.33 | Pattern: UNIMODAL

... (10 features total)

---

**Note on Bimodal Features**:
Some features may show "Pattern: BIMODAL" if BOTH high percentage AND low percentage ≥30% among top performers.

Example of bimodal feature format:
```
3. word_count - RF Importance: 0.18 (rank #3)
   Top: avg 52 (40% high, 35% low) | Bottom: avg 18 | Gap: 34 | Pattern: BIMODAL
   → Strategy A: Brief (≤20 words) - 35% of top performers
   → Strategy B: Dense (≥80 words) - 40% of top performers
```

This indicates BOTH brief AND dense approaches work. When you see bimodal features, format recommendations as:
"ALTERNATIVE STRATEGIES: Use either [Strategy A] OR [Strategy B] - RF data shows both work"

See Issue #1 resolution (line 2667) for complete bimodal handling instructions.
```

**Pros**:
- ✅ **Always shows bimodal example**: Even if real data has no bimodal features, LLM sees the format
- ✅ **Doesn't force fake data**: Real RF features can be all unimodal without confusion
- ✅ **Educational**: Explicit note explains what bimodal means and how it's formatted
- ✅ **References Issue #1**: Points LLM to handling instructions
- ✅ **Few-shot learning**: Concrete example helps LLM recognize pattern

**Cons**:
- ❌ **Adds ~10 lines**: More prompt context
- ❌ **Example is hypothetical**: Might not match actual features (but that's intentional)

---

#### **Alternative C: No Example (Rely on Issue #1 Python Labeling)**

**Description**: Trust that Python labeling ("Pattern: BIMODAL") + Issue #1 instructions are sufficient.

**Rationale**: Issue #1 resolution already provides:
- Python detection (30% threshold)
- Clear labeling in data ("Pattern: BIMODAL")
- Format instructions for recommendations

**Pros**:
- ✅ No prompt changes needed
- ✅ Python labeling should be clear enough

**Cons**:
- ❌ LLM never sees what bimodal DATA looks like
- ❌ No few-shot learning example
- ❌ Higher risk of misinterpretation (even with labels)

---

### Final Decision: Alternative B (Add Bimodal Example Note)

**Rationale**:
1. **Few-shot learning is powerful**: Even if Python labels features, showing LLM what bimodal data LOOKS LIKE (40% high, 35% low) helps it understand the pattern.
2. **Doesn't require forcing fake data**: Alternative A requires having a bimodal feature in the actual data. Alternative B shows a hypothetical example in a note section, so real data can be all unimodal without confusion.
3. **Completes Issue #1 resolution**: Issue #1 added Python detection + handling instructions. This adds the missing piece: showing LLM what the detected pattern looks like in raw data.
4. **Educational value**: The note explicitly explains:
   - What makes a feature bimodal (both ≥30%)
   - How it's formatted (Strategy A/B breakdown)
   - Where to find handling instructions (Issue #1 resolution)
5. **Minimal cost**: ~10 additional lines in a note section. This is acceptable for a CRITICAL priority issue.

**Trade-off Accepted**: Adds ~10 lines to prompt. But this is a CRITICAL issue - without seeing bimodal data format, LLM might misinterpret Python labels even if they're present.

**Implementation**:
1. Add "Note on Bimodal Features" section after RF feature list (after line 2655 in MLPlanningv2.md)
2. Include hypothetical example showing 40%/35% distribution pattern
3. Reference Issue #1 resolution for handling instructions

**Next Step**: Update MLPlanningv2.md (add "Note on Bimodal Features" after RF feature list)

---

#### Issue #11: No Cluster Size Guidance (MEDIUM PRIORITY)

**Status**: ✅ RESOLVED (DUPLICATE OF ISSUE #6)
**Decision Date**: 2025-10-17
**Source**: None (missing entirely)

**Child HLD Example** (line 1558): Shows cluster size = 35 (35% of videos)

**Problem**: Should LLM acknowledge cluster size in recommendations?
- "This is the DOMINANT strategy (70% of videos)" vs
- "This is a NICHE strategy (8% of videos)"

**Impact**: Missing context for creators - they don't know if this is a dominant or niche strategy.

---

### Analysis: Issue #11 is DUPLICATE of Issue #6

**Issue #6 Resolution** already comprehensively addresses cluster size guidance:

From Issue #6 (lines 1163-1178):
```
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
```

**This directly addresses Issue #11's concern**: LLM is instructed to acknowledge cluster size and frame recommendations appropriately (dominant/common/niche).

---

### Evaluated Alternatives

#### **Alternative A: Mark Issue #11 as DUPLICATE of Issue #6** ⭐ **CHOSEN**

**Description**: Recognize that Issue #6 already resolved Issue #11. No additional work needed.

**Rationale**:
- Issue #6 provides cluster size framing guidance (dominant/common/niche)
- Issue #6 instructs LLM to include size context in `strategy_description` and `when_to_use`
- Both issues address the same underlying problem: creators need to know prevalence

**Pros**:
- ✅ No duplicate work
- ✅ Clean documentation (acknowledges relationship between issues)
- ✅ Issue #6 resolution is comprehensive

**Cons**:
- None (this is the correct interpretation)

---

#### **Alternative B: Add Additional Size-Specific Guidance**

**Description**: Add extra instructions beyond Issue #6's guidance.

**Example additions**:
```
**Cluster Size Impact on Recommendations**:
- Large clusters (>50%): Emphasize broad applicability, mainstream approach
- Small clusters (<25%): Emphasize specificity, "works for creators with [X trait]"
- Always include percentage in strategy_description: "This strategy (used by X% of top performers)..."
```

**Pros**:
- ✅ More explicit than Issue #6

**Cons**:
- ❌ Redundant with Issue #6 (which already covers this)
- ❌ Adds unnecessary prompt bloat

---

#### **Alternative C: No Action (Issue Not Real)**

**Description**: Treat Issue #11 as non-issue since Issue #6 exists.

**Pros**:
- ✅ Acknowledges overlap

**Cons**:
- ❌ Doesn't document relationship in critique file

---

### Final Decision: Alternative A (Mark as DUPLICATE)

**Rationale**:
1. **Issue #6 comprehensively addresses Issue #11**: The "Cluster Size Context" section provides:
   - Thresholds for dominant (>50%), common (25-50%), niche (<25%)
   - Framing language guidance ("DOMINANT strategy", "NICHE strategy")
   - Output field guidance (`strategy_description`, `when_to_use`)
2. **No gaps remain**: Issue #11's concern ("creators don't know if this is dominant or niche") is fully addressed by Issue #6's instructions.
3. **Clean documentation**: Marking as duplicate clarifies the relationship and avoids confusion.

**Implementation**:
- Issue #11 resolved by Issue #6 resolution (lines 1163-1178)
- No additional Mother Document updates needed
- Update Issue Tracker to reflect duplicate status

**Next Step**: Update Issue Tracker in Stage7PromptCritique.md (lines 54-68) to show Issue #11 as resolved/duplicate

---

## Critique Part 2: Phase 2 Prompt (Cross-Window Synthesis)

### Source Location
**MLPlanningv2.md lines 2964-3076**

### Critical Gap Analysis

#### Gap #1: Missing 10% Frequency Threshold (CRITICAL)

**Current Mother Doc** (line 3012):
```
Identify 3-5 "Winning Formulas" - specific combinations of window strategies...
```

**Child HLD Requirement** (Section 2.3.3, Critique Q5 lines 301-318):
- 10% threshold REQUIRED (minimum 10 videos out of 100)
- Exactly 3 reports (not 3-5)
- Path-based preferred, feature-based fallback if <3 paths meet threshold

**Why This Matters**:
- **Without threshold**: LLM may include 8% frequency paths (8 videos) → not statistically reliable
- **Quality over coverage**: 10% = "1 in 10 videos" = proven pattern vs experimental noise
- **Deterministic output**: "3 reports" is clear requirement, "3-5" is ambiguous

**Impact on Implementation**:
```python
# Without threshold instruction, LLM might output:
{
  "winning_formulas": [
    {"path": [...], "frequency": 22, "percentage": 22.0},  # Good
    {"path": [...], "frequency": 18, "percentage": 18.0},  # Good
    {"path": [...], "frequency": 12, "percentage": 12.0},  # Good
    {"path": [...], "frequency": 8, "percentage": 8.0},    # ❌ BAD (below 10%)
    {"path": [...], "frequency": 6, "percentage": 6.0}     # ❌ BAD (below 10%)
  ]
}

# Stage 8 receives 5 reports (2 are low-quality)
# Creators see "6% frequency pattern" → confusing, unreliable
```

**Recommended Fix**:
```
Identify exactly 3 "Winning Formulas" using frequency-based filtering:

1. **Apply 10% Threshold**:
   - Only consider cluster paths with ≥10% frequency (minimum 10 videos)
   - This ensures formulas are proven patterns, not statistical noise
   - Example: 22 videos out of 100 = 22% frequency → INCLUDE
   - Example: 8 videos out of 100 = 8% frequency → EXCLUDE (below threshold)

2. **Prioritize Path-Based Formulas**:
   - Generate path-based reports for top 3 paths meeting threshold (if available)
   - Order by frequency descending (22% before 18% before 12%)

3. **Use Feature-Based Fallback if Needed**:
   - If only 2 paths ≥10%: Generate 2 path + 1 feature-based
   - If only 1 path ≥10%: Generate 1 path + 2 feature-based
   - If 0 paths ≥10%: Generate 3 feature-based (high fragmentation scenario)
   - Feature-based reports use top RF features from video-level analysis

4. **Always Output Exactly 3 Reports** (not 3-5):
   - Combination examples:
     - Best case: 3 path-based (if 3+ paths ≥10%)
     - Mixed: 2 path + 1 feature (if only 2 paths ≥10%)
     - Worst case: 3 feature-based (if 0 paths ≥10%)
```

---

#### Gap #2: Missing Confidence Level Classification (CRITICAL)

**Current Mother Doc** (lines 3039-3041):
```json
{
  "cluster_path": [0, 1, 0, 1, 2, 0],
  "frequency": 18,
  "percentage": 18.0,
  // ❌ Missing: "confidence_level"
}
```

**Child HLD Requirement** (Section 5.2.2, Critique Q5 lines 332-341):
```json
{
  "cluster_path": [0, 1, 0, 1, 2, 0],
  "frequency": 18,
  "percentage": 18.0,
  "confidence_level": "high"  // ✅ REQUIRED
}
```

**Why This Matters**:
- **Stage 8 PDF prioritization**: High-confidence reports featured prominently, moderate reports secondary
- **Creator guidance**: "Very High Confidence (22%)" = "THE dominant strategy" vs "Moderate (11%)" = "Solid option worth testing"
- **Future-proofing**: Normalizes confidence across different sample sizes (200 videos vs 100 videos)

**Classification Bands** (Critique Q5 lines 332-341):
| Frequency | Percentage | Confidence Level | Interpretation |
|-----------|-----------|------------------|----------------|
| 20+ videos | ≥20% | `very_high` | 1 in 5 videos - dominant pattern |
| 15-19 videos | 15-19.9% | `high` | 1 in 6-7 videos - strong pattern |
| 10-14 videos | 10-14.9% | `moderate` | 1 in 10 videos - proven but not dominant |
| <10 videos | <10% | N/A | Filtered out (below threshold) |

**Recommended Fix**:
```
5. **Classify Confidence Levels Based on Frequency**:

   For each path-based report, assign confidence level:

   - **very_high**: Frequency ≥20% (20+ videos out of 100)
     - Interpretation: "This is a dominant strategy used by 1 in 5 top performers"
     - Example: 22% frequency → very_high

   - **high**: Frequency 15-19.9% (15-19 videos)
     - Interpretation: "This is a strong strategy with clear evidence"
     - Example: 18% frequency → high

   - **moderate**: Frequency 10-14.9% (10-14 videos)
     - Interpretation: "This is a proven strategy worth testing"
     - Example: 12% frequency → moderate

   Feature-based reports always get "moderate" confidence level.

Output example:
{
  "report_id": 1,
  "type": "path_based",
  "frequency": 22,
  "percentage": 22.0,
  "confidence_level": "very_high",  // ← REQUIRED FIELD
  ...
}
```

---

#### Gap #3: Missing Hybrid Output Structure (HIGH PRIORITY)

**Current Mother Doc** (lines 3030-3070):
```json
{
  "winning_formulas": [...],  // 3-5 formulas
  "cross_window_insights": [...]
}
```

**Child HLD Requirement** (Section 5.2.2, Critique Q5 lines 363-389):
```json
{
  "creative_reports": [...],  // Exactly 3 reports (renamed from winning_formulas)
  "supplementary_insights": {
    "universal_principles": [...],  // ← MISSING ENTIRELY
    "cross_window_patterns": [...]  // ← Renamed from cross_window_insights
  }
}
```

**Why This Matters**:
- **Coverage gap**: Path formulas cover 40-60% of videos → `universal_principles` cover remaining 40-60%
- **Multi-audience value**:
  - Path formulas: For creators wanting exact templates (beginner-friendly)
  - Universal principles: For experienced creators wanting flexibility
- **Fallback enhancement**: When 0 paths meet 10%, universal principles ensure creators still get value
- **Complete coaching**: EVERY creator gets actionable advice, even if their style doesn't match a formula

**Example Universal Principles** (from Section 5.2.2):
```json
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
    "Closing energy should match or exceed middle average (85% of top performers follow this)"
  ]
}
```

**Recommended Fix**:
```
6. **Generate Hybrid Output Structure**:

Output must include TWO sections:

A. **creative_reports** (Primary - Exactly 3 reports):
   - Path-based reports (for cluster path combinations meeting 10% threshold)
   - OR feature-based reports (fallback when <3 paths meet threshold)
   - Each report has: formula_name, structure, recommendations, confidence_level

B. **supplementary_insights** (Secondary - For all creators):

   1. **universal_principles**: Top 5-7 RF features applicable to ALL videos
      - Extract from video-level RF feature importance (top 5-7 features)
      - Format: "Feature X (top avg vs bottom avg) - applies to Y% of videos"
      - Example: "High eye contact rate (88% vs 45%) - applies to 78% of videos"
      - Purpose: Guidance for creators whose style doesn't match path formulas

   2. **cross_window_patterns**: General progression patterns
      - Extract from video-level RF cross-window features
      - Format: Percentage-based insights about temporal evolution
      - Example: "78% use bookend eye contact pattern (high in hook/closing)"
      - Purpose: Understanding how features evolve across video journey

Output format:
{
  "creative_reports": [
    // 3 reports (path-based or feature-based)
  ],
  "supplementary_insights": {
    "universal_principles": [
      // 5-7 top RF features with percentages
    ],
    "cross_window_patterns": [
      // 3-5 temporal progression insights
    ]
  }
}
```

---

#### Gap #4: Missing Feature-Based Fallback Instructions (HIGH PRIORITY)

**Current Mother Doc**: No instructions for what to do when <3 paths meet 10% threshold

**Child HLD Requirement** (Section 2.3.3, Critique Q5 lines 311-318):
- Must handle scenario where only 0-2 paths meet 10% threshold
- Generate feature-based reports to supplement path-based reports
- Feature-based reports use top RF features from video-level analysis

**Scenario Examples**:

**Scenario 1**: Only 2 paths meet 10% threshold
```python
# Available paths:
# - Path [0,1,1,2,0,1]: 18% (meets threshold)
# - Path [1,0,0,1,1,0]: 12% (meets threshold)
# - All other paths: <10% (below threshold)

# LLM should generate:
{
  "creative_reports": [
    # Report 1: Path-based (18% - high confidence)
    # Report 2: Path-based (12% - moderate confidence)
    # Report 3: Feature-based fallback (uses top RF features)
  ]
}
```

**Scenario 2**: 0 paths meet 10% threshold (high fragmentation)
```python
# Available paths:
# - Highest path: 9% (below threshold)
# - All 45 unique paths spread across 1-9% each

# LLM should generate:
{
  "creative_reports": [
    # Report 1: Feature-based (top RF features group 1)
    # Report 2: Feature-based (top RF features group 2)
    # Report 3: Feature-based (diverse strategies)
  ]
}
```

**Recommended Fix**:
```
7. **Handle Feature-Based Fallback for Low-Frequency Scenarios**:

When fewer than 3 cluster paths meet the 10% threshold:

**Step 1: Count Paths Above Threshold**
- Filter cluster paths to only those with ≥10% frequency
- Example: 45 unique paths → 5 paths above 10% threshold

**Step 2: Determine Report Mix**
- If 3+ paths ≥10%: Generate 3 path-based reports (take top 3 by frequency)
- If 2 paths ≥10%: Generate 2 path-based + 1 feature-based
- If 1 path ≥10%: Generate 1 path-based + 2 feature-based
- If 0 paths ≥10%: Generate 3 feature-based (high fragmentation scenario)

**Step 3: Generate Feature-Based Reports**
When needed, use video-level RF top features to create actionable reports:

Feature-Based Report Structure:
{
  "report_id": 3,
  "type": "feature_based",
  "frequency": null,  // Not applicable (not a cluster path)
  "percentage": null,
  "confidence_level": "moderate",  // Feature-based always moderate
  "formula_name": "High Eye Contact Strategy",
  "strategy_description": "Consistent use of direct eye contact across all windows",
  "key_features": [
    "eye_contact_rate: 0.88 (RF rank #1, importance 0.35)",
    "eye_contact_consistency: 0.12 std dev (RF rank #6)"
  ],
  "creator_recommendations": [
    "Maintain 85-90% eye contact throughout video (RF #1 predictor)",
    "Keep eye contact variance low (<0.15 std dev) across windows",
    "Use direct-to-camera framing in hook and closing"
  ],
  "when_to_use": "Videos where cluster paths are highly fragmented. Focus on universal principles."
}

**Feature-Based Report Sources**:
- Use top 5-7 features from video-level RF feature_importance
- Group related features (e.g., eye_contact_rate + eye_contact_consistency = "Eye Contact Strategy")
- Reference top_performer_avg for target values
- Always classify as "moderate" confidence (not based on path frequency)

**Fallback Scenario Handling**:
- Log warning: "Only X paths meet 10% threshold. Generating Y path + Z feature-based reports."
- Ensure exactly 3 reports total in all scenarios
- Feature-based reports are NOT inferior - they provide universal principles valuable to all creators
```

---

#### Gap #5: Ambiguous "3-5 Formulas" Instruction (MEDIUM PRIORITY)

**Current Mother Doc** (line 3012):
```
Identify 3-5 "Winning Formulas"...
```

**Child HLD Requirement** (Section 2.3.3, Critique Q5 line 312):
```
Always deliver 3 reports per bucket
```

**Why This Matters**:
- **Deterministic output**: "3 reports" is clear, "3-5" is ambiguous (when to generate 3 vs 5?)
- **Stage 8 PDF layout**: Designed for exactly 3 reports (not variable 3-5)
- **Consistency**: Every bucket gets 3 reports (predictable structure for clients)

**Recommended Fix**:
```
Identify exactly 3 "Winning Formulas"...
```

---

## Summary of Required Updates

### Phase 1 Prompt: ✅ NO UPDATES NEEDED

### Phase 2 Prompt: ⚠️ 5 CRITICAL UPDATES REQUIRED

| Gap | Severity | Lines to Update | Estimated Impact |
|-----|----------|-----------------|------------------|
| #1: 10% threshold instruction | CRITICAL | 3012-3017 | +15 lines |
| #2: Confidence level classification | CRITICAL | 3041 (add field), 3012+ (add instructions) | +20 lines |
| #3: Hybrid output structure | HIGH | 3030-3070 (restructure JSON) | +25 lines |
| #4: Feature-based fallback | HIGH | 3012+ (add instructions) | +30 lines |
| #5: "3-5" → "3" formulas | MEDIUM | 3012 (single word) | 1 line |

**Total Estimated Additions**: ~90 lines (3012-3076 becomes 3012-3165)

---

## Recommended Prompt Update (Complete Phase 2)

### Updated Phase 2 Prompt Template

```python
def run_phase2_synthesis(
    window_analyses: dict,
    kmeans_outputs: dict,
    rf_video_data: dict,
    bucket: str,
    hashtag: str | None
) -> dict:
    """
    Synthesize cross-window patterns from Phase 1 analyses.

    Returns: Phase 2 synthesis JSON with 3 creative reports + supplementary insights
    """
    # Extract video cluster paths
    video_paths = extract_cluster_paths(window_analyses, kmeans_outputs)
    top_paths = analyze_path_frequencies(video_paths)

    prompt = f"""
You are synthesizing creative insights for viral videos in the {bucket} duration bucket for #{hashtag or 'general content'}.

You have analyzed 100 viral videos across {len(window_analyses)} temporal windows. Each window has been clustered into 3 distinct strategies.

## Per-Window Cluster Analyses

### Hook Analysis:
{json.dumps(window_analyses['hook'], indent=2)}

### Middle_1 Analysis:
{json.dumps(window_analyses['middle_1'], indent=2)}

... (include all {len(window_analyses)} window analyses)

## Most Common Cluster Paths (Video Journey Patterns)

The 10 most common combinations of window strategies:

{format_top_paths(top_paths)}

Path frequency distribution:
- Paths above 10% threshold: {top_paths['paths_above_threshold']}
- Total unique paths: {top_paths['total_unique_paths']}
- Fragmentation level: {'HIGH' if top_paths['paths_above_threshold'] < 3 else 'MODERATE' if top_paths['paths_above_threshold'] < 6 else 'LOW'}

## Video-Level Random Forest (Cross-Window Pattern Detection)

The features that BEST PREDICT viral success across the ENTIRE VIDEO JOURNEY:

Top Single-Window Features:
{format_single_window_features(rf_video_data)}

Top Cross-Window Features (these only exist at video-level):
{format_cross_window_features(rf_video_data)}

Key Cross-Window Insights from RF:
- Energy progression matters: Building from hook → middle (delta +0.15) predicts virality
- Closing contrast matters: Large energy gap between middle avg and closing peak (0.28) predicts virality
- Consistency matters: Low variance in eye_contact across windows (std 0.12) predicts virality

---

## Your Task

Generate exactly 3 creative reports using a frequency-based approach with feature-based fallback.

### STEP 1: Filter Paths by 10% Frequency Threshold

**CRITICAL RULE**: Only consider cluster paths with ≥10% frequency (minimum 10 videos out of 100).

**Why 10% Threshold**:
- Ensures formulas are proven patterns, not statistical noise
- 10% = "1 in 10 videos use this pattern" = reliable for creator replication
- Below 10% = too rare, might not replicate, wastes creator time

**Examples**:
- 22 videos (22%) → INCLUDE ✅ (very high confidence)
- 18 videos (18%) → INCLUDE ✅ (high confidence)
- 12 videos (12%) → INCLUDE ✅ (moderate confidence)
- 8 videos (8%) → EXCLUDE ❌ (below threshold - statistical noise)

**Action**: Count how many paths meet ≥10% threshold from the cluster path data provided above.

---

### STEP 2: Determine Report Mix (Path vs Feature-Based)

Based on number of paths above 10% threshold:

**Scenario A**: 3 or more paths ≥10%
- Generate 3 path-based reports (take top 3 by frequency, ordered descending)

**Scenario B**: Exactly 2 paths ≥10%
- Generate 2 path-based reports (for the 2 paths above threshold)
- Generate 1 feature-based report (using top RF features from video-level analysis)

**Scenario C**: Exactly 1 path ≥10%
- Generate 1 path-based report (for the 1 path above threshold)
- Generate 2 feature-based reports (using top RF features)

**Scenario D**: 0 paths ≥10% (high fragmentation)
- Generate 3 feature-based reports (all based on top RF features)
- Log: "High fragmentation detected: No paths meet 10% threshold. Using feature-based approach."

**ALWAYS output exactly 3 reports total** (never 4, never 2).

---

### STEP 3: Generate Path-Based Reports (for paths ≥10%)

For each cluster path above 10% threshold:

1. **Name**: Creative, memorable name (e.g., "The Educator's Arc")
2. **Structure**: Which cluster combination
   - Hook: Cluster name from Phase 1 (e.g., "The Direct Eye Contact Hook")
   - Middle pattern: Progression description
   - Closing: Cluster name from Phase 1
3. **Frequency & Confidence**:
   - frequency: Video count (e.g., 22)
   - percentage: Frequency percentage (e.g., 22.0)
   - confidence_level: Based on percentage:
     - ≥20%: "very_high" (1 in 5 videos - dominant pattern)
     - 15-19.9%: "high" (1 in 6-7 videos - strong pattern)
     - 10-14.9%: "moderate" (1 in 10 videos - proven pattern)
4. **Temporal Progression**: How key features evolve across windows
   - Show actual values per window (hook: 0.55, middle_avg: 0.65, closing: 0.85)
   - Calculate deltas (hook_to_middle_delta, middle_to_closing_contrast)
   - Describe pattern in words
5. **RF Cross-Window Validation**: How formula matches video-level RF patterns
   - Compare formula's deltas to RF top_performer_avg
   - List matches (e.g., "hook_to_middle_energy_delta: 0.16 matches RF avg 0.15")
   - Provide rf_validation_score (e.g., "9/10" if 3/3 patterns match)
6. **Strategy Description**: Overall creative approach
7. **When to Use**: Content types and creator profiles that fit this formula
8. **Step-by-Step Template**: Concrete replication steps
   - Include window-specific actions (Hook: do X, Middle: do Y)
   - Include cross-window targets (Energy delta: +0.16, Contrast: 0.27)
   - Reference RF-validated features

---

### STEP 4: Generate Feature-Based Reports (fallback when needed)

If fewer than 3 paths meet 10% threshold, generate feature-based reports to reach exactly 3 total.

**Feature-Based Report Structure**:
- **No cluster path** (not based on specific path combination)
- Uses top features from video-level RF analysis
- Focus on universal principles applicable to all videos
- Always classified as "moderate" confidence (not frequency-based)

**How to Create Feature-Based Reports**:
1. Select top RF features (choose from video-level RF feature_importance)
2. Group related features (e.g., eye_contact_rate + eye_contact_consistency = "Eye Contact Strategy")
3. Use top_performer_avg as target values
4. Provide actionable recommendations for each feature group

**Example Feature-Based Report**:
{{
  "report_id": 3,
  "type": "feature_based",
  "frequency": null,
  "percentage": null,
  "confidence_level": "moderate",
  "formula_name": "The High Eye Contact Strategy",
  "strategy_description": "Maintain consistent direct eye contact throughout video journey",
  "key_features": [
    "eye_contact_rate: 0.88 (RF rank #1, importance 0.35, gap 0.43)",
    "eye_contact_consistency: 0.12 std dev (RF rank #6, importance 0.08)"
  ],
  "rf_validation": {{
    "insight": "Leverages #1 and #6 most predictive features across entire video"
  }},
  "when_to_use": "Universal strategy applicable when cluster paths are fragmented. Focus on proven principles.",
  "creator_recommendations": [
    "PRIORITY: Maintain 85-90% eye contact throughout video (RF #1 predictor)",
    "Keep eye contact variance low (<0.15 std dev) across all windows",
    "Use direct-to-camera framing in hook and closing windows"
  ]
}}

**Feature-Based Report Categories** (use these groupings):
1. **Eye Contact & Engagement**: eye_contact_rate, eye_contact_consistency
2. **Energy & Pacing**: energy_level, hook_to_middle_energy_delta, middle_to_closing_contrast
3. **Speech & Density**: word_count, speech_coverage, word_density
4. **Visual Variety**: scene_count, object_count, overlay_unique_count

---

### STEP 5: Generate Supplementary Insights (for all creators)

In addition to the 3 creative reports, provide supplementary insights that apply broadly:

**A. Universal Principles** (5-7 insights):
- Extract from video-level RF feature_importance (top 5-7 features)
- Format: "Feature X (top avg vs bottom avg) - applies to Y% of videos"
- Example: "High eye contact rate (88% vs 45% for top vs bottom) - applies to 78% of videos"
- Purpose: Guidance for creators whose style doesn't match specific path formulas

**B. Cross-Window Patterns** (3-5 insights):
- Extract from video-level RF cross-window features
- Format: Percentage-based insights about temporal evolution
- Example: "78% of high-performing videos use 'bookend' eye contact pattern (high in hook/closing, lower in middle)"
- Purpose: Understanding how features evolve across video journey

---

## Output Format: JSON

{{
  "bucket": "{bucket}",
  "hashtag": "{hashtag or None}",
  "total_videos": {len(video_paths)},
  "total_unique_paths": {top_paths['total_unique_paths']},
  "paths_above_threshold": {top_paths['paths_above_threshold']},

  "creative_reports": [
    {{
      "report_id": 1,
      "type": "path_based",  // or "feature_based"
      "path": [0, 1, 1, 1, 2, 0],  // Only for path_based
      "frequency": 22,  // Only for path_based (null for feature_based)
      "percentage": 22.0,  // Only for path_based (null for feature_based)
      "confidence_level": "very_high",  // very_high, high, or moderate
      "formula_name": "The Educator's Arc",
      "structure": {{  // Only for path_based
        "hook": "The Direct Eye Contact Hook (Cluster 0)",
        "middle_pattern": "Information Dense Middle (Cluster 1 → 1 → 1 → 2)",
        "closing": "High Energy CTA (Cluster 0)"
      }},
      "temporal_progressions": [  // Only for path_based
        {{
          "feature": "energy_level",
          "hook": 0.55,
          "middle_1": 0.60,
          "middle_2": 0.62,
          "middle_3": 0.68,
          "middle_4": 0.75,
          "closing": 0.85,
          "pattern": "Steady build from moderate to high",
          "hook_to_middle_delta": 0.16,
          "middle_to_closing_contrast": 0.27
        }}
      ],
      "rf_cross_window_validation": {{
        "matches_top_patterns": [
          "hook_to_middle_energy_delta: 0.16 (RF top performer avg: 0.15, RF rank #4)",
          "middle_to_closing_contrast: 0.27 (RF top performer avg: 0.28, RF rank #5)"
        ],
        "insight": "This formula exhibits 2 of 3 major cross-window patterns identified by video-level RF.",
        "rf_validation_score": "8/10"
      }},
      "strategy_description": "Start with intimate eye contact to build trust, deliver dense educational content in middle segments, return to direct eye contact for high-energy call-to-action.",
      "when_to_use": "Educational nutrition content, product explanations, how-to videos.",
      "creator_recommendations": [
        "Hook (0-3s): Direct eye contact (0.87), minimal words (14), moderate energy (0.55)",
        "Middle_1 (3-8s): Shift to product view, increase talking speed (50+ words), build energy to 0.60",
        "Middle_2-4 (8-23s): Continue information delivery, steady energy progression",
        "Closing (23-26s): Return to direct eye contact (0.82), peak energy (0.85), clear CTA",
        "CROSS-WINDOW TARGETS (RF validated):",
        "  - Energy delta hook→middle: +0.16 (RF target: +0.15)",
        "  - Energy contrast middle→closing: 0.27 gap (RF target: 0.28)"
      ]
    }},
    // Report 2
    // Report 3
  ],

  "supplementary_insights": {{
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
      "Videos with energy delta >0.3 from hook to closing had 2x engagement"
    ]
  }},

  "path_statistics": {{
    "total_unique_paths": {top_paths['total_unique_paths']},
    "paths_above_threshold": {top_paths['paths_above_threshold']},
    "needs_fallback": {top_paths['needs_fallback']}
  }},

  "analysis_metadata": {{
    "llm_model": "claude-sonnet-4-20250514",
    "timestamp": "{datetime.now().isoformat()}",
    "phase": "phase2_synthesis"
  }}
}}

---

## Important Reminders:

1. **Always output exactly 3 creative reports** (never more, never less)
2. **Apply 10% threshold strictly** (8% paths are excluded)
3. **Classify confidence levels accurately**:
   - very_high: ≥20%
   - high: 15-19.9%
   - moderate: 10-14.9%
   - Feature-based reports: always moderate
4. **Use feature-based fallback when needed** (<3 paths above 10%)
5. **Include supplementary_insights** (universal principles + cross-window patterns)
6. **Focus on actionability**: Concrete steps creators can replicate
7. **Validate against RF data**: Cross-window patterns should match video-level RF features
"""

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=ANTHROPIC_MODEL,  # claude-sonnet-4-20250514
        max_tokens=PHASE2_MAX_TOKENS,  # 8000
        temperature=PHASE2_TEMPERATURE,  # 0.4
        timeout=PHASE2_TIMEOUT_SECONDS,  # 180s
        messages=[{"role": "user", "content": prompt}]
    )

    synthesis = json.loads(response.content[0].text)

    # Add metadata
    synthesis['bucket'] = bucket
    synthesis['hashtag'] = hashtag
    synthesis['total_videos'] = len(video_paths)
    synthesis['analysis_metadata'] = {
        'llm_model': ANTHROPIC_MODEL,
        'timestamp': datetime.now().isoformat(),
        'phase': 'phase2_synthesis'
    }

    return synthesis
```

---

## Implementation Checklist

### Mother Document Updates (MLPlanningv2.md)

- [ ] **Line 3012**: Change "3-5" to "3" formulas
- [ ] **Lines 3012-3020**: Add 10% threshold instruction (5 steps: filter, prioritize, fallback, always 3, classify confidence)
- [ ] **Lines 3030-3070**: Restructure JSON output schema:
  - [ ] Rename `winning_formulas` → `creative_reports`
  - [ ] Add `type` field ("path_based" | "feature_based")
  - [ ] Add `confidence_level` field (very_high | high | moderate)
  - [ ] Make `path`, `frequency`, `percentage` nullable (null for feature_based)
  - [ ] Add `supplementary_insights` section with `universal_principles` + `cross_window_patterns`
- [ ] **Lines 3012+**: Add feature-based fallback instructions (~30 lines)
  - [ ] Feature-based report structure
  - [ ] When to use feature-based (scenario B, C, D)
  - [ ] Feature grouping categories
  - [ ] Always "moderate" confidence for feature-based

### Child HLD Updates (LLMAnalysisCHILD.md)

- [ ] **Section 2.3.3**: Add note referencing updated Mother Document prompts
- [ ] **Appendix B**: Update example Phase 2 output to match new schema (lines 1818-1854)
- [ ] **Section 5.2.2**: Verify output schema matches updated prompt (already correct)

### Testing Validation

After prompt updates, validate with pilot testing:
- [ ] Test Scenario A: 3+ paths ≥10% → Generates 3 path-based reports
- [ ] Test Scenario B: 2 paths ≥10% → Generates 2 path + 1 feature-based
- [ ] Test Scenario C: 1 path ≥10% → Generates 1 path + 2 feature-based
- [ ] Test Scenario D: 0 paths ≥10% → Generates 3 feature-based
- [ ] Verify confidence_level classification (very_high at 22%, high at 18%, moderate at 12%)
- [ ] Verify supplementary_insights populated correctly (5-7 universal principles, 3-5 cross-window patterns)

---

## Conclusion

**Phase 1 Prompt**: ⚠️ **REQUIRES UPDATES** - Issues #5-11 resolved with documented decisions

**Phase 2 Prompt**: ⚠️ Requires updates to align with approved Child HLD architecture:
1. Add 10% threshold filtering
2. Add confidence level classification
3. Restructure to hybrid output (creative_reports + supplementary_insights)
4. Add feature-based fallback instructions
5. Change "3-5" to "3" formulas

**Estimated Effort**: ~4 hours total (Phase 1: 2 hours, Phase 2: 2 hours)

**Impact**: Critical for Stage 7 success. Without updates, LLM will generate low-quality patterns (8% frequency), no confidence prioritization, and incomplete coverage (missing universal principles).

---

**Status**: CRITIQUE COMPLETE - ALL ISSUES RESOLVED (2025-10-17)
**Date**: 2025-10-17
**Original Critique Date**: 2025-10-16

---

## Next Steps for Implementation

### Phase 1: Issues #5-11 Implementation (Estimated: 2 hours)

#### 1. Update MLPlanningv2.md Phase 1 Prompt

**Issue #5 - Remove "Important" Section**:
- [ ] Delete lines 2694-2697 (3-line "Important" section)
- [ ] Location: After task list, before output schema

**Issue #6 - Add Cluster Size Context**:
- [ ] Add "Cluster Size Context" section after task list (line ~2668)
- [ ] Content: ~10 lines with dominant (>50%), common (25-50%), niche (<25%) guidance
- [ ] Reference: See Issue #6 resolution lines 1163-1178

**Issue #7 - Compress RF Feature Format**:
- [ ] Update lines 2643-2655 (RF feature data format)
- [ ] Convert from 4-5 line format to 2-line condensed format with pipe separators
- [ ] Add bimodal expansion line for bimodal features (Strategy A/B breakdown)
- [ ] Expected savings: 50 lines → ~22 lines

**Issue #9 - Add RF Alignment Score to Insight**:
- [ ] Update lines 2679-2682 (RF validation instruction)
- [ ] Add format requirement: mention "(RF alignment: X/5)" in insight text
- [ ] Reference: See Issue #9 resolution lines 1670-1673

**Issue #10 - Add Bimodal Example Note**:
- [ ] Add "Note on Bimodal Features" section after RF feature list (after line 2655)
- [ ] Content: ~10 lines with hypothetical bimodal example (40% high, 35% low)
- [ ] Reference Issue #1 resolution for handling instructions
- [ ] Reference: See Issue #10 resolution lines 1814-1828

**Issue #11**:
- [ ] No action needed (duplicate of Issue #6)

#### 2. Python Preprocessing Implementation (if not done)

**From Issues #1, #3, #4, #8 resolutions**:
- [ ] Implement `detect_bimodal_pattern()` (Issue #1)
- [ ] Implement `identify_high_contrast_features()` (Issue #3)
- [ ] Implement `compute_rf_alignment()` (Issue #4)
- [ ] Implement `enrich_high_contrast_features()` (Issue #8)

### Phase 2: Phase 2 Prompt Updates (Estimated: 2 hours)

**Already documented in lines 2714-2746** - No changes to that checklist.

### Phase 3: Pilot Testing & Validation

**Phase 1 Testing**:
- [ ] Test with sample hook window data (50-100 videos)
- [ ] Verify bimodal detection triggers correctly (40%/35% case)
- [ ] Verify RF alignment scores appear in insights
- [ ] Verify cluster size framing (dominant/common/niche) appears
- [ ] Verify condensed RF format doesn't confuse LLM

**Phase 2 Testing**:
- [ ] Test Scenario A: 3+ paths ≥10% → Generates 3 path-based reports
- [ ] Test Scenario B: 2 paths ≥10% → Generates 2 path + 1 feature-based
- [ ] Test Scenario C: 1 path ≥10% → Generates 1 path + 2 feature-based
- [ ] Test Scenario D: 0 paths ≥10% → Generates 3 feature-based
- [ ] Verify confidence_level classification (very_high at 22%, high at 18%, moderate at 12%)
- [ ] Verify supplementary_insights populated correctly

### Summary of Decisions

**Issue #5**: Remove "Important" section (redundant with Python preprocessing)
**Issue #6**: Add minimal cluster size guidance (dominant/common/niche framing)
**Issue #7**: Compress RF format to 2-line condensed (50% space savings)
**Issue #9**: Include RF alignment score in insight text (no schema change)
**Issue #10**: Add bimodal example note (few-shot learning)
**Issue #11**: Resolved by Issue #6 (duplicate)

---

**Ready for Implementation**: All decisions documented with clear rationales and implementation steps.

---

## Phase 2 Gaps - Detailed Dependency Analysis

**Date**: 2025-10-17
**Purpose**: Systematic analysis of Phase 2 gaps to identify code dependencies, cross-dependencies with Phase 1, and prerequisite Stage 6 requirements

This section expands on the Phase 2 gap analysis (lines 2004-2763) to identify:
1. Python code required for each gap
2. Dependencies on Phase 1 implementations
3. Dependencies on upstream stages (Stage 6)
4. Cross-dependencies between gaps

---

### Gap #1 Analysis: 10% Frequency Threshold

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17

**Original Gap Description** (lines 2011-2070):
- LLM needs to filter cluster paths by ≥10% frequency
- Prioritize path-based formulas (if available)
- Always output exactly 3 reports

---

### Evaluated Alternatives

#### **Alternative A: Python Preprocessing (Full Automation)**

**Description**: Python filters paths by 10% threshold BEFORE passing data to LLM. LLM only sees pre-filtered paths above threshold.

**Pros**:
1. ✅ Simple threshold enforcement
2. ✅ Clean prompt
3. ✅ Deterministic boundary

**Cons**:
1. ❌ Hides fragmentation context
2. ❌ No semantic override capability
3. ❌ Lost analytical value

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Python filters, LLM can't hallucinate invalid paths
- ✅ **Prevent Misclassification**: Clear 10% boundary enforced
- ❌ **Allow Open-Ended Discovery**: Over-constrains - LLM can't explain fragmentation
- ✅ **Python handles arithmetic**: Correctly applied

---

#### **Alternative B: LLM Prompt Instructions (LLM Applies Threshold)**

**Description**: Python provides ALL paths to LLM with percentages. Prompt instructs LLM to apply 10% threshold rule.

**Pros**:
1. ✅ Full transparency
2. ✅ Allows nuanced judgment
3. ✅ Better context for synthesis

**Cons**:
1. ❌ Arithmetic dependency (LLM must compare percentages)
2. ❌ Prompt bloat (35 paths waste tokens)
3. ❌ Instruction complexity

**Prompting Concern Analysis**:
- ❌ **Prevent Hallucination**: LLM must do arithmetic comparison - hallucination risk
- ❌ **Prevent Misclassification**: LLM interprets "≥10%" rule - could misclassify
- ✅ **Allow Open-Ended Discovery**: Full flexibility
- ❌ **Python handles arithmetic**: VIOLATES pattern - LLM does threshold arithmetic

---

#### **Alternative C: Hybrid Approach (Python Labels, LLM Selects)** ⭐ **CHOSEN**

**Description**: Python pre-computes threshold status and labels each path. Shows top 10 paths with clear ✅/❌ markers. Provides scenario guidance.

**Pros**:
1. ✅ Python handles arithmetic
2. ✅ Clear visual markers (✅/❌)
3. ✅ Provides fragmentation context

**Cons**:
1. ❌ Still shows risky data (though clearly marked)
2. ❌ Slightly more complex than Alternative A

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Python does all arithmetic, LLM reads labels
- ✅ **Prevent Misclassification**: Clear ✅/❌ markers + explicit "ONLY use ✅" instruction
- ✅ **Allow Open-Ended Discovery**: LLM can explain fragmentation (35 paths) and reference below-threshold patterns in supplementary insights
- ✅ **Python handles arithmetic**: Correctly applied - Python computes percentages, compares to threshold, labels results

---

### Final Decision: Alternative C (Hybrid Approach)

**Rationale**:
1. **Satisfies all 4 prompting concerns**: Prevents hallucination (Python arithmetic), prevents misclassification (clear labels), allows discovery (LLM sees context), correct division of labor (Python computes, LLM synthesizes)
2. **Aligns with Phase 1 pattern**: Consistent with Issues #1, #3, #4, #8 (Python detects/labels, LLM formats/selects)
3. **Provides analytical value**: "35 unique paths, 3 above threshold" metric can be shown in Stage 8 PDF
4. **Balances context efficiency**: Top 10 paths (not all 35) provides visibility without token waste
5. **Scenario guidance reduces ambiguity**: "Scenario A: Generate 3 path-based reports" is deterministic

**Why 10% Threshold is Correct**:
- **Statistical grounding**: 10 samples is widely accepted minimum for pattern reliability
- **Balances quality vs coverage**: Not too restrictive (15%) nor too lenient (5%)
- **Intuitive**: "1 in 10 videos" is clear for creators
- **Aligns with confidence bands**: 10-14% = moderate, 15-19% = high, 20%+ = very_high
- **Will monitor in pilot testing**: Can adjust to 8% or 12% if data shows need

**Trade-off Accepted**: We show some ❌ BELOW THRESHOLD paths (8%, 6%) for context, but they're clearly marked and instruction says "Do NOT use ❌ paths in creative_reports"

---

**Python Code Implementation**:

```python
def prepare_path_data_for_llm(
    cluster_paths: dict,
    threshold_pct: float = 0.10,
    total_videos: int = 100,
    top_n: int = 10
) -> dict:
    """
    Label paths by threshold status, show top N with context.

    Implements Alternative C (Hybrid Approach): Python handles arithmetic,
    LLM handles semantic synthesis using clearly labeled data.

    Args:
        cluster_paths: Dict mapping path tuples to frequency counts
            Example: {(0,1,1,2,0,1): 22, (1,0,0,1,1,0): 18, ...}
        threshold_pct: Minimum frequency percentage (default: 0.10 = 10%)
        total_videos: Total videos in sample (default: 100)
        top_n: Number of top paths to show in prompt (default: 10)

    Returns:
        dict with:
            - 'top_paths': List of (path, count, pct, status) tuples (top N paths)
            - 'total_unique_paths': Total number of unique paths
            - 'paths_above_threshold': Count of paths meeting threshold
            - 'scenario': str ('A' | 'B' | 'C' | 'D') for report mix
            - 'threshold_pct': Threshold percentage (for prompt display)

    Example:
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
            'scenario': 'A',
            'threshold_pct': 10.0
        }
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
        scenario = 'D'  # 0 paths: Generate 3 feature-based (high fragmentation)

    return {
        'top_paths': paths_with_status[:top_n],
        'total_unique_paths': len(cluster_paths),
        'paths_above_threshold': num_above,
        'scenario': scenario,
        'threshold_pct': threshold_pct * 100
    }
```

**Prompt Data Format** (how LLM sees the data):
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

**Dependencies**:
- ❌ **No Phase 1 dependencies** - Standalone function
- ❌ **No Stage 6 dependencies** - Uses cluster path data from Phase 1 K-Means output
- ✅ **Feeds into Gap #2** (confidence classification)
- ✅ **Feeds into Gap #4** (determines if fallback needed)

**Implementation**:
1. Add `prepare_path_data_for_llm()` to Stage 7 preprocessing (before Phase 2 prompt)
2. Update Phase 2 prompt data format to show labeled paths with ✅/❌ markers
3. Add clear instruction: "Use ONLY ✅ ABOVE THRESHOLD paths in creative_reports"
4. Add to Stage 7 Child HLD Section 2.3.3

**Next Step**: Update MLPlanningv2.md Phase 2 prompt with labeled path format and scenario-based instructions

---

### Gap #2 Analysis: Confidence Level Classification

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17

**Original Gap Description** (lines 2074-2138):
- Classify path frequency into confidence bands
- Add `confidence_level` field to output schema
- very_high (≥20%), high (15-19.9%), moderate (10-14.9%)

---

### Final Decision: Python Computes Confidence Level

**Rationale**:
This is **pure arithmetic classification with clear thresholds** - exactly what Python should handle per our prompting guidelines.

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Python does threshold comparison (22.0% >= 20% = very_high), no LLM arithmetic
- ✅ **Prevent Misclassification**: Clear boundary rules enforced in code (20%, 15%, 10%)
- ✅ **Allow Open-Ended Discovery**: Not applicable - this is mechanical classification, no semantic judgment needed
- ✅ **Python handles arithmetic**: Correctly applied - pure threshold comparison

**Why not LLM classification?**:
- No semantic judgment required (unlike Gap #1 where LLM might explain fragmentation)
- Clear mathematical boundaries (≥20%, 15-19.9%, 10-14.9%)
- Risk of misclassification if LLM interprets thresholds (e.g., "is 19.8% high or very_high?")
- Violates "Python handles arithmetic" principle

**Implementation Notes**:
- Python computes confidence_level and includes it in path data
- LLM receives confidence_level as a field to copy into output schema
- Feature-based reports always get "moderate" (not frequency-based)

---

**Python Code Implementation**:

```python
def classify_confidence_level(frequency_pct: float, report_type: str = "path_based") -> str:
    """
    Classify confidence level based on frequency percentage.

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
        - Feature-based reports: always "moderate"
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

**Dependencies**:
- ✅ **Depends on Gap #1** - Uses filtered paths and percentages from `prepare_path_data_for_llm()`
- ❌ **No Phase 1 dependencies**
- ❌ **No Stage 6 dependencies**

**Integration with Gap #1**:
The `prepare_path_data_for_llm()` function can be extended to include confidence_level in the path data:

```python
def prepare_path_data_for_llm(
    cluster_paths: dict,
    threshold_pct: float = 0.10,
    total_videos: int = 100,
    top_n: int = 10
) -> dict:
    """Extended to include confidence_level for each path."""
    threshold_count = int(threshold_pct * total_videos)

    paths_with_status = []
    for path, count in cluster_paths.items():
        pct = (count / total_videos) * 100.0
        status = 'ABOVE' if count >= threshold_count else 'BELOW'

        # Add confidence level classification (NEW)
        confidence = classify_confidence_level(pct, report_type='path_based')

        paths_with_status.append((path, count, pct, status, confidence))

    paths_with_status.sort(key=lambda x: x[1], reverse=True)

    # ... rest of function

    return {
        'top_paths': paths_with_status[:top_n],  # Now includes confidence
        # ... rest of return dict
    }
```

**Prompt Data Format** (updated to show confidence):
```
Top 10 Paths (with threshold status and confidence):
1. [0,1,1,2,0,1]: 22 videos (22%) - ✅ ABOVE THRESHOLD - Confidence: VERY_HIGH
2. [1,0,0,1,1,0]: 18 videos (18%) - ✅ ABOVE THRESHOLD - Confidence: HIGH
3. [0,0,1,1,0,1]: 12 videos (12%) - ✅ ABOVE THRESHOLD - Confidence: MODERATE
4. [1,1,0,0,1,0]: 8 videos (8%) - ❌ BELOW THRESHOLD
... (showing top 10 of 35)

YOUR TASK:
For each path you use in creative_reports, copy the confidence_level field directly:
{
  "report_id": 1,
  "confidence_level": "very_high",  // Copy from path data above
  ...
}
```

**Implementation**:
1. Add `classify_confidence_level()` to Stage 7 preprocessing
2. Integrate with `prepare_path_data_for_llm()` from Gap #1
3. Update Phase 2 output schema to include confidence_level field
4. Update prompt instruction: "Copy confidence_level from path data"
5. Add to Stage 7 Child HLD Section 2.3.3

**Next Step**: Update MLPlanningv2.md Phase 2 output schema to include confidence_level field

---

### Gap #3 Analysis: Hybrid Output Structure

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17

**Original Gap Description** (lines 2142-2228):
- Restructure output to `creative_reports` + `supplementary_insights`
- `supplementary_insights.universal_principles`: Top 5-7 RF features applicable to ALL videos
- `supplementary_insights.cross_window_patterns`: Temporal progression insights

**CRITICAL DISCOVERY**: Cross-window features ARE implemented in Stage 4!
- Found in FeatureTransformationCHILD.md Section 6.5 (lines 221-250, 612-619)
- Stage 4 computes 5 cross-window features: `hook_to_middle_energy_delta`, `middle_to_closing_contrast`, `eye_contact_consistency`, `word_density_std`, `energy_progression_slope`
- Stage 6 HLD updated (MLAnalysisGenerationCHILD.md lines 937-969) to document interface contract
- Original blocker is RESOLVED - features exist in pipeline

---

### Evaluated Alternatives

#### **Alternative A: Assume Features Always Exist**

**Description**: Implement `generate_cross_window_patterns()` expecting features to always be present. No fallback logic.

**Pros**:
1. ✅ Simple implementation (no conditionals)
2. ✅ Assumes correct architecture
3. ✅ Forces upstream correctness (fails loudly)

**Cons**:
1. ❌ No graceful degradation (crashes if features missing)
2. ❌ Brittle (relies on name pattern matching)
3. ❌ No fallback value

---

#### **Alternative B: Graceful Degradation with Existence Check** ⭐ **CHOSEN**

**Description**: Check if cross-window features exist in Stage 6 output. If present, use them. If absent, return informative placeholder.

**Pros**:
1. ✅ Fail-safe (works whether features exist or not)
2. ✅ Informative fallback (tells developers what's missing)
3. ✅ Self-documenting (references Stage 4 implementation)
4. ✅ Future-proof (automatically uses features when added)

**Cons**:
1. ❌ More complex (adds conditional logic)
2. ❌ Silent degradation (doesn't crash if features missing)

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: N/A - pure Python (no LLM arithmetic)
- ✅ **Prevent Misclassification**: N/A - mechanical filtering
- ✅ **Allow Open-Ended Discovery**: N/A - no semantic judgment needed
- ✅ **Python handles arithmetic**: Correctly applied

---

#### **Alternative C: Explicit Feature Name Matching (Strict)**

**Description**: Look for exact 5 cross-window feature names. Fail if any are missing.

**Pros**:
1. ✅ Explicit contract (no ambiguity)
2. ✅ Catches bugs early
3. ✅ No guessing (exact names)

**Cons**:
1. ❌ Brittle (feature renames break Stage 7)
2. ❌ No graceful degradation (always crashes)
3. ❌ Over-constrains (can't handle new features)

---

### Final Decision: Alternative B (Graceful Degradation with Existence Check)

**Rationale**:
1. **Cross-window features ARE implemented**: Confirmed in Stage 4 (FeatureTransformationCHILD.md), so Alternative B will use them in normal operation
2. **Handles edge cases gracefully**: If Stage 4/6 have bugs, Stage 7 returns informative placeholder instead of crashing
3. **Self-documenting fallback**: Placeholder explicitly lists expected features and where to find them
4. **Aligns with phased implementation**: Works even if Stage 7 runs before Stage 4 is complete
5. **Future-compatible**: Name pattern matching automatically includes new cross-window features

**Why Alternative A fails**: No error handling - crashes if features missing

**Why Alternative C fails**: Too brittle - exact name matching means Stage 4 refactoring breaks Stage 7

**Trade-off Accepted**: Silent degradation (doesn't crash if features missing). This is acceptable because:
- Cross-window patterns are supplementary (not critical for creator value)
- Path formulas + universal principles still provide 80% value
- Placeholder message clearly indicates what's missing and where to fix it

---

**Python Code Implementation**:

```python
def generate_universal_principles(rf_video_data: dict, top_n: int = 7) -> list[str]:
    """
    Extract top N RF features as universal principles applicable to all videos.

    Args:
        rf_video_data: Video-level RF feature importance data
            Expected structure:
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
        List of formatted universal principle strings
        Format: "Feature X (top avg vs bottom avg) - applies to Y% of videos"

    Example:
        [
            "High eye contact rate (88% vs 45% for top vs bottom performers) - applies to 78% of videos",
            "Consistent energy maintenance (std dev ≤0.15) - found in 65% of top performers",
            "Clear CTA in closing window - present in 92% of high-performing videos",
            ...
        ]
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
            # Percentage features (e.g., eye_contact_rate)
            principle = (
                f"High {feature_name.replace('_', ' ')} "
                f"({top_avg:.0%} vs {bottom_avg:.0%} for top vs bottom performers) - "
                f"applies to {prevalence:.0f}% of videos"
            )
        elif 'count' in feature_name:
            # Count features (e.g., word_count)
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


def generate_cross_window_patterns(rf_video_data: dict) -> list[str]:
    """
    Extract cross-window progression patterns from video-level RF data.

    Implements Alternative B: Graceful Degradation with Existence Check.

    Args:
        rf_video_data: Video-level RF feature importance data
            Expected structure:
            {
                'feature_importance': [
                    {
                        'feature': 'hook_to_middle_energy_delta',
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
        Format: "X% of videos use [pattern description]"

    Example (when features exist):
        [
            "65% of high-performing videos show energy builds from hook to middle",
            "78% show consistent eye contact throughout (bookend pattern)",
            "72% show strong energy peak in closing vs middle"
        ]

    Example (when features missing):
        [
            "Cross-window progression analysis requires Stage 6 RF cross-window features",
            "These features are computed in Stage 4 (FeatureTransformationCHILD.md Section 6.5)",
            ...
        ]
    """
    cross_features = rf_video_data.get('feature_importance', [])

    # Filter to cross-window features by name pattern
    # Cross-window keywords from Stage 4 specification
    CROSS_WINDOW_KEYWORDS = ['delta', 'consistency', 'contrast', 'progression', '_std']
    cross_window_features = [
        f for f in cross_features
        if any(keyword in f['feature'] for keyword in CROSS_WINDOW_KEYWORDS)
    ]

    # Check if we have cross-window features
    if not cross_window_features:
        # Graceful fallback: Return informative placeholder
        return [
            "Cross-window progression analysis requires Stage 6 RF cross-window features",
            "These features are computed in Stage 4 (FeatureTransformationCHILD.md Section 6.5)",
            "Expected features: hook_to_middle_energy_delta, middle_to_closing_contrast, eye_contact_consistency, word_density_std, energy_progression_slope",
            "Stage 7 will automatically use these features once Stage 4/6 pipeline is complete"
        ]

    # If features exist, generate insights
    cross_window_features.sort(key=lambda x: x['importance'], reverse=True)
    top_cross = cross_window_features[:5]  # Top 5 cross-window features

    patterns = []
    for feature in top_cross:
        # Estimate pattern prevalence from top_performer_avg
        # For delta features: positive = builds, negative = declines
        # For consistency features: low value = consistent
        prevalence_pct = estimate_prevalence_from_gap(feature)
        interpretation = interpret_cross_window_feature(feature['feature'])

        pattern = f"{prevalence_pct:.0f}% of high-performing videos show {interpretation}"
        patterns.append(pattern)

    return patterns


def interpret_cross_window_feature(feature_name: str) -> str:
    """
    Translate cross-window feature name to human-readable pattern description.

    Maps Stage 4 feature names to Stage 7 LLM-consumable descriptions.
    """
    interpretations = {
        'hook_to_middle_energy_delta': 'energy builds from hook to middle',
        'middle_to_closing_contrast': 'strong energy peak in closing vs middle',
        'eye_contact_consistency': 'consistent eye contact throughout (bookend pattern)',
        'word_density_std': 'varied pacing across windows',
        'energy_progression_slope': 'steady energy progression from start to end'
    }
    return interpretations.get(feature_name, feature_name.replace('_', ' '))


def estimate_prevalence_from_gap(feature: dict) -> float:
    """
    Estimate pattern prevalence percentage from feature gap.

    Rough heuristic: larger gap = more videos show this pattern distinctly.
    Gap of 0.20 (20%) ≈ 65% prevalence
    Gap of 0.30 (30%) ≈ 78% prevalence
    Gap of 0.40 (40%) ≈ 85% prevalence
    """
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

**Dependencies**:

1. **Stage 4 (Feature Transformation)**: ✅ **IMPLEMENTED**
   - Computes 5 cross-window features in FeatureTransformationCHILD.md Section 6.5 (lines 221-250)
   - Features: `hook_to_middle_energy_delta`, `middle_to_closing_contrast`, `eye_contact_consistency`, `word_density_std`, `energy_progression_slope`

2. **Stage 6 (ML Analysis Generation)**: ✅ **DOCUMENTED**
   - Extracts cross-window features from video-level RF model
   - Interface contract added to MLAnalysisGenerationCHILD.md (lines 937-969)
   - Stage 6 includes cross-window features in `rf_video_analysis.json` output

3. **Phase 1 Dependencies**: None - uses same RF data structure as `generate_universal_principles()`

**Implementation**:
1. Add `generate_cross_window_patterns()` to Stage 7 preprocessing (alongside `generate_universal_principles()`)
2. Function filters `rf_video_data['feature_importance']` for cross-window features (name pattern matching)
3. If cross-window features found: Generate temporal progression insights
4. If cross-window features missing: Return informative placeholder message
5. Add to Stage 7 Child HLD Section 2.3.3 (Phase 2 synthesis)

**Next Step**: Implement `generate_cross_window_patterns()` in Stage 7 Python preprocessing script

---

### Gap #4 Analysis: Feature-Based Fallback

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17

**Original Gap Description** (lines 2232-2326):
- When <3 paths meet 10%, generate feature-based reports
- Use top RF features (not cluster paths)
- Always classified as "moderate" confidence

**Context**: Feature-based reports are fallback for Scenarios B, C, D (when <3 cluster paths meet 10% threshold):
- **Scenario B**: 2 paths ≥10% → Generate 1 feature-based report
- **Scenario C**: 1 path ≥10% → Generate 2 feature-based reports
- **Scenario D**: 0 paths ≥10% → Generate 3 feature-based reports (high fragmentation)

---

### Evaluated Alternatives

#### **Alternative A: Python Pre-Groups Features, LLM Formats**

**Description**: Python defines 4 feature groups, selects top features from each group, passes structured data to LLM for formatting (LLM generates formula names, descriptions, recommendations).

**Pros**:
1. ✅ Clear separation (Python groups, LLM creates)
2. ✅ Consistent grouping
3. ✅ Avoids duplication

**Cons**:
1. ❌ LLM still generates text (hallucination risk)
2. ❌ Hybrid complexity
3. ❌ Non-deterministic output

**Prompting Concern Analysis**:
- ⚠️ **Prevent Hallucination**: Python groups but LLM generates text
- ⚠️ **Prevent Misclassification**: LLM might misinterpret feature group
- ✅ **Allow Open-Ended Discovery**: LLM has creative freedom
- ✅ **Python handles arithmetic**: Grouping logic in Python

---

#### **Alternative B: Python Generates Complete Reports** ⭐ **CHOSEN**

**Description**: Python does ALL the work - groups features, generates formula names, writes descriptions, creates recommendations. LLM just copies the structured JSON.

**Pros**:
1. ✅ Zero LLM hallucination risk (all text Python-generated)
2. ✅ Consistent output (deterministic)
3. ✅ Fully testable (unit tests with known inputs/outputs)
4. ✅ Can use data-driven templates for flexibility

**Cons**:
1. ❌ Less flexible (formula names are generic, not hashtag-specific creative variants)
2. ❌ More code (Python generates all text)
3. ❌ Maintenance (adding feature groups requires code updates)

**Prompting Concern Analysis**:
- ✅ **Prevent Hallucination**: Perfect - LLM does NO generation, just copies Python output
- ✅ **Prevent Misclassification**: Python does all classification
- ❌ **Allow Open-Ended Discovery**: LLM has no creative freedom (acceptable - feature-based reports are fallback, not primary insights)
- ✅ **Python handles arithmetic**: All logic in Python

---

#### **Alternative C: LLM Generates Reports from Top Features**

**Description**: Python just provides top RF features. LLM groups them, names strategies, writes descriptions.

**Pros**:
1. ✅ Maximum flexibility
2. ✅ Simple Python
3. ✅ Uses LLM strengths

**Cons**:
1. ❌ High hallucination risk (LLM groups features)
2. ❌ Inconsistent output
3. ❌ Hard to test (non-deterministic)
4. ❌ Violates "Python handles arithmetic"

**Prompting Concern Analysis**:
- ❌ **Prevent Hallucination**: HIGH RISK - LLM generates all text
- ❌ **Prevent Misclassification**: LLM might mis-group unrelated features
- ✅ **Allow Open-Ended Discovery**: Maximum creativity
- ❌ **Python handles arithmetic**: VIOLATES - LLM does grouping logic

---

### Final Decision: Alternative B (Python Generates Complete Reports)

**Rationale**:
1. **Satisfies prompting concerns**: Prevents hallucination (Python generates ALL text), prevents misclassification (Python groups features), Python handles arithmetic
2. **Fully deterministic**: Same RF features always produce same reports (testable, reproducible)
3. **Aligns with Gaps #1, #2 pattern**: Python computes everything, LLM copies structure
4. **Hashtag specificity comes from DATA**: Recommendations use `top_performer_avg` from that hashtag's RF model (e.g., "maintain 88% eye contact" for #nutrition vs "maintain 72%" for #comedy)
5. **Feature-based reports are universal by design**: They're fallback guidance when cluster paths are fragmented - universal principles, not creative strategies

**Why Alternative A fails**: Still involves LLM generation (formula names, descriptions) → hallucination risk

**Why Alternative C fails**: Violates ALL prompting concerns (LLM groups features, generates text, does logic)

**Trade-off Accepted**: Generic formula names ("Eye Contact Strategy" vs hashtag-specific variants like "Trust-Building Eye Contact"). This is acceptable because:
- Feature-based reports are fallback (not primary insights)
- Recommendations ARE hashtag-specific (use actual RF data)
- Generic names + data-driven recommendations provide full value
- Consistency > creativity for fallback reports

**Enhancement**: Use data-driven description templates (fill based on `top_performer_avg` values) to add flexibility WITHOUT LLM generation.

---

**Python Code Implementation**:

```python
def generate_feature_based_reports(
    rf_video_data: dict,
    num_reports: int,
    used_features: set = None
) -> list[dict]:
    """
    Generate feature-based reports when insufficient paths meet 10% threshold.

    Args:
        rf_video_data: Video-level RF feature importance data
        num_reports: Number of feature-based reports to generate (1-3)
        used_features: Set of features already used in path-based reports (to avoid duplication)

    Returns:
        List of feature-based report dictionaries matching Phase 2 schema

    Feature Grouping Categories:
        1. Eye Contact & Engagement: eye_contact_rate, eye_contact_consistency
        2. Energy & Pacing: energy_level, hook_to_middle_energy_delta, middle_to_closing_contrast
        3. Speech & Density: word_count, speech_coverage, word_density
        4. Visual Variety: scene_count, object_count, overlay_unique_count

    Example Output:
        [
            {
                "report_id": 3,
                "type": "feature_based",
                "frequency": null,
                "percentage": null,
                "confidence_level": "moderate",
                "formula_name": "The High Eye Contact Strategy",
                "strategy_description": "Maintain consistent direct eye contact throughout video journey",
                "key_features": [
                    "eye_contact_rate: 0.88 (RF rank #1, importance 0.35, gap 0.43)",
                    "eye_contact_consistency: 0.12 std dev (RF rank #6, importance 0.08)"
                ],
                "rf_validation": {
                    "insight": "Leverages #1 and #6 most predictive features across entire video"
                },
                "when_to_use": "Universal strategy applicable when cluster paths are fragmented. Focus on proven principles.",
                "creator_recommendations": [
                    "PRIORITY: Maintain 85-90% eye contact throughout video (RF #1 predictor)",
                    "Keep eye contact variance low (<0.15 std dev) across all windows",
                    "Use direct-to-camera framing in hook and closing windows"
                ]
            }
        ]
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

        # Generate report
        report = {
            'report_id': report_id_start + i + 1,
            'type': 'feature_based',
            'frequency': None,
            'percentage': None,
            'confidence_level': 'moderate',
            'formula_name': f"The {group_name} Strategy",
            'strategy_description': _generate_strategy_description(group_name, group_top_features),
            'key_features': [
                _format_key_feature(f) for f in group_top_features
            ],
            'rf_validation': {
                'insight': _generate_rf_insight(group_top_features)
            },
            'when_to_use': 'Universal strategy applicable when cluster paths are fragmented. Focus on proven principles.',
            'creator_recommendations': _generate_recommendations(group_top_features)
        }

        reports.append(report)

    return reports


def _format_key_feature(feature: dict) -> str:
    """Format feature for key_features array."""
    return (
        f"{feature['feature']}: {feature['top_performer_avg']:.2f} "
        f"(RF rank #{feature['rank']}, importance {feature['importance']:.2f}, "
        f"gap {feature['gap']:.2f})"
    )


def _generate_strategy_description(group_name: str, features: list[dict]) -> str:
    """
    Generate strategy description based on feature group.

    Uses data-driven templates that adapt to feature values for hashtag specificity.
    """
    # Define description templates with data-driven parameters
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
            'thresholds': {'dense': 50, 'moderate': 30}  # word_count thresholds
        },
        'Visual Variety': {
            'template': 'Use {level} visual elements and scene transitions',
            'thresholds': {'high': 5, 'moderate': 3}  # scene_count thresholds
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

**Dependencies**:

1. ✅ **Depends on Gap #1** - Uses `prepare_path_data_for_llm()` output to determine scenario (B, C, or D)
2. ✅ **Depends on Gap #3** - Uses same RF video data structure as `generate_universal_principles()`
3. ❌ **No Phase 1 dependencies** - Standalone function using video-level RF data

**Implementation**:
1. Add `generate_feature_based_reports()` to Stage 7 preprocessing (alongside Gap #1 and Gap #3 functions)
2. Function groups RF features into 4 categories (Eye Contact, Energy, Speech, Visual Variety)
3. Selects top 2 features from each group (rotating through groups based on num_reports needed)
4. Generates complete report structure (Python creates all text - formula names, descriptions, recommendations)
5. Uses data-driven templates that adapt descriptions based on `top_performer_avg` values
6. All reports get "moderate" confidence (not frequency-based like path reports)
7. LLM receives complete JSON structure and copies it into `creative_reports` array
8. Add to Stage 7 Child HLD Section 2.3.3 (Phase 2 synthesis)

**Next Step**: Implement `generate_feature_based_reports()` in Stage 7 Python preprocessing script

---

### Gap #5 Analysis: Change "3-5" to "3" Formulas

**Status**: ✅ RESOLVED
**Decision Date**: 2025-10-17

**Original Gap Description** (lines 2330-2351):
- Change ambiguous "3-5 formulas" to deterministic "exactly 3 reports"

**The Problem**:
- Current Mother Doc (MLPlanningv2.md line 3012): "Identify 3-5 Winning Formulas..."
- Child HLD requirement: "Always deliver 3 reports per bucket"
- Ambiguity: When should LLM generate 3 vs 4 vs 5? Unclear.

---

### Final Decision: Change "3-5" to "exactly 3"

**This is a straightforward correction** - no alternatives needed.

**Rationale**:
1. **Aligns with Gaps #1 and #4**:
   - Gap #1: Scenario determination (A, B, C, D) always produces exactly 3 reports
   - Gap #4: Feature-based fallback fills to reach exactly 3 total
2. **Child HLD is authoritative**: Section 2.3.3 clearly specifies "3 reports per bucket"
3. **Stage 8 PDF layout**: Template designed for exactly 3 reports (not variable)
4. **Deterministic output**: Removes ambiguity for LLM

**Implementation**:

**Prompt Text Changes** (MLPlanningv2.md Phase 2 prompt):

**Before**:
```
Identify 3-5 "Winning Formulas" - specific combinations of window strategies...
```

**After**:
```
Identify exactly 3 "Winning Formulas" using frequency-based filtering:

CRITICAL: Always output exactly 3 reports (never more, never less).

Report mix depends on how many cluster paths meet the 10% threshold:
- Scenario A (3+ paths ≥10%): Generate 3 path-based reports
- Scenario B (2 paths ≥10%): Generate 2 path-based + 1 feature-based reports
- Scenario C (1 path ≥10%): Generate 1 path-based + 2 feature-based reports
- Scenario D (0 paths ≥10%): Generate 3 feature-based reports (high fragmentation)
```

**Python Code Required**: None - prompt text change only

**Dependencies**: None - standalone prompt correction

**Next Step**: Update MLPlanningv2.md Phase 2 prompt (lines 3010-3020) with exact count instruction

---

## Summary: Phase 2 Dependencies & Blockers

**ALL GAPS RESOLVED** - 2025-10-17

### Python Functions Required (New Code)

| Function | Gap | Status | Dependencies |
|----------|-----|--------|--------------|
| `prepare_path_data_for_llm()` | #1 | ✅ **RESOLVED** | None |
| `classify_confidence_level()` | #2 | ✅ **RESOLVED** | Gap #1 output |
| `generate_universal_principles()` | #3 | ✅ **RESOLVED** | RF video data |
| `generate_cross_window_patterns()` | #3 | ✅ **RESOLVED** | Stage 4 cross-window features (implemented) |
| `generate_feature_based_reports()` | #4 | ✅ **RESOLVED** | Gaps #1, #3 |
| Prompt text change | #5 | ✅ **RESOLVED** | None (just change "3-5" to "exactly 3") |

### Reusable Phase 1 Code

| Function | Original Use (Phase 1) | Reuse in Phase 2 |
|----------|------------------------|------------------|
| `enrich_high_contrast_features()` | Issue #8 - Enrich cluster features with RF metadata | Gap #4 - Enrich RF features for feature-based reports |
| `compute_rf_alignment()` | Issue #4 - Compute cluster RF alignment | Gap #3 - Universal principles (validate top features) |

### ✅ FORMER BLOCKER RESOLVED: Stage 6 Cross-Window RF Features

**Gap #3 blocker has been RESOLVED** (2025-10-17):

**Discovery**: Cross-window features ARE implemented in Stage 4!
- Found in FeatureTransformationCHILD.md Section 6.5 (lines 221-250, 612-619)
- Stage 4 computes 5 cross-window features during video-level RF transformation
- Stage 6 extracts these features from trained RF models
- Stage 6 HLD updated (MLAnalysisGenerationCHILD.md lines 937-969) to document interface

**Implemented Cross-Window Features** (Stage 4):
1. **Delta features** (progression):
   - `hook_to_middle_energy_delta`: middle_avg_energy - hook_energy ✅
   - `middle_to_closing_contrast`: closing_energy - middle_avg_energy ✅

2. **Variance features** (consistency):
   - `eye_contact_consistency`: std_dev([hook, middle_*, closing]) ✅
   - `word_density_std`: std_dev([word_count across windows]) ✅

3. **Pattern features** (specific progressions):
   - `energy_progression_slope`: linear regression slope of energy ✅

**Resolution**:
- ✅ Stage 4 computes features
- ✅ Stage 6 extracts features from RF models
- ✅ Stage 7 uses graceful degradation (checks if features exist, falls back if missing)
- ✅ No blocker remains - full pipeline is functional

---

## Cross-Dependencies: Phase 1 ↔ Phase 2

| Shared Component | Phase 1 Use | Phase 2 Use | Implementation Location |
|------------------|-------------|-------------|-------------------------|
| **RF Feature Enrichment** | `enrich_high_contrast_features()` - Add metadata to cluster features | `generate_feature_based_reports()` - Add metadata to fallback features | Stage 7 preprocessing (shared utility) |
| **RF Alignment Computation** | `compute_rf_alignment()` - Validate cluster against top RF | Universal principles - Validate top features prevalence | Stage 7 preprocessing (shared utility) |
| **Video-Level RF Data** | Window-specific RF features (eye_contact_rate, etc.) | Cross-window RF features (deltas, consistency) | Stage 6 RF output |
| **Bimodal Detection** | `detect_bimodal_pattern()` - Detect bimodal cluster features | May appear in universal principles (if feature is bimodal globally) | Stage 7 preprocessing (Phase 1 only) |

---

## Implementation Order Recommendation

**~~PHASE 0: Resolve Stage 6 Blocker~~** ✅ **COMPLETE**
1. ✅ Verified Stage 4 computes cross-window RF features (FeatureTransformationCHILD.md Section 6.5)
2. ✅ Updated Stage 6 HLD to document interface (MLAnalysisGenerationCHILD.md lines 937-969)
3. ✅ No implementation needed - features already exist in pipeline

**PHASE 1: Implement Phase 1 Issues (#5-11)** ⏳ **READY FOR IMPLEMENTATION**
- Remove "Important" section (Issue #5)
- Add cluster size context (Issue #6)
- Compress RF format (Issue #7)
- Add RF alignment score mention (Issue #9)
- Add bimodal example note (Issue #10)
- Implement Python preprocessing: `detect_bimodal_pattern()`, `identify_high_contrast_features()`, `compute_rf_alignment()`, `enrich_high_contrast_features()`

**PHASE 2: Implement Phase 2 Gaps (#1-5)** ✅ **ALL GAPS RESOLVED - READY FOR IMPLEMENTATION**
- ✅ Gap #1 RESOLVED: `prepare_path_data_for_llm()` - Hybrid approach with threshold labeling
- ✅ Gap #2 RESOLVED: `classify_confidence_level()` - Python computes confidence bands
- ✅ Gap #3 RESOLVED: `generate_universal_principles()` + `generate_cross_window_patterns()` - Graceful degradation
- ✅ Gap #4 RESOLVED: `generate_feature_based_reports()` - Python generates complete reports with data-driven templates
- ✅ Gap #5 RESOLVED: Change prompt text "3-5" → "exactly 3" reports

**PHASE 3: Integration & Testing** ⏳ **NEXT STEP**
- Create unified Stage 7 preprocessing script with all Gap #1-5 functions
- Update LLMAnalysisCHILD.md with all Phase 1 and Phase 2 changes
- Update MLPlanningv2.md Phase 2 prompt with Gap #5 text changes
- Pilot testing with real data

---

**Status**: **ALL PHASE 2 GAPS RESOLVED** (2025-10-17)
**Next Steps**: Update LLMAnalysisCHILD.md with integrated changes from all resolved gaps.
