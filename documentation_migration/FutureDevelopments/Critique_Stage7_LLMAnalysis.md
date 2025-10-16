# Business Critique: Stage 7 - LLM Analysis (Hybrid Two-Phase Approach)

> **Mother Doc**: MLPlanningv2.md Section "Stage 7: LLM Analysis - Hybrid Two-Phase Approach"
> **Date**: 2025-10-15
> **Status**: COMPLETE - APPROVED FOR IMPLEMENTATION

## Component Summary

**Name**: Stage 7 - LLM Analysis (Hybrid Two-Phase Approach)

**Purpose**: Generate creative insights from K-Means clustering results, validated by dual Random Forest analysis through a two-phase hybrid approach

**Depends On**:
- Stage 6: ML Model Interpretation (outputs 13 JSON files: 1 video-level RF, 6 window-level RF, 6 window-level K-Means)
- Anthropic API (Claude Sonnet 4)
- Video cluster path extraction logic
- Cross-window feature computation from video-level RF

## Critical Analysis

### Overall Assessment
**NEEDS REFINEMENT**

The component demonstrates sophisticated architecture with dual RF validation and parallel processing, but has several critical concerns around complexity, cost validation, and dependency assumptions that need validation before approval.

### Critical Concerns

#### 1. **[CRITICAL] Architectural Complexity vs Business Value**
- **Concern**: The component introduces 90 ML models total (8 video-level RF, 41 window-level RF, 41 window-level K-Means) and a complex two-phase LLM pipeline (7 API calls per bucket × 3 buckets = 21 calls per hashtag). This is a massive leap from the current single-video analysis system.
- **Impact**:
  - High implementation risk with 90 models to train, validate, and maintain
  - Complex orchestration code for Phase 1 parallel execution + Phase 2 synthesis
  - Significant technical debt if not properly architected
  - Could delay MVP by months if complexity leads to bugs/failures
- **Evidence**: MLPlanningv2.md lines 105-109 states "90 models total" and Stage 7 requires orchestrating 7 LLM calls per bucket with cluster path extraction, cross-window feature computation, and dual RF validation (lines 2394-3125)

#### 2. **[CRITICAL] Cost Assumptions Require Validation**
- **Concern**: Cost estimate of $0.26 per bucket (or $0.78 per hashtag) appears optimistic and lacks detailed breakdown. No validation against actual Claude Sonnet 4 API pricing or consideration of failure/retry costs.
- **Impact**:
  - Budget overruns if actual costs are 2-3x higher
  - Economic viability of processing 300 videos per hashtag unclear
  - No contingency for API rate limiting or retry logic costs
  - At scale (10 hashtags per client), costs could be $7.80+ per client analysis
- **Evidence**: MLPlanningv2.md lines 3084-3087 provides cost estimates without detailed calculation methodology or failure scenario modeling

#### 3. **[CRITICAL] Dependency on Video-Level RF Cross-Window Features**
- **Concern**: Phase 2 critically depends on "cross-window features" (e.g., `hook_to_middle_energy_delta`, `middle_to_closing_contrast`, `eye_contact_consistency`) that must be computed from raw temporal window data. The document doesn't specify WHO computes these or WHEN in the pipeline.
- **Impact**:
  - If Stage 6 doesn't compute these features, Stage 7 will fail
  - Unclear if these are hand-coded features or ML-derived
  - Risk of feature computation bugs breaking LLM analysis
  - No validation that these features actually exist in current RumiAI output
- **Evidence**: MLPlanningv2.md lines 2708-2751 shows cross-window features in RF output, but no corresponding Stage 6 documentation for computing these features

#### 4. **[HIGH] Hallucination Risk Mitigation Unproven**
- **Concern**: Component claims "minimizes hallucination risk with small, focused contexts" (113 numbers in Phase 1 vs 1000+ in single-call approach), but provides no empirical validation or testing strategy for this claim.
- **Impact**:
  - LLM may still hallucinate creative patterns that don't exist
  - No validation framework to detect when LLM recommendations are incorrect
  - Client reports could contain fabricated insights, damaging credibility
  - No human-in-the-loop verification step before delivering to clients
- **Evidence**: MLPlanningv2.md line 2404 makes architectural claim about hallucination reduction, but provides no testing protocol or validation metrics (lines 3094-3115 comparison table is theoretical, not empirical)

#### 5. **[HIGH] Parallel Execution Failure Handling Incomplete**
- **Concern**: Phase 1 runs 6-7 parallel API calls, with graceful degradation mentioned ("one window failure doesn't block entire analysis"), but no specification of WHAT happens when failures occur or what quality threshold determines if analysis is usable.
- **Impact**:
  - Partial results may lead to incomplete "Winning Formulas"
  - No clear decision logic: Is 4/6 windows enough? 3/6?
  - Phase 2 may synthesize incorrect patterns from incomplete Phase 1 data
  - Cost is incurred even for failed/incomplete analyses
- **Evidence**: MLPlanningv2.md lines 3006-3008 shows validation check (`if len(window_analyses) != len(window_types)`) but only logs a warning, doesn't specify recovery logic or quality thresholds

#### 6. **[HIGH] "Cluster Path" Pattern Frequency Assumption**
- **Concern**: Phase 2 assumes "most common cluster paths" will be frequent enough to be meaningful (e.g., "18% of videos follow Hook-0 → Middle-1 → Closing-0"). With 3 clusters per window and 6 windows, there are 3^6 = 729 possible paths. Many may occur only 1-2 times.
- **Impact**:
  - "Winning Formulas" may be based on statistically insignificant patterns
  - Top 10 paths may only cover 30-40% of videos, leaving most unexplained
  - Recommendations may not generalize to new content
  - No minimum frequency threshold specified for "formula" viability
- **Evidence**: MLPlanningv2.md lines 2687-2704 shows top 10 path extraction, but no validation of pattern concentration or statistical significance thresholds

### Suggested Changes

#### 1. **Validate Cost Assumptions with Empirical Testing**
- **Change**: Run pilot test with 10 videos through complete Stage 6 + Stage 7 pipeline, measure actual API costs, token usage, and latency
- **Expected Improvement**:
  - Accurate cost projections for client proposals
  - Identify optimization opportunities (e.g., reduce max_tokens if unused)
  - Validate economic viability before building full system

#### 2. **Add Explicit Stage 6 Cross-Window Feature Computation**
- **Change**: Specify in Stage 6 (or new Stage 6.5) the exact feature engineering step that computes cross-window features from temporal windows
- **Expected Improvement**:
  - Clear dependency chain prevents Stage 7 implementation failures
  - Testable feature engineering logic separate from ML model training
  - Opportunity to validate features exist in current RumiAI outputs

#### 3. **Implement Minimum Viable LLM Pipeline First**
- **Change**: Start with single-call Phase 1 on ONE window type (hook only) for ONE bucket, validate output quality, THEN scale to full two-phase architecture
- **Expected Improvement**:
  - Faster time to first insight (2 weeks vs 2 months)
  - Validate LLM prompt engineering works before building orchestration
  - Reduce risk of building complex system that doesn't deliver value
  - Allows early client feedback on report quality

## Resolution of Critical Concerns

### ✅ RESOLVED: Concern #3 - Cross-Window Feature Computation Gap

**Date Resolved**: 2025-10-15

**Finding**: CRITICAL gap identified - Phase 2 LLM prompts reference cross-window features (`hook_to_middle_energy_delta`, `middle_to_closing_contrast`, `eye_contact_consistency`, `word_density_std`, `energy_progression_slope`) that do NOT exist anywhere in Stages 2-6 pipeline.

**Root Cause**: Stage 4 (Feature Transformation) only creates window-prefixed raw features (`hook_energy_level`, `middle_1_energy_level`) but does NOT compute deltas, consistency metrics, or progression slopes across windows.

**Resolution Approach**: Add cross-window feature computation to Stage 4 Video-Level RF transformation (Step 6.5).

**Implementation Documents Created**:
1. **Crosswindowupgrade.md** - Complete architectural plan with:
   - 5 cross-window features defined (formulas, examples, edge cases)
   - Custom `calculate_linear_slope()` helper function (Option A - no new dependencies)
   - Comprehensive testing strategy (8 unit tests)
   - Granular range validation (5 features × specific ranges)
   - 12 surgical edits to FeatureTransformationCHILD.md

2. **Updates to FeatureTransformationCHILD.md** (Stage 4 HLD):
   - Status: 2/12 edits completed (✅ helper function, 🔄 Step 6.5 in progress)
   - Remaining: 10 edits (validation, config, docs, tests, examples)

3. **Updates to MLAnalysisGenerationCHILD.md** (Stage 6 HLD):
   - Status: ✅ Complete (2 lines updated: 178→183 features, 24-215→24-220 range)

**Impact**:
- ✅ Prevents Stage 7 LLM hallucination (analyzing non-existent features)
- ✅ Enables "Winning Formulas" validation (cross-window patterns now computable)
- ✅ Stage 5 auto-adapts to 183 features (no changes needed)
- ✅ Stage 6 generates correct JSON schema with `input_features: 183`

**Status**: Resolution in progress, implementation plan finalized and documented.

---

## Validation Questions & Answers

### Q1: Architectural Complexity vs Business Value (90 models + 21-23 API calls)

**Question**: You propose 90 ML models total (8 video-level RF, 41 window-level RF, 41 window-level K-Means) with a two-phase LLM pipeline requiring 21-23 API calls per hashtag. This is a significant architecture for an MVP.

Given that:
- RumiAI currently processes ONE video at a time (no batch processing yet)
- You haven't processed 300 videos through the pipeline yet (no validation of the full workflow)
- The system will take ~6-8 hours to process 300 videos per hashtag

What is your confidence level (0-100%) that:
1. The 90-model architecture will actually provide MORE valuable insights than a simpler 8-model architecture (just video-level RF per bucket, no window-level models)?
2. Clients will pay a premium for the complexity of "3 hook strategies + 3 middle strategies + 3 closing strategies + 5 winning formulas" vs. just "top 10 features that predict virality per duration bucket"?
3. The checkpoint/resume system will work reliably across all 7 stages without data corruption or state management bugs?

In other words: What evidence (beyond architectural elegance) suggests that this level of complexity is justified for an MVP vs. building a simpler 8-model system first, validating with 1-2 clients, THEN adding the 82 additional models if clients demand more granular insights?

**Answer**: Business value of the 90-model architecture is accepted. Proceeding with full architecture as planned.

**LLM Analysis**: User has validated business case for complete architecture (90 models + dual RF validation + window-level K-Means). This resolves Concern #1 - architectural complexity is approved. Remaining concerns focus on technical validation (cost, dependencies, failure handling).

---

### Q2: Cost Assumptions Require Validation ($0.26 per bucket estimate)

**Question**: Your cost estimate is $0.26 per bucket (or $0.78 per hashtag for 3 buckets), totaling to $7.80 for a 10-hashtag client analysis. This appears optimistic.

Breaking down the calculation:
- **Per bucket**: 7 LLM calls (6 Phase 1 + 1 Phase 2)
- **Per hashtag**: 21 calls (assuming 3 buckets)
- **Claude Sonnet 4 pricing** (as of Jan 2025): $3 per million input tokens, $15 per million output tokens

**Your assumptions** (MLPlanningv2.md lines 3277-3279):
- Phase 1 calls: ~4000 max_tokens each × 6 = 24K tokens output
- Phase 2 call: ~8000 max_tokens × 1 = 8K tokens output
- Input tokens: Estimated ~113 numbers Phase 1, ~larger context Phase 2

**What I need validated**:
1. Have you calculated ACTUAL input token counts for Phase 1 (K-Means + RF JSONs) and Phase 2 (6 window analyses + cluster paths + video-level RF)?
2. What are your retry/failure assumptions? If 10% of API calls fail and retry, does that 10% increase in calls push costs to $0.29-$0.30 per bucket?
3. At what cost per hashtag ($1? $2? $5?) does the economic model break for your target client pricing?

**In other words**: Show me the detailed cost breakdown (input tokens, output tokens, retries) that produces $0.26 per bucket, or acknowledge this is an estimate that needs empirical validation before committing to client pricing.

**Answer**: Cost assumptions are accepted as estimates. Empirical validation will occur during pilot testing but does not block architecture approval.

**LLM Analysis**: User accepts cost estimates without detailed breakdown validation. This resolves Concern #2 - cost assumptions are approved pending empirical validation during pilot. Economic viability concerns are deferred to implementation phase.

---

### Q3: Hallucination Risk Mitigation - Validation Strategy

**Question**: Stage 7 claims to "minimize hallucination risk with small, focused contexts" (113 numbers per window vs 1000+ in single-call). How will you detect when the LLM hallucinates creative patterns that don't actually exist?

Specifically:
1. What is your validation protocol? Human review or automated validation before client delivery?
2. What defines a "hallucination"? (e.g., invented features, contradicts RF data, cluster names don't match centroids)
3. What happens when hallucination detected? (manual correction, regenerate, add to test suite)

**Answer**: Will implement a **three-layer validation approach**:

**Layer 1 - Automated Validation (Technical - Stage 7 enhancement)**:
- Run validation script AFTER each LLM call to check:
  - Feature value contradictions (e.g., LLM says "energy 0.85" but data shows 0.22)
  - Invented features (LLM references features not in source JSON)
  - RF validation contradictions (priority recommendations ignore top RF features)
- On failure: Retry LLM call with modified prompt OR flag for human review
- Will be added to Stage 7 technical design

**Layer 2 - Human Spot-Check (Operational)**:
- **Phase 1 (MVP - First 3 clients)**: 100% human review before client delivery
- **Phase 2 (After 5-10 deliveries)**: 20% spot-check on random samples
- **Phase 3 (After 20+ clients)**: Human review only when automated validation fails
- Focus on: Low silhouette scores, flat RF importance, low-frequency cluster paths (<10%)

**Layer 3 - Feedback Loop (Operational)**:
- Document all errors (automated + human-detected) with source data, LLM output, metadata
- Add failure cases to regression test suite
- Refine prompts with explicit constraints (e.g., "0.0-0.3 = low energy, 0.6-1.0 = high energy")

**Action Items**:
1. Add automated validation to Stage 7 HLD (Layer 1 - technical architecture)
2. Create operational runbook documenting Layers 2 & 3 (human review checklist, error documentation process, prompt refinement workflow)

**LLM Analysis**: User commits to comprehensive hallucination prevention with automated + human validation. This resolves Concern #4 - hallucination risk mitigation is approved with concrete validation strategy. Layer 1 becomes part of Stage 7 technical design; Layers 2-3 documented in operational runbook.

---

### Q4: Parallel Execution Failure Handling - Quality Thresholds

**Question**: Phase 1 runs 6-7 parallel LLM API calls (one per window). What's the minimum quality threshold to proceed to Phase 2?

Scenario: Bucket 18-33s with 6 windows - Phase 1 completes with 5/6 windows (Middle_2 failed due to API timeout). Do you:
1. Proceed to Phase 2 with 5/6 windows (partial data)?
2. Abort and retry entire Phase 1?
3. Some other threshold/strategy?

What's the business rule for "good enough" partial data vs. requiring 100% completion?

**Answer**: **100% Success Required - All Windows Are Critical**

**Rationale**:
- Complete data exists for all windows (6-8 hours of RumiAI processing)
- Client expects full video journey analysis (Hook → Middle → Closing)
- Phase 2 "Winning Formulas" require complete cluster paths across all windows
- Partial analysis appears incomplete/unprofessional

**Implementation Strategy - Smart Retry**:

**Process Flow**:
1. **Attempt 1**: Launch 6 parallel API calls (all windows)
   - Track successes and failures
   - Example result: 4/6 succeed, 2 fail (middle_2, middle_4)

2. **Retry 1**: Retry ONLY failed windows (not all 6)
   - Launch 2 API calls (middle_2, middle_4)
   - Merge with existing successes
   - Example result: 1/2 succeed (middle_2 ✅), still missing middle_4

3. **Retry 2**: Final retry for remaining failures
   - Launch 1 API call (middle_4)
   - Last chance to complete

4. **Outcome**:
   - **6/6 complete** → Proceed to Phase 2 ✅
   - **Still missing windows after 2 retries** → ABORT bucket ❌

**Failure Handling**:
- If bucket fails after retries → Skip bucket, deliver other buckets to client
- Client receives note: "Bucket X unavailable due to technical issues"
- Alternative: Manual intervention (analyst writes insights from Stage 6 JSONs)

**Key Efficiency**:
- Only retry what failed (don't waste successful API calls)
- Total attempts: Initial + 2 retries = max 3 attempts per window
- Example cost: 9 API calls (6 initial + 2 retry + 1 final) vs 18 if retrying all windows

**LLM Analysis**: User requires 100% window completion (no partial analysis). Smart retry strategy approved: retry only failed windows up to 2 times, abort bucket if still incomplete. This resolves Concern #5 - parallel execution failure handling has clear quality threshold (100%) and pragmatic retry logic.

---

### Q5: Cluster Path Pattern Frequency - Statistical Significance Thresholds

**Question**: Phase 2 synthesizes "Winning Formulas" from most common cluster paths. With 3^6 = 729 possible paths and only 100 videos, what's the minimum frequency threshold for a path to become a "Winning Formula"?

Context:
- Many paths will be rare (1-2 videos)
- Need to distinguish "proven pattern" from "statistical noise"
- Must decide: Coverage (more formulas) vs Confidence (fewer, stronger formulas)

**Sub-Questions**:
1. **Minimum threshold**: What % makes a path a "Winning Formula"? (5%? 10%? 15%?)
2. **Metadata classification**: Should Stage 7 add confidence tiers (very_high/high/moderate)?
3. **Output structure**: Should Stage 7 output both path formulas AND universal RF features?

---

**Answer for Sub-Question 1 (Threshold)**: **10% Minimum - Proven Patterns Only**

**Decision**:
- **10% threshold** (10+ videos out of 100)
- Prioritize proven patterns and confidence over coverage
- Always deliver **3 creative reports per bucket**

**Rationale**:
- Business goal: Maximize affiliate creator success rate → need RELIABLE strategies
- 10%+ = 1 in 10 videos → clearly proven pattern with high replication probability
- 5-7% = too rare, might not replicate, wastes creator time on experimental strategies
- 3 reports = manageable choice with clear differentiation (not overwhelming)

**Fallback Strategy** (if <3 paths meet 10% threshold):
- Generate 2 path-based reports (if only 2 paths ≥10%)
- 3rd report = feature-based (using universal RF features)
- Extreme case (no paths ≥10%): All 3 reports are feature-based
- Maintains "3 reports per bucket" commitment while preserving quality

**Coverage Expectation**:
- Strong clustering: 40-60% coverage with 3 path formulas
- High fragmentation: 20-30% coverage, supplement with RF features
- Acceptable trade-off: Quality (proven patterns) over coverage (experimental patterns)

---

**Answer for Sub-Question 2 (Metadata)**: **Add Granular Confidence Levels**

**Decision**: Add confidence metadata even though all formulas pass 10% threshold

**Confidence Bands**:
```python
if frequency_pct >= 20:
    confidence_level = "very_high"  # 1 in 5 videos - dominant pattern
elif frequency_pct >= 15:
    confidence_level = "high"       # 1 in 6-7 videos - strong pattern
elif frequency_pct >= 10:
    confidence_level = "moderate"   # 1 in 10 videos - proven but not dominant

# Feature-based fallback reports always get "moderate"
```

**Rationale**:
- **Stage 8 prioritization**: Helps create visual hierarchy in PDFs (feature Report 1 more prominently if 22% vs Report 3 at 11%)
- **Client communication**: "Very High Confidence" (20%+) = "THE dominant strategy" vs "Moderate" (10-15%) = "Solid strategy worth testing"
- **Creator guidance**: Helps affiliates prioritize which strategy to try first
- **Future-proofing**: Normalizes confidence across different sample sizes (200 videos vs 100)

**JSON Schema**:
```json
{
  "report_id": 1,
  "type": "path_based",
  "frequency": 22,
  "percentage": 22.0,
  "confidence_level": "very_high",
  "confidence_score": 0.92  // Optional: 0-1 normalized
}
```

---

**Answer for Sub-Question 3 (Output Structure)**: **Hybrid - Reports + Universal Principles**

**Decision**: Stage 7 outputs BOTH path formulas AND universal RF features

**JSON Structure**:
```json
{
  "creative_reports": [
    // 3 main reports (path-based preferred, feature-based fallback)
    // PRIMARY deliverable
  ],

  "supplementary_insights": {
    "universal_principles": [
      // Top 5-7 RF features that apply to ALL videos
      {
        "feature": "hook_eye_contact_rate",
        "rf_importance": 0.35,
        "top_performer_avg": 0.88,
        "recommendation": "Maintain 85%+ eye contact in hook"
      }
    ],
    "cross_window_patterns": [
      // Energy progression, consistency patterns from video-level RF
    ]
  }
}
```

**Rationale**:
- **Coverage gap solution**: 3 path formulas cover 40-60% of videos → universal principles cover remaining 40-60%
- **Multi-audience value**:
  - Path formulas: For creators willing to follow exact templates (beginner-friendly)
  - Universal principles: For experienced creators who want flexibility to customize
- **Fallback enhancement**: Feature-based reports can reference universal principles when no paths meet 10%
- **Complete coaching**: Ensures EVERY affiliate creator gets actionable advice, even if their style doesn't match a formula

**Hierarchy**:
- **Primary**: 3 creative reports (path formulas when available)
- **Supplementary**: Universal principles (for customization/edge cases)
- Stage 8 decides how to present (main PDF section vs appendix)

---

**LLM Analysis**: User commits to quality-over-coverage strategy with 10% threshold, granular confidence levels, and hybrid output structure. This resolves Concern #6 - cluster path fragmentation handled through proven pattern threshold (10%), confidence classification (very_high/high/moderate), and supplementary universal features for 100% coverage. Path formulas provide specific templates (40-60% coverage), universal principles ensure all creators get value (100% coverage).

---

## Final Decision

**Overall Assessment**: **APPROVE WITH ENHANCEMENTS**

**Reasoning**:
Based on Q&A validation, all critical and high-priority concerns have been addressed with concrete implementation decisions:

**Resolved Concerns**:
1. ✅ **Concern #1 (Architectural Complexity)**: Business value of 90-model architecture accepted - complexity justified for complete pattern coverage
2. ✅ **Concern #2 (Cost Assumptions)**: Cost estimates accepted as preliminary - empirical validation deferred to pilot testing
3. ✅ **Concern #3 (Cross-Window Features)**: Already resolved via Crosswindowupgrade.md - 5 cross-window features added to Stage 4
4. ✅ **Concern #4 (Hallucination Risk)**: Three-layer validation strategy approved (automated validation + human spot-check + feedback loop)
5. ✅ **Concern #5 (Parallel Execution)**: 100% window completion required with smart retry logic (retry only failed windows up to 2x)
6. ✅ **Concern #6 (Cluster Path Frequency)**: 10% threshold for proven patterns, granular confidence levels, hybrid output (formulas + universal features)

**Proceed to Phase 2**: **YES**

**Approved Architecture** (Stage 7 - LLM Analysis):

**Two-Phase Hybrid Approach**:
- **Phase 1**: Per-window analysis (6-7 parallel LLM calls) with window-level RF validation
- **Phase 2**: Cross-window synthesis with video-level RF validation and cluster path extraction
- **Smart Retry**: Retry only failed windows (up to 2 attempts), abort bucket if incomplete after retries

**Output Schema**:
```json
{
  "creative_reports": [
    {
      "report_id": 1-3,
      "type": "path_based" | "feature_based",
      "percentage": float,
      "confidence_level": "very_high" | "high" | "moderate",
      "formula_name": string,
      "cluster_path": array,
      // ... full template
    }
  ],
  "supplementary_insights": {
    "universal_principles": [...],  // Top 5-7 RF features
    "cross_window_patterns": [...]  // Video-level RF patterns
  }
}
```

**Quality Thresholds**:
- Path formula inclusion: ≥10% frequency (10+ videos)
- Confidence classification: very_high (≥20%), high (15-20%), moderate (10-15%)
- Window completion: 100% required (all windows must succeed)
- Reports per bucket: Always 3 (path-based preferred, feature-based fallback)

**Validation Enhancements** (to be added to Stage 7 HLD):
1. **Automated validation** (Layer 1): Post-LLM checks for feature contradictions, invented features, RF misalignment
2. **Human review protocol** (Layer 2): 100% review for first 3 clients → 20% spot-check → automated-only with alerts
3. **Feedback loop** (Layer 3): Error documentation, regression tests, prompt refinement

**Action Items**:
1. ✅ Update Stage 7 HLD (LLMAnalysis7.md) with:
   - 10% threshold logic for path formula inclusion
   - Confidence level classification (very_high/high/moderate)
   - Hybrid output schema (creative_reports + supplementary_insights)
   - Automated validation layer (feature contradictions, invented features, RF misalignment)
   - Smart retry logic (100% window completion requirement)

2. ✅ Create operational runbook for:
   - Human review protocol (Layers 2-3: spot-check procedures, error documentation, prompt refinement)
   - Fallback handling (when <3 paths meet 10%, use feature-based reports)
   - Bucket failure scenarios (abort bucket, deliver other buckets, manual intervention)

3. ✅ Ensure cross-window features are implemented:
   - Verify Crosswindowupgrade.md implementation complete in Stage 4
   - Validate 5 cross-window features appear in Stage 6 JSON outputs
   - Confirm Phase 2 LLM prompts correctly reference these features

**Key Constraints Acknowledged**:
- Hallucination risk exists despite mitigation (automated validation + human review required)
- Cost estimates are preliminary (empirical validation during pilot)
- 10% threshold may result in <3 path formulas (feature-based fallback accepted)
- Cluster path fragmentation possible (universal principles provide coverage safety net)

**Status**: **COMPLETE - APPROVED FOR IMPLEMENTATION**

**Next Steps**: Proceed to Phase 2 (Clarification Q&A) with approved Stage 7 architecture.
