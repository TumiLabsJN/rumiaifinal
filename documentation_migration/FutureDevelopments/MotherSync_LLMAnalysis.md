# Mother Document Sync: Proposed Changes from LLM Analysis (Stage 7) Work

> **Trigger**: Stage 7 Child HLD work revealed Mother Doc Stage 7 prompt outdated/incomplete
> **Component**: Stage 7 - LLM Analysis (Hybrid Two-Phase Approach)
> **Phase Outputs Reviewed**:
>   - Critique_Stage7_LLMAnalysis.md (Phase 1)
>   - QA_LLMAnalysis.md (Phase 2)
>   - LLMAnalysisCHILD.md (Phase 3)
>   - Stage7PromptCritique.md (Prompt Analysis)
> **Date**: 2025-10-16
> **Status**: PENDING APPROVAL

---

## Summary

**Total Changes Proposed**: 1 major update (Phase 2 prompt revision)
**Impact Scope**: Level 2 (Mother Doc Stage 7 - affects Stage 7 implementation only)

**Affected Docs**:
- MLPlanningv2.md (direct changes to Stage 7 Section lines 3012-3076)
- LLMAnalysisCHILD.md (already aligned with proposed changes - no updates needed)

**Root Cause**: Mother Document Phase 2 prompt (lines 3012-3076) was written before Critique Q5 decisions (10% threshold, confidence levels, hybrid output structure, feature-based fallback). Child HLD incorporates approved Critique decisions but Mother prompt doesn't reflect them.

**Impact if Not Updated**:
- LLM will generate 3-5 formulas without frequency filtering → includes 8% frequency paths (unreliable patterns)
- No confidence levels → Stage 8 cannot prioritize reports in PDF generation
- Missing `supplementary_insights` → only 40-60% of videos covered by path formulas
- No fallback logic → LLM confused when high fragmentation (<3 paths meet 10% threshold)

---

## Proposed Changes

### Change 1: [Category 3 - Outdated Information (Mother)] Update Phase 2 Prompt with Critique Q5 Decisions

**Issue Type**: Outdated Information (Mother)

**Current State**:
- **Mother Section**: MLPlanningv2.md lines 3012-3076 (Stage 7: Phase 2 LLM Prompt)
- **Current Text** (line 3012):
  ```
  Identify 3-5 "Winning Formulas" - specific combinations of window strategies...
  ```
- **Current Output Schema** (lines 3030-3070):
  ```json
  {
    "winning_formulas": [...],  // 3-5 formulas
    "cross_window_insights": [...]
  }
  ```

**Problem Discovered**:
- **By Comparing**: LLMAnalysisCHILD.md Section 5.2.2 (Output Schema) vs MLPlanningv2.md lines 3012-3076
- **Evidence from Child HLD**:
  - **Section 5.2.2** (Output Schema): Defines `creative_reports` array with exactly 3 reports, `confidence_level` field, hybrid structure with `supplementary_insights`
  - **Section 2.3.3** (Detailed Process): Documents 10% threshold filtering, feature-based fallback logic, confidence classification
  - **Appendix A Decision Log**: Decision 3 (lines in Child HLD) documents 10% threshold rationale
- **Evidence from Critique**:
  - **Critique_Stage7_LLMAnalysis.md** Q5 (lines 301-407): User approved 10% threshold, confidence levels, hybrid output, feature-based fallback
  - **Stage7PromptCritique.md**: Documents 5 critical gaps in Mother prompt (10% threshold, confidence levels, hybrid structure, fallback, "3-5" ambiguity)

**Gaps Identified** (from Stage7PromptCritique.md):
1. **Gap #1**: Missing 10% frequency threshold instruction (CRITICAL)
2. **Gap #2**: Missing confidence level classification (CRITICAL)
3. **Gap #3**: Missing hybrid output structure (`creative_reports` + `supplementary_insights`) (HIGH)
4. **Gap #4**: Missing feature-based fallback instructions (HIGH)
5. **Gap #5**: Ambiguous "3-5 formulas" should be "3 reports" (MEDIUM)

**Proposed Update**:

Replace MLPlanningv2.md lines 3012-3076 with the following updated Phase 2 prompt:

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
      "path": [0, 1, 1, 1, 2, 0],  // Only for path_based (null for feature_based)
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

**Rationale**:

**Why This Change is Needed**:
1. **Critique Q5 Approved Decisions**: User explicitly approved 10% threshold, confidence levels, hybrid output during Critique Phase (lines 301-407 of Critique doc)
2. **Child HLD Implementation Ready**: LLMAnalysisCHILD.md Section 5.2.2 defines complete output schema matching this prompt
3. **Quality Assurance**: Without threshold filtering, LLM includes unreliable 8% frequency patterns
4. **Stage 8 Dependency**: PDF generation (Stage 8) requires confidence levels for report prioritization
5. **Coverage Completeness**: supplementary_insights ensures 100% creator coverage (path formulas only cover 40-60%)
6. **Fallback Logic**: Handles high fragmentation scenarios (0-2 paths above threshold) gracefully

**Comparison - What Changed**:
- **OLD**: "Identify 3-5 Winning Formulas" (ambiguous, no filtering)
- **NEW**: "Generate exactly 3 reports" with 10% threshold filtering + fallback logic
- **OLD**: No confidence_level field
- **NEW**: confidence_level classification (very_high/high/moderate)
- **OLD**: `winning_formulas` array only
- **NEW**: `creative_reports` + `supplementary_insights` (hybrid structure)
- **OLD**: No instructions for <3 paths scenario
- **NEW**: Feature-based fallback for Scenarios B, C, D
- **OLD**: ~65 lines of prompt
- **NEW**: ~200 lines of detailed prompt (3x more specific)

**Impact**: Stage 7 implementation only (no Foundation or other Component Children affected)

**Priority**: **CRITICAL** - Blocks Stage 7 implementation if not updated

---

## Change Summary by Priority

### [CRITICAL] Changes (must apply)
1. **Change 1**: Update Phase 2 prompt (lines 3012-3076) with 10% threshold, confidence levels, hybrid output, feature-based fallback

### [HIGH] Changes (should apply)
*None - all changes bundled into single prompt update*

### [LOW] Changes (optional)
*None*

---

## Recommended Action

**Option A: Apply Change** ⭐ **RECOMMENDED**
- Update MLPlanningv2.md lines 3012-3076 with revised Phase 2 prompt
- No cascade needed (only Stage 7 affected, Child HLD already aligned)
- Estimated effort: 10 minutes (copy-paste replacement)

**Option B: Apply Partial Update**
- Update only 10% threshold + confidence levels (Gaps #1, #2)
- Defer hybrid output structure + fallback logic (Gaps #3, #4)
- NOT RECOMMENDED: Creates incomplete prompt, Stage 8 still can't prioritize reports

**Option C: Reject Change**
- Keep Mother Doc as-is
- Acknowledge Child HLD and Mother Doc diverge on Phase 2 prompt
- Stage 7 implementers use Child HLD as source of truth
- NOT RECOMMENDED: Mother Doc becomes outdated

---

## User Decision

**Selected Option**: [A - Apply Change] ✅

**Changes to Apply**: [Change 1: Phase 2 prompt updated (lines 3012-3260)] ✅

**Status**: APPLIED
**Applied Date**: 2025-10-16

---

## Additional Context

### Related Documents
- **Stage7PromptCritique.md**: Complete analysis of prompt gaps (5 identified)
- **Critique_Stage7_LLMAnalysis.md** Q5: User approval of 10% threshold, confidence levels, hybrid output
- **LLMAnalysisCHILD.md** Section 5.2.2: Output schema matching proposed prompt

### Implementation Notes
After Mother Doc update:
1. ✅ LLMAnalysisCHILD.md already aligned (no changes needed)
2. ✅ No Foundation updates needed (Stage 7-specific change)
3. ✅ No other Component Children affected (Stage 7 standalone)
4. ✅ Ready for Stage 7 implementation immediately after update

### Verification Checklist
After applying changes:
- [✓] MLPlanningv2.md lines 3012-3260 replaced with new prompt
- [✓] New prompt includes 5-step process (STEP 1-5 headers present)
- [✓] Output schema includes `creative_reports` + `supplementary_insights`
- [✓] Output schema includes `confidence_level` field
- [✓] Prompt mentions "exactly 3 reports" (not "3-5")
- [✓] Prompt includes Scenarios A-D for different path counts
- [✓] Prompt includes feature-based report structure and categories

**All Gaps Resolved**:
- ✅ Gap #1: 10% threshold present (lines 3016, 3029, 3035, 3050)
- ✅ Gap #2: confidence_level field present (lines 3068, 3111, 3171)
- ✅ Gap #3: supplementary_insights structure present (lines 3216, 3257)
- ✅ Gap #4: feature_based fallback logic present (lines 3089-3134)
- ✅ Gap #5: "exactly 3 reports" language present (lines 3012, 3052, 3091)

---

**Date Created**: 2025-10-16
**Author**: AI Documentation Architect
**Approval Status**: PENDING USER REVIEW
