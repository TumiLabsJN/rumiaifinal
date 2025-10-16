# Content Analysis - Stage 7 Integration Design

> **Purpose**: Document decisions for how Stage 7 (LLM Report Generation) consumes Content Analysis outputs
> **Parent**: ContentAnalysisCHILD.md
> **Date**: 2025-10-14
> **Status**: IN PROGRESS

---

## Context

ContentAnalysisCHILD.md defines what Content Analysis produces (classification schema). This document addresses **how Stage 7 consumes and integrates** those outputs to generate creative strategy reports.

---

## Integration Questions & Decisions

### Question 1: Performance Group Labeling

**Context**: Stage 7 needs to distinguish top performers from bottom performers for contrastive analysis.

**Current State**: Classification output has no `performance_group` field. Stage 7 must cross-reference `selection_manifest.json` to determine if a video is top or bottom.

**Options**:
- **Option A**: Add `"performance_group": "top"|"bottom"` field to classification output
  - Pro: Self-contained classification, no manifest lookup needed
  - Pro: Simpler Stage 7 logic
  - Con: Redundant data (manifest already has this)
  - Con: Increases classification file size slightly

- **Option B**: Stage 7 cross-references manifest
  - Pro: Single source of truth (manifest)
  - Pro: Smaller classification files
  - Con: Stage 7 must load manifest + classifications
  - Con: More complex data loading logic

**Decision**: **Option A - Add `performance_group` field to classification output**

**Rationale**:
1. **Alignment with "Individual Files" Decision**: We already decided to keep individual classification files (240KB) to preserve expensive LLM output. Adding 1.2KB (0.5% increase) is negligible and consistent with that philosophy.

2. **Classification Self-Sufficiency**: Each classification file becomes a complete, interpretable unit. A developer/QA can inspect `7526250443832331550_content.json` and immediately understand "this is a top performer with problem_solution hook" without cross-referencing external files.

3. **Stage 7 Simplicity**: Stage 7's primary job is report generation, not data wrangling. Removing the manifest cross-reference step reduces complexity in the most critical stage.

4. **Input Source Already Available**: Stage 2.7 (classification stage) already loads `selection_manifest.json` to know which videos to classify. Adding the field requires ~2 lines of code:
   ```python
   performance_group = "top" if video_id in manifest['top_performers'] else "bottom"
   classification['performance_group'] = performance_group
   ```

5. **Practical "Single Source of Truth"**: While manifest is the original source, once Stage 2.7 runs, classifications become the active artifact consumed by Stage 7. They should be self-contained.

6. **Debugging & Validation**: QA can spot-check classifications without loading manifests. Example: "Wait, this video has performance_group: 'bottom' but uses problem_solution hook 60% top frequency pattern - is this an outlier or misclassification?"

**Trade-off Accepted**: 1.2KB redundancy (0.000001% of modern disk space) in exchange for simpler downstream logic and better file interpretability.

**Implementation Note**: Stage 2.7 will add this field during classification generation. Example output:
```json
{
  "video_id": "123456789",
  "performance_group": "top",
  "content_category": "recipe_tutorial",
  "hook_strategy": "problem_solution",
  ...
}
```

---

### Question 2: Field Selection for Aggregation

**Context**: Stage 7 Python code will aggregate 120 classification files before sending stats to LLM. The classification schema has 22 fields (11 core + 11 caption_analysis subfields, excluding video_id and note). We must decide which fields to aggregate.

**Complete Field Inventory**:

**Core Fields (9 aggregatable)**:
- content_category (string - PRIMARY classification)
- hook_strategy (string)
- audience_pain_points (array[string])
- trending_keywords (array[string])
- engagement_drivers (array[string])
- content_tactics (array[string])
- confidence (enum: high/medium/low)
- transcript_available (boolean)
- ~~video_id, note~~ (not aggregated)

**caption_analysis Subfields (12 aggregatable)**:
- caption_hook_type (enum)
- caption_cta_type (enum)
- caption_cta_present (boolean)
- brand_mention_present (boolean)
- influencer_tag_present (boolean)
- emoji_usage (enum)
- caption_length (enum)
- hashtag_count (integer)
- hashtag_placement (enum)
- hashtag_strategy.broad_count (integer)
- hashtag_strategy.niche_count (integer)
- hashtag_strategy.branded_count (integer)

**Options Considered**:
- **Option A**: Core content fields only (7 aggregations) - Minimal, ignores all caption data
- **Option B**: Core + caption essentials (13 aggregations) - Balanced depth + efficiency
- **Option C**: All 21 aggregatable fields (21 aggregations) - Comprehensive, potential bloat

**Decision**: **Option B - Core + Caption Essentials (13 aggregations)**

**Fields to Aggregate**:

1. **Core Content (7 fields)**:
   - `content_category` → Counter (frequency distribution)
   - `hook_strategy` → Counter (frequency distribution)
   - `engagement_drivers` → Counter (flatten arrays, count occurrences)
   - `content_tactics` → Counter (flatten arrays, count occurrences)
   - `audience_pain_points` → Counter (flatten arrays, count occurrences)
   - `trending_keywords` → Counter (flatten arrays, count occurrences)
   - `confidence` → Counter (quality gate distribution)

2. **Caption Strategy (6 fields)**:
   - `caption_cta_type` → Counter (which CTAs work best)
   - `emoji_usage` → Counter (none/light/moderate/heavy distribution)
   - `caption_length` → Counter (short/medium/long distribution)
   - `hashtag_count` → Distribution stats (mean, min, max, std)
   - `hashtag_strategy` → Average broad/niche/branded counts per group
   - `transcript_available` → Ratio (percentage with speech)

**Rationale**:

1. **Actionable Caption Guidance**: Answers "How should I write my caption?"
   - "Use 7 hashtags (5 niche, 2 broad)"
   - "Keep captions short"
   - "Use light emojis"
   - "Include link_in_bio CTA"

2. **ROI on LLM Extraction**: Stage 2.7 costs $0.12 to extract 12 caption_analysis fields. Option B uses the 6 most valuable fields (50% utilization). Option A wastes 100% of caption extraction.

3. **80/20 Rule**: These 13 fields cover 80% of actionable insights:
   - Core 6 taxonomy fields → content strategy ("use problem_solution hook")
   - Caption CTA/emoji/length → formatting guidance ("short captions with link_in_bio")
   - Hashtag count + strategy → optimization ("7 hashtags: 5 niche, 2 broad")
   - Confidence → quality filtering ("exclude low-confidence classifications")

4. **Skipped Fields Have Low Differentiating Value**:
   - `caption_hook_type`: Redundant with `hook_strategy` (video hook > caption hook)
   - `caption_cta_present`: 90%+ have CTAs → not a differentiator
   - `brand_mention_present`, `influencer_tag_present`: Vary by niche/network, not universal advice
   - `hashtag_placement`: "End" dominates 80%+ → not actionable differentiator

5. **Token Efficiency**: ~1.5K tokens (vs 1K for Option A, 2K for Option C) - optimal depth without bloat

6. **Reporting Focus**: 13 dimensions is digestible for LLM. Option C's 21 fields risk overwhelming synthesis, diluting focus on high-impact differentiators.

**Trade-offs Accepted**:
- Missing `caption_hook_type` (caption vs video hook is redundant)
- Missing 3 boolean fields (`caption_cta_present`, `brand_mention_present`, `influencer_tag_present`) - marginal differentiators
- Missing `hashtag_placement` - low variance, not actionable
- Can upgrade to all 21 fields in V2 if customers request more depth

**Implementation Pattern**:
```python
# Stage 7 Python aggregation
def aggregate_classifications(classifications_top, classifications_bottom):
    # Filter by confidence first
    high_conf_top = [c for c in classifications_top if c['confidence'] in ['high', 'medium']]
    high_conf_bottom = [c for c in classifications_bottom if c['confidence'] in ['high', 'medium']]

    stats = {
        # Core content (7 fields)
        'content_category': {
            'top': Counter([c['content_category'] for c in high_conf_top]),
            'bottom': Counter([c['content_category'] for c in high_conf_bottom])
        },
        # ... (6 more core fields)

        # Caption strategy (6 fields)
        'hashtag_count': {
            'top': {'mean': 7.2, 'min': 5, 'max': 9, 'std': 1.3},
            'bottom': {'mean': 14.5, 'min': 2, 'max': 30, 'std': 9.1}
        },
        # ... (5 more caption fields)
    }

    return stats  # ~1.5K tokens to LLM
```

---

### Question 3: Effect Size Calculation Documentation

**Context**: Stage 7 calculates effect sizes to identify differentiating patterns (e.g., "problem_solution hook is 2.5x more common in top performers").

**What is Effect Size?**
```python
# Example calculation
top_frequency = 24 / 40  # 60% of top performers use "problem_solution"
bottom_frequency = 4 / 20  # 20% of bottom performers use "problem_solution"
effect_size = top_frequency / bottom_frequency  # 3.0x

# Report: "problem_solution hook is 3.0x more common in top performers (60% vs 20%)"
```

**Question**: Should we add effect size calculation guidance to ContentAnalysisCHILD.md?

**Options**:
- **Option A**: Add to Section 3.5 "Stage 7 Integration Pattern"
  - Include formula + examples in main body
  - Pro: Emphasizes integration importance
  - Con: Mixes Stage 2.7 concerns (what we produce) with Stage 7 concerns (how they consume)

- **Option B**: Add to Appendix D "Stage 7 Integration Examples"
  - Detailed pseudocode in appendix
  - Pro: Optional reading, keeps main sections lean
  - Con: Still duplicates Stage 7 logic, creates sync issues

- **Option C**: Don't add to ContentAnalysisCHILD.md
  - Document in Stage7_LLMReportGenerationCHILD.md only
  - Pro: Clear separation of concerns, single source of truth
  - Con: Reader must cross-reference to see full picture

**Decision**: **Option C - Don't Add to ContentAnalysisCHILD.md**

**Rationale**:

1. **Separation of Concerns**:
   - ContentAnalysisCHILD.md = "what Content Analysis produces" (classification schema)
   - Stage7_LLMReportGenerationCHILD.md = "how Stage 7 consumes and processes" (aggregation + effect size)
   - Effect size is Stage 7's processing logic, not Content Analysis's output contract

2. **Single Source of Truth**:
   - If effect size formula changes (e.g., add confidence intervals, use Cohen's d), update ONE place (Stage7 docs)
   - Options A & B create maintenance burden (update two docs for same logic)

3. **Precedent from Other Stages**:
   - Stage 3 (ML Feature Extraction) doesn't document how Stage 4 (Random Forest Training) uses features
   - Stage 4 docs explain feature selection, importance calculation, etc.
   - Each stage documents what it does, not what downstream stages do with its output

4. **ContentAnalysisCHILD.md Already References Stage 7**:
   - Section 3.2 Output Contracts: "Enables contrastive analysis ('60% of top use X vs 20% of bottom')"
   - Section 10.3 Related Docs: Links to Stage 7 as downstream consumer
   - Reader knows where to look for integration details

5. **ContentAnalysisCHILDpt2.md Serves as Bridge**:
   - This document specifically addresses integration questions
   - Questions 1-2 document Stage 2.7 → Stage 7 handoff decisions
   - This is the right place for cross-stage concerns, not the main CHILD doc

6. **Keep ContentAnalysisCHILD.md Concise**:
   - Effect size calculation = ~5 lines Python in Stage 7
   - Documenting it in ContentAnalysisCHILD.md adds ~50-100 lines
   - Better to keep CHILD doc lean (930 lines) vs bloated (980+ lines)

**Cross-Reference Strategy**:
- **ContentAnalysisCHILD.md Section 10.3**: Add note "See Stage7_LLMReportGenerationCHILD.md for aggregation patterns and effect size calculations"
- **Stage7CHILD.md Section 3.1** (future): "See ContentAnalysisCHILD.md Section 5.2 for classification schema details"

**Trade-offs Accepted**:
- Reader must cross-reference Stage7CHILD.md for full integration picture
- ContentAnalysisCHILDpt2.md bridges the gap (acceptable interim solution)

---

### Question 4: ML Feature Cross-Reference

**Context**: This question was relevant when considering ML×content correlation for MVP.

**Decision**: **Not Applicable - Deferred to V2**

**Rationale**:
- **MVP Scope**: Separate ML and Content insights (no correlation per Token Budget decision lines 181-212)
- **Token Budget**: 99K tokens with correlation → 12K tokens without (87% reduction)
- **Sample Size**: 120 videos insufficient for reliable correlations
- **Dev Effort**: 4.5 days additional development for marginal MVP benefit

**V2 Documentation**:
When correlation is implemented in V2:
- **Stage2.6correlation.md**: Already documents complete technical design (600 lines)
  - **Section 6.4 NEW**: Documents prerequisite to reverse aggregation decision OR use Option C (Python-computed correlations)
- **Stage7_LLMReportGenerationCHILD.md Section 3.1** (future): Will document loading aggregated_features.csv and joining by video_id
- **ContentAnalysisCHILD.md**: No changes needed (output schema remains unchanged)

**MVP Reality**:
- Stage 7 receives: Aggregated content stats (1.5K tokens) + Aggregated ML stats (10K tokens) = 11.5K tokens (separate sections)
- Report sections: "ML Insights" + "Content Insights" (no cross-reference required)
- No need to document ML feature access in ContentAnalysisCHILD.md for MVP

---

## Additional Decision: Individual vs Aggregated Output Files

**Context**: Since MVP won't include ML × content correlation, should Stage 2.7 output individual classification files or one aggregated summary per bucket?

**Options**:
- **Option A**: 120 individual files (2KB each, 240KB total per hashtag)
- **Option B**: 1 aggregated summary per bucket (5KB, contains only counts/distributions)

**Decision**: **Option A - Keep Individual Files**

**Rationale**:
1. **Preserve expensive data**: Stage 2.7 costs $0.12 in LLM calls - aggregation is free (<1s Python), disaggregation is impossible without re-running
2. **Future flexibility**: Enables V2 correlation, report examples with video_ids, quality validation, client-specific queries
3. **Negligible cost**: 240KB vs 5KB difference is meaningless (disk space: ~$0.000001)
4. **Stage 7 simplicity**: Aggregation in Stage 7 is trivial (5 lines pandas, <1 second)
5. **Debugging capability**: Can inspect individual classifications for validation

**Trade-offs Accepted**: Stage 7 must perform aggregation (negligible effort)

---

## Token Budget & MVP Scope Decision

**Discovery**: Stage 7 LLM input with individual data approaches token limits

**Token Calculation**:
- ML Analysis (with individual video features): ~79,000 tokens
- Content Classifications (120 individual files): ~20,000 tokens
- **Total**: ~99,000 tokens (at 100K recommended limit edge)

**Decision**: **MVP ships with separate ML and Content insights (no correlation)**

**Stage 7 Data Flow Decision**: **Aggregate in Python, Send Stats to LLM**

**Stage 7 Approach**:
- Pre-aggregate ML analysis in Stage 6: Send feature importance + distributions only (~10K tokens)
- **Pre-aggregate content in Stage 7: Load 120 files, Python aggregates to frequency distributions (~2K tokens)**
- **Total**: ~12K tokens (87% reduction)

**Rationale for Python Aggregation**:
1. **Token Economics**: 90% reduction (20K → 2K tokens), saves $0.06 per report
2. **Task Alignment**: Python handles arithmetic (counting, percentages), LLM handles insights (synthesis, recommendations)
3. **Deterministic Accuracy**: Python guarantees correct counts, no LLM miscounting risk
4. **MVP Scope**: Without ML×content correlation, LLM only needs frequency distributions for contrastive analysis
5. **Data Preserved**: Individual files on disk enable V2 features (correlation, outlier detection, video examples)

**What's Lost by Aggregating**:
- ❌ Multi-field co-occurrence patterns (e.g., "X + Y combo appears in 40% top, 0% bottom")
  - **Acceptable**: Explicitly deferred to V2 per Stage2.6correlation.md (needs larger sample, statistical algorithms)
- ❌ Outlier detection (e.g., "this bottom performer uses all top tactics")
  - **Acceptable**: Nice-to-have, not critical for MVP actionable insights
- ❌ Video-specific examples in report
  - **Acceptable**: Creators can't access other creators' videos anyway

**What's Kept by Aggregating**:
- ✅ Frequency distributions: "60% of top use problem_solution vs 20% of bottom"
- ✅ Effect sizes: "problem_solution is 3x more common in top performers"
- ✅ Contrastive comparisons: All "top vs bottom" differentiators
- ✅ Distribution statistics: min/max/mean/std for numeric fields (hashtag_count)
- ✅ Confidence filtering: Filter low-confidence classifications before aggregating

**Upgrade Path**: V2 can send individual files if customers demand deeper pattern discovery

**What Creators Get (MVP)**:
1. **ML Insights**: "Eye contact rate is #1 predictor (0.88 top vs 0.45 bottom)"
2. **Content Insights**: "60% top use problem_solution hook vs 20% bottom"
3. Both are independently actionable

**What's Deferred to V2** (see Stage2.6correlation.md):
- ML × content correlation: "Direct_to_camera drives 1.42x higher eye_contact_rate"
- Mechanistic explanations: Why tactics work scientifically
- Effort: 4.5 days dev, $0.45/hashtag cost increase
- Trigger: Customer demand OR sample size increase to 100+ videos

---

## Summary of Decisions

### Decision 1: Performance Group Labeling (Question 1)
**Decision**: Add `performance_group` field to classification output

**Impact**:
- Classification files become self-contained (no manifest cross-reference needed)
- Stage 7 simplicity (filter by field vs manifest lookup)
- 1.2KB redundancy accepted (0.5% increase)

### Decision 2: Field Selection for Aggregation (Question 2)
**Decision**: Aggregate 13 fields (7 core content + 6 caption strategy)

**Impact**:
- Actionable caption guidance (hashtag count/strategy, emoji, length, CTA)
- 50% utilization of caption_analysis extraction ($0.12 LLM cost)
- ~1.5K tokens for aggregated stats (vs 20K for individual files = 93% reduction)

### Decision 3: Effect Size Calculation Documentation (Question 3)
**Decision**: Don't add to ContentAnalysisCHILD.md (document in Stage7CHILD.md only)

**Impact**:
- Clear separation of concerns (Stage 2.7 = what we produce, Stage 7 = how they consume)
- Single source of truth for effect size logic
- ContentAnalysisCHILD.md stays lean (930 lines vs 980+ with appendix)

### Decision 4: ML Feature Cross-Reference (Question 4)
**Decision**: Not Applicable - Deferred to V2

**Impact**:
- MVP ships without ML×content correlation
- 87% token reduction (99K → 12K)
- V2 implementation documented in Stage2.6correlation.md Section 6.4

### Supporting Decision: Stage 7 Data Flow (Token Budget Section)
**Decision**: Aggregate in Python, Send Stats to LLM

**Impact**:
- 90% token reduction (20K → 2K for content)
- Python handles arithmetic, LLM handles synthesis
- $0.06 savings per report, $6 per 100 reports

---

## Resulting Changes to ContentAnalysisCHILD.md

### Required Updates:

1. **Section 5.2.2 - Video Classification Output Schema**:
   - Add `performance_group` field (line ~720)
   - Type: enum ("top" | "bottom")
   - Description: "Performance group classification (from selection_manifest)"

2. **Section 10.3 - Related Child Docs**:
   - Update Stage 7 bullet to add: "See Stage7_LLMReportGenerationCHILD.md for aggregation patterns and effect size calculations"

3. **No Other Changes Required**:
   - Output schema remains 22 fields (now 23 with performance_group)
   - File structure unchanged (individual files preserved on disk)
   - Stage 2.6/2.7 logic unchanged (classification process same)

### Optional Updates (Can Defer):

- Add brief note in Section 3.2 Output Contracts about Stage 7 aggregation approach
- Add performance_group to example JSONs in Appendix B (lines 744-773, 1171-1200)

---

## Next Steps

### Immediate (Before Implementation):
1. ✅ Update ContentAnalysisCHILD.md Section 5.2.2 to add `performance_group` field
2. ✅ Update ContentAnalysisCHILD.md Section 10.3 cross-reference note
3. ✅ Update example JSONs in Appendix B to include `performance_group: "top"` field

### Implementation Phase:
1. Stage 2.7 Classification: Add 2 lines to inject `performance_group` field during classification generation
2. Stage 7 Aggregation: Implement 13-field aggregation function (~30 lines Python)
3. Stage 7 LLM Prompt: Update template to receive aggregated stats (~1.5K tokens)

### V2 Planning (Future):
1. Review customer demand for ML×content correlation
2. If implementing correlation: Follow Stage2.6correlation.md Section 6.4 (Option C recommended)
3. Monitor sample size growth (trigger at 100+ videos per bucket for reliable correlations)

### Documentation Debt:
- Create Stage7_LLMReportGenerationCHILD.md when Stage 7 implementation begins
- Document effect size calculation logic in Stage 7 docs (not Content Analysis docs)
- Update MLPlanningv2.md to reference ContentAnalysisCHILD.md for Stages 2.6/2.7

---

**Status**: ContentAnalysisCHILDpt2.md COMPLETE
**Date**: 2025-10-14
**All Questions Answered**: Yes (Q1-Q4 + Token Budget + Data Flow)
