# ML × Content Correlation Analysis

> **Purpose**: Technical design for correlating ML features with content patterns
> **Status**: V2 Feature (Deferred from MVP)
> **Date**: 2025-10-14
> **Related**: ContentAnalysisCHILD.md, MLPlanningv2.md Stage 6

---

## 1. What Is ML × Content Correlation?

**Definition**: Identifying statistical relationships between content patterns (from Content Analysis) and ML features (from RumiAI temporal analysis).

**Example Correlation**:
- **Content Pattern**: "Videos using `direct_to_camera` presentation tactic"
- **ML Feature**: `hook_eye_contact_rate` (quantitative metric)
- **Correlation**: Videos with `direct_to_camera` have 1.42x higher `hook_eye_contact_rate` on average
- **Insight**: "Film yourself speaking directly to camera - this drives eye contact, which is a high-importance ML feature for virality"

---

## 2. Why Is This Valuable?

### 2.1 Business Value

**Without Correlation (MVP)**:
```
Report Section 1: ML Insights
- "Eye contact rate is the #1 predictor of virality"
- "Top performers average 88% eye contact vs 45% in bottom"
- Actionable: "Maintain high eye contact"

Report Section 2: Content Insights
- "75% of top performers use direct_to_camera vs 30% of bottom"
- Actionable: "Film yourself speaking directly to camera"
```

**With Correlation (V2)**:
```
Report Section 3: Why These Tactics Work (ML × Content)
- "Direct_to_camera presentation drives 1.42x higher eye contact rates"
- "Before_after_reveal drives 2.1x higher scene_count (scene cuts between shots)"
- "Problem_solution hooks drive 1.8x higher word_count in opening 3 seconds"
- Insight: "These tactics work BECAUSE they trigger high-importance ML features"
```

### 2.2 Value Propositions

1. **Explanatory Power**: Moves from "what works" to "why it works scientifically"
2. **Trust Building**: Data-backed mechanism explanations increase creator confidence
3. **Competitive Moat**: Unique insight layer competitors can't easily replicate
4. **Optimization Guidance**: "If you can't do direct_to_camera, find other ways to increase eye contact"
5. **Discovery of Surprising Patterns**: "Vulnerability_shown tactic correlates with lower energy_level - authenticity over hype"

---

## 3. Technical Architecture

### 3.1 Data Sources

**Input 1: Content Classifications** (Stage 2.7 output)
```json
{
  "video_id": "7526250443832331550",
  "content_category": "wellness_practice",
  "hook_strategy": "direct_statement",
  "content_tactics": ["direct_to_camera", "personal_story", "vulnerability_shown"],
  "engagement_drivers": ["personal_testimony"],
  ...
}
```
- 120 files per hashtag (40 per bucket × 3 buckets)
- Each file: ~2KB
- Fields to correlate: Arrays (content_tactics, engagement_drivers) and strings (hook_strategy, content_category)

**Input 2: ML Features** (Stage 3 output)
```csv
video_id,hook_eye_contact_rate,hook_scene_count,hook_word_count,middle_1_word_count,...
7526250443832331550,0.85,3,15,55,...
7428596413707144481,0.62,5,8,42,...
...
```
- `aggregated_features.csv` per bucket
- 40-80 videos × 65-215 features (bucket-dependent)
- Quantitative features: floats (0.0-1.0 for rates) or ints (counts)

**Input 3: ML Feature Importance** (Stage 6 output)
```json
{
  "feature_importance": [
    {
      "feature": "hook_eye_contact_rate",
      "importance": 0.23,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45
    },
    ...
  ]
}
```
- Ranked features by importance
- Use to prioritize which correlations matter most

### 3.2 Correlation Pipeline

**Step 1: Load and Join Data**
```python
def load_correlation_data(bucket_path):
    """
    Load and join content classifications with ML features.

    Returns:
        pd.DataFrame: Merged dataset with video_id as key
    """
    # Load content classifications
    content_files = glob(f"{bucket_path}/content_analysis/*_content.json")
    content_data = []
    for file in content_files:
        classification = load_json(file)
        content_data.append({
            'video_id': classification['video_id'],
            'content_category': classification['content_category'],
            'hook_strategy': classification['hook_strategy'],
            'content_tactics': classification['content_tactics'],  # Array
            'engagement_drivers': classification['engagement_drivers'],  # Array
            'performance_group': classification.get('performance_group', 'unknown')
        })

    content_df = pd.DataFrame(content_data)

    # Load ML features
    ml_df = pd.read_csv(f"{bucket_path}/ml_analysis/aggregated_features.csv")

    # Join by video_id
    merged = content_df.merge(ml_df, on='video_id', how='inner')

    # Result: DataFrame with content fields + ~185 ML feature columns
    return merged
```

**Step 2: Compute Correlations for Array Fields (content_tactics, engagement_drivers)**
```python
def correlate_tactics_with_ml(merged_df, ml_feature_importance):
    """
    Find correlations between content tactics and ML features.

    Args:
        merged_df: DataFrame with content + ML columns
        ml_feature_importance: Ranked ML features from Stage 6

    Returns:
        list: Correlation insights ranked by effect size × ML importance
    """
    correlations = []

    # Get top 15 ML features (focus on high-importance features)
    top_ml_features = [f['feature'] for f in ml_feature_importance[:15]]

    # Get unique tactics across all videos
    all_tactics = set()
    for tactics_list in merged_df['content_tactics']:
        all_tactics.update(tactics_list)

    # For each tactic × ML feature combination
    for tactic in all_tactics:
        # Split videos into with/without tactic
        has_tactic = merged_df['content_tactics'].apply(lambda x: tactic in x)
        videos_with = merged_df[has_tactic]
        videos_without = merged_df[~has_tactic]

        # Skip if sample size too small
        if len(videos_with) < 5 or len(videos_without) < 5:
            continue

        for ml_feature in top_ml_features:
            avg_with = videos_with[ml_feature].mean()
            avg_without = videos_without[ml_feature].mean()

            # Calculate effect size
            if avg_without > 0:
                effect_size = avg_with / avg_without
            else:
                effect_size = float('inf') if avg_with > 0 else 1.0

            # Threshold: 30%+ difference
            if effect_size > 1.3 or effect_size < 0.77:  # 1.3x higher or 0.77x lower (30% diff)
                # Get ML importance for this feature
                ml_importance = next((f['importance'] for f in ml_feature_importance if f['feature'] == ml_feature), 0)

                # Composite score: effect_size × ml_importance
                correlation_strength = abs(effect_size - 1.0) * ml_importance

                correlations.append({
                    'content_pattern_type': 'tactic',
                    'content_pattern': tactic,
                    'ml_feature': ml_feature,
                    'ml_importance': ml_importance,
                    'videos_with_tactic': len(videos_with),
                    'videos_without_tactic': len(videos_without),
                    'avg_with': avg_with,
                    'avg_without': avg_without,
                    'effect_size': effect_size,
                    'direction': 'increases' if effect_size > 1.0 else 'decreases',
                    'correlation_strength': correlation_strength,
                    'insight': generate_insight(tactic, ml_feature, effect_size, ml_importance)
                })

    # Sort by correlation_strength (highest impact first)
    correlations.sort(key=lambda x: x['correlation_strength'], reverse=True)

    return correlations


def generate_insight(tactic, ml_feature, effect_size, ml_importance):
    """
    Generate natural language insight.
    """
    direction = "increases" if effect_size > 1.0 else "decreases"
    magnitude = f"{effect_size:.2f}x" if effect_size > 1.0 else f"{1/effect_size:.2f}x lower"

    # Humanize feature names
    feature_display = ml_feature.replace('_', ' ').replace('hook ', 'opening ').title()

    importance_label = "high-importance" if ml_importance > 0.15 else "medium-importance" if ml_importance > 0.08 else "low-importance"

    return (
        f"Videos with '{tactic}' presentation have {magnitude} {feature_display}, "
        f"which is a {importance_label} viral predictor (importance: {ml_importance:.2f})"
    )
```

**Step 3: Compute Correlations for String Fields (hook_strategy, content_category)**
```python
def correlate_hooks_with_ml(merged_df, ml_feature_importance):
    """
    Find correlations between hook strategies and ML features.
    Similar logic to tactics, but for categorical string field.
    """
    correlations = []

    # Get unique hook strategies
    unique_hooks = merged_df['hook_strategy'].unique()

    # For each hook strategy
    for hook in unique_hooks:
        videos_with_hook = merged_df[merged_df['hook_strategy'] == hook]
        videos_without_hook = merged_df[merged_df['hook_strategy'] != hook]

        if len(videos_with_hook) < 5:
            continue

        # Compare against top ML features
        for ml_feature in [f['feature'] for f in ml_feature_importance[:15]]:
            avg_with = videos_with_hook[ml_feature].mean()
            avg_without = videos_without_hook[ml_feature].mean()

            effect_size = avg_with / avg_without if avg_without > 0 else 1.0

            if effect_size > 1.3 or effect_size < 0.77:
                # [Similar correlation object as above]
                correlations.append({...})

    return correlations
```

**Step 4: Statistical Significance Testing (Optional but Recommended)**
```python
from scipy import stats

def add_statistical_significance(correlation, videos_with, videos_without, ml_feature):
    """
    Add p-value to determine if correlation is statistically significant.
    """
    with_values = videos_with[ml_feature]
    without_values = videos_without[ml_feature]

    # T-test for difference in means
    t_stat, p_value = stats.ttest_ind(with_values, without_values)

    correlation['p_value'] = p_value
    correlation['significant'] = p_value < 0.05  # 95% confidence

    return correlation
```

---

## 4. Output Schema

**Correlation Analysis JSON** (Stage 7 input)

**File**: `ml_analysis/ml_content_correlations.json`

```json
{
  "bucket": "33_60s",
  "hashtag": "nutrition",
  "total_videos": 40,
  "correlations": [
    {
      "content_pattern_type": "tactic",
      "content_pattern": "direct_to_camera",
      "ml_feature": "hook_eye_contact_rate",
      "ml_importance": 0.23,
      "videos_with_tactic": 30,
      "videos_without_tactic": 10,
      "avg_with": 0.75,
      "avg_without": 0.53,
      "effect_size": 1.42,
      "direction": "increases",
      "correlation_strength": 0.097,
      "p_value": 0.012,
      "significant": true,
      "insight": "Videos with 'direct_to_camera' presentation have 1.42x Opening Eye Contact Rate, which is a high-importance viral predictor (importance: 0.23)"
    },
    {
      "content_pattern_type": "driver",
      "content_pattern": "before_after_reveal",
      "ml_feature": "hook_scene_count",
      "ml_importance": 0.18,
      "videos_with_tactic": 25,
      "videos_without_tactic": 15,
      "avg_with": 5.2,
      "avg_without": 2.8,
      "effect_size": 1.86,
      "direction": "increases",
      "correlation_strength": 0.155,
      "p_value": 0.003,
      "significant": true,
      "insight": "Videos with 'before_after_reveal' driver have 1.86x Opening Scene Count (scene cuts for before/after shots), which is a high-importance viral predictor (importance: 0.18)"
    },
    {
      "content_pattern_type": "hook",
      "content_pattern": "problem_solution",
      "ml_feature": "hook_word_count",
      "ml_importance": 0.15,
      "videos_with_tactic": 24,
      "videos_without_tactic": 16,
      "avg_with": 22.4,
      "avg_without": 12.1,
      "effect_size": 1.85,
      "direction": "increases",
      "correlation_strength": 0.128,
      "p_value": 0.008,
      "significant": true,
      "insight": "Videos with 'problem_solution' hook have 1.85x Opening Word Count (problem statement requires explanation), which is a medium-importance viral predictor (importance: 0.15)"
    }
  ],
  "summary": {
    "total_correlations_found": 15,
    "significant_correlations": 12,
    "top_3_strongest": [
      "before_after_reveal → hook_scene_count (1.86x, p=0.003)",
      "direct_to_camera → hook_eye_contact_rate (1.42x, p=0.012)",
      "problem_solution → hook_word_count (1.85x, p=0.008)"
    ]
  }
}
```

**Size**: ~8KB for 15 correlations (~8,000 tokens when sent to Stage 7 LLM)

---

## 5. Stage 7 Integration

### 5.1 LLM Prompt Enhancement

**With Correlation Data**:
```python
prompt = f"""
Generate creative strategy report.

ML INSIGHTS (Top Performers vs Bottom):
- eye_contact_rate: 0.88 vs 0.45 (gap: 0.43, importance: 0.23)
- scene_count: 4.2 vs 2.1 (gap: 2.1x, importance: 0.18)

CONTENT INSIGHTS (Top vs Bottom):
- direct_to_camera: 75% vs 30% (2.5x more common)
- problem_solution hook: 60% vs 20% (3x more common)

ML × CONTENT CORRELATIONS (WHY tactics work):
{json.dumps(correlations['top_3_strongest'])}

Generate report with sections:
1. What Works (ML + Content combined with WHY explanations)
2. Actionable Checklist

Use friendly tone. Focus on replicable tactics with scientific backing.
"""
```

### 5.2 Report Output Example

**Section: "Why Direct-to-Camera Presentation Works"**
```
📊 Data-Backed Insight:

75% of top-performing videos use direct-to-camera presentation, compared to just 30% of
low-performing videos (2.5x more common).

🔬 Why This Works (Science):

Our ML analysis reveals that direct-to-camera videos achieve 1.42x higher eye contact
rates (75% vs 53%). Eye contact rate is the #1 predictor of virality in our models
(importance: 0.23/1.0).

✅ Actionable:

Film yourself speaking directly into the camera lens. Avoid b-roll-only videos or
voiceovers. Your face + eye contact = connection = shares.
```

---

## 6. Implementation Effort

### 6.1 Development Tasks

| Task | Effort | Complexity | Risk |
|------|--------|------------|------|
| Data loading & joining | 0.5 day | Low | Low (standard pandas operations) |
| Correlation computation (tactics) | 1 day | Medium | Medium (array field handling) |
| Correlation computation (hooks, categories) | 0.5 day | Low | Low (string field grouping) |
| Statistical significance testing | 0.5 day | Medium | Medium (scipy dependency, interpretation) |
| Output schema generation | 0.5 day | Low | Low |
| Stage 7 integration (LLM prompt) | 0.5 day | Low | Low |
| Testing (unit + integration) | 1 day | Medium | High (must validate correlations make sense) |
| **Total** | **4.5 days** | **Medium** | **Medium** |

### 6.2 Dependencies

**Required**:
- pandas (existing)
- numpy (existing)
- scipy (NEW - for statistical testing)

**Stage Dependencies**:
- Stage 2.7 must complete (content classifications exist)
- Stage 3 must complete (aggregated_features.csv exists)
- Stage 6 must complete (feature importance ranking exists)

### 6.3 Risks

1. **Small Sample Sizes**: With 40 videos per bucket, some tactics may have <5 examples → spurious correlations
   - Mitigation: Minimum sample size threshold (5 videos), p-value filtering

2. **Correlation ≠ Causation**: Correlation doesn't prove tactic CAUSES ML feature change
   - Mitigation: Report language emphasizes "associated with" not "causes"

3. **Multiple Comparisons Problem**: Testing 20 tactics × 15 ML features = 300 tests → 5% false positive rate = 15 spurious correlations
   - Mitigation: Bonferroni correction (adjust p-value threshold) or focus on top 5 correlations only

4. **Token Budget**: Adding 8K tokens for correlations + 20K for content + 10K for ML = 38K total
   - Mitigation: Send only top 10 correlations (reduces to ~5K tokens)

### 6.4 Prerequisite: Reverse Aggregation Decision

**Current MVP Architecture** (per ContentAnalysisCHILDpt2.md Question 2):
```python
# Stage 7 current plan (MVP)
def generate_content_insights():
    # Load 120 individual files
    classifications = load_all_classifications()

    # AGGREGATE in Python (13 fields → frequency distributions)
    stats = aggregate_classifications(classifications)  # ~1.5K tokens

    # Send ONLY aggregated stats to LLM
    prompt = f"Content patterns: {stats}"  # LLM doesn't see individual videos
    return llm.generate(prompt)
```

**Correlation Architecture Requirement**:
```python
# Stage 7 with correlation (V2)
def generate_content_insights_with_correlation():
    # Load 120 individual files
    classifications = load_all_classifications()  # INDIVIDUAL files, not aggregated

    # Compute correlations (requires video-level data)
    correlations = correlate_content_with_ml(classifications, ml_features)

    # Send individual classifications OR correlations to LLM
    prompt = f"""
    Content classifications: {classifications}  # 120 videos, ~20K tokens
    ML correlations: {correlations}  # ~5K tokens
    """
    return llm.generate(prompt)
```

**Breaking Change Required**:

| Aspect | MVP (Aggregated) | V2 Correlation (Individual) | Impact |
|--------|------------------|----------------------------|---------|
| **Python processing** | Aggregate to stats | Skip aggregation, compute correlations | Rewrite Stage 7 data flow |
| **LLM input** | 13 field frequencies (~1.5K tokens) | 120 individual files (~20K tokens) | 13x token increase |
| **Stage 7 logic** | Simple aggregation (5 lines) | Complex correlation (200+ lines) | Significant refactor |
| **Cost per report** | Baseline | +$0.055 (18.5K extra tokens) | 15% cost increase |

**Implementation Considerations**:

1. **Can't Have Both**: Either aggregate OR send individual files, not both
   - Aggregating loses video-level granularity needed for correlation
   - Must choose one architecture

2. **Code Changes Required**:
   - Stage 7 `aggregate_classifications()` function → delete or disable
   - Stage 7 `generate_content_insights()` → rewrite to use individual files
   - Stage 7 LLM prompt template → include individual classifications or correlations

3. **Token Budget Impact**:
   - MVP: 1.5K (content) + 10K (ML) = 11.5K tokens
   - V2 Option A: 20K (content individual) + 5K (correlations) + 10K (ML) = 35K tokens
   - V2 Option B: 5K (correlations only) + 10K (ML) = 15K tokens (if skipping individual content)

4. **Migration Paths**:

   **Option A: Replace Aggregation with Individual Files**
   - Delete aggregation logic
   - Send 120 individual classifications to LLM
   - LLM performs contrastive analysis from raw data
   - Pros: LLM has full flexibility, can discover unexpected patterns
   - Cons: 20K token overhead, LLM may miscount, slower processing

   **Option B: Add Correlation Alongside Aggregation (Hybrid)**
   - Keep aggregation for content-only insights
   - Add correlation for ML×content insights
   - Send both to LLM (aggregated stats + correlations)
   - Pros: Best of both worlds
   - Cons: 36.5K tokens total (MVP 11.5K → V2 36.5K = 217% increase)

   **Option C: Compute Correlations in Python, Send Insights Only ⭐ RECOMMENDED**
   - Keep aggregation for content-only insights
   - Compute correlations in Python (don't send raw data)
   - Send correlation insights to LLM: "direct_to_camera → 1.42x eye_contact_rate"
   - Pros: Only 3.5K token increase (15K total), Python does data crunching, LLM does synthesis
   - Cons: LLM can't discover correlations we didn't compute

**Recommended Approach**: **Option C - Compute Correlations in Python**

```python
# V2 Stage 7 with Option C
def generate_insights_with_correlation():
    # Load data
    classifications = load_all_classifications()
    ml_features = load_ml_features()

    # Python aggregates content (MVP logic, unchanged)
    content_stats = aggregate_classifications(classifications)  # 1.5K tokens

    # Python computes correlations (NEW V2 logic)
    correlations = correlate_content_with_ml(classifications, ml_features)
    correlation_insights = format_top_10_correlations(correlations)  # 5K tokens

    # LLM receives both (no individual files)
    prompt = f"""
    CONTENT PATTERNS: {content_stats}  # 1.5K tokens
    ML×CONTENT CORRELATIONS: {correlation_insights}  # 5K tokens
    ML FEATURE IMPORTANCE: {ml_importance}  # 10K tokens
    Total: 16.5K tokens (vs MVP 11.5K = +43% increase)
    """
    return llm.generate(prompt)
```

**Why Option C is Superior**:
1. **Preserves MVP Architecture**: Aggregation logic remains, just adds correlation layer
2. **Token Efficient**: 5K for correlations vs 20K for individual files (75% savings)
3. **Task Alignment**: Python handles arithmetic (correlation computation), LLM handles synthesis (insight generation)
4. **Non-Breaking Change**: Can ship V2 without rewriting Stage 7 from scratch
5. **Deterministic**: Python guarantees correct correlation calculations, no LLM miscounting risk

**Decision Needed Before V2 Implementation**:
- ✅ Confirm: Use Option C (Python-computed correlations, no individual files sent to LLM)
- ✅ Update ContentAnalysisCHILDpt2.md if choosing Option A (must reverse Question 2 decision)
- ✅ Document correlation output schema for LLM prompt (similar to section 4 above, but formatted as insights not raw JSON)

---

## 7. Performance & Cost Impact

### 7.1 Processing Time

**Correlation computation**:
- Load data: ~1 second
- Compute correlations (20 tactics × 15 features): ~5 seconds
- Statistical testing: ~2 seconds
- Total: **~8 seconds per bucket**

**Not a bottleneck** - adds 8s to pipeline that already takes 5-10 minutes per bucket.

### 7.2 Token Cost

**Stage 7 LLM input**:
- Without correlation: 12K tokens (aggregated only)
- With correlation (top 10): 17K tokens
- With correlation (all 15): 20K tokens

**Cost increase**: ~5K tokens = ~$0.015 per report (Sonnet)
- 10 hashtags × 3 buckets = 30 reports
- Total cost increase: 30 × $0.015 = **$0.45 per full hashtag analysis**

**Negligible cost increase** (< 10% of total pipeline cost).

---

## 8. Trade-Offs Summary

### 8.1 MVP (Without Correlation)

**Pros**:
- ✅ Faster to market (save 1 week dev time)
- ✅ Simpler Stage 7 pipeline
- ✅ Lower token cost (12K vs 20K)
- ✅ Both insight types independently actionable

**Cons**:
- ❌ No mechanistic explanations ("why")
- ❌ Lower perceived sophistication
- ❌ Missed competitive differentiation

### 8.2 V2 (With Correlation)

**Pros**:
- ✅ Deeper insights with mechanistic explanations
- ✅ Stronger competitive moat (unique insight layer)
- ✅ Higher perceived value ("scientific backing")
- ✅ Discovery of surprising patterns

**Cons**:
- ❌ Additional 1 week development time
- ❌ More complex pipeline (3 data sources)
- ❌ Requires statistical validation (avoid spurious correlations)
- ❌ Small sample size risks (40 videos may be insufficient)

---

## 9. Recommendation

### 9.1 Ship MVP First

**Reasoning**:
1. Both ML and content insights are independently valuable
2. Validate customer demand before investing 1 week
3. Small sample sizes (40 videos) make correlations risky for MVP
4. Faster time to market = earlier feedback

### 9.2 V2 Trigger Conditions

Add correlation when:
1. **Customer demand**: "I want to know WHY these tactics work"
2. **Sample size increase**: If buckets scale to 100+ videos → more reliable correlations
3. **Competitive pressure**: Competitors add similar features
4. **Usage data**: High engagement with ML + content sections → users care about depth

### 9.3 V2 Implementation Timeline

**Phase 1** (Week 1): Core correlation engine
- Data loading, joining, correlation computation
- Output JSON schema

**Phase 2** (Week 2): Statistical validation + Stage 7 integration
- P-value testing, Bonferroni correction
- LLM prompt enhancement, report generation

**Phase 3** (Week 3): Testing + refinement
- Validate correlations make sense (not garbage)
- User testing with 2-3 beta customers

**Total**: 3 weeks from decision to production

---

## 10. Example Correlations (Hypothetical)

Based on content patterns and expected ML relationships:

| Content Pattern | ML Feature | Expected Effect | Mechanism |
|----------------|------------|-----------------|-----------|
| `direct_to_camera` | `hook_eye_contact_rate` | 1.4x higher | Direct gaze at camera = eye contact detection |
| `before_after_reveal` | `hook_scene_count` | 2.0x higher | Before/after shots = scene cuts |
| `problem_solution` hook | `hook_word_count` | 1.8x higher | Problem explanation requires words |
| `vulnerability_shown` | `emotional_valence` | 0.6x (lower) | Vulnerability = lower positive emotion |
| `specific_metrics_mentioned` | `middle_word_count` | 1.5x higher | Explaining metrics requires speech |
| `personal_story` | `speech_coverage` | 1.3x higher | Storytelling = more speaking time |
| `question` hook | `hook_duration` | 0.9x (shorter) | Questions are concise |
| `product_demonstration` | `element_count` | 1.6x higher | Product = visual element detection |

These are **hypotheses** - actual correlations must be computed from real data.

---

## 11. Future Enhancements (V3+)

1. **Multi-Tactic Correlations**: "Videos combining X + Y have 2.5x higher Z"
2. **Temporal Correlations**: "Tactic X in hook drives feature Y in middle segments"
3. **Causal Analysis**: Use propensity score matching to approach causality
4. **Cross-Hashtag Correlations**: "Tactic X works across all hashtags" vs "Tactic Y is niche-specific"
5. **Negative Correlations**: "Avoid tactic X - it reduces feature Y"

---

## 12. References

- **ContentAnalysisCHILD.md**: Content classification schema
- **MLPlanningv2.md Stage 3**: Aggregated features CSV schema
- **MLPlanningv2.md Stage 6**: ML feature importance output
- **Statistical Methods**: T-tests, Bonferroni correction for multiple comparisons

---

**Document Status**: Technical design complete, awaiting product decision on V2 prioritization

**Last Updated**: 2025-10-14
**Author**: Claude Code (Phase 3 exploration)
