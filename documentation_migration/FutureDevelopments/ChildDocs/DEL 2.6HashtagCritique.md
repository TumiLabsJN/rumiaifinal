# Stage 2.6 Discovery Prompt - Comprehensive Critique & Refinement

> **Purpose**: Systematic refinement of Discovery prompt for hashtag content pattern analysis
> **Created**: 2025-01-28
> **Status**: In Progress - Section-by-Section Review

---

## Overview

This document critiques and refines the Stage 2.6 Discovery prompt through 12 systematic audits. Each section presents 3 alternatives with recommendation and final decision.

**Prompt Context**:
- **Input**: 50 video transcripts from top-performing videos in a hashtag
- **Output**: JSON with 6 pattern categories (content types, hooks, pain points, keywords, drivers, tactics)
- **Model**: Claude 3.5 Sonnet
- **Cost**: ~$0.75 per hashtag
- **Duration**: ~45 seconds

---

## Section 1: System Message

**Current**:
```
You are an expert TikTok content strategist and pattern recognition analyst. Your specialty is identifying natural content patterns, creative hooks, audience psychology, and viral tactics from video transcripts. You think in terms of actionable creative strategies that content creators can replicate.
```

**Questions**:
1. Should we emphasize "frequency-based" pattern discovery?
2. Should we add "Be objective and data-driven, not prescriptive"?
3. Is "TikTok content strategist" too specific?

**Decision**: ✅ Alternative C (Balanced Hybrid)

**Final System Message**:
```
You are an expert content analyst specializing in short-form video patterns. Identify recurring patterns in the transcripts based on frequency and evidence. Be objective and data-driven: report patterns that actually appear in the data, not prescriptive advice. Patterns should be actionable for content creators but grounded in observed behavior.
```

**Rationale**: Balances expertise with objectivity, platform-agnostic ("short-form video"), prevents hallucination while maintaining output quality, sets clear dual mandate (actionable + evidence-based).

---

## Section 2: Task Framing

**Current**:
```
Analyze the following {sample_size} TikTok video transcripts from the #{hashtag} hashtag.

Your task is to identify natural content patterns across 6 categories. Be specific and actionable - content creators will use these patterns to guide their video creation.
```

**Questions**:
1. Should we explicitly state "Identify patterns present in AT LEAST 10% of videos (5+ videos)"?
2. Should we warn "Do not create patterns for single videos or rare occurrences"?
3. Should we add context: "These are TOP-PERFORMING videos"?

**Decision**: ✅ Option 2 (Percentage with Floor)

**Final Task Framing**:
```
Analyze the following {sample_size} video transcripts from the #{hashtag} hashtag.

Your task is to identify recurring content patterns across 6 categories. Focus on patterns that appear in AT LEAST 10% of videos (minimum 3 videos). Do not create patterns for isolated or single-video elements.

Patterns should be specific and actionable so content creators can replicate them.
```

**Rationale**: Percentage-based threshold scales with configurable sample size (10-200 videos), floor of 3 videos prevents degenerate patterns in small samples, explicit guard rail against single-video patterns, maintains actionability requirement.

---

## Section 3: Category 1 - Content Categories

**Current Instructions**:
```
Identify the TYPES of videos that exist in this hashtag. Focus on the primary content format or purpose.

Examples of content categories:
- recipe_tutorial: Step-by-step cooking instructions
- supplement_review: Product recommendations and personal experiences
- myth_busting: Debunking common misconceptions
```

**Questions**:
1. Are examples too specific to nutrition hashtag? (May bias LLM for fitness/beauty)
2. Should examples be generic: "tutorial", "review", "storytelling"?
3. Should we add: "Limit to 5-10 categories maximum"?

**Decision**: ✅ Option A2 (Range with Flexibility)

**Final Instructions**:
```
Identify the TYPES of videos that exist in this hashtag. Focus on the primary content format or purpose.

Examples to show naming style only (these are NOT limits):
- instructional_walkthrough: Teaching or how-to format
- personal_perspective: Opinion, review, or commentary
- narrative_content: Story-driven or journey-based

Create NEW category names based on patterns you observe in THIS specific hashtag (#{hashtag}). Do not limit yourself to these examples - they only demonstrate the naming format (snake_case, 2-4 words, descriptive).

Discover the categories that reflect actual content patterns in these transcripts. Typically this will be 3-8 categories, but return only as many as you genuinely observe - do not force patterns to reach a target number.
```

**Rationale**: 3 abstract examples minimize domain anchoring (vs 7+ examples), triple disclaimers prevent template matching, "typically 3-8" sets realistic expectation without hard constraint, explicit escape clause prevents forcing artificial patterns, no minimum floor allows homogeneous content to have fewer categories.

---

## Section 4: Category 2 - Hook Strategies

**Current Instructions**:
```
Identify HOW videos open in the first 3-5 seconds. What technique captures attention?

Examples of hook strategies:
- problem_solution: Starts with a problem, promises a solution
- shocking_fact: Opens with surprising statistic or claim
```

**Questions**:
1. Problem: Transcripts don't have timestamps - how does LLM know "first 3-5 seconds"?
2. Should we say: "Identify how videos open (first 1-2 sentences of transcript)"?
3. Are examples too leading?

**Decision**: ✅ Alternative B (Opening Clause Analysis)

**Final Instructions**:
```
Identify HOW videos open. Analyze the OPENING PHRASE (first 5-10 words of each transcript) to detect attention-grabbing techniques.

Examples to show naming style only (these are NOT limits):
- question_hook: Opens with a question
- direct_address: Speaks directly to viewer
- surprising_claim: Unexpected statement or fact

Create NEW hook strategy names based on opening patterns you observe in THIS specific hashtag (#{hashtag}). Focus on the rhetorical technique, not specific content. Typically this will be 2-5 hook strategies, but return only as many as you genuinely observe.
```

**Rationale**: "First 5-10 words" operationalizes "first 3-5 seconds" without timestamps (2.5-3 words/second speech rate = ~2-4 seconds), word count is concrete and executable for LLM, maintains minimal anchoring with 3 abstract examples, "rhetorical technique not content" prevents over-specificity.

---

## Section 5: Categories 3-6 (Simple Lists)

**Current Instructions**:
```
## CATEGORY 3: Audience Pain Points
Identify the PROBLEMS or STRUGGLES mentioned in the content.
Return as simple string list: ["bloating", "low_energy"]

## CATEGORY 4: Trending Keywords
Identify SPECIFIC TERMS and PHRASES that appear frequently.
Return as simple string list: ["protein", "gut_health"]
```

**Questions**:
1. Should we add: "Extract verbatim phrases from transcripts"?
2. How to distinguish "Pain Points" vs "Keywords"? (Overlap risk)
3. Should we add frequency threshold: "Only include terms in 5+ transcripts"?

**Decision**: ✅ Alternative D (Grounded Discovery with Flexibility)

**Final Instructions**:
```
For Categories 3-6 below: Extract phrases (2-4 words) that appear or are IMPLIED in at least 10% of videos (minimum 3). Phrases can be verbatim quotes OR interpretations of what's shown/discussed. Return as simple string lists.

GROUNDING RULE: Every term you list must be traceable to specific transcripts. If you cannot point to at least 3 transcripts showing this pattern, do not include it.

## CATEGORY 3: Audience Pain Points
Identify PROBLEMS, STRUGGLES, or UNMET NEEDS mentioned or implied. Include:
- Explicit problems stated ("I have bloating")
- Implied problems from solutions shown ("I started doing X and Y went away" → Y is pain point)
- Challenges discussed

## CATEGORY 4: Trending Keywords
Identify TOPICS, METHODS, SOLUTIONS, or CONCEPTS mentioned or implied (excluding problems from Category 3). Include:
- Specific terms used repeatedly
- Methods or practices discussed
- Solutions or approaches mentioned

## CATEGORY 5: Engagement Drivers
Identify CONTENT FEATURES or TECHNIQUES that make content compelling (not topics). Include:
- Storytelling devices mentioned or used
- Proof elements described ("I show before/after photos")
- Engagement tactics visible in how creators speak

## CATEGORY 6: Content Tactics
Identify PRESENTATION STYLES or FORMATS mentioned or implied. Include:
- Delivery methods described or evident from speech patterns
- Visual approaches mentioned ("I'm going to show you on screen")
- Structural formats implied by how content flows
```

**Rationale**: Balances all 3 concerns - (1) Grounding rule prevents hallucination while allowing interpretation, (2) "(excluding problems from Cat 3)" + "(not topics)" prevents misclassification, (3) "mentioned or IMPLIED" allows open-ended discovery. Human curation is safety net: false positives (hallucination/misclassification) easy to fix, false negatives (missing patterns) costly. Trade-off favors comprehensive discovery over strict extraction.

---

## Section 6: Output Format

**Current**:
```json
{
  "content_categories": [{
    "name": "recipe_tutorial",
    "frequency": 32,
    "percentage": 64.0,
    "examples": ["step by step recipe"],
    "representative_video_ids": ["7526250443832331550"]
  }]
}
```

**Questions**:
1. Should we require minimum 2 examples (not 1)?
2. Should we require minimum 2 representative video IDs?
3. Should percentage be calculated by LLM or Python post-process?

**Decision**: ✅ Alternative B (Python Post-Processes Calculations)

**Final Output Schema**:
```json
{
  "hashtag": "{hashtag}",
  "analysis_date": "{iso8601_timestamp}",
  "sample_size": {sample_size},
  "discovered_patterns": {
    "content_categories": [
      {
        "name": "instructional_walkthrough",
        "frequency": 28,
        "examples": ["step by step tutorial", "here's how to make"],
        "representative_video_ids": ["7526250443832331550", "7428596413707144481"]
      }
    ],
    "hook_strategies": [same structure],
    "audience_pain_points": ["chronic bloating", "low energy"],
    "trending_keywords": ["protein intake", "gut health"],
    "engagement_drivers": ["before after reveal"],
    "content_tactics": ["direct to camera", "voiceover"]
  }
}
```

**Requirements**:
- Categories 1-2: Provide 2-3 examples and 2-3 representative_video_ids per pattern
- NO percentage field (Python calculates: frequency / sample_size * 100)
- Categories 3-6: Simple string lists

**Rationale**: LLMs make math errors, Python is deterministic. Video IDs essential for curator verification. Percentage is derived data computed in post-processing. Flexible 2-3 examples/IDs prevents LLM padding with weak examples.

---

## Section 7: Final Instructions

**Current**:
```
1. Analyze ALL {sample_size} transcripts carefully
2. Identify patterns that appear in at least 10% of videos (5+ videos)
3. Use descriptive snake_case names (short but clear)
4. Return ONLY the JSON output (no additional text)
```

**Questions**:
1. Should we add: "If you cannot identify 5 patterns in a category, return fewer"?
2. Should we add: "Do not make up patterns - only report observations"?
3. Should we add examples of BAD outputs to avoid?

**Decision**: ✅ Alternative B (Moderate Guard Rails)

**Final Instructions**:
```
FINAL INSTRUCTIONS:

1. Analyze ALL {sample_size} transcripts carefully - every pattern must be grounded in observed data
2. Return only patterns you genuinely observe - if a category has fewer patterns, that's acceptable
3. Use descriptive snake_case names (short but clear, 2-4 words)
4. Return valid JSON only - no commentary, explanations, or additional text outside the JSON structure

DO NOT make up patterns to fill categories. Quality over quantity.
```

**Rationale**: Explicitly addresses flexibility (Question 1) with "if a category has fewer patterns, that's acceptable". Directly addresses anti-hallucination (Question 2) with "DO NOT make up patterns" + reinforces grounding from Section 5. Avoids negative examples (Question 3) - positive framing more effective. Adds "2-4 words" naming specificity. Each line serves distinct purpose without redundancy.

---

## Section 8: Clarity Audit (Pattern Boundaries)

**ChatGPT Suggestion**:
Add a "⚠️ Pattern Boundaries" block:
- Only identify patterns that describe recurrent creative structures, not isolated storytelling elements
- Each transcript can belong to multiple categories, but do not create sub-patterns unless they appear in ≥10% of videos

**Questions**:
1. How to define "recurrent creative structures" vs "isolated storytelling elements"?
2. Should we give examples of what NOT to do?
3. Should this be in System Message or Task Instructions?

**Decision**: ✅ Alternative A (Skip - Already Covered)

**No additional instructions needed.**

**Rationale**: The 10% threshold (minimum 3 videos) already prevents "isolated storytelling elements" by definition. Section 2 explicitly states "Do not create patterns for isolated or single-video elements". Section 5 grounding rule requires "traceable to at least 3 transcripts". Adding abstract "pattern boundaries" creates conceptual confusion - LLM needs concrete frequency requirements, not philosophical distinctions. Existing safeguards across Sections 1, 2, 5, and 7 already comprehensively address this concern. The 10% frequency threshold IS the pattern boundary.

---

## Section 9: Control Audit (LLM Output Stability)

**ChatGPT Suggestion**:
Add control rules:
- Validate that all representative_video_ids exist in the provided {transcripts_json}
- Round all percentages to one decimal place
- Do not include text commentary or "explanation" keys outside the JSON

**Questions**:
1. Should video ID validation be in prompt or Python post-processing?
2. Should we enforce JSON-only output more strictly?
3. Should we add output schema validation examples?

**Note**: ChatGPT's "round all percentages" suggestion is obsolete - Section 6 decided Python calculates percentages post-LLM.

**Decision**: ✅ Alternative A (All Validation in Python)

**No additional instructions needed.**

**Rationale**: Video ID validation must be in Python - LLM cannot reliably verify IDs exist in provided JSON. JSON-only enforcement already covered in Section 7 line 4: "Return valid JSON only - no commentary, explanations, or additional text outside the JSON structure". Schema validation examples contradict minimalist philosophy from Sections 3-4. Separation of concerns: LLM discovers patterns, Python validates output (schema compliance, valid video IDs, frequency bounds, malformed JSON). Section 6 established this precedent (Python calculates percentages). Prompt should guide discovery, not enumerate validation rules.

---

## Section 10: Structural Audit (Output Schema)

**ChatGPT Suggestion**:
Upgrade Categories 3-6 from simple string lists to frequency objects:
```json
"audience_pain_points": [
  {"term": "bloating", "frequency": 14, "percentage": 28.0},
  {"term": "low_energy", "frequency": 9, "percentage": 18.0}
]
```

**Questions**:
1. Does this add value over simple string lists?
2. Trade-off: Richer data vs simpler curation workflow?
3. Should all 4 categories (pain points, keywords, drivers, tactics) use this format?

**Decision**: ✅ Alternative A (Keep Simple String Lists)

**Current schema remains:**
```json
"audience_pain_points": ["chronic bloating", "low energy"],
"trending_keywords": ["protein intake", "gut health"],
"engagement_drivers": ["before after reveal"],
"content_tactics": ["direct to camera", "voiceover"]
```

**Rationale**: Frequency data doesn't change curation decisions - curator evaluates semantic quality ("Is this a real pain point?"), not frequency validity. Simple strings enable fastest curation workflow, consistent with Section 5 decision (Grounded Discovery with human safety net). Frequency unused in Stage 2.7 Classification (only checks presence/absence). Categories 1-2 have frequency because they're structural patterns with examples + video IDs (richer objects) - Categories 3-6 are qualitative term lists where presence matters, not frequency count. If analytics needed, Python can count term occurrences in transcripts deterministically vs trusting LLM counting.

---

## Section 11: Logic Audit (Analytical Quality)

**ChatGPT Suggestions**:
- Normalize comparisons by transcript length (don't over-weight longer scripts)
- Merge semantically identical terms (e.g., "gut_health" covers "gut" and "digestive_health")
- Prefer actions over topics when ambiguous (e.g., "showing_meal_prep" vs "healthy_food")

**Questions**:
1. Should we add explicit instruction: "Weight patterns by frequency, not transcript length"?
2. Should we instruct: "Merge synonyms under canonical terms"?
3. How to define "actions over topics" clearly?

**Decision**: ✅ Alternative A (Skip All - Already Addressed)

**No additional instructions needed.**

**Rationale**: Length normalization is non-issue - frequency metric is "number of videos exhibiting pattern", not "total mentions across transcripts". Transcript length already inherently normalized by design. Synonym consolidation better handled by human curator with domain context - LLM can't judge if variants represent true synonyms or distinct framings in specific hashtag. False negative (LLM doesn't merge) costs 30 seconds of curator work; false positive (LLM over-merges) loses data. Actions vs topics already addressed in Section 5: Category 5 explicitly states "(not topics)", Category 6 defines "PRESENTATION STYLES or FORMATS". Clear category boundaries sufficient without vague "prefer actions" instruction. Trust LLM to discover, Python to validate, curator to consolidate.

---

## Section 12: Scalability Audit (Metadata Tracking)

**ChatGPT Suggestion**:
Add top-level metadata fields:
```json
{
  "model_version": "claude-3-5-sonnet-20241022",
  "prompt_version": "v1.2",
  "taxonomy_summary": "6 recurring content archetypes emphasizing personal transformation"
}
```

**Questions**:
1. Should metadata be in LLM output or added by Python wrapper?
2. What value does "taxonomy_summary" provide?
3. Should we track prompt version for reproducibility?

**Decision**: ✅ Keep Current Schema (No Additional Metadata)

**Schema remains as defined in Section 6:**
```json
{
  "hashtag": "guthealth",
  "analysis_date": "2025-01-28T14:32:00Z",
  "sample_size": 50,
  "discovered_patterns": { ... }
}
```

**Rationale**: Current schema already contains essential operational metadata (hashtag, timestamp, sample size). Additional metadata fields (model_version, prompt_version, taxonomy_summary) add complexity without clear immediate value. Taxonomy summary is redundant - the discovered patterns ARE the taxonomy. Version tracking can be implemented later if A/B testing or reproducibility analysis becomes necessary. Keep schema simple until proven need for additional fields. Consistent with minimalist philosophy throughout all 12 sections.

---

## Final Prompt (After All Decisions)

This is the production-ready Stage 2.6 Discovery prompt incorporating all decisions from Sections 1-12.

---

### SYSTEM MESSAGE

```
You are an expert content analyst specializing in short-form video patterns. Identify recurring patterns in the transcripts based on frequency and evidence. Be objective and data-driven: report patterns that actually appear in the data, not prescriptive advice. Patterns should be actionable for content creators but grounded in observed behavior.
```

---

### USER MESSAGE

```
Analyze the following {sample_size} video transcripts from the #{hashtag} hashtag.

Your task is to identify recurring content patterns across 6 categories. Focus on patterns that appear in AT LEAST 10% of videos (minimum 3 videos). Do not create patterns for isolated or single-video elements.

Patterns should be specific and actionable so content creators can replicate them.

---

## CATEGORY 1: Content Categories

Identify the TYPES of videos that exist in this hashtag. Focus on the primary content format or purpose.

Examples to show naming style only (these are NOT limits):
- instructional_walkthrough: Teaching or how-to format
- personal_perspective: Opinion, review, or commentary
- narrative_content: Story-driven or journey-based

Create NEW category names based on patterns you observe in THIS specific hashtag (#{hashtag}). Do not limit yourself to these examples - they only demonstrate the naming format (snake_case, 2-4 words, descriptive).

Discover the categories that reflect actual content patterns in these transcripts. Typically this will be 3-8 categories, but return only as many as you genuinely observe - do not force patterns to reach a target number.

---

## CATEGORY 2: Hook Strategies

Identify HOW videos open. Analyze the OPENING PHRASE (first 5-10 words of each transcript) to detect attention-grabbing techniques.

Examples to show naming style only (these are NOT limits):
- question_hook: Opens with a question
- direct_address: Speaks directly to viewer
- surprising_claim: Unexpected statement or fact

Create NEW hook strategy names based on opening patterns you observe in THIS specific hashtag (#{hashtag}). Focus on the rhetorical technique, not specific content. Typically this will be 2-5 hook strategies, but return only as many as you genuinely observe.

---

## CATEGORIES 3-6: Simple Lists

For Categories 3-6 below: Extract phrases (2-4 words) that appear or are IMPLIED in at least 10% of videos (minimum 3). Phrases can be verbatim quotes OR interpretations of what's shown/discussed. Return as simple string lists.

GROUNDING RULE: Every term you list must be traceable to specific transcripts. If you cannot point to at least 3 transcripts showing this pattern, do not include it.

### CATEGORY 3: Audience Pain Points

Identify PROBLEMS, STRUGGLES, or UNMET NEEDS mentioned or implied. Include:
- Explicit problems stated ("I have bloating")
- Implied problems from solutions shown ("I started doing X and Y went away" → Y is pain point)
- Challenges discussed

### CATEGORY 4: Trending Keywords

Identify TOPICS, METHODS, SOLUTIONS, or CONCEPTS mentioned or implied (excluding problems from Category 3). Include:
- Specific terms used repeatedly
- Methods or practices discussed
- Solutions or approaches mentioned

### CATEGORY 5: Engagement Drivers

Identify CONTENT FEATURES or TECHNIQUES that make content compelling (not topics). Include:
- Storytelling devices mentioned or used
- Proof elements described ("I show before/after photos")
- Engagement tactics visible in how creators speak

### CATEGORY 6: Content Tactics

Identify PRESENTATION STYLES or FORMATS mentioned or implied. Include:
- Delivery methods described or evident from speech patterns
- Visual approaches mentioned ("I'm going to show you on screen")
- Structural formats implied by how content flows

---

## OUTPUT FORMAT

Return your analysis as valid JSON with this exact structure:

{
  "hashtag": "{hashtag}",
  "analysis_date": "{iso8601_timestamp}",
  "sample_size": {sample_size},
  "discovered_patterns": {
    "content_categories": [
      {
        "name": "instructional_walkthrough",
        "frequency": 28,
        "examples": ["step by step tutorial", "here's how to make"],
        "representative_video_ids": ["7526250443832331550", "7428596413707144481"]
      }
    ],
    "hook_strategies": [
      {
        "name": "question_hook",
        "frequency": 15,
        "examples": ["did you know that", "have you ever wondered"],
        "representative_video_ids": ["7526250443832331550", "7428596413707144481"]
      }
    ],
    "audience_pain_points": ["chronic bloating", "low energy"],
    "trending_keywords": ["protein intake", "gut health"],
    "engagement_drivers": ["before after reveal"],
    "content_tactics": ["direct to camera", "voiceover"]
  }
}

Requirements:
- Categories 1-2: Provide 2-3 examples and 2-3 representative_video_ids per pattern
- DO NOT include a "percentage" field - this will be calculated automatically by Python post-processing
- Categories 3-6: Simple string lists only (no objects, no extra fields)

---

## FINAL INSTRUCTIONS

1. Analyze ALL {sample_size} transcripts carefully - every pattern must be grounded in observed data
2. Return only patterns you genuinely observe - if a category has fewer patterns, that's acceptable
3. Use descriptive snake_case names (short but clear, 2-4 words)
4. Return valid JSON only - no commentary, explanations, or additional text outside the JSON structure

DO NOT make up patterns to fill categories. Quality over quantity.

---

TRANSCRIPTS:

{transcripts_json}
```

---

### TEMPLATE VARIABLES

When implementing this prompt, replace these placeholders:

- `{sample_size}`: Number of transcripts (e.g., 50)
- `{hashtag}`: Hashtag name without # symbol (e.g., "guthealth")
- `{iso8601_timestamp}`: Current timestamp in ISO 8601 format (e.g., "2025-01-28T14:32:00Z")
- `{transcripts_json}`: JSON array of transcript objects with video_id and transcript_text fields

---

### PYTHON POST-PROCESSING

After receiving LLM output, Python must:

1. **Calculate percentages** (see TI Section 4.2.5):
   ```python
   for pattern in taxonomy['discovered_patterns']['content_categories']:
       pattern['percentage'] = round((pattern['frequency'] / sample_size) * 100, 1)
   ```

2. **Validate output** (see TI Section 4.2.6):
   - All representative_video_ids exist in provided transcripts
   - All frequency values ≤ sample_size
   - Schema compliance (correct fields, correct types)
   - JSON is well-formed

3. **Save to disk** for manual curation workflow

---

## Change Log

| Date | Section | Decision | Rationale |
|------|---------|----------|-----------|
| 2025-01-28 | Document Created | - | Initialized with 12 critique sections |
