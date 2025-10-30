# Closing Strategies Implementation Guide

> **Enhancement**: Add closing_strategies as 7th taxonomy category to Stage 2.6/2.7
> **Version**: 1.0
> **Date**: 2025-01-29
> **Status**: Implementation Ready
> **Related Documents**:
> - ContentAnalysisCHILD.md (HLD)
> - ContentAnalysisCHILDTI.md (TI)

---

## Table of Contents

2. [Current State vs Target State](#2-current-state-vs-target-state)
3. [Schema Modifications](#3-schema-modifications)
4. [Prompt Modifications](#4-prompt-modifications)
5. [Helper Functions](#5-helper-functions)
6. [Validation Updates](#6-validation-updates)
8. [Testing & Validation](#8-testing--validation)

---

## 1. Overview & Rationale

### 1.2 Technical Context

**Current Taxonomy**: 6 categories
1. content_categories (semantic)
2. hook_strategies (semantic)
3. audience_pain_points (list)
4. trending_keywords (list)
5. engagement_drivers (list)
6. content_tactics (list)

**Target Taxonomy**: 7 categories (add closing_strategies as 3rd semantic category)

### 1.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Semantic category** (name + definition) | Mirrors hook_strategies structure, requires LLM to discover patterns |
| **Single selection** (not multi-select) | Each video has one primary closing approach |
| **Transcript-based** analysis | Classify based on spoken closing patterns (including "no verbal close" as valid pattern) |
| **Last 10 words** analysis | Sufficient to detect CTA patterns without full context |

---

## 2. Current State vs Target State

### 2.1 Taxonomy Structure

**CURRENT** (6 categories):
```json
{
  "hashtag": "nutrition",
  "content_categories": [...],
  "hook_strategies": [...],
  "audience_pain_points": [...],
  "trending_keywords": [...],
  "engagement_drivers": [...],
  "content_tactics": [...]
}
```

**TARGET** (7 categories):
```json
{
  "hashtag": "nutrition",
  "content_categories": [...],
  "hook_strategies": [...],
  "closing_strategies": [             // NEW
    {
      "name": "direct_cta",
      "definition": "Ends with explicit call-to-action directing viewer to link in bio or specific action"
    },
    {
      "name": "cliffhanger_ending",
      "definition": "Creates anticipation for next video by leaving story or information incomplete"
    },
    {
      "name": "question_prompt",
      "definition": "Ends with question to drive comments or engagement"
    }
  ],
  "audience_pain_points": [...],
  "trending_keywords": [...],
  "engagement_drivers": [...],
  "content_tactics": [...]
}
```

### 2.2 Classification Output

**CURRENT** (12 fields):
```json
{
  "video_id": "7526250443832331550",
  "taxonomy_version": "stage2.6_output",
  "content_category": "wellness_practice",
  "hook_strategy": "personal_story",
  "pain_points": ["gut_health", "bloating"],
  "keywords": ["probiotics"],
  "engagement_drivers": ["before_after"],
  "content_tactics": ["direct_address"],
  "caption_analysis": {...},
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

**TARGET** (13 fields):
```json
{
  "video_id": "7526250443832331550",
  "taxonomy_version": "stage2.6_output",
  "content_category": "wellness_practice",
  "hook_strategy": "personal_story",
  "closing_strategy": "direct_cta",      // NEW - required field
  "pain_points": ["gut_health", "bloating"],
  "keywords": ["probiotics"],
  "engagement_drivers": ["before_after"],
  "content_tactics": ["direct_address"],
  "caption_analysis": {...},
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

---

## 3. Schema Modifications

### 3.1 Input Schema: Curated Taxonomy

**File**: `ContentAnalysisCHILDTI.md` Section 3.2 - Input Schema 4

**BEFORE**:
```python
CuratedTaxonomySchema = {
    "hashtag": str,                     # Required
    "content_categories": list[dict],   # Required, Range: 2-10 items
    "hook_strategies": list[dict],      # Required, Range: 2-10 items
    "audience_pain_points": list[str],  # Required, Range: 2-15 items
    "trending_keywords": list[str],     # Required, Range: 2-15 items
    "engagement_drivers": list[str],    # Required, Range: 2-15 items
    "content_tactics": list[str],       # Required, Range: 2-15 items
}
```

**AFTER**:
```python
CuratedTaxonomySchema = {
    "hashtag": str,                     # Required
    "content_categories": list[dict],   # Required, Range: 2-10 items
    "hook_strategies": list[dict],      # Required, Range: 2-10 items
    "closing_strategies": list[dict],   # NEW - Required, Range: 2-10 items
    "audience_pain_points": list[str],  # Required, Range: 2-15 items
    "trending_keywords": list[str],     # Required, Range: 2-15 items
    "engagement_drivers": list[str],    # Required, Range: 2-15 items
    "content_tactics": list[str],       # Required, Range: 2-15 items
}

# Nested schema for closing_strategies items (NEW):
ClosingStrategySchema = {
    "name": str,               # Required, strategy identifier (snake_case)
                               # Examples: "direct_cta", "cliffhanger_ending",
                               #           "social_proof_close", "link_reminder",
                               #           "question_prompt", "teaser_next_video"

    "definition": str,         # Required, >10 chars, human-readable description
                               # Example: "Ends with explicit call-to-action
                               #           directing viewer to link in bio"
}
```

**Validation Rules** (NEW):
```python
# For closing_strategies:
assert len(closing_strategies) >= 2, "Must have at least 2 closing strategies"
assert len(closing_strategies) <= 10, "Maximum 10 closing strategies"
assert all('name' in cs and 'definition' in cs for cs in closing_strategies), "All must have name and definition"
assert all(len(cs['definition']) >= 10 for cs in closing_strategies), "Definitions minimum 10 chars"
assert all(re.match(r'^[a-z0-9_]+$', cs['name']) for cs in closing_strategies), "Names must be snake_case"
```

---

### 3.2 Output Schema: Video Classification

**File**: `ContentAnalysisCHILDTI.md` Section 3.3 - Output Schema 2

**BEFORE**:
```python
VideoClassificationSchema = {
    "video_id": str,                    # Required
    "taxonomy_version": str,            # Required, always "stage2.6_output"
    "content_category": str,            # Required, from taxonomy
    "hook_strategy": str,               # Required, from taxonomy
    "pain_points": list[str],           # Required (can be empty array)
    "keywords": list[str],              # Required (can be empty array)
    "engagement_drivers": list[str],    # Required (can be empty array)
    "content_tactics": list[str],       # Required (can be empty array)
    "caption_analysis": dict,           # Required (8 subfields)
    "confidence": str,                  # Required, ["high", "medium", "low"]
    "transcript_available": bool,       # Required
    "note": str,                        # Optional (can be None)
}
```

**AFTER**:
```python
VideoClassificationSchema = {
    "video_id": str,                    # Required
    "taxonomy_version": str,            # Required, always "stage2.6_output"
    "content_category": str,            # Required, from taxonomy
    "hook_strategy": str,               # Required, from taxonomy
    "closing_strategy": str,            # NEW - Required, from taxonomy
                                        # Primary closing approach
                                        # Example: "direct_cta"
    "pain_points": list[str],           # Required (can be empty array)
    "keywords": list[str],              # Required (can be empty array)
    "engagement_drivers": list[str],    # Required (can be empty array)
    "content_tactics": list[str],       # Required (can be empty array)
    "caption_analysis": dict,           # Required (8 subfields)
    "confidence": str,                  # Required, ["high", "medium", "low"]
    "transcript_available": bool,       # Required
    "note": str,                        # Optional (can be None)
}
```

**Field Count**: 12 → 13 fields

---

### 3.3 Output Schema: Raw Discovery

**File**: `ContentAnalysisCHILDTI.md` Section 3.3 - Output Schema 1

**BEFORE**:
```python
RawDiscoverySchema = {
    "hashtag": str,
    "analysis_date": str,
    "sample_size": int,
    "discovered_patterns": {
        "content_categories": list[dict],   # Frequency-based patterns
        "hook_strategies": list[dict],      # Frequency-based patterns
        "audience_pain_points": list[str],  # Simple list
        "trending_keywords": list[str],     # Simple list
        "engagement_drivers": list[str],    # Simple list
        "content_tactics": list[str]        # Simple list
    }
}
```

**AFTER**:
```python
RawDiscoverySchema = {
    "hashtag": str,
    "analysis_date": str,
    "sample_size": int,
    "discovered_patterns": {
        "content_categories": list[dict],   # Frequency-based patterns
        "hook_strategies": list[dict],      # Frequency-based patterns
        "closing_strategies": list[dict],   # NEW - Frequency-based patterns
        "audience_pain_points": list[str],  # Simple list
        "trending_keywords": list[str],     # Simple list
        "engagement_drivers": list[str],    # Simple list
        "content_tactics": list[str]        # Simple list
    }
}

# Nested schema for closing_strategies patterns (NEW):
DiscoveredClosingStrategySchema = {
    "name": str,                        # Required, strategy identifier
                                        # Example: "direct_cta"

    "frequency": int,                   # Required, count of videos with this pattern
                                        # Example: 28 (out of 50 sampled)

    "percentage": float,                # ADDED BY PYTHON post-LLM (calculate_percentages)
                                        # Example: 56.0

    "examples": list[str],              # Required, 2-3 example phrases
                                        # Example: ["click link in bio", "follow for more tips"]

    "representative_video_ids": list[str],  # Required, video IDs showing this pattern
                                            # Example: ["7526250443832331550", "7428596413707144481"]
}
```

**Category Count**: 6 → 7 categories

---

## 4. Prompt Modifications

### 4.1 Discovery Prompt (Stage 2.6)

**File**: `ml_pipeline/stage2_content_analysis/discovery.py`
**Function**: `discover_patterns_llm()`
**Lines to Modify**: Prompt construction section

**COMPLETE MODIFIED PROMPT**:

```python
def discover_patterns_llm(
    transcripts: list[dict],
    hashtag: str
) -> dict:
    """
    Discover content patterns using LLM (Claude 3.5 Sonnet).

    MODIFIED: Now discovers 7 categories (added closing_strategies)
    """

    # System message (UNCHANGED)
    system_message = """You are an expert content analyst specializing in short-form video patterns. Identify recurring patterns in the transcripts based on frequency and evidence. Be objective and data-driven: report patterns that actually appear in the data, not prescriptive advice. Patterns should be actionable for content creators but grounded in observed behavior."""

    # Main prompt (MODIFIED - added closing_strategies section)
    prompt = f"""Analyze the following {len(transcripts)} video transcripts from the #{hashtag} hashtag.

Your task is to identify recurring content patterns across 7 categories. Focus on patterns that appear in AT LEAST 10% of videos (minimum 5 videos). Do not create patterns for isolated or single-video elements.

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

## CATEGORY 3: Closing Strategies *** NEW CATEGORY ***

Identify HOW videos end. Analyze the LAST 10 WORDS of each transcript to detect ending techniques and calls-to-action.

Examples to show naming style only (these are NOT limits):
- direct_cta: Explicit call-to-action at end
- question_prompt: Ends with question to viewers
- cliffhanger_ending: Creates anticipation for next video

Create NEW closing strategy names based on ending patterns you observe in THIS specific hashtag (#{hashtag}). Focus on the rhetorical technique used to close the video, not specific content. Typically this will be 2-5 closing strategies, but return only as many as you genuinely observe.

---

## CATEGORIES 4-7: Simple Lists

For Categories 4-7 below: Extract phrases (2-4 words) that appear or are IMPLIED in at least 10% of videos (minimum 5). Phrases can be verbatim quotes OR interpretations of what's shown/discussed. Return as simple string lists.

GROUNDING RULE: Every term you list must be traceable to specific transcripts. If you cannot point to at least 5 transcripts showing this pattern, do not include it.

### CATEGORY 4: Audience Pain Points

Identify PROBLEMS, STRUGGLES, or UNMET NEEDS mentioned or implied. Include:
- Explicit problems stated ("I have bloating")
- Implied problems from solutions shown ("I started doing X and Y went away" → Y is pain point)
- Challenges discussed

### CATEGORY 5: Trending Keywords

Identify TOPICS, METHODS, SOLUTIONS, or CONCEPTS mentioned or implied (excluding problems from Category 4). Include:
- Specific terms used repeatedly
- Methods or practices discussed
- Solutions or approaches mentioned

### CATEGORY 6: Engagement Drivers

Identify CONTENT FEATURES or TECHNIQUES that make content compelling (not topics). Include:
- Storytelling devices mentioned or used
- Proof elements described ("I show before/after photos")
- Engagement tactics visible in how creators speak

### CATEGORY 7: Content Tactics

Identify PRESENTATION STYLES or FORMATS mentioned or implied. Include:
- Delivery methods described or evident from speech patterns
- Visual approaches mentioned ("I'm going to show you on screen")
- Structural formats implied by how content flows

---

## OUTPUT FORMAT

Return your analysis as valid JSON with this exact structure:

{{
  "hashtag": "{hashtag}",
  "analysis_date": "{datetime.utcnow().isoformat()}Z",
  "sample_size": {len(transcripts)},
  "discovered_patterns": {{
    "content_categories": [
      {{
        "name": "instructional_walkthrough",
        "frequency": 28,
        "examples": ["step by step tutorial", "here's how to make"],
        "representative_video_ids": ["7526250443832331550", "7428596413707144481"]
      }}
    ],
    "hook_strategies": [
      {{
        "name": "question_hook",
        "frequency": 15,
        "examples": ["did you know that", "have you ever wondered"],
        "representative_video_ids": ["7526250443832331550", "7428596413707144481"]
      }}
    ],
    "closing_strategies": [
      {{
        "name": "direct_cta",
        "frequency": 32,
        "examples": ["click the link in my bio", "follow for more tips like this"],
        "representative_video_ids": ["7526250443832331550", "7428596413707144481"]
      }}
    ],
    "audience_pain_points": ["chronic bloating", "low energy"],
    "trending_keywords": ["protein intake", "gut health"],
    "engagement_drivers": ["before after reveal"],
    "content_tactics": ["direct to camera", "voiceover"]
  }}
}}

Requirements:
- Categories 1-3: Provide 2-3 examples and 2-3 representative_video_ids per pattern
- DO NOT include a "percentage" field - this will be calculated automatically by Python post-processing
- Categories 4-7: Simple string lists only (no objects, no extra fields)

---

## FINAL INSTRUCTIONS

1. Analyze ALL {len(transcripts)} transcripts carefully - every pattern must be grounded in observed data
2. For closing_strategies: Focus on the LAST 10 WORDS of each transcript
3. Return only patterns you genuinely observe - if a category has fewer patterns, that's acceptable
4. Use descriptive snake_case names (short but clear, 2-4 words)
5. Return valid JSON only - no commentary, explanations, or additional text outside the JSON structure

DO NOT make up patterns to fill categories. Quality over quantity.

---

TRANSCRIPTS:

{json.dumps([{{'video_id': t['video_id'], 'text': t['text']}} for t in transcripts], indent=2)}
"""

    # Rest of function implementation (API call, retry logic, etc.) remains UNCHANGED
    # ...
```

**Key Changes**:
1. ✅ Title changed from "6 categories" → "7 categories"
2. ✅ Added complete Category 3: Closing Strategies section (35 lines)
3. ✅ Renumbered Categories 3-6 → 4-7
4. ✅ Added closing_strategies to output format example
5. ✅ Added instruction: "Focus on LAST 10 WORDS of each transcript"

---

### 4.2 Classification Prompt (Stage 2.7)

**File**: `ml_pipeline/stage2_content_analysis/classification.py`
**Function**: `classify_video_llm()`
**Lines to Modify**: Prompt construction section

**COMPLETE MODIFIED PROMPT**:

```python
def classify_video_llm(
    video_id: str,
    transcript: dict,
    caption: str,
    hashtags: list[str],
    taxonomy: dict,
    client: anthropic.Anthropic
) -> dict:
    """
    Classify single video using LLM + taxonomy.

    MODIFIED: Now classifies closing_strategy (13 fields instead of 12)
    """

    # System message (UNCHANGED)
    system_message = """You are an expert content classifier specializing in short-form video analysis. Your task is to accurately classify videos using a predefined taxonomy that was empirically discovered from real video data in this hashtag (Stage 2.6).

Be objective and evidence-based: select classifications that best match the video content based on transcript, caption, and hashtags. Use taxonomy categories EXACTLY as defined - do not reinterpret or expand their meaning. When evidence is ambiguous, note lower confidence rather than forcing a classification."""

    # Build main prompt (MODIFIED - added closing_strategies)
    transcript_text = transcript['text'] if transcript['available'] else "(No transcript available - classify using caption and hashtags)"

    prompt = f"""## ZONE 1: TAXONOMY & CORE CLASSIFICATION

### Provided Taxonomy

**Category 1: Content Categories** (Single Selection)
{json.dumps(taxonomy['content_categories'], indent=2)}

**Category 2: Hook Strategies** (Single Selection)
{json.dumps(taxonomy['hook_strategies'], indent=2)}

**Category 3: Closing Strategies** (Single Selection) *** NEW ***
{json.dumps(taxonomy['closing_strategies'], indent=2)}

**Category 4: Audience Pain Points** (Multiple Selection)
{json.dumps(taxonomy['audience_pain_points'], indent=2)}

**Category 5: Trending Keywords** (Multiple Selection)
{json.dumps(taxonomy['trending_keywords'], indent=2)}

**Category 6: Engagement Drivers** (Multiple Selection)
{json.dumps(taxonomy['engagement_drivers'], indent=2)}

**Category 7: Content Tactics** (Multiple Selection)
{json.dumps(taxonomy['content_tactics'], indent=2)}

---

### Video Data

**Video ID**: {video_id}

**Transcript**:
{transcript_text}

**Caption**:
{caption}

**Hashtags**:
{json.dumps(hashtags)}

---

### Classification Task

Select the best-matching categories from the taxonomy above:

**Categories 1-3: Single Selection (REQUIRED)** *** CHANGED FROM 1-2 ***

**Content Category**: Select exactly ONE category that best describes the primary content format.

**Hook Strategy**: Select exactly ONE strategy that best describes how the video opens.

**Closing Strategy**: Select exactly ONE strategy that best describes how the video ends. *** NEW ***

**IMPORTANT**: You MUST copy the category name EXACTLY as written in the taxonomy. Do not paraphrase, abbreviate, or modify the string. Mismatched spelling or underscores will cause system errors.

**If no perfect match exists**: Select the closest matching category from the taxonomy. Set confidence=low and document the mismatch in the note field (e.g., "Video ends with no clear CTA, closest match is direct_cta").

**String Matching**: Copy category names character-for-character from taxonomy above.

**Categories 4-7: Multiple Selection (0-N)** *** RENUMBERED FROM 3-6 ***

Select ALL applicable items from the taxonomy that are clearly present in the video:

- **Audience Pain Points**: Problems explicitly stated OR strongly implied from solutions discussed
- **Trending Keywords**: Topics/methods explicitly mentioned OR clearly central to the content
- **Engagement Drivers**: Tactics described OR evident from how creator speaks
- **Content Tactics**: Presentation styles explicitly mentioned OR observable from transcript patterns

**GROUNDING RULE**: Only select items that are:
1. **Explicitly mentioned**: Can quote a direct phrase (e.g., "I had bloating")
2. **Strongly implied**: Clear evidence from context (e.g., "I started X and it went away" → X addresses implied problem)

If uncertain whether implication is strong enough, do NOT select. Empty arrays `[]` are acceptable.

**Evidence requirement**: For each selection, you should be able to explain WHY it's selected with specific evidence from transcript or caption.

---

## ZONE 2: CAPTION & HASHTAG ANALYSIS

Analyze caption structure and hashtag strategy as a secondary task after completing Zone 1.

**Caption Hook Type**: How does the caption open? (first 5-10 words)
- statement: Declarative ("This changed my life", "Best product ever")
- question: Interrogative ("Did you know?", "Have you tried?")
- command: Imperative ("Try this now", "Follow for more")
- teaser: Creates curiosity ("You won't believe…", "Wait till the end")

**Call-to-Action**:
- cta_type: link_in_bio, save_post, comment, follow, share, tag_friend, none
- brand_mention_present: Does caption mention a brand/product? (true/false)
- influencer_tag_present: Does caption tag another creator? (true/false)

**Caption Metrics** (simplified levels):
- emoji_usage: none (0), some (1-4), many (5+)
- caption_length: short (<100 chars), long (100+ chars)

**Hashtag Analysis**:
- hashtag_count: Total number of hashtags (integer)
- hashtag_placement: end (all at end), mixed (throughout caption), none

**Note**: Do not attempt to categorize hashtags as broad/niche/branded - this requires view count data not available.

---

## ZONE 3: OUTPUT & CONFIDENCE

### Evidence Handling & Fallback Logic

**Hook Strategy** (required single selection):
- **Primary**: Use transcript opening (first 5-10 words spoken)
- **Fallback**: If transcript empty, use caption opening (first 5-10 words written)
- **Caveat**: Caption opening may not reflect actual video opening - this is acceptable as "best available evidence"

**Closing Strategy** (required single selection): *** NEW ***
- **Primary**: Use transcript ending (last 10 words spoken)
- **If no verbal closing detected**: Classify based on what you observe (e.g., "no_verbal_close" may be a discovered pattern in the taxonomy)
- **Only if transcript completely unavailable**: Use full caption + hashtags to infer closing approach, mark confidence=low
- **Note**: Videos ending with music/silence (no speech) are a valid pattern - do NOT force a CTA classification if none exists verbally

**Content Category** (required single selection):
- **Primary**: Classify from full transcript + caption alignment
- **Fallback**: If transcript empty, classify from caption + hashtags only

**Note Field** (dynamic context for low-confidence scenarios):
- Empty transcript → "Classified from caption/hashtags only - no transcript available"
- Conflicting evidence → "Transcript suggests X, caption suggests Y - selected X (transcript priority)"
- Forced match → "No perfect taxonomy match, selected closest: [category_name]"
- Multiple issues → Combine messages: "No transcript + forced closing match to [category]"

**Evidence Priority** (transparent but enforced):
When evidence conflicts: transcript > caption > hashtags. Document conflicts in note field.
When evidence is weak: still make best-effort classification, but set confidence=low and explain in note.

---

### Confidence Assessment

Assign confidence based on two factors: (1) How well video matches taxonomy, (2) Quality of evidence

**high**:
- Video clearly matches selected categories (no ambiguity in taxonomy fit)
- Strong evidence from transcript and/or caption
- All selections can be justified with explicit phrases or strong implications
- Hook and closing strategies clearly identifiable from transcript

**medium**:
- Video partially matches taxonomy OR selection required inference
- Evidence from transcript OR caption, but not both aligning
- Some selections based on reasonable but not explicit evidence
- Hook or closing strategy identifiable but not perfect match

**low**:
- Forced match for required categories (no perfect taxonomy fit)
- Limited evidence (empty transcript, minimal caption)
- Selections based on weak inference or hashtags alone
- Hook or closing strategy unclear or missing from transcript

**Tie-breakers**:
- If transcript unavailable but caption is rich → can be medium (not automatically low)
- If perfect taxonomy match but only hashtags available → medium (good match, weak evidence)
- If closing has no verbal cue (music/silence) → low confidence for closing_strategy

---

### Output Format

Return a single JSON object with ALL 13 fields present. Do not add fields beyond this schema. *** CHANGED FROM 12 ***

**Required fields** (must be non-null):
- video_id: String (provided in input)
- taxonomy_version: Always use "stage2.6_output"
- content_category: String (exactly one from taxonomy)
- hook_strategy: String (exactly one from taxonomy)
- closing_strategy: String (exactly one from taxonomy) *** NEW ***
- confidence: "high"|"medium"|"low"
- transcript_available: true|false
- note: String with explanation OR null if high confidence

**Multi-select fields** (empty arrays [] allowed):
- pain_points: Array of strings from taxonomy ([] if none apply)
- keywords: Array of strings from taxonomy ([] if none apply)
- engagement_drivers: Array of strings from taxonomy ([] if none apply)
- content_tactics: Array of strings from taxonomy ([] if none apply)

**Caption analysis object** (all 8 subfields required):
- caption_analysis: {{
    hook_type, cta_type, brand_mention_present, influencer_tag_present,
    emoji_usage, caption_length, hashtag_count, hashtag_placement
  }}

**JSON FORMATTING RULES**:
- Use lowercase true/false for booleans (not True/False)
- Always include note field (use null if not needed, don't omit)
- Empty arrays should be [] (not null)
- Copy string values exactly (including underscores and capitalization)
- No additional fields beyond this schema

---

### FINAL INSTRUCTIONS

Before submitting, verify all requirements are met:

**Critical Requirements (System Errors)**
✓ **Exact Strings**: Copy category names character-for-character from taxonomy (e.g., "direct_cta" NOT "direct cta")
   - Mismatched spelling or underscores will cause system error
✓ **Complete Schema**: All 13 fields present (see Output Format section) *** CHANGED FROM 12 ***
✓ **JSON Only**: No text outside JSON structure

**Classification Quality**
✓ **Evidence-Based**: All selections traceable to transcript/caption/hashtags - do not invent patterns
   - Quality over quantity: empty arrays [] better than wrong selections
✓ **Closest Match**: If perfect taxonomy match unclear, select closest category and set confidence=low
✓ **Note Field**: Explain when confidence=low (forced match, missing transcript, conflicts)
✓ **Evidence Priority**: transcript > caption > hashtags (see Zone 3)
✓ **Closing Analysis**: Focus on LAST 10 WORDS of transcript for closing_strategy *** NEW ***

Your classifications feed Stage 7 contrastive analysis - accuracy is critical.
"""

    # Rest of function implementation (API call, retry logic, etc.) remains UNCHANGED
    # ...
```

**Key Changes**:
1. ✅ Added Category 3: Closing Strategies to taxonomy section
2. ✅ Changed "Categories 1-2" → "Categories 1-3" for single selection
3. ✅ Renumbered "Categories 3-6" → "Categories 4-7" for multiple selection
4. ✅ Added closing_strategy to evidence handling section with fallback logic
5. ✅ Added closing_strategy to output format (13 fields)
6. ✅ Added closing analysis instruction: "Focus on LAST 10 WORDS of transcript"
7. ✅ Updated confidence assessment to mention closing strategy identification

---

## 5. Helper Functions

### 5.1 Extract Transcript Ending

**File**: `ml_pipeline/stage2_content_analysis/utils.py` (NEW FUNCTION)

**Purpose**: Extract last N words from transcript for closing analysis

**Implementation**:

```python
import re
from typing import Optional


def extract_transcript_ending(text: str, max_words: int = 10) -> str:
    """
    Extract last N words from transcript for closing strategy analysis.

    HARDENED against edge cases:
    - Ellipsis handling (strips trailing punctuation)
    - Empty/None transcripts (returns "")
    - No punctuation (splits by whitespace only)
    - Multiple whitespace/newlines (normalizes with split())

    Args:
        text: Full transcript text from Whisper
        max_words: Number of words to extract from end (default: 10)

    Returns:
        str: Last N words, or full text if shorter, or empty string if no text

    Edge Cases Tested:
        >>> extract_transcript_ending("Text here...")  # Ellipsis
        "Text here"

        >>> extract_transcript_ending("Word " * 5)  # Exactly N words
        "Word Word Word Word Word"

        >>> extract_transcript_ending("no punctuation here")  # No sentence markers
        "no punctuation here"

        >>> extract_transcript_ending("text  \n\n  more")  # Whitespace
        "text more"
    """
    # DEFENSE 1: Handle empty/None input
    if not text or not text.strip():
        return ""

    # DEFENSE 4: Normalize whitespace (handles multiple spaces, newlines, tabs)
    # .split() with no args splits on ANY whitespace and removes empty strings
    text = text.strip()
    words = text.split()  # Handles "text  \n\n  more" → ["text", "more"]

    if not words:
        return ""

    # DEFENSE 2: Handle transcripts shorter than max_words
    if len(words) <= max_words:
        # Return all words joined (already normalized)
        return " ".join(words)

    # DEFENSE 3: Extract last N words (no regex needed - whitespace already handled)
    ending_words = words[-max_words:]
    ending_text = " ".join(ending_words)

    # DEFENSE 1 (Ellipsis): Strip trailing punctuation (., !, ?, ...)
    # This handles "... text here..." → "text here"
    ending_text = ending_text.rstrip('.!?…')  # Note: … is single char ellipsis

    return ending_text.strip()


def extract_transcript_opening(text: str, max_words: int = 10) -> str:
    """
    Extract first N words from transcript for hook strategy analysis.

    COMPANION FUNCTION to extract_transcript_ending for symmetry.
    Uses same hardened approach (whitespace normalization, no regex).

    Args:
        text: Full transcript text from Whisper
        max_words: Number of words to extract from beginning (default: 10)

    Returns:
        str: First N words, or full text if shorter, or empty string if no text

    Examples:
        >>> extract_transcript_opening("This is why you need to try this. It changed my life.", max_words=10)
        "This is why you need to try this"

        >>> extract_transcript_opening("Short text", max_words=10)
        "Short text"
    """
    # Handle empty or None input
    if not text or not text.strip():
        return ""

    # Normalize whitespace (handles multiple spaces, newlines, tabs)
    text = text.strip()
    words = text.split()  # Split on any whitespace

    if not words:
        return ""

    # Handle transcripts shorter than max_words
    if len(words) <= max_words:
        return " ".join(words)

    # Extract first N words
    opening_words = words[:max_words]
    return " ".join(opening_words)


**Usage in Classification**:

```python
# In classify_video_llm() function (OPTIONAL ENHANCEMENT):
# Add these extractions to prompt for LLM transparency

if transcript['available'] and transcript['text']:
    transcript_ending = extract_transcript_ending(transcript['text'], max_words=10)
    transcript_opening = extract_transcript_opening(transcript['text'], max_words=10)
else:
    transcript_ending = ""
    transcript_opening = ""

# Add to prompt to make hook/closing analysis explicit:
prompt = f"""...
### Video Data

**Video ID**: {video_id}

**Transcript Opening** (first 5-10 words):
{transcript_opening or "(empty)"}

**Transcript Ending** (last 10 words):
{transcript_ending or "(empty)"}

**Full Transcript**:
{transcript_text}

**Caption**:
{caption}
...
"""
```

**Note**: The extraction is OPTIONAL for classification prompt. The LLM can analyze full transcript independently. Including extractions makes the hook/closing analysis more explicit but adds prompt tokens.

---

#### Edge Case Defense Summary

Both helper functions use the **same hardened approach**:

| Edge Case | Defense Mechanism | Code Implementation |
|-----------|------------------|---------------------|
| **1. Ellipsis (...)** | `.rstrip('.!?…')` removes trailing punctuation | `extract_transcript_ending()` line 826 |
| **2. Exact N words** | `if len(words) <= max_words: return all` | Both functions, early return |
| **3. No punctuation** | Use `.split()` only (no regex), works on any text | Both functions, line ~810 |
| **4. Multiple whitespace** | `.split()` with no args normalizes all whitespace | Both functions, line ~810 |
| **5. Empty input** | `if not text or not text.strip(): return ""` | Both functions, first check |

**Key Design Decision**: By using `.split()` with no arguments instead of regex sentence splitting, we:
- ✅ Avoid regex failures on edge cases (ellipsis, no punctuation)
- ✅ Handle all whitespace types automatically (spaces, tabs, newlines)
- ✅ Simplify code (fewer lines, easier to test)
- ✅ Maintain consistency across both extraction functions

---

### 5.2 Validate Closing Strategies in Taxonomy

**File**: `ml_pipeline/stage2_content_analysis/validation.py`

**Function to ADD**:

```python
def validate_closing_strategies_in_taxonomy(taxonomy: dict) -> bool:
    """
    Validate closing_strategies field in curated taxonomy.

    Called by validate_curated_taxonomy() after manual curation.

    Args:
        taxonomy: Curated taxonomy dict with closing_strategies field

    Returns:
        bool: True if valid, raises ValueError if invalid

    Raises:
        ValueError: If validation fails with specific error message

    Validation Rules:
        - closing_strategies field must exist
        - Must be list (not string or dict)
        - Must have at least 2 items (minimum 2 patterns)
        - Must have at most 10 items (recommended maximum)
        - Each item must be dict with 'name' and 'definition'
        - Names must be snake_case (lowercase, numbers, underscores only)
        - Definitions must be >= 10 characters
        - No duplicate names
    """
    import re

    # Rule 1: Field exists
    if 'closing_strategies' not in taxonomy:
        raise ValueError(
            "Taxonomy missing required field: 'closing_strategies'. "
            "Add this field with at least 2 closing strategy patterns."
        )

    closing_strategies = taxonomy['closing_strategies']

    # Rule 2: Must be list
    if not isinstance(closing_strategies, list):
        raise ValueError(
            f"closing_strategies must be array, got {type(closing_strategies)}"
        )

    # Rule 3: Minimum count
    if len(closing_strategies) < 2:
        raise ValueError(
            f"closing_strategies must have at least 2 items, found {len(closing_strategies)}. "
            "Add more closing strategy patterns from raw discovery."
        )

    # Rule 4: Maximum count (warning, not error)
    if len(closing_strategies) > 10:
        logger.warning(
            f"closing_strategies has {len(closing_strategies)} items (recommended maximum: 10). "
            "Consider consolidating similar patterns."
        )

    # Rule 5: Validate each pattern
    for i, strategy in enumerate(closing_strategies):
        # Must be dict
        if not isinstance(strategy, dict):
            raise ValueError(
                f"closing_strategies[{i}] must be object, got {type(strategy)}"
            )

        # Must have 'name' and 'definition'
        if 'name' not in strategy or 'definition' not in strategy:
            raise ValueError(
                f"closing_strategies[{i}] missing 'name' or 'definition'. "
                f"Found fields: {list(strategy.keys())}"
            )

        name = strategy['name']
        definition = strategy['definition']

        # Name must be snake_case
        if not re.match(r'^[a-z0-9_]+$', name):
            raise ValueError(
                f"closing_strategies[{i}] name '{name}' must be snake_case "
                "(lowercase letters, numbers, underscores only). "
                "Example: 'direct_cta' not 'Direct CTA'"
            )

        # Definition minimum length
        if len(definition) < 10:
            raise ValueError(
                f"closing_strategies[{i}] '{name}' definition too short: "
                f"'{definition}' (minimum 10 chars). "
                "Provide clear description of this closing technique."
            )

    # Rule 6: No duplicates
    names = [s['name'] for s in closing_strategies]
    duplicates = [n for n in names if names.count(n) > 1]
    if duplicates:
        raise ValueError(
            f"closing_strategies has duplicate names: {set(duplicates)}. "
            "Each closing strategy must have unique name."
        )

    return True
```

**Integration into validate_curated_taxonomy()**:

```python
def validate_curated_taxonomy(taxonomy_path: str) -> bool:
    """
    Validate manually curated taxonomy file.

    MODIFIED: Now validates 7 categories (added closing_strategies)
    """
    # ... existing code ...

    # Step 4: Validate semantic categories (categories 1-3)  *** CHANGED FROM 1-2 ***
    for category_type in ['content_categories', 'hook_strategies', 'closing_strategies']:  # ADDED closing_strategies
        categories = taxonomy[category_type]

        # ... existing validation logic for semantic categories ...

    # Step 5: Validate simple list categories (categories 4-7)  *** CHANGED FROM 3-6 ***
    for category_type in ['audience_pain_points', 'trending_keywords',
                          'engagement_drivers', 'content_tactics']:
        # ... existing validation logic for simple lists ...

    # Step 6: All validations passed
    logger.info(
        f"✅ Taxonomy validation passed: {taxonomy_path}\n"
        f"   - {len(taxonomy['content_categories'])} content categories\n"
        f"   - {len(taxonomy['hook_strategies'])} hook strategies\n"
        f"   - {len(taxonomy['closing_strategies'])} closing strategies\n"  # ADDED
        f"   - {len(taxonomy['audience_pain_points'])} pain points\n"
        f"   - {len(taxonomy['trending_keywords'])} keywords\n"
        f"   - {len(taxonomy['engagement_drivers'])} engagement drivers\n"
        f"   - {len(taxonomy['content_tactics'])} content tactics"
    )

    return True
```

---

## 6. Validation Updates

### 6.1 Discovery Input Validation

**File**: `ml_pipeline/stage2_content_analysis/validation.py`
**Function**: `validate_discovery_inputs()`

**CHANGE**: Update required field count check

**BEFORE**:
```python
required_patterns = [
    'content_categories', 'hook_strategies', 'audience_pain_points',
    'trending_keywords', 'engagement_drivers', 'content_tactics'
]
# ... check for 6 pattern categories ...
```

**AFTER**:
```python
required_patterns = [
    'content_categories', 'hook_strategies', 'closing_strategies',  # ADDED closing_strategies
    'audience_pain_points', 'trending_keywords', 'engagement_drivers', 'content_tactics'
]
# ... check for 7 pattern categories ...
```

---

### 6.2 Discovery Output Validation

**File**: `ml_pipeline/stage2_content_analysis/validation.py`
**Function**: `validate_discovery_output()`

**CHANGE**: Update category count check

**BEFORE**:
```python
required_patterns = [
    'content_categories', 'hook_strategies', 'audience_pain_points',
    'trending_keywords', 'engagement_drivers', 'content_tactics'
]
patterns = raw_taxonomy['discovered_patterns']
missing = [f for f in required_patterns if f not in patterns]
if missing:
    raise ValueError(f"Discovered patterns missing categories: {missing}")
```

**AFTER**:
```python
required_patterns = [
    'content_categories', 'hook_strategies', 'closing_strategies',  # ADDED
    'audience_pain_points', 'trending_keywords', 'engagement_drivers', 'content_tactics'
]
patterns = raw_taxonomy['discovered_patterns']
missing = [f for f in required_patterns if f not in patterns]
if missing:
    raise ValueError(f"Discovered patterns missing categories: {missing}")

# Additional validation for closing_strategies structure
for category in ['content_categories', 'hook_strategies', 'closing_strategies']:  # ADDED closing_strategies
    for pattern in patterns[category]:
        required_fields = ['name', 'frequency', 'examples']
        missing = [f for f in required_fields if f not in pattern]
        if missing:
            raise ValueError(
                f"Pattern in {category} missing fields: {missing}. Pattern: {pattern}"
            )
```

---

### 6.3 Classification Input Validation

**File**: `ml_pipeline/stage2_content_analysis/validation.py`
**Function**: `validate_classification_inputs()`

**CHANGE**: Validate closing_strategies exists in taxonomy

**BEFORE**:
```python
required_fields = [
    'content_categories', 'hook_strategies', 'audience_pain_points',
    'trending_keywords', 'engagement_drivers', 'content_tactics'
]
missing = [f for f in required_fields if f not in taxonomy]
if missing:
    raise ValueError(f"Taxonomy missing required fields: {missing}")
```

**AFTER**:
```python
required_fields = [
    'content_categories', 'hook_strategies', 'closing_strategies',  # ADDED
    'audience_pain_points', 'trending_keywords', 'engagement_drivers', 'content_tactics'
]
missing = [f for f in required_fields if f not in taxonomy]
if missing:
    raise ValueError(f"Taxonomy missing required fields: {missing}")

# Check closing_strategies is non-empty
if not taxonomy['closing_strategies']:
    raise ValueError("Taxonomy field 'closing_strategies' is empty. Must have at least 2 items.")
```

---

### 6.4 Classification Output Validation

**File**: `ml_pipeline/stage2_content_analysis/validation.py`
**Function**: `validate_classification_output()`

**CHANGE**: Check for closing_strategy field (13 fields total)

**BEFORE**:
```python
required_fields = [
    'video_id', 'taxonomy_version', 'content_category', 'hook_strategy',
    'pain_points', 'keywords', 'engagement_drivers', 'content_tactics',
    'caption_analysis', 'confidence', 'transcript_available', 'note'
]
missing = [f for f in required_fields if f not in classification]
if missing:
    raise ValueError(f"Classification missing required fields: {missing}")
```

**AFTER**:
```python
required_fields = [
    'video_id', 'taxonomy_version', 'content_category', 'hook_strategy',
    'closing_strategy',  # ADDED (field 5 of 13)
    'pain_points', 'keywords', 'engagement_drivers', 'content_tactics',
    'caption_analysis', 'confidence', 'transcript_available', 'note'
]
missing = [f for f in required_fields if f not in classification]
if missing:
    raise ValueError(f"Classification missing required fields: {missing}")

# Additional validation: closing_strategy must be non-empty string
if not classification['closing_strategy'] or not isinstance(classification['closing_strategy'], str):
    raise ValueError(
        f"closing_strategy must be non-empty string, got {type(classification['closing_strategy'])}"
    )
```

---

## 7. Testing & Validation

### 7.1 Unit Tests

**File**: `tests/test_validation_closing_strategies.py` (NEW FILE)

```python
import pytest
from ml_pipeline.stage2_content_analysis.validation import (
    validate_curated_taxonomy,
    validate_classification_output
)
from ml_pipeline.stage2_content_analysis.utils import (
    extract_transcript_ending,
    extract_transcript_opening
)


class TestClosingStrategiesValidation:
    """Test suite for closing_strategies validation."""

    def test_valid_taxonomy_with_closing_strategies(self):
        """Test that valid taxonomy with closing_strategies passes validation."""
        valid_taxonomy = {
            "hashtag": "nutrition",
            "content_categories": [{"name": "recipe", "definition": "Cooking tutorial"}],
            "hook_strategies": [{"name": "question", "definition": "Opens with question"}],
            "closing_strategies": [
                {"name": "direct_cta", "definition": "Explicit call-to-action at end"},
                {"name": "question_prompt", "definition": "Ends with question to viewers"}
            ],
            "audience_pain_points": ["bloating"],
            "trending_keywords": ["protein"],
            "engagement_drivers": ["before_after"],
            "content_tactics": ["voiceover"]
        }

        # Should not raise
        assert validate_curated_taxonomy(valid_taxonomy) is True

    def test_missing_closing_strategies_field(self):
        """Test that taxonomy without closing_strategies raises error."""
        invalid_taxonomy = {
            "hashtag": "nutrition",
            "content_categories": [{"name": "recipe", "definition": "Cooking tutorial"}],
            "hook_strategies": [{"name": "question", "definition": "Opens with question"}],
            # Missing closing_strategies field
            "audience_pain_points": ["bloating"],
            "trending_keywords": ["protein"],
            "engagement_drivers": ["before_after"],
            "content_tactics": ["voiceover"]
        }

        with pytest.raises(ValueError, match="missing required field.*closing_strategies"):
            validate_curated_taxonomy(invalid_taxonomy)

    def test_empty_closing_strategies_array(self):
        """Test that empty closing_strategies array raises error."""
        invalid_taxonomy = {
            "hashtag": "nutrition",
            "content_categories": [{"name": "recipe", "definition": "Cooking tutorial"}],
            "hook_strategies": [{"name": "question", "definition": "Opens with question"}],
            "closing_strategies": [],  # Empty array
            "audience_pain_points": ["bloating"],
            "trending_keywords": ["protein"],
            "engagement_drivers": ["before_after"],
            "content_tactics": ["voiceover"]
        }

        with pytest.raises(ValueError, match="must have at least 2 items"):
            validate_curated_taxonomy(invalid_taxonomy)

    def test_closing_strategy_wrong_naming(self):
        """Test that non-snake_case closing strategy names raise error."""
        invalid_taxonomy = {
            "hashtag": "nutrition",
            "content_categories": [{"name": "recipe", "definition": "Cooking tutorial"}],
            "hook_strategies": [{"name": "question", "definition": "Opens with question"}],
            "closing_strategies": [
                {"name": "Direct CTA", "definition": "Explicit call-to-action at end"},  # Wrong: not snake_case
                {"name": "question_prompt", "definition": "Ends with question"}
            ],
            "audience_pain_points": ["bloating"],
            "trending_keywords": ["protein"],
            "engagement_drivers": ["before_after"],
            "content_tactics": ["voiceover"]
        }

        with pytest.raises(ValueError, match="must be snake_case"):
            validate_curated_taxonomy(invalid_taxonomy)

    def test_valid_classification_with_closing_strategy(self):
        """Test that valid classification with closing_strategy passes."""
        valid_classification = {
            "video_id": "7526250443832331550",
            "taxonomy_version": "stage2.6_output",
            "content_category": "recipe",
            "hook_strategy": "question",
            "closing_strategy": "direct_cta",  # NEW FIELD
            "pain_points": ["bloating"],
            "keywords": ["protein"],
            "engagement_drivers": [],
            "content_tactics": [],
            "caption_analysis": {
                "hook_type": "statement",
                "cta_type": "link_in_bio",
                "brand_mention_present": False,
                "influencer_tag_present": False,
                "emoji_usage": "some",
                "caption_length": "long",
                "hashtag_count": 5,
                "hashtag_placement": "end"
            },
            "confidence": "high",
            "transcript_available": True,
            "note": None
        }

        assert validate_classification_output(valid_classification) is True

    def test_missing_closing_strategy_field_in_classification(self):
        """Test that classification without closing_strategy raises error."""
        invalid_classification = {
            "video_id": "7526250443832331550",
            "taxonomy_version": "stage2.6_output",
            "content_category": "recipe",
            "hook_strategy": "question",
            # Missing closing_strategy field
            "pain_points": [],
            "keywords": [],
            "engagement_drivers": [],
            "content_tactics": [],
            "caption_analysis": {...},
            "confidence": "high",
            "transcript_available": True,
            "note": None
        }

        with pytest.raises(ValueError, match="missing required fields.*closing_strategy"):
            validate_classification_output(invalid_classification)


class TestTranscriptExtractionHelpers:
    """Test suite for transcript ending extraction."""

    def test_extract_ending_from_long_transcript(self):
        """Test extraction from transcript longer than max_words."""
        text = "This is a long transcript with many words. At the end I want to say click the link in my bio for more."
        result = extract_transcript_ending(text, max_words=10)
        assert result == "click the link in my bio for more."

    def test_extract_ending_from_short_transcript(self):
        """Test extraction from transcript shorter than max_words."""
        text = "Short text here."
        result = extract_transcript_ending(text, max_words=10)
        assert result == "Short text here."

    def test_extract_ending_from_empty_transcript(self):
        """Test extraction from empty transcript."""
        result = extract_transcript_ending("", max_words=10)
        assert result == ""

    def test_extract_ending_with_multiple_sentences(self):
        """Test that extraction takes last 10 words regardless of sentences."""
        text = "First sentence here. Second sentence here. Last sentence is the closing one."
        result = extract_transcript_ending(text, max_words=10)
        # Should return last 10 words: "Second sentence here. Last sentence is the closing one"
        # Note: punctuation stripped
        assert "Last sentence is the closing one" in result
```

**Run tests**:
```bash
pytest tests/test_validation_closing_strategies.py -v
```

---

### 7.2 Integration Test

**File**: `tests/test_closing_strategies_integration.py` (NEW FILE)

```python
import pytest
import os
from ml_pipeline.stage2_content_analysis.discovery import discover_patterns_llm
from ml_pipeline.stage2_content_analysis.classification import classify_video_llm


class TestClosingStrategiesIntegration:
    """Integration test for closing_strategies end-to-end flow."""

    @pytest.fixture
    def sample_transcripts(self):
        """Sample transcripts with clear closings."""
        return [
            {
                "video_id": "7111111111111111111",
                "text": "Here's how to make a protein smoothie. Add banana and protein powder. Blend it up. Click the link in my bio for the recipe.",
                "bucket": "33_60s"
            },
            {
                "video_id": "7222222222222222222",
                "text": "Let me show you my morning routine. I wake up at 6am and do yoga. Follow for more tips like this.",
                "bucket": "33_60s"
            },
            {
                "video_id": "7333333333333333333",
                "text": "This supplement changed my life. I take it every morning. What do you think? Let me know in the comments.",
                "bucket": "60_90s"
            }
        ]

    def test_discovery_returns_closing_strategies(self, sample_transcripts):
        """Test that discovery finds closing_strategies category."""
        # Skip if no API key (CI environment)
        if not os.environ.get('ANTHROPIC_API_KEY'):
            pytest.skip("ANTHROPIC_API_KEY not set")

        result = discover_patterns_llm(sample_transcripts, hashtag="test")

        # Check output structure
        assert 'discovered_patterns' in result
        assert 'closing_strategies' in result['discovered_patterns']

        # Check closing_strategies is list of dicts with correct fields
        closing_strategies = result['discovered_patterns']['closing_strategies']
        assert isinstance(closing_strategies, list)
        assert len(closing_strategies) >= 1

        for strategy in closing_strategies:
            assert 'name' in strategy
            assert 'frequency' in strategy
            assert 'examples' in strategy
            assert 'representative_video_ids' in strategy

    def test_classification_returns_closing_strategy_field(self, sample_transcripts):
        """Test that classification includes closing_strategy field."""
        # Skip if no API key
        if not os.environ.get('ANTHROPIC_API_KEY'):
            pytest.skip("ANTHROPIC_API_KEY not set")

        # Mock taxonomy with closing_strategies
        taxonomy = {
            "content_categories": [{"name": "tutorial", "definition": "How-to content"}],
            "hook_strategies": [{"name": "direct_statement", "definition": "States topic upfront"}],
            "closing_strategies": [
                {"name": "direct_cta", "definition": "Explicit call-to-action"},
                {"name": "question_prompt", "definition": "Ends with question"}
            ],
            "audience_pain_points": ["confusion"],
            "trending_keywords": ["protein"],
            "engagement_drivers": ["tips"],
            "content_tactics": ["voiceover"]
        }

        # Classify first sample video
        video = sample_transcripts[0]
        transcript = {"text": video['text'], "available": True}

        result = classify_video_llm(
            video_id=video['video_id'],
            transcript=transcript,
            caption="",
            hashtags=[],
            taxonomy=taxonomy,
            client=None  # Will be initialized in function
        )

        # Check closing_strategy field exists and is valid
        assert 'closing_strategy' in result
        assert result['closing_strategy'] in ['direct_cta', 'question_prompt']
        assert result['confidence'] in ['high', 'medium', 'low']
```

---

## 8. Implementation Checklist

### 8.1 Pre-Implementation

- [ ] **Feasibility Study**: Sample 100 transcripts, verify ≥60% have clear closings
- [ ] **Stakeholder Approval**: Get sign-off on +7% cost increase
- [ ] **Backup Current System**: Tag codebase, backup data (Section 10.1)

---

### 8.2 Code Changes

#### Schema Updates
- [ ] Update `CuratedTaxonomySchema` - add closing_strategies field (Section 3.1)
- [ ] Update `VideoClassificationSchema` - add closing_strategy field (Section 3.2)
- [ ] Update `RawDiscoverySchema` - add closing_strategies to discovered_patterns (Section 3.3)

#### Prompt Updates
- [ ] Modify `discover_patterns_llm()` prompt - add Category 3: Closing Strategies (Section 4.1)
- [ ] Modify `classify_video_llm()` prompt - add closing_strategies taxonomy + output field (Section 4.2)
- [ ] Test prompts manually with Claude API (3 sample transcripts)

#### Helper Functions
- [ ] Add `extract_transcript_ending()` to utils.py (Section 5.1)
- [ ] Add `extract_transcript_opening()` to utils.py (Section 5.1)
- [ ] Write unit tests for extraction functions (Section 7.1)

#### Validation Updates
- [ ] Update `validate_discovery_inputs()` - check 7 categories (Section 6.1)
- [ ] Update `validate_discovery_output()` - check 7 categories (Section 6.2)
- [ ] Update `validate_classification_inputs()` - check closing_strategies exists (Section 6.3)
- [ ] Update `validate_classification_output()` - check 13 fields (Section 6.4)
- [ ] Update `validate_curated_taxonomy()` - validate closing_strategies structure (Section 6)
- [ ] Write validation tests (Section 8.1)

---

### 8.3 Testing

#### Unit Tests
- [ ] Run all validation tests: `pytest tests/test_validation_closing_strategies.py -v`
- [ ] Run extraction helper tests: `pytest tests/test_transcript_extraction.py -v`
- [ ] Verify 100% pass rate

#### Integration Test
- [ ] Run discovery on 1 test hashtag (e.g., #nutrition)
- [ ] Verify raw discovery contains closing_strategies with 2-5 patterns
- [ ] Manually curate taxonomy
- [ ] Run taxonomy validation: `validate_curated_taxonomy(taxonomy_path)`
- [ ] Run classification on 120 videos
- [ ] Verify all classifications have closing_strategy field
- [ ] Check confidence distribution (target: >70% high)

#### Manual Spot-Check
- [ ] Sample 20 random videos from classification output
- [ ] For each: read transcript ending + verify classified closing_strategy
- [ ] Calculate accuracy (target: ≥80%)
- [ ] Document false positives/negatives in report

---

### 8.4 Documentation

- [ ] Update ContentAnalysisCHILD.md (HLD) - add closing_strategies to Section 5 schemas
- [ ] Update ContentAnalysisCHILDTI.md (TI) - reflect schema changes in Sections 3, 4, 5, 6
- [ ] Update QUICK_REFERENCE.md - mention 7 categories instead of 6
- [ ] Update README - add closing_strategies to feature list
- [ ] Create this ClosingStrategies.md implementation guide
- [ ] Add inline code comments for new functions

---

### 8.5 Deployment

- [ ] Code review with team
- [ ] Merge feature branch to main
- [ ] Deploy to staging environment
- [ ] Run end-to-end test on staging (1 full hashtag pipeline)
- [ ] Monitor logs for errors
- [ ] Deploy to production
- [ ] Monitor first 5 hashtag runs in production
- [ ] Verify zero schema errors

---

### 8.6 Post-Deployment Validation

- [ ] Run 5 hashtags through full pipeline (discovery + classification)
- [ ] Verify closing_strategies discovered for each hashtag
- [ ] Check average confidence (should be >70% high)
- [ ] Review manual curator feedback (is closing_strategies curation easy?)
- [ ] Monitor API costs (should be +7% as predicted)
- [ ] Monitor latency (should be +9% as predicted)

---

### 11.7 Rollback Criteria

**Trigger rollback if any of these occur:**

- [ ] ❌ Schema validation errors in production (>5% of videos)
- [ ] ❌ LLM discovery consistently fails (>20% timeouts)
- [ ] ❌ Classification confidence <50% high (data quality issue)
- [ ] ❌ Manual curator reports closing_strategies too ambiguous
- [ ] ❌ API costs exceed +15% (should be ~7%)
