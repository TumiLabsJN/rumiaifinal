# Content Analysis - High-Level Design

> **Parent**: MLPlanningv2.md - Stages 2.6 & 2.7
> **Version**: 1.0
> **Last Updated**: 2025-10-14
> **Status**: Draft

---

## 1. Context & Business Goal

### 1.1 What Problem Does This Solve?

RumiAI extracts 60+ quantitative ML features (eye_contact_rate, scene_count, energy_level) from TikTok videos, but rich qualitative content data (transcripts, captions, hashtags) cannot be effectively fed to Random Forest or K-Means without lossy transformations (TF-IDF, embeddings). This results in missing critical creative insights like "60% of top videos use 'problem_solution' hook strategy" or "videos with 'vulnerability_shown' tactic have 2x engagement."

Content Analysis solves this by creating a semi-structured analysis layer that preserves semantic richness. Using LLM-powered taxonomy-based classification, it extracts observable content patterns (hook strategies, engagement drivers, caption tactics) that complement ML features, enabling Stage 7 to generate comprehensive creative reports with both quantitative ("eye_contact_rate importance: 0.23") and qualitative ("problem_solution hook used 3x more in top videos") insights.

### 1.2 Where This Fits in Pipeline

**Foundation Dependencies**: This component depends on MLPlanningv2.md Part 1 for:
- Client directory structure (Section: Client Architecture & Storage)
- CLI parameter definitions (Section: CLI Command Structure)
- Configuration schemas (Section: Configuration Schemas)

```
Stage 2: Video Processing (RumiAI Pipeline)
   ↓ Output: insights/{video_id}_temporal_windows_updated.json (N files)
   ↓         unified_analysis/{video_id}.json (metadata + timeline)
   ↓         speech_transcriptions/{video_id}_whisper.json (transcripts)
Stage 2.5: Bucket Selection
   ↓ Output: selection_manifest.json (top 3 buckets, video lists)
   ↓         ml_training_data/{hashtag}/bucket_{duration}/ (organized features)
Stage 2.6: Content Analysis Discovery [THIS COMPONENT - Discovery]
   ↓ Output: content_taxonomies/{hashtag}_raw_discovery.json
   ↓         content_taxonomies/{hashtag}_taxonomy.json (after manual curation)
Stage 2.7: Content Analysis Classification [THIS COMPONENT - Classification]
   ↓ Output: content_analysis/{video_id}_content.json (120 files: 40 per bucket × 3 buckets)
Stage 3-6: ML Training (Random Forest + K-Means)
Stage 7: LLM Report Generation (consumes both ML + content insights)
```

### 1.3 Success Criteria

- [ ] Stage 2.6 completes discovery in < 120 seconds for 50 transcripts (Sonnet API)
- [ ] Stage 2.7 classifies 120 videos in < 15 minutes (Haiku API, 40 per bucket × 3 buckets)
- [ ] Taxonomy schema validates: all 6 fields present (content_categories, hook_strategies, audience_pain_points, trending_keywords, engagement_drivers, content_tactics)
- [ ] Classification output includes complete schema: 12 fields total (6 core + 1 caption_analysis object with 8 subfields + 5 metadata fields)
- [ ] Classification uses refined prompt with 3-zone structure and grounding rules
- [ ] Enables contrastive analysis: Stage 7 can query "60% of top use X vs 20% of bottom"
- [ ] Zero data loss on API failures: 3 retries with exponential backoff, fail-fast with clear error
- [ ] File paths follow ML pipeline architecture: `/data/clients/{client_id}/hashtags/{cluster_id}/top_contrastive/`

---

## 2. Architecture & Design

### 2.1 High-Level Approach

**Two-Stage Pipeline with Human-in-the-Loop**:

**Stage 2.6 (Discovery - One-Time per Hashtag)**:
Sample 50 transcripts (stratified even across top 3 buckets, top performers only), use Claude 3.5 Sonnet to discover natural content patterns (categories, hooks, pain points), output raw discovery JSON with examples/frequencies. Human curator reviews, filters noise (e.g., remove patterns <10% frequency), refines terminology, saves curated taxonomy. Cost: ~$0.75 per hashtag.

**Stage 2.7 (Classification - Every Run)**:
Load curated taxonomy, classify 120 videos (20 top + 20 bottom per bucket × 3 buckets) using Claude 3 Haiku. For each video: extract transcript (speech_transcriptions/), caption + hashtags (unified_analysis/), classify using taxonomy, output structured JSON with content_category, hook_strategy, engagement_drivers, content_tactics, caption_analysis. Cost: ~$0.12 per hashtag.

**Key Design Principles**:
- Expensive smart model (Sonnet) for creative discovery, cheap fast model (Haiku) for repetitive classification (15x cost savings)
- Manual gate between stages ensures taxonomy quality (aligned with business goals)
- Two-step execution: `--stop-after discovery` → manual curation → `--resume-from classification`
- Graceful degradation: empty transcripts handled via caption/hashtag-only classification
- Fail-fast with checkpointing: API failures retry 3x, then stop for investigation

### 2.2 Data Flow

```
Stage 2.6 Discovery:
Input: selection_manifest.json (from Stage 2.5)
       → Top 3 buckets list, video IDs per bucket
       50 transcripts sampled (17 per bucket, stratified even)
       speech_transcriptions/{video_id}_whisper.json → "text" field
   ↓
Process: LLM Discovery (Sonnet, 30s)
   → Analyze 50 transcripts
   → Identify patterns: categories, hooks, pain points, keywords, drivers, tactics
   → Return JSON with name, frequency, examples per pattern
   ↓
Output: content_taxonomies/{hashtag}_raw_discovery.json (~10KB)
        → Automated pattern discovery output
   ↓
[MANUAL STEP]: Human curator reviews, filters, refines
   ↓
Output: content_taxonomies/{hashtag}_taxonomy.json (~5KB)
        → Curated, production-ready taxonomy

Stage 2.7 Classification:
Input: content_taxonomies/{hashtag}_taxonomy.json (from Stage 2.6)
       selection_manifest.json (video IDs to classify)
       speech_transcriptions/{video_id}_whisper.json → "text"
       unified_analysis/{video_id}.json → "metadata.description", "metadata.hashtags"
   ↓
Process: LLM Classification (Haiku, 5 min for 120 videos)
   → For each video (40 per bucket × 3 buckets):
      → Load transcript + caption + hashtags
      → Classify using taxonomy with 3-zone prompt structure
      → Structure output: 12 fields (6 core + 1 caption_analysis object with 8 subfields + 5 metadata)
   ↓
Output: bucket_{duration}/content_analysis/{video_id}_content.json × 120
        → Per-bucket classification files (~2KB each)
```

### 2.3 Detailed Process

#### Step 2.3.1: Stage 2.6 - Discovery Sampling

**Purpose**: Select 50 representative transcripts from top performers across top 3 buckets for pattern discovery

**Logic**:
```python
# Source: QA Q8 (sampling strategy decisions)
def sample_transcripts_for_discovery(manifest_path, sample_size=50):
    """
    Sample transcripts stratified evenly across top 3 buckets.

    Args:
        manifest_path: Path to selection_manifest.json from Stage 2.5
        sample_size: Total transcripts to sample (default: 50, configurable)

    Returns:
        list: Sampled video IDs with transcript text
    """
    # Load manifest (Source: QA Q2 - Option B: Stage 2.5 outputs manifest)
    manifest = load_json(manifest_path)
    top_3_buckets = manifest['selected_buckets']  # e.g., ["33_60s", "60_90s", "90_120s"]

    # Stratified even sampling (Source: QA Q8 Point 2 - Option B)
    samples_per_bucket = sample_size // 3  # ~17 per bucket

    sampled_transcripts = []
    for bucket in top_3_buckets:
        # Sample from top performers only (Source: QA Q8 Point 3 - Option A)
        top_performers = manifest['videos_by_bucket'][bucket]['top_performers']

        # Random sample
        sampled_ids = random.sample(top_performers, min(samples_per_bucket, len(top_performers)))

        # Load transcripts (Source: QA Q3 - transcript schema)
        for video_id in sampled_ids:
            transcript_path = f"/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json"
            transcript_data = load_json(transcript_path)
            sampled_transcripts.append({
                "video_id": video_id,
                "text": transcript_data['text'],  # Complete transcript
                "bucket": bucket
            })

    return sampled_transcripts
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Bucket has < 17 videos | Sample all available | Rare (buckets typically have 40-80 videos after selection) |
| Transcript file missing | Skip video, log warning | Fail gracefully, use available transcripts |
| Empty transcript (no speech) | Include in sample | May reveal "no-speech" content patterns |
| Sample size not divisible by 3 | Remainder distributed to first buckets | 50÷3 = 16, 17, 17 |

#### Step 2.3.2: Stage 2.6 - LLM Discovery

**Purpose**: Use Claude 3.5 Sonnet to discover natural content patterns from 50 transcripts

**Logic**:
```python
# Source: QA Q7 (LLM API configuration), QA Q10 (taxonomy schema)
def discover_patterns_llm(transcripts, hashtag):
    """
    Discover content patterns using LLM.

    Args:
        transcripts: List of transcript dicts with video_id, text, bucket
        hashtag: str, hashtag name (e.g., "nutrition")

    Returns:
        dict: Raw discovery JSON with patterns, frequencies, examples
    """
    # Prepare prompt (Source: 2.6HashtagCritique.md - Final Prompt)
    # Note: System message configured separately in API call
    system_message = """You are an expert content analyst specializing in short-form video patterns. Identify recurring patterns in the transcripts based on frequency and evidence. Be objective and data-driven: report patterns that actually appear in the data, not prescriptive advice. Patterns should be actionable for content creators but grounded in observed behavior."""

    prompt = f"""Analyze the following {len(transcripts)} video transcripts from the #{hashtag} hashtag.

Your task is to identify recurring content patterns across 6 categories. Focus on patterns that appear in AT LEAST 10% of videos (minimum 3 videos). Do not create patterns for isolated or single-video elements.

Patterns should be specific and actionable so content creators can replicate them.

[Complete prompt details - see 2.6HashtagCritique.md Final Prompt section or TI Section 4.2 for full prompt text]

Return JSON with structure:
{{
  "hashtag": "{hashtag}",
  "analysis_date": "{datetime.utcnow().isoformat()}Z",
  "sample_size": {len(transcripts)},
  "discovered_patterns": {{
    "content_categories": [{{"name": "...", "frequency": N, "examples": [...], "representative_video_ids": [...]}}],
    "hook_strategies": [...],
    "audience_pain_points": [...],
    "trending_keywords": [...],
    "engagement_drivers": [...],
    "content_tactics": [...]
  }}
}}

Note: Do NOT include percentage field - calculated by Python post-processing (see TI Section 4.2.5)

Transcripts:
{json.dumps([{{'video_id': t['video_id'], 'text': t['text']} for t in transcripts])}
"""

    # Call Anthropic API (Source: 2.6HashtagCritique.md - Final Prompt)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        timeout=120,  # Source: QA Q11 - 120s timeout
        system=system_message,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response
    raw_taxonomy = json.loads(response.content[0].text)

    # Save raw discovery (Source: QA Q12 - file paths)
    output_path = f"/data/clients/{{client_id}}/hashtags/{hashtag}/top_contrastive/content_taxonomies/{hashtag}_raw_discovery.json"
    save_json(output_path, raw_taxonomy)

    logger.info(f"✅ Discovery complete: {output_path}")
    logger.info(f"📝 Next: Manually curate and save to content_taxonomies/{hashtag}_taxonomy.json")

    return raw_taxonomy
```

**Note on Percentage Field**: The LLM returns discovery JSON without the `percentage` field. Python post-processing calculates percentages using the `calculate_percentages()` function (see TI Section 4.2.5) before saving to disk. This ensures:
- No LLM math errors (deterministic calculation)
- Validation that frequency ≤ sample_size (detects hallucination)
- Consistent rounding to 1 decimal place

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| LLM timeout (>120s) | Retry 3x with backoff (1s, 2s, 4s), then fail | Source: QA Q7, Q11 - fail-fast after retries |
| Invalid JSON response | Retry 3x, then fail with clear error | LLM occasionally produces malformed JSON |
| Very low pattern frequency (<5%) | Include in raw output, curator filters | Human decides what's actionable |
| Patterns missing a field | Log warning, include partial data | Curator can fix during manual review |

#### Step 2.3.3: Stage 2.6 - Manual Curation (External to Pipeline)

**Purpose**: Human reviews raw discovery, filters noise, refines terminology, saves production taxonomy

**Process** (manual, ~2 hours per hashtag):
1. Review `raw_discoveries/{hashtag}_raw.json`
2. Filter patterns: remove <10% frequency, brand-specific, hyper-granular categories
3. Refine names: "videos_mentioning_coffee" → remove, "recipe_tutorial" → keep
4. Add definitions for semantic categories (content_categories, hook_strategies)
5. Save curated taxonomy to `content_taxonomies/{hashtag}_taxonomy.json`

**Curated Taxonomy Schema** (Source: QA Q10 - Option B hybrid):
```json
{
  "hashtag": "nutrition",
  "content_categories": [
    {"name": "recipe_tutorial", "definition": "Step-by-step cooking instructions"},
    {"name": "supplement_review", "definition": "Product reviews for supplements"}
  ],
  "hook_strategies": [
    {"name": "problem_solution", "definition": "Starts with problem, promises solution"},
    {"name": "direct_statement", "definition": "Opens with bold declarative fact"}
  ],
  "audience_pain_points": ["bloating", "low_energy"],
  "trending_keywords": ["protein", "gut_health"],
  "engagement_drivers": ["before_after_reveal", "specific_metrics_mentioned"],
  "content_tactics": ["personal_story", "direct_to_camera"]
}
```

#### Step 2.3.4: Stage 2.7 - Video Classification

**Purpose**: Classify 120 videos (40 per bucket × 3) using curated taxonomy

**Logic**:
```python
# Source: QA Q9 (classification scope), QA Q6 (output schema), QA Q7 (LLM config)
def classify_videos(manifest_path, taxonomy_path, hashtag, client_id):
    """
    Classify videos using saved taxonomy.

    Args:
        manifest_path: Path to selection_manifest.json
        taxonomy_path: Path to curated taxonomy JSON
        hashtag: str, hashtag name
        client_id: str, client ID for file paths

    Returns:
        int: Number of videos classified
    """
    # Load taxonomy (Source: QA Q10)
    taxonomy = load_json(taxonomy_path)

    # Load manifest (Source: QA Q2)
    manifest = load_json(manifest_path)
    top_3_buckets = manifest['selected_buckets']

    # Initialize Anthropic client (Source: QA Q7)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    classified_count = 0

    for bucket in top_3_buckets:
        # Select 20 top + 20 bottom per bucket (Source: QA Q9 - Option B)
        videos_to_classify = (
            manifest['videos_by_bucket'][bucket]['top_performers'][:20] +
            manifest['videos_by_bucket'][bucket]['bottom_performers'][:20]
        )

        for video_id in videos_to_classify:
            # Check cache (Source: QA Q7 - avoid re-classification)
            output_path = f"/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/buckets/bucket_{bucket}/content_analysis/{video_id}_content.json"
            if os.path.exists(output_path):
                logger.info(f"⏭️  Skipping {video_id} (cached)")
                continue

            # Load input data (Source: QA Q3, Q5)
            transcript = load_transcript(video_id)  # Returns {"text": "...", "available": True/False}
            caption, hashtags = load_caption_and_hashtags(video_id)

            # Classify with LLM
            classification = classify_video_llm(
                video_id=video_id,
                transcript=transcript,
                caption=caption,
                hashtags=hashtags,
                taxonomy=taxonomy,
                client=client
            )

            # Save output (Source: QA Q12)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_json(output_path, classification)
            logger.info(f"✅ Classified: {video_id}")

            classified_count += 1

            # Inter-request delay (Source: QA Q7 - 0.5s safety buffer)
            time.sleep(0.5)

    return classified_count


def load_transcript(video_id):
    """
    Load transcript from speech_transcriptions/.
    Source: QA Q3 (transcript input schema)
    """
    path = f"/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json"
    try:
        data = load_json(path)
        text = data.get('text', '')
        return {
            "text": text,
            "available": len(text) > 0
        }
    except FileNotFoundError:
        logger.warning(f"Transcript not found: {video_id}")
        return {"text": "", "available": False}


def load_caption_and_hashtags(video_id):
    """
    Load caption and hashtags from unified_analysis/.
    Source: QA Q5 (caption/hashtag input schema)
    """
    path = f"/home/jorge/rumiaifinal/unified_analysis/{video_id}.json"
    try:
        data = load_json(path)
        caption = data.get('metadata', {}).get('description', '')
        hashtags_array = data.get('metadata', {}).get('hashtags', [])
        hashtag_names = [h.get('name', '') for h in hashtags_array if h.get('name')]
        return caption, hashtag_names
    except FileNotFoundError:
        logger.warning(f"Unified analysis not found: {video_id}")
        return "", []


def classify_video_llm(video_id, transcript, caption, hashtags, taxonomy, client):
    """
    Classify single video using LLM + taxonomy.
    Source: 2.7ClassificationCritique.md (Final Refined Prompt)
    """
    # System message (Source: 2.7ClassificationCritique.md Section 1)
    system_message = """You are an expert content classifier specializing in short-form video analysis. Your task is to accurately classify videos using a predefined taxonomy that was empirically discovered from real video data in this hashtag (Stage 2.6).

Be objective and evidence-based: select classifications that best match the video content based on transcript, caption, and hashtags. Use taxonomy categories EXACTLY as defined - do not reinterpret or expand their meaning. When evidence is ambiguous, note lower confidence rather than forcing a classification."""

    # Build user prompt with 3-zone structure
    prompt = f"""## ZONE 1: TAXONOMY & CORE CLASSIFICATION

### Provided Taxonomy

**Category 1: Content Categories** (Single Selection)
{json.dumps(taxonomy['content_categories'], indent=2)}

**Category 2: Hook Strategies** (Single Selection)
{json.dumps(taxonomy['hook_strategies'], indent=2)}

**Category 3: Audience Pain Points** (Multiple Selection)
{json.dumps(taxonomy['audience_pain_points'])}

**Category 4: Trending Keywords** (Multiple Selection)
{json.dumps(taxonomy['trending_keywords'])}

**Category 5: Engagement Drivers** (Multiple Selection)
{json.dumps(taxonomy['engagement_drivers'])}

**Category 6: Content Tactics** (Multiple Selection)
{json.dumps(taxonomy['content_tactics'])}

---

### Video Data

**Video ID**: {video_id}

**Transcript**:
{transcript['text'] if transcript['available'] else "(No transcript available - classify using caption and hashtags)"}

**Caption**:
{caption}

**Hashtags**:
{json.dumps(hashtags)}

---

### Classification Task

Select the best-matching categories from the taxonomy above:

**Categories 1-2: Single Selection (REQUIRED)**

**Content Category**: Select exactly ONE category that best describes the primary content format.

**Hook Strategy**: Select exactly ONE strategy that best describes how the video opens.

**IMPORTANT**: You MUST copy the category name EXACTLY as written in the taxonomy. Do not paraphrase, abbreviate, or modify the string. Mismatched spelling or underscores will cause system errors.

**If no perfect match exists**: Select the closest matching category from the taxonomy. Set confidence=low and document the mismatch in the note field (e.g., "Video is comedy skit, closest match is wellness_practice").

**String Matching**: Copy category names character-for-character from taxonomy above.

**Categories 3-6: Multiple Selection (0-N)**

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

---

## ZONE 3: OUTPUT & CONFIDENCE

### Evidence Handling & Fallback Logic

**Hook Strategy** (required single selection):
- **Primary**: Use transcript opening (first 5-10 words spoken)
- **Fallback**: If transcript empty, use caption opening (first 5-10 words written)
- **Caveat**: Caption opening may not reflect actual video opening - this is acceptable as "best available evidence"

**Content Category** (required single selection):
- **Primary**: Classify from full transcript + caption alignment
- **Fallback**: If transcript empty, classify from caption + hashtags only

**Note Field** (dynamic context for low-confidence scenarios):
- Empty transcript → "Classified from caption/hashtags only - no transcript available"
- Conflicting evidence → "Transcript suggests X, caption suggests Y - selected X (transcript priority)"
- Forced match → "No perfect taxonomy match, selected closest: [category_name]"
- Multiple issues → Combine messages: "No transcript + forced match to [category]"

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

**medium**:
- Video partially matches taxonomy OR selection required inference
- Evidence from transcript OR caption, but not both aligning
- Some selections based on reasonable but not explicit evidence

**low**:
- Forced match for required categories (no perfect taxonomy fit)
- Limited evidence (empty transcript, minimal caption)
- Selections based on weak inference or hashtags alone

**Tie-breakers**:
- If transcript unavailable but caption is rich → can be medium (not automatically low)
- If perfect taxonomy match but only hashtags available → medium (good match, weak evidence)

---

### Output Format

Return a single JSON object with ALL 12 fields present. Do not add fields beyond this schema.

**Required fields** (must be non-null):
- video_id: String (provided in input)
- taxonomy_version: Always use "stage2.6_output"
- content_category: String (exactly one from taxonomy)
- hook_strategy: String (exactly one from taxonomy)
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
✓ **Exact Strings**: Copy category names character-for-character from taxonomy (e.g., "wellness_practice" NOT "wellness")
   - Mismatched spelling or underscores will cause system error
✓ **Complete Schema**: All 12 fields present (see Output Format section)
✓ **JSON Only**: No text outside JSON structure

**Classification Quality**
✓ **Evidence-Based**: All selections traceable to transcript/caption/hashtags - do not invent patterns
   - Quality over quantity: empty arrays [] better than wrong selections
✓ **Closest Match**: If perfect taxonomy match unclear, select closest category and set confidence=low
✓ **Note Field**: Explain when confidence=low (forced match, missing transcript, conflicts)
✓ **Evidence Priority**: transcript > caption > hashtags (see Zone 3)

Your classifications feed Stage 7 contrastive analysis - accuracy is critical.
    """

    # Call API with retry logic (Source: QA Q7 - 3 retries with backoff)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Source: QA Q7 - Haiku for classification
                max_tokens=1024,
                timeout=30,  # Source: QA Q11 - 30s per-video timeout
                system=system_message,  # Source: 2.7ClassificationCritique.md Section 1
                messages=[{"role": "user", "content": prompt}]
            )

            classification = json.loads(response.content[0].text)
            return classification

        except (TimeoutError, anthropic.APIError) as e:
            if attempt < 2:
                delay = [1, 2, 4][attempt]  # Exponential backoff
                logger.warning(f"API failed for {video_id}, retry {attempt+1} in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"API failed for {video_id} after 3 retries")
                raise  # Re-raise after final retry
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Empty transcript | Classify using caption/hashtags only, set transcript_available=false | Source: QA Q4 - Option B |
| Missing caption | Use empty string, classification uses transcript + hashtags | Captions are optional |
| Missing hashtags | Use empty array, classification uses transcript + caption | Hashtags are optional |
| LLM timeout (>30s per video) | Retry 3x with backoff (1s, 2s, 4s), then fail | Source: QA Q7, Q11 |
| Invalid JSON response | Retry 3x, then fail with error | LLM occasionally produces malformed JSON |
| Classification exceeds 15min total | Fail with timeout error, investigate API slowness | Source: QA Q11 - 15min overall timeout |

---

## 3. Dependencies & Integration

### 3.1 Input Dependencies

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **System setup** | MLPlanningv2.md Part 1 (Client Architecture) | Directory structure | `/data/clients/{client_id}/hashtags/{cluster_id}/` | Fail-fast if directories don't exist |
| **Selection manifest** | Stage 2.5 output | JSON | `selected_buckets`, `videos_by_bucket`, `top_performers`, `bottom_performers` | Fail-fast: "selection_manifest.json not found. Run Stage 2.5 first." |
| **Transcripts** | Stage 2 (Whisper service) | JSON | `text` field (string, can be empty) | Gracefully handle missing/empty: classify using caption/hashtags only |
| **Captions** | Stage 2 (unified_analysis) | JSON | `metadata.description` (string, can be empty) | Gracefully handle missing: use empty string |
| **Hashtags** | Stage 2 (unified_analysis) | JSON | `metadata.hashtags` (array of objects with `name` field) | Gracefully handle missing: use empty array |
| **Environment variable** | System config | String | `ANTHROPIC_API_KEY` | Fail-fast: "ANTHROPIC_API_KEY not set" |

### 3.2 Output Contracts

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| **Raw discovery** | JSON (~10KB) | `discovered_patterns` with 6 categories, each with name/frequency/examples/video_ids | Human curator (manual review) | Check all 6 pattern categories present |
| **Curated taxonomy** | JSON (~5KB) | 6 required fields: `content_categories` (array of objects with name/definition), `hook_strategies` (array of objects), `audience_pain_points` (array of strings), `trending_keywords` (array of strings), `engagement_drivers` (array of strings), `content_tactics` (array of strings) | Stage 2.7 (classification input) | Validate: all 6 fields non-empty, definitions >10 chars for semantic categories |
| **Video classifications** | JSON (~2KB each, 120 total) | 12 fields total (6 core + 1 caption_analysis object with 8 subfields + 5 metadata) using refined 3-zone prompt (see Section 5.2) | Stage 7 (LLM Report Generation) | Check: all required fields present, confidence in [high, medium, low], arrays are valid |

### 3.3 Cross-Stage Dependencies

**This feature depends on**:
- **Stage 2 (Video Processing)**: Must complete successfully - requires transcripts, captions, hashtags for all selected videos
- **Stage 2.5 (Bucket Selection)**: Must be enhanced to output `selection_manifest.json` with video lists per bucket (NEW requirement - see QA Q2)

**This feature is required by**:
- **Stage 7 (LLM Report Generation)**: Expects content classification JSONs for contrastive analysis ("60% of top videos use X vs 20% of bottom")

**Failure Impact**:
- If Stage 2.6 fails: Stage 2.7 cannot run (no taxonomy available). Manual intervention required to complete curation.
- If Stage 2.7 fails mid-classification: Partial classifications saved (cached). Resume from checkpoint: re-run Stage 2.7, skips cached videos.
- If Stage 7 runs without content analysis: Reports are incomplete (no qualitative insights). Content Analysis is hard dependency per Critique Phase 1 Q5.

### 3.4 External Dependencies

**Python Libraries**:
```python
import anthropic  # 0.18.0+ (Anthropic Claude API client)
import json  # stdlib
import os  # stdlib
import time  # stdlib
import random  # stdlib
import logging  # stdlib
```

**File System**:
- Read access:
  - `/home/jorge/rumiaifinal/speech_transcriptions/` (transcripts)
  - `/home/jorge/rumiaifinal/unified_analysis/` (captions/hashtags)
  - `/data/clients/{client_id}/hashtags/{cluster_id}/top_contrastive/selection_manifest.json` (video lists)
  - `/data/clients/{client_id}/hashtags/{cluster_id}/top_contrastive/content_taxonomies/{hashtag}_taxonomy.json` (curated taxonomy)
- Write access:
  - `/data/clients/{client_id}/hashtags/{cluster_id}/top_contrastive/content_taxonomies/` (raw discovery + curated taxonomy)
  - `/data/clients/{client_id}/hashtags/{cluster_id}/top_contrastive/buckets/bucket_{duration}/content_analysis/` (classifications)

**Environment Variables**:
- `ANTHROPIC_API_KEY`: API key for Claude (required, fail-fast if missing)
- `LOG_LEVEL`: Logging verbosity (optional, default: INFO)

**External Services**:
- **Anthropic Claude API**:
  - Models: Claude 3.5 Sonnet (discovery), Claude 3 Haiku (classification)
  - Rate limits: Sonnet ~10 req/s, Haiku ~50 req/s (within our usage patterns)
  - Cost: ~$0.75/hashtag discovery (one-time), ~$0.12/hashtag classification (per run)
  - Timeout: 120s for discovery, 30s per video classification
  - Retry: 3 attempts with exponential backoff (1s, 2s, 4s)

---

## 4. Configuration & Parameters

### 4.1 CLI Parameters (if applicable)

Content Analysis is invoked as part of ML training pipeline. Parameters inherited from main pipeline:

| Parameter | Type | Default | Valid Values | Impact | Example |
|-----------|------|---------|--------------|--------|---------|
| `--hashtag` | str | Required | Any hashtag name | Determines which hashtag to analyze | `--hashtag nutrition` |
| `--client` | str | Required | Client ID | Determines file path structure | `--client acme` |
| `--stop-after` | str | None | `discovery`, `classification`, etc. | Stop pipeline after specified stage | `--stop-after discovery` |
| `--resume-from` | str | None | `classification`, etc. | Resume pipeline from specified stage | `--resume-from classification` |

### 4.2 Internal Configuration

```python
# Stage 2.6 Discovery Configuration
DISCOVERY_SAMPLE_SIZE = 50  # Total transcripts to sample (configurable)
DISCOVERY_MODEL = "claude-3-5-sonnet-20241022"
DISCOVERY_TIMEOUT = 120  # seconds (2 minutes)
DISCOVERY_WARNING_THRESHOLD = 60  # seconds (log warning if exceeded)

# Stage 2.7 Classification Configuration
CLASSIFICATION_MODEL = "claude-3-haiku-20240307"
CLASSIFICATION_PER_VIDEO_TIMEOUT = 30  # seconds
CLASSIFICATION_OVERALL_TIMEOUT = 900  # seconds (15 minutes)
CLASSIFICATION_WARNING_THRESHOLD = 600  # seconds (10 minutes)
CLASSIFICATION_VIDEOS_PER_BUCKET = 40  # 20 top + 20 bottom

# LLM API Configuration
LLM_API_TIMEOUT = 30  # seconds per API call
LLM_INTER_REQUEST_DELAY = 0.5  # seconds between requests (safety buffer)
LLM_MAX_RETRIES = 3  # retry attempts before failing
LLM_RETRY_DELAYS = [1, 2, 4]  # exponential backoff (seconds)

# Taxonomy Validation
REQUIRED_TAXONOMY_FIELDS = [
    'content_categories',
    'hook_strategies',
    'audience_pain_points',
    'trending_keywords',
    'engagement_drivers',
    'content_tactics'
]
MIN_DEFINITION_LENGTH = 10  # characters (for semantic categories)

# File Paths (relative to /data/clients/{client_id}/hashtags/{cluster_id}/top_contrastive/)
TAXONOMY_DIR = "content_taxonomies"
RAW_DISCOVERY_FILE = "{hashtag}_raw_discovery.json"
CURATED_TAXONOMY_FILE = "{hashtag}_taxonomy.json"
CONTENT_ANALYSIS_DIR = "buckets/bucket_{duration}/content_analysis"
CLASSIFICATION_FILE = "{video_id}_content.json"

# Transcript/Caption Sources (absolute paths)
TRANSCRIPT_DIR = "/home/jorge/rumiaifinal/speech_transcriptions"
UNIFIED_ANALYSIS_DIR = "/home/jorge/rumiaifinal/unified_analysis"
```

---

## 5. Data Schemas

### 5.1 Input Schema

#### 5.1.1 Selection Manifest (from Stage 2.5)

**File**: `selection_manifest.json`

| Field | Type | Range | Nulls? | Description | Example |
|-------|------|-------|--------|-------------|---------|
| `hashtag` | string | - | No | Hashtag name (without #) | "nutrition" |
| `selected_buckets` | array[string] | 3 items | No | Top 3 duration buckets selected | ["33_60s", "60_90s", "90_120s"] |
| `videos_by_bucket` | object | - | No | Video IDs organized by bucket | See below |
| `videos_by_bucket.{bucket}.top_performers` | array[string] | 40-100 items | No | Video IDs of top performers | ["7526250443832331550", ...] |
| `videos_by_bucket.{bucket}.bottom_performers` | array[string] | 10-25 items | No | Video IDs of bottom performers | ["7428596413707144481", ...] |
| `total_videos` | int | 150-375 | No | Total videos across all buckets | 300 |
| `timestamp` | string | ISO 8601 | No | Manifest creation timestamp | "2025-10-14T10:30:00Z" |

#### 5.1.2 Transcript (from Stage 2 - Whisper)

**File**: `speech_transcriptions/{video_id}_whisper.json`

| Field | Type | Range | Nulls? | Description | Example |
|-------|------|-------|--------|-------------|---------|
| `text` | string | 0-5000 chars | No | Complete transcript (can be empty) | "this is why every woman needs to start yoni steaming..." |
| `segments` | array[object] | - | No | Timestamped segments (optional, not used) | [...] |
| `words` | array[object] | - | No | Word-level data (optional, not used) | [...] |

**Note**: Only `text` field is used. Empty string indicates no speech in video.

#### 5.1.3 Caption and Hashtags (from Stage 2 - unified_analysis)

**File**: `unified_analysis/{video_id}.json`

| Field Path | Type | Range | Nulls? | Description | Example |
|------------|------|-------|--------|-------------|---------|
| `metadata.description` | string | 0-2200 chars | Yes | Creator-written caption | "this is why every woman needs to start yoni steaming..." |
| `metadata.hashtags` | array[object] | 0-30 items | Yes | Hashtag objects | `[{"id": "...", "name": "yonisteam"}, ...]` |
| `metadata.hashtags[].name` | string | - | Yes | Hashtag name (without #) | "yonisteam" |

**Extraction Logic**:
```python
caption = unified_analysis['metadata']['description']  # String or empty
hashtags_array = unified_analysis['metadata']['hashtags']  # Array or empty
hashtag_names = [h['name'] for h in hashtags_array if h.get('name')]  # Extract names only
```

#### 5.1.4 Curated Taxonomy (from Stage 2.6)

**File**: `content_taxonomies/{hashtag}_taxonomy.json`

| Field | Type | Range | Nulls? | Description | Example |
|-------|------|-------|--------|-------------|---------|
| `hashtag` | string | - | No | Hashtag name | "nutrition" |
| `content_categories` | array[object] | 2-10 items | No | Semantic categories with definitions | `[{"name": "recipe_tutorial", "definition": "Step-by-step cooking instructions"}]` |
| `hook_strategies` | array[object] | 2-10 items | No | Hook patterns with definitions | `[{"name": "problem_solution", "definition": "Starts with problem, promises solution"}]` |
| `audience_pain_points` | array[string] | 2-15 items | No | Pain points (simple strings) | `["bloating", "low_energy"]` |
| `trending_keywords` | array[string] | 2-15 items | No | Keywords (simple strings) | `["protein", "gut_health"]` |
| `engagement_drivers` | array[string] | 2-15 items | No | Tactics (simple strings) | `["before_after_reveal", "specific_metrics_mentioned"]` |
| `content_tactics` | array[string] | 2-15 items | No | Presentation styles (simple strings) | `["personal_story", "direct_to_camera"]` |

**Validation Rules**:
- All 6 fields must be present and non-empty
- `content_categories` and `hook_strategies` must have objects with `name` and `definition` fields
- Definitions must be >10 characters
- Other fields (pain_points, keywords, drivers, tactics) are simple string arrays

### 5.2 Output Schema

#### 5.2.1 Raw Discovery (Stage 2.6 Output)

**File**: `content_taxonomies/{hashtag}_raw_discovery.json`

| Field | Type | Description |
|-------|------|-------------|
| `hashtag` | string | Hashtag name |
| `analysis_date` | string | ISO 8601 timestamp |
| `sample_size` | int | Number of transcripts analyzed (typically 50) |
| `discovered_patterns` | object | Container for all pattern categories |
| `discovered_patterns.content_categories` | array[object] | Discovered content types |
| `discovered_patterns.content_categories[].name` | string | Category identifier |
| `discovered_patterns.content_categories[].frequency` | int | Count of videos with this pattern |
| `discovered_patterns.content_categories[].percentage` | float | Calculated by Python post-processing (frequency / sample_size * 100) |
| `discovered_patterns.content_categories[].examples` | array[string] | 2-3 example phrases |
| `discovered_patterns.content_categories[].representative_video_ids` | array[string] | Video IDs showing this pattern |
| `discovered_patterns.hook_strategies` | array[object] | Same structure as content_categories |
| `discovered_patterns.audience_pain_points` | array[string] | Simple string list (e.g., ["bloating", "low energy"]) |
| `discovered_patterns.trending_keywords` | array[string] | Simple string list (e.g., ["protein", "gut health"]) |
| `discovered_patterns.engagement_drivers` | array[string] | Simple string list (e.g., ["before after reveal"]) |
| `discovered_patterns.content_tactics` | array[string] | Simple string list (e.g., ["direct to camera"]) |

#### 5.2.2 Video Classification (Stage 2.7 Output)

**File**: `bucket_{duration}/content_analysis/{video_id}_content.json`

**Complete Schema** (Source: QA Q6 - final locked schema):

| Field | Type | Range | Nulls? | Description |
|-------|------|-------|--------|-------------|
| `video_id` | string | - | No | Video identifier |
| `taxonomy_version` | string | "stage2.6_output" | No | Links classification to taxonomy source (always "stage2.6_output") |
| `content_category` | string | From taxonomy | No | Primary content type (e.g., "recipe_tutorial") |
| `hook_strategy` | string | From taxonomy | No | Opening pattern (e.g., "problem_solution") |
| `pain_points` | array[string] | From taxonomy | No | Detected pain points (can be empty array) - renamed from audience_pain_points |
| `keywords` | array[string] | From taxonomy | No | Detected keywords (can be empty array) - renamed from trending_keywords |
| `engagement_drivers` | array[string] | From taxonomy | No | Shareability tactics (can be empty array) |
| `content_tactics` | array[string] | From taxonomy | No | Presentation styles (can be empty array) |
| `caption_analysis` | object | - | No | Caption-specific analysis (8 subfields) - simplified from 13 subfields |
| `caption_analysis.hook_type` | string | statement, question, command, teaser | No | How caption opens (simplified from 6 to 4 types) |
| `caption_analysis.cta_type` | string | link_in_bio, save_post, comment, follow, share, tag_friend, none | No | Call-to-action type |
| `caption_analysis.brand_mention_present` | boolean | - | No | Whether brand/product mentioned |
| `caption_analysis.influencer_tag_present` | boolean | - | No | Whether influencer tagged |
| `caption_analysis.emoji_usage` | string | none, some, many | No | Emoji density (simplified from 4 to 3 levels) |
| `caption_analysis.caption_length` | string | short, long | No | Caption length category (simplified from 3 to 2 levels) |
| `caption_analysis.hashtag_count` | int | 0-30 | No | Number of hashtags |
| `caption_analysis.hashtag_placement` | string | end, mixed, none | No | Where hashtags appear |
| `confidence` | string | high, medium, low | No | Classification confidence |
| `transcript_available` | boolean | - | No | Whether transcript was used (false = caption/hashtag only) |
| `note` | string | - | Yes | Optional note (e.g., "Classified using caption and hashtags only") |

**Example** (complete):
```json
{
  "video_id": "7526250443832331550",
  "taxonomy_version": "stage2.6_output",
  "content_category": "wellness_practice",
  "hook_strategy": "direct_statement",
  "pain_points": ["menstrual_discomfort", "feminine_wellness"],
  "keywords": ["yoni", "steaming", "holistic", "tcm"],
  "engagement_drivers": ["personal_testimony", "product_recommendation"],
  "content_tactics": ["direct_to_camera", "product_demonstration"],
  "caption_analysis": {
    "hook_type": "statement",
    "cta_type": "link_in_bio",
    "brand_mention_present": true,
    "influencer_tag_present": true,
    "emoji_usage": "some",
    "caption_length": "long",
    "hashtag_count": 9,
    "hashtag_placement": "end"
  },
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

---

## 6. Error Handling & Validation

### 6.1 Input Validation

**Stage 2.6 Discovery Validation**:
```python
def validate_discovery_inputs(manifest_path, sample_size):
    """
    Validate inputs before discovery.
    Source: QA Q2, Q3, Q8
    """
    # 1. Check manifest exists
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"selection_manifest.json not found at {manifest_path}. "
            "Did Stage 2.5 complete successfully?"
        )

    # 2. Load and validate manifest structure
    manifest = load_json(manifest_path)
    required_fields = ['hashtag', 'selected_buckets', 'videos_by_bucket']
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")

    # 3. Check we have 3 buckets
    if len(manifest['selected_buckets']) != 3:
        raise ValueError(
            f"Expected 3 selected buckets, found {len(manifest['selected_buckets'])}. "
            "Stage 2.5 may have failed."
        )

    # 4. Check each bucket has videos
    for bucket in manifest['selected_buckets']:
        if bucket not in manifest['videos_by_bucket']:
            raise ValueError(f"Bucket {bucket} missing from videos_by_bucket")

        top_performers = manifest['videos_by_bucket'][bucket].get('top_performers', [])
        if len(top_performers) < 10:
            raise ValueError(
                f"Bucket {bucket} has only {len(top_performers)} top performers. "
                f"Need at least 10 for sampling."
            )

    # 5. Check sample size is reasonable
    if sample_size < 10:
        raise ValueError(f"Sample size too small: {sample_size}. Minimum is 10.")
    if sample_size > 200:
        logger.warning(f"Sample size very large: {sample_size}. May exceed LLM token limits.")

    # 6. Check ANTHROPIC_API_KEY set
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set with: export ANTHROPIC_API_KEY=sk-ant-..."
        )
```

**Stage 2.7 Classification Validation**:
```python
def validate_classification_inputs(taxonomy_path, manifest_path):
    """
    Validate inputs before classification.
    Source: QA Q10 (taxonomy validation)
    """
    # 1. Check taxonomy exists
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(
            f"Curated taxonomy not found at {taxonomy_path}. "
            "Run Stage 2.6 discovery and complete manual curation first."
        )

    # 2. Load and validate taxonomy structure
    taxonomy = load_json(taxonomy_path)

    # Check all required fields present
    required_fields = [
        'content_categories', 'hook_strategies', 'audience_pain_points',
        'trending_keywords', 'engagement_drivers', 'content_tactics'
    ]
    missing = [f for f in required_fields if f not in taxonomy]
    if missing:
        raise ValueError(f"Taxonomy missing required fields: {missing}")

    # Check all fields non-empty
    for field in required_fields:
        if not taxonomy[field]:
            raise ValueError(f"Taxonomy field '{field}' is empty")

    # Check semantic categories have definitions
    for category in taxonomy['content_categories']:
        if 'name' not in category or 'definition' not in category:
            raise ValueError(f"content_categories missing name or definition: {category}")
        if len(category['definition']) < 10:
            raise ValueError(
                f"Definition too short for '{category['name']}': "
                f"'{category['definition']}' (min 10 chars)"
            )

    for strategy in taxonomy['hook_strategies']:
        if 'name' not in strategy or 'definition' not in strategy:
            raise ValueError(f"hook_strategies missing name or definition: {strategy}")
        if len(strategy['definition']) < 10:
            raise ValueError(
                f"Definition too short for '{strategy['name']}': "
                f"'{strategy['definition']}' (min 10 chars)"
            )

    # 3. Check manifest exists (same as discovery validation)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"selection_manifest.json not found at {manifest_path}. "
            "Did Stage 2.5 complete successfully?"
        )
```

### 6.2 Error Cases

| Error | Detection | Handling | User Message | Exit Code |
|-------|-----------|----------|--------------|-----------|
| **Missing manifest** | `os.path.exists()` | Fail-fast | `"selection_manifest.json not found at {path}. Did Stage 2.5 complete successfully?"` | 1 |
| **Invalid manifest structure** | JSON validation | Fail-fast | `"Manifest missing required fields: {fields}. Stage 2.5 may have failed."` | 2 |
| **Missing ANTHROPIC_API_KEY** | `os.environ.get()` | Fail-fast | `"ANTHROPIC_API_KEY not set. Set with: export ANTHROPIC_API_KEY=sk-ant-..."` | 3 |
| **Missing taxonomy** | `os.path.exists()` | Fail-fast | `"Curated taxonomy not found. Run Stage 2.6 discovery and complete manual curation first."` | 4 |
| **Invalid taxonomy structure** | Schema validation | Fail-fast | `"Taxonomy missing required field: {field}"` or `"Definition too short for '{name}': {definition}"` | 5 |
| **LLM API timeout (discovery)** | Timeout after 120s | Retry 3x (1s, 2s, 4s), then fail | `"⏰ Discovery timeout (>120s). Retry {N}/3..."` then `"❌ Discovery failed after 3 retries. Check status.anthropic.com"` | 6 |
| **LLM API timeout (per video)** | Timeout after 30s | Retry 3x (1s, 2s, 4s), then fail | `"⏰ Video {video_id} timeout (>30s). Retry {N}/3..."` then `"❌ Video {video_id} failed after 3 retries"` | 7 |
| **LLM API timeout (overall)** | Stage 2.7 exceeds 15min | Fail-fast | `"❌ Classification timeout (>15 min). Processed {N}/120 videos. API may be slow. Check status.anthropic.com"` | 8 |
| **Invalid JSON response from LLM** | JSON parse error | Retry 3x, then fail | `"⚠️ LLM returned invalid JSON for {video_id}. Retry {N}/3..."` then `"❌ Invalid JSON after 3 retries: {error}"` | 9 |
| **Write permission denied** | File write exception | Fail-fast | `"Cannot write to {path}. Check directory exists and has write permissions."` | 10 |
| **Missing transcript file** | `FileNotFoundError` | Warn + continue (use caption/hashtags) | `"⚠️ Transcript not found: {video_id}. Classifying using caption/hashtags only."` | 0 (warning) |
| **Missing unified_analysis file** | `FileNotFoundError` | Warn + continue (use transcript only) | `"⚠️ Unified analysis not found: {video_id}. Classifying using transcript only."` | 0 (warning) |
| **Insufficient videos in bucket** | Row count check | Warn + continue | `"⚠️ Bucket {bucket} has only {count} top performers (expected >10). Sampling all available."` | 0 (warning) |

**Warning vs Fatal Distinction**:
- **Fatal errors (exit >0)**: Configuration issues, missing required data, API failures after retries
- **Warnings (exit 0)**: Missing optional data (transcripts, captions can be empty), video count mismatches

### 6.3 Output Validation

**Stage 2.6 Discovery Output Validation**:
```python
def validate_discovery_output(raw_taxonomy):
    """
    Validate raw discovery JSON before saving.
    Source: QA Q10 (raw discovery schema)
    """
    required_top_level = ['hashtag', 'analysis_date', 'sample_size', 'discovered_patterns']
    missing = [f for f in required_top_level if f not in raw_taxonomy]
    if missing:
        raise ValueError(f"Discovery output missing fields: {missing}")

    # Check discovered_patterns has all 6 categories
    required_patterns = [
        'content_categories', 'hook_strategies', 'audience_pain_points',
        'trending_keywords', 'engagement_drivers', 'content_tactics'
    ]
    patterns = raw_taxonomy['discovered_patterns']
    missing = [f for f in required_patterns if f not in patterns]
    if missing:
        raise ValueError(f"Discovered patterns missing categories: {missing}")

    # Check each pattern array is non-empty
    for category in required_patterns:
        if not patterns[category]:
            logger.warning(f"Discovery found 0 patterns for {category}. This is unusual.")

    # Check pattern objects have required fields
    # Note: percentage added by Python post-processing (see TI Section 4.2.5)
    for category in ['content_categories', 'hook_strategies']:
        for pattern in patterns[category]:
            required_fields = ['name', 'frequency', 'examples']
            missing = [f for f in required_fields if f not in pattern]
            if missing:
                raise ValueError(
                    f"Pattern in {category} missing fields: {missing}. Pattern: {pattern}"
                )
```

**Stage 2.7 Classification Output Validation**:
```python
def validate_classification_output(classification):
    """
    Validate classification JSON before saving.
    Source: QA Q6 (complete output schema)
    """
    # Check all 12 core fields present
    core_fields = [
        'video_id', 'taxonomy_version', 'content_category', 'hook_strategy',
        'pain_points', 'keywords', 'engagement_drivers', 'content_tactics',
        'caption_analysis', 'confidence', 'transcript_available', 'note'
    ]
    missing = [f for f in core_fields if f not in classification]
    if missing:
        raise ValueError(f"Classification missing core fields: {missing}")

    # Check confidence value
    if classification['confidence'] not in ['high', 'medium', 'low']:
        raise ValueError(
            f"Invalid confidence value: {classification['confidence']}. "
            f"Must be high, medium, or low."
        )

    # Check caption_analysis has all 8 subfields
    caption_fields = [
        'hook_type', 'cta_type', 'brand_mention_present',
        'influencer_tag_present', 'emoji_usage', 'caption_length',
        'hashtag_count', 'hashtag_placement'
    ]
    caption_analysis = classification['caption_analysis']
    missing = [f for f in caption_fields if f not in caption_analysis]
    if missing:
        raise ValueError(f"caption_analysis missing fields: {missing}")

    # Check arrays are actually arrays
    array_fields = ['pain_points', 'keywords', 'engagement_drivers', 'content_tactics']
    for field in array_fields:
        if not isinstance(classification[field], list):
            raise ValueError(f"Field {field} must be array, got {type(classification[field])}")
```

---

## 7. Performance & Scalability

### 7.1 Performance Targets

**Stage 2.6 Discovery**:
- **Expected time**: 30 seconds (50 transcripts, Sonnet)
- **Warning threshold**: 60 seconds (log warning if exceeded)
- **Timeout**: 120 seconds (fail if exceeded)
- **Memory**: < 500 MB (50 transcripts ~50KB each = 2.5MB input)
- **Network**: ~50K tokens input, ~2K tokens output

**Stage 2.7 Classification**:
- **Expected time**: 5 minutes (120 videos × 2.5s each with Haiku)
- **Warning threshold**: 10 minutes (log warning if exceeded)
- **Per-video timeout**: 30 seconds (fail individual video if exceeded)
- **Overall timeout**: 15 minutes (fail entire stage if exceeded)
- **Memory**: < 100 MB (per-video processing, no batch accumulation)
- **Network**: ~2K tokens per video (120 videos = ~240K tokens total)

**Cost Targets**:
- Discovery: ~$0.75 per hashtag (one-time)
- Classification: ~$0.12 per hashtag (per run)
- Total first run: ~$0.87 per hashtag
- Total subsequent runs: ~$0.12 per hashtag

### 7.2 Measured Performance

Not yet measured (component not implemented). Expected to meet targets based on:
- Anthropic API typical latency: 1-2s for Haiku classification prompts
- Sequential processing avoids rate limit issues
- 0.5s inter-request delay adds 60s total (acceptable overhead)

### 7.3 Bottlenecks & Mitigations

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| **LLM API latency variance** | Occasionally >30s per video | Anthropic API load fluctuations | 4x timeout buffer (30s vs ~7s expected), 3 retries with backoff | High |
| **Sequential classification** | 5 min for 120 videos (vs <1 min if parallel) | API rate limit avoidance + simplicity | Accept trade-off (5 min is acceptable for background task) | Low |
| **Manual curation time** | ~2 hours per hashtag | Human review required for business-relevant taxonomy | Semi-automation after 2-3 hashtags (learned heuristics reduce to 30 min) | Medium |
| **Large transcript token usage** | Rare: >5K tokens if 10-min video | Very long videos produce massive transcripts | Warn if transcript >3K tokens, consider truncation in future | Low |
| **Network failures** | Intermittent API failures | Internet connectivity, API downtime | 3 retries with exponential backoff, fail-fast for investigation | High |

### 7.4 Scalability Limits

- **Max videos per hashtag**: 1000 (sequential classification: 1000 × 2.5s = ~42 minutes, still acceptable)
- **Max hashtags per run**: No inherent limit (taxonomies are independent)
- **Max transcript size**: ~10K tokens (Anthropic limit is 200K, but transcripts rarely exceed 5K)
- **Min videos per hashtag**: 50 (below this, pattern discovery unreliable)

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Test Discovery Sampling**:
- [ ] Sample 50 transcripts from manifest with 3 buckets → returns 50 (17+17+16)
- [ ] Sample stratified evenly → each bucket contributes ~17 transcripts
- [ ] Sample from top performers only → no bottom performer IDs in results
- [ ] Handle bucket with <17 videos → samples all available, continues
- [ ] Missing transcript file → logs warning, skips video, continues

**Test LLM Prompt Generation**:
- [ ] Generate discovery prompt with 50 transcripts → includes all 6 pattern categories
- [ ] Generate classification prompt with taxonomy → includes all taxonomy fields
- [ ] Handle empty transcript → prompt includes "Transcript: (empty)" + caption/hashtags
- [ ] Handle missing caption → prompt includes "Caption: (empty)" + transcript/hashtags

**Test Taxonomy Validation**:
- [ ] Valid taxonomy (all 6 fields, definitions >10 chars) → passes
- [ ] Missing field → raises ValueError with field name
- [ ] Empty field → raises ValueError
- [ ] Short definition (<10 chars) → raises ValueError with category name
- [ ] Missing name or definition in semantic category → raises ValueError

**Test Classification Output Validation**:
- [ ] Valid classification (all 22 fields) → passes
- [ ] Missing core field → raises ValueError
- [ ] Invalid confidence value → raises ValueError
- [ ] Missing caption_analysis subfield → raises ValueError
- [ ] Non-array for array field → raises ValueError

### 8.2 Integration Tests

**Test End-to-End Discovery (Stage 2.6)**:
- [ ] Use real manifest with 10 videos per bucket (30 total)
- [ ] Run discovery with sample_size=10 (mock LLM response)
- [ ] Verify raw_discovery.json created with correct schema
- [ ] Manually create curated taxonomy
- [ ] Verify taxonomy validates successfully

**Test End-to-End Classification (Stage 2.7)**:
- [ ] Use 5 real videos (transcripts + captions + hashtags)
- [ ] Create mock taxonomy with 2-3 categories per field
- [ ] Run classification (mock LLM responses)
- [ ] Verify 5 classification JSON files created
- [ ] Verify all files have complete schema (22 fields)

**Test Checkpoint Resume**:
- [ ] Classify 10 videos, stop mid-process
- [ ] Resume classification → skips cached videos, completes remaining

**Test Error Propagation**:
- [ ] Missing manifest → Stage 2.6 fails with clear message
- [ ] Missing taxonomy → Stage 2.7 fails with clear message
- [ ] LLM API timeout → retries 3x, then fails with specific video ID

**Test Empty Transcript Handling** (Source: QA Q4 - Option B):
- [ ] Video with empty transcript → classification uses caption/hashtags only
- [ ] Output has `transcript_available: false`
- [ ] Output has `note: "Classified using caption and hashtags only"`

### 8.3 Test Data

**Sample Manifest** (`tests/fixtures/selection_manifest_sample.json`):
```json
{
  "hashtag": "nutrition",
  "selected_buckets": ["33_60s", "60_90s", "90_120s"],
  "videos_by_bucket": {
    "33_60s": {
      "top_performers": ["7526250443832331550", "7428596413707144481"],
      "bottom_performers": ["7480428850522950920"]
    },
    "60_90s": {
      "top_performers": ["video_id_4", "video_id_5"],
      "bottom_performers": ["video_id_6"]
    },
    "90_120s": {
      "top_performers": ["video_id_7", "video_id_8"],
      "bottom_performers": ["video_id_9"]
    }
  },
  "total_videos": 9
}
```

**Sample Transcript** (`tests/fixtures/7526250443832331550_whisper.json`):
```json
{
  "text": "this is why every woman needs to start yoni steaming. I want to invest in a boiler pot eventually so I can sit on it even longer! my stool is linked in my amazon under TCM",
  "segments": [],
  "words": []
}
```

**Sample Curated Taxonomy** (`tests/fixtures/nutrition_taxonomy.json`):
```json
{
  "hashtag": "nutrition",
  "content_categories": [
    {"name": "wellness_practice", "definition": "Traditional or alternative health practices"},
    {"name": "product_recommendation", "definition": "Product reviews or recommendations"}
  ],
  "hook_strategies": [
    {"name": "direct_statement", "definition": "Opens with bold declarative statement"},
    {"name": "problem_solution", "definition": "Starts with problem, promises solution"}
  ],
  "audience_pain_points": ["menstrual_discomfort", "stress"],
  "trending_keywords": ["holistic", "wellness"],
  "engagement_drivers": ["personal_testimony", "product_link"],
  "content_tactics": ["direct_to_camera", "product_demonstration"]
}
```

**Expected Classification Output** (`tests/fixtures/7526250443832331550_content_expected.json`):
```json
{
  "video_id": "7526250443832331550",
  "taxonomy_version": "stage2.6_output",
  "content_category": "wellness_practice",
  "hook_strategy": "direct_statement",
  "pain_points": ["menstrual_discomfort"],
  "keywords": ["holistic"],
  "engagement_drivers": ["personal_testimony", "product_link"],
  "content_tactics": ["direct_to_camera", "product_demonstration"],
  "caption_analysis": {
    "hook_type": "statement",
    "cta_type": "link_in_bio",
    "brand_mention_present": true,
    "influencer_tag_present": true,
    "emoji_usage": "some",
    "caption_length": "long",
    "hashtag_count": 9,
    "hashtag_placement": "end"
  },
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

### 8.4 Test Execution

```bash
# Unit tests
pytest tests/test_content_analysis_discovery.py -v
pytest tests/test_content_analysis_classification.py -v
pytest tests/test_taxonomy_validation.py -v

# Integration tests
pytest tests/test_content_analysis_integration.py -v

# End-to-end test with mock LLM
pytest tests/test_content_analysis_e2e.py -v --mock-llm

# Test with real LLM (requires ANTHROPIC_API_KEY)
pytest tests/test_content_analysis_e2e.py -v --real-llm

# Coverage report
pytest --cov=content_analysis --cov-report=html
```

---

## 9. Future Enhancements

### 9.1 Planned Improvements

**Phase 2: Semi-Automated Taxonomy Curation**
- Current: Fully manual curation (~2 hours per hashtag)
- Future: After 2-3 hashtags, apply learned heuristics (auto-filter <10% frequency, brand-specific terms, etc.)
- Impact: Reduce curation time from 2 hours → 30 minutes (70% reduction)

**Phase 3: Universal Taxonomy with Hashtag Extensions**
- Current: Separate taxonomy per hashtag
- Future: If 70%+ overlap discovered across hashtags, consolidate to universal taxonomy with minor hashtag-specific extensions
- Impact: 10 hashtags × 2 hours = 20 hours → 1 base taxonomy (2 hours) + 10 extensions (30 min each) = 7 hours total

**Phase 4: Upgrade Haiku to Sonnet for Classification Quality**
- Current: Haiku for classification (~$0.12/hashtag)
- Future: If Haiku misclassification rate >20%, upgrade to Sonnet (~$4.50/hashtag)
- Impact: Higher accuracy at 37x cost increase ($4.38 more per hashtag)

**Phase 5: Parallel Classification Processing**
- Current: Sequential (120 videos × 2.5s = 5 minutes)
- Future: Parallel with rate limiter (120 videos in ~30 seconds)
- Impact: 10x speedup, but requires careful rate limit management

### 9.2 Known Limitations

- **No confidence scoring for individual fields**: `confidence` is overall, not per-field (can't say "content_category: high confidence, hook_strategy: low confidence")
- **Fixed taxonomy per hashtag**: Cannot dynamically add categories mid-run (must re-run discovery)
- **No multi-label content categories**: Videos can only have ONE primary category (not "recipe_tutorial + supplement_review")
- **Hashtag broad/niche/branded categorization is heuristic-based**: LLM estimates view counts, not actual TikTok API data
- **No video visual analysis**: Classification uses transcript/caption/hashtags only (doesn't analyze visual content like "direct_to_camera" detection from video frames)

---

## 10. References & Related Docs

### 10.1 Parent Document

- **ContentAnalysis.md - Stages 2.6 & 2.7**
  - Original design document for Content Analysis
  - Brainstorm and conceptual framework
  - Pipeline integration architecture

### 10.2 Mother Document Foundation

- **MLPlanningv2.md Part 1: Foundation** (shared across all stages)
  - Section: Client Architecture & Storage - Provides directory paths for file I/O
  - Section: CLI Command Structure - Defines CLI parameters (hashtag, client, stop-after, resume-from)
  - Section: Configuration Schemas - Defines config.json and selection_manifest structure
  - Appendix A: Glossary - System-wide term definitions

**Key Sections Referenced in This Stage**:
- Client Architecture: `/data/clients/{client_id}/hashtags/{cluster_id}/top_contrastive/` base path
- CLI Command Structure: `--hashtag`, `--client`, `--stop-after discovery`, `--resume-from classification`
- Configuration Schemas: `selection_manifest.json` structure (NEW requirement from QA Q2)

### 10.3 Related Child Docs

- **Stage 2 (Video Processing)** (upstream)
  - Produces transcripts (`speech_transcriptions/{video_id}_whisper.json`)
  - Produces captions/hashtags (`unified_analysis/{video_id}.json`)
  - Required for Content Analysis inputs

- **Stage 2.5 (Bucket Selection)** (upstream)
  - Must be enhanced to produce `selection_manifest.json` (NEW requirement)
  - Provides video lists per bucket for sampling/classification

- **Stage 7 (LLM Report Generation)** (downstream)
  - Consumes content classification JSONs (`content_analysis/{video_id}_content.json`)
  - Combines ML insights + content insights for comprehensive creative reports
  - Enables contrastive analysis ("60% of top use X vs 20% of bottom")

### 10.4 External References

- **Anthropic Claude API Documentation**: https://docs.anthropic.com/claude/reference/messages_post
- **Claude 3.5 Sonnet Model Card**: https://www.anthropic.com/claude/sonnet
- **Claude 3 Haiku Model Card**: https://www.anthropic.com/claude/haiku
- **RumiAI temporal_windows schema**: `SystemArchitecturev2.md` (lines 395-460)
- **RumiAI feature definitions**: `TotalFeatures.md` (complete 61 feature matrix)

### 10.5 Phase Documentation

- **Critique_ContentAnalysis.md** (Phase 1 output)
  - Business validation and critical concerns
  - Final Decision: APPROVE with REFINEMENT
  - Key decisions: Manual gate (Option B), contrastive analysis required

- **QA_ContentAnalysis.md** (Phase 2 output)
  - 13 Q&A covering all technical clarifications
  - Input/output schemas, error handling, performance targets
  - Testing strategy and edge case handling

---

## Appendix A: Decision Log

### Decision 1: Two-Step Pipeline with Manual Gate

**Context**: Content Analysis requires human curation of taxonomy (business decisions about actionability, granularity, terminology). Phase 1 Critique Q4-Q5 identified need for workflow coordination.

**Alternatives Considered**:
- Option A: Graceful Degradation - Stage 7 checks if taxonomy exists, skips content analysis if missing
- Option B: Manual Gate - Explicit checkpoint, Stage 7 requires content analysis (hard dependency)
- Option C: Status Tracking System - Track curation state, show warnings

**Rationale**: Option B selected because content analysis is core business requirement, not optional enhancement. Reports without content classifications are incomplete per business value proposition.

**Trade-offs**: Adds manual intervention step between Stage 2.6 and Stage 2.7. Pipeline must pause for human curation (~2 hours). Accepted because taxonomy quality is critical for actionable insights.

**Date**: 2025-10-14 (Phase 1 Q5)

### Decision 2: Sonnet for Discovery, Haiku for Classification

**Context**: LLM costs can be 15x different between models. Need to balance quality vs cost. Phase 2 Q7 Point 2 evaluated model selection.

**Alternatives Considered**:
- Option A: Sonnet for both - Highest quality, $4.50 per hashtag
- Option B: Haiku for both - Lowest cost, $0.30 per hashtag
- Option C: Sonnet/Haiku split - Balanced approach, $0.87 per hashtag first run

**Rationale**: Option C selected. Discovery requires sophisticated pattern recognition and creative category naming (Sonnet's strength). Classification is repetitive pattern matching against predefined taxonomy (Haiku can handle). 15x cost savings for classification ($4.20 → $0.30) justifies split architecture.

**Trade-offs**: Risk of Haiku missing subtle patterns in classification. Accepted with validation: if Haiku misclassification rate >20% in testing, upgrade to Sonnet for Stage 2.7.

**Date**: 2025-10-14 (Phase 2 Q7)

### Decision 3: Stratified Even Sampling (50 transcripts across 3 buckets)

**Context**: Discovery needs representative sample from top performers. Videos in different duration buckets may have different narrative patterns. Phase 2 Q8 evaluated sampling strategies.

**Alternatives Considered**:
- Option A: Single "winning bucket" - All 50 from bucket with most videos
- Option B: Stratified even - ~17 from each of top 3 buckets
- Option C: Stratified proportional - Sample weighted by bucket size

**Rationale**: Option B selected. Ensures taxonomy discovers patterns across all duration ranges (e.g., 90-120s videos may have unique patterns like longer narrative arcs that 33-60s don't). 50 transcripts split evenly (17+17+16) gives sufficient representation per bucket.

**Trade-offs**: Undersamples largest bucket slightly. Accepted because 17 samples still sufficient for pattern discovery, and bucket diversity more valuable than bucket size weighting.

**Date**: 2025-10-14 (Phase 2 Q8)

### Decision 4: Classify Both Top and Bottom Performers (120 videos total)

**Context**: Business value proposition is identifying what DIFFERENTIATES viral from non-viral content. Phase 2 Q9 evaluated classification scope.

**Alternatives Considered**:
- Option A: Top only (60 videos) - $0.06/hashtag, no contrastive analysis
- Option B: Selective contrastive (120 videos: 20 top + 20 bottom per bucket) - $0.12/hashtag, enables contrastive
- Option C: Full contrastive (all 300 videos) - $0.30/hashtag, more robust statistics

**Rationale**: Option B selected. Without bottom classifications, can't prove if pattern is differentiator (e.g., if "personal_story" is 80% in top but also 75% in bottom, it's not actionable insight). 120 videos (40 per bucket × 3) provides sufficient data for "60% vs 30%" style insights.

**Trade-offs**: 2x cost vs top-only ($0.12 vs $0.06). Accepted because $0.06 savings ($0.60 for 10 hashtags) undermines entire business value of identifying viral differentiators.

**Date**: 2025-10-14 (Phase 2 Q9)

### Decision 5: Observable Features over Subjective Ratings

**Context**: Original schema (ContentAnalysis.md lines 366-372) had subjective ratings like "authenticity: very_high". Phase 2 Q6 identified reliability concerns.

**Alternatives Considered**:
- Option A: Minimal schema - String arrays only
- Option B: Structured with definitions - Add definitions for semantic categories
- Option C: Rich with ratings - Include subjective quality assessments

**Rationale**: Option B selected with modification: Replace subjective ratings (authenticity, relatability, educational_value, entertainment_factor) with observable content_tactics (personal_story, direct_to_camera, vulnerability_shown). LLM can reliably detect presence/absence of tactics (boolean) better than rate quality (subjective 1-5 scale).

**Trade-offs**: Loses some semantic richness (can't say "high authenticity"). Accepted because observable features are actionable ("Use direct_to_camera") vs vague ratings ("Increase authenticity to 8/10").

**Date**: 2025-10-14 (Phase 2 Q6)

---

## Appendix B: Example Data

### B.1 Sample Raw Discovery Output

**File**: `content_taxonomies/nutrition_raw_discovery.json` (excerpt, 3 patterns shown)

```json
{
  "hashtag": "nutrition",
  "analysis_date": "2025-10-14T10:30:00Z",
  "sample_size": 50,
  "discovered_patterns": {
    "content_categories": [
      {
        "name": "recipe_tutorial",
        "frequency": 32,
        "percentage": 64.0,  // Added by Python post-processing
        "examples": [
          "protein smoothie recipe for breakfast",
          "easy meal prep for busy professionals",
          "high protein low calorie dinner ideas"
        ],
        "representative_video_ids": ["7526250443832331550", "7428596413707144481"]
      },
      {
        "name": "supplement_review",
        "frequency": 18,
        "percentage": 36.0,  // Added by Python post-processing
        "examples": [
          "best magnesium supplement for sleep",
          "protein powder taste test",
          "vitamin D deficiency symptoms"
        ],
        "representative_video_ids": ["7480428850522950920"]
      }
    ],
    "hook_strategies": [
      {
        "name": "problem_solution",
        "frequency": 27,
        "percentage": 54.0,  // Added by Python post-processing
        "examples": [
          "struggling with bloating? try this",
          "low energy all day? here's why",
          "can't lose weight? you're missing this"
        ],
        "representative_video_ids": ["video_id_3"]
      }
    ]
  }
}
```

### B.2 Sample Curated Taxonomy

**File**: `content_taxonomies/nutrition_taxonomy.json` (complete)

```json
{
  "hashtag": "nutrition",
  "content_categories": [
    {
      "name": "recipe_tutorial",
      "definition": "Step-by-step cooking or meal preparation instructions"
    },
    {
      "name": "supplement_review",
      "definition": "Product reviews or recommendations for vitamins, proteins, supplements"
    },
    {
      "name": "wellness_practice",
      "definition": "Traditional or alternative health practices (e.g., yoni steaming, oil pulling)"
    }
  ],
  "hook_strategies": [
    {
      "name": "problem_solution",
      "definition": "Opens by stating a problem, then promises or teases a solution"
    },
    {
      "name": "direct_statement",
      "definition": "Opens with bold declarative statement or fact"
    },
    {
      "name": "question",
      "definition": "Opens with question directed at viewer"
    }
  ],
  "audience_pain_points": [
    "bloating",
    "low_energy",
    "weight_loss_plateau",
    "poor_sleep",
    "digestive_issues",
    "menstrual_discomfort"
  ],
  "trending_keywords": [
    "protein",
    "gut_health",
    "metabolism",
    "inflammation",
    "holistic",
    "macro_tracking"
  ],
  "engagement_drivers": [
    "before_after_reveal",
    "specific_metrics_mentioned",
    "relatable_struggle",
    "product_recommendation",
    "personal_testimony"
  ],
  "content_tactics": [
    "personal_story",
    "direct_to_camera",
    "specific_actionable_steps",
    "vulnerability_shown",
    "product_demonstration",
    "transformation_narrative"
  ]
}
```

### B.3 Sample Video Classification Output

**File**: `bucket_33_60s/content_analysis/7526250443832331550_content.json` (complete)

```json
{
  "video_id": "7526250443832331550",
  "taxonomy_version": "stage2.6_output",
  "content_category": "wellness_practice",
  "hook_strategy": "direct_statement",
  "pain_points": ["menstrual_discomfort"],
  "keywords": ["holistic", "wellness"],
  "engagement_drivers": ["personal_testimony", "product_recommendation"],
  "content_tactics": ["direct_to_camera", "product_demonstration", "personal_story"],
  "caption_analysis": {
    "hook_type": "statement",
    "cta_type": "link_in_bio",
    "brand_mention_present": true,
    "influencer_tag_present": true,
    "emoji_usage": "some",
    "caption_length": "long",
    "hashtag_count": 9,
    "hashtag_placement": "end"
  },
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

### B.4 Sample Classification with Empty Transcript

**File**: `bucket_60_90s/content_analysis/video_no_speech_content.json` (illustrates QA Q4 - Option B)

```json
{
  "video_id": "video_no_speech_example",
  "taxonomy_version": "stage2.6_output",
  "content_category": "recipe_tutorial",
  "hook_strategy": "direct_statement",
  "pain_points": [],
  "keywords": ["protein", "meal_prep"],
  "engagement_drivers": ["specific_actionable_steps"],
  "content_tactics": ["visual_demonstration"],
  "caption_analysis": {
    "hook_type": "command",
    "cta_type": "save_post",
    "brand_mention_present": false,
    "influencer_tag_present": false,
    "emoji_usage": "many",
    "caption_length": "short",
    "hashtag_count": 5,
    "hashtag_placement": "end"
  },
  "confidence": "medium",
  "transcript_available": false,
  "note": "Classified using caption and hashtags only"
}
```

---

## Appendix C: Pseudocode (Complete)

### C.1 Stage 2.6 Discovery - Full Pipeline

```python
def run_discovery(client_id, hashtag, manifest_path, sample_size=50):
    """
    Complete Stage 2.6 discovery pipeline.

    Args:
        client_id: str, client identifier
        hashtag: str, hashtag name (e.g., "nutrition")
        manifest_path: str, path to selection_manifest.json
        sample_size: int, transcripts to sample (default 50, configurable)

    Returns:
        str: Path to raw_discovery.json

    Raises:
        ValueError: if validation fails
        TimeoutError: if LLM exceeds 120s timeout after 3 retries
    """
    logger.info(f"=== Stage 2.6: Content Discovery - #{hashtag} ===")

    # === 1. Validate Inputs ===
    logger.info("Step 1: Validating inputs")
    validate_discovery_inputs(manifest_path, sample_size)

    # === 2. Sample Transcripts ===
    logger.info(f"Step 2: Sampling {sample_size} transcripts")
    transcripts = sample_transcripts_for_discovery(manifest_path, sample_size)
    logger.info(f"  ✅ Sampled {len(transcripts)} transcripts")
    logger.info(f"     {len([t for t in transcripts if t['bucket'] == transcripts[0]['bucket']])} from first bucket")

    # === 3. LLM Discovery ===
    logger.info("Step 3: Running LLM pattern discovery (Sonnet)")
    start_time = time.time()

    raw_taxonomy = discover_patterns_llm(transcripts, hashtag)

    elapsed = time.time() - start_time
    if elapsed > 60:
        logger.warning(f"⚠️  Discovery took {elapsed:.0f}s (>60s warning threshold)")
    logger.info(f"  ✅ Discovery complete in {elapsed:.0f}s")

    # === 4. Validate Output ===
    logger.info("Step 4: Validating discovery output")
    validate_discovery_output(raw_taxonomy)

    # === 5. Save Raw Discovery ===
    base_path = f"/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/content_taxonomies"
    os.makedirs(base_path, exist_ok=True)
    output_path = os.path.join(base_path, f"{hashtag}_raw_discovery.json")
    save_json(output_path, raw_taxonomy)

    logger.info(f"✅ Stage 2.6 Complete: {output_path}")
    logger.info(f"")
    logger.info(f"📝 MANUAL STEP REQUIRED:")
    logger.info(f"   1. Review: {output_path}")
    logger.info(f"   2. Curate: Filter patterns, refine terminology")
    logger.info(f"   3. Save to: {base_path}/{hashtag}_taxonomy.json")
    logger.info(f"   4. Resume with: --resume-from classification")
    logger.info(f"")

    return output_path
```

### C.2 Stage 2.7 Classification - Full Pipeline

```python
def run_classification(client_id, hashtag, manifest_path, taxonomy_path):
    """
    Complete Stage 2.7 classification pipeline.

    Args:
        client_id: str, client identifier
        hashtag: str, hashtag name
        manifest_path: str, path to selection_manifest.json
        taxonomy_path: str, path to curated taxonomy.json

    Returns:
        int: Number of videos classified

    Raises:
        ValueError: if validation fails
        TimeoutError: if overall classification exceeds 15 min
    """
    logger.info(f"=== Stage 2.7: Content Classification - #{hashtag} ===")

    # === 1. Validate Inputs ===
    logger.info("Step 1: Validating inputs")
    validate_classification_inputs(taxonomy_path, manifest_path)

    # === 2. Load Taxonomy & Manifest ===
    logger.info("Step 2: Loading taxonomy and manifest")
    taxonomy = load_json(taxonomy_path)
    manifest = load_json(manifest_path)
    top_3_buckets = manifest['selected_buckets']
    logger.info(f"  Taxonomy: {len(taxonomy['content_categories'])} categories, {len(taxonomy['hook_strategies'])} hooks")
    logger.info(f"  Buckets: {', '.join(top_3_buckets)}")

    # === 3. Initialize LLM Client ===
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # === 4. Classify Videos ===
    logger.info("Step 3: Classifying videos (Haiku)")
    start_time = time.time()
    classified_count = 0
    total_videos = len(top_3_buckets) * 40  # 40 per bucket

    for bucket in top_3_buckets:
        logger.info(f"  Processing bucket: {bucket}")

        # Select 20 top + 20 bottom
        videos_to_classify = (
            manifest['videos_by_bucket'][bucket]['top_performers'][:20] +
            manifest['videos_by_bucket'][bucket]['bottom_performers'][:20]
        )

        for i, video_id in enumerate(videos_to_classify, 1):
            # Check cache
            output_path = get_classification_path(client_id, hashtag, bucket, video_id)
            if os.path.exists(output_path):
                logger.debug(f"    [{i}/40] ⏭️  Skipping {video_id} (cached)")
                continue

            # Load inputs
            transcript = load_transcript(video_id)
            caption, hashtags = load_caption_and_hashtags(video_id)

            # Classify (with retry logic)
            try:
                classification = classify_video_llm(
                    video_id=video_id,
                    transcript=transcript,
                    caption=caption,
                    hashtags=hashtags,
                    taxonomy=taxonomy,
                    client=client
                )

                # Validate output
                validate_classification_output(classification)

                # Save
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_json(output_path, classification)

                logger.info(f"    [{i}/40] ✅ Classified: {video_id} (confidence: {classification['confidence']})")
                classified_count += 1

                # Inter-request delay
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"    [{i}/40] ❌ Failed: {video_id} - {str(e)}")
                raise  # Fail-fast

            # Check overall timeout (15 min)
            elapsed = time.time() - start_time
            if elapsed > 900:  # 15 minutes
                raise TimeoutError(
                    f"Classification timeout (>15 min). "
                    f"Processed {classified_count}/{total_videos} videos. "
                    f"API may be slow. Check status.anthropic.com"
                )

        logger.info(f"  ✅ Bucket {bucket} complete")

    # === 5. Summary ===
    elapsed = time.time() - start_time
    if elapsed > 600:  # 10 min warning threshold
        logger.warning(f"⚠️  Classification took {elapsed/60:.1f} min (>10 min warning threshold)")

    logger.info(f"")
    logger.info(f"✅ Stage 2.7 Complete:")
    logger.info(f"   Classified: {classified_count} videos")
    logger.info(f"   Time: {elapsed/60:.1f} minutes")
    logger.info(f"   Rate: {elapsed/classified_count:.1f}s per video")
    logger.info(f"   Output: {get_classification_dir(client_id, hashtag)}")
    logger.info(f"")

    return classified_count


def get_classification_path(client_id, hashtag, bucket, video_id):
    """Helper: Get full path for classification output"""
    base = f"/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive"
    return f"{base}/buckets/bucket_{bucket}/content_analysis/{video_id}_content.json"


def get_classification_dir(client_id, hashtag):
    """Helper: Get classification directory for summary"""
    base = f"/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive"
    return f"{base}/buckets/bucket_*/content_analysis/"
```

### C.3 Helper Functions

```python
def save_json(path, data):
    """Save JSON with pretty formatting"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """Load JSON from file"""
    with open(path, 'r') as f:
        return json.load(f)
```

---

## Document Metadata

**Creation Date**: 2025-10-14
**Last Modified**: 2025-10-14
**Authors**: Claude Code (Phase 3 HLD Generation)
**Reviewers**: [Pending]
**Approved By**: [Pending]
**Next Review Date**: [After implementation]

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-14 | Claude Code | Initial draft from Phase 3 generation (Critique + QA_ContentAnalysis) |
