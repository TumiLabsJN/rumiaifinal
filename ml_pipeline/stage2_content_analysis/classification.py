"""
Stage 2.7: Video Classification
Classify all videos using curated taxonomy + LLM

Source: ContentAnalysisCHILDTI.md Section 4 (Function 4.3)
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import anthropic

from .utils import load_json, save_json
from .validation import (
    validate_classification_inputs,
    validate_business_rules_classification,
    validate_classification_output
)
from .error_handlers import handle_graceful_skip
from .checkpoint import load_checkpoint, save_checkpoint, update_checkpoint
from .cost_tracking import log_estimated_cost, log_actual_cost

logger = logging.getLogger(__name__)

# Module-level stats for tracking extraction behavior
_extraction_stats = {"clean_responses": 0, "cleaned_responses": 0, "total": 0}


def extract_json(response_text: str, video_id: str = None) -> Dict[str, Any]:
    """
    Extract JSON from API response that may contain extra content.

    Handles:
    - Markdown code fences (```json ... ```)
    - Extra text before/after JSON
    - Whitespace issues

    Args:
        response_text: Raw text from API response
        video_id: Optional video ID for debug logging

    Returns:
        dict: Parsed JSON object

    Raises:
        ValueError: If no valid JSON found or JSON is malformed
    """
    original_text = response_text
    text = response_text.strip()

    # Step 1: Strip markdown code fences
    if text.startswith('```'):
        # Handle ```json or just ```
        text = text[3:].lstrip()
        if text.startswith('json'):
            text = text[4:].lstrip()

    if text.endswith('```'):
        text = text[:-3].rstrip()

    text = text.strip()

    # Step 2: Find first complete JSON object using brace counting
    # This handles cases where LLM returns multiple JSON objects
    first_brace = text.find('{')

    if first_brace == -1:
        # Log the failure case for debugging
        logger.error(
            f"No JSON braces found in response for {video_id}. "
            f"Response preview: {original_text[:200]}..."
        )
        raise ValueError("No valid JSON braces found in response")

    # Step 3: Find matching closing brace by counting depth
    brace_count = 0
    last_brace = -1

    for i in range(first_brace, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                # Found matching closing brace for first opening brace
                last_brace = i
                break

    if last_brace == -1:
        logger.error(
            f"No matching closing brace found for {video_id}. "
            f"first_brace={first_brace}, text preview: {repr(text[first_brace:first_brace+200])}"
        )
        raise ValueError("No matching closing brace found")

    # Step 4: Extract first complete JSON object
    json_text = text[first_brace:last_brace + 1]

    # Step 4: Attempt to parse
    try:
        parsed = json.loads(json_text)

        # Track extraction stats
        _extraction_stats["total"] += 1
        if len(json_text) < len(text):
            _extraction_stats["cleaned_responses"] += 1
            # Log if we extracted extra content (for monitoring)
            extra_before = text[:first_brace].strip()
            extra_after = text[last_brace + 1:].strip()
            if extra_before or extra_after:
                logger.debug(
                    f"Extracted JSON for {video_id}, removed extra content. "
                    f"Before: {len(extra_before)} chars, After: {len(extra_after)} chars"
                )
        else:
            _extraction_stats["clean_responses"] += 1

        return parsed

    except json.JSONDecodeError as e:
        # JSON extraction found braces but content is malformed
        # Enhanced debugging: Log complete context for troubleshooting
        logger.error(
            f"JSONDecodeError for {video_id}: {str(e)}\n"
            f"  original_text length: {len(original_text)}\n"
            f"  original_text preview: {repr(original_text[:500])}\n"
            f"  json_text length: {len(json_text)}\n"
            f"  json_text: {repr(json_text)}\n"
            f"  first_brace: {first_brace}, last_brace: {last_brace}"
        )
        raise ValueError(f"Extracted text is not valid JSON: {str(e)}")


def classify_video_llm(
    video_id: str,
    transcript: Dict[str, Any],
    caption: str,
    hashtags: List[str],
    taxonomy: Dict[str, Any],
    client: anthropic.Anthropic
) -> Dict[str, Any]:
    """
    Classify single video using LLM + taxonomy.

    Source: ContentAnalysisCHILDTI.md Section 4.3

    Args:
        video_id: Video identifier
        transcript: {"text": str, "available": bool}
        caption: Creator-written caption (can be empty string)
        hashtags: List of hashtag names without # (can be empty list)
        taxonomy: Curated taxonomy from Stage 2.6
        client: Initialized Anthropic API client

    Returns:
        dict: Classification JSON with 12 fields

    Raises:
        TimeoutError: If LLM exceeds 30s timeout per video after 3 retries
        ValueError: If LLM returns invalid JSON after 3 retries
    """
    # Step 1: Build classification prompt with taxonomy + video data
    # Source: 2.7ClassificationCritique.md - Final Refined Prompt

    # Build system message
    system_message = """You are an expert content classifier specializing in short-form video analysis. Your task is to accurately classify videos using a predefined taxonomy that was empirically discovered from real video data in this hashtag (Stage 2.6).

Be objective and evidence-based: select classifications that best match the video content based on transcript, caption, and hashtags. Use taxonomy categories EXACTLY as defined - do not reinterpret or expand their meaning. When evidence is ambiguous, note lower confidence rather than forcing a classification."""

    # Build main prompt
    transcript_text = transcript['text'] if transcript['available'] else "(No transcript available - classify using caption and hashtags)"

    prompt = f"""## ZONE 1: TAXONOMY & CORE CLASSIFICATION

### Provided Taxonomy

**Category 1: Content Categories** (Single Selection)
{json.dumps(taxonomy['content_categories'], indent=2)}

**Category 2: Hook Strategies** (Single Selection)
{json.dumps(taxonomy['hook_strategies'], indent=2)}

**Category 3: Closing Strategies** (Single Selection)
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

**Categories 1-3: Single Selection (REQUIRED)**

**Content Category**: Select exactly ONE category that best describes the primary content format.

**Hook Strategy**: Select exactly ONE strategy that best describes how the video opens.

**Closing Strategy**: Select exactly ONE strategy that best describes how the video ends.

**IMPORTANT**: You MUST copy the category name EXACTLY as written in the taxonomy. Do not paraphrase, abbreviate, or modify the string. Mismatched spelling or underscores will cause system errors.

**If no perfect match exists**: Select the closest matching category from the taxonomy. Set confidence=low and document the mismatch in the note field (e.g., "Video is comedy skit, closest match is wellness_practice").

**String Matching**: Copy category names character-for-character from taxonomy above.

**Categories 4-7: Multiple Selection (0-N)**

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

**CRITICAL - String Matching for caption_analysis:** You must copy field values EXACTLY as written below. Do not use synonyms, variations, or values from taxonomy categories.

### Correct Output Format ✅

{{
  "caption_analysis": {{
    "hook_type": "statement",
    "cta_type": "link_in_bio",
    "brand_mention_present": true,
    "influencer_tag_present": false,
    "emoji_usage": "some",
    "caption_length": "short",
    "hashtag_count": 3,
    "hashtag_placement": "end"
  }}
}}

### Wrong Output Format ❌

{{
  "caption_analysis": {{
    "hook_type": "declarative_statement",  ← WRONG! Use "statement"
    "cta_type": "bio_link",                ← WRONG! Use "link_in_bio"
    "emoji_usage": "few"                   ← WRONG! Use "some"
  }}
}}

### Allowed Values (Copy Exactly)

**hook_type** - How caption opens (first 5-10 words):
- "statement" | "question" | "command" | "teaser"

**cta_type** - Call-to-action in caption:
- "link_in_bio" | "save_post" | "comment" | "follow" | "share" | "tag_friend" | "none"

**emoji_usage** - Count emojis in caption:
- "none" | "some" | "many"

**caption_length** - Character count:
- "short" | "long"

**hashtag_placement** - Where hashtags appear:
- "end" | "mixed" | "none"

**brand_mention_present** - Caption mentions brand/product:
- true | false

**influencer_tag_present** - Caption tags another creator:
- true | false

**hashtag_count** - Total number of hashtags:
- integer (e.g., 0, 3, 15)

### Field Definitions

- **hook_type**: "statement" (declarative), "question" (interrogative), "command" (imperative), "teaser" (curiosity-building)
- **emoji_usage**: "none" (0 emojis), "some" (1-4 emojis), "many" (5+ emojis)
- **caption_length**: "short" (< 100 characters), "long" (≥ 100 characters)
- **hashtag_placement**: "end" (all hashtags at caption end), "mixed" (hashtags throughout), "none" (no hashtags present)

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

Return a single JSON object with ALL 13 fields present. Do not add fields beyond this schema.

**Required fields** (must be non-null):
- video_id: String (provided in input)
- taxonomy_version: Always use "stage2.6_output"
- content_category: String (exactly one from taxonomy)
- hook_strategy: String (exactly one from taxonomy)
- closing_strategy: String (exactly one from taxonomy)
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

    # Step 2: Call API with retry logic (3 attempts with exponential backoff)
    for attempt in range(3):
        try:
            # Step 2.1: Make API call with Haiku model
            # Updated: M8 Enhancement - Add latency tracking
            start_time = time.time()
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                timeout=30,  # 30 seconds per video
                system=system_message,
                messages=[{"role": "user", "content": prompt}]
            )

            # Step 2.1a: Log actual cost with latency
            # Source: ContentAnalysisCHILDTI.md Section 4.6.4
            log_actual_cost(response, "haiku", start_time)

            # Step 2.2: Parse response with robust extraction
            response_text = response.content[0].text

            # Optional debug logging (only if DEBUG level enabled)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Raw API response for {video_id} ({len(response_text)} chars): "
                    f"{response_text[:500]}..."
                )

            # Extract and parse JSON (handles markdown fences and extra content)
            classification = extract_json(response_text, video_id)

            # Step 2.3: Validate output schema before returning
            validate_classification_output(classification)

            # Step 2.4: Return successful classification
            return classification

        except (TimeoutError, anthropic.APIError) as e:
            # Step 2.5: Handle timeout/API errors (retry with backoff)
            if attempt < 2:
                delay = [1, 2, 4][attempt]  # Exponential backoff
                logger.warning(f"API failed for {video_id}, retry {attempt+1} in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"API failed for {video_id} after 3 retries")
                raise  # Re-raise after final retry

        except json.JSONDecodeError as e:
            # Step 2.6: Handle invalid JSON (retry)
            if attempt < 2:
                delay = [1, 2, 4][attempt]
                logger.warning(f"Invalid JSON for {video_id}, retry {attempt+1} in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"Invalid JSON for {video_id} after 3 retries: {str(e)}")
                raise ValueError(f"LLM returned invalid JSON: {str(e)}")

    # Unreachable (for type checker)
    raise RuntimeError("Unexpected retry loop exit")


def build_video_data_cache(target_dir) -> Dict[str, Dict[str, Any]]:
    """
    Build hash map of caption/hashtag data from selected_videos.json files.

    Called ONCE at start of classification stage for O(1) lookups.
    Reads directly from selected_videos.json (no extraction needed).

    Args:
        target_dir: Target directory (Path or str)

    Returns:
        dict: {video_id: {'caption': str, 'hashtags': list, 'text_language': str}}

    Source: Direct reading solution (replaces Stage 2.6.5 extraction)
    """
    from pathlib import Path

    target_dir = Path(target_dir)
    video_data_map = {}

    # Load winner_analysis.json to get winning buckets
    winner_analysis_path = target_dir / "winner_analysis.json"
    try:
        winner_analysis = load_json(str(winner_analysis_path))
        winning_buckets = winner_analysis.get('winning_buckets', [])
    except FileNotFoundError:
        logger.error(f"winner_analysis.json not found at {winner_analysis_path}")
        raise

    # Load caption/hashtag data from each winning bucket's selected_videos.json
    for bucket_name in winning_buckets:
        selected_videos_path = target_dir / "buckets" / f"bucket_{bucket_name}" / "selected_videos.json"

        try:
            selected_videos = load_json(str(selected_videos_path))

            # Build hash map from videos array
            for video in selected_videos.get('videos', []):
                video_id = video.get('id')
                if video_id:
                    video_data_map[video_id] = {
                        'caption': video.get('text', ''),
                        'hashtags': video.get('hashtags', []),
                        'text_language': video.get('textLanguage', 'unknown')
                    }

        except FileNotFoundError:
            logger.warning(f"selected_videos.json not found for bucket {bucket_name}")
            continue
        except Exception as e:
            logger.error(f"Error loading selected_videos.json for bucket {bucket_name}: {e}")
            continue

    logger.info(f"✓ Cached caption/hashtag data for {len(video_data_map)} videos from selected_videos.json")
    return video_data_map


def load_video_data(video_id: str, video_data_cache: Dict[str, Dict[str, Any]] = None) -> tuple:
    """
    Load transcript, caption, and hashtags for a video.

    Args:
        video_id: Video identifier
        video_data_cache: Pre-built hash map from build_video_data_cache()
                         If None, falls back to legacy file reading (backward compat)

    Returns:
        tuple: (transcript_dict, caption_str, hashtags_list)
    """
    RUMIAI_ROOT = os.environ.get('RUMIAI_ROOT', '/home/jorge/rumiaifinal')
    transcript_path = f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json"

    # Load transcript (unchanged - still from global location)
    try:
        transcript_data = load_json(transcript_path)
        transcript = {
            'text': transcript_data.get('text', ''),
            'available': True
        }
    except FileNotFoundError:
        transcript = {'text': '', 'available': False}
        logger.warning(f"No transcript for {video_id}")

    # Load caption and hashtags from cache (O(1)) or fall back to file reading
    if video_data_cache is not None:
        # Use cache (direct reading from selected_videos.json)
        video_data = video_data_cache.get(video_id, {})
        caption = video_data.get('caption', '')
        hashtags = video_data.get('hashtags', [])

        if not caption and not hashtags:
            logger.warning(f"No caption/hashtags in cache for {video_id}")
    else:
        # Legacy fallback: Try to load from extracted files (backward compatibility)
        caption_path = f"{RUMIAI_ROOT}/video_captions/{video_id}_caption.json"
        hashtags_path = f"{RUMIAI_ROOT}/video_hashtags/{video_id}_hashtags.json"

        try:
            caption_data = load_json(caption_path)
            caption = caption_data.get('text', '')
        except FileNotFoundError:
            caption = ''

        try:
            hashtags_data = load_json(hashtags_path)
            hashtags = hashtags_data.get('hashtags', [])
        except FileNotFoundError:
            hashtags = []

    return transcript, caption, hashtags


def classify_single_video_with_save(
    video_id: str,
    taxonomy: Dict[str, Any],
    client: anthropic.Anthropic,
    output_dir: str,
    manifest: Optional[Dict[str, Any]] = None,
    validation_cache: Optional[Dict[str, Any]] = None,
    video_data_cache: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Classify video using dual-flow approach (if manifest provided) or legacy single-flow.

    DUAL-FLOW MODE (manifest provided):
    - Flow 1 (valid transcript): Full classification (transcript + caption)
    - Flow 2 (invalid transcript): Caption analysis only

    LEGACY MODE (manifest=None):
    - Always attempts full classification (backwards compatible)

    Source: ContentAnalysispt2.md Step 4 (Dual-Flow Classification)

    Args:
        video_id: Video identifier
        taxonomy: Curated taxonomy
        client: Anthropic API client
        output_dir: Directory to save classification result
        manifest: Optional selection_manifest.json (enables dual-flow)
        validation_cache: Optional validation cache for checking transcript validity
        video_data_cache: Optional pre-built hash map of caption/hashtag data

    Returns:
        dict: Classification result

    Raises:
        Exception: Any error during classification
    """
    # Load video data (with cache for O(1) caption/hashtag lookup)
    transcript, caption, hashtags = load_video_data(video_id, video_data_cache)

    # DUAL-FLOW MODE: Route based on transcript validity
    if manifest is not None and validation_cache is not None:
        # Get bucket and performer type for this video
        bucket, performer_type = get_bucket_for_video(video_id, manifest)

        # Check if transcript is valid
        is_valid_transcript = validation_cache.get(video_id, {}).get('is_valid', False)

        if is_valid_transcript:
            # Flow 1: Full classification
            logger.info(f"📋 Flow 1 (With Transcript): {video_id} [bucket: {bucket}]")

            llm_output = classify_video_with_transcript(
                video_id=video_id,
                transcript=transcript,
                caption=caption,
                hashtags=hashtags,
                taxonomy=taxonomy,
                client=client
            )

            # Save RAW LLM output
            raw_output_dir = f"{output_dir}/raw_llm_output/bucket_{bucket}"
            os.makedirs(raw_output_dir, exist_ok=True)
            raw_output_path = f"{raw_output_dir}/{video_id}_raw.json"
            save_json(raw_output_path, llm_output)

            # Normalize schema (adds hashtag_count)
            classification = normalize_classification_schema(
                llm_output=llm_output,
                video_id=video_id,
                caption=caption,
                transcript_available=True,
                flow_type="full"
            )

        else:
            # Flow 2: Caption only
            logger.info(f"📋 Flow 2 (Caption Only): {video_id} [bucket: {bucket}]")

            llm_output = classify_caption_only(
                video_id=video_id,
                caption=caption,
                hashtags=hashtags,
                client=client
            )

            # Save RAW LLM output
            raw_output_dir = f"{output_dir}/raw_llm_output/bucket_{bucket}"
            os.makedirs(raw_output_dir, exist_ok=True)
            raw_output_path = f"{raw_output_dir}/{video_id}_raw.json"
            save_json(raw_output_path, llm_output)

            # Normalize schema (fills in defaults)
            classification = normalize_classification_schema(
                llm_output=llm_output,
                video_id=video_id,
                caption=caption,
                transcript_available=False,
                flow_type="caption_only"
            )

        # Add bucket metadata
        classification['bucket'] = bucket
        classification['performer_type'] = performer_type  # "top" or "bottom"

        # Save VALIDATED output (final)
        validated_output_dir = f"{output_dir}/validated/bucket_{bucket}"
        os.makedirs(validated_output_dir, exist_ok=True)
        validated_output_path = f"{validated_output_dir}/{video_id}_content.json"
        save_json(validated_output_path, classification)

    else:
        # LEGACY MODE: Original single-flow classification
        logger.info(f"📋 Legacy Mode (Single-Flow): {video_id}")

        # Validate business rules
        validate_business_rules_classification(video_id, transcript, caption, hashtags)

        # Classify video (original flow)
        classification = classify_video_llm(
            video_id=video_id,
            transcript=transcript,
            caption=caption,
            hashtags=hashtags,
            taxonomy=taxonomy,
            client=client
        )

        # Save classification
        output_path = f"{output_dir}/{video_id}_content.json"
        save_json(output_path, classification)

    return classification


def classify_all_videos_sequential(
    videos: List[str],
    taxonomy: Dict[str, Any],
    client: anthropic.Anthropic,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    manifest: Optional[Dict[str, Any]] = None,
    validation_cache: Optional[Dict[str, Any]] = None,
    video_data_cache: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Classify videos sequentially with checkpoint/resume support.

    Dual-Flow Support: If manifest and validation_cache provided, uses Flow 1/Flow 2 routing.

    Source: ContentAnalysisCHILDTI.md Section 4.4.1, 4.5.4 + ContentAnalysispt2.md Step 4

    Args:
        videos: List of video IDs to classify
        taxonomy: Curated taxonomy
        client: Anthropic API client
        output_dir: Directory to save results
        checkpoint_path: Optional path to checkpoint file
        manifest: Optional manifest for dual-flow classification
        validation_cache: Optional validation cache for dual-flow classification
        video_data_cache: Optional pre-built hash map of caption/hashtag data

    Returns:
        dict: {"completed": int, "failed": int, "failed_ids": list[str]}
    """
    # Step 1: Load checkpoint (if enabled)
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        remaining_videos = [v for v in videos if v not in checkpoint["completed"]]
        logger.info(
            f"📋 Sequential mode with checkpoints: "
            f"{len(checkpoint['completed'])} already completed, "
            f"{len(remaining_videos)} remaining"
        )
    else:
        remaining_videos = videos
        checkpoint = None

    completed = 0
    failed = 0
    failed_ids = []

    # Step 2: Process remaining videos
    for i, video_id in enumerate(remaining_videos):
        try:
            # Classify and save (with optional dual-flow parameters and cache)
            classify_single_video_with_save(
                video_id, taxonomy, client, output_dir, manifest, validation_cache, video_data_cache
            )

            # Update checkpoint (if enabled)
            if checkpoint_path:
                update_checkpoint(checkpoint, video_id, "completed", checkpoint_path)

            completed += 1
            logger.debug(f"✅ Classified {i+1}/{len(remaining_videos)}: {video_id}")

        except Exception as e:
            # Update checkpoint for failure (if enabled)
            if checkpoint_path:
                update_checkpoint(checkpoint, video_id, "failed", checkpoint_path)

            failed += 1
            failed_ids.append(video_id)
            logger.error(f"❌ Failed {i+1}/{len(remaining_videos)}: {video_id} - {str(e)}")

    # Step 3: Log extraction stats
    if _extraction_stats["total"] > 0:
        logger.info(
            f"📊 Extraction stats: {_extraction_stats['cleaned_responses']}/{_extraction_stats['total']} "
            f"responses needed cleaning ({_extraction_stats['cleaned_responses'] / _extraction_stats['total'] * 100:.1f}%)"
        )

    # Step 4: Return results
    total_completed = completed + (len(checkpoint["completed"]) if checkpoint else 0)
    return {
        "completed": total_completed,
        "failed": failed,
        "failed_ids": failed_ids
    }


def classify_all_videos_parallel(
    videos: List[str],
    taxonomy: Dict[str, Any],
    client: anthropic.Anthropic,
    output_dir: str,
    max_workers: int = 5,
    checkpoint_path: Optional[str] = None,
    manifest: Optional[Dict[str, Any]] = None,
    validation_cache: Optional[Dict[str, Any]] = None,
    video_data_cache: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Classify videos in parallel with checkpoint/resume support.

    Dual-Flow Support: If manifest and validation_cache provided, uses Flow 1/Flow 2 routing.

    Source: ContentAnalysisCHILDTI.md Section 4.4.2, 4.5.5 + ContentAnalysispt2.md Step 4

    Args:
        videos: List of video IDs
        taxonomy: Curated taxonomy
        client: Anthropic API client
        output_dir: Directory to save results
        max_workers: Concurrent workers (default: 5)
        checkpoint_path: Optional checkpoint file path
        manifest: Optional manifest for dual-flow classification
        validation_cache: Optional validation cache for dual-flow classification
        video_data_cache: Optional pre-built hash map of caption/hashtag data

    Returns:
        dict: {"completed": int, "failed": int, "failed_ids": list[str]}
    """
    # Step 1: Load checkpoint and filter completed videos
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        remaining_videos = [v for v in videos if v not in checkpoint["completed"]]
        checkpoint_lock = threading.Lock()  # Thread-safe checkpoint updates
        logger.info(
            f"🚀 Parallel mode with checkpoints: "
            f"{len(checkpoint['completed'])} already completed, "
            f"{len(remaining_videos)} remaining"
        )
    else:
        remaining_videos = videos
        checkpoint = None
        checkpoint_lock = None

    completed = 0
    failed = 0
    failed_ids = []

    # Step 2: Submit all tasks to thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all classification tasks (with optional dual-flow parameters and cache)
        future_to_video = {
            executor.submit(
                classify_single_video_with_save,
                video_id, taxonomy, client, output_dir, manifest, validation_cache, video_data_cache
            ): video_id
            for video_id in remaining_videos
        }

        # Step 3: Process results as they complete
        for future in as_completed(future_to_video):
            video_id = future_to_video[future]
            try:
                result = future.result()  # Blocks until this video completes

                # Update checkpoint (thread-safe)
                if checkpoint_path:
                    with checkpoint_lock:
                        update_checkpoint(checkpoint, video_id, "completed", checkpoint_path)

                completed += 1
                logger.debug(f"✅ Classified ({completed}/{len(remaining_videos)}): {video_id}")

            except Exception as e:
                # Update checkpoint for failure (thread-safe)
                if checkpoint_path:
                    with checkpoint_lock:
                        update_checkpoint(checkpoint, video_id, "failed", checkpoint_path)

                failed += 1
                failed_ids.append(video_id)
                logger.error(f"❌ Failed classification: {video_id} - {str(e)}")

    # Step 4: Log extraction stats
    if _extraction_stats["total"] > 0:
        logger.info(
            f"📊 Extraction stats: {_extraction_stats['cleaned_responses']}/{_extraction_stats['total']} "
            f"responses needed cleaning ({_extraction_stats['cleaned_responses'] / _extraction_stats['total'] * 100:.1f}%)"
        )

    # Step 5: Return results
    total_completed = completed + (len(checkpoint["completed"]) if checkpoint else 0)
    return {
        "completed": total_completed,
        "failed": failed,
        "failed_ids": failed_ids
    }


def classify_all_videos(
    client_id: str,
    hashtag: str,
    analysis_type: str,
    taxonomy: Dict[str, Any],
    manifest: Dict[str, Any],
    anthropic_client: anthropic.Anthropic,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive",
    parallel: bool = False,
    max_workers: int = 5,
    checkpoint_enabled: bool = True,
    validation_cache: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Classify all videos with configurable parallel/sequential execution and checkpoints.

    Dual-Flow Support (ContentAnalysispt2.md Step 4):
    - If validation_cache provided: Routes to Flow 1 (transcript) or Flow 2 (caption-only)
    - If validation_cache=None: Uses legacy single-flow classification

    Source: ContentAnalysisCHILDTI.md Section 4.4 + ContentAnalysispt2.md Step 4

    Args:
        client_id: Client identifier
        hashtag: Target name
        analysis_type: "hashtag", "competitor", or "creator"
        taxonomy: Curated taxonomy from Stage 2.6
        manifest: Selection manifest from Stage 2.5
        anthropic_client: Initialized Anthropic API client
        analysis_mode: "top" or "recent"
        selection_strategy: "contrastive" or "top"
        parallel: Enable parallel mode (default: False)
        max_workers: Workers for parallel mode (default: 5)
        checkpoint_enabled: Enable checkpoint/resume (default: True)
        validation_cache: Optional validation cache for dual-flow classification

    Returns:
        dict: Summary statistics with mode, duration, completed/failed counts
    """
    # Collect all videos from all buckets
    all_videos = []
    for bucket in manifest['selected_buckets']:
        bucket_videos = manifest['videos_by_bucket'][bucket]
        all_videos.extend(bucket_videos['top_performers'])
        all_videos.extend(bucket_videos.get('bottom_performers', []))

    # Log estimated cost
    log_estimated_cost("classification", video_count=len(all_videos))

    # Determine output directory using PathBuilder
    from foundation.paths import PathBuilder
    path_builder = PathBuilder()
    target_dir = path_builder.get_target_dir(
        client_id=client_id,
        analysis_type=analysis_type,
        target=hashtag,
        analysis_mode=analysis_mode,
        selection_strategy=selection_strategy
    )
    output_dir = str(target_dir / "content_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Set up checkpoint path
    checkpoint_path = None
    if checkpoint_enabled:
        checkpoint_dir = str(target_dir / ".checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{checkpoint_dir}/classification_checkpoint.json"

    # Build video data cache (caption/hashtag data from selected_videos.json)
    logger.info("Building caption/hashtag cache from selected_videos.json...")
    video_data_cache = build_video_data_cache(target_dir)

    # Execute classification
    start_time = time.time()

    if parallel:
        logger.info(f"🚀 Starting parallel classification: {len(all_videos)} videos, {max_workers} workers")
        results = classify_all_videos_parallel(
            all_videos, taxonomy, anthropic_client, output_dir, max_workers, checkpoint_path,
            manifest, validation_cache, video_data_cache
        )
    else:
        logger.info(f"📋 Starting sequential classification: {len(all_videos)} videos")
        results = classify_all_videos_sequential(
            all_videos, taxonomy, anthropic_client, output_dir, checkpoint_path,
            manifest, validation_cache, video_data_cache
        )

    # Calculate summary statistics
    duration = time.time() - start_time
    summary = {
        "total": len(all_videos),
        "completed": results["completed"],
        "failed": results["failed"],
        "failed_ids": results["failed_ids"],
        "mode": "parallel" if parallel else "sequential",
        "duration_seconds": round(duration, 2)
    }

    logger.info(
        f"✅ Classification complete: {summary['completed']}/{summary['total']} videos "
        f"({summary['mode']} mode, {summary['duration_seconds']}s)"
    )

    return summary


def run_classification_stage(
    client_id: str,
    hashtag: str,
    analysis_type: str,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive",
    parallel: bool = None,
    max_workers: int = 5,
    checkpoint_enabled: bool = True
) -> Dict[str, Any]:
    """
    Run complete Stage 2.7 classification pipeline with error handling.

    Source: ContentAnalysisCHILDTI.md Section 6.3

    Args:
        client_id: Client identifier (e.g., "acme_corp")
        hashtag: Target name (e.g., "nutrition" or "cindyprado28")
        analysis_type: "hashtag", "competitor", or "creator"
        analysis_mode: "top" or "recent" (default: "top")
        selection_strategy: "contrastive" or "top" (default: "contrastive")
        parallel: Enable parallel mode (default: reads ENABLE_PARALLEL_CLASSIFICATION env var)
        max_workers: Workers for parallel mode (default: reads MAX_CLASSIFICATION_WORKERS env var or 5)
        checkpoint_enabled: Enable checkpoint/resume (default: True)

    Returns:
        dict: Summary statistics from classification

    Raises:
        FileNotFoundError: If taxonomy or manifest missing
        ValueError: If validation fails
    """
    logger.info(f"=" * 80)
    logger.info(f"STAGE 2.7: VIDEO CLASSIFICATION")
    logger.info(f"Client: {client_id}, Target: {hashtag}, Type: {analysis_type}")
    logger.info(f"=" * 80)

    # Step 1: Read environment variables for configuration
    # Source: ContentAnalysisCHILDTI.md Section 9.1
    if parallel is None:
        parallel = os.environ.get('ENABLE_PARALLEL_CLASSIFICATION', 'false').lower() == 'true'

    max_workers = int(os.environ.get('MAX_CLASSIFICATION_WORKERS', str(max_workers)))

    # Step 2: Construct paths using PathBuilder
    from foundation.paths import PathBuilder, sanitize_target
    path_builder = PathBuilder()
    target_dir = path_builder.get_target_dir(
        client_id=client_id,
        analysis_type=analysis_type,
        target=hashtag,
        analysis_mode=analysis_mode,
        selection_strategy=selection_strategy
    )
    target_sanitized = sanitize_target(hashtag, analysis_type)
    taxonomy_path = str(target_dir / "content_taxonomies" / f"{target_sanitized}_taxonomy.json")
    manifest_path = str(target_dir / "selection_manifest.json")

    # Step 3: Validate inputs
    logger.info("Step 1/3: Validating inputs...")
    validate_classification_inputs(taxonomy_path, manifest_path)
    logger.info("✓ Input validation passed")

    # Step 4: Load taxonomy and manifest
    logger.info("Step 2/4: Loading taxonomy and manifest...")
    taxonomy = load_json(taxonomy_path)
    manifest = load_json(manifest_path)
    logger.info("✓ Files loaded")

    # Step 4a: Load validation cache (for dual-flow classification)
    logger.info("Step 3/4: Loading transcript validation cache...")
    try:
        from .transcript_validation import load_validation_cache
        validation_cache = load_validation_cache(
            client_id, hashtag, analysis_type, analysis_mode, selection_strategy
        )
        logger.info(f"✓ Validation cache loaded - dual-flow classification enabled")
    except FileNotFoundError:
        logger.warning(
            "⚠️  Validation cache not found - using legacy single-flow classification.\n"
            "   Run Stage 2.5.1 (Transcript Validation) for dual-flow support."
        )
        validation_cache = None
    except Exception as e:
        logger.warning(f"⚠️  Failed to load validation cache: {e} - using legacy mode")
        validation_cache = None

    # Step 5: Initialize Anthropic client
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Step 6: Classify all videos
    logger.info("Step 4/4: Classifying all videos (Claude Haiku)...")
    summary = classify_all_videos(
        client_id=client_id,
        hashtag=hashtag,
        analysis_type=analysis_type,
        taxonomy=taxonomy,
        manifest=manifest,
        anthropic_client=anthropic_client,
        analysis_mode=analysis_mode,
        selection_strategy=selection_strategy,
        parallel=parallel,
        max_workers=max_workers,
        checkpoint_enabled=checkpoint_enabled,
        validation_cache=validation_cache
    )

    logger.info(f"=" * 80)
    logger.info(f"STAGE 2.7 COMPLETE")
    logger.info(f"Mode: {summary['mode']}")
    logger.info(f"Total videos: {summary['total']}")
    logger.info(f"Completed: {summary['completed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Duration: {summary['duration_seconds']}s")
    logger.info(f"=" * 80)

    return summary

# ============================================================================
# STEP 4: DUAL-FLOW CLASSIFICATION (ContentAnalysispt2.md)
# Added: 2025-10-31
# Purpose: Classify videos with valid OR invalid transcripts
# ============================================================================


def get_bucket_for_video(
    video_id: str,
    manifest: Dict[str, Any]
) -> tuple:
    """
    Find which bucket a video belongs to and its performer type.

    Args:
        video_id: Video identifier
        manifest: selection_manifest.json from Stage 2.5

    Returns:
        tuple: (bucket, performer_type) where:
            - bucket: str (e.g., "33-60s")
            - performer_type: str ("top" or "bottom")

    Raises:
        ValueError: If video not found in any bucket
    """
    for bucket, bucket_data in manifest['videos_by_bucket'].items():
        if video_id in bucket_data['top_performers']:
            return bucket, "top"
        if video_id in bucket_data['bottom_performers']:
            return bucket, "bottom"

    raise ValueError(f"Video {video_id} not found in any bucket")


def normalize_classification_schema(
    llm_output: Dict[str, Any],
    video_id: str,
    caption: str,
    transcript_available: bool,
    flow_type: str
) -> Dict[str, Any]:
    """
    Normalize LLM output to final 15-field schema.
    
    M10 FIX: Calculate hashtag_count from caption (not from LLM).
    
    Args:
        llm_output: Validated output from LLM
        video_id: Video identifier
        caption: Video caption text
        transcript_available: True if Flow 1, False if Flow 2
        flow_type: "full" or "caption_only"
        
    Returns:
        Normalized classification with all 15 fields
    """
    # Calculate hashtag_count from caption (M10 FIX)
    hashtag_count = len([word for word in (caption or "").split() if word.startswith('#')])
    
    if flow_type == "full":
        # Flow 1: Use all LLM fields, just add hashtag_count
        normalized = llm_output.copy()
        normalized['caption_analysis']['hashtag_count'] = hashtag_count
        
    else:  # caption_only
        # Flow 2: Construct full schema with defaults
        normalized = {
            "video_id": video_id,
            "taxonomy_version": "none_no_transcript",
            "content_category": None,
            "hook_strategy": None,
            "closing_strategy": None,
            "pain_points": [],
            "keywords": [],
            "engagement_drivers": [],
            "content_tactics": [],
            "caption_analysis": llm_output.get("caption_analysis", {}),
            "confidence": "n/a",
            "transcript_available": False,
            "note": "No valid transcript - caption analysis only"
        }
        normalized['caption_analysis']['hashtag_count'] = hashtag_count
    
    return normalized


def classify_video_with_transcript(
    video_id: str,
    transcript: str,
    caption: str,
    hashtags: List[str],
    taxonomy: Dict[str, Any],
    client: Any
) -> Dict[str, Any]:
    """
    Flow 1: Full classification using transcript.
    
    LLM outputs ALL 13 fields (Python adds hashtag_count later).
    Uses physical zone separation to prevent hallucination.
    
    Source: ContentAnalysispt2.md Step 4 (Lines 700-1010)
    
    Args:
        video_id: Video identifier
        transcript: Transcript text (validated as non-empty)
        caption: Caption text
        hashtags: List of hashtags
        taxonomy: Curated taxonomy from Stage 2.6
        client: Anthropic API client
        
    Returns:
        Dict with 13 fields (transcript_available=True)
    """
    import json
    
    system_message = """You are an expert TikTok content classifier. Your task is to classify videos using a data-driven taxonomy discovered from real transcripts in this hashtag.

Be objective and evidence-based. Only select classifications that have explicit support in the transcript. The taxonomy was created from analyzing top-performing videos, so you must ground all selections in what was actually said."""

    prompt = f"""# CLASSIFICATION TASK

You will classify a TikTok video using the taxonomy below. The taxonomy was empirically discovered from analyzing 60 top-performing video transcripts in this hashtag.

**Video ID**: {video_id}

---

## ZONE 1: CONTENT CLASSIFICATION (Transcript Only)

### Taxonomy (Data-Driven Patterns)

{json.dumps(taxonomy, indent=2)}

---

### Video Transcript

Below is what was SPOKEN in the video. Use ONLY this transcript for content classification.

```
{transcript}
```

**Note**: You will see the caption and hashtags in Zone 2, but do NOT use them for ANY content classification fields below. Zone 1 uses transcript ONLY.

---

### Classification Instructions

Classify this video using the taxonomy above. Copy all category names EXACTLY as written (character-for-character).

#### Categories 1-3: Single Selection (REQUIRED)

Select exactly ONE option for each field:

**content_category**:
- What type of content is this? (e.g., "wellness_routine", "supplement_review")
- Select ONE from taxonomy that best matches the video's overall format and purpose
- Copy the category name EXACTLY from taxonomy

**hook_strategy**:
- How does the video open?
- Analyze the first 5-10 spoken words in the transcript
- Select ONE from taxonomy that matches how the speaker began
- Copy the strategy name EXACTLY from taxonomy

**closing_strategy**:
- How does the video end?
- Analyze the last 10 spoken words in the transcript
- Select ONE from taxonomy that matches how the speaker concluded
- Copy the strategy name EXACTLY from taxonomy

---

#### Categories 4-7: Multiple Selection (0-N items)

Select ZERO or MORE items from taxonomy. Empty arrays [] are acceptable and encouraged when no match exists.

**pain_points**:
- Which problems or challenges are mentioned in the transcript?
- Only include if:
  1. **Explicitly stated** (e.g., "I had bloating") OR
  2. **Strongly implied from solutions** (e.g., "I started probiotics and my gut issues went away" → "gut_issues" is pain point)
- Do NOT infer based on video topic alone
- Copy pain point names EXACTLY from taxonomy
- Return [] if no pain points mentioned

**keywords**:
- What topics, methods, or products are central to this video?
- Only include if explicitly mentioned in the transcript
- Do NOT infer from context
- Copy keyword names EXACTLY from taxonomy
- Return [] if no keywords match

**engagement_drivers**:
- What tactics does the creator use to engage viewers?
- Only include if observable from what the creator says or how they present
- Examples: "before_after_reveal", "relatable_struggles", "humor_and_satire"
- Copy driver names EXACTLY from taxonomy
- Return [] if no drivers evident

**content_tactics**:
- What presentation styles are used in this video?
- Only include if evident from transcript or explicitly mentioned
- Examples: "direct_to_camera", "voiceover_narration", "step_by_step"
- Copy tactic names EXACTLY from taxonomy
- Return [] if no tactics evident

---

### Evidence Rules (CRITICAL)

**Source Restrictions**:
- ALL 7 fields (content_category, hook_strategy, closing_strategy, pain_points, keywords, engagement_drivers, content_tactics): **Transcript ONLY**

**Selection Rules**:
- Only select items that exist in the taxonomy
- Copy names character-for-character (exact match)
- Empty arrays acceptable if no match
- Do NOT infer or hallucinate

---

### Examples: VALID Classifications ✅

**Example 1: Explicit pain point**
```
Transcript: "I had terrible bloating after every meal"
→ pain_points: ["bloating"] ✅
Reasoning: Explicitly stated problem
```

**Example 2: Implied pain point from solution**
```
Transcript: "I started taking probiotics and my gut issues completely went away"
→ pain_points: ["gut_issues"] ✅
Reasoning: Solution mentioned → problem implied (gut issues went away)
```

**Example 3: Empty array when no match**
```
Transcript: "I love my morning wellness routine"
→ pain_points: [] ✅
Reasoning: No problems mentioned, empty array is correct
```

---

### Examples: INVALID Classifications (Hallucination) ❌

**Example 1: Inferring from topic alone**
```
Transcript: "I take supplements every morning"
→ pain_points: ["brain_fog", "fatigue"] ❌ WRONG!
Reasoning: Brain fog and fatigue are NOT mentioned in transcript
Should be: pain_points: [] ✅
```

**Example 2: Inventing categories not in taxonomy**
```
Transcript: "Best green smoothie recipe"
→ content_category: "smoothie_recipe" ❌ WRONG!
Reasoning: "smoothie_recipe" doesn't exist in taxonomy
Should use: closest match from actual taxonomy (e.g., "recipe_tutorial")
```

**Example 3: Using caption/hashtags for content fields**
```
Transcript: "I love wellness routines" (no mention of matcha)
Caption: "#matcha #wellness" (you'll see this in Zone 2)
→ keywords: ["matcha"] ❌ WRONG!
Reasoning: Not mentioned in transcript
Should be: keywords: [] ✅
```

---

## ZONE 2: CAPTION ANALYSIS (Caption-Based)

### Caption & Hashtags Data

You are NOW seeing the caption and hashtags for the first time. Use these ONLY for caption analysis below. Do NOT go back and modify Zone 1 classifications.

**Caption**:
```
{caption if caption else "(No caption available)"}
```

**Hashtags**:
```
{json.dumps(hashtags) if hashtags else "[]"}
```

---

### Caption Analysis Instructions

Analyze the caption structure. These fields are INDEPENDENT of the transcript taxonomy (Zone 1).

**CRITICAL WARNING**: Use the exact string values specified below. Do NOT use variations. Do NOT use values from the transcript taxonomy.

---

### Caption Hook Type (select ONE)

Copy EXACTLY one of these strings:
- `"statement"`
- `"question"`
- `"command"`
- `"teaser"`

**DO NOT USE**: "declarative", "declarative_statement", "interrogative", or any other variation
**DO NOT USE**: Values from transcript hook_strategies taxonomy

**Examples**:
- Caption: "This changed my life" → hook_type: `"statement"` ✅
- Caption: "Did you know this?" → hook_type: `"question"` ✅
- Caption: "Try this now" → hook_type: `"command"` ✅

---

### Call-to-Action Type (select ONE)

Copy EXACTLY one of these strings:
- `"link_in_bio"`
- `"save_post"`
- `"comment"`
- `"follow"`
- `"share"`
- `"tag_friend"`
- `"none"`

---

## ZONE 3: OUTPUT FORMAT

**M10 FIX: hashtag_count removed from LLM output - Python calculates it**

Return valid JSON with 12 fields below (Python will add hashtag_count). Use lowercase `true`/`false` for booleans.

```json
{{
  "video_id": "{video_id}",
  "taxonomy_version": "stage2.6_output",
  "content_category": "...",
  "hook_strategy": "...",
  "closing_strategy": "...",
  "pain_points": [...],
  "keywords": [...],
  "engagement_drivers": [...],
  "content_tactics": [...],
  "caption_analysis": {{
    "hook_type": "...",
    "cta_type": "..."
  }},
  "confidence": "high",
  "transcript_available": true,
  "note": null
}}
```

### Confidence Assessment

**"high"**:
- Video clearly matches selected taxonomy categories
- Strong transcript evidence for all selections
- Confident in classifications

**"medium"**:
- Partial matches OR some inference required
- Weaker evidence but reasonable classification
- Some uncertainty

**"low"**:
- Limited evidence OR forced match
- Uncertain classifications
- Difficult to classify

### Note Field

- Use `null` for normal classifications
- Use string for issues: `"Transcript quality poor but analyzable"`

---

## FINAL CHECKLIST

Before submitting your response, verify:

1. ✅ Zone 1: Used ONLY transcript for ALL 7 content classification fields
2. ✅ All taxonomy category names copied EXACTLY (character-for-character)
3. ✅ Empty arrays [] used when no match (not guessing)
4. ✅ Zone 2: Caption fields use exact values: `"statement"` NOT `"declarative"`
5. ✅ Output is valid JSON with all 13 fields
6. ✅ No hallucinated classifications (check examples above)
7. ✅ No invented categories not in taxonomy
8. ✅ Did NOT use caption/hashtags for Zone 1 content fields
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        timeout=30,
        system=system_message,
        messages=[{"role": "user", "content": prompt}]
    )

    return extract_json(response.content[0].text, video_id)


def classify_caption_only(
    video_id: str,
    caption: str,
    hashtags: List[str],
    client: Any
) -> Dict[str, Any]:
    """
    Flow 2: Caption analysis only (no valid transcript).
    
    LLM outputs ONLY caption_analysis (2 fields).
    Python constructs full 15-field schema with defaults.
    
    Source: ContentAnalysispt2.md Step 4 (Lines 1015-1107)
    
    Args:
        video_id: Video identifier
        caption: Caption text
        hashtags: List of hashtags
        client: Anthropic API client
        
    Returns:
        Dict with caption_analysis only (Python will fill rest)
    """
    import json
    
    system_message = """You are analyzing TikTok video captions and hashtags. The video has NO valid transcript (music/noise only). Return ONLY caption analysis fields."""

    prompt = f"""# CAPTION ANALYSIS TASK

Video has NO valid transcript. Analyze caption and hashtags ONLY.

**Video ID**: {video_id}

---

## VIDEO DATA

**Caption**:
```
{caption or "(No caption available)"}
```

**Hashtags**:
```
{json.dumps(hashtags) if hashtags else "[]"}
```

---

## INSTRUCTIONS

Analyze the caption structure using the exact values specified below.

### Caption Hook Type (select ONE)

Copy EXACTLY one of these strings:
- `"statement"`
- `"question"`
- `"command"`
- `"teaser"`

**DO NOT USE**: "declarative", "declarative_statement", "interrogative", or any variation

### Call-to-Action Type (select ONE)

Copy EXACTLY one of these strings:
- `"link_in_bio"`
- `"save_post"`
- `"comment"`
- `"follow"`
- `"share"`
- `"tag_friend"`
- `"none"`

---

## OUTPUT FORMAT

**M10 FIX: hashtag_count removed - Python calculates it deterministically**

Return ONLY these 2 caption fields (Python will construct full schema including hashtag_count):

```json
{{
  "caption_analysis": {{
    "hook_type": "...",
    "cta_type": "..."
  }}
}}
```

**Note**: Python will add hashtag_count and all other 13 fields to create the full 15-field output.
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=128,
        timeout=15,
        system=system_message,
        messages=[{"role": "user", "content": prompt}]
    )

    return extract_json(response.content[0].text, video_id)
