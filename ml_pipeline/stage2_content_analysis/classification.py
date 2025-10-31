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

from .utils import load_json, save_json, construct_path
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

    # Step 2: Find JSON boundaries
    first_brace = text.find('{')
    last_brace = text.rfind('}')

    if first_brace == -1 or last_brace == -1:
        # Log the failure case for debugging
        logger.error(
            f"No JSON braces found in response for {video_id}. "
            f"Response preview: {original_text[:200]}..."
        )
        raise ValueError("No valid JSON braces found in response")

    if last_brace <= first_brace:
        logger.error(
            f"Invalid brace order for {video_id}: first={first_brace}, last={last_brace}"
        )
        raise ValueError("Invalid JSON brace positions")

    # Step 3: Extract JSON block
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
        # Log more context for debugging
        logger.error(
            f"Extracted text is not valid JSON for {video_id}: {str(e)}. "
            f"Extracted: {json_text[:300]}..."
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


def load_video_data(video_id: str) -> tuple:
    """
    Load transcript, caption, and hashtags for a video.

    Returns:
        tuple: (transcript_dict, caption_str, hashtags_list)
    """
    RUMIAI_ROOT = os.environ.get('RUMIAI_ROOT', '/home/jorge/rumiaifinal')
    transcript_path = f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json"
    caption_path = f"{RUMIAI_ROOT}/video_captions/{video_id}_caption.json"
    hashtags_path = f"{RUMIAI_ROOT}/video_hashtags/{video_id}_hashtags.json"

    # Load transcript
    try:
        transcript_data = load_json(transcript_path)
        transcript = {
            'text': transcript_data.get('text', ''),
            'available': True
        }
    except FileNotFoundError:
        transcript = {'text': '', 'available': False}
        logger.warning(f"No transcript for {video_id}")

    # Load caption
    try:
        caption_data = load_json(caption_path)
        caption = caption_data.get('text', '')
    except FileNotFoundError:
        caption = ''

    # Load hashtags
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
    output_dir: str
) -> Dict[str, Any]:
    """
    Helper function: Load video data, classify, and save result.

    Source: ContentAnalysisCHILDTI.md Section 4.4.2

    Args:
        video_id: Video identifier
        taxonomy: Curated taxonomy
        client: Anthropic API client
        output_dir: Directory to save classification result

    Returns:
        dict: Classification result

    Raises:
        Exception: Any error during classification
    """
    # Load video data
    transcript, caption, hashtags = load_video_data(video_id)

    # Validate business rules
    validate_business_rules_classification(video_id, transcript, caption, hashtags)

    # Classify video
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
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify videos sequentially with checkpoint/resume support.

    Source: ContentAnalysisCHILDTI.md Section 4.4.1, 4.5.4

    Args:
        videos: List of video IDs to classify
        taxonomy: Curated taxonomy
        client: Anthropic API client
        output_dir: Directory to save results
        checkpoint_path: Optional path to checkpoint file

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
            # Classify and save
            classify_single_video_with_save(video_id, taxonomy, client, output_dir)

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
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify videos in parallel with checkpoint/resume support.

    Source: ContentAnalysisCHILDTI.md Section 4.4.2, 4.5.5

    Args:
        videos: List of video IDs
        taxonomy: Curated taxonomy
        client: Anthropic API client
        output_dir: Directory to save results
        max_workers: Concurrent workers (default: 5)
        checkpoint_path: Optional checkpoint file path

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
        # Submit all classification tasks
        future_to_video = {
            executor.submit(
                classify_single_video_with_save,
                video_id, taxonomy, client, output_dir
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
    checkpoint_enabled: bool = True
) -> Dict[str, Any]:
    """
    Classify all videos with configurable parallel/sequential execution and checkpoints.

    Source: ContentAnalysisCHILDTI.md Section 4.4

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

    # Execute classification
    start_time = time.time()

    if parallel:
        logger.info(f"🚀 Starting parallel classification: {len(all_videos)} videos, {max_workers} workers")
        results = classify_all_videos_parallel(
            all_videos, taxonomy, anthropic_client, output_dir, max_workers, checkpoint_path
        )
    else:
        logger.info(f"📋 Starting sequential classification: {len(all_videos)} videos")
        results = classify_all_videos_sequential(
            all_videos, taxonomy, anthropic_client, output_dir, checkpoint_path
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
    logger.info("Step 2/3: Loading taxonomy and manifest...")
    taxonomy = load_json(taxonomy_path)
    manifest = load_json(manifest_path)
    logger.info("✓ Files loaded")

    # Step 5: Initialize Anthropic client
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Step 6: Classify all videos
    logger.info("Step 3/3: Classifying all videos (Claude Haiku)...")
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
        checkpoint_enabled=checkpoint_enabled
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
