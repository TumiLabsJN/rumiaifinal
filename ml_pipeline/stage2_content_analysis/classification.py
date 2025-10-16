"""
Stage 2.7: Video Classification
Classify all videos using curated taxonomy + LLM

Source: ContentAnalysisCHILDTI.md Section 4 (Function 4.3)
"""

import os
import json
import logging
import time
from typing import Dict, List, Any
import anthropic

from .utils import load_json, save_json, construct_path
from .validation import (
    validate_classification_inputs,
    validate_business_rules_classification,
    validate_classification_output
)
from .error_handlers import handle_graceful_skip

logger = logging.getLogger(__name__)


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

**Category 3: Audience Pain Points** (Multiple Selection)
{json.dumps(taxonomy['audience_pain_points'], indent=2)}

**Category 4: Trending Keywords** (Multiple Selection)
{json.dumps(taxonomy['trending_keywords'], indent=2)}

**Category 5: Engagement Drivers** (Multiple Selection)
{json.dumps(taxonomy['engagement_drivers'], indent=2)}

**Category 6: Content Tactics** (Multiple Selection)
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

**Note**: Do not attempt to categorize hashtags as broad/niche/branded - this requires view count data not available.

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

    # Step 2: Call API with retry logic (3 attempts with exponential backoff)
    for attempt in range(3):
        try:
            # Step 2.1: Make API call with Haiku model
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                timeout=30,  # 30 seconds per video
                system=system_message,
                messages=[{"role": "user", "content": prompt}]
            )

            # Step 2.2: Parse response
            classification = json.loads(response.content[0].text)

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


def classify_all_videos(
    client_id: str,
    hashtag: str,
    taxonomy: Dict[str, Any],
    manifest: Dict[str, Any],
    anthropic_client: anthropic.Anthropic,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive"
) -> Dict[str, int]:
    """
    Classify all videos in top 3 buckets using taxonomy.

    Source: ContentAnalysisCHILDTI.md Section 4.3 (orchestration function)

    Args:
        client_id: Client identifier
        hashtag: Hashtag name
        taxonomy: Curated taxonomy from Stage 2.6
        manifest: Selection manifest from Stage 2.5
        anthropic_client: Initialized Anthropic API client
        analysis_mode: "top" or "recent"
        selection_strategy: "contrastive" or "top"

    Returns:
        dict: Statistics {
            'total_videos': int,
            'successful': int,
            'skipped': int,
            'failed': int
        }
    """
    stats = {'total_videos': 0, 'successful': 0, 'skipped': 0, 'failed': 0}

    # Process each bucket
    for bucket in manifest['selected_buckets']:
        logger.info(f"Processing bucket {bucket}...")

        bucket_videos = manifest['videos_by_bucket'][bucket]
        all_videos = bucket_videos['top_performers'] + bucket_videos.get('bottom_performers', [])

        stats['total_videos'] += len(all_videos)

        # Classify each video
        for video_id in all_videos:
            try:
                # Load video data
                transcript_path = f"/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json"
                caption_path = f"/home/jorge/rumiaifinal/video_captions/{video_id}_caption.json"
                hashtags_path = f"/home/jorge/rumiaifinal/video_hashtags/{video_id}_hashtags.json"

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

                # Validate business rules
                validate_business_rules_classification(video_id, transcript, caption, hashtags)

                # Classify video
                classification = classify_video_llm(
                    video_id=video_id,
                    transcript=transcript,
                    caption=caption,
                    hashtags=hashtags,
                    taxonomy=taxonomy,
                    client=anthropic_client
                )

                # Save classification
                output_dir = construct_path(
                    client_id=client_id,
                    hashtag=hashtag,
                    analysis_mode=analysis_mode,
                    selection_strategy=selection_strategy,
                    bucket=bucket,
                    file_type="bucket_content_analysis"
                )
                output_path = f"{output_dir}/{video_id}_content.json"
                save_json(output_path, classification)

                stats['successful'] += 1
                logger.debug(f"✓ Classified {video_id}")

            except Exception as e:
                logger.error(f"✗ Failed to classify {video_id}: {e}")
                stats['failed'] += 1

    return stats


def run_classification_stage(
    client_id: str,
    hashtag: str,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive"
) -> Dict[str, int]:
    """
    Run complete Stage 2.7 classification pipeline with error handling.

    Source: ContentAnalysisCHILDTI.md Section 6.3

    Args:
        client_id: Client identifier (e.g., "acme_corp")
        hashtag: Hashtag name (e.g., "nutrition")
        analysis_mode: "top" or "recent" (default: "top")
        selection_strategy: "contrastive" or "top" (default: "contrastive")

    Returns:
        dict: Statistics from classification

    Raises:
        FileNotFoundError: If taxonomy or manifest missing
        ValueError: If validation fails
    """
    logger.info(f"=" * 80)
    logger.info(f"STAGE 2.7: VIDEO CLASSIFICATION")
    logger.info(f"Client: {client_id}, Hashtag: #{hashtag}")
    logger.info(f"=" * 80)

    # Step 1: Construct paths
    taxonomy_path = construct_path(
        client_id=client_id,
        hashtag=hashtag,
        analysis_mode=analysis_mode,
        selection_strategy=selection_strategy,
        file_type="taxonomy"
    )
    manifest_path = construct_path(
        client_id=client_id,
        hashtag=hashtag,
        analysis_mode=analysis_mode,
        selection_strategy=selection_strategy,
        file_type="selection_manifest"
    )

    # Step 2: Validate inputs
    logger.info("Step 1/3: Validating inputs...")
    validate_classification_inputs(taxonomy_path, manifest_path)
    logger.info("✓ Input validation passed")

    # Step 3: Load taxonomy and manifest
    logger.info("Step 2/3: Loading taxonomy and manifest...")
    taxonomy = load_json(taxonomy_path)
    manifest = load_json(manifest_path)
    logger.info("✓ Files loaded")

    # Step 4: Initialize Anthropic client
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Step 5: Classify all videos
    logger.info("Step 3/3: Classifying all videos (Claude Haiku)...")
    stats = classify_all_videos(
        client_id=client_id,
        hashtag=hashtag,
        taxonomy=taxonomy,
        manifest=manifest,
        anthropic_client=anthropic_client,
        analysis_mode=analysis_mode,
        selection_strategy=selection_strategy
    )

    logger.info(f"=" * 80)
    logger.info(f"STAGE 2.7 COMPLETE")
    logger.info(f"Total videos: {stats['total_videos']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"=" * 80)

    return stats
