"""
Stage 2.6: Pattern Discovery
Discover content patterns from sample transcripts using LLM

Source: ContentAnalysisCHILDTI.md Section 4 (Functions 4.1, 4.2, 4.2.5)
"""

import os
import json
import random
import logging
import time
from datetime import datetime
from typing import List, Dict, Any
import anthropic

from .utils import load_json, save_json, construct_path
from .validation import validate_discovery_inputs, validate_business_rules_sampling, validate_discovery_output
from .error_handlers import handle_graceful_skip
from .cost_tracking import log_estimated_cost, log_actual_cost
from .transcript_validation import load_validation_cache

logger = logging.getLogger(__name__)


def sample_transcripts_for_discovery(
    manifest_path: str,
    sample_size: int = 50,
    validation_cache: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Sample transcripts stratified evenly across top 3 buckets.

    Source: ContentAnalysisCHILDTI.md Section 4.1
    Updated: 2.6FilterImprovement.md Section "Implementation Part 4"

    Args:
        manifest_path: Path to selection_manifest.json from Stage 2.5
        sample_size: Total transcripts to sample (default: 50, max: 100)
                    If fewer videos available, uses all available (does not fail)
        validation_cache: Optional validation cache dict from Stage 2.5.5
                         If provided, only samples valid transcripts
                         If None, samples all transcripts (legacy behavior)

    Returns:
        list[dict]: Sampled video IDs with transcript text and bucket assignment
                    Format: [{"video_id": str, "text": str, "bucket": str}, ...]

    Raises:
        FileNotFoundError: If manifest_path does not exist
        ValueError: If manifest missing required fields or insufficient samples
    """
    # Step 1: Load manifest from Stage 2.5
    # Source: ContentAnalysisCHILD.md Section 2.3.1 lines 131-133
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)

    # Step 2: Validate manifest structure
    # Source: ContentAnalysisCHILD.md Section 6.1 (Input Validation)
    required_fields = ['hashtag', 'selected_buckets', 'videos_by_bucket']
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")

    # Step 3: Extract top 3 buckets
    # Source: ContentAnalysisCHILD.md Section 2.3.1 line 133
    top_3_buckets = manifest['selected_buckets']  # e.g., ["33-60s", "60-90s", "90-120s"]

    # Step 4: Cap sample_size at 100 (configurable maximum)
    sample_size = min(sample_size, 100)

    # Step 5: Calculate total available top performers across all buckets
    total_available = 0
    for bucket in top_3_buckets:
        if bucket in manifest['videos_by_bucket']:
            total_available += len(manifest['videos_by_bucket'][bucket]['top_performers'])

    # Step 6: Adjust sample_size if fewer videos available (don't fail)
    actual_sample_size = min(sample_size, total_available)

    if actual_sample_size < sample_size:
        logger.warning(
            f"Requested {sample_size} samples but only {total_available} top performers available. "
            f"Using all {actual_sample_size} available videos."
        )

    logger.info(f"Sampling {actual_sample_size} transcripts from top 3 buckets: {top_3_buckets}")

    # Step 7: Calculate samples per bucket (stratified even sampling)
    # Source: ContentAnalysisCHILD.md Section 2.3.1 line 136
    samples_per_bucket = actual_sample_size // 3  # ~17 per bucket for sample_size=50

    # Step 8: Initialize results container
    sampled_transcripts = []

    # Step 9: Sample from each bucket
    for bucket in top_3_buckets:
        # Step 9.1: Validate bucket exists in manifest
        if bucket not in manifest['videos_by_bucket']:
            logger.warning(f"Bucket {bucket} not in videos_by_bucket, skipping")
            continue

        # Step 9.2: Extract top performers only
        # Source: ContentAnalysisCHILD.md Section 2.3.1 line 141
        top_performers = manifest['videos_by_bucket'][bucket]['top_performers']

        # Step 9.2a: FILTER to valid video IDs if validation cache provided
        # Source: 2.6FilterImprovement.md Section "Implementation Part 4"
        if validation_cache is not None:
            valid_video_ids = [
                vid for vid in top_performers
                if vid in validation_cache and validation_cache[vid]['is_valid']
            ]

            invalid_rate = (len(top_performers) - len(valid_video_ids)) / len(top_performers) if len(top_performers) > 0 else 0
            logger.info(
                f"Bucket {bucket}: {len(valid_video_ids)}/{len(top_performers)} valid transcripts "
                f"({invalid_rate:.1%} invalid rate)"
            )

            # Use filtered valid IDs for sampling
            top_performers = valid_video_ids

        # Step 9.3: Random sample (handle case where bucket has < samples_per_bucket videos)
        # Source: ContentAnalysisCHILD.md Section 2.3.1 lines 144
        # Use all available if bucket has fewer than target
        sample_count = min(samples_per_bucket, len(top_performers))
        sampled_ids = random.sample(top_performers, sample_count) if len(top_performers) > 0 else []

        logger.info(f"Bucket {bucket}: Sampling {sample_count} videos from {len(top_performers)} available")

        # Step 6.4: Load transcripts for sampled videos
        for video_id in sampled_ids:
            # Step 6.4.1: Construct transcript path
            # Source: ContentAnalysisCHILD.md Section 2.3.1 line 148
            # Updated: M1 Enhancement - Use RUMIAI_ROOT environment variable
            RUMIAI_ROOT = os.environ.get('RUMIAI_ROOT', '/home/jorge/rumiaifinal')
            transcript_path = f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json"

            # Step 6.4.2: Load transcript (handle missing files gracefully)
            # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases table
            try:
                transcript_data = load_json(transcript_path)
                text = transcript_data.get('text', '')

                # Step 6.4.3: Include even if empty (may reveal no-speech patterns)
                # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases row 3
                sampled_transcripts.append({
                    "video_id": video_id,
                    "text": text,
                    "bucket": bucket
                })
            except FileNotFoundError:
                # Step 6.4.4: Log warning and skip video
                # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases row 2
                logger.warning(f"Transcript not found: {video_id}, skipping")
                continue

    # Step 7: Validate we have sufficient samples
    if len(sampled_transcripts) < 10:
        raise ValueError(
            f"Insufficient transcripts sampled: {len(sampled_transcripts)}. "
            f"Minimum 10 required for pattern discovery."
        )

    logger.info(f"Successfully sampled {len(sampled_transcripts)} transcripts")

    # Step 8: Run business rules validation
    validate_business_rules_sampling(sampled_transcripts)

    # Step 9: Return sampled transcripts
    return sampled_transcripts


def discover_patterns_llm(
    transcripts: List[Dict[str, Any]],
    hashtag: str,
    client_id: str,
    analysis_type: str,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive"
) -> Dict[str, Any]:
    """
    Discover content patterns using LLM (Claude 3.5 Sonnet).

    Source: ContentAnalysisCHILDTI.md Section 4.2

    Args:
        transcripts: List of transcript dicts with video_id, text, bucket
        hashtag: str, target name (e.g., "nutrition" or "cindyprado28")
        client_id: str, client identifier (e.g., "acme_corp")
        analysis_type: str, "hashtag", "competitor", or "creator"
        analysis_mode: str, "top" or "recent" (default: "top")
        selection_strategy: str, "contrastive" or "top" (default: "contrastive")

    Returns:
        dict: Raw discovery JSON with patterns, frequencies, examples

    Raises:
        TimeoutError: If LLM exceeds 120s timeout after 3 retries
        ValueError: If LLM returns invalid JSON after 3 retries
    """
    # Step 1: Prepare prompt with taxonomy discovery instructions
    # Source: 2.6HashtagCritique.md - Final Prompt (After All Decisions)
    # Note: System message configured separately in API call
    system_message = """You are an expert content analyst specializing in short-form video patterns. Identify recurring patterns in the transcripts based on frequency and evidence. Be objective and data-driven: report patterns that actually appear in the data, not prescriptive advice. Patterns should be actionable for content creators but grounded in observed behavior."""

    prompt = f"""Analyze the following {len(transcripts)} video transcripts from the #{hashtag} hashtag.

Your task is to identify recurring content patterns across 6 categories. Focus on patterns that appear in AT LEAST 5% of videos (minimum 2 videos). Do not create patterns for isolated or single-video elements.

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
    "audience_pain_points": ["chronic bloating", "low energy"],
    "trending_keywords": ["protein intake", "gut health"],
    "engagement_drivers": ["before after reveal"],
    "content_tactics": ["direct to camera", "voiceover"]
  }}
}}

Requirements:
- Categories 1-2: Provide 2-3 examples and 2-3 representative_video_ids per pattern
- DO NOT include a "percentage" field - this will be calculated automatically by Python post-processing
- Categories 3-6: Simple string lists only (no objects, no extra fields)

---

## FINAL INSTRUCTIONS

1. Analyze ALL {len(transcripts)} transcripts carefully - every pattern must be grounded in observed data
2. Return only patterns you genuinely observe - if a category has fewer patterns, that's acceptable
3. Use descriptive snake_case names (short but clear, 2-4 words)
4. Return valid JSON only - no commentary, explanations, or additional text outside the JSON structure

DO NOT make up patterns to fill categories. Quality over quantity.

---

TRANSCRIPTS:

{json.dumps([{'video_id': t['video_id'], 'text': t['text']} for t in transcripts], indent=2)}
"""

    # Step 2: Initialize Anthropic client
    # Source: ContentAnalysisCHILD.md Section 2.3.2 line 224
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Step 2a: Log estimated cost
    # Source: ContentAnalysisCHILDTI.md Section 4.6.3
    log_estimated_cost("discovery", sample_size=len(transcripts))

    # Step 3: Call API with retry logic (3 attempts with exponential backoff)
    # Source: ContentAnalysisCHILD.md Section 2.3.2 Edge Cases table
    for attempt in range(3):
        try:
            logger.info(f"Calling Claude Sonnet API for discovery (attempt {attempt + 1}/3)...")

            # Step 3.1: Make API call with Sonnet model
            # Source: 2.6HashtagCritique.md - Final Prompt (System + User messages)
            # Updated: M8 Enhancement - Add latency tracking
            # Updated: 2025-01-17 - Upgraded to Sonnet 4.5 for better pattern discovery
            start_time = time.time()
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                timeout=120,  # 2 minutes
                system=system_message,
                messages=[{"role": "user", "content": prompt}]
            )

            # Step 3.1a: Log actual cost with latency
            # Source: ContentAnalysisCHILDTI.md Section 4.6.3
            log_actual_cost(response, "sonnet", start_time)

            # Step 3.2: Extract response text
            response_text = response.content[0].text

            logger.info(f"Received response from Claude Sonnet ({len(response_text)} chars)")

            # Step 3.3: Parse JSON response (handle markdown code fences)
            # Source: ContentAnalysisCHILD.md Section 2.3.2 line 234
            # Updated: 2025-10-25 - Use parse_llm_json() to handle markdown wrapping
            from ml_pipeline.stage2_content_analysis.utils import parse_llm_json
            raw_taxonomy = parse_llm_json(response_text)

            # Step 3.4: Validate response structure before returning
            # Source: ContentAnalysisCHILD.md Section 6.3 (Output Validation)
            validate_discovery_output(raw_taxonomy)

            # Step 3.5: Calculate percentages for patterns
            raw_taxonomy = calculate_percentages(raw_taxonomy, len(transcripts))

            # Step 3.6: Save raw discovery to file using PathBuilder
            # Source: ContentAnalysisCHILD.md Section 2.3.2 lines 237-238
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
            output_path = str(target_dir / "content_taxonomies" / f"{target_sanitized}_raw_discovery.json")
            save_json(output_path, raw_taxonomy)

            # Step 3.7: Log success and manual curation instructions
            # Source: ContentAnalysisCHILD.md Section 2.3.2 lines 240-242
            logger.info(f"✅ Discovery complete: {output_path}")
            logger.info(f"📝 Next: Manually curate and save to {hashtag}_taxonomy.json")

            # Step 3.8: Return successful result
            return raw_taxonomy

        except TimeoutError as e:
            # Step 3.9: Handle timeout (retry with backoff)
            # Source: ContentAnalysisCHILD.md Section 2.3.2 Edge Cases row 1
            if attempt < 2:
                delay = [1, 2, 4][attempt]  # Exponential backoff
                logger.warning(f"⏰ Discovery timeout (>120s). Retry {attempt+1}/3 in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"❌ Discovery failed after 3 retries. Check status.anthropic.com")
                raise

        except json.JSONDecodeError as e:
            # Step 3.10: Handle invalid JSON (retry)
            # Source: ContentAnalysisCHILD.md Section 2.3.2 Edge Cases row 2
            if attempt < 2:
                delay = [1, 2, 4][attempt]
                logger.warning(f"⚠️ LLM returned invalid JSON. Retry {attempt+1}/3 in {delay}s...")
                logger.debug(f"Response text: {response_text[:500]}")
                time.sleep(delay)
            else:
                logger.error(f"❌ Invalid JSON after 3 retries: {str(e)}")
                logger.error(f"Response text: {response_text[:1000]}")
                raise ValueError(f"LLM returned invalid JSON after 3 retries: {str(e)}")

    # Unreachable (for type checker)
    raise RuntimeError("Unexpected retry loop exit")


def calculate_percentages(raw_taxonomy: Dict[str, Any], sample_size: int) -> Dict[str, Any]:
    """
    Add percentage field to discovery patterns post-LLM.

    Source: ContentAnalysisCHILDTI.md Section 4.2.5

    Args:
        raw_taxonomy: Raw discovery JSON from LLM (without percentage fields)
        sample_size: Number of transcripts analyzed (e.g., 50)

    Returns:
        dict: Discovery JSON with percentage fields added

    Raises:
        ValueError: If frequency > sample_size (data integrity check)
    """
    # Step 1: Process Categories 1-2 (content_categories, hook_strategies)
    # These have frequency field, simple lists don't
    for category in ['content_categories', 'hook_strategies']:
        for pattern in raw_taxonomy['discovered_patterns'][category]:
            frequency = pattern['frequency']

            # Step 1.1: Validate frequency doesn't exceed sample size
            if frequency > sample_size:
                raise ValueError(
                    f"Pattern '{pattern['name']}' frequency ({frequency}) "
                    f"exceeds sample size ({sample_size}). LLM hallucination detected."
                )

            # Step 1.2: Calculate percentage (round to 1 decimal place)
            pattern['percentage'] = round((frequency / sample_size) * 100, 1)

    # Step 2: Return enriched taxonomy
    return raw_taxonomy


def run_discovery_stage(
    client_id: str,
    hashtag: str,
    analysis_type: str,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive",
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Run complete Stage 2.6 discovery pipeline with error handling.

    Source: ContentAnalysisCHILDTI.md Section 6.3

    Args:
        client_id: Client identifier (e.g., "acme_corp")
        hashtag: Target name (e.g., "nutrition" or "cindyprado28")
        analysis_type: "hashtag", "competitor", or "creator"
        analysis_mode: "top" or "recent" (default: "top")
        selection_strategy: "contrastive" or "top" (default: "contrastive")
        sample_size: Number of transcripts to sample (default: 50)

    Returns:
        dict: Raw discovery JSON with patterns

    Raises:
        FileNotFoundError: If manifest or transcripts missing
        ValueError: If validation fails or insufficient data
        TimeoutError: If LLM API timeout after retries
    """
    logger.info(f"=" * 80)
    logger.info(f"STAGE 2.6: CONTENT PATTERN DISCOVERY")
    logger.info(f"Client: {client_id}, Target: {hashtag}, Type: {analysis_type}, Sample Size: {sample_size}")
    logger.info(f"=" * 80)

    # Step 1: Construct manifest path using PathBuilder
    from foundation.paths import PathBuilder
    path_builder = PathBuilder()
    target_dir = path_builder.get_target_dir(
        client_id=client_id,
        analysis_type=analysis_type,
        target=hashtag,  # Will be sanitized by PathBuilder
        analysis_mode=analysis_mode,
        selection_strategy=selection_strategy
    )
    manifest_path = str(target_dir / "selection_manifest.json")

    # Step 2: Validate inputs
    logger.info("Step 1/4: Validating inputs...")
    validate_discovery_inputs(manifest_path, sample_size)
    logger.info("✓ Input validation passed")

    # Step 2a: Load validation cache if available (Stage 2.5.5)
    # Source: 2.6FilterImprovement.md Section "Implementation Part 4"
    logger.info("Step 2/4: Loading transcript validation cache...")
    try:
        validation_cache = load_validation_cache(
            client_id, hashtag, analysis_type, analysis_mode, selection_strategy
        )
        logger.info("✓ Validation cache loaded - will filter invalid transcripts")
    except FileNotFoundError:
        logger.warning(
            "⚠️  No validation cache found. Sampling without filtering.\n"
            "   Run Stage 2.5.5 (validate_all_transcripts) first for better results."
        )
        validation_cache = None

    # Step 3: Sample transcripts
    logger.info("Step 3/4: Sampling transcripts from top 3 buckets...")
    sampled_transcripts = sample_transcripts_for_discovery(
        manifest_path, sample_size, validation_cache
    )
    logger.info(f"✓ Sampled {len(sampled_transcripts)} transcripts")

    # Step 4: Run LLM discovery
    logger.info("Step 4/4: Running LLM pattern discovery (Claude Sonnet)...")
    raw_taxonomy = discover_patterns_llm(
        sampled_transcripts,
        hashtag,
        client_id,
        analysis_type,
        analysis_mode,
        selection_strategy
    )
    logger.info("✓ Discovery complete")

    logger.info(f"=" * 80)
    logger.info(f"STAGE 2.6 COMPLETE")
    logger.info(f"Discovered {len(raw_taxonomy['discovered_patterns']['content_categories'])} content categories")
    logger.info(f"Discovered {len(raw_taxonomy['discovered_patterns']['hook_strategies'])} hook strategies")
    logger.info(f"=" * 80)

    return raw_taxonomy
