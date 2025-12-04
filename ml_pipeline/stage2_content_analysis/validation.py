"""
Validation Functions for Content Analysis Stage (2.6 & 2.7)

Source: ContentAnalysisCHILDTI.md Section 5
"""

import os
import logging
from typing import Dict, List, Any
from .utils import load_json

logger = logging.getLogger(__name__)


# ===== STAGE 2.6 DISCOVERY INPUT VALIDATION =====
# Source: ContentAnalysisCHILD.md Section 6.1 (lines 784-835)

def validate_discovery_inputs(manifest_path: str, sample_size: int):
    """
    Validate inputs before discovery.

    Source: ContentAnalysisCHILDTI.md Section 5.1

    Args:
        manifest_path: Path to selection_manifest.json from Stage 2.5
        sample_size: Number of transcripts to sample (typically 50)

    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest structure invalid or parameters out of range
    """
    # Validation 1: Check manifest exists
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 790-795
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"selection_manifest.json not found at {manifest_path}. "
            "Did Stage 2.5 complete successfully?"
        )

    # Validation 2: Load and validate manifest structure
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 797-801
    manifest = load_json(manifest_path)
    required_fields = ['hashtag', 'selected_buckets', 'videos_by_bucket']
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")

    # Validation 3: Check we have at least 1 bucket (allow 1-3 for small datasets)
    # Modified: Original required exactly 3, relaxed for small competitor accounts
    # See: /home/jorge/rumiaifinal/UpgradeBucketz.md for full rationale
    if len(manifest['selected_buckets']) < 1:
        raise ValueError(
            f"Expected at least 1 selected bucket, found {len(manifest['selected_buckets'])}. "
            "Stage 2.5 may have failed."
        )
    if len(manifest['selected_buckets']) < 3:
        logger.warning(
            f"Only {len(manifest['selected_buckets'])} bucket(s) selected (typically 3). "
            f"Small dataset - proceeding with limited buckets."
        )

    # Validation 4: Check each bucket has videos
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 810-821
    for bucket in manifest['selected_buckets']:
        if bucket not in manifest['videos_by_bucket']:
            raise ValueError(f"Bucket {bucket} missing from videos_by_bucket")

        top_performers = manifest['videos_by_bucket'][bucket].get('top_performers', [])
        if len(top_performers) < 1:
            raise ValueError(
                f"Bucket {bucket} has no top performers. "
                f"Cannot proceed with empty bucket."
            )
        if len(top_performers) < 10:
            logger.warning(
                f"⚠️  Bucket {bucket} has only {len(top_performers)} top performers "
                f"(recommended: 10+). Proceeding with limited sample - taxonomy quality may be reduced."
            )

    # Validation 5: Check sample size is reasonable
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 823-827
    if sample_size < 10:
        raise ValueError(f"Sample size too small: {sample_size}. Minimum is 10.")
    if sample_size > 200:
        logger.warning(f"Sample size very large: {sample_size}. May exceed LLM token limits.")

    # Validation 6: Check ANTHROPIC_API_KEY set
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 829-834
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set with: export ANTHROPIC_API_KEY=sk-ant-..."
        )


# ===== STAGE 2.7 CLASSIFICATION INPUT VALIDATION =====
# Source: ContentAnalysisCHILD.md Section 6.1 (lines 838-893)

def validate_classification_inputs(taxonomy_path: str, manifest_path: str):
    """
    Validate inputs before classification.

    Source: ContentAnalysisCHILDTI.md Section 5.1

    Args:
        taxonomy_path: Path to curated taxonomy JSON
        manifest_path: Path to selection_manifest.json

    Raises:
        FileNotFoundError: If taxonomy or manifest doesn't exist
        ValueError: If taxonomy structure invalid
    """
    # Validation 1: Check taxonomy exists
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 844-849
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(
            f"Curated taxonomy not found at {taxonomy_path}. "
            "Run Stage 2.6 discovery and complete manual curation first."
        )

    # Validation 2: Load and validate taxonomy structure
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 851-866
    taxonomy = load_json(taxonomy_path)

    # Check all required fields present
    required_fields = [
        'content_categories', 'hook_strategies', 'closing_strategies',
        'audience_pain_points', 'trending_keywords', 'engagement_drivers', 'content_tactics'
    ]
    missing = [f for f in required_fields if f not in taxonomy]
    if missing:
        raise ValueError(f"Taxonomy missing required fields: {missing}")

    # Check all fields non-empty
    for field in required_fields:
        if not taxonomy[field]:
            raise ValueError(f"Taxonomy field '{field}' is empty")

    # Validation 3: Check semantic categories have definitions
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 868-885
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

    for strategy in taxonomy['closing_strategies']:
        if 'name' not in strategy or 'definition' not in strategy:
            raise ValueError(f"closing_strategies missing name or definition: {strategy}")
        if len(strategy['definition']) < 10:
            raise ValueError(
                f"Definition too short for '{strategy['name']}': "
                f"'{strategy['definition']}' (min 10 chars)"
            )

    # Validation 4: Check manifest exists (same as discovery validation)
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 887-892
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"selection_manifest.json not found at {manifest_path}. "
            "Did Stage 2.5 complete successfully?"
        )


# ===== BUSINESS LOGIC VALIDATION =====
# Source: ContentAnalysisCHILD.md Section 2.3.X Edge Cases tables

def validate_business_rules_sampling(sampled_transcripts: List[Dict[str, Any]]):
    """
    Validate business rules during sampling.

    Source: ContentAnalysisCHILDTI.md Section 5.2

    Args:
        sampled_transcripts: List of sampled transcript dicts with video_id, text, bucket
    """
    # Rule 1: Bucket with < 17 videos
    # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases row 1
    # Handling: Sample all available (warn, don't fail)
    bucket_counts = {}
    for transcript in sampled_transcripts:
        bucket = transcript['bucket']
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    for bucket, count in bucket_counts.items():
        if count < 17:
            logger.warning(
                f"⚠️  Bucket {bucket} has only {count} sampled transcripts (expected ~17). "
                f"Bucket may have insufficient videos."
            )

    # Rule 2: Empty transcripts included
    # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases row 3
    # Handling: Allow (may reveal no-speech patterns)
    empty_count = sum(1 for t in sampled_transcripts if not t['text'])
    if empty_count > 0:
        logger.info(
            f"ℹ️  {empty_count}/{len(sampled_transcripts)} transcripts are empty (no speech). "
            f"Including for potential no-speech pattern detection."
        )


def validate_business_rules_classification(
    video_id: str,
    transcript: Dict[str, Any],
    caption: str,
    hashtags: List[str]
):
    """
    Validate business rules during classification.

    Source: ContentAnalysisCHILDTI.md Section 5.2

    Args:
        video_id: Video identifier
        transcript: Transcript dict with 'text' field
        caption: Video caption text
        hashtags: List of hashtag strings
    """
    # Rule 1: Empty transcript handling
    # Source: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases row 1
    # Handling: Classify using caption + hashtags only (warn, don't fail)
    if not transcript.get('text', ''):
        logger.warning(
            f"⚠️  Video {video_id} has empty transcript. "
            f"Classifying using caption and hashtags only."
        )

    # Rule 2: Missing caption handling
    # Source: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases row 2
    # Handling: Use empty string, continue with transcript + hashtags
    if not caption:
        logger.debug(f"Video {video_id} has no caption. Using transcript + hashtags.")

    # Rule 3: Missing hashtags handling
    # Source: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases row 3
    # Handling: Use empty array, continue with transcript + caption
    if not hashtags:
        logger.debug(f"Video {video_id} has no hashtags. Using transcript + caption.")


# ===== DISCOVERY OUTPUT VALIDATION =====
# Source: ContentAnalysisCHILD.md Section 6.3 (lines 920-955)

def validate_discovery_output(raw_taxonomy: Dict[str, Any]):
    """
    Validate raw discovery JSON before saving.

    Source: ContentAnalysisCHILDTI.md Section 5.3

    Args:
        raw_taxonomy: LLM-generated discovery output

    Raises:
        ValueError: If output structure invalid
    """
    # Validation 1: Check top-level fields
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 926-929
    required_top_level = ['hashtag', 'analysis_date', 'sample_size', 'discovered_patterns']
    missing = [f for f in required_top_level if f not in raw_taxonomy]
    if missing:
        raise ValueError(f"Discovery output missing fields: {missing}")

    # Validation 2: Check discovered_patterns has all 7 categories
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 931-937
    required_patterns = [
        'content_categories', 'hook_strategies', 'closing_strategies',
        'audience_pain_points', 'trending_keywords', 'engagement_drivers', 'content_tactics'
    ]
    patterns = raw_taxonomy['discovered_patterns']
    missing = [f for f in required_patterns if f not in patterns]
    if missing:
        raise ValueError(f"Discovered patterns missing categories: {missing}")

    # Validation 3: Check each pattern array is non-empty (warn only)
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 939-944
    for category in required_patterns:
        if not patterns[category]:
            logger.warning(f"Discovery found 0 patterns for {category}. This is unusual.")

    # Validation 4: Check pattern objects have required fields
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 946-954
    for category in ['content_categories', 'hook_strategies']:
        for pattern in patterns[category]:
            required_fields = ['name', 'frequency', 'examples']
            missing = [f for f in required_fields if f not in pattern]
            if missing:
                raise ValueError(
                    f"Pattern in {category} missing fields: {missing}. Pattern: {pattern}"
                )


# ===== CLASSIFICATION OUTPUT VALIDATION =====
# Source: 2.7ClassificationCritique.md - Section 9 (Refined Schema)
# Updated: Reflects refined prompt with 12 fields (simplified from 23 fields)

def validate_classification_output(classification: Dict[str, Any]):
    """
    Validate classification JSON before saving.

    Source: ContentAnalysisCHILDTI.md Section 5.3
            2.7ClassificationCritique.md Section 9 - Final Schema

    Args:
        classification: LLM-generated classification output

    Raises:
        ValueError: If output structure invalid
    """
    # Validation 1: Check all 13 top-level fields present
    # Source: 2.7ClassificationCritique.md Section 9 - Required fields
    required_fields = [
        'video_id', 'taxonomy_version', 'content_category', 'hook_strategy',
        'closing_strategy', 'pain_points', 'keywords', 'engagement_drivers', 'content_tactics',
        'caption_analysis', 'confidence', 'transcript_available', 'note'
    ]
    missing = [f for f in required_fields if f not in classification]
    if missing:
        raise ValueError(f"Classification missing required fields: {missing}")

    # Validation 2: Check taxonomy_version is correct
    # Source: 2.7ClassificationCritique.md Section 9
    if classification['taxonomy_version'] != 'stage2.6_output':
        raise ValueError(
            f"Invalid taxonomy_version: {classification['taxonomy_version']}. "
            f"Must be 'stage2.6_output'."
        )

    # Validation 3: Check confidence value
    # Source: 2.7ClassificationCritique.md Section 5 - Confidence Assessment
    if classification['confidence'] not in ['high', 'medium', 'low']:
        raise ValueError(
            f"Invalid confidence value: {classification['confidence']}. "
            f"Must be high, medium, or low."
        )

    # Validation 4: Check caption_analysis has all 8 subfields (simplified from 12)
    # Source: 2.7ClassificationCritique.md Section 6 - Caption Analysis Fields
    caption_fields = [
        'hook_type', 'cta_type', 'brand_mention_present', 'influencer_tag_present',
        'emoji_usage', 'caption_length', 'hashtag_count', 'hashtag_placement'
    ]
    caption_analysis = classification['caption_analysis']
    missing = [f for f in caption_fields if f not in caption_analysis]
    if missing:
        raise ValueError(f"caption_analysis missing fields: {missing}")

    # Validation 5: Check arrays are actually arrays
    # Note: Field names changed from audience_pain_points → pain_points, trending_keywords → keywords
    array_fields = ['pain_points', 'keywords', 'engagement_drivers', 'content_tactics']
    for field in array_fields:
        if not isinstance(classification[field], list):
            raise ValueError(f"Field {field} must be array, got {type(classification[field])}")

    # Validation 6: Check boolean fields are booleans
    # Source: 2.7ClassificationCritique.md Section 9 - JSON Formatting Rules
    boolean_fields = ['transcript_available']
    caption_boolean_fields = ['brand_mention_present', 'influencer_tag_present']
    for field in boolean_fields:
        if not isinstance(classification[field], bool):
            raise ValueError(f"Field {field} must be boolean, got {type(classification[field])}")
    for field in caption_boolean_fields:
        if not isinstance(caption_analysis[field], bool):
            raise ValueError(f"caption_analysis.{field} must be boolean, got {type(caption_analysis[field])}")
