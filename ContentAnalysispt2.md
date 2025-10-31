# Stage 2 Content Analysis - Implementation Guide

> **Date**: 2025-10-31
> **Status**: Ready for Implementation
> **Related Files**:
> - `ml_pipeline/stage2_content_analysis/utils.py`
> - `ml_pipeline/stage2_content_analysis/validate_transcripts.py`
> - `ml_pipeline/stage2_content_analysis/discovery.py`
> - `ml_pipeline/stage2_content_analysis/classification.py`
> **Output Paths**:
> - `/data/clients/{client}/hashtags/{hashtag}/top_contrastive/content_analysis/raw_llm_output/`
> - `/data/clients/{client}/hashtags/{hashtag}/top_contrastive/content_analysis/validated/`

---

## Table of Contents

1. [Overview](#overview)
2. [Solution Architecture](#solution-architecture)
   - [Transcript Analysis Results](#transcript-analysis-results)
3. [Implementation Steps](#implementation-steps)
   - [Step 1: Shared Validation Function](#step-1-shared-validation-function)
   - [Step 2: Upstream Transcript Validation (Stage 2.5.1)](#step-2-upstream-transcript-validation-stage-251)
   - [Step 3: Updated Stage 2.6 Sampling (Adaptive)](#step-3-updated-stage-26-sampling-adaptive)
   - [Step 4: Dual-Flow Stage 2.7 Classification](#step-4-dual-flow-stage-27-classification)
4. [Execution Flow](#execution-flow)
5. [Expected Impact](#expected-impact)

---

## Overview

This guide documents the complete implementation of Stage 2 Content Analysis (Stages 2.5.1, 2.6, and 2.7), which classifies TikTok videos using LLM-based taxonomy discovery and dual-flow classification.

### **Key Problem Solved**

**42% of transcripts are invalid** (music/noise only), causing:
- Taxonomy contamination during discovery
- LLM hallucinations during classification
- Inconsistent output formatting

### **Solution Highlights**

1. **Upstream Transcript Validation** - Validate all transcripts once, use everywhere
2. **Adaptive Sampling** - 60 transcripts (20 per bucket) with adaptive compensation
3. **Dual-Flow Classification** - Separate flows for valid vs invalid transcripts
4. **Validation Layer** - Auto-correct LLM output formatting
5. **Dual Outputs** - Save both raw LLM responses and validated production files

### **Performance & Cost Estimates**

**M12 FIX: Added resource planning estimates**

| Stage | Duration | Cost | Notes |
|-------|----------|------|-------|
| Stage 2.5.1 | ~2 min | $0 | Transcript validation (local processing) |
| Stage 2.6 | ~1 min | ~$0.75 | Discovery using Sonnet 4.5, 60 transcripts |
| Stage 2.7 (300 videos) | ~75 min | ~$0.30 | Classification using Haiku, dual-flow |
| **Total Pipeline** | **~78 min** | **~$1.05** | For typical 300-video hashtag |

**Cost Breakdown by Flow:**
- Flow 1 (valid transcript): ~$0.001 per video (full classification)
- Flow 2 (invalid transcript): ~$0.0005 per video (50% cheaper - caption only)

**Disk Space:**
- Raw LLM outputs: ~500KB (300 files × ~1.5KB avg)
- Validated outputs: ~1MB (300 files × ~3KB avg)
- Total: ~1.5MB per hashtag

---

## Solution Architecture

### **Transcript Analysis Results**

Analysis of 898 transcripts revealed critical data quality issues:

**Overall Statistics:**
- Total transcripts: 898
- Valid transcripts: 520 (57.9%)
- **Invalid transcripts: 378 (42.1%)** ← Nearly HALF are unusable!

**Invalid Pattern Breakdown:**

| Pattern Type | Count | % of Total | Examples |
|--------------|-------|-----------|----------|
| **Empty/Too Short** | 273 | 30.4% | `""` (empty), `"[Music]"` (1 word), `"Thank you"` (2 words) |
| **Music Brackets** | 58 | 6.5% | `"[Music] [Music] [Music]"`, `"[MUSIC]"` |
| **Song Lyrics** | 22 | 2.4% | `"♪ Give me your kiss tonight ♪ ♪ On the phone ♪"` |
| **Sound Effects** | 8 | 0.9% | `"[clicking] [clicking]"`, `"[typing]"`, `"[sizzling]"` |
| **Mixed Valid** | 8 | 0.9% | Has music notes BUT also real speech (VALID) |
| **Other Noise** | 9 | 1.0% | `"[SPEAKING SPANISH]"`, `"[knocking on door]"` |

**Specific Pattern Examples:**

1. **Music Bracket Variations:**
   - `"[Music]"` - Most common (single bracket)
   - `"[Music] [Music] [Music] [Music] [Music]"` - Repeated (15 occurrences)
   - `"[music]"` - Lowercase variant
   - `"[MUSIC]"` - Uppercase variant
   - `"(upbeat music)"` - Parentheses variant

2. **Lyrics Detection (3+ ♪ symbols):**
   - `"♪ Give me your kiss tonight ♪ ♪ On the phone ♪"`
   - `"♪ She's got a, she's got a way ♪ ♪ She's got a way ♪"`
   - Pure symbols: `"♪ ♪ ♪ ♪ ♪ ♪ ♪ ♪ ♪ ♪"`

3. **Sound Effect Markers:**
   - `"[clicking] [clicking] [clicking]..."` (repeated)
   - `"[typing] [typing] [typing]..."`
   - `"[sizzling] [sizzling] [sizzling]..."`
   - `"[knocking on door]"`
   - `"[birds chirping] [water splashing]"`

4. **Mixed Content (VALID):**
   - `"I got one of them, Canada man was on one of them. ♪ There is one on my promise ♪"` ✅
   - Has real speech + some music notes → Should be classified as VALID

With 42% of transcripts invalid, upstream validation is critical to prevent taxonomy contamination and hallucinations.

---

## Error Handling Philosophy

**M4 FIX: Documented error handling approach**

This implementation uses **context-appropriate error handling** strategies:

### **Stage Boundaries: Fail-Fast**
Errors at stage boundaries (missing files, invalid inputs) raise exceptions and halt pipeline:
- **Stage 2.5.1:** Missing manifest → FileNotFoundError (halt)
- **Stage 2.6:** Manifest not validated → ValueError (halt or auto-fix)
- **Stage 2.7:** Missing taxonomy → FileNotFoundError (halt)

**Rationale:** Stage transitions represent critical dependencies. Better to fail early than produce invalid outputs.

### **LLM Outputs: Graceful Auto-Correction**
Validation layer auto-corrects LLM formatting issues and logs warnings:
- Invalid `hook_type` → Default to "statement" (continue)
- Invalid `cta_type` → Default to "none" (continue)
- Multi-select in single field → Keep first item (continue)
- Category not in taxonomy → Fuzzy match or fallback (continue)

**Rationale:** LLM errors are recoverable. Auto-correction prevents pipeline failures from minor formatting issues while preserving data.

**M13 FIX: Already addressed via strict_mode parameter**
Production code supports `strict_mode=False` (default) for auto-correction. See validation.py:298+ and earlier answer A11-A12.

### **Missing Data: Sensible Defaults**
Missing non-critical data uses defaults and logs warnings:
- Missing video duration → Default to 0.0 (continue with warning)
- Missing caption → Empty string (continue)
- Missing hashtags → Empty array (continue)

**Rationale:** Non-critical data shouldn't block classification. Defaults allow processing while warnings flag issues for review.

### **Summary Table**

| Error Type | Strategy | Example | Outcome |
|------------|----------|---------|---------|
| Missing prerequisite file | **Fail-fast** | selection_manifest.json not found | Raise FileNotFoundError, halt |
| Invalid stage input | **Fail-fast** | Manifest missing validation results | Raise ValueError or auto-fix |
| LLM formatting error | **Auto-correct** | hook_type="declarative" instead of "statement" | Correct + warn, continue |
| LLM invalid category | **Auto-correct** | Category not in taxonomy | Fuzzy match + warn, continue |
| Missing optional data | **Default** | Video duration not found | Use 0.0 + warn, continue |

---

## Implementation Steps

```
Stage 2.5 → selection_manifest.json (N videos per bucket, dynamic)
            ↓
Stage 2.5.1 → Validate ALL transcripts → Enhanced manifest with validation results
            ↓
Stage 2.6 → Sample 60 from VALID top performers (20 per bucket, adaptive) → Taxonomy discovery
            ↓
Stage 2.7 → Classify ALL videos using dual flows → Save to bucket subdirectories:
            - Flow 1 (valid transcripts): Full classification
            - Flow 2 (invalid transcripts): Caption analysis only
```

---

## Step 1: Shared Validation Function

**File:** `ml_pipeline/stage2_content_analysis/utils.py`

```python
def is_transcript_invalid(transcript: Dict[str, Any]) -> bool:
    """
    Detect invalid transcripts (music/lyrics/noise only).
    Used by Stage 2.5.1 (validation), Stage 2.6 (sampling), Stage 2.7 (flow routing).

    Invalid patterns (from 898 transcript analysis):
    - 273 (30.4%): Empty/too short (≤3 words)
    - 58 (6.5%): Music brackets only
    - 22 (2.4%): Song lyrics (3+ ♪ symbols)
    - 8 (0.9%): Sound effects only
    - 9 (1.0%): Other noise markers

    Returns True if transcript should not be used for content classification.
    """
    import re

    if not transcript.get('available', False):
        return True

    text = transcript.get('text', '').strip()

    # Empty or too short (≤3 words)
    if not text or len(text.split()) <= 3:
        return True

    # Remove all music/sound markers
    clean_text = re.sub(r'\[.*?\]|\(.*?\)', '', text)  # Bracketed: [Music], (upbeat music)
    clean_text = re.sub(r'[♪♫]+', '', clean_text)      # Music notes
    clean_text = clean_text.strip()

    # If <10 words remain after removing markers, it's invalid
    if len(clean_text.split()) < 10:
        return True

    return False
```

---

## Step 2: Upstream Transcript Validation (Stage 2.5.1)

**File:** `ml_pipeline/stage2_content_analysis/validate_transcripts.py`

```python
def validate_all_transcripts_in_manifest(
    manifest_path: str,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Validate ALL transcripts in selection manifest ONCE.

    Handles dynamic video counts (e.g., 100, 200, 150 per bucket).
    Validates top + bottom performers.

    Args:
        manifest_path: Path to selection_manifest.json from Stage 2.5
        output_path: Path to save enhanced manifest (default: overwrite original)

    Returns:
        Enhanced manifest with validation results per bucket
    """
    from .utils import is_transcript_invalid, load_transcript

    manifest = load_json(manifest_path)

    logger.info("🔍 Validating ALL transcripts in manifest...")

    total_valid = 0
    total_invalid = 0

    # Validate transcripts for each bucket
    for bucket in manifest['selected_buckets']:
        bucket_data = manifest['videos_by_bucket'][bucket]

        # Get ALL video IDs dynamically (top + bottom, whatever manifest contains)
        top_performers = bucket_data.get('top_performers', [])
        bottom_performers = bucket_data.get('bottom_performers', [])
        all_video_ids = top_performers + bottom_performers

        valid_ids = []
        invalid_ids = []

        logger.info(f"  Validating bucket {bucket}: {len(all_video_ids)} videos")

        # Validate each transcript
        for video_id in all_video_ids:
            transcript = load_transcript(video_id)

            if is_transcript_invalid(transcript):
                invalid_ids.append(video_id)
                total_invalid += 1
            else:
                valid_ids.append(video_id)
                total_valid += 1

        # Add validation results to bucket
        bucket_data['validated_transcripts'] = {
            'valid': valid_ids,
            'invalid': invalid_ids,
            'valid_count': len(valid_ids),
            'invalid_count': len(invalid_ids)
        }

        logger.info(
            f"    ✅ {len(valid_ids)} valid ({len(valid_ids)/len(all_video_ids)*100:.1f}%)\n"
            f"    ❌ {len(invalid_ids)} invalid ({len(invalid_ids)/len(all_video_ids)*100:.1f}%)"
        )

    # Add overall statistics
    manifest['validation_summary'] = {
        'total_videos': total_valid + total_invalid,
        'valid_transcripts': total_valid,
        'invalid_transcripts': total_invalid,
        'validation_rate': total_valid / (total_valid + total_invalid),
        'validated_at': datetime.utcnow().isoformat() + 'Z'
    }

    # Save enhanced manifest
    if output_path is None:
        output_path = manifest_path  # Overwrite original

    save_json(output_path, manifest)

    logger.info(
        f"\n✅ Validation complete:\n"
        f"   Total: {total_valid + total_invalid}\n"
        f"   Valid: {total_valid} ({total_valid/(total_valid + total_invalid)*100:.1f}%)\n"
        f"   Invalid: {total_invalid} ({total_invalid/(total_valid + total_invalid)*100:.1f}%)"
    )

    return manifest


def load_transcript(video_id: str) -> Dict[str, Any]:
    """Load transcript from disk."""
    # L17 FIX: Default path is dev convenience - always set RUMIAI_ROOT in production
    RUMIAI_ROOT = os.environ.get('RUMIAI_ROOT', '/home/jorge/rumiaifinal')
    transcript_path = f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json"

    try:
        transcript_data = load_json(transcript_path)
        return {
            'text': transcript_data.get('text', ''),
            'available': True
        }
    except FileNotFoundError:
        return {
            'text': '',
            'available': False
        }
```

**Enhanced Manifest Schema:**
```json
{
  "hashtag": "wellnesspt2_test5",
  "selected_buckets": ["33-60s", "60-90s", "90-120s"],
  "videos_by_bucket": {
    "33-60s": {
      "top_performers": [/* dynamic count */],
      "bottom_performers": [/* dynamic count */],

      "validated_transcripts": {
        "valid": [/* video IDs */],
        "invalid": [/* video IDs */],
        "valid_count": 46,
        "invalid_count": 34
      }
    }
  },
  "validation_summary": {
    "total_videos": 300,
    "valid_transcripts": 174,
    "invalid_transcripts": 126,
    "validation_rate": 0.58,
    "validated_at": "2025-01-30T10:30:00Z"
  }
}
```

---

## Step 3: Updated Stage 2.6 Sampling (Adaptive)

**File:** `ml_pipeline/stage2_content_analysis/discovery.py`

```python
def sample_transcripts_for_discovery(
    manifest_path: str,
    sample_size: int = 60,
    random_seed: Optional[int] = None
) -> list[dict]:
    """
    Sample 60 valid transcripts with adaptive distribution across buckets.

    Strategy:
    - Target: 20 per bucket (balanced duration representation)
    - If bucket has <20 valid: Take all (signal: weak bucket)
    - Compensate shortfall from buckets with surplus
    - Never fails: Adapts to available data

    Requires pre-validated manifest from Stage 2.5.1.
    **Auto-runs Stage 2.5.1 if validation results missing.**

    Args:
        manifest_path: Path to VALIDATED selection_manifest.json
        sample_size: Total transcripts (default: 60)
        random_seed: Optional seed for reproducible sampling (M9 FIX)
                     If None, sampling is random (non-reproducible)
                     Useful for testing and debugging taxonomy issues

    Returns:
        list[dict]: Sampled valid transcripts (~20 per bucket, adaptive)
    """
    from .utils import load_transcript
    from .validate_transcripts import validate_all_transcripts_in_manifest

    # M9 FIX: Set random seed if provided for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        logger.info(f"🎲 Random seed set to {random_seed} for reproducible sampling")

    manifest = load_json(manifest_path)

    # C1 FIX: Auto-run validation if missing
    if 'validation_summary' not in manifest:
        logger.warning(
            "⚠️  Manifest not validated. Auto-running Stage 2.5.1 transcript validation..."
        )
        manifest = validate_all_transcripts_in_manifest(manifest_path)
        logger.info("✅ Auto-validation complete. Proceeding with sampling...")

    top_3_buckets = manifest['selected_buckets']
    # L18 FIX: 20 samples per bucket = balanced duration representation across 3 buckets
    target_per_bucket = sample_size // 3  # 20 samples per bucket (default: 60 // 3)

    # STEP 1: Assess available valid transcripts per bucket
    bucket_valid_counts = {}
    bucket_valid_ids = {}

    for bucket in top_3_buckets:
        bucket_data = manifest['videos_by_bucket'][bucket]
        top_performers = bucket_data['top_performers']
        valid_ids = bucket_data['validated_transcripts']['valid']

        valid_top_performers = [vid for vid in top_performers if vid in valid_ids]
        bucket_valid_counts[bucket] = len(valid_top_performers)
        bucket_valid_ids[bucket] = valid_top_performers

    # STEP 2: Determine sampling plan
    sampling_plan = {}
    shortfall = 0
    surplus_buckets = []

    for bucket in top_3_buckets:
        available = bucket_valid_counts[bucket]

        if available < target_per_bucket:
            # Weak bucket: take all available
            sampling_plan[bucket] = available
            shortfall += (target_per_bucket - available)
            logger.warning(
                f"⚠️  Bucket {bucket} has only {available} valid top performers "
                f"(target: {target_per_bucket}). Taking all {available}."
            )
        else:
            # Strong bucket: initially plan for target, may increase later
            sampling_plan[bucket] = target_per_bucket
            surplus_buckets.append(bucket)

    # STEP 3: Distribute shortfall among surplus buckets
    if shortfall > 0 and surplus_buckets:
        extra_per_surplus = shortfall // len(surplus_buckets)
        remainder = shortfall % len(surplus_buckets)

        for i, bucket in enumerate(surplus_buckets):
            extra = extra_per_surplus
            if i < remainder:  # Distribute remainder to first N buckets
                extra += 1

            # Ensure we don't exceed available
            max_possible = bucket_valid_counts[bucket]
            sampling_plan[bucket] = min(sampling_plan[bucket] + extra, max_possible)

        logger.info(
            f"📊 Compensating shortfall of {shortfall} transcripts "
            f"from {len(surplus_buckets)} surplus buckets"
        )

    # STEP 4: Sample according to plan
    sampled_transcripts = []

    for bucket in top_3_buckets:
        sample_count = sampling_plan[bucket]
        valid_ids = bucket_valid_ids[bucket]

        if sample_count == 0:
            logger.warning(f"⚠️  Bucket {bucket}: No valid transcripts available, skipping")
            continue

        # Sample exactly according to plan
        sampled_ids = random.sample(valid_ids, sample_count)

        # Load transcripts
        for video_id in sampled_ids:
            transcript = load_transcript(video_id)
            sampled_transcripts.append({
                "video_id": video_id,
                "text": transcript['text'],
                "bucket": bucket
            })

        logger.info(
            f"  ✅ Bucket {bucket}: Sampled {sample_count} from {len(valid_ids)} valid "
            f"({'balanced' if sample_count == target_per_bucket else 'adapted'})"
        )

    total_sampled = len(sampled_transcripts)
    logger.info(
        f"✅ Total sampled: {total_sampled} transcripts "
        f"({'balanced' if total_sampled == sample_size else 'adapted distribution'})"
    )

    # STEP 5: Warn if we couldn't reach target
    if total_sampled < sample_size:
        logger.warning(
            f"⚠️  Only sampled {total_sampled}/{sample_size} transcripts. "
            f"Some buckets had very few valid top performers."
        )

    return sampled_transcripts
```

**Example Scenarios:**

**Scenario 1: All Buckets Healthy**
```
Bucket 33-60s: 46 valid → Sample 20
Bucket 60-90s: 48 valid → Sample 20
Bucket 90-120s: 44 valid → Sample 20
Total: 60 ✅ (balanced)
```

**Scenario 2: One Weak Bucket**
```
Bucket 33-60s: 12 valid → Sample 12 (all)
Bucket 60-90s: 50 valid → Sample 24 (compensate +4)
Bucket 90-120s: 48 valid → Sample 24 (compensate +4)
Total: 60 ✅ (adapted)
```

**Scenario 3: Insufficient Total**
```
Bucket 33-60s: 8 valid → Sample 8 (all)
Bucket 60-90s: 12 valid → Sample 12 (all)
Bucket 90-120s: 18 valid → Sample 18 (all)
Total: 38 ⚠️ (proceeds with reduced sample)
```

---

## Step 4: Dual-Flow Stage 2.7 Classification

**File:** `ml_pipeline/stage2_content_analysis/classification.py`

**C8 FIX - Checkpoint/Resume Support:**

Production code already has checkpoint/resume functionality via:
- `checkpoint.py`: `load_checkpoint()`, `save_checkpoint()`, `update_checkpoint()`
- Usage: Sequential and parallel modes support checkpoint files for resuming interrupted classification runs
- Atomic writes prevent corruption if process crashes mid-save
- See production code: `classification.py:553-721` (classify_all_videos_sequential/parallel)

**C7 FIX - Utility Functions:**

The following utility functions are already implemented in production:
- `load_json(file_path)` → In `utils.py:63-94`
- `save_json(file_path, data)` → In `utils.py:97-137` (atomic write pattern)
- `load_video_data(video_id)` → Defined below for spec completeness (exists in production)

### **Main Classification Function:**

```python
def classify_single_video_with_save(
    video_id: str,
    validated_manifest: Dict[str, Any],
    taxonomy: Dict[str, Any],
    client: anthropic.Anthropic,
    output_base_dir: str
) -> Dict[str, Any]:
    """
    Classify video using dual-flow approach based on pre-validation results.

    Flow 1 (valid transcript): Full classification
    Flow 2 (invalid transcript): Caption analysis only

    Outputs are organized by bucket subdirectories.
    """
    # Load video data
    transcript, caption, hashtags = load_video_data(video_id)

    # Check validation status from manifest (no re-validation!)
    bucket = get_bucket_for_video(video_id, validated_manifest)
    valid_ids = validated_manifest['videos_by_bucket'][bucket]['validated_transcripts']['valid']

    if video_id in valid_ids:
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

        # Save RAW LLM output immediately (before validation)
        raw_output_dir = f"{output_base_dir}/raw_llm_output/bucket_{bucket}"
        os.makedirs(raw_output_dir, exist_ok=True)
        raw_output_path = f"{raw_output_dir}/{video_id}_raw.json"
        save_json(raw_output_path, llm_output)

        # Validate LLM output
        validated_output = validate_classification_output(
            llm_output=llm_output,
            taxonomy=taxonomy,
            flow_type="full"
        )

        # Normalize schema (M10 FIX: pass caption for hashtag_count calculation)
        classification = normalize_classification_schema(
            llm_output=validated_output,
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

        # Save RAW LLM output immediately (before validation)
        raw_output_dir = f"{output_base_dir}/raw_llm_output/bucket_{bucket}"
        os.makedirs(raw_output_dir, exist_ok=True)
        raw_output_path = f"{raw_output_dir}/{video_id}_raw.json"
        save_json(raw_output_path, llm_output)

        # Validate LLM output
        validated_output = validate_classification_output(
            llm_output=llm_output,
            taxonomy=None,
            flow_type="caption_only"
        )

        # Normalize schema (M10 FIX: pass caption for hashtag_count calculation)
        classification = normalize_classification_schema(
            llm_output=validated_output,
            video_id=video_id,
            caption=caption,
            transcript_available=False,
            flow_type="caption_only"
        )

    # Add bucket metadata (C3 FIX: pass client_id and hashtag)
    classification['bucket'] = bucket
    # Extract client_id and hashtag from manifest for get_video_duration()
    client_id = validated_manifest.get('client_id')  # Assume added to manifest
    hashtag = validated_manifest.get('hashtag')
    classification['video_duration'] = get_video_duration(video_id, client_id, hashtag)

    # Save VALIDATED output (final production output)
    validated_output_dir = f"{output_base_dir}/validated/bucket_{bucket}"
    os.makedirs(validated_output_dir, exist_ok=True)
    validated_output_path = f"{validated_output_dir}/{video_id}_content.json"
    save_json(validated_output_path, classification)

    return classification
```

### **Flow 1: Full Classification (Valid Transcript)**

**M5 FIX: Token & Cost Estimates**

**Typical Token Usage (with M10 fix - hashtag_count removed from LLM):**
- Input tokens: ~3,500 (system: ~200, taxonomy: ~2,000, transcript: ~800, prompt: ~500)
- Output tokens: ~380-480 (12 fields + 2 caption subfields, Python adds hashtag_count)
- **Total cost per video:** ~$0.0046 (Haiku pricing: $0.25 input / $1.25 output per 1M tokens)
- **Savings:** ~$0.0001 per video vs asking LLM for hashtag_count

**Large Taxonomy Warning:**
- If curated taxonomy >5,000 tokens: May approach context limits
- Recommendation: Test with sample video before full batch
- Monitor: Check response.usage.input_tokens in production logs

**M14 FIX: Zone Separation Design Choice**

**Note on Zone Boundaries:**
Zone separation is prompt-based guidance, not technical enforcement. The LLM receives all data (transcript + caption + hashtags) in a single call but is instructed to use specific sources per zone. This is intentional:
- **Trade-off:** 2-call approach (truly separate) would be 2× cost and slower
- **Mitigation:** Extensive prompt examples showing correct vs incorrect usage
- **Validation:** Post-processing validates output but cannot detect cross-contamination
- **Risk:** Low in practice; Haiku follows instructions well with clear zone headers

```python
def classify_video_with_transcript(
    video_id: str,
    transcript: Dict[str, Any],
    caption: str,
    hashtags: List[str],
    taxonomy: Dict[str, Any],
    client: anthropic.Anthropic
) -> Dict[str, Any]:
    """
    Flow 1: Full classification using transcript.
    LLM outputs ALL 13 fields.
    Uses physical zone separation to prevent hallucination.
    """

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
{transcript['text']}
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
        # L18 FIX: 1024 tokens = max classification output (13 fields with arrays, ~400-500 typical)
        max_tokens=1024,
        # L18 FIX: 30s timeout = typical Haiku response time for classification task
        timeout=30,
        system=system_message,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)
```

### **Flow 2: Caption Only (Invalid Transcript)**

```python
def classify_caption_only(
    video_id: str,
    caption: str,
    hashtags: List[str],
    client: anthropic.Anthropic
) -> Dict[str, Any]:
    """
    Flow 2: Caption analysis only (no transcript).
    LLM outputs ONLY caption_analysis (3 fields).
    Python constructs full 13-field schema.
    """

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
        # L18 FIX: 128 tokens = caption analysis only (2 fields, minimal output)
        max_tokens=128,
        # L18 FIX: 15s timeout = shorter for simple caption-only task
        timeout=15,
        system=system_message,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)
```

### **Output Validation & Normalization (Post-Processing)**

**File:** `ml_pipeline/stage2_content_analysis/classification.py`

```python
def validate_classification_output(
    llm_output: Dict[str, Any],
    taxonomy: Dict[str, Any],
    flow_type: str
) -> Dict[str, Any]:
    """
    Validate and clean LLM output before normalization.
    Ensures correct formatting for downstream processes.

    Args:
        llm_output: Raw output from LLM
        taxonomy: Taxonomy used for classification (Flow 1 only)
        flow_type: "full" (Flow 1) or "caption_only" (Flow 2)

    Returns:
        Validated and cleaned output
    """
    validated = llm_output.copy()

    # Validate caption_analysis fields (both flows)
    if "caption_analysis" in validated:
        caption = validated["caption_analysis"]

        # Validate hook_type
        valid_hook_types = ["statement", "question", "command", "teaser"]
        if caption.get("hook_type") not in valid_hook_types:
            logger.warning(
                f"Invalid hook_type: {caption.get('hook_type')}. "
                f"Defaulting to 'statement'"
            )
            caption["hook_type"] = "statement"

        # Validate cta_type
        valid_cta_types = ["link_in_bio", "save_post", "comment", "follow", "share", "tag_friend", "none"]
        if caption.get("cta_type") not in valid_cta_types:
            logger.warning(
                f"Invalid cta_type: {caption.get('cta_type')}. "
                f"Defaulting to 'none'"
            )
            caption["cta_type"] = "none"

        # M10 FIX: hashtag_count not expected from LLM (Python will add it)
        # If LLM included it anyway, remove it - will be recalculated by Python
        if "hashtag_count" in caption:
            logger.debug("Removing hashtag_count from LLM output (Python will calculate)")
            del caption["hashtag_count"]

    # Flow 1: Validate taxonomy-based fields
    if flow_type == "full" and taxonomy:
        # Ensure all classification fields are arrays
        array_fields = ["content_category", "hook_strategy", "closing_strategy",
                       "pain_points", "keywords", "engagement_drivers", "content_tactics"]

        for field in array_fields:
            if field in validated and not isinstance(validated[field], list):
                logger.warning(f"{field} is not an array. Converting to array.")
                validated[field] = [validated[field]] if validated[field] else []

        # Validate content_category, hook_strategy, closing_strategy are single items
        single_fields = ["content_category", "hook_strategy", "closing_strategy"]
        for field in single_fields:
            if field in validated and isinstance(validated[field], list):
                if len(validated[field]) > 1:
                    logger.warning(
                        f"{field} has multiple items: {validated[field]}. "
                        f"Keeping only first item."
                    )
                    validated[field] = [validated[field][0]]
                elif len(validated[field]) == 0:
                    logger.warning(f"{field} is empty array. This should be single selection.")

        # C2 FIX: Validate taxonomy category names exist
        from difflib import get_close_matches

        # Validate content_category
        if "content_category" in validated and validated["content_category"]:
            category_names = [cat["name"] for cat in taxonomy.get("content_categories", [])]
            user_category = validated["content_category"][0] if isinstance(validated["content_category"], list) else validated["content_category"]

            if user_category not in category_names:
                # Find closest match
                matches = get_close_matches(user_category, category_names, n=1, cutoff=0.6)
                if matches:
                    logger.warning(
                        f"content_category '{user_category}' not in taxonomy. "
                        f"Did you mean '{matches[0]}'? Using closest match."
                    )
                    validated["content_category"] = [matches[0]]
                else:
                    logger.error(
                        f"content_category '{user_category}' not in taxonomy and no close match found. "
                        f"Available: {category_names}. Using first taxonomy category as fallback."
                    )
                    validated["content_category"] = [category_names[0]] if category_names else []

        # Validate hook_strategy
        if "hook_strategy" in validated and validated["hook_strategy"]:
            strategy_names = [strat["name"] for strat in taxonomy.get("hook_strategies", [])]
            user_strategy = validated["hook_strategy"][0] if isinstance(validated["hook_strategy"], list) else validated["hook_strategy"]

            if user_strategy not in strategy_names:
                matches = get_close_matches(user_strategy, strategy_names, n=1, cutoff=0.6)
                if matches:
                    logger.warning(
                        f"hook_strategy '{user_strategy}' not in taxonomy. "
                        f"Using closest match '{matches[0]}'."
                    )
                    validated["hook_strategy"] = [matches[0]]
                else:
                    logger.error(
                        f"hook_strategy '{user_strategy}' not in taxonomy. "
                        f"Available: {strategy_names}. Using first as fallback."
                    )
                    validated["hook_strategy"] = [strategy_names[0]] if strategy_names else []

        # Validate closing_strategy
        if "closing_strategy" in validated and validated["closing_strategy"]:
            closing_names = [strat["name"] for strat in taxonomy.get("closing_strategies", [])]
            user_closing = validated["closing_strategy"][0] if isinstance(validated["closing_strategy"], list) else validated["closing_strategy"]

            if user_closing not in closing_names:
                matches = get_close_matches(user_closing, closing_names, n=1, cutoff=0.6)
                if matches:
                    logger.warning(
                        f"closing_strategy '{user_closing}' not in taxonomy. "
                        f"Using closest match '{matches[0]}'."
                    )
                    validated["closing_strategy"] = [matches[0]]
                else:
                    logger.error(
                        f"closing_strategy '{user_closing}' not in taxonomy. "
                        f"Available: {closing_names}. Using first as fallback."
                    )
                    validated["closing_strategy"] = [closing_names[0]] if closing_names else []

    # Validate confidence field (Flow 1 only)
    if flow_type == "full":
        valid_confidence = ["high", "medium", "low"]
        if validated.get("confidence") not in valid_confidence:
            logger.warning(
                f"Invalid confidence: {validated.get('confidence')}. "
                f"Defaulting to 'medium'"
            )
            validated["confidence"] = "medium"

    return validated


def normalize_classification_schema(
    llm_output: Dict[str, Any],
    video_id: str,
    caption: str,
    transcript_available: bool,
    flow_type: str
) -> Dict[str, Any]:
    """
    Normalize LLM output to consistent 15-field schema.
    Assumes output has been validated by validate_classification_output().

    M10 FIX: Python calculates hashtag_count deterministically.

    Args:
        llm_output: Validated LLM output (caption_analysis has only hook_type + cta_type)
        video_id: Video identifier
        caption: Caption text (for calculating hashtag_count)
        transcript_available: Whether transcript is valid
        flow_type: "full" (Flow 1) or "caption_only" (Flow 2)

    Returns:
        Complete 15-field classification schema
    """

    # M10 FIX: Calculate hashtag_count in Python (deterministic, accurate)
    hashtag_count = caption.count('#') if caption else 0

    if flow_type == "caption_only":
        # Flow 2: Construct from minimal LLM output
        normalized = {
            "video_id": video_id,
            "taxonomy_version": "stage2.6_output",

            # Content fields (empty for Flow 2)
            "content_category": [],
            "hook_strategy": [],
            "closing_strategy": [],
            "pain_points": [],
            "keywords": [],
            "engagement_drivers": [],
            "content_tactics": [],

            # Caption analysis (from LLM + Python)
            "caption_analysis": {
                "hook_type": llm_output["caption_analysis"]["hook_type"],
                "cta_type": llm_output["caption_analysis"]["cta_type"],
                "hashtag_count": hashtag_count  # Python-calculated
            },

            # Metadata
            "confidence": "low",
            "transcript_available": False,
            "note": "No valid transcript - content classification unavailable"
            # Note: bucket and video_duration added by classify_single_video_with_save
        }

    else:  # flow_type == "full"
        # Flow 1: LLM output + Python-calculated hashtag_count
        normalized = {
            "video_id": llm_output.get("video_id", video_id),
            "taxonomy_version": llm_output.get("taxonomy_version", "stage2.6_output"),
            "content_category": llm_output.get("content_category", []),
            "hook_strategy": llm_output.get("hook_strategy", []),
            "closing_strategy": llm_output.get("closing_strategy", []),
            "pain_points": llm_output.get("pain_points", []),
            "keywords": llm_output.get("keywords", []),
            "engagement_drivers": llm_output.get("engagement_drivers", []),
            "content_tactics": llm_output.get("content_tactics", []),
            "caption_analysis": {
                "hook_type": llm_output["caption_analysis"]["hook_type"],
                "cta_type": llm_output["caption_analysis"]["cta_type"],
                "hashtag_count": hashtag_count  # Python-calculated
            },
            "confidence": llm_output.get("confidence", "medium"),
            "transcript_available": True,
            "note": llm_output.get("note", None)
            # Note: bucket and video_duration added by classify_single_video_with_save
        }

    return normalized
```

**Final Output Schema (15 fields total):**

**M6 FIX: Canonical schema definition - single source of truth**

| # | Field | Type | Source | Required | Description |
|---|-------|------|--------|----------|-------------|
| 1 | `video_id` | string | Input | ✅ | TikTok video identifier |
| 2 | `bucket` | string | Python | ✅ | Duration bucket (e.g., "33-60s") |
| 3 | `video_duration` | float | Python | ✅ | Video duration in seconds |
| 4 | `taxonomy_version` | string | LLM/Python | ✅ | Always "stage2.6_output" |
| 5 | `content_category` | string | LLM (Flow 1) / [] (Flow 2) | ✅ | Single category from taxonomy |
| 6 | `hook_strategy` | string | LLM (Flow 1) / [] (Flow 2) | ✅ | Single strategy from taxonomy |
| 7 | `closing_strategy` | string | LLM (Flow 1) / [] (Flow 2) | ✅ | Single strategy from taxonomy |
| 8 | `pain_points` | array[string] | LLM (Flow 1) / [] (Flow 2) | ✅ | Multiple or empty |
| 9 | `keywords` | array[string] | LLM (Flow 1) / [] (Flow 2) | ✅ | Multiple or empty |
| 10 | `engagement_drivers` | array[string] | LLM (Flow 1) / [] (Flow 2) | ✅ | Multiple or empty |
| 11 | `content_tactics` | array[string] | LLM (Flow 1) / [] (Flow 2) | ✅ | Multiple or empty |
| 12 | `caption_analysis` | object | LLM + Python | ✅ | See subfields below |
| 12.1 | `caption_analysis.hook_type` | string | LLM | ✅ | "statement" \| "question" \| "command" \| "teaser" |
| 12.2 | `caption_analysis.cta_type` | string | LLM | ✅ | "link_in_bio" \| "save_post" \| "comment" \| "follow" \| "share" \| "tag_friend" \| "none" |
| 12.3 | `caption_analysis.hashtag_count` | integer | **Python (M10 FIX)** | ✅ | Calculated via `caption.count('#')` |
| 13 | `confidence` | string | LLM (Flow 1) / "low" (Flow 2) | ✅ | "high" \| "medium" \| "low" |
| 14 | `transcript_available` | boolean | Python | ✅ | true (Flow 1) / false (Flow 2) |
| 15 | `note` | string \| null | LLM (Flow 1) / string (Flow 2) | ✅ | Explanation or null |

**Flow 1 (Valid Transcript):** LLM outputs fields 4-11 + 12.1-12.2 + 13, Python adds 1-3, 12.3, 14-15
**Flow 2 (Invalid Transcript):** LLM outputs fields 12.1-12.2 only, Python constructs all others (including 12.3)

**Example Output:**
```json
{
  "video_id": "7469467763740904746",
  "bucket": "33-60s",
  "video_duration": 45.2,
  "taxonomy_version": "stage2.6_output",
  "content_category": "wellness_routine",
  "hook_strategy": "direct_question",
  "closing_strategy": "follow_cta",
  "pain_points": ["bloating", "low_energy"],
  "keywords": ["morning_routine", "supplements"],
  "engagement_drivers": ["before_after_reveal"],
  "content_tactics": ["direct_to_camera"],
  "caption_analysis": {
    "hook_type": "statement",
    "cta_type": "save_post",
    "hashtag_count": 3
  },
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

---

### **Helper Functions:**

```python
def get_bucket_for_video(video_id: str, manifest: Dict[str, Any]) -> str:
    """Find which bucket a video belongs to."""
    for bucket in manifest['selected_buckets']:
        bucket_data = manifest['videos_by_bucket'][bucket]
        all_videos = bucket_data.get('top_performers', []) + bucket_data.get('bottom_performers', [])
        if video_id in all_videos:
            return bucket
    raise ValueError(f"Video {video_id} not found in manifest")


def get_video_duration(video_id: str, client_id: str, hashtag: str) -> float:
    """
    Load video duration from temporal windows file.

    C3 FIX: Pass client/hashtag to construct exact path (no expensive glob search).

    Args:
        video_id: Video identifier
        client_id: Client identifier (e.g., "acme_corp")
        hashtag: Hashtag name (e.g., "nutrition")

    Returns:
        float: Video duration in seconds, or 0.0 if not found
    """
    from foundation.paths import PathBuilder

    path_builder = PathBuilder()

    # Try to find temporal windows file in target directory
    target_dir = path_builder.get_target_dir(
        client_id=client_id,
        analysis_type="hashtag",
        target=hashtag,
        analysis_mode="top",
        selection_strategy="contrastive"
    )

    # Search in buckets subdirectory
    import glob
    pattern = f"{target_dir}/buckets/**/analysis/insights/{video_id}_temporal_windows_updated.json"
    matches = glob.glob(pattern, recursive=True)

    if matches:
        temporal_data = load_json(matches[0])
        return temporal_data.get('duration', 0.0)

    logger.warning(f"Duration not found for video {video_id}, defaulting to 0.0")
    return 0.0  # Fallback if not found


def load_video_data(video_id: str) -> tuple:
    """
    Load transcript, caption, and hashtags for a video.

    C7 FIX: Define function that was referenced but not shown in spec.

    Returns:
        tuple: (transcript_dict, caption_str, hashtags_list)

    Note: This function exists in production code (classification.py:467-504).
          Reproduced here for spec completeness.
    """
    # L17 FIX: Default path is dev convenience - always set RUMIAI_ROOT in production
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
```

---

## Integration with Main Orchestrator

**Critical Context:** All Stage 2 improvements are orchestrated by `rumiai_ml_batch.py`. The auto-run logic in Stage 2.6 is a **fallback safety mechanism**, not the primary execution path.

### **Flow Control & Backward Compatibility**

**DEFAULT BEHAVIOR: Upgraded Flow (NEW)**

This implementation is a **production upgrade**, not a gradual rollout:
- ✅ **Default:** NEW upgraded flow (dual-flow classification, bucket organization)
- ❌ **No silent fallbacks:** If upgraded flow fails → HARD FAIL (don't fall back to old code)
- 🔧 **Old code:** ONLY activated via explicit manual override

**Rationale:**
- Silent fallbacks mask problems and create inconsistent outputs
- Hard fails surface issues immediately for debugging
- Old code is emergency rollback only (not production path)

**Implementation Strategy:**

```python
# Environment variable controls flow version
USE_LEGACY_FLOW = os.environ.get('USE_LEGACY_FLOW', 'false').lower() == 'true'

# Feature flag for dual-flow classification (default: enabled)
ENABLE_DUAL_FLOW_CLASSIFICATION = os.environ.get('ENABLE_DUAL_FLOW_CLASSIFICATION', 'true').lower() == 'true'

# Feature flag for bucket-organized outputs (default: enabled)
ENABLE_BUCKET_ORGANIZATION = os.environ.get('ENABLE_BUCKET_ORGANIZATION', 'true').lower() == 'true'

if USE_LEGACY_FLOW:
    logger.warning("⚠️  LEGACY FLOW ENABLED (manual override)")
    logger.warning("    This uses old classification logic (single flow, flat structure)")
    logger.warning("    Only use for emergency rollback or testing")
    # Use old code path
else:
    # NEW UPGRADED FLOW (DEFAULT)
    if not ENABLE_DUAL_FLOW_CLASSIFICATION:
        raise ValueError(
            "Dual-flow classification is REQUIRED in upgraded flow. "
            "If you need old behavior, set USE_LEGACY_FLOW=true (not recommended)."
        )

    if not ENABLE_BUCKET_ORGANIZATION:
        raise ValueError(
            "Bucket-organized outputs are REQUIRED in upgraded flow. "
            "If you need old behavior, set USE_LEGACY_FLOW=true (not recommended)."
        )

    # Proceed with upgraded flow
    run_stage_2_5_1(...)  # NEW: Transcript validation
    run_discovery_stage(...)  # UPGRADED: Adaptive sampling
    run_classification_stage(...)  # UPGRADED: Dual-flow classification
```

**Emergency Rollback (Manual Override):**

```bash
# To use old code (NOT RECOMMENDED - for emergency only)
export USE_LEGACY_FLOW=true
python rumiai_ml_batch.py --client test --hashtag wellness

# Logs will show:
# ⚠️  LEGACY FLOW ENABLED (manual override)
# ⚠️  This uses old classification logic
# ⚠️  Only use for emergency rollback or testing
```

**Hard Fail Examples:**

```python
# Example 1: Stage 2.5.1 validation fails
try:
    run_stage_2_5_1(client_id, hashtag, analysis_type)
except Exception as e:
    logger.error("❌ HARD FAIL: Stage 2.5.1 validation failed")
    logger.error(f"   Error: {str(e)}")
    logger.error("   SOLUTION: Fix validation issue or check transcripts exist")
    logger.error("   DO NOT fall back to old code - this must work")
    raise  # Hard fail - don't continue

# Example 2: Dual-flow classification logic fails
try:
    classification = classify_single_video_with_save(...)
except Exception as e:
    logger.error(f"❌ HARD FAIL: Dual-flow classification failed for video {video_id}")
    logger.error(f"   Error: {str(e)}")
    logger.error("   SOLUTION: Debug classification logic")
    logger.error("   DO NOT fall back to old code - this must work")
    raise  # Hard fail - don't continue

# Example 3: Bucket organization fails
try:
    os.makedirs(f"{output_base_dir}/validated/bucket_{bucket}", exist_ok=True)
except Exception as e:
    logger.error("❌ HARD FAIL: Cannot create bucket-organized directory structure")
    logger.error(f"   Error: {str(e)}")
    logger.error("   SOLUTION: Check disk permissions or available space")
    logger.error("   DO NOT fall back to flat structure - this must work")
    raise  # Hard fail - don't continue
```

**Deployment Checklist:**

Before deploying upgraded flow:
1. ✅ Verify `USE_LEGACY_FLOW` NOT set in production environment
2. ✅ Test upgraded flow on single hashtag end-to-end
3. ✅ Verify bucket-organized outputs created correctly
4. ✅ Test checkpoint/resume with upgraded structure
5. ✅ Clear old checkpoints from previous runs
6. ✅ Document rollback procedure (set USE_LEGACY_FLOW=true if needed)

---

### **Checkpoint System Compatibility**

**Core Logic:** Checkpoint system (`checkpoint.py`) **requires NO changes** - it tracks video IDs, not file paths.

**Version Tracking:** Add version to checkpoint to prevent mixed-mode processing:

```python
# checkpoint.py enhancement (recommended)

def save_checkpoint(checkpoint_path: str, completed_ids: list, failed_ids: list):
    """Save checkpoint with version tracking."""

    checkpoint = {
        "version": "2.0",  # NEW: Version tracking
        "completed": completed_ids,
        "failed": failed_ids,
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "output_structure": "bucket_organized"  # Track structure type
    }

    # Atomic write (existing logic)
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(temp_path, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint with version validation and hard-fail on mismatch."""

    if not os.path.exists(checkpoint_path):
        return {"completed": [], "failed": [], "version": "2.0"}

    checkpoint = load_json(checkpoint_path)
    checkpoint_version = checkpoint.get("version", "1.0")  # Old = v1.0

    if checkpoint_version == "1.0":
        # HARD FAIL - do not allow mixed versions
        raise ValueError(
            "❌ INCOMPATIBLE CHECKPOINT DETECTED\n"
            f"   Checkpoint version: {checkpoint_version} (old flat structure)\n"
            f"   Current code version: 2.0 (bucket-organized structure)\n"
            "\n"
            "   SOLUTIONS:\n"
            "   1. [RECOMMENDED] Delete checkpoint to restart with new structure:\n"
            f"      rm {checkpoint_path}\n"
            "\n"
            "   2. [EMERGENCY ONLY] Use legacy flow:\n"
            "      export USE_LEGACY_FLOW=true\n"
            "\n"
            "   Mixed-version processing creates inconsistent outputs. Hard fail enforced."
        )

    return checkpoint
```

**Migration Strategy (Hard Fail Approach):**

```python
# In rumiai_ml_batch.py - before Stage 2.7

# Check for old checkpoints BEFORE processing
checkpoint_path = f"{output_base_dir}/.checkpoint.json"

try:
    checkpoint = load_checkpoint(checkpoint_path)
except ValueError as e:
    logger.error(str(e))
    logger.error("\n🛑 PIPELINE HALTED: Checkpoint version mismatch")
    logger.error("   Action required: Delete old checkpoint or use legacy flow")
    sys.exit(1)  # Hard fail - do not continue

# Proceed with classification (checkpoint version validated)
run_classification_stage(...)
```

**Why Hard Fail on Version Mismatch:**

| Scenario | Old Approach (Silent) | New Approach (Hard Fail) |
|----------|----------------------|--------------------------|
| Resume with v1.0 checkpoint | Skips completed videos, writes new videos to new location | ❌ HALT: Version mismatch detected |
| Result (old) | Mixed outputs: some in flat, some in bucket structure | N/A - pipeline stops |
| Result (new) | N/A - pipeline stops | ✅ User deletes checkpoint, restarts clean |
| Data Quality | ❌ Inconsistent, hard to debug | ✅ Consistent structure |

**Automatic Checkpoint Cleanup (Deployment):**

```python
# Add to deployment script or rumiai_ml_batch.py startup

def check_and_clear_old_checkpoints(client_id: str, hashtag: str):
    """
    Check for v1.0 checkpoints and offer to clear them on deployment.
    Only runs once on first upgraded execution.
    """

    base_dir = f"/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/content_analysis"
    checkpoint_path = f"{base_dir}/.checkpoint.json"

    if not os.path.exists(checkpoint_path):
        return  # No checkpoint, proceed

    try:
        checkpoint = load_json(checkpoint_path)
        version = checkpoint.get("version", "1.0")

        if version == "1.0":
            logger.warning("=" * 80)
            logger.warning("⚠️  OLD CHECKPOINT DETECTED (v1.0)")
            logger.warning(f"   Location: {checkpoint_path}")
            logger.warning("   This checkpoint was created with old code (flat structure)")
            logger.warning("   New code uses bucket-organized structure (v2.0)")
            logger.warning("=" * 80)
            logger.warning("\n📋 OPTIONS:")
            logger.warning("   1. Delete checkpoint → Restart classification with new structure")
            logger.warning("   2. Keep checkpoint → Use legacy flow (USE_LEGACY_FLOW=true)")
            logger.warning("   3. Exit → Manual investigation")
            logger.warning("=" * 80)

            response = input("Delete checkpoint and restart? (yes/no/exit): ").lower()

            if response == 'yes':
                os.remove(checkpoint_path)
                logger.info("✅ Checkpoint deleted. Classification will restart from beginning.")
                logger.info("   All videos will be reprocessed with new dual-flow logic.")
            elif response == 'no':
                logger.error("❌ Cannot proceed with v1.0 checkpoint and v2.0 code.")
                logger.error("   Set USE_LEGACY_FLOW=true to use old code, or delete checkpoint.")
                sys.exit(1)
            else:
                logger.info("Exiting for manual investigation.")
                sys.exit(0)

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


# Call at pipeline start
if not USE_LEGACY_FLOW:
    check_and_clear_old_checkpoints(client_id, hashtag)
```

**Testing Checkpoint Compatibility:**

```bash
# Test 1: Clean start (no checkpoint)
rm /data/clients/test/hashtags/wellness/top_contrastive/content_analysis/.checkpoint.json
python rumiai_ml_batch.py --client test --hashtag wellness
# Expected: Creates v2.0 checkpoint, runs upgraded flow

# Test 2: Resume with v2.0 checkpoint (same version)
python rumiai_ml_batch.py --client test --hashtag wellness
# Expected: Resumes from checkpoint, continues upgraded flow

# Test 3: v1.0 checkpoint with upgraded code (version mismatch)
# Manually edit checkpoint: {"version": "1.0", "completed": [...]}
python rumiai_ml_batch.py --client test --hashtag wellness
# Expected: HARD FAIL with clear error message, prompts to delete checkpoint

# Test 4: Legacy flow with v1.0 checkpoint (compatible)
export USE_LEGACY_FLOW=true
python rumiai_ml_batch.py --client test --hashtag wellness
# Expected: Resumes with old code (emergency rollback scenario)
```

---

### **Main Orchestrator Updates**

**File:** `rumiai_ml_batch.py`

Add Stage 2.5.1 as explicit step between Stage 2.5 and Stage 2.6:

```python
def run_stage_2_5_1(client_id: str, hashtag: str, analysis_type: str, analysis_mode: str = "top", selection_strategy: str = "contrastive"):
    """
    Stage 2.5.1: Validate ALL transcripts in manifest.

    This is a NEW stage added between 2.5 and 2.6 to ensure transcript quality
    before pattern discovery. Validates transcripts once, results used by Stage 2.6
    (sampling) and Stage 2.7 (flow routing).

    Args:
        client_id: Client identifier (e.g., "acme_corp")
        hashtag: Hashtag name (e.g., "nutrition")
        analysis_type: "hashtag", "competitor", or "creator"
        analysis_mode: "top" or "recent" (default: "top")
        selection_strategy: "contrastive" or "top" (default: "contrastive")

    Outputs:
        Enhanced selection_manifest.json with validation results

    Raises:
        FileNotFoundError: If selection_manifest.json doesn't exist
    """
    from ml_pipeline.stage2_content_analysis.validate_transcripts import validate_all_transcripts_in_manifest
    from foundation.paths import PathBuilder
    import logging

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("STAGE 2.5.1: TRANSCRIPT VALIDATION")
    logger.info(f"Client: {client_id}, Hashtag: #{hashtag}, Type: {analysis_type}")
    logger.info("=" * 80)

    # Construct manifest path
    path_builder = PathBuilder()
    target_dir = path_builder.get_target_dir(
        client_id=client_id,
        analysis_type=analysis_type,
        target=hashtag,
        analysis_mode=analysis_mode,
        selection_strategy=selection_strategy
    )
    manifest_path = str(target_dir / "selection_manifest.json")

    # Run validation
    try:
        enhanced_manifest = validate_all_transcripts_in_manifest(manifest_path)

        logger.info("=" * 80)
        logger.info("STAGE 2.5.1 COMPLETE")
        logger.info(f"Valid: {enhanced_manifest['validation_summary']['valid_transcripts']}")
        logger.info(f"Invalid: {enhanced_manifest['validation_summary']['invalid_transcripts']}")
        logger.info(f"Rate: {enhanced_manifest['validation_summary']['validation_rate']:.1%}")
        logger.info("=" * 80)

        return enhanced_manifest

    except FileNotFoundError as e:
        logger.error(f"❌ Stage 2.5.1 failed: {str(e)}")
        logger.error("   Make sure Stage 2.5 completed successfully.")
        raise
    except Exception as e:
        logger.error(f"❌ Stage 2.5.1 failed with unexpected error: {str(e)}")
        raise
```

### **Updated Pipeline Flow in rumiai_ml_batch.py**

**Before (Old Flow):**
```python
def run_ml_pipeline(client_id: str, hashtag: str, analysis_type: str):
    """Main ML pipeline orchestrator."""

    # ... Stage 1 (discovery) ...

    # Stage 2.5: File organization
    run_stage_2_5(client_id, hashtag, analysis_type)

    # Stage 2.6: Pattern discovery
    from ml_pipeline.stage2_content_analysis.discovery import run_discovery_stage
    run_discovery_stage(client_id, hashtag, analysis_type)

    # Stage 2.7: Video classification
    from ml_pipeline.stage2_content_analysis.classification import run_classification_stage
    run_classification_stage(client_id, hashtag, analysis_type)

    # ... Stage 3+ ...
```

**After (New Flow with Stage 2.5.1):**
```python
def run_ml_pipeline(client_id: str, hashtag: str, analysis_type: str):
    """Main ML pipeline orchestrator with flow control."""

    # FLOW CONTROL: Check version and enforce upgraded flow (DEFAULT)
    USE_LEGACY_FLOW = os.environ.get('USE_LEGACY_FLOW', 'false').lower() == 'true'

    if USE_LEGACY_FLOW:
        logger.warning("=" * 80)
        logger.warning("⚠️  LEGACY FLOW ENABLED (manual override)")
        logger.warning("    Using old classification logic (single flow, flat structure)")
        logger.warning("    Only use for emergency rollback or testing")
        logger.warning("=" * 80)
        # Run old code path
        run_old_stage_2_pipeline(client_id, hashtag, analysis_type)
        return

    # NEW UPGRADED FLOW (DEFAULT PATH)
    logger.info("=" * 80)
    logger.info("🚀 UPGRADED STAGE 2 PIPELINE (v2.0)")
    logger.info("   - Transcript validation (Stage 2.5.1)")
    logger.info("   - Adaptive sampling (20 per bucket)")
    logger.info("   - Dual-flow classification (valid/invalid transcripts)")
    logger.info("   - Bucket-organized outputs")
    logger.info("=" * 80)

    # Check for old checkpoints and offer cleanup
    check_and_clear_old_checkpoints(client_id, hashtag)

    # ... Stage 1 (discovery) ...

    # Stage 2.5: File organization
    logger.info("Starting Stage 2.5: File Organization...")
    run_stage_2_5(client_id, hashtag, analysis_type)
    logger.info("✅ Stage 2.5 complete\n")

    # Stage 2.5.1: Transcript validation (NEW!)
    logger.info("Starting Stage 2.5.1: Transcript Validation...")
    try:
        manifest = run_stage_2_5_1(client_id, hashtag, analysis_type)
    except Exception as e:
        logger.error("❌ HARD FAIL: Stage 2.5.1 validation failed")
        logger.error(f"   Error: {str(e)}")
        logger.error("   SOLUTION: Fix validation issue or check transcripts exist")
        logger.error("   DO NOT fall back to old code")
        raise  # Hard fail
    logger.info("✅ Stage 2.5.1 complete\n")

    # Stage 2.6: Pattern discovery (adaptive sampling)
    logger.info("Starting Stage 2.6: Pattern Discovery...")
    from ml_pipeline.stage2_content_analysis.discovery import run_discovery_stage
    run_discovery_stage(client_id, hashtag, analysis_type)
    # Note: Stage 2.6 will auto-run validation if Stage 2.5.1 was somehow skipped
    logger.info("✅ Stage 2.6 complete\n")

    # Manual curation checkpoint
    logger.info("⏸️  MANUAL CURATION REQUIRED")
    logger.info(f"   1. Review: data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/content_taxonomies/{hashtag}_raw_discovery.json")
    logger.info(f"   2. Curate and save as: {hashtag}_taxonomy.json")
    logger.info(f"   3. Run taxonomy validation: validate_curated_taxonomy()")
    logger.info(f"   4. Resume with Stage 2.7")
    input("Press Enter when manual curation is complete...")

    # Stage 2.7: Video classification (dual-flow)
    logger.info("Starting Stage 2.7: Video Classification...")
    from ml_pipeline.stage2_content_analysis.classification import run_classification_stage
    run_classification_stage(client_id, hashtag, analysis_type)
    # Note: Stage 2.7 reads validation results from manifest for flow routing
    logger.info("✅ Stage 2.7 complete\n")

    # ... Stage 3+ ...
```

### **Fallback Safety: Auto-run in Stage 2.6**

The auto-run logic added to `sample_transcripts_for_discovery()` serves as a **safety net** for:

1. **Manual Execution**: Developer runs Stage 2.6 directly without orchestrator
2. **Debugging**: Testing Stage 2.6 in isolation
3. **Recovery**: Re-running Stage 2.6 after orchestrator crash

**Primary Path (Orchestrator):**
```
Orchestrator → Explicit Stage 2.5.1 call → Validation runs once
                                        ↓
                           Manifest enhanced with validation results
                                        ↓
              Stage 2.6 reads results (no auto-run triggered)
```

**Fallback Path (Manual/Debug):**
```
Developer runs Stage 2.6 directly → Auto-detects missing validation
                                  ↓
                   Auto-runs Stage 2.5.1 → Validation runs
                                         ↓
                        Continues with sampling
```

### **Error Handling in Orchestrator**

**Stage 2.5.1 Validation Failures:**

```python
# In rumiai_ml_batch.py

try:
    run_stage_2_5_1(client_id, hashtag, analysis_type)
except FileNotFoundError:
    logger.error("❌ Pipeline halted: selection_manifest.json not found")
    logger.error("   Diagnosis: Stage 2.5 may have failed")
    logger.error("   Action: Check Stage 2.5 logs and re-run")
    raise
except Exception as e:
    logger.error(f"❌ Pipeline halted: Stage 2.5.1 validation error: {str(e)}")
    logger.error("   Diagnosis: Transcript loading or validation logic failed")
    logger.error("   Action: Check transcripts exist in speech_transcriptions/")
    raise
```

**Insufficient Valid Transcripts Warning:**

```python
# After Stage 2.5.1
manifest = run_stage_2_5_1(client_id, hashtag, analysis_type)
validation_rate = manifest['validation_summary']['validation_rate']

if validation_rate < 0.30:  # Less than 30% valid
    logger.warning("=" * 80)
    logger.warning("⚠️  WARNING: Very low valid transcript rate!")
    logger.warning(f"   Only {validation_rate:.1%} of transcripts are valid")
    logger.warning("   This hashtag may be music-heavy or have poor audio quality")
    logger.warning("   Stage 2.6 discovery may produce weak taxonomy")
    logger.warning("   Consider:")
    logger.warning("     1. Selecting different hashtag")
    logger.warning("     2. Adjusting validation thresholds")
    logger.warning("     3. Proceeding with reduced expectations")
    logger.warning("=" * 80)

    response = input("Continue anyway? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Pipeline halted by user")
        return
```

### **Visual Flow Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│                    rumiai_ml_batch.py                           │
│                    (Main Orchestrator)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2.5: File Organization                                    │
│ Output: selection_manifest.json (N videos per bucket)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2.5.1: Transcript Validation (NEW!)                       │
│ Input:  selection_manifest.json                                 │
│ Action: Validate ALL transcripts (once)                         │
│ Output: Enhanced manifest with validation results               │
│         - valid: [video_ids]                                    │
│         - invalid: [video_ids]                                  │
│         - validation_rate: 0.58                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2.6: Pattern Discovery (Adaptive Sampling)                │
│ Input:  Enhanced manifest with validation results               │
│ Action: Sample 60 VALID transcripts (20 per bucket, adaptive)   │
│         Auto-runs Stage 2.5.1 if validation missing (fallback)  │
│ Output: {hashtag}_raw_discovery.json                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ MANUAL CURATION CHECKPOINT                                      │
│ Human reviews raw_discovery.json and creates taxonomy.json      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2.7: Video Classification (Dual-Flow)                     │
│ Input:  Enhanced manifest with validation results               │
│         Curated taxonomy.json                                   │
│ Action: Classify ALL videos                                     │
│         - Flow 1 (valid): Full classification                   │
│         - Flow 2 (invalid): Caption analysis only               │
│ Output: Dual outputs (raw + validated) by bucket                │
└─────────────────────────────────────────────────────────────────┘
```

### **Testing the Integration**

**Full Pipeline Test:**
```bash
# Run complete pipeline with new Stage 2.5.1
python rumiai_ml_batch.py \
    --client test_client \
    --hashtag wellness_test \
    --analysis-type hashtag

# Expected logs:
# STAGE 2.5: FILE ORGANIZATION
# ✅ Stage 2.5 complete
#
# STAGE 2.5.1: TRANSCRIPT VALIDATION
# 🔍 Validating ALL transcripts in manifest...
# ✅ Validation complete: 174/300 valid (58.0%)
# ✅ Stage 2.5.1 complete
#
# STAGE 2.6: PATTERN DISCOVERY
# ✓ Loaded validation cache - will filter invalid transcripts
# ✅ Sampled 60 transcripts
# ✅ Stage 2.6 complete
```

**Manual Stage Execution (Tests Fallback):**
```bash
# Test auto-run fallback by running Stage 2.6 directly
python -c "
from ml_pipeline.stage2_content_analysis.discovery import run_discovery_stage
run_discovery_stage('test_client', 'wellness_test', 'hashtag')
"

# Expected behavior:
# - Detects missing validation results
# - Auto-runs Stage 2.5.1
# - Proceeds with sampling
```

---

## Execution Flow

```
Stage 2.5: Create selection_manifest.json
           ↓
Stage 2.5.1: Validate ALL transcripts (once)
             - Output: Enhanced manifest with validation results
             - Stats: ~58% valid, ~42% invalid
           ↓
Stage 2.6: Sample 60 from VALID top performers (adaptive)
           - Target: 20 per bucket (balanced duration representation)
           - If bucket <20: Take all, compensate from surplus buckets
           - Use pre-validation results (no re-checking)
           - Send ~60 valid transcripts to LLM
           ↓
Stage 2.7: Classify ALL videos → Validate → Save dual outputs
           - For each video:
             * Check manifest validation results
             * Route to Flow 1 (valid) or Flow 2 (invalid)
             * Validate LLM output (format correction)
             * Normalize to 15-field schema
             * Add bucket + duration metadata
             * Save RAW: /content_analysis/raw_llm_output/bucket_{bucket}/{video_id}_raw.json
             * Save FINAL: /content_analysis/validated/bucket_{bucket}/{video_id}_content.json
```

**Output Structure:**
```
/data/clients/{client}/hashtags/{hashtag}/top_contrastive/
└── content_analysis/
    ├── raw_llm_output/              (Raw LLM responses - debugging/auditing)
    │   ├── bucket_33-60s/
    │   │   ├── 7468785508596927786_raw.json
    │   │   └── ... (100 files)
    │   ├── bucket_60-90s/
    │   │   ├── 7469467763740904746_raw.json
    │   │   └── ... (100 files)
    │   └── bucket_90-120s/
    │       └── ... (100 files)
    │
    └── validated/                    (Final validated outputs - production)
        ├── bucket_33-60s/
        │   ├── 7468785508596927786_content.json
        │   └── ... (100 files)
        ├── bucket_60-90s/
        │   ├── 7469467763740904746_content.json
        │   └── ... (100 files)
        └── bucket_90-120s/
            └── ... (100 files)
```

**Output Types:**

**Raw LLM Output** (`raw_llm_output/bucket_{bucket}/{video_id}_raw.json`):
- Direct output from LLM before validation
- Used for debugging prompt issues
- Used for auditing validation corrections
- Format varies (Flow 1: 13 fields, Flow 2: 1 field)

**Validated Output** (`validated/bucket_{bucket}/{video_id}_content.json`):
- Final production output after validation + normalization
- Used by downstream ML models and report generation
- Guaranteed consistent 15-field schema
- All issues corrected by validation layer

---

## Expected Impact

**Before (Old Code v1.0):**
- Taxonomy contaminated with 42% invalid transcripts
- Stage 2.7 validates each video individually (repeated work)
- Single prompt tries to handle both valid/invalid cases
- 50 transcripts sampled randomly (no guaranteed bucket balance)
- All outputs in flat directory structure
- No checkpoint versioning (mixed outputs possible on upgrade)

**After (Upgraded Code v2.0 - DEFAULT):**
- Taxonomy created from 100% valid transcripts
- Validation done once, used everywhere
- Dual flows: clear separation, optimal prompts per case
- Flow 2: 50% cheaper and faster (caption only)
- 60 transcripts with adaptive sampling (20 per bucket target)
- Weak buckets automatically signal lower video quality for that duration
- Outputs organized by bucket for ML model training
- Bucket + duration metadata in each JSON
- Dual outputs: raw LLM responses + validated production files
- Validation layer catches and corrects formatting issues automatically
- **Hard-fail on errors** (no silent fallbacks to old code)
- **Checkpoint versioning** (prevents mixed-mode processing)
- **Explicit legacy override** (USE_LEGACY_FLOW=true for emergency only)

---

## Implementation Summary: Flow Control Strategy

**Key Design Decisions:**

1. **✅ DEFAULT = Upgraded Flow (v2.0)**
   - New code runs by default
   - No environment variables needed for normal operation
   - Production-ready from day one

2. **❌ NO Silent Fallbacks**
   - Upgraded flow fails → Pipeline HALTS with error
   - Errors surface immediately for debugging
   - No masking of issues with automatic degradation

3. **🔧 Legacy Override = Emergency Only**
   - Set `USE_LEGACY_FLOW=true` to use old code
   - Requires explicit manual action
   - Logs prominent warnings
   - Only for emergency rollback scenarios

4. **🔒 Checkpoint Versioning = Hard Fail**
   - v1.0 checkpoint + v2.0 code → HALT
   - v2.0 checkpoint + v2.0 code → Continue
   - Prevents mixed-mode outputs (some flat, some bucket-organized)
   - Forces clean restart or explicit legacy mode

**Implementation Checklist:**

- [ ] Add `USE_LEGACY_FLOW` environment variable check (defaults to 'false')
- [ ] Add checkpoint version field to save_checkpoint() (v2.0)
- [ ] Add checkpoint version validation to load_checkpoint() (hard-fail on v1.0)
- [ ] Add `check_and_clear_old_checkpoints()` function to orchestrator
- [ ] Wrap Stage 2.5.1 in try/except with hard-fail error messages
- [ ] Wrap dual-flow classification in try/except with hard-fail error messages
- [ ] Add deployment checklist verification (USE_LEGACY_FLOW not set)
- [ ] Test all 4 checkpoint scenarios (clean start, v2.0 resume, v1.0 mismatch, legacy mode)
- [ ] Document emergency rollback procedure (USE_LEGACY_FLOW=true)

**Deployment Workflow:**

```bash
# 1. Deploy new code to server
git pull origin main

# 2. Verify environment (no USE_LEGACY_FLOW set)
env | grep USE_LEGACY_FLOW
# Expected: (no output)

# 3. Test on single hashtag first
python rumiai_ml_batch.py --client test --hashtag wellness_small

# 4. Check logs for upgraded flow banner
# Expected:
# 🚀 UPGRADED STAGE 2 PIPELINE (v2.0)
#    - Transcript validation (Stage 2.5.1)
#    - Adaptive sampling (20 per bucket)
#    - Dual-flow classification
#    - Bucket-organized outputs

# 5. Verify outputs in new structure
ls /data/clients/test/hashtags/wellness_small/top_contrastive/content_analysis/
# Expected:
# raw_llm_output/
# validated/

# 6. If issues arise, emergency rollback
export USE_LEGACY_FLOW=true
python rumiai_ml_batch.py --client test --hashtag wellness_small
# Logs will show: ⚠️  LEGACY FLOW ENABLED (manual override)

# 7. After fixes, unset legacy flag and retry
unset USE_LEGACY_FLOW
python rumiai_ml_batch.py --client test --hashtag wellness_small
```

**Error Scenarios & Responses:**

| Error | Current Behavior | Action |
|-------|------------------|--------|
| Stage 2.5.1 validation fails | ❌ HARD FAIL, logs error, raises exception | Fix transcript issues, re-run |
| Dual-flow classification fails | ❌ HARD FAIL, logs error, raises exception | Debug classification logic |
| Bucket directory creation fails | ❌ HARD FAIL, logs error, raises exception | Check disk permissions |
| v1.0 checkpoint detected | ❌ HARD FAIL, prompts user to delete or use legacy | Delete checkpoint or set USE_LEGACY_FLOW=true |
| LLM API timeout (retryable) | ⚠️  Retry 3x with backoff, then hard fail | Wait for retry or investigate API |
| Invalid taxonomy category | ⚠️  Auto-correct with fuzzy match, log warning | Review logs, fix if needed |

This implementation ensures **production stability** while providing **emergency escape hatches** if needed.

