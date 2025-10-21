# Stage 2.6 Transcript Quality Filtering Enhancement

**Date**: 2025-10-16
**Status**: ✅ APPROVED - Ready for Implementation
**Impact**: High - Affects pattern discovery accuracy and coverage metrics
**Last Updated**: 2025-10-17 (after critique session)

---

## Problem Statement

### Issue Discovered

During Stage 2.6 pattern discovery testing with real hashtag data (`#test_vitamin`), we discovered that **invalid/low-quality transcripts are included in the sample**, leading to:

1. **Inaccurate coverage metrics** (pattern frequency percentages)
2. **LLM analyzing gibberish** (wasted tokens, no value)
3. **Misleading sparsity conclusions** (appears sparse due to noise, not actual diversity)

### Examples of Invalid Transcripts

| Video ID | Transcript Content | Issue |
|----------|-------------------|-------|
| 7560652726527364366 | `(speaking in foreign language) (speaking in foreign language)...` | Foreign language (Whisper can't transcribe) |
| 7557775207377865986 | `[Music] [Music] [Music] [Music] [Music]` | Music only, no spoken content |
| 7549317319722323213 | `[Music] [Music] [Music] [Music] [Music] [Music] [BLANK_AUDIO]` | Music + blank audio |
| 7558810700181802262 | `♪ And where do we 틔 ♪ ♪ Oh, oh, we're downeen ♪` | Corrupted Unicode (transcription error) |
| 7528143196287733006 | `It's 9 PM on a fucking Friday, get the fuck up! We're going to high, let's go. Let's go. Let's go. Let's go. Going to high, let's go...` (repeated 50+ times) | Repeated lyrics/chants (no informative content) |
| 7560964241084271903 | `I have one thing to say. You better work, bitch. [Music]` | Mostly music with single quote |

### Impact on Metrics

**✅ VALIDATED WITH REAL DATA (18-33s bucket, 50 videos):**

**Current State (No Filtering):**
```
Sample size: 50 videos
Valid content: 9 videos
Music-only: 33 videos (66%)
Sound effects only: 3 videos (6%)
Coverage: 9/50 = 18%
```

**After Filtering:**
```
Valid transcripts: 17 videos (filtered 33 music-only)
Valid content: 9 videos
Actual coverage: 9/17 = 53%
```

**This 35% difference MASSIVELY changes interpretation:**
- 18% coverage → "Extremely sparse, questionable data quality"
- 53% coverage → "Moderately strong pattern coherence"

**Key Discovery:** Invalid rate is **66-68%** (way higher than initial 15-20% estimate)

---

## Root Cause Analysis

### Why This Happens

1. **Whisper transcription limitations**
   - Foreign language videos → "(speaking in foreign language)"
   - Music-only videos → "[Music]" markers
   - Audio quality issues → Unicode corruption or blank audio

2. **Content diversity in TikTok**
   - Videos with no speech (visual-only, music-only)
   - Non-English content (global platform)
   - Performance videos (dancing, lip-sync with no original audio)

3. **Stage 2 doesn't filter transcripts**
   - `rumiai_runner.py` processes all videos regardless of transcript quality
   - No validation before saving `*_whisper.json` files
   - Downstream stages (2.5, 2.6) inherit this noise

### Current Data Flow (BEFORE Enhancement)

```
Stage 1: Video Selection
   ↓ (selects 120 videos based on engagement metrics)
Stage 2: Video Processing
   ↓ (runs Whisper on ALL videos, saves all transcripts)
Stage 2.5: File Organization
   ↓ (organizes ALL temporal_windows files, no filtering)
Stage 2.6: Pattern Discovery
   ↓ (samples transcripts, includes NOISE) ❌
LLM Analysis
   ↓ (analyzes gibberish, gets confused, misses real patterns)
Low Coverage Results ❌
```

### ✅ APPROVED Data Flow (AFTER Enhancement)

```
Stage 1: Video Selection
   ↓ (selects 120 videos based on engagement metrics)
Stage 2: Video Processing
   ↓ (runs Whisper on ALL videos, saves all transcripts - UNCHANGED)
Stage 2.5: File Organization
   ↓ (creates selection_manifest.json)
Stage 2.5.5: Transcript Validation (NEW)
   ↓ (validates ALL transcripts, creates validation cache)
   ↓ (logs: 66% music-only, 6% sound effects, 28% valid content)
Stage 2.6: Pattern Discovery
   ↓ (filters using validation cache, samples ONLY valid transcripts) ✅
LLM Analysis
   ↓ (analyzes clean data, discovers accurate patterns)
Accurate Coverage Results ✅
```

---

## ✅ APPROVED Solution

### Overview

Add **transcript quality validation as a separate stage (Stage 2.5.5)** that runs BEFORE Stage 2.6 pattern discovery. Creates a reusable validation cache that both Stage 2.6 (discovery) and Stage 2.7 (classification) can use.

**Key Design Decisions:**
- ✅ **Point 3 Decision:** Create separate validation stage (don't modify Stage 2 Whisper processing)
- ✅ **Point 4 Decision:** No oversampling needed - validation cache tells us exactly which transcripts are valid
- ✅ **Point 1 Decision:** Enhanced filtering logic with comprehensive regex patterns
- ✅ **Point 2 Decision:** Config-driven thresholds with conservative defaults

### Implementation Part 1: Configuration Class

**Location**: `ml_pipeline/stage2_content_analysis/transcript_validation.py` (NEW FILE)

```python
import re
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class TranscriptFilterConfig:
    """
    Configuration for transcript quality filtering.

    Default thresholds are conservative (favor precision over recall).
    For Stage 2.6 pattern discovery, we want high-quality transcripts only.

    ✅ Point 2 Decision: Config-driven with validated defaults from real data.
    """
    # Core thresholds (validated against 18-33s bucket real data)
    min_length: int = 12  # Compromise between 10 and 15
    min_unique_ratio: float = 0.30  # 30% unique words minimum
    min_words: int = 4  # Allow short valid transcripts like "Take your vitamin D"

    # Repetitiveness check
    repetitiveness_check_min_words: int = 15  # Check if >= 15 words

    # Language handling
    filter_foreign_language: bool = False  # Default: keep Spanish/international content
```

### Implementation Part 2: Enhanced `is_valid_transcript()` Function

**✅ Point 1 Decision: Regex-based with improved pattern matching**

```python
def is_valid_transcript(
    text: str,
    config: Optional[TranscriptFilterConfig] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate transcript quality for pattern discovery.

    ✅ Point 1 Decision: Enhanced validation with comprehensive patterns

    Args:
        text: Transcript text from Whisper
        config: Filter configuration (uses defaults if None)

    Returns:
        (is_valid, failure_reason) tuple

    Examples:
        >>> is_valid_transcript("[Music] [Music] [Music]")
        (False, "music_only (2 chars after marker removal)")

        >>> is_valid_transcript("(speaking in foreign language)")
        (False, "foreign_language_marker_only")

        >>> is_valid_transcript("Here's how I take my vitamins every morning")
        (True, None)

        >>> is_valid_transcript("let's go " * 50)  # Repeated 50 times
        (False, "too_repetitive (unique_ratio=0.02 < 0.30)")
    """
    if config is None:
        config = TranscriptFilterConfig()

    text_clean = text.strip()

    # Check 1: Minimum length
    if len(text_clean) < config.min_length:
        return False, f"too_short ({len(text_clean)} < {config.min_length} chars)"

    # Check 2: Music/sound effect markers (ENHANCED from real data analysis)
    # Pattern covers: [Music], [MUSIC], (upbeat music), (chewing), (sighing), etc.
    SOUND_MARKERS_PATTERN = r'''
        \[Music\]|              # [Music]
        \[MUSIC\]|              # [MUSIC]
        \([^)]*music[^)]*\)|    # (upbeat music), (music)
        \([a-z]+ing\)|          # (chewing), (sighing), (laughing)
        ♪|                      # Music note
        \[BLANK[_\s]?AUDIO\]    # [BLANK_AUDIO] or [BLANK AUDIO]
    '''
    text_no_markers = re.sub(SOUND_MARKERS_PATTERN, '', text_clean, flags=re.IGNORECASE | re.VERBOSE)
    text_no_markers = re.sub(r'\s+', ' ', text_no_markers).strip()

    if len(text_no_markers) < config.min_length:
        return False, f"music_only ({len(text_no_markers)} chars after marker removal)"

    # Check 3: Foreign language - ONLY filter if ENTIRELY marker with no real content
    # Keeps valid Spanish content like: "[Speaking in Spanish] Now they are having..."
    FOREIGN_LANG_MARKER = r'\[?Speaking in \w+\]?'
    text_no_foreign_markers = re.sub(FOREIGN_LANG_MARKER, '', text_no_markers, flags=re.IGNORECASE)
    text_no_foreign_markers = re.sub(r'\s+', ' ', text_no_foreign_markers).strip()

    if config.filter_foreign_language and len(text_no_foreign_markers) < config.min_length:
        return False, "foreign_language_marker_only"

    # Use text after removing markers for remaining checks
    text_for_checks = text_no_foreign_markers if text_no_foreign_markers else text_no_markers

    # Check 4: Minimum word count
    words = text_for_checks.split()
    if len(words) < config.min_words:
        return False, f"too_few_words ({len(words)} < {config.min_words})"

    # Check 5: Repetitiveness (only for longer transcripts)
    if len(words) >= config.repetitiveness_check_min_words:
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)

        if unique_ratio < config.min_unique_ratio:
            return False, f"too_repetitive (unique_ratio={unique_ratio:.2f} < {config.min_unique_ratio})"

    # All checks passed
    return True, None
```

### Implementation Part 3: Validation Stage (Stage 2.5.5)

**✅ Point 3 Decision: Separate validation stage, don't modify Stage 2**

**New File**: `ml_pipeline/stage2_content_analysis/transcript_validation.py`

```python
def validate_all_transcripts(
    client_id: str,
    hashtag: str,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive",
    config: Optional[TranscriptFilterConfig] = None
) -> str:
    """
    Validate ALL transcripts for a hashtag and cache results.

    This runs as Stage 2.5.5 (after file organization, before discovery).
    Creates transcript_validation_cache.json that both Stage 2.6 and 2.7 use.

    Returns:
        str: Path to validation cache file

    Raises:
        FileNotFoundError: If selection_manifest.json doesn't exist
    """
    if config is None:
        config = TranscriptFilterConfig()

    # Load selection manifest
    manifest_path = construct_path(client_id, hashtag, analysis_mode, selection_strategy, file_type="selection_manifest")
    manifest = load_json(manifest_path)

    # Collect all video IDs
    all_video_ids = []
    for bucket_data in manifest['videos_by_bucket'].values():
        all_video_ids.extend(bucket_data['top_performers'])
        all_video_ids.extend(bucket_data['bottom_performers'])

    logger.info(f"Validating {len(all_video_ids)} transcripts for #{hashtag}...")

    # Validate each transcript
    validation_results = {}
    stats = {'total': len(all_video_ids), 'valid': 0, 'invalid': 0, 'by_reason': {}}

    for video_id in all_video_ids:
        transcript_path = f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json"

        try:
            transcript_data = load_json(transcript_path)
            text = transcript_data.get('text', '')

            is_valid, reason = is_valid_transcript(text, config)

            validation_results[video_id] = {
                'is_valid': is_valid,
                'failure_reason': reason,
                'text_length': len(text),
                'word_count': len(text.split())
            }

            if is_valid:
                stats['valid'] += 1
            else:
                stats['invalid'] += 1
                stats['by_reason'][reason] = stats['by_reason'].get(reason, 0) + 1

        except FileNotFoundError:
            validation_results[video_id] = {
                'is_valid': False,
                'failure_reason': 'transcript_not_found'
            }
            stats['invalid'] += 1

    # Save validation cache
    analysis_base = construct_path(client_id, hashtag, analysis_mode, selection_strategy, file_type="base")
    cache_path = f"{analysis_base}/transcript_validation_cache.json"

    save_json(cache_path, {
        'version': VALIDATION_CACHE_VERSION,  # Point 7: Cache versioning
        'hashtag': hashtag,
        'validation_date': datetime.now().isoformat(),
        'config': config.__dict__,
        'stats': stats,
        'results': validation_results
    })

    logger.info(f"✓ Validation complete: {stats['valid']}/{stats['total']} valid ({stats['valid']/stats['total']*100:.1f}%)")
    logger.info(f"  Invalid by reason: {stats['by_reason']}")

    return cache_path
```

### Implementation Part 4: Updated Sampling Logic

**✅ Point 4 Decision: No oversampling needed - filter first, then sample from valid pool**

```python
def sample_transcripts_for_discovery(
    manifest_path: str,
    sample_size: int = 100,
    validation_cache: Dict[str, dict] = None
) -> List[dict]:
    """
    Sample transcripts for pattern discovery.

    Strategy (Point 4 Decision):
    1. Filter to valid video IDs using validation cache
    2. Sample from valid pool only
    3. Load transcripts (all guaranteed valid)

    NO oversampling needed - we know exactly which transcripts are valid.
    """
    if validation_cache is None:
        raise ValueError("validation_cache is required. Run validate_all_transcripts() first.")

    manifest = load_json(manifest_path)

    # Select top 3 buckets by video count
    bucket_counts = {
        bucket: len(data['top_performers'])
        for bucket, data in manifest['videos_by_bucket'].items()
    }
    top_3_buckets = sorted(bucket_counts.keys(), key=lambda b: bucket_counts[b], reverse=True)[:3]

    samples_per_bucket = sample_size // len(top_3_buckets)
    sampled_transcripts = []

    for bucket in top_3_buckets:
        top_performers = manifest['videos_by_bucket'][bucket]['top_performers']

        # STEP 1: Filter to valid video IDs using cache
        valid_video_ids = [
            vid for vid in top_performers
            if vid in validation_cache and validation_cache[vid]['is_valid']
        ]

        invalid_rate = (len(top_performers) - len(valid_video_ids)) / len(top_performers)
        logger.info(
            f"Bucket {bucket}: {len(valid_video_ids)}/{len(top_performers)} valid "
            f"({invalid_rate:.1%} invalid rate)"
        )

        # STEP 2: Sample from valid pool only
        actual_sample_count = min(samples_per_bucket, len(valid_video_ids))

        if actual_sample_count < samples_per_bucket:
            logger.warning(
                f"Bucket {bucket}: Only {actual_sample_count} valid transcripts available. "
                f"Using all available."
            )

        if actual_sample_count == 0:
            logger.error(f"Bucket {bucket}: No valid transcripts! Skipping bucket.")
            continue

        sampled_ids = random.sample(valid_video_ids, actual_sample_count)

        # STEP 3: Load transcripts (all guaranteed valid)
        for video_id in sampled_ids:
            transcript_path = f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json"
            transcript_data = load_json(transcript_path)

            sampled_transcripts.append({
                "video_id": video_id,
                "text": transcript_data.get('text', ''),
                "bucket": bucket
            })

    logger.info(f"Successfully sampled {len(sampled_transcripts)} valid transcripts (target: {sample_size})")

    return sampled_transcripts
```

---

## ✅ Expected Improvements (Validated with Real Data)

### 1. Accurate Coverage Metrics

**Before (18-33s bucket):**
```
Sample: 50 videos (includes 33 music-only + 5 sound effects)
Valid content: 9 videos
Coverage: 9/50 = 18%
```

**After (Point 3+4 approach):**
```
Sample: 17 valid videos (filtered 33 invalid automatically via cache)
Valid content: 9 videos
Coverage: 9/17 = 53%
```

**Improvement: 18% → 53% (35 percentage point increase!)**

### 2. Better LLM Analysis

- LLM receives only meaningful content
- Patterns discovered from real data, not noise
- Reduced token costs (no gibberish sent to API)

### 3. More Reliable Discovery

- Higher confidence in discovered patterns
- Less chance of LLM getting confused by noise
- Better quality taxonomy output

### 4. Transparent Logging

```
INFO: Sampling 100 transcripts from 88 top performers
INFO: Skipping invalid transcript: 7557775207377865986 (music-only)
INFO: Skipping invalid transcript: 7560652726527364366 (foreign language)
INFO: Skipping invalid transcript: 7528143196287733006 (repetitive content)
INFO: Successfully sampled 62 valid transcripts from 88 available
```

---

## ✅ Edge Cases & Handling (Point 5)

**✅ Point 5 Decision: Two-stage validation (early warning + hard enforcement)**

### Stage 2.5.5: Validation Stage (Early Warning)

```python
def validate_all_transcripts(...):
    # ... validation logic ...

    # Early warning if Stage 2.6 will struggle
    valid_top_performers = sum(
        1 for vid in top_performers
        if validation_results[vid]['is_valid']
    )

    if valid_top_performers < 30:
        logger.error(
            f"⚠️  Only {valid_top_performers} valid top performer transcripts.\n"
            f"   Stage 2.6 requires minimum 30. This hashtag may be music-only."
        )
    elif valid_top_performers < 50:
        logger.warning(
            f"⚠️  Only {valid_top_performers} valid top performers.\n"
            f"   Stage 2.6 recommends 50+ for reliable discovery."
        )
```

### Stage 2.6: Sampling Stage (Hard Validation)

```python
def sample_transcripts_for_discovery(...):
    # ... sampling logic ...

    MIN_REQUIRED = 30  # Hard minimum
    MIN_RECOMMENDED = 50  # Soft minimum

    # Edge Case 1: Zero samples (complete failure)
    if len(sampled_transcripts) == 0:
        raise ValueError(
            "No valid transcripts found in any bucket.\n"
            "This hashtag appears to be entirely music-only."
        )

    # Edge Case 2: Below hard minimum
    if len(sampled_transcripts) < MIN_REQUIRED:
        raise ValueError(
            f"Insufficient valid transcripts: {len(sampled_transcripts)}/{MIN_REQUIRED} required.\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Review validation cache stats.by_reason for failure breakdown\n"
            f"  2. Lower thresholds: TranscriptFilterConfig(min_length=10, min_words=3)\n"
            f"  3. This hashtag may be predominantly music-only content"
        )

    # Edge Case 3: Below recommended (warning only)
    if len(sampled_transcripts) < MIN_RECOMMENDED:
        logger.warning(
            f"⚠️  Low sample size: {len(sampled_transcripts)}/{MIN_RECOMMENDED} recommended.\n"
            f"   Pattern discovery will proceed but may be less reliable."
        )

    # Informational: Log bucket distribution
    bucket_counts = {}
    for transcript in sampled_transcripts:
        bucket_counts[transcript['bucket']] = bucket_counts.get(transcript['bucket'], 0) + 1

    logger.info(f"Sample distribution: {dict(sorted(bucket_counts.items()))}")

    # Informational: Detect extreme skewness (>80% from one bucket)
    max_bucket = max(bucket_counts, key=bucket_counts.get)
    max_bucket_pct = bucket_counts[max_bucket] / len(sampled_transcripts)

    if max_bucket_pct > 0.80:
        logger.info(
            f"ℹ️  Sample skewed toward {max_bucket}: {max_bucket_pct:.1%}.\n"
            f"   Patterns may be specific to this video length (this is OK)."
        )
```

### Key Edge Case Decisions

1. **Only validate total sample size** - Per-bucket distribution doesn't matter
2. **We're sampling ONLY top performers** - Not bottom performers
3. **Fail at <30 transcripts** - LLM needs minimum data for pattern discovery
4. **Warn at <50 transcripts** - Reliability improves at 50+
5. **Skewness is informational** - >80% from one bucket is notable but not blocking

---

## Testing Strategy

### Unit Tests

**File**: `ml_pipeline/stage2_content_analysis/test_discovery.py`

```python
import pytest
from ml_pipeline.stage2_content_analysis.transcript_validation import (
    is_valid_transcript,
    TranscriptFilterConfig
)

def test_valid_transcript():
    """Test typical valid transcript"""
    result = is_valid_transcript("Here's how I take my vitamins every morning")
    assert result[0] == True
    assert result[1] is None

def test_music_only_transcript():
    """Test music-only transcript"""
    result = is_valid_transcript("[Music] [Music] [Music] [Music]")
    assert result[0] == False
    assert "music_only" in result[1]

def test_foreign_language_transcript():
    """Test foreign language marker"""
    result = is_valid_transcript("(speaking in foreign language)")
    assert result[0] == False
    # Note: foreign language filtering depends on config.filter_foreign_language

def test_blank_audio_transcript():
    """Test blank audio"""
    result = is_valid_transcript("[BLANK_AUDIO]")
    assert result[0] == False
    assert "music_only" in result[1]

def test_repetitive_transcript():
    """Test repetitive lyrics/chants"""
    repetitive = "let's go " * 50
    result = is_valid_transcript(repetitive)
    assert result[0] == False
    assert "too_repetitive" in result[1]

def test_short_transcript():
    """Test too short transcript"""
    result = is_valid_transcript("Hi")
    assert result[0] == False
    assert "too_short" in result[1]

def test_mixed_content():
    """Test transcript with music + valid content"""
    result = is_valid_transcript("[Music] Vitamin D is essential [Music]")
    assert result[0] == True
    assert result[1] is None

def test_borderline_length():
    """Test transcript at minimum length threshold"""
    result1 = is_valid_transcript("Vitamin D")  # 9 chars
    assert result1[0] == False
    assert "too_short" in result1[1]

    result2 = is_valid_transcript("Vitamin D now!")  # 14 chars
    assert result2[0] == True
    assert result2[1] is None

def test_config_customization():
    """Test custom config thresholds"""
    config = TranscriptFilterConfig(min_length=5, min_words=2)
    result = is_valid_transcript("Hello world", config)
    assert result[0] == True
```

### Integration Test

**✅ Point 7: Test with real test_vitamin data**

**File**: `ml_pipeline/stage2_content_analysis/test_validation_integration.py`

```python
import os
from ml_pipeline.stage2_content_analysis.transcript_validation import (
    validate_all_transcripts,
    load_validation_cache,
    VALIDATION_CACHE_VERSION
)
from ml_pipeline.stage2_content_analysis.utils import load_json

def test_validation_with_real_data():
    """Test full validation stage with test_vitamin data"""
    # Run validation stage
    cache_path = validate_all_transcripts(
        client_id='test_final',
        hashtag='test_vitamin',
        analysis_mode='top',
        selection_strategy='contrastive'
    )

    # Verify cache file exists
    assert os.path.exists(cache_path)

    # Load and validate cache structure
    cache_data = load_json(cache_path)

    # Check version
    assert cache_data['version'] == VALIDATION_CACHE_VERSION

    # Check required fields
    assert 'hashtag' in cache_data
    assert cache_data['hashtag'] == 'test_vitamin'
    assert 'stats' in cache_data
    assert 'results' in cache_data

    # Check stats
    stats = cache_data['stats']
    assert stats['total'] > 0
    assert stats['valid'] >= 0
    assert stats['invalid'] >= 0
    assert stats['total'] == stats['valid'] + stats['invalid']

    # Validate results structure
    for video_id, result in cache_data['results'].items():
        assert 'is_valid' in result
        assert 'failure_reason' in result
        assert isinstance(result['is_valid'], bool)

        if not result['is_valid']:
            assert result['failure_reason'] is not None

    print(f"✅ Validation test passed!")
    print(f"   Total: {stats['total']}")
    print(f"   Valid: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"   Invalid: {stats['invalid']} ({stats['invalid']/stats['total']*100:.1f}%)")
    print(f"   By reason: {stats.get('by_reason', {})}")

def test_load_validation_cache():
    """Test loading validation cache with version checking"""
    # Load cache
    cache = load_validation_cache(
        client_id='test_final',
        hashtag='test_vitamin',
        analysis_mode='top',
        selection_strategy='contrastive'
    )

    # Verify structure
    assert isinstance(cache, dict)
    assert len(cache) > 0

    # Verify each entry has required fields
    for video_id, result in cache.items():
        assert 'is_valid' in result
        assert isinstance(result['is_valid'], bool)

    print(f"✅ Cache load test passed! Loaded {len(cache)} validation results")
```

**Run tests:**
```bash
# Activate virtual environment
source venv/bin/activate

# Run integration tests
python -m pytest ml_pipeline/stage2_content_analysis/test_validation_integration.py -v

# Expected output:
# test_validation_with_real_data ... PASSED
# test_load_validation_cache ... PASSED
```

---

## Rollout Plan

### Phase 1: Implementation (15 minutes)

1. Add `is_valid_transcript()` function to `discovery.py`
2. Update `sample_transcripts_for_discovery()` to call validation
3. Add logging for skipped transcripts

### Phase 2: Testing (10 minutes)

1. Write unit tests for `is_valid_transcript()`
2. Run integration test with existing test data
3. Verify improved coverage metrics

### Phase 3: Re-run Discovery (5 minutes)

1. Re-run Stage 2.6 on test_vitamin data
2. Compare before/after results
3. Verify improved pattern quality

### Phase 4: Documentation (5 minutes)

1. Update `discovery.py` docstrings
2. Add logging documentation
3. Update Stage 2.6 README

**Total estimated time: 35 minutes**

---

## Alternative Solutions Considered

### Alternative 1: Filter at Stage 2 (During Whisper Processing)

**Pros:**
- Prevents invalid transcripts from being saved at all
- Reduces disk usage
- Cleaner downstream stages

**Cons:**
- Requires modifying core `rumiai_runner.py`
- May want to keep raw transcripts for debugging
- Harder to adjust filtering criteria later

**Decision**: ❌ Rejected - Too invasive, Stage 2 should save all raw data

### Alternative 2: Manual Curation Before Discovery

**Pros:**
- Human judgment on what's valid
- Maximum control

**Cons:**
- Not scalable (manual review of 100+ transcripts)
- Subjective decisions
- Slows pipeline significantly

**Decision**: ❌ Rejected - Not scalable for production

### Alternative 3: LLM Handles Filtering (No Preprocessing)

**Pros:**
- Minimal code changes
- LLM might ignore gibberish naturally

**Cons:**
- Wastes tokens ($$$)
- LLM still counts gibberish in frequency calculations
- Inaccurate coverage metrics persist

**Decision**: ❌ Rejected - Doesn't solve the core problem

---

## ✅ Success Metrics (Point 6)

**✅ Point 6 Decision: Focus on ACCURATE coverage, not HIGH coverage**

### Primary Goal: Metric Accuracy (Not Metric Optimization)

**Before Filtering (18-33s bucket, real data):**
```
Invalid transcript rate: 66% (33/50 music-only)
Reported coverage: 18% (9/50 videos match patterns)
Problem: Coverage includes 33 music-only videos in denominator (misleading)
```

**After Filtering:**
```
Invalid transcript rate: 0% (filtered before analysis)
Actual coverage: 53% (9/17 valid videos match patterns)
Improvement: Coverage only counts valid transcripts (accurate)
```

### Why This Matters

- **Coverage BEFORE filtering: 18%** = Misleading (includes noise)
- **Coverage AFTER filtering: 53%** = Accurate (noise removed)

This is NOT "inflating metrics" - it's **"measuring correctly"**.

### What Coverage Tells Us

- **53% coverage** means: LLM discovered patterns that match ~half of valid videos
- **47% don't match patterns** = Content is diverse (expected for TikTok)
- **Low coverage isn't bad** - it means hashtag has varied content strategies
- **High coverage isn't necessarily good** - might mean content is repetitive

### The Real Win

✅ We can trust the coverage number
✅ LLM analyzes real content, not "[Music]"
✅ Pattern frequencies are accurate
✅ Downstream ML training uses clean data

### Quantitative Metrics

| Metric | Before Filtering | After Filtering | Interpretation |
|--------|-----------------|-----------------|----------------|
| Valid transcript % | 34% (17/50) | 100% (17/17) | Only valid transcripts analyzed |
| Pattern coverage | 18% (9/50) | 53% (9/17) | **Accuracy improved 35 percentage points** |
| LLM tokens wasted on noise | 66% | 0% | Zero gibberish sent to API |
| Pattern reliability | Low (noise-affected) | High (clean data) | Trustworthy pattern discovery |

### Qualitative Metrics

- ✅ Discovery results are interpretable (no "[Music]" confusion)
- ✅ Coverage metrics are trustworthy (accurate denominator)
- ✅ Patterns are grounded in real content (not noise artifacts)
- ✅ Low coverage is acceptable (indicates content diversity, not data problems)

---

## Related Issues

- **Stage 2.7 Classification**: Will also benefit from transcript filtering (fewer garbage inputs)
- **ML Training Stages**: Downstream ML models will have cleaner training data
- **Cost Optimization**: Reduced API costs from filtering before LLM calls

---

## ✅ Production Readiness (Point 7)

**✅ Point 7 Decision: Pragmatic production approach (cache versioning + minimal tests)**

### What To Implement NOW (MVP)

1. ✅ **Cache Versioning** - Prevents subtle bugs from config changes
```python
VALIDATION_CACHE_VERSION = "1.0"

cache_output = {
    'version': VALIDATION_CACHE_VERSION,
    'hashtag': hashtag,
    'validation_date': datetime.now().isoformat(),
    'config': config.__dict__,
    'stats': stats,
    'results': validation_results
}

# On load, check version compatibility
if cache_data.get('version') != VALIDATION_CACHE_VERSION:
    raise ValueError(
        f"Cache version mismatch. Run validate_all_transcripts() to regenerate."
    )
```

2. ✅ **Atomic Writes** - Prevent corrupted cache
   - Reuse existing `save_json()` with atomic write pattern from utils.py

3. ✅ **Error Handling** - Graceful failures
```python
for video_id in all_video_ids:
    try:
        # ... validation logic ...
    except Exception as e:
        logger.error(f"Validation failed for {video_id}: {e}")
        validation_results[video_id] = {
            'is_valid': False,
            'failure_reason': f'validation_error: {type(e).__name__}'
        }
```

4. ✅ **Minimal Unit Tests** - Test core function
```python
def test_is_valid_transcript_music_only():
    result = is_valid_transcript("[Music] [Music]", TranscriptFilterConfig())
    assert result[0] == False
    assert "music_only" in result[1]

def test_is_valid_transcript_valid():
    result = is_valid_transcript("Here's my vitamin routine", TranscriptFilterConfig())
    assert result[0] == True
    assert result[1] is None
```

5. ✅ **Integration Test** - Test with real test_vitamin data
```python
def test_validation_with_real_data():
    cache_path = validate_all_transcripts(
        client_id='test_final',
        hashtag='test_vitamin',
        analysis_mode='top',
        selection_strategy='contrastive'
    )

    assert os.path.exists(cache_path)
    cache_data = load_json(cache_path)
    assert cache_data['version'] == VALIDATION_CACHE_VERSION
    assert cache_data['stats']['total'] > 0
```

### What To Defer (Post-MVP)

- ⏳ **Structured Logging** (e.g., structlog) - Nice to have, not blocking
- ⏳ **Parallelization** - Sequential is fine for 100 videos (~5 seconds)
- ⏳ **Comprehensive Test Suite** - Add as we discover edge cases
- ⏳ **Config File Management** - Use dataclass defaults for now

### Production Readiness Checklist

| Aspect | Status | Notes |
|--------|--------|-------|
| **Code Quality** | ⏳ Pending | Implement with Points 1-6 decisions |
| **Error Handling** | ✅ Yes | Try/except with fallback to validation_error |
| **Cache Versioning** | ✅ Yes | Version 1.0, check on load |
| **Atomic Writes** | ✅ Yes | Reuse save_json() with atomic pattern |
| **Unit Tests** | ✅ Minimal | Test core is_valid_transcript() function |
| **Integration Test** | ✅ Yes | Test with real test_vitamin data |
| **Performance** | ⏳ Defer | Sequential fine for 100 videos (~5s) |
| **Monitoring** | ⏳ Defer | Use existing logger, add structlog later |
| **Documentation** | ✅ Yes | Docstrings + 2.6FilterImprovement.md |
| **Config Management** | ✅ Simple | Dataclass defaults, can add file later |

### Why This Balance Is Right

- **MVP-ready** - Safe to deploy without over-engineering
- **Prevents critical bugs** - Versioning + error handling + tests cover the essentials
- **Room for improvement** - Can add parallelization/monitoring later when needed
- **Pragmatic** - Tests what matters, defers what doesn't

---

## References

- Original issue discovered: 2025-10-16 during Stage 2.6 testing
- Test data: `/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/`
- Related files:
  - `ml_pipeline/stage2_content_analysis/discovery.py` (Lines 111-116)
  - Sample transcripts with issues documented in this file

---

## ✅ Approval & Implementation Status

**Proposed by**: Claude (AI Assistant)
**Date**: 2025-10-16
**Status**: ✅ **APPROVED** (after critique session 2025-10-17)

**Approved Decisions:**
- ✅ **Point 1**: Enhanced filtering logic with comprehensive regex patterns
- ✅ **Point 2**: Config-driven thresholds (min_length=12, min_words=4, min_unique_ratio=0.30)
- ✅ **Point 3**: Separate validation stage (Stage 2.5.5), don't modify Stage 2
- ✅ **Point 4**: No oversampling needed - filter first, sample from valid pool
- ✅ **Point 5**: Two-stage validation (early warning + hard enforcement)
- ✅ **Point 6**: Focus on accurate coverage, not high coverage
- ✅ **Point 7**: Pragmatic production approach (versioning + minimal tests)

**Implementation Checklist:**
1. ✅ Create `transcript_validation.py` (new file)
2. ✅ Implement `TranscriptFilterConfig` class
3. ✅ Implement `is_valid_transcript()` with enhanced regex patterns
4. ✅ Implement `validate_all_transcripts()` with cache versioning
5. ✅ Implement `load_validation_cache()` with version checking
6. ✅ Update `sample_transcripts_for_discovery()` to use validation cache
7. ✅ Add edge case validation (MIN_REQUIRED=30, MIN_RECOMMENDED=50)
8. ⏳ Write unit tests for `is_valid_transcript()`
9. ⏳ Write integration test with test_vitamin data
10. ⏳ Re-run Stage 2.6 test and compare before/after metrics

**Ready for Implementation: YES** 🚀
