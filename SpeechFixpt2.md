# Speech Coverage Fix Part 2: Non-Speech Tag Filtering

**Document Version**: 1.0
**Date**: 2025-01-30
**Status**: CRITICAL BUG - Immediate Fix Required
**Impact**: All videos with background music or non-speech audio

---

## 1. Bug Outline

### 1.1 The Problem
**Speech_coverage shows 1.0 (100%) for videos with NO actual speech**

#### Current Behavior (WRONG)
```json
// E3NoFace video output
{
  "speech_coverage": 1.0,  // Says 100% speech coverage
  "word_count": 0,         // But zero actual words!
  "energy_level": 0.009    // Has audio energy (music playing)
}
```

#### What's Actually Happening
1. Video has background music, no speech
2. Whisper detects this and outputs: `[Music]`
3. Our code treats `[Music]` as a speech segment
4. Speech coverage = music duration / video duration = 10s/10s = 1.0
5. **Result**: Music is counted as speech!

### 1.2 Scope of Impact

Found these non-speech patterns in production data:
- `[Music]` - Most common, background music
- `[MUSIC]` - Uppercase variant
- `[BLANK_AUDIO]` - Silent sections
- **Sung lyrics with ♪ symbols** - Actual lyrics transcribed but should not count as speech

Potential tags from Whisper documentation:
- `[Applause]`
- `[Laughter]`
- `[Silence]`
- `[Background noise]`
- `[Inaudible]`

**CRITICAL DISCOVERY**: Whisper handles two music scenarios differently:
1. **Instrumental music** → `[Music]` tags (correctly filtered)
2. **Music with sung lyrics** → Normal transcription with ♪ symbols (currently counted as speech!)

### 1.3 Why This Matters

**ML Training Corruption**:
- Model thinks music videos have 100% speech
- Creates false correlation: high speech_coverage ≠ actual talking
- Engagement predictions will be wrong

**Analytics Distortion**:
- "90% of videos have speech" - FALSE, includes music
- Speech-based metrics are meaningless

---

## 2. Proposed Fix

### 2.1 Core Solution
Filter out non-speech segments AND sung lyrics in `calculate_speech_metrics_for_window()`:

```python
def calculate_speech_metrics_for_window(speech_segments, start, end, duration, video_id=None):
    """
    Calculate speech coverage and word count for a temporal window.
    FILTERS OUT non-speech tags like [Music], [Applause], AND sung lyrics with ♪ symbols.
    """

    # No speech is valid - return zeros
    if not speech_segments:
        return 0.0, 0

    def is_non_speech_segment(text):
        """Returns True if text is non-speech (music tags or sung lyrics)"""
        text_stripped = text.strip()

        # Check for bracketed non-speech tags
        NON_SPEECH_TAGS = [
            '[Music]', '[MUSIC]', '[music]',
            '[Applause]', '[APPLAUSE]', '[applause]',
            '[Laughter]', '[LAUGHTER]', '[laughter]',
            '[BLANK_AUDIO]', '[Silence]', '[silence]',
            '[Background noise]', '[Inaudible]', '[inaudible]'
        ]

        if text_stripped in NON_SPEECH_TAGS:
            return True

        # Check for sung lyrics (musical note symbols)
        if '♪' in text or '♫' in text or '🎵' in text or '🎶' in text:
            return True

        # Check if text is ONLY non-speech tags with whitespace
        cleaned_text = text
        for pattern in NON_SPEECH_TAGS:
            cleaned_text = cleaned_text.replace(pattern, '')

        return not cleaned_text.strip()  # True if nothing left after removing tags

    # Filter segments
    actual_speech_segments = []
    for segment in speech_segments:
        text = segment.get('text', '')

        if not is_non_speech_segment(text):
            actual_speech_segments.append(segment)
        else:
            logger.debug(f"Filtering non-speech: {text[:50]}...")

    # NOW calculate with only actual speech
    if not actual_speech_segments:
        return 0.0, 0  # No actual speech

    # Rest of existing calculation logic...
    total_speech_duration = 0.0
    total_word_count = 0

    for segment in actual_speech_segments:
        # ... existing calculation code
```

### 2.2 Alternative: Regex-Based Filtering
More robust but slightly slower:

```python
import re

NON_SPEECH_REGEX = re.compile(r'^\s*(\[[^\]]+\]\s*)+$')

def is_non_speech_segment(text):
    """Returns True if text contains ONLY non-speech tags"""
    return bool(NON_SPEECH_REGEX.match(text))
```

---

## 3. Deep Discovery: Implementation Risks

### 3.1 Immediate Deployment Risks (No Rollback!)

#### Risk 1: Sudden Metric Changes (AMPLIFIED)
**What happens**:
- Yesterday: "80% videos have speech"
- Today (post-fix): "15% videos have speech" (even lower due to sung lyrics filtering)

**Impact**:
- Dashboards show massive drop
- Executives panic about "broken metrics"
- Data team thinks pipeline failed

**Mitigation**:
- Send email NOW: "Speech metrics will correct on [date]"
- Add annotation to all dashboards
- Log the change: `logger.warning("METRIC CHANGE: Filtering non-speech tags as of 2025-01-30")`

#### Risk 2: ML Model Confusion
**What happens**:
- Models trained on old data expect speech_coverage=1.0 for music videos
- New data has speech_coverage=0.0 for same videos
- Model performance degrades immediately

**Impact**:
- Predictions become erratic
- A/B tests show weird results
- Recommendation quality drops

**Mitigation**:
- MUST retrain models within 48 hours
- Consider dual-metric period: `speech_coverage_v1` and `speech_coverage_v2`
- Alert ML team before deployment

#### Risk 3: Hidden Dependencies
**What happens**:
Some code might DEPEND on the bug!

```python
# Someone's hacky workaround that will break:
if speech_coverage == 1.0 and word_count == 0:
    # They figured out this means music!
    is_music_video = True
```

**Impact**:
- Random features break
- Music video detection fails
- Categorization pipelines fail

**Mitigation**:
```bash
# Search for dangerous patterns
grep -r "speech_coverage.*1\.0.*word_count.*0" .
grep -r "word_count.*0.*speech_coverage.*1" .
```

#### Risk 4: Whisper Output Variations
**What happens**:
Different Whisper versions/models use different tags:
- `whisper-tiny`: `[Music]`
- `whisper-large`: `♪♪♪`
- `whisper.cpp`: `[MUSIC]`
- Future version: `<music>`

**Impact**:
- Fix works for some videos, not others
- Inconsistent results across batches
- New Whisper version breaks everything

**Mitigation**:
Make detection MORE aggressive:
```python
def is_likely_non_speech(text):
    # Strip all special chars and check what's left
    import re
    cleaned = re.sub(r'[\[\]()<>♪♫🎵🎶\s]+', '', text)

    NON_SPEECH_WORDS = {'music', 'applause', 'laughter', 'silence',
                        'blank', 'audio', 'inaudible', 'background', 'noise'}

    # If cleaned text is empty or only contains non-speech indicators
    if not cleaned:
        return True

    if cleaned.lower() in NON_SPEECH_WORDS:
        return True

    return False
```

### 3.2 Data Quality Risks

#### Risk 5: Legitimate Speech Edge Cases
**What happens**:
Someone actually SAYS "[Music]" or talks about music:
- "Today we're reviewing [Music] by Madonna" → Filtered incorrectly
- "Type [BLANK_AUDIO] to fix the bug" → Filtered incorrectly
- "This song has ♪ in the title" → Filtered incorrectly
- Someone singing a cappella (no instruments) → May get ♪ symbols and be filtered

**Impact**:
- Real speech gets filtered out
- Word counts become wrong
- Transcription quality appears to drop

**Detection**:
```python
# Check if brackets appear mid-sentence
if len(segment.get('words', [])) > 1:
    # Multiple words = probably real speech
    # Even if one word is "[Music]"
    pass
```

#### Risk 6: Mixed Content Segments
**What happens**:
Segment contains both speech and music tag:
```json
{
  "text": "Hello everyone [Music] welcome back",
  "start": 0.0,
  "end": 5.0
}
```

**Current fix behavior**: Keeps the segment (has real content)
**But**: Duration calculation includes the [Music] portion

**Better approach**:
```python
# Remove tags from text but keep segment
cleaned_text = text
for pattern in NON_SPEECH_PATTERNS:
    cleaned_text = cleaned_text.replace(pattern, '')

if cleaned_text.strip():
    # Use segment but adjust word count
    # Count only non-tag words
    pass
```

### 3.3 Performance Risks

#### Risk 7: Processing Speed Impact
**What happens**:
- Every segment now needs tag checking
- 300 videos × 50 segments = 15,000 checks
- String operations are expensive

**Impact**:
- Pipeline slows by 5-10%
- Timeout errors on long videos
- Batch processing takes longer

**Optimization**:
```python
# Pre-compile patterns ONCE
NON_SPEECH_SET = frozenset(['[Music]', '[MUSIC]', ...])  # O(1) lookup

# Quick check first
if text[0] == '[':  # Only check if starts with bracket
    if text in NON_SPEECH_SET:
        # Filter out
```

### 3.4 Edge Cases from Hell

#### Risk 8: Compound Tags
```
"[Music][Music][Music][Applause][Music]"  # Live performance
"[Laughter] [Inaudible] [Laughter]"       # Comedy show
```

#### Risk 9: Partial Tags
```
"[Mus"          # Corrupted segment
"Music]"        # Missing opening bracket
"[Music"        # Missing closing bracket
```

#### Risk 10: Unicode Variants
```
"【Music】"     # Asian brackets
"〔Music〕"     # Different bracket style
"［ＭＵＳＩＣ］" # Full-width characters
```

---

## 4. Recommended Implementation Strategy

### Given: No rollback, aggressive immediate deployment

1. **Add comprehensive logging** (30 minutes)
   ```python
   logger.warning(f"SPEECH_FIX_V2: Filtered {len(filtered)} non-speech segments from video {video_id}")
   ```

2. **Deploy to shadow mode FIRST** (2 hours)
   - Run both calculations
   - Log differences
   - Verify fix works on real data

3. **Alert all stakeholders** (Before deployment)
   - Email: Data team, ML team, Analytics team
   - Slack: #data-pipeline channel
   - Dashboard annotation

4. **Deploy aggressively** (1 hour window)
   - Push fix
   - Monitor error rates
   - Watch processing times
   - Check metric changes

5. **Post-deployment** (Next 24 hours)
   - Monitor for edge cases
   - Collect filtered segments for analysis
   - Prepare for hot fixes

---

## 5. Success Metrics

After deployment, we should see:

1. **E3NoFace (music only)**:
   - Before: `speech_coverage: 1.0, word_count: 0`
   - After: `speech_coverage: 0.0, word_count: 0` ✓

2. **Normal speech videos**:
   - No change (no tags to filter)

3. **Videos with sung lyrics**:
   - Before: `speech_coverage: 1.0, word_count: 50` (lyrics counted)
   - After: `speech_coverage: 0.0, word_count: 0` ✓

4. **Mixed content videos** (talking + background music):
   - Speech coverage drops to actual talking portions only
   - Sung lyrics no longer counted as speech

5. **Processing performance**:
   - < 5% slowdown acceptable
   - No timeout increases

---

## 6. Conclusion

This fix is **necessary and urgent**. The current bug corrupts:
- ML training data
- Analytics accuracy
- User understanding

Yes, aggressive deployment without rollback is risky. But every day we wait, we're training models on lies and making decisions on false metrics.

The key to success: **COMMUNICATION**. Tell everyone before you deploy. Log everything. Monitor aggressively.

**Recommended action**: Deploy TODAY with shadow mode first (2 hours), then full deployment.

---

## Document History
- v1.0 (2025-01-30): Initial analysis and risk assessment for speech_coverage non-speech tag filtering