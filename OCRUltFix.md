# OCR Ultimate Fix - Handling Persistent OCR Errors

**Status**: Proposed
**Date**: 2025-10-02
**Context**: Troubleshooting segment_3 showing 3 overlays when it should show 0

---

## Problem

After implementing the timestamp fix (using `'start'` field instead of `'timestamp'`), segment_3 still shows 3 overlays:

1. **"weight loss is really just"** - 0.11 overlap
2. **"gvm"** - 0.11 overlap
3. **"sayoh"** - 0.10 overlap

Analysis reveals:
- **"gvm"** is an OCR error for **"gym"** (v→y character substitution)
- **"sayoh"** is an OCR error for **"say oh"** (word merging - missing space)
- Both should be classified as **captions** (they match speech perfectly if OCR was correct)
- Current fuzzy matching threshold (0.7) is too strict for these heavy OCR errors

## Current OCR Error Handling

**Location**: `temporal_compute.py:1045-1053`

```python
def fix_ocr_errors(text: str) -> str:
    """Fix common OCR errors that break fuzzy matching."""
    if not text:
        return text
    # Split merged words (e.g., "formeI" → "forme I")
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)
    # Fix pipe symbols (e.g., "modeland|" → "modeland ")
    text = re.sub(r'(\w)\|', r'\1 ', text)
    return text
```

**Limitations**:
- Only handles capital letter splits
- Only handles pipe symbols
- Doesn't catch character substitutions (v→y)
- Doesn't catch lowercase word merging (sayoh)

---

## Proposed Solutions

### Option 1: Character Substitution Mapping (Simple)

**Approach**: Maintain a dictionary of known OCR errors

```python
def fix_ocr_errors(text: str) -> str:
    """Fix common OCR errors that break fuzzy matching."""
    if not text:
        return text

    # Specific known OCR errors (exact matches)
    text_lower = text.lower()
    ocr_fixes = {
        'gvm': 'gym',
        'sayoh': 'say oh',
        # Add more as discovered
    }
    if text_lower in ocr_fixes:
        return ocr_fixes[text_lower]

    # Pattern-based fixes
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)  # Split on capitals
    text = re.sub(r'(\w)\|', r'\1 ', text)          # Fix pipe symbols

    return text
```

**Pros**:
- Simple and fast
- Precise - no false positives
- Easy to add new fixes as discovered

**Cons**:
- Brittle - only fixes known errors
- Doesn't scale to unknown OCR errors
- Requires manual maintenance

---

### Option 2: Edit Distance Fuzzy Matching (Robust)

**Approach**: Use Levenshtein distance for short words

```python
def calculate_fuzzy_similarity(text1, text2, max_edit_distance=2):
    """
    Calculate similarity allowing for OCR errors.
    Uses both character similarity AND edit distance.
    """
    # Character-level similarity (current approach)
    char_sim = SequenceMatcher(None, text1, text2).ratio()

    # Edit distance check for short words (OCR errors)
    if len(text1) <= 4:  # Short words like "gym", "gvm"
        import Levenshtein  # Requires: pip install python-Levenshtein
        edit_dist = Levenshtein.distance(text1, text2)
        if edit_dist <= 1:  # 1 character difference
            return 0.8  # High similarity for close matches

    return char_sim
```

**Pros**:
- Handles unknown OCR errors
- Scales to new errors automatically
- Good for single-character substitutions

**Cons**:
- Requires external library (`python-Levenshtein`)
- More computationally expensive
- May cause false positives

---

### Option 3: Phonetic Matching (Speech-Specific)

**Approach**: Use phonetic algorithms for speech-to-caption matching

```python
import jellyfish  # Requires: pip install jellyfish

def are_phonetically_similar(text1, text2):
    """Check if words sound similar (good for speech captions)"""
    return jellyfish.metaphone(text1) == jellyfish.metaphone(text2)
```

**Pros**:
- Perfect for speech-to-text scenarios
- Handles homophones

**Cons**:
- Doesn't help with visual OCR errors (v→y)
- Requires external library
- "gym" and "gvm" don't sound the same

---

### Option 4: Lower Threshold (Simplest)

**Approach**: Accept lower overlap scores as captions

```python
# Change from:
LOW_SPEECH_THRESHOLD = 0.3   # <30% = definitely overlay

# To:
LOW_SPEECH_THRESHOLD = 0.15  # <15% = definitely overlay
```

**Example impact**:
- "gvm" (0.11) → Still overlay ❌
- Need to go even lower (0.10) to catch these

**Pros**:
- No code changes
- Immediate fix

**Cons**:
- May cause false positives (real overlays → captions)
- Doesn't address root cause
- Brittle solution

---

## Recommended Solution

**Hybrid Approach**: Option 1 (specific fixes) + Enhanced word-level matching

### Part 1: Expand `fix_ocr_errors`

```python
def fix_ocr_errors(text: str) -> str:
    """Fix common OCR errors that break fuzzy matching."""
    if not text:
        return text

    # Specific known OCR errors (exact matches)
    text_lower = text.lower()
    ocr_fixes = {
        'gvm': 'gym',
        'sayoh': 'say oh',
        # Add more as discovered through testing
    }
    if text_lower in ocr_fixes:
        return ocr_fixes[text_lower]

    # Pattern-based fixes
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)  # Split on capitals
    text = re.sub(r'(\w)\|', r'\1 ', text)          # Fix pipe symbols

    return text
```

### Part 2: Relax threshold for short words

In `calculate_speech_overlap`, after calculating overlap:

```python
# Use the higher of character similarity or word overlap
overlap_ratio = max(char_similarity, word_overlap_ratio)

# Special handling for single short words (likely OCR errors)
text_words = set(text_normalized.split())
if len(text_words) == 1:
    single_word = list(text_words)[0]
    if len(single_word) <= 4:  # Short word
        # "gvm" vs "gym" gets 0.66 char similarity
        if char_similarity > 0.5:  # Relaxed threshold
            overlap_ratio = max(overlap_ratio, 0.7)  # Boost to caption threshold

return overlap_ratio
```

**Rationale**:
- Known errors get exact fixes (fast path)
- Unknown short words get relaxed matching (safety net)
- Longer texts maintain strict matching (avoid false positives)
- Scalable: add new fixes as discovered

---

## Testing

**Before fix**:
- "gvm" → 0.11 overlap → OVERLAY ❌
- "sayoh" → 0.10 overlap → OVERLAY ❌

**After fix**:
- "gvm" → exact match → "gym" → 1.00 overlap → CAPTION ✅
- "sayoh" → exact match → "say oh" → word match → 1.00 overlap → CAPTION ✅

**Edge case protection**:
- Unknown 3-letter OCR error → 0.66 char similarity → boosted to 0.7 → CAPTION ✅
- Real 3-letter overlay → 0.20 char similarity → stays 0.20 → OVERLAY ✅

---

## Implementation

1. Update `fix_ocr_errors` function in `temporal_compute.py:1045`
2. Add OCR fixes dictionary with known errors
3. Add short-word handling in `calculate_speech_overlap`
4. Test with segment_3 (should show 0-1 overlays instead of 3)
5. Monitor for new OCR errors and add to dictionary

---

## Open Questions

1. **"weight loss is really just"** - Appears after speech ends. Is this a caption or thematic overlay?
   - Text: 24.67-27.27s
   - Speech: 21.40-24.68s
   - Overlap: only 0.01s
   - **Raincheck for now**

2. Should we look backwards in time to match text with earlier speech?
3. What's the acceptable time gap between speech ending and caption appearing?

---

## Related Documents

- `SpeechxCaption.md` - Original fix for caption detection
- `temporal_compute.py:1045-1053` - Current OCR fix implementation
- `temporal_compute.py:1141-1147` - Speech overlap calculation
