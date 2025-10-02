# Overlay Detection Problem Analysis

**Date**: 2025-10-02
**Context**: Attempting to reliably distinguish between captions (speech transcription) and overlays (graphical text elements) in TikTok videos

---

## The Goal

Classify detected text into two categories:
1. **Captions**: Bottom subtitle-style text that transcribes speech
2. **Overlays**: Graphical text elements (titles, labels, emphasis text, etc.)

---

## Current Approach

**Logic:**
```
IF OCR_text matches Whisper_transcript:
    → It's a caption (transcribing speech)
ELSE:
    → It's an overlay (graphical element)
```

**Implementation:**
- Extract text via EasyOCR (timeline-based)
- Extract speech via Whisper
- Calculate speech overlap using fuzzy matching
- High overlap (>0.7) → Caption
- Low overlap (<0.3) → Overlay
- Uncertain (0.3-0.7) → Use persistence as tiebreaker

---

## The Four Barriers

### Barrier 1: OCR Copies Captions Incorrectly

**Problem:** EasyOCR produces errors that break fuzzy matching with Whisper transcripts.

**Examples:**
| OCR Output | Actual Text | Error Type |
|------------|-------------|------------|
| "gvm" | "gym" | Character substitution (v→y) |
| "sayoh" | "say oh" | Missing space |
| "thisagain:" | "this again" | Missing space + punctuation |
| "talking hereg" | "talking here" | Extra character |
| "you knowz but; uhz" | "you know but uh" | Multiple character errors + punctuation |
| "knoWg" | "know" | Case + extra character |
| "So we'regonna trV filming" | "So we're gonna try filming" | Missing space + character error |

**Impact:**
- Even when caption DOES match speech, OCR errors break the match
- Results in false overlay classification
- Requires extensive error dictionary or better OCR model

**Current Mitigation:**
- Dictionary fixes: `{'gvm': 'gym', 'sayoh': 'say oh'}`
- Edit distance fallback (Levenshtein) for ≤4 letter words
- Pattern-based fixes (camelCase splitting, pipe removal)

**Limitation:**
- Dictionary requires manual updates for each new error discovered
- Edit distance only helps with small words
- Cannot fix all error types (e.g., "you knowz but; uhz" has distance=5 from "you know but uh")

---

### Barrier 2: OCR Text ≠ Whisper Text Granularity

**Problem:** OCR captures partial text visible in frame, Whisper captures complete sentences.

**Example:**
- **OCR**: "So we're gonna try filming"
- **Whisper**: "So we're gonna try filming this again. We have one team here with this. Here's"

**Why This Happens:**
- OCR detects text as it appears/disappears frame-by-frame
- Captions show 3-5 words at a time for readability
- Whisper transcribes complete thoughts/sentences
- Single Whisper segment might span multiple OCR detections

**Impact:**
- Perfect OCR would still not match Whisper exactly
- Fuzzy matching helps but requires aggressive thresholds
- Aggressive thresholds increase false positives

**Example of Partial Matching:**
```
Whisper segment: "This is very good you know the mix of fruits and Nutella tastes amazing"

OCR detections:
- 21s: "This is very good"          → Partial match ✓
- 22s: "this is very good. You"     → Partial match ✓
- 23s: "good"                        → Partial match ✓
- 24s: "the mix of uh, fruitsand"   → Partial match ✓ (with OCR error)
- 25s: "Nutella tastes amazing"     → Partial match ✓
```

**No Clean Solution:** This is a fundamental difference in how OCR vs speech-to-text operate.

---

### Barrier 3: Caption/Speech Timing Misalignment

**Problem:** Text persistence across frames + segment boundaries create timing issues.

**Example from video 595997271203511:**
- Speech: "So we're gonna try filming this again" (0.0s - 3.44s)
- Caption appears at: 0.5s, 0.67s, 1.0s, 2.0s, **3.0s**
- Hook segment ends at 3.0s, segment_1 starts at 3.0s

**What Happens:**
```
Caption at 0.5s  → Speech active (0.0-3.44s) → overlap=0.75 → CAPTION ✓
Caption at 0.67s → Speech active (0.0-3.44s) → overlap=1.00 → CAPTION ✓
Caption at 1.0s  → Speech active (0.0-3.44s) → overlap=1.00 → CAPTION ✓
Caption at 2.0s  → Speech active (0.0-3.44s) → overlap=1.00 → CAPTION ✓
Caption at 3.0s  → Segment boundary          → overlap=0.00 → OVERLAY ❌
```

**Why 3.0s Gets 0.00 Overlap:**
- Text timestamp: 3.0s
- Speech ends: 3.44s (technically still active)
- BUT: 3.0s falls exactly at segment boundary
- segment_1 (3.0-11.8s) processing sees text at 3.0s
- Speech segment (0.0-6.8s) is available, but text at boundary gets edge case handling

**Impact:**
- Same caption text appears in BOTH caption and overlay lists
- Requires cross-category deduplication (favor captions over overlays)
- Cross-dedup helps but doesn't eliminate all boundary cases

**Current Mitigation:**
- Grace period: 0.5s tolerance for captions appearing after speech ends
- Cross-category deduplication: Remove overlays that match captions (similarity >0.85)
- Use start time instead of midpoint for timestamp comparison

**Limitation:**
- Segment boundaries always create artificial cutoffs
- Grace period can't solve texts appearing at exact boundary (3.0s)
- Text persistence means same text gets different classifications at different timestamps

---

### Barrier 4: The Logic IS Sound But Fundamentally Limited

**Problem:** Even if Barriers 1-3 were solved, the logic cannot handle all cases.

**Stress Test Case (video 595997271203511):**
- User intentionally added overlay text "This is very good"
- While speaking the words "This is very good"
- OCR detects: "This is very good" at 21s-23s
- Speech says: "This is very good you know..." at 21.28s-23.42s
- Match quality: 1.00 (perfect match)
- **Classification: CAPTION** (because it matches speech)
- **Reality: OVERLAY** (graphical element that happens to match speech)

**The Inverse Problem:**
```
Current logic assumes:
  Text matches speech = Caption
  Text doesn't match speech = Overlay

But reality:
  Captions match speech ✓
  Overlays CAN ALSO match speech ✓
```

**Why This Is Unfixable with Current Approach:**
- We're inferring **semantic meaning** (caption vs overlay) from **technical signals** (text similarity + timing)
- There's no way to distinguish:
  - Bottom caption saying "This is very good" (caption)
  - vs. Graphical overlay saying "This is very good" (overlay)
  - When both appear while speaker says "This is very good"

**Visual Metadata Doesn't Help:**
- All texts show `position: "right", size: "medium", style: "normal"`
- No distinction between captions and overlays in current OCR output
- Would need actual screen coordinates (e.g., y-position < 20% = caption)

---

## Test Results

### Video 7480428850522950920 (Real Content)
**Before fixes:**
- segment_3: 3 overlays ("gvm", "sayoh", "weight loss is really just")

**After fixes:**
- segment_3: 0 overlays ✅

**What worked:**
- Dictionary fixed OCR errors
- Grace period fixed timing issues
- Cross-category dedup fixed text persistence

---

### Video 595997271203511 (Stress Test)
**Expected (User's Manual Analysis):**
- Hook: 0 overlays
- segment_1: 0 overlays
- segment_2: 0 overlays
- segment_3: 1 overlay ("This is very good")
- segment_4: 1 overlay ("keep talking")

**Actual (Current Implementation):**
- Hook: 1 overlay ("thisagain:")
- segment_1: 2 overlays ("So we're gonna try filming", "thisagain:")
- segment_2: 2 overlays ("talking hereg", "you knowz but; uhz")
- segment_3: 3 overlays (including OCR garbage like "knoWg")
- segment_4: 3 overlays (including OCR garbage like "QJaOWE", "itoseehowum;")

**What broke:**
1. **Barrier 1**: New OCR errors not in dictionary
2. **Barrier 3**: Boundary timing at 3.0s
3. **Barrier 4**: Stress test overlays match speech, get classified as captions

---

## Why We Keep Going in Circles

1. **Fix video A** → Works for video A ✓
2. **Test video B** → Breaks due to new OCR errors or edge cases ❌
3. **Add more fixes** → Works for video B ✓
4. **Test video C** → New edge cases appear ❌
5. **Repeat forever...**

**Root Cause:** We're treating a **fundamental limitation** (Barrier 4) as an **edge case problem**.

---

## The Compound Effect

The barriers compound each other:

```
OCR errors (Barrier 1)
  → Need fuzzy matching
    → More false positives

Partial text (Barrier 2)
  → Need aggressive fuzzy thresholds
    → More false positives

Timing issues (Barrier 3)
  → Need grace periods & cross-dedup
    → More false positives
    → More edge cases

Stress test (Barrier 4)
  → Proves approach is fundamentally limited
    → No amount of tuning will fix this
```

---

## Current Mitigations and Their Limits

### 1. OCR Error Dictionary
```python
ocr_fixes = {
    'gvm': 'gym',
    'sayoh': 'say oh',
}
```
**Limitation:** Requires manual discovery and maintenance. Cannot scale.

### 2. Edit Distance Fallback
- Levenshtein distance for ≤4 letter words
- Boost overlap if distance=1 or distance=2

**Limitation:** Only helps small words. Can't fix compound errors like "you knowz but; uhz".

### 3. Grace Period
- 0.5s tolerance for captions appearing after speech ends

**Limitation:** Doesn't solve segment boundary issues (exact 3.0s timestamp).

### 4. Cross-Category Deduplication
- Remove overlays that also appear as captions (similarity >0.85)

**Limitation:** Only works when text appears in both lists. Doesn't catch overlays that match speech but never get classified as captions.

### 5. Start Time Instead of Midpoint
- Use text start time for timestamp comparison

**Limitation:** Helps with timing but doesn't eliminate boundary issues.

---

## What Would Actually Fix This?

### Option 1: Better Visual Positioning Data
**Requirements:**
- Capture actual screen coordinates from OCR (x, y, width, height)
- Define caption region (e.g., bottom 20% of screen, centered)
- Text outside caption region = overlay, regardless of speech match

**Pros:**
- Would solve Barrier 4 completely
- More reliable classification

**Cons:**
- Requires modifying OCR pipeline (timeline_builder.py)
- May not work if captions aren't always in same position
- Doesn't solve Barriers 1-3

### Option 2: Machine Learning Classifier
**Requirements:**
- Collect ground truth data (manually label 100+ videos)
- Train classifier on features: text content, timing, position, style, speech overlap
- Use ML model instead of rule-based logic

**Pros:**
- Can learn patterns we can't articulate
- Handles edge cases better

**Cons:**
- Requires significant data labeling effort
- Model training and maintenance overhead
- Still limited by OCR quality (Barrier 1)

### Option 3: Accept It's Unreliable - Simplify
**Change:**
- Remove `overlay_unique_count` as a feature
- Replace with `has_text_overlay: boolean`
- Only track whether ANY non-caption text exists

**Pros:**
- Simple, less error-prone
- Binary classification is easier than counting
- Still provides value (filter videos with/without overlays)

**Cons:**
- Loses granularity (can't distinguish 1 vs 10 overlays)
- Doesn't solve the fundamental problem, just reduces its impact

---

## Recommendation

**Accept Barrier 4 is unfixable with current architecture.**

**Short-term (Easy Problem):**
- Add more OCR errors to dictionary as discovered
- Accept that hook/segment_1/segment_2 will have 1-2 false overlays
- Focus on making it "good enough" rather than "perfect"

**Long-term Decision:**
Either:
1. **Invest in visual positioning data** (Option 1) - if overlay count is critical
2. **Simplify to boolean** (Option 3) - if overlay count is nice-to-have
3. **Remove overlay feature entirely** - if not valuable enough to maintain

**Current state:** Consuming disproportionate effort (2+ hours of debugging) for one feature among 60+ features.

---

## Files Referenced

- `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py` - Classification logic
- `/home/jorge/rumiaifinal/rumiai_v2/processors/timeline_builder.py` - OCR processing
- `/home/jorge/rumiaifinal/OCR_Caption_Overlay_Fix.md` - Previous fix documentation
- `/home/jorge/rumiaifinal/unified_analysis/7480428850522950920.json` - Test video 1 (fixed)
- `/home/jorge/rumiaifinal/unified_analysis/595997271203511.json` - Test video 2 (stress test)

---

## Key Insight

**We're trying to infer semantic meaning (caption vs overlay) from imperfect technical signals (OCR + speech timing).**

The barriers aren't just bugs to fix—they represent fundamental limitations of the approach. No amount of edge case handling will make this reliable without architectural changes (visual positioning data or ML classifier).
