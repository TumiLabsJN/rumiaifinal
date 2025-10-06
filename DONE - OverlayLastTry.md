# Overlay Detection: Last Try Analysis

**Date**: 2025-10-02
**Goal**: Explore untried approaches to distinguish captions from overlays

---

## Key Question from User

> "Are we really giving up on Overlay? There is nothing we did not try?"

**Answer:** We haven't tried position-based clustering or other statistical approaches yet!

---

## Data Exploration: Video 595997271203511

### Position Distribution
```
left_medium_normal:   1 text  (1.4%)  → "QJaOWE" (OCR garbage)
right_medium_normal: 71 texts (98.6%) → Everything else
```

**Finding:** Position metadata has almost NO variance. All real texts (captions + overlays) are at "right_medium_normal".

### Temporal Distribution
Text appears at these timestamps:
```
0-3s    (4 seconds)   → Hook period
16-32s  (17 seconds)  → Main content
47-50s  (4 seconds)   → Closing
```

**Gaps:**
- 3-16s: No text (13 second gap)
- 32-47s: No text (15 second gap)

### Text Density (texts per timestamp)
```
Normal density: 1-4 texts per timestamp
High density:
  - Time 31s: 7 texts ← Stress test overlay location!
  - Time 27s: 5 texts
```

**Hypothesis:** Overlays might create density spikes when they appear alongside captions?

---

## What Position Metadata Tells Us

### Test Video 595997271203511
- **71 texts at "right"** (98.6%)
- **1 text at "left"** (1.4%) - OCR garbage "QJaOWE"

### Test Video 7480428850522950920
- **82 texts at "right"** (100%)
- **0 texts at other positions**

### Conclusion
**Position clustering WON'T work** - captions and overlays use the same position in these videos.

**Why:** TikTok users place both captions and overlays wherever they want. No consistent "caption region" to rely on.

---

## Approaches We HAVEN'T Tried

### Approach 1: Text Density Spike Detection

**Hypothesis:** When an overlay appears, it adds text ON TOP of existing captions, creating a density spike.

**Evidence:**
- Time 31s has 7 texts (highest density)
- Contains stress test overlay "Keep" / "Kee talkinq"
- Also contains captions: "basically I will keep talking", "about random thingshere"
- Also contains OCR garbage: "itoseehowum;", "toseehowum;"

**How It Would Work:**
```python
# Calculate baseline density
baseline = median(texts_per_timestamp)  # e.g., 2-3 texts/timestamp

# Detect spikes
for timestamp in timeline:
    if len(texts_at[timestamp]) > baseline * 2:
        # Density spike detected
        # One of these texts is likely an overlay
        # Which one?
```

**Problem:** Still need to identify WHICH text in the spike is the overlay vs caption.

---

### Approach 2: Outlier Text Detection

**Hypothesis:** Overlays might have unusual characteristics that make them statistical outliers.

**Metrics to test:**
1. **Text length** - Are overlays consistently shorter/longer?
2. **Persistence** - Do overlays persist differently than captions?
3. **Speech overlap variance** - Do overlays have unstable overlap scores?
4. **Character patterns** - Do overlays use different punctuation/capitalization?

**Let me test Text Length:**

From video 595997271203511, short texts (<10 chars):
```
"yeah,"  (5 chars) - Caption fragment
"yeah;"  (5 chars) - Caption fragment (OCR variant)
"good"   (4 chars) - Caption fragment
"knoWg"  (5 chars) - OCR garbage
"always:" (7 chars) - Caption fragment
"41ia"   (4 chars) - OCR garbage
"And, um," (8 chars) - Caption fragment
```

**Finding:** Both captions and OCR garbage can be short. Length alone doesn't distinguish overlays.

---

### Approach 3: Multi-Appearance Pattern Analysis

**Hypothesis:** Captions appear continuously as speech flows. Overlays appear sporadically.

**Test: Time gaps between appearances**

Caption text "So we're gonna try filming":
- Appears at: 0s, 1s, 2s, 3s (continuous)

OCR garbage "thisagain:":
- Appears at: 0s, 1s, 2s, 3s (also continuous!)

Overlay "Keep" / "Kee talkinq":
- Appears at: 31s only (isolated)

**Finding:** Continuous appearance = caption. Isolated appearance = could be overlay OR caption fragment.

**Problem:** How do we distinguish isolated captions from isolated overlays?

---

### Approach 4: Speech Overlap + Persistence Combined

**Current approach:** Speech overlap alone
**New approach:** Use BOTH signals simultaneously

**Logic:**
```python
if speech_overlap > 0.7:
    if appears_continuously:
        → CAPTION (high confidence)
    else:
        → Uncertain (could be overlay matching speech)

elif speech_overlap < 0.3:
    → OVERLAY (high confidence)

else:  # 0.3-0.7
    if appears_continuously:
        → CAPTION (medium confidence)
    else:
        → OVERLAY (medium confidence)
```

**Problem with stress test:**
- Overlay "Keep" appears at 31s
- Speech says "keep" at 30.37-30.84s
- Overlap = 1.00 (perfect match)
- Appears once (isolated)
- Logic says: "Uncertain - could be overlay matching speech"

**Still can't definitively classify it as overlay!**

---

### Approach 5: Comparative Analysis Within Timestamp

**Hypothesis:** At timestamps with multiple texts, the text that DOESN'T fit the pattern is the overlay.

**Example at 31s (7 texts):**
```
1. "talking"                        - Partial match "talking" in speech ✓
2. "basically I will keep talking"  - Exact match in speech ✓
3. "about random thingshere"        - Match in speech ✓
4. "itoseehowum;"                   - NO match (OCR garbage) ✗
5. "Keep"                           - Match "keep" in speech ✓
6. "Kee talkinq"                    - Match "keep talking" in speech ✓
7. "toseehowum;"                    - NO match (OCR garbage) ✗
```

**Which is the overlay?**
- Texts 4, 7 are clearly OCR garbage (no speech match)
- Texts 1, 2, 3 are clearly captions (good speech match + lowercase)
- Texts 5, 6 are candidates (match speech BUT capitalized differently)

**Pattern:** "Keep" and "Kee talkinq" start with capital K, others are lowercase.

**Insight:** Capitalization might distinguish overlays from captions!

Let me test this theory...

---

### Approach 6: Capitalization Pattern Analysis

**Hypothesis:** Captions follow speech text style (usually lowercase or sentence case). Overlays might use stylized capitalization for emphasis.

**Test on known captions:**
```
"So we're gonna try filming"    - Sentence case ✓
"It works very well. And"       - Sentence case ✓
"basically I will keep talking" - Lowercase ✓
"about random thingshere"       - Lowercase ✓
```

**Test on stress test overlays:**
```
"Keep"       - Single word, capitalized
"Kee talkinq" - Capitalized
```

**Test on OCR garbage:**
```
"QJaOWE"     - Mixed case (garbage)
"knoWg"      - Mixed case (garbage)
"itoseehowum;" - Lowercase
```

**Finding:** Pattern is inconsistent. OCR garbage also has weird capitalization.

---

### Approach 7: Consensus Voting System

**Idea:** Combine MULTIPLE weak signals to make a stronger decision.

**Weak signals we have:**
1. Speech overlap (0.0 - 1.0)
2. Text persistence (continuous vs isolated)
3. Position (mostly useless)
4. Text length
5. Capitalization pattern
6. Density spike presence
7. OCR confidence (if available)
8. Timing alignment with segment boundaries

**Voting system:**
```python
votes = {
    'caption': 0,
    'overlay': 0
}

# Vote 1: Speech overlap
if overlap > 0.7:
    votes['caption'] += 3  # Strong vote
elif overlap < 0.3:
    votes['overlay'] += 3  # Strong vote
else:
    votes['overlay'] += 1  # Weak vote (uncertain = probably overlay)

# Vote 2: Persistence
if appears_continuously:
    votes['caption'] += 2
else:
    votes['overlay'] += 1

# Vote 3: In density spike
if in_density_spike:
    votes['overlay'] += 1

# Vote 4: Capitalization
if has_unusual_capitalization:
    votes['overlay'] += 1

# Vote 5: Cross-category dedup
if also_appears_as_caption_elsewhere:
    votes['caption'] += 3  # Strong vote

# Final classification
classification = max(votes, key=votes.get)
```

**Would this work for stress test?**

"Keep" at 31s:
- Speech overlap = 1.00 → caption +3
- Appears isolated → overlay +1
- In density spike (7 texts) → overlay +1
- Unusual capitalization → overlay +1
- Total: caption=3, overlay=3 → **TIE**

**Still doesn't definitively solve it!**

---

## Approach 8: Inverse Detection - Find Captions First

**Current approach:** Try to identify overlays directly

**New approach:** Identify captions with high confidence, EVERYTHING ELSE is overlay

**How it works:**
```python
# Step 1: Find the "caption stream"
# - Texts that appear continuously (adjacent timestamps)
# - AND match speech well (>0.7 overlap)
# - AND follow consistent style patterns

caption_stream = find_continuous_matching_texts()

# Step 2: Everything NOT in caption stream = overlay
overlays = all_texts - caption_stream
```

**Would this work?**

Caption stream candidates:
- 0-3s: "So we're gonna try filming this again" variations
- 16-32s: "It works very well... basically I will keep talking..." variations
- 47-50s: Closing captions

Overlay candidates:
- Isolated texts that don't fit the stream
- Texts in density spikes that aren't part of continuous flow
- "Keep" at 31s - **isolated, not part of continuous flow**

**This might actually work!**

---

## Approach 9: Time-Window Coherence

**Hypothesis:** Captions form coherent sequences within short time windows. Overlays are random insertions.

**How it works:**
```python
# For each 3-second window:
window_texts = texts_in(t, t+3)

# Check if texts form a coherent sequence
if forms_coherent_sentence(window_texts):
    → All texts in window are CAPTIONS
else:
    → Some text is breaking coherence (likely overlay)
```

**Example at 30-33s:**
```
Texts: ["basically I will keep talking", "about random things", "Keep", "Kee talkinq"]

Speech: "basically I will keep talking about random things here to see how"

Coherence check:
- "basically I will keep talking" → matches start of speech ✓
- "about random things" → continues coherently ✓
- "Keep" → breaks coherence (duplicate of word already in text) ✗
- "Kee talkinq" → breaks coherence ✗
```

**Insight:** "Keep" / "Kee talkinq" are REDUNDANT with "keep talking" already in the caption stream!

**This could be the key insight!**

---

## The Breakthrough: Redundancy Detection

### Key Insight

**Captions don't repeat themselves.** If you're transcribing speech, you write each phrase once.

**Overlays CAN overlap with captions.** The stress test deliberately adds "Keep talking" as an overlay while caption already shows "keep talking".

### How to Detect Redundancy

```python
# Within each time window (e.g., 3 seconds):
def detect_redundant_texts(texts, window_size=3.0):
    # Group texts by normalized content
    for text_a in texts:
        for text_b in texts:
            if text_a == text_b:
                continue

            # Check if text_b is substring of text_a
            if normalize(text_b) in normalize(text_a):
                # text_b is redundant!
                # Mark it as overlay
```

### Apply to Stress Test

At timestamp 31s:
```
Texts:
1. "basically I will keep talking"
2. "Keep"

normalize("Keep") = "keep"
normalize("basically I will keep talking") = "basically i will keep talking"

Check: "keep" in "basically i will keep talking" → YES!

→ "Keep" is REDUNDANT
→ "Keep" is OVERLAY ✓
```

**This works!**

---

## Testing Redundancy Detection Logic

### Case 1: Stress Test Overlay "Keep"
```
Caption: "basically I will keep talking"
Overlay: "Keep"
Result: "keep" ⊂ "basically i will keep talking" → OVERLAY ✓
```

### Case 2: Stress Test Overlay "Kee talkinq"
```
Caption: "basically I will keep talking"
Overlay: "Kee talkinq"

normalize("Kee talkinq") = "kee talkinq"
Wait... "talkinq" has a 'q' instead of 'g' (OCR error)

Need fuzzy substring matching:
- "kee" similar to "keep" (edit distance=1)
- "talkinq" similar to "talking" (edit distance=1)

Result: Fuzzy redundancy detected → OVERLAY ✓
```

### Case 3: Normal Caption Fragment
```
Caption stream:
- "It works very well. And"
- "yeah,"

Are these redundant? NO
- "yeah" not substring of "it works very well and"
- They're separate phrases

Result: NOT redundant → CAPTION ✓
```

### Case 4: OCR Garbage "knoWg"
```
Caption stream: "you know but uh"
OCR garbage: "knoWg"

normalize("knoWg") = "knowg"
"knowg" similar to "know" (edit distance=1)

Is this redundant?
- "know" IS substring of "you know but uh"
- But "knoWg" appears at DIFFERENT timestamp than the caption

Temporal check: Are they within 1 second?
- Caption "you know" at 20.0s
- "knoWg" at 23.0s
- Gap: 3 seconds → NOT redundant

Result: NOT redundant → But still OVERLAY due to low speech overlap
```

### Case 5: Caption Persistence (Same Text Multiple Times)
```
"So we're gonna try filming" appears at:
- 0.67s
- 1.0s
- 2.0s

Are these redundant with each other? YES
But they're the SAME text (exact duplicates)

→ After deduplication, only count once as CAPTION ✓
```

---

## The Refined Algorithm: Redundancy + Speech Overlap

```python
def classify_texts(texts_in_window, speech_segments):
    # Step 1: Calculate speech overlap for all texts
    for text in texts_in_window:
        text.speech_overlap = calculate_overlap(text, speech_segments)

    # Step 2: Separate by overlap threshold
    high_overlap = [t for t in texts if t.speech_overlap > 0.7]
    low_overlap = [t for t in texts if t.speech_overlap < 0.3]
    uncertain = [t for t in texts if 0.3 <= t.speech_overlap <= 0.7]

    # Step 3: Within high_overlap texts, detect redundancy
    captions = []
    overlays = []

    # Sort by length (longer texts first - they're likely the "source")
    high_overlap.sort(key=lambda t: len(t.text), reverse=True)

    for text in high_overlap:
        # Check if this text is redundant with any identified caption
        is_redundant = False
        for caption in captions:
            if is_fuzzy_substring(text.normalized, caption.normalized):
                # This text is redundant with an existing caption
                is_redundant = True
                break

        if is_redundant:
            overlays.append(text)  # Redundant high-overlap text = overlay
        else:
            captions.append(text)  # Non-redundant high-overlap = caption

    # Step 4: Low overlap = definitely overlays
    overlays.extend(low_overlap)

    # Step 5: Uncertain - use persistence as tiebreaker
    for text in uncertain:
        if appears_continuously(text):
            captions.append(text)
        else:
            overlays.append(text)

    return captions, overlays
```

---

## Would This Solve Our Test Cases?

### Video 595997271203511 (Stress Test)

**Hook (0-3s):**
- High overlap: "So we're gonna try filming" variations
- Low overlap: "thisagain:" (OCR error)
- Redundancy: None (each caption is unique phrase)
- **Result: 0 overlays (just OCR error)** ✓

**segment_1 (3-11.8s):**
- Texts at 3.0s boundary: "So we're gonna try filming", "thisagain:"
- These are from previous segment, now out of temporal context
- Speech overlap at 3.0s = 0.00 (speech moved on)
- **Result: 2 overlays** (boundary artifacts)
- **Expected: 0 overlays** ❌

**segment_3 (20.6-29.4s):**
- High overlap: "This is very good", "you know", "the mix of fruits and Nutella"
- Redundancy: None detected
- **Expected: User says 1 overlay ("This is very good")**
- **Result: 0 overlays** ❌

Wait, the stress test "This is very good" is NOT redundant because it appears alone in its timestamp, not alongside the full caption.

Let me reconsider...

---

## The Limitation of Redundancy Detection

Redundancy detection only works when:
1. The overlay appears **at the same timestamp** as the caption it duplicates
2. OR within a small time window (e.g., 1-3 seconds)

**But in the stress test:**
- Caption "This is very good you know the mix..." appears across 21-27s
- Overlay "This is very good" appears at... let me check exactly when

Actually, looking back at the debug output:
```
[DEBUG] Text: 'This is very good' at 21s → overlap=1.00
[DEBUG] Text: 'this is very good. You' at 22s → overlap=1.00
```

Both appear! So redundancy WOULD detect:
- "This is very good" (exact)
- "this is very good. You" (longer version)
- The shorter one is substring of longer → REDUNDANT → OVERLAY ✓

**This actually might work!**

---

## Revised Assessment: Would Redundancy Work?

### For Stress Test Overlays: YES ✓
- "Keep" is substring of "basically I will keep talking" → Detected
- "This is very good" appears alongside "this is very good. You" → Detected

### For OCR Errors: PARTIALLY
- "thisagain:" vs "this again" (in speech)
  - Would need fuzzy matching to connect them
  - If caption also has OCR error variant, redundancy would detect
  - If caption is correct, redundancy won't help (different timestamps)

### For Boundary Artifacts: NO ❌
- Text at 3.0s gets 0.00 overlap
- Already classified as overlay by speech overlap
- Redundancy doesn't help here

---

## The Remaining Problem: Boundary Artifacts

**Root cause:** Captions persisting at segment boundaries get 0.00 speech overlap.

**Why:**
- Caption appears at timestamp 3.0s
- Segment_1 starts at 3.0s
- Speech that matches the caption ended at 2.65s
- Grace period (0.5s) gives us until 3.15s
- Text at 3.0s should be within grace period...

**Wait, let me recalculate:**
- Speech segment: 0.0-6.8s
- Caption at 3.0s
- 3.0s is WITHIN 0.0-6.8s!
- Why does it get overlap=0.00?

**Hypothesis:** The segment boundary logic might be cutting off speech segments incorrectly.

**This might be a BUG, not a fundamental limitation!**

---

## Final Diagnosis: Three Remaining Issues

### Issue 1: Boundary Bug (Fixable)
- Texts at segment boundaries getting 0.00 overlap incorrectly
- Should be getting overlap based on full speech segment availability
- **Fix:** Debug speech segment extraction at boundaries

### Issue 2: OCR Errors (Partially Fixed)
- Dictionary + edit distance helps but doesn't catch all errors
- "thisagain:", "talking hereg", "you knowz but; uhz" still breaking through
- **Fix:** Expand dictionary, improve normalization

### Issue 3: Stress Test Detection (Newly Solvable!)
- Redundancy detection can identify overlays that duplicate caption content
- Works when overlay appears near its matching caption
- **Fix:** Implement redundancy detection within time windows

---

## Recommended Implementation: Redundancy + Speech Overlap

```python
def process_text_overlays_v3(window_texts, speech_segments):
    # Step 1: Calculate speech overlap
    for text in window_texts:
        text.speech_overlap = calculate_overlap(text, speech_segments)
        text.normalized = normalize_text(text.content)

    # Step 2: Initial classification by overlap
    high_overlap_texts = [t for t in window_texts if t.speech_overlap > 0.7]
    low_overlap_texts = [t for t in window_texts if t.speech_overlap < 0.3]
    uncertain_texts = [t for t in window_texts if 0.3 <= t.speech_overlap <= 0.7]

    # Step 3: Redundancy detection within high_overlap group
    captions = []
    overlays = []

    # Sort by timestamp, then by length (longer first)
    high_overlap_texts.sort(key=lambda t: (t.timestamp, -len(t.normalized)))

    for text in high_overlap_texts:
        is_redundant = False

        # Check against already-classified captions
        for caption in captions:
            # Only check within 3-second window
            if abs(text.timestamp - caption.timestamp) <= 3.0:
                # Check if text is fuzzy substring of caption
                if fuzzy_substring_match(text.normalized, caption.normalized, threshold=0.85):
                    is_redundant = True
                    break

        if is_redundant:
            overlays.append(text)
        else:
            captions.append(text)

    # Step 4: Low overlap = overlays
    overlays.extend(low_overlap_texts)

    # Step 5: Uncertain = use persistence tiebreaker
    for text in uncertain_texts:
        if appears_continuously(text):
            captions.append(text)
        else:
            overlays.append(text)

    # Step 6: Cross-category deduplication (existing logic)
    overlays = deduplicate_against_captions(overlays, captions)

    return captions, overlays

def fuzzy_substring_match(short_text, long_text, threshold=0.85):
    """Check if short_text is a fuzzy substring of long_text."""
    if short_text in long_text:
        return True

    # Fuzzy check: sliding window over long_text
    short_len = len(short_text)
    for i in range(len(long_text) - short_len + 1):
        window = long_text[i:i+short_len]
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, short_text, window).ratio()
        if similarity >= threshold:
            return True

    return False
```

---

## Conclusion

### What We Discovered

1. **Position clustering DOESN'T work** - all texts use same position metadata
2. **Redundancy detection DOES work** - overlays often duplicate existing caption content
3. **Boundary artifacts might be a BUG** - not a fundamental limitation
4. **Combining multiple signals** - redundancy + speech overlap + persistence gives us the best chance

### Next Steps

1. **Fix boundary bug** - Debug why text at 3.0s gets 0.00 overlap
2. **Implement redundancy detection** - Add fuzzy substring matching within time windows
3. **Expand OCR dictionary** - Add more common errors as discovered
4. **Test on both videos** - Verify improvements

### Confidence Level

**Medium-High (70%)** that redundancy detection + boundary fix will solve both test cases.

**Why not 100%?**
- Still depends on OCR quality (Barrier 1)
- Still depends on temporal alignment (Barrier 3)
- But adds a NEW signal (redundancy) that directly addresses Barrier 4

### This IS Worth Trying

We haven't exhausted all options. Redundancy detection is a genuinely new approach that could break through the Barrier 4 limitation.
