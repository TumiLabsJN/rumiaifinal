# Overlay Detection: The Solution

**Date**: 2025-10-02
**Discovery**: We already have the data we need - we just weren't using it!

---

## The Problem We Were Solving Wrong

We've been trying to distinguish captions from overlays using:
- Speech overlap matching
- OCR error correction
- Timing analysis
- Redundancy detection

**All unnecessary!** The answer is in the bounding box coordinates we're already capturing but discarding.

---

## What We Have: Video 907954733475671

### Bounding Box Format
```json
{
  "text": "So we're gonna try filming",
  "bbox": [43.0, 715.0, 492.0, 60.0]
}
```

Format: `[x, y, width, height]`

### Y-Position Analysis

**Top Region (y=269-272) - OVERLAYS:**
```
"Keep"        → y=269  (Stress test overlay ✓)
"Kee talkinq" → y=271  (Stress test overlay ✓)
"talking"     → y=272
```

**Middle Region (y=572-574):**
```
"This is very good" → y=572
"good"              → y=572
"This is very"      → y=574
```

**Bottom Region (y=684-798) - CAPTIONS:**
```
"basically I will keep talking"  → y=684
"the mix of uh, fruitsand"       → y=692
"So we're gonna try filming"     → y=708-711
"It works very well. And"        → y=716-718
"Nutella tastes amazing"         → y=733
"about random thingshere"        → y=735
"let's continue"                 → y=742
"continuing these tests"         → y=757
"thisagain:"                     → y=761
"always:"                        → y=787
"QJaOWE"                          → y=798 (OCR garbage)
```

### Key Findings

**Maximum y-coordinate**: 847 (y + height)
**Estimated video height**: ~900-1080 pixels (typical TikTok portrait)

**Percentages:**
- Top overlays: y=270 → **30% from top**
- Bottom captions: y=684-798 → **76-89% from top**

**Clear separation!** There's a ~400 pixel gap between overlays and captions.

---

## The Simple Solution

### Step 1: Calculate Y-Position Percentage

```python
def get_vertical_position(bbox, video_height=1080):
    """Get vertical position as percentage from top."""
    y = bbox[1]  # y-coordinate
    height = bbox[3]  # text height
    y_center = y + (height / 2)

    return y_center / video_height
```

### Step 2: Classify Based on Y-Position

```python
def classify_text_by_position(text_entry, video_height=1080):
    """Classify text as caption or overlay based on vertical position."""
    y_percent = get_vertical_position(text_entry['bbox'], video_height)

    if y_percent < 0.5:
        # Top half of screen
        return 'overlay'
    elif y_percent > 0.7:
        # Bottom 30% of screen
        return 'caption'
    else:
        # Middle region (50-70%) - uncertain
        # Use speech overlap as tiebreaker
        if text_entry['speech_overlap'] > 0.7:
            return 'caption'
        else:
            return 'overlay'
```

### Step 3: Handle Missing Video Dimensions

If we don't know video height, estimate from bounding boxes:

```python
def estimate_video_height(text_annotations):
    """Estimate video height from text bounding boxes."""
    if not text_annotations:
        return 1080  # Default TikTok portrait

    max_y_bottom = max(bbox[1] + bbox[3] for bbox in text_annotations)

    # Add margin (text rarely goes to absolute bottom)
    estimated_height = max_y_bottom * 1.1

    # Round to common video heights
    if estimated_height < 800:
        return 720
    elif estimated_height < 1200:
        return 1080
    else:
        return 1920
```

---

## Implementation in Timeline Builder

### Current Code (timeline_builder.py:435-464)

```python
def _extract_position(self, annotation: Dict[str, Any]) -> str:
    """Extract position information from annotation."""
    if 'x' in annotation and 'y' in annotation:
        x = annotation['x']
        y = annotation['y']

        if x < 0.33:
            h_pos = 'left'
        elif x > 0.67:
            h_pos = 'right'
        else:
            h_pos = 'center'

        if y < 0.33:
            v_pos = 'top'
        # ... (incomplete - y is not returned)
```

**Problem**: Y-coordinate is calculated but never returned!

### Fixed Code

```python
def _extract_position(self, annotation: Dict[str, Any]) -> str:
    """Extract position information from annotation."""
    if 'bbox' in annotation:
        bbox = annotation['bbox']
        # bbox format: [x, y, width, height]
        x = bbox[0]
        y = bbox[1]
        width = bbox[2]
        height = bbox[3]

        # Estimate video dimensions from context or use defaults
        video_width = 1080  # TikTok typical
        video_height = 1920  # TikTok typical

        # Calculate center positions
        x_center = (x + width / 2) / video_width
        y_center = (y + height / 2) / video_height

        # Vertical classification (most important)
        if y_center < 0.5:
            v_pos = 'top'
        elif y_center > 0.7:
            v_pos = 'bottom'
        else:
            v_pos = 'middle'

        # Horizontal classification
        if x_center < 0.33:
            h_pos = 'left'
        elif x_center > 0.67:
            h_pos = 'right'
        else:
            h_pos = 'center'

        return f"{v_pos}_{h_pos}"  # e.g., "bottom_center", "top_right"

    # Fallback for old format
    if 'position' in annotation:
        return annotation['position']

    return 'unknown'
```

### Store Additional Position Data

```python
# In timeline_builder.py:199-209
entry = TimelineEntry(
    start=timestamp,
    end=Timestamp(timestamp.seconds + duration),
    entry_type='text',
    data={
        'text': text,
        'position': self._extract_position(text_annotation),
        'bbox': text_annotation.get('bbox', []),  # Store raw bbox
        'size': text_annotation.get('size', 'medium'),
        'style': text_annotation.get('style', 'normal')
    }
)
```

---

## Updated Classification Logic

### In temporal_compute.py

```python
def process_text_overlays_v4(window_texts, speech_segments, video_height=1080):
    """Classify texts using vertical position + speech overlap."""

    captions = []
    overlays = []

    for text_entry in window_texts:
        # Extract vertical position
        bbox = text_entry.get('data', {}).get('bbox', [])

        if len(bbox) >= 4:
            y = bbox[1]
            height = bbox[3]
            y_center = y + (height / 2)
            y_percent = y_center / video_height

            # Primary classification: vertical position
            if y_percent < 0.5:
                # Top half = overlay (high confidence)
                overlays.append(text_entry)

            elif y_percent > 0.7:
                # Bottom 30% = caption (high confidence)
                captions.append(text_entry)

            else:
                # Middle region = uncertain
                # Use speech overlap as tiebreaker
                text_content = text_entry.get('data', {}).get('text', '')
                timestamp = text_entry.get('start', 0)
                speech_overlap = calculate_speech_overlap(
                    text_content, timestamp, speech_segments
                )

                if speech_overlap > 0.7:
                    captions.append(text_entry)
                else:
                    overlays.append(text_entry)
        else:
            # No bbox data - fallback to speech overlap only
            text_content = text_entry.get('data', {}).get('text', '')
            timestamp = text_entry.get('start', 0)
            speech_overlap = calculate_speech_overlap(
                text_content, timestamp, speech_segments
            )

            if speech_overlap > 0.7:
                captions.append(text_entry)
            else:
                overlays.append(text_entry)

    # Deduplicate
    caption_unique_texts = deduplicate(captions)
    overlay_unique_texts = deduplicate(overlays)

    # Cross-category dedup (favor captions)
    overlay_unique_texts = remove_caption_duplicates(
        overlay_unique_texts, caption_unique_texts
    )

    return {
        'overlay_unique_count': len(overlay_unique_texts),
        'caption_unique_count': len(caption_unique_texts),
        'has_captions': len(caption_unique_texts) > 0
    }
```

---

## Expected Results with New Approach

### Video 907954733475671 (Stress Test)

**Using Y-Position Classification:**

**Top Region (y<50%):**
- "Keep" (y=30%) → **OVERLAY** ✓
- "Kee talkinq" (y=30%) → **OVERLAY** ✓

**Bottom Region (y>70%):**
- All caption texts (y=76-89%) → **CAPTION** ✓

**Middle Region (50-70%):**
- "This is very good" (y=63%) → Check speech overlap → CAPTION

**Result:**
- Hook: 0 overlays ✓ (captions at y=78-89%)
- segment_1: 0 overlays ✓ (captions at y=78-89%)
- segment_2: 0 overlays ✓ (captions at y=79-85%)
- segment_3: 0 overlays ✓ ("This is very good" at y=63% with high speech overlap = caption)
- segment_4: 2 overlays ✓ ("Keep", "Kee talkinq" at y=30%)

**Wait - the user said segment_3 should have 1 overlay ("This is very good") and segment_4 should have 1 overlay ("keep talking").**

Let me reconsider the middle region logic...

---

## Refined Middle Region Handling

**The Issue**: "This is very good" at y=63% has high speech overlap, so it gets classified as caption. But the user says it's an overlay.

**Two possibilities:**

### Possibility 1: Different "This is very good" instances
- Caption "This is very good you know..." at y=79% (bottom)
- Overlay "This is very good" at y=63% (middle)

Let me check if there are multiple y-positions for same text...

### Possibility 2: Middle region should default to overlay
```python
# Middle region (50-70%)
if speech_overlap > 0.9:  # Very high threshold
    captions.append(text_entry)
else:
    overlays.append(text_entry)  # Default to overlay
```

**Reasoning**: Real captions should be in bottom region. Anything in middle is likely an overlay, even if it matches speech.

---

## Conservative Classification Strategy

**Principle**: Captions have consistent placement (bottom region). Anything outside that region is suspect.

```python
def classify_by_position_conservative(y_percent, speech_overlap):
    """Conservative classification - only bottom region is caption."""

    if y_percent > 0.7:
        # Bottom 30% = caption zone
        return 'caption'

    elif y_percent > 0.5:
        # Middle (50-70%) = probably overlay
        # Only classify as caption if VERY high speech overlap
        if speech_overlap > 0.95:
            return 'caption'
        else:
            return 'overlay'

    else:
        # Top half (<50%) = definitely overlay
        return 'overlay'
```

---

## Why This Works

### Solves All 4 Barriers:

**Barrier 1 (OCR Errors)**:
- ✓ Doesn't rely on text matching
- ✓ Position is independent of OCR quality

**Barrier 2 (Text Granularity)**:
- ✓ Doesn't need to match OCR with Whisper
- ✓ Position is the primary signal

**Barrier 3 (Timing Issues)**:
- ✓ Position is constant across frames
- ✓ No timing alignment needed

**Barrier 4 (Logic Limitation)**:
- ✓ Stress test overlays at y=30% → classified as overlay
- ✓ Even though they match speech!
- ✓ Position overrides speech matching

---

## Implementation Priority

### Phase 1: Quick Win (1 hour)
1. Update `_extract_position()` to return vertical position
2. Store bbox in timeline entry data
3. Add y-position classification to `process_text_overlays()`

### Phase 2: Refinement (30 min)
1. Estimate video height from bounding boxes
2. Tune thresholds (0.5, 0.7) based on test videos
3. Handle missing bbox data gracefully

### Phase 3: Validation (1 hour)
1. Test on both stress test videos
2. Test on 10+ real TikTok videos
3. Adjust thresholds if needed

**Total effort**: 2.5 hours (vs. days of fighting OCR errors)

---

## Confidence Level

**95%** that this will solve the overlay detection problem.

**Why 95% not 100%:**
- Need to verify bbox format is consistent across videos
- Need to tune exact threshold values (0.5, 0.7)
- Some videos might have unusual caption placement

**But this is fundamentally sound** because:
- We're using spatial information (the RIGHT signal)
- We're not fighting OCR errors
- We're not trying to infer meaning from text matching
- The data shows clear spatial separation (270 vs 684-798)

---

## Next Step

Implement Phase 1 and test on video 907954733475671.

If it works, this replaces ALL previous attempts:
- ❌ Dictionary fixes
- ❌ Edit distance
- ❌ Grace periods
- ❌ Redundancy detection
- ❌ Cross-category dedup

With:
- ✅ Y-position percentage
- ✅ Simple threshold check
- ✅ Speech overlap as tiebreaker for middle region only
