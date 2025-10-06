# OCR Overlay vs Caption Classification

**Document Version:** 1.0
**Last Updated:** 2025-10-02
**System:** RumiAI v2.0
**Status:** Production Standard

---

## Overview

This document describes RumiAI's system for distinguishing between **text overlays** (creator-added graphics/titles) and **captions** (speech transcriptions) in TikTok videos using OCR data and spatial analysis.

**Why This Matters:**
- Overlay count is a critical ML feature for virality prediction
- Misclassification inflates overlay counts with UI elements and captions
- Position-based classification is more reliable than text-matching heuristics

---

## Classification Strategy

### **Primary Signal: Vertical Position (Y-coordinate)**

Text position on screen is the strongest indicator of whether it's an overlay or caption:

| Y-Position | Zone | Classification | Reasoning |
|------------|------|----------------|-----------|
| **y < 40%** | Top | **OVERLAY** | Creator graphics typically appear in top portion |
| **40% ≤ y ≤ 60%** | Middle | **OVERLAY** | Middle zone is ambiguous but more often overlays |
| **y > 60%** | Bottom | **CAPTION** | TikTok captions always appear at bottom of screen |

**Threshold Configuration:**
```python
# Located in: temporal_compute.py, classify_text_hybrid()
TOP_THRESHOLD = 0.4      # Below 40% = top zone (overlay)
BOTTOM_THRESHOLD = 0.6   # Above 60% = bottom zone (caption)
```

### **Secondary Signal: Speech Overlap (for middle zone only)**

For texts in the middle zone (40-60%), we use adaptive speech overlap thresholds:

| Zone | Speech Overlap Threshold | Logic |
|------|-------------------------|--------|
| Top (y<40%) | 0.9 | Very high bar - most top text is overlays regardless of speech match |
| Middle (40-60%) | 0.6 | Moderate - if moderate speech overlap, likely caption |
| Bottom (y>60%) | **N/A** | Position alone determines classification (always caption) |

**Key Decision (2025-10-02):** Bottom zone (y>60%) is **always** classified as caption, regardless of speech overlap. This prevents UI elements and edge-case text from being misclassified as overlays.

---

## Multi-Stage Overlay Counting Pipeline

Once texts are classified, overlays go through a multi-stage deduplication pipeline to handle OCR errors and temporal variations:

```
Raw OCR Detections
        ↓
1. CLASSIFICATION (Position + Adaptive Fuzzy)
   └─→ Split into: overlays (y<60%) | captions (y≥60%)
        ↓
2. SPATIAL CLUSTERING (100ms time buckets, 50px y-proximity)
   └─→ Merge OCR fragments at same location
        ↓
3. TEMPORAL CLUSTERING (0.5s buckets, 50px y-proximity)
   └─→ Group same overlay appearing over time
        ↓
4. AGGRESSIVE FUZZY MATCHING (0.7 char, 0.6 token similarity)
   └─→ Deduplicate OCR errors ("F8kg" vs "-8kg")
        ↓
5. CROSS-CATEGORY DEDUPLICATION (0.85 threshold)
   └─→ Remove overlays that also appear as captions
        ↓
    FINAL UNIQUE OVERLAY COUNT
```

**Note:** Captions only go through simple fuzzy deduplication (no spatial/temporal clustering).

---

## Spatial & Temporal Clustering

### **Spatial Clustering (Stage 2)**

**Purpose:** Merge OCR fragments detected in the same frame

**Logic:**
- Group by **100ms time buckets** (same timestamp = same frame)
- Within each bucket, group by **y-position proximity** (±50px)
- Merge texts left-to-right (sorted by x-coordinate)
- **Preserve timestamp and y_pos** for temporal clustering

**Configuration:**
```python
# Located in: temporal_compute.py, spatial_cluster_overlays_with_bbox()
SPATIAL_TIME_BUCKET = 0.1    # 100ms buckets (same frame grouping)
SPATIAL_Y_THRESHOLD = 50     # ±50 pixels vertical proximity
```

**Example:**
```
Input (3 fragments at same timestamp):
  3.0s  y=276  "Lunch: Chicken BLT"
  3.0s  y=304  "Wrap"
  3.0s  y=308  "Raspberries"

Output (1 merged text):
  3.0s  y=296  "Lunch: Chicken BLT Wrap Raspberries"
```

**Key Feature:** Returns `List[Dict]` with `text`, `timestamp`, `y_pos` to preserve spatial information for temporal clustering.

---

### **Temporal Clustering (Stage 3)**

**Purpose:** Merge same overlay appearing at different times (OCR detecting it across multiple frames)

**Logic:**
- Group by **0.5s time buckets** (captures same overlay over time)
- Within each bucket, group by **y-position proximity** (±50px)
- **Spatial proximity merging:** Texts at same y-position in same time bucket are automatically merged (keep longest version)
- Cross-bucket fuzzy matching handles remaining duplicates

**Configuration:**
```python
# Located in: temporal_compute.py, temporal_cluster_overlays()
TEMPORAL_BUCKET_SIZE = 0.5   # 0.5s buckets for temporal grouping
TEMPORAL_Y_THRESHOLD = 50    # ±50 pixels for y-proximity grouping
```

**Example:**
```
Input (2 detections at different times, same y-position):
  3.0s  y=524.7  "from someone who loss -8kg"
  3.2s  y=531.0  "F8kg"  (OCR error for "-8kg")

Time bucketing:
  Both → bucket 3.0 (round(3.0/0.5)*0.5 = 3.0, round(3.2/0.5)*0.5 = 3.0)

Y-proximity grouping:
  |531.0 - 524.7| = 6.3px < 50px → SAME GROUP

Output (1 merged overlay):
  "from someone who loss -8kg"  (longest version kept)
```

**Critical Logic:** If texts are:
1. In same temporal bucket (within 0.5s)
2. At same y-position (within 50px)

→ They are **automatically merged** as the same overlay, regardless of text similarity

---

### **Aggressive Fuzzy Matching (Stage 4)**

**Purpose:** Handle OCR errors and minor variations

**Thresholds:**
```python
# Located in: temporal_compute.py, should_merge_texts_aggressive()
AGGRESSIVE_CHAR_THRESHOLD = 0.7   # Character-level similarity
AGGRESSIVE_TOKEN_THRESHOLD = 0.6  # Token-level similarity
```

**Matching Rules:**
1. **Exact match:** Texts are identical (case-insensitive)
2. **Character similarity ≥ 0.7:** Catches OCR errors like "1oss" vs "loss"
3. **Token similarity ≥ 0.6:** Catches partial matches like "what eat" vs "what I eat in a day"
4. **Substring match:** One text contained in another (minimum 3 chars)
5. **Common word ratio ≥ 0.5:** Texts share ≥50% of tokens

**Example:**
```python
"Lunch: Chicken BLT Wrap" vs "Lunch Chicken BLT"
  - Char similarity: 0.82 → MERGE ✓
  - Token similarity: 0.75 → MERGE ✓
```

---

## Code Architecture

### **Main Functions**

#### 1. `classify_text_hybrid()` - Position + Adaptive Fuzzy Classification
**Location:** `temporal_compute.py:1044-1103`

**Input:**
```python
text_entry = {
    'timestamp': 3.0,
    'data': {
        'text': 'Sample Text',
        'bbox': [x, y, width, height]  # y is critical
    }
}
speech_segments = [...]  # Whisper transcription
video_height = 1080  # Default
```

**Output:**
```python
('overlay', 'top_zone_y0.25_overlap0.15')  # or
('caption', 'bottom_zone_y0.72_overlap0.94')
```

**Logic:**
```python
# Extract y-coordinate from bbox
bbox = text_entry.get('data', {}).get('bbox', [])
y_pos = bbox[1] if len(bbox) > 1 else None
y_percent = y_pos / video_height

# Position-based classification
if y_percent < 0.4:
    # TOP ZONE: Overlay with high threshold (0.9)
    if speech_overlap > 0.9:
        return 'caption'
    else:
        return 'overlay'

elif y_percent > 0.6:
    # BOTTOM ZONE: Always caption (position is primary signal)
    return 'caption'

else:
    # MIDDLE ZONE: Adaptive threshold (0.6)
    if speech_overlap > 0.6:
        return 'caption'
    else:
        return 'overlay'
```

---

#### 2. `spatial_cluster_overlays_with_bbox()` - Spatial Fragment Merging
**Location:** `temporal_compute.py:185-265`

**Input:** `List[Dict]` with `timestamp`, `data.text`, `data.bbox`
**Output:** `List[Dict]` with `text`, `timestamp`, `y_pos`

**Key Innovation:** Preserves timestamp and y-position for temporal clustering

**Algorithm:**
```python
1. Group by 100ms time buckets (same frame)
2. Within each bucket:
   a. Sort by x-position (left to right)
   b. Group by y-position proximity (±50px)
   c. Merge texts in each y-group (left-to-right order)
3. Return merged texts with preserved metadata
```

---

#### 3. `temporal_cluster_overlays()` - Temporal Grouping
**Location:** `temporal_compute.py:267-296`

**Input:** `List[Dict]` from spatial clustering
**Output:** `List[str]` of unique overlay texts

**Algorithm:**
```python
1. Group by 0.5s time buckets
2. Within each bucket, call spatial_cluster_within_bucket()
3. Apply aggressive_fuzzy_matching() across all buckets
```

---

#### 4. `spatial_cluster_within_bucket()` - Y-Proximity Merging
**Location:** `temporal_compute.py:298-341`

**Critical Logic:**
```python
# Group texts by y-position proximity (±50px)
for entry in bucket_entries:
    y_pos = entry.get('y_pos', 0)
    text = entry.get('text', '')

    # Find group with similar y-position
    for group in y_groups:
        if abs(y_pos - group['y_avg']) < 50:
            group['texts'].append(text)
            # Merge automatically - keep longest version

# If multiple texts at same y-position → same overlay
# Keep longest version (most complete OCR reading)
longest = max(group['texts'], key=len)
```

**This is the key fix:** Texts at same y-position in same time bucket are automatically considered the same overlay, regardless of text content.

---

#### 5. `process_text_overlays()` - Main Pipeline Orchestrator
**Location:** `temporal_compute.py:1375-1472`

**Full Pipeline:**
```python
# Step 1: Extract text timeline entries
window_texts = [entries from text_timeline in time window]

# Step 2: HYBRID CLASSIFICATION
for entry in window_texts:
    classification, reason = classify_text_hybrid(entry, speech_segments)
    if classification == 'caption':
        caption_entries.append(entry)
    else:
        overlay_entries.append(entry)

# Step 3: OVERLAY COUNTING PIPELINE (multi-stage)
overlay_spatial = spatial_cluster_overlays_with_bbox(overlay_entries)
overlay_temporal = temporal_cluster_overlays(overlay_spatial)
overlay_unique_texts = aggressive_fuzzy_matching(overlay_temporal)

# Step 4: CAPTION DEDUPLICATION (simple fuzzy only)
caption_texts = [e.get('data', {}).get('text', '') for e in caption_entries]
caption_unique_texts = deduplicate_with_fuzzy_matching(caption_texts)

# Step 5: CROSS-CATEGORY DEDUPLICATION
# Remove overlays that also appear as captions (0.85 threshold)
overlay_unique_texts_filtered = [...]

# Return counts
return {
    'overlay_unique_count': len(overlay_unique_texts_filtered),
    'has_captions': len(caption_unique_texts) > 0,
    ...
}
```

---

## Tunable Parameters Summary

| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| **TOP_THRESHOLD** | 0.4 | `classify_text_hybrid()` | Top zone boundary (y < 40%) |
| **BOTTOM_THRESHOLD** | 0.6 | `classify_text_hybrid()` | Bottom zone boundary (y > 60%) |
| **SPEECH_THRESHOLD_TOP** | 0.9 | `classify_text_hybrid()` | Speech overlap for top zone |
| **SPEECH_THRESHOLD_MIDDLE** | 0.6 | `classify_text_hybrid()` | Speech overlap for middle zone |
| **SPATIAL_TIME_BUCKET** | 0.1s | `spatial_cluster_overlays_with_bbox()` | Time bucket for same-frame grouping |
| **SPATIAL_Y_THRESHOLD** | 50px | `spatial_cluster_overlays_with_bbox()` | Y-proximity for spatial grouping |
| **TEMPORAL_BUCKET_SIZE** | 0.5s | `temporal_cluster_overlays()` | Time bucket for temporal grouping |
| **TEMPORAL_Y_THRESHOLD** | 50px | `spatial_cluster_within_bucket()` | Y-proximity for temporal grouping |
| **AGGRESSIVE_CHAR_THRESHOLD** | 0.7 | `should_merge_texts_aggressive()` | Character similarity threshold |
| **AGGRESSIVE_TOKEN_THRESHOLD** | 0.6 | `should_merge_texts_aggressive()` | Token similarity threshold |
| **CROSS_CATEGORY_THRESHOLD** | 0.85 | `process_text_overlays()` | Overlay-caption dedup threshold |

---

## Design Decisions & Rationale

### **Decision 1: Bottom Zone Always Caption (y > 60%)**
**Date:** 2025-10-02
**Rationale:**
- TikTok UI always places captions at bottom
- UI elements (view counts, timestamps) appear at bottom corners
- Even with low speech overlap, bottom text should be caption
- Position is more reliable than speech matching for bottom zone

**Before Fix:**
```
"808" at y=967 (90% down) → speech_overlap=0.1 → classified as OVERLAY ✗
```

**After Fix:**
```
"808" at y=967 (90% down) → y>60% → classified as CAPTION ✓
```

---

### **Decision 2: Spatial Proximity = Same Overlay**
**Date:** 2025-10-02
**Rationale:**
- OCR detects same overlay at slightly different times (frame variations)
- If texts are at same y-position (±50px) within 0.5s, they're the same overlay
- Text similarity is unreliable (OCR errors: "F8kg" vs "-8kg")
- Spatial proximity is the ground truth signal

**Before Fix:**
```
3.0s  y=524  "from someone who loss -8kg"
3.2s  y=531  "F8kg"
→ 2 overlays (fuzzy matching failed) ✗
```

**After Fix:**
```
3.0s  y=524  "from someone who loss -8kg"
3.2s  y=531  "F8kg"
→ Same bucket (3.0s), y-diff=6px → 1 overlay ✓
```

---

### **Decision 3: Preserve Metadata Through Pipeline**
**Date:** 2025-10-02
**Rationale:**
- Original spatial clustering returned `List[str]`, losing timestamp/position
- Temporal clustering couldn't use time/position for grouping
- Changed to return `List[Dict]` with `text`, `timestamp`, `y_pos`

**Impact:**
- Temporal clustering now has full spatial-temporal context
- Enables y-proximity grouping within time buckets
- More accurate overlay merging

---

## Known Edge Cases

### **1. Middle Zone Ambiguity (40-60%)**
- Some overlays appear in middle zone
- Some captions appear in middle zone (less common)
- Current: Classify as overlay unless moderate speech overlap (>0.6)
- **Tuning:** Adjust `SPEECH_THRESHOLD_MIDDLE` if needed

### **2. Multi-Line Overlays**
- Large overlays may span multiple y-positions
- Spatial clustering merges if within 50px
- **Tuning:** Increase `SPATIAL_Y_THRESHOLD` if large titles are split

### **3. Fast-Changing Overlays**
- If same overlay appears >0.5s apart, may count as 2
- Example: Overlay appears at 1s, then again at 2s
- **Tuning:** Increase `TEMPORAL_BUCKET_SIZE` to catch longer gaps

### **4. TikTok UI Elements**
- View counts, timestamps, profile icons can be OCR'd
- Usually at extreme positions (top-left, bottom-right corners)
- Bottom ones classified as captions (acceptable false positive)
- Top ones may be classified as overlays (rare, low confidence text)

---

## Testing & Validation

### **Test Cases**

**Test Video 7480428850522950920:**
- Segment 1 (3.0-14.8s): Expected 1 overlay ("from someone who loss -8kg")
  - Previously: 2 overlays (F8kg counted separately)
  - After fix: 1 overlay ✓

**Test Video 7459548276413435178:**
- Segment 3 (23.4-33.6s): Expected 2 overlays (Lunch + Dinner)
  - Previously: 4 overlays (308/808 UI elements counted)
  - After fix: 2 overlays ✓

### **Regression Tests**
- Video with 0 text overlays should return 0
- Video with captions only should return 0 overlays, has_captions=True
- Video with overlays + captions should separate correctly

---

## Future Improvements

1. **Machine Learning Classifier**
   - Train model on labeled overlay/caption data
   - Features: y-position, speech overlap, text length, confidence
   - Would eliminate manual thresholds

2. **X-Position Filtering**
   - Filter out extreme x-positions (far left/right edges)
   - Reduces TikTok UI element false positives

3. **Confidence-Based Filtering**
   - Use OCR confidence scores
   - Low confidence text likely UI artifacts

4. **Multi-Language Speech Matching**
   - Current speech overlap assumes English
   - Would need language-aware tokenization

---

## Related Documents

- `OverlayHybridApproach.md` - Original design spec for hybrid classification approach
- `MLFeaturesGIGO.md` - ML feature engineering decisions (includes overlay_count rationale)
- `FixOCR6.md` - OCR deduplication fixes (temporal/spatial clustering history)

---

## Changelog

**v1.0 - 2025-10-02**
- Initial documentation
- Documented bottom zone always caption rule
- Documented spatial proximity merging logic
- Added tunable parameters table
- Added code architecture section
