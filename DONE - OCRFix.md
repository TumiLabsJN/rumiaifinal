# OCR Text Deduplication Bug Fix

**Created**: 2024-09-29
**Status**: Critical Bug - Immediate Fix Required
**Impact**: Temporal window analysis incorrectly reports overlay_unique_count=0 when text is still visible

---

## 1. Bug Description

### The Problem
OCR text detection uses **global deduplication** that prevents tracking text persistence across temporal windows. When the same text appears multiple times throughout a video, only the **first occurrence** is recorded.

### Example Case
Video03TextsCaptions.mp4 showed "New Text" overlay from 6s-12s:
- **Expected**: "New Text" detected in both Segment 2 (5.67-8.33s) and Segment 3 (8.33-11s)
- **Actual**: "New Text" only recorded at 6.0s, Segment 3 shows overlay_unique_count=0

### Root Cause Location
```python
# /home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py, line 618
if text_clean not in seen_texts:  # Global deduplication
    seen_texts.add(text_clean)
    # ... append to results
```

### Why This Is Wrong
1. **Temporal analysis breaks**: Can't track when text appears/disappears
2. **Overlay persistence metrics fail**: Can't calculate how long text stays on screen
3. **Window-based counts wrong**: Text visible in window but not counted

---

## 2. Solution: Time-Window Deduplication

**Concept**: Only deduplicate texts within 1-second time windows to track persistence

**Why 1-Second Windows**:
1-second windows balance detection granularity with data growth. Since OCR samples ~5 frames/second, a 1-second window ensures we capture at least one detection cycle while preventing frame-level duplicates. This granularity is sufficient to track text persistence for our 3-second temporal windows without excessive data redundancy.

```python
# Current Problem (Global Deduplication):
Frame at 6.0s: Detects "New Text" → Stores it ✅
Frame at 9.0s: Detects "New Text" → Ignored (already seen) ❌
Frame at 10.0s: Detects "New Text" → Ignored (already seen) ❌
Result: Only ONE entry at 6.0s, Segment 3 gets overlay_unique_count=0

# Solution (Time-Window Deduplication):
Window 6 (6.0-6.9s): "New Text" → Stores once ✅
Window 9 (9.0-9.9s): "New Text" → Stores once ✅ (new window!)
Window 10 (10.0-10.9s): "New Text" → Stores once ✅ (new window!)
Result: THREE entries, Segment 3 correctly gets overlay_unique_count=1
```

**Why 1-Second Windows Work**:
- Prevents frame-level spam (if OCR runs 5 times per second, we don't want 5 identical entries)
- Tracks text persistence (shows text was visible at 6s, 9s, AND 10s)
- Enables accurate temporal analysis (each window can find what text was present)
- Reasonable data growth (10-15x instead of 60x)

**Note**: We are NOT adding an overlay freshness metric to avoid architectural complexity. This means we cannot distinguish new overlays from continuing ones across windows - this limitation is documented in MLimitations.md.

---

## 3. Implementation Plan
**File**: `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py`

```python
# Inside the _run_ocr_on_frames method, after line 603:

# Line 603, REPLACE:
seen_texts = set()

# WITH:
seen_texts_by_window = {}  # {time_window: set_of_texts}
TIME_WINDOW = 1.0  # 1-second windows (local constant)

# Line 604: LEAVE UNCHANGED
seen_stickers = set()  # Keep this line as-is (dead code but don't delete)

# Line 606, at the beginning of the for loop, ADD:
for frame_data in ocr_frames:
    try:
        # ADD THIS: Calculate time window
        time_window = int(frame_data.timestamp / TIME_WINDOW)
        if time_window not in seen_texts_by_window:
            seen_texts_by_window[time_window] = set()

        # Rest of existing code continues unchanged
        # Run OCR in thread
        results = await asyncio.to_thread(
            reader.readtext, frame_data.image
        )

# Line 618-619, REPLACE:
if text_clean not in seen_texts:
    seen_texts.add(text_clean)

# WITH:
if text_clean not in seen_texts_by_window[time_window]:
    seen_texts_by_window[time_window].add(text_clean)

# Line 656, REPLACE:
'unique_texts': len(seen_texts),

# WITH:
'unique_texts': sum(len(texts) for texts in seen_texts_by_window.values()),

# OPTIONAL: Add debug logging after line 650 (before result = {):
logger.debug(f"OCR time-window deduplication: {len(seen_texts_by_window)} windows, "
            f"{sum(len(t) for t in seen_texts_by_window.values())} unique texts total")
```

**Note**:
- Line 604 (`seen_stickers = set()`) is LEFT UNCHANGED - don't delete it
- Empty time windows (no text detected) are handled correctly - they contribute 0 to the count
- Sticker deduplication (lines 637-646) remains unchanged (dead code per StickersProblem.md)

---

## 4. Risk Analysis

### 4.1 Data Size Impact

**Current State**:
- 60 frames analyzed → ~12 text detections stored (80% deduplication)
- File size: ~5KB per video

**After Fix (Time-Window Deduplication)**:
- 60 frames → ~120-180 text detections (10-15x increase)
- File size: ~50-75KB
- **Risk Level: LOW** - Modern systems handle this easily

### 4.2 Performance Impact

**Processing Time**:
- Current: ~7.5s for OCR on 60 frames
- After fix: No change (deduplication is negligible time)
- **Risk Level: NONE**

**Memory Usage - Dictionary Storage**:
- One dictionary entry per second of video
- 10-minute video = 600 dictionary entries
- Worst case: 600 windows × 10 texts × 50 bytes = ~300KB
- Realistic: 600 windows × 3 texts × 50 bytes = ~90KB
- **Risk Level: NEGLIGIBLE** - Less than 0.03% of typical 1GB memory allocation
- Python garbage collects automatically when function exits

**Memory Usage - Detection Storage**:
- Current: ~90MB for OCR processing
- After fix: +1-5MB for storing additional detections
- **Risk Level: LOW**

### 4.3 Downstream System Impact

**temporal_compute.py**:
```python
# Currently handles multiple detections per text
def process_text_overlays(text_timeline: List[Dict], start: float, end: float, ...):
    # Already has logic to handle multiple occurrences
    segment_texts = [t for t in text_timeline
                    if start <= t['timestamp'] < end]
```
- **Risk Level: NONE** - Already designed to handle multiple detections

**Timeline Builder**:
- Expects list of text annotations with timestamps
- No changes needed
- **Risk Level: NONE**

### 4.4 Edge Cases and Benefits

**Risk 1: Flickering Text**
- Text that appears/disappears rapidly could create many entries
- **Mitigation**: Time-window deduplication (Option A) handles this

**Risk 2: Moving Text**
- Scrolling text creates multiple "unique" detections
- **Current**: Would be deduplicated (wrong)
- **After fix**: Correctly tracked as multiple positions
- **Risk Level: POSITIVE** - This is desired behavior

**Risk 3: OCR Errors**
- Same text detected with slight variations ("TEST" vs "TEST.")
- **Current**: Both stored as unique
- **After fix**: No change, still unique
- **Mitigation**: Could add fuzzy matching later if needed

### 4.5 Storage & Database Impact

**JSON File Storage**:
- 10-15x size increase per video
- 1000 videos: ~50MB → ~500MB-750MB
- **Risk Level: LOW** - Negligible for modern storage

**Database** (if used):
- More rows in text_annotations table
- Indexes on (video_id, timestamp) remain efficient
- **Risk Level: LOW**

---

## 5. Testing Strategy

### Test Cases

1. **Persistent Text Test**
   - Text visible for 10+ seconds
   - Should appear in multiple temporal windows
   - Verify overlay_persistence calculation

2. **Flickering Text Test**
   - Text appears/disappears every 0.5s
   - Should deduplicate within 1s windows
   - Verify reasonable data size

3. **Moving Text Test**
   - Scrolling text across screen
   - Should track position changes
   - Verify not over-deduplicated

4. **Multiple Overlays Test**
   - 3+ different texts visible simultaneously
   - All should be detected in each window
   - Verify overlay_unique_count

### Expected Results After Fix
```python
# Video03TextsCaptions.mp4
Segment 2 (5.67-8.33s):
  overlay_unique_count: 1 ("New Text" present)

Segment 3 (8.33-11.0s):
  overlay_unique_count: 1 ("New Text" still present) ✅ (was 0 before fix)
```

### Validation Commands
```bash
# Test the fix
python3 test_manual_videos.py Video03TextsCaptions.mp4

# Verify the key success criteria
grep "segment_3" -A 20 insights/Video03.json | grep overlay_unique_count
# Expected: overlay_unique_count changes from 0 (before fix) to 1 (after fix)
# This proves persistent text is now tracked across windows
```

---

## 6. Immediate Production Implementation

### Implementation Order:
1. **FIRST**: Implement OCR time-window deduplication (ml_services_unified.py)
2. **SECOND**: Add overlay_freshness metric (temporal_compute.py)
3. **THIRD**: Test with Video03TextsCaptions.mp4 to verify fix
4. **DONE**: Code is in production

### No Rollback Strategy:
- This is an aggressive production fix
- No staging environment - direct to production
- Testing happens on production code
- If issues arise, fix forward (no reverting)

---

## 7. Success Metrics

---

### Immediate Success Criteria
✅ "New Text" detected in both Segment 2 and Segment 3
✅ overlay_unique_count correctly reflects visible overlays per window
✅ File size increase < 20x (5KB → <100KB per video)
✅ No performance degradation > 10%

### Known Limitation
- Cannot distinguish new overlays from continuing ones (no freshness metric)
- Each window sees what overlays are present, but not if they're new or persistent
- This is an acceptable trade-off for architectural simplicity

---

## Summary

This fix addresses a critical bug where persistent text overlays were invisible to temporal analysis after their first appearance. The solution uses time-window deduplication to track text throughout the video while preventing frame-level spam.

The implementation is straightforward (~20 lines of code changes) and maintains backward compatibility for existing features.

---

*Priority: IMMEDIATE - This bug causes incorrect feature extraction that undermines temporal analysis accuracy.*