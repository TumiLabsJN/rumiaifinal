# OCR Text Overlay Upgrade
**Created**: 2025-01-16
**Status**: Phase 1 Implementation

---

## Executive Summary
Upgrade OCR text detection to distinguish between marketing overlays and speech captions, providing separate metrics for each type. This fixes the critical issue where speech captions are incorrectly counted as unique marketing texts.

---

## Problem Statement
Current system reports 6 unique texts when there's actually 1 marketing overlay + speech captions. The OCR detects both:
- Marketing overlay: "THIS 'SEXY GREEN TEA' BOOSTS FAT BURN 🔥"  
- Speech captions: Auto-generated subtitles matching the narration

This misleads ML models about actual marketing text usage.

---

## Phase 1: Hybrid Classification Implementation

> ⚠️ **BREAKING CHANGE WARNING** ⚠️
> 
> This implementation **REMOVES** the `unique_text_count` metric entirely because it was fundamentally incorrect (counted captions as marketing text). 
> 
> **Any downstream systems using `unique_text_count` WILL BREAK** and must be updated to use:
> - `overlay_unique_count` for marketing text analysis
> - `caption_unique_count` for subtitle analysis
> - Sum both if you need total text count
> 
> This is an intentional breaking change to fix a critical data quality issue.

### Implementation Status
**This is a design specification.** The code shown provides interfaces and logic flow to guide implementation.

Actual implementation will require:
- Completing helper function bodies (shown as `pass`)
- Integration with existing temporal_compute.py
- Testing with production data
- Validation of threshold values

### Approach
Combine **temporal clustering patterns** (primary signal) with **speech matching** (secondary signal) to classify texts as overlay vs caption.

**Key Insights**: 
1. Captions and overlays have fundamentally different temporal behaviors:
   - **Captions**: Change rapidly (every 1-3 seconds) synchronized with speech rate
   - **Overlays**: Persist longer (5-30+ seconds) for marketing visibility

2. Temporal pattern is more reliable than speech matching:
   - Captions can be paraphrased or translated (low speech match)
   - Marketing overlays can quote speech (high speech match)
   - But temporal behavior is consistent: captions change rapidly, overlays persist

This allows us to use a pattern-weighted approach: trust temporal patterns first, use speech matching for verification or when patterns are unclear.

### Implementation Steps

#### 1. Add Classification Helper Functions

**Note**: These helpers are either existing functions or straightforward implementations. Signatures shown for clarity.

```python
# Classification thresholds (based on typical behavior, to be validated with data)
CAPTION_CHANGE_RATE = 0.5  # Captions typically change >0.5 times/sec with speech
OVERLAY_CHANGE_RATE = 0.2  # Marketing overlays typically change <0.2 times/sec
CLUSTER_GAP_THRESHOLD = 1.0  # Texts >1.0s apart are considered separate events

# Text normalization helper
def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison (lowercase, remove punctuation)."""
    pass

# Classification helpers
def calculate_speech_overlap(text: str, timestamp: float, speech_segments: List[Dict]) -> float:
    """Calculate % word overlap between text and speech at timestamp."""
    # Find overlapping speech segment
    # Normalize and compare words
    # Return overlap ratio 0.0-1.0

def cluster_texts_temporally(text_entries: List[Dict]) -> List[List[Dict]]:
    """
    Group texts appearing within CLUSTER_GAP_THRESHOLD seconds of each other.
    Uses 1.0s gap because captions change rapidly (with speech) while overlays persist longer.
    """
    # Sort by timestamp
    # Group texts within temporal proximity (1.0s gap separates different text events)
    # Return list of clusters

def analyze_cluster_pattern(cluster: List[Dict]) -> str:
    """
    Determine if cluster shows sequential or persistent pattern using change rate.
    Captions change rapidly (>0.5/sec), overlays persist (<0.2/sec).
    """
    if len(cluster) < 2:
        return 'unknown'
    
    # Calculate time span and unique texts
    time_span = cluster[-1]['timestamp'] - cluster[0]['timestamp']
    unique_texts = len(set(t['text'] for t in cluster))
    
    if time_span == 0:
        return 'unknown'
    
    # Single text that persists = overlay behavior
    if unique_texts == 1:
        return 'persistent'
    
    # Calculate rate of text transitions per second
    # (unique_texts - 1) = number of transitions between different texts
    change_rate = (unique_texts - 1) / time_span
    
    if change_rate > CAPTION_CHANGE_RATE:
        return 'sequential'  # Rapid changes = captions
    elif change_rate < OVERLAY_CHANGE_RATE:
        return 'persistent'  # Few changes = overlay
    else:
        return 'mixed'  # Unclear, use speech matching

# Metric calculation helpers  
def get_empty_text_metrics() -> Dict[str, Any]:
    """Return default metrics when no texts are present."""
    return {
        'overlay_unique_count': 0,
        'overlay_coverage': 0.0,
        'overlay_persistence': 0.0,
        'caption_unique_count': 0,
        'caption_coverage': 0.0,
        'caption_density': 0.0,
        'max_simultaneous_texts': 0,
        'text_appearance_count': 0,
        'text_change_count': 0,
        'avg_text_lifespan': 0.0
    }

def calculate_coverage(texts: List[Dict], duration: float) -> float:
    """Calculate percentage of time window that has text visible."""
    pass

def calculate_avg_persistence(texts: List[Dict]) -> float:
    """Calculate average seconds each unique text persists on screen."""
    pass

def calculate_max_simultaneous(texts: List[Dict]) -> int:
    """Calculate maximum number of texts visible at same time."""
    pass

def calculate_avg_lifespan(texts: List[Dict]) -> float:
    """Calculate average lifespan of all text appearances."""
    pass

def count_text_changes(texts: List[Dict]) -> int:
    """Count number of text transition events."""
    pass
```

#### 2. Update process_text_overlays Function
```python
def process_text_overlays(text_timeline: List[Dict], start: float, end: float, 
                         duration: float, speech_segments: List[Dict] = None) -> Dict[str, Any]:
    """
    Process OCR text with overlay vs caption classification.
    """
    # Handle edge cases
    if not text_timeline:
        return get_empty_text_metrics()
    if speech_segments is None:
        speech_segments = []
    
    # Step 1: Filter texts in window
    window_texts = [t for t in text_timeline if start <= t['timestamp'] < end]
    
    # Early return if no texts in this window
    if not window_texts:
        return get_empty_text_metrics()
    
    # Step 2: Cluster texts temporally
    text_clusters = cluster_texts_temporally(window_texts)
    
    # Step 3: Classify each cluster
    overlay_texts = []
    caption_texts = []
    
    for cluster in text_clusters:
        pattern = analyze_cluster_pattern(cluster)  # Uses change rate
        
        for text_entry in cluster:
            speech_overlap = calculate_speech_overlap(
                text_entry['text'], 
                text_entry['timestamp'],
                speech_segments
            )
            
            # Pattern-weighted classification: trust temporal pattern over speech
            if pattern == 'sequential':
                # High change rate strongly indicates captions
                # Speech overlap adds confidence but doesn't override
                caption_texts.append(text_entry)
                # Note: Even without speech match, rapid changes = captions (could be descriptions, translations)
                    
            elif pattern == 'persistent':
                # Low change rate strongly indicates overlay
                # Speech overlap is secondary signal
                overlay_texts.append(text_entry)
                # Note: Even with speech match, persistence = overlay (marketing often quotes speech)
                    
            else:  # pattern == 'mixed' or 'unknown'
                # No clear pattern - must rely on speech matching
                # Use 0.5 as single consistent threshold
                if speech_overlap > 0.5:
                    caption_texts.append(text_entry)
                else:
                    overlay_texts.append(text_entry)
    
    # Step 4: Calculate separate metrics
    return {
        # Overlay metrics (marketing text)
        'overlay_unique_count': len(set(normalize_text(t['text']) for t in overlay_texts)),
        'overlay_coverage': calculate_coverage(overlay_texts, duration),
        'overlay_persistence': calculate_avg_persistence(overlay_texts),
        
        # Caption metrics (speech subtitles)
        'caption_unique_count': len(set(normalize_text(t['text']) for t in caption_texts)),
        'caption_coverage': calculate_coverage(caption_texts, duration),
        'caption_density': len(caption_texts) / duration if duration > 0 else 0,
        
        # REMOVED: unique_text_count (was fundamentally wrong, counting captions as marketing)
        # Downstream systems must now explicitly use overlay_unique_count or caption_unique_count
        
        # Keep other processing metrics that aren't ambiguous
        'max_simultaneous_texts': calculate_max_simultaneous(all_texts),
        'text_appearance_count': len(all_appearance_events),
        'text_change_count': count_text_changes(all_texts),
        'avg_text_lifespan': calculate_avg_lifespan(all_texts)
    }
```

#### 3. Integration Points
- **Location**: `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`
- **Function**: `process_text_overlays()` at line 474
- **Called from**: `process_segment()` which passes `speech_segments` from timelines

#### 4. Expected Output Changes
```json
{
  "hook": {
    // New overlay-specific metrics (marketing text only)
    "overlay_unique_count": 1,        // "BOOSTS FAT BURN" 
    "overlay_coverage": 0.8,          // 80% of window has overlay
    "overlay_persistence": 2.4,       // Avg seconds overlay persists
    
    // New caption-specific metrics (speech subtitles only)
    "caption_unique_count": 3,        // 3 different caption lines
    "caption_coverage": 0.6,          // 60% has captions
    "caption_density": 1.5,           // 1.5 captions per second
    
    // REMOVED: unique_text_count (was misleading)
    // Processing metrics (kept, work on all texts)
    "max_simultaneous_texts": 2,      // Max texts on screen at once
    "text_appearance_count": 7,       // Total text appearance events
    "text_change_count": 6,           // Number of text transitions
    "avg_text_lifespan": 1.5          // Avg seconds each text visible
  }
}
```

**Breaking Change**: `unique_text_count` has been removed because it was fundamentally incorrect (counted captions as marketing text). Downstream systems must update to use:
- `overlay_unique_count` for marketing text analysis
- `caption_unique_count` for subtitle analysis
- Sum both if total count needed

### Testing Strategy
1. Test on video 7515687288257465630 (known to have 1 overlay + captions)
2. Verify overlay_unique_count = 1 (not 6)
3. Verify caption_unique_count > 0
4. Test on videos without captions
5. Test on videos without overlays

### Success Criteria
- ✅ Correctly identifies "BOOSTS FAT BURN" as overlay (count = 1)
- ✅ Correctly identifies speech subtitles as captions
- ✅ No regression in existing metrics
- ✅ Works across different video styles

---

## Phase 2: Future Enhancements (NOT YET IMPLEMENTED)
- Add confidence scores to classifications
- Consider font size/style differences
- Add position-based hints (without hard rules)
- Cache normalized text for performance
- Add ML model for edge cases

---

## Decision Log
- **2025-01-16**: Chose hybrid approach over pure speech matching to handle edge cases
- **2025-01-16**: Added temporal clustering to detect caption vs overlay patterns  
- **2025-01-16**: Decision 1 - Set 1.0s clustering gap based on caption vs overlay temporal behaviors (captions change every 1-3s with speech, overlays persist 5-30s+)
- **2025-01-16**: Decision 2 - Use rate-based pattern detection: >0.5 changes/sec = sequential (captions), <0.2 changes/sec = persistent (overlays), middle = use speech matching
- **2025-01-16**: Decision 3 - Pattern-weighted classification: trust temporal pattern over speech matching (patterns are more reliable than word overlap)
- **2025-01-16**: Decision 4 - Remove `unique_text_count` entirely (was fundamentally wrong). Force downstream to explicitly choose overlay or caption metrics

---

## Notes
- Speech matching alone fails when marketing text quotes the narration
- Position-based rules are unreliable (captions can appear anywhere)
- Temporal patterns are strong signals: captions replace, overlays persist
- This is a Phase 1 implementation - we can iterate based on results