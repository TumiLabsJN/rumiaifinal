# OCR Text Overlay Upgrade

## Problem Statement
Current implementation counts OCR text detections incorrectly:
- **Issue**: Same text overlay detected multiple times (e.g., "SEXY GREEN TEA" detected 7 times in 3 seconds)
- **Result**: `text_count: 7` is misleading - suggests 7 different texts when it's actually 1 persistent overlay
- **Impact**: ML models receive incorrect signals about content pacing and information density

## Solution: Replace text_count with 6 meaningful metrics

### New Metrics
```python
# Replace:
"text_count": 7  # Misleading - counts duplicate detections

# With:
"unique_text_count": 1          # Number of distinct text strings
"max_simultaneous_texts": 1     # Peak concurrent texts on screen
"text_appearance_count": 7      # Raw detection events (for debugging)
"text_coverage": 1.0            # Fraction of window with ANY text visible (0-1)
"avg_text_lifespan": 3.0        # Average seconds each unique text is shown
"text_change_count": 0          # Major transitions (clean screen <-> has text)
```

## Implementation

### Step 1: Add text processing function to temporal_compute.py
```python
# Add after existing helper functions (around line 470)
from typing import List, Dict, Any
import numpy as np
import re

def process_text_overlays(text_timeline: List[Dict], start: float, end: float, 
                         duration: float) -> Dict[str, Any]:
    """
    Process text overlay detections to compute meaningful metrics.
    Replaces simple text_count with nuanced metrics.
    
    Design decisions:
    1. Gap detection: >1s between detections = separate text appearance
    2. Text normalization: Lowercase + remove punctuation for grouping
    3. Event-based processing: O(n log n) performance instead of sampling
    4. Major transitions only: Count clean screen <-> has text changes
    5. Position ignored: Keep metrics simple, no spatial analysis
    6. No text complexity: Let ML discover patterns from core metrics
    """
    
    # Constants
    GAP_THRESHOLD = 1.0  # Seconds between detections to consider text gone
    PERSIST_BUFFER = 0.5  # Seconds to assume text persists after last detection
    
    def normalize_text(text: str) -> str:
        """Normalize text for grouping similar OCR detections."""
        # Convert to lowercase
        text = text.lower()
        # Remove emojis and special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    # Group detections by normalized text content
    text_groups = {}
    for entry in text_timeline:
        timestamp = entry.get('timestamp', 0)
        if start <= timestamp < end:
            text_content = entry.get('data', {}).get('text', '')
            if text_content:
                normalized = normalize_text(text_content)
                if normalized:  # Skip empty after normalization
                    if normalized not in text_groups:
                        text_groups[normalized] = []
                    text_groups[normalized].append(timestamp)
    
    unique_text_count = len(text_groups)
    
    # Handle empty case
    if unique_text_count == 0:
        return {
            'unique_text_count': 0,
            'max_simultaneous_texts': 0,
            'text_appearance_count': 0,
            'text_coverage': 0.0,
            'avg_text_lifespan': 0.0,
            'text_change_count': 0
        }
    
    # Calculate lifespan of each unique text (accounting for gaps)
    text_lifespans = {}
    text_appearances = {}  # Track separate appearances
    
    for text, timestamps in text_groups.items():
        timestamps.sort()
        appearances = []
        current_appearance = [timestamps[0]]
        
        for i in range(1, len(timestamps)):
            # Check if gap is too large
            if timestamps[i] - timestamps[i-1] > GAP_THRESHOLD:
                # End current appearance, start new one
                appearances.append(current_appearance)
                current_appearance = [timestamps[i]]
            else:
                current_appearance.append(timestamps[i])
        appearances.append(current_appearance)
        
        text_appearances[text] = appearances
        
        # Calculate total lifespan (sum of all appearances)
        total_lifespan = 0
        for appearance in appearances:
            first = appearance[0]
            last = appearance[-1]
            # Add buffer for each appearance, but don't exceed segment
            lifespan = min(last + PERSIST_BUFFER, end) - first
            total_lifespan += lifespan
        
        text_lifespans[text] = total_lifespan
    
    avg_text_lifespan = sum(text_lifespans.values()) / len(text_lifespans) if text_lifespans else 0
    
    # Build list of all text appear/disappear events (event-based approach)
    events = []
    for text, appearance_list in text_appearances.items():
        for appearance in appearance_list:
            first = appearance[0]
            last = appearance[-1]
            # Text appears
            events.append((first, 'appear', text))
            # Text disappears (with buffer)
            events.append((min(last + PERSIST_BUFFER, end), 'disappear', text))
    
    # Sort events by time
    events.sort(key=lambda x: x[0])
    
    # Process events to calculate metrics
    active_texts = set()
    max_simultaneous_texts = 0
    time_with_text = 0.0
    text_change_count = 0
    prev_time = start
    was_empty = True  # Start with assumption of clean screen
    
    for event_time, event_type, text_id in events:
        # Update coverage calculation BEFORE updating active_texts
        if len(active_texts) > 0:
            time_with_text += (event_time - prev_time)
        
        # Update active texts
        if event_type == 'appear':
            active_texts.add(text_id)
        else:  # disappear
            active_texts.discard(text_id)
        
        # Check for major transitions (empty <-> has text)
        current_empty = (len(active_texts) == 0)
        if was_empty and not current_empty:
            text_change_count += 1  # Text appeared on clean screen
        elif not was_empty and current_empty:
            text_change_count += 1  # All text disappeared
        
        max_simultaneous_texts = max(max_simultaneous_texts, len(active_texts))
        was_empty = current_empty
        prev_time = event_time
    
    # Final coverage calculation
    if len(active_texts) > 0:
        time_with_text += (end - prev_time)
    
    text_coverage = time_with_text / duration if duration > 0 else 0.0
    
    # Count raw appearances for debugging
    text_appearance_count = len([e for e in text_timeline 
                                if start <= e.get('timestamp', 0) < end])
    
    return {
        'unique_text_count': unique_text_count,
        'max_simultaneous_texts': max_simultaneous_texts,
        'text_appearance_count': text_appearance_count,
        'text_coverage': float(text_coverage),
        'avg_text_lifespan': float(avg_text_lifespan),
        'text_change_count': text_change_count
    }
```

### Step 2: Modify process_segment function

Replace text counting section in process_segment:

```python
# REMOVE:
    segment_text = [t for t in timelines.get('text_overlay_timeline', []) 
                   if start <= t.get('timestamp', 0) < end]
    text_count = len(segment_text)

# ADD:
    # Process text overlays with advanced metrics
    text_metrics = process_text_overlays(
        timelines.get('text_overlay_timeline', []),
        start, end, duration
    )
    
    # Keep stickers as is for now
    segment_stickers = [s for s in timelines.get('sticker_timeline', [])
                       if start <= s.get('timestamp', 0) < end]
    sticker_count = len(segment_stickers)
```

### Step 3: Update element_count calculation

```python
# REMOVE:
    total_elements = (text_count + sticker_count + object_count + 
                     gesture_count + expression_count + scene_count)

# ADD:
    total_elements = (text_metrics['unique_text_count'] + sticker_count + 
                     object_count + gesture_count + expression_count + scene_count)
```

### Step 4: Update return dictionary

```python
# In return statement, REPLACE:
    'text_count': text_count,

# WITH:
    **text_metrics,  # Unpacks all 6 text metrics
```

## Example Outputs

### Static overlay (current video)
```python
"unique_text_count": 1
"max_simultaneous_texts": 1  
"text_coverage": 1.0         # 100% coverage
"avg_text_lifespan": 3.0     # Full 3 seconds
"text_change_count": 0       # No changes
```

### Rapid text changes
```python
"unique_text_count": 5
"max_simultaneous_texts": 1
"text_coverage": 0.8         # 80% has text
"avg_text_lifespan": 0.48    # Each text ~0.5 seconds  
"text_change_count": 2       # Text appears, then disappears (2 major transitions)
```

### Multiple simultaneous texts
```python
"unique_text_count": 3
"max_simultaneous_texts": 3
"text_coverage": 1.0
"avg_text_lifespan": 3.0     # All persist full window
"text_change_count": 0       # All static
```

## ML Benefits
1. **Better signal quality**: Distinguishes between rapid changes vs persistent overlays
2. **Cognitive load measurement**: max_simultaneous_texts indicates reading complexity
3. **Pacing analysis**: text_change_count and avg_text_lifespan reveal content rhythm
4. **Non-collinear features**: Each metric captures distinct aspect

## Files to Update
- `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`
- `/home/jorge/rumiaifinal/test_temporal_compute_v2.py` (update expected features)
- `/home/jorge/rumiaifinal/ImprovementsMLMVP.md` (mark as complete)