# Basic Speech Content Indicators

## Problem Statement

Our current temporal window features are completely blind to speech content. We have metrics about:
- **Quantity**: word_count, speech_coverage
- **Delivery**: audio energy, pacing
- **Timing**: segment durations

But we have **ZERO insight** into:
- **WHAT is being said**
- Content type (tutorial vs entertainment vs sales)
- Engagement techniques in language
- Linguistic energy/style

This is a critical gap - two videos with identical word_count and energy_level could be completely different content types.

## Why This Matters Despite Short Windows

Even with only 10-20 words per 3-10 second window, content patterns are distinguishable:

### Hook Examples (3s, ~15 words):
- "Hey guys welcome back to my channel today" → Personal vlog
- "Follow these three steps to get perfect" → Tutorial
- "You won't believe what happened when I" → Story/clickbait
- "This product will change your life forever" → Sales/marketing

These are fundamentally different content strategies that our ML model cannot currently distinguish.

## Proposed Implementation

### Core Indicators

Despite potential sparsity, implement these basic content signals:

#### 1. Greeting Detection (`has_greeting`)
- **Patterns**: "hey", "hello", "hi ", "welcome", "what's up"
- **Location**: Usually hook, rare in middle/closing
- **Signal**: Personal/casual content style
- **ML Value**: Distinguishes vlogs from tutorials

#### 2. Question Detection (`has_question`)
- **Patterns**: "?", "how ", "what ", "why ", "when ", "where ", "can you"
- **Location**: Can appear anywhere
- **Signal**: Engagement technique, audience interaction
- **ML Value**: Identifies interactive vs declarative content

#### 3. Instruction Detection (`has_instruction`)
- **Patterns**: "first ", "then ", "next ", "step ", "make sure", "don't forget"
- **Location**: Usually middle segments
- **Signal**: Tutorial/educational content
- **ML Value**: Strong indicator of how-to content

#### 4. Call-to-Action Detection (`has_speech_cta`)
- **Patterns**: "subscribe", "follow", "like", "comment", "share", "click", "link in bio"
- **Location**: Often in closing, but can appear anywhere
- **Signal**: Direct audience engagement request, marketing intent
- **ML Value**: Strong indicator of promotional/influencer content

### Implementation Considerations

#### Sparsity Handling
- Accept that many windows will have zeros
- The signal when present is valuable
- ML models handle sparse features well

#### Normalization
- All features are binary flags: 0 or 1
- No normalization needed
- Simple, sparse, and effective

#### Per-Window Application
Apply to all temporal windows:
- Hook (critical for greeting detection)
- Each middle segment
- Closing (critical for question/CTA detection)

## Expected Output Example

```json
"hook": {
  "word_count": 15,              // Already exists
  "speech_coverage": 1.0,        // Already exists
  "has_greeting": 1,             // NEW
  "has_question": 0,             // NEW
  "has_instruction": 0,          // NEW
  "has_speech_cta": 0            // NEW
},
"middle_segment_1": {
  "word_count": 28,
  "speech_coverage": 0.95,
  "has_greeting": 0,             // Rare in middle
  "has_question": 1,             // "How do we fix this?"
  "has_instruction": 1,          // "First, open the settings"
  "has_speech_cta": 0            // No CTA in middle
}
```

## Value Proposition

Even with sparsity, these indicators enable:
1. **Content Type Classification**: Tutorial vs vlog vs sales
2. **Engagement Pattern Recognition**: Questions and greetings placement
3. **Energy Assessment**: Linguistic energy beyond audio metrics
4. **Quality Signals**: Prepared (consistent WPM) vs spontaneous (variable WPM)

## Implementation Priority

**HIGH PRIORITY** - This is our only window into speech content. Without it, we're analyzing how people speak but not what they're saying. Even sparse signals are better than complete blindness to content.

## Technical Implementation

### Function Definition
```python
def calculate_speech_content_indicators(speech_segments, start, end, duration):
    """
    Calculate speech content indicators for a temporal window.

    This function follows the same pattern as calculate_speech_metrics_for_window()
    and other calculate_* functions in temporal_compute.py.

    Args:
        speech_segments: List of speech segments from Whisper
        start: Window start time in seconds
        end: Window end time in seconds
        duration: Window duration (end - start)

    Returns:
        dict: Speech content indicators (has_greeting, has_question,
              has_instruction, has_speech_cta)
    """
    # Collect all text in window
    window_text = ""
    for segment in speech_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)

        if seg_end <= start or seg_start >= end:
            continue

        seg_text = segment.get('text', '')
        window_text += " " + seg_text

    # Prepare for analysis
    text_lower = window_text.lower()
    words = window_text.split()
    word_count = len(words)

    # Content indicators
    greetings = ['hey', 'hello', 'hi ', 'welcome', "what's up"]
    has_greeting = 1 if any(g in text_lower[:50] for g in greetings) else 0

    questions = ['how ', 'what ', 'why ', 'when ', 'where ', 'can you']
    has_question = 1 if ('?' in window_text or any(q in text_lower for q in questions)) else 0

    instructions = ['first ', 'then ', 'next ', 'step ', 'make sure', "don't forget"]
    has_instruction = 1 if any(i in text_lower for i in instructions) else 0

    cta_patterns = ['subscribe', 'follow', 'like', 'comment', 'share', 'click', 'link in bio']
    has_speech_cta = 1 if any(cta in text_lower for cta in cta_patterns) else 0

    return {
        'has_greeting': has_greeting,
        'has_question': has_question,
        'has_instruction': has_instruction,
        'has_speech_cta': has_speech_cta
    }
```

### Integration in process_segment()
```python
# In process_segment() function, after line ~1084:

# Calculate speech metrics using proportional approach
speech_coverage, word_count = calculate_speech_metrics_for_window(
    speech_segments, start, end, duration
)

# Calculate speech content indicators (NEW)
speech_content_indicators = calculate_speech_content_indicators(
    speech_segments, start, end, duration
)

# Later in the return statement:
return {
    'start': start,
    'end': end,
    'duration': duration,
    # Text overlay metrics
    **text_metrics,
    # ... other metrics ...
    'speech_coverage': speech_coverage,
    'word_count': word_count,
    **speech_content_indicators,  # Unpack speech content indicators (NEW)
    # ... rest of metrics ...
}
```

## Conclusion

While these indicators will be sparse in our 3-10 second windows, they provide the **only insight into speech content** in our feature set. The value of knowing whether someone said "hello" or "first step" far outweighs the sparsity concern. This is essential for content type classification and should be implemented.