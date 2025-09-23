# Caption Sentiment Analysis Implementation Plan

## High-Level Design (HLD)

### Problem Statement
Based on ImprovementsMLMVP.md:
- `captionSentiment` is currently missing from our feature set
- TikTok captions are critical engagement signals that often drive virality
- Caption emotion (controversy, humor, inspiration) correlates with video performance
- Need to analyze the text description/caption that creators write for their videos

### Current State
- Caption data is available in `unified_analysis/{video_id}.json` under `metadata.description`
- Caption text contains the creator's written description with hashtags
- temporal_compute.py already accesses metadata and extracts description (line 1494)
- No sentiment analysis is currently performed on this text

### Proposed Solution
Implement a lightweight, dependency-free sentiment analysis that:
1. Analyzes the caption/description text from metadata
2. Returns a normalized sentiment score between -1 (negative) and 1 (positive)
3. Adds `caption_sentiment` to the metadata output section
4. Uses rule-based approach to avoid external dependencies

### Architecture Decision
- **Location**: Add to temporal_compute.py in the metadata section
- **Why not emotional_journey**: Caption is metadata, not video content emotion
- **Integration point**: After extracting metadata.description, before returning metadata dict

## Implementation Process

### Step 1: Create Sentiment Analysis Function

```python
def calculate_caption_sentiment(caption_text: str) -> float:
    """
    Calculate sentiment score for video caption/description.
    Uses rule-based approach without external dependencies.

    Args:
        caption_text: The video description/caption text

    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive)
    """
```

#### Sentiment Rules:
1. **Positive indicators** (+0.2 each):
   - Words: "love", "amazing", "perfect", "beautiful", "awesome", "great", "best"
   - Emojis: ❤️, 😍, 🔥, ✨, 💯, 🙌
   - Patterns: Multiple exclamation marks (!!), "so good", "can't wait"

2. **Negative indicators** (-0.2 each):
   - Words: "hate", "awful", "terrible", "worst", "bad", "disgusting", "fail"
   - Emojis: 😢, 😭, 😡, 💔, 👎
   - Patterns: "don't like", "can't stand", "waste"

3. **Intensity modifiers**:
   - "very", "really", "extremely": multiply score by 1.5
   - "not" before positive word: reverse sentiment
   - ALL CAPS: multiply score by 1.2 (emphasis)

4. **Normalization**:
   - Cap final score between -1 and 1
   - Return 0.0 for empty or neutral captions

### Step 2: Integrate into temporal_compute.py

**Location**: In the `compute_temporal_windows()` function, after line 1494 where description is extracted

```python
# Around line 1494-1495 in compute_temporal_windows
'description': metadata.get('description', ''),

# ADD NEW CODE HERE:
caption_text = metadata.get('description', '')
caption_sentiment = calculate_caption_sentiment(caption_text)
```

### Step 3: Add to Output Structure

**Location**: In the metadata return dictionary (around line 1490-1495)

```python
'metadata': {
    'video_id': video_id,
    'duration': video_duration,
    'digg_count': metadata.get('diggCount', metadata.get('likes', 0)),
    # ... existing fields ...
    'description': metadata.get('description', ''),
    'caption_sentiment': caption_sentiment  # NEW FIELD
}
```

## Testing Plan

### Test Cases

1. **Positive Caption Test**:
   - Input: "This is amazing! I love this hack ❤️ #amazing #love"
   - Expected: Score > 0.5

2. **Negative Caption Test**:
   - Input: "This is the worst thing ever 😭 don't try this"
   - Expected: Score < -0.3

3. **Neutral Caption Test**:
   - Input: "Green tea recipe #greentea #recipe"
   - Expected: Score near 0.0

4. **Empty Caption Test**:
   - Input: ""
   - Expected: 0.0

5. **Mixed Sentiment Test**:
   - Input: "Started bad but ended amazing! ❤️"
   - Expected: Small positive score (competing sentiments)

### Validation Method
1. Run on test video: 7515687288257465630
2. Check caption: "Burn more calories at rest, reduce sugar and carb cravings, balance blood sugar, and boost skin clarity ❤️"
3. Expected result: Slight positive (health benefits + heart emoji)

## Implementation Checklist

- [ ] Create `calculate_caption_sentiment()` function
- [ ] Add positive word list and scoring
- [ ] Add negative word list and scoring
- [ ] Add emoji sentiment mapping
- [ ] Implement intensity modifiers
- [ ] Add normalization (-1 to 1 range)
- [ ] Integrate into compute_temporal_windows()
- [ ] Add caption_sentiment to metadata output
- [ ] Test with various caption examples
- [ ] Update ImprovementsMLMVP.md as DONE

## Benefits

1. **No Dependencies**: Pure Python implementation, no external libraries needed
2. **Fast**: O(n) complexity where n is caption length
3. **Interpretable**: Clear rules for sentiment scoring
4. **Domain-Specific**: Can tune for TikTok-specific language and emojis
5. **Extensible**: Easy to add more patterns and rules

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Sarcasm detection | Accept limitation, most TikTok captions are direct |
| Emoji encoding issues | Use try-except blocks around emoji detection |
| Language other than English | Return 0.0 for non-ASCII heavy text |
| Overly simplistic | Start simple, can enhance with ML later if needed |

## Future Enhancements (Post-MVP)

1. Add TikTok-specific slang sentiment patterns
2. Weight hashtag sentiment separately
3. Detect questions and CTAs as engagement signals
4. Add language detection to handle non-English captions
5. Consider TextBlob or VADER for more sophisticated analysis

## Success Criteria

- Caption sentiment is calculated for all videos
- Sentiment scores are normalized between -1 and 1
- No external dependencies added
- Processing time < 0.01s per caption
- Test coverage for edge cases