# Fix Speech: Speech Coverage Calculation Issues
**Created**: 2025-01-16
**Status**: Solution Ready for Implementation
**Issue**: Speech coverage showing 0 in hook when person is clearly talking
**Solution**: Proportional calculation for partial segments
**Breaking Change**: Yes - Expected and Acceptable

---

## Problem Statement

### Current Behavior
In video 7515687288257465630 (hook section 0-3s):
- **Current output**: `speech_coverage: 0.0`, `word_count: 0`
- **Expected**: Should show speech metrics (person is clearly talking)
- **Actual speech**: Person is clearly speaking throughout the hook

### Root Cause
The filtering logic only includes segments COMPLETELY within the window:

```python
# CURRENT (WRONG):
segment_speech = [s for s in speech_segments
                 if s.get('start', 0) >= start and s.get('end', 0) <= end]
```

Since Whisper's first segment runs from 0.0s to 4.56s, it doesn't fit completely in the 0-3s hook window, so it gets excluded entirely.

---

## Breaking Change Notice

This fix introduces a **breaking change** to speech metrics calculation. Videos processed before and after this implementation will have significantly different values for:
- `speech_coverage`: Will increase (often from 0% to actual coverage)
- `word_count`: Will increase (often from 0 to actual word count)

**This breaking change is expected and acceptable** because:
1. The current implementation is fundamentally broken (missing speech that exists)
2. The new values are objectively more accurate
3. Backward compatibility is not required for this analytics pipeline
4. All videos can be reprocessed as needed

### Example Impact
For video 7515687288257465630:
- **Hook (3s)**: `speech_coverage` changes from 0.0 to 1.0, `word_count` from 0 to 11
- **Segment 1 (7.6s)**: Now correctly calculates partial overlaps
- **Segment 5 (7.6s)**: Now handles multiple partial segments
- **Closing (3s)**: Now works for short end segments

---

## The Solution: Proportional Calculation

### Core Concept
When a speech segment partially overlaps with our analysis window, calculate what proportion of the segment falls within the window and count that proportion of the words.

**Example:**
- Segment: 0-4.56s with 16 words
- Window: 0-3s (hook)
- Overlap: 3s out of 4.56s = 65.8%
- Count: 65.8% of 16 words ≈ 11 words

---

## Implementation

### Location
File: `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`

**Scope Note**: The fix is only applied to `temporal_compute.py` as it's the active code path. Other files containing speech_segments references (precompute_functions.py, precompute_functions_full.py) are deprecated and being replaced by the temporal_compute module.

### Function Placement
The `calculate_speech_metrics_for_window` function will be placed at **module level** in `temporal_compute.py`, at line 749 (after the SEGMENT PROCESSING section comment, immediately before the `process_segment` function). This placement ensures:
- **Testability**: Can be unit tested independently
- **Reusability**: Available for other functions if needed
- **Proximity**: Close to its primary usage location (process_segment uses it)
- **Clean organization**: Follows Python best practices for helper functions

**Note**: The function handling segment processing is named `process_segment`, not `compute_segment_metrics` as originally documented.

### New Implementation

```python
import logging

# Module-level function placed before compute_segment_metrics
def calculate_speech_metrics_for_window(speech_segments, start, end, duration):
    """
    Calculate speech coverage and word count for a temporal window.
    Uses proportional calculation for segments that partially overlap.
    
    This is a module-level function for testability and potential reuse.
    
    Args:
        speech_segments: List of speech segments from Whisper
        start: Window start time in seconds
        end: Window end time in seconds  
        duration: Window duration (end - start)
    
    Returns:
        tuple: (speech_coverage, word_count)
    
    Raises:
        ValueError: If segment data is corrupted (invalid timestamps, zero duration)
                   or if duration parameter doesn't match end - start
    """
    # Validate duration parameter
    expected_duration = end - start
    if abs(duration - expected_duration) > 0.001:  # Allow tiny floating point differences
        raise ValueError(f"Duration {duration} doesn't match end-start {expected_duration}")
    
    # No speech is valid - return zeros
    if not speech_segments:
        return 0.0, 0
    
    total_speech_duration = 0.0
    total_word_count = 0.0
    
    for segment in speech_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)
        seg_text = segment.get('text', '')
        
        # Validate segment data - fail fast on corruption
        if seg_start > seg_end:
            raise ValueError(f"Corrupted segment: start {seg_start} > end {seg_end}")
        if seg_start == seg_end:
            raise ValueError(f"Zero-duration segment at {seg_start}s - indicates corrupted data")
        if seg_start < 0 or seg_end < 0:
            raise ValueError(f"Invalid negative timestamps: {seg_start} to {seg_end}")
        
        # Handle None text gracefully (empty speech segment)
        if seg_text is None:
            seg_text = ''
        
        # Check if segment overlaps with window at all
        if seg_start < end and seg_end > start:
            # Calculate the overlap duration
            overlap_start = max(seg_start, start)
            overlap_end = min(seg_end, end)
            overlap_duration = overlap_end - overlap_start
            
            # Calculate what proportion of the segment is in our window
            segment_duration = seg_end - seg_start
            # segment_duration is guaranteed > 0 due to validation above
            proportion_in_window = overlap_duration / segment_duration
            
            # Add the overlap duration to total
            total_speech_duration += overlap_duration
            
            # Count proportional words
            segment_words = len(seg_text.split())
            words_in_window = segment_words * proportion_in_window
            total_word_count += words_in_window
    
    # Calculate coverage as percentage of window
    raw_coverage = total_speech_duration / duration if duration > 0 else 0
    
    # Cap at 100% but log if exceeded (indicates overlapping segments)
    if raw_coverage > 1.0:
        # Note: logging should be imported at module level
        logger = logging.getLogger(__name__)
        logger.debug(f"Speech coverage {raw_coverage:.1%} exceeds 100% - overlapping segments detected")
        # DEBUG level chosen because:
        # 1. Function handles gracefully by capping at 100%
        # 2. Overlapping segments can be normal (interviews, multiple speakers)
        # 3. Doesn't clutter production logs with non-critical warnings
    
    speech_coverage = min(1.0, raw_coverage)
    
    # Round word count to nearest integer
    word_count = int(round(total_word_count))
    
    return speech_coverage, word_count
```

### Integration with Existing Code

The `calculate_speech_metrics_for_window` function, placed at module level right before `process_segment`, will be called from within `process_segment` to handle ALL temporal windows (hook, middle segments, closing).

### Code Patterns to Replace

Find and replace the following patterns in `process_segment` (only in temporal_compute.py):

```python
# FIND THIS PATTERN (speech segment filtering):
segment_speech = [s for s in speech_segments
                 if s.get('start', 0) >= start and s.get('end', 0) <= end]
# OR THIS (partially fixed version):
segment_speech = [s for s in speech_segments
                 if s.get('start', 0) < end and s.get('end', 0) > start]

# REPLACE WITH:
# Calculate speech metrics using proportional approach
speech_coverage, word_count = calculate_speech_metrics_for_window(
    speech_segments, start, end, duration
)

# ALSO FIND AND REMOVE this calculation logic (usually appears later in the function):
speech_duration = sum(s.get('end', 0) - s.get('start', 0) for s in segment_speech)
speech_coverage = speech_duration / duration if duration > 0 else 0
word_count = sum(len(s.get('text', '').split()) for s in segment_speech)

# These are now calculated by the function above, so remove them
```

---

## Known Limitations

### Performance Characteristics
The current implementation iterates through all speech segments for each temporal window:
- **Complexity**: O(n × m) where n = segments, m = windows
- **Typical case**: 50 segments × 7 windows = 350 iterations
- **Performance impact**: Negligible (milliseconds)
- **Decision**: Optimization not needed unless profiling shows bottleneck

For videos with unusually high segment counts (>200), performance remains acceptable as the calculation is simple arithmetic. Premature optimization would add complexity without measurable benefit.

### Word Distribution Assumption
The implementation assumes words are distributed evenly across a segment's duration. This is a simplification that works well in most cases but has known inaccuracies:

**Works Well For:**
- Steady, continuous speech
- Short segments (2-5 seconds)  
- Tutorial/instructional content with consistent pacing
- General analytics where approximate counts are sufficient

**Less Accurate For:**
- Speech with long pauses: "I... [2 second pause] ...love this!"
- Variable speed: Fast introduction followed by slow emphasis
- Dramatic delivery with intentional rhythm changes
- Segments longer than 5 seconds with uneven pacing

**Why This Is Acceptable:**
1. **Analytics tolerance**: For engagement metrics, ±2-3 words is negligible
2. **Rounding absorbs errors**: Integer rounding helps smooth small inaccuracies
3. **Consistency over precision**: All videos have the same systematic bias
4. **Practical trade-off**: Perfect accuracy would require word-level timestamps (10x processing)

**Example of Impact:**
```
Segment: "Hello everyone... [long pause] ...welcome back!" (4 words, 0-4s)
Window: 0-2s (first half of segment)

Actual: "Hello everyone..." (2 words in first 2s)
Calculated: 4 words * 0.5 = 2 words ✓ (happens to be correct)

But if: "Hello everyone welcome back... [long pause]" 
Actual: 4 words in first 2s
Calculated: 4 words * 0.5 = 2 words ✗ (undercount by 2)
```

This limitation is documented and accepted as the error margin is within acceptable bounds for video analytics purposes.

---

## Why This Solution Works

### Advantages
1. **Accurate Duration**: Speech coverage reflects actual speaking time in window
2. **Fair Word Distribution**: Words are distributed proportionally across time
3. **Logical Metrics**: Coverage capped at 100% with DEBUG-level logging for data issues
4. **Handles Edge Cases**: Works for any overlap pattern
5. **Duration Agnostic**: Works equally well for 3s hook or 7.6s middle segments
6. **Data Quality Visibility**: DEBUG logs when overlapping segments detected (non-intrusive)

### Handles Variable Window Durations

The solution is particularly robust for middle segments with varying durations:

**Example - Different Window Sizes:**
```
Speech Segment: 8.0s to 12.0s (20 words)

Window 1 (3.0-10.6s, duration=7.6s):
- Overlap: 8.0-10.6 = 2.6s
- Proportion: 2.6/4.0 = 65%
- Words: 13, Coverage: 34.2%

Window 2 (10.6-18.2s, duration=7.6s):
- Overlap: 10.6-12.0 = 1.4s  
- Proportion: 1.4/4.0 = 35%
- Words: 7, Coverage: 18.4%

Window 3 (18.2-25.8s, duration=7.6s):
- No overlap
- Words: 0, Coverage: 0%
```

The algorithm naturally adapts to any window duration, ensuring fair distribution of speech metrics across all temporal segments.

---

## Test Cases

**Testing Approach**: Use `python test_temporal_compute_v2.py "VIDEO_ID"` for verification. Primary test will be with video 7515687288257465630 where the bug was identified. Additional manual testing with other videos will be performed separately.

### Test 1: Partial Overlap at Start (Current Bug)
```python
# Segment: 0.0 to 4.56s, "Sample speech content here" (16 words)
# Window: 0.0 to 3.0s (hook)

# Expected calculation:
# Overlap: 0.0 to 3.0 = 3.0s
# Proportion: 3.0 / 4.56 = 0.658
# Words: 16 * 0.658 = 10.5 → 11 words
# Coverage: 3.0 / 3.0 = 100%
```

### Test 2: Partial Overlap at End
```python
# Segment: 2.5 to 5.0s, "Example text here" (3 words)
# Window: 0.0 to 3.0s (hook)

# Expected calculation:
# Overlap: 2.5 to 3.0 = 0.5s
# Proportion: 0.5 / 2.5 = 0.2
# Words: 3 * 0.2 = 0.6 → 1 word
# Coverage: 0.5 / 3.0 = 16.7%
```

### Test 3: Multiple Overlapping Segments
```python
# Segment 1: 2.0 to 4.0s, "First part" (2 words)
# Segment 2: 3.5 to 6.0s, "Second part here" (3 words)
# Window: 3.0 to 5.0s (duration = 2.0s)

# Expected calculation:
# Segment 1 overlap: 3.0 to 4.0 = 1.0s, proportion = 0.5, words = 1
# Segment 2 overlap: 3.5 to 5.0 = 1.5s, proportion = 0.6, words = 1.8 → 2
# Overlap between segments: 3.5 to 4.0 = 0.5s
# Actual speech duration: 1.0 + 1.5 - 0.5 = 2.0s
# Coverage: 2.0 / 2.0 = 100%, Words = 3

# Note: Coverage >100% only occurs with multiple simultaneous speakers
# (e.g., interview with overlapping speech) or corrupted timestamp data.
# Whisper typically produces non-overlapping segments for single speakers.
```

### Test 4: Validation Errors (Should Raise ValueError)
```python
# Test 4a: Negative timestamps
# Segment: -1.0 to 2.0s
# Expected: ValueError("Invalid negative timestamps: -1.0 to 2.0")

# Test 4b: Zero duration
# Segment: 3.0 to 3.0s  
# Expected: ValueError("Zero-duration segment at 3.0s - indicates corrupted data")

# Test 4c: End before start
# Segment: 5.0 to 4.0s
# Expected: ValueError("Corrupted segment: start 5.0 > end 4.0")

# Test 4d: Wrong duration parameter
# Window: 0.0 to 3.0s, but duration=5.0 passed
# Expected: ValueError("Duration 5.0 doesn't match end-start 3.0")
```

### Test 5: No Speech (Valid Case)
```python
# No segments in speech_segments list
# Window: 0.0 to 3.0s

# Expected:
# speech_coverage: 0.0
# word_count: 0
# No errors raised
```

---

## Implementation Checklist

- [ ] Add `calculate_speech_metrics_for_window` function at line 749 in temporal_compute.py (after SEGMENT PROCESSING comment, before `process_segment`)
- [ ] Import logging at module level
- [ ] Find and replace the segment filtering logic (search for `segment_speech =`)
- [ ] Find and remove the old calculation logic (search for `speech_duration = sum`)
- [ ] Test with video 7515687288257465630 using `python test_temporal_compute_v2.py "7515687288257465630"`
- [ ] Verify hook shows speech coverage > 0
- [ ] Verify word counts are reasonable
- [ ] Manual testing with other videos will be done separately by user
- [ ] Test validation: corrupted timestamps should raise ValueError
- [ ] Test validation: zero-duration segments should raise ValueError
- [ ] Test validation: wrong duration parameter should raise ValueError
- [ ] Test validation: videos with no speech should return 0,0 gracefully
- [ ] Update any unit tests that check speech metrics
- [ ] Add unit tests for the new `calculate_speech_metrics_for_window` function

---

## Notes

- This fix applies to ALL temporal windows (hook, middle segments, closing)
- The proportional approach is the best balance of accuracy and simplicity
- Function placement at module level enables independent testing and potential reuse
- The module-level placement follows Python conventions for helper functions
- **Breaking change is intentional and required** - old metrics were incorrect
- No backward compatibility needed - reprocess videos as required
- **Word distribution assumption is a known, acceptable limitation** for analytics use cases
- **Validation strategy: Fail fast on corrupted data, but handle no-speech gracefully**
- **Performance is acceptable** - Typical videos have <50 segments × 7 windows = 350 iterations (milliseconds)
- Future improvement: Whisper could provide word-level timestamps for perfect accuracy
- Debug information can be added via logging if needed in the future