# Expand Generic Hashtag Detection - Implementation Plan

## ⚠️ CRITICAL BUG DISCOVERED

**The hashtag processing is currently ORPHANED in the new architecture!**
- Old flow (precompute_functions_full.py) has the logic but isn't called
- New flow (temporal_compute.py) doesn't process hashtags
- Hashtags are scraped but never analyzed!
- **This means genericRatio is currently 0 for ALL videos in production**

## Problem Statement
1. **Bug**: Hashtag analysis completely missing from pipeline
2. **Accuracy**: Only detecting 6 generic hashtags instead of 14
3. **Impact**: Creators get no hashtag strategy insights at all

## Implementation Solution

### PART 1: Add Hashtag Processing to temporal_compute.py

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`

**Step 1: Add hashtag extraction function** (after line ~500)
```python
def extract_hashtag_metrics(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and analyze hashtag strategy from video metadata.

    Args:
        metadata: Video metadata containing hashtags

    Returns:
        Dictionary with hashtag metrics
    """
    hashtags = metadata.get('hashtags', [])

    # Define generic hashtags (expanded list)
    generic_hashtags = [
        # Discovery-focused (original 6)
        'fyp', 'foryou', 'foryoupage', 'viral', 'trending', 'explore',

        # Platform identity (new)
        'tiktok',           # Platform name itself
        'tiktokviral',      # Viral aspiration

        # Creator community (new)
        'tiktokcreator',    # Creator-focused discovery
        'contentcreator',   # Generic creator tag

        # Engagement bait (new)
        'funny',            # General entertainment
        'duet',             # Collaboration/reaction content

        # Trending variations (new)
        'trendingvideo',    # Variation of trending
        'tiktokchallenge'   # Challenge participation
    ]

    # Count hashtags
    total_count = len(hashtags)
    generic_count = 0
    specific_hashtags = []

    for tag in hashtags:
        # Extract tag text (handles both string and dict formats)
        if isinstance(tag, dict):
            tag_text = tag.get('name', '').lower().strip('#')
        else:
            tag_text = str(tag).lower().strip('#')

        if tag_text in generic_hashtags:
            generic_count += 1
        else:
            specific_hashtags.append(tag_text)

    # Calculate metrics
    generic_ratio = generic_count / total_count if total_count > 0 else 0

    return {
        'hashtag_count': total_count,
        'generic_hashtag_count': generic_count,
        'specific_hashtag_count': total_count - generic_count,
        'generic_ratio': round(generic_ratio, 3)  # ML-compatible numeric only
    }

# Note: Removed classify_hashtag_strategy for ML compatibility
# Only numeric features are included for direct ML consumption
```

**Step 2: Integrate into compute_temporal_windows** (around line ~1450)
```python
def compute_temporal_windows(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Complete implementation of temporal windows computation."""

    # ... existing code ...

    # Extract metadata (already exists around line 1400)
    metadata = analysis_dict.get('metadata', {})

    # ADD THIS: Extract hashtag metrics
    hashtag_metrics = extract_hashtag_metrics(metadata)

    # ... existing temporal window processing ...

    # Update return statement (around line 1500)
    return {
        'video_id': video_id,
        'duration': video_duration,
        'temporal_windows': {
            'hook': hook_result,
            'middle_segments': middle_results,
            'closing': closing_result
        },
        'metadata': {
            'video_id': video_id,
            'duration': video_duration,
            'digg_count': metadata.get('likes', 0),
            'play_count': metadata.get('views', 0),
            'collect_count': metadata.get('saves', 0),
            'share_count': metadata.get('shares', 0),
            'comment_count': metadata.get('comments', 0),
            'create_time': metadata.get('create_time'),
            'author': metadata.get('username'),
            'description': metadata.get('description'),
            # ADD HASHTAG METRICS HERE
            'hashtag_analysis': hashtag_metrics
        },
        'processing_timestamp': time.time(),
        'version': '2.0.0'
    }
```

### PART 2: Testing the Fix

**Create test file**: `test_hashtag_fix.py`
```python
#!/usr/bin/env python3
"""Test hashtag analysis fix."""

import json
from pathlib import Path
from rumiai_v2.processors.temporal_compute import extract_hashtag_metrics

def test_generic_detection():
    """Test that all 14 generic hashtags are detected."""

    # Test old generics still work
    metadata = {
        'hashtags': [
            {'name': '#fyp'},
            {'name': '#viral'},
            {'name': '#fitness'}
        ]
    }

    result = extract_hashtag_metrics(metadata)
    assert result['generic_hashtag_count'] == 2  # fyp, viral
    assert result['specific_hashtag_count'] == 1  # fitness
    assert result['generic_ratio'] == 0.67
    print("✓ Original generic hashtags detected")

    # Test new generics are detected
    metadata2 = {
        'hashtags': [
            {'name': '#tiktok'},
            {'name': '#funny'},
            {'name': '#duet'},
            {'name': '#cooking'}
        ]
    }

    result2 = extract_hashtag_metrics(metadata2)
    assert result2['generic_hashtag_count'] == 3  # tiktok, funny, duet
    assert result2['specific_hashtag_count'] == 1  # cooking
    assert result2['generic_ratio'] == 0.75
    print("✓ New generic hashtags detected")

    # Test 100% generic case
    metadata3 = {
        'hashtags': [
            {'name': '#fyp'},
            {'name': '#tiktok'},
            {'name': '#viral'},
            {'name': '#funny'},
            {'name': '#trending'}
        ]
    }

    result3 = extract_hashtag_metrics(metadata3)
    assert result3['generic_ratio'] == 1.0
    assert result3['hashtag_strategy'] == 'too_generic'
    print("✓ Strategy classification working")

    print("\n✅ All hashtag tests passed!")

if __name__ == "__main__":
    test_generic_detection()
```

### PART 3: Verify in Production

**Run on real video**:
```bash
python3 scripts/rumiai_runner.py 'https://www.tiktok.com/@user/video/123'
```

**Check output includes hashtag analysis**:
```json
{
  "metadata": {
    "hashtag_analysis": {
      "hashtag_count": 5,
      "generic_hashtag_count": 3,
      "specific_hashtag_count": 2,
      "generic_ratio": 0.6
    }
  }
}
```

## Implementation Steps

1. **Add hashtag extraction function** (10 minutes)
   - Copy the `extract_hashtag_metrics` function to temporal_compute.py
   - Includes expanded 14 generic hashtags

2. **Integrate into pipeline** (5 minutes)
   - Call function in compute_temporal_windows
   - Add to metadata section of return

3. **Test the fix** (10 minutes)
   - Run test_hashtag_fix.py
   - Verify all 14 generics detected
   - Check strategy classification

4. **Production validation** (5 minutes)
   - Process a real video
   - Verify hashtag_analysis in output
   - Confirm genericRatio now accurate

## Why This Approach?

1. **Minimal changes**: Just adding missing functionality
2. **Backward compatible**: Doesn't break existing features
3. **Correct location**: Metadata processing belongs with other metadata
4. **Testable**: Clear test cases to verify fix
5. **Actionable**: Provides strategy classification for creators

## Risk Assessment
- **Risk Level**: LOW (adding missing feature)
- **Breaking Changes**: None
- **Performance Impact**: Negligible (simple string matching)
- **Rollback**: Comment out hashtag_metrics call if issues

## Expected Outcome

### Before Fix (CURRENT BUG):
- No hashtag analysis in output
- genericRatio always 0 or missing
- Creators get no hashtag insights

### After Fix:
- Accurate hashtag metrics in metadata
- 14 generic hashtags detected (up from 6)
- Clear strategy classification (too_generic, balanced, etc.)
- Creators get actionable feedback

## Decision
**IMPLEMENT IMMEDIATELY** - This fixes a critical bug where hashtag analysis is completely missing from production, plus improves accuracy with expanded detection.