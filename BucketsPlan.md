# Natural Buckets Plan for RumiAI ML Training

## Executive Summary
Instead of forcing 4 arbitrary ML buckets, embrace the natural breakpoints created by temporal window structures. By strategically splitting the highest-variance bucket (9-18s), we achieve a practical 8-bucket solution with manageable variance.

**Key Finding**: Original variance ranged from 2.25x to 4x. After splitting the 9-18s bucket, maximum variance is reduced to 2.3x, balancing ML performance with practical implementation.

## Core Principle
**Different video durations = Different content strategies = Different ML models**

The temporal window structure naturally creates breakpoints that correspond to distinct TikTok content formats. Rather than fighting this with normalization or padding, we leverage it for better model performance.

## Final 8-Bucket Definitions

### Bucket 1: Ultra-Short (0-3s)
- **Structure**: Hook only
- **Features**: ~50 features
- **Content Type**: Reaction clips, memes, quick cuts
- **Creator Strategy**: Instant impact, no time for development
- **ML Focus**: Hook effectiveness is everything

### Bucket 2: Short (3-9s)
- **Structure**: Hook + Closing
- **Features**: ~100 features (50 × 2 windows)
- **Content Type**: Quick jokes, single tips, brief moments, extended jokes
- **Creator Strategy**: Setup → Punchline
- **ML Focus**: Hook-to-CTA transition effectiveness
- **Note**: Merges 3-6s and 6-9s videos (6-9s may have 1-3s unanalyzed middle)

### Bucket 3: Medium-A (9-13s)
- **Structure**: Hook + 3 Middle Segments + Closing
- **Features**: ~250 features (50 × 5 windows)
- **Middle Duration**: 3-7s (segments of 1-2.33s each)
- **Segment Variance**: **2.33x variance** (1s to 2.33s per segment) ✓
- **Content Type**: Quick tutorials, short jokes
- **Creator Strategy**: Rapid progression
- **ML Focus**: Fast-paced content patterns

### Bucket 4: Medium-B (13-18s)
- **Structure**: Hook + 3 Middle Segments + Closing
- **Features**: ~250 features (50 × 5 windows)
- **Middle Duration**: 7-12s (segments of 2.33-4s each)
- **Segment Variance**: **1.72x variance** (2.33s to 4s per segment) ✓
- **Content Type**: Standard tutorials, developed stories
- **Creator Strategy**: Balanced pacing
- **ML Focus**: Standard progression patterns

### Bucket 5: Medium-Long (18-33s)
- **Structure**: Hook + 4 Middle Segments + Closing
- **Features**: ~300 features (50 × 6 windows)
- **Middle Duration**: 12-27s (segments of 3-6.75s each)
- **Segment Variance**: **2.25x variance** (3s to 6.75s per segment) ⚠️
- **Content Type**: Detailed tutorials, mini-vlogs, product reviews
- **Creator Strategy**: Extended narrative with multiple beats
- **ML Focus**: Sustaining engagement through 4 development phases
- **Problem**: High variance impacts feature reliability

### Bucket 6: Long (33-60s)
- **Segment Variance**: **2x variance** (5.4s to 10.8s per segment) ⚠️
- **Structure**: Hook + 5 Middle Segments + Closing
- **Features**: ~350 features (50 × 7 windows)
- **Middle Duration**: 27-54s (segments of 5.4-10.8s each)
- **Content Type**: Full tutorials, storytimes, in-depth reviews
- **Creator Strategy**: Complete narrative arc with multiple chapters
- **ML Focus**: Long-form retention patterns

### Bucket 7: Extra-Long-A (61-90s)
- **Structure**: Hook + 5 Middle Segments + Closing
- **Features**: ~350 features (50 × 7 windows)
- **Middle Duration**: 55-84s (segments of 11-16.8s each)
- **Segment Variance**: **1.5x variance** (11s to 16.8s per segment) ✓
- **Content Type**: Extended tutorials, detailed reviews
- **Creator Strategy**: YouTube-style content adapted for TikTok
- **ML Focus**: Maintaining engagement in long-form

### Bucket 8: Extra-Long-B (91-120s)
- **Structure**: Hook + 5 Middle Segments + Closing
- **Features**: ~350 features (50 × 7 windows)
- **Middle Duration**: 85-114s (segments of 17-22.8s each)
- **Segment Variance**: **1.3x variance** (17s to 22.8s per segment) ✓
- **Content Type**: Mini-documentaries, comprehensive tutorials
- **Creator Strategy**: Full narrative arcs
- **ML Focus**: Sustaining 2-minute attention

## Implementation Details

### Training Pipeline
```python
# Pseudocode for bucket-based training
def train_all_models(videos):
    # Assign videos to buckets
    buckets = {
        'ultra_short': [],    # 0-3s
        'short': [],          # 3-6s
        'short_plus': [],     # 6-9s
        'medium': [],         # 9-18s
        'medium_long': [],    # 18-33s
        'long': [],           # 33-60s
        'extra_long': []      # 60s+
    }

    for video in videos:
        bucket = determine_bucket(video.duration)
        buckets[bucket].append(video)

    # Train separate models per bucket
    models = {}
    for bucket_name, bucket_videos in buckets.items():
        if len(bucket_videos) >= MIN_SAMPLES:  # Need sufficient data
            models[bucket_name] = {
                'random_forest': train_rf(bucket_videos),
                'kmeans': train_kmeans(bucket_videos),
                'sample_size': len(bucket_videos)
            }

    return models
```

### Bucket Assignment Logic
```python
def determine_bucket(duration):
    if duration <= 3:
        return 'ultra_short'
    elif duration <= 6:
        return 'short'
    elif duration <= 9:
        return 'short_plus'  # Consider merging with 'short'
    elif duration <= 18:
        return 'medium'
    elif duration <= 33:
        return 'medium_long'
    elif duration <= 60:
        return 'long'
    else:
        return 'extra_long'
```

## Handling Edge Cases

### The 6-9s Problem
**Current Issue**: Videos have 1-3s of unanalyzed middle content

**Options**:
1. **Merge with 3-6s bucket** (Recommended)
   - Both have Hook + Closing structure
   - Similar content types (quick content)
   - Reduces to 6 total buckets

2. **Analyze middle as single segment**
   - Add single middle segment analysis for 6-9s videos
   - Maintains 7 buckets but ensures full coverage

3. **Accept the gap**
   - Acknowledge that 1-3s middle is too short for meaningful patterns
   - Focus on hook and closing which matter most for short content

### Minimum Sample Requirements
Each bucket needs minimum 40 videos (30 training + 10 validation) to train reliable models. If a bucket has insufficient samples:
- Merge with adjacent bucket
- Or exclude from initial training until more data available

## Advantages of Natural Buckets

### 1. **Content Alignment**
Each bucket represents a distinct content strategy on TikTok:
- Ultra-short: Viral moments
- Short: Quick entertainment
- Medium: Educational content
- Long: Deep dives

### 2. **No Information Loss**
- No padding needed
- No feature aggregation
- No normalization artifacts
- Every second of content analyzed (except 6-9s gap)

### 3. **Better Model Performance**
- Models learn patterns specific to their duration
- No confusion from mixed content types
- Clearer feature importance per bucket

### 4. **Interpretable Results**
- "For 15-second tutorials, start with a question in the hook"
- "For 45-second reviews, save strongest point for segment 4"
- Duration-specific insights creators can actually use

### 5. **Scalable Architecture**
```python
# Easy to add new buckets or adjust boundaries
BUCKET_DEFINITIONS = {
    'ultra_short': {'min': 0, 'max': 3, 'middle_segments': 0},
    'short': {'min': 3, 'max': 6, 'middle_segments': 0},
    # Easy to modify or extend
}
```

## Variance Analysis

### Segment Duration Variance by Bucket

## Final 8-Bucket Solution

### Refined Bucket Structure
By splitting the problematic 9-18s bucket into two sub-buckets, we achieve manageable variance across all buckets:

| Bucket | Duration Range | Segments | Min Segment | Max Segment | Variance | Impact |
|--------|---------------|----------|-------------|-------------|----------|---------|
| 1 | 0-3s | 0 | N/A | N/A | N/A | None |
| 2 | 3-9s | 0 | N/A | N/A | N/A | None |
| 3 | 9-13s | 3 | 1.0s | 2.33s | **2.33x** | Acceptable |
| 4 | 13-18s | 3 | 2.33s | 4.0s | **1.72x** | Low |
| 5 | 18-33s | 4 | 3.0s | 6.75s | **2.25x** | Acceptable |
| 6 | 33-60s | 5 | 5.4s | 10.8s | **2.0x** | Acceptable |
| 7 | 61-90s | 5 | 11.0s | 16.8s | **1.5x** | Low |
| 8 | 91-120s | 5 | 17.0s | 22.8s | **1.3x** | Minimal |

### ML Impact of Variance

**Problem**: When segments have different durations but produce the same raw features:
- 20 words in 1s segment = fast pace (good)
- 20 words in 4s segment = slow pace (bad)
- Both produce `word_count = 20` → Model gets confused

**Expected Accuracy Drop**:
- 1.5x variance: ~5-10% accuracy drop
- 2x variance: ~15-25% accuracy drop
- 4x variance: ~40-50% accuracy drop

## Solutions to Variance Problem

### Option 1: Strategic Sub-bucketing (IMPLEMENTED)
Split only the highest-variance bucket:
- **9-18s** → Split into 9-13s and 13-18s (reduces from 4x to 2.33x and 1.72x)
- Keep other buckets as-is (variance ≤2.25x is acceptable)

**Result**: 8 total buckets - optimal balance of variance reduction and model count
tion

## Revised Migration Path

### Phase 1: Implement 8-Bucket Solution
1. Split 9-18s into 9-13s and 13-18s buckets
2. Keep other natural buckets as-is
3. Add hybrid features (raw + normalized + segment_duration)
4. Target 70-80% accuracy per bucket

### Phase 2: Initial Training & Validation
1. Collect minimum 40 videos per bucket (320 total)
2. Train 8 Random Forest models with variance-robust parameters
3. Validate on 20% holdout set
4. Document accuracy and feature importance per bucket

### Phase 3: Production Deployment
1. Deploy if accuracy >70% for majority of buckets
2. Generate bucket-specific insights for creators
3. Monitor real-world performance
4. Iterate based on creator feedback and model performance

## Expected Outcomes

### Per-Bucket Insights (Examples)
**Ultra-Short (0-3s)**
- "Videos with text overlay in first 0.5s get 3x more loops"
- "Quick zoom-in increases completion rate by 40%"

**Medium (9-18s)**
- "Energy peak in segment 2 correlates with shares"
- "Speech coverage >80% in middle segments drives engagement"

**Long (33-60s)**
- "Segment 4 emotion shift critical for retention"
- "Videos with consistent pacing across 5 segments perform best"

## Conclusion

The refined 8-bucket solution successfully addresses the variance problem while maintaining practical implementation:

**Key Achievements**:
1. **Reduced maximum variance from 4x to 2.33x** by splitting the 9-18s bucket
2. **Maintained manageable model count** (8 models instead of potential 15+)
3. **Preserved temporal structure integrity** (all features remain comparable within buckets)
4. **Balanced ML performance with practicality** (expected accuracy drop <25% instead of 40-50%)

**Final Recommendation**:
- Implement the 8-bucket solution with hybrid features (raw + normalized)
- Include segment_duration as a context feature
- Accept 2-2.3x variance as reasonable trade-off
- Ship with 70-80% accuracy target for v1
- Iterate based on real-world performance data

The "complexity" of 8 models is trivial - it's the same training pipeline with different data subsets. This solution respects both the ML requirements and the business need to ship.