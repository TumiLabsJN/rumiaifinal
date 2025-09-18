# ImprovementsMLMVP - Implementation Improvements for MVP

## Executive Summary

This document tracks HOW to implement and improve features for the RumiAI MVP. All improvements support the piecewise segment architecture: hook (0-3s) + middle segments (3-5 segments depending on video length) + closing (last 3s). Phase 2 and Phase 3 enhancements are documented in Future - MLrevolutions.md.

## Summary Table

| Priority | Category | Improvement | Difficulty | Time Est | Explanation of Importance | Technical Debt Resolved | Dependencies | Global Feature | Temporal Feature | Applicable for Both? | RF Transform | RF Difficulty | KM Transform | KM Difficulty |
|----------|----------|-------------|------------|----------|---------------------------|------------------------|--------------|----------------|------------------|---------------------|--------------|---------------|--------------|---------------|
| P0 | Architecture | Temporal Windows as Single Source of Truth | Medium | High | Enables consistent temporal pattern detection across all features | Removes 5+ redundant features, fixes mixed architecture | None | NO | YES | NO | None needed | None | None needed | None |
| P0 | Raw Data | Multimodal Counts in Windows | Easy | Low | Allows ML to discover text-speech-gesture correlations independently | Replaces pre-computed multimodal features | Temporal Windows | YES | NO | NO | None needed | Low | Scale | Low | DONE |
| P0 | Raw Data | Overlay Counts in Windows | Easy | Low | Enables ML to sum totals and discover distribution patterns | Replaces global totalOverlays, totalStickers, totalTextOverlays | Temporal Windows | YES | NO | NO | None needed | Low | Scale | Low | DONE |
| P0 | Raw Data | Per-Window Density Extremes | Easy | Low | Captures peak and floor creative intensity with temporal localization | Replaces global maxDensity/minDensity with actionable timing context | Temporal Windows | NO | YES | NO | None needed | Low | Scale | Low | DONE |
| P0 | Bug Fix | Missing shortestScene Metric | Easy | Low | Scene pacing missing min value of min/avg/max trio | Completes scene duration distribution metrics | scene_durations already calculated | YES | NO | NO | None needed | Low | Scale | Low | DONE |
| P0 | ML Principle | Remove background_noise_ratio Interpretation | Easy | Low | Violates ML discovery principle with flawed interpretation | Removes pre-computed assumption that variance = noise | None | DONE (temporal_compute uses raw energy_variance) | NO | NO | Remove feature | None | Remove feature | None |
| P0 | Bug Fix | Missing pacingVariation Implementation | Easy | Low | Referenced in wrapper but never calculated | Provides speaking speed consistency metric | WPM calculations per segment | DONE | YES | NO | None needed | Low | Scale | Low |
| P1 | Raw Data | Multi-person Metrics | Medium | Medium | Captures collaboration dynamics critical for viral content | Fixes broken subjectCount, enables group analysis | None | YES | YES | YES | None needed | Low | Scale | Low |
| P1 | Raw Data | Audio Energy Metrics (avg, peaks) | Easy | Low | Completes speech intensity patterns for emotion detection | Replaces semantic features like burstPattern | None | YES | YES | YES | None needed | Low | Scale | Low |
| P1 | Raw Data | Pitch and Spectral Voice Metrics | Medium | Medium | Captures emotional expression through acoustic features | Replaces interpretive emotion features | None | YES | YES | YES | None needed | Low | Log transform+scale | Med |
| P1 | ML-Compatible Transformations | Basic Speech Content Indicators | Easy | Low | Identifies content style through simple pattern matching on transcript data | Distinguishes tutorial vs casual vs energetic without NLP dependencies | None | YES | NO | NO | One-hot encode 4 categories | Low | Label encode (0-3)+scale | Low | DONE ✅ |
| P1 | Raw Data | Caption Sentiment Analysis | Easy | Low | Critical text emotion signal for engagement prediction | Replaces hardcoded placeholder in emotional_journey | Caption data already available | YES | NO | NO | None needed | Low | Scale to [-1,1] | Low | SKIPPED - Low ML value |
| P1 | Raw Data | Creative Density Climax Moment | Easy | Low | Identifies peak production intensity timing for alignment analysis | Completes climax moment system for coordination patterns | density_per_second calculation | NO | YES | NO | Extract position (0-1) | Low | Scale | Low |
| P1 | ML-Compatible Transformations | Normalize Climax Moments to Position | Easy | Low | Enables cross-video comparison and alignment analysis | Fixes inconsistent formats (strings vs dicts) for ML compatibility | Existing climax calculations | NO | YES | NO | Extract position (0-1) | Low | Scale | Low |
| P1 | Raw Data | Emotion Distribution Ratios | Easy | Low | Complete emotional composition for pattern discovery | Replaces oversimplified dominantEmotion with temporal distributions only | Expression timeline data | NO | YES | NO | None needed | Low | Scale | Low |
| P1 | Raw Data | Temporal Face Size Metrics | Medium | Medium | Captures framing patterns and intimacy progression through video | Replaces global average with temporal window face sizes | Temporal Windows, Face detection | NO | YES | NO | None needed | Low | Scale | Low | SKIPPED - Redundant with framing ratios (RF learns variance) |
| P1 | Raw Data | Temporal Eye Contact Metrics | Medium | Medium | Reveals audience connection patterns throughout video journey | Replaces global average with per-window eye contact rates | Temporal Windows, Eye tracking data | NO | YES | NO | None needed | Low | Scale | Low |
| P1 | Raw Data | Temporal Face Visibility Metrics | Medium | Medium | Shows face presence patterns to identify content strategy | Replaces global average with per-window face visibility rates | Temporal Windows, Face detection | NO | YES | NO | None needed | Low | Scale | Low |
| P1 | Raw Data | Temporal Framing Changes | Easy | Low | Reveals where shot type dynamics occur in video structure | Adds per-window framing change counts | Temporal Windows, Framing progression data | NO | YES | NO | None needed | Low | Scale | Low | ALREADY CAPTURED - Framing ratios encode changes |
| P1 | Raw Data | Temporal Framing Consistency | Easy | Low | Shows stability patterns through video journey | Replaces global with per-window consistency scores | Temporal Windows, framing_volatility calculation | NO | YES | NO | None needed | Low | Scale | Low | ALREADY CAPTURED - Derivable from framing ratios |
| P1 | Raw Data | Temporal Framing Distribution | Easy | Low | Reveals shot composition evolution through video | Replaces global with per-window shot type percentages | Temporal Windows, shot_type_distribution data | NO | YES | NO | None needed | Low | Scale | Low |
| P1 | Raw Data | Temporal Gaze Variance | Easy | Low | Shows eye contact consistency patterns through video | Replaces categorical gazeSteadiness with per-window numerical variance | Temporal Windows, gaze timeline | NO | YES | NO | None needed | Low | Scale | Low | DONE ✅ |
| P1 | Raw Data | Temporal Scene Duration Metrics | Easy | Low | Reveals pacing evolution through video journey | Replaces global with per-window scene duration metrics | Temporal Windows, scene boundaries | DONE | YES | NO | None needed | Low | Scale | Low |
| P1 | Raw Data | Temporal Speech Rhythm Metrics | Easy | Low | Tracks speech delivery patterns through video journey | Adds per-window avg/longest segment durations | Temporal Windows, speech segments | NO | YES | NO | None needed | Low | Scale | Low | SKIPPED - Redundant with speech_coverage/word_count |
| P1 | Raw Data | Temporal Speech Pacing Variation | Easy | Low | Reveals speaking consistency patterns through video | Shows where steady vs variable pacing occurs | Temporal Windows, speech segments | NO | YES | NO | None needed | Low | Scale | Low | SKIPPED - Insufficient samples for variance in 3-10s windows |
| P1 | Raw Data | Temporal Vocabulary Diversity | Easy | Low | Tracks vocabulary richness evolution through video | Shows scripted vs natural speech patterns per window | Temporal Windows, unique/total words per window | NO | YES | NO | None needed | Low | Scale | Low | SKIPPED - Sample too small (10-30 words, need 100+) |
| P1 | Raw Data | Temporal Overlay Metrics (Duration, Variety & Persistence) | Easy | Low | Reveals complete text display strategy through video | Shows reading time allowance, variety patterns, and text persistence per window with min/avg/max durations | Temporal Windows, overlay timestamps | NO | YES | NO | None needed | Low | Scale | Low | DONE |
| P1 | Raw Data | Basic Sticker Metrics | Easy | Low | Captures platform-native visual language usage patterns | Fills gap where stickers are counted but not analyzed separately from text | Temporal Windows, stickerTimeline | YES | YES | YES | None needed | Low | Scale | Low | SKIPPED - HSV sticker detection too unreliable |
| P1 | Raw Data | Expand Generic Hashtag Detection | Easy | Low | More accurate genericRatio calculations for discovery strategy | Expands from 6 to 14 generic hashtags per documentation | None | YES | NO | NO | None needed | Low | Scale | Low | DONE ✅ - Fixed orphaned bug + expanded to 14 |
| P1 | Raw Data | Expand hasHook Pattern Detection | Easy | Low | Better viral hook detection coverage from ~5% to ~40% | Expands from 7 to 50+ proven hook patterns | None | YES | NO | NO | None needed | Low | None needed | None | SKIPPED - Low reliability due to transcript dependency |
| P1 | ML-Compatible Transformations | Simplify ctaFeatures Structure | Easy | Low | Removes redundancy and flattens for ML consumption | Eliminates hasCTA duplicate and derivable ctaCount | None | YES | NO | NO | None needed | Low | Scale | Low | SKIPPED - Same verbatim text matching issues as hook detection |
| P2 | Raw Data | Scene Duration Variance | Easy | Low | Reveals pacing consistency within temporal windows | Complements averageSceneDuration with spread | None | YES | NO | NO | None needed | Low | Scale | Low | DONE |
| P2 | Raw Data | Quiet Period Metrics | Medium | Medium | Captures strategic pauses and cognitive rest patterns | Fixes variable array incompatibility of quietMoments | Temporal Windows | NO | YES | NO | None needed | Low | Scale | Low |
| P2 | Raw Data | Silence Duration Metrics | Easy | Low | Completes pause pattern analysis with duration info | Replaces silencePeriods variable array | None | YES | YES | YES | None needed | Low | Scale | Low | SKIPPED - Insufficient variance in 3-10s windows (0-1 pauses) |
| P2 | Raw Data | Enhanced Emotion Metrics | Easy | Low | Captures emotional complexity without sequence challenges | Adds variety and depth beyond dominant emotion | None | YES | NO | NO | None needed | Low | Scale | Low |
| P2 | Raw Data | Enhanced Gesture Metrics | Easy | Low | Captures gesture diversity and communication style | Adds variety and types beyond simple count | None | YES | NO | NO | None needed | Low | Scale | Low | SKIPPED - Gesture classification broken + minimal variance in 3-10s windows |
| P2 | Raw Data | Enhanced Object Metrics | Medium | Medium | Enables content type classification through objects | Identifies viral niches (pets, food, tech) | None | YES | NO | NO | One-hot encode categories | Med | Label encode+scale | Med |
| P2 | ML-Compatible Transformations | Text Content Classification Metrics | Easy | Low | Distinguishes marketing patterns and content styles through multi-instance classification | Counts text types instead of losing quantity with binary flags | None | YES | NO | NO | None needed | Low | Scale | Low | SKIPPED - Verbatim matching + low variance in 3-10s windows |
| P3 | Raw Data | Speech Segmentation Metrics | Easy | Low | Reveals speaking rhythm and delivery style patterns | Partially overlaps with silentMoments but adds segment perspective | Quiet Period Metrics (P2) | YES | YES | YES | None needed | Low | Scale | Low | SKIPPED - Arbitrary Whisper segmentation + low variance |
| P4 | Transferred - No | accelerationPattern | NONE | NONE | Captures overall pacing patterns across windows | NONE | NONE | YES | NO | NO | One-hot encode 4 categories | Low | Label encode (0-3) + scale | Low |
| P4 | Transferred - No | stdDeviation | NONE | NONE | Measures consistency vs variation in density across entire video | NONE | NONE | YES | NO | NO | None needed | Low | Scale/normalize | Low |
| P4 | Transferred - No | emojiCount | NONE | NONE | Total emoji usage in caption | NONE | NONE | YES | NO | NO | None needed | Low | Scale | Low |
| P4 | Transferred - No | hashtagCount | NONE | NONE | Total number of hashtags used | NONE | NONE | YES | NO | NO | None needed | Low | Scale | Low |
| P4 | Transferred - No | linkPresent | NONE | NONE | Binary flag for external link in caption | NONE | NONE | YES | NO | NO | None needed | Low | None needed | None |
| P4 | Transferred - No | mentionCount | NONE | NONE | Total number of @mentions in caption | NONE | NONE | YES | NO | NO | None needed | Low | Scale | Low |
| P4 | Transferred - No | publishDayOfWeek | NONE | NONE | Day posted (0=Mon, 6=Sun) | NONE | NONE | YES | NO | NO | None needed | Low | Cyclical encoding (sin/cos) | Med |
| P4 | Transferred - No | publishHour | NONE | NONE | Hour posted (0-23) | NONE | NONE | YES | NO | NO | None needed | Low | Cyclical encoding (sin/cos) | Med |
| P4 | Transferred - No | videoDuration | NONE | NONE | Video length in seconds | NONE | NONE | YES | NO | NO | None needed | Low | Log transform + scale | Low |
| P4 | Transferred - No | volatility | NONE | NONE | Normalized variation measure, enables cross-video comparison | NONE | NONE | YES | NO | NO | None needed | Low | None needed | None |
| P4 | Transferred - No | callToAction | NONE | NONE | Binary flag for CTA presence in caption | NONE | NONE | YES | NO | NO | None needed | Low | None needed | None |
| P4 | Transferred - No | wordCount | NONE | NONE | Total words in caption | NONE | NONE | YES | NO | NO | None needed | Low | Scale | Low |
| Umb | Umbrella Entry | totalWords | NONE | NONE | Total number of words detected, content volume metric | | | YES | YES | YES | None needed | Low | Log transform + scale | Low |
| Umb | Umbrella Entry | totalElements | NONE | NONE | Temporal Windows as Single Source of Truth | | | NO | YES | NO | None needed | Low | None needed | Low |
| Umb | Umbrella Entry | sceneChangeCount | NONE | NONE | Temporal Windows as Single Source of Truth | | | NO | YES | NO | None needed | Low | None needed | Low |
| Umb | Umbrella Entry | densityExtremes | NONE | NONE | Per-Window Density Extremes | | | NO | YES | NO | None needed | Low | Scale/normalize | Low |
| Umb | Umbrella Entry | eyeContactRate | NONE | NONE | Temporal Eye Contact Metrics | | | NO | YES | NO | None needed | Low | Scale to [0,1] | Low |
| Umb | Umbrella Entry | faceVisibilityRate | NONE | NONE | Temporal Face Visibility Metrics | | | NO | YES | NO | None needed | Low | Scale to [0,1] | Low |
| Umb | Umbrella Entry | averageFaceSize | NONE | NONE | Temporal Face Size Metrics | | | NO | YES | NO | None needed | Low | Scale | Low |
| Umb | Umbrella Entry | avgSegmentDuration | NONE | NONE | Temporal Speech Rhythm Metrics | | | NO | YES | NO | None needed | Low | Log transform + scale | Low |
| Umb | Umbrella Entry | longestSegment | NONE | NONE | Temporal Speech Rhythm Metrics | | | NO | YES | NO | None needed | Low | Log transform + scale | Low |
| Umb | Umbrella Entry | shortestScene | NONE | NONE | Temporal Scene Duration Metrics | | | YES | NO | NO | None needed | Low | Log transform + scale | Low | DONE |
| Umb | Umbrella Entry | shortestSegment | NONE | NONE | Temporal Speech Rhythm Metrics | | | NO | YES | NO | None needed | Low | Log transform + scale | Low |
| Umb | Umbrella Entry | speechCoverage | NONE | NONE | Temporal Windows as Single Source of Truth | | | NO | YES | NO | None needed | Low | Already [0,1] | Low |
| Umb | Umbrella Entry | averageSceneDuration | NONE | NONE | Temporal Scene Duration Metrics | | | SKIP (deterministic) | YES | NO | None needed | Low | Log transform + scale | Low |
| Umb | Umbrella Entry | longestScene | NONE | NONE | Temporal Scene Duration Metrics | | | YES | NO | NO | None needed | Low | Log transform + scale | Low | DONE |
| Umb | Umbrella Entry | faceSizeVariance | NONE | NONE | Temporal Face Size Metrics | | | NO | YES | NO | None needed | Low | Scale | Low |
| Umb | Umbrella Entry | framingChanges | NONE | NONE | Temporal Framing Changes | | | NO | YES | NO | None needed | Low | Scale | Low |
| Umb | Umbrella Entry | framingDistribution | NONE | NONE | Temporal Framing Distribution | | | NO | YES | NO | None needed (3 values) | Low | Already normalized [0,1] | Low |
| Umb | Umbrella Entry | energyVariance | NONE | NONE | Remove background_noise_ratio Interpretation | | | DONE (raw metric) | NO | NO | None needed | Low | Scale | Low |
| Umb | Umbrella Entry | vocabularyDiversity | NONE | NONE | Temporal Vocabulary Diversity | | | NO | YES | NO | None needed | Low | Already [0,1] | Low |
| Umb | Umbrella Entry | avgOverlayDuration | NONE | NONE | Temporal Overlay Metrics | | | NO | YES | NO | None needed | Low | Scale | Low |
| Umb | Umbrella Entry | uniqueOverlayCount | NONE | NONE | Temporal Overlay Metrics | | | NO | YES | NO | None needed | Low | Scale | Low |

---

## P0: Blocking Issues (Must Fix for MVP)

### Temporal Windows as Single Source of Truth

#### Problem Statement
- Currently have mixed architecture with global counts and window-specific metrics
- Disconnected data makes it hard for ML to learn relationships
- Redundancy between global totals and window sums
- Some features exist outside temporal framework

#### Explanation of Difficulty
- **Medium**: Requires refactoring existing data structure
- Large-scale architectural change affecting multiple flows
- Must ensure backward compatibility during migration
- Need to update all feature extraction pipelines

#### Solution Design
- Move ALL temporal and count metrics into temporal windows
- Remove redundant global counts from features_base
- Derive global totals from window sums when needed
- Ensure piecewise segments in middle window

#### Implementation Details
```python
# Remove from features_base:
- cd_totalElements
- vo_totalOverlays  
- sa_totalWords
- sp_totalScenes
- ej_emotionTransitions
- sa_speechCoverage  # Replace global with temporal versions

# Add to temporal windows:
"hook_window": {
  "hook_text_count": 4,
  "hook_word_count": 12,
  "hook_element_count": 25,  # Sum of: text + stickers + objects + gestures + expressions + scenes
  "hook_scene_count": 2,
  "hook_speech_coverage": 0.67  # 2 seconds of speech in 3-second hook
},
"middle_window": {
  # Overall middle
  "middle_text_count": 2,
  "middle_word_count": 45,
  "middle_element_count": 87,  # Sum of all 6 element types in middle window
  # Piecewise segments (3-5 depending on video length)
  "middle_segment_1_text_count": 1,
  "middle_segment_1_element_count": 28,
  "middle_segment_2_text_count": 0,
  "middle_segment_2_element_count": 31,
  "middle_segment_3_text_count": 1,
  "middle_segment_3_element_count": 28,
  "middle_segment_4_text_count": 2,  # if video > 60s
  "middle_segment_4_element_count": 30,  # if video > 60s
  "middle_segment_5_text_count": 1,   # if video > 90s
  "middle_segment_5_element_count": 25   # if video > 90s
},
"closing_window": {
  "closing_text_count": 3,
  "closing_word_count": 18,
  "closing_element_count": 42,  # Sum of all 6 element types in closing
  "closing_scene_count": 3,
  "closing_speech_coverage": 0.85  # Percentage of closing window with speech
}

# Speech coverage calculation for temporal windows:
def calculate_temporal_speech_coverage(speech_segments, video_duration):
    """Calculate speech coverage percentage per temporal window"""
    
    # Hook (0-3s)
    hook_speech_time = sum(seg['duration'] for seg in speech_segments 
                          if seg['start'] < 3)
    hook_speech_coverage = min(hook_speech_time / 3.0, 1.0)
    
    # Middle window overall (3s to last 3s)
    middle_start = 3
    middle_end = max(6, video_duration - 3)
    middle_speech_time = sum(seg['duration'] for seg in speech_segments 
                            if middle_start <= seg['start'] < middle_end)
    middle_speech_coverage = middle_speech_time / (middle_end - middle_start)
    
    # Middle segments (3-5 based on duration)
    middle_segments = get_middle_segments(video_duration)
    segment_coverages = {}
    for segment_name, (start, end) in middle_segments.items():
        segment_speech_time = sum(seg['duration'] for seg in speech_segments 
                                 if start <= seg['start'] < end)
        segment_coverages[f'middle_{segment_name}_speech_coverage'] = \
            segment_speech_time / (end - start)
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_speech_time = sum(seg['duration'] for seg in speech_segments 
                             if seg['start'] >= closing_start)
    closing_speech_coverage = min(closing_speech_time / 3.0, 1.0)
    
    return {
        'hook_speech_coverage': round(hook_speech_coverage, 2),
        'middle_speech_coverage': round(middle_speech_coverage, 2),
        **{k: round(v, 2) for k, v in segment_coverages.items()},
        'closing_speech_coverage': round(closing_speech_coverage, 2)
    }

# IMPORTANT: element_count must include ALL 6 types for avg_density compatibility:
# 1. Text overlays (OCR-detected)
# 2. Stickers/emojis
# 3. Objects (YOLO-detected)
# 4. Gestures (MediaPipe hand gestures)
# 5. Facial expressions
# 6. Scene changes
```

#### Dependencies
- None - this is the foundational change

---


### Missing shortestScene Metric

#### Problem Statement
- Scene pacing analysis has longestScene but missing shortestScene
- Cannot see pacing floor (fastest cut in video)
- Incomplete min/avg/max trio for scene duration distribution
- Temporal versions exist in P1 but global metric missing

#### Explanation of Difficulty
- **Easy**: Simple calculation from existing scene durations
- Scene durations already calculated for averageSceneDuration
- Just needs to find minimum duration

#### Solution Design
- Calculate minimum of all scene durations
- Add to global scene_pacing metrics
- Completes the min/avg/max trio

#### Implementation Details
```python
# In compute_scene_pacing_metrics (precompute_functions_full.py)
# After calculating avg_scene_duration and longest_scene:

# Calculate shortest scene (pacing floor)
shortest_scene = min(scene_durations) if scene_durations else 0

# Add to returned metrics:
metrics = {
    'avg_scene_duration': avg_scene_duration,
    'longest_scene': longest_scene,
    'shortest_scene': shortest_scene,  # ADD THIS LINE
    'scenes_per_minute': scenes_per_minute,
    # ... rest of metrics
}
```

#### Dependencies
- None - uses existing scene_durations list

---

### Remove background_noise_ratio Interpretation

#### Problem Statement
- energyVariance is being misused to derive background_noise_ratio
- Assumes high variance = background noise (flawed logic)
- Violates "let ML discover patterns" principle
- Pre-computed interpretation that could be wrong

#### Explanation of Difficulty
- **Easy**: Simple removal of derived metric
- Keep energyVariance as raw metric
- Remove background_noise_ratio calculation

#### Solution Design
- Remove background_noise_ratio derivation
- Keep energyVariance as pure statistical measure
- Let ML discover what variance patterns mean

#### Implementation Details
```python
# In compute_speech_analysis_metrics (precompute_functions_full.py)
# REMOVE this entire block (lines ~2935-2944):

# Background noise from audio energy variance (high variance = more noise variation)
# Low energy variance (< 0.01) suggests consistent audio = less background noise
# if energy_variance is not None and energy_variance > 0:
#     # Map energy variance to noise ratio (inverse relationship)
#     # Low variance = low noise, high variance = high noise
#     background_noise_ratio = min(1.0, energy_variance * 100)  # Scale to 0-1
# else:
#     background_noise_ratio = 0.0  # No data available
# 
# metrics['background_noise_ratio'] = round(background_noise_ratio, 3)

# KEEP only:
metrics['energy_variance'] = float(energy_variance)
```

#### Dependencies
- None - removal only

---

### Missing pacingVariation Implementation

#### Problem Statement
- pacingVariation referenced in professional wrapper but never calculated
- Missing critical speaking rhythm consistency metric
- Unable to detect steady vs variable speaking patterns
- Currently returns 0 for all videos

#### Explanation of Difficulty
- **Easy**: Calculate standard deviation of WPM across segments
- Speech segments already exist with word counts
- Simple statistical calculation

#### Solution Design
- Calculate WPM for each speech segment
- Compute standard deviation of WPM values
- Return as pacing_std metric

#### Implementation Details
```python
# In compute_speech_analysis_metrics (precompute_functions_full.py)
# After calculating speech segments:

def calculate_pacing_variation(speech_segments):
    """Calculate standard deviation of speaking speed changes"""
    
    if not speech_segments or len(speech_segments) < 2:
        return 0
    
    # Calculate WPM for each segment
    segment_wpms = []
    for segment in speech_segments:
        duration = segment.get('duration', 0)
        word_count = len(segment.get('text', '').split())
        
        if duration > 0:
            wpm = (word_count / duration) * 60
            segment_wpms.append(wpm)
    
    # Calculate standard deviation
    if len(segment_wpms) > 1:
        pacing_std = np.std(segment_wpms)
    else:
        pacing_std = 0
    
    return round(pacing_std, 2)

# Add to metrics:
metrics['pacing_std'] = calculate_pacing_variation(speech_segments)
```

#### Dependencies
- Speech segments with duration and text

---

### Multimodal Counts in Windows ✅ DONE

#### Problem Statement
- ML cannot discover text-speech-gesture correlations
- Missing modality data in temporal windows
- Must rely on pre-computed features like multimodalMoments
- Cannot identify multimodal patterns by window

#### Explanation of Difficulty
- **Easy**: Simple counting within existing timelines
- Data already exists in processing pipeline
- Just needs aggregation to windows and segments
- Low risk of breaking existing features

#### Solution Design
- Add modality counts to each temporal window
- Include counts in piecewise segments for middle
- Use binary presence indicators where appropriate
- Defer sync metrics to Phase 2

#### Implementation Details
```python
# Add to each window:
"hook_window": {
  "hook_speech_coverage": 0.67,  # Changed from speech_present to match implementation
  "hook_word_count": 12,  # Changed from speech_words to match implementation
  "hook_gesture_count": 2,
  "hook_text_count": 4  # Already exists
},
"middle_window": {
  # Overall counts
  "middle_speech_coverage": 0.75,
  "middle_gesture_count": 5,
  # Piecewise segments (3-5 based on duration)
  "middle_segment_1_speech_coverage": 0.8,
  "middle_segment_2_speech_coverage": 0.9,
  "middle_segment_3_speech_coverage": 0.5,
  "middle_segment_4_speech_coverage": 0.7,  # if exists
  "middle_segment_5_speech_coverage": 0.4   # if exists
}
```

#### Dependencies
- Temporal Windows architecture must be complete

---

### Overlay Counts in Windows ✅ DONE

#### Problem Statement
- Global overlay counts (totalOverlays, totalStickers, totalTextOverlays) exist outside temporal framework
- Creates redundancy with window-specific counts
- ML receives mixed signals about where to find overlay information
- Cannot discover distribution patterns from global totals

#### Explanation of Difficulty
- **Easy**: Simple counting within existing overlay timeline
- Overlay detection already complete
- Just needs aggregation to windows and segments
- Direct summation with no complex logic

#### Solution Design
- Move ALL overlay counts into temporal windows
- Separate text overlays from stickers for granularity
- Calculate totals through window summation
- Remove global counts from features_base

#### Implementation Details
```python
# Remove from features_base:
- vo_totalOverlays
- vo_totalStickers  
- vo_totalTextOverlays

# Add to temporal windows (IMPLEMENTED):
"hook_window": {
  "hook_text_count": 3,       # Text overlays
  "hook_sticker_count": 1     # Stickers
  # NOTE: overlay_count omitted to avoid deterministic feature (sum of above)
},
"middle_window": {
  # Overall middle counts
  "middle_overlay_count": 8,
  "middle_text_overlay_count": 5,
  "middle_sticker_count": 3,
  # Piecewise segments (3-5 based on duration)
  "middle_segment_1_overlay_count": 2,
  "middle_segment_2_overlay_count": 3,
  "middle_segment_3_overlay_count": 3,
  # Repeat for text and stickers separately
}

# ML derives totals:
total_overlays = sum(all_window_overlay_counts)
total_text = sum(all_window_text_counts)
total_stickers = sum(all_window_sticker_counts)
```

#### Dependencies
- Temporal Windows architecture must be complete

---


---

### Per-Window Density Extremes

#### Problem Statement
- Global maxDensity and minDensity lose critical timing information
- Can't distinguish hook spike from climactic middle peak
- Can't tell if minimum is strategic pause or dead spot
- Same extreme values could mean different patterns based on placement
- ML needs temporal context to understand intensity variation patterns

#### Explanation of Difficulty
- **Easy**: Simple max() and min() operations on existing per-second densities
- Already calculating density_per_second in creative_density
- Just need to slice by temporal windows and find extremes
- No new data extraction required

#### Solution Design
- Calculate max AND min density for each temporal window
- Provides peak and floor intensity WITH timing context
- Enables pattern discovery (consistent vs variable, crescendo vs front-loaded)
- Reveals production consistency and strategic pauses
- Small data addition (two values per window)

#### Implementation Details
```python
# Add to temporal windows after density calculation:
"hook_window": {
  "hook_element_count": 25,      # Existing
  "hook_max_density": 12,        # NEW: Peak second in hook (elements/second)
  "hook_min_density": 2          # NEW: Floor second in hook
}

"middle_window": {
  "middle_element_count": 87,    # Existing
  "middle_max_density": 32,      # NEW: Peak second in middle
  "middle_min_density": 0        # NEW: Has empty second (strategic pause?)
}

"middle_segments": {
  "middle_segment_1_element_count": 28,
  "middle_segment_1_max_density": 15,     # NEW: Peak in segment 1
  "middle_segment_1_min_density": 0,      # NEW: Has pause in segment 1
  
  "middle_segment_2_element_count": 31,
  "middle_segment_2_max_density": 32,     # NEW: Spike located here!
  "middle_segment_2_min_density": 4,      # NEW: Maintains baseline activity
  
  "middle_segment_3_element_count": 28,
  "middle_segment_3_max_density": 10,     # NEW: Peak in segment 3
  "middle_segment_3_min_density": 2       # NEW: Floor in segment 3
}

"closing_window": {
  "closing_element_count": 42,   # Existing
  "closing_max_density": 18,     # NEW: Peak second in closing
  "closing_min_density": 8       # NEW: Strong sustained finish (no drops)
}

# Implementation in precompute_creative_density.py:
# We already have: density_per_second = [5, 9, 6, 2, 0, 4, ...]

hook_max_density = max(density_per_second[0:3])
hook_min_density = min(density_per_second[0:3])
middle_max_density = max(density_per_second[3:-3])
middle_min_density = min(density_per_second[3:-3])
closing_max_density = max(density_per_second[-3:])
closing_min_density = min(density_per_second[-3:])

# For middle segments (assuming 3 segments for 18s video):
segment_size = 4  # (18-6)/3 = 4 seconds per segment
middle_segment_1_max = max(density_per_second[3:7])
middle_segment_1_min = min(density_per_second[3:7])
middle_segment_2_max = max(density_per_second[7:11])
middle_segment_2_min = min(density_per_second[7:11])
middle_segment_3_max = max(density_per_second[11:15])
middle_segment_3_min = min(density_per_second[11:15])
```

#### Value for ML
- **Timing context**: Distinguishes opening hook spike from climactic peak
- **Pattern discovery**: ML can learn crescendo vs steady vs explosive patterns
- **Consistency metrics**: (max - min) per window shows variability vs consistency
- **Engagement signals**: 
  - High hook_max → attention grab
  - Low middle_min → breathing room
  - High closing_min → sustained energy finish
- **Production quality**: 
  - Consistent min across windows → maintained baseline production
  - Variable min → strategic use of pauses

#### Notes from Analysis
- Global maxDensity/minDensity alone is like knowing extremes without context
- Video A and B could have same extremes but different engagement based on placement
- Per-window extremes preserve WHERE peaks and valleys occur
- Replaces creative_density.maxDensity and minDensity with temporally-aware metrics
- Architecturally perfect fit: follows temporal window structure, no redundancy

#### Dependencies
- Temporal Windows must be implemented
- density_per_second calculation must exist

---

## P1: High-Value Improvements

### Multi-person Metrics

#### Problem Statement
- Current subjectCount is broken (returns MAX not AVG)
- Cannot track multi-person dynamics in windows
- Missing collaboration patterns crucial for viral content
- No visibility into group vs solo content strategies

#### Explanation of Difficulty
- **Medium**: Requires reliable face detection across frames
- Must handle detection failures and partial faces
- Need to aggregate across variable timestamps
- Edge cases like crowds and posters

#### Solution Design
- Create comprehensive multiPersonRate metric
- Calculate for each temporal window and segment
- Include average, max, and distribution metrics
- Replace broken subjectCount entirely

#### Implementation Details
```python
def calculate_multi_person_metrics(face_detection_timeline):
    faces_per_frame = []
    for timestamp, data in face_detection_timeline.items():
        face_count = len(data.get('faces', []))
        faces_per_frame.append(face_count)
    
    return {
        'multiPersonRate': np.mean([1 if f > 1 else 0 for f in faces_per_frame]),
        'avgPersonCount': np.mean(faces_per_frame),
        'maxPersonCount': max(faces_per_frame) if faces_per_frame else 0,
        'soloRate': np.mean([1 if f == 1 else 0 for f in faces_per_frame]),
        'emptyRate': np.mean([1 if f == 0 else 0 for f in faces_per_frame])
    }

# Apply to windows:
"hook_multiPersonRate": 0.25,
"middle_segment_1_multiPersonRate": 0.60,
"middle_segment_2_multiPersonRate": 0.45,
"middle_segment_3_multiPersonRate": 0.30,
"middle_segment_4_multiPersonRate": 0.25,  # if exists
"middle_segment_5_multiPersonRate": 0.20,  # if exists
"closing_multiPersonRate": 0.10
```

#### Dependencies
- None - independent metric

---

### Audio Energy Metrics (avg, peaks) - PARTIALLY DONE

#### Problem Statement
- Have energyVariance but missing average energy level
- Cannot identify emphatic moments without peak detection
- Incomplete picture of speech dynamics
- Cannot replace semantic features like burstPattern

#### Implementation Status
- **Average Energy**: ✅ DONE - Already implemented as `energy_level` (which is np.mean of RMS frames)
- **Energy Peaks**: ❌ SKIPPED - Not compatible with 3-10s temporal windows

#### Reason for Skipping Peaks
With temporal windows of 3 seconds (hook/closing) and 3-10 seconds (middle segments), peak counts would:
1. Be mostly 0-1 per window (low signal)
2. Scale with duration rather than content (biased metric)
3. Duplicate information already captured by energy_variance

The existing metrics provide complete audio dynamics:
- `energy_level`: Average loudness (already implemented)
- `energy_variance`: Variation in emphasis
- `energy_max`: Peak intensity
- `burst_pattern`: Where energy concentrates (front/middle/back)

#### Dependencies
- None - uses existing audio extraction

---

### Temporal Speech Rhythm Metrics [SKIPPED - Redundant with speech_coverage/word_count]

#### Problem Statement
- Currently only have global avgSegmentDuration across entire video
- Cannot see how speech delivery patterns evolve through video journey
- Missing visibility into hook speaking rhythm vs body vs closing pace
- No insight into whether speaker accelerates or maintains steady rhythm

#### Resolution - SKIPPED
**Not implementing - redundant with existing features:**
- Speech rhythm is already captured by `speech_coverage` + `word_count`
- Average segment duration measures Whisper's arbitrary segmentation, not actual rhythm
- Real speech pace = word_count / (duration * speech_coverage)
- In 3-7s windows, only 1-2 segments = statistically meaningless averages
- ML can derive all rhythm patterns from existing features

#### Original Explanation of Difficulty
- **Easy**: Simple calculation of average segment duration per window
- Speech segments already tracked with timestamps
- Just needs aggregation to temporal windows
- Straightforward statistical calculations

#### Solution Design
- Calculate average speech segment duration for each temporal window
- Replace global avgSegmentDuration with temporal versions to avoid correlation
- Track hook, middle, and closing speech rhythm patterns
- Identify acceleration/deceleration patterns
- Enables discovery of speech pacing strategies

#### Implementation Details
```python
# In compute_speech_analysis_metrics (precompute_functions_full.py)
# After current avg_segment_duration calculation:

def calculate_temporal_speech_rhythm(speech_segments, video_duration):
    """Calculate speech rhythm metrics per temporal window"""
    
    temporal_metrics = {}
    
    # Hook (0-3s)
    hook_segments = [s for s in speech_segments if s['start'] < 3]
    if hook_segments:
        hook_durations = [s['duration'] for s in hook_segments]
        temporal_metrics['hook_avg_segment_duration'] = round(np.mean(hook_durations), 2)
        temporal_metrics['hook_longest_segment'] = round(max(hook_durations), 2)
        temporal_metrics['hook_shortest_segment'] = round(min(hook_durations), 2)
        temporal_metrics['hook_segment_count'] = len(hook_segments)
    else:
        temporal_metrics['hook_avg_segment_duration'] = 0
        temporal_metrics['hook_longest_segment'] = 0
        temporal_metrics['hook_shortest_segment'] = 0
        temporal_metrics['hook_segment_count'] = 0
    
    # Middle segments (3-5 based on duration)
    middle_segments = get_middle_segments(video_duration)
    for i, (start, end, segment_name) in enumerate(middle_segments):
        segment_speeches = [s for s in speech_segments 
                          if start <= s['start'] < end]
        if segment_speeches:
            segment_durations = [s['duration'] for s in segment_speeches]
            temporal_metrics[f'middle_{segment_name}_avg_segment_duration'] = round(np.mean(segment_durations), 2)
            temporal_metrics[f'middle_{segment_name}_longest_segment'] = round(max(segment_durations), 2)
            temporal_metrics[f'middle_{segment_name}_shortest_segment'] = round(min(segment_durations), 2)
            temporal_metrics[f'middle_{segment_name}_segment_count'] = len(segment_speeches)
        else:
            temporal_metrics[f'middle_{segment_name}_avg_segment_duration'] = 0
            temporal_metrics[f'middle_{segment_name}_longest_segment'] = 0
            temporal_metrics[f'middle_{segment_name}_shortest_segment'] = 0
            temporal_metrics[f'middle_{segment_name}_segment_count'] = 0
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_segments = [s for s in speech_segments if s['start'] >= closing_start]
    if closing_segments:
        closing_durations = [s['duration'] for s in closing_segments]
        temporal_metrics['closing_avg_segment_duration'] = round(np.mean(closing_durations), 2)
        temporal_metrics['closing_longest_segment'] = round(max(closing_durations), 2)
        temporal_metrics['closing_shortest_segment'] = round(min(closing_durations), 2)
        temporal_metrics['closing_segment_count'] = len(closing_segments)
    else:
        temporal_metrics['closing_avg_segment_duration'] = 0
        temporal_metrics['closing_longest_segment'] = 0
        temporal_metrics['closing_shortest_segment'] = 0
        temporal_metrics['closing_segment_count'] = 0
    
    # Calculate rhythm pattern (steady vs variable)
    hook_rhythm = temporal_metrics['hook_avg_segment_duration']
    closing_rhythm = temporal_metrics['closing_avg_segment_duration']
    
    if hook_rhythm > 0 and closing_rhythm > 0:
        rhythm_ratio = closing_rhythm / hook_rhythm
        if rhythm_ratio < 0.7:
            temporal_metrics['speech_rhythm_pattern'] = 'accelerating'
        elif rhythm_ratio > 1.3:
            temporal_metrics['speech_rhythm_pattern'] = 'decelerating'
        else:
            temporal_metrics['speech_rhythm_pattern'] = 'steady'
    else:
        temporal_metrics['speech_rhythm_pattern'] = 'sparse'
    
    return temporal_metrics

# Example output:
"hook_avg_segment_duration": 1.2,
"hook_longest_segment": 2.1,
"hook_shortest_segment": 0.5,
"hook_segment_count": 2,
"middle_segment_1_avg_segment_duration": 0.8,
"middle_segment_1_longest_segment": 1.5,
"middle_segment_1_shortest_segment": 0.3,
"middle_segment_1_segment_count": 4,
"closing_avg_segment_duration": 0.6,
"closing_longest_segment": 0.9,
"closing_shortest_segment": 0.2,
"closing_segment_count": 3,
"speech_rhythm_pattern": "accelerating"
```

#### Dependencies
- Speech segments with accurate timestamps
- Temporal window architecture (P0)

---

### Temporal Speech Pacing Variation [SKIPPED - Insufficient samples for variance]

#### Problem Statement
- Currently only have global pacingVariation (once P0 implemented)
- Cannot see where speaking consistency changes through video
- Missing visibility into steady hooks vs variable middles vs climactic endings
- No insight into pacing strategy evolution

#### Resolution - SKIPPED
**Not implementing - statistically flawed for our window sizes:**
- Variance requires ≥5 samples for meaningful calculation
- 3-second windows have only 1-2 speech segments
- 7-second windows have only 2-3 speech segments
- Standard deviation of 1-3 values is statistically meaningless
- Pacing changes already visible from word_count differences between windows
- Rule: Variation only works when `Event Frequency × Window Duration ≥ 5`

#### Original Explanation of Difficulty
- **Easy**: Calculate standard deviation of WPM per temporal window
- Speech segments already have timestamps and word counts
- Statistical calculation per window

#### Solution Design
- Calculate pacing variation for each temporal window
- Track hook, middle, and closing consistency patterns
- Identify where speakers maintain steady vs variable pacing

#### Implementation Details
```python
# In compute_speech_analysis_metrics (precompute_functions_full.py)
# After global pacing_std calculation:

def calculate_temporal_pacing_variation(speech_segments, video_duration):
    """Calculate pacing variation per temporal window"""
    
    temporal_metrics = {}
    
    # Hook (0-3s)
    hook_segments = [s for s in speech_segments if s['start'] < 3]
    hook_wpms = []
    for segment in hook_segments:
        duration = segment.get('duration', 0)
        word_count = len(segment.get('text', '').split())
        if duration > 0:
            hook_wpms.append((word_count / duration) * 60)
    
    temporal_metrics['hook_pacing_variation'] = round(np.std(hook_wpms), 2) if len(hook_wpms) > 1 else 0
    
    # Middle segments (3-5 based on duration)
    middle_segments = get_middle_segments(video_duration)
    for i, (start, end, segment_name) in enumerate(middle_segments):
        segment_speeches = [s for s in speech_segments 
                          if start <= s['start'] < end]
        segment_wpms = []
        for segment in segment_speeches:
            duration = segment.get('duration', 0)
            word_count = len(segment.get('text', '').split())
            if duration > 0:
                segment_wpms.append((word_count / duration) * 60)
        
        temporal_metrics[f'middle_{segment_name}_pacing_variation'] = round(np.std(segment_wpms), 2) if len(segment_wpms) > 1 else 0
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_segments = [s for s in speech_segments if s['start'] >= closing_start]
    closing_wpms = []
    for segment in closing_segments:
        duration = segment.get('duration', 0)
        word_count = len(segment.get('text', '').split())
        if duration > 0:
            closing_wpms.append((word_count / duration) * 60)
    
    temporal_metrics['closing_pacing_variation'] = round(np.std(closing_wpms), 2) if len(closing_wpms) > 1 else 0
    
    # Identify consistency pattern
    hook_var = temporal_metrics['hook_pacing_variation']
    closing_var = temporal_metrics['closing_pacing_variation']
    
    if hook_var < 10 and closing_var < 10:
        temporal_metrics['pacing_consistency_pattern'] = 'steady_throughout'
    elif hook_var < 10 and closing_var > 20:
        temporal_metrics['pacing_consistency_pattern'] = 'steady_to_variable'
    elif hook_var > 20 and closing_var < 10:
        temporal_metrics['pacing_consistency_pattern'] = 'variable_to_steady'
    else:
        temporal_metrics['pacing_consistency_pattern'] = 'variable_throughout'
    
    return temporal_metrics

# Example output:
"hook_pacing_variation": 5.2,  # Steady opening
"middle_segment_1_pacing_variation": 12.5,  # Building variety
"middle_segment_2_pacing_variation": 25.3,  # Peak variation
"closing_pacing_variation": 8.1,  # Settling down
"pacing_consistency_pattern": "steady_to_variable"
```

#### Dependencies
- Speech segments with accurate timestamps
- Global pacingVariation implementation (P0)
- Temporal window architecture (P0)

---

### Temporal Vocabulary Diversity [SKIPPED - Sample too small]

#### Problem Statement
- Currently only have global vocabularyDiversity metric
- Cannot track vocabulary richness evolution through video
- Missing visibility into scripted vs natural speech patterns per window
- No insight into where vocabulary complexity changes (simple hook, complex middle, etc.)

#### Resolution - SKIPPED
**Not implementing - sample size too small for meaningful metric:**
- Vocabulary diversity requires 100+ words for statistical stability (Zipf's Law)
- Hook has only ~11 words, segments have ~30 words
- Unique/total ratio swings wildly with each word at this scale
- Example: 10 words all unique = 1.0, add one repeat = 0.91 (10% swing from 1 word!)
- Word count already captures vocabulary opportunity implicitly
- Would produce noise, not signal

#### Original Explanation of Difficulty
- **Easy**: Calculate unique/total word ratio per temporal window
- Word counts already available from speech transcripts
- Simple ratio calculation per window

#### Solution Design
- Calculate vocabulary diversity for each temporal window
- Track uniqueWords and totalWords per window
- Compute ratio to show vocabulary richness progression
- Identify patterns like "simple-to-complex" or "rich-throughout"

#### Implementation Details
```python
# In compute_speech_analysis_metrics (precompute_functions_full.py)
# After global vocabulary_diversity calculation:

def calculate_temporal_vocabulary_diversity(speech_segments, video_duration):
    """Calculate vocabulary diversity per temporal window"""
    
    temporal_metrics = {}
    
    # Hook (0-3s)
    hook_segments = [s for s in speech_segments if s['start'] < 3]
    hook_words = []
    for segment in hook_segments:
        hook_words.extend(segment.get('text', '').lower().split())
    
    if hook_words:
        hook_unique = len(set(hook_words))
        hook_total = len(hook_words)
        temporal_metrics['hook_vocabulary_diversity'] = round(hook_unique / hook_total, 3)
        temporal_metrics['hook_unique_words'] = hook_unique
        temporal_metrics['hook_total_words'] = hook_total
    else:
        temporal_metrics['hook_vocabulary_diversity'] = 0
        temporal_metrics['hook_unique_words'] = 0
        temporal_metrics['hook_total_words'] = 0
    
    # Middle segments (3-5 segments based on duration)
    middle_start = 3
    middle_end = video_duration - 3
    num_segments = 3 if video_duration <= 18 else (4 if video_duration <= 33 else 5)
    segment_duration = (middle_end - middle_start) / num_segments
    
    for i in range(num_segments):
        seg_start = middle_start + (i * segment_duration)
        seg_end = seg_start + segment_duration
        segment_words = []
        
        for segment in speech_segments:
            if seg_start <= segment['start'] < seg_end:
                segment_words.extend(segment.get('text', '').lower().split())
        
        if segment_words:
            unique = len(set(segment_words))
            total = len(segment_words)
            temporal_metrics[f'middle_segment_{i+1}_vocabulary_diversity'] = round(unique / total, 3)
            temporal_metrics[f'middle_segment_{i+1}_unique_words'] = unique
            temporal_metrics[f'middle_segment_{i+1}_total_words'] = total
        else:
            temporal_metrics[f'middle_segment_{i+1}_vocabulary_diversity'] = 0
            temporal_metrics[f'middle_segment_{i+1}_unique_words'] = 0
            temporal_metrics[f'middle_segment_{i+1}_total_words'] = 0
    
    # Closing (last 3s)
    closing_segments = [s for s in speech_segments if s['start'] >= video_duration - 3]
    closing_words = []
    for segment in closing_segments:
        closing_words.extend(segment.get('text', '').lower().split())
    
    if closing_words:
        closing_unique = len(set(closing_words))
        closing_total = len(closing_words)
        temporal_metrics['closing_vocabulary_diversity'] = round(closing_unique / closing_total, 3)
        temporal_metrics['closing_unique_words'] = closing_unique
        temporal_metrics['closing_total_words'] = closing_total
    else:
        temporal_metrics['closing_vocabulary_diversity'] = 0
        temporal_metrics['closing_unique_words'] = 0
        temporal_metrics['closing_total_words'] = 0
    
    # Identify vocabulary pattern
    hook_div = temporal_metrics['hook_vocabulary_diversity']
    closing_div = temporal_metrics['closing_vocabulary_diversity']
    
    if hook_div > 0.7 and closing_div > 0.7:
        temporal_metrics['vocabulary_pattern'] = 'rich_throughout'
    elif hook_div < 0.5 and closing_div > 0.7:
        temporal_metrics['vocabulary_pattern'] = 'simple_to_complex'
    elif hook_div > 0.7 and closing_div < 0.5:
        temporal_metrics['vocabulary_pattern'] = 'complex_to_simple'
    elif hook_div < 0.5 and closing_div < 0.5:
        temporal_metrics['vocabulary_pattern'] = 'simple_throughout'
    else:
        temporal_metrics['vocabulary_pattern'] = 'variable'
    
    return temporal_metrics

# Example output:
"hook_vocabulary_diversity": 0.450,  # Simple, repetitive hook
"hook_unique_words": 9,
"hook_total_words": 20,
"middle_segment_1_vocabulary_diversity": 0.625,
"middle_segment_2_vocabulary_diversity": 0.710,  # Richer vocabulary
"closing_vocabulary_diversity": 0.520,
"vocabulary_pattern": "simple_to_complex"
```

#### Value for ML
- Distinguishes scripted (low diversity) from natural speech (high diversity)
- Identifies educational content (high diversity) vs casual/viral (lower diversity)
- Reveals content complexity progression strategy
- Enables discovery of optimal vocabulary patterns for engagement

#### Notes from Analysis
- Type-Token Ratio is a standard linguistic metric for vocabulary richness
- Pre-normalized to [0,1] for cross-video comparison
- Temporal version reveals where complexity changes occur
- Can identify "dumbing down" for broader audience vs maintaining sophistication

#### Dependencies
- Speech transcript with word segmentation
- Temporal window architecture (P0)
- Global vocabularyDiversity already implemented

---

### Temporal Overlay Metrics (Duration, Variety & Persistence) - PARTIALLY DONE

#### Problem Statement
- Currently only have global avgOverlayDuration and uniqueOverlayCount metrics
- Cannot track reading time strategy evolution through video
- Missing visibility into hook quick-flash vs body sustained text patterns
- No insight into where creators give viewers processing time

#### Implementation Status
✅ **Already Implemented:**
- `overlay_unique_count`: Number of unique overlays (variety metric)
- `overlay_coverage`: Percentage of time with overlays visible
- `overlay_persistence`: Average lifespan of overlays (avg duration)
- `has_captions`: Binary caption presence

❌ **SKIPPED (Incompatible with temporal windows):**
- `min_overlay_duration`: Would often equal max in windows with 1-2 overlays
- `max_overlay_duration`: Capped by window size, limited variance
- `avg_overlay_duration`: Redundant with existing `overlay_persistence`

#### Reason for Skipping Additional Metrics
In 3-10 second windows:
1. **Sparse data**: Most windows have 0-2 overlays
2. **No variance**: With 1 overlay, min = max = avg
3. **Duration cap**: Max duration limited by window size (3s window → max 3s overlay)
4. **Already captured**: `overlay_persistence` already provides average duration

The existing metrics (`overlay_unique_count`, `overlay_coverage`, `overlay_persistence`) provide comprehensive overlay analysis appropriate for our temporal window structure.

#### Implementation Details
```python
# In compute_visual_overlay_metrics (precompute_functions_full.py)
# After global avg_text_display_duration calculation:

def calculate_temporal_overlay_metrics(text_overlay_timeline, video_duration):
    """Calculate comprehensive overlay metrics per temporal window"""
    
    temporal_metrics = {}
    
    # Parse all overlay durations with timestamps
    overlay_durations = []
    for timestamp, data in text_overlay_timeline.items():
        try:
            parts = timestamp.split('-')
            if len(parts) == 2:
                start = float(parts[0])
                end = float(parts[1].replace('s', ''))
                duration = end - start
                overlay_durations.append({
                    'start': start,
                    'duration': duration
                })
        except:
            pass
    
    # Parse overlay texts with timestamps for unique counting
    overlay_texts = []
    for timestamp, data in text_overlay_timeline.items():
        text = data.get('text', '')
        if text:
            try:
                start = float(timestamp.split('-')[0])
                overlay_texts.append({
                    'start': start,
                    'text': text.lower().strip()
                })
            except:
                pass
    
    # Hook (0-3s)
    hook_durations = [o['duration'] for o in overlay_durations if o['start'] < 3]
    hook_texts = [o['text'] for o in overlay_texts if o['start'] < 3]
    hook_unique_texts = set(hook_texts)
    
    # Calculate text persistence (total time text visible / window duration)
    hook_text_visible_time = sum(hook_durations)
    hook_persistence_ratio = min(hook_text_visible_time / 3.0, 1.0)  # Cap at 1.0 for overlapping text
    
    if hook_durations:
        temporal_metrics['hook_avg_overlay_duration'] = round(np.mean(hook_durations), 2)
        temporal_metrics['hook_min_overlay_duration'] = round(min(hook_durations), 2)
        temporal_metrics['hook_max_overlay_duration'] = round(max(hook_durations), 2)
        temporal_metrics['hook_text_persistence_ratio'] = round(hook_persistence_ratio, 2)
        temporal_metrics['hook_overlay_count'] = len(hook_durations)
        temporal_metrics['hook_unique_overlay_count'] = len(hook_unique_texts)
        temporal_metrics['hook_repetition_ratio'] = round(1 - (len(hook_unique_texts) / len(hook_texts)), 2) if hook_texts else 0
    else:
        temporal_metrics['hook_avg_overlay_duration'] = 0
        temporal_metrics['hook_min_overlay_duration'] = 0
        temporal_metrics['hook_max_overlay_duration'] = 0
        temporal_metrics['hook_text_persistence_ratio'] = 0
        temporal_metrics['hook_overlay_count'] = 0
        temporal_metrics['hook_unique_overlay_count'] = 0
        temporal_metrics['hook_repetition_ratio'] = 0
    
    # Middle segments (3-5 segments based on duration)
    middle_start = 3
    middle_end = video_duration - 3
    num_segments = 3 if video_duration <= 18 else (4 if video_duration <= 33 else 5)
    segment_duration = (middle_end - middle_start) / num_segments
    
    for i in range(num_segments):
        seg_start = middle_start + (i * segment_duration)
        seg_end = seg_start + segment_duration
        
        segment_durations = [o['duration'] for o in overlay_durations 
                            if seg_start <= o['start'] < seg_end]
        segment_texts = [o['text'] for o in overlay_texts
                        if seg_start <= o['start'] < seg_end]
        segment_unique_texts = set(segment_texts)
        
        if segment_durations:
            temporal_metrics[f'middle_segment_{i+1}_avg_overlay_duration'] = round(np.mean(segment_durations), 2)
            temporal_metrics[f'middle_segment_{i+1}_overlay_count'] = len(segment_durations)
            temporal_metrics[f'middle_segment_{i+1}_unique_overlay_count'] = len(segment_unique_texts)
            temporal_metrics[f'middle_segment_{i+1}_repetition_ratio'] = round(1 - (len(segment_unique_texts) / len(segment_texts)), 2) if segment_texts else 0
        else:
            temporal_metrics[f'middle_segment_{i+1}_avg_overlay_duration'] = 0
            temporal_metrics[f'middle_segment_{i+1}_overlay_count'] = 0
            temporal_metrics[f'middle_segment_{i+1}_unique_overlay_count'] = 0
            temporal_metrics[f'middle_segment_{i+1}_repetition_ratio'] = 0
    
    # Closing (last 3s)
    closing_durations = [o['duration'] for o in overlay_durations 
                         if o['start'] >= video_duration - 3]
    closing_texts = [o['text'] for o in overlay_texts if o['start'] >= video_duration - 3]
    closing_unique_texts = set(closing_texts)
    
    if closing_durations:
        temporal_metrics['closing_avg_overlay_duration'] = round(np.mean(closing_durations), 2)
        temporal_metrics['closing_min_overlay_duration'] = round(min(closing_durations), 2)
        temporal_metrics['closing_max_overlay_duration'] = round(max(closing_durations), 2)
        temporal_metrics['closing_overlay_count'] = len(closing_durations)
        temporal_metrics['closing_unique_overlay_count'] = len(closing_unique_texts)
        temporal_metrics['closing_repetition_ratio'] = round(1 - (len(closing_unique_texts) / len(closing_texts)), 2) if closing_texts else 0
    else:
        temporal_metrics['closing_avg_overlay_duration'] = 0
        temporal_metrics['closing_min_overlay_duration'] = 0
        temporal_metrics['closing_max_overlay_duration'] = 0
        temporal_metrics['closing_overlay_count'] = 0
        temporal_metrics['closing_unique_overlay_count'] = 0
        temporal_metrics['closing_repetition_ratio'] = 0
    
    # Identify reading time pattern
    hook_avg = temporal_metrics['hook_avg_overlay_duration']
    closing_avg = temporal_metrics['closing_avg_overlay_duration']
    
    if hook_avg < 1.0 and closing_avg > 2.0:
        temporal_metrics['reading_time_pattern'] = 'quick_hook_sustained_cta'
    elif hook_avg > 2.0 and closing_avg < 1.0:
        temporal_metrics['reading_time_pattern'] = 'clear_hook_rapid_close'
    elif hook_avg < 1.0 and closing_avg < 1.0:
        temporal_metrics['reading_time_pattern'] = 'rapid_throughout'
    elif hook_avg > 2.0 and closing_avg > 2.0:
        temporal_metrics['reading_time_pattern'] = 'sustained_throughout'
    else:
        temporal_metrics['reading_time_pattern'] = 'variable'
    
    return temporal_metrics

# Example output:
"hook_avg_overlay_duration": 0.8,  # Quick flash text
"hook_overlay_count": 3,
"hook_unique_overlay_count": 3,  # All different messages
"hook_repetition_ratio": 0.0,  # No repetition
"middle_segment_1_avg_overlay_duration": 1.5,
"middle_segment_1_unique_overlay_count": 2,  # Some repetition
"middle_segment_1_repetition_ratio": 0.33,
"middle_segment_2_avg_overlay_duration": 2.1,  # Giving more reading time
"middle_segment_2_unique_overlay_count": 4,
"closing_avg_overlay_duration": 3.2,  # Sustained CTA
"closing_overlay_count": 1,
"closing_unique_overlay_count": 1,  # Single focused message
"closing_repetition_ratio": 0.0,
"reading_time_pattern": "quick_hook_sustained_cta"
```

#### Value for ML
- Distinguishes attention-grabbing quick text from informational sustained text
- Identifies cognitive load management strategies  
- Reveals where creators prioritize comprehension vs energy
- Shows variety (high unique count) vs emphasis (repetition) strategies
- Identifies information progression patterns (diverse → focused)
- Enables discovery of optimal reading time and diversity patterns for engagement

#### Notes from Analysis
- Quick hooks (< 1s) create urgency but risk missing message
- Sustained CTAs (> 2s) ensure message delivery
- Middle variation shows content complexity progression
- Pattern should align with speech pacing for coherence
- Window-local unique counts show variety within each segment
- Repetition ratio reveals emphasis strategy (0 = all unique, 1 = all same)
- Educational content tends toward high unique counts
- Viral content often uses repetition for memorability

#### Dependencies
- Text overlay timeline with timestamps
- Temporal window architecture (P0)
- Global avgOverlayDuration already implemented

---

### Basic Sticker Metrics

#### Problem Statement
- Stickers are counted in total but never analyzed separately from text overlays
- Missing platform-native visual language patterns (🔥💯😂 are TikTok's vocabulary)
- No visibility into sticker usage strategy across video segments
- Cannot distinguish sticker-heavy content from text-heavy content
- Sticker types (static/animated) not leveraged for pattern discovery

#### Explanation of Difficulty
- **Easy**: Extract counts from existing stickerTimeline data
- Timeline already contains timestamps and sticker_type
- Simple counting and density calculations per temporal window

#### Solution Design
- Calculate sticker counts per temporal window (parallel to text overlay counts)
- Track sticker density (stickers per second) globally and per window
- Extract sticker type distribution (static vs animated percentages)
- Keep stickers separate from text overlays (different cognitive processing)

#### Implementation Details
```python
# In compute_visual_overlay_metrics (precompute_functions_full.py)
# After text overlay calculations:

def calculate_sticker_metrics(sticker_timeline, video_duration):
    """Calculate sticker-specific metrics separate from text overlays"""
    
    metrics = {}
    
    # Global metrics
    total_stickers = len(sticker_timeline)
    metrics['total_sticker_count'] = total_stickers
    metrics['sticker_density'] = round(total_stickers / video_duration, 2) if video_duration > 0 else 0
    
    # Sticker type distribution
    sticker_types = {'static': 0, 'animated': 0}
    sticker_timestamps = []
    
    for timestamp, data in sticker_timeline.items():
        sticker_type = data.get('sticker_type', 'static')
        sticker_types[sticker_type] = sticker_types.get(sticker_type, 0) + 1
        
        try:
            start = float(timestamp.split('-')[0])
            sticker_timestamps.append(start)
        except:
            pass
    
    # Type distribution as percentages
    if total_stickers > 0:
        metrics['static_sticker_ratio'] = round(sticker_types['static'] / total_stickers, 2)
        metrics['animated_sticker_ratio'] = round(sticker_types.get('animated', 0) / total_stickers, 2)
    else:
        metrics['static_sticker_ratio'] = 0
        metrics['animated_sticker_ratio'] = 0
    
    # Temporal sticker counts
    # Hook (0-3s)
    hook_stickers = [t for t in sticker_timestamps if t < 3]
    metrics['hook_sticker_count'] = len(hook_stickers)
    metrics['hook_sticker_density'] = round(len(hook_stickers) / 3, 2)
    
    # Middle segments (3-5 segments based on duration)
    middle_start = 3
    middle_end = video_duration - 3
    num_segments = 3 if video_duration <= 18 else (4 if video_duration <= 33 else 5)
    segment_duration = (middle_end - middle_start) / num_segments
    
    for i in range(num_segments):
        seg_start = middle_start + (i * segment_duration)
        seg_end = seg_start + segment_duration
        
        segment_stickers = [t for t in sticker_timestamps 
                           if seg_start <= t < seg_end]
        
        metrics[f'middle_segment_{i+1}_sticker_count'] = len(segment_stickers)
        metrics[f'middle_segment_{i+1}_sticker_density'] = round(
            len(segment_stickers) / segment_duration, 2
        )
    
    # Closing (last 3s)
    closing_stickers = [t for t in sticker_timestamps if t >= video_duration - 3]
    metrics['closing_sticker_count'] = len(closing_stickers)
    metrics['closing_sticker_density'] = round(len(closing_stickers) / 3, 2)
    
    # Sticker usage pattern
    hook_count = metrics['hook_sticker_count']
    closing_count = metrics['closing_sticker_count']
    
    if hook_count > 3 and closing_count < 2:
        metrics['sticker_pattern'] = 'front_loaded'
    elif hook_count < 2 and closing_count > 3:
        metrics['sticker_pattern'] = 'climax_emphasis'
    elif hook_count > 3 and closing_count > 3:
        metrics['sticker_pattern'] = 'bookend'
    elif total_stickers > 10:
        metrics['sticker_pattern'] = 'continuous'
    else:
        metrics['sticker_pattern'] = 'minimal'
    
    return metrics

# Example output:
"total_sticker_count": 12,
"sticker_density": 0.4,  # Per second
"static_sticker_ratio": 0.75,
"animated_sticker_ratio": 0.25,
"hook_sticker_count": 5,  # Heavy sticker opening
"hook_sticker_density": 1.67,
"middle_segment_1_sticker_count": 2,
"middle_segment_2_sticker_count": 3,
"closing_sticker_count": 2,
"closing_sticker_density": 0.67,
"sticker_pattern": "front_loaded"
```

#### Value for ML
- Distinguishes platform-native creators (high sticker use) from traditional editors
- Identifies emotional amplification strategies (stickers as emphasis)
- Reveals engagement patterns specific to sticker-heavy content
- Separates instant-read stickers from processing-heavy text overlays
- Animated vs static ratios indicate production sophistication

#### Notes from Analysis
- Stickers are instant-read unlike text (no duration needed)
- Platform-native content uses stickers as punctuation
- Heavy sticker use correlates with younger audience targeting
- Animated stickers suggest higher production effort
- Sticker clustering often coincides with beat drops or emphasis moments

#### Dependencies
- stickerTimeline data structure
- Temporal window architecture (P0)
- Sticker type classification in OCR pipeline

---

### Pitch and Spectral Voice Metrics

#### Problem Statement
- Missing fundamental acoustic features for emotion
- Cannot capture voice brightness or tension
- No objective measures of emotional expression
- Relying on interpretive features instead of raw acoustics

#### Explanation of Difficulty
- **Medium**: Requires careful pitch extraction
- Must handle unvoiced segments and noise
- Need normalization within speaker
- Multiple spectral features to compute

#### Solution Design
- Add four core spectral metrics
- avgPitch and pitchVariance for fundamental frequency
- spectralCentroid for voice brightness
- zeroCrossingRate for voice texture
- Calculate for all windows and segments

#### Implementation Details
```python
def calculate_pitch_metrics(audio_timeline, sample_rate=22050):
    pitches, magnitudes = librosa.piptrack(y=audio_timeline, sr=sample_rate)
    voiced_pitches = pitches[pitches > 0]
    
    if len(voiced_pitches) == 0:
        return {'avgPitch': 0.0, 'pitchVariance': 0.0}
    
    return {
        'avgPitch': round(np.mean(voiced_pitches), 2),
        'pitchVariance': round(np.var(voiced_pitches), 2)
    }

def calculate_spectral_centroid(audio_timeline, sample_rate=22050):
    cent = librosa.feature.spectral_centroid(y=audio_timeline, sr=sample_rate)
    avg_centroid = np.mean(cent)
    normalized = (avg_centroid - 100) / 3900  # Normalize to speech range
    return round(np.clip(normalized, 0, 1), 3)

def calculate_zero_crossing_rate(audio_timeline):
    zcr = librosa.feature.zero_crossing_rate(audio_timeline)
    avg_zcr = np.mean(zcr)
    normalized = (avg_zcr - 0.02) / 0.08  # Normalize to speech range
    return round(np.clip(normalized, 0, 1), 3)

# Apply to all windows and segments:
"hook_avgPitch": 245.5,
"hook_pitchVariance": 1250.3,
"hook_spectralCentroid": 0.65,
"hook_zeroCrossingRate": 0.45,
# Middle segments (3-5 based on duration)
"middle_segment_1_avgPitch": 235.2,
"middle_segment_2_avgPitch": 240.8,
"middle_segment_3_avgPitch": 255.3,
"middle_segment_4_avgPitch": 238.5,           # if video > 60s
"middle_segment_4_pitchVariance": 1180.5,     # if video > 60s
"middle_segment_4_spectralCentroid": 0.62,    # if video > 60s
"middle_segment_4_zeroCrossingRate": 0.48,    # if video > 60s
"middle_segment_5_avgPitch": 242.1,           # if video > 90s
"middle_segment_5_pitchVariance": 1220.8,     # if video > 90s
"middle_segment_5_spectralCentroid": 0.64,    # if video > 90s
"middle_segment_5_zeroCrossingRate": 0.46     # if video > 90s
```

#### Dependencies
- None - uses existing LibROSA integration

---

### Basic Speech Content Indicators

#### Problem Statement
- Cannot distinguish content styles (tutorial vs casual vs energetic)
- Missing engagement signals like questions and greetings
- No visibility into speaking preparation level (fillers)
- Acoustic features alone don't reveal content structure

#### Explanation of Difficulty
- **Easy**: Simple pattern matching on existing transcripts
- No new dependencies or NLP libraries needed
- Fast processing (~0.01s per video)
- Straightforward keyword detection

#### Solution Design
- Add content type flags through keyword detection
- Calculate speaking style metrics from existing data
- Apply to all temporal windows and segments
- Zero-dependency implementation using basic string operations

#### Implementation Details
```python
def analyze_speech_content(transcript, duration):
    text_lower = transcript.lower()
    words = transcript.split()
    
    # Content type indicators
    greetings = ['hey', 'hello', 'hi ', 'what\'s up', 'welcome']
    has_greeting = any(g in text_lower[:50] for g in greetings)
    
    questions = ['how ', 'what ', 'why ', 'when ', 'where ', 'can you']
    has_question = '?' in transcript or any(q in text_lower for q in questions)
    
    instructions = ['first ', 'then ', 'next ', 'step ', 'you need', 
                   'make sure', 'don\'t forget', 'remember to']
    has_instruction = any(i in text_lower for i in instructions)
    
    exclamation_ratio = transcript.count('!') / max(len(words), 1)
    
    # Speaking style metrics
    speech_speed = len(words) / (duration / 60)  # words per minute
    
    fillers = ['um', 'uh', 'like', 'you know', 'i mean', 'basically']
    filler_count = sum(1 for w in words if w.lower() in fillers)
    filler_ratio = filler_count / max(len(words), 1)
    
    sentences = transcript.split('.')
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
    avg_sentence_length = sum(sentence_lengths) / max(len(sentence_lengths), 1)
    
    return {
        'has_greeting': has_greeting,
        'has_question': has_question,
        'has_instruction': has_instruction,
        'exclamation_ratio': exclamation_ratio,
        'speech_speed': speech_speed,
        'filler_word_ratio': filler_ratio,
        'sentence_length_avg': avg_sentence_length
    }

# Apply to windows:
"hook_window": {
    "hook_has_greeting": true,           # Opening engagement
    "hook_has_question": false,          # Viewer interaction
    "hook_has_instruction": false,       # Tutorial indicator
    "hook_exclamation_ratio": 0.15,     # Energy level
    "hook_speech_speed": 145,           # Words per minute
    "hook_filler_word_ratio": 0.08,     # Casual vs prepared
    "hook_sentence_length_avg": 8.5     # Complexity
},
"middle_window": {
    # Overall middle
    "middle_has_greeting": false,
    "middle_has_question": true,
    "middle_has_instruction": true,
    # Piecewise segments (3-5 based on duration)
    "middle_segment_1_has_greeting": false,
    "middle_segment_1_speech_speed": 155,
    "middle_segment_2_has_instruction": true,
    "middle_segment_2_filler_word_ratio": 0.05,
    "middle_segment_3_has_greeting": false,
    "middle_segment_3_has_instruction": false,
    "middle_segment_3_speech_speed": 150,
    "middle_segment_3_filler_word_ratio": 0.06,
    "middle_segment_4_has_instruction": true,      # if video > 60s
    "middle_segment_4_speech_speed": 158,          # if video > 60s
    "middle_segment_5_has_question": true,         # if video > 90s
    "middle_segment_5_speech_speed": 162           # if video > 90s
},
"closing_window": {
    "closing_has_question": true,        # Call to action
    "closing_exclamation_ratio": 0.25,  # Energy finish
    "closing_speech_speed": 160
}
```

#### Dependencies
- None - uses existing transcript data from speech processing

#### Status: DONE ✅
- Implemented 4 binary indicators: has_greeting, has_question, has_instruction, has_speech_cta
- Added to all temporal windows (hook, middle segments, closing)
- Function: calculate_speech_content_indicators() in temporal_compute.py
- Tested and working in production (Sept 17, 2025)
- Note: Simplified from 7 metrics to 4 to avoid collinearity (removed WPM, exclamation metrics, filler ratios)

---

### Caption Sentiment Analysis - SKIPPED

#### Problem Statement
- captionSentiment currently hardcoded to "neutral" in emotional_journey
- Critical engagement signal missing from TikTok caption analysis
- Caption emotion often drives virality (controversy, humor, inspiration)
- Architecturally misplaced in emotional_journey (should be in metadata_analysis)

#### Status: SKIPPED ❌
**Reason for skipping:** Low ML value for the complexity

**Analysis showed:**
- Single static feature (not temporal) - doesn't vary across windows
- Weak correlation with virality compared to video content features
- Most captions cluster around neutral sentiment
- Better signals already captured (has_question, word_count, hashtags)
- Would only be useful for edge cases (sarcasm detection, clickbait)

**ML Usefulness: 3/10**
- The model would likely ignore it in favor of stronger temporal features
- Resources better spent on temporal features that vary across windows

#### Explanation of Difficulty
- **Easy**: Simple sentiment analysis on existing caption text
- Caption data already available in metadata_analysis
- Basic NLP library (TextBlob or similar) for sentiment
- One-line implementation after library import

#### Solution Design
- Move from emotional_journey to metadata_analysis where caption is processed
- Add sentiment analysis to existing caption processing pipeline
- Return normalized sentiment score (-1 to 1)
- Architecturally correct: metadata analyzing metadata

#### Implementation Details
```python
# In compute_metadata_analysis_metrics (line ~1001)
def compute_metadata_analysis_metrics(static_metadata, metadata_summary, video_duration):
    # Existing: Caption text extraction
    caption_text = metadata_summary.get('description', '')
    if not caption_text:
        caption_text = static_metadata.get('text', '')
    
    # Existing caption analysis
    words = caption_text.split()
    word_count = len(words)
    has_question = int('?' in caption_text)
    # ... other existing features ...
    
    # NEW: Add sentiment analysis
    from textblob import TextBlob
    
    # Calculate sentiment (-1 to 1)
    if caption_text:
        try:
            sentiment = TextBlob(caption_text).sentiment.polarity
            caption_sentiment = round(sentiment, 3)  # -1 (negative) to 1 (positive)
        except:
            caption_sentiment = 0.0  # Neutral if analysis fails
    else:
        caption_sentiment = 0.0
    
    # Add to return structure
    return {
        # ... existing features ...
        'captionSentiment': caption_sentiment,  # NEW
        'wordCount': word_count,
        'hasQuestion': has_question,
        # ... rest of features ...
    }
```

#### Value for ML
- **Sentiment-engagement correlation**: Controversial captions drive comments
- **Content alignment**: Caption sentiment vs video emotion alignment
- **Viral patterns**: Positive captions for dance, negative for rants
- **Cross-validation**: Caption emotion vs speech emotion for authenticity

#### Notes from Investigation
- Caption data flows: Apify → VideoMetadata.description → metadata_analysis
- metadata_analysis already processes caption for wordCount, hasQuestion, CTAs
- All caption features currently in metadata_analysis, not emotional_journey
- Architecturally belongs with other caption analysis, not video emotion
- Currently placeholder in emotional_journey should be deprecated

#### Dependencies
- TextBlob or similar lightweight sentiment library
- Caption data (already available in metadata_analysis)

---

### Creative Density Climax Moment - SKIPPED

#### Problem Statement
- Have maxDensity value but missing WHEN the peak occurs
- Cannot align density peaks with emotional/speech/visual climaxes
- Lose timing information critical for pattern discovery
- ML cannot determine if production peaks align with emotional peaks

#### Implementation Status
❌ **SKIPPED** - Not compatible with temporal window structure

#### Reason for Skipping
Within our 3-10 second temporal windows, tracking climax position is not meaningful:

1. **Limited variance**:
   - Hook/Closing (3s): Only 3 possible positions (seconds 0, 1, 2)
   - Middle segments (~7.6s): Only ~8 possible positions
   - Most peaks would randomly fall in middle seconds

2. **Signal already captured**:
   - Comparing `max_density` across windows shows progression
   - ML can see if hook > middle > closing (front-loaded)
   - Or if middle_segment_2 has highest max_density (middle peak)

3. **Within-window position is noise**:
   - Whether peak is at second 1 vs 2 in a 3-second window doesn't indicate strategy
   - Too granular to be meaningful for ML

The existing approach of comparing `max_density` values across windows already provides the density progression signal without the noise of within-window positions.

#### Original Design Context
This feature was designed for full-video analysis where knowing if peak occurs at 30s vs 60s matters. It doesn't translate well to our temporal window architecture.

#### Dependencies
- Would have required density_per_second calculation
- Not applicable to temporal windows

---

### Normalize Climax Moments to Position

#### Problem Statement
- Climax moments have inconsistent formats (strings "28s" vs dicts vs missing)
- Cannot compare across videos of different durations
- speech_analysis.climaxMoment is completely missing
- ML models need normalized numerical values, not timestamp strings
- Prevents alignment analysis across speech/visual/emotional/density climaxes

#### Explanation of Difficulty
- **Easy**: Simple normalization calculations
- Just dividing timestamp by duration
- speech_analysis needs addition but straightforward
- Format standardization is mechanical change

#### Solution Design
- Convert all climaxMoment features to normalized position (0.0-1.0)
- Add missing speech_analysis.climaxMoment
- Ensure consistent float output across all domains
- Enable direct numerical comparison for ML

#### Implementation Details

**1. Fix emotional_journey.climaxMoment (precompute_professional.py ~line 465):**
```python
# Current:
climax_moment = f"{int(climax_moment[0])}s"

# Change to:
if emotion_timestamps:
    peak = max(emotion_timestamps, key=lambda x: x[2])  # x[2] is confidence
    emotional_climax_position = round(peak[0] / duration, 3)  # Normalize to 0-1
else:
    emotional_climax_position = None

# Update return structure:
"climaxMoment": emotional_climax_position,  # Now 0.45 instead of "45s"
```

**2. Fix visual_overlay_analysis.climaxMoment (precompute_professional.py ~line 236):**
```python
# Current:
"climaxMoment": max(overlay_peaks, key=lambda x: x["intensity"])["timestamp"] if overlay_peaks else None,

# Change to:
if overlay_peaks:
    peak_timestamp = max(overlay_peaks, key=lambda x: x["intensity"])["timestamp"]
    # Parse "0-8s" format to get start second
    start_second = float(peak_timestamp.split('-')[0])
    visual_climax_position = round(start_second / duration, 3)
else:
    visual_climax_position = None

"climaxMoment": visual_climax_position,  # Now 0.13 instead of "0-8s"
```

**3. Add speech_analysis.climaxMoment (new implementation needed):**
```python
# In speech_analysis computation (location TBD - need audio energy timeline):
def find_speech_climax(audio_timeline, duration):
    if not audio_timeline:
        return None
    
    # Find peak audio energy moment
    peak_entry = max(audio_timeline.items(), key=lambda x: x[1].get('energy', 0))
    peak_timestamp = parse_timestamp_to_seconds(peak_entry[0])
    
    return round(peak_timestamp / duration, 3)

# Add to speech_analysis result:
"climaxMoment": find_speech_climax(audio_timeline, duration)
```

#### Value for ML
- **Direct comparison**: All climax positions now comparable (0.3 vs 0.7)
- **Alignment scoring**: Can calculate variance of climax positions for coordination
- **Pattern discovery**: ML can find optimal climax positioning patterns
- **Duration agnostic**: 0.5 means midpoint regardless of video length
- **Complete system**: All four climaxes (emotional, visual, speech, density) aligned

#### Notes from Analysis
- Test outputs show inconsistent formats need fixing
- speech_analysis completely missing climaxMoment despite being in feature list
- Normalization to 0-1 essential for cross-video comparison
- Enables powerful alignment analysis: std([emotional_climax, visual_climax, speech_climax, density_climax])
- Fixes current ML incompatibility with string timestamps

#### Dependencies
- Existing climax calculations must be present
- Duration must be known
- Audio energy timeline needed for speech_climax

---

### Emotion Distribution Ratios

#### Problem Statement
- Currently only have dominantEmotion which loses critical information (40% joy reported as "joy")
- No visibility into emotional composition and mix
- Cannot detect multi-emotion patterns (horror-comedy, dramatic thriller)
- Missing emotional journey progression through video
- Single label oversimplifies complex emotional content

#### Explanation of Difficulty
- **Easy**: Simple ratio calculations from existing emotion detections
- Already have emotion data in expression_timeline
- Just need to count and normalize
- Straightforward percentage calculations

#### Solution Design
- Calculate emotion ratios for each temporal window (hook/middle/closing) only
- Remove global emotion distribution to avoid collinearity
- Provide complete emotional composition percentages per window
- Temporal-only approach shows emotional journey without redundancy

#### Implementation Details
```python
# In compute_emotional_journey_analysis_professional (precompute_professional.py)
# After line ~344 where dominant_emotion is calculated:

def calculate_emotion_distribution(emotions_list):
    """Calculate percentage distribution of emotions"""
    if not emotions_list:
        return {"neutral": 1.0}
    
    total = len(emotions_list)
    emotion_counts = {}
    
    # Count each emotion
    for emotion in emotions_list:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Convert to ratios
    emotion_distribution = {}
    for emotion in ['joy', 'surprise', 'neutral', 'sad', 'anger', 'fear', 'disgust']:
        count = emotion_counts.get(emotion, 0)
        emotion_distribution[f"{emotion}_ratio"] = round(count / total, 3)
    
    return emotion_distribution

# Extract emotions by window
hook_emotions = []
middle_emotions = []
closing_emotions = []

for timestamp, data in expression_timeline.items():
    time_sec = parse_timestamp_to_seconds(timestamp)
    emotion = data.get('emotion', 'neutral')
    
    if time_sec < 3:
        hook_emotions.append(emotion)
    elif time_sec >= duration - 3:
        closing_emotions.append(emotion)
    else:
        middle_emotions.append(emotion)

# Calculate temporal distributions only (no global to avoid collinearity)
hook_emotion_distribution = calculate_emotion_distribution(hook_emotions)
middle_emotion_distribution = calculate_emotion_distribution(middle_emotions)
closing_emotion_distribution = calculate_emotion_distribution(closing_emotions)

# Add to emotional_core_metrics (line ~364):
emotional_core_metrics = {
    # ... existing metrics ...
    # Removed globalEmotionDistribution to avoid collinearity
    "hookEmotionDistribution": hook_emotion_distribution,
    "middleEmotionDistribution": middle_emotion_distribution,
    "closingEmotionDistribution": closing_emotion_distribution,
    # ... rest of metrics ...
}
```

#### Value for ML
- **Genre classification**: 30% fear + 30% joy + 40% surprise = horror-comedy pattern
- **Journey patterns**: Fear hook → neutral middle → joy closing = redemption arc
- **Engagement prediction**: High fear_ratio in hook correlates with completion
- **Content matching**: Compare promised emotion (thumbnail) with delivered distribution
- **Complexity scoring**: Even distribution = emotional rollercoaster, skewed = mood piece

#### Notes from Analysis
- Replaces oversimplified dominantEmotion with complete picture
- 40% joy, 30% fear, 30% surprise all captured instead of just "joy"
- Enables emotional journey tracking through temporal windows
- Global distribution for overall composition, window distributions for progression
- Total of 28 new features (7 emotions × 4 contexts)

#### Dependencies
- Expression timeline with emotion labels
- Temporal window definitions (hook/middle/closing)
- Emotion detection must be functioning

---

### Temporal Eye Contact Metrics

#### Problem Statement
- Currently only have global eyeContactRate which averages across entire video
- Loses critical engagement patterns (attention-grabbing hook, demonstration middle, CTA closing)
- Cannot identify where creators establish parasocial connection
- Missing visibility into trust-building progression
- Single average masks strategic eye contact placement

#### Explanation of Difficulty
- **Medium**: Requires processing eye tracking data per temporal window
- Must handle variable middle segments based on video duration
- Need to aggregate frame-level eye contact into window rates
- Requires existing eye tracking timeline data

#### Solution Design
- Calculate eye contact rate for each temporal window (hook/middle/closing)
- Process middle segments individually (3-5 segments based on duration)
- Replace global metric with temporal versions to avoid correlation
- Provide both window-level and segment-level granularity

#### Implementation Details
```python
# In compute_person_framing_analysis (precompute_functions_full.py)
# After current eyeContactRate calculation (~line 265):

def calculate_temporal_eye_contact(eye_contact_timeline, video_duration):
    """Calculate eye contact rate per temporal window"""
    import numpy as np
    
    # Initialize results
    temporal_metrics = {}
    
    # Hook (0-3s)
    hook_frames = []
    for timestamp, has_eye_contact in eye_contact_timeline.items():
        time_sec = parse_timestamp_to_seconds(timestamp)
        if time_sec < 3:
            hook_frames.append(has_eye_contact)
    
    if hook_frames:
        temporal_metrics['hook_eye_contact_rate'] = np.mean(hook_frames)
    else:
        temporal_metrics['hook_eye_contact_rate'] = 0
    
    # Middle segments (3-5 based on duration)
    middle_segments = get_middle_segments(video_duration)
    
    for segment_name, (start, end) in middle_segments.items():
        segment_frames = []
        for timestamp, has_eye_contact in eye_contact_timeline.items():
            time_sec = parse_timestamp_to_seconds(timestamp)
            if start <= time_sec < end:
                segment_frames.append(has_eye_contact)
        
        if segment_frames:
            temporal_metrics[f'middle_{segment_name}_eye_contact_rate'] = np.mean(segment_frames)
        else:
            temporal_metrics[f'middle_{segment_name}_eye_contact_rate'] = 0
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_frames = []
    for timestamp, has_eye_contact in eye_contact_timeline.items():
        time_sec = parse_timestamp_to_seconds(timestamp)
        if time_sec >= closing_start:
            closing_frames.append(has_eye_contact)
    
    if closing_frames:
        temporal_metrics['closing_eye_contact_rate'] = np.mean(closing_frames)
    else:
        temporal_metrics['closing_eye_contact_rate'] = 0
    
    return temporal_metrics

# Add to person_framing_metrics dictionary:
person_framing_metrics = {
    # ... existing metrics ...
    # Remove global eyeContactRate to avoid correlation with temporal versions
    **calculate_temporal_eye_contact(eye_contact_timeline, duration),
    # ... rest of metrics ...
}
```

#### Value for ML
- **Hook engagement**: High eye contact (>0.7) in hook correlates with viewer retention
- **Tutorial patterns**: Low middle eye contact (<0.3) indicates demonstration focus
- **CTA effectiveness**: Rising eye contact in closing (>0.6) drives conversions
- **Parasocial strength**: Consistent high rates (>0.5) predict follower loyalty
- **Content style**: Eye contact distribution reveals presenter vs demonstrator approach

#### Notes from Analysis
- Eye contact differs from face presence - face can be visible without looking at camera
- Critical for influencer, educational, and sales content
- Temporal patterns reveal strategic engagement moments
- Global average of 0.4 could hide hook=0.8, middle=0.2, closing=0.6 pattern
- Complements face size metrics for complete intimacy analysis

#### Dependencies
- Eye tracking data from MediaPipe iris/gaze detection
- Temporal window definitions (hook/middle/closing)
- Video duration for segment calculation
- Existing eyeContactRate calculation infrastructure

---

### Temporal Face Visibility Metrics

#### Problem Statement
- Currently only have global faceVisibilityRate which averages across entire video
- Loses critical content strategy patterns (personality hook, demonstration middle, personal closing)
- Cannot identify where creators show their face vs product/hands
- Missing visibility into content type progression
- Single average masks strategic face placement decisions

#### Explanation of Difficulty
- **Medium**: Requires processing face detection data per temporal window
- Must handle variable middle segments based on video duration
- Need to aggregate frame-level face presence into window rates
- Requires existing face detection timeline data

#### Solution Design
- Calculate face visibility rate for each temporal window (hook/middle/closing)
- Process middle segments individually (3-5 segments based on duration)
- Replace global metric with temporal versions to avoid correlation
- Provide both window-level and segment-level granularity

#### Implementation Details
```python
# In compute_person_framing_metrics (precompute_functions_full.py)
# After current face_visibility_rate calculation (~line 1882):

def calculate_temporal_face_visibility(expression_timeline, video_duration):
    """Calculate face visibility rate per temporal window"""
    import numpy as np
    
    # Initialize results
    temporal_metrics = {}
    
    # Hook (0-3s)
    hook_faces = 0
    hook_total = 0
    for timestamp, data in expression_timeline.items():
        time_sec = parse_timestamp_to_seconds(timestamp)
        if time_sec < 3:
            hook_total += 1
            if data.get('emotion'):
                hook_faces += 1
    
    temporal_metrics['hook_face_visibility_rate'] = hook_faces / hook_total if hook_total > 0 else 0
    
    # Middle segments (3-5 based on duration)
    middle_segments = get_middle_segments(video_duration)
    
    for segment_name, (start, end) in middle_segments.items():
        segment_faces = 0
        segment_total = 0
        for timestamp, data in expression_timeline.items():
            time_sec = parse_timestamp_to_seconds(timestamp)
            if start <= time_sec < end:
                segment_total += 1
                if data.get('emotion'):
                    segment_faces += 1
        
        if segment_total > 0:
            temporal_metrics[f'middle_{segment_name}_face_visibility_rate'] = segment_faces / segment_total
        else:
            temporal_metrics[f'middle_{segment_name}_face_visibility_rate'] = 0
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_faces = 0
    closing_total = 0
    for timestamp, data in expression_timeline.items():
        time_sec = parse_timestamp_to_seconds(timestamp)
        if time_sec >= closing_start:
            closing_total += 1
            if data.get('emotion'):
                closing_faces += 1
    
    temporal_metrics['closing_face_visibility_rate'] = closing_faces / closing_total if closing_total > 0 else 0
    
    return temporal_metrics

# Add to metrics dictionary:
metrics = {
    # ... existing metrics ...
    # Remove global face_visibility_rate to avoid correlation with temporal versions
    **calculate_temporal_face_visibility(expression_timeline, duration),
    # ... rest of metrics ...
}
```

#### Value for ML
- **Content type classification**: High face visibility (>0.8) = talking head, Low (<0.3) = product/tutorial
- **Hook strategy**: Face in hook (>0.7) establishes personality-driven content
- **Tutorial patterns**: Low middle face visibility (<0.4) indicates hands-on demonstration
- **Closing effectiveness**: Face return in closing (>0.6) personalizes CTA
- **Genre identification**: Face visibility patterns distinguish vlog, tutorial, reaction, product review

#### Notes from Analysis
- Face visibility differs from eye contact - face can be present without looking at camera
- Critical for content type classification and engagement prediction
- Temporal patterns reveal strategic content decisions
- Global average of 0.5 could hide hook=0.9, middle=0.2, closing=0.8 pattern
- Works synergistically with eye contact metrics for complete personality analysis

#### Dependencies
- Face detection from MediaPipe/expression timeline
- Temporal window definitions (hook/middle/closing)
- Video duration for segment calculation
- Existing face_visibility_rate calculation infrastructure

---

### Temporal Framing Changes [ALREADY CAPTURED - Framing ratios encode changes]

#### Problem Statement
- Currently only have global framingChanges count across entire video
- Cannot identify where shot type dynamics occur (dynamic hook vs steady middle)
- Missing visibility into framing variety patterns through video journey
- Single count masks strategic placement of framing changes
- Cannot distinguish front-loaded dynamics from evenly distributed changes

#### Resolution
**Already captured in existing framing ratios!** The close_ratio, medium_ratio, wide_ratio, and none_ratio distributions implicitly encode framing changes:
- Single dominant ratio (e.g., medium_ratio: 1.0) = no framing changes
- Equal distribution across types = many framing changes
- ML models can derive framing dynamism from ratio entropy
- No additional implementation needed - existing ratios contain all necessary information

#### Original Solution (Not Needed)
- ~~Count framing changes within each temporal window (hook/middle/closing)~~
- ~~Process middle segments individually (3-5 segments based on duration)~~
- ~~Remove global sum to avoid multicollinearity (temporal counts contain all information)~~
- ~~Track both count and rate (changes per second) per window~~

#### Implementation Details
```python
# In compute_person_framing_metrics (precompute_functions_full.py)
# After current framing_changes calculation (~line 2030):

def calculate_temporal_framing_changes(framing_progression, video_duration):
    """Calculate framing changes per temporal window"""
    
    temporal_metrics = {}
    
    # Hook (0-3s)
    hook_changes = 0
    for i in range(len(framing_progression) - 1):
        if framing_progression[i]['end'] <= 3:
            # Check if this segment transitions to a different type
            if framing_progression[i]['type'] != framing_progression[i+1]['type']:
                hook_changes += 1
    
    temporal_metrics['hook_framing_changes'] = hook_changes
    temporal_metrics['hook_framing_change_rate'] = hook_changes / 3  # changes per second
    
    # Middle segments (3-5 based on duration)
    middle_segments = get_middle_segments(video_duration)
    total_middle_changes = 0
    
    for segment_name, (start, end) in middle_segments.items():
        segment_changes = 0
        segment_duration = end - start
        
        for i in range(len(framing_progression) - 1):
            # Check if transition happens within this segment
            if start <= framing_progression[i]['end'] < end:
                if framing_progression[i]['type'] != framing_progression[i+1]['type']:
                    segment_changes += 1
        
        temporal_metrics[f'middle_{segment_name}_framing_changes'] = segment_changes
        temporal_metrics[f'middle_{segment_name}_framing_change_rate'] = segment_changes / segment_duration
        total_middle_changes += segment_changes
    
    # Overall middle metrics
    middle_duration = max(3, video_duration - 3) - 3
    temporal_metrics['middle_framing_changes'] = total_middle_changes
    temporal_metrics['middle_framing_change_rate'] = total_middle_changes / middle_duration if middle_duration > 0 else 0
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_changes = 0
    
    for i in range(len(framing_progression) - 1):
        if framing_progression[i]['start'] >= closing_start:
            if framing_progression[i]['type'] != framing_progression[i+1]['type']:
                closing_changes += 1
    
    temporal_metrics['closing_framing_changes'] = closing_changes
    temporal_metrics['closing_framing_change_rate'] = closing_changes / 3
    
    return temporal_metrics

# Add to metrics:
metrics.update({
    'framing_changes': framing_changes,  # Keep global
    **calculate_temporal_framing_changes(framing_progression, duration)
})
```

#### Value for ML
- **Hook dynamics**: High framing changes (>2) in hook indicates attention-grabbing opening
- **Middle stability**: Low middle changes (<1) suggests focused demonstration or explanation
- **Closing variety**: Increased closing changes signals dynamic CTA or climax
- **Pacing patterns**: Front-loaded (hook heavy) vs distributed vs back-loaded dynamics
- **Production quality**: Strategic framing changes vs random camera movement

#### Notes from Analysis
- Different from scene cuts - specifically tracks shot TYPE changes
- Complements scene_pacing metrics by adding framing variety dimension
- Rate metrics (changes per second) normalize across different window durations
- Works with framing_volatility for complete dynamics picture

#### Dependencies
- Framing progression data from existing analysis
- Temporal window definitions (hook/middle/closing)
- Video duration for segment calculation
- Existing framing_changes calculation

---

### Temporal Framing Consistency [ALREADY CAPTURED - Derivable from framing ratios]

#### Problem Statement
- Currently only have global framing_volatility metric
- Cannot identify where stability/instability occurs in video
- Missing visibility into production quality patterns through journey
- Single score masks strategic stability choices (dynamic hook, steady middle)

#### Resolution
**Already captured in existing framing ratios!** Framing consistency (how often shot type changes) is derivable from the distribution:
- High consistency = One dominant ratio (e.g., medium_ratio: 0.95)
- Low consistency = Equal distribution across types (e.g., 0.33 each)
- ML models can compute consistency as max(ratios) or 1 - entropy(ratios)
- No additional implementation needed - existing ratios contain all necessary information

#### Original Solution (Not Needed)
- ~~Calculate framing volatility per temporal window~~
- ~~Convert to consistency scores (1 - volatility)~~
- ~~Replace global metric with temporal versions to avoid collinearity~~
- ~~Remove global framing_volatility from features_base~~
- ~~Provide stability progression through video~~

#### Implementation Details
```python
# In compute_person_framing_metrics (precompute_functions_full.py)
# After temporal framing changes calculation:

def calculate_temporal_framing_consistency(framing_progression, video_duration):
    """Calculate framing consistency per temporal window"""
    
    temporal_metrics = {}
    
    # Hook (0-3s)
    hook_changes = 0
    hook_frames = 0
    for seg in framing_progression:
        if seg['start'] < 3:
            # Count frames in hook
            overlap_start = max(0, seg['start'])
            overlap_end = min(3, seg['end'])
            hook_frames += (overlap_end - overlap_start)
            
            # Check for changes within hook
            if seg['end'] <= 3 and seg != framing_progression[-1]:
                hook_changes += 1
    
    hook_volatility = hook_changes / (3 * 30) if hook_frames > 0 else 0  # Assume 30fps
    temporal_metrics['hook_framing_consistency'] = round(1.0 - hook_volatility, 2)
    
    # Middle segments
    middle_segments = get_middle_segments(video_duration)
    
    for segment_name, (start, end) in middle_segments.items():
        segment_changes = 0
        segment_duration = end - start
        
        for i, seg in enumerate(framing_progression[:-1]):
            # Count changes within this middle segment
            if start <= seg['end'] <= end:
                segment_changes += 1
        
        segment_volatility = segment_changes / (segment_duration * 30) if segment_duration > 0 else 0
        temporal_metrics[f'middle_{segment_name}_framing_consistency'] = round(1.0 - segment_volatility, 2)
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_changes = 0
    
    for i, seg in enumerate(framing_progression[:-1]):
        if seg['start'] >= closing_start:
            closing_changes += 1
    
    closing_volatility = closing_changes / (3 * 30) if video_duration > 3 else 0
    temporal_metrics['closing_framing_consistency'] = round(1.0 - closing_volatility, 2)
    
    return temporal_metrics

# Replace global with temporal metrics:
# Remove global framing_volatility from features_base
metrics.update(calculate_temporal_framing_consistency(framing_progression, duration))
```

#### Value for ML
- **Production quality patterns**: High consistency = professional, low = amateur or creative choice
- **Hook strategy**: Low hook consistency (dynamic) often correlates with higher retention
- **Tutorial identification**: High middle consistency (>0.8) indicates steady demonstration
- **Content type**: Consistency patterns distinguish vlogs, tutorials, music videos, action content
- **Intentionality detection**: Consistent low stability = creative choice, variable = amateur

#### Notes from Analysis
- Inverse of volatility provides stability perspective
- Works with framingChanges for complete dynamics picture
- Professional content often has strategic inconsistency (dynamic hooks)
- Amateur content has unintentional inconsistency throughout

#### Dependencies
- Framing progression data
- Temporal window definitions
- Global framing_volatility calculation (already implemented)
- Frame rate assumption (30fps) or actual frame rate data

---

### Temporal Framing Distribution

#### Problem Statement
- Currently only have global framingDistribution showing overall shot type percentages
- Cannot see how shot composition evolves through video journey
- Missing visibility into strategic framing choices (intimate hooks, wide demonstrations)
- Single distribution masks compositional storytelling patterns
- Cannot identify content type from framing evolution

#### Explanation of Difficulty
- **Easy**: Aggregate existing shot type data by temporal windows
- Shot type data already tracked in camera_distance_timeline
- Just needs percentage calculations per window
- Straightforward implementation

#### Solution Design
- Calculate shot type distribution for each temporal window (hook/middle/closing)
- Provide percentages of close/medium/far shots per window
- Replace global distribution with temporal versions to avoid correlation
- Enable shot composition pattern analysis

#### Implementation Details
```python
# In compute_person_framing_metrics (precompute_functions_full.py)
# After current shot_type_distribution calculation (~line 1611):

def calculate_temporal_framing_distribution(camera_distance_timeline, video_duration):
    """Calculate shot type distribution per temporal window"""
    
    temporal_distributions = {}
    
    # Hook (0-3s)
    hook_counts = {'close': 0, 'medium': 0, 'far': 0}
    hook_total = 0
    
    for timestamp, distance in camera_distance_timeline.items():
        time_sec = parse_timestamp_to_seconds(timestamp)
        if time_sec < 3:
            hook_counts[distance] += 1
            hook_total += 1
    
    if hook_total > 0:
        temporal_distributions['hook_framing_distribution'] = {
            shot: round(count / hook_total, 2) 
            for shot, count in hook_counts.items()
        }
    else:
        temporal_distributions['hook_framing_distribution'] = {'close': 0, 'medium': 0, 'far': 0}
    
    # Middle segments (3-5 based on duration)
    middle_segments = get_middle_segments(video_duration)
    
    for segment_name, (start, end) in middle_segments.items():
        segment_counts = {'close': 0, 'medium': 0, 'far': 0}
        segment_total = 0
        
        for timestamp, distance in camera_distance_timeline.items():
            time_sec = parse_timestamp_to_seconds(timestamp)
            if start <= time_sec < end:
                segment_counts[distance] += 1
                segment_total += 1
        
        if segment_total > 0:
            temporal_distributions[f'middle_{segment_name}_framing_distribution'] = {
                shot: round(count / segment_total, 2)
                for shot, count in segment_counts.items()
            }
        else:
            temporal_distributions[f'middle_{segment_name}_framing_distribution'] = {'close': 0, 'medium': 0, 'far': 0}
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_counts = {'close': 0, 'medium': 0, 'far': 0}
    closing_total = 0
    
    for timestamp, distance in camera_distance_timeline.items():
        time_sec = parse_timestamp_to_seconds(timestamp)
        if time_sec >= closing_start:
            closing_counts[distance] += 1
            closing_total += 1
    
    if closing_total > 0:
        temporal_distributions['closing_framing_distribution'] = {
            shot: round(count / closing_total, 2)
            for shot, count in closing_counts.items()
        }
    else:
        temporal_distributions['closing_framing_distribution'] = {'close': 0, 'medium': 0, 'far': 0}
    
    return temporal_distributions

# Add to metrics:
metrics.update({
    # Remove global shot_type_distribution to avoid correlation with temporal versions
    'framing_distribution': shot_type_distribution,  # Add alias for professional wrapper
    **calculate_temporal_framing_distribution(camera_distance_timeline, duration)
})
```

#### Value for ML
- **Content type classification**: Tutorial (close-heavy), vlog (medium-heavy), performance (far-heavy)
- **Hook strategy**: Close hooks (>0.6) for intimacy, wide hooks for context establishment
- **Production patterns**: Professional videos show strategic distribution shifts
- **Engagement prediction**: Close-up CTAs (closing close% > 0.5) drive conversions
- **Genre fingerprints**: Each content type has characteristic distribution evolution

#### Notes from Analysis
- Different from faceSizeVariance which measures movement/stability
- Complements framingChanges which counts transitions
- Shows WHERE specific shot types are used strategically
- Essential for understanding visual storytelling patterns

#### Dependencies
- Camera distance timeline data
- Temporal window definitions (hook/middle/closing)
- Video duration for segment calculation
- Existing shot_type_distribution calculation

---

### Temporal Gaze Variance (Replace gazeSteadiness) [DONE ✅]

#### Problem Statement
- Currently gazeSteadiness returns categorical 'high'/'medium'/'low' based on arbitrary thresholds
- Loses continuous variance information through bucketing
- Pre-interprets what constitutes "steady" instead of letting ML discover
- Semantic categorization violates raw data principle
- ML cannot learn optimal variance levels for different content types
- No temporal visibility into where variance occurs

#### Explanation of Difficulty
- **Easy**: Simple change from categorical to numerical output
- Variance calculation already exists
- Just needs normalization and temporal segmentation
- Remove interpretation layer

#### Solution Design
- Replace categorical gazeSteadiness with temporal numerical variance (0-1)
- Calculate variance per temporal window only (no global to avoid collinearity)
- 0 = perfectly steady (no variation in eye contact)
- 1 = maximum variation
- Let ML determine optimal variance for engagement
- Remove gazeSteadiness from features entirely

#### Implementation Status
**Implemented successfully!**
- Added `calculate_gaze_variance()` function to temporal_compute.py
- Integrated into `process_segment()` to compute per-window variance
- Tested with real data - all windows show meaningful variance values
- Variance ranges from 0.003 to 0.014 across windows (realistic spread)

#### Implementation Details
```python
# Added to temporal_compute.py:
def calculate_gaze_variance(timeline_entries, start, end):
    """Calculate variance in eye contact scores within window"""
    # Collect eye contact scores from gaze entries
    # Return variance or 0 if insufficient data

# Integrated into process_segment():
gaze_variance = calculate_gaze_variance(
    timelines.get('timeline', {}).get('entries', []), start, end
)

# Original implementation plan (for reference):
def calculate_temporal_gaze_variance(gaze_timeline, video_duration):
    """Calculate gaze variance per temporal window"""
    
    temporal_metrics = {}
    
    # Hook (0-3s)
    hook_scores = []
    for timestamp, gaze_data in gaze_timeline.items():
        if parse_timestamp_to_seconds(timestamp) < 3:
            if gaze_data.get('eye_contact', 0) > 0:
                hook_scores.append(gaze_data['eye_contact'])
    
    if len(hook_scores) > 1:
        hook_variance = statistics.variance(hook_scores)
        temporal_metrics['hook_gaze_variance'] = round(min(hook_variance * 4, 1.0), 3)
    else:
        temporal_metrics['hook_gaze_variance'] = 0.0 if hook_scores else None
    
    # Middle segments
    middle_segments = get_middle_segments(video_duration)
    for segment_name, (start, end) in middle_segments.items():
        segment_scores = []
        for timestamp, gaze_data in gaze_timeline.items():
            time_sec = parse_timestamp_to_seconds(timestamp)
            if start <= time_sec < end:
                if gaze_data.get('eye_contact', 0) > 0:
                    segment_scores.append(gaze_data['eye_contact'])
        
        if len(segment_scores) > 1:
            segment_variance = statistics.variance(segment_scores)
            temporal_metrics[f'middle_{segment_name}_gaze_variance'] = round(min(segment_variance * 4, 1.0), 3)
        else:
            temporal_metrics[f'middle_{segment_name}_gaze_variance'] = 0.0 if segment_scores else None
    
    # Closing window
    closing_start = max(3, video_duration - 3)
    closing_scores = []
    for timestamp, gaze_data in gaze_timeline.items():
        if parse_timestamp_to_seconds(timestamp) >= closing_start:
            if gaze_data.get('eye_contact', 0) > 0:
                closing_scores.append(gaze_data['eye_contact'])
    
    if len(closing_scores) > 1:
        closing_variance = statistics.variance(closing_scores)
        temporal_metrics['closing_gaze_variance'] = round(min(closing_variance * 4, 1.0), 3)
    else:
        temporal_metrics['closing_gaze_variance'] = 0.0 if closing_scores else None
    
    return temporal_metrics

# Replace gazeSteadiness with temporal gaze variance only:
# Remove metrics['gaze_steadiness'] 
# Do NOT add global gaze_variance to avoid collinearity
metrics.update(calculate_temporal_gaze_variance(gaze_timeline, duration))
```

#### Value for ML
- **Continuous signal**: Preserves full variance information
- **Content differentiation**: ML discovers optimal variance per genre
- **Quality indicator**: ML learns if consistency matters for engagement
- **No pre-judgment**: Algorithm determines meaning of variance levels
- **Temporal patterns**: Variance changes through video reveal presentation style

#### Notes from Analysis
- Removes semantic interpretation layer
- Provides raw statistical measure
- Normalization ensures comparability across videos
- Missing data handled explicitly (None) rather than 'unknown'
- Enables discovery of non-linear relationships

#### Dependencies
- Eye contact scores from gaze timeline
- Statistics module for variance calculation
- Temporal windows for per-segment analysis

---

### Temporal Scene Duration Metrics

#### Problem Statement
- Currently only have global averageSceneDuration and longestScene across entire video
- Cannot see how pacing evolves through video journey
- Missing visibility into hook dynamics vs steady middle vs climactic ending
- Single average and max mask strategic pacing changes
- Cannot identify where sustained shots are used for emphasis
- Cannot detect acceleration or deceleration patterns

#### Explanation of Difficulty
- **Easy**: Simple calculation of averages within temporal windows
- Scene boundaries already detected
- Just needs segmentation by temporal windows
- Straightforward statistical calculation

#### Solution Design
- Calculate average scene duration per temporal window (hook/middle/closing)
- Include scene count per window for context
- Maintain global metric for compatibility
- Enable pacing progression analysis

#### Implementation Details
```python
# In compute_scene_pacing_metrics (precompute_functions_full.py)
# After current avg_scene_duration calculation:

def calculate_temporal_scene_durations(scene_timeline, video_duration):
    """Calculate average scene duration per temporal window"""
    import numpy as np
    
    temporal_metrics = {}
    
    # Get scene boundaries as timestamps
    scene_changes = [0]  # Start of video
    for timestamp in sorted(scene_timeline.keys()):
        if scene_timeline[timestamp].get('is_scene_change'):
            scene_changes.append(parse_timestamp_to_seconds(timestamp))
    scene_changes.append(video_duration)  # End of video
    
    # Calculate scene durations
    scene_durations = []
    for i in range(len(scene_changes) - 1):
        duration = scene_changes[i+1] - scene_changes[i]
        scene_durations.append((scene_changes[i], duration))
    
    # Hook (0-3s)
    hook_durations = [d for start, d in scene_durations if start < 3]
    if hook_durations:
        temporal_metrics['hook_avg_scene_duration'] = round(np.mean(hook_durations), 2)
        temporal_metrics['hook_longest_scene'] = round(max(hook_durations), 2)
        temporal_metrics['hook_shortest_scene'] = round(min(hook_durations), 2)
        temporal_metrics['hook_scene_count'] = len(hook_durations)
        # Calculate cuts per second (more honest for short windows)
        temporal_metrics['hook_cuts_per_second'] = round(len(hook_durations) / 3.0, 2)
    else:
        temporal_metrics['hook_avg_scene_duration'] = 3.0  # Single scene through hook
        temporal_metrics['hook_longest_scene'] = 3.0
        temporal_metrics['hook_shortest_scene'] = 3.0
        temporal_metrics['hook_scene_count'] = 1
        temporal_metrics['hook_cuts_per_second'] = 0.33  # 1 scene in 3s
    
    # Middle segments (3-5 based on duration)
    middle_segments = get_middle_segments(video_duration)
    
    for segment_name, (start, end) in middle_segments.items():
        segment_durations = [d for s, d in scene_durations if start <= s < end]
        
        if segment_durations:
            temporal_metrics[f'middle_{segment_name}_avg_scene_duration'] = round(np.mean(segment_durations), 2)
            temporal_metrics[f'middle_{segment_name}_longest_scene'] = round(max(segment_durations), 2)
            temporal_metrics[f'middle_{segment_name}_shortest_scene'] = round(min(segment_durations), 2)
            temporal_metrics[f'middle_{segment_name}_scene_count'] = len(segment_durations)
            # Calculate cuts per second for this segment
            segment_duration = end - start
            temporal_metrics[f'middle_{segment_name}_cuts_per_second'] = round(len(segment_durations) / segment_duration, 2)
        else:
            temporal_metrics[f'middle_{segment_name}_avg_scene_duration'] = end - start
            temporal_metrics[f'middle_{segment_name}_longest_scene'] = end - start
            temporal_metrics[f'middle_{segment_name}_shortest_scene'] = end - start
            temporal_metrics[f'middle_{segment_name}_scene_count'] = 1
            temporal_metrics[f'middle_{segment_name}_cuts_per_second'] = round(1.0 / (end - start), 2)
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_durations = [d for start, d in scene_durations if start >= closing_start]
    
    if closing_durations:
        temporal_metrics['closing_avg_scene_duration'] = round(np.mean(closing_durations), 2)
        temporal_metrics['closing_longest_scene'] = round(max(closing_durations), 2)
        temporal_metrics['closing_shortest_scene'] = round(min(closing_durations), 2)
        temporal_metrics['closing_scene_count'] = len(closing_durations)
        # Calculate cuts per second
        temporal_metrics['closing_cuts_per_second'] = round(len(closing_durations) / 3.0, 2)
    else:
        temporal_metrics['closing_avg_scene_duration'] = 3.0
        temporal_metrics['closing_longest_scene'] = 3.0
        temporal_metrics['closing_shortest_scene'] = 3.0
        temporal_metrics['closing_scene_count'] = 1
        temporal_metrics['closing_cuts_per_second'] = 0.33
    
    # Calculate pacing ratios relative to video average
    global_cuts_per_second = len(scene_changes) / video_duration if video_duration > 0 else 1
    
    if global_cuts_per_second > 0:
        temporal_metrics['hook_pacing_ratio'] = round(temporal_metrics['hook_cuts_per_second'] / global_cuts_per_second, 2)
        temporal_metrics['closing_pacing_ratio'] = round(temporal_metrics['closing_cuts_per_second'] / global_cuts_per_second, 2)
        
        # Calculate middle average pacing ratio
        middle_cuts_sum = sum([v for k, v in temporal_metrics.items() 
                              if 'middle' in k and 'cuts_per_second' in k])
        middle_segments_count = len([k for k in temporal_metrics.keys() 
                                    if 'middle' in k and 'cuts_per_second' in k])
        if middle_segments_count > 0:
            middle_avg_cuts = middle_cuts_sum / middle_segments_count
            temporal_metrics['middle_pacing_ratio'] = round(middle_avg_cuts / global_cuts_per_second, 2)
    
    return temporal_metrics

# Add to metrics:
metrics.update({
    # Remove global avg_scene_duration to avoid correlation with temporal versions
    **calculate_temporal_scene_durations(scene_timeline, duration)
})
```

#### Value for ML
- **Hook pacing**: Fast cuts (<1s avg) grab attention, slow (>3s) establish context
- **Middle evolution**: Acceleration indicates building excitement, long takes for explanation
- **Closing dynamics**: Quick cuts for urgency, sustained shots for emphasis
- **Genre patterns**: Music videos (all fast, short max), tutorials (slow middle, long sustained), vlogs (varied)
- **Quality indicator**: Strategic pacing changes vs random cutting
- **Extremes reveal strategy**: Long takes in middle = demonstration, in closing = emotional appeal

#### Notes from Analysis
- Complements scene count metrics with duration perspective
- Reveals pacing strategy evolution
- Average duration more stable than count for short windows
- Pacing pattern detection helps identify content strategy

#### Dependencies
- Scene boundary detection from timeline
- Temporal window definitions (hook/middle/closing)
- Video duration for segment calculation
- Existing scene_timeline data

---

### Expand Generic Hashtag Detection

#### Problem Statement
- Currently only detects 6 generic hashtags (fyp, foryou, foryoupage, viral, trending, explore)
- Missing 8 additional generic hashtags documented in FeaturesMLMVP.md
- genericRatio calculation is inaccurate due to incomplete list
- Underestimates use of discovery-focused hashtag strategy

#### Explanation of Difficulty
- **Easy**: Simply adding more hashtags to existing list
- No architectural changes needed
- Just expanding the hashtag array

#### Solution Design
- Expand from 6 to 14 generic hashtags
- Include platform-specific tags (tiktok, tiktokcreator, contentcreator)
- Add engagement tags (funny, duet, smallbusiness)
- Cover trending variations (trendingvideo, tiktokchallenge)

#### Implementation Details
```python
# In precompute_functions_full.py, expand generic_hashtags ~line 1070:

generic_hashtags = [
    # Current hashtags
    'fyp', 'foryou', 'foryoupage', 'viral', 'trending', 'explore',
    
    # Missing hashtags from FeaturesMLMVP.md
    'tiktok',           # Platform name itself
    'funny',            # General entertainment
    'duet',             # Collaboration/reaction content
    'smallbusiness',    # Commercial discovery
    'trendingvideo',    # Variation of trending
    'tiktokcreator',    # Creator-focused discovery
    'contentcreator',   # Generic creator tag
    'tiktokchallenge'   # Challenge participation
]
```

#### Expected Outcome
- More accurate genericRatio calculations
- Better identification of discovery-focused content strategy
- Proper classification of generic vs niche hashtags
- Improved signal for ML models about content targeting

#### Dependencies
- None - simple list expansion

#### Future Considerations
- Generic hashtags evolve with platform trends
- May need quarterly updates to stay current
- Consider regional variations (different languages/markets)

---

### Expand hasHook Pattern Detection

#### Problem Statement
- Currently only detects 7 hook patterns (wait for it, watch till, won't believe, pov:, story time, here's how, the secret)
- Misses majority of viral hook patterns used on TikTok
- Weak signal due to limited coverage
- Many proven engagement patterns not captured

#### Explanation of Difficulty
- **Easy**: Simply adding more patterns to existing list
- No architectural changes needed
- Just expanding the pattern array

#### Solution Design
- Expand from 7 patterns to 50+ proven viral hooks
- Include numerical hooks (99%, 90%, x things)
- Add controversy hooks (unpopular opinion, hot take)
- Cover instructional hooks (3 ways, 5 tips, here's why)
- Include emotional hooks (confession, warning, breaking)

#### Implementation Details
```python
# In precompute_functions_full.py, expand has_hook detection ~line 1085:

has_hook = int(any(pattern in caption_lower for pattern in [
    # Current patterns
    'wait for it', 'watch till', "won't believe", 'pov:', 
    'story time', "here's how", 'the secret',
    
    # Expanded patterns
    "you won't believe what happens next", "the secret behind", 
    "this is what happens when", "if you're seeing this",
    "the truth they don't want you to know", "stop scrolling",
    "is it just me", "i can't believe i just discovered",
    "it's not me it's you", "did you know that",
    "are you having trouble", "have you ever", 
    "what if i told you", "why is no one talking about",
    "how would you react",
    
    # Numerical hooks
    "99%", "90%", "x things", "3 ways", "5 tips", 
    "number one",
    
    # Problem/solution hooks
    "struggling with", "this mistake", "stop doing", 
    "instead try", "here's why", "the reason why",
    
    # Discovery hooks
    "things you didn't know", "nobody talks about", 
    "the truth about", "everything you know", 
    "hard truth", "underestimated",
    
    # Story hooks
    "this changed my", "what happened when", 
    "the story of how", "i challenged myself", 
    "here's what happened",
    
    # Controversy/urgency hooks
    "warning:", "breaking:", "don't hate me", 
    "unpopular opinion", "hot take:", "confession:"
]))
```

#### Expected Outcome
- Increased detection rate from ~5% to ~40% of videos with hooks
- Better signal for ML models
- Captures wider variety of engagement strategies
- More representative of actual TikTok content patterns

#### Dependencies
- None - simple pattern list expansion

---

### Simplify ctaFeatures Structure

#### Problem Statement
- ctaFeatures currently returns unnecessary redundant fields
- hasCTA duplicates the callToAction binary feature already kept
- ctaCount is directly derivable from summing the 5 CTA type flags
- Dictionary structure needs flattening for ML consumption
- Current structure creates redundancy and potential confusion

#### Explanation of Difficulty
- **Easy**: Simple removal of redundant fields
- Just need to modify return structure
- No complex logic changes required

#### Solution Design
- Remove hasCTA field (duplicate of callToAction)
- Remove ctaCount field (derivable from sum)
- Keep only the 5 specific CTA types as features
- Flatten dictionary to individual binary features for ML

#### Implementation Details
```python
# In precompute_functions_full.py, modify detect_cta_features() ~line 1091:

def detect_cta_features(text):
    """ML-ready CTA detection - returns only specific types"""
    text_lower = text.lower()
    
    # Only the 5 specific CTA types (removed hasCTA and ctaCount)
    cta_features = {
        'ctaFollow': 0,
        'ctaLike': 0,
        'ctaComment': 0,
        'ctaShare': 0,
        'ctaUrgency': 0
    }
    
    # Check each CTA type (existing logic)
    if any(p in text_lower for p in ['follow me', 'follow for', 'hit follow']):
        cta_features['ctaFollow'] = 1
    if any(p in text_lower for p in ['drop a like', 'hit like', 'double tap', 'like if']):
        cta_features['ctaLike'] = 1
    if any(p in text_lower for p in ['comment below', 'let me know', 'drop a comment', 'tell me']):
        cta_features['ctaComment'] = 1
    if any(p in text_lower for p in ['share this', 'tag someone', 'send this to']):
        cta_features['ctaShare'] = 1
    if any(p in text_lower for p in ['limited time', 'act now', 'last chance', 'today only', 'ends soon']):
        cta_features['ctaUrgency'] = 1
    
    # REMOVED: ctaCount and hasCTA calculations
    
    return cta_features
```

#### Expected Outcome
- Clean structure with only 5 binary features
- No redundancy with callToAction
- No derivable fields
- Direct ML consumption without flattening

#### Dependencies
- Must keep callToAction feature separately
- Ensure ML pipeline expects 5 individual features not nested dict

---

### Temporal Face Size Metrics

#### Problem Statement
- averageFaceSize only provides global average across entire video
- distanceVariation hardcoded to 0, misses filming dynamics
- Loses critical framing patterns (close-up hooks, pulled-back middle, close-up CTAs)
- Can't detect intimacy progression, camera stability, or strategic prominence changes
- Single average masks important storytelling through camera distance

#### Explanation of Difficulty
- **Medium**: Requires applying existing calculation to temporal windows
- Face size calculation already works
- Need to segment by our temporal architecture
- Must calculate both averages and variances per window

#### Solution Design
- Calculate average AND variance of face size per temporal window
- Hook (0-3s), Middle segments (varies by duration), Closing (last 3s)
- Maintains consistency with temporal window architecture
- Replaces both global averageFaceSize and unimplemented distanceVariation
- Enables ML to learn optimal face size progressions and filming dynamics

#### Implementation Details
```python
# In precompute_person_framing.py, add temporal calculations:

def calculate_temporal_face_metrics(face_timeline, video_duration):
    """Calculate average and variance of face size per temporal window"""
    import numpy as np
    
    # Hook (0-3s)
    hook_faces = [f['size'] for f in face_timeline if f['timestamp'] < 3]
    hook_avg_face_size = np.mean(hook_faces) if hook_faces else 0
    hook_face_size_variance = np.var(hook_faces) if len(hook_faces) > 1 else 0
    
    # Middle segments (based on duration)
    middle_segments = get_middle_segments(video_duration)
    middle_metrics = {}
    
    for segment_name, (start, end) in middle_segments.items():
        segment_faces = [f['size'] for f in face_timeline 
                        if start <= f['timestamp'] < end]
        middle_metrics[f'middle_{segment_name}_avg_face_size'] = \
            np.mean(segment_faces) if segment_faces else 0
        middle_metrics[f'middle_{segment_name}_face_size_variance'] = \
            np.var(segment_faces) if len(segment_faces) > 1 else 0
    
    # Closing (last 3s)
    closing_start = max(3, video_duration - 3)
    closing_faces = [f['size'] for f in face_timeline 
                    if f['timestamp'] >= closing_start]
    closing_avg_face_size = np.mean(closing_faces) if closing_faces else 0
    closing_face_size_variance = np.var(closing_faces) if len(closing_faces) > 1 else 0
    
    return {
        'hook_avg_face_size': hook_avg_face_size,
        'hook_face_size_variance': hook_face_size_variance,
        **middle_metrics,
        'closing_avg_face_size': closing_avg_face_size,
        'closing_face_size_variance': closing_face_size_variance
    }
```

#### Expected Outcome
- Captures framing strategy evolution through video
- Shows how creators use camera distance for engagement
- Enables pattern discovery (close-up hooks, demonstration pull-backs)
- Reveals professional vs amateur framing progressions
- Identifies filming dynamics (stable tripod vs handheld movement)
- Distinguishes intentional zooms from erratic camera work
- Fully replaces need for global distanceVariation metric

#### Dependencies
- Temporal window architecture must be properly implemented
- Face detection must be functioning

---

## P2: Optimizations (Nice to Have)

### Scene Duration Variance

#### Problem Statement
- Have averageSceneDuration but missing variance
- Cannot measure pacing consistency within windows
- No visibility into chaotic vs steady pacing
- Incomplete statistical picture (mean without spread)

#### Explanation of Difficulty
- **Easy**: Simple numpy standard deviation
- Calculation alongside existing mean
- No new data extraction needed
- 5 minutes to implement

#### Solution Design
- Add sceneDurationVariance metric
- Calculate standard deviation of scene durations
- Apply to all windows and segments
- Complements existing averageSceneDuration

#### Implementation Details
```python
def calculate_scene_duration_variance(scene_timeline):
    scene_durations = [scene['duration'] for scene in scene_timeline]
    if len(scene_durations) < 2:
        return 0.0
    return round(np.std(scene_durations), 3)

# Add alongside averageSceneDuration:
"hook_sceneDurationVariance": 0.45,
"middle_segment_1_sceneDurationVariance": 1.23,
"middle_segment_2_sceneDurationVariance": 0.67,
"middle_segment_3_sceneDurationVariance": 2.10,
"middle_segment_4_sceneDurationVariance": 1.85,  # if exists
"middle_segment_5_sceneDurationVariance": 1.45,  # if exists
"closing_sceneDurationVariance": 0.30
```

#### Dependencies
- None - uses existing scene data

---

### Quiet Period Metrics

#### Problem Statement
- quietMoments returns variable array incompatible with ML
- Cannot identify where quiet periods occur in windows
- Missing strategic pause patterns
- No visibility into cognitive rest distribution

#### Explanation of Difficulty
- **Medium**: Requires tracking zero-overlay segments
- Must handle segment boundaries carefully
- Need both window and segment level metrics
- Variable middle window adds complexity

#### Solution Design
- Add quiet ratio and period metrics
- Calculate for windows and piecewise segments
- Include binary indicators for start/end quiet
- Replace variable array with fixed metrics

#### Implementation Details
```python
def calculate_quiet_metrics(overlay_timeline, window_start, window_end):
    quiet_segments = []
    current_quiet_start = None
    
    for t in range(window_start, window_end):
        has_overlay = overlay_timeline.get(t, 0) > 0
        
        if not has_overlay and current_quiet_start is None:
            current_quiet_start = t
        elif has_overlay and current_quiet_start is not None:
            quiet_segments.append((current_quiet_start, t))
            current_quiet_start = None
    
    total_quiet = sum(end - start for start, end in quiet_segments)
    window_duration = window_end - window_start
    
    return {
        'quiet_ratio': total_quiet / window_duration,
        'quiet_periods': len(quiet_segments),
        'longest_quiet': max([e-s for s,e in quiet_segments]) if quiet_segments else 0,
        'has_quiet_start': overlay_timeline.get(window_start, 0) == 0,
        'ends_quiet': overlay_timeline.get(window_end-1, 0) == 0
    }

# Apply to all windows and segments:
"hook_quiet_ratio": 0.33,
"hook_quiet_periods": 2,
"hook_longest_quiet": 0.8,
"middle_segment_1_quiet_ratio": 0.40,
"middle_segment_1_quiet_periods": 3,
"middle_segment_2_quiet_ratio": 0.00,
"middle_segment_2_quiet_periods": 0,
"middle_segment_3_quiet_ratio": 0.35,
"middle_segment_3_quiet_periods": 2,
"middle_segment_4_quiet_ratio": 0.25,      # if video > 60s
"middle_segment_4_quiet_periods": 1,       # if video > 60s
"middle_segment_5_quiet_ratio": 0.30,      # if video > 90s
"middle_segment_5_quiet_periods": 2,       # if video > 90s
"closing_quiet_ratio": 0.15,
"closing_quiet_periods": 1
```

#### Dependencies
- Temporal Windows must be implemented first
- Overlay timeline data must be available

---

### Silence Duration Metrics

#### Problem Statement
- Have silenceRatio and silentMoments but missing durations
- Cannot distinguish short vs long pauses
- Missing average and maximum pause lengths
- Incomplete pause pattern analysis

#### Explanation of Difficulty
- **Easy**: Simple calculations on existing data
- Silence periods already detected
- Just need mean and max calculations
- 5-10 lines of code total

#### Solution Design
- Add avgSilenceDuration for typical pause length
- Add maxSilenceGap for dramatic pauses
- Calculate for all windows and segments
- Work with existing silence metrics

#### Implementation Details
```python
def calculate_silence_duration_metrics(silence_periods):
    if not silence_periods:
        return {'avgSilenceDuration': 0.0, 'maxSilenceGap': 0.0}
    
    durations = [end - start for start, end in silence_periods]
    
    return {
        'avgSilenceDuration': round(np.mean(durations), 3),
        'maxSilenceGap': round(max(durations), 3)
    }

# Apply to all windows and segments:
"hook_avgSilenceDuration": 0.25,
"hook_maxSilenceGap": 0.45,
"middle_segment_1_avgSilenceDuration": 0.35,
"middle_segment_1_maxSilenceGap": 1.20,
"middle_segment_2_avgSilenceDuration": 0.30,
"middle_segment_2_maxSilenceGap": 0.95,
"middle_segment_3_avgSilenceDuration": 0.28,
"middle_segment_3_maxSilenceGap": 0.75,
"middle_segment_4_avgSilenceDuration": 0.32,      # if video > 60s
"middle_segment_4_maxSilenceGap": 1.10,           # if video > 60s
"middle_segment_5_avgSilenceDuration": 0.40,      # if video > 90s
"middle_segment_5_maxSilenceGap": 1.50,           # if video > 90s
"closing_avgSilenceDuration": 0.20,
"closing_maxSilenceGap": 0.35
```

#### Dependencies
- Temporal Windows architecture
- Overlay timeline data

---

### Enhanced Emotion Metrics - DONE (Ratios Only)

#### Problem Statement
- Currently only capture dominant emotion per window
- Missing emotional variety/range information
- Cannot identify secondary emotional themes
- ML lacks signal about emotional complexity vs monotone delivery

#### Implementation Status
✅ **Already Implemented:**
- All 7 emotion ratios (joy_ratio, sadness_ratio, anger_ratio, fear_ratio, disgust_ratio, surprise_ratio, neutral_ratio)
- Consistent features even when emotions are missing (initialized to 0)
- Complete emotional distribution per window

❌ **SKIPPED (Redundant/Derivable):**
- `emotion_variety`: Completely derivable by counting non-zero ratios
- `dominant_emotion`: Derivable by finding max ratio
- `secondary_emotion`: Derivable by finding second-highest ratio

#### Reason for Skipping Additional Metrics
The proposed additions are 100% derivable from existing ratios:
- Variety = count(ratio > 0)
- Dominant = argmax(ratios)
- Secondary = argsort(ratios)[1]

Adding them would be pure redundancy. ML models can easily derive these patterns from the ratios. The current 7 emotion ratios provide complete emotional information without redundancy beyond the inherent sum-to-1 constraint.

#### Solution Design
The existing implementation with 7 emotion ratios is optimal. No additional metrics needed.

#### Implementation Details
```python
# No removals needed - these are additions

# Add to temporal windows:
"hook_window": {
  "hook_emotion_variety": 3,              # Count of unique emotions
  "hook_secondary_emotion": "sad"         # Second most common emotion
},
"middle_window": {
  # Overall middle
  "middle_emotion_variety": 5,
  "middle_secondary_emotion": "neutral",
  # Piecewise segments (3-5 based on duration)
  "middle_segment_1_emotion_variety": 2,
  "middle_segment_1_secondary_emotion": "happy",
  "middle_segment_2_emotion_variety": 3,
  "middle_segment_2_secondary_emotion": "surprised",
  "middle_segment_3_emotion_variety": 2,
  "middle_segment_3_dominant_emotion": "sad",
  "middle_segment_3_secondary_emotion": "neutral",
  "middle_segment_4_emotion_variety": 3,         # if video > 60s
  "middle_segment_4_dominant_emotion": "angry",  # if video > 60s
  "middle_segment_4_secondary_emotion": "sad",   # if video > 60s
  "middle_segment_5_emotion_variety": 2,         # if video > 90s
  "middle_segment_5_dominant_emotion": "happy",  # if video > 90s
  "middle_segment_5_secondary_emotion": "excited" # if video > 90s
},
"closing_window": {
  "closing_emotion_variety": 2,
  "closing_secondary_emotion": "excited"
}

# Example calculation:
from collections import Counter

emotions_in_window = ["happy", "sad", "happy", "angry", "sad"]
emotion_counts = Counter(emotions_in_window)

emotion_variety = len(emotion_counts)  # 3 unique emotions
most_common = emotion_counts.most_common(2)
secondary_emotion = most_common[1][0] if len(most_common) > 1 else None
```

#### Dependencies
- None - uses existing emotion detection

---

### Enhanced Gesture Metrics

#### Problem Statement
- Currently only capture gesture count and presence per window
- Missing gesture type diversity information
- Cannot identify primary communication gestures
- ML lacks signal about gesture variety vs repetitive movements

#### Explanation of Difficulty
- **Easy**: Simple counting and classification of existing gesture data
- MediaPipe already provides gesture type classification
- Just needs aggregation and ranking logic
- Standard categorical encoding for gesture types

#### Solution Design
- Add gesture variety count for each window
- Track dominant gesture type (most common)
- Track secondary gesture type (second most common)
- Apply to all temporal windows and segments

#### Implementation Details
```python
# No removals needed - these are additions

# Add to temporal windows:
"hook_window": {
  "hook_gesture_variety": 3,              # Count of unique gesture types
  "hook_dominant_gesture": "pointing",    # Most common gesture type
  "hook_secondary_gesture": "open_palm"   # Second most common gesture
},
"middle_window": {
  # Overall middle
  "middle_gesture_variety": 5,
  "middle_dominant_gesture": "thumbs_up",
  "middle_secondary_gesture": "pointing",
  # Piecewise segments (3-5 based on duration)
  "middle_segment_1_gesture_variety": 2,
  "middle_segment_1_dominant_gesture": "waving",
  "middle_segment_1_secondary_gesture": "pointing",
  "middle_segment_2_gesture_variety": 3,
  "middle_segment_2_dominant_gesture": "open_palm",
  "middle_segment_2_secondary_gesture": "victory",
  "middle_segment_3_gesture_variety": 4,
  "middle_segment_3_dominant_gesture": "thumbs_up",
  "middle_segment_3_secondary_gesture": "pointing",
  "middle_segment_4_gesture_variety": 2,         # if video > 60s
  "middle_segment_4_dominant_gesture": "waving", # if video > 60s
  "middle_segment_4_secondary_gesture": "ok_sign", # if video > 60s
  "middle_segment_5_gesture_variety": 3,         # if video > 90s
  "middle_segment_5_dominant_gesture": "closed_fist", # if video > 90s
  "middle_segment_5_secondary_gesture": "rock_sign" # if video > 90s
},
"closing_window": {
  "closing_gesture_variety": 2,
  "closing_dominant_gesture": "pointing",
  "closing_secondary_gesture": "thumbs_up"
}

# Example calculation:
from collections import Counter

gestures_in_window = ["pointing", "open_palm", "pointing", "waving", "pointing"]
gesture_counts = Counter(gestures_in_window)

gesture_variety = len(gesture_counts)  # 3 unique gesture types
most_common = gesture_counts.most_common(2)
dominant_gesture = most_common[0][0] if most_common else None
secondary_gesture = most_common[1][0] if len(most_common) > 1 else None

# Common MediaPipe gesture types:
# pointing, open_palm, closed_fist, thumbs_up, thumbs_down,
# victory, ok_sign, waving, rock_sign, call_me
```

#### Dependencies
- MediaPipe gesture detection must be running
- Gesture type classification must be enabled

---

### Enhanced Object Metrics

#### Problem Statement
- Currently no systematic object tracking in temporal windows
- Missing content type classification signals
- Cannot identify viral content niches (pet videos, cooking, tech)
- ML lacks context about what appears in frame beyond people

#### Explanation of Difficulty
- **Medium**: Requires aggregating YOLO detection results
- Object detection already runs but not aggregated to windows
- Need to handle multiple objects per frame
- Must map YOLO's 80+ object classes to meaningful categories

#### Solution Design
- Add object count and variety for each window
- Track dominant and secondary object types
- Apply to all temporal windows and segments
- Focus on content-defining objects (pets, food, electronics)

#### Implementation Details
```python
# No removals needed - these are additions

# Add to temporal windows:
"hook_window": {
  "hook_object_count": 5,               # Total objects detected
  "hook_object_variety": 3,             # Count of unique object types
  "hook_dominant_object": "dog",        # Most common object type
  "hook_secondary_object": "person"     # Second most common object
},
"middle_window": {
  # Overall middle
  "middle_object_count": 12,
  "middle_object_variety": 6,
  "middle_dominant_object": "food",
  "middle_secondary_object": "plate",
  # Piecewise segments (3-5 based on duration)
  "middle_segment_1_object_count": 4,
  "middle_segment_1_object_variety": 2,
  "middle_segment_1_dominant_object": "dog",
  "middle_segment_1_secondary_object": "toy",
  "middle_segment_2_object_count": 5,
  "middle_segment_2_object_variety": 3,
  "middle_segment_2_dominant_object": "person",
  "middle_segment_2_secondary_object": "chair",
  "middle_segment_3_object_count": 8,
  "middle_segment_3_object_variety": 5,
  "middle_segment_3_dominant_object": "laptop",
  "middle_segment_3_secondary_object": "phone",
  "middle_segment_4_object_count": 6,            # if video > 60s
  "middle_segment_4_object_variety": 4,          # if video > 60s
  "middle_segment_4_dominant_object": "car",    # if video > 60s
  "middle_segment_4_secondary_object": "bicycle", # if video > 60s
  "middle_segment_5_object_count": 7,            # if video > 90s
  "middle_segment_5_object_variety": 3,          # if video > 90s
  "middle_segment_5_dominant_object": "pizza",  # if video > 90s
  "middle_segment_5_secondary_object": "cake"   # if video > 90s
},
"closing_window": {
  "closing_object_count": 3,
  "closing_object_variety": 2,
  "closing_dominant_object": "person",
  "closing_secondary_object": "phone"
}

# Example calculation:
from collections import Counter

objects_in_window = ["dog", "person", "dog", "toy", "dog", "person"]
object_counts = Counter(objects_in_window)

object_count = len(objects_in_window)  # 6 total objects
object_variety = len(object_counts)  # 3 unique types
most_common = object_counts.most_common(2)
dominant_object = most_common[0][0] if most_common else None  # "dog"
secondary_object = most_common[1][0] if len(most_common) > 1 else None  # "person"

# Key YOLO object categories for content classification:
# Animals: dog, cat, bird, horse
# Food: pizza, sandwich, cake, apple, banana
# Electronics: laptop, phone, tv, keyboard
# Vehicles: car, bicycle, motorcycle, bus
# Sports: ball, tennis racket, skateboard, surfboard
# Furniture: chair, table, couch, bed
```

#### Dependencies
- YOLO object detection must be running
- Object timeline data must be available

---

### Text Content Classification Metrics

#### Problem Statement
- Cannot distinguish between marketing-heavy vs organic content
- Missing quantity information (1 CTA vs 5 CTAs treated the same)
- No visibility into text content strategies (hashtag-heavy, emoji-rich)
- Binary presence flags lose multi-instance patterns

#### Explanation of Difficulty
- **Easy**: Simple keyword matching and counting
- Text data already extracted from overlays
- Pattern matching for common content types
- No NLP libraries or embeddings needed

#### Solution Design
- Classify each text overlay by content type
- Count instances of each type per window
- Calculate proportional ratios for normalization
- Apply to all temporal windows and segments

#### Implementation Details
```python
def classify_text_overlays(text_overlays):
    cta_keywords = ['subscribe', 'follow', 'like', 'comment', 'share', 
                    'tap', 'click', 'swipe', 'link', 'bio', 'dm']
    
    counts = {
        'cta_text_count': 0,
        'caption_text_count': 0,
        'hashtag_text_count': 0,
        'emoji_text_count': 0
    }
    
    for text in text_overlays:
        text_lower = text.lower()
        
        # Check for CTA keywords
        if any(keyword in text_lower for keyword in cta_keywords):
            counts['cta_text_count'] += 1
        else:
            counts['caption_text_count'] += 1  # Default to caption
        
        # Check for hashtags
        if '#' in text:
            counts['hashtag_text_count'] += 1
        
        # Check for emojis (simplified check)
        if any(ord(c) > 127 for c in text):  # Non-ASCII often emojis
            counts['emoji_text_count'] += 1
    
    # Calculate ratios
    total = len(text_overlays)
    if total > 0:
        counts['cta_text_ratio'] = counts['cta_text_count'] / total
        counts['hashtag_text_ratio'] = counts['hashtag_text_count'] / total
        counts['emoji_text_ratio'] = counts['emoji_text_count'] / total
    
    return counts

# Apply to temporal windows:
"hook_window": {
    "hook_cta_text_count": 1,           # Number of CTA texts
    "hook_caption_text_count": 2,       # Number of descriptive texts
    "hook_hashtag_text_count": 1,       # Number of hashtag texts
    "hook_emoji_text_count": 1,         # Number of texts with emojis
    "hook_cta_text_ratio": 0.33,        # Proportion that are CTAs
    "hook_hashtag_text_ratio": 0.33,    # Proportion with hashtags
    "hook_emoji_text_ratio": 0.33       # Proportion with emojis
},
"middle_window": {
    # Overall middle
    "middle_cta_text_count": 2,
    "middle_caption_text_count": 3,
    "middle_hashtag_text_count": 1,
    "middle_emoji_text_count": 2,
    "middle_cta_text_ratio": 0.4,
    # Piecewise segments (3-5 based on duration)
    "middle_segment_1_cta_text_count": 0,
    "middle_segment_1_caption_text_count": 1,
    "middle_segment_2_cta_text_count": 1,
    "middle_segment_2_hashtag_text_count": 1,
    "middle_segment_3_cta_text_count": 2,
    "middle_segment_3_caption_text_count": 1,
    "middle_segment_3_cta_text_ratio": 0.67,
    "middle_segment_4_cta_text_count": 1,           # if video > 60s
    "middle_segment_4_hashtag_text_count": 2,       # if video > 60s
    "middle_segment_4_emoji_text_ratio": 0.4,       # if video > 60s
    "middle_segment_5_cta_text_count": 0,           # if video > 90s
    "middle_segment_5_caption_text_count": 3,       # if video > 90s
    "middle_segment_5_hashtag_text_ratio": 0.2      # if video > 90s
},
"closing_window": {
    "closing_cta_text_count": 2,        # Strong CTA finish
    "closing_caption_text_count": 0,
    "closing_hashtag_text_count": 0,
    "closing_emoji_text_count": 1,
    "closing_cta_text_ratio": 1.0       # All texts are CTAs
}

# ML insights:
# High cta_text_count in closing = Strong marketing push
# High hashtag_text_ratio overall = Discovery-optimized
# Emoji_text_count correlation with engagement
# Multiple CTAs in hook = Aggressive strategy
```

#### Dependencies
- Text overlay extraction must be functioning
- Overlay Counts in Windows (P0) for base text_count

---

## P3: Future Enhancements (Post-MVP)

### Speech Segmentation Metrics

#### Problem Statement
- Cannot distinguish speaking rhythm patterns (choppy vs smooth)
- Missing delivery style signals (rapid-fire vs continuous)
- No visibility into speech continuity within windows
- Speech fragmentation patterns not captured

#### Explanation of Difficulty
- **Easy**: Simple counting of continuous speech blocks
- Speech timeline data already exists
- Just need to identify gaps and count segments
- Minimal processing overhead

#### Solution Design
- Count distinct continuous speech blocks per window
- Calculate fragmentation rate (segments per second)
- Apply to all temporal windows and segments
- Provides rhythm metrics without interpretation

#### Implementation Details
```python
def count_speech_segments(speech_timeline, window_start, window_end):
    segments = []
    current_segment = None
    
    for timestamp, has_speech in speech_timeline.items():
        if window_start <= timestamp <= window_end:
            if has_speech and current_segment is None:
                current_segment = {'start': timestamp}
            elif not has_speech and current_segment:
                current_segment['end'] = timestamp
                segments.append(current_segment)
                current_segment = None
    
    segment_count = len(segments)
    fragmentation = segment_count / (window_end - window_start)
    
    return {
        'speech_segment_count': segment_count,
        'speech_fragmentation': round(fragmentation, 2)
    }

# Apply to temporal windows:
"hook_window": {
    "hook_speech_segment_count": 3,        # 3 distinct speech blocks
    "hook_speech_fragmentation": 1.0       # 1 segment per second
},
"middle_window": {
    # Overall middle
    "middle_speech_segment_count": 12,
    "middle_speech_fragmentation": 0.8,
    # Piecewise segments (3-5 based on duration)
    "middle_segment_1_speech_segment_count": 5,
    "middle_segment_1_speech_fragmentation": 1.67,
    "middle_segment_2_speech_segment_count": 3,
    "middle_segment_2_speech_fragmentation": 1.0,
    "middle_segment_3_speech_segment_count": 4,
    "middle_segment_3_speech_fragmentation": 1.33,
    "middle_segment_4_speech_segment_count": 3,     # if video > 60s
    "middle_segment_4_speech_fragmentation": 1.0,   # if video > 60s
    "middle_segment_5_speech_segment_count": 5,     # if video > 90s
    "middle_segment_5_speech_fragmentation": 1.67   # if video > 90s
},
"closing_window": {
    "closing_speech_segment_count": 2,
    "closing_speech_fragmentation": 0.67
}

# ML insights:
# High fragmentation = Rapid tips/list format
# Low fragmentation = Storytelling/explanation
# Hook fragmentation predicts content style
```

#### Dependencies
- Quiet Period Metrics (P2) provides related silence patterns
- Speech timeline data must be available

---


## Implementation Order

### Critical Path
1. **First**: Temporal Windows as Single Source of Truth (P0) - Everything depends on this
2. **Second**: Multimodal Counts (P0) - Requires temporal windows
3. **Parallel**: All P1 improvements can be done simultaneously
4. **Last**: P2 improvements after P1 complete

### Time Estimates
- **Week 1**: Complete P0 architectural changes
- **Week 2**: Implement P1 improvements in parallel
- **Week 3**: Add P2 optimizations and testing

### Success Metrics
- All features compatible with piecewise segments
- No redundant global counts
- ML can discover patterns without pre-computed correlations
- Processing time remains under 60 seconds

---

## Notes

### Piecewise Segment Adaptation
All improvements MUST support:
- Hook (0-3s)
- Middle segments: 3-5 segments based on video duration
  - Videos 15-30s: 3 segments
  - Videos 30-60s: 4 segments  
  - Videos 60s+: 5 segments
- Closing (last 3s)

**Segment Calculation**:
```python
def calculate_middle_segments(video_duration):
    middle_start = 3  # After hook
    middle_end = video_duration - 3  # Before closing
    middle_duration = middle_end - middle_start
    
    if middle_duration <= 12:
        num_segments = 3
    elif middle_duration <= 27:
        num_segments = 4
    else:
        num_segments = 5
    
    # Divide middle into equal segments
    segment_duration = middle_duration / num_segments
    return num_segments, segment_duration
```

This granularity enables ML to discover pacing changes and progression patterns within the middle section that were previously invisible.

### Excluded from MVP
- Phase 2 sync metrics (text-gesture-speech alignment)
- Deep learning preparations
- Validation requirements
- Performance benchmarks

These belong in Future - MLrevolutions.md document.

---

## Outcome Metrics (Target Components)

These metrics are post-publication outcomes used to calculate engagement rate (our ML target).
They must NOT be included as features to avoid data leakage.

### Metrics to Extract but NOT Use as Features
- commentCount: Total comments received
- likeCount: Total likes received
- shareCount: Total shares received
- viewCount: Total views received (denominator for engagement rate)
- engagementRate: (Likes + Comments + Shares) / Views - THE key success metric and likely ML target

### Important Notes
- These are OUTCOMES, not FEATURES
- Cannot be controlled during video creation
- Including these as features would cause circular reasoning
- Must be fetched from API but stored separately from feature sets