# FeaturesMLMVPv2 - Validated Features for ML Pipeline

**Created**: 2025-01-09
**Purpose**: Final validated feature list after crosschecking with ImprovementsMLMVP.md and architectural compatibility

## Overview

This document contains features validated through the Feature Validation Framework, ensuring:
- No duplicity with planned improvements
- Architectural compatibility (Global vs Temporal)
- Clear transformation requirements for RF and K-means

## Feature Table

| Source | Feature | Window Type | Reason | RF Transform | RF Difficulty | KM Transform | KM Difficulty | In Improvements? | Notes |
|--------|---------|-------------|--------|--------------|---------------|--------------|---------------|------------------|-------|
| creative_density | accelerationPattern | Global-Derived | Captures overall pacing patterns across windows | One-hot encode 4 categories | Low | Label encode (0-3) + scale | Low | No | Uses aggregated middle window density (avg of 3-5 segments) to remain video-length agnostic |
| creative_density | elementCounts | Global-Inherent | Raw counts of each element type, content composition volume | None needed | Low | None needed | Low | Yes - P0 Derived Global Metrics | Covered by total_text_count, total_sticker_count, etc. in P0 improvements. Provides global composition data |
| creative_density | maxDensity | Global-Derived | Peak creative intensity, shows production ceiling | None needed | Low | Scale/normalize | Low | Yes - P0 Per-Window Density Extremes | Replaced by hook_max_density, middle_max_density, closing_max_density for temporal context |
| creative_density | minDensity | Global-Derived | Baseline activity level, shows production floor | None needed | Low | Scale/normalize | Low | Yes - P0 Per-Window Density Extremes | Replaced by hook_min_density, middle_min_density, closing_min_density for temporal context |
| creative_density | sceneChangeCount | Global-Inherent | Fundamental editing metric, shows production pace and engagement | None needed | Low | None needed | Low | Yes - P0 (both global and temporal) | Covered by total_transition_count globally and hook/middle/closing_scene_count per window |
| creative_density | stdDeviation | Global-Derived | Measures consistency vs variation in density across entire video | None needed | Low | Scale/normalize | Low | No | Captures global variance across all seconds that per-window extremes can't provide |
| creative_density | totalElements | Global-Inherent | Absolute production volume, shows total content effort | None needed | Low | None needed | Low | Yes - P0 Derived Global Metrics | Replaced by total_elements_derived, valuable for within-bucket comparisons |
| creative_density | volatility | Global-Derived | Normalized variation measure, enables cross-video comparison | None needed | Low | None needed | Low | No | Coefficient of variation (std/avg), comparable across all videos regardless of scale |
| emotional_journey | captionSentiment | Global-Inherent | Critical text emotion signal for TikTok engagement | None needed | Low | Scale to [-1,1] | Low | Yes - P1 Caption Sentiment Analysis | Moving to metadata_analysis where caption data exists. Currently hardcoded placeholder |
| emotional_journey | climaxMoment | Global-Derived | Captures emotional peak timing and type, unique from density peaks | Extract position (0-1) | Low | Already normalized | Low | Yes - P1 Normalize Climax Moments | Currently string format, needs normalization to position |
| visual_overlay_analysis | climaxMoment | Global-Derived | Peak text overlay moment, message delivery climax timing | Extract position (0-1) | Low | Already normalized | Low | Yes - P1 Normalize Climax Moments | Currently timestamp string, needs normalization |
| speech_analysis | climaxMoment | Global-Derived | Peak audio energy timestamp, emphasis point | Extract position (0-1) | Low | Already normalized | Low | Yes - P1 Normalize Climax Moments | Partially implemented - defaults to video_duration/2 when audio energy analysis unavailable. Needs full implementation |
| creative_density | creativeDensityClimax | Global-Derived | Peak production intensity timing for alignment analysis | None needed | Low | Already normalized (0-1) | Low | Yes - P1 Creative Density Climax Moment | New feature to complete climax system |
| emotional_journey | globalEmotionDistribution | Global-Derived | Complete emotional composition across entire video | None needed (7 ratios) | Low | Already normalized (0-1) | Low | Yes - P1 Emotion Distribution Ratios | Replaces oversimplified dominantEmotion with full distribution |
| emotional_journey | hookEmotionDistribution | Temporal | Emotional composition in opening 3 seconds | None needed (7 ratios) | Low | Already normalized (0-1) | Low | Yes - P1 Emotion Distribution Ratios | Shows emotional hook strategy |
| emotional_journey | middleEmotionDistribution | Temporal | Emotional composition in middle section | None needed (7 ratios) | Low | Already normalized (0-1) | Low | Yes - P1 Emotion Distribution Ratios | Tracks emotional journey development |
| emotional_journey | closingEmotionDistribution | Temporal | Emotional composition in final 3 seconds | None needed (7 ratios) | Low | Already normalized (0-1) | Low | Yes - P1 Emotion Distribution Ratios | Reveals emotional resolution pattern |
| metadata_analysis | callToAction | Global-Inherent | Binary flag for CTA presence in caption | None needed | Low | None needed | Low | No | Direct engagement signal, conversion predictor |
| metadata_analysis | captionLength | Global-Inherent | Raw character count of caption | None needed | Low | Log transform + normalize | Low | No | Proxy for caption complexity and content strategy |
| metadata_analysis | ctaFollow | Global-Inherent | Binary flag for follow CTA in caption | None needed | Low | None needed | Low | Yes - P1 Simplify ctaFeatures | Specific engagement ask type |
| metadata_analysis | ctaLike | Global-Inherent | Binary flag for like CTA in caption | None needed | Low | None needed | Low | Yes - P1 Simplify ctaFeatures | Specific engagement ask type |
| metadata_analysis | ctaComment | Global-Inherent | Binary flag for comment CTA in caption | None needed | Low | None needed | Low | Yes - P1 Simplify ctaFeatures | Specific engagement ask type |
| metadata_analysis | ctaShare | Global-Inherent | Binary flag for share CTA in caption | None needed | Low | None needed | Low | Yes - P1 Simplify ctaFeatures | Specific engagement ask type |
| metadata_analysis | ctaUrgency | Global-Inherent | Binary flag for urgency CTA in caption | None needed | Low | None needed | Low | Yes - P1 Simplify ctaFeatures | Time-pressure engagement driver |
| metadata_analysis | emojiCount | Global-Inherent | Total emoji usage in caption | None needed | Low | Scale | Low | No | Proxy for emotional expression, platform-native communication |
| metadata_analysis | genericRatio | Global-Derived | Percentage of common discovery hashtags | None needed | Low | None needed | Low | Yes - P1 Expand Generic Hashtags | Shows discovery vs niche targeting strategy balance |
| metadata_analysis | hasHook | Global-Inherent | Binary flag for attention-grabbing caption start | None needed | Low | None needed | Low | Yes - P1 Expand Hook Patterns | Detects viral hook patterns in caption opening |
| metadata_analysis | hashtagCount | Global-Inherent | Total number of hashtags used | None needed | Low | Scale | Low | No | Fundamental discoverability metric, SEO strategy indicator |
| metadata_analysis | linkPresent | Global-Inherent | Binary flag for external link in caption | None needed | Low | None needed | Low | No | Commercial/promotional indicator, external traffic intent |
| metadata_analysis | mentionCount | Global-Inherent | Total number of @mentions in caption | None needed | Low | Scale | Low | No | Collaboration indicator, cross-audience potential |
| metadata_analysis | publishDayOfWeek | Global-Inherent | Day posted (0=Mon, 6=Sun) | None needed | Low | Cyclical encoding (sin/cos) | Medium | No | Audience availability patterns, weekend vs weekday strategy |
| metadata_analysis | publishHour | Global-Inherent | Hour posted (0-23) | None needed | Low | Cyclical encoding (sin/cos) | Medium | No | Peak time targeting, algorithm boost windows |
| metadata_analysis | videoDuration | Global-Inherent | Video length in seconds | None needed | Low | Log transform + scale | Low | No | Fundamental context metric, affects all other features |
| metadata_analysis | wordCount | Global-Inherent | Total words in caption | None needed | Low | Scale | Low | No | Caption complexity, denominator for densities |
| person_framing | averageFaceSize | Global-Derived | Average face size as % of frame | None needed | Low | Scale | Low | Yes - P1 Temporal Face Size | Face prominence, intimacy indicator |
| person_framing | faceSizeVariance | Temporal | Face size variance per temporal window | None needed | Low | Scale | Low | Yes - P1 Temporal Face Size | Filming dynamics - calculated for hook/middle/closing windows. Replaces distanceVariation |
| person_framing | eyeContactRate | Global-Derived | Direct audience connection metric, trust-building indicator | None needed | Low | Scale to [0,1] | Low | Yes - P1 Temporal Eye Contact | Percentage of frames with eye contact. Currently global average, needs temporal implementation (hook_eye_contact_rate, middle_eye_contact_rate, closing_eye_contact_rate) to reveal engagement patterns |
| person_framing | faceVisibilityRate | Global-Derived | Human presence indicator, content type signal | None needed | Low | Scale to [0,1] | Low | Yes - P1 Temporal Face Visibility | Percentage of frames with face detected. Distinguishes personality-driven from product/tutorial content. Needs temporal implementation (hook/middle/closing) to show face presence patterns |
| person_framing | framingChanges | Global-Inherent | Shot type transition count, visual dynamics indicator | None needed | Low | Scale | Low | Yes - P1 Temporal Framing Changes | Number of shot type changes (close→medium→wide). Different from scene cuts - tracks framing variety. Needs temporal version to show where dynamics occur |
| person_framing | framingConsistency | Global-Derived | Camera stability measure, production quality indicator | None needed | Low | Already [0,1] | Low | Yes - P0 Fix + P1 Temporal | Inverse of framing_volatility (1-volatility). Currently missing (P0 bug), returns 0. Needs implementation then temporal version |
| person_framing | framingDistribution | Global-Derived | Complete breakdown of shot types used | None needed (3 values) | Low | Already normalized [0,1] | Low | Yes - P1 Temporal Distribution | Shot type percentages (close%, medium%, far%). More informative than dominantFraming. Needs temporal to show composition evolution |
| person_framing | gaze_variance | Global-Derived | Eye contact consistency measure | None needed | Low | Already [0,1] | Low | Yes - P1 Replace categorical + Temporal | Normalized variance of eye contact scores. Replaces categorical gazeSteadiness. 0=steady, 1=variable. Needs temporal for confidence patterns |
| scene_pacing | averageSceneDuration | Global-Derived | Average scene length in seconds, editing rhythm | None needed | Low | Log transform + scale | Low | Yes - P1 Temporal Scene Duration | Fundamental pacing metric. Shows overall editing speed. Needs temporal to reveal pacing evolution |
| scene_pacing | longestScene | Global-Derived | Duration of longest scene in seconds | None needed | Low | Log transform + scale | Low | Yes - P1 Temporal Scene Extremes | Shows pacing ceiling. Completes min/avg/max trio. Needs temporal to show where long takes occur |
| scene_pacing | shortestScene | Global-Derived | Duration of shortest scene in seconds | None needed | Low | Log transform + scale | Low | Yes - P0 Global + P1 Temporal | Shows pacing floor (fastest cut). Currently missing globally, needs P0 implementation. Temporal versions (hook/middle/closing_shortest_scene) already in P1 |
| scene_pacing | scenesPerMinute | Global-Derived | Rate of scene changes per minute | None needed | Low | Scale | Low | Yes - P1 Temporal Scene Rates | Industry-standard cut frequency metric. Inverse of averageSceneDuration. Temporal version will show acceleration patterns |
| speech_analysis | avgSegmentDuration | Global-Derived | Average continuous speech length, speaking rhythm indicator | None needed | Low | Log transform + scale | Low | Yes - P1 Temporal Speech Rhythm | Measures speech delivery style, how long someone speaks before pausing. Temporal versions (hook/middle/closing) will track rhythm evolution |
| speech_analysis | energyVariance | Global-Derived | Variation in audio energy/volume levels, measures audio dynamics | None needed | Low | Scale | Low | Yes - P0 Remove Interpretation | Raw statistical variance of audio energy. Currently misused to derive background_noise_ratio which violates ML discovery. Keep variance, remove noise interpretation |
| speech_analysis | pacingVariation | Global-Derived | Standard deviation of speaking speed changes, rhythm consistency | None needed | Low | Scale | Low | Yes - P0 Implementation + P1 Temporal | Measures speaking speed consistency. Currently missing (only referenced in wrapper). Needs P0 global implementation and P1 temporal versions (hook/middle/closing) |
| speech_analysis | longestSegment | Global-Derived | Maximum continuous speech duration, shows speech delivery ceiling | None needed | Low | Log transform + scale | Low | Yes - P1 Temporal Speech Rhythm | Longest uninterrupted speech segment. Already implemented globally. Temporal versions (hook/middle/closing) already included in P1 Speech Rhythm |
| speech_analysis | shortestSegment | Global-Derived | Minimum continuous speech duration, shows speech editing floor | None needed | Low | Log transform + scale | Low | Yes - P1 Temporal Speech Rhythm | Shortest speech segment, reveals editing aggressiveness. Already implemented but not in original feature list. Needs temporal versions added to P1 |
| speech_analysis | speechCoverage | Global-Derived | Percentage of video containing speech, content type indicator | None needed | Low | Already [0,1] | Low | Yes - P0 Temporal Windows | Ratio of speech time to video duration. Already implemented. Temporal versions (middle_speech_coverage, segment coverage) exist in P0 improvements |
| speech_analysis | totalWords | Global-Inherent | Total number of words detected, content volume metric | None needed | Low | Log transform + scale | Low | Yes - P0 Temporal Windows + Derived Global | Currently implemented globally. P0 moves to temporal windows (hook_word_count, middle_word_count) with global derived as total_speech_words_derived |
| speech_analysis | vocabularyDiversity | Global-Derived | Ratio of unique words to total words (Type-Token Ratio) | None needed | Low | Already [0,1] | Low | Yes - P1 Temporal Vocabulary Diversity | Standard linguistic metric (uniqueWords/totalWords). Measures vocabulary richness, distinguishes scripted vs natural speech. Pre-normalized for cross-video comparison. P1 adds temporal versions to track evolution |
| visual_overlay_analysis | avgOverlayDuration | Global-Derived | Average text display duration across video | None needed | Low | Scale | Low | Yes - P1 Temporal Overlay Metrics | Measures reading time adequacy (avg_text_display_duration in code). Quick flash (<1s) vs sustained (>2s). P1 adds temporal to show reading strategy evolution |
| visual_overlay_analysis | uniqueOverlayCount | Global-Inherent | Number of distinct text overlays | None needed | Low | Scale | Low | Yes - P1 Temporal Overlay Metrics | Variety vs repetition indicator (unique_text_count in code). P1 adds window-local unique counts to show information progression patterns |

## Statistics

- **Total Features**: 56
- **Global Features**: 52
- **Temporal Features**: 4

## Notes

- Source column is for human reference only - ML models treat all features equally
- Window Type indicates if feature is calculated once (Global) or per window/segment (Temporal)
- Transformations are required for algorithm compatibility

## Climax Moment System

The four climax moments (emotional, visual, speech, density) form a coordination analysis system:
- All normalized to 0-1 position for cross-video comparison
- Alignment patterns reveal production quality and content strategy
- Variance across climaxes indicates coordination level
- Essential for discovering optimal peak timing patterns

## Emotion Distribution System

Complete emotional composition tracking replacing single dominantEmotion:
- Each distribution contains 7 emotion ratios (joy, surprise, neutral, sad, anger, fear, disgust)
- Global distribution shows overall emotional composition
- Temporal distributions (hook/middle/closing) reveal emotional journey
- Enables genre classification and emotional arc pattern discovery

## Feature Notes

### accelerationPattern
- **Implementation**: Compares density between Hook, Middle (aggregated), and Closing windows
- **Video Length Handling**: Uses average middle density across 3-5 segments to remain comparable across different video durations
- **Categories**: "steady_increase", "steady_decrease", "peak_middle", "peak_end", "variable"

