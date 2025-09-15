# Features Selected for ML MVP Pipeline
**Created**: 2025-01-28
**Purpose**: Consolidated table of features passing benchmark criteria across all flows

## Important Notes

### Feature Selection Criteria
- Features listed here are compatible with **AT LEAST ONE** ML algorithm (Random Forest and/or K-means)
- A feature only needs to work with one algorithm to be included in this document
- Features incompatible with one algorithm will also appear in FutureExtraFeatures.md with rejection reasons

### "Seems Repetitive?" Column
- This flags redundancy **WITHIN each flow only** (intra-flow repetition)
- We do NOT flag repetition between different flows in Phase 1
- Example: Flag if creative_density has two similar density metrics
- DO NOT flag if creative_density metric is similar to scene_pacing metric

## Phase 1: Individual Flow Features
*Note: Duplicates across flows are intentionally preserved in Phase 1. Cross-flow deduplication happens in Phase 2.*

## Creative Density Features

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|---------------|-------------------|---------------|---------------|-------------------|---------------|-------------------|-------|
| creative_density | accelerationPattern | Captures overall pacing patterns not covered by temporal window slopes | Yes | One-hot encode 4 binary features | Low | Yes | Label encode 0-3 then scale | Low | No | Complements temporal slopes with overall pattern |
| creative_density | avgDensity | Fundamental metric for overall video busy-ness, provides global baseline | Yes | Already numerical | Low | Yes | Scale to 0-1 range | Low | No | Raw measurement, not captured by temporal windows |
| creative_density | deadZones | Critical for engagement - identifies content gaps of 2+ seconds | Yes | Extract count, total duration, max duration | Medium | Yes | Extract features then scale | Medium | Maybe | Related to emptySeconds but captures continuous periods |
| creative_density | elementCooccurrence | Shows which element pairs work together, content composition patterns | Yes | Use counts as features directly | Low | Yes | Use counts then scale | Low | No | Complete pair co-occurrence data |
| creative_density | elementCounts | Raw counts of each element type, content composition volume | Yes | Use 6 values as features | Low | Yes | Use 6 values then scale | Low | No | Complements density (rate) with absolute volumes |
| creative_density | emptySeconds | Measures overall sparsity, total empty time regardless of continuity | Yes | Extract count and percentage of video | Low | Yes | Count and percentage then scale | Low | Maybe | Related to deadZones but captures scattered empty seconds |
| creative_density | maxDensity | Peak creative intensity, shows production ceiling | Yes | Already numerical | Low | Yes | Scale to 0-1 range | Low | No | Fundamental statistic, complements avgDensity |
| creative_density | minDensity | Baseline activity level, shows production floor | Yes | Already numerical | Low | Yes | Scale to 0-1 range | Low | No | Completes min/avg/max trio |
| creative_density | sceneChangeCount | Fundamental editing metric, shows production pace and engagement | Yes | Already numerical | Low | Yes | Scale to 0-1 range | Low | No | Raw count of cuts, unique from density metrics |
| creative_density | stdDeviation | Measures consistency vs variation in density across entire video | Yes | Already numerical | Low | Yes | Scale to 0-1 range | Low | Maybe | Related to volatility but pure statistical measure |
| creative_density | totalElements | Absolute production volume, shows total content effort | Yes | Already numerical | Low | Yes | Scale to 0-1 range | Low | Maybe | Related to avgDensity×duration but captures absolute scale |
| creative_density | volatility | Normalized variation measure, enables cross-video comparison | Yes | Already numerical | Low | Yes | Scale to 0-1 range | Low | Maybe | Coefficient of variation (stdDev/avg), different from absolute variation |

### Creative Density - Repetition Analysis
*For features flagged as Yes/Maybe in "Seems Repetitive?" above*

| Feature | Related Features | Repetition Type | Explanation |
|---------|-----------------|-----------------|-------------|
| deadZones | emptySeconds | Complementary | Both track empty periods but differently - deadZones captures continuous 2+ second gaps, emptySeconds captures individual empty seconds |
| emptySeconds | deadZones | Complementary | Captures total scattered empty seconds vs continuous gaps - different sparsity measures |
| stdDeviation | volatility | Statistical Variant | Both measure variation but stdDeviation is absolute spread, volatility is normalized (CV) |
| totalElements | avgDensity, elementCounts | Statistical Variant | Related to avgDensity×duration and sum of elementCounts, but provides absolute scale |
| volatility | stdDeviation | Statistical Variant | Coefficient of variation (stdDev/avg), provides normalized variation vs absolute |

---

## Emotional Journey Features

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|---------------|-------------------|---------------|---------------|-------------------|---------------|-------------------|-------|
| emotional_journey | captionSentiment | Critical text emotion signal for TikTok engagement | Yes | Already numerical | Low | Yes | Scale to [-1,1] | Low | No | Only text sentiment feature, pragmatic exception |
| emotional_journey | climaxMoment | Captures emotional peak timing and type, unique from density peaks | Yes | Flatten: extract timestamp, emotion, confidence as 3 features | Low | Yes | Extract confidence as numeric, encode emotion, scale | Medium | No | Shows WHERE and WHAT emotion peaks |
| emotional_journey | dominantEmotion | Overall emotional baseline, frequency-based characterization | Yes | One-hot encode (7-10 categories) | Low | Yes | Label encode (0-9) + scale | Low | No | Most common emotion, different from peak intensity |
| emotional_journey | emotionalContrastMoments | Captures contrast magnitude and surprise factor, key for engagement | Yes | Extract count, first transition time, max contrast | Medium | Partial | Extract count + avg transition time | Medium | Maybe | Related to transitions but captures contrast intensity |
| emotional_journey | emotionalDiversity | Measures emotional variety vs monotony, distribution evenness | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Shows HOW dominant the dominant emotion is |
| emotional_journey | emotionalIntensity | Overall emotional strength/power, key engagement metric | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Average intensity vs climax peak intensity |
| emotional_journey | emotionProgression | Fixed 4-section emotional journey, complements temporal density | Yes | Flatten: 4 sections × 2 values = 8 features | Low | Yes | Extract intensities, encode dominants, scale | Medium | No | Shows emotional narrative through time |
| emotional_journey | emotionTransitions | Simple count of emotion changes, emotional dynamism metric | Yes | Already numerical | Low | Yes | Scale | Low | Maybe | Count of changes, complements contrast magnitude |
| emotional_journey | first_emotion_transition | Timing of first emotional change, shows when dynamism starts | Yes | Already numerical (time) | Low | Yes | Scale, handle nulls | Low | Maybe | May overlap with hook window if early |

### Emotional Journey - Repetition Analysis
*For features flagged as Yes/Maybe in "Seems Repetitive?" above*

| Feature | Related Features | Repetition Type | Explanation |
|---------|-----------------|-----------------|-------------|
| emotionalContrastMoments | emotionTransitions, transitionSmoothness | Complementary | All track transitions but contrast captures magnitude/intensity of shifts |
| emotionTransitions | emotionalContrastMoments | Complementary | Simple count vs magnitude/timing of transitions |
| first_emotion_transition | hook_0to3s_surprise_score | Complementary | Timing of first change vs intensity of opening - may overlap if < 3s |

---

## Person Framing Features

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|---------------|-------------------|---------------|---------------|-------------------|---------------|-------------------|-------|
| person_framing | averageFaceSize | Face prominence indicator, intimacy/connection proxy | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Raw spatial measurement |
| person_framing | distanceVariation | Visual dynamics indicator, filming variety measure | Yes | Already numerical | Low | Yes | Scale | Low | No | Raw statistical variation |
| person_framing | dominantFraming | Most common shot type (close/medium/wide) | Yes | One-hot encode (close/medium/wide) | Low | Yes | Label encode (0-2) + scale | Low | No | Simple 3-category objective classification |
| person_framing | eyeContactRate | Direct audience connection metric | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Critical for trust/engagement |
| person_framing | faceVisibilityRate | Human presence indicator, content type signal | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Key for personality-driven content |
| person_framing | framingChanges | Number of shot type/framing transitions | Yes | Already numerical | Low | Yes | Scale | Low | No | Raw count of editing transitions, visual dynamics |
| person_framing | framingConsistency | How stable the framing remains (0-1) | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Raw stability measure, complements framingChanges |
| person_framing | framingDistribution | Breakdown of shot types used | Yes | Flatten: extract percentages for each type | Low | Yes | Extract percentages, scale | Low | No | Complete shot composition breakdown (close%, medium%, wide%) |
| person_framing | gazeSteadiness | How stable eye direction is | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Eye movement stability, confidence/trust indicator |
| person_framing | stabilityScore | Camera shake/stability (0-1) | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Camera motion steadiness, different from framingConsistency (shot changes) |

### Person Framing - Repetition Analysis
*For features flagged as Yes/Maybe in "Seems Repetitive?" above*

| Feature | Related Features | Repetition Type | Explanation |
|---------|-----------------|-----------------|-------------|
| *To be added after identifying repetitive features* | | | |

---

## Visual Overlay Features

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|---------------|-------------------|---------------|---------------|-------------------|---------------|-------------------|-------|
| visual_overlay_analysis | avgOverlayDuration | Reading time adequacy, cognitive load indicator, pacing characteristic | Yes | Already numerical | Low | Yes | Scale | Low | No | How long overlays persist vs when they appear |
| visual_overlay_analysis | burstPatterns | Attention-grabbing technique count, intensity moments, production style | Yes | Already numerical | Low | Yes | Scale | Low | No | Counts clustered overlay events, not temporal position |
| visual_overlay_analysis | climaxMoment | Peak text overlay moment, message delivery climax timing | Yes | Extract timestamp, convert to relative position | Medium | Partial | Extract timestamp only | Medium | No | Text peak vs overall density peak, different signals |
| visual_overlay_analysis | uniqueOverlayCount | Number of distinct overlay contents | Yes | Already numerical | Low | Yes | Scale | Low | No | Captures content variety vs repetition, inherently global metric |

### Visual Overlay - Repetition Analysis
*For features flagged as Yes/Maybe in "Seems Repetitive?" above*

| Feature | Related Features | Repetition Type | Explanation |
|---------|-----------------|-----------------|-------------|
| *To be added after identifying repetitive features* | | | |

---

## Scene Pacing Features

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|---------------|-------------------|---------------|---------------|-------------------|---------------|-------------------|-------|
| scene_pacing | averageSceneDuration | Average scene length in seconds, editing rhythm | Yes | Already numerical | Low | Yes | Scale (log transform) | Low | No | Fundamental pacing metric, works with temporal windows |
| scene_pacing | longestScene | Duration of longest scene in seconds | Yes | Already numerical | Low | Yes | Scale (log transform) | Low | No | Shows pacing range/variety, complements average |
| scene_pacing | scenesPerMinute | Rate of scene changes per minute | Yes | Already numerical | Low | Yes | Scale | Low | No | Cut frequency metric, complements averageSceneDuration |
| scene_pacing | shortestScene | Duration of shortest scene in seconds | Yes | Already numerical | Low | Yes | Scale (log transform) | Low | No | Shows pacing floor, completes min/avg/max trio |
| scene_pacing | totalScenes | Total number of scene changes/cuts | Yes | Already numerical | Low | Yes | Scale | Low | No | Absolute cut count, complements rate metrics |

### Scene Pacing - Repetition Analysis
*For features flagged as Yes/Maybe in "Seems Repetitive?" above*

| Feature | Related Features | Repetition Type | Explanation |
|---------|-----------------|-----------------|-------------|
| *To be added after identifying repetitive features* | | | |

---

## Speech Analysis Features

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|---------------|-------------------|---------------|---------------|-------------------|---------------|-------------------|-------|
| speech_analysis | avgSegmentDuration | Average continuous speech length, speaking rhythm | Yes | Already numerical | Low | Yes | Scale (log transform) | Low | No | Different from visual pacing, shows speech delivery style |
| speech_analysis | climaxMoment | Peak audio energy timestamp, emphasis point | Yes | Already numerical (position) | Low | Yes | Scale to [0,1] | Low | No | Objective peak detection, works as binary per window |
| speech_analysis | energyVariance | Variation in audio energy/volume levels | Yes | Already numerical | Low | Yes | Scale | Low | No | Core of energy trio with avgAudioEnergy and peaks |
| speech_analysis | longestSegment | Maximum continuous speech duration | Yes | Already numerical | Low | Yes | Scale (log transform) | Low | No | Complements avgSegmentDuration with range info |
| speech_analysis | pacingVariation | Standard deviation of speaking speed changes | Yes | Already numerical | Low | Yes | Scale | Low | No | Objective pace dynamics, complements wordsPerMinute |
| speech_analysis | repetitionRate | Frequency of repeated words/phrases | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Important for memorable content, TikTok-specific pattern |
| speech_analysis | silenceRatio | Percentage of video that is silent | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Overall silence percentage, complements silentMoments |
| speech_analysis | silentMoments | Count of significant pauses for effect | Yes | Already numerical | Low | Yes | Scale | Low | No | Pause frequency, complements silenceRatio |
| speech_analysis | speechCoverage | Percentage of video containing speech | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Voice content ratio, different from silenceRatio |
| speech_analysis | totalWords | Total number of words detected | Yes | Already numerical | Low | Yes | Scale (log transform) | Low | No | Content volume indicator, different from WPM and coverage |
| speech_analysis | uniqueWords | Count of distinct words used | Yes | Already numerical | Low | Yes | Scale | Low | No | Vocabulary diversity, complements totalWords with variety metric |
| speech_analysis | wordsPerMinute | Speaking rate in words per minute | Yes | Already numerical | Low | Yes | Scale | Low | No | Fundamental speech speed metric, essential for pace analysis |

### Speech Analysis - Repetition Analysis
*For features flagged as Yes/Maybe in "Seems Repetitive?" above*

| Feature | Related Features | Repetition Type | Explanation |
|---------|-----------------|-----------------|-------------|
| *To be added after identifying repetitive features* | | | |

---

## Visual Overlay Features

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|---------------|-------------------|---------------|---------------|-------------------|---------------|-------------------|-------|
| *To be added after visual_overlayMLA.md review* | | | | | | | | | | |

### Visual Overlay - Repetition Analysis
*For features flagged as Yes/Maybe in "Seems Repetitive?" above*

| Feature | Related Features | Repetition Type | Explanation |
|---------|-----------------|-----------------|-------------|
| *To be added after identifying repetitive features* | | | |

---

## Metadata Analysis Features

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|---------------|-------------------|---------------|---------------|-------------------|---------------|-------------------|-------|
| metadata_analysis | callToAction | Binary flag for CTA presence, direct engagement signal | Yes | Already binary (0/1) | Low | Yes | Scale to [0,1] | Low | No | Simple CTA detection |
| metadata_analysis | captionLength | Raw character count, proxy for caption complexity | Yes | Already numerical | Low | Yes | Log transform + normalize | Low | No | Important for caption strategy |
| metadata_analysis | commentCount | Core engagement metric, potential ML target | Yes | Already numerical | Low | Yes | Log scale + normalize | Low | No | Fundamental success metric |
| metadata_analysis | ctaFeatures | Granular CTA breakdown (follow/like/comment/share/urgency) | Yes | Flatten: extract 5 boolean features | Low | Yes | Extract booleans, scale | Low | No | Detailed engagement drivers |
| metadata_analysis | emojiCount | Total emoji usage, proxy for emotional expression | Yes | Already numerical | Low | Yes | Scale | Low | No | Platform-native communication |
| metadata_analysis | emojiDensity | Emojis per word ratio, style indicator | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Normalized emoji intensity |
| metadata_analysis | engagementRate | THE key success metric: (L+C+S)/Views | Yes | Already numerical | Low | Yes | Log scale + normalize | Low | No | Industry-standard KPI |
| metadata_analysis | genericRatio | Percentage of common discovery hashtags | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Update needed: fyp, viral, tiktok, foryou, funny, duet, smallbusiness, trending, explore, foryoupage, trendingvideo, tiktokcreator, contentcreator, tiktokchallenge |
| metadata_analysis | hasExclamation | Excitement/enthusiasm indicator via punctuation | Yes | Already binary (0/1) | Low | Yes | Scale to [0,1] | Low | No | Simple engagement signal |
| metadata_analysis | hasHook | Attention-grabbing caption start detection | Yes | Already binary (0/1) | Low | Yes | Scale to [0,1] | Low | No | Update needed: wait for it, watch till, won't believe, pov:, story time, here's how, the secret, you won't believe what happens next, the secret behind, this is what happens when, if you're seeing this, the truth they don't want you to know, stop scrolling, is it just me, i can't believe i just discovered, it's not me it's you, did you know that, are you having trouble, have you ever, what if i told you, why is no one talking about, how would you react, 99%, 90%, x things, 3 ways, 5 tips, number one, struggling with, this mistake, stop doing, instead try, here's why, the reason why, things you didn't know, nobody talks about, the truth about, everything you know, hard truth, underestimated, this changed my, what happened when, the story of how, i challenged myself, here's what happened, warning:, breaking:, don't hate me, unpopular opinion, hot take:, confession: |
| metadata_analysis | hashtagBreakdown | Absolute counts of generic vs niche hashtags | Partial | Extract generic_count, niche_count only | Medium | No | Too interpretive | High | Maybe | MODIFICATION NEEDED: Currently returns dict with {total, generic, niche, genericRatio, strategy}. For ML, flatten to just 2 features: generic_count (int) and niche_count (int). Remove 'total' (redundant with hashtagCount), remove 'genericRatio' (keeping as separate feature), remove 'strategy' (semantic interpretation). In precompute_functions_full.py lines 1183-1189, change return to only include generic and niche counts as separate numerical features, not nested dict |
| metadata_analysis | hashtagCount | Total number of hashtags used | Yes | Already numerical | Low | Yes | Scale | Low | No | Fundamental discoverability metric |
| metadata_analysis | hasQuestion | Inquiry/discussion driver via punctuation | Yes | Already binary (0/1) | Low | Yes | Scale to [0,1] | Low | No | Questions drive comments specifically |
| metadata_analysis | likeCount | Core engagement metric, potential ML target | Yes | Already numerical | Low | Yes | Log scale + normalize | Low | No | Fundamental success metric |
| metadata_analysis | linkPresent | Commercial/promotional content indicator | Yes | Already binary (0/1) | Low | Yes | Scale to [0,1] | Low | No | External traffic intent signal |
| metadata_analysis | mentionCount | Collaboration/social engagement indicator | Yes | Already numerical | Low | Yes | Scale | Low | No | Cross-audience traffic potential |
| metadata_analysis | mentionDensity | Mentions per word ratio, tagging intensity | Yes | Already numerical | Low | Yes | Scale to [0,1] | Low | No | Normalized mention strategy |
| metadata_analysis | publishDayOfWeek | Day posted (0=Mon, 6=Sun), timing strategy | Yes | Already numerical (0-6) | Low | Yes | Cyclical encoding (sin/cos) | Medium | No | Critical for audience availability patterns |
| metadata_analysis | publishHour | Hour posted (0-23), peak time targeting | Yes | Already numerical (0-23) | Low | Yes | Cyclical encoding (sin/cos) | Medium | No | Essential for algorithm boost windows |
| metadata_analysis | shareCount | CRITICAL virality metric, strongest engagement | Yes | Already numerical | Low | Yes | Log scale + normalize | Low | No | Best predictor of viral success |
| metadata_analysis | videoDuration | Fundamental video characteristic, length in seconds | Yes | Already numerical | Low | Yes | Scale (log transform) | Low | No | Context for all metrics, affects engagement |
| metadata_analysis | viewCount | Essential reach metric, denominator for engagement | Yes | Already numerical | Low | Yes | Log scale + normalize | Low | No | Base for all percentage calculations |
| metadata_analysis | wordCount | Word-based caption measure, denominator for densities | Yes | Already numerical | Low | Yes | Scale | Low | No | Needed for emojiDensity and mentionDensity |

### Metadata Analysis - Repetition Analysis
*For features flagged as Yes/Maybe in "Seems Repetitive?" above*

| Feature | Related Features | Repetition Type | Explanation |
|---------|-----------------|-----------------|-------------|
| *To be added after identifying repetitive features* | | | |

---

## Temporal Markers Features

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|---------------|-------------------|---------------|---------------|-------------------|---------------|-------------------|-------|
| *To be added after temporal_markersMLA.md review* | | | | | | | | | | |

### Temporal Markers - Repetition Analysis
*For features flagged as Yes/Maybe in "Seems Repetitive?" above*

| Feature | Related Features | Repetition Type | Explanation |
|---------|-----------------|-----------------|-------------|
| *To be added after identifying repetitive features* | | | |

---

## Next Steps

- Complete review of remaining 8 visual_overlay features
- Review temporal_markers features  
- Cross-flow deduplication after all reviews complete
