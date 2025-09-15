# Failed Validation Features

**Created**: 2025-01-09
**Purpose**: Track features that failed validation for FeaturesMLMVPv2.md and document reasons for rejection

## Overview

Features listed here were reviewed but rejected during the validation process due to:
- Duplicity with ImprovementsMLMVP.md
- Architectural incompatibility
- Redundancy violations
- Other validation failures

## Failed Features Table

| Source | Feature | Failed Reason | Covered By | Severity | Action Taken | Notes |
|--------|---------|---------------|------------|----------|--------------|-------|
| creative_density | avgDensity | Redundancy - covered by P0 improvements | ImprovementsMLMVP.md P0 "Derived Global Metrics" as avg_density_derived | High | Added to P0 improvements | Violates "no redundant storage" principle - can be derived from temporal window counts |
| creative_density | deadZones | Threshold too strict for short-form | None - feature not useful | High | Skip entirely | 2+ second threshold with ZERO elements is catastrophic failure, not useful pattern. Would be 0 for 99% of videos |
| creative_density | elementCooccurrence | Pre-computed correlation ML should discover | Temporal window element counts | High | Skip - violates ML discovery principle | ML can discover element relationships from window counts. Pre-computing correlations removes ML's ability to find patterns |
| creative_density | emptySeconds | Threshold too strict, not actionable | Temporal window counts indicate sparsity | High | Skip entirely | Requires ALL 6 element types = 0, rarely occurs. ML can infer sparsity from low counts in windows |
| emotional_journey | emotionalContrastMoments | Pre-computed valence violates ML discovery | Emotion distributions per window | High | Skip - violates ML discovery principle | Pre-defines positive/negative valence. With per-window emotion distributions, ML can discover contrast patterns naturally |
| emotional_journey | emotionalDiversity | Redundancy with emotion distributions | Emotion distribution ratios (P1) | High | Skip - redundant storage | Directly derivable from emotion distributions. Including both would create multicollinearity, especially problematic for K-means clustering |
| emotional_journey | emotionalIntensity | Measures detection confidence not emotion | None - technical artifact | Medium | Skip entirely | Average MediaPipe confidence score. Reflects detection quality (lighting, face clarity) not actual emotional intensity. Not a content feature |
| emotional_journey | emotionProgression | Architectural incompatibility | Emotion distributions per window (P1) | High | Skip - incompatible architecture | Uses arbitrary 10 equal sections instead of Hook/Middle/Closing. Variable window sizes make cross-video comparison invalid. Also uses flawed confidence as intensity |
| emotional_journey | emotionTransitions | Redundancy - covered by P0 improvements | Per-window transition counts in P0 | High | Already in P0 improvements | Global count will be replaced by hook/middle/closing_emotion_transitions. Global can be derived from sum |
| emotional_journey | first_emotion_transition | Not implemented - phantom feature | Per-window emotion transitions | High | Skip - never implemented | Feature described but no code exists. Value captured by temporal window transitions. ML can discover when dynamism starts from window data |
| metadata_analysis | commentCount | Outcome metric not content feature | None - target variable component | High | Skip - not controllable | Post-publication metric that cannot be controlled during creation. Part of engagement rate target, not a predictive feature. Would cause data leakage |
| metadata_analysis | hasExclamation | Too simplistic to be meaningful | ctaUrgency and other features | Low | Skip - weak signal | Single punctuation mark provides minimal signal. No context or intensity. Better urgency/excitement signals in ctaUrgency and emojiCount |
| metadata_analysis | hashtagBreakdown | Complete redundancy with existing features | hashtagCount + genericRatio | High | Skip - derivable | generic_count = hashtagCount × genericRatio, niche_count = hashtagCount × (1-genericRatio). Perfect multicollinearity risk, especially for K-means |
| metadata_analysis | hasQuestion | Too simplistic to be meaningful | CTA features and hasHook | Low | Skip - weak signal | Single punctuation mark provides minimal signal. No context or intensity. Better engagement signals in ctaComment and question-based hooks |
| person_framing | distanceVariation | Not implemented - replaced by temporal | faceSizeVariance per window | High | Skip - never implemented | Hardcoded to 0. Completely replaced by temporal face size variance metrics which show WHERE dynamics occur |
| person_framing | dominantFraming | Redundant bucketing of face size | Temporal face size averages | Medium | Skip - derivable | Just buckets avg face size into 3 categories. ML can learn optimal thresholds from raw face sizes. Current implementation uses mean not mode |
| person_framing | stabilityScore | Technical impossibility - requires optical flow | Component metrics exist separately | High | Skip - cannot implement properly | True camera shake detection needs optical flow/motion vectors. Face tracking conflates subject and camera movement. Composite metric would mix unrelated stability types. ML can use existing components (framing_volatility, faceSizeVariance) |
| scene_pacing | totalScenes | Redundancy - mathematically derivable | sceneChangeCount + 1 | High | Skip - redundant storage | Total number of scenes = sceneChangeCount + 1 always. Complete redundancy with sceneChangeCount from creative_density. Violates "no redundant storage" principle |
| speech_analysis | repetitionRate | Low signal value without filtering | None - requires complex NLP | High | Skip entirely | Simple repetition rate includes filler words ("and", "um", "like") making it noisy. Meaningful repetition requires stop word filtering and content/function word separation. Current implementation tracks all phrases without discrimination |
| speech_analysis | silenceRatio | Redundancy - mathematically derivable | speechCoverage | High | Skip - redundant storage | Perfect mathematical redundancy: silenceRatio = 1 - speechCoverage. Including both would cause perfect multicollinearity, especially problematic for K-means clustering |
| speech_analysis | silentMoments | Pre-computed interpretation + arbitrary threshold | Raw pause data without thresholds | High | Skip - violates ML discovery | Uses arbitrary 1-second threshold to define "significant" pause. Categorizes as strategic/dramatic/awkward (interpretive). Temporal resolution limits would miss sub-second pauses anyway |
| speech_analysis | wordsPerMinute | Redundancy - mathematically derivable | totalWords, speechCoverage, videoDuration | High | Skip - redundant storage | Perfect mathematical redundancy: wordsPerMinute = totalWords / (speechCoverage × videoDuration) × 60. Creates multicollinearity. ML can discover if high totalWords + high speechCoverage = fast pace. LLM interpretation layer can calculate WPM when needed for human-readable output |
| visual_overlay_analysis | burstPatterns | Pre-computed interpretation + temporal granularity mismatch | Temporal window overlay counts | High | Skip - violates ML discovery + architecture limitation | Uses arbitrary 5-second windows and 3+ overlay threshold to define "burst". Temporal window architecture (Hook/Middle/Closing) loses second-level granularity needed for true burst detection. Would require deep learning with sequential data to properly capture 1-3 second clustering patterns |
| person_framing | global_framing_changes | Perfect multicollinearity with temporal counts | Temporal framing changes per window | High | Remove global count - keep temporal only | Global framing_changes = sum(hook + middle + closing changes). Perfect deterministic relationship provides no new information for ML. Creates multicollinearity where models cannot determine feature importance |
| improvements | Derived Global Metrics | Perfect multicollinearity - deterministic sums | Temporal Windows element counts | High | Remove P0 improvement entirely | Creates global sums (total_elements_derived, total_overlays_derived, etc.) that are perfect mathematical sums of temporal window counts. Violates "no redundant storage" principle. ML models cannot determine feature importance between parts and sum. K-means clustering especially affected by multicollinearity |
| improvements | cuts_per_second | Deterministic derivation - perfect formula | sceneChangeCount / videoDuration | High | Remove - deterministic calculation | Perfect mathematical formula: cuts_per_second = sceneChangeCount / videoDuration. Creates multicollinearity with existing features. ML models can learn this relationship independently. Including both raw features and their ratio prevents models from determining feature importance |
| improvements | framingConsistency | Deterministic derivation - perfect inverse | 1 - framing_volatility | High | Remove - deterministic calculation | Perfect mathematical formula: framingConsistency = 1.0 - framing_volatility. Pure inverse relationship adds no new information. Creates perfect negative correlation with framing_volatility. ML models cannot distinguish between the two features |
| metadata_analysis | captionLength | High collinearity with existing features | wordCount + emojiCount + hashtagCount + mentionCount | High | Remove - redundant combination | Character count largely redundant with word count plus special character features. Hashtag/mention length signals not valuable per requirements. Punctuation density shown to be weak signal (see hasExclamation/hasQuestion). Creates multicollinearity for K-means clustering |

## Column Definitions

- **Source**: Original analysis flow from FeaturesMLMVP.md
- **Feature**: Feature name that failed validation
- **Failed Reason**: Primary reason for rejection (Duplicity/Redundancy/Architectural/Hardcoded)
- **Covered By**: Where the functionality is already handled (if applicable)
- **Severity**: Impact level (High = critical redundancy, Medium = partial overlap, Low = minor issue)
- **Action Taken**: What was done (Added to improvements/Removed/Deferred to Phase 2)
- **Notes**: Additional context about the failure

## Statistics

- **Total Failed**: 28
- **Due to Redundancy**: 13
- **Due to Threshold/Calibration**: 2
- **Due to ML Discovery Violation**: 4
- **Due to Architecture**: 2
- **Due to Hardcoded/Fake**: 2
- **Due to Technical Artifact**: 1
- **Due to Outcome Metric**: 1
- **Due to Weak Signal**: 3
- **Due to Technical Impossibility**: 1

## Categories of Failure

### Redundancy Violations
Features that violate "no redundant storage" principle by duplicating information available in temporal windows.

### Duplicity with Improvements
Features already covered by P0/P1/P2/P3 improvements in ImprovementsMLMVP.md.

### Architectural Incompatibility
Features that don't fit Global-Inherent, Global-Derived, or Temporal window structure.

### Hardcoded/Fake Data
Features with no real implementation or static values.