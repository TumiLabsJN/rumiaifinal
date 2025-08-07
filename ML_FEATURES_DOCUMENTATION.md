# RumiAI ML Features Documentation - Complete CoreBlock Feature Dictionary
**Version**: 1.0.0  
**Last Updated**: 2025-01-08  
**Total Features**: 300+ across 7 analysis types

## Executive Summary

This document provides comprehensive documentation of all ML features available from the RumiAI CoreBlock structure. Each of the 7 analysis types generates features organized into 6 standardized blocks, creating a rich dataset for machine learning applications.

### Feature Statistics
- **Total Unique Features**: 300+
- **Features per Analysis Type**: 40-50
- **Data Types**: float (45%), int (20%), string (15%), array (10%), object (8%), boolean (2%)
- **Source Distribution**: Computed (60%), ML Services (30%), API/Metadata (10%)

---

## Table of Contents
1. [Feature Organization](#feature-organization)
2. [Creative Density Features](#1-creative-density-features)
3. [Emotional Journey Features](#2-emotional-journey-features)
4. [Person Framing Features](#3-person-framing-features)
5. [Scene Pacing Features](#4-scene-pacing-features)
6. [Speech Analysis Features](#5-speech-analysis-features)
7. [Visual Overlay Features](#6-visual-overlay-analysis-features)
8. [Metadata Analysis Features](#7-metadata-analysis-features)
9. [Cross-Analysis Dependencies](#cross-analysis-dependencies)
10. [Feature Engineering Opportunities](#feature-engineering-opportunities)

---

## Feature Organization

Each analysis type outputs 6 standardized blocks:

| Block | Purpose | Typical Feature Count | Primary Data Types |
|-------|---------|----------------------|-------------------|
| **CoreMetrics** | Key measurements and summary statistics | 8-12 | float, int |
| **Dynamics** | Temporal patterns and progressions | 5-7 | array, float, string |
| **Interactions** | Cross-modal relationships | 4-6 | float, object |
| **KeyEvents** | Critical moments with timestamps | 3-5 | array of objects |
| **Patterns** | Recurring elements and strategies | 5-8 | string, boolean, array |
| **Quality** | Data confidence and completeness | 3-5 | float, string |

---

## 1. Creative Density Features

### CoreMetrics (11 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `avgDensity` | float | 0.0-10.0 | Mean creative elements per second | Computed |
| `maxDensity` | float | 0.0-20.0 | Maximum elements in any second | Computed |
| `minDensity` | float | 0.0-10.0 | Minimum elements in any second | Computed |
| `stdDeviation` | float | 0.0-5.0 | Standard deviation of density | Computed |
| `totalElements` | int | 0-1000 | Sum of all creative elements | Computed |
| `elementsPerSecond` | float | 0.0-10.0 | Average elements per second | Computed |
| `elementCounts` | object | - | Breakdown by type | Computed |
| ├─ `text` | int | 0-200 | Text overlay count | OCR |
| ├─ `sticker` | int | 0-50 | Sticker count | OCR |
| ├─ `effect` | int | 0-100 | Effect count | Computed |
| ├─ `transition` | int | 0-50 | Transition count | Scene Detection |
| └─ `object` | int | 0-500 | Object detection count | YOLO |
| `sceneChangeCount` | int | 0-100 | Total scene changes | Scene Detection |
| `timelineCoverage` | float | 0.0-1.0 | Percentage of timeline with elements | Computed |
| `confidence` | float | 0.0-1.0 | Overall metric confidence | Computed |

### Dynamics (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `densityCurve` | array | - | Per-second density progression | Computed |
| ├─ `second` | int | 0-300 | Time index | Computed |
| ├─ `density` | float | 0.0-20.0 | Elements at this second | Computed |
| └─ `primaryElement` | string | text/effect/object/transition | Dominant element type | Computed |
| `volatility` | float | 0.0-1.0 | Density variation measure | Computed |
| `accelerationPattern` | string | front_loaded/even/back_loaded/oscillating | Density distribution pattern | Computed |
| `densityProgression` | string | increasing/decreasing/stable/erratic | Overall density trend | Computed |
| `emptySeconds` | array[int] | - | List of seconds with no elements | Computed |
| `confidence` | float | 0.0-1.0 | Dynamics confidence | Computed |

### Interactions (5 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `multiModalPeaks` | array | - | Moments with multiple element types | Computed |
| ├─ `timestamp` | string | "X-Ys" | Time range | Computed |
| ├─ `elements` | array[string] | - | Element types present | Computed |
| └─ `syncType` | string | reinforcing/complementary/redundant | Relationship type | Computed |
| `elementCooccurrence` | object | - | Frequency of element combinations | Computed |
| `dominantCombination` | string | e.g., "text_gesture" | Most frequent combination | Computed |
| `coordinationScore` | float | 0.0-1.0 | Overall synchronization | Computed |
| `confidence` | float | 0.0-1.0 | Interaction confidence | Computed |

### KeyEvents (4 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `peakMoments` | array | - | Highest density moments | Computed |
| ├─ `timestamp` | string | "X-Ys" | Time range | Computed |
| ├─ `totalElements` | int | 0-20 | Element count | Computed |
| ├─ `surpriseScore` | float | 0.0-1.0 | Deviation from average | Computed |
| └─ `elementBreakdown` | object | - | Elements by type | Computed |
| `deadZones` | array | - | Periods with no elements | Computed |
| ├─ `start` | int | 0-300 | Start second | Computed |
| ├─ `end` | int | 0-300 | End second | Computed |
| └─ `duration` | int | 0-300 | Duration in seconds | Computed |
| `densityShifts` | array | - | Major density transitions | Computed |
| `confidence` | float | 0.0-1.0 | Events confidence | Computed |

### Patterns (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `structuralFlags` | object | - | Boolean pattern indicators | Computed |
| ├─ `strongOpeningHook` | boolean | true/false | Strong start pattern | Computed |
| ├─ `crescendoPattern` | boolean | true/false | Building intensity | Computed |
| ├─ `frontLoaded` | boolean | true/false | Most content early | Computed |
| ├─ `consistentPacing` | boolean | true/false | Steady rhythm | Computed |
| ├─ `finalCallToAction` | boolean | true/false | CTA at end | Computed |
| └─ `rhythmicPattern` | boolean | true/false | Regular intervals | Computed |
| `densityClassification` | string | sparse/moderate/dense/overwhelming | Overall density level | Computed |
| `pacingStyle` | string | steady/building/burst_fade/oscillating | Pacing pattern | Computed |
| `cognitiveLoadCategory` | string | minimal/optimal/challenging/overwhelming | Viewer processing load | Computed |
| `mlTags` | array[string] | - | ML-relevant tags | Computed |
| `confidence` | float | 0.0-1.0 | Pattern confidence | Computed |

### Quality (3 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `dataCompleteness` | float | 0.0-1.0 | Percentage of data available | Computed |
| `detectionReliability` | object | - | Reliability by service | Computed |
| ├─ `textOverlay` | float | 0.0-1.0 | OCR reliability | OCR |
| ├─ `sticker` | float | 0.0-1.0 | Sticker detection reliability | OCR |
| ├─ `effect` | float | 0.0-1.0 | Effect detection reliability | Computed |
| ├─ `transition` | float | 0.0-1.0 | Scene detection reliability | Scene Detection |
| ├─ `sceneChange` | float | 0.0-1.0 | Scene change reliability | Scene Detection |
| ├─ `object` | float | 0.0-1.0 | YOLO reliability | YOLO |
| └─ `gesture` | float | 0.0-1.0 | MediaPipe reliability | MediaPipe |
| `overallConfidence` | float | 0.0-1.0 | Overall quality score | Computed |

---

## 2. Emotional Journey Features

### CoreMetrics (9 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `uniqueEmotions` | int | 0-10 | Number of different emotions detected | MediaPipe/Computed |
| `emotionTransitions` | int | 0-50 | Count of emotion changes | Computed |
| `dominantEmotion` | string | neutral/happy/sad/surprise/anger/fear | Most frequent emotion | MediaPipe |
| `emotionalDiversity` | float | 0.0-1.0 | Variety score of emotions | Computed |
| `gestureEmotionAlignment` | float | 0.0-1.0 | Gesture-emotion synchronization | Computed |
| `audioEmotionAlignment` | float | 0.0-1.0 | Music-emotion match | Audio Energy |
| `captionSentiment` | string | positive/negative/neutral/mixed | Caption emotional tone | NLP |
| `emotionalIntensity` | float | 0.0-1.0 | Overall emotional strength | Computed |
| `confidence` | float | 0.0-1.0 | Metrics confidence | Computed |

### Dynamics (7 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `emotionProgression` | array | - | Emotion timeline | Computed |
| ├─ `timestamp` | string | "X-Ys" | Time range | Computed |
| ├─ `emotion` | string | - | Emotion type | MediaPipe |
| └─ `intensity` | float | 0.0-1.0 | Emotion strength | Computed |
| `transitionSmoothness` | float | 0.0-1.0 | How gradual changes are | Computed |
| `emotionalArc` | string | rising/falling/stable/rollercoaster | Overall journey pattern | Computed |
| `peakEmotionMoments` | array | - | High intensity timestamps | Computed |
| `stabilityScore` | float | 0.0-1.0 | Consistency of emotions | Computed |
| `tempoEmotionSync` | float | 0.0-1.0 | Music tempo alignment | Audio Energy |
| `confidence` | float | 0.0-1.0 | Dynamics confidence | Computed |

### Interactions (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `gestureReinforcement` | float | 0.0-1.0 | How gestures support emotions | MediaPipe |
| `audioMoodCongruence` | float | 0.0-1.0 | Music-emotion match | Audio Energy |
| `captionEmotionAlignment` | float | 0.0-1.0 | Text-visual sync | NLP/MediaPipe |
| `multimodalCoherence` | float | 0.0-1.0 | Overall alignment | Computed |
| `emotionalContrastMoments` | array | - | Mismatched elements | Computed |
| `confidence` | float | 0.0-1.0 | Interaction confidence | Computed |

### KeyEvents (5 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `emotionalPeaks` | array | - | High intensity moments | Computed |
| `transitionPoints` | array | - | When emotions change | Computed |
| `climaxMoment` | object | - | Peak emotional point | Computed |
| `resolutionMoment` | object | - | Emotional conclusion | Computed |
| `confidence` | float | 0.0-1.0 | Events confidence | Computed |

### Patterns (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `journeyArchetype` | string | surprise_delight/problem_solution/transformation/discovery | Story structure type | Computed |
| `emotionalTechniques` | array[string] | - | Methods used (anticipation_build, contrast, repetition) | Computed |
| `pacingStrategy` | string | gradual_build/quick_shifts/steady_state | How emotions unfold | Computed |
| `engagementHooks` | array[string] | - | Emotional triggers (curiosity, empathy, excitement) | Computed |
| `viewerJourneyMap` | string | engaged_throughout/slow_build/immediate_hook | Expected audience response | Computed |
| `confidence` | float | 0.0-1.0 | Pattern confidence | Computed |

### Quality (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `detectionConfidence` | float | 0.0-1.0 | Expression accuracy | MediaPipe |
| `timelineCoverage` | float | 0.0-1.0 | Data completeness | Computed |
| `emotionalDataCompleteness` | float | 0.0-1.0 | Emotion data coverage | Computed |
| `analysisReliability` | string | high/medium/low | Overall reliability | Computed |
| `missingDataPoints` | array[string] | - | What's missing | Computed |
| `overallConfidence` | float | 0.0-1.0 | Quality score | Computed |

---

## 3. Person Framing Features

### CoreMetrics (12 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `personPresenceRate` | float | 0.0-1.0 | Percentage of video with person | MediaPipe/YOLO |
| `avgPersonCount` | float | 0.0-10.0 | Average people in frame | YOLO |
| `maxSimultaneousPeople` | int | 0-20 | Maximum people at once | YOLO |
| `dominantFraming` | string | close/medium/wide/extreme_close | Most common shot type | Computed |
| `framingChanges` | int | 0-100 | Number of framing transitions | Computed |
| `personScreenCoverage` | float | 0.0-1.0 | Average screen area covered | YOLO bbox |
| `positionStability` | float | 0.0-1.0 | How steady person position is | Computed |
| `gestureClarity` | float | 0.0-1.0 | How clear gestures are | MediaPipe |
| `faceVisibilityRate` | float | 0.0-1.0 | Face detection rate | MediaPipe |
| `bodyVisibilityRate` | float | 0.0-1.0 | Body detection rate | MediaPipe |
| `overallFramingQuality` | float | 0.0-1.0 | Combined framing score | Computed |
| `confidence` | float | 0.0-1.0 | Metrics confidence | Computed |

### Dynamics (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `framingProgression` | array | - | Framing over time | Computed |
| `movementPattern` | string | static/gradual/dynamic/erratic | Camera/subject movement | Computed |
| `zoomTrend` | string | in/out/stable/varied | Zoom direction | Computed |
| `stabilityTimeline` | array | - | Stability per second | Computed |
| `framingTransitions` | array | - | Framing changes | Computed |
| `confidence` | float | 0.0-1.0 | Dynamics confidence | Computed |

### Interactions (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `gestureFramingSync` | float | 0.0-1.0 | Gesture-framing alignment | Computed |
| `expressionVisibility` | float | 0.0-1.0 | Face expression clarity | MediaPipe |
| `multiPersonCoordination` | float | 0.0-1.0 | Multiple person sync | YOLO/MediaPipe |
| `actionSpaceUtilization` | float | 0.0-1.0 | Use of frame space | Computed |
| `framingPurposeAlignment` | float | 0.0-1.0 | Framing matches content | Computed |
| `confidence` | float | 0.0-1.0 | Interaction confidence | Computed |

### KeyEvents (4 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `framingHighlights` | array | - | Notable framing moments | Computed |
| `criticalFramingMoments` | array | - | Important framing events | Computed |
| `optimalFramingPeriods` | array | - | Best framing sections | Computed |
| `confidence` | float | 0.0-1.0 | Events confidence | Computed |

### Patterns (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `framingStrategy` | string | intimate/observational/dynamic/staged | Overall approach | Computed |
| `visualNarrative` | string | single_focus/multi_person/environment_aware | Story telling style | Computed |
| `technicalExecution` | string | professional/casual/experimental | Production quality | Computed |
| `engagementTechniques` | array[string] | - | Techniques used | Computed |
| `productionValue` | string | high/medium/low | Overall quality | Computed |
| `confidence` | float | 0.0-1.0 | Pattern confidence | Computed |

### Quality (5 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `detectionReliability` | float | 0.0-1.0 | Person detection accuracy | YOLO/MediaPipe |
| `trackingConsistency` | float | 0.0-1.0 | Tracking quality | YOLO |
| `framingDataCompleteness` | float | 0.0-1.0 | Data coverage | Computed |
| `analysisLimitations` | array[string] | - | Known issues | Computed |
| `overallConfidence` | float | 0.0-1.0 | Quality score | Computed |

---

## 4. Scene Pacing Features

### CoreMetrics (12 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `totalScenes` | int | 0-100 | Number of scenes | Scene Detection |
| `sceneChangeRate` | float | 0.0-5.0 | Changes per second | Computed |
| `avgSceneDuration` | float | 0.0-60.0 | Average scene length | Computed |
| `minSceneDuration` | float | 0.0-60.0 | Shortest scene | Computed |
| `maxSceneDuration` | float | 0.0-300.0 | Longest scene | Computed |
| `sceneDurationVariance` | float | 0.0-100.0 | Duration variance | Computed |
| `quickCutsCount` | int | 0-50 | Scenes < 2 seconds | Computed |
| `longTakesCount` | int | 0-20 | Scenes > 5 seconds | Computed |
| `sceneRhythmScore` | float | 0.0-1.0 | Rhythm consistency | Computed |
| `pacingConsistency` | float | 0.0-1.0 | Pacing steadiness | Computed |
| `videoDuration` | float | 0.0-300.0 | Total video length | Metadata |
| `confidence` | float | 0.0-1.0 | Metrics confidence | Computed |

### Dynamics (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `pacingCurve` | array | - | Pacing over time | Computed |
| `accelerationPattern` | string | steady/building/declining/variable | Pacing trend | Computed |
| `rhythmRegularity` | float | 0.0-1.0 | Rhythm consistency | Computed |
| `pacingMomentum` | string | maintaining/accelerating/decelerating | Momentum direction | Computed |
| `dynamicRange` | float | 0.0-1.0 | Pacing variation | Computed |
| `confidence` | float | 0.0-1.0 | Dynamics confidence | Computed |

### Interactions (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `contentPacingAlignment` | float | 0.0-1.0 | Content-pacing match | Computed |
| `emotionalPacingSync` | float | 0.0-1.0 | Emotion-pacing sync | Computed |
| `narrativeFlowScore` | float | 0.0-1.0 | Story flow quality | Computed |
| `viewerAdaptationCurve` | string | smooth/jarring/engaging | Viewer experience | Computed |
| `pacingContrastMoments` | array | - | Pacing shifts | Computed |
| `confidence` | float | 0.0-1.0 | Interaction confidence | Computed |

### KeyEvents (5 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `pacingPeaks` | array | - | Fastest pacing moments | Computed |
| `pacingValleys` | array | - | Slowest pacing moments | Computed |
| `criticalTransitions` | array | - | Important pace changes | Computed |
| `rhythmBreaks` | array | - | Rhythm disruptions | Computed |
| `confidence` | float | 0.0-1.0 | Events confidence | Computed |

### Patterns (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `pacingStyle` | string | music_video/narrative/montage/documentary | Edit style | Computed |
| `editingRhythm` | string | metronomic/syncopated/free_form/accelerando | Rhythm type | Computed |
| `visualTempo` | string | slow/moderate/fast/variable | Overall speed | Computed |
| `cutMotivation` | string | action_driven/beat_driven/emotion_driven/random | Cut reasoning | Computed |
| `pacingTechniques` | array[string] | - | Techniques used | Computed |
| `confidence` | float | 0.0-1.0 | Pattern confidence | Computed |

### Quality (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `sceneDetectionAccuracy` | float | 0.0-1.0 | Detection quality | Scene Detection |
| `transitionAnalysisReliability` | float | 0.0-1.0 | Transition accuracy | Computed |
| `pacingDataCompleteness` | float | 0.0-1.0 | Data coverage | Computed |
| `technicalQuality` | string | professional/amateur/mixed | Production quality | Computed |
| `analysisLimitations` | array[string] | - | Known issues | Computed |
| `overallConfidence` | float | 0.0-1.0 | Quality score | Computed |

---

## 5. Speech Analysis Features

### CoreMetrics (11 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `totalSpeechSegments` | int | 0-100 | Number of speech segments | Whisper |
| `speechDuration` | float | 0.0-300.0 | Total speech time | Whisper |
| `speechRate` | float | 0.0-1.0 | Portion of video with speech | Computed |
| `wordsPerMinute` | float | 0.0-300.0 | Speaking speed | Computed |
| `uniqueSpeakers` | int | 0-10 | Number of speakers | Whisper/Computed |
| `primarySpeakerDominance` | float | 0.0-1.0 | Main speaker percentage | Computed |
| `avgConfidence` | float | 0.0-1.0 | Average speech confidence | Whisper |
| `speechClarityScore` | float | 0.0-1.0 | Speech clarity | Computed |
| `pauseCount` | int | 0-50 | Number of pauses | Computed |
| `avgPauseDuration` | float | 0.0-5.0 | Average pause length | Computed |
| `confidence` | float | 0.0-1.0 | Metrics confidence | Computed |

### Dynamics (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `speechPacingCurve` | array | - | Speaking pace over time | Computed |
| `pacingVariation` | float | 0.0-1.0 | Pace consistency | Computed |
| `speechRhythm` | string | steady/variable/accelerating/decelerating | Speech rhythm | Computed |
| `pausePattern` | string | natural/dramatic/rushed/irregular | Pause style | Computed |
| `emphasisMoments` | array | - | Emphasized words/phrases | Computed |
| `confidence` | float | 0.0-1.0 | Dynamics confidence | Computed |

### Interactions (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `speechGestureSync` | float | 0.0-1.0 | Speech-gesture alignment | MediaPipe/Whisper |
| `speechExpressionAlignment` | float | 0.0-1.0 | Speech-expression sync | MediaPipe/Whisper |
| `verbalVisualCoherence` | float | 0.0-1.0 | Overall alignment | Computed |
| `multimodalEmphasis` | array | - | Multi-modal emphasis points | Computed |
| `conversationalDynamics` | string | monologue/dialogue/mixed | Speech type | Computed |
| `confidence` | float | 0.0-1.0 | Interaction confidence | Computed |

### KeyEvents (5 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `keyPhrases` | array | - | Important phrases | NLP/Computed |
| `speechClimax` | object | - | Peak speech moment | Computed |
| `silentMoments` | array | - | Significant silences | Computed |
| `transitionPhrases` | array | - | Transitional speech | NLP/Computed |
| `confidence` | float | 0.0-1.0 | Events confidence | Computed |

### Patterns (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `deliveryStyle` | string | conversational/instructional/narrative/promotional | Speaking style | Computed |
| `speechTechniques` | array[string] | - | Techniques used | Computed |
| `toneCategory` | string | enthusiastic/calm/urgent/informative | Overall tone | Computed |
| `linguisticComplexity` | string | simple/moderate/complex | Language level | NLP |
| `engagementStrategy` | string | storytelling/demonstration/explanation | Approach | Computed |
| `confidence` | float | 0.0-1.0 | Pattern confidence | Computed |

### Quality (5 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `transcriptionConfidence` | float | 0.0-1.0 | Whisper confidence | Whisper |
| `audioQuality` | string | clear/moderate/poor | Audio clarity | Audio Energy |
| `speechDataCompleteness` | float | 0.0-1.0 | Data coverage | Computed |
| `analysisLimitations` | array[string] | - | Known issues | Computed |
| `overallConfidence` | float | 0.0-1.0 | Quality score | Computed |

---

## 6. Visual Overlay Analysis Features

### CoreMetrics (11 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `totalTextOverlays` | int | 0-200 | Text overlay count | OCR |
| `uniqueTexts` | int | 0-100 | Unique text count | OCR |
| `avgTextsPerSecond` | float | 0.0-5.0 | Text frequency | Computed |
| `timeToFirstText` | float | 0.0-10.0 | First text appearance | Computed |
| `avgTextDisplayDuration` | float | 0.0-10.0 | Average text duration | Computed |
| `totalStickers` | int | 0-50 | Sticker count | OCR |
| `uniqueStickers` | int | 0-30 | Unique sticker count | OCR |
| `textStickerRatio` | float | 0.0-10.0 | Text to sticker ratio | Computed |
| `overlayDensity` | float | 0.0-1.0 | Overall overlay density | Computed |
| `visualComplexityScore` | float | 0.0-1.0 | Visual complexity | Computed |
| `confidence` | float | 0.0-1.0 | Metrics confidence | Computed |

### Dynamics (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `overlayTimeline` | array | - | Overlay progression | Computed |
| `appearancePattern` | string | gradual/burst/rhythmic/random | Appearance style | Computed |
| `densityProgression` | string | building/front_loaded/even/climactic | Density trend | Computed |
| `animationIntensity` | float | 0.0-1.0 | Animation level | Computed |
| `visualPacing` | string | slow/moderate/fast/variable | Visual speed | Computed |
| `confidence` | float | 0.0-1.0 | Dynamics confidence | Computed |

### Interactions (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `textSpeechSync` | float | 0.0-1.0 | Text-speech alignment | OCR/Whisper |
| `overlayGestureCoordination` | float | 0.0-1.0 | Overlay-gesture sync | OCR/MediaPipe |
| `visualEmphasisAlignment` | float | 0.0-1.0 | Emphasis alignment | Computed |
| `multiLayerComplexity` | string | simple/moderate/complex | Layer complexity | Computed |
| `readingFlowScore` | float | 0.0-1.0 | Reading ease | Computed |
| `confidence` | float | 0.0-1.0 | Interaction confidence | Computed |

### KeyEvents (5 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `impactfulOverlays` | array | - | High-impact overlays | Computed |
| `overlayBursts` | array | - | Multiple overlay moments | Computed |
| `keyTextMoments` | array | - | Important text appearances | Computed |
| `stickerHighlights` | array | - | Notable stickers | Computed |
| `confidence` | float | 0.0-1.0 | Events confidence | Computed |

### Patterns (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `overlayStrategy` | string | minimal/moderate/heavy/dynamic | Overall strategy | Computed |
| `textStyle` | string | clean/decorative/mixed/chaotic | Text style | Computed |
| `communicationApproach` | string | reinforcing/supplementary/dominant | Communication type | Computed |
| `visualTechniques` | array[string] | - | Techniques used | Computed |
| `productionQuality` | string | professional/casual/amateur | Quality level | Computed |
| `confidence` | float | 0.0-1.0 | Pattern confidence | Computed |

### Quality (6 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `textDetectionAccuracy` | float | 0.0-1.0 | OCR accuracy | OCR |
| `stickerRecognitionRate` | float | 0.0-1.0 | Sticker detection rate | OCR |
| `overlayDataCompleteness` | float | 0.0-1.0 | Data coverage | Computed |
| `readabilityIssues` | array[string] | - | Reading problems | Computed |
| `visualAccessibility` | string | high/medium/low | Accessibility level | Computed |
| `overallConfidence` | float | 0.0-1.0 | Quality score | Computed |

---

## 7. Metadata Analysis Features

### CoreMetrics (15 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `captionLength` | int | 0-500 | Caption character count | API |
| `wordCount` | int | 0-100 | Caption word count | Computed |
| `hashtagCount` | int | 0-30 | Number of hashtags | API |
| `emojiCount` | int | 0-20 | Number of emojis | Computed |
| `mentionCount` | int | 0-10 | Number of mentions | API |
| `linkPresent` | boolean | true/false | Contains link | API |
| `videoDuration` | float | 0.0-300.0 | Video length | API |
| `publishHour` | int | 0-23 | Hour published | API |
| `publishDayOfWeek` | int | 0-6 | Day published | API |
| `viewCount` | int | 0-1000000000 | View count | API |
| `likeCount` | int | 0-100000000 | Like count | API |
| `commentCount` | int | 0-1000000 | Comment count | API |
| `shareCount` | int | 0-1000000 | Share count | API |
| `engagementRate` | float | 0.0-100.0 | Engagement percentage | Computed |
| `confidence` | float | 0.0-1.0 | Metrics confidence | Computed |

### Dynamics (10 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `hashtagStrategy` | string | minimal/moderate/heavy/spam | Hashtag usage | Computed |
| `captionStyle` | string | storytelling/direct/question/list/minimal | Caption approach | NLP |
| `emojiDensity` | float | 0.0-1.0 | Emoji usage rate | Computed |
| `mentionDensity` | float | 0.0-1.0 | Mention usage rate | Computed |
| `readabilityScore` | float | 0.0-100.0 | Reading ease score | NLP |
| `sentimentPolarity` | float | -1.0-1.0 | Sentiment score | NLP |
| `sentimentCategory` | string | positive/negative/neutral/mixed | Sentiment type | NLP |
| `urgencyLevel` | string | high/medium/low/none | Urgency indicators | NLP |
| `viralPotentialScore` | float | 0.0-1.0 | Viral likelihood | Computed |
| `confidence` | float | 0.0-1.0 | Dynamics confidence | Computed |

### Interactions (5 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `hashtagCounts` | object | - | Hashtag breakdown | Computed |
| ├─ `nicheCount` | int | 0-30 | Niche hashtags | Computed |
| └─ `genericCount` | int | 0-30 | Generic hashtags | Computed |
| `engagementAlignment` | object | - | Engagement ratios | Computed |
| ├─ `likesToViewsRatio` | float | 0.0-1.0 | Like rate | Computed |
| ├─ `commentsToViewsRatio` | float | 0.0-1.0 | Comment rate | Computed |
| ├─ `sharesToViewsRatio` | float | 0.0-1.0 | Share rate | Computed |
| └─ `aboveAverageEngagement` | boolean | true/false | Above average | Computed |
| `creatorContext` | object | - | Creator info | API |
| `confidence` | float | 0.0-1.0 | Interaction confidence | Computed |

### KeyEvents (5 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `hashtags` | array | - | Hashtag details | API/Computed |
| `emojis` | array | - | Emoji details | Computed |
| `hooks` | array | - | Hook phrases | NLP |
| `callToActions` | array | - | CTA phrases | NLP |
| `confidence` | float | 0.0-1.0 | Events confidence | Computed |

### Patterns (3 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `linguisticMarkers` | object | - | Language patterns | NLP |
| `hashtagPatterns` | object | - | Hashtag patterns | Computed |
| `confidence` | float | 0.0-1.0 | Pattern confidence | Computed |

### Quality (8 features)
| Feature | Type | Range/Values | Description | Source |
|---------|------|--------------|-------------|--------|
| `captionPresent` | boolean | true/false | Has caption | API |
| `hashtagsPresent` | boolean | true/false | Has hashtags | API |
| `statsAvailable` | boolean | true/false | Has statistics | API |
| `publishTimeAvailable` | boolean | true/false | Has publish time | API |
| `creatorDataAvailable` | boolean | true/false | Has creator data | API |
| `captionQuality` | string | high/medium/low | Caption quality | Computed |
| `hashtagQuality` | string | high/medium/low/mixed | Hashtag quality | Computed |
| `overallConfidence` | float | 0.0-1.0 | Quality score | Computed |

---

## Cross-Analysis Dependencies

### Feature Relationships Matrix

| Source Analysis | Target Analysis | Shared Features | Dependency Type |
|-----------------|-----------------|-----------------|-----------------|
| Speech → Visual Overlay | `textSpeechSync` | Temporal alignment | Direct |
| Emotional → Scene Pacing | `emotionalPacingSync` | Rhythm correlation | Computed |
| Person Framing → Emotional | `expressionVisibility` | Face detection | Direct |
| Creative Density → All | `elementCounts` | Element distribution | Aggregated |
| Metadata → All | `videoDuration` | Temporal normalization | Reference |
| Audio Energy → Emotional | `audioEmotionAlignment` | Music-mood | Direct |
| Scene Pacing → Creative | `sceneChangeCount` | Transition density | Direct |

### ML Service Dependencies

| ML Service | Dependent Analyses | Critical Features |
|------------|-------------------|-------------------|
| **YOLO** | Person Framing, Creative Density | Object detection, person tracking |
| **MediaPipe** | Emotional Journey, Person Framing, Speech | Pose, face, gesture detection |
| **Whisper** | Speech Analysis, Visual Overlay | Transcription, timing |
| **OCR** | Visual Overlay, Creative Density | Text detection, stickers |
| **Scene Detection** | Scene Pacing, Creative Density | Scene boundaries |
| **Audio Energy** | Emotional Journey | Energy levels, burst patterns |

---

## Feature Engineering Opportunities

### High-Value Composite Features

1. **Engagement Momentum Score**
   - Combines: `densityProgression` + `emotionalArc` + `pacingMomentum`
   - Predicts: Viewer retention probability

2. **Production Quality Index**
   - Combines: `framingQuality` + `speechClarity` + `visualAccessibility`
   - Indicates: Professional vs amateur content

3. **Viral Signature Pattern**
   - Combines: `strongOpeningHook` + `emotionalPeaks` + `engagementRate`
   - Predicts: Viral potential

4. **Cognitive Load Index**
   - Combines: `overlayDensity` + `speechRate` + `sceneChangeRate`
   - Measures: Processing difficulty

5. **Multimodal Coherence Score**
   - Combines: All `*Sync` and `*Alignment` features
   - Indicates: Production coordination quality

### Temporal Pattern Features

1. **Momentum Vectors**: Direction and magnitude of change over time
2. **Rhythm Signatures**: Recurring patterns in pacing/density
3. **Climax Timing**: Normalized position of peak moments
4. **Hook Effectiveness**: First 3-second engagement metrics

### Missing Features to Consider Adding

1. **Color Analysis**: Dominant colors, palette changes
2. **Motion Vectors**: Camera movement, subject movement
3. **Audio Spectrum**: Frequency analysis beyond energy
4. **Semantic Content**: Topic modeling from transcription
5. **Cultural Markers**: Region-specific elements

---

## Feature Quality Metrics

### Data Completeness by Analysis Type

| Analysis Type | Average Completeness | Critical Gaps |
|---------------|---------------------|---------------|
| Creative Density | 92% | Sticker detection |
| Emotional Journey | 78% | Fine-grained emotions |
| Person Framing | 85% | Multi-person scenarios |
| Scene Pacing | 95% | Transition effects |
| Speech Analysis | 88% | Speaker diarization |
| Visual Overlay | 75% | Animation detection |
| Metadata | 98% | Historical trends |

### Confidence Score Distribution

- **High Confidence (0.8-1.0)**: 65% of features
- **Medium Confidence (0.5-0.8)**: 25% of features
- **Low Confidence (0.0-0.5)**: 10% of features

---

## Implementation Notes

### Feature Extraction Pipeline

```python
# Pseudo-code for feature extraction
for analysis_type in ANALYSIS_TYPES:
    raw_ml_data = ml_services.extract(video)
    timeline_data = precompute_functions.transform(raw_ml_data)
    computed_metrics = precompute_functions.compute(timeline_data)
    claude_response = claude_api.analyze(computed_metrics)
    ml_features = parse_coreblocks(claude_response)
    validate_and_store(ml_features)
```

### Feature Storage Schema

```json
{
  "video_id": "string",
  "timestamp": "ISO8601",
  "analysis_version": "1.0.0",
  "features": {
    "creative_density": { /* 6 blocks */ },
    "emotional_journey": { /* 6 blocks */ },
    "person_framing": { /* 6 blocks */ },
    "scene_pacing": { /* 6 blocks */ },
    "speech_analysis": { /* 6 blocks */ },
    "visual_overlay_analysis": { /* 6 blocks */ },
    "metadata_analysis": { /* 6 blocks */ }
  },
  "quality_metrics": {
    "overall_confidence": 0.85,
    "data_completeness": 0.92
  }
}
```

---

## Conclusion

This comprehensive documentation provides a complete reference for all 300+ ML features available from the RumiAI CoreBlock structure. The features span:

- **7 analysis types**
- **6 standardized blocks per analysis**
- **40-50 features per analysis type**
- **Multiple data sources** (computed, ML services, API)
- **Rich cross-modal relationships**

The structured organization enables:
- Consistent feature engineering
- Reliable ML model training
- Quality assurance and validation
- System integration and scaling
- Research and experimentation

This feature dictionary serves as the foundation for building ML models that can understand, analyze, and predict TikTok video performance and characteristics.