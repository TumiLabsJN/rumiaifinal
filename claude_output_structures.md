# Claude ML Pipeline Output Structures

This document details the exact JSON output structure for each of the 7 ML analysis flows that Claude produces when processing TikTok videos through the RumiAI pipeline.

**Note (2025-08-05)**: Output structures remain unchanged. The unified ML implementation only fixed the input data flow to ensure Claude receives real ML data via the `ml_data` field instead of empty arrays.

## Overview

Each ML flow outputs a standardized 6-block structure:
1. **CoreMetrics** - Basic measurements and counts
2. **Dynamics** - Temporal changes and progressions  
3. **Interactions** - Element relationships and synchronization
4. **KeyEvents** - Specific moments and occurrences
5. **Patterns** - Recurring behaviors and strategies
6. **Quality** - Data confidence and completeness

---

## 1. Creative Density

### Input
Precomputed density metrics including average density, peak moments, element distribution, and timeline coverage.

### Output Structure

```json
{
  "densityCoreMetrics": {
    "avgDensity": float,
    "maxDensity": float,
    "minDensity": float,
    "stdDeviation": float,
    "totalElements": int,
    "elementsPerSecond": float,
    "elementCounts": {
      "text": int,
      "sticker": int,
      "effect": int,
      "transition": int,
      "object": int,
      "gesture": int,
      "expression": int
    },
    "sceneChangeCount": int,
    "timelineCoverage": float,
    "confidence": float
  },
  
  "densityDynamics": {
    "densityCurve": [
      {"second": int, "density": float, "primaryElement": string}
    ],
    "volatility": float,
    "accelerationPattern": "front_loaded" | "even" | "back_loaded" | "oscillating",
    "densityProgression": "increasing" | "decreasing" | "stable" | "erratic",
    "emptySeconds": [int],
    "confidence": float
  },
  
  "densityInteractions": {
    "multiModalPeaks": [
      {
        "timestamp": string,
        "elements": [string],
        "syncType": "reinforcing" | "complementary" | "redundant"
      }
    ],
    "elementCooccurrence": {
      "text_effect": int,
      "text_transition": int,
      "effect_sceneChange": int,
      "gesture_text": int,
      "expression_gesture": int
    },
    "dominantCombination": string,
    "coordinationScore": float,
    "confidence": float
  },
  
  "densityKeyEvents": {
    "peakMoments": [
      {
        "timestamp": string,
        "totalElements": int,
        "surpriseScore": float,
        "elementBreakdown": {
          "text": int,
          "effect": int,
          "transition": int,
          "gesture": int
        }
      }
    ],
    "deadZones": [
      {"start": int, "end": int, "duration": int}
    ],
    "densityShifts": [
      {
        "timestamp": int,
        "from": "high" | "medium" | "low",
        "to": "high" | "medium" | "low",
        "magnitude": float
      }
    ],
    "confidence": float
  },
  
  "densityPatterns": {
    "structuralFlags": {
      "strongOpeningHook": boolean,
      "crescendoPattern": boolean,
      "frontLoaded": boolean,
      "consistentPacing": boolean,
      "finalCallToAction": boolean,
      "rhythmicPattern": boolean
    },
    "densityClassification": "sparse" | "moderate" | "dense" | "overwhelming",
    "pacingStyle": "steady" | "building" | "burst_fade" | "oscillating",
    "cognitiveLoadCategory": "minimal" | "optimal" | "challenging" | "overwhelming",
    "mlTags": [string],
    "confidence": float
  },
  
  "densityQuality": {
    "dataCompleteness": float,
    "detectionReliability": {
      "textOverlay": float,
      "sticker": float,
      "effect": float,
      "transition": float,
      "sceneChange": float,
      "object": float,
      "gesture": float
    },
    "overallConfidence": float
  }
}
```

---

## 2. Emotional Journey

### Input
Expression timeline, gesture timeline, audio mood indicators, and caption sentiment.

### Output Structure

```json
{
  "emotionalCoreMetrics": {
    "uniqueEmotions": int,
    "emotionTransitions": int,
    "dominantEmotion": string,
    "emotionalDiversity": float,
    "gestureEmotionAlignment": float,
    "audioEmotionAlignment": float,
    "captionSentiment": "positive" | "negative" | "neutral",
    "emotionalIntensity": float,
    "confidence": float
  },
  
  "emotionalDynamics": {
    "emotionProgression": [
      {
        "timestamp": string,
        "emotion": string,
        "intensity": float
      }
    ],
    "transitionSmoothness": float,
    "emotionalArc": "rising" | "falling" | "stable" | "rollercoaster",
    "peakEmotionMoments": [
      {
        "timestamp": string,
        "emotion": string,
        "intensity": float
      }
    ],
    "stabilityScore": float,
    "tempoEmotionSync": float,
    "confidence": float
  },
  
  "emotionalInteractions": {
    "gestureReinforcement": float,
    "audioMoodCongruence": float,
    "captionEmotionAlignment": float,
    "multimodalCoherence": float,
    "emotionalContrastMoments": [
      {
        "timestamp": string,
        "conflict": string
      }
    ],
    "confidence": float
  },
  
  "emotionalKeyEvents": {
    "emotionalPeaks": [
      {
        "timestamp": string,
        "emotion": string,
        "trigger": string
      }
    ],
    "transitionPoints": [
      {
        "timestamp": string,
        "from": string,
        "to": string,
        "trigger": string
      }
    ],
    "climaxMoment": {
      "timestamp": string,
      "emotion": string,
      "intensity": float
    },
    "resolutionMoment": {
      "timestamp": string,
      "emotion": string,
      "closure": boolean
    },
    "confidence": float
  },
  
  "emotionalPatterns": {
    "journeyArchetype": "surprise_delight" | "problem_solution" | "transformation" | "discovery",
    "emotionalTechniques": [string],
    "pacingStrategy": "gradual_build" | "quick_shifts" | "steady_state",
    "engagementHooks": [string],
    "viewerJourneyMap": "engaged_throughout" | "slow_build" | "immediate_hook",
    "confidence": float
  },
  
  "emotionalQuality": {
    "detectionConfidence": float,
    "timelineCoverage": float,
    "emotionalDataCompleteness": float,
    "analysisReliability": "high" | "medium" | "low",
    "missingDataPoints": [string],
    "overallConfidence": float
  }
}
```

---

## 3. Person Framing

### Input
Object detection timeline, camera distance, gestures, expressions, and framing metrics.

### Output Structure

```json
{
  "personFramingCoreMetrics": {
    "personPresenceRate": float,
    "avgPersonCount": float,
    "maxSimultaneousPeople": int,
    "dominantFraming": "close" | "medium" | "wide",
    "framingChanges": int,
    "personScreenCoverage": float,
    "positionStability": float,
    "gestureClarity": float,
    "faceVisibilityRate": float,
    "bodyVisibilityRate": float,
    "overallFramingQuality": float,
    "confidence": float
  },
  
  "personFramingDynamics": {
    "framingProgression": [
      {
        "timestamp": string,
        "distance": string,
        "coverage": float
      }
    ],
    "movementPattern": "static" | "gradual" | "dynamic",
    "zoomTrend": "in" | "out" | "stable" | "varied",
    "stabilityTimeline": [
      {"second": int, "stability": float}
    ],
    "framingTransitions": [
      {
        "timestamp": int,
        "from": string,
        "to": string
      }
    ],
    "confidence": float
  },
  
  "personFramingInteractions": {
    "gestureFramingSync": float,
    "expressionVisibility": float,
    "multiPersonCoordination": float,
    "actionSpaceUtilization": float,
    "framingPurposeAlignment": float,
    "confidence": float
  },
  
  "personFramingKeyEvents": {
    "framingHighlights": [
      {
        "timestamp": string,
        "type": string,
        "impact": "high" | "medium" | "low"
      }
    ],
    "criticalFramingMoments": [
      {
        "timestamp": int,
        "event": string,
        "framing": string
      }
    ],
    "optimalFramingPeriods": [
      {
        "start": int,
        "end": int,
        "reason": string
      }
    ],
    "confidence": float
  },
  
  "personFramingPatterns": {
    "framingStrategy": "intimate" | "observational" | "dynamic" | "staged",
    "visualNarrative": "single_focus" | "multi_person" | "environment_aware",
    "technicalExecution": "professional" | "casual" | "experimental",
    "engagementTechniques": [string],
    "productionValue": "high" | "medium" | "low",
    "confidence": float
  },
  
  "personFramingQuality": {
    "detectionReliability": float,
    "trackingConsistency": float,
    "framingDataCompleteness": float,
    "analysisLimitations": [string],
    "overallConfidence": float
  }
}
```

---

## 4. Scene Pacing

### Input
Scene change timeline with durations, transitions, and rhythm metrics.

### Output Structure

```json
{
  "scenePacingCoreMetrics": {
    "totalScenes": int,
    "sceneChangeRate": float,
    "avgSceneDuration": float,
    "minSceneDuration": float,
    "maxSceneDuration": float,
    "sceneDurationVariance": float,
    "quickCutsCount": int,
    "longTakesCount": int,
    "sceneRhythmScore": float,
    "pacingConsistency": float,
    "videoDuration": float,
    "confidence": float
  },
  
  "scenePacingDynamics": {
    "pacingCurve": [
      {
        "second": int,
        "cutsPerSecond": float,
        "intensity": "low" | "medium" | "high"
      }
    ],
    "accelerationPattern": "steady" | "building" | "declining" | "variable",
    "rhythmRegularity": float,
    "pacingMomentum": "maintaining" | "accelerating" | "decelerating",
    "dynamicRange": float,
    "confidence": float
  },
  
  "scenePacingInteractions": {
    "contentPacingAlignment": float,
    "emotionalPacingSync": float,
    "narrativeFlowScore": float,
    "viewerAdaptationCurve": "smooth" | "jarring" | "engaging",
    "pacingContrastMoments": [
      {
        "timestamp": int,
        "shift": string,
        "impact": float
      }
    ],
    "confidence": float
  },
  
  "scenePacingKeyEvents": {
    "pacingPeaks": [
      {
        "timestamp": string,
        "cutsPerSecond": float,
        "intensity": string
      }
    ],
    "pacingValleys": [
      {
        "timestamp": string,
        "sceneDuration": float,
        "type": string
      }
    ],
    "criticalTransitions": [
      {
        "timestamp": int,
        "fromPace": string,
        "toPace": string,
        "effect": string
      }
    ],
    "rhythmBreaks": [
      {
        "timestamp": int,
        "expectedDuration": float,
        "actualDuration": float
      }
    ],
    "confidence": float
  },
  
  "scenePacingPatterns": {
    "pacingStyle": "music_video" | "narrative" | "montage" | "documentary",
    "editingRhythm": "metronomic" | "syncopated" | "free_form" | "accelerando",
    "visualTempo": "slow" | "moderate" | "fast" | "variable",
    "cutMotivation": "action_driven" | "beat_driven" | "emotion_driven" | "random",
    "pacingTechniques": [string],
    "confidence": float
  },
  
  "scenePacingQuality": {
    "sceneDetectionAccuracy": float,
    "transitionAnalysisReliability": float,
    "pacingDataCompleteness": float,
    "technicalQuality": "professional" | "amateur" | "mixed",
    "analysisLimitations": [string],
    "overallConfidence": float
  }
}
```

---

## 5. Speech Analysis

### Input
Speech timeline, expression timeline, gesture timeline, and speech metrics.

### Output Structure

```json
{
  "speechCoreMetrics": {
    "totalSpeechSegments": int,
    "speechDuration": float,
    "speechRate": float,
    "wordsPerMinute": float,
    "uniqueSpeakers": int,
    "primarySpeakerDominance": float,
    "avgConfidence": float,
    "speechClarityScore": float,
    "pauseCount": int,
    "avgPauseDuration": float,
    "confidence": float
  },
  
  "speechDynamics": {
    "speechPacingCurve": [
      {
        "timestamp": string,
        "wordsPerSecond": float,
        "intensity": "low" | "moderate" | "high"
      }
    ],
    "pacingVariation": float,
    "speechRhythm": "steady" | "variable" | "accelerating" | "decelerating",
    "pausePattern": "natural" | "dramatic" | "rushed" | "irregular",
    "emphasisMoments": [
      {
        "timestamp": float,
        "word": string,
        "emphasisType": string
      }
    ],
    "confidence": float
  },
  
  "speechInteractions": {
    "speechGestureSync": float,
    "speechExpressionAlignment": float,
    "verbalVisualCoherence": float,
    "multimodalEmphasis": [
      {
        "timestamp": float,
        "speech": string,
        "gesture": string,
        "alignment": float
      }
    ],
    "conversationalDynamics": "monologue" | "dialogue" | "mixed",
    "confidence": float
  },
  
  "speechKeyEvents": {
    "keyPhrases": [
      {
        "timestamp": float,
        "phrase": string,
        "significance": string
      }
    ],
    "speechClimax": {
      "timestamp": float,
      "text": string,
      "intensity": float
    },
    "silentMoments": [
      {
        "start": float,
        "end": float,
        "duration": float,
        "purpose": string
      }
    ],
    "transitionPhrases": [
      {
        "timestamp": float,
        "phrase": string,
        "function": string
      }
    ],
    "confidence": float
  },
  
  "speechPatterns": {
    "deliveryStyle": "conversational" | "instructional" | "narrative" | "promotional",
    "speechTechniques": [string],
    "toneCategory": "enthusiastic" | "calm" | "urgent" | "informative",
    "linguisticComplexity": "simple" | "moderate" | "complex",
    "engagementStrategy": "storytelling" | "demonstration" | "explanation",
    "confidence": float
  },
  
  "speechQuality": {
    "transcriptionConfidence": float,
    "audioQuality": "clear" | "moderate" | "poor",
    "speechDataCompleteness": float,
    "analysisLimitations": [string],
    "overallConfidence": float
  }
}
```

---

## 6. Visual Overlay Analysis

### Input
Text overlay timeline, sticker timeline, visual complexity metrics, and overlay patterns.

### Output Structure

```json
{
  "overlaysCoreMetrics": {
    "totalTextOverlays": int,
    "uniqueTexts": int,
    "avgTextsPerSecond": float,
    "timeToFirstText": float,
    "avgTextDisplayDuration": float,
    "totalStickers": int,
    "uniqueStickers": int,
    "textStickerRatio": float,
    "overlayDensity": float,
    "visualComplexityScore": float,
    "confidence": float
  },
  
  "overlaysDynamics": {
    "overlayTimeline": [
      {
        "timestamp": string,
        "overlayCount": int,
        "type": string,
        "density": "low" | "medium" | "high"
      }
    ],
    "appearancePattern": "gradual" | "burst" | "rhythmic" | "random",
    "densityProgression": "building" | "front_loaded" | "even" | "climactic",
    "animationIntensity": float,
    "visualPacing": "slow" | "moderate" | "fast" | "variable",
    "confidence": float
  },
  
  "overlaysInteractions": {
    "textSpeechSync": float,
    "overlayGestureCoordination": float,
    "visualEmphasisAlignment": float,
    "multiLayerComplexity": "simple" | "moderate" | "complex",
    "readingFlowScore": float,
    "confidence": float
  },
  
  "overlaysKeyEvents": {
    "impactfulOverlays": [
      {
        "timestamp": string,
        "text": string,
        "impact": "high" | "medium" | "low",
        "reason": string
      }
    ],
    "overlayBursts": [
      {
        "timestamp": string,
        "count": int,
        "purpose": string
      }
    ],
    "keyTextMoments": [
      {
        "timestamp": float,
        "text": string,
        "function": string
      }
    ],
    "stickerHighlights": [
      {
        "timestamp": float,
        "sticker": string,
        "purpose": string
      }
    ],
    "confidence": float
  },
  
  "overlaysPatterns": {
    "overlayStrategy": "minimal" | "moderate" | "heavy" | "dynamic",
    "textStyle": "clean" | "decorative" | "mixed" | "chaotic",
    "communicationApproach": "reinforcing" | "supplementary" | "dominant",
    "visualTechniques": [string],
    "productionQuality": "professional" | "casual" | "amateur",
    "confidence": float
  },
  
  "overlaysQuality": {
    "textDetectionAccuracy": float,
    "stickerRecognitionRate": float,
    "overlayDataCompleteness": float,
    "readabilityIssues": [string],
    "visualAccessibility": "high" | "medium" | "low",
    "overallConfidence": float
  }
}
```

---

## 7. Metadata Analysis

### Input
Caption text, hashtags, engagement stats, publishing data, and creator information.

### Output Structure

```json
{
  "metadataCoreMetrics": {
    "captionLength": int,
    "wordCount": int,
    "hashtagCount": int,
    "emojiCount": int,
    "mentionCount": int,
    "linkPresent": boolean,
    "videoDuration": float,
    "publishHour": int,
    "publishDayOfWeek": int,
    "viewCount": int,
    "likeCount": int,
    "commentCount": int,
    "shareCount": int,
    "engagementRate": float,
    "confidence": float
  },
  
  "metadataDynamics": {
    "hashtagStrategy": "minimal" | "moderate" | "heavy" | "spam",
    "captionStyle": "storytelling" | "direct" | "question" | "list" | "minimal",
    "emojiDensity": float,
    "mentionDensity": float,
    "readabilityScore": float,
    "sentimentPolarity": float,
    "sentimentCategory": "positive" | "negative" | "neutral" | "mixed",
    "urgencyLevel": "high" | "medium" | "low" | "none",
    "viralPotentialScore": float,
    "confidence": float
  },
  
  "metadataInteractions": {
    "hashtagCounts": {
      "nicheCount": int,
      "genericCount": int
    },
    "engagementAlignment": {
      "likesToViewsRatio": float,
      "commentsToViewsRatio": float,
      "sharesToViewsRatio": float,
      "aboveAverageEngagement": boolean
    },
    "creatorContext": {
      "username": string,
      "verified": boolean
    },
    "confidence": float
  },
  
  "metadataKeyEvents": {
    "hashtags": [
      {
        "tag": string,
        "position": int,
        "type": "generic" | "niche",
        "estimatedReach": "high" | "medium" | "low"
      }
    ],
    "emojis": [
      {
        "emoji": string,
        "count": int,
        "sentiment": "positive" | "negative" | "neutral",
        "emphasis": boolean
      }
    ],
    "hooks": [
      {
        "text": string,
        "position": "start" | "middle" | "end",
        "type": "curiosity" | "promise" | "question" | "challenge",
        "strength": float
      }
    ],
    "callToActions": [
      {
        "text": string,
        "type": "follow" | "like" | "comment" | "share" | "visit",
        "explicitness": "direct" | "implied",
        "urgency": "high" | "medium" | "low"
      }
    ],
    "confidence": float
  },
  
  "metadataPatterns": {
    "linguisticMarkers": {
      "questionCount": int,
      "exclamationCount": int,
      "capsLockWords": int,
      "personalPronounCount": int
    },
    "hashtagPatterns": {
      "leadWithGeneric": boolean,
      "allCaps": boolean
    },
    "confidence": float
  },
  
  "metadataQuality": {
    "captionPresent": boolean,
    "hashtagsPresent": boolean,
    "statsAvailable": boolean,
    "publishTimeAvailable": boolean,
    "creatorDataAvailable": boolean,
    "captionQuality": "high" | "medium" | "low" | "empty",
    "hashtagQuality": "mixed" | "spammy" | "none",
    "overallConfidence": float
  }
}
```

---

## Notes

1. **Unified Analysis** is not a Claude output - it's the input data structure that feeds all these analysis flows.

2. **Field Types**:
   - `float`: Decimal numbers (0.0 - 1.0 for scores/ratios)
   - `int`: Whole numbers
   - `string`: Text values
   - `boolean`: true/false
   - `[type]`: Array of specified type
   - `"option1" | "option2"`: Enumerated string values

3. **Confidence Scores**: Each block includes a confidence score (0.0-1.0) indicating data quality and reliability.

4. **Timestamp Format**: Timestamps are typically in "X-Ys" format (e.g., "5-6s") or float seconds.

5. **Optional Fields**: Some fields may be null or empty arrays when data is unavailable.

This structure ensures consistent ML feature extraction across all video analyses, enabling reliable model training and performance tracking.