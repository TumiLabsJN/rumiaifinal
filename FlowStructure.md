# TikTok ML Pipeline Overview

## System Status (Updated 2025-08-07 - Post Deep Investigation)
**🟡 PARTIALLY WORKING**: ML extraction successful but critical bugs found
- ✅ ML data extraction improved (2.2% → ~100%) - VERIFIED
- ✅ Claude uses data effectively (0.86-0.90 confidence justified) - VERIFIED
- ✅ Unified frame extraction working (4x → 1x reduction) - VERIFIED
- ❌ 3/7 prompts failing due to data format mismatches - NEEDS FIX
- ❌ Metadata pipeline broken (wrong params, field names) - NEEDS FIX
- ❌ Scene detection threshold too high (27.0 vs 20.0) - NEEDS FIX
- ❌ Sticker detection hardcoded to empty - NEEDS INTEGRATION

## Data Flow
Raw Video → Unified Frame Extraction → ML Processing (Working) → MLAnalysisResult → ml_data field → Unified Analysis → Precompute Functions (3 FAILING) → Claude Analysis → ML Features

## Critical Format Issues Found
- **objectTimeline**: Creates list format but functions expect dict with 'objects' key
- **Metadata**: Wrong field names (playCount vs views, diggCount vs likes)
- **Scene Detection**: Threshold 27.0 missing changes (needs 20.0)

## Unified ML Architecture (IMPLEMENTED 2025-08-05)

### Frame Extraction Pipeline
```
Video Input
    ↓
UnifiedFrameManager (unified_frame_manager.py)
    ├── Extract frames ONCE (145 frames @ 2 FPS for 72s video)
    ├── LRU Cache (max 5 videos, 2GB limit)
    └── Share frames with all ML services
        ├── YOLO: 100 frames (uniform sampling)
        ├── MediaPipe: 180 frames (all frames)
        ├── OCR: 60 frames (adaptive sampling)
        └── Scene: All frames (boundary detection)
```

### ML Services Flow
```
UnifiedMLServices (ml_services_unified.py)
    ├── Lazy Model Loading (load only when needed)
    ├── Native Async (asyncio.to_thread)
    ├── Timeout Protection (10 min max)
    └── Individual Service Methods
        ├── run_yolo_detection() → ONLY YOLO
        ├── run_mediapipe_analysis() → ONLY MediaPipe
        ├── run_ocr_analysis() → ONLY OCR (was run_ocr_detection - FIXED)
        └── run_whisper_transcription() → ONLY Whisper
```

## 7 Analysis Flows
1. **Creative Density** - Measures content density and pacing
2. **Emotional Journey** - Tracks emotional progression
3. **Person Framing** - Analyzes human presence and positioning
4. **Scene Pacing** - Evaluates scene changes and rhythm
5. **Speech Analysis** - Examines speech patterns and sync
6. **Visual Overlay** - Analyzes text/sticker overlays
7. **Metadata Analysis** - Processes caption and engagement data

## Output Structure (All Flows)
Each flow outputs 6 standardized blocks:
1. **CoreMetrics** - Basic measurements and counts
2. **Dynamics** - Temporal changes and progressions
3. **Interactions** - Element relationships and synchronization
4. **KeyEvents** - Specific moments and occurrences
5. **Patterns** - Recurring behaviors and strategies
6. **Quality** - Data confidence and completeness

**IMPORTANT**: All prompts MUST use these exact block names without prefixes. Person framing previously used `personFramingCoreMetrics` etc. - this has been fixed.

## Data Dependencies

| Flow | Required Timelines | Status |
|------|-------------------|--------|
| Creative Density | textOverlay, sticker, gesture, object, expression, sceneChange, effect, transition | ✅ Working |
| Emotional Journey | expression, gesture, audioRatio + caption | ✅ Working |
| Person Framing | object, cameraDistance, gesture, expression | ❌ FAILING - list/dict mismatch |
| Scene Pacing | sceneChange | ❌ FAILING - list/dict mismatch |
| Speech Analysis | speech, expression, gesture | ✅ Working |
| Visual Overlay | textOverlay, sticker, gesture, speech, object | ❌ FAILING - tuple unpacking error |
| Metadata | static_metadata, metadata_summary | ✅ Working but all zeros (bugs)

## Confidence Scoring
- Each block includes a confidence score (0.0-1.0)
- Confidence reflects data quality and detection reliability
- Lower confidence indicates missing or uncertain data
- Used for ML sample weighting during training

## Implementation Details (2025-08-05)

### ML Data Field Solution
**Implementation**: UnifiedAnalysis.to_dict() now includes ml_data field
```python
# In analysis.py lines 126-142:
result['ml_data'] = {}
for service in required_models:
    if service in self.ml_results and self.ml_results[service].success:
        result['ml_data'][service] = self.ml_results[service].data
    else:
        result['ml_data'][service] = {}
```

**Result**: 
- Precompute functions get expected ml_data field
- ML services provide real detections
- Data flows correctly: MLAnalysisResult → .data → ml_data → precompute → Claude

**Verification Complete (2025-08-05)**: 
- ✅ E2E test created and executed successfully
- ✅ unified_analysis.json contains ml_data field with real detections:
  - YOLO: Detected objects (cake, bowl) with confidence scores
  - MediaPipe: Detected 25 poses, 10 faces
  - Scene Detection: Found 27 scene changes
- ✅ Claude received actual ML data and returned valid 6-block structures
- ✅ Total cost for 7 prompts: $0.0097 (Haiku model)

---

# Creative Density

## Input - Structured Data JSON

```json
"timelines": {
      "textOverlayTimeline": {
        "0-1s": {
          "frame": 1,
          "texts": [
            {
              "text": "AMAZING PASTA RECIPE",
              "category": "title",
              "confidence": 0.95,
              "bbox": {"x1": 100, "y1": 50, "x2": 500, "y2": 100}
            }
          ]
        },
        "5-6s": {
          "frame": 6,
          "texts": [
            {
              "text": "Add fresh basil",
              "category": "instruction",
              "confidence": 0.88
            }
          ]
        }
      },
      "stickerTimeline": {
        "2-3s": {
          "frame": 3,
          "stickers": []  // CURRENTLY HARDCODED EMPTY - needs fix
        }
      },
      "effectTimeline": {
        "3-4s": {
          "frame": 4,
          "effects": [{"type": "blur", "intensity": 0.5}]
        }
      },
      "transitionTimeline": {
        "5-6s": {
          "frame": 6,
          "type": "fade",
          "duration": 0.5
        }
      },
      "sceneChangeTimeline": {
        "5-6s": {
          "frame": 6,
          "type": "shot_change",
          "startTime": 5.0,
          "endTime": 10.0
        }
      },
      "objectTimeline": {
        "0-1s": {
          "frame": 1,
          "objects": {"person": 1, "bowl": 1},
          "total_objects": 2
        }
      },
      "gestureTimeline": {
        "0-1s": {
          "frame": 1,
          "gestures": ["open_palm"],
          "dominant": "open_palm"
        }
      },
      "expressionTimeline": {
        "0-1s": {
          "frame": 1,
          "expression": "neutral",
          "confidence": 0.95
        }
      }
    },
    "duration": 30
```

## Metric Example

### 1. Core Metrics (8):
  - average_density
  - max_density
  - min_density
  - std_deviation
  - total_creative_elements
  - element_distribution (dict with 7 sub-metrics)
  - scene_changes
  - timeline_coverage

### 2. Dynamics (5):
  - density_curve (array of density per second)
  - density_volatility
  - acceleration_pattern
  - density_progression
  - empty_seconds (list)

### 3. Interactions (3):
  - multi_modal_peaks (array)
  - element_cooccurrence (dict)
  - dominant_combination

### 4. Key Events (3):
  - peak_density_moments (array)
  - dead_zones (array)
  - density_shifts (array)

### 5. Patterns (5):
  - structural_patterns (dict with 6 boolean flags)
  - density_classification
  - pacing_style
  - cognitive_load_category
  - ml_tags (array)

### 6. Quality Metrics (3):
  - data_completeness
  - detection_reliability (dict with 7 reliability scores)
  - overall_confidence

### Additional Compatibility Fields (9):
  - creative_density_score
  - elements_per_second
  - density_pattern
  - density_pattern_flags
  - creative_ml_tags
  - timeline_frame_counts (dict with 7 counts)
  - duration_seconds
  - patterns_identified
  - peak_moments

## Claude Prompt

```
Goal: Extract creative density features as structured data for ML analysis

  Input File: unified_analysis/[video_id].json

  You will receive precomputed creative density metrics:
  - `average_density`: Mean creative elements per second
  - `max_density`: Maximum elements in any second
  - `min_density`: Minimum elements in any second
  - `std_deviation`: Standard deviation of density
  - `total_creative_elements`: Sum of all creative elements
  - `element_distribution`: Counts by type (text, sticker, effect, transition, object)
  - `peak_density_moments`: Top 5-10 peaks with timestamp, total_elements, surprise_score, breakdown
  - `density_pattern_flags`: Boolean flags (strong_opening_hook, crescendo_pattern, front_loaded, etc.)
  - `density_curve`: Per-second density and primary_element
  - `scene_changes`: Total count from sceneChangeTimeline
  - `timeline_frame_counts`: effect_count, transition_count, object_detection_frames, etc.
  - `density_volatility`: Measure of density variation
  - `multi_modal_peaks`: Moments where multiple element types converge
  - `element_cooccurrence`: Frequency of element type combinations
  - `dead_zones`: Periods with zero creative elements
  - `density_shifts`: Transitions between high and low density periods

  Output the following 6 modular blocks:

  1. densityCoreMetrics
  {
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
      "object": int
    },
    "sceneChangeCount": int,
    "timelineCoverage": float,
    "confidence": float
  }

  2. densityDynamics
  {
    "densityCurve": [
      {"second": 0, "density": 3, "primaryElement": "text"},
      {"second": 2, "density": 1, "primaryElement": "effect"}
    ],
    "volatility": float,
    "accelerationPattern": "front_loaded" | "even" | "back_loaded" | "oscillating",
    "densityProgression": "increasing" | "decreasing" | "stable" | "erratic",
    "emptySeconds": [7, 8, 9],
    "confidence": float
  }

  3. densityInteractions
  {
    "multiModalPeaks": [
      {
        "timestamp": "5-6s",
        "elements": ["text", "effect", "transition"],
        "syncType": "reinforcing" | "complementary" | "redundant"
      }
    ],
    "elementCooccurrence": {
      "text_effect": 5,
      "text_transition": 3,
      "effect_sceneChange": 2
    },
    "dominantCombination": "text_gesture",
    "coordinationScore": float,
    "confidence": float
  }

  4. densityKeyEvents
  {
    "peakMoments": [
      {
        "timestamp": "5-6s",
        "totalElements": 8,
        "surpriseScore": 0.85,
        "elementBreakdown": {"text": 3, "effect": 2, "transition": 1, "gesture": 2}
      }
    ],
    "deadZones": [
      {"start": 7, "end": 9, "duration": 3}
    ],
    "densityShifts": [
      {
        "timestamp": 5,
        "from": "high",
        "to": "low",
        "magnitude": 0.7
      }
    ],
    "confidence": float
  }

  5. densityPatterns
  {
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
    "mlTags": ["hook_driven", "multi_modal", "text_heavy"],
    "confidence": float
  }

  6. densityQuality
  {
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
```

## Claude Prompt Data Input

```json
{
  "precomputed_creative_density_metrics": {
    "average_density": 3.5,
    "max_density": 8,
    ...
  },
  "video_duration": 30
}
```

---

# Emotional Journey

## Input - Structured Data JSON

```json
{
  "expressionTimeline": {
    "0-1s": {"expression": "neutral", "confidence": 0.95},
    "2-3s": {"expression": "happy", "confidence": 0.88},
    "5-6s": {"expression": "surprise", "confidence": 0.92}
  },
  "gestureTimeline": {
    "1-2s": {"gestures": ["thumbs_up"], "dominant": "thumbs_up"},
    "3-4s": {"gestures": ["open_palm"], "dominant": "open_palm"}
  },
  "audioRatioTimeline": {
    "speech_to_music_ratio": 0.3,
    "has_voiceover": true,
    "music_energy": "high"
  },
  "caption": "Wait for it... 🔥 This is amazing!"
}
```

## Metric Example

### 1. Core Metrics (8):
  - unique_emotions: Number of different expressions
  - emotion_transitions: Count of expression changes
  - dominant_emotion: Most frequent expression
  - emotional_diversity: Variety score
  - gesture_emotion_alignment: Sync percentage
  - audio_emotion_alignment: Music-emotion match
  - caption_sentiment: Positive/negative/neutral
  - emotional_intensity: Overall strength

### 2. Dynamics (6):
  - emotion_progression: Array of emotions over time
  - transition_smoothness: How gradual changes are
  - emotional_arc: Overall journey pattern
  - peak_emotion_moments: High intensity timestamps
  - stability_score: Consistency of emotions
  - tempo_emotion_sync: Music tempo alignment

### 3. Interactions (5):
  - gesture_reinforcement: How gestures support emotions
  - audio_mood_congruence: Music-emotion match
  - caption_emotion_alignment: Text-visual sync
  - multimodal_coherence: Overall alignment
  - emotional_contrast_moments: Mismatched elements

### 4. Key Events (4):
  - emotional_peaks: High intensity moments
  - transition_points: When emotions change
  - climax_moment: Peak emotional point
  - resolution_moment: Emotional conclusion

### 5. Patterns (5):
  - journey_archetype: Story structure type
  - emotional_techniques: Methods used
  - pacing_strategy: How emotions unfold
  - engagement_hooks: Emotional triggers
  - viewer_journey_map: Expected audience response

### 6. Quality (3):
  - detection_confidence: Expression accuracy
  - timeline_coverage: Data completeness
  - analysis_reliability: Overall confidence

## Claude Prompt

```
Goal: Extract emotional journey features as structured data for ML analysis

  You will receive precomputed emotional metrics:
  - Expression timeline with confidence scores
  - Gesture timeline with dominant gestures
  - Audio mood indicators (speech_to_music_ratio, music_energy)
  - Caption text for sentiment context
  - Emotion transitions and progression data
  - Gesture-emotion alignment scores
  - Peak emotional moments

  Output the following 6 modular blocks:

  1. emotionalCoreMetrics
  {
    "uniqueEmotions": int,
    "emotionTransitions": int,
    "dominantEmotion": string,
    "emotionalDiversity": float,
    "gestureEmotionAlignment": float,
    "audioEmotionAlignment": float,
    "captionSentiment": "positive" | "negative" | "neutral",
    "emotionalIntensity": float,
    "confidence": float
  }

  2. emotionalDynamics
  {
    "emotionProgression": [
      {"timestamp": "0-2s", "emotion": "neutral", "intensity": 0.5},
      {"timestamp": "2-5s", "emotion": "happy", "intensity": 0.8}
    ],
    "transitionSmoothness": float,
    "emotionalArc": "rising" | "falling" | "stable" | "rollercoaster",
    "peakEmotionMoments": [{"timestamp": "5s", "emotion": "surprise", "intensity": 0.9}],
    "stabilityScore": float,
    "tempoEmotionSync": float,
    "confidence": float
  }

  3. emotionalInteractions
  {
    "gestureReinforcement": float,
    "audioMoodCongruence": float,
    "captionEmotionAlignment": float,
    "multimodalCoherence": float,
    "emotionalContrastMoments": [
      {"timestamp": "3s", "conflict": "happy_face_sad_music"}
    ],
    "confidence": float
  }

  4. emotionalKeyEvents
  {
    "emotionalPeaks": [
      {"timestamp": "5s", "emotion": "surprise", "trigger": "reveal"}
    ],
    "transitionPoints": [
      {"timestamp": "2s", "from": "neutral", "to": "happy", "trigger": "music_change"}
    ],
    "climaxMoment": {"timestamp": "8s", "emotion": "joy", "intensity": 0.95},
    "resolutionMoment": {"timestamp": "28s", "emotion": "satisfaction", "closure": true},
    "confidence": float
  }

  5. emotionalPatterns
  {
    "journeyArchetype": "surprise_delight" | "problem_solution" | "transformation" | "discovery",
    "emotionalTechniques": ["anticipation_build", "contrast", "repetition"],
    "pacingStrategy": "gradual_build" | "quick_shifts" | "steady_state",
    "engagementHooks": ["curiosity", "empathy", "excitement"],
    "viewerJourneyMap": "engaged_throughout" | "slow_build" | "immediate_hook",
    "confidence": float
  }

  6. emotionalQuality
  {
    "detectionConfidence": float,
    "timelineCoverage": float,
    "emotionalDataCompleteness": float,
    "analysisReliability": "high" | "medium" | "low",
    "missingDataPoints": ["audio_analysis"],
    "overallConfidence": float
  }
```

## Claude Prompt Data Input

```json
{
  "precomputed_emotional_metrics": {
    "emotion_counts": {"neutral": 5, "happy": 8, "surprise": 3},
    "emotion_transitions": 6,
    "emotion_sequence": ["neutral", "happy", "happy", "surprise"],
    ...
  },
  "expression_timeline": {...},
  "gesture_timeline": {...},
  "audio_ratio_timeline": {...},
  "caption": "Wait for it... 🔥"
}
```

---

# Person Framing

## Input - Structured Data JSON

```json
{
  "objectTimeline": {
    "0-1s": {
      "objects": [
        {
          "class": "person",
          "bbox": [100, 50, 300, 400],
          "confidence": 0.95,
          "id": "person_1"
        }
      ],
      "frame": 0
    },
    "1-2s": {
      "objects": [
        {
          "class": "person",
          "bbox": [150, 30, 350, 450],
          "confidence": 0.92,
          "id": "person_1"
        }
      ],
      "frame": 30
    }
  },
  "cameraDistanceTimeline": {
    "0-5s": "medium",
    "5-10s": "close",
    "10-15s": "wide"
  },
  "gestureTimeline": {
    "2-3s": {"gestures": ["pointing"], "dominant": "pointing"},
    "5-6s": {"gestures": ["open_palm"], "dominant": "open_palm"}
  },
  "expressionTimeline": {
    "0-2s": {"expression": "neutral", "confidence": 0.95},
    "2-4s": {"expression": "happy", "confidence": 0.88}
  }
}
```

## Metric Example

### Core Metrics (11):
1. person_presence_rate: 0.95 (95% of video has person)
2. avg_person_count: 1.2
3. max_simultaneous_people: 2
4. dominant_framing: "medium"
5. framing_changes: 4
6. person_screen_coverage: 0.35 (35% of screen)
7. position_stability: 0.8
8. gesture_clarity_score: 0.85
9. face_visibility_rate: 0.9
10. body_visibility_rate: 0.75
11. overall_framing_quality: 0.88

## Claude Prompt

```
Goal: Extract person framing features as structured data for ML analysis

  You will receive precomputed person framing metrics:
  - Person detection rates and counts
  - Camera distance progression
  - Screen coverage and positioning
  - Gesture and expression visibility
  - Framing quality indicators

  Output the following 6 modular blocks:

  1. personFramingCoreMetrics
  {
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
  }

  2. personFramingDynamics
  {
    "framingProgression": [
      {"timestamp": "0-5s", "distance": "medium", "coverage": 0.35}
    ],
    "movementPattern": "static" | "gradual" | "dynamic",
    "zoomTrend": "in" | "out" | "stable" | "varied",
    "stabilityTimeline": [{"second": 0, "stability": 0.9}],
    "framingTransitions": [
      {"timestamp": 5, "from": "medium", "to": "close"}
    ],
    "confidence": float
  }

  3. personFramingInteractions
  {
    "gestureFramingSync": float,
    "expressionVisibility": float,
    "multiPersonCoordination": float,
    "actionSpaceUtilization": float,
    "framingPurposeAlignment": float,
    "confidence": float
  }

  4. personFramingKeyEvents
  {
    "framingHighlights": [
      {"timestamp": "5-8s", "type": "close_up_emotion", "impact": "high"}
    ],
    "criticalFramingMoments": [
      {"timestamp": 10, "event": "person_exit", "framing": "wide"}
    ],
    "optimalFramingPeriods": [
      {"start": 5, "end": 15, "reason": "gesture_clarity"}
    ],
    "confidence": float
  }

  5. personFramingPatterns
  {
    "framingStrategy": "intimate" | "observational" | "dynamic" | "staged",
    "visualNarrative": "single_focus" | "multi_person" | "environment_aware",
    "technicalExecution": "professional" | "casual" | "experimental",
    "engagementTechniques": ["direct_address", "gesture_emphasis", "emotional_close_ups"],
    "productionValue": "high" | "medium" | "low",
    "confidence": float
  }

  6. personFramingQuality
  {
    "detectionReliability": float,
    "trackingConsistency": float,
    "framingDataCompleteness": float,
    "analysisLimitations": ["occlusion_periods", "motion_blur"],
    "overallConfidence": float
  }
```

---

# Scene Pacing

## Input - Structured Data JSON

```json
{
  "sceneChangeTimeline": {
    "0.0-3.5s": {
      "scene_number": 1,
      "start_time": 0.0,
      "end_time": 3.5,
      "duration": 3.5,
      "frame_start": 0,
      "frame_end": 105
    },
    "3.5-7.2s": {
      "scene_number": 2,
      "start_time": 3.5,
      "end_time": 7.2,
      "duration": 3.7,
      "frame_start": 105,
      "frame_end": 216
    },
    "7.2-12.0s": {
      "scene_number": 3,
      "start_time": 7.2,
      "end_time": 12.0,
      "duration": 4.8,
      "frame_start": 216,
      "frame_end": 360
    }
  },
  "video_duration": 30.0
}
```

## Metric Example

### Core Metrics (31 features):
- total_scenes: 8
- scene_change_rate: 0.27 (changes per second)
- avg_scene_duration: 3.75
- min_scene_duration: 1.2
- max_scene_duration: 7.5
- scene_duration_variance: 2.3
- quick_cuts_count: 3 (scenes < 2s)
- long_takes_count: 2 (scenes > 5s)
- scene_rhythm_score: 0.75
- pacing_consistency: 0.82
- video_duration: 30.0

### Additional metrics include:
- Scene duration distribution
- Pacing acceleration patterns
- Rhythm regularity
- Cut intensity moments
- Scene transition patterns

## Claude Prompt

```
Goal: Extract scene pacing features as structured data for ML analysis

  You will receive precomputed scene pacing metrics:
  - Scene count and durations
  - Scene change rate and rhythm
  - Quick cuts vs long takes
  - Pacing consistency measures
  - Scene transition patterns

  Output the following 6 modular blocks:

  1. scenePacingCoreMetrics
  {
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
  }

  2. scenePacingDynamics
  {
    "pacingCurve": [
      {"second": 0, "cutsPerSecond": 0.2, "intensity": "low"},
      {"second": 10, "cutsPerSecond": 0.5, "intensity": "high"}
    ],
    "accelerationPattern": "steady" | "building" | "declining" | "variable",
    "rhythmRegularity": float,
    "pacingMomentum": "maintaining" | "accelerating" | "decelerating",
    "dynamicRange": float,
    "confidence": float
  }

  3. scenePacingInteractions
  {
    "contentPacingAlignment": float,
    "emotionalPacingSync": float,
    "narrativeFlowScore": float,
    "viewerAdaptationCurve": "smooth" | "jarring" | "engaging",
    "pacingContrastMoments": [
      {"timestamp": 15, "shift": "fast_to_slow", "impact": 0.8}
    ],
    "confidence": float
  }

  4. scenePacingKeyEvents
  {
    "pacingPeaks": [
      {"timestamp": "10-12s", "cutsPerSecond": 0.8, "intensity": "high"}
    ],
    "pacingValleys": [
      {"timestamp": "20-25s", "sceneDuration": 5.0, "type": "long_take"}
    ],
    "criticalTransitions": [
      {"timestamp": 15, "fromPace": "fast", "toPace": "slow", "effect": "emphasis"}
    ],
    "rhythmBreaks": [
      {"timestamp": 18, "expectedDuration": 2.5, "actualDuration": 0.8}
    ],
    "confidence": float
  }

  5. scenePacingPatterns
  {
    "pacingStyle": "music_video" | "narrative" | "montage" | "documentary",
    "editingRhythm": "metronomic" | "syncopated" | "free_form" | "accelerando",
    "visualTempo": "slow" | "moderate" | "fast" | "variable",
    "cutMotivation": "action_driven" | "beat_driven" | "emotion_driven" | "random",
    "pacingTechniques": ["match_cut", "jump_cut", "cross_cut", "smash_cut"],
    "confidence": float
  }

  6. scenePacingQuality
  {
    "sceneDetectionAccuracy": float,
    "transitionAnalysisReliability": float,
    "pacingDataCompleteness": float,
    "technicalQuality": "professional" | "amateur" | "mixed",
    "analysisLimitations": ["motion_blur_periods", "transition_effects"],
    "overallConfidence": float
  }
```

---

# Speech Analysis

## Input - Structured Data JSON

```json
{
  "speechTimeline": {
    "2.0-5.5s": {
      "text": "Check this out, this is amazing",
      "confidence": 0.95,
      "speaker": "primary",
      "word_timestamps": [
        {"word": "Check", "start": 2.0, "end": 2.3},
        {"word": "this", "start": 2.3, "end": 2.5}
      ]
    },
    "8.0-12.0s": {
      "text": "You won't believe what happens next",
      "confidence": 0.92,
      "speaker": "primary"
    }
  },
  "expressionTimeline": {
    "2-3s": {"expression": "neutral", "confidence": 0.95},
    "8-9s": {"expression": "excited", "confidence": 0.88}
  },
  "gestureTimeline": {
    "2-3s": {"gestures": ["pointing"], "dominant": "pointing"},
    "9-10s": {"gestures": ["open_palm"], "dominant": "open_palm"}
  },
  "video_duration": 30.0
}
```

## Metric Example

### Core Metrics (30 features):
- total_speech_segments: 5
- speech_duration: 12.5
- speech_rate: 0.42 (portion of video with speech)
- words_per_minute: 145
- unique_speakers: 1
- primary_speaker_dominance: 1.0
- avg_confidence: 0.93
- speech_clarity_score: 0.88
- pause_count: 4
- avg_pause_duration: 1.2

### Additional metrics include:
- Speech-gesture synchronization
- Expression-speech alignment
- Speech pacing patterns
- Emphasis detection
- Dialogue vs monologue classification

## Claude Prompt

```
Goal: Extract speech analysis features as structured data for ML analysis

  You will receive precomputed speech metrics:
  - Speech segments with timing and confidence
  - Word count and speech rate
  - Speaker identification
  - Speech-gesture synchronization
  - Speech-expression alignment
  - Pause patterns and emphasis

  Output the following 6 modular blocks:

  1. speechCoreMetrics
  {
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
  }

  2. speechDynamics
  {
    "speechPacingCurve": [
      {"timestamp": "0-5s", "wordsPerSecond": 2.5, "intensity": "moderate"}
    ],
    "pacingVariation": float,
    "speechRhythm": "steady" | "variable" | "accelerating" | "decelerating",
    "pausePattern": "natural" | "dramatic" | "rushed" | "irregular",
    "emphasisMoments": [
      {"timestamp": 8.5, "word": "amazing", "emphasisType": "volume"}
    ],
    "confidence": float
  }

  3. speechInteractions
  {
    "speechGestureSync": float,
    "speechExpressionAlignment": float,
    "verbalVisualCoherence": float,
    "multimodalEmphasis": [
      {"timestamp": 9, "speech": "believe", "gesture": "open_palm", "alignment": 0.9}
    ],
    "conversationalDynamics": "monologue" | "dialogue" | "mixed",
    "confidence": float
  }

  4. speechKeyEvents
  {
    "keyPhrases": [
      {"timestamp": 2.5, "phrase": "check this out", "significance": "hook"}
    ],
    "speechClimax": {"timestamp": 10, "text": "happens next", "intensity": 0.9},
    "silentMoments": [
      {"start": 5.5, "end": 8.0, "duration": 2.5, "purpose": "anticipation"}
    ],
    "transitionPhrases": [
      {"timestamp": 15, "phrase": "but wait", "function": "pivot"}
    ],
    "confidence": float
  }

  5. speechPatterns
  {
    "deliveryStyle": "conversational" | "instructional" | "narrative" | "promotional",
    "speechTechniques": ["direct_address", "rhetorical_questions", "repetition"],
    "toneCategory": "enthusiastic" | "calm" | "urgent" | "informative",
    "linguisticComplexity": "simple" | "moderate" | "complex",
    "engagementStrategy": "storytelling" | "demonstration" | "explanation",
    "confidence": float
  }

  6. speechQuality
  {
    "transcriptionConfidence": float,
    "audioQuality": "clear" | "moderate" | "poor",
    "speechDataCompleteness": float,
    "analysisLimitations": ["background_noise", "overlapping_speech"],
    "overallConfidence": float
  }
```

---

# Visual Overlay Analysis

## Input - Structured Data JSON

```json
{
  "textOverlayTimeline": {
    "0-3s": {
      "text": "Wait for it...",
      "position": "center",
      "style": "bold",
      "animation": "fade_in"
    },
    "5-8s": {
      "text": "Mind = Blown 🤯",
      "position": "top",
      "style": "impact",
      "animation": "bounce"
    }
  },
  "stickerTimeline": {
    "3-5s": {
      "sticker_id": "fire_emoji",
      "position": "bottom_right",
      "size": "medium"
    }
  },
  "gestureTimeline": {
    "2-3s": {"gestures": ["pointing"], "dominant": "pointing"},
    "6-7s": {"gestures": ["mind_blown"], "dominant": "mind_blown"}
  },
  "speechTimeline": {
    "2-5s": {
      "text": "Watch what happens here",
      "confidence": 0.95
    }
  },
  "objectTimeline": {
    "0-10s": {
      "objects": [{"class": "person", "bbox": [100, 100, 200, 300]}]
    }
  },
  "video_duration": 30.0
}
```

## Metric Example

### Core Metrics (40 features):
- total_text_overlays: 8
- unique_texts: 6
- avg_texts_per_second: 0.27
- time_to_first_text: 0.5
- avg_text_display_duration: 3.2
- total_stickers: 4
- unique_stickers: 3
- text_sticker_ratio: 2.0
- overlay_density: 0.4
- visual_complexity_score: 0.65

### Additional metrics include:
- Text appearance patterns
- Animation usage
- Position distribution
- Text-speech synchronization
- Gesture-overlay coordination
- Reading time adequacy
- Visual hierarchy scores

## Claude Prompt

```
Goal: Extract visual overlay features as structured data for ML analysis

  You will receive precomputed visual overlay metrics:
  - Text overlay frequency and timing
  - Sticker usage patterns
  - Text-speech synchronization
  - Gesture-overlay coordination
  - Visual density measures
  - Overlay positioning patterns

  Output the following 6 modular blocks:

  1. overlaysCoreMetrics
  {
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
  }

  2. overlaysDynamics
  {
    "overlayTimeline": [
      {"timestamp": "0-3s", "overlayCount": 1, "type": "text", "density": "low"}
    ],
    "appearancePattern": "gradual" | "burst" | "rhythmic" | "random",
    "densityProgression": "building" | "front_loaded" | "even" | "climactic",
    "animationIntensity": float,
    "visualPacing": "slow" | "moderate" | "fast" | "variable",
    "confidence": float
  }

  3. overlaysInteractions
  {
    "textSpeechSync": float,
    "overlayGestureCoordination": float,
    "visualEmphasisAlignment": float,
    "multiLayerComplexity": "simple" | "moderate" | "complex",
    "readingFlowScore": float,
    "confidence": float
  }

  4. overlaysKeyEvents
  {
    "impactfulOverlays": [
      {"timestamp": "5s", "text": "Mind = Blown", "impact": "high", "reason": "climax"}
    ],
    "overlayBursts": [
      {"timestamp": "10-12s", "count": 5, "purpose": "information_dump"}
    ],
    "keyTextMoments": [
      {"timestamp": 0, "text": "Wait for it", "function": "hook"}
    ],
    "stickerHighlights": [
      {"timestamp": 3, "sticker": "fire", "purpose": "emphasis"}
    ],
    "confidence": float
  }

  5. overlaysPatterns
  {
    "overlayStrategy": "minimal" | "moderate" | "heavy" | "dynamic",
    "textStyle": "clean" | "decorative" | "mixed" | "chaotic",
    "communicationApproach": "reinforcing" | "supplementary" | "dominant",
    "visualTechniques": ["text_animation", "emoji_emphasis", "color_coding"],
    "productionQuality": "professional" | "casual" | "amateur",
    "confidence": float
  }

  6. overlaysQuality
  {
    "textDetectionAccuracy": float,
    "stickerRecognitionRate": float,
    "overlayDataCompleteness": float,
    "readabilityIssues": ["text_too_fast", "overlapping_elements"],
    "visualAccessibility": "high" | "medium" | "low",
    "overallConfidence": float
  }
```

---

# Metadata Analysis

## Input - Structured Data

```json
{
  "static_metadata": {
    "captionText": "Wait for it... 🔥 This is the best pasta recipe ever! #cooking #foodie #recipe",
    "hashtags": [
      {"name": "cooking", "id": "cooking"},
      {"name": "foodie", "id": "foodie"},
      {"name": "recipe", "id": "recipe"}
    ],
    "createTime": "2024-03-15T10:30:00Z",
    "author": {
      "username": "chef_amazing",
      "verified": true
    },
    "stats": {
      "views": 1500000,
      "likes": 250000,
      "comments": 15000,
      "shares": 8000
    }
  },
  "metadata_summary": {
    "videoLength": 30,
    "engagement": {
      "totalEngagements": 273000,
      "engagementRate": 18.2
    }
  },
  "video_duration": 30
}
```

## Metric Example

### Core Metrics (45 features):
- caption_length: 89
- word_count: 15
- hashtag_count: 3
- emoji_count: 1
- mention_count: 0
- link_present: false
- publish_hour: 10
- publish_day_of_week: 4
- engagement_rate: 18.2
- viral_potential_score: 0.85

### Additional metrics include:
- Hashtag classification (niche vs generic)
- Caption style and sentiment
- Hook detection
- CTA identification
- Linguistic markers
- Quality assessments

## Claude Prompt

```
Goal: Extract metadata features as structured data for ML analysis

  You will receive precomputed metadata metrics:
  - Caption analysis (length, words, emojis, mentions)
  - Hashtag details and classification
  - Engagement metrics and ratios
  - Publishing time data
  - Creator information
  - Sentiment and readability scores
  - Detected hooks and CTAs

  Output the following 6 modular blocks:

  1. metadataCoreMetrics
  {
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
  }

  2. metadataDynamics
  {
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
  }

  3. metadataInteractions
  {
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
  }

  4. metadataKeyEvents
  {
    "hashtags": [
      {
        "tag": "#cooking",
        "position": 1,
        "type": "niche",
        "estimatedReach": "medium"
      }
    ],
    "emojis": [
      {
        "emoji": "🔥",
        "count": 1,
        "sentiment": "positive",
        "emphasis": true
      }
    ],
    "hooks": [
      {
        "text": "wait for it",
        "position": "start",
        "type": "curiosity",
        "strength": 0.8
      }
    ],
    "callToActions": [],
    "confidence": float
  }

  5. metadataPatterns
  {
    "linguisticMarkers": {
      "questionCount": 0,
      "exclamationCount": 1,
      "capsLockWords": 0,
      "personalPronounCount": 0
    },
    "hashtagPatterns": {
      "leadWithGeneric": false,
      "allCaps": false
    },
    "confidence": float
  }

  6. metadataQuality
  {
    "captionPresent": true,
    "hashtagsPresent": true,
    "statsAvailable": true,
    "publishTimeAvailable": true,
    "creatorDataAvailable": true,
    "captionQuality": "high",
    "hashtagQuality": "mixed",
    "overallConfidence": 0.95
  }
```

---

# Pipeline Summary

## Data Flow
```
Unified Analysis → Precompute Functions → Precomputed Metrics → Claude Prompts → ML Features
```

Each analysis flow:
1. Takes specific timelines from unified analysis
2. Computes domain-specific metrics
3. Sends metrics to Claude with structured prompt
4. Claude outputs 6 standardized blocks
5. Blocks are used as ML training features

The modular structure ensures consistency across all video types and enables reliable ML model training.