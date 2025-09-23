# ML MVP 2.0 - Architecture Decisions

## Date: 2025-08-26
## Participants: Jorge, Claude, GPT-4

---

## Executive Summary
Defined architecture for ML feature engineering pipeline with ~TBC features (pending feature audit) from RumiAI video analysis, targeting Random Forest and K-means models with duration-aware temporal analysis.

> **Feature Count Note**: Final feature count to be confirmed after audit for:
> - Removal of interpreted/semantic features
> - Elimination of redundant/duplicate features
> - Validation of RF/K-means compatibility
> - Integration of new temporal event features (~30 estimated)

> **Related Documentation**: This architecture is implemented in **[MLProjectsGrassrootsv2.md](./MLProjectsGrassrootsv2.md)**, which covers the end-to-end ML training pipeline. While this document (MLMVP2) focuses on the canonical JSON structure and feature engineering design, MLProjectsGrassrootsv2 details the operational implementation.

---

## 1. Core Architecture Decision: Single Canonical JSON

### Chosen Approach
**One canonical JSON per video as source of truth**, with materialized train-ready artifacts for each model and duration bucket.

### Structure
```json
{
  "video_id": "abc123",
  "duration_sec": 70,
  "duration_bucket": "31-60s",
  
  "features_base": {
    "cd_avgDensity": 24.3,
    "cd_totalElements": 170,
    "ej_uniqueEmotions": 2,
    "pf_averageFaceSize": 9.86,
    "sp_totalScenes": 4,
    "sa_totalWords": 28,
    "vo_totalOverlays": 7,
    "// ... features TBC after audit ...": 0
  },
  
  "temporal_summaries": {
    "hook_window": {
      "hook_0to3s_density": 52,
      "hook_0to3s_surprise_score": 0.89,
      "hook_0to3s_has_question": true,
      "hook_0to3s_face_visible": true,
      "hook_0to3s_motion_intensity": 0.76,
      "hook_0to3s_text_count": 4,
      "hook_0to3s_emotion": "surprise",
      "hook_effectiveness_score": 0.84
    },
    "middle_window": {
      "len_sec": 64,
      "shape": {
        "peak_value": 62,
        "peak_position": 0.58,
        "oscillations": 2,
        "trend_slope": -0.15,
        "variance": 12.3,
        "cv": 0.28
      },
      "bins": {
        "early_density": 35,
        "mid_density": 62,
        "late_density": 41
      },
      "piecewise": {
        "slope_early": 2.1,
        "slope_mid": 0.2,
        "slope_late": -1.8,
        "break_pos_1": 0.33,
        "break_pos_2": 0.67
      },
      "rhythm": {
        "burstiness": 1.8,
        "cut_rate_slope": 0.15,
        "spectral_centroid": 0.45
      },
      "is_present": true
    },
    "closing_window": {
      "closing_3s_density": 48,
      "closing_3s_has_cta": true,
      "closing_3s_cta_type": "follow",
      "closing_3s_gesture_present": true,
      "closing_3s_text_count": 3,
      "closing_3s_emotion": "excitement",
      "closing_3s_face_visible": true,
      "closing_effectiveness_score": 0.79
    }
  },
  
  "audit": {
    "schema_version": "1.0.0",
    "extractor_version": "1.4.2",
    "extracted_at": "2025-08-26T10:44:17Z"
  }
}
```

### Why This Over Alternatives
- **Single source of truth prevents schema drift** (vs multiple base JSONs)
- **All features in one place** makes governance tractable
- **Fixed schema** enables CI/CD validation
- **Versioning is simple** with audit trail

---

## 2. Model-Specific Feature Requirements

### Why Classical ML Over Deep Learning
> **Model Selection Rationale**: We chose Random Forest + K-means over deep learning approaches (Transformers, CNN/LSTM) because:
> - **Data Scale**: 60 videos per bucket is ideal for classical ML, insufficient for deep learning
> - **Interpretability**: Content creators need to understand WHY patterns work (RF provides clear feature importance)
> - **Infrastructure**: No GPU requirements, runs on standard hardware
> - **Temporal Patterns**: Our Hook/Middle/Closing windows extract temporal insights without needing recurrent architectures

### Random Forest vs K-means Need Different Features
**Key Insight**: RF and K-means don't need identical feature subsets

| Aspect | Random Forest | K-means |
|--------|--------------|---------|
| Feature Count | 24/25 features from creative_density | 20/25 features from creative_density |
| Categorical Handling | One-hot encoding works well | Label encoding (assumes ordinality) |
| Complex Structures | Can handle nested arrays with extraction | Needs flattened numerical only |
| Scaling | Not required | Essential |
| High Dimensionality | Handles well | Curse of dimensionality |

### Materialized Artifacts Per Model
```
artifacts/model=rf/bucket=0to15s/train_rf_0to15s.parquet
artifacts/model=km/bucket=31to60s/train_km_31to60s.parquet
```

---

## 3. Temporal Analysis Architecture

### The Problem We Solved
**Initial Issue**: Fixed temporal bins (0-3s, 3-10s, 10-20s) don't scale across video durations
- 15s video: `mid_10to20s` doesn't exist
- 70s video: `late_rest` is 50 seconds (72% of video!)

### The Solution: Non-Overlapping Temporal Windows with Rich Middle Analysis

#### Core Principle: Clean Boundaries
To avoid double-counting and feature correlation issues, we use non-overlapping temporal windows:
- **Hook Window**: First 3 seconds (user scroll decision)
- **Middle Window**: Everything between hook and closing (narrative development)  
- **Closing Window**: Last 3 seconds (conversion moment)

#### Hook Window (User Behavior-Based)
The hook represents the critical scroll-decision moment that is universal across all video lengths. Users scrolling through TikTok/Reels don't check video duration - they make watch/skip decisions in the first 3 seconds.

```json
"hook_window": {
  "hook_0to3s_density": 52,          // Element density in hook
  "hook_0to3s_surprise_score": 0.89, // Surprise/novelty factor
  "hook_0to3s_has_question": true,   // Question posed to viewer
  "hook_0to3s_face_visible": true,   // Human face present
  "hook_0to3s_motion_intensity": 0.76, // Movement/action level
  "hook_0to3s_text_count": 4,        // Text overlays in hook
  "hook_0to3s_emotion": "surprise",  // Dominant emotion
  "hook_effectiveness_score": 0.84   // Composite hook strength
}
```

**Why Hook Window is Special:**
- **Duration-agnostic**: Always 0-3 seconds regardless of video length
- **Behavior-driven**: Matches actual user scroll patterns
- **Platform-universal**: Same behavior on TikTok, Reels, YouTube Shorts
- **Highest impact**: Most predictive of engagement/retention

#### Closing Window (Engagement Conversion-Based)
The closing window captures the critical CTA moment where viewers decide to follow/share/save. Like the hook, this is duration-agnostic - CTAs happen in the final 3 seconds regardless of total length.

```json
"closing_window": {
  "closing_3s_density": 48,           // Element density in closing
  "closing_3s_has_cta": true,         // CTA present
  "closing_3s_cta_type": "follow",    // Type: follow/like/share/buy
  "closing_3s_gesture_present": true, // Pointing/gesture for emphasis
  "closing_3s_text_count": 3,         // CTA text overlays
  "closing_3s_emotion": "excitement", // Final emotion
  "closing_3s_face_visible": true,    // Still engaging vs turned away
  "closing_effectiveness_score": 0.79 // CTA strength composite
}
```

**Why Closing Window is Critical:**
- **Precise CTA timing**: Most CTAs occur at -3s to -1s mark
- **Not diluted**: Final 20% of 60s video = 12s (too broad)
- **Conversion moment**: Where follow/share decisions happen
- **Loop consideration**: Sets up video replay on platforms

#### Middle Window (Narrative Analysis) - ENHANCED WITH EVENT-CENTRIC APPROACH

**GPT's Critique**: "Temporal analysis must not collapse the middle narrative arc into something too coarse. The insights between hook and CTA are often what make or break a viral TikTok (reveals, twists, pacing shifts, 'second hooks')."

**Our Solution**: Dual-layer approach combining continuous analysis with discrete event extraction.

The middle window captures everything between hook and closing, providing rich temporal insights without overlap.

##### Layer 1: Continuous Analysis (Existing)
```json
"middle_window": {
  "len_sec": 54,  // For 60s video: 3s to 57s
  "shape": {
    "peak_value": 62,
    "peak_position": 0.58,    // Peak at 58% through middle
    "oscillations": 2,         // Number of peaks detected
    "trend_slope": -0.15,      // Overall trend
    "variance": 12.3,
    "cv": 0.28                 // Coefficient of variation
  },
  "bins": {
    "early_density": 35,       // First third of middle
    "mid_density": 62,         // Middle third
    "late_density": 41         // Last third
  },
  "piecewise": {              // For videos > 30s
    "slope_early": 2.1,        // Rising action
    "slope_mid": 0.2,          // Plateau
    "slope_late": -1.8,        // Falling action
    "break_pos_1": 0.33,       // First transition
    "break_pos_2": 0.67        // Second transition
  },
  "rhythm": {
    "burstiness": 1.8,         // Temporal regularity
    "cut_rate_slope": 0.15,    // Acceleration of cuts
    "spectral_centroid": 0.45  // Frequency of changes
  },
  
  // Layer 2: Event-Centric Features (NEW)
  "temporal_events": [  // For Claude/human interpretation
    {"time": 0.24, "type": "emotion_peak", "subtype": "surprise", "intensity": 0.88},
    {"time": 0.24, "type": "density_peak", "subtype": "visual_burst", "intensity": 0.91},
    {"time": 0.57, "type": "motion_peak", "subtype": "camera_zoom", "intensity": 0.83}
  ],
  
  // Fixed ML features derived from temporal_events
  "event_count": 3,
  "g1_time": 0.24,       // Global event 1 position
  "g1_type_id": 3,       // 3 = emotion (see encoding map below)
  "g1_mag": 0.88,        // Event 1 magnitude
  "g2_time": 0.24,       
  "g2_type_id": 1,       // 1 = density
  "g2_mag": 0.91,
  "g3_time": 0.57,
  "g3_type_id": 2,       // 2 = motion
  "g3_mag": 0.83,
  "g4_time": null,       // Unused slot
  "g4_type_id": 0,
  "g4_mag": null,
  "g5_time": null,
  "g5_type_id": 0,
  "g5_mag": null,
  
  // Distance and alignment metrics
  "g12_distance": 0.0,           // Events 1&2 aligned
  "g23_distance": 0.33,          // Distance between event 2 and 3
  "hook_to_first_peak": 0.06,   // 0.24 - 0.18 (hook end)
  "last_peak_to_cta": 0.18,     // 0.75 (CTA start) - 0.57
  "global_peak_spread": 0.33,   // max_time - min_time
  "g1_flows_aligned_count": 2,  // emotion + density peaked together
  
  "is_present": true           // False for videos ≤ 6s
}
```

**Adaptive Granularity Based on Duration:**
- **≤15s**: Shape stats only (middle too short for bins) + 1-2 events max
- **16-30s**: Shape + thirds within middle + 2-3 events
- **31-60s**: Shape + thirds (mapped from quartiles) + piecewise + 3-4 events
- **61-120s**: Shape + thirds (mapped from quintiles) + piecewise + rhythm + 4-5 events

##### Layer 2: Event-Centric Analysis Implementation

**Event Count by Duration:**
- **0-15s**: 1-2 events (one main reveal/peak)
- **16-30s**: 2-3 events (hook → reveal → reinforcement)
- **31-60s**: 3-4 events (multiple narrative beats)
- **61-120s**: 4-5 events (complex narrative with multiple hooks)

**Event Type Encoding Map:**
```python
# Label encoding for event types
0 = none/null
1 = density        # Visual element density peaks
2 = motion         # High motion/action peaks
3 = emotion        # Emotional intensity peaks
4 = text_overlay   # Text density peaks
5 = face_close     # Close-up face moments
6 = speech_emphasis # Vocal emphasis peaks
7 = scene_change   # Scene transition clusters
8 = gesture        # Significant gesture peaks
```

**Event Selection Policy:**
1. Detect peaks across all analysis flows (creative_density, emotional_journey, etc.)
2. Rank all events by: intensity → novelty → recency
3. Keep top K per flow (K=2) and top K global (K=3-5 based on duration)
4. Apply alignment tolerance: max(3% of duration, 300ms)
5. If events fall within tolerance, keep higher intensity and record alignment

**Fixed Feature Structure:**
- Always 5 event slots (g1 through g5)
- Use null values for empty slots
- Include explicit event_count feature
- Distance and alignment metrics for pattern discovery

**Why This Dual-Layer Approach Works:**
- **Preserves Continuous Narrative**: Shape/bins/piecewise capture overall flow
- **Captures Critical Moments**: Events identify specific reveals, twists, "second hooks"
- **ML-Compatible**: Fixed features work with Random Forest and K-means
- **Human-Interpretable**: temporal_events array for Claude/analysis
- **No Information Loss**: Both continuous and discrete patterns preserved

### Duration-Specific Middle Window Analysis

#### 0-15s Videos (Short Form)
**Middle Window Coverage**: 3s to 12s (for 15s video) = 9s of content

```json
"middle_window": {
  "len_sec": 9,
  "shape": {
    "peak_value": 45,
    "peak_position": 0.67,    // Peak at 6s mark (67% through middle)
    "oscillations": 1,         // Can detect 1-2 peaks max
    "trend_slope": 0.8,       
    "variance": 8.2,
    "cv": 0.19
  },
  "bins": null,              // TOO SHORT for bins
  "piecewise": null,         // TOO SHORT for piecewise
  "rhythm": null,            // TOO SHORT for rhythm
  "is_present": true
}
```
**Peak Detection**: 1-2 major peaks with positions
**Example**: Fashion video with outfit reveal at 9s mark

#### 16-30s Videos (Medium)
**Middle Window Coverage**: 3s to 27s (for 30s video) = 24s of content

```json
"middle_window": {
  "len_sec": 24,
  "shape": {
    "peak_value": 72,
    "peak_position": 0.42,    // Peak at ~10s into middle
    "oscillations": 2,         // Can detect 2-3 peaks
    "trend_slope": -0.3,
    "variance": 15.4,
    "cv": 0.31
  },
  "bins": {                  // Simple thirds
    "early_density": 35,     // 3-11s average
    "mid_density": 72,       // 11-19s average  
    "late_density": 28       // 19-27s average
  },
  "piecewise": null,         // Not for <30s
  "rhythm": null,            // Not for <30s
  "is_present": true
}
```
**Peak Detection**: 2-3 distinct peaks
**Example**: Tutorial with intro→demo→recap structure visible in bins

#### 31-60s Videos (Long)
**Middle Window Coverage**: 3s to 57s (for 60s video) = 54s of content

```json
"middle_window": {
  "len_sec": 54,
  "shape": {
    "peak_value": 85,
    "peak_position": 0.33,    // Early peak at ~18s
    "oscillations": 3,         // Can detect 3-4 peaks
    "trend_slope": 0.1,
    "variance": 22.7,
    "cv": 0.38
  },
  "bins": {                  // Quartiles → 3 bins
    "early_density": 48,     // avg(q1,q2): 3-30s
    "mid_density": 85,       // q3: 30-43s
    "late_density": 52       // q4: 43-57s
  },
  "piecewise": {
    "slope_early": 3.2,      // Sharp rise
    "slope_mid": -0.5,       // Gentle fall
    "slope_late": 1.8,       // Rise again
    "break_pos_1": 0.33,     // Transition at 18s
    "break_pos_2": 0.67      // Transition at 36s
  },
  "rhythm": {
    "burstiness": 2.1,       // Irregular pacing
    "cut_rate_slope": 0.25,  // Accelerating
    "spectral_centroid": 0.6
  },
  "is_present": true
}
```
**Peak Detection**: 3-4 major peaks with precise timing
**Example**: Story with setup→conflict→resolution→twist all visible

#### 61-120s Videos (Extra Long)
**Middle Window Coverage**: 3s to 117s (for 120s video) = 114s of content

```json
"middle_window": {
  "len_sec": 114,
  "shape": {
    "peak_value": 92,
    "peak_position": 0.25,    // Early peak
    "oscillations": 5,         // Can detect 5-6 peaks!
    "trend_slope": -0.2,
    "variance": 28.3,
    "cv": 0.42
  },
  "bins": {                  // Quintiles → 3 bins
    "early_density": 55,     // avg(q1,q2): 3-48s
    "mid_density": 92,       // q3: 48-71s
    "late_density": 44       // avg(q4,q5): 71-117s
  },
  "piecewise": {
    "slope_early": 2.5,
    "slope_mid": -3.1,       // Sharp drop
    "slope_late": 0.8,
    "break_pos_1": 0.25,     // at ~28s
    "break_pos_2": 0.75      // at ~85s
  },
  "rhythm": {
    "burstiness": 3.2,       // Very bursty
    "cut_rate_slope": -0.1,  // Decelerating
    "spectral_centroid": 0.35
  },
  "is_present": true
}
```
**Peak Detection**: 5-6 major peaks with complex patterns
**Example**: Multi-segment content with recurring themes

### Peak Detection Capabilities

| Duration | Max Peaks | Peak Resolution | What We Can Identify |
|----------|-----------|-----------------|----------------------|
| **0-15s** | 1-2 peaks | ~3-4s apart | Single climax, maybe a secondary moment |
| **16-30s** | 2-3 peaks | ~6-8s apart | Opening burst, main peak, possible third |
| **31-60s** | 3-4 peaks | ~10-12s apart | Multiple story beats, complex patterns |
| **61-120s** | 5-6 peaks | ~15-20s apart | Full narrative arcs with multiple climaxes |

### The 3-Bin Mapping Strategy
To maintain fixed schema while capturing adaptive detail:
- **Thirds** (16-30s): Direct mapping to early/mid/late
- **Quartiles** (31-60s): [q1, q2, q3, q4] → [avg(q1,q2), q3, q4]
- **Quintiles** (61-120s): [q1, q2, q3, q4, q5] → [avg(q1,q2), q3, avg(q4,q5)]

This preserves the essential shape while keeping consistent column structure.

### The Universal Temporal Features
Every video gets these exact features regardless of length:

**Hook Window (8 features)**
1. `hook_0to3s_density`
2. `hook_0to3s_surprise_score`
3. `hook_0to3s_has_question`
4. `hook_0to3s_face_visible`
5. `hook_0to3s_motion_intensity`
6. `hook_0to3s_text_count`
7. `hook_0to3s_emotion`
8. `hook_effectiveness_score`

**Middle Window Shape (6 features)**
9. `middle_peak_value`
10. `middle_peak_position`
11. `middle_oscillations`
12. `middle_trend_slope`
13. `middle_variance`
14. `middle_cv`

**Middle Window Bins (3 features)**
15. `middle_early_density`
16. `middle_mid_density`
17. `middle_late_density`

**Middle Window Piecewise (5 features, videos >30s)**
18. `middle_slope_early`
19. `middle_slope_mid`
20. `middle_slope_late`
21. `middle_break_pos_1`
22. `middle_break_pos_2`

**Middle Window Rhythm (3 features, videos >30s)**
23. `middle_burstiness`
24. `middle_cut_rate_slope`
25. `middle_spectral_centroid`

**Closing Window (8 features)**
26. `closing_3s_density`
27. `closing_3s_has_cta`
28. `closing_3s_cta_type`
29. `closing_3s_gesture_present`
30. `closing_3s_text_count`
31. `closing_3s_emotion`
32. `closing_3s_face_visible`
33. `closing_effectiveness_score`

**Metadata (2 features)**
34. `middle_is_present`
35. `middle_len_sec`

**Total: 35 Core Temporal Features** (with additional features for longer videos)

---

## Architectural Decision: Multimodal Synchronization for MVP

### The Challenge

During the RumiAI ML MVP design, we faced a critical decision about capturing multimodal relationships (text overlays, speech, and gestures occurring together). This decision impacts how ML models learn cross-modal patterns that may drive viral engagement.

### The Options Considered

**Option 1: Window-Level Counts (Selected for MVP)**
- Track counts of each modality per temporal window
- Example: `hook_text_count: 4`, `hook_gesture_count: 3`
- ML discovers: Co-occurrence patterns within windows
- Limitation: Cannot detect precise timing alignment

**Option 2: Synchronization Metrics (Deferred to Phase 2)**
- Calculate timing distances between modalities
- Example: `text_gesture_sync_rate: 0.75` (% within 0.5s)
- ML discovers: Quality of coordination, not just co-occurrence
- Complexity: O(n×m) distance calculations, normalization challenges

### Why We Chose Counts-Only for MVP

1. **Video Length Compatibility**: Our temporal windows handle videos from 7-120 seconds with variable middle sections. Counts naturally scale, while sync metrics require complex normalization across different durations.

2. **Implementation Simplicity**: Counting occurrences is O(n) with low risk. Calculating pairwise distances is O(n×m) with edge cases around window boundaries and missing modalities.

3. **80/20 Rule**: Window-level co-occurrence captures ~70% of the multimodal signal. If text and gestures both appear in the hook's 3 seconds, they're likely coordinated enough for engagement.

4. **ML Discovery Path**: Even without sync metrics, ML can learn:
   - Hook windows need text+gesture combinations
   - Closing windows benefit from speech+text CTAs
   - Middle sections vary by content type

### Implications for Feature Engineering

This architectural decision led to removing several pre-computed synchronization features:
- `multimodalMoments` → Removed (pre-computed alignments)
- `overlayGestureSync` → Removed (requires timing precision)
- `overlaySpeechAlignment` → Removed (requires timing precision)
- `crossModalCoherence` → Removed (average of sync metrics)

Instead, we rely on temporal window counts that ML combines to discover patterns.

### Future Evolution Path

If MVP models show multimodal co-occurrence strongly predicts engagement, Phase 2 can add:
- Sub-second timing arrays within windows
- Statistical sync metrics (avg distance, sync rate)
- Lead/lag relationships between modalities

The temporal window architecture supports this evolution without breaking existing features.

---

## 4. Duration Buckets

### Defined Buckets
- **0-15s**: Short form
- **16-30s**: Medium
- **31-60s**: Long
- **61-120s**: Extra long

### Do We Still Need Buckets?
With the hybrid temporal features (proportional bins + hook window), **duration buckets may no longer be necessary** for the canonical JSON. The universal temporal features work across all durations:

- **Hook Window**: Same 0-3s window for all videos (user behavior is duration-agnostic)
- **Proportional Bins**: Scale to any video length
- **Normalized Positions**: Already in 0-1 range

### When to Consider Buckets:
1. **If models show different feature importance by duration** (hook matters more in 15s)
2. **If training separate specialist models** (RF_15s vs RF_60s)
3. **If clustering patterns differ significantly** by duration

### Recommendation:
Start without duration-specific JSONs. Include `duration_bucket` as a feature in the canonical JSON. Only split if performance analysis shows significant benefit.

---

## 5. Feature Engineering Decisions

### Feature Namespacing Convention
```
cd_ = creative_density
ej_ = emotional_journey
pf_ = person_framing
sp_ = scene_pacing
sa_ = speech_analysis
vo_ = visual_overlay_analysis
md_ = metadata_analysis
```

Benefits:
- Easy feature selection: `df.filter(regex='^cd_')`
- Clear ownership and source
- Prevents naming collisions

### Features Requiring Transformation

#### For Random Forest
- **Categorical**: One-hot encode → `accelerationPattern` becomes 4 binary features
- **Arrays**: Extract statistics → `densityCurve` becomes mean, std, trend
- **Dicts**: Flatten to features → `elementCounts` becomes 6 numerical features

#### For K-means
- **Categorical**: Label encode + scale → `accelerationPattern` becomes 0-3 scaled
- **Arrays**: Extract statistics + scale
- **Complex structures**: Often excluded (multiModalPeaks, peakMoments)
- **All numerical**: Scale to [0,1] range

### Features to Exclude
- `densityProgression`: Hardcoded to "stable" - no variation
- For K-means: Complex nested structures that don't translate to meaningful distances

---

## 6. Implementation Plan

### Phase 1: Schema Definition
- [ ] Conduct feature audit to eliminate redundancies
- [ ] Lock feature catalog with TBC features (after audit)
- [ ] Define canonical JSON schema
- [ ] Create validation rules

### Phase 2: ETL Pipeline
- [ ] Implement canonical JSON writer
- [ ] Build proportional bin calculator
- [ ] Create RF artifact generator
- [ ] Create K-means artifact generator with scaling

### Phase 3: Validation
- [ ] CI/CD schema validation
- [ ] Unit tests for edge cases (very short/long videos)
- [ ] Feature count assertions (must equal TBC after audit)

### Phase 4: Model Training
- [ ] Train RF models per duration bucket
- [ ] Train K-means models per duration bucket
- [ ] Persist scalers and PCA models

---

## 7. Key Decisions Made

### ✅ Accepted
1. **Single canonical JSON** per video (not multiple base files)
2. **Separate RF and K-means artifacts** (different preprocessing needs)
3. **Proportional temporal bins** for duration-agnostic comparison
4. **Feature namespacing** with prefixes
5. **Feature count TBC** after redundancy audit
6. **4 duration buckets** as metadata (may not need separate models)
7. **Hybrid temporal approach** (proportional + hook/closing windows + absolute windows)
8. **Hook Window** as primary temporal feature class (8 features for 0-3s window)
9. **Closing Window** as conversion temporal feature class (8 features for last 3s)

### ❌ Rejected
1. ~~Separate base and temporal JSONs~~ (causes schema drift)
2. ~~Fixed temporal bins for all durations~~ (doesn't scale)
3. ~~Same feature set for RF and K-means~~ (suboptimal performance)
4. ~~Comma-separated format~~ (use pipe `|` delimited for sheets)

---

## 8. Data Flow

```
Video → ML Services → Timelines → Precompute Functions
                                           ↓
                                   Canonical JSON
                                           ↓
                        ┌─────────────────┼─────────────────┐
                        ↓                                     ↓
                  ETL for RF                            ETL for K-means
                        ↓                                     ↓
              RF Artifacts (one-hot)              KM Artifacts (scaled)
                        ↓                                     ↓
                  RF Training                          K-means Training
```

---

## 9. Example Feature Counts

### Creative Density Analysis
- **Total Features**: 25
- **RF Adaptable**: 24/25 (96%)
- **K-means Adaptable**: 20/25 (80%)
- **Excluded**: densityProgression (hardcoded)

### By Data Type Success Rate
- **Numerical (float/int)**: 100% adaptable for both
- **Categorical (string)**: 100% for RF, 83% for K-means
- **Dict (fixed structure)**: 100% adaptable for both
- **Array-variable**: 100% for RF, 40% for K-means

---

## 10. Storage Layout

```
canonical/
  date=20250826/
    duration_bucket=0to15s/
      part-00000.parquet
    duration_bucket=16to30s/
      part-00000.parquet

artifacts/
  model=rf/
    bucket=0to15s/
      date=20250826/
        train_rf_0to15s.parquet
  model=km/
    bucket=31to60s/
      date=20250826/
        train_km_31to60s.parquet
```

---

## 11. Next Steps

1. **Conduct feature audit** to eliminate redundancies and validate ML compatibility
2. **Create feature catalog** with all TBC features defined
2. **Build ETL pipeline** for canonical JSON generation
3. **Implement temporal bin calculator** for proportional features
4. **Create model-specific transformers** for RF and K-means
5. **Set up validation framework** for schema compliance

---

## Appendix: Temporal Feature Enhancement Details

For comprehensive discussion of the temporal event-centric approach, including:
- Detailed rationale for 5 key implementation decisions
- GPT's 3 gaps analysis and our solutions
- ML compatibility considerations
- Alternative approaches considered

See: **[TemporalFeatures2708.md](./TemporalFeatures2708.md)**

### Key Decisions Summary
1. **Variable event counts by duration** (1-2 for 15s up to 5 for 120s)
2. **Fixed feature slots with nulls** for ML compatibility
3. **Event type tracking with label encoding** (0-8 categorical)
4. **No semantic labels** - leave interpretation to Claude
5. **Dual-layer architecture** - temporal_events array + fixed ML features

### Feature Audit Notes
Prior to finalizing feature count, we must:
- Remove interpreted/semantic features unsuitable for ML
- Eliminate redundant features across flows
- Validate Random Forest and K-means compatibility
- Integrate ~30 new temporal event features

Expected final count: ~150-190 features (TBC after audit)

---

## References

- creative_densityMLA.md - Feature adaptability analysis
- TemporalFeatures2708.md - Event-centric temporal analysis design
- Original RumiAI architecture documents
- GPT-4 consultation on temporal analysis critique

---

*Document created: 2025-08-26*
*Last updated: 2025-08-27*