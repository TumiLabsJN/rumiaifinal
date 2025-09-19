# Temporal Features Enhancement Discussion - 2024-08-27

## Goal
Address GPT's critique: "temporal analysis must not collapse the middle narrative arc into something too coarse. The insights between hook and CTA are often what make or break a viral TikTok (reveals, twists, pacing shifts, 'second hooks')."

## Key Questions & Decisions

### 1. How many events to track? ✅ ANSWERED: Variable by Duration

**Decision**: Variable events based on video duration, aligning with adaptive granularity philosophy.

**Event Count by Duration Bucket:**
```
0-15s:   1-2 events max (one main reveal/peak)
16-30s:  2-3 events (hook → reveal → reinforcement)
31-60s:  3-4 events (multiple narrative beats)
61-120s: 4-5 events (complex narrative with multiple hooks)
```

**Rationale:**
- Shorter videos naturally have fewer meaningful peaks
- Longer videos have room for complex narratives with multiple hooks
- Matches real content patterns:
  - 15s video: Usually ONE main moment (outfit reveal, punchline)
  - 60s video: Setup → twist → second hook → climax → callback

**Implementation Approach:**
```json
"middle_window": {
  "event_count": 2,  // For a 20s video
  "event_1": {...},  // Populated
  "event_2": {...},  // Populated  
  "event_3": null,   // Empty for this duration
  "event_4": null,   // Empty for this duration
  "event_5": null    // Empty for this duration
}
```

**Benefits:**
- Variable richness based on content length
- Predictable schema structure (always max 5 slots)
- ML models can use event_count to know how many are valid
- No feature explosion for short videos

---

### 2. Should events be in an array or fixed features? ✅ ANSWERED: Fixed Features

**Decision**: Use fixed feature slots (e.g., `event_1_pos`, `event_1_type`, `event_1_mag`) rather than variable arrays.

**Critical Context - We Have 174+ ML Features Already**:
- Hand-counted ~235 total features across all flows
- After removing interpreted features: ~174 ML-ready features
- Adding event tracking: 15 additional features (5 events × 3 properties each)
- Total increase: Only ~8.6% more features (174 → 189)

**Why Fixed Features Over Arrays**:

1. **Direct ML Compatibility**
   - Random Forest requires fixed-width feature vectors
   - Each feature becomes a column in the training matrix
   - No intermediate transformation needed
   ```
   | video_id | event_1_pos | event_1_type | event_1_mag | event_2_pos | ...
   |----------|------------|--------------|-------------|-------------|---
   | vid_001  | 0.25       | density      | 0.89        | 0.58        | ...
   | vid_002  | 0.33       | emotion      | 0.92        | null        | ...
   ```

2. **Consistency with Existing Architecture**
   - Current RumiAI features already use fixed structure
   - Example: `elementCounts` is a dict with 6 fixed keys, not a variable array
   - Maintains pattern consistency across the 174+ existing features

3. **Arrays Would Require Transformation Anyway**
   ```python
   # Array approach would need flattening:
   events = [{"pos": 0.25, "type": "density", "mag": 0.89}, ...]
   # Must transform to:
   features["event_1_pos"] = events[0]["pos"] if len(events) > 0 else None
   features["event_1_type"] = events[0]["type"] if len(events) > 0 else None
   # So why not start with fixed features?
   ```

**CRITICAL CLARIFICATION - This Does NOT Replace Curve Analysis**:

We maintain a **two-layer approach** that preserves all temporal richness:

**Layer 1: Continuous Curve Analysis (EXISTING - UNCHANGED)**
```json
{
  "densityCurve": [20, 25, 30, 45, 60, 55, 40, 35, 30, 28, 25, 22],
  "emotionProgression": ["neutral", "happy", "excited", "satisfied"],
  "shape": {
    "trend_slope": -0.15,
    "variance": 12.3,
    "oscillations": 2,
    "peak_position": 0.33
  },
  "bins": {
    "early_density": 35,
    "mid_density": 62,
    "late_density": 41
  },
  "piecewise": {
    "slope_early": 2.1,
    "slope_mid": 0.2,
    "slope_late": -1.8
  }
}
```

**Layer 2: Discrete Event Extraction (NEW ADDITION)**
```json
{
  "event_1_pos": 0.33,      // Position of first key moment
  "event_1_type": "density", // What type of peak
  "event_1_mag": 60,        // Magnitude of the peak
  "event_2_pos": 0.75,      // Second key moment
  "event_2_type": "emotion",
  "event_2_mag": 0.92
}
```

**Why Both Layers Are Essential**:
- **Curves** capture the overall narrative flow, pacing, trends
- **Events** identify specific critical moments (reveals, twists, "second hooks")
- Together they provide both continuous and discrete temporal understanding

**Analogy for GPT**: 
Think of it like analyzing a movie:
- Layer 1 (Curves) = The overall tension graph, pacing rhythm, emotional arc
- Layer 2 (Events) = Specific timestamps of jump scares, plot twists, key reveals
- You need both to fully understand the viewing experience

**Implementation Structure**:
```json
"middle_window": {
  // EXISTING CURVE ANALYSIS (unchanged)
  "shape": { /* 6 features */ },
  "bins": { /* 3 features */ },
  "piecewise": { /* 5 features for 30s+ */ },
  "rhythm": { /* 3 features for 60s+ */ },
  
  // NEW EVENT EXTRACTION (additional)
  "event_count": 3,
  "event_1_pos": 0.25,
  "event_1_type": "density",
  "event_1_mag": 0.89,
  "event_2_pos": 0.58,
  "event_2_type": "emotion", 
  "event_2_mag": 0.92,
  "event_3_pos": 0.83,
  "event_3_type": "motion",
  "event_3_mag": 0.76,
  "event_4_pos": null,  // Unused for this video
  "event_4_type": null,
  "event_4_mag": null,
  "event_5_pos": null,
  "event_5_type": null,
  "event_5_mag": null
}
```

**Feature Count Summary**:
- Existing ML features: ~174
- New event features: 16 (5 events × 3 properties + 1 count)
- Total: ~190 features
- Increase: Only 9.2%

This approach maximizes temporal insight while maintaining ML compatibility and schema consistency.

### 3. How to handle videos with fewer peaks than slots? ✅ ANSWERED: Null Values + Event Count

**Decision**: Use null values for empty event slots, paired with an explicit `event_count` feature.

**The Problem**:
- We have 5 event slots maximum
- A 15s video might only have 1 meaningful peak
- What do we put in event slots 2-5?

**Options Considered**:

1. **Null/Empty Values (RECOMMENDED)**
   ```json
   "event_count": 1,
   "event_1_pos": 0.58,
   "event_1_type": "density",
   "event_1_mag": 0.89,
   "event_2_pos": null,
   "event_2_type": null,
   "event_2_mag": null
   ```
   - ✅ Random Forest handles nulls naturally (treats as missing)
   - ✅ Clear semantic meaning: "no event here"
   - ✅ Standard data science practice
   - ⚠️ K-means needs preprocessing (exclude nulls from distance calc)

2. **Zero Padding**
   ```json
   "event_2_pos": 0.0,
   "event_2_type": "none",
   "event_2_mag": 0.0
   ```
   - ❌ Dangerous! 0.0 position = "event at video start"
   - ❌ Creates false patterns
   - ❌ ML might learn incorrect associations

3. **Repeat Last Valid Event**
   ```json
   "event_2_pos": 0.58,  // Same as event 1
   "event_2_type": "density",
   "event_2_mag": 0.89
   ```
   - ❌ Creates false importance weighting
   - ❌ Can't distinguish "1 event" from "multiple events same location"
   - ❌ Distorts statistical patterns

4. **Sentinel Values (-1)**
   ```json
   "event_2_pos": -1,
   "event_2_type": "none",
   "event_2_mag": -1
   ```
   - ✅ Clear distinction from real values
   - ✅ Works with all ML algorithms
   - ⚠️ Requires preprocessing and documentation
   - ⚠️ Less standard than nulls

5. **Hierarchical Features (exists flags)**
   ```json
   "event_2_exists": 0,
   "event_2_pos": null,
   "event_2_type": null
   ```
   - ✅ Explicit presence signaling
   - ❌ Doubles feature count (adds 5 "exists" flags)
   - ❌ More complex schema

**Why Null + Count is Optimal**:

1. **Explicit Signal**: `event_count` tells ML exactly how many real events exist
   - Becomes a valuable feature itself
   - "Videos with 3 events have 2x engagement vs 1 event"
   - Clear pattern learning: "optimal event count for duration"

2. **Standard Practice**: Nulls are the data science standard for "missing/not applicable"
   - Every ML practitioner understands nulls
   - Built-in library support
   - No ambiguity in interpretation

3. **Algorithm Compatibility**:
   - **Random Forest**: Handles nulls natively (splits ignore missing)
   - **K-means**: Simple preprocessing - exclude null features from distance
   - **Deep Learning**: Standard masking layers handle nulls

4. **Clean Semantics**: 
   - null = "no event at this position"
   - Not confused with 0 (start position) or -1 (artificial)
   - Maintains data integrity

5. **Feature Value of Count**:
   ```python
   # ML can learn patterns like:
   if duration <= 15 and event_count > 2:
       prediction = "overwhelming/chaotic"
   elif duration >= 60 and event_count < 2:
       prediction = "monotonous/boring"
   elif event_count == 3:
       prediction = "optimal pacing"
   ```

**Implementation Example**:
```json
// 15s video with 1 peak
{
  "event_count": 1,
  "event_1_pos": 0.67,
  "event_1_type": "density",
  "event_1_mag": 0.92,
  "event_2_pos": null,
  "event_2_type": null,
  "event_2_mag": null,
  // ... events 3-5 all null
}

// 60s video with 4 peaks
{
  "event_count": 4,
  "event_1_pos": 0.15,
  "event_1_type": "emotion",
  "event_1_mag": 0.75,
  "event_2_pos": 0.42,
  "event_2_type": "density",
  "event_2_mag": 0.89,
  // ... all 4 events populated
  "event_5_pos": null,  // Only this one empty
  "event_5_type": null,
  "event_5_mag": null
}
```

### 4. Should we track event types (density/emotion/motion)? ✅ ANSWERED: Yes, with Label Encoding

**Decision**: Track event types using label encoding to identify which flow/analysis generated each peak.

**Critical Understanding - Event Types are Composite Signals**:
Each "event type" represents a moment when multiple related features within a flow combine to create a significant peak:

- **Flow**: `emotional_journey` (the analysis module)
- **Event Type**: `emotion` (a peak moment detected)
- **Contributing Features**: 
  - `emotionalIntensity` = 0.92
  - `uniqueEmotions` = 4
  - `emotionTransitions` = rapid
  - `gestureEmotionAlignment` = high
  - All spike together → "emotion peak detected"

**Analogy**: 
- Features = Individual instruments in orchestra
- Peak/Event = Crescendo moment when instruments play together
- Event Type = Which section led (strings, brass, percussion)

**Available Event Types from Our Flows**:
- `density` - Visual element density peaks (from creative_density)
- `emotion` - Emotional intensity peaks (from emotional_journey)
- `motion` - High motion/action peaks (from scene_pacing)
- `text_overlay` - Text density peaks (from visual_overlay_analysis)
- `face_close` - Close-up face moments (from person_framing)
- `speech_emphasis` - Vocal emphasis peaks (from speech_analysis)
- `scene_change` - Scene transition clusters (from scene_pacing)
- `gesture` - Significant gesture peaks (from person_framing)

**Options Considered**:

1. **Track Event Types with Categories (RECOMMENDED)**
   ```json
   "event_1_type": "emotion",
   "event_2_type": "density",
   "event_3_type": "text_overlay"
   ```
   - ✅ ML learns which types matter most
   - ✅ Identifies multi-modal reinforcement
   - ✅ Enables pattern learning: "emotion early + density middle = viral"
   - ⚠️ Needs encoding for ML algorithms

2. **No Types - Position + Magnitude Only**
   ```json
   "event_1_pos": 0.58,
   "event_1_mag": 0.92
   ```
   - ✅ Simpler schema
   - ❌ Loses critical information about WHAT happened
   - ❌ Can't distinguish emotional climax from visual burst
   - ❌ ML blind to flow-specific patterns

3. **Binary Multi-Flow Flags**
   ```json
   "event_1_is_emotion": 1,
   "event_1_is_density": 0,
   "event_1_is_motion": 0
   ```
   - ✅ Direct ML compatibility
   - ❌ Feature explosion: 5 events × 8 types = 40 extra features
   - ❌ Overly complex for the value gained

**Why Track Event Types is Critical**:

1. **Content Style Recognition**:
   - Tutorials: `text_overlay` + `speech_emphasis` peaks
   - Comedy: `emotion` + `motion` peaks  
   - Fashion: `density` + `face_close` peaks
   - ML learns these patterns automatically

2. **Multi-Modal Power Detection**:
   ```json
   // When emotion and density peaks align:
   "event_1_pos": 0.58, "event_1_type": "emotion",
   "event_2_pos": 0.58, "event_2_type": "density"
   // ML learns: "Synchronized peaks = viral moment"
   ```

3. **Duration-Specific Importance**:
   - 15s videos: `density` peaks matter most
   - 60s videos: `emotion` journey critical
   - ML discovers these without hardcoding

**Recommended Implementation - Label Encoding**:
```json
{
  "event_1_pos": 0.58,
  "event_1_type": 3,        // 3 = emotion
  "event_1_mag": 0.92,
  
  "event_2_pos": 0.58,  
  "event_2_type": 1,        // 1 = density (same time = reinforcement!)
  "event_2_mag": 0.89
}

// Encoding Map:
// 0 = none/null
// 1 = density
// 2 = motion
// 3 = emotion
// 4 = text_overlay
// 5 = face_close
// 6 = speech_emphasis
// 7 = scene_change
// 8 = gesture
```

**Why Label Encoding Over One-Hot**:
- Keeps feature count low (1 integer vs 8 binary features)
- Random Forest handles ordinal data well
- Can always one-hot encode later if needed
- Simplifies null handling (type = 0)

**Example Showing Value**:
```python
# A viral reveal moment might generate:
events = [
  {"pos": 0.58, "type": "emotion", "mag": 0.92},     # Surprise
  {"pos": 0.58, "type": "density", "mag": 0.89},     # Visual burst
  {"pos": 0.59, "type": "motion", "mag": 0.85},      # Camera move
  {"pos": 0.60, "type": "text_overlay", "mag": 0.78} # "WOW!" text
]

# ML learns: Multiple types clustering = major moment
# Single type isolation = less impactful
```

**Implementation Note**: 
During peak detection, we scan all flows, identify peaks above threshold in each, rank by magnitude, and select top 5 across all flows while ensuring diversity (max 3 from same flow type).

### 5. Do we need semantic labels (reveal/twist/hook)? ✅ ANSWERED: No, Leave to Claude

**Decision**: Skip semantic labels in ML features. Let Claude handle human interpretation during report generation.

**The Key Insight - Different Layers, Different Purposes**:

**ML Layer (Random Forest/K-means)**:
- Needs: Objective, numerical patterns
- Gets: Position (0.67) + Type (3=emotion) + Magnitude (0.92)
- Learns: "Events at 0.6-0.7 position with high magnitude correlate with engagement"

**Claude Layer (Report Generation)**:
- Receives: ML-discovered patterns
- Adds: Human semantic meaning
- Outputs: "The video builds to a reveal at the 40-second mark, a classic fashion content strategy"

**Why NOT to Add Semantic Labels for ML**:

1. **ML Doesn't Need Words**:
   - ML learns from numerical patterns, not human concepts
   - Position + type + magnitude already contains all signal needed
   - Model discovers "emotion peak at 0.67 = effective" without us labeling it "reveal"

2. **Avoid Human Bias**:
   - Let data tell us what patterns matter
   - Don't impose our assumptions about what constitutes a "hook" or "twist"
   - What we think is a "reveal" might not be what drives engagement

3. **Labeling Complexity**:
   - Would require manual rules or human annotation
   - Same event could be "reveal" in one context, "reinforcement" in another
   - Subjective and error-prone

4. **Feature Economy**:
   - Already adding 16 features for events (5 events × 3 properties + count)
   - Another categorical feature means more encoding
   - Diminishing returns on feature addition

5. **Clean Separation of Concerns**:
   - ML: Pattern discovery (objective)
   - Claude: Pattern interpretation (subjective)
   - Don't mix the layers

**The Optimal Flow**:
```
Raw Data → Feature Extraction → ML Training → Claude Interpretation
         ↑                     ↑             ↑
         No labels needed      Learns patterns  Adds semantic meaning
         Objective metrics     Statistical      "This is a reveal"
```

**Example Showing the Separation**:

**ML Sees**:
```json
{
  "event_1_pos": 0.25,
  "event_1_type": 3,  // emotion
  "event_1_mag": 0.87,
  
  "event_2_pos": 0.67,
  "event_2_type": 1,  // density
  "event_2_mag": 0.92,
  
  "event_3_pos": 0.67,
  "event_3_type": 3,  // emotion
  "event_3_mag": 0.89
}
```

**ML Learns** (without labels):
- Events at 0.67 with type=1 & type=3 together → high engagement
- This pattern is 3x more common in viral videos
- Confidence: 0.87

**Claude Interprets** (adds semantic meaning):
```
"The video employs a 'curiosity-to-reveal' structure with an emotional 
hook at 15 seconds (25% through middle), building to a synchronized 
visual and emotional climax at the 40-second mark. This dual-peak 
moment (density + emotion) creates the 'wow factor' that drives shares."
```

**Why This Separation Works**:
- ML stays objective and data-driven
- Claude provides the narrative that humans understand
- No contamination between statistical learning and human interpretation
- Each layer does what it's best at

**Implementation Note**: 
During feature extraction, we simply record position, type, and magnitude. No attempt to categorize as "hook", "reveal", or "twist". Claude receives the ML model's findings (feature importance, peak patterns) and crafts the human narrative.

---

## GPT's Feedback & Remaining Gaps

### ✅ Strong Improvements Acknowledged by GPT
1. **Event-Centric Middle**: Added timestamped, event-level markers (exactly what critique asked for)
2. **Contextualization of Peaks**: Features now tie peaks to creative elements (answers "why" not just "when")
3. **Multi-Scale Middle Windows**: Both micro-peaks and macro arcs captured
4. **Schema Flexibility**: Fixed-schema for ML stability + event arrays for richer analysis

### ⚠️ Gap 1: Redundancy & Noise ✅ SOLUTION: Feature Audit Process

**GPT's Concern**: 
- Multiple overlapping measures (e.g., `climaxMoment` in multiple domains, `stabilityScore` under different labels)
- Risk of feature bloat → training noise → weaker interpretability
- ~235 total features with likely redundancies

**Our Solution Strategy**:
After finalizing this temporal enhancement plan, we will conduct a systematic feature audit:

1. **Cross-Reference Against Temporal Plan**: 
   - Review all 235 features against the event-centric approach defined here
   - Identify which existing features are superseded by new event tracking
   - Example: If we track `emotion` peaks with position/magnitude, do we still need separate `emotionalPeaks`, `climaxMoment`, AND `peakEmotionMoments`?

2. **Redundancy Elimination Matrix**:
   ```
   Feature A            | Feature B              | Action
   --------------------|------------------------|--------
   climaxMoment        | event_1 (highest mag)  | Remove climaxMoment
   emotionalPeaks      | event_type="emotion"   | Remove emotionalPeaks
   stabilityScore      | transitionSmoothness   | Keep one, remove other
   ```

3. **Feature Selection Criteria**:
   - **Keep**: Features that provide unique signal not captured by events
   - **Remove**: Features that duplicate event information
   - **Merge**: Similar features that can be combined
   - **Transform**: Array features that should become event entries

4. **Expected Outcome**:
   - Reduce from ~235 to ~150-180 features
   - Each feature has clear, unique purpose
   - No overlapping temporal measures
   - Clean separation between continuous (curves) and discrete (events) features

**Implementation Note**: 
This audit will be conducted AFTER this temporal plan is finalized, using the event architecture as the reference standard. Each of the 235 features will be evaluated for redundancy with the new temporal event system.

### ⚠️ Gap 2: Temporal Anchoring ✅ SOLUTION: Dual-Layer with Bridge

**GPT's Concern**:
- Events scattered across categories without unified temporal index
- Peaks "live in silos" making cross-modal pattern discovery harder
- No way to see when emotion peaks align with density peaks

**Our Solution (Aligned with GPT's Recommendation)**:
Implement a dual-layer approach with a deterministic bridge between them:

**Layer 1: Temporal Events Array (Source of Truth for Cross-Modal Narrative)**
```json
"temporal_events": [
  {"time": 0.24, "type": "emotion_peak", "subtype": "surprise", "intensity": 0.88},
  {"time": 0.24, "type": "density_peak", "subtype": "visual_burst", "intensity": 0.91},
  {"time": 0.57, "type": "motion_peak", "subtype": "camera_zoom", "intensity": 0.83}
]
```
- Chronologically ordered
- All flows unified on single timeline
- Perfect for human analysis and Claude interpretation

**Layer 2: Fixed Features for ML (Deterministic Transformation)**

**Selection Policy**:
1. Rank all events by: intensity → novelty → recency
2. Keep top K per flow (K=2) and top K global (K=3)
3. Apply alignment tolerance: max(3% of duration, 300ms)
4. If events fall within tolerance, keep higher intensity and record tie

**Fixed Feature Structure**:
```python
# Per-flow events (2 max per flow)
"emotion_event_1_time": 0.24,
"emotion_event_1_mag": 0.88,
"emotion_event_2_time": null,  # Only 1 emotion event
"emotion_event_2_mag": null,

"density_event_1_time": 0.24,
"density_event_1_mag": 0.91,
# ... for each flow type

# Global top 3 events
"g1_time": 0.24,
"g1_mag": 0.91,
"g1_type_id": 4,  # density_peak

"g2_time": 0.57,
"g2_mag": 0.83,
"g2_type_id": 6,  # motion_peak

"g3_time": null,  # Only 2 events in this example
"g3_mag": null,
"g3_type_id": 0,

# Distance metrics
"g12_distance": 0.33,           # Distance between event 1 and 2
"g23_distance": null,            # No event 3
"g13_distance": null,
"hook_to_first_peak": 0.06,     # 0.24 - 0.18 (hook ends at 3s = 0.18 in normalized time)
"last_peak_to_cta": 0.18,       # 0.75 (CTA starts) - 0.57
"global_peak_spread": 0.33,     # max_time - min_time
"global_cluster_score": 0.72,   # 1 / mean_nearest_neighbor_distance

# Alignment metrics
"g1_flows_aligned_count": 2,    # emotion + density at same time
"g2_flows_aligned_count": 1,    # only motion
"g3_flows_aligned_count": 0,

# Optional binary flags for key pairs
"emotion_density_aligned": 1,   # Both peaked at 0.24
"emotion_motion_aligned": 0,
"density_motion_aligned": 0
```

**Encoding Rules**:
- **Types/Subtypes**: Integer dictionary with lookup JSON in repo
- **K-means View**: Drop categoricals, standardize numerics with robust scaling
- **Random Forest View**: Keep categoricals as one-hot, raw/min-max scale distances

**Unit Test Examples**:
- Single middle peak → g1_time near middle, low spread
- Three spaced peaks → monotonic g1 < g2 < g3, high spread  
- Strong multi-modal moments → flows_aligned_count ≥ 2

**Why This Solution Works**:
1. **Preserves Narrative Fidelity**: temporal_events array captures full story
2. **ML-Compatible**: Fixed features with deterministic selection
3. **Cross-Modal Discovery**: Distance and alignment metrics reveal patterns
4. **Minimal Feature Explosion**: Compact representation (~30 new features)

**Implementation Priority**:
The temporal_events array becomes the canonical representation. Fixed ML features are derived views materialized during training.

### ⚠️ Gap 3: Model Adaptation ✅ ADDRESSED: Same Root Cause as Gap 1

**GPT's Concern**:
- "For event arrays, you only specify 'extract count + avg + max.' This still collapses temporal richness back into summary stats for the ML models"
- Current approach loses temporal information by reducing arrays to summaries

**Our Analysis**:
This gap is fundamentally the same issue as Gap 1 (Redundancy & Noise). Both stem from our current practice of extracting multiple summary statistics from temporal arrays:

**The Problem Pattern**:
```python
# Current approach that creates both redundancy AND loses temporal info
emotionalPeaks = [0.3, 0.7, 0.9]  
→ peak_count: 3
→ peak_avg: 0.63
→ peak_max: 0.9
→ peak_std: 0.25
# Lost: WHERE these peaks occurred, their relationships, their context
```

**Our Solution**:
The new event-centric fixed features (from Gap 2) eliminate this problem entirely:
```python
# New approach preserves temporal information without redundancy
"event_1_pos": 0.25,  # WHERE it happened
"event_1_type": 3,     # WHAT happened  
"event_1_mag": 0.92,  # HOW intense
"event_2_pos": 0.58,  # Preserves sequence
# No redundant summary stats, actual temporal data preserved
```

**Implementation Note**:
This will be resolved automatically during our feature audit (Gap 1). As we eliminate redundant summary statistics and replace them with our event-centric approach, we simultaneously:
1. Remove redundancy (Gap 1)
2. Preserve temporal richness (Gap 3)

Both gaps have the same root cause and the same solution: moving from summary statistics to explicit temporal event tracking.

---

## Potential Solutions
*[To be populated after all gaps addressed]*

---

## Final Recommendation
*[To be determined]*