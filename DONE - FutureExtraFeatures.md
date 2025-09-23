# Future Extra Features - Not Selected for MVP
**Created**: 2025-01-28
**Purpose**: Document features not selected for ML MVP but potentially valuable in future iterations

## Phase 1 Improvements - NEW [03/09 onwards]

### Description
These initial improvements were identified after the realization and understanding of our:
1. Temporal window architecture (Hook/Middle/Closing with piecewise segments)
2. ML vs Interpreted Features importance (raw metrics over semantic interpretations)

These improvements would enrich the Phase 1 ML project by adding valuable raw metrics that follow our architectural principles.

### Object Focus Metrics

| Priority | Category | Improvement | Difficulty | Time Est | Explanation of Importance | Technical Debt Resolved | Dependencies |
|----------|----------|-------------|------------|----------|---------------------------|------------------------|--------------|
| P3 | Raw Data | Object Screen Time Metrics | Medium | Medium | Captures content focus duration and persistence patterns | Complements dominant_object with temporal depth | Enhanced Object Metrics (P2) |

**Details**:
- **What it adds**: `dominant_object_time`, `dominant_object_ratio`, `object_switch_rate`
- **Why valuable**: Reveals content strategy (quick product shots vs sustained focus)
- **Why not P1/P2**: Dominant/secondary objects already capture main content type signals
- **Implementation complexity**: Requires tracking object persistence across frames within windows




## Overview [OLD]

### What This Document Contains
- Features that are incompatible with RF and/or K-means algorithms
- Features can appear here even if they're in FeaturesMLMVP.md (if rejected by one algorithm but accepted by another)
- Documents specific rejection reasons for each algorithm
- Tracks what would need to change for future inclusion

### Key Points
- **"N/A" in Reason for Removal column** = That algorithm actually accepts this feature (it's in FeaturesMLMVP.md)
- **Both columns have removal reasons** = Feature is completely rejected (not in FeaturesMLMVP.md)
- **This is NOT just for completely rejected features** - it documents ALL algorithm incompatibilities

### Examples
- Feature works for RF but not K-means: Appears in BOTH FeaturesMLMVP.md (kept) and here (KM rejection documented)
- Feature works for neither: Appears ONLY here with both rejection reasons

## Column Definitions

**Reason for Removal RF/KM**:
- N/A = Feature included in shortlist for this algorithm
- Incompatible = Not compatible with algorithm
- Module Needed = Requires additional module installation
- Interpreted Feature = Semantic/interpreted feature not suitable for ML
- Repetitive Feature = Duplicates information from other features
- Debugging Work = Not worth debugging for MVP

**Difficulty** (Transformation difficulty to make feature adaptable):
- N/A = Feature cannot be made adaptable (incompatible with algorithm)
- Low = Simple transformation would make it work
- Medium = Moderate transformation complexity required
- High = Complex transformation with significant effort needed

**Future Value**:
- High = High value for deep learning with 1000+ videos
- Med = Medium value for deep learning
- Low = Low value for deep learning  
- None = No value for deep learning

**What Needs to Change**:
- N/A = Incompatible feature with algorithm
- Module Installation = Install additional modules
- Debug = Debug current implementation

---

## Creative Density Features

| Feature | Reason for Removal RF | Reason for Removal KM | Difficulty | Future Value | What Needs to Change | Notes |
|---------|----------------------|----------------------|------------|--------------|---------------------|-------|
| cognitiveLoadCategory | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw density metrics instead | Semantic interpretation of density, ML should discover patterns from raw features |
| densityClassification | Repetitive Feature | Repetitive Feature | N/A | None | Use avgDensity directly | Direct derivative of avgDensity, loses information through categorization |
| densityCurve | Temporal Redundancy | Temporal Redundancy | High | Med | Use MLMVP2 temporal windows | Replaced by temporal windows (hook/middle/closing) and shape features |
| densityProgression | Incompatible | Incompatible | N/A | None | Debug | Hardcoded to "stable" - no variation across videos, zero predictive power |
| densityShifts | Temporal Redundancy | Temporal Redundancy | High | Low | Use MLMVP2 temporal features | Covered by oscillations, variance, piecewise slopes, and volatility |
| dominantCombination | Repetitive Feature | Repetitive Feature | Medium | Low | Use elementCooccurrence | Direct derivative of elementCooccurrence, which contains complete pair data |
| elementsPerSecond | Repetitive Feature | Repetitive Feature | N/A | None | Use avgDensity | Duplicate/variant of avgDensity, both measure elements per second rate |
| mlTags | Debugging Work | Incompatible | High | Low | Define tag taxonomy | Variable text array, high complexity, unclear value, only works for RF with high info loss |
| multiModalPeaks | Debugging Work | Incompatible | High | Med | Simplify structure | Complex nested structure, high info loss, signal partially captured by other features |
| pacingStyle | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw progression metrics | Semantic interpretation, redundant with accelerationPattern and temporal features |
| peakMoments | Temporal Redundancy | Incompatible | High | Med | Use MLMVP2 peak features | Complex structure, covered by peak_value/position and temporal bins |
| structuralFlags | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw metrics | 6 semantic flags derived from avgDensity/stdDev, human-defined thresholds |
| timelineCoverage | Repetitive Feature | Repetitive Feature | N/A | None | Use emptySeconds percentage | Direct inverse of emptySeconds percentage, creates multicollinearity |

---

## Emotional Journey Features

| Feature | Reason for Removal RF | Reason for Removal KM | Difficulty | Future Value | What Needs to Change | Notes |
|---------|----------------------|----------------------|------------|--------------|---------------------|-------|
| audioEmotionAlignment | Module Needed | Module Needed | N/A | Med | Add raw audio pipeline | No raw audio data available, interpreted alignment score without foundation |
| emotionalArc | Interpreted Feature | Incompatible | N/A | Low | Use temporal bins and shape features | Semantic story patterns, captured by MLMVP2 temporal windows |
| emotionalContrastMoments | N/A | Debugging Work | Medium | Med | Simplify extraction | Kept for RF, partial for KM due to high info loss in extraction |
| emotionalPeaks | Temporal Redundancy | Debugging Work | Medium | Low | Use climaxMoment + temporal peaks | Secondary peaks less valuable, covered by climaxMoment and MLMVP2 peaks |
| emotionalTechniques | Interpreted Feature | Incompatible | Medium | Low | Use raw production metrics | Semantic labels of techniques, lacks raw data foundation |
| engagementHooks | Interpreted Feature | Debugging Work | Medium | Low | Use raw engagement metrics | Semantic judgment of "designed to capture", ML should discover hooks |

---


---

## Person Framing Features

| Feature | Reason for Removal RF | Reason for Removal KM | Difficulty | Future Value | What Needs to Change | Notes |
|---------|----------------------|----------------------|------------|--------------|---------------------|-------|
| cinematicStyle | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw framing metrics | Semantic categorization of visual style (documentary, cinematic, handheld). ML should discover styles from raw metrics like face size, stability, framing changes. Subjective human-imposed categories |
| closeUpMoments | Temporal Redundancy | Temporal Redundancy | Medium | Low | Use averageFaceSize + temporal windows | Variable array of close-up timestamps. Creates multicollinearity with temporal windows. ML can discover close-up importance from averageFaceSize + temporal patterns without redundant tracking |
| compositionRule | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw spatial metrics | Semantic categorization of filming techniques (rule of thirds, golden ratio). ML should discover compositional patterns from raw spatial data. Pre-imposed artistic rules limit learning. Subjective interpretation |
| compositionScore | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw metrics | Subjective quality judgment (0-1). Who defines "quality"? Black box calculation. ML should discover what composition drives engagement, not be told human aesthetic standards |
| framingAppropriate | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw framing data | Subjective judgment of whether framing "suits" content. Who decides appropriateness? ML should discover what framing works, not be told pre-imposed rules |
| framingProgression | Temporal Redundancy | Incompatible | High | Low | Use temporal windows | Variable-length array of shot segments with timing. Duplicates framingChanges info. Creates competing temporal segmentation vs our hook/middle/closing windows. High extraction complexity |
| framingTechnique | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw motion/stability metrics | Semantic film technique labels (Dutch angle, POV, tracking). Pre-imposed cinematography categories. ML should discover techniques from raw data. Subjective categorization |
| framingTransitions | Repetitive Feature | Repetitive Feature | N/A | None | Use framingChanges | Exact duplicate of framingChanges - both count shot type transitions. Same information with different name. Already keeping framingChanges |
| groupShots | Incompatible | Incompatible | High | Med | Need temporal-compatible metrics | Variable array of group moments. Can't extract temporal window counts. Total count breaks our window pattern. Need proper multiPersonRate metric instead |
| interactionZones | Interpreted Feature | Incompatible | High | Low | Use raw spatial positions | Semantic proxemics zones (personal/social/public). Academic theory imposed on TikTok. Variable array incompatible with temporal windows. Depends on multi-person tracking |
| keySubjectMoments | Interpreted Feature | Temporal Redundancy | High | Low | Use temporal windows × face metrics | "Important" is subjective judgment. Variable array hard to map to temporal windows. Redundant with averageFaceSize/eyeContactRate in windows. ML should discover importance |
| movementPattern | Debugging Work | Debugging Work | High | Low | Debug or use raw metrics | HARDCODED to "static" - zero variation! Even if fixed, creates semantic categories (static/slow/moderate/dynamic). Better captured by stabilityScore, framingChanges, distanceVariation |
| multiPersonDynamics | Interpreted Feature | Incompatible | High | Med | Need semantic analysis | Complex dict describing HOW people interact (talking/dancing/etc). Semantic interpretation of behaviors. Can't capture with multiPersonRate (quantity ≠ quality). Temporal window incompatible |
| primarySubject | Repetitive Feature | Repetitive Feature | N/A | None | Use multiPersonRate | Derived from face count (single/multiple/none). Oversimplified 3-category bucketing. multiPersonRate provides continuous data. ML learns optimal thresholds from raw metrics |
| professionalLevel | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw quality metrics | Subjective "professional" quality score (0-1). Likely hardcoded to 0.6. Black box calculation. ML should discover quality from stabilityScore, framingConsistency, etc. |
| speakerFraming | Cross-flow Dependency | Cross-flow Dependency | Medium | Low | ML correlation discovery | Conditional feature requiring speech detection. Complex dict with framing during speech. ML can discover speech×framing patterns. Temporal window mapping challenges |
| socialDistance | Conditional Feature | Conditional Feature | Medium | Low | Use multiPersonRate + face sizes | Only meaningful with 2+ people. Undefined for solo content. Temporal averaging problematic with varying person counts. Better captured by multiPersonRate × averageFaceSize |
| visualEngagement | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw framing metrics | Subjective "engaging" judgment. Circular reasoning (using engagement to predict engagement). Likely hardcoded. ML should discover what framing drives engagement |
| subjectCount | Debugging Work | Debugging Work | N/A | None | Replace with multiPersonRate | Currently broken (returns MAX not AVG). Will be replaced by multiPersonRate.avgPersonCount which correctly calculates average people per frame |

---

## Scene Pacing Features

| Feature | Reason for Removal RF | Reason for Removal KM | Difficulty | Future Value | What Needs to Change | Notes |
|---------|----------------------|----------------------|------------|--------------|---------------------|-------|
| accelerationPoints | Temporal Redundancy | Incompatible | High | Low | Use temporal windows | Pre-defined acceleration detection. Variable array incompatible with windows. ML should discover acceleration patterns from raw pace metrics per window. Human-imposed thresholds |
| audioVisualSync | Module Needed | Module Needed | N/A | Med | Add audio pipeline | NO AUDIO ANALYSIS EXISTS. Cannot measure sync without audio data. Likely hardcoded/fake. Would require audio libraries outside zero-cost MVP scope |
| beatMatching | Module Needed | Module Needed | N/A | Med | Add audio pipeline | NO AUDIO EXTRACTION. Beat detection impossible without audio. Certainly fake/hardcoded. More complex than audioVisualSync - needs music analysis |
| climaxTiming | Temporal Redundancy | Temporal Redundancy | Medium | Low | Use temporal windows | Single point (0-1) incompatible with window ranges. Subjective climax identification. ML discovers peak patterns from comparing pacing across hook/middle/closing windows |
| decelerationPoints | Temporal Redundancy | Incompatible | High | Low | Use temporal windows | Same issues as accelerationPoints. Variable array incompatible with windows. Pre-defined deceleration detection. ML discovers slowing patterns from pace metrics per window |
| editingQuality | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw metrics | Subjective "quality" judgment (0-1). Likely hardcoded. Black box calculation. ML should discover what editing quality drives engagement, not be told standards |
| editingStyle | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw pacing metrics | Semantic categories (cut/montage/continuous). Pre-labels before ML analysis. ML discovers patterns, then Claude interprets - not reverse. Likely hardcoded |
| emotionalPacing | Interpreted Feature | Interpreted Feature | N/A | Low | ML correlation discovery | Double interpretation: emotion detection + "matching" judgment. Semantic categories (steady/building/volatile). ML should discover pacing-emotion relationships, not be told |
| engagementPacing | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw pacing metrics | Circular reasoning: using engagement to predict engagement. Subjective "engaging" judgment. Likely hardcoded. ML should discover what pacing engages |
| flowScore | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw transition metrics | Subjective "smoothness" judgment (0-1). Black box calculation. Likely hardcoded. ML should discover what flow patterns work, not be told "smooth" is good |
| narrativeFlow | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw pacing patterns | Double interpretation: identify story + judge support. Semantic categories (linear/circular/episodic). Assumes narrative exists. Literary theory imposed on TikTok |
| overallScore | Interpreted Feature | Interpreted Feature | N/A | Low | Use individual metrics | Black box combination score (0-1). Hides individual signals. Subjective "quality" judgment. ML should determine what metrics matter, not pre-combined scores |
| pacingPattern | Interpreted Feature | Interpreted Feature | N/A | Low | Use variance metrics | Pre-imposed categories (consistent/varied/erratic). Single label loses temporal detail. ML discovers patterns from scene duration variance per window |
| pacingProgression | Temporal Redundancy | Temporal Redundancy | Medium | Low | Use temporal windows | Fixed array with own segmentation (quartiles). Competes with hook/middle/closing windows. Redundant - averageSceneDuration × windows shows progression |
| pacingScore | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw pacing metrics | Subjective "quality" score (0-1). Black box combination. Same as overallScore. ML should determine what pacing matters. Likely hardcoded |
| pacingShifts | Debugging Work | Debugging Work | High | Low | Debug or use variance | NOT IMPLEMENTED - returns empty array. Type mismatch (doc says int, code returns array). No calculation exists. Even if fixed, interpretive "major" threshold |
| rhythmConsistency | Repetitive Feature | Repetitive Feature | N/A | None | Use sceneDurationVariance | Inverse of variance (1-normalized_variance). Semantic wrapper on statistical measure. ML should determine if consistency matters, not be told consistency=good |
| rhythmStructure | Module Needed | Module Needed | N/A | Low | Add audio pipeline | Musical-style rhythm notation. NO AUDIO/MUSIC ANALYSIS. Cannot detect musical rhythm without audio. Would require beat detection |
| sceneRhythm | Interpreted Feature | Interpreted Feature | N/A | Low | Use sceneDurationVariance | Musical categories (regular/syncopated/free) without audio. Subjective pattern labels. Redundant with variance. ML should discover timing patterns |
| temporalFlow | Interpreted Feature | Interpreted Feature | N/A | Low | Use temporal windows | Vague "progression style" (0-1). Single value loses all temporal detail. Against window philosophy. Similar to removed flowScore/narrativeFlow |
| transitionSmoothing | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw cut metrics | Subjective "smoothness" judgment (0-1). Likely hardcoded. ML should discover what transitions work. Pre-imposed aesthetic standards |
| transitionTypes | Incompatible | Partial | High | Med | See Potential Upgrades section | Variable array of transition types. Can't map to temporal windows. Complex detection likely not implemented. Only extracts diversity score |
| viewerRetention | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw pacing metrics | Circular reasoning: using retention to predict retention. Black box "estimation" (0-1). ML should discover pacing-retention relationship. Likely hardcoded |

---

## Potential Upgrades - Explained

**Description**: During initial feature analysis, certain improvements were identified that could enhance ML capabilities but weren't selected for MVP due to complexity or implementation constraints. This section documents these potential upgrades to preserve the analysis and guide future development.

### Transition Type Detection

**1. Importance**

*Pros:*
- Captures editing sophistication (hard cuts vs fades vs dissolves)
- Differentiates professional from amateur editing styles
- Could identify platform-specific transition trends
- Provides granular editing rhythm information beyond simple cut counts

*Cons:*
- TikTok videos predominantly use hard cuts (95%+)
- In-app effects are handled separately from scene transitions
- Limited impact on engagement compared to pacing metrics
- Complex visual analysis required for accurate detection

**2. Potential Implementation Strategy**

Simple approach (hardCutRate vs softTransitionRate):
```python
def calculate_transition_metrics(scene_changes):
    """
    Categorize transitions as hard cuts or soft transitions
    based on frame difference analysis
    """
    hard_cuts = 0
    soft_transitions = 0
    
    for change in scene_changes:
        # Threshold: >80% pixel change = hard cut
        if change['pixel_diff'] > 0.8:
            hard_cuts += 1
        else:
            soft_transitions += 1
    
    return {
        'hardCutRate': hard_cuts / len(scene_changes),
        'softTransitionRate': soft_transitions / len(scene_changes)
    }
```

Advanced approach (specific transition types):
- **fadeCount**: Detect gradual brightness changes to black/white
- **dissolveCount**: Identify gradual blending between scenes
- **wipeCount**: Detect directional transitions
- **hardCutCount**: Sudden frame changes

**3. Integration with Temporal Windows**

Replace variable array with simple counts:
- `hook_hardCutCount`: Number of hard cuts in first 3 seconds
- `middle_fadeCount`: Number of fades in 30-70% duration
- `closing_softTransitionRate`: Percentage of soft transitions in last 10 seconds

This maintains compatibility with our temporal window architecture while providing transition type information.

**4. Potential Implementation Challenges**

a) **Detection Accuracy**
   - Frame differencing alone may misclassify transitions
   - Motion blur can appear as soft transition
   - Solution: Multi-frame analysis around cut points

b) **Computational Cost**
   - Analyzing frames around each cut adds processing time
   - Solution: Sample frames or use existing scene detection data

c) **Platform Effects**
   - TikTok in-app transitions differ from traditional video editing
   - May need platform-specific transition categories
   - Solution: Focus on basic hard/soft distinction initially

d) **Threshold Tuning**
   - Pixel difference thresholds need calibration
   - Different video styles require different thresholds
   - Solution: Use adaptive thresholds based on video characteristics

**Priority**: LOW - While technically feasible, the predominance of hard cuts in TikTok content limits the value of detailed transition analysis for engagement prediction.

---

## Speech Analysis Features

| Feature | Reason for Removal RF | Reason for Removal KM | Difficulty | Future Value | What Needs to Change | Notes |
|---------|----------------------|----------------------|------------|--------------|---------------------|-------|
| body_language_congruence | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw gesture metrics | Subjective "matching" judgment. Requires semantic understanding of speech + gestures. Black box congruence algorithm. ML should discover patterns |
| burstPattern | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw energy metrics | Semantic categories (regular/irregular/clustered). Single label loses temporal detail. ML should discover patterns from avgAudioEnergy + energyVariance + peaks |
| clarity | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw speech metrics | Subjective articulation assessment. Black box clarity algorithm. Cultural/accent bias. ML should discover from wordsPerMinute, silenceRatio what patterns work |
| confidence | Meta-data Feature | Meta-data Feature | N/A | Low | Separate quality category | Analysis quality metric, not content feature. Should group all flow confidence scores separately for data filtering, not ML training |
| deliveryStyle | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw speech metrics | Semantic categories (conversational/dramatic/informative). Single label loses temporal variation. ML should discover from WPM, energy, silence patterns |
| emotionalRange | Interpreted Feature | Interpreted Feature | N/A | Low | Use pitch/spectral metrics | Black box emotion detection. Vague "range" calculation. Replaced by avgPitch, pitchVariance, spectralCentroid, zeroCrossingRate |
| emphasisTechniques | Interpreted Feature | Variable Array | N/A | Low | Use raw metrics | Vague technique categories. Variable array incompatible with windows. Redundant with energyVariance, repetitionRate, silenceRatio |
| engagement | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw metrics | Circular reasoning - engagement predicting engagement. Black box scoring. ML should discover what's engaging from raw features |
| expression_peaks_during_speech | Pre-correlated Feature | Pre-correlated Feature | N/A | Low | Separate raw metrics | Pre-imposed multimodal correlation. ML should discover from separate emotion peaks and speech coverage if timing matters |
| gesture_emphasis_moments | Pre-correlated Feature | Variable Array | N/A | Low | Separate raw metrics | Pre-imposed gesture-speech alignment. Variable array incompatible. ML should discover from separate gesture and speech peaks |
| hasAudioEnergy | Redundant Feature | Redundant Feature | N/A | Low | Use energy metrics | Binary flag too simplistic. Redundant with energyVariance, avgAudioEnergy, speechCoverage. Almost always TRUE |
| lip_sync_quality | Technical QA Feature | Technical QA Feature | N/A | Low | Not engagement-relevant | Production quality metric. Black box sync assessment. Technical baseline, not engagement driver |
| multiModalCoherence | Pre-correlated Feature | Pre-correlated Feature | N/A | Low | Separate raw metrics | Pre-imposed alignment judgment. Black box coherence. ML should discover from separate speech, gesture, visual data |
| narrativeStyle | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw speech metrics | Semantic categories (storytelling/instructional/promotional). Single label loses variation. ML should discover from pace, pauses, repetition |
| overallScore | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw metrics | Ultimate black box score. Subjective quality judgment. Circular reasoning. ML needs raw features not pre-combined scores |
| silencePeriods | Variable Array | Variable Array | N/A | Medium | Use summary stats | Variable array incompatible with windows. Better captured by silenceRatio + silentMoments + avgSilenceDuration + maxSilenceGap |
| speechBursts | Variable Array | Variable Array | N/A | Low | Use pace metrics | Variable array incompatible. Already captured by wordsPerMinute + pacingVariation + wpmProgression. Bursts show in existing metrics |
| speechDensity | Redundant Feature | Redundant Feature | N/A | Low | Use wordsPerMinute | Perfectly redundant with WPM (just WPM/60). Creates multicollinearity. No unique information |
| speechEmotionAlignment | Pre-correlated Feature | Pre-correlated Feature | N/A | Low | Separate raw metrics | Pre-imposed emotion matching. Black box alignment. ML should discover from separate facial emotion and speech features |
| speechGestureSync | Pre-correlated Feature | Pre-correlated Feature | N/A | Low | Separate raw metrics | Pre-imposed gesture-speech sync. Black box correlation. ML should discover from separate gesture and speech timelines |
| speechTextOverlap | Pre-correlated Feature | Pre-correlated Feature | N/A | Low | Separate raw metrics | Pre-imposed text-speech overlap. Black box calculation. ML should discover from separate text overlay and speech coverage |
| verbalHooks | Interpreted Feature | Variable Array | N/A | Low | Use raw metrics | Semantic hook detection. Variable array incompatible. Subjective attention-grabbing. ML should discover from hook window speech |
| vocabularyDiversity | Redundant Feature | Redundant Feature | N/A | Low | Use components | Perfectly derivable from uniqueWords/totalWords. Creates multicollinearity. ML can discover ratio from raw counts |

---

## Visual Overlay Features

| Feature | Reason for Removal RF | Reason for Removal KM | Difficulty | Future Value | What Needs to Change | Notes |
|---------|----------------------|----------------------|------------|--------------|---------------------|-------|
| crossModalCoherence | Repetitive Feature | Repetitive Feature | N/A | None | Use component features | Simple average of overlaySpeechAlignment and overlayGestureSync. ML can learn this relationship. Creates redundancy with its components |
| ctaMoments | High Data Loss | High Data Loss | Medium | High | Text embeddings needed | 80-95% data loss when adapting. Loses actual CTA text content which is critical. Keyword-based detection is semantic interpretation. Redundant with closing window CTA features. Without embeddings, just weak count |
| engagementArchetype | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw overlay patterns | Semantic categorization (educational/entertainment/promotional). Pre-imposed content taxonomy. ML should discover archetypes from raw features, not be given human labels |
| multimodalMoments | Pre-computed Correlation | Pre-computed Correlation | Medium | Low | Add raw data to temporal windows | Pre-computes text-speech-gesture alignment with 1-second threshold. ML should discover correlations from raw timelines. See FeaturesMLMVP.md "Improve Raw Data" section for MVP implementation. 70-85% data loss in adaptation |
| multimodalReinforcementCount | Pre-computed Correlation | Pre-computed Correlation | Low | None | Add raw data to temporal windows | Count of multimodalMoments. ML should count co-occurrences from raw data per window. Resolved by FeaturesMLMVP.md "Improve Raw Data" section MVP implementation. Redundant with multimodalMoments |
| overlayAcceleration | Temporal Architecture | Temporal Architecture | Low | Low | Add overlay slopes to temporal windows | Better captured by overlay-specific slopes in temporal windows. See FeaturesMLMVP.md "Improve Raw Data" section for MVP implementation. Keeps all acceleration patterns in one consistent location |
| overlayDensity | Architectural Redundancy | Architectural Redundancy | Low | None | Use temporal window counts | Derivable from window overlay counts / duration. Violates "temporal windows as single source of truth" principle. See FeaturesMLMVP.md "Improve Raw Data" section |
| overlayFrequency | Architectural Redundancy | Architectural Redundancy | Low | None | Use temporal window counts | Same as overlayDensity but per minute instead of per second. Derivable from window counts / duration. Violates temporal windows principle |
| overlayGestureSync | MVP Scope Limitation | MVP Scope Limitation | Low | Medium | Add sync metrics in Phase 2 | Requires complex timing calculations. MVP uses counts only per architectural decision. See FeaturesMLMVP.md "Multimodal Coordination" section |
| overlayPeaks | Temporal Redundancy | Variable Array | Medium | Low | Use temporal window peaks | Redundant with middle_window peak_value/position. Variable array loses 70-85% data when adapted. Should be in temporal windows per architecture |
| overlayProgression | Competing Architecture | Competing Architecture | Low | Low | Use temporal windows | Creates competing temporal segmentation (10 segments vs 3 windows). Violates single temporal architecture principle. Overlay patterns captured by window-specific metrics |
| overlaySpeechAlignment | MVP Scope Limitation | MVP Scope Limitation | Low | Medium | Add sync metrics in Phase 2 | Same as overlayGestureSync but for text-speech. Requires timing precision. MVP uses counts only per architectural decision. See FeaturesMLMVP.md "Multimodal Coordination" section |
| overlayStrategy | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw window counts | Bucketed version of overlayDensity (minimal/moderate/heavy). Arbitrary thresholds. ML should discover patterns from raw counts, not pre-imposed categories |
| overlayTechniques | Interpreted Feature | Variable Array | Medium | Low | Use raw metrics | Pre-labeled technique taxonomy (direct_cta, rhythmic_timing, etc). Derived from removed features. ML should discover techniques from raw data. 50-95% data loss in adaptation |
| pacingPattern | Duplicate Feature | Duplicate Feature | N/A | None | Already removed as overlayAcceleration | Exact duplicate of overlayAcceleration with different name. Same calculation, same values. Covered by temporal window slopes and between-window patterns |
| quietMoments | Variable Array | Variable Array | Medium | Medium | Add quiet period metrics | Variable array of timestamps incompatible with RF/K-means. 70-85% data loss. Replaced by window/segment quiet ratios. See FeaturesMLMVP.md "Quiet Period Metrics" section |
| rhythmConsistency | Semantic Interpretation | Semantic Interpretation | Low | None | Add overlayTimingVariance if needed | Measures overlay timing regularity as "consistency". Derivative metric (1-variance). Not temporal window compatible. ML should discover if regular timing helps |
| timeToFirstOverlay | Temporal Redundancy | Temporal Redundancy | Low | None | Use hook_overlay_count > 0 | Single timestamp redundant with temporal windows. If hook has overlays, starts in 0-3s. If not, check middle segments. Precise sub-window timing not critical |
| totalOverlays | Architectural Redundancy | Architectural Redundancy | Low | None | Sum window overlay counts | Global count violates temporal windows as single source of truth. Moved to windows in ImprovementsMLMVP.md. ML derives total from window sums |
| totalStickers | Architectural Redundancy | Architectural Redundancy | Low | None | Sum window sticker counts | Global count violates temporal windows principle. Already included in "Overlay Counts in Windows" improvement. ML derives from hook/middle/closing sticker counts |
| totalTextOverlays | Architectural Redundancy | Architectural Redundancy | Low | None | Sum window text overlay counts | Global count violates temporal windows principle. Already included in "Overlay Counts in Windows" improvement. ML derives from hook/middle/closing text overlay counts |
| uniqueOverlayRatio | Derivative Metric | Derivative Metric | Low | None | Use unique_count / total_derived | Pure derivative: unique_overlay_count / total_overlays. With "Derived Global Metrics" improvement, ML can easily calculate. Avoids redundant storage |
| temporalDistribution | Semantic Interpretation | Semantic Interpretation | Low | Medium | Use window overlay counts | Pre-categorized as front/even/back_loaded. Arbitrary thresholds. Completely redundant with temporal window counts. ML discovers distribution from raw counts |

---

## Metadata Analysis Features

| Feature | Reason for Removal RF | Reason for Removal KM | Difficulty | Future Value | What Needs to Change | Notes |
|---------|----------------------|----------------------|------------|--------------|---------------------|-------|
| hasCaption | Repetitive Feature | Repetitive Feature | N/A | None | Use wordCount > 0 | Directly derived from wordCount > 0, perfect redundancy with captionLength and wordCount |
| hashtags | Debugging Work | Incompatible | High | High | Text embedding or topic modeling | Variable text array with infinite possibilities. One-hot encoding would create extreme dimensionality and sparsity. Strategy captured via hashtagCount, genericRatio, hashtagBreakdown. WITH DEEP LEARNING: Becomes highly valuable! Options: 1) Text embeddings (BERT/Word2Vec) to convert hashtags to semantic vectors, captures meaning (#cooking ≈ #recipes). 2) Sequence models (LSTM/Transformer) process variable-length lists. 3) Hybrid: top 1000 one-hot + embedding fallback for rare tags. 4) Pre-trained TikTok hashtag embeddings from millions of posts. DL handles infinite hashtags, learns semantic relationships, and captures trend participation - making this a goldmine feature for 1000+ videos |
| hashtagStrategy | Repetitive Feature | Repetitive Feature | N/A | None | Use hashtagCount directly | Just binned hashtagCount: 0=none, 1-2=minimal, 3-7=moderate, 8-15=heavy, 16+=spam. ML can learn optimal thresholds from raw count. Arbitrary categorization loses granularity |
| keyMentions | Debugging Work | Incompatible | High | Med | Social graph features or influence scores | Variable text array with infinite usernames. One-hot encoding impossible (even more users than hashtags). Can only extract count, likely redundant with mentionCount. Identity of mentions lost (influencers vs friends). WITH DEEP LEARNING: Could embed usernames, track influence scores, or build social graph features |
| primaryEmojis | Debugging Work | Partial | Medium | Med | One-hot top 50-100 emojis | Variable emoji array. While more manageable than hashtags (~3000 possible vs infinite), still requires one-hot encoding top 50-100 emojis for MVP. Valuable emotional/content signals (😂=humor, ❤️=romantic, 🔥=hype) but complexity not worth it for MVP. We capture emoji usage via emojiCount and emojiDensity. WITH DEEP LEARNING: Could use emoji embeddings or semantic groupings |
| strategy | Interpreted Feature | Interpreted Feature | N/A | Low | Use raw features | High-level semantic categorization of content strategy. Human-imposed categories like "educational", "entertainment", "promotional". ML should discover strategies from raw features, not be told them. Classic over-interpretation |

---

## Temporal Markers Features

| Feature | Reason for Removal RF | Reason for Removal KM | Difficulty | Future Value | What Needs to Change | Notes |
|---------|----------------------|----------------------|------------|--------------|---------------------|-------|
| *To be added during temporal_markers review* | | | | | | |

---

## Summary Statistics

| Flow | Features Removed | Primary Removal Reasons |
|------|------------------|------------------------|
| creative_density | TBD | TBD |
| emotional_journey | TBD | TBD |
| person_framing | TBD | TBD |
| scene_pacing | TBD | TBD |
| speech_analysis | TBD | TBD |
| visual_overlay | TBD | TBD |
| metadata_analysis | TBD | TBD |
| temporal_markers | TBD | TBD |

---

## Common Removal Patterns
*To be identified after all flows reviewed*

1. **Debugging Work**: Features that need fixes but not worth it for MVP
2. **Incompatible**: Features that can't work with RF or KM algorithms
3. **Module Needed**: Features requiring additional module installations
4. **Interpreted Features**: Semantic features that obscure raw patterns
5. **Repetitive Features**: Features that duplicate information from others
6. **Temporal Redundancy**: Features better captured by MLMVP2's temporal windows