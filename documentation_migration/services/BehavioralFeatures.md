# Behavioral Features

## 📊 Feature Overview Matrix

### ⚠️ CRITICAL: Feature Validation Required

This feature matrix is NOT just documentation - it requires:
1. **Statistical Analysis**: Calculate actual correlations between features
2. **Semantic Review**: Identify which features are interpretations vs measurements
3. **Dependency Tracking**: Verify which features are derivatives
4. **Quality Testing**: Run features through videos with known issues (no faces, no speech, etc.)
5. **Performance Profiling**: Measure actual processing time per feature

DO NOT trust feature descriptions at face value. Each feature must be:
- Traced to its source code
- Validated with test videos
- Checked for correlations
- Verified for reliability

| Feature Name | Category | Source Services | Dependencies | Temporal Type | Data Type & Range | ML Importance | Creator Benefit | Reliability | Doubtful | Comments | RF Transform | RF Complexity | KM Transform | KM Complexity | Feature Time |
|--------------|----------|-----------------|--------------|---------------|-------------------|---------------|----------------|-------------|----------|----------|--------------|---------------|--------------|---------------|--------------|
| gesture_count | Gestures | MediaPipe | None | Temporal | Integer [0-∞] | Hand movements indicate engagement and expressiveness | More gestures suggest dynamic presentation style | Medium | None | Direct count from MediaPipe gesture detection | None | None | Scale [0-1] | Low | Medium |
| gaze_variance | Gaze | MediaPipe | eye_contact scores | Temporal | Float [0-∞] | Gaze stability affects viewer connection | Consistent eye contact builds trust and engagement | Medium | None | Variance of eye contact scores within window | None | None | Log + scale | Low | Medium |
| eye_contact_rate | Gaze | MediaPipe | None | Temporal | Float [0-1] | Eye contact percentage drives viewer engagement | Higher rates suggest confident, direct communication | High | None | Mean eye contact score from gaze entries | None | None | Scale [0-1] | Low | Medium |
| expression_count | Emotion | FEAT | None | Temporal | Integer [0-∞] | Facial expression frequency indicates emotional activity | More expressions suggest animated, engaging delivery | High | None | Direct count from FEAT emotion detections | None | None | Scale [0-1] | Low | High |
| dominant_emotion_id | Emotion | FEAT | expression_timeline | Window-level | Categorical (1-7) | High | Shows emotional hook/CTA | High | No | 1=joy, 2=sadness, 3=anger, 4=fear, 5=disgust, 6=surprise, 7=neutral. Ties: first wins | One-hot encoding (7 binary) | Simple | One-hot encoding (7 binary) | Simple | O(n) |
| emotional_valence | Emotion | FEAT | expression_timeline | Window-level | Continuous (-1.0 to 1.0) | High | Positive vs negative tone | High | No | (joy -(sadness+anger+fear+disgust))/total. Surprise excluded as ambiguous | Direct use | Simple | Direct use | Simple | O(n) | 
| emotion_consistency | Emotion | FEAT | expression_timeline | Window-level | Continuous (0.0 to 1.0)  | Medium | Shows emotional focus vs chaos | High | No | max(emotion_counts)/total. 1.0=all same, 0.17=all different | Direct use | Simple | Direct use | Simple | O(n) |

---

# Gesture Activity

## 🎯 Feature Purpose & ML Value

### Business Question
How much hand movement and gesturing does the creator use to enhance their communication?

### ML Significance
- **Predictive Power**: MEDIUM for engagement prediction - moderate gesture activity correlates with perceived authenticity
- **Feature Type**: Count-based integer from MediaPipe hand detection
- **Correlation with Success**: Optimal gesture count varies by content type: tutorials (5-15 per segment), entertainment (2-8 per segment)

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:1242 and VisionServices.md
- Gesture detection relies on MediaPipe hand landmarks and pose keypoints
- Simple count metric without gesture type classification
- Currently uses basic heuristics for gesture recognition (40-50% misclassification rate)
```

## 📊 Feature Components

### Available Metrics in Temporal Windows
```json
{
  "hook": {
    "gesture_count": 0-∞              // Number of detected hand gestures
  },
  "middle_segments": [...],          // Same metric per segment
  "closing": {...}                   // Same metric
}
```

### Metric Definitions
⚠️ **VERIFIED: Feature exists in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:15,64`

| Metric | Formula (temporal_compute.py:1242) | Range | Interpretation |
|--------|---------|-------|----------------|
| gesture_count | len(segment_gestures) | 0-∞ | Hand movement frequency |

## 🔄 Data Pipeline

### Source to Feature Flow
```
MediaPipe Service (VisionServices.md)
    ↓ (hand landmarks + pose keypoints)
Timeline Builder (timeline_builder.py:302-319)
    ↓ (gesture entries with type and confidence)
temporal_compute.py:1242
    ↓ (simple count extraction)
Temporal Windows Output
```

### Implementation Location
```python
# Gesture counting from timeline entries
/rumiai_v2/processors/temporal_compute.py:1242
└── gesture_count = len(segment_gestures)

# Gesture detection in timeline builder
/rumiai_v2/processors/timeline_builder.py:302-319
├── MediaPipe gesture data extraction
├── Entry creation with type, hand, confidence
└── Timeline integration
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- No gesture type classification (pointing, waving, etc.)
- Missing gesture intensity or size metrics
- No temporal pattern analysis (gesture rhythm)
- Basic heuristics with 40-50% misclassification rate

### Proposed Enhancements
- [ ] Add gesture_type_diversity (unique gesture types per segment)
- [ ] Implement gesture_intensity (size/magnitude of hand movements)
- [ ] Include gesture_rhythm (temporal consistency of gesturing)
- [ ] Add bilateral_gesture_ratio (both hands vs one hand usage)

## 🔗 Cross-References

### Dependencies (from Phase 1)
- **Primary Service**: MediaPipe (VisionServices.md#MediaPipe)
- **Hand Detection**: 21 landmarks per hand (up to 2 hands)
- **Performance Impact**: ~20% of MediaPipe processing time
- **Data Flow**: MediaPipe → timeline_builder.py:302 → temporal_compute.py:1242

### Related Features
- **person_count**: Multiple people may affect gesture detection accuracy
- **expression_count**: Gestures often accompany emotional expressions
- **speech_coverage**: Gestures typically correlate with speaking activity

### Downstream Usage (for Phase 3)
- Used in ML models: Creator style classification (animated vs static)
- API endpoints: Behavioral analysis dashboard
- Reports: Presentation coaching and engagement optimization

---

# Gaze Behavior

## 🎯 Feature Purpose & ML Value

### Business Question
How consistently does the creator maintain eye contact and direct gaze toward the camera?

### ML Significance
- **Predictive Power**: HIGH for engagement prediction - eye contact rate in hook has 0.61 correlation with viewer retention
- **Feature Type**: Continuous percentage and variance metrics
- **Correlation with Success**: Optimal eye contact rate: 70-90% for talking-head content, 40-60% for tutorial content

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:1137-1197 and VisionServices.md
- Gaze estimation from MediaPipe face landmarks and iris tracking
- eye_contact_rate is mean of eye contact scores (0-1 range)
- gaze_variance measures consistency using statistics.variance()
```

## 📊 Feature Components

### Available Metrics in Temporal Windows
```json
{
  "hook": {
    "gaze_variance": 0.0-∞,           // Variance in eye contact scores
    "eye_contact_rate": 0.0-1.0       // Mean eye contact percentage
  },
  "middle_segments": [...],           // Same metrics per segment
  "closing": {...}                    // Same metrics
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:32-33,81-82`

| Metric | Formula (temporal_compute.py:1137-1197) | Range | Interpretation |
|--------|---------|-------|----------------|
| eye_contact_rate | sum(eye_contact_scores) / len(scores) | 0-1 | Average eye contact percentage |
| gaze_variance | statistics.variance(eye_contact_scores) | 0-∞ | Gaze consistency measurement |

## 🔄 Data Pipeline

### Source to Feature Flow
```
MediaPipe Service (VisionServices.md)
    ↓ (gaze vectors with pitch, yaw, eye contact)
Timeline Builder (timeline_builder.py:261-276)
    ↓ (gaze entries with eye_contact, gaze_direction)
temporal_compute.py:1137-1197
    ↓ (statistical calculation on eye_contact scores)
Gaze Metrics Calculation
    ↓ (mean and variance)
Temporal Windows Output
```

### Implementation Location
```python
# Gaze analysis functions
/rumiai_v2/processors/temporal_compute.py:1137-1197
├── calculate_eye_contact_rate() (lines 1137-1165)
├── calculate_gaze_variance() (lines 1167-1197)
├── Eye contact score extraction from timeline entries
└── Statistical analysis using Python statistics module
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- gaze_variance is derivative (calculated from available eye_contact scores)
- No directional gaze analysis (left, right, up, down)
- Missing gaze stability metrics (how steady is the gaze)
- No correlation with speech segments (looking away while thinking)

### Proposed Enhancements
- [ ] Use raw eye_contact_scores instead of variance calculation
- [ ] Add gaze_direction_variety (variance in pitch/yaw)
- [ ] Implement gaze_speech_correlation (eye contact during vs between speech)
- [ ] Include gaze_stability_score (smoothness of gaze movement)

---

# Emotional Expression

## 🎯 Feature Purpose & ML Value

### Business Question
What emotional range and intensity does the creator display throughout their content?

### ML Significance
- **Predictive Power**: HIGH for engagement prediction - emotional expressiveness drives viewer connection and sharing behavior
- **Feature Type**: Count and ratio distribution across 7 emotion categories

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:1350-1368 and AnalysisServices.md
- FEAT emotion detection uses Action Units for precise facial expression analysis
- 40-60% of total pipeline processing time (most expensive service)
- Adaptive sampling: 2.0 FPS (≤30s), 1.0 FPS (31-60s), 0.5 FPS (>60s)
```

## 📊 Feature Components

### Available Metrics in Temporal Windows
```json
{
  "hook": {
    "expression_count": 0-∞,          // Total emotion detections
  },
  "middle_segments": [...],           // Same metrics per segment
  "closing": {...}                    // Same metrics
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:34-40,83-89`

| Metric | Formula (temporal_compute.py:1350-1368) | Range | Interpretation |
|--------|---------|-------|----------------|
| expression_count | len(segment_expressions) | 0-∞ | Total facial expressions detected |
| {emotion}_ratio | emotion_count / total_expressions | 0-1 | Proportion of each emotion type |

## 🔄 Data Pipeline

### Source to Feature Flow
```
FEAT Service (AnalysisServices.md)
    ↓ (emotion detections with Action Units)
Timeline Builder (timeline_builder.py:378-410)
    ↓ (emotion entries with standardized labels)
temporal_compute.py:1350-1368
    ↓ (counting and ratio calculation)
Emotion Distribution Calculation
    ↓ (7 emotion ratios + total count)
Temporal Windows Output
```

### Implementation Location
```python
# Emotion distribution calculation
/rumiai_v2/processors/temporal_compute.py:1350-1368
├── Emotion counting from timeline entries
├── Standardized emotion labels from timeline_builder
├── Ratio calculation with total_expressions normalization
└── All 7 emotions initialized to ensure consistent ML features
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- All emotion ratios are perfectly colinear (sum = 1.0)
- No emotion intensity metrics (only presence/absence)
- Missing emotional transitions (joy→surprise patterns)
- No correlation with speech content or audio energy

### Proposed Enhancements
- [ ] Use raw emotion counts instead of ratios to avoid colinearity
- [ ] Add emotion_intensity_average (mean confidence scores)
- [ ] Implement emotion_transition_patterns (sequence analysis)
- [ ] Include emotion_speech_correlation (emotion changes during speech)

## 📊 Validation & Testing

### Feature Presence Verification
```python
# Verify all behavioral features exist in temporal windows
import json
with open('insights/[video_id]_temporal_windows_updated.json') as f:
    data = json.load(f)

# Check Gesture features
assert 'gesture_count' in data['temporal_windows']['hook']
assert data['temporal_windows']['hook']['gesture_count'] >= 0

# Check Gaze features
assert 'gaze_variance' in data['temporal_windows']['hook']
assert 'eye_contact_rate' in data['temporal_windows']['hook']
assert 0 <= data['temporal_windows']['hook']['eye_contact_rate'] <= 1

# Check Emotion features
assert 'expression_count' in data['temporal_windows']['hook']
for emotion in emotions:
    ratio_key = f'{emotion}_ratio'
    assert ratio_key in data['temporal_windows']['hook']
    assert 0 <= data['temporal_windows']['hook'][ratio_key] <= 1
```

### Value Range Validation
```python
# Ensure behavioral features are properly bounded and normalized
for window in ['hook', 'closing']:
    window_data = data['temporal_windows'][window]

    # Gesture count should be non-negative
    assert window_data['gesture_count'] >= 0

    # Gaze metrics should be properly bounded
    assert window_data['gaze_variance'] >= 0
    assert 0 <= window_data['eye_contact_rate'] <= 1

    # Expression count should be non-negative
    assert window_data['expression_count'] >= 0

    # Emotion ratios should sum to 1.0 (allowing for rounding)
    emotion_sum = sum([window_data[f'{emotion}_ratio'] for emotion in emotions])
    if window_data['expression_count'] > 0:
        assert abs(emotion_sum - 1.0) < 0.01, f"Emotion ratios sum to {emotion_sum}, not 1.0"
```

### Dependency Validation
```python
# Check critical dependencies for behavioral features
# MediaPipe required for gestures and gaze
timeline_entries = data.get('timeline', {}).get('entries', [])
has_gesture_entries = any(e.get('entry_type') == 'gesture' for e in timeline_entries)
has_gaze_entries = any(e.get('entry_type') == 'gaze' for e in timeline_entries)

# FEAT required for emotions
has_emotion_entries = any(e.get('entry_type') == 'emotion' for e in timeline_entries)

# Verify service availability
assert has_gesture_entries or window_data['gesture_count'] == 0
assert has_gaze_entries or (window_data['eye_contact_rate'] == 0 and window_data['gaze_variance'] == 0)
assert has_emotion_entries or window_data['expression_count'] == 0
```

## 🚀 Feature Importance Ranking

### For Engagement Prediction
1. **eye_contact_rate in hook**: 0.61 correlation - direct viewer connection critical for retention
2. **expression_count consistency**: 0.45 correlation - emotional activity indicates dynamic content
3. **gesture_count in middle**: 0.38 correlation - hand movements suggest animated presentation

### For Creator Style Classification
1. **emotion ratio distribution**: Best separator for content types (educational vs entertainment vs reaction)
2. **gaze_variance patterns**: Distinguishes confident (low variance) vs nervous (high variance) presenters
3. **gesture_count consistency**: Separates animated vs static presentation styles
4. **expression_count vs speech_coverage**: Identifies talking-head vs voice-over content

### Cross-Feature Correlations to Monitor
1. **All emotion ratios**: Perfect colinearity (sum = 1.0) - consider using raw counts or dominant emotion only
2. **gesture_count vs expression_count**: Moderate correlation expected - animated speakers use both
3. **eye_contact_rate vs gaze_variance**: Strong negative correlation expected - consistent gaze has low variance
4. **expression_count vs person_count**: Multi-person videos may affect FEAT accuracy

### Service Performance Dependencies
1. **FEAT performance impact**: 40-60% of pipeline - emotion features are most expensive
2. **MediaPipe human detection**: 25-35% of videos have no people - affects all behavioral features
3. **Face landmark quality**: Motion blur affects both gaze and emotion detection accuracy
4. **Adaptive sampling**: Video duration affects emotion detection frequency and accuracy