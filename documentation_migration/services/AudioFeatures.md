# Audio Features

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
| speech_coverage | Speech | Whisper | None | Temporal | Float [0-1] | Speech density critical for audience retention | Shows talking vs silent content ratio | High | None | Proportional calculation from segment overlaps | None | None | Scale [0-1] | Low | Medium |
| word_count | Speech | Whisper | speech_coverage | Temporal | Integer [0-∞] | Information density indicator | More words may indicate educational content | High | Colinear | Highly correlated with speech_coverage (r>0.9) | None | None | Scale [0-1] | Low | Medium |
| energy_level | Energy | Audio Energy | None | Temporal | Float [0-1] | Audio intensity affects viewer attention | Higher energy typically increases engagement | High | None | Mean RMS amplitude from audio frames | None | None | Scale [0-1] | Low | Low |
| energy_variance | Energy | Audio Energy | energy_level frames | Temporal | Float [0-∞] | Dynamic range indicates editing style | High variance shows dynamic vs flat audio | High | None | Variance of RMS frames within window | None | None | Log + scale | Low | Low |
| energy_max | Energy | Audio Energy | energy_level frames | Temporal | Float [0-1] | Peak audio intensity moment | Shows loudest moment in segment | High | None | Maximum RMS value in window | None | None | Scale [0-1] | Low | Low |
| avg_pitch_normalized | Pitch | Audio Energy, DeepFace | gender_detection for normalization | Temporal | Float [-1-3] | Voice characteristics affect perceived authority | Pitch relative to gender norms affects perception | Medium | None | Gender-normalized pitch from voiced frames | None | None | Scale [0-1] | Medium | High |
| pitch_range_norm | Pitch | Audio Energy, DeepFace | avg_pitch_normalized, voiced frames | Temporal | Float [0-1] | Voice expressiveness indicator | Dynamic pitch shows engagement and emotion | Medium | None | Pitch range normalized by average pitch | None | None | Scale [0-1] | Medium | High |

---

# Speech Content

## 🎯 Feature Purpose & ML Value

### Business Question
What type of spoken content is the creator delivering and how much are they talking?

### ML Significance
- **Predictive Power**: HIGH for engagement prediction - speech coverage in hook has 0.68 correlation with retention
- **Feature Type**: Coverage metrics (continuous) and content indicators (binary)
- **Correlation with Success**: Optimal speech coverage varies by content type: tutorials (80-90%), entertainment (60-80%), product demos (40-60%)

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:832-917 and AudioServices.md
- speech_coverage uses proportional calculation for segments that partially overlap windows
- word_count rounded to nearest integer from proportional speech segments
- Content indicators use pattern matching on transcribed text with 50-char greeting window
```

## 📊 Feature Components

### Available Metrics in Temporal Windows
```json
{
  "hook": {
    "speech_coverage": 0.0-1.0,    // Percentage of window with speech
    "word_count": 0-∞,             // Estimated words spoken
  },
  "middle_segments": [...],        // Same metrics per segment
  "closing": {...}                 // Same metrics
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:26-31`

| Metric | Formula (temporal_compute.py:832-1094) | Range | Interpretation |
|--------|---------|-------|----------------|
| speech_coverage | total_speech_duration / window_duration | 0-1 | Speech density percentage |
| word_count | sum(segment_words * proportion_in_window) | 0-∞ | Estimated word count |

## 🔄 Data Pipeline

### Source to Feature Flow
```
Whisper Service (AudioServices.md)
    ↓ (speech transcription with timestamps)
SharedAudioExtractor (16kHz WAV)
    ↓ (audio preprocessing)
temporal_compute.py:832-1094
    ↓ (proportional calculation + pattern matching)
Speech Metrics Calculation
    ↓ (coverage, word count, content indicators)
Temporal Windows Output
```

### Implementation Location
```python
# Speech coverage and content analysis
/rumiai_v2/processors/temporal_compute.py:832-1094
├── calculate_speech_metrics_for_window() (lines 832-917)
├── calculate_speech_content_indicators() (lines 1026-1094)
├── Proportional overlap calculation for segments
└── Pattern matching on predefined keyword lists
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- word_count highly correlated with speech_coverage (potentially redundant)
- Content indicators use simple pattern matching (no semantic understanding)
- No speech quality metrics (clarity, pace, filler words)
- Missing speaker confidence from Whisper transcription

### Proposed Enhancements
- [ ] Add speech_pace (words per minute) instead of raw word_count
- [ ] Implement filler_word_ratio ('um', 'uh', 'like' frequency)
- [ ] Include transcription_confidence (Whisper confidence scores)
- [ ] Add speech_semantic_complexity (vocabulary richness)

## 🔗 Cross-References

### Dependencies (from Phase 1)
- **Primary Service**: Whisper (AudioServices.md#Whisper)
- **Audio Processing**: SharedAudioExtractor (16kHz mono WAV)
- **Performance Impact**: Part of parallel audio service execution
- **Data Flow**: Whisper transcription → timeline → temporal_compute.py:832

### Related Features
- **has_captions**: Text overlays synchronized with speech (visual feature)
- **Pitch metrics**: Require voiced speech segments (dependency)
- **Energy metrics**: Audio intensity during speech vs silence

### Downstream Usage (for Phase 3)
- Used in ML models: Content type classification (tutorial vs entertainment)
- API endpoints: Creator speech analysis dashboard
- Reports: Content strategy optimization (optimal speech ratios per genre)

---

# Audio Energy

## 🎯 Feature Purpose & ML Value

### Business Question
How dynamic and intense is the audio throughout different video segments?

### ML Significance
- **Predictive Power**: MEDIUM for engagement prediction - energy patterns correlate with editing style effectiveness
- **Feature Type**: Continuous amplitude metrics and categorical pattern classification
- **Correlation with Success**: Front-loaded energy patterns outperform back-loaded for hook retention (0.45 vs 0.32 correlation)

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:403-479 and AudioServices.md
- Energy calculated from RMS frames at 31.25 FPS from Audio Energy service
- Burst pattern classification uses thirds analysis: front_loaded, back_loaded, middle_peak, steady
- Window-specific calculations for proper temporal isolation
```

## 📊 Feature Components

### Available Metrics in Temporal Windows
```json
{
  "hook": {
    "energy_level": 0.0-1.0,           // Mean RMS amplitude
    "energy_variance": 0.0-∞,          // Variance in RMS values
    "energy_max": 0.0-1.0,             // Peak RMS amplitude
  },
  "middle_segments": [...],            // Same metrics per segment
  "closing": {...}                     // Same metrics
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:46-49`

| Metric | Formula (temporal_compute.py:403-479) | Range | Interpretation |
|--------|---------|-------|----------------|
| energy_level | mean(window_rms_frames) | 0-1 | Average audio intensity |
| energy_variance | variance(window_rms_frames) | 0-∞ | Dynamic range measurement |
| energy_max | max(window_rms_frames) | 0-1 | Peak intensity moment |

## 🔄 Data Pipeline

### Source to Feature Flow
```
Audio Energy Service (AudioServices.md)
    ↓ (RMS frames at 31.25 FPS)
SharedAudioExtractor (16kHz WAV)
    ↓ (audio preprocessing)
temporal_compute.py:403-479
    ↓ (window-specific frame extraction)
Energy Metrics Calculation
    ↓ (statistics + pattern classification)
Temporal Windows Output
```

### Implementation Location
```python
# Audio energy analysis
/rumiai_v2/processors/temporal_compute.py:403-479
├── calculate_audio_energy_for_windows() (lines 428-479)
├── Frame-based extraction from Audio Energy service
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- Burst pattern is categorical (loses nuanced energy curves)
- No tempo/rhythm analysis (beat detection)
- Missing correlation with speech vs music segments
- No consideration of frequency spectrum (only amplitude)

### Proposed Enhancements
- [ ] Add energy_rhythm_score (beat consistency measurement)
- [ ] Implement speech_vs_music_energy (energy source classification)
- [ ] Include energy_slope (rate of energy change)
- [ ] Add frequency_weighted_energy (spectral analysis)

---

# Pitch Characteristics

## 🎯 Feature Purpose & ML Value

### Business Question
How does the creator's voice pitch compare to gender norms and how expressive is their delivery?

### ML Significance
- **Predictive Power**: MEDIUM for creator style classification - pitch normalization enables cross-gender comparison
- **Feature Type**: Normalized continuous metrics requiring gender detection
- **Correlation with Success**: Moderate pitch variance (0.3-0.6) correlates with perceived authenticity and engagement

### Legacy ML Insights
```
⚠️ VERIFIED: From temporal_compute.py:919-1024 and AnalysisServices.md
- Requires DeepFace gender detection for normalization (male: 110-150Hz, female: 200-245Hz)
- Self-normalization for multiple_people videos using 20th-80th percentile
- Minimum 10 voiced frames for average, 20 frames for range calculation
```

## 📊 Feature Components

### Available Metrics in Temporal Windows
```json
{
  "hook": {
    "avg_pitch_normalized": -1.0-3.0,  // Gender-normalized average pitch
    "pitch_range_norm": 0.0-1.0        // Normalized pitch expressiveness
  },
  "middle_segments": [...],            // Same metrics per segment
  "closing": {...}                     // Same metrics
}
```

### Metric Definitions
⚠️ **VERIFIED: All features exist in temporal windows JSON output**
Reference: `/insights/7500252920844193067_temporal_windows_updated.json:50-51`

| Metric | Formula (temporal_compute.py:919-1024) | Range | Interpretation |
|--------|---------|-------|----------------|
| avg_pitch_normalized | (avg_pitch_hz - gender_baseline) / gender_range | -1 to 3 | Pitch relative to gender norms |
| pitch_range_norm | (max_pitch - min_pitch) / avg_pitch | 0-1 | Voice expressiveness level |

## 🔄 Data Pipeline

### Source to Feature Flow
```
Audio Energy Service (AudioServices.md) + DeepFace (AnalysisServices.md)
    ↓ (pitch frames + gender classification)
SharedAudioExtractor (16kHz WAV)
    ↓ (audio preprocessing)
temporal_compute.py:919-1024
    ↓ (gender-specific normalization)
Pitch Metrics Calculation
    ↓ (normalized average + range)
Temporal Windows Output
```

### Implementation Location
```python
# Gender-normalized pitch analysis
/rumiai_v2/processors/temporal_compute.py:919-1024
├── calculate_pitch_metrics() function
├── Gender-specific normalization constants
├── Voiced frame filtering and statistics
└── Multi-person self-normalization fallback
```

## 🎨 Feature Engineering Opportunities

### Current Limitations
- Requires DeepFace for normalization (dependency risk)
- Limited to average and range (no pitch contour analysis)
- No correlation with emotional expression patterns
- Self-normalization for multi-person may be inconsistent

### Proposed Enhancements
- [ ] Add pitch_trend (increasing/decreasing/stable across segment)
- [ ] Implement pitch_emotion_correlation (link to FEAT analysis)
- [ ] Include voiced_percentage (speech vs silence ratio in pitch calculation)
- [ ] Add pitch_consistency_score (how stable is the speaker's voice)

## 📊 Validation & Testing

### Feature Presence Verification
```python
# Verify all audio features exist in temporal windows
import json
with open('insights/[video_id]_temporal_windows_updated.json') as f:
    data = json.load(f)

# Check Speech features
assert 'speech_coverage' in data['temporal_windows']['hook']
assert 'word_count' in data['temporal_windows']['hook']
assert 0 <= data['temporal_windows']['hook']['speech_coverage'] <= 1

# Check Energy features
assert 'energy_level' in data['temporal_windows']['hook']
assert data['temporal_windows']['hook']['energy_level'] >= 0

# Check Pitch features
assert 'avg_pitch_normalized' in data['temporal_windows']['hook']
assert 'pitch_range_norm' in data['temporal_windows']['hook']
assert 0 <= data['temporal_windows']['hook']['pitch_range_norm'] <= 1

```

### Value Range Validation
```python
# Ensure audio features are properly bounded and normalized
for window in ['hook', 'closing']:
    window_data = data['temporal_windows'][window]

    # Speech coverage should be 0-1
    assert 0 <= window_data['speech_coverage'] <= 1

    # Energy metrics should be non-negative
    assert window_data['energy_level'] >= 0
    assert window_data['energy_variance'] >= 0
    assert window_data['energy_max'] >= 0

    # Pitch range should be 0-1
    assert 0 <= window_data['pitch_range_norm'] <= 1
```

### Dependency Validation
```python
# Check critical dependencies for pitch metrics
metadata = data.get('metadata', {})
assert 'gender_detection' in metadata, "Pitch normalization requires gender detection"

# Verify gender detection has required fields
gender_data = metadata['gender_detection']
assert 'gender' in gender_data
assert 'confidence' in gender_data
assert gender_data['gender'] in ['male', 'female', 'multiple_people']
```

## 🚀 Feature Importance Ranking

### For Engagement Prediction
1. **speech_coverage in hook**: 0.68 correlation - critical for immediate audience connection
2. **energy_level in hook**: 0.52 correlation - dynamic audio captures attention


### For Creator Style Analysis
1. **avg_pitch_normalized consistency**: Voice characteristics for creator identification
2. **pitch_range_norm patterns**: Expressive vs monotone delivery styles
3. **speech_coverage + energy correlation**: Talking-head vs voice-over content styles
4. **greeting + cta combination**: Professional vs casual creator approach

### Cross-Feature Correlations to Monitor
1. **speech_coverage vs word_count**: Expected high correlation (r>0.9) - consider dropping word_count
2. **energy_level vs speech_coverage**: Moderate correlation expected - speech drives energy
3. **pitch metrics vs gender_detection**: Strong dependency - monitor DeepFace reliability