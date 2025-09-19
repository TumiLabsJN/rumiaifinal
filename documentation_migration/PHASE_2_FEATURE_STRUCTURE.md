# PHASE 2: Feature Documentation Structure

## ⚠️ CRITICAL CONTEXT
Phase 2 focuses on ML FEATURES and BUSINESS VALUE, not technical implementation (Phase 1) or system architecture (Phase 3).
We're mapping features to temporal windows and explaining their ML significance.

## 📋 Phase 2 Scope
Document WHAT features provide ML value and HOW they appear in temporal windows.
Build upon Phase 1's service documentation to show feature derivation.

### ⚠️ CRITICAL REQUIREMENT
**ONLY document features that exist in the actual temporal windows JSON output.**
Reference file: `/insights/[video_id]_temporal_windows_updated.json`
Every feature in the JSON must appear in exactly ONE of the 4 feature documents.
NO phantom features that don't exist in the output.

## 📁 Documents to Create & Feature Coverage

### Document-to-Feature Mapping (4 Documents Total)
Every feature in temporal windows JSON must be documented in exactly ONE document:

1. **VisualFeatures.md** (~19 features)
   - Person Framing: `close_ratio`, `medium_ratio`, `wide_ratio`, `none_ratio`, `average_face_size`
   - Creative Density: `element_count`, `max_density`, `min_density`, `avg_density`
   - Text Overlays: `overlay_unique_count`, `overlay_coverage`, `overlay_persistence`, `has_captions`
   - Scene Pacing: `scene_count`, `shortest_scene`, `longest_scene`, `scene_duration_variance`, `changes_per_second`
   - Object Detection: `object_count`, `person_count`

2. **AudioFeatures.md** (~12 features)
   - Speech: `speech_coverage`, `word_count`, `has_greeting`, `has_question`, `has_instruction`, `has_speech_cta`
   - Energy: `energy_level`, `energy_variance`, `energy_max`, `burst_pattern`
   - Pitch: `avg_pitch_normalized`, `pitch_range_norm`

3. **BehavioralFeatures.md** (~11 features)
   - Gestures: `gesture_count`
   - Gaze: `gaze_variance`, `eye_contact_rate`
   - Emotion: `expression_count`, `joy_ratio`, `sadness_ratio`, `anger_ratio`, `fear_ratio`, `disgust_ratio`, `surprise_ratio`, `neutral_ratio`

4. **EngagementMetadata.md** (~13 features)
   - Virality Metrics: `digg_count`, `play_count`, `collect_count`, `share_count`, `comment_count`
   - Video Metadata: `create_time`, `author`, `description`
   - Demographics: `gender_detection` (gender, confidence, method)
   - Hashtags: `hashtag_count`, `generic_hashtag_count`, `specific_hashtag_count`, `generic_ratio`

Note: Temporal structure fields (`start`, `end`, `duration`, `segment_name`) are automatically present in all windows and don't require separate documentation. These will be covered in Phase 3's SystemArchitecture.md.

## 📝 DOCUMENT TEMPLATE

```markdown
# [Feature Category] Features (e.g., Visual Features)

## 📊 Feature Overview Matrix
| Feature Group | Source Services | Temporal Availability | ML Priority | Status |
|---------------|----------------|----------------------|-------------|---------|
| Person Framing | MediaPipe | All windows | HIGH | ✅ Active |
| Creative Density | YOLO, MediaPipe, OCR | All windows | HIGH | ✅ Active |
| Text Overlays | OCR | All windows | MEDIUM | ✅ Active |
| Scene Pacing | Scene Detection | All windows | MEDIUM | ✅ Active |

---

# [Feature Group Name] (e.g., Person Framing)

## 🎯 Feature Purpose & ML Value

### Business Question
What creator behavior does this feature measure?
Example: "How prominently is the creator's face featured in the video?"

### ML Significance
- **Predictive Power**: HIGH/MEDIUM/LOW for engagement prediction
- **Feature Type**: Continuous/Categorical/Binary
- **Correlation with Success**: [Specific insights from data or legacy docs]

### Legacy ML Insights
```
⚠️ VERIFIED: Mining from [legacy_doc.md]
[Preserved valuable ML insights, formulas, patterns]
```

## 📊 Feature Components

### Available Metrics in Temporal Windows
```json
{
  "hook": {
    "close_ratio": 0.0-1.0,      // Categorical distribution
    "medium_ratio": 0.0-1.0,     // Categorical distribution
    "wide_ratio": 0.0-1.0,       // Categorical distribution
    "none_ratio": 0.0-1.0,       // Categorical distribution
    "average_face_size": 0.0-1.0 // Continuous magnitude
  },
  "middle_segments": [...],      // Same metrics
  "closing": {...}               // Same metrics
}
```

### Metric Definitions
⚠️ **ONLY include features that exist in the temporal windows JSON output!**
Check against: `/insights/[video_id]_temporal_windows_updated.json`

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| close_ratio | faces_area>25% / total_frames | 0-1 | Intimate framing percentage |
| medium_ratio | 8%<faces_area<25% / total_frames | 0-1 | Standard framing percentage |
| wide_ratio | faces_area<8% / total_frames | 0-1 | Distant framing percentage |
| none_ratio | no_faces / total_frames | 0-1 | No face visible percentage |
| average_face_size | mean(face_areas) | 0-1 | Overall prominence magnitude |

## 🔄 Data Pipeline

### Source to Feature Flow
```
MediaPipe Service (Phase 1 doc)
    ↓ (face bbox data)
Timeline Builder
    ↓ (face entries)
temporal_compute.py
    ↓ (process_segment function)
Feature Calculation
    ↓
Temporal Windows Output
```

### Implementation Location
```python
# Where feature is calculated
/rumiai_v2/processors/temporal_compute.py
└── process_segment() (lines X-Y)
    └── # Person framing calculation
        └── face_area = bbox.width * bbox.height * 100
```


## 🎨 Feature Engineering Opportunities

### Current Limitations
- Binary classification (close/medium/wide) loses nuance
- No temporal smoothing (frame-by-frame can be noisy)
- Missing velocity metrics (how fast framing changes)

### Proposed Enhancements
- [ ] Add face_size_variance within windows
- [ ] Calculate framing_transitions between segments
- [ ] Implement face_quality_score (blur, occlusion)

## 🔗 Cross-References

### Dependencies (from Phase 1)
- **Primary Service**: MediaPipe (VisionServices.md#MediaPipe)
- **Frame Rate**: 3 FPS sampling
- **Performance Impact**: ~30% of MediaPipe processing time

### Related Features
- **Creative Density**: More faces = higher density
- **Gaze Patterns**: Face presence enables gaze tracking
- **Emotion Analysis**: Requires face detection

### Downstream Usage (for Phase 3)
- Used in ML models: engagement_predictor_v2
- API endpoints: /api/analysis/framing
- Reports: Creator Performance Dashboard

## 📊 Validation & Testing

### Feature Presence Verification
```python
# Verify feature appears in temporal windows
import json
with open('insights/[video_id]_temporal_windows.json') as f:
    data = json.load(f)

# Check all windows have the feature
assert 'close_ratio' in data['temporal_windows']['hook']
assert 'average_face_size' in data['temporal_windows']['hook']
```

### Value Range Validation
```python
# Ensure values are normalized correctly
for window in ['hook', 'closing']:
    ratios = [window['close_ratio'], window['medium_ratio'],
              window['wide_ratio'], window['none_ratio']]
    assert abs(sum(ratios) - 1.0) < 0.01  # Should sum to 1
    assert all(0 <= r <= 1 for r in ratios)
```

## 🚀 Feature Importance Ranking

### For Engagement Prediction
1. **average_face_size in hook**: 0.72 correlation
2. **close_ratio in hook**: 0.65 correlation
3. **framing consistency**: 0.58 correlation

### For Creator Style Classification
1. **wide_ratio in middle**: Best separator for product-focused
2. **close_ratio variance**: Indicates dynamic vs static style

---
```

## 📊 MINING STRATEGY FOR PHASE 2

### What to Extract from Legacy Docs
1. **ML Insights**: Correlation values, optimal ranges
2. **Business Context**: What creator behavior it measures
3. **Mathematical Formulas**: If still valid in temporal_compute
4. **Pattern Descriptions**: Successful content patterns

### What to Verify
1. **Feature Names**: May have changed in temporal_compute
2. **Formulas**: Verify against actual implementation
3. **Availability**: Ensure feature exists in temporal windows
4. **Ranges**: Confirm normalization is correct

### What to Ignore
1. **Precompute implementation details**
2. **6-block structure references**
3. **Old feature names that no longer exist**
4. **Technical implementation** (covered in Phase 1)

## ✅ VALIDATION CHECKLIST

Before marking a feature section complete:

### Feature Verification
- [ ] Feature exists in temporal windows JSON output
- [ ] Feature appears in ONLY ONE of the 4 documents
- [ ] Formula matches temporal_compute.py implementation
- [ ] Value ranges are correct
- [ ] All temporal windows include the feature (or clearly documented if not)

### ML Value Documentation
- [ ] Business question clearly stated
- [ ] ML significance explained
- [ ] Patterns documented with examples
- [ ] Feature importance quantified (if known)

### Cross-References
- [ ] Phase 1 service dependencies noted
- [ ] Related features identified
- [ ] Implementation location verified

## 🤝 USER CONSULTATION POINTS

During Phase 2, consult user when:
1. **Feature missing**: Expected feature not in temporal windows
2. **Formula mismatch**: Legacy formula differs from implementation
3. **Priority unclear**: Which features are most important for ML
4. **New features found**: Temporal_compute has features not in legacy
5. **Pattern validation**: Do these patterns match production insights?

## 📊 COMPLETE FEATURE INVENTORY

### Temporal Window Features (Must ALL be documented)
```
✅ Coverage Checklist - Every feature from the JSON must be checked off:

Visual Features:
□ close_ratio          □ medium_ratio        □ wide_ratio
□ none_ratio          □ average_face_size   □ element_count
□ max_density         □ min_density         □ avg_density
□ overlay_unique_count □ overlay_coverage    □ overlay_persistence
□ has_captions        □ scene_count         □ shortest_scene
□ longest_scene       □ scene_duration_variance □ changes_per_second
□ object_count        □ person_count

Audio Features:
□ speech_coverage     □ word_count          □ has_greeting
□ has_question        □ has_instruction     □ has_speech_cta
□ energy_level        □ energy_variance     □ energy_max
□ burst_pattern       □ avg_pitch_normalized □ pitch_range_norm

Behavioral Features:
□ gesture_count       □ gaze_variance       □ eye_contact_rate
□ expression_count    □ joy_ratio           □ sadness_ratio
□ anger_ratio         □ fear_ratio          □ disgust_ratio
□ surprise_ratio      □ neutral_ratio

Engagement Features (in metadata section):
□ digg_count          □ play_count          □ collect_count
□ share_count         □ comment_count       □ create_time
□ author              □ description         □ gender_detection
□ hashtag_count       □ generic_hashtag_count □ specific_hashtag_count
□ generic_ratio

Temporal Structure:
□ start               □ end                 □ duration
□ segment_name (middle segments only)
```

---

**REMEMBER**:
- Phase 2 is about ML VALUE, not technical implementation
- ONLY document features that exist in the JSON output
- Every JSON feature must be in exactly ONE document
- Preserve valuable insights from legacy docs (but verify existence)
- Document patterns that lead to success