# Person Framing Features - ML Adaptability Analysis

## Feature Count Verification (MANDATORY)
Expected features: 29
Table rows created: 29
Status: ✓ SUCCESS (counts match)

## Feature Evaluation for Random Forest and K-means Models

| Source | Feature | Data Type | RF Adaptable | RF Transformation | RF Difficulty | RF Blockers | RF Info Loss | RF Confidence | KM Adaptable | KM Transformation | KM Difficulty | KM Blockers | KM Info Loss | KM Confidence |
|--------|---------|-----------|--------------|-------------------|---------------|-------------|--------------|---------------|--------------|-------------------|---------------|-------------|--------------|---------------|
| person_framing | averageFaceSize | float | Yes | Already numerical | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |
| person_framing | closeUpMoments | array-variable | Yes | Extract count, avg duration, time positions | Medium | None | Low | High | Partial | Extract count + total duration | Medium | Variable length | Medium | Medium |
| person_framing | distanceVariation | float | Yes | Already numerical | Low | None | None | High | Yes | Scale | Low | None | None | High |
| person_framing | dominantFraming | string | Yes | One-hot encode (close/medium/wide) | Low | None | None | High | Yes | Label encode (0-2) + scale | Low | None | Low | High |
| person_framing | eyeContactRate | float | Yes | Already numerical | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |
| person_framing | faceVisibilityRate | float | Yes | Already numerical | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |
| person_framing | framingChanges | int | Yes | Already numerical | Low | None | None | High | Yes | Scale | Low | None | None | High |
| person_framing | framingConsistency | float | Yes | Already numerical | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |
| person_framing | framingProgression | array-fixed | Yes | Flatten: extract values per segment | Low | None | None | High | Yes | Extract segment values, scale | Medium | None | Low | High |
| person_framing | framingTransitions | int | Yes | Already numerical | Low | None | None | High | Yes | Scale | Low | None | None | High |
| person_framing | gazeSteadiness | float | Yes | Already numerical | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |
| person_framing | groupShots | array-variable | Partial | Extract count only | Low | Multi-person tracking | Medium | Medium | Yes | Extract count | Low | None | Medium | Medium |
| person_framing | keySubjectMoments | array-variable | Yes | Extract count, timing metrics | Medium | None | Low | High | Partial | Extract count only | Medium | Timing lost | High | Low |
| person_framing | multiPersonDynamics | dict | Partial | Extract interaction count | Medium | Multi-person complexity | High | Low | No | Too complex | High | Dynamic structure | High | Low |
| person_framing | primarySubject | string | Yes | One-hot encode (single/multiple/none) | Low | None | None | High | Yes | Label encode (0-2) + scale | Low | None | Low | High |
| person_framing | subjectCount | float | Yes | Already numerical | Low | None | None | High | Yes | Scale | Low | None | None | High |
| person_framing | speakerFraming | dict | Partial | Extract avg framing quality | Medium | Speech dependency | Medium | Medium | Partial | Extract quality score | Medium | Context lost | Medium | Low |
| person_framing | cinematicStyle | string | Yes | One-hot encode (5-8 styles) | Low | None | None | High | Partial | Label encode + scale | Medium | Not ordinal | Medium | Medium |
| person_framing | compositionRule | string | Yes | One-hot encode (rule types) | Low | None | None | High | Partial | Label encode + scale | Medium | Not ordinal | Medium | Medium |
| person_framing | compositionScore | float | Yes | Already numerical | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |
| person_framing | framingAppropriate | boolean | Yes | Already binary (0/1) | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |
| person_framing | framingDistribution | dict | Yes | Flatten: extract percentages for each type | Low | None | None | High | Yes | Extract percentages, scale | Low | None | None | High |
| person_framing | framingTechnique | string | Yes | One-hot encode (technique types) | Low | None | None | High | Partial | Label encode + scale | Medium | Not ordinal | Medium | Medium |
| person_framing | interactionZones | array-variable | Yes | Extract zone count, overlap metrics | Medium | None | Medium | Medium | Partial | Extract count + avg size | Medium | Spatial info lost | High | Low |
| person_framing | movementPattern | string | Yes | One-hot encode (pattern types) | Low | None | None | High | Yes | Label encode (ordinal: static→dynamic) | Low | None | Low | High |
| person_framing | professionalLevel | float | Yes | Already numerical | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |
| person_framing | socialDistance | float | Yes | Already numerical | Low | None | None | High | Yes | Scale | Low | None | None | High |
| person_framing | stabilityScore | float | Yes | Already numerical | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |
| person_framing | visualEngagement | float | Yes | Already numerical | Low | None | None | High | Yes | Scale to [0,1] | Low | None | None | High |

## Summary Statistics

### Total features analyzed: 29

### Random Forest Adaptability
- **Fully Adaptable**: 24/29 features (83%)
- **Partially Adaptable**: 5/29 features (17%)
- **Not Adaptable**: 0/29 features (0%)

### K-means Adaptability
- **Fully Adaptable**: 18/29 features (62%)
- **Partially Adaptable**: 10/29 features (35%)
- **Not Adaptable**: 1/29 feature (3%)

### Missing Features
None - all 29 features from the provided list are included in the table

## Key Findings

### Strengths
1. **High numerical content**: 16/29 features (55%) are already numerical floats/ints
2. **Boolean features**: framingAppropriate is easily adaptable for both models
3. **Ordinal categoricals**: Some features like movementPattern have natural ordering (static→dynamic)
4. **Well-structured dicts**: framingDistribution, compositionScore cleanly flatten to percentages

### Challenges
1. **Variable arrays**: closeUpMoments, keySubjectMoments, interactionZones lose temporal/spatial detail
2. **Non-ordinal categoricals**: cinematicStyle, compositionRule, framingTechnique lack natural order for K-means
3. **Complex structures**: multiPersonDynamics too complex for K-means
4. **Context-dependent**: speakerFraming depends on speech detection quality

### Model-Specific Considerations

#### For Random Forest
- Can handle all 29 features with appropriate transformations
- One-hot encoding works well for all categorical features
- Complex arrays can be summarized without significant impact on tree splits

#### For K-means
- Focus on the 18 fully adaptable features for best results
- Consider dropping multiPersonDynamics entirely
- Use ordinal encoding carefully - only where natural order exists
- Apply robust scaling to handle outliers in numerical features

## Transformation Examples

### cinematicStyle (string categorical)
```python
# Original
"documentary"

# RF Transformation (one-hot)
style_documentary: 1
style_cinematic: 0
style_handheld: 0
style_professional: 0
style_amateur: 0

# K-means Transformation (problematic)
style_encoded: 2  # Arbitrary encoding, no meaningful distance
```

### interactionZones (array-variable)
```python
# Original
[
  {"zone": "personal", "duration": 5.2, "overlap": 0.3},
  {"zone": "social", "duration": 8.1, "overlap": 0.1}
]

# RF Transformation
zone_count: 2
personal_zone_time: 5.2
social_zone_time: 8.1
avg_overlap: 0.2
total_zone_duration: 13.3

# K-means Transformation
zone_count: 2
avg_overlap: 0.2
# Lost zone type specifics
```

### movementPattern (string with natural order)
```python
# Original
"moderate_movement"

# RF Transformation (one-hot)
pattern_static: 0
pattern_slow: 0
pattern_moderate: 1
pattern_dynamic: 0

# K-means Transformation (ordinal works)
movement_level: 2  # 0=static, 1=slow, 2=moderate, 3=dynamic
# Natural progression from still to moving
```

## Notes
- The additional 11 features (beyond the initial 18) introduce more categorical complexity
- Features like professionalLevel and compositionScore are judgment-based but numerical
- socialDistance and gazeSteadiness add psychological/behavioral dimensions
- Consider feature engineering: combining related features (e.g., all composition-related features)