# Dance Feature Detection with Current RumiAI Services

## Executive Summary
RumiAI currently collects all necessary data for dance detection through MediaPipe and Audio Energy services but doesn't extract movement features. This document outlines multiple approaches to add dance/movement analysis using existing infrastructure.

## Current State
- **MediaPipe**: Collects 33 pose landmarks per frame but only uses visibility metrics
- **Audio Energy**: Provides 31 RMS frames/second for rhythm analysis
- **Missing**: Frame-to-frame movement calculations and audio-movement correlation

## Approach 1: Minimal Implementation (4 Features)
*Lowest effort, highest impact - can be added to temporal_compute.py immediately*

### Features to Add:
1. **movement_intensity** (Float 0-1)
   - Calculate Euclidean distance between pose landmarks across consecutive frames
   - Average movement across all 33 body points
   - Normalize by frame rate (higher FPS shouldn't mean higher intensity)

2. **vertical_movement** (Float 0-1)
   - Track hip center Y-coordinate variance over time
   - Detects jumping, bouncing, squatting
   - Key indicator of energetic dance

3. **movement_periodicity** (Float 0-1)
   - FFT on movement intensity signal
   - Identifies repetitive movement patterns
   - High value = likely choreographed/rhythmic movement

4. **has_significant_movement** (Boolean)
   - True if movement_intensity > 0.3 for >50% of segment
   - Simple binary classifier for dance vs static content

### Implementation:
```python
def calculate_minimal_dance_features(poses, window_duration):
    if len(poses) < 2:
        return {
            'movement_intensity': 0.0,
            'vertical_movement': 0.0,
            'movement_periodicity': 0.0,
            'has_significant_movement': False
        }

    # Extract movement between frames
    movements = []
    hip_positions = []

    for i in range(1, len(poses)):
        prev_pose = poses[i-1]['landmarks']
        curr_pose = poses[i]['landmarks']

        # Calculate average movement
        total_movement = 0
        for j in range(33):  # 33 pose landmarks
            dist = euclidean_distance(prev_pose[j], curr_pose[j])
            total_movement += dist

        movements.append(total_movement / 33)

        # Track hip center
        hip_y = (curr_pose[23].y + curr_pose[24].y) / 2
        hip_positions.append(hip_y)

    # Calculate features
    movement_intensity = np.mean(movements)
    vertical_movement = np.std(hip_positions)

    # Periodicity via FFT
    if len(movements) > 10:
        fft = np.fft.fft(movements)
        freqs = np.fft.fftfreq(len(movements))
        peak_freq = freqs[np.argmax(np.abs(fft[1:]))+1]
        movement_periodicity = min(abs(peak_freq) * 10, 1.0)  # Scale to 0-1
    else:
        movement_periodicity = 0.0

    has_significant_movement = movement_intensity > 0.3

    return {
        'movement_intensity': round(movement_intensity, 4),
        'vertical_movement': round(vertical_movement, 4),
        'movement_periodicity': round(movement_periodicity, 4),
        'has_significant_movement': has_significant_movement
    }
```

## Approach 2: Rhythm-Synchronized Features (8 Features)
*Combines movement with audio for rhythm detection*

### Additional Features:
5. **beat_sync_score** (Float 0-1)
   - Correlate movement peaks with audio energy peaks
   - High correlation = movements match music beat
   - Critical for identifying "on-beat" dancing

6. **movement_audio_correlation** (Float -1 to 1)
   - Pearson correlation between movement intensity and audio RMS
   - Indicates if movement follows music dynamics

7. **rhythm_consistency** (Float 0-1)
   - Variance in time between movement peaks
   - Low variance = consistent rhythm
   - High variance = freestyle or no rhythm

8. **anticipated_beat_ratio** (Float 0-1)
   - Percentage of movement peaks that occur 50-100ms before audio peaks
   - Professional dancers often move slightly ahead of beat

### Implementation:
```python
def calculate_rhythm_features(poses, audio_energy, fps=30):
    movement_signal = extract_movement_signal(poses)
    audio_signal = audio_energy['rms_frames']

    # Resample to same rate (audio is 31fps, video is variable)
    movement_resampled = resample_signal(movement_signal, len(audio_signal))

    # Find peaks
    movement_peaks = find_peaks(movement_resampled, height=0.3)
    audio_peaks = find_peaks(audio_signal, height=np.mean(audio_signal))

    # Calculate beat synchronization
    beat_sync = calculate_peak_alignment(movement_peaks, audio_peaks)

    # Correlation
    correlation = np.corrcoef(movement_resampled, audio_signal)[0,1]

    # Rhythm consistency
    if len(movement_peaks) > 2:
        peak_intervals = np.diff(movement_peaks)
        rhythm_consistency = 1 - (np.std(peak_intervals) / np.mean(peak_intervals))
    else:
        rhythm_consistency = 0.0

    # Anticipated beats (dancer moves before beat)
    anticipated = count_anticipated_peaks(movement_peaks, audio_peaks,
                                         anticipation_window=0.1*fps)
    anticipated_ratio = anticipated / len(movement_peaks) if movement_peaks else 0

    return {
        'beat_sync_score': beat_sync,
        'movement_audio_correlation': correlation,
        'rhythm_consistency': rhythm_consistency,
        'anticipated_beat_ratio': anticipated_ratio
    }
```

## Approach 3: Advanced Body Analysis (12 Features)
*Detailed pose analysis for professional dance detection*

### Additional Features:
9. **body_symmetry** (Float 0-1)
   - Compare left and right side poses
   - Choreographed dances often have symmetric movements
   - Calculate: 1 - mean_distance(left_side, mirrored_right_side)

10. **limb_extension** (Float 0-1)
    - Average extension of arms and legs from body center
    - Fully extended = 1.0, close to body = 0.0
    - Indicates expansive vs contained movement

11. **pose_diversity** (Float 0-1)
    - Number of unique pose clusters / total frames
    - High diversity = complex choreography
    - Low diversity = repetitive or minimal movement

12. **movement_smoothness** (Float 0-1)
    - Inverse of movement acceleration variance
    - Smooth transitions vs jerky movements
    - Professional dance has controlled, smooth transitions

### Implementation:
```python
def calculate_body_features(poses):
    symmetry_scores = []
    extensions = []
    pose_clusters = []
    accelerations = []

    for i, pose in enumerate(poses):
        landmarks = pose['landmarks']

        # Body symmetry
        left_indices = [11, 13, 15, 23, 25, 27, 29, 31]
        right_indices = [12, 14, 16, 24, 26, 28, 30, 32]

        symmetry = calculate_symmetry(landmarks[left_indices],
                                     landmarks[right_indices])
        symmetry_scores.append(symmetry)

        # Limb extension
        center = calculate_body_center(landmarks)
        avg_extension = calculate_avg_limb_distance(landmarks, center)
        extensions.append(avg_extension)

        # Pose clustering (simplified)
        pose_vector = flatten_landmarks(landmarks)
        pose_clusters.append(pose_vector)

        # Movement smoothness (requires 3+ frames)
        if i >= 2:
            accel = calculate_acceleration(poses[i-2:i+1])
            accelerations.append(accel)

    # Cluster poses to find unique positions
    unique_poses = len(set(cluster_poses(pose_clusters, eps=0.1)))
    pose_diversity = unique_poses / len(poses)

    # Smoothness from acceleration consistency
    movement_smoothness = 1 - (np.std(accelerations) / (np.mean(accelerations) + 0.001))

    return {
        'body_symmetry': np.mean(symmetry_scores),
        'limb_extension': np.mean(extensions),
        'pose_diversity': pose_diversity,
        'movement_smoothness': movement_smoothness
    }
```

## Approach 4: Semantic Dance Classification (4 Categories)
*High-level dance type identification*

### Features:
13. **dance_style_category** (Categorical)
    - Categories: "none", "freestyle", "choreographed", "partner"
    - Based on movement patterns and symmetry

14. **dance_confidence** (Float 0-1)
    - Weighted combination of all dance indicators
    - Threshold at 0.5 for binary "is_dancing" classification

15. **movement_energy_level** (Categorical)
    - "static", "low", "medium", "high"
    - Based on movement intensity and vertical movement

16. **sync_quality** (Categorical)
    - "no_music", "off_beat", "on_beat", "professional"
    - Based on beat_sync_score thresholds

### Implementation:
```python
def classify_dance(all_features):
    # Dance style classification
    if all_features['movement_intensity'] < 0.1:
        dance_style = 'none'
    elif all_features['body_symmetry'] > 0.7 and all_features['rhythm_consistency'] > 0.6:
        dance_style = 'choreographed'
    elif all_features['person_count'] > 1 and all_features['movement_intensity'] > 0.3:
        dance_style = 'partner'
    else:
        dance_style = 'freestyle'

    # Dance confidence (weighted combination)
    dance_confidence = (
        all_features['movement_intensity'] * 0.2 +
        all_features['beat_sync_score'] * 0.3 +
        all_features['movement_periodicity'] * 0.2 +
        all_features['rhythm_consistency'] * 0.2 +
        all_features['body_symmetry'] * 0.1
    )

    # Energy level
    if all_features['movement_intensity'] < 0.1:
        energy_level = 'static'
    elif all_features['movement_intensity'] < 0.3:
        energy_level = 'low'
    elif all_features['movement_intensity'] < 0.6:
        energy_level = 'medium'
    else:
        energy_level = 'high'

    # Sync quality
    if all_features.get('beat_sync_score', 0) < 0.2:
        sync_quality = 'no_music'
    elif all_features['beat_sync_score'] < 0.5:
        sync_quality = 'off_beat'
    elif all_features['beat_sync_score'] < 0.8:
        sync_quality = 'on_beat'
    else:
        sync_quality = 'professional'

    return {
        'dance_style_category': dance_style,
        'dance_confidence': round(dance_confidence, 4),
        'movement_energy_level': energy_level,
        'sync_quality': sync_quality
    }
```

## Integration Plan

### Phase 1: Minimal Implementation (1-2 days)
1. Add movement calculation to MediaPipe processing
2. Store pose deltas in timeline
3. Add 4 basic features to temporal_compute.py
4. Test on videos with known dance content

### Phase 2: Rhythm Features (2-3 days)
1. Implement peak detection for audio and movement
2. Add correlation calculations
3. Add rhythm features to temporal windows
4. Validate against dance vs non-dance videos

### Phase 3: Advanced Features (3-5 days)
1. Implement pose clustering
2. Add body symmetry calculations
3. Add semantic classification
4. Full testing and optimization

## Performance Considerations

### Computational Cost:
- **Minimal features**: +0.5-1s per video (negligible)
- **Rhythm features**: +2-3s per video (peak detection)
- **Advanced features**: +5-8s per video (clustering)
- **All features**: +10-12s per video total

### Memory Impact:
- Pose landmarks already in memory
- Additional arrays for movement signals: ~100KB per video
- Clustering may require ~1MB for temporary storage

### Optimization Options:
1. **Subsample frames**: Process every 3rd frame for faster computation
2. **Parallel processing**: Calculate features in parallel with other services
3. **GPU acceleration**: Use CuPy for numpy operations if available
4. **Caching**: Store computed movement signals for reuse

## Validation Strategy

### Test Dataset Requirements:
- 20 videos with professional dance
- 20 videos with casual movement
- 20 videos with no movement (talking heads)
- 20 videos with camera movement but static subject

### Success Metrics:
- Dance classification accuracy > 85%
- Beat synchronization correlation > 0.7 for dance videos
- False positive rate < 10% for non-dance content
- Processing time increase < 15%

## Recommended Implementation Order

1. **Start with Approach 1** (Minimal - 4 features)
   - Immediate value with minimal effort
   - Can be deployed within days

2. **Add Approach 2** (Rhythm - 4 more features)
   - High value for TikTok dance content
   - Leverages audio data we already have

3. **Consider Approach 3** if ML models need it
   - More complex, longer implementation
   - Wait for ML training feedback

4. **Add Approach 4** for creator reports
   - Semantic categories useful for reports
   - Build on top of other features

## Code Location Changes

### Files to Modify:
1. `/rumiai_v2/processors/temporal_compute.py`
   - Add `calculate_dance_features()` function
   - Call from `aggregate_segment_features()`

2. `/rumiai_v2/processors/timeline_builder.py`
   - Store pose landmarks in timeline (currently discarded)
   - Add movement delta entries

3. `/rumiai_v2/api/ml_services_unified.py`
   - Return full pose landmarks, not just visibility
   - Optional: pre-calculate movement deltas

### New Files to Create:
1. `/rumiai_v2/utils/dance_detection.py`
   - All dance-specific calculations
   - Peak detection and correlation functions
   - Pose clustering utilities

## Conclusion

RumiAI already has all the raw data needed for comprehensive dance detection. By adding movement calculations to our existing MediaPipe and Audio Energy data, we can extract 12-16 dance-specific features with minimal changes to the codebase. The recommended minimal implementation (4 features) can be completed in 1-2 days and would significantly improve our ability to analyze TikTok's dance content.