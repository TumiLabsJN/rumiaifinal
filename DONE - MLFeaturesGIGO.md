# ML Features GIGO (Garbage In, Garbage Out) - Feature Removal Plan

## Executive Summary
Remove redundant features from our current unified JSON output.

## Features to Remove

### Person Framing features
We currently generate 5 features for person framing:
1. `close_ratio` - Percentage of time face area > 25%
2. `medium_ratio` - Percentage of time face area 8-25%
3. `wide_ratio` - Percentage of time face area 0-8%
4. `none_ratio` - Percentage of time no face detected
5. `average_face_size` - Mean face area as fraction of frame (0-1)

#### Why This Is Problematic

##### 1. Perfect Multicollinearity
```python
close_ratio + medium_ratio + wide_ratio + none_ratio = 1.0  # ALWAYS
```
This violates fundamental ML assumptions and causes:
- Unstable model coefficients
- Overfitting
- Reduced interpretability
- Singular matrix errors in some algorithms

##### 2. Information Redundancy
The 4 ratios are just discretized versions of `average_face_size`:
```python
# Current semantic bucketing
if face_area > 0.25:    → close_ratio++
elif face_area > 0.08:   → medium_ratio++
elif face_area > 0:      → wide_ratio++
else:                    → none_ratio++
```

##### 3. Arbitrary Thresholds
The 25% and 8% thresholds are arbitrary and may not align with:
- Actual viral content patterns
- Different video styles (dance vs tutorial vs product)
- Cultural differences in framing preferences

#### Proposed Solution

##### Remove These Features
```json
{
  "close_ratio": 0.6,
  "medium_ratio": 0.3,
  "wide_ratio": 0.1,
  "none_ratio": 0.0
}
```
**Action**: Remove these from temporal_windows output

##### Keep Only This Feature
```json
{
  "average_face_size": 0.1875
}
```
**Action**: Keep only this feature (continuous 0-1)

#### Implementation Plan

##### Step 1: Update temporal_compute.py
```python
# BEFORE (line 1450)
framing_dist = {
    'close_ratio': framing_counts['close'] / total_framing if total_framing > 0 else 0,
    'medium_ratio': framing_counts['medium'] / total_framing if total_framing > 0 else 0,
    'wide_ratio': framing_counts['wide'] / total_framing if total_framing > 0 else 0,
    'none_ratio': framing_counts['none'] / total_framing if total_framing > 0 else 0
}

# AFTER
# Simply remove the framing_dist calculation and inclusion
```

##### Step 2: Update Window Features Dictionary
```python
# BEFORE
window_features = {
    **emotion_dist,
    **framing_dist,  # Remove this line
    'average_face_size': round(average_face_size, 4),
    ...
}

# AFTER
window_features = {
    **emotion_dist,
    'average_face_size': round(average_face_size, 4),  # Keep only this for framing
    ...
}
```

#### Step 3: Update Tests
Update test files to expect only `average_face_size`:
- `/test_temporal_compute_v2.py`
- `/test_person_framing_v2.py`
- `/test_average_face_size.py`

---

### Creative Density Features
We currently generate 4 derived features for visual complexity:
1. `element_count` - Sum of all visual elements (objects + gestures + expressions + scenes + text)
2. `avg_density` - element_count / duration (redundant calculation)
3. `max_density` - Peak per-second element count (useful, but see note)
4. `min_density` - Minimum per-second element count (useful, but see note)

Plus the component features that are already in the output:
- `object_count` - YOLO detected objects
- `gesture_count` - MediaPipe gestures
- `expression_count` - FEAT expressions
- `scene_count` - Scene changes
- `overlay_unique_count` - Text overlays

#### Why This Is Problematic

##### 1. Pure Derivative Features
```python
element_count = object_count + gesture_count + expression_count + scene_count + text_count
avg_density = element_count / duration  # Double derivative!
```
These add zero new information - ML models can learn these relationships.

##### 2. Forces Equal Weighting
By pre-summing, we assume all element types contribute equally to "complexity". But maybe:
- Gestures matter more than objects for engagement
- Scene changes are more impactful than text overlays
- The interaction between elements matters more than the sum

##### 3. Redundant Computation
`avg_density` is just `element_count / duration`, which the ML model can compute if needed.

#### Proposed Solution for Density Features

##### Remove These Features
```json
{
  "element_count": 22,
  "avg_density": 4.5
}
```
**Action**: Remove these from temporal_windows output

##### Keep These Features
```json
{
  "object_count": 5,
  "gesture_count": 3,
  "expression_count": 8,
  "scene_count": 2,
  "overlay_unique_count": 4,
  "max_density": 8,
  "min_density": 2
}
```
**Action**: Keep all these features

#### Why Keep max_density and min_density?
These capture **temporal variation within windows** that cannot be reconstructed from window-level counts:
- A steady 5 elements throughout: `max=5, min=5`
- A burst of 50 elements then nothing: `max=50, min=0`
Both scenarios have same total count but different patterns!

#### Implementation Plan for Density Features

##### Step 1: Update temporal_compute.py
```python
# BEFORE (lines 1292-1293, 1320)
total_elements = (total_unique_texts +
                 object_count + gesture_count + expression_count + scene_count)
avg_density = total_elements / duration if duration > 0 else 0

# AFTER
# Remove element_count calculation entirely
# Remove avg_density calculation
# Keep max_density and min_density calculations as they capture unique temporal info
```

##### Step 2: Update Window Features Dictionary
```python
# BEFORE
window_features = {
    ...
    'object_count': object_count,
    'gesture_count': gesture_count,
    'expression_count': expression_count,
    'scene_count': scene_count,
    'element_count': total_elements,  # Remove this line
    'max_density': max_density,
    'min_density': min_density,
    'avg_density': avg_density,       # Remove this line
    ...
}

# AFTER
window_features = {
    ...
    'object_count': object_count,      # Keep components
    'gesture_count': gesture_count,
    'expression_count': expression_count,
    'scene_count': scene_count,
    'overlay_unique_count': overlay_unique_count,
    'max_density': max_density,        # Keep temporal variation
    'min_density': min_density,
    ...
}
```

##### Step 3: Update Density Tests
- `/test_temporal_compute_v2.py` - Remove element_count and avg_density assertions
- Update TotalFeatures.md documentation

---

### Scene Pacing Features
We currently generate a derivative feature for editing pace:
1. `changes_per_second` - scene_count / duration

Plus the raw features that are already in the output:
- `scene_count` - Number of scene changes in the segment
- `duration` - Length of the temporal window/segment
- `shortest_scene` - Minimum scene duration
- `longest_scene` - Maximum scene duration
- `scene_duration_variance` - Statistical variance of scene lengths

#### Why changes_per_second Is Problematic

##### 1. Pure Arithmetic Derivative
```python
changes_per_second = scene_count / duration  # Just division!
```
The ML model has both inputs and can compute this if needed.

##### 2. Forces Linear Relationship
By pre-computing the division, we assume the relationship is linear. But maybe:
- Log relationship works better: `log(scene_count) / duration`
- Square root scaling: `scene_count / sqrt(duration)`
- Interaction with other features: `(scene_count * energy_level) / duration`

##### 3. No Information Gain
This adds zero new information - it's just a convenience calculation that the ML model can learn.

#### Proposed Solution for Scene Pacing

##### Remove This Feature
```json
{
  "changes_per_second": 0.5
}
```
**Action**: Remove this from temporal_windows output

##### Keep These Features
```json
{
  "scene_count": 5,
  "duration": 10,
  "shortest_scene": 0.5,
  "longest_scene": 3.2,
  "scene_duration_variance": 1.8
}
```
**Action**: Keep all these features

##### Why Keep scene_duration_variance?
Unlike `changes_per_second`, variance cannot be reconstructed from other features:
- Requires ALL individual scene durations
- We only provide min/max, not the full distribution
- Captures pacing consistency that min/max alone cannot show

#### Implementation Plan for Scene Pacing

##### Step 1: Update temporal_compute.py
```python
# BEFORE (line 1321)
changes_per_second = scene_count / duration if duration > 0 else 0

# AFTER
# Remove changes_per_second calculation entirely
```

##### Step 2: Update Window Features Dictionary
```python
# BEFORE
window_features = {
    ...
    'shortest_scene': shortest_scene,
    'longest_scene': longest_scene,
    'scene_duration_variance': scene_duration_variance,
    'changes_per_second': changes_per_second,  # Remove this line
    ...
}

# AFTER
window_features = {
    ...
    'shortest_scene': shortest_scene,
    'longest_scene': longest_scene,
    'scene_duration_variance': scene_duration_variance,
    # changes_per_second removed - ML can compute from scene_count/duration
    ...
}
```

##### Step 3: Update Scene Pacing Tests
- Remove changes_per_second assertions from tests
- Verify scene_count and duration are still present for ML to use

---

### Semantic Speech Features
We currently generate 4 binary features from keyword pattern matching:
1. `has_greeting` - Looks for "hey", "hello", "welcome" in first 50 chars
2. `has_question` - Searches for "?" or question words like "how", "what", "why"
3. `has_instruction` - Detects instructional words like "first", "step", "make sure"
4. `has_speech_cta` - Finds CTAs like "subscribe", "follow", "link in bio"

Plus the actual useful speech features:
- `speech_coverage` - Percentage of window with speech (0-1)
- `word_count` - Total words spoken (information density)

#### Why Semantic Features Are Problematic

##### 1. Arbitrary Keyword Lists
```python
greetings = ['hey', 'hello', 'hi ', 'welcome', "what's up", 'good morning']
# But what about: "yo", "sup", "ladies and gentlemen", "fam", cultural greetings?
```
These pre-defined lists miss 70%+ of actual patterns and impose English-centric bias.

##### 2. Context Blindness
Same words have different meanings:
- "Follow me" (CTA) vs "Follow these steps" (instruction)
- "Check this out" (CTA) vs "Check your settings" (instruction)
- "What's up" (greeting) vs "What's up with that?" (question)

##### 3. Worse Than Nothing
These features create harmful false signals:
- `has_greeting = 0` doesn't mean no greeting, just means we missed it
- Models learn spurious correlations from incomplete detection
- Wastes model capacity on noise instead of real patterns

##### 4. Incompatible with RF/K-means
Random Forest and K-means cannot:
- Process raw text to discover patterns
- Learn which words actually matter
- Understand context or semantics
So we're stuck with bad keyword matching that misleads more than helps.

#### Proposed Solution for Speech Features

##### Remove These Features
```json
{
  "has_greeting": 1,
  "has_question": 1,
  "has_instruction": 1,
  "has_speech_cta": 1
}
```
**Action**: Remove all these from temporal_windows output

##### Keep These Features
```json
{
  "speech_coverage": 0.75,
  "word_count": 127
}
```
**Action**: Keep both these features

##### Why Not Replace with Text Samples?
RF/K-means cannot process text strings. Options like:
- Providing first/last N words - Can't be used
- Better NLP models - Incompatible with RF/K-means
- Embeddings - Requires different ML architecture

Given these constraints, it's better to remove bad features than keep misleading ones.

#### Implementation Plan for Speech Features

##### Step 1: Remove calculate_speech_content_indicators Function
```python
# REMOVE ENTIRE FUNCTION (around line 600-650 in temporal_compute.py)
def calculate_speech_content_indicators(speech_segments, start, end, duration):
    # DELETE THIS ENTIRE FUNCTION
```

##### Step 2: Update Window Features Dictionary
```python
# BEFORE
window_features = {
    ...
    'speech_coverage': speech_coverage,
    'word_count': word_count,
    **speech_content_indicators,  # Remove this line
    ...
}

# AFTER
window_features = {
    ...
    'speech_coverage': speech_coverage,
    'word_count': word_count,
    # Semantic speech features removed - arbitrary keywords worse than nothing
    ...
}
```

##### Step 3: Remove Function Call
```python
# BEFORE (around line 500)
speech_content_indicators = calculate_speech_content_indicators(
    speech_segments, start, end, duration
)

# AFTER
# Removed - semantic keyword matching creates false signals
```

##### Step 4: Update Tests
- Remove assertions for has_greeting, has_question, has_instruction, has_speech_cta
- Document that text analysis is not supported with RF/K-means architecture

---

### Audio Energy Features
We currently generate a semantic classification for audio patterns:
1. `burst_pattern` - Classifies energy distribution as "steady", "burst", "fade", or "variable"

Plus the actual numeric features that contain all the information:
- `energy_level` - Mean audio energy (RMS)
- `energy_variance` - Variance in audio energy
- `energy_max` - Peak audio energy

#### Why burst_pattern Is Problematic

##### 1. Arbitrary Algorithm
```python
def calculate_burst_pattern_for_window(window_rms):
    # Divides window into thirds
    third_size = len(window_rms) // 3
    front_avg = np.mean(window_rms[:third_size])
    middle_avg = np.mean(window_rms[third_size:2*third_size])
    back_avg = np.mean(window_rms[2*third_size:])

    # Arbitrary thresholds
    if front_avg > middle_avg * 1.5:  # Why 1.5?
        return "burst"
    # etc...
```
Why thirds? Why 1.5x ratio? These are arbitrary choices that may not match actual patterns.

##### 2. Information Loss
Reduces rich temporal energy curves to just 4 categories:
- **"steady"** - But how steady? Variance already tells us this
- **"burst"** - Where? How strong? Max and variance show this
- **"fade"** - Linear? Exponential? Lost in categorization
- **"variable"** - Meaningless catch-all category

##### 3. Redundant with Numeric Features
ML models can learn from energy_level, variance, and max:
- High variance → dynamic audio
- High max vs mean → has spikes/bursts
- Low variance → steady audio
- Patterns across segments → pacing changes

##### 4. One-Hot Encoding Overhead
In Random Forest, this becomes 4 binary features instead of letting the model learn from the continuous values.

#### Proposed Solution for Audio Energy

##### Remove This Feature
```json
{
  "burst_pattern": "variable"
}
```
**Action**: Remove this from temporal_windows output

##### Keep These Features
```json
{
  "energy_level": 0.73,
  "energy_variance": 0.24,
  "energy_max": 0.95
}
```
**Action**: Keep all these features

##### Why Raw Metrics Are Better
The ML model can learn complex relationships:
- Maybe `energy_max / energy_level` ratio matters more
- Maybe `sqrt(variance) * energy_level` predicts engagement
- Maybe the pattern differs by video duration

Pre-categorizing into "burst" prevents discovering these patterns.

#### Implementation Plan for Audio Energy

##### Step 1: Remove calculate_burst_pattern_for_window Function
```python
# REMOVE ENTIRE FUNCTION (around line 250-280 in temporal_compute.py)
def calculate_burst_pattern_for_window(window_rms: np.ndarray) -> str:
    # DELETE THIS ENTIRE FUNCTION
```

##### Step 2: Update calculate_audio_energy_for_windows
```python
# BEFORE
results[f'{window_name}_energy_level'] = float(np.mean(window_rms))
results[f'{window_name}_energy_variance'] = float(np.var(window_rms))
results[f'{window_name}_energy_max'] = float(np.max(window_rms))
results[f'{window_name}_burst_pattern'] = calculate_burst_pattern_for_window(window_rms)

# AFTER
results[f'{window_name}_energy_level'] = float(np.mean(window_rms))
results[f'{window_name}_energy_variance'] = float(np.var(window_rms))
results[f'{window_name}_energy_max'] = float(np.max(window_rms))
# burst_pattern removed - arbitrary categorization loses information
```

##### Step 3: Update process_segment Function
```python
# BEFORE (around line 520)
burst_pattern = calculate_burst_pattern_for_window(segment_rms)

# AFTER
# Removed - let ML learn patterns from energy metrics
```

##### Step 4: Update Window Features Dictionary
```python
# BEFORE
window_features = {
    ...
    'energy_level': energy_level,
    'energy_variance': energy_variance,
    'energy_max': energy_max,
    'burst_pattern': burst_pattern,  # Remove this line
    ...
}

# AFTER
window_features = {
    ...
    'energy_level': energy_level,
    'energy_variance': energy_variance,
    'energy_max': energy_max,
    # burst_pattern removed - ML learns from numeric features
    ...
}
```

##### Step 5: Update Hook and Closing Data
```python
# BEFORE
hook_data['burst_pattern'] = audio_energy_metrics.get('hook_burst_pattern', 'steady')
closing_data['burst_pattern'] = audio_energy_metrics.get('closing_burst_pattern', 'steady')

# AFTER
# Remove both lines - burst_pattern no longer needed
```

---

### Emotion Distribution Features
We currently generate 7 ratio features for emotion distribution:
1. `joy_ratio` - Percentage of detections showing joy
2. `sadness_ratio` - Percentage of detections showing sadness
3. `anger_ratio` - Percentage of detections showing anger
4. `fear_ratio` - Percentage of detections showing fear
5. `disgust_ratio` - Percentage of detections showing disgust
6. `surprise_ratio` - Percentage of detections showing surprise
7. `neutral_ratio` - Percentage of detections showing neutral

Plus the raw count that's already in the output:
- `expression_count` - Total number of emotion detections

#### Why This Is Problematic

##### 1. Perfect Multicollinearity
```python
joy_ratio + sadness_ratio + anger_ratio + fear_ratio +
disgust_ratio + surprise_ratio + neutral_ratio = 1.0  # ALWAYS
```
This is identical to the framing ratio problem - creates unstable ML models.

##### 2. Information Redundancy
All ratios are derived from the same emotion counts:
```python
# Current calculation
emotion_counts = {'joy': 3, 'sadness': 1, 'neutral': 2}
total = 6
joy_ratio = 3/6 = 0.5
sadness_ratio = 1/6 = 0.167
neutral_ratio = 2/6 = 0.333
# They always sum to 1.0
```

##### 3. Limited Samples Per Window
With our sampling rates:
- Short videos (≤30s): 6 frames per 3-second window
- Medium videos (30-60s): 3 frames per 3-second window
- Long videos (>60s): ~2 frames per 3-second window

Ratios from 2-6 samples are statistically questionable.

#### Proposed Solution

##### Remove These Features
```json
{
  "joy_ratio": 0.5,
  "sadness_ratio": 0.167,
  "anger_ratio": 0.0,
  "fear_ratio": 0.0,
  "disgust_ratio": 0.0,
  "surprise_ratio": 0.0,
  "neutral_ratio": 0.333
}
```
**Action**: Remove all these from temporal_windows output

##### Keep Only This Feature (Already Exists)
```json
{
  "expression_count": 6
}
```
**Action**: Keep this feature (raw count of detections)

#### Implementation Plan

##### Step 1: Update temporal_compute.py
```python
# BEFORE (lines 1399-1410)
all_emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']
emotion_counts = {emotion: 0 for emotion in all_emotions}

for e in segment_expressions:
    emotion = e.get('emotion', 'neutral')
    if emotion in emotion_counts:
        emotion_counts[emotion] += 1

total_emotions = len(segment_expressions)
emotion_dist = {
    f'{emotion}_ratio': count / total_emotions if total_emotions > 0 else 0
    for emotion, count in emotion_counts.items()
}

# AFTER
# Remove emotion_dist calculation entirely
# IMPORTANT: Keep emotion_counts calculation - it's needed for the new features
# The emotion_counts dictionary is still calculated but emotion_dist is removed
```

##### Step 2: Update Window Features Dictionary
```python
# BEFORE
window_features = {
    ...
    'expression_count': expression_count,
    **emotion_dist,  # Remove this line
    'average_face_size': round(average_face_size, 4),
    ...
}

# AFTER
window_features = {
    ...
    'expression_count': expression_count,
    # emotion ratios removed - perfect multicollinearity
    'average_face_size': round(average_face_size, 4),
    ...
}
```

##### Step 3: Update Emotion Tests
- Remove assertions for all 7 emotion ratios
- Ensure expression_count remains
- Add tests for new emotion features (see below)

---

## New Features to Add

### Emotion Features - Non-Collinear Replacements

To replace the removed emotion ratios while preserving emotional information without multicollinearity, we'll add 3 new features:

#### 1. dominant_emotion_id
**Purpose**: Identifies the most frequent emotion as a numeric value
**Range**: 1-7 (categorical encoded as numeric)
**Encoding**:
- joy = 1
- sadness = 2
- anger = 3
- fear = 4
- disgust = 5
- surprise = 6
- neutral = 7

**Tie Handling**: When multiple emotions have the same count, the first in the order above wins. This ensures deterministic, reproducible results. The `emotion_consistency` feature will indicate low dominance (e.g., 0.5) when ties occur.

#### 2. emotional_valence
**Purpose**: Measures positive vs negative emotional tone
**Range**: -1.0 to +1.0
**Formula**: `(joy_count - (sadness + anger + fear + disgust)) / total_count`
- Positive values = positive emotional tone
- Negative values = negative emotional tone
- Zero = balanced emotions

**Note on Surprise**: Surprise is intentionally excluded from this calculation as it can be either positive or negative depending on context. Surprise information is preserved in `dominant_emotion_id` if it's the most frequent emotion.

#### 3. emotion_consistency
**Purpose**: Measures how unified the emotional state is
**Range**: 0.17 to 1.0
**Formula**: `max(emotion_counts) / total_count`
- 1.0 = All detections show same emotion (consistent)
- 0.17 = All 6 emotions equally distributed (chaotic)
- Typical values: 0.5-0.8 (dominant emotion with some variation)

### Implementation Plan for New Features

#### Step 1: Add New Emotion Feature Calculations
```python
# Add this AFTER emotion_counts calculation in temporal_compute.py (after line 1407)
# NOTE: This depends on emotion_counts being calculated (lines 1399-1407)
# Do NOT remove the emotion_counts calculation, only remove emotion_dist

# Calculate new non-collinear emotion features
total_emotions = len(segment_expressions)

# Feature 1: Dominant emotion (categorical as numeric)
emotion_encoding = {
    'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4,
    'disgust': 5, 'surprise': 6, 'neutral': 7
}

if total_emotions > 0:
    # Find dominant emotion (with deterministic tie handling)
    max_count = max(emotion_counts.values())
    dominant_emotion = None
    # First emotion in this order wins ties
    for emotion in ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']:
        if emotion_counts.get(emotion, 0) == max_count:
            dominant_emotion = emotion
            break
    dominant_emotion_id = emotion_encoding[dominant_emotion]

    # Feature 2: Emotional valence (-1 to +1)
    # Note: Surprise excluded as it's ambiguous (can be positive or negative)
    positive_count = emotion_counts.get('joy', 0)
    negative_count = (emotion_counts.get('sadness', 0) +
                     emotion_counts.get('anger', 0) +
                     emotion_counts.get('fear', 0) +
                     emotion_counts.get('disgust', 0))
    # Neutral and surprise don't affect valence
    emotional_valence = (positive_count - negative_count) / total_emotions

    # Feature 3: Emotion consistency (how unified)
    max_emotion_count = max(emotion_counts.values())
    emotion_consistency = max_emotion_count / total_emotions
else:
    # No emotions detected - use defaults
    dominant_emotion_id = 7  # neutral
    emotional_valence = 0.0
    emotion_consistency = 0.0
```

#### Step 2: Update Window Features Dictionary
```python
# Updated window features with new emotion features
window_features = {
    ...
    'expression_count': expression_count,
    # New emotion features (replace removed ratios)
    'dominant_emotion_id': dominant_emotion_id,
    'emotional_valence': round(emotional_valence, 4),
    'emotion_consistency': round(emotion_consistency, 4),
    # Continue with other features
    'average_face_size': round(average_face_size, 4),
    ...
}
```

#### Step 3: Expected Output in Unified JSON

##### Before (with problematic ratios):
```json
{
  "temporal_windows": {
    "hook": {
      "expression_count": 6,
      "joy_ratio": 0.5,
      "sadness_ratio": 0.167,
      "anger_ratio": 0.0,
      "fear_ratio": 0.0,
      "disgust_ratio": 0.0,
      "surprise_ratio": 0.167,
      "neutral_ratio": 0.167
    }
  }
}
```

##### After (with new features):
```json
{
  "temporal_windows": {
    "hook": {
      "expression_count": 6,
      "dominant_emotion_id": 1,
      "emotional_valence": 0.5,
      "emotion_consistency": 0.5
    }
  }
}
```
**Feature meanings**:
- `dominant_emotion_id`: 1 = joy is most common
- `emotional_valence`: 0.5 = positive leaning
- `emotion_consistency`: 0.5 = 50% same emotion

#### Step 4: Benefits of New Features

1. **No Multicollinearity**: These features are independent, don't sum to any constant
2. **Information Preserved**:
   - Which emotion dominates (dominant_emotion_id)
   - Overall emotional tone (emotional_valence)
   - Emotional variety/chaos (emotion_consistency)
   - Total emotional activity (expression_count)
3. **ML-Friendly**: All numeric, compatible with Random Forest and K-means
4. **Scalable**: Works whether we have 2 or 20 emotion samples per window

#### Step 5: Testing the New Features

Add these test cases:
```python
def test_new_emotion_features():
    # Test case 1: Joy dominant
    emotions = ['joy', 'joy', 'joy', 'neutral']
    expected = {
        'dominant_emotion_id': 1,  # joy
        'emotional_valence': 0.75,  # 3 joy / 4 total
        'emotion_consistency': 0.75  # 3 out of 4 same
    }

    # Test case 2: Mixed emotions (all tied)
    emotions = ['joy', 'sadness', 'anger', 'neutral']
    expected = {
        'dominant_emotion_id': 1,  # all tied at 1, joy wins (first in order)
        'emotional_valence': -0.25,  # 1 positive - 2 negative / 4
        'emotion_consistency': 0.25  # each appears once
    }

    # Test case 2b: Tie between two emotions
    emotions = ['joy', 'joy', 'anger', 'anger']
    expected = {
        'dominant_emotion_id': 1,  # joy=2, anger=2, joy wins (first in order)
        'emotional_valence': 0.0,  # 2 positive - 2 negative / 4
        'emotion_consistency': 0.5  # half are same emotion
    }

    # Test case 4: Surprise handling
    emotions = ['joy', 'surprise', 'surprise', 'neutral']
    expected = {
        'dominant_emotion_id': 6,  # surprise most common
        'emotional_valence': 0.25,  # only joy counts as positive (1/4)
        'emotion_consistency': 0.5  # surprise appears 2/4 times
    }

    # Test case 3: No emotions
    emotions = []
    expected = {
        'dominant_emotion_id': 7,  # default neutral
        'emotional_valence': 0.0,
        'emotion_consistency': 0.0
    }
```

### Summary of Changes

#### Features Removed (19 total):
- 4 framing ratios (close, medium, wide, none)
- 2 density features (element_count, avg_density)
- 1 scene pacing (changes_per_second)
- 4 semantic speech (has_greeting, has_question, has_instruction, has_speech_cta)
- 1 audio pattern (burst_pattern)
- 7 emotion ratios (joy, sadness, anger, fear, disgust, surprise, neutral)

#### Features Added (3 new):
- dominant_emotion_id (categorical emotion as numeric)
- emotional_valence (positive vs negative)
- emotion_consistency (how unified)

#### Net Result:
- **Before**: 19 problematic features + others
- **After**: 3 new meaningful features + others
- **Reduction**: 16 features total
- **Benefit**: No multicollinearity, better ML performance

---

## Known Risks

### Implementation Risks
- New emotion features are untested on production data
- Performance impact of feature changes unknown
- Emotional patterns may differ significantly across video categories

### Data Risks
- With only 2-6 frames per window, emotion features may be noisy
- Tie-breaking rule (first wins) introduces slight bias toward joy
- Removing semantic features eliminates all text-based signals

### Mitigation Notes
- Monitor model performance metrics after deployment
- Keep raw data to recalculate features if needed
- Consider gradual rollout to subset of videos first
- Maintain ability to add back removed features to training data if performance degrades

---

## Output Validation

After implementing these changes, run a test video and verify the JSON output:

### Features That Should Be Absent
```json
{
  "temporal_windows": {
    "hook": {
      // These should NOT appear:
      "close_ratio": ❌,
      "medium_ratio": ❌,
      "wide_ratio": ❌,
      "none_ratio": ❌,
      "element_count": ❌,
      "avg_density": ❌,
      "changes_per_second": ❌,
      "has_greeting": ❌,
      "has_question": ❌,
      "has_instruction": ❌,
      "has_speech_cta": ❌,
      "burst_pattern": ❌,
      "joy_ratio": ❌,
      "sadness_ratio": ❌,
      "anger_ratio": ❌,
      "fear_ratio": ❌,
      "disgust_ratio": ❌,
      "surprise_ratio": ❌,
      "neutral_ratio": ❌
    }
  }
}
```

### Features That Should Be Present with Valid Ranges
```json
{
  "temporal_windows": {
    "hook": {
      // Existing features (keep):
      "average_face_size": 0.0-1.0,
      "expression_count": ≥0,
      "object_count": ≥0,
      "gesture_count": ≥0,
      "scene_count": ≥0,
      "speech_coverage": 0.0-1.0,
      "word_count": ≥0,
      "energy_level": 0.0-1.0,
      "energy_variance": ≥0,
      "energy_max": 0.0-1.0,

      // New emotion features:
      "dominant_emotion_id": 1-7,
      "emotional_valence": -1.0 to 1.0,
      "emotion_consistency": 0.0-1.0
    }
  }
}
```

### Quick Validation Script
```python
def validate_temporal_window(window_data):
    """Validate a single temporal window has correct features"""

    # Features that should NOT exist
    removed_features = [
        'close_ratio', 'medium_ratio', 'wide_ratio', 'none_ratio',
        'element_count', 'avg_density', 'changes_per_second',
        'has_greeting', 'has_question', 'has_instruction', 'has_speech_cta',
        'burst_pattern', 'joy_ratio', 'sadness_ratio', 'anger_ratio',
        'fear_ratio', 'disgust_ratio', 'surprise_ratio', 'neutral_ratio'
    ]

    # Check removed features are absent
    for feature in removed_features:
        assert feature not in window_data, f"Found removed feature: {feature}"

    # Validate new features exist and have correct ranges
    assert 1 <= window_data['dominant_emotion_id'] <= 7
    assert -1.0 <= window_data['emotional_valence'] <= 1.0
    assert 0.0 <= window_data['emotion_consistency'] <= 1.0

    print("✅ Validation passed!")
```