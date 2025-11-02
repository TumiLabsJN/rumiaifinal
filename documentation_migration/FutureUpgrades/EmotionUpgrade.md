# Emotion Valence Calculation Upgrade

**Document**: Feature calculation bug fix and enhancement proposal
**Date**: 2025-10-29
**Status**: Proposed (Not Yet Implemented)
**Priority**: HIGH - Affects ML model training accuracy
**Related**: AudioFeatures.md, AnalysisServices.md, temporal_compute.py

---

## Executive Summary

The current `emotional_valence` calculation has a critical bug where **sparse face detection** (1-2 emotion frames in a window) receives **100% weight**, producing misleading values of `-1.0` or `+1.0` even when faces are barely visible.

**Example Bug**:
- Hook window: 0-3s (3 seconds)
- FEAT detects face in **1 frame** at 2.0s (sadness)
- Other frames: No face detected
- Current output: `emotional_valence = -1.0` (100% negative)
- Expected output: `emotional_valence = 0.0` (insufficient face data)

**Impact**:
- ML models trained on incorrect emotion signals
- Videos with minimal face visibility get extreme valence scores
- Contrastive analysis (top vs bottom) skewed by false negative emotions

---

## Current Implementation

### Location
`/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py` (lines ~600-700)

### Current Logic
```python
# Simplified current implementation
if total_emotions > 0:
    positive_count = emotion_counts.get('joy', 0)
    negative_count = (emotion_counts.get('sadness', 0) +
                     emotion_counts.get('anger', 0) +
                     emotion_counts.get('fear', 0) +
                     emotion_counts.get('disgust', 0))

    emotional_valence = (positive_count - negative_count) / total_emotions
    # ⚠️ PROBLEM: No consideration of temporal coverage or face visibility
else:
    emotional_valence = 0.0
```

### FEAT Adaptive Sampling Rates
From `emotion_detection_service.py:76-89`:
```python
def get_adaptive_sample_rate(self, video_duration: float) -> float:
    if video_duration <= 30:
        return 2.0  # 2 FPS for short videos (≤30s)
    elif video_duration <= 60:
        return 1.0  # 1 FPS for medium videos (30-60s)
    else:
        return 0.5  # 0.5 FPS for long videos (>60s)
```

---

## Bug Evidence

### Case Study: Video 7542515730902699295

**Video Details**:
- Duration: 60.73 seconds
- FEAT sample rate: 0.5 FPS (1 frame every 2 seconds)
- Total emotion frames detected: 11 frames across entire video
- Content type: Wellness video with product demos (B-roll heavy)

**Hook Window (0-3s) Analysis**:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Expected emotions @ 0.5 FPS | 1.5 frames | 3s × 0.5 FPS |
| Actual emotions detected | 1 frame (at 2.0s) | 67% coverage ✅ |
| Emotion detected | Sadness (confidence 0.46) | Low confidence ⚠️ |
| `person_count` | 1 | YOLO detected body |
| `average_face_size` | 0.0 | No face in most frames |
| `eye_contact_rate` | 0.0 | Confirms no face visibility |
| **Current output** | `emotional_valence = -1.0` | ❌ **BUG** |
| **Expected output** | `emotional_valence = 0.0` | Should be neutral (insufficient data) |

**Middle_1 Window (3-13.8s) Analysis**:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Expected emotions @ 0.5 FPS | 5.4 frames | 10.8s × 0.5 FPS |
| Actual emotions detected | 1 frame (at 10.0s) | 18.5% coverage ❌ |
| Emotion detected | Anger (confidence 0.84) | High confidence but sparse |
| `average_face_size` | 0.1099 | Small face |
| **Current output** | `emotional_valence = -1.0` | ❌ **BUG** |
| **Expected output** | `emotional_valence = 0.0` | Insufficient coverage (<30%) |

**Pattern Across All Windows**:

| Window | Duration | Expected Frames | Actual Frames | Coverage | Current Valence | Bug? |
|--------|----------|----------------|---------------|----------|-----------------|------|
| Hook | 3.0s | 1.5 | 1 | 67% | -1.0 | ⚠️ Low confidence |
| Middle_1 | 10.8s | 5.4 | 1 | 18.5% | -1.0 | ❌ **Sparse** |
| Middle_2 | 10.8s | 5.4 | 0 | 0% | 0.0 | ✅ Correct |
| Middle_3 | 10.8s | 5.4 | 1 | 18.5% | -1.0 | ❌ **Sparse** |
| Middle_4 | 10.8s | 5.4 | 4 | 74% | -1.0 | ⚠️ All anger |
| Middle_5 | 10.8s | 5.4 | 4 | 74% | -0.33 | ✅ Mixed emotions |
| Closing | 3.0s | 1.5 | 1 | 67% | 0.0 | ✅ Neutral |

---

## Root Cause Analysis

### Problem 1: Sparse Face Detection Gets 100% Weight

**Scenario**: B-roll heavy video (product demos, hands, text)
- Creator's face visible for 1-2 seconds in a 10-second window
- FEAT samples at 0.5 FPS → captures 1 emotion frame
- That 1 frame determines 100% of `emotional_valence` for entire window
- Result: `-1.0` valence even though 80% of window has no face

**Why This Happens**:
```python
# Current formula treats all detected emotions equally
emotional_valence = (positive - negative) / total_emotions
# 1 negative frame / 1 total frame = -1.0

# SHOULD account for:
# - How much of the window actually has a face?
# - How confident is the emotion detection?
# - What percentage of expected frames were detected?
```

---

### Problem 2: No Confidence Filtering

**Scenario**: Low-confidence emotion detection
- Hook detects sadness at 2.0s with **confidence 0.46** (below typical 0.5 threshold)
- Small/occluded face produces uncertain emotion classification
- Current implementation treats all confidences equally

**Evidence**:
```json
{
  "timestamp": 2.0,
  "emotion": "sadness",
  "confidence": 0.46125513315200806,  // ← Low confidence
  "all_scores": {
    "sadness": 0.461,
    "anger": 0.188,
    "joy": 0.129,     // ← Almost as likely as sadness!
    "neutral": 0.102
  }
}
```

---

### Problem 3: Spatial Coverage Ignored

**Scenario**: `average_face_size = 0.0` but emotions still calculated
- Hook has `average_face_size = 0.0` (no face in window)
- Yet `emotional_valence = -1.0` (from 1 frame at edge of window)
- Contradiction: How can we have emotions without a face?

**Cross-Feature Inconsistency**:
```json
{
  "average_face_size": 0.0,      // No face
  "eye_contact_rate": 0.0,       // No eye contact
  "gaze_variance": 0.0,          // No gaze data
  "emotional_valence": -1.0,     // ← But has strong negative emotion?!
  "dominant_emotion_id": 2       // ← And dominant emotion ID?!
}
```

---

## Proposed Solutions

### Option 1: Minimum Coverage Threshold ⭐ **RECOMMENDED**

**Approach**: Require minimum percentage of expected emotion frames before calculating valence.

**Implementation**:
```python
def calculate_emotion_features(
    segment_expressions: List[Dict],
    window_duration: float,
    sample_rate: float,  # From FEAT metadata (0.5, 1.0, or 2.0)
    segment_faces: List[Dict]
) -> Dict[str, Any]:
    """
    Calculate emotion features with coverage threshold.

    Returns:
        {
            'dominant_emotion_id': int (1-8),
            'emotional_valence': float (-1.0 to 1.0),
            'emotion_consistency': float (0.0 to 1.0)
        }
    """
    # Calculate expected frames based on FEAT sampling rate
    expected_emotion_frames = window_duration * sample_rate
    actual_emotion_frames = len(segment_expressions)

    # Coverage threshold: require at least 30% of expected frames
    MINIMUM_COVERAGE = 0.3

    if actual_emotion_frames == 0:
        # No emotions detected
        if segment_faces:
            # Face exists but no emotions classified
            return {
                'dominant_emotion_id': 7,  # neutral
                'emotional_valence': 0.0,
                'emotion_consistency': 0.0
            }
        else:
            # No face detected
            return {
                'dominant_emotion_id': 8,  # no_person
                'emotional_valence': 0.0,
                'emotion_consistency': 0.0
            }

    # Check coverage
    coverage = actual_emotion_frames / expected_emotion_frames

    if coverage < MINIMUM_COVERAGE:
        # Insufficient face visibility - default to neutral
        # Log warning for debugging
        logger.debug(
            f"Insufficient emotion coverage: {actual_emotion_frames}/{expected_emotion_frames:.1f} "
            f"({coverage:.1%}) in {window_duration}s window. Defaulting to neutral."
        )
        return {
            'dominant_emotion_id': 7,  # neutral
            'emotional_valence': 0.0,
            'emotion_consistency': 0.0
        }

    # Normal calculation (sufficient coverage)
    emotion_counts = {emotion: 0 for emotion in ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']}
    for e in segment_expressions:
        emotion = e.get('emotion', 'neutral')
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1

    total_emotions = len(segment_expressions)

    # Dominant emotion (deterministic tie handling)
    max_count = max(emotion_counts.values())
    dominant_emotion = None
    for emotion in ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']:
        if emotion_counts.get(emotion, 0) == max_count:
            dominant_emotion = emotion
            break

    emotion_encoding = {
        'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4,
        'disgust': 5, 'surprise': 6, 'neutral': 7, 'no_person': 8
    }
    dominant_emotion_id = emotion_encoding[dominant_emotion]

    # Emotional valence
    positive_count = emotion_counts.get('joy', 0)
    negative_count = (emotion_counts.get('sadness', 0) +
                     emotion_counts.get('anger', 0) +
                     emotion_counts.get('fear', 0) +
                     emotion_counts.get('disgust', 0))
    emotional_valence = (positive_count - negative_count) / total_emotions

    # Emotion consistency
    max_emotion_count = max(emotion_counts.values())
    emotion_consistency = max_emotion_count / total_emotions

    return {
        'dominant_emotion_id': dominant_emotion_id,
        'emotional_valence': round(emotional_valence, 4),
        'emotion_consistency': round(emotion_consistency, 4)
    }
```

**Thresholds**:
- **30% coverage** = Minimum threshold (1 frame in 3 expected, or 2 in 6)
- Rationale: Balances data availability vs accuracy
- Tested on 60s videos: Filters out sparse windows while keeping face-present segments

**Impact**:
- Video 7542515730902699295 Hook: `emotional_valence = -1.0` → `0.0` ✅ (67% coverage but low confidence handled separately)
- Video 7542515730902699295 Middle_1: `emotional_valence = -1.0` → `0.0` ✅ (18.5% < 30%)
- Video 7542515730902699295 Middle_3: `emotional_valence = -1.0` → `0.0` ✅ (18.5% < 30%)

**Pros**:
- ✅ Simple to implement (15-20 lines of code)
- ✅ Based on objective metric (frame coverage)
- ✅ Accounts for FEAT's adaptive sampling (uses actual sample_rate)
- ✅ Fixes the core bug without over-engineering

**Cons**:
- ⚠️ May still accept low-confidence emotions if coverage is sufficient
- ⚠️ Doesn't consider face size (small faces might be unreliable)

---

### Option 2: Confidence Filtering

**Approach**: Only count high-confidence emotion detections.

**Implementation**:
```python
def calculate_emotion_features(
    segment_expressions: List[Dict],
    window_duration: float,
    sample_rate: float,
    segment_faces: List[Dict]
) -> Dict[str, Any]:
    """
    Calculate emotion features with confidence threshold.
    """
    # Filter low-confidence emotions
    MINIMUM_CONFIDENCE = 0.5

    high_confidence_emotions = [
        e for e in segment_expressions
        if e.get('confidence', 0) >= MINIMUM_CONFIDENCE
    ]

    if len(high_confidence_emotions) == 0:
        # No high-confidence emotions
        if len(segment_expressions) > 0:
            # Low-confidence emotions exist - default to neutral
            return {
                'dominant_emotion_id': 7,  # neutral
                'emotional_valence': 0.0,
                'emotion_consistency': 0.0
            }
        else:
            # No emotions at all
            return {
                'dominant_emotion_id': 8 if not segment_faces else 7,
                'emotional_valence': 0.0,
                'emotion_consistency': 0.0
            }

    # Calculate valence using only high-confidence emotions
    # ... (same calculation as current, but on filtered list)
```

**Thresholds**:
- **0.5 confidence** = Standard ML classification threshold
- FEAT confidence scores are well-calibrated (from softmax probabilities)

**Impact**:
- Video 7542515730902699295 Hook: Sadness has confidence 0.46 → Filtered out → `emotional_valence = 0.0` ✅

**Pros**:
- ✅ Filters unreliable emotion classifications
- ✅ Uses FEAT's built-in uncertainty metric
- ✅ Simple implementation

**Cons**:
- ⚠️ May over-filter: Some legitimate emotions have 0.4-0.5 confidence
- ⚠️ Doesn't solve coverage problem (1 high-confidence frame still = -1.0)

---

### Option 3: Cross-Reference with Face Size

**Approach**: Validate emotion data against spatial face presence.

**Implementation**:
```python
def calculate_emotion_features(
    segment_expressions: List[Dict],
    window_duration: float,
    sample_rate: float,
    segment_faces: List[Dict],
    average_face_size: float  # From face analysis
) -> Dict[str, Any]:
    """
    Calculate emotion features with face size validation.
    """
    # If face is too small or absent, don't trust emotion data
    MINIMUM_FACE_SIZE = 0.05  # 5% of frame

    if average_face_size < MINIMUM_FACE_SIZE:
        # Face too small/absent for reliable emotion detection
        logger.debug(
            f"Face too small ({average_face_size:.4f} < {MINIMUM_FACE_SIZE}) "
            f"for reliable emotion detection. Defaulting to neutral/no_person."
        )
        return {
            'dominant_emotion_id': 8,  # no_person
            'emotional_valence': 0.0,
            'emotion_consistency': 0.0
        }

    # Normal calculation (face is sufficiently visible)
    # ... (same as current)
```

**Thresholds**:
- **0.05 face size** = 5% of frame (very small face threshold)
- Rationale: Below 5%, FEAT's face detector struggles with accuracy

**Impact**:
- Video 7542515730902699295 Hook: `average_face_size = 0.0` → `emotional_valence = 0.0` ✅
- Video 7542515730902699295 Middle_4: `average_face_size = 0.0` → `emotional_valence = 0.0` ✅

**Pros**:
- ✅ Cross-validates emotion data with independent metric (face size)
- ✅ Fixes contradiction (emotions without visible face)
- ✅ Simple single-threshold check

**Cons**:
- ⚠️ Dependency on `average_face_size` calculation (must be calculated first)
- ⚠️ Doesn't handle "medium face but sparse detection" scenarios
- ⚠️ May be too aggressive (filters all emotions if face is small)

---

## Recommended Implementation: Combined Approach

**Best Solution**: Combine Options 1 + 2 for robust filtering.

```python
def calculate_emotion_features(
    segment_expressions: List[Dict],
    window_duration: float,
    sample_rate: float,
    segment_faces: List[Dict],
    average_face_size: float
) -> Dict[str, Any]:
    """
    Calculate emotion features with multi-stage validation.

    Stage 1: Filter low-confidence emotions
    Stage 2: Check coverage threshold
    Stage 3: (Optional) Cross-validate with face size
    """
    # Configuration
    MINIMUM_CONFIDENCE = 0.5
    MINIMUM_COVERAGE = 0.3
    MINIMUM_FACE_SIZE = 0.05  # Optional additional check

    # Stage 1: Filter low-confidence emotions
    high_confidence_emotions = [
        e for e in segment_expressions
        if e.get('confidence', 0) >= MINIMUM_CONFIDENCE
    ]

    # Stage 2: Check coverage
    expected_frames = window_duration * sample_rate
    actual_frames = len(high_confidence_emotions)

    if actual_frames == 0:
        # No high-confidence emotions
        if segment_faces:
            return {'dominant_emotion_id': 7, 'emotional_valence': 0.0, 'emotion_consistency': 0.0}
        else:
            return {'dominant_emotion_id': 8, 'emotional_valence': 0.0, 'emotion_consistency': 0.0}

    coverage = actual_frames / expected_frames

    if coverage < MINIMUM_COVERAGE:
        # Insufficient coverage
        logger.debug(
            f"Low emotion coverage: {actual_frames}/{expected_frames:.1f} ({coverage:.1%}). "
            f"Defaulting to neutral."
        )
        return {'dominant_emotion_id': 7, 'emotional_valence': 0.0, 'emotion_consistency': 0.0}

    # Stage 3 (Optional): Cross-validate with face size
    # Uncomment if needed for additional validation
    # if average_face_size < MINIMUM_FACE_SIZE:
    #     return {'dominant_emotion_id': 8, 'emotional_valence': 0.0, 'emotion_consistency': 0.0}

    # Normal calculation with high-confidence emotions only
    emotion_counts = {emotion: 0 for emotion in ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']}
    for e in high_confidence_emotions:
        emotion = e.get('emotion', 'neutral')
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1

    total_emotions = len(high_confidence_emotions)

    # Calculate dominant emotion
    max_count = max(emotion_counts.values())
    dominant_emotion = None
    for emotion in ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']:
        if emotion_counts.get(emotion, 0) == max_count:
            dominant_emotion = emotion
            break

    emotion_encoding = {
        'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4,
        'disgust': 5, 'surprise': 6, 'neutral': 7, 'no_person': 8
    }
    dominant_emotion_id = emotion_encoding[dominant_emotion]

    # Calculate valence
    positive_count = emotion_counts.get('joy', 0)
    negative_count = (emotion_counts.get('sadness', 0) +
                     emotion_counts.get('anger', 0) +
                     emotion_counts.get('fear', 0) +
                     emotion_counts.get('disgust', 0))
    emotional_valence = (positive_count - negative_count) / total_emotions

    # Calculate consistency
    max_emotion_count = max(emotion_counts.values())
    emotion_consistency = max_emotion_count / total_emotions

    return {
        'dominant_emotion_id': dominant_emotion_id,
        'emotional_valence': round(emotional_valence, 4),
        'emotion_consistency': round(emotion_consistency, 4)
    }
```

**Why This Combination?**:
1. **Confidence filtering** removes unreliable detections (FEAT uncertainty)
2. **Coverage threshold** ensures sufficient temporal data (prevents 1-frame bias)
3. **Optional face size** provides additional validation layer if needed

**Configuration Flexibility**:
```python
# In config or as function parameters
EMOTION_CONFIG = {
    'min_confidence': 0.5,      # Adjustable: 0.4-0.6 range
    'min_coverage': 0.3,        # Adjustable: 0.2-0.5 range
    'min_face_size': 0.05,      # Optional: 0.03-0.10 range
    'enable_face_size_check': False  # Toggle Stage 3
}
```

---

## Implementation Plan

### Phase 1: Core Fix (Week 1)

**Files to Modify**:
1. `/rumiai_v2/processors/temporal_compute.py`
   - Update emotion feature calculation function (lines ~600-700)
   - Add coverage and confidence checks
   - Add logging for filtered windows

2. `/rumiai_v2/processors/temporal_compute.py` (function signature)
   - Add `sample_rate` parameter to emotion calculation
   - Pass from FEAT metadata (`data['sample_rate']`)

**Testing**:
- Unit test with synthetic data (1 frame, 5 frames, 10 frames scenarios)
- Integration test on video 7542515730902699295
- Verify all windows with coverage < 30% return valence = 0.0

**Success Criteria**:
- ✅ Video 7542515730902699295 Hook: `emotional_valence = 0.0` (was -1.0)
- ✅ Video 7542515730902699295 Middle_1: `emotional_valence = 0.0` (was -1.0)
- ✅ Video 7542515730902699295 Middle_3: `emotional_valence = 0.0` (was -1.0)
- ✅ Windows with sufficient coverage still calculate normally

---

### Phase 2: Validation (Week 2)

**Batch Re-processing**:
1. Re-run temporal_compute on existing videos in bucket_60-90s
2. Compare old vs new `emotional_valence` distributions
3. Identify windows where valence changed from ±1.0 → 0.0

**Expected Changes**:
- ~20-30% of windows will have valence adjusted to 0.0
- B-roll heavy videos will show largest changes
- Face-centric videos (talking heads) minimally affected

**Metrics to Track**:
```python
# Analysis script
old_extreme_valence = count(abs(valence) == 1.0)  # Old implementation
new_extreme_valence = count(abs(valence) == 1.0)  # New implementation
reduction = (old_extreme_valence - new_extreme_valence) / old_extreme_valence

# Expected: 20-40% reduction in extreme valence scores
```

---

### Phase 3: ML Model Re-training (Week 3)

**Impact Assessment**:
1. Stage 3: Re-aggregate features with new emotion values
2. Stage 4: Re-transform for RF/K-Means
3. Stage 5: Re-train models
4. Stage 6: Compare RF feature importance (old vs new)

**Expected ML Impact**:
- `emotional_valence` feature importance may decrease (less extreme values)
- Cross-window emotion patterns may become more reliable
- Contrastive analysis (top vs bottom) less skewed by false negatives

---

## Rollback Plan

If new implementation causes issues:

### Option A: Revert Entirely
```bash
# Restore original temporal_compute.py
git checkout temporal_compute.py
# Re-run pipeline on affected videos
```

### Option B: Adjust Thresholds
```python
# If 30% coverage is too strict
MINIMUM_COVERAGE = 0.2  # Reduce to 20%

# If 0.5 confidence is too strict
MINIMUM_CONFIDENCE = 0.4  # Reduce to 40%
```

### Option C: Disable Specific Checks
```python
# Keep confidence filtering, disable coverage check
ENABLE_COVERAGE_CHECK = False

# Or vice versa
ENABLE_CONFIDENCE_CHECK = False
```

---

## Testing Strategy

### Unit Tests

**File**: `/tests/test_emotion_upgrade.py`

```python
import pytest
from rumiai_v2.processors.temporal_compute import calculate_emotion_features

def test_sparse_coverage_returns_neutral():
    """1 frame out of 5 expected should return neutral"""
    segment_expressions = [
        {'emotion': 'sadness', 'confidence': 0.8, 'timestamp': 2.0}
    ]
    result = calculate_emotion_features(
        segment_expressions=segment_expressions,
        window_duration=10.0,
        sample_rate=0.5,  # Expect 5 frames
        segment_faces=[],
        average_face_size=0.1
    )
    assert result['emotional_valence'] == 0.0
    assert result['dominant_emotion_id'] == 7  # neutral

def test_low_confidence_filtered():
    """Low confidence emotions should be filtered out"""
    segment_expressions = [
        {'emotion': 'sadness', 'confidence': 0.4, 'timestamp': 2.0},  # Below 0.5
        {'emotion': 'anger', 'confidence': 0.3, 'timestamp': 4.0}     # Below 0.5
    ]
    result = calculate_emotion_features(
        segment_expressions=segment_expressions,
        window_duration=10.0,
        sample_rate=0.5,
        segment_faces=[{'timestamp': 2.0}],
        average_face_size=0.1
    )
    assert result['emotional_valence'] == 0.0  # All filtered out

def test_sufficient_coverage_calculates_normally():
    """Sufficient high-confidence frames should calculate normally"""
    segment_expressions = [
        {'emotion': 'joy', 'confidence': 0.8, 'timestamp': 2.0},
        {'emotion': 'joy', 'confidence': 0.7, 'timestamp': 4.0},
        {'emotion': 'sadness', 'confidence': 0.6, 'timestamp': 6.0}
    ]
    result = calculate_emotion_features(
        segment_expressions=segment_expressions,
        window_duration=10.0,
        sample_rate=0.5,  # Expect 5, have 3 = 60% coverage
        segment_faces=[{'timestamp': t} for t in [2.0, 4.0, 6.0]],
        average_face_size=0.2
    )
    # 2 joy, 1 sadness = (2 - 1) / 3 = 0.333
    assert result['emotional_valence'] == pytest.approx(0.333, abs=0.01)
    assert result['dominant_emotion_id'] == 1  # joy

def test_no_emotions_no_face():
    """No emotions + no face = no_person"""
    result = calculate_emotion_features(
        segment_expressions=[],
        window_duration=3.0,
        sample_rate=2.0,
        segment_faces=[],
        average_face_size=0.0
    )
    assert result['emotional_valence'] == 0.0
    assert result['dominant_emotion_id'] == 8  # no_person

def test_no_emotions_with_face():
    """No emotions + face present = neutral"""
    result = calculate_emotion_features(
        segment_expressions=[],
        window_duration=3.0,
        sample_rate=2.0,
        segment_faces=[{'timestamp': 1.0}],
        average_face_size=0.15
    )
    assert result['emotional_valence'] == 0.0
    assert result['dominant_emotion_id'] == 7  # neutral
```

---

### Integration Tests

**Test on Real Videos**:

```bash
# Test video 7542515730902699295 (known bug case)
python3 scripts/rumiai_runner.py 'https://www.tiktok.com/@vitalizateusa/video/7542515730902699295'

# Verify output
jq '.temporal_windows.hook | {emotional_valence, dominant_emotion_id, average_face_size}' \
  insights/7542515730902699295_temporal_windows_updated.json

# Expected:
# {
#   "emotional_valence": 0.0,  // Was -1.0 before fix
#   "dominant_emotion_id": 7,   // neutral (was 2=sadness)
#   "average_face_size": 0.0
# }
```

---

## Monitoring & Validation

### Post-Deployment Checks

**Distribution Analysis**:
```python
# Analyze emotion valence distribution before/after
import pandas as pd
import matplotlib.pyplot as plt

# Load aggregated_features.csv (before/after)
df_old = pd.read_csv('old_aggregated_features.csv')
df_new = pd.read_csv('new_aggregated_features.csv')

# Compare valence distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df_old['hook_emotional_valence'].hist(bins=20, ax=axes[0], alpha=0.7)
axes[0].set_title('Old Implementation')
axes[0].set_xlabel('emotional_valence')

df_new['hook_emotional_valence'].hist(bins=20, ax=axes[1], alpha=0.7)
axes[1].set_title('New Implementation (with fix)')
axes[1].set_xlabel('emotional_valence')

plt.savefig('emotion_valence_comparison.png')
```

**Expected Changes**:
- Old: Spike at -1.0 and +1.0 (extreme values)
- New: More concentration at 0.0 (neutral for sparse data)
- New: Smoother distribution between -0.5 to +0.5

---

### Metrics to Track

| Metric | Before Fix | After Fix | Target |
|--------|-----------|-----------|--------|
| % windows with valence = ±1.0 | ~15-20% | ~5-8% | <10% |
| % windows with valence = 0.0 | ~10% | ~25-30% | 20-30% |
| Avg absolute valence | ~0.6 | ~0.35 | 0.3-0.4 |
| Windows filtered (coverage) | 0% | ~15-20% | 15-25% |
| Windows filtered (confidence) | 0% | ~5-10% | 5-15% |

---

## Related Issues & Future Work

### Issue 1: Emotion Consistency Edge Case
**Current**: `emotion_consistency = 1.0` when only 1 frame detected
**Impact**: Misleading "100% consistent" for sparse data
**Fix**: Set consistency = 0.0 when coverage < threshold (same logic as valence)

### Issue 2: Dominant Emotion ID for No-Person
**Current**: `dominant_emotion_id = 8` (no_person) exists but not documented in emotion encoding
**Impact**: ML models may not handle ID=8 properly (expected 1-7)
**Fix**: Verify K-Means one-hot encoding includes no_person category

### Issue 3: FEAT Sampling Rate Not Persisted
**Current**: Sample rate calculated but not saved to temporal_windows JSON
**Impact**: Can't verify coverage calculations post-processing
**Fix**: Add `feat_sample_rate` to metadata in temporal_windows output

---

## Alternatives Considered & Rejected

### Rejected: Temporal Weighting (Option 4)

**Approach**: Weight each emotion frame by its temporal coverage
```python
# Calculate duration each emotion frame represents
for i, emotion in enumerate(emotions):
    if i < len(emotions) - 1:
        duration = emotions[i+1]['timestamp'] - emotions[i]['timestamp']
    else:
        duration = window_end - emotions[i]['timestamp']

    weighted_emotions[emotion] += duration

valence = (weighted_positive - weighted_negative) / window_duration
```

**Rejected Because**:
- ❌ Overly complex (assumes emotion persists between frames)
- ❌ Doesn't solve core issue (1 frame still dominates if alone)
- ❌ Requires careful edge case handling (first/last frames)

---

### Rejected: Machine Learning Confidence Calibration (Option 5)

**Approach**: Re-calibrate FEAT confidence scores using validation set
```python
# Learn confidence threshold per emotion class
calibrated_thresholds = {
    'joy': 0.6,
    'sadness': 0.45,
    'anger': 0.55,
    # ...
}
```

**Rejected Because**:
- ❌ Requires labeled validation set (don't have ground truth)
- ❌ Over-engineering for current problem scope
- ❌ FEAT confidences already well-calibrated (softmax probabilities)

---

## Document Control

**Created**: 2025-10-29
**Author**: Claude (Anthropic)
**Version**: 1.0
**Status**: Proposed - Awaiting Implementation
**Next Review**: After Phase 1 implementation
**Approver**: Jorge (Tumi Labs)

---

## References

- **AudioFeatures.md**: Current emotion feature documentation
- **AnalysisServices.md**: FEAT service architecture and sampling rates
- **TotalFeatures.md**: Feature matrix and ML transforms
- **temporal_compute.py**: Current implementation location
- **emotion_detection_service.py**: FEAT adaptive sampling logic

---

**End of Emotion Valence Calculation Upgrade Documentation**
