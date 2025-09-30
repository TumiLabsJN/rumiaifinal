# MLimitations.md - Known Limitations of RumiAI's ML Services

This document catalogues the known limitations, edge cases, and potential improvements for each ML service in the RumiAI pipeline. Understanding these limitations is crucial for interpreting results and planning future enhancements.

## 1. Emotion Detection (FEAT)

### Current Implementation
- **Service**: FEAT (Facial Expression Analysis Toolkit)
- **Model**: ResMaskNet + AU detection
- **Claimed Accuracy**: 87% on AffectNet dataset

### Known Limitations

#### 1.1 Expression Detection Issues
- **Partial Expressions**: Requires complete stereotypical expressions
  - Example: Angry eyebrows alone → classified as "neutral"
  - Needs: eyebrows + eyes + mouth for reliable detection
- **Natural vs Posed**: Trained primarily on posed expressions
  - Poor performance on subtle, natural expressions
  - Expects "textbook" emotional displays

#### 1.2 Technical Constraints
- **Processing Speed**: 1-2 FPS on GPU (very slow)
- **Memory Usage**: ~2GB for model loading
- **Single Frame**: No temporal context (analyzes frames independently)
- **Face Angle**: Degrades significantly with non-frontal faces
- **Lighting**: Sensitive to shadows and poor lighting

#### 1.3 Classification Issues
- **Binary Emotion Assumption**: Forces single emotion label
- **Cultural Bias**: Trained on Western expression datasets
- **Age/Gender Bias**: Better accuracy on adult faces
- **Ambiguous Expressions**: Struggles with mixed emotions

### Better Alternatives

#### Commercial APIs (Highest Accuracy)
| Service | Pros | Cons | Use Case |
|---------|------|------|----------|
| **AWS Rekognition** | • 8 emotions + confidence<br>• Handles partial expressions<br>• Multi-face support | • Requires internet<br>• $0.001 per image<br>• Privacy concerns | Production with budget |
| **Azure Face API** | • Real-time capable<br>• Returns valence/arousal<br>• Good documentation | • Cloud dependency<br>• Subscription model<br>• Rate limits | Enterprise integration |
| **Google Cloud Vision** | • Robust to angles<br>• Joy/sorrow/anger/surprise<br>• Likelihood scores | • Fewer emotions<br>• Google ecosystem lock-in | Already using GCP |
| **Hume AI** | • 30+ granular emotions<br>• Multimodal (face+voice)<br>• State-of-the-art | • Expensive<br>• Beta access<br>• Complex integration | Research/high-accuracy needs |

#### Open Source Alternatives
| Model | Pros | Cons | Implementation Effort |
|-------|------|------|----------------------|
| **DeepFace** | • Already in codebase<br>• 7 emotions<br>• Faster than FEAT | • 48% accuracy FER2013<br>• Same stereotype issues<br>• No AUs | 3-4 hours |
| **FER (fer2013)** | • Lightweight<br>• Real-time capable<br>• Easy integration | • Lower accuracy<br>• Basic emotions only | 2 hours |
| **EmoNet** | • Valence + arousal<br>• More nuanced<br>• Handles ambiguity | • Newer/less tested<br>• Different output format | 1 day |
| **HSEmotion** | • Higher accuracy<br>• Pyramid feature extraction<br>• Recent (2022) | • Less documentation<br>• Requires PyTorch | 1-2 days |

### Recommended Improvements

1. **Immediate (Low Effort)**:
   - Document expression requirements in test videos
   - Add confidence thresholds to filter uncertain detections
   - Log when faces are non-frontal

2. **Short-term (Medium Effort)**:
   - Implement emotion smoothing across frames
   - Add fallback to DeepFace when FEAT confidence is low
   - Cache emotion models in RAM

3. **Long-term (High Effort)**:
   - Evaluate commercial APIs for cost/benefit
   - Train custom model on TikTok-specific expressions
   - Implement multimodal emotion (face + voice + text)

---

## 2. Overlay Novelty Detection (Not Implemented)

### Current Implementation
- **Status**: NOT IMPLEMENTED (architectural decision)
- **Issue**: Cannot distinguish new overlays from continuing ones across temporal windows

### Known Limitations

#### 2.1 Lost Information
- **No Freshness Metric**: Can't tell if overlay in current window is new or continuing from previous
- **Example**: "Subscribe" appears in all windows - each counts as present but can't identify it started in hook
- **Impact**: ML models can't learn if new overlays vs persistent ones affect engagement differently

#### 2.2 Why Not Implemented
- **Architectural Complexity**: Would require inter-window communication
- **Current Design**: Windows are processed independently by design
- **Trade-off**: Accepted limitation for cleaner, more maintainable architecture
- **Decision Date**: 2024-09-29 (see OCRFix.md)

### What Would Work
- **Post-Processing**: Calculate freshness after all windows processed
- **Timeline Pre-Processing**: Mark overlays as new/continuing before window segmentation
- **Stateful Processing**: Pass window state through pipeline (complex, breaks isolation)

### Current Workaround
- Use `overlay_persistence` metric to infer if text is long-lasting
- Combine with `overlay_coverage` to understand presence patterns
- Accept that novelty/freshness information is lost
- Focus on presence rather than novelty for ML features

### Impact on Analysis
- Cannot distinguish "attention-grabbing new text" from "persistent CTAs"
- Cannot track overlay introduction patterns (e.g., staggered reveals)
- Cannot identify if closing has "new call-to-action" vs repeated message

## 3. Gesture Detection (MediaPipe)

### Current Implementation
- **Service**: MediaPipe GestureRecognizer
- **Processing**: Frame-by-frame detection at 5 FPS
- **Problem**: Detects gestures as point events, not time ranges
- **Counting Strategy**: Counts ALL hand interactions, including unrecognized gestures

### Critical Design Flaw: Gestures as Point Events

#### The Problem
Gestures are **continuous actions over time**, not instantaneous events. Current implementation:
```python
# WRONG: Treats each frame detection as separate gesture
frame_0.2s: pointing detected → count = 1
frame_0.4s: pointing detected → count = 2  # Same gesture!
frame_0.6s: pointing detected → count = 3  # Still same gesture!
```

A single 3-second pointing gesture gets counted as 15-26 separate gestures.

#### Band-Aid Fix (Currently Implemented)
Deduplication at aggregation time in `temporal_compute.py`:
- Groups detections within 0.8s as same gesture
- Reduces count by ~95%
- Preserves raw data but fixes counting

### Architecturally Correct Solution: Gestures as Time Ranges

#### Design Principle
**Gestures ARE time ranges with lifecycle events:**
- START: Gesture begins
- CONTINUE: Gesture sustained
- END: Gesture completes
- TRANSITION: Gesture morphs into another

#### Correct Implementation
```python
class GestureTimeline:
    """Represents gestures as time ranges, not points"""

    def build_timeline(self, frame_detections):
        """Convert frame detections to gesture ranges"""
        gestures = []
        current = None

        for detection in frame_detections:
            if not current:
                # Start new gesture range
                current = GestureRange(
                    type=detection.type,
                    hand=detection.hand,
                    start_time=detection.timestamp,
                    end_time=detection.timestamp,
                    confidence_samples=[detection.confidence]
                )
            elif (current.matches(detection) and
                  detection.timestamp - current.end_time <= CONTINUATION_THRESHOLD):
                # Extend current gesture
                current.end_time = detection.timestamp
                current.confidence_samples.append(detection.confidence)
            else:
                # Finalize current gesture, start new one
                current.duration = current.end_time - current.start_time
                current.stability = np.std(current.confidence_samples)
                gestures.append(current)
                current = GestureRange(...)

        return gestures

class GestureRange:
    """A gesture as it should be: a time range with properties"""
    def __init__(self, type, hand, start_time, end_time):
        self.type = type              # pointing, thumbs_up, etc.
        self.hand = hand              # left, right
        self.start_time = start_time  # When gesture began
        self.end_time = end_time      # When gesture ended
        self.duration = None          # How long held
        self.stability = None         # How steady (confidence variance)
        self.transition_to = None     # Next gesture if morphed
```

#### Benefits of Correct Architecture

1. **Natural Data Model**
   - One record per gesture, not per frame
   - Matches human understanding of gestures
   - Enables duration-based features

2. **Rich Features**
   ```python
   # Current (broken): Just count
   gesture_count = 26  # Wrong!

   # Correct: Multiple meaningful features
   gesture_count = 2
   avg_gesture_duration = 1.5
   gesture_stability = 0.92
   gesture_transitions = 1
   longest_gesture = 2.3
   gesture_types = ['pointing', 'thumbs_up']
   ```

3. **Transition Tracking**
   ```python
   # Can detect gesture flows
   pointing → open_palm → thumbs_up
   # Useful for engagement analysis
   ```

4. **Storage Efficiency**
   - Current: 100 detections for 20-second video
   - Correct: 3-5 gesture ranges for same video
   - 95% reduction in data points

5. **Semantic Accuracy**
   - Distinguishes sustained vs repeated gestures
   - Captures gesture emphasis through duration
   - Preserves intentionality

#### Implementation Path

**Phase 1: Data Model Change**
- Modify `timeline_builder.py` to create gesture ranges
- Update Timeline class to support range entries
- Keep raw detections for debugging

**Phase 2: Feature Engineering**
- Add gesture_duration, gesture_stability metrics
- Track gesture transitions
- Calculate gesture velocity (how quickly changing)

**Phase 3: Deprecate Band-Aid**
- Remove deduplication logic from temporal_compute
- Update all downstream consumers
- Migrate historical data

#### Why Not Implemented Now

1. **Breaking Change**: Requires pipeline refactor
2. **Time Constraint**: Band-aid fix takes 3 minutes, this takes 2 days
3. **Good Enough**: Band-aid gives 95% accuracy improvement
4. **Risk**: More complex = more potential bugs

### Alternative Architectures Considered

#### Event-Driven Detection
Only detect when hand motion exceeds threshold:
- Pros: Efficient, natural gesture boundaries
- Cons: Might miss subtle gestures

#### State Machine
Track gesture state per hand:
- Pros: Handles complex transitions
- Cons: Complex to implement and debug

#### ML-Based Deduplication
Train model to identify same vs different:
- Pros: Learns from data
- Cons: Requires labeled training data

### Recommendation
Implement **Gestures as Time Ranges** in next major version. Until then, the band-aid deduplication fix provides adequate accuracy improvement with minimal risk.

### Critical Limitation: Gesture Recognition Vocabulary

#### The Problem
MediaPipe's GestureRecognizer has extremely limited gesture vocabulary and poor recognition accuracy.

**Claimed Support vs Reality:**
- **Documented gestures**: Pointing_Up, Thumb_Up, Victory, etc.
- **Actually recognizes**: Only 5-6 gestures reliably
- **Missing common gestures**: Pointing (horizontal), waving, counting, explaining gestures

**Real Example from Video05:**
```python
# User performed pointing gesture for 2 seconds
# MediaPipe detection results:
Hook (0-3s): 26 detections ALL type='none'
Entire video: 0 pointing gestures recognized

# Actual recognized gestures:
none: 72 detections (unrecognized hand positions)
open_palm: 3 detections
thumbs_up: 4 detections
victory: 2 detections
```

#### Current Workaround
**We count ALL hand interactions including 'none' type**:
- Rationale: Any hand gesture shows engagement, even if unrecognized
- Better reflects real human communication
- Prevents undercounting natural gesturing

**What 'gesture_count' actually measures:**
- NOT specific gesture types
- BUT hand interaction frequency
- Includes pointing, waving, explaining, and other natural gestures
- More accurate for engagement analysis than strict gesture recognition

#### Impact
1. **Feature Interpretation**:
   - `gesture_count` = hand activity count, not specific gesture count
   - Higher counts may indicate unrecognized but active gesturing

2. **ML Models**:
   - Learn hand activity patterns, not specific gesture patterns
   - Still useful for engagement prediction

3. **Future Enhancement Options**:
   - Add `recognized_gesture_count` for only identified gestures
   - Add `gesture_recognition_rate` = recognized/total
   - Consider alternative gesture recognition models

#### Why Not Fixed
- Current approach (counting all hand activity) is actually MORE useful for engagement analysis
- Filtering to only recognized gestures would undercount real human communication
- MediaPipe's vocabulary is too limited for TikTok's diverse gesture landscape

---

## 4. Emoji Detection (EasyOCR)

### Current Implementation
- **Service**: EasyOCR for text detection
- **Capability**: Detects alphanumeric text only
- **Status**: CANNOT detect emojis or pictographic symbols

### The Problem

#### Test Case: Video03TextsCaptions.mp4
- **Input**: Fire emoji (🔥) displayed at 12-14s in video
- **Expected**: Detect and count emoji as overlay
- **Actual**: EasyOCR cannot see emoji at all

#### Investigation Results
```
OCR detections in closing window (11-14s):
  11-12s: "New Text", "TEST"
  12-13s: "New Text"
  13-14s: "THANK YOU"

Missing: 🔥 emoji (not detected)
```

### What We Tried

#### Attempt 1: Emoji Normalization
**Implementation**: Modified `normalize_text()` in temporal_compute.py to handle emojis:
```python
emoji_mappings = {
    '🔥': '[fire]',
    '❤️': '[heart]',
    '💯': '[100]',
    # ... 20+ common emojis
}

# For unknown emojis, create unique identifier
if not text and original:
    text = f"[emoji_{hashlib.md5(original.encode()).hexdigest()[:6]}]"
```

**Result**: Code works IF emoji is detected, but EasyOCR never detects emojis in the first place.

#### Root Cause
EasyOCR is fundamentally a **text-only OCR system**:
- Trained on alphanumeric characters and punctuation
- Cannot recognize pictographic symbols, emojis, or icons
- No amount of post-processing can fix undetected content

### Impact on Analysis

#### What's Lost
1. **Emoji Overlays**: Popular TikTok emojis (🔥, ❤️, 💯, etc.) are invisible to our system
2. **Engagement Signals**: Emojis often indicate emotional peaks or CTAs
3. **Trend Detection**: Cannot track emoji usage patterns
4. **Cultural Context**: Emojis carry meaning that affects viewer engagement

#### Metrics Affected
- `overlay_unique_count`: Undercounts when emojis present
- `overlay_coverage`: Lower than reality when emojis displayed
- Missing potential features like `emoji_count`, `emoji_diversity`

### Alternative Solutions (Not Implemented)

#### Option 1: Symbol-Aware OCR
- **PaddleOCR**: Claims some emoji support (limited)
- **TrOCR (Transformers)**: Better with symbols but slower
- **Cons**: Major pipeline change, still incomplete emoji coverage

#### Option 2: Object Detection for Emojis
- Train YOLO on emoji dataset
- Detect emojis as objects, not text
- **Pros**: Could work for common emojis
- **Cons**: Need labeled emoji dataset, increases processing time

#### Option 3: Hybrid Approach
- Use EasyOCR for text
- Add emoji-specific detector
- Merge results in timeline
- **Pros**: Best coverage
- **Cons**: Complex, doubles OCR processing time

### Current Workaround
**None**. Emojis are completely invisible to our analysis pipeline.

### Recommendation
Accept this limitation for now. Emoji detection would require:
1. Replacing or supplementing EasyOCR (major change)
2. Significant processing overhead
3. Limited benefit for current ML objectives

Document for creators that emoji overlays won't be tracked in analytics.

## 5. Gender Detection (DeepFace)

### Current Implementation
- **Service**: DeepFace analyze() with gender action
- **Backend**: MTCNN face detector (improved from opencv)
- **Sampling**: 3-10 frames depending on video duration
- **Confidence Threshold**: 75% minimum to vote

### Known Limitations

#### 5.1 Presentation Bias
- **Makeup Dependency**: Model heavily biased toward performative gender presentation
  - Female with makeup → Correctly classified
  - Female without makeup → Often misclassified as male
- **Real Example**: Test video of female (just woken up) → 85% confidence "male"
- **Root Cause**: Training data bias (women typically with makeup, men without)

#### 5.2 Technical Constraints
- **Frame Selection**: Naive linear interpolation (0%, 25%, 50%, 75%, 100%)
  - May sample intro/outro frames with poor angles
  - No scene awareness or face quality checks
- **Processing Time**: ~1.7 seconds per frame with MTCNN
- **Single Modal**: Visual-only, no voice analysis integration

#### 5.3 Classification Issues
- **Binary Assumption**: Forces male/female classification
- **Low Confidence Handling**: Returns null if all frames <75% confidence
- **Cultural Bias**: Western-centric facial feature interpretation
- **Age Sensitivity**: Less accurate on children/elderly

### Better Alternatives

#### Multi-Modal Approaches
| Method | Pros | Cons | Implementation |
|--------|------|------|----------------|
| **Face + Voice** | • More robust classification<br>• Catches visual errors | • Requires voice presence<br>• Complex fusion logic | 1 week |
| **Creator History** | • Learn from past videos<br>• Account-level consistency | • Requires data persistence<br>• Cold start problem | 3 days |
| **Self-Identification** | • 100% accurate<br>• Respects identity | • Requires UI/input<br>• Not automated | 1 day |

#### Alternative Models
| Model | Accuracy | Speed | Notes |
|-------|----------|-------|--------|
| **FairFace** | Better on diverse faces | Similar | Addresses some bias issues |
| **InsightFace** | High accuracy | Slower | Better with angles |
| **Custom Model** | Can target TikTok data | Varies | Requires training data |

### Recommended Improvements

#### Short-term (1-2 days)
1. **Smart Frame Selection**: Skip first/last 10% of video
2. **Face Quality Filtering**: Check face size, angle, blur before using frame
3. **Confidence Weighting**: Weight high-confidence frames more in voting

#### Medium-term (1 week)
1. **Multi-Modal Fusion**: Combine face + voice pitch for better accuracy
2. **Scene-Aware Sampling**: Sample from stable scenes, not transitions
3. **Fallback Strategy**: Use creator's historical gender if current detection fails

#### Long-term (2-4 weeks)
1. **Custom Model Training**: Train on TikTok-specific dataset with diverse presentations
2. **Non-Binary Support**: Move beyond male/female classification
3. **Presentation-Invariant Features**: Focus on bone structure, not cosmetic features

### Current Workaround
**Removed avg_pitch_normalized feature** - Eliminated dependency on gender detection for pitch analysis. System now uses gender-independent pitch_scatter_ratio instead.

---

## 6. Person Count (ByteTrack + Temporal Windows)

### Current Implementation
- **Service**: YOLO v8 for detection + ByteTrack for tracking
- **Processing**: Object detection at 5 FPS with track IDs
- **Counting Strategy**: Maximum simultaneous persons visible within 0.2s temporal windows

### Known Limitations

#### 6.1 Sequential vs Simultaneous People
- **The Problem**: Cannot distinguish between multiple people appearing sequentially vs same person with track fragmentation
- **Example Case**: Video10TwoPeople.mp4 segment 3 (8.33-11.0s)
  - Person A visible 8.33-9.0s (ID 2)
  - Person B visible 9.0-11.0s (ID 4)
  - Result: person_count = 1 (maximum simultaneous)
  - Reality: 2 different people appeared sequentially
- **Design Decision**: Counting maximum simultaneous was chosen to prevent track fragmentation over-counting

#### 6.2 Track Fragmentation Trade-off
- **What We Fixed**: Same person getting multiple IDs no longer causes over-counting
- **What We Lost**: Total unique people who appear in segment (useful for cast complexity)
- **The Dilemma**:
  - Count unique IDs → Over-counts due to fragmentation
  - Count max simultaneous → Under-counts sequential appearances
  - No perfect solution without person re-identification

#### 6.3 Rapid Re-ID Within Window
- **Edge Case**: Person loses and regains tracking within same 0.2s window
- **Probability**: ~0.1% of cases (requires track loss and re-acquisition within 200ms)
- **Impact**: Would count same person twice within that window
- **Decision**: Accepted as extremely rare edge case

### What person_count Actually Represents

**Current Metric**: Maximum number of people visible simultaneously in any 0.2s window within the segment

**NOT Captured**:
- Total unique people who appeared in segment
- Cast complexity for sequential appearances
- Quick cuts between different people (testimonial patterns)

### Better Alternatives

#### Hybrid Metrics (Recommended)
```python
# Provide multiple counts for ML models to choose
person_count_max = 1      # Current: Max simultaneous
person_count_total = 2    # Future: Total unique (with smart dedup)
person_turnover = 1       # Future: How many person changes
person_consistency = 0.5  # Future: Same people throughout?
```

#### Person Re-Identification
| Method | Pros | Cons | Implementation |
|--------|------|------|----------------|
| **Face Recognition** | • Accurate person matching<br>• Handles re-ID perfectly | • Privacy concerns<br>• Computationally expensive | 1 week |
| **Appearance Similarity** | • No face needed<br>• Uses clothing/body | • Changes in angle affect<br>• Less accurate | 3 days |
| **Spatial-Temporal Heuristics** | • Fast<br>• No ML needed | • Many edge cases<br>• Parameter tuning | 2 days |

### Impact on ML Analysis

#### For Viral Pattern Detection
- **Lost Patterns**: Quick testimonials, reaction compilations, before/after sequences
- **Preserved Patterns**: Group dynamics, solo vs multi-person content
- **Workaround**: Combine with scene_count to infer sequential patterns

#### For Engagement Prediction
- **Still Useful**: Distinguishes solo from group content
- **Less Granular**: Can't detect "3 testimonials" pattern
- **Compensated By**: Other features like scene changes, emotion variety

### Recommended Improvements

#### Short-term (Already Implemented)
1. ✅ Temporal window approach to handle track fragmentation
2. ✅ Filter fallback IDs (>=10000 or tracked=False)
3. ✅ Separate person and object counting logic

#### Medium-term (1 week)
1. Add `person_count_total` as additional feature (with caveat documentation)
2. Track person "turnover" - how many ID changes occur
3. Add confidence score for counting accuracy

#### Long-term (2-4 weeks)
1. Implement basic person re-identification using appearance features
2. Create "cast complexity" metric combining multiple signals
3. Add temporal patterns (e.g., "alternating speakers" detection)

### Current Workaround
**Use maximum simultaneous count** - Provides conservative, consistent metric that won't explode due to fragmentation. Better to under-count sequential appearances than over-count due to tracking errors.

### Why Not Fixed Completely
1. **No Perfect Solution**: Without person re-identification, cannot distinguish sequential people from fragmented tracking
2. **Business Priority**: Current approach prevents gross over-counting which is more harmful to ML models
3. **Acceptable Trade-off**: Losing sequential cast information is better than 3x over-counting from fragmentation
4. **Future Enhancement**: Proper fix requires person re-identification system (significant effort)

