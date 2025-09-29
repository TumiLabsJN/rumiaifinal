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

---