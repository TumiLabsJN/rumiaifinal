| Feature Name | Category | Source Services | Dependencies | Temporal Type | Data Type & Range | ML Importance | Creator Benefit | Reliability | Doubtful | Comments | RF Transform | RF Complexity | KM Transform | KM Complexity | Feature Time |
|--------------|----------|-----------------|--------------|---------------|-------------------|---------------|----------------|-------------|----------|----------|--------------|---------------|--------------|---------------|--------------|
| average_face_size | Person Framing | MediaPipe | None | Temporal | Float [0-1] | Overall face prominence magnitude | Continuous intimacy metric vs discrete ratios | High | None | Mean of face bbox areas in percentage | Direct | Low | Scale [0-1] | Low | Low |
| overlay_unique_count | Text Overlays | OCR | None | Temporal | Integer [0-∞] | Unique marketing text overlay count | More overlays may indicate professional production | High | None | Count of unique text overlays (not captions) | Direct | Low | Log + scale | Low | Medium |
| has_captions | Text Overlays | OCR, Whisper | speech segments for classification | Temporal | Boolean | Presence of speech-synchronized captions | Accessibility and engagement for sound-off viewing | High | None | Binary: detected speech-synchronized text | One-hot (2) | Low | Label encode | Low | Medium |
| scene_count | Scene Pacing | Scene Detection | None | Temporal | Integer [0-∞] | Number of scene changes | More cuts indicate dynamic editing style | High | None | Direct count from scene detection algorithm | Direct | Low | Log + scale | Low | Medium |
| shortest_scene | Scene Pacing | Scene Detection | scene timestamps | Temporal | Float [0-∞] | Duration of shortest scene in seconds | Shows editing pace extremes | High | None | Minimum scene duration calculated from timestamps | Direct | Low | Log + scale | Low | Medium |
| longest_scene | Scene Pacing | Scene Detection | scene timestamps | Temporal | Float [0-∞] | Duration of longest scene in seconds | Shows editing pace extremes | High | None | Maximum scene duration calculated from timestamps | Direct | Low | Log + scale | Low | Medium |
| scene_duration_variance | Scene Pacing | Scene Detection | scene durations | Temporal | Float [0-∞] | Variance in scene durations | High variance indicates dynamic vs consistent pacing | High | Derivative | Variance of scene durations - use durations directly | Direct | Low | Log + scale | Medium | Medium |
| object_count | Object Detection | YOLO + ByteTrack | None | Temporal | Integer [0-∞] | Non-person objects detected | Props, products, and scene elements (excludes persons) | High | None | Direct count from YOLO detections with ByteTrack tracking (excluding persons) | Direct | Low | Log + scale | Low | Medium |
| person_count | Object Detection | YOLO + ByteTrack | object detections with className='person' | Temporal | Integer [0-∞] | Maximum unique persons visible simultaneously | Multi-person content affects viewer engagement | High | Optimized | ByteTrack tracking with 95% dominant track threshold, improved continuity | Direct | Low | Log + scale | Low | Medium |
| speech_coverage | Speech | Whisper | None | Temporal | Float [0-1] | Speech density critical for audience retention | Shows talking vs silent content ratio | High | None | Proportional calculation from segment overlaps | Direct | Low | Scale [0-1] | Low | Medium |
| word_count | Speech | Whisper | speech_coverage | Temporal | Integer [0-∞] | Information density indicator | More words may indicate educational content | High | Colinear | Highly correlated with speech_coverage (r>0.9) | Direct | Low | Log + scale | Low | Medium |
| energy_level | Energy | Audio Energy | None | Temporal | Float [0-1] | Audio intensity affects viewer attention | Higher energy typically increases engagement | High | None | Mean RMS amplitude from audio frames | Direct | Low | Scale [0-1] | Low | Low |
| energy_variance | Energy | Audio Energy | energy_level frames | Temporal | Float [0-∞] | Dynamic range indicates editing style | High variance shows dynamic vs flat audio | High | None | Variance of RMS frames within window | Direct | Low | Log + scale | Low | Low |
| energy_max | Energy | Audio Energy | energy_level frames | Temporal | Float [0-1] | Peak audio intensity moment | Shows loudest moment in segment | High | None | Maximum RMS value in window | Direct | Low | Scale [0-1] | Low | Low |
| pitch_scatter_ratio | Pitch | Audio Energy | voiced frames | Temporal | Float [0-1] | Pitch instability/scatter measure | High values indicate unstable pitch (whisper), low values indicate controlled pitch | Medium | None | Pitch scatter relative to window average | Direct | Low | Scale [0-1] | Medium | High |
| gesture_count | Gestures | MediaPipe | None | Temporal | Integer [0-∞] | Hand movements indicate engagement and expressiveness | More gestures suggest dynamic presentation style | Medium | None | Direct count from MediaPipe gesture detection | Direct | Low | Log + scale | Low | Medium |
| gaze_variance | Gaze | MediaPipe | eye_contact scores | Temporal | Float [0-∞] | Gaze stability affects viewer connection | Consistent eye contact builds trust and engagement | Medium | None | Variance of eye contact scores within window | Direct | Low | Log + scale | Low | Medium |
| eye_contact_rate | Gaze | MediaPipe | None | Temporal | Float [0-1] | Eye contact percentage drives viewer engagement | Higher rates suggest confident, direct communication | High | None | Mean eye contact score from gaze entries | Direct | Low | Scale [0-1] | Low | Medium |
| digg_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | Direct viral success indicator | Shows total likes received | High | None | Direct measurement from TikTok API | Direct | Low | Log + scale | Low | Low |
| play_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | View count drives algorithm ranking | Shows total video views | High | None | Direct measurement from TikTok API | Direct | Low | Log + scale | Low | Low |
| collect_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | Saves indicate high-value content | Shows bookmark/save actions | High | None | Direct measurement from TikTok API | Direct | Low | Log + scale | Low | Low |
| share_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | Shares drive organic distribution | Shows forward/share actions | High | None | Direct measurement from TikTok API | Direct | Low | Log + scale | Low | Low |
| comment_count | Virality Metrics | Apify Scraper | None | Global | Integer [0-∞] | Comments indicate engagement depth | Shows comment thread activity | High | None | Direct measurement from TikTok API | Direct | Low | Log + scale | Low | Low |
| create_time | Video Metadata | Apify Scraper | None | Global | String (ISO 8601) | Temporal patterns affect performance | Shows when content was published | High | None | Direct timestamp from TikTok metadata | Extract-date | Medium | Cyclical | Medium | Low |
| author | Video Metadata | Apify Scraper | None | Global | String | Creator identity for account analysis | Shows username/unique identifier | High | None | Direct creator info from TikTok metadata | Exclude | None | Exclude | None | Low |
| description | Video Metadata | Apify Scraper | None | Global | String | Caption text for content analysis | Shows video description/caption | High | None | Direct caption text from TikTok metadata | ContentAnalysis | None | ContentAnalysis | None | Low |
| video_id | Video Metadata | Apify Scraper | None | Global | String | Unique identifier for tracking | Internal video identification | High | None | Direct TikTok video identifier | Exclude | None | Exclude | None | Low |
| duration | Video Metadata | Apify Scraper | None | Global | Float [1-600] | Video length affects engagement patterns | Shows total video duration in seconds | High | None | Direct duration from TikTok metadata | Direct | Low | Scale [0-1] | Low | Low |
| gender_detection | Demographics | DeepFace | None | Global | Object | Required for pitch normalization | Shows detected gender and confidence | Medium | None | Gender classification from face analysis | Extract gender_label + one-hot | Low | Extract gender_label + label encode | Low | Medium |
| hashtag_analysis | Hashtags | Hashtag Analysis | Apify metadata | Global | Object | Strategy analysis for viral patterns | Shows hashtag strategy metrics | High | None | Analysis of hashtag types and patterns | ContentAnalysis | None | ContentAnalysis | None | Low |
| processing_timestamp | System Fields | System | None | Global | Float (timestamp) | Pipeline execution tracking | Internal processing metadata | High | None | System timestamp when processing completed | Exclude | None | Exclude | None | None |
| version | System Fields | System | None | Global | String | Pipeline version tracking | Internal version identification | High | None | Temporal compute version identifier | Exclude | None | Exclude | None | None |
| dominant_emotion_id | Emotion | FEAT | expression_timeline | Window-level | Categorical (1-7) | High | Shows emotional hook/CTA | High | No | 1=joy, 2=sadness, 3=anger, 4=fear, 5=disgust, 6=surprise, 7=neutral. Ties: first wins | One-hot (7) | Low | One-hot (7) | Low | O(n) |
| emotional_valence | Emotion | FEAT | expression_timeline | Window-level | Continuous (-1.0 to 1.0) | High | Positive vs negative tone | High | No | (joy -(sadness+anger+fear+disgust))/total. Surprise excluded as ambiguous | Direct | Low | Shift + scale [0-1] | Low | O(n) |
| emotion_consistency | Emotion | FEAT | expression_timeline | Window-level | Continuous (0.0 to 1.0)  | Medium | Shows emotional focus vs chaos | High | No | max(emotion_counts)/total. 1.0=all same, 0.17=all different | Direct | Low | Scale [0-1] | Low | O(n) |

---

## Feature Transform Guide

### Transform Definitions

**Random Forest (RF) Transforms:**
- **Direct**: Use raw values without transformation (RF handles mixed scales well)
- **One-hot**: Convert categorical to binary columns (e.g., 7 emotions → 7 binary features)
- **Label-encode**: Convert categorical strings to integers (e.g., author names → 0, 1, 2, ...)
- **Extract-date**: Parse timestamps into temporal features (hour, day_of_week, month, is_weekend)
- **Extract-fields**: Extract nested fields from complex objects (gender_detection: extract gender_label only)
- **ContentAnalysis**: Text/semantic features analyzed separately (description, hashtag_analysis)
- **Exclude**: System metadata not used in model (video_id, processing_timestamp, version)

**K-Means (KM) Transforms:**
- **Scale [0-1]**: Min-max normalization for already-normalized features
- **Log + scale**: Log1p transform (handles zeros) followed by min-max scaling (for skewed count distributions)
- **Label encode + scale**: Encode categories as integers, then scale to [0-1]
- **One-hot**: Binary encoding for categorical features (same as RF)
- **Cyclical**: Sin/cos encoding for temporal features (hour → hour_sin, hour_cos)
- **Shift + scale**: Shift negative ranges to positive, then scale (e.g., emotional_valence [-1,1] → [0,1])
- **ContentAnalysis**: Text/semantic features analyzed separately (description, hashtag_analysis)
- **Exclude**: Same as RF (system metadata)

### Transform Rationale

**Why Different Transforms for RF vs K-Means?**

**Random Forest:**
- Tree-based algorithm, **scale-invariant** (doesn't care if feature ranges differ)
- Handles **mixed data types** naturally (numerical + categorical)
- Can use **raw counts** directly (scene_count=5 vs scene_count=50 both work)
- **One-hot encoding** helps with categorical features (creates decision splits)
- **Minimal preprocessing** needed (simpler pipeline)

**K-Means:**
- Distance-based algorithm, **scale-sensitive** (euclidean distance calculation)
- Requires **all numerical features** with similar scales
- **Raw counts skew distances** (100,000 views dominates 5 text overlays)
- **Log + scale** compresses skewed distributions (viral outliers don't dominate)
- **Cyclical encoding** preserves circular nature (hour 23 is close to hour 0)
- **Heavy preprocessing** required (complex pipeline)

### Transform Complexity

**Low Complexity:**
- Direct pass-through (no transformation)
- Simple scaling (already [0-1] features)
- Label encoding (string → int mapping)
- One-hot encoding (low cardinality features)

**Medium Complexity:**
- Extract-date features (parse timestamp, create 5+ features)
- Extract-fields (navigate nested objects, extract values)
- Log + scale (two-step transformation)
- Cyclical encoding (sin/cos calculation)

**High Complexity:**
- Text-embed (TF-IDF vectorization or neural embeddings)
- Text-embed + PCA (dimensionality reduction)
- Handling missing values in complex objects

### Feature Groups by Transform Type

**Already Normalized [0-1] (Both RF and KM use directly):**
- average_face_size, speech_coverage, energy_level, energy_max
- pitch_scatter_ratio, eye_contact_rate, emotion_consistency

**Count Features (RF: direct, KM: log + scale):**
- overlay_unique_count, scene_count, object_count, person_count
- gesture_count, word_count, digg_count, play_count, collect_count
- share_count, comment_count

**Variance Features (RF: direct, KM: log + scale):**
- scene_duration_variance, energy_variance, gaze_variance
- (Variance distributions are typically right-skewed)

**Duration Features (RF: direct, KM: log + scale):**
- shortest_scene, longest_scene
- (Scene durations can have extreme outliers)

**Categorical Features (Both: one-hot or label-encode):**
- has_captions (one-hot: 2 categories)
- dominant_emotion_id (one-hot: 7 categories)

**Temporal Features (RF: extract-date, KM: cyclical):**
- create_time → hour, day_of_week, month, is_weekend
- KM uses sin/cos to preserve cyclical nature

**Content Analysis Features (Excluded from ML, analyzed separately):**
- description → ContentAnalysis system (caption/hook/CTA pattern analysis)
- hashtag_analysis → ContentAnalysis system (hashtag strategy analysis)

**Complex Objects (Both: extract-fields + transform):**
- gender_detection → extract gender_label only (one-hot for RF, label-encode for KM)

**System Metadata (Both: exclude):**
- video_id, processing_timestamp, version, author

### Validation Checklist

**Before Training Random Forest:**
- [ ] All categorical features one-hot encoded or label-encoded
- [ ] Text features converted to numerical (TF-IDF or embeddings)
- [ ] Complex objects flattened to individual features
- [ ] System metadata excluded
- [ ] No missing values (impute or drop)

**Before Training K-Means:**
- [ ] All features are numerical (no strings)
- [ ] All features scaled to similar ranges ([0-1] or standardized)
- [ ] Count features log-transformed to reduce skewness
- [ ] Temporal features cyclically encoded (sin/cos)
- [ ] Text embeddings scaled after dimensionality reduction
- [ ] Negative ranges shifted to positive (emotional_valence)
- [ ] System metadata excluded
- [ ] No missing values (impute or drop)

### Implementation Reference

See `feature_transforms.json` for machine-readable configuration mapping each feature to its RF/KM transform with parameters.