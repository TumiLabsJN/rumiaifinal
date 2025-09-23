# Gender Detection Services and Methods
# Decision = Deepface

## Available Services for Gender Detection

### 1. Face-Based Gender Detection APIs

#### Amazon Rekognition
```python
# Detects gender from face with confidence score
response = rekognition.detect_faces(
    Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
    Attributes=['ALL']
)
# Returns: Gender {Value: 'Male', Confidence: 99.2}
```
- **Accuracy**: 90-95% on clear faces
- **Cost**: $0.001 per image
- **Issues**: Requires visible face, front-facing

#### Microsoft Azure Face API
```python
face_attributes = face_client.face.detect_with_url(
    url=image_url,
    return_face_attributes=['age', 'gender', 'emotion']
)
# Returns: gender: 'male' or 'female'
```
- **Accuracy**: 92-96% on clear faces
- **Cost**: $0.001 per 1000 calls
- **Note**: Being phased out for ethical reasons

#### Google Cloud Vision API
```python
# Note: Google REMOVED gender detection in 2020
# They only provide face detection, not gender
# This was an ethical decision
```

#### Face++ (Megvii)
```python
# Commercial API, primarily used in Asia
result = facepp.detect(
    image_file=image,
    return_attributes='gender,age'
)
# Returns: gender with confidence
```
- **Accuracy**: 95%+ claimed
- **Privacy concerns**: Chinese company, data sovereignty issues

### 2. Open Source Face Analysis

#### DeepFace (Facebook Research)
```python
from deepface import DeepFace

result = DeepFace.analyze(
    img_path="image.jpg",
    actions=['age', 'gender', 'race', 'emotion']
)
# Returns: {'gender': 'Woman', 'gender_confidence': 0.92}
```
- **Accuracy**: 91-94% on clear faces
- **Cost**: Free, runs locally
- **Requirements**: 1-2GB model download

#### InsightFace
```python
import insightface

app = insightface.app.FaceAnalysis()
faces = app.get(image)
# Returns gender, age, embedding
```
- **Accuracy**: 93-96%
- **Performance**: Fast, optimized
- **Models**: Multiple options (Buffalo, RetinaFace)

### 3. Voice-Based Gender Detection

#### Hume AI
```python
# Emotion + demographic detection from voice
result = hume_client.predict(
    audio_file=audio,
    models=['prosody', 'demographics']
)
# Returns gender probability from voice
```
- **Accuracy**: 85-88% from voice
- **Multimodal**: Can combine face + voice

#### pyAudioAnalysis
```python
from pyAudioAnalysis import audioTrainTest as aT

# Requires training on labeled data
result = aT.file_classification(audio_file, model_path, "svm")
# Returns: male/female classification
```
- **Accuracy**: 75-82% voice only
- **Limitation**: Needs training data

### 4. Multimodal Approaches

#### Combined Face + Voice
```python
def detect_gender_multimodal(video_path):
    # 1. Extract faces
    face_gender = deepface_analysis(video_frames)
    face_confidence = 0.94
    
    # 2. Extract voice features
    pitch_avg = extract_pitch(audio)
    voice_gender_prob = pitch_to_gender_probability(pitch_avg)
    
    # 3. Combine with weights
    if face_confidence > 0.9:
        return face_gender  # Trust face if high confidence
    else:
        # Weight by confidence
        combined = (face_gender * face_confidence + 
                   voice_gender_prob * 0.5)
        return combined
```

---

## Our Current System Capabilities

### What we already have:

1. **MediaPipe Face Detection**
   - We detect faces and face bounding boxes
   - Could add gender classification layer
   - Would need to integrate gender model

2. **Emotion Detection Service**
   - Already analyzing faces
   - Could extend to include gender
   - Same frames, additional analysis

3. **Audio Analysis Pipeline**
   - Already extracting pitch
   - Could add gender classification
   - But accuracy issues (70% max)

---

## Implementation Options for RumiAI

### Option 1: Add DeepFace to Emotion Service
```python
# In emotion_detection_service.py
from deepface import DeepFace

class EmotionDetectionService:
    def analyze_frame(self, frame):
        # Existing emotion detection
        emotion_result = self.detect_emotion(frame)
        
        # Add gender detection
        analysis = DeepFace.analyze(
            frame,
            actions=['gender'],
            enforce_detection=False
        )
        
        gender = analysis[0]['gender'] if analysis else 'unknown'
        gender_confidence = analysis[0]['gender_confidence'] if analysis else 0
        
        return {
            **emotion_result,
            'gender': gender,
            'gender_confidence': gender_confidence
        }
```

**Pros**:
- High accuracy (91-94%)
- Free, runs locally
- Already processing same frames

**Cons**:
- Extra processing time (+100-200ms per frame)
- Large model size (1-2GB)
- Ethical concerns

### Option 2: AWS Rekognition Integration
```python
# New service: gender_detection_service.py
import boto3

class GenderDetectionService:
    def __init__(self):
        self.client = boto3.client('rekognition')
    
    async def detect_gender(self, frame_samples):
        # Sample 5-10 frames across video
        gender_votes = []
        
        for frame in frame_samples:
            response = self.client.detect_faces(
                Image={'Bytes': frame},
                Attributes=['GENDER']
            )
            
            if response['FaceDetails']:
                gender = response['FaceDetails'][0]['Gender']['Value']
                confidence = response['FaceDetails'][0]['Gender']['Confidence']
                gender_votes.append((gender.lower(), confidence))
        
        # Majority vote
        return self.aggregate_votes(gender_votes)
```

**Pros**:
- Very accurate (95%+)
- Handles edge cases well
- No local compute needed

**Cons**:
- Costs money ($0.001 per image)
- Requires AWS setup
- Network latency
- Privacy concerns (sending faces to cloud)

### Option 3: Hybrid Confidence-Based
```python
def detect_gender_with_confidence(video_data):
    """
    Try multiple methods, use most confident
    """
    results = []
    
    # 1. Face-based (if faces detected)
    if video_data.get('face_detections'):
        face_gender = detect_from_face(video_data['faces'])
        results.append(('face', face_gender, 0.92))
    
    # 2. Voice-based (if speech exists)
    if video_data.get('speech_coverage') > 0.5:
        voice_gender = detect_from_voice(video_data['pitch'])
        results.append(('voice', voice_gender, 0.75))
    
    # 3. Metadata (if creator profile exists)
    if video_data.get('creator_id'):
        profile_gender = get_creator_profile_gender()
        results.append(('profile', profile_gender, 1.0))
    
    # Return highest confidence
    return max(results, key=lambda x: x[2])
```

---

## Ethical and Practical Considerations

### Why Major Companies Are Moving Away

1. **Google**: Removed gender detection in 2020
2. **Microsoft**: Limiting access, phasing out
3. **IBM**: Exited facial recognition entirely

**Reasons**:
- Reinforces binary gender assumptions
- High error rates for minorities
- Trans and non-binary exclusion
- Legal liability (GDPR, CCPA)
- Reputation risk

### If We Must Do It

**Best Practices**:
1. Make it optional (creators opt-in)
2. Allow manual override/correction
3. Use "masculine/feminine" not "male/female"
4. Include "not detected" option
5. Never use for access control or pricing

---

## Recommendation for RumiAI

### Short Term: Don't Implement Gender Detection

**Use log-scale normalization instead**:
```python
def normalize_pitch_without_gender(pitch_hz):
    """Gender-agnostic normalization"""
    return np.log2(pitch_hz / 110)  # Musical scale
```

### Long Term: Optional Creator Self-Identification

```python
# Let creators specify if they want
creator_profile = {
    'voice_type': 'tenor',  # Or alto, soprano, bass, etc.
    'preferred_baseline': 165,  # Hz
    'normalization': 'musical'  # Or 'statistical', 'none'
}
```

**This approach**:
- ✅ Respects creator identity
- ✅ More accurate (self-reported)
- ✅ Avoids ethical issues
- ✅ Builds trust
- ✅ Legal compliance

### If Forced to Implement

Use **DeepFace** locally:
- Free, no cloud privacy issues
- 91-94% accurate on faces
- Can run on same frames as emotion detection
- Add 200ms processing time per video

But clearly document:
- It's optional
- It can be wrong
- It's for normalization only
- Creators can override

---

## The Bottom Line

**We CAN detect gender** (91-95% accuracy with faces)
**We SHOULDN'T** (ethical, legal, trust issues)
**We DON'T NEED TO** (log-scale works without it)

If business absolutely requires it, use DeepFace locally with opt-in and override options.