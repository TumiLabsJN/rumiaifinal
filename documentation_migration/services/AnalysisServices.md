# Analysis Services

## ⚠️ Legacy Information Warning
This document references current implementation verified through code inspection at:
- `/home/jorge/rumiaifinal/rumiai_v2/ml_services/emotion_detection_service.py`
- `/home/jorge/rumiaifinal/rumiai_v2/ml_services/deepface_gender_service_simple.py`
- `/home/jorge/rumiaifinal/rumiai_v2/processors/video_analyzer.py`

## 🔄 Parallel Execution Architecture
**IMPORTANT**: All services in this group run concurrently with other ML services, NOT sequentially.
- Services are launched in parallel through video_analyzer.py
- Individual service timing cannot be isolated in production without instrumentation
- Services compete for system resources (CPU, memory, GPU, I/O)
- Performance measurements show aggregate time for all parallel services

## 📦 Batch Processing Clarification
**CRITICAL DISTINCTION** - Two types of "batching":
1. **Frame Batching** (WITHIN one video):
   - Processing multiple frames together for efficiency
   - Example: FEAT processes 8 frames per batch on GPU (line 126: batch_size = 8 if self.device == 'cuda' else 4)
   - Example: DeepFace analyzes 2-7 frames based on duration (lines 44-51)
   - This is an optimization technique for single video processing

2. **Video Batching** (NOT IMPLEMENTED):
   - RumiAI processes ONE video at a time
   - No parallel processing of multiple videos
   - Each video goes through the complete pipeline sequentially

## 📊 Service Overview Matrix
| Service | Purpose | Status | Currently Using | GPU Compatible | Output Type | Self-Contained |
|---------|---------|--------|-----------------|----------------|-------------|----------------|
| FEAT | Emotion detection via Action Units | ✅ Production | GPU/CPU | ✅ Yes (CUDA) | ML Data + Timeline | ✅ Yes |
| DeepFace | Gender detection for pitch normalization | ✅ Production | CPU (subprocess) | ⚠️ Optional | ML Data only | ✅ Yes |

---

# FEAT Emotion Detection Service

## 🔄 Execution Context
**This service runs in parallel with other ML services through `video_analyzer.py`.**
- Launched concurrently with YOLO, MediaPipe, Whisper, OCR, Scene Detection, Audio Energy, DeepFace (video_analyzer.py:148-150)
- GPU usage: Competes with YOLO, Whisper for GPU resources
- Thread flexibility: '⚠️ Env vars' (video_analyzer.py:50)

## 🎯 Service Purpose
- **Single sentence**: Detects emotions and Action Units using state-of-the-art facial expression analysis
- **Input type**: Video frames (sampled adaptively based on duration)
- **Output type**: Emotions, confidence scores, Action Units, emotional journey analysis

## ⚡ Performance Profile
```
Execution Time (To Be Confirmed - Pending Instrumentation Tests):
- 60-second video: TBC seconds
- 120-second video: TBC seconds

Resource Usage (To Be Confirmed - Pending Instrumentation Tests):
- Memory: TBC GB peak
- CPU: TBC% average (TBC cores)
- GPU Compatible: ✅ Yes (CUDA)
- GPU Usage (if compatible): TBC% average

Configuration:
- Parallelizable: Yes (batch processing)
- Frame Batching: Yes (8 frames/batch on GPU, 4 on CPU) [line 126]
- Video Processing: Sequential (one video at a time)
- Current Status: ✅ Optimized

IMPORTANT DISTINCTION:
- Frame Batching: Processing multiple frames together within ONE video (for efficiency)
- Video Processing: RumiAI ALWAYS processes one video at a time (never parallel videos)
```

## 📹 Frame Sampling Strategy
```
⚠️ VERIFIED through code inspection (emotion_detection_service.py:76-89)

Adaptive Sampling Rate (based on video duration):
- ≤30s videos: 2.0 FPS [line 85]
- 31-60s videos: 1.0 FPS [line 87]
- >60s videos: 0.5 FPS [line 89]

Actual Implementation:
# From emotion_detection_service.py:76-89
def get_adaptive_sample_rate(self, video_duration: float) -> float:
    if video_duration <= 30:
        return 2.0  # 2 FPS for short videos (max 60 frames)
    elif video_duration <= 60:
        return 1.0  # 1 FPS for medium videos (max 60 frames)
    else:
        return 0.5  # 0.5 FPS for long videos (max 60 frames)

Implementation Location:
└── /rumiai_v2/ml_services/emotion_detection_service.py
    └── get_adaptive_sample_rate() method
        └── Lines: 76-89

Rationale: "Ensures consistent processing time (~15-30s) across all video lengths" [line 79]
Trade-offs: Longer videos have lower temporal resolution but stable processing time
⚠️ Known Issues: None observed in code
```

## 🔍 Self-Containment Check
- [x] Works without precompute imports (verified - no precompute imports found)
- [x] No circular dependencies (uses scipy_compat patches [line 20])
- [x] Clear service boundaries (subprocess for face detection)
- [ ] Isolated test exists: /test_feat_isolation.py (not found)

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                     OUTPUT
Video Frames ─────────> FEAT Processing ─────────> Emotion + AU Data
                           ├── Face Detection (retinaface) [line 49]
                           ├── AU Detection (xgb) [line 51]
                           └── Emotion Classification (resmasknet) [line 52]
```

### Data Flow Pipeline
```
1. Input Stage
   └── Adaptive frame sampling based on video duration

2. Processing Stage
   ├── Step 1: Save frames to temp files [lines 334-343]
   ├── Step 2: Face detection with RetinaFace [line 49]
   ├── Step 3: Action Unit detection with XGBoost [line 51]
   └── Step 4: Emotion classification with ResNet [line 52]

3. Output Stage
   └── {emotions, action_units, timeline, metrics, processing_stats}
```

### Component Breakdown

#### Face Detection (RetinaFace)
- **Model**: 'retinaface' [line 49]
- **Threshold**: FaceScore > 0.5 [lines 382, 400]
- **Multi-face handling**: Selects largest face [lines 388-393]

#### Action Unit Detection (XGBoost)
- **Model**: 'xgb' [line 51]
- **AU columns**: AU01-AU26 detected [lines 258-259]
- **Threshold**: AU intensity > 0.5 for active [line 415]

#### Emotion Classification (ResNet)
- **Model**: 'resmasknet' [line 52]
- **Emotions**: anger, disgust, fear, happiness→joy, sadness, surprise, neutral [lines 63-71]
- **Accuracy claim**: "87% accuracy on AffectNet" [line 31]

### Timeline Integration
```python
# From timeline_builder.py:378-398
timeline_builder.py:_add_emotion_entries() (line 378)
├── Entry type: 'emotion' [line 393]
├── Data structure: {emotion, confidence, action_units, all_scores} [lines 396-398]
└── Validation: ml_validator.validate_emotion_data() [line 385]
```

## 📁 File Structure & Key Locations
```
Service Implementation:
├── /rumiai_v2/ml_services/emotion_detection_service.py (651 lines total)
├── /rumiai_v2/processors/video_analyzer.py (lines 580-654)
└── /scipy_compat.py (compatibility patches, line 20)

Timeline Integration:
└── /rumiai_v2/processors/timeline_builder.py
    └── _add_emotion_entries() (lines 378-398)

Tests:
├── /test_feat_isolation.py (NOT FOUND)
└── /benchmark_emotion.py (NOT FOUND)
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios (from code inspection)
| Failure | Cause | Implementation | Recovery | Line Reference |
|---------|-------|----------------|----------|----------------|
| No faces detected | B-roll/text/animation | no_face_rate > 0.8 | Return 'no_people' type | lines 206-216 |
| Multiple faces | Duets/groups | len(frame_predictions) > 1 | Select largest face | lines 388-393 |
| FEAT crash | Memory/dependency | Exception in _detect_batch | Re-raise (fail fast) | lines 198-199 |
| Face score low | Poor detection | FaceScore <= 0.5 | Mark as no_face | lines 382-384 |
| Import error | scipy compatibility | ImportError on feat | Apply patches first | lines 19-23 |

### Graceful Degradation Strategy (verified in code)
- **No people handling**: Returns valid structure with 'no_people' video_type [lines 207-216]
- **Processing stats**: Tracks successful_frames, no_face_frames [lines 114-119]
- **Multi-face selection**: Picks face with largest area [lines 389-392]
- **Temp file cleanup**: Always cleans up temp files [lines 349-354]

## 🐛 Current Issues & Future Fixes

### Priority: HIGH 🔴
- **Issue**: scipy compatibility patches required
- **Evidence**: scipy_compat import required [line 20]
- **Current Workaround**: ensure_feat_compatibility() called before FEAT import
- **Files Affected**: emotion_detection_service.py lines 19-23

### Priority: MEDIUM 🟡
- **Issue**: Temporary file I/O overhead
- **Evidence**: FEAT requires file paths, not arrays [lines 331-343]
- **Impact**: Creates temp files for every frame batch
- **Current Implementation**: tempfile.NamedTemporaryFile used [line 340]

## 🧪 Testing & Validation

### Two Types of Testing

#### 1. Functional Testing (Isolation)
- **Purpose**: Verify service works correctly
- **NOT for**: Performance measurement
- **Use**: Debugging, dependency verification
- **Location**: `/test_feat_isolation.py` (TODO - not found)

#### 2. Performance Testing (Full Pipeline)
- **Purpose**: Measure actual performance with resource competition
- **Status**: Pending proper instrumentation setup
- **Metrics**: Currently marked as TBC (To Be Confirmed)
- **Note**: Uses ThreadMonitor for instrumentation [line 584]

## 📈 Optimization Opportunities
- [x] **Frame Batching**: Already implemented (8 GPU / 4 CPU) [line 126]
- [x] **GPU Acceleration**: torch.cuda.is_available() check [line 649]
- [x] **Adaptive sampling**: Based on video duration [lines 76-89]
- [ ] **Direct array input**: Currently requires temp files

## 🔄 Dependencies
```
External Libraries (verified from imports):
├── feat (line 47: from feat import Detector)
├── torch (line 12: import torch)
├── cv2 (line 9: import cv2)
├── numpy (line 7: import numpy as np)
└── scipy (via scipy_compat patches, line 20)

Internal Dependencies:
├── ThreadMonitor (video_analyzer.py:584)
├── MLAnalysisResult (video_analyzer.py:645-654)
└── Timeline/TimelineEntry (timeline_builder.py:392-398)
```

---

# DeepFace Gender Detection Service

## 🔄 Execution Context
**This service runs in parallel with other ML services through `video_analyzer.py`.**
- Launched concurrently with all other services (video_analyzer.py:149)
- Subprocess isolation to avoid TensorFlow memory corruption [line 3-5 comments]
- Thread flexibility: '⚠️ Env vars' (video_analyzer.py:56)

## 🎯 Service Purpose
- **Single sentence**: Detects gender for pitch normalization in voice analysis
- **Input type**: Video path (passed to subprocess) [line 37]
- **Output type**: Gender classification with confidence score [lines 63-68]

## ⚡ Performance Profile
```
Execution Time (To Be Confirmed - Pending Instrumentation Tests):
- 60-second video: TBC seconds
- 120-second video: TBC seconds

Resource Usage (To Be Confirmed - Pending Instrumentation Tests):
- Memory: TBC GB peak (subprocess isolated)
- CPU: TBC% average (TBC cores)
- GPU Compatible: ⚠️ Optional (TensorFlow backend)
- GPU Usage: Currently CPU-only due to subprocess isolation

Configuration:
- Parallelizable: No (sequential frame processing)
- Frame Batching: No (processes frames individually) [lines 76-114]
- Video Processing: Sequential (one video at a time)
- Current Status: ✅ Stable (subprocess isolation)
- Timeout: 30 seconds [line 49]
```

## 📹 Frame Sampling Strategy
```
⚠️ VERIFIED through code inspection (run_deepface_gender.py:44-51)

Adaptive Sampling (based on video duration):
- <5s videos: 2 frames [line 45]
- 5-15s videos: 3 frames [line 47]
- 15-30s videos: 5 frames [line 49]
- >30s videos: 7 frames [line 51]

Actual Implementation:
# From run_deepface_gender.py:44-51
if duration < 5:
    num_frames = 2
elif duration < 15:
    num_frames = 3
elif duration < 30:
    num_frames = 5
else:
    num_frames = 7

# Line 54: Frames sampled evenly
frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

Implementation Location:
└── /scripts/run_deepface_gender.py
    └── analyze_video() function
        └── Lines: 44-54

Face Size Filter: 120x120 pixels minimum [line 94]
Multi-person Detection: Returns 'multiple_people' [line 119]
```

## 🔍 Self-Containment Check
- [x] Works without precompute imports (verified - standalone script)
- [x] No circular dependencies (subprocess isolation) [lines 38-43]
- [x] Clear service boundaries (JSON communication) [lines 70-76]
- [ ] Isolated test exists: /test_deepface_isolation.py (not found)

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                     OUTPUT
Video Path ─────────> Subprocess Runner ─────────> JSON Result
   [line 37]              ├── Python3 subprocess [line 37]
                         ├── run_deepface_gender.py
                         └── JSON parsing [lines 73-76]
```

### Data Flow Pipeline
```
1. Input Stage (deepface_gender_service_simple.py)
   └── Video path validation [lines 32-33]

2. Processing Stage (run_deepface_gender.py)
   ├── Step 1: Extract frames based on duration [lines 44-62]
   ├── Step 2: DeepFace.analyze per frame [lines 78-84]
   ├── Step 3: Filter faces >120x120px [lines 88-95]
   └── Step 4: Vote aggregation [lines 139-159]

3. Output Stage
   └── JSON output to stdout [line 193]
```

### Special Cases Handling (verified in code)

#### Multiple People Detection
- **Detection**: multi_person_count > 0 [line 117]
- **Output**: {'gender': 'multiple_people'} [line 119]
- **Note**: "use self-normalization for pitch" [line 125]

#### No Face Detection
- **Check**: not gender_votes [line 129]
- **Output**: {'gender': None, 'error': 'no_faces_detected'} [lines 131-136]

#### Subprocess Timeout
- **Timeout**: 30 seconds [deepface_gender_service_simple.py:49]
- **Output**: {'error': 'timeout_30s'} [line 58]

## 📁 File Structure & Key Locations
```
Service Implementation:
├── /rumiai_v2/ml_services/deepface_gender_service_simple.py (102 lines)
├── /scripts/run_deepface_gender.py (199 lines)
└── /rumiai_v2/processors/video_analyzer.py (lines 656-713)

Script Path Resolution:
└── Path(__file__).parent.parent.parent / 'scripts' / 'run_deepface_gender.py' [line 24]

Output:
└── ml_data['deepface_gender'] (not added to timeline)
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios (from code inspection)
| Failure | Code Location | Impact | Recovery | Line Reference |
|---------|---------------|--------|----------|----------------|
| Subprocess timeout | asyncio.TimeoutError | No gender data | 30s timeout, return error | simple.py:46-59 |
| Multiple people | multi_person_count > 0 | Ambiguous | Return 'multiple_people' | gender.py:117-126 |
| No faces | len(gender_votes) == 0 | No data | Return None + error | gender.py:129-136 |
| Process failed | returncode != 0 | Subprocess error | Return error dict | simple.py:61-68 |
| No JSON output | json_start < 0 | Parse fail | Return error dict | simple.py:73-84 |
| Face too small | width < 120 or height < 120 | Filtered | Skip face | gender.py:94 |

### Subprocess Communication (verified)
- **Command**: ['python3', script_path, video_path] [line 37]
- **Timeout handling**: asyncio.wait_for with 30s [lines 46-59]
- **JSON extraction**: Finds '{' in stdout to skip warnings [lines 73-76]
- **Process cleanup**: process.kill() on timeout [lines 52-53]

## 🐛 Current Issues & Future Fixes

### Priority: MEDIUM 🟡
- **Issue**: Subprocess overhead per video
- **Evidence**: New subprocess for each analyze() call [lines 38-43]
- **Impact**: ~500ms startup overhead
- **Current Implementation**: asyncio.create_subprocess_exec [lines 39-43]

### Priority: LOW 🟢
- **Issue**: OpenCV backend less accurate
- **Evidence**: detector_backend='opencv' [gender.py:82]
- **Alternative**: Could use 'retinaface' for better accuracy

## 🧪 Testing & Validation

### Two Types of Testing

#### 1. Functional Testing (Isolation)
- **Purpose**: Verify gender detection accuracy
- **Script mode**: Can run standalone [gender.py:171-199]
- **Usage**: python3 scripts/run_deepface_gender.py <video_path>

#### 2. Performance Testing (Full Pipeline)
- **Purpose**: Measure subprocess overhead
- **Instrumentation**: ThreadMonitor in video_analyzer.py
- **Metrics**: processing_ms tracked [gender.py:124, 135, 167]

## 📈 Optimization Opportunities
- [ ] **Service Mode**: Replace subprocess with long-running service
- [ ] **GPU Support**: Currently CPU-only (TF_CPP_MIN_LOG_LEVEL='3') [line 19]
- [ ] **Detector backend**: Switch from 'opencv' to 'retinaface'
- [x] **Face filtering**: Already filters <120px faces [line 94]

## 🔄 Dependencies
```
External Libraries (from imports in run_deepface_gender.py):
├── deepface (line 21: from deepface import DeepFace)
├── cv2 (line 14: import cv2)
├── numpy (line 15: import numpy as np)
└── tensorflow (suppressed warnings, line 19)

Internal Dependencies:
├── MLAnalysisResult (video_analyzer.py)
├── ThreadMonitor (video_analyzer.py)
└── subprocess isolation (deepface_gender_service_simple.py)
```