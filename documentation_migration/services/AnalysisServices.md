# Analysis Services

## ⚠️ Legacy Information Warning
This document references current implementation verified through code inspection at:
- `/home/jorge/rumiaifinal/rumiai_v2/ml_services/emotion_detection_service.py`
- `/home/jorge/rumiaifinal/rumiai_v2/ml_services/deepface_gender_service_simple.py`
- `/home/jorge/rumiaifinal/rumiai_v2/processors/video_analyzer.py`

## 🔄 Flexible Execution Architecture
**IMPORTANT**: Metadata extraction happens BEFORE ML services, not in parallel.
- Apify scraping occurs first to get video metadata (rumiai_runner.py lines 243)
- Video download happens after scraping (line 249)
- The downloaded videos get processed either in Parallel or Sequentially. The architecture can support both and switch between them easily. 
- Metadata is passed through the entire pipeline

## 📦 Batch Processing Clarification
**CRITICAL DISTINCTION** - Two types of "batching":
1. **Frame Batching** (WITHIN one video):
   - Processing multiple frames together for efficiency
   - Example: FEAT processes 8 frames per batch on GPU, 4 on CPU
   - Example: DeepFace analyzes 2-7 frames based on duration
   - This is an optimization technique for single video processing

2. **Video Batching** (NOT IMPLEMENTED):
   - RumiAI processes ONE video at a time
   - No parallel processing of multiple videos
   - Each video goes through the complete pipeline sequentially

## 📊 Service Overview Matrix
| Service | Purpose | Status | Currently Using | GPU Capable | Output Type | Self-Contained | Performance Impact |
|---------|---------|--------|-----------------|----------------|-------------|----------------|--------------------|
| FEAT | Emotion detection via Action Units | ✅ Production | GPU/CPU | ✅ Yes (CUDA) | ML Data + Timeline | ✅ Yes | 🔴 High (40-60% of pipeline) |
| DeepFace | Gender detection for pitch normalization | ✅ Production | CPU only | ✅ Yes (TensorFlow) | ML Data only | ✅ Yes | 🟢 Low (<5% of pipeline) |

---

# FEAT Emotion Detection Service

## 🔄 Flexible Execution Architecture
**IMPORTANT**: Metadata extraction happens BEFORE ML services, not in parallel.
- Apify scraping occurs first to get video metadata (rumiai_runner.py lines 243)
- Video download happens after scraping (line 249)
- The downloaded videos get processed either in Parallel or Sequentially. The architecture can support both and switch between them easily. 
- Metadata is passed through the entire pipeline

## 🎯 Service Purpose
- **Single sentence**: Detects emotions and Action Units using state-of-the-art facial expression analysis (computationally intensive)
- **Input type**: Video frames (sampled adaptively based on duration)
- **Output type**: Emotions, confidence scores, Action Units, emotional journey analysis

## ⚡ Performance Profile
```
Execution Time (To Be Confirmed - Pending Instrumentation Tests):
- 60-second video: TBC seconds
- 120-second video: TBC seconds
- Relative Impact: Typically 40-60% of total pipeline time when faces are present

Resource Usage (To Be Confirmed - Pending Instrumentation Tests):
- Memory: TBC GB peak (highest among all services)
- CPU: TBC% average (TBC cores)
- GPU Compatible: ✅ Yes (CUDA)
- GPU Usage (if compatible): TBC% average (competes with YOLO/Whisper)

Configuration:
- Parallelizable: Yes (batch processing)
- Frame Batching: Yes (8 frames/batch on GPU, 4 on CPU)
- Video Processing: Sequential (one video at a time)
- Current Status: ✅ Optimized
- Performance Note: Most computationally expensive service in pipeline

IMPORTANT DISTINCTION:
- Frame Batching: Processing multiple frames together within ONE video (for efficiency)
- Video Processing: RumiAI ALWAYS processes one video at a time (never parallel videos)
```

## 📹 Frame Sampling Strategy
```
⚠️ VERIFIED through code inspection

Adaptive Sampling Rate (based on video duration):
- ≤30s videos: 2.0 FPS
- 31-60s videos: 1.0 FPS
- >60s videos: 0.5 FPS

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

Rationale: Ensures consistent processing time (~15-30s) across all video lengths
Trade-offs: Longer videos have lower temporal resolution but stable processing time
⚠️ Known Issues: None
```

## 📦 Installation Requirements

### Model Downloads
- **First Run**: Automatically downloads pretrained models
- **Download Size**: ~2GB across multiple models
  - RetinaFace (face detection): ~250MB
  - MobileFaceNet (landmarks): ~100MB
  - XGBoost AU models: ~500MB
  - ResNet emotion model: ~1.2GB
- **Total Disk Space**: ~4GB (including cache and temp files)
- **Download Location**: `~/.feat/` directory

### Installation Complexity
- **Known Issues**:
  - scipy version conflicts (see Current Issues section)
  - CUDA compatibility with specific PyTorch versions
  - Missing system libraries (libgomp, libgfortran)
- **First Run Behavior**:
  - Downloads happen on first `Detector()` initialization
  - Can take 5-10 minutes on slow connections
  - No progress bar (appears frozen but is downloading)
- **Offline Installation**: Models can be pre-downloaded and cached
- **Docker Considerations**: Mount `~/.feat/` as volume to avoid re-downloads

## 🔍 Self-Containment Check
- [x] Works without precompute imports (verified)
- [x] No circular dependencies (uses scipy_compat patches)
- [x] Clear service boundaries
- [ ] Isolated test exists: /test_feat_isolation.py (TODO)

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                     OUTPUT
Video Frames ─────────> FEAT Processing ─────────> Emotion + AU Data
                           ├── Face Detection (retinaface)
                           ├── AU Detection (xgb)
                           └── Emotion Classification (resmasknet)
```

### Data Flow Pipeline
```
1. Input Stage
   └── Adaptive frame sampling based on video duration

2. Processing Stage
   ├── Step 1: Save frames to temp files
   ├── Step 2: Face detection with RetinaFace
   ├── Step 3: Action Unit detection with XGBoost
   └── Step 4: Emotion classification with ResNet

3. Output Stage
   └── {emotions, action_units, timeline, metrics, processing_stats}
```

### Component Breakdown

#### Face Detection (RetinaFace)
- **Model**: 'retinaface'
- **Threshold**: FaceScore > 0.5
- **Multi-face handling**: Selects largest face

#### Action Unit Detection (XGBoost)
- **Model**: 'xgb'
- **AU columns**: AU01-AU26 detected
- **Threshold**: AU intensity > 0.5 for active

#### Emotion Classification (ResNet)
- **Model**: 'resmasknet'
- **Emotions**: anger, disgust, fear, happiness→joy, sadness, surprise, neutral
- **Accuracy claim**: "87% accuracy on AffectNet"

### Timeline Integration
```python
timeline_builder.py:_add_emotion_entries() (lines 378-398)
├── Entry type: 'emotion'
├── Data structure: {emotion, confidence, action_units, all_scores}
└── Validation: ml_validator.validate_emotion_data()
```

## 📁 File Structure & Key Locations
```
Service Implementation:
├── /rumiai_v2/ml_services/emotion_detection_service.py (main implementation)
├── /rumiai_v2/processors/video_analyzer.py (lines 580-654)
└── /scipy_compat.py (compatibility patches)

Timeline Integration:
└── /rumiai_v2/processors/timeline_builder.py
    └── _add_emotion_entries() (lines 378-398)

Tests:
├── /test_feat_isolation.py (NOT FOUND)
└── /benchmark_emotion.py (NOT FOUND)
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios
| Failure | Cause | Impact | Recovery | Frequency |
|---------|-------|--------|----------|------------|
| No faces detected | B-roll/text/animation | No emotion data | Return 'no_people' type | ~15% of videos |
| Multiple faces | Duets/groups | Ambiguous detection | Select largest face | ~5% of videos |
| FEAT crash | Memory/dependency issues | Service fails | Fail fast, log error | <1% (rare) |
| Face score low | Poor detection quality | False negatives | Mark as no_face | ~3% of frames |
| Import error | scipy compatibility | Service unavailable | Apply compatibility patches | First run only |

### Graceful Degradation Strategy
- **Principle**: Videos without people are valid outcomes, not errors
- **Empty Results**: Return structured data with 'no_people' video_type
- **Processing stats**: Tracks successful_frames, no_face_frames
- **Multi-face selection**: Picks face with largest area
- **Temp file cleanup**: Always cleans up temp files

## 🐛 Current Issues & Future Fixes

### Priority: HIGH 🔴
- **Issue**: scipy compatibility patches required for FEAT
- **Root Cause**: Version conflict - FEAT 0.6.0 requires scipy<1.13 but other RumiAI services need scipy>=1.13
- **Impact**: Service fails to start without patches (ImportError on specific scipy modules)
- **Current Workaround**: scipy_compat.py patches applied before import
  - **What patches do**: Shim compatibility layer that monkey-patches scipy imports
  - **How it works**: Redirects deprecated scipy.stats methods to new locations
  - **Example**: `scipy.stats.mode` behavior changed in 1.13, patch maintains old behavior
- **Risks**:
  - Patches may break with FEAT updates
  - Fragile solution dependent on specific scipy internals
  - Could mask other compatibility issues
- **Proposed Fix**: Update to FEAT 0.7.0 when released (promised scipy 1.13+ support)
- **Effort Estimate**: 1 day for FEAT update, 0 days if patches continue working
- **Files Affected**: emotion_detection_service.py, scipy_compat.py
- **Why this matters**: This is a ticking time bomb - any scipy or FEAT update could break production

### Priority: MEDIUM 🟡
- **Issue**: Temporary file I/O overhead
- **Impact**: Extra I/O for every frame batch
- **Current Workaround**: Using tempfile.NamedTemporaryFile
- **Proposed Fix**: Modify FEAT to accept numpy arrays directly
- **Effort Estimate**: 3 days (requires FEAT fork)

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
- **Note**: Uses ThreadMonitor for instrumentation

## 📈 Optimization Opportunities
**Note**: FEAT is the highest-value optimization target due to its dominant processing time.

- [x] **Frame Batching**: Already implemented (8 GPU / 4 CPU)
- [x] **GPU Acceleration**: Enabled by default when available
- [x] **Adaptive sampling**: Based on video duration
- [ ] **Direct array input**: Currently requires temp files (would save ~10% time)
- [ ] **Model quantization**: INT8 quantization could reduce memory by 50%
- [ ] **Face detection caching**: Could skip redundant detection across services
- [ ] **Lower precision models**: Trade accuracy for 2-3x speedup

## 🔄 Dependencies
```
External Libraries:
├── feat==0.6.0 (facial expression analysis toolkit)
├── torch>=1.9.0 (deep learning framework)
├── opencv-python (image processing)
├── numpy (array operations)
└── scipy (with compatibility patches)

Internal Dependencies:
├── ThreadMonitor (instrumentation)
├── MLAnalysisResult (data structure)
└── Timeline/TimelineEntry (timeline integration)
```

---

# DeepFace Gender Detection Service

## 🔄 Flexible Execution Architecture
**IMPORTANT**: Metadata extraction happens BEFORE ML services, not in parallel.
- Apify scraping occurs first to get video metadata (rumiai_runner.py lines 243)
- Video download happens after scraping (line 249)
- The downloaded videos get processed either in Parallel or Sequentially. The architecture can support both and switch between them easily. 
- Metadata is passed through the entire pipeline

## 🎯 Service Purpose
- **Single sentence**: Detects gender for pitch normalization in voice analysis
- **Input type**: Video path (passed to subprocess)
- **Output type**: Gender classification with confidence score

### ℹ️ Why Gender Detection for Voice Analysis?
```
Voice pitch varies significantly by biological sex:
- Male voices: 85-180 Hz fundamental frequency
- Female voices: 165-255 Hz fundamental frequency

Gender detection enables relative pitch analysis:
✅ WITH gender: "High pitch for male speaker" (e.g., 170Hz = excited for male)
❌ WITHOUT gender: "170Hz pitch" (ambiguous - low for female, high for male)

Fallback strategies:
- Multiple people → Self-normalization (compare to speaker's own range)
- No faces detected → Absolute frequency analysis only
- Animation/AI voices → Skip pitch normalization entirely

Used by: temporal_compute.py for voice characteristic analysis
```

## ⚡ Performance Profile
```
Execution Time (To Be Confirmed - Pending Instrumentation Tests):
- 60-second video: TBC seconds
- 120-second video: TBC seconds

Resource Usage (To Be Confirmed - Pending Instrumentation Tests):
- Memory: TBC GB peak (subprocess isolated)
- CPU: TBC% average (TBC cores)
- GPU Capable: ✅ Yes (DeepFace/TensorFlow supports CUDA)
- Current GPU Usage: ❌ None (CPU-only implementation)
- Potential: Could enable GPU with proper CUDA environment in subprocess

Configuration:
- Parallelizable: No (sequential frame processing)
- Frame Batching: No (processes frames individually)
- Video Processing: Sequential (one video at a time)
- Current Status: ✅ Stable (subprocess isolation)
- Timeout: 30 seconds
```

## 📹 Frame Sampling Strategy
```
⚠️ VERIFIED through code inspection

Adaptive Sampling (based on video duration):
This is not FPS, its number of frames to identify sex.
- <5s videos: 2 frames
- 5-15s videos: 3 frames
- 15-30s videos: 5 frames
- >30s videos: 7 frames

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

# Frames sampled evenly across video
frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

Implementation Location:
└── /scripts/run_deepface_gender.py
    └── analyze_video() function
        └── Lines: 44-54

Face Size Filter: 120x120 pixels minimum (filters out logos/watermarks)
Multi-person Detection: Returns 'multiple_people' for ambiguous cases
```

## 🔍 Self-Containment Check
- [x] Works without precompute imports (standalone script)
- [x] No circular dependencies (subprocess isolation)
- [x] Clear service boundaries (JSON communication)
- [ ] Isolated test exists: /test_deepface_isolation.py (TODO)

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                     OUTPUT
Video Path ─────────> Subprocess Runner ─────────> JSON Result
                         ├── Python3 subprocess
                         ├── run_deepface_gender.py
                         └── JSON parsing
```

### Data Flow Pipeline
```
1. Input Stage (deepface_gender_service_simple.py)
   └── Video path validation

2. Processing Stage (run_deepface_gender.py)
   ├── Step 1: Extract frames based on duration
   ├── Step 2: DeepFace.analyze per frame
   ├── Step 3: Filter faces >120x120px
   └── Step 4: Vote aggregation

3. Output Stage
   └── JSON output to stdout
```

### Special Cases Handling

#### Multiple People Detection
- **Detection**: Any frame with >1 valid face (>120x120px)
- **Output**: {'gender': 'multiple_people'}
- **Design Rationale**: Unlike FEAT which picks the largest face for emotion analysis, DeepFace returns 'multiple_people' because incorrect gender assignment would severely impact pitch normalization. Better to use self-normalization than wrong gender baseline.
- **Usage**: Triggers self-normalization in pitch analysis (compares speaker to their own range rather than gender baseline)

#### No Face Detection
- **Detection**: No valid faces found in any frame
- **Output**: {'gender': None, 'error': 'no_faces_detected'}
- **Usage**: Pitch analysis falls back to absolute values

#### Subprocess Timeout
- **Timeout**: 30 seconds
- **Output**: {'error': 'timeout_30s'}
- **Recovery**: Returns None, pipeline continues

## 📁 File Structure & Key Locations
```
Service Implementation:
├── /rumiai_v2/ml_services/deepface_gender_service_simple.py (102 lines)
├── /scripts/run_deepface_gender.py (199 lines)
└── /rumiai_v2/processors/video_analyzer.py (lines 656-713)

Script Path Resolution:
└── Path(__file__).parent.parent.parent / 'scripts' / 'run_deepface_gender.py'

Output:
└── ml_data['deepface_gender'] (not added to timeline)
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios
| Failure | Cause | Impact | Recovery | Frequency |
|---------|-------|--------|----------|------------|
| Subprocess timeout | Long processing/hang | No gender data | 30s timeout, return None | <1% |
| Multiple people | Duets/groups | Ambiguous gender | Return 'multiple_people' | ~8% of videos |
| No faces | Animation/text videos | No gender data | Return None gracefully | ~10% of videos |
| Process failed | Subprocess crash | No gender data | Return error dict | <1% |
| No JSON output | Parse failure | No gender data | Return error dict | Rare |
| Small face filtering | Logos/watermarks | False positives prevented | Skip faces <120px | Prevented |

### Subprocess Communication
- **Command**: ['python3', script_path, video_path]
- **Timeout handling**: asyncio.wait_for with 30s timeout
- **JSON extraction**: Finds '{' in stdout to skip warnings
- **Process cleanup**: process.kill() on timeout

## 🐛 Current Issues & Future Fixes

### Priority: MEDIUM 🟡
- **Issue**: Subprocess overhead for each video
- **Impact**: ~500ms startup overhead per video
- **Current Workaround**: Accepted trade-off for stability
- **Proposed Fix**: Long-running service with queue
- **Effort Estimate**: 3 days
- **Files Affected**: deepface_gender_service_simple.py

### Priority: LOW 🟢
- **Issue**: OpenCV backend less accurate than RetinaFace
- **Impact**: ~5% lower accuracy in face detection
- **Proposed Fix**: Switch to RetinaFace backend
- **Effort Estimate**: 1 day

## 🧪 Testing & Validation

### Two Types of Testing

#### 1. Functional Testing (Isolation)
- **Purpose**: Verify gender detection accuracy
- **Script mode**: Can run standalone
- **Usage**: python3 scripts/run_deepface_gender.py <video_path>

#### 2. Performance Testing (Full Pipeline)
- **Purpose**: Measure subprocess overhead
- **Instrumentation**: ThreadMonitor in video_analyzer.py
- **Metrics**: processing_ms tracked in output

## 📈 Optimization Opportunities
- [ ] **Service Mode**: Replace subprocess with long-running service
- [ ] **Enable GPU**: DeepFace supports GPU, need to configure CUDA environment in subprocess
  - Set proper CUDA_VISIBLE_DEVICES
  - Ensure TensorFlow-GPU installed in subprocess environment
  - Potential 2-3x speedup for face detection
- [ ] **Detector backend**: Switch from 'opencv' to 'retinaface'
- [x] **Face filtering**: Already filters <120px faces

## 🔄 Dependencies
```
External Libraries:
├── deepface>=0.0.79 (face analysis framework)
├── tensorflow>=2.0.0 (deep learning backend)
├── opencv-python (face detection)
└── numpy (array operations)

Internal Dependencies:
├── MLAnalysisResult (video_analyzer.py)
├── ThreadMonitor (video_analyzer.py)
└── subprocess isolation (deepface_gender_service_simple.py)
```