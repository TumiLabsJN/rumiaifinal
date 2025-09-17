# RumiAI Service Integration Reference

A comprehensive guide for integrating new ML services into RumiAI based on real implementation experience.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Dependency Hell](#dependency-hell)
3. [Integration Patterns](#integration-patterns)
4. [Technical Implementation](#technical-implementation)
5. [Edge Cases & Troubleshooting](#edge-cases--troubleshooting)
6. [Service Resource Mapping](#service-resource-mapping)
7. [Validation & Testing](#validation--testing)
8. [DeepFace Case Study](#deepface-case-study)
9. [Critical Pitfalls: The Gesture Recognition Lesson](#critical-pitfalls-the-gesture-recognition-lesson)

---

## Architecture Overview

**RumiAI Pipeline Flow:**
```
Video → Unified Frame Manager → ML Services (parallel) → Timeline Entries → Temporal Compute → Unified JSON
                                        ↓
                                 unified_analysis/*.json
                                        ↓
                              insights/*_temporal_windows_updated.json
```

**Key Components:**
- **Unified Frame Manager**: Extract frames ONCE, analyze MANY times
- **ML Services**: Parallel processing with service-specific frame sampling
- **Timeline Entries**: Standardized format for all temporal events
- **Temporal Compute**: Aggregates all data into temporal windows
- **Unified JSON**: Final output in `insights/` folder

---

## Dependency Hell

**⚠️ CRITICAL: Before planning any implementation, ensure service compatibility with Python 3.12.3**

### Pre-Implementation Checklist
1. **Python Version Compatibility**
   - Test service with Python 3.12.3 specifically
   - Check all dependencies for version conflicts
   - Document any known incompatibilities

2. **Common Incompatibilities Found**
   - TensorFlow 2.16 + ThreadPoolExecutor + Python 3.12 = Memory corruption
   - Some older PyTorch versions have asyncio conflicts
   - OpenCV headless vs full version conflicts

3. **Dependency Resolution Strategy**
   ```bash
   # Test in isolated environment first
   python3.12 -m venv test_env
   source test_env/bin/activate
   pip install <new_service>
   python -c "import <new_service>; print('Success')"
   ```

### When Dependency Conflicts Occur
- **Option 1:** Find compatible version
- **Option 2:** Use subprocess isolation (see DeepFace case)
- **Option 3:** Docker containerization
- **Option 4:** Alternative library

---

## Integration Patterns

### 1. Direct ML Data Flow Pattern
**Use when:** Service provides video-level metadata that doesn't need temporal interpretation

**Examples:** `audio_energy`, `deepface_gender`, emotion aggregates, scene counts

**Characteristics:**
- Output goes directly to `ml_data[service_name]`
- No timeline entries created
- Video-level attributes (not frame-by-frame events)
- Used for statistical analysis or metadata enrichment

**Implementation:**
```python
# In video_analyzer.py
analyses['service_name'] = self.service.analyze(video_path)

# In analysis.py to_dict()
result['ml_data'][service_name] = self.ml_results[service_name].data
```

### 2. Timeline-Based Flow Pattern
**Use when:** Service produces temporal events that need frame-accurate positioning

**Examples:** `yolo`, `whisper`, `mediapipe`, `ocr`, `scene_detection`

**Characteristics:**
- Creates timeline entries with start/end timestamps
- Frame-by-frame or segment-based events
- Used for temporal analysis and window computation
- Enables cross-service temporal correlation

**Implementation:**
```python
# In timeline_builder.py
timeline.add_entry(TimelineEntry(
    entry_type='detection',
    start_time=frame_time,
    end_time=frame_time + duration,
    data={'confidence': 0.9, 'class': 'person'}
))
```

### Decision Framework

#### Direct ML Data Flow
✅ **Use when:**
- Video-level metadata (gender, scene count, audio stats)
- No temporal interpretation needed
- Statistical/aggregate data
- Cross-service dependencies (e.g., gender → pitch normalization)

❌ **Don't use when:**
- Frame-accurate timing required
- Temporal correlation with other events needed
- Interactive/real-time analysis required

#### Timeline-Based Flow
✅ **Use when:**
- Frame-by-frame detection/analysis
- Temporal events with start/end times
- Cross-service correlation needed
- Window-based analysis required

❌ **Don't use when:**
- Video-level attributes only
- No temporal component
- Pure metadata/statistics

---

## Technical Implementation

### Frame Sampling Strategy

#### Decision Rules
- **All Frames**: Use for critical detections where missing frames impacts accuracy
  - Gestures, facial expressions, eye contact tracking
  - Performance Rule: If total processing < 1 second for 60s video, process all frames
- **Adaptive Sampling**: Use where temporal consistency is less critical
  - Text detection (OCR), object detection
  - Sample every Nth frame or use keyframe detection
- **Batch Processing**: Process 10-20 frames per batch for memory/threading efficiency

#### Frame Extraction Pattern
```python
# In unified_frame_manager.py
def get_frames_for_service(frames, service_name):
    if service_name in ['mediapipe', 'gesture']:
        return frames  # All frames for gesture accuracy
    elif service_name in ['ocr', 'yolo']:
        return frames[::3]  # Every 3rd frame
    return frames
```

### Service Architecture Requirements

#### 1. Core Service Class
```python
class NewMLService:
    def __init__(self, config: ServiceConfig = None):
        self.config = config or ServiceConfig.from_env()
        # Thread-local storage for model loading
        self._thread_local = threading.local()

    async def analyze(self, video_path: str) -> Dict[str, Any]:
        # Must return schema-compliant data
        pass
```

#### 2. Configuration Management
```python
@dataclass
class ServiceConfig:
    timeout: int = 30
    use_gpu: bool = False
    thread_workers: int = 2

    @classmethod
    def from_env(cls):
        return cls(
            timeout=int(os.getenv('SERVICE_TIMEOUT', '30')),
            use_gpu=os.getenv('SERVICE_USE_GPU', 'false').lower() == 'true'
        )
```

#### 3. Integration Points
- **Analysis Registration:** Add to `required_models` in `analysis.py`
- **Service Import:** Import in `video_analyzer.py`
- **Timeline Integration:** Add builder logic if timeline-based
- **Temporal Windows:** Add extraction logic in `temporal_compute.py`
- **Unified JSON Output:** For Direct ML Data services, explicitly extract in `temporal_compute.py`:
  ```python
  # CRITICAL: Direct ML Data services need explicit extraction
  service_data = ml_data.get('service_name', {})
  if service_data:
      calculated_metadata['service_name'] = service_data
  ```
  **Note:** Data in `unified_analysis/*.json` is NOT automatically in `insights/*_temporal_windows_updated.json`

### Technical Architecture Decisions

#### Standard Service Integration
✅ **Use when:**
- Compatible with existing ML framework stack
- No threading/memory conflicts
- Standard Python async/await patterns work

#### Subprocess Integration
✅ **Use when:**
- Memory corruption or threading conflicts
- Incompatible ML framework dependencies
- Need complete process isolation
- Library has known asyncio/threading issues

---

## Edge Cases & Troubleshooting

### Memory Corruption Issues
**Problem:** TensorFlow/PyTorch + ThreadPoolExecutor + certain Python versions cause segfaults

**DeepFace Example:**
- Error: `free(): double free detected in tcache 2`
- Cause: TensorFlow 2.16 + ThreadPoolExecutor + Python 3.12 conflict
- Solution: Subprocess isolation via standalone script

**Detection Symptoms:**
- Segmentation faults during concurrent model loading
- Memory corruption errors in logs
- Inconsistent crashes during parallel processing

**Resolution Pattern:**
1. Create standalone script in `scripts/`
2. Implement subprocess wrapper service
3. Use `asyncio.create_subprocess_exec()` for isolation
4. Maintain same API interface for seamless integration

### False Positive Detection
**Problem:** ML models detecting artifacts as valid targets

**DeepFace Example:**
- Issue: Logos/watermarks detected as faces
- Impact: False "multiple people" classification
- Solution: Size-based filtering (minimum 120x120 pixels)

**Prevention Strategies:**
- Implement confidence thresholds
- Add size/area filtering for detection services
- Use multiple validation criteria
- Provide debug modes for investigation

### Threading & GPU Conflicts
**Common Issues:**
- TensorFlow threading conflicts with asyncio
- GPU memory not released between model loads
- CUDA context issues in multi-threaded environments

**Solutions:**
- Force CPU usage: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`
- Thread-local model loading
- Explicit GPU memory cleanup
- Single-threaded TensorFlow configuration

---

## Service Resource Mapping

### Current Services Status

| Service Name | Purpose | GPU Support | Current Mode | Dependencies | Integration Pattern | Notes |
|-------------|---------|-------------|--------------|--------------|-------------------|-------|
| **YOLO** | Object detection | GPU/CPU | GPU | ultralytics | Timeline | Frame-by-frame detection |
| **DeepFace** | Gender detection | GPU/CPU | CPU (forced) | tensorflow 2.16 | Direct ML Data | Subprocess isolation due to TF conflict |
| **Whisper** | Speech transcription | GPU/CPU | CPU | whisper.cpp | Timeline | Using whisper.cpp for speed |
| **MediaPipe** | Pose/Face/Gesture/Gaze | CPU only | CPU | mediapipe | Timeline | Includes gesture recognition & eye gaze |
| **OCR** | Text detection | GPU/CPU | GPU | easyocr | Timeline | GPU significantly faster |
| **Scene Detection** | Scene changes | CPU | CPU | scenedetect | Timeline | Frame difference analysis |
| **Audio Energy** | Audio analysis | CPU | CPU | librosa | Direct ML Data | RMS energy calculation |
| **FEAT** | Emotion detection | GPU/CPU | GPU | py-feat | Timeline | Face emotion analysis |

### Resource Allocation Strategy

#### GPU Priority (when available)
1. **High Priority:** OCR, YOLO, FEAT (significant speedup)
2. **Medium Priority:** Whisper (if using PyTorch version)
3. **Low Priority:** DeepFace (forced CPU due to conflicts)

#### CPU-Only Services
- MediaPipe (no GPU support)
- Scene Detection (minimal GPU benefit)
- Audio Energy (CPU sufficient)

#### Memory Requirements
- **High:** YOLO (~2GB), OCR (~1.5GB)
- **Medium:** FEAT (~1GB), DeepFace (~800MB)
- **Low:** MediaPipe (~500MB), Audio Energy (~200MB)

---

## Validation & Testing

### 1. Dependency Testing
```python
def test_python_compatibility():
    import sys
    assert sys.version_info[:3] == (3, 12, 3), "Must use Python 3.12.3"

def test_import_compatibility():
    try:
        import new_service
        assert True
    except ImportError as e:
        pytest.fail(f"Service import failed: {e}")
```

### 2. Unit Testing
```python
def test_service_basic_functionality():
    service = NewMLService()
    result = await service.analyze('test_video.mp4')
    assert 'required_field' in result
    assert result['confidence'] > 0
```

### 3. Integration Testing
```python
def test_full_pipeline_integration():
    # Run complete pipeline with test video
    result = run_rumiai_pipeline('test_video_url')

    # Verify service data in unified analysis
    assert 'new_service' in result['ml_data']

    # Verify temporal windows if applicable
    if service_is_temporal:
        assert service_data_in_temporal_windows(result)
```

### 4. Performance Testing
- Memory usage monitoring
- Processing time benchmarks
- Concurrent processing validation
- GPU/CPU resource utilization

### 5. Edge Case Testing
- Empty videos
- Very short/long videos
- Multiple people scenarios
- Low quality/resolution videos
- Videos with artifacts/watermarks

---

## DeepFace Case Study

### Background
DeepFace gender detection service for enabling gender-specific pitch normalization in audio processing.

### Implementation Journey

#### Initial Approach (Failed)
- **Pattern:** Direct ML Data Flow with ThreadPoolExecutor
- **Integration:** Standard service class with async wrapper
- **Problem:** Memory corruption from TensorFlow + ThreadPoolExecutor conflict

#### Dependency Discovery
```bash
# Initial test revealed conflict
Python 3.12.3 + TensorFlow 2.16 + ThreadPoolExecutor = CRASH
└── Error: free(): double free detected in tcache 2
```

#### Root Cause Analysis
```
TensorFlow 2.16 + ThreadPoolExecutor + Python 3.12 = Memory Corruption
├── 14 debugging attempts tried
├── Thread-local storage failed
├── TensorFlow downgrade failed
├── Force CPU failed
├── Single-thread TF config failed
└── Only subprocess isolation worked
```

#### Final Solution (Successful)
- **Pattern:** Direct ML Data Flow via subprocess
- **Implementation:**
  - Standalone script: `scripts/run_deepface_gender.py`
  - Wrapper service: `DeepFaceGenderServiceSimple`
  - Subprocess execution: `asyncio.create_subprocess_exec()`

#### Key Lessons

1. **Subprocess ≠ Band-aid Fix**
   - Sometimes the architecturally correct solution
   - Provides complete isolation from incompatible libraries
   - Maintains clean API interfaces

2. **False Positive Handling**
   - Original: 99.99% female → "multiple_people" (false positive from logo)
   - Solution: Face size filtering (>120x120 pixels)
   - Result: Accurate gender detection

3. **Integration Points**
   - Added to `required_models` in `analysis.py`
   - Gender data appears in both `ml_data` and convenience field
   - Included in temporal windows metadata

4. **Unified JSON Integration Challenge**
   - **Issue:** Gender data was in `unified_analysis/*.json` but NOT in `insights/*_temporal_windows_updated.json`
   - **Root Cause:** Temporal windows file is post-processed output, not the complete unified analysis
   - **Solution:** Modified `temporal_compute.py` to extract gender from `ml_data` and add to metadata:
   ```python
   # In temporal_compute.py
   gender_data = ml_data.get('deepface_gender', {})
   if gender_data:
       calculated_metadata['gender_detection'] = {
           'gender': gender_data.get('gender'),
           'confidence': gender_data.get('confidence', 0.0),
           'method': gender_data.get('method', 'deepface')
       }
   ```
   - **Learning:** Direct ML Data services need explicit extraction in temporal compute for unified JSON visibility

#### Architecture Benefits
```
Production Pipeline → DeepFaceGenderServiceSimple → subprocess → scripts/run_deepface_gender.py
                                                               ↓
                                                        No memory conflicts
                                                        Complete TF isolation
                                                        Easy debugging/testing
```

### Configuration Lessons
- **Thread Safety:** Use subprocess for incompatible libraries
- **Error Handling:** Distinguish critical vs expected failures
- **Output Validation:** Implement schema compliance checking
- **Performance:** Subprocess overhead < memory corruption debugging time

---

## Quick Reference Checklist

### New Service Integration
- [ ] **Test Python 3.12.3 compatibility**
- [ ] **Check for dependency conflicts**
- [ ] Determine integration pattern (Direct vs Timeline)
- [ ] Test for memory/threading conflicts
- [ ] Implement schema-compliant output
- [ ] Add to required_models list
- [ ] Create configuration dataclass
- [ ] Add error handling for critical vs expected failures
- [ ] Implement unit and integration tests
- [ ] Document edge cases and solutions
- [ ] Add performance benchmarks
- [ ] Update temporal windows if needed
- [ ] **Update Service Resource Mapping table**
- [ ] **Document GPU/CPU requirements**
- [ ] **For Direct ML Data: Add extraction in temporal_compute.py for unified JSON**
- [ ] **Verify data appears in insights/*_temporal_windows_updated.json**

---

## Future Considerations

### Monitoring & Observability
- Health check endpoints
- Performance metrics collection
- Error rate tracking
- Resource usage monitoring

### Service Templates
```bash
# Generate new service boilerplate
python scripts/generate_service.py --name MyService --type direct_ml
```

### Migration Patterns
- Standard → Subprocess migration guide
- Backward compatibility strategies
- Rollback procedures
- A/B testing new services

---

## Critical Pitfalls: The Gesture Recognition Lesson

### The Silent Integration Failure

**What Happened:** Gesture recognition service worked perfectly (detected 124 gestures) but the entire pipeline appeared to succeed while actually failing silently.

### Root Cause: Timeline Extraction Conflict

**The Bug:**
```python
# ❌ WRONG: Multiple extraction methods for same service
# Method 1 (old code)
timelines['gestureTimeline'][timestamp_key] = gesture  # Direct assignment

# Method 2 (new code)
timelines['gestureTimeline'][timestamp_key] = {'gestures': [], 'confidence': 0}  # Structured

# Result: Old overwrote new → KeyError: 'gestures' → Temporal compute failed silently
```

**The Fix:**
```python
# ✅ CORRECT: Single extraction method
# Only use ONE approach consistently
timelines['gestureTimeline'][timestamp_key] = {'gestures': [...], 'confidence': ...}
```

### Critical Lessons

#### 1. **Silent Failures Are Real**
- Processing can appear successful while output is broken
- Always verify data appears in `insights/*_temporal_windows_updated.json`
- Don't trust intermediate success messages

#### 2. **Data Structure Conflicts Cascade**
- One extraction conflict in temporal_compute.py breaks everything
- New timeline structure vs old extraction = silent failure
- When changing data structures, update ALL consuming code

#### 3. **Test the Final Output**
```bash
# ✅ CORRECT: Verify final unified JSON
cat insights/[video_id]_temporal_windows_updated.json | jq '.metadata.your_service'

# Test temporal compute directly
python3 test_temporal_compute_v2.py [video_id]

# ❌ WRONG: Only testing service in isolation
python3 test_your_service.py  # Works perfectly but integration fails!
```

#### 4. **Common Integration Failures**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Service works, no data in unified JSON | Missing extraction in temporal_compute.py | Add extraction logic |
| KeyError in temporal compute | Conflicting data structures | Use single extraction pattern |
| Processing succeeds, output missing | Silent failure in temporal compute | Check final JSON output |
| Partial data in output | Multiple extraction methods | Remove duplicate extractions |

### Prevention Checklist

- [ ] **Single Extraction Method**: Only ONE way to extract your service data
- [ ] **Final Output Verification**: Check `insights/*_temporal_windows_updated.json`
- [ ] **End-to-End Testing**: Test from video input to final JSON output
- [ ] **Data Structure Consistency**: New structures must update ALL consumers
- [ ] **No Silent Failures**: Add explicit validation of final output

### The Key Takeaway

> **"A service that works perfectly in isolation can break the entire pipeline through integration conflicts. Always validate the final unified JSON output, not just your service."**

The gesture recognition bug taught us that technical success (124 gestures detected!) means nothing if integration fails. One small conflict in data extraction cascaded to break temporal compute while appearing to succeed.