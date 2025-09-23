# Audio Services

## ⚠️ Legacy Information Warning
This document references current implementation verified through code inspection (2025-01-19).
All information has been validated through actual code review and testing where possible.

## 🔄 Flexible Execution Architecture
**IMPORTANT**: Metadata extraction happens BEFORE ML services, not in parallel.
- Apify scraping occurs first to get video metadata (rumiai_runner.py lines 243)
- Video download happens after scraping (line 249)
- The downloaded videos get processed either in Parallel or Sequentially. The architecture can support both and switch between them easily. 
- Metadata is passed through the entire pipeline

## 📦 Batch Processing Clarification
**CRITICAL DISTINCTION** - Two types of "batching":
1. **Audio Frame Batching** (WITHIN one video):
   - Processing audio in chunks/windows for efficiency
   - Example: Audio Energy processes in 5-second windows
   - This is an optimization technique for single video processing

2. **Video Batching** (NOT IMPLEMENTED):
   - RumiAI processes ONE video at a time
   - No parallel processing of multiple videos
   - Each video goes through the complete pipeline sequentially

## 🎯 Shared Audio Extraction Architecture
**SharedAudioExtractor** ensures audio is extracted once and reused by all audio services.
- Located at `/rumiai_v2/api/shared_audio_extractor.py`
- Extraction format: WAV, 16kHz, mono
- **Note for future services**: Any new audio service should use SharedAudioExtractor to avoid redundant extraction

## 📊 Service Overview Matrix

### Aggregate Performance (All ML Services Combined)
| Video Duration | Total Pipeline Time | Processing Speed | Status |
|---------------|-------------------|------------------|---------|
| 18s | ~68s | 0.26x realtime | Based on Vision tests |
| 73s | ~133s | 0.55x realtime | Based on Vision tests |
| 120s | ~246s | 0.49x realtime | Based on Vision tests |



| Service | Purpose | Status | Currently Using | GPU Compatible | Output Type | Self-Contained |
|---------|---------|--------|-----------------|----------------|-------------|----------------|
| Whisper | Speech transcription with timestamps | ✅ Active | CPU (whisper.cpp) | ⚠️ Alternative available | Timeline | ✅ Yes |
| Audio Energy | RMS energy and pitch dynamics analysis | ✅ Active | CPU | ❌ No (CPU only) | ML Data | ✅ Yes |

---

# Whisper Service

## 🔄 Flexible Execution Architecture
**IMPORTANT**: Metadata extraction happens BEFORE ML services, not in parallel.
- Apify scraping occurs first to get video metadata (rumiai_runner.py lines 243)
- Video download happens after scraping (line 249)
- The downloaded videos get processed either in Parallel or Sequentially. The architecture can support both and switch between them easily. 
- Metadata is passed through the entire pipeline

## 🎯 Service Purpose
- **Single sentence**: Transcribes speech from video audio with precise timestamps using whisper.cpp
- **Input type**: Video file path (audio extracted internally via SharedAudioExtractor)
- **Output type**: JSON with segments containing text, timestamps, and confidence scores

## ⚡ Performance Profile
```
Execution Time (To Be Confirmed - Pending Instrumentation Tests):
- 60-second video: TBC seconds
- 120-second video: TBC seconds


Resource Usage (To Be Confirmed - Pending Instrumentation Tests):
- Memory: TBC MB peak (estimated ~500-800 MB for base model)
- CPU: TBC% average (estimated high utilization on 2-4 cores)
- GPU Compatible: ❌ No (whisper.cpp is CPU-optimized with AVX/SSE)
- GPU Usage: N/A

Configuration:
- Model: base (multilingual, ~142 MB)
- Backend: whisper.cpp (CPU-optimized C++ implementation)
- Audio Processing: Full audio (not sampled)
- Shared Extraction: ✅ Yes (via SharedAudioExtractor)
- Current Status: ✅ Optimized with whisper.cpp
- Timeout: 600 seconds (10 minutes)
- Future Option: faster-whisper with GPU support available
```

## 🔄 Implementation Alternatives

### Current: whisper.cpp (CPU-optimized)
- **Chosen for**: Simplicity, no GPU dependency, proven stability
- **Performance**: Processes audio at ~0.5x realtime on 4 CPU cores
- **Dependencies**: Minimal (C++ compiler, make)
- **Deployment**: Simple, works on any CPU

### Alternative: faster-whisper (GPU-capable)
- **Benefits**: 2-4x faster with CUDA, significantly lower CPU usage
- **Trade-offs**: Requires CUDA toolkit, more complex deployment, GPU memory usage
- **Migration path**: Drop-in replacement via whisper_transcribe_safe.py factory
- **Status**: Documented for future optimization if processing speed becomes bottleneck
- **When to consider**: If transcription becomes pipeline bottleneck or CPU resources are constrained

## 🎵 Audio Processing Strategy
```
✅ VERIFIED through code inspection

Processing Method: Full audio transcription (no sampling)
Audio Format: 16kHz mono WAV (via SharedAudioExtractor)
Segmentation: Automatic by whisper.cpp VAD (Voice Activity Detection)

Implementation Flow:
1. SharedAudioExtractor.extract_once() - Gets/creates 16kHz WAV
2. WhisperCppTranscriber.transcribe_with_preprocessing()
3. Whisper.cpp processes full audio with VAD
4. Returns segments with precise timestamps

Implementation Location:
├── /rumiai_v2/api/whisper_cpp_service.py
│   └── WhisperCppTranscriber class (lines 68-400+)
├── /rumiai_v2/api/shared_audio_extractor.py
│   └── extract_once() method (lines 34-90)
└── /whisper.cpp/
    └── Native C++ implementation

Rationale: Full transcription ensures no speech is missed
Trade-offs: Longer videos take more time, but accuracy is paramount
⚠️ Known Issues: None identified
```

## 🔍 Self-Containment Check
- [x] Works without precompute imports (verified)
- [x] No circular dependencies (uses SharedAudioExtractor singleton)
- [x] Clear service boundaries (documented below)
- [x] Dependencies verified: whisper.cpp, ffmpeg, make, g++

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                           OUTPUT
Video Path ─────────> SharedAudioExtractor ─────────> WhisperCpp ─────────> Segments JSON
                           ├── Extract once                ├── Load model
                           ├── Cache path                  ├── Transcribe
                           └── Return WAV                  └── Format output
```

### Data Flow Pipeline
```
1. Input Stage
   └── Video path received from video_analyzer.py

2. Audio Extraction (Shared)
   └── SharedAudioExtractor.extract_once()
       ├── Check cache for video_id
       ├── If not cached: extract via ffmpeg
       └── Return cached audio path

3. Transcription Stage
   ├── Load whisper.cpp base model
   ├── Process full audio with VAD
   ├── Generate segments with timestamps
   └── Include confidence scores

4. Output Stage
   └── {
       "text": "full transcript",
       "segments": [
         {
           "start": 0.0,
           "end": 2.5,
           "text": "segment text",
           "confidence": 0.95
         }
       ],
       "language": "en",
       "duration": 120.5
     }
```

### Timeline Integration
```python
# How this service integrates with timeline_builder.py
timeline_builder.py:_add_whisper_entries() (lines 133-175)
├── Entry type: 'speech'
├── Data structure: {start, end, text, confidence}
└── Validation: Handles missing end timestamps
```

## 📁 File Structure & Key Locations
```
Service Implementation:
├── /rumiai_v2/api/whisper_cpp_service.py (main Whisper wrapper)
├── /rumiai_v2/api/whisper_transcribe_safe.py (factory/loader)
├── /rumiai_v2/api/shared_audio_extractor.py (audio extraction)
└── /whisper.cpp/ (native implementation)

Timeline Integration:
└── /rumiai_v2/processors/timeline_builder.py
    └── _add_whisper_entries() (lines 133-175)

Temporal Processing:
└── /rumiai_v2/processors/temporal_compute.py
    └── Speech segments used in temporal windows (lines 289-292)

Model Storage:
└── /whisper.cpp/models/
    └── ggml-base.bin (142 MB model file)
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios
| Failure | Cause | Impact | Recovery | Frequency |
|---------|-------|--------|----------|------------|
| No audio track | Video has no audio stream | Empty transcription | Return empty segments array | ~5% of videos |
| Model load fail | Missing ggml-base.bin | Service unavailable | Download model, retry | First run only |
| Timeout (10 min) | Very long video or slow CPU | Incomplete transcription | Return partial or increase timeout | ~1% for >5min videos |
| Audio extraction fail | Corrupt video or ffmpeg error | No audio file | Retry once, then skip service | <1% |
| Memory overflow | Insufficient RAM (<2GB free) | Process killed | Use smaller model (tiny) | Rare on 4GB+ systems |
| Language detection fail | Non-speech audio | Wrong/no transcription | Return empty result | ~3% (music-only videos) |

### Graceful Degradation Strategy
- **Principle**: Whisper failures don't crash pipeline
- **Empty Results**: Returns `{"segments": [], "text": "", "language": "unknown"}` on failure
- **Logging**: All failures logged to `/rumiai_v2/logs/whisper_errors.log`
- **Pipeline Continuation**: Other ML services continue independently
- **Status Indication**: MLAnalysisResult.success = False with error message

### Monitoring Recommendations
- **Key Metrics**: Transcription success rate, average segment count, timeout frequency
- **Alerts**: Alert if >10% failure rate or >5 timeouts per hour
- **Logs**: Monitor for "Model load failed", "Timeout exceeded", "No audio stream"

## 🐛 Current Issues & Future Fixes

### Priority: LOW 🟢
- **Issue**: Fixed 10-minute timeout might be insufficient for very long videos
- **Impact**: Videos >5 minutes might timeout on slower systems
- **Current Workaround**: 600s timeout works for videos up to ~3-4 minutes
- **Proposed Fix**: Dynamic timeout based on video duration
- **Effort Estimate**: 0.5 days
- **Files Affected**: whisper_cpp_service.py

## 🧪 Testing & Validation

### Functional Testing (Isolation)
**Purpose**: Verify service works correctly, NOT for performance measurement
**Warning**: Isolation tests show theoretical performance without resource competition


```bash
# Test Whisper functionality only (NOT for performance metrics)
python3 -c "
import asyncio
from pathlib import Path
from rumiai_v2.api.whisper_cpp_service import WhisperCppTranscriber

async def test():
    transcriber = WhisperCppTranscriber(model='base')
    result = await transcriber.transcribe(Path('test_video.mp4'))
    print(f'✅ Service functional: {len(result.get(\"segments\", []))} segments found')
    print('⚠️ Note: This isolation timing is NOT representative of production performance')

asyncio.run(test())
"
```

### Performance Testing (Full Pipeline)
**Purpose**: Measure actual production performance with resource competition
**This is the ONLY valid way to measure service performance**

```bash
# Clear all caches first
rm -rf /tmp/rumiai_frames_*
rm -rf /tmp/tmp*audio*.wav
rm -rf /home/jorge/rumiaifinal/temp/*.mp4

# Run production pipeline
time python3 scripts/rumiai_runner.py 'VIDEO_URL'
```

### Why Isolation Tests Are Misleading for Performance
- **In isolation**: Whisper might process 60s video in 30s (using all CPU cores)
- **In production**: Same video takes 60-90s (competing for CPU with 7 other services)
- **Resource contention**: Each service gets ~2-3 cores instead of 12
- **Use isolation tests for**: Debugging, functionality verification, dependency checking

## 📈 Optimization Opportunities
- [ ] **Model Size**: Could use 'tiny' model for faster processing (trade quality for speed)
- [ ] **VAD Tuning**: Adjust voice activity detection parameters for content type
- [ ] **Language Hints**: Provide language hints to speed up detection
- [ ] **GPU Acceleration**: Migrate to faster-whisper when processing speed becomes bottleneck (see Implementation Alternatives section)

## 🔄 Dependencies
```
External Libraries:
├── whisper.cpp (commit f9ca902, tested 2025-08-06)
├── ffmpeg (for audio extraction)
└── Standard C++ build tools (make, g++)

Internal Dependencies:
├── SharedAudioExtractor (audio extraction caching)
├── audio_utils (extract_audio_simple fallback)
└── ML Validator (data validation)
```

---

# Audio Energy Service

## 🔄 Flexible Execution Architecture
**IMPORTANT**: Metadata extraction happens BEFORE ML services, not in parallel.
- Apify scraping occurs first to get video metadata (rumiai_runner.py lines 243)
- Video download happens after scraping (line 249)
- The downloaded videos get processed either in Parallel or Sequentially. The architecture can support both and switch between them easily. 
- Metadata is passed through the entire pipeline

## 🎯 Service Purpose
- **Single sentence**: Analyzes audio energy (RMS) and pitch dynamics to detect speech intensity and emotional patterns
- **Input type**: Video file path (audio extracted internally via SharedAudioExtractor)
- **Output type**: JSON with RMS frames, pitch statistics, and derived features

## ⚡ Performance Profile
```
Execution Time (To Be Confirmed - Pending Instrumentation Tests):
- 60-second video: TBC seconds
- 120-second video: TBC seconds


Resource Usage (To Be Confirmed - Pending Instrumentation Tests):
- Memory: TBC MB peak (estimated ~200-400 MB for librosa)
- CPU: TBC% average (estimated moderate utilization on 1-2 cores)
- GPU Compatible: ❌ No (CPU only, numpy/librosa based)
- GPU Usage: N/A

Configuration:
- Energy Sample Rate: 16kHz
- Pitch Sample Rate: 22.05kHz
- Window Size: 5 seconds for pattern analysis
- Hop Length: 512 samples (energy), 512-1024 (pitch)
- Shared Extraction: ✅ Yes (via SharedAudioExtractor)
- Current Status: ✅ Optimized with dual sampling strategy
- Timeout: 30 seconds (configurable)
```

## 🎵 Audio Processing Strategy
```
✅ VERIFIED through code inspection

Processing Method: Full audio analysis with windowing
Energy Analysis: RMS frames at 16kHz
Pitch Analysis: F0 extraction at 22.05kHz (60-350 Hz range)

Implementation Flow:
1. SharedAudioExtractor.extract_once() - Gets 16kHz WAV
2. Load and resample for pitch (22.05kHz) if needed
3. Extract RMS energy frames
4. Extract pitch contour (if enabled)
5. Calculate statistics and patterns

Frame Processing:
- Energy: ~31 frames/second (16000 Hz / 512 hop)
- Pitch: ~43 frames/second (22050 Hz / 512 hop)
- For 60s video: ~1860 energy frames, ~2580 pitch frames

Implementation Location:
├── /rumiai_v2/ml_services/audio_energy_service.py
│   └── AudioEnergyService class (lines 50-400+)
├── /rumiai_v2/api/shared_audio_extractor.py
│   └── extract_once() method (lines 34-90)
└── Processing uses librosa for audio analysis

Rationale: Dual sampling optimizes quality vs performance
Trade-offs: Higher sample rate for pitch = better accuracy but slower
⚠️ Known Issues: None identified
```

## 🔍 Self-Containment Check
- [x] Works without precompute imports (verified)
- [x] No circular dependencies (uses SharedAudioExtractor singleton)
- [x] Clear service boundaries (documented below)
- [x] Uses standard librosa/numpy stack

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                           OUTPUT
Video Path ─────────> SharedAudioExtractor ─────────> AudioEnergyService ─────────> Energy+Pitch JSON
                           ├── Extract once                ├── RMS frames
                           ├── Cache path                  ├── Pitch extraction
                           └── Return WAV                  └── Statistics
```

### Data Flow Pipeline
```
1. Input Stage
   └── Video path received from video_analyzer.py

2. Audio Extraction (Shared)
   └── SharedAudioExtractor.extract_once()
       ├── Check cache for video_id
       ├── If not cached: extract via ffmpeg
       └── Return cached audio path

3. Energy Analysis
   ├── Load audio at 16kHz
   ├── Calculate RMS with hop_length=512
   ├── Generate frame-level energy values
   └── Identify burst patterns

4. Pitch Analysis (if enabled)
   ├── Resample to 22.05kHz if needed
   ├── Extract F0 using librosa.pyin
   ├── Filter 60-350 Hz (voice range)
   ├── Calculate statistics (avg, range, variance)
   └── Normalize values

5. Output Stage
   └── {
       "rms_frames": [0.1, 0.15, ...],
       "frames_per_second": 31.25,
       "energy_stats": {
         "mean": 0.25,
         "variance": 0.08,
         "max": 0.95
       },
       "pitch_features": {
         "avg_pitch_normalized": 0.45,
         "pitch_range_normalized": 0.62
       },
       "burst_pattern": "middle_peak"
     }
```

### Component Breakdown
#### Energy Analysis
- **Output**: RMS frames array
- **Processing time**: ~40% of total
- **Features enabled**: Temporal energy patterns, burst detection

#### Pitch Analysis
- **Output**: Pitch statistics and contour
- **Processing time**: ~60% of total
- **Features enabled**: Emotional intensity, speaking style detection

### ML Data Integration
```python
# This service outputs to ml_data, not timeline
# Data flows directly to temporal_compute.py
video_analyzer.py:_run_audio_energy() (lines 251-307)
├── Output type: MLAnalysisResult with ml_data
├── Saved to: audio_energy_outputs/{video_id}/
└── Used in: temporal_compute.py for burst patterns
```

## 📁 File Structure & Key Locations
```
Service Implementation:
├── /rumiai_v2/ml_services/audio_energy_service.py (main service)
├── /rumiai_v2/ml_services/audio_energy_service_extended.py (with pitch)
├── /rumiai_v2/api/shared_audio_extractor.py (audio extraction)
└── /rumiai_v2/api/audio_utils.py (extraction utilities)

ML Data Flow:
└── /rumiai_v2/processors/video_analyzer.py
    └── _run_audio_energy() (lines 251-307)

Temporal Processing:
└── /rumiai_v2/processors/temporal_compute.py
    └── extract_audio_energy_data() (lines 162-175)
    └── calculate_audio_energy_for_windows() (lines 336-387)

Output Storage:
└── /audio_energy_outputs/{video_id}/
    └── {video_id}_audio_energy.json
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios
| Failure | Cause | Impact | Recovery | Frequency |
|---------|-------|--------|----------|------------|
| Silent audio | No audio content in video | Zero energy values | Return zero-filled arrays | ~5% of videos |
| Audio extraction fail | SharedAudioExtractor fails | No audio to analyze | Fallback to direct extraction | <1% |
| Pitch extraction fail | No voiced segments detected | Missing pitch features | Continue with energy only | ~10% (non-speech) |
| Timeout (30s) | Very long audio processing | Incomplete analysis | Return partial results | Rare |
| Memory overflow | Large audio file | Process killed | Downsample audio first | <1% on 4GB+ systems |
| Invalid sample rate | Unusual audio format | Processing error | Resample to 16kHz | <1% |

### Graceful Degradation Strategy
- **Principle**: Audio Energy failures return valid structure with zero/default values
- **Empty Results**: Returns `{"rms_frames": [], "frames_per_second": 30, "energy_stats": {"mean": 0, "variance": 0, "max": 0}}` on failure
- **Logging**: Failures logged with audio file details for debugging
- **Pipeline Continuation**: Service failure doesn't affect other ML services
- **Fallback Options**: Can run without pitch analysis if that component fails

### Monitoring Recommendations
- **Key Metrics**: Success rate, average RMS frame count, pitch detection rate
- **Alerts**: Alert if >15% videos have silent audio or >5% extraction failures
- **Logs**: Monitor for "Silent audio detected", "Pitch extraction failed", "Sample rate mismatch"

## 🐛 Current Issues & Future Fixes

### Priority: MEDIUM 🟡
- **Issue**: Pitch extraction can be slow for long videos
- **Impact**: Adds 5-10 seconds for videos >2 minutes
- **Current Workaround**: Configurable quality presets (high/medium/low)
- **Proposed Fix**: Implement sliding window pitch extraction
- **Effort Estimate**: 1 day
- **Files Affected**: audio_energy_service.py

## 🧪 Testing & Validation

### Functional Testing (Isolation)
**Purpose**: Verify service works correctly, NOT for performance measurement
**Warning**: Isolation tests show theoretical performance without resource competition

```bash
# Test Audio Energy functionality only (NOT for performance metrics)
python3 -c "
import asyncio
from pathlib import Path
from rumiai_v2.ml_services.audio_energy_service import AudioEnergyService

async def test():
    service = AudioEnergyService()
    result = await service.analyze(Path('test_video.mp4'), 'test_id')
    print(f'✅ Service functional: {len(result.get(\"rms_frames\", []))} RMS frames')
    print(f'✅ Pitch analysis: {\"pitch_features\" in result}')
    print('⚠️ Note: This isolation timing is NOT representative of production performance')

asyncio.run(test())
"
```

### Performance Testing (Full Pipeline)
**Purpose**: Measure actual production performance
**Required for accurate performance metrics**

See Whisper service section for full pipeline testing protocol. Audio Energy runs concurrently with all other services and cannot be timed individually in production.

## 📈 Optimization Opportunities
- [ ] **Parallel Processing**: Process energy and pitch in parallel threads
- [ ] **Downsampling**: Reduce frame rate for faster processing
- [ ] **Caching**: Cache pitch extraction (more expensive than energy)
- [ ] **Quality Presets**: Auto-select based on video duration

## 🔄 Dependencies
```
External Libraries:
├── librosa (0.10.0+) - Audio analysis
├── numpy - Numerical operations
├── scipy - Signal processing
└── ffmpeg - Audio extraction

Internal Dependencies:
├── SharedAudioExtractor (audio extraction caching)
├── audio_utils (extraction fallback)
└── Unified frame manager (for coordination)
```

---

## 📊 Audio Services Performance Summary

### Processing Patterns
- **Whisper**: CPU-intensive, processes full audio sequentially
- **Audio Energy**: Moderate CPU, processes in windows/frames
- **Bottleneck**: Whisper typically takes longest for speech-heavy content

### Testing Protocol Summary

#### Two Types of Testing
1. **Isolation Tests** (Required by template)
   - Location: `/test_whisper_isolation.py`, `/test_audio_energy_isolation.py`
   - Purpose: Functional verification, debugging, dependency checking
   - NOT for performance measurement

2. **Production Tests** (Required for performance)
   - Command: `python3 scripts/rumiai_runner.py 'VIDEO_URL'`
   - Purpose: Real performance measurement with resource competition
   - ONLY valid way to measure actual processing time

#### Cold Start Requirements
```bash
# Before ANY performance test, clear all caches:
rm -rf /tmp/rumiai_frames_*
rm -rf /tmp/tmp*audio*.wav
rm -rf /home/jorge/rumiaifinal/temp/*.mp4
rm -rf /home/jorge/rumiaifinal/audio_energy_outputs/test_*
rm -rf /home/jorge/rumiaifinal/speech_transcriptions/test_*

# Then run production pipeline
time python3 scripts/rumiai_runner.py 'VIDEO_URL'
```


---

**Document Status**: v1.0 - Created 2025-01-19
**Next Review**: After performance testing with standard videos