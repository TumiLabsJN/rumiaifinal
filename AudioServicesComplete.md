# Audio Services Documentation - Python-Only Processing Pipeline

## Overview
The RumiAI audio services provide speech transcription and audio energy analysis as part of the ML pipeline. These services operate with shared audio file architecture for efficiency and are optimized for WSL2 environments.

**Key Components:**
- **Whisper.cpp Service**: Fast CPU-based transcription using C++ implementation
- **Audio Energy Service**: Real-time audio dynamics analysis using librosa
- **Shared Architecture**: Single audio extraction for both services

## Current Status
✅ **Fully Operational** - Both services integrated and performing optimally
- Whisper.cpp: 1.56x realtime speed (3-5 seconds for 3-second audio)
- Audio Energy: Real-time analysis with 5-second windows
- Cost: $0.00 (part of Python-only processing)

---

## 1. Architecture

### 1.1 Shared Audio File Design
**Location**: `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py`

The system extracts audio once and shares it between both services:
```python
# Single audio extraction
temp_audio = await extract_audio_simple(video_path)

# Parallel service execution
whisper_task = transcriber.transcriber.transcribe(temp_audio)
energy_task = energy_service.analyze(temp_audio)
results = await asyncio.gather(whisper_task, energy_task)

# Cleanup after both complete
os.unlink(temp_audio)
```

### 1.2 File Structure
```
rumiai_v2/
├── api/
│   ├── whisper_cpp_service.py      # Whisper.cpp subprocess wrapper
│   ├── whisper_transcribe_safe.py  # Service interface wrapper
│   ├── ml_services_unified.py      # Integrated pipeline
│   └── audio_utils.py              # Audio extraction utilities
└── ml_services/
    ├── __init__.py
    └── audio_energy_service.py     # Energy analysis service
```

---

## 2. Whisper.cpp Implementation

### 2.1 Technical Details
**Location**: `/home/jorge/rumiaifinal/rumiai_v2/api/whisper_cpp_service.py`

**Features:**
- Subprocess-based execution of compiled whisper.cpp binary
- Automatic dependency checking and model validation
- JSON output parsing to match Python Whisper format
- Version pinned to commit f9ca90256bf691642407e589db1a36562c461db7

**Configuration:**
```python
# Optimized for WSL2 with 10 cores
threads = 10  # Hardcoded for WSL2 setup
beam_size_options = 1  # Greedy decoding (bo=1)
beam_size = 1  # No beam search (bs=1)
output_json = True  # JSON format for parsing
```

**Binary Location**: `/home/jorge/rumiaifinal/whisper.cpp/main`
**Model Path**: `/home/jorge/rumiaifinal/whisper.cpp/models/ggml-base.bin`

### 2.2 WSL2 Performance Optimization

#### Problem Solved
- **Before**: 30+ seconds for 3-second audio (WSL2 limited to 4 cores)
- **After**: 3-5 seconds for 3-second audio (WSL2 configured with 10 cores)
- **Speedup**: 6-10x improvement

#### WSL2 Configuration
Create `.wslconfig` in Windows user directory (`C:\Users\[YourUsername]\.wslconfig`):

```ini
[wsl2]
# CPU Configuration
processors=10           # Assign 10 cores out of 16 to WSL2

# Memory Configuration
memory=24GB            # Assign 24GB RAM (adjust based on your total RAM)

# Swap Configuration
swap=8GB               # 8GB swap space for overflow

# Network Configuration
localhostForwarding=true    # Allow localhost forwarding

# Performance Optimizations
pageReporting=false         # Disable page reporting to reduce CPU overhead
guiApplications=false       # Disable GUI app support (saves resources)

# File System
kernelCommandLine = vsyscall=emulate    # Better compatibility
```

#### Apply Configuration
```powershell
# In Windows PowerShell (as Administrator):
wsl --shutdown
# Wait 10 seconds, then restart WSL2
```

#### Verify Setup
```bash
# In WSL2:
nproc  # Should output: 10
lscpu | grep "CPU(s):"  # Should show 10 CPUs
```

### 2.3 Performance Benchmarks

| Audio Length | Before Fix | After Fix | Speedup |
|-------------|------------|-----------|---------|
| 3 seconds   | 30+ sec    | 3-5 sec   | ~6-10x  |
| 30 seconds  | Timeout    | 30-40 sec | N/A     |
| 120 seconds | Timeout    | 2-3 min   | N/A     |

**Current Performance**: 1.56x realtime (excellent)

---

## 3. Audio Energy Service

### 3.1 Implementation Details
**Location**: `/home/jorge/rumiaifinal/rumiai_v2/ml_services/audio_energy_service.py`

**Features:**
- Real audio energy analysis using librosa
- 5-second window energy analysis
- Percentile-based normalization (preserves dynamics)
- Burst pattern detection (front_loaded, back_loaded, middle_peak, steady)
- Climax timestamp detection
- Energy variance calculation

**Configuration:**
```python
sample_rate = 16000  # Standard rate for audio processing
window_seconds = 5   # 5-second windows for energy analysis
```

### 3.2 Output Format
```json
{
  "avgEnergy": 0.65,
  "maxEnergy": 0.95,
  "energyVariance": 0.12,
  "energyPeaks": [
    {"timestamp": 15.0, "energy": 0.95},
    {"timestamp": 45.0, "energy": 0.88}
  ],
  "burstPattern": "middle_peak",
  "climaxTimestamp": 15.0,
  "windowEnergies": [0.45, 0.52, 0.95, 0.88, 0.65, ...]
}
```

---

## 4. Testing & Validation

### 4.1 Test Commands

#### Test Whisper.cpp
```bash
cd /home/jorge/rumiaifinal

# Quick test (3-second audio)
ffmpeg -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 -ac 1 test_3s.wav -y
time ./whisper.cpp/main -m whisper.cpp/models/ggml-base.bin -f test_3s.wav -t 10 -bo 1 -bs 1 -oj

# Python integration test
python3 test_whisper_cpp.py
```

#### Test Audio Energy
```bash
python3 test_audio_energy.py
```

#### Test Full Pipeline
```bash
# With environment variables set for Python-only flow
python3 scripts/rumiai_runner.py "VIDEO_URL"
```

### 4.2 Validation Checklist
- [ ] WSL2 shows 10 available cores (`nproc` returns 10)
- [ ] 3-second audio processes in <10 seconds
- [ ] 120-second audio completes without timeout
- [ ] No thread contention warnings in logs
- [ ] CPU usage stays below 100% (no oversubscription)
- [ ] Both services produce valid JSON output
- [ ] Pipeline integration test passes

---

## 5. Output Files

### 5.1 File Locations
- **Whisper**: `ml_outputs/[video_id]_whisper.json`
- **Energy**: `ml_outputs/[video_id]_audio_energy.json`
- **Transcriptions**: `speech_transcriptions/[video_id]_whisper.json`

### 5.2 Whisper Output Format
```json
{
  "text": "Full transcription text...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.28,
      "text": "Segment text"
    }
  ],
  "language": "en",
  "duration": 66.0,
  "backend": "whisper.cpp",
  "model": "whisper.cpp-base"
}
```

---

## 6. Dependencies & Setup

### 6.1 Required Packages
```bash
# Python packages
pip install aiohttp        # Async HTTP operations
pip install opencv-python  # Video processing
pip install librosa       # Audio energy analysis
pip install numpy==1.26.4 # Compatible version for librosa

# System dependencies
sudo apt-get install ffmpeg        # Audio extraction
sudo apt-get install build-essential  # For compiling whisper.cpp
```

### 6.2 Whisper.cpp Setup
```bash
cd /home/jorge/rumiaifinal
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make -j10  # Compile with 10 threads

# Download base model
cd models
bash ./download-ggml-model.sh base
```

### 6.3 Known Issues
- numpy version conflict: opencv-python wants 2.x, librosa needs 1.26.x
- Solution: Use numpy 1.26.4 (librosa requirement takes precedence)

---

## 7. Troubleshooting

### 7.1 Whisper Performance Issues

#### If WSL2 Still Shows 4 Cores:
1. Verify `.wslconfig` location: `C:\Users\[YourUsername]\.wslconfig`
2. Check for typos in configuration
3. Ensure `wsl --shutdown` ran as Administrator
4. Try `wsl --terminate <distro-name>` then restart
5. Restart Windows if necessary

#### If Performance Still Poor After Core Upgrade:

**Option 1: Use Tiny Model**
```bash
cd /home/jorge/rumiaifinal/whisper.cpp/models
bash ./download-ggml-model.sh tiny
# Update whisper_cpp_service.py to use tiny model
```

**Option 2: Check CPU Throttling**
```bash
# Monitor CPU usage during transcription
htop  # Watch thread usage
watch -n 1 "cat /proc/cpuinfo | grep MHz"  # Check for throttling
```

**Option 3: Process in Chunks**
For very long audio, consider splitting into 30-second chunks.

### 7.2 Audio Energy Issues

#### If librosa Import Fails:
```bash
pip uninstall librosa
pip install --no-cache-dir librosa
```

#### If Energy Values All Zero:
- Check audio file is valid WAV format
- Verify sample rate matches expected (16000 Hz)
- Check audio file has actual content (not silence)

### 7.3 Integration Issues

#### If Services Timeout:
- Increase timeout in ml_services_unified.py
- Check WSL2 CPU allocation
- Consider using tiny model for faster processing

#### If Shared Audio File Missing:
- Verify extract_audio_simple() completes successfully
- Check temp directory permissions
- Ensure ffmpeg is installed and working

---

## 8. Key Architectural Decisions

1. **No Singleton Pattern**: Whisper.cpp instances are lightweight
2. **Hardcoded Threads**: 10 threads optimized for WSL2 setup
3. **No Quantization**: Standard models for multilingual support
4. **Greedy Decoding**: Maximum speed over accuracy (bo=1, bs=1)
5. **Fail-Fast Approach**: No graceful degradation for missing dependencies
6. **Real Energy Only**: No fake/estimated energy values
7. **Shared Audio File**: Single extraction for efficiency

---

## 9. Future Enhancements

1. **Model Options**: Consider tiny model for even faster processing
2. **Chunk Processing**: Implement for very long videos (>5 minutes)
3. **Progress Monitoring**: Add real-time transcription progress
4. **GPU Acceleration**: Consider if CUDA available in future
5. **Dynamic Threading**: Auto-detect optimal thread count
6. **Caching**: Cache transcriptions for repeated videos

---

## 10. Notes for Fresh Claude Instance

If starting fresh, inform Claude:
```
The RumiAI audio services use whisper.cpp (C++ binary via subprocess) for transcription and librosa for energy analysis. WSL2 is configured with 10 cores via .wslconfig.

Key configurations:
- Whisper: 10 threads, greedy decoding (bo=1, bs=1), base model
- Energy: 5-second windows, percentile normalization
- Both services share a single extracted audio file
- Expected performance: 1.56x realtime for transcription

Verify with: nproc (should return 10)
Test with: python3 test_whisper_cpp.py
```

---

## Related Documentation
- `ML_DATA_PROCESSING_PIPELINE.md` - Overall ML pipeline architecture
- `rumiai_v2/api/whisper_cpp_service.py` - Whisper implementation
- `rumiai_v2/ml_services/audio_energy_service.py` - Energy implementation
- `rumiai_v2/api/ml_services_unified.py` - Integration point