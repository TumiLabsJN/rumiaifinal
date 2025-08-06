# Whisper.cpp WSL2 Performance Optimization Test

## Problem Statement
Whisper.cpp transcription is extremely slow in WSL2, taking 30+ seconds for 3 seconds of audio. Investigation revealed WSL2 was only allocated 4 CPU cores despite the host system having 16 cores and 22 threads.

## Root Cause
1. **WSL2 Default Limitation**: By default, WSL2 only uses 50% of available memory and a limited number of CPU cores
2. **Thread Contention**: Code was using 10 threads on only 4 available cores, causing massive contention
3. **Suboptimal Settings**: Using beam search (bo=5, bs=5) instead of greedy decoding

## Solution Implemented

### 1. WSL2 Configuration
Created `.wslconfig` file in Windows user directory (`C:\Users\[YourUsername]\.wslconfig`):

```ini
[wsl2]
# CPU Configuration
processors=10           # Assign 10 cores out of 16 to WSL2

# Memory Configuration
memory=24GB            # Assign 24GB RAM (adjust based on your total RAM)

# Swap Configuration
swap=8GB               # 8GB swap space for overflow

# Network Configuration
localhostForwarding=true    # Allow localhost forwarding between Windows and WSL

# Performance Optimizations
pageReporting=false         # Disable page reporting to reduce CPU overhead
guiApplications=false       # Disable GUI app support if not needed (saves resources)

# File System
kernelCommandLine = vsyscall=emulate    # Better compatibility with some applications
```

### 2. Code Optimizations
Updated `/home/jorge/rumiaifinal/rumiai_v2/api/whisper_cpp_service.py`:
- Thread count: 10 threads (matching WSL2 allocated cores)
- Beam search: Greedy decoding (bo=1, bs=1) for 5-10x speed improvement
- Timeout: 120 seconds for long audio files

## Testing Procedure

### Step 1: Apply WSL2 Configuration
```powershell
# In Windows PowerShell (as Administrator):
wsl --shutdown
# Wait 10 seconds, then restart WSL2
```

### Step 2: Verify Core Allocation
```bash
# In WSL2:
nproc  # Should output: 10
lscpu | grep "CPU(s):"  # Should show 10 CPUs
```

### Step 3: Performance Benchmarks

#### Test 1: Short Audio (3 seconds)
```bash
cd /home/jorge/rumiaifinal

# Create test audio
ffmpeg -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 -ac 1 test_3s.wav -y

# Benchmark
time ./whisper.cpp/main -m whisper.cpp/models/ggml-base.bin -f test_3s.wav -t 10 -bo 1 -bs 1 -oj

# Expected: ~3-5 seconds (vs 30+ seconds before)
```

#### Test 2: Medium Audio (30 seconds)
```bash
# Create test audio
ffmpeg -f lavfi -i "sine=frequency=440:duration=30" -ar 16000 -ac 1 test_30s.wav -y

# Benchmark
time ./whisper.cpp/main -m whisper.cpp/models/ggml-base.bin -f test_30s.wav -t 10 -bo 1 -bs 1 -oj

# Expected: ~30-40 seconds
```

#### Test 3: Full Length (120 seconds)
```bash
# Create test audio
ffmpeg -f lavfi -i "sine=frequency=440:duration=120" -ar 16000 -ac 1 test_120s.wav -y

# Benchmark with timeout
time timeout 180 ./whisper.cpp/main -m whisper.cpp/models/ggml-base.bin -f test_120s.wav -t 10 -bo 1 -bs 1 -oj

# Expected: ~2-3 minutes (vs timeout before)
```

#### Test 4: Python Integration
```bash
# Test the full pipeline
python3 test_whisper_cpp.py

# Expected: Should complete successfully within timeout
```

## Performance Expectations

| Audio Length | Before Fix | After Fix | Speedup |
|-------------|------------|-----------|---------|
| 3 seconds   | 30+ sec    | 3-5 sec   | ~6-10x  |
| 30 seconds  | Timeout    | 30-40 sec | N/A     |
| 120 seconds | Timeout    | 2-3 min   | N/A     |

## Troubleshooting

### If WSL2 still shows 4 cores:
1. Verify `.wslconfig` is in the correct location: `C:\Users\[YourUsername]\.wslconfig`
2. Check for typos in the configuration
3. Ensure you ran `wsl --shutdown` as Administrator
4. Try `wsl --terminate <distro-name>` then restart
5. Restart Windows if necessary

### If performance is still poor after core upgrade:

#### Option 1: Use Tiny Model
```bash
cd /home/jorge/rumiaifinal/whisper.cpp/models
bash ./download-ggml-model.sh tiny

# Update whisper_cpp_service.py to use tiny model:
# Change: model: str = "base"
# To: model: str = "tiny"
```

#### Option 2: Further Reduce Beam Size
Already at minimum (bo=1, bs=1 for greedy decoding)

#### Option 3: Check for CPU throttling
```bash
# Monitor CPU usage during transcription
top -H  # Watch thread usage
htop    # Better visualization

# Check for thermal throttling
watch -n 1 "cat /proc/cpuinfo | grep MHz"
```

#### Option 4: Process in chunks
For very long audio, consider splitting into 30-second chunks and processing in parallel.

## Validation Checklist

- [ ] WSL2 shows 10 available cores (`nproc` returns 10)
- [ ] 3-second audio processes in <10 seconds
- [ ] 120-second audio completes without timeout
- [ ] No thread contention warnings in logs
- [ ] CPU usage stays below 100% (no oversubscription)
- [ ] Python integration test passes

## Notes for Fresh Claude Instance

If starting fresh, tell Claude:

```
I'm running RumiAI on WSL2 with Ubuntu. The host system has 16 cores and 22 threads, but WSL2 is configured to use 10 cores via .wslconfig. 

The Whisper.cpp integration in /home/jorge/rumiaifinal/rumiai_v2/api/whisper_cpp_service.py is configured to use:
- 10 threads (matching WSL2 cores)
- Greedy decoding (bo=1, bs=1) for speed
- Base model by default

Please verify that `nproc` returns 10 before testing. If it returns 4, the WSL2 configuration hasn't been applied and needs `wsl --shutdown` from Windows PowerShell.

The expected performance for base model with 10 cores:
- 3-second audio: 3-5 seconds
- 120-second audio: 2-3 minutes

If performance is significantly worse, check for thread contention or consider using the tiny model.
```

## Related Files
- `/home/jorge/rumiaifinal/rumiai_v2/api/whisper_cpp_service.py` - Main service implementation
- `/home/jorge/rumiaifinal/rumiai_v2/api/whisper_transcribe_safe.py` - Wrapper service
- `/home/jorge/rumiaifinal/test_whisper_cpp.py` - Integration test
- `C:\Users\[YourUsername]\.wslconfig` - WSL2 configuration (Windows side)