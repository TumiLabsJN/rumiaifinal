# Services Performance Analysis

## Comprehensive Performance Metrics

Last Updated: 2025-09-19

### Complete Service Performance Matrix

| Service | Purpose | Status | Currently Using | GPU Compatible | Output Type | Self-Contained | Thread Count | Optimal Threads | Peak Memory (MB) | Memory 18s | Memory 73s | Memory 120s | Time 18s | Time 73s | Time 120s |
|---------|---------|--------|-----------------|----------------|-------------|----------------|--------------|-----------------|------------------|------------|------------|-------------|----------|----------|-----------|
| **YOLO** | Object detection and tracking | ✅ Active | CPU | ⚠️ Optional (CUDA) | Timeline | ✅ Yes | 1-2 | 2 (1.05x) | 959.7 | 880.6 | 959.7 | N/A | 2.0s | 3.0s | 4.0s |
| **MediaPipe** | Human pose, face, hands, gaze detection | ✅ Active | CPU | ❌ No (CPU only) | Timeline | ✅ Yes | 1 | N/A (Fixed) | 399.2 | 388.0 | 399.2 | N/A | 3.0s | 7.0s | 7.0s |
| **OCR** | Text overlay detection and recognition | ✅ Active | CPU | ⚠️ Optional (CUDA) | Timeline | ✅ Yes | 1 | 2 (1.26x) | 90.6 | 90.4 | 90.6 | N/A | 22.6s | 6.5s | 17.1s |
| **Scene Detection** | Scene boundary and cut detection | ✅ Active | CPU | ❌ No (CPU only) | ML Data | ✅ Yes | 1-2 | N/A (Single) | 41.8 | 40.2 | 41.8 | N/A | 0.5s | 1.5s | 3.0s |
| **Whisper** | Speech transcription with timestamps | ✅ Active | CPU (whisper.cpp) | ⚠️ Alternative | Timeline | ✅ Yes | 1 | 4 (1.15x) | 0.0 | 0.0 | 0.0 | N/A | 2.0s | 12.1s | 26.1s |
| **Audio Energy** | RMS energy and pitch dynamics analysis | ✅ Active | CPU | ❌ No (CPU only) | ML Data | ✅ Yes | 1 | N/A (Single) | 102.1 | 102.1 | 101.1 | N/A | 2.0s | 3.5s | 5.0s |
| **Emotion Detection** | Emotion detection via Action Units | ✅ Production | GPU/CPU | ✅ Yes (CUDA) | ML Data + Timeline | ✅ Yes | 2 | N/A (Fixed) | 1143.1 | 1027.0 | 1143.1 | N/A | 9.5s | 37.1s | 74.0s |
| **DeepFace Gender** | Gender detection for pitch normalization | ✅ Production | CPU only | ✅ Yes (TensorFlow) | ML Data only | ✅ Yes | 1 | N/A (Fixed) | 9.1 | 3.1 | 0.0 | N/A | 5.0s | 9.1s | 6.0s |

## Performance Summary

### Thread Optimization
- **Configurable Services**: Only 3 services (Whisper, YOLO, OCR) support thread configuration via environment variables
- **Fixed Threading**: 5 services use internal/fixed threading that cannot be optimized
- **Total Thread Usage**: 9 threads created across all services in typical run

### Memory Usage Patterns
- **Highest Memory**: Emotion Detection (1143 MB in 73s video) - accounts for ~40% of total memory
- **Lowest Memory**: DeepFace Gender (3.1 MB) and Whisper (0 MB reported due to measurement timing)
- **Total Pipeline Memory**: ~2531 MB for 18s video, ~2736 MB for 73s video

### Total Pipeline Processing Time

| Video Duration | Sequential Mode | Parallel Mode | Processing/Duration Ratio |
|----------------|-----------------|---------------|--------------------------|
| **18s video** | 48.4s | 83.4s | 2.7x (Sequential) |
| **73s video** | 81.5s | 113.7s | 1.1x (Sequential) |
| **120s video** | 177.5s | 167.4s | 1.5x (Sequential) |

*Note: Times from Phase 2 instrumentation tests. Sequential mode generally faster due to better resource utilization.*

### Processing Time Analysis
- **Bottleneck Service (18s video)**: OCR at 22.6s (48% of pipeline time)
- **Bottleneck Service (73s video)**: Emotion Detection at 37.1s (46% of pipeline time)
- **Bottleneck Service (120s video)**: Emotion Detection at 74.0s
- **Fastest Services**: Scene Detection (0.5-3s) consistently quick

### Optimization Recommendations

#### Thread Configuration
```bash
# Add to .bashrc or export before running
export WHISPER_THREADS=4    # 1.15x speedup
export CV2_THREADS=2         # 1.05x speedup (YOLO)
export OMP_NUM_THREADS=2     # 1.26x speedup (OCR)
```

#### Memory Optimization Priorities
1. **Emotion Detection**: Investigate memory reduction strategies (1027 MB)
2. **YOLO**: Consider batch processing to reduce memory (880 MB)
3. **MediaPipe**: Fixed memory usage, limited optimization potential (388 MB)

#### Performance Bottlenecks
1. **Short Videos (< 30s)**: OCR is the primary bottleneck
2. **Long Videos (> 60s)**: Emotion Detection becomes the bottleneck
3. **Parallel Processing**: Benefits all video lengths after Phase 2 instrumentation

## Service Categories by Resource Usage

### High Resource Services (⚠️ Optimize First)
- **Emotion Detection**: 1027 MB memory, 2 threads, 9.5-74s processing
- **OCR**: Variable performance (22.6s for 18s video), thread-optimizable

### Medium Resource Services (🟡 Monitor)
- **YOLO**: 880 MB memory, thread-optimizable, consistent performance
- **MediaPipe**: 388 MB memory, fixed threading, scales linearly
- **Whisper**: Low memory but high CPU, benefits from thread optimization

### Low Resource Services (✅ Efficient)
- **DeepFace Gender**: 3.1 MB memory, fast processing
- **Scene Detection**: 40.2 MB memory, <3s processing
- **Audio Energy**: 102 MB memory, linear scaling

## Notes
- Memory measurements from Phase 2 instrumentation with enhanced monitoring
- Processing times from sequential mode execution
- Thread optimization only applies to Whisper, YOLO, and OCR
- N/A values indicate data not yet collected for those video durations
- Peak memory represents maximum memory delta during service execution