# Services Performance Analysis

## Comprehensive Performance Metrics

Last Updated: 2025-10-02

### Complete Service Performance Matrix

| Service | Purpose | Status | Currently Using | GPU Compatible | Output Type | Self-Contained | Thread Count | Optimal Threads | Peak Memory (MB) | Memory 18s | Memory 73s | Memory 120s | Time 18s | Time 73s | Time 120s |
|---------|---------|--------|-----------------|----------------|-------------|----------------|--------------|-----------------|------------------|------------|------------|-------------|----------|----------|-----------|
| **YOLO** | Object detection and tracking | ✅ Active | GPU/CPU | ✅ Auto-GPU (CUDA) | Timeline | ✅ Yes | 15-16 | 16 (Optimal) | 6000.0 | 5891.3 | 5897.3 | N/A | 30.6s | 28.6s | 30.0s |
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
- **Total Thread Usage**: 23-24 threads created across all services in typical run (YOLO now uses 15-16 threads)

### Memory Usage Patterns
- **Highest Memory**: YOLO (5891-6000 MB GPU memory) - accounts for ~60% of total memory with GPU acceleration
- **Second Highest**: Emotion Detection (1143-1185 MB) - accounts for ~12% of total memory
- **Lowest Memory**: DeepFace Gender (0-9 MB) and Whisper (0-0.2 MB)
- **Total Pipeline Memory**: ~9935 MB for 65s video (including GPU memory)

### Total Pipeline Processing Time

| Video Duration | Sequential Mode | Parallel Mode | Processing/Duration Ratio |
|----------------|-----------------|---------------|--------------------------|
| **18s video** | 48.4s | 83.4s | 2.7x (Sequential) |
| **73s video** | 81.5s | 113.7s | 1.1x (Sequential) |
| **120s video** | 177.5s | 167.4s | 1.5x (Sequential) |

*Note: Times from Phase 2 instrumentation tests. Sequential mode generally faster due to better resource utilization.*

### Processing Time Analysis
- **Current Bottleneck Service**: Emotion Detection at 36.9-41.3s (26-30% of pipeline time)
- **Second Bottleneck**: YOLO at 28.6-30.6s (20-22% of pipeline time) with GPU acceleration
- **Third Bottleneck**: MediaPipe at 24.6-25.1s (18% of pipeline time)
- **Fastest Services**: Scene Detection (0.5s) and Audio Energy (0.5s) consistently quick

### Optimization Recommendations

#### Thread Configuration
```bash
# Add to .bashrc or export before running
export WHISPER_THREADS=4    # 1.15x speedup
export CV2_THREADS=16        # Optimal for YOLO GPU processing (15-16 threads)
export OMP_NUM_THREADS=2     # 1.26x speedup (OCR)
```

#### Memory Optimization Priorities
1. **YOLO**: GPU memory management strategies (5891-6000 MB GPU memory)
2. **Emotion Detection**: Investigate memory reduction strategies (1185 MB)
3. **MediaPipe**: Memory efficiency improved with negative deltas (-3627 MB)

#### Performance Bottlenecks
1. **Short Videos (< 30s)**: OCR is the primary bottleneck
2. **Long Videos (> 60s)**: Emotion Detection becomes the bottleneck
3. **Parallel Processing**: Benefits all video lengths after Phase 2 instrumentation

## Service Categories by Resource Usage

### High Resource Services (⚠️ Optimize First)
- **YOLO**: 5891-6000 MB GPU memory, 15-16 threads, 28.6-30.6s processing
- **Emotion Detection**: 1185 MB memory, 2 threads, 36.9-41.3s processing

### Medium Resource Services (🟡 Monitor)
- **MediaPipe**: Memory efficient with cleanup (-3627 MB delta), fixed threading, 24.6-25.1s processing
- **OCR**: 469-487 MB memory, thread-optimizable, 13.0-16.1s processing
- **Whisper**: Low memory (0-0.2 MB) but moderate CPU, benefits from thread optimization, 11.5-15.0s processing

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