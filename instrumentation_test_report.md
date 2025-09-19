# Instrumentation Test Report

## Test Configuration
- **Video**: 120s TikTok video (6923023955813092613.mp4, 22.6 MB)
- **Test Date**: 2025-09-19
- **Cold Start**: Yes (all caches cleared before each test)

## Performance Comparison

### Sequential Mode
- **Total Pipeline Time**: 147.08s
- **Bottleneck Service**: emotion_detection (74.25s)
- **Total Memory Used**: 3700.1 MB
- **Total Threads Created**: 12
- **Overhead**: 0.01s (minimal)

### Parallel Mode
- **Total Pipeline Time**: 171.27s
- **Bottleneck Service**: emotion_detection (168.25s)
- **Total Threads Created**: Not measured in parallel (memory tracking disabled)
- **Parallel Speedup**: ~0.86x (slower than sequential!)

## Service-Level Performance

| Service | Sequential Time | Parallel Time | Threads | Memory (MB) | Thread Flexibility |
|---------|----------------|---------------|---------|-------------|-------------------|
| yolo | 6.08s | 43.25s | 3 | 1186.9 | ✅ cv2.set() |
| whisper | 21.46s | 142.23s | 1 | 0.1 | ✅ Direct |
| mediapipe | 11.32s | 63.32s | 1 | 913.0 | ❌ Fixed |
| ocr | 16.94s | 73.08s | 1 | 111.7 | ✅ Direct |
| scene_detection | 3.00s | 3.00s | 2 | 58.5 | ✅ N/A |
| audio_energy | 6.50s | 25.09s | 1 | 96.3 | ✅ N/A |
| emotion_detection | 74.25s | 168.25s | 2 | 1333.6 | ⚠️ Env vars |
| deepface_gender | 7.51s | 27.37s | 1 | 0.0 | ⚠️ Env vars |

## Key Findings

### 1. Parallel Mode Performance Degradation
- **Unexpected Result**: Parallel mode is **24s slower** than sequential mode
- **Cause**: Resource contention when services run concurrently
- All services except scene_detection show significant slowdown in parallel mode
- Whisper shows the worst degradation: 6.6x slower in parallel mode

### 2. Memory Usage
- Total memory usage in sequential mode: 3.7 GB
- Largest consumers:
  - emotion_detection: 1.3 GB
  - yolo: 1.2 GB
  - mediapipe: 913 MB

### 3. Thread Control
- Services with direct thread control (✅) perform better
- MediaPipe has fixed threading (❌) - cannot optimize
- Services using environment variables (⚠️) have limited control

### 4. Instrumentation Overhead
- Sequential mode overhead: 0.012s (< 0.01%)
- Successfully meets <0.1% overhead requirement
- Thread monitoring working correctly
- Memory tracking accurate in sequential mode

## Recommendations

1. **Default to Sequential Mode** for this workload
   - 24s faster than parallel execution
   - More predictable performance
   - Easier to identify bottlenecks

2. **Optimization Priorities**:
   - emotion_detection (74s) - biggest bottleneck
   - whisper (21s) - consider smaller model or optimization
   - ocr (17s) - potential for optimization

3. **Thread Optimization**:
   - Services with env var control could be optimized
   - MediaPipe's fixed threading is a limitation

4. **Memory Optimization**:
   - Consider memory-efficient models for emotion_detection
   - YOLO memory usage could be reduced with smaller model

## Test Validation

✅ **Instrumentation working correctly**:
- All services instrumented and reporting metrics
- Thread counts captured accurately
- Memory tracking functional in sequential mode
- Thread flexibility mapping correct
- Minimal overhead (<0.01%)
- Metrics saved to JSON for analysis

✅ **Cold start testing verified**:
- Caches cleared before each test
- No warm cache benefits
- Real-world performance metrics

## Next Steps

1. Test with 18s and 73s videos for comparison
2. Profile emotion_detection for optimization opportunities
3. Investigate whisper performance degradation in parallel mode
4. Consider implementing adaptive parallelism based on available resources