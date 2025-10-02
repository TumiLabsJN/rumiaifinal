# YOLO GPU Acceleration - Implementation Plan

## Problem Statement

**Current Status**: YOLO object detection is running on **CPU** despite CUDA being available on the system.

**Evidence**:
- CUDA available: ✅ True (version 12.1)
- YOLO device: ❌ **cpu** (confirmed via runtime check)
- Code location: `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py:69-83`

**Impact**:
- **Performance**: CPU processing is 10-100x slower than GPU
- **Batch Processing Time**: 300 videos taking 6-8 hours instead of 30-60 minutes
- **Cost**: Wasted compute resources and extended processing windows

---

## Root Cause Analysis

### Code Investigation

**File**: `ml_services_unified.py`

**Current Implementation** (lines 69-83):
```python
async def _load_yolo_model(self):
    """Load YOLO model asynchronously"""
    try:
        from ultralytics import YOLO

        model_path = '/home/jorge/rumiaifinal/yolov8n.pt'
        if os.path.exists(model_path):
            model = await asyncio.to_thread(YOLO, model_path)
        else:
            model = await asyncio.to_thread(YOLO, 'yolov8n.pt')

        return model  # ❌ No device assignment - defaults to CPU
    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        return None
```

**Issue**: No explicit device assignment after model loading. Ultralytics YOLO defaults to CPU when device is not specified during initialization.

### Usage Points

YOLO model is used at:
- **Line 288**: `model.track()` - Object tracking with persistence
- **Line 234**: `await self._ensure_model_loaded('yolo')` - Lazy loading trigger

---

## Proposed Solution

### Option 1: Explicit GPU Assignment (Recommended)

**Approach**: Add `.to('cuda')` after model initialization

**Implementation**:
```python
async def _load_yolo_model(self):
    """Load YOLO model asynchronously with GPU support"""
    try:
        from ultralytics import YOLO
        import torch

        model_path = '/home/jorge/rumiaifinal/yolov8n.pt'
        if os.path.exists(model_path):
            model = await asyncio.to_thread(YOLO, model_path)
        else:
            model = await asyncio.to_thread(YOLO, 'yolov8n.pt')

        # Force GPU usage if available
        if torch.cuda.is_available():
            model.to('cuda')
            logger.info(f"YOLO loaded on GPU: {model.device}")
        else:
            logger.warning("YOLO using CPU (CUDA not available)")

        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        return None
```

**Pros**:
- ✅ Simple, minimal code change
- ✅ Automatic fallback to CPU if CUDA unavailable
- ✅ Clear logging of device usage
- ✅ No changes needed to `model.track()` calls

**Cons**:
- None significant

---

### Option 2: Device Parameter During Initialization

**Approach**: Pass device during YOLO instantiation

**Implementation**:
```python
async def _load_yolo_model(self):
    """Load YOLO model asynchronously with GPU support"""
    try:
        from ultralytics import YOLO
        import torch

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading YOLO on device: {device}")

        model_path = '/home/jorge/rumiaifinal/yolov8n.pt'
        if os.path.exists(model_path):
            model = await asyncio.to_thread(YOLO, model_path, device=device)
        else:
            model = await asyncio.to_thread(YOLO, 'yolov8n.pt', device=device)

        logger.info(f"YOLO loaded successfully on {model.device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        return None
```

**Pros**:
- ✅ Device set during initialization (potentially cleaner)
- ✅ Automatic fallback to CPU

**Cons**:
- ⚠️ Ultralytics YOLO may not support `device` parameter in constructor (needs verification)

---

## Recommended Approach

**Use Option 1**: Explicit GPU assignment with `.to('cuda')`

**Rationale**:
- Proven to work with Ultralytics YOLO
- Clear separation: load model → assign device
- Better logging and diagnostics
- Fail-safe fallback to CPU

---

## Implementation Steps

### 1. Verify Current Behavior
```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check current YOLO device
python3 -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); print(f'Device: {m.device}')"
```

**Expected Output**:
- CUDA: True
- Device: cpu ❌ (current state)

---

### 2. Update `ml_services_unified.py`

**File**: `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py`

**Change**: Lines 69-83

**Before**:
```python
async def _load_yolo_model(self):
    """Load YOLO model asynchronously"""
    try:
        from ultralytics import YOLO

        model_path = '/home/jorge/rumiaifinal/yolov8n.pt'
        if os.path.exists(model_path):
            model = await asyncio.to_thread(YOLO, model_path)
        else:
            model = await asyncio.to_thread(YOLO, 'yolov8n.pt')

        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        return None
```

**After**:
```python
async def _load_yolo_model(self):
    """Load YOLO model asynchronously with GPU support"""
    try:
        from ultralytics import YOLO
        import torch

        model_path = '/home/jorge/rumiaifinal/yolov8n.pt'
        if os.path.exists(model_path):
            model = await asyncio.to_thread(YOLO, model_path)
        else:
            model = await asyncio.to_thread(YOLO, 'yolov8n.pt')

        # Force GPU usage if available
        if torch.cuda.is_available():
            model.to('cuda')
            logger.info(f"✓ YOLO loaded on GPU: {model.device}")
        else:
            logger.warning("⚠ YOLO using CPU (CUDA not available)")

        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        return None
```

---

### 3. Test GPU Acceleration

**Create Test Script**: `test_yolo_gpu.py`

```python
import asyncio
import time
import numpy as np
from rumiai_v2.api.ml_services_unified import UnifiedMLServices

async def test_yolo_gpu():
    """Test YOLO GPU acceleration"""

    services = UnifiedMLServices()

    # Load YOLO model
    model = await services._ensure_model_loaded('yolo')

    print(f"YOLO Device: {model.device}")

    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Benchmark CPU vs GPU
    print("\n=== Benchmarking YOLO Performance ===")

    # Warm-up run
    _ = model.track(dummy_frame, persist=True, verbose=False)

    # Timed runs
    num_runs = 10
    start = time.time()

    for i in range(num_runs):
        results = model.track(dummy_frame, persist=True, verbose=False)

    elapsed = time.time() - start
    avg_time = elapsed / num_runs

    print(f"Average inference time: {avg_time:.4f}s per frame")
    print(f"Estimated FPS: {1/avg_time:.2f}")
    print(f"\nFor 300 videos × 100 frames:")
    print(f"  Total time: {(300 * 100 * avg_time) / 3600:.2f} hours")

if __name__ == "__main__":
    asyncio.run(test_yolo_gpu())
```

**Expected Results**:
- **CPU**: ~0.5-2s per frame → 4-17 hours for 300 videos
- **GPU**: ~0.01-0.05s per frame → 0.08-0.4 hours for 300 videos

---

### 4. Verify in Production

**Monitor Logs**: Check for GPU confirmation message during model load

```bash
# Run actual video analysis
python3 rumiai_runner.py --video test_video.mp4

# Look for log output:
# "✓ YOLO loaded on GPU: cuda:0"
```

**Check GPU Usage**:
```bash
# Monitor GPU utilization during video processing
nvidia-smi -l 1
```

**Expected**: GPU utilization should spike to 30-90% during YOLO processing

---

## Performance Impact

### Before (CPU)

**Single Video Processing**:
- YOLO inference: ~0.5-2s per frame
- 100 frames per video: 50-200 seconds
- 300 videos: **4-17 hours**

### After (GPU)

**Single Video Processing**:
- YOLO inference: ~0.01-0.05s per frame
- 100 frames per video: 1-5 seconds
- 300 videos: **5-25 minutes**

**Speedup**: **10-100x faster**

---

## Risks & Mitigation

### Risk 1: CUDA Out of Memory

**Scenario**: Large batch sizes cause GPU memory overflow

**Mitigation**:
- Current batch size: 10 frames (line 244 in `ml_services_unified.py`)
- Monitor GPU memory: `nvidia-smi`
- Reduce batch size if needed: `batch_size = 5`

### Risk 2: CUDA Not Available

**Scenario**: System doesn't have CUDA or GPU

**Mitigation**:
- Automatic fallback to CPU in code
- Warning logged: "⚠ YOLO using CPU (CUDA not available)"
- No breaking changes

### Risk 3: Model Transfer Overhead

**Scenario**: `.to('cuda')` takes time during model load

**Mitigation**:
- One-time cost during lazy loading
- Amortized over thousands of frames
- Negligible compared to inference speedup

---

## Validation Checklist

**Before Deployment**:
- [ ] Verify CUDA available: `torch.cuda.is_available() == True`
- [ ] Check GPU memory: `nvidia-smi` shows available memory
- [ ] Review batch size: 10 frames per batch (line 244)

**After Deployment**:
- [ ] Confirm log message: "✓ YOLO loaded on GPU: cuda:0"
- [ ] Monitor GPU utilization: `nvidia-smi -l 1` shows activity
- [ ] Benchmark single video: <30 seconds total processing
- [ ] Test 10-video batch: Verify speedup vs historical baseline

**Regression Tests**:
- [ ] Object detection accuracy unchanged
- [ ] Tracking IDs persist correctly
- [ ] No CUDA errors in logs
- [ ] CPU fallback works if GPU unavailable

---

## Related Systems

### Other ML Services Using GPU

**MediaPipe** (line 85-118):
- ✅ Already configured to use GPU when available (line 129)
- Uses `torch.cuda.is_available()` check

**DeepFace/FEAT** (external library):
- Check if GPU support can be enabled
- May require additional configuration

**Whisper** (audio transcription):
- Check if Whisper model supports CUDA
- May benefit from GPU acceleration

---

## Future Optimizations

### 1. Dynamic Batch Sizing

**Current**: Fixed batch size of 10 frames

**Optimization**: Adjust batch size based on GPU memory
```python
# Detect available GPU memory
gpu_mem = torch.cuda.get_device_properties(0).total_memory
batch_size = min(50, gpu_mem // (1024**3))  # 1GB per batch estimate
```

### 2. Multi-GPU Support

**Future**: Distribute YOLO processing across multiple GPUs
```python
if torch.cuda.device_count() > 1:
    model.to(f'cuda:{gpu_id % torch.cuda.device_count()}')
```

### 3. Mixed Precision Inference

**Optimization**: Use FP16 for 2x speedup
```python
model.to('cuda')
# Enable mixed precision
with torch.cuda.amp.autocast():
    results = model.track(frame)
```

---

## Implementation Priority

**Priority**: **HIGH** - Critical performance optimization

**Effort**: Low (5-10 minute code change)

**Impact**: High (10-100x speedup for YOLO processing)

**Risk**: Low (automatic CPU fallback, no breaking changes)

**Timeline**:
- Day 1: Implement code change (5 minutes)
- Day 1: Test with single video (10 minutes)
- Day 1: Validate with 10-video batch (30 minutes)
- Day 2: Deploy to production, monitor logs

---

## Success Metrics

**Quantitative**:
- YOLO device shows `cuda:0` in logs ✅
- GPU utilization >30% during video processing ✅
- Video processing time reduced by >80% ✅

**Qualitative**:
- No CUDA errors in production logs
- Batch processing completes in <1 hour (vs 6-8 hours)
- Object detection quality unchanged

---

## References

- Ultralytics YOLO Docs: https://docs.ultralytics.com/modes/predict/#inference-arguments
- PyTorch CUDA Docs: https://pytorch.org/docs/stable/cuda.html
- Code Location: `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py:69-83`
