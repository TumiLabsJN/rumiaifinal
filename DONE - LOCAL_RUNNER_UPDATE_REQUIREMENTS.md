# Local Video Runner Update Requirements

## Executive Summary
Based on a comprehensive analysis comparing `rumiai_runner.py` (production) with `local_video_runner.py` (test), multiple critical updates are needed to ensure the test flow matches production output exactly.

## 🚨 Critical Changes Required

### 1. **Complete Data Structure Update** ✅ HIGH PRIORITY
The production runner has changed the `complete_data` structure in the output files.

**Current (local_video_runner.py line 329):**
```python
complete_data = {
    'analysis_type': analysis_type,
    'success': True,
    'result': json.dumps(result),
    'parsed_response': ml_data
}
```

**Required (from production):**
```python
complete_data = {
    "analysis_type": analysis_type,
    "success": True,
    "result": json.dumps(result_data),
    "parsed_response": ml_data,
    "source": "python_precompute",
    "processing_mode": "python_only",
    "version": "2.0"
}
```

**Impact:** Output files will have different structure, affecting downstream processing.

### 2. **SharedAudioExtractor Integration** ✅ MEDIUM PRIORITY
Production now uses a shared audio extraction service to avoid redundant audio processing.

**Add to local_video_runner.py:**
- Import: `from rumiai_v2.api.shared_audio_extractor import SharedAudioExtractor`
- Add cleanup after processing (around line 120):
```python
# Cleanup SharedAudioExtractor cache for this video
try:
    from rumiai_v2.api.shared_audio_extractor import SharedAudioExtractor
    SharedAudioExtractor.cleanup(video_id)
    self.logger.info(f"♻️ Cleaned up shared audio cache for {video_id}")
except Exception as cleanup_error:
    self.logger.warning(f"Failed to cleanup audio cache: {cleanup_error}")
```

**Impact:** Memory optimization and prevents audio file accumulation.

### 3. **Convert to ML Format Method** ✅ CRITICAL
The production and test runners use COMPLETELY DIFFERENT approaches for converting prefixed data to ML format.

**Production (rumiai_runner.py lines 104-124):** Uses dynamic prefix-based conversion
```python
def convert_to_ml_format(self, prefixed_data: dict, insight_type: str) -> dict:
    prefix = self.get_prefix_for_type(insight_type)  # Gets "density", "emotional", etc.
    
    for key, value in prefixed_data.items():
        if key.startswith(prefix):
            # Remove prefix and capitalize first letter
            new_key = key[len(prefix):]
            new_key = new_key[0].upper() + new_key[1:] if new_key else key
            ml_data[new_key] = value
```

**Local (local_video_runner.py lines 213-309):** Uses hardcoded mappings
```python
key_mappings = {
    'densityCoreMetrics': 'CoreMetrics',
    'densityDynamics': 'Dynamics',
    # ... 70+ hardcoded mappings
}
```

**ACTION REQUIRED:** Replace the entire `convert_to_ml_format` method in local_video_runner.py with the production version AND add the `get_prefix_for_type` method.

### 4. **File Saving Mechanism** ✅ HIGH PRIORITY
Production uses `FileHandler` with atomic writes, while local uses direct file operations.

**Production:** Uses `self.insights_handler.save_json()` with atomic writes
**Local:** Uses `open(file, 'w')` directly

**ACTION REQUIRED:** Either:
- Option A: Import and use FileHandler for atomic writes
- Option B: Keep direct writes but ensure consistency

### 5. **Unified Analysis Saving** ✅ MEDIUM PRIORITY
Production saves unified analysis using the model's built-in `save_to_file()` method with atomic writes.

**Production (line 284):**
```python
unified_analysis.save_to_file(str(unified_path))
```

**Local (lines 106-112):**
```python
with open(analysis_path, 'w') as f:
    json.dump(analysis.to_dict(), f, indent=2)
```

**ACTION REQUIRED:** Use `analysis.save_to_file()` method instead of manual JSON dump.

### 6. **GPU Verification** ✅ LOW PRIORITY
Production includes GPU verification at startup, local doesn't.

**Add to `__init__` method:**
```python
self._verify_gpu()
```

And add the method (optional for testing):
```python
def _verify_gpu(self) -> None:
    """Verify GPU/CUDA availability at startup."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"✅ GPU available: {device_name} with {memory:.1f}GB VRAM")
    except:
        self.logger.warning("⚠️ No GPU detected, using CPU")
```

### 7. **New ML Services** ✅ VERIFIED
Production has added:
- `librosa` - Audio analysis library (used in audio_energy_service.py)
- `FEAT` - Facial expression analysis (used in emotion_detection_service.py)
- `SharedAudioExtractor` - Shared audio caching service

These are encapsulated within the ML services and should work automatically.

### 8. **Report Generation** ✅ HIGH PRIORITY
Production generates a comprehensive report with metrics, local doesn't.

**Production includes:**
```python
def _generate_report(self, analysis, prompt_results):
    # Returns detailed report with:
    # - successful_prompts count
    # - total_cost and total_tokens
    # - memory usage statistics
    # - completion status details
```

**Local:** No report generation at all

**ACTION REQUIRED:** Add report generation or ensure output compatibility.

### 9. **Return Value Structure** ✅ CRITICAL
The return values are completely different between production and local.

**Production returns:**
```python
{
    'success': True,
    'video_id': video_id,
    'outputs': {
        'video': str(video_path),
        'unified': str(unified_path),
        'temporal': str(temporal_path),
        'insights': str(insights_dir)
    },
    'report': report,  # Detailed report object
    'metrics': self.metrics.get_all()  # Performance metrics
}
```

**Local returns:**
```python
{
    'video_id': video_id,
    'duration': duration,
    'ml_results_count': len(ml_results),
    'timeline_entries': len(analysis.timeline.entries),
    'analyses_generated': list(results.keys()),
    'success': True
}
```

**ACTION REQUIRED:** Align return structure or ensure consuming code handles both formats.

### 10. **Metrics and Performance Tracking** ✅ MEDIUM PRIORITY
Production tracks detailed metrics, local doesn't.

**Production has:**
- `Metrics()` class for timing operations
- `VideoProcessingMetrics()` for video-specific metrics
- Memory usage tracking with `_get_memory_usage()`
- Memory threshold checking with `_check_memory_threshold()`

**Local:** No metrics tracking

**ACTION REQUIRED:** Consider adding metrics if performance monitoring is needed.

### 11. **Missing Imports** ✅ LOW PRIORITY
Production imports several modules that local doesn't:

**Production only:**
- `dotenv` for environment variables
- `psutil` for system monitoring
- `gc` for garbage collection
- `time` for timing
- `argparse` for command-line arguments
- `Metrics`, `VideoProcessingMetrics` for tracking

**Impact:** These are mostly for monitoring and don't affect output directly.

### 12. **Python-Only Processing Mode** ✅ VERIFIED
Both runners use Python-only processing through the same `COMPUTE_FUNCTIONS` import.
**No changes needed.**

## 📋 Implementation Checklist

### Immediate Actions Required (in priority order):
1. [ ] **CRITICAL:** Replace `convert_to_ml_format` method with production version
2. [ ] **CRITICAL:** Add `get_prefix_for_type` method from production
3. [ ] **CRITICAL:** Align return value structure (may break consuming code if not handled)
4. [ ] **HIGH:** Update `complete_data` structure in `save_results()` method (line 329)
5. [ ] **HIGH:** Add report generation method (or stub it out)
6. [ ] **HIGH:** Consider using FileHandler for atomic file writes
7. [ ] **MEDIUM:** Use `analysis.save_to_file()` instead of manual JSON dump (line 111)
8. [ ] **MEDIUM:** Add SharedAudioExtractor import and cleanup
9. [ ] **MEDIUM:** Add metrics tracking (optional but helpful for debugging)
10. [ ] **LOW:** Add GPU verification method (optional for testing)
11. [ ] **LOW:** Add missing imports if needed (dotenv, psutil, etc.)

### Testing After Updates:
1. [ ] Run `python3 scripts/verify_sync.py` to check synchronization
2. [ ] Process test video: `python3 scripts/local_video_runner.py test_videos/video_01_highenergy_peaks.mp4`
3. [ ] Compare output structure with production using `check_all_formats.py`
4. [ ] Verify all 3 files are generated per analysis type (complete, ml, result)

## 🔍 Verification Commands

```bash
# After making changes, verify with:
python3 scripts/verify_sync.py

# Test with known video
python3 scripts/local_video_runner.py test_videos/video_01_highenergy_peaks.mp4

# Check output structure
ls -la insights/video_01_highenergy_peaks/*/*.json | head -20

# Compare with production (if available)
python3 scripts/compare_ml_results.py <prod_id> video_01_highenergy_peaks creative_density
```

## ⚠️ Most Important Finding

**The `convert_to_ml_format` method difference is CRITICAL** - this directly affects how the ML format JSON files are generated. The production version uses a smart prefix-based approach while the local version uses hardcoded mappings that may not cover all cases or new analysis types.

## 📝 Complete Summary of Differences Found

| Component | Production | Local | Impact |
|-----------|------------|-------|---------|
| `convert_to_ml_format` | Dynamic prefix-based | Hardcoded mappings | **CRITICAL** - ML file structure |
| Return value structure | Complex with outputs, report, metrics | Simple summary | **CRITICAL** - API compatibility |
| `complete_data` fields | Has source, processing_mode, version | Missing these fields | **HIGH** - Output structure |
| Report generation | `_generate_report()` with detailed metrics | None | **HIGH** - Missing analysis summary |
| File saving | FileHandler with atomic writes | Direct file writes | **MEDIUM** - Data integrity |
| Unified analysis save | Uses model's save_to_file() | Manual JSON dump | **MEDIUM** - Consistency |
| Metrics tracking | Comprehensive Metrics class | None | **MEDIUM** - Performance monitoring |
| Memory tracking | `_get_memory_usage()` and threshold checks | None | **MEDIUM** - Resource monitoring |
| SharedAudioExtractor | Has cleanup | No cleanup | **LOW** - Memory usage |
| GPU verification | Checks at startup | No check | **LOW** - Debugging info |
| Missing imports | dotenv, psutil, gc, argparse, Metrics | None | **LOW** - Monitoring features |

## 🔍 Additional Differences in Error Handling

### Failed Analysis Handling
**Production:** Doesn't save failed analyses, just logs error and continues
```python
except Exception as e:
    logger.error(f"Precompute {func_name} failed: {e}")
    prompt_results[func_name] = {}  # Empty dict
```

**Local:** Saves error information in results
```python
except Exception as e:
    self.logger.error(f"    ❌ {func_name} failed: {str(e)}")
    results[func_name] = {
        'error': str(e),
        'success': False
    }
```

This means local might generate error files that production wouldn't create.

## 📝 Final Notes

- The fail-fast architecture is preserved in both runners
- All 7 analysis types + temporal markers are implemented identically
- The same ML services (YOLO, MediaPipe, OCR, Whisper, Scene Detection) are called
- New ML services (librosa, FEAT) are encapsulated within existing service structure
- **CRITICAL FINDING: The convert_to_ml_format difference could cause significant output inconsistencies**
- **CRITICAL FINDING: Return value structures are completely different, which could break consuming code**
- **Total differences found: 12 major differences affecting output, performance tracking, and error handling**

## Confidence Level
After this comprehensive analysis, I have **high confidence** that these are the major differences. However, subtle differences in data processing logic within shared modules could still exist.

## Last Updated
2025-08-20

## Git Reference
- Last sync commit: ebfba83e9512679fc74dc96740891af7b67e0669
- Latest production changes: 788b31c (Speech Fixed)