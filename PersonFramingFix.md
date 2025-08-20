# Person Framing Fix - ML Cache Removal

**CRITICAL**: Stale cache from Aug 12 lacks bbox field (added Aug 14), causing zero metrics  
**Date**: 2025-08-20  
**Problem**: Person framing returns zeros due to missing bbox in cached ML data  
**Solution**: Remove ML output caching from video_analyzer.py  
**Impact**: +5 seconds processing, 100% data accuracy  

---

## IMMEDIATE ACTIONS

**Do these steps NOW to fix person framing:**

1. **Clear stale cache** (takes 5 seconds):
   ```bash
   rm -rf human_analysis_outputs/7482079864242146603/
   ```

2. **Remove MediaPipe caching** (most critical fix):
   ```bash
   # Edit rumiai_v2/processors/video_analyzer.py
   # Delete lines 168-180 in _run_mediapipe()
   ```

3. **Test the fix** (proves cache removal works):
   ```bash
   python3 scripts/rumiai_runner.py [video_url]
   
   # Verify bbox now exists:
   grep bbox human_analysis_outputs/7482079864242146603/*.json
   # Should show: "bbox": {"x": 0.295, "y": 0.345...}
   
   # Verify metrics now work:
   grep averageFaceSize insights/*/person_framing/*.json
   # Should show: "averageFaceSize": 12.49 (not 0)
   ```

4. **Then remove caching from remaining 4 ML services** (see Implementation section)

---

## Problem

Person framing analysis returns all zeros despite faces being detected:
```json
"averageFaceSize": 0,        // Should be >0 with faces
"faceVisibilityRate": 0.0,   // Should be >0 with faces  
"eyeContactRate": 0.0        // Should be >0 with gaze data
```

---

## Proof of Root Cause

**BEFORE (with stale cache from Aug 12):**
```bash
$ ls -la human_analysis_outputs/7482079864242146603/*.json
-rw-r--r-- Aug 12 16:43 7482079864242146603_human_analysis.json  # OLD!

$ grep bbox human_analysis_outputs/7482079864242146603/*.json
# NO RESULTS - bbox field missing

$ grep averageFaceSize insights/*/person_framing/*.json
"averageFaceSize": 0  # BROKEN
```

**AFTER (fresh ML run):**
```bash
$ rm -rf human_analysis_outputs/7482079864242146603/  # Clear cache
$ python3 scripts/rumiai_runner.py [video_url]

$ grep bbox human_analysis_outputs/7482079864242146603/*.json
"bbox": {"x": 0.295, "y": 0.345, "width": 0.394, "height": 0.222}  # FIXED!

$ grep averageFaceSize insights/*/person_framing/*.json  
"averageFaceSize": 12.49  # WORKING!
```

---

## Root Cause

**The caching bug in `video_analyzer.py`**:

1. Aug 12: Video processed, MediaPipe creates cache without bbox field
2. Aug 14: Bbox feature added to code
3. Aug 20: Same video uses stale cache → missing bbox → zero metrics

```python
# BROKEN: Lines 168-180 in video_analyzer.py
if output_path.exists():
    with open(output_path, 'r') as f:
        data = json.load(f)  # OLD DATA WITHOUT BBOX!
    return MLAnalysisResult(...)
```

---

## Solution: Remove ML Caching

**Why this works**:
- Frame extraction (expensive, 10-30s) remains cached ✅
- ML inference (cheap, 1-2s) runs fresh every time ✅
- Eliminates all cache-related bugs permanently ✅

---

## Implementation

### Remove caching from ALL 5 ML methods in `video_analyzer.py`:

**1. MediaPipe** (`_run_mediapipe`, lines 168-180):
```python
# DELETE lines 168-180:
if output_path.exists():
    logger.info(f"Using existing MediaPipe output: {output_path}")
    with open(output_path, 'r') as f:
        data = json.load(f)
    return MLAnalysisResult(
        model_name='mediapipe',
        model_version='0.10',
        success=True,
        data=data,
        processing_time=0.0
    )
```

**2. YOLO** (`_run_yolo`, lines 86-97):
```python
# DELETE lines 86-97:
if output_path.exists():
    logger.info(f"Using existing YOLO output: {output_path}")
    with open(output_path, 'r') as f:
        data = json.load(f)
    return MLAnalysisResult(
        model_name='yolo',
        model_version='8x',
        success=True,
        data=data,
        processing_time=0.0
    )
```

**3. OCR** (`_run_ocr`, lines 213-224):
```python
# DELETE lines 213-224:
if output_path.exists():
    logger.info(f"Using existing OCR output: {output_path}")
    with open(output_path, 'r') as f:
        data = json.load(f)
    return MLAnalysisResult(
        model_name='ocr',
        model_version='1.0',
        success=True,
        data=data,
        processing_time=0.0
    )
```

**4. Whisper** (`_run_whisper`, lines 122-133):
```python
# DELETE lines 122-133:
if output_path.exists():
    logger.info(f"Using existing Whisper output: {output_path}")
    with open(output_path, 'r') as f:
        segments = json.load(f)
    return MLAnalysisResult(
        model_name='whisper',
        model_version='base',
        success=True,
        data={'segments': segments},
        processing_time=0.0
    )
```

**5. Emotion** (`_run_emotion_detection`, lines 365-376):
```python
# DELETE lines 365-376:
if output_path.exists():
    logger.info(f"Using existing emotion output: {output_path}")
    with open(output_path, 'r') as f:
        data = json.load(f)
    return MLAnalysisResult(
        model_name='emotion',
        model_version='1.0',
        success=True,
        data=data,
        processing_time=0.0
    )
```

---

## Testing

```bash
# 1. Clear old cache
rm -rf human_analysis_outputs/7482079864242146603/

# 2. Process video
python3 scripts/rumiai_runner.py [video_url]

# 3. Verify metrics are non-zero
cat insights/*/person_framing/*.json | grep -E "averageFaceSize|faceVisibilityRate"

# Expected:
# "averageFaceSize": 12.49     ✅
# "faceVisibilityRate": 1.02   ✅
```

---

## Impact

**Performance**:
- Before: ~75 seconds (with cache)
- After: ~80 seconds (+5 seconds)
- Worth it: 100% data accuracy

**Data Quality**:
- Before: Stale cache → wrong results
- After: Fresh ML data → accurate metrics

---

## Risk Assessment

**Performance Impact:**
- **Expected**: +5 seconds for ML inference per video
- **Worst case**: +30-120 seconds if frame extraction cache also fails
- **Development impact**: Slower testing cycles without cache

**Code Dependencies Found:**
- **temporal_markers.py (line 439)**: Checks for YOLO cache file existence
  - Fix: Update to use ML results directly instead of file check
- **error_handler.py**: Suggests cache clearing in error messages
  - Fix: Update error messages to reflect new no-cache approach

**System Impact:**
- **No downstream dependencies** on cache files
- **No monitoring systems** affected
- **No backup/restore** procedures impacted
- **Git already ignores** all ML output directories

**Risk Level: LOW** - Safe to proceed with identified fixes

---

## Summary

Remove ML output caching to ensure fresh data flows to Python analysis functions. The 5-second performance cost is negligible compared to guaranteed accuracy in the $0.00 Python-only pipeline.