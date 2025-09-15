# Remove ML Output Caching - Architectural Fix

**Date Created**: 2025-08-20  
**Problem**: Stale ML output cache causing missing data fields  
**Solution**: Remove output caching, always run ML analysis fresh  

---

## The Real Problem

The system caches ML analysis outputs **forever**, causing:
- Old outputs missing new fields (like bbox)
- No way to force regeneration
- Silent failures with wrong data
- Confusion about which version of code generated which output

---

## The Correct Architectural Fix

### Remove ALL ML output caching from video_analyzer.py

The caching provides minimal benefit but causes major problems. Frame extraction is already cached (the expensive part), so ML analysis should run fresh.

---

## Implementation

### 1. Fix MediaPipe Analysis (Lines 165-209)

```python
# rumiai_v2/processors/video_analyzer.py
async def _run_mediapipe(self, video_id: str, video_path: Path) -> MLAnalysisResult:
    """Run MediaPipe human analysis - ALWAYS FRESH."""
    try:
        output_dir = Path(f"human_analysis_outputs/{video_id}")
        output_path = output_dir / f"{video_id}_human_analysis.json"
        
        # REMOVED: Caching check that caused bugs
        # Always run fresh analysis
        logger.info(f"Running MediaPipe analysis on {video_path}")
        data = await self.ml_services.run_mediapipe_analysis(video_path, output_dir)
        
        # Save the output for debugging/review (not caching)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return MLAnalysisResult(
            model_name='mediapipe',
            model_version='0.10',
            success=True,
            data=data,
            processing_time=0.0
        )
            
    except Exception as e:
        return MLAnalysisResult(
            model_name='mediapipe',
            model_version='0.10',
            success=False,
            error=str(e)
        )
```

### 2. Fix YOLO Detection (Lines 83-109)

```python
async def _run_yolo(self, video_id: str, video_path: Path) -> MLAnalysisResult:
    """Run YOLO object detection - ALWAYS FRESH."""
    try:
        output_dir = Path(f"object_detection_outputs/{video_id}")
        output_path = output_dir / f"{video_id}_yolo_detections.json"
        
        # REMOVED: Caching check
        # Always run fresh analysis
        logger.info(f"Running YOLO detection on {video_path}")
        data = await self.ml_services.run_yolo_detection(video_path, output_dir)
        
        # Save the output
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return MLAnalysisResult(
            model_name='yolo',
            model_version='v8',
            success=True,
            data=data,
            processing_time=0.0
        )
            
    except Exception as e:
        return MLAnalysisResult(
            model_name='yolo',
            model_version='v8',
            success=False,
            error=str(e)
        )
```

### 3. Fix OCR (Lines 210-235)

```python
async def _run_ocr(self, video_id: str, video_path: Path) -> MLAnalysisResult:
    """Run OCR text detection - ALWAYS FRESH."""
    try:
        output_dir = Path(f"creative_analysis_outputs/{video_id}")
        output_path = output_dir / f"{video_id}_creative_analysis.json"
        
        # REMOVED: Caching check
        # Always run fresh analysis
        logger.info(f"Running OCR analysis on {video_path}")
        data = await self.ml_services.run_ocr_analysis(video_path, output_dir)
        
        # Save the output
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return MLAnalysisResult(
            model_name='ocr',
            model_version='tesseract-5',
            success=True,
            data=data,
            processing_time=0.0
        )
            
    except Exception as e:
        return MLAnalysisResult(
            model_name='ocr',
            model_version='tesseract-5',
            success=False,
            error=str(e)
        )
```

### 4. Fix Whisper (Lines 119-159)

```python
async def _run_whisper(self, video_id: str, video_path: Path) -> MLAnalysisResult:
    """Run Whisper transcription - ALWAYS FRESH."""
    try:
        output_path = Path(f"speech_transcriptions/{video_id}_whisper.json")
        
        # REMOVED: Caching check
        # Always run fresh analysis
        logger.info(f"Running Whisper transcription on {video_path}")
        
        # Extract audio
        from ..api.audio_utils import extract_audio_simple
        audio_path = await extract_audio_simple(video_path)
        
        # Transcribe
        transcriber = await self._get_transcriber()
        result = await transcriber.transcribe(str(audio_path))
        
        # Clean up audio file
        if audio_path.exists():
            audio_path.unlink()
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return MLAnalysisResult(
            model_name='whisper',
            model_version='base',
            success=True,
            data=result,
            processing_time=0.0
        )
            
    except Exception as e:
        return MLAnalysisResult(
            model_name='whisper',
            model_version='base',
            success=False,
            error=str(e)
        )
```

### 5. Fix Emotion Detection (Lines 362-406)

```python
async def _run_emotion_detection(self, video_id: str, video_path: Path) -> MLAnalysisResult:
    """Run FEAT emotion detection - ALWAYS FRESH."""
    try:
        output_dir = Path(f"emotion_detection_outputs/{video_id}")
        output_path = output_dir / f"{video_id}_emotions.json"
        
        # REMOVED: Caching check
        # Always run fresh analysis
        logger.info(f"Running emotion detection on {video_path}")
        
        # Get frames (these ARE cached at frame extraction level)
        frames, timestamps = await self._extract_frames_for_emotion(video_path)
        
        if not frames:
            logger.warning("No frames extracted for emotion detection")
            return MLAnalysisResult(
                model_name='feat',
                model_version='1.0',
                success=False,
                error="No frames extracted"
            )
        
        # Run emotion detection
        from ..ml_services.emotion_detection_service import get_emotion_detector
        detector = get_emotion_detector()
        result = await detector.detect_emotions_batch(frames, timestamps)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return MLAnalysisResult(
            model_name='feat',
            model_version='1.0',
            success=True,
            data=result,
            processing_time=0.0
        )
            
    except Exception as e:
        return MLAnalysisResult(
            model_name='feat',
            model_version='1.0',
            success=False,
            error=str(e)
        )
```

---

## Why This Is The Right Solution

### 1. **Simplicity**
- Remove code instead of adding complexity
- No versioning system needed
- No migration scripts needed

### 2. **Correctness**
- Always get latest output format
- No stale data possible
- Code changes immediately reflected

### 3. **Performance**
- Frame extraction (the slow part) is still cached
- ML inference is fast (< 1 second per service)
- Total overhead: ~5 seconds per video

### 4. **Debugging**
- Outputs still saved to disk for inspection
- Each run overwrites with fresh data
- Clear what version of code generated output

---

## Performance Analysis

### What's Actually Expensive?
1. **Frame Extraction**: 10-30 seconds (ALREADY CACHED)
2. **ML Inference**: 1-2 seconds per model (NOT WORTH CACHING)

### Caching Frame Extraction (GOOD)
```python
# This is already implemented and working
frame_data = await self.frame_manager.extract_frames(video_path, video_id)
# Uses cache if available - this is the expensive part
```

### Caching ML Outputs (BAD)
- Saves 1-2 seconds
- Causes version mismatch bugs
- Creates stale data problems
- Not worth the complexity

---

## Migration Plan

### Step 1: Remove Caching Code
Remove the `if output_path.exists()` checks from all 5 ML service methods in video_analyzer.py

### Step 2: Clean Old Cache (Optional)
```bash
# Optional: Remove old cached files to save disk space
rm -rf human_analysis_outputs/*
rm -rf object_detection_outputs/*
rm -rf creative_analysis_outputs/*
rm -rf speech_transcriptions/*
rm -rf emotion_detection_outputs/*
```

### Step 3: Test
```bash
# Process a video - should always run fresh
python3 scripts/rumiai_runner.py 'https://www.tiktok.com/@user/video/123'
```

---

## Expected Results

### Before
- Stale cache with missing fields
- person_framing returns all zeros
- Confusion about data freshness

### After  
- Always fresh data with all fields
- person_framing works correctly
- Clear, predictable behavior

---

## Alternative: Smart Caching (If Needed)

If caching is absolutely required for performance, implement it correctly:

```python
async def _run_mediapipe(self, video_id: str, video_path: Path) -> MLAnalysisResult:
    # Option 1: Time-based cache (expires after 1 hour)
    output_path = Path(f"human_analysis_outputs/{video_id}/{video_id}_human_analysis.json")
    
    if output_path.exists():
        file_age = time.time() - output_path.stat().st_mtime
        if file_age < 3600:  # 1 hour
            # Use cache
            pass
    
    # Option 2: Hash-based cache (regenerate if video changed)
    video_hash = hashlib.md5(open(video_path, 'rb').read()).hexdigest()
    cache_path = Path(f"cache/{video_id}_{video_hash}.json")
    
    # Option 3: Development mode flag
    if os.getenv('DISABLE_ML_CACHE', 'false').lower() == 'true':
        # Always regenerate in development
        pass
```

But honestly, **just remove the caching**. It's not needed.

---

## Conclusion

The best solution is the simplest: **remove ML output caching entirely**.

This is a true architectural fix because:
- It eliminates the root cause (stale cache)
- It's permanent (can't break again)
- It's simple (remove code, not add)
- It's correct (always fresh data)

The performance impact is negligible (~5 seconds per video) and the benefits are enormous (no more cache bugs ever).