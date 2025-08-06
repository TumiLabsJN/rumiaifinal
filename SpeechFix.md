# Speech Analysis Fix - Pragmatic Approach
**Date**: 2025-08-05  
**Author**: Claude  
**Status**: Implementation Plan (Revised)  
**Issue**: Whisper transcription failing with "Cannot set attribute 'src' directly" error

---

## Executive Summary

The speech analysis pipeline is failing due to a Whisper/PyTorch compatibility issue. Instead of rebuilding the entire architecture, we'll take a pragmatic approach: fix the immediate issue first, then add minimal audio extraction only if necessary.

**Original estimate**: 8-12 hours  
**Revised estimate**: 30 minutes to 3 hours

---

## Root Cause Analysis

### The Error
```
"Cannot set attribute 'src' directly. Use '_unsafe_update_src()' and manually clear `.hash` of all callers instead."
```

This is a **known PyTorch tensor modification issue** that occurs when:
1. Whisper tries to modify internal tensor attributes
2. Version mismatch between Whisper and PyTorch
3. Whisper's internal ffmpeg audio loading fails

### Current State Analysis
- Some videos transcribe successfully (evidence from other videos in the system)
- The architecture isn't fundamentally broken
- The issue is likely configuration/version specific

---

## Solution Approach: Try Simple First

### Phase 1: Fix Whisper Configuration (30 minutes)

#### 1.1 Update `whisper_transcribe_safe.py`

```python
# Location: rumiai_v2/api/whisper_transcribe_safe.py

# Fix 1: Add explicit parameters to prevent tensor modification
result = await asyncio.to_thread(
    model.transcribe,
    str(video_path),
    language=language,
    word_timestamps=True,
    fp16=False,
    verbose=False,  # ADD: Prevents progress bar tensor updates
    task='transcribe',  # ADD: Explicit task specification
    without_timestamps=False,  # ADD: Explicit timestamp handling
    condition_on_previous_text=False  # ADD: Prevents context tensor updates
)

# Fix 2: Handle Python version compatibility
# Replace line 64:
# async with asyncio.timeout(timeout):  # Python 3.11+ only

# With:
try:
    result = await asyncio.wait_for(
        asyncio.to_thread(
            model.transcribe,
            str(video_path),
            # ... parameters above
        ),
        timeout=timeout
    )
except asyncio.TimeoutError:
    logger.error(f"Transcription timed out after {timeout}s")
    return self._empty_result(error=f"Timeout after {timeout}s")
```

#### 1.2 Test Immediately
```bash
cd /home/jorge/rumiaifinal
./venv/bin/python -c "
from rumiai_v2.api.whisper_transcribe_safe import get_transcriber
import asyncio

async def test():
    t = get_transcriber()
    result = await t.transcribe(Path('temp/7428757192624311594.mp4'))
    print(f'Success: {len(result.get(\"text\", \"\"))} chars')
    
asyncio.run(test())
"
```

**If this works, we're done. Skip to Phase 4.**

---

### Phase 2: Fix Version Dependencies (30 minutes) - ONLY IF PHASE 1 FAILS

#### 2.1 Pin Compatible Versions
```bash
# Create requirements-whisper.txt
openai-whisper==20230314
torch==2.0.1
torchaudio==2.0.2
ffmpeg-python==0.2.0

# Install
pip install -r requirements-whisper.txt
```

#### 2.2 Clear Whisper Cache
```bash
rm -rf ~/.cache/whisper
```

#### 2.3 Test Again
Run the same test from Phase 1.2

**If this works, we're done. Skip to Phase 4.**

---

### Phase 3: Add Minimal Audio Extraction (2 hours) - ONLY IF PHASES 1-2 FAIL

#### 3.1 Create Simple Audio Extractor
```python
# Location: rumiai_v2/api/audio_utils.py

import subprocess
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

async def extract_audio_simple(video_path: Path) -> Path:
    """
    Extract audio from video using ffmpeg.
    Returns path to temporary audio file.
    """
    # Create temp file with .wav extension
    temp_audio = tempfile.NamedTemporaryFile(
        suffix='.wav',
        delete=False,
        dir='/tmp'
    )
    temp_audio_path = Path(temp_audio.name)
    temp_audio.close()
    
    try:
        # Use ffmpeg to extract audio (16kHz mono WAV - Whisper's preferred format)
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            str(temp_audio_path)
        ]
        
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {stderr.decode()}")
            raise Exception(f"ffmpeg extraction failed")
            
        return temp_audio_path
        
    except Exception as e:
        # Clean up on failure
        if temp_audio_path.exists():
            temp_audio_path.unlink()
        raise
```

#### 3.2 Update Whisper Transcriber
```python
# In whisper_transcribe_safe.py, modify transcribe method:

async def transcribe(self, video_path: Path, 
                    timeout: int = 600,
                    language: Optional[str] = None) -> Dict[str, Any]:
    
    # Try direct transcription first
    try:
        # ... existing direct transcription code ...
    except Exception as e:
        logger.warning(f"Direct transcription failed: {e}, trying audio extraction")
        
        # Fallback: Extract audio first
        from .audio_utils import extract_audio_simple
        temp_audio = None
        
        try:
            temp_audio = await extract_audio_simple(video_path)
            
            # Transcribe extracted audio
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    model.transcribe,
                    str(temp_audio),
                    # ... parameters ...
                ),
                timeout=timeout
            )
            
            # ... process result ...
            
        finally:
            # Clean up temp audio
            if temp_audio and temp_audio.exists():
                temp_audio.unlink()
```

---

### Phase 4: Improve Error Handling (30 minutes)

#### 4.1 Fix Error State Representation
```python
# In whisper_transcribe_safe.py

def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
    """Return empty result with proper error state"""
    result = {
        'text': '',
        'segments': [],
        'language': 'unknown', 
        'duration': 0,
        'metadata': {
            'model': self._model_size or 'base',
            'processed': False,
            'success': False  # ADD: Explicit failure flag
        }
    }
    if error:
        result['metadata']['error'] = error  # Move error to metadata
    return result
```

#### 4.2 Update Timeline Builder
```python
# In timeline_builder.py, when processing whisper data:

if whisper_data.get('metadata', {}).get('success', True):
    # Process normally
else:
    logger.warning(f"Skipping failed transcription: {whisper_data.get('metadata', {}).get('error')}")
    # Don't add empty segments
```

---

## Testing Checklist

### Immediate Tests (After Each Phase)
- [ ] Test with problematic video (7428757192624311594)
- [ ] Test with known working video
- [ ] Verify error states are properly represented
- [ ] Check logs for clear error messages

### Final Validation
- [ ] Run full pipeline: `python test_unified_pipeline_e2e.py temp/7428757192624311594.mp4`
- [ ] Verify speech segments appear in timeline
- [ ] Verify Claude receives actual transcription data
- [ ] Check no regression on working videos

---

## Decision Tree

```
Start → Try Phase 1 (Config Fix)
         ├─ Works? → Done! (30 mins)
         └─ Fails? → Try Phase 2 (Version Fix)
                      ├─ Works? → Done! (1 hour)
                      └─ Fails? → Try Phase 3 (Audio Extraction)
                                   ├─ Works? → Done! (3 hours)
                                   └─ Fails? → Investigate deeper issue
```

---

## Why This Approach Is Better

1. **Tries simplest fix first** - Often it's just configuration
2. **Preserves working functionality** - No risk to videos that work
3. **Minimal new code** - Less to maintain
4. **Fast validation** - Know within 30 minutes if simple fix works
5. **No new dependencies** - Uses existing libraries
6. **Follows YAGNI principle** - You Aren't Gonna Need It (fallback services)

---

## What We're NOT Doing (And Why)

### ❌ Full Audio Extraction Service
**Why not**: Whisper already does this internally with ffmpeg. We're just duplicating work.

### ❌ Fallback Transcription Services  
**Why not**: If Whisper fails, it's usually not a transcription issue but an audio/format issue. Another service won't help.

### ❌ Audio Caching System
**Why not**: Audio files are large (50MB for 5 mins). Caching would fill disk quickly for marginal benefit.

### ❌ Complex Retry Logic
**Why not**: If it fails once, it'll likely fail again. Fix the root cause instead.

---

## Success Metrics

- **Primary Goal**: Video 7428757192624311594 transcribes successfully
- **No Regression**: Existing working videos still work
- **Clear Errors**: Failed transcriptions show clear error messages
- **Performance**: No significant slowdown (<5s added)

---

## Timeline

### Best Case: 30 minutes
Phase 1 works, problem solved

### Likely Case: 1 hour  
Phase 1 + Phase 2 needed

### Worst Case: 3 hours
All phases needed + testing

### Compare to Original: 8-12 hours
**Time saved: 5-11 hours**

---

## Conclusion

This pragmatic approach:
1. **Fixes the immediate problem** without overengineering
2. **Preserves what works** while fixing what doesn't
3. **Minimizes new code** and technical debt
4. **Provides fast feedback** on what actually works

Remember: The best code is no code. The best architecture change is no architecture change. Fix the bug, not the architecture (unless the architecture IS the bug).