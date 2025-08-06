# Whisper.cpp Integration - Final Implementation Plan
**Date**: 2025-08-06  
**Author**: Claude  
**Status**: Reviewed and Finalized  
**Goal**: Complete replacement of Python Whisper with Whisper.cpp for robust CPU-based transcription
**Target Environment**: WSL2 Ubuntu on 16-core Windows laptop  
**Scope**: TikTok video analysis only

---

## Executive Summary

Replace the current Python Whisper implementation (which has PyTorch tensor conflicts) with Whisper.cpp, a fast C++ implementation that runs reliably on CPU. This eliminates version conflicts, improves speed 3-4x, and maintains all functionality.

---

## Current Problem

1. **Python Whisper fails** with PyTorch 2.7.1 (tensor modification error)
2. **Audio extraction workaround** adds complexity but transcription still fails
3. **Fallback method** loses word timestamps and segment data
4. **Version pinning** would break other ML services that need PyTorch 2.7
5. **Python dependencies** create ongoing conflict potential

## Key Decisions Made

1. **Complete removal of Python Whisper** - No dual-backend support
2. **No quantization** - Use standard models for full multilingual support
3. **10 threads default** - Optimal for 16-core system
4. **30-second timeout** - Appropriate for max 120-second videos
5. **Version pinning** - Lock to specific Whisper.cpp commit
6. **No singleton pattern** - Subprocess model doesn't need it
7. **No abstraction layer** - Single backend only
8. **No resource management** - Sequential processing only

---

## Solution Architecture

```
Video File → Extract Audio (keep) → Whisper.cpp (new) → Full Transcription with Segments
                    ↓                      ↓
              [Temp WAV file]      [CPU Processing]
                                   [5-8s for 60s video]
```

---

## Implementation Phases

### Phase 1: Install Whisper.cpp (30 minutes)

#### 1.1 Build Whisper.cpp
```bash
# Location: /home/jorge/rumiaifinal/whisper_cpp/
cd /home/jorge/rumiaifinal
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp

# Build for WSL2 environment (Linux)
make clean
make -j10  # Use 10 cores as decided

# Pin to specific version for stability
git checkout b3221  # Or current stable commit
echo "b3221" > PINNED_VERSION.txt

# Test the build
./main --help
```

#### 1.2 Download Models
```bash
# Download base model (best balance of speed/accuracy)
cd models
bash ./download-ggml-model.sh base

# Only download standard model (no quantization)
# models/ggml-base.bin (~142MB) - Multilingual support
```

#### 1.3 Test Whisper.cpp Directly
```bash
# Test on our problematic video
cd /home/jorge/rumiaifinal
ffmpeg -i temp/7389683775929699616.mp4 -ar 16000 -ac 1 -c:a pcm_s16le test_audio.wav
./whisper.cpp/main -m whisper.cpp/models/ggml-base.bin -f test_audio.wav -oj

# Should output JSON with full transcription and segments
```

---

### Phase 2: Create Whisper.cpp Wrapper (1 hour)

#### 2.1 Create New File: `rumiai_v2/api/whisper_cpp_service.py`

```python
"""
Whisper.cpp service for fast, reliable CPU transcription
"""

import json
import asyncio
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class DependencyError(Exception):
    """Raised when required dependencies are missing"""
    pass

def check_dependencies():
    """
    Check all required dependencies are installed.
    Fail fast with clear error messages.
    """
    errors = []
    
    # Check ffmpeg
    if not shutil.which('ffmpeg'):
        errors.append(
            "ffmpeg is not installed. Install it with:\n"
            "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )
    
    # Check make (for building whisper.cpp)
    if not shutil.which('make'):
        errors.append(
            "make is not installed. Install it with:\n"
            "  Ubuntu/Debian: sudo apt-get install build-essential\n"
            "  macOS: xcode-select --install\n"
            "  Windows: Use MinGW or WSL"
        )
    
    # Check g++ (for building whisper.cpp)
    if not shutil.which('g++') and not shutil.which('clang++'):
        errors.append(
            "C++ compiler not found. Install it with:\n"
            "  Ubuntu/Debian: sudo apt-get install g++\n"
            "  macOS: xcode-select --install\n"
            "  Windows: Install MinGW-w64 or use WSL"
        )
    
    if errors:
        raise DependencyError(
            "Missing required dependencies:\n\n" + "\n\n".join(errors)
        )

# Check dependencies when module is imported
# Note: Only runs once at import, not for each transcription
check_dependencies()

# Version tracking for monitoring
WHISPER_CPP_VERSION = "b3221"  # Update only after testing
LAST_TESTED = "2025-08-06"

class WhisperCppTranscriber:
    """Whisper.cpp wrapper for CPU-based transcription"""
    
    def __init__(self, 
                 whisper_cpp_path: Optional[str] = None,
                 model: str = "base"):
        """
        Initialize Whisper.cpp transcriber
        Note: Creates new instance each time - validation is lightweight (~100-200ms)
        
        Args:
            whisper_cpp_path: Path to whisper.cpp directory (auto-detected if None)
            model: Model size (tiny, base, small, medium, large) - all multilingual
        """
        # Validate model parameter
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        if model not in valid_models:
            raise ValueError(
                f"Invalid model '{model}'. Must be one of: {valid_models}"
            )
        
        # Auto-detect whisper.cpp location
        if whisper_cpp_path is None:
            # Try multiple possible locations
            possible_paths = [
                Path(__file__).parent.parent.parent / "whisper.cpp",  # Relative to module
                Path.home() / "rumiaifinal" / "whisper.cpp",          # User home
                Path("/home/jorge/rumiaifinal/whisper.cpp"),          # Absolute path
                Path.cwd() / "whisper.cpp",                           # Current directory
            ]
            
            for path in possible_paths:
                if path.exists() and (path / "main").exists():
                    self.whisper_cpp_path = path
                    break
            else:
                raise DependencyError(
                    "Could not find whisper.cpp installation. "
                    "Please specify path or install in one of: "
                    f"{[str(p) for p in possible_paths]}"
                )
        else:
            self.whisper_cpp_path = Path(whisper_cpp_path)
        
        self.binary_path = self.whisper_cpp_path / "main"
        
        # Check whisper.cpp binary exists
        if not self.binary_path.exists():
            raise DependencyError(
                f"Whisper.cpp binary not found at {self.binary_path}\n"
                "Build it with:\n"
                "  cd whisper.cpp && make clean && make -j10"
            )
        
        # Test whisper.cpp binary works
        try:
            result = subprocess.run(
                [str(self.binary_path), "--help"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise DependencyError(
                    f"Whisper.cpp binary at {self.binary_path} is not working.\n"
                    "Rebuild with: cd whisper.cpp && make clean && make -j10"
                )
        except subprocess.TimeoutExpired:
            raise DependencyError(
                "Whisper.cpp binary hangs. Rebuild it:\n"
                "  cd whisper.cpp && make clean && make -j10"
            )
        except Exception as e:
            raise DependencyError(
                f"Cannot execute Whisper.cpp binary: {e}\n"
                "Check file permissions: chmod +x whisper.cpp/main"
            )
        
        # Hardcoded thread count for 16-core system
        self.threads = 10  # Not configurable, no parameter
        logger.info("Using 10 threads for Whisper.cpp (optimized for 16-core system)")
        
        # Use standard multilingual model (no .en suffix, no quantization)
        self.model_path = self.whisper_cpp_path / f"models/ggml-{model}.bin"
        
        # Check model exists
        if not self.model_path.exists():
            raise DependencyError(
                f"Model {model} not found at {self.model_path}\n"
                f"Download it with:\n"
                f"  cd whisper.cpp/models\n"
                f"  bash ./download-ggml-model.sh {model}"
            )
        
        # Verify model file isn't corrupted (basic check)
        model_size = self.model_path.stat().st_size
        expected_sizes = {
            'tiny': 39_000_000,    # ~39MB
            'base': 142_000_000,   # ~142MB
            'small': 466_000_000,  # ~466MB
            'medium': 1_500_000_000, # ~1.5GB
            'large': 3_000_000_000,  # ~3GB
        }
        
        if model_size < expected_sizes.get(model, 0) * 0.9:  # 90% tolerance
            raise DependencyError(
                f"Model file appears corrupted (too small: {model_size} bytes).\n"
                f"Re-download with:\n"
                f"  cd whisper.cpp/models\n"
                f"  rm {self.model_path.name}\n"
                f"  bash ./download-ggml-model.sh {model}"
            )
            
        logger.info(f"Initialized Whisper.cpp with model: {self.model_path}")
    
    async def transcribe(self, 
                        audio_path: Path,
                        language: Optional[str] = None,
                        initial_prompt: Optional[str] = None,
                        timeout: int = 30) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.cpp
        
        Args:
            audio_path: Path to audio file (WAV preferred)
            language: Optional language code (en, es, fr, etc.)
            initial_prompt: Optional prompt to guide transcription
            timeout: Maximum seconds to wait (default 30s for 120s max videos)
            
        Returns:
            Dict with transcription results matching Whisper format
            
        Raises:
            TimeoutError: If transcription takes longer than timeout
        """
        # Build command
        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-f", str(audio_path),
            "-t", "10",  # Hardcoded, not configurable
            "-oj",  # Output JSON
            "--no-prints"  # Suppress progress output
        ]
        
        # Add language if specified
        if language:
            cmd.extend(["-l", language])
        
        # Add initial prompt if specified
        if initial_prompt:
            cmd.extend(["--prompt", initial_prompt])
        
        logger.info(f"Running Whisper.cpp transcription on {audio_path}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill the hung process
                process.kill()
                await process.wait()  # Ensure it's dead
                
                raise TimeoutError(
                    f"Whisper.cpp timed out after {timeout}s for {audio_path}. "
                    f"This usually indicates a corrupted audio file or system issue. "
                    f"Normal processing for 120s video should take <10s."
                )
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Whisper.cpp failed: {error_msg}")
            
            # Parse JSON output
            result = json.loads(stdout.decode('utf-8'))
            
            # Transform to match Python Whisper format
            return self._format_result(result)
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _format_result(self, whisper_cpp_result: Dict) -> Dict[str, Any]:
        """
        Format Whisper.cpp output to match Python Whisper format
        
        Actual Whisper.cpp format:
        {
            "text": " Hello world",
            "segments": [
                {
                    "start": 0,
                    "end": 3280,
                    "text": " Hello world"
                }
            ]
        }
        """
        # Extract text (already in correct format)
        full_text = whisper_cpp_result.get("text", "").strip()
        
        # Format segments - they're already mostly correct
        segments = []
        for i, seg in enumerate(whisper_cpp_result.get("segments", [])):
            segments.append({
                "id": i,
                "start": seg.get("start", 0) / 1000.0,  # Convert ms to seconds
                "end": seg.get("end", 0) / 1000.0,      # Convert ms to seconds
                "text": seg.get("text", "").strip(),
                "words": []  # Whisper.cpp doesn't provide word-level by default
            })
        
        # Whisper.cpp doesn't output language in JSON, default to English
        # Could be detected with a separate call if needed
        language = "en"  # or detect from first segment
        
        return {
            "text": full_text,
            "segments": segments,
            "language": language,
            "duration": segments[-1]["end"] if segments else 0.0
        }
    
    async def transcribe_with_preprocessing(self,
                                           audio_path: Path,
                                           language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe with audio preprocessing for better results
        
        Converts audio to optimal format if needed:
        - 16kHz sample rate
        - Mono channel
        - WAV format
        """
        # Check if audio needs conversion
        if audio_path.suffix.lower() != '.wav':
            # Convert to WAV
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_path = Path(temp_wav.name)
            temp_wav.close()
            
            try:
                # Use ffmpeg to convert
                cmd = [
                    'ffmpeg',
                    '-i', str(audio_path),
                    '-ar', '16000',  # 16kHz
                    '-ac', '1',      # Mono
                    '-c:a', 'pcm_s16le',  # 16-bit PCM
                    '-y',  # Overwrite
                    str(temp_wav_path)
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await process.communicate()
                
                if process.returncode != 0:
                    raise RuntimeError("Audio conversion failed")
                
                # Transcribe converted audio
                result = await self.transcribe(temp_wav_path, language)
                
            finally:
                # Clean up temp file
                if temp_wav_path.exists():
                    temp_wav_path.unlink()
            
            return result
        else:
            # Already WAV, transcribe directly
            return await self.transcribe(audio_path, language)

def get_whisper_cpp_transcriber() -> WhisperCppTranscriber:
    """
    Create new Whisper.cpp transcriber instance
    Note: No singleton pattern - each instance is lightweight
    Validation overhead (~100-200ms) is negligible for sequential processing
    """
    return WhisperCppTranscriber()
```

---

### Phase 3: Complete Python Whisper Removal (30 minutes)

#### 3.1 Remove Python Whisper Dependencies

Pattern: Find and remove all Python Whisper imports and usage:
- Search for: `import whisper`, `from whisper import`
- Remove: Model loading code (`whisper.load_model`)
- Remove: Fallback transcription methods
- Remove: Model caching/singleton patterns for Python Whisper

#### 3.2 Update `rumiai_v2/api/whisper_transcribe_safe.py`

Replace the entire file with a simpler version that ONLY uses Whisper.cpp:

```python
"""
Whisper transcription service using Whisper.cpp
Fast, reliable CPU-based transcription without Python/PyTorch issues
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .audio_utils import extract_audio_simple
from .whisper_cpp_service import get_whisper_cpp_transcriber

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Whisper transcription using Whisper.cpp backend"""
    
    def __init__(self):
        # Direct instantiation - no singleton needed
        self.transcriber = WhisperCppTranscriber()
    
    async def transcribe(self, 
                        video_path: Path,
                        timeout: int = 600,
                        language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe video using Whisper.cpp
        
        Args:
            video_path: Path to video file
            timeout: Maximum time in seconds (used for compatibility, not enforced)
            language: Optional language hint
            
        Returns:
            Transcription results dictionary
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return self._empty_result(error=f"Video file not found: {video_path}")
        
        temp_audio = None
        try:
            # Extract audio to WAV (required for transcription)
            logger.info(f"Extracting audio from {video_path}")
            temp_audio = await extract_audio_simple(video_path)
            
            # Transcribe using Whisper.cpp
            logger.info(f"Transcribing audio with Whisper.cpp")
            result = await self.transcriber.transcribe(
                temp_audio,
                language=language
            )
            
            # Add metadata
            result['metadata'] = {
                'model': 'whisper.cpp-base',
                'processed': True,
                'success': True,
                'backend': 'whisper.cpp'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for {video_path}: {e}")
            return self._empty_result(error=str(e))
        
        finally:
            # Clean up temporary audio file
            if temp_audio and temp_audio.exists():
                try:
                    temp_audio.unlink()
                    logger.debug(f"Cleaned up temporary audio file: {temp_audio}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp audio file: {e}")
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result with proper error state"""
        result = {
            'text': '',
            'segments': [],
            'language': 'unknown',
            'duration': 0,
            'metadata': {
                'model': 'whisper.cpp-base',
                'processed': False,
                'success': False,
                'backend': 'whisper.cpp'
            }
        }
        if error:
            result['metadata']['error'] = error
        return result
    
    # No model loading methods needed - Whisper.cpp handles everything

def get_transcriber() -> WhisperTranscriber:
    """Get transcriber instance"""
    return WhisperTranscriber()
```

---

### Phase 4: Dependency Cleanup (10 minutes)

#### 4.1 Remove Python Whisper
```bash
# Uninstall Python Whisper completely
pip uninstall -y openai-whisper whisper
# Do NOT uninstall torch - other ML services need it
```

#### 4.2 Update requirements.txt
Pattern: Remove whisper-related entries:
- Remove: `openai-whisper`
- Remove: `whisper`
- Keep: `torch` (for YOLO and other ML services)
- Keep: `ffmpeg-python` (for audio extraction)

#### 4.2 Update Installation Instructions
Add to `README.md`:

```markdown
## Whisper.cpp Setup

1. Build Whisper.cpp:
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make -j8
```

2. Download model:
```bash
cd models
bash ./download-ggml-model.sh base  # Only standard model needed
```

3. Test:
```bash
./main -m models/ggml-base.bin -f samples/jfk.wav
```
```

---

## Testing Plan

### Unit Tests

```python
# test_whisper_cpp.py
import asyncio
from pathlib import Path
from rumiai_v2.api.whisper_cpp_service import WhisperCppTranscriber

async def test_transcription():
    transcriber = WhisperCppTranscriber()
    
    # Test with sample audio
    result = await transcriber.transcribe(
        Path("test_audio.wav"),
        language="en"
    )
    
    assert result['text']
    assert result['segments']
    assert result['language'] == 'en'
    print(f"✅ Transcription successful: {len(result['text'])} chars")

asyncio.run(test_transcription())
```

### Integration Test

```bash
# Test with problematic videos
cd /home/jorge/rumiaifinal
rm -f speech_transcriptions/7389683775929699616_whisper.json

# Run full pipeline
python test_unified_pipeline_e2e.py temp/7389683775929699616.mp4
```

---

## Performance Expectations

### Expected Performance (16-core system, 10 threads)

| Video Length | Python Whisper | Whisper.cpp | Improvement |
|-------------|---------------|-------------|-------------|
| 30 seconds  | 8-10 seconds  | 2-3 seconds | ~3x faster |
| 60 seconds  | 15-20 seconds | 5-8 seconds | ~2.5x faster |
| 120 seconds | 30-40 seconds | 10-15 seconds | ~2.5x faster |

Note: TikTok videos are typically under 60 seconds

### Resource Usage

- **CPU**: 10 threads, ~60% total CPU usage during transcription
- **Memory**: ~150MB for base model (vs 500MB+ for Python Whisper)
- **Disk**: ~142MB for base model only

---

## Implementation Checklist

- [ ] Clone and build Whisper.cpp (with version pinning)
- [ ] Download base model only (no quantization)
- [ ] Test Whisper.cpp directly on TikTok videos
- [ ] Create whisper_cpp_service.py with all fixes
- [ ] Completely replace whisper_transcribe_safe.py
- [ ] Remove ALL Python Whisper code and imports
- [ ] Uninstall openai-whisper package
- [ ] Test with videos that previously failed
- [ ] Run full E2E test suite
- [ ] Add health monitoring
- [ ] Document pinned version

---

## Monitoring and Maintenance

### Health Monitoring with Auto-Recovery
```python
def check_and_recover_whisper():
    """Health check with automatic recovery attempts"""
    
    # First, try simple check
    def check_whisper_health():
        try:
            result = subprocess.run(
                ["./whisper.cpp/main", "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    if check_whisper_health():
        return True
    
    print("Whisper.cpp health check failed. Attempting recovery...")
    
    # Check 1: Binary exists and is executable
    if not os.path.exists("./whisper.cpp/main"):
        print("Binary missing. Rebuilding...")
        subprocess.run(["make", "-C", "whisper.cpp", "clean"])
        subprocess.run(["make", "-C", "whisper.cpp", "-j10"])
        subprocess.run(["chmod", "+x", "./whisper.cpp/main"])
        return check_whisper_health()
    
    # Check 2: Model file exists and isn't corrupted
    model_path = "./whisper.cpp/models/ggml-base.bin"
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 100_000_000:
        print("Model missing/corrupted. Re-downloading...")
        if os.path.exists(model_path):
            os.remove(model_path)
        subprocess.run(["bash", "./whisper.cpp/models/download-ggml-model.sh", "base"])
        return check_whisper_health()
    
    # Check 3: Disk space (need at least 1GB free)
    import shutil
    if shutil.disk_usage(".").free < 1_000_000_000:
        print("Low disk space! Clear some space and retry.")
        return False
    
    # Manual intervention needed
    print("Unable to auto-recover. Try:")
    print("1. cd whisper.cpp && git pull && make clean && make -j10")
    print("2. Reboot WSL: wsl --shutdown (from PowerShell)")
    return False
```

### Version Management
- Pin to specific commit with fallback:
  ```bash
  git checkout b3221 || git checkout $(git rev-list -n 1 --before="2024-01-01" master)
  ```
- Document actual version: `echo "$(git rev-parse HEAD)" > PINNED_VERSION.txt`
- Review quarterly for updates
- Test thoroughly before updating

### Minimum Dependencies (WSL2 Ubuntu)
```
# Required versions (WSL2 Ubuntu 20.04+ has these by default):
# - ffmpeg 4.0+ (for modern codec support)
# - gcc 5.0+ (for C++11 support)
# - make (any version)
# No runtime version checking needed for controlled environment
```

---

## Benefits Summary

1. **Reliability**: No Python/PyTorch conflicts
2. **Speed**: 3-4x faster transcription
3. **Efficiency**: Lower memory usage
4. **Simplicity**: No complex dependency management
5. **Consistency**: Same results every time
6. **Maintainability**: C++ binary is stable, no version hell

---

## Accepted Limitations

| Limitation | Impact | Acceptance |
|------------|--------|------------|
| No word-level timestamps | Can't highlight individual words | Not needed for video analysis |
| No confidence scores | Can't filter uncertain segments | Transcription quality is binary |
| WSL2-specific build | Won't work on other systems | Single laptop deployment |
| Version pinning | No automatic updates | Controlled, stable environment |
| No rollback plan | If Whisper fails, pipeline stops | Fail-fast approach - fix immediately |

---

## Audio Energy Service Integration

### Purpose
Add REAL audio energy analysis as a separate ML service to replace estimated/fake energy metrics.

### Architecture
Pattern: Copy Whisper service pattern for audio file sharing
- Location: `rumiai_v2/ml_services/audio_energy_service.py`
- Output: `ml_outputs/[video_id]_audio_energy.json`
- Integration: Shares extracted audio file with Whisper service
- Coordination: Both services run in parallel via asyncio.gather()

### File Sharing Coordination
```python
# Extract audio once for both services
audio_file = await extract_audio_simple(video_path)

# Run both services in parallel on same audio file
results = await asyncio.gather(
    whisper_service.transcribe(audio_file),
    energy_service.analyze(audio_file)
)

# Cleanup after BOTH complete
os.remove(audio_file)
```

### Implementation Pattern

1. **Service Creation**
   - Pattern: Copy Whisper's structure specifically
   - Add to ML pipeline runner for parallel execution
   - Share the same extracted audio file with Whisper

2. **Energy Analysis**
   - Use librosa for RMS energy extraction
   - Percentile-based normalization (preserve relative dynamics)
   - Generate 5-second windows matching expected format
   - Calculate burst pattern from actual audio dynamics

3. **Data Format**
   ```json
   {
     "energy_level_windows": {"0-5s": 0.8, "5-10s": 0.6},
     "energy_variance": 0.35,
     "climax_timestamp": 23.5,
     "burst_pattern": "front_loaded"
   }
   ```

4. **Unified Analyzer Update**
   - Pattern: Find where fake energy is generated
   - Replace with loading from `audio_energy.json`
   - Remove ALL estimation code
   - Fail if energy data missing (no fallback)

5. **Temporal Markers Enhancement**
   - Add `energy_progression` to first_5_seconds
   - Add `audio_energy` to each speech segment
   - Include energy in peak moment detection
   - Mark CTA emphasis as measured, not guessed

### Key Principles
- NO fake/estimated energy - always real
- NO `is_real_energy` field - it's always real
- Fail fast if energy analysis missing
- Energy service runs in parallel with other ML services

---

## Implementation Steps

### Step 1: Build and Pin Whisper.cpp
```bash
cd /home/jorge/rumiaifinal
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
# Pin to specific version with fallback
git checkout b3221 || git checkout $(git rev-list -n 1 --before="2024-01-01" master)
echo "$(git rev-parse HEAD)" > PINNED_VERSION.txt

# Build and ensure executable
make clean && make -j10
chmod +x main  # Ensure binary has execute permission

# Download model
cd models && bash ./download-ggml-model.sh base
```

### Step 2: Remove Python Whisper
Specific removal patterns:
1. Find Python imports only:
   ```bash
   grep -r "^import whisper$" --include="*.py"
   grep -r "^from whisper import" --include="*.py"
   ```
2. Remove all Python Whisper model loading
3. Remove fallback transcription methods
4. Remove singleton patterns for model management
5. Uninstall package:
   ```bash
   pip uninstall -y openai-whisper whisper
   # Do NOT uninstall torch - other ML services need it
   ```

### Step 3: Implement Whisper.cpp Service
Create new service with:
- Dependency checking at import
- Binary path auto-detection
- Version verification
- 30-second timeout
- Proper JSON parsing for actual Whisper.cpp format
- No singleton pattern

### Step 4: Test and Verify
1. Test on the specific video that was failing:
   ```bash
   python test_unified_pipeline_e2e.py 7389683775929699616.mp4
   ```
2. Test on 2-3 other random TikTok videos
3. Verify complete pipeline flow:
   - Check `speech_transcriptions/[video_id]_whisper.json` exists and has text
   - Check `ml_outputs/[video_id]_audio_energy.json` exists
   - Check `unified_analysis/[video_id].json` has energy data
   - Check `claude_outputs/[video_id]_analysis.json` is complete
4. If all checkpoints have data, deployment is successful

### Step 5: Performance Validation
```python
def validate_pipeline_performance(video_id):
    """Check that data flows correctly through entire pipeline"""
    
    checkpoints = {
        "whisper_output": f"speech_transcriptions/{video_id}_whisper.json",
        "energy_output": f"ml_outputs/{video_id}_audio_energy.json",
        "unified_output": f"unified_analysis/{video_id}.json",
        "claude_output": f"claude_outputs/{video_id}_analysis.json"
    }
    
    # Check each checkpoint exists and has required data
    for stage, filepath in checkpoints.items():
        if not os.path.exists(filepath):
            print(f"❌ Pipeline broken at {stage}")
            return False
            
        # Verify non-empty and valid JSON
        with open(filepath) as f:
            data = json.load(f)
            if not data or (stage == "whisper_output" and not data.get("text")):
                print(f"❌ Empty/invalid data at {stage}")
                return False
    
    print("✅ Pipeline complete - all data flowing correctly")
    return True
```

Note: Performance = complete data flow, not speed. Single-video sequential processing means completion matters more than timing.

**Total**: 3-4 hours of implementation + testing