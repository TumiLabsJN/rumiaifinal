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
WHISPER_CPP_VERSION = "f9ca90256bf691642407e589db1a36562c461db7"  # Update only after testing
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
        
        # Hardcoded thread count - use all 10 cores allocated to WSL2
        self.threads = 10  # Not configurable, no parameter
        logger.info("Using 10 threads for Whisper.cpp (WSL2 allocated cores)")
        
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
                        timeout: int = 120) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.cpp
        
        Args:
            audio_path: Path to audio file (WAV preferred)
            language: Optional language code (en, es, fr, etc.)
            initial_prompt: Optional prompt to guide transcription
            timeout: Maximum seconds to wait (default 120s for long videos)
            
        Returns:
            Dict with transcription results matching Whisper format
            
        Raises:
            TimeoutError: If transcription takes longer than timeout
        """
        # Build command with speed optimizations
        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-f", str(audio_path),
            "-t", "10",  # Use all 10 WSL2 cores
            "-bo", "1",  # Greedy decoding for maximum speed
            "-bs", "1",  # Greedy decoding (no beam search)
            "-oj",  # Output JSON to file
        ]
        
        # Add language if specified
        if language:
            cmd.extend(["-l", language])
        
        # Add initial prompt if specified
        if initial_prompt:
            cmd.extend(["--prompt", initial_prompt])
        
        logger.info(f"Running Whisper.cpp transcription on {audio_path}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Determine the output JSON file path
        json_output_path = Path(str(audio_path) + ".json")
        
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
                    f"Consider using a smaller model (tiny) or checking system resources."
                )
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Whisper.cpp failed: {error_msg}")
            
            # Read the generated JSON file
            if not json_output_path.exists():
                raise RuntimeError(f"Whisper.cpp did not generate output file: {json_output_path}")
            
            with open(json_output_path, 'r') as f:
                result = json.load(f)
            
            # Clean up the JSON file
            try:
                json_output_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors
            
            # Transform to match Python Whisper format
            return self._format_result(result)
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _format_result(self, whisper_cpp_result: Dict) -> Dict[str, Any]:
        """
        Format Whisper.cpp output to match Python Whisper format
        
        Actual Whisper.cpp JSON format:
        {
            "result": {"language": "en"},
            "transcription": [
                {
                    "offsets": {"from": 0, "to": 2580},
                    "text": " (chiming)"
                }
            ]
        }
        """
        # Extract transcription segments
        transcription = whisper_cpp_result.get("transcription", [])
        
        # Build full text from all segments
        full_text = " ".join(seg.get("text", "").strip() for seg in transcription)
        
        # Format segments
        segments = []
        for i, seg in enumerate(transcription):
            offsets = seg.get("offsets", {})
            segments.append({
                "id": i,
                "start": offsets.get("from", 0) / 1000.0,  # Convert ms to seconds
                "end": offsets.get("to", 0) / 1000.0,      # Convert ms to seconds
                "text": seg.get("text", "").strip(),
                "words": []  # Whisper.cpp doesn't provide word-level by default
            })
        
        # Extract language from result
        language = whisper_cpp_result.get("result", {}).get("language", "en")
        
        return {
            "text": full_text,
            "segments": segments,
            "language": language,
            "duration": segments[-1]["end"] if segments else 0.0
        }
    
    async def transcribe_with_preprocessing(self,
                                           audio_path: Path,
                                           language: Optional[str] = None,
                                           video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe with audio preprocessing for better results
        
        Converts audio to optimal format if needed:
        - 16kHz sample rate
        - Mono channel
        - WAV format
        
        Args:
            audio_path: Path to audio file or video file
            language: Optional language code (en, es, fr, etc.)
            video_id: Optional video ID for shared extraction caching
        """
        # Use SharedAudioExtractor if video_id provided (indicates video processing)
        if video_id:
            from .shared_audio_extractor import SharedAudioExtractor
            
            logger.info(f"Using SharedAudioExtractor for video {video_id}")
            # Get the shared audio file (will extract only once across all services)
            audio_path = await SharedAudioExtractor.extract_once(
                str(audio_path), 
                video_id, 
                service_name="whisper_cpp"
            )
            # SharedAudioExtractor always returns WAV format, so directly transcribe
            return await self.transcribe(audio_path, language)
        
        # Legacy path for non-video audio files
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