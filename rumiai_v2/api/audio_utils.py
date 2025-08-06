"""
Simple audio extraction utility for Whisper
"""

import asyncio
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
        
        logger.info(f"Extracting audio from {video_path} to {temp_audio_path}")
        
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {stderr.decode()}")
            raise Exception(f"ffmpeg extraction failed: {stderr.decode()[:200]}")
        
        # Verify file was created and has size
        if not temp_audio_path.exists():
            raise Exception("Audio file was not created")
        
        file_size = temp_audio_path.stat().st_size
        if file_size == 0:
            raise Exception("Audio file is empty")
            
        logger.info(f"Successfully extracted audio: {file_size / 1024 / 1024:.1f} MB")
        return temp_audio_path
        
    except Exception as e:
        # Clean up on failure
        if temp_audio_path.exists():
            temp_audio_path.unlink()
        raise