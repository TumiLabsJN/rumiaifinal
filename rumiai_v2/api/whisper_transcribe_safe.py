"""
Whisper transcription service using Whisper.cpp
Fast, reliable CPU-based transcription without Python/PyTorch issues
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .audio_utils import extract_audio_simple
from .whisper_cpp_service import WhisperCppTranscriber

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