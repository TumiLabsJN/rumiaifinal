"""
Safe Whisper transcription with async support and timeout protection
"""

import asyncio
import logging
import whisper
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Async-native Whisper transcription with singleton model management"""
    
    _instance = None
    _model = None
    _model_size = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def load_model(self, model_size: str = "base"):
        """Load model once, reuse for all transcriptions"""
        if self._model is None or self._model_size != model_size:
            logger.info(f"Loading Whisper model: {model_size}")
            try:
                # Load model asynchronously
                self._model = await asyncio.to_thread(
                    whisper.load_model, model_size
                )
                self._model_size = model_size
                logger.info(f"Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
        return self._model
    
    async def transcribe(self, video_path: Path, 
                        timeout: int = 600,
                        language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe video with timeout protection
        
        Args:
            video_path: Path to video file
            timeout: Maximum time in seconds (default 10 minutes)
            language: Optional language hint
            
        Returns:
            Transcription results dictionary
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return self._empty_result()
        
        try:
            # Ensure model is loaded
            model = await self.load_model()
            
            # Run transcription with timeout
            async with asyncio.timeout(timeout):
                logger.info(f"Starting transcription of {video_path}")
                
                result = await asyncio.to_thread(
                    model.transcribe,
                    str(video_path),
                    language=language,
                    word_timestamps=True,
                    fp16=False  # Avoid FP16 issues on CPU
                )
                
                # Format segments
                segments = []
                for seg in result.get('segments', []):
                    segment = {
                        'id': seg.get('id', 0),
                        'start': float(seg.get('start', 0)),
                        'end': float(seg.get('end', 0)),
                        'text': str(seg.get('text', '')).strip(),
                        'words': seg.get('words', [])
                    }
                    segments.append(segment)
                
                return {
                    'text': str(result.get('text', '')).strip(),
                    'segments': segments,
                    'language': str(result.get('language', 'unknown')),
                    'duration': float(result.get('duration', 0)),
                    'metadata': {
                        'model': self._model_size,
                        'processed': True
                    }
                }
            
        except asyncio.TimeoutError:
            logger.error(f"Transcription timed out after {timeout}s for {video_path}")
            return self._empty_result(error=f"Timeout after {timeout}s")
            
        except Exception as e:
            logger.error(f"Transcription failed for {video_path}: {e}")
            return self._empty_result(error=str(e))
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result structure"""
        result = {
            'text': '',
            'segments': [],
            'language': 'unknown',
            'duration': 0,
            'metadata': {
                'model': self._model_size or 'base',
                'processed': False
            }
        }
        if error:
            result['error'] = error
        return result

def get_transcriber() -> WhisperTranscriber:
    """Get singleton transcriber instance"""
    return WhisperTranscriber()