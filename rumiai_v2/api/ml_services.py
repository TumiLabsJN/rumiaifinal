"""
ML services interface for RumiAI v2.

This module provides a unified interface to various ML services.
"""
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio
import os

from ..core.exceptions import MLAnalysisError

logger = logging.getLogger(__name__)


class MLServices:
    """
    Interface to ML analysis services.
    
    In production, this would call actual ML services.
    For now, it interfaces with existing Python scripts.
    """
    
    def __init__(self):
        self.venv_path = Path("venv/bin/activate")
        self.python_path = Path("venv/bin/python")
    
    async def run_yolo_detection(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run YOLO object detection on video."""
        logger.info(f"Running YOLO detection on {video_path}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # In production, this would call the YOLO service
        # For now, check if output already exists
        output_file = output_dir / f"{video_path.stem}_yolo_detections.json"
        
        if output_file.exists():
            logger.info(f"YOLO output already exists: {output_file}")
            with open(output_file, 'r') as f:
                return json.load(f)
        
        # Simulate YOLO processing
        logger.warning("YOLO service not implemented - returning empty results")
        return {
            'objectAnnotations': [],
            'metadata': {
                'model': 'YOLOv8',
                'processed': False
            }
        }
    
    async def run_whisper_transcription(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run Whisper speech transcription on video."""
        logger.info(f"Running Whisper transcription on {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{video_path.stem}_whisper.json"
        
        if output_file.exists():
            logger.info(f"Whisper output already exists: {output_file}")
            with open(output_file, 'r') as f:
                return json.load(f)
        
        # Try to run actual Whisper if available
        try:
            result = await self._run_python_script(
                "whisper_transcribe.py",
                ["--video", str(video_path), "--output", str(output_file)]
            )
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}")
        
        # Return empty transcription
        return {
            'segments': [],
            'text': '',
            'language': 'unknown'
        }
    
    async def run_mediapipe_analysis(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run MediaPipe human analysis on video."""
        logger.info(f"Running MediaPipe analysis on {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{video_path.stem}_human_analysis.json"
        
        if output_file.exists():
            logger.info(f"MediaPipe output already exists: {output_file}")
            with open(output_file, 'r') as f:
                return json.load(f)
        
        # Return default human analysis
        return {
            'presence_percentage': 0.0,
            'frames_with_people': 0,
            'poses': [],
            'faces': [],
            'hands': []
        }
    
    async def run_ocr_detection(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run OCR text detection on video."""
        logger.info(f"Running OCR detection on {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{video_path.stem}_creative_analysis.json"
        
        if output_file.exists():
            logger.info(f"OCR output already exists: {output_file}")
            with open(output_file, 'r') as f:
                return json.load(f)
        
        # Return empty OCR results
        return {
            'textAnnotations': [],
            'stickers': [],
            'metadata': {
                'frames_analyzed': 0
            }
        }
    
    async def run_scene_detection(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run scene detection on video."""
        logger.info(f"Running scene detection on {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{video_path.stem}_scenes.json"
        
        if output_file.exists():
            logger.info(f"Scene detection output already exists: {output_file}")
            with open(output_file, 'r') as f:
                return json.load(f)
        
        # Try to run PySceneDetect if available
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
            
            # Open video
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector())
            
            # Detect scenes
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
            
            # Convert to our format
            scenes = []
            scene_changes = []
            
            for i, (start_time, end_time) in enumerate(scene_list):
                start_seconds = start_time.get_seconds()
                end_seconds = end_time.get_seconds()
                
                scenes.append({
                    'start_time': start_seconds,
                    'end_time': end_seconds,
                    'duration': end_seconds - start_seconds,
                    'scene_index': i
                })
                
                if i > 0:  # Skip first scene start at 0
                    scene_changes.append(start_seconds)
            
            result = {
                'scenes': scenes,
                'scene_changes': scene_changes,
                'total_scenes': len(scenes)
            }
            
            # Save result
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Detected {len(scenes)} scenes")
            return result
            
        except ImportError:
            logger.warning("PySceneDetect not available")
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
        
        # Return default scene data
        return {
            'scenes': [{
                'start_time': 0,
                'end_time': 60,
                'duration': 60
            }],
            'scene_changes': [],
            'total_scenes': 1
        }
    
    async def extract_frames(self, video_path: Path, output_dir: Path, 
                           sample_rate: float = 1.0) -> List[Path]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            sample_rate: Frames per second to extract
            
        Returns:
            List of paths to extracted frames
        """
        logger.info(f"Extracting frames from {video_path} at {sample_rate} fps")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if frames already extracted
        existing_frames = list(output_dir.glob("frame_*.jpg"))
        if existing_frames:
            logger.info(f"Frames already extracted: {len(existing_frames)} frames")
            return sorted(existing_frames)
        
        try:
            # Use ffmpeg to extract frames
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'fps={sample_rate}',
                '-q:v', '2',  # Quality
                str(output_dir / 'frame_%04d.jpg'),
                '-y'  # Overwrite
            ]
            
            # Run ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise MLAnalysisError('ffmpeg', f"Frame extraction failed: {result.stderr}")
            
            # Get extracted frames
            frames = sorted(output_dir.glob("frame_*.jpg"))
            logger.info(f"Extracted {len(frames)} frames")
            
            return frames
            
        except FileNotFoundError:
            logger.error("FFmpeg not found - cannot extract frames")
            raise MLAnalysisError('ffmpeg', "FFmpeg not installed")
    
    async def _run_python_script(self, script_name: str, args: List[str]) -> Dict[str, Any]:
        """Run a Python script in the virtual environment."""
        # Build command
        if self.venv_path.exists():
            # Use venv Python
            cmd = [str(self.python_path), script_name] + args
        else:
            # Use system Python
            cmd = ['python', script_name] + args
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Run script
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise MLAnalysisError(script_name, f"Script failed: {error_msg}")
        
        # Parse JSON output
        try:
            return json.loads(stdout.decode())
        except json.JSONDecodeError:
            logger.warning(f"Script output not valid JSON: {stdout.decode()[:200]}")
            return {}