"""
Video analyzer orchestrator for RumiAI v2.

This module orchestrates ML analysis of videos.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
import logging
import json

from ..core.models import MLAnalysisResult
from ..core.exceptions import MLAnalysisError

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Orchestrate ML analysis of videos."""
    
    def __init__(self, ml_services):
        """
        Initialize with ML services.
        
        Args:
            ml_services: ML service client for running analyses
        """
        self.ml_services = ml_services
    
    async def analyze_video(self, video_id: str, video_path: Path) -> Dict[str, MLAnalysisResult]:
        """
        Run all ML analyses on video.
        
        Returns dictionary of model_name -> MLAnalysisResult
        """
        logger.info(f"Starting ML analysis for video {video_id}")
        
        # Define all ML analyses to run
        analyses = {
            'yolo': self._run_yolo,
            'whisper': self._run_whisper,
            'mediapipe': self._run_mediapipe,
            'ocr': self._run_ocr,
            'scene_detection': self._run_scene_detection
        }
        
        # Run analyses in parallel
        tasks = {}
        for model_name, analysis_func in analyses.items():
            logger.info(f"Scheduling {model_name} analysis")
            tasks[model_name] = asyncio.create_task(
                analysis_func(video_id, video_path)
            )
        
        # Wait for all to complete
        results = {}
        for model_name, task in tasks.items():
            try:
                result = await task
                results[model_name] = result
                logger.info(f"{model_name} analysis completed (success={result.success})")
            except Exception as e:
                logger.error(f"{model_name} analysis failed with exception: {e}")
                results[model_name] = MLAnalysisResult(
                    model_name=model_name,
                    model_version='unknown',
                    success=False,
                    error=str(e)
                )
        
        return results
    
    async def _run_yolo(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run YOLO object detection."""
        try:
            # First check if output already exists
            output_dir = Path(f"object_detection_outputs/{video_id}")
            output_path = output_dir / f"{video_id}_yolo_detections.json"
            
            if output_path.exists():
                logger.info(f"Using existing YOLO output: {output_path}")
                with open(output_path, 'r') as f:
                    data = json.load(f)
                
                return MLAnalysisResult(
                    model_name='yolo',
                    model_version='v8',
                    success=True,
                    data=data,
                    processing_time=0.0
                )
            
            # Run actual YOLO detection via ML services
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
    
    async def _run_whisper(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run Whisper speech transcription."""
        try:
            # Check for existing transcription
            output_dir = Path("speech_transcriptions")
            output_path = output_dir / f"{video_id}_whisper.json"
            
            if output_path.exists():
                logger.info(f"Using existing Whisper output: {output_path}")
                with open(output_path, 'r') as f:
                    data = json.load(f)
                
                return MLAnalysisResult(
                    model_name='whisper',
                    model_version='base',
                    success=True,
                    data=data,
                    processing_time=0.0
                )
            
            # Run actual Whisper transcription
            logger.info(f"Running Whisper transcription on {video_path}")
            data = await self.ml_services.run_whisper_transcription(video_path, output_dir)
            
            # Save the output
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return MLAnalysisResult(
                model_name='whisper',
                model_version='base',
                success=True,
                data=data,
                processing_time=0.0
            )
                
        except Exception as e:
            return MLAnalysisResult(
                model_name='whisper',
                model_version='base',
                success=False,
                error=str(e)
            )
    
    async def _run_mediapipe(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run MediaPipe human analysis."""
        try:
            # Check for existing analysis
            output_dir = Path(f"human_analysis_outputs/{video_id}")
            output_path = output_dir / f"{video_id}_human_analysis.json"
            
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
            
            # Run actual MediaPipe analysis
            logger.info(f"Running MediaPipe analysis on {video_path}")
            data = await self.ml_services.run_mediapipe_analysis(video_path, output_dir)
            
            # Save the output
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
    
    async def _run_ocr(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run OCR text detection."""
        try:
            # Check for existing OCR results
            output_dir = Path(f"creative_analysis_outputs/{video_id}")
            output_path = output_dir / f"{video_id}_creative_analysis.json"
            
            if output_path.exists():
                logger.info(f"Using existing OCR output: {output_path}")
                with open(output_path, 'r') as f:
                    data = json.load(f)
                
                return MLAnalysisResult(
                    model_name='ocr',
                    model_version='tesseract-5',
                    success=True,
                    data=data,
                    processing_time=0.0
                )
            
            # Run actual OCR detection
            logger.info(f"Running OCR on {video_path}")
            data = await self.ml_services.run_ocr_detection(video_path, output_dir)
            
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
    
    async def _run_scene_detection(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run scene detection."""
        try:
            # Check for existing scene detection
            output_dir = Path(f"scene_detection_outputs/{video_id}")
            output_path = output_dir / f"{video_id}_scenes.json"
            
            if output_path.exists():
                logger.info(f"Using existing scene detection output: {output_path}")
                with open(output_path, 'r') as f:
                    data = json.load(f)
                
                return MLAnalysisResult(
                    model_name='scene_detection',
                    model_version='pyscenedetect-0.6',
                    success=True,
                    data=data,
                    processing_time=0.0
                )
            
            # Run actual scene detection
            logger.info(f"Running scene detection on {video_path}")
            data = await self.ml_services.run_scene_detection(video_path, output_dir)
            
            # Save the output
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return MLAnalysisResult(
                model_name='scene_detection',
                model_version='pyscenedetect-0.6',
                success=True,
                data=data,
                processing_time=0.0
            )
                
        except Exception as e:
            return MLAnalysisResult(
                model_name='scene_detection',
                model_version='pyscenedetect-0.6',
                success=False,
                error=str(e)
            )