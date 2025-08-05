"""
ML Services API - Delegates to Unified ML Services
Maintains backward compatibility while using unified frame extraction
Each method runs ONLY its specific service, not all services
"""

import logging
from pathlib import Path
from typing import Dict, Any
from .ml_services_unified import UnifiedMLServices

logger = logging.getLogger(__name__)

class MLServices:
    """
    ML Services wrapper that uses unified frame extraction
    Maintains API compatibility with existing code
    """
    
    def __init__(self):
        self.unified_services = UnifiedMLServices()
        self._video_id_cache = {}  # Map video_path to video_id
        
    def _get_video_id(self, video_path: Path) -> str:
        """Generate or retrieve video ID from path"""
        video_str = str(video_path)
        if video_str not in self._video_id_cache:
            # Extract video ID from path or generate one
            if video_path.stem.isdigit():
                video_id = video_path.stem
            else:
                import hashlib
                video_id = hashlib.md5(video_str.encode()).hexdigest()[:12]
            self._video_id_cache[video_str] = video_id
        return self._video_id_cache[video_str]
        
    async def run_all_ml_services(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Run all ML services on video using unified frame extraction
        This is the only method that runs everything at once
        """
        video_id = self._get_video_id(video_path)
        return await self.unified_services.analyze_video(video_path, video_id, output_dir)
        
    async def run_yolo_detection(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run ONLY YOLO object detection"""
        video_id = self._get_video_id(video_path)
        
        # Extract frames (will use cache if available)
        frame_data = await self.unified_services.frame_manager.extract_frames(
            video_path, video_id
        )
        
        if not frame_data.get('success') or not frame_data.get('frames'):
            return self.unified_services._empty_yolo_result()
        
        # Run ONLY YOLO on the frames
        return await self.unified_services._run_yolo_on_frames(
            frame_data['frames'], 
            video_id, 
            output_dir
        )
        
    async def run_whisper_transcription(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run ONLY Whisper transcription"""
        video_id = self._get_video_id(video_path)
        
        # Whisper doesn't need frames, run directly
        return await self.unified_services._run_whisper_on_video(
            video_path, video_id, output_dir
        )
        
    async def run_mediapipe_analysis(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run ONLY MediaPipe analysis"""
        video_id = self._get_video_id(video_path)
        
        # Extract frames (will use cache if available)
        frame_data = await self.unified_services.frame_manager.extract_frames(
            video_path, video_id
        )
        
        if not frame_data.get('success') or not frame_data.get('frames'):
            return self.unified_services._empty_mediapipe_result()
        
        # Run ONLY MediaPipe on the frames
        return await self.unified_services._run_mediapipe_on_frames(
            frame_data['frames'],
            video_id,
            output_dir
        )
        
    async def run_ocr_analysis(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run ONLY OCR analysis"""
        video_id = self._get_video_id(video_path)
        
        # Extract frames (will use cache if available)
        frame_data = await self.unified_services.frame_manager.extract_frames(
            video_path, video_id
        )
        
        if not frame_data.get('success') or not frame_data.get('frames'):
            return self.unified_services._empty_ocr_result()
        
        # Run ONLY OCR on the frames
        return await self.unified_services._run_ocr_on_frames(
            frame_data['frames'],
            video_id,
            output_dir
        )
        
    async def run_scene_detection(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Run scene detection (existing implementation)"""
        # Scene detection continues to work independently
        # This is already working in the current system
        from scenedetect import detect, ContentDetector
        
        try:
            scenes = detect(str(video_path), ContentDetector())
            
            scene_list = []
            for i, (start, end) in enumerate(scenes):
                scene_list.append({
                    'scene_number': i + 1,
                    'start_time': start.get_seconds(),
                    'end_time': end.get_seconds(),
                    'duration': (end - start).get_seconds()
                })
                
            return {
                'scenes': scene_list,
                'total_scenes': len(scene_list),
                'metadata': {'processed': True}
            }
            
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return {
                'scenes': [],
                'total_scenes': 0,
                'metadata': {'processed': False, 'error': str(e)}
            }
            
    async def cleanup(self):
        """Clean up resources"""
        await self.unified_services.cleanup()