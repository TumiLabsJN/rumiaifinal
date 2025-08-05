"""
Unified ML Services - All services use shared frame extraction
Native async implementation without unnecessary wrappers
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from ..processors.unified_frame_manager import get_frame_manager, FrameData
from .whisper_transcribe_safe import get_transcriber

logger = logging.getLogger(__name__)

class UnifiedMLServices:
    """
    ML services that use unified frame extraction
    Processes frames once, analyzes many times
    With lazy model loading and proper async patterns
    """
    
    def __init__(self):
        self.frame_manager = get_frame_manager()
        self._models = {}  # Models loaded on demand
        self._model_locks = {
            'yolo': asyncio.Lock(),
            'mediapipe': asyncio.Lock(),
            'ocr': asyncio.Lock(),
            'whisper': asyncio.Lock()
        }
        
    async def _ensure_model_loaded(self, model_name: str):
        """Load model only when needed (lazy loading)"""
        if model_name in self._models:
            return self._models[model_name]
        
        async with self._model_locks[model_name]:
            # Double-check after acquiring lock
            if model_name in self._models:
                return self._models[model_name]
            
            logger.info(f"Loading {model_name} model...")
            
            if model_name == 'yolo':
                self._models['yolo'] = await self._load_yolo_model()
            elif model_name == 'mediapipe':
                self._models['mediapipe'] = await self._load_mediapipe_models()
            elif model_name == 'ocr':
                self._models['ocr'] = await self._load_ocr_model()
            elif model_name == 'whisper':
                self._models['whisper'] = get_transcriber()
            
            logger.info(f"{model_name} model loaded successfully")
        
        return self._models.get(model_name)
        
    async def _load_yolo_model(self):
        """Load YOLO model asynchronously"""
        try:
            from ultralytics import YOLO
            
            model_path = '/home/jorge/rumiaifinal/yolov8n.pt'
            if os.path.exists(model_path):
                model = await asyncio.to_thread(YOLO, model_path)
            else:
                model = await asyncio.to_thread(YOLO, 'yolov8n.pt')
            
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            return None
        
    async def _load_mediapipe_models(self):
        """Load MediaPipe models asynchronously"""
        try:
            import mediapipe as mp
            
            # Load models in thread to avoid blocking
            def load_mp():
                return {
                    'pose': mp.solutions.pose.Pose(
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    ),
                    'face': mp.solutions.face_detection.FaceDetection(
                        min_detection_confidence=0.5
                    ),
                    'hands': mp.solutions.hands.Hands(
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                }
            
            return await asyncio.to_thread(load_mp)
        except Exception as e:
            logger.error(f"Failed to load MediaPipe: {e}")
            return None
        
    async def _load_ocr_model(self):
        """Load OCR model asynchronously"""
        try:
            import easyocr
            import torch
            
            use_gpu = torch.cuda.is_available()
            reader = await asyncio.to_thread(
                easyocr.Reader, ['en'], gpu=use_gpu
            )
            logger.info(f"EasyOCR loaded (GPU: {use_gpu})")
            return reader
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            return None
        
    async def analyze_video(self, video_path: Path, video_id: str, output_dir: Path) -> Dict[str, Any]:
        """
        Analyze video with all ML services using unified frame extraction
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            output_dir: Output directory for results
            
        Returns:
            Combined ML analysis results
        """
        # Extract frames once with timeout protection
        try:
            async with asyncio.timeout(600):  # 10 minute timeout for extraction
                logger.info(f"Extracting frames for video: {video_id}")
                frame_data = await self.frame_manager.extract_frames(video_path, video_id)
        except asyncio.TimeoutError:
            logger.error(f"Frame extraction timed out for {video_id}")
            return self._empty_results()
        
        if not frame_data.get('success') or not frame_data.get('frames'):
            logger.error(f"Frame extraction failed: {frame_data.get('error')}")
            return self._empty_results()
        
        frames = frame_data['frames']
        metadata = frame_data['metadata']
        
        logger.info(f"Extracted {len(frames)} frames, analyzing with ML services...")
        
        # Run all ML services in parallel with individual timeouts
        async def run_with_timeout(coro, timeout_seconds, service_name):
            try:
                async with asyncio.timeout(timeout_seconds):
                    return await coro
            except asyncio.TimeoutError:
                logger.error(f"{service_name} timed out after {timeout_seconds}s")
                return None
        
        # Run services with appropriate timeouts
        results = await asyncio.gather(
            run_with_timeout(
                self._run_yolo_on_frames(frames, video_id, output_dir),
                300, "YOLO"  # 5 minutes
            ),
            run_with_timeout(
                self._run_mediapipe_on_frames(frames, video_id, output_dir),
                300, "MediaPipe"  # 5 minutes
            ),
            run_with_timeout(
                self._run_ocr_on_frames(frames, video_id, output_dir),
                300, "OCR"  # 5 minutes
            ),
            run_with_timeout(
                self._run_whisper_on_video(video_path, video_id, output_dir),
                600, "Whisper"  # 10 minutes for audio
            ),
            return_exceptions=True
        )
        
        # Handle results
        yolo_result = results[0] if results[0] and not isinstance(results[0], Exception) else self._empty_yolo_result()
        mediapipe_result = results[1] if results[1] and not isinstance(results[1], Exception) else self._empty_mediapipe_result()
        ocr_result = results[2] if results[2] and not isinstance(results[2], Exception) else self._empty_ocr_result()
        whisper_result = results[3] if results[3] and not isinstance(results[3], Exception) else self._empty_whisper_result()
        
        return {
            'yolo': yolo_result,
            'mediapipe': mediapipe_result,
            'ocr': ocr_result,
            'whisper': whisper_result,
            'metadata': {
                'video_id': video_id,
                'duration': metadata.duration if metadata else 0,
                'fps': metadata.fps if metadata else 0,
                'total_frames': metadata.total_frames if metadata else 0,
                'extracted_frames': len(frames),
                'cache_hit': frame_data.get('cache_hit', False),
                'degraded': frame_data.get('degraded', False)
            }
        }
        
    async def _run_yolo_on_frames(self, 
                                 frames: List[FrameData], 
                                 video_id: str,
                                 output_dir: Path) -> Dict[str, Any]:
        """Run YOLO on pre-extracted frames using native async"""
        # Load model only when needed
        model = await self._ensure_model_loaded('yolo')
        if not model:
            logger.warning("YOLO model not available")
            return self._empty_yolo_result()
            
        # Get frames optimized for YOLO
        yolo_frames = self.frame_manager.get_frames_for_service(frames, 'yolo')
        logger.info(f"Running YOLO on {len(yolo_frames)} frames")
        
        # Process frames in batches for efficiency
        batch_size = 10
        annotations = []
        
        for i in range(0, len(yolo_frames), batch_size):
            batch = yolo_frames[i:i+batch_size]
            
            # Process batch in thread pool
            batch_results = await asyncio.to_thread(
                self._process_yolo_batch, model, batch
            )
            annotations.extend(batch_results)
        
        result = {
            'objectAnnotations': annotations,
            'metadata': {
                'model': 'YOLOv8n',
                'processed': True,
                'frames_analyzed': len(yolo_frames),
                'objects_detected': len(annotations)
            }
        }
        
        # Save results
        output_file = output_dir / f"{video_id}_yolo_detections.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
    
    def _process_yolo_batch(self, model, frames: List[FrameData]) -> List[Dict]:
        """Process a batch of frames with YOLO (sync)"""
        results = []
        for frame_data in frames:
            detections = model(frame_data.image, verbose=False)
            
            for detection in detections:
                if detection.boxes is not None:
                    for box in detection.boxes:
                        results.append({
                            'trackId': f"obj_{frame_data.frame_number}_{int(box.cls)}",
                            'className': model.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'timestamp': frame_data.timestamp,
                            'bbox': box.xyxy[0].tolist() if len(box.xyxy) > 0 else [0,0,0,0],
                            'frame_number': frame_data.frame_number
                        })
        return results
        
    async def _run_mediapipe_on_frames(self,
                                      frames: List[FrameData],
                                      video_id: str,
                                      output_dir: Path) -> Dict[str, Any]:
        """Run MediaPipe on pre-extracted frames"""
        models = await self._ensure_model_loaded('mediapipe')
        if not models:
            logger.warning("MediaPipe models not available")
            return self._empty_mediapipe_result()
            
        # Get frames for MediaPipe (using all frames per documentation)
        mp_frames = self.frame_manager.get_frames_for_service(frames, 'mediapipe')
        logger.info(f"Running MediaPipe on {len(mp_frames)} frames")
        
        # Process in batches
        batch_size = 20
        all_poses = []
        all_faces = []
        all_hands = []
        
        for i in range(0, len(mp_frames), batch_size):
            batch = mp_frames[i:i+batch_size]
            
            batch_results = await asyncio.to_thread(
                self._process_mediapipe_batch, models, batch
            )
            
            all_poses.extend(batch_results['poses'])
            all_faces.extend(batch_results['faces'])
            all_hands.extend(batch_results['hands'])
        
        result = {
            'poses': all_poses,
            'faces': all_faces,
            'hands': all_hands,
            'gestures': [],  # Would need additional processing
            'presence_percentage': (len(all_poses) / len(mp_frames) * 100) if mp_frames else 0,
            'frames_with_people': len(all_poses),
            'metadata': {
                'frames_analyzed': len(mp_frames),
                'processed': True,
                'poses_detected': len(all_poses),
                'faces_detected': len(all_faces),
                'hands_detected': len(all_hands)
            }
        }
        
        # Save results
        output_file = output_dir / f"{video_id}_human_analysis.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
    
    def _process_mediapipe_batch(self, models, frames: List[FrameData]) -> Dict:
        """Process a batch of frames with MediaPipe (sync)"""
        import cv2
        
        poses = []
        faces = []
        hands = []
        
        for frame_data in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame_data.image, cv2.COLOR_BGR2RGB)
            
            # Process pose
            if models['pose']:
                pose_results = models['pose'].process(rgb_frame)
                if pose_results.pose_landmarks:
                    poses.append({
                        'timestamp': frame_data.timestamp,
                        'frame_number': frame_data.frame_number,
                        'landmarks': len(pose_results.pose_landmarks.landmark),
                        'visibility': sum(lm.visibility for lm in pose_results.pose_landmarks.landmark) / 33
                    })
                    
            # Process face
            if models['face']:
                face_results = models['face'].process(rgb_frame)
                if face_results.detections:
                    faces.append({
                        'timestamp': frame_data.timestamp,
                        'frame_number': frame_data.frame_number,
                        'count': len(face_results.detections),
                        'confidence': face_results.detections[0].score[0] if face_results.detections else 0
                    })
                    
            # Process hands
            if models['hands']:
                hand_results = models['hands'].process(rgb_frame)
                if hand_results.multi_hand_landmarks:
                    hands.append({
                        'timestamp': frame_data.timestamp,
                        'frame_number': frame_data.frame_number,
                        'count': len(hand_results.multi_hand_landmarks)
                    })
                    
        return {'poses': poses, 'faces': faces, 'hands': hands}
        
    async def _run_ocr_on_frames(self,
                                frames: List[FrameData],
                                video_id: str,
                                output_dir: Path) -> Dict[str, Any]:
        """Run OCR on pre-extracted frames"""
        reader = await self._ensure_model_loaded('ocr')
        if not reader:
            logger.warning("OCR model not available")
            return self._empty_ocr_result()
            
        # Get frames optimized for OCR (adaptive sampling)
        ocr_frames = self.frame_manager.get_frames_for_service(frames, 'ocr')
        logger.info(f"Running OCR on {len(ocr_frames)} frames")
        
        # Process frames
        text_annotations = []
        seen_texts = set()
        
        for frame_data in ocr_frames:
            try:
                # Run OCR in thread
                results = await asyncio.to_thread(
                    reader.readtext, frame_data.image
                )
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.5 and len(text.strip()) > 2:
                        text_clean = text.strip()
                        
                        # Deduplicate similar texts
                        if text_clean not in seen_texts:
                            seen_texts.add(text_clean)
                            
                            bbox_list = [[float(pt[0]), float(pt[1])] for pt in bbox]
                            
                            text_annotations.append({
                                'text': text_clean,
                                'confidence': float(confidence),
                                'timestamp': frame_data.timestamp,
                                'bbox': bbox_list,
                                'frame_number': frame_data.frame_number
                            })
                            
            except Exception as e:
                logger.warning(f"OCR failed on frame {frame_data.frame_number}: {e}")
        
        result = {
            'textAnnotations': text_annotations,
            'stickers': [],  # Would need sticker detection
            'metadata': {
                'frames_analyzed': len(ocr_frames),
                'unique_texts': len(seen_texts),
                'processed': True
            }
        }
        
        # Save results
        output_file = output_dir / f"{video_id}_creative_analysis.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    async def _run_whisper_on_video(self,
                                   video_path: Path,
                                   video_id: str,
                                   output_dir: Path) -> Dict[str, Any]:
        """Run Whisper on video audio (doesn't need frames)"""
        transcriber = await self._ensure_model_loaded('whisper')
        if not transcriber:
            logger.warning("Whisper model not available")
            return self._empty_whisper_result()
        
        # Transcribe with 10 minute timeout
        result = await transcriber.transcribe(video_path, timeout=600)
        
        # Save results
        output_file = output_dir / f"{video_id}_whisper.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results for all services"""
        return {
            'yolo': self._empty_yolo_result(),
            'mediapipe': self._empty_mediapipe_result(),
            'ocr': self._empty_ocr_result(),
            'whisper': self._empty_whisper_result(),
            'metadata': {'processed': False, 'error': 'Frame extraction failed'}
        }
        
    def _empty_yolo_result(self) -> Dict[str, Any]:
        """Empty YOLO result"""
        return {
            'objectAnnotations': [],
            'metadata': {'processed': False, 'model': 'YOLOv8'}
        }
        
    def _empty_mediapipe_result(self) -> Dict[str, Any]:
        """Empty MediaPipe result"""
        return {
            'poses': [],
            'faces': [],
            'hands': [],
            'gestures': [],
            'presence_percentage': 0.0,
            'frames_with_people': 0,
            'metadata': {'frames_analyzed': 0, 'processed': False}
        }
        
    def _empty_ocr_result(self) -> Dict[str, Any]:
        """Empty OCR result"""
        return {
            'textAnnotations': [],
            'stickers': [],
            'metadata': {'frames_analyzed': 0, 'processed': False}
        }
        
    def _empty_whisper_result(self) -> Dict[str, Any]:
        """Empty Whisper result"""
        return {
            'text': '',
            'segments': [],
            'language': 'unknown',
            'duration': 0,
            'metadata': {'processed': False}
        }
        
    async def cleanup(self):
        """Clean up resources"""
        await self.frame_manager.cleanup()
        
        # Close MediaPipe models if loaded
        if 'mediapipe' in self._models and self._models['mediapipe']:
            models = self._models['mediapipe']
            if 'pose' in models and models['pose']:
                models['pose'].close()
            if 'face' in models and models['face']:
                models['face'].close()
            if 'hands' in models and models['hands']:
                models['hands'].close()