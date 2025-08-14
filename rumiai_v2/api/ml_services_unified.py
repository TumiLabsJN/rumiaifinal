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
from .audio_utils import extract_audio_simple
from ..ml_services.audio_energy_service import get_audio_energy_service

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
            'whisper': asyncio.Lock(),
            'audio_energy': asyncio.Lock()
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
            elif model_name == 'audio_energy':
                self._models['audio_energy'] = get_audio_energy_service()
            
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
                models = {
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
                    ),
                    'face_mesh': mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=5,  # Support multi-person detection
                        refine_landmarks=True,  # Enable iris landmarks (468-477)
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                }
                
                # Fail-fast: Verify FaceMesh loaded successfully
                if not models['face_mesh']:
                    raise RuntimeError("FaceMesh failed to initialize - failing fast")
                
                return models
            
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
                self._run_audio_services(video_path, video_id, output_dir),
                600, "Audio Services"  # 10 minutes for audio extraction + processing
            ),
            return_exceptions=True
        )
        
        # Handle results
        yolo_result = results[0] if results[0] and not isinstance(results[0], Exception) else self._empty_yolo_result()
        mediapipe_result = results[1] if results[1] and not isinstance(results[1], Exception) else self._empty_mediapipe_result()
        ocr_result = results[2] if results[2] and not isinstance(results[2], Exception) else self._empty_ocr_result()
        
        # Audio services returns a tuple (whisper_result, energy_result)
        if results[3] and not isinstance(results[3], Exception):
            whisper_result, energy_result = results[3]
        else:
            whisper_result = self._empty_whisper_result()
            energy_result = self._empty_audio_energy_result()
        
        return {
            'yolo': yolo_result,
            'mediapipe': mediapipe_result,
            'ocr': ocr_result,
            'whisper': whisper_result,
            'audio_energy': energy_result,
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
        all_gaze = []  # ADD: Collect gaze data from batches
        
        for i in range(0, len(mp_frames), batch_size):
            batch = mp_frames[i:i+batch_size]
            
            batch_results = await asyncio.to_thread(
                self._process_mediapipe_batch, models, batch
            )
            
            all_poses.extend(batch_results['poses'])
            all_faces.extend(batch_results['faces'])
            all_hands.extend(batch_results['hands'])
            all_gaze.extend(batch_results.get('gaze', []))  # ADD: Aggregate gaze data
        
        result = {
            'poses': all_poses,
            'faces': all_faces,
            'hands': all_hands,
            'gaze': all_gaze,  # ADD: Include gaze in result
            'gestures': [],  # Would need additional processing
            'presence_percentage': (len(all_poses) / len(mp_frames) * 100) if mp_frames else 0,
            'frames_with_people': len(all_poses),
            'metadata': {
                'frames_analyzed': len(mp_frames),
                'processed': True,
                'poses_detected': len(all_poses),
                'faces_detected': len(all_faces),
                'hands_detected': len(all_hands),
                'gaze_detected': len(all_gaze)  # ADD: Track gaze detections
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
        gaze_data = []  # ADD: Gaze detection results
        
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
                    
            # Process face with bbox extraction (FIX 2A)
            if models['face']:
                face_results = models['face'].process(rgb_frame)
                if face_results.detections:
                    detection = face_results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    
                    faces.append({
                        'timestamp': frame_data.timestamp,
                        'frame_number': frame_data.frame_number,
                        'count': len(face_results.detections),
                        'confidence': detection.score[0],
                        'bbox': {  # ADD: Capture actual bbox data
                            'x': bbox.xmin,
                            'y': bbox.ymin,
                            'width': bbox.width,
                            'height': bbox.height
                        }
                    })
                    
                    # Process gaze using FaceMesh (only if face detected - early exit optimization)
                    if models.get('face_mesh'):
                        mesh_results = models['face_mesh'].process(rgb_frame)
                        if mesh_results.multi_face_landmarks:
                            gaze_info = self._compute_gaze_from_mesh(
                                mesh_results.multi_face_landmarks[0],
                                frame_data.timestamp
                            )
                            if gaze_info:
                                gaze_data.append(gaze_info)
                    
            # Process hands
            if models['hands']:
                hand_results = models['hands'].process(rgb_frame)
                if hand_results.multi_hand_landmarks:
                    hands.append({
                        'timestamp': frame_data.timestamp,
                        'frame_number': frame_data.frame_number,
                        'count': len(hand_results.multi_hand_landmarks)
                    })
                    
        return {'poses': poses, 'faces': faces, 'hands': hands, 'gaze': gaze_data}
    
    def _compute_gaze_from_mesh(self, face_landmarks, timestamp):
        """Compute gaze direction from MediaPipe FaceMesh landmarks
        
        Note: Requires refine_landmarks=True for iris detection (landmarks 468-477)
        """
        
        # VERIFIED MediaPipe FaceMesh landmark indices
        # Iris landmarks (only available with refine_landmarks=True)
        LEFT_IRIS_CENTER = 468   # Center of left iris (landmark 468)
        RIGHT_IRIS_CENTER = 473  # Center of right iris (landmark 473)
        
        # Eye corner landmarks for calculating eye center
        LEFT_EYE_INNER = 133     # Inner corner of left eye
        LEFT_EYE_OUTER = 33      # Outer corner of left eye  
        RIGHT_EYE_INNER = 362    # Inner corner of right eye
        RIGHT_EYE_OUTER = 263    # Outer corner of right eye
        
        try:
            # Get iris positions (requires refine_landmarks=True)
            left_iris = face_landmarks.landmark[LEFT_IRIS_CENTER]
            right_iris = face_landmarks.landmark[RIGHT_IRIS_CENTER]
            
            # Calculate eye centers from eye corners
            left_eye_inner = face_landmarks.landmark[LEFT_EYE_INNER]
            left_eye_outer = face_landmarks.landmark[LEFT_EYE_OUTER]
            left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2
            left_eye_center_y = (left_eye_inner.y + left_eye_outer.y) / 2
            
            right_eye_inner = face_landmarks.landmark[RIGHT_EYE_INNER]
            right_eye_outer = face_landmarks.landmark[RIGHT_EYE_OUTER]
            right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2
            right_eye_center_y = (right_eye_inner.y + right_eye_outer.y) / 2
            
            # Calculate gaze offset: iris position relative to eye center
            left_gaze_offset_x = left_iris.x - left_eye_center_x
            left_gaze_offset_y = left_iris.y - left_eye_center_y
            right_gaze_offset_x = right_iris.x - right_eye_center_x
            right_gaze_offset_y = right_iris.y - right_eye_center_y
            
            # Average horizontal and vertical offsets
            avg_gaze_offset_x = (left_gaze_offset_x + right_gaze_offset_x) / 2
            avg_gaze_offset_y = (left_gaze_offset_y + right_gaze_offset_y) / 2
            
            # Calculate total gaze offset magnitude (both axes)
            gaze_magnitude = (avg_gaze_offset_x**2 + avg_gaze_offset_y**2) ** 0.5
            
            # Classify eye contact based on offset magnitude
            # Small offset = looking at camera, large offset = looking away
            eye_contact_threshold = 0.02  # Tunable parameter (in normalized coordinates)
            is_looking_at_camera = gaze_magnitude < eye_contact_threshold
            
            # Determine gaze direction (considering both axes)
            if is_looking_at_camera:
                gaze_direction = 'camera'
                eye_contact_score = max(0, 1.0 - gaze_magnitude / eye_contact_threshold)
            else:
                # Determine primary gaze direction
                if abs(avg_gaze_offset_x) > abs(avg_gaze_offset_y):
                    # Horizontal gaze is stronger
                    gaze_direction = 'right' if avg_gaze_offset_x > 0 else 'left'
                else:
                    # Vertical gaze is stronger
                    gaze_direction = 'down' if avg_gaze_offset_y > 0 else 'up'
                eye_contact_score = 0.0
            
            return {
                'timestamp': timestamp,
                'eye_contact': eye_contact_score,
                'gaze_direction': gaze_direction,
                'gaze_offset_x': avg_gaze_offset_x,
                'gaze_offset_y': avg_gaze_offset_y,
                'gaze_magnitude': gaze_magnitude,
                'confidence': 0.8  # Can be enhanced with landmark visibility scores
            }
            
        except (IndexError, AttributeError):
            # No face/landmarks detected - return None (not an error)
            # Calling code will handle empty gaze data
            return None
        
    async def _run_ocr_on_frames(self,
                                frames: List[FrameData],
                                video_id: str,
                                output_dir: Path) -> Dict[str, Any]:
        """Run OCR on pre-extracted frames WITH inline sticker detection"""
        import cv2
        import numpy as np
        
        reader = await self._ensure_model_loaded('ocr')
        if not reader:
            logger.warning("OCR model not available")
            return self._empty_ocr_result()
            
        # Get frames optimized for OCR (adaptive sampling)
        ocr_frames = self.frame_manager.get_frames_for_service(frames, 'ocr')
        logger.info(f"Running OCR on {len(ocr_frames)} frames")
        
        # Add inline sticker detection function
        def detect_stickers_inline(image_array):
            """Fast HSV-based sticker detection (3-5ms per frame)"""
            try:
                # Convert to HSV for color detection
                hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1]
                
                # Threshold for high saturation (stickers/graphics) - LOWERED from 180 to 120
                _, binary = cv2.threshold(saturation, 120, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                stickers = []
                for contour in contours[:5]:  # Limit to 5 per frame for performance
                    area = cv2.contourArea(contour)
                    if 200 < area < 15000:  # Sticker size range - WIDENED from 500-5000 to 200-15000
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        # Classify as sticker with relaxed shape constraints - WIDENED from 0.8-1.2 to 0.3-3.0
                        if 0.3 < aspect_ratio < 3.0:
                            stickers.append({
                                'bbox': [int(x), int(y), int(w), int(h)],
                                'confidence': 0.7,
                                'type': 'sticker'
                            })
                return stickers
            except Exception as e:
                logger.debug(f"Sticker detection failed: {e}")
                return []
        
        # Process OCR and stickers on same frames
        text_annotations = []
        sticker_detections = []
        seen_texts = set()
        seen_stickers = set()  # Deduplicate stickers
        
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
                            
                            # Convert polygon bbox to flat format [x, y, width, height]
                            if bbox and len(bbox) >= 4:
                                xs = [float(pt[0]) for pt in bbox]
                                ys = [float(pt[1]) for pt in bbox]
                                bbox_list = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
                            else:
                                bbox_list = [0, 0, 0, 0]  # Fallback for invalid bbox
                            
                            text_annotations.append({
                                'text': text_clean,
                                'confidence': float(confidence),
                                'timestamp': frame_data.timestamp,
                                'bbox': bbox_list,
                                'frame_number': frame_data.frame_number
                            })
                
                # ADD: Inline sticker detection (adds ~3-5ms)
                frame_stickers = detect_stickers_inline(frame_data.image)
                for sticker in frame_stickers:
                    # Create unique key for deduplication
                    sticker_key = f"{sticker['bbox'][0]}_{sticker['bbox'][1]}"
                    if sticker_key not in seen_stickers:
                        seen_stickers.add(sticker_key)
                        sticker['timestamp'] = frame_data.timestamp
                        sticker['frame_number'] = frame_data.frame_number
                        sticker_detections.append(sticker)
                            
            except Exception as e:
                logger.warning(f"Frame processing failed on frame {frame_data.frame_number}: {e}")
        
        result = {
            'textAnnotations': text_annotations,
            'stickers': sticker_detections,  # NOW POPULATED!
            'metadata': {
                'frames_analyzed': len(ocr_frames),
                'unique_texts': len(seen_texts),
                'stickers_detected': len(sticker_detections),  # ADD THIS
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
    
    async def _run_audio_services(self,
                                 video_path: Path,
                                 video_id: str,
                                 output_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Run Whisper and Audio Energy services on shared audio file"""
        # Extract audio once for both services
        logger.info(f"Extracting audio for {video_id}")
        temp_audio = None
        
        try:
            temp_audio = await extract_audio_simple(video_path)
            
            # Load both services
            transcriber = await self._ensure_model_loaded('whisper')
            energy_service = await self._ensure_model_loaded('audio_energy')
            
            if not transcriber:
                logger.warning("Whisper model not available")
                whisper_result = self._empty_whisper_result()
            else:
                # Transcribe using the extracted audio directly
                logger.info(f"Running Whisper transcription on {video_id}")
                whisper_result = await transcriber.transcriber.transcribe(
                    temp_audio,
                    timeout=600
                )
                whisper_result['metadata'] = {
                    'model': 'whisper.cpp-base',
                    'processed': True,
                    'success': True,
                    'backend': 'whisper.cpp'
                }
                
                # Save Whisper results
                output_file = output_dir / f"{video_id}_whisper.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(whisper_result, f, indent=2)
            
            if not energy_service:
                logger.warning("Audio energy service not available")
                energy_result = self._empty_audio_energy_result()
            else:
                # Analyze energy using the same audio file
                logger.info(f"Running audio energy analysis on {video_id}")
                energy_result = await energy_service.analyze(temp_audio)
                
                # Save energy results
                await energy_service.save_results(energy_result, video_id, output_dir)
            
            return whisper_result, energy_result
            
        finally:
            # Clean up temporary audio file after BOTH services complete
            if temp_audio and temp_audio.exists():
                try:
                    temp_audio.unlink()
                    logger.debug(f"Cleaned up temporary audio file: {temp_audio}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp audio: {e}")
        
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results for all services"""
        return {
            'yolo': self._empty_yolo_result(),
            'mediapipe': self._empty_mediapipe_result(),
            'ocr': self._empty_ocr_result(),
            'whisper': self._empty_whisper_result(),
            'audio_energy': self._empty_audio_energy_result(),
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
    
    def _empty_audio_energy_result(self) -> Dict[str, Any]:
        """Empty Audio Energy result"""
        return {
            'energy_level_windows': {},
            'energy_variance': 0.0,
            'climax_timestamp': 0.0,
            'burst_pattern': 'unknown',
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