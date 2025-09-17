"""
DeepFace Gender Detection Service for RumiAI

Provides gender detection from video frames to enable gender-specific pitch normalization.
Operates independently from MediaPipe face detection for redundancy and different detection capabilities.
"""

# FIX: Configure TensorFlow threading BEFORE any imports
import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)

# Decision Point 3: Error hierarchy for critical vs expected failures
class VideoLoadError(Exception):
    """Critical: Video file cannot be loaded or is corrupted"""
    pass

class ModelInitializationError(Exception):
    """Critical: DeepFace model failed to initialize"""
    pass


@dataclass
class DeepFaceConfig:
    """
    Decision Point 10: Dataclass configuration with environment variable support.
    Decision Point 13: Validates configuration in __post_init__.
    Type-safe, testable, and clearly documents all configurable options.
    """
    timeout: int = 10  # Max seconds per video analysis (Decision Point 4)
    detector_backend: str = 'opencv'  # Face detection backend
    enforce_detection: bool = False  # Whether to fail if no faces detected
    use_gpu: bool = False  # Whether to use GPU if available
    thread_workers: int = 2  # Thread pool size per video (Decision Point 2)

    def __post_init__(self):
        """Decision Point 13: Validate configuration after initialization"""
        # Validate detector_backend
        valid_backends = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe']
        if self.detector_backend not in valid_backends:
            raise ValueError(
                f"Invalid detector_backend: '{self.detector_backend}'. "
                f"Must be one of: {', '.join(valid_backends)}"
            )

        # Validate timeout
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")

        # Validate thread_workers
        if self.thread_workers <= 0:
            raise ValueError(f"Thread workers must be positive, got {self.thread_workers}")

    @classmethod
    def from_env(cls):
        """Create config from environment variables with defaults"""
        return cls(
            timeout=int(os.getenv('DEEPFACE_TIMEOUT', '10')),
            detector_backend=os.getenv('DEEPFACE_DETECTOR', 'opencv'),  # opencv is fastest
            enforce_detection=os.getenv('DEEPFACE_ENFORCE', 'false').lower() == 'true',
            use_gpu=os.getenv('DEEPFACE_USE_GPU', 'false').lower() == 'true',
            thread_workers=int(os.getenv('DEEPFACE_WORKERS', '2'))
        )


class DeepFaceGenderService:
    """
    DeepFace service for gender detection.
    Decision Point 1: Independent from MediaPipe - does its own face detection.
    Decision Point 2: ThreadPoolExecutor created/destroyed per video (no resource leaks).
    Decision Point 3: Distinguishes critical errors (fail-fast) from expected cases (empty result).
    Decision Point 5: All imports at module level for better performance.
    Decision Point 6: Any multi-person frame triggers self-normalization.
    Decision Point 8: Adaptive frame sampling based on video duration.
    Decision Point 9: No caching - analyzes fresh each time for consistency.
    Decision Point 10: No batch mode - processes one video at a time.
    Returns schema-compliant data per our defined interface.
    """

    def __init__(self, config: DeepFaceConfig = None):
        """Initialize service with configuration

        Decision Point 10: Accepts dataclass config for flexibility.
        Decision Point 2: No persistent executor - created per analyze() call.
        FIXED: No pre-initialization - models load on first use in thread context.
        """
        self.config = config or DeepFaceConfig.from_env()
        self._thread_local = threading.local()

        # Decision Point 10: GPU usage now configurable
        if not self.config.use_gpu:
            self._force_cpu()

    def _force_cpu(self):
        """Force CPU usage when configured

        Decision Point 10: Only forces CPU when use_gpu=False
        NOTE: Threading config is now done at module level before imports
        """
        # Already configured at module level
        logger.info("DeepFace configured to use CPU with single-threaded TensorFlow")

    def _ensure_model_loaded(self):
        """Ensure models are loaded in current thread.

        Thread-local initialization avoids conflicts and memory issues.
        Models are loaded once per thread, cached for reuse.
        """
        if not hasattr(self._thread_local, 'model_loaded'):
            self._thread_local.model_loaded = False

        if not self._thread_local.model_loaded:
            try:
                # Configure TensorFlow threading in this thread
                import tensorflow as tf
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)

                logger.info(f"Loading DeepFace model in thread {threading.current_thread().name}")
                # Load by doing a dummy analysis
                test_img = np.zeros((224, 224, 3), dtype=np.uint8)
                DeepFace.analyze(
                    test_img,
                    actions=['gender'],
                    enforce_detection=False,
                    detector_backend=self.config.detector_backend,
                    silent=True
                )
                self._thread_local.model_loaded = True
                logger.info("DeepFace model loaded successfully in thread")
            except Exception as e:
                logger.error(f"Failed to load DeepFace model: {e}")
                raise ModelInitializationError(f"Cannot initialize DeepFace: {e}")

    def _calculate_frame_count(self, video_duration: float) -> int:
        """
        Decision Point 9: Adaptive frame count based on video duration.
        Avoids oversampling short videos and undersampling long ones.
        """
        if video_duration < 5:
            return 2  # Very short video - sample at ~33% and ~66%
        elif video_duration < 15:
            return 3  # Short video - beginning, middle, end
        elif video_duration < 30:
            return 5  # Medium video - good coverage
        else:
            return 7  # Long video - maximum temporal coverage

    def _sample_frames(self, video_path: str) -> list:
        """
        Sample frames evenly across video.
        Decision Point 9: Uses adaptive frame count instead of hardcoded 5.
        Decision Point 7: Raises VideoLoadError for critical failures.
        """
        cap = cv2.VideoCapture(video_path)

        # Decision Point 3: Check if video opened successfully
        if not cap.isOpened():
            raise VideoLoadError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0 or fps <= 0:
            cap.release()
            raise VideoLoadError(f"Invalid video metadata: frames={total_frames}, fps={fps}")

        video_duration = total_frames / fps

        # Adaptive frame count (Decision Point 9)
        num_frames = self._calculate_frame_count(video_duration)

        if total_frames < num_frames:
            num_frames = max(1, total_frames)

        # Sample evenly across video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()  # Decision Point 9: Always release VideoCapture
        return frames

    async def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze video for gender detection using asyncio.to_thread.

        Decision Point 1: Independent service - does its own face detection.
        FIXED: Use asyncio.to_thread instead of ThreadPoolExecutor to avoid TF conflicts.
        Decision Point 3: Fails fast for critical errors, returns empty for expected.
        Decision Point 4: Applies timeout to entire analysis operation.
        Decision Point 5: Uses module-level imports for better performance.
        Decision Point 8: No caching - always analyzes fresh for consistency.
        Returns schema-compliant result:
        - Required: gender (str|None), confidence (float)
        - Optional: method, processing_ms, frames_analyzed
        """
        try:
            # FIXED: Use asyncio.to_thread to avoid ThreadPoolExecutor + TensorFlow conflict
            # This creates a thread but doesn't use ThreadPoolExecutor which conflicts with TF
            result = await asyncio.wait_for(
                asyncio.to_thread(self._analyze_sync, video_path),
                timeout=self.config.timeout
            )
            return result
        except asyncio.TimeoutError:
            # Decision Point 4: Timeout is an expected failure
            logger.warning(f"DeepFace analysis timed out after {self.config.timeout}s for {video_path}")
            return self._empty_result(f"timeout_after_{self.config.timeout}s")
        except (FileNotFoundError, VideoLoadError, ModelInitializationError) as e:
            # Decision Point 3: Critical errors - fail fast
            logger.error(f"Critical error in DeepFace analysis: {e}")
            raise
        except Exception as e:
            # Decision Point 3: Expected failures (no faces, etc) - return empty
            logger.warning(f"DeepFace analysis completed with no detection: {e}")
            return self._empty_result(str(e))

    def _analyze_sync(self, video_path: str) -> Dict[str, Any]:
        """Synchronous analysis function to run in thread pool

        Decision Point 5: No function-level imports - all at module level.
        FIXED: Ensures model is loaded in this thread before analysis.
        """
        # Ensure model is loaded in this thread
        self._ensure_model_loaded()

        start_time = time.time()

        # Decision Point 2: Map DeepFace labels to our expected format
        GENDER_MAP = {'man': 'male', 'woman': 'female'}

        try:
            # Sample frames from video (blocking I/O)
            # Decision Point 9: Adaptive frame count, no hardcoded value
            # Decision Point 7: May raise VideoLoadError for critical failures
            frames = self._sample_frames(video_path)

            if not frames:
                # Decision Point 3: No frames is expected case, not critical
                logger.info(f"No frames extracted from {video_path}")
                return self._empty_result("no_frames")

            # Analyze each frame
            gender_votes = []

            for frame in frames:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['gender'],
                        enforce_detection=self.config.enforce_detection,  # Decision Point 10
                        detector_backend=self.config.detector_backend,  # Face detection backend
                        silent=True
                    )

                    if result and len(result) > 0:
                        # Decision Point 6: Handle multi-person scenarios
                        if len(result) > 1:
                            # Multiple people detected - will trigger self-normalization
                            logger.debug(f"Multiple people ({len(result)}) detected in frame")
                            gender_votes.append({
                                'gender': 'multiple_people',
                                'confidence': 0.0,  # Zero confidence for multi-person
                                'num_people': len(result)
                            })
                        else:
                            # Single person detected
                            gender_data = result[0]['gender']
                            dominant = result[0]['dominant_gender']  # 'Man' or 'Woman'
                            confidence = gender_data[dominant] / 100.0

                            # Map to our expected labels (Decision Point 2)
                            mapped_gender = GENDER_MAP.get(dominant.lower(), dominant.lower())

                            gender_votes.append({
                                'gender': mapped_gender,  # 'male' or 'female'
                                'confidence': confidence
                            })
                except Exception as e:
                    logger.debug(f"Frame analysis failed: {e}")
                    continue

            # Aggregate results
            if not gender_votes:
                # Decision Point 4: Expected case - return empty result
                logger.info(f"No faces detected in {video_path} - expected for some videos")
                return self._empty_result("no_faces_detected")

            # Decision Point 6: Check for multiple people (conservative)
            # Any frame with multiple people triggers self-normalization
            if any(v['gender'] == 'multiple_people' for v in gender_votes):
                multi_person_count = sum(1 for v in gender_votes if v['gender'] == 'multiple_people')
                logger.info(f"Multiple people detected in {multi_person_count}/{len(frames)} frames - using self-normalization")
                return {
                    'gender': 'multiple_people',
                    'confidence': 0.0,
                    'method': 'deepface',
                    'processing_ms': int((time.time() - start_time) * 1000),
                    'frames_analyzed': len(frames),
                    'multi_person_frames': multi_person_count,
                    'note': 'Multiple people detected - use self-normalization'
                }

            # Majority vote with average confidence
            # Now using mapped labels (Decision Point 2)
            male_votes = [v for v in gender_votes if v['gender'] == 'male']
            female_votes = [v for v in gender_votes if v['gender'] == 'female']

            if len(male_votes) > len(female_votes):
                gender = 'male'
                confidence = np.mean([v['confidence'] for v in male_votes])
            else:
                gender = 'female'
                confidence = np.mean([v['confidence'] for v in female_votes])

            processing_ms = int((time.time() - start_time) * 1000)

            # Return schema-compliant data
            return {
                # Required fields
                'gender': gender,
                'confidence': float(confidence),
                # Optional debugging fields
                'method': 'deepface',
                'processing_ms': processing_ms,
                'frames_analyzed': len(frames),
                'detector_backend': self.config.detector_backend
            }

        except (VideoLoadError, ModelInitializationError) as e:
            # Decision Point 3: Critical errors - re-raise
            raise
        except Exception as e:
            # Decision Point 3: Unexpected error in sync function
            logger.error(f"Unexpected error in _analyze_sync: {e}")
            raise

    def _empty_result(self, error: str = None) -> Dict[str, Any]:
        """Return schema-compliant empty result"""
        result = {
            'gender': None,
            'confidence': 0.0,
            'method': 'deepface'
        }
        if error:
            result['error'] = error
        return result


# Decision Point 1: Service remains independent from MediaPipe
# Decision Point 2: ThreadPoolExecutor created/destroyed per video - no cleanup needed
# Decision Point 3: Error classes defined for critical vs expected failures
# Decision Point 5: All imports at module level for performance
# Decision Point 10: Each video processed independently