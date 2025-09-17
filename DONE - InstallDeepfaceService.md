# DeepFace Gender Detection Service Installation Guide

**Created**: 2025-01-17
**Purpose**: Install DeepFace as a gender detection service to provide gender data for pitch normalization
**Architecture Pattern**: Direct ML Data Flow with Single Video Processing

## Decision Points Summary

| Decision | Choice | Rationale |
|----------|--------|----------|
| **1. Service Independence** | Keep DeepFace separate | Each service does own face detection |
| **2. Thread Pool Lifecycle** | Create/destroy per video | No resource leaks, auto-cleanup |
| **3. Error Handling** | Critical vs Expected | Fail-fast for critical, empty for expected |
| **4. Timeout Implementation** | asyncio.wait_for | Entire analysis times out as expected failure |
| **5. Import Pattern** | Module-level imports | Better performance, standard practice |
| **6. Multi-Person Logic** | Any frame triggers | Conservative approach for safety |
| **7. Error Handling** | Critical vs Expected | Fail-fast for critical, empty for expected |
| **8. Parallel Execution** | Always parallel | Let OS handle scheduling |
| **9. Frame Sampling** | Adaptive count | Scale with video duration |
| **10. Caching** | No caching | Fresh analysis each time |
| **11. VideoAnalyzer Integration** | Comment instructions | Preserve existing signature |
| **12. Pitch Integration** | Show data access only | Implementation in spectralspeech.md |
| **13. Config Validation** | __post_init__ validation | Fail fast with clear errors |

---

## Executive Summary

DeepFace will be installed as a **Direct ML Data service** that bypasses the timeline and feeds directly into `ml_data['gender_detection']`. The service:
- Runs in parallel with audio extraction (Decision Point 8)
- Processes videos one at a time (no batch mode)
- Creates fresh thread pool per video (Decision Point 2)
- Applies configurable timeout to analysis (Decision Point 4)
- Uses adaptive frame sampling (Decision Point 7)
- Analyzes fresh each time without caching (Decision Point 8)
- Configurable via environment variables (Decision Point 10)

---

## Architecture Decision

### Why Direct ML Data Flow (Not Timeline)

| Aspect | Reasoning |
|--------|-----------|
| **Data Type** | Video-level attribute, not temporal event |
| **Output** | Single gender + confidence per video |
| **Usage** | Normalization parameter for pitch metrics |
| **Performance** | Parallel execution with audio service |
| **Processing Mode** | Single video at a time (Decision Point 9) |
| **Independence** | Separate from MediaPipe face detection (Decision Point 1) |

### Data Flow Path
```
Video → [DeepFace || Audio] → ml_data['gender_detection'] → consumers (e.g., pitch normalization)
         (parallel services)                                    (see spectralspeech.md for usage)
```

---

## Installation Roadmap

### Phase 1: Environment Setup

#### 1.1 Install Dependencies

**Python 3.12.3 Compatibility Requirements:**
- DeepFace: 0.0.92+ (latest version recommended)
- TensorFlow: 2.16.1+ (supports Python 3.12)

```bash
# For Python 3.12.3 - Install specific versions
pip install deepface>=0.0.92
pip install tensorflow>=2.16.1,<2.17  # TF 2.16.x works best with Python 3.12

# Alternative: If you encounter issues, use explicit versions
pip install deepface==0.0.92 tensorflow==2.16.1

# For GPU support (optional but faster)
pip install tensorflow[and-cuda]>=2.16.1  # Includes CUDA dependencies

# Verify installation
python3 -c "from deepface import DeepFace; print('DeepFace installed successfully')"
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
```

**Note on Face Detection:**
DeepFace performs its own face detection independent of MediaPipe (Decision Point 1). The `detector_backend` parameter controls the face detection method (opencv, retinaface, mtcnn, ssd), with 'opencv' being fastest and 'retinaface' most accurate.

#### 1.2 Model Download
```python
# First run will download models (~500MB)
# Run this separately to download models before production
from deepface import DeepFace

# This will download the gender model on first use
test_result = DeepFace.analyze(
    img_path="test_image.jpg",
    actions=['gender'],
    enforce_detection=False
)
print("Models downloaded successfully")
```

---

### Phase 2: Service Implementation

#### 2.1 Create DeepFace Service
**File**: `rumiai_v2/ml_services/deepface_gender_service.py`

```python
import asyncio
import logging
import os
import time
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
    Maps DeepFace labels ('Man'/'Woman') to our format ('male'/'female').
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
        """
        self.config = config or DeepFaceConfig.from_env()
        self.model_loaded = False
        # No executor stored - Decision Point 2: created fresh per video

        # Decision Point 10: GPU usage now configurable
        if not self.config.use_gpu:
            self._force_cpu()

        self._initialize_model()

    def _force_cpu(self):
        """Force CPU usage when configured

        Decision Point 10: Only forces CPU when use_gpu=False
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("DeepFace configured to use CPU")

    def _initialize_model(self):
        """Pre-load DeepFace models

        Decision Point 3: Model initialization failure is critical - fail fast.
        """
        try:
            logger.info("Loading DeepFace gender detection model...")
            # Dummy analysis to load model
            test_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(
                test_img,
                actions=['gender'],
                enforce_detection=self.config.enforce_detection,  # Decision Point 10: Configurable
                detector_backend=self.config.detector_backend,  # Face detection method
                prog_bar=False,
                silent=True
            )
            self.model_loaded = True
            logger.info("DeepFace model loaded successfully")
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
        Analyze video for gender detection using thread pool executor.

        Decision Point 1: Independent service - does its own face detection.
        Decision Point 2: Creates fresh ThreadPoolExecutor per video (auto-cleanup).
        Decision Point 3: Fails fast for critical errors, returns empty for expected.
        Decision Point 4: Applies timeout to entire analysis operation.
        Decision Point 5: Uses module-level imports for better performance.
        Decision Point 8: No caching - always analyzes fresh for consistency.
        Returns schema-compliant result:
        - Required: gender (str|None), confidence (float)
        - Optional: method, processing_ms, frames_analyzed
        """
        loop = asyncio.get_event_loop()

        # Decision Point 2: Create fresh executor for this video
        # Automatically cleaned up when context exits
        with ThreadPoolExecutor(max_workers=self.config.thread_workers) as executor:
            try:
                # Decision Point 4: Apply timeout to entire analysis
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        self._analyze_sync,
                        video_path
                    ),
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
        """
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
                        prog_bar=False,
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


# Decision Point 5: No global state - service managed by VideoAnalyzer instance
# Each VideoAnalyzer creates and manages its own DeepFaceGenderService
# Decision Point 2: ThreadPoolExecutor created/destroyed per video - no cleanup needed
# Decision Point 3: Error classes defined for critical vs expected failures
# Decision Point 9: Each video processed independently
```

---

### Phase 3: Integration with Video Analyzer

**Decision Point 11**: Integration preserves existing VideoAnalyzer signature and code.

#### 3.1 Add to Video Analyzer
**File**: `rumiai_v2/processors/video_analyzer.py`

**Important**: The examples below show where to add code. Keep all existing VideoAnalyzer code unchanged except for the specific additions shown.

```python
# Add imports at top
from rumiai_v2.ml_services.deepface_gender_service import (
    DeepFaceGenderService,
    VideoLoadError,
    ModelInitializationError
)

class VideoAnalyzer:
    def __init__(self,
                 # Keep all existing parameters unchanged
                 ):
        # Keep all existing initialization code
        # Add only this line at the end of __init__:
        self.deepface_service = None  # Lazy load DeepFace service

    # Add to analyses dictionary (around line 39-47)
    def _get_analyses(self):
        return {
            'yolo': self._run_yolo,
            'whisper': self._run_whisper,
            'mediapipe': self._run_mediapipe,
            'ocr': self._run_ocr,
            'scene_detection': self._run_scene_detection,
            'audio_energy': self._run_audio_energy,
            'emotion_detection': self._run_emotion_detection,
            'deepface_gender': self._run_deepface_gender  # NEW
        }

    async def _run_deepface_gender(self, video_path: Path) -> Tuple[str, Any]:
        """
        Run DeepFace gender detection.
        Decision Point 8: Runs in parallel with audio service.
        """
        try:
            # Initialize service if needed (Decision Point 9: one video at a time)
            if self.deepface_service is None:
                # Decision Point 10: Service uses environment configuration
                # Decision Point 2: Service handles its own thread pool lifecycle
                self.deepface_service = DeepFaceGenderService()

            result = await self.deepface_service.analyze(str(video_path))
            return 'gender_detection', result

        except (FileNotFoundError, VideoLoadError, ModelInitializationError) as e:
            # Decision Point 3: Critical errors should propagate
            logger.error(f"Critical error in gender detection for {video_path}: {e}")
            raise
        except Exception as e:
            # Decision Point 3: Expected failures return empty result
            logger.warning(f"Gender detection returned no result for {video_path}: {e}")
            return 'gender_detection', {
                'gender': None,
                'confidence': 0.0,
                'error': str(e)
            }
```

#### 3.2 Ensure Parallel Execution
**Modification to run_analysis method**:

```python
async def run_analysis(self, video_path: Path, selected_analyses: List[str] = None):
    """Run selected ML analyses on video

    Decision Point 6: Always run ALL analyses in parallel.
    No hardcoded service names or special cases.
    Let asyncio and OS handle CPU scheduling.
    """
    if not selected_analyses:
        selected_analyses = list(self.analyses.keys())

    # Build list of tasks for all selected analyses
    tasks = []
    task_names = []

    for analysis_name in selected_analyses:
        if analysis_name in self.analyses:
            tasks.append(self.analyses[analysis_name](video_path))
            task_names.append(analysis_name)
        else:
            logger.warning(f"Unknown analysis requested: {analysis_name}")

    if not tasks:
        return {}

    # Run ALL analyses in parallel (Decision Point 6)
    # CPU-bound services (audio, deepface) will share CPU cores
    # I/O-bound services benefit from async
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle any exceptions
    final_results = {}
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logger.error(f"Analysis {name} failed: {result}")
            # Store error state but continue with other results
            final_results[name] = {'error': str(result)}
        elif isinstance(result, tuple) and len(result) == 2:
            # Result is (name, data) tuple
            final_results[result[0]] = result[1]
        else:
            # Unexpected format
            logger.warning(f"Unexpected result format from {name}")
            final_results[name] = result

    logger.info(f"Completed {len(final_results)} analyses in parallel")
    return final_results
```

---

### Phase 4: Accessing Gender Data

**Decision Point 12**: This document shows how to access DeepFace data. Actual pitch normalization is in spectralspeech.md.

#### 4.1 Data Access Interface
**File**: `rumiai_v2/processors/temporal_compute.py` (or any consumer)

```python
# Example of accessing DeepFace gender data from ml_data
def any_function_needing_gender(ml_data: Dict[str, Any]):
    """
    DeepFace provides gender data in ml_data['gender_detection']
    """
    # Access the gender detection results
    gender_data = ml_data.get('gender_detection', {})

    # Extract the detected gender
    gender = gender_data.get('gender')
    # Possible values: 'male', 'female', 'multiple_people', or None

    # Get the confidence score
    confidence = gender_data.get('confidence', 0.0)
    # Range: 0.0 to 1.0

    # Check for multi-person scenario
    if gender == 'multiple_people':
        # Multiple people detected - handle accordingly
        # For pitch normalization, this triggers self-normalization
        multi_person_frames = gender_data.get('multi_person_frames', 0)
        logger.info(f"Multiple people in {multi_person_frames} frames")

    # Optional debugging info
    detector_used = gender_data.get('detector_backend')  # e.g., 'opencv'
    processing_ms = gender_data.get('processing_ms')  # e.g., 2500
    frames_analyzed = gender_data.get('frames_analyzed')  # e.g., 5
```

**Note**: For the actual pitch normalization implementation using this gender data, see `spectralspeech.md`.

---

### Phase 5: Testing Strategy

#### 5.1 Unit Test for Service
**File**: `tests/test_deepface_service.py`

```python
import pytest
from rumiai_v2.ml_services.deepface_gender_service import DeepFaceGenderService

@pytest.mark.asyncio
async def test_deepface_schema_compliance():
    """Test schema compliance for gender detection output"""
    # Decision Point 10: Test with custom config
    from rumiai_v2.ml_services.deepface_gender_service import DeepFaceConfig
    test_config = DeepFaceConfig(timeout=5, detector_backend='opencv')
    service = DeepFaceGenderService(config=test_config)
    result = await service.analyze("test_video.mp4")

    # Required fields must exist
    assert 'gender' in result
    assert 'confidence' in result
    assert 0.0 <= result['confidence'] <= 1.0

    # Gender must be valid (mapped from DeepFace's 'Man'/'Woman')
    # Decision Point 6: Also includes 'multiple_people' for multi-person videos
    assert result['gender'] in ['male', 'female', 'multiple_people', None]

@pytest.mark.asyncio
async def test_gender_label_mapping():
    """Test Decision Point 2: Gender label mapping"""
    # Mock DeepFace to return 'Man' or 'Woman'
    # Verify service returns 'male' or 'female'
    service = DeepFaceGenderService()

    # Would need to mock DeepFace.analyze to test mapping
    # Expected: 'Man' -> 'male', 'Woman' -> 'female'

@pytest.mark.asyncio
async def test_error_handling():
    """Test Decision Point 3: Distinguish critical vs expected failures"""
    from rumiai_v2.ml_services.deepface_gender_service import (
        DeepFaceGenderService,
        VideoLoadError,
        ModelInitializationError
    )
    service = DeepFaceGenderService()

    # Test critical error - corrupted video should raise
    with pytest.raises(VideoLoadError):
        await service.analyze("nonexistent_video.mp4")

    # Test expected case - no faces should return empty result
    # Would need to mock DeepFace.analyze to return empty list
    # result = await service.analyze("no_faces_video.mp4")
    # assert result['gender'] is None
    # assert result['confidence'] == 0.0
    # assert 'error' in result

@pytest.mark.asyncio
async def test_configuration():
    """Test Decision Point 10 & 13: Dataclass configuration and validation"""
    import os
    import pytest
    from rumiai_v2.ml_services.deepface_gender_service import DeepFaceConfig

    # Test environment variable loading
    os.environ['DEEPFACE_TIMEOUT'] = '15'
    os.environ['DEEPFACE_DETECTOR'] = 'retinaface'
    config = DeepFaceConfig.from_env()
    assert config.timeout == 15
    assert config.detector_backend == 'retinaface'

    # Test with custom config
    custom_config = DeepFaceConfig(timeout=20, use_gpu=True)
    service = DeepFaceGenderService(config=custom_config)
    assert service.config.timeout == 20
    assert service.config.use_gpu == True

    # Test validation (Decision Point 13)
    with pytest.raises(ValueError, match="Invalid detector_backend"):
        DeepFaceConfig(detector_backend='invalid')

    with pytest.raises(ValueError, match="Timeout must be positive"):
        DeepFaceConfig(timeout=-1)

    with pytest.raises(ValueError, match="Thread workers must be positive"):
        DeepFaceConfig(thread_workers=0)

@pytest.mark.asyncio
async def test_adaptive_frame_sampling():
    """Test Decision Point 9: Adaptive frame count"""
    service = DeepFaceGenderService()  # Uses default config

    # Test frame count calculation
    assert service._calculate_frame_count(3.0) == 2  # Very short
    assert service._calculate_frame_count(10.0) == 3  # Short
    assert service._calculate_frame_count(20.0) == 5  # Medium
    assert service._calculate_frame_count(60.0) == 7  # Long

@pytest.mark.asyncio
async def test_multi_person_handling():
    """Test Decision Point 6: Multi-person detection (conservative)"""
    service = DeepFaceGenderService()

    # Mock scenario: even 1 frame with multiple people triggers fallback
    # result = await service.analyze("multi_person_video.mp4")
    # assert result['gender'] == 'multiple_people'
    # assert result['confidence'] == 0.0
    # assert 'multi_person_frames' in result
    # assert 'self-normalization' in result['note']

@pytest.mark.asyncio
async def test_timeout_handling():
    """Test Decision Point 4: Timeout implementation"""
    from rumiai_v2.ml_services.deepface_gender_service import DeepFaceConfig

    # Test with very short timeout
    config = DeepFaceConfig(timeout=0.1)  # 100ms timeout
    service = DeepFaceGenderService(config=config)

    # Should timeout on real video
    result = await service.analyze("long_video.mp4")
    assert result['gender'] is None
    assert 'timeout' in result.get('error', '')

@pytest.mark.asyncio
async def test_deepface_performance():
    """Test performance benchmarks for DeepFace service"""
    import time

    service = DeepFaceGenderService()  # Default 10s timeout
    start = time.time()
    result = await service.analyze("60_second_video.mp4")
    duration = time.time() - start

    # Should complete in 2-3s (well under 10s timeout)
    # Decision Point 9: Long videos (>30s) use 7 frames
    assert duration < 5.0, f"DeepFace took {duration:.1f}s, expected <5s"
    assert 'processing_ms' in result
```

#### 5.2 Integration Test
```python
@pytest.mark.asyncio
async def test_parallel_execution():
    """Test Decision Point 8: Parallel processing with audio service"""
    analyzer = VideoAnalyzer()

    # Time both services running
    import time
    start = time.time()

    results = await analyzer.run_analysis(
        Path("test_video.mp4"),
        selected_analyses=['audio_energy', 'deepface_gender']
    )

    duration = time.time() - start

    # Should complete in ~10s (max of both services)
    assert duration < 15.0, "Parallel execution too slow"
    assert 'gender_detection' in results
    assert 'audio_energy' in results
```

---

### Phase 6: ~~Batch Processing~~ Single Video Processing

**Decision Point 9**: Videos are processed one at a time, not in batch mode.
- Each video gets its own VideoAnalyzer instance
- No memory accumulation between videos
- Simpler architecture without batch state management
- OpenCV VideoCapture always released after each video
- ThreadPoolExecutor created/destroyed per video (Decision Point 2)

---

## Deployment Checklist

### Pre-Production
- [ ] Install DeepFace package
- [ ] Download models (run test analysis)
- [ ] Verify GPU availability (optional but faster)
- [ ] Create service file
- [ ] Add to video_analyzer
- [ ] Update temporal_compute
- [ ] Run unit tests
- [ ] Run integration tests

### Production Deployment
- [ ] Deploy to staging environment
- [ ] Test with 10 videos
- [ ] Verify parallel execution (<10s for 60s video)
- [ ] Test with 100 videos (processed one at a time)
- [ ] Monitor memory usage (~500MB for model)
- [ ] Deploy to production
- [ ] Monitor error rates (<1%)

### Performance Monitoring
- [ ] Track processing time per video
- [ ] Monitor sequential processing efficiency
- [ ] Check memory stability
- [ ] Log gender detection confidence distribution
- [ ] Track failure rates by video type

---

## Configuration Options

### Environment Variables (Decision Point 10)
```bash
# All configuration via environment variables
DEEPFACE_TIMEOUT=10          # Max seconds for entire analysis - times out as expected failure (default: 10)
DEEPFACE_DETECTOR=opencv     # Face detector: opencv, retinaface, mtcnn, ssd, dlib, mediapipe (default: opencv)
DEEPFACE_ENFORCE=false       # Fail if no faces detected (default: false)
DEEPFACE_USE_GPU=false       # Use GPU if available (default: false)
DEEPFACE_WORKERS=2           # Thread pool workers (default: 2)

# Note: Invalid configuration will raise ValueError on service initialization (Decision Point 13)

# Note: Frame count is adaptive based on video duration (Decision Point 7)
# Not configurable via environment variables
```

### Service Configuration Example (Decision Point 10)
```python
# Use default configuration from environment
service = DeepFaceGenderService()

# Or provide custom configuration for testing
from rumiai_v2.ml_services.deepface_gender_service import DeepFaceConfig

# Decision Point 13: Config validation will catch any invalid values
test_config = DeepFaceConfig(
    timeout=5,
    detector_backend='retinaface',  # Valid backend (validated in __post_init__)
    enforce_detection=True,
    use_gpu=True,
    thread_workers=4
)
service = DeepFaceGenderService(config=test_config)
```

---

## Troubleshooting Guide

### Common Issues (Updated with Decision Points)

| Issue | Solution |
|-------|----------|
| **ImportError: No module named 'deepface'** | Run `pip install deepface` |
| **Too many frames analyzed** | Decision Point 9: Adaptive frame count based on video duration |
| **Same video analyzed multiple times** | Decision Point 10: By design - no caching for consistency |
| **Model download fails** | Check internet connection, manually download to ~/.deepface/weights |
| **OOM errors** | Since we process one video at a time with fresh thread pools (Decision Point 2), this shouldn't occur |
| **Slow processing** | Expected 2-3s on CPU (not the bottleneck vs audio 8-12s) |
| **Event loop blocked** | Thread pool executor prevents blocking (Decision Point 2) |
| **Analysis hangs forever** | Timeout after configured seconds (Decision Point 4) |
| **Gender mismatch** | DeepFace returns 'Man'/'Woman', we map to 'male'/'female' |
| **No faces detected** | Normal for some videos, returns gender=None |
| **Multiple people detected** | Any frame with multiple people triggers self-normalization (Decision Point 6) |
| **TensorFlow warnings** | Can be suppressed with `TF_CPP_MIN_LOG_LEVEL=2` |

### Error Recovery (Decision Point 3: Distinguish Critical vs Expected)
- **Critical Errors** (fail-fast, raise exception):
  - `VideoLoadError`: Cannot open/read video file
  - `ModelInitializationError`: DeepFace model failed to load
  - `FileNotFoundError`: Video file doesn't exist
- **Expected Cases** (return empty result):
  - No faces detected in video
  - DeepFace analysis returns empty
  - Face detection confidence too low
  - Analysis timeout after configured seconds (Decision Point 4)
- Log all errors with appropriate level (error for critical, warning for expected)
- Always return schema-compliant data or raise clear exception

---

## Success Metrics

### Week 1
- [ ] Service installed and tested
- [ ] Parallel execution verified
- [ ] 10 test videos processed successfully

### Week 2
- [ ] 100+ videos processed sequentially
- [ ] Memory usage stable at ~500MB
- [ ] Processing time 2-3s per video (adaptive frames)
- [ ] Integration with calculate_pitch_metrics verified
- [ ] Gender detection accuracy >90% on clear faces
- [ ] Multi-person videos correctly trigger self-normalization

### Month 1
- [ ] 10,000+ videos processed
- [ ] Error rate <1%
- [ ] Pitch normalization using gender data
- [ ] Improved ML model accuracy with normalized features

---

## References

- [DeepFace Documentation](https://github.com/serengil/deepface)
- [Spectral Speech Analysis Design](spectralspeech.md)
- [Post-Refactor Architecture](postrefactorflow.md)
- [Gender Detection Services Analysis](gender_detection_services.md)

---

## Appendix A: Requirements.txt Addition

Add to `requirements_ml.txt`:
```txt
# Gender Detection for Pitch Normalization
# Python 3.12.3 compatible versions
deepface>=0.0.92
tensorflow>=2.16.1,<2.17  # TF 2.16.x for Python 3.12 support
tf-keras>=2.16.0  # Keras backend for TF 2.16+

# Optional: For better performance
# tensorflow[and-cuda]>=2.16.1  # Uncomment for GPU support
```

---

## Appendix B: Quick Start Script

```bash
#!/bin/bash
# quick_start_deepface.sh

echo "Installing DeepFace service..."

# Install package
pip install deepface

# Create service directory
mkdir -p rumiai_v2/ml_services

# Download models
python -c "
from deepface import DeepFace
import numpy as np
test = np.zeros((224,224,3), dtype=np.uint8)
DeepFace.analyze(test, actions=['gender'], enforce_detection=False, silent=True)
print('Models downloaded successfully')
"

# Run basic test
python -c "
from pathlib import Path
import asyncio
from rumiai_v2.ml_services.deepface_gender_service import DeepFaceGenderService

async def test():
    service = DeepFaceGenderService()
    result = await service.analyze('test_video.mp4')
    print(f'Gender: {result.get(\"gender\")}, Confidence: {result.get(\"confidence\")}')

asyncio.run(test())
"

echo "DeepFace service installation complete!"
```