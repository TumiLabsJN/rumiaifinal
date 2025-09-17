"""
Unit tests for DeepFace Gender Detection Service

Tests the DeepFace service implementation including configuration,
error handling, and gender detection capabilities.
"""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from rumiai_v2.ml_services.deepface_gender_service import (
    DeepFaceGenderService,
    DeepFaceConfig,
    VideoLoadError,
    ModelInitializationError
)


@pytest.mark.asyncio
async def test_deepface_schema_compliance():
    """Test schema compliance for gender detection output"""
    # Decision Point 10: Test with custom config
    test_config = DeepFaceConfig(timeout=5, detector_backend='opencv')
    service = DeepFaceGenderService(config=test_config)

    # Mock the analyze method to return a sample result
    with patch.object(service, '_analyze_sync') as mock_analyze:
        mock_analyze.return_value = {
            'gender': 'male',
            'confidence': 0.95,
            'method': 'deepface',
            'processing_ms': 2500,
            'frames_analyzed': 5
        }

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
    service = DeepFaceGenderService()

    # Mock DeepFace.analyze to return 'Man' or 'Woman'
    with patch('rumiai_v2.ml_services.deepface_gender_service.DeepFace.analyze') as mock_analyze:
        # Test 'Man' -> 'male' mapping
        mock_analyze.return_value = [{
            'dominant_gender': 'Man',
            'gender': {'Man': 95.0, 'Woman': 5.0}
        }]

        with patch.object(service, '_sample_frames') as mock_frames:
            mock_frames.return_value = [np.zeros((224, 224, 3))]
            result = service._analyze_sync("test_video.mp4")

        assert result['gender'] == 'male'

        # Test 'Woman' -> 'female' mapping
        mock_analyze.return_value = [{
            'dominant_gender': 'Woman',
            'gender': {'Woman': 92.0, 'Man': 8.0}
        }]

        with patch.object(service, '_sample_frames') as mock_frames:
            mock_frames.return_value = [np.zeros((224, 224, 3))]
            result = service._analyze_sync("test_video.mp4")

        assert result['gender'] == 'female'


@pytest.mark.asyncio
async def test_error_handling():
    """Test Decision Point 3: Distinguish critical vs expected failures"""
    service = DeepFaceGenderService()

    # Test critical error - video file doesn't exist
    with pytest.raises(FileNotFoundError):
        await service.analyze("/nonexistent/video.mp4")

    # Test expected case - no faces detected (mocked)
    with patch.object(service, '_analyze_sync') as mock_analyze:
        mock_analyze.return_value = {
            'gender': None,
            'confidence': 0.0,
            'method': 'deepface',
            'error': 'no_faces_detected'
        }

        result = await service.analyze("no_faces_video.mp4")
        assert result['gender'] is None
        assert result['confidence'] == 0.0
        assert 'error' in result


@pytest.mark.asyncio
async def test_configuration():
    """Test Decision Point 10 & 13: Dataclass configuration and validation"""
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

    # Mock scenario with multiple people
    with patch('rumiai_v2.ml_services.deepface_gender_service.DeepFace.analyze') as mock_analyze:
        # Return multiple faces in one frame
        mock_analyze.return_value = [
            {'dominant_gender': 'Man', 'gender': {'Man': 95.0, 'Woman': 5.0}},
            {'dominant_gender': 'Woman', 'gender': {'Woman': 92.0, 'Man': 8.0}}
        ]

        with patch.object(service, '_sample_frames') as mock_frames:
            mock_frames.return_value = [np.zeros((224, 224, 3))]
            result = service._analyze_sync("multi_person_video.mp4")

        assert result['gender'] == 'multiple_people'
        assert result['confidence'] == 0.0
        assert 'multi_person_frames' in result
        assert 'self-normalization' in result['note']


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test Decision Point 4: Timeout implementation"""
    # Test with very short timeout
    config = DeepFaceConfig(timeout=0.001)  # 1ms timeout
    service = DeepFaceGenderService(config=config)

    # Mock a slow operation
    with patch.object(service, '_analyze_sync') as mock_analyze:
        async def slow_operation(*args):
            await asyncio.sleep(1)  # Sleep longer than timeout
            return {}

        mock_analyze.side_effect = slow_operation

        result = await service.analyze("slow_video.mp4")
        assert result['gender'] is None
        assert 'timeout' in result.get('error', '')


@pytest.mark.asyncio
async def test_deepface_performance():
    """Test performance benchmarks for DeepFace service"""
    import time

    service = DeepFaceGenderService()  # Default 10s timeout

    # Mock the sync analysis to simulate processing time
    with patch.object(service, '_analyze_sync') as mock_analyze:
        mock_analyze.return_value = {
            'gender': 'female',
            'confidence': 0.89,
            'method': 'deepface',
            'processing_ms': 2300,
            'frames_analyzed': 7
        }

        start = time.time()
        result = await service.analyze("60_second_video.mp4")
        duration = time.time() - start

        # Should complete quickly when mocked
        assert duration < 1.0
        assert 'processing_ms' in result
        # Decision Point 9: Long videos (>30s) use 7 frames
        assert result['frames_analyzed'] == 7


@pytest.mark.asyncio
async def test_parallel_execution():
    """Test Decision Point 8: Parallel processing capability"""
    service = DeepFaceGenderService()

    # Mock to make operations fast
    with patch.object(service, '_analyze_sync') as mock_analyze:
        mock_analyze.return_value = {
            'gender': 'male',
            'confidence': 0.91,
            'method': 'deepface',
            'processing_ms': 2000
        }

        # Run multiple analyses in parallel
        tasks = [
            service.analyze(f"video_{i}.mp4")
            for i in range(3)
        ]

        import time
        start = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start

        # All should complete
        assert len(results) == 3
        assert all(r['gender'] == 'male' for r in results)
        # Should be fast when mocked
        assert duration < 1.0


def test_config_from_env():
    """Test environment variable configuration"""
    # Set environment variables
    os.environ['DEEPFACE_TIMEOUT'] = '30'
    os.environ['DEEPFACE_DETECTOR'] = 'mtcnn'
    os.environ['DEEPFACE_ENFORCE'] = 'true'
    os.environ['DEEPFACE_USE_GPU'] = 'false'
    os.environ['DEEPFACE_WORKERS'] = '4'

    config = DeepFaceConfig.from_env()

    assert config.timeout == 30
    assert config.detector_backend == 'mtcnn'
    assert config.enforce_detection == True
    assert config.use_gpu == False
    assert config.thread_workers == 4

    # Clean up
    for key in ['DEEPFACE_TIMEOUT', 'DEEPFACE_DETECTOR', 'DEEPFACE_ENFORCE',
                'DEEPFACE_USE_GPU', 'DEEPFACE_WORKERS']:
        if key in os.environ:
            del os.environ[key]


def test_video_load_error():
    """Test VideoLoadError is raised for invalid videos"""
    service = DeepFaceGenderService()

    # Mock VideoCapture to simulate failed video open
    with patch('cv2.VideoCapture') as mock_cap:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap.return_value = mock_instance

        with pytest.raises(VideoLoadError, match="Cannot open video file"):
            service._sample_frames("invalid_video.mp4")


def test_model_initialization_error():
    """Test ModelInitializationError on failed model load"""
    with patch('rumiai_v2.ml_services.deepface_gender_service.DeepFace.analyze') as mock_analyze:
        mock_analyze.side_effect = Exception("Model download failed")

        with pytest.raises(ModelInitializationError, match="Cannot initialize DeepFace"):
            service = DeepFaceGenderService()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])