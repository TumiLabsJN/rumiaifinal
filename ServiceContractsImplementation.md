# Service Contracts Implementation - RumiAI Python-Only Flow

**Version**: 1.0.0  
**Last Updated**: 2025-01-08  
**Purpose**: Complete implementation details for all service contracts  
**Parent Document**: ServiceContracts.md

## Overview

This document contains the complete, copy-paste ready implementations for all service contracts in the RumiAI Python-only processing pipeline. Each contract follows strict fail-fast principles with zero tolerance for data normalization or graceful degradation.

## Table of Contents

1. [Base Contract Infrastructure](#1-base-contract-infrastructure)
2. [Critical Path Contracts](#2-critical-path-contracts)
3. [ML Service Contracts](#3-ml-service-contracts)
4. [Pipeline Stage Contracts](#4-pipeline-stage-contracts)
5. [Data Transformation Contracts](#5-data-transformation-contracts)
6. [External Service Contracts](#6-external-service-contracts)
7. [Domain-Specific Contract Templates](#7-domain-specific-contract-templates)
8. [Supporting Contracts](#8-supporting-contracts)
9. [Contract Registry Implementation](#9-contract-registry-implementation)

---

## 1. Base Contract Infrastructure

### Required: rumiai_v2/contracts/base.py

This file must be created before any contract implementation:

```python
# rumiai_v2/contracts/base.py
"""
Base classes and exceptions for all service contracts.
This is the SINGLE SOURCE for contract exceptions.
"""

class ServiceContractViolation(Exception):
    """
    Unified exception raised when any contract validation fails.
    
    Args:
        message: Description of the validation failure
        component: Optional component name for better error tracking
        
    Example:
        raise ServiceContractViolation("Invalid frame size", component="YOLO")
        # Output: "[YOLO] Invalid frame size"
    """
    def __init__(self, message: str, component: str = None):
        self.component = component
        super().__init__(f"[{component}] {message}" if component else message)


class ContractValidator:
    """
    Base class for all contract validators.
    All validators should inherit from this class.
    """
    
    @staticmethod
    def validate(data: any) -> None:
        """
        Standard validation method signature.
        
        Raises:
            ServiceContractViolation: On any validation failure
        """
        raise NotImplementedError("Validators must implement validate()")
```

---

## 2. Critical Path Contracts

These contracts are blocking issues that must be implemented first.

### 2.1 FEAT Integration Contracts
## 11. Detailed Contract Implementations

### Base Contract Infrastructure

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/base_contract.py

import os
import re
import logging
import numpy as np
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Union
from datetime import datetime

# Import existing error handling infrastructure
from rumiai_v2.utils.logger import Logger
from rumiai_v2.core.error_handler import RumiAIErrorHandler
from rumiai_v2.core.exceptions import ValidationError, RumiAIError

# Setup module logger
logger = logging.getLogger(__name__)

class ServiceContractViolation(ValidationError):
    """
    Raised when a service contract is violated - FAIL FAST
    Extends ValidationError to integrate with existing error handling
    """
    def __init__(self, message: str, contract_name: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.contract_name = contract_name
        self.context = context or {}
        self.timestamp = datetime.utcnow().isoformat()
        
        # Log the violation with full context
        logger.error(
            f"Service Contract Violation: {message}",
            extra={
                'contract_name': contract_name,
                'context': context,
                'timestamp': self.timestamp
            },
            exc_info=True
        )
        
        # Always create debug dump for contract violations
        # Contract violations are serious enough to warrant full debugging info
        self._create_debug_dump()
    
    def _create_debug_dump(self):
        """Create debug dump for contract violation"""
        try:
            error_handler = RumiAIErrorHandler()
            dump_id = error_handler.create_debug_dump(
                error_type="ServiceContractViolation",
                error_message=str(self),
                context=self.context,
                contract_name=self.contract_name
            )
            logger.info(f"Debug dump created: {dump_id}")
        except Exception as e:
            logger.warning(f"Failed to create debug dump: {e}")

class BaseServiceContract:
    """
    Base class for all service contracts with common validation patterns
    Integrated with RumiAI error logging system
    """
    
    def __init__(self, contract_name: str = None):
        """
        Initialize contract with logging and error handling
        
        Args:
            contract_name: Name of the contract for logging purposes
        """
        self.contract_name = contract_name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.contract_name}")
        self.violation_count = 0
        self.validation_count = 0
        self.debug_dumps_created = []  # Track debug dumps for this contract
        
        # Initialize error handler for this contract
        self.error_handler = RumiAIErrorHandler()
        
        # Log contract initialization
        self.logger.debug(f"Initialized contract: {self.contract_name}")
    
    @staticmethod
    def contract_enforced(contract_name: str, validation_method: str = None, validate_inputs: bool = True, validate_outputs: bool = False):
        """
        Decorator for automatic contract enforcement at function boundaries
        
        Args:
            contract_name: Name of contract in registry
            validation_method: Specific validation method to call (auto-detected if None)
            validate_inputs: Whether to validate function inputs
            validate_outputs: Whether to validate function outputs
        
        Usage:
            @BaseServiceContract.contract_enforced('feat', 'validate_feat_input')
            def process_frames_with_feat(frames, timestamps, duration):
                # Inputs automatically validated before execution
                return feat_results
        """
        def decorator(func):
            from functools import wraps
            from .registry import get_registry
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # Get contract from registry
                    registry = get_registry()
                    contract = registry.get(contract_name)
                    
                    if not contract:
                        # Log warning but don't fail - graceful degradation
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Contract '{contract_name}' not found in registry")
                        return func(*args, **kwargs)
                    
                    # Auto-detect validation method if not specified
                    method_name = validation_method
                    if not method_name:
                        # Try common naming patterns
                        func_name = func.__name__
                        possible_methods = [
                            f'validate_{func_name}_input',
                            f'validate_{func_name}',
                            f'validate_input',
                            'validate'
                        ]
                        
                        for possible in possible_methods:
                            if hasattr(contract, possible):
                                method_name = possible
                                break
                    
                    # Validate inputs before function execution
                    if validate_inputs and method_name:
                        validation_method = getattr(contract, method_name, None)
                        if validation_method:
                            try:
                                # Call validation with function arguments
                                validation_method(*args, **kwargs)
                                contract.logger.debug(f"Input validation passed for {func.__name__}")
                            except Exception as e:
                                contract.logger.error(f"Input validation failed for {func.__name__}: {e}")
                                raise
                    
                    # Execute original function
                    result = func(*args, **kwargs)
                    
                    # Validate outputs if requested
                    if validate_outputs and method_name:
                        output_method_name = method_name.replace('_input', '_output')
                        output_validation = getattr(contract, output_method_name, None)
                        if output_validation:
                            try:
                                output_validation(result)
                                contract.logger.debug(f"Output validation passed for {func.__name__}")
                            except Exception as e:
                                contract.logger.error(f"Output validation failed for {func.__name__}: {e}")
                                raise
                    
                    return result
                    
                except Exception as e:
                    # Re-raise contract violations and other exceptions
                    raise
            
            # Add metadata to decorated function
            wrapper._contract_enforced = True
            wrapper._contract_name = contract_name
            wrapper._validation_method = validation_method
            
            return wrapper
        return decorator
    
    def validate_or_fail(self, condition: bool, message: str, context: Dict[str, Any] = None) -> None:
        """
        Core validation method - fails fast on contract violation
        
        Args:
            condition: Boolean condition to check
            message: Error message if condition fails
            context: Additional context for debugging
        """
        self.validation_count += 1
        
        if not condition:
            self.violation_count += 1
            
            # Log before raising
            self.logger.error(
                f"Contract validation failed in {self.contract_name}: {message}",
                extra={'context': context, 'violation_count': self.violation_count}
            )
            
            # Create debug dump directly for immediate debugging
            try:
                dump_id = self.error_handler.handle_contract_violation(
                    service=self.contract_name,
                    expected="condition to be True",
                    got="False",
                    context=context or {}
                )
                self.debug_dumps_created.append(dump_id)
                self.logger.info(f"Debug dump created for investigation: {dump_id}")
            except Exception as e:
                self.logger.warning(f"Could not create debug dump: {e}")
            
            raise ServiceContractViolation(
                message=f"Contract violation: {message}",
                contract_name=self.contract_name,
                context=context
            )
        
        # Log successful validation at debug level
        self.logger.debug(f"Validation passed: {message[:50]}...")
    
    def validate_type(self, value: Any, expected_type: type, field_name: str) -> None:
        """
        Validate type with descriptive error and logging
        
        Args:
            value: Value to check
            expected_type: Expected type or tuple of types
            field_name: Name of field for error message
        """
        if not isinstance(value, expected_type):
            actual_type = type(value).__name__
            expected = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
            
            self.validate_or_fail(
                False,
                f"{field_name} must be {expected}, got {actual_type}",
                context={
                    'field_name': field_name,
                    'expected_type': expected,
                    'actual_type': actual_type,
                    'value_sample': str(value)[:100] if value else None
                }
            )
    
    def validate_range(self, value: float, min_val: float, max_val: float, field_name: str) -> None:
        """
        Validate numeric range with logging
        
        Args:
            value: Numeric value to check
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            field_name: Name of field for error message
        """
        self.validate_or_fail(
            min_val <= value <= max_val,
            f"{field_name} must be between {min_val} and {max_val}, got {value}",
            context={
                'field_name': field_name,
                'value': value,
                'min_val': min_val,
                'max_val': max_val
            }
        )
    
    def normalize_path(self, path_input: Union[str, Path]) -> Path:
        """
        Standardize path handling - converts strings to Path objects
        
        Args:
            path_input: String path or Path object
            
        Returns:
            Path object
        """
        if isinstance(path_input, str):
            return Path(path_input).expanduser().resolve()
        elif isinstance(path_input, Path):
            return path_input.expanduser().resolve()
        else:
            raise TypeError(f"Expected str or Path, got {type(path_input).__name__}")
    
    def validate_file_exists(self, file_path: Union[str, Path], description: str) -> None:
        """
        Validate file existence with logging
        
        Args:
            file_path: Path to check (str or Path object)
            description: Description of the file for error message
        """
        file_path = self.normalize_path(file_path)
        
        self.validate_or_fail(
            file_path.exists(),
            f"{description} not found: {file_path}",
            context={
                'file_path': str(file_path),
                'description': description,
                'absolute_path': str(file_path.absolute()) if file_path else None
            }
        )
    
    def validate_not_empty(self, value: Any, field_name: str) -> None:
        """
        Validate non-empty value with logging
        
        Args:
            value: Value to check for emptiness
            field_name: Name of field for error message
        """
        self.validate_or_fail(
            bool(value),
            f"{field_name} cannot be empty",
            context={
                'field_name': field_name,
                'value_type': type(value).__name__,
                'value_length': len(value) if hasattr(value, '__len__') else None
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics for monitoring
        
        Returns:
            Dictionary with validation and violation counts
        """
        return {
            'contract_name': self.contract_name,
            'validation_count': self.validation_count,
            'violation_count': self.violation_count,
            'success_rate': (self.validation_count - self.violation_count) / self.validation_count 
                          if self.validation_count > 0 else 1.0,
            'debug_dumps_created': len(self.debug_dumps_created),
            'last_debug_dump': self.debug_dumps_created[-1] if self.debug_dumps_created else None
        }
    
    def log_stats(self):
        """Log contract validation statistics"""
        stats = self.get_stats()
        self.logger.info(
            f"Contract stats for {self.contract_name}: "
            f"{stats['validation_count']} validations, "
            f"{stats['violation_count']} violations, "
            f"{stats['success_rate']:.2%} success rate, "
            f"{stats['debug_dumps_created']} debug dumps created"
        )
        if stats['last_debug_dump']:
            self.logger.info(f"Last debug dump for investigation: {stats['last_debug_dump']}")
```

### Automatic Contract Enforcement

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/base_contract.py (continued)

class ValidationContext:
    """Context manager for automatic contract validation scopes"""
    
    def __init__(self, contract_name: str, operation: str = None):
        self.contract_name = contract_name
        self.operation = operation
        self.contract = None
        self.validation_stack = []
    
    def __enter__(self):
        from .registry import get_registry
        
        registry = get_registry()
        self.contract = registry.get(self.contract_name)
        
        if self.contract:
            self.contract.logger.debug(f"Entering validation context: {self.operation or 'unnamed'}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.contract:
            if exc_type is None:
                self.contract.logger.debug(f"Validation context completed: {self.operation or 'unnamed'}")
            else:
                self.contract.logger.error(f"Validation context failed: {self.operation or 'unnamed'} - {exc_val}")
        return False  # Don't suppress exceptions
    
    def validate_step(self, step_name: str, *args, **kwargs):
        """Validate a specific step within the context"""
        if self.contract:
            validation_method = f'validate_{step_name}'
            method = getattr(self.contract, validation_method, None)
            if method:
                method(*args, **kwargs)
                self.validation_stack.append(step_name)


# Usage Examples for Automatic Contract Enforcement
"""
Example 1: Decorator-based automatic validation

from rumiai_v2.contracts.base_contract import BaseServiceContract

class VideoProcessor:
    @BaseServiceContract.contract_enforced('feat', 'validate_feat_input')
    def process_emotions(self, frames, timestamps, duration):
        # Inputs automatically validated before execution
        # No manual contract calls needed!
        return self.feat_detector.detect_emotions(frames)
    
    @BaseServiceContract.contract_enforced('frame_manager', validate_inputs=True, validate_outputs=True)
    def extract_frames(self, video_path: Union[str, Path]):
        # Both input and output validation automatic
        return self.frame_extractor.extract(video_path)

Example 2: Context manager approach

from rumiai_v2.contracts.base_contract import ValidationContext

def process_video_pipeline(video_url):
    with ValidationContext('orchestrator', 'video_processing') as ctx:
        # All operations in this block are monitored
        
        ctx.validate_step('pipeline_input', video_url)
        video_data = scrape_video(video_url)
        
        ctx.validate_step('video_file', video_data['file_path'])
        frames = extract_frames(video_data['file_path'])
        
        with ValidationContext('feat', 'emotion_detection') as feat_ctx:
            feat_ctx.validate_step('feat_input', frames, timestamps, duration)
            emotions = detect_emotions(frames)
            feat_ctx.validate_step('feat_output', emotions)
        
        return build_timeline(emotions, video_data)

Example 3: Function registration with auto-validation (Future Enhancement)

registry = get_registry()
registry.register_function(
    'process_frames_with_feat',
    process_frames_with_feat,
    input_contract='feat.validate_feat_input',
    output_contract='feat.validate_feat_output'
)

# All calls automatically validated
result = registry.call_function('process_frames_with_feat', frames, timestamps, duration)
"""
```

---

## P0 CRITICAL: FEAT Emotion Detection Contracts

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/feat_contracts.py

import os
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base_contract import BaseServiceContract, ServiceContractViolation

# Setup module logger
logger = logging.getLogger(__name__)

class FEATServiceContract(BaseServiceContract):
    """Service contracts for FEAT emotion detection integration"""
    
    # Valid emotions that FEAT can output
    VALID_FEAT_EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    # Mapping to RumiAI emotion names
    EMOTION_MAPPING = {
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'happiness': 'joy',  # Map happiness to joy
        'sadness': 'sadness',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }
    
    # Frame sampling tolerance - allows variance to account for:
    # - Rounding errors in FPS calculations
    # - Variable frame rates in source videos  
    # - Frame extraction timing inconsistencies
    FRAME_COUNT_TOLERANCE = 0.1  # 10% tolerance
    
    def __init__(self):
        """Initialize FEAT contract with logging"""
        super().__init__(contract_name="FEATServiceContract")
    
    def validate_feat_initialization(self) -> None:
        """
        Validate FEAT can be initialized - fail fast if not available
        Called once at startup
        """
        self.logger.info("Validating FEAT initialization...")
        
        # Check FEAT is installed
        try:
            import feat
            self.logger.debug("FEAT library import successful")
        except ImportError as e:
            self.validate_or_fail(
                False,
                "FEAT not installed. Install with: pip install py-feat==0.6.0",
                context={'error': str(e), 'install_command': 'pip install py-feat==0.6.0'}
            )
        
        # Check model directory exists
        feat_model_dir = Path.home() / '.feat' / 'models'
        if not feat_model_dir.exists():
            # Models will be downloaded on first run, this is OK
            self.logger.info(f"FEAT models will be downloaded to {feat_model_dir}")
        else:
            self.logger.debug(f"FEAT models found at {feat_model_dir}")
        
        # Check GPU availability matches configuration
        if os.getenv('USE_GPU_FOR_FEAT', 'true').lower() == 'true':
            try:
                import torch
                if not torch.cuda.is_available():
                    self.logger.warning("GPU requested for FEAT but not available, falling back to CPU")
            except ImportError:
                self.logger.warning("PyTorch not available for GPU check")
        
        # Try to initialize detector to verify it works
        try:
            from feat import Detector
            detector = Detector(
                face_model='retinaface',
                landmark_model='mobilefacenet', 
                au_model='xgb',
                emotion_model='resmasknet',
                device='cpu'  # Use CPU for validation
            )
            del detector  # Clean up
            self.logger.debug("FEAT detector test initialization successful")
        except Exception as e:
            self.validate_or_fail(
                False,
                f"FEAT detector initialization failed: {e}",
                context={'error': str(e), 'action': 'initialize_detector'}
            )
    
    def validate_feat_input(self, 
                           frames: List[np.ndarray],
                           timestamps: List[float],
                           video_duration: float,
                           mediapipe_data: Dict[str, Any] = None) -> None:
        """
        Validate input data for FEAT processing
        Called before each FEAT detection batch
        
        Args:
            frames: Video frames for processing
            timestamps: Frame timestamps
            video_duration: Total video duration
            mediapipe_data: Optional MediaPipe face detection data for validation
        """
        # Validate frames list
        self.validate_not_empty(frames, "frames")
        self.validate_type(frames, list, "frames")
        
        # Validate timestamps
        self.validate_not_empty(timestamps, "timestamps")
        self.validate_type(timestamps, list, "timestamps")
        
        # Check frames and timestamps match
        if len(frames) != len(timestamps):
            self.validate_or_fail(
                False,
                f"Frames count ({len(frames)}) must match timestamps count ({len(timestamps)})",
                context={'frames_count': len(frames), 'timestamps_count': len(timestamps)}
            )
        
        # Validate each frame
        for i, frame in enumerate(frames):
            if not isinstance(frame, np.ndarray):
                self.validate_or_fail(
                    False,
                    f"Frame {i} must be numpy array, got {type(frame)}",
                    context={'frame_index': i, 'actual_type': type(frame).__name__}
                )
            
            # Check frame dimensions (should be HxWxC)
            if len(frame.shape) != 3:
                self.validate_or_fail(
                    False,
                    f"Frame {i} must be 3D array (HxWxC), got shape {frame.shape}",
                    context={'frame_index': i, 'shape': frame.shape}
                )
            
            # Check color channels (should be 3 for BGR/RGB)
            if frame.shape[2] != 3:
                self.validate_or_fail(
                    False,
                    f"Frame {i} must have 3 color channels, got {frame.shape[2]}",
                    context={'frame_index': i, 'channels': frame.shape[2], 'expected': 3}
                )
            
            # Check frame is not empty
            if frame.size == 0:
                self.validate_or_fail(
                    False,
                    f"Frame {i} is empty",
                    context={'frame_index': i}
                )
        
        # Validate timestamps are within video duration
        for i, ts in enumerate(timestamps):
            self.validate_range(ts, 0, video_duration, f"timestamp[{i}]")
        
        # Validate timestamps are monotonically increasing
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                self.validate_or_fail(
                    False,
                    f"Timestamps must be increasing: {timestamps[i-1]} -> {timestamps[i]}",
                    context={'index': i, 'previous': timestamps[i-1], 'current': timestamps[i]}
                )
        
        # Validate adaptive sampling rate
        sample_rate = self._calculate_sample_rate(video_duration)
        expected_max_frames = int(video_duration * sample_rate)
        max_allowed_frames = int(expected_max_frames * (1 + self.FRAME_COUNT_TOLERANCE))
        
        if len(frames) > max_allowed_frames:
            self.validate_or_fail(
                False,
                f"Too many frames ({len(frames)}) for {video_duration}s video. "
                f"Expected ~{expected_max_frames} at {sample_rate} FPS, "
                f"allowing {self.FRAME_COUNT_TOLERANCE*100:.0f}% tolerance = max {max_allowed_frames}",
                context={
                    'frame_count': len(frames),
                    'expected_max': expected_max_frames,
                    'max_allowed': max_allowed_frames,
                    'tolerance_percent': self.FRAME_COUNT_TOLERANCE * 100,
                    'sample_rate': sample_rate,
                    'video_duration': video_duration
                }
            )
        
        # If MediaPipe face data is provided, validate it for FEAT compatibility
        if mediapipe_data is not None:
            self.logger.debug("Validating MediaPipe face data for FEAT integration...")
            self.validate_mediapipe_face_input(mediapipe_data, frames)
    
    def validate_mediapipe_face_input(self, mediapipe_data: Dict[str, Any], frames: List[np.ndarray]) -> None:
        """
        Validate MediaPipe face detection data before FEAT processing
        This ensures MediaPipe provides the required face data structure for FEAT
        
        Args:
            mediapipe_data: Output from MediaPipe face detection
            frames: Video frames that were processed
        """
        self.logger.debug("Validating MediaPipe face data for FEAT integration...")
        
        # Basic structure validation
        self.validate_type(mediapipe_data, dict, "mediapipe_data")
        
        # Required top-level fields from MediaPipe
        required_fields = ['faces', 'success', 'frame_count']
        for field in required_fields:
            if field not in mediapipe_data:
                self.validate_or_fail(
                    False,
                    f"MediaPipe data missing required field: {field}",
                    context={'field': field, 'available_fields': list(mediapipe_data.keys())}
                )
        
        # Validate success status
        if not mediapipe_data.get('success', False):
            self.validate_or_fail(
                False,
                "MediaPipe face detection failed",
                context={'mediapipe_data': mediapipe_data}
            )
        
        # Validate faces structure
        faces = mediapipe_data['faces']
        self.validate_type(faces, list, "faces")
        
        # Validate frame count consistency
        expected_frame_count = len(frames)
        actual_frame_count = mediapipe_data.get('frame_count', 0)
        if actual_frame_count != expected_frame_count:
            self.validate_or_fail(
                False,
                f"MediaPipe frame count ({actual_frame_count}) doesn't match input frames ({expected_frame_count})",
                context={'expected': expected_frame_count, 'actual': actual_frame_count}
            )
        
        # Validate each face detection entry
        for i, face_entry in enumerate(faces):
            self._validate_face_detection_entry(face_entry, i, frames[i] if i < len(frames) else None)
    
    def _validate_face_detection_entry(self, face_entry: Dict[str, Any], frame_index: int, frame: np.ndarray) -> None:
        """
        Validate individual face detection entry from MediaPipe
        
        Args:
            face_entry: Single frame's face detection data
            frame_index: Index of the frame
            frame: The actual frame array for bounds checking
        """
        self.validate_type(face_entry, dict, f"faces[{frame_index}]")
        
        # Required fields for each face entry
        required_fields = ['timestamp', 'detections']
        for field in required_fields:
            if field not in face_entry:
                self.validate_or_fail(
                    False,
                    f"Face entry {frame_index} missing field: {field}",
                    context={'frame_index': frame_index, 'field': field, 'available_fields': list(face_entry.keys())}
                )
        
        # Validate timestamp
        timestamp = face_entry['timestamp']
        self.validate_type(timestamp, (int, float), f"faces[{frame_index}].timestamp")
        if timestamp < 0:
            self.validate_or_fail(
                False,
                f"Face entry {frame_index} has negative timestamp: {timestamp}",
                context={'frame_index': frame_index, 'timestamp': timestamp}
            )
        
        # Validate detections
        detections = face_entry['detections']
        self.validate_type(detections, list, f"faces[{frame_index}].detections")
        
        # Each detection should have proper structure
        for j, detection in enumerate(detections):
            self._validate_face_detection(detection, frame_index, j, frame)
    
    def _validate_face_detection(self, detection: Dict[str, Any], frame_index: int, detection_index: int, frame: np.ndarray) -> None:
        """
        Validate individual face detection from MediaPipe
        
        Args:
            detection: Single face detection
            frame_index: Frame index
            detection_index: Detection index within frame
            frame: Frame array for bounds checking
        """
        detection_id = f"faces[{frame_index}].detections[{detection_index}]"
        self.validate_type(detection, dict, detection_id)
        
        # Required fields for FEAT processing
        required_fields = ['bbox', 'confidence']
        for field in required_fields:
            if field not in detection:
                self.validate_or_fail(
                    False,
                    f"Detection {detection_id} missing field: {field}",
                    context={'detection_id': detection_id, 'field': field, 'available_fields': list(detection.keys())}
                )
        
        # Validate confidence score
        confidence = detection['confidence']
        self.validate_range(confidence, 0.0, 1.0, f"{detection_id}.confidence")
        
        # Validate bounding box
        bbox = detection['bbox']
        self._validate_face_bbox(bbox, detection_id, frame)
        
        # Optional: Validate landmarks if present (FEAT can use them)
        if 'landmarks' in detection:
            landmarks = detection['landmarks']
            self.validate_type(landmarks, list, f"{detection_id}.landmarks")
            
            # MediaPipe face landmarks should be 468 points
            expected_landmark_count = 468
            if len(landmarks) != expected_landmark_count:
                self.logger.warning(
                    f"{detection_id} has {len(landmarks)} landmarks, expected {expected_landmark_count}"
                )
    
    def _validate_face_bbox(self, bbox: List[float], detection_id: str, frame: np.ndarray) -> None:
        """
        Validate face bounding box coordinates
        
        Args:
            bbox: Bounding box [x, y, width, height]
            detection_id: Detection identifier for error messages
            frame: Frame array for bounds checking
        """
        self.validate_type(bbox, list, f"{detection_id}.bbox")
        
        # Should have 4 coordinates
        if len(bbox) != 4:
            self.validate_or_fail(
                False,
                f"{detection_id}.bbox must have 4 coordinates [x, y, width, height], got {len(bbox)}",
                context={'detection_id': detection_id, 'bbox_length': len(bbox), 'bbox': bbox}
            )
        
        x, y, width, height = bbox
        
        # Validate coordinate types
        for i, coord in enumerate(['x', 'y', 'width', 'height']):
            if not isinstance(bbox[i], (int, float)):
                self.validate_or_fail(
                    False,
                    f"{detection_id}.bbox.{coord} must be numeric, got {type(bbox[i])}",
                    context={'detection_id': detection_id, 'coordinate': coord, 'value': bbox[i]}
                )
        
        # Validate positive dimensions
        if width <= 0 or height <= 0:
            self.validate_or_fail(
                False,
                f"{detection_id}.bbox has invalid dimensions: width={width}, height={height}",
                context={'detection_id': detection_id, 'width': width, 'height': height}
            )
        
        # Validate bbox is within frame bounds
        if frame is not None:
            frame_height, frame_width = frame.shape[:2]
            
            # Check bbox doesn't exceed frame boundaries
            if x < 0 or y < 0 or (x + width) > frame_width or (y + height) > frame_height:
                self.validate_or_fail(
                    False,
                    f"{detection_id}.bbox [{x}, {y}, {width}, {height}] exceeds frame bounds [{frame_width}x{frame_height}]",
                    context={
                        'detection_id': detection_id,
                        'bbox': [x, y, width, height],
                        'frame_dimensions': [frame_width, frame_height],
                        'x_exceeds': x < 0 or (x + width) > frame_width,
                        'y_exceeds': y < 0 or (y + height) > frame_height
                    }
                )
    
    def validate_feat_output(self, feat_results: Dict[str, Any]) -> None:
        """
        Validate FEAT detection output
        Called after each FEAT detection
        """
        self.validate_type(feat_results, dict, "feat_results")
        
        # Required fields in FEAT output
        required_fields = ['emotions', 'processing_stats', 'timeline']
        for field in required_fields:
            if field not in feat_results:
                self.validate_or_fail(
                    False,
                    f"Missing required field: {field}",
                    context={'field': field, 'available_fields': list(feat_results.keys())}
                )
        
        # Validate emotions list
        emotions = feat_results.get('emotions', [])
        self.validate_type(emotions, list, "emotions")
        
        for i, emotion_data in enumerate(emotions):
            self.validate_type(emotion_data, dict, f"emotions[{i}]")
            
            # Check required emotion fields
            if 'timestamp' not in emotion_data:
                self.validate_or_fail(
                    False,
                    f"emotions[{i}] missing timestamp",
                    context={'index': i, 'emotion_data': emotion_data}
                )
            if 'emotion' not in emotion_data:
                self.validate_or_fail(
                    False,
                    f"emotions[{i}] missing emotion",
                    context={'index': i, 'emotion_data': emotion_data}
                )
            if 'confidence' not in emotion_data:
                self.validate_or_fail(
                    False,
                    f"emotions[{i}] missing confidence",
                    context={'index': i, 'emotion_data': emotion_data}
                )
            
            # Validate emotion value
            emotion = emotion_data['emotion']
            if emotion not in self.VALID_FEAT_EMOTIONS:
                self.validate_or_fail(
                    False,
                    f"Invalid emotion '{emotion}'. Must be one of {self.VALID_FEAT_EMOTIONS}",
                    context={'emotion': emotion, 'valid_emotions': self.VALID_FEAT_EMOTIONS}
                )
            
            # Validate confidence range
            confidence = emotion_data['confidence']
            self.validate_range(confidence, 0.0, 1.0, f"emotions[{i}].confidence")
        
        # Validate processing stats
        stats = feat_results.get('processing_stats', {})
        self.validate_type(stats, dict, "processing_stats")
        
        if 'video_type' not in stats:
            self.validate_or_fail(
                False,
                "processing_stats missing video_type",
                context={'stats': stats}
            )
        
        valid_video_types = ['people_detected', 'no_people', 'feat_unavailable']
        if stats['video_type'] not in valid_video_types:
            self.validate_or_fail(
                False,
                f"Invalid video_type '{stats['video_type']}'. Must be one of {valid_video_types}",
                context={'video_type': stats['video_type'], 'valid_types': valid_video_types}
            )
        
        # Validate timeline format
        timeline = feat_results.get('timeline', {})
        self.validate_type(timeline, dict, "timeline")
        
        # Each timeline entry should be "X-Ys" format
        for time_key, entry in timeline.items():
            if not re.match(r'^\d+-\d+s$', time_key):
                self.validate_or_fail(
                    False,
                    f"Invalid timeline key format: {time_key}. Expected 'X-Ys'",
                    context={'time_key': time_key}
                )
            
            self.validate_type(entry, dict, f"timeline[{time_key}]")
            
            # If face detected, must have emotion and confidence
            if not entry.get('no_face', False):
                if 'emotion' not in entry or 'confidence' not in entry:
                    self.validate_or_fail(
                        False,
                        f"timeline[{time_key}] must have emotion and confidence",
                        context={'time_key': time_key, 'entry': entry}
                    )
    
    def validate_expression_timeline(self, expression_timeline: Dict[str, Dict]) -> None:
        """
        Validate expression timeline format for precompute functions
        This is the final output format expected by emotional journey analysis
        """
        self.validate_type(expression_timeline, dict, "expression_timeline")
        
        for time_key, entry in expression_timeline.items():
            # Validate time key format
            if not re.match(r'^\d+-\d+s$', time_key):
                self.validate_or_fail(
                    False,
                    f"Invalid time key format: {time_key}. Expected 'X-Ys'",
                    context={'time_key': time_key}
                )
            
            self.validate_type(entry, dict, f"expression_timeline[{time_key}]")
            
            # Required fields
            if 'emotion' not in entry:
                self.validate_or_fail(
                    False,
                    f"{time_key} missing 'emotion' field",
                    context={'time_key': time_key, 'entry': entry}
                )
            if 'confidence' not in entry:
                self.validate_or_fail(
                    False,
                    f"{time_key} missing 'confidence' field",
                    context={'time_key': time_key, 'entry': entry}
                )
            
            # Validate emotion is in RumiAI format (after mapping)
            valid_rumiai_emotions = list(self.EMOTION_MAPPING.values())
            if entry['emotion'] not in valid_rumiai_emotions:
                self.validate_or_fail(
                    False,
                    f"Invalid emotion '{entry['emotion']}' in {time_key}. "
                    f"Must be one of {valid_rumiai_emotions}",
                    context={'time_key': time_key, 'emotion': entry['emotion'], 'valid_emotions': valid_rumiai_emotions}
                )
            
            # Validate confidence
            self.validate_range(entry['confidence'], 0.0, 1.0, f"{time_key}.confidence")
    
    def _calculate_sample_rate(self, video_duration: float) -> float:
        """Calculate adaptive sample rate based on video duration"""
        self.logger.debug(f"Calculating sample rate for {video_duration}s video")
        if video_duration <= 30:
            return 2.0  # 2 FPS for short videos
        elif video_duration <= 60:
            return 1.0  # 1 FPS for medium videos
        else:
            return 0.5  # 0.5 FPS for long videos
```

---

## CRITICAL: Core Pipeline Contracts

### 1. Main Orchestrator Contract

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/orchestrator_contracts.py

import re
import os
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from .base_contract import BaseServiceContract, ServiceContractViolation

class OrchestratorContract(BaseServiceContract):
    """Service contracts for main pipeline orchestrator"""
    
    def __init__(self):
        """Initialize orchestrator contract with logging"""
        super().__init__(contract_name="OrchestratorContract")
    
    # Valid TikTok URL patterns
    TIKTOK_URL_PATTERNS = [
        r'https?://(?:www\.)?tiktok\.com/@[\w\.-]+/video/\d+',
        r'https?://(?:www\.)?tiktok\.com/t/[\w]+',
        r'https?://vm\.tiktok\.com/[\w]+',
    ]
    
    def validate_pipeline_input(self, video_input: str) -> Dict[str, Any]:
        """
        Validate main pipeline input - video URL or ID
        Returns validated input type and extracted data
        """
        self.validate_not_empty(video_input, "video_input")
        self.validate_type(video_input, str, "video_input")
        
        # Check if it's a URL or video ID
        if video_input.startswith('http'):
            return self._validate_video_url(video_input)
        else:
            return self._validate_video_id(video_input)
    
    def _validate_video_url(self, url: str) -> Dict[str, Any]:
        """Validate TikTok video URL"""
        # Check URL format
        is_valid = any(re.match(pattern, url) for pattern in self.TIKTOK_URL_PATTERNS)
        if not is_valid:
            self.validate_or_fail(
                False,
                f"Invalid TikTok URL format: {url}\n"
                f"Expected format: https://www.tiktok.com/@username/video/VIDEO_ID",
                context={'url': url, 'expected_patterns': self.TIKTOK_URL_PATTERNS}
            )
        
        # Parse URL
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https']:
            self.validate_or_fail(
                False,
                f"URL must use http or https scheme: {url}",
                context={'url': url, 'scheme': parsed.scheme}
            )
        
        # Extract video ID if possible
        video_id_match = re.search(r'/video/(\d+)', url)
        video_id = video_id_match.group(1) if video_id_match else None
        
        return {
            'input_type': 'url',
            'url': url,
            'video_id': video_id,
            'validated': True
        }
    
    def _validate_video_id(self, video_id: str) -> Dict[str, Any]:
        """Validate TikTok video ID"""
        # TikTok video IDs are typically 19 digits
        if not re.match(r'^\d{17,21}$', video_id):
            self.validate_or_fail(
                False,
                f"Invalid video ID format: {video_id}. Expected 17-21 digit number",
                context={'video_id': video_id, 'expected_format': '17-21 digit number'}
            )
        
        return {
            'input_type': 'id',
            'video_id': video_id,
            'validated': True
        }
    
    def validate_pipeline_configuration(self) -> None:
        """
        Validate Python-only pipeline configuration
        Called at startup to ensure all required flags are set
        """
        # Check Python-only mode is enabled
        if os.getenv('USE_PYTHON_ONLY_PROCESSING', 'false').lower() != 'true':
            self.validate_or_fail(
                False,
                "USE_PYTHON_ONLY_PROCESSING must be true for Python-only flow",
                context={'current_value': os.getenv('USE_PYTHON_ONLY_PROCESSING', 'false'), 'required': 'true'}
            )
        
        if os.getenv('USE_ML_PRECOMPUTE', 'false').lower() != 'true':
            self.validate_or_fail(
                False,
                "USE_ML_PRECOMPUTE must be true for Python-only flow",
                context={'current_value': os.getenv('USE_ML_PRECOMPUTE', 'false'), 'required': 'true'}
            )
        
        # Check all precompute flags are enabled
        required_precompute_flags = [
            'PRECOMPUTE_CREATIVE_DENSITY',
            'PRECOMPUTE_EMOTIONAL_JOURNEY',
            'PRECOMPUTE_PERSON_FRAMING',
            'PRECOMPUTE_SCENE_PACING',
            'PRECOMPUTE_SPEECH_ANALYSIS',
            'PRECOMPUTE_VISUAL_OVERLAY',
            'PRECOMPUTE_METADATA'
        ]
        
        for flag in required_precompute_flags:
            if os.getenv(flag, 'false').lower() != 'true':
                self.validate_or_fail(
                    False,
                    f"{flag} must be true for complete Python-only analysis",
                    context={'flag': flag, 'current_value': os.getenv(flag, 'false'), 'required': 'true'}
                )
        
        # Verify output directory is writable
        output_dir = Path('insights')
        try:
            output_dir.mkdir(exist_ok=True)
            test_file = output_dir / '.write_test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            self.validate_or_fail(
                False,
                f"Cannot write to output directory: {e}",
                context={'error': str(e), 'output_dir': str(output_dir)}
            )
    
    def validate_pipeline_state(self, state: Dict[str, Any]) -> None:
        """
        Validate pipeline state during execution
        Called at key checkpoints
        """
        self.validate_type(state, dict, "pipeline_state")
        
        # Check required state fields
        required_fields = ['video_id', 'stage', 'ml_complete', 'precompute_complete']
        for field in required_fields:
            if field not in state:
                self.validate_or_fail(
                    False,
                    f"Pipeline state missing {field}",
                    context={'field': field, 'available_fields': list(state.keys())}
                )
        
        # Validate stage progression
        valid_stages = [
            'initialized', 'downloading', 'extracting_frames', 
            'ml_processing', 'building_timeline', 'precomputing', 'complete'
        ]
        if state['stage'] not in valid_stages:
            self.validate_or_fail(
                False,
                f"Invalid pipeline stage: {state['stage']}. Must be one of {valid_stages}",
                context={'stage': state['stage'], 'valid_stages': valid_stages}
            )
        
        # ML must complete before precompute
        if state['precompute_complete'] and not state['ml_complete']:
            self.validate_or_fail(
                False,
                "Cannot complete precompute before ML processing",
                context={'precompute_complete': state['precompute_complete'], 'ml_complete': state['ml_complete']}
            )
```

### 2. Frame Manager Contract

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/frame_contracts.py

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import cv2
from .base_contract import BaseServiceContract, ServiceContractViolation

class FrameManagerContract(BaseServiceContract):
    """Service contracts for unified frame manager"""
    
    def __init__(self):
        """Initialize frame manager contract with logging"""
        super().__init__(contract_name="FrameManagerContract")
    
    # Maximum video duration in seconds (2 hours)
    MAX_VIDEO_DURATION = 7200
    
    # Maximum frames to extract
    MAX_FRAMES = 1000
    
    def validate_video_input(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate video file before frame extraction
        Returns video metadata
        """
        # Normalize path input
        video_path = self.normalize_path(video_path)
        
        # Check file exists
        self.validate_file_exists(video_path, "Video file")
        
        # Check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
        if video_path.suffix.lower() not in valid_extensions:
            self.validate_or_fail(
                False,
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported: {valid_extensions}",
                context={'suffix': video_path.suffix, 'valid_extensions': valid_extensions}
            )
        
        # Check file size (max 2GB)
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 2048:
            self.validate_or_fail(
                False,
                f"Video file too large: {file_size_mb:.1f}MB. Maximum: 2048MB",
                context={'file_size_mb': file_size_mb, 'max_size_mb': 2048}
            )
        
        # Open video to get metadata
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.validate_or_fail(
                    False,
                    f"Cannot open video file: {video_path}",
                    context={'video_path': str(video_path)}
                )
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
        except Exception as e:
            self.validate_or_fail(
                False,
                f"Error reading video metadata: {e}",
                context={'error': str(e), 'video_path': str(video_path)}
            )
        
        # Validate video properties
        if fps <= 0:
            self.validate_or_fail(False, "Invalid video FPS", context={'fps': fps})
        if frame_count <= 0:
            self.validate_or_fail(False, "Video has no frames", context={'frame_count': frame_count})
        if width <= 0 or height <= 0:
            self.validate_or_fail(False, "Invalid video dimensions", context={'width': width, 'height': height})
        if duration > self.MAX_VIDEO_DURATION:
            self.validate_or_fail(
                False,
                f"Video too long: {duration:.1f}s. Maximum: {self.MAX_VIDEO_DURATION}s",
                context={'duration': duration, 'max_duration': self.MAX_VIDEO_DURATION}
            )
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'file_size_mb': file_size_mb
        }
    
    def validate_extraction_params(self, params: Dict[str, Any], video_metadata: Dict) -> None:
        """
        Validate frame extraction parameters
        """
        self.validate_type(params, dict, "extraction_params")
        
        # Validate service-specific requirements
        if 'yolo_frames' in params:
            yolo_frames = params['yolo_frames']
            if not 1 <= yolo_frames <= 200:
                self.validate_or_fail(
                    False,
                    f"YOLO frames must be 1-200, got {yolo_frames}",
                    context={'yolo_frames': yolo_frames, 'min': 1, 'max': 200}
                )
        
        if 'mediapipe_sample_rate' in params:
            sample_rate = params['mediapipe_sample_rate']
            if not 0.1 <= sample_rate <= 30:
                self.validate_or_fail(
                    False,
                    f"MediaPipe sample rate must be 0.1-30 FPS, got {sample_rate}",
                    context={'sample_rate': sample_rate, 'min': 0.1, 'max': 30}
                )
        
        if 'ocr_frames' in params:
            ocr_frames = params['ocr_frames']
            if not 1 <= ocr_frames <= 100:
                self.validate_or_fail(
                    False,
                    f"OCR frames must be 1-100, got {ocr_frames}",
                    context={'ocr_frames': ocr_frames, 'min': 1, 'max': 100}
                )
        
        # Calculate total frames to extract
        total_frames = 0
        if 'yolo_frames' in params:
            total_frames += params['yolo_frames']
        if 'mediapipe_sample_rate' in params:
            total_frames += int(video_metadata['duration'] * params['mediapipe_sample_rate'])
        if 'ocr_frames' in params:
            total_frames += params['ocr_frames']
        
        if total_frames > self.MAX_FRAMES:
            self.validate_or_fail(
                False,
                f"Total frames to extract ({total_frames}) exceeds maximum ({self.MAX_FRAMES})",
                context={'total_frames': total_frames, 'max_frames': self.MAX_FRAMES}
            )
    
    def validate_frame_output(self, frame_data: Dict[str, Any]) -> None:
        """
        Validate extracted frame data
        """
        self.validate_type(frame_data, dict, "frame_data")
        
        # Check required fields
        required_fields = ['success', 'frames', 'metadata']
        for field in required_fields:
            if field not in frame_data:
                self.validate_or_fail(
                    False,
                    f"Frame data missing {field}",
                    context={'field': field, 'available_fields': list(frame_data.keys())}
                )
        
        if not frame_data['success']:
            if 'error' in frame_data:
                self.validate_or_fail(
                    False,
                    f"Frame extraction failed: {frame_data['error']}",
                    context={'error': frame_data['error']}
                )
            else:
                self.validate_or_fail(
                    False,
                    "Frame extraction failed with no error message",
                    context={'frame_data': frame_data}
                )
        
        # Validate frames
        frames = frame_data['frames']
        self.validate_type(frames, list, "frames")
        self.validate_not_empty(frames, "frames")
        
        for i, frame in enumerate(frames):
            if not isinstance(frame, np.ndarray):
                self.validate_or_fail(
                    False,
                    f"Frame {i} must be numpy array, got {type(frame)}",
                    context={'frame_index': i, 'actual_type': type(frame).__name__}
                )
            
            # Check dimensions
            if len(frame.shape) != 3:
                self.validate_or_fail(
                    False,
                    f"Frame {i} must be 3D (HxWxC), got shape {frame.shape}",
                    context={'frame_index': i, 'shape': frame.shape}
                )
            
            # Check not empty
            if frame.size == 0:
                self.validate_or_fail(
                    False,
                    f"Frame {i} is empty",
                    context={'frame_index': i}
                )
        
        # Validate metadata
        metadata = frame_data['metadata']
        self.validate_type(metadata, dict, "metadata")
        
        required_metadata = ['extraction_fps', 'total_frames', 'video_duration']
        for field in required_metadata:
            if field not in metadata:
                self.validate_or_fail(
                    False,
                    f"Frame metadata missing {field}",
                    context={'field': field, 'available_fields': list(metadata.keys())}
                )
    
    def validate_cache_access(self, video_id: str, cache_dir: Union[str, Path]) -> None:
        """
        Validate frame cache access
        """
        cache_dir = self.normalize_path(cache_dir)
        self.validate_not_empty(video_id, "video_id")
        
        # Check cache directory exists and is writable
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.validate_or_fail(
                    False,
                    f"Cannot create cache directory: {e}",
                    context={'error': str(e), 'cache_dir': str(cache_dir)}
                )
        
        if not cache_dir.is_dir():
            self.validate_or_fail(
                False,
                f"Cache path is not a directory: {cache_dir}",
                context={'cache_dir': str(cache_dir)}
            )
        
        # Check we can write to cache
        try:
            test_file = cache_dir / f".test_{video_id}"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            self.validate_or_fail(
                False,
                f"Cannot write to cache directory: {e}",
                context={'error': str(e), 'cache_dir': str(cache_dir)}
            )
```

### 3. Audio Extraction Contract

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/audio_contracts.py

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union
import wave
from .base_contract import BaseServiceContract, ServiceContractViolation

class AudioExtractionContract(BaseServiceContract):
    """Service contracts for audio extraction"""
    
    def __init__(self):
        """Initialize audio extraction contract with logging"""
        super().__init__(contract_name="AudioExtractionContract")
    
    def validate_ffmpeg_availability(self) -> None:
        """
        Validate ffmpeg is available
        Called once at startup
        """
        self.logger.info("Checking ffmpeg availability...")
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                self.validate_or_fail(
                    False,
                    f"ffmpeg not working properly: {result.stderr}",
                    context={'return_code': result.returncode, 'stderr': result.stderr}
                )
        except FileNotFoundError:
            self.validate_or_fail(
                False,
                "ffmpeg not found. Install with: sudo apt-get install ffmpeg",
                context={'install_command': 'sudo apt-get install ffmpeg'}
            )
        except subprocess.TimeoutExpired:
            self.validate_or_fail(
                False,
                "ffmpeg version check timed out",
                context={'timeout': 5}
            )
    
    def validate_audio_extraction_input(self, video_path: Union[str, Path]) -> None:
        """
        Validate input for audio extraction
        """
        video_path = self.normalize_path(video_path)
        self.validate_file_exists(video_path, "Video file for audio extraction")
        
        # Check file size
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 2048:
            self.validate_or_fail(
                False,
                f"Video file too large for audio extraction: {file_size_mb:.1f}MB",
                context={'file_size_mb': file_size_mb, 'max_size_mb': 2048}
            )
    
    def validate_audio_output(self, audio_path: Union[str, Path], expected_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate extracted audio file
        Returns audio metadata
        """
        audio_path = self.normalize_path(audio_path)
        self.validate_file_exists(audio_path, "Extracted audio file")
        
        # Check file extension
        if audio_path.suffix.lower() != '.wav':
            self.validate_or_fail(
                False,
                f"Audio must be WAV format, got {audio_path.suffix}",
                context={'expected': '.wav', 'actual': audio_path.suffix}
            )
        
        # Open WAV file to validate
        try:
            with wave.open(str(audio_path), 'rb') as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                frames = wav.getnframes()
                duration = frames / sample_rate if sample_rate > 0 else 0
        except Exception as e:
            self.validate_or_fail(
                False,
                f"Invalid WAV file: {e}",
                context={'error': str(e), 'file': str(audio_path)}
            )
        
        # Validate audio properties
        if channels != 1:
            self.validate_or_fail(
                False,
                f"Audio must be mono (1 channel), got {channels} channels",
                context={'expected_channels': 1, 'actual_channels': channels}
            )
        
        if sample_rate != 16000:
            self.validate_or_fail(
                False,
                f"Audio must be 16kHz sample rate, got {sample_rate}Hz",
                context={'expected_rate': 16000, 'actual_rate': sample_rate}
            )
        
        if duration <= 0:
            self.validate_or_fail(
                False,
                "Audio file has no duration",
                context={'file': str(audio_path)}
            )
        
        # Check duration matches video if provided
        if expected_duration is not None:
            tolerance = 1.0  # 1 second tolerance
            if abs(duration - expected_duration) > tolerance:
                self.validate_or_fail(
                    False,
                    f"Audio duration ({duration:.1f}s) doesn't match video duration ({expected_duration:.1f}s)",
                    context={'audio_duration': duration, 'expected_duration': expected_duration, 'tolerance': 1.0}
                )
        
        return {
            'channels': channels,
            'sample_rate': sample_rate,
            'duration': duration,
            'file_size_mb': audio_path.stat().st_size / (1024 * 1024)
        }
```

---

## HIGH Priority: External Dependency Contracts

### 1. Apify Client Contract

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/apify_contracts.py

import re
from typing import Dict, Any, Optional
from datetime import datetime
from .base_contract import BaseServiceContract, ServiceContractViolation

class ApifyContract(BaseServiceContract):
    """Service contracts for Apify TikTok scraping"""
    
    def __init__(self):
        """Initialize Apify contract with logging"""
        super().__init__(contract_name="ApifyContract")
    
    def validate_apify_request(self, url: str, api_key: str) -> None:
        """
        Validate Apify API request parameters
        """
        self.validate_not_empty(url, "TikTok URL")
        self.validate_not_empty(api_key, "Apify API key")
        
        # Validate API key format (typically 32 chars)
        if not re.match(r'^[a-zA-Z0-9]{20,40}$', api_key):
            self.validate_or_fail(
                False,
                "Invalid Apify API key format",
                context={'api_key_length': len(api_key), 'expected_format': '20-40 alphanumeric characters'}
            )
    
    def validate_video_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Validate scraped video metadata from Apify
        """
        self.validate_type(metadata, dict, "video_metadata")
        
        # Required fields from Apify response
        required_fields = [
            'id', 'video_url', 'caption', 'duration', 
            'width', 'height', 'created_at'
        ]
        
        for field in required_fields:
            if field not in metadata:
                self.validate_or_fail(
                    False,
                    f"Missing required metadata field: {field}",
                    context={'field': field, 'available_fields': list(metadata.keys())}
                )
        
        # Validate video ID
        video_id = metadata['id']
        if not re.match(r'^\d{17,21}$', str(video_id)):
            self.validate_or_fail(
                False,
                f"Invalid video ID from Apify: {video_id}",
                context={'video_id': video_id, 'expected_format': '17-21 digit number'}
            )
        
        # Validate video URL
        video_url = metadata['video_url']
        if not video_url or not video_url.startswith('http'):
            self.validate_or_fail(
                False,
                f"Invalid video URL from Apify: {video_url}",
                context={'video_url': video_url}
            )
        
        # Validate duration
        duration = metadata['duration']
        self.validate_type(duration, (int, float), "duration")
        if duration <= 0 or duration > 600:  # Max 10 minutes
            self.validate_or_fail(
                False,
                f"Invalid video duration: {duration}s. Must be 0-600s",
                context={'duration': duration, 'min': 0, 'max': 600}
            )
        
        # Validate dimensions
        width = metadata['width']
        height = metadata['height']
        if width <= 0 or height <= 0:
            self.validate_or_fail(
                False,
                f"Invalid video dimensions: {width}x{height}",
                context={'width': width, 'height': height}
            )
        
        # Validate engagement metrics if present
        if 'likes' in metadata:
            self.validate_type(metadata['likes'], int, "likes")
            if metadata['likes'] < 0:
                self.validate_or_fail(
                    False,
                    "Likes cannot be negative",
                    context={'likes': metadata['likes']}
                )
        
        if 'views' in metadata:
            self.validate_type(metadata['views'], int, "views")
            if metadata['views'] < 0:
                self.validate_or_fail(
                    False,
                    "Views cannot be negative",
                    context={'views': metadata['views']}
                )
```

### 2. Timeline Builder Contract

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/timeline_contracts.py

from typing import Dict, List, Any
from .base_contract import BaseServiceContract, ServiceContractViolation

class TimelineContract(BaseServiceContract):
    """Service contracts for timeline builder"""
    
    def __init__(self):
        """Initialize timeline contract with logging"""
        super().__init__(contract_name="TimelineContract")
    
    VALID_ENTRY_TYPES = [
        'object', 'speech', 'text', 'scene', 'gesture', 
        'expression', 'pose', 'audio_energy'
    ]
    
    def validate_ml_results_input(self, ml_results: Dict[str, Any]) -> None:
        """
        Validate ML results before timeline building
        """
        self.validate_type(ml_results, dict, "ml_results")
        
        # Check expected ML services
        expected_services = ['yolo', 'mediapipe', 'whisper', 'ocr', 'scene_detection']
        
        for service in expected_services:
            if service not in ml_results:
                self.logger.warning(f"Missing ML service results: {service}")
                continue
            
            result = ml_results[service]
            self.validate_type(result, dict, f"ml_results[{service}]")
            
            # Each result should have success flag
            if 'success' not in result:
                self.validate_or_fail(
                    False,
                    f"ML result {service} missing success flag",
                    context={'service': service}
                )
    
    def validate_timeline_entry(self, entry: Dict[str, Any], index: int) -> None:
        """
        Validate individual timeline entry
        """
        self.validate_type(entry, dict, f"timeline_entry[{index}]")
        
        # Required fields
        required_fields = ['start', 'entry_type', 'data']
        for field in required_fields:
            if field not in entry:
                self.validate_or_fail(
                    False,
                    f"Timeline entry {index} missing {field}",
                    context={'index': index, 'field': field, 'available_fields': list(entry.keys())}
                )
        
        # Validate entry type
        if entry['entry_type'] not in self.VALID_ENTRY_TYPES:
            self.validate_or_fail(
                False,
                f"Invalid entry type '{entry['entry_type']}' at index {index}. "
                f"Must be one of {self.VALID_ENTRY_TYPES}",
                context={'index': index, 'entry_type': entry['entry_type'], 'valid_types': self.VALID_ENTRY_TYPES}
            )
        
        # Validate timestamp
        start = entry['start']
        self.validate_type(start, (int, float), f"entry[{index}].start")
        if start < 0:
            self.validate_or_fail(
                False,
                f"Timeline entry {index} has negative start time: {start}",
                context={'index': index, 'start': start}
            )
        
        # If end time present, validate it
        if 'end' in entry and entry['end'] is not None:
            end = entry['end']
            self.validate_type(end, (int, float), f"entry[{index}].end")
            if end < start:
                self.validate_or_fail(
                    False,
                    f"Timeline entry {index} has end ({end}) before start ({start})",
                    context={'index': index, 'start': start, 'end': end}
                )
    
    def validate_timeline_consistency(self, timeline: List[Dict], duration: float) -> None:
        """
        Validate timeline consistency and completeness
        """
        self.validate_type(timeline, list, "timeline")
        
        if not timeline:
            self.logger.warning("Empty timeline")
            return
        
        # Validate each entry
        for i, entry in enumerate(timeline):
            self.validate_timeline_entry(entry, i)
        
        # Check entries are sorted by start time
        for i in range(1, len(timeline)):
            if timeline[i]['start'] < timeline[i-1]['start']:
                self.validate_or_fail(
                    False,
                    f"Timeline not sorted: entry {i} starts before {i-1}",
                    context={'index': i, 'current_start': timeline[i]['start'], 'previous_start': timeline[i-1]['start']}
                )
        
        # Check no entries exceed video duration
        for i, entry in enumerate(timeline):
            if entry['start'] > duration:
                self.validate_or_fail(
                    False,
                    f"Timeline entry {i} starts after video end ({duration}s)",
                    context={'index': i, 'start': entry['start'], 'duration': duration}
                )
            if 'end' in entry and entry['end'] and entry['end'] > duration:
                self.validate_or_fail(
                    False,
                    f"Timeline entry {i} ends after video end ({duration}s)",
                    context={'index': i, 'end': entry['end'], 'duration': duration}
                )
        
        # Check for timeline coverage (warning only)
        timeline_coverage = self._calculate_timeline_coverage(timeline, duration)
        if timeline_coverage < 0.5:
            self.logger.warning(f"Timeline only covers {timeline_coverage:.1%} of video")
    
    def _calculate_timeline_coverage(self, timeline: List[Dict], duration: float) -> float:
        """Calculate percentage of video covered by timeline entries"""
        self.logger.debug(f"Calculating timeline coverage for {len(timeline)} entries")
        if duration <= 0:
            return 0.0
        
        covered_time = set()
        for entry in timeline:
            start = int(entry['start'])
            end = int(entry.get('end', start + 1))
            for t in range(start, min(end, int(duration))):
                covered_time.add(t)
        
        return len(covered_time) / duration if duration > 0 else 0.0
```

### 3. FFmpeg Subprocess Contract

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/subprocess_contracts.py

import subprocess
import shlex
from pathlib import Path
from typing import List, Optional, Tuple, Union
from .base_contract import BaseServiceContract, ServiceContractViolation

class SubprocessContract(BaseServiceContract):
    """Service contracts for subprocess executions"""
    
    def __init__(self):
        """Initialize subprocess contract with logging"""
        super().__init__(contract_name="SubprocessContract")
    
    # Allowed commands (whitelist approach)
    ALLOWED_COMMANDS = ['ffmpeg', 'whisper.cpp/main', 'make', 'g++']
    
    # Dangerous shell patterns that could indicate command injection
    # These patterns are blocked to prevent shell injection attacks
    DANGEROUS_PATTERNS = [
        '&&',    # Command chaining
        '||',    # Command chaining with OR
        ';',     # Command separator
        '|',     # Pipe to another command
        '>',     # Output redirection
        '<',     # Input redirection
        '`',     # Command substitution (backticks)
        '$(',    # Command substitution (modern)
        '${',    # Variable substitution
        '\\n',   # Newline injection
        '\\r',   # Carriage return injection
        '&',     # Background execution
        '>>',    # Append redirection
        '2>',    # Stderr redirection
        '$((',   # Arithmetic expansion
    ]
    
    def validate_subprocess_command(self, command: List[str]) -> None:
        """
        Validate subprocess command before execution
        Security-focused validation
        """
        self.validate_not_empty(command, "command")
        self.validate_type(command, list, "command")
        
        if not command:
            self.validate_or_fail(False, "Empty command", context={'command': command})
        
        # Check command is in whitelist
        base_command = Path(command[0]).name
        if base_command not in self.ALLOWED_COMMANDS:
            # Check if it's a path to an allowed command
            allowed = False
            for allowed_cmd in self.ALLOWED_COMMANDS:
                if allowed_cmd in str(command[0]):
                    allowed = True
                    break
            
            if not allowed:
                self.validate_or_fail(
                    False,
                    f"Command '{base_command}' not in allowed list: {self.ALLOWED_COMMANDS}",
                    context={'command': base_command, 'allowed': self.ALLOWED_COMMANDS}
                )
        
        # Check for dangerous patterns
        command_str = ' '.join(command)
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in command_str:
                self.validate_or_fail(
                    False,
                    f"Dangerous pattern '{pattern}' detected in command - possible injection attack",
                    context={
                        'command': command_str, 
                        'dangerous_pattern': pattern,
                        'all_dangerous_patterns': self.DANGEROUS_PATTERNS
                    }
                )
    
    def validate_ffmpeg_command(self, 
                               input_file: Union[str, Path],
                               output_file: Union[str, Path],
                               args: List[str]) -> None:
        """
        Validate ffmpeg command specifically
        """
        input_file = self.normalize_path(input_file)
        output_file = self.normalize_path(output_file)
        self.validate_file_exists(input_file, "FFmpeg input file")
        
        # Check output directory exists
        output_dir = output_file.parent
        if not output_dir.exists():
            self.validate_or_fail(
                False,
                f"Output directory doesn't exist: {output_dir}",
                context={'output_dir': str(output_dir)}
            )
        
        # Validate common ffmpeg arguments
        dangerous_args = ['-f', 'rawvideo', '-filter_complex']
        for arg in args:
            if arg in dangerous_args:
                self.logger.warning(f"Using potentially dangerous ffmpeg arg: {arg}")
    
    def validate_subprocess_result(self,
                                  result: subprocess.CompletedProcess,
                                  expected_output_file: Optional[Union[str, Path]] = None) -> None:
        """
        Validate subprocess execution result
        """
        if expected_output_file:
            expected_output_file = self.normalize_path(expected_output_file)
        # Check return code
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown error"
            self.validate_or_fail(
                False,
                f"Subprocess failed with code {result.returncode}: {error_msg}",
                context={'returncode': result.returncode, 'error': error_msg}
            )
        
        # Check expected output file was created
        if expected_output_file and not expected_output_file.exists():
            self.validate_or_fail(
                False,
                f"Expected output file not created: {expected_output_file}",
                context={'expected_file': str(expected_output_file)}
            )
```

---

## MEDIUM Priority: Data Flow and I/O Contracts

### 1. File Handler Contract

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/file_contracts.py

import json
from pathlib import Path
from typing import Any, Dict, Union
from .base_contract import BaseServiceContract, ServiceContractViolation

class FileHandlerContract(BaseServiceContract):
    """Service contracts for file I/O operations"""
    
    def __init__(self):
        """Initialize file handler contract with logging"""
        super().__init__(contract_name="FileHandlerContract")
    
    # Maximum file sizes
    MAX_JSON_SIZE_MB = 100
    MAX_LOG_SIZE_MB = 50
    
    def validate_json_write(self, data: Any, file_path: Union[str, Path]) -> None:
        """
        Validate data before writing to JSON
        """
        file_path = self.normalize_path(file_path)
        # Check data is JSON serializable
        try:
            json_str = json.dumps(data)
        except (TypeError, ValueError) as e:
            self.validate_or_fail(
                False,
                f"Data not JSON serializable: {e}",
                context={'error': str(e)}
            )
        
        # Check size
        size_mb = len(json_str.encode()) / (1024 * 1024)
        if size_mb > self.MAX_JSON_SIZE_MB:
            self.validate_or_fail(
                False,
                f"JSON too large: {size_mb:.1f}MB. Maximum: {self.MAX_JSON_SIZE_MB}MB",
                context={'size_mb': size_mb, 'max_mb': self.MAX_JSON_SIZE_MB}
            )
        
        # Check output path
        output_dir = file_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.validate_or_fail(
                    False,
                    f"Cannot create output directory: {e}",
                    context={'error': str(e), 'directory': str(output_dir)}
                )
    
    def validate_json_read(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate JSON file before reading
        Returns parsed JSON
        """
        file_path = self.normalize_path(file_path)
        self.validate_file_exists(file_path, "JSON file")
        
        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > self.MAX_JSON_SIZE_MB:
            self.validate_or_fail(
                False,
                f"JSON file too large: {size_mb:.1f}MB. Maximum: {self.MAX_JSON_SIZE_MB}MB",
                context={'size_mb': size_mb, 'max_mb': self.MAX_JSON_SIZE_MB, 'file': str(file_path)}
            )
        
        # Try to parse JSON
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.validate_or_fail(
                False,
                f"Invalid JSON in {file_path}: {e}",
                context={'error': str(e), 'file': str(file_path)}
            )
        
        return data
```

### 2. Configuration Contract

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/config_contracts.py

import os
from pathlib import Path
from typing import Dict, Any
from .base_contract import BaseServiceContract, ServiceContractViolation

class ConfigurationContract(BaseServiceContract):
    """Service contracts for configuration and settings"""
    
    def __init__(self):
        """Initialize configuration contract with logging"""
        super().__init__(contract_name="ConfigurationContract")
    
    def validate_environment_variables(self) -> Dict[str, str]:
        """
        Validate required environment variables for Python-only flow
        Returns validated config
        """
        config = {}
        
        # Core Python-only flags
        required_flags = {
            'USE_PYTHON_ONLY_PROCESSING': 'true',
            'USE_ML_PRECOMPUTE': 'true',
            'PRECOMPUTE_CREATIVE_DENSITY': 'true',
            'PRECOMPUTE_EMOTIONAL_JOURNEY': 'true',
            'PRECOMPUTE_PERSON_FRAMING': 'true',
            'PRECOMPUTE_SCENE_PACING': 'true',
            'PRECOMPUTE_SPEECH_ANALYSIS': 'true',
            'PRECOMPUTE_VISUAL_OVERLAY': 'true',
            'PRECOMPUTE_METADATA': 'true'
        }
        
        for var, expected in required_flags.items():
            value = os.getenv(var, 'false').lower()
            if value != expected:
                self.validate_or_fail(
                    False,
                    f"{var} must be '{expected}' for Python-only flow, got '{value}'",
                    context={'variable': var, 'expected': expected, 'actual': value}
                )
            config[var] = value
        
        # Optional but recommended
        optional_vars = {
            'VIDEO_CACHE_DIR': 'temp/video_cache',
            'FRAME_CACHE_DIR': 'temp/frame_cache',
            'OUTPUT_DIR': 'insights',
            'LOG_LEVEL': 'INFO'
        }
        
        for var, default in optional_vars.items():
            config[var] = os.getenv(var, default)
        
        return config
    
    def validate_model_paths(self) -> None:
        """
        Validate ML model files exist
        """
        model_checks = [
            ('~/.feat/models', 'FEAT emotion models'),
            ('whisper.cpp/models/ggml-base.bin', 'Whisper model'),
            ('yolov8n.pt', 'YOLO model (will auto-download)')
        ]
        
        for path_str, description in model_checks:
            path = Path(path_str).expanduser()
            if not path.exists():
                self.logger.warning(f"{description} not found at {path}")
```

### 3. ML Data Extractor Contract

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/ml_extractor_contracts.py

from typing import Dict, Any, List
from .base_contract import BaseServiceContract, ServiceContractViolation

class MLExtractorContract(BaseServiceContract):
    """Service contracts for ML data extraction"""
    
    def __init__(self):
        """Initialize ML extractor contract with logging"""
        super().__init__(contract_name="MLExtractorContract")
    
    # Maximum context sizes for different analysis types
    MAX_CONTEXT_SIZES = {
        'creative_density': 50000,
        'emotional_journey': 40000,
        'person_framing': 30000,
        'scene_pacing': 20000,
        'speech_analysis': 35000,
        'visual_overlay': 40000,
        'metadata_analysis': 15000
    }
    
    def validate_extraction_request(self, 
                                   analysis_type: str,
                                   ml_data: Dict[str, Any]) -> None:
        """
        Validate ML data extraction request
        """
        # Check analysis type is valid
        if analysis_type not in self.MAX_CONTEXT_SIZES:
            self.validate_or_fail(
                False,
                f"Unknown analysis type: {analysis_type}",
                context={'analysis_type': analysis_type, 'valid_types': list(self.MAX_CONTEXT_SIZES.keys())}
            )
        
        self.validate_type(ml_data, dict, "ml_data")
        self.validate_not_empty(ml_data, "ml_data")
        
        # Check required ML services for each analysis type
        required_services = self._get_required_services(analysis_type)
        for service in required_services:
            if service not in ml_data or not ml_data[service]:
                self.logger.warning(f"Missing {service} data for {analysis_type}")
    
    def validate_extracted_context(self,
                                  context: Dict[str, Any],
                                  analysis_type: str) -> None:
        """
        Validate extracted context size and structure
        """
        self.validate_type(context, dict, "context")
        
        # Check context size
        import json
        context_str = json.dumps(context)
        context_size = len(context_str)
        
        max_size = self.MAX_CONTEXT_SIZES.get(analysis_type, 50000)
        if context_size > max_size:
            self.validate_or_fail(
                False,
                f"Context too large for {analysis_type}: {context_size} bytes (max: {max_size})",
                context={'analysis_type': analysis_type, 'context_size': context_size, 'max_size': max_size}
            )
    
    def _get_required_services(self, analysis_type: str) -> List[str]:
        """Get required ML services for each analysis type"""
        self.logger.debug(f"Getting required services for {analysis_type}")
        requirements = {
            'creative_density': ['yolo', 'ocr', 'mediapipe', 'scene_detection'],
            'emotional_journey': ['mediapipe'],
            'person_framing': ['yolo', 'mediapipe'],
            'scene_pacing': ['scene_detection'],
            'speech_analysis': ['whisper', 'mediapipe'],
            'visual_overlay': ['ocr', 'whisper'],
            'metadata_analysis': []
        }
        return requirements.get(analysis_type, [])
```

### 4. Remaining MEDIUM Priority Contracts

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/validation_contracts.py

from typing import Dict, Any, List
import re
from .base_contract import BaseServiceContract, ServiceContractViolation

class TemporalMarkerContract(BaseServiceContract):
    """Service contracts for temporal markers"""
    
    def __init__(self):
        """Initialize temporal marker contract with logging"""
        super().__init__(contract_name="TemporalMarkerContract")
    
    def validate_temporal_markers(self, markers: Dict[str, Any], duration: float) -> None:
        """Validate temporal marker format and consistency"""
        self.validate_type(markers, dict, "temporal_markers")
        
        # Validate each marker timestamp
        for marker_type, marker_list in markers.items():
            if not isinstance(marker_list, list):
                continue
                
            for i, marker in enumerate(marker_list):
                if 'timestamp' in marker:
                    ts = marker['timestamp']
                    if isinstance(ts, str):
                        # Validate "X-Ys" format
                        if not re.match(r'^\d+-\d+s$', ts):
                            self.validate_or_fail(
                                False,
                                f"Invalid timestamp format in {marker_type}[{i}]: {ts}",
                                context={'marker_type': marker_type, 'index': i, 'timestamp': ts, 'expected_format': 'X-Ys'}
                            )
                    elif isinstance(ts, (int, float)):
                        # Validate numeric timestamp
                        self.validate_range(ts, 0, duration, f"{marker_type}[{i}].timestamp")


class CacheValidationContract(BaseServiceContract):
    """Service contracts for cache operations"""
    
    def __init__(self):
        """Initialize cache validation contract with logging"""
        super().__init__(contract_name="CacheValidationContract")
    
    def validate_cache_save(self, data: Any, cache_key: str, cache_dir: Union[str, Path]) -> None:
        """Validate data before saving to cache"""
        cache_dir = self.normalize_path(cache_dir)
        self.validate_not_empty(cache_key, "cache_key")
        
        # Validate cache key format (alphanumeric + underscore)
        if not re.match(r'^[a-zA-Z0-9_]+$', cache_key):
            self.validate_or_fail(
                False,
                f"Invalid cache key format: {cache_key}",
                context={'cache_key': cache_key, 'expected_format': 'alphanumeric + underscore'}
            )
        
        # Check cache directory
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)


class PrecomputeOutputContract(BaseServiceContract):
    """Service contracts for precompute function outputs"""
    
    def __init__(self):
        """Initialize precompute output contract with logging"""
        super().__init__(contract_name="PrecomputeOutputContract")
    
    REQUIRED_BLOCKS = [
        'CoreMetrics', 'Dynamics', 'Interactions', 
        'KeyEvents', 'Patterns', 'Quality'
    ]
    
    def validate_precompute_output(self, output: Dict[str, Any], analysis_type: str) -> None:
        """Validate precompute function output format"""
        self.validate_type(output, dict, "precompute_output")
        
        # Check all 6 blocks are present
        for block in self.REQUIRED_BLOCKS:
            block_name = f"{analysis_type}{block}"
            if block_name not in output:
                self.validate_or_fail(
                    False,
                    f"Missing required block: {block_name}",
                    context={'block_name': block_name, 'available_blocks': list(output.keys())}
                )
            
            # Each block should have confidence score
            block_data = output[block_name]
            if isinstance(block_data, dict) and 'confidence' in block_data:
                self.validate_range(
                    block_data['confidence'], 0.0, 1.0, 
                    f"{block_name}.confidence"
                )


class WhisperBinaryContract(BaseServiceContract):
    """Service contracts for whisper.cpp binary"""
    
    def __init__(self):
        """Initialize whisper binary contract with logging"""
        super().__init__(contract_name="WhisperBinaryContract")
    
    def validate_whisper_binary(self) -> None:
        """Validate whisper.cpp binary is available"""
        whisper_path = Path('whisper.cpp/main')
        self.validate_file_exists(whisper_path, "whisper.cpp binary")
        
        # Check it's executable
        if not os.access(whisper_path, os.X_OK):
            self.validate_or_fail(
                False,
                f"whisper.cpp binary is not executable",
                context={'whisper_path': str(whisper_path)}
            )
    
    def validate_whisper_output(self, output: str) -> Dict[str, Any]:
        """Validate and parse whisper.cpp output"""
        self.validate_not_empty(output, "whisper output")
        
        # Parse output format (timestamps and text)
        segments = []
        for line in output.strip().split('\n'):
            # Expected format: [00:00:00.000 --> 00:00:03.000]  Text here
            match = re.match(r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s+(.+)', line)
            if match:
                start_time, end_time, text = match.groups()
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text.strip()
                })
        
        if not segments:
            self.validate_or_fail(
                False,
                "No valid segments in whisper output",
                context={'output_length': len(output), 'output_sample': output[:200] if output else None}
            )
        
        return {'segments': segments}
```

---

## 12. Contract Registry and Usage

### Central Contract Registry

```python
# Location: /home/jorge/rumiaifinal/rumiai_v2/contracts/registry.py

import logging
from typing import Dict, Type, Optional
from threading import Lock

from .base_contract import BaseServiceContract
from .feat_contracts import FEATServiceContract
from .orchestrator_contracts import OrchestratorContract
from .frame_contracts import FrameManagerContract
from .audio_contracts import AudioExtractionContract
from .apify_contracts import ApifyContract
from .timeline_contracts import TimelineContract


## 3. ML Service Contracts

### 3.1 OCRContract

```python
class OCRContract(BaseServiceContract):
    """
    Contract for OCR/Creative Analysis service validation.
    Based on actual output from creative_analysis_outputs/*.json
    """
    
    def __init__(self):
        super().__init__(contract_name="OCRContract")
    
    def validate_input(self, frames: List[Any]) -> None:
        """Validate input frames for OCR processing"""
        self.validate_type(frames, list, "frames")
        self.validate_or_fail(
            len(frames) > 0,
            "frames list cannot be empty"
        )
        
        for i, frame in enumerate(frames):
            # Validate frame has required attributes
            self.validate_or_fail(
                hasattr(frame, 'image'),
                f"Frame {i} missing 'image' attribute"
            )
            self.validate_or_fail(
                hasattr(frame, 'timestamp'),
                f"Frame {i} missing 'timestamp' attribute"
            )
            self.validate_or_fail(
                hasattr(frame, 'frame_number'),
                f"Frame {i} missing 'frame_number' attribute"
            )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """
        Validate OCR output structure.
        Based on actual creative_analysis.json files.
        """
        self.validate_type(result, dict, "result")
        
        # Required top-level fields
        self.validate_or_fail(
            'textAnnotations' in result,
            "Missing required field 'textAnnotations'"
        )
        self.validate_or_fail(
            'stickers' in result,
            "Missing required field 'stickers'"
        )
        self.validate_or_fail(
            'metadata' in result,
            "Missing required field 'metadata'"
        )
        
        # Validate textAnnotations structure
        text_annotations = result['textAnnotations']
        self.validate_type(text_annotations, list, "textAnnotations")
        
        for i, ann in enumerate(text_annotations):
            self.validate_type(ann, dict, f"textAnnotation[{i}]")
            
            # Required fields per annotation
            self.validate_or_fail(
                'text' in ann,
                f"textAnnotation[{i}] missing 'text' field"
            )
            self.validate_or_fail(
                'confidence' in ann,
                f"textAnnotation[{i}] missing 'confidence' field"
            )
            self.validate_or_fail(
                'timestamp' in ann,
                f"textAnnotation[{i}] missing 'timestamp' field"
            )
            self.validate_or_fail(
                'bbox' in ann,
                f"textAnnotation[{i}] missing 'bbox' field"
            )
            self.validate_or_fail(
                'frame_number' in ann,
                f"textAnnotation[{i}] missing 'frame_number' field"
            )
            
            # Validate field types and values
            self.validate_type(ann['text'], str, f"textAnnotation[{i}].text")
            self.validate_range(ann['confidence'], 0.0, 1.0, f"textAnnotation[{i}].confidence")
            self.validate_type(ann['timestamp'], (int, float), f"textAnnotation[{i}].timestamp")
            self.validate_type(ann['frame_number'], int, f"textAnnotation[{i}].frame_number")
            
            # Validate bbox format [x, y, width, height]
            bbox = ann['bbox']
            self.validate_type(bbox, list, f"textAnnotation[{i}].bbox")
            self.validate_or_fail(
                len(bbox) == 4,
                f"textAnnotation[{i}].bbox must have 4 values [x,y,w,h], got {len(bbox)}"
            )
            for j, val in enumerate(bbox):
                self.validate_type(val, (int, float), f"textAnnotation[{i}].bbox[{j}]")
        
        # Validate stickers structure
        stickers = result['stickers']
        self.validate_type(stickers, list, "stickers")
        
        for i, sticker in enumerate(stickers):
            self.validate_type(sticker, dict, f"sticker[{i}]")
            
            # Required sticker fields
            self.validate_or_fail(
                'bbox' in sticker,
                f"sticker[{i}] missing 'bbox' field"
            )
            self.validate_or_fail(
                'confidence' in sticker,
                f"sticker[{i}] missing 'confidence' field"
            )
            self.validate_or_fail(
                'timestamp' in sticker,
                f"sticker[{i}] missing 'timestamp' field"
            )
            self.validate_or_fail(
                'frame_number' in sticker,
                f"sticker[{i}] missing 'frame_number' field"
            )
            
            # Validate sticker field types
            bbox = sticker['bbox']
            self.validate_type(bbox, list, f"sticker[{i}].bbox")
            self.validate_or_fail(
                len(bbox) == 4,
                f"sticker[{i}].bbox must have 4 values, got {len(bbox)}"
            )
        
        # Validate metadata
        metadata = result['metadata']
        self.validate_type(metadata, dict, "metadata")
        self.validate_or_fail(
            'frames_analyzed' in metadata,
            "metadata missing 'frames_analyzed'"
        )
        self.validate_or_fail(
            'processed' in metadata,
            "metadata missing 'processed'"
        )
```

### 3.2 YOLOContract

```python
class YOLOContract(BaseServiceContract):
    """
    Contract for YOLO object detection service validation.
    Based on actual output from object_detection_outputs/*.json
    """
    
    def __init__(self):
        super().__init__(contract_name="YOLOContract")
    
    def validate_input(self, frames: List[Any]) -> None:
        """Validate input frames for YOLO processing"""
        self.validate_type(frames, list, "frames")
        self.validate_or_fail(
            len(frames) > 0,
            "frames list cannot be empty"
        )
        
        for i, frame in enumerate(frames):
            # Validate frame has required attributes
            self.validate_or_fail(
                hasattr(frame, 'image'),
                f"Frame {i} missing 'image' attribute"
            )
            self.validate_or_fail(
                hasattr(frame, 'timestamp'),
                f"Frame {i} missing 'timestamp' attribute"
            )
            self.validate_or_fail(
                hasattr(frame, 'frame_number'),
                f"Frame {i} missing 'frame_number' attribute"
            )
            
            # Validate image is numpy array
            import numpy as np
            self.validate_or_fail(
                isinstance(frame.image, np.ndarray),
                f"Frame {i} image must be numpy array"
            )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """
        Validate YOLO output structure.
        Based on actual yolo_detections.json files.
        """
        self.validate_type(result, dict, "result")
        
        # Required top-level fields
        self.validate_or_fail(
            'objectAnnotations' in result,
            "Missing required field 'objectAnnotations'"
        )
        self.validate_or_fail(
            'metadata' in result,
            "Missing required field 'metadata'"
        )
        
        # Validate objectAnnotations
        annotations = result['objectAnnotations']
        self.validate_type(annotations, list, "objectAnnotations")
        
        for i, obj in enumerate(annotations):
            self.validate_type(obj, dict, f"object[{i}]")
            
            # Required fields per detection (from actual JSON)
            required_fields = ['trackId', 'className', 'confidence', 'timestamp', 'bbox', 'frame_number']
            for field in required_fields:
                self.validate_or_fail(
                    field in obj,
                    f"object[{i}] missing required field '{field}'"
                )
            
            # Validate trackId format (e.g., "obj_0_53")
            track_id = obj['trackId']
            self.validate_type(track_id, str, f"object[{i}].trackId")
            self.validate_or_fail(
                track_id.startswith('obj_'),
                f"object[{i}].trackId must start with 'obj_', got '{track_id}'"
            )
            
            # Validate className
            self.validate_type(obj['className'], str, f"object[{i}].className")
            self.validate_or_fail(
                len(obj['className']) > 0,
                f"object[{i}].className cannot be empty"
            )
            
            # Validate confidence
            self.validate_range(obj['confidence'], 0.0, 1.0, f"object[{i}].confidence")
            
            # Validate timestamp
            self.validate_type(obj['timestamp'], (int, float), f"object[{i}].timestamp")
            self.validate_or_fail(
                obj['timestamp'] >= 0,
                f"object[{i}].timestamp must be non-negative"
            )
            
            # Validate bbox (YOLO uses xyxy format: [x1, y1, x2, y2])
            bbox = obj['bbox']
            self.validate_type(bbox, list, f"object[{i}].bbox")
            self.validate_or_fail(
                len(bbox) == 4,
                f"object[{i}].bbox must have 4 values [x1,y1,x2,y2], got {len(bbox)}"
            )
            
            # Validate bbox values are numeric
            for j, val in enumerate(bbox):
                self.validate_type(val, (int, float), f"object[{i}].bbox[{j}]")
            
            # Validate frame_number
            self.validate_type(obj['frame_number'], int, f"object[{i}].frame_number")
            self.validate_or_fail(
                obj['frame_number'] >= 0,
                f"object[{i}].frame_number must be non-negative"
            )
        
        # Validate metadata
        metadata = result['metadata']
        self.validate_type(metadata, dict, "metadata")
        
        # Required metadata fields
        self.validate_or_fail(
            'model' in metadata,
            "metadata missing 'model'"
        )
        self.validate_or_fail(
            'processed' in metadata,
            "metadata missing 'processed'"
        )
        self.validate_or_fail(
            'frames_analyzed' in metadata,
            "metadata missing 'frames_analyzed'"
        )
        self.validate_or_fail(
            'objects_detected' in metadata,
            "metadata missing 'objects_detected'"
        )
        
        # Validate metadata consistency
        self.validate_or_fail(
            metadata['objects_detected'] == len(annotations),
            f"metadata.objects_detected ({metadata['objects_detected']}) doesn't match actual count ({len(annotations)})"
        )
```

### 3.3 MediaPipeContract

```python
class MediaPipeContract(BaseServiceContract):
    """
    Contract for MediaPipe human analysis service validation.
    Based on actual output from human_analysis_outputs/*.json
    
    CRITICAL: Handles edge cases:
    - Video with no people detected (valid - empty arrays)
    - Face detected but no pose (e.g., close-up face shots)
    - Hands detected but no face/pose (e.g., hand-only shots)
    - All arrays can be independently empty
    """
    
    def __init__(self):
        super().__init__(contract_name="MediaPipeContract")
    
    def validate_input(self, frames: List[Any]) -> None:
        """Validate input frames for MediaPipe processing"""
        self.validate_type(frames, list, "frames")
        self.validate_or_fail(
            len(frames) > 0,
            "frames list cannot be empty"
        )
        
        for i, frame in enumerate(frames):
            # Validate frame has required attributes
            self.validate_or_fail(
                hasattr(frame, 'image'),
                f"Frame {i} missing 'image' attribute"
            )
            self.validate_or_fail(
                hasattr(frame, 'timestamp'),
                f"Frame {i} missing 'timestamp' attribute"
            )
            self.validate_or_fail(
                hasattr(frame, 'frame_number'),
                f"Frame {i} missing 'frame_number' attribute"
            )
            
            # Validate image is numpy array
            import numpy as np
            self.validate_or_fail(
                isinstance(frame.image, np.ndarray),
                f"Frame {i} image must be numpy array"
            )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """
        Validate MediaPipe output structure.
        Based on actual human_analysis.json files.
        
        Edge case handling:
        - Empty poses/faces/hands arrays are VALID (no people in video)
        - presence_percentage can be 0.0 (valid)
        - frames_with_people can be 0 (valid)
        - Each detection type validated independently
        """
        self.validate_type(result, dict, "result")
        
        # Required top-level fields
        required_fields = ['poses', 'faces', 'hands', 'gestures', 
                          'presence_percentage', 'frames_with_people', 'metadata']
        for field in required_fields:
            self.validate_or_fail(
                field in result,
                f"Missing required field '{field}'"
            )
        
        # Validate poses array (can be empty)
        poses = result['poses']
        self.validate_type(poses, list, "poses")
        
        for i, pose in enumerate(poses):
            self.validate_type(pose, dict, f"pose[{i}]")
            
            # Required pose fields
            self.validate_or_fail(
                'timestamp' in pose,
                f"pose[{i}] missing 'timestamp'"
            )
            self.validate_or_fail(
                'frame_number' in pose,
                f"pose[{i}] missing 'frame_number'"
            )
            self.validate_or_fail(
                'landmarks' in pose,
                f"pose[{i}] missing 'landmarks'"
            )
            self.validate_or_fail(
                'visibility' in pose,
                f"pose[{i}] missing 'visibility'"
            )
            
            # Validate pose values
            self.validate_type(pose['timestamp'], (int, float), f"pose[{i}].timestamp")
            self.validate_type(pose['frame_number'], int, f"pose[{i}].frame_number")
            
            # MediaPipe has exactly 33 pose landmarks
            self.validate_or_fail(
                pose['landmarks'] == 33,
                f"pose[{i}].landmarks must be 33, got {pose['landmarks']}"
            )
            
            # Visibility is average of all landmark visibilities
            self.validate_range(pose['visibility'], 0.0, 1.0, f"pose[{i}].visibility")
        
        # Validate faces array (can be empty)
        faces = result['faces']
        self.validate_type(faces, list, "faces")
        
        for i, face in enumerate(faces):
            self.validate_type(face, dict, f"face[{i}]")
            
            # Required face fields
            self.validate_or_fail(
                'timestamp' in face,
                f"face[{i}] missing 'timestamp'"
            )
            self.validate_or_fail(
                'frame_number' in face,
                f"face[{i}] missing 'frame_number'"
            )
            self.validate_or_fail(
                'count' in face,
                f"face[{i}] missing 'count'"
            )
            self.validate_or_fail(
                'confidence' in face,
                f"face[{i}] missing 'confidence'"
            )
            
            # Validate face values
            self.validate_type(face['timestamp'], (int, float), f"face[{i}].timestamp")
            self.validate_type(face['frame_number'], int, f"face[{i}].frame_number")
            self.validate_type(face['count'], int, f"face[{i}].count")
            self.validate_or_fail(
                face['count'] >= 1,
                f"face[{i}].count must be >= 1 when face detected"
            )
            self.validate_range(face['confidence'], 0.0, 1.0, f"face[{i}].confidence")
        
        # Validate hands array (can be empty)
        hands = result['hands']
        self.validate_type(hands, list, "hands")
        
        for i, hand in enumerate(hands):
            self.validate_type(hand, dict, f"hand[{i}]")
            
            # Required hand fields
            self.validate_or_fail(
                'timestamp' in hand,
                f"hand[{i}] missing 'timestamp'"
            )
            self.validate_or_fail(
                'frame_number' in hand,
                f"hand[{i}] missing 'frame_number'"
            )
            self.validate_or_fail(
                'count' in hand,
                f"hand[{i}] missing 'count'"
            )
            
            # Validate hand values
            self.validate_type(hand['timestamp'], (int, float), f"hand[{i}].timestamp")
            self.validate_type(hand['frame_number'], int, f"hand[{i}].frame_number")
            self.validate_type(hand['count'], int, f"hand[{i}].count")
            self.validate_or_fail(
                hand['count'] >= 1,
                f"hand[{i}].count must be >= 1 when hands detected"
            )
        
        # Validate gestures (currently always empty but must be present)
        gestures = result['gestures']
        self.validate_type(gestures, list, "gestures")
        
        # Validate presence_percentage (can be 0.0 if no people)
        presence_pct = result['presence_percentage']
        self.validate_type(presence_pct, (int, float), "presence_percentage")
        self.validate_range(presence_pct, 0.0, 100.0, "presence_percentage")
        
        # Validate frames_with_people (can be 0 if no people)
        frames_with_people = result['frames_with_people']
        self.validate_type(frames_with_people, int, "frames_with_people")
        self.validate_or_fail(
            frames_with_people >= 0,
            f"frames_with_people must be non-negative, got {frames_with_people}"
        )
        
        # Validate metadata
        metadata = result['metadata']
        self.validate_type(metadata, dict, "metadata")
        
        # Required metadata fields
        self.validate_or_fail(
            'frames_analyzed' in metadata,
            "metadata missing 'frames_analyzed'"
        )
        self.validate_or_fail(
            'processed' in metadata,
            "metadata missing 'processed'"
        )
        
        # Optional but common metadata fields
        if 'poses_detected' in metadata:
            self.validate_or_fail(
                metadata['poses_detected'] == len(poses),
                f"metadata.poses_detected mismatch"
            )
        if 'faces_detected' in metadata:
            self.validate_or_fail(
                metadata['faces_detected'] == len(faces),
                f"metadata.faces_detected mismatch"
            )
        if 'hands_detected' in metadata:
            self.validate_or_fail(
                metadata['hands_detected'] == len(hands),
                f"metadata.hands_detected mismatch"
            )
        
        # EDGE CASE: No consistency checks between poses/faces/hands
        # They can be independently empty or populated
        # Example: Close-up face shot has faces but no pose
        # Example: Hand gesture video has hands but no faces
```

### 3.4 WhisperContract

```python
class WhisperContract(BaseServiceContract):
    """
    Contract for Whisper speech transcription service validation.
    Based on actual output from speech_transcriptions/*_whisper.json
    
    CRITICAL EDGE CASES:
    - Video with no speech: text="", segments=[], language="en" or "unknown"
    - Video with only silence: Valid, same as no speech
    - Video with speech gaps: Segments may have time gaps
    - Failed transcription: metadata.success=false but structure still valid
    """
    
    def __init__(self):
        super().__init__(contract_name="WhisperContract")
    
    def validate_input(self, video_path: Union[str, Path], timeout: int = 600) -> None:
        """Validate input video file for Whisper processing"""
        # Convert to Path for validation
        video_path = self.normalize_path(video_path)
        
        # Check file exists
        self.validate_or_fail(
            video_path.exists(),
            f"Video file does not exist: {video_path}"
        )
        
        # Check it's a file not directory
        self.validate_or_fail(
            video_path.is_file(),
            f"Path is not a file: {video_path}"
        )
        
        # Check supported video/audio extensions
        valid_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.mp3', '.wav', '.m4a'}
        self.validate_or_fail(
            video_path.suffix.lower() in valid_extensions,
            f"Unsupported file format: {video_path.suffix}"
        )
        
        # Validate timeout
        self.validate_type(timeout, int, "timeout")
        self.validate_or_fail(
            0 < timeout <= 3600,  # Max 1 hour
            f"timeout must be between 1 and 3600 seconds, got {timeout}"
        )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """
        Validate Whisper output structure.
        Based on actual whisper.json files.
        
        EDGE CASES HANDLED:
        1. No speech in video: text="", segments=[], duration>0 ✓ VALID
        2. Silent video: Same as no speech ✓ VALID
        3. Partial transcription failure: Some segments present ✓ VALID
        4. Complete failure: text="", segments=[], metadata.success=false ✓ VALID
        """
        self.validate_type(result, dict, "result")
        
        # Required top-level fields
        required_fields = ['text', 'segments', 'language', 'duration']
        for field in required_fields:
            self.validate_or_fail(
                field in result,
                f"Missing required field '{field}'"
            )
        
        # Validate text field (EDGE CASE: Empty string is VALID for no speech)
        text = result['text']
        self.validate_type(text, str, "text")
        # DO NOT validate that text has content - empty is valid!
        
        # Validate segments array (EDGE CASE: Empty array is VALID for no speech)
        segments = result['segments']
        self.validate_type(segments, list, "segments")
        # DO NOT validate that segments has items - empty is valid!
        
        # EDGE CASE VALIDATION: Ensure consistency between text and segments
        # If no text, there should be no segments
        if not text.strip() and len(segments) > 0:
            # This would be inconsistent - no text but has segments
            self.validate_or_fail(
                False,
                f"Inconsistent state: empty text but {len(segments)} segments present"
            )
        
        # If we have segments, we should have some text
        # (unless it's a failure case)
        if segments:
            # Calculate total text from segments
            segment_texts = [s.get('text', '') for s in segments]
            combined_text = ' '.join(segment_texts).strip()
            
            if combined_text and not text.strip():
                self.validate_or_fail(
                    False,
                    "Segments contain text but top-level text is empty"
                )
        
        # Validate each segment (only if segments exist)
        for i, segment in enumerate(segments):
            self.validate_type(segment, dict, f"segment[{i}]")
            
            # Required segment fields
            required_segment_fields = ['id', 'start', 'end', 'text']
            for field in required_segment_fields:
                self.validate_or_fail(
                    field in segment,
                    f"segment[{i}] missing '{field}'"
                )
            
            # Validate segment id
            self.validate_type(segment['id'], int, f"segment[{i}].id")
            self.validate_or_fail(
                segment['id'] == i,
                f"segment[{i}].id should be {i}, got {segment['id']}"
            )
            
            # Validate timing
            start = segment['start']
            end = segment['end']
            self.validate_type(start, (int, float), f"segment[{i}].start")
            self.validate_type(end, (int, float), f"segment[{i}].end")
            
            self.validate_or_fail(
                start >= 0,
                f"segment[{i}].start must be non-negative"
            )
            self.validate_or_fail(
                end > start,
                f"segment[{i}].end ({end}) must be after start ({start})"
            )
            
            # Validate segment text
            self.validate_type(segment['text'], str, f"segment[{i}].text")
            # Segment text CAN be empty (pause/breathing/noise)
            
            # Validate words array if present
            if 'words' in segment:
                words = segment['words']
                self.validate_type(words, list, f"segment[{i}].words")
                # Words array can be empty even with text (no word-level timing)
        
        # Validate language
        # EDGE CASE: For silent videos, language might be 'unknown' or a default like 'en'
        language = result['language']
        self.validate_type(language, str, "language")
        self.validate_or_fail(
            len(language) >= 2,
            f"Invalid language code: '{language}'"
        )
        # Common values: 'en', 'es', 'unknown', 'english', etc.
        
        # Validate duration
        duration = result['duration']
        self.validate_type(duration, (int, float), "duration")
        self.validate_or_fail(
            duration >= 0,
            f"duration must be non-negative, got {duration}"
        )
        # EDGE CASE: Duration > 0 even if no speech detected (video has length)
        
        # Validate metadata if present
        if 'metadata' in result:
            metadata = result['metadata']
            self.validate_type(metadata, dict, "metadata")
            
            # Check success flag if present
            if 'success' in metadata:
                self.validate_type(metadata['success'], bool, "metadata.success")
                # EDGE CASE: success=false is VALID
                # It means transcription attempted but may have failed
                # The output structure is still valid
                
            if 'processed' in metadata:
                self.validate_type(metadata['processed'], bool, "metadata.processed")
                
            if 'model' in metadata:
                self.validate_type(metadata['model'], str, "metadata.model")
                
            if 'backend' in metadata:
                self.validate_type(metadata['backend'], str, "metadata.backend")
        
        # LOG edge cases for debugging
        if not text.strip() and not segments:
            self.logger.info("WhisperContract: Valid empty transcription (no speech detected)")
        elif 'metadata' in result and result['metadata'].get('success') == False:
            self.logger.warning("WhisperContract: Transcription marked as failed but structure valid")
```

### 3.5 AudioEnergyContract

```python
class AudioEnergyContract(BaseServiceContract):
    """
    Contract for Audio Energy analysis service validation.
    Based on actual output from ml_outputs/*_audio_energy.json
    
    CRITICAL: Analyzes audio dynamics using librosa RMS energy
    - Window-based energy levels (default 5-second windows)
    - Burst pattern detection (front/back/middle/steady)
    - Climax timestamp (peak energy moment)
    - Handles both success and failure states
    """
    
    def __init__(self):
        super().__init__(contract_name="AudioEnergyContract")
    
    def validate_input(self, audio_path: Union[str, Path]) -> None:
        """
        Validate input audio file for energy analysis.
        Note: Service receives extracted WAV file from video.
        """
        # Convert to Path for validation
        audio_path = self.normalize_path(audio_path)
        
        # Check file exists
        self.validate_or_fail(
            audio_path.exists(),
            f"Audio file does not exist: {audio_path}"
        )
        
        # Check it's a file not directory
        self.validate_or_fail(
            audio_path.is_file(),
            f"Path is not a file: {audio_path}"
        )
        
        # Service primarily receives WAV files from extraction
        # But librosa can handle various formats
        valid_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4'}
        self.validate_or_fail(
            audio_path.suffix.lower() in valid_extensions,
            f"Unsupported audio format: {audio_path.suffix}"
        )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """
        Validate Audio Energy output structure.
        Based on actual audio_energy.json files.
        
        EDGE CASES:
        - Short audio (<5s): Single window like "0-3s" 
        - Silent audio: All energy values near 0, variance 0
        - Processing failure: Empty windows, metadata.success=False
        - Very long audio: Many windows
        """
        self.validate_type(result, dict, "result")
        
        # Required top-level fields
        required_fields = ['energy_level_windows', 'energy_variance', 
                          'climax_timestamp', 'burst_pattern', 'metadata']
        for field in required_fields:
            self.validate_or_fail(
                field in result,
                f"Missing required field '{field}'"
            )
        
        # Duration is usually present but might be missing in error cases
        if 'duration' in result:
            duration = result['duration']
            self.validate_type(duration, (int, float), "duration")
            self.validate_or_fail(
                duration >= 0,
                f"duration must be non-negative, got {duration}"
            )
        else:
            duration = None
        
        # Validate energy_level_windows
        energy_windows = result['energy_level_windows']
        self.validate_type(energy_windows, dict, "energy_level_windows")
        
        # EDGE CASE: Empty windows indicates processing failure
        if not energy_windows:
            # Check if this is an error state
            metadata = result.get('metadata', {})
            if not metadata.get('success', True):
                # This is expected for failure cases
                self.logger.info("AudioEnergyContract: Empty windows due to processing failure")
            else:
                self.logger.warning("AudioEnergyContract: Empty windows but success=True")
        
        # Validate each window
        prev_end = None
        for window_key, energy_value in energy_windows.items():
            # Validate window key format (e.g., "0-5s", "5-10s", "0-3s")
            self.validate_type(window_key, str, f"window key")
            self.validate_or_fail(
                window_key.endswith('s'),
                f"Window key must end with 's': {window_key}"
            )
            
            # Parse window timing
            try:
                time_part = window_key[:-1]  # Remove 's'
                parts = time_part.split('-')
                self.validate_or_fail(
                    len(parts) == 2,
                    f"Window key must have format 'start-end': {window_key}"
                )
                
                start = float(parts[0])  # Can be float for precision
                end = float(parts[1])
                
                self.validate_or_fail(
                    start >= 0,
                    f"Window start must be non-negative: {window_key}"
                )
                self.validate_or_fail(
                    end > start,
                    f"Window end must be after start: {window_key}"
                )
                
                # Check window continuity (with small tolerance for rounding)
                if prev_end is not None:
                    gap = abs(start - prev_end)
                    self.validate_or_fail(
                        gap < 1.0,  # Allow up to 1 second gap for rounding
                        f"Gap between windows: prev_end={prev_end}, current_start={start}"
                    )
                prev_end = end
                
            except (ValueError, IndexError) as e:
                self.validate_or_fail(
                    False,
                    f"Invalid window key format '{window_key}': {e}"
                )
            
            # Validate energy value (normalized 0-1)
            self.validate_type(energy_value, (int, float), f"energy[{window_key}]")
            self.validate_range(energy_value, 0.0, 1.0, f"energy[{window_key}]")
        
        # Validate energy_variance
        variance = result['energy_variance']
        self.validate_type(variance, (int, float), "energy_variance")
        self.validate_or_fail(
            variance >= 0,
            f"energy_variance must be non-negative, got {variance}"
        )
        # After normalization, variance should typically be <= 1
        if variance > 1.0:
            self.logger.warning(f"AudioEnergyContract: Unusually high variance {variance}")
        
        # Validate climax_timestamp
        climax = result['climax_timestamp']
        self.validate_type(climax, (int, float), "climax_timestamp")
        self.validate_or_fail(
            climax >= 0,
            f"climax_timestamp must be non-negative, got {climax}"
        )
        
        # Cross-validation: climax should be within duration (with tolerance)
        if duration is not None and duration > 0:
            # Allow 0.1s tolerance for floating point precision
            self.validate_or_fail(
                climax <= duration + 0.1,
                f"climax_timestamp ({climax}) exceeds duration ({duration})"
            )
        
        # Validate burst_pattern
        burst_pattern = result['burst_pattern']
        self.validate_type(burst_pattern, str, "burst_pattern")
        valid_patterns = {'front_loaded', 'back_loaded', 'middle_peak', 'steady', 'unknown'}
        self.validate_or_fail(
            burst_pattern in valid_patterns,
            f"Invalid burst_pattern: {burst_pattern}, must be one of {valid_patterns}"
        )
        
        # Validate metadata
        metadata = result['metadata']
        self.validate_type(metadata, dict, "metadata")
        
        # Required metadata fields
        required_metadata = ['processed', 'success', 'method']
        for field in required_metadata:
            self.validate_or_fail(
                field in metadata,
                f"metadata missing required field '{field}'"
            )
        
        # Validate metadata values
        self.validate_type(metadata['processed'], bool, "metadata.processed")
        self.validate_type(metadata['success'], bool, "metadata.success")
        self.validate_type(metadata['method'], str, "metadata.method")
        
        # Validate method
        self.validate_or_fail(
            metadata['method'] == 'librosa_rms',
            f"Unexpected method: {metadata['method']}, expected 'librosa_rms'"
        )
        
        # Optional metadata fields
        if 'sample_rate' in metadata:
            sr = metadata['sample_rate']
            self.validate_type(sr, int, "metadata.sample_rate")
            # Common sample rates: 16000, 22050, 44100, 48000
            self.validate_or_fail(
                sr in {8000, 16000, 22050, 44100, 48000},
                f"Unusual sample_rate: {sr}"
            )
        
        if 'window_seconds' in metadata:
            ws = metadata['window_seconds']
            self.validate_type(ws, int, "metadata.window_seconds")
            self.validate_or_fail(
                1 <= ws <= 60,
                f"window_seconds out of reasonable range: {ws}"
            )
        
        if 'error' in metadata:
            # Error case - validate error message
            self.validate_type(metadata['error'], str, "metadata.error")
            self.logger.warning(f"AudioEnergyContract: Error state - {metadata['error']}")
        
        # EDGE CASE logging
        if not energy_windows and metadata.get('success') == False:
            self.logger.info("AudioEnergyContract: Processing failure (expected)")
        elif variance == 0.0 and energy_windows:
            self.logger.info("AudioEnergyContract: Zero variance (steady/silent audio)")
        elif len(energy_windows) == 1:
            self.logger.info(f"AudioEnergyContract: Short audio, single window")
```

### 3.6 SceneDetectionContract

```python
class SceneDetectionContract(BaseServiceContract):
    """
    Contract for Scene Detection service validation.
    Based on actual output from scene_detection_outputs/*_scenes.json
    
    Uses PySceneDetect for content-based scene/shot boundary detection.
    Scenes are consecutive, non-overlapping segments of video.
    """
    
    def __init__(self):
        super().__init__(contract_name="SceneDetectionContract")
    
    def validate_input(self, video_path: Union[str, Path]) -> None:
        """Validate input video file for scene detection"""
        # Convert to Path for validation
        video_path = self.normalize_path(video_path)
        
        # Check file exists
        self.validate_or_fail(
            video_path.exists(),
            f"Video file does not exist: {video_path}"
        )
        
        # Check it's a file not directory
        self.validate_or_fail(
            video_path.is_file(),
            f"Path is not a file: {video_path}"
        )
        
        # Check supported video extensions
        valid_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.mpg', '.mpeg', '.m4v'}
        self.validate_or_fail(
            video_path.suffix.lower() in valid_extensions,
            f"Unsupported video format: {video_path.suffix}"
        )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """
        Validate Scene Detection output structure.
        Based on actual scenes.json files.
        
        EDGE CASES:
        - Single scene video: scenes array has one entry (no cuts)
        - Empty scenes: Error during processing
        - Very short scenes: Can be < 0.1s (single frame difference)
        - Floating point precision: Times may have rounding errors
        """
        self.validate_type(result, dict, "result")
        
        # Required top-level fields (based on actual JSON)
        required_fields = ['scenes', 'total_scenes', 'metadata']
        for field in required_fields:
            self.validate_or_fail(
                field in result,
                f"Missing required field '{field}'"
            )
        
        # Validate scenes array
        scenes = result['scenes']
        self.validate_type(scenes, list, "scenes")
        
        # Validate total_scenes
        total_scenes = result['total_scenes']
        self.validate_type(total_scenes, int, "total_scenes")
        self.validate_or_fail(
            total_scenes >= 0,
            f"total_scenes must be non-negative, got {total_scenes}"
        )
        
        # Validate count consistency
        self.validate_or_fail(
            total_scenes == len(scenes),
            f"total_scenes ({total_scenes}) doesn't match actual count ({len(scenes)})"
        )
        
        # Validate metadata
        metadata = result['metadata']
        self.validate_type(metadata, dict, "metadata")
        
        # Required metadata field
        self.validate_or_fail(
            'processed' in metadata,
            "metadata missing required field 'processed'"
        )
        processed = metadata['processed']
        self.validate_type(processed, bool, "metadata.processed")
        
        # Handle error case
        if 'error' in metadata:
            self.validate_type(metadata['error'], str, "metadata.error")
            self.logger.warning(f"SceneDetectionContract: Error - {metadata['error']}")
            
            # In error state, we expect processed=false
            if not processed:
                # Scenes should typically be empty, but don't enforce strictly
                # as partial processing might have occurred
                if len(scenes) > 0:
                    self.logger.warning(f"SceneDetectionContract: Error state but {len(scenes)} scenes present (partial processing?)")
            return  # Skip scene validation for error case
        
        # EDGE CASE: No scenes detected (but not an error)
        if len(scenes) == 0 and processed:
            self.logger.warning("SceneDetectionContract: No scenes detected but marked as processed")
            return
        
        # Validate each scene
        prev_end_time = None
        tolerance = 0.01  # 10ms tolerance for floating point errors
        
        for i, scene in enumerate(scenes):
            self.validate_type(scene, dict, f"scene[{i}]")
            
            # Required scene fields (based on actual JSON structure)
            required_scene_fields = ['scene_number', 'start_time', 'end_time', 'duration']
            for field in required_scene_fields:
                self.validate_or_fail(
                    field in scene,
                    f"scene[{i}] missing required field '{field}'"
                )
            
            # Validate scene_number (1-indexed in actual output)
            scene_num = scene['scene_number']
            self.validate_type(scene_num, int, f"scene[{i}].scene_number")
            self.validate_or_fail(
                scene_num == i + 1,
                f"scene[{i}].scene_number should be {i+1}, got {scene_num}"
            )
            
            # Validate timing fields
            start_time = scene['start_time']
            end_time = scene['end_time']
            duration = scene['duration']
            
            self.validate_type(start_time, (int, float), f"scene[{i}].start_time")
            self.validate_type(end_time, (int, float), f"scene[{i}].end_time")
            self.validate_type(duration, (int, float), f"scene[{i}].duration")
            
            # Validate time values are sensible
            self.validate_or_fail(
                start_time >= 0,
                f"scene[{i}].start_time must be non-negative, got {start_time}"
            )
            self.validate_or_fail(
                end_time > start_time,
                f"scene[{i}].end_time ({end_time}) must be after start_time ({start_time})"
            )
            self.validate_or_fail(
                duration > 0,
                f"scene[{i}].duration must be positive, got {duration}"
            )
            
            # Validate duration consistency (with tolerance)
            calculated_duration = end_time - start_time
            self.validate_or_fail(
                abs(duration - calculated_duration) < tolerance,
                f"scene[{i}].duration ({duration}) doesn't match calculated ({calculated_duration})"
            )
            
            # Validate scene continuity and non-overlap
            if i == 0:
                # First scene should start near 0
                self.validate_or_fail(
                    abs(start_time) < tolerance,
                    f"First scene should start at ~0, got {start_time}"
                )
            else:
                # Validate no gap or overlap with previous scene
                gap = abs(start_time - prev_end_time)
                self.validate_or_fail(
                    gap < tolerance,
                    f"scene[{i}] not continuous with previous: gap of {gap}s"
                )
                
                # Ensure no overlap
                self.validate_or_fail(
                    start_time >= prev_end_time - tolerance,
                    f"scene[{i}] overlaps with previous scene"
                )
            
            prev_end_time = end_time
            
            # Warn about very short scenes
            if duration < 0.1:  # Less than 100ms
                self.logger.warning(f"scene[{i}]: Very short duration {duration}s")
            
            # Optional fields (might be present in some implementations)
            if 'scene_id' in scene:
                # scene_id is 0-indexed in implementation
                self.validate_type(scene['scene_id'], int, f"scene[{i}].scene_id")
            if 'start_frame' in scene:
                self.validate_type(scene['start_frame'], int, f"scene[{i}].start_frame")
                self.validate_or_fail(
                    scene['start_frame'] >= 0,
                    f"scene[{i}].start_frame must be non-negative"
                )
            if 'end_frame' in scene:
                self.validate_type(scene['end_frame'], int, f"scene[{i}].end_frame")
                if 'start_frame' in scene:
                    self.validate_or_fail(
                        scene['end_frame'] > scene['start_frame'],
                        f"scene[{i}].end_frame must be after start_frame"
                    )
        
        # Log edge cases for successful processing
        if processed and scenes:
            if len(scenes) == 1:
                self.logger.info("SceneDetectionContract: Single scene (no cuts detected)")
            elif len(scenes) > 100:
                self.logger.warning(f"SceneDetectionContract: Many scenes detected ({len(scenes)}), possible over-segmentation")
            
            # Check for very short last scene (possible truncation)
            if scenes[-1]['duration'] < 0.1:
                self.logger.warning(f"SceneDetectionContract: Last scene very short ({scenes[-1]['duration']}s)")
```

## 4. Compute Function Contracts

### 4.1 BaseComputeContract

```python
class BaseComputeContract(BaseServiceContract):
    """
    Base contract for all compute functions.
    Provides common validation for timeline-based compute functions.
    
    All compute functions follow the pattern:
    - Input: timelines dict + duration
    - Output: analysis dict with specific structure
    """
    
    def __init__(self, contract_name: str = "BaseComputeContract"):
        super().__init__(contract_name=contract_name)
        
        # Known timeline types and their expected structures
        self.known_timelines = {
            'textOverlayTimeline': dict,
            'objectTimeline': dict,
            'sceneChangeTimeline': (dict, list),  # Can be dict or list
            'gestureTimeline': dict,
            'expressionTimeline': dict,
            'stickerTimeline': dict,
            'speechTimeline': dict,
            'personTimeline': dict,
            'cameraDistanceTimeline': dict,
            'audioEnergyTimeline': dict,
            'sceneTimeline': (dict, list)
        }
    
    def validate_timeline_structure(self, timeline_dict: Dict, timeline_name: str) -> None:
        """
        Validate individual timeline meets contract requirements.
        
        CONTRACT:
        - Must be dict type (or list for some timelines)
        - Dict keys must be timestamp strings in "X-Ys" format
        - Values can be any type (timeline-specific)
        """
        # List timelines don't need timestamp validation
        if isinstance(timeline_dict, list):
            return
            
        if not isinstance(timeline_dict, dict):
            self.validate_or_fail(
                False,
                f"{timeline_name} must be dict, got {type(timeline_dict).__name__}"
            )
        
        for timestamp, data in timeline_dict.items():
            # Validate timestamp format
            self.validate_type(timestamp, str, f"{timeline_name} timestamp")
            
            self.validate_or_fail(
                '-' in timestamp and timestamp.endswith('s'),
                f"{timeline_name} timestamp format invalid: '{timestamp}', expected 'X-Ys' format"
            )
            
            # Validate timestamp parts are numeric
            try:
                parts = timestamp[:-1].split('-')  # Remove 's' and split
                start = float(parts[0])
                end = float(parts[1])
                
                self.validate_or_fail(
                    start >= 0 and end >= 0,
                    f"{timeline_name} timestamp '{timestamp}' has negative values"
                )
                self.validate_or_fail(
                    start < end,
                    f"{timeline_name} timestamp '{timestamp}' start >= end"
                )
            except (ValueError, IndexError) as e:
                self.validate_or_fail(
                    False,
                    f"{timeline_name} timestamp '{timestamp}' has invalid numeric parts: {e}"
                )
    
    def validate_compute_input(self, timelines: Dict[str, Any], duration: Union[int, float]) -> None:
        """
        Universal input validation for ALL compute functions.
        
        INPUTS:
        - timelines: dict containing timeline data
        - duration: video duration in seconds (positive)
        """
        # Validate timelines is dict
        self.validate_type(timelines, dict, "timelines")
        
        # Validate duration
        self.validate_type(duration, (int, float), "duration")
        self.validate_or_fail(
            duration > 0,
            f"duration must be positive, got {duration}"
        )
        self.validate_or_fail(
            duration <= 3600,  # 1 hour sanity check
            f"duration unreasonably large: {duration} seconds (> 1 hour)"
        )
        
        # Validate known timeline types
        for timeline_name, expected_type in self.known_timelines.items():
            if timeline_name in timelines:
                timeline = timelines[timeline_name]
                
                # Check type
                if isinstance(expected_type, tuple):
                    self.validate_or_fail(
                        isinstance(timeline, expected_type),
                        f"{timeline_name} must be one of {expected_type}, got {type(timeline).__name__}"
                    )
                else:
                    self.validate_type(timeline, expected_type, timeline_name)
                
                # Validate structure if non-empty dict
                if isinstance(timeline, dict) and timeline:
                    self.validate_timeline_structure(timeline, timeline_name)
        
        # Warn about unknown timelines
        for key, value in timelines.items():
            if key.endswith('Timeline') and key not in self.known_timelines:
                self.logger.warning(f"Unknown timeline type: {key}")
                # Still must be valid structure
                self.validate_or_fail(
                    isinstance(value, (dict, list)),
                    f"Unknown timeline {key} must be dict or list, got {type(value).__name__}"
                )
    
    def validate_confidence_score(self, score: float, field_name: str) -> None:
        """Validate confidence scores are in [0, 1] range"""
        self.validate_type(score, (int, float), field_name)
        self.validate_range(score, 0.0, 1.0, field_name)
```

### 4.2 CreativeDensityContract

```python
class CreativeDensityContract(BaseComputeContract):
    """
    Contract for compute_creative_density_analysis function.
    Analyzes the density and distribution of creative elements.
    """
    
    def __init__(self):
        super().__init__(contract_name="CreativeDensityContract")
    
    def validate_input(self, timelines: Dict[str, Any], duration: Union[int, float]) -> None:
        """Validate input for creative density computation"""
        self.validate_compute_input(timelines, duration)
        
        # Optional but commonly used timelines
        expected_timelines = [
            'textOverlayTimeline',
            'stickerTimeline',
            'objectTimeline',
            'gestureTimeline'
        ]
        
        # Warn if none of the expected timelines are present
        if not any(t in timelines for t in expected_timelines):
            self.logger.warning(
                f"CreativeDensityContract: No creative timelines found. Expected one of: {expected_timelines}"
            )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Validate creative density analysis output"""
        self.validate_type(result, dict, "result")
        
        # Required top-level key
        self.validate_or_fail(
            'density_analysis' in result,
            "Missing required key 'density_analysis'"
        )
        
        analysis = result['density_analysis']
        self.validate_type(analysis, dict, "density_analysis")
        
        # Required analysis fields
        required_fields = [
            'overall_density',
            'peak_density_window',
            'creative_momentum',
            'density_distribution'
        ]
        
        for field in required_fields:
            self.validate_or_fail(
                field in analysis,
                f"density_analysis missing required field '{field}'"
            )
        
        # Validate overall_density
        self.validate_confidence_score(
            analysis['overall_density'],
            "overall_density"
        )
        
        # Validate peak_density_window
        peak_window = analysis['peak_density_window']
        if peak_window is not None:
            self.validate_type(peak_window, dict, "peak_density_window")
            self.validate_or_fail(
                'window' in peak_window,
                "peak_density_window missing 'window'"
            )
            self.validate_or_fail(
                'density' in peak_window,
                "peak_density_window missing 'density'"
            )
            self.validate_confidence_score(
                peak_window['density'],
                "peak_density_window.density"
            )
        
        # Validate creative_momentum
        self.validate_type(
            analysis['creative_momentum'],
            str,
            "creative_momentum"
        )
        valid_momentums = {'increasing', 'decreasing', 'steady', 'variable'}
        self.validate_or_fail(
            analysis['creative_momentum'] in valid_momentums,
            f"Invalid creative_momentum: {analysis['creative_momentum']}"
        )
        
        # Validate density_distribution
        distribution = analysis['density_distribution']
        self.validate_type(distribution, dict, "density_distribution")
        
        # Each window in distribution should have valid format
        for window, density in distribution.items():
            self.validate_or_fail(
                window.endswith('s') and '-' in window,
                f"Invalid window format in distribution: {window}"
            )
            self.validate_confidence_score(density, f"distribution[{window}]")
```

### 4.3 EmotionalMetricsContract

```python
class EmotionalMetricsContract(BaseComputeContract):
    """
    Contract for compute_emotional_metrics function.
    Analyzes emotional content and journey throughout video.
    """
    
    def __init__(self):
        super().__init__(contract_name="EmotionalMetricsContract")
    
    def validate_input(self, timelines: Dict[str, Any], duration: Union[int, float]) -> None:
        """Validate input for emotional metrics computation"""
        self.validate_compute_input(timelines, duration)
        
        # Expected timelines for emotional analysis
        expected_timelines = [
            'expressionTimeline',
            'speechTimeline',
            'gestureTimeline'
        ]
        
        if not any(t in timelines for t in expected_timelines):
            self.logger.warning(
                f"EmotionalMetricsContract: No emotional timelines found. Expected one of: {expected_timelines}"
            )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Validate emotional metrics output"""
        self.validate_type(result, dict, "result")
        
        # Required top-level key
        self.validate_or_fail(
            'emotional_analysis' in result,
            "Missing required key 'emotional_analysis'"
        )
        
        analysis = result['emotional_analysis']
        self.validate_type(analysis, dict, "emotional_analysis")
        
        # Required fields
        required_fields = [
            'dominant_emotion',
            'emotional_variance',
            'emotional_peaks',
            'emotional_journey'
        ]
        
        for field in required_fields:
            self.validate_or_fail(
                field in analysis,
                f"emotional_analysis missing required field '{field}'"
            )
        
        # Validate dominant_emotion
        dominant = analysis['dominant_emotion']
        if dominant is not None:
            self.validate_type(dominant, str, "dominant_emotion")
        
        # Validate emotional_variance
        self.validate_confidence_score(
            analysis['emotional_variance'],
            "emotional_variance"
        )
        
        # Validate emotional_peaks
        peaks = analysis['emotional_peaks']
        self.validate_type(peaks, list, "emotional_peaks")
        for i, peak in enumerate(peaks):
            self.validate_type(peak, dict, f"emotional_peak[{i}]")
            self.validate_or_fail(
                'timestamp' in peak,
                f"emotional_peak[{i}] missing 'timestamp'"
            )
            self.validate_or_fail(
                'emotion' in peak,
                f"emotional_peak[{i}] missing 'emotion'"
            )
            self.validate_or_fail(
                'intensity' in peak,
                f"emotional_peak[{i}] missing 'intensity'"
            )
            self.validate_confidence_score(
                peak['intensity'],
                f"emotional_peak[{i}].intensity"
            )
        
        # Validate emotional_journey
        journey = analysis['emotional_journey']
        self.validate_type(journey, str, "emotional_journey")
        valid_journeys = {
            'positive_arc', 'negative_arc', 'neutral', 
            'volatile', 'crescendo', 'decrescendo'
        }
        self.validate_or_fail(
            journey in valid_journeys,
            f"Invalid emotional_journey: {journey}"
        )
```

### 4.4 SpeechAnalysisContract

```python
class SpeechAnalysisContract(BaseComputeContract):
    """
    Contract for compute_speech_analysis_metrics function.
    Analyzes speech patterns, pacing, and verbal content.
    """
    
    def __init__(self):
        super().__init__(contract_name="SpeechAnalysisContract")
    
    def validate_input(self, timelines: Dict[str, Any], duration: Union[int, float], 
                      transcript: str = None, speech_segments: list = None) -> None:
        """Validate input for speech analysis computation"""
        self.validate_compute_input(timelines, duration)
        
        # Optional transcript validation
        if transcript is not None:
            self.validate_type(transcript, str, "transcript")
        
        # Optional speech_segments validation
        if speech_segments is not None:
            self.validate_type(speech_segments, list, "speech_segments")
            for i, segment in enumerate(speech_segments):
                self.validate_type(segment, dict, f"speech_segment[{i}]")
                self.validate_or_fail(
                    'start' in segment and 'end' in segment,
                    f"speech_segment[{i}] missing start/end"
                )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Validate speech analysis output"""
        self.validate_type(result, dict, "result")
        
        # Required top-level key
        self.validate_or_fail(
            'speech_analysis' in result,
            "Missing required key 'speech_analysis'"
        )
        
        analysis = result['speech_analysis']
        self.validate_type(analysis, dict, "speech_analysis")
        
        # Required fields
        required_fields = [
            'speech_rate',
            'silence_ratio',
            'speech_clarity',
            'verbal_energy'
        ]
        
        for field in required_fields:
            self.validate_or_fail(
                field in analysis,
                f"speech_analysis missing required field '{field}'"
            )
        
        # Validate speech_rate (words per minute, can be 0)
        speech_rate = analysis['speech_rate']
        self.validate_type(speech_rate, (int, float), "speech_rate")
        self.validate_or_fail(
            speech_rate >= 0,
            f"speech_rate must be non-negative, got {speech_rate}"
        )
        
        # Validate silence_ratio
        self.validate_confidence_score(
            analysis['silence_ratio'],
            "silence_ratio"
        )
        
        # Validate speech_clarity
        self.validate_confidence_score(
            analysis['speech_clarity'],
            "speech_clarity"
        )
        
        # Validate verbal_energy
        verbal_energy = analysis['verbal_energy']
        self.validate_type(verbal_energy, str, "verbal_energy")
        valid_energies = {'high', 'medium', 'low', 'variable', 'none'}
        self.validate_or_fail(
            verbal_energy in valid_energies,
            f"Invalid verbal_energy: {verbal_energy}"
        )
```

### 4.5 ScenePacingContract

```python
class ScenePacingContract(BaseComputeContract):
    """
    Contract for compute_scene_pacing_metrics function.
    Analyzes scene transitions and pacing rhythm.
    """
    
    def __init__(self):
        super().__init__(contract_name="ScenePacingContract")
    
    def validate_input(self, timelines: Dict[str, Any], duration: Union[int, float],
                      scene_timeline: Any = None) -> None:
        """Validate input for scene pacing computation"""
        self.validate_compute_input(timelines, duration)
        
        # Scene timeline can be in timelines or passed separately
        if scene_timeline is None and 'sceneTimeline' not in timelines:
            self.logger.warning(
                "ScenePacingContract: No scene timeline found for pacing analysis"
            )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Validate scene pacing output"""
        self.validate_type(result, dict, "result")
        
        # Required top-level key
        self.validate_or_fail(
            'pacing_analysis' in result,
            "Missing required key 'pacing_analysis'"
        )
        
        analysis = result['pacing_analysis']
        self.validate_type(analysis, dict, "pacing_analysis")
        
        # Required fields
        required_fields = [
            'average_scene_duration',
            'pacing_type',
            'scene_density',
            'rhythm_consistency'
        ]
        
        for field in required_fields:
            self.validate_or_fail(
                field in analysis,
                f"pacing_analysis missing required field '{field}'"
            )
        
        # Validate average_scene_duration
        avg_duration = analysis['average_scene_duration']
        self.validate_type(avg_duration, (int, float), "average_scene_duration")
        self.validate_or_fail(
            avg_duration >= 0,
            f"average_scene_duration must be non-negative, got {avg_duration}"
        )
        
        # Validate pacing_type
        pacing_type = analysis['pacing_type']
        self.validate_type(pacing_type, str, "pacing_type")
        valid_types = {'fast', 'medium', 'slow', 'variable', 'accelerating', 'decelerating'}
        self.validate_or_fail(
            pacing_type in valid_types,
            f"Invalid pacing_type: {pacing_type}"
        )
        
        # Validate scene_density
        self.validate_confidence_score(
            analysis['scene_density'],
            "scene_density"
        )
        
        # Validate rhythm_consistency
        self.validate_confidence_score(
            analysis['rhythm_consistency'],
            "rhythm_consistency"
        )
```

### 4.6 PersonFramingContract

```python
class PersonFramingContract(BaseComputeContract):
    """
    Contract for compute_person_framing_metrics function.
    Analyzes how people are framed and positioned in video.
    """
    
    def __init__(self):
        super().__init__(contract_name="PersonFramingContract")
    
    def validate_input(self, timelines: Dict[str, Any], duration: Union[int, float]) -> None:
        """Validate input for person framing computation"""
        self.validate_compute_input(timelines, duration)
        
        # Expected timelines for framing analysis
        expected_timelines = [
            'personTimeline',
            'cameraDistanceTimeline',
            'expressionTimeline'
        ]
        
        if not any(t in timelines for t in expected_timelines):
            self.logger.warning(
                f"PersonFramingContract: No person-related timelines found. Expected one of: {expected_timelines}"
            )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Validate person framing output"""
        self.validate_type(result, dict, "result")
        
        # Required top-level key
        self.validate_or_fail(
            'framing_analysis' in result,
            "Missing required key 'framing_analysis'"
        )
        
        analysis = result['framing_analysis']
        self.validate_type(analysis, dict, "framing_analysis")
        
        # Required fields
        required_fields = [
            'dominant_framing',
            'framing_variety',
            'person_prominence',
            'framing_stability'
        ]
        
        for field in required_fields:
            self.validate_or_fail(
                field in analysis,
                f"framing_analysis missing required field '{field}'"
            )
        
        # Validate dominant_framing
        dominant_framing = analysis['dominant_framing']
        self.validate_type(dominant_framing, str, "dominant_framing")
        valid_framings = {
            'close-up', 'medium-shot', 'wide-shot', 
            'extreme-close-up', 'full-shot', 'variable', 'none'
        }
        self.validate_or_fail(
            dominant_framing in valid_framings,
            f"Invalid dominant_framing: {dominant_framing}"
        )
        
        # Validate framing_variety
        self.validate_confidence_score(
            analysis['framing_variety'],
            "framing_variety"
        )
        
        # Validate person_prominence
        self.validate_confidence_score(
            analysis['person_prominence'],
            "person_prominence"
        )
        
        # Validate framing_stability
        self.validate_confidence_score(
            analysis['framing_stability'],
            "framing_stability"
        )
```

### 4.7 MetadataAnalysisContract

```python
class MetadataAnalysisContract(BaseComputeContract):
    """
    Contract for compute_metadata_analysis_metrics function.
    Analyzes video metadata and technical properties.
    """
    
    def __init__(self):
        super().__init__(contract_name="MetadataAnalysisContract")
    
    def validate_input(self, static_metadata: Dict[str, Any], 
                      metadata_summary: Dict[str, Any], 
                      video_duration: Union[int, float]) -> None:
        """Validate input for metadata analysis"""
        # Validate static_metadata
        self.validate_type(static_metadata, dict, "static_metadata")
        
        # Validate metadata_summary
        self.validate_type(metadata_summary, dict, "metadata_summary")
        
        # Validate video_duration
        self.validate_type(video_duration, (int, float), "video_duration")
        self.validate_or_fail(
            video_duration > 0,
            f"video_duration must be positive, got {video_duration}"
        )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Validate metadata analysis output"""
        self.validate_type(result, dict, "result")
        
        # Required top-level key
        self.validate_or_fail(
            'metadata_analysis' in result,
            "Missing required key 'metadata_analysis'"
        )
        
        analysis = result['metadata_analysis']
        self.validate_type(analysis, dict, "metadata_analysis")
        
        # Required fields
        required_fields = [
            'video_quality',
            'production_value',
            'technical_consistency',
            'format_compliance'
        ]
        
        for field in required_fields:
            self.validate_or_fail(
                field in analysis,
                f"metadata_analysis missing required field '{field}'"
            )
        
        # Validate video_quality
        video_quality = analysis['video_quality']
        self.validate_type(video_quality, str, "video_quality")
        valid_qualities = {'high', 'medium', 'low', 'variable'}
        self.validate_or_fail(
            video_quality in valid_qualities,
            f"Invalid video_quality: {video_quality}"
        )
        
        # Validate production_value
        self.validate_confidence_score(
            analysis['production_value'],
            "production_value"
        )
        
        # Validate technical_consistency
        self.validate_confidence_score(
            analysis['technical_consistency'],
            "technical_consistency"
        )
        
        # Validate format_compliance
        self.validate_confidence_score(
            analysis['format_compliance'],
            "format_compliance"
        )
```

### 4.8 VisualOverlayContract

```python
class VisualOverlayContract(BaseComputeContract):
    """
    Contract for compute_visual_overlay_metrics function.
    Analyzes visual overlays, text, and graphics.
    """
    
    def __init__(self):
        super().__init__(contract_name="VisualOverlayContract")
    
    def validate_input(self, timelines: Dict[str, Any], duration: Union[int, float]) -> None:
        """Validate input for visual overlay computation"""
        self.validate_compute_input(timelines, duration)
        
        # Expected timelines for overlay analysis
        expected_timelines = [
            'textOverlayTimeline',
            'stickerTimeline',
            'gestureTimeline'
        ]
        
        if not any(t in timelines for t in expected_timelines):
            self.logger.warning(
                f"VisualOverlayContract: No overlay timelines found. Expected one of: {expected_timelines}"
            )
    
    def validate_output(self, result: Dict[str, Any]) -> None:
        """Validate visual overlay output"""
        self.validate_type(result, dict, "result")
        
        # Required top-level key
        self.validate_or_fail(
            'visual_analysis' in result,
            "Missing required key 'visual_analysis'"
        )
        
        analysis = result['visual_analysis']
        self.validate_type(analysis, dict, "visual_analysis")
        
        # Required fields
        required_fields = [
            'overlay_density',
            'text_prominence',
            'visual_complexity',
            'overlay_timing'
        ]
        
        for field in required_fields:
            self.validate_or_fail(
                field in analysis,
                f"visual_analysis missing required field '{field}'"
            )
        
        # Validate overlay_density
        self.validate_confidence_score(
            analysis['overlay_density'],
            "overlay_density"
        )
        
        # Validate text_prominence
        self.validate_confidence_score(
            analysis['text_prominence'],
            "text_prominence"
        )
        
        # Validate visual_complexity
        visual_complexity = analysis['visual_complexity']
        self.validate_type(visual_complexity, str, "visual_complexity")
        valid_complexities = {'simple', 'moderate', 'complex', 'overwhelming'}
        self.validate_or_fail(
            visual_complexity in valid_complexities,
            f"Invalid visual_complexity: {visual_complexity}"
        )
        
        # Validate overlay_timing
        overlay_timing = analysis['overlay_timing']
        self.validate_type(overlay_timing, str, "overlay_timing")
        valid_timings = {'synchronized', 'delayed', 'anticipatory', 'random'}
        self.validate_or_fail(
            overlay_timing in valid_timings,
            f"Invalid overlay_timing: {overlay_timing}"
        )
```
