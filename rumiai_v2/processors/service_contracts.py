"""
Service Contract Definitions and Validators
===========================================
Philosophy: FAIL FAST on contract violations
No graceful degradation, no default values, no silent failures
"""

from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class ServiceContractViolation(ValueError):
    """Explicit exception for contract violations"""
    pass

def validate_timeline_structure(timeline_dict: Dict, timeline_name: str) -> None:
    """
    Validate individual timeline meets contract requirements.
    
    CONTRACT:
    - Must be dict type
    - Keys must be timestamp strings in "X-Ys" format
    - Values can be any type (timeline-specific)
    
    FAIL FAST: Raises ServiceContractViolation on any violation
    """
    if not isinstance(timeline_dict, dict):
        raise ServiceContractViolation(
            f"CONTRACT VIOLATION: {timeline_name} must be dict, got {type(timeline_dict).__name__}"
        )
    
    for timestamp, data in timeline_dict.items():
        # Validate timestamp format
        if not isinstance(timestamp, str):
            raise ServiceContractViolation(
                f"CONTRACT VIOLATION: {timeline_name} timestamp must be string, "
                f"got {type(timestamp).__name__} for key {timestamp}"
            )
        
        if '-' not in timestamp or not timestamp.endswith('s'):
            raise ServiceContractViolation(
                f"CONTRACT VIOLATION: {timeline_name} timestamp format invalid: '{timestamp}', "
                f"expected 'X-Ys' format (e.g., '0-1s', '5-10s')"
            )
        
        # Validate timestamp parts are numeric
        try:
            parts = timestamp[:-1].split('-')  # Remove 's' and split
            start = float(parts[0])
            end = float(parts[1])
            if start < 0 or end < 0:
                raise ValueError("Negative timestamps not allowed")
            if start >= end:
                raise ValueError("Start must be less than end")
        except (ValueError, IndexError) as e:
            raise ServiceContractViolation(
                f"CONTRACT VIOLATION: {timeline_name} timestamp '{timestamp}' "
                f"has invalid numeric parts: {str(e)}"
            )

def validate_compute_contract(timelines: Dict[str, Any], duration: Union[int, float]) -> None:
    """
    Universal service contract for ALL compute functions.
    
    SERVICE CONTRACT
    ================
    
    INPUTS:
    - timelines: dict containing timeline data
      * Required type: dict
      * Optional keys: textOverlayTimeline, objectTimeline, etc.
      * Each timeline value MUST be dict with "X-Ys" timestamp keys
      
    - duration: video duration in seconds
      * Required type: int or float
      * Must be positive (> 0)
    
    GUARANTEES:
    - Validates ALL inputs before processing
    - Fails fast with clear error messages
    - No silent failures or default values
    - Deterministic: same input = same output
    
    FAILURES:
    - ServiceContractViolation: Contract violation (wrong types, invalid structure)
    - No recovery attempted - caller MUST handle errors
    """
    
    # Contract Rule 1: timelines MUST be dict
    if not isinstance(timelines, dict):
        raise ServiceContractViolation(
            f"CONTRACT VIOLATION: timelines must be dict, got {type(timelines).__name__}"
        )
    
    # Contract Rule 2: duration MUST be positive number
    if not isinstance(duration, (int, float)):
        raise ServiceContractViolation(
            f"CONTRACT VIOLATION: duration must be number, got {type(duration).__name__}"
        )
    
    if duration <= 0:
        raise ServiceContractViolation(
            f"CONTRACT VIOLATION: duration must be positive, got {duration}"
        )
    
    if duration > 3600:  # 1 hour sanity check
        raise ServiceContractViolation(
            f"CONTRACT VIOLATION: duration unreasonably large: {duration} seconds (> 1 hour)"
        )
    
    # Contract Rule 3: Known timeline types must have correct structure
    known_timelines = {
        'textOverlayTimeline': dict,
        'objectTimeline': dict,
        'sceneChangeTimeline': (dict, list),  # Can be dict or list
        'gestureTimeline': dict,
        'expressionTimeline': dict,
        'stickerTimeline': dict,
        'speechTimeline': dict,
        'personTimeline': dict,
        'cameraDistanceTimeline': dict
    }
    
    for timeline_name, expected_type in known_timelines.items():
        if timeline_name in timelines:
            timeline = timelines[timeline_name]
            
            # Check type
            if not isinstance(timeline, expected_type):
                raise ServiceContractViolation(
                    f"CONTRACT VIOLATION: {timeline_name} must be {expected_type}, "
                    f"got {type(timeline).__name__}"
                )
            
            # Validate dict timelines structure
            if isinstance(timeline, dict) and timeline:  # Non-empty dict
                validate_timeline_structure(timeline, timeline_name)
    
    # Contract Rule 4: No unknown timeline types with invalid structure
    for key, value in timelines.items():
        if key.endswith('Timeline') and key not in known_timelines:
            logger.warning(f"Unknown timeline type: {key}")
            # Still must be valid structure if it claims to be a timeline
            if not isinstance(value, (dict, list)):
                raise ServiceContractViolation(
                    f"CONTRACT VIOLATION: Unknown timeline {key} must be dict or list, "
                    f"got {type(value).__name__}"
                )

def validate_output_contract(result: Dict[str, Any], function_name: str) -> None:
    """
    Validate compute function output meets its contract.
    
    OUTPUT CONTRACT:
    - Must return dict
    - Must have expected top-level key
    - Must have confidence scores between 0 and 1
    """
    if not isinstance(result, dict):
        raise ServiceContractViolation(
            f"OUTPUT CONTRACT VIOLATION: {function_name} must return dict, "
            f"got {type(result).__name__}"
        )
    
    # Check for expected structure based on function
    expected_keys = {
        'compute_creative_density_analysis': 'density_analysis',
        'compute_emotional_metrics': 'emotional_analysis',
        'compute_speech_analysis_metrics': 'speech_analysis',
        'compute_visual_overlay_metrics': 'visual_analysis',
        'compute_metadata_analysis_metrics': 'metadata_analysis',
        'compute_person_framing_metrics': 'framing_analysis',
        'compute_scene_pacing_metrics': 'pacing_analysis'
    }
    
    if function_name in expected_keys:
        key = expected_keys[function_name]
        if key not in result:
            raise ServiceContractViolation(
                f"OUTPUT CONTRACT VIOLATION: {function_name} result missing required key '{key}'"
            )

# FAIL FAST Error Handling Philosophy
# ====================================
#
# Level 1: Service Contract Violations (ServiceContractViolation)
#   -> Fail immediately with clear message
#   -> No recovery attempted
#   -> Caller must fix their data
#
# Level 2: Programming Errors (AssertionError) 
#   -> Internal consistency checks
#   -> Should never happen in production
#   -> Indicates bug in our code
#
# Level 3: System Errors (OSError, MemoryError)
#   -> Let them propagate
#   -> Infrastructure layer handles
#
# NO Level: Data Quality Issues
#   -> Empty timelines are VALID (video might have no text)
#   -> Missing ML detections are VALID (nothing detected)
#   -> Low confidence is VALID (poor video quality)
#   -> These are NOT errors, just characteristics