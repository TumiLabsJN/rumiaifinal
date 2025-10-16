"""
Stage 2: Content Analysis (Stages 2.6 & 2.7)
Pattern discovery and video classification using LLM

Source: ContentAnalysisCHILDTI.md
"""

from .discovery import (
    sample_transcripts_for_discovery,
    discover_patterns_llm,
    calculate_percentages,
    run_discovery_stage
)

from .classification import (
    classify_video_llm,
    classify_all_videos,
    run_classification_stage
)

from .validation import (
    validate_discovery_inputs,
    validate_classification_inputs,
    validate_business_rules_sampling,
    validate_business_rules_classification,
    validate_discovery_output,
    validate_classification_output
)

from .error_handlers import (
    handle_missing_input_file,
    handle_api_timeout_with_retry,
    handle_invalid_json_response,
    handle_graceful_skip,
    handle_api_rate_limit
)

from .utils import (
    load_json,
    save_json,
    construct_path,
    get_logger
)

__all__ = [
    # Discovery (Stage 2.6)
    'sample_transcripts_for_discovery',
    'discover_patterns_llm',
    'calculate_percentages',
    'run_discovery_stage',

    # Classification (Stage 2.7)
    'classify_video_llm',
    'classify_all_videos',
    'run_classification_stage',

    # Validation
    'validate_discovery_inputs',
    'validate_classification_inputs',
    'validate_business_rules_sampling',
    'validate_business_rules_classification',
    'validate_discovery_output',
    'validate_classification_output',

    # Error Handlers
    'handle_missing_input_file',
    'handle_api_timeout_with_retry',
    'handle_invalid_json_response',
    'handle_graceful_skip',
    'handle_api_rate_limit',

    # Utils
    'load_json',
    'save_json',
    'construct_path',
    'get_logger'
]

__version__ = '1.0.0'
__author__ = 'Tumi Labs'
__description__ = 'Content Analysis Stage: Pattern Discovery & Video Classification'
