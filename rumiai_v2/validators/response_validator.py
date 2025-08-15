"""
Response validator for 6-block ML outputs.
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ResponseValidator:
    """Validates analysis results for 6-block ML output format."""
    
    # Expected blocks in 6-block format (generic names)
    EXPECTED_BLOCKS = [
        'CoreMetrics',
        'Dynamics', 
        'Interactions',
        'KeyEvents',
        'Patterns',
        'Quality'
    ]
    
    # Prompt-specific block name mappings
    BLOCK_NAME_MAPPINGS = {
        'creative_density': {
            'densityCoreMetrics': 'CoreMetrics',
            'densityDynamics': 'Dynamics',
            'densityInteractions': 'Interactions',
            'densityKeyEvents': 'KeyEvents',
            'densityPatterns': 'Patterns',
            'densityQuality': 'Quality'
        },
        'emotional_journey': {
            'emotionalCoreMetrics': 'CoreMetrics',
            'emotionalDynamics': 'Dynamics',
            'emotionalInteractions': 'Interactions',
            'emotionalKeyEvents': 'KeyEvents',
            'emotionalPatterns': 'Patterns',
            'emotionalQuality': 'Quality'
        },
        'speech_analysis': {
            'speechCoreMetrics': 'CoreMetrics',
            'speechDynamics': 'Dynamics',
            'speechInteractions': 'Interactions',
            'speechKeyEvents': 'KeyEvents',
            'speechPatterns': 'Patterns',
            'speechQuality': 'Quality'
        },
        'visual_overlay_analysis': {
            'overlaysCoreMetrics': 'CoreMetrics',
            'overlaysDynamics': 'Dynamics',
            'overlaysInteractions': 'Interactions',
            'overlaysKeyEvents': 'KeyEvents',
            'overlaysPatterns': 'Patterns',
            'overlaysQuality': 'Quality'
        },
        'metadata_analysis': {
            'metadataCoreMetrics': 'CoreMetrics',
            'metadataDynamics': 'Dynamics',
            'metadataInteractions': 'Interactions',
            'metadataKeyEvents': 'KeyEvents',
            'metadataPatterns': 'Patterns',
            'metadataQuality': 'Quality'
        },
        'person_framing': {
            'framingCoreMetrics': 'CoreMetrics',
            'framingDynamics': 'Dynamics',
            'framingInteractions': 'Interactions',
            'framingKeyEvents': 'KeyEvents',
            'framingPatterns': 'Patterns',
            'framingQuality': 'Quality'
        },
        'scene_pacing': {
            'scenePacingCoreMetrics': 'CoreMetrics',
            'scenePacingDynamics': 'Dynamics',
            'scenePacingInteractions': 'Interactions',
            'scenePacingKeyEvents': 'KeyEvents',
            'scenePacingPatterns': 'Patterns',
            'scenePacingQuality': 'Quality'
        }
    }
    
    # Expected fields for each block type
    BLOCK_SCHEMAS = {
        'CoreMetrics': {
            'required': ['summary', 'key_metrics'],
            'optional': ['confidence', 'data_quality']
        },
        'Dynamics': {
            'required': ['temporal_patterns', 'transitions'],
            'optional': ['flow_analysis', 'pacing']
        },
        'Interactions': {
            'required': ['cross_modal_sync', 'engagement_factors'],
            'optional': ['coherence_score', 'alignment_issues']
        },
        'KeyEvents': {
            'required': ['critical_moments', 'timestamps'],
            'optional': ['event_impact', 'event_context']
        },
        'Patterns': {
            'required': ['recurring_elements', 'style_signatures'],
            'optional': ['uniqueness_factors', 'consistency']
        },
        'Quality': {
            'required': ['technical_quality', 'creative_quality'],
            'optional': ['production_value', 'recommendations']
        }
    }
    
    @classmethod
    def validate_6block_response(cls, response_text: str, prompt_type: str) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
        """
        Validate a 6-block format response.
        
        Args:
            response_text: Raw result text from analysis
            analysis_type: Type of analysis for context
            
        Returns:
            Tuple of (is_valid, parsed_data, errors)
        """
        errors = []
        
        # Step 1: Try to parse as JSON
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
            return False, None, errors
        
        # Step 2: Check if it's a dict
        if not isinstance(data, dict):
            errors.append("Response is not a JSON object")
            return False, None, errors
        
        # Step 3: Check for required blocks (handle prompt-specific names)
        # First try to detect which naming scheme is used
        found_blocks = set(data.keys())
        
        # Check if we have prompt-specific block names
        prompt_type_detected = None
        for pt, mapping in cls.BLOCK_NAME_MAPPINGS.items():
            if any(block_name in found_blocks for block_name in mapping.keys()):
                prompt_type_detected = pt
                logger.info(f"Detected prompt type from block names: {pt}")
                break
        
        # Map blocks to generic names if needed
        if prompt_type_detected and prompt_type_detected in cls.BLOCK_NAME_MAPPINGS:
            mapping = cls.BLOCK_NAME_MAPPINGS[prompt_type_detected]
            normalized_data = {}
            for block_name, block_data in data.items():
                if block_name in mapping:
                    normalized_data[mapping[block_name]] = block_data
                else:
                    normalized_data[block_name] = block_data
            data = normalized_data
            logger.info(f"Normalized blocks: {list(data.keys())}")
        
        # Now check for missing blocks
        missing_blocks = []
        for block in cls.EXPECTED_BLOCKS:
            if block not in data:
                missing_blocks.append(block)
        
        if missing_blocks:
            errors.append(f"Missing required blocks: {', '.join(missing_blocks)}")
            # Don't return yet - check what we do have
        
        # Step 4: Validate each block's structure
        for block_name, block_data in data.items():
            if block_name not in cls.EXPECTED_BLOCKS:
                errors.append(f"Unexpected block: {block_name}")
                continue
            
            if not isinstance(block_data, dict):
                errors.append(f"{block_name} is not a JSON object")
                continue
            
            # Skip field validation for prompt-specific formats
            # The prompts define their own required fields
            if prompt_type_detected:
                continue
            
            # Check required fields if schema exists (only for generic format)
            if block_name in cls.BLOCK_SCHEMAS:
                schema = cls.BLOCK_SCHEMAS[block_name]
                for field in schema.get('required', []):
                    if field not in block_data:
                        errors.append(f"{block_name} missing required field: {field}")
        
        # Step 5: Check data quality
        if len(errors) == 0:
            # Additional quality checks
            total_content_length = sum(
                len(json.dumps(block_data)) 
                for block_data in data.values() 
                if isinstance(block_data, dict)
            )
            
            if total_content_length < 500:
                errors.append("Response seems too short - possible incomplete generation")
            
            # Check for empty blocks
            for block_name, block_data in data.items():
                if isinstance(block_data, dict) and len(block_data) == 0:
                    errors.append(f"{block_name} is empty")
        
        # Determine if valid
        is_valid = len(errors) == 0 and len(missing_blocks) == 0
        
        if not is_valid:
            logger.warning(f"Invalid 6-block response for {prompt_type}: {'; '.join(errors)}")
        else:
            logger.info(f"Valid 6-block response for {prompt_type}")
        
        return is_valid, data if is_valid else None, errors
    
    @classmethod
    def extract_text_blocks(cls, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract 6-block structure from text response.
        
        Some responses might have the structure but not be valid JSON.
        This method attempts to extract the blocks.
        """
        blocks = {}
        current_block = None
        current_content = []
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            # Check if this line is a block header
            stripped = line.strip()
            if stripped in cls.EXPECTED_BLOCKS or stripped.rstrip(':') in cls.EXPECTED_BLOCKS:
                # Save previous block if exists
                if current_block:
                    blocks[current_block] = '\n'.join(current_content).strip()
                
                # Start new block
                current_block = stripped.rstrip(':')
                current_content = []
            elif current_block:
                current_content.append(line)
        
        # Save last block
        if current_block:
            blocks[current_block] = '\n'.join(current_content).strip()
        
        # Check if we found all blocks
        if all(block in blocks for block in cls.EXPECTED_BLOCKS):
            logger.info("Successfully extracted 6-block structure from text")
            return blocks
        
        return None
    
    @classmethod
    def validate_response(cls, response_text: str, prompt_type: str, expected_format: str = 'v2') -> Tuple[bool, Any, List[str]]:
        """
        General response validator that handles both v1 and v2 formats.
        
        Args:
            response_text: Raw response text
            prompt_type: Type of prompt
            expected_format: 'v1' or 'v2'
            
        Returns:
            Tuple of (is_valid, parsed_data, errors)
        """
        if expected_format == 'v2':
            return cls.validate_6block_response(response_text, prompt_type)
        else:
            # For v1, just check if it's valid JSON
            try:
                data = json.loads(response_text)
                return True, data, []
            except json.JSONDecodeError as e:
                return False, None, [f"Invalid JSON: {str(e)}"]