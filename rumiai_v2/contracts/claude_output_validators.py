"""
Claude API Output Validators
Validates Claude responses match actual 6-block structure for ML training

IMPORTANT DISCOVERY: All Claude outputs use GENERIC block names,
not prefixed! Verified from test_outputs/*.json files.
"""

class ClaudeOutputValidators:
    """Validate Claude responses match actual 6-block structure"""
    
    # ACTUAL block names from real outputs - NOT prefixed!
    REQUIRED_BLOCKS = [
        'CoreMetrics',
        'Dynamics', 
        'Interactions',
        'KeyEvents',
        'Patterns',
        'Quality'
    ]
    
    # Block-specific required fields per prompt type
    BLOCK_REQUIRED_FIELDS = {
        'creative_density': {
            'CoreMetrics': ['avgDensity', 'maxDensity', 'totalElements', 'confidence'],
            'Dynamics': ['densityCurve', 'volatility', 'accelerationPattern', 'confidence'],
            'Interactions': ['multiModalPeaks', 'elementCooccurrence', 'confidence'],
            'KeyEvents': ['peakMoments', 'deadZones', 'confidence'],
            'Patterns': ['densityPattern', 'temporalFlow', 'confidence'],
            'Quality': ['dataCompleteness', 'detectionReliability', 'overallConfidence']
        },
        'emotional_journey': {
            'CoreMetrics': ['dominantEmotion', 'emotionalIntensity', 'emotionTransitions', 'confidence'],
            'Dynamics': ['emotionalArc', 'emotionalVelocity', 'confidence'],
            'Interactions': ['emotionGestureSync', 'audioEmotionAlignment', 'confidence'],
            'KeyEvents': ['emotionalPeaks', 'neutralZones', 'confidence'],
            'Patterns': ['emotionalPattern', 'emotionalFlow', 'confidence'],
            'Quality': ['dataCompleteness', 'detectionReliability', 'overallConfidence']
        },
        'person_framing': {
            'CoreMetrics': ['avgFaceSize', 'presencePercentage', 'framingChanges', 'confidence'],
            'Dynamics': ['framingProgression', 'distancePattern', 'confidence'],
            'Interactions': ['multiPersonFrames', 'framingTransitions', 'confidence'],
            'KeyEvents': ['closeupMoments', 'wideShots', 'confidence'],
            'Patterns': ['framingPattern', 'cameraMovement', 'confidence'],
            'Quality': ['dataCompleteness', 'detectionReliability', 'overallConfidence']
        },
        'scene_pacing': {
            'CoreMetrics': ['avgSceneDuration', 'sceneCount', 'pacingSpeed', 'confidence'],
            'Dynamics': ['pacingCurve', 'rhythmPattern', 'confidence'],
            'Interactions': ['sceneMusicSync', 'transitionSmoothing', 'confidence'],
            'KeyEvents': ['quickCuts', 'longTakes', 'confidence'],
            'Patterns': ['pacingPattern', 'editingStyle', 'confidence'],
            'Quality': ['dataCompleteness', 'detectionReliability', 'overallConfidence']
        },
        'speech_analysis': {
            'CoreMetrics': ['wordCount', 'speechRate', 'silenceRatio', 'confidence'],
            'Dynamics': ['speechPacing', 'volumePattern', 'confidence'],
            'Interactions': ['speechVisualSync', 'captionAlignment', 'confidence'],
            'KeyEvents': ['keyPhrases', 'silentMoments', 'confidence'],
            'Patterns': ['speechPattern', 'deliveryStyle', 'confidence'],
            'Quality': ['dataCompleteness', 'transcriptionReliability', 'overallConfidence']
        },
        'visual_overlay_analysis': {
            'CoreMetrics': ['overlayCount', 'screenCoverage', 'overlayTypes', 'confidence'],
            'Dynamics': ['overlayTiming', 'overlayDensity', 'confidence'],
            'Interactions': ['textMotionSync', 'overlayCoordination', 'confidence'],
            'KeyEvents': ['maxOverlayMoments', 'cleanFrames', 'confidence'],
            'Patterns': ['overlayPattern', 'designStyle', 'confidence'],
            'Quality': ['dataCompleteness', 'detectionReliability', 'overallConfidence']
        },
        'metadata_analysis': {
            'CoreMetrics': ['totalViews', 'engagementRate', 'shareability', 'confidence'],
            'Dynamics': ['viralityPotential', 'trendAlignment', 'confidence'],
            'Interactions': ['audioVisualSync', 'metadataConsistency', 'confidence'],
            'KeyEvents': ['hookMoment', 'callToAction', 'confidence'],
            'Patterns': ['contentPattern', 'formatStyle', 'confidence'],
            'Quality': ['dataCompleteness', 'metadataReliability', 'overallConfidence']
        }
    }
    
    @classmethod
    def validate_claude_response(cls, prompt_type: str, response: dict) -> tuple[bool, str]:
        """Validate Claude's response has correct 6-block structure
        
        Key findings from actual outputs:
        1. All 7 prompt types use the SAME 6 generic block names
        2. NO prefixes (just CoreMetrics, not densityCoreMetrics)
        3. 5 blocks have 'confidence', Quality block has 'overallConfidence'
        4. Each prompt type has different fields within blocks
        """
        
        # 1. Check response is a dict
        if not isinstance(response, dict):
            return False, f"Response must be a dict, got {type(response)}"
        
        # 2. Check all 6 blocks exist (generic names, not prefixed!)
        missing_blocks = []
        for block in cls.REQUIRED_BLOCKS:
            if block not in response:
                missing_blocks.append(block)
        
        if missing_blocks:
            return False, f"Missing blocks: {missing_blocks}. Expected: {cls.REQUIRED_BLOCKS}"
        
        # 3. Check for extra blocks
        extra_blocks = [k for k in response.keys() if k not in cls.REQUIRED_BLOCKS]
        if extra_blocks:
            return False, f"Unexpected blocks: {extra_blocks}"
        
        # 4. Validate each block structure
        for block_name in cls.REQUIRED_BLOCKS:
            block_data = response[block_name]
            
            # Block must be a dict
            if not isinstance(block_data, dict):
                return False, f"Block {block_name} must be a dict, got {type(block_data)}"
            
            # Check confidence field (Quality uses 'overallConfidence')
            if block_name == 'Quality':
                if 'overallConfidence' not in block_data:
                    return False, f"Quality block missing 'overallConfidence'"
                conf = block_data['overallConfidence']
            else:
                if 'confidence' not in block_data:
                    return False, f"Block {block_name} missing 'confidence' field"
                conf = block_data['confidence']
            
            # Validate confidence value
            if conf is not None:  # Some blocks might have null confidence
                if not isinstance(conf, (int, float)):
                    return False, f"Invalid confidence type in {block_name}: {type(conf)}"
                if not 0 <= conf <= 1:
                    return False, f"Invalid confidence in {block_name}: {conf}"
            
            # Check prompt-specific required fields
            if prompt_type in cls.BLOCK_REQUIRED_FIELDS:
                required_fields = cls.BLOCK_REQUIRED_FIELDS[prompt_type].get(block_name, [])
                for field in required_fields:
                    if field not in block_data:
                        return False, f"Block {block_name} missing required field '{field}' for {prompt_type}"
        
        # 5. Additional validation for specific blocks
        
        # CoreMetrics should have numeric metrics
        core_metrics = response.get('CoreMetrics', {})
        for key, value in core_metrics.items():
            if key != 'confidence' and 'Count' in key:
                if value is not None and not isinstance(value, (int, float)):
                    return False, f"CoreMetrics.{key} should be numeric, got {type(value)}"
        
        # Dynamics should have arrays or patterns
        dynamics = response.get('Dynamics', {})
        for key, value in dynamics.items():
            if 'Curve' in key and value is not None and not isinstance(value, list):
                return False, f"Dynamics.{key} should be a list, got {type(value)}"
        
        # KeyEvents should have arrays
        key_events = response.get('KeyEvents', {})
        for key, value in key_events.items():
            if key != 'confidence' and isinstance(value, list):
                # Validate timestamp format in events
                for event in value:
                    if isinstance(event, dict) and 'timestamp' in event:
                        ts = event['timestamp']
                        if not isinstance(ts, (int, float, str)):
                            return False, f"Invalid timestamp type in KeyEvents.{key}"
        
        return True, "Valid"
    
    @classmethod
    def validate_all_prompt_types(cls, responses: dict) -> dict:
        """Validate responses for all 7 prompt types"""
        results = {}
        for prompt_type in cls.BLOCK_REQUIRED_FIELDS.keys():
            if prompt_type in responses:
                is_valid, message = cls.validate_claude_response(prompt_type, responses[prompt_type])
                results[prompt_type] = {'valid': is_valid, 'message': message}
            else:
                results[prompt_type] = {'valid': False, 'message': 'Response not found'}
        return results