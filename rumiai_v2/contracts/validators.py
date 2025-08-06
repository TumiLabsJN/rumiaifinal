"""
ML Service Output Validators
Executable validation functions - not string descriptions or schemas
Single source of truth for all validation logic
"""

class MLServiceValidators:
    """Central location for ALL ML service validation functions"""
    
    @staticmethod
    def validate_ocr(output: dict) -> tuple[bool, str]:
        """Validate OCR service output against contract
        Expected structure from: creative_analysis_outputs/{id}_creative_analysis.json
        """
        try:
            if 'textAnnotations' not in output:
                return True, "No text annotations (valid empty result)"
            
            for i, ann in enumerate(output['textAnnotations']):
                # Check bbox structure
                if 'bbox' not in ann:
                    return False, f"Annotation {i} missing bbox"
                
                bbox = ann['bbox']
                if not isinstance(bbox, list) or len(bbox) != 4:
                    return False, f"Annotation {i}: bbox must be [x,y,w,h], got {bbox}"
                
                if not all(isinstance(x, (int, float)) for x in bbox):
                    return False, f"Annotation {i}: bbox has non-numeric values"
                
                # Check other required fields
                if 'text' not in ann or not isinstance(ann['text'], str):
                    return False, f"Annotation {i}: missing or invalid text field"
                
                if 'confidence' not in ann:
                    return False, f"Annotation {i}: missing confidence field"
                
                conf = ann['confidence']
                if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
                    return False, f"Annotation {i}: invalid confidence {conf}"
                    
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def validate_yolo(output: dict) -> tuple[bool, str]:
        """Validate YOLO detection output
        Expected structure from: object_detection_outputs/{id}_yolo_detections.json
        """
        try:
            if 'objectAnnotations' not in output:
                return True, "No objects detected (valid empty result)"
            
            for i, obj in enumerate(output['objectAnnotations']):
                # Check required fields
                required = ['trackId', 'className', 'confidence', 'bbox', 'timestamp', 'frame_number']
                for field in required:
                    if field not in obj:
                        return False, f"Object {i} missing field: {field}"
                
                # Validate bbox format (should be [x, y, width, height])
                bbox = obj['bbox']
                if not isinstance(bbox, list) or len(bbox) != 4:
                    return False, f"Object {i}: bbox must have 4 values, got {len(bbox)}"
                
                if not all(isinstance(x, (int, float)) for x in bbox):
                    return False, f"Object {i}: bbox has non-numeric values"
                
                # Validate confidence
                if not 0 <= obj['confidence'] <= 1:
                    return False, f"Object {i}: confidence {obj['confidence']} out of range"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def validate_mediapipe(output: dict) -> tuple[bool, str]:
        """Validate MediaPipe human analysis output
        Expected structure from: human_analysis_outputs/{id}_human_analysis.json
        """
        try:
            # Check required top-level fields
            required = ['poses', 'faces', 'hands', 'gestures', 'metadata']
            for field in required:
                if field not in output:
                    return False, f"Missing field: {field}"
            
            # Validate poses
            for i, pose in enumerate(output.get('poses', [])):
                if 'timestamp' not in pose:
                    return False, f"Pose {i} missing timestamp"
                if 'landmarks' not in pose:
                    return False, f"Pose {i} missing landmarks"
                if pose['landmarks'] != 33:  # MediaPipe has 33 pose landmarks
                    return False, f"Pose {i}: expected 33 landmarks, got {pose['landmarks']}"
            
            # Validate faces
            for i, face in enumerate(output.get('faces', [])):
                if 'timestamp' not in face:
                    return False, f"Face {i} missing timestamp"
                if 'confidence' not in face:
                    return False, f"Face {i} missing confidence"
            
            # Validate hands
            for i, hand in enumerate(output.get('hands', [])):
                if 'timestamp' not in hand:
                    return False, f"Hand {i} missing timestamp"
                if 'count' not in hand:
                    return False, f"Hand {i} missing count"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def validate_whisper(output: dict) -> tuple[bool, str]:
        """Validate Whisper transcription output
        Expected structure from: speech_transcriptions/{id}_whisper.json
        """
        try:
            # Check required fields
            required = ['text', 'segments', 'language', 'duration']
            for field in required:
                if field not in output:
                    return False, f"Missing field: {field}"
            
            # Validate segments
            for i, segment in enumerate(output.get('segments', [])):
                if 'start' not in segment:
                    return False, f"Segment {i} missing start time"
                if 'end' not in segment:
                    return False, f"Segment {i} missing end time"
                if 'text' not in segment:
                    return False, f"Segment {i} missing text"
                
                # Validate timing
                if segment['start'] < 0:
                    return False, f"Segment {i}: negative start time"
                if segment['end'] <= segment['start']:
                    return False, f"Segment {i}: end time not after start"
            
            # Empty transcription is valid (video might have no speech)
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def validate_audio_energy(output: dict) -> tuple[bool, str]:
        """Validate Audio Energy analysis output
        Expected structure from: ml_outputs/{id}_audio_energy.json
        """
        try:
            # Check required fields
            required = ['energy_level_windows', 'energy_variance', 'climax_timestamp', 'burst_pattern']
            for field in required:
                if field not in output:
                    return False, f"Missing field: {field}"
            
            # Validate energy windows
            windows = output['energy_level_windows']
            if not isinstance(windows, dict):
                return False, "energy_level_windows must be a dictionary"
            
            for window, energy in windows.items():
                if not isinstance(energy, (int, float)):
                    return False, f"Window {window}: energy must be numeric"
                if not 0 <= energy <= 1:
                    return False, f"Window {window}: energy {energy} out of range [0,1]"
            
            # Validate burst pattern
            valid_patterns = ['front_loaded', 'back_loaded', 'middle_peak', 'steady', 'unknown']
            if output['burst_pattern'] not in valid_patterns:
                return False, f"Invalid burst_pattern: {output['burst_pattern']}"
            
            # Validate variance
            if not 0 <= output['energy_variance'] <= 1:
                return False, f"energy_variance {output['energy_variance']} out of range"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def validate_scene_detection(output: dict) -> tuple[bool, str]:
        """Validate Scene Detection output
        Expected structure from: scene_detection_outputs/{id}_scenes.json
        """
        try:
            if 'scenes' not in output:
                return False, "Missing 'scenes' field"
            
            scenes = output['scenes']
            if not isinstance(scenes, list):
                return False, "'scenes' must be a list"
            
            # Validate each scene
            prev_end = 0
            for i, scene in enumerate(scenes):
                # Check required fields
                required = ['scene_number', 'start_time', 'end_time', 'duration']
                for field in required:
                    if field not in scene:
                        return False, f"Scene {i} missing field: {field}"
                
                # Validate scene number
                if scene['scene_number'] != i + 1:
                    return False, f"Scene {i}: incorrect scene_number"
                
                # Validate timing
                if scene['start_time'] < 0:
                    return False, f"Scene {i}: negative start_time"
                if scene['end_time'] <= scene['start_time']:
                    return False, f"Scene {i}: end_time not after start_time"
                
                # Check continuity (scenes should be consecutive)
                if i > 0 and abs(scene['start_time'] - prev_end) > 0.01:
                    return False, f"Scene {i}: gap or overlap with previous scene"
                
                prev_end = scene['end_time']
                
                # Validate duration
                expected_duration = scene['end_time'] - scene['start_time']
                if abs(scene['duration'] - expected_duration) > 0.01:
                    return False, f"Scene {i}: duration mismatch"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    # Registry for easy lookup
    VALIDATORS = {
        'ocr': validate_ocr,
        'yolo': validate_yolo,
        'mediapipe': validate_mediapipe,
        'whisper': validate_whisper,
        'audio_energy': validate_audio_energy,
        'scene_detection': validate_scene_detection
    }
    
    @classmethod
    def validate(cls, service_name: str, output: dict) -> tuple[bool, str]:
        """Main entry point for validation"""
        if service_name not in cls.VALIDATORS:
            return False, f"Unknown service: {service_name}"
        
        validator = cls.VALIDATORS[service_name]
        return validator(output)