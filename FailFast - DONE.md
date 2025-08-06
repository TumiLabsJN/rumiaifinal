# Fail-Fast Error Handling Strategy for RumiAI
**Date**: 2025-08-06  
**Author**: System Architecture Team  
**Status**: Final - Post-Critique Revision  
**Priority**: CRITICAL  
**Timeline**: 2-3 days realistic implementation

---

## Executive Summary

The current RumiAI pipeline silently handles data structure errors and continues processing with partial results. This document outlines a **fail-fast strategy** where data structure mismatches and schema violations immediately halt execution, preventing silent data corruption and making debugging significantly easier.

**Key principle: Fix problems at their source, not with band-aid validation layers.**

---

## Problem Statement

### Current Issue: OCR Bounding Box Format Mismatch
```python
# Expected format (flat array):
"bbox": [x, y, width, height]  # e.g., [100, 50, 200, 100]

# Actual format from EasyOCR (polygon coordinates):
"bbox": [
    [100.0, 131.0],  # Top-left
    [217.0, 131.0],  # Top-right  
    [217.0, 152.0],  # Bottom-right
    [100.0, 152.0]   # Bottom-left
]
```

**Current behavior**: Error is logged, OCR data skipped, pipeline continues  
**Result**: Incomplete data, silent failure, difficult debugging  
**Risk**: Downstream processes receive partial data without knowing

---

## Solution: Fix at Source + Enforce Contracts + Smart Logging

### Core Philosophy
**"Invalid data structures are bugs, not runtime conditions"**

Data structure errors indicate fundamental problems that MUST be fixed at the source, not worked around at runtime.

---

## Implementation Plan

### Point 1: Fix OCR Service Output Format

**Fix the source, not the consumer.**

#### Discovery Phase
```bash
# Find where OCR processing actually happens
grep -r "textAnnotations" --include="*.py"
grep -r "easyocr" --include="*.py"
grep -r "bbox.*float.*pt" --include="*.py"
```

**ACTUAL LOCATION FOUND:** `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py:412`

```python
# In rumiai_v2/api/ml_services_unified.py - Line 412
# Current problematic code:
bbox_list = [[float(pt[0]), float(pt[1])] for pt in bbox]

# FIX: Convert polygon to flat format at the source
if bbox and len(bbox) >= 4:
    xs = [float(pt[0]) for pt in bbox]
    ys = [float(pt[1]) for pt in bbox]
    bbox_list = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
else:
    bbox_list = [0, 0, 0, 0]  # Fallback for invalid bbox
```

**Output location:** `creative_analysis_outputs/{video_id}/{video_id}_creative_analysis.json`

**Result**: Timeline builder and all consumers receive consistent format.

---

### Point 2: Central Validation with Executable Functions

**Location**: `rumiai_v2/contracts/validators.py`

```python
"""
ML Service Output Validators
Executable validation functions - not string descriptions or schemas
Single source of truth for all validation logic
"""

class MLServiceValidators:
    """Central location for ALL ML service validation functions"""
    
    @staticmethod
    def validate_ocr(output: dict) -> tuple[bool, str]:
        """Validate OCR service output against contract"""
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
            valid_patterns = ['front_loaded', 'back_loaded', 'middle_peak', 'steady']
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
```

**Usage in each service (one-line call, <1ms overhead):**
```python
# In each ML service
from rumiai_v2.contracts.validators import MLServiceValidators

class OCRService:
    def run_analysis(self, video_path):
        result = self._process_ocr(video_path)
        
        # Single line validation call
        is_valid, error_msg = MLServiceValidators.validate('ocr', result)
        if not is_valid:
            raise ValueError(f"OCR contract violation: {error_msg}")
            
        return result
```

---

### Point 3: Claude Input/Output Validation

**Location**: `rumiai_v2/contracts/claude_validators.py`

```python
"""
Claude API Validators
Ensures prompts have required data and responses match 6-block structure
"""

class ClaudePromptValidators:
    """Validate precomputed data feeding into Claude prompts"""
    
    REQUIRED_FIELDS = {
        'creative_density': [
            'average_density', 'max_density', 'element_distribution',
            'peak_density_moments', 'density_pattern_flags', 'density_curve'
        ],
        'emotional_journey': [
            'expression_timeline', 'gesture_counts', 'emotional_peaks'
        ],
        # ... all 7 prompt types
    }
    
    @classmethod
    def validate_prompt_data(cls, prompt_type: str, data: dict) -> tuple[bool, str]:
        """Validate data has all required fields before sending to Claude"""
        required = cls.REQUIRED_FIELDS.get(prompt_type, [])
        missing = [f for f in required if f not in data or data[f] is None]
        
        if missing:
            return False, f"Missing required fields: {missing}"
            
        return True, "Valid"


class ClaudeOutputValidators:
    """Validate Claude responses for ML training compatibility
    
    IMPORTANT DISCOVERY: All Claude outputs use GENERIC block names,
    not prefixed! Verified from test_outputs/*.json files.
    """
    
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
        # ... other 5 prompt types follow same pattern
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
        
        # Check all 6 blocks exist (generic names, not prefixed!)
        missing_blocks = []
        for block in cls.REQUIRED_BLOCKS:
            if block not in response:
                missing_blocks.append(block)
        
        if missing_blocks:
            return False, f"Missing blocks: {missing_blocks}. Expected: {cls.REQUIRED_BLOCKS}"
        
        # Check for extra blocks
        extra_blocks = [k for k in response.keys() if k not in cls.REQUIRED_BLOCKS]
        if extra_blocks:
            return False, f"Unexpected blocks: {extra_blocks}"
        
        # Validate each block structure
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
        
        return True, "Valid"
```

---

### Point 4: Smart Error Handling with Logging

**Location**: `rumiai_v2/core/error_handler.py`

```python
"""
Error Handler with actionable messages, debug dumps, and session memory
"""

import json
import traceback
import random
from datetime import datetime, timedelta
from pathlib import Path

class RumiAIErrorHandler:
    """Centralized error handling with logging and debugging"""
    
    def __init__(self):
        # Use environment variables with fallbacks to project-relative paths
        base_dir = Path(os.getenv('RUMIAI_BASE_DIR', Path.cwd()))
        self.log_dir = Path(os.getenv('RUMIAI_LOG_DIR', base_dir / 'logs'))
        self.debug_dir = Path(os.getenv('RUMIAI_DEBUG_DIR', base_dir / 'debug_dumps'))
        
        # Create directory structure
        (self.log_dir / "errors").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "pipeline").mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def handle_contract_violation(self, service: str, expected: any, got: any, context: dict):
        """Handle ML service contract violations with full debugging support"""
        
        # 1. Generate actionable message
        error_msg = self._create_actionable_message(service, expected, got)
        
        # 2. Create debug dump
        dump_id = self._create_debug_dump(service, context, expected, got)
        
        # 3. Log to structured file (for parsing/analysis)
        self._log_structured({
            'timestamp': datetime.now().isoformat(),
            'error_type': 'contract_violation',
            'service': service,
            'video_id': context.get('video_id'),
            'dump_id': dump_id,
            'expected': str(expected),
            'got': str(got)[:500]  # Truncate large outputs
        })
        
        # 4. Log human-readable
        self._log_human_readable(f"""
=== CONTRACT VIOLATION ===
Time: {datetime.now()}
Service: {service}
Video: {context.get('video_id')}
Debug ID: {dump_id}

What went wrong:
{error_msg}

To investigate:
1. Check debug dump: debug_dumps/{dump_id}/
2. View full context: cat debug_dumps/{dump_id}/context.json
3. Replay error: python tools/replay_error.py {dump_id}
===========================
        """)
        
        # 5. Print to console for immediate feedback
        print(f"\n❌ {error_msg}\n📁 Debug saved: {dump_id}\n")
        
        # 6. Exit with specific code
        sys.exit(10)  # Contract violation exit code
    
    def _create_actionable_message(self, service: str, expected: any, got: any) -> str:
        """Generate user-friendly error message with fix instructions"""
        
        messages = {
            'ocr': f"""
OCR Output Format Error:
  Expected: Flat bbox format [x, y, width, height]
  Got: {type(got)} format
  
  Fix:
  1. Check OCR service is using latest version
  2. Clear OCR cache: rm -rf creative_analysis_outputs/*
  3. Verify EasyOCR version: pip show easyocr
  4. Check debug dump for full output structure
            """,
            'yolo': f"""
YOLO Output Format Error:
  Expected: {expected}
  Got: Invalid format
  
  Fix:
  1. Check YOLO model version
  2. Clear detection cache: rm -rf object_detection_outputs/*
  3. Verify YOLO installation: yolo version
            """,
            # ... other services with specific fix instructions
        }
        
        return messages.get(service, f"Service {service} contract violation")
    
    def _create_debug_dump(self, service: str, context: dict, expected: any, got: any) -> str:
        """Save complete state for debugging"""
        
        dump_id = f"{int(datetime.now().timestamp())}_{random.randint(1000,9999)}"
        dump_path = self.debug_dir / dump_id
        dump_path.mkdir(parents=True)
        
        # Save everything needed to debug
        (dump_path / "error.txt").write_text(f"""
Service: {service}
Expected: {expected}
Got: {got}
Stack trace:
{traceback.format_exc()}
        """)
        
        # Save full context
        with open(dump_path / "context.json", 'w') as f:
            json.dump(context, f, indent=2, default=str)
        
        # Save replay command
        (dump_path / "replay_command.txt").write_text(
            f"python tools/replay_error.py {dump_id}"
        )
        
        return dump_id
```

---

### Point 5: Session Context Memory

**Location**: `rumiai_v2/core/session_context.py`

```python
"""
Load context from previous errors/runs for new CLI sessions
Logs become persistent knowledge base between sessions
"""

class SessionContextLoader:
    """Load context from previous errors for intelligent CLI sessions"""
    
    def __init__(self):
        # Use same configurable paths as error handler
        base_dir = Path(os.getenv('RUMIAI_BASE_DIR', Path.cwd()))
        self.log_dir = Path(os.getenv('RUMIAI_LOG_DIR', base_dir / 'logs'))
        self.debug_dir = Path(os.getenv('RUMIAI_DEBUG_DIR', base_dir / 'debug_dumps'))
    
    def get_recent_context(self, video_id: str = None) -> dict:
        """Get context from recent logs for pattern detection"""
        
        context = {
            'recent_errors': self._get_recent_errors(limit=5),
            'last_successful_run': self._get_last_success(),
            'known_issues': self._get_patterns(),
        }
        
        if video_id:
            context['video_history'] = self._get_video_history(video_id)
        
        # Provide smart suggestions based on patterns
        if self._detect_ocr_pattern(context):
            context['suggestion'] = "OCR format issues detected. Run: python tools/fix_ocr.py"
        
        return context
    
    def _get_patterns(self) -> dict:
        """Identify recurring issues from logs"""
        patterns = {}
        
        # Analyze last 7 days of errors
        for day in range(7):
            date = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
            # FIX: Use proper base path
            error_file = self.log_dir / "errors" / f"{date}_errors.json"
            
            if error_file.exists():
                try:
                    with open(error_file) as f:
                        # Each line is a JSON object (newline-delimited JSON)
                        for line in f:
                            error = json.loads(line)
                            error_type = error.get('error_type')
                            patterns[error_type] = patterns.get(error_type, 0) + 1
                except (json.JSONDecodeError, IOError) as e:
                    # Log file might be corrupted or in progress
                    print(f"Warning: Could not read {error_file}: {e}")
                    continue
        
        return patterns
```

---

### Point 6: FAIL FAST Philosophy

**New Error Handling Philosophy:**

```markdown
## Error Philosophy: FAIL FAST, FIX IMMEDIATELY

- Contract violation? FAIL
- Unexpected data? FAIL
- Missing service? FAIL
- Partial results? FAIL
- Can't understand the data? FAIL

The ONLY acceptable retry is for network timeouts.
Everything else is a bug that must be fixed.

NO ROLLBACK NEEDED:
- Single-user laptop system
- Git for version control
- Fail-fast is the goal
```

---

## Directory Structure for Logs

```
/home/jorge/rumiaifinal/
├── logs/
│   ├── errors/
│   │   ├── 2024-01-15_errors.json      # Structured for analysis
│   │   └── 2024-01-15_errors.txt       # Human-readable
│   ├── pipeline/
│   │   └── 2024-01-15_pipeline.log     # Full execution logs
│   └── summary/
│       └── daily_summary.json          # Aggregated stats
│
└── debug_dumps/
    └── 1704946245_7891/                # Timestamp_RandomID
        ├── error.txt                    # Error and stack trace
        ├── context.json                 # Full context
        ├── ml_outputs/                  # Actual ML outputs
        └── replay_command.txt           # How to reproduce
```

---

## Migration Strategy

### Realistic Timeline: 2-3 Days

**Day 1:**
- Morning: Find and understand OCR issue (2 hours)
- Afternoon: Fix OCR service, start validators (4 hours)

**Day 2:**
- Morning: Complete validators for all 6 services (3 hours)
- Afternoon: Integration and testing (4 hours)

**Day 3:**
- Morning: Fix edge cases found in testing (3 hours)
- Afternoon: Error handler, logging, documentation (3 hours)

**Total: ~20 hours of work = 2-3 days real time**

---

## What We're NOT Doing (Avoiding Technical Debt)

### ❌ NOT Adding Complex Validation Frameworks
No Pydantic schemas or complex type systems - simple functions only

### ❌ NOT Supporting Multiple Formats
No code that handles both polygon and flat bbox "just in case"

### ❌ NOT Adding Format Converters in Consumers
Fix at the source only

### ❌ NOT Implementing Rollback Infrastructure
Single-user system, git is sufficient

### ❌ NOT Gracefully Degrading Data Errors
Data errors are bugs. Bugs must be fixed, not worked around.

---

## Benefits

1. **Immediate Error Detection**: Problems surface at their source with clear messages
2. **Actionable Errors**: Every error tells you exactly how to fix it
3. **Debug Dumps**: Complete state saved for reproduction
4. **Session Memory**: Logs provide context between CLI sessions
5. **Pattern Detection**: Identifies recurring issues automatically
6. **Data Integrity**: No partial or corrupted data in pipeline
7. **No Technical Debt**: No accumulation of workarounds

---

## Discovery Commands

```bash
# Find actual ML service locations
find . -name "*.py" -exec grep -l "textAnnotations\|objectAnnotations\|poses" {} \;

# Discover ML service output structures
for service in yolo mediapipe whisper audio_energy scene_detection; do
    find . -name "*${service}*.json" -type f | head -1 | xargs cat | jq 'keys'
done

# Verify Claude output block names
find test_outputs -name "*.json" | xargs -I {} sh -c 'echo "File: {}"; cat {} | jq "keys"'

# Check actual OCR processing location
grep -n "bbox.*float.*pt" rumiai_v2/api/ml_services_unified.py
```

## Data Structure Reference

| Service | Output Location | Key Structure |
|---------|----------------|---------------|
| OCR | creative_analysis_outputs/{id}_creative_analysis.json | textAnnotations[] with bbox issue at line 412 |
| YOLO | object_detection_outputs/{id}_yolo_detections.json | objectAnnotations[] with bbox[4] |
| MediaPipe | human_analysis_outputs/{id}_human_analysis.json | poses[], faces[], hands[] |
| Whisper | speech_transcriptions/{id}_whisper.json | text, segments[], language |
| Audio Energy | ml_outputs/{id}_audio_energy.json | energy_level_windows{}, burst_pattern |
| Scene Detection | scene_detection_outputs/{id}_scenes.json | scenes[] with continuous timing |
| Claude Outputs | test_outputs/{id}_{type}.json | 6 generic blocks: CoreMetrics, Dynamics, etc. |

## Success Criteria

- [ ] OCR fix applied at rumiai_v2/api/ml_services_unified.py:412
- [ ] All 6 ML services have complete, tested validation functions
- [ ] All validation logic in single MLServiceValidators class
- [ ] Claude validators use correct generic block names (not prefixed)
- [ ] Claude Quality block validated with 'overallConfidence' not 'confidence'
- [ ] Error handler uses environment variables for paths
- [ ] Session context loader uses proper base paths
- [ ] Expected JSON structure documented
- [ ] No graceful degradation for data structure errors

---

## Conclusion

This approach fixes the root cause (OCR output format) and prevents future issues through:
1. **Executable validation functions** (not string descriptions)
2. **Single source of truth** for validation logic
3. **Actionable error messages** with specific fix instructions
4. **Debug dumps** for complete error reproduction
5. **Session memory** from logs for intelligent CLI behavior

By failing fast on data errors while maintaining network resilience, we ensure data integrity without creating technical debt through complex validation layers.

**Remember: It's better to fail loudly with bad data than to succeed silently with wrong results.**

**Realistic implementation time: 2-3 days**  
**Technical debt prevented: Immeasurable**