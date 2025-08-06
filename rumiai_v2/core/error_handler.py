"""
Error Handler with actionable messages, debug dumps, and session memory
Centralized error handling for fail-fast strategy
"""

import os
import sys
import json
import traceback
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

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
        
        # Log the paths being used (helpful for debugging)
        print(f"RumiAI Error Handler initialized:")
        print(f"  Log directory: {self.log_dir}")
        print(f"  Debug directory: {self.debug_dir}")
    
    def handle_contract_violation(self, service: str, expected: Any, got: Any, context: dict):
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
    
    def _create_actionable_message(self, service: str, expected: Any, got: Any) -> str:
        """Generate user-friendly error message with fix instructions"""
        
        messages = {
            'ocr': f"""
OCR Output Format Error:
  Expected: Flat bbox format [x, y, width, height]
  Got: {type(got)} format
  
  Fix:
  1. Check OCR fix was applied: grep -n "bbox_list" rumiai_v2/api/ml_services_unified.py
  2. Clear OCR cache: rm -rf creative_analysis_outputs/*
  3. Verify EasyOCR version: pip show easyocr
  4. Check debug dump for full output structure
            """,
            'yolo': f"""
YOLO Output Format Error:
  Expected: {expected}
  Got: Invalid format
  
  Fix:
  1. Check YOLO model version: yolo version
  2. Clear detection cache: rm -rf object_detection_outputs/*
  3. Verify model file: ls -la yolov8n.pt
  4. Check GPU memory: nvidia-smi
            """,
            'mediapipe': f"""
MediaPipe Output Format Error:
  Expected: poses[], faces[], hands[] structure
  Got: Invalid format
  
  Fix:
  1. Check MediaPipe version: pip show mediapipe
  2. Clear analysis cache: rm -rf human_analysis_outputs/*
  3. Test with sample image: python -m mediapipe.solutions.pose
            """,
            'whisper': f"""
Whisper Output Format Error:
  Expected: text, segments[], language, duration
  Got: Invalid format
  
  Fix:
  1. Check Whisper.cpp build: whisper.cpp/main --help
  2. Clear transcription cache: rm -rf speech_transcriptions/*
  3. Test with sample audio: whisper.cpp/main -m base.bin sample.wav
            """,
            'audio_energy': f"""
Audio Energy Output Format Error:
  Expected: energy_level_windows, burst_pattern
  Got: Invalid format
  
  Fix:
  1. Check librosa version: pip show librosa
  2. Clear energy cache: rm -rf ml_outputs/*audio_energy*
  3. Verify audio extraction: ffmpeg -i video.mp4 -acodec pcm_s16le test.wav
            """,
            'scene_detection': f"""
Scene Detection Output Format Error:
  Expected: scenes[] with continuous timing
  Got: Invalid format
  
  Fix:
  1. Check PySceneDetect version: pip show scenedetect
  2. Clear scene cache: rm -rf scene_detection_outputs/*
  3. Test detection: scenedetect -i video.mp4 detect-content list-scenes
            """,
            'claude': f"""
Claude Response Format Error:
  Expected: 6 blocks (CoreMetrics, Dynamics, Interactions, KeyEvents, Patterns, Quality)
  Got: Invalid structure
  
  Fix:
  1. Check prompt template: cat prompt_templates/creative_density_v2.txt
  2. Verify API key: echo $ANTHROPIC_API_KEY | head -c 10
  3. Test with minimal prompt: curl https://api.anthropic.com/v1/messages
  4. Check rate limits and quotas
            """
        }
        
        return messages.get(service, f"Service {service} contract violation\nExpected: {expected}\nGot: {got}")
    
    def _create_debug_dump(self, service: str, context: dict, expected: Any, got: Any) -> str:
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
        
        # If we have the actual output, save it
        if isinstance(got, dict):
            with open(dump_path / "actual_output.json", 'w') as f:
                json.dump(got, f, indent=2, default=str)
        
        return dump_id
    
    def _log_structured(self, error_data: dict):
        """Log error in structured format (newline-delimited JSON)"""
        date = datetime.now().strftime("%Y-%m-%d")
        error_file = self.log_dir / "errors" / f"{date}_errors.json"
        
        with open(error_file, 'a') as f:
            f.write(json.dumps(error_data) + '\n')
    
    def _log_human_readable(self, message: str):
        """Log error in human-readable format"""
        date = datetime.now().strftime("%Y-%m-%d")
        error_file = self.log_dir / "errors" / f"{date}_errors.txt"
        
        with open(error_file, 'a') as f:
            f.write(message + '\n')
    
    def handle_timeout(self, service: str, timeout: int, context: dict):
        """Handle service timeout errors"""
        
        error_msg = f"""
{service.upper()} Service Timeout:
  Service failed to complete within {timeout} seconds
  
  This usually indicates:
  - File too large for processing
  - System resource constraints
  - Model loading issues
  
  Try:
  1. Check system resources: top, nvidia-smi
  2. Process smaller files first
  3. Increase timeout in configuration
  4. Check service logs: tail -f logs/pipeline/*.log
        """
        
        dump_id = self._create_debug_dump(service, context, f"completion within {timeout}s", "timeout")
        
        self._log_structured({
            'timestamp': datetime.now().isoformat(),
            'error_type': 'timeout',
            'service': service,
            'video_id': context.get('video_id'),
            'dump_id': dump_id,
            'timeout': timeout
        })
        
        print(f"\n⏱️ {service} timeout after {timeout}s\n📁 Debug saved: {dump_id}\n")
        sys.exit(11)  # Timeout exit code
    
    def handle_model_load_failure(self, service: str, error: Exception, context: dict):
        """Handle model loading failures"""
        
        error_msg = f"""
{service.upper()} Model Load Failure:
  {str(error)}
  
  Fix:
  1. Check model file exists
  2. Verify CUDA/GPU availability if needed
  3. Check available memory
  4. Reinstall the service dependencies
        """
        
        dump_id = self._create_debug_dump(service, context, "model loaded", str(error))
        
        self._log_structured({
            'timestamp': datetime.now().isoformat(),
            'error_type': 'model_load_failure',
            'service': service,
            'error': str(error),
            'dump_id': dump_id
        })
        
        print(f"\n🔴 {service} model load failed\n📁 Debug saved: {dump_id}\n")
        sys.exit(12)  # Model load failure exit code