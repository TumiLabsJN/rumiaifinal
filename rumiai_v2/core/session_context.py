"""
Session Context Loader
Load context from previous errors/runs for new CLI sessions
Logs become persistent knowledge base between sessions
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
        elif self._detect_timeout_pattern(context):
            context['suggestion'] = "Multiple timeouts detected. Consider: 1) Reducing video size, 2) Increasing timeout limits"
        elif self._detect_model_load_pattern(context):
            context['suggestion'] = "Model loading issues detected. Check GPU memory: nvidia-smi"
        
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
    
    def _get_recent_errors(self, limit: int = 5) -> List[Dict]:
        """Get most recent errors with full context"""
        errors = []
        
        # Check today and previous days until we have enough errors
        for day in range(30):  # Look back up to 30 days
            if len(errors) >= limit:
                break
                
            date = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
            error_file = self.log_dir / "errors" / f"{date}_errors.json"
            
            if error_file.exists():
                try:
                    with open(error_file) as f:
                        day_errors = []
                        for line in f:
                            day_errors.append(json.loads(line))
                        
                        # Add most recent first
                        day_errors.reverse()
                        errors.extend(day_errors[:limit - len(errors)])
                except Exception:
                    continue
        
        return errors[:limit]
    
    def _get_last_success(self) -> Optional[Dict]:
        """Find the last successful pipeline run"""
        
        # Check pipeline logs for successful runs
        for day in range(7):
            date = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
            pipeline_log = self.log_dir / "pipeline" / f"{date}_pipeline.log"
            
            if pipeline_log.exists():
                try:
                    with open(pipeline_log) as f:
                        # Read lines in reverse to find most recent success
                        lines = f.readlines()
                        for line in reversed(lines):
                            if "Pipeline completed successfully" in line:
                                # Parse the success entry
                                return {
                                    'date': date,
                                    'message': line.strip()
                                }
                except Exception:
                    continue
        
        return None
    
    def _get_video_history(self, video_id: str) -> List[Dict]:
        """Get processing history for a specific video"""
        history = []
        
        # Search all error logs for this video
        for day in range(30):
            date = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
            error_file = self.log_dir / "errors" / f"{date}_errors.json"
            
            if error_file.exists():
                try:
                    with open(error_file) as f:
                        for line in f:
                            error = json.loads(line)
                            if error.get('video_id') == video_id:
                                history.append(error)
                except Exception:
                    continue
        
        return history
    
    def _detect_ocr_pattern(self, context: dict) -> bool:
        """Detect if OCR errors are recurring"""
        patterns = context.get('known_issues', {})
        ocr_errors = patterns.get('contract_violation', 0)
        
        # Check if recent errors include OCR
        recent = context.get('recent_errors', [])
        recent_ocr = sum(1 for e in recent if e.get('service') == 'ocr')
        
        # If more than 3 OCR errors in last 7 days or 2+ in recent
        return ocr_errors > 3 or recent_ocr >= 2
    
    def _detect_timeout_pattern(self, context: dict) -> bool:
        """Detect if timeout errors are recurring"""
        patterns = context.get('known_issues', {})
        timeout_errors = patterns.get('timeout', 0)
        
        # If more than 5 timeouts in last 7 days
        return timeout_errors > 5
    
    def _detect_model_load_pattern(self, context: dict) -> bool:
        """Detect if model loading errors are recurring"""
        patterns = context.get('known_issues', {})
        model_errors = patterns.get('model_load_failure', 0)
        
        # If more than 2 model load failures in last 7 days
        return model_errors > 2
    
    def get_summary_report(self) -> str:
        """Generate a summary report of recent activity"""
        context = self.get_recent_context()
        patterns = context.get('known_issues', {})
        recent_errors = context.get('recent_errors', [])
        last_success = context.get('last_successful_run')
        
        report = ["=== RumiAI Pipeline Status Report ===\n"]
        
        # Last success
        if last_success:
            report.append(f"✅ Last successful run: {last_success['date']}")
        else:
            report.append("⚠️ No successful runs in the last 7 days")
        
        # Error patterns
        if patterns:
            report.append("\n📊 Error patterns (last 7 days):")
            for error_type, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  - {error_type}: {count} occurrences")
        
        # Recent errors
        if recent_errors:
            report.append(f"\n🔴 Recent errors ({len(recent_errors)}):")
            for error in recent_errors[:3]:  # Show top 3
                report.append(f"  - {error.get('service', 'unknown')}: {error.get('error_type', 'unknown')}")
                if error.get('video_id'):
                    report.append(f"    Video: {error['video_id']}")
        
        # Suggestions
        if context.get('suggestion'):
            report.append(f"\n💡 Suggestion: {context['suggestion']}")
        
        report.append("\n" + "=" * 40)
        
        return "\n".join(report)
    
    def get_video_report(self, video_id: str) -> str:
        """Generate a report for a specific video"""
        context = self.get_recent_context(video_id)
        history = context.get('video_history', [])
        
        report = [f"=== Video Processing History: {video_id} ===\n"]
        
        if not history:
            report.append("No processing history found for this video")
        else:
            report.append(f"Processing attempts: {len(history)}")
            
            # Group by error type
            error_types = {}
            for error in history:
                error_type = error.get('error_type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            report.append("\nError breakdown:")
            for error_type, count in error_types.items():
                report.append(f"  - {error_type}: {count}")
            
            # Most recent error
            if history:
                latest = history[0]
                report.append(f"\nMost recent error:")
                report.append(f"  Type: {latest.get('error_type')}")
                report.append(f"  Service: {latest.get('service')}")
                report.append(f"  Time: {latest.get('timestamp')}")
                if latest.get('dump_id'):
                    report.append(f"  Debug dump: {latest['dump_id']}")
        
        return "\n".join(report)