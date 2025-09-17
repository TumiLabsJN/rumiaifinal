"""
Simple DeepFace Gender Detection Service

This implementation uses subprocess isolation to avoid memory corruption issues
with TensorFlow in mixed ML environments.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any
import subprocess

logger = logging.getLogger(__name__)


class DeepFaceGenderServiceSimple:
    """
    Simple wrapper that runs DeepFace in subprocess to avoid memory issues.
    """

    def __init__(self):
        self.script_path = Path(__file__).parent.parent.parent / 'scripts' / 'run_deepface_gender.py'
        if not self.script_path.exists():
            raise FileNotFoundError(f"DeepFace runner script not found: {self.script_path}")

    async def analyze(self, video_path: str) -> Dict[str, Any]:
        """Run gender detection via subprocess."""

        # Check video exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            # Run analysis in subprocess
            cmd = ['python3', str(self.script_path), video_path]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30  # 30 second timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    'gender': None,
                    'confidence': 0.0,
                    'method': 'deepface',
                    'error': 'timeout_30s'
                }

            if process.returncode != 0:
                logger.error(f"DeepFace subprocess failed: {stderr.decode()}")
                return {
                    'gender': None,
                    'confidence': 0.0,
                    'method': 'deepface',
                    'error': 'subprocess_failed'
                }

            # Parse JSON output (script outputs JSON to stdout)
            output = stdout.decode()
            # Find JSON in output (skip any warnings)
            json_start = output.find('{')
            if json_start >= 0:
                json_str = output[json_start:]
                return json.loads(json_str)
            else:
                logger.error(f"No JSON output from DeepFace script")
                return {
                    'gender': None,
                    'confidence': 0.0,
                    'method': 'deepface',
                    'error': 'no_json_output'
                }

        except Exception as e:
            logger.error(f"DeepFace subprocess error: {e}")
            return {
                'gender': None,
                'confidence': 0.0,
                'method': 'deepface',
                'error': str(e)
            }

    async def health_check(self) -> Dict[str, Any]:
        """Check if service is healthy."""
        return {
            'service': 'deepface_gender_simple',
            'status': 'healthy' if self.script_path.exists() else 'unhealthy',
            'mode': 'subprocess',
            'script': str(self.script_path)
        }