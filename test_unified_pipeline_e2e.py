#!/usr/bin/env python3
"""
End-to-End Test for Unified ML Pipeline
Tests the complete flow: Frame Extraction → ML Processing → ml_data → Claude Analysis
"""

import asyncio
import argparse
import json
import time
import psutil
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add rumiai_v2 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rumiai_v2.api.ml_services import MLServices
from rumiai_v2.processors.video_analyzer import VideoAnalyzer
from rumiai_v2.processors.timeline_builder import TimelineBuilder
from rumiai_v2.processors.ml_data_extractor import MLDataExtractor
from rumiai_v2.processors.prompt_builder import PromptBuilder
from rumiai_v2.processors.precompute_functions import get_compute_function
from rumiai_v2.api.claude_client import ClaudeClient
from rumiai_v2.validators.response_validator import ResponseValidator
from rumiai_v2.core.models.prompt import PromptType
from rumiai_v2.prompts.prompt_manager import PromptManager
from rumiai_v2.config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedPipelineE2ETester:
    """Comprehensive E2E tester for unified ML pipeline"""
    
    def __init__(self):
        # Load settings
        self.settings = Settings()
        
        self.ml_services = MLServices()
        self.video_analyzer = VideoAnalyzer(self.ml_services)
        self.timeline_builder = TimelineBuilder()
        self.ml_extractor = MLDataExtractor()
        
        # Initialize prompt manager and builder
        self.prompt_manager = PromptManager()
        self.prompt_builder = PromptBuilder(self.prompt_manager.templates)
        
        # Initialize Claude client with API key
        if not self.settings.claude_api_key:
            raise ValueError("CLAUDE_API_KEY environment variable not set")
        self.claude_client = ClaudeClient(self.settings.claude_api_key)
        
        self.validator = ResponseValidator()
        
        # Performance tracking
        self.metrics = {
            'frame_extraction_time': 0,
            'ml_processing_time': {},
            'claude_processing_time': {},
            'total_time': 0,
            'memory_peak_mb': 0,
            'memory_start_mb': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'json_sizes_kb': {},
            'token_counts': {},
            'api_costs': {},
            'total_cost': 0,
            'timeouts': {
                'frame_extraction': False,
                'whisper': False,
                'yolo': False,
                'mediapipe': False,
                'ocr': False,
                'scene': False,
                'claude_calls': {}  # Per flow
            }
        }
        
        # Validation results
        self.validation_results = {
            'ml_data_present': False,
            'ml_data_populated': {},
            'claude_outputs': {},
            'block_validation': {},
            'errors': []
        }
        
    async def test_video(self, video_path: Path) -> Dict[str, Any]:
        """Run complete pipeline test on a single video"""
        print(f"\n{'='*60}")
        print(f"Testing video: {video_path.name}")
        print(f"{'='*60}")
        
        # Track overall time and memory
        start_time = time.time()
        self.metrics['memory_start_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Step 1: Extract video metadata
            video_id = video_path.stem
            duration = await self._get_video_duration(video_path)
            print(f"Video ID: {video_id}")
            print(f"Duration: {duration:.1f} seconds")
            
            # Step 2: Run ML analysis (includes unified frame extraction)
            print("\n--- ML Analysis Phase ---")
            ml_results = await self._run_ml_analysis(video_id, video_path)
            
            # Step 3: Build unified analysis
            print("\n--- Building Unified Analysis ---")
            video_metadata = {
                'video_id': video_id,
                'duration': duration,
                'width': 1920,  # Would get from actual video
                'height': 1080,
                'fps': 30.0
            }
            
            unified_analysis = self.timeline_builder.build_timeline(
                video_id, video_metadata, ml_results
            )
            
            # Step 4: Verify ml_data field
            print("\n--- Verifying ml_data Field ---")
            analysis_dict = unified_analysis.to_dict(legacy_mode=False)
            self._verify_ml_data(analysis_dict)
            
            # Save unified analysis for inspection
            unified_path = Path(f"test_outputs/{video_id}_unified_analysis.json")
            unified_path.parent.mkdir(exist_ok=True)
            with open(unified_path, 'w') as f:
                json.dump(analysis_dict, f, indent=2)
            print(f"Saved unified analysis to: {unified_path}")
            
            # Step 5: Run all 7 Claude analysis flows
            print("\n--- Claude Analysis Phase ---")
            await self._run_claude_analyses(video_id, analysis_dict, unified_analysis)
            
            # Step 6: Calculate totals
            self.metrics['total_time'] = time.time() - start_time
            self.metrics['memory_peak_mb'] = max(
                self.metrics['memory_peak_mb'],
                psutil.Process().memory_info().rss / 1024 / 1024
            )
            
            # Generate report
            return self._generate_report(video_id)
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            self.validation_results['errors'].append(str(e))
            return self._generate_report(video_id, failed=True)
    
    async def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using OpenCV"""
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0
    
    async def _run_ml_analysis(self, video_id: str, video_path: Path) -> Dict[str, Any]:
        """Run ML analysis and track metrics"""
        ml_start = time.time()
        
        try:
            # This will use unified frame extraction internally
            ml_results = await asyncio.wait_for(
                self.video_analyzer.analyze_video(video_id, video_path),
                timeout=300  # 5 minute overall ML timeout
            )
        except asyncio.TimeoutError:
            self.metrics['timeouts']['ml_processing'] = True
            logger.error("ML analysis timed out after 5 minutes")
            # Return empty results but continue
            ml_results = {}
        
        # Track individual ML service times (would need to instrument internally)
        self.metrics['ml_processing_time'] = {
            'total': time.time() - ml_start,
            'yolo': 0,  # Would track internally
            'mediapipe': 0,
            'ocr': 0,
            'whisper': 0,
            'scene': 0
        }
        
        # Check for specific service timeouts
        for service in ['whisper', 'yolo', 'mediapipe', 'ocr', 'scene_detection']:
            if service in ml_results:
                result = ml_results[service]
                # Check if the service returned a timeout error
                if hasattr(result, 'data') and isinstance(result.data, dict):
                    if result.data.get('error', '').startswith('Timeout'):
                        self.metrics['timeouts'][service] = True
                        logger.warning(f"{service} timed out but pipeline continued")
        
        # Check frame manager cache stats
        frame_manager = self.ml_services.unified_services.frame_manager
        # Would need to add cache stats tracking to frame_manager
        
        return ml_results
    
    def _verify_ml_data(self, analysis_dict: Dict[str, Any]):
        """Verify ml_data field exists and contains real data"""
        # Check field exists
        self.validation_results['ml_data_present'] = 'ml_data' in analysis_dict
        
        if not self.validation_results['ml_data_present']:
            self.validation_results['errors'].append("ml_data field missing!")
            return
        
        ml_data = analysis_dict['ml_data']
        
        # Check each service has data
        for service in ['yolo', 'mediapipe', 'ocr', 'whisper', 'scene_detection']:
            if service in ml_data:
                data = ml_data[service]
                
                # Check if populated (not empty)
                is_populated = False
                if service == 'yolo' and data.get('objectAnnotations'):
                    is_populated = len(data['objectAnnotations']) > 0
                elif service == 'mediapipe' and (data.get('poses') or data.get('faces')):
                    is_populated = True
                elif service == 'ocr' and data.get('textAnnotations'):
                    is_populated = len(data['textAnnotations']) > 0
                elif service == 'whisper' and data.get('text'):
                    is_populated = len(data['text'].strip()) > 0
                elif service == 'scene_detection' and data.get('scenes'):
                    is_populated = len(data['scenes']) > 0
                
                self.validation_results['ml_data_populated'][service] = is_populated
                
                if is_populated:
                    print(f"✅ {service}: Contains real data")
                else:
                    print(f"⚠️  {service}: Empty (may be expected)")
            else:
                self.validation_results['ml_data_populated'][service] = False
                print(f"❌ {service}: Missing from ml_data")
    
    async def _run_claude_analyses(self, video_id: str, analysis_dict: Dict[str, Any], unified_analysis=None):
        """Run all 7 Claude analysis flows"""
        prompt_types = [
            PromptType.CREATIVE_DENSITY,
            PromptType.EMOTIONAL_JOURNEY,
            PromptType.PERSON_FRAMING,
            PromptType.SCENE_PACING,
            PromptType.SPEECH_ANALYSIS,
            PromptType.VISUAL_OVERLAY,
            PromptType.METADATA_ANALYSIS
        ]
        
        for prompt_type in prompt_types:
            print(f"\n--- Testing {prompt_type.value} ---")
            
            try:
                # Extract ML data for this prompt
                if unified_analysis:
                    # Use the UnifiedAnalysis object if provided
                    context = self.ml_extractor.extract_for_prompt(
                        unified_analysis, prompt_type
                    )
                else:
                    # Fallback: create a minimal context from dict
                    from rumiai_v2.core.models import PromptContext
                    context = PromptContext(
                        video_id=video_id,
                        prompt_type=prompt_type,
                        duration=analysis_dict.get('duration', 0),
                        metadata=analysis_dict.get('metadata', {}),
                        ml_data=analysis_dict.get('ml_data', {}),
                        timelines=analysis_dict.get('timeline', {})
                    )
                
                # Run precompute function
                compute_fn = get_compute_function(prompt_type)
                if compute_fn:
                    precomputed = compute_fn(analysis_dict)
                    # Add precomputed data to the ml_data
                    if isinstance(precomputed, dict):
                        context.ml_data.update(precomputed)
                
                # Build prompt
                prompt = self.prompt_builder.build_prompt(context)
                
                # Track prompt size
                prompt_size_kb = len(prompt.encode('utf-8')) / 1024
                self.metrics['json_sizes_kb'][prompt_type.value] = prompt_size_kb
                print(f"Prompt size: {prompt_size_kb:.1f} KB")
                
                # Call Claude
                claude_start = time.time()
                try:
                    # Create context data for Claude
                    context_data = {
                        'prompt_type': prompt_type.value,
                        'video_id': video_id
                    }
                    
                    # Call Claude with timeout
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.claude_client.send_prompt,
                            prompt,
                            context_data,
                            timeout=60
                        ),
                        timeout=60  # 1 minute timeout per Claude call
                    )
                except asyncio.TimeoutError:
                    self.metrics['timeouts']['claude_calls'][prompt_type.value] = True
                    logger.error(f"Claude call timed out for {prompt_type.value}")
                    self.validation_results['claude_outputs'][prompt_type.value] = False
                    self.validation_results['errors'].append(f"{prompt_type.value}: Claude timeout after 60s")
                    continue
                    
                claude_time = time.time() - claude_start
                self.metrics['claude_processing_time'][prompt_type.value] = claude_time
                
                # Track tokens and cost from response
                if response.success and response.tokens_used > 0:
                    # The response already has tokens and cost info
                    self.metrics['token_counts'][prompt_type.value] = {
                        'input': response.tokens_used,  # We don't have separate counts
                        'output': 0  # Can't differentiate without metadata
                    }
                    self.metrics['api_costs'][prompt_type.value] = response.estimated_cost
                    self.metrics['total_cost'] += response.estimated_cost
                    
                    print(f"Total tokens: {response.tokens_used}")
                    print(f"Cost: ${response.estimated_cost:.4f}")
                
                # Validate response
                is_valid, parsed_data, errors = self.validator.validate_response(
                    response.response, prompt_type.value
                )
                
                if is_valid:
                    print(f"✅ Valid 6-block structure")
                    self._deep_validate_blocks(parsed_data, prompt_type)
                else:
                    print(f"❌ Invalid structure: {errors}")
                    self.validation_results['errors'].append(
                        f"{prompt_type.value}: {', '.join(errors)}"
                    )
                
                # Save output
                output_path = Path(f"test_outputs/{video_id}_{prompt_type.value}.json")
                with open(output_path, 'w') as f:
                    json.dump(parsed_data if is_valid else {}, f, indent=2)
                
                self.validation_results['claude_outputs'][prompt_type.value] = is_valid
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                self.validation_results['claude_outputs'][prompt_type.value] = False
                self.validation_results['errors'].append(f"{prompt_type.value}: {str(e)}")
    
    def _deep_validate_blocks(self, data: Dict[str, Any], prompt_type: PromptType):
        """Deep validation of 6-block structure"""
        expected_blocks = self._get_expected_blocks(prompt_type)
        block_results = {}
        
        for block_name in expected_blocks:
            if block_name in data:
                block_data = data[block_name]
                
                # Validate data types and ranges
                issues = []
                
                # Check confidence scores
                if 'confidence' in block_data:
                    conf = block_data['confidence']
                    if not (0.0 <= conf <= 1.0):
                        issues.append(f"Confidence {conf} out of range")
                
                # Check counts are non-negative
                for key, value in block_data.items():
                    if 'count' in key.lower() and isinstance(value, (int, float)):
                        if value < 0:
                            issues.append(f"{key} is negative: {value}")
                
                # Check arrays have reasonable lengths
                for key, value in block_data.items():
                    if isinstance(value, list) and len(value) > 1000:
                        issues.append(f"{key} has {len(value)} items (suspicious)")
                
                block_results[block_name] = {
                    'present': True,
                    'issues': issues
                }
            else:
                block_results[block_name] = {
                    'present': False,
                    'issues': ['Block missing']
                }
        
        self.validation_results['block_validation'][prompt_type.value] = block_results
    
    def _get_expected_blocks(self, prompt_type: PromptType) -> List[str]:
        """Get expected block names for each prompt type"""
        prefix_map = {
            PromptType.CREATIVE_DENSITY: 'density',
            PromptType.EMOTIONAL_JOURNEY: 'emotional',
            PromptType.PERSON_FRAMING: 'personFraming',
            PromptType.SCENE_PACING: 'scenePacing',
            PromptType.SPEECH_ANALYSIS: 'speech',
            PromptType.VISUAL_OVERLAY: 'overlays',
            PromptType.METADATA_ANALYSIS: 'metadata'
        }
        
        prefix = prefix_map.get(prompt_type, 'unknown')
        
        return [
            f"{prefix}CoreMetrics",
            f"{prefix}Dynamics",
            f"{prefix}Interactions",
            f"{prefix}KeyEvents",
            f"{prefix}Patterns",
            f"{prefix}Quality"
        ]
    
    def _generate_report(self, video_id: str, failed: bool = False) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("TEST REPORT")
        print("="*60)
        
        # Performance metrics
        print("\n--- Performance Metrics ---")
        print(f"Total time: {self.metrics['total_time']:.2f} seconds")
        print(f"ML processing: {self.metrics['ml_processing_time'].get('total', 0):.2f} seconds")
        print(f"Memory usage: {self.metrics['memory_start_mb']:.0f} MB → {self.metrics['memory_peak_mb']:.0f} MB")
        print(f"Cache performance: {self.metrics['cache_hits']} hits, {self.metrics['cache_misses']} misses")
        
        # Cost analysis
        print("\n--- Cost Analysis ---")
        print(f"Total API cost: ${self.metrics['total_cost']:.4f}")
        for flow, cost in self.metrics['api_costs'].items():
            tokens = self.metrics['token_counts'].get(flow, {})
            print(f"  {flow}: ${cost:.4f} ({tokens.get('input', 0)} in / {tokens.get('output', 0)} out)")
        
        # Timeout events
        print("\n--- Timeout Events ---")
        timeout_occurred = False
        for service, timed_out in self.metrics['timeouts'].items():
            if service == 'claude_calls':
                for flow, flow_timeout in timed_out.items():
                    if flow_timeout:
                        print(f"⏱️  Claude {flow}: Timed out after 60s")
                        timeout_occurred = True
            elif timed_out:
                print(f"⏱️  {service}: Timed out (gracefully handled)")
                timeout_occurred = True
        
        if not timeout_occurred:
            print("No timeouts occurred")
        
        # Validation results
        print("\n--- Validation Results ---")
        print(f"ml_data field present: {'✅' if self.validation_results['ml_data_present'] else '❌'}")
        
        print("\nML data population:")
        for service, populated in self.validation_results['ml_data_populated'].items():
            print(f"  {service}: {'✅ Has data' if populated else '⚠️  Empty'}")
        
        print("\nClaude analysis results:")
        for flow, valid in self.validation_results['claude_outputs'].items():
            print(f"  {flow}: {'✅ Valid' if valid else '❌ Invalid'}")
        
        # Errors
        if self.validation_results['errors']:
            print("\n--- Errors ---")
            for error in self.validation_results['errors']:
                print(f"❌ {error}")
        
        # Overall result
        all_claude_valid = all(self.validation_results['claude_outputs'].values())
        ml_data_ok = self.validation_results['ml_data_present']
        has_real_data = any(self.validation_results['ml_data_populated'].values())
        
        success = all_claude_valid and ml_data_ok and has_real_data and not failed
        
        print("\n" + "="*60)
        print(f"OVERALL RESULT: {'✅ PASS' if success else '❌ FAIL'}")
        print("="*60)
        
        # Save detailed report
        report = {
            'video_id': video_id,
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'metrics': self.metrics,
            'validation': self.validation_results
        }
        
        report_path = Path(f"test_outputs/{video_id}_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        
        return report

async def main():
    parser = argparse.ArgumentParser(description='Test Unified ML Pipeline E2E')
    parser.add_argument('video_path', type=Path, help='Path to video file to test')
    parser.add_argument('--save-outputs', action='store_true', 
                       help='Save all intermediate outputs')
    
    args = parser.parse_args()
    
    # Validate video exists
    if not args.video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Create output directory
    Path("test_outputs").mkdir(exist_ok=True)
    
    # Run test
    tester = UnifiedPipelineE2ETester()
    report = await tester.test_video(args.video_path)
    
    # Exit with appropriate code
    sys.exit(0 if report.get('success') else 1)

if __name__ == "__main__":
    asyncio.run(main())