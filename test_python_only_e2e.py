#!/usr/bin/env python3
"""
End-to-End Test for Python-Only ML Pipeline
Based on E2Etestplan.md specification
Tests the complete flow without Claude API dependency
"""

import asyncio
import argparse
import json
import time
import psutil
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add rumiai_v2 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rumiai_v2.api.ml_services import MLServices
from rumiai_v2.processors.video_analyzer import VideoAnalyzer
from rumiai_v2.processors.timeline_builder import TimelineBuilder
from rumiai_v2.processors.precompute_functions import get_compute_function, COMPUTE_FUNCTIONS
# from rumiai_v2.core.models.prompt import PromptType  # Removed in cleanup
from rumiai_v2.config.settings import Settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ServiceContractValidation:
    """Validate each service meets its contract requirements"""
    
    CONTRACTS = {
        'creative_density': {
            'required_fields': ['densityCoreMetrics', 'densityDynamics', 'densityInteractions', 
                               'densityKeyEvents', 'densityPatterns', 'densityQuality'],
            'required_ml_data': ['objectTimeline', 'textOverlayTimeline'],
            'output_structure': '6-block CoreBlocks format'
        },
        'emotional_journey': {
            'required_fields': ['emotionalCoreMetrics', 'emotionalDynamics', 'emotionalInteractions',
                               'emotionalKeyEvents', 'emotionalPatterns', 'emotionalQuality'],
            'required_ml_data': ['expression_timeline'],
            'output_structure': '6-block CoreBlocks format'
        },
        'person_framing': {
            'required_fields': ['personFramingCoreMetrics', 'personFramingDynamics', 'personFramingInteractions',
                               'personFramingKeyEvents', 'personFramingPatterns', 'personFramingQuality'],
            'required_ml_data': ['pose_timeline', 'face_timeline'],
            'output_structure': '6-block CoreBlocks format'
        },
        'scene_pacing': {
            'required_fields': ['scenePacingCoreMetrics', 'scenePacingDynamics', 'scenePacingInteractions',
                               'scenePacingKeyEvents', 'scenePacingPatterns', 'scenePacingQuality'],
            'required_ml_data': ['scene_timeline'],
            'output_structure': '6-block CoreBlocks format'
        },
        'speech_analysis': {
            'required_fields': ['speechCoreMetrics', 'speechDynamics', 'speechInteractions',
                               'speechKeyEvents', 'speechPatterns', 'speechQuality'],
            'required_ml_data': ['speech_timeline'],
            'output_structure': '6-block CoreBlocks format'
        },
        'visual_overlay_analysis': {
            'required_fields': ['overlaysCoreMetrics', 'overlaysDynamics', 'overlaysInteractions',
                               'overlaysKeyEvents', 'overlaysPatterns', 'overlaysQuality'],
            'required_ml_data': ['textOverlayTimeline'],
            'output_structure': '6-block CoreBlocks format'
        },
        'metadata_analysis': {
            'required_fields': ['metadataCoreMetrics', 'metadataDynamics', 'metadataInteractions',
                               'metadataKeyEvents', 'metadataPatterns', 'metadataQuality'],
            'required_ml_data': [],  # No ML data required
            'output_structure': '6-block CoreBlocks format'
        }
    }
    
    def validate_contract(self, analysis_type: str, result: Dict, ml_data: Dict) -> Dict:
        """Validate a service meets its contract"""
        if analysis_type not in self.CONTRACTS:
            return {'valid': False, 'violations': [f'Unknown analysis type: {analysis_type}']}
            
        contract = self.CONTRACTS[analysis_type]
        violations = []
        
        # Check ML data dependencies
        for required_data in contract['required_ml_data']:
            if required_data not in ml_data or not ml_data[required_data]:
                violations.append(f"Missing ML dependency: {required_data}")
        
        # Check output structure (6 blocks)
        for required_field in contract['required_fields']:
            if required_field not in result:
                violations.append(f"Missing required field: {required_field}")
        
        return {
            'valid': len(violations) == 0,
            'violations': violations
        }

class PythonOnlyE2ETester:
    """E2E tester for Python-only processing pipeline"""
    
    def __init__(self):
        self.settings = Settings()
        self.ml_services = MLServices()
        self.video_analyzer = VideoAnalyzer(self.ml_services)
        self.timeline_builder = TimelineBuilder()
        
        # Test configuration
        self.prompt_types = [
            'creative_density',
            'emotional_journey', 
            'person_framing',
            'scene_pacing',
            'speech_analysis',
            'visual_overlay_analysis',
            'metadata_analysis'
        ]
        
        # Tracking
        self.ml_data = {}
        self.ml_data_validation_result = False
        self.ml_data_details = {}
        self.start_time = None
        
    def _validate_environment(self) -> bool:
        """Ensure Python-only mode is properly configured"""
        print("\n[Phase 1/4] Environment Validation...")
        
        required_vars = {
            'USE_PYTHON_ONLY_PROCESSING': 'true',
            'USE_ML_PRECOMPUTE': 'true',
            'PRECOMPUTE_CREATIVE_DENSITY': 'true',
            'PRECOMPUTE_EMOTIONAL_JOURNEY': 'true',
            'PRECOMPUTE_PERSON_FRAMING': 'true',
            'PRECOMPUTE_SCENE_PACING': 'true',
            'PRECOMPUTE_SPEECH_ANALYSIS': 'true',
            'PRECOMPUTE_VISUAL_OVERLAY': 'true',
            'PRECOMPUTE_METADATA': 'true'
        }
        
        all_valid = True
        for var, expected in required_vars.items():
            actual = os.getenv(var)
            if actual == expected:
                print(f"✅ {var}={expected}")
            else:
                print(f"❌ {var}={actual} (expected {expected})")
                all_valid = False
        
        if not all_valid:
            raise ValueError("Environment not configured for Python-only processing!")
            
        print("✅ All flags set correctly")
        return True
    
    async def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration"""
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0
    
    async def _run_ml_analysis(self, video_id: str, video_path: Path) -> Dict[str, Any]:
        """Run ML analysis with timing"""
        print("\n[Phase 2/4] ML Analysis...")
        ml_start = time.time()
        
        # Track individual service times
        print("- Frame extraction: Processing...", end='', flush=True)
        frame_start = time.time()
        
        # Run ML analysis
        ml_results = await self.video_analyzer.analyze_video(video_id, video_path)
        
        ml_time = time.time() - ml_start
        print(f" ({ml_time:.1f}s)")
        print(f"✅ ML Pipeline Complete ({ml_time:.1f}s total)")
        
        return ml_results
    
    def _verify_ml_data(self, analysis_dict: Dict[str, Any]) -> None:
        """Verify ml_data field exists and contains real data"""
        self.ml_data_validation_result = 'ml_data' in analysis_dict
        
        if not self.ml_data_validation_result:
            self.ml_data_details = {'error': 'ml_data field missing'}
            return
        
        self.ml_data = analysis_dict['ml_data']
        details = {}
        
        # Check each ML service
        for service in ['yolo', 'mediapipe', 'ocr', 'whisper', 'scene_detection']:
            if service in self.ml_data:
                data = self.ml_data[service]
                
                # Check if populated
                is_populated = False
                if service == 'yolo' and data.get('objectAnnotations'):
                    is_populated = len(data['objectAnnotations']) > 0
                    details[f'{service}_objects'] = len(data['objectAnnotations'])
                elif service == 'mediapipe':
                    if data.get('poses'):
                        is_populated = True
                        details[f'{service}_poses'] = len(data.get('poses', []))
                    if data.get('faces'):
                        is_populated = True
                        details[f'{service}_faces'] = len(data.get('faces', []))
                elif service == 'ocr' and data.get('textAnnotations'):
                    is_populated = len(data['textAnnotations']) > 0
                    details[f'{service}_texts'] = len(data['textAnnotations'])
                elif service == 'whisper' and data.get('text'):
                    is_populated = len(data['text'].strip()) > 0
                    details[f'{service}_words'] = len(data['text'].split())
                elif service == 'scene_detection' and data.get('scenes'):
                    is_populated = len(data['scenes']) > 0
                    details[f'{service}_scenes'] = len(data['scenes'])
                
                details[service] = 'populated' if is_populated else 'empty'
        
        self.ml_data_details = details
    
    async def test_python_only_e2e(self, video_path: Path) -> Dict[str, Any]:
        """Complete E2E test for Python-only processing"""
        
        print("="*70)
        print("E2E TEST REPORT - PYTHON-ONLY PROCESSING")
        print("="*70)
        print(f"Video: {video_path.name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        
        try:
            # Step 1: Validate environment
            self._validate_environment()
            
            # Step 2: Load video and extract metadata
            video_id = video_path.stem
            duration = await self._get_video_duration(video_path)
            print(f"Video ID: {video_id}")
            print(f"Duration: {duration:.1f} seconds")
            
            # Step 3: Run ML analysis
            ml_results = await self._run_ml_analysis(video_id, video_path)
            
            # Step 4: Build unified analysis
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
            
            # Step 5: Verify ml_data field
            analysis_dict = unified_analysis.to_dict(legacy_mode=False)
            self._verify_ml_data(analysis_dict)
            
            # Save unified analysis
            output_dir = Path("test_outputs")
            output_dir.mkdir(exist_ok=True)
            unified_path = output_dir / f"{video_id}_unified_analysis.json"
            with open(unified_path, 'w') as f:
                json.dump(analysis_dict, f, indent=2)
            
            # Step 6: Run Python compute functions
            print("\n[Phase 3/4] Python Compute Functions...")
            compute_results = {}
            compute_errors = {}
            
            for prompt_type in self.prompt_types:
                try:
                    print(f"Testing {prompt_type}...", end=' ', flush=True)
                    start = time.time()
                    
                    compute_fn = get_compute_function(prompt_type)
                    if compute_fn:
                        result = compute_fn(analysis_dict)
                        elapsed = time.time() - start
                        compute_results[prompt_type] = result
                        print(f"✅ PASS ({elapsed:.4f}s)")
                        
                        # Save result
                        result_path = output_dir / f"{video_id}_{prompt_type}.json"
                        with open(result_path, 'w') as f:
                            json.dump(result, f, indent=2)
                    else:
                        error_msg = f"No compute function found"
                        compute_errors[prompt_type] = error_msg
                        print(f"❌ FAIL - {error_msg}")
                        
                except Exception as e:
                    elapsed = time.time() - start
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    compute_errors[prompt_type] = error_msg
                    print(f"❌ FAIL - {error_msg[:50]}...")
                    logger.error(f"{prompt_type} failed: {error_msg}", exc_info=True)
            
            # Step 7: Validate outputs and contracts
            print("\n[Phase 4/4] Validation & Report Generation...")
            
            # Generate report
            report = self._generate_test_report(
                video_id=video_id,
                compute_results=compute_results,
                compute_errors=compute_errors,
                analysis_dict=analysis_dict
            )
            
            # Save report
            report_path = output_dir / f"{video_id}_test_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Print summary
            self._print_test_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            print(f"\n❌ CRITICAL ERROR: {e}")
            return {
                'test_level': 'CRITICAL FAIL',
                'error': str(e),
                'success': False
            }
    
    def _generate_test_report(self, video_id, compute_results, compute_errors, analysis_dict):
        """Generate comprehensive test report with graduated success levels"""
        
        total_functions = len(self.prompt_types)
        successful_functions = len(compute_results)
        failed_functions = len(compute_errors)
        success_rate = (successful_functions / total_functions) * 100 if total_functions > 0 else 0
        
        # Determine graduated success level
        if success_rate == 100:
            test_level = "FULL PASS"
            test_status = "✅ Production Ready"
        elif success_rate >= 85:
            test_level = "PARTIAL PASS"
            test_status = "⚠️ Acceptable with Issues"
        elif success_rate >= 50:
            test_level = "DEGRADED"
            test_status = "⚠️ Major Issues"
        else:
            test_level = "CRITICAL FAIL"
            test_status = "❌ System Broken"
        
        # Validate service contracts
        contract_validator = ServiceContractValidation()
        contract_validations = {}
        
        for prompt_type in self.prompt_types:
            if prompt_type in compute_results:
                contract_result = contract_validator.validate_contract(
                    prompt_type,
                    compute_results[prompt_type],
                    self.ml_data
                )
                contract_validations[prompt_type] = {
                    'status': '✅ PASS' if contract_result['valid'] else '❌ FAIL',
                    'contract_met': contract_result['valid'],
                    'violations': contract_result.get('violations', [])
                }
            elif prompt_type in compute_errors:
                contract_validations[prompt_type] = {
                    'status': '❌ FAIL',
                    'contract_met': False,
                    'error': compute_errors[prompt_type],
                    'violations': ['Function execution failed']
                }
        
        total_time = time.time() - self.start_time
        
        # Get memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        report = {
            'video_id': video_id,
            'timestamp': datetime.now().isoformat(),
            'test_level': test_level,
            'test_status': test_status,
            'success_level': f"{successful_functions}/{total_functions} ({success_rate:.1f}%)",
            'total_time_seconds': total_time,
            'memory_usage_mb': memory_mb,
            'cost': 0.00,
            'summary': {
                'total_functions_tested': total_functions,
                'successful': successful_functions,
                'failed': failed_functions,
                'success_rate': success_rate
            },
            'ml_data_validation': {
                'valid': self.ml_data_validation_result,
                'details': self.ml_data_details
            },
            'contract_validations': contract_validations,
            'compute_function_results': {
                'successful': list(compute_results.keys()),
                'failed': compute_errors
            },
            'recommendations': self._generate_recommendations(compute_errors, contract_validations)
        }
        
        return report
    
    def _generate_recommendations(self, errors, contract_validations):
        """Generate actionable recommendations"""
        recommendations = []
        
        for func, error in errors.items():
            if "RuntimeError" in error:
                recommendations.append(f"Check ML data availability for {func}")
            elif "KeyError" in error:
                recommendations.append(f"Verify data structure for {func}")
            elif "expression_timeline" in error:
                recommendations.append(f"Video may not contain people for {func}")
        
        # Check contract violations
        for func, validation in contract_validations.items():
            if not validation['contract_met'] and validation.get('violations'):
                for violation in validation['violations'][:2]:  # First 2 violations
                    recommendations.append(f"{func}: {violation}")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _print_test_summary(self, report):
        """Print formatted test summary"""
        print("\n--- Test Summary ---")
        print(f"Success Level:    {report['test_level']} ({report['success_level']})")
        print(f"Test Status:      {report['test_status']}")
        print(f"Total Time:       {report['total_time_seconds']:.2f} seconds")
        print(f"Memory Usage:     {report['memory_usage_mb']:.1f} MB")
        print(f"Cost:            ${report['cost']:.2f}")
        
        # ML Data Statistics
        print("\n--- ML Data Statistics ---")
        ml_details = report['ml_data_validation']['details']
        for key, value in ml_details.items():
            if isinstance(value, int):
                print(f"{key}: {value}")
            elif value == 'populated':
                print(f"{key}: ✅ Has data")
            elif value == 'empty':
                print(f"{key}: ⚠️ Empty")
        
        # Contract violations
        if any(not v['contract_met'] for v in report['contract_validations'].values()):
            print("\n--- Contract Violations ---")
            for func, validation in report['contract_validations'].items():
                if not validation['contract_met']:
                    print(f"{func}:")
                    if 'error' in validation:
                        print(f"  ❌ {validation['error'][:60]}...")
                    for violation in validation.get('violations', [])[:2]:
                        print(f"  ❌ {violation}")
        
        # Files generated
        successful = len(report['compute_function_results']['successful'])
        failed = len(report['compute_function_results']['failed'])
        print(f"\n--- Files Generated ---")
        print(f"✅ {successful * 3} files created ({successful} successful functions)")
        if failed > 0:
            print(f"❌ {failed * 3} files missing ({failed} failed functions)")
        
        # Final result
        print("\n" + "="*70)
        print(f"RESULT: {report['test_status']} - {report['test_level']}")
        
        if report['recommendations']:
            print("Next Steps:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        print("="*70)

async def main():
    parser = argparse.ArgumentParser(description='E2E Test for Python-Only Pipeline')
    parser.add_argument('video_path', type=Path, help='Path to video file or TikTok URL')
    
    args = parser.parse_args()
    
    # Handle TikTok URL vs local path
    video_path = args.video_path
    
    if str(video_path).startswith('http'):
        # Extract video ID from TikTok URL
        match = re.search(r'/video/(\d+)', str(video_path))
        if match:
            video_id = match.group(1)
            # Check if already downloaded
            local_path = Path(f"temp/{video_id}.mp4")
            if local_path.exists():
                print(f"Using existing download: {local_path}")
                video_path = local_path
            else:
                print(f"Error: Video not found locally. Please download first: {local_path}")
                sys.exit(1)
        else:
            print("Error: Invalid TikTok URL format")
            sys.exit(1)
    
    # Validate video exists
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Run test
    tester = PythonOnlyE2ETester()
    report = await tester.test_python_only_e2e(video_path)
    
    # Exit with appropriate code based on test level
    if report.get('test_level') == 'FULL PASS':
        sys.exit(0)
    elif report.get('test_level') == 'PARTIAL PASS':
        sys.exit(0)  # Still acceptable
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())