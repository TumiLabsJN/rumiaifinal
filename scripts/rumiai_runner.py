#!/usr/bin/env python3
"""
Main entry point for RumiAI v2.

CRITICAL: Must maintain backward compatibility with existing Node.js calls.
Supports both old and new calling conventions.
"""
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
import os
import time
import psutil
import gc

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rumiai_v2.api import ApifyClient, MLServices
from rumiai_v2.processors import (
    VideoAnalyzer, TimelineBuilder, TemporalMarkerProcessor,
    get_compute_function, COMPUTE_FUNCTIONS
)
from rumiai_v2.core.models import VideoMetadata
from rumiai_v2.config import Settings
from rumiai_v2.utils import FileHandler, Logger, Metrics, VideoProcessingMetrics
from rumiai_v2.validators import ResponseValidator

# Configure logging
logger = Logger.setup('rumiai_v2', level=os.getenv('LOG_LEVEL', 'INFO'))

# FAIL-FAST: Validate ML dependencies before anything else
try:
    from rumiai_v2.core.ml_dependency_validator import MLDependencyValidator
    MLDependencyValidator.validate_all()
    logger.info("✅ ML dependencies validated")
except Exception as e:
    logger.error(f"❌ ML dependency validation failed: {e}")
    if os.getenv('USE_PYTHON_ONLY_PROCESSING') == 'true':
        # In Python-only mode, ML dependencies are CRITICAL
        print(f"\n{'='*60}", file=sys.stderr)
        print("CRITICAL ERROR: ML Dependencies Missing", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("="*60, file=sys.stderr)
        sys.exit(1)


class RumiAIRunner:
    """
    Main orchestrator for RumiAI v2.
    
    CRITICAL: Maintains backward compatibility with old system.
    """
    
    def __init__(self):
        """
        Initialize runner.
        """
        self.settings = Settings()
        self.metrics = Metrics()
        self.video_metrics = VideoProcessingMetrics()
        
        # Initialize file handlers
        self.file_handler = FileHandler(self.settings.output_dir)
        self.unified_handler = FileHandler(self.settings.unified_dir)
        self.insights_handler = FileHandler(self.settings.insights_dir)
        # temporal_handler removed - using insights_handler for temporal markers now
        
        # Initialize clients
        self.apify = ApifyClient(self.settings.apify_token)
        self.ml_services = MLServices()
        
        # Initialize processors
        self.video_analyzer = VideoAnalyzer(self.ml_services)
        self.timeline_builder = TimelineBuilder()
        self.temporal_processor = TemporalMarkerProcessor()
        
        # Verify GPU availability at startup
        self._verify_gpu()
    
    def get_prefix_for_type(self, insight_type: str) -> str:
        """Get the prefix used in RESULT format for each insight type."""
        prefix_map = {
            'creative_density': 'density',
            'emotional_journey': 'emotional',
            'person_framing': 'personFraming',
            'scene_pacing': 'scenePacing', 
            'speech_analysis': 'speech',
            'visual_overlay_analysis': 'visualOverlay',
            'metadata_analysis': 'metadata',
            'temporal_markers': None  # No prefix needed
        }
        return prefix_map.get(insight_type, insight_type)
    
    def convert_to_ml_format(self, prefixed_data: dict, insight_type: str) -> dict:
        """Convert prefixed format to ML format by removing type prefix."""
        if insight_type == 'temporal_markers':
            return prefixed_data  # No conversion needed
        
        ml_data = {}
        prefix = self.get_prefix_for_type(insight_type)  # e.g., "density", "emotional"
        
        if not prefix:
            return prefixed_data
        
        for key, value in prefixed_data.items():
            if key.startswith(prefix):
                # Remove prefix and capitalize first letter
                new_key = key[len(prefix):]
                new_key = new_key[0].upper() + new_key[1:] if new_key else key
                ml_data[new_key] = value
            else:
                ml_data[key] = value
        
        return ml_data
    
    def save_analysis_result(self, video_id: str, analysis_type: str, data: dict) -> Path:
        """Save analysis result in 3-file backward compatible format."""
        from datetime import datetime
        import json
        
        # Generate single timestamp for all 3 files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directory structure
        analysis_dir = self.insights_handler.get_path(video_id, analysis_type)
        self.insights_handler.ensure_dir(analysis_dir)
        
        # Initialize paths to None for cleanup in case of error
        complete_path = None
        ml_path = None
        result_path = None
        
        try:
            # 1. Prepare the three formats
            result_data = data  # Already in prefixed format from precompute
            ml_data = self.convert_to_ml_format(data, analysis_type)
            
            # 2. Create _complete file content
            complete_data = {
                "prompt_type": analysis_type,
                "success": True,
                "response": json.dumps(result_data),
                "parsed_response": ml_data
            }
            
            # 3. Generate file paths with same timestamp
            complete_path = analysis_dir / f"{analysis_type}_complete_{timestamp}.json"
            ml_path = analysis_dir / f"{analysis_type}_ml_{timestamp}.json"
            result_path = analysis_dir / f"{analysis_type}_result_{timestamp}.json"
            
            # 4. Save all 3 files atomically (fail fast on any error)
            self.insights_handler.save_json(complete_path, complete_data)
            self.insights_handler.save_json(ml_path, ml_data)
            self.insights_handler.save_json(result_path, result_data)
            
            logger.info(f"Saved 3-file set for {analysis_type} to {analysis_dir}")
            return complete_path
            
        except Exception as e:
            # Fail fast - log and re-raise
            logger.error(f"Failed to save {analysis_type} for {video_id}: {e}")
            # Clean up any partial files if they exist
            for path in [complete_path, ml_path, result_path]:
                if path and path.exists():
                    path.unlink()
            raise
    
    def _verify_gpu(self) -> None:
        """Verify GPU/CUDA availability at startup."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"✅ GPU available: {device_name} with {memory:.1f}GB VRAM")
                print(f"🎮 GPU: {device_name} ({memory:.1f}GB VRAM)")
            else:
                logger.warning("⚠️ No GPU detected, using CPU (will be slower)")
                print("⚠️ WARNING: No GPU detected, processing will be slower")
        except ImportError:
            logger.warning("PyTorch not installed, cannot check GPU availability")
        except Exception as e:
            logger.warning(f"Could not verify GPU: {e}")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get system memory
        virtual_memory = psutil.virtual_memory()
        
        return {
            'process_rss_gb': memory_info.rss / 1024**3,
            'process_vms_gb': memory_info.vms / 1024**3,
            'system_percent': virtual_memory.percent,
            'system_available_gb': virtual_memory.available / 1024**3
        }
    
    def _check_memory_threshold(self, threshold_gb: float = 4.0) -> bool:
        """Check if we're approaching memory limits."""
        memory = self._get_memory_usage()
        
        if memory['process_rss_gb'] > threshold_gb:
            logger.warning(f"High memory usage: {memory['process_rss_gb']:.1f}GB")
            # Force garbage collection
            gc.collect()
            return True
        
        if memory['system_percent'] > 90:
            logger.warning(f"System memory critically low: {memory['system_percent']:.1f}%")
            return True
        
        return False
    
    async def process_video_url(self, video_url: str) -> Dict[str, Any]:
        """
        Process a video from URL (new mode).
        
        This is the main entry point for new system.
        """
        logger.info(f"🚀 Starting processing for: {video_url}")
        self.metrics.start_timer('total_processing')
        
        try:
            # Step 1: Scrape video metadata
            print("📊 scraping_metadata... (0%)")
            video_metadata = await self._scrape_video(video_url)
            video_id = video_metadata.video_id
            print(f"✅ Video ID: {video_id}")
            
            # Step 2: Download video
            print("📊 downloading_video... (10%)")
            video_path = await self._download_video(video_metadata)
            print(f"✅ Downloaded to: {video_path}")
            
            # Step 3: Run ML analysis
            print("📊 running_ml_analysis... (20%)")
            # Check memory before ML analysis
            initial_memory = self._get_memory_usage()
            logger.info(f"Memory before ML: {initial_memory['process_rss_gb']:.1f}GB")
            
            ml_results = await self._run_ml_analysis(video_id, video_path)
            
            # Check memory after ML analysis
            post_ml_memory = self._get_memory_usage()
            logger.info(f"Memory after ML: {post_ml_memory['process_rss_gb']:.1f}GB")
            
            if self._check_memory_threshold():
                print("⚠️ High memory usage detected, forcing garbage collection...")
            
            # Step 4: Build unified timeline
            print("📊 building_timeline... (50%)")
            unified_analysis = self.timeline_builder.build_timeline(
                video_id, 
                video_metadata.to_dict(), 
                ml_results
            )
            
            # Step 5: Generate temporal markers
            print("📊 generating_temporal_markers... (60%)")
            temporal_markers = self.temporal_processor.generate_markers(unified_analysis)
            unified_analysis.temporal_markers = temporal_markers
            
            # Save temporal markers separately for compatibility
            temporal_path = self.save_analysis_result(video_id, "temporal_markers", temporal_markers)
            
            # Step 6: Save unified analysis
            print("📊 saving_analysis... (65%)")
            unified_path = self.unified_handler.get_path(f"{video_id}.json")
            unified_analysis.save_to_file(str(unified_path))
            
            print("📊 running_precompute_functions... (70%)")
            prompt_results = {}
            for func_name, func in COMPUTE_FUNCTIONS.items():
                try:
                    result = func(unified_analysis.to_dict())
                    prompt_results[func_name] = result
                    # Save each insight to the insights folder
                    if result:  # Only save if result is not empty
                        self.save_analysis_result(video_id, func_name, result)
                except Exception as e:
                    logger.error(f"Precompute {func_name} failed: {e}")
                    prompt_results[func_name] = {}
            
            # Step 8: Generate final report
            print("📊 generating_report... (95%)")
            report = self._generate_report(unified_analysis, prompt_results)
            
            self.metrics.stop_timer('total_processing')
            self.video_metrics.record_video(success=True)
            
            print("📊 completed... (100%)")
            logger.info(f"✅ Processing complete! Total time: {self.metrics.get_time('total_processing'):.1f}s")
            
            # Return result in format expected by Node.js
            return {
                'success': True,
                'video_id': video_id,
                'outputs': {
                    'video': str(video_path),
                    'unified': str(unified_path),
                    'temporal': str(temporal_path),
                    'insights': str(self.insights_handler.base_dir / video_id)
                },
                'report': report,
                'metrics': self.metrics.get_all()
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            self.video_metrics.record_video(success=False)
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'metrics': self.metrics.get_all()
            }
    
    async def _scrape_video(self, video_url: str) -> VideoMetadata:
        """Scrape video metadata from TikTok."""
        self.metrics.start_timer('scraping')
        try:
            metadata = await self.apify.scrape_video(video_url)
            self.metrics.stop_timer('scraping')
            return metadata
        except Exception as e:
            self.metrics.stop_timer('scraping')
            raise
    
    async def _download_video(self, video_metadata: VideoMetadata) -> Path:
        """Download video file."""
        self.metrics.start_timer('download')
        try:
            video_path = await self.apify.download_video(
                video_metadata.download_url,
                video_metadata.video_id,
                self.settings.temp_dir
            )
            self.metrics.stop_timer('download')
            return video_path
        except Exception as e:
            self.metrics.stop_timer('download')
            raise
    
    async def _run_ml_analysis(self, video_id: str, video_path: Path) -> Dict[str, Any]:
        """Run all ML analyses on video."""
        self.metrics.start_timer('ml_analysis')
        try:
            results = await self.video_analyzer.analyze_video(video_id, video_path)
            
            # Log ML timing
            for model_name, result in results.items():
                if result.processing_time > 0:
                    self.video_metrics.record_ml_time(model_name, result.processing_time)
            
            self.metrics.stop_timer('ml_analysis')
            return results
        except Exception as e:
            self.metrics.stop_timer('ml_analysis')
            raise
    
    def _generate_report(self, analysis, prompt_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final analysis report."""
        # Handle both PromptResult objects and dictionary results from precompute functions
        successful_prompts = 0
        total_cost = 0.0
        total_tokens = 0
        
        for r in prompt_results.values():
            if hasattr(r, 'success'):  # PromptResult object
                if r.success:
                    successful_prompts += 1
                    if hasattr(r, 'estimated_cost') and r.estimated_cost:
                        total_cost += r.estimated_cost
                    if hasattr(r, 'tokens_used') and r.tokens_used:
                        total_tokens += r.tokens_used
            elif isinstance(r, dict) and r:  # Non-empty dictionary from precompute
                successful_prompts += 1
                # Precompute functions are free and don't use tokens
                total_cost += 0.0
                total_tokens += 0
        
        # Get final memory usage
        final_memory = self._get_memory_usage()
        
        return {
            'video_id': analysis.video_id,
            'duration': analysis.timeline.duration,
            'ml_analyses_complete': analysis.is_complete(),
            'ml_completion_details': analysis.get_completion_status(),
            'temporal_markers_generated': analysis.temporal_markers is not None,
            'prompts_successful': successful_prompts,
            'prompts_total': len(prompt_results),
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'prompt_details': {
                name: {
                    'success': result.success if hasattr(result, 'success') else bool(result),
                    'tokens': result.tokens_used if hasattr(result, 'tokens_used') else 0,
                    'cost': result.estimated_cost if hasattr(result, 'estimated_cost') else 0.0,
                    'time': result.processing_time if hasattr(result, 'processing_time') else 0.001
                }
                for name, result in prompt_results.items()
            },
            'processing_metrics': self.metrics.get_all(),
            'video_metrics': self.video_metrics.get_summary(),
            'memory_usage': {
                'final_process_gb': final_memory['process_rss_gb'],
                'peak_process_gb': max(final_memory['process_rss_gb'], 4.0),  # Estimate peak
                'system_percent': final_memory['system_percent']
            },
            'feature_flags': {
                'ml_precompute': True,  # Always True in Python-only mode
                'python_only': True,   # Python-only mode enabled
                'output_format': getattr(self.settings, 'output_format_version', '2.0')
            }
        }


def main():
    """
    Main entry point.
    
    CRITICAL: Exit codes must match Node.js expectations:
    - 0: Success
    - 1: General failure  
    - 2: Invalid arguments
    - 3: API failure
    - 4: ML processing failure
    """
    parser = argparse.ArgumentParser(description='RumiAI v2 Video Processor')
    
    # Support multiple calling conventions
    parser.add_argument('video_input', nargs='?', help='Video URL (must start with http:// or https://)')
    parser.add_argument('--video-url', help='Video URL to process')
    parser.add_argument('--config-dir', help='Configuration directory')
    parser.add_argument('--output-format', choices=['json', 'text'], default='json')
    
    args = parser.parse_args()
    
    # Determine input
    video_url = None
    
    if args.video_url:
        video_url = args.video_url
    elif args.video_input:
        # Only accept URLs
        if args.video_input.startswith('http'):
            video_url = args.video_input
        else:
            logger.error(f"Error: '{args.video_input}' is not a valid URL")
            logger.error("Please provide a complete TikTok URL starting with http:// or https://")
            sys.exit(1)
    else:
        print("Usage: rumiai_runner.py <video_url>", file=sys.stderr)
        sys.exit(2)
    
    try:
        # Create runner
        runner = RumiAIRunner()
        
        # Run processing
        logger.info(f"Processing video URL: {video_url}")
        result = asyncio.run(runner.process_video_url(video_url))
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()