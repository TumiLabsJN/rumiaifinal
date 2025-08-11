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
    
    def __init__(self, legacy_mode: bool = False):
        """
        Initialize runner.
        
        Args:
            legacy_mode: If True, operate in backward compatibility mode
        """
        self.legacy_mode = legacy_mode
        self.settings = Settings()
        self.metrics = Metrics()
        self.video_metrics = VideoProcessingMetrics()
        
        # Initialize file handlers
        self.file_handler = FileHandler(self.settings.output_dir)
        self.unified_handler = FileHandler(self.settings.unified_dir)
        self.insights_handler = FileHandler(self.settings.insights_dir)
        self.temporal_handler = FileHandler(self.settings.temporal_dir)
        
        # Initialize clients
        self.apify = ApifyClient(self.settings.apify_token)
        self.ml_services = MLServices()
        
        # Initialize processors
        self.video_analyzer = VideoAnalyzer(self.ml_services)
        self.timeline_builder = TimelineBuilder()
        self.temporal_processor = TemporalMarkerProcessor()
        
        # Verify GPU availability at startup
        self._verify_gpu()
    
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
            temporal_path = self.temporal_handler.get_path(
                f"{video_id}_{int(time.time())}.json"
            )
            self.temporal_handler.save_json(temporal_path, temporal_markers)
            
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
    
    async def process_video_id(self, video_id: str) -> Dict[str, Any]:
        """
        Process a video by ID (legacy mode).
        
        This maintains compatibility with old Python script calls.
        """
        logger.info(f"🔄 Processing video ID in legacy mode: {video_id}")
        
        try:
            # Load existing analysis data
            unified_path = self.unified_handler.get_path(f"{video_id}.json")
            if not unified_path.exists():
                # Try old path structure
                old_path = Path(f"unified_analysis/{video_id}.json")
                if old_path.exists():
                    unified_path = old_path
                else:
                    raise FileNotFoundError(f"No unified analysis found for {video_id}")
            
            # Load unified analysis
            from ..core.models import UnifiedAnalysis
            unified_analysis = UnifiedAnalysis.load_from_file(str(unified_path))
            
            # Generate temporal markers if missing
            if not unified_analysis.temporal_markers:
                print("🔄 Generating temporal markers...")
                temporal_markers = self.temporal_processor.generate_markers(unified_analysis)
                unified_analysis.temporal_markers = temporal_markers
                
                # Save temporal markers
                temporal_path = self.temporal_handler.get_path(
                    f"{video_id}_{int(time.time())}.json"
                )
                self.temporal_handler.save_json(temporal_path, temporal_markers)
            
            print("🧠 Running precompute functions...")
            prompt_results = {}
            for func_name, func in COMPUTE_FUNCTIONS.items():
                try:
                    result = func(unified_analysis.to_dict())
                    prompt_results[func_name] = result
                except Exception as e:
                    logger.error(f"Precompute {func_name} failed: {e}")
                    prompt_results[func_name] = {}
            
            # Generate report
            report = self._generate_report(unified_analysis, prompt_results)
            
            return {
                'success': True,
                'video_id': video_id,
                'prompts_completed': len([r for r in prompt_results.values() if r.success]),
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Legacy processing failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
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
        successful_prompts = sum(1 for r in prompt_results.values() if r.success)
        total_cost = sum(r.estimated_cost for r in prompt_results.values() if r.success and r.estimated_cost)
        total_tokens = sum(r.tokens_used for r in prompt_results.values() if r.success and r.tokens_used)
        
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
                    'success': result.success,
                    'tokens': result.tokens_used,
                    'cost': result.estimated_cost,
                    'time': result.processing_time
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
                'ml_precompute': self.settings.use_ml_precompute,
                'claude_sonnet': self.settings.use_claude_sonnet,
                'output_format': self.settings.output_format_version
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
    parser.add_argument('video_input', nargs='?', help='Video URL or ID')
    parser.add_argument('--video-url', help='Video URL to process')
    parser.add_argument('--video-id', help='Video ID to process (legacy mode)')
    parser.add_argument('--config-dir', help='Configuration directory')
    parser.add_argument('--output-format', choices=['json', 'text'], default='json')
    
    args = parser.parse_args()
    
    # Determine mode and input
    video_url = None
    video_id = None
    legacy_mode = False
    
    if args.video_url:
        video_url = args.video_url
    elif args.video_id:
        video_id = args.video_id
        legacy_mode = True
    elif args.video_input:
        # Auto-detect URL vs ID
        if args.video_input.startswith('http'):
            video_url = args.video_input
        else:
            video_id = args.video_input
            legacy_mode = True
    else:
        print("Usage: rumiai_runner.py <video_url_or_id>", file=sys.stderr)
        sys.exit(2)
    
    try:
        # Create runner
        runner = RumiAIRunner(legacy_mode=legacy_mode)
        
        # Run processing
        
            logger.info(f"Running in legacy mode for video ID: {video_id}")
            result = asyncio.run(runner.process_video_id(video_id))
        
if __name__ == "__main__":
    main()