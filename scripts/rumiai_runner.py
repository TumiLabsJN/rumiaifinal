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

from rumiai_v2.api import ClaudeClient, ApifyClient, MLServices
from rumiai_v2.processors import (
    VideoAnalyzer, TimelineBuilder, TemporalMarkerProcessor,
    MLDataExtractor, PromptBuilder, OutputAdapter,
    get_compute_function, COMPUTE_FUNCTIONS
)
from rumiai_v2.prompts import PromptManager
from rumiai_v2.core.models import PromptType, PromptBatch, VideoMetadata
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
        self.claude = ClaudeClient(self.settings.claude_api_key, self.settings.claude_model)
        self.ml_services = MLServices()
        
        # Initialize processors
        self.video_analyzer = VideoAnalyzer(self.ml_services)
        self.timeline_builder = TimelineBuilder()
        self.temporal_processor = TemporalMarkerProcessor()
        self.ml_extractor = MLDataExtractor()
        self.prompt_builder = PromptBuilder(self.settings._prompt_templates)
        self.prompt_manager = PromptManager()
        self.output_adapter = OutputAdapter()
        
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
            
            # Step 7: Run Claude prompts
            print("📊 running_claude_prompts... (70%)")
            # Use v2 if feature flag is enabled
            if self.settings.use_ml_precompute:
                prompt_results = await self._run_claude_prompts_v2(unified_analysis)
            else:
                prompt_results = await self._run_claude_prompts(unified_analysis)
            
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
            
            # Run Claude prompts
            print("🧠 Running Claude prompts...")
            # Use v2 if feature flag is enabled
            if self.settings.use_ml_precompute:
                prompt_results = await self._run_claude_prompts_v2(unified_analysis)
            else:
                prompt_results = await self._run_claude_prompts(unified_analysis)
            
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
    
    async def _run_claude_prompts(self, analysis) -> Dict[str, Any]:
        """Run all Claude prompts."""
        self.metrics.start_timer('claude_prompts')
        
        # Define prompts to run
        prompt_types = [
            PromptType.CREATIVE_DENSITY,
            PromptType.EMOTIONAL_JOURNEY,
            PromptType.SPEECH_ANALYSIS,
            PromptType.VISUAL_OVERLAY,
            PromptType.METADATA_ANALYSIS,
            PromptType.PERSON_FRAMING,
            PromptType.SCENE_PACING
        ]
        
        # Create prompt batch
        batch = PromptBatch(
            video_id=analysis.video_id,
            prompts=prompt_types
        )
        
        # Process each prompt
        for i, prompt_type in enumerate(prompt_types):
            # Progress output for Node.js
            progress = int((i / len(prompt_types)) * 100)
            print(f"\n[{'█' * (i+1)}{'░' * (len(prompt_types)-i-1)}] {i+1}/{len(prompt_types)} ({progress}%)")
            print(f"🎬 Running {prompt_type.value} for video {analysis.video_id}")
            
            try:
                # Extract relevant data
                context = self.ml_extractor.extract_for_prompt(analysis, prompt_type)
                
                # Build prompt
                prompt_text = self.prompt_builder.build_prompt(context)
                
                # Log prompt info
                print(f"📏 Payload size: {context.get_size_bytes() / 1024:.1f}KB")
                
                # Send to Claude
                self.metrics.start_timer(f'prompt_{prompt_type.value}')
                result = self.claude.send_prompt(
                    prompt_text,
                    {
                        'video_id': analysis.video_id,
                        'prompt_type': prompt_type.value
                    },
                    timeout=self.settings.prompt_timeouts.get(prompt_type.value, 60)
                )
                prompt_time = self.metrics.stop_timer(f'prompt_{prompt_type.value}')
                
                # Record metrics
                self.video_metrics.record_prompt_time(prompt_type.value, prompt_time)
                if result.success:
                    self.video_metrics.record_prompt_cost(prompt_type.value, result.estimated_cost)
                
                # Save result
                self._save_prompt_result(analysis.video_id, prompt_type.value, result)
                
                # Add to batch
                batch.add_result(result)
                
                if result.success:
                    print(f"✅ {prompt_type.value} completed successfully!")
                    print(f"⏱️  {prompt_type.value} completed in {prompt_time:.1f}s")
                else:
                    print(f"❌ {prompt_type.value} failed: {result.error}")
                
                # Delay between prompts
                if i < len(prompt_types) - 1 and self.settings.prompt_delay > 0:
                    print(f"⏳ Waiting {self.settings.prompt_delay}s before next prompt...")
                    await asyncio.sleep(self.settings.prompt_delay)
                    
            except Exception as e:
                logger.error(f"Prompt {prompt_type.value} failed with exception: {str(e)}")
                print(f"❌ {prompt_type.value} crashed: {str(e)}")
                
                # Create failed result
                from rumiai_v2.core.models import PromptResult
                result = PromptResult(
                    prompt_type=prompt_type,
                    success=False,
                    error=str(e)
                )
                batch.add_result(result)
        
        self.metrics.stop_timer('claude_prompts')
        
        # Summary
        print(f"\n📊 Prompt Summary: {batch.get_success_rate()*100:.0f}% success rate")
        print(f"💰 Total cost: ${batch.get_total_cost():.4f}")
        
        return batch.results
    
    async def _run_claude_prompts_v2(self, analysis) -> Dict[str, Any]:
        """Run Claude prompts with ML precompute (v2 mode)."""
        self.metrics.start_timer('claude_prompts')
        logger.info("Using ML precompute mode (v2)")
        
        # Define prompts to run
        prompt_configs = [
            ('creative_density', PromptType.CREATIVE_DENSITY),
            ('emotional_journey', PromptType.EMOTIONAL_JOURNEY),
            ('speech_analysis', PromptType.SPEECH_ANALYSIS),
            ('visual_overlay_analysis', PromptType.VISUAL_OVERLAY),
            ('metadata_analysis', PromptType.METADATA_ANALYSIS),
            ('person_framing', PromptType.PERSON_FRAMING),
            ('scene_pacing', PromptType.SCENE_PACING)
        ]
        
        # Create prompt batch
        batch = PromptBatch(
            video_id=analysis.video_id,
            prompts=[pt for _, pt in prompt_configs]
        )
        
        # Get video data from analysis
        video_duration = analysis.timeline.duration if hasattr(analysis, 'timeline') else 0
        
        # Process each prompt
        for i, (compute_name, prompt_type) in enumerate(prompt_configs):
            # Progress output for Node.js
            progress = int((i / len(prompt_configs)) * 100)
            print(f"\n[{'█' * (i+1)}{'░' * (len(prompt_configs)-i-1)}] {i+1}/{len(prompt_configs)} ({progress}%)")
            print(f"🎬 Running {prompt_type.value} for video {analysis.video_id}")
            
            try:
                # Check if we should use Python-only processing (fail-fast mode)
                if self.settings.use_python_only_processing:
                    # Python-only mode: NO fallbacks, precompute must work or fail
                    compute_func = get_compute_function(compute_name)
                    if not compute_func:
                        raise RuntimeError(f"Python-only mode requires precompute function for {compute_name}, but none found")
                    
                    # Run precompute function
                    print(f"🔧 Python-only: Running precompute for {compute_name}...")
                    precomputed_metrics = compute_func(analysis.to_dict())
                    
                    if not precomputed_metrics:
                        raise RuntimeError(f"Python-only mode: precompute function for {compute_name} returned empty/None result")
                    
                    # Skip Claude entirely - use Python-computed metrics
                    logger.info(f"Python-only mode: Using precomputed metrics for {prompt_type.value}")
                    print(f"⚡ Python-only mode: Bypassing Claude for {prompt_type.value}")
                    
                    # Import PromptResult if needed
                    from rumiai_v2.core.models import PromptResult
                    
                    # Create result object with precomputed data
                    result = PromptResult(
                        prompt_type=prompt_type,
                        success=True,
                        response=json.dumps(precomputed_metrics, indent=2),  # Formatted string for backward compatibility
                        parsed_response=precomputed_metrics,  # Actual dict for ML training
                        processing_time=0.001,  # Near-instant
                        tokens_used=0,          # No tokens!
                        estimated_cost=0.0      # Free!
                    )
                    prompt_time = 0.001
                else:
                    # Legacy mode: Try precompute first, fallback to legacy extraction
                    compute_func = get_compute_function(compute_name)
                    if not compute_func:
                        logger.warning(f"No compute function found for {compute_name}, falling back to legacy")
                        # Fall back to legacy extraction
                        context = self.ml_extractor.extract_for_prompt(analysis, prompt_type)
                        prompt_text = self.prompt_builder.build_prompt(context)
                        precomputed_metrics = None
                    else:
                        # Run precompute function
                        print(f"🔧 Running precompute for {compute_name}...")
                        precomputed_metrics = compute_func(analysis.to_dict())
                        
                        # Build context for prompt
                        context = {
                            'video_id': analysis.video_id,
                            'video_duration': video_duration,
                            'precomputed_metrics': precomputed_metrics,
                            'prompt_type': prompt_type.value
                        }
                        
                        # Format prompt using new manager
                        prompt_text = self.prompt_manager.format_prompt(compute_name, context)
                    
                    # Validate prompt size (only in legacy mode)
                    is_valid, size_kb = self.prompt_manager.validate_prompt_size(prompt_text)
                    print(f"📏 Payload size: {size_kb}KB")
                    
                    if not is_valid:
                        raise ValueError(f"Prompt too large: {size_kb}KB")
                    
                    # Calculate dynamic timeout based on size
                    base_timeout = self.settings.prompt_timeouts.get(prompt_type.value, 60)
                    size_factor = max(1, size_kb / 50)  # Scale timeout for larger prompts
                    dynamic_timeout = int(base_timeout * size_factor)
                    
                    # Use Sonnet if feature flag is enabled
                    model = "claude-3-5-sonnet-20241022" if self.settings.use_claude_sonnet else self.settings.claude_model
                    
                    # Send to Claude
                    self.metrics.start_timer(f'prompt_{prompt_type.value}')
                    result = self.claude.send_prompt(
                        prompt_text,
                        {
                            'video_id': analysis.video_id,
                            'prompt_type': prompt_type.value,
                            'model': model
                        },
                        timeout=dynamic_timeout
                    )
                    prompt_time = self.metrics.stop_timer(f'prompt_{prompt_type.value}')
                
                # Handle 6-block output format validation
                if result.success and self.settings.output_format_version == 'v2':
                    # Validate response
                    is_valid, parsed_data, validation_errors = ResponseValidator.validate_6block_response(
                        result.response, 
                        prompt_type.value
                    )
                    
                    if is_valid and parsed_data:
                        logger.info(f"Received valid 6-block response for {prompt_type.value}")
                        
                        # Store parsed data for later use
                        result.parsed_response = parsed_data
                        
                        # Convert to legacy format if output format is v1
                        if self.settings.output_format_version == 'v1':
                            legacy_response = self.output_adapter.convert_6block_to_legacy(parsed_data, prompt_type.value)
                            result.response = json.dumps(legacy_response, indent=2)
                    else:
                        # Try to extract structure from text if JSON parsing failed
                        extracted = ResponseValidator.extract_text_blocks(result.response)
                        if extracted:
                            logger.warning(f"Extracted 6-block structure from text for {prompt_type.value}")
                            result.parsed_response = extracted
                        else:
                            logger.error(f"Invalid 6-block response for {prompt_type.value}: {', '.join(validation_errors)}")
                            # Mark as failed if we can't parse the response
                            result.success = False
                            result.error = f"Invalid response format: {'; '.join(validation_errors)}"
                
                # Record metrics
                self.video_metrics.record_prompt_time(prompt_type.value, prompt_time)
                if result.success:
                    self.video_metrics.record_prompt_cost(prompt_type.value, result.estimated_cost)
                
                # Save result
                self._save_prompt_result(analysis.video_id, prompt_type.value, result)
                
                # Add to batch
                batch.add_result(result)
                
                if result.success:
                    print(f"✅ {prompt_type.value} completed successfully!")
                    print(f"⏱️  {prompt_type.value} completed in {prompt_time:.1f}s")
                else:
                    print(f"❌ {prompt_type.value} failed: {result.error}")
                
                # Check memory after each prompt
                if self._check_memory_threshold(threshold_gb=3.5):
                    print("⚠️ Memory threshold reached, cleaning up...")
                    gc.collect()
                    await asyncio.sleep(2)  # Give system time to free memory
                
                # Delay between prompts
                if i < len(prompt_configs) - 1 and self.settings.prompt_delay > 0:
                    print(f"⏳ Waiting {self.settings.prompt_delay}s before next prompt...")
                    await asyncio.sleep(self.settings.prompt_delay)
                    
            except Exception as e:
                logger.error(f"Prompt {prompt_type.value} failed with exception: {str(e)}")
                print(f"❌ {prompt_type.value} crashed: {str(e)}")
                
                # Create failed result
                from rumiai_v2.core.models import PromptResult
                result = PromptResult(
                    prompt_type=prompt_type,
                    success=False,
                    error=str(e)
                )
                batch.add_result(result)
        
        self.metrics.stop_timer('claude_prompts')
        
        # Summary
        print(f"\n📊 Prompt Summary: {batch.get_success_rate()*100:.0f}% success rate")
        print(f"💰 Total cost: ${batch.get_total_cost():.4f}")
        
        return batch.results
    
    def _save_prompt_result(self, video_id: str, prompt_name: str, result) -> None:
        """Save individual prompt result."""
        # Create directory structure
        prompt_dir = self.insights_handler.get_path(video_id, prompt_name)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        result_path = prompt_dir / f"{prompt_name}_result_{timestamp}.txt"
        complete_path = prompt_dir / f"{prompt_name}_complete_{timestamp}.json"
        ml_path = prompt_dir / f"{prompt_name}_ml_{timestamp}.json"
        
        if result.success:
            # Save response text (for backward compatibility)
            with open(result_path, 'w') as f:
                f.write(result.response)
            
            # Save ML-friendly structured data if available
            if result.parsed_response:
                self.insights_handler.save_json(ml_path, result.parsed_response, indent=2)
        
        # Save complete data (includes both response string and parsed_response)
        self.insights_handler.save_json(complete_path, result.to_dict())
    
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
        if legacy_mode:
            logger.info(f"Running in legacy mode for video ID: {video_id}")
            result = asyncio.run(runner.process_video_id(video_id))
        else:
            logger.info(f"Running in new mode for video URL: {video_url}")
            result = asyncio.run(runner.process_video_url(video_url))
        
        # Output result
        if args.output_format == 'json':
            print(json.dumps(result, indent=2))
        else:
            if result['success']:
                print(f"✅ Success! Video {result.get('video_id', 'unknown')} processed.")
                if 'report' in result:
                    print(f"Report: {json.dumps(result['report'], indent=2)}")
            else:
                print(f"❌ Failed! Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        
        # Exit with appropriate code
        if result['success']:
            sys.exit(0)
        else:
            error_type = result.get('error_type', '')
            if 'API' in error_type:
                sys.exit(3)
            elif 'ML' in error_type:
                sys.exit(4)
            else:
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"🔴 FATAL ERROR: {type(e).__name__}: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()