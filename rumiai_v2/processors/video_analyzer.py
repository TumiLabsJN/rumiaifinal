"""
Video analyzer orchestrator for RumiAI v2.

This module orchestrates ML analysis of videos.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
import logging
import json
import time
import threading
import psutil
import os

from ..core.models import MLAnalysisResult
from ..core.exceptions import MLAnalysisError

logger = logging.getLogger(__name__)


class ThreadMonitor:
    """
    Monitor thread creation and memory usage for a specific service.
    Thread counting is accurate in sequential mode, approximate in parallel mode.
    Memory delta is accurate in sequential mode, not shown in parallel mode.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.initial_threads = threading.active_count()
        self.peak_threads = self.initial_threads
        self.running = True
        self.thread_flexibility = self._get_thread_flexibility()

        # Memory tracking
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Enhanced monitoring with sampling
        self.thread_names = set()
        self.samples = []
        self.cpu_times_start = self.process.cpu_times()

        # Start thread sampling
        self.thread = threading.Thread(target=self._sample_threads)
        self.thread.daemon = True
        self.thread.start()

    def _get_thread_flexibility(self) -> str:
        """Determine thread control flexibility for this service."""
        flexibility_map = {
            'whisper': '✅ Direct',
            'yolo': '✅ cv2.set()',
            'emotion_detection': '⚠️ Env vars',
            'mediapipe': '❌ Fixed',
            'ocr': '✅ Direct',
            'audio_energy': '✅ N/A',
            'scene_detection': '✅ N/A',
            'deepface': '⚠️ Env vars',
            'deepface_gender': '⚠️ Env vars',  # Alternative name
        }
        return flexibility_map.get(self.service_name, '❓ Unknown')

    def _sample_threads(self):
        """Sample thread count and resources every 0.5 seconds."""
        while self.running:
            try:
                current_threads = threading.active_count()
                self.peak_threads = max(self.peak_threads, current_threads)

                # Track thread names for debugging
                for thread in threading.enumerate():
                    if thread.name not in self.thread_names:
                        self.thread_names.add(thread.name)

                # Collect resource samples
                self.samples.append({
                    'time': time.time(),
                    'threads': current_threads,
                    'cpu_percent': self.process.cpu_percent(),
                    'memory_mb': self.process.memory_info().rss / 1024**2
                })
            except Exception as e:
                logger.warning(f"Thread sampling error for {self.service_name}: {e}")
            time.sleep(0.5)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ALWAYS called, even on exceptions."""
        self.stop()
        return False  # Don't suppress exceptions

    def stop(self):
        """Stop monitoring and return thread and memory statistics."""
        if hasattr(self, '_cached_stats'):
            return self._cached_stats  # Already stopped, return cached

        self.running = False
        self.thread.join(timeout=1)

        # Calculate memory delta (safely handle errors)
        try:
            end_memory = self.process.memory_info().rss / 1024 / 1024
            memory_delta = end_memory - self.start_memory
        except Exception as e:
            logger.warning(f"Memory measurement failed for {self.service_name}: {e}")
            memory_delta = 0

        # Calculate threads created
        threads_created = self.peak_threads - self.initial_threads

        # Check if we're in sequential mode
        is_sequential = os.getenv('PARALLEL_MODE', 'false').lower() != 'true'

        # Calculate CPU time if available
        cpu_time = 0
        if hasattr(self, 'cpu_times_start'):
            try:
                cpu_times_end = self.process.cpu_times()
                cpu_time = (cpu_times_end.user - self.cpu_times_start.user +
                           cpu_times_end.system - self.cpu_times_start.system)
            except Exception:
                pass

        # Calculate peak memory from samples if available
        peak_memory_mb = end_memory if memory_delta else 0
        if self.samples:
            peak_memory_mb = max((s['memory_mb'] for s in self.samples if s.get('memory_mb') is not None), default=end_memory)

        self._cached_stats = {
            'threads_created': max(0, threads_created),
            'peak_threads': self.peak_threads,
            'memory_delta_mb': round(memory_delta, 1) if is_sequential else None,
            'thread_flexibility': self.thread_flexibility,
            'cpu_time': round(cpu_time, 2) if cpu_time else None,
            'peak_memory_mb': round(peak_memory_mb, 1) if self.samples else None,
            'unique_threads': len(self.thread_names) if hasattr(self, 'thread_names') else 0,
            'sample_count': len(self.samples) if hasattr(self, 'samples') else 0
        }
        return self._cached_stats


class SystemSnapshot:
    """Capture comprehensive system state between services."""

    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
            self.available = True
        except ImportError:
            logger.warning("psutil not installed - system snapshots disabled")
            self.available = False

    def capture(self) -> Dict[str, Any]:
        """Capture current system state."""
        if not self.available:
            return {}

        vm = self.psutil.virtual_memory()

        return {
            'timestamp': time.time(),
            'cpu': {
                'percent': self.psutil.cpu_percent(interval=0.1),
                'load_avg': os.getloadavg(),
            },
            'memory': {
                'available_mb': vm.available / 1024**2,
                'used_mb': vm.used / 1024**2,
                'percent': vm.percent,
            },
            'process': {
                'threads': self.process.num_threads(),
                'connections': len(self.process.connections(kind='all')) if hasattr(self.process, 'connections') else 0,
            }
        }


class VideoAnalyzer:
    """Orchestrate ML analysis of videos."""
    
    def __init__(self, ml_services):
        """
        Initialize with ML services.

        Args:
            ml_services: ML service client for running analyses
        """
        self.ml_services = ml_services
        # Decision Point 11: Add DeepFace service attribute
        self.deepface_service = None  # Lazy load DeepFace service
    
    async def analyze_video(self, video_id: str, video_path: Path) -> Dict[str, MLAnalysisResult]:
        """
        Run all ML analyses on video.

        Returns dictionary of model_name -> MLAnalysisResult
        """
        logger.info(f"Starting ML analysis for video {video_id}")

        # Record pipeline start time for reporting
        pipeline_start = time.time()

        # Check execution mode
        sequential_mode = os.getenv('PARALLEL_MODE', 'false').lower() != 'true'  # Default to sequential
        if sequential_mode:
            logger.info("Running services sequentially (default mode)")

        # Define all ML analyses to run
        analyses = {
            'yolo': self._run_yolo,
            'whisper': self._run_whisper,
            'mediapipe': self._run_mediapipe,
            'ocr': self._run_ocr,
            'scene_detection': self._run_scene_detection,
            'audio_energy': self._run_audio_energy,
            'emotion_detection': self._run_emotion_detection,  # NEW - FEAT integration
            'deepface_gender': self._run_deepface_gender  # NEW - DeepFace gender detection
        }

        results = {}

        if sequential_mode:
            # SEQUENTIAL MODE: Run services one-by-one for accurate measurement
            # Initialize system snapshot for optimization detection
            system_snapshot = SystemSnapshot()
            optimization_warnings = []

            for model_name, analysis_func in analyses.items():
                logger.info(f"Starting {model_name} analysis (sequential)")

                # Capture system state before (if available)
                system_before = system_snapshot.capture() if system_snapshot.available else {}

                try:
                    result = await analysis_func(video_id, video_path)
                    results[model_name] = result

                    # Capture system state after
                    system_after = system_snapshot.capture() if system_snapshot.available else {}

                    # Real-time optimization detection
                    if system_after and system_before:
                        # Memory pressure warning
                        if system_after.get('memory', {}).get('percent', 0) > 80:
                            warning = f"⚠️ Memory pressure after {model_name}: {system_after['memory']['percent']:.1f}%"
                            logger.warning(warning)
                            optimization_warnings.append(warning)

                        # CPU saturation warning
                        if system_after.get('cpu', {}).get('percent', 0) > 90:
                            warning = f"⚠️ CPU saturated after {model_name}: {system_after['cpu']['percent']:.1f}%"
                            logger.warning(warning)
                            optimization_warnings.append(warning)

                    # Service-specific optimization detection
                    if result.threads_created > 10:
                        logger.warning(f"⚠️ {model_name} created {result.threads_created} threads")

                    if result.memory_delta_mb is not None and result.memory_delta_mb > 500:
                        logger.warning(f"⚠️ {model_name} used {result.memory_delta_mb:.0f}MB memory")

                    # Log detailed metrics in sequential mode
                    if hasattr(result, 'processing_time'):
                        logger.info(f"{model_name} completed: {result.processing_time:.1f}s, "
                                  f"threads: {result.threads_created}, "
                                  f"memory: {'+' + str(result.memory_delta_mb) + 'MB' if result.memory_delta_mb else 'N/A'}")
                    else:
                        logger.info(f"{model_name} analysis completed (success={result.success})")

                    # Garbage collection after heavy services
                    if model_name in ['emotion_detection', 'mediapipe'] and system_after:
                        if system_after.get('memory', {}).get('percent', 0) > 70:
                            import gc
                            gc.collect()
                            logger.info("🧹 Forced garbage collection after heavy service")

                except Exception as e:
                    logger.error(f"{model_name} analysis failed: {e}")
                    results[model_name] = MLAnalysisResult(
                        model_name=model_name,
                        model_version='unknown',
                        success=False,
                        error=str(e),
                        processing_time=0.0,
                        start_time=time.time(),
                        end_time=time.time(),
                        threads_created=0,
                        memory_delta_mb=0.0,
                        thread_flexibility='❓ Unknown'
                    )
        else:
            # PARALLEL MODE: Only when PARALLEL_MODE=true
            logger.info("Running services in parallel (PARALLEL_MODE=true)")
            tasks = {}
            for model_name, analysis_func in analyses.items():
                logger.info(f"Scheduling {model_name} analysis (parallel)")
                tasks[model_name] = asyncio.create_task(
                    analysis_func(video_id, video_path)
                )

            # Wait for all to complete
            for model_name, task in tasks.items():
                try:
                    result = await task
                    results[model_name] = result
                    logger.info(f"{model_name} analysis completed (success={result.success})")
                except Exception as e:
                    logger.error(f"{model_name} analysis failed with exception: {e}")
                    results[model_name] = MLAnalysisResult(
                        model_name=model_name,
                        model_version='unknown',
                        success=False,
                        error=str(e),
                        processing_time=0.0,
                        start_time=time.time(),
                        end_time=time.time(),
                        threads_created=0,
                        memory_delta_mb=0.0,
                        thread_flexibility='❓ Unknown'
                    )

        # Generate timing report
        pipeline_end = time.time()
        self._generate_timing_report(video_id, video_path, results, pipeline_start, pipeline_end)

        return results
    
    async def _run_yolo(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run YOLO object detection."""
        start_time = time.time()

        with ThreadMonitor('yolo') as monitor:
            try:
                # First check if output already exists
                output_dir = Path(f"object_detection_outputs/{video_id}")
                output_path = output_dir / f"{video_id}_yolo_detections.json"

                # Always run fresh analysis (cache removed for accuracy)
                logger.info(f"Running YOLO detection on {video_path}")
                data = await self.ml_services.run_yolo_detection(video_path, output_dir)

                # Save the output
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='yolo',
                    model_version='v8',
                    success=True,
                    data=data,
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'],
                    thread_flexibility=stats['thread_flexibility']
                )

            except Exception as e:
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='yolo',
                    model_version='v8',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )
    
    async def _run_whisper(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run Whisper speech transcription."""
        start_time = time.time()

        # Use context manager for guaranteed cleanup
        with ThreadMonitor('whisper') as monitor:
            try:
                # Always run fresh analysis (cache removed for accuracy)
                output_dir = Path("speech_transcriptions")
                output_path = output_dir / f"{video_id}_whisper.json"

                # Run actual Whisper transcription
                logger.info(f"Running Whisper transcription on {video_path}")
                data = await self.ml_services.run_whisper_transcription(video_path, output_dir)

                # Save the output
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                # Get stats before returning
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='whisper',
                    model_version='base',
                    success=True,
                    data=data,
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'],
                    thread_flexibility=stats['thread_flexibility']
                )

            except Exception as e:
                # Context manager ensures monitor.stop() is called via __exit__
                stats = monitor.stop()  # Get stats even on failure

                return MLAnalysisResult(
                    model_name='whisper',
                    model_version='base',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )
    
    async def _run_mediapipe(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run MediaPipe human analysis."""
        start_time = time.time()

        with ThreadMonitor('mediapipe') as monitor:
            try:
                # Always run fresh analysis (cache removed for accuracy)
                output_dir = Path(f"human_analysis_outputs/{video_id}")
                output_path = output_dir / f"{video_id}_human_analysis.json"

                # Run actual MediaPipe analysis
                logger.info(f"Running MediaPipe analysis on {video_path}")
                data = await self.ml_services.run_mediapipe_analysis(video_path, output_dir)

                # Save the output
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='mediapipe',
                    model_version='0.10',
                    success=True,
                    data=data,
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'],
                    thread_flexibility=stats['thread_flexibility']
                )

            except Exception as e:
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='mediapipe',
                    model_version='0.10',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )
    
    async def _run_ocr(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run OCR text detection."""
        start_time = time.time()

        with ThreadMonitor('ocr') as monitor:
            try:
                # Always run fresh analysis (cache removed for accuracy)
                output_dir = Path(f"creative_analysis_outputs/{video_id}")
                output_path = output_dir / f"{video_id}_creative_analysis.json"

                # Run actual OCR detection
                logger.info(f"Running OCR on {video_path}")
                data = await self.ml_services.run_ocr_analysis(video_path, output_dir)

                # Save the output
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='ocr',
                    model_version='tesseract-5',
                    success=True,
                    data=data,
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'],
                    thread_flexibility=stats['thread_flexibility']
                )

            except Exception as e:
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='ocr',
                    model_version='tesseract-5',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )
    
    async def _run_scene_detection(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run scene detection."""
        start_time = time.time()

        with ThreadMonitor('scene_detection') as monitor:
            try:
                # Check for existing scene detection
                output_dir = Path(f"scene_detection_outputs/{video_id}")
                output_path = output_dir / f"{video_id}_scenes.json"

                if output_path.exists():
                    logger.info(f"Using existing scene detection output: {output_path}")
                    with open(output_path, 'r') as f:
                        data = json.load(f)

                    stats = monitor.stop()

                    return MLAnalysisResult(
                        model_name='scene_detection',
                        model_version='pyscenedetect-0.6',
                        success=True,
                        data=data,
                        processing_time=time.time() - start_time,
                        start_time=start_time,
                        end_time=time.time(),
                        threads_created=stats['threads_created'],
                        memory_delta_mb=stats['memory_delta_mb'],
                        thread_flexibility=stats['thread_flexibility']
                    )

                # Run actual scene detection
                logger.info(f"Running scene detection on {video_path}")
                data = await self.ml_services.run_scene_detection(video_path, output_dir)

                # Save the output
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='scene_detection',
                    model_version='pyscenedetect-0.6',
                    success=True,
                    data=data,
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'],
                    thread_flexibility=stats['thread_flexibility']
                )

            except Exception as e:
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='scene_detection',
                    model_version='pyscenedetect-0.6',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )
    
    async def _run_audio_energy(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run audio energy analysis using librosa."""
        start_time = time.time()

        with ThreadMonitor('audio_energy') as monitor:
            try:
                # Check if output already exists
                output_dir = Path(f"audio_energy_outputs/{video_id}")
                output_path = output_dir / f"{video_id}_energy.json"

                if output_path.exists():
                    logger.info(f"Using existing audio energy output: {output_path}")
                    with open(output_path, 'r') as f:
                        data = json.load(f)

                    stats = monitor.stop()

                    return MLAnalysisResult(
                        model_name='audio_energy',
                        model_version='librosa-0.11',
                        success=True,
                        data=data,
                        processing_time=time.time() - start_time,
                        start_time=start_time,
                        end_time=time.time(),
                        threads_created=stats['threads_created'],
                        memory_delta_mb=stats['memory_delta_mb'],
                        thread_flexibility=stats['thread_flexibility']
                    )

                # Extract audio from video first
                audio_path = Path(f"temp/{video_id}_audio.wav")
                if not audio_path.exists():
                    logger.info(f"Extracting audio from {video_path}")
                    import subprocess
                    cmd = [
                        'ffmpeg', '-i', str(video_path),
                        '-vn', '-acodec', 'pcm_s16le',
                        '-ar', '16000', '-ac', '1',
                        str(audio_path), '-y'
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"Audio extraction failed: {result.stderr}")

                # Run audio energy analysis with video_id for shared extraction
                from rumiai_v2.ml_services.audio_energy_service import AudioEnergyService
                service = AudioEnergyService()
                data = await service.analyze(audio_path, video_id=video_id)

                # Save results
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='audio_energy',
                    model_version='librosa-0.11',
                    success=True,
                    data=data,
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'],
                    thread_flexibility=stats['thread_flexibility']
                )

            except Exception as e:
                logger.error(f"Audio energy analysis failed: {e}")
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='audio_energy',
                    model_version='librosa-0.11',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )
    
    async def _run_emotion_detection(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """Run FEAT emotion detection."""
        start_time = time.time()

        with ThreadMonitor('emotion_detection') as monitor:
            try:
                # Always run fresh analysis (cache removed for accuracy)
                output_dir = Path(f"emotion_detection_outputs/{video_id}")
                output_path = output_dir / f"{video_id}_emotions.json"

                # Run actual emotion detection
                logger.info(f"Running FEAT emotion detection on {video_path}")
                from ..ml_services.emotion_detection_service import get_emotion_detector

                # Get detector instance
                detector = get_emotion_detector()

                # Extract frames for emotion detection
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0

                # Adaptive sampling based on video duration
                sample_rate = detector.get_adaptive_sample_rate(duration)
                frame_interval = int(fps / sample_rate) if sample_rate > 0 else int(fps)

                frames = []
                timestamps = []
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        frames.append(frame)
                        timestamps.append(frame_count / fps)

                    frame_count += 1

                    # Limit to reasonable number of frames
                    if len(frames) >= 60:
                        break

                cap.release()

                # Run emotion detection
                data = await detector.detect_emotions_batch(frames, timestamps)

                # Add video metadata
                data['video_id'] = video_id
                data['video_duration'] = duration
                data['frames_analyzed'] = len(frames)
                data['sample_rate'] = sample_rate

                # Save output
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='emotion_detection',
                    model_version='feat-0.6.0',
                    success=True,
                    data=data,
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'],
                    thread_flexibility=stats['thread_flexibility']
                )

            except Exception as e:
                logger.error(f"Emotion detection failed: {e}")
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='emotion_detection',
                    model_version='feat-0.6.0',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )

    def _generate_timing_report(
        self,
        video_id: str,
        video_path: Path,
        results: Dict[str, MLAnalysisResult],
        pipeline_start: float,
        pipeline_end: float
    ) -> None:
        """Generate comprehensive timing report with mode-aware features."""
        import os
        from datetime import datetime

        # Determine execution mode
        sequential_mode = os.getenv('PARALLEL_MODE', 'false').lower() != 'true'

        # Calculate pipeline timing
        total_pipeline_time = pipeline_end - pipeline_start

        # Build base report
        report = {
            "video_id": video_id,
            "video_path": str(video_path),
            "timestamp": datetime.now().isoformat(),
            "execution_mode": "sequential" if sequential_mode else "parallel",
            "pipeline": {
                "start_time": pipeline_start,
                "end_time": pipeline_end,
                "total_time": total_pipeline_time,
            },
            "services": {}
        }

        # Add service-level metrics
        for service_name, result in results.items():
            service_metrics = {
                "success": result.success,
                "processing_time": result.processing_time,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "threads_created": result.threads_created,
                "thread_flexibility": result.thread_flexibility
            }

            # Only include memory_delta_mb in sequential mode (Decision 5)
            if os.getenv('PARALLEL_MODE', 'false').lower() != 'true':  # Sequential is default
                service_metrics["memory_delta_mb"] = result.memory_delta_mb

            if not result.success:
                service_metrics["error"] = result.error

            report["services"][service_name] = service_metrics

        # Add mode-specific analysis
        if sequential_mode:
            # Sequential mode - detailed analysis
            report["analysis"] = self._generate_sequential_analysis(results)
            report["optimization_suggestions"] = self._generate_optimization_suggestions(results)
        else:
            # Parallel mode - different metrics
            report["analysis"] = self._generate_parallel_analysis(results)

        # Thread analysis
        total_threads = sum(r.threads_created for r in results.values())
        report["thread_analysis"] = {
            "total_threads_created": total_threads,
            "threads_per_service": {
                service: result.threads_created
                for service, result in results.items()
            },
            "flexibility_summary": {
                service: result.thread_flexibility
                for service, result in results.items()
            }
        }

        # Save the report
        metrics_dir = Path("instrumentation_metrics")
        metrics_dir.mkdir(exist_ok=True)

        # Always save to JSON file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = metrics_dir / f"{video_id}_{timestamp_str}_metrics.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Metrics saved to {report_path}")

        # Display summary
        self._display_report_summary(report, sequential_mode)

    async def _run_deepface_gender(self, video_id: str, video_path: Path) -> MLAnalysisResult:
        """
        Run DeepFace gender detection.
        Decision Point 8: Runs in parallel with audio service.
        FIXED: Uses subprocess isolation to avoid memory corruption.
        """
        start_time = time.time()

        with ThreadMonitor('deepface_gender') as monitor:
            try:
                # Initialize service if needed
                if self.deepface_service is None:
                    # Use simple subprocess version to avoid memory corruption
                    from rumiai_v2.ml_services.deepface_gender_service_simple import (
                        DeepFaceGenderServiceSimple
                    )
                    self.deepface_service = DeepFaceGenderServiceSimple()

                # Run gender detection
                logger.info(f"Running DeepFace gender detection on {video_path}")
                result = await self.deepface_service.analyze(str(video_path))

                # Save results to output directory
                output_dir = Path(f"gender_detection_outputs/{video_id}")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{video_id}_gender.json"

                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f"Gender detection saved to {output_path}")

                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='deepface_gender',
                    model_version='deepface-0.0.92',
                    success=True,
                    data=result,
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'],
                    thread_flexibility=stats['thread_flexibility']
                )

            except (FileNotFoundError, VideoLoadError, ModelInitializationError) as e:
                # Decision Point 3: Critical errors should propagate
                logger.error(f"Critical error in gender detection for {video_path}: {e}")
                stats = monitor.stop()
                raise
            except Exception as e:
                # Decision Point 3: Expected failures return empty result
                logger.warning(f"Gender detection returned no result for {video_path}: {e}")
                stats = monitor.stop()

                return MLAnalysisResult(
                    model_name='deepface_gender',
                    model_version='deepface-0.0.92',
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                    start_time=start_time,
                    end_time=time.time(),
                    threads_created=stats['threads_created'],
                    memory_delta_mb=stats['memory_delta_mb'] if stats['memory_delta_mb'] is not None else 0.0,
                    thread_flexibility=stats['thread_flexibility']
                )
    def _generate_sequential_analysis(self, results: Dict[str, MLAnalysisResult]) -> Dict:
        """Generate analysis specific to sequential mode."""
        successful = [r for r in results.values() if r.success]

        # Find bottlenecks
        if successful:
            bottleneck = max(successful, key=lambda r: r.processing_time)
            total_time = sum(r.processing_time for r in successful if r.processing_time is not None)

            return {
                "bottleneck_service": bottleneck.model_name,
                "bottleneck_time": bottleneck.processing_time,
                "bottleneck_percentage": (bottleneck.processing_time / total_time * 100) if total_time > 0 else 0,
                "total_memory_used": sum(r.memory_delta_mb for r in successful if r.memory_delta_mb is not None),
                "total_threads_created": sum(r.threads_created for r in successful if r.threads_created is not None),
                "services_succeeded": len(successful),
                "services_failed": len(results) - len(successful)
            }
        return {}

    def _generate_optimization_suggestions(self, results: Dict[str, MLAnalysisResult]) -> List[Dict]:
        """Generate optimization suggestions based on metrics."""
        suggestions = []

        for service_name, result in results.items():
            if not result.success:
                continue

            # High thread usage
            if result.threads_created > 20:
                suggestions.append({
                    'service': service_name,
                    'issue': 'thread_explosion',
                    'threads_created': result.threads_created,
                    'suggestion': 'Consider limiting threads using environment variables'
                })

            # High memory usage
            if result.memory_delta_mb is not None and result.memory_delta_mb > 1000:
                suggestions.append({
                    'service': service_name,
                    'issue': 'high_memory',
                    'memory_mb': result.memory_delta_mb,
                    'suggestion': f'Used {result.memory_delta_mb:.1f}MB memory',
                })

            # Parallelization opportunity
            if result.thread_flexibility and '✅' in result.thread_flexibility and result.threads_created <= 2:
                suggestions.append({
                    'service': service_name,
                    'issue': 'underutilized',
                    'current_threads': result.threads_created,
                    'suggestion': 'Could benefit from more threads'
                })

        return suggestions

    def _generate_parallel_analysis(self, results: Dict[str, MLAnalysisResult]) -> Dict:
        """Generate analysis specific to parallel mode."""
        successful = [r for r in results.values() if r.success]

        if successful:
            slowest = max(successful, key=lambda r: r.end_time) if successful else None
            fastest_finish = min(r.end_time for r in successful) if successful else 0

            return {
                "mode": "parallel",
                "slowest_service": slowest.model_name if slowest else None,
                "services_succeeded": len(successful),
                "services_failed": len(results) - len(successful),
                "thread_info": "Thread counts approximate due to parallel execution",
                "memory_info": "Memory tracking disabled in parallel mode"
            }
        return {}

    def _display_report_summary(self, report: Dict, sequential_mode: bool) -> None:
        """Display a summary of the report to console."""
        logger.info("\n" + "="*60)
        logger.info("INSTRUMENTATION REPORT")
        logger.info("="*60)
        logger.info(f"Mode: {'Sequential' if sequential_mode else 'Parallel'}")
        logger.info(f"Pipeline Time: {report['pipeline']['total_time']:.2f}s")

        if sequential_mode and 'analysis' in report:
            analysis = report['analysis']
            if 'bottleneck_service' in analysis:
                logger.info(f"Bottleneck: {analysis['bottleneck_service']} ({analysis['bottleneck_time']:.2f}s)")
                logger.info(f"Memory Used: {analysis.get('total_memory_used', 0):.1f} MB")
                logger.info(f"Threads Created: {analysis.get('total_threads_created', 0)}")

        if 'optimization_suggestions' in report and report['optimization_suggestions']:
            logger.info("\nOptimization Suggestions:")
            for suggestion in report['optimization_suggestions'][:3]:  # Top 3
                logger.info(f"  - {suggestion['service']}: {suggestion['suggestion']}")

        logger.info("="*60)
