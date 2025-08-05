"""
Unified Frame Manager - Extract once, use everywhere
Handles frame extraction, caching, and distribution to ML services
With LRU cache, size limits, and error recovery
"""

import os
import cv2
import asyncio
import numpy as np
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class FrameData:
    """Container for extracted frame data"""
    frame_number: int
    timestamp: float
    image: np.ndarray
    
@dataclass
class VideoMetadata:
    """Video properties"""
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int

class FrameSamplingConfig:
    """
    Frame sampling configuration based on:
    - FRAME_PROCESSING_DOCUMENTATION.md requirements
    - Model computational requirements
    - Accuracy vs performance tradeoffs
    """
    
    CONFIGS = {
        'yolo': {
            'max_frames': 100,  # ~2 FPS for 60s video (matches doc line 206)
            'strategy': 'uniform',
            'rationale': 'Object detection needs consistent temporal coverage'
        },
        'mediapipe': {
            'max_frames': 180,  # All frames per doc line 207
            'strategy': 'all',
            'rationale': 'Human motion requires high temporal resolution'
        },
        'ocr': {
            'max_frames': 60,  # ~1 FPS for 60s video (matches doc line 208)
            'strategy': 'adaptive',
            'rationale': 'Text appears more at beginning/end (titles/CTAs)'
        },
        'scene': {
            'max_frames': None,  # All frames needed for accurate detection
            'strategy': 'all',
            'rationale': 'Scene boundaries require full temporal analysis'
        },
    }
    
    @classmethod
    def get_config(cls, service_name: str) -> Dict:
        """Get config with documentation"""
        return cls.CONFIGS.get(service_name, {
            'max_frames': 50,
            'strategy': 'uniform',
            'rationale': 'Default conservative sampling'
        })
    
class UnifiedFrameManager:
    """
    Centralized frame extraction and distribution with LRU cache
    Extracts frames once, provides to all ML services
    """
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 max_memory_frames: int = 500,
                 max_cache_size_gb: float = 2.0,
                 max_cache_videos: int = 5):
        """
        Initialize frame manager with cache limits
        
        Args:
            cache_dir: Directory for frame cache (None = temp dir)
            max_memory_frames: Max frames to keep in memory per video
            max_cache_size_gb: Maximum cache size in GB (reduced to 2GB)
            max_cache_videos: Maximum number of videos to cache (5 for debugging)
        """
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix="rumiai_frames_"))
        self.max_memory_frames = max_memory_frames
        self.max_cache_size_gb = max_cache_size_gb
        self.max_cache_videos = max_cache_videos
        
        # LRU cache implementation
        self._frame_cache = OrderedDict()  # Maintains insertion order for LRU
        self._metadata_cache = {}  # video_id -> metadata
        self._cache_sizes = {}  # Track size of each cached video
        self._total_cache_size = 0
        
    async def extract_frames(self, 
                           video_path: Path, 
                           video_id: str,
                           target_fps: Optional[float] = None,
                           retry_count: int = 3) -> Dict[str, Any]:
        """
        Extract frames from video with adaptive sampling and retry logic
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for caching
            target_fps: Target frame rate (None = adaptive)
            retry_count: Number of retry attempts on failure
            
        Returns:
            Dictionary with frames, metadata, and status
        """
        # Check cache first with LRU update
        if video_id in self._frame_cache:
            # Move to end (most recently used)
            self._frame_cache.move_to_end(video_id)
            logger.info(f"Using cached frames for {video_id}")
            return {
                'frames': self._frame_cache[video_id]['frames'],
                'metadata': self._metadata_cache[video_id],
                'cache_hit': True,
                'success': True
            }
            
        logger.info(f"Extracting frames from {video_path}")
        
        # Try extraction with retries
        last_error = None
        
        for attempt in range(retry_count):
            try:
                # Try extraction with timeout (10 minutes)
                async with asyncio.timeout(600):
                    # Get video metadata
                    metadata = await self._get_video_metadata(video_path)
                    
                    # Determine extraction FPS
                    if target_fps is None:
                        target_fps = self._calculate_adaptive_fps(metadata.duration)
                    
                    # Try extraction
                    frames = await self._extract_frames_async(
                        video_path, metadata, target_fps
                    )
                    
                    if frames and len(frames) > 0:
                        # Add to cache with eviction
                        await self._add_to_cache_with_eviction(video_id, frames, metadata)
                        
                        return {
                            'frames': frames,
                            'metadata': metadata,
                            'cache_hit': False,
                            'success': True
                        }
                        
            except asyncio.TimeoutError:
                last_error = f"Timeout on attempt {attempt + 1}"
                logger.warning(f"Frame extraction timeout, attempt {attempt + 1}/{retry_count}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Frame extraction failed: {e}, attempt {attempt + 1}/{retry_count}")
                
            # Exponential backoff between retries
            if attempt < retry_count - 1:
                await asyncio.sleep(2 ** attempt)
        
        # All retries failed - try minimal extraction as fallback
        logger.error(f"All extraction attempts failed, trying minimal extraction")
        
        try:
            frames = await self._minimal_frame_extraction(video_path, num_frames=10)
            if frames:
                basic_metadata = await self._get_basic_metadata(video_path)
                return {
                    'frames': frames,
                    'metadata': basic_metadata,
                    'degraded': True,
                    'error': last_error,
                    'success': True
                }
        except Exception as e:
            logger.error(f"Even minimal extraction failed: {e}")
        
        # Complete failure
        return {
            'frames': [],
            'metadata': None,
            'error': f"Frame extraction failed after {retry_count} attempts: {last_error}",
            'success': False
        }
        
    async def _get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract video metadata"""
        def extract_metadata():
            cap = cv2.VideoCapture(str(video_path))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0
                
                return VideoMetadata(
                    fps=fps,
                    total_frames=total_frames,
                    duration=duration,
                    width=width,
                    height=height
                )
            finally:
                cap.release()
                
        return await asyncio.to_thread(extract_metadata)
        
    async def _get_basic_metadata(self, video_path: Path) -> VideoMetadata:
        """Get basic metadata for fallback"""
        try:
            return await self._get_video_metadata(video_path)
        except:
            # Return minimal metadata
            return VideoMetadata(
                fps=30.0,  # Assume standard
                total_frames=0,
                duration=0,
                width=1920,
                height=1080
            )
        
    def _calculate_adaptive_fps(self, duration: float) -> float:
        """Calculate optimal FPS based on video duration (from FRAME_PROCESSING_DOCUMENTATION.md)"""
        if duration < 30:
            return 5.0  # 5 FPS for short videos
        elif duration < 60:
            return 3.0  # 3 FPS for medium videos
        elif duration < 120:
            return 2.0  # 2 FPS for long videos
        else:
            return 1.0  # 1 FPS for very long videos
            
    async def _extract_frames_async(self, 
                                   video_path: Path, 
                                   metadata: VideoMetadata,
                                   target_fps: float) -> List[FrameData]:
        """Extract frames asynchronously"""
        
        def extract_opencv():
            """OpenCV extraction method"""
            frames = []
            cap = cv2.VideoCapture(str(video_path))
            
            try:
                # Calculate frame interval
                source_fps = metadata.fps
                interval = int(source_fps / target_fps) if source_fps > target_fps else 1
                
                frame_count = 0
                extracted_count = 0
                max_frames = int(metadata.duration * target_fps)  # Limit total frames
                
                while cap.isOpened() and extracted_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Extract at interval
                    if frame_count % interval == 0:
                        timestamp = frame_count / source_fps
                        frames.append(FrameData(
                            frame_number=frame_count,
                            timestamp=timestamp,
                            image=frame
                        ))
                        extracted_count += 1
                        
                    frame_count += 1
                    
                logger.info(f"Extracted {len(frames)} frames at {target_fps} FPS using OpenCV")
                return frames
                
            finally:
                cap.release()
        
        return await asyncio.to_thread(extract_opencv)
            
    async def _minimal_frame_extraction(self, video_path: Path, num_frames: int = 10) -> List[FrameData]:
        """Minimal extraction as last resort - extract just a few key frames"""
        def extract_minimal():
            frames = []
            cap = cv2.VideoCapture(str(video_path))
            
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                if total_frames <= 0:
                    return frames
                
                # Extract frames at uniform intervals
                interval = max(1, total_frames // num_frames)
                
                for i in range(0, total_frames, interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(FrameData(
                            frame_number=i,
                            timestamp=i / fps if fps > 0 else 0,
                            image=frame
                        ))
                    
                    if len(frames) >= num_frames:
                        break
                        
                logger.info(f"Minimal extraction: got {len(frames)} frames")
                return frames
                
            finally:
                cap.release()
                
        return await asyncio.to_thread(extract_minimal)
        
    async def _add_to_cache_with_eviction(self, video_id: str, frames: List[FrameData], metadata: VideoMetadata):
        """Add to cache with LRU eviction based on size limits"""
        # Calculate size
        if frames:
            sample_size = min(10, len(frames))
            frame_size = sum(f.image.nbytes for f in frames[:sample_size]) / sample_size
            estimated_size_mb = (frame_size * len(frames)) / (1024 * 1024)
        else:
            estimated_size_mb = 0
        
        # Evict if needed (LRU)
        while (len(self._frame_cache) >= self.max_cache_videos or 
               self._total_cache_size + estimated_size_mb > self.max_cache_size_gb * 1024):
            
            if not self._frame_cache:
                break
                
            # Remove least recently used
            oldest_id, oldest_data = self._frame_cache.popitem(last=False)
            self._total_cache_size -= self._cache_sizes.get(oldest_id, 0)
            del self._cache_sizes[oldest_id]
            
            # Clean disk cache if exists
            disk_path = self.cache_dir / oldest_id
            if disk_path.exists():
                shutil.rmtree(disk_path, ignore_errors=True)
            
            logger.info(f"Evicted {oldest_id} from cache (LRU)")
        
        # Decide on caching strategy based on size
        if len(frames) > self.max_memory_frames:
            # Too many frames for memory - save to disk and keep subset in memory
            await self._save_frames_to_disk(video_id, frames, metadata)
            # Keep every 5th frame in memory for quick access
            memory_frames = frames[::5]
        else:
            # Keep all frames in memory
            memory_frames = frames
        
        # Add to cache
        self._frame_cache[video_id] = {'frames': memory_frames, 'full_on_disk': len(frames) > self.max_memory_frames}
        self._metadata_cache[video_id] = metadata
        self._cache_sizes[video_id] = estimated_size_mb
        self._total_cache_size += estimated_size_mb
        
    async def _save_frames_to_disk(self, 
                                  video_id: str, 
                                  frames: List[FrameData],
                                  metadata: VideoMetadata):
        """Save frames to disk for large videos"""
        frame_dir = self.cache_dir / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_file = frame_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'fps': metadata.fps,
                'total_frames': metadata.total_frames,
                'duration': metadata.duration,
                'width': metadata.width,
                'height': metadata.height,
                'extracted_frames': len(frames)
            }, f)
            
        # Save frames in parallel
        def save_frame(frame_data: FrameData, idx: int):
            frame_file = frame_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(frame_file), frame_data.image)
            return {
                'index': idx,
                'frame_number': frame_data.frame_number,
                'timestamp': frame_data.timestamp,
                'file': str(frame_file)
            }
            
        tasks = []
        for idx, frame in enumerate(frames):
            task = asyncio.to_thread(save_frame, frame, idx)
            tasks.append(task)
            
        frame_index = await asyncio.gather(*tasks)
        
        # Save index
        index_file = frame_dir / "frame_index.json"
        with open(index_file, 'w') as f:
            json.dump(frame_index, f)
            
        logger.info(f"Saved {len(frames)} frames to {frame_dir}")
        
    def get_frames_for_service(self, 
                              frames: List[FrameData],
                              service_name: str,
                              max_frames: Optional[int] = None) -> List[FrameData]:
        """
        Get optimally sampled frames for specific ML service
        Based on FRAME_PROCESSING_DOCUMENTATION.md requirements
        
        Args:
            frames: All extracted frames
            service_name: Name of ML service (yolo, mediapipe, ocr, etc.)
            max_frames: Override maximum frames for this service
            
        Returns:
            Sampled frames appropriate for the service
        """
        if not frames:
            return []
            
        config = FrameSamplingConfig.get_config(service_name)
        
        if max_frames:
            config['max_frames'] = max_frames
            
        if config['strategy'] == 'all' or config['max_frames'] is None:
            return frames
            
        if len(frames) <= config['max_frames']:
            return frames
            
        # Uniform sampling
        if config['strategy'] == 'uniform':
            interval = len(frames) // config['max_frames']
            return frames[::interval][:config['max_frames']]
            
        # Adaptive sampling (more frames at beginning and end)
        elif config['strategy'] == 'adaptive':
            total = config['max_frames']
            beginning = total // 3
            middle = total // 3
            end = total - beginning - middle
            
            result = []
            result.extend(frames[:beginning])  # First third
            
            # Middle third (sampled)
            middle_start = len(frames) // 3
            middle_end = 2 * len(frames) // 3
            middle_frames = frames[middle_start:middle_end]
            if len(middle_frames) > middle:
                interval = len(middle_frames) // middle
                result.extend(middle_frames[::interval][:middle])
            else:
                result.extend(middle_frames)
                
            result.extend(frames[-end:])  # Last third
            
            return result
            
        return frames[:config['max_frames']]
        
    async def cleanup(self):
        """Clean up resources and temporary files"""
        logger.info(f"Cache cleanup: {len(self._frame_cache)} videos, "
                   f"{self._total_cache_size:.1f}MB cached")
        
        # Clear memory cache
        self._frame_cache.clear()
        self._metadata_cache.clear()
        self._cache_sizes.clear()
        self._total_cache_size = 0
        
        # Remove temporary files
        if self.cache_dir.exists() and self.cache_dir.name.startswith("rumiai_frames_"):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        
    def get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Get cached video metadata"""
        return self._metadata_cache.get(video_id)
        
    def get_cached_frames(self, video_id: str) -> Optional[List[FrameData]]:
        """Get cached frames if available"""
        if video_id in self._frame_cache:
            return self._frame_cache[video_id].get('frames')
        return None

# Singleton instance
_frame_manager_instance = None

def get_frame_manager() -> UnifiedFrameManager:
    """Get singleton frame manager instance"""
    global _frame_manager_instance
    if _frame_manager_instance is None:
        _frame_manager_instance = UnifiedFrameManager()
    return _frame_manager_instance