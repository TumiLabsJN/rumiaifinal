"""
Audio Energy Service for real audio dynamics analysis
Uses librosa to extract actual energy metrics from audio files
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class AudioEnergyService:
    """Service for analyzing audio energy dynamics"""
    
    def __init__(self):
        """Initialize the audio energy analyzer"""
        self.sample_rate = 16000  # Standard rate for audio processing
        self.window_seconds = 5  # 5-second windows for energy analysis
        
    async def analyze(self, audio_path: Path) -> Dict[str, Any]:
        """
        Analyze audio energy from WAV file
        
        Args:
            audio_path: Path to audio file (WAV format)
            
        Returns:
            Dictionary with energy metrics
        """
        try:
            # Import librosa here to avoid startup penalty if not used
            import librosa
            
            # Run librosa analysis in thread pool to avoid blocking
            result = await asyncio.to_thread(
                self._analyze_energy_sync,
                audio_path,
                librosa
            )
            
            return result
            
        except ImportError:
            logger.error("librosa not installed. Install with: pip install librosa")
            raise
        except Exception as e:
            logger.error(f"Energy analysis failed for {audio_path}: {e}")
            return self._empty_result(error=str(e))
    
    def _analyze_energy_sync(self, audio_path: Path, librosa) -> Dict[str, Any]:
        """
        Synchronous energy analysis using librosa
        
        Args:
            audio_path: Path to audio file
            librosa: The librosa module
            
        Returns:
            Energy analysis results
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        
        # Get duration
        duration = len(y) / sr
        
        # Calculate energy for 5-second windows
        samples_per_window = self.window_seconds * sr
        hop_samples = int(512)  # Hop length used in RMS
        frames_per_window = samples_per_window // hop_samples
        
        energy_windows = {}
        window_energies = []
        
        for i in range(0, len(rms), frames_per_window):
            window_start = i * hop_samples / sr
            window_end = min((i + frames_per_window) * hop_samples / sr, duration)
            window_key = f"{int(window_start)}-{int(window_end)}s"
            
            # Get RMS values for this window
            window_rms = rms[i:i+frames_per_window]
            if len(window_rms) > 0:
                # Use mean RMS for the window
                window_energy = float(np.mean(window_rms))
                window_energies.append(window_energy)
                energy_windows[window_key] = window_energy
        
        # Normalize using percentile-based approach (preserve dynamics)
        if window_energies:
            # Use 95th percentile for normalization to avoid outliers
            p95 = np.percentile(window_energies, 95)
            if p95 > 0:
                # Normalize to 0-1 range
                for key in energy_windows:
                    energy_windows[key] = min(1.0, energy_windows[key] / p95)
        
        # Calculate variance
        energy_variance = float(np.var(window_energies)) if window_energies else 0.0
        
        # Find climax (peak energy moment)
        climax_timestamp = 0.0
        if len(rms) > 0:
            peak_frame = np.argmax(rms)
            climax_timestamp = float(peak_frame * hop_samples / sr)
        
        # Determine burst pattern
        burst_pattern = self._determine_burst_pattern(window_energies)
        
        return {
            "energy_level_windows": energy_windows,
            "energy_variance": energy_variance,
            "climax_timestamp": climax_timestamp,
            "burst_pattern": burst_pattern,
            "duration": duration,
            "metadata": {
                "processed": True,
                "success": True,
                "method": "librosa_rms",
                "sample_rate": sr,
                "window_seconds": self.window_seconds
            }
        }
    
    def _determine_burst_pattern(self, energies: list) -> str:
        """
        Determine the burst pattern from energy values
        
        Args:
            energies: List of energy values for each window
            
        Returns:
            Burst pattern type: "front_loaded", "back_loaded", "middle_peak", or "steady"
        """
        if not energies or len(energies) < 2:
            return "steady"
        
        # Divide into thirds
        third_size = len(energies) // 3
        if third_size == 0:
            third_size = 1
            
        front_third = energies[:third_size]
        middle_third = energies[third_size:2*third_size] if len(energies) > third_size else []
        back_third = energies[2*third_size:] if len(energies) > 2*third_size else []
        
        # Calculate average energy for each third
        front_avg = np.mean(front_third) if front_third else 0
        middle_avg = np.mean(middle_third) if middle_third else 0
        back_avg = np.mean(back_third) if back_third else 0
        
        # Determine pattern based on which third has highest average
        max_avg = max(front_avg, middle_avg, back_avg)
        
        # Need at least 20% difference to be considered a pattern
        threshold = max_avg * 0.2
        
        if abs(max_avg - front_avg) < threshold and abs(max_avg - back_avg) < threshold:
            return "steady"
        elif front_avg == max_avg:
            return "front_loaded"
        elif back_avg == max_avg:
            return "back_loaded"
        elif middle_avg == max_avg:
            return "middle_peak"
        else:
            return "steady"
    
    def _empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty result with error state"""
        result = {
            "energy_level_windows": {},
            "energy_variance": 0.0,
            "climax_timestamp": 0.0,
            "burst_pattern": "unknown",
            "duration": 0.0,
            "metadata": {
                "processed": False,
                "success": False,
                "method": "librosa_rms"
            }
        }
        if error:
            result["metadata"]["error"] = error
        return result
    
    async def save_results(self, results: Dict[str, Any], video_id: str, output_dir: Path):
        """
        Save energy analysis results to JSON file
        
        Args:
            results: Energy analysis results
            video_id: Video identifier
            output_dir: Directory to save results
        """
        output_path = output_dir / f"{video_id}_audio_energy.json"
        
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Saved audio energy results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save energy results: {e}")
            raise


def get_audio_energy_service() -> AudioEnergyService:
    """Get audio energy service instance"""
    return AudioEnergyService()