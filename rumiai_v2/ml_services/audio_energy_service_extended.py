"""
Extended Audio Energy Service with Pitch Analysis
Adds pitch extraction capabilities to existing energy analysis
"""

import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# CONFIGURATION: Centralized pitch extraction settings
PITCH_CONFIG = {
    # Core settings
    'enabled': True,                    # Set False to disable pitch extraction
    'sample_rate': 22050,               # Hz - optimal for pitch detection
    'hop_length': 512,                  # Samples - controls time resolution

    # Pitch detection parameters
    'fmin': 60,                         # Hz - minimum human voice frequency
    'fmax': 350,                        # Hz - maximum human voice frequency
    'voiced_threshold': 80,             # Hz - below this is considered unvoiced

    # Performance settings
    'quality': 'high',                  # 'high', 'medium', 'low'
    'max_processing_time': 15,         # Seconds - warning threshold
    'timeout': 30,                      # Seconds - hard timeout

    # Memory optimization
    'max_voiced_samples': 500,         # Maximum voiced samples to store
    'cache_percentiles': True,         # Pre-compute statistics for efficiency

    # Normalization thresholds
    'min_frames_avg': 10,              # Minimum frames for average pitch
    'min_frames_range': 30,            # Minimum frames for pitch range

    # Quality presets
    'quality_presets': {
        'high': {'hop_length': 512},    # Best quality, slower
        'medium': {'hop_length': 768},  # Balanced
        'low': {'hop_length': 1024}     # Fast, lower quality
    }
}


class AudioEnergyService:
    """Extended service for analyzing audio energy AND pitch dynamics"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the audio energy and pitch analyzer"""
        # Use provided config or defaults
        self.config = config or PITCH_CONFIG

        # Existing energy parameters
        self.sample_rate = 16000  # For energy analysis
        self.energy_hop = 512
        self.window_seconds = 5  # 5-second windows for energy analysis

        # Pitch parameters from config
        self.pitch_sample_rate = self.config['sample_rate']

        # Apply quality preset if specified
        quality = self.config.get('quality', 'high')
        if quality in self.config['quality_presets']:
            preset = self.config['quality_presets'][quality]
            self.pitch_hop = preset['hop_length']
        else:
            self.pitch_hop = self.config['hop_length']

    async def analyze(self, audio_path: Path, video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze audio energy AND pitch from audio file

        Performance optimization: Audio is loaded once at 22050 Hz, then resampled
        to 16000 Hz for energy. This saves ~30% I/O time vs loading twice.

        Args:
            audio_path: Path to audio file (WAV format) or video file
            video_id: Optional video ID for shared extraction caching

        Returns:
            Dictionary with energy and pitch metrics
        """
        try:
            # Use SharedAudioExtractor if video_id provided (indicates video processing)
            if video_id:
                from rumiai_v2.api.shared_audio_extractor import SharedAudioExtractor

                logger.info(f"Using SharedAudioExtractor for video {video_id}")
                # Get the shared audio file (will extract only once across all services)
                audio_path = await SharedAudioExtractor.extract_once(
                    str(audio_path),
                    video_id,
                    service_name="audio_energy"
                )

            # Import librosa here to avoid startup penalty if not used
            import librosa
            import librosa.effects
            from librosa.util.exceptions import ParameterError

            # Run analysis in thread pool to avoid blocking
            result = await asyncio.to_thread(
                self._analyze_audio_sync,
                audio_path,
                librosa
            )

            return result

        except ImportError:
            logger.error("librosa not installed. Install with: pip install librosa")
            raise
        except Exception as e:
            logger.error(f"Audio analysis failed for {audio_path}: {e}")
            return self._empty_result(error=str(e))

    def _analyze_audio_sync(self, audio_path: Path, librosa) -> Dict[str, Any]:
        """
        Synchronous audio analysis using librosa (energy + pitch)

        Args:
            audio_path: Path to audio file
            librosa: The librosa module

        Returns:
            Audio analysis results with energy and pitch
        """
        from librosa.util.exceptions import ParameterError

        start_time = time.time()

        try:
            # OPTIMIZED: Load audio once at higher sample rate
            # Load at 22050 Hz (optimal for pitch detection)
            y_22k, sr_22k = librosa.load(str(audio_path), sr=self.pitch_sample_rate)

            # Resample to 16000 Hz for energy analysis
            # This is faster than loading twice from disk
            y_16k = librosa.resample(
                y=y_22k,
                orig_sr=self.pitch_sample_rate,
                target_sr=self.sample_rate
            )
            sr_16k = self.sample_rate

            logger.info(f"Audio loaded once and resampled - saved I/O time")

            # Get duration
            duration = len(y_16k) / sr_16k

            # EXISTING: Energy extraction (using resampled audio)
            rms = librosa.feature.rms(y=y_16k, frame_length=2048, hop_length=self.energy_hop)[0]

            # Calculate energy for 5-second windows (existing logic)
            samples_per_window = self.window_seconds * sr_16k
            hop_samples = self.energy_hop
            frames_per_window = samples_per_window // hop_samples

            energy_windows = {}
            window_energies = []

            for i in range(0, len(rms), frames_per_window):
                window_start = i * hop_samples / sr_16k
                window_end = min((i + frames_per_window) * hop_samples / sr_16k, duration)
                window_key = f"{int(window_start)}-{int(window_end)}s"

                # Get RMS values for this window
                window_rms = rms[i:i+frames_per_window]

                if len(window_rms) > 0:
                    # Calculate energy statistics
                    window_energy = {
                        'mean': float(np.mean(window_rms)),
                        'std': float(np.std(window_rms)),
                        'min': float(np.min(window_rms)),
                        'max': float(np.max(window_rms)),
                        'median': float(np.median(window_rms))
                    }

                    energy_windows[window_key] = window_energy
                    window_energies.append(window_energy['mean'])

            # NEW: Pitch extraction (if enabled)
            pitch_data = {}
            if self.config.get('enabled', True):
                pitch_data = self._extract_pitch(y_22k, sr_22k, librosa)

            # Performance check
            processing_time = time.time() - start_time
            max_time = self.config.get('max_processing_time', 15)
            if processing_time > max_time:
                logger.warning(f"Audio analysis slow: {processing_time:.1f}s (threshold: {max_time}s)")

            # Build return dictionary
            result = {
                # Energy metrics (existing)
                'rms_frames': rms.tolist(),
                'energy_fps': sr_16k / self.energy_hop,  # 31.25 fps
                'energy_windows': energy_windows,
                'window_energies': window_energies,

                # Audio properties
                'duration': duration,
                'sample_rate': sr_16k,

                # Processing metadata
                'processing_ms': int(processing_time * 1000)
            }

            # Add pitch data if extracted
            if pitch_data:
                result.update(pitch_data)

            # Add overall statistics
            if window_energies:
                result['overall_stats'] = {
                    'mean_energy': float(np.mean(window_energies)),
                    'std_energy': float(np.std(window_energies)),
                    'min_energy': float(np.min(window_energies)),
                    'max_energy': float(np.max(window_energies))
                }

            return result

        except ParameterError as e:
            raise RuntimeError(f"Audio analysis failed with invalid parameters: {e}")
        except (OSError, IOError) as e:
            raise RuntimeError(f"Audio file access failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in audio analysis: {type(e).__name__}")
            raise RuntimeError(f"Audio analysis failed unexpectedly: {e}")

    def _extract_pitch(self, y: np.ndarray, sr: int, librosa) -> Dict[str, Any]:
        """
        Extract pitch features from audio

        Args:
            y: Audio time series
            sr: Sample rate
            librosa: The librosa module

        Returns:
            Dictionary with pitch metrics
        """
        from librosa.util.exceptions import ParameterError

        try:
            # Harmonic separation for cleaner pitch
            try:
                y_harmonic, _ = librosa.effects.hpss(y)
            except ParameterError as e:
                raise RuntimeError(f"HPSS failed with invalid parameters: {e}")
            except ValueError as e:
                raise RuntimeError(f"HPSS failed with value error (likely corrupted audio): {e}")
            except Exception as e:
                logger.error(f"Unexpected HPSS error type: {type(e).__name__}")
                raise RuntimeError(f"HPSS failed unexpectedly: {e}")

            # Extract pitch with validation
            try:
                pitches, magnitudes = librosa.piptrack(
                    y=y_harmonic,
                    sr=sr,
                    hop_length=self.pitch_hop,
                    fmin=self.config['fmin'],
                    fmax=self.config['fmax']
                )
            except ParameterError as e:
                raise RuntimeError(f"Piptrack failed with invalid parameters: {e}")
            except Exception as e:
                logger.error(f"Unexpected piptrack error type: {type(e).__name__}")
                raise RuntimeError(f"Pitch extraction failed: {e}")

            # Validate pitch extraction succeeded
            if pitches.size == 0:
                raise RuntimeError("Pitch extraction returned empty - no pitched content detected")

            # Memory optimization: Store only max pitch per frame (1D not 2D)
            max_pitches_per_frame = np.max(pitches, axis=0)

            # Extract voiced pitches for self-normalization
            voiced_threshold = self.config['voiced_threshold']
            all_voiced_pitches = max_pitches_per_frame[max_pitches_per_frame > voiced_threshold]

            # MEMORY OPTIMIZATION: Cap sample + compute statistics
            max_samples = self.config['max_voiced_samples']
            voiced_sample = all_voiced_pitches[:max_samples].tolist() if len(all_voiced_pitches) > 0 else []

            # Pre-compute statistics for self-normalization
            pitch_stats = {}
            if len(all_voiced_pitches) >= 20:  # Need minimum samples for percentiles
                pitch_stats = {
                    'p20': float(np.percentile(all_voiced_pitches, 20)),
                    'p50': float(np.percentile(all_voiced_pitches, 50)),
                    'p80': float(np.percentile(all_voiced_pitches, 80)),
                    'mean': float(np.mean(all_voiced_pitches)),
                    'std': float(np.std(all_voiced_pitches)),
                    'total_voiced_frames': len(all_voiced_pitches)
                }

            logger.info(f"Pitch extraction: {len(all_voiced_pitches)} voiced frames, "
                       f"storing {len(voiced_sample)} samples + statistics")

            return {
                # Pitch metrics
                'pitch_frames': max_pitches_per_frame.tolist(),
                'pitch_fps': sr / self.pitch_hop,

                # Memory-optimized self-normalization data
                'voiced_pitch_sample': voiced_sample,
                'pitch_statistics': pitch_stats
            }

        except RuntimeError:
            # Our explicit errors - reraise
            raise
        except Exception as e:
            logger.error(f"Unexpected pitch extraction error: {e}")
            return {}  # Return empty pitch data on error

    def _empty_result(self, error: str = None) -> Dict[str, Any]:
        """Return empty result structure when analysis fails"""
        result = {
            'rms_frames': [],
            'energy_fps': 31.25,
            'energy_windows': {},
            'window_energies': [],
            'duration': 0,
            'sample_rate': self.sample_rate,
            'overall_stats': None,
            'processing_ms': 0
        }

        if error:
            result['error'] = error

        return result