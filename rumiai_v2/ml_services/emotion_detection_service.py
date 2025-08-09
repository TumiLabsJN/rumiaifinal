"""
Real emotion detection service using FEAT (Facial Expression Analysis Toolkit)
Replaces fake MediaPipe emotion mapping with state-of-the-art AU-based detection
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import cv2
import logging
from pathlib import Path
import torch
import os

# Apply FEAT compatibility patches BEFORE importing FEAT
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from scipy_compat import ensure_feat_compatibility
    ensure_feat_compatibility()
except ImportError:
    print("⚠️ FEAT compatibility patches not found, attempting direct FEAT import")

logger = logging.getLogger(__name__)

class EmotionDetectionService:
    """
    Real emotion detection using FEAT's hybrid approach
    Combines Action Units (AUs) with direct emotion classification
    87% accuracy on AffectNet dataset
    """
    
    def __init__(self, gpu: bool = True):
        """
        Initialize FEAT emotion detector with adaptive sampling
        
        Args:
            gpu: Use GPU if available (processes at 2-4 FPS on consumer GPUs)
        """
        # FEAT uses ResNet-50 for emotion detection + AU detection
        # Trained on AffectNet (420K images) + BP4D+ for AUs
        device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
        
        # Initialize with best models
        try:
            from feat import Detector
            self.detector = Detector(
                face_model='retinaface',  # Best face detection
                landmark_model='mobilefacenet',
                au_model='xgb',  # XGBoost for Action Units
                emotion_model='resmasknet',  # ResNet for emotions
                device=device
            )
        except Exception as e:
            raise RuntimeError(f"FEAT initialization failed: {e}")
        
        self.device = device
        # Sample rate now determined dynamically based on video duration
        
        # FEAT emotion categories (already matches RumiAI)
        # anger, disgust, fear, happiness, sadness, surprise, neutral
        self.emotion_mapping = {
            'anger': 'anger',
            'disgust': 'disgust', 
            'fear': 'fear',
            'happiness': 'joy',
            'sadness': 'sadness',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
        
        logger.info(f"FEAT emotion detector initialized (Device: {device})")
    
    def get_adaptive_sample_rate(self, video_duration: float) -> float:
        """
        Adaptive sampling based on video duration
        Ensures consistent processing time (~15-30s) across all video lengths
        
        Returns:
            Sample rate in FPS
        """
        if video_duration <= 30:
            return 2.0  # 2 FPS for short videos (max 60 frames)
        elif video_duration <= 60:
            return 1.0  # 1 FPS for medium videos (max 60 frames)
        else:
            return 0.5  # 0.5 FPS for long videos (max 60 frames)
    
    async def detect_emotions_batch(self, 
                                   frames: List[np.ndarray],
                                   timestamps: List[float]) -> Dict[str, Any]:
        """
        Detect emotions in batch of frames using FEAT
        Handles videos with no people gracefully (B-roll, text, animation)
        
        Args:
            frames: List of frame arrays (BGR format)
            timestamps: Corresponding timestamps
            
        Returns:
            Emotion detection results with AUs and confidence scores
            Always returns structured data - never crashes for videos without people
        """
        results = {
            'emotions': [],
            'action_units': [],
            'timeline': {},
            'dominant_emotion': None,
            'emotion_transitions': [],
            'confidence_scores': [],
            'au_activations': {},
            'processing_stats': {
                'total_frames': len(frames),
                'successful_frames': 0,
                'no_face_frames': 0,
                'video_type': 'unknown'
            }
        }
        
        emotion_counts = {}
        prev_emotion = None
        
        # Process frames in batches for GPU efficiency
        batch_size = 8 if self.device == 'cuda' else 4
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_timestamps = timestamps[i:i+batch_size]
            
            try:
                # Run FEAT detection on batch
                detections = await asyncio.to_thread(
                    self._detect_batch, batch_frames
                )
                
                # Process each detection
                for j, detection in enumerate(detections):
                    timestamp = batch_timestamps[j]
                    
                    if detection.get('no_face'):
                        # No face detected in this frame (normal for B-roll, text, etc.)
                        results['processing_stats']['no_face_frames'] += 1
                        time_key = f"{int(timestamp)}-{int(timestamp)+1}s"
                        results['timeline'][time_key] = {'no_face': True}
                        continue
                    
                    # Valid emotion detection
                    results['processing_stats']['successful_frames'] += 1
                    
                    # Store raw detection with AUs
                    results['emotions'].append({
                        'timestamp': timestamp,
                        'emotion': detection['emotion'],
                        'confidence': detection['confidence'],
                        'all_scores': detection['emotion_scores'],
                        'action_units': detection['action_units'],
                        'au_intensities': detection['au_intensities']
                    })
                    
                    # Store Action Units
                    results['action_units'].append({
                        'timestamp': timestamp,
                        'aus': detection['action_units'],
                        'intensities': detection['au_intensities']
                    })
                    
                    # Build timeline
                    time_key = f"{int(timestamp)}-{int(timestamp)+1}s"
                    results['timeline'][time_key] = detection
                    
                    # Track transitions
                    if prev_emotion and prev_emotion != detection['emotion']:
                        results['emotion_transitions'].append({
                            'timestamp': timestamp,
                            'from': prev_emotion,
                            'to': detection['emotion']
                        })
                    prev_emotion = detection['emotion']
                    
                    # Count for dominant emotion
                    emotion = detection['emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    # Track confidence
                    results['confidence_scores'].append(detection['confidence'])
                    
                    # Track AU activations
                    for au in detection['action_units']:
                        if au not in results['au_activations']:
                            results['au_activations'][au] = 0
                        results['au_activations'][au] += 1
                        
            except Exception as e:
                # FEAT crashed - this is a bug, let it fail fast
                logger.error(f"FEAT detection crashed at batch {batch_timestamps[0]}: {e}")
                raise  # Re-raise the exception to fail fast
        
        # Determine video type based on processing results
        total_frames = len(frames)
        success_rate = results['processing_stats']['successful_frames'] / total_frames if total_frames > 0 else 0
        no_face_rate = results['processing_stats']['no_face_frames'] / total_frames if total_frames > 0 else 0
        
        if no_face_rate > 0.8:  # 80%+ frames have no faces
            results['processing_stats']['video_type'] = 'no_people'
            results['dominant_emotion'] = None
            results['metrics'] = {
                'video_type': 'no_people',
                'detection_rate': 0.0,
                'suitable_for_emotion_analysis': False,
                'reason': 'No faces detected in video (B-roll, text, animation)',
                'processing_stats': results['processing_stats']
            }
            return results  # Valid outcome - not an error
        
        else:
            # Normal video with people detected
            results['processing_stats']['video_type'] = 'people_detected'
            
            # Calculate dominant emotion
            if emotion_counts:
                results['dominant_emotion'] = max(emotion_counts, key=emotion_counts.get)
            
            # Calculate metrics (trust FEAT's internal AU-emotion correlation)
            results['metrics'] = {
                'video_type': 'people_detected',
                'unique_emotions': len(set(emotion_counts.keys())),
                'transition_count': len(results['emotion_transitions']),
                'avg_confidence': np.mean(results['confidence_scores']) if results['confidence_scores'] else 0,
                'emotion_diversity': self._calculate_diversity(emotion_counts),
                'detection_rate': success_rate,
                'suitable_for_emotion_analysis': True,
                'most_active_aus': sorted(results['au_activations'].items(), key=lambda x: x[1], reverse=True)[:5],
                'processing_stats': results['processing_stats']
            }
        
        return results
    
    def _validate_feat_output(self, predictions) -> bool:
        """
        Validate FEAT output format matches our expectations
        Returns True if valid, logs errors and returns False otherwise
        """
        if predictions is None:
            logger.error("FEAT returned None")
            return False
            
        if not hasattr(predictions, 'columns'):
            logger.error(f"FEAT output is not a DataFrame, got {type(predictions)}")
            return False
        
        # Check for required columns (may vary by FEAT version)
        required_columns = {
            'face_detection': ['FaceScore', 'FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight'],
            'emotions': ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral'],
            'action_units': ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 
                           'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26']
        }
        
        missing = []
        for category, cols in required_columns.items():
            for col in cols:
                if col not in predictions.columns:
                    missing.append(f"{category}.{col}")
        
        if missing:
            logger.warning(f"FEAT output missing expected columns: {missing[:5]}...")  # Log first 5
            # Don't fail - FEAT versions may vary
        
        return len(predictions) > 0
    
    def _safe_extract_emotion_scores(self, pred) -> Dict[str, float]:
        """
        Safely extract emotion scores with fallbacks for different FEAT versions
        """
        # Try multiple possible column names for each emotion
        emotion_mappings = {
            'anger': ['anger', 'angry', 'Anger'],
            'disgust': ['disgust', 'Disgust'],
            'fear': ['fear', 'Fear', 'afraid'],
            'happiness': ['happiness', 'happy', 'joy', 'Happy'],
            'sadness': ['sadness', 'sad', 'Sad'],
            'surprise': ['surprise', 'surprised', 'Surprise'],
            'neutral': ['neutral', 'Neutral']
        }
        
        scores = {}
        for emotion, possible_names in emotion_mappings.items():
            found = False
            for name in possible_names:
                if name in pred.index:
                    scores[emotion] = float(pred[name])
                    found = True
                    break
            if not found:
                scores[emotion] = 0.0
                
        return scores
    
    def _safe_extract_face_bbox(self, pred) -> List[float]:
        """
        Safely extract face bounding box with coordinate system conversion if needed
        """
        # FEAT may use different coordinate conventions
        bbox_mappings = [
            ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight'],
            ['face_x', 'face_y', 'face_width', 'face_height'],
            ['x', 'y', 'width', 'height'],
            ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
        ]
        
        for mapping in bbox_mappings:
            if all(col in pred.index for col in mapping):
                return [float(pred[col]) for col in mapping]
        
        # Return empty bbox if not found
        return [0, 0, 0, 0]
    
    def _detect_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Detect emotions in batch using FEAT (synchronous)
        
        Returns list of detections for each frame
        If FEAT crashes, let it fail fast - this indicates a bug to fix
        """
        import tempfile
        import os
        
        # FEAT expects image file paths, not numpy arrays
        temp_files = []
        try:
            # Save frames to temporary files
            for i, frame in enumerate(frames):
                # Convert BGR to RGB for saving
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix=f'_frame_{i}.jpg', delete=False)
                cv2.imwrite(temp_file.name, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))  # cv2.imwrite expects BGR
                temp_files.append(temp_file.name)
                temp_file.close()
            
            # Run FEAT detection on file paths
            predictions = self.detector.detect_image(temp_files)
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass  # File already deleted or doesn't exist
        
        # Validate output format on first batch
        if not hasattr(self, '_format_validated'):
            if self._validate_feat_output(predictions):
                self._format_validated = True
                logger.info(f"FEAT output format validated: {len(predictions.columns)} columns")
            else:
                logger.warning("FEAT output format unexpected but continuing")
                self._format_validated = True
        
        results = []
        for i in range(len(frames)):
            # Handle different FEAT output structures
            if 'frame' in predictions.columns:
                # FEAT may include frame index
                frame_predictions = predictions[predictions['frame'] == i]
            elif len(predictions) == len(frames):
                # Simple 1:1 mapping
                frame_predictions = predictions.iloc[i:i+1]
            else:
                # Unexpected format - try to get row i
                try:
                    frame_predictions = predictions.iloc[i:i+1]
                except IndexError:
                    results.append({'no_face': True, 'reason': 'index_error'})
                    continue
            
            if len(frame_predictions) == 0 or frame_predictions.iloc[0].get('FaceScore', 0) <= 0.5:
                # No face detected
                results.append({'no_face': True, 'face_score': 0.0})
                continue
            
            # Multiple faces: pick the largest one (duets, groups, etc.)
            if len(frame_predictions) > 1:
                # Calculate face areas and pick largest
                face_areas = (frame_predictions['FaceRectWidth'] * frame_predictions['FaceRectHeight'])
                largest_idx = face_areas.idxmax()
                pred = frame_predictions.loc[largest_idx]
                logger.debug(f"Frame {i}: Multiple faces detected ({len(frame_predictions)}), using largest")
            else:
                pred = frame_predictions.iloc[0]
            
            # Check if selected face meets minimum threshold (safe access)
            face_score = float(pred.get('FaceScore', 0)) if 'FaceScore' in pred.index else 0.0
            
            if face_score > 0.5:
                # Extract emotions using safe method
                emotion_scores = self._safe_extract_emotion_scores(pred)
                
                # Get dominant emotion
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                mapped_emotion = self.emotion_mapping.get(dominant_emotion, 'neutral')
                
                # Extract Action Units safely
                action_units = []
                au_intensities = {}
                for au_col in [col for col in pred.index if col.startswith('AU') and col[2:].isdigit()]:
                    try:
                        au_num = int(au_col[2:])
                        au_intensity = float(pred[au_col])
                        if au_intensity > 0.5:  # AU is active
                            action_units.append(au_num)
                            au_intensities[au_num] = au_intensity
                    except (ValueError, TypeError):
                        continue  # Skip malformed AU columns
                
                results.append({
                    'emotion': mapped_emotion,
                    'confidence': emotion_scores[dominant_emotion],
                    'emotion_scores': {self.emotion_mapping.get(k, k): v for k, v in emotion_scores.items()},
                    'action_units': action_units,
                    'au_intensities': au_intensities,
                    'face_bbox': self._safe_extract_face_bbox(pred)
                })
            else:
                # No face detected - this is normal for many videos
                results.append({'no_face': True, 'face_score': face_score})
        
        return results
    
    def _calculate_diversity(self, emotion_counts: Dict) -> float:
        """Calculate emotion diversity score (0-1)"""
        if not emotion_counts:
            return 0.0
            
        total = sum(emotion_counts.values())
        probabilities = [count/total for count in emotion_counts.values()]
        
        # Shannon entropy normalized to 0-1
        entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
        max_entropy = np.log(len(emotion_counts))
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    
    async def analyze_emotional_journey(self, 
                                       frames: List[np.ndarray],
                                       timestamps: List[float],
                                       audio_features: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete emotional journey analysis
        
        Args:
            frames: Video frames
            timestamps: Frame timestamps
            audio_features: Optional audio emotion features
            
        Returns:
            Full emotional journey analysis matching RumiAI format
        """
        # Get base emotion detection
        emotion_data = await self.detect_emotions_batch(frames, timestamps)
        
        # Build emotional arc
        arc = self._classify_emotional_arc(emotion_data['emotions'])
        
        # Find emotional peaks (high confidence moments)
        peaks = self._find_emotional_peaks(emotion_data['emotions'])
        
        # Classify journey archetype
        archetype = self._classify_journey_archetype(arc, peaks)
        
        return {
            # Raw data for other services
            'raw_emotions': emotion_data,
            
            # Formatted for RumiAI Emotional Journey
            'emotionalCoreMetrics': {
                'uniqueEmotions': emotion_data['metrics']['unique_emotions'],
                'emotionTransitions': emotion_data['metrics']['transition_count'],
                'dominantEmotion': emotion_data['dominant_emotion'],
                'emotionalDiversity': emotion_data['metrics']['emotion_diversity'],
                'avgConfidence': emotion_data['metrics']['avg_confidence'],
                'detectionRate': emotion_data['metrics']['detection_rate']
            },
            
            'emotionalDynamics': {
                'emotionProgression': emotion_data['emotions'],
                'transitionPoints': emotion_data['emotion_transitions'],
                'emotionalArc': arc,
                'stabilityScore': self._calculate_stability(emotion_data['emotions'])
            },
            
            'emotionalKeyEvents': {
                'emotionalPeaks': peaks,
                'climaxMoment': peaks[0] if peaks else None,
                'resolutionMoment': self._find_resolution(emotion_data['emotions'])
            },
            
            'emotionalPatterns': {
                'journeyArchetype': archetype,
                'emotionalTechniques': self._detect_techniques(emotion_data),
                'viewerJourneyMap': self._predict_viewer_journey(archetype)
            }
        }
    
    def _classify_emotional_arc(self, emotions: List[Dict]) -> str:
        """Classify the overall emotional arc pattern"""
        if not emotions:
            return 'stable'
            
        # Map emotions to valence scores
        valence_map = {
            'joy': 1.0, 'surprise': 0.5, 'neutral': 0.0,
            'sadness': -0.5, 'anger': -0.8, 'fear': -0.7, 'disgust': -0.6
        }
        
        valences = [valence_map.get(e['emotion'], 0) for e in emotions]
        
        # Analyze trend
        first_third = np.mean(valences[:len(valences)//3])
        last_third = np.mean(valences[2*len(valences)//3:])
        
        if last_third > first_third + 0.3:
            return 'rising'
        elif first_third > last_third + 0.3:
            return 'falling'
        elif np.std(valences) > 0.5:
            return 'rollercoaster'
        else:
            return 'stable'
    
    def _find_emotional_peaks(self, emotions: List[Dict], threshold: float = 0.8) -> List[Dict]:
        """Find high-intensity emotional moments"""
        peaks = []
        
        for i, emotion in enumerate(emotions):
            if emotion['confidence'] >= threshold:
                # Check if it's a local maximum
                is_peak = True
                window = 2  # Check 2 frames before and after
                
                for j in range(max(0, i-window), min(len(emotions), i+window+1)):
                    if j != i and emotions[j]['confidence'] > emotion['confidence']:
                        is_peak = False
                        break
                
                if is_peak:
                    peaks.append({
                        'timestamp': emotion['timestamp'],
                        'emotion': emotion['emotion'],
                        'intensity': emotion['confidence'],
                        'type': 'peak'
                    })
        
        return sorted(peaks, key=lambda x: x['intensity'], reverse=True)[:5]
    
    def _classify_journey_archetype(self, arc: str, peaks: List[Dict]) -> str:
        """Classify the emotional journey archetype"""
        if not peaks:
            return 'steady_state'
            
        peak_emotions = [p['emotion'] for p in peaks[:3]]
        
        if arc == 'rising' and 'joy' in peak_emotions:
            return 'surprise_delight'
        elif arc == 'falling' and any(e in peak_emotions for e in ['sadness', 'anger']):
            return 'problem_solution'
        elif arc == 'rollercoaster':
            return 'transformation'
        elif 'surprise' in peak_emotions:
            return 'discovery'
        else:
            return 'narrative'
    
    def _calculate_stability(self, emotions: List[Dict]) -> float:
        """Calculate emotional stability score"""
        if len(emotions) < 2:
            return 1.0
            
        transitions = sum(
            1 for i in range(1, len(emotions))
            if emotions[i]['emotion'] != emotions[i-1]['emotion']
        )
        
        # Normalize: 0 transitions = 1.0, many transitions = low score
        max_transitions = len(emotions) - 1
        stability = 1.0 - (transitions / max_transitions)
        
        return stability
    
    def _find_resolution(self, emotions: List[Dict]) -> Optional[Dict]:
        """Find emotional resolution moment (return to neutral/positive)"""
        if len(emotions) < 5:
            return None
            
        # Check last 20% of video
        last_segment = emotions[int(len(emotions) * 0.8):]
        
        for emotion in last_segment:
            if emotion['emotion'] in ['neutral', 'joy']:
                return {
                    'timestamp': emotion['timestamp'],
                    'emotion': emotion['emotion'],
                    'type': 'resolution'
                }
        
        return None
    
    def _detect_techniques(self, emotion_data: Dict) -> List[str]:
        """Detect emotional techniques used"""
        techniques = []
        
        if emotion_data['metrics']['transition_count'] > 5:
            techniques.append('rapid_shifts')
        
        if emotion_data['metrics']['emotion_diversity'] > 0.7:
            techniques.append('emotional_variety')
            
        if any(t['from'] == 'neutral' and t['to'] in ['joy', 'surprise'] 
               for t in emotion_data['emotion_transitions']):
            techniques.append('surprise_reveal')
            
        if emotion_data['dominant_emotion'] in ['joy', 'surprise']:
            techniques.append('positive_framing')
            
        return techniques
    
    def _predict_viewer_journey(self, archetype: str) -> str:
        """Predict viewer emotional journey based on archetype"""
        journey_map = {
            'surprise_delight': 'engaged_throughout',
            'problem_solution': 'tension_release',
            'transformation': 'emotional_investment',
            'discovery': 'curiosity_driven',
            'narrative': 'story_following',
            'steady_state': 'passive_viewing'
        }
        
        return journey_map.get(archetype, 'variable_engagement')


def get_emotion_detector() -> EmotionDetectionService:
    """Factory function to get FEAT emotion detector instance"""
    use_gpu = torch.cuda.is_available()
    # Uses adaptive sampling based on video duration
    return EmotionDetectionService(gpu=use_gpu)