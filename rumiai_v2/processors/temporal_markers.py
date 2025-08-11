"""
Temporal marker processor for RumiAI v2.

CRITICAL: This replaces ALL the broken temporal marker generators.
This is the ONLY implementation.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json
import sys

from ..core.models import Timestamp, Timeline, UnifiedAnalysis
from ..core.exceptions import MLAnalysisError, handle_error
from ..core.validators import MLDataValidator

logger = logging.getLogger(__name__)


class TemporalMarkerProcessor:
    """
    Generate temporal markers from unified analysis.
    
    Generates temporal markers from unified ML analysis.
    Used in Python-only processing pipeline.
    """
    
    def __init__(self):
        self.validator = MLDataValidator()
    
    def generate_markers(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """
        Generate temporal markers for video.
        
        Generate temporal markers from ML data for video analysis.
        Returns dictionary with first_5_seconds, cta_window, peak_moments, etc.
        """
        try:
            # Extract data from ML results
            yolo_data = analysis.get_ml_data('yolo')
            whisper_data = analysis.get_ml_data('whisper')
            ocr_data = analysis.get_ml_data('ocr')
            mediapipe_data = analysis.get_ml_data('mediapipe')
            scene_data = analysis.get_ml_data('scene_detection')
            
            # Log what data we have
            logger.info(f"Generating temporal markers for {analysis.video_id}")
            logger.info(f"Available ML data: YOLO={yolo_data is not None}, "
                       f"Whisper={whisper_data is not None}, "
                       f"OCR={ocr_data is not None}, "
                       f"MediaPipe={mediapipe_data is not None}, "
                       f"Scene={scene_data is not None}")
            
            # Generate markers for different time windows
            markers = {
                'first_5_seconds': self._analyze_opening(analysis, 0, 5),
                'cta_window': self._analyze_cta_window(analysis),
                'peak_moments': self._find_peak_moments(analysis),
                'engagement_curve': self._calculate_engagement_curve(analysis),
                'metadata': {
                    'video_id': analysis.video_id,
                    'duration': analysis.timeline.duration,
                    'generated_at': datetime.utcnow().isoformat(),
                    'ml_data_available': {
                        'yolo': yolo_data is not None,
                        'whisper': whisper_data is not None,
                        'ocr': ocr_data is not None,
                        'mediapipe': mediapipe_data is not None,
                        'scene': scene_data is not None
                    }
                }
            }
            
            return markers
            
        except Exception as e:
            # Log error and return empty markers structure
            logger.error(f"Temporal marker generation failed: {str(e)}", exc_info=True)
            # ALWAYS return valid JSON structure
            return self._get_empty_markers(analysis.video_id, str(e))
    
    def _analyze_opening(self, analysis: UnifiedAnalysis, start: float, end: float) -> Dict[str, Any]:
        """Analyze opening seconds of video."""
        timeline_entries = analysis.timeline.get_entries_in_range(start, end)
        
        return {
            'density_progression': self._calculate_density_progression(timeline_entries, start, end),
            'text_moments': self._extract_text_moments(timeline_entries),
            'emotion_sequence': self._extract_emotion_sequence(analysis, start, end),
            'gesture_moments': self._extract_gesture_moments(timeline_entries),
            'object_appearances': self._extract_object_appearances(timeline_entries),
            'speech_segments': self._extract_speech_segments(timeline_entries),
            'scene_changes': self._extract_scene_changes(analysis, start, end)
        }
    
    def _analyze_cta_window(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """Analyze call-to-action window (typically last 15% of video)."""
        duration = analysis.timeline.duration
        cta_start = duration * 0.85  # Last 15%
        
        timeline_entries = analysis.timeline.get_entries_in_range(cta_start, duration)
        
        return {
            'time_range': f'{int(cta_start)}-{int(duration)}s',
            'cta_appearances': self._find_cta_keywords(timeline_entries),
            'gesture_sync': self._analyze_gesture_sync(timeline_entries),
            'object_focus': self._analyze_object_focus(timeline_entries),
            'speech_emphasis': self._analyze_speech_emphasis(timeline_entries),
            'text_overlays': self._extract_text_moments(timeline_entries)
        }
    
    def _find_peak_moments(self, analysis: UnifiedAnalysis) -> List[Dict[str, Any]]:
        """Find peak engagement moments in the video."""
        peaks = []
        
        # Calculate activity density per second
        density = self._calculate_activity_density(analysis)
        
        # Find peaks (local maxima above threshold)
        threshold = sum(density) / len(density) * 1.5 if density else 0
        
        for i in range(1, len(density) - 1):
            if density[i] > threshold and density[i] > density[i-1] and density[i] > density[i+1]:
                # This is a peak
                peak_time = float(i)
                peaks.append({
                    'timestamp': peak_time,
                    'intensity': density[i],
                    'events': self._get_events_at_time(analysis, peak_time)
                })
        
        # Sort by intensity and return top 5
        peaks.sort(key=lambda p: p['intensity'], reverse=True)
        return peaks[:5]
    
    def _calculate_density_progression(self, entries: List[Any], start: float, end: float) -> List[int]:
        """Calculate information density per second."""
        duration = int(end - start)
        density = [0] * duration
        
        for i in range(duration):
            second_start = start + i
            second_end = start + i + 1
            
            # Count events in this second
            count = 0
            for entry in entries:
                if entry.start.seconds >= second_start and entry.start.seconds < second_end:
                    count += 1
                    # Weight different event types
                    if entry.entry_type == 'text':
                        count += 2  # Text is more important
                    elif entry.entry_type == 'speech':
                        count += 1
            
            density[i] = min(count, 10)  # Cap at 10 for normalization
        
        return density
    
    def _extract_text_moments(self, entries: List[Any]) -> List[Dict[str, Any]]:
        """Extract text overlay moments."""
        text_moments = []
        
        for entry in entries:
            if entry.entry_type == 'text':
                text_moments.append({
                    'timestamp': entry.start.to_string(),
                    'text': entry.data.get('text', ''),
                    'duration': entry.duration if entry.end else 1.0,
                    'position': entry.data.get('position', 'unknown')
                })
        
        return text_moments[:10]  # Limit to 10 most important
    
    def _extract_emotion_sequence(self, analysis: UnifiedAnalysis, start: float, end: float) -> List[str]:
        """Extract emotion sequence from speech and visual cues."""
        emotions = []
        duration = int(end - start)
        
        # Simple emotion detection based on speech content
        speech_entries = [e for e in analysis.timeline.get_entries_in_range(start, end) 
                         if e.entry_type == 'speech']
        
        for i in range(duration):
            second_time = start + i
            emotion = 'neutral'  # Default
            
            # Check speech at this time
            for entry in speech_entries:
                if entry.contains_time(second_time):
                    text = entry.data.get('text', '').lower()
                    # Simple keyword-based emotion detection
                    if any(word in text for word in ['amazing', 'love', 'great', 'awesome']):
                        emotion = 'positive'
                    elif any(word in text for word in ['warning', 'careful', 'don\'t', 'avoid']):
                        emotion = 'warning'
                    elif any(word in text for word in ['urgent', 'now', 'quick', 'fast']):
                        emotion = 'urgent'
                    break
            
            emotions.append(emotion)
        
        return emotions
    
    def _extract_gesture_moments(self, entries: List[Any]) -> List[Dict[str, Any]]:
        """Extract gesture/movement moments."""
        gesture_moments = []
        
        for entry in entries:
            if entry.entry_type in ['gesture', 'movement', 'action']:
                gesture_moments.append({
                    'timestamp': entry.start.to_string(),
                    'type': entry.data.get('action', 'movement'),
                    'confidence': entry.data.get('confidence', 0.5)
                })
        
        return gesture_moments[:5]
    
    def _extract_object_appearances(self, entries: List[Any]) -> List[Dict[str, Any]]:
        """Extract object appearance moments."""
        objects = {}
        
        for entry in entries:
            if entry.entry_type == 'object':
                obj_class = entry.data.get('class', 'unknown')
                if obj_class not in objects:
                    objects[obj_class] = {
                        'class': obj_class,
                        'first_appearance': entry.start.to_string(),
                        'count': 0
                    }
                objects[obj_class]['count'] += 1
        
        # Return top 5 most frequent objects
        return sorted(objects.values(), key=lambda x: x['count'], reverse=True)[:5]
    
    def _extract_speech_segments(self, entries: List[Any]) -> List[Dict[str, Any]]:
        """Extract speech segments."""
        segments = []
        
        for entry in entries:
            if entry.entry_type == 'speech':
                segments.append({
                    'start': entry.start.to_string(),
                    'end': entry.end.to_string() if entry.end else entry.start.to_string(),
                    'text': entry.data.get('text', '')[:50] + '...' if len(entry.data.get('text', '')) > 50 else entry.data.get('text', '')
                })
        
        return segments[:5]
    
    def _extract_scene_changes(self, analysis: UnifiedAnalysis, start: float, end: float) -> List[float]:
        """Extract scene change timestamps."""
        scene_entries = [e for e in analysis.timeline.get_entries_in_range(start, end)
                        if e.entry_type == 'scene_change']
        
        return [entry.start.seconds for entry in scene_entries]
    
    def _find_cta_keywords(self, entries: List[Any]) -> List[Dict[str, Any]]:
        """Find call-to-action keywords in text and speech."""
        cta_keywords = ['follow', 'like', 'subscribe', 'comment', 'share', 'link', 'bio', 
                       'check out', 'click', 'tap', 'swipe', 'buy', 'shop', 'order']
        
        cta_appearances = []
        
        for entry in entries:
            if entry.entry_type in ['text', 'speech']:
                text = entry.data.get('text', '').lower()
                for keyword in cta_keywords:
                    if keyword in text:
                        cta_appearances.append({
                            'timestamp': entry.start.to_string(),
                            'keyword': keyword,
                            'context': text[:100],
                            'type': entry.entry_type
                        })
        
        return cta_appearances
    
    def _analyze_gesture_sync(self, entries: List[Any]) -> Dict[str, Any]:
        """Analyze gesture synchronization with speech."""
        gesture_count = sum(1 for e in entries if e.entry_type in ['gesture', 'movement'])
        speech_count = sum(1 for e in entries if e.entry_type == 'speech')
        
        return {
            'gesture_count': gesture_count,
            'speech_count': speech_count,
            'sync_ratio': gesture_count / speech_count if speech_count > 0 else 0
        }
    
    def _analyze_object_focus(self, entries: List[Any]) -> List[str]:
        """Analyze which objects are focused on."""
        object_counts = {}
        
        for entry in entries:
            if entry.entry_type == 'object':
                obj_class = entry.data.get('class', 'unknown')
                object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
        
        # Return top 3 objects
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        return [obj[0] for obj in sorted_objects[:3]]
    
    def _analyze_speech_emphasis(self, entries: List[Any]) -> Dict[str, Any]:
        """Analyze speech emphasis patterns."""
        speech_entries = [e for e in entries if e.entry_type == 'speech']
        
        total_duration = sum(e.duration for e in speech_entries)
        word_count = sum(len(e.data.get('text', '').split()) for e in speech_entries)
        
        return {
            'speech_duration': total_duration,
            'word_count': word_count,
            'words_per_second': word_count / total_duration if total_duration > 0 else 0,
            'segment_count': len(speech_entries)
        }
    
    def _calculate_activity_density(self, analysis: UnifiedAnalysis) -> List[float]:
        """Calculate activity density per second across entire video."""
        duration = int(analysis.timeline.duration)
        density = [0.0] * duration
        
        for entry in analysis.timeline.entries:
            second = int(entry.start.seconds)
            if second < duration:
                # Weight different activities
                weight = 1.0
                if entry.entry_type == 'text':
                    weight = 3.0
                elif entry.entry_type == 'scene_change':
                    weight = 2.0
                elif entry.entry_type == 'speech':
                    weight = 1.5
                
                density[second] += weight
        
        return density
    
    def _get_events_at_time(self, analysis: UnifiedAnalysis, timestamp: float) -> List[str]:
        """Get all events happening at a specific timestamp."""
        events = []
        entries = analysis.timeline.get_entries_at_time(timestamp)
        
        for entry in entries:
            event_desc = f"{entry.entry_type}"
            if entry.entry_type == 'text':
                event_desc += f": {entry.data.get('text', '')[:30]}"
            elif entry.entry_type == 'object':
                event_desc += f": {entry.data.get('class', 'unknown')}"
            
            events.append(event_desc)
        
        return events
    
    def _calculate_engagement_curve(self, analysis: UnifiedAnalysis) -> List[float]:
        """Calculate engagement curve across video."""
        # Sample every 5 seconds
        sample_rate = 5
        num_samples = int(analysis.timeline.duration / sample_rate) + 1
        curve = []
        
        for i in range(num_samples):
            time = i * sample_rate
            end_time = min(time + sample_rate, analysis.timeline.duration)
            
            # Calculate engagement score for this window
            entries = analysis.timeline.get_entries_in_range(time, end_time)
            
            score = 0.0
            # Text overlays are highly engaging
            score += sum(2.0 for e in entries if e.entry_type == 'text')
            # Scene changes keep attention
            score += sum(1.5 for e in entries if e.entry_type == 'scene_change')
            # Speech provides information
            score += sum(1.0 for e in entries if e.entry_type == 'speech')
            # Objects add visual interest
            score += sum(0.5 for e in entries if e.entry_type == 'object')
            
            # Normalize to 0-1 range
            normalized_score = min(score / 10.0, 1.0)
            curve.append(round(normalized_score, 2))
        
        return curve
    
    def _get_empty_markers(self, video_id: str, error: str = None) -> Dict[str, Any]:
        """Return empty but valid marker structure."""
        return {
            'first_5_seconds': {
                'density_progression': [0, 0, 0, 0, 0],
                'text_moments': [],
                'emotion_sequence': ['neutral'] * 5,
                'gesture_moments': [],
                'object_appearances': [],
                'speech_segments': [],
                'scene_changes': []
            },
            'cta_window': {
                'time_range': 'last 15%',
                'cta_appearances': [],
                'gesture_sync': {'gesture_count': 0, 'speech_count': 0, 'sync_ratio': 0},
                'object_focus': [],
                'speech_emphasis': {'speech_duration': 0, 'word_count': 0, 'words_per_second': 0, 'segment_count': 0},
                'text_overlays': []
            },
            'peak_moments': [],
            'engagement_curve': [0.0],
            'metadata': {
                'video_id': video_id,
                'error': error if error else 'Generation failed - empty markers returned',
                'generated_at': datetime.utcnow().isoformat()
            }
        }


def main():
    """
    Main entry point for command-line execution.
    
    For testing and standalone execution only.
    Production uses direct function calls from rumiai_runner.py.
    """
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Generate temporal markers for video')
    parser.add_argument('--video-path', required=True, help='Path to video file')
    parser.add_argument('--video-id', required=True, help='Video ID')
    parser.add_argument('--deps', required=True, help='JSON string of dependency paths')
    
    args = parser.parse_args()
    
    try:
        # Parse dependencies
        deps = json.loads(args.deps)
        
        # Load ML results
        ml_results = {}
        
        # Load YOLO data
        if 'yolo' in deps and Path(deps['yolo']).exists():
            with open(deps['yolo'], 'r') as f:
                yolo_data = json.load(f)
                ml_results['yolo'] = MLAnalysisResult(
                    model_name='yolo',
                    model_version='v8',
                    success=True,
                    data=MLDataValidator.validate_yolo_data(yolo_data, args.video_id)
                )
        
        # Load other ML results similarly...
        # (In production, this would load all ML results)
        
        # Create unified analysis
        from ..core.models import Timeline, UnifiedAnalysis
        
        # Get video duration (simplified - in production would read from video file)
        duration = 60.0  # Default
        
        timeline = Timeline(video_id=args.video_id, duration=duration)
        analysis = UnifiedAnalysis(
            video_id=args.video_id,
            video_metadata={'duration': duration},
            timeline=timeline,
            ml_results=ml_results
        )
        
        # Generate temporal markers
        processor = TemporalMarkerProcessor()
        markers = processor.generate_markers(analysis)
        
        # Output JSON to stdout
        print(json.dumps(markers, indent=2))
        
    except Exception as e:
        # Log error to stderr
        print(f"Error: {str(e)}", file=sys.stderr)
        
        # Always output valid JSON to stdout
        empty_markers = {
            'first_5_seconds': {
                'density_progression': [0, 0, 0, 0, 0],
                'text_moments': [],
                'emotion_sequence': ['neutral'] * 5,
                'gesture_moments': [],
                'object_appearances': [],
                'speech_segments': [],
                'scene_changes': []
            },
            'cta_window': {
                'time_range': 'last 15%',
                'cta_appearances': [],
                'gesture_sync': {'gesture_count': 0, 'speech_count': 0, 'sync_ratio': 0},
                'object_focus': [],
                'speech_emphasis': {'speech_duration': 0, 'word_count': 0, 'words_per_second': 0, 'segment_count': 0},
                'text_overlays': []
            },
            'peak_moments': [],
            'engagement_curve': [0.0],
            'metadata': {
                'video_id': args.video_id,
                'error': str(e),
                'generated_at': datetime.utcnow().isoformat()
            }
        }
        print(json.dumps(empty_markers, indent=2))
        sys.exit(0)  # Exit cleanly even on error


if __name__ == '__main__':
    main()