"""
Timeline builder for RumiAI v2.

This module builds the unified timeline from ML analysis results.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import json

from ..core.models import Timeline, TimelineEntry, Timestamp, UnifiedAnalysis, MLAnalysisResult
from ..core.validators import MLDataValidator, TimestampValidator
from ..core.exceptions import ValidationError, TimelineError, handle_error

logger = logging.getLogger(__name__)


class TimelineBuilder:
    """Build unified timeline from ML analysis results."""
    
    def __init__(self):
        self.ml_validator = MLDataValidator()
        self.ts_validator = TimestampValidator()
    
    def build_timeline(self, video_id: str, video_metadata: Dict[str, Any], 
                      ml_results: Dict[str, MLAnalysisResult]) -> UnifiedAnalysis:
        """
        Build complete unified analysis with timeline.
        
        CRITICAL: This combines all ML results into a single timeline.
        Must handle missing/failed analyses gracefully.
        """
        logger.info(f"Building timeline for video {video_id}")
        
        # Extract video duration
        duration = video_metadata.get('duration', 0)
        if duration <= 0:
            logger.warning(f"Invalid duration {duration} for video {video_id}, using default 60s")
            duration = 60  # Default duration
        
        # Create timeline
        timeline = Timeline(video_id=video_id, duration=float(duration))
        
        # Add entries from each ML model
        builders = {
            'yolo': self._add_yolo_entries,
            'whisper': self._add_whisper_entries,
            'ocr': self._add_ocr_entries,
            'mediapipe': self._add_mediapipe_entries,
            'scene_detection': self._add_scene_entries
        }
        
        for model_name, builder_func in builders.items():
            if model_name in ml_results and ml_results[model_name].success:
                try:
                    logger.info(f"Adding {model_name} entries to timeline")
                    builder_func(timeline, ml_results[model_name].data)
                    logger.info(f"Successfully added {model_name} entries")
                except Exception as e:
                    logger.error(f"Failed to add {model_name} entries: {str(e)}")
                    # Continue with other models even if one fails
            else:
                logger.warning(f"Skipping {model_name} - not available or failed")
        
        # Log timeline statistics
        logger.info(f"Timeline built with {len(timeline.entries)} entries")
        entry_types = {}
        for entry in timeline.entries:
            entry_types[entry.entry_type] = entry_types.get(entry.entry_type, 0) + 1
        logger.info(f"Entry types: {entry_types}")
        
        # Create unified analysis
        analysis = UnifiedAnalysis(
            video_id=video_id,
            video_metadata=video_metadata,
            timeline=timeline,
            ml_results=ml_results,
            processing_metadata={
                'timeline_entries': len(timeline.entries),
                'entry_types': entry_types,
                'duration': duration
            }
        )
        
        return analysis
    
    def _add_yolo_entries(self, timeline: Timeline, yolo_data: Dict[str, Any]) -> None:
        """Add YOLO object detection entries."""
        # Validate and normalize data
        yolo_data = self.ml_validator.validate_yolo_data(yolo_data, timeline.video_id)
        
        # Track objects across frames
        object_tracks = {}
        
        for annotation in yolo_data.get('objectAnnotations', []):
            obj_class = annotation.get('class', 'unknown')
            
            for frame_data in annotation.get('frames', []):
                # Parse timestamp
                timestamp = self.ts_validator.validate_timestamp(
                    frame_data.get('timestamp'), 
                    f"YOLO {obj_class} timestamp"
                )
                
                if not timestamp:
                    continue
                
                # Create entry
                entry = TimelineEntry(
                    start=timestamp,
                    end=None,  # Object detection is instantaneous
                    entry_type='object',
                    data={
                        'class': obj_class,
                        'confidence': frame_data.get('confidence', 0),
                        'bbox': frame_data.get('bbox', []),
                        'track_id': annotation.get('track_id', None)
                    }
                )
                
                timeline.add_entry(entry)
                
                # Track object appearances
                if obj_class not in object_tracks:
                    object_tracks[obj_class] = []
                object_tracks[obj_class].append(timestamp.seconds)
        
        # Log object statistics
        for obj_class, timestamps in object_tracks.items():
            logger.info(f"YOLO: {obj_class} detected {len(timestamps)} times")
    
    def _add_whisper_entries(self, timeline: Timeline, whisper_data: Dict[str, Any]) -> None:
        """Add Whisper transcription entries."""
        # Validate and normalize data
        whisper_data = self.ml_validator.validate_whisper_data(whisper_data, timeline.video_id)
        
        total_segments = 0
        total_duration = 0
        
        for segment in whisper_data.get('segments', []):
            # Parse timestamps
            start_ts = self.ts_validator.validate_timestamp(
                segment.get('start'), 
                "Whisper segment start"
            )
            end_ts = self.ts_validator.validate_timestamp(
                segment.get('end'), 
                "Whisper segment end"
            )
            
            if not start_ts:
                continue
            
            # If end is missing, estimate based on text length
            if not end_ts:
                text_length = len(segment.get('text', '').split())
                estimated_duration = text_length * 0.3  # ~0.3 seconds per word
                end_ts = Timestamp(start_ts.seconds + estimated_duration)
            
            # Create entry
            entry = TimelineEntry(
                start=start_ts,
                end=end_ts,
                entry_type='speech',
                data={
                    'text': segment.get('text', ''),
                    'confidence': segment.get('confidence', 0),
                    'language': whisper_data.get('language', 'unknown'),
                    'words': segment.get('words', [])  # Word-level timestamps if available
                }
            )
            
            timeline.add_entry(entry)
            total_segments += 1
            total_duration += end_ts.seconds - start_ts.seconds
        
        logger.info(f"Whisper: Added {total_segments} segments, {total_duration:.1f}s total speech")
    
    def _add_ocr_entries(self, timeline: Timeline, ocr_data: Dict[str, Any]) -> None:
        """Add OCR text detection entries."""
        # Validate and normalize data
        ocr_data = self.ml_validator.validate_ocr_data(ocr_data, timeline.video_id)
        
        text_count = 0
        sticker_count = 0
        
        # Add text annotations
        for text_annotation in ocr_data.get('textAnnotations', []):
            # Parse timestamp
            timestamp = self._extract_timestamp_from_annotation(text_annotation)
            if not timestamp:
                continue
            
            # Estimate duration based on text length
            text = text_annotation.get('text', '')
            duration = max(1.0, len(text) * 0.1)  # At least 1 second, 0.1s per character
            
            entry = TimelineEntry(
                start=timestamp,
                end=Timestamp(timestamp.seconds + duration),
                entry_type='text',
                data={
                    'text': text,
                    'position': self._extract_position(text_annotation),
                    'size': text_annotation.get('size', 'medium'),
                    'style': text_annotation.get('style', 'normal')
                }
            )
            
            timeline.add_entry(entry)
            text_count += 1
        
        # Add stickers
        for sticker in ocr_data.get('stickers', []):
            timestamp = self._extract_timestamp_from_annotation(sticker)
            if not timestamp:
                continue
            
            entry = TimelineEntry(
                start=timestamp,
                end=Timestamp(timestamp.seconds + 2.0),  # Stickers typically last 2 seconds
                entry_type='sticker',
                data={
                    'type': sticker.get('type', 'emoji'),
                    'value': sticker.get('value', ''),
                    'position': self._extract_position(sticker)
                }
            )
            
            timeline.add_entry(entry)
            sticker_count += 1
        
        logger.info(f"OCR: Added {text_count} text overlays and {sticker_count} stickers")
    
    def _add_mediapipe_entries(self, timeline: Timeline, mediapipe_data: Dict[str, Any]) -> None:
        """Add MediaPipe human analysis entries."""
        # Validate and normalize data
        mediapipe_data = self.ml_validator.validate_mediapipe_data(mediapipe_data, timeline.video_id)
        
        # Add pose data if available
        poses = mediapipe_data.get('poses', [])
        for pose in poses:
            timestamp = self._extract_timestamp_from_annotation(pose)
            if not timestamp:
                continue
            
            entry = TimelineEntry(
                start=timestamp,
                end=None,
                entry_type='pose',
                data={
                    'landmarks': pose.get('landmarks', []),
                    'action': pose.get('action', 'unknown'),
                    'confidence': pose.get('confidence', 0)
                }
            )
            
            timeline.add_entry(entry)
        
        # Add face data if available
        faces = mediapipe_data.get('faces', [])
        for face in faces:
            timestamp = self._extract_timestamp_from_annotation(face)
            if not timestamp:
                continue
            
            entry = TimelineEntry(
                start=timestamp,
                end=None,
                entry_type='face',
                data={
                    'landmarks': face.get('landmarks', []),
                    'emotion': face.get('emotion', 'neutral'),
                    'gaze_direction': face.get('gaze_direction', 'unknown')
                }
            )
            
            timeline.add_entry(entry)
        
        # Add gesture data if available
        gestures = mediapipe_data.get('gestures', [])
        for gesture in gestures:
            timestamp = self._extract_timestamp_from_annotation(gesture)
            if not timestamp:
                continue
            
            entry = TimelineEntry(
                start=timestamp,
                end=Timestamp(timestamp.seconds + 0.5),  # Gestures are brief
                entry_type='gesture',
                data={
                    'type': gesture.get('type', 'unknown'),
                    'hand': gesture.get('hand', 'unknown'),
                    'confidence': gesture.get('confidence', 0)
                }
            )
            
            timeline.add_entry(entry)
        
        logger.info(f"MediaPipe: Added {len(poses)} poses, {len(faces)} faces, {len(gestures)} gestures")
    
    def _add_scene_entries(self, timeline: Timeline, scene_data: Dict[str, Any]) -> None:
        """Add scene detection entries."""
        # Validate and normalize data
        scene_data = self.ml_validator.validate_scene_detection_data(scene_data, timeline.video_id)
        
        scene_count = 0
        
        # Add scene changes as instantaneous events
        for scene_change in scene_data.get('scene_changes', []):
            timestamp = self.ts_validator.validate_timestamp(scene_change, "Scene change")
            if not timestamp:
                continue
            
            entry = TimelineEntry(
                start=timestamp,
                end=None,  # Scene changes are instantaneous
                entry_type='scene_change',
                data={
                    'transition_type': 'cut',  # Could be enhanced to detect different types
                    'scene_index': scene_count
                }
            )
            
            timeline.add_entry(entry)
            scene_count += 1
        
        # Also add scene segments if available
        for i, scene in enumerate(scene_data.get('scenes', [])):
            if not isinstance(scene, dict):
                continue
            
            start_ts = self.ts_validator.validate_timestamp(
                scene.get('start_time'), 
                f"Scene {i} start"
            )
            end_ts = self.ts_validator.validate_timestamp(
                scene.get('end_time'), 
                f"Scene {i} end"
            )
            
            if start_ts and end_ts:
                entry = TimelineEntry(
                    start=start_ts,
                    end=end_ts,
                    entry_type='scene',
                    data={
                        'scene_index': i,
                        'duration': scene.get('duration', end_ts.seconds - start_ts.seconds)
                    }
                )
                
                timeline.add_entry(entry)
        
        logger.info(f"Scene Detection: Added {scene_count} scene changes")
    
    def _extract_timestamp_from_annotation(self, annotation: Dict[str, Any]) -> Optional[Timestamp]:
        """Extract timestamp from various annotation formats."""
        # Try different timestamp fields
        timestamp_fields = ['timestamp', 'time', 'start_time', 'frame_time', 't']
        
        for field in timestamp_fields:
            if field in annotation:
                ts = self.ts_validator.validate_timestamp(annotation[field])
                if ts:
                    return ts
        
        # Try frame number with FPS
        if 'frame' in annotation or 'frame_number' in annotation:
            frame = annotation.get('frame', annotation.get('frame_number'))
            fps = annotation.get('fps', 30)  # Default 30 FPS
            try:
                timestamp_seconds = float(frame) / float(fps)
                return Timestamp(timestamp_seconds)
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _extract_position(self, annotation: Dict[str, Any]) -> str:
        """Extract position information from annotation."""
        # Try different position representations
        if 'position' in annotation:
            return annotation['position']
        
        if 'bbox' in annotation:
            bbox = annotation['bbox']
            if isinstance(bbox, list) and len(bbox) >= 2:
                x = bbox[0]
                if x < 0.33:
                    return 'left'
                elif x > 0.67:
                    return 'right'
                else:
                    return 'center'
        
        if 'x' in annotation and 'y' in annotation:
            x = annotation['x']
            y = annotation['y']
            
            if x < 0.33:
                h_pos = 'left'
            elif x > 0.67:
                h_pos = 'right'
            else:
                h_pos = 'center'
            
            if y < 0.33:
                v_pos = 'top'
            elif y > 0.67:
                v_pos = 'bottom'
            else:
                v_pos = 'middle'
            
            return f"{v_pos}-{h_pos}"
        
        return 'center'  # Default
    
    def merge_timelines(self, primary: UnifiedAnalysis, *others: UnifiedAnalysis) -> UnifiedAnalysis:
        """
        Merge multiple unified analyses into one.
        
        Used when combining results from different processing runs.
        """
        # Start with primary analysis
        merged_timeline = Timeline(
            video_id=primary.video_id,
            duration=primary.timeline.duration
        )
        
        # Add all entries from primary
        for entry in primary.timeline.entries:
            merged_timeline.add_entry(entry)
        
        # Merge ML results
        merged_ml_results = primary.ml_results.copy()
        
        # Add entries from other analyses
        for other in others:
            # Verify same video
            if other.video_id != primary.video_id:
                logger.warning(f"Skipping merge of different video: {other.video_id} != {primary.video_id}")
                continue
            
            # Add timeline entries
            for entry in other.timeline.entries:
                merged_timeline.add_entry(entry)
            
            # Merge ML results (prefer successful results)
            for model_name, result in other.ml_results.items():
                if model_name not in merged_ml_results or not merged_ml_results[model_name].success:
                    merged_ml_results[model_name] = result
        
        # Create merged analysis
        merged = UnifiedAnalysis(
            video_id=primary.video_id,
            video_metadata=primary.video_metadata,
            timeline=merged_timeline,
            ml_results=merged_ml_results,
            temporal_markers=primary.temporal_markers,
            processing_metadata={
                **primary.processing_metadata,
                'merged': True,
                'merge_sources': len(others) + 1
            }
        )
        
        return merged