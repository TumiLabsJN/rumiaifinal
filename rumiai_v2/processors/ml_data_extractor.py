"""
ML data extractor for RumiAI v2.

This module extracts relevant ML data for each Claude prompt type.
"""
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

from ..core.models import UnifiedAnalysis, Timeline, TimelineEntry, PromptType, PromptContext
from ..core.exceptions import ValidationError, handle_error

logger = logging.getLogger(__name__)


class MLDataExtractor:
    """Extract relevant ML data for each prompt type."""
    
    def extract_for_prompt(self, analysis: UnifiedAnalysis, prompt_type: PromptType) -> PromptContext:
        """
        Extract ML data specific to prompt type.
        
        CRITICAL: This data is sent to Claude API. Size matters for cost and latency.
        """
        # Base context all prompts need
        base_context = {
            'video_id': analysis.video_id,
            'duration': analysis.timeline.duration,
            'metadata': self._extract_video_metadata(analysis)
        }
        
        # Prompt-specific extractors
        extractors = {
            PromptType.CREATIVE_DENSITY: self._extract_creative_density,
            PromptType.EMOTIONAL_JOURNEY: self._extract_emotional_journey,
            PromptType.SPEECH_ANALYSIS: self._extract_speech_analysis,
            PromptType.VISUAL_OVERLAY: self._extract_visual_overlay,
            PromptType.METADATA_ANALYSIS: self._extract_metadata_analysis,
            PromptType.PERSON_FRAMING: self._extract_person_framing,
            PromptType.SCENE_PACING: self._extract_scene_pacing
        }
        
        if prompt_type not in extractors:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Extract prompt-specific data
        prompt_data = extractors[prompt_type](analysis)
        
        # Create prompt context
        context = PromptContext(
            video_id=analysis.video_id,
            prompt_type=prompt_type,
            duration=analysis.timeline.duration,
            metadata=base_context['metadata'],
            ml_data=prompt_data.get('ml_data', {}),
            timelines=prompt_data.get('timelines', {}),
            temporal_markers=analysis.temporal_markers
        )
        
        # Log context size
        size_kb = context.get_size_bytes() / 1024
        logger.info(f"Extracted context for {prompt_type.value}: {size_kb:.1f}KB")
        
        return context
    
    def _extract_video_metadata(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """Extract relevant video metadata."""
        metadata = analysis.video_metadata.copy()
        
        # Remove large fields not needed for prompts
        fields_to_remove = ['downloadUrl', 'coverUrl', 'music']
        for field in fields_to_remove:
            metadata.pop(field, None)
        
        # Add computed fields
        if 'views' in metadata and metadata['views'] > 0:
            metadata['engagement_metrics'] = {
                'like_rate': metadata.get('likes', 0) / metadata['views'],
                'comment_rate': metadata.get('comments', 0) / metadata['views'],
                'share_rate': metadata.get('shares', 0) / metadata['views'],
                'save_rate': metadata.get('saves', 0) / metadata['views']
            }
        
        return metadata
    
    def _extract_creative_density(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """Extract data for creative density analysis."""
        ocr_data = analysis.get_ml_data('ocr') or {}
        
        # Extract text timeline
        text_timeline = {}
        text_entries = analysis.timeline.get_entries_by_type('text')
        
        for entry in text_entries:
            timestamp_key = entry.start.to_string()
            if timestamp_key not in text_timeline:
                text_timeline[timestamp_key] = []
            text_timeline[timestamp_key].append({
                'text': entry.data.get('text', ''),
                'position': entry.data.get('position', 'center'),
                'size': entry.data.get('size', 'medium')
            })
        
        # Extract sticker timeline
        sticker_timeline = {}
        stickers = ocr_data.get('stickers', [])
        
        for sticker in stickers:
            if 'timestamp' in sticker:
                ts = entry.start.to_string()
                sticker_timeline[ts] = sticker
        
        # Calculate density metrics
        density_buckets = analysis.timeline.get_density_buckets(bucket_size=5.0)
        
        return {
            'ml_data': {
                'total_text_elements': len(text_entries),
                'total_stickers': len(stickers),
                'unique_text_positions': len(set(e.data.get('position', 'center') for e in text_entries))
            },
            'timelines': {
                'text_timeline': text_timeline,
                'sticker_timeline': sticker_timeline,
                'density_buckets': density_buckets
            }
        }
    
    def _extract_emotional_journey(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """Extract data for emotional journey analysis."""
        # Get speech data for sentiment
        speech_entries = analysis.timeline.get_entries_by_type('speech')
        
        # Create emotional timeline
        emotional_timeline = {}
        
        # Sample every 2 seconds
        sample_rate = 2.0
        num_samples = int(analysis.timeline.duration / sample_rate) + 1
        
        for i in range(num_samples):
            time = i * sample_rate
            timestamp_key = f"{int(time)}s"
            
            # Get all events at this time
            entries = analysis.timeline.get_entries_at_time(time)
            
            emotional_timeline[timestamp_key] = {
                'speech_active': any(e.entry_type == 'speech' for e in entries),
                'text_count': sum(1 for e in entries if e.entry_type == 'text'),
                'scene_change': any(e.entry_type == 'scene_change' for e in entries),
                'object_count': sum(1 for e in entries if e.entry_type == 'object'),
                'events': [e.entry_type for e in entries]
            }
        
        # Extract key moments
        scene_changes = analysis.timeline.get_entries_by_type('scene_change')
        
        return {
            'ml_data': {
                'total_speech_segments': len(speech_entries),
                'total_scene_changes': len(scene_changes),
                'average_scene_duration': analysis.timeline.duration / (len(scene_changes) + 1)
            },
            'timelines': {
                'emotional_timeline': emotional_timeline,
                'scene_changes': [sc.start.seconds for sc in scene_changes],
                'speech_density': self._calculate_speech_density(analysis)
            }
        }
    
    def _extract_speech_analysis(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """Extract data for speech analysis."""
        import os
        
        whisper_data = analysis.get_ml_data('whisper') or {}
        audio_energy_data = analysis.get_ml_data('audio_energy') or {}  # ADD AUDIO ENERGY
        speech_entries = analysis.timeline.get_entries_by_type('speech')
        
        # Build speech segments with full text
        speech_segments = []
        for entry in speech_entries:
            segment = {
                'start': entry.start.seconds,
                'end': entry.end.seconds if entry.end else entry.start.seconds + 1,
                'text': entry.data.get('text', ''),
                'confidence': entry.data.get('confidence', 0.0)
            }
            speech_segments.append(segment)
        
        # Calculate speech metrics
        total_speech_time = sum(s['end'] - s['start'] for s in speech_segments)
        speech_percentage = (total_speech_time / analysis.timeline.duration) * 100 if analysis.timeline.duration > 0 else 0
        
        # Word frequency analysis
        all_text = ' '.join(s['text'] for s in speech_segments).lower()
        words = all_text.split()
        word_count = len(words)
        
        # Simple word frequency (top 20)
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # ERROR HANDLING: Check for speech + missing audio energy (FAIL FAST)
        has_speech = bool(speech_entries) or bool(whisper_data.get('segments', []))
        has_audio_energy = bool(audio_energy_data.get('energy_level_windows', {}))
        
        if os.getenv('USE_PYTHON_ONLY_PROCESSING') == 'true':
            if has_speech and not has_audio_energy:
                # FAIL FAST: Speech detected but no audio energy data
                raise RuntimeError(
                    "CRITICAL: Speech detected but audio energy data missing. "
                    "Audio energy analysis is required for speech analysis in Python-only mode."
                )
        
        # BACKWARD COMPATIBILITY: No audio = continue with empty energy data
        if not has_audio_energy:
            audio_energy_data = {
                'energy_level_windows': {},
                'energy_variance': 0.0,
                'climax_timestamp': 0.0,
                'burst_pattern': 'none',
                'metadata': {'processed': True, 'success': True, 'no_audio': True}
            }
        
        return {
            'ml_data': {
                'total_segments': len(speech_segments),
                'total_speech_time': total_speech_time,
                'speech_percentage': speech_percentage,
                'word_count': word_count,
                'words_per_minute': (word_count / total_speech_time * 60) if total_speech_time > 0 else 0,
                'language': whisper_data.get('language', 'unknown'),
                'top_words': dict(top_words),
                'audio_energy': audio_energy_data  # ADD AUDIO ENERGY DATA
            },
            'timelines': {
                'speech_segments': speech_segments[:50],  # Limit to 50 segments
                'speech_gaps': self._find_speech_gaps(speech_segments, analysis.timeline.duration)
            }
        }
    
    def _extract_visual_overlay(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """
        Extract data for visual overlay analysis.
        
        CRITICAL: This prompt often causes crashes due to data size and format issues.
        """
        # Get all visual elements
        text_entries = analysis.timeline.get_entries_by_type('text')
        object_entries = analysis.timeline.get_entries_by_type('object')
        
        # Create visual timeline with controlled size
        visual_timeline = {}
        
        # Sample every 0.5 seconds to control data size
        sample_rate = 0.5
        num_samples = min(int(analysis.timeline.duration / sample_rate) + 1, 200)  # Cap at 200 samples
        
        for i in range(num_samples):
            time = i * sample_rate
            timestamp_key = f"{time:.1f}s"
            
            # Get visual elements at this time
            current_entries = analysis.timeline.get_entries_at_time(time)
            
            visual_elements = {
                'text': [e.data.get('text', '')[:50] for e in current_entries if e.entry_type == 'text'][:3],  # Limit text
                'objects': [e.data.get('class', '') for e in current_entries if e.entry_type == 'object'][:5],  # Limit objects
                'has_person': any(e.data.get('class', '') in ['person', 'face'] for e in current_entries if e.entry_type == 'object')
            }
            
            if visual_elements['text'] or visual_elements['objects']:
                visual_timeline[timestamp_key] = visual_elements
        
        # Calculate overlay density
        overlay_density = []
        for i in range(0, int(analysis.timeline.duration), 5):
            entries_in_window = analysis.timeline.get_entries_in_range(i, i + 5)
            text_count = sum(1 for e in entries_in_window if e.entry_type == 'text')
            overlay_density.append(min(text_count / 5, 1.0))  # Normalize to 0-1
        
        return {
            'ml_data': {
                'total_text_overlays': len(text_entries),
                'total_objects_detected': len(object_entries),
                'unique_object_classes': len(set(e.data.get('class', '') for e in object_entries)),
                'average_text_per_second': len(text_entries) / analysis.timeline.duration if analysis.timeline.duration > 0 else 0
            },
            'timelines': {
                'visual_timeline': visual_timeline,  # Already size-limited
                'overlay_density': overlay_density,
                'text_appearance_times': [e.start.seconds for e in text_entries][:50]  # Limit to 50
            }
        }
    
    def _extract_metadata_analysis(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """Extract data for metadata analysis (hashtags, caption, etc)."""
        metadata = analysis.video_metadata
        
        # Extract hashtags
        hashtags = metadata.get('hashtags', [])
        hashtag_names = [h.get('name', '') for h in hashtags if isinstance(h, dict)]
        
        # Extract caption/description
        description = metadata.get('description', '')
        
        # Simple keyword extraction from description
        desc_words = description.lower().split()
        desc_keywords = [w for w in desc_words if len(w) > 4 and not w.startswith('#')]
        
        # Music analysis
        music = metadata.get('music', {})
        
        return {
            'ml_data': {
                'hashtag_count': len(hashtags),
                'hashtag_names': hashtag_names,
                'description_length': len(description),
                'description_word_count': len(desc_words),
                'has_music': bool(music),
                'music_original': music.get('musicOriginal', False),
                'author_verified': metadata.get('author', {}).get('verified', False)
            },
            'timelines': {
                # Metadata doesn't have timeline data, but we can correlate with content
                'content_metadata_alignment': self._analyze_content_metadata_alignment(analysis, hashtag_names, desc_keywords)
            }
        }
    
    def _extract_person_framing(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """Extract data for person framing analysis."""
        mediapipe_data = analysis.get_ml_data('mediapipe') or {}
        
        # Get person/face detections from YOLO
        object_entries = analysis.timeline.get_entries_by_type('object')
        person_entries = [e for e in object_entries if e.data.get('class', '') in ['person', 'face']]
        
        # Calculate person presence timeline
        person_timeline = {}
        
        # Sample every second
        for i in range(int(analysis.timeline.duration)):
            timestamp_key = f"{i}s"
            entries_at_time = [e for e in person_entries if e.contains_time(float(i))]
            
            if entries_at_time:
                person_timeline[timestamp_key] = {
                    'person_count': len(entries_at_time),
                    'positions': [self._get_position_description(e.data.get('bbox', [])) for e in entries_at_time]
                }
        
        # Calculate presence metrics
        frames_with_person = len(person_timeline)
        presence_percentage = (frames_with_person / analysis.timeline.duration * 100) if analysis.timeline.duration > 0 else 0
        
        return {
            'ml_data': {
                'presence_percentage': mediapipe_data.get('presence_percentage', presence_percentage),
                'frames_with_people': mediapipe_data.get('frames_with_people', frames_with_person),
                'face_screen_time': mediapipe_data.get('face_screen_time', 0),
                'eye_contact_ratio': mediapipe_data.get('eye_contact_ratio', 0),
                'primary_actions': mediapipe_data.get('primary_actions', [])
            },
            'timelines': {
                'person_timeline': person_timeline,
                'person_appearance_pattern': self._analyze_person_pattern(person_timeline, analysis.timeline.duration)
            }
        }
    
    def _extract_scene_pacing(self, analysis: UnifiedAnalysis) -> Dict[str, Any]:
        """Extract data for scene pacing analysis."""
        scene_data = analysis.get_ml_data('scene_detection') or {}
        scene_entries = analysis.timeline.get_entries_by_type('scene_change')
        
        # Calculate scene durations
        scene_durations = []
        for i in range(len(scene_entries)):
            if i < len(scene_entries) - 1:
                duration = scene_entries[i+1].start.seconds - scene_entries[i].start.seconds
            else:
                duration = analysis.timeline.duration - scene_entries[i].start.seconds
            scene_durations.append(duration)
        
        # Calculate pacing metrics
        if scene_durations:
            avg_scene_duration = sum(scene_durations) / len(scene_durations)
            min_scene_duration = min(scene_durations)
            max_scene_duration = max(scene_durations)
        else:
            avg_scene_duration = analysis.timeline.duration
            min_scene_duration = analysis.timeline.duration
            max_scene_duration = analysis.timeline.duration
        
        # Pacing curve (scene changes per 5-second window)
        pacing_curve = []
        for i in range(0, int(analysis.timeline.duration), 5):
            changes_in_window = sum(
                1 for sc in scene_entries 
                if i <= sc.start.seconds < i + 5
            )
            pacing_curve.append(changes_in_window)
        
        return {
            'ml_data': {
                'total_scenes': len(scene_entries) + 1,  # +1 for initial scene
                'scene_changes': len(scene_entries),
                'average_scene_duration': avg_scene_duration,
                'min_scene_duration': min_scene_duration,
                'max_scene_duration': max_scene_duration,
                'scenes_per_minute': (len(scene_entries) / analysis.timeline.duration * 60) if analysis.timeline.duration > 0 else 0
            },
            'timelines': {
                'scene_change_times': [sc.start.seconds for sc in scene_entries],
                'scene_durations': scene_durations,
                'pacing_curve': pacing_curve
            }
        }
    
    def _calculate_speech_density(self, analysis: UnifiedAnalysis) -> List[float]:
        """Calculate speech density over time."""
        # 5-second buckets
        bucket_size = 5
        num_buckets = int(analysis.timeline.duration / bucket_size) + 1
        density = []
        
        speech_entries = analysis.timeline.get_entries_by_type('speech')
        
        for i in range(num_buckets):
            start = i * bucket_size
            end = min((i + 1) * bucket_size, analysis.timeline.duration)
            
            # Calculate speech time in this bucket
            speech_time = 0
            for entry in speech_entries:
                if entry.end:
                    # Calculate overlap with bucket
                    overlap_start = max(entry.start.seconds, start)
                    overlap_end = min(entry.end.seconds, end)
                    if overlap_end > overlap_start:
                        speech_time += overlap_end - overlap_start
            
            # Normalize to 0-1
            density.append(min(speech_time / bucket_size, 1.0))
        
        return density
    
    def _find_speech_gaps(self, segments: List[Dict[str, Any]], duration: float) -> List[Dict[str, float]]:
        """Find gaps in speech."""
        gaps = []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s['start'])
        
        # Check gap at beginning
        if sorted_segments and sorted_segments[0]['start'] > 2.0:
            gaps.append({
                'start': 0,
                'end': sorted_segments[0]['start'],
                'duration': sorted_segments[0]['start']
            })
        
        # Check gaps between segments
        for i in range(len(sorted_segments) - 1):
            gap_start = sorted_segments[i]['end']
            gap_end = sorted_segments[i + 1]['start']
            gap_duration = gap_end - gap_start
            
            if gap_duration > 2.0:  # Only significant gaps
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration': gap_duration
                })
        
        # Check gap at end
        if sorted_segments and sorted_segments[-1]['end'] < duration - 2.0:
            gaps.append({
                'start': sorted_segments[-1]['end'],
                'end': duration,
                'duration': duration - sorted_segments[-1]['end']
            })
        
        return gaps[:10]  # Limit to 10 most significant gaps
    
    def _analyze_content_metadata_alignment(self, analysis: UnifiedAnalysis, 
                                          hashtags: List[str], 
                                          keywords: List[str]) -> Dict[str, Any]:
        """Analyze how well content aligns with metadata."""
        # Simple alignment check - does speech/text mention hashtags/keywords?
        speech_entries = analysis.timeline.get_entries_by_type('speech')
        text_entries = analysis.timeline.get_entries_by_type('text')
        
        all_content_text = ' '.join(
            e.data.get('text', '') for e in speech_entries + text_entries
        ).lower()
        
        hashtag_mentions = {}
        for tag in hashtags:
            tag_lower = tag.lower().replace('#', '')
            if tag_lower in all_content_text:
                hashtag_mentions[tag] = True
        
        keyword_mentions = {}
        for keyword in keywords[:20]:  # Limit to top 20
            if keyword in all_content_text:
                keyword_mentions[keyword] = True
        
        return {
            'hashtag_alignment': len(hashtag_mentions) / len(hashtags) if hashtags else 0,
            'keyword_alignment': len(keyword_mentions) / len(keywords) if keywords else 0,
            'mentioned_hashtags': list(hashtag_mentions.keys()),
            'mentioned_keywords': list(keyword_mentions.keys())
        }
    
    def _get_position_description(self, bbox: List[float]) -> str:
        """Convert bounding box to position description."""
        if not bbox or len(bbox) < 4:
            return 'unknown'
        
        x, y, w, h = bbox[:4]
        
        # Simple position description based on x coordinate
        if x < 0.33:
            horizontal = 'left'
        elif x > 0.67:
            horizontal = 'right'
        else:
            horizontal = 'center'
        
        # Vertical position
        if y < 0.33:
            vertical = 'top'
        elif y > 0.67:
            vertical = 'bottom'
        else:
            vertical = 'middle'
        
        return f"{vertical}-{horizontal}"
    
    def _analyze_person_pattern(self, person_timeline: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """Analyze pattern of person appearances."""
        if not person_timeline:
            return {
                'pattern': 'no_person',
                'consistency': 0,
                'average_duration': 0
            }
        
        # Find continuous segments
        segments = []
        current_segment_start = None
        
        for i in range(int(duration)):
            timestamp = f"{i}s"
            if timestamp in person_timeline:
                if current_segment_start is None:
                    current_segment_start = i
            else:
                if current_segment_start is not None:
                    segments.append({
                        'start': current_segment_start,
                        'end': i,
                        'duration': i - current_segment_start
                    })
                    current_segment_start = None
        
        # Handle last segment
        if current_segment_start is not None:
            segments.append({
                'start': current_segment_start,
                'end': duration,
                'duration': duration - current_segment_start
            })
        
        # Analyze pattern
        if not segments:
            pattern = 'no_person'
        elif len(segments) == 1:
            pattern = 'continuous'
        elif len(segments) <= 3:
            pattern = 'intermittent'
        else:
            pattern = 'frequent_cuts'
        
        avg_duration = sum(s['duration'] for s in segments) / len(segments) if segments else 0
        
        return {
            'pattern': pattern,
            'segment_count': len(segments),
            'average_duration': avg_duration,
            'total_screen_time': sum(s['duration'] for s in segments)
        }