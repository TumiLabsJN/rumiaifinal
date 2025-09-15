# Phase 2 Integration Decisions

## Decision 1: Async/Sync Mismatch
**Context**: `compute_temporal_windows` is synchronous but needs to be called from async context in rumiai_runner.py. The codebase uses GPU for ML models and async for resource coordination.

**Solution**: Use asyncio.to_thread() to bridge sync/async
```python
# In rumiai_runner.py
async def process_video(self, video_path: str, video_id: str):
    # ... extract timelines, metadata, speech ...
    
    # Run synchronous computation in thread pool
    result = await asyncio.to_thread(
        compute_temporal_windows,
        timelines,
        video_metadata,
        speech_segments,
        audio_path
    )
```

**Rationale**:
- Keeps temporal_compute synchronous (pure Python computation, no GPU)
- Doesn't block event loop (important for GPU service coordination)
- Consistent with how audio_energy_service handles CPU-bound librosa operations
- Frees async loop for GPU/ML service management
- Matches existing codebase patterns for CPU-bound work

## Decision 2: Wrong Import Structure
**Context**: P0Phase2.md had `from rumiai_v2.utils.video_utils import get_video_metadata` but this module doesn't exist. Need to determine how to get video metadata for temporal computation.

**Investigation**:
- Current flow gets metadata from Apify scraping (VideoMetadata object)
- Converts to dict via `video_metadata.to_dict()`
- Apify provides social metrics but field names don't match temporal_compute expectations
- Critical metadata needed:
  - Duration (for temporal windows)
  - Outcome metrics (ML targets: views, likes, comments, shares, engagement)
  - Global features (caption metrics, publish time, etc.)

**Solution**: Use Apify metadata with transformation function
```python
import re
from datetime import datetime

def transform_metadata_for_temporal(apify_metadata: Dict) -> Dict:
    """Transform Apify metadata to temporal compute format"""
    
    description = apify_metadata.get('description', '')
    create_time_str = apify_metadata.get('createTime', '')
    
    # Parse datetime for temporal features
    try:
        dt = datetime.fromisoformat(create_time_str.replace('Z', '+00:00'))
        publish_hour = dt.hour
        publish_day = dt.weekday()
    except:
        publish_hour = 0
        publish_day = 0
    
    # Extract caption metrics
    mention_count = len(re.findall(r'@\w+', description))
    emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', description))
    word_count = len(description.split())
    link_present = 1 if re.search(r'https?://\S+', description) else 0
    cta_present = 1 if any(cta in description.lower() for cta in 
                          ['link in bio', 'comment', 'follow', 'like', 'share']) else 0
    
    return {
        # Critical
        'video_id': apify_metadata.get('id', ''),
        'duration': apify_metadata.get('duration', 0),
        
        # Outcome metrics (ML targets)
        'view_count': apify_metadata.get('views', 0),
        'like_count': apify_metadata.get('likes', 0),
        'comment_count': apify_metadata.get('comments', 0),
        'share_count': apify_metadata.get('shares', 0),
        'engagement_rate': apify_metadata.get('engagementRate', 0.0),
        
        # Global metadata features
        'caption_length': len(description),
        'hashtag_count': len(apify_metadata.get('hashtags', [])),
        'mention_count': mention_count,
        'emoji_count': emoji_count,
        'word_count': word_count,
        'link_present': link_present,
        'call_to_action': cta_present,
        'publish_hour': publish_hour,
        'publish_day_of_week': publish_day,
        'has_soundtrack': bool(apify_metadata.get('music', {})),
    }
```

**Rationale**:
- Preserves all critical outcome metrics (ML targets)
- Provides all required metadata for temporal windows
- Extracts additional ML features from caption
- Single source of truth (Apify) avoids sync issues
- Transformation is lightweight and fast

## Decision 3: Service Integration Strategy
**Context**: P0Phase2.md showed extracting timelines separately, but the existing architecture already handles this through VideoAnalyzer → TimelineBuilder → UnifiedAnalysis flow.

**Investigation**:
- Studied RUMIAI_CORE_ARCHITECTURE_PRINCIPLES.md - Python-only, fail-fast design
- Reviewed Codemappingfinal.md - VideoAnalyzer orchestrates ML services
- Analyzed ML_DATA_PROCESSING_PIPELINE.md - UnifiedAnalysis provides all timelines
- Current flow: VideoAnalyzer runs ML → TimelineBuilder creates unified timeline → Precompute functions process

**Solution**: Use existing architecture, don't duplicate timeline extraction
```python
# WRONG: Don't re-run ML services or extract timelines manually
# The existing pipeline already does this in video_analyzer.py

# RIGHT: Use the existing flow in rumiai_runner.py
async def process_video_url(self, video_url: str):
    # ... existing scraping and download ...
    
    # Step 3: ML analysis (already extracts all timelines)
    ml_results = await self.video_analyzer.analyze_video(video_id, video_path)
    
    # Step 4: Build unified timeline (already combines all ML data)
    unified_analysis = self.timeline_builder.build_timeline(
        video_id, 
        video_metadata.to_dict(), 
        ml_results
    )
    
    # Step 5: NEW - Run temporal compute
    # Transform metadata for temporal compute
    temporal_metadata = transform_metadata_for_temporal(video_metadata.to_dict())
    
    # Extract speech segments from unified analysis
    speech_segments = extract_speech_segments(unified_analysis)
    
    # Run temporal computation in thread pool
    temporal_result = await asyncio.to_thread(
        compute_temporal_windows,
        unified_analysis.to_dict(),  # Contains all timelines
        temporal_metadata,
        speech_segments,
        None  # Audio path if needed
    )
    
    # Save temporal analysis
    self.save_analysis_result(video_id, "temporal_windows", temporal_result)
```

**Rationale**:
- Leverage existing VideoAnalyzer → TimelineBuilder flow
- UnifiedAnalysis already contains all timelines (text, sticker, gaze, emotion, etc.)
- No duplication of ML service calls
- Maintains fail-fast architecture principle
- Follows existing data transformation pipeline