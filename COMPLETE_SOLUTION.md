# Complete Architectural Fix for ML Data Extraction

## Current State: Band-Aid Solution
Our proposed fix only addresses 1 of 7 technical debt items, leaving the architecture messy and prone to future failures.

## The Complete Solution

### Phase 1: Clean Up Technical Debt

#### 1.1 Remove Dead Code (lines 96-106)
```python
# DELETE THIS ENTIRE BLOCK:
def compute_creative_density_wrapper(analysis_data):
    """Wrapper with format compatibility"""
    # Extract data using helpers
    yolo_objects = extract_yolo_data(analysis_data)
    mediapipe_data = extract_mediapipe_data(analysis_data)
    ocr_data = extract_ocr_data(analysis_data)
    whisper_data = extract_whisper_data(analysis_data)
    
    # Continue with existing logic...
    # (rest of the function remains the same)
```
**Why**: Eliminates confusion, removes abandoned code, single source of truth

#### 1.2 Add Import for Logging
```python
import logging
logger = logging.getLogger(__name__)
```

### Phase 2: Implement Robust Extraction with Validation

#### 2.1 Complete _extract_timelines_from_analysis Function
```python
def _extract_timelines_from_analysis(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract timeline data from unified analysis with validation and error handling.
    
    This function transforms raw ML data into timeline format required by compute functions.
    Uses helper functions to handle multiple data format variations defensively.
    
    Args:
        analysis_dict: Unified analysis dict from analysis.to_dict()
        
    Returns:
        Dict with timeline keys (textOverlayTimeline, objectTimeline, etc.)
        Each timeline uses timestamp keys like '0-1s', '1-2s'
    """
    timeline_data = analysis_dict.get('timeline', {})
    ml_data = analysis_dict.get('ml_data', {})
    
    # Initialize all timeline types
    timelines = {
        'textOverlayTimeline': {},
        'stickerTimeline': {},
        'speechTimeline': {},
        'objectTimeline': {},
        'gestureTimeline': {},
        'expressionTimeline': {},
        'sceneTimeline': {},
        'sceneChangeTimeline': {},
        'personTimeline': {},
        'cameraDistanceTimeline': {}
    }
    
    # Track extraction metrics for monitoring
    extraction_metrics = {
        'video_id': analysis_dict.get('video_id', 'unknown'),
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    # ========== OCR EXTRACTION WITH VALIDATION ==========
    try:
        # Get available data count for validation
        ocr_available = len(ml_data.get('ocr', {}).get('textAnnotations', []))
        
        # Use helper for robust extraction
        ocr_data = extract_ocr_data(ml_data)
        ocr_annotations = ocr_data.get('textAnnotations', [])
        ocr_extracted = len(ocr_annotations)
        
        # Validate extraction success
        if ocr_available > 0 and ocr_extracted == 0:
            logger.error(f"OCR extraction failed: {ocr_available} annotations available but 0 extracted")
            extraction_metrics['ocr_failure'] = True
        elif ocr_extracted > 0:
            logger.info(f"OCR extraction successful: {ocr_extracted}/{ocr_available} annotations")
        
        # Transform to timeline format
        for annotation in ocr_annotations:
            timestamp = annotation.get('timestamp', 0)
            start = int(timestamp)
            end = start + 1
            timestamp_key = f"{start}-{end}s"
            
            # Derive position from bbox
            bbox = annotation.get('bbox', [0, 0, 0, 0])
            y_pos = bbox[1] if len(bbox) > 1 else 0
            
            # Classify position based on frame coordinates
            if y_pos < 200:
                position = 'top'
            elif y_pos > 500:
                position = 'bottom'
            else:
                position = 'center'
            
            timelines['textOverlayTimeline'][timestamp_key] = {
                'text': annotation.get('text', ''),
                'position': position,
                'size': 'medium',  # Could be derived from bbox dimensions
                'confidence': annotation.get('confidence', 0.9),
                'bbox': bbox
            }
        
        # Handle stickers
        stickers = ocr_data.get('stickers', [])
        for sticker in stickers:
            timestamp = sticker.get('timestamp', 0)
            timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
            timelines['stickerTimeline'][timestamp_key] = sticker
            
        extraction_metrics['ocr_extracted'] = ocr_extracted
        extraction_metrics['stickers_extracted'] = len(stickers)
        
    except Exception as e:
        logger.error(f"OCR extraction error: {e}", exc_info=True)
        extraction_metrics['ocr_error'] = str(e)
    
    # ========== YOLO EXTRACTION WITH VALIDATION ==========
    try:
        # Get available data count
        yolo_available = len(ml_data.get('yolo', {}).get('objectAnnotations', []))
        
        # Use helper for extraction
        yolo_objects = extract_yolo_data(ml_data)
        yolo_extracted = len(yolo_objects)
        
        # Validate
        if yolo_available > 0 and yolo_extracted == 0:
            logger.error(f"YOLO extraction failed: {yolo_available} objects available but 0 extracted")
            extraction_metrics['yolo_failure'] = True
        elif yolo_extracted > 0:
            logger.info(f"YOLO extraction successful: {yolo_extracted}/{yolo_available} objects")
        
        # Transform to timeline format
        for obj in yolo_objects:
            timestamp = obj.get('timestamp', 0)
            timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
            
            if timestamp_key not in timelines['objectTimeline']:
                timelines['objectTimeline'][timestamp_key] = []
            
            timelines['objectTimeline'][timestamp_key].append({
                'class': obj.get('className', 'unknown'),
                'confidence': obj.get('confidence', 0.5),
                'trackId': obj.get('trackId', ''),
                'bbox': obj.get('bbox', [])
            })
            
        extraction_metrics['yolo_extracted'] = yolo_extracted
        
    except Exception as e:
        logger.error(f"YOLO extraction error: {e}", exc_info=True)
        extraction_metrics['yolo_error'] = str(e)
    
    # ========== WHISPER EXTRACTION WITH VALIDATION ==========
    try:
        # Get available data count
        whisper_available = len(ml_data.get('whisper', {}).get('segments', []))
        
        # Use helper for extraction
        whisper_data = extract_whisper_data(ml_data)
        segments = whisper_data.get('segments', [])
        whisper_extracted = len(segments)
        
        # Validate
        if whisper_available > 0 and whisper_extracted == 0:
            logger.error(f"Whisper extraction failed: {whisper_available} segments available but 0 extracted")
            extraction_metrics['whisper_failure'] = True
        elif whisper_extracted > 0:
            logger.info(f"Whisper extraction successful: {whisper_extracted}/{whisper_available} segments")
        
        # Transform to timeline format
        for segment in segments:
            start = int(segment.get('start', 0))
            end = int(segment.get('end', start + 1))
            timestamp_key = f"{start}-{end}s"
            
            timelines['speechTimeline'][timestamp_key] = {
                'text': segment.get('text', ''),
                'confidence': segment.get('confidence', 0.9),
                'start_time': segment.get('start', 0),
                'end_time': segment.get('end', 0)
            }
            
        extraction_metrics['whisper_extracted'] = whisper_extracted
        
    except Exception as e:
        logger.error(f"Whisper extraction error: {e}", exc_info=True)
        extraction_metrics['whisper_error'] = str(e)
    
    # ========== MEDIAPIPE EXTRACTION WITH VALIDATION ==========
    try:
        # Use helper for extraction
        mediapipe_data = extract_mediapipe_data(ml_data)
        
        poses = mediapipe_data.get('poses', [])
        faces = mediapipe_data.get('faces', [])
        hands = mediapipe_data.get('hands', [])
        gestures = mediapipe_data.get('gestures', [])
        
        # Transform poses to person timeline
        for pose in poses:
            timestamp = pose.get('timestamp', '0-1s')
            timelines['personTimeline'][timestamp] = pose
        
        # Transform faces to expression timeline
        for face in faces:
            timestamp = face.get('timestamp', '0-1s')
            timelines['expressionTimeline'][timestamp] = face
        
        # Transform gestures to gesture timeline
        for gesture in gestures:
            timestamp = gesture.get('timestamp', '0-1s')
            timelines['gestureTimeline'][timestamp] = gesture
            
        extraction_metrics['mediapipe_poses'] = len(poses)
        extraction_metrics['mediapipe_faces'] = len(faces)
        extraction_metrics['mediapipe_gestures'] = len(gestures)
        
        if poses or faces or gestures:
            logger.info(f"MediaPipe extraction: {len(poses)} poses, {len(faces)} faces, {len(gestures)} gestures")
            
    except Exception as e:
        logger.error(f"MediaPipe extraction error: {e}", exc_info=True)
        extraction_metrics['mediapipe_error'] = str(e)
    
    # ========== SCENE EXTRACTION (Keep existing working code) ==========
    # This already works, don't break it
    timeline_entries = timeline_data.get('entries', [])
    scenes = []
    scene_changes = []
    current_scene_start = 0
    
    for entry in timeline_entries:
        if entry.get('entry_type') == 'scene_change':
            start_str = entry.get('start', '0s')
            if isinstance(start_str, str) and start_str.endswith('s'):
                try:
                    start_seconds = float(start_str[:-1])
                except ValueError:
                    start_seconds = 0
            else:
                start_seconds = 0
            
            scene_changes.append(start_seconds)
            scenes.append({
                'start': current_scene_start,
                'end': start_seconds
            })
            current_scene_start = start_seconds
    
    # Add final scene
    duration = timeline_data.get('duration', 0)
    if current_scene_start < duration:
        scenes.append({
            'start': current_scene_start,
            'end': duration
        })
    
    # Convert to timeline format
    for i, scene in enumerate(scenes):
        start = scene['start']
        end = scene['end']
        timestamp = f"{int(start)}-{int(end)}s"
        timelines['sceneTimeline'][timestamp] = {
            'scene_number': i + 1,
            'duration': end - start
        }
    
    # Create scene change timeline
    for i, change_time in enumerate(scene_changes):
        timestamp = f"{int(change_time)}-{int(change_time)}s"
        timelines['sceneChangeTimeline'][timestamp] = {
            'type': 'scene_change',
            'scene_index': i + 1,
            'actual_time': change_time
        }
    
    extraction_metrics['scenes_extracted'] = len(scenes)
    
    # ========== LOG EXTRACTION SUMMARY ==========
    total_ml_extracted = (
        extraction_metrics.get('ocr_extracted', 0) +
        extraction_metrics.get('yolo_extracted', 0) +
        extraction_metrics.get('whisper_extracted', 0) +
        extraction_metrics.get('scenes_extracted', 0)
    )
    
    logger.info(f"Timeline extraction complete for {extraction_metrics['video_id']}: "
                f"{total_ml_extracted} total elements extracted")
    
    # Log metrics for monitoring (could be sent to metrics service)
    if any('failure' in k for k in extraction_metrics.keys()):
        logger.warning(f"Extraction had failures: {extraction_metrics}")
    
    # Add extraction metrics to timelines for debugging
    timelines['_extraction_metrics'] = extraction_metrics
    
    return timelines
```

### Phase 3: Add Tests

#### 3.1 Create test_timeline_extraction.py
```python
def test_extraction_with_ml_data():
    """Test that extraction works with real ML data"""
    # Load test data with known ML detections
    with open('test_data/sample_analysis.json') as f:
        analysis = json.load(f)
    
    timelines = _extract_timelines_from_analysis(analysis)
    
    # Verify extraction
    assert len(timelines['textOverlayTimeline']) > 0, "OCR extraction failed"
    assert len(timelines['objectTimeline']) > 0, "YOLO extraction failed"
    assert len(timelines['speechTimeline']) > 0, "Whisper extraction failed"
    
def test_extraction_with_missing_data():
    """Test graceful handling of missing ML data"""
    analysis = {'ml_data': {}, 'timeline': {}}
    
    timelines = _extract_timelines_from_analysis(analysis)
    
    # Should not crash, should return empty timelines
    assert timelines is not None
    assert isinstance(timelines['textOverlayTimeline'], dict)

def test_extraction_validation_logs_warnings():
    """Test that validation catches extraction failures"""
    # Create data where ML exists but extraction would fail
    analysis = {
        'ml_data': {
            'ocr': {'textAnnotations': [1, 2, 3]},  # Data exists
        }
    }
    
    with capture_logs() as logs:
        timelines = _extract_timelines_from_analysis(analysis)
        
    # Should log warning about extraction failure
    assert any('extraction failed' in log for log in logs)
```

### Phase 4: Add Monitoring

#### 4.1 Create extraction_monitor.py
```python
class ExtractionMonitor:
    """Monitor extraction health and alert on failures"""
    
    def __init__(self):
        self.failure_threshold = 0.1  # Alert if >10% extractions fail
        self.recent_extractions = []
    
    def record_extraction(self, metrics):
        """Record extraction metrics"""
        self.recent_extractions.append(metrics)
        
        # Keep last 100 extractions
        if len(self.recent_extractions) > 100:
            self.recent_extractions.pop(0)
        
        # Check failure rate
        failures = sum(1 for m in self.recent_extractions 
                      if 'failure' in str(m))
        failure_rate = failures / len(self.recent_extractions)
        
        if failure_rate > self.failure_threshold:
            self.alert_extraction_failures(failure_rate)
    
    def alert_extraction_failures(self, rate):
        """Send alert about high failure rate"""
        logger.critical(f"High extraction failure rate: {rate:.1%}")
        # Could send to Slack, PagerDuty, etc.
```

## Benefits of Complete Solution

### What We Gain
1. **Clean Architecture**: No dead code, single source of truth
2. **Observability**: Know when extraction fails
3. **Reliability**: Handle errors gracefully
4. **Maintainability**: Clear, documented code
5. **Debuggability**: Metrics and logging
6. **Preventive**: Catches issues before they become 98% data loss

### Technical Debt Eliminated
- ✅ Dead code removed
- ✅ Duplicate functions eliminated
- ✅ Validation added
- ✅ Error handling implemented
- ✅ Monitoring in place
- ✅ Format assumptions documented
- ✅ Tests added

### Risk Mitigation
- Format changes detected immediately
- Extraction failures logged and alerted
- Graceful degradation on errors
- Metrics track system health

## Implementation Priority

### Must Have (Do First)
1. Remove dead code (5 min)
2. Fix extraction with helpers (30 min)
3. Add basic validation logging (15 min)

### Should Have (Do Next)
1. Add error handling (20 min)
2. Add extraction metrics (20 min)
3. Document the changes (10 min)

### Nice to Have (Do Later)
1. Add comprehensive tests (1 hour)
2. Set up monitoring dashboard (2 hours)
3. Add alerting system (1 hour)

## Total Effort
- Band-aid fix: 30 minutes (but leaves debt)
- Complete fix: 2-3 hours (eliminates debt)
- ROI: Prevent future 98% data loss incidents

## Conclusion
The band-aid fix would work but leave us vulnerable to similar issues. The complete fix takes slightly longer but creates a robust, maintainable system that won't silently lose 98% of data again.