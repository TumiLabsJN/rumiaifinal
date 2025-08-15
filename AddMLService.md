# Adding a New ML Service to RumiAI

This guide documents the complete process for adding a new ML service to the RumiAI video analysis pipeline, based on lessons learned from implementing MediaPipe gesture recognition.

## Architecture Overview

RumiAI follows a **unified frame extraction → ML processing → timeline integration** architecture:

```
Video → Unified Frame Manager → ML Services (parallel) → Timeline Entries → Compute Functions → Professional Wrappers
```

## Key Integration Points

When adding a new ML service, you must understand and integrate with these 5 critical components:

### 1. Unified Frame Extraction (`unified_frame_manager.py`)
- **Purpose**: Extract frames ONCE, analyze MANY times for efficiency
- **Frame Types**: Different services get optimized frame sets (YOLO gets fewer frames, OCR gets adaptive sampling, MediaPipe gets all frames)
- **Key Method**: `get_frames_for_service(frames, service_name)` - defines which frames your service processes

### 2. ML Services Pipeline (`ml_services_unified.py`)
- **Purpose**: Parallel ML processing with lazy loading and async patterns
- **Integration Points**:
  - Add service to `_model_locks` dictionary
  - Implement `_ensure_model_loaded(model_name)` case
  - Add service call to `analyze_video()` gather block
  - Implement `_run_[service]_on_frames()` method
  - Add empty result fallback method

### 3. Timeline Builder (`timeline_builder.py`)
- **Purpose**: Convert ML detections into timeline entries with timestamps
- **Key Pattern**: ML results → standardized timeline entries with `entry_type`, `timestamp`, `data`
- **Note**: Can usually process new data types without changes if following standard format

### 4. Timeline Extraction (`precompute_functions.py`)
- **Purpose**: Extract timeline data for compute functions
- **Integration Point**: `_extract_timelines_from_analysis()` method
- **Pattern**: Extract your service's timeline entries and add to appropriate data structures

### 5. Professional Wrappers (`precompute_functions_full.py`)
- **Purpose**: Compute professional metrics using timeline data
- **Integration Points**: 
  - Visual Overlay: Add new visual elements detection
  - Person Framing: Add behavioral metrics (eye contact, gestures, etc.)
  - Hook Detection: Add engagement signals

## Implementation Checklist

### Phase 1: Service Integration
- [ ] **Frame Strategy**: Decide sampling strategy (all frames vs. adaptive sampling)
- [ ] **Model Loading**: Add to `_model_locks` and `_ensure_model_loaded()`
- [ ] **Batch Processing**: Implement `_process_[service]_batch()` for efficiency
- [ ] **Results Format**: Define output schema (similar to existing services)
- [ ] **Error Handling**: Add empty result fallbacks and timeout handling

### Phase 2: Timeline Integration  
- [ ] **Timeline Format**: Ensure results follow `{entry_type, timestamp, data}` pattern
- [ ] **Timeline Extraction**: Add extraction logic in `_extract_timelines_from_analysis()`
- [ ] **Data Aggregation**: Handle batch aggregation if processing in chunks

### Phase 3: Compute Functions
- [ ] **Visual Overlay**: Add new detections to CTA reinforcement logic
- [ ] **Person Framing**: Add behavioral signals (confidence, engagement, etc.)
- [ ] **Hook Detection**: Add engagement metrics if applicable

### Phase 4: Performance & Testing
- [ ] **Performance Analysis**: Measure processing time per frame and total overhead
- [ ] **Frame Sampling Decision**: Use performance data to decide on sampling strategy
- [ ] **Integration Testing**: Test with real videos end-to-end
- [ ] **Error Recovery**: Verify graceful degradation when service fails

## Critical Design Patterns

### 1. Lazy Loading Pattern
```python
# Add to _model_locks
self._model_locks = {
    'existing_service': asyncio.Lock(),
    'your_service': asyncio.Lock()  # ADD THIS
}

# Add to _ensure_model_loaded
elif model_name == 'your_service':
    self._models['your_service'] = await self._load_your_service()
```

### 2. Batch Processing Pattern  
```python
def _process_your_service_batch(self, model, frames: List[FrameData]) -> List[Dict]:
    """Process frames in batch for efficiency"""
    results = []
    for frame_data in frames:
        # Process individual frame
        detection = model.process(frame_data.image)
        results.append({
            'timestamp': frame_data.timestamp,
            'frame_number': frame_data.frame_number,
            'data': detection
        })
    return results
```

### 3. Timeline Integration Pattern
```python
# In _extract_timelines_from_analysis()
for entry in timeline_entries:
    if entry.get('entry_type') == 'your_service':
        # Extract and process your service's timeline data
        your_data = entry.get('data', {})
        # Add to appropriate timeline structures
```

### 4. Async Parallel Processing
```python
# Add to analyze_video() gather block
results = await asyncio.gather(
    run_with_timeout(self._run_yolo_on_frames(...), 300, "YOLO"),
    run_with_timeout(self._run_your_service_on_frames(...), 300, "YourService"),  # ADD
    return_exceptions=True
)
```

## Performance Considerations

### Frame Sampling Strategy
- **All Frames**: Use for critical detections (gestures, facial expressions) where missing frames impacts accuracy
- **Adaptive Sampling**: Use for text/object detection where temporal consistency is less critical  
- **Performance Rule**: If total processing time < 1 second for 60s video, process all frames

### Memory Management
- **Batch Size**: Use 10-20 frame batches to balance memory usage and threading efficiency
- **Resource Cleanup**: Implement proper model cleanup in service destructors
- **Shared Resources**: Use singleton pattern for heavy models to avoid multiple loads

## Critical Pitfalls and Architectural Lessons

### 1. **CRITICAL: Timeline Structure Conflicts**
**The Gesture Recognition Bug** - This caused 6 out of 8 analysis flows to fail silently:

- **Root Cause**: Conflicting data extraction methods in `_extract_timelines_from_analysis()`
- **What Happened**: 
  - Old code: `timelines['gestureTimeline'][timestamp_key] = gesture` (direct assignment)
  - New code: `timelines['gestureTimeline'][timestamp_key] = {'gestures': [], 'confidence': 0}` (structured)
  - Old code overwrote new structure → `KeyError: 'gestures'` → ALL compute functions failed
- **Impact**: Only metadata_analysis and temporal_markers completed (don't use gesture data)
- **Lesson**: **NEVER have two different data extraction methods for the same service**

**Prevention Strategy**:
```python
# ❌ BAD: Multiple extraction methods
# Method 1 (old)
timelines['serviceTimeline'][key] = raw_data

# Method 2 (new) 
timelines['serviceTimeline'][key] = {'structured': data}

# ✅ GOOD: Single extraction method
# Only use timeline entries approach
```

### 2. **Silent Failure Detection**
- **Problem**: Processing appeared successful but 6/8 flows silently failed
- **Detection**: Always verify complete output structure in insights folder
- **Prevention**: Add validation that ALL expected analysis flows completed

### 3. **Data Structure Evolution Conflicts**
- **Problem**: New timeline structure (`timeline.entries[]`) vs old structure (`timeline_data`)
- **Impact**: Precompute functions expected old structure, timeline builder created new structure
- **Lesson**: When changing data structures, update ALL consuming code simultaneously

### 4. **Common Technical Pitfalls**

1. **Frame Format Confusion**: Ensure RGB/BGR format consistency across services
2. **Timeline Schema Mismatch**: Follow existing `entry_type` patterns exactly
3. **Missing Error Handling**: Always provide empty result fallbacks
4. **Performance Assumptions**: Measure actual processing time before deciding on sampling
5. **Integration Testing Gap**: Test full pipeline, not just service in isolation
6. **Timeline Extraction Missing**: Service works but compute functions can't access data
7. **Conflicting Extraction Logic**: Multiple ways to extract same data type (CRITICAL)

## Testing Strategy

### Essential Testing Phases

1. **Unit Test**: Service in isolation with sample frames
2. **Integration Test**: Full video processing pipeline 
3. **Performance Test**: Measure processing time per frame
4. **Real Video Test**: Test with actual content containing target features
5. **Failure Test**: Verify graceful degradation when model fails to load

### **CRITICAL: Complete Pipeline Validation**
After implementing your service, always verify:

```bash
# 1. Check that ALL 8 analysis flows completed
ls insights/[video_id]/
# Expected: creative_density, emotional_journey, metadata_analysis, 
#          person_framing, scene_pacing, speech_analysis, 
#          temporal_markers, visual_overlay_analysis

# 2. Verify timeline extraction works
python3 -c "
from rumiai_v2.processors.precompute_functions import _extract_timelines_from_analysis
import json
with open('unified_analysis/[video_id].json', 'r') as f:
    analysis = json.load(f)
timelines = _extract_timelines_from_analysis(analysis)
print('Timeline keys:', list(timelines.keys()))
print('[service]Timeline entries:', len(timelines.get('[service]Timeline', {})))
"

# 3. Test all compute functions individually
python3 -c "
from rumiai_v2.processors import COMPUTE_FUNCTIONS
import json
with open('unified_analysis/[video_id].json', 'r') as f:
    analysis = json.load(f)
for func_name, func in COMPUTE_FUNCTIONS.items():
    try:
        result = func(analysis)
        print(f'✓ {func_name}: {\"SUCCESS\" if result else \"EMPTY\"}')
    except Exception as e:
        print(f'❌ {func_name}: {e}')
"
```

### **Silent Failure Prevention**
- **Never trust** "processing completed successfully" without verifying output
- **Always count** output files in insights folder
- **Test with multiple videos** to catch edge cases
- **Verify data flow** from service → timeline → compute functions

### **Debugging Cascade Failures**
When compute functions mysteriously fail:

```python
# 1. Test timeline extraction directly
from rumiai_v2.processors.precompute_functions import _extract_timelines_from_analysis
try:
    timelines = _extract_timelines_from_analysis(analysis)
    print("✓ Timeline extraction successful")
except Exception as e:
    print(f"❌ Timeline extraction failed: {e}")
    # This is where the gesture bug manifested

# 2. Check for data structure conflicts
timeline_data = analysis.get('timeline', {})
entries = timeline_data.get('entries', [])
service_entries = [e for e in entries if e.get('entry_type') == 'your_service']
print(f"Service entries found: {len(service_entries)}")

# 3. Verify no conflicting extraction methods
# Search for: timelines['serviceTimeline'][key] = 
# Should only have ONE pattern in _extract_timelines_from_analysis()
```

**Red Flags**:
- Processing completes but insights folder has fewer than 8 directories
- Compute functions fail with KeyError on data structure access
- Timeline extraction throws exceptions during processing
- Different parts of code extract same service data differently

## Documentation Requirements

When implementation is complete, update:
- Service-specific documentation in `docs/` directory
- Performance benchmarks in main README
- Timeline schema documentation
- Compute function integration notes

## Real-World Example: Gesture Recognition Implementation

The gesture recognition service demonstrates both successful patterns and critical failures:

### **What Went Right**
1. **Service**: `gesture_recognizer_service.py` - Singleton MediaPipe service
2. **Integration**: Added to `ml_services_unified.py` with lazy loading  
3. **Timeline**: Gesture events flow through timeline builder
4. **Performance**: 2-3ms per frame, processes all frames (360ms total overhead negligible)
5. **Detection**: Successfully detected open_palm, thumbs_up, pointing gestures
6. **Enhancement**: Creative density analysis uses gesture data for CTA reinforcement

### **What Went Wrong (The Critical Bug)**
1. **Timeline Extraction Conflict**: Two different extraction methods in same function
2. **Silent Cascade Failure**: 6 out of 8 analysis flows failed silently
3. **Data Structure Mismatch**: New timeline builder vs old precompute expectations
4. **False Success**: Processing appeared complete but output was degraded

### **Lessons Learned**
- ✅ **Technical Success**: Gesture detection worked perfectly (124 gestures detected)
- ❌ **Integration Failure**: Architectural mismatch broke entire pipeline
- 🔧 **Fix Required**: Remove conflicting extraction method
- 📊 **Result**: Full pipeline restoration with gesture enhancement

### **Key Takeaway**
Adding a new ML service is not just about the service itself - it's about ensuring **every downstream dependency** works correctly. One small conflict in data extraction can cascade to break the entire analysis pipeline while appearing to succeed.

**Always validate the complete end-to-end pipeline**, not just your individual service.