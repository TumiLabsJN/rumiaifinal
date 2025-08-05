# RumiAI Core Architecture Principles
**Non-Negotiable Design Requirements**  
**Last Updated**: 2025-08-05 - Post Unified ML Implementation & Testing Fixes

*This document defines the fundamental architectural principles that MUST be maintained in RumiAI. These are not suggestions - they are requirements.*

---

## 1. Unified Analysis Structure (CRITICAL)

### Principle
All ML outputs MUST flow through a single `UnifiedAnalysis` object that serves as the single source of truth for video analysis.

### Requirements
- **ONE analysis object per video** containing ALL ML results
- **Consistent timeline structure** across all ML models
- **Temporal alignment** of all ML outputs to video timestamps
- **No scattered outputs** - everything must be accessible from UnifiedAnalysis

### Implementation
```python
UnifiedAnalysis:
  - video_id: str (unique identifier)
  - timeline: TimelineData (0-1s, 1-2s, etc.)
  - ml_results: Dict[str, MLAnalysisResult]
    - yolo: object detections
    - whisper: transcription
    - mediapipe: human analysis
    - ocr: text detection
    - scene: scene boundaries
  - temporal_markers: List[TemporalMarker]
  - metadata: VideoMetadata
  
# CRITICAL: to_dict() must include ml_data field
def to_dict():
    result['ml_data'] = {
        'yolo': ml_results['yolo'].data if success else {},
        'mediapipe': ml_results['mediapipe'].data if success else {},
        # ... all ML services
    }
```

### Why This Matters
- Claude prompts need consistent data structure
- Temporal correlation requires unified timeline
- Debugging requires single source of truth
- Cost optimization depends on efficient data access

---

## 2. Unified Frame Extraction Pipeline (MANDATORY)

### Principle
Frames must be extracted ONCE per video and shared across all ML services.

### Requirements
- **Single video decode** - NEVER decode the same video multiple times
- **Shared frame pool** - All ML services use the same extracted frames
- **LRU caching** with size limits (10GB max, 100 videos max)
- **Adaptive sampling** based on video duration:
  - <30s: 5 FPS
  - 30-60s: 3 FPS
  - >60s: 2 FPS

### Service-Specific Frame Distribution (Implemented in unified_frame_manager.py)
```python
FrameSamplingConfig.CONFIGS = {
    'yolo': {
        'max_frames': 100,  # ~2 FPS for 60s video
        'strategy': 'uniform',
        'rationale': 'Object detection needs consistent temporal coverage'
    },
    'mediapipe': {
        'max_frames': 180,  # All frames
        'strategy': 'all',
        'rationale': 'Human motion requires high temporal resolution'
    },
    'ocr': {
        'max_frames': 60,  # ~1 FPS for 60s video
        'strategy': 'adaptive',
        'rationale': 'Text appears more at beginning/end (titles/CTAs)'
    },
    'scene': {
        'max_frames': None,  # All frames needed
        'strategy': 'all',
        'rationale': 'Scene boundaries require full temporal analysis'
    }
}
```

### Anti-Patterns to AVOID
❌ Each service reading video independently
❌ Extracting frames multiple times
❌ Unlimited frame caching
❌ Fixed FPS regardless of duration

---

## 3. ML Service Architecture (ENFORCED)

### Principle
ML services must be modular, lazy-loaded, and individually callable.

### Requirements
- **Lazy model loading** - Load models only when needed
- **Individual service methods** - Each method runs ONLY its service
- **Native async** - Use asyncio.to_thread, NOT custom thread wrappers
- **Timeout protection** - 10-minute max for any operation
- **Error recovery** - Retry logic with exponential backoff

### Correct Implementation
```python
async def run_yolo_detection(video_path, output_dir):
    # This runs ONLY YOLO, not all services
    frames = await extract_frames(video_path)
    return await run_yolo_on_frames(frames)

async def run_all_ml_services(video_path, output_dir):
    # This is the ONLY method that runs everything
    frames = await extract_frames(video_path)
    results = await asyncio.gather(
        run_yolo_on_frames(frames),
        run_mediapipe_on_frames(frames),
        run_ocr_on_frames(frames),
        run_whisper_on_video(video_path)
    )
```

---

## 4. 6-Block Output Format (REQUIRED FOR CLAUDE)

### Principle
All Claude responses MUST follow the 6-block structure for ML-friendly parsing.

### Requirements
Each prompt response contains exactly 6 blocks:
1. **CoreMetrics** - Key measurements and summary
2. **Dynamics** - Temporal patterns and transitions
3. **Interactions** - Cross-modal relationships
4. **KeyEvents** - Critical moments with timestamps
5. **Patterns** - Recurring elements and signatures
6. **Quality** - Technical and creative assessment

### Validation
- ResponseValidator MUST validate all Claude responses
- Missing blocks = failed response
- Empty blocks = failed response
- Blocks under 500 chars total = likely incomplete

### Backward Compatibility
- OutputAdapter converts 6-block to legacy format when needed
- NEVER modify 6-block structure itself
- Always preserve original 6-block response

---

## 5. Temporal Marker System (ESSENTIAL)

### Principle
Temporal markers provide semantic understanding of video progression.

### Requirements
- **Generated from unified analysis** - Not from raw ML data
- **Confidence thresholds** - Only high-confidence markers
- **Type classification**:
  - scene_change
  - emotional_peak
  - action_moment
  - text_appearance
  - speech_segment
- **Saved separately** for compatibility

---

## 6. Cost Control Architecture (CRITICAL)

### Principle
Every Claude API call must be justified by value and optimized for cost.

### Requirements
- **Precompute functions** extract only necessary data
- **Prompt size limits** - Validate before sending
- **Batch processing** - Group related prompts
- **Result caching** - Never reprocess same video
- **Feature flags** for expensive operations

---

## 7. Error Handling Philosophy (NON-NEGOTIABLE)

### Principle
The system must degrade gracefully, never fail completely.

### Requirements
- **Service isolation** - One service failure doesn't crash pipeline
- **Fallback strategies**:
  1. Try primary method
  2. Retry with exponential backoff
  3. Try alternative method
  4. Return degraded results
  5. Mark as degraded but continue
- **Empty results** are valid (but marked as failed)
- **NEVER throw unhandled exceptions** to Node.js layer

### Example
```python
try:
    frames = await extract_frames_opencv(video)
except:
    try:
        frames = await extract_frames_ffmpeg(video)
    except:
        frames = await extract_minimal_frames(video, count=10)
        mark_as_degraded()
```

---

## 8. Data Flow Integrity (MANDATORY)

### Principle
Data must flow unidirectionally through well-defined stages.

### Pipeline Stages
```
1. Video Input
   ↓
2. Frame Extraction (ONCE)
   ↓
3. ML Analysis (PARALLEL)
   ↓
4. Unified Analysis Assembly
   ↓
5. Temporal Marker Generation
   ↓
6. Precompute Functions
   ↓
7. Claude Prompts
   ↓
8. 6-Block Response Validation
   ↓
9. Output Storage
```

### Rules
- **No backward data flow**
- **No skipping stages**
- **Each stage validates its inputs**
- **Each stage provides typed outputs**

---

## 9. Testing Requirements (MANDATORY)

### Principle
Every component must be testable in isolation.

### Requirements
- **CLI interfaces** for all major components
- **Argument parsing** not hardcoded paths
- **Progress output** for long operations
- **Deterministic outputs** where possible
- **Performance metrics** in test output

### Test Coverage
- Frame extraction with various video formats
- Each ML service independently
- Unified pipeline end-to-end
- Cache eviction behavior
- Error recovery paths
- Memory limit compliance

---

## 10. Backward Compatibility (CRITICAL)

### Principle
New implementations must not break existing Node.js integrations.

### Requirements
- **Exit codes** must match expectations:
  - 0: Success
  - 1: General failure
  - 2: Invalid arguments
  - 3: API failure
  - 4: ML processing failure
- **Output formats** support both v1 and v2
- **File paths** remain consistent
- **API contracts** unchanged

---

## 11. Documentation Standards (REQUIRED)

### Principle
Code behavior must be self-documenting and verifiable.

### Requirements
- **Docstrings** for all public methods
- **Type hints** for all parameters and returns
- **Constants** for magic numbers
- **Configuration** externalized, not hardcoded
- **Logging** at key decision points

---

## Common Violations to Avoid

### ❌ NEVER DO THIS:
1. Process video multiple times for different services
2. Load all models at startup regardless of need
3. Keep unlimited frames in memory
4. Run all services when only one is requested
5. Ignore frame extraction failures
6. Send unvalidated prompts to Claude
7. Mix v1 and v2 output formats
8. Hardcode paths or credentials
9. Catch exceptions without logging
10. Assume operations will complete quickly
11. Use inconsistent block naming across prompts (all must use CoreMetrics, Dynamics, etc.)
12. Call non-existent API methods (verify method names match implementation)

### ✅ ALWAYS DO THIS:
1. Extract frames once, share everywhere
2. Load models lazily when needed
3. Implement cache limits and eviction
4. Run only requested services
5. Implement retry logic with fallbacks
6. Validate prompt size and structure
7. Use OutputAdapter for format conversion
8. Use environment variables and configs
9. Log errors with context
10. Implement timeouts on all operations

---

## Enforcement

These principles are enforced through:
1. **Code reviews** - PRs must demonstrate compliance
2. **Automated tests** - CI/CD validates requirements
3. **Performance monitoring** - Production metrics tracked
4. **Cost monitoring** - Claude API costs tracked per video

---

## Implementation Status (2025-08-05)

✅ **Fully Implemented**:
- Unified Frame Manager with LRU cache (unified_frame_manager.py)
- ML Services with lazy loading (ml_services_unified.py)
- Async Whisper transcription (whisper_transcribe_safe.py)
- ML data field in UnifiedAnalysis (analysis.py lines 126-142)
- Individual service methods (ml_services.py)
- Standardized prompt block naming (fixed person_framing_v2.txt)
- OCR service integration (fixed method name in video_analyzer.py)

✅ **Verified Through Testing**:
- ML data field appears correctly in unified_analysis JSON
- End-to-end data flow from ML → precompute → Claude works
- All 7 Claude prompts receive real ML data and return valid 6-block structures
- Frame extraction happens only once per video (verified with test)
- Cost tracking functional ($0.0097 for complete 7-prompt analysis)

## Conclusion

These architectural principles are the foundation of RumiAI's reliability, efficiency, and maintainability. They are non-negotiable because:

1. **Cost Control**: Violating these principles wastes money on API calls
2. **Performance**: The system becomes unusably slow without optimization
3. **Reliability**: Production systems need predictable behavior
4. **Maintainability**: Future developers need clear patterns
5. **Scalability**: The system must handle growth

Every line of code should respect these principles. When in doubt, refer to this document.