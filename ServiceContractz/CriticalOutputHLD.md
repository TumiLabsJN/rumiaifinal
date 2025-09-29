# Critical Output Validation Contract - High Level Design (HLD)

**Version**: 1.0
**Last Updated**: January 2025
**Status**: Design Phase
**Scope**: ML Service Output Validation for RumiAI Pipeline

## 1. Executive Summary

### 1.1 Purpose
The Critical Output Validation Contract ensures data quality, structural integrity, and logical boundaries of ML service outputs before they propagate through the RumiAI pipeline. This prevents cascade failures, data corruption, and subtle boundary violations that could affect batch processing of 60-300 videos.

### 1.2 Problem Statement
Currently, the RumiAI pipeline has no systematic validation of ML service outputs, leading to:
- Silent data corruption that propagates downstream
- Difficult debugging when failures occur deep in the pipeline
- Wasted processing time on videos with malformed data
- Unreliable ML training data quality

### 1.3 Solution Overview
Implement a validation layer that intercepts ML service outputs and validates both their structure and logical boundaries against service-specific contracts before allowing pipeline continuation. This includes verifying timestamps are within video duration, confidence scores are normalized, and coordinates are within frame boundaries.

---

## 2. Architectural Context

### 2.1 Current Pipeline Flow
```
Video → ML Service → [No Validation] → Timeline Builder → Temporal Compute → ML Training
                            ↑
                     Problem: Invalid data propagates
```

### 2.2 Proposed Pipeline Flow
```
Video → ML Service → [Validation Contract] → Timeline Builder → Temporal Compute → ML Training
           ↓                    ↓
    Video Metadata      Structure + Boundaries
    (duration, dims)         Validated
                            ↓
                    Valid: Continue
                    Invalid: Fail-fast with clear error
```

### 2.3 Integration Points
- **Location**: Between ML service execution and result storage
- **Timing**: Immediately after each service completes
- **Scope**: All 8 ML services (YOLO, Whisper, MediaPipe, OCR, Scene Detection, Audio Energy, FEAT, DeepFace)
- **Inputs**: Service output + video metadata (duration, width, height, fps)

---

## 3. Design Principles

### 3.1 Core Principles
1. **Fail-Fast Within Video**: When validation fails for any service, stop processing that video immediately
2. **Clear Errors**: Provide actionable error messages with remediation
3. **Non-Invasive**: Wrap existing code without architectural changes
4. **Progressive**: Support both strict (production) and lenient (debugging) modes
5. **Observable**: Track validation metrics for monitoring
6. **Atomic Video Processing**: Each video either has all features or is marked failed

### 3.2 What We Validate (Structure + Boundaries Approach)
- **Structure**: Required fields exist with correct types
- **Completeness**: Critical data is present
- **Consistency**: Logical relationships are maintained (e.g., start < end)
- **Temporal Boundaries**: Timestamps within [0, video_duration]
- **Spatial Boundaries**: Coordinates within frame dimensions or normalized [0,1]
- **Value Boundaries**: Confidence scores within [0,1], counts match expected ranges

### 3.3 What We DON'T Validate
- **Accuracy**: Whether detected objects are correct
- **Quality**: Whether transcription is accurate
- **Performance**: Processing speed or resource usage
- **Business Logic**: Domain-specific rules
- **Content Presence**: Whether services found "interesting" data (empty results are valid)

---

## 4. Functional Requirements

### 4.1 Validation Capabilities

#### 4.1.1 Service-Specific Validation
Each ML service has unique output structure and boundaries requiring tailored validation. **Any validation failure stops the entire video processing**:

| Service | Structure Validations | Boundary Validations | Failure Impact |
|---------|----------------------|---------------------|----------------|
| YOLO | objectAnnotations list, trackId | timestamp ≤ duration, confidence ∈ [0,1], bbox within frame | **Stops video** - No features generated |
| Whisper | segments array, text | start/end ≤ duration, start < end | **Stops video** - Partial features discarded |
| MediaPipe | poses/faces/hands lists | landmarks ∈ [0,1], timestamp ≤ duration | **Stops video** - Partial features discarded |
| OCR | textAnnotations list | position within frame bounds | **Stops video** - Partial features discarded |
| Scene Detection | scene_segments | start/end ≤ duration, start < end | **Stops video** - Partial features discarded |
| Audio Energy | rms_frames array | frame_count ≈ duration×31 (±20%), rms ∈ [0,1] | **Stops video** - Partial features discarded |
| FEAT | emotion predictions | 7 values each ∈ [0,1] | **Stops video** - Partial features discarded |
| DeepFace | gender classification | confidence ∈ [0,1] | **Stops video** - Partial features discarded |

#### 4.1.2 Severity Levels
```
CRITICAL → Pipeline must stop (missing required structure, boundary violations)
WARNING  → Quality issue but can continue (missing optional data)
INFO     → Informational only (empty but structurally valid results)
```

**Empty Results Policy**: Services returning valid but empty data (e.g., YOLO finds no objects, Whisper detects no speech) are treated as **INFO level**. This is normal for TikTok's diverse content and provides valuable training data representing "nothing detected" scenarios.

#### 4.1.3 Validation Modes
- **Strict Mode** (Production): Warnings treated as critical
- **Lenient Mode** (Development): Warnings logged but allowed

### 4.2 Error Handling

#### 4.2.1 Response Structure Examples

**Critical/Warning Errors:**
```json
{
  "valid": false,
  "service": "yolo",
  "severity": "critical",
  "error_message": "Missing 'objectAnnotations' field",
  "remediation": "YOLO must return {'objectAnnotations': [], 'metadata': {...}}",
  "details": {
    "actual_keys": ["error", "timestamp"],
    "expected_keys": ["objectAnnotations", "metadata"]
  }
}
```

**INFO Level (Empty but Valid):**
```json
{
  "valid": true,
  "service": "yolo",
  "severity": "info",
  "message": "No objects detected (normal for text-only or abstract videos)",
  "details": {
    "result_type": "empty_valid",
    "objects_found": 0
  }
}
```

#### 4.2.2 Remediation Guidance
Each validation failure includes:
- What went wrong
- Why it's a problem
- How to fix it
- Example of correct structure

### 4.3 Metrics & Monitoring

#### 4.3.1 Tracked Metrics (First-Failure Tracking)
Due to fail-fast behavior, metrics focus on **where videos first fail** rather than per-service pass rates:
- Total videos processed
- Videos successfully completed (all services passed)
- First point of failure per video
- Most problematic service identification
- Validation performance overhead

#### 4.3.2 Reporting Structure
```json
{
  "summary": {
    "total_videos": 300,
    "successful_videos": 250,
    "failed_videos": 50,
    "success_rate": 0.833,
    "failure_points": {
      "yolo": 15,           // 15 videos failed at YOLO (never ran other services)
      "whisper": 10,        // 10 videos failed at Whisper
      "mediapipe": 5,       // 5 videos failed at MediaPipe
      "audio_energy": 20    // 20 videos failed at Audio Energy
    },
    "most_problematic_service": "audio_energy",
    "avg_validation_time_ms": 3.2
  }
}
```

**Note**: Service metrics don't show pass/fail rates because fail-fast means later services run fewer times. Instead, we track which service causes videos to fail.

#### 4.3.3 Why First-Failure Tracking
Traditional per-service metrics are misleading with fail-fast:
- If YOLO fails, Whisper never runs → Whisper appears to have 100% success
- Services later in pipeline always show better metrics (survivorship bias)
- First-failure tracking shows the **true bottlenecks** in your pipeline

This approach answers the critical question: "Which service is blocking the most videos from completing?"

---

## 5. Non-Functional Requirements

### 5.1 Performance
- **Overhead**: < 10ms per validation
- **Memory**: < 10MB for validation logic
- **Scalability**: Support 300+ videos in batch

### 5.2 Reliability
- **Availability**: No external dependencies
- **Fault Tolerance**: Validation failures don't crash pipeline
- **Recovery**: Clear state for retry attempts

### 5.3 Maintainability
- **Extensibility**: Easy to add new services
- **Configuration**: Validation rules configurable
- **Testing**: Unit testable validation logic

---

## 6. Use Cases

### 6.1 Single Video Processing (Fail-Fast)
```
1. Video processed by YOLO
2. YOLO returns result
3. Validator checks result structure and boundaries
4. If valid → Continue to Whisper
5. If invalid → Log error, stop processing this video
6. No further services (Whisper, MediaPipe, etc.) are executed
7. Video marked as failed with clear reason
```

### 6.2 Batch Processing (300 Videos)
```
1. Start processing video N
2. Run YOLO → Validate output
3. If invalid → Stop video N (don't run other services), record "failed_at: yolo", go to N+1
4. If valid → Continue with Whisper → Validate
5. If any service fails → Stop video N immediately, record failure point, go to N+1
6. If all services pass → Mark video N complete, go to N+1
7. After batch → Report first-failure statistics (where videos failed, not service pass rates)
```

### 6.3 ML Training Pipeline
```
1. Load 60 videos for duration bucket
2. Check which videos completed all validations
3. Failed videos (incomplete features) → Exclude entirely
4. Successful videos (all features valid) → Include in training
5. Train model with complete, validated videos only
6. Report: X complete videos used, Y failed videos excluded
```

---

## 7. Validation Rules by Service

### 7.1 YOLO Object Detection
**Purpose**: Ensure object tracking data structure and boundaries
```
Structure Requirements:
- objectAnnotations: array (can be empty)
- metadata.frames_analyzed: number

Each annotation needs (if any):
- className: string
- confidence: float
- timestamp: float
- trackId: string (optional but recommended)
- bbox: [x, y, width, height]

Boundary Requirements:
- timestamp ≤ video.duration
- 0 ≤ confidence ≤ 1
- bbox.x + bbox.width ≤ video.width
- bbox.y + bbox.height ≤ video.height

Empty Results: objectAnnotations: [] → INFO (normal for abstract/text videos)
```

### 7.2 Whisper Speech Transcription
**Purpose**: Ensure speech segments structure and temporal validity
```
Structure Requirements:
- segments: array (can be empty)

Each segment needs (if any):
- start: float
- end: float
- text: string
- confidence: float (optional)

Boundary Requirements:
- 0 ≤ start ≤ video.duration
- 0 ≤ end ≤ video.duration
- start < end (temporal ordering)
- 0 ≤ confidence ≤ 1 (if present)

Empty Results: segments: [] → INFO (normal for music-only videos)
```

### 7.3 MediaPipe Human Analysis
**Purpose**: Ensure multi-modal human detection with valid coordinates
```
Structure Requirements:
- poses: array (can be empty)
- faces: array (can be empty)
- hands: array (can be empty)
- gestures: array (can be empty)
- gaze: array (can be empty)

Boundary Requirements:
- All landmarks: x, y ∈ [0, 1] (normalized)
- timestamp ≤ video.duration
- confidence ∈ [0, 1] (if present)

Empty Results: All arrays empty → INFO (normal for object-only videos)
```

### 7.4 Audio Energy Analysis
**Purpose**: Ensure audio frame data with correct alignment
```
Structure Requirements:
- rms_frames: array of floats
- duration: float

Boundary Requirements:
- 0 ≤ rms_value ≤ 1 for each frame
- Frame count ≈ duration × 31 (±20% tolerance)
- |actual_frames - expected_frames| / expected_frames < 0.2
```

### 7.5 FEAT Emotion Detection
**Purpose**: Ensure emotion predictions with valid probabilities
```
Structure Requirements:
- predictions: array (can be empty if no faces)

Each prediction needs (if any):
- 7 emotions: anger, disgust, fear, joy, sadness, surprise, neutral

Boundary Requirements:
- 0 ≤ emotion_value ≤ 1 for each emotion
- Sum of emotions should ≈ 1.0 (±0.1 tolerance for rounding)

Empty Results: predictions: [] → INFO (normal when no faces detected)
```

---

## 8. Success Criteria

### 8.1 Immediate Success (Week 1)
- ✅ Catches 80% of structural errors
- ✅ Catches 90% of boundary violations
- ✅ Correctly identifies empty results as valid (INFO level)
- ✅ Provides clear error messages with valid ranges
- ✅ No false positives on valid data or normal empty results

### 8.2 Short-term Success (Month 1)
- ✅ 50% reduction in debugging time
- ✅ 95% of videos complete all services successfully
- ✅ Clear identification of most problematic services via failure-point metrics
- ✅ ML training uses only fully-validated videos
- ✅ No partial feature sets in training data

### 8.3 Long-term Success (Quarter 1)
- ✅ Validation becomes standard practice
- ✅ New services automatically include validation
- ✅ Historical validation data informs service improvements

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Over-validation | False positives block valid data | Start with shadow mode, tune boundaries |
| Boundary check overhead | Additional validation time | <5ms per check, metadata already available |
| Maintenance burden | Rules become outdated | Document thoroughly, version contracts |
| Service evolution | Output formats change | Backward-compatible validation with deprecation warnings |

---

## 10. Implementation Phases

### Phase 1: Core Validation (Week 1)
- Implement structure + boundary validation
- Cover YOLO, Whisper, MediaPipe with full boundary checks
- Deploy in shadow mode for observation

### Phase 2: Complete Coverage (Week 2)
- Add remaining services
- Implement severity levels
- Add first-failure metrics tracking
- Identify bottleneck services

### Phase 3: Production Hardening (Week 3)
- Enable strict mode
- Add monitoring dashboards
- Document common failures

### Phase 4: ML Integration (Week 4)
- Ensure only fully-validated videos reach ML pipeline
- Report complete vs failed video counts
- Generate data quality reports

---

## 11. Service Evolution Strategy

### 11.1 Backward Compatible Validation
When services change output formats, validators maintain backward compatibility:

```python
# Example: YOLO format change
def validate_yolo(result, metadata):
    # Try new format first (preferred)
    if "detections" in result:  # v2 format
        return validate_v2_format(result, metadata)

    # Support old format with deprecation warning
    if "objectAnnotations" in result:  # v1 format
        logger.warning(f"YOLO using deprecated v1 format, will be removed {deprecation_date}")
        return validate_v1_format(result, metadata)

    # Unrecognized format
    raise ValidationError("Unrecognized YOLO output format")
```

### 11.2 Migration Process
1. **Week 1-2**: Add new format support, maintain old format
2. **Week 3-4**: Log warnings for old format usage
3. **Week 5-6**: Increase warning severity, alert teams
4. **Week 7**: Remove old format support (with notice)

### 11.3 Benefits of Backward Compatibility
- **Zero downtime** during format transitions
- **Services upgrade independently** without coordination
- **Clear deprecation path** with warnings before removal
- **Graceful degradation** if services roll back
- **Single validator codebase** instead of version proliferation

---

## 12. Processing Flow Clarification

### 12.1 Fail-Fast Behavior
The validation contract enforces **fail-fast within each video**:
```python
# Pseudocode showing fail-fast behavior
for video in video_list:  # Sequential processor (not this contract)
    try:
        # This contract's scope starts here
        yolo_result = run_yolo(video)
        validate(yolo_result)  # Fails here? Stop this video

        whisper_result = run_whisper(video)
        validate(whisper_result)  # Fails here? Stop this video

        # ... continue through all services
        mark_video_complete(video)

    except ValidationError as e:
        mark_video_failed(video, failed_at_service=e.service)
        # DON'T run remaining services for this video
        continue  # Sequential processor moves to next video
```

### 12.2 Data Consistency Guarantee
This fail-fast approach with global mode ensures:
- **No partial feature sets**: Videos have either ALL features or NONE
- **Uniform quality**: All features validated at same strictness level
- **Clear failure attribution**: Metrics show exactly which service causes most failures
- **Predictable ML input**: Training data always has complete, uniformly-validated feature vectors
- **Efficient compute usage**: Don't waste resources on known-bad videos
- **Simple configuration**: One mode setting, no per-service complexity
- **Actionable metrics**: First-failure tracking identifies bottleneck services

---

## 13. Dependencies

### 13.1 Technical Dependencies
- Python 3.8+ (type hints, dataclasses)
- No external libraries required
- Access to service output samples
- Video metadata (duration, width, height, fps) - already available

### 13.2 Configuration Requirements
- Single global mode setting (strict/lenient) for entire pipeline
- All services validate at same strictness level
- Mode can be changed via configuration but applies to all services uniformly

### 13.3 Organizational Dependencies
- Agreement on validation rules
- Service team cooperation
- Monitoring infrastructure

---

## 14. Appendix

### 14.1 Sample Validation Flow
```python
# Pseudocode
result = ml_service.process(video)
validation = validator.validate(service_name, result)

if validation.severity == CRITICAL:
    raise PipelineError(validation.error_message)
elif validation.severity == WARNING:
    logger.warning(validation.error_message)
    if strict_mode:
        raise PipelineError(validation.error_message)

# Continue with valid data
timeline_builder.add(result)
```

### 14.2 References
- [RumiAI System Architecture](../documentation_migration/services/SystemArchitecturev2.md)
- [ML Roadmap](../MLROADMAP.md)
- [Service Contracts v2](../ServiceContractsv2.md)

---

## Document History
- v1.0 (2025-01-26): Initial HLD created