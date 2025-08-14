# Person Framing Critical Fixes - Gaze Detection & Face Distance

## Executive Summary

Person framing has **two equally critical issues** that need fixing:
1. **Missing Gaze Detection**: `eyeContactRate: 0` - No eye contact analysis implemented
2. **Missing Face Distance**: `averageFaceSize: 0` - No camera distance classification

Both issues have working infrastructure but missing data generation.

## Problem Analysis

### What's Working ✅
- **MediaPipe face detection**: 59/60 frames detected with high confidence (0.7-0.9)
- **Timeline builder (partial)**: Has code for gaze entries, missing code for face entries
- **Person framing computation**: Ready to consume both metrics once data flows

### Two Critical Missing Components ❌

#### Problem 1: No Gaze Detection
- **Impact**: `eyeContactRate: 0`, no `gazeSteadiness` metric
- **Root Cause**: MediaPipe doesn't process eye/iris landmarks
- **Result**: Empty gaze array, no eye contact analysis

#### Problem 2: No Face Size/Distance Data  
- **Impact**: `averageFaceSize: 0`, wrong camera distance classification
- **Root Cause**: FEAT doesn't provide face bbox, MediaPipe faces not used for sizing
- **Result**: Camera distance always 'wide', incorrect shot type classification

## Technical Analysis

### Current MediaPipe Implementation (Gaze Missing)
**Location**: `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py:_process_mediapipe_batch`

```python
def _process_mediapipe_batch(self, models, frames: List[FrameData]) -> Dict:
    """Process a batch of frames with MediaPipe (sync)"""
    poses = []
    faces = []  # ← Face detection exists but no gaze analysis
    hands = []
    
    for frame_data in frames:
        # Process pose ✅
        # Process face ✅ (but no gaze)
        # Process hands ✅
        # Process gaze ❌ MISSING
        
    return {'poses': poses, 'faces': faces, 'hands': hands}
    # Missing: 'gaze': gaze_data
```

### Technical Details: What's Actually Missing

1. **Gaze Processing**: MediaPipe doesn't run FaceMesh, so no iris landmarks generated
2. **Face Bbox Capture**: MediaPipe face detection runs but bbox not saved to output
3. **Timeline Builder Gap**: No code to convert face detections to timeline entries (only handles poses)

**Note**: Timeline builder has gaze entry code (`timeline_builder.py:261-278`) but it receives empty array because MediaPipe never generates gaze data.

## Architectural Understanding: Service Separation

### Correct Service Responsibilities for Person Framing:
1. **FEAT**: Emotion detection only (NOT face bbox)
2. **MediaPipe Face Detection**: Face presence, bbox, camera distance
3. **MediaPipe FaceMesh**: Gaze detection and eye contact

### Integration with Existing FEAT Service:
- **FEAT continues unchanged**: Still provides emotion detection and action units
- **FEAT bbox expectation removed**: Stop looking for face_bbox in FEAT output
- **MediaPipe provides face metrics**: Face size, visibility, camera distance
- **Data precedence**: 
  - Emotions → FEAT (authoritative)
  - Face presence/size → MediaPipe (authoritative)
  - When both detect faces → Use FEAT for emotions, MediaPipe for spatial metrics
- **No conflicts**: Each service handles distinct metrics, no overlap

## Solutions: Two Parallel Fixes Required

Both fixes are equally important and should be implemented together for complete person framing functionality.

## Fix 1: Add Gaze Detection Using FaceMesh

### Step 1A: Add FaceMesh Model Loading
**Location**: `ml_services_unified.py:_ensure_model_loaded`

```python
async def _ensure_model_loaded(self, service_name: str):
    if service_name == 'mediapipe':
        if not self.mediapipe_models:
            import mediapipe as mp
            
            # Fail-fast: Let initialization errors bubble up
            # No try/except - if FaceMesh fails, the service should crash
            self.mediapipe_models = {
                'pose': mp.solutions.pose.Pose(),
                'face': mp.solutions.face_detection.FaceDetection(),
                'hands': mp.solutions.hands.Hands(),
                'face_mesh': mp.solutions.face_mesh.FaceMesh(  # ← ADD THIS
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            }
            
            # Verify FaceMesh loaded successfully (fail-fast)
            if not self.mediapipe_models['face_mesh']:
                raise RuntimeError("FaceMesh failed to initialize - failing fast")
```

### Step 1B: Add Gaze Detection to Batch Processing
**Location**: `ml_services_unified.py:_process_mediapipe_batch`

```python
def _process_mediapipe_batch(self, models, frames: List[FrameData]) -> Dict:
    poses = []
    faces = []
    hands = []
    gaze_data = []  # ← ADD THIS
    
    for frame_data in frames:
        # Existing processing...
        
        # ADD: Process gaze using FaceMesh
        if models['face_mesh']:
            mesh_results = models['face_mesh'].process(rgb_frame)
            if mesh_results.multi_face_landmarks:
                gaze_info = self._compute_gaze_from_mesh(
                    mesh_results.multi_face_landmarks[0], 
                    frame_data.timestamp
                )
                if gaze_info:
                    gaze_data.append(gaze_info)
    
    return {
        'poses': poses, 
        'faces': faces, 
        'hands': hands,
        'gaze': gaze_data  # ← ADD THIS
    }
```

### Step 1C: Implement Gaze Computation Helper
**Location**: `ml_services_unified.py` (new method)

```python
def _compute_gaze_from_mesh(self, face_landmarks, timestamp):
    """Compute gaze direction from MediaPipe FaceMesh landmarks
    
    Note: Requires refine_landmarks=True for iris detection (landmarks 468-477)
    """
    
    # VERIFIED MediaPipe FaceMesh landmark indices
    # Iris landmarks (only available with refine_landmarks=True)
    LEFT_IRIS_CENTER = 468   # Center of left iris (landmark 468)
    RIGHT_IRIS_CENTER = 473  # Center of right iris (landmark 473)
    
    # Eye corner landmarks for calculating eye center
    LEFT_EYE_INNER = 133     # Inner corner of left eye
    LEFT_EYE_OUTER = 33      # Outer corner of left eye  
    RIGHT_EYE_INNER = 362    # Inner corner of right eye
    RIGHT_EYE_OUTER = 263    # Outer corner of right eye
    
    try:
        # Get iris positions (requires refine_landmarks=True)
        left_iris = face_landmarks.landmark[LEFT_IRIS_CENTER]
        right_iris = face_landmarks.landmark[RIGHT_IRIS_CENTER]
        
        # Calculate eye centers from eye corners
        left_eye_inner = face_landmarks.landmark[LEFT_EYE_INNER]
        left_eye_outer = face_landmarks.landmark[LEFT_EYE_OUTER]
        left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2
        left_eye_center_y = (left_eye_inner.y + left_eye_outer.y) / 2
        
        right_eye_inner = face_landmarks.landmark[RIGHT_EYE_INNER]
        right_eye_outer = face_landmarks.landmark[RIGHT_EYE_OUTER]
        right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2
        right_eye_center_y = (right_eye_inner.y + right_eye_outer.y) / 2
        
        # Calculate gaze offset: iris position relative to eye center
        left_gaze_offset_x = left_iris.x - left_eye_center_x
        left_gaze_offset_y = left_iris.y - left_eye_center_y
        right_gaze_offset_x = right_iris.x - right_eye_center_x
        right_gaze_offset_y = right_iris.y - right_eye_center_y
        
        # Average horizontal and vertical offsets
        avg_gaze_offset_x = (left_gaze_offset_x + right_gaze_offset_x) / 2
        avg_gaze_offset_y = (left_gaze_offset_y + right_gaze_offset_y) / 2
        
        # Calculate total gaze offset magnitude (both axes)
        gaze_magnitude = (avg_gaze_offset_x**2 + avg_gaze_offset_y**2) ** 0.5
        
        # Classify eye contact based on offset magnitude
        # Small offset = looking at camera, large offset = looking away
        eye_contact_threshold = 0.02  # Tunable parameter (in normalized coordinates)
        is_looking_at_camera = gaze_magnitude < eye_contact_threshold
        
        # Determine gaze direction (considering both axes)
        if is_looking_at_camera:
            gaze_direction = 'camera'
            eye_contact_score = max(0, 1.0 - gaze_magnitude / eye_contact_threshold)
        else:
            # Determine primary gaze direction
            if abs(avg_gaze_offset_x) > abs(avg_gaze_offset_y):
                # Horizontal gaze is stronger
                gaze_direction = 'right' if avg_gaze_offset_x > 0 else 'left'
            else:
                # Vertical gaze is stronger
                gaze_direction = 'down' if avg_gaze_offset_y > 0 else 'up'
            eye_contact_score = 0.0
        
        return {
            'timestamp': timestamp,
            'eye_contact': eye_contact_score,
            'gaze_direction': gaze_direction,
            'gaze_offset_x': avg_gaze_offset_x,
            'gaze_offset_y': avg_gaze_offset_y,
            'gaze_magnitude': gaze_magnitude,
            'confidence': 0.8  # Can be enhanced with landmark visibility scores
        }
        
    except (IndexError, AttributeError):
        # No face/landmarks detected - return None (not an error)
        # Calling code will handle empty gaze data
        return None
```

### Step 1D: Update Result Structure
**Location**: `ml_services_unified.py:_run_mediapipe_on_frames`

```python
async def _run_mediapipe_on_frames(self, frames, video_id, output_dir):
    # ... existing batch processing ...
    
    all_poses = []
    all_faces = []
    all_hands = []
    all_gaze = []  # ← ADD THIS
    
    for i in range(0, len(mp_frames), batch_size):
        batch_results = await asyncio.to_thread(
            self._process_mediapipe_batch, models, batch
        )
        
        all_poses.extend(batch_results['poses'])
        all_faces.extend(batch_results['faces'])
        all_hands.extend(batch_results['hands'])
        all_gaze.extend(batch_results.get('gaze', []))  # ← ADD THIS
    
    result = {
        'poses': all_poses,
        'faces': all_faces,
        'hands': all_hands,
        'gaze': all_gaze,  # ← ADD THIS
        'gestures': [],
        'presence_percentage': (len(all_poses) / len(mp_frames) * 100) if mp_frames else 0,
        'frames_with_people': len(all_poses),
        'metadata': {
            'frames_analyzed': len(mp_frames),
            'processed': True,
            'poses_detected': len(all_poses),
            'faces_detected': len(all_faces),
            'hands_detected': len(all_hands),
            'gaze_detected': len(all_gaze)  # ← ADD THIS
        }
    }
```

## Fix 2: Face Size/Camera Distance Using MediaPipe Bbox

### Step 2A: Capture Bbox Data from MediaPipe
**Location**: `ml_services_unified.py:_process_mediapipe_batch`

MediaPipe Face Detection provides bbox data that we're not capturing:

```python
# Process face with bbox extraction
if models['face']:
    face_results = models['face'].process(rgb_frame)
    if face_results.detections:
        detection = face_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        faces.append({
            'timestamp': frame_data.timestamp,
            'frame_number': frame_data.frame_number,
            'count': len(face_results.detections),
            'confidence': detection.score[0],
            'bbox': {  # ADD THIS
                'x': bbox.xmin,
                'y': bbox.ymin,
                'width': bbox.width,    # Fraction of frame width
                'height': bbox.height   # Fraction of frame height
            }
        })
```

### Step 2B: Enhance PersonTimeline with Face Data
**Location**: `precompute_functions.py:_extract_timelines_from_analysis`

Instead of adding a new parameter, enhance the existing personTimeline to include face bbox data:

```python
def _extract_timelines_from_analysis(analysis_dict):
    # ... existing code ...
    
    # Transform poses to timeline (existing code around line 443)
    for pose in mediapipe_data.get('poses', []):
        timestamp = pose.get('timestamp', 0)
        timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
        timelines['personTimeline'][timestamp_key] = {
            'detected': True,
            'pose_confidence': pose.get('confidence', 0.8),
            'face_bbox': None,  # Will be filled if face detected at same time
            'face_confidence': None
        }
    
    # ADD THIS: Merge face data into personTimeline
    for face in mediapipe_data.get('faces', []):
        timestamp = face.get('timestamp', 0)
        timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
        
        # Create or update personTimeline entry with face data
        if timestamp_key in timelines['personTimeline']:
            # Person already detected via pose, add face data
            timelines['personTimeline'][timestamp_key]['face_bbox'] = face.get('bbox')
            timelines['personTimeline'][timestamp_key]['face_confidence'] = face.get('confidence')
        else:
            # Face detected but no pose, create entry
            timelines['personTimeline'][timestamp_key] = {
                'detected': True,
                'pose_confidence': None,
                'face_bbox': face.get('bbox'),
                'face_confidence': face.get('confidence', 0)
            }
```

**Why This is Better**: 
- Maintains architectural consistency - personTimeline becomes the single source for all person-related detections (poses AND faces)
- No need for separate faceTimeline - faces are part of person detection
- Timeline entries ARE created for faces, just merged into existing personTimeline

### Step 2C: Use Real Bbox for Camera Distance
**Location**: `precompute_functions_full.py:compute_person_framing_metrics`

```python
def compute_person_framing_metrics(expression_timeline, object_timeline, camera_distance_timeline,
                                  person_timeline, enhanced_human_data, duration, gaze_timeline=None):
    # NO NEW PARAMETER NEEDED - face data now in person_timeline
    
    # Extract face bbox data from enhanced personTimeline
    face_sizes = []
    face_frames = 0
    
    for timestamp, person_data in person_timeline.items():
        if person_data.get('face_bbox'):
            face_frames += 1
            bbox = person_data['face_bbox']
            # Calculate face area as percentage of frame
            face_area = bbox['width'] * bbox['height'] * 100
            face_sizes.append(face_area)
    
    # If no valid bboxes found, default to 0
    avg_face_size = sum(face_sizes) / len(face_sizes) if face_sizes else 0
    
    # Use real percentages for classification
    if avg_face_size > 25:  # Face >25% of frame = close-up
        avg_camera_distance = 'close-up'
        dominant_shot_type = 'close_up'
    elif avg_face_size > 10:  # Face 10-25% = medium
        avg_camera_distance = 'medium'
        dominant_shot_type = 'medium_shot'
    elif avg_face_size > 0:  # Face <10% = wide
        avg_camera_distance = 'wide'
        dominant_shot_type = 'wide_shot'
    else:  # No face size data
        avg_camera_distance = 'unknown'
        dominant_shot_type = 'unknown'
```

### Step 2D: Add Gaze Steadiness to Professional Wrapper
**Location**: `precompute_professional_wrappers.py:convert_to_person_framing_professional`

The professional wrapper needs to include `gazeSteadiness`:

```python
def convert_to_person_framing_professional(basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "personFramingCoreMetrics": {
            "eyeContactRate": basic_metrics.get('eye_contact_rate', 0),
            # ... other fields
        },
        "personFramingDynamics": {
            "gazeSteadiness": basic_metrics.get('gaze_steadiness', 'unknown'),  # ← ADD THIS
            "framingProgression": basic_metrics.get('framing_progression', []),
            # ... other fields
        }
    }
```

## Testing & Validation

### Test Both Fixes Together

#### Gaze Detection Tests
1. **Eye Contact**: Person looking at camera → `eyeContactRate > 0.5`
2. **Gaze Direction**: Person looking away → `gaze_direction: 'left'/'right'`
3. **Mixed Patterns**: Varied gaze → `gazeSteadiness: 'medium'`

#### Face Distance Tests
1. **Close-up**: High confidence faces → `averageFaceSize > 30`, `cameraDistance: 'close-up'`
2. **Medium Shot**: Medium confidence → `averageFaceSize: 15-30`, `cameraDistance: 'medium'`
3. **Wide Shot**: Low confidence → `averageFaceSize < 15`, `cameraDistance: 'wide'`

### Debug Script

```python
#!/usr/bin/env python3
"""Test gaze detection implementation"""
import json
from pathlib import Path

# Load MediaPipe output
mp_file = Path('ml_outputs/VIDEO_ID_human_analysis.json')
if mp_file.exists():
    with open(mp_file, 'r') as f:
        mp_data = json.load(f)
    
    gaze_data = mp_data.get('gaze', [])
    print(f"Gaze detections: {len(gaze_data)}")
    
    if gaze_data:
        eye_contacts = [g for g in gaze_data if g['eye_contact'] > 0.5]
        print(f"Eye contact frames: {len(eye_contacts)}/{len(gaze_data)}")
        print(f"Eye contact rate: {len(eye_contacts)/len(gaze_data):.2f}")
        
        # Show sample gaze data
        for i, gaze in enumerate(gaze_data[:3]):
            print(f"Frame {i}: {gaze}")
    else:
        print("❌ No gaze data detected")
```

## Error Handling Strategy

### Fail-Fast Philosophy

**Initialization Errors → Fail Fast:**
- If FaceMesh fails to load: **Crash immediately** (don't hide problems)
- Model loading errors should bubble up to surface configuration issues

**Runtime Missing Data → Blank Output:**
- No faces detected: Return empty/zero metrics (not an error)
- No landmarks found: Return None for gaze data
- Missing bbox: Use 0 for face size

**This approach:**
1. **Surfaces problems early** during development
2. **Handles normal cases** (no people in frame) gracefully
3. **Avoids silent failures** that hide bugs

### Performance Impact

**Negligible Impact - No Optimization Needed:**

Based on FRAME_PROCESSING_PYTHON_ONLY.md:
- **Current processing**: 3 FPS for 60s videos = 180 frames total
- **FaceMesh overhead**: ~15-30ms per frame
- **Total additional time**: 2.7-5.4 seconds
- **Impact on 10-15 minute processing**: **<1%**

**Simple Early-Exit Optimization:**
```python
# Only run FaceMesh when face is detected
if face_results.detections:
    if models.get('face_mesh'):
        mesh_results = models['face_mesh'].process(rgb_frame)
        # Process gaze
# No face = skip FaceMesh automatically
```

**Why Frame Sampling is NOT Needed:**
1. Already at low FPS (3 FPS from adaptive sampling)
2. Impact is negligible (<1% of total time)
3. Would reduce gaze accuracy
4. MediaPipe needs temporal consistency

**Conclusion**: The performance impact is so small it doesn't require special handling beyond early-exit when no face is detected.

## Expected Results

### Before Fix
```json
{
  "personFramingCoreMetrics": {
    "eyeContactRate": 0,
    "faceVisibilityRate": 0
  }
}
```

### After Fix
```json
{
  "personFramingCoreMetrics": {
    "eyeContactRate": 0.68,
    "faceVisibilityRate": 0.85
  },
  "personFramingDynamics": {
    "gazeSteadiness": "medium"
  }
}
```

## Implementation Summary

### Both Fixes Required (Parallel Implementation)

#### Fix 1: Gaze Detection (25 minutes)
1. Add FaceMesh model loading
2. Implement iris-based gaze computation
3. Integrate gaze data into pipeline
4. Update result structures

#### Fix 2: Face Distance Detection (15 minutes)
1. Use MediaPipe confidence as face size proxy
2. Map confidence to camera distance
3. Update professional wrapper

**Total Time**: ~40 minutes (can be done in parallel)

### Critical Success Metrics
- ✅ `eyeContactRate > 0` when person looking at camera
- ✅ `averageFaceSize > 0` with proper camera distance classification
- ✅ `gazeSteadiness` metric populated
- ✅ Correct shot type identification (close/medium/wide)

## Alternative Approaches Considered

### Option 1: Use FEAT for Gaze (Rejected)
- **Pros**: Already integrated, might have gaze features
- **Cons**: FEAT is for emotion detection, not optimized for gaze
- **Verdict**: Wrong tool for the job

### Option 2: External Gaze Detection Library (Rejected)
- **Pros**: Potentially more accurate
- **Cons**: New dependency, integration complexity, performance overhead
- **Verdict**: Over-engineering for current needs

### Option 3: Simple Face Center Heuristic (Rejected)
- **Pros**: Very simple to implement
- **Cons**: Inaccurate, doesn't work with head movements
- **Verdict**: Too crude for useful results

### Option 4: MediaPipe FaceMesh (Selected ✅)
- **Pros**: Already have MediaPipe, proven iris detection, integrates cleanly
- **Cons**: Slightly more complex than heuristics
- **Verdict**: Best balance of accuracy and simplicity

## Conclusion

Person framing requires **two equally critical fixes** to function properly:

### Fix 1: Gaze Detection Implementation
**Impact**: Without this, `eyeContactRate` remains 0 and eye contact analysis is impossible
**Solution**: Add MediaPipe FaceMesh with iris landmark detection

### Fix 2: Face Distance Detection  
**Impact**: Without this, `averageFaceSize` remains 0 and camera distance is always wrong
**Solution**: Use MediaPipe face confidence as proxy for face size/distance

### Implementation Approach

Both fixes should be implemented **together** as they:
- Address different critical metrics
- Use the same MediaPipe infrastructure
- Are equally important for person framing analysis
- Can be developed in parallel

The result will be complete person framing functionality with accurate eye contact metrics and proper camera distance classification.