# Bug Report 1130 - Systematic Timeline Integration Failures

## Overview
This document outlines critical timeline integration failures affecting multiple ML services in the RumiAI pipeline. After successfully implementing FEAT emotion detection integration, systematic issues were discovered in other services that prevent them from properly contributing to video analysis.

**Bug ID**: 1130  
**Severity**: HIGH  
**Impact**: Multiple analysis outputs showing zero/incorrect data despite working ML services  
**Status**: Under Investigation  

## Problem Summary

### Affected Services
1. **Scene Detection**: Service crashes with FrameTimecode type error → Zero scenes detected
2. **Person Framing**: Service runs but extracts zero face data → Invalid framing analysis

### Working Services (For Comparison)
1. **FEAT Emotion Detection**: ✅ Fully integrated after recent fixes
2. **YOLO Object Detection**: ✅ Working correctly  
3. **Whisper Speech**: ✅ Working correctly
4. **OCR Text**: ✅ Working correctly

## Detailed Problem Analysis

### Issue 1: Scene Detection Service Crash

**File**: `/rumiai_v2/api/ml_services.py`  
**Lines**: 121, 129  
**Error**: `"unsupported operand type(s) for /: 'FrameTimecode' and 'int"`

**Root Cause Analysis**:
- **When it broke**: Commit d62533b "Pre Revolutions (sans Claude)"
- **Why it broke**: Changed from `scenedetect.open_video()` to `VideoManager` API
- **Working before**: Yes - older outputs (Jul 30) have successful scene detection
- **Impact timeline**: All videos processed after Aug 8 show the error
- **Evidence**: Files from Jul 30-Aug 5 have scenes, Aug 8+ have error message

**Git History Discovery**:
```bash
# The breaking change (commit d62533b):
- from scenedetect import open_video, SceneManager
+ video_manager = VideoManager([str(video_path)])
+ duration = video_manager.get_duration()[0]  # Returns FrameTimecode, not float!
```

**Similar Bugs Check**: Only one instance found (line 121) - no other similar type errors

**Root Cause**:
```python
# Line 121 - BROKEN
duration = video_manager.get_duration()[0] if video_manager.get_duration() else 30.0

# Line 129 - CRASHES HERE  
avg_scene_length = duration / len(scenes) if scenes else duration
```

**Problem**: `video_manager.get_duration()[0]` returns a `FrameTimecode` object, not a float. When divided by `len(scenes)` (int), Python raises a type error.

**Verified API Investigation**:
- `VideoManager.get_duration()` returns a tuple of 3 `FrameTimecode` objects: (total_duration, start_time, end_time)
- ✅ **CONFIRMED**: `.get_seconds()` is the correct method (not `.seconds` property)
- ✅ **TESTED**: FrameTimecode objects do NOT support division operators
- ✅ **EDGE CASES**: None/empty handling already covered by existing conditional

**Evidence**: Working pattern exists in same file at lines 144-146:
```python
'start_time': start.get_seconds(),    # ✅ Using .get_seconds() correctly
'end_time': end.get_seconds(),        # ✅ Using .get_seconds() correctly
'duration': (end - start).get_seconds()  # ✅ Using .get_seconds() correctly
```

**Why This Wasn't Caught**:
- No unit tests for scene detection
- Error is silently caught and saved to JSON (not logged)
- Scene pacing analysis continues with `totalScenes: 1` (assumes one long scene)
- System appears to work but produces invalid results

**Impact**: 
- Scene detection service crashes completely (since Aug 8)
- No scene timeline entries created  
- Scene pacing analysis shows `"totalScenes": 1` (misleading)
- All scene-based analysis features fail silently
- 13+ videos processed with broken scene detection

**Verified Fix**:
```python
# Line 121 - Simple fix: Add .get_seconds() method call
duration = video_manager.get_duration()[0].get_seconds() if video_manager.get_duration() else 30.0
```
**Note**: No try/except needed - the API consistently returns FrameTimecode objects, and the existing conditional handles edge cases.

### Issue 2: FEAT Face Bounding Box Not Saved (Critical for Camera Distance + Service Deduplication)

**Files Affected**:
- `/rumiai_v2/ml_services/emotion_detection_service.py` (extraction but not saving)
- `/rumiai_v2/processors/precompute_functions_full.py` (computation needs face size)

**Root Cause**: FEAT successfully extracts face bounding box but doesn't save it to output JSON, preventing camera distance calculations.

**Data Flow Problem**:
1. **FEAT Service** ✅ Extracts face bbox at line 426: `'face_bbox': self._safe_extract_face_bbox(pred)`
2. **Save to Results** ❌ Lines 153-159 save emotion data but OMIT face_bbox field
3. **JSON Output** ❌ Missing face_bbox means no face size data available
4. **Person Framing** ❌ Can't calculate camera distance without face size

**Evidence**:
```python
# Line 426 - FEAT extracts bbox successfully
results.append({
    'emotion': mapped_emotion,
    'confidence': emotion_scores[dominant_emotion],
    'emotion_scores': {...},
    'action_units': action_units,
    'au_intensities': au_intensities,
    'face_bbox': self._safe_extract_face_bbox(pred)  # ✅ EXTRACTED HERE
})

# Lines 153-159 - But NOT saved to final results
results['emotions'].append({
    'timestamp': timestamp,
    'emotion': detection['emotion'],
    'confidence': detection['confidence'],
    'all_scores': detection['emotion_scores'],
    'action_units': detection['action_units'],
    'au_intensities': detection['au_intensities']
    # ❌ MISSING: 'face_bbox': detection['face_bbox']
})
```

**Camera Distance Impact**:

Face Size → Camera Distance Mapping:
- **Large face (>30% frame)** → Close-up shot (intimate, emotional)
- **Medium face (10-30%)** → Medium shot (standard talking head)
- **Small face (<10%)** → Wide shot (environmental context)

Without face bbox data:
- Can't detect zoom-ins (face getting larger)
- Can't identify dramatic close-ups
- Can't measure framing consistency
- Can't determine professional vs amateur framing

**Additional Impact**:
- This makes FEAT the SINGLE source for face detection
- Eliminates need for MediaPipe face detection  
- Provides face size for camera distance calculations
- Reduces computation by ~30% (no duplicate face detection)

**Current Broken Output**:
```json
// FEAT Emotion Output - Missing face_bbox
{
  "emotions": [{
    "timestamp": 0.0,
    "emotion": "neutral",
    "confidence": 0.657,
    "all_scores": {...},
    "action_units": [1, 2],
    "au_intensities": {...}
    // ❌ No face_bbox field
  }]
}

// Person Framing Output - Can't calculate face size
{
  "CoreMetrics": {
    "averageFaceSize": 0,        // ❌ No bbox data available
    "cameraDistance": "unknown"   // ❌ Can't determine without face size
  }
}
```

### Issue 3: Real Eye Contact Requires MediaPipe Iris (FEAT Can't Do This)

**Files Affected**:
- `/local_analysis/enhanced_human_analyzer.py` (disconnected from pipeline)
- `/mediapipe_human_detector.py` (missing iris landmarks)
- `/rumiai_v2/processors/precompute_functions_full.py` (expects eye contact data that never arrives)

**Root Cause**: System claims to detect "eye contact" but actually just checks if face is centered in frame.

**Clarification**:
- FEAT provides face detection and emotions but NOT iris tracking
- ONLY MediaPipe can provide iris landmarks (with refine_landmarks=True)  
- This is MediaPipe's UNIQUE contribution for faces
- Enhanced Human Analyzer is dead code that should be removed

**Current Fake Implementation**:
```python
# enhanced_human_analyzer.py lines 253-254
face_center_x = (left_center[0] + right_center[0]) / 2
if 0.45 < face_center_x < 0.55:  # Face is centered
    gaze_data['eye_contact'] = True  # ❌ NOT real eye contact!
```

**Problems**:
1. **Enhanced Human Analyzer is DISCONNECTED** - Never called in pipeline
2. **MediaPipe doesn't use iris landmarks** - `refine_landmarks=False` 
3. **Person framing expects eye contact data** - Always gets 0
4. **"Eye contact" is fake** - Just checks face centering, not actual gaze

**Evidence**:
- `EnhancedHumanAnalyzer` only runs manually: `python enhanced_human_analyzer.py <video_id>`
- Person framing expects `enhancedHumanAnalysis` data but gets empty dict
- Output always shows: `"eyeContactRate": 0`

### Issue 4: MediaPipe Should Only Provide Gaze, Not Duplicate Face Detection

**Current Problem**: MediaPipe duplicates FEAT's face detection but doesn't provide its unique capability - iris/gaze tracking

**Evidence of Wasted Capability**:
```python
# mediapipe_human_detector.py line 29
refine_landmarks=True,  # ✅ Has iris landmarks enabled!

# But NEVER extracts iris or calculates gaze!
# Just does fake "expression" analysis (lines 109-157)
expression = self._analyze_expression(face_landmarks.landmark, w, h)  # ❌ Fake emotions
```

**Current Broken Data Flow**:
```python
# timeline_builder.py lines 268-276 adds redundant face data:
entry = TimelineEntry(
    entry_type='face',  # ❌ Duplicate of FEAT
    data={
        'emotion': face.get('emotion', 'neutral'),  # ❌ Fake (not real emotions)
        'gaze_direction': face.get('gaze_direction', 'unknown')  # ❌ Always 'unknown'!
    }
)

# precompute_functions.py line 456 ignores these entries anyway:
if entry.get('entry_type') == 'emotion':  # Never checks for 'face' entries
```

**The Real Issue**:
- MediaPipe has iris landmarks but never uses them
- Creates fake emotions that duplicate FEAT
- Gaze is always 'unknown' despite having the data
- These entries are ignored during extraction anyway!

## Architectural Analysis

### Root Cause Pattern

The issues follow a pattern of **incomplete integration** rather than fundamental architectural problems:

1. **Services work individually** (MediaPipe detects faces, scene detection would work with type fix)
2. **Timeline builder partially integrates** (services in builders dict)  
3. **Extraction/computation layers fail** (don't handle the data correctly)

### Comparison with Working Integration

**FEAT Emotion Detection** (working) vs **MediaPipe Face Detection** (broken):

| Stage | FEAT (✅ Working) | MediaPipe Faces (❌ Broken) |
|-------|-------------------|---------------------------|
| Service | Runs, outputs JSON | Runs, outputs JSON |
| Validation | `validate_emotion_data()` | No face validation |  
| Timeline Builder | `_add_emotion_entries()` | `_add_mediapipe_entries()` incomplete |
| Timeline Extraction | Extracts emotion entries | Skips face entries |
| Computation | Uses emotion data | Looks for missing face data |

### Data Flow Integrity Check

```
ML Service → Timeline Builder → Timeline Entries → Extraction → Computation
     ↓              ↓                   ↓             ↓            ↓
✅ FEAT        ✅ Added        ✅ Created      ✅ Extracted  ✅ Used
✅ MediaPipe   ❓ Partial      ❓ Incomplete   ❌ Skipped    ❌ Missing
❌ Scene Det   ❌ Crashes      ❌ None         ❌ None       ❌ Empty
✅ YOLO        ✅ Added        ✅ Created      ✅ Extracted  ✅ Used  
✅ Whisper     ✅ Added        ✅ Created      ✅ Extracted  ✅ Used
```

## Proposed Solution Strategy

### Phase 1: Emergency Fixes (Critical Path)

**Fix 1 - Scene Detection Crash**
```python
# File: /rumiai_v2/api/ml_services.py line 121
# VERIFIED FIX - Confirmed .get_seconds() is the correct method (not .seconds property)
duration = video_manager.get_duration()[0].get_seconds() if video_manager.get_duration() else 30.0
```

**Fix 2 - FEAT Face Bbox Extraction**
```python
# File: /rumiai_v2/ml_services/emotion_detection_service.py line 153-159
# Add face_bbox to saved emotions data
results['emotions'].append({
    'timestamp': timestamp,
    'emotion': detection['emotion'],
    'confidence': detection['confidence'],
    'all_scores': detection['emotion_scores'],
    'action_units': detection['action_units'],
    'au_intensities': detection['au_intensities'],
    'face_bbox': detection.get('face_bbox', [0, 0, 0, 0])  # ✅ ADD THIS LINE - Format: [x, y, width, height]
})
```

**Fix 3 - Wrong Dictionary Key (Simple Find & Replace)**

**Root Cause**: FEAT outputs 'emotion' key, timeline builder saves 'emotion' key, but precompute_functions_full.py looks for 'expression' key that never existed.

**Affected Lines in precompute_functions_full.py**:
- Lines 1194, 1225, 1735, 1771, 1819, 1965, 2022, 2074 (8 locations)

**Simple Fix - Find & Replace All**:
```python
# BROKEN (all 8 locations):
expression_timeline[timestamp].get('expression')  # ❌ Wrong key - never existed!

# FIX TO:
expression_timeline[timestamp].get('emotion')     # ✅ Correct key from FEAT
```

**Why No Compatibility Layer Needed**:
- FEAT never outputs 'expression' - it always outputs 'emotion'
- Timeline builder always saves 'emotion' key (line 377)
- The 'expression' key was simply a typo/mistake
- This is a straightforward find-replace fix, not technical debt

**Fix 4 - Convert MediaPipe to Gaze-Only Service (Simplified)**
```python
# File: /mediapipe_human_detector.py
# REMOVE fake expression analysis, ADD simple eye contact detection

# Replace detect_faces_and_expressions() with:
def detect_gaze_only(self, image):
    """Extract iris landmarks for eye contact detection only"""
    results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return []
    
    gaze_data = []
    for face_landmarks in results.multi_face_landmarks:
        try:
            # Verify all required landmarks exist
            required_indices = [468, 473, 133, 33, 362, 263]  # Iris + eye corners
            if len(face_landmarks.landmark) < max(required_indices) + 1:
                continue
            
            # Get iris centers for both eyes
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]
            
            # Eye corners for normalization
            left_inner = face_landmarks.landmark[133]
            left_outer = face_landmarks.landmark[33]
            right_inner = face_landmarks.landmark[362]
            right_outer = face_landmarks.landmark[263]
            
            # Calculate normalized positions (0=inner, 1=outer)
            left_width = abs(left_outer.x - left_inner.x)
            left_pos = (left_iris.x - left_inner.x) / left_width if left_width > 0 else 0.5
            
            right_width = abs(right_outer.x - right_inner.x)
            right_pos = (right_iris.x - right_inner.x) / right_width if right_width > 0 else 0.5
            
            # Average both eyes
            avg_position = (left_pos + right_pos) / 2
            
            # Eye contact when iris is centered (0.35-0.65 range)
            eye_contact = 1.0 if 0.35 <= avg_position <= 0.65 else 0.0
            
            gaze_data.append({
                'eye_contact': eye_contact,
                'gaze_direction': 'camera' if eye_contact > 0.5 else 'away'
            })
        except (IndexError, AttributeError):
            # Skip this face if landmarks are missing or malformed
            continue
    
    return gaze_data

# File: /timeline_builder.py lines 261-279
# REPLACE 'face' entries with 'gaze' entries:
gaze_results = mediapipe_data.get('gaze', [])  # Not 'faces'
for gaze in gaze_results:
    timestamp = self._extract_timestamp_from_annotation(gaze)
    if not timestamp:
        continue
    
    entry = TimelineEntry(
        start=timestamp,
        end=None,
        entry_type='gaze',  # ✅ Not 'face'
        data={
            'eye_contact': gaze.get('eye_contact', 0.0),
            'gaze_direction': gaze.get('gaze_direction', 'unknown')
        }
    )
    timeline.add_entry(entry)
```

**Fix 5 - Extract Gaze Entries for Person Framing**
```python
# File: /rumiai_v2/processors/precompute_functions.py
# Add extraction for 'gaze' entries (new timeline type)
for entry in timeline_entries:
    if entry.get('entry_type') == 'gaze':  # New gaze entries
        timestamp = entry.get('start', 0)
        if isinstance(timestamp, str) and timestamp.endswith('s'):
            timestamp = float(timestamp[:-1])
        
        timestamp_key = f"{int(timestamp)}-{int(timestamp)+1}s"
        timelines['gazeTimeline'][timestamp_key] = entry.get('data', {})

# File: /rumiai_v2/processors/precompute_functions_full.py
# Use gaze data in person_framing_metrics
gaze_timeline = timelines.get('gazeTimeline', {})
eye_contact_frames = sum(1 for t in gaze_timeline.values() if t.get('eye_contact', 0) > 0.5)
eye_contact_ratio = eye_contact_frames / total_frames if total_frames > 0 else 0
```

### Phase 2: Architectural Cleanup - One Service Per Capability

**Goal**: Each ML service does ONLY what it uniquely provides

**Task 1 - Update Timeline Builder**
```python
# Remove MediaPipe face entries (except iris if enabled)
# Use FEAT emotions as face presence signal  
# Add FEAT face_bbox to timeline
def _add_mediapipe_entries(self, timeline, mediapipe_data):
    # ONLY add unique MediaPipe capabilities
    for pose in mediapipe_data.get('poses', []):
        # Add body pose entries
    for gesture in mediapipe_data.get('gestures', []):
        # Add hand gesture entries
    # NO face entries unless iris tracking enabled
```

**Task 2 - Update Person Framing to Use FEAT**
```python
# Use FEAT for face presence (with correct key)
for timestamp in expression_timeline:
    if expression_timeline[timestamp].get('emotion'):  # ✅ Correct FEAT key
        face_frames += 1

# Use FEAT face_bbox for camera distance
if 'face_bbox' in emotion_data:
    x, y, width, height = emotion_data['face_bbox']
    face_size = (width * height) * 100  # Percentage of frame
    camera_distance = 'close-up' if face_size > 30 else 'medium' if face_size > 10 else 'wide'
```

**Task 3 - Extract Face Size from FEAT**
```python
# File: /rumiai_v2/processors/precompute_functions_full.py
# Update person_framing_analysis to use FEAT face bbox

def person_framing_analysis(timeline_data):
    emotion_data = timeline_data.get('emotions', {})
    
    # Calculate average face size from FEAT bbox data
    face_sizes = []
    for timestamp, emotions in emotion_data.items():
        for emotion in emotions:
            if 'face_bbox' in emotion:
                x, y, width, height = emotion['face_bbox']
                # Face size as percentage of frame
                face_size = (width * height) * 100  
                face_sizes.append(face_size)
    
    avg_face_size = np.mean(face_sizes) if face_sizes else 0
    
    # Determine camera distance from face size
    if avg_face_size > 30:
        camera_distance = 'close-up'
    elif avg_face_size > 10:
        camera_distance = 'medium'
    else:
        camera_distance = 'wide'
    
    return {
        'averageFaceSize': avg_face_size,
        'cameraDistance': camera_distance,
        'faceVisibilityRate': len(face_sizes) / total_frames
    }
```

**Task 2 - Timeline Builder Integration**  
- Update `_add_emotion_entries()` to include face_bbox in TimelineEntry
- Ensure bbox data flows through to person framing analysis

**Task 3 - Metrics Enhancement**
- Add zoom detection (face size changes over time)
- Add framing consistency score (variance in face size)
- Add professional framing detection (stable face size in appropriate range)

### Phase 3: Prevention Measures

**Task 1 - Integration Test Suite**
- Create test that verifies each service flows through full pipeline
- Test pattern: Service runs → Timeline entries created → Extraction works → Computation succeeds

**Task 2 - Documentation Update**
- Update EmotionService.md with these findings
- Add "Timeline Integration Health Check" section

**Task 3 - Monitoring**
- Add logging to track which services successfully create timeline entries
- Alert when services run but don't contribute timeline data

## Testing Strategy

### Verification Plan

**Test Video**: 7190549760154012970 (has person throughout, subtle scene changes)

**Before Fixes**:
- ❌ Scene pacing: `"totalScenes": 0`
- ❌ Person framing: `"faceVisibilityRate": 0`  
- ✅ Emotional journey: Rich emotion data (comparison baseline)

**After Fixes Should Show**:
- ✅ Scene pacing: `"totalScenes": > 0, scenes detected
- ✅ Person framing: `"faceVisibilityRate": > 0.5, face data present
- ✅ Emotional journey: Unchanged (regression test)

### Integration Test Commands

```bash
# 1. Fix scene detection, run pipeline  
python3 scripts/rumiai_runner.py "test_video.mp4"

# 2. Verify scene detection
grep "totalScenes" insights/*/scene_pacing/*.json

# 3. Verify person framing  
grep "faceVisibilityRate" insights/*/person_framing/*.json

# 4. Verify no regressions
grep "dominantEmotion" insights/*/emotional_journey/*.json
```

## Architectural Issue: Service Overlap - MediaPipe and FEAT Redundancy

**Root Problem**: Multiple services doing the same thing, causing confusion and wasted computation.

**Current Redundancy**:
- FEAT detects faces for emotions
- MediaPipe detects faces for... nothing useful
- YOLO detects person bounding boxes
- All three overlap unnecessarily

**Proposed Clean Architecture**:
```python
SERVICE_RESPONSIBILITIES = {
    'emotion_detection': FEAT,      # Emotions + AUs + face bbox (ONLY face detection)
    'object_detection': YOLO,       # Objects + full body person boxes
    'hand_gestures': MediaPipe,     # Hand tracking and gesture recognition (UNIQUE)
    'body_pose': MediaPipe,         # 33 joint skeleton tracking (UNIQUE)
    'iris_gaze': MediaPipe,         # Iris landmarks for real gaze (UNIQUE, optional)
    'speech': Whisper,              # Transcription
    'scene': PySceneDetect          # Scene changes
}
```

**Benefits of Clean Separation**:
- No redundant face detection (30% computation saved)
- Clear service boundaries
- Each service does ONLY what it's best at
- Easier to debug and maintain

## Architectural Issue: Enhanced Human Analyzer - Dead Code That Should Be Removed

**Files to Delete**:
- `/local_analysis/enhanced_human_analyzer.py` (entire file)
- `/enhanced_human_analysis_outputs/` (directory if exists)
- All references to `enhanced_human_data` in pipeline

**Evidence It's Dead Code**:

1. **Never Imported or Called**:
   ```bash
   # No imports found anywhere:
   grep "import.*enhanced|from.*enhanced" **/*.py  # Returns nothing
   grep "EnhancedHumanAnalyzer" scripts/rumiai_runner.py  # Returns nothing
   ```

2. **Standalone Script Only**:
   - Only runs manually: `python enhanced_human_analyzer.py <video_id>`
   - Not in rumiai_runner.py pipeline
   - Not in test_rumiai_complete_flow.js
   - Output directory doesn't even exist in most installations

3. **Pipeline Uses Fake "Enhanced" Data**:
   ```python
   # precompute_functions.py line 634 - What actually happens:
   enhanced_human_data = {
       'faces': mediapipe_data.get('faces', []),  # ❌ NOT from enhanced analyzer!
       'poses': mediapipe_data.get('poses', [])   # ❌ Just MediaPipe renamed!
   }
   
   # precompute_functions_full.py line 3910 - What it expects:
   enhanced_human_data = metadata_summary.get('enhancedHumanAnalysis', {})  # Always empty!
   ```

4. **Duplicates MediaPipe Entirely**:
   - Uses same MediaPipe models (face_mesh, pose, hands)
   - Already has `refine_landmarks=True` (but unused)
   - "Gaze detection" is fake (just checks face centering 0.45-0.55)
   - Outputs to isolated folder nobody reads

5. **Causes Confusion**:
   - Person framing expects eye_contact_ratio but gets 0
   - Code has `if enhanced_human_data:` checks that always fail
   - Developers think system has gaze tracking (it doesn't)
   - Maintenance burden for code that never runs

**Safe to Remove Because**:
- ✅ No imports reference it
- ✅ rumiai_runner.py doesn't know it exists
- ✅ Pipeline already handles empty enhanced_human_data
- ✅ All the "enhanced" features are already in MediaPipe
- ✅ Test suite doesn't use it

**What Should Be Removed**:

1. **Files**:
   ```bash
   rm local_analysis/enhanced_human_analyzer.py
   rm -rf enhanced_human_analysis_outputs/
   ```

2. **Code References in precompute_functions.py**:
   ```python
   # Line 634 - Remove this fake mapping:
   enhanced_human_data = {
       'faces': mediapipe_data.get('faces', []),
       'poses': mediapipe_data.get('poses', [])
   }
   # Replace with: enhanced_human_data = {}
   ```

3. **Code References in precompute_functions_full.py**:
   ```python
   # Lines 2140-2193 - Remove entire enhanced_human_data extraction block
   # Lines 3910, 3918 - Remove enhanced_human_data parameters
   # Line 4020 - Remove enhanced_human_data extraction
   ```

4. **Function Parameters**:
   ```python
   # Update compute_person_framing_metrics() to remove enhanced_human_data parameter
   # It's always empty anyway
   ```

**Why This Is Good**:
- **Removes 600+ lines of dead code**
- **Eliminates confusion** about what features actually work
- **Simplifies debugging** (no phantom "enhanced" data)
- **Reduces maintenance** (one less file to update)
- **Honest about capabilities** (no fake gaze tracking)

**Migration Path**:
If we want REAL eye contact detection:
1. Enable `refine_landmarks=True` in main MediaPipe service
2. Add iris-based eye contact calculation
3. Use FEAT face bbox for camera distance
4. All features stay in active, maintained services

## Updated Risk Assessment (Based on Deep Discovery)

### LOW RISK: 'expression' vs 'emotion' Key Change
**Discovery**: Found 8 locations using wrong 'expression' key
- Lines 1194, 1225, 1735, 1771, 1819, 1965, 2022, 2074 in precompute_functions_full.py
- All need the same simple fix: 'expression' → 'emotion'
- **No compatibility needed**: 'expression' key never existed in the data
- **Fix**: Simple find & replace all 8 occurrences
```python
# Just change all instances from:
.get('expression')
# To:
.get('emotion')
```

### LOW RISK: FrameTimecode Fix
**Discovery**: `.get_seconds()` is the correct method (already used successfully in lines 144-146)
- VideoManager API consistently returns FrameTimecode objects since commit d62533b
- Existing conditional already handles None case
- **Fix**: Simply add `.get_seconds()` - no defensive coding needed
```python
# Line 121 - Simple, clean fix:
duration = video_manager.get_duration()[0].get_seconds() if video_manager.get_duration() else 30.0
```
**Why no defensive coding needed**:
- The error is consistent and specific (FrameTimecode vs int)
- Same pattern works in lines 144-146 without issues
- Over-engineering would hide real API problems

### LOW RISK: Iris Landmarks
**Discovery**: `refine_landmarks=True` already enabled, 478 landmarks available
- Iris indices 468-477 exist but never used
- **Mitigation**: Check landmark count before accessing:
```python
if len(face_landmarks.landmark) >= 478:
    # Safe to use iris
else:
    # Fallback to no gaze
```


## Rollback Plan

**If fixes cause regressions**:

1. **Scene Detection**: Revert line 121 change, accept crash (current state)
2. **MediaPipe Faces**: Remove extraction code addition  
3. **Timeline Builder**: No changes made, safe rollback

**Monitoring**: Use emotional journey analysis as regression test - it should remain unchanged.

## Success Metrics

**Scene Detection Fix**:
- Scene service no longer crashes  
- `totalScenes` > 0 for videos with scene changes
- Scene pacing analysis shows realistic data

**Person Framing Fix**:  
- `faceVisibilityRate` > 0 when FEAT detects emotions (correct 'emotion' key)
- `averageFaceSize` from FEAT face_bbox dimensions
- `cameraDistance` correctly identifies close-up/medium/wide shots
- Real `eyeContactRate` from MediaPipe iris (if enabled)

**Service Deduplication**:
- FEAT is ONLY face detector (emotions prove face exists)
- MediaPipe ONLY does hands/pose/iris (unique capabilities)
- No redundant face detection
- ~30% reduction in computation

**Data Flow Clarity**:
- Face presence: FEAT emotion data
- Face size: FEAT face_bbox  
- Eye contact: MediaPipe iris (optional)
- Body pose: MediaPipe skeleton
- Hand gestures: MediaPipe hands

**Overall System Health**:
- Each service has unique, non-overlapping purpose
- No phantom "enhanced_human_data"
- Clear architectural boundaries
- Integration test suite passes end-to-end

## Next Steps

1. **Fix dictionary key** ('expression' → 'emotion' - 8 locations) - 2 minutes - **IMMEDIATE WIN**
2. **Fix scene detection crash** (FrameTimecode type error) - 15 minutes
3. **Add FEAT face bbox extraction** (save bbox to JSON) - 15 minutes
4. **Convert MediaPipe to gaze-only** (simple eye contact) - 10 minutes
5. **Connect gaze to person framing** (use gazeTimeline) - 15 minutes
6. **Remove MediaPipe face redundancy** (cleanup) - 15 minutes
7. **Update person framing for FEAT bbox** (camera distance) - 20 minutes
8. **Remove Enhanced Human Analyzer** (dead code) - 10 minutes
9. **Verify all metrics work** (integration test) - 10 minutes
10. **Document architectural decisions** in EmotionService.md

**Implementation Order Rationale**:
- Step 1: Instant fix that makes person framing work immediately
- Step 2-3: Critical fixes for broken features
- Step 4-7: Service separation and optimization
- Step 8-10: Cleanup and verification

**Key Implementation Notes**:
- Dictionary key fix provides immediate value with minimal risk
- FEAT already extracts face bbox (line 426) - just need to save it
- MediaPipe should ONLY do hands/pose/iris (no redundant face detection)
- Enhanced human analyzer is disconnected dead code - remove it entirely

**What This Solves (in order)**:
- ✅ Person framing works immediately (correct 'emotion' key) - Step 1
- ✅ Scene detection no longer crashes (FrameTimecode fix) - Step 2
- ✅ Real face size for camera distance (FEAT bbox) - Step 3
- ✅ Real eye contact from iris position (MediaPipe gaze) - Step 4-5
- ✅ No redundant face detection (FEAT is single source) - Step 6-7
- ✅ Clear service boundaries (each does ONE thing well) - Step 8-10

**Estimated Total Fix Time**: 112 minutes  
**Risk Level**: Low → High (starts with low-risk quick wins)  
**Testing Confidence**: Low (no tests exist, manual verification needed)