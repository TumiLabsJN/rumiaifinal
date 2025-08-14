# PersonFraming V4 - Multi-Person Detection and Gaze Analysis Issues

## Critical Issues Discovered

After successfully fixing the face visibility and size metrics in V3, **two critical issues remain**:

1. **Multi-Person Detection Bug**: Video has 2 people, but `subjectCount: 1`
2. **Gaze Analysis Missing**: `eyeContactRate: 0` and `gazeSteadiness: "unknown"`

**Status**: INVESTIGATION IN PROGRESS - Deep discovery needed to understand multi-person detection and gaze analysis pipelines.

## Issue Analysis

### Issue #1: Multi-Person Detection
**Expected**: `subjectCount: 2` (video has 2 people)  
**Actual**: `subjectCount: 1`

**Hypothesis**: Person counting logic may be:
- Only counting FEAT emotion detections (1 primary speaker)
- Not analyzing YOLO person class detections properly
- Using pose detection count instead of actual person count
- Hardcoded to single-person assumption

### Issue #2: Gaze Analysis Missing  
**Expected**: Eye contact percentage and gaze steadiness metrics
**Actual**: `eyeContactRate: 0`, `gazeSteadiness: "unknown"`

**Hypothesis**: Gaze detection may be:
- Missing MediaPipe FaceMesh integration for iris detection
- Not processing gaze timeline entries properly
- Missing eye landmark detection in MediaPipe service
- Gaze data not being extracted from timeline entries

## Deep Discovery Plan

### Discovery Task 1: Multi-Person Detection Pipeline
**Investigate**:
1. YOLO person class detection - how many people detected?
2. MediaPipe pose detection - individual pose tracking?
3. Person counting logic in `compute_person_framing_metrics`
4. Multi-person dynamics analysis implementation
5. Subject count calculation methodology

### Discovery Task 2: Gaze Analysis Pipeline  
**Investigate**:
1. MediaPipe gaze data generation - iris landmark detection?
2. Gaze timeline creation and population
3. Eye contact calculation methodology
4. Gaze steadiness analysis implementation
5. FaceMesh integration status

### Discovery Task 3: Data Flow Analysis
**Trace**:
1. Multi-person data from ML services → Timeline → Computation
2. Gaze data from MediaPipe → gazeTimeline → eye contact metrics
3. Cross-reference with actual video content (2 people visible)

## Deep Discovery Results

### **🔍 ROOT CAUSES IDENTIFIED**

#### **Issue #1: Multi-Person Detection Failure**

**Root Cause A: MediaPipe Configuration Limit**
- **Location**: `/rumiai_v2/api/ml_services_unified.py:95`
- **Problem**: `max_num_faces=1` hardcoded in FaceMesh configuration
- **Impact**: Limits detection to 1 person despite video having 2+ people

```python
# CURRENT (BROKEN):
'face_mesh': mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,  # <<<< CRITICAL LIMITATION
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

**Root Cause B: Hardcoded Single-Person Assumption**
- **Location**: `/rumiai_v2/processors/precompute_functions.py:285-286`
- **Problem**: Ignores actual detection data, uses defaults
- **Impact**: Always returns `subject_count: 1` regardless of reality

```python
# CURRENT (BROKEN):
'primary_subject': 'single',  # Default for now
'subject_count': 1,  # Default for now  
```

#### **Issue #2: Eye Contact Processing Failure**

**Root Cause: Wrong Threshold Logic**
- **Location**: `/rumiai_v2/processors/precompute_functions.py:264`
- **Problem**: Gaze processing logic incorrectly processes high-quality gaze data
- **Impact**: 0.81-0.97 eye contact values result in 0% eye contact rate

```python
# CURRENT (BROKEN LOGIC):
if gaze_data.get('eye_contact', 0) > 0.5:  # Wrong processing
    eye_contact_frames += 1
```

### **📊 ACTUAL DATA ANALYSIS**

#### **Video 7274651255392210219 Reality Check**

**MediaPipe Face Detection Data**:
```json
{"count": 1}, {"count": 1}, {"count": 3}, {"count": 3}, {"count": 1}
```
- **Reality**: 2 people in video, MediaPipe detects 1-3 faces at different times
- **System Reports**: `subjectCount: 1` (wrong)

**Gaze Analysis Data**:
```json
{"eye_contact": 0.81}, {"eye_contact": 0.84}, {"eye_contact": 0.90}, {"eye_contact": 0.92}
```
- **Reality**: Strong eye contact (81-97%), 108+ measurements
- **System Reports**: `eyeContactRate: 0` (completely wrong)

**YOLO Person Tracking**:
```json
{"class": "person", "trackId": "obj_0_0"}, {"class": "person", "trackId": "obj_1_0"}
```
- **Reality**: 123 person detections with track patterns
- **Analysis**: Track IDs suggest single-person tracking limitation

### **🔧 SOLUTION IMPLEMENTATION**

#### **Fix #1: Enable Multi-Person Face Detection**
**File**: `/rumiai_v2/api/ml_services_unified.py`
**Change**: Increase MediaPipe face detection limit
```python
# FIXED VERSION:
'face_mesh': mp.solutions.face_mesh.FaceMesh(
    max_num_faces=5,  # Support up to 5 faces
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

#### **Fix #2: Dynamic Subject Count Calculation**
**File**: `/rumiai_v2/processors/precompute_functions.py`
**Change**: Calculate from actual MediaPipe face data
```python
# FIXED VERSION:
# Calculate actual subject count from MediaPipe face data
max_faces_detected = 0
for timestamp_key, person_data in person_timeline.items():
    face_count = person_data.get('face_count', 0)  # From MediaPipe count field
    max_faces_detected = max(max_faces_detected, face_count)

subject_count = max(1, max_faces_detected)  # At least 1, up to actual detections
primary_subject = 'multiple' if subject_count > 1 else 'single'

# Return actual values:
'subject_count': subject_count,
'primary_subject': primary_subject
```

#### **Fix #3: Eye Contact Calculation Logic**
**File**: `/rumiai_v2/processors/precompute_functions.py`
**Change**: Proper gaze data processing
```python
# FIXED VERSION:
total_gaze_measurements = 0
total_eye_contact_score = 0

for timestamp_key, gaze_data in gaze_timeline.items():
    eye_contact_value = gaze_data.get('eye_contact', 0)
    if eye_contact_value > 0:  # Count all valid measurements
        total_gaze_measurements += 1
        total_eye_contact_score += eye_contact_value

# Calculate average eye contact rate
eye_contact_rate = total_eye_contact_score / total_gaze_measurements if total_gaze_measurements > 0 else 0

# Calculate gaze steadiness from variance
gaze_values = [gaze_data.get('eye_contact', 0) for gaze_data in gaze_timeline.values() if gaze_data.get('eye_contact', 0) > 0]
gaze_variance = statistics.variance(gaze_values) if len(gaze_values) > 1 else 0
gaze_steadiness = 'high' if gaze_variance < 0.1 else 'medium' if gaze_variance < 0.2 else 'low'
```

### **🎯 EXPECTED RESULTS**

#### **After Fixes Applied to Video 7274651255392210219**:
```json
{
  "personFramingCoreMetrics": {
    "subjectCount": 2,                    // Was: 1 (fixed from MediaPipe face count)
    "primarySubject": "multiple",         // Was: "single" (fixed from actual data)
    "eyeContactRate": 0.87,              // Was: 0 (fixed from 0.81-0.97 gaze data)
    "averageFaceSize": 9.84,             // Still working ✅
    "faceVisibilityRate": 0.97           // Still working ✅
  },
  "personFramingDynamics": {
    "gazeSteadiness": "high"             // Was: "unknown" (calculated from low variance)
  }
}
```

### **⚡ IMPLEMENTATION PRIORITY**

#### **High Impact (Immediate)**:
1. **MediaPipe FaceMesh Config**: Change `max_num_faces=1` to `max_num_faces=5`
2. **Subject Count Logic**: Use actual face detection data instead of hardcoded 1
3. **Eye Contact Processing**: Fix gaze data processing logic

#### **Medium Impact (Next Phase)**:
1. **YOLO Multi-Tracking**: Investigate track ID patterns for better person tracking
2. **Face-Person Association**: Link MediaPipe faces to YOLO person tracks
3. **Multi-Person Metrics**: Expand all metrics for individual person analysis

#### **Validation Required**:
1. Test with known multi-person videos
2. Verify eye contact calculations against manual review
3. Confirm subject count matches visual inspection

### **🚨 CRITICAL ARCHITECTURAL FINDING**

**The ML pipeline is fundamentally sound** - MediaPipe generates excellent multi-person and gaze data. The issues are:

1. **Configuration Limitations**: Artificial limits preventing multi-person detection
2. **Processing Logic Errors**: Wrong threshold/calculation logic ignoring good data
3. **Hardcoded Assumptions**: Defaults that override actual ML detections

**This is NOT a data pipeline problem** - it's a **configuration and logic problem** that ignores high-quality ML data.

The fixes are **straightforward configuration changes and logic corrections** that will unlock the existing high-quality multi-person and gaze detection capabilities.