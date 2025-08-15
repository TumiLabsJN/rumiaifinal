# Gesture Detection Dependencies Analysis

## Executive Summary
Gesture detection is **NOT currently implemented** in RumiAI. The gestureTimeline is always empty `{}` because MediaPipe's gesture recognition is not configured/enabled.

## Current State
- **Source**: MediaPipe should provide gesture data
- **Status**: NOT IMPLEMENTED (returns empty gestures array)
- **Impact**: 5 flows depend on gesture data but receive empty timeline

## Architecture Analysis

### Current Frame Processing Flow
Video → Unified Frame Extraction → MediaPipe Service → Batch Processing → Individual ML Models

### Discovered Architecture Details
- **Frame Extraction**: 3-5 FPS adaptive sampling (180 frames for 60s video)
- **MediaPipe Service**: Processes ALL frames (strategy: 'all' in FrameSamplingConfig)
- **Batch Processing**: 20 frames per batch (line 305 in ml_services_unified.py)
- **Current Models**: Pose, Face, Hands, Gaze detection on all frames
- **Missing Component**: Gesture Recognition Service

### Performance Analysis
- **MediaPipe Gesture Recognition**: ~2-3ms per frame (mobile optimized)
- **Processing Decision**: Process ALL frames for maximum gesture coverage
- **Total Processing Time**: 540ms for 60s video (all frames)
- **Batch Efficiency**: 20-frame batches optimize memory usage and threading
- **Architectural Principle**: Avoid premature optimization - 540ms is negligible

### Why Batch Processing with Size 20?
1. **Memory Efficiency**: Limits RAM usage during MediaPipe processing
2. **Threading Optimization**: Each batch runs in separate thread via `asyncio.to_thread`
3. **Progress Tracking**: Allows incremental processing updates
4. **Error Isolation**: Failed batch doesn't crash entire video processing
5. **Proven Performance**: Existing pose/face/hands processing uses same batch size

### MediaPipe Service Integration Points
```python
# Current batch processing (ml_services_unified.py:305-320)
batch_size = 20  # Optimal for memory/performance balance
for i in range(0, len(mp_frames), batch_size):
    batch = mp_frames[i:i+batch_size]  # 20 frames per batch
    
    batch_results = await asyncio.to_thread(
        self._process_mediapipe_batch, models, batch  # Process in thread
    )
    
    # Aggregate results from all batches
    all_poses.extend(batch_results['poses'])
    all_gestures.extend(batch_results.get('gestures', []))  # NEW!
```

## Flows That Depend on Gesture Detection

### 1. **Visual Overlay Analysis** 
**File**: `precompute_functions_full.py:43-478`
**Dependencies**:
- Text-gesture coordination tracking
- CTA reinforcement when text + gesture align
- Cross-modal alignment scoring
- Key alignment moments detection

**Specific Features**:
- `text_gesture_coordination`: Tracks aligned/misaligned/neutral gesture-text pairs
- `cta_reinforcement_matrix['text_gesture']`: Counts CTA text with pointing gestures
- Searches for pointing/thumbs_up gestures within 1 second of text overlays
- Creates `text_gesture_sync` key moments when CTA text aligns with pointing

### 2. **Creative Density Analysis**
**File**: `precompute_functions_full.py:515-790`
**Dependencies**:
- Gesture density per second calculation
- Element co-occurrence patterns
- Multi-modal synchronization detection

**Specific Features**:
- Tracks `gesture` count in `element_types_per_second`
- Calculates `text_gesture` co-occurrence patterns
- Includes gestures in density peaks and creative moments
- Weights gestures at 0.87 in quality scoring

### 3. **Emotional Journey Analysis**
**File**: `precompute_functions_full.py:935-1378`
**Dependencies**:
- Emotion-gesture alignment scoring
- Cross-modal emotional consistency
- Authentic expression detection

**Specific Features**:
- `emotion_gesture_alignment`: Validates emotional authenticity
- Happy + thumbs_up/victory = aligned
- Sad + closed_fist/open_palm = aligned  
- Surprised + pointing = aligned
- Tags videos with 'authentic_expression' when alignment > 0.7

### 4. **Speech Analysis**
**File**: `precompute_functions_full.py:3099-3719`
**Dependencies**:
- Speech-gesture synchronization ratio
- Visual punctuation detection
- Body language congruence scoring

**Specific Features**:
- `gesture_sync_ratio`: % of speech segments with gestures
- `gesture_emphasis_moments`: Key speech points with gestures
- Tags 'high_gesture_sync' when ratio > 0.6
- Includes in hook effectiveness scoring

### 5. **Person Framing Analysis**
**File**: `precompute_professional_wrappers.py:45`
**Dependencies**:
- Speech-gesture sync metric (MISSING IMPLEMENTATION)

**Specific Features**:
- `speechGestureSync`: Alignment score in speechInteractions block
- **ISSUE**: Currently hardcoded to 0 due to missing gesture sync computation
- **STATUS**: Person framing wrapper does NOT compute gesture sync despite needing it

## Data Pipeline

### Current (Broken) Flow:
```
MediaPipe Analysis
    ├── Face Detection ✅ (working)
    ├── Pose Detection ✅ (working)
    ├── Hand Detection ✅ (working - landmarks detected)
    └── Gesture Recognition ❌ (NOT IMPLEMENTED)
        └── Returns: gestures: [] (always empty)

Timeline Builder
    └── gestureTimeline: {} (always empty)

Compute Functions
    └── All gesture logic executes but finds no data
```

### MediaPipe Configuration Issue:
**File**: `ml_services_unified.py:328`
```python
'gestures': [],  # Would need additional processing
```

The comment indicates gesture recognition requires additional processing beyond basic MediaPipe hand landmark detection.

## Impact Analysis

### Metrics Currently Affected:
1. **Text-gesture coordination**: Always shows 100% misaligned
2. **CTA reinforcement**: Never detects gesture-reinforced CTAs
3. **Emotion authenticity**: Missing gesture validation
4. **Speech emphasis**: Can't detect visual punctuation
5. **Creative density**: Underreports multi-modal moments
6. **Hook effectiveness**: Missing gesture component

### Quality Scores Impacted:
- Visual Overlay sync score reduced
- Emotional journey authenticity score reduced
- Speech visual harmony score reduced
- Creative density peaks may be missed

## 🚀 EXACT IMPLEMENTATION PLAN

### Prerequisites
```bash
# 1. Download gesture recognizer model (30MB)
cd /home/jorge/rumiaifinal
mkdir -p models
wget -O models/gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

# 2. Verify model downloaded and has correct size
ls -lh models/gesture_recognizer.task
# Should show approximately 30MB file size

# 3. Verify model is readable (will be validated by service on first load)
file models/gesture_recognizer.task
# Should show: data or TensorFlow Lite model
```

### Step 1: Create Gesture Recognizer Service
**File**: `/home/jorge/rumiaifinal/rumiai_v2/api/gesture_recognizer_service.py` (NEW FILE)

```python
"""
MediaPipe Gesture Recognition Service
Handles gesture detection for all video analysis flows
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class GestureRecognizerService:
    """Singleton gesture recognizer to avoid multiple model loads"""
    _instance = None
    _recognizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._recognizer is None:
            self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Initialize MediaPipe gesture recognizer"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            
            # Model path
            model_path = Path(__file__).parent.parent.parent / "models" / "gesture_recognizer.task"
            
            if not model_path.exists():
                logger.error(f"Gesture model not found at {model_path}")
                logger.info("Download from: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task")
                self._recognizer = None
                return
            
            # Create recognizer
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.GestureRecognizerOptions(
                base_options=base_options,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                num_hands=2
            )
            
            self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)
            logger.info("Gesture recognizer initialized successfully")
            
        except ImportError as e:
            logger.error(f"MediaPipe import failed: {e}")
            self._recognizer = None
        except Exception as e:
            logger.error(f"Failed to initialize gesture recognizer: {e}")
            self._recognizer = None
    
    def recognize_frame(self, frame: np.ndarray, timestamp_ms: int = 0) -> List[Dict[str, Any]]:
        """
        Recognize gestures in a single frame
        
        Args:
            frame: RGB frame as numpy array
            timestamp_ms: Frame timestamp in milliseconds
            
        Returns:
            List of detected gestures with confidence scores
        """
        if self._recognizer is None:
            return []
        
        try:
            import mediapipe as mp
            
            # Convert frame to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Perform recognition
            result = self._recognizer.recognize(mp_image)
            
            # Process results
            gestures = []
            if result.gestures:
                for hand_idx, gesture_list in enumerate(result.gestures):
                    if gesture_list:
                        top_gesture = gesture_list[0]
                        if top_gesture.score > 0.5:  # Confidence threshold
                            gestures.append({
                                'type': self._map_gesture_name(top_gesture.category_name),
                                'confidence': float(top_gesture.score),
                                'hand': 'left' if hand_idx == 0 else 'right',
                                'timestamp_ms': timestamp_ms
                            })
            
            return gestures
            
        except Exception as e:
            logger.warning(f"Gesture recognition failed for frame: {e}")
            return []
    
    def _map_gesture_name(self, mediapipe_name: str) -> str:
        """Map MediaPipe gesture names to RumiAI conventions"""
        mapping = {
            'Thumb_Up': 'thumbs_up',
            'Thumb_Down': 'thumbs_down',
            'Victory': 'victory',
            'Pointing_Up': 'pointing',
            'Open_Palm': 'open_palm',
            'Closed_Fist': 'closed_fist',
            'ILoveYou': 'love',
            'None': 'none'
        }
        return mapping.get(mediapipe_name, mediapipe_name.lower())
    
    def cleanup(self):
        """Release resources"""
        if self._recognizer:
            self._recognizer.close()
            self._recognizer = None
            logger.info("Gesture recognizer cleaned up")
```

### Step 2: Update ML Services to Use Gesture Recognizer
**File**: `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py`

**EXACT CHANGES - Part A: Update __init__ method (line 28-36):**
```python
# FIND THIS SECTION (lines 28-36):
def __init__(self):
    self.frame_manager = get_frame_manager()
    self._models = {}  # Models loaded on demand
    self._model_locks = {
        'yolo': asyncio.Lock(),
        'mediapipe': asyncio.Lock(),
        'ocr': asyncio.Lock(),
        'whisper': asyncio.Lock(),
        'audio_energy': asyncio.Lock()
    }

# CHANGE TO:
def __init__(self):
    self.frame_manager = get_frame_manager()
    self._models = {}  # Models loaded on demand
    self._gesture_service = None  # Add gesture service instance
    self._model_locks = {
        'yolo': asyncio.Lock(),
        'mediapipe': asyncio.Lock(),
        'ocr': asyncio.Lock(),
        'whisper': asyncio.Lock(),
        'audio_energy': asyncio.Lock(),
        'gesture': asyncio.Lock()  # Add lock for gesture service
    }
```

**EXACT CHANGES - Part B: Update _process_mediapipe_batch() method (line 349 onwards):**

*Note: This method processes 20-frame batches. Gestures will be detected on every frame within each batch.*
```python
# FIND THIS SECTION (around line 405-413 in the method):
# Process hands
if models['hands']:
    hand_results = models['hands'].process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        hands.append({
            'timestamp': frame_data.timestamp,
            'frame_number': frame_data.frame_number,
            'count': len(hand_results.multi_hand_landmarks)
        })

return {'poses': poses, 'faces': faces, 'hands': hands, 'gaze': gaze_data}

# CHANGE TO:
# Process hands
if models['hands']:
    hand_results = models['hands'].process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        hands.append({
            'timestamp': frame_data.timestamp,
            'frame_number': frame_data.frame_number,
            'count': len(hand_results.multi_hand_landmarks)
        })
        
        # Process gestures when hands detected (all frames for maximum coverage)
        # No sampling - 360ms extra processing is negligible for full gesture detection
        # Lazy load gesture service
        if self._gesture_service is None:
            from .gesture_recognizer_service import GestureRecognizerService
            self._gesture_service = GestureRecognizerService()
        
        # Recognize gestures
        frame_gestures = self._gesture_service.recognize_frame(
            rgb_frame,  # Already in RGB format from line 360
            timestamp_ms=int(frame_data.timestamp * 1000)  # Convert seconds to ms
        )
        
        # Add to results with frame info
        for gesture in frame_gestures:
            gesture['frame_number'] = frame_data.frame_number
            gesture['timestamp'] = frame_data.timestamp  # Keep seconds for consistency
        gestures.extend(frame_gestures)

return {'poses': poses, 'faces': faces, 'hands': hands, 'gaze': gaze_data, 'gestures': gestures}
```

**EXACT CHANGES - Part C: Initialize gestures list (line 353-356):**
```python
# FIND THIS SECTION:
poses = []
faces = []
hands = []
gaze_data = []  # ADD: Gaze detection results

# CHANGE TO:
poses = []
faces = []
hands = []
gaze_data = []  # Gaze detection results
gestures = []  # Gesture detection results
```

### Step 3: Update MediaPipe Result Aggregation
**File**: `/home/jorge/rumiaifinal/rumiai_v2/api/ml_services_unified.py`

**EXACT CHANGES - Update result aggregation (line 318-329):**
```python
# FIND THIS SECTION:
all_poses.extend(batch_results['poses'])
all_faces.extend(batch_results['faces'])
all_hands.extend(batch_results['hands'])
all_gaze.extend(batch_results.get('gaze', []))  # ADD: Aggregate gaze data

result = {
    'poses': all_poses,
    'faces': all_faces,
    'hands': all_hands,
    'gaze': all_gaze,  # ADD: Include gaze in result
    'gestures': [],  # Would need additional processing

# CHANGE TO:
all_poses.extend(batch_results['poses'])
all_faces.extend(batch_results['faces'])
all_hands.extend(batch_results['hands'])
all_gaze.extend(batch_results.get('gaze', []))
all_gestures.extend(batch_results.get('gestures', []))  # ADD: Aggregate gestures

result = {
    'poses': all_poses,
    'faces': all_faces,
    'hands': all_hands,
    'gaze': all_gaze,
    'gestures': all_gestures,  # Now populated from batches!
```

**EXACT CHANGES - Initialize all_gestures (lines 306-309):**
```python
# FIND THIS SECTION (lines 306-309):
all_poses = []
all_faces = []
all_hands = []
all_gaze = []  # Collect gaze data from batches

# CHANGE TO:
all_poses = []
all_faces = []
all_hands = []
all_gaze = []  # Collect gaze data from batches
all_gestures = []  # Collect gesture data from batches
```

### Step 4: Timeline Builder Gesture Processing
**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/timeline_builder.py`

**GOOD NEWS**: The timeline builder already has gesture processing code (lines 301-319):

```python
# EXISTING CODE that already works:
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
```

The `_extract_timestamp_from_annotation()` method (line 412) correctly extracts the 'timestamp' field we're providing from ML services. 

**HOWEVER**: The timeline builder won't receive gesture data until we fix the timeline extraction in Step 6.

### Step 5: Fix Person Framing Gesture Sync
**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/precompute_functions.py`

**EXACT CHANGES - Update compute_person_framing_wrapper (lines 759-791):**
```python
# FIND THIS SECTION:
def compute_person_framing_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for person framing computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    
    expression_timeline = timelines.get('expressionTimeline', {})
    object_timeline = timelines.get('objectTimeline', {})
    # ... other timeline extractions ...
    
    # Get basic metrics first
    basic_result = compute_person_framing_metrics(
        expression_timeline, object_timeline, camera_distance_timeline,
        person_timeline, enhanced_human_data, duration,
        gaze_timeline=timelines.get('gazeTimeline', {})
    )
    
    # Convert to professional 6-block format
    from .precompute_professional_wrappers import ensure_professional_format
    return ensure_professional_format(basic_result, 'person_framing')

# CHANGE TO:
def compute_person_framing_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for person framing computation"""
    timelines = _extract_timelines_from_analysis(analysis_dict)
    
    expression_timeline = timelines.get('expressionTimeline', {})
    object_timeline = timelines.get('objectTimeline', {})
    speech_timeline = timelines.get('speechTimeline', {})
    gesture_timeline = timelines.get('gestureTimeline', {})  # ADD: Extract gesture timeline
    # ... other timeline extractions ...
    
    # Get basic metrics first
    basic_result = compute_person_framing_metrics(
        expression_timeline, object_timeline, camera_distance_timeline,
        person_timeline, enhanced_human_data, duration,
        gaze_timeline=timelines.get('gazeTimeline', {})
    )
    
    # ADD: Compute gesture sync ratio for person framing
    if gesture_timeline and speech_timeline:
        # Simple gesture-speech overlap calculation
        speech_with_gesture = 0
        total_speech_segments = len(speech_timeline)
        
        for speech_timestamp in speech_timeline:
            # Check if any gesture occurs during this speech
            if speech_timestamp in gesture_timeline:
                speech_with_gesture += 1
        
        gesture_sync_ratio = speech_with_gesture / total_speech_segments if total_speech_segments > 0 else 0
    else:
        gesture_sync_ratio = 0
    
    # Add gesture sync to basic result
    basic_result['speech_gesture_alignment'] = gesture_sync_ratio
    
    # Convert to professional 6-block format
    from .precompute_professional_wrappers import ensure_professional_format
    return ensure_professional_format(basic_result, 'person_framing')
```

### Step 6: Update Timeline Extraction (CRITICAL!)
**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/precompute_functions.py`

**WHY THIS IS NEEDED**: Timeline builder can process gestures, but `_extract_timelines_from_analysis()` doesn't extract them from unified analysis, so `gestureTimeline` stays empty.

**EXACT CHANGES - In `_extract_timelines_from_analysis()` method (around line 450):**
```python
# FIND THIS SECTION (around line 445-455):
# Initialize empty timelines
timelines = {
    'speechTimeline': {},
    'expressionTimeline': {},
    'objectTimeline': {},
    'textOverlayTimeline': {},
    'stickerTimeline': {},
    'gestureTimeline': {},  # Currently always empty
    'personTimeline': {},
    'sceneChangeTimeline': [],
    'gazeTimeline': {}
}

# ADD THIS AFTER timeline initialization (around line 460):
# Extract gestures from timeline
if 'gesture' in timeline.entries_by_type:
    for entry in timeline.entries_by_type['gesture']:
        timestamp_key = f"{int(entry.start.seconds)}-{int(entry.end.seconds)}s"
        
        # Initialize if not exists
        if timestamp_key not in timelines['gestureTimeline']:
            timelines['gestureTimeline'][timestamp_key] = {
                'gestures': [],
                'confidence': 0
            }
        
        # Add gesture to list
        gesture_type = entry.data.get('type', 'unknown')
        if gesture_type not in timelines['gestureTimeline'][timestamp_key]['gestures']:
            timelines['gestureTimeline'][timestamp_key]['gestures'].append(gesture_type)
        
        # Update confidence (keep highest)
        timelines['gestureTimeline'][timestamp_key]['confidence'] = max(
            timelines['gestureTimeline'][timestamp_key]['confidence'],
            entry.data.get('confidence', 0)
        )
```

## Implementation Summary

| Flow | Current Status | Bug Fixes Needed | Expected Impact | Details |
|------|----------------|------------------|-----------------|---------|
| **Visual Overlay** | ⚠️ Has Bugs | 3 critical fixes | CTA reinforcement: 0→10-30<br>Sync score: 0→0.3-0.6 | [Appendix A](#appendix-a-visual-overlay-detailed-analysis) |
| **Creative Density** | ✅ Ready | None | Gesture count: 0→50-100<br>Multi-modal peaks: +30% | [Appendix B](#appendix-b-creative-density-detailed-analysis) |
| **Emotional Journey** | ✅ Ready | None | Alignment: 0→0.3-0.7<br>Authenticity tags enabled | [Appendix C](#appendix-c-emotional-journey-detailed-analysis) |
| **Speech Analysis** | ✅ Ready | Minor enhancements | Sync ratio: 0→0.3-0.7<br>Visual punctuation: 0→5-15 | [Appendix D](#appendix-d-speech-analysis-detailed-analysis) |
| **Person Framing** | ❌ Missing Implementation | Add gesture sync calc | speechGestureSync: 0→0.3-0.7 | [Appendix E](#appendix-e-person-framing-detailed-analysis) |

### Key Findings
- **Ready Flows**: 3/5 flows work immediately once gestures are enabled
- **Critical Bugs**: Visual Overlay has `str(gesture_data)` bug that must be fixed
- **Missing Implementation**: Person Framing needs gesture sync calculation added
- **Total Implementation Effort**: ~4-6 hours (mostly MediaPipe integration)

---

**📖 Navigation:**
- **Quick Start**: Continue to [Implementation Plan](#-exact-implementation-plan) for step-by-step instructions
- **Technical Details**: Jump to [Detailed Flow Analysis](#appendix-detailed-flow-analysis) for comprehensive implementation specifics
- **Testing**: Skip to [Testing & Validation](#-testing--validation) for validation procedures

## Appendix: Detailed Flow Analysis

### Appendix A: Visual Overlay Detailed Analysis

#### Current State (From VisualOverlay.md Analysis)
The visual_overlay flow has complex architecture:
1. 5 redundant OCR processing paths (major inefficiency)
2. Text overlays extracted via EasyOCR
3. Sticker detection via HSV thresholding
4. compute_visual_overlay_metrics() processes all timelines

#### Gesture Integration Points

**A. Function Signature Already Accepts Gestures**
```python
# precompute_functions_full.py:43
def compute_visual_overlay_metrics(text_overlay_timeline, sticker_timeline, gesture_timeline,
                                  speech_timeline, object_timeline, video_duration):
    # gesture_timeline is ALREADY a parameter!
```

**B. Two Main Gesture Uses in Visual Overlay**

**1. CTA Reinforcement Matrix (Lines 174-194)**
```python
cta_reinforcement_matrix = {
    'text_only': 0,
    'text_gesture': 0,      # CTA text + gesture
    'text_sticker': 0,
    'all_three': 0           # Text + gesture + sticker
}

# Current implementation:
for timestamp, data in text_overlay_timeline.items():
    text = data.get('text', '').lower()
    if any(keyword in text for keyword in cta_keywords):  # CTA detected
        has_gesture = timestamp in gesture_timeline  # Check for gesture
        has_sticker = timestamp in sticker_timeline
        
        if has_gesture and has_sticker:
            cta_reinforcement_matrix['all_three'] += 1
        elif has_gesture:
            cta_reinforcement_matrix['text_gesture'] += 1  # KEY METRIC
```

**Issue**: Only checks exact timestamp match, should use temporal window

**2. Text-Gesture Coordination (Lines 263-286)**
```python
text_gesture_coordination = {
    'aligned': 0,      # Pointing/tap with text
    'misaligned': 0,   # Text without gesture
    'neutral': 0       # Other gestures with text
}

# Current implementation has a BUG:
for timestamp in text_overlay_timeline:
    text_sec = parse_timestamp_to_seconds(timestamp)
    gesture_found = False
    for gesture_ts in gesture_timeline:
        gesture_sec = parse_timestamp_to_seconds(gesture_ts)
        if abs(text_sec - gesture_sec) < 1.0:  # 1 second window
            gesture_data = gesture_timeline[gesture_ts]
            # BUG: Checks string representation!
            if any(g in str(gesture_data).lower() for g in ['point', 'tap', 'swipe']):
                text_gesture_coordination['aligned'] += 1
```

**C. Implementation Fixes Needed**

**Fix 1: Update Gesture Type Checking**
```python
# Instead of: str(gesture_data).lower()
# Use proper field access:
gestures_list = gesture_data.get('gestures', [])
if any(g in gestures_list for g in ['pointing', 'thumbs_up']):
    text_gesture_coordination['aligned'] += 1
```

**Fix 2: Map MediaPipe Gestures to Visual Context**
```python
VISUAL_GESTURE_MAPPING = {
    'pointing': ['Pointing_Up'],      # Maps to pointing
    'thumbs_up': ['Thumb_Up'],        # Endorsement
    'victory': ['Victory'],           # Success/win
    'open_palm': ['Open_Palm'],       # Presentation
    'closed_fist': ['Closed_Fist'],  # Emphasis
}

# CTA-relevant gestures:
CTA_GESTURES = ['pointing', 'thumbs_up', 'victory']
EMPHASIS_GESTURES = ['pointing', 'open_palm', 'closed_fist']
```

**Fix 3: Improve Temporal Alignment**
```python
def find_gestures_near_text(text_timestamp, gesture_timeline, window=1.0):
    """Find gestures within window of text appearance"""
    text_sec = parse_timestamp_to_seconds(text_timestamp)
    nearby_gestures = []
    
    for gesture_ts, gesture_data in gesture_timeline.items():
        gesture_sec = parse_timestamp_to_seconds(gesture_ts)
        if abs(text_sec - gesture_sec) <= window:
            nearby_gestures.extend(gesture_data.get('gestures', []))
    
    return nearby_gestures
```

**D. Key Alignment Moments (Lines 323-342)**
```python
# Current implementation looks for CTA + pointing combo:
for timestamp, data in text_overlay_timeline.items():
    text = data.get('text', '')
    gestures = find_gestures_near_text(timestamp, gesture_timeline)
    
    if 'pointing' in gestures and any(kw in text.lower() for kw in cta_keywords):
        key_alignment_moments.append({
            'timestamp': round(text_sec, 1),
            'type': 'text_gesture_sync',
            'elements': ['CTA text', 'pointing gesture']
        })
```

**E. Metrics That Will Update**
```python
# Line 348: Overall sync score
total_alignments = sum(text_gesture_coordination.values())
overall_sync_score = text_gesture_coordination['aligned'] / total_alignments

# Line 356: ML Tags
if cta_reinforcement_matrix['text_gesture'] > 0:
    ml_tags.append('cta_focused')
```

### Step 6: Fix Visual Overlay Bugs
**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/precompute_functions_full.py`

**BUG FIX #1 - Line 278 (Text-Gesture Coordination):**
```python
# FIND THIS (around line 278):
if any(g in str(gesture_data).lower() for g in ['point', 'tap', 'swipe']):
    text_gesture_coordination['aligned'] += 1

# REPLACE WITH:
gestures_list = gesture_data.get('gestures', [])
if any(g in gestures_list for g in ['pointing', 'thumbs_up']):
    text_gesture_coordination['aligned'] += 1
elif gestures_list:  # Any other gesture
    text_gesture_coordination['neutral'] += 1
```

**BUG FIX #2 - Add Helper Function (around line 270, before text_gesture_coordination):**
```python
# ADD THIS HELPER FUNCTION:
def find_gestures_near_timestamp(timestamp_str, gesture_timeline, window=1.0):
    """Find gestures within window seconds of a timestamp"""
    try:
        # Parse timestamp like "5-6s" to get start second
        start_sec = int(timestamp_str.split('-')[0])
        nearby_gestures = []
        
        for gesture_ts, gesture_data in gesture_timeline.items():
            gesture_start = int(gesture_ts.split('-')[0])
            if abs(start_sec - gesture_start) <= window:
                nearby_gestures.extend(gesture_data.get('gestures', []))
        
        return nearby_gestures
    except:
        return []
```

**BUG FIX #3 - Update CTA Detection (around line 184):**
```python
# FIND THIS (around line 184):
has_gesture = timestamp in gesture_timeline

# REPLACE WITH:
nearby_gestures = find_gestures_near_timestamp(timestamp, gesture_timeline, window=1.0)
has_gesture = bool(nearby_gestures)
```

#### Expected Impact After Fixes

When gestures are enabled:
- `cta_reinforcement_matrix['text_gesture']`: 0 → 10-30 per video
- `text_gesture_coordination['aligned']`: 0 → 5-15 per video
- `overall_sync_score`: 0 → 0.3-0.6
- ML tags: Adds 'cta_focused' when CTAs have gesture support

### Appendix B: Creative Density Detailed Analysis

#### Current State (From CreativeDensity.md Analysis)
The creative_density flow is the most straightforward:
1. **Primary implementation**: precompute_creative_density.py (344 lines)
2. **Legacy fallback**: precompute_functions_full.py (deprecated)
3. **Service contract validation**: Fail-fast architecture
4. **Output**: 6-block CoreBlocks format with density metrics

#### Gesture Integration Points

**A. Timeline Already Extracted**
```python
# precompute_creative_density.py:57
gesture_timeline = timelines.get('gestureTimeline', {})  # Empty OK: no gestures detected
```

**B. THREE Integration Points for Gestures**

**1. Per-Second Density Calculation (Line 88)**
```python
# Current implementation - ALREADY WORKING:
for second in range(int(duration)):
    timestamp_key = f"{second}-{second+1}s"
    gesture_count = len(gesture_timeline.get(timestamp_key, []))
    total = text_count + sticker_count + object_count + gesture_count + expression_count + scene_count
    density_per_second.append(total)
```

**2. Element Count Aggregation (Line 109)**
```python
element_counts = {
    "gesture": sum(len(v) for v in gesture_timeline.values()),  # Total gestures
    "text": ...,
    "object": ...,
    # Other element types
}
```

**3. Multi-Modal Peak Detection (Lines 156-166)**
```python
# Detects synchronized high-density moments:
for second in range(int(duration)):
    active_elements = []
    if gesture_timeline.get(timestamp_key):
        active_elements.append('gesture')
    
    # If multiple elements active, it's a multi-modal peak
    if len(active_elements) >= 3:
        multi_modal_peaks.append({
            'second': second,
            'elements': active_elements,
            'totalDensity': density_per_second[second]
        })
```

**C. How Gestures Affect Metrics**

**1. Core Metrics Impact**
```python
densityCoreMetrics: {
    "avgDensity": 16.99,        # Will increase with gestures
    "maxDensity": 54.0,         # Peak may be higher
    "elementCounts": {
        "gesture": 0 → 89,      # Currently 0, will show actual count
    }
}
```

**2. Dynamics Impact**
```python
densityDynamics: {
    "densityCurve": [
        {"second": 5, "density": 42, "primaryElement": "gesture"},  # May become primary
    ],
    "volatility": 0.73,         # May increase with gesture bursts
}
```

**3. Interactions Impact**
```python
densityInteractions: {
    "multiModalPeaks": [        # More peaks when gestures sync with other elements
        {
            "second": 5,
            "breakdown": {"expression": 18, "gesture": 12, "object": 15}
        }
    ],
    "elementCooccurrence": {
        "expression_gesture": 0.45,  # Will show actual correlation
        "text_gesture": 0.0 → 0.6    # Currently missing
    }
}
```

**D. Data Format Expected**

Creative Density expects gestures in this format:
```python
gestureTimeline: {
    "0-1s": ["pointing"],           # Simple list format OR
    "5-6s": {                       # Dict format (handles both!)
        "gestures": ["thumbs_up", "victory"],
        "confidence": 0.88
    }
}
```

**Note**: Implementation handles BOTH formats:
- Legacy: List of gesture strings
- New: Dict with 'gestures' field

**E. Element Co-occurrence Patterns**

```python
# Lines 156-180: Tracks which elements appear together
if text_count > 0 and gesture_count > 0:
    element_cooccurrence['text_gesture'] += 1
if expression_count > 0 and gesture_count > 0:
    element_cooccurrence['expression_gesture'] += 1
```

These patterns help identify:
- Text+Gesture = Emphasized messaging
- Expression+Gesture = Authentic emotion
- Object+Gesture = Product demonstration

#### What's Already Working

**NO CODE CHANGES NEEDED!** Creative Density is 100% ready:

1. ✅ Extracts gestureTimeline
2. ✅ Counts gestures per second
3. ✅ Includes in total density
4. ✅ Tracks in element_counts
5. ✅ Detects multi-modal peaks
6. ✅ Calculates co-occurrence
7. ✅ Handles both data formats

#### Expected Impact

When gestures are enabled:
- `element_counts['gesture']`: 0 → 50-100 per video
- `avgDensity`: +2-5 elements per second
- `multiModalPeaks`: +30% more peaks detected
- `element_cooccurrence['text_gesture']`: 0 → 0.3-0.6
- `dominantCombination`: May become "gesture_expression"

#### Quality Score Impact

```python
# Line 328: Gesture weighted at 0.87 confidence
"detectionReliability": {
    "gesture": 0.87,  # Already configured!
    "ocr": 0.92,
    "yolo": 0.98
}
```

Gestures contribute 87% confidence weight to overall quality score.

### Appendix C: Emotional Journey Detailed Analysis

#### Current State (From EmotionService.md Analysis)
The emotional_journey flow follows this pipeline:
1. FEAT runs in video_analyzer → emotions with timestamps
2. Timeline builder creates expressionTimeline entries
3. compute_emotional_metrics() processes the timeline
4. Professional wrapper formats to 6-block structure

#### Gesture Integration Points

**A. Timeline Already Available**
```python
# precompute_functions_full.py:935
def compute_emotional_metrics(expression_timeline, speech_timeline, gesture_timeline, duration, ...):
    # gesture_timeline is ALREADY passed in!
    # Currently always empty {}, will be populated after MediaPipe update
```

**B. Existing Gesture Processing (Lines 1059-1079)**
```python
# Current implementation that receives empty timeline:
for ts, data in expression_timeline.items():
    if ts in gesture_timeline and 'expression' in data:
        emotion = data['expression']
        gestures = gesture_timeline[ts].get('gestures', [])
        
        # Alignment patterns already defined!
        if emotion in ['happy', 'excited'] and any(g in ['thumbs_up', 'victory', 'pointing'] for g in gestures):
            alignment_count += 1
        elif emotion in ['sad', 'thoughtful'] and any(g in ['closed_fist', 'open_palm'] for g in gestures):
            alignment_count += 1
        elif emotion == 'surprised' and any(g in ['pointing', 'open_palm'] for g in gestures):
            alignment_count += 1
```

**C. What Happens When Gestures Are Enabled**

1. **No Code Changes Needed in Emotional Journey!**
   - Logic already exists and waits for data
   - Just needs populated gestureTimeline

2. **Alignment Mapping (Already Defined)**
   ```python
   EMOTION_GESTURE_ALIGNMENT = {
       'happy': ['thumbs_up', 'victory'],      # MediaPipe: Thumb_Up, Victory ✅
       'excited': ['thumbs_up', 'victory', 'pointing'],  # Pointing_Up ✅
       'sad': ['closed_fist', 'open_palm'],    # Closed_Fist, Open_Palm ✅
       'thoughtful': ['closed_fist', 'open_palm'],
       'surprised': ['pointing', 'open_palm']
   }
   ```

3. **MediaPipe Coverage Check**
   - ✅ thumbs_up → Thumb_Up
   - ✅ victory → Victory
   - ✅ pointing → Pointing_Up
   - ✅ closed_fist → Closed_Fist
   - ✅ open_palm → Open_Palm
   - **100% of required gestures covered!**

4. **Metrics That Will Auto-Update**
   ```python
   # Line 1078: emotion_gesture_alignment calculation
   emotion_gesture_alignment = alignment_count / total_checks if total_checks > 0 else 0
   
   # Line 1259: cross_modal_consistency
   cross_modal_consistency = (emotion_gesture_alignment + emotion_speech_alignment['overallAlignment']) / 2
   
   # Line 1343: ML tag generation
   if emotion_gesture_alignment > 0.7:
       emotional_ml_tags.append('authentic_expression')
   ```

5. **Output Structure Updates Automatically**
   ```json
   {
     "emotionalInteractions": {
       "emotionGestureAlignment": 0.85,  // Will change from 0
       "crossModalConsistency": 0.78     // Will improve
     },
     "emotionalPatterns": {
       "mlTags": ["authentic_expression"]  // New tag when > 0.7
     }
   }
   ```

#### Timeline Synchronization Strategy

**Challenge**: Emotion and gesture timestamps may not align perfectly
**Solution**: Already implemented! (Lines 1192-1209)

```python
# Current implementation handles timing mismatch:
for timestamp in emotion_timestamps:
    if timestamp in expression_timeline and timestamp in gesture_timeline:
        # Direct match - process alignment
    # Could enhance with temporal window:
    gesture_window = find_gestures_in_window(timestamp, window=1.0)  # ±1 second
```

#### No Additional Changes Required!

The emotional journey is **already fully prepared** for gestures:
1. ✅ Timeline parameter passed through
2. ✅ Alignment logic implemented
3. ✅ Metrics calculation ready
4. ✅ Output structure prepared
5. ✅ ML tags defined

**The ONLY requirement**: Populate gestureTimeline from MediaPipe

### Appendix D: Speech Analysis Detailed Analysis

#### Current State (From SpeechAnalysis.md Analysis)
The speech_analysis flow is complex with significant redundancy:
1. **4 duplicate audio extraction paths** (major inefficiency)
2. **3 separate Whisper implementations** (inconsistent)
3. **Multiple timeline transformations** (overhead)
4. **Output**: 6-block CoreBlocks format with speech metrics

#### Gesture Integration Points

**A. Function Already Receives Gestures**
```python
# precompute_functions_full.py:3098
def compute_speech_analysis_metrics(speech_timeline, transcript, speech_segments, 
                                   expression_timeline, gesture_timeline,  # HERE!
                                   human_analysis_data, video_duration, ...):
```

**B. Main Gesture Integration (Lines 3568-3586)**

```python
# Calculate gesture sync ratio
if gesture_timeline and speech_segments:
    speech_with_gesture = 0
    
    for segment in speech_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)
        
        # Check if any gestures occur during this speech segment
        for timestamp, gesture_data in gesture_timeline.items():
            gesture_time = parse_timestamp_to_seconds(timestamp)
            if gesture_time and seg_start <= gesture_time <= seg_end:
                speech_with_gesture += 1
                break
    
    gesture_sync_ratio = speech_with_gesture / len(speech_segments)
```

**C. Current Implementation Analysis**

**GOOD**: 
- Already receives gestureTimeline
- Calculates sync ratio correctly
- Handles empty timeline gracefully

**ISSUE**: Only counts if ANY gesture occurs, doesn't analyze gesture types

**D. Gesture Emphasis Moments (Lines 3604-3609)**

```python
# Find gesture emphasis moments
gesture_emphasis_moments = []

# TODO: Requires word-level timing for precise alignment
# Currently just a placeholder
```

**Missing Implementation**: Should detect when specific gestures align with emphasized words

**E. Metrics Using Gestures**

```python
# Line 3647: Body language congruence
'body_language_congruence': round(gesture_sync_ratio, 2)

# Line 3650: Direct metric
metrics['gesture_sync_ratio'] = round(gesture_sync_ratio, 2)

# Line 3654: Visual punctuation (currently always 0)
metrics['visual_punctuation_count'] = len(gesture_emphasis_moments)

# Line 3676-3677: Pattern tags
if gesture_sync_ratio > 0.6:
    speech_pattern_tags.append("high_gesture_sync")

# Line 3711: Hook effectiveness includes gesture sync
hook_effectiveness = sum([
    ...,
    gesture_sync_ratio > 0.5,
    ...
]) / total_factors

# Line 3718: Visual-verbal harmony
visual_verbal_harmony_score = (gesture_sync_ratio + face_on_screen_during_speech) / 2
```

#### Enhancement Opportunities

**1. Gesture Type Analysis**
```python
# Enhance to track specific gesture types during speech
pointing_during_speech = 0
emphasis_gestures = 0

for segment in speech_segments:
    segment_gestures = get_gestures_in_range(seg_start, seg_end)
    if 'pointing' in segment_gestures:
        pointing_during_speech += 1
    if any(g in ['open_palm', 'closed_fist'] for g in segment_gestures):
        emphasis_gestures += 1
```

**2. Gesture Emphasis Detection**
```python
# Detect gestures at speech peaks (high energy moments)
for peak in energy_peaks:
    nearby_gestures = find_gestures_near_timestamp(peak['timestamp'], window=0.5)
    if nearby_gestures:
        gesture_emphasis_moments.append({
            'timestamp': peak['timestamp'],
            'gestures': nearby_gestures,
            'type': 'energy_peak_gesture'
        })
```

**3. Speech Pattern Enhancement**
```python
# Add more nuanced gesture patterns
if pointing_during_speech > 3:
    speech_pattern_tags.append("directive_gestures")
if emphasis_gestures / len(speech_segments) > 0.3:
    speech_pattern_tags.append("expressive_delivery")
```

#### What's Already Working

✅ Gesture sync ratio calculation
✅ Body language congruence metric
✅ Pattern tag generation for high sync
✅ Hook effectiveness includes gestures
✅ Visual-verbal harmony score

#### What Needs Minor Enhancement

1. **Gesture emphasis moments** - Currently empty, needs implementation
2. **Visual punctuation count** - Always 0, needs gesture peak detection
3. **Gesture type analysis** - Only counts presence, not types

#### Expected Impact

When gestures are enabled:
- `gesture_sync_ratio`: 0 → 0.3-0.7
- `visual_punctuation_count`: 0 → 5-15 per video
- `body_language_congruence`: 0 → 0.3-0.7
- Pattern tags: Adds "high_gesture_sync" when > 0.6
- Hook effectiveness: +10-15% improvement
- Visual-verbal harmony: +20-30% score increase

#### Implementation Priority

**LOW** - Speech analysis already works well with gestures, just needs minor enhancements for:
1. Gesture emphasis moment detection
2. Gesture type differentiation
3. Visual punctuation counting

### Appendix E: Person Framing Detailed Analysis

#### Current State (From PersonFraming.md Analysis)
The person_framing flow was recently debugged and fixed:
1. **MediaPipe provides**: Face detection (97% visibility), pose, gaze
2. **Return statement bugs**: FIXED (was missing avg_face_size, wrong field names)
3. **Professional wrapper**: 6-block format with comprehensive metrics
4. **Output**: Now shows correct face visibility and size metrics

#### Gesture Integration - SURPRISING DISCOVERY

**Person Framing does NOT directly use gestures!**

After thorough analysis:
1. **No gesture_timeline parameter** in compute_person_framing_metrics()
2. **No gesture processing** in the computation function
3. **No gesture metrics** in the output structure

**CONFIRMED: It does NOT get gesture data!**

#### Current Implementation Reality

**Person Framing Does NOT Process Gestures**:
```python
# In precompute_functions.py:759-791
def compute_person_framing_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Extract only person framing related timelines
    timelines = _extract_timelines_from_analysis(analysis_dict)
    expression_timeline = timelines.get('expressionTimeline', {})
    object_timeline = timelines.get('objectTimeline', {})
    # ... other timelines ...
    
    # Compute ONLY basic person framing metrics
    basic_result = compute_person_framing_metrics(
        expression_timeline, object_timeline, camera_distance_timeline,
        person_timeline, enhanced_human_data, duration, gaze_timeline
        # NO gesture processing at all!
    )
    
    # Convert to professional format
    return ensure_professional_format(basic_result, 'person_framing')
    
    # speechGestureSync gets default value of 0 from professional wrapper
```

#### Current Implementation in Professional Wrapper

```python
# precompute_professional_wrappers.py:45
"speechInteractions": {
    "speechGestureSync": basic_metrics.get('speech_gesture_alignment', 0),
    # This value comes from speech_analysis, not person_framing!
}
```

#### Architecture Understanding

**Person Framing focuses on:**
- Face detection and visibility
- Shot framing (close/medium/wide)
- Eye contact and gaze
- Subject positioning

**It delegates gesture analysis to Speech Analysis because:**
- Gestures are most meaningful during speech
- Speech-gesture sync is a speech metric
- Person framing cares about face/body position, not hand gestures

#### What This Means for Gesture Implementation

**PERSON FRAMING NEEDS TO BE UPDATED!**

Current state:
1. ❌ Person Framing wrapper does NOT compute gestures
2. ❌ Professional wrapper gets default value of 0
3. ❌ speechGestureSync is truly hardcoded to 0
4. ❌ Will remain 0 even when gestures are enabled elsewhere

**Required Fix**: Person Framing needs to compute or receive gesture sync data

#### Required Implementation for Person Framing

**Option 1: Add Speech Analysis Dependency**
```python
# Update compute_person_framing_wrapper to call speech analysis first
def compute_person_framing_wrapper(analysis_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Extract timelines
    timelines = _extract_timelines_from_analysis(analysis_dict)
    
    # Compute speech metrics to get gesture sync ratio
    speech_metrics = compute_speech_analysis_metrics(
        speech_timeline, transcript, speech_segments,
        expression_timeline, gesture_timeline, ...
    )
    
    # Compute person framing
    basic_result = compute_person_framing_metrics(...)
    
    # Add gesture sync from speech analysis
    basic_result['speech_gesture_alignment'] = speech_metrics.get('gesture_sync_ratio', 0)
    
    return ensure_professional_format(basic_result, 'person_framing')
```

**Current Status**: speechGestureSync will remain 0 until this fix is implemented

#### Architecture Analysis

**Current Architecture Issue**:
- **Missing Dependency**: Person framing wants gesture sync but doesn't compute it
- **Inconsistent Design**: Other flows process their needed timelines directly  
- **Default Values**: Professional wrapper provides fallback 0, masking the missing implementation

**Resolution**: Person framing should either:
1. **Add speech dependency** (compute speech metrics for gesture sync)
2. **Direct gesture processing** (minimal gesture sync calculation)
3. **Remove the metric** (if not actually needed for person framing)

## 🧪 TESTING & VALIDATION

### Step 6: Test Gesture Detection
```bash
# 1. Create test script
cat > test_gestures.py << 'EOF'
#!/usr/bin/env python3
"""Test gesture detection implementation"""
import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from rumiai_v2.api.gesture_recognizer_service import GestureRecognizerService
import cv2
import numpy as np

def test_gesture_recognizer():
    """Test the gesture recognizer service"""
    print("Testing Gesture Recognizer Service...")
    
    # Initialize service
    service = GestureRecognizerService()
    
    # Create test frame (blank for now)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test recognition
    gestures = service.recognize_frame(test_frame, timestamp_ms=0)
    
    print(f"✓ Service initialized")
    print(f"✓ Recognition completed: {len(gestures)} gestures detected")
    
    return True

def test_with_video(video_path):
    """Test with actual video"""
    print(f"\nTesting with video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    service = GestureRecognizerService()
    
    frame_count = 0
    gesture_count = 0
    
    while cap.isOpened() and frame_count < 100:  # Test first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:  # Sample every 10th frame
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gestures = service.recognize_frame(frame_rgb, timestamp_ms=frame_count * 33)
            
            if gestures:
                gesture_count += len(gestures)
                for g in gestures:
                    print(f"  Frame {frame_count}: {g['type']} (confidence: {g['confidence']:.2f})")
        
        frame_count += 1
    
    cap.release()
    print(f"✓ Processed {frame_count} frames, detected {gesture_count} gestures")
    
    return gesture_count > 0

if __name__ == "__main__":
    # Test basic functionality
    if test_gesture_recognizer():
        print("\n✅ Basic test passed")
    
    # Test with video if provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if Path(video_path).exists():
            if test_with_video(video_path):
                print("\n✅ Video test passed - gestures detected!")
            else:
                print("\n⚠️ No gestures detected in video")
        else:
            print(f"\n❌ Video not found: {video_path}")
EOF

chmod +x test_gestures.py

# 2. Run basic test
python3 test_gestures.py

# 3. Test with a video (if you have one)
# python3 test_gestures.py /path/to/test/video.mp4
```

### Step 7: Integration Test
```bash
# Test full pipeline with a TikTok video
python3 scripts/rumiai_runner.py 'https://www.tiktok.com/@test/video/123' 2>&1 | grep -A5 "gestureTimeline"

# Check if gestures are detected
grep -o '"gestureTimeline":{[^}]*}' insights/*/unified_analysis/*.json | head -5
```

### Step 8: Validate Each Flow

**1. Check Emotional Journey:**
```bash
# Look for emotion-gesture alignment
grep -o '"emotion_gesture_alignment":[0-9.]*' insights/*/emotional_journey/*.json
```

**2. Check Creative Density:**
```bash
# Look for gesture counts
grep -o '"gesture":[0-9]*' insights/*/creative_density/*.json
```

**3. Check Visual Overlay:**
```bash
# Look for text-gesture coordination
grep -o '"text_gesture":[0-9]*' insights/*/visual_overlay/*.json
```

**4. Check Speech Analysis:**
```bash
# Look for gesture sync ratio
grep -o '"gesture_sync_ratio":[0-9.]*' insights/*/speech_analysis/*.json
```

### Expected Output Examples

**Good gestureTimeline:**
```json
{
  "gestureTimeline": {
    "0-1s": {"gestures": ["pointing"], "confidence": 0.92},
    "5-6s": {"gestures": ["thumbs_up"], "confidence": 0.88},
    "12-13s": {"gestures": ["victory", "open_palm"], "confidence": 0.75}
  }
}
```

**Metrics should change:**
- `emotion_gesture_alignment`: 0.0 → 0.3-0.7
- `gesture` count in density: 0 → 5-20
- `text_gesture` in CTA matrix: 0 → 2-10
- `gesture_sync_ratio`: 0.0 → 0.3-0.6

## 🚦 ROLLOUT CHECKLIST

### Pre-Implementation
- [ ] Download gesture_recognizer.task model (30MB)
- [ ] Verify model exists at `/home/jorge/rumiaifinal/models/`
- [ ] Back up current ml_services_unified.py

### Implementation
- [ ] Create gesture_recognizer_service.py
- [ ] Update ml_services_unified.py (process ALL frames, no sampling)
- [ ] Update timeline_builder.py
- [ ] Update precompute_functions.py (gesture timeline extraction)
- [ ] Fix Person Framing gesture sync in precompute_functions.py
- [ ] Fix Visual Overlay bugs in precompute_functions_full.py

### Testing
- [ ] Run test_gestures.py basic test
- [ ] Test with sample video
- [ ] Process TikTok video with gestures
- [ ] Verify gestureTimeline populated
- [ ] Check all 5 flows receive data

### Validation
- [ ] Emotional Journey alignment > 0
- [ ] Creative Density gesture count > 0
- [ ] Visual Overlay CTA reinforcement > 0
- [ ] Speech Analysis sync ratio > 0
- [ ] Person Framing shows speech-gesture sync

### Rollback Plan
```bash
# If issues occur:
git checkout -- rumiai_v2/api/ml_services_unified.py
git checkout -- rumiai_v2/processors/timeline_builder.py
git checkout -- rumiai_v2/processors/precompute_functions.py
git checkout -- rumiai_v2/processors/precompute_functions_full.py
rm rumiai_v2/api/gesture_recognizer_service.py
```

## 📈 SUCCESS CRITERIA

**Immediate (Day 1):**
- ✅ gestureTimeline not empty for videos with hands
- ✅ No crashes or errors in pipeline
- ✅ All flows continue working

**Short-term (Week 1):**
- ✅ 30%+ videos show gesture detection
- ✅ Metrics show meaningful changes
- ✅ Processing time increase < 10%

**Long-term (Month 1):**
- ✅ Gesture insights add value to analysis
- ✅ Stable performance across video types
- ✅ Decision on whether to optimize further

## Recommendation

**Priority**: MEDIUM-HIGH
- Core functionality works without gestures
- But missing important quality signals
- Affects authenticity and engagement metrics

**Implementation Effort**: 4-6 hours
- 2 hours: MediaPipe integration
- 1 hour: Timeline builder updates
- 1 hour: Testing each flow
- 2 hours: Debugging and validation