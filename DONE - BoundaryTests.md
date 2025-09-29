# Boundary Tests - Complete Testing Plan

## Discovery Process: How We Got Here

### Initial Assumption
We originally thought we needed to test EVERY feature at temporal window boundaries (hook/middle/closing). With 40+ features × 3 boundaries = 120+ tests needed. This would be exhausting and time-consuming.

### Key Discovery #1: Shared Boundary Logic
By examining `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`, we discovered ALL features use the SAME `process_segment()` function with identical timestamp filtering:

```python
# Every feature uses this pattern:
segment_data = [item for item in timeline
                if start <= item.get('timestamp', 0) < end]
```

This means if boundary filtering works for ONE feature, it works for ALL similar features.

### Key Discovery #2: Feature Categories
Through documentation review, we identified that features fall into distinct categories:

1. **Discrete Events** (90% of features)
   - Text overlays, expressions, words, gestures, objects
   - Use standard timestamp filtering: `if start <= timestamp < end`
   - ONE test validates ALL of these

2. **Continuous Sampling** (Audio features)
   - Energy levels, pitch, RMS values
   - Sample audio frames at intervals
   - Need separate test for frame selection

3. **Spanning Events** (Scene features)
   - Scenes have both start_time and end_time
   - Can span across boundaries
   - Use different filtering: `if not (end <= start or start >= end)`
   - Need separate test for spanning behavior

4. **Sampled Features** (FEAT, some YOLO)
   - Don't analyze every frame but still use discrete timestamps
   - Covered by discrete events test

### The Breakthrough
We realized we only need ONE test per category, not one test per feature. This reduces testing from 120+ tests to just 5 focused tests!

---

## Boundary Testing Plan (5 Tests Total)

### Test 1: Hook/Middle Boundary - Discrete Events ✅

**Purpose**: Verify the 3-second boundary for ALL discrete event features

**Test Video Specifications**:
- Duration: 10 seconds
- Frame rate: 30 FPS recommended
- Resolution: 1080x1920 (vertical)

**Event Timeline**:
```
2.95s: Text overlay "HOOK-TEXT" appears
2.99s: Person enters frame (for object detection)
3.00s: Text overlay "MIDDLE-TEXT" appears
3.01s: Different person enters frame
3.05s: Person makes thumbs-up gesture
```

**Recording Instructions**:
1. Use video editor with precise timestamp control
2. Add text overlays at EXACT timestamps (use keyframes)
3. Have people enter frame at precise moments
4. Include clear gesture at 3.05s

**What to Verify**:
```json
{
  "hook": {
    "overlay_unique_count": 1,    // Only "HOOK-TEXT"
    "person_count": 1,             // First person at 2.99s
    "gesture_count": 0             // No gestures before 3.0s
  },
  "middle_segments": [{
    "overlay_unique_count": 1,    // Only "MIDDLE-TEXT"
    "person_count": 1,             // Second person at 3.01s
    "gesture_count": 1             // Thumbs-up at 3.05s
  }]
}
```

**What This Proves**: ALL discrete event features respect the 3.0s boundary

---

### Test 2: Middle/Closing Boundary - Discrete Events ✅

**Purpose**: Verify the last-3-seconds boundary for discrete events

**Test Video Specifications**:
- Duration: 10 seconds
- Last 3 seconds = 7.0s to 10.0s

**Event Timeline**:
```
6.95s: Text overlay "MIDDLE-END" appears
6.99s: Person shows happy expression
7.00s: Text overlay "CLOSING-START" appears
7.01s: Person shows sad expression
7.05s: Person makes wave gesture
```

**Recording Instructions**:
1. Use same person throughout for expression changes
2. Make expressions very obvious (exaggerated)
3. Ensure text overlays are clearly visible

**What to Verify**:
```json
{
  "middle_segments": [{
    "overlay_unique_count": 1,    // Only "MIDDLE-END"
    "expression_count": 1,         // Happy at 6.99s
    "dominant_emotion": "happy"
  }],
  "closing": {
    "overlay_unique_count": 1,    // Only "CLOSING-START"
    "expression_count": 2,         // Sad at 7.01s, plus any at 7.05s
    "dominant_emotion": "sad",
    "gesture_count": 1             // Wave at 7.05s
  }
}
```

---

### Test 3: Scene Spanning Boundary 🎬

**Purpose**: Verify scenes that span boundaries appear in BOTH windows

**Test Video Specifications**:
- Duration: 10 seconds
- 4 distinct scenes with cuts

**Scene Timeline**:
```
Scene 1: 0.0s - 2.5s   (Indoor shot)     → Fully in hook
Scene 2: 2.5s - 3.5s   (Outdoor shot)    → SPANS hook/middle
Scene 3: 3.5s - 7.0s   (Different room)  → Fully in middle
Scene 4: 7.0s - 10.0s  (Close-up shot)   → Fully in closing
```

**Recording Instructions**:
1. Film 4 completely different scenes
2. Edit together with hard cuts at exact timestamps
3. Make scenes visually distinct (different locations/angles)

**What to Verify**:
```json
{
  "hook": {
    "scene_count": 2,              // Scene 1 + Scene 2 (spanning)
    "scene_changes": [0.0, 2.5]    // When scenes start
  },
  "middle_segments": [{
    "scene_count": 2,              // Scene 2 (spanning) + Scene 3
    "scene_changes": [2.5, 3.5]    // Both scenes detected
  }],
  "closing": {
    "scene_count": 1,              // Only Scene 4
    "scene_changes": [7.0]
  }
}
```

**Critical Finding**: Scene 2 (2.5s-3.5s) appears in BOTH hook AND middle

---

### Test 4: Audio Continuous Sampling 🎵

**Purpose**: Verify audio frame sampling respects boundaries

**Test Video Specifications**:
- Duration: 10 seconds
- Clear audio changes at boundaries

**Audio Timeline**:
```
0.00s - 2.99s: LOUD consistent tone (1000Hz sine wave)
3.00s - 6.99s: Complete SILENCE
7.00s - 10.0s: MEDIUM volume tone (500Hz sine wave)
```

**Recording Instructions**:
1. Use audio generator for precise tones
2. Edit audio track separately from video
3. Ensure clean cuts at exactly 3.00s and 7.00s
4. Export at 16kHz sample rate minimum

**What to Verify**:
```json
{
  "hook": {
    "energy_level": 0.8,           // High (normalized)
    "energy_variance": 0.01,       // Low (consistent tone)
    "has_speech": false            // Pure tone, no speech
  },
  "middle_segments": [{
    "energy_level": 0.0,           // Near zero
    "energy_variance": 0.0,        // No variance in silence
    "has_speech": false
  }],
  "closing": {
    "energy_level": 0.4,           // Medium
    "energy_variance": 0.01,       // Low (consistent)
    "has_speech": false
  }
}
```

---

### Test 5: Short Video Edge Case ⚠️

**Purpose**: Verify overlapping windows behavior when video is too short

**Test Video Specifications**:
- Duration: 5 seconds (deliberately short)
- Hook: 0-3s
- Closing: 2-5s (last 3 seconds)
- **Overlap zone: 2-3s**

**Event Timeline**:
```
1.0s: Text overlay "IN-HOOK" appears
2.5s: Scene change occurs (in overlap zone!)
4.0s: Text overlay "IN-CLOSING" appears
```

**Recording Instructions**:
1. Keep video exactly 5 seconds
2. Place scene change in the overlap zone deliberately
3. Make text overlays very clear

**What to Verify**:
```json
{
  "hook": {
    "exists": true,
    "duration": 3.0,
    "overlay_unique_count": 1,    // "IN-HOOK"
    "scene_count": 2               // Both scenes (change at 2.5s)
  },
  "middle_segments": null,          // NO MIDDLE (too short)
  "closing": {
    "exists": true,
    "duration": 3.0,
    "overlay_unique_count": 1,    // "IN-CLOSING"
    "scene_count": 2               // Both scenes (change at 2.5s)
  }
}
```

**Critical Finding**: Scene at 2.5s appears in BOTH hook and closing due to overlap

---

## Implementation Guide

### Creating Test Videos

**Option A: Manual Creation (Most Reliable)**
1. Film raw footage with timer visible
2. Edit in DaVinci Resolve/Premiere/Final Cut
3. Use frame-accurate editing (not time-based)
4. Add text overlays with keyframes
5. Export at exactly 30 FPS

**Option B: Programmatic Generation**
```python
import cv2
import numpy as np
from moviepy.editor import VideoClip, TextClip, CompositeVideoClip

def create_boundary_test_1():
    # Create 10-second video at 30 FPS
    duration = 10
    fps = 30

    def make_frame(t):
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

        # Add text overlays at precise times
        if 2.95 <= t < 3.00:
            # Add "HOOK-TEXT" overlay
            pass
        elif 3.00 <= t < 3.05:
            # Add "MIDDLE-TEXT" overlay
            pass

        return frame

    clip = VideoClip(make_frame, duration=duration)
    clip.write_videofile("boundary_test_1.mp4", fps=fps)
```

### Running Tests

```bash
# Process test videos
python3 test_manual_videos.py boundary_test_1.mp4
python3 test_manual_videos.py boundary_test_2.mp4
python3 test_manual_videos.py boundary_test_3.mp4
python3 test_manual_videos.py boundary_test_4.mp4
python3 test_manual_videos.py boundary_test_5.mp4
```

### Validating Results

```python
import json

def validate_boundary_test_1(video_id):
    # Load the temporal windows output
    with open(f'insights/{video_id}_temporal_windows_updated.json') as f:
        data = json.load(f)

    windows = data['temporal_windows']

    # Check hook
    assert windows['hook']['overlay_unique_count'] == 1, "Hook should have 1 overlay"
    assert windows['hook']['person_count'] == 1, "Hook should have 1 person"

    # Check middle
    middle = windows['middle_segments'][0] if windows['middle_segments'] else {}
    assert middle.get('overlay_unique_count') == 1, "Middle should have 1 overlay"
    assert middle.get('gesture_count') == 1, "Middle should have 1 gesture"

    print("✅ Boundary Test 1 PASSED")
```

---

## Why This Plan Works

### Testing Efficiency
- **Without this plan**: 40+ features × 3 boundaries = 120+ tests
- **With this plan**: 5 focused tests
- **Reduction**: 96% fewer tests with same confidence

### Coverage Guarantee
1. Test 1-2 prove discrete event filtering works
2. Test 3 proves scene spanning works
3. Test 4 proves audio sampling works
4. Test 5 proves edge cases work
5. Together they prove the ENTIRE boundary system works

### Mathematical Proof
Since all discrete features use:
```python
[item for item in timeline if start <= item.get('timestamp', 0) < end]
```

If this works for text overlays, it MUST work for expressions, words, gestures, etc. They literally execute the same code!

---

## Common Pitfalls to Avoid

1. **Imprecise Timestamps**: Even 0.01s matters! Use frame-accurate editing
2. **Wrong FPS**: Ensure exactly 30 FPS export for predictable timestamps
3. **Audio Drift**: Audio can desync from video - check carefully
4. **Scene Detection Sensitivity**: Make scene changes obvious (different locations)
5. **Expression Detection**: Ensure face is clearly visible and well-lit

---

## Conclusion

This 5-test plan provides complete boundary validation with 96% less effort than naive testing. The key insight is that shared code means shared behavior - if boundaries work for one feature type, they work for all similar types.