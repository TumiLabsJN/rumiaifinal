# Frame Processing Pipeline - Technical Documentation

## Overview
This document provides a comprehensive guide to the frame processing pipeline in RumiAI v2, detailing how video frames are extracted and processed by various ML models.

## 1. Frame Extraction Process

### 1.1 Adaptive Frame Extraction
**Location**: `automated_video_pipeline.py`, lines 1051-1112

The system uses adaptive frame sampling based on video duration to balance processing efficiency and analysis quality:

```python
# automated_video_pipeline.py, lines 1051-1067
def extract_frames_ffmpeg(video_path, output_dir, video_duration):
    """Extract frames using adaptive FPS based on video duration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Adaptive FPS selection
    if video_duration < 30:
        fps = 5  # 5 FPS for short videos
    elif video_duration < 60:
        fps = 3  # 3 FPS for medium videos
    else:
        fps = 2  # 2 FPS for long videos
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={fps}',
        '-q:v', '2',  # High quality JPEG
        f'{output_dir}/frame_%06d.jpg'
    ]
```

**Output**: JPEG frames saved as `frame_000001.jpg`, `frame_000002.jpg`, etc.

### 1.2 Alternative Frame Extraction
**Location**: `python/frame_sampler.py`, lines 34-89

An alternative OpenCV-based frame extraction with the same adaptive logic:

```python
# python/frame_sampler.py, lines 65-75
if duration < 30:
    target_fps = 5
elif duration <= 60:
    target_fps = 3
else:
    target_fps = 2

frame_interval = int(video_fps / target_fps)
```

## 2. ML Model Processing

### 2.1 YOLO Object Detection + DeepSort Tracking
**Location**: `python/object_tracking.py`, lines 103-175

#### Processing Rate Calculation:
```python
# python/object_tracking.py, lines 108-117
# Adaptive frame skipping based on video duration
video_length = total_frames / fps
if video_length < 30:
    frame_skip = 12  # ~2.5 FPS for 30fps video
elif video_length < 60:
    frame_skip = 15  # ~2 FPS for 30fps video
else:
    frame_skip = 15  # ~2 FPS for 30fps video
```

#### Processing Logic:
- Reads video directly (not extracted frames)
- Processes every Nth frame based on `frame_skip`
- Maintains object tracking continuity using DeepSort
- GPU acceleration if available

**Output Format**:
```json
{
  "frame_1": {
    "timestamp": "0:00:00.033",
    "objects": [
      {
        "id": 1,
        "class": "person",
        "confidence": 0.92,
        "bbox": [100, 200, 300, 400]
      }
    ]
  }
}
```

### 2.2 MediaPipe Human Analysis
**Location**: `python/mediapipe_human_detector.py`, lines 456-652

#### Processing Logic:
```python
# python/mediapipe_human_detector.py, lines 478-482
# Process all extracted frames
extracted_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
for i, frame_path in enumerate(extracted_frames):
    frame = cv2.imread(frame_path)
    results = process_frame(frame)
```

**Components Analyzed**:
- Face detection & landmarks (468 points)
- Hand detection & tracking (21 landmarks per hand)
- Pose detection (33 body landmarks)
- Holistic model for coordinated detection

**Output**: Detailed pose, face, and hand landmark data for each frame

### 2.3 OCR Text Detection
**Location**: `detect_tiktok_creative_elements.py`, lines 309-373

#### Adaptive Sampling Strategy:
```python
# detect_tiktok_creative_elements.py, lines 317-329
if num_frames <= 100:  # ~20 seconds at 5fps
    step = 2  # Process every 2nd frame (~2.5 FPS effective)
elif num_frames <= 200:  # ~40-60 seconds
    step = 3  # Process 1 of every 3 frames (~1.67 FPS effective)
else:
    step = 1  # Process every frame for longer videos
```

**Processing Features**:
- EasyOCR with GPU support
- Text bounding box detection
- Font size estimation
- Text persistence tracking across frames

### 2.4 Enhanced Human Analysis
**Location**: `enhanced_human_analyzer.py`, lines 234-432

#### Processing All Extracted Frames:
```python
# enhanced_human_analyzer.py, lines 256-260
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    results = analyze_frame(frame_path)
```

**Advanced Features**:
- Body pose analysis (33 landmarks)
- Gaze detection and eye contact ratio
- Scene segmentation (person vs background)
- Action recognition (walking, sitting, dancing, etc.)
- Temporal marker generation

## 3. Pipeline Orchestration

### 3.1 Main Orchestrator
**Location**: `server/services/LocalVideoAnalyzer.js`, lines 145-367

The orchestrator runs all ML models in parallel:

```javascript
// server/services/LocalVideoAnalyzer.js, lines 234-256
const analysisPromises = [];

// Run all analyses in parallel
analysisPromises.push(this.runWhisperTranscription(videoId, videoPath));
analysisPromises.push(this.runYoloDetection(videoId, framesDir));
analysisPromises.push(this.runMediaPipeAnalysis(videoId, framesDir));
analysisPromises.push(this.runEnhancedHumanAnalysis(videoId, framesDir));
analysisPromises.push(this.runOCRAnalysis(videoId, framesDir));
analysisPromises.push(this.runSceneDetection(videoId, videoPath));
analysisPromises.push(this.runCLIPLabeling(videoId, videoPath));
analysisPromises.push(this.runContentModeration(videoId, videoPath));

const results = await Promise.all(analysisPromises);
```

### 3.2 Timeline Synchronization
**Location**: `python/timeline_synchronizer.py`

All model outputs are synchronized to a unified timeline with consistent timestamps.

## 4. Dependencies and Requirements

### Python Dependencies:
```txt
opencv-python==4.8.1.78
ultralytics==8.0.200  # YOLO
mediapipe==0.10.7
easyocr==1.7.0
torch>=2.0.0
numpy>=1.24.0
```

### System Requirements:
- FFmpeg for video processing
- CUDA-capable GPU (optional but recommended)
- Minimum 8GB RAM (16GB+ recommended)
- Python 3.8+

## 5. Performance Characteristics

For a 60-second video at 30fps:
- **Frame Extraction**: 180 frames at 3 FPS
- **YOLO Processing**: ~120 frames (2 FPS effective)
- **MediaPipe**: All 180 extracted frames
- **OCR**: ~60 frames (1 of 3 frames)
- **Enhanced Human**: All 180 extracted frames
- **Total Processing Time**: ~2-5 minutes depending on GPU

## 6. Output Structure

All outputs follow a consistent timeline format:
```json
{
  "0-1s": { /* aggregated data for first second */ },
  "1-2s": { /* aggregated data for second second */ },
  // ... continues for entire video duration
}
```

This structure allows for easy temporal alignment across all ML models and enables synchronized multi-modal analysis.