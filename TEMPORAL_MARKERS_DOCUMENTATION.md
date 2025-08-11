# Temporal Markers - Complete Technical Documentation

**Version**: 2.0.0  
**Last Updated**: 2025-01-11  
**Status**: Fully integrated as 8th analysis type in Python-only pipeline

## 1. Overview

Temporal markers are a sophisticated feature in RumiAI that analyzes specific time windows in videos to extract engagement patterns. The system focuses on two critical periods:

- **First X seconds** (default: 5 seconds) - The "hook window" where viewers decide to continue watching
- **Last Y%** (default: 15%) - The "CTA window" where creators typically place calls-to-action

## Integration Status

Temporal markers have been fully integrated as the **8th analysis type** in RumiAI's Python-only processing pipeline:

- ✅ Added to all 3 documentation files (ML_FEATURES, FlowStructure, Codemappingfinal)
- ✅ Integrated with backward compatibility (3-file output structure)
- ✅ Works in both URL and Video ID processing flows
- ✅ Zero-cost processing ($0.00) with instant execution (0.001s)

## 2. Architecture

### 2.1 Core Components

```
┌─────────────────────┐
│   Video Input       │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  ML Analysis Suite  │
│  - YOLO             │
│  - Whisper          │
│  - MediaPipe        │
│  - OCR              │
│  - Scene Detection  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Temporal Processor  │
│  - First 5s         │
│  - Last 15%         │
│  - Peak Moments     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Python-Only Mode   │
│  - No Claude API    │
│  - $0.00 cost       │
│  - 0.001s speed     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 3-File JSON Output  │
│  - complete.json    │
│  - ml.json          │
│  - result.json      │
└─────────────────────┘
```

### 2.2 Key Files and Locations

#### Python Components:
- **Main Processor**: `rumiai_v2/processors/temporal_markers.py`
- **Precompute Registration**: `rumiai_v2/processors/precompute_functions.py`
- **Runner Integration**: `scripts/rumiai_runner.py`
- **Legacy Generator**: `python/TemporalMarkerGenerator.py`
- **Extractors**: `python/temporal_marker_extractors.py`
- **Fixed Implementation**: `python/generate_temporal_markers_fixed.py`

#### Output Structure:
- **Location**: `insights/{video_id}/temporal_markers/`
- **Files Generated**:
  - `temporal_markers_complete_{timestamp}.json` - Full response with metadata
  - `temporal_markers_ml_{timestamp}.json` - Unprefixed ML format
  - `temporal_markers_result_{timestamp}.json` - Prefixed RESULT format

#### Configuration:
- **Settings**: `rumiai_v2/config/settings.py` (hardcoded enabled)
- **Main Config**: `config/temporal_markers.json`

## 3. Time Window Configuration

### 3.1 First X Seconds (Hook Window)
```python
# rumiai_v2/processors/temporal_markers.py, line 98
first_x_seconds = 5  # Hardcoded to 5 seconds
```

**Purpose**: Analyze the critical opening moments where viewers decide to continue watching.

### 3.2 Last Y% (CTA Window)
```python
# rumiai_v2/processors/temporal_markers.py, line 99
cta_start = duration * 0.85  # Last 15% of video
cta_end = duration
```

**Purpose**: Detect calls-to-action and engagement drivers in the video's conclusion.

## 4. Data Processing Pipeline

### 4.1 Input Data Structure
```json
{
  "video_id": "123456789",
  "duration": 60.0,
  "yolo_data": { /* object detection timeline */ },
  "whisper_data": { /* speech transcription */ },
  "mediapipe_data": { /* human analysis */ },
  "ocr_data": { /* text detection */ },
  "scene_data": { /* scene changes */ }
}
```

### 4.2 Processing Steps

1. **Data Loading**
   ```python
   # Load ML analysis outputs
   yolo_data = load_json(deps['yolo'])
   mediapipe_data = load_json(deps['mediapipe'])
   ocr_data = load_json(deps['ocr'])
   whisper_data = load_json(deps['whisper'])
   scene_data = load_json(deps['scenes'])
   ```

2. **Time Window Extraction**
   ```python
   # Extract first 5 seconds
   first_5s_data = extract_time_window(all_data, 0, 5)
   
   # Extract CTA window (last 15%)
   cta_start = duration * 0.85
   cta_data = extract_time_window(all_data, cta_start, duration)
   ```

3. **Density Calculation**
   ```python
   # Calculate activity density per second
   density_weights = {
       'text': 3.0,      # Text overlays have high impact
       'scene_change': 2.0,  # Scene changes are significant
       'speech': 1.5,    # Speech adds engagement
       'object': 1.0,    # Objects provide context
       'gesture': 1.2    # Gestures enhance communication
   }
   ```

4. **Pattern Extraction**
   - Emotion sequences from speech content
   - Gesture patterns from MediaPipe
   - Object appearances from YOLO
   - Text overlay timing from OCR
   - Speech segments from Whisper

## 5. Output Format

### 5.1 Complete Temporal Marker Structure
```json
{
  "first_5_seconds": {
    "density_progression": [0, 2, 5, 8, 6],  // Per-second density (0-10)
    "text_moments": [
      {
        "timestamp": "0-1s",
        "text": "Amazing trick!",
        "position": "center",
        "size": "large"
      }
    ],
    "emotion_sequence": ["neutral", "excited", "positive", "excited", "positive"],
    "gesture_moments": [
      {
        "timestamp": "2-3s",
        "gesture": "pointing",
        "confidence": 0.92
      }
    ],
    "object_appearances": [
      {
        "timestamp": "1-2s",
        "object": "person",
        "action": "appears",
        "position": "center"
      }
    ],
    "speech_segments": [
      {
        "timestamp": "0.5-2.5s",
        "text": "Hey everyone, watch this!",
        "emotion": "excited"
      }
    ],
    "scene_changes": [1.5, 3.2]  // Timestamps of scene transitions
  },
  
  "cta_window": {
    "time_range": "51.0-60.0s",
    "cta_appearances": [
      {
        "timestamp": "55-56s",
        "type": "text",
        "content": "Follow for more!",
        "confidence": 0.95
      }
    ],
    "gesture_sync": {
      "pointing_during_cta": 0.8,  // 80% of CTAs have pointing
      "hand_gestures": ["pointing", "thumbs_up"]
    },
    "object_focus": [
      {
        "object": "product",
        "screen_time": 0.7,  // 70% of CTA window
        "position_changes": 2
      }
    ],
    "speech_emphasis": {
      "volume_increase": true,
      "speed_change": -0.1,  // 10% slower
      "keywords": ["follow", "like", "subscribe"]
    },
    "text_overlays": [
      {
        "text": "Link in bio",
        "duration": 3.5,
        "position": "bottom"
      }
    ]
  },
  
  "peak_moments": [
    {
      "timestamp": "8-9s",
      "density": 9.5,
      "triggers": ["scene_change", "text_reveal", "gesture"]
    },
    {
      "timestamp": "23-24s",
      "density": 8.2,
      "triggers": ["object_reveal", "speech_emphasis"]
    }
  ],
  
  "engagement_curve": [
    {"second": 0, "engagement": 0.2},
    {"second": 1, "engagement": 0.5},
    // ... continues for entire video
  ],
  
  "metadata": {
    "video_id": "123456789",
    "duration": 60.0,
    "generated_at": "2024-01-01T12:00:00Z",
    "version": "2.0",
    "config": {
      "first_x_seconds": 5,
      "cta_percentage": 0.15,
      "density_weights": { /* ... */ }
    }
  }
}
```

## 6. ML Data Sources

### 6.1 YOLO (Object Detection)
- **Input**: Video frames at ~2-2.5 FPS
- **Output**: Objects with bounding boxes, IDs, and tracking
- **Used for**: Object appearances, focus analysis

### 6.2 Whisper (Speech-to-Text)
- **Input**: Audio track
- **Output**: Timestamped transcriptions
- **Used for**: Speech segments, CTA keywords, emotion detection

### 6.3 MediaPipe (Human Analysis)
- **Input**: Extracted frames (2-5 FPS)
- **Output**: Pose, face, and hand landmarks
- **Used for**: Gesture detection, human presence

### 6.4 OCR (Text Detection)
- **Input**: Extracted frames
- **Output**: Text with bounding boxes
- **Used for**: Text overlays, on-screen CTAs

### 6.5 PySceneDetect
- **Input**: Video file
- **Output**: Scene change timestamps
- **Used for**: Scene transition markers

## 7. Algorithms and Logic

### 7.1 Density Calculation Algorithm
```python
def calculate_density(events, weights):
    """Calculate activity density for a time window"""
    total_weight = 0
    
    for event_type, count in events.items():
        weight = weights.get(event_type, 1.0)
        total_weight += count * weight
    
    # Normalize to 0-10 scale
    normalized = min(10, total_weight / 3)
    return round(normalized, 1)
```

### 7.2 Emotion Detection
```python
EMOTION_KEYWORDS = {
    'excited': ['amazing', 'wow', 'incredible', 'awesome'],
    'positive': ['great', 'love', 'perfect', 'beautiful'],
    'negative': ['bad', 'hate', 'terrible', 'worst'],
    'urgent': ['now', 'quick', 'fast', 'hurry'],
    'warning': ['careful', 'watch out', 'attention', 'warning']
}

def detect_emotion(text):
    """Detect emotion from speech text"""
    text_lower = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return emotion
    return 'neutral'
```

### 7.3 Peak Moment Detection
```python
def find_peak_moments(density_curve, threshold_multiplier=1.5):
    """Find moments of peak engagement"""
    average_density = sum(density_curve) / len(density_curve)
    threshold = average_density * threshold_multiplier
    
    peaks = []
    for i, density in enumerate(density_curve):
        if density > threshold:
            # Check if local maximum
            is_peak = True
            if i > 0 and density_curve[i-1] >= density:
                is_peak = False
            if i < len(density_curve)-1 and density_curve[i+1] >= density:
                is_peak = False
            
            if is_peak:
                peaks.append({
                    'timestamp': f'{i}-{i+1}s',
                    'density': density
                })
    
    # Return top 5 peaks
    return sorted(peaks, key=lambda x: x['density'], reverse=True)[:5]
```

## 8. Integration with Python-Only Pipeline

### 8.1 Python-Only Processing
Temporal markers now bypass Claude API entirely:

```python
# scripts/rumiai_runner.py
if self.settings.use_python_only_processing:
    # Direct Python computation - no Claude API
    from rumiai_v2.processors.temporal_markers import generate_markers
    markers = generate_markers(analysis.to_dict())
    
    # Save in 3-file backward compatible format
    self.save_analysis_result(video_id, 'temporal_markers', markers)
```

### 8.2 Backward Compatibility
The system maintains 3-file output for legacy compatibility:

```python
def save_analysis_result(self, video_id: str, analysis_type: str, data: dict) -> Path:
    """Save analysis result in 3-file backward compatible format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = self.output_dir / video_id / analysis_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save complete file (full response with metadata)
    complete_data = {
        "prompt_type": analysis_type,
        "success": True,
        "response": json.dumps(data),
        "parsed_response": self.convert_to_unprefixed_format(data, analysis_type)
    }
    
    # 2. Save ml file (unprefixed format)
    ml_data = self.convert_to_ml_format(data, analysis_type)
    
    # 3. Save result file (prefixed format for legacy)
    result_data = data  # Already in prefixed format
    
    return output_dir
```

### 8.3 Settings Configuration
```python
# rumiai_v2/config/settings.py
self.precompute_enabled_prompts = {
    'creative_density': True,
    'emotional_journey': True,
    'person_framing': True,
    'scene_pacing': True,
    'speech_analysis': True,
    'visual_overlay_analysis': True,
    'metadata_analysis': True,
    'temporal_markers': True  # Added as 8th type
}
```

## 9. Safety and Limits

### 9.1 Size Limits
- Maximum text length: 100 characters
- Maximum events in first 5s: 50
- Maximum CTA events: 20
- Maximum marker JSON size: 50KB
- Maximum processing time: 60 seconds

### 9.2 Error Handling
```python
try:
    markers = generate_temporal_markers(video_data)
except MemoryError:
    logger.error("Insufficient memory for temporal markers")
    return None
except TimeoutError:
    logger.error("Temporal marker generation timed out")
    return None
except Exception as e:
    logger.error(f"Temporal marker generation failed: {e}")
    return None
```

### 9.3 Circuit Breaker Pattern
```javascript
// server/services/TemporalMarkerService.js
class TemporalMarkerService {
    constructor() {
        this.failureCount = 0;
        this.maxFailures = 3;
        this.isEnabled = true;
    }
    
    async generateMarkers(videoData) {
        if (!this.isEnabled) {
            return null;
        }
        
        try {
            const markers = await this._generate(videoData);
            this.failureCount = 0;  // Reset on success
            return markers;
        } catch (error) {
            this.failureCount++;
            if (this.failureCount >= this.maxFailures) {
                this.isEnabled = false;
                logger.error('Temporal markers disabled due to repeated failures');
            }
            throw error;
        }
    }
}
```

## 10. Performance Characteristics

### 10.1 Processing Time
- Average: 2-5 seconds per video
- Depends on: Video duration, ML data size
- Timeout: 60 seconds

### 10.2 Memory Usage
- Base: ~50MB
- Per minute of video: ~10MB
- Maximum: 500MB (safety limit)

### 10.3 CPU Usage
- Single-threaded Python process
- No GPU required
- Minimal CPU impact (<5% on modern systems)

## 11. Dependencies

### 11.1 Python Dependencies
No additional dependencies - uses existing packages:
- numpy (for numerical operations)
- Standard library (json, datetime, etc.)

### 11.2 System Requirements
- Python 3.8+
- 1GB free memory
- Access to ML analysis outputs

### 11.3 Configuration Files
- `config/temporal_markers.json` - Main configuration
- Environment variables:
  - `ENABLE_TEMPORAL_MARKERS` (true/false)
  - `TEMPORAL_ROLLOUT_PERCENTAGE` (0-100)
  - `TEMPORAL_MARKERS_TIMEOUT` (seconds)

## 12. Monitoring and Metrics

### 12.1 Metrics Collected
```json
{
  "video_id": "123456789",
  "processing_time_ms": 2341,
  "marker_size_bytes": 8456,
  "first_5s_events": 23,
  "cta_events": 8,
  "peak_moments": 3,
  "success": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### 12.2 Metric Storage
- Location: `metrics/temporal_markers/`
- Format: JSON files by date
- Retention: 30 days

## 13. Usage Examples

### 13.1 Command Line
```bash
# Generate temporal markers for a video
python python/generate_temporal_markers_fixed.py \
    --video-path /path/to/video.mp4 \
    --video-id 123456789 \
    --deps '{
        "yolo": "/path/to/yolo.json",
        "mediapipe": "/path/to/mediapipe.json",
        "ocr": "/path/to/ocr.json",
        "whisper": "/path/to/whisper.json",
        "scenes": "/path/to/scenes.json"
    }'
```

### 13.2 Python Integration
```python
from rumiai_v2.processors.temporal_markers import TemporalMarkerProcessor

# Initialize processor
processor = TemporalMarkerProcessor()

# Generate markers
markers = processor.generate_markers(unified_analysis)

# Save to file
with open('temporal_markers.json', 'w') as f:
    json.dump(markers, f, indent=2)
```

### 13.3 Node.js Integration
```javascript
const TemporalMarkerService = require('./services/TemporalMarkerService');

const service = new TemporalMarkerService();
const markers = await service.generateMarkers({
    videoPath: '/path/to/video.mp4',
    videoId: '123456789',
    dependencies: {
        yolo: '/path/to/yolo.json',
        // ... other dependencies
    }
});
```

## 14. Recent Updates (2025-01-11)

### 14.1 Full Integration as 8th Analysis Type
- ✅ Added to `ML_FEATURES_DOCUMENTATION_V2.md` as Section 8
- ✅ Added to `FlowStructure.md` analysis table and descriptions
- ✅ Added to `Codemappingfinal.md` Python Compute Functions table
- ✅ Integrated with 3-file backward compatibility system
- ✅ Works in both URL and Video ID processing flows

### 14.2 Key Changes Made
1. **Documentation Updates**: Updated all 3 core documentation files to reflect 8 analysis types
2. **Backward Compatibility**: Implemented 3-file output (complete/ml/result) for legacy support
3. **Python-Only Mode**: Fully integrated with zero-cost Python processing pipeline
4. **Unified Save Method**: Uses common `save_analysis_result()` for consistency

### 14.3 Lessons Learned
- Documentation files are part of the system architecture, not just reference
- Backward compatibility is critical for legacy system integration
- Prefix conversion between ML and RESULT formats must be handled properly
- All 8 analysis types must be consistently reflected across documentation

## 15. Future Enhancements

### 15.1 Planned Features
- Configurable time windows (not just 5s and 15%)
- Machine learning-based window detection
- Real-time temporal marker generation
- Multi-language CTA detection
- Emotion detection from facial expressions

### 15.2 Optimization Opportunities
- Parallel processing of time windows
- Caching of frequently accessed patterns
- GPU acceleration for pattern matching
- Streaming processing for long videos

This comprehensive documentation provides all necessary information to understand, use, and refactor the temporal markers system in RumiAI.