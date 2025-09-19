# Local Video Testing Guide

## Quick Start

Process any local video file directly:
```bash
python3 scripts/local_video_runner.py <video_path>
```

### Example
```bash
python3 scripts/local_video_runner.py test_videos/video_01_highenergy_peaks.mp4
```

### Output Location
Results are saved to: `insights/{video_id}/`

Each analysis type creates a separate folder:
- `speech_analysis/`
- `scene_pacing/`
- `visual_overlay_analysis/`
- `temporal_markers/`
- `creative_density/`
- `emotional_journey/`
- `person_framing/`
- `metadata_analysis/`

### File Format
Each analysis produces a `*_complete_YYYYMMDD_HHMMSS.json` file containing:
```json
{
  "prompt_type": "analysis_type",
  "success": true,
  "parsed_response": {
    // The actual analysis results in 6-block format
  }
}
```

## Prerequisites

### Required Dependencies
```bash
# Check Python version (3.8+ required)
python3 --version

# Check ML dependencies are installed
python3 -c "from rumiai_v2.api import MLServices; print('✓ ML Services available')"

# Check if video processing tools are available
ffprobe -version > /dev/null 2>&1 && echo "✓ ffmpeg installed" || echo "✗ Install ffmpeg"
```

### Supported Video Formats
- MP4 (recommended)
- Any format supported by OpenCV and ffmpeg
- Video must have an audio track (even if silent) for speech analysis

## How It Works

1. **ML Analysis**: Runs YOLO, Whisper, MediaPipe, OCR, Scene Detection, Audio Energy, and Emotion Detection
2. **Timeline Building**: Combines all ML outputs into a unified timeline
3. **Temporal Markers**: Generates engagement markers and patterns
4. **Precompute Functions**: Runs 7 analysis types to generate final insights
5. **Results Saved**: Each analysis saved as JSON in `insights/{video_id}/`

## Common Issues

| Issue | Solution |
|-------|----------|
| `MLServices missing` | Ensure you're in the project root directory |
| `No audio track` | Add silent audio track with: `ffmpeg -i input.mp4 -f lavfi -i anullsrc -shortest -c:v copy output.mp4` |
| `Out of memory` | Process shorter videos or reduce resolution |

## Next Steps

After processing a video:

1. **Review the outputs** in `insights/{video_id}/`
2. **Create a golden dataset** based on what you actually filmed (see `golden_dataset_v2_instructions.md`)
3. **Compare results** with golden dataset (comparison script coming soon)

---

## TODO: Comparison Script

*A working comparison script that validates outputs against golden datasets will be added here.*

## TODO: Integration with Golden Dataset Workflow

*Instructions for creating golden datasets from filmed videos and validating results will be added here.*