# Test Manual Videos - Design Document

## Purpose
Create a test script that processes manually downloaded TikTok videos through the RumiAI pipeline, mirroring production behavior while bypassing the Apify scraping layer.

## Goal
Test the core ML processing pipeline (ML Services → Timeline Builder → Temporal Compute → Output) using local video files, without requiring network access or API dependencies.

## Scope

### What This Test Covers ✅
- **ML SERVICES LAYER**: All 8 ML services (YOLO, Whisper, MediaPipe, OCR, Scene Detection, Audio Energy, FEAT Emotion, DeepFace Gender)
- **TIMELINE BUILDER**: Merging ML outputs into unified timeline
- **TEMPORAL COMPUTE**: Extracting 60+ features per temporal window
- **OUTPUT LAYER**: Generating final JSON in `/insights/` directory

### What This Test Skips ❌
- **INPUT LAYER**: Apify scraping (uses local files instead)
- **Metadata Collection**: TikTok engagement metrics (uses mock data)
- **Video Download**: Files already in `/home/jorge/rumiaifinal/temp/`

## Design Principles

### 1. True Mirror Philosophy - Monkey-Patch Approach
- Use **actual production pipeline** by inheriting from RumiAIRunner
- Override only the Apify-specific methods (2 methods total)
- Maintain identical data flow and processing logic
- Preserve all production behaviors (logging, error handling, checkpointing)

### 2. Minimal Mock Data Strategy with Complete Fields
- Use **zero or minimal values** for all engagement metrics (Decision Point 2)
- Provide **ALL required VideoMetadata fields** to prevent AttributeErrors (Decision Point 5)
- Keep **values simple** (zeros/empty) but **structure complete**
- Accept that **some features will produce zero values** due to minimal data

### 3. Direct Video Path Return
- Videos already in `/home/jorge/rumiaifinal/temp/` directory
- No copying needed - return paths directly (Decision Point 6)
- Avoids unnecessary I/O and duplication
- Production expects videos in temp dir, which is where they already are

### 4. No Production Code Changes
```python
# Import production runner AS-IS
import asyncio
import json
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

from scripts.rumiai_runner import RumiAIRunner
from rumiai_v2.core.models.video import VideoMetadata

logger = logging.getLogger(__name__)

# Create test subclass that overrides only Apify methods
class TestManualVideosRunner(RumiAIRunner):
    async def _scrape_video(self, video_url: str) -> VideoMetadata:
        """Override to return mock metadata instead of Apify scraping."""
        video_path = self.video_mapping.get(video_url)
        if not video_path:
            raise ValueError(f"No local video for URL: {video_url}")
        return self.create_mock_metadata(video_url, video_path)

    async def _download_video(self, video_metadata: VideoMetadata) -> Path:
        """Override to return local file path instead of downloading."""
        video_path = self.video_mapping.get(video_metadata.url)
        if not video_path or not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return video_path
```

**Rationale**: Analysis showed rumiai_runner.py has excellent Apify separation with only 2 clean interface points to override.

### 5. Comparison with Existing Tests

| Aspect | rumiai_runner.py | test_temporal_compute_v2.py | test_manual_videos.py |
|--------|------------------|------------------------------|----------------------|
| **Input** | TikTok URL | Saved JSON | Fake URL → Local MP4 |
| **Apify Scraping** | ✅ Yes | ❌ No | ❌ No (mocked) |
| **ML Services** | ✅ Runs all | ❌ Uses old data | ✅ Runs all (via production) |
| **Timeline Builder** | ✅ Yes | ❌ Uses old timeline | ✅ Yes (via production) |
| **Temporal Compute** | ✅ Yes | ✅ Yes | ✅ Yes (via production) |
| **Pipeline Code** | Full pipeline | Component only | Full pipeline (inherited) |
| **Purpose** | Production | Unit test | Integration test |

## Implementation Strategy

### Phase 1: Video Discovery and URL Mapping

```python
class TestManualVideosRunner(RumiAIRunner):
    def _build_video_mapping(self) -> Dict[str, Path]:
        """
        Map fake TikTok URLs to local video files.
        This allows us to call runner.run() with URLs just like production.
        """
        mapping = {}
        video_files = list(self.video_dir.glob("*.mp4"))

        for video_path in video_files:
            # Create fake but valid-looking TikTok URL
            fake_url = f"https://www.tiktok.com/@testuser/video/{video_path.stem}"
            mapping[fake_url] = video_path

        logger.info(f"Mapped {len(mapping)} local videos to fake URLs")
        return mapping

    def create_mock_metadata(self, video_url: str, video_path: Path) -> VideoMetadata:
        """
        Create VideoMetadata with ALL required fields to ensure production compatibility.

        Decision Point 2: Use minimal mock data values for simplicity.
        Decision Point 5: Provide ALL VideoMetadata fields (even with zero values)
        because production code may access any field and we want to avoid AttributeErrors.

        This combines minimal values with complete field coverage.
        """
        # Extract actual video properties
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(frame_count / fps) if fps > 0 else 0
        cap.release()

        # Generate simple video_id from filename
        video_id = video_path.stem.replace('_', '')
        if not video_id.isdigit():
            video_id = str(abs(hash(video_id)) % 10**15)  # Fake TikTok-like ID

        # Return VideoMetadata with ALL required fields to avoid AttributeErrors
        # Decision Point 5: Must provide every field that VideoMetadata class expects
        return VideoMetadata(
            video_id=video_id,
            url=video_url,
            username="testuser",  # Required field (not 'author' string)
            description="Test video",  # Minimal but valid
            duration=duration,  # Actual duration from video
            views=0,  # Minimal value
            likes=0,  # Minimal value
            comments=0,  # Minimal value
            shares=0,  # Minimal value
            saves=0,  # Minimal value
            create_time=datetime.now(),  # Must be datetime object
            download_url=video_url,  # Required even if not downloading
            cover_url="",  # Required but can be empty
            hashtags=[],  # Required but can be empty list
            music={},  # Required but can be empty dict
            author={},  # Required but can be empty dict
            engagement_rate=0.0  # Required field
        )
```

### Phase 2: ML Processing Pipeline

```python
class TestManualVideosRunner(RumiAIRunner):
    """Inherits full production pipeline, overrides only Apify methods."""

    def __init__(self, video_dir: Path = Path("/home/jorge/rumiaifinal/temp")):
        super().__init__()
        self.video_dir = video_dir
        self.video_mapping = self._build_video_mapping()

    def _build_video_mapping(self) -> Dict[str, Path]:
        """Map fake URLs to local video files."""
        mapping = {}
        for video_path in self.video_dir.glob("*.mp4"):
            fake_url = f"https://tiktok.com/@test/{video_path.stem}"
            mapping[fake_url] = video_path
        return mapping

    async def _scrape_video(self, video_url: str) -> VideoMetadata:
        """Override Apify scraping with mock metadata."""
        video_path = self.video_mapping.get(video_url)
        if not video_path:
            raise ValueError(f"No local video for URL: {video_url}")

        return self.create_mock_metadata(video_url, video_path)

    async def _download_video(self, video_metadata: VideoMetadata) -> Path:
        """Override Apify download by returning local file.

        Decision Point 6: Direct return (no copy) since videos are already in temp dir.
        Videos are in /home/jorge/rumiaifinal/temp/ which is where production expects them.
        """
        video_path = self.video_mapping.get(video_metadata.url)
        if not video_path or not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Videos are already in temp directory, just return the path
        return video_path

    def create_mock_metadata(self, video_url: str, video_path: Path) -> VideoMetadata:
        """Create minimal VideoMetadata matching production structure.
        See full implementation above with zero/minimal values."""
        # Implementation must match Phase 1 exactly
        # Extract video duration
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(frame_count / fps) if fps > 0 else 0
        cap.release()

        # Generate video_id
        video_id = video_path.stem.replace('_', '')
        if not video_id.isdigit():
            video_id = str(abs(hash(video_id)) % 10**15)

        # Return complete VideoMetadata
        return VideoMetadata(
            video_id=video_id,
            url=video_url,
            username="testuser",
            description="Test video",
            duration=duration,
            views=0,
            likes=0,
            comments=0,
            shares=0,
            saves=0,
            create_time=datetime.now(),
            download_url=video_url,
            cover_url="",
            hashtags=[],
            music={},
            author={},
            engagement_rate=0.0
        )

# Usage: Run exact production pipeline with local videos
runner = TestManualVideosRunner()
await runner.process_video_url("https://tiktok.com/@test/my_video")  # Uses local my_video.mp4
```

### Phase 3: Output Validation

```python
import json
from pathlib import Path

def validate_output(video_id: str) -> bool:
    """
    Verify the pipeline produced expected outputs.
    """
    expected_files = [
        f"unified_analysis/{video_id}.json",
        f"insights/{video_id}_temporal_windows_updated.json"
    ]

    for file_path in expected_files:
        if not Path(file_path).exists():
            print(f"❌ Missing output: {file_path}")
            return False

        # Validate JSON structure
        with open(file_path) as f:
            data = json.load(f)

        # Check for required fields
        if 'temporal_windows' not in data and 'temporal' in file_path:
            print(f"❌ Invalid structure in {file_path}")
            return False

    return True
```

### Phase 4: Single Video Testing

```python
import asyncio
from pathlib import Path

async def test_single_video(video_filename: str):
    """
    Test a single video through the production pipeline.
    Tests are run one at a time for better debugging and isolation.
    """
    runner = TestManualVideosRunner(
        video_dir=Path("/home/jorge/rumiaifinal/temp")
    )

    # Create fake URL for this specific video
    fake_url = f"https://tiktok.com/@test/{Path(video_filename).stem}"

    if fake_url not in runner.video_mapping:
        print(f"❌ Video not found: {video_filename}")
        print(f"Available videos: {list(runner.video_mapping.keys())}")
        return False

    print(f"\n{'='*60}")
    print(f"Testing: {video_filename}")
    print(f"URL: {fake_url}")
    print('='*60)

    try:
        # Run EXACT production pipeline
        await runner.process_video_url(fake_url)

        # Extract video_id and validate outputs
        video_id = Path(video_filename).stem
        success = validate_output(f"manual_{video_id}")

        if success:
            print(f"✅ Test passed: {video_filename}")
        else:
            print(f"⚠️ Test completed but validation failed: {video_filename}")

        return success

    except Exception as e:
        print(f"❌ Test failed: {video_filename}")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Entry point for single video testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_manual_videos.py <video_filename>")
        print("Example: python test_manual_videos.py example_video.mp4")
        sys.exit(1)

    video_file = sys.argv[1]
    success = asyncio.run(test_single_video(video_file))
    sys.exit(0 if success else 1)
```

## Key Differences from Production

### 1. Metadata Handling
- **Production**: Gets rich metadata from TikTok (hashtags, engagement, creator info)
- **Test**: Uses minimal mock metadata with zero/default values
- **Impact**: Features dependent on hashtags/engagement will produce zero or minimal values

### 2. Video Source
- **Production**: Downloads from TikTok CDN to temp directory
- **Test**: Uses pre-existing files in `/home/jorge/rumiaifinal/temp/`
- **Impact**: No network dependencies, no file copying, faster testing

### 3. Error Handling
- **Production**: Handles Apify failures, network issues
- **Test**: Simplified error handling for local file issues
- **Impact**: Cleaner test flow, fewer edge cases

## Expected Outputs

For each video `example_video.mp4`, the pipeline will create:

```
unified_analysis/
└── manual_example_video.json          # ML service outputs + timeline

insights/
└── manual_example_video_temporal_windows_updated.json  # Final features

test_outputs/
└── manual_videos_summary.json         # Test run summary
```

## Success Criteria

1. **Functional Success**
   - All 8 ML services run successfully
   - Timeline builds without errors
   - Temporal compute produces all 60+ features per window
   - Output JSONs have correct structure
   - Pipeline completes without crashes (some features may be zero due to minimal mock data)

2. **Performance Baseline**
   - 60-second video processes in < 3 minutes
   - Memory usage stays under 4GB
   - No service crashes or timeouts

3. **Data Quality**
   - Features have reasonable values (no NaN/null)
   - Temporal windows align with video duration
   - All expected fields present in output

## Usage

```bash
# Test a single video
python test_manual_videos.py example_video.mp4

# Test another video
python test_manual_videos.py my_test_video.mp4

# Videos must be in /home/jorge/rumiaifinal/temp/
```

## Advantages Over Current Tests

1. **End-to-End Testing**: Tests complete pipeline, not just one component
2. **Custom Videos**: Can test edge cases with specific video content
3. **Offline Testing**: No dependency on network or APIs
4. **Debugging**: One-at-a-time testing makes debugging easier
5. **Isolation**: Each test runs independently, no interference

## Limitations

1. **Minimal Metadata**: Can't properly test hashtag-dependent or engagement-based features
2. **No Apify Testing**: Doesn't validate scraping layer
3. **Static Test Set**: Limited to pre-downloaded videos
4. **Zero Engagement**: Virality and engagement ratio features will be meaningless

## Future Enhancements

1. **Metadata Injection**: Allow JSON sidecar files with real TikTok metadata
2. **Golden Dataset**: Compare outputs against known-good results
3. **Performance Tracking**: Track processing time trends
4. **Parallel Processing**: Process multiple videos simultaneously
5. **Selective Service Testing**: Run subset of ML services for speed