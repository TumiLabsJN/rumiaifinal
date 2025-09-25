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

### 2. Minimal Mock Data Strategy
- Use **zero or minimal values** for all engagement metrics
- Provide **only required fields** to prevent crashes
- Keep **test simple and fast** without complex data generation
- Accept that **some features will produce zero values**

### 3. No Production Code Changes
```python
# Import production runner AS-IS
from scripts.rumiai_runner import RumiAIRunner
from rumiai_v2.core.models.video import VideoMetadata
from pathlib import Path
from datetime import datetime

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

### 4. Comparison with Existing Tests

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
        Create minimal VideoMetadata with only required fields to avoid crashes.
        Uses zero/minimal values to keep the test simple and focused on ML pipeline.

        Decision: Use minimal mock data (Option A) for simplicity and speed.
        Some features dependent on engagement may produce zero values.
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

        # Return VideoMetadata with all required fields (production expects these)
        return VideoMetadata(
            video_id=video_id,
            url=video_url,
            username="testuser",  # Required: changed from 'author'
            description="Test video",
            duration=duration,
            views=0,  # Required: changed from 'play_count'
            likes=0,  # Required: changed from 'digg_count'
            comments=0,  # Required: changed from 'comment_count'
            shares=0,  # Required: changed from 'share_count'
            saves=0,  # Required: changed from 'collect_count'
            create_time=datetime.now(),  # Required: datetime object, not timestamp
            download_url=video_url,  # Required field
            cover_url="",  # Required field
            hashtags=[],  # Empty list is valid
            music={},  # Required: empty dict is valid
            author={},  # Required: empty dict is valid
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
        """Override Apify download by returning local file."""
        video_path = self.video_mapping.get(video_metadata.url)
        if not video_path or not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Copy to temp dir to mimic production behavior
        temp_path = self.settings.temp_dir / f"{video_metadata.video_id}.mp4"
        shutil.copy2(video_path, temp_path)
        return temp_path

    def create_mock_metadata(self, video_url: str, video_path: Path) -> VideoMetadata:
        """Create minimal VideoMetadata matching production structure.
        See full implementation above with zero/minimal values."""
        # Implementation shown in Phase 1 above
        # Cannot call super() as parent class doesn't have this method
        # Would implement the full method here (see Phase 1)

# Usage: Run exact production pipeline with local videos
runner = TestManualVideosRunner()
await runner.process_video_url("https://tiktok.com/@test/my_video")  # Uses local my_video.mp4
```

### Phase 3: Output Validation

```python
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

### Phase 4: Batch Processing

```python
async def main():
    """
    Process all videos in temp directory using production pipeline.
    """
    runner = TestManualVideosRunner(
        video_dir=Path("/home/jorge/rumiaifinal/temp")
    )

    # Get all mapped URLs
    test_urls = list(runner.video_mapping.keys())
    print(f"Found {len(test_urls)} videos to process")

    results = []
    for url in test_urls:
        video_name = runner.video_mapping[url].name
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print(f"URL: {url}")
        print('='*60)

        try:
            # Run EXACT production pipeline
            # This will call all production code except the 2 overridden methods
            await runner.process_video_url(url)

            # Extract video_id from URL for validation
            video_id = url.split('/')[-1]

            # Validate outputs using production paths
            success = validate_output(f"manual_{video_id}")

            results.append({
                'video': video_name,
                'video_id': f"manual_{video_id}",
                'success': success,
                'url': url
            })

            print(f"✅ Completed: {video_name}")

        except Exception as e:
            print(f"❌ Failed: {video_name}")
            print(f"   Error: {str(e)}")
            logger.exception(f"Failed processing {url}")

            results.append({
                'video': video_name,
                'success': False,
                'error': str(e),
                'url': url
            })

    # Print summary report
    print_summary(results)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
```

## Key Differences from Production

### 1. Metadata Handling
- **Production**: Gets rich metadata from TikTok (hashtags, engagement, creator info)
- **Test**: Uses minimal mock metadata with zero/default values
- **Impact**: Features dependent on hashtags/engagement will produce zero or minimal values

### 2. Video Source
- **Production**: Downloads from TikTok CDN after Apify provides URL
- **Test**: Reads from local `/temp` directory
- **Impact**: No network dependencies, faster testing

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
# Basic usage with minimal mock data
python test_manual_videos.py

# With specific directory
python test_manual_videos.py --video-dir /custom/path

# With parallel processing
python test_manual_videos.py --parallel

# Validate only (don't process)
python test_manual_videos.py --validate-only
```

## Advantages Over Current Tests

1. **End-to-End Testing**: Tests complete pipeline, not just one component
2. **Custom Videos**: Can test edge cases with specific video content
3. **Offline Testing**: No dependency on network or APIs
4. **Debugging**: Easier to debug with local files
5. **Regression Testing**: Can maintain suite of test videos

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