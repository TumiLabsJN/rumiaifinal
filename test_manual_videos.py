#!/usr/bin/env python3
"""
Test Manual Videos - Implementation
Processes manually downloaded TikTok videos through the RumiAI pipeline,
mirroring production behavior while bypassing the Apify scraping layer.
"""

import asyncio
import json
import logging
import sys
import traceback
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

# Add project to path for imports
sys.path.insert(0, '/home/jorge/rumiaifinal')

from scripts.rumiai_runner import RumiAIRunner
from rumiai_v2.core.models.video import VideoMetadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestManualVideosRunner(RumiAIRunner):
    """Inherits full production pipeline, overrides only Apify methods."""

    def __init__(self, video_dir: Path = Path("/home/jorge/rumiaifinal/temp")):
        """Initialize with video directory and build mapping."""
        super().__init__()
        self.video_dir = video_dir
        self.video_mapping = self._build_video_mapping()
        logger.info(f"Initialized with {len(self.video_mapping)} videos from {video_dir}")

    def _build_video_mapping(self) -> Dict[str, Path]:
        """
        Map fake TikTok URLs to local video files.
        This allows us to call runner.process_video_url() with URLs just like production.
        """
        mapping = {}

        if not self.video_dir.exists():
            logger.warning(f"Video directory does not exist: {self.video_dir}")
            return mapping

        video_files = list(self.video_dir.glob("*.mp4"))

        for video_path in video_files:
            # Create fake but valid-looking TikTok URL
            fake_url = f"https://www.tiktok.com/@testuser/video/{video_path.stem}"
            mapping[fake_url] = video_path
            logger.debug(f"Mapped {video_path.name} -> {fake_url}")

        logger.info(f"Mapped {len(mapping)} local videos to fake URLs")
        return mapping

    async def _scrape_video(self, video_url: str) -> VideoMetadata:
        """Override Apify scraping with mock metadata."""
        video_path = self.video_mapping.get(video_url)
        if not video_path:
            raise ValueError(f"No local video for URL: {video_url}")

        logger.info(f"Creating mock metadata for {video_path.name}")
        return self.create_mock_metadata(video_url, video_path)

    async def _download_video(self, video_metadata: VideoMetadata) -> Path:
        """
        Override Apify download by returning local file.

        Decision Point 6: Direct return (no copy) since videos are already in temp dir.
        Videos are in /home/jorge/rumiaifinal/temp/ which is where production expects them.
        """
        video_path = self.video_mapping.get(video_metadata.url)
        if not video_path or not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Returning local video path: {video_path}")
        # Videos are already in temp directory, just return the path
        return video_path

    def create_mock_metadata(self, video_url: str, video_path: Path) -> VideoMetadata:
        """
        Create VideoMetadata with ALL required fields to ensure production compatibility.

        Decision Point 2: Use minimal mock data values for simplicity.
        Decision Point 5: Provide ALL VideoMetadata fields (even with zero values)
        because production code may access any field and we want to avoid AttributeErrors.

        This combines minimal values with complete field coverage.
        """
        # Extract actual video properties
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = int(frame_count / fps) if fps > 0 else 0
            cap.release()
            logger.debug(f"Video duration: {duration}s (fps={fps}, frames={frame_count})")
        except Exception as e:
            logger.warning(f"Could not extract video duration: {e}")
            duration = 30  # Default fallback

        # Generate simple video_id from filename
        video_id = video_path.stem.replace('_', '')
        if not video_id.isdigit():
            video_id = str(abs(hash(video_id)) % 10**15)  # Fake TikTok-like ID

        logger.debug(f"Generated video_id: {video_id}")

        # Return VideoMetadata with ALL required fields to avoid AttributeErrors
        # Decision Point 5: Must provide every field that VideoMetadata class expects
        metadata = VideoMetadata(
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

        logger.info(f"Created mock metadata for video_id={video_id}, duration={duration}s")
        return metadata


def validate_output(video_id: str) -> bool:
    """
    Verify the pipeline produced expected outputs.
    """
    expected_files = [
        f"unified_analysis/{video_id}.json",
        f"insights/{video_id}_temporal_windows_updated.json"
    ]

    all_valid = True

    for file_path_str in expected_files:
        file_path = Path(file_path_str)
        if not file_path.exists():
            logger.error(f"❌ Missing output: {file_path}")
            all_valid = False
            continue

        # Validate JSON structure
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Check for required fields based on file type
            if 'temporal_windows_updated' in file_path_str:
                if 'temporal_windows' not in data:
                    logger.error(f"❌ Missing 'temporal_windows' in {file_path}")
                    all_valid = False
                else:
                    # Check for the three required temporal windows
                    windows = data['temporal_windows']
                    required_windows = ['hook', 'middle_segments', 'closing']
                    for window in required_windows:
                        if window not in windows:
                            logger.error(f"❌ Missing '{window}' in temporal_windows")
                            all_valid = False
                    logger.info(f"✅ Valid structure in {file_path}")

            elif 'unified_analysis' in file_path_str:
                # Check for timeline and ml_data
                if 'timeline' not in data and 'ml_data' not in data:
                    logger.error(f"❌ Missing timeline/ml_data in {file_path}")
                    all_valid = False
                else:
                    logger.info(f"✅ Valid structure in {file_path}")

        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {file_path}: {e}")
            all_valid = False
        except Exception as e:
            logger.error(f"❌ Error validating {file_path}: {e}")
            all_valid = False

    return all_valid


async def test_single_video(video_filename: str) -> bool:
    """
    Test a single video through the production pipeline.
    Tests are run one at a time for better debugging and isolation.
    """
    runner = TestManualVideosRunner(
        video_dir=Path("/home/jorge/rumiaifinal/temp")
    )

    # Create fake URL for this specific video
    video_stem = Path(video_filename).stem
    fake_url = f"https://www.tiktok.com/@testuser/video/{video_stem}"

    if fake_url not in runner.video_mapping:
        print(f"❌ Video not found: {video_filename}")
        print(f"Available videos in {runner.video_dir}:")
        for url, path in runner.video_mapping.items():
            print(f"  - {path.name}")
        return False

    print(f"\n{'='*60}")
    print(f"Testing: {video_filename}")
    print(f"URL: {fake_url}")
    print('='*60)

    try:
        # Run EXACT production pipeline
        print("\n🚀 Starting production pipeline...")
        result = await runner.process_video_url(fake_url)

        # Extract video_id and validate outputs
        # The mock metadata prepends 'manual_' to non-numeric IDs
        video_id = video_stem.replace('_', '')
        if not video_id.isdigit():
            video_id = str(abs(hash(video_id)) % 10**15)

        print(f"\n🔍 Validating outputs for video_id={video_id}...")
        success = validate_output(video_id)

        if success:
            print(f"\n✅ Test passed: {video_filename}")
            print(f"   All expected outputs generated and valid")
        else:
            print(f"\n⚠️ Test completed but validation failed: {video_filename}")
            print(f"   Check logs above for specific validation errors")

        return success

    except Exception as e:
        print(f"\n❌ Test failed: {video_filename}")
        print(f"   Error: {str(e)}")
        logger.exception("Full error traceback:")
        traceback.print_exc()
        return False


# Entry point for single video testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_manual_videos.py <video_filename>")
        print("Example: python test_manual_videos.py example_video.mp4")
        print("\nVideos must be in /home/jorge/rumiaifinal/temp/")
        sys.exit(1)

    video_file = sys.argv[1]

    # Setup event loop and run test
    try:
        success = asyncio.run(test_single_video(video_file))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)