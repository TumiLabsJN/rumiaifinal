#!/usr/bin/env python3
"""
Test DeepFace gender detection on a single video file.

Usage:
    python scripts/test_deepface_video.py <video_path>
"""

import sys
import asyncio
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rumiai_v2.ml_services.deepface_gender_service import (
    DeepFaceGenderService,
    DeepFaceConfig,
    VideoLoadError
)


async def test_video(video_path: str):
    """Test DeepFace gender detection on a video."""

    print(f"Testing DeepFace gender detection on: {video_path}")
    print("-" * 50)

    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return

    try:
        # Create service with custom config for testing
        config = DeepFaceConfig(
            timeout=30,  # 30 seconds timeout
            detector_backend='opencv',  # Fast backend
            enforce_detection=False,  # Don't fail if no faces
            use_gpu=False,  # Use CPU
            thread_workers=2
        )

        print(f"Configuration:")
        print(f"  Timeout: {config.timeout}s")
        print(f"  Detector: {config.detector_backend}")
        print(f"  GPU: {config.use_gpu}")
        print()

        # Initialize service
        print("Initializing DeepFace service...")
        service = DeepFaceGenderService(config=config)
        print("✓ Service initialized")
        print()

        # Run analysis
        print("Running gender detection...")
        result = await service.analyze(video_path)

        # Display results
        print("\nResults:")
        print("-" * 50)

        if result.get('gender'):
            print(f"Gender: {result['gender']}")
            print(f"Confidence: {result['confidence']:.2%}")

            if result['gender'] == 'multiple_people':
                print(f"Multiple people detected in {result.get('multi_person_frames', 'N/A')} frames")
                print("→ Will use self-normalization for pitch")
        else:
            print("No gender detected")
            if 'error' in result:
                print(f"Reason: {result['error']}")

        # Display metadata
        print(f"\nProcessing details:")
        print(f"  Processing time: {result.get('processing_ms', 0)/1000:.2f}s")
        print(f"  Frames analyzed: {result.get('frames_analyzed', 'N/A')}")
        print(f"  Detector used: {result.get('detector_backend', 'N/A')}")
        print(f"  Method: {result.get('method', 'N/A')}")

        # Save result to file
        output_path = Path(f"gender_detection_outputs/test_{Path(video_path).stem}_gender.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nResult saved to: {output_path}")

    except VideoLoadError as e:
        print(f"Error loading video: {e}")
        return 1
    except asyncio.TimeoutError:
        print(f"Analysis timed out after {config.timeout} seconds")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_deepface_video.py <video_path>")
        print("\nExample:")
        print("  python scripts/test_deepface_video.py sample_videos/test.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    # Run async function
    exit_code = asyncio.run(test_video(video_path))
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()