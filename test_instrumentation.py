#!/usr/bin/env python3
"""Test script for instrumented ML pipeline."""

import os
import sys
import asyncio
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/jorge/rumiaifinal')

async def download_video(url: str, video_id: str) -> Path:
    """Download video using yt-dlp."""
    output_path = Path(f'/home/jorge/rumiaifinal/temp/{video_id}.mp4')

    if output_path.exists():
        print(f"  Video already downloaded: {output_path}")
        return output_path

    print(f"  Downloading {video_id}...")
    cmd = [
        'yt-dlp',
        '-f', 'best[ext=mp4]/mp4',
        '-o', str(output_path),
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Download failed: {result.stderr}")
        return None

    print(f"  Downloaded to: {output_path}")
    return output_path

async def test_instrumentation():
    """Run instrumented pipeline test."""

    # Standard test videos from PHASE_1_SERVICE_STRUCTURE.md
    test_videos = [
        {
            'url': 'https://www.tiktok.com/@meganlock_/video/7428596413707144481',
            'id': '7428596413707144481',
            'name': '18s_video',
            'duration': '18s'
        },
        {
            'url': 'https://www.tiktok.com/@janemukbangs/video/7500252920844193067',
            'id': '7500252920844193067',
            'name': '73s_video',
            'duration': '73s'
        },
        {
            'url': 'https://www.tiktok.com/@jakedoesmusicsometimes/video/6923023955813092613',
            'id': '6923023955813092613',
            'name': '120s_video',
            'duration': '120s'
        }
    ]

    # Test both sequential and parallel modes
    test_modes = [
        ('parallel', {'SEQUENTIAL_TEST': 'false', 'METRICS_DISPLAY': 'summary'}),
        ('sequential', {'SEQUENTIAL_TEST': 'true', 'METRICS_DISPLAY': 'summary'})
    ]

    for video_info in test_videos:
        print(f"\n{'='*80}")
        print(f"Testing with: {video_info['name']} ({video_info['duration']})")
        print(f"URL: {video_info['url']}")
        print(f"{'='*80}")

        # Download video if needed
        video_path = await download_video(video_info['url'], video_info['id'])
        if not video_path or not video_path.exists():
            print(f"Failed to get video: {video_info['id']}")
            continue

        for mode_name, env_vars in test_modes:
            print(f"\n{'-'*60}")
            print(f"Testing in {mode_name.upper()} mode")
            print(f"Environment: {env_vars}")
            print(f"{'-'*60}")

            # Set environment variables
            for key, value in env_vars.items():
                os.environ[key] = value

            try:
                # Import after setting env vars
                from rumiai_v2.processors.video_analyzer import VideoAnalyzer
                from rumiai_v2.core.models.analysis import UnifiedAnalysis
                from rumiai_v2.core.models.timeline import Timeline
                from rumiai_v2.api.ml_services import MLServices

                # Initialize services and analyzer
                ml_services = MLServices()
                analyzer = VideoAnalyzer(ml_services)

                # Create video metadata
                video_id = video_path.stem
                video_metadata = {
                    'filename': video_path.name,
                    'path': str(video_path),
                    'size_mb': video_path.stat().st_size / (1024*1024)
                }

                # Initialize unified analysis
                timeline = Timeline(video_id=video_id, duration=0)
                unified_analysis = UnifiedAnalysis(
                    video_id=video_id,
                    video_metadata=video_metadata,
                    timeline=timeline
                )

                # Run analysis
                print(f"\nStarting {mode_name} analysis...")
                start_time = time.time()

                await analyzer.analyze_video(
                    video_id=video_id,
                    video_path=video_path,
                    unified_analysis=unified_analysis
                )

                total_time = time.time() - start_time
                print(f"\nTotal execution time: {total_time:.2f}s")

                # Check results
                print("\nAnalysis Results:")
                print(f"  Completed: {unified_analysis.is_complete()}")
                print(f"  Services run: {list(unified_analysis.ml_results.keys())}")

                # Show any errors
                errors = unified_analysis.get_errors()
                if errors:
                    print("\nErrors encountered:")
                    for service, error in errors.items():
                        print(f"  {service}: {error}")

            except Exception as e:
                print(f"\nError during {mode_name} test: {e}")
                import traceback
                traceback.print_exc()

            # Small delay between tests
            await asyncio.sleep(2)

    print(f"\n{'='*80}")
    print("Instrumentation test complete!")
    print(f"Check instrumentation_metrics/ directory for detailed reports")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(test_instrumentation())