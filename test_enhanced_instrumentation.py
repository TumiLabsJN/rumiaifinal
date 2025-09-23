#!/usr/bin/env python3
"""
Test enhanced instrumentation features from Phase 2
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the rumiai_v2 directory to Python path
sys.path.insert(0, '/home/jorge/rumiaifinal')

# Set sequential mode as default
os.environ['PARALLEL_MODE'] = 'false'  # Ensure sequential

async def test_enhanced_features():
    """Test the enhanced instrumentation features."""
    print("=" * 60)
    print("TESTING ENHANCED INSTRUMENTATION FEATURES")
    print("=" * 60)

    # Test video (short for quick testing)
    test_video_url = 'https://www.tiktok.com/@meganlock_/video/7428596413707144481'
    test_video_id = '7428596413707144481'

    print("\n1. Testing SystemSnapshot class...")
    from rumiai_v2.processors.video_analyzer import SystemSnapshot
    snapshot = SystemSnapshot()
    if snapshot.available:
        state = snapshot.capture()
        print(f"✅ SystemSnapshot working - CPU: {state['cpu']['percent']:.1f}%, "
              f"Memory: {state['memory']['percent']:.1f}%")
    else:
        print("⚠️ SystemSnapshot not available (psutil missing)")

    print("\n2. Testing Enhanced ThreadMonitor...")
    from rumiai_v2.processors.video_analyzer import ThreadMonitor
    with ThreadMonitor('test_service') as monitor:
        import time
        time.sleep(1)  # Simulate work
        stats = monitor.stop()

    print(f"✅ ThreadMonitor enhanced stats:")
    print(f"   - Threads created: {stats['threads_created']}")
    print(f"   - Peak threads: {stats['peak_threads']}")
    print(f"   - CPU time: {stats.get('cpu_time', 'N/A')}")
    print(f"   - Peak memory: {stats.get('peak_memory_mb', 'N/A')} MB")
    print(f"   - Samples collected: {stats.get('sample_count', 0)}")

    print("\n3. Testing Enhanced Report Generation...")
    print("Creating mock report to test new features...")
    from rumiai_v2.processors.video_analyzer import VideoAnalyzer
    from rumiai_v2.core.models import MLAnalysisResult
    import time

    # Create a minimal analyzer
    analyzer = VideoAnalyzer(None)  # ml_services not needed for report generation

    # Create mock results
    mock_results = {
        'yolo': MLAnalysisResult(
            model_name='yolo',
            model_version='v8',
            success=True,
            data={'mock': 'data'},
            processing_time=5.2,
            start_time=time.time() - 5.2,
            end_time=time.time(),
            threads_created=3,
            memory_delta_mb=250.5,
            thread_flexibility='✅ cv2.set()'
        ),
        'whisper': MLAnalysisResult(
            model_name='whisper',
            model_version='base',
            success=True,
            data={'mock': 'data'},
            processing_time=12.8,
            start_time=time.time() - 12.8,
            end_time=time.time(),
            threads_created=25,  # High thread count for warning
            memory_delta_mb=1200.3,  # High memory for warning
            thread_flexibility='✅ Direct'
        )
    }

    # Generate report
    analyzer._generate_timing_report(
        test_video_id,
        Path('test_video.mp4'),
        mock_results,
        time.time() - 18.0,
        time.time()
    )

    # Check if report was generated
    metrics_dir = Path("instrumentation_metrics")
    if metrics_dir.exists():
        reports = list(metrics_dir.glob(f"{test_video_id}_*_metrics.json"))
        if reports:
            latest_report = max(reports, key=lambda p: p.stat().st_mtime)
            with open(latest_report) as f:
                report = json.load(f)

            print(f"✅ Report generated: {latest_report.name}")

            # Check for new features
            if 'analysis' in report:
                print("✅ Sequential analysis present")
                analysis = report['analysis']
                if 'bottleneck_service' in analysis:
                    print(f"   - Bottleneck: {analysis['bottleneck_service']}")
                if 'total_memory_used' in analysis:
                    print(f"   - Memory used: {analysis['total_memory_used']:.1f} MB")

            if 'optimization_suggestions' in report:
                print(f"✅ Optimization suggestions: {len(report['optimization_suggestions'])} found")
                for suggestion in report['optimization_suggestions'][:2]:
                    print(f"   - {suggestion['service']}: {suggestion['suggestion']}")
        else:
            print("⚠️ No report generated")
    else:
        print("⚠️ Metrics directory not found")

    print("\n4. Testing Optimization Detection...")
    print("Check the logs above for real-time warnings like:")
    print("   - ⚠️ Memory pressure warnings")
    print("   - ⚠️ Thread creation warnings")
    print("   - 🧹 Garbage collection notices")

    print("\n" + "=" * 60)
    print("ENHANCED INSTRUMENTATION TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_enhanced_features())