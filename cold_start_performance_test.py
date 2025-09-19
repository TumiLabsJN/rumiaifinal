#!/usr/bin/env python3
"""
Cold Start Performance Test for RumiAI Services
Tests each video with complete cache clearing and process restart
"""

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

class ColdStartTester:
    def __init__(self):
        self.test_videos = [
            {
                'name': '18s_video',
                'url': 'https://www.tiktok.com/@meganlock_/video/7428596413707144481',
                'expected_duration': 18
            },
            {
                'name': '73s_video',
                'url': 'https://www.tiktok.com/@janemukbangs/video/7500252920844193067',
                'expected_duration': 73
            },
            {
                'name': '120s_video',
                'url': 'https://www.tiktok.com/@jakedoesmusicsometimes/video/6923023955813092613',
                'expected_duration': 120
            }
        ]

        self.results = []
        self.results_file = Path('/home/jorge/rumiaifinal/cold_start_results.json')

    def clear_all_caches(self):
        """Remove ALL caches and temporary files"""
        print("\n🧹 Clearing all caches...")

        # 1. Clear frame caches in /tmp
        tmp_patterns = [
            '/tmp/rumiai_frames_*',
            '/tmp/tmp*frame*.jpg',
            '/tmp/vision_test_*',
            '/tmp/rumiai_run.log'
        ]

        for pattern in tmp_patterns:
            subprocess.run(f'rm -rf {pattern}', shell=True, capture_output=True)

        # 2. Clear video downloads
        temp_dir = Path('/home/jorge/rumiaifinal/temp')
        if temp_dir.exists():
            print(f"  ✓ Clearing {len(list(temp_dir.glob('*.mp4')))} video files")
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)

        # 3. Clear Python cache
        cache_dirs = [
            '/home/jorge/rumiaifinal/rumiai_v2/__pycache__',
            '/home/jorge/rumiaifinal/rumiai_v2/**/__pycache__'
        ]

        for pattern in cache_dirs:
            subprocess.run(f'rm -rf {pattern}', shell=True, capture_output=True)

        # 4. Clear insights output (keep for reference but clear test outputs)
        test_insights = Path('/home/jorge/rumiaifinal/insights')
        for test_dir in test_insights.glob('test_*'):
            shutil.rmtree(test_dir)

        # 5. Kill any running Python processes (except this one)
        current_pid = os.getpid()
        result = subprocess.run(
            "ps aux | grep python | grep -v grep | awk '{print $2}'",
            shell=True, capture_output=True, text=True
        )

        for pid in result.stdout.strip().split('\n'):
            if pid and int(pid) != current_pid:
                try:
                    subprocess.run(f'kill -9 {pid}', shell=True, capture_output=True)
                except:
                    pass

        print("  ✓ All caches cleared")

    def run_single_test(self, video_config):
        """Run a single cold-start test"""
        print(f"\n{'='*60}")
        print(f"Testing: {video_config['name']}")
        print(f"URL: {video_config['url']}")
        print(f"Expected duration: {video_config['expected_duration']}s")
        print('='*60)

        # Clear all caches before test
        self.clear_all_caches()

        # Give system a moment to settle
        time.sleep(2)

        # Record start time
        start_time = time.time()

        # Run the actual production command
        print("\n📊 Running production command...")
        print(f"python3 scripts/rumiai_runner.py '{video_config['url']}'")

        result = subprocess.run(
            ['python3', 'scripts/rumiai_runner.py', video_config['url']],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for 120s videos
        )

        # Record end time
        end_time = time.time()
        total_time = end_time - start_time

        # Parse results
        test_result = {
            'video': video_config['name'],
            'url': video_config['url'],
            'expected_duration': video_config['expected_duration'],
            'total_processing_time': total_time,
            'success': result.returncode == 0,
            'timestamp': datetime.now().isoformat()
        }

        if result.returncode == 0:
            print(f"✅ Success! Total time: {total_time:.2f}s")

            # Try to find and parse the output
            insights_dir = Path('/home/jorge/rumiaifinal/insights')
            latest_dir = max(insights_dir.glob('*'), key=os.path.getctime)

            # Check for temporal windows output
            temporal_file = latest_dir / 'temporal_windows_updated.json'
            if temporal_file.exists():
                with open(temporal_file, 'r') as f:
                    temporal_data = json.load(f)
                    test_result['actual_duration'] = temporal_data.get('duration', 0)

            # Check for ML service outputs
            for service_file in ['yolo_detections.json', 'mediapipe_analysis.json',
                                'ocr_results.json', 'scene_detection.json']:
                service_path = latest_dir / service_file
                if service_path.exists():
                    service_name = service_file.split('_')[0]
                    test_result[f'{service_name}_completed'] = True

        else:
            print(f"❌ Failed with return code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            test_result['error'] = result.stderr[:1000] if result.stderr else 'Unknown error'

        # Store result
        self.results.append(test_result)

        # Save intermediate results
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        return test_result

    def run_all_tests(self):
        """Run all cold-start tests"""
        print("\n" + "="*60)
        print("COLD START PERFORMANCE TESTING")
        print("="*60)
        print("\nThis will test each video with complete cache clearing.")
        print("Each test runs the production command:")
        print("  python3 scripts/rumiai_runner.py 'VIDEO_URL'")

        for video_config in self.test_videos:
            try:
                self.run_single_test(video_config)
            except subprocess.TimeoutExpired:
                print(f"⏱️ Test timed out after 600 seconds")
                self.results.append({
                    'video': video_config['name'],
                    'error': 'Timeout after 600 seconds',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                self.results.append({
                    'video': video_config['name'],
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        print(f"\n{'Video':<15} {'Expected':<10} {'Actual':<10} {'Time(s)':<10} {'Status':<10}")
        print("-"*60)

        for result in self.results:
            video = result['video']
            expected = f"{result.get('expected_duration', '?')}s"
            actual = f"{result.get('actual_duration', '?'):.1f}s" if 'actual_duration' in result else 'N/A'
            time_taken = f"{result.get('total_processing_time', 0):.2f}" if 'total_processing_time' in result else 'N/A'
            status = '✅' if result.get('success') else '❌'

            print(f"{video:<15} {expected:<10} {actual:<10} {time_taken:<10} {status:<10}")

        print(f"\n✓ Results saved to: {self.results_file}")

        # Calculate averages
        successful_tests = [r for r in self.results if r.get('success')]
        if successful_tests:
            avg_time = sum(r['total_processing_time'] for r in successful_tests) / len(successful_tests)
            print(f"\nAverage processing time: {avg_time:.2f}s")

            # Processing speed analysis
            for result in successful_tests:
                if 'actual_duration' in result:
                    speed = result['actual_duration'] / result['total_processing_time']
                    print(f"  {result['video']}: {speed:.2f}x realtime")

if __name__ == "__main__":
    tester = ColdStartTester()
    tester.run_all_tests()