#!/usr/bin/env python3
"""
Cold Start Performance Comparison Test for Sequential vs Parallel Modes
Tests each video in both modes with complete cache clearing
"""

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

class ModeComparisonTester:
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

        self.results = {
            'parallel': [],
            'sequential': []
        }
        self.results_file = Path('/home/jorge/rumiaifinal/mode_comparison_results.json')

    def clear_all_caches(self):
        """Remove ALL caches and temporary files - proven to work"""
        print("\n🧹 Clearing all caches...")

        # 1. Clear frame caches in /tmp
        tmp_patterns = [
            '/tmp/rumiai_frames_*',
            '/tmp/tmp*frame*.jpg',
            '/tmp/vision_test_*',
            '/tmp/rumiai_run.log',
            '/tmp/*.wav',  # Audio files
            '/tmp/*.mp3'
        ]

        for pattern in tmp_patterns:
            subprocess.run(f'rm -rf {pattern}', shell=True, capture_output=True)

        # 2. Clear video downloads
        temp_dir = Path('/home/jorge/rumiaifinal/temp')
        if temp_dir.exists():
            for video in temp_dir.glob('*.mp4'):
                print(f"  Removing cached video: {video.name}")
                video.unlink()

        # 3. Clear ALL service outputs (including subdirectories!)
        service_dirs = [
            '/home/jorge/rumiaifinal/yolo_outputs',
            '/home/jorge/rumiaifinal/whisper_outputs',
            '/home/jorge/rumiaifinal/mediapipe_outputs',
            '/home/jorge/rumiaifinal/ocr_outputs',
            '/home/jorge/rumiaifinal/scene_detection_outputs',
            '/home/jorge/rumiaifinal/audio_energy_outputs',
            '/home/jorge/rumiaifinal/emotion_detection_outputs',
            '/home/jorge/rumiaifinal/gender_detection_outputs',
            '/home/jorge/rumiaifinal/creative_analysis_outputs'
        ]

        for service_dir in service_dirs:
            if Path(service_dir).exists():
                shutil.rmtree(service_dir, ignore_errors=True)
                Path(service_dir).mkdir(exist_ok=True)  # Recreate empty

        # 4. Clear Python cache
        cache_dirs = [
            '/home/jorge/rumiaifinal/__pycache__',
            '/home/jorge/rumiaifinal/**/__pycache__'
        ]

        for pattern in cache_dirs:
            subprocess.run(f'rm -rf {pattern}', shell=True, capture_output=True)

        # 5. Clear insights output
        test_insights = Path('/home/jorge/rumiaifinal/insights')
        if test_insights.exists():
            for test_dir in test_insights.glob('test_*'):
                shutil.rmtree(test_dir, ignore_errors=True)

        # 6. Kill any lingering Python processes (except this one)
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

        print("  ✓ Cache clearing complete")

    def run_single_test(self, video_config, mode):
        """Run a single test with specified mode"""
        print(f"\n{'='*60}")
        print(f"Testing: {video_config['name']} in {mode.upper()} mode")
        print(f"URL: {video_config['url']}")
        print(f"Expected duration: {video_config['expected_duration']}s")
        print(f"{'='*60}")

        # Clear all caches
        self.clear_all_caches()

        # Give system a moment to settle
        time.sleep(2)

        # Set environment for the mode
        env = os.environ.copy()
        if mode == 'sequential':
            env['SEQUENTIAL_TEST'] = 'true'
            env['METRICS_DISPLAY'] = 'summary'
        else:
            # Ensure sequential mode is OFF for parallel
            if 'SEQUENTIAL_TEST' in env:
                del env['SEQUENTIAL_TEST']
            env['METRICS_DISPLAY'] = 'summary'

        # Record start time
        start_time = time.time()

        # Run the actual production command
        print(f"\n📊 Running in {mode} mode...")
        print(f"python3 scripts/rumiai_runner.py '{video_config['url']}'")

        try:
            result = subprocess.run(
                ['python3', 'scripts/rumiai_runner.py', video_config['url']],
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            # Record end time
            end_time = time.time()
            total_time = end_time - start_time

            # Parse results
            test_result = {
                'video': video_config['name'],
                'url': video_config['url'],
                'mode': mode,
                'expected_duration': video_config['expected_duration'],
                'total_processing_time': total_time,
                'success': result.returncode == 0,
                'timestamp': datetime.now().isoformat()
            }

            if result.returncode == 0:
                print(f"✅ Success! Total time: {total_time:.2f}s")

                # Try to find instrumentation metrics
                metrics_dir = Path('/home/jorge/rumiaifinal/instrumentation_metrics')
                if metrics_dir.exists():
                    # Find the most recent metrics file
                    metrics_files = sorted(metrics_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
                    if metrics_files:
                        with open(metrics_files[0], 'r') as f:
                            metrics = json.load(f)
                            test_result['instrumentation'] = {
                                'bottleneck_service': metrics['pipeline'].get('bottleneck_service'),
                                'max_service_time': metrics['pipeline'].get('max_service_time'),
                                'total_memory_mb': metrics['pipeline'].get('total_memory_mb'),
                                'services': {}
                            }
                            # Add individual service times
                            for service, data in metrics['services'].items():
                                test_result['instrumentation']['services'][service] = {
                                    'time': data.get('processing_time', 0),
                                    'success': data.get('success', False)
                                }

                # Check for actual video duration
                insights_dir = Path('/home/jorge/rumiaifinal/insights')
                if insights_dir.exists():
                    latest_dirs = sorted(insights_dir.glob('*'), key=os.path.getctime, reverse=True)
                    if latest_dirs:
                        temporal_file = latest_dirs[0] / 'temporal_windows_updated.json'
                        if temporal_file.exists():
                            with open(temporal_file, 'r') as f:
                                temporal_data = json.load(f)
                                test_result['actual_duration'] = temporal_data.get('duration', 0)
            else:
                print(f"❌ Failed with return code: {result.returncode}")
                test_result['error'] = result.stderr[-1000:] if result.stderr else 'Unknown error'

            return test_result

        except subprocess.TimeoutExpired:
            print(f"⏱️ Test timed out after 600 seconds")
            return {
                'video': video_config['name'],
                'mode': mode,
                'total_processing_time': 600,
                'success': False,
                'error': 'Timeout after 600s'
            }

    def compare_results(self):
        """Compare parallel vs sequential results"""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON: PARALLEL vs SEQUENTIAL")
        print("="*80)

        comparison = []

        for video_config in self.test_videos:
            video_name = video_config['name']

            # Find results for this video in both modes
            parallel_result = next((r for r in self.results['parallel'] if r['video'] == video_name), None)
            sequential_result = next((r for r in self.results['sequential'] if r['video'] == video_name), None)

            if parallel_result and sequential_result:
                parallel_time = parallel_result.get('total_processing_time', 0)
                sequential_time = sequential_result.get('total_processing_time', 0)

                speedup = (parallel_time - sequential_time) / parallel_time * 100 if parallel_time > 0 else 0

                comparison.append({
                    'video': video_name,
                    'duration': video_config['expected_duration'],
                    'parallel_time': parallel_time,
                    'sequential_time': sequential_time,
                    'difference': sequential_time - parallel_time,
                    'speedup_percent': speedup,
                    'faster_mode': 'sequential' if sequential_time < parallel_time else 'parallel'
                })

        # Print comparison table
        print(f"\n{'Video':<15} {'Duration':<10} {'Parallel':<12} {'Sequential':<12} {'Difference':<12} {'Faster':<10}")
        print("-"*80)

        for comp in comparison:
            print(f"{comp['video']:<15} {comp['duration']:<10}s "
                  f"{comp['parallel_time']:<12.2f}s {comp['sequential_time']:<12.2f}s "
                  f"{comp['difference']:<+12.2f}s {comp['faster_mode']:<10}")

        # Calculate averages
        if comparison:
            avg_parallel = sum(c['parallel_time'] for c in comparison) / len(comparison)
            avg_sequential = sum(c['sequential_time'] for c in comparison) / len(comparison)
            avg_difference = avg_sequential - avg_parallel

            print("-"*80)
            print(f"{'AVERAGE':<15} {'':<10} {avg_parallel:<12.2f}s {avg_sequential:<12.2f}s {avg_difference:<+12.2f}s")

            # Overall recommendation
            print("\n" + "="*80)
            print("RECOMMENDATION")
            print("="*80)

            if avg_sequential < avg_parallel:
                improvement = (avg_parallel - avg_sequential) / avg_parallel * 100
                print(f"✅ SEQUENTIAL mode is {improvement:.1f}% faster on average")
                print(f"   Average time saved: {avg_parallel - avg_sequential:.2f}s per video")
            else:
                improvement = (avg_sequential - avg_parallel) / avg_sequential * 100
                print(f"✅ PARALLEL mode is {improvement:.1f}% faster on average")
                print(f"   Average time saved: {avg_sequential - avg_parallel:.2f}s per video")

        return comparison

    def run_all_tests(self):
        """Run all tests in both modes"""
        print("\n" + "="*80)
        print("COLD START PERFORMANCE COMPARISON TEST")
        print("="*80)
        print("This test will run each video in both parallel and sequential modes")
        print("with complete cache clearing between each test.")
        print("\nTest videos:")
        for video in self.test_videos:
            print(f"  - {video['name']}: {video['expected_duration']}s")

        # Test each video in both modes
        for video_config in self.test_videos:
            # Test in parallel mode
            result = self.run_single_test(video_config, 'parallel')
            self.results['parallel'].append(result)

            # Test in sequential mode
            result = self.run_single_test(video_config, 'sequential')
            self.results['sequential'].append(result)

        # Compare results
        comparison = self.compare_results()

        # Save full results
        full_results = {
            'test_date': datetime.now().isoformat(),
            'results': self.results,
            'comparison': comparison
        }

        with open(self.results_file, 'w') as f:
            json.dump(full_results, f, indent=2)

        print(f"\n✅ Full results saved to: {self.results_file}")

        # Print individual service comparisons if available
        self.print_service_comparison()

    def print_service_comparison(self):
        """Print detailed service-level comparison"""
        print("\n" + "="*80)
        print("SERVICE-LEVEL COMPARISON (120s video)")
        print("="*80)

        # Find 120s video results with instrumentation
        parallel_120 = next((r for r in self.results['parallel']
                            if r['video'] == '120s_video' and 'instrumentation' in r), None)
        sequential_120 = next((r for r in self.results['sequential']
                              if r['video'] == '120s_video' and 'instrumentation' in r), None)

        if parallel_120 and sequential_120:
            print(f"\n{'Service':<20} {'Parallel Time':<15} {'Sequential Time':<15} {'Difference':<12}")
            print("-"*60)

            services = set()
            if 'services' in parallel_120['instrumentation']:
                services.update(parallel_120['instrumentation']['services'].keys())
            if 'services' in sequential_120['instrumentation']:
                services.update(sequential_120['instrumentation']['services'].keys())

            for service in sorted(services):
                p_time = parallel_120['instrumentation']['services'].get(service, {}).get('time', 0)
                s_time = sequential_120['instrumentation']['services'].get(service, {}).get('time', 0)
                diff = s_time - p_time

                print(f"{service:<20} {p_time:<15.2f}s {s_time:<15.2f}s {diff:<+12.2f}s")

            # Print bottleneck info
            print(f"\nBottleneck Service:")
            print(f"  Parallel: {parallel_120['instrumentation'].get('bottleneck_service')} "
                  f"({parallel_120['instrumentation'].get('max_service_time', 0):.2f}s)")
            print(f"  Sequential: {sequential_120['instrumentation'].get('bottleneck_service')} "
                  f"({sequential_120['instrumentation'].get('max_service_time', 0):.2f}s)")

            if sequential_120['instrumentation'].get('total_memory_mb'):
                print(f"\nMemory Usage (Sequential): {sequential_120['instrumentation']['total_memory_mb']:.1f} MB")

if __name__ == "__main__":
    tester = ModeComparisonTester()
    tester.run_all_tests()