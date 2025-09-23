#!/usr/bin/env python3
"""
Thread Scaling Analysis for ML Services
Tests each service with different thread counts to find optimal configuration
"""

import os
import sys
import subprocess
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List
import statistics

class ThreadScalingTester:
    def __init__(self):
        self.test_video = {
            'url': 'https://www.tiktok.com/@meganlock_/video/7428596413707144481',
            'id': '7428596413707144481',
            'duration': 18  # Short video for quick tests
        }

        self.thread_counts = [1, 2, 4, 8, 16]
        self.services_to_test = {
            'whisper': {'env_var': 'WHISPER_THREADS', 'flexibility': 'Direct'},
            'yolo': {'env_var': 'CV2_THREADS', 'flexibility': 'cv2.set()'},
            'ocr': {'env_var': 'OMP_NUM_THREADS', 'flexibility': 'Direct'},
        }

        self.results = {}

    def clear_caches(self):
        """Clear all caches for clean test."""
        # Clear temp frames
        patterns = ['/tmp/rumiai_frames_*', '/tmp/tmp*frame*.jpg']
        for pattern in patterns:
            subprocess.run(f'rm -rf {pattern}', shell=True, capture_output=True)

        # Clear service outputs
        output_dirs = Path('/home/jorge/rumiaifinal').glob('*_outputs')
        for dir in output_dirs:
            if dir.exists():
                shutil.rmtree(dir, ignore_errors=True)
                dir.mkdir(exist_ok=True)

        time.sleep(2)

    def run_single_test(self, service: str, thread_count: int) -> Dict:
        """Test a single service with specific thread count."""
        print(f"  Testing {service} with {thread_count} threads...")

        # Create a minimal test script that only runs one service
        test_script = f'''
import os
import sys
os.environ['{self.services_to_test[service]["env_var"]}'] = '{thread_count}'
os.environ['SERVICES_TO_RUN'] = '{service}'  # Only run this service
sys.path.insert(0, '/home/jorge/rumiaifinal')

from scripts.rumiai_runner import RumiAIProcessor
import asyncio

async def test():
    processor = RumiAIProcessor()
    await processor.process_video_url('{self.test_video["url"]}')

asyncio.run(test())
'''

        # Write and run test script
        test_file = Path('/tmp/thread_test.py')
        test_file.write_text(test_script)

        start_time = time.time()
        result = subprocess.run(
            ['python3', str(test_file)],
            capture_output=True,
            text=True,
            timeout=120
        )
        execution_time = time.time() - start_time

        return {
            'thread_count': thread_count,
            'execution_time': execution_time,
            'success': result.returncode == 0
        }

    def test_service(self, service: str):
        """Test a service with all thread counts."""
        print(f"\nTesting {service}...")
        service_results = []

        for thread_count in self.thread_counts:
            # Run 3 times for average
            runs = []
            for run in range(3):
                self.clear_caches()
                result = self.run_single_test(service, thread_count)
                runs.append(result['execution_time'])
                print(f"    Run {run+1}: {result['execution_time']:.2f}s")

            avg_time = statistics.mean(runs)
            std_dev = statistics.stdev(runs) if len(runs) > 1 else 0

            service_results.append({
                'threads': thread_count,
                'avg_time': avg_time,
                'std_dev': std_dev
            })

            print(f"  {thread_count} threads: {avg_time:.2f}s ± {std_dev:.2f}s")

        self.results[service] = service_results

        # Find optimal
        optimal = min(service_results, key=lambda x: x['avg_time'])
        print(f"✅ Optimal for {service}: {optimal['threads']} threads")

    def generate_report(self):
        """Generate optimization report."""
        print("\n" + "="*60)
        print("THREAD SCALING RESULTS")
        print("="*60)

        optimal_config = {}
        for service, results in self.results.items():
            optimal = min(results, key=lambda x: x['avg_time'])
            baseline = next(r for r in results if r['threads'] == 1)
            speedup = baseline['avg_time'] / optimal['avg_time']

            optimal_config[service] = {
                'threads': optimal['threads'],
                'speedup': speedup,
                'env_var': self.services_to_test[service]['env_var']
            }

            print(f"\n{service}:")
            print(f"  Optimal: {optimal['threads']} threads")
            print(f"  Speedup: {speedup:.2f}x vs single thread")

        # Save configuration
        print("\n" + "="*60)
        print("OPTIMAL CONFIGURATION")
        print("="*60)
        print("Add to your .bashrc or export before running:")
        for service, config in optimal_config.items():
            print(f"export {config['env_var']}={config['threads']}")

        # Save to file
        config_file = Path('/home/jorge/rumiaifinal/optimal_threads.json')
        with open(config_file, 'w') as f:
            json.dump(optimal_config, f, indent=2)
        print(f"\n✅ Configuration saved to {config_file}")

    def run_all_tests(self):
        """Run complete analysis."""
        print("THREAD SCALING ANALYSIS")
        print("="*60)
        print(f"Testing with thread counts: {self.thread_counts}")
        print(f"Each configuration tested 3 times\n")

        for service in self.services_to_test:
            self.test_service(service)

        self.generate_report()

if __name__ == "__main__":
    tester = ThreadScalingTester()
    if len(sys.argv) > 1 and sys.argv[1] in tester.services_to_test:
        tester.test_service(sys.argv[1])
        tester.generate_report()
    else:
        tester.run_all_tests()