#!/usr/bin/env python3
"""
Automated test runner for RumiAI v2.

Runs all tests and generates report.
NO HUMAN INTERVENTION REQUIRED.
"""
import unittest
import sys
import json
from pathlib import Path
import time
from io import StringIO
import traceback


class TestResult(unittest.TestResult):
    """Custom test result to capture detailed information."""
    
    def __init__(self):
        super().__init__()
        self.test_results = []
        self.start_time = time.time()
    
    def startTest(self, test):
        super().startTest(test)
        self.test_start = time.time()
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_results.append({
            'test': str(test),
            'status': 'PASS',
            'time': time.time() - self.test_start
        })
    
    def addError(self, test, err):
        super().addError(test, err)
        self.test_results.append({
            'test': str(test),
            'status': 'ERROR',
            'time': time.time() - self.test_start,
            'error': self._exc_info_to_string(err, test)
        })
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.test_results.append({
            'test': str(test),
            'status': 'FAIL',
            'time': time.time() - self.test_start,
            'error': self._exc_info_to_string(err, test)
        })
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_results.append({
            'test': str(test),
            'status': 'SKIP',
            'time': 0,
            'reason': reason
        })


def run_all_tests():
    """Run all tests and generate report."""
    print("🧪 RumiAI v2 Automated Test Suite")
    print("=" * 50)
    
    # Discover all tests
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests with custom result
    result = TestResult()
    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=TestResult,
        stream=sys.stdout
    )
    
    # Capture output
    test_output = StringIO()
    runner.stream = test_output
    
    # Run the tests
    runner.run(suite)
    
    # Generate report
    total_time = time.time() - result.start_time
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    # Count results
    passed = sum(1 for r in result.test_results if r['status'] == 'PASS')
    failed = sum(1 for r in result.test_results if r['status'] == 'FAIL')
    errors = sum(1 for r in result.test_results if r['status'] == 'ERROR')
    skipped = sum(1 for r in result.test_results if r['status'] == 'SKIP')
    total = len(result.test_results)
    
    # Print summary
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"💥 Errors: {errors}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    
    if failed + errors > 0:
        print("\n❌ FAILED TESTS:")
        for test_result in result.test_results:
            if test_result['status'] in ['FAIL', 'ERROR']:
                print(f"\n{test_result['test']}:")
                print(test_result.get('error', 'No error details'))
    
    # Generate JSON report
    report = {
        'summary': {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'duration': total_time,
            'success_rate': (passed / total * 100) if total > 0 else 0
        },
        'tests': result.test_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save report
    report_path = test_dir / 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Exit code based on results
    if failed + errors == 0:
        print("\n✅ ALL TESTS PASSED! The Pure Big Bang implementation is ready!")
        return 0
    else:
        print("\n❌ TESTS FAILED! Please fix the issues before deployment.")
        return 1


def run_single_test(test_name):
    """Run a single test file."""
    print(f"🧪 Running single test: {test_name}")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test
        exit_code = run_single_test(sys.argv[1])
    else:
        # Run all tests
        exit_code = run_all_tests()
    
    sys.exit(exit_code)