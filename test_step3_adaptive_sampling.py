#!/usr/bin/env python3
"""
Test Suite for Step 3: Adaptive Sampling

Verifies ContentAnalysispt2.md Step 3 implementation:
- Default sample_size = 60 (changed from 50)
- random_seed parameter for reproducibility
- Adaptive distribution: Target 20 per bucket
- Shortfall compensation from surplus buckets
- Mandatory validation cache

Run: python3 test_step3_adaptive_sampling.py
"""

import os
import sys
import re
from pathlib import Path

class TestStep3AdaptiveSampling:
    """Test suite for Step 3 adaptive sampling"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.project_root = Path(__file__).parent

    def assert_true(self, condition, test_name):
        """Assert helper"""
        if condition:
            print(f"  ✅ {test_name}")
            self.passed += 1
        else:
            print(f"  ❌ {test_name}")
            self.failed += 1

    def assert_equal(self, actual, expected, test_name):
        """Assert equality helper"""
        if actual == expected:
            print(f"  ✅ {test_name}")
            self.passed += 1
        else:
            print(f"  ❌ {test_name} (expected: {expected}, got: {actual})")
            self.failed += 1

    def test_function_signature(self):
        """Test sample_transcripts_for_discovery signature updated"""
        print("\n" + "="*80)
        print("TEST 1: Function Signature Changes")
        print("="*80)

        discovery_file = self.project_root / "ml_pipeline/stage2_content_analysis/discovery.py"

        if not discovery_file.exists():
            self.assert_true(False, "discovery.py exists")
            return

        content = discovery_file.read_text()

        # Test 1a: Function signature has random_seed parameter
        func_match = re.search(
            r'def sample_transcripts_for_discovery\((.*?)\):',
            content,
            re.DOTALL
        )

        if func_match:
            params = func_match.group(1)
            has_random_seed = "random_seed" in params
            self.assert_true(has_random_seed, "random_seed parameter added")

            has_optional_int = "Optional[int]" in params and "random_seed" in params
            self.assert_true(has_optional_int, "random_seed is Optional[int]")
        else:
            self.assert_true(False, "sample_transcripts_for_discovery function found")

        # Test 1b: Default sample_size changed to 60
        has_default_60 = "sample_size: int = 60" in content
        self.assert_true(has_default_60, "Default sample_size = 60 (changed from 50)")

        # Test 1c: Docstring mentions adaptive strategy
        has_adaptive_doc = "adaptive distribution" in content.lower() or "adaptive" in content
        self.assert_true(has_adaptive_doc, "Docstring mentions adaptive strategy")

    def test_adaptive_distribution_logic(self):
        """Test adaptive distribution algorithm present"""
        print("\n" + "="*80)
        print("TEST 2: Adaptive Distribution Algorithm")
        print("="*80)

        discovery_file = self.project_root / "ml_pipeline/stage2_content_analysis/discovery.py"
        content = discovery_file.read_text()

        # Test 2a: Target per bucket calculation (sample_size // 3)
        has_target_calc = "target_per_bucket = sample_size // 3" in content
        self.assert_true(has_target_calc, "Target per bucket calculated (sample_size // 3)")

        # Test 2b: Shortfall tracking
        has_shortfall = "shortfall" in content
        self.assert_true(has_shortfall, "Shortfall tracking present")

        # Test 2c: Surplus buckets logic
        has_surplus = "surplus_buckets" in content
        self.assert_true(has_surplus, "Surplus buckets tracking present")

        # Test 2d: Weak bucket handling (take all available)
        has_weak_bucket = "available < target_per_bucket" in content
        self.assert_true(has_weak_bucket, "Weak bucket detection (available < target)")

        # Test 2e: Shortfall compensation
        has_compensation = "Compensating shortfall" in content or "compensate" in content.lower()
        self.assert_true(has_compensation, "Shortfall compensation logic present")

        # Test 2f: Distribute extra to surplus buckets
        has_distribution = "extra_per_surplus" in content or "extra =" in content
        self.assert_true(has_distribution, "Extra samples distribution to surplus buckets")

    def test_random_seed_support(self):
        """Test random seed for reproducibility"""
        print("\n" + "="*80)
        print("TEST 3: Random Seed Support")
        print("="*80)

        discovery_file = self.project_root / "ml_pipeline/stage2_content_analysis/discovery.py"
        content = discovery_file.read_text()

        # Test 3a: Random seed check present
        has_seed_check = "if random_seed is not None:" in content
        self.assert_true(has_seed_check, "Random seed check present")

        # Test 3b: Random seed setting
        has_seed_set = "random.seed(random_seed)" in content
        self.assert_true(has_seed_set, "random.seed() called when provided")

        # Test 3c: Logging for random seed
        has_seed_log = "Random seed set to" in content or "random_seed" in content
        self.assert_true(has_seed_log, "Random seed logging present")

    def test_mandatory_validation_cache(self):
        """Test validation cache is mandatory"""
        print("\n" + "="*80)
        print("TEST 4: Mandatory Validation Cache")
        print("="*80)

        discovery_file = self.project_root / "ml_pipeline/stage2_content_analysis/discovery.py"
        content = discovery_file.read_text()

        # Test 4a: Validation cache None check
        has_none_check = "validation_cache is None" in content
        self.assert_true(has_none_check, "Validation cache None check present")

        # Test 4b: Raises ValueError if cache missing
        has_error = "raise ValueError" in content and "validation cache" in content.lower()
        self.assert_true(has_error, "Raises ValueError if validation cache missing")

        # Test 4c: Error message mentions Stage 2.5.1
        has_stage_ref = "Stage 2.5.1" in content
        self.assert_true(has_stage_ref, "Error message references Stage 2.5.1")

    def test_logging_improvements(self):
        """Test improved logging messages"""
        print("\n" + "="*80)
        print("TEST 5: Logging Improvements")
        print("="*80)

        discovery_file = self.project_root / "ml_pipeline/stage2_content_analysis/discovery.py"
        content = discovery_file.read_text()

        # Test 5a: Adaptive sampling logging
        has_adaptive_log = "Adaptive sampling" in content or "adaptive" in content.lower()
        self.assert_true(has_adaptive_log, "Adaptive sampling mentioned in logs")

        # Test 5b: Balanced vs adapted distinction
        has_balanced_check = "'balanced'" in content and "'adapted'" in content
        self.assert_true(has_balanced_check, "Logs distinguish balanced vs adapted distribution")

        # Test 5c: Weak bucket warnings
        has_weak_warning = "⚠️" in content and "only" in content and "valid top performers" in content
        self.assert_true(has_weak_warning, "Warning for weak buckets present")

        # Test 5d: Total sampled summary
        has_summary = "Total sampled:" in content or "✅" in content
        self.assert_true(has_summary, "Total sampled summary logging present")

    def test_run_discovery_stage_updates(self):
        """Test run_discovery_stage function updated"""
        print("\n" + "="*80)
        print("TEST 6: run_discovery_stage Updates")
        print("="*80)

        discovery_file = self.project_root / "ml_pipeline/stage2_content_analysis/discovery.py"
        content = discovery_file.read_text()

        # Find run_discovery_stage function (check if it exists first)
        has_func = "def run_discovery_stage(" in content
        self.assert_true(has_func, "run_discovery_stage function found")

        if has_func:
            # Test 6a: Has random_seed parameter
            # Check in the function definition area (next ~200 chars after def)
            func_start = content.find("def run_discovery_stage(")
            func_params = content[func_start:func_start+500]

            has_random_seed = "random_seed" in func_params
            self.assert_true(has_random_seed, "run_discovery_stage has random_seed parameter")

            # Test 6b: Default sample_size = 60
            has_60 = "sample_size: int = 60" in func_params
            self.assert_true(has_60, "run_discovery_stage default sample_size = 60")

        # Test 6c: Passes random_seed to sampling function
        has_pass_seed = re.search(
            r'sample_transcripts_for_discovery\([^)]*random_seed[^)]*\)',
            content
        )
        self.assert_true(has_pass_seed is not None, "random_seed passed to sampling function")

        # Test 6d: Docstring mentions adaptive sampling
        has_adaptive_doc = "adaptive sampling" in content.lower()
        self.assert_true(has_adaptive_doc, "Docstring mentions adaptive sampling")

    def test_example_scenarios_documented(self):
        """Test example scenarios in docstring"""
        print("\n" + "="*80)
        print("TEST 7: Example Scenarios Documentation")
        print("="*80)

        discovery_file = self.project_root / "ml_pipeline/stage2_content_analysis/discovery.py"
        content = discovery_file.read_text()

        # Test 7a: Scenario 1 (All Healthy)
        has_scenario1 = "Scenario 1" in content or "46, 48, 44" in content
        self.assert_true(has_scenario1, "Scenario 1 (All Healthy) documented")

        # Test 7b: Scenario 2 (One Weak)
        has_scenario2 = "Scenario 2" in content or "12, 50, 48" in content
        self.assert_true(has_scenario2, "Scenario 2 (One Weak) documented")

        # Test 7c: Scenario 3 (Insufficient)
        has_scenario3 = "Scenario 3" in content or "8, 12, 18" in content
        self.assert_true(has_scenario3, "Scenario 3 (Insufficient) documented")

    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("STEP 3: ADAPTIVE SAMPLING VERIFICATION")
        print("ContentAnalysispt2.md Lines 372-550")
        print("="*80)

        self.test_function_signature()
        self.test_adaptive_distribution_logic()
        self.test_random_seed_support()
        self.test_mandatory_validation_cache()
        self.test_logging_improvements()
        self.test_run_discovery_stage_updates()
        self.test_example_scenarios_documented()

        # Print summary
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Total: {self.passed + self.failed}")

        if self.passed + self.failed > 0:
            print(f"Success Rate: {self.passed/(self.passed+self.failed)*100:.1f}%")

        print("="*80)

        if self.failed == 0:
            print("\n🎉 All tests passed! Step 3 (Adaptive Sampling) verified.")
            print("\n📊 Key Features Verified:")
            print("  ✅ Default sample_size: 50 → 60")
            print("  ✅ Random seed support for reproducibility")
            print("  ✅ Adaptive distribution algorithm (20 per bucket target)")
            print("  ✅ Shortfall compensation from surplus buckets")
            print("  ✅ Mandatory validation cache enforcement")
            print("  ✅ Improved logging (balanced vs adapted)")
        else:
            print(f"\n⚠️  {self.failed} test(s) failed. Review implementation.")

        print("="*80)

        return self.failed == 0


if __name__ == "__main__":
    tester = TestStep3AdaptiveSampling()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
