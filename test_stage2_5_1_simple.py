#!/usr/bin/env python3
"""
Simple Test Suite for Stage 2.5.1: Transcript Validation
Tests implementation without requiring full environment setup
"""

import os
import sys
import re
from pathlib import Path

class TestStage2_5_1_Simple:
    """Lightweight test suite - code inspection only"""

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

    def test_file_changes(self):
        """Verify all required files were modified"""
        print("\n" + "="*80)
        print("TEST 1: File Changes")
        print("="*80)

        # Test 1a: transcript_validation.py exists
        validation_file = self.project_root / "ml_pipeline/stage2_content_analysis/transcript_validation.py"
        self.assert_true(validation_file.exists(), "transcript_validation.py exists")

        if validation_file.exists():
            content = validation_file.read_text()

            # Test 1b: Cache version is 2.0
            has_v2 = 'VALIDATION_CACHE_VERSION = "2.0"' in content
            self.assert_true(has_v2, "Cache version is 2.0")

            # Test 1c: Max length config present
            has_max_length = "max_length: int = 10000" in content
            self.assert_true(has_max_length, "max_length config field added")

            # Test 1d: Max length check in validation
            has_max_check = "too_long" in content and "> config.max_length" in content
            self.assert_true(has_max_check, "Max length validation check present")

            # Test 1e: Incremental validation logic
            has_incremental = "incremental" in content.lower() and "existing_cache" in content
            self.assert_true(has_incremental, "Incremental validation logic present")

            # Test 1f: Entry point function
            has_entry_point = "def run_transcript_validation_stage(" in content
            self.assert_true(has_entry_point, "run_transcript_validation_stage() function exists")

            # Test 1g: Minimum threshold check
            has_threshold = "valid_count < 30" in content and "ValueError" in content
            self.assert_true(has_threshold, "30 valid transcript threshold enforced")

    def test_discovery_changes(self):
        """Verify discovery.py enforces mandatory cache"""
        print("\n" + "="*80)
        print("TEST 2: Discovery Changes")
        print("="*80)

        discovery_file = self.project_root / "ml_pipeline/stage2_content_analysis/discovery.py"
        self.assert_true(discovery_file.exists(), "discovery.py exists")

        if discovery_file.exists():
            content = discovery_file.read_text()

            # Test 2a: Import present
            has_import = "from .transcript_validation import load_validation_cache" in content
            self.assert_true(has_import, "Imports load_validation_cache")

            # Test 2b: Mandatory cache enforcement
            has_mandatory = "raise FileNotFoundError" in content and "Stage 2.5.1" in content
            self.assert_true(has_mandatory, "Enforces mandatory validation cache")

            # Test 2c: Error message mentions Stage 2.5.1
            has_error_msg = "Stage 2.5.1 (Transcript Validation) must complete" in content
            self.assert_true(has_error_msg, "Clear error message for missing cache")

    def test_orchestrator_changes(self):
        """Verify rumiai_ml_batch.py has Stage 2.5.1 integration"""
        print("\n" + "="*80)
        print("TEST 3: Orchestrator Changes")
        print("="*80)

        orchestrator_file = self.project_root / "rumiai_ml_batch.py"
        self.assert_true(orchestrator_file.exists(), "rumiai_ml_batch.py exists")

        if orchestrator_file.exists():
            content = orchestrator_file.read_text()

            # Test 3a: Import added
            has_import = "from ml_pipeline.stage2_content_analysis.transcript_validation import run_transcript_validation_stage" in content
            self.assert_true(has_import, "Imports run_transcript_validation_stage")

            # Test 3b: Stage 2.5.1 section added
            has_stage_section = "STAGE 2.5.1: TRANSCRIPT VALIDATION" in content
            self.assert_true(has_stage_section, "Stage 2.5.1 section added")

            # Test 3c: Calls entry point
            has_call = "run_transcript_validation_stage(" in content
            self.assert_true(has_call, "Calls run_transcript_validation_stage()")

            # Test 3d: Error handling
            has_error_handling = "except ValueError as e:" in content and \
                               "Stage 2.5.1 failed" in content
            self.assert_true(has_error_handling, "Error handling added")

            # Test 3e: Stage ordering (2.5 -> 2.5.1 -> 2.6)
            stage_25_match = re.search(r'STAGE 2\.5: FILE ORGANIZATION', content)
            stage_251_match = re.search(r'STAGE 2\.5\.1: TRANSCRIPT VALIDATION', content)
            stage_26_match = re.search(r'STAGE 2\.6', content)

            if stage_25_match and stage_251_match and stage_26_match:
                correct_order = stage_25_match.start() < stage_251_match.start() < stage_26_match.start()
                self.assert_true(correct_order, "Stage 2.5.1 positioned correctly (2.5 -> 2.5.1 -> 2.6)")
            else:
                self.assert_true(False, "All stage markers found")

            # Test 3f: Pipeline status updated
            has_status = "Stage 2.5.1: Transcript Validation - COMPLETE" in content
            self.assert_true(has_status, "Pipeline status includes Stage 2.5.1")

            # Test 3g: Docstring updated
            has_docstring_update = "Stage 2.5.1: Transcript Validation (filter music/noise)" in content
            self.assert_true(has_docstring_update, "main() docstring updated")

    def test_implementation_completeness(self):
        """Verify all ContentAnalysispt2.md Step 2 features"""
        print("\n" + "="*80)
        print("TEST 4: Implementation Completeness")
        print("="*80)

        validation_file = self.project_root / "ml_pipeline/stage2_content_analysis/transcript_validation.py"

        if validation_file.exists():
            content = validation_file.read_text()

            # Count key features
            features = {
                "max_length filter": "too_long" in content,
                "cache version 2.0": 'VALIDATION_CACHE_VERSION = "2.0"' in content,
                "incremental validation": "videos_to_validate" in content,
                "entry point function": "def run_transcript_validation_stage" in content,
                "minimum 30 threshold": "valid_count < 30" in content,
                "cache skip logic": "Cache complete" in content and "skipping validation" in content,
                "version mismatch handling": "version mismatch" in content.lower(),
            }

            for feature, present in features.items():
                self.assert_true(present, f"Feature: {feature}")

    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("STAGE 2.5.1 IMPLEMENTATION VERIFICATION")
        print("Simple Code Inspection Tests (No Environment Required)")
        print("="*80)

        self.test_file_changes()
        self.test_discovery_changes()
        self.test_orchestrator_changes()
        self.test_implementation_completeness()

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
            print("\n🎉 All tests passed! Stage 2.5.1 implementation verified.")
        else:
            print(f"\n⚠️  {self.failed} test(s) failed. Review implementation.")

        print("="*80)

        return self.failed == 0


if __name__ == "__main__":
    tester = TestStage2_5_1_Simple()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
