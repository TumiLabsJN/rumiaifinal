#!/usr/bin/env python3
"""
Quick integration test for Stage 5 ML Model Training.

Tests that the integration code is syntactically correct and
can be called with proper parameters.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all Stage 5 imports work."""
    print("=" * 80)
    print("TEST 1: Import Verification")
    print("=" * 80)

    try:
        from rumiai_v2.processors.model_training import (
            run_stage5_training,
            StageInputError,
            InsufficientDataError,
            ModelTrainingError,
            ValidationError
        )
        print("✓ All Stage 5 imports successful")
        print(f"  - run_stage5_training: {type(run_stage5_training)}")
        print(f"  - StageInputError: {StageInputError}")
        print(f"  - InsufficientDataError: {InsufficientDataError}")
        print(f"  - ModelTrainingError: {ModelTrainingError}")
        print(f"  - ValidationError: {ValidationError}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_function_signature():
    """Test that run_stage5_training has correct signature."""
    print("\n" + "=" * 80)
    print("TEST 2: Function Signature Verification")
    print("=" * 80)

    try:
        from rumiai_v2.processors.model_training import run_stage5_training
        import inspect

        sig = inspect.signature(run_stage5_training)
        params = list(sig.parameters.keys())

        print(f"Function signature: {sig}")
        print(f"Parameters: {params}")

        # Expected parameters from TI Section 11.4
        expected_params = ['bucket_path', 'config', 'selection_strategy']

        if params == expected_params:
            print(f"✓ Function signature matches TI specification")
            print(f"  Expected: {expected_params}")
            print(f"  Actual:   {params}")
            return True
        else:
            print(f"✗ Function signature mismatch")
            print(f"  Expected: {expected_params}")
            print(f"  Actual:   {params}")
            return False

    except Exception as e:
        print(f"✗ Signature check failed: {e}")
        return False


def test_orchestrator_syntax():
    """Test that rumiai_ml_batch.py has valid syntax."""
    print("\n" + "=" * 80)
    print("TEST 3: Orchestrator Syntax Check")
    print("=" * 80)

    try:
        import py_compile
        py_compile.compile('rumiai_ml_batch.py', doraise=True)
        print("✓ rumiai_ml_batch.py syntax is valid")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ Syntax error in rumiai_ml_batch.py: {e}")
        return False


def test_stage5_code_present():
    """Test that Stage 5 code exists in orchestrator."""
    print("\n" + "=" * 80)
    print("TEST 4: Stage 5 Code Presence Check")
    print("=" * 80)

    try:
        with open('rumiai_ml_batch.py', 'r') as f:
            content = f.read()

        checks = {
            "Import statement": "from rumiai_v2.processors.model_training import",
            "Stage 5 header": "STAGE 5: ML MODEL TRAINING",
            "run_stage5_training call": "run_stage5_training(",
            "StageInputError handler": "except StageInputError",
            "Checkpoint creation": 'stage_5_ml_model_training',
            "Pipeline status updated": "Stage 5: ML Model Training - COMPLETE"
        }

        all_passed = True
        for check_name, search_string in checks.items():
            if search_string in content:
                print(f"✓ {check_name} found")
            else:
                print(f"✗ {check_name} NOT FOUND")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"✗ File read failed: {e}")
        return False


def test_exception_handlers():
    """Test that all exception handlers are present."""
    print("\n" + "=" * 80)
    print("TEST 5: Exception Handler Verification")
    print("=" * 80)

    try:
        with open('rumiai_ml_batch.py', 'r') as f:
            content = f.read()

        exceptions = [
            "except StageInputError",
            "except InsufficientDataError",
            "except ModelTrainingError",
            "except ValidationError",
            "except (IOError, OSError)",
            "except Exception"
        ]

        all_found = True
        for exc in exceptions:
            if exc in content:
                print(f"✓ {exc} handler found")
            else:
                print(f"✗ {exc} handler NOT FOUND")
                all_found = False

        return all_found

    except Exception as e:
        print(f"✗ Exception handler check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "STAGE 5 INTEGRATION TEST SUITE" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    tests = [
        ("Import Verification", test_imports),
        ("Function Signature", test_function_signature),
        ("Orchestrator Syntax", test_orchestrator_syntax),
        ("Stage 5 Code Presence", test_stage5_code_present),
        ("Exception Handlers", test_exception_handlers)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")

    print("=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n🎉 All tests passed! Stage 5 integration is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
