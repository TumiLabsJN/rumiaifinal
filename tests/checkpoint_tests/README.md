# Stage 3 Checkpoint Fallback Test Suite

## Overview

This test suite validates the graceful degradation and CSV fallback validation feature implemented for Stage 3 checkpoint handling.

## Feature Being Tested

**Problem**: If Stage 3 checkpoint write fails (disk full, permissions, etc.), the pipeline would crash even though the CSV output is valid.

**Solution**:
1. **Graceful degradation**: Stage 3 logs warning but doesn't raise exception when checkpoint write fails
2. **CSV fallback validation**: Orchestrator validates Stage 3 completion via CSV if checkpoint is missing

## Test Scripts

### 1. `test_stage3_checkpoint_normal.py`
**Purpose**: Validate normal checkpoint flow (happy path)

**Tests**:
- Stage 3 creates `aggregated_features.csv` successfully
- Stage 3 creates `stage_3_checkpoint.json` successfully
- Checkpoint has correct schema with all required fields
- Orchestrator validates via checkpoint (primary path)

**Expected Result**: ✓ All validations pass

---

### 2. `test_stage3_checkpoint_failure.py`
**Purpose**: Validate graceful degradation when checkpoint write fails

**Tests**:
- Stage 3 creates CSV successfully
- Checkpoint write fails (simulated via read-only directory)
- Stage 3 logs WARNING (not ERROR)
- Stage 3 returns successfully (no exception raised)
- CSV is valid and usable
- Orchestrator uses CSV fallback validation

**Expected Result**: ✓ Pipeline continues despite checkpoint failure

**Key Validation**: Stage 3 does NOT raise exception when checkpoint fails

---

### 3. `test_orchestrator_fallback.py`
**Purpose**: Validate all orchestrator validation paths

**Tests** (5 scenarios):
1. **Primary path**: Checkpoint exists and `status == "completed"` → Proceed ✓
2. **Fallback path**: Checkpoint missing, CSV exists → Proceed ✓
3. **Fail path**: Both checkpoint and CSV missing → Skip bucket ✓
4. **Edge case**: Checkpoint exists but `status != "completed"` → Skip bucket ✓
5. **Edge case**: CSV exists but empty (0 bytes) → Skip bucket ✓

**Expected Result**: ✓ All 5 scenarios pass

---

### 4. `run_all_tests.sh`
**Purpose**: Master test runner that executes all tests

**Usage**:
```bash
cd tests/checkpoint_tests
./run_all_tests.sh
```

**Output**: Color-coded summary of all test results

---

## Running the Tests

### Run All Tests (Recommended)
```bash
cd /home/jorge/rumiaifinal/tests/checkpoint_tests
./run_all_tests.sh
```

### Run Individual Tests
```bash
cd /home/jorge/rumiaifinal/tests/checkpoint_tests

# Test 1: Normal flow
python3 test_stage3_checkpoint_normal.py

# Test 2: Graceful degradation
python3 test_stage3_checkpoint_failure.py

# Test 3: Orchestrator fallback
python3 test_orchestrator_fallback.py
```

---

## Expected Output

### Successful Test Run
```
========================================
Stage 3 Checkpoint Fallback Test Suite
========================================

Running: Test 1: Normal Checkpoint Flow
TEST RESULT: ✓ PASSED
✓ Test 1: Normal Checkpoint Flow PASSED

Running: Test 2: Checkpoint Write Failure
TEST RESULT: ✓ PASSED
✓ Test 2: Checkpoint Write Failure PASSED

Running: Test 3: Orchestrator Fallback Logic
TEST RESULT: ✓ ALL SCENARIOS PASSED
✓ Test 3: Orchestrator Fallback Logic PASSED

========================================
TEST SUITE SUMMARY
========================================

Total tests:  3
Passed:       3
Failed:       0

========================================
✓ ALL TESTS PASSED
========================================

Validated features:
  ✓ Normal checkpoint creation and validation
  ✓ Graceful degradation on checkpoint write failure
  ✓ CSV fallback validation in orchestrator
  ✓ Stage 3 continues successfully despite checkpoint failure
  ✓ Pipeline doesn't break when checkpoint write fails

The checkpoint fallback feature is working correctly!
```

---

## Test Data

All tests create temporary directories with minimal synthetic data:
- Test videos: Minimal JSON with required temporal window fields
- CSVs: Small 10-row CSVs with essential columns
- Checkpoints: Valid JSON with Stage 3 checkpoint schema

**Cleanup**: All tests clean up their temporary directories after completion.

---

## Test Coverage

| Scenario | Test Script | Coverage |
|----------|-------------|----------|
| Normal checkpoint creation | Test 1 | ✓ Primary path validation |
| Checkpoint write failure | Test 2 | ✓ Graceful degradation |
| CSV fallback validation | Test 2, Test 3 | ✓ Fallback path validation |
| Both checkpoint and CSV missing | Test 3 | ✓ Fail path validation |
| Invalid checkpoint status | Test 3 | ✓ Edge case handling |
| Empty CSV | Test 3 | ✓ Edge case handling |

**Total coverage**: 6 distinct scenarios across 8 test cases

---

## Troubleshooting

### Test Fails: "Stage 3 raised exception"
**Cause**: Graceful degradation not working (checkpoint failure raises exception)

**Fix**: Check `scripts/stage3_aggregation.py` lines 624-639 - should log WARNING, not raise

---

### Test Fails: "Orchestrator would skip bucket"
**Cause**: Fallback validation not implemented in orchestrator

**Fix**: Check `rumiai_ml_batch.py` lines 1152-1196 - should have CSV fallback logic

---

### Test Fails: "CSV not created"
**Cause**: Stage 3 aggregation logic issue (unrelated to checkpoint feature)

**Fix**: Check Stage 3 implementation and test data setup

---

## Files Modified by Feature

1. **`scripts/stage3_aggregation.py`** (lines 624-639)
   - Graceful degradation: Don't raise on checkpoint write failure

2. **`rumiai_ml_batch.py`** (lines 1152-1196)
   - CSV fallback validation: Check CSV if checkpoint missing

---

## Philosophy

**Checkpoint** = Performance optimization (enables skip logic)
**CSV** = Source of truth (actual data Stage 4 needs)

**If checkpoint fails**:
- ✅ Stage 3 succeeds (CSV is valid)
- ✅ Stage 4 proceeds (validates via CSV)
- ⚠️ Stage 3 re-runs on resume (10 min cost, acceptable trade-off)

**Better than**:
- ❌ Pipeline breaks
- ❌ User must debug checkpoint issue
- ❌ Valid CSV sits unused

---

## Author

Created as part of Stage 3 checkpoint fallback feature implementation
Date: 2025-01-28
