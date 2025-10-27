#!/bin/bash
#
# Master Test Runner: Stage 3 Checkpoint Fallback Feature
#
# Tests the graceful degradation and CSV fallback validation
# implemented for Stage 3 checkpoint handling.
#
# Usage:
#   ./run_all_tests.sh
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test script directory
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEST_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Stage 3 Checkpoint Fallback Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Testing graceful degradation and CSV fallback validation"
echo "Test directory: $TEST_DIR"
echo ""

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_script="$2"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: $test_name${NC}"
    echo -e "${BLUE}========================================${NC}"

    if python3 "$test_script"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        return 0
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}✗ $test_name FAILED${NC}"
        return 1
    fi
}

# Test 1: Normal checkpoint flow
echo ""
run_test "Test 1: Normal Checkpoint Flow" "test_stage3_checkpoint_normal.py" || true

echo ""
echo ""

# Test 2: Checkpoint write failure (graceful degradation)
run_test "Test 2: Checkpoint Write Failure" "test_stage3_checkpoint_failure.py" || true

echo ""
echo ""

# Test 3: Orchestrator fallback validation
run_test "Test 3: Orchestrator Fallback Logic" "test_orchestrator_fallback.py" || true

echo ""
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TEST SUITE SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"

if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
else
    echo "Failed:       0"
fi

echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Validated features:"
    echo "  ✓ Normal checkpoint creation and validation"
    echo "  ✓ Graceful degradation on checkpoint write failure"
    echo "  ✓ CSV fallback validation in orchestrator"
    echo "  ✓ Stage 3 continues successfully despite checkpoint failure"
    echo "  ✓ Pipeline doesn't break when checkpoint write fails"
    echo ""
    echo "The checkpoint fallback feature is working correctly!"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Please review the test output above for details."
    exit 1
fi
