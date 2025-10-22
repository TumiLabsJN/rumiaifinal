#!/usr/bin/env python3
"""
Simple Stage 7 Integration Test (No External Dependencies)

Tests the Stage 7 integration code by directly importing only the helper
functions without loading the entire rumiai_ml_batch module.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Test if we can at least compile the file
print("="*80)
print("STAGE 7 INTEGRATION - SIMPLE TEST")
print("="*80)
print()

print("Test 1: Python syntax validation...")
try:
    import py_compile
    py_compile.compile('/home/jorge/rumiaifinal/rumiai_ml_batch.py', doraise=True)
    print("✓ PASS: rumiai_ml_batch.py has valid Python syntax")
except SyntaxError as e:
    print(f"✗ FAIL: Syntax error in rumiai_ml_batch.py - {e}")
    sys.exit(1)

print("\nTest 2: Verify Stage 7 code sections exist...")

# Read the file and check for key sections
with open('/home/jorge/rumiaifinal/rumiai_ml_batch.py', 'r') as f:
    content = f.read()

checks = [
    ("Stage 7 import", "from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_llm_analysis_main"),
    ("validate_stage7_prerequisites function", "def validate_stage7_prerequisites(bucket_path: str, bucket: str)"),
    ("validate_stage7_outputs function", "def validate_stage7_outputs(bucket_path: str, bucket: str)"),
    ("handle_stage7_error function", "def handle_stage7_error(error: Exception, bucket_path: str)"),
    ("cleanup_stage7_partial_outputs function", "def cleanup_stage7_partial_outputs(bucket_path: str)"),
    ("Stage 7 orchestration code", "# ===== STAGE 7: LLM ANALYSIS - HYBRID TWO-PHASE APPROACH ====="),
    ("ANTHROPIC_API_KEY validation", "anthropic_api_key = os.getenv(\"ANTHROPIC_API_KEY\")"),
    ("Stage 7 execution call", "stage7_llm_analysis_main("),
    ("Stage 7 final status", "print(\"✓ Stage 7: LLM Analysis - COMPLETE\")"),
    ("Stage 7 in pipeline complete message", "PIPELINE EXECUTION COMPLETE (Stages 0-7)"),
]

all_passed = True
for check_name, check_string in checks:
    if check_string in content:
        print(f"✓ PASS: {check_name}")
    else:
        print(f"✗ FAIL: {check_name} - NOT FOUND")
        all_passed = False

print("\nTest 3: Verify Stage 7 integration structure...")

# Check that Stage 7 is in the right place (after Stage 6, before FINAL STATUS)
stage6_pos = content.find("logger.info(\"Stage 6 completed for all buckets\")")
stage7_pos = content.find("# ===== STAGE 7: LLM ANALYSIS - HYBRID TWO-PHASE APPROACH =====")
final_status_pos = content.find("# ===== FINAL STATUS =====")

if stage6_pos < stage7_pos < final_status_pos:
    print("✓ PASS: Stage 7 positioned correctly (after Stage 6, before FINAL STATUS)")
else:
    print(f"✗ FAIL: Stage 7 positioning incorrect")
    print(f"  Stage 6 pos: {stage6_pos}")
    print(f"  Stage 7 pos: {stage7_pos}")
    print(f"  Final Status pos: {final_status_pos}")
    all_passed = False

# Check helper functions are before main()
validate_prereq_pos = content.find("def validate_stage7_prerequisites")
main_func_pos = content.find("def main():")

if 0 < validate_prereq_pos < main_func_pos:
    print("✓ PASS: Helper functions positioned before main()")
else:
    print(f"✗ FAIL: Helper functions positioning incorrect")
    all_passed = False

print("\nTest 4: Verify error handling coverage...")

error_patterns = [
    ("FileNotFoundError handling", "except FileNotFoundError as e:"),
    ("ValueError handling", "except ValueError as e:"),
    ("RuntimeError handling", "except RuntimeError as e:"),
    ("IOError/OSError handling", "except (IOError, OSError) as e:"),
    ("Generic Exception handling", "except Exception as e:"),
]

for check_name, pattern in error_patterns:
    # Count occurrences in Stage 7 section
    stage7_section = content[stage7_pos:final_status_pos]
    if pattern in stage7_section:
        count = stage7_section.count(pattern)
        print(f"✓ PASS: {check_name} ({count} occurrence(s))")
    else:
        print(f"✗ FAIL: {check_name} - NOT FOUND in Stage 7 section")
        all_passed = False

print("\nTest 5: Verify Stage 7 summary logging...")

summary_checks = [
    ("stage7_summaries dict", "stage7_summaries = {}"),
    ("Stage 7 summary logging", "logger.info(\n                f\"Stage 7 Summary:"),
    ("Stage 7 JSON count", "total_llm_jsons = sum(s['json_files_generated']"),
]

for check_name, pattern in summary_checks:
    if pattern in content:
        print(f"✓ PASS: {check_name}")
    else:
        print(f"✗ FAIL: {check_name} - NOT FOUND")
        all_passed = False

print("\nTest 6: Verify bucket iteration pattern...")

# Check Stage 7 uses the same bucket iteration pattern as Stages 3-6
bucket_iteration = "for bucket_name in winning_buckets:"
stage7_section = content[stage7_pos:final_status_pos]

if bucket_iteration in stage7_section:
    print("✓ PASS: Bucket iteration pattern matches Stages 3-6")
else:
    print("✗ FAIL: Bucket iteration pattern not found")
    all_passed = False

print("\nTest 7: Verify configuration patterns...")

config_patterns = [
    ("Bucket path construction", "bucket_path = analysis_base / f\"buckets/bucket_{bucket_name}\"", True),  # Must be in Stage 7 section
    ("CLI args access", "hashtag=cli_args.target", True),  # Must be in Stage 7 section
    ("BUCKET_WINDOWS usage", "BUCKET_WINDOWS", False),  # Can be anywhere (used in helper functions)
]

for check_name, pattern, must_be_in_stage7 in config_patterns:
    if must_be_in_stage7:
        search_area = stage7_section
        location = "Stage 7 section"
    else:
        search_area = content
        location = "anywhere in file"

    if pattern in search_area:
        print(f"✓ PASS: {check_name} (found in {location})")
    else:
        print(f"✗ FAIL: {check_name} - NOT FOUND in {location}")
        all_passed = False

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

if all_passed:
    print("\n✅ All Stage 7 integration structure tests passed!")
    print()
    print("Note: These tests verify the integration code structure and syntax.")
    print("Full integration tests (with actual LLM API calls) require:")
    print("  1. Installing anthropic package: pip install anthropic")
    print("  2. Setting ANTHROPIC_API_KEY environment variable")
    print("  3. Running full pipeline: python3 rumiai_ml_batch.py --client test --target '#test'")
    sys.exit(0)
else:
    print("\n❌ Some tests failed - review integration code")
    sys.exit(1)
