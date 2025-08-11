#!/usr/bin/env python3
"""Analyze and categorize test files for cleanup"""

import os
import re

test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]

categories = {
    'python_only_critical': [],
    'ml_validation': [],
    'debug_scripts': [],
    'duplicates': []
}

for f in sorted(test_files):
    with open(f, 'r') as file:
        content = file.read()
        
    # Check characteristics
    has_main = 'if __name__' in content
    line_count = len(content.split('\n'))
    has_claude = 'claude' in content.lower()
    has_ml_services = 'MLServices' in content or 'ml_services' in content
    has_precompute = 'precompute' in content.lower() or 'compute_function' in content.lower()
    has_timeline = 'timeline' in content.lower()
    has_validation = 'validate' in content.lower() or 'assert' in content
    
    # Get description
    doc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    description = doc_match.group(1).strip()[:80] if doc_match else 'No description'
    
    # Categorize based on file name and content
    if 'python_only_e2e' in f:
        categories['python_only_critical'].append((f, line_count, "E2E test for Python-only flow"))
    elif 'unified_ml_pipeline' in f:
        categories['python_only_critical'].append((f, line_count, "Tests unified ML pipeline"))
    elif 'audio_energy' in f:
        categories['ml_validation'].append((f, line_count, "Tests audio energy service"))
    elif 'whisper' in f:
        categories['ml_validation'].append((f, line_count, "Tests whisper integration"))
    elif 'ml_fixes' in f:
        categories['ml_validation'].append((f, line_count, "Validates ML bug fixes"))
    elif 'p0_fixes' in f:
        categories['ml_validation'].append((f, line_count, "Tests P0 critical fixes"))
    elif 'extraction_fixed' in f:
        categories['duplicates'].append((f, line_count, "Duplicate of test_extraction.py"))
    elif 'extraction' in f:
        categories['debug_scripts'].append((f, line_count, "Debug timeline extraction"))
    elif 'failfast' in f:
        categories['debug_scripts'].append((f, line_count, "Debug fail-fast implementation"))
    elif 'analysis_structure' in f:
        categories['debug_scripts'].append((f, line_count, "Debug analysis.to_dict()"))
    elif 'helper' in f:
        categories['debug_scripts'].append((f, line_count, "Debug helper functions"))
    elif 'which_wrapper' in f:
        categories['debug_scripts'].append((f, line_count, "Debug wrapper selection"))
    elif 'format_defense' in f:
        categories['debug_scripts'].append((f, line_count, "Debug format handling"))
    else:
        categories['debug_scripts'].append((f, line_count, description))

print("=" * 70)
print("TEST FILE ANALYSIS FOR SMART CLEANUP")
print("=" * 70)

print("\n✅ KEEP - Critical Python-only Tests:")
keep_lines = 0
for f, lines, desc in categories['python_only_critical']:
    print(f"  • {f:30} ({lines:4} lines) - {desc}")
    keep_lines += lines

print("\n✅ KEEP - ML Validation Tests:")
for f, lines, desc in categories['ml_validation']:
    print(f"  • {f:30} ({lines:4} lines) - {desc}")
    keep_lines += lines

print("\n❌ DELETE - Debug/One-time Scripts:")
delete_lines = 0
for f, lines, desc in categories['debug_scripts']:
    print(f"  • {f:30} ({lines:4} lines) - {desc}")
    delete_lines += lines

print("\n❌ DELETE - Duplicates:")
for f, lines, desc in categories['duplicates']:
    print(f"  • {f:30} ({lines:4} lines) - {desc}")
    delete_lines += lines

print("\n" + "=" * 70)
total_keep = len(categories['python_only_critical']) + len(categories['ml_validation'])
total_delete = len(categories['debug_scripts']) + len(categories['duplicates'])
print(f"Summary: Keep {total_keep} files ({keep_lines} lines), Delete {total_delete} files ({delete_lines} lines)")
print(f"Space saved: {delete_lines} lines of test code")
print("=" * 70)