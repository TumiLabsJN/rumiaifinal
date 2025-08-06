#!/usr/bin/env python3
"""Test which compute_creative_density_wrapper is actually used"""

import sys
from pathlib import Path

# Add rumiai_v2 to path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.precompute_functions import (
    COMPUTE_FUNCTIONS,
    compute_creative_density_wrapper
)

# Check which function is in COMPUTE_FUNCTIONS
func_in_dict = COMPUTE_FUNCTIONS['creative_density']

print(f"Function in COMPUTE_FUNCTIONS: {func_in_dict}")
print(f"Function code object: {func_in_dict.__code__}")
print(f"Function line number: {func_in_dict.__code__.co_firstlineno}")
print(f"Function var names: {func_in_dict.__code__.co_varnames[:5]}")

# The first one (line 96) takes 'analysis_data'
# The second one (line 366) takes 'analysis_dict'
if 'analysis_data' in func_in_dict.__code__.co_varnames:
    print("\n✗ Using FIRST definition (line 96) with helpers - but this is wrong input!")
elif 'analysis_dict' in func_in_dict.__code__.co_varnames:
    print("\n✗ Using SECOND definition (line 366) with _extract_timelines_from_analysis")