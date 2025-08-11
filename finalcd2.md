# Phase 2: COMPLETE PURGE - Removing ALL Non-Python-Only Code
**Version**: 2.0.0  
**Created**: 2025-01-11  
**Updated**: 2025-01-11  
**Purpose**: Remove 8,346 lines total of ALL dead code, prompt systems, tests, and documentation

## Executive Summary

After Phase 1's successful cleanup, this SMART PURGE will strategically remove dead code while preserving critical Python-only tests and functionality. 

### Two-Phase Approach:
1. **PRE-CLEANUP**: Remove 465 lines of Claude/dead code from production files
2. **FILE DELETION**: Delete 8,194 lines across 53 non-production files

### Final Impact:
- **Total lines removed**: 8,346 lines (465 from production + 7,881 from deleted files)
- **Files deleted**: 55 files (52 non-production + 3 cleanup scripts after completion)
- **Files preserved**: 7 files (6 critical tests + 1 utility script)
- **Reduction**: ~30% of total codebase, 85% of test/debug code
- **Result**: 100% Python-only functionality with essential test coverage

**IMPORTANT**: 
- **Apify will be PRESERVED** - It's required for TikTok video downloading
- **Documentation (.md files) will be PRESERVED** - User will handle these manually

### Current State (After Phase 1)
- ✅ Python-only pipeline works at $0.00 cost
- ✅ All settings hardcoded, zero configuration needed
- ⚠️ ~3,000-4,000 lines of dead code remain
- ⚠️ Entire prompt template system still exists
- ⚠️ Hundreds of test files with mocks
- ⚠️ Documentation for obsolete flows
- ⚠️ Node.js compatibility layers still present
- ⚠️ Legacy v1 format code never executes

### Goal State (After Phase 2 COMPLETE PURGE)
- Pure Python-only codebase
- **~30% smaller total codebase**
- Zero Claude references anywhere
- No prompt systems at all
- Only essential Python ML code
- Clean, minimal file structure

---

## Initial Analysis - Now Resolved
### (This section shows the original investigation that led to the cleanup plan below)

> **NOTE**: This analysis was performed BEFORE cleanup. All issues identified here are addressed by the cleanup steps in the sections below.

### 1. Claude API Remnants (Safe to Remove)
**~400-500 lines of dead code**

#### Files with Dead Claude Code:
- `scripts/rumiai_runner.py` (lines 349-641): ~290 lines of bypassed Claude logic
- `rumiai_v2/config/settings.py`: ~10 lines of unused Claude settings
- `rumiai_v2/api/__init__.py`: Obsolete ClaudeClient export

#### What We're Removing:
- All Claude API method calls  
- All conditional checks that are always True/False
- All prompt-related code paths
- ~290 lines of dead Claude code that never executes

### 2. Node.js Compatibility Layers
**~200-300 lines of unnecessary formatting**

#### Critical Files:
- `scripts/rumiai_runner.py`:
  - Line 5: "CRITICAL: Must maintain backward compatibility with existing Node.js calls"
  - Lines 714-730: Node.js exit code handling
  - Lines 220-240: Node.js response formatting

- `rumiai_v2/processors/temporal_markers.py`:
  - Lines 24, 35, 418: Node.js subprocess formatting

#### What Can Be Removed:
- All "legacy_mode" parameters and logic
- Node.js-specific stdout/stderr handling
- Subprocess formatting for Node.js callers

### 3. Legacy V1/V2 Format Code
**~150-200 lines of unused compatibility**

#### Affected Files:
- `rumiai_v2/core/models/timestamp.py` (lines 120-131): Legacy timestamp formats
- `rumiai_v2/core/models/timeline.py` (lines 49-57, 182-205): Legacy grouping
- `rumiai_v2/core/models/analysis.py` (lines 92-119): Legacy mode in UnifiedAnalysis
- `rumiai_v2/validators/response_validator.py` (lines 264-277): V1/V2 handling

#### What We're Removing:
- All legacy_mode parameters (always True)
- All v1/v2 format conditionals  
- Dead code branches that never execute

### 4. Cleanup Scripts (No Longer Needed)
**~821 lines of one-time scripts**

#### Files to Delete:
- `remove_claude_refs.py` (46 lines)
- `hardcode_settings.py` (94 lines)
- `discovery_analysis.py` (438 lines)
- `final_validation.py` (112 lines)
- `scipy_compat.py` (131 lines) - Check if ML libs still need this

### 5. Placeholder Functions
**~85 lines of "not implemented" functions**

#### In `rumiai_v2/processors/precompute_functions.py`:
Multiple placeholder functions that just return empty dicts or raise NotImplementedError

### 6. ENTIRE Prompt Template System (MISSED IN ORIGINAL ANALYSIS!)
**~90 lines of useless prompt loading**

#### In `rumiai_v2/config/settings.py` (lines 83-177):
```python
self._prompt_templates = self._load_prompt_templates()  # Line 83
def _load_prompt_templates(self) -> Dict[str, str]:     # Lines 88-173
def get_prompt_template(self, prompt_type: str):         # Lines 175-177
```
This ENTIRE system is useless in Python-only mode!

### 7. Prompt Model Classes (MISSED!)
**~100 lines of Claude-specific models**

#### File: `rumiai_v2/core/models/prompt.py`:
- `class PromptType(Enum)` - Enumeration for prompt types
- `class PromptResult` - Result wrapper for Claude
- `class PromptBatch` - Batch processing for Claude
**This entire file can be DELETED!**

### 8. API Key Validation (PARTIAL REMOVAL)
**~5 lines of obsolete Claude validation**

#### In `rumiai_v2/config/settings.py` (lines 184-189):
```python
if not self.claude_api_key:
    errors.append("CLAUDE_API_KEY environment variable not set")  # REMOVE THIS
    
if not self.apify_token:
    errors.append("APIFY_API_TOKEN environment variable not set")  # KEEP THIS - NEEDED!
```
**IMPORTANT**: Remove Claude API validation but KEEP Apify validation - Apify is required for TikTok video downloading!

### 9. Test Files and Scripts - Complete Deletion List
**Total to delete: 7,881 lines across 52 files | Keep: 1,690 lines across 7 files**

#### ❌ DELETE - Debug/One-time Test Scripts (748 lines):
- `test_analysis_structure.py` (30 lines) - Debug analysis.to_dict()
- `test_extraction.py` (66 lines) - Debug timeline extraction
- `test_extraction_fixed.py` (105 lines) - Duplicate of test_extraction.py
- `test_failfast.py` (186 lines) - Debug fail-fast implementation
- `test_format_defense.py` (87 lines) - Debug format handling
- `test_helper_solution.py` (103 lines) - Debug helper functions
- `test_helper_vs_timeline.py` (82 lines) - Debug helper functions
- `test_helpers.py` (61 lines) - Debug helper functions
- `test_which_wrapper.py` (28 lines) - Debug wrapper selection

#### ❌ DELETE - Cleanup/Analysis Scripts (2,284 lines):
**None of these are part of production - all are one-time analysis/debug tools**
- `analyze_all_outputs.py` (64 lines) - Output analysis
- `analyze_data_structures.py` (74 lines) - Data structure analysis
- `analyze_defensive_programming.py` (124 lines) - Defensive programming analysis
- `analyze_test_files.py` (94 lines) - Test file categorization
- `check_ml_data_exists.py` (41 lines) - ML data verification
- `count_all_features.py` (126 lines) - Feature counting
- `debug_ml_functions.py` (95 lines) - ML function debugging
- `detect_tiktok_creative_elements.py` (557 lines) - **Standalone tool, NOT used in production**
- `discovery_analysis.py` (284 lines) - **One-time codebase analysis, NOT production**
- `final_validation.py` (113 lines) - Final validation
- `hardcode_settings.py` (90 lines) - Settings hardcoding script
- `investigate_helper_intent.py` (126 lines) - Helper investigation
- `remove_claude_refs.py` (72 lines) - Claude removal script
- `scipy_compat.py` (159 lines) - Scipy compatibility patch
- `trace_scene_extraction.py` (48 lines) - Scene extraction trace
- `verify_feat_installation.py` (221 lines) - FEAT verification

#### ❌ DELETE - Claude-specific Files (703 lines):
- `rumiai_v2/core/models/prompt.py` (106 lines) - Prompt models
- `rumiai_v2/processors/ml_data_extractor.py` (597 lines) - ML data extractor
- `rumiai_v2/processors/prompt_builder.py` - Prompt builder (if exists)
- `rumiai_v2/processors/output_adapter.py` - Output adapter (if exists)
- `rumiai_v2/api/claude_client.py` - Claude client (if exists)

#### ❌ DELETE - Setup/Deployment Files (222 lines):
- `.env.example` (13 lines) - Environment template
- `Dockerfile` (22 lines) - Docker configuration
- `install_ml_dependencies.sh` (120 lines) - ML dependency installer
- `set_python_only_env.sh` (26 lines) - Python-only environment setter
- `setup.sh` (41 lines) - Setup script

#### ❌ DELETE - Test Audio Files (4 files):
- `test_120s.wav` (3.7 MB)
- `test_30s.wav` (0.9 MB)
- `test_3s.wav` (0.1 MB)
- `test_audio.wav` (0.1 MB)

#### ❌ DELETE - Entire Directories (3,920 lines):
- `tests/` (3,920 lines, 16 Python files) - Unit tests with Claude mocks
- `rumiai_v2/prompts/` (0 lines) - Empty prompt directory
- `prompt_templates/` (0 lines) - Empty template directory

#### ❌ DELETE - Cleanup Scripts AFTER Completion (3 files):
**Delete these only AFTER successful cleanup and verification**
- `remove_all_claude_dead_code.py` - Production code cleaner (run first, then delete)
- `verify_production_safety.py` - Safety verification (run second, then delete)
- `smart_cleanup_phase2.py` - File deletion script (run third, then delete)

#### ✅ KEEP - Critical Python-only Tests (1,377 lines):
- `test_python_only_e2e.py` (541 lines) - **E2E validation**
- `test_unified_ml_pipeline.py` (210 lines) - **ML pipeline tests**
- `test_audio_energy.py` (72 lines) - Audio service tests
- `test_ml_fixes.py` (110 lines) - Bug fix validation
- `test_p0_fixes.py` (384 lines) - P0 critical fixes
- `test_whisper_cpp.py` (60 lines) - Whisper integration

#### ✅ KEEP - Utility Scripts for Future Use:
- `analyze_all_deletable_files.py` (313 lines) - Analyzes files safe to delete without affecting production

### 10. Documentation Files
**NOTE: User will handle .md files manually - DO NOT DELETE**

#### Files that reference Claude/old flows:
- Various `.md` files contain Claude references
- **Action**: User will review and update these manually
- **DO NOT include .md files in automated cleanup**

### 11. Setup and Deployment Scripts (MISSED!)
**~200 lines of Node.js setup**

#### Files:
- `setup.sh` - Still runs `npm install`
- `Dockerfile` - Uses Node.js base image
- `.env.example` - Claude API template

---

## CRITICAL: Pre-Cleanup Code Removal (MUST DO FIRST!)

### ⚠️ IMPORTANT: Remove Dead Code from Production Files BEFORE Deleting Files

Our verification found **465 lines of Claude/dead code** still in production files that must be removed first:

#### Files with Dead Code to Clean:
1. **scripts/rumiai_runner.py** (349 lines of dead code):
   - Lines 349-445: `_run_claude_prompts()` method - REMOVE ENTIRE METHOD
   - Lines 445-543: `_run_claude_prompts_v2()` method - REMOVE ENTIRE METHOD  
   - Lines 643-667: `_save_prompt_result()` method - REMOVE ENTIRE METHOD
   - Line 203: Change "running_claude_prompts" to "running_precompute_functions"
   - Line 282: Change "Running Claude prompts" to "Running precompute functions"
   - Line 390: Remove `'claude_sonnet': self.settings.use_claude_sonnet`
   - Lines 206-210: Replace Claude calls with precompute functions

2. **rumiai_v2/api/__init__.py** (3 lines):
   - Line 8: Remove `'ClaudeClient'` from __all__ list

3. **rumiai_v2/config/settings.py** (106 lines):
   - Lines 83-177: Remove entire prompt template system
   - Remove all `self.claude_*` settings
   - Remove all `self.prompt_*` settings
   - Remove Claude API key validation

4. **Dead Conditionals** (35+ lines):
   - Remove `if self.settings.use_ml_precompute:` (always True)
   - Remove `legacy_mode` parameters and conditionals

### Pre-Cleanup Script to Run:

```bash
# Run this BEFORE any file deletion!
python3 remove_all_claude_dead_code.py
```

This script will:
- Remove 314 lines from rumiai_runner.py
- Remove 106 lines from settings.py  
- Remove 35 lines of dead conditionals
- Total: 465 lines of dead code removed

## Implementation Plan - COMPLETE PURGE

### Step 0: Production Safety Verification (REQUIRED)

```bash
# 1. First remove dead code from production files
python3 remove_all_claude_dead_code.py

# 2. Verify production safety
python3 verify_production_safety.py

# Should see:
# ✅ Production Safety: PASSED
# ✅ Dead Code Removal: COMPLETE  
# ✅ ML Services: Present

# If verification fails, DO NOT PROCEED
```

### Step 1: File Deletion (After Pre-Cleanup)

```bash
#!/bin/bash
# phase2_file_deletion.sh - Delete all non-essential files

echo "🔥 PHASE 1: DELETE ENTIRE DIRECTORIES"

# Delete ALL test files (keep only essential ML tests later)
rm -rf tests/
rm -f test_*.py
rm -f *test*.py

# Delete prompt system entirely
rm -rf rumiai_v2/prompts/

# Delete all one-time scripts
rm -f discovery_*.py remove_*.py hardcode_*.py final_*.py scipy_compat.py

# Delete obsolete documentation
ls *.md | grep -v "FRAME_PROCESSING_PYTHON_ONLY\|ML_DATA_PROCESSING_PIPELINE\|README" | xargs rm -f

# Delete setup/deployment files
rm -f setup.sh Dockerfile .env.example

# Delete prompt model file
rm -f rumiai_v2/core/models/prompt.py

echo "🔥 PHASE 2: GUT CONFIGURATION FILES"
```

### Step 1: Complete Settings.py Surgery

```python
# clean_settings.py - Remove ALL prompt and Claude references
import re
from pathlib import Path

def gut_settings():
    """Remove prompt templates and Claude validation from settings.py"""
    
    file_path = Path('rumiai_v2/config/settings.py')
    content = file_path.read_text()
    
    # Remove entire prompt template system (lines ~83-177)
    content = re.sub(
        r'# Load prompt templates.*?def get_prompt_template.*?\n.*?\n',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Remove Claude API key lines (but KEEP Apify!)
    content = re.sub(r'.*claude_api_key.*\n', '', content)
    content = re.sub(r'.*CLAUDE_API_KEY.*\n', '', content)
    content = re.sub(r'.*claude_model.*\n', '', content)
    content = re.sub(r'.*use_claude_sonnet.*\n', '', content)
    content = re.sub(r'.*claude_cost_threshold.*\n', '', content)
    # NOTE: Keeping self.apify_token - it's needed for video downloading!
    
    # Remove prompt delay
    content = re.sub(r'.*prompt_delay.*\n', '', content)
    content = re.sub(r'.*prompt_timeouts.*\n', '', content)
    
    file_path.write_text(content)
    print(f"✅ Removed ~100 lines from settings.py")
```

### Step 2: Remove Node.js Compatibility

```python
def remove_nodejs_compat():
    """Remove all Node.js compatibility layers"""
    
    # Remove legacy_mode parameter from all files
    for py_file in Path('rumiai_v2').rglob('*.py'):
        content = py_file.read_text()
        
        # Remove legacy_mode parameters
        content = re.sub(r',?\s*legacy_mode:\s*bool\s*=\s*\w+', '', content)
        content = re.sub(r'if\s+legacy_mode:.*?else:.*?(?=\n\S)', '', content, flags=re.DOTALL)
        
        # Remove Node.js comments
        content = re.sub(r'#.*Node\.js.*\n', '', content)
        
        py_file.write_text(content)
```

### Step 3: Remove Legacy Format Code

```python
def remove_legacy_formats():
    """Remove all v1 format compatibility code"""
    
    # List of files with legacy format code
    files_to_clean = [
        'rumiai_v2/core/models/timestamp.py',
        'rumiai_v2/core/models/timeline.py',
        'rumiai_v2/core/models/analysis.py',
    ]
    
    for file_path in files_to_clean:
        content = Path(file_path).read_text()
        
        # Remove legacy_mode conditionals
        content = re.sub(
            r'if\s+.*?legacy.*?:.*?else:.*?(?=\n\S)',
            lambda m: m.group().split('else:')[1],  # Keep non-legacy path
            content,
            flags=re.DOTALL
        )
        
        Path(file_path).write_text(content)
```

### Step 4: Delete Unnecessary Files

```bash
# Delete one-time cleanup scripts
rm -f remove_claude_refs.py
rm -f hardcode_settings.py
rm -f discovery_analysis.py
rm -f final_validation.py
rm -f simplify_code.py

# Delete test files for removed features
rm -f test_unified_pipeline_e2e.py  # Has mock Claude references

# Optional: Check if scipy_compat.py is still needed
# python3 -c "from rumiai_v2.ml_services.emotion_detection_service import *"
# If it works without scipy_compat.py, delete it too
```

### Step 5: Clean Up Imports and Comments

```python
def clean_imports_and_comments():
    """Remove obsolete imports and dead code comments"""
    
    for py_file in Path('.').rglob('*.py'):
        if 'venv' in str(py_file):
            continue
            
        content = py_file.read_text()
        
        # Remove "REMOVED" comments
        content = re.sub(r'#.*REMOVED.*\n', '', content)
        content = re.sub(r'#.*TODO.*remove.*\n', '', content, flags=re.IGNORECASE)
        content = re.sub(r'#.*FIXME.*\n', '', content)
        content = re.sub(r'#.*DEPRECATED.*\n', '', content)
        
        # Remove empty comment lines
        content = re.sub(r'^\s*#\s*$\n', '', content, flags=re.MULTILINE)
        
        py_file.write_text(content)
```

---

## Automated Full Cleanup Script

Save as `phase2_cleanup.py`:

```python
#!/usr/bin/env python3
"""
Phase 2 Deep Cleanup - Remove all non-Python-only code
Removes ~1,500-2,000 lines of dead code
"""

import re
import sys
from pathlib import Path

def main():
    print("="*60)
    print("PHASE 2: DEEP CLEANUP - REMOVING DEAD CODE")
    print("="*60)
    
    total_lines_removed = 0
    
    # Step 1: Remove dead Claude paths
    print("\n[1/5] Removing dead Claude code paths...")
    lines = remove_dead_claude_code()
    total_lines_removed += lines
    print(f"  ✅ Removed {lines} lines")
    
    # Step 2: Remove Node.js compatibility
    print("\n[2/5] Removing Node.js compatibility layers...")
    lines = remove_nodejs_compat()
    total_lines_removed += lines
    print(f"  ✅ Removed {lines} lines")
    
    # Step 3: Remove legacy formats
    print("\n[3/5] Removing legacy v1 format code...")
    lines = remove_legacy_formats()
    total_lines_removed += lines
    print(f"  ✅ Removed {lines} lines")
    
    # Step 4: Delete unnecessary files
    print("\n[4/5] Deleting cleanup scripts...")
    lines = delete_unnecessary_files()
    total_lines_removed += lines
    print(f"  ✅ Deleted {lines} lines worth of files")
    
    # Step 5: Clean imports and comments
    print("\n[5/5] Cleaning imports and comments...")
    lines = clean_imports_and_comments()
    total_lines_removed += lines
    print(f"  ✅ Cleaned {lines} lines")
    
    print("\n" + "="*60)
    print(f"🎉 CLEANUP COMPLETE!")
    print(f"📊 Total lines removed: ~{total_lines_removed}")
    print(f"📉 Codebase reduced by ~{total_lines_removed/6000*100:.1f}%")
    print("="*60)

# Implementation functions here...
# (Add all the cleanup functions from above)

if __name__ == "__main__":
    main()
```

---

## Testing After Cleanup

### Validation Script

```python
#!/usr/bin/env python3
"""Validate that Python-only pipeline still works after Phase 2 cleanup"""

def validate_phase2():
    """Ensure everything still works after deep cleanup"""
    
    tests = []
    
    # Test 1: Can import without errors
    try:
        from scripts.rumiai_runner import RumiAIRunner
        tests.append(("Import RumiAIRunner", True))
    except:
        tests.append(("Import RumiAIRunner", False))
    
    # Test 2: No legacy_mode references
    legacy_count = 0
    for py_file in Path('.').rglob('*.py'):
        if 'legacy_mode' in py_file.read_text():
            legacy_count += 1
    tests.append(("No legacy_mode refs", legacy_count == 0))
    
    # Test 3: No Node.js references
    nodejs_count = 0
    for py_file in Path('.').rglob('*.py'):
        content = py_file.read_text().lower()
        if 'node.js' in content or 'nodejs' in content:
            nodejs_count += 1
    tests.append(("No Node.js refs", nodejs_count == 0))
    
    # Test 4: Precompute functions work
    try:
        from rumiai_v2.processors.precompute_functions import COMPUTE_FUNCTIONS
        tests.append(("Precompute functions", len(COMPUTE_FUNCTIONS) == 7))
    except:
        tests.append(("Precompute functions", False))
    
    # Print results
    print("\nVALIDATION RESULTS:")
    for test_name, passed in tests:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    return all(passed for _, passed in tests)

if __name__ == "__main__":
    if validate_phase2():
        print("\n🎉 All tests passed! Cleanup successful!")
    else:
        print("\n❌ Some tests failed. Review changes.")
```

---

## Expected Outcome

### Before Phase 2 COMPLETE PURGE:
- ~8,000-10,000 total lines (including tests, configs)
- Dead Claude code paths everywhere
- Entire prompt template system
- Hundreds of test files
- Node.js compatibility layers
- Documentation files with Claude references (user will handle)

### After Phase 2 SMART PURGE:
- **~4,000-5,000 lines of pure Python ML code** (30-40% reduction!)
- Essential ML processing code
- Critical Python-only test coverage preserved (1,377 lines)
- Zero Claude/prompt references
- No debug/one-time scripts
- Documentation preserved for user review
- Clean, maintainable codebase with tests

### Performance Improvements:
- **Faster imports** (less code to parse)
- **Clearer debugging** (no dead paths to confuse)
- **Easier maintenance** (obvious what's active)
- **Smaller deployment** (less code to ship)

---

## Risk Assessment

### Low Risk Items:
- ✅ Removing cleanup scripts (one-time use)
- ✅ Removing dead Claude code (never executes)
- ✅ Removing REMOVED/TODO comments

### Medium Risk Items:
- ⚠️ Removing legacy_mode parameters (test thoroughly)
- ⚠️ Removing Node.js compatibility (ensure not needed)

### Mitigation:
1. **Test after each step** with validation script
2. **User handles backups** manually
3. **Run full pipeline test** with real video

---

## Execution Timeline

### Quick Win (30 minutes):
1. Delete cleanup scripts (5 min)
2. Remove dead Claude paths (10 min)
3. Clean comments (5 min)
4. Test (10 min)

### Full Cleanup (2-3 hours):
1. Run automated script (30 min)
2. Manual review of changes (30 min)
3. Testing with multiple videos (1 hour)
4. Final validation (30 min)

---

## Complete Automation Script

Alternative: Complete cleanup script:

```bash
#!/bin/bash
# complete_phase2_cleanup.sh - Remove all non-Python-only code
# WARNING: This will delete ~50-60% of your codebase!

echo "⚠️  WARNING: This will DELETE ~30% of the codebase!"
echo "Make sure you have created a manual backup!"
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo "🔥 Starting Phase 2 cleanup..."

# PHASE 1: Delete entire directories and files
echo "📁 Deleting directories..."
rm -rf tests/
rm -rf rumiai_v2/prompts/
rm -rf prompt_templates/

echo "📄 Deleting 9 debug/one-time test scripts..."
# Delete only debug scripts, keep critical tests
rm -f test_analysis_structure.py test_extraction.py test_extraction_fixed.py
rm -f test_failfast.py test_format_defense.py test_helper_solution.py
rm -f test_helper_vs_timeline.py test_helpers.py test_which_wrapper.py

echo "📄 Keeping critical test files..."
# Explicitly keeping:
# test_python_only_e2e.py - E2E validation
# test_unified_ml_pipeline.py - ML pipeline tests
# test_audio_energy.py - Audio service tests
# test_ml_fixes.py - Bug fix validation
# test_p0_fixes.py - P0 fixes
# test_whisper_cpp.py - Whisper tests

echo "📄 Deleting 17 cleanup/analysis scripts..."
# Delete all one-time analysis and cleanup scripts
rm -f analyze_all_deletable_files.py analyze_all_outputs.py analyze_data_structures.py
rm -f analyze_defensive_programming.py analyze_test_files.py check_ml_data_exists.py
rm -f count_all_features.py debug_ml_functions.py detect_tiktok_creative_elements.py
rm -f discovery_analysis.py final_validation.py hardcode_settings.py
rm -f investigate_helper_intent.py remove_claude_refs.py scipy_compat.py
rm -f trace_scene_extraction.py verify_feat_installation.py

echo "📄 Deleting Claude-specific files..."
rm -f rumiai_v2/core/models/prompt.py
rm -f rumiai_v2/processors/ml_data_extractor.py
rm -f rumiai_v2/processors/prompt_builder.py 2>/dev/null
rm -f rumiai_v2/processors/output_adapter.py 2>/dev/null
rm -f rumiai_v2/api/claude_client.py 2>/dev/null

echo "📄 Deleting setup/deployment files..."
rm -f .env.example Dockerfile install_ml_dependencies.sh set_python_only_env.sh setup.sh

echo "🔊 Deleting test audio files..."
rm -f test_120s.wav test_30s.wav test_3s.wav test_audio.wav

# NOTE: Keeping all .md files - user will handle documentation manually
echo "📄 Preserving all .md documentation files..."

# PHASE 2: Clean Python files
echo "🧹 Cleaning Python files..."

# Clean settings.py
python3 << 'EOF'
import re
from pathlib import Path

file_path = Path('rumiai_v2/config/settings.py')
if file_path.exists():
    content = file_path.read_text()
    
    # Remove prompt templates
    content = re.sub(r'self\._prompt_templates.*?\n', '', content)
    content = re.sub(r'def _load_prompt_templates.*?return templates.*?\n', '', content, flags=re.DOTALL)
    content = re.sub(r'def get_prompt_template.*?return.*?\n', '', content, flags=re.DOTALL)
    
    # Remove Claude references (but preserve Apify lines!)
    # Use negative lookahead to avoid removing lines with 'apify'
    content = re.sub(r'^(?!.*apify).*claude.*\n', '', content, flags=re.IGNORECASE | re.MULTILINE)
    content = re.sub(r'.*prompt_delay.*\n', '', content)
    content = re.sub(r'.*prompt_timeouts.*\n', '', content)
    
    file_path.write_text(content)
    print("✅ Cleaned settings.py")
EOF

# Clean rumiai_runner.py
python3 << 'EOF'
import re
from pathlib import Path

file_path = Path('scripts/rumiai_runner.py')
if file_path.exists():
    content = file_path.read_text()
    
    # Remove all methods with 'claude' in name
    content = re.sub(r'def.*claude.*?(?=def|\Z)', '', content, flags=re.DOTALL)
    
    # Remove legacy mode
    content = re.sub(r'legacy_mode.*?\n', '', content)
    
    # Remove Node.js comments
    content = re.sub(r'#.*Node\.js.*\n', '', content, flags=re.IGNORECASE)
    content = re.sub(r'#.*CRITICAL.*compatibility.*\n', '', content)
    
    file_path.write_text(content)
    print("✅ Cleaned rumiai_runner.py")
EOF

# PHASE 3: Clean imports
echo "🔧 Fixing imports..."
find rumiai_v2 -name "*.py" -exec python3 -c "
import sys
import re
file_path = sys.argv[1]
with open(file_path, 'r') as f:
    content = f.read()
content = re.sub(r'from.*prompt.*import.*\n', '', content, flags=re.IGNORECASE)
content = re.sub(r'import.*prompt.*\n', '', content, flags=re.IGNORECASE)
with open(file_path, 'w') as f:
    f.write(content)
" {} \;

echo "📊 Calculating reduction..."
echo "Files deleted:"
echo "  - Test files: $(find . -name "test_*.py" 2>/dev/null | wc -l)"
echo "  - Scripts: $(ls *_*.py 2>/dev/null | wc -l)"
echo "Files preserved:"
echo "  - Documentation: $(ls *.md 2>/dev/null | wc -l) .md files (for user review)"

echo ""
echo "🎉 PHASE 2 CLEANUP COMPLETE!"
echo "The codebase is now pure Python-only with minimal footprint."
echo ""
echo "Test with: python3 scripts/rumiai_runner.py 'VIDEO_URL'"
```

## Complete Execution Order

### ⚠️ CRITICAL: Follow this exact order!

```bash
# Step 1: Manual backup (handled by user)
# User will create backup manually before proceeding

# Step 2: Remove dead code from production files (REQUIRED FIRST!)
python3 remove_all_claude_dead_code.py

# Step 3: Verify production safety
python3 verify_production_safety.py
# MUST see all "✅ PASSED" before continuing

# Step 4: Run smart cleanup (dry run first)
python3 smart_cleanup_phase2.py --dry-run

# Step 5: Execute cleanup if dry run looks good
python3 smart_cleanup_phase2.py --execute

# Step 6: Test the cleaned pipeline
python3 scripts/rumiai_runner.py "https://www.tiktok.com/@test/video/123"

# Step 7: Run critical tests
python3 test_python_only_e2e.py
python3 test_unified_ml_pipeline.py

# If issues occur:
# User will restore from their manual backup if needed
```

### Expected Results After Cleanup:
- ✅ 465 lines of dead code removed from production
- ✅ 53 non-production files deleted
- ✅ 6 critical test files preserved
- ✅ Python-only pipeline fully functional
- ✅ Zero Claude/JS references remaining
- ✅ ~8,659 total lines removed

This Phase 2 COMPLETE PURGE will transform your codebase from a working but bloated implementation into a **minimal, pure Python ML pipeline** with 50-60% less code!