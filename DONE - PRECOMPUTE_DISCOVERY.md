# PRECOMPUTE Flow - Complete Discovery Map
**Generated**: 2025-01-23
**Purpose**: Ensure safe removal of legacy PRECOMPUTE code without breaking TEMPORAL flow

## 🎯 Critical Question for Each Location:
**"Is this needed for rumiai_runner.py TEMPORAL flow?"**

---

## 📊 Discovery Results

### 1. PRECOMPUTE Files (5 files in /rumiai_v2/processors/)
```
✅ SAFE TO DELETE - Not used by temporal flow
- precompute_professional_wrappers.py
- precompute_functions.py
- precompute_professional.py
- precompute_creative_density.py
- precompute_functions_full.py
```

### 2. Production Code (rumiai_runner.py)
```python
# Line 31: IMPORTS BUT DOESN'T USE
from rumiai_v2.processors import (
    get_compute_function, COMPUTE_FUNCTIONS  # ← Imported but unused
)

# Lines 300-305: ALREADY COMMENTED OUT
# for func_name, func in COMPUTE_FUNCTIONS.items():  # ← Already disabled
```
**Status**: ✅ SAFE - Already disabled, temporal flow works without it

### 3. Local Video Runner (DELETE)
```python
# /scripts/local_video_runner.py - Lines 194-202
from rumiai_v2.processors import COMPUTE_FUNCTIONS
for func_name, func in COMPUTE_FUNCTIONS.items():  # ← ACTIVELY USES
    result = func(analysis_dict)
```
**Status**: ✅ CAN DELETE - Legacy test script, not needed for production
**Action**: Delete entire file (production uses rumiai_runner.py)

### 4. Test Files (DELETE)
```python
# test_python_only_e2e.py - Line 91
compute_fn = get_compute_function(prompt_type)  # ← Uses precompute

# test_ml_fixes.py
for name in COMPUTE_FUNCTIONS:  # ← References precompute

# Other test files that import MLServices (but not precompute directly):
- test_parallel.py (KEEP - tests parallel processing)
- test_single_video.py (KEEP - tests single video flow)
- test_bug_fixes.py (KEEP - tests bug fixes)
- test_unified_ml_pipeline.py (KEEP - tests unified pipeline)
- test_instrumentation.py (KEEP - tests instrumentation)
```
**Status**: ✅ CAN DELETE - test_python_only_e2e.py and test_ml_fixes.py are legacy tests
**Action**: Delete both test files (they test precompute, not temporal)

### 5. Configuration Files
```python
# /rumiai_v2/config/settings.py - Line ~50
'creative_density': True,  # HARDCODED

# /rumiai_v2/config/constants.py - Line ~30
'creative_density',  # In some list

# /rumiai_v2/validators/response_validator.py
'creative_density': { ... }  # Validation schema

# /rumiai_v2/processors/service_contracts.py
'compute_creative_density_analysis': 'density_analysis',
```
**Status**: ⚠️ LEGACY - Not used by temporal flow but may cause warnings
**Action**: Clean up after main removal

### 6. Import Chain (__init__.py)
```python
# /rumiai_v2/processors/__init__.py
from .precompute_functions import get_compute_function, COMPUTE_FUNCTIONS
```
**Status**: ⚠️ BREAKING - Exports these symbols
**Action**: Remove exports after updating importers

---

## 🔍 Deep Verification Commands

### Check if rumiai_runner.py actually uses COMPUTE_FUNCTIONS:
```bash
# 1. Check if COMPUTE_FUNCTIONS is ever called (not just imported)
grep -n "COMPUTE_FUNCTIONS\[" scripts/rumiai_runner.py
grep -n "get_compute_function(" scripts/rumiai_runner.py
# Result: NO ACTUAL USAGE (only commented out)

# 2. Verify temporal flow runs without it
python3 scripts/rumiai_runner.py "VIDEO_URL"  # Works fine
```

### Find all ACTUAL function calls (not just imports):
```bash
# Find where functions are CALLED, not just imported
grep -r "get_compute_function(" --include="*.py" | grep -v "^#"
grep -r "COMPUTE_FUNCTIONS\[" --include="*.py" | grep -v "^#"
grep -r "for.*in COMPUTE_FUNCTIONS" --include="*.py" | grep -v "^#"
```

---

## ✅ Safe Deletion Checklist

### Phase 1: Delete Obsolete Scripts (5 min)
- [ ] Delete `scripts/local_video_runner.py` (legacy test runner)
- [ ] Delete `test_python_only_e2e.py` (tests precompute flow)
- [ ] Delete `test_ml_fixes.py` (tests old ML fixes)

### Phase 2: Migrate Scene Detection (20 min)
- [ ] Copy scene_detection to UnifiedMLServices
- [ ] Update ml_services.py to inherit from UnifiedMLServices
- [ ] Test scene detection works

### Phase 3: Remove Core Imports & Files (10 min)
- [ ] Remove imports from `rumiai_runner.py` line 31
- [ ] Remove 'creative_density': 'density' mapping from `rumiai_runner.py` line ~300
- [ ] Remove exports from `processors/__init__.py`
- [ ] Delete 5 precompute*.py files from processors/
- [ ] Delete `scripts/verify_sync.py`
- [ ] Test: `python3 scripts/rumiai_runner.py VIDEO_URL`

### Phase 4: Complete Cleanup (20 min)
- [ ] Remove from config/settings.py ('creative_density': True)
- [ ] Remove from config/constants.py ('creative_density' in lists)
- [ ] Remove from validators/response_validator.py (creative_density schema)
- [ ] Remove from processors/service_contracts.py (compute_creative_density_analysis)
- [ ] Update core/error_handler.py (remove creative_density reference)
- [ ] Update scripts/compare_ml_results.py (remove/update creative_density logic)
- [ ] Delete prompt_templates/creative_density*.txt
- [ ] Delete prompt_templates/visual_hierarchy*.txt
- [ ] Delete prompt_templates/emotion_dynamics*.txt
- [ ] Final test: `python3 scripts/rumiai_runner.py VIDEO_URL`

---

## 🚨 Breaking Change Matrix

| Component | Uses PRECOMPUTE? | Needed for Temporal? | Safe to Delete? |
|-----------|-----------------|---------------------|-----------------|
| rumiai_runner.py | Imported only | NO | ✅ Remove import |
| temporal_compute.py | NO | - | ✅ Nothing to do |
| local_video_runner.py | YES (active) | NO | ✅ DELETE FILE |
| test_python_only_e2e.py | YES (active) | NO | ✅ DELETE FILE |
| test_ml_fixes.py | YES (active) | NO | ✅ DELETE FILE |
| ml_services.py | NO | NO | ✅ Keep (scene detection) |
| video_analyzer.py | NO | NO | ✅ Nothing to do |

---

## 🎯 Final Verification Test

After all deletions, run this test suite:
```bash
# 1. Main production flow
python3 scripts/rumiai_runner.py "TEST_VIDEO_URL"

# 2. Check temporal output exists
ls -la insights/*_temporal_windows_updated.json

# 3. Verify 50+ features
cat insights/*_temporal_windows_updated.json | jq '.temporal_windows.hook | keys | length'

# 4. Check no import errors
python3 -c "from rumiai_v2.processors import temporal_compute"
python3 -c "from rumiai_v2.processors import VideoAnalyzer"

# 5. Run any critical tests
pytest test_temporal_compute.py  # If exists
```

---

## 💡 Recommendation

**The complete removal approach**:
1. Delete the 3 scripts that use COMPUTE_FUNCTIONS (they're not needed)
2. Migrate scene detection to UnifiedMLServices
3. Remove all imports/exports and delete precompute files
4. Clean up ALL remaining references in configs and scripts

**Time estimate**: 55 minutes total
**Risk level**: LOW (systematic removal)
**Rollback**: Simple - just git revert if issues

**Result**: ZERO precompute references remaining in codebase (except .md docs)

---

## 📝 Notes
- Scene detection migration is SEPARATE from this cleanup
- ml_services.py deletion is SEPARATE (needs scene detection migration first)
- The temporal flow is ALREADY independent and working