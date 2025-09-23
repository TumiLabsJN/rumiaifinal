# TemporalOnly.md - Legacy PRECOMPUTE Removal Roadmap

## 🎯 Objective
Complete removal of all legacy PRECOMPUTE flow code while preserving the integrity of the TEMPORAL flow pipeline.

## 📊 Current State Assessment (UPDATED 2025-01-23)

### ✅ Already Completed:
- Precompute functions removed from `ml_services.py` (reduced to 164 lines)
- COMPUTE_FUNCTIONS already commented out in `rumiai_runner.py` (line 302)
- **rumiai_runner.py production flow is clean** - only imports but never uses COMPUTE_FUNCTIONS

### ⚠️ Still Remaining:
- **5 precompute files** in `/rumiai_v2/processors/`:
  - precompute_professional_wrappers.py
  - precompute_functions.py
  - precompute_professional.py
  - precompute_creative_density.py
  - precompute_functions_full.py
- **3 scripts actively use COMPUTE_FUNCTIONS**:
  - local_video_runner.py (line 197)
  - test_python_only_e2e.py (line 91)
  - test_ml_fixes.py
- Scene detection needs migration to UnifiedMLServices
- Imports and exports need cleanup

## ✅ Critical Discovery: Production is Safe!

**rumiai_runner.py (production) verification**:
```python
# Line 31: Imports but NEVER uses
from rumiai_v2.processors import (
    get_compute_function, COMPUTE_FUNCTIONS  # ← Imported but unused
)

# Lines 300-305: Already commented out
# for func_name, func in COMPUTE_FUNCTIONS.items():  # ← Already disabled
```

✅ **Temporal flow works without any precompute dependencies**

---

## 🗑️ Simplified Removal Roadmap

### Phase 1: Delete Scripts Using COMPUTE_FUNCTIONS (5 minutes)
**Priority**: SIMPLIFIED ✅ - These scripts are not needed

#### 1.1 Delete obsolete scripts
```bash
# These scripts are legacy and not part of production flow:
rm scripts/local_video_runner.py     # Old test runner - not needed
rm test_python_only_e2e.py          # Old test - not needed
rm test_ml_fixes.py                 # Old test - not needed
```

#### 1.2 Justification
- **local_video_runner.py**: Legacy test script, production uses rumiai_runner.py
- **test_python_only_e2e.py**: Tests precompute flow, not temporal flow
- **test_ml_fixes.py**: Tests old ML fixes for precompute

#### 1.3 No validation needed
- These are not production scripts
- rumiai_runner.py is the only production entry point

---

### Phase 2: Scene Detection Migration (20 minutes)
**Priority**: MEDIUM 🟡 - Required before deleting ml_services.py

#### 2.1 Copy scene_detection to UnifiedMLServices
```python
# From: rumiai_v2/api/ml_services.py (lines 111-161)
# To: rumiai_v2/api/ml_services_unified.py

# Add to UnifiedMLServices class:
async def run_scene_detection(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
    # [Copy implementation from ml_services.py]
```

#### 2.2 Update ml_services.py to delegate
```python
# Make MLServices inherit from UnifiedMLServices
from .ml_services_unified import UnifiedMLServices

class MLServices(UnifiedMLServices):
    """Compatibility wrapper"""
    pass
```

#### 2.3 Test scene detection
```bash
python3 -c "
from rumiai_v2.api.ml_services_unified import UnifiedMLServices
# Verify scene detection available
"
```

---

### Phase 3: Remove Imports & Files (10 minutes)
**Priority**: LOW 🟢 - Safe after Phase 1 & 2

#### 3.1 Clean rumiai_runner.py
```python
# Line 31: Remove imports
# get_compute_function, COMPUTE_FUNCTIONS  ← Delete these

# Line ~300: Remove mapping
# 'creative_density': 'density',  ← Delete this line
```

#### 3.2 Clean processors/__init__.py
```python
# Remove:
from .precompute_functions import get_compute_function, COMPUTE_FUNCTIONS
# Remove from __all__ list
```

#### 3.3 Delete precompute files
```bash
rm rumiai_v2/processors/precompute_*.py
```

#### 3.4 Delete verify_sync.py
```bash
rm scripts/verify_sync.py  # Uses COMPUTE_FUNCTIONS
```

#### 3.5 Final test
```bash
python3 scripts/rumiai_runner.py "TEST_VIDEO_URL"
```

---

### Phase 4: Complete Config & Code Cleanup (20 minutes)
**Priority**: MEDIUM 🟡 - Removes all remaining references

#### 4.1 Configuration Files
```python
# /rumiai_v2/config/settings.py
# Remove: 'creative_density': True,

# /rumiai_v2/config/constants.py
# Remove: 'creative_density' from any lists

# /rumiai_v2/validators/response_validator.py
# Remove: 'creative_density': { ... } validation schema

# /rumiai_v2/processors/service_contracts.py
# Remove: 'compute_creative_density_analysis': 'density_analysis',
```

#### 4.2 Error Handler
```python
# /rumiai_v2/core/error_handler.py
# Remove line: "1. Check prompt template: cat prompt_templates/creative_density_v2.txt"
```

#### 4.3 Compare ML Results Script
```python
# /scripts/compare_ml_results.py
# Remove or update:
# - Lines referencing 'creative_density'
# - Default analysis_type = 'creative_density'
# - creative_density comparison logic
```

#### 4.4 Clean up any prompt template files
```bash
# Remove old prompt templates if they exist
rm -f prompt_templates/creative_density*.txt
rm -f prompt_templates/visual_hierarchy*.txt
rm -f prompt_templates/emotion_dynamics*.txt
```

**Note**: Documentation (.md files) will be handled separately - they don't affect code execution

---

## 🚨 Breaking Change Matrix

| Component | Uses PRECOMPUTE? | Blocks Temporal? | Action Required |
|-----------|-----------------|------------------|-----------------|
| **rumiai_runner.py** | Import only | ❌ NO | Remove import |
| **temporal_compute.py** | ❌ NO | ❌ NO | Nothing |
| **local_video_runner.py** | ✅ YES (active) | ❌ NO | DELETE (not needed) |
| **test_python_only_e2e.py** | ✅ YES (active) | ❌ NO | DELETE (not needed) |
| **test_ml_fixes.py** | ✅ YES | ❌ NO | DELETE (not needed) |
| **ml_services.py** | ❌ NO | ❌ NO | Can delete after scene migration |

---

## 🎯 Execution Checklist

### Pre-Flight
- [x] Verify rumiai_runner.py doesn't use COMPUTE_FUNCTIONS (CONFIRMED)
- [x] Identify all active users of COMPUTE_FUNCTIONS (3 files found)
- [ ] Create backup branch: `git checkout -b remove-precompute-final`

### Execution Order
1. [ ] **Phase 1**: Delete 3 obsolete scripts (local_video_runner.py, test_python_only_e2e.py, test_ml_fixes.py)
2. [ ] **Phase 2**: Migrate scene detection to unified
3. [ ] **Test Point**: Verify scene detection works
4. [ ] **Phase 3**: Remove imports, delete precompute files, delete verify_sync.py
5. [ ] **Phase 4**: Complete config and code cleanup (all remaining references)
6. [ ] **Final Test**: Full pipeline test
7. [ ] **Commit**: Create PR with changes

### Post-Removal Verification
```bash
# 1. Production flow works
python3 scripts/rumiai_runner.py "VIDEO_URL"

# 2. Temporal output exists
ls -la insights/*_temporal_windows_updated.json

# 3. Has 50+ features
cat insights/*_temporal_windows_updated.json | jq '.temporal_windows.hook | keys | length'

# 4. No import errors
python3 -c "from rumiai_v2.processors import temporal_compute"

# 5. All services work
python3 -c "from rumiai_v2.api.ml_services_unified import UnifiedMLServices"
```

---

## ⚠️ Rollback Plan

If issues arise:
```bash
# Immediate rollback
git checkout main
git branch -D remove-precompute-final

# Debug
grep -r "COMPUTE_FUNCTIONS" --include="*.py"
python3 scripts/rumiai_runner.py "TEST" 2>&1 | grep -i error
```

---

## 📊 Success Criteria

✅ **Removal is successful when**:
1. Zero precompute*.py files in processors/
2. rumiai_runner.py runs without COMPUTE_FUNCTIONS
3. All 8 ML services operational
4. Temporal flow produces identical output
5. No import errors
6. Scene detection works through UnifiedMLServices

---

## 🎯 Expected Outcomes

### Code Reduction
- **Files deleted**: 5 precompute files
- **Lines removed**: ~2000+ lines of legacy code
- **Complexity**: Significant reduction

### Risk Assessment
- **Production risk**: ✅ LOW (rumiai_runner already clean)
- **Test impact**: MEDIUM (3 test files need updates)
- **Time required**: ~1 hour total

---

## 📅 Timeline

**Total Duration**: 55 minutes

1. **Phase 1** (5 min): Delete 3 obsolete scripts
2. **Phase 2** (20 min): Migrate scene detection
3. **Phase 3** (10 min): Remove imports and delete files
4. **Phase 4** (20 min): Complete config and code cleanup

---

## 🔍 Key Insight

**The production pipeline (rumiai_runner.py) is already independent of PRECOMPUTE!**

Only utility scripts and tests need updating. This makes the removal much safer than initially thought.

---

**Document Status**: Ready for Execution
**Last Updated**: 2025-01-23
**Discovery**: Complete via PRECOMPUTE_DISCOVERY.md