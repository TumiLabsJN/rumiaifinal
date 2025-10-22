# Fix Checkpoint Bug - Path Mismatch Between Stage 0 and Stage 2

**Date**: 2025-10-21
**Discovered During**: Test run with @vitalproteins competitor analysis
**Severity**: Critical - Blocks Stage 2.5 and all subsequent stages
**Affects**: All analysis types (hashtag, competitor, creator) with prefixed targets (#, @)

---

## Problem Analysis

### Root Cause

**Path construction inconsistency** between Stage 0 (foundation/paths.py) and Stage 2 (stage2_processing).

**Stage 0 (PathBuilder)**:
- Uses `sanitize_target()` to strip prefixes (`#`, `@`) before creating directories
- Example: `@vitalproteins` → `vitalproteins`
- Creates: `/data/clients/test_run/competitors/vitalproteins/top_top/`

**Stage 2 (get_bucket_path, bucket_init)**:
- Uses raw `config['target']` without sanitization
- Example: `@vitalproteins` → `@vitalproteins` (@ kept)
- Writes to: `/data/clients/test_run/competitors/@vitalproteins/top_top/`

**Result**:
- Stage 0 creates bucket directories at `vitalproteins/`
- Stage 2 writes checkpoints to `@vitalproteins/`
- Stage 2.5 looks for checkpoints at `vitalproteins/` (uses Stage 0's path logic)
- Stage 2.5 fails with: `FileNotFoundError: Checkpoint not found for bucket 13-18s`

---

### Evidence

**Test Run Details**:
```
Client: test_run
Target: @vitalproteins
Analysis Type: competitor
Videos Processed: 27/30 (successful)
```

**Directory Structure Created**:
```
/data/clients/test_run/competitors/
├── vitalproteins/top_top/          # Created by Stage 0 (PathBuilder)
│   └── buckets/
│       └── bucket_13-18s/
│           └── checkpoints/         # EMPTY (Stage 2.5 looks here)
└── @vitalproteins/top_top/         # Created by Stage 2 (get_bucket_path)
    └── buckets/
        └── bucket_13-18s/
            └── checkpoints/
                └── stage_2_checkpoint.json  # Written here, but never found!
```

**Checkpoint Files Found**:
```bash
$ find /home/jorge/rumiaifinal/data/clients/test_run/competitors -name "stage_2_checkpoint.json"
/data/clients/test_run/competitors/@vitalproteins/top_top/buckets/bucket_13-18s/checkpoints/stage_2_checkpoint.json
/data/clients/test_run/competitors/@vitalproteins/top_top/buckets/bucket_33-60s/checkpoints/stage_2_checkpoint.json
/data/clients/test_run/competitors/@vitalproteins/top_top/buckets/bucket_9-13s/checkpoints/stage_2_checkpoint.json
```

All checkpoints written to `@vitalproteins` (with @), but Stage 2.5 searches in `vitalproteins` (without @).

---

### Scope of Impact

**Affected Analysis Types**:
- ✅ **Hashtag** (`#nutrition`) - Stage 0 creates `nutrition/`, Stage 2 writes to `#nutrition/`
- ✅ **Competitor** (`@vitalproteins`) - Stage 0 creates `vitalproteins/`, Stage 2 writes to `@vitalproteins/`
- ✅ **Creator** (`@handle`) - Stage 0 creates `handle/`, Stage 2 writes to `@handle/`

**Confirmed Broken Runs**:
- `test_run/competitors/vitalproteins` - checkpoints in `@vitalproteins` directory
- `test_run/hashtags/fitness` - empty checkpoint directories (likely same issue)

**Pipeline Stages Affected**:
- ✅ Stage 0-2: Work correctly (videos process successfully)
- ❌ Stage 2.5: **FAILS** (cannot find checkpoints)
- ❌ Stage 3+: **BLOCKED** (Stage 2.5 is prerequisite)

---

## Code Analysis

### Files with Duplicated Path Construction

**1. `foundation/paths.py` (Stage 0 - CORRECT)**
```python
def get_target_dir(...):
    sanitized_target = sanitize_target(target, analysis_type)  # ✅ Strips @ and #
    return (
        self.base_path / "clients" / client_id /
        analysis_type_plural / sanitized_target / mode_strategy
    )
```

**2. `ml_pipeline/stage2_processing/utils.py` (Stage 2 - BROKEN)**
```python
def get_bucket_path(config: dict, bucket_name: str) -> str:
    data_root = os.getenv('DATA_ROOT', '/data')
    analysis_base = (
        f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/"
        f"{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"  # ❌ Uses raw target
    )
    return f"{analysis_base}buckets/bucket_{bucket_name}/"
```

**3. `ml_pipeline/stage2_processing/bucket_init.py` (Stage 2 - BROKEN)**
```python
def ensure_bucket_exists(bucket_name: str, config: dict) -> str:
    data_root = os.getenv('DATA_ROOT', '/data')
    analysis_base = (
        f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/"
        f"{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"  # ❌ Uses raw target
    )
    bucket_path = f"{analysis_base}buckets/bucket_{bucket_name}/"
```

**Path Construction Count**:
- Foundation: 1 implementation (✅ correct)
- Stage 2: 2 implementations (❌ both broken)
- **Total**: 3 places with path logic = **DRY violation**

---

## Solution: Option A (Centralize in PathBuilder)

### Approach

Make ALL stages use PathBuilder for path construction instead of duplicating logic.

**Key Principle**: Single Source of Truth (SSOT) for all path construction.

### Implementation Plan

#### Step 1: Extend PathBuilder

**File**: `foundation/paths.py`

Add new method:
```python
def get_bucket_path(
    self,
    client_id: str,
    analysis_type: str,
    target: str,
    analysis_mode: str,
    selection_strategy: str,
    bucket_name: str
) -> Path:
    """
    Get full bucket directory path.

    Centralizes bucket path construction to ensure consistency across all stages.
    Automatically applies target sanitization (strips @ and # prefixes).

    Args:
        client_id: Client identifier
        analysis_type: "hashtag", "competitor", or "creator"
        target: Target with prefix (#nutrition, @brand)
        analysis_mode: "top" or "recent"
        selection_strategy: "contrastive" or "top"
        bucket_name: Duration range (e.g., "18-33s")

    Returns:
        Path: Full bucket directory path
        Example: /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s

    Source: FixCheckpointBug.md - Centralized path construction fix
    """
    target_dir = self.get_target_dir(
        client_id=client_id,
        analysis_type=analysis_type,
        target=target,
        analysis_mode=analysis_mode,
        selection_strategy=selection_strategy
    )

    return target_dir / "buckets" / f"bucket_{bucket_name}"
```

#### Step 2: Refactor stage2_processing/utils.py

**File**: `ml_pipeline/stage2_processing/utils.py`

**Before**:
```python
def get_bucket_path(config: dict, bucket_name: str) -> str:
    data_root = os.getenv('DATA_ROOT', '/data')
    analysis_base = (
        f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/"
        f"{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"
    )
    return f"{analysis_base}buckets/bucket_{bucket_name}/"
```

**After**:
```python
def get_bucket_path(config: dict, bucket_name: str) -> str:
    """
    Get bucket path using centralized PathBuilder.

    REFACTORED (FixCheckpointBug.md): Delegates to PathBuilder.get_bucket_path()
    to ensure consistency with Stage 0 path construction. This fixes the bug where
    Stage 2 wrote checkpoints to @vitalproteins while Stage 0 created directories
    at vitalproteins (@ stripped).

    Args:
        config: dict, loaded from config.json
        bucket_name: str, duration range (e.g., "18-33s")

    Returns:
        str: Full bucket directory path with trailing slash

    Source: VideoProcessingTI.md Section 4 (Helper Function) + FixCheckpointBug.md
    """
    from foundation.paths import PathBuilder

    path_builder = PathBuilder()
    bucket_path = path_builder.get_bucket_path(
        client_id=config['client_id'],
        analysis_type=config['analysis_type'],
        target=config['target'],
        analysis_mode=config['analysis_mode'],
        selection_strategy=config['selection_strategy'],
        bucket_name=bucket_name
    )

    # Return as string with trailing slash (backward compatibility)
    return f"{bucket_path}/"
```

#### Step 3: Refactor stage2_processing/bucket_init.py

**File**: `ml_pipeline/stage2_processing/bucket_init.py`

**Before** (lines 124-129):
```python
data_root = os.getenv('DATA_ROOT', '/data')
analysis_base = (
    f"{data_root}/clients/{config['client_id']}/{config['analysis_type']}s/"
    f"{config['target']}/{config['analysis_mode']}_{config['selection_strategy']}/"
)
bucket_path = f"{analysis_base}buckets/bucket_{bucket_name}/"
```

**After**:
```python
from foundation.paths import PathBuilder

# REFACTORED (FixCheckpointBug.md): Use PathBuilder for consistent path construction
path_builder = PathBuilder()
bucket_path = path_builder.get_bucket_path(
    client_id=config['client_id'],
    analysis_type=config['analysis_type'],
    target=config['target'],
    analysis_mode=config['analysis_mode'],
    selection_strategy=config['selection_strategy'],
    bucket_name=bucket_name
)

# Convert Path to string for os.makedirs compatibility
bucket_path = str(bucket_path) + "/"
```

#### Step 4: Add Regression Test

**New File**: `ml_pipeline/tests/test_path_consistency.py`

```python
"""
Test that all stages use consistent path construction.

Prevents regression of the @ and # prefix bug discovered in FixCheckpointBug.md.
Ensures Stage 2, 2.5, 3, etc. all use the same paths as Stage 0 PathBuilder.

Source: FixCheckpointBug.md
"""

import pytest
from pathlib import Path
from foundation.paths import PathBuilder
from ml_pipeline.stage2_processing.utils import get_bucket_path


@pytest.mark.parametrize("target,analysis_type,expected_sanitized", [
    ("#nutrition", "hashtag", "nutrition"),
    ("@vitalproteins", "competitor", "vitalproteins"),
    ("@creator_handle", "creator", "creator_handle"),
    ("#Fitness & Health!", "hashtag", "fitness_health"),
])
def test_stage2_paths_match_pathbuilder(target, analysis_type, expected_sanitized):
    """
    Ensure Stage 2 get_bucket_path() matches PathBuilder output.

    Regression test for FixCheckpointBug.md - prevents path mismatch where
    Stage 0 creates vitalproteins/ but Stage 2 writes to @vitalproteins/.
    """

    config = {
        'client_id': 'test_client',
        'analysis_type': analysis_type,
        'target': target,
        'analysis_mode': 'top',
        'selection_strategy': 'contrastive'
    }
    bucket_name = "18-33s"

    # Stage 2 path
    stage2_path = get_bucket_path(config, bucket_name)

    # Stage 0 PathBuilder path
    path_builder = PathBuilder()
    stage0_path = path_builder.get_bucket_path(
        client_id=config['client_id'],
        analysis_type=config['analysis_type'],
        target=config['target'],
        analysis_mode=config['analysis_mode'],
        selection_strategy=config['selection_strategy'],
        bucket_name=bucket_name
    )

    # Compare (strip trailing slash from stage2_path)
    assert stage2_path.rstrip('/') == str(stage0_path), \
        f"Path mismatch for {target}:\n" \
        f"  Stage 0 PathBuilder: {stage0_path}\n" \
        f"  Stage 2 get_bucket_path: {stage2_path}\n" \
        f"  This violates SSOT principle from FixCheckpointBug.md"

    # Verify target sanitization happened
    assert expected_sanitized in str(stage0_path), \
        f"Expected sanitized target '{expected_sanitized}' in path: {stage0_path}"

    # Verify no @ or # in path (critical: this is the bug we're fixing)
    assert '@' not in str(stage0_path), \
        f"@ symbol should be stripped (FixCheckpointBug.md): {stage0_path}"
    assert '#' not in str(stage0_path), \
        f"# symbol should be stripped (FixCheckpointBug.md): {stage0_path}"


def test_pathbuilder_get_bucket_path_method_exists():
    """Verify PathBuilder has get_bucket_path method (Option A requirement)."""
    pb = PathBuilder()
    assert hasattr(pb, 'get_bucket_path'), \
        "PathBuilder must have get_bucket_path() method per FixCheckpointBug.md Option A"
```

#### Step 5: Audit Other Stages

**Check these files for similar path construction bugs**:
```bash
# Find all places using config['target'] for path construction
grep -r "config\['target'\]" ml_pipeline/ --include="*.py" | grep -v test
```

**Known locations to check**:
- `ml_pipeline/stage2_5_organize/file_organizer.py` - Uses `.lstrip('#').lstrip('@')` (hacky, should use PathBuilder)
- `ml_pipeline/stage3_*/` - Check if Stage 3 constructs paths
- `ml_pipeline/stage4_*/` - Check if Stage 4 constructs paths
- `ml_pipeline/stage5_*/` - Check if Stage 5 constructs paths

---

## Migration for Existing Broken Data

### Current State

Checkpoints exist in wrong locations:
```
/data/clients/test_run/competitors/@vitalproteins/top_top/buckets/bucket_*/checkpoints/
```

But Stage 2.5 searches in:
```
/data/clients/test_run/competitors/vitalproteins/top_top/buckets/bucket_*/checkpoints/
```

### Migration Options

**Option 1: Copy checkpoints to correct location (preserves work)**
```bash
# For vitalproteins test run
for bucket in 13-18s 33-60s 9-13s; do
    cp /home/jorge/rumiaifinal/data/clients/test_run/competitors/@vitalproteins/top_top/buckets/bucket_${bucket}/checkpoints/*.json \
       /home/jorge/rumiaifinal/data/clients/test_run/competitors/vitalproteins/top_top/buckets/bucket_${bucket}/checkpoints/
done

# Verify
find /home/jorge/rumiaifinal/data/clients/test_run/competitors/vitalproteins -name "stage_2_checkpoint.json"
```

**Option 2: Delete wrong directory and re-run (clean slate)**
```bash
# Delete incorrect directory
rm -rf /home/jorge/rumiaifinal/data/clients/test_run/competitors/@vitalproteins

# Re-run Stage 2 (with fix applied)
python rumiai_ml_batch.py --client test_run --analysis-type competitor --target "@vitalproteins" ...
```

**Option 3: Migration script (for production data)**
```python
# Script: migrate_checkpoint_paths.py
"""
Migrate checkpoints from @-prefixed directories to sanitized directories.

Handles:
- @vitalproteins → vitalproteins
- #nutrition → nutrition
- Any other prefix combinations

Source: FixCheckpointBug.md
"""

import os
import shutil
from pathlib import Path
from foundation.paths import sanitize_target

def migrate_checkpoints(data_root: str = "/data"):
    """Migrate all misplaced checkpoints to correct locations."""
    # TODO: Implement migration logic
    pass
```

**Recommendation**: Use Option 1 for test data, Option 3 for production

---

## Validation

### Pre-Fix Validation

**Reproduce the bug**:
```bash
# Run pipeline with @vitalproteins (should fail at Stage 2.5)
python rumiai_ml_batch.py --client test_run --analysis-type competitor --target "@vitalproteins" --video-count 10

# Expected error:
# ✗ Stage 2.5 failed: Checkpoint not found for bucket 13-18s
```

### Post-Fix Validation

**Test 1: Unit tests pass**
```bash
pytest ml_pipeline/tests/test_path_consistency.py -v
```

**Test 2: Integration test with competitor**
```bash
# Clean up old data
rm -rf /home/jorge/rumiaifinal/data/clients/test_run/competitors/vitalproteins
rm -rf /home/jorge/rumiaifinal/data/clients/test_run/competitors/@vitalproteins

# Run pipeline
python rumiai_ml_batch.py --client test_run --analysis-type competitor --target "@vitalproteins" --video-count 10

# Verify checkpoints in CORRECT location (no @ prefix)
find /home/jorge/rumiaifinal/data/clients/test_run/competitors/vitalproteins -name "stage_2_checkpoint.json"

# Should show:
# /data/clients/test_run/competitors/vitalproteins/top_top/buckets/bucket_13-18s/checkpoints/stage_2_checkpoint.json
# /data/clients/test_run/competitors/vitalproteins/top_top/buckets/bucket_33-60s/checkpoints/stage_2_checkpoint.json
# /data/clients/test_run/competitors/vitalproteins/top_top/buckets/bucket_9-13s/checkpoints/stage_2_checkpoint.json
```

**Test 3: Integration test with hashtag**
```bash
# Test hashtag to ensure # prefix is also stripped
python rumiai_ml_batch.py --client test_run --analysis-type hashtag --target "#nutrition" --video-count 10

# Verify checkpoints in nutrition/ (not #nutrition/)
find /home/jorge/rumiaifinal/data/clients/test_run/hashtags/nutrition -name "stage_2_checkpoint.json"
```

**Test 4: Stage 2.5 continues successfully**
```bash
# Full pipeline should complete through Stage 2.5
python rumiai_ml_batch.py --client test_run --analysis-type competitor --target "@vitalproteins" --video-count 10

# Should show:
# ✓ Stage 2: Video Processing - COMPLETE
# ✓ Stage 2.5: File Organization - COMPLETE  # ← Previously failed here
# ✓ Stage 2.6/2.7: Content Analysis - COMPLETE (or paused for manual step)
```

---

## Alternative Solutions Considered

### Option B: Simple Fix + Regression Test

**Approach**: Import `sanitize_target()` in Stage 2 files

**Pros**:
- ✅ Minimal code changes (2 files)
- ✅ Quick to implement
- ✅ Low risk

**Cons**:
- ❌ Still violates DRY (3 places with path logic)
- ❌ Fragile (easy to forget in new stages)
- ❌ Doesn't address root cause (code duplication)

**Verdict**: ❌ Rejected - Band-aid solution that doesn't fix architectural problem

---

## Risk Assessment

### Risks of Implementing Option A

**1. Breaking Existing Code** - LOW RISK
- Function signatures unchanged (`get_bucket_path` still returns `str`)
- Backward compatible (trailing slash preserved)
- No changes to external APIs

**2. Import Dependency** - MEDIUM RISK
- Stage 2 now imports from `foundation.paths`
- Could cause circular dependency if foundation imports Stage 2
- **Mitigation**: Foundation should never import from ml_pipeline (already true)

**3. PathBuilder Instantiation Cost** - LOW RISK
- Creates new PathBuilder object on each call
- Performance impact negligible (< 1ms per call)
- Called ~30-100 times per pipeline run (once per video)

**4. Test Coverage Gaps** - MEDIUM RISK
- New code needs thorough testing
- Integration tests required
- **Mitigation**: Add comprehensive test suite (Step 4)

**5. Migration Complexity** - HIGH RISK (for production data)
- Existing runs have checkpoints in wrong locations
- Migration required for any in-progress pipelines
- **Mitigation**: Provide migration script (Option 3)

---

## Timeline

**Estimated Implementation Time**:
1. Add PathBuilder.get_bucket_path(): **30 minutes**
2. Refactor stage2_processing/utils.py: **15 minutes**
3. Refactor stage2_processing/bucket_init.py: **15 minutes**
4. Add regression tests: **45 minutes**
5. Audit other stages: **60 minutes**
6. Test and validate: **90 minutes**

**Total**: ~4 hours

---

## Related Issues

**Similar Bugs to Check**:
1. Do Stage 3, 4, 5 construct paths? (Need audit)
2. Does Stage 2.5 file organization have similar issue? (Checked: uses `.lstrip('#').lstrip('@')` - hacky workaround)
3. Are there other places using raw `config['target']`? (Need grep audit)

**Documentation Updates Needed**:
- VideoProcessingTI.md - Update Section 4 (Helper Functions)
- FoundationCHILD.md - Document new PathBuilder.get_bucket_path() method
- Add inline comments explaining why PathBuilder is used

---

## Decision

**✅ APPROVED: Implement Option A (Centralize in PathBuilder)**

**Rationale**:
- Fixes root cause (code duplication), not just symptoms
- Establishes Single Source of Truth for path construction
- Prevents future regressions with same pattern
- Clean architectural solution despite higher upfront cost

**Next Steps**:
1. Implement Step 1-5 from Solution section
2. Run validation tests
3. Migrate test data (vitalproteins, fitness)
4. Document in changelog
5. Monitor for similar issues in other stages

---

**Document Maintainer**: Claude (Sonnet 4.5)
**Last Updated**: 2025-10-21
**Status**: Solution Designed, Awaiting Implementation
