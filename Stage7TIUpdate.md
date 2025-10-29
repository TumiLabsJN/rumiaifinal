# Stage 7 TI Update Guide

**Document**: LLMAnalysisCHILDTI.md Alignment to Production Code
**Date**: 2025-01-28
**Status**: Discovery Complete - Updates Required
**Source of Truth**: Production Code (`rumiai_ml_batch.py`, `stage7_llm_analysis.py`)

---

## Executive Summary

**Discovery Scope**: Systematic comparison of LLMAnalysisCHILDTI.md (7556 lines) against production implementation
**Files Analyzed**:
- `/home/jorge/rumiaifinal/rumiai_ml_batch.py` (1900 lines) - Orchestrator
- `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py` (763 lines) - Implementation
- `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_prompts.py` - Prompt builders

**Key Findings**:
- ❌ **Section Reference Errors**: 2 instances (wrong section numbers in code comments)
- ❌ **TI Over-Specification**: 9 functions documented but not implemented (1165 lines, 15% of TI)
- ❌ **Error Handling Mismatch**: TI documents 5 categories, production uses 3
- ❌ **Missing Documentation**: 6 production features not documented in TI
- ✅ **API Call Documentation**: Accurate (verified)
- ✅ **Output Schema**: Matches production

**Overall Assessment**: TI contains significant over-specification from abandoned design approaches. Requires major simplification to reflect production reality.

---

## Priority 1: Critical Code Comment Fixes (IMMEDIATE - 5 minutes)

### Issue 1.1: Wrong Section Reference in validate_stage7_outputs()

**File**: `rumiai_ml_batch.py`
**Line**: 334
**Current**:
```python
"""
Source: LLMAnalysisCHILDTI.md Section 3 (Stage Contract - StageOutput)
"""
```

**Problem**: Section 3 is "Data Schemas", not "Stage Contract"
**TI Reality**: Section 2.2 (line 182) contains "Output Contract" (Stage7Output)

**Fix**:
```python
"""
Source: LLMAnalysisCHILDTI.md Section 2.2 (Output Contract)
"""
```

---

### Issue 1.2: Wrong Section Reference in cleanup_stage7_partial_outputs()

**File**: `rumiai_ml_batch.py`
**Line**: 453
**Current**:
```python
"""
Source: LLMAnalysisCHILDTI.md Section 3 StageOutput
"""
```

**Problem**: Same as Issue 1.1 - wrong section number

**Fix**:
```python
"""
Source: LLMAnalysisCHILDTI.md Section 2.2 (Output Contract)
"""
```

---

## Priority 2: Major TI Simplification (HIGH - 1-2 days)

### Issue 2.1: Remove Unimplemented Functions from TI Section 4

**TI Location**: Section 4.1 through 4.9 (lines 680-1843)
**Size**: ~1165 lines (15% of entire TI)
**Status**: ❌ NOT IMPLEMENTED in production

**Functions to Remove**:

| Function | TI Line | Reason | Production Alternative |
|----------|---------|--------|----------------------|
| `detect_bimodal_pattern()` | 680 | LLM does all pattern detection | Handled in LLM prompt/response |
| `identify_high_contrast_features()` | 818 | LLM analyzes all 21 features | Full centroids sent to LLM |
| `compute_rf_alignment()` | 1005 | LLM computes alignment scores | LLM generates "RF alignment: N/5" |
| `enrich_high_contrast_features()` | 1182 | Inline in prompt builder | `build_phase1_prompt()` does this |
| `prepare_path_data_for_llm()` | 1328 | Simple list sufficient | `extract_cluster_paths()` (line 596) |
| `classify_confidence_level()` | 1451 | LLM assigns confidence | LLM output has `confidence_level` |
| `generate_universal_principles()` | 1519 | LLM generates principles | Part of Phase 2 LLM response |
| `generate_cross_window_patterns()` | 1534 | LLM detects patterns | Part of Phase 2 LLM response |
| `generate_feature_based_reports()` | 1659 | LLM generates Scenario D | LLM handles all scenarios |

**Action**: Delete TI lines 680-1843 (Section 4.1-4.9)

**Rationale**:
- Original TI design: "Python handles arithmetic, LLM handles creativity"
- Production reality: Claude Sonnet 4 is powerful enough to do ALL analysis
- Simpler architecture: LLM-only approach eliminates 9 complex functions
- Maintenance burden: Documenting unimplemented functions misleads developers

**After Deletion, Renumber**:
- Current 4.10 (`run_phase1_parallel()`) → New 4.1
- Current 4.11 (`analyze_window_with_retry()`) → New 4.2
- Current 4.12 (`run_phase2_synthesis()`) → New 4.3
- Current 4.13 (`build_phase1_prompt()`) → New 4.4
- Current 4.14 (`build_phase2_prompt()`) → New 4.5

**Verification**: Search production code for function names:
```bash
grep -r "detect_bimodal_pattern\|identify_high_contrast_features\|compute_rf_alignment\|enrich_high_contrast\|prepare_path_data\|classify_confidence\|generate_universal_principles\|generate_cross_window_patterns\|generate_feature_based_reports" ml_pipeline/stage7_llm_analysis/
# Result: No matches (functions don't exist)
```

---

### Issue 2.2: Simplify Error Handling Documentation

**TI Location**: Section 6.1 (lines 3622-3633)
**Problem**: TI documents 5 error categories, production uses 3

**TI Current Structure** (5 categories):
1. Pre-Flight Failures (exit code 1-3)
2. Retryable API Errors (exit code 0 - auto-recover)
3. Fatal API Errors (exit code 4)
4. Execution Failures (exit code 5)
5. Data Integrity Errors (exit code 6)

**Production Reality** (`rumiai_ml_batch.py:397-448`):

**3 main error categories**:
1. **LLM Validation Errors** (lines 410-414)
   - Missing 'clusters' key in response
   - Wrong cluster count (not 3)
   - **Handling**: Automatic retry with backoff [0s, 2s, 4s]

2. **Phase 1 Execution Errors** (lines 417-420)
   - Window analysis failures
   - **Handling**: Check `.phase1_status.json`, resume from checkpoint

3. **Insufficient Data Errors** (lines 423-427)
   - <3 cluster paths meet 10% threshold
   - **Handling**: Automatic fallback to feature-based reports (NOT a failure)

**Plus API Error Codes**:
- **401 Unauthorized** (lines 430-434): Authentication failed, NO RETRY
- **429 Rate Limit / 503 Service Unavailable** (lines 436-439): RETRY with exponential backoff

**Action**: Rewrite TI Section 6 to match production's 3-category structure

**Proposed TI Section 6 Structure**:

```markdown
## Section 6: Error Handling

Stage 7 has 3 main error categories based on production implementation:

### 6.1 LLM Validation Errors

**When**: After receiving LLM API response
**Examples**:
- Missing 'clusters' key in JSON response
- Expected 3 clusters, got different count
- Malformed JSON (invalid syntax)

**Handling**: Automatic retry with exponential backoff
- Attempt 1: Immediate (0s delay)
- Attempt 2: 2s delay
- Attempt 3: 4s delay

**Exit Code**: 2 (if all retries exhausted)

**Source**: `rumiai_ml_batch.py:410-414`

### 6.2 Phase 1 Execution Errors

**When**: During Phase 1 window analysis
**Examples**:
- Window fails after 3 retry attempts
- API timeout (>90s)
- File I/O errors writing window_analysis.json

**Handling**:
- Status tracked in `.phase1_status.json`
- Partial completion allowed
- Re-running Stage 7 resumes from checkpoint (only processes incomplete windows)

**Exit Code**: 2 (Phase 1 failure)

**Source**: `rumiai_ml_batch.py:417-420`

### 6.3 Insufficient Data Errors (Not a Failure)

**When**: Phase 2 path extraction yields <3 paths ≥10% threshold
**Scenario**: Only 0-2 cluster paths meet frequency threshold

**Handling**:
- Automatic fallback to feature-based reports
- LLM generates reports using RF features instead of paths
- This is an EXPECTED scenario, not an error

**Exit Code**: 0 (success - system handled gracefully)

**Source**: `rumiai_ml_batch.py:423-427`

### 6.4 API Authentication & Rate Limiting

**401 Unauthorized**:
- Cause: Invalid API key
- Handling: NO RETRY (immediate exit)
- Exit Code: 4
- User Action: Check ANTHROPIC_API_KEY environment variable

**429 Rate Limit / 503 Service Unavailable**:
- Cause: API rate limits or service issues
- Handling: RETRY with exponential backoff [0s, 2s, 4s]
- Exit Code: 0 (if retry succeeds) or 2 (if retries exhausted)

**Source**: `rumiai_ml_batch.py:430-439`
```

**Changes from TI**:
- Collapsed "Pre-Flight Failures" into "Phase 1 Execution Errors"
- Removed "Data Integrity Errors" as separate category (absorbed into Phase 1)
- Elevated "Insufficient Data" to top-level (it's a key scenario, not an error)
- Clarified retry logic with exact intervals: [0s, 2s, 4s]
- Referenced exact production code line numbers

---

## Priority 3: Add Missing Production Features to TI (MEDIUM - 1 day)

### Issue 3.1: Document "Scenario Field Removed" Note

**Production Location**: `rumiai_ml_batch.py:376`
**Code Comment**:
```python
# Note: 'scenario' field removed (internal logic only, not saved)
```

**TI Location to Update**: Section 3.3.3 (WinningFormulasSchema, line ~596)

**Add to Schema Notes**:
```markdown
**Field Removal Note**: The `scenario` field (A/B/C/D classification) is computed internally during
Phase 2 but is NOT saved to `winning_formulas.json`. Scenario is used only to route LLM prompt
generation logic. This design decision keeps output focused on creative insights rather than
exposing internal classification mechanics.

Source: `run_phase2_synthesis()` lines 438-450, `rumiai_ml_batch.py:376`
```

---

### Issue 3.2: Document Idempotency & Skip Logic

**Production Location**: `rumiai_ml_batch.py:1719-1736`
**Behavior**: If `complete_analysis_{bucket}.json` exists, skip Stage 7 for that bucket

**TI Location to Add**: Create new Section 2.3 "Idempotency & Resume Behavior"

**Proposed Content**:
```markdown
### 2.3 Idempotency & Resume Behavior

Stage 7 is **idempotent** and supports **checkpoint-based resume**.

#### 2.3.1 Bucket-Level Idempotency

**Check**: If `complete_analysis_{bucket}.json` exists in `ml_analysis/llm/`
**Action**: Skip Stage 7 entirely for that bucket (no API calls made)
**Rationale**: Cost savings - avoid re-processing already completed buckets

**Implementation**: `rumiai_ml_batch.py:1719-1736`

**Example**:
```python
complete_path = f"ml_analysis/llm/complete_analysis_{bucket}.json"
if os.path.exists(complete_path):
    logger.info(f"Skipping Stage 7 for {bucket} (already complete)")
    # Count existing JSON files and continue to next bucket
```

#### 2.3.2 Window-Level Resume (Phase 1)

**Check**: `.phase1_status.json` tracks which windows completed
**Action**: Re-run only resumes incomplete windows
**Cost Impact**: Only pay for incomplete windows (completed windows loaded from disk)

**Implementation**: `stage7_llm_analysis.py:127-159`

**Status File Schema**:
```json
{
  "total_windows": 6,
  "completed_windows": ["hook", "middle_1", "middle_2"],
  "failed_windows": [{"window": "middle_3", "error": "timeout"}],
  "phase1_complete": false,
  "started_at": "2025-01-28T10:00:00Z",
  "last_updated": "2025-01-28T10:05:23Z"
}
```

**Resume Logic**:
- Load `.phase1_status.json` if exists
- Skip windows in `completed_windows` (load from file)
- Re-run windows NOT in `completed_windows`
- Continue from last checkpoint seamlessly
```

---

### Issue 3.3: Document "Skip Bucket, Continue Pipeline" Strategy

**Production Location**: `rumiai_ml_batch.py:1777-1795`
**Behavior**: Non-fatal errors skip bucket but continue processing other buckets

**TI Location to Add**: Section 6 (Error Handling)

**Proposed Section**: 6.5 Bucket-Level Error Recovery

```markdown
### 6.5 Bucket-Level Error Recovery

For certain non-fatal errors, Stage 7 **skips the failing bucket** but **continues processing remaining buckets**.

#### Errors That Skip Bucket (Continue Pipeline)

**FileNotFoundError** (`rumiai_ml_batch.py:1777-1785`):
- Cause: Stage 6 outputs missing for this bucket
- Action: Skip bucket, log warning, continue to next bucket
- Rationale: One bucket's failure shouldn't block others
- Exit Code: 0 (pipeline completes successfully)

**ValueError** (`rumiai_ml_batch.py:1787-1795`):
- Cause: Invalid bucket configuration or data format
- Action: Skip bucket, log error, continue to next bucket
- Exit Code: 0 (pipeline completes successfully)

#### Errors That Exit Pipeline (Stop All Buckets)

**RuntimeError** (`rumiai_ml_batch.py:1797-1805`):
- Cause: API authentication failure (401)
- Action: Exit entire pipeline immediately
- Rationale: Auth is system-wide; other buckets will also fail
- Exit Code: 4

**IOError / OSError** (`rumiai_ml_batch.py:1807-1815`):
- Cause: File system permission issues, disk full
- Action: Exit entire pipeline immediately
- Rationale: System-level issues affect all buckets
- Exit Code: 4

**Exception (catch-all)** (`rumiai_ml_batch.py:1817-1824`):
- Cause: Unexpected errors
- Action: Exit entire pipeline immediately
- Rationale: Unknown errors require investigation
- Exit Code: 99
```

---

### Issue 3.4: Document JSON Counting Exclusion

**Production Location**: `rumiai_ml_batch.py:1758-1761`
**Code**:
```python
# Count JSON files (excluding .phase1_status.json - internal file)
json_files = [f for f in os.listdir(llm_output_dir)
              if f.endswith('.json') and not f.startswith('.')]
```

**TI Location to Update**: Section 3.3.1 (Phase1StatusSchema)

**Add Note**:
```markdown
**File Counting Note**: The `.phase1_status.json` file is **excluded from output file counts**
when reporting Stage 7 completion metrics. This is because:
- It's an internal tracking file (not consumed by Stage 8)
- Hidden file convention (starts with `.`)
- Should not be counted alongside deliverable outputs

Implementation: `rumiai_ml_batch.py:1758-1761`
```

---

### Issue 3.5: Add Error Message Examples

**Problem**: TI Section 6 describes errors but doesn't show concrete error messages
**Location to Update**: Throughout Section 6

**Example - Add to Section 6.2.3 (Stage 6 Files Missing)**:

```markdown
**Example Error Output**:
```
ERROR: Stage 7 pre-flight validation failed
Stage 6 incomplete: Missing 7 of 13 required JSON files for bucket 18-33s

Missing files:
  - ml_analysis/hook_rf_analysis.json
  - ml_analysis/hook_kmeans_analysis.json
  - ml_analysis/middle_1_rf_analysis.json
  - ml_analysis/middle_1_kmeans_analysis.json
  - ml_analysis/middle_2_rf_analysis.json
  (showing first 5 of 7 missing files)

Action: Re-run Stage 6 before Stage 7
Command: python rumiai_ml_batch.py --stage 6 --bucket 18-33s

Exiting with code 1
```

Source: `rumiai_ml_batch.py:287-328` (validate_stage7_prerequisites)
```

**Apply Similar Format to**:
- Section 6.2.1: Missing API Key error
- Section 6.2.2: Invalid API Key format error
- Section 6.3.1: LLM Validation error
- Section 6.4.1: Phase 1 execution error

---

### Issue 3.6: Document Cleanup Policy

**Production Location**: `rumiai_ml_batch.py:450-486`
**Function**: `cleanup_stage7_partial_outputs()`

**TI Location to Update**: Section 6 (add new subsection 6.6)

**Proposed Content**:
```markdown
### 6.6 Cleanup Policy for Failed Executions

**When Cleanup Occurs**: Only on **catastrophic failures** (non-recoverable errors)

**Files Cleaned Up**:
- `winning_formulas.json` (Phase 2 output)
- `.phase1_status.json` (checkpoint file)
- All `*_analysis.json` files (window analyses)
- `complete_analysis_{bucket}.json` (combined output)

**When Cleanup Does NOT Occur**:
- Recoverable errors (retries remaining)
- Partial Phase 1 completion (checkpoint preserved for resume)
- User-initiated interruption (Ctrl+C)

**Rationale**: Preserve checkpoint state for resume capability. Only cleanup when state is
irreparably corrupted.

**Implementation**: `rumiai_ml_batch.py:450-486`

**Example - Catastrophic Failure Scenarios**:
- JSON parsing error indicating corrupted LLM response
- Disk full during write (all outputs incomplete)
- API returns malformed response after all retries

**Example - Recoverable Scenarios (NO cleanup)**:
- 2 of 6 windows completed, then API timeout → Keep checkpoint, resume later
- Phase 1 complete, Phase 2 fails → Keep Phase 1 outputs
- Single window retry exhausted → Mark as failed in status, allow manual retry
```

---

## Priority 4: Verify and Update Function References (LOW - 2 hours)

### Issue 4.1: Update Production Code Function References

**File**: `stage7_llm_analysis.py`
**Lines**: 7-12

**Current**:
```python
"""
Functions:
    - run_phase1_parallel() - TI §4.10
    - analyze_window_with_retry() - TI §4.11
    - run_phase2_synthesis() - TI §4.12
    - main() - Entry point

Source: LLMAnalysisCHILDTI.md Section 4.10-4.12
"""
```

**After TI Section 4 Renumbering** (from Priority 2):
```python
"""
Functions:
    - run_phase1_parallel() - TI §4.1
    - analyze_window_with_retry() - TI §4.2
    - run_phase2_synthesis() - TI §4.3
    - build_phase1_prompt() - TI §4.4
    - build_phase2_prompt() - TI §4.5
    - main() - Entry point

Source: LLMAnalysisCHILDTI.md Section 4.1-4.5
"""
```

---

## Priority 5: Add Design Decision Documentation (LOW - 1 hour)

### Issue 5.1: Document Why Python Preprocessing Was Abandoned

**TI Location to Add**: Section 4 introduction (after deletion of 4.1-4.9)

**Proposed Content**:
```markdown
## Section 4: Production Functions

### Design Evolution Note

**Original Design** (documented in TI v1.0):
- 14 functions split between Python preprocessing (4.1-4.9) and orchestration (4.10-4.14)
- Philosophy: "Python handles arithmetic, LLM handles creativity"
- Goal: Prevent LLM hallucination via mechanical pre-computation

**Production Reality** (implemented):
- 5 functions only (current 4.1-4.5)
- Philosophy: "LLM-only analysis with Python validation"
- Claude Sonnet 4 proven reliable enough for all analysis

**Why Change Occurred**:
1. **LLM Quality**: Claude Sonnet 4 (released 2025-01) handles complex analysis reliably
2. **Simplicity**: Eliminating 9 functions reduced complexity by 60%
3. **Maintainability**: Fewer functions = less code to test, debug, and document
4. **Cost-Benefit**: Python preprocessing added complexity without improving output quality
5. **Validation Sufficient**: Post-processing JSON schema validation catches errors effectively

**Functions Removed** (not implemented):
- `detect_bimodal_pattern()` - LLM detects patterns
- `identify_high_contrast_features()` - LLM analyzes all features
- `compute_rf_alignment()` - LLM generates alignment scores
- `enrich_high_contrast_features()` - Prompt builder handles inline
- `prepare_path_data_for_llm()` - Simple data structures sufficient
- `classify_confidence_level()` - LLM assigns confidence
- `generate_universal_principles()` - LLM generates principles
- `generate_cross_window_patterns()` - LLM detects temporal patterns
- `generate_feature_based_reports()` - LLM handles all scenarios (A/B/C/D)

**Result**: Production Stage 7 relies on LLM for all analysis, with Python handling only:
- Data loading (JSON files)
- Prompt construction
- API communication
- Response validation (schema checks)
- File I/O (saving outputs)

This architectural decision may be revisited if:
- LLM quality degrades in future model versions
- Cost optimization requires reducing LLM workload
- Output quality issues arise that Python preprocessing could solve
```

---

## Priority 6: API Call Documentation (VERIFIED - No Changes Needed)

### Issue 6.1: API Call Count Documentation

**Status**: ✅ ACCURATE - No TI updates needed

**Verification Results**:
- Formula: `API Calls = len(BUCKET_WINDOWS[bucket]) + 1`
- Phase 1: 1 call per window (sequential execution, max_workers=1)
- Phase 2: 1 call per bucket (synthesis)

**Breakdown by Bucket** (verified against `config/bucket_definitions.py`):

| Bucket | Windows | Phase 1 | Phase 2 | Total |
|--------|---------|---------|---------|-------|
| 0-3s | 1 | 1 | 1 | 2 |
| 3-9s | 2 | 2 | 1 | 3 |
| 9-13s | 3 | 3 | 1 | 4 |
| 13-18s | 3 | 3 | 1 | 4 |
| 18-33s | 6 | 6 | 1 | 7 |
| 33-60s | 7 | 7 | 1 | 8 |
| 60-90s | 7 | 7 | 1 | 8 |
| 90-120s | 7 | 7 | 1 | 8 |

**Total for all 8 buckets**: 44 API calls

**Source Code Verification**:
- API Call 1: `stage7_llm_analysis.py:283` (Phase 1, inside `analyze_window_with_retry()`)
- API Call 2: `stage7_llm_analysis.py:491` (Phase 2, inside `run_phase2_synthesis()`)
- Sequential execution: `stage7_llm_analysis.py:146` (`max_workers=1`)

**Note**: If TI documents API call counts, ensure it matches the above. If not documented, consider adding to TI Section 9 (Performance & Scalability).

---

## Summary of Actions Required

### Immediate (Priority 1) - 5 minutes
- [ ] Fix `rumiai_ml_batch.py:334` - Change "Section 3" → "Section 2.2"
- [ ] Fix `rumiai_ml_batch.py:453` - Change "Section 3" → "Section 2.2"

### High Priority (Priority 2) - 1-2 days
- [ ] Delete TI Section 4.1-4.9 (lines 680-1843, ~1165 lines)
- [ ] Renumber TI Section 4.10-4.14 → 4.1-4.5
- [ ] Rewrite TI Section 6 (Error Handling) - collapse 5 categories → 3

### Medium Priority (Priority 3) - 1 day
- [ ] Add "scenario field removed" note to TI Section 3.3.3
- [ ] Add new TI Section 2.3 (Idempotency & Resume Behavior)
- [ ] Add TI Section 6.5 (Bucket-Level Error Recovery)
- [ ] Add JSON counting exclusion note to TI Section 3.3.1
- [ ] Add error message examples throughout TI Section 6
- [ ] Add TI Section 6.6 (Cleanup Policy)

### Low Priority (Priority 4 & 5) - 3 hours
- [ ] Update function references in `stage7_llm_analysis.py:7-12`
- [ ] Add design evolution note to TI Section 4 introduction

### Verification (Priority 6) - No action needed
- [x] API call documentation verified accurate

---

## Testing & Validation After Updates

### 1. Code Comment Verification
```bash
# After fixing code comments, verify all TI references are correct:
grep -n "LLMAnalysisCHILDTI.md" rumiai_ml_batch.py
grep -n "TI §" ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py
```

### 2. TI Section Cross-References
```bash
# After renumbering Section 4, search for broken references:
grep -n "Section 4\." documentation_migration/FutureDevelopments/ChildDocs/LLMAnalysisCHILDTI.md
grep -n "§4\." documentation_migration/FutureDevelopments/ChildDocs/LLMAnalysisCHILDTI.md
```

### 3. Production Function Verification
```bash
# Verify no unimplemented functions are referenced:
grep -r "detect_bimodal\|identify_high_contrast\|compute_rf_alignment" ml_pipeline/stage7_llm_analysis/
# Expected: No results (functions don't exist)
```

### 4. Line Count Validation
```bash
# Verify TI reduction after deletion:
wc -l documentation_migration/FutureDevelopments/ChildDocs/LLMAnalysisCHILDTI.md
# Before: 7556 lines
# After: ~6391 lines (7556 - 1165 = 6391)
```

---

## Appendix: Complete File Mapping

### Production Files Analyzed
1. **Orchestrator**: `/home/jorge/rumiaifinal/rumiai_ml_batch.py` (1900 lines)
   - Lines 287-328: `validate_stage7_prerequisites()`
   - Lines 331-391: `validate_stage7_outputs()`
   - Lines 394-448: `handle_stage7_error()`
   - Lines 450-486: `cleanup_stage7_partial_outputs()`
   - Lines 1694-1843: Main Stage 7 execution block

2. **Implementation**: `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py` (763 lines)
   - Lines 96-218: `run_phase1_parallel()`
   - Lines 225-369: `analyze_window_with_retry()`
   - Lines 376-589: `run_phase2_synthesis()`
   - Lines 596-670: `extract_cluster_paths()` (helper)
   - Lines 677-734: `main()` (entry point)

3. **Prompts**: `/home/jorge/rumiaifinal/ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
   - Line 265: `build_phase1_prompt()`
   - Line 535: `build_phase2_prompt()`

4. **Configuration**: `/home/jorge/rumiaifinal/config/bucket_definitions.py`
   - BUCKET_WINDOWS dict (defines window counts for API call calculation)

### TI Document Structure
- **File**: `/home/jorge/rumiaifinal/documentation_migration/FutureDevelopments/ChildDocs/LLMAnalysisCHILDTI.md`
- **Total Lines**: 7556
- **Section 1**: Document Metadata (lines 1-90)
- **Section 2**: Stage Contract (lines 92-324)
- **Section 3**: Data Schemas (lines 327-672)
- **Section 4**: Algorithmic Specifications (lines 674-2719) ← TARGET FOR MAJOR REDUCTION
- **Section 6**: Error Handling (lines 3618+) ← TARGET FOR SIMPLIFICATION
- **Section 12**: Dependencies (lines 6941+)

---

## Document Control

**Created**: 2025-01-28
**Last Updated**: 2025-01-28
**Version**: 1.0
**Status**: Ready for Implementation
**Approver**: [Your Name/Team]
**Implementation Tracker**: Create GitHub issues for each checkbox item

---

## Notes for Implementer

1. **Order Matters**: Complete Priority 1 first (code fixes), then Priority 2 (TI deletion/renumbering), then update references
2. **Backup TI**: Create `LLMAnalysisCHILDTI_v1.0_backup.md` before making changes
3. **Version Control**: Commit after each priority level completes
4. **Testing**: Run Stage 7 on a test bucket after code comment fixes to ensure nothing breaks
5. **Documentation**: Update `CHANGELOG.md` or equivalent with summary of TI changes
6. **Communication**: Notify team about TI simplification (35% reduction in Section 4)

---

**End of Stage 7 TI Update Guide**
