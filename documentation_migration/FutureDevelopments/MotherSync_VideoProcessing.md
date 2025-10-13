# Mother Document Sync: Proposed Changes from VideoProcessing Work

> **Trigger**: Child HLD work for VideoProcessing revealed Mother doc issues
> **Component**: VideoProcessing (Stage 2)
> **Phase Outputs Reviewed**:
>   - VideoProcessingCHILD.md (Phase 3 - Issues 8-17 resolved)
> **Date**: 2025-01-28
> **Status**: APPLIED
> **Applied Date**: 2025-01-28

---

## Summary

**Total Changes Proposed**: 1

**Impact Scope**:
- Level 1 (Single Component): 1 change (VideoProcessingCHILD.md only)

**Affected Child Docs**: VideoProcessingCHILD.md

---

## Proposed Changes

### Change 1: [Broken Reference] MLPlanningv2.md Stage 2 Line Number Reference

**Issue Type**: Broken Reference

**Current State**:
- **Child Section**: VideoProcessingCHILD.md Section 10.1 (Line 1398)
- **Current Text**:
  ```markdown
  - **MLPlanningv2.md - Stage 2: Video Processing** (lines 644-708)
  ```

**Problem Discovered**:
- **By Comparing**: VideoProcessingCHILD.md Line 1398 vs actual MLPlanningv2.md content
- **Evidence**:
  - Child doc references "lines 644-708" for Stage 2
  - Actual Stage 2 section in MLPlanningv2.md: Lines 733-917
  - Lines 644-708 contain Stage 1.4 and 1.5 (Video Selection and Interactive Confirmation)
  - Broken reference would confuse TI generator looking for Stage 2 details

**Root Cause**:
- MLPlanningv2.md may have been reorganized after VideoProcessingCHILD.md was created
- OR initial reference was incorrect

**Proposed Update**:

**Option A: Update Child Doc (VideoProcessingCHILD.md Line 1398)**
```markdown
- **MLPlanningv2.md - Stage 2: Video Processing** (lines 733-917)
  - High-level stage overview
  - Stage position in pipeline
  - Input/output contracts
```

**Option B: Add Section Markers to Mother Doc**
Add clear section markers in MLPlanningv2.md that don't rely on line numbers:
```markdown
## Stage 2: Video Processing (RumiAI Pipeline) {#stage-2}
```
Then update Child to reference by anchor: `MLPlanningv2.md#stage-2`

**Rationale**:
- Line number references are fragile (break when Mother doc edited)
- TI generator needs accurate references to understand stage context
- Prevents TI from reading wrong section (Stage 1 instead of Stage 2)

**Impact Scope**: Level 1 - Only VideoProcessingCHILD.md references this section

**Priority**: CRITICAL
- Broken reference blocks TI from finding correct Mother content
- Could cause TI to implement wrong stage logic

**Recommended Solution**: Option A (immediate fix) + Option B (long-term improvement)
1. Fix line numbers in VideoProcessingCHILD.md now (733-917)
2. Add section anchors to MLPlanningv2.md for future robustness

---

## Change Summary by Priority

### [CRITICAL] Changes (must apply)
1. Change 1: Fix Stage 2 line number reference (644-708 → 733-917)

### [HIGH] Changes (should apply)
None

### [LOW] Changes (optional)
None

---

## Recommended Action

**Option A: Fix Child Doc Only** (quick fix)
- Update VideoProcessingCHILD.md Line 1398: change "644-708" to "733-917"
- No Mother doc changes needed
- No re-audit needed (fixing broken reference, not changing design)
- Estimated effort: 1 minute

**Option B: Fix Child + Add Section Anchors** (robust fix)
- Update VideoProcessingCHILD.md Line 1398: change "644-708" to "733-917"
- Add section anchors to MLPlanningv2.md (all stages)
- Update all Child docs to use anchor references instead of line numbers
- Estimated effort: 15-30 minutes

**Option C: Do Nothing**
- Leave incorrect reference
- TI generator must manually find correct Stage 2 section
- Risk of confusion

---

## User Decision

**Selected Option**: A (Fix Child Doc Only - Use Section Headers)

**Changes Applied**:
1. ✅ VideoProcessingCHILD.md Line 3: Changed from `(Lines 656-708)` to `Section "Stage 2: Video Processing (RumiAI Pipeline)"`
2. ✅ VideoProcessingCHILD.md Line 1398: Changed from `(lines 644-708)` to `Section "Stage 2: Video Processing (RumiAI Pipeline)"`

**Status**: APPLIED

---

## Additional Notes

### Other Observations (No Action Required)

**1. Mother Doc Child Document Reference is Outdated**
- **Location**: MLPlanningv2.md Line 788-789
- **Current Text**:
  ```markdown
  **Child Documents**:
  - MLCheckpointResume.md (checkpoint/resume system design)
  ```
- **Actual Child Doc**: VideoProcessingCHILD.md (not MLCheckpointResume.md)
- **Impact**: Low priority - Mother references non-existent child doc, but doesn't block TI
- **Note**: This could be addressed in future Mother doc update if needed

**2. Mother Doc Mentions Future TI Document**
- **Location**: MLPlanningv2.md Line 791-792
- **Text**: `VideoProcessingTI.md (rumiai_runner integration, checkpoint logic, error handling)`
- **Status**: This is correct - TI document will be generated from VideoProcessingCHILD.md
- **No action needed**: This is forward-looking planning, not an error

---

## Cascade: Child Docs Requiring Re-Audit

**If Option A (Fix Child Doc Only) is chosen**:
- No re-audit needed - this is a reference fix, not a design change

**If Option B (Add Section Anchors) is chosen**:
- [ ] VideoProcessingCHILD.md - Update to use anchors instead of line numbers
- [ ] VideoDiscoveryCHILD.md - Update Mother references to use anchors (if applicable)
- [ ] FeatureAggregationCHILD.md - Update Mother references to use anchors (if applicable)
- [ ] All other Child docs with MLPlanningv2.md references

---

**Completion Criteria**:
- [x] VideoProcessingCHILD.md Line 3 updated (line number reference → section header)
- [x] VideoProcessingCHILD.md Line 1398 updated (line number reference → section header)
- [ ] (Deferred) Section anchors added to MLPlanningv2.md
- [ ] (Deferred) All Child docs updated to use section anchors
- [x] MotherSync status updated to APPLIED

---

## Additional Issues Identified (Not Fixed)

**5 Line Number References to SystemArchitecturev2.md (External Doc)**:
- Line 603: `SystemArchitecturev2.md (lines 395-460)`
- Line 671: `SystemArchitecturev2.md (lines 395-460)`
- Line 1009: `SystemArchitecturev2.md lines 395-460`
- Line 1436: `lines 1-534`
- Line 1437: `SystemArchitecturev2.md (lines 395-460)`

**Status**: Not fixed (external documentation, stability unknown)
**Decision**: User opted to fix only CRITICAL Mother document references
**Future Action**: Consider converting these to section headers if SystemArchitecturev2.md changes frequently
