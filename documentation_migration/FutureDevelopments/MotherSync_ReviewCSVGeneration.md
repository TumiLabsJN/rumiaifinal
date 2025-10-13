# Mother Document Sync: Proposed Changes from ReviewCSVGeneration Work

> **Trigger**: Child HLD work revealed Stage 2.4 should be relocated to Stage 3.4 with simplified design
> **Component**: Review CSV Generation (formerly Pipeline Validation)
> **Phase Outputs Reviewed**:
>   - Critique_ReviewCSVGeneration.md (Phase 1)
>   - QA_ReviewCSVGeneration.md (Phase 2)
>   - ReviewCSVGenerationCHILD.md (Phase 3)
> **Date**: 2025-01-09
> **Status**: PENDING APPROVAL

## Summary

**Total Changes Proposed**: 2
**Impact Scope**:
- Level 1 (Single Component): 2 changes (Stage 2.4 removal, Stage 3.4 addition)

**Affected Child Docs**:
- ReviewCSVGenerationCHILD.md (will need Section 1.2 reference updated after Mother change)

## Proposed Changes

### Change 1: [Outdated Info] Remove Stage 2.4: Pipeline Validation

**Issue Type**: Outdated Information

**Current State**:
- **Mother Section**: Stage 2.4: Pipeline Validation (lines 799-916)
- **Current Text**:
  ```
  ## Stage 2.4: Pipeline Validation

  **Purpose**: Detect feature outliers and edge cases in real-time to prevent bad data from entering ML training

  **Input**:
  - `temporal_windows_updated.json` per video (from Stage 2.2)
  - `validation/rolling_stats.json` (bucket-level running statistics)

  **Process**:

  ### 2.4.1: Rolling Statistics Tracking
  [complex automated validation with IQR outliers, z-scores, investigation packages, notifications]
  ```

**Problem Discovered**:
- **During**: Phase 1 Business Critique
- **Evidence**:
  - Critique_ReviewCSVGeneration.md Q1-Q6 conversation revealed simpler approach
  - User decided: "Is there a way to just have an Excel of the analyzed videos, for me to manually revise the outliers?"
  - Phase 1 Decision: Create separate `video_review.csv` for manual Excel review instead of automated detection
  - User confirmed: "Yes, this is sexy" (approving simpler approach)
  - Phase 2 Q3: Two separate CSV files decided (aggregated_features.csv for ML, video_review.csv for human review)

**Proposed Update**:
```markdown
REMOVE entire Stage 2.4 section (lines 799-916)

Rationale: Original design (complex rolling stats + anomaly detection) replaced with simpler approach (CSV export for manual Excel review). New design belongs in Stage 3.4 since primary work happens during Feature Aggregation, not Video Processing.
```

**Rationale**:
- Phase 1 Critique revealed 90% complexity reduction possible with Excel-based manual review
- Automated detection system (rolling stats, IQR, z-scores, investigation packages) is over-engineered for the use case
- User confirmed manual review workload is acceptable
- Main implementation happens in Stage 3 (CSV generation), not Stage 2

**Impact Scope**: Level 1 - Only ReviewCSVGenerationCHILD.md references this (will update to reference Stage 3.4)

**Priority**: HIGH

---

### Change 2: [Missing Content] Add Stage 3.4: Review CSV Generation

**Issue Type**: Missing Foundation Content (relocated from Stage 2.4)

**Current State**:
- **Mother Section**: Stage 3 has NO subsections (only "Stage 3: Feature Aggregation" at line 918)
- **Current Text**: Stage 3.4 doesn't exist

**Problem Discovered**:
- **During**: Phase 2 Q&A and Phase 3 Child HLD generation
- **Evidence**:
  - ReviewCSVGenerationCHILD.md Section 1.2 states: "Stage 3.4: Review CSV Generation [THIS COMPONENT]"
  - Phase 1 Critique decided component belongs in Stage 3, not Stage 2 (User: "Option A" - move to Stage 3.4)
  - User feedback: "Since we modified this development, should it still really go in phase 2.4? It may not be compatible at that stage of processual flow"
  - The primary deliverable (video_review.csv) is generated in Stage 3, alongside aggregated_features.csv

**Proposed Update**:
```markdown
## Stage 3.4: Review CSV Generation

**Purpose**: Generate video_review.csv for manual outlier investigation in Excel

**Why Separate from Stage 3.1-3.3?**:
- Stage 3.1-3.3 generates `aggregated_features.csv` (ML training input, ~65-215 columns)
- Stage 3.4 generates `video_review.csv` (human review, same features + url column)
- Review CSV is OPTIONAL - deleting it doesn't impact ML pipeline

**Input**:
- `temporal_windows_updated.json` (N files per bucket, with metadata.url)
- Note: Requires Stage 2 modification - temporal_compute.py must include `url` in calculated_metadata

**Process**:
1. Load all temporal_windows_updated.json files for bucket
2. Extract features (same logic as aggregated_features.csv)
3. Check metadata.url presence (skip videos with missing url, log warning)
4. Build CSV rows: [video_id, url, duration, all_features]
5. Save as `bucket_{duration}/validation/video_review.csv`

**Output**:
- `bucket_{duration}/validation/video_review.csv`
- Row count: N videos (same as aggregated_features.csv, minus videos with missing url)
- Column count: ~67-217 columns (video_id + url + duration + all temporal features)

**User Workflow**:
1. Open video_review.csv in Excel
2. Apply conditional formatting to highlight outliers (Excel built-in feature)
3. Click `url` column to watch flagged videos on TikTok
4. Investigate why outliers occurred (encoding issues, edge cases, RumiAI bugs)
5. All videos still proceed to ML training (no exclusions)

**Stage 2 Prerequisite**:
Modify `temporal_compute.py` (line ~2650) to pass url through metadata:
```python
calculated_metadata = {
    'video_id': video_id,
    'duration': video_duration,
    'url': metadata.get('url'),  # ← ADD THIS LINE
    'digg_count': metadata.get('likes', 0),
    ...
}
```

**Error Handling**:
- Videos with missing url: Skip from review CSV, log warning, still included in aggregated_features.csv
- All videos missing url: Log error, skip video_review.csv generation, continue pipeline
- Disk full: Fail fast

**Child Documents**:
- ReviewCSVGenerationCHILD.md (complete HLD with schemas, tests, pseudocode)

**Future TI Document**:
- ReviewCSVGenerationTI.md (implementation of dual CSV generation logic)

**Related Features**:
- Phase 1: Manual Outlier Investigation (simplified from automated Pipeline Validation)
```

**Rationale**:
- Relocates simplified component to correct stage (Stage 3 where CSV generation happens)
- Documents the dual-CSV pattern (ML vs human review separation)
- Clarifies that this is an investigation tool, not a data quality gate
- Shows user workflow (Excel conditional formatting + clickable URLs)
- Notes Stage 2 prerequisite (temporal_compute.py modification)

**Impact Scope**: Level 1 - Only ReviewCSVGenerationCHILD.md references this (will verify Section 1.2 matches)

**Priority**: HIGH

---

## Change Summary by Priority

### [CRITICAL] Changes (must apply)
None

### [HIGH] Changes (should apply)
1. Change 1: Remove Stage 2.4 (outdated complex validation)
2. Change 2: Add Stage 3.4 (simplified review CSV generation)

### [LOW] Changes (optional)
None

## Recommended Action

**Option B: Apply [CRITICAL] + [HIGH]** (recommended)
- Remove Mother Stage 2.4 (lines 799-916)
- Add Mother Stage 3.4 (after line 1040, before Stage 4)
- Verify ReviewCSVGenerationCHILD.md Section 1.2 references are correct
- Estimated effort: 15 minutes (Mother update + Child verification)

**Alternative Options**:

**Option A: Apply All Changes** (same as Option B - only 2 changes total)

**Option C: Apply [CRITICAL] Only** (skip both changes - no critical issues)

**Option D: Reject Changes**
- Keep Mother Stage 2.4 as-is (complex automated validation)
- Child HLD will contradict Mother doc (Child implements simpler approach)
- Future developers may be confused by mismatch

## User Decision

**Selected Option**: A (Apply all changes)

**Changes to Apply**: 1, 2 (both)

**Status**: APPLIED

**Applied Date**: 2025-01-09

---

## Additional Context

### Why This Relocation Makes Sense

**Original Design (Stage 2.4)**:
- Real-time anomaly detection DURING video processing
- Rolling statistics updated per video
- Investigation packages created immediately
- Notifications sent during processing

**Simplified Design (Stage 3.4)**:
- Batch CSV generation AFTER all videos processed
- No real-time detection (manual Excel review instead)
- No investigation packages (user clicks URL to watch video)
- No automated notifications (user applies conditional formatting)

**Key Difference**: Original was **reactive automation** (detect and flag during processing). New design is **proactive human review** (export data, user investigates manually).

**Stage Fit**:
- Stage 2 is about individual video processing (per-video operations)
- Stage 3 is about aggregation across videos (batch operations on N videos)
- New design aggregates ALL videos into one CSV → belongs in Stage 3

### Design Evolution Summary

**Phase 1 Findings**:
- Automated detection system complexity: 90% higher than needed
- User prefers manual Excel review (built-in conditional formatting)
- Investigation packages unnecessary (user can watch TikTok URL directly)

**Phase 2 Decisions**:
- Separate video_review.csv from aggregated_features.csv (ML purity)
- Include ALL features (user flexibility)
- Skip videos with missing url (log warning only)
- Mirror ML training data exactly (real reflection principle)

**Phase 3 Implementation**:
- 2 code changes: temporal_compute.py (1 line), Stage 3 (dual CSV generation)
- Complexity: ~100 lines vs ~500+ lines (original design)
- No new dependencies (just pandas, json, pathlib)

---

## Cascade: Child Docs Requiring Re-Audit

**Due to Mother MLPlanningv2.md updates:**

- [✓] ReviewCSVGenerationCHILD.md - Impact: Verified Section 1.2 and 10.1 already reference Stage 3.4 correctly

**Re-audit Results**:
1. ✅ Section 1.2 pipeline diagram: Correctly shows "Stage 3.4: Review CSV Generation [THIS COMPONENT]"
2. ✅ Section 10.1 references: Correctly lists "MLPlanningv2.md Section 3.4: Review CSV Generation (this component)"
3. ✅ No changes needed - Child HLD was written anticipating Stage 3.4 placement

**Conclusion**: Child HLD fully synchronized with Mother doc. No further updates required.

---

**Document Metadata**
- Version: 1.0
- Created: 2025-01-09
- Status: PENDING APPROVAL
- Impact: 1 Mother section removal, 1 Mother section addition, 1 Child doc verification
