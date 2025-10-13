# Mother Document Sync: Proposed Changes from VideoDiscovery Work

> **Trigger**: Child HLD work revealed Mother doc missing interactive confirmation feature
> **Component**: VideoDiscovery (Stage 1)
> **Phase Outputs Reviewed**:
>   - VideoDiscoveryCHILD.md (recent update adding Section 2.4: Interactive Confirmation)
> **Date**: 2025-10-08
> **Status**: PENDING APPROVAL

## Summary

**Total Changes Proposed**: 3
**Impact Scope**:
- Level 1 (Single Component): 3 changes (Stage 1 only)

**Affected Child Docs**: VideoDiscoveryCHILD.md only (already updated)

## Proposed Changes

### Change 1: [Missing Foundation] Stage 1 - Add Section 1.5 "Interactive Confirmation"

**Issue Type**: Missing Foundation Content

**Current State**:
- **Mother Section**: Stage 1 (lines 588-683)
- **Current Text**:
  ```
  ### 1.4: Video Selection Per Bucket (Strategy-Specific)
  [ends at line 683]

  **Output**:
  - Selected video list (per bucket)
  - Typical: ~300 videos total (3 buckets × ~100 videos each)
  - Format: List of video URLs/IDs for Stage 2 processing

  [No Section 1.5 exists - jumps to Example Workflow at line 710]
  ```

**Problem Discovered**:
- **By Comparing**: VideoDiscoveryCHILD.md Section 2.4 vs MLPlanningv2.md Stage 1
- **Evidence**: Child doc (lines 754-964) has detailed interactive confirmation feature with:
  - Display format examples
  - Full pseudocode (2 functions: `confirm_bucket_selection()` and `show_detailed_bucket_analysis()`)
  - User action documentation (Y/n/details)
  - Edge cases table
  - Automation bypass with `--auto-confirm` flag
- **Mother doc**: No mention of this feature, Stage 1 appears to flow directly to Stage 2

**Proposed Update**:
```markdown
### 1.5: Interactive Confirmation

After bucket selection completes, CLI displays summary and prompts user to confirm before proceeding to Stage 2:

```
Stage 1 Complete: Video Discovery & Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Selected Buckets (by winner concentration):

  1. 15-30s  →  28 videos  (32.0% of winners)
  2. 30-45s  →  24 videos  (24.0% of winners)
  3. 45-60s  →  20 videos  (16.0% of winners)

Total: 72 videos across 3 buckets

Proceed to Stage 2 (Download & Analysis)? [Y/n/details]
```

**Purpose**: Allow user to review bucket selection and abort before expensive operations (downloads, ML inference).

**User Options**:
- `Y` or `Enter`: Proceed to Stage 2
- `n`: Abort (exit code 130)
- `details`: Show full bucket analysis including runners-up

**Bypass**: Use `--auto-confirm` flag to skip prompt for CI/CD pipelines.

**Child Document**: VideoDiscoveryCHILD.md Section 2.4 (full implementation details)
```

**Rationale**:
- Adds cost control gate before expensive Stage 2 operations
- Provides transparency about what will be processed
- Enables user to verify bucket selection before committing resources
- Child doc has full implementation (210 lines), Mother needs high-level summary

**Impact Scope**: Level 1 - Only VideoDiscoveryCHILD.md references Stage 1 workflow

**Priority**: HIGH

---

### Change 2: [Missing Foundation] Stage 0 - Add `--auto-confirm` CLI flag

**Issue Type**: Missing Foundation Content

**Current State**:
- **Mother Section**: Stage 0 Command Structure (lines 250-264)
- **Current Text**:
  ```bash
  python rumiai_ml_batch.py \
    --client "client_name" \
    --analysis-type {hashtag|competitor|creator} \
    --target "{target}" \
    --analysis-mode {top|recent} \
    --selection-strategy {contrastive|top} \
    --video-count N \
    --date-filter last_N_days \
    --report-type {single|comparison} \
    --report-audience {client|internal|creator}
  ```

**Problem Discovered**:
- **By Comparing**: VideoDiscoveryCHILD.md Section 2.4 vs MLPlanningv2.md Stage 0
- **Evidence**: Child doc (line 840, 947, 950) references `--auto-confirm` flag:
  - "Skip prompt if auto-confirm enabled (CLI flag or config)"
  - "Use `--auto-confirm` CLI flag (see FoundationCHILD.md Section 4.1) to skip prompt"
  - "Enables unattended execution for CI/CD pipelines"
- **Mother doc**: Command structure doesn't list this flag

**Proposed Update**:
```markdown
python rumiai_ml_batch.py \
  --client "client_name" \
  --analysis-type {hashtag|competitor|creator} \    # Stage 0.1: Target Type
  --target "{target}" \
  --analysis-mode {top|recent} \                     # Stage 0.2: Analysis Mode
  --selection-strategy {contrastive|top} \           # Stage 0.3: Selection Strategy
  --video-count N \                                  # Stage 0.4: Video Count
  --date-filter last_N_days \                        # Stage 0.5: Date Filter
  --report-type {single|comparison} \                # Stage 0.6: Report Type
  --report-audience {client|internal|creator} \      # Stage 0.7: Report Audience
  --auto-confirm                                     # Skip interactive prompts (CI/CD)
```

**Rationale**:
- Completes CLI flag documentation
- Explains automation bypass for CI/CD pipelines
- Aligns Mother with Child implementation

**Impact Scope**: Level 1 - Only VideoDiscoveryCHILD.md uses this flag (Stage 1 confirmation prompt)

**Priority**: HIGH

---

### Change 3: [Outdated Info] Update Stage 1 Example Workflow

**Issue Type**: Outdated Information

**Current State**:
- **Mother Section**: Stage 1 Example Workflow (lines 710-721)
- **Current Text**:
  ```
  **Example Workflow**:
  ```
  Scraped: 800 videos (all-time)
  ↓ Apply date_filter: last_90_days
  Filtered: 600 videos (within date range)
  ↓ Analyze top 100 performers (success-based distribution)
  Top 100 winners: 18-33s (45%), 33-60s (30%), 13-18s (20%), 9-13s (5%)
  ↓ Select top 3 winning buckets
  Process: 18-33s, 33-60s, 13-18s (95% of winners)
  ↓ Apply selection strategy (contrastive, N=100)
  Per bucket: 100 videos (80 top + 20 bottom)
  ```
  ```

**Problem Discovered**:
- **By Comparing**: Example workflow ends with "Per bucket: 100 videos" but doesn't show interactive confirmation step
- **Evidence**: Workflow should show confirmation prompt before proceeding to Stage 2
- Current workflow implies automatic progression to Stage 2

**Proposed Update**:
```markdown
**Example Workflow**:
```
Scraped: 800 videos (all-time)
↓ Apply date_filter: last_90_days
Filtered: 600 videos (within date range)
↓ Analyze top 100 performers (success-based distribution)
Top 100 winners: 18-33s (45%), 33-60s (30%), 13-18s (20%), 9-13s (5%)
↓ Select top 3 winning buckets
Process: 18-33s, 33-60s, 13-18s (95% of winners)
↓ Apply selection strategy (contrastive, N=100)
Per bucket: 100 videos (80 top + 20 bottom)
↓ Interactive confirmation (unless --auto-confirm)
User reviews and confirms → Proceed to Stage 2
```
```

**Rationale**:
- Accurately reflects Stage 1 workflow including new confirmation step
- Shows where user interaction occurs in pipeline
- Documents automation bypass behavior

**Impact Scope**: Level 1 - Example workflow documentation only

**Priority**: LOW (nice-to-have for completeness)

---

## Change Summary by Priority

### [HIGH] Changes (should apply)
1. Change 1: Add Stage 1 Section 1.5 "Interactive Confirmation" (new processing step)
2. Change 2: Add `--auto-confirm` CLI flag to Stage 0 command structure (new flag)

### [LOW] Changes (optional)
1. Change 3: Update Stage 1 example workflow (documentation clarity)

## Recommended Action

**Option B: Apply [HIGH] Only** (recommended)
- Update Mother doc with 2 high-priority changes
- Skip [LOW] change for now (can add later)
- Re-audit VideoDiscoveryCHILD.md (verify references are valid)
- Estimated effort: 10-15 minutes (quick Mother updates + verification)

**Option A: Apply All Changes**
- Update Mother doc with all 3 proposed changes
- Re-audit VideoDiscoveryCHILD.md
- Estimated effort: 15-20 minutes

**Option C: Apply Change 1 Only**
- Add Section 1.5 only (most critical)
- Defer CLI flag documentation and example workflow
- Re-audit VideoDiscoveryCHILD.md
- Estimated effort: 5-10 minutes

**Option D: Reject Changes**
- Keep Mother doc as-is
- VideoDiscoveryCHILD.md documents interactive confirmation feature
- Mother remains high-level (no interactive confirmation detail)

## User Decision

**Selected Option**: D (Reject Changes)

**Changes to Apply**: None

**Status**: REJECTED

**Rejection Date**: 2025-10-08

**Rationale**:
- Interactive confirmation is a Stage 1-specific UX feature (not architectural)
- Does not affect data flow or Stage 2+ planning
- Properly documented in VideoDiscoveryCHILD.md Section 2.4 (210 lines)
- Mother doc remains focused on high-level architecture (what happens)
- Child doc documents implementation details (how it happens)
- No cross-stage impact - Stage 2 receives identical input format
- Maintains clean separation: Mother = architecture, Child = implementation

---

## Notes

**Why This Sync is Needed**:
- VideoDiscoveryCHILD.md was just updated with new interactive confirmation feature (Section 2.4)
- This feature is production-ready and should be reflected in Mother doc
- Without sync, Mother appears to show automatic Stage 1→2 progression (misleading)

**Impact on Other Stages**:
- No impact on other stages (Stage 1 feature only)
- Stage 2 receives same input format (selected_videos.json per bucket)
- Only Stage 1 workflow changes (adds confirmation gate)

**Alternative Approach** (if rejected):
- Keep Mother doc as high-level overview
- VideoDiscoveryCHILD.md documents full implementation
- Accept that Mother doesn't show all workflow details
