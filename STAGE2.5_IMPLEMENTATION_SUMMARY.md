# Stage 2.5: File Organization - Implementation Summary

**Implementation Date**: 2025-10-13
**Source Specification**: FileOrganizationCHILDTI.md
**Status**: ✅ Complete - Ready for Integration Testing

---

## Overview

Stage 2.5 implements checkpoint-driven batch file organization with detection-based resume. It organizes temporal_windows files from the flat `/insights/` directory into bucket-specific directories, enabling Stage 3 (Feature Aggregation) to process videos efficiently within their duration-specific groups.

## What Was Implemented

### Core Modules Created

```
/home/jorge/rumiaifinal/ml_pipeline/stage2_5_organize/
├── __init__.py              # Package exports
├── main.py                  # Main orchestration (stage_2_5_file_organization_main)
├── file_organizer.py        # Core functions (4 main functions)
└── validation.py            # Input/output validation
```

### Key Features

1. **Checkpoint-Driven Discovery**
   - Reads `winner_analysis.json` to determine which 3 buckets to process
   - Reads Stage 2 checkpoints to get completed video IDs
   - Only organizes files that Stage 2 successfully processed

2. **Detection-Based Resume**
   - No Stage 2.5 checkpoint needed (checks filesystem state)
   - Automatically skips already-organized files
   - Idempotent (safe to re-run without side effects)

3. **Data Integrity Validation**
   - Detects duplicate video IDs across buckets (fail-fast)
   - Validates checkpoint schemas
   - Handles partial Stage 2 completion gracefully

4. **File Operations**
   - Atomic file moves (same filesystem)
   - Creates target directories automatically
   - Handles missing files gracefully (warning, continue)

---

## Architecture Integration

### How It Works

```
Stage 2 Output (FLAT)
└── /home/jorge/rumiaifinal/insights/
    ├── 7428596413707144481_temporal_windows_updated.json
    ├── 7428596413707144482_temporal_windows_updated.json
    └── ... (300 files, mixed durations across all buckets)

Stage 2.5 Processes:
1. Load winner_analysis.json → ["18-33s", "33-60s", "13-18s"]
2. Load Stage 2 checkpoints per winning bucket → completed_video_ids
3. Build file list (video_id → source/target paths)
4. Validate no duplicates
5. Move files to bucket directories

Stage 2.5 Output (ORGANIZED)
└── /data/clients/acme/hashtags/nutrition/top_contrastive/buckets/
    ├── bucket_18-33s/analysis/insights/
    │   ├── 7428596413707144481_temporal_windows_updated.json
    │   └── 7428596413707144482_temporal_windows_updated.json
    ├── bucket_33-60s/analysis/insights/
    │   └── 7428596413707144483_temporal_windows_updated.json
    └── bucket_13-18s/analysis/insights/
        └── 7428596413707144484_temporal_windows_updated.json

Stage 3 (Feature Aggregation)
└── Reads bucket-organized files ✓
```

---

## Data Flow

### Input (from Stage 1 & Stage 2)
```json
// File 1: winner_analysis.json (from Stage 1.3)
{
  "top_3_buckets": ["18-33s", "33-60s", "13-18s"],
  "top_100_distribution": {"18-33s": 45, "33-60s": 30, "13-18s": 20},
  "winner_coverage": 95.0
}

// File 2: stage_2_checkpoint.json (per winning bucket, from Stage 2)
{
  "stage": "video_processing",
  "bucket": "18-33s",
  "total_videos": 100,
  "completed": 98,
  "failed": 2,
  "status": "completed",
  "completed_video_ids": ["7428596413707144481", "7428596413707144482", ...]
}

// File 3: temporal_windows_updated.json (flat directory, from Stage 2)
Location: /home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json
Schema: 60+ features per temporal window (not modified by Stage 2.5)
```

### Output (to Stage 3)
```
Organized files by bucket:
{
  "18-33s": [
    "/data/.../bucket_18-33s/analysis/insights/7428596413707144481.json",
    "/data/.../bucket_18-33s/analysis/insights/7428596413707144482.json"
  ],
  "33-60s": [
    "/data/.../bucket_33-60s/analysis/insights/7428596413707144483.json"
  ],
  "13-18s": [
    "/data/.../bucket_13-18s/analysis/insights/7428596413707144484.json"
  ]
}

Organization summary:
{
  "moved_count": 150,
  "skipped_already_organized": 50,
  "missing_count": 2,
  "total_processed": 202,
  "winning_buckets": ["18-33s", "33-60s", "13-18s"]
}

Empty source directory:
/home/jorge/rumiaifinal/insights/ → (empty - all files moved)
```

---

## Usage Example

### From Python (Pipeline Orchestration)
```python
from ml_pipeline.stage2_5_organize import stage_2_5_file_organization_main

# After Stage 2 completes, organize files by bucket
analysis_base = "/data/clients/acme/hashtags/nutrition/top_contrastive/"

summary = stage_2_5_file_organization_main(analysis_base)

print(f"Moved: {summary['moved_count']} files")
print(f"Skipped: {summary['skipped_already_organized']} files (already organized)")
print(f"Missing: {summary['missing_count']} files")
print(f"Winning buckets: {summary['winning_buckets']}")
```

### Command Line (Future Enhancement)
```bash
# Organize files after Stage 2 completes
python -m ml_pipeline.stage2_5_organize \
  --analysis-base /data/clients/acme/hashtags/nutrition/top_contrastive/
```

---

## Resume Example

### Scenario: Organization Interrupted at File 150/300

**First Run** (interrupted):
```bash
# Start organizing 300 files
→ Moved: video_001 → 18-33s (1/300)
→ Moved: video_002 → 18-33s (2/300)
...
→ Moved: video_150 → 33-60s (150/300)
[Power outage - process killed]
```

**Resume** (automatic, no flag needed):
```bash
# Re-run same command
✓ Loaded winning buckets: ['18-33s', '33-60s', '13-18s']
✓ Built file list: 300 files across 3 buckets

DEBUG: Already organized: video_001 → 18-33s
DEBUG: Already organized: video_002 → 18-33s
...
DEBUG: Already organized: video_150 → 33-60s
INFO: Moved: video_151 → 13-18s (1/300)
...

Organization complete:
  Total files:  300
  Moved:        150  ← Only remaining files
  Already done: 150  ← Detected automatically
  Missing:      0
  Processed:    300/300
```

---

## Error Handling

### Fail-Fast Errors

**Missing winner_analysis.json**:
```
FileNotFoundError: winner_analysis.json not found at:
  /data/clients/acme/hashtags/nutrition/top_contrastive/winner_analysis.json

This file is created by Stage 1.3 (Winner Analysis).
Stage 2.5 requires this file to know which buckets to organize.

Solutions:
  1. Complete Stage 1 (Video Discovery & Winner Analysis)
  2. Check if Stage 1 completed successfully
  3. Verify analysis_base path is correct
```

**Duplicate Video ID Across Buckets**:
```
ValueError: Video ID '7428596413707144481' appears in multiple buckets:
  - Bucket: 18-33s
  - Bucket: 33-60s

This indicates checkpoint corruption or Stage 2 bug.
Each video should belong to exactly one bucket based on duration.

Solutions:
  1. Re-run Stage 2 to regenerate checkpoints
  2. Manually inspect checkpoints and remove duplicate entries
```

### Non-Fatal Errors

**Missing Source File**:
```
WARNING: Missing source and target for video 7428596413707144481.
Stage 2 checkpoint indicated completion, but file doesn't exist.
→ Continuing with other files...
```

**File Move Failure** (permissions):
```
ERROR: Failed to move 7428596413707144482: Permission denied
→ Continuing with other files...
```

---

## Performance Characteristics

### Expected Processing Time

Based on FileOrganizationCHILDTI.md Section 7.1:

| Metric | N=100 files | N=300 files | Notes |
|--------|-------------|-------------|-------|
| Load winner_analysis.json | < 0.1s | < 0.1s | Small JSON file |
| Load checkpoints (3 buckets) | < 0.3s | < 0.3s | 3 small JSON files |
| File existence checks | 0.5s | 1.5s | Fast filesystem checks |
| File moves | 1-2s | 3-4s | Atomic renames within filesystem |
| **Total time** | **2-3s** | **5-6s** | Linear scaling |

**Key Assumption**: Source `/insights/` and target `/data/` on **same filesystem**
- Same filesystem: File move is atomic rename (instant)
- Different filesystems: File move is copy+delete (10x slower)

### Bottlenecks

| Bottleneck | Impact | Mitigation | Priority |
|------------|--------|------------|----------|
| File moves across filesystems | 10x slower | Ensure `/insights/` and `/data/` on same filesystem | High |
| Sequential file processing | Linear scaling | Acceptable for current scale (300 files in ~5s) | Low |

---

## Testing Recommendations

### Unit Tests Needed

```python
# ml_pipeline/tests/test_stage2_5_organize.py

def test_load_winning_buckets_valid():
    """Test loading valid winner_analysis.json"""
    pass

def test_load_winning_buckets_missing():
    """Test FileNotFoundError raised when winner_analysis.json missing"""
    pass

def test_load_winning_buckets_invalid_schema():
    """Test ValueError raised when schema invalid"""
    pass

def test_build_file_list_complete_checkpoints():
    """Test building file list from completed checkpoints"""
    pass

def test_build_file_list_partial_checkpoints():
    """Test warning logged for status='in_progress' checkpoints"""
    pass

def test_detect_duplicates_none():
    """Test validation passes when no duplicates"""
    pass

def test_detect_duplicates_detected():
    """Test ValueError raised when duplicate video_id detected"""
    pass

def test_organize_files_fresh_run():
    """Test organizing files from scratch (all files moved)"""
    pass

def test_organize_files_resume():
    """Test detection-based resume (skips already-organized files)"""
    pass

def test_organize_files_missing_source():
    """Test warning logged for missing source file"""
    pass

def test_validate_inputs_valid():
    """Test input validation passes for valid inputs"""
    pass

def test_validate_inputs_missing_analysis_base():
    """Test ValueError raised for missing analysis_base"""
    pass

def test_validate_output_valid():
    """Test output validation passes for valid organized files"""
    pass
```

### Integration Tests Needed

```python
# ml_pipeline/tests/test_stage2_5_integration.py

def test_end_to_end_organize_5_files():
    """
    Test complete file organization flow with 5 real files.

    Setup:
    - Create test winner_analysis.json (1 bucket: 18-33s)
    - Create test stage_2_checkpoint.json (5 completed videos)
    - Create 5 test temporal_windows_updated.json files in flat directory

    Verify:
    - All 5 files moved to bucket_18-33s/analysis/insights/
    - Flat directory empty
    - Stage 3 can read organized files
    """
    pass

def test_resume_after_interruption():
    """
    Test auto-resume without --resume flag.

    Setup:
    - Organize 10 files, interrupt after 5
    - Re-run Stage 2.5

    Verify:
    - First 5 files detected as already organized (skipped)
    - Only remaining 5 files moved
    - No duplicates created
    """
    pass

def test_missing_winner_analysis():
    """
    Test fail-fast on missing winner_analysis.json

    Setup:
    - Delete winner_analysis.json

    Verify:
    - FileNotFoundError raised with clear message
    """
    pass

def test_duplicate_video_id():
    """
    Test fail-fast on duplicate video ID across buckets.

    Setup:
    - Manually create corrupted checkpoints with same video_id in 2 buckets

    Verify:
    - ValueError raised with detailed error message
    """
    pass
```

---

## Next Steps

### Immediate Tasks

1. **Integration with Stage 2**
   - Call Stage 2.5 after Stage 2 completes for each winning bucket
   - Pass `analysis_base` path to `stage_2_5_file_organization_main()`

2. **Update Main Pipeline Orchestrator**
   - Modify `rumiai_ml_batch.py` to orchestrate Stage 1 → Stage 2 → **Stage 2.5** → Stage 3

3. **Testing**
   - Unit tests for all functions
   - Integration test with 5 real temporal_windows files
   - Resume behavior test

4. **Documentation**
   - Update QUICK_REFERENCE.md with Stage 2.5 usage
   - Add troubleshooting guide

### Future Enhancements (from FileOrganizationCHILD.md Section 9)

- **Phase 2**: Parallel file moves (ThreadPoolExecutor for N > 500 files)
- **Phase 3**: Verification mode (`--verify` flag for post-move integrity checks)
- **Phase 4**: Progress bar for user feedback
- **Phase 5**: Rollback capability for partial failures

---

## Dependencies

### Python Packages (Standard Library Only)
- `os` - File system operations
- `json` - JSON file loading
- `shutil` - File move operations
- `logging` - Stage execution logging

### External Services
- None (pure filesystem operations)

### Foundation Modules (Already Implemented)
- `foundation.paths` - Path construction (not strictly required, Stage 2.5 builds paths directly)
- `foundation.constants` - Bucket definitions

---

## File Locations

### Implementation Files
```
/home/jorge/rumiaifinal/ml_pipeline/stage2_5_organize/
├── __init__.py              (15 lines)
├── main.py                  (95 lines)
├── file_organizer.py        (290 lines)
└── validation.py            (70 lines)

Total: ~470 lines of production code
```

### Documentation
```
/home/jorge/rumiaifinal/
├── documentation_migration/FutureDevelopments/ChildDocs/
│   ├── FileOrganizationCHILD.md        (HLD - Design)
│   └── FileOrganizationCHILDTI.md      (TI - Implementation Spec)
├── STAGE2_BUGFIX_STAGE2.5_CLARIFICATION.md
└── STAGE2.5_IMPLEMENTATION_SUMMARY.md   (This file)
```

---

## Implementation Completeness

✅ **Completed** (from FileOrganizationCHILDTI.md):
- [x] Section 2: Stage Contract (StageInput, StageOutput)
- [x] Section 3: Data Schemas (all schemas)
- [x] Section 4: Algorithmic Specifications (all 4 functions)
- [x] Section 5: Validation Rules (input/output validation)
- [x] Section 6: Error Handling (all error conditions)
- [x] Section 8: File Structure & Integration
- [x] Section 9: Configuration & Environment

⏳ **Pending**:
- [ ] Section 13: Test Specifications (unit tests, integration tests)
- [ ] Integration with Stage 2 output
- [ ] Integration with Stage 3 input
- [ ] CLI wrapper (optional enhancement)

---

## Comparison: Stage 2 vs Stage 2.5

| Aspect | Stage 2 (Video Processing) | Stage 2.5 (File Organization) |
|--------|---------------------------|------------------------------|
| **Purpose** | Process videos through RumiAI | Organize outputs by bucket |
| **Input** | Video URLs (from Stage 1) | temporal_windows files (flat) |
| **Output** | Flat directory of JSON files | Bucket-organized JSON files |
| **Duration** | 8-10 hours (300 videos) | ~5 seconds (300 file moves) |
| **Resume** | Checkpoint per video | Detection-based (filesystem) |
| **Checkpoint** | Required (one per bucket) | Not needed (idempotent) |
| **Complexity** | High (subprocess, ML pipeline) | Low (pure file operations) |
| **Status** | ✅ Implemented & Fixed | ✅ Implemented |

---

## Contact & Support

**Implementation Source**: FileOrganizationCHILDTI.md (v1.0)
**Implementation Date**: 2025-10-13
**Implemented By**: Claude Code
**Review Status**: Pending integration testing

For questions or issues, reference:
- Technical specification: `documentation_migration/FutureDevelopments/ChildDocs/FileOrganizationCHILDTI.md`
- High-level design: `documentation_migration/FutureDevelopments/ChildDocs/FileOrganizationCHILD.md`
- This summary: `STAGE2.5_IMPLEMENTATION_SUMMARY.md`
