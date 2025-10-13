# Clarification Q&A: File Organization (Stage 2.5)

> **Mother Doc**: MLPlanningv2.md Section 2.5 "File Organization (Bucket Assignment)"
> **Phase 1**: Critique_FileOrganization.md
> **Date**: 2025-01-09
> **Status**: IN PROGRESS

## Questions by Category

### Input/Output Contracts

#### Q1: [CRITICAL] Input Source Directory & File Pattern
**Question**: What is the exact absolute path to the source directory where Stage 2 outputs temporal_windows_updated.json files? Is it Option A (`/home/jorge/rumiaifinal/insights/`), Option B (analysis-specific path), or something else? Also confirm exact filename pattern.

**Answer**: Option A - `/home/jorge/rumiaifinal/insights/` (single global directory for all videos)

**Filename Pattern**: Confirmed as `{video_id}_temporal_windows_updated.json` where `{video_id}` is the TikTok video ID (e.g., `7428596413707144481_temporal_windows_updated.json`)

**For HLD Section**: 5.1 (Input Schema), 3.1 (Input Dependencies)

**Notes**: This confirms Stage 2 (rumiai_runner.py) currently saves all outputs to a single flat directory regardless of client/hashtag/bucket. Stage 2.5 will read from this global location and organize into bucket-specific directories.

### Dependencies & Integration

### Edge Cases & Validation

### Performance & Scale

### Error Handling

### Testing

## Completeness Check

[Will be filled at end - see Step 6]

## Proceed to Phase 3

[Will be filled at end - see Step 6]
