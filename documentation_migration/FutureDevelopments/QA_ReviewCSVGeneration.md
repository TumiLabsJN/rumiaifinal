# Clarification Q&A: Review CSV Generation

> **Mother Doc**: MLPlanningv2.md Section 3.4 "Review CSV Generation" (relocated from Section 2.4)
> **Phase 1**: Critique_ReviewCSVGeneration.md
> **Date**: 2025-10-08
> **Status**: COMPLETE

## Questions by Category

### Input/Output Contracts

#### Q1: [CRITICAL] What is the exact field name for the video URL in the Apify scrape output?

**Answer**: `webVideoUrl` (found in `/home/jorge/rumiaifinal/foundation/schemas.py:70`)

**For HLD Section**: 3.1 (Input Dependencies - tracing webVideoUrl from Stage 1 → Stage 3), 5.1 (Input Schema - data flow), 2.3 (Detailed Process - field mapping)

**Notes**: The field is defined as `webVideoUrl: str = Field(..., description="TikTok web URL")` in the VideoMetadata schema. This field comes from Apify's hashtag/profile scraper output and needs to be passed through to aggregated_features.csv in Stage 3.

### Dependencies & Integration

#### Q2: [CRITICAL] How should webVideoUrl (metadata.url) be passed from Stage 1 to Stage 3?

**Answer**: The TikTok URL already flows through the pipeline as `metadata.url`:

**Current Flow** (verified in codebase):
1. Stage 1: Apify scrapes video → VideoMetadata contains `webVideoUrl`
2. rumiai_runner.py: Maps `webVideoUrl` → `metadata.url` in unified_analysis.json
3. temporal_compute.py: Receives `metadata` from unified_analysis (Line 2516)
4. **Missing step**: temporal_compute.py needs to include `url` in `calculated_metadata` (currently not passed through)

**Verified Evidence**:
- unified_analysis.json contains: `metadata.url = "https://www.tiktok.com/@user/video/123"`
- temporal_compute.py Line 2643-2667: `calculated_metadata` is built from input metadata
- temporal_compute.py does NOT currently include `url` field in calculated_metadata

**Required Change**:
Add one line to temporal_compute.py `calculated_metadata` dict (around line 2650):
```python
calculated_metadata = {
    'video_id': video_id,
    'duration': video_duration,
    'url': metadata.get('url'),  # ← ADD THIS LINE
    'digg_count': metadata.get('likes', 0),
    ...
}
```

**Rationale for modifying Stage 2 output**:
- temporal_windows_updated.json already contains metadata passthrough (engagement counts, author, timestamps)
- Adding `url` follows existing pattern in temporal_compute.py
- Keeps Stage 3 simple: read ONE file instead of TWO (temporal + unified_analysis)
- Single source of truth for all video data (features + metadata)

**For HLD Section**: 2.3 (Detailed Process - 1-line code change in temporal_compute.py), 3.3 (Cross-Stage Dependencies - Stage 3 reads url from temporal_windows_updated.json), 5.2 (Output Schema - temporal_windows_updated.json metadata.url field)

### Edge Cases & Validation

#### Q3: [CRITICAL] Should URL be added to aggregated_features.csv or a separate review file?

**Answer**: Create a SEPARATE file for manual review to keep aggregated_features.csv clean for ML training.

**Problem with adding URL to aggregated_features.csv**:
- aggregated_features.csv feeds directly into ML models (Stage 5: Random Forest + K-Means)
- URL is non-numeric, non-categorical metadata - not a feature
- Would require dropping URL before ML or risk model contamination

**Approved Solution - Two Output Files from Stage 3**:

**File 1: `aggregated_features.csv`** (existing, ML training)
- Columns: video_id, duration, hook_scene_count, hook_word_count, ... (~65-215 features depending on bucket)
- Purpose: ML model training input (Stage 5)
- NO url column

**File 2: `video_review.csv`** (NEW, human review)
- Columns: video_id, url, duration, + subset of key features (10-20 features for outlier detection)
- Purpose: Manual Excel review with clickable URLs
- Location: `bucket_{duration}/validation/video_review.csv`
- Features included: scene_count, word_count, eye_contact_rate, emotion ratios, energy_level (most likely outlier indicators)

**User Workflow**:
1. Open `video_review.csv` in Excel
2. Apply conditional formatting to highlight outliers
3. Click `url` column to watch flagged videos on TikTok
4. Investigate why outliers occurred (encoding issues, edge cases, etc.)
5. All videos still proceed to ML training via aggregated_features.csv

**Rationale**:
- Separation of concerns: ML data vs human review data
- No preprocessing needed (aggregated_features.csv stays ML-ready)
- Cleaner architecture (review file is optional, can be deleted without impacting ML pipeline)

**For HLD Section**: 5.2 (Output Schema - two CSV files), 2.3 (Detailed Process - dual output generation), 3.2 (Output Contracts - file purposes)

#### Q4: [CRITICAL] Which features should be included in video_review.csv?

**Answer**: Include ALL features (same as aggregated_features.csv) plus url and video_id.

**video_review.csv structure**:
- Column 1: `video_id` (for reference)
- Column 2: `url` (clickable TikTok link)
- Column 3: `duration` (context)
- Columns 4-N: ALL temporal window features (hook_*, middle_*, closing_* - same as aggregated_features.csv)

**Rationale**:
- User can apply conditional formatting to ANY column to spot outliers
- No need to pre-select "important" features - user decides what to investigate
- Excel handles large column counts well (65-215 columns depending on bucket)
- Same data as ML input, just with url added for convenience

**Column count by bucket**:
- Bucket 0-3s, 3-9s: ~67 columns (video_id + url + ~65 features)
- Bucket 9-13s, 13-18s: ~157 columns (video_id + url + ~155 features)
- Bucket 18-33s: ~187 columns (video_id + url + ~185 features)
- Bucket 33-60s, 60-90s, 90-120s: ~217 columns (video_id + url + ~215 features)

**For HLD Section**: 5.2 (Output Schema - complete column list), 2.3 (Detailed Process - extract all features + url)

### Performance & Scale

[Questions will be filled iteratively]

### Error Handling

#### Q5: [HIGH] What happens if metadata.url is missing from temporal_windows_updated.json?

**Answer**: Option A - Skip that video row entirely (exclude from review CSV).

**Rationale**:
- If url is missing, the review CSV is useless for that video (can't click to watch)
- No point including a row that can't be investigated
- Keeps video_review.csv clean (only videos that CAN be reviewed)
- Log warning: "Video {video_id} excluded from video_review.csv - missing url"

**Error Handling Logic**:
```python
# In Stage 3 video_review.csv generation
for video in temporal_windows_jsons:
    url = video['metadata'].get('url')
    if not url:
        logger.warning(f"Video {video_id} excluded from review CSV - missing url")
        continue  # Skip this video
    # Add row to video_review.csv
```

**Impact**:
- aggregated_features.csv: Still includes this video (url not required for ML)
- video_review.csv: Excludes this video (url required for manual review)
- ML pipeline: Unaffected (video still trains)

**For HLD Section**: 6.2 (Error Cases - missing url handling), 6.1 (Input Validation - url presence check), 2.3 (Detailed Process - conditional row inclusion)

### Testing

#### Q6: [HIGH] What's a realistic test scenario for validating video_review.csv generation?

**Answer**: video_review.csv should be IDENTICAL to aggregated_features.csv except for the url column insertion. Both files reflect the ACTUAL videos selected per bucket.

**Real Reflection Principle**:
- If bucket_18-33s has 100 videos → both CSVs have 100 rows
- If bucket_33-60s has 80 videos → both CSVs have 80 rows
- Each bucket's video_review.csv = exact mirror of what goes into ML training for that bucket

**Test Validation** (per bucket):
1. Both files have same row count (same N videos)
2. Both files have same feature columns in same order
3. video_review.csv has ONE extra column: `url` (inserted at position 2, after video_id)
4. All feature values match between files (row-by-row comparison)
5. video_review.csv url column contains valid TikTok URLs

**Test Data Source**:
- Use existing bucket: `/home/jorge/rumiaifinal/data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s/`
- Process 5-10 videos through Stage 2 → temporal_windows_updated.json (with url in metadata)
- Run Stage 3 → generates both CSVs
- Verify both CSVs have same data, video_review.csv just adds url

**Expected Output Example** (bucket_18-33s, 10 test videos):
```
bucket_18-33s/ml_analysis/aggregated_features.csv: 10 rows × 185 columns
bucket_18-33s/validation/video_review.csv:        10 rows × 186 columns
                                                                ↑ url column added
```

**Why This Matters**:
- User reviews in Excel the EXACT data that ML trains on
- No surprises: outliers spotted in review = outliers in ML training
- Per-bucket: Each bucket's review file matches its ML input exactly

**For HLD Section**: 8.3 (Test Data - file comparison test per bucket), 8.1 (Unit Tests - CSV validation logic), 2.3 (Detailed Process - per-bucket dual CSV generation)

## Completeness Check

Can write these HLD sections without TODOs or gaps?

- [✓] **Section 2 (Architecture & Design)?**
  - 2.1: High-level approach - YES (simplified: add url to temporal_compute, generate video_review.csv in Stage 3)
  - 2.2: Data flow - YES (url flows: Apify → unified_analysis → temporal_compute → Stage 3)
  - 2.3: Detailed process - YES (2 code changes: temporal_compute.py line, Stage 3 dual CSV generation)

- [✓] **Section 3 (Dependencies & Integration)?**
  - 3.1: Input dependencies - YES (depends on temporal_windows_updated.json with metadata.url)
  - 3.2: Output contracts - YES (two CSV files: aggregated_features.csv, video_review.csv)
  - 3.3: Cross-stage dependencies - YES (Stage 2 outputs url → Stage 3 consumes it)
  - 3.4: External dependencies - NONE (no external services)

- [✓] **Section 5 (Data Schemas)?**
  - 5.1: Input schema - YES (temporal_windows_updated.json with metadata.url field)
  - 5.2: Output schema - YES (video_review.csv: video_id, url, duration, all features; ~186 columns for bucket 18-33s)

- [✓] **Section 6 (Error Handling)?**
  - 6.1: Input validation - YES (check metadata.url presence)
  - 6.2: Error cases - YES (missing url → skip video from review CSV, log warning)
  - 6.3: Output validation - MINIMAL (CSV row count verification)

- [✓] **Section 8 (Testing Strategy)?**
  - 8.1-8.3: Test cases - YES (row count match, feature value match, url column validation)

## Proceed to Phase 3

**Ready for HLD Generation**: YES

**All critical information gathered**:
- Q1: Field name is `webVideoUrl` (Apify) → `metadata.url` (unified_analysis)
- Q2: url flows through existing metadata chain, just needs passthrough in temporal_compute.py
- Q3: Two separate CSV files (ML vs human review) - cleaner architecture
- Q4: Include ALL features in video_review.csv for maximum flexibility
- Q5: Skip videos with missing url from review CSV (log warning)
- Q6: video_review.csv mirrors aggregated_features.csv exactly (same rows, same features, +url)

**Component Renamed**: "Pipeline Validation" → "Feature Quality Review" (investigation tool, not data gate)

**Status**: COMPLETE
