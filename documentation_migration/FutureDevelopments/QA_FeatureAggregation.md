# Clarification Q&A: Feature Aggregation

> **Mother Doc**: MLPlanningv2.md Stage 3 "Feature Aggregation"
> **Phase 1**: Critique_FeatureAggregation.md
> **Date**: 2025-01-09
> **Status**: IN PROGRESS

## Questions by Category

### Input/Output Contracts

#### Q1 [CRITICAL]: What is the exact file path pattern for input temporal_windows_updated.json files?

**Discovery Results**:

**Current State** (from file system exploration):
- `/home/jorge/rumiaifinal/insights/{video_id}_temporal_windows_updated.json` (flat structure, 8 files found)
- This appears to be for development/testing only

**Expected Production State** (from directory structure discovery):
- `/home/jorge/rumiaifinal/data/clients/{client_name}/{target_type}/{target_name}/{strategy}/buckets/bucket_{duration}/analysis/insights/{video_id}_temporal_windows_updated.json`
- Example: `/data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_13-18s/analysis/insights/7428596_temporal_windows_updated.json`

**Directory Structure Confirmed**:
```
data/clients/{client}/hashtags/{hashtag}/top_contrastive/buckets/
├── bucket_13-18s/
│   ├── analysis/
│   │   ├── insights/           ← temporal_windows_updated.json files
│   │   ├── service_debug/
│   │   └── unified/
│   ├── ml_analysis/            ← aggregated_features.csv OUTPUT
│   ├── validation/             ← video_review.csv OUTPUT (Stage 3.4)
│   ├── videos/
│   ├── selected_videos.json
│   └── ...
```

**Decision**: Stage 3 will read ONLY from bucket-organized directories.

**Rationale**:
- **Separation of concerns**: Stage 2 processes videos (no bucket knowledge), Stage 3 aggregates by bucket (assumes organized inputs)
- **Architectural consistency**: Mirrors Stage 1 pattern (organizes selected_videos.json by bucket)
- **Performance**: Stage 3 doesn't need to open every file to filter by duration

**NEW COMPONENT REQUIRED: Stage 2.5 (File Organization)**
- **Purpose**: Move temporal_windows_updated.json files from `/insights/` to bucket directories
- **Input**: `/insights/{video_id}_temporal_windows_updated.json` (flat, N files mixed)
- **Process**: Read duration → determine bucket → move to `bucket_{duration}/analysis/insights/`
- **Output**: Bucket directories populated, ready for Stage 3
- **When**: Runs ONCE after all Stage 2 video processing completes (batch operation)

**Stage 3 Input Path** (confirmed):
- `bucket_{duration}/analysis/insights/{video_id}_temporal_windows_updated.json`
- Example: `data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s/analysis/insights/7428596_temporal_windows_updated.json`

**For HLD Sections**:
- 1.2 (Where This Fits in Pipeline - add Stage 2.5 dependency)
- 3.1 (Input Dependencies - requires Stage 2.5 completion)
- 5.1 (Input Schema - bucket directory path)
- 10.4 (Related Components - note Stage 2.5 requirement)

#### Q2 [CRITICAL]: What is the exact output file path and how is Stage 3 invoked?

**Discovery Results**:

**Invocation Pattern Decision**: One invocation per bucket (Option A)

**Invocation Command**:
```bash
python3 scripts/stage3_aggregation.py \
  --bucket-path="data/clients/{client}/{target_type}/{target_name}/{strategy}/buckets/bucket_{duration}"
```

**Example**:
```bash
python3 scripts/stage3_aggregation.py \
  --bucket-path="data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s"
```

**Output Path**: `{bucket-path}/ml_analysis/aggregated_features.csv`

**Full Example Output Path**:
- `data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s/ml_analysis/aggregated_features.csv`

**Stage 3 Process**:
1. Receive `--bucket-path` as CLI parameter
2. Read all files from `{bucket-path}/analysis/insights/*_temporal_windows_updated.json`
3. Aggregate features per video (21 features × N windows = 45-150 columns)
4. Write to `{bucket-path}/ml_analysis/aggregated_features.csv`
5. Stage 3.4 generates `{bucket-path}/validation/video_review.csv` (optional, human review)

**Rationale for Option A (One Invocation Per Bucket)**:
- **Parallelization**: 3 qualified buckets can run simultaneously
  ```bash
  python3 scripts/stage3_aggregation.py --bucket-path="...bucket_18-33s" &
  python3 scripts/stage3_aggregation.py --bucket-path="...bucket_33-60s" &
  python3 scripts/stage3_aggregation.py --bucket-path="...bucket_60-90s" &
  ```
- **Failure Resilience**: One bucket fails, others continue (easy resume)
- **Architectural Consistency**: Stage 1 operates on individual buckets via `selected_videos.json`
- **Separation of Concerns**: Stage 3 doesn't need to know about "all buckets", just processes the one it's given
- **Debugging**: Easy to test single bucket

**Orchestration** (future consideration):
- Shell script or orchestration layer iterates through qualified buckets
- Launches N parallel Stage 3 invocations
- Handles retry logic for failed buckets

**Alternative Rejected** (Option B: One invocation for all buckets):
- Simpler orchestration BUT:
  - No parallelization (sequential processing)
  - Failure in one bucket stops all buckets
  - Harder to test/debug individual buckets
  - Stage 3 needs "all buckets" context awareness

**For HLD Sections**:
- 3.2 (Output Contracts - exact output path pattern)
- 5.2 (Output Schema - aggregated_features.csv structure)
- 2.1 (Invocation - CLI parameters)
- 2.3 (Detailed Process - how bucket context is determined)
- 6.1 (Parallel Processing - per-bucket invocation enables parallelization)

### Dependencies & Integration

#### Q3 [HIGH]: What are Stage 3's hard dependencies and how are they validated?

**Context**: Mother Stage 3 depends on Stage 2 completion. With the addition of Stage 2.5, dependencies have changed.

**Hard Dependencies**:

1. **Stage 2.5 Completion** (NEW - critical prerequisite)
   - **What**: All `temporal_windows_updated.json` files organized into bucket directories
   - **Why**: Stage 3 reads from `{bucket-path}/analysis/insights/*.json`
   - **Validation**: Check that `{bucket-path}/analysis/insights/` exists and contains files
   - **Failure Mode**: If directory empty or missing → Stage 3 should fail fast with clear error

2. **Stage 1 Completion** (existing)
   - **What**: `selected_videos.json` per bucket (defines which videos were selected)
   - **Why**: Stage 3 should only process videos that passed Stage 1 selection
   - **Question**: Does Stage 3 need to cross-reference `selected_videos.json`?
   - **Or**: Does Stage 2.5 already ensure only selected videos are in `analysis/insights/`?

3. **Bucket Directory Structure** (infrastructure)
   - **What**: `{bucket-path}/ml_analysis/` directory must exist or be creatable
   - **Validation**: Check write permissions before processing
   - **Failure Mode**: If cannot create directory → fail fast

**Dependency Validation Strategy**:

**Option A: Defensive Validation** (check all prerequisites)
```python
# Stage 3 startup checks
1. Validate --bucket-path parameter exists
2. Check {bucket-path}/analysis/insights/ exists and has *.json files
3. Check {bucket-path}/ml_analysis/ is writable (create if missing)
4. (Optional) Cross-reference selected_videos.json
5. If any check fails → exit with clear error message
```

**Option B: Fail Fast** (let errors surface naturally)
```python
# Stage 3 minimal checks
1. Validate --bucket-path parameter exists
2. Attempt to read insights directory (let FileNotFoundError surface)
3. Attempt to write output (let PermissionError surface)
```

**Decisions**:

**Decision 1: Cross-Reference selected_videos.json?**
- **Answer**: NO - Trust Stage 2.5 (Option A)
- **Rationale**: Stage 2.5's responsibility is to organize selected videos. If it organizes wrong videos, that's a Stage 2.5 bug. Stage 3 processes what it's given (separation of concerns).

**Decision 2: Validation Strategy**
- **Answer**: Defensive Validation (Option A)
- **Rationale**: Better user experience with clear error messages before processing begins
- **Example**: "❌ Stage 3 Error: No JSON files found in bucket_18-33s/analysis/insights/. Did Stage 2.5 complete?"
- vs generic: "FileNotFoundError: [Errno 2] No such file or directory"

**Stage 3 Startup Validation** (implementation):
```python
def validate_dependencies(bucket_path: Path):
    """Validate all prerequisites before processing"""

    # 1. Check bucket path exists
    if not bucket_path.exists():
        raise ValueError(f"Bucket path does not exist: {bucket_path}")

    # 2. Check insights directory exists and has files
    insights_dir = bucket_path / "analysis" / "insights"
    if not insights_dir.exists():
        raise ValueError(f"Insights directory missing: {insights_dir}. Did Stage 2.5 complete?")

    json_files = list(insights_dir.glob("*_temporal_windows_updated.json"))
    if len(json_files) == 0:
        raise ValueError(f"No temporal_windows_updated.json files found in {insights_dir}. Did Stage 2.5 complete?")

    # 3. Check ml_analysis directory is writable
    ml_analysis_dir = bucket_path / "ml_analysis"
    ml_analysis_dir.mkdir(parents=True, exist_ok=True)

    # Test write permissions
    test_file = ml_analysis_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        raise ValueError(f"Cannot write to {ml_analysis_dir}. Check permissions.")

    return len(json_files)  # Return count for logging
```

**For HLD Sections**:
- 1.2 (Where This Fits in Pipeline - dependency diagram)
- 3.1 (Input Dependencies - list all hard dependencies)
- 7.2 (Validation Strategy - startup checks with pseudocode)
- 8.1 (Error Handling - dependency failure modes)

### Edge Cases & Validation

#### Q4 [HIGH]: What happens if a bucket has mixed duration videos after Stage 2.5?

**Context**: Stage 2.5 reads duration from each `temporal_windows_updated.json` and moves it to the appropriate bucket directory. But what if duration doesn't match the bucket?

**Scenario**:
- File: `7428596_temporal_windows_updated.json`
- Contains: `metadata.duration = 20.5` (should go to bucket_18-33s)
- But Stage 2.5 puts it in: `bucket_13-18s/` (wrong bucket)

**Question**: Should Stage 3 validate that all videos in a bucket match the expected duration range?

**Option A: Trust Stage 2.5** (no validation)
- Assume Stage 2.5 organized correctly
- Process all videos in bucket regardless of actual duration
- Simpler, faster

**Option B: Validate and Skip** (defensive)
```python
# For bucket_18-33s, expected range: [18.0, 33.0]
for video_file in json_files:
    duration = data['metadata']['duration']
    if not (18.0 <= duration < 33.0):
        logger.warning(f"Video {video_id} duration {duration}s outside bucket range [18-33s]. Skipping.")
        continue  # Skip this video
```

**Option C: Validate and Fail**
```python
# Strict validation - fail entire bucket if ANY video mismatched
for video_file in json_files:
    duration = data['metadata']['duration']
    if not (18.0 <= duration < 33.0):
        raise ValueError(f"Video {video_id} has duration {duration}s, expected [18-33s]. Stage 2.5 bug?")
```

**Decision**: **Option A (Trust Stage 2.5)**

**Rationale**:
- **Separation of concerns**: Stage 2.5's job is correct organization. If it fails, that's a Stage 2.5 bug to fix there, not Stage 3's concern.
- **Simplicity**: Stage 3 processes the bucket it's given, trusting the organization is correct.
- **Architectural consistency**: Other stages trust their inputs (Stage 2 trusts Stage 1's selected_videos.json).
- **Focus**: Keeps Stage 3 focused on aggregation, not validation of upstream components.

**Implementation**: No duration range validation in Stage 3. Process all videos found in bucket directory.

**For HLD Sections**:
- 7.2 (Validation - NO duration range checks)
- 8.2 (Error Handling - trust upstream organization)

#### Q5 [MEDIUM]: How should Stage 3 handle videos with corrupted/malformed JSON files?

**Context**: From Phase 1 Critique, we decided "graceful error handling - skip videos with null middle_segments, log error, continue batch."

**Question**: What's the complete error handling strategy for individual video files during aggregation?

**Scenarios**:

1. **File exists but JSON is malformed** (syntax error)
   ```python
   # File contains: {"metadata": {"duration": 20.5, ...
   # Missing closing brace
   ```

2. **File is valid JSON but missing required fields**
   ```python
   # Missing metadata.duration
   # Missing temporal_windows
   # Missing middle_segments (decided: skip video, log error)
   ```

3. **File has null/None values in critical fields**
   ```python
   {"metadata": {"duration": null, ...}}
   {"temporal_windows": {"middle_segments": null}}
   ```

**Proposed Error Handling Strategy**:

```python
for video_file in json_files:
    try:
        # Load JSON
        with open(video_file) as f:
            data = json.load(f)

        # Extract video_id from filename
        video_id = video_file.stem.replace('_temporal_windows_updated', '')

        # Validate required fields exist
        if 'metadata' not in data:
            logger.error(f"Video {video_id}: Missing 'metadata'. Skipping.")
            continue

        if 'temporal_windows' not in data:
            logger.error(f"Video {video_id}: Missing 'temporal_windows'. Skipping.")
            continue

        # Check middle_segments (Phase 1 decision)
        middle_segments = data['temporal_windows'].get('middle_segments')
        if middle_segments is None or len(middle_segments) == 0:
            logger.error(f"Video {video_id}: null or empty middle_segments. Skipping.")
            continue

        # Extract features and add to aggregation
        features = extract_features(data)
        aggregated_data.append(features)

    except json.JSONDecodeError as e:
        logger.error(f"Video {video_file.name}: Malformed JSON - {e}. Skipping.")
        continue

    except Exception as e:
        logger.error(f"Video {video_file.name}: Unexpected error - {e}. Skipping.")
        continue

# After loop: check if we have ANY valid videos
if len(aggregated_data) == 0:
    raise ValueError(f"No valid videos processed in bucket {bucket_path}. Check logs.")
```

**Key Decisions**:
- **Individual file errors**: Skip video, log error, continue processing
- **All files fail**: Raise error (don't create empty CSV)
- **Partial success**: Create CSV with N valid videos (where N < total files)

**Decision**: Agreed

**Error Handling Strategy Confirmed**:
1. ✅ Skip individual bad videos, continue processing
2. ✅ Log errors for each skipped video (specific error messages)
3. ✅ Fail only if ALL videos fail (0 valid videos processed)
4. ✅ Allow partial success (create CSV with N valid videos, where N < total files)

**Implementation Notes**:
- Use try/except blocks for each video file
- Catch `json.JSONDecodeError` for malformed JSON
- Validate required fields: `metadata`, `temporal_windows`, `middle_segments`
- Check for null/empty middle_segments (Phase 1 decision)
- Final check: Raise error if `len(aggregated_data) == 0`

**For HLD Sections**:
- 8.2 (Error Handling - per-video error strategy with pseudocode)
- 2.3 (Detailed Process - error handling in aggregation loop)

### Performance & Scale

#### Q6 [MEDIUM]: What are the memory constraints for Stage 3 and how should large buckets be handled?

**Context**: Stage 3 loads all videos from a bucket into memory to create the aggregated CSV.

**Scale Estimates**:
- **Typical bucket**: 30-100 videos
- **Large bucket**: 200+ videos (possible for popular duration ranges)
- **Per-video data size**: ~50-100 KB (temporal_windows_updated.json file size)
- **In-memory representation**: ~150 columns × N videos

**Memory Calculation Example**:
```
Bucket with 100 videos, 150 features each:
- Raw JSON loading: 100 videos × 75 KB = 7.5 MB
- Pandas DataFrame: 100 rows × 150 columns × 8 bytes = 120 KB
- Total estimated: ~10 MB per bucket
```

**Question**: Should Stage 3 load all videos into memory at once, or use streaming/batching?

**Option A: Load All Into Memory** (current approach)
```python
aggregated_data = []
for video_file in json_files:
    data = json.load(video_file)
    features = extract_features(data)
    aggregated_data.append(features)

# Convert to DataFrame at end
df = pd.DataFrame(aggregated_data)
df.to_csv(output_path)
```

**Benefits**:
- Simple implementation
- Fast for typical buckets (30-100 videos)
- Pandas handles CSV writing efficiently

**Risks**:
- Very large buckets (1000+ videos) could cause memory issues
- But: Given bucket-specific models, unlikely to have 1000+ videos in one bucket

**Option B: Streaming/Batching**
```python
# Write CSV in chunks
with open(output_path, 'w') as f:
    writer = None
    for video_file in json_files:
        data = json.load(video_file)
        features = extract_features(data)

        if writer is None:
            writer = csv.DictWriter(f, fieldnames=features.keys())
            writer.writeheader()

        writer.writerow(features)
```

**Benefits**:
- Constant memory usage regardless of bucket size
- Can handle any number of videos

**Costs**:
- More complex code
- Harder to debug (can't inspect DataFrame before writing)
- Minimal benefit given expected bucket sizes

**Recommendation**: **Option A (Load All Into Memory)**

**Rationale**:
- Expected bucket sizes (30-200 videos) = ~10-20 MB memory (trivial)
- Simpler code, easier debugging
- Pandas provides better error handling and data validation
- If we ever hit memory issues with 1000+ video buckets, that indicates a modeling problem (bucket-specific models shouldn't have that many videos)

**Decision**: **Option A (Load All Into Memory)** - Agreed

**Rationale Confirmed**:
- Expected bucket sizes (30-200 videos) = ~10-20 MB memory (trivial for modern systems)
- Simpler implementation and debugging
- Pandas provides robust CSV writing and data validation
- If 1000+ video buckets occur, that indicates modeling issue (bucket-specific models shouldn't have that scale)

**Implementation**:
- Load all video JSON files into list of dictionaries
- Convert to pandas DataFrame
- Write to CSV in single operation
- No streaming/batching needed

**For HLD Sections**:
- 6.2 (Performance Considerations - memory usage is not a concern)
- 2.3 (Detailed Process - in-memory aggregation approach)

### Error Handling

#### Q7 [HIGH]: What happens if Stage 3 completes but the output CSV is corrupted or incomplete?

**Context**: Stage 3 writes `aggregated_features.csv`. What if the write operation partially fails?

**Scenarios**:

1. **Disk full during write**
   - DataFrame created successfully
   - `df.to_csv()` fails mid-write
   - Result: Partial CSV file exists (corrupted)

2. **Process killed during write**
   - SIGKILL, out-of-memory killer, system crash
   - Result: Partial CSV file exists

3. **Permission denied mid-write**
   - Write starts successfully
   - Permissions change during write
   - Result: Partial CSV file

**Question**: How should Stage 3 ensure output integrity?

**Option A: No Protection** (write directly)
```python
df.to_csv(output_path)  # If fails, partial file may exist
```

**Option B: Atomic Write** (write to temp, then rename)
```python
import tempfile
import shutil

# Write to temporary file first
temp_path = output_path.with_suffix('.tmp')
df.to_csv(temp_path)

# Atomic rename (only if write succeeded)
shutil.move(temp_path, output_path)
```

**Benefits of Option B**:
- If write fails, `aggregated_features.csv` doesn't exist (no partial file)
- Downstream Stage 4 won't see corrupted data
- Atomic rename on most filesystems (POSIX)

**Option C: Write + Validate**
```python
# Write file
df.to_csv(output_path)

# Validate it's readable and has correct shape
validation_df = pd.read_csv(output_path)
if len(validation_df) != len(df):
    raise ValueError("CSV validation failed - row count mismatch")
```

**Recommendation**: **Option B (Atomic Write)**

**Rationale**:
- Minimal code overhead (2 extra lines)
- Prevents downstream stages from consuming corrupted data
- Standard pattern for critical file writes
- Option C adds validation but doesn't prevent partial writes

**Decision**: **Option B (Atomic Write)** - Agreed

**Implementation Confirmed**:
```python
# Write to temporary file first
temp_path = output_path.with_suffix('.tmp')
df.to_csv(temp_path, index=False)

# Atomic rename (only if write succeeded)
shutil.move(temp_path, output_path)
```

**Benefits**:
- If write fails, `aggregated_features.csv` doesn't exist (no partial corruption)
- Downstream Stage 4 won't encounter corrupted data
- Atomic rename on POSIX filesystems
- Minimal code overhead (2 extra lines)

**Cleanup on Failure**:
- If `df.to_csv(temp_path)` fails, temp file may exist
- Add try/finally to clean up temp file on failure
```python
temp_path = output_path.with_suffix('.tmp')
try:
    df.to_csv(temp_path, index=False)
    shutil.move(temp_path, output_path)
finally:
    if temp_path.exists():
        temp_path.unlink()  # Clean up temp file if rename failed
```

**For HLD Sections**:
- 8.3 (Error Handling - output integrity with atomic write pattern)
- 2.3 (Detailed Process - atomic write implementation)

### Testing

#### Q8 [MEDIUM]: What test cases are needed to validate Stage 3 functionality?

**Context**: Need to define test strategy for Stage 3 before implementation.

**Test Categories**:

**1. Happy Path Tests**:
- **Test**: Single bucket with N valid videos → produces aggregated_features.csv
- **Validation**:
  - Output file exists at correct path
  - Row count = N videos
  - Column count matches bucket configuration (45/108/129/150)
  - All feature values are numeric (no NaN from extraction errors)

**2. Error Handling Tests**:
- **Test 2a**: Bucket with malformed JSON file
  - Expected: Skip bad file, log error, process remaining videos
  - Validation: Row count = N-1, error logged

- **Test 2b**: Bucket with video missing middle_segments
  - Expected: Skip video (Phase 1 decision), log error
  - Validation: Row count = N-1, error logged

- **Test 2c**: Bucket where ALL videos fail
  - Expected: Raise ValueError, no CSV created
  - Validation: Error raised, output file doesn't exist

**3. Edge Case Tests**:
- **Test 3a**: Bucket with single video (N=1)
  - Expected: CSV with 1 row

- **Test 3b**: Bucket with mixed window counts (2-window vs 5-window videos)
  - Note: Shouldn't happen if Stage 2.5 works correctly
  - Expected: Process all (per Q4 decision - trust Stage 2.5)

- **Test 3c**: Empty insights directory
  - Expected: Fail fast in validation (Q3 decision)

**4. Integration Tests**:
- **Test 4a**: Real temporal_windows_updated.json files from development
  - Use `/home/jorge/rumiaifinal/insights/*.json` (8 files)
  - Expected: Successful aggregation

- **Test 4b**: Verify feature extraction matches FeatureTransformation.md spec
  - Spot-check: hook_scene_count, middle_1_word_count, closing_audio_energy
  - Validation: Values match manual calculation

**5. Output Integrity Tests**:
- **Test 5a**: Simulate disk full during write
  - Expected: No partial CSV (atomic write), temp file cleaned up

- **Test 5b**: Verify CSV is readable by pandas
  - Expected: `pd.read_csv(output)` succeeds, shape matches

**Question**: Are there any additional test scenarios we should cover?

**For HLD Sections**:
- 9.1 (Testing Strategy - test categories)
- 9.2 (Test Cases - detailed scenarios)

## Completeness Check

### Questions Asked by Category

**Input/Output Contracts**: 3 questions
- Q1 [CRITICAL]: Input file path pattern → Decided: Stage 2.5 organizes files into bucket directories
- Q2 [CRITICAL]: Output file path and invocation → Decided: One invocation per bucket, write to `{bucket-path}/ml_analysis/`
- Q9 [LOW]: CSV column naming convention → Decided: Flat with underscores (`hook_scene_count`)

**Dependencies & Integration**: 1 question
- Q3 [HIGH]: Hard dependencies and validation → Decided: Trust Stage 2.5, defensive validation at startup

**Edge Cases & Validation**: 2 questions
- Q4 [HIGH]: Mixed duration videos in bucket → Decided: Trust Stage 2.5, no duration validation
- Q5 [MEDIUM]: Corrupted/malformed JSON handling → Decided: Skip bad videos, log errors, fail if all fail

**Performance & Scale**: 1 question
- Q6 [MEDIUM]: Memory constraints for large buckets → Decided: Load all into memory (10-20 MB trivial)

**Error Handling**: 3 questions
- Q7 [HIGH]: Output CSV corruption → Decided: Atomic write (temp file + rename)
- Q10 [LOW]: Logging strategy → Decided: Progress every 10 videos
- Q11 [LOW]: Duplicate video_ids → Decided: Keep first, skip duplicates with warning

**Testing**: 1 question
- Q8 [MEDIUM]: Test cases needed → Defined: 5 categories (happy path, error handling, edge cases, integration, output integrity)

**Observability**: 1 question
- Q12 [LOW]: Summary file generation → Decided: Generate aggregation_summary.json (separate from CSV)

**Feature Engineering**: 1 question
- Q13 [CRITICAL]: Cross-window feature derivation for video-level RF → Decided: Stage 4 derives explicit cross-window features

**Total Questions**: 13 questions across 7 categories

### Coverage Assessment

**Well-Covered Topics**:
- ✅ Input/output contracts (file paths, invocation pattern, column naming, summary file)
- ✅ Error handling strategy (graceful, atomic writes, duplicates)
- ✅ Dependencies (Stage 2.5, validation approach)
- ✅ Performance considerations (memory usage)
- ✅ Testing strategy (comprehensive test categories)
- ✅ Logging and observability (progress logs, summary file)
- ✅ Edge cases (corrupted JSON, duplicates, empty buckets, mixed durations)

**All Identified Gaps Addressed**: ✅
- Column naming: Q9 ✅
- Logging strategy: Q10 ✅
- Duplicate handling: Q11 ✅
- Summary/observability: Q12 ✅

---

#### Q9 [LOW]: What is the exact CSV column naming convention?

**Context**: Need to specify exact column names for aggregated_features.csv.

**Options**:

**Option A: Flat names with underscores**
```
video_id, duration, create_time, gender_detection,
hook_scene_count, hook_word_count, hook_audio_energy, ...,
middle_1_scene_count, middle_1_word_count, middle_1_audio_energy, ...,
middle_2_scene_count, middle_2_word_count, middle_2_audio_energy, ...,
closing_scene_count, closing_word_count, closing_audio_energy, ...
```

**Option B: Dotted notation**
```
video_id, duration, create_time, gender_detection,
hook.scene_count, hook.word_count, hook.audio_energy,
middle_1.scene_count, middle_1.word_count, middle_1.audio_energy,
closing.scene_count, closing.word_count, closing.audio_energy
```

**Option C: Prefixed categories**
```
video_id, meta_duration, meta_create_time, meta_gender_detection,
feat_hook_scene_count, feat_hook_word_count, feat_hook_audio_energy,
feat_middle_1_scene_count, feat_middle_1_word_count, feat_middle_1_audio_energy,
feat_closing_scene_count, feat_closing_word_count, feat_closing_audio_energy
```

**Recommendation**: **Option A (Flat with underscores)**

**Rationale**:
- Pandas-friendly (no issues with `df.hook_scene_count` access)
- Matches Python naming conventions (snake_case)
- Easy to read in Excel
- Consistent with common ML practice
- Avoids confusion with dots (could be interpreted as dict access)

**Column Ordering**:
1. `video_id` (primary key)
2. Metadata: `duration`, `create_time`, `gender_detection`
3. Hook features: `hook_scene_count`, `hook_word_count`, ...
4. Middle features: `middle_1_scene_count`, `middle_1_word_count`, ..., `middle_N_scene_count`, ...
5. Closing features: `closing_scene_count`, `closing_word_count`, ...

**Decision**: **Option A (Flat with underscores)** - Agreed

**Column Naming Convention Confirmed**:
```
video_id, duration, create_time, gender_detection,
hook_scene_count, hook_word_count, hook_audio_energy, ...,
middle_1_scene_count, middle_1_word_count, middle_1_audio_energy, ...,
middle_2_scene_count, middle_2_word_count, middle_2_audio_energy, ...,
closing_scene_count, closing_word_count, closing_audio_energy, ...
```

**Implementation**:
- Snake_case naming (Python convention)
- Window identifier: `hook_`, `middle_{N}_`, `closing_`
- Feature name: feature from FeatureTransformation.md (e.g., `scene_count`, `word_count`)
- No prefixes (no `feat_`, no `meta_`)

**Column Ordering**:
1. `video_id` (primary key)
2. Metadata: `duration`, `create_time`, `gender_detection`
3. Hook features: alphabetically within hook
4. Middle features: `middle_1_*`, `middle_2_*`, ... (alphabetically within each middle)
5. Closing features: alphabetically within closing

**For HLD Sections**: 5.2 (Output Schema - exact column names and ordering)

---

#### Q10 [LOW]: What logging strategy should Stage 3 use?

**Context**: Stage 3 needs to log progress, errors, and completion status.

**Logging Requirements**:

**Log Levels Needed**:
- **INFO**: Normal progress (e.g., "Processing bucket_18-33s: 45 videos found")
- **WARNING**: Skipped videos (e.g., "Video 7428596: null middle_segments. Skipping.")
- **ERROR**: Critical failures (e.g., "No valid videos processed in bucket")
- **DEBUG**: Detailed extraction info (optional, for debugging)

**What to Log**:

**Startup**:
```
INFO: Stage 3 starting - bucket_path: data/clients/.../bucket_18-33s
INFO: Found 45 temporal_windows_updated.json files
INFO: Validation complete - ml_analysis/ directory writable
```

**Processing**:
```
INFO: Processing video 1/45 (video_id: 7428596)
WARNING: Video 238506: Malformed JSON. Skipping.
INFO: Processing video 2/45 (video_id: 238506412723073)
```

**Completion**:
```
INFO: Successfully processed 43/45 videos (2 skipped)
INFO: Created aggregated_features.csv - 43 rows × 108 columns
INFO: Stage 3 complete - Duration: 12.3s
```

**Options**:

**Option A: Progress on Every Video**
- Log "Processing video N/M" for each video
- Verbose but helpful for debugging stuck processes

**Option B: Progress Every 10 Videos**
- Log "Processed 10/45 videos", "Processed 20/45 videos", etc.
- Less verbose, still shows progress

**Option C: Silent Processing** (only log errors and summary)
- No per-video logs
- Only log skipped videos (warnings) and final summary
- Cleanest logs

**Recommendation**: **Option B (Progress Every 10 Videos)**

**Rationale**:
- Balances observability with log verbosity
- For typical buckets (30-100 videos), you'll see 3-10 progress messages
- Still shows the process is alive (not stuck)
- Errors/warnings always logged immediately

**Decision**: **Option B (Progress Every 10 Videos)** - Agreed

**Logging Strategy Confirmed**:

**Startup Logs** (INFO):
```
INFO: Stage 3 Feature Aggregation starting
INFO: Bucket path: data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s
INFO: Found 45 temporal_windows_updated.json files
INFO: Validation complete - ml_analysis/ directory writable
```

**Processing Logs** (INFO every 10 videos):
```
INFO: Processed 10/45 videos
INFO: Processed 20/45 videos
INFO: Processed 30/45 videos
INFO: Processed 40/45 videos
```

**Error/Warning Logs** (immediate, always logged):
```
WARNING: Video 238506: Malformed JSON - JSONDecodeError at line 45. Skipping.
WARNING: Video 7428596: null middle_segments. Skipping.
ERROR: Video 9876543: Missing 'metadata' field. Skipping.
```

**Completion Logs** (INFO):
```
INFO: Successfully processed 43/45 videos (2 skipped)
INFO: Skipped reasons: malformed_json=1, null_middle_segments=1
INFO: Created aggregated_features.csv - 43 rows × 108 columns
INFO: Stage 3 complete - Duration: 12.3s
```

**Implementation**:
```python
for i, video_file in enumerate(json_files, start=1):
    # Process video...

    # Log progress every 10 videos
    if i % 10 == 0:
        logger.info(f"Processed {i}/{total_videos} videos")
```

**For HLD Sections**: 8.4 (Logging Strategy - levels, frequency, content, examples)

---

#### Q11 [LOW]: How should Stage 3 handle duplicate video_ids?

**Context**: What if two files in the insights directory have the same video_id?

**Scenario**:
```
bucket_18-33s/analysis/insights/
├── 7428596_temporal_windows_updated.json
├── 7428596_temporal_windows_updated (1).json  ← Duplicate video_id
```

**This could happen if**:
- Stage 2 processed the same video twice (bug)
- Manual file copying error
- File system issues

**Options**:

**Option A: Process Both (Allow Duplicates)**
```python
# aggregated_features.csv will have:
# Row 1: video_id=7428596, features...
# Row 2: video_id=7428596, features... (duplicate)
```
- Pro: Simple, no validation needed
- Con: Duplicate rows in ML training data (could skew model)

**Option B: Keep First, Skip Duplicates**
```python
seen_video_ids = set()
for video_file in json_files:
    video_id = extract_video_id(video_file)
    if video_id in seen_video_ids:
        logger.warning(f"Duplicate video_id {video_id} found in {video_file.name}. Skipping.")
        continue
    seen_video_ids.add(video_id)
    # Process video
```
- Pro: No duplicate rows in output
- Con: Which file is "correct"? First one might not be.

**Option C: Fail on Duplicate**
```python
seen_video_ids = set()
for video_file in json_files:
    video_id = extract_video_id(video_file)
    if video_id in seen_video_ids:
        raise ValueError(f"Duplicate video_id {video_id} detected. Fix Stage 2 or insights directory.")
    seen_video_ids.add(video_id)
```
- Pro: Forces fix upstream (Stage 2 bug or manual error)
- Con: Fails entire bucket on duplicate

**Recommendation**: **Option B (Keep First, Skip Duplicates)**

**Rationale**:
- Graceful handling (doesn't fail entire bucket)
- Prevents duplicate rows in ML training
- Logs warning so user knows there's an issue
- Consistent with "skip bad videos" philosophy (Q5)

**Decision**: **Option B (Keep First, Skip Duplicates)** - Agreed

**Duplicate Handling Strategy Confirmed**:

**Implementation**:
```python
seen_video_ids = set()
skipped_reasons = defaultdict(int)

for video_file in json_files:
    # Extract video_id from filename
    video_id = video_file.stem.replace('_temporal_windows_updated', '')

    # Check for duplicate
    if video_id in seen_video_ids:
        logger.warning(f"Duplicate video_id {video_id} found in {video_file.name}. Skipping.")
        skipped_reasons['duplicate_video_id'] += 1
        continue

    seen_video_ids.add(video_id)

    # Process video...
```

**Behavior**:
- First occurrence of video_id: Processed normally
- Subsequent occurrences: Skipped with WARNING log
- Result: No duplicate rows in aggregated_features.csv

**Example Log Output**:
```
WARNING: Duplicate video_id 7428596 found in 7428596_temporal_windows_updated (1).json. Skipping.
INFO: Successfully processed 44/45 videos (1 skipped)
INFO: Skipped reasons: duplicate_video_id=1
```

**Benefits**:
- Prevents duplicate rows in ML training data
- Graceful handling (doesn't fail entire bucket)
- Logs warning so user can investigate upstream issue
- Consistent with Q5 decision (skip bad videos, continue processing)

**For HLD Sections**: 8.2 (Error Handling - duplicate detection and logging)

---

#### Q12 [LOW]: Should Stage 3 output a summary/stats file alongside the CSV?

**Context**: Would it be useful to have a metadata file documenting what Stage 3 produced?

**Proposed Summary File**: `{bucket-path}/ml_analysis/aggregation_summary.json`

**Example Content**:
```json
{
  "bucket_path": "data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s",
  "timestamp": "2025-01-09T14:32:15Z",
  "duration_seconds": 12.3,
  "input_files_found": 45,
  "videos_processed": 43,
  "videos_skipped": 2,
  "skipped_reasons": {
    "malformed_json": 1,
    "null_middle_segments": 1
  },
  "output_csv": {
    "path": "ml_analysis/aggregated_features.csv",
    "rows": 43,
    "columns": 108,
    "column_names": ["video_id", "duration", "create_time", ...]
  },
  "stage_version": "3.0.0"
}
```

**Use Cases**:
- Debugging: Quickly see why some videos were skipped
- Auditing: Track when bucket was last processed
- Validation: Verify column count matches expectation (108 for this bucket)
- Monitoring: Detect buckets with high skip rates

**Options**:

**Option A: Generate Summary File**
- Write `aggregation_summary.json` alongside CSV
- Minimal code overhead (~20 lines)
- Helpful for debugging/monitoring

**Option B: No Summary File**
- Just log to console/file
- Simpler, fewer files to manage

**Recommendation**: **Option A (Generate Summary File)**

**Rationale**:
- Very low implementation cost
- Helpful for debugging (especially skip reasons)
- Enables future monitoring/dashboards
- Self-documenting output (know when CSV was generated)

**Decision**: **Option A (Generate Summary File)** - Agreed

**Summary File Strategy Confirmed**:

**Output Files** (Stage 3 produces TWO separate files):
```
bucket_18-33s/ml_analysis/
├── aggregated_features.csv          ← CLEAN ML training data (consumed by Stage 4)
└── aggregation_summary.json         ← Metadata (for debugging, NOT consumed by Stage 4)
```

**aggregation_summary.json Schema**:
```json
{
  "bucket_path": "data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s",
  "timestamp": "2025-01-09T14:32:15Z",
  "duration_seconds": 12.3,
  "input_files_found": 45,
  "videos_processed": 43,
  "videos_skipped": 2,
  "skipped_reasons": {
    "malformed_json": 1,
    "null_middle_segments": 1,
    "duplicate_video_id": 0
  },
  "output_csv": {
    "path": "ml_analysis/aggregated_features.csv",
    "rows": 43,
    "columns": 108,
    "column_names": ["video_id", "duration", "create_time", "gender_detection", "hook_scene_count", ...]
  },
  "stage_version": "3.0.0"
}
```

**Key Separation**:
- **aggregated_features.csv**: ONLY features, NO metadata, consumed by Stage 4 ML training
- **aggregation_summary.json**: Metadata about aggregation process, NOT consumed by downstream stages

**Use Cases**:
- Debugging: "Why were 2 videos skipped in bucket_18-33s?"
- Auditing: "When was this bucket last processed?"
- Validation: "Does column count (108) match expected for 3-window bucket?"
- Monitoring: "Which buckets have high skip rates?"

**Implementation**:
```python
summary = {
    "bucket_path": str(bucket_path),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "duration_seconds": round(end_time - start_time, 2),
    "input_files_found": total_files,
    "videos_processed": len(aggregated_data),
    "videos_skipped": total_files - len(aggregated_data),
    "skipped_reasons": dict(skipped_reasons),
    "output_csv": {
        "path": "ml_analysis/aggregated_features.csv",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns)
    },
    "stage_version": "3.0.0"
}

summary_path = bucket_path / "ml_analysis" / "aggregation_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
```

**For HLD Sections**:
- 3.2 (Output Contracts - two separate files)
- 5.3 (Summary Schema - JSON structure)
- 2.3 (Detailed Process - summary generation)

---

### Feature Engineering

## Proceed to Phase 3

### Phase 2 Status: COMPLETE ✅

**Total Questions Asked**: 13
**Total Decisions Made**: 13
**Coverage**: All identified knowledge gaps addressed

### Key Decisions Summary

**Architecture**:
- Stage 2.5 required (file organization from flat `/insights/` to bucket directories)
- Stage 3 invocation: One per bucket (enables parallelization)
- Trust upstream stages (no duration validation, no selected_videos.json cross-reference)

**Input/Output**:
- Input: `{bucket-path}/analysis/insights/*_temporal_windows_updated.json`
- Output 1: `{bucket-path}/ml_analysis/aggregated_features.csv` (clean ML data)
- Output 2: `{bucket-path}/ml_analysis/aggregation_summary.json` (metadata)
- Column naming: Flat with underscores (`hook_scene_count`, `middle_1_word_count`)

**Error Handling**:
- Graceful: Skip bad videos, log warnings, continue processing
- Fail only if ALL videos fail (don't create empty CSV)
- Atomic writes (temp file + rename) for output integrity
- Duplicate video_ids: Keep first, skip subsequent with warning

**Performance**:
- Load all videos into memory (10-20 MB per bucket, trivial)
- No streaming/batching needed

**Observability**:
- Log progress every 10 videos
- Generate summary JSON with skip reasons, timestamps, stats
- Always log errors/warnings immediately

**Testing**:
- 5 test categories defined (happy path, error handling, edge cases, integration, output integrity)

**Feature Engineering** (NEW):
- **Q13 [CRITICAL]**: Stage 4 must derive cross-window features for video-level RF
- Cross-window features include: energy deltas, contrast gaps, consistency (std dev), ratios
- Stage 4 produces THREE outputs: `rf_video_transformed.csv` (190+ features), `{window}_rf_transformed.csv`, `{window}_km_transformed.csv`
- Video-level RF trains on ~190 features (126 raw + ~15 cross-window + 4 temporal + 2 gender + 1 target)
- Critical for LLM analysis: Explicit features like `hook_to_middle_energy_delta: 0.12` are interpretable

### Ready for Phase 3: Child HLD Generation ✅

**Next Steps**:
1. Generate `FeatureAggregationCHILD.md` following Phase3_ChildHLD.md template
2. Use decisions from this QA document to populate HLD sections
3. Include Stage 2.5 dependency in pipeline diagram

**Estimated Child HLD Sections**:
- Section 1.2: Pipeline diagram showing Stage 2.5 → Stage 3 → Stage 4
- Section 2.3: Detailed process with error handling pseudocode
- Section 3.1: Input dependencies (Stage 2.5 completion)
- Section 3.2: Output contracts (dual CSV + JSON)
- Section 5.1: Input schema (temporal_windows_updated.json)
- Section 5.2: Output schema (aggregated_features.csv columns)
- Section 5.3: Summary schema (aggregation_summary.json)
- Section 6.2: Performance considerations (memory usage)
- Section 7.2: Validation strategy (defensive startup checks)
- Section 8.2: Error handling (skip bad videos, duplicates, atomic writes)
- Section 8.4: Logging strategy (every 10 videos)
- Section 9.1-9.2: Testing strategy and test cases

---

**QA Document Status**: FINALIZED
**Date**: 2025-01-09
**Ready to Proceed**: YES
