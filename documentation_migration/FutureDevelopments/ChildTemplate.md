# [Feature Name] - High-Level Design

> **Parent**: MLPlanningv2.md - Stage X
> **Version**: 1.0
> **Last Updated**: [YYYY-MM-DD]
> **Status**: [Draft | Review | Approved]

---

## 1. Context & Business Goal

<!-- PURPOSE: Provide business context and justification. TI generator needs to understand WHY this feature exists. -->

### 1.1 What Problem Does This Solve?
<!-- 2-3 sentences explaining the business problem this feature addresses.
     Copy from MLROADMAP.md or MLPlanningv2.md if relevant.
     Example: "Manual video selection is time-consuming and inconsistent. This feature automates
     selection using engagement metrics, ensuring reproducible and data-driven choices." -->

[TODO: Add problem statement]

### 1.2 Where This Fits in Pipeline
<!-- Show the exact stage flow from MLPlanningv2.md. TI needs to understand inputs/outputs
     from adjacent stages to validate integration points. -->

**Foundation Dependencies**: This stage depends on FoundationCHILD.md for:
- Client directory structure (Section 2: Client Architecture & Storage)
- CLI parameter definitions (Section 4: CLI Command Structure)
- Config.json schema (Section 5: Configuration Schemas)

```
Stage 0: Configuration (see FoundationCHILD.md)
   ↓ CLI parameters + Directory structure + Config
Stage W: [Previous Stage Name]
   ↓ Output: [exact format, e.g., "temporal_windows_updated.json (N files)"]
Stage X: [THIS FEATURE]
   ↓ Output: [exact format, e.g., "aggregated_features.csv (N rows, M cols)"]
Stage Y: [Next Stage Name]
```

### 1.3 Success Criteria
<!-- Measurable outcomes that define "done". TI generator will use these for test assertions. -->

- [ ] [Criterion 1 - e.g., "Process 300 videos in < 5 minutes"]
- [ ] [Criterion 2 - e.g., "No data loss on checkpoint resume"]
- [ ] [Criterion 3 - e.g., "Output schema matches Stage Y requirements exactly"]

---

## 2. Architecture & Design

<!-- PURPOSE: Core technical design. This is the PRIMARY section TI generator reads. -->

### 2.1 High-Level Approach
<!-- 3-5 sentences explaining the technical strategy.
     Example: "We use pandas for CSV I/O and scikit-learn for feature scaling. Log transformation
     handles right-skewed distributions. MinMaxScaler ensures all features are [0,1] for K-Means." -->

[TODO: Add approach description]

### 2.2 Data Flow
<!-- Visual representation of data transformation. TI generator uses this for function structure. -->

```
Input: [exact format, location, example: "ml_analysis/aggregated_features.csv"]
       Schema: (N videos, ~185 features)
   ↓
Process Step 1: [e.g., "Load CSV and validate schema"]
   ↓
Process Step 2: [e.g., "Apply log transformation to count features"]
   ↓
Process Step 3: [e.g., "MinMax scale all features to [0,1]"]
   ↓
Output: [exact format, location, example: "ml_analysis/rf_transformed.csv"]
        Schema: (N videos, ~190 features)
```

### 2.3 Detailed Process

<!-- PURPOSE: Step-by-step implementation logic. TI generator converts this to actual code. -->

#### Step 2.3.1: [Sub-process Name - e.g., "Input Validation"]

**Purpose**: [One line - e.g., "Ensure input CSV has required columns and valid data ranges"]

**Logic**:
```python
# 10-20 lines pseudocode with comments
# Use realistic variable names that TI can adopt
# Show exact structure, edge cases, error handling

# Example:
def validate_input(df, required_cols):
    """
    Validate aggregated features CSV has required schema.

    Args:
        df: pandas DataFrame from aggregated_features.csv
        required_cols: list of column names (from Section 5.1)

    Raises:
        ValueError: if schema invalid or data out of range
    """
    # Check required columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Validate duration range (3-120 seconds per bucket definitions)
    if not df['duration'].between(3, 120).all():
        invalid = df[~df['duration'].between(3, 120)]
        raise ValueError(f"Invalid duration in {len(invalid)} rows")

    # Check for nulls
    if df.isnull().any().any():
        raise ValueError("Found null values in input")
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Empty input CSV | Raise ValueError | No videos to process - fail fast |
| Missing optional column (e.g., gender) | Use default value (null) | Gender is optional metadata |
| Duration out of range | Raise ValueError | Invalid data from Stage 3 - fail fast |

#### Step 2.3.2: [Sub-process Name]

**Purpose**: [One line]

**Logic**:
```python
# Repeat structure for each major sub-process
# Show complete logic, not just stubs
```

**Edge Cases**:
| Scenario | Handling | Rationale |
|----------|----------|-----------|

#### Step 2.3.3: [Additional sub-processes as needed]

---

## 3. Dependencies & Integration

<!-- PURPOSE: Explicit contracts with other stages. TI generator uses this for imports and validation. -->

### 3.1 Input Dependencies

<!-- List ALL inputs this feature requires. TI generator validates these exist before running. -->

| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| **Foundation setup** | FoundationCHILD.md (FoundationTI.md implementation) | Directory structure + config.json | client_id, bucket, analysis_mode, selection_strategy, video_count, base_paths | Fail-fast if directories don't exist or config.json missing |
| Aggregated features CSV | Stage 3 output | CSV (N rows, ~185 cols) | `hook_scene_count`, `duration`, `create_time` | Fail-fast with error message |
| Config parameters | FoundationCHILD.md Section 4 (CLI flags) | config.json | `video_count`, `strategy` | Read from config.json created by Foundation |
| Bucket directory | FoundationCHILD.md Section 2 (Client Architecture) | Directory path | Must exist with write access | Fail-fast if Foundation didn't create directories |

### 3.2 Output Contracts

<!-- List ALL outputs this feature produces. TI generator ensures these are created. -->

| Output | Format | Schema | Consumers | Validation |
|--------|--------|--------|-----------|------------|
| rf_transformed.csv | CSV | (N rows, ~190 cols) | Stage 5 (ML Training) | Assert column count matches bucket expectations |
| km_transformed.csv | CSV | (N rows, ~375 cols) | Stage 5 (ML Training) | Assert all values in [0,1] range |
| transformation_log.json | JSON | `{applied: [], skipped: []}` | Debugging, auditing | None (optional output) |

### 3.3 Cross-Stage Dependencies

<!-- Explicit stage ordering requirements. TI generator uses this for execution flow validation. -->

**This feature depends on**:
- **Stage 1 (Video Selection)**: Must complete successfully (video list finalized)
- **Stage 2 (Video Processing)**: Must complete successfully (all temporal_windows JSONs exist)
- **Stage 3 (Feature Aggregation)**: Must complete successfully (aggregated_features.csv exists with valid schema)

**This feature is required by**:
- **Stage 5 (ML Training)**: Expects `rf_transformed.csv` and `km_transformed.csv` in exact format
- **Stage 7 (Report Generation)**: Indirectly requires this (via Stage 5 models)

**Failure Impact**:
- If this stage fails: Stage 5 cannot run (no transformed features)
- Checkpoint: Resume from this stage without re-running Stages 1-3

### 3.4 External Dependencies

<!-- Libraries, file system, environment variables. TI generator uses this for imports and setup. -->

**Python Libraries**:
```python
import pandas as pd  # 2.0.0+
import numpy as np  # 1.24.0+
from sklearn.preprocessing import MinMaxScaler  # scikit-learn 1.3.0+
```

**File System**:
- Read access: `/data/clients/{client_id}/buckets/{bucket}/analysis/`
- Write access: `/data/clients/{client_id}/buckets/{bucket}/ml_analysis/`

**Environment Variables**:
- `DATA_ROOT`: Root directory for client data (default: `/data`)
- `LOG_LEVEL`: Logging verbosity (default: `INFO`)

**External Services**: None (pure computational stage)

---

## 4. Configuration & Parameters

<!-- PURPOSE: All tunable values. TI generator uses this for config parsing and defaults. -->

### 4.1 CLI Parameters (if applicable)

<!-- Parameters passed by user. TI generator validates these and applies defaults. -->

| Parameter | Type | Default | Valid Values | Impact | Example |
|-----------|------|---------|--------------|--------|---------|
| `--video-count` | int | 100 | 40-300 | Number of videos to process per bucket | `--video-count 150` |
| `--strategy` | str | `contrastive` | `contrastive`, `top` | Changes RF target variable creation | `--strategy top` |
| `--bucket` | str | Required | `0-3s`, `9-13s`, ... | Which duration bucket to process | `--bucket 18-33s` |

### 4.2 Internal Configuration

<!-- Hardcoded constants. TI generator uses these exact values unless specified otherwise. -->

```python
# Random Forest hyperparameters (used in transformation context)
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42

# K-Means parameters
N_CLUSTERS = 3
KMEANS_RANDOM_STATE = 42

# Feature scaling parameters
LOG_FEATURES = ['scene_count', 'word_count', 'element_count']  # Right-skewed features
RATE_FEATURES = ['eye_contact_rate', 'speech_coverage', 'joy_ratio']  # Already [0,1]

# File paths (relative to bucket directory)
AGGREGATED_CSV = "ml_analysis/aggregated_features.csv"
RF_OUTPUT = "ml_analysis/rf_transformed.csv"
KM_OUTPUT = "ml_analysis/km_transformed.csv"
```

---

## 5. Data Schemas

<!-- PURPOSE: Exact data structures. TI generator uses this for validation and type hints. -->

### 5.1 Input Schema

**File**: `ml_analysis/aggregated_features.csv`

<!-- IMPORTANT: List ALL columns that TI must validate. Include types, ranges, nullable. -->

| Column | Type | Range | Nulls? | Description | Source |
|--------|------|-------|--------|-------------|--------|
| `hook_scene_count` | int | 0-20 | No | Scene cuts in hook window (0-3s) | Stage 3 aggregation |
| `hook_eye_contact_rate` | float | 0.0-1.0 | No | Eye contact proportion in hook | Stage 3 aggregation |
| `hook_word_count` | int | 0-200 | No | Words spoken in hook | Stage 3 aggregation |
| `middle_1_scene_count` | int | 0-20 | No | Scene cuts in middle segment 1 | Stage 3 aggregation |
| `middle_1_word_count` | int | 0-200 | No | Words in middle segment 1 | Stage 3 aggregation |
| `closing_energy_level` | float | 0.0-1.0 | No | Audio energy in closing window | Stage 3 aggregation |
| `duration` | float | 3.0-120.0 | No | Total video duration (seconds) | Metadata |
| `create_time` | datetime | - | No | Video publish timestamp | Metadata |
| `gender` | str | `male`, `female`, `null` | Yes | Detected gender (optional) | Metadata |
| `gender_confidence` | float | 0.0-1.0 | Yes | Gender detection confidence | Metadata |

**Total Columns**: ~185 (varies by bucket - see Section 5.1.1)

#### 5.1.1 Bucket-Specific Column Counts

| Bucket | Total Windows | Base Features per Window | Metadata | Total Columns |
|--------|---------------|--------------------------|----------|---------------|
| 0-3s, 3-9s | 2 (Hook + Closing) | 30 × 2 = 60 | 5 | ~65 |
| 9-13s, 13-18s | 5 (Hook + 3 Middle + Closing) | 30 × 5 = 150 | 5 | ~155 |
| 18-33s | 6 (Hook + 4 Middle + Closing) | 30 × 6 = 180 | 5 | ~185 |
| 33-60s, 60-90s, 90-120s | 7 (Hook + 5 Middle + Closing) | 30 × 7 = 210 | 5 | ~215 |

### 5.2 Output Schema

**File 1**: `ml_analysis/rf_transformed.csv`

<!-- RF-specific transformations. List NEW columns added and any columns removed. -->

| Column | Type | Range | Nulls? | Description | Transformation |
|--------|------|-------|--------|-------------|----------------|
| `hook_scene_count` | int | 0-20 | No | Same as input | Unchanged |
| `hook_eye_contact_rate` | float | 0.0-1.0 | No | Same as input | Unchanged |
| `hour` | int | 0-23 | No | Hour of day from create_time | Extracted |
| `day_of_week` | int | 0-6 | No | Day of week (0=Monday) | Extracted |
| `is_weekend` | int | 0, 1 | No | 1 if Sat/Sun, 0 otherwise | Derived |
| `is_business_hours` | int | 0, 1 | No | 1 if 9am-5pm, 0 otherwise | Derived |
| `gender_male` | int | 0, 1 | No | One-hot encoded gender | One-hot |
| `gender_female` | int | 0, 1 | No | One-hot encoded gender | One-hot |
| `is_top_performer` | int | 0, 1 | No | Target variable (contrastive only) | Computed |

**Removed Columns**: `create_time` (replaced with temporal features)

**Total Columns**: ~190 for bucket 18-33s (185 original + 5 new temporal features, - 1 removed)

**File 2**: `ml_analysis/km_transformed.csv`

<!-- K-Means-specific transformations. Show log/scale transformations. -->

| Column | Type | Range | Nulls? | Description | Transformation |
|--------|------|-------|--------|-------------|----------------|
| `hook_scene_count_log` | float | 0.0-~3.0 | No | Log-transformed scene count | log1p(x) |
| `hook_scene_count_scaled` | float | 0.0-1.0 | No | MinMax scaled log value | MinMaxScaler |
| `hook_eye_contact_rate_scaled` | float | 0.0-1.0 | No | MinMax scaled rate | MinMaxScaler |
| `hour_sin` | float | -1.0-1.0 | No | Cyclical encoding (sin) | sin(2π * hour/24) |
| `hour_cos` | float | -1.0-1.0 | No | Cyclical encoding (cos) | cos(2π * hour/24) |
| `day_sin` | float | -1.0-1.0 | No | Cyclical encoding (sin) | sin(2π * day/7) |
| `day_cos` | float | -1.0-1.0 | No | Cyclical encoding (cos) | cos(2π * day/7) |
| `gender_male` | int | 0, 1 | No | One-hot encoded gender | One-hot |
| `gender_female` | int | 0, 1 | No | One-hot encoded gender | One-hot |

**Removed Columns**: All original raw features (replaced with log/scaled versions), `create_time`

**Total Columns**: ~375 for bucket 18-33s (count features × 2 transformations + rate features × 1 + temporal + one-hot)

---

## 6. Error Handling & Validation

<!-- PURPOSE: All error scenarios. TI generator uses this for try/catch blocks and assertions. -->

### 6.1 Input Validation

<!-- Validation logic that runs BEFORE processing. TI implements this first. -->

```python
def validate_input(df, bucket):
    """
    Validate aggregated features CSV before transformation.

    Args:
        df: pandas DataFrame from aggregated_features.csv
        bucket: str, bucket name (e.g., "18-33s")

    Raises:
        ValueError: if validation fails with specific error message
    """
    # 1. Check required columns exist
    required_cols = get_required_columns(bucket)  # From Section 5.1
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 2. Validate data types
    for col, expected_type in COLUMN_TYPES.items():
        if col in df.columns and not df[col].dtype == expected_type:
            raise ValueError(f"Column {col} has type {df[col].dtype}, expected {expected_type}")

    # 3. Validate value ranges
    if not df['duration'].between(3, 120).all():
        invalid = df[~df['duration'].between(3, 120)]
        raise ValueError(f"Invalid duration in {len(invalid)} rows: {invalid['duration'].tolist()}")

    # 4. Check for nulls in required columns
    required_non_null = [c for c in required_cols if c not in ['gender', 'gender_confidence']]
    if df[required_non_null].isnull().any().any():
        null_cols = df[required_non_null].columns[df[required_non_null].isnull().any()].tolist()
        raise ValueError(f"Found null values in required columns: {null_cols}")

    # 5. Validate row count matches video_count parameter
    # (This is a warning, not an error)
    expected_count = get_video_count_from_config()
    if len(df) != expected_count:
        logger.warning(f"Expected {expected_count} videos, found {len(df)}")
```

### 6.2 Error Cases

<!-- Comprehensive error table. TI generator creates specific error messages for each. -->

| Error | Detection | Handling | User Message | Exit Code |
|-------|-----------|----------|--------------|-----------|
| Missing input file | `os.path.exists(csv_path)` | Fail-fast | `"Aggregated CSV not found at {path}. Did Stage 3 complete successfully?"` | 1 |
| Invalid CSV format | `pd.read_csv()` exception | Fail-fast | `"Failed to parse CSV: {error}. Check file is valid CSV format."` | 2 |
| Missing required column | Column validation | Fail-fast | `"Missing columns: {cols}. Required: {required_cols}"` | 3 |
| Invalid duration range | Range validation | Fail-fast | `"Invalid duration in {count} rows. Expected 3-120s, found {values}"` | 4 |
| Null values in required fields | `.isnull().any()` | Fail-fast | `"Found null values in {cols}. Stage 3 may have failed."` | 5 |
| Insufficient videos | Row count check | Warn + continue | `"Only {count} videos (expected {expected}). Proceeding with available data."` | 0 (warning) |
| Write permission denied | File write exception | Fail-fast | `"Cannot write to {path}. Check permissions."` | 6 |
| Out of memory | MemoryError exception | Fail-fast | `"Insufficient memory. Try processing smaller batches."` | 7 |

### 6.3 Output Validation

<!-- Validation that runs AFTER processing. TI implements this at end of pipeline. -->

```python
def validate_output(df_rf, df_km, bucket):
    """
    Validate transformed CSVs before saving.

    Args:
        df_rf: pandas DataFrame for RF
        df_km: pandas DataFrame for K-Means
        bucket: str, bucket name

    Raises:
        AssertionError: if output schema invalid
    """
    # 1. Check RF output column count
    expected_rf_cols = get_expected_rf_columns(bucket)
    assert len(df_rf.columns) == len(expected_rf_cols), \
        f"RF output has {len(df_rf.columns)} columns, expected {len(expected_rf_cols)}"

    # 2. Check K-Means output all values in [0,1] (after scaling)
    scaled_cols = [c for c in df_km.columns if c.endswith('_scaled')]
    for col in scaled_cols:
        assert df_km[col].between(0, 1).all(), \
            f"Column {col} has values outside [0,1]: {df_km[col].min()}-{df_km[col].max()}"

    # 3. Check no nulls introduced during transformation
    assert not df_rf.isnull().any().any(), "RF output contains null values"
    assert not df_km.isnull().any().any(), "K-Means output contains null values"

    # 4. Check target variable exists (contrastive only)
    if strategy == 'contrastive':
        assert 'is_top_performer' in df_rf.columns, "Missing target variable"
        assert df_rf['is_top_performer'].isin([0, 1]).all(), "Invalid target values"
```

---

## 7. Performance & Scalability

<!-- PURPOSE: Performance targets and bottlenecks. TI generator uses this for optimization. -->

### 7.1 Performance Targets

<!-- Measurable performance goals. TI generator can add instrumentation to verify. -->

- **Throughput**: Process 300 videos (bucket 18-33s) in < 5 minutes
- **Memory**: Peak usage < 2GB for 300 videos
- **Disk I/O**: < 10s for CSV read/write operations
- **CPU**: < 50% average utilization (single-threaded)

### 7.2 Measured Performance

<!-- If known, provide actual measurements. Helps TI generator understand realistic expectations. -->

| Metric | Bucket 18-33s (N=100) | Bucket 18-33s (N=300) | Notes |
|--------|------------------------|------------------------|-------|
| Total time | 1.2 minutes | 3.8 minutes | Linear scaling |
| Memory peak | 750 MB | 2.1 GB | Proportional to N |
| CSV read | 2s | 5s | pandas `read_csv` |
| Transformation | 45s | 2.5 minutes | Bottleneck: MinMaxScaler fit |
| CSV write | 3s | 8s | pandas `to_csv` |

### 7.3 Bottlenecks & Mitigations

<!-- Known performance issues and how to address them. -->

| Bottleneck | Impact | Cause | Mitigation | Priority |
|------------|--------|-------|------------|----------|
| CSV I/O | 30s for N=300 | pandas single-threaded read | Use `pyarrow` engine: `pd.read_csv(engine='pyarrow')` | Medium |
| MinMaxScaler fit | 2.5 min for N=300 | Scikit-learn computes min/max per feature | Cache scalers per bucket, reuse across runs | High |
| Memory growth | 2.1 GB peak | pandas loads full CSV into memory | Use chunked processing if N > 500 | Low |
| One-hot encoding | 10s for N=300 | pandas `get_dummies` | Acceptable performance, no optimization needed | Low |

### 7.4 Scalability Limits

<!-- When does this break? TI generator can add validation checks. -->

- **Max videos per bucket**: 1000 (memory constraint, 7GB peak)
- **Max features**: 500 (K-Means performance degrades with high dimensionality)
- **Min videos per bucket**: 10 (below this, statistics unreliable)

---

## 8. Testing Strategy

<!-- PURPOSE: Test plan. TI generator uses this to create test suite. -->

### 8.1 Unit Tests

<!-- Test individual functions in isolation. TI generator creates pytest fixtures. -->

- [ ] **Test input validation**
  - Empty CSV (raises ValueError)
  - Missing required columns (raises ValueError with column names)
  - Invalid duration values (raises ValueError with offending rows)
  - Null values in required columns (raises ValueError)
  - Valid input (passes without error)

- [ ] **Test feature transformation**
  - Log transformation: `log1p(0) = 0`, `log1p(10) ≈ 2.4`
  - MinMax scaling: min → 0, max → 1, midpoint → 0.5
  - One-hot encoding: gender='male' → [1, 0], gender='female' → [0, 1], gender=null → [0, 0]
  - Cyclical encoding: hour=0 → sin=0, cos=1; hour=12 → sin=0, cos=-1

- [ ] **Test edge cases**
  - Single video (N=1)
  - Large batch (N=1000)
  - Missing optional gender field (use default null)
  - All videos same value (variance=0, handle division by zero)

- [ ] **Test output validation**
  - RF output has correct column count per bucket
  - K-Means scaled columns all in [0,1]
  - No nulls introduced during transformation
  - Target variable binary (0 or 1 only)

### 8.2 Integration Tests

<!-- Test interaction with adjacent stages. TI generator creates end-to-end tests. -->

- [ ] **End-to-end: Stage 3 → Stage 4 → Stage 5**
  - Use real `aggregated_features.csv` from Stage 3 (10 videos, bucket 18-33s)
  - Run transformation
  - Validate `rf_transformed.csv` and `km_transformed.csv` exist
  - Verify Stage 5 can load outputs without error

- [ ] **Checkpoint resume**
  - Not applicable (Stage 4 is fast, no checkpointing needed)

- [ ] **Error propagation**
  - Stage 3 missing output → Stage 4 fails with clear message
  - Stage 4 transformation error → Stage 5 does not run

### 8.3 Test Data

<!-- Sample data for tests. TI generator uses this to create fixtures. -->

**File**: `tests/fixtures/aggregated_features_bucket_18-33s_sample.csv`

```csv
hook_scene_count,hook_eye_contact_rate,middle_1_word_count,middle_2_word_count,middle_3_word_count,middle_4_word_count,closing_energy_level,duration,create_time,gender
3,0.85,55,48,62,51,0.75,18.5,2025-01-15 14:30:00,female
5,0.62,42,38,45,40,0.82,22.1,2025-01-16 09:15:00,male
2,0.91,68,71,65,69,0.68,25.3,2025-01-17 18:45:00,female
```

**Expected RF Output**: `tests/fixtures/rf_transformed_expected.csv`

```csv
hook_scene_count,hook_eye_contact_rate,middle_1_word_count,...,hour,is_weekend,is_top_performer
3,0.85,55,...,14,0,1
5,0.62,42,...,9,0,1
2,0.91,68,...,18,0,1
```

**Expected KM Output**: `tests/fixtures/km_transformed_expected.csv`

```csv
hook_scene_count_log,hook_scene_count_scaled,hook_eye_contact_rate_scaled,...,hour_sin,hour_cos
1.386,0.45,0.78,...,0.5,0.866
1.791,0.85,0.15,...,-0.5,0.866
1.099,0.12,1.0,...,-0.866,-0.5
```

### 8.4 Test Execution

```bash
# Run unit tests
pytest tests/test_feature_transformation.py -v

# Run integration tests
pytest tests/test_stage4_integration.py -v

# Run with coverage
pytest --cov=feature_transformation --cov-report=html
```

---

## 9. Future Enhancements

<!-- PURPOSE: Planned improvements. TI generator ignores this section (not for current implementation). -->

### 9.1 Planned Improvements

<!-- Features to add in future phases. Document now for future reference. -->

- **Phase 2: Auto-detect optimal feature scaling per bucket**
  - Current: Fixed log transformation for counts
  - Future: Analyze distribution skewness, apply BoxCox/Yeo-Johnson automatically
  - Impact: Better K-Means clustering quality

- **Phase 3: Parallel processing across buckets**
  - Current: Process one bucket at a time
  - Future: Run 3 buckets in parallel (independent transformations)
  - Impact: 3x speedup for full hashtag analysis

- **Phase 4: Feature selection based on RF importance**
  - Current: Use all features
  - Future: Drop low-importance features (< 0.01 importance)
  - Impact: Faster training, reduced overfitting

### 9.2 Known Limitations

<!-- Current constraints or technical debt. TI generator acknowledges but doesn't fix. -->

- **Manual feature selection**: No automatic feature engineering (e.g., polynomial features, interactions)
- **Fixed n_clusters=3**: No elbow method or silhouette analysis for optimal K
- **No missing data imputation**: Fails if required fields are null (could use mean/median imputation)
- **Single-threaded**: pandas operations not parallelized (could use Dask for large N)

---

## 10. References & Related Docs

<!-- PURPOSE: Links to other documentation. TI generator uses this for additional context if needed. -->

### 10.1 Parent Document

- **MLPlanningv2.md Section X.Y "[Section Title]"**
  - High-level stage overview
  - Stage position in pipeline
  - Input/output contracts

### 10.2 Mother Document Foundation

- **MLPlanningv2.md Part 1: Foundation** (shared across all stages)
  - Section 1 "System Goals": Success criteria and objectives
  - Section 2 "Client Architecture": Directory paths used in this stage
  - Section 3 "Configuration Dimensions": CLI parameters affecting this stage
  - Section 4 "CLI Command Structure": Complete command syntax
  - Section 5 "Configuration Schemas": config.json, Apify metadata, checkpoints
  - **Appendix A "Glossary"**: System-wide term definitions (FEAT, temporal windows, buckets, etc.)

**Key Sections Referenced in This Stage**:
- Section 2 "Client Architecture": Provides base directory paths for file I/O
- Section 4 "CLI Command Structure": Defines CLI parameters this stage reads
- Section 5 "Configuration Schemas": Defines config.json structure this stage depends on
- **Appendix A "Glossary"**: For all term definitions, see Mother Part 1 Glossary

### 10.3 Related Child Docs

<!-- Upstream and downstream child docs. TI generator may reference for integration. -->

- **FeatureAggregation.md** (Stage 3)
  - Produces `aggregated_features.csv` (input to this stage)
  - Defines exact column names and temporal window structure

- **MLModelTraining.md** (Stage 5)
  - Consumes `rf_transformed.csv` and `km_transformed.csv` (outputs from this stage)
  - Defines expected schema and validation requirements

### 10.3 External References

<!-- Documentation for libraries, APIs, or external systems. -->

- **Scikit-learn MinMaxScaler**: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
- **Pandas `get_dummies`**: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
- **RumiAI temporal_windows schema**: `SystemArchitecturev2.md` (lines 395-460)
- **Feature definitions**: `TotalFeatures.md` (complete feature list with descriptions)

### 10.4 Code References

<!-- If existing code exists, link to it. TI generator can review for patterns. -->

- **Existing transformation code** (if any): `/path/to/existing/code.py`
- **Utility functions**: `/utils/feature_utils.py` (reusable transformation helpers)

---

## Appendix A: Decision Log

<!-- PURPOSE: Document key design decisions with rationale. Helps future developers understand WHY things were designed this way. -->

**Purpose**: Record major design decisions, alternatives considered, and trade-offs accepted.

**Decision 1**: [Design choice made]
- **Context**: [Situation that required this decision]
- **Alternatives Considered**:
  - Option A: [Description] - Rejected because [reason]
  - Option B: [Description] - Rejected because [reason]
- **Rationale**: [Why chosen approach is best]
- **Trade-offs**: [What was sacrificed by choosing this approach]
- **Date**: [Decision date]

**Decision 2**: [Design choice made]
- **Context**: [Situation that required this decision]
- **Alternatives Considered**: [Options evaluated]
- **Rationale**: [Why chosen approach is best]
- **Trade-offs**: [What was sacrificed]
- **Date**: [Decision date]

**Decision 3**: [Design choice made]
- **Context**: [Situation that required this decision]
- **Alternatives Considered**: [Options evaluated]
- **Rationale**: [Why chosen approach is best]
- **Trade-offs**: [What was sacrificed]
- **Date**: [Decision date]

[TODO: Add 3-5 key design decisions during Phase 3 generation from Phase 1 Critique + Phase 2 Q&A]

**Examples of design decisions to document**:
- Sequential vs parallel processing approach
- Skip-on-fail vs fail-fast error policy
- Checkpoint strategy (frequency, granularity)
- State storage method (JSON, SQLite, in-memory)
- Signal handling approach (SIGUSR1, SIGTERM)
- Data structure choices (list vs dict, DataFrame vs arrays)

---

## Appendix B: Example Data

<!-- PURPOSE: Concrete examples. TI generator uses this to understand data format visually. -->

### B.1 Sample Input (3 rows, 10 columns shown)

**File**: `ml_analysis/aggregated_features.csv`

```csv
hook_scene_count,hook_eye_contact_rate,hook_word_count,middle_1_word_count,middle_2_word_count,closing_energy_level,duration,create_time,gender,gender_confidence
3,0.85,15,55,48,0.75,18.5,2025-01-15 14:30:00,female,0.92
5,0.62,8,42,38,0.82,22.1,2025-01-16 09:15:00,male,0.88
2,0.91,22,68,71,0.68,25.3,2025-01-17 18:45:00,female,0.95
```

### B.2 Sample RF Output (3 rows, 12 columns shown)

**File**: `ml_analysis/rf_transformed.csv`

```csv
hook_scene_count,hook_eye_contact_rate,hook_word_count,middle_1_word_count,middle_2_word_count,closing_energy_level,hour,day_of_week,is_weekend,gender_male,gender_female,is_top_performer
3,0.85,15,55,48,0.75,14,2,0,0,1,1
5,0.62,8,42,38,0.82,9,3,0,1,0,1
2,0.91,22,68,71,0.68,18,4,0,0,1,1
```

### B.3 Sample KM Output (3 rows, 10 columns shown)

**File**: `ml_analysis/km_transformed.csv`

```csv
hook_scene_count_log,hook_scene_count_scaled,hook_eye_contact_rate_scaled,hook_word_count_log,hook_word_count_scaled,hour_sin,hour_cos,day_sin,day_cos,gender_female
1.386,0.45,0.78,2.773,0.62,0.5,0.866,0.434,0.901,1
1.791,0.85,0.15,2.197,0.35,-0.5,0.866,0.782,0.623,0
1.099,0.12,1.0,3.135,0.88,-0.866,-0.5,0.975,0.223,1
```

---

## Appendix C: Pseudocode (Complete)

<!-- PURPOSE: Detailed implementation logic. TI generator converts this to production code. -->

### C.1 Full Transformation Pipeline

```python
def transform_features(input_csv_path, output_dir, bucket, strategy, video_count):
    """
    Transform aggregated features for RF and K-Means models.

    Args:
        input_csv_path: str, path to aggregated_features.csv
        output_dir: str, directory to save transformed CSVs
        bucket: str, bucket name (e.g., "18-33s")
        strategy: str, "contrastive" or "top"
        video_count: int, expected number of videos

    Returns:
        tuple: (rf_csv_path, km_csv_path)

    Raises:
        ValueError: if input validation fails
        IOError: if file I/O fails
    """
    # ===== 1. Load Input =====
    logger.info(f"Loading aggregated features from {input_csv_path}")
    df = pd.read_csv(input_csv_path, parse_dates=['create_time'])

    # ===== 2. Validate Input =====
    logger.info("Validating input schema and data quality")
    validate_input(df, bucket)  # See Section 6.1

    # ===== 3. Random Forest Transformation =====
    logger.info("Transforming features for Random Forest")
    df_rf = df.copy()

    # 3.1: Extract temporal features from create_time
    df_rf['hour'] = df_rf['create_time'].dt.hour
    df_rf['day_of_week'] = df_rf['create_time'].dt.dayofweek
    df_rf['is_weekend'] = (df_rf['day_of_week'] >= 5).astype(int)
    df_rf['is_business_hours'] = ((df_rf['hour'] >= 9) & (df_rf['hour'] <= 17)).astype(int)

    # 3.2: One-hot encode categorical features
    if 'gender' in df_rf.columns:
        df_rf = pd.get_dummies(df_rf, columns=['gender'], prefix='gender', dummy_na=False)

    # 3.3: Add target variable (contrastive strategy only)
    if strategy == 'contrastive':
        top_count = int(video_count * 0.8)
        df_rf['is_top_performer'] = (df_rf.index < top_count).astype(int)

    # 3.4: Drop original create_time (replaced with temporal features)
    df_rf.drop(columns=['create_time'], inplace=True)

    # 3.5: Save RF transformed CSV
    rf_output_path = os.path.join(output_dir, "rf_transformed.csv")
    df_rf.to_csv(rf_output_path, index=False)
    logger.info(f"Saved RF transformed features: {rf_output_path} ({len(df_rf)} rows, {len(df_rf.columns)} cols)")

    # ===== 4. K-Means Transformation =====
    logger.info("Transforming features for K-Means")
    df_km = df.copy()

    # 4.1: Log + scale for right-skewed features (counts)
    count_features = [col for col in df_km.columns if any(x in col for x in ['scene_count', 'word_count', 'element_count'])]
    for feature in count_features:
        # Log transformation
        df_km[f'{feature}_log'] = np.log1p(df_km[feature])
        # MinMax scaling
        min_val = df_km[f'{feature}_log'].min()
        max_val = df_km[f'{feature}_log'].max()
        if max_val > min_val:  # Avoid division by zero
            df_km[f'{feature}_scaled'] = (df_km[f'{feature}_log'] - min_val) / (max_val - min_val)
        else:
            df_km[f'{feature}_scaled'] = 0.5  # All same value → midpoint
        # Drop original feature
        df_km.drop(columns=[feature, f'{feature}_log'], inplace=True)

    # 4.2: Scale [0-1] for already-normalized features (rates, ratios)
    rate_features = [col for col in df_km.columns if any(x in col for x in ['_rate', '_ratio', '_coverage', 'energy_level'])]
    for feature in rate_features:
        min_val = df_km[feature].min()
        max_val = df_km[feature].max()
        if max_val > min_val:
            df_km[f'{feature}_scaled'] = (df_km[feature] - min_val) / (max_val - min_val)
        else:
            df_km[f'{feature}_scaled'] = 0.5
        df_km.drop(columns=[feature], inplace=True)

    # 4.3: Cyclical encoding for create_time
    df_km['hour_sin'] = np.sin(2 * np.pi * df_km['create_time'].dt.hour / 24)
    df_km['hour_cos'] = np.cos(2 * np.pi * df_km['create_time'].dt.hour / 24)
    df_km['day_sin'] = np.sin(2 * np.pi * df_km['create_time'].dt.dayofweek / 7)
    df_km['day_cos'] = np.cos(2 * np.pi * df_km['create_time'].dt.dayofweek / 7)
    df_km.drop(columns=['create_time'], inplace=True)

    # 4.4: One-hot encode categorical features
    if 'gender' in df_km.columns:
        df_km = pd.get_dummies(df_km, columns=['gender'], prefix='gender', dummy_na=False)

    # 4.5: Save K-Means transformed CSV
    km_output_path = os.path.join(output_dir, "km_transformed.csv")
    df_km.to_csv(km_output_path, index=False)
    logger.info(f"Saved K-Means transformed features: {km_output_path} ({len(df_km)} rows, {len(df_km.columns)} cols)")

    # ===== 5. Validate Outputs =====
    logger.info("Validating output schemas")
    validate_output(df_rf, df_km, bucket)  # See Section 6.3

    # ===== 6. Return Paths =====
    return rf_output_path, km_output_path


def validate_input(df, bucket):
    """See Section 6.1 for full implementation"""
    pass  # Detailed in Section 6.1


def validate_output(df_rf, df_km, bucket):
    """See Section 6.3 for full implementation"""
    pass  # Detailed in Section 6.3
```

### C.2 Helper Functions

```python
def get_required_columns(bucket):
    """
    Get list of required columns for a given bucket.

    Args:
        bucket: str, bucket name (e.g., "18-33s")

    Returns:
        list: column names required in aggregated_features.csv
    """
    # Base columns (always required)
    base_cols = ['duration', 'create_time']

    # Temporal window columns (bucket-specific)
    base_features = [
        'scene_count', 'eye_contact_rate', 'word_count', 'speech_coverage',
        'energy_level', 'joy_ratio', 'surprise_ratio', 'close_ratio'
        # ... (all ~30 base features)
    ]

    # Hook columns (always 1 window)
    hook_cols = [f'hook_{feat}' for feat in base_features]

    # Middle columns (varies by bucket)
    middle_segment_count = {
        '0-3s': 0, '3-9s': 0,
        '9-13s': 3, '13-18s': 3,
        '18-33s': 4,
        '33-60s': 5, '60-90s': 5, '90-120s': 5
    }

    middle_cols = []
    for i in range(1, middle_segment_count[bucket] + 1):
        middle_cols.extend([f'middle_{i}_{feat}' for feat in base_features])

    # Closing columns (always 1 window)
    closing_cols = [f'closing_{feat}' for feat in base_features]

    return base_cols + hook_cols + middle_cols + closing_cols
```

---

## Document Metadata

**Creation Date**: [YYYY-MM-DD]
**Last Modified**: [YYYY-MM-DD]
**Authors**: [Your Name]
**Reviewers**: [Reviewer Names]
**Approved By**: [Approver Name]
**Next Review Date**: [YYYY-MM-DD]

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-28 | [Name] | Initial draft |
| 1.1 | [Date] | [Name] | [Description] |
