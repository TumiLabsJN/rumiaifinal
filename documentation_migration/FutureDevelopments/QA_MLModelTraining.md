# Clarification Q&A: ML Model Training (Stage 5)

> **Mother Doc**: MLPlanningv2.md - Stage 5: ML Model Training (Lines 1624-1992)
> **Phase 1**: Critique_MLModelTraining.md
> **Date**: 2025-10-14
> **Status**: COMPLETE

## Questions by Category

### Input/Output Contracts

#### Q1: [CRITICAL] What happens if Stage 4 output files are missing or incomplete?

**Context**: Section 5 (lines 1644-1647) states Stage 5 expects files from Stage 4:
- `ml_analysis/rf_transformed.csv` (video-level RF, ~190 features)
- `ml_analysis/{window}_rf_transformed.csv` (window-level RF, 22 features × 6 windows)
- `ml_analysis/{window}_km_transformed.csv` (window-level K-Means, ~30 features × 6 windows)

**Answer**: **Alternative A - Strict Fail-Fast** (Fail on ANY missing file)

**Behavior**:
- If ANY file is missing → Immediate failure before training ANY models
- Error: "Stage 4 incomplete: Missing {file_path}. Run Stage 4 first."
- Empty CSV (0 rows) treated same as missing file
- Zero tolerance for partial data

**Validation Logic**:
```python
# Validate ALL required files exist before training ANY models
required_files = [
    'ml_analysis/rf_transformed.csv',
    'ml_analysis/hook_rf_transformed.csv',
    'ml_analysis/middle_1_rf_transformed.csv',
    # ... all window RF files
    'ml_analysis/hook_km_transformed.csv',
    # ... all window K-Means files
]

for file_path in required_files:
    if not exists(file_path):
        raise StageInputError(f"Stage 4 incomplete: Missing {file_path}. Run Stage 4 first.")
    if row_count(file_path) == 0:
        raise StageInputError(f"Stage 4 output empty: {file_path} has 0 rows.")
```

**Rationale**:
- Aligns with checkpoint/resume architecture (Foundation line 63)
- Prevents incomplete model architectures (90-model design from Phase 1 Q2)
- Fail-fast is established pattern (consistent with Section 2.4 validation)
- Avoids downstream chaos (Stage 6/7 expect complete model sets)

**For HLD Section**: 6.1 (Input Validation), 6.2 (Error Cases)

#### Q2: [CRITICAL] Where exactly should trained models be saved?

**Context**: Section 5.5 (lines 1877-1902) shows detailed file paths, but Part 1 (lines 171-175) shows simplified generic names. Which is correct?

**Answer**: **Alternative A - Use Section 5.5 Detailed Naming** (follows Part 1's established pattern)

**File Naming Convention**:
```
models/
├── rf_video_{bucket}.pkl          # Video-level RF: rf_video_18-33s.pkl
├── rf_{window}_{bucket}.pkl       # Window-level RF: rf_hook_18-33s.pkl, rf_middle_1_18-33s.pkl
├── {window}_kmeans_{bucket}.pkl   # K-Means: hook_kmeans_18-33s.pkl, middle_1_kmeans_18-33s.pkl
├── {window}_scalers_{bucket}.pkl  # Scalers: hook_scalers_18-33s.pkl, middle_1_scalers_18-33s.pkl
└── model_metrics.json             # Metrics (all models combined)
```

**Example for bucket 18-33s** (20 files total):
```
models/
├── rf_video_18-33s.pkl
├── rf_hook_18-33s.pkl
├── rf_middle_1_18-33s.pkl
├── rf_middle_2_18-33s.pkl
├── rf_middle_3_18-33s.pkl
├── rf_middle_4_18-33s.pkl
├── rf_closing_18-33s.pkl
├── hook_kmeans_18-33s.pkl
├── middle_1_kmeans_18-33s.pkl
├── middle_2_kmeans_18-33s.pkl
├── middle_3_kmeans_18-33s.pkl
├── middle_4_kmeans_18-33s.pkl
├── closing_kmeans_18-33s.pkl
├── hook_scalers_18-33s.pkl
├── middle_1_scalers_18-33s.pkl
├── middle_2_scalers_18-33s.pkl
├── middle_3_scalers_18-33s.pkl
├── middle_4_scalers_18-33s.pkl
├── closing_scalers_18-33s.pkl
└── model_metrics.json
```

**Rationale**:
- **Follows Part 1's established pattern** (lines 157-168): `hook_rf_transformed.csv`, `middle_1_km_transformed.csv`
- Part 1 uses **flat directories with descriptive file names**, NOT nested subdirectories
- Pattern consistency: Stage 4 `hook_rf_transformed.csv` → Stage 5 `rf_hook_18-33s.pkl`
- Part 1 lines 171-175 are simplified placeholders (cannot represent 90-model architecture)
- Self-documenting filenames aid debugging and Stage 6 loading
- Flat structure simpler than nested subdirectories

**Action Item**: Update Part 1 (lines 171-175) to show detailed file naming convention from Section 5.5

**For HLD Section**: 3.2 (Output Contracts), 5.2 (Output Schema), 8 (File Structure)

#### Q3: [CRITICAL] Are model hyperparameters hardcoded or configurable?

**Context**: Section 5.1 (lines 1668-1672) and 5.3 (lines 1789-1793) show specific hyperparameters:
- RandomForest: `n_estimators=100, max_depth=10, random_state=42`
- K-Means: `n_clusters=3, random_state=42, n_init=10`

**Answer**: **Alternative B - Configurable via Configuration File**

**Configuration File Location**: `config/model_hyperparameters.json`

**Configuration Structure**:
```json
{
  "random_forest": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "kmeans": {
    "n_clusters": 3,
    "random_state": 42,
    "n_init": 10
  }
}
```

**Implementation Logic**:
```python
def load_model_config():
    """Load hyperparameters from config with fallback to hardcoded defaults."""
    try:
        with open('config/model_hyperparameters.json') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to hardcoded defaults
        return {
            "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "kmeans": {"n_clusters": 3, "random_state": 42, "n_init": 10}
        }

# Usage
config = load_model_config()
rf_video = RandomForestClassifier(**config["random_forest"])
kmeans = KMeans(**config["kmeans"])
```

**Rationale**:
- Balances simplicity (Alternative A) and flexibility (Alternative C)
- Aligns with Part 1 configuration patterns (line 134: `config.json` per mode/strategy)
- Enables A/B testing without code deployment
- Per-bucket tuning (Alternative C) is premature - can add later if data shows need
- Fail-safe: Missing config file → graceful fallback to hardcoded defaults

**Behavior**:
- Config file exists → Use hyperparameters from file
- Config file missing → Use hardcoded defaults (log warning)
- Config file malformed → Fail with error "Invalid model_hyperparameters.json: {error}"

**For HLD Section**: 2.3 (Detailed Process), 4 (Configuration), 6.1 (Input Validation)

#### Q4: [CRITICAL] What exact metrics should be saved to `model_metrics.json`?

**Context**: Section 5.6 (lines 1910-1960) shows comprehensive metrics structure. Is this the complete and final schema?

**Answer**: **Alternative A - Use Exactly Section 5.6 Schema** (As-Is)

**Complete Schema** (from Section 5.6):
```json
{
  "bucket": "18-33s",
  "total_videos": 100,
  "video_level_rf": {
    "model_type": "random_forest",
    "input_features": 190,
    "accuracy": 0.87,
    "precision": 0.89,
    "recall": 0.84,
    "f1_score": 0.86,
    "top_feature": "hook_eye_contact_rate",
    "top_feature_importance": 0.22,
    "purpose": "Cross-window pattern detection"
  },
  "window_level_rf": {
    "hook": {
      "model_type": "random_forest",
      "input_features": 21,
      "accuracy": 0.82,
      "precision": 0.85,
      "recall": 0.78,
      "top_feature": "eye_contact_rate",
      "top_feature_importance": 0.35
    },
    "middle_1": {...},
    "middle_2": {...},
    "middle_3": {...},
    "middle_4": {...},
    "closing": {...}
  },
  "window_level_kmeans": {
    "hook": {
      "model_type": "kmeans",
      "input_features": 30,
      "n_clusters": 3,
      "inertia": 12.5,
      "silhouette_score": 0.68,
      "cluster_sizes": [35, 42, 23]
    },
    "middle_1": {...},
    "middle_2": {...},
    "middle_3": {...},
    "middle_4": {...},
    "closing": {...}
  }
}
```

**Rationale**:
- **Stage 6 extracts detailed feature importance from .pkl models** (Section 6.1, line 2011)
- model_metrics.json is for quick validation ("Did training succeed?"), not comprehensive analysis
- Avoids redundancy: Feature importance stored in .pkl, extracted by Stage 6, not duplicated here
- Aligns with Part 1 (line 175) showing single model_metrics.json file
- Standard ML convention: model_metrics = performance summary (not complete analysis dump)

**Purpose of this file**:
- Quick sanity check after training completes
- Validate model performance is reasonable (accuracy >0.80)
- Verify top feature makes intuitive sense
- Confirm cluster sizes are balanced (~33 videos each)

**What NOT included** (intentionally):
- Full feature importance rankings (extracted by Stage 6 from .pkl files)
- Confusion matrices (not needed for unsupervised K-Means)
- Cluster centroids (stored in .pkl files, analyzed by Stage 6)

**For HLD Section**: 5.2 (Output Schema), 3.2 (Output Contracts)

### Dependencies & Integration

[Questions will be filled iteratively]

#### Q5: [HIGH] Should models be trained sequentially or in parallel?

**Context**: Section 5 shows sequential training code (lines 1716-1738). For bucket 18-33s with 13 models: sequential = ~26s, parallel = ~2-4s.

**Answer**: **Alternative A - Sequential Training**

**Rationale**:
- 78 seconds for 3 buckets is only **0.36% of total pipeline time** (3.6-4.8 hours from Phase 1 Q4)
- Aligns with Foundation's sequential processing philosophy (line 111)
- Easier debugging during MVP (clear error: "rf_hook_18-33s.pkl failed")
- No resource contention (sklearn RF already uses multi-threading internally)
- Checkpoint/resume easier with sequential (know exactly which model failed)

**Implementation**: Train one model at a time with clear progress logging

**For HLD Section**: 2.3 (Detailed Process), 7.1 (Performance Targets)

---

#### Q6: [CRITICAL] What exact data is used to train the models?

**Context**: Need to understand complete data flow from Stage 1 through Stage 5.

**Answer**: **Alternative A - Use Stage 4 Transformed Data Directly**

**Complete Pipeline Data Flow**:

```
Stage 1: Video Discovery & Selection
  ↓ Output: Selected video URLs/IDs (~300 videos, 3 buckets)

Stage 2: Video Processing (RumiAI)
  ↓ Output: temporal_windows_updated.json (N files, 1 per video)
  ↓ Contains: Per-window features (hook, middle segments, closing)

Stage 2.5: File Organization
  ↓ Output: temporal_windows_updated.json organized into bucket directories

Stage 3: Feature Aggregation
  ↓ Output: aggregated_features.csv (N videos, ~65-215 features per bucket)
  ↓ Format: Flattened temporal windows into single row per video
  ↓ Example columns: hook_scene_count, middle_1_scene_count, ..., closing_scene_count

Stage 4: Feature Transformation
  ↓ Output (3 pipelines):
  ↓ 1. rf_transformed.csv (~190 features) - Video-level RF input
  ↓ 2. {window}_rf_transformed.csv (22 features) - Window-level RF input
  ↓ 3. {window}_km_transformed.csv (~39 features) - K-Means input

Stage 5: ML Model Training ← WE ARE HERE
  ↓ Input: Stage 4 transformed CSVs
  ↓ Process: Load CSVs → Train models → Save .pkl files
```

**Training Data Sources**:

1. **Video-Level RF** uses `ml_analysis/rf_transformed.csv`:
   - Shape: (100 videos, ~190 features)
   - Features: All temporal windows + derived features (hour, day_of_week, gender_encoded)
   - Labels: `is_top_performer` column (1 for top 80%, 0 for bottom 20%)

2. **Window-Level RF** uses `ml_analysis/{window}_rf_transformed.csv`:
   - Shape: (100 videos, 22 features) per window
   - Features: 21 base features per window + `is_top_performer` label
   - Example: `hook_rf_transformed.csv` has hook-specific features only

3. **K-Means** uses `ml_analysis/{window}_km_transformed.csv`:
   - Shape: (100 videos, ~39 features) per window
   - Features: 21 base features + derived features (log transforms, scaled)
   - NO labels: K-Means is unsupervised

**Why Stage 4 outputs are used directly**:
- Stage 4 exists specifically to prepare data for Stage 5
- All transformations (log, scale, one-hot encoding) already applied
- No need to re-transform or re-fetch
- Aligns with checkpoint/resume architecture (Stage 4 complete → Stage 5 can run)

**Training Code Example**:
```python
# For Random Forest (Video-Level)
X = pd.read_csv('ml_analysis/rf_transformed.csv')
y = X['is_top_performer']
X = X.drop(['is_top_performer', 'video_id'], axis=1)
rf_video.fit(X, y)

# For Random Forest (Window-Level)
X = pd.read_csv('ml_analysis/hook_rf_transformed.csv')
y = X['is_top_performer']
X = X.drop(['is_top_performer'], axis=1)
rf_hook.fit(X, y)

# For K-Means (Window-Level)
X = pd.read_csv('ml_analysis/hook_km_transformed.csv')
X = X.drop(['video_id'], axis=1)  # No labels
kmeans_hook.fit(X)
```

**For HLD Section**: 3.1 (Input Dependencies), 5.1 (Input Schema), 2.2 (Data Flow)

### Edge Cases & Validation

#### Q7: [CRITICAL] What happens if a bucket has insufficient videos for training?

**Context**: Section 5 trains models per bucket. Contrastive expects N=100 (80 top + 20 bottom), but Stage 1.6 (lines 897-915) shows buckets may have fewer videos.

**Scenario**: Bucket 18-33s has only 45 videos (not 100). Should Stage 5 train anyway, fail, or skip?

**Answer**: **Alternative A - Fail-Fast on Insufficient Data**

**Minimum Thresholds**:
- **Contrastive mode**: min 50 videos (40 top + 10 bottom, bare minimum for 80/20 split)
- **Top mode**: min 30 videos (descriptive analysis only)

**Behavior**:
- If bucket has < minimum → Fail immediately with clear error
- Error message: "Bucket 18-33s has 45 videos (min 50 required for contrastive mode). Re-run Stage 1 with lower --video-count or skip this bucket."
- No training occurs for this bucket

**Validation Logic**:
```python
MIN_VIDEOS_CONTRASTIVE = 50
MIN_VIDEOS_TOP = 30

def validate_bucket_data(bucket, videos, mode):
    min_required = MIN_VIDEOS_CONTRASTIVE if mode == "contrastive" else MIN_VIDEOS_TOP

    if len(videos) < min_required:
        raise InsufficientDataError(
            f"Bucket {bucket} has {len(videos)} videos (min {min_required} required for {mode} mode). "
            f"Re-run Stage 1 with lower --video-count or skip this bucket."
        )
```

**Rationale**:
- Stage 1 already has flexible thresholds (Section 1.6, lines 908-911) - if Stage 5 receives insufficient data, it's a Stage 1 failure
- Statistical validity: N=45 with 3 K-Means clusters = 15 per cluster (unstable), 80/20 split = 36/9 (9 minority samples insufficient)
- Prevents garbage models → garbage reports → wasted client money
- Clear error forces user to adjust Stage 1 configuration (correct behavior)
- Alternative "train anyway" creates unreliable downstream analysis (Stage 6/7)

**For HLD Section**: 6.1 (Input Validation), 6.2 (Error Cases), 7.2 (Scale Limitations)

---

#### Q8: [HIGH] What happens if model training fails mid-bucket?

**Context**: Section 5 trains 13 models per bucket sequentially. If training fails partway through (e.g., rf_middle_2_18-33s.pkl fails), what happens to partially trained models?

**Scenario**: Bucket 18-33s has 3 models trained, 1 fails, 9 not started. Should Stage 5 keep partial models, delete all, or retry?

**Answer**: **Alternative C - Fail-Fast with Clean Bucket Directory**

**Behavior**:
- Train models directly to `models/` directory (no temporary directory)
- On failure → Delete ALL models for this bucket immediately
- Fail with clear error message
- No resume support - user must re-run Stage 5 from scratch for this bucket

**Implementation**:
```python
def train_bucket_models(bucket, mode):
    """Train all models for bucket. Atomic: all succeed or all deleted."""
    trained_models = []

    try:
        # Train video-level RF
        rf_video = train_video_rf(bucket)
        joblib.dump(rf_video, f'models/rf_video_{bucket}.pkl')
        trained_models.append(f'models/rf_video_{bucket}.pkl')

        # Train window-level RF
        for window in windows:
            rf_window = train_window_rf(bucket, window)
            joblib.dump(rf_window, f'models/rf_{window}_{bucket}.pkl')
            trained_models.append(f'models/rf_{window}_{bucket}.pkl')

        # Train K-Means
        for window in windows:
            kmeans = train_kmeans(bucket, window)
            joblib.dump(kmeans, f'models/{window}_kmeans_{bucket}.pkl')
            trained_models.append(f'models/{window}_kmeans_{bucket}.pkl')

    except Exception as e:
        # Clean up ALL models for this bucket
        for model_path in trained_models:
            if os.path.exists(model_path):
                os.remove(model_path)
        raise ModelTrainingError(
            f"Bucket {bucket} training failed: {e}. All models deleted. Re-run Stage 5."
        )
```

**Error Message Example**:
```
ERROR: Bucket 18-33s training failed at rf_middle_2_18-33s.pkl: NaN values in feature data.
Action: All 3 partially trained models deleted. Fix data issue and re-run Stage 5.
```

**Rationale**:
- Model training is FAST (~26 seconds per bucket from Phase 1 Q4) - re-training all models costs only ~26 seconds
- Foundation checkpoints at video level (Stage 2, expensive: 60-80s per video), NOT at sub-stage level (cheap operations)
- Clean failure semantics: Partial model sets never exist in `models/` → Either complete set OR empty directory
- Simple downstream validation: "Check if all 13 models exist" (not "check which models are missing")
- Checkpoint-based resume (Alternative B) overhead > training time saved

**Result**: Either bucket has complete model set (13 files) OR no models (0 files). Never partial.

**For HLD Section**: 6.2 (Error Cases), 6.4 (Recovery Procedures)

---

### Performance & Scale

#### Q9: [HIGH] What is the acceptable training time for Stage 5?

**Context**: Phase 1 Q4 determined Stage 5 adds ~3.75 minutes for 90 models. In practice, only 3 active buckets train (~86 seconds total for typical scenario).

**Answer**: **Alternative C - No Hard Target (Best-Effort with Logging)**

**Expected Performance**:
- **Typical 3-bucket scenario**: 30-90 seconds per bucket (depends on model count: 3-15 models)
- **Total for 3 buckets**: 90-270 seconds (~1.5-4.5 minutes)
- **Varies by hardware**: Development machine vs CI/CD vs production server

**Behavior**:
- No enforced timeout
- Log actual training time per bucket
- Log warning if > 5 minutes per bucket (suggests performance issue)
- Never fail due to timeout

**Implementation**:
```python
import time

start_time = time.time()

# Train all models for bucket
train_bucket_models(bucket, mode)

elapsed = time.time() - start_time
print(f"✓ Bucket {bucket} training complete: {elapsed:.1f}s ({len(models)} models)")

# Log warning if suspiciously slow
if elapsed > 300:  # 5 minutes per bucket
    logger.warning(
        f"Bucket {bucket} training took {elapsed:.1f}s (expected <120s). "
        f"Check for performance issues."
    )
```

**Performance Guidelines**:
- **Expected**: 30-90 seconds per bucket
- **Acceptable**: 90-300 seconds per bucket (slower hardware)
- **Warning**: > 5 minutes per bucket (log warning, continue)
- **Likely bug**: > 30 minutes per bucket (suggests infinite loop or hardware failure)

**Rationale**:
- Training time depends on hardware (CPU cores, RAM, disk) - hard timeout breaks on slower hardware
- Stage 5 is NOT user-facing (no interactive latency requirement) - 1 minute vs 5 minutes doesn't impact user experience
- Foundation Success Criteria specifies "< 5 hours for 200 video batch" (total pipeline), no per-stage timeouts
- Stage 5 is 0.5-1% of total pipeline time (1-2 minutes of 3.6-4.8 hours) - not a bottleneck
- If training takes 10+ minutes, it's obvious something is wrong (don't need timeout to detect)

**For HLD Section**: 7.1 (Performance Targets), 7.3 (Bottleneck Analysis)

---

### Error Handling

#### Q10: [CRITICAL] What information should be logged when training fails?

**Context**: Section 5 shows training code but doesn't specify logging requirements. When training fails (e.g., NaN values, memory error), what diagnostic information should be captured?

**Answer**: **Alternative C - Balanced Logging (Error + Context, No Data Dump)**

**Logged Information** (on failure):
```
ERROR: Bucket 18-33s training failed at rf_hook_18-33s.pkl
Exception: ValueError: Input contains NaN
Stack trace: [first 10 lines]
Input file: ml_analysis/hook_rf_transformed.csv
Input shape: (100 videos, 22 features)
NaN count: 3 values in 2 columns
Hyperparameters: {n_estimators: 100, max_depth: 10, random_state: 42}
Completed models before failure: ['rf_video_18-33s.pkl']
Training duration before failure: 1.2s
```

**Implementation**:
```python
import logging
import traceback

logger = logging.getLogger('stage5_training')

try:
    train_and_save_model(model_name, X, y, hyperparameters)
except Exception as e:
    # Log comprehensive error context
    logger.error(f"""
Bucket {bucket} training failed at {model_name}
Exception: {type(e).__name__}: {str(e)}
Stack trace: {traceback.format_exc(limit=10)}
Input file: {input_file_path}
Input shape: {X.shape}
NaN count: {X.isna().sum().sum()} values in {X.isna().any().sum()} columns
Hyperparameters: {hyperparameters}
Completed models before failure: {completed_models}
Training duration before failure: {elapsed:.1f}s
""")
    # Clean up and re-raise
    cleanup_partial_models(bucket, completed_models)
    raise
```

**What is logged**:
- **WHAT failed**: Model name, file path, input shape
- **WHY it failed**: Exception type and message, stack trace (first 10 lines)
- **CONTEXT**: Hyperparameters, completed models, training duration, NaN count
- **NOT logged**: Actual feature values, video IDs (privacy concerns)

**Rationale**:
- Provides actionable debugging information without data dump
- Alternative A too minimal ("Input contains NaN" → which column? how many?)
- Alternative B leaks sensitive data (feature values, PII) and fills disk with large logs
- Aligns with Foundation's logging pattern (log metadata, not data)
- Enables self-service debugging (user can inspect CSV file with provided path)

**For HLD Section**: 6.2 (Error Cases), 9 (Logging Strategy)

---

### Testing

#### Q11: [HIGH] What tests are required before deploying Stage 5?

**Context**: Phase 1 created Stage5Tests.md with comprehensive test suite (6 unit tests + 1 integration test). Are these sufficient for production deployment?

**Answer**: **Alternative A - Use Stage5Tests.md As-Is (Sufficient)**

**Testing Requirements** (from Stage5Tests.md):

**Layer 1: Unit Tests** (6 tests, ~10 seconds total)
- Test #1: Binomial test baseline (validates 80% baseline, not 50%)
- Test #2: Feature name normalization (CRITICAL - prevents feature name mismatch bug)
- Test #3: K-Means feature ranking logic (HIGH - validates variance-based ranking)
- Test #4: Success rate calculation
- Test #5: Silhouette score with correct X matrix
- Test #6: Confidence scoring tiers (GOLD/SILVER/BRONZE/EXPLORATORY)

**Layer 2: Integration Test** (1 test, ~30 seconds)
- Test #7: Stage 4 → Stage 5 integration with REAL Stage 4 output data
- Validates: Feature name overlap ≥ 15/21 features

**Layer 3: Manual Validation** (human review, ~30 minutes on first production run)
- Review top 5 features for each model - confirm intuitive sense
- Validate cluster patterns make sense
- Sign off on validation results

**Total Test Time**: ~40 seconds (automated) + 30 minutes (manual validation, first run only)

**Test Execution**:
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run integration test (requires Stage 4 fixtures)
pytest tests/integration/ -v

# Manual validation
# Open Stage5Tests.md Layer 3 and follow checklist
```

**Rationale**:
- Stage5Tests.md designed during Phase 1 Business Critique specifically to prevent HIGH-RISK bugs
- Covers CRITICAL paths: Feature name mismatch (guaranteed bug), K-Means ranking (complex logic), baseline validation (80% not 50%)
- Integration test uses real Stage 4 data (not mocked)
- E2E pipeline test (Alternative B) belongs in system-level tests, not Stage 5 unit tests (too slow: 10-15 minutes)
- Smoke test (Alternative C) redundant with integration test
- MVP philosophy: Test critical paths, add more tests post-MVP if bugs emerge

**Test Status**: ✅ All tests specified in Stage5Tests.md (complete implementation provided)

**For HLD Section**: 8 (Testing Strategy)

---

## Completeness Check

Can write these HLD sections without TODOs or gaps?

### Section 2 (Architecture & Design)
- **2.1: High-level approach**: ✅ YES - Q6 provides complete pipeline context (Stage 1-5 data flow)
- **2.2: Data flow**: ✅ YES - Q6 documents Stage 4 → Stage 5 flow, Q2 documents output file structure
- **2.3: Detailed process**: ✅ YES - Q5 (sequential training), Q3 (hyperparameters), Section 5 provides training code

### Section 3 (Dependencies & Integration)
- **3.1: Input dependencies**: ✅ YES - Q1 (validation), Q6 (Stage 4 CSVs), Q7 (minimum thresholds)
- **3.2: Output contracts**: ✅ YES - Q2 (file naming), Q4 (model_metrics.json schema)
- **3.3: Cross-stage dependencies**: ✅ YES - Q6 provides full Stage 1-5 context
- **3.4: External dependencies**: ✅ YES - sklearn, joblib (from Section 5 code examples)

### Section 5 (Data Schemas)
- **5.1: Input schema**: ✅ YES - Q6 documents exact CSVs (rf_transformed.csv, hook_rf_transformed.csv, hook_km_transformed.csv)
- **5.2: Output schema**: ✅ YES - Q2 (file naming), Q4 (model_metrics.json complete schema)

### Section 6 (Error Handling)
- **6.1: Input validation**: ✅ YES - Q1 (missing files), Q7 (insufficient videos), Q3 (config validation)
- **6.2: Error cases**: ✅ YES - Q8 (mid-bucket failure), Q10 (error logging), Q1 (missing/empty files)
- **6.3: Output validation**: ✅ YES - Q8 (atomic bucket training ensures complete model sets)

### Section 7 (Performance & Scale)
- **7.1: Performance targets**: ✅ YES - Q9 (no hard timeout, 30-90s per bucket expected)
- **7.2: Scale limitations**: ✅ YES - Q7 (min 50 videos contrastive, min 30 videos top mode)
- **7.3: Bottlenecks**: ✅ YES - Q9 (Stage 5 is 0.5-1% of pipeline, not a bottleneck)

### Section 8 (Testing Strategy)
- **8.1-8.3: Test cases**: ✅ YES - Q11 (Stage5Tests.md provides complete test suite: 6 unit + 1 integration + manual validation)

### Section 4 (Configuration)
- **4.1: Configuration files**: ✅ YES - Q3 (config/model_hyperparameters.json with fallback to hardcoded defaults)

## Proceed to Phase 3

**Ready for HLD Generation**: ✅ YES

**All critical information gathered**:
- ✅ Input/Output contracts fully specified (Q1, Q2, Q4, Q6)
- ✅ Dependencies clear (Q6 pipeline context)
- ✅ Edge cases handled (Q1, Q7, Q8)
- ✅ Performance expectations set (Q9)
- ✅ Error handling specified (Q8, Q10)
- ✅ Testing strategy complete (Q11 references Stage5Tests.md)
- ✅ Configuration approach defined (Q3)
- ✅ Training approach decided (Q5 sequential)

**No missing information**. Ready for Phase 3 (Child HLD Generation).

**Status**: ✅ COMPLETE
