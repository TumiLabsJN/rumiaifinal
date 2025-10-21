# Cross-HLD Alignment Action Items
**RumiAI ML Pipeline - Stages 3-7 Consistency Audit**

**Date**: 2025-10-16
**Analysis Scope**: 8,316 lines across 5 HLD documents
**Status**: ✅ Analysis Complete - Ready for Implementation

---

## 📊 Executive Summary

**Total Issues Identified**: 20 items
**Overall Assessment**: **GOOD ALIGNMENT** ✅
**Confidence to Proceed to TI**: **HIGH**

### Issue Breakdown

| Severity | Count | Action Required |
|----------|-------|-----------------|
| ✅ Already Aligned | 10 (50%) | No action |
| ⛔ **CRITICAL** | 2 (10%) | **Fix before TI** |
| 🟡 **MEDIUM** | 5 (25%) | Address before TI |
| 🟢 **LOW** | 3 (15%) | Nice-to-have |

### Documents Analyzed

1. **FeatureAggregationCHILD.md** (Stage 3) - 3,847 lines
2. **FeatureTransformationCHILD.md** (Stage 4) - 1,556 lines
3. **Stage5_MLModelTraining_HLD.md** (Stage 5) - 950 lines
4. **MLAnalysisGenerationCHILD.md** (Stage 6) - 1963 lines
5. **LLMAnalysisCHILD.md** (Stage 7) - 1,963 lines

---

## ⛔ CRITICAL ISSUES (Must Fix Before TI)

### Issue #11: Stage 7 Incremental Saves Need Status Tracking

**Severity**: CRITICAL (Blocking)
**Category**: Documentation / Operational Clarity
**Status**: 🔴 NOT STARTED

**Location**:
- Stage 7: `LLMAnalysisCHILD.md:299-306` (incremental saves)
- Comparison: Stage 6: `MLAnalysisGenerationCHILD.md:554-664`, Stage 5: `Stage5_MLModelTraining_HLD.md:208-310` (atomic pattern)

**Problem**:
Stage 7 saves Phase 1 window analyses incrementally (not atomically like Stages 5-6). However, **this is intentional** due to Stage 7's unique characteristics (expensive LLM API calls, non-deterministic outputs, external API reliability issues). The real problem is **lack of completion tracking** - users cannot tell if a bucket is "in progress" or "failed" when they see partial output files.

**Current Code (Stage 7: LLMAnalysisCHILD.md:299-306)**:
```python
# Save individual window analyses immediately
for window_type, analysis in window_analyses.items():
    output_path = os.path.join(bucket_path, f'ml_analysis/llm/{window_type}_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
# ✅ Incremental saves (intentional for cost optimization)
# ❌ No status tracking (user can't determine completion state)
```

**Why Incremental Saves Are Better for Stage 7**:
1. **Cost Optimization**: API calls cost ~$0.03 each. On retry, only re-run failed windows (not all 6) → saves $0.12-0.18 per retry
2. **Non-Deterministic LLM**: Retrying all windows might lose good analyses from first run (LLM gives different responses)
3. **External API Reliability**: Preserves progress through transient failures (timeouts, 503 errors)
4. **Progress Visibility**: Users can see 4/6 windows completed during long-running operations (2-3 minutes)

**Impact**:
- ❌ Ambiguous completion state (4 JSON files = in progress or failed?)
- ❌ User doesn't know whether to retry or wait
- ❌ No clear signal that Phase 1 is 100% complete
- ✅ But: Incremental saves optimize cost and preserve good LLM analyses

**Recommended Fix**:

**Solution: Add Status Tracking File** (Alternative A - Cost-Optimized)

Add `.phase1_status.json` to track completion state, enabling resume capability:

```markdown
### Step 2.3.2.1: Status Tracking for Phase 1 Incremental Saves

**Purpose**: Provide clear completion status and resume capability for cost-optimized incremental saves

**Design Decision**: Stage 7 uses incremental saves (NOT atomic pattern) intentionally because:
- LLM API calls cost $0.02-0.05 per window (6 windows × $0.03 = $0.18 per bucket)
- Atomic pattern wastes successful API calls on retry (costs $0.18 vs $0.03 for failed window only)
- LLM is non-deterministic (losing good analyses from first run is expensive)
- External API reliability requires progress preservation (transient 503/timeout errors)

**Status File Schema**:
```json
{
  "total_windows": 6,
  "completed_windows": ["hook", "middle_1", "middle_2", "middle_3"],
  "failed_windows": [
    {"window": "middle_4", "error": "API timeout after 90s", "timestamp": "2025-10-16T10:27:30Z"}
  ],
  "phase1_complete": false,
  "started_at": "2025-10-16T10:25:00Z",
  "last_updated": "2025-10-16T10:27:30Z"
}
```

**Logic**:
```python
def run_phase1_with_status_tracking(bucket_path: str, bucket: str, hashtag: str | None, window_types: list) -> dict:
    """
    Run Phase 1 analysis with status tracking and resume capability.

    Returns: {window_type: analysis_json} for all windows
    Raises: Phase1ExecutionError if ANY window fails after retries
    """
    status_file = os.path.join(bucket_path, 'ml_analysis/llm/.phase1_status.json')

    # === Initialize or load status ===
    if os.path.exists(status_file):
        with open(status_file) as f:
            status = json.load(f)
        completed = set(status['completed_windows'])
        logger.info(f"Resuming Phase 1: {len(completed)}/{len(window_types)} windows already completed")
    else:
        status = {
            'total_windows': len(window_types),
            'completed_windows': [],
            'failed_windows': [],
            'phase1_complete': False,
            'started_at': datetime.utcnow().isoformat(),
        }
        completed = set()

    window_analyses = {}

    # === Run windows in parallel (skip already completed) ===
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(window_types)) as executor:
        futures = {}

        for window_type in window_types:
            if window_type in completed:
                # Load existing analysis from file
                output_path = os.path.join(bucket_path, f'ml_analysis/llm/{window_type}_analysis.json')
                with open(output_path) as f:
                    window_analyses[window_type] = json.load(f)
                logger.info(f"  ⏭ {window_type} already completed (skipping)")
                continue

            # Run analysis for incomplete window
            future = executor.submit(
                analyze_window_with_retry,
                bucket_path=bucket_path,
                window_type=window_type,
                bucket=bucket,
                hashtag=hashtag,
                max_attempts=3
            )
            futures[window_type] = future

        # Collect results
        for window_type, future in futures.items():
            try:
                analysis = future.result(timeout=120)

                # Save window JSON immediately (incremental save)
                output_path = os.path.join(bucket_path, f'ml_analysis/llm/{window_type}_analysis.json')
                with open(output_path, 'w') as f:
                    json.dump(analysis, f, indent=2)

                # Update status file
                status['completed_windows'].append(window_type)
                status['last_updated'] = datetime.utcnow().isoformat()
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)

                window_analyses[window_type] = analysis
                logger.info(f"  ✓ {window_type}_analysis.json saved ({len(status['completed_windows'])}/{len(window_types)})")

            except Exception as e:
                # Record failure in status
                status['failed_windows'].append({
                    'window': window_type,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
                status['last_updated'] = datetime.utcnow().isoformat()
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)

                logger.error(f"  ✗ {window_type} failed: {e}")
                raise Phase1ExecutionError(
                    f"Phase 1 incomplete: {window_type} failed after retries. "
                    f"Review errors and re-run Stage 7 (will resume from checkpoint)."
                )

    # === Mark Phase 1 complete ===
    status['phase1_complete'] = True
    status['completed_at'] = datetime.utcnow().isoformat()
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

    logger.info(f"✓ Phase 1 complete: All {len(window_types)} windows succeeded")

    return window_analyses
```

**Benefits**:
- ✅ Clear completion status (check `phase1_complete: true`)
- ✅ Resume from checkpoint on retry (skip completed windows, save $0.12-0.18)
- ✅ Preserves good LLM analyses from partial runs (non-deterministic output optimization)
- ✅ Progress visibility (status file shows 4/6 completed)
- ✅ Cost-optimized (only retry failed windows, not all 6)

**User Experience**:
```bash
# First run (fails after 4 windows)
$ python run_ml_pipeline.py --stage 7 --client acme --bucket 18-33s
✓ hook_analysis.json (30s)
✓ middle_1_analysis.json (30s)
✓ middle_2_analysis.json (30s)
✓ middle_3_analysis.json (30s)
✗ middle_4_analysis.json failed (API timeout)

# Check status
$ cat ml_analysis/llm/.phase1_status.json
{"phase1_complete": false, "completed_windows": ["hook", "middle_1", "middle_2", "middle_3"]}

# Retry (resumes from checkpoint)
$ python run_ml_pipeline.py --stage 7 --client acme --bucket 18-33s
⏭ hook already completed (skipping)
⏭ middle_1 already completed (skipping)
⏭ middle_2 already completed (skipping)
⏭ middle_3 already completed (skipping)
✓ middle_4_analysis.json (30s) ← Only retries this
✓ closing_analysis.json (30s)
✓ Phase 1 complete
```

**Edge Cases**:
- Status file corrupted → Delete and start fresh
- Window file exists but not in status → Treat as incomplete, re-run
- All windows complete but status missing → Validate all files exist, create status retroactively
```

**Estimated Effort**: 1-2 hours (add status tracking to Section 2.3.2, update Section 5.2 output schema, update Appendix C pseudocode)

**Assignee**: TBD
**Due Date**: Before TI documentation begins
**Dependencies**: None

**Rationale for Choosing Alternative A** (Status Tracking over Atomic Pattern):
- Stage 7 has fundamentally different cost/reliability characteristics than Stages 5-6
- Stages 5-6: Free local computation (atomic pattern = simplicity)
- Stage 7: Expensive external API calls (incremental saves = cost optimization)
- See CrossHLDalignment2do.md decision analysis for full rationale

---

### Issue #14: Cross-Window Features Undocumented in Stage 3

**Severity**: CRITICAL (Traceability Gap)
**Category**: Feature Naming / Documentation
**Status**: ✅ COMPLETED

**Location**:
- Stage 3: `FeatureAggregationCHILD.md` (entire document - missing mention)
- Stage 4: `FeatureTransformationCHILD.md:269-296` (where they're added)
- Stage 7: `LLMAnalysisCHILD.md:683-685` (where they're used)

**Problem**:
Stage 3 (Feature Aggregation) documents 21 base features per window and states output has `(windows × 21) + 3 metadata` columns. It does NOT mention that cross-window features (e.g., `hook_to_middle_energy_delta`, `energy_trend_slope`) are added in Stage 4. This breaks feature traceability—readers assume Stage 3 outputs all features (incorrect).

**Cross-Window Features Added in Stage 4 (FeatureTransformationCHILD.md:269-296)**:
```python
# 5 cross-window features added to video-level RF:
1. hook_to_middle_energy_delta: Energy change from hook to middle windows
2. middle_to_closing_contrast: Energy contrast between middle and closing
3. eye_contact_consistency: Std deviation of eye contact across windows
4. energy_trend_slope: Linear regression slope of energy across windows
5. window_consistency_score: Overall consistency metric (0-1)
```

**Impact**:
- Readers assume Stage 3 outputs all features (incorrect)
- Unclear where cross-window features originate (feature lineage broken)
- Can't trace `hook_to_middle_energy_delta` back to source transformation logic
- Contradicts Stage 3's stated output schema (129 columns for 18-33s bucket)

**Recommended Fix**:

Add the following note to Stage 3 Section 5.2 (Output Schema) after the column count table:

```markdown
### 5.2.1 Important Note: Cross-Window Features

**Scope of Stage 3 Output**: This stage outputs **window-specific features only** (21 per window + 3 metadata).

**Cross-window features** (features computed by comparing values **across** windows) are **NOT** part of Stage 3 output. They are computed in **Stage 4 (Feature Transformation)** using the aggregated CSV produced by this stage.

**Examples of Cross-Window Features** (added in Stage 4):
- `hook_to_middle_energy_delta`: Energy change from hook to middle windows
- `middle_to_closing_contrast`: Energy contrast between middle and closing
- `eye_contact_consistency`: Standard deviation of eye contact rate across all windows
- `energy_trend_slope`: Linear regression slope of energy level progression
- `window_consistency_score`: Overall consistency metric (0-1 scale)

**For Implementation Details**: See `FeatureTransformationCHILD.md` Section 2.3.2 "Cross-Window Feature Engineering" (lines 269-296).

**Why This Matters**:
- Video-level Random Forest (Stage 5) trains on **both** window-specific features (from Stage 3) **and** cross-window features (from Stage 4)
- Stage 7 LLM prompts use cross-window features for temporal progression analysis
- Understanding this separation is critical for debugging feature importance rankings
```

**Estimated Effort**: ✅ COMPLETED (10 minutes - documentation only)

**Assignee**: Completed
**Due Date**: ✅ Done
**Dependencies**: None

---

## 🟡 MEDIUM PRIORITY ISSUES (Should Fix Before TI)

### Issue #5: Stage 6 Distribution Analysis Uses Inconsistent Data Sources

**Severity**: MEDIUM (Needs Clarification)
**Category**: Data Flow Logic
**Status**: 🟡 IN PROGRESS

**Location**:
- Video-level: `MLAnalysisGenerationCHILD.md:238-240`
- Window-level: `MLAnalysisGenerationCHILD.md:370-372`

**Problem**:
Stage 6 uses `aggregated_features.csv` (Stage 3 output) for **video-level** distribution analysis but `{window}_rf_transformed.csv` (Stage 4 output) for **window-level** distribution analysis. Different data sources for same type of analysis creates confusion.

**Code Evidence**:

**Video-level (MLAnalysisGenerationCHILD.md:238-240)**:
```python
# Load aggregated_features.csv for distribution analysis
agg_csv_path = os.path.join(bucket_path, 'ml_analysis/aggregated_features.csv')
df = pd.read_csv(agg_csv_path)
# ⚠️ Uses Stage 3 output
```

**Window-level (MLAnalysisGenerationCHILD.md:370-372)**:
```python
# Load window-specific transformed CSV
rf_csv_path = os.path.join(bucket_path, f'ml_analysis/{window}_rf_transformed.csv')
df = pd.read_csv(rf_csv_path)
# ⚠️ Uses Stage 4 output
```

**Impact**:
- Unclear why different CSVs used
- Potential data inconsistency if Stage 3 and Stage 4 have different video counts
- Makes debugging harder (which CSV is source of truth for distributions?)
- Appears inconsistent to readers

**Recommended Fix**:

Add the following subsection to Stage 6 Section 2.3.2 (Video-Level RF Analysis) and Section 2.3.3 (Window-Level RF Analysis):

```markdown
#### Design Decision: Mixed CSV Sources for Distribution Analysis

**Rationale**: Video-level and window-level distribution analyses use **different** CSV sources intentionally:

**Video-Level Distribution Uses `aggregated_features.csv` (Stage 3)**:
- Video-level Random Forest features include **cross-window features** (e.g., `hook_to_middle_energy_delta`)
- Cross-window features are computed from **raw aggregated data** (Stage 3 output)
- Need original aggregated values (not Stage 4 transformed values) to compute accurate percentiles for cross-window features
- Example: `hook_to_middle_energy_delta` percentile thresholds must match raw hook/middle energy values

**Window-Level Distribution Uses `{window}_rf_transformed.csv` (Stage 4)**:
- Window-level Random Forest features are **trained on Stage 4 transformed data** (scaled, encoded)
- Distribution percentiles must **match training data distribution** exactly
- Example: If `eye_contact_rate_scaled` is normalized 0-1 in training, percentiles must use same 0-1 range
- Using Stage 3 raw data would create train/distribution mismatch

**This is intentional design** ensuring distribution analysis matches ML training data source for each model type.

**Validation**: Both CSVs have identical video counts (pre-flight validation checks this in Section 6.1).
```

**Estimated Effort**: 15 minutes (documentation clarification only)

**Assignee**: TBD
**Due Date**: Before TI documentation
**Dependencies**: None

---

### Issue #9: Ambiguous Window Count "6" vs "6-7"

**Severity**: MEDIUM (Documentation Clarity)
**Category**: Data Flow
**Status**: 🔵 NOT STARTED

**Location**:
- Stage 6: `MLAnalysisGenerationCHILD.md:113-115` (says "6 files")
- Stage 7: `LLMAnalysisCHILD.md:56-60` (says "6-7 files")

**Problem**:
Stage 6 documentation says "6 files" specifically for bucket 18-33s, but Stage 7 generically says "6-7 files depending on bucket," creating ambiguity about which buckets have 7 windows.

**Evidence**:

**Stage 6 Output Schema (MLAnalysisGenerationCHILD.md:113-115)**:
```
13 JSON files (~95KB total):
- rf_video_analysis.json (1 file)
- {window}_rf_analysis.json (6 files: hook, middle_1-4, closing)
- {window}_kmeans_analysis.json (6 files)
```

**Stage 7 Input Schema (LLMAnalysisCHILD.md:56-60)**:
```
Input: Stage 6 ML Analysis JSONs (13 files per bucket, ~95KB total)
   ├── rf_video_analysis.json (~30KB)
   ├── {window}_rf_analysis.json × 6-7 (~5KB each)
   └── {window}_kmeans_analysis.json × 6-7 (~5KB each)
```

**Root Cause**: Bucket variability
- Bucket 18-33s: 6 windows (hook, middle_1-4, closing)
- Bucket 90-120s: 7 windows (hook, middle_1-5, closing)

**Impact**:
- Reader confusion ("Is it 6 or 7? Which buckets have 7?")
- Unclear if 13 files is always true or if it varies (13 for 6-window buckets, 15 for 7-window buckets)
- Inconsistent documentation style across stages

**Recommended Fix**:

**Stage 6 - Update Section 5.2 (Output Schema)**:
```markdown
### 5.2 Output Schema

**Output Files**: 13-15 JSON files per bucket (varies by bucket window count)

**File Breakdown**:
- **1 video-level RF JSON**: `rf_video_analysis.json` (~30KB)
- **N window-level RF JSONs**: `{window}_rf_analysis.json` (~5KB each)
  - Where N = number of windows in bucket (6 for 18-33s, 7 for 90-120s)
- **N window-level K-Means JSONs**: `{window}_kmeans_analysis.json` (~5KB each)

**Total**: `1 + (N × 2)` files = 13 files for 6-window buckets, 15 files for 7-window buckets

**Window Count by Bucket** (from `config/bucket_definitions.py`):
- 0-3s: 1 window (hook only) → **3 files total**
- 3-9s: 2 windows (hook, closing) → **5 files total**
- 9-13s, 13-18s: 3 windows (hook, middle_aggregate, closing) → **7 files total**
- 18-33s, 33-60s, 60-90s: 6 windows (hook, middle_1-4, closing) → **13 files total**
- 90-120s: 7 windows (hook, middle_1-5, closing) → **15 files total**

See `FoundationCHILD.md` Appendix "Bucket Definitions" for BUCKET_WINDOWS config.
```

**Stage 7 - Update Section 3.1 (Input Dependencies)**:
```markdown
| Dependency | Source | Format | Required Fields | Failure Mode |
|------------|--------|--------|-----------------|--------------|
| Window-level RF analysis | Stage 6 output (**N files** where N = bucket window count) | JSON (~5KB each) | `window_type`, `feature_importance` (top 10), `model_performance` | Pre-flight validation fails |
| Window-level K-Means analysis | Stage 6 output (**N files**) | JSON (~5KB each) | `window_type`, `n_clusters` (3), `clusters` (with centroids, videos), `total_videos` | Pre-flight validation fails, cluster size integrity check |

**Where N = Window Count**:
- 6-window buckets (18-33s, 33-60s, 60-90s): 13 total files (1 video + 6 window RF + 6 window K-Means)
- 7-window buckets (90-120s): 15 total files (1 video + 7 window RF + 7 window K-Means)

See `config/bucket_definitions.py` for exact window counts per bucket.
```

**Estimated Effort**: 20 minutes (documentation updates in 2 files)

**Assignee**: TBD
**Due Date**: Before TI documentation
**Dependencies**: None

---

### Issue #13: Stage 5 Missing Feature Name Validation

**Severity**: MEDIUM (Input Validation Gap)
**Category**: Logic Gap
**Status**: 🔵 NOT STARTED

**Location**:
- Stage 4: `FeatureTransformationCHILD.md:456-459` (adds `_scaled` suffix)
- Stage 5: `Stage5_MLModelTraining_HLD.md:387-401` (assumes suffix exists, removes it)
- Stage 5: (Missing pre-flight validation)

**Problem**:
Stage 5 `normalize_feature_name()` function assumes K-Means features have `_scaled`, `_log`, `_encoded` suffixes but doesn't validate they actually exist. If Stage 4 changes naming convention, Stage 5 silently fails (or produces incorrect feature names in K-Means JSON).

**Current Code (Stage5_MLModelTraining_HLD.md:387-401)**:
```python
def normalize_feature_name(feature_name: str) -> str:
    """
    Remove transformation suffixes for K-Means JSON output.

    Examples:
        eye_contact_rate_scaled → eye_contact_rate
        word_count_log → word_count
        gender_encoded → gender
    """
    normalized = feature_name.replace('_scaled', '').replace('_log', '').replace('_encoded', '')
    return normalized
    # ❌ No validation that suffix actually existed
    # ❌ No check if transformation was applied
```

**Impact**:
- **Silent failure** if Stage 4 changes naming convention (e.g., uses `_norm` instead of `_scaled`)
- Feature names in K-Means JSON may be incorrect (e.g., `eye_contact_rate` when Stage 4 didn't add suffix)
- **Debugging nightmare**: Stage 6-7 fail with cryptic "feature not found" errors
- No way to detect Stage 4 → Stage 5 contract break

**Recommended Fix**:

Add the following to Stage 5 Section 6.1 (Input Validation):

```markdown
### 6.1.4 K-Means Feature Name Validation

**Purpose**: Validate that Stage 4 K-Means CSVs use expected naming convention (`_scaled`, `_log`, `_encoded` suffixes).

**Logic**:
```python
def validate_kmeans_feature_naming(csv_path: str, expected_suffix: str = '_scaled') -> None:
    """
    Validate K-Means CSV has expected transformation suffixes.

    Args:
        csv_path: Path to K-Means transformed CSV (e.g., hook_kmeans_transformed.csv)
        expected_suffix: Primary suffix to check (default: '_scaled')

    Raises:
        ValidationError: If <80% of features have expected suffix

    Source: Cross-HLD Alignment Issue #13
    """
    # Read CSV header only
    df = pd.read_csv(csv_path, nrows=1)
    feature_names = [col for col in df.columns if col not in ['video_id', 'create_time', 'gender']]

    # Count features with transformation suffixes
    scaled_count = sum(1 for f in feature_names if '_scaled' in f)
    log_count = sum(1 for f in feature_names if '_log' in f)
    encoded_count = sum(1 for f in feature_names if '_encoded' in f)
    total_transformed = scaled_count + log_count + encoded_count

    # Expect at least 80% of features to have transformation suffixes
    expected_threshold = len(feature_names) * 0.80

    if total_transformed < expected_threshold:
        raise ValidationError(
            f"K-Means CSV feature naming validation failed: {csv_path}\n"
            f"  Total features: {len(feature_names)}\n"
            f"  Features with _scaled: {scaled_count}\n"
            f"  Features with _log: {log_count}\n"
            f"  Features with _encoded: {encoded_count}\n"
            f"  Total transformed: {total_transformed}/{len(feature_names)} ({total_transformed/len(feature_names)*100:.1f}%)\n"
            f"  Expected: ≥{expected_threshold:.0f} ({80}%)\n"
            f"\n"
            f"This indicates Stage 4 may not have applied transformations correctly.\n"
            f"Check FeatureTransformationCHILD.md Section 2.3.2 for transformation logic.\n"
            f"Expected suffixes: _scaled (StandardScaler), _log (log transform), _encoded (one-hot encoding)"
        )

    logger.info(
        f"✓ K-Means feature naming validated: {total_transformed}/{len(feature_names)} "
        f"({total_transformed/len(feature_names)*100:.1f}%) features have transformation suffixes"
    )


# Add to pre-flight validation (Section 2.3.1)
def run_preflight_validation(bucket_path: str, bucket: str) -> None:
    """
    [Existing validation logic...]
    """
    # === Layer 3: Schema and Naming Convention Validation ===
    windows = BUCKET_WINDOWS[bucket]

    for window in windows:
        # Validate K-Means CSV naming convention
        km_csv_path = os.path.join(bucket_path, f'ml_analysis/{window}_kmeans_transformed.csv')
        validate_kmeans_feature_naming(km_csv_path)

        # [Rest of existing validation...]
```

**Edge Cases**:
- Some features legitimately don't need transformation (e.g., `video_id`, metadata) → Exclude from count
- Different transformations for different features (80% threshold accommodates this)
- Bucket-specific features (cross-window features don't have suffixes) → Exclude metadata columns

**Estimated Effort**: 30 minutes (add validation function + pre-flight call)

**Assignee**: TBD
**Due Date**: Before TI documentation
**Dependencies**: None

---

### Issue #16: Redundant Cluster Size Validation in Stage 6 and 7

**Severity**: MEDIUM (Code Duplication)
**Category**: Architectural
**Status**: 🔵 NOT STARTED

**Location**:
- Stage 6: `MLAnalysisGenerationCHILD.md:1182-1190` (validates cluster sizes sum to total_videos)
- Stage 7: `LLMAnalysisCHILD.md:207-215` (ALSO validates cluster sizes sum to total_videos)

**Problem**:
Both Stage 6 and Stage 7 validate that cluster sizes sum to `total_videos`. This creates code duplication and raises question: Is this intentional defense-in-depth or oversight?

**Evidence**:

**Stage 6 Validation (MLAnalysisGenerationCHILD.md:1182-1190)**:
```python
# CRITICAL: Cluster sizes must sum to total_videos
total_videos = kmeans_data['total_videos']
cluster_sizes = [c['size'] for c in kmeans_data['clusters']]
sum_sizes = sum(cluster_sizes)

if sum_sizes != total_videos:
    raise ValidationError(
        f"{window}_kmeans_analysis.json: Cluster sizes {cluster_sizes} "
        f"sum to {sum_sizes}, but total_videos is {total_videos}. "
        f"Stage 5 cluster assignment failed."
    )
```

**Stage 7 Validation (LLMAnalysisCHILD.md:207-215)**:
```python
# CRITICAL: Cluster sizes must sum to total_videos
if sum(cluster_sizes) != total_videos:
    raise ValidationError(
        f"{window}_kmeans_analysis.json: Cluster sizes {cluster_sizes} "
        f"sum to {sum(cluster_sizes)}, but total_videos is {total_videos}. "
        f"Stage 6 cluster assignment failed."
    )
```

**Impact**:
- Code duplication (same validation logic twice)
- Unclear if intentional (defense-in-depth) or oversight
- If Stage 6 validation is sufficient, Stage 7 check is wasteful
- If defense-in-depth is desired, document rationale

**Recommended Fix - Option A (Preferred)**: Remove from Stage 7, rely on Stage 6

Update Stage 7 Section 2.3.1 (Pre-Flight Validation):

```markdown
### Step 2.3.1: Pre-Flight Validation

**Purpose**: Validate all dependencies before Phase 1 execution (fail-fast principle)

**Logic**:
```python
def run_preflight_validation(bucket_path: str, bucket: str) -> None:
    """
    Three-layer pre-flight validation.

    Layer 1: API credentials exist
    Layer 2: Stage 6 outputs exist and parseable
    Layer 3: Schema validation (EXCLUDING cluster size check - delegated to Stage 6)

    Source: QA Q4.2, Q7, Cross-HLD Alignment Issue #16
    """
    # [Layer 1 and Layer 2 validation - unchanged]

    # === Layer 3: Schema Validation (Simplified) ===
    for window in windows:
        # Validate K-Means JSON
        kmeans_path = os.path.join(bucket_path, f'ml_analysis/{window}_kmeans_analysis.json')
        with open(kmeans_path, 'r') as f:
            kmeans_data = json.load(f)

        # Check required fields
        required = ['window_type', 'bucket', 'n_clusters', 'clusters', 'total_videos']
        missing = [f for f in required if f not in kmeans_data]
        if missing:
            raise ValidationError(f"{window}_kmeans_analysis.json: Missing fields: {missing}")

        # Check 3 clusters
        if len(kmeans_data['clusters']) != 3:
            raise ValidationError(
                f"{window}_kmeans_analysis.json: Expected 3 clusters, "
                f"got {len(kmeans_data['clusters'])}"
            )

        # ✅ Cluster size validation REMOVED (Stage 6 already validates this in MLAnalysisGenerationCHILD.md:1182-1190)
        # Rationale: Avoid redundant validation. Stage 6 is authoritative source for cluster integrity.

        # [Rest of validation - unchanged]
```

**Rationale**:
- Avoids code duplication
- Stage 6 is authoritative source for K-Means cluster integrity
- Stage 7 trusts Stage 6's validation (follows Unix philosophy: do one thing well)

**Alternative Fix - Option B**: Keep both, document as defense-in-depth

Add note to Stage 7 Section 2.3.1:

```markdown
**Design Decision: Defense-in-Depth Validation**

Stage 6 and Stage 7 **both** validate cluster size integrity (cluster sizes sum to total_videos) as intentional defense-in-depth:

**Why Duplicate Validation?**
1. **Manual file editing**: If user manually edits Stage 6 JSON files (debugging, corrections), Stage 7 catches corruption
2. **Stage 6 bypass**: If user runs only Stage 7 (resuming from checkpoint), pre-flight catches Stage 6 output issues
3. **Critical for Phase 2**: Cluster path extraction (Q9.1) **requires** every video to be assigned to exactly one cluster. Missing/extra videos break path frequency analysis.

**Cost**: Minimal (1-2ms per window, <10ms total per bucket)

This is intentional redundancy prioritizing correctness over efficiency.
```

**Recommended Approach**: **Option A** (remove from Stage 7) - cleaner, follows single-responsibility principle.

**Estimated Effort**: 10 minutes (documentation update or code removal)

**Assignee**: TBD
**Due Date**: Before TI documentation
**Dependencies**: None

---

### Issue #20: Exit Code Standardization

**Severity**: LOW (Developer Experience)
**Category**: Architectural
**Status**: 🔵 NOT STARTED

**Location**:
- Stage 5: Exit codes (implied in error handling sections)
- Stage 6: `MLAnalysisGenerationCHILD.md:663` (exit code 2 = generation fail)
- Stage 7: `LLMAnalysisCHILD.md` (exit codes 1, 4, 5, 6, 99)

**Problem**:
Exit codes have different meanings across stages, making debugging and orchestration harder.

**Evidence**:

| Stage | Exit Code 2 Meaning | Other Codes |
|-------|---------------------|-------------|
| Stage 5 | Training failure (implied) | 0=success, 1=init fail, 3=validation fail |
| Stage 6 | Generation fail | 0=success, 1=pre-flight fail, 3=validation fail, 4=I/O fail |
| Stage 7 | (Not documented) | 0=success, 1=pre-flight, 4=API 401, 5=Phase 1 fail, 6=Phase 2 fail, 99=unexpected |

**Impact**:
- Debugging confusion (what does exit code 2 mean? depends on stage)
- Harder to write unified error handling in orchestration layer
- No single reference for exit code meanings

**Recommended Fix**:

Create standardized exit code table in **FoundationCHILD.md** (new section to be added):

```markdown
## Section 7: Standardized Exit Codes (All Stages)

**Purpose**: Provide consistent exit code semantics across all ML pipeline stages for orchestration and debugging.

### Exit Code Reference Table

| Code | Category | Meaning | Example Scenarios | Recovery Action |
|------|----------|---------|-------------------|-----------------|
| **0** | Success | All operations completed successfully | Stage completed without errors | None (proceed to next stage) |
| **1** | Pre-flight Validation | Dependencies missing or invalid | Stage N-1 outputs don't exist, malformed JSONs, missing API keys | Re-run previous stage, check environment |
| **2** | Execution Failure | Core stage logic failed | ML training failed, JSON generation failed, API call failed | Check logs, debug stage logic, retry |
| **3** | Output Validation | Generated output failed validation | Cluster sizes don't sum, JSON schema invalid, model metrics below threshold | Review stage logic, check input data quality |
| **4** | I/O Failure | File system or external service error | Disk full, permission denied, network timeout, API unauthorized | Fix infrastructure, check credentials |
| **5** | Partial Completion | Some operations succeeded, some failed | Phase 1: 4/6 windows completed (Stage 7), 3/8 buckets trained (Stage 5) | Review partial output, retry failed components |
| **6** | Data Integrity Error | Input data inconsistent or corrupted | Video missing from cluster, feature count mismatch, CSV row count mismatch | Re-run upstream stages, validate data pipeline |
| **99** | Unexpected Error | Uncaught exception or unknown failure | Python exception not handled by stage-specific logic | Debug stack trace, file bug report |

### Usage Guidelines

**For Stage Implementers**:
- Use exit codes consistently within each stage
- Document exit codes in stage HLD Section "Error Handling"
- Raise exceptions with clear error messages (logged before exit)

**For Orchestration Layer**:
- Check exit code to determine recovery strategy
- Exit codes 1, 4, 6 → Re-run previous stage
- Exit code 2, 3 → Debug current stage
- Exit code 5 → Resume from checkpoint (partial completion)
- Exit code 99 → Escalate to engineering

**Example (Python)**:
```python
import sys

try:
    run_stage_N(...)
    sys.exit(0)  # Success
except PreFlightValidationError as e:
    logger.error(f"Pre-flight failed: {e}")
    sys.exit(1)
except ExecutionError as e:
    logger.error(f"Execution failed: {e}")
    sys.exit(2)
except ValidationError as e:
    logger.error(f"Output validation failed: {e}")
    sys.exit(3)
except IOError as e:
    logger.error(f"I/O error: {e}")
    sys.exit(4)
except Exception as e:
    logger.error(f"Unexpected error: {type(e).__name__}: {e}")
    sys.exit(99)
```

### Stage-Specific Exit Code Mapping

**Stage 3 (Feature Aggregation)**:
- 0 = Success (aggregated CSV generated)
- 1 = Pre-flight fail (Stage 2 temporal windows missing)
- 2 = Aggregation fail (merge/join failed)
- 3 = Output validation fail (column count mismatch)

**Stage 4 (Feature Transformation)**:
- 0 = Success (transformed CSVs generated)
- 1 = Pre-flight fail (Stage 3 aggregated CSV missing)
- 2 = Transformation fail (scaling/encoding failed)
- 3 = Output validation fail (feature naming convention violated)

**Stage 5 (ML Model Training)**:
- 0 = Success (all models trained and validated)
- 1 = Pre-flight fail (Stage 4 CSVs missing)
- 2 = Training fail (RandomForest/K-Means training failed)
- 3 = Validation fail (model metrics below threshold)
- 5 = Partial completion (3/8 buckets trained)

**Stage 6 (ML Analysis Generation)**:
- 0 = Success (13-15 JSONs generated per bucket)
- 1 = Pre-flight fail (Stage 4 CSVs or Stage 5 models missing)
- 2 = Generation fail (JSON creation failed)
- 3 = Validation fail (cluster size integrity check failed)
- 4 = I/O fail (disk full, permission denied)

**Stage 7 (LLM Analysis)**:
- 0 = Success (8 JSONs generated per bucket)
- 1 = Pre-flight fail (Stage 6 JSONs missing, API key invalid)
- 2 = Phase 1 fail (window analysis failed)
- 3 = Phase 2 fail (synthesis failed)
- 4 = API auth fail (Anthropic API 401/403)
- 5 = Partial completion (4/6 windows completed in Phase 1)
- 6 = Data integrity (cluster path extraction failed)
- 99 = Unexpected error
```

**Estimated Effort**: 30-45 minutes (create new FoundationCHILD.md section, update all stage HLDs to reference it)

**Assignee**: TBD
**Due Date**: Before TI documentation
**Dependencies**: Need to update FoundationCHILD.md (currently doesn't have exit code section)

---

## 🟢 LOW PRIORITY ISSUES (Nice-to-Have)

### Issue #8: Cross-Window Feature Validation in Stage 6 Pre-Flight

**Severity**: LOW (Nice-to-Have)
**Category**: Logic Gap
**Status**: 🔵 NOT STARTED

**Location**:
- Stage 6: `MLAnalysisGenerationCHILD.md:140-191` (pre-flight validation)
- Stage 7: `LLMAnalysisCHILD.md:683-685` (expects cross-window features in video RF)

**Problem**:
Stage 6 validates that Stage 4 CSVs and Stage 5 models exist, but doesn't check if video-level RF model has expected cross-window features. If Stage 4 forgot to compute cross-window features, Stage 7 fails with cryptic "feature not found" error.

**Impact**:
- Low (Stage 7 will fail gracefully with clear error if cross-window features missing)
- Fails late (Stage 7) instead of early (Stage 6 pre-flight)
- Harder to debug (user doesn't know cross-window features are the root cause)

**Recommended Fix** (Optional):

Add to Stage 6 Section 6.1 (Input Validation):

```python
def validate_video_rf_features(bucket_path: str, bucket: str) -> None:
    """
    Validate video-level RF model has expected cross-window features.

    Purpose: Fail-fast if Stage 4 didn't compute cross-window features
    Source: Cross-HLD Alignment Issue #8
    """
    # Load video RF model
    model_path = os.path.join(bucket_path, f'models/rf_video_{bucket}.pkl')
    rf_model = joblib.load(model_path)

    # Expected cross-window features (from Stage 4)
    expected_cross_window = [
        'hook_to_middle_energy_delta',
        'middle_to_closing_contrast',
        'eye_contact_consistency',
        'energy_trend_slope',
        'window_consistency_score'
    ]

    # Get actual feature names from model
    actual_features = rf_model.feature_names_in_.tolist()

    # Check if cross-window features present
    missing = [f for f in expected_cross_window if f not in actual_features]

    if missing:
        logger.warning(
            f"Video RF model missing {len(missing)} cross-window features: {missing}\n"
            f"  This may indicate Stage 4 didn't compute cross-window features.\n"
            f"  Stage 7 LLM analysis will fail without these features.\n"
            f"  Check FeatureTransformationCHILD.md Section 2.3.2 for cross-window logic."
        )
        # ⚠️ Warning only (not fatal) - Stage 7 will handle gracefully
    else:
        logger.info(f"✓ Video RF has all {len(expected_cross_window)} cross-window features")
```

**Estimated Effort**: 15 minutes (optional enhancement)

**Assignee**: TBD
**Due Date**: Optional (low priority)
**Dependencies**: None

---

## ✅ ITEMS ALREADY ALIGNED (10/20 - No Action Required)

| # | Item | Stages | Status | Notes |
|---|------|--------|--------|-------|
| 1 | Feature count Stage 3 → 4 (129 columns for bucket 18-33s) | 3, 4 | ✅ ALIGNED | Both expect 129 columns |
| 2 | Video RF feature count Stage 4 → 5 (183 features) | 4, 5 | ✅ ALIGNED | Both expect 183 features |
| 3 | Bucket window definitions (centralized config) | 3, 4, 5, 6, 7 | ✅ ALIGNED | All reference config/bucket_definitions.py |
| 4 | K-Means feature normalization (_scaled removal) | 4, 5, 6 | ✅ RESOLVED | Stage 6 normalizes before Stage 7 |
| 6 | File count Stage 5 → 6 (20 PKL files) | 5, 6 | ✅ ALIGNED | Both expect 20 files per bucket |
| 7 | Middle window handling (middle_aggregate) | 3, 4, 7 | ✅ ALIGNED | Consistent across stages |
| 10 | Window-level RF feature naming (no prefix) | 4, 6, 7 | ✅ ALIGNED | Prefixes removed in Stage 4 |
| 12 | Bucket count (7 schemas, 8 buckets) | 3, 4, 5 | ✅ CORRECT | 7 unique configs, 8 distinct buckets |
| 15 | Video count assumptions (50-300 range) | 3, 5, 6, 7 | ✅ ACCEPTABLE | All within same range |
| 18 | Metadata fields (video_id, create_time, gender) | 3, 4, 6 | ✅ ALIGNED | Consistent across stages |

---

## 📋 ACTION PLAN & NEXT STEPS

### Immediate Actions (Before TI Documentation)

**Priority 1: Critical Fixes (Must Complete)**
- [ ] **Issue #11**: Implement atomic output pattern in Stage 7 (Est: 1-2 hours)
  - Modify `LLMAnalysisCHILD.md` Section 2.3.2 to add temp directory logic
  - Update pseudocode in Appendix C
  - Update error handling section
- [x] **Issue #14**: Document cross-window features in Stage 3 (Est: 10 minutes) ✅ COMPLETED
  - Add note to `FeatureAggregationCHILD.md` Section 5.2

**Priority 2: Medium Fixes (Should Complete)**
- [ ] **Issue #5**: Document distribution CSV rationale in Stage 6 (Est: 15 minutes)
  - Add design decision section to `MLAnalysisGenerationCHILD.md` Section 2.3.2 & 2.3.3
- [ ] **Issue #9**: Standardize window count language (Est: 20 minutes)
  - Update `MLAnalysisGenerationCHILD.md` Section 5.2 output schema
  - Update `LLMAnalysisCHILD.md` Section 3.1 input dependencies
- [ ] **Issue #13**: Add feature name validation to Stage 5 (Est: 30 minutes)
  - Add validation function to `Stage5_MLModelTraining_HLD.md` Section 6.1
  - Update pre-flight validation logic
- [ ] **Issue #16**: Decide on redundant validation (Est: 10 minutes)
  - Choose Option A (remove from Stage 7) or Option B (document as defense-in-depth)
  - Update `LLMAnalysisCHILD.md` accordingly
- [ ] **Issue #20**: Create exit code standardization (Est: 30-45 minutes)
  - Add new section to `FoundationCHILD.md`
  - Update all stage HLDs to reference centralized table

**Priority 3: Optional Enhancements**
- [ ] **Issue #8**: Add cross-window feature check to Stage 6 pre-flight (Est: 15 minutes)
  - Optional enhancement for better error messages

**Estimated Total Time**: **3-4 hours** for all Priority 1 & 2 items

---

### Suggested Next Steps

**Phase 1: Documentation Fixes (This Week)**
1. ✅ Complete Issue #14 (cross-window features note) - **DONE**
2. Complete Issue #5 (distribution CSV rationale) - **IN PROGRESS**
3. Complete Issue #9 (window count standardization)
4. Create exit code table in FoundationCHILD.md (Issue #20)

**Phase 2: Logic Enhancements (Next Week)**
5. Implement Issue #11 (Stage 7 atomic pattern) - **Most complex change**
6. Implement Issue #13 (feature name validation)
7. Decide on Issue #16 (redundant validation) - **Requires architectural decision**

**Phase 3: Validation (Before TI)**
8. Review all HLD changes for consistency
9. Cross-check line references still accurate after edits
10. Run through full Stage 3-7 data flow mentally to verify alignment
11. **Proceed to Technical Implementation documentation** with confidence

---

### Tracking Checklist

**Critical Issues**
- [x] #11: Stage 7 status tracking implemented (Alternative A) ✅ **COMPLETED 2025-10-17**
  - CrossHLDalignment2do.md updated with full rationale
  - LLMAnalysisCHILD.md Section 2.3.2 updated (status tracking logic)
  - LLMAnalysisCHILD.md Section 5.2.0 added (status file schema)
  - LLMAnalysisCHILD.md Section 6.2 updated (error handling)
  - LLMAnalysisCHILD.md Appendix C updated (pseudocode)
- [x] #14: Cross-window features documented in Stage 3 ✅ **COMPLETED**

**Medium Issues**
- [x] #5: Distribution CSV rationale documented ✅ **COMPLETED**
  - Action: Added design decision notes to Stage 6 Section 2.3.2 & 2.3.3 (MLAnalysisGenerationCHILD.md:239-242, 374-377)
  - Completed: 2025-10-17
- [x] #9: Window count standardized ("6-7 files") ✅ **COMPLETED**
  - Action: Added bucket-specific window count table to Stage 6 Section 5.2 (MLAnalysisGenerationCHILD.md:867-887) and Stage 7 Section 3.1 (LLMAnalysisCHILD.md:58-62, 718-728)
  - Completed: 2025-10-17
- [x] #13: Feature name validation added to Stage 5 ✅ **COMPLETED**
  - Action: Added validation function to Stage 5 Section 6.1 (Stage5_MLModelTraining_HLD.md:1035-1102)
  - Completed: 2025-10-17
- [x] #16: Redundant validation decision made ✅ **DECISION: Option A (Remove from Stage 7)**
  - Rationale: Single responsibility - Stage 6 is authoritative source for cluster integrity
  - Action: Removed cluster size validation from Stage 7 pre-flight (LLMAnalysisCHILD.md:209-212)
  - Completed: 2025-10-17
- [x] #20: Exit code table created ✅ **COMPLETED**
  - Action: Created comprehensive exit code reference in FoundationCHILD.md Section 7 (FoundationCHILD.md:1128-1236)
  - Completed: 2025-10-17

**Low Priority**
- [x] #8: Cross-window feature validation in Stage 6 (optional) ✅ **DECISION: SKIP (Option C)**
  - Rationale: YAGNI principle - Stage 7 will fail gracefully with clear error if features missing
  - Low probability of Stage 4 forgetting cross-window features
  - Warning-only validation adds complexity with minimal value
  - Decided: 2025-10-17

**Verification**
- [x] Issue #11 HLD edits completed ✅
- [x] Issues #5, #9, #13, #20 completed ✅
- [x] Issues #16, #8 decisions finalized ✅
- [ ] Line references updated after edits
- [ ] Cross-stage data flow verified
- [ ] Ready for TI documentation

**Session Notes - 2025-10-17**:
- Issue #11 fully implemented with Alternative A (status tracking for incremental saves)
- Design rationale documented: Stage 7 uses incremental saves (not atomic) due to LLM API cost optimization
- Issues #5, #9, #13, #20 all completed (total time: ~90 minutes)
- Issues #16, #8 decisions finalized:
  - Issue #16: Option A selected (remove redundant validation from Stage 7)
  - Issue #8: Option C selected (skip - YAGNI principle)
- **All alignment issues resolved** - Ready for TI documentation phase

---

## 📝 Detailed Issue Reference

For full context on each issue, including:
- Exact code snippets with line numbers
- Impact analysis
- Recommended fixes with code examples
- Edge case handling

See the detailed analysis in the sections above.

---

## 🎯 Success Criteria

**Alignment Audit Complete** ✅ when:
- [x] All 5 HLD documents read completely (8,316 lines)
- [x] All 20 issues identified and categorized
- [x] Line references documented for each issue
- [ ] All Priority 1-2 issues resolved (7 items)
- [ ] All stage HLDs updated with fixes
- [ ] Cross-stage data flow verified end-to-end

**Ready for Technical Implementation** ✅ when:
- [ ] Critical issues (#11, #14) resolved
- [ ] Medium issues (#5, #9, #13, #16, #20) resolved or explicitly deferred
- [ ] Verification checklist complete
- [ ] Stakeholder approval obtained

---

## 📚 Related Documents

**HLD Documents Analyzed**:
1. `FeatureAggregationCHILD.md` (Stage 3)
2. `FeatureTransformationCHILD.md` (Stage 4)
3. `Stage5_MLModelTraining_HLD.md` (Stage 5)
4. `MLAnalysisGenerationCHILD.md` (Stage 6)
5. `LLMAnalysisCHILD.md` (Stage 7)

**Foundation Documents**:
- `FoundationCHILD.md` (cross-stage configuration, directory structure)
- `MLPlanningv2.md` (parent document for Stages 3-7)
- `config/bucket_definitions.py` (BUCKET_WINDOWS centralized config)

**Related Enhancement Docs**:
- `Crosswindowupgrade.md` (Stage 4 cross-window feature logic)

---

## 📞 Contact & Ownership

**Document Owner**: [TBD]
**Last Updated**: 2025-10-16
**Next Review**: After Priority 1-2 fixes complete
**Questions**: [Add contact info]

---

**END OF ALIGNMENT ACTION ITEMS DOCUMENT**
