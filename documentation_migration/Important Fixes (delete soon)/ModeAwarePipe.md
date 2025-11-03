# Mode-Aware Pipeline Fix Documentation

## Problem
TOP mode (`selection_strategy: "top"`) fails at Stage 5 validation with missing RF model files.

**Error:**
```
Stage 5 output validation failed for bucket 3-9s: Expected output missing:
/home/jorge/rumiaifinal/data/clients/rollo/competitors/gnclivewell/top_top/buckets/bucket_3-9s/models/rf_video_3-9s.pkl
```

**Root Cause:**
- TOP mode only has top performers (no bottom 20% for comparison)
- Random Forest requires labeled data (top vs bottom classes)
- Stage 5 correctly skips RF training when single class detected
- Stages 6, 7, and orchestrator validation assume RF models always exist

---

## Discovery Results

### Stage 5: ✅ Correctly Handles TOP Mode
**File:** `rumiai_v2/processors/model_training.py:460-471`

```python
# Already detects single class and skips RF
unique_labels = X_check['is_top_performer'].unique()
can_train_rf = len(unique_labels) >= 2

if not can_train_rf:
    logger.info("Skipping Random Forest: Single class detected in 'top' mode")
    # Creates model_metrics.json with trained: False
```

**Outputs in TOP mode:**
- ✅ `model_metrics.json` (with `trained: False` for RF)
- ✅ K-Means models for all windows
- ❌ NO RF models (expected behavior)

---

### Stage 6: ❌ Breaks - Hard RF Requirements
**File:** `ml_pipeline/stage6_analysis/ml_analysis_generation.py`

**Issue 1: Pre-flight validation (Lines 110-136)**
```python
required_stage5_files = [
    f'models/rf_video_{bucket}.pkl',  # ❌ ASSUMES ALWAYS EXISTS
]
for window in windows:
    required_stage5_files.append(f'models/rf_{window}_{bucket}.pkl')  # ❌ ASSUMES
```

**Issue 2: RF JSON generation (Lines 166-287)**
```python
def generate_video_rf_json(bucket_path: str, bucket: str) -> dict:
    model_path = os.path.join(bucket_path, f'models/rf_video_{bucket}.pkl')
    rf_model = joblib.load(model_path)  # ❌ CRASHES IF MISSING
```

**Issue 3: Distribution analysis (Lines 234-273)**
```python
bottom_performers = df[df['is_top_performer'] == 0][feature_name]  # ❌ EMPTY
bottom_avg = float(bottom_performers.mean())  # ❌ NaN
```

---

### Stage 7: ❌ Assumes Contrastive Data
**File:** `ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py:409-421`

```python
rf_video_path = os.path.join(bucket_path, 'ml_analysis/rf_video_analysis.json')
if not os.path.exists(rf_video_path):
    raise FileNotFoundError(f"RF video file not found")  # ❌ FAILS TOP MODE
```

**File:** `ml_pipeline/stage7_llm_analysis/stage7_prompts.py:386-389`
```python
prompt += f"Top: avg {feature['top_performer_avg']:.2f} "
prompt += f"Bottom: avg {feature['bottom_performer_avg']:.2f}"  # ❌ ASSUMES EXISTS
```

---

### Orchestrator: ❌ Enforces RF Files
**File:** `rumiai_ml_batch.py`

**Lines 177-222: validate_stage_6_prerequisites()**
```python
required_files.append(bucket_path_obj / "models" / f"rf_video_{bucket_name}.pkl")  # Line 205
for window in windows:
    required_files.append(bucket_path_obj / "models" / f"rf_{window}_{bucket_name}.pkl")  # Line 208
```

**Lines 288-330: validate_stage7_prerequisites()**
```python
required_files = [
    os.path.join(ml_analysis_dir, "rf_video_analysis.json"),  # Line 311
]
```

---

## Solution: Conditional RF Logic

### Pattern
Instead of hard-coding RF requirements:
1. Check `model_metrics.json` first to see if RF was trained
2. If `trained: false` → skip RF validation/generation
3. If `trained: true` → require RF files (backward compatible)

---

### Fix 1: Stage 6 Validation with Cross-Check
**File:** `ml_pipeline/stage6_analysis/ml_analysis_generation.py:71-163`

```python
def validate_stage_dependencies(bucket_path: str, bucket: str, windows: List[str]) -> None:
    # Load model_metrics.json to check if RF was trained
    metrics_path = os.path.join(bucket_path, 'models/model_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    rf_trained = metrics.get('video_level_rf', {}).get('trained', True)

    # Cross-validate: Check if video-level RF file exists
    rf_video_path = os.path.join(bucket_path, f'models/rf_video_{bucket}.pkl')
    rf_video_exists = os.path.exists(rf_video_path)

    # Case 1: Metrics says trained but file missing → Stage 5 incomplete
    if rf_trained and not rf_video_exists:
        raise ValueError(
            f"Stage 5 incomplete: model_metrics.json says RF trained, "
            f"but rf_video_{bucket}.pkl missing. Re-run Stage 5."
        )

    # Case 2: Metrics says not trained but file exists → Stale from previous run
    elif not rf_trained and rf_video_exists:
        logger.warning(
            f"⚠️ Stale RF model files detected from previous run. "
            f"model_metrics.json says trained=False (TOP mode), ignoring stale files."
        )
        # Trust metrics - don't change rf_trained

    if rf_trained:
        # Require RF model files
        required_stage5_files = [f'models/rf_video_{bucket}.pkl']
        for window in windows:
            required_stage5_files.append(f'models/rf_{window}_{bucket}.pkl')
        logger.info("RF models expected (trained=True)")
    else:
        # Skip RF files
        required_stage5_files = []
        logger.info("RF models NOT expected (trained=False - TOP mode)")

    # K-Means always required
    for window in windows:
        required_stage5_files.append(f'models/{window}_kmeans_{bucket}.pkl')
```

**Decision: Trust metrics.json, fail fast on inconsistency**
- RF files cannot be created in TOP mode (Stage 5 logic prevents this)
- If `trained=True` but files missing → Stage 5 failed, hard fail
- If `trained=False` but files exist → stale from previous CONTRASTIVE run, warn + ignore

---

### Fix 2: Stage 6 RF JSON Generation
**File:** `ml_pipeline/stage6_analysis/ml_analysis_generation.py:166-287`

```python
def generate_video_rf_json(bucket_path: str, bucket: str) -> Optional[dict]:
    """Returns None if RF model not found (TOP mode)"""
    model_path = os.path.join(bucket_path, f'models/rf_video_{bucket}.pkl')

    if not os.path.exists(model_path):
        logger.info("RF video model not found (TOP mode) - skipping")
        return None

    # Proceed with RF analysis...
```

**File:** `ml_pipeline/stage6_analysis/ml_analysis_generation.py:290-386`

```python
def generate_window_rf_json(bucket_path: str, bucket: str, window: str) -> Optional[dict]:
    """Returns None if RF model not found (TOP mode)"""
    model_path = os.path.join(bucket_path, f'models/rf_{window}_{bucket}.pkl')

    if not os.path.exists(model_path):
        logger.info(f"RF window model not found for {window} (TOP mode) - skipping")
        return None

    # Proceed with RF analysis...
```

---

### Fix 3: Stage 6 Main Logic
**File:** `ml_pipeline/stage6_analysis/ml_analysis_generation.py:641-737`

```python
def generate_ml_analysis_jsons(bucket_path: str, bucket: str, windows: List[str]) -> int:
    # Generate Video-Level RF JSON (optional)
    video_rf_json = generate_video_rf_json(bucket_path, bucket)
    if video_rf_json is not None:
        # Save JSON
        logger.info("✓ Generated rf_video_analysis.json")
    else:
        logger.info("⏭ Skipped rf_video_analysis.json (RF not trained)")

    # Generate Window-Level RF JSONs (optional)
    for window in windows:
        window_rf_json = generate_window_rf_json(bucket_path, bucket, window)
        if window_rf_json is not None:
            # Save JSON
            logger.info(f"✓ Generated {window}_rf_analysis.json")
        else:
            logger.info(f"⏭ Skipped {window}_rf_analysis.json (RF not trained)")

    # K-Means always generated
    for window in windows:
        window_km_json = generate_window_kmeans_json(bucket_path, bucket, window)
        # Save JSON
        logger.info(f"✓ Generated {window}_kmeans_analysis.json")
```

---

### Fix 4: Orchestrator Stage 6 Validation with Cross-Check
**File:** `rumiai_ml_batch.py:177-222`

```python
def validate_stage_6_prerequisites(bucket_path: str) -> None:
    # Check model_metrics.json first
    metrics_path = bucket_path_obj / "models" / "model_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    rf_trained = metrics.get('video_level_rf', {}).get('trained', True)

    # Cross-validate: Check if video-level RF file exists
    rf_video_path = bucket_path_obj / "models" / f"rf_video_{bucket_name}.pkl"
    rf_video_exists = rf_video_path.exists()

    # Case 1: Metrics says trained but file missing → Stage 5 incomplete
    if rf_trained and not rf_video_exists:
        raise ValueError(
            f"Stage 5 incomplete: model_metrics.json says RF trained, "
            f"but rf_video_{bucket_name}.pkl missing. Re-run Stage 5."
        )

    # Case 2: Metrics says not trained but file exists → Stale from previous run
    elif not rf_trained and rf_video_exists:
        logger.warning(
            f"⚠️ Stale RF model files detected from previous run. "
            f"model_metrics.json says trained=False (TOP mode), ignoring stale files."
        )
        # Trust metrics - don't change rf_trained

    if rf_trained:
        # RF was trained - require RF model files
        required_files.append(bucket_path_obj / "models" / f"rf_video_{bucket_name}.pkl")
        for window in windows:
            required_files.append(bucket_path_obj / "models" / f"rf_{window}_{bucket_name}.pkl")
        logger.info("Stage 6 validation: RF models required")
    else:
        logger.info("Stage 6 validation: RF models NOT required (TOP mode)")

    # K-Means models (always required)
    for window in windows:
        required_files.append(bucket_path_obj / "models" / f"{window}_kmeans_{bucket_name}.pkl")
```

---

### Fix 5: Orchestrator Stage 7 Validation
**File:** `rumiai_ml_batch.py:288-330`

```python
def validate_stage7_prerequisites(bucket_path: str, bucket: str) -> None:
    # Video-level RF (optional - check if exists)
    rf_video_path = os.path.join(ml_analysis_dir, "rf_video_analysis.json")
    if os.path.exists(rf_video_path):
        required_files.append(rf_video_path)
        logger.info("Stage 7: RF video analysis found (CONTRASTIVE mode)")
    else:
        logger.info("Stage 7: RF video analysis NOT found (TOP mode - K-Means only)")

    # Window-level files (RF optional, K-Means required)
    for window in window_types:
        # RF analysis (optional)
        rf_path = os.path.join(ml_analysis_dir, f"{window}_rf_analysis.json")
        if os.path.exists(rf_path):
            required_files.append(rf_path)

        # K-Means analysis (required)
        km_path = os.path.join(ml_analysis_dir, f"{window}_kmeans_analysis.json")
        required_files.append(km_path)
```

---

### Fix 6: Stage 7 Graceful Degradation
**File:** `ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py:372-589`

```python
def run_phase2_synthesis(bucket_path: str, window_analyses: Dict[str, dict],
                        bucket: str, hashtag: Optional[str]) -> dict:
    # Load RF video-level data (OPTIONAL)
    rf_video_path = os.path.join(bucket_path, 'ml_analysis/rf_video_analysis.json')

    if os.path.exists(rf_video_path):
        with open(rf_video_path, 'r') as f:
            rf_video_data = json.load(f)
        logger.info("✓ Loaded RF video data")
    else:
        rf_video_data = None
        logger.info("⚠ RF video data not found (TOP mode) - K-Means only")

    # Build prompt with optional RF data
    prompt = build_phase2_prompt(
        window_analyses=window_analyses,
        rf_video_data=rf_video_data,  # Can be None
        bucket=bucket
    )
```

**File:** `ml_pipeline/stage7_llm_analysis/stage7_prompts.py`

```python
def build_phase2_prompt(
    window_analyses: Dict[str, dict],
    rf_video_data: Optional[dict],  # NEW: Can be None in TOP mode
    bucket: str
) -> str:
    if rf_video_data is not None:
        # CONTRASTIVE mode - include RF feature analysis
        prompt += "## Cross-Window Feature Importance (Random Forest)\n"
        for feature in rf_video_data.get('feature_importance', []):
            prompt += f"- {feature['feature']}: {feature['importance']:.3f}\n"
            prompt += f"  Top: {feature['top_performer_avg']:.2f}, "
            prompt += f"Bottom: {feature['bottom_performer_avg']:.2f}\n"
    else:
        # TOP mode - skip RF analysis
        prompt += "## Analysis Mode: TOP PERFORMERS ONLY\n"
        prompt += "Note: Only top performers analyzed (no comparison group).\n"
        prompt += "Focus on common patterns using K-Means clustering.\n"
```

---

## Testing Checklist

- [ ] Run Stage 5 in TOP mode → verify RF skipped, K-Means trained
- [ ] Check `model_metrics.json` has `trained: false` for RF
- [ ] Run Stage 6 in TOP mode → verify no crash, only K-Means JSONs created
- [ ] Verify `rf_video_analysis.json` NOT created in TOP mode
- [ ] Run Stage 7 in TOP mode → verify graceful K-Means-only analysis
- [ ] Run full pipeline CONTRASTIVE mode → verify no regressions
- [ ] Validate checkpoint/resume works in both modes

---

## File Summary

**Files to modify (7 files):**
1. `ml_pipeline/stage6_analysis/ml_analysis_generation.py` (4 functions)
2. `rumiai_ml_batch.py` (2 validation functions)
3. `ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py` (1 function)
4. `ml_pipeline/stage7_llm_analysis/stage7_prompts.py` (1 function)

**Key locations:**
- Stage 6 validation: Lines 71-163
- Stage 6 RF JSON gen: Lines 166-287, 290-386, 641-737
- Orchestrator Stage 6: Lines 177-222
- Orchestrator Stage 7: Lines 288-330
- Stage 7 synthesis: Lines 372-589
- Stage 7 prompts: build_phase2_prompt function

**Pattern:** Check `model_metrics.json → trained: false` → skip RF logic
