# Clarification Q&A: Stage 7 - LLM Analysis

> **Mother Doc**: MLPlanningv2.md Section "Stage 7: LLM Analysis - Hybrid Two-Phase Approach" (lines 2587-3299)
> **Phase 1**: Critique_Stage7_LLMAnalysis.md
> **Date**: 2025-10-16
> **Status**: IN PROGRESS

## Questions by Category

### Input/Output Contracts

#### Q1: [CRITICAL] Output File Directory Structure - ml_analysis/llm/ Subdirectory

**Context**: Stage 7 section (lines 2754, 2828-2833, 3096, 3240) shows output files saved to `ml_analysis/llm/` subdirectory:
- Phase 1: `save_json(f'ml_analysis/llm/{window_type}_analysis.json', analysis)` (line 2754)
- Phase 2: `save_json('ml_analysis/llm/winning_formulas.json', synthesis)` (line 3096)
- Complete: `save_json(f'ml_analysis/llm/complete_analysis_{bucket}.json', complete_analysis)` (line 3240)

However, **MLAnalysisGenerationCHILD.md** (Stage 6 - our source of truth for paths) shows Stage 6 outputs are saved directly to `ml_analysis/`:
- `{bucket_path}/ml_analysis/rf_video_analysis.json` (line 114)
- No `llm/` subdirectory mentioned in Stage 6

**Questions**:
1. Does Stage 7 create a **new subdirectory** `ml_analysis/llm/` for its outputs?
   - If yes, when is this directory created? (Pre-flight validation? First save?)

2. What is the **complete absolute path** for Stage 7 outputs? Is it:
   - `/data/clients/{client_id}/buckets/bucket_{bucket}/ml_analysis/llm/{window}_analysis.json`?

3. Should Stage 7 use the **same bucket_path pattern** as Stage 6?
   - Stage 6 receives: `bucket_path` as absolute path (e.g., `/data/clients/acme/buckets/bucket_18-33s`)
   - Stage 7 should receive: Same `bucket_path` parameter from orchestrator?

4. For the **relative paths in code** (lines 2754, 3096, 3240), are these relative to `bucket_path`?
   - Example: `os.path.join(bucket_path, 'ml_analysis/llm/hook_analysis.json')`?

**For HLD Section**: 3.2 (Output Contracts), 5.2 (Output Schema - file paths), 8.1 (File Structure)

**Answer**:

**Directory Structure Approved**: Stage 7 creates **new subdirectory** `ml_analysis/llm/` for separation of concerns.

**Complete File Paths**:
```
/data/clients/{client_id}/buckets/bucket_{bucket}/
├── ml_analysis/
│   ├── rf_video_analysis.json             # Stage 6 outputs (13 files)
│   ├── hook_rf_analysis.json
│   ├── hook_kmeans_analysis.json
│   ├── ... (10 more Stage 6 files)
│   └── llm/                                # Stage 7 outputs (NEW subdirectory)
│       ├── hook_analysis.json              # Phase 1 (6-7 files)
│       ├── middle_1_analysis.json
│       ├── middle_2_analysis.json
│       ├── middle_3_analysis.json
│       ├── middle_4_analysis.json
│       ├── closing_analysis.json
│       ├── winning_formulas.json           # Phase 2
│       └── complete_analysis_{bucket}.json # Combined Phase 1 + Phase 2
```

**Implementation Details**:
1. **Directory Creation**: Stage 7 creates `ml_analysis/llm/` during pre-flight setup (before Phase 1 execution)
   ```python
   llm_output_dir = os.path.join(bucket_path, 'ml_analysis/llm')
   os.makedirs(llm_output_dir, exist_ok=True)
   ```

2. **Absolute Path Pattern**: Same as Stage 6
   - Stage 7 receives: `bucket_path` parameter (absolute path, e.g., `/data/clients/acme/buckets/bucket_18-33s`)
   - Constructs output paths: `os.path.join(bucket_path, 'ml_analysis/llm/hook_analysis.json')`

3. **Input File Paths**: Stage 6 outputs loaded from `bucket_path/ml_analysis/`
   - `os.path.join(bucket_path, 'ml_analysis/rf_video_analysis.json')`
   - `os.path.join(bucket_path, f'ml_analysis/{window}_rf_analysis.json')`
   - `os.path.join(bucket_path, f'ml_analysis/{window}_kmeans_analysis.json')`

**Rationale**:
- **Separation of concerns**: ML model insights (Stage 6) vs LLM creative insights (Stage 7) in separate directories
- **Debugging convenience**: Can delete `llm/` directory and re-run Stage 7 without affecting Stage 6 outputs
- **Future scalability**: Enables multiple LLM output types (e.g., `llm/creative_reports/`, `llm/technical_analysis/`)
- **Consistency**: Follows Stage 6's pattern of using subdirectories (e.g., `ml_analysis/.tmp/` for temp files)

**Notes**: This creates 8 output files per bucket (6-7 Phase 1 + 1 Phase 2 + 1 complete analysis), totaling ~40-50KB

### Dependencies & Integration

#### Q2: [CRITICAL] Critique Approved Decisions - Mandatory Integration into HLD

**Context**: Critique_Stage7_LLMAnalysis.md (Phase 1) has approved several critical design decisions that MUST be integrated into the HLD. These are not optional - they are **source of truth** requirements.

**Approved Decisions from Critique (Q3, Q4, Q5)**:

**1. Smart Retry Logic** (Critique Q4, lines 251-281):
- Retry ONLY failed windows (not all 6-7)
- Maximum 2 retry attempts per window
- 100% window completion required
- Abort bucket if still incomplete after retries
- Example: 6 initial + 2 retry + 1 final = 9 API calls (vs 18 if retrying all windows)

**2. Automated Validation Layer** (Critique Q3, lines 205-210 - Layer 1):
- Post-LLM validation script checks AFTER each LLM call:
  - Feature value contradictions (LLM says "high energy 0.85" but data shows 0.22)
  - Invented features (LLM references features not in source JSON)
  - RF validation contradictions (priority recommendations ignore top RF features)
- On failure: Retry LLM call with modified prompt OR flag for human review

**3. Path Frequency Filtering** (Critique Q5, lines 301-407):
- **10% threshold** for path formula inclusion (minimum 10 videos out of 100)
- Confidence levels: very_high (≥20%), high (15-20%), moderate (10-15%)
- **Fallback strategy**: If <3 paths meet 10%, use feature-based reports
- Always deliver 3 reports per bucket (path-based preferred, feature-based fallback)
- **Hybrid output structure**:
  ```json
  {
    "creative_reports": [
      {
        "report_id": 1-3,
        "type": "path_based" | "feature_based",
        "frequency": int,
        "percentage": float,
        "confidence_level": "very_high" | "high" | "moderate",
        // ... existing fields
      }
    ],
    "supplementary_insights": {
      "universal_principles": [...],  // Top 5-7 RF features from video-level RF
      "cross_window_patterns": [...]  // Cross-window features
    }
  }
  ```

**For HLD Sections**:
- Section 2.3 (Detailed Process): Smart retry logic, automated validation post-LLM
- Section 5.2 (Output Schema): Hybrid output structure with confidence_level, supplementary_insights
- Section 6.2 (Error Cases): Retry logic (max 2 attempts), abort conditions
- Section 6.3 (Output Validation): Automated validation checks (feature contradictions, invented features, RF misalignment)

**Notes**:
- These are mandatory requirements, not design options
- Mother Doc (MLPlanningv2.md) will be updated manually after HLD creation
- Critique decisions override any conflicting Mother Doc content

[Questions will be filled iteratively]

### Edge Cases & Validation

[Questions will be filled iteratively]

### Performance & Scale

[Questions will be filled iteratively]

### Error Handling

[Questions will be filled iteratively]

### Testing

[Questions will be filled iteratively]

## Completeness Check

[Will be filled at end - see Step 6]

## Proceed to Phase 3

[Will be filled at end - see Step 6]
