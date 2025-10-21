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

#### Q3: [CRITICAL] Window Configuration for Different Buckets - Bucket-Aware Processing

**Context**: Different duration buckets have different window structures. Stage 6 (MLAnalysisGenerationCHILD.md lines 754-765) uses centralized configuration from `config.bucket_definitions.BUCKET_WINDOWS`.

**Questions & Approved Answers**:

**Q3.1: Should Stage 7 use the same centralized config as Stage 6?**

**Answer**: ✅ **YES** - Use `config.bucket_definitions.BUCKET_WINDOWS`

**Rationale**:
- Consistency across stages (Stage 4, 5, 6 all use it)
- Single source of truth prevents divergence
- If bucket structure changes, only one place to update

**Implementation**:
```python
from config.bucket_definitions import BUCKET_WINDOWS

def run_stage7_llm_analysis(bucket_path: str, bucket: str) -> dict:
    windows = BUCKET_WINDOWS[bucket]  # e.g., ['hook', 'middle_1', ..., 'closing']
    # Use windows for Phase 1 iteration
```

---

**Q3.2: For buckets 9-13s, 13-18s (middle_aggregate), does Stage 7 generate only 3 Phase 1 JSONs?**

**Answer**: ✅ **YES** - Only 3 Phase 1 JSONs, 3 LLM calls, 3-position cluster paths

**Rationale**:
- `middle_aggregate` is treated identically to other windows (Mother Doc line 2525)
- Phase 1 analyzes each window independently → 3 windows = 3 calls
- Phase 2 cluster paths: `[0, 1, 2]` where position 1 = middle_aggregate cluster

**Implementation**:
```python
# Bucket 9-13s
windows = ['hook', 'middle_aggregate', 'closing']  # 3 windows from config
# Phase 1: 3 parallel LLM calls
# Phase 2 cluster paths: [0, 1, 2], [1, 1, 0], etc. (3^3 = 27 possible paths)
# Example path: "Hook-Cluster0 → Middle_Aggregate-Cluster1 → Closing-Cluster2"
```

---

**Q3.3: For bucket 0-3s (only hook, no closing), does Stage 7 skip Phase 2?**

**Answer**: ✅ **SKIP Phase 2** - Generate modified Phase 1 report only

**Rationale**:
- **Phase 1**: Single LLM call for hook analysis (3 clusters, defining features, recommendations)
- **Phase 2 doesn't make sense**: "Winning Formulas" require temporal progression (hook → middle → closing)
  - With only 1 window, there's no "journey" or "arc"
  - Cluster paths would be single values: `[0]`, `[1]`, `[2]` (not meaningful patterns)
  - No cross-window features (hook_to_middle_energy_delta doesn't exist)

**Alternative Output**: Generate `hook_analysis.json` + simplified `bucket_summary_0-3s.json`:
```json
{
  "bucket": "0-3s",
  "total_videos": 100,
  "note": "Single-window bucket - no temporal progression analysis available",
  "hook_strategies": [
    {
      "cluster_id": 0,
      "name": "The Direct Eye Contact Hook",
      "size": 35,
      "percentage": 35.0,
      "strategy_description": "...",
      "creator_recommendations": [...]
    },
    // ... clusters 1 and 2
  ],
  "recommendation": "Videos in this bucket are extremely short (0-3s). Focus on immediate impact. Choose one of the 3 hook strategies based on content type."
}
```

**Implementation**:
```python
if len(windows) == 1:
    # Bucket 0-3s: Skip Phase 2, generate simplified summary
    return generate_single_window_summary(window_analyses['hook'], bucket)
elif len(windows) >= 2:
    # All other buckets: Run Phase 2 normally
    return run_phase2_synthesis(window_analyses, kmeans_outputs, rf_video_data, bucket, hashtag)
```

---

**Q3.4: What is the minimum number of windows required for Phase 2 synthesis?**

**Answer**: ✅ **Minimum 2 windows** (hook + closing)

**Rationale**:
- **2 windows (bucket 3-9s)**: Minimal temporal progression possible
  - Cluster paths: `[0, 0]`, `[0, 1]`, `[1, 0]`, etc. (3^2 = 9 possible paths)
  - Can analyze: "opening strategy → closing strategy"
  - Video-level RF can compute: `hook_to_closing_energy_delta`

- **Why not 3 minimum?**
  - Bucket 3-9s exists and should receive insights (don't skip)
  - Even hook → closing progression is valuable ("how do they close after that hook?")

---

**Summary Table - Bucket-Aware Processing**:

| Bucket | Windows | Phase 1 Calls | Phase 2? | Cluster Path Length | Total Possible Paths |
|--------|---------|---------------|----------|---------------------|----------------------|
| 0-3s | 1 (hook) | 1 | ❌ NO (simplified summary) | N/A | N/A |
| 3-9s | 2 (hook, closing) | 2 | ✅ YES | 2 | 9 (3^2) |
| 9-13s | 3 (hook, middle_agg, closing) | 3 | ✅ YES | 3 | 27 (3^3) |
| 13-18s | 3 (hook, middle_agg, closing) | 3 | ✅ YES | 3 | 27 (3^3) |
| 18-33s | 6 | 6 | ✅ YES | 6 | 729 (3^6) |
| 33-60s | 7 | 7 | ✅ YES | 7 | 2187 (3^7) |
| 60-90s | 7 | 7 | ✅ YES | 7 | 2187 (3^7) |
| 90-120s | 7 | 7 | ✅ YES | 7 | 2187 (3^7) |

**For HLD Section**: 2.3 (Detailed Process - bucket-aware iteration), 6.1 (Input Validation - window count checks), 6.2 (Error Cases - bucket 0-3s special handling), 8.1 (Testing - edge case buckets)

**Notes**:
- This creates a pragmatic approach that handles all bucket types correctly
- Doesn't force meaningless analysis (0-3s Phase 2)
- Maintains consistency with centralized config
- Provides value even for edge case buckets (3-9s gets insights)

[Questions will be filled iteratively]

### Performance & Scale

#### Q4: [CRITICAL] Anthropic API Configuration & Credentials

**Context**: Mother Doc Stage 7 (lines 2700-2706) shows Anthropic API usage for LLM calls.

**Questions & Approved Answers**:

**Q4.1: Where is ANTHROPIC_API_KEY stored?**

**Answer**: ✅ **Environment variable in `.env` file**

**Implementation**:
```bash
# .env file (in project root)
ANTHROPIC_API_KEY=sk-ant-api03-...
```

```python
# stage7_llm_analysis.py
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
```

---

**Q4.2: Should Stage 7 validate API key exists before Phase 1?**

**Answer**: ✅ **YES** - Pre-flight validation required

**Rationale**:
- Fail-fast principle: Better to fail immediately than after processing 5/6 windows
- Clear error message: "Missing ANTHROPIC_API_KEY" vs cryptic API auth error
- Consistent with Stage 6 pre-flight validation pattern

**Implementation**:
```python
def validate_api_credentials():
    """
    Validate Anthropic API key exists before Phase 1.

    Raises:
        PreFlightValidationError: If API key missing or invalid
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise PreFlightValidationError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Add to .env file: ANTHROPIC_API_KEY=sk-ant-api03-..."
        )

    if not api_key.startswith("sk-ant-"):
        raise PreFlightValidationError(
            f"Invalid ANTHROPIC_API_KEY format: {api_key[:10]}... "
            "Expected format: sk-ant-api03-..."
        )

    logger.info("✓ API credentials validated")
```

**Pre-flight validation order**:
1. Validate API key exists and has correct format
2. Validate Stage 6 outputs exist (13 JSON files)
3. Create output directory (`ml_analysis/llm/`)
4. Run Phase 1

**Optional - Connection Test**: Test API with minimal request?
- **Not recommended**: Adds latency (~1-2s) and costs ($0.001)
- **Better**: Fail at first actual Phase 1 call with clear retry logic

---

**Q4.3: Is model version hardcoded or configurable?**

**Answer**: ✅ **Configurable constant in `config/llm_config.py`**

**Rationale**:
- Model versions change (Claude releases new models)
- Testing flexibility (can test with cheaper models like `claude-haiku-4`)
- Cost optimization options
- Single update point (not scattered code)

**Implementation**:
```python
# config/llm_config.py
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # Production model
ANTHROPIC_MODEL_TEST = "claude-haiku-4"       # Optional: cheaper for testing

# stage7_llm_analysis.py
from config.llm_config import ANTHROPIC_MODEL

response = client.messages.create(
    model=ANTHROPIC_MODEL,  # From config, not hardcoded
    max_tokens=...,
    temperature=...,
    messages=[...]
)
```

**Benefits**:
- Update model version in one place
- Easy A/B testing of model versions
- Can switch to cheaper models for development/testing

---

**Q4.4: Are max_tokens, temperature values final or configurable?**

**Answer**: ✅ **Configurable constants with sensible defaults (not CLI parameters)**

**Rationale**:
- Prompt engineering iteration: May need tuning based on output quality
- Cost control: Lower max_tokens = lower cost
- Different phases have different needs (Phase 1 vs Phase 2)
- **Not CLI parameters**: These are technical LLM tuning, not user-facing

**Implementation**:
```python
# config/llm_config.py
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Phase 1: Per-window analysis
PHASE1_MAX_TOKENS = 4000
PHASE1_TEMPERATURE = 0.3  # Lower = more consistent/focused

# Phase 2: Cross-window synthesis
PHASE2_MAX_TOKENS = 8000
PHASE2_TEMPERATURE = 0.4  # Slightly higher = more creative connections

# Validation layer (automated checks)
VALIDATION_MAX_TOKENS = 1000  # Short responses for yes/no validation
VALIDATION_TEMPERATURE = 0.1  # Very low = deterministic
```

**Usage in code**:
```python
from config.llm_config import (
    ANTHROPIC_MODEL,
    PHASE1_MAX_TOKENS,
    PHASE1_TEMPERATURE,
    PHASE2_MAX_TOKENS,
    PHASE2_TEMPERATURE
)

# Phase 1
response = client.messages.create(
    model=ANTHROPIC_MODEL,
    max_tokens=PHASE1_MAX_TOKENS,
    temperature=PHASE1_TEMPERATURE,
    messages=[{"role": "user", "content": prompt}]
)

# Phase 2
response = client.messages.create(
    model=ANTHROPIC_MODEL,
    max_tokens=PHASE2_MAX_TOKENS,
    temperature=PHASE2_TEMPERATURE,
    messages=[{"role": "user", "content": prompt}]
)
```

**Why not CLI parameters?**
- Users shouldn't need to know what temperature/max_tokens mean
- These are internal LLM tuning values, not business parameters
- Changing them requires understanding LLM behavior (technical expertise)

**When to adjust these values?** (During pilot testing)
- Temperature too low (0.1): Outputs are repetitive/boring
- Temperature too high (0.7): Outputs are inconsistent/unreliable
- Max_tokens too low: Outputs truncated mid-sentence
- Max_tokens too high: Wasting money on unused tokens

**For HLD Section**: 3.4 (External Dependencies), 4.2 (Internal Configuration - llm_config.py), 6.1 (Input Validation - API key pre-flight check), 6.2 (Error Cases - API failures)

[Questions will be filled iteratively]

### Error Handling

#### Q5: [HIGH] API Failure Handling - Specific Error Types and Retry Logic

**Context**: Critique Q4 approved smart retry logic (retry only failed windows, max 2 attempts). Need to clarify specific retry implementation details.

**Questions & Approved Answers**:

**Q5.1: Which API errors should trigger retry vs immediate abort?**

**Answer**:

### **Retry (Recoverable Errors)**:
- **429 Rate Limiting**: Anthropic throttling → Wait and retry
- **503 Service Unavailable**: Temporary Anthropic outage → Retry
- **502 Bad Gateway**: Network/proxy issue → Retry
- **Connection Timeout**: Network dropped → Retry
- **500 Internal Server Error**: Temporary Anthropic issue → Retry (1 retry only)

### **Abort (Fatal Errors - Don't Retry)**:
- **401 Unauthorized**: Invalid API key → Pre-flight should catch this, abort immediately
- **400 Bad Request**: Invalid prompt format → Our code bug, retry won't fix
- **403 Forbidden**: Account issue → Retry won't fix
- **422 Unprocessable Entity**: Prompt too long or violates policy → Retry won't fix
- **Quota Exceeded**: Account limit reached → Retry won't fix

**Implementation**:
```python
# config/llm_config.py
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}
FATAL_STATUS_CODES = {400, 401, 403, 422}

def should_retry_api_error(error) -> bool:
    """Determine if API error is retryable."""
    if hasattr(error, 'status_code'):
        if error.status_code in FATAL_STATUS_CODES:
            return False
        if error.status_code in RETRYABLE_STATUS_CODES:
            return True

    # Network errors (connection timeout, DNS failure)
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    return False  # Unknown errors → don't retry (fail-fast)
```

---

**Q5.2: Retry backoff strategy?**

**Answer**: ✅ **Exponential backoff with jitter**

**Rationale**:
- **Rate limiting (429)**: Exponential backoff prevents hammering API during throttling
- **Temporary outages (503)**: Gives service time to recover
- **Jitter**: Prevents thundering herd (multiple concurrent retries hitting at same time)

**Implementation**:
```python
import time
import random

def retry_with_backoff(attempt: int, max_wait: int = 30) -> None:
    """
    Wait with exponential backoff before retry.

    Args:
        attempt: Retry attempt number (1, 2, ...)
        max_wait: Maximum wait time in seconds
    """
    # Exponential: 2s, 4s, 8s, 16s, 32s (capped at max_wait)
    base_wait = min(2 ** attempt, max_wait)

    # Add jitter: ±25% randomness
    jitter = random.uniform(0.75, 1.25)
    wait_time = base_wait * jitter

    logger.info(f"Retry attempt {attempt}: waiting {wait_time:.1f}s before retry")
    time.sleep(wait_time)

# Usage in Phase 1 parallel execution:
for attempt in range(1, 3):  # Max 2 retries
    try:
        response = client.messages.create(...)
        break  # Success
    except Exception as e:
        if should_retry_api_error(e) and attempt < 2:
            retry_with_backoff(attempt)
        else:
            raise  # Final failure or non-retryable
```

**Backoff Schedule**:
| Attempt | Base Wait | With Jitter (±25%) | Max Total Time |
|---------|-----------|-------------------|----------------|
| 1st call | 0s | 0s | - |
| Retry 1 | 2s | 1.5-2.5s | ~2s |
| Retry 2 | 4s | 3-5s | ~6s |

**Why not use Anthropic SDK's built-in retry?**
- We need **window-level granularity**: Only retry failed windows, not all 6
- Built-in retry would retry entire parallel batch, wasting successful calls
- We need custom logic for our multi-window architecture

---

**Q5.3: Timeout handling - How long to wait for single LLM call?**

**Answer**: ✅ **Conservative timeouts with 2x safety margin**

**Rationale for Conservative Timeouts**:
- **API variability**: Anthropic response times spike during peak hours (5s typical → 30s during high load)
- **Larger prompts**: Phase 1 includes 113-167 numbers, Phase 2 includes 6-7 window analyses
- **Cost of premature timeout**: Aborting bucket after 6 hours of video processing is expensive
- **Retry overhead**: Each timeout → wait → retry → potential 2nd timeout = wasted 2+ minutes

**Approved Timeouts**:
```python
# config/llm_config.py

# Phase 1: Per-window analysis (conservative)
PHASE1_TIMEOUT_SECONDS = 90   # 90 seconds
# Rationale:
# - Typical: 5-10s
# - 99th percentile: 30-45s (during API high load)
# - 90s = 2x safety margin prevents spurious timeouts

# Phase 2: Cross-window synthesis (very conservative)
PHASE2_TIMEOUT_SECONDS = 180  # 180 seconds (3 minutes)
# Rationale:
# - Larger context: 6 window analyses (each ~1-2KB) = 6-12KB input
# - Complex reasoning: Cross-window pattern detection
# - Typical: 15-30s
# - 99th percentile: 60-90s
# - 180s = 2x safety margin for worst-case

# Validation layer (automated checks)
VALIDATION_TIMEOUT_SECONDS = 30  # Short, focused validation task
```

**Timeout Comparison**:
| Phase | Timeout | Typical Response | 99th Percentile | Safety Margin |
|-------|---------|------------------|-----------------|---------------|
| Phase 1 | **90s** | 5-10s | 30-45s | 2x |
| Phase 2 | **180s** | 15-30s | 60-90s | 2x |
| Validation | **30s** | 2-5s | 10-15s | 2x |

**Timeout Handling**:
```python
from anthropic import Anthropic
from config.llm_config import PHASE1_TIMEOUT_SECONDS

def analyze_window_with_timeout(window_type, kmeans_data, rf_data, bucket, hashtag):
    """Phase 1 analysis with conservative timeout."""
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=PHASE1_MAX_TOKENS,
            temperature=PHASE1_TEMPERATURE,
            timeout=PHASE1_TIMEOUT_SECONDS,  # 90 seconds
            messages=[{"role": "user", "content": prompt}]
        )
        return parse_analysis(response)

    except TimeoutError as e:
        logger.warning(f"Timeout for {window_type} after {PHASE1_TIMEOUT_SECONDS}s")
        raise  # Will be caught by smart retry logic
```

**Timeout in Smart Retry Flow**:
```python
# Window fails with timeout → Retry logic kicks in
# Retry 1: Wait 2s (backoff), try again with 90s timeout
# Retry 2: Wait 4s (backoff), try again with 90s timeout
# Still timeout? → Mark window as failed, abort bucket
```

**Why Conservative Is Better**:
- **False positive prevention**: Avoids aborting due to temporary API slowness
- **Bucket protection**: After 6 hours of video processing, losing bucket to premature timeout is unacceptable
- **Negligible downside**: If actual failure (network down), waiting 90s vs 60s doesn't matter

---

**Q5.4: Partial/Invalid JSON response - Retry or abort?**

**Answer**: ✅ **Validate JSON, retry with increased max_tokens if truncated**

**Rationale**:
- **Root cause**: Response hits max_tokens limit mid-JSON (incomplete output)
- **Detection**: `json.JSONDecodeError` or `response.stop_reason == 'max_tokens'`
- **Fix**: Increase max_tokens by 50% and retry (automatic remediation)

**Implementation**:
```python
def parse_and_validate_json(response, window_type: str, kmeans_data, rf_data, attempt: int = 1):
    """
    Parse LLM response as JSON with automatic retry on truncation.

    Args:
        response: Anthropic API response
        window_type: Window being analyzed
        kmeans_data: K-Means data for retry
        rf_data: RF data for retry
        attempt: Current attempt number (1 or 2)

    Returns:
        dict: Parsed and validated JSON

    Raises:
        ValidationError: If JSON invalid after retries
    """
    try:
        # Extract JSON from response
        content = response.content[0].text
        analysis = json.loads(content)

        # Validate structure (has required fields)
        required_fields = ['window_type', 'clusters']
        missing = [f for f in required_fields if f not in analysis]

        if missing:
            raise ValidationError(f"Missing required fields: {missing}")

        return analysis

    except json.JSONDecodeError as e:
        # Check if truncation (finish_reason = 'max_tokens')
        if response.stop_reason == 'max_tokens':
            logger.warning(
                f"{window_type}: JSON truncated at max_tokens. "
                f"Increasing limit by 50% and retrying (attempt {attempt})"
            )

            if attempt < 2:
                # Retry with 50% more tokens
                increased_max_tokens = int(PHASE1_MAX_TOKENS * 1.5)  # 4000 → 6000
                logger.info(f"  Retrying with max_tokens={increased_max_tokens}")

                # Make new API call with increased limit
                response_retry = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=increased_max_tokens,
                    temperature=PHASE1_TEMPERATURE,
                    timeout=PHASE1_TIMEOUT_SECONDS,
                    messages=[{"role": "user", "content": build_phase1_prompt(window_type, kmeans_data, rf_data)}]
                )

                # Recursive call with attempt incremented
                return parse_and_validate_json(response_retry, window_type, kmeans_data, rf_data, attempt + 1)
            else:
                # Still truncated after retry → Fatal error
                raise ValidationError(
                    f"{window_type}: JSON still truncated after increasing max_tokens to {increased_max_tokens}. "
                    f"LLM output too verbose. Manual prompt refinement needed."
                )
        else:
            # Invalid JSON for other reason (malformed output)
            raise ValidationError(
                f"{window_type}: Invalid JSON response: {e}. "
                f"LLM generated malformed output. Response: {content[:200]}..."
            )
```

**Truncation Retry Strategy**:
| Attempt | max_tokens | Outcome |
|---------|------------|---------|
| 1st call | 4000 | Truncated (stop_reason='max_tokens') |
| Automatic retry | 6000 (50% increase) | Success ✅ (fits in 5500 tokens) |
| If still truncated | N/A | Abort - prompt needs refinement |

**Why Not Always Use Higher max_tokens?**
- **Cost**: Higher max_tokens = higher cost ($15/million output tokens)
- **Rare issue**: Most responses fit in 4000 tokens (typical: 2000-3000)
- **Adaptive**: Only increase when needed (pay-per-use efficiency)

**Other JSON Validation Errors**:
- **Missing required fields**: Treat as LLM error → Log and retry (counts as failed window attempt)
- **Wrong data types**: Treat as validation failure → Log and retry
- **Invented features**: Automated validation layer catches this (Critique Q3 Layer 1)

**For HLD Section**: 6.2 (Error Cases - detailed retry logic with backoff, timeout handling, truncation recovery), 6.3 (Output Validation - JSON parsing failures)

**Notes**:
- Conservative timeouts (90s/180s) protect against API variability and prevent losing buckets
- Exponential backoff with jitter prevents API hammering during rate limiting
- Automatic truncation recovery (increase max_tokens) handles edge cases gracefully
- Clear error classification (retry vs abort) ensures efficient failure handling

[Questions will be filled iteratively]

### Testing

#### Q6: [HIGH] Testing Strategy - Test Data and Validation Approach

**Context**: Testing Stage 7 is challenging due to dependencies on real API calls, Stage 6 ML outputs, and non-deterministic LLM responses.

**Questions & Approved Answers**:

**Q6.1: Test Data Strategy - Where does test data come from?**

**Answer**: ✅ **Two-phase testing: Synthetic fixtures (Phase 1) + Real Stage 2.6 data (Phase 2)**

**Phase 1 - Synthetic JSON Fixtures (Logic Validation)**:
- **Purpose**: Rigorous testing of edge cases and logic validation with controlled inputs
- **Scope**: All 8 buckets (0-3s through 90-120s)
- **Scale**: Realistic-scale fixtures (10-20 videos per bucket to test 10% threshold logic)
- **Benefits**:
  - Controlled inputs reveal logic flaws
  - Fast execution (no video processing pipeline)
  - Deterministic outputs (predictable test results)

**Phase 2 - Real Stage 2.6 Data (Integration Testing)**:
- **Purpose**: Integration testing with actual ML outputs from real video processing
- **Source**: Real videos processed through Stage 2.6 (with `temporal_windows_updated.json` outputs)
- **Pipeline**: Stage 2.6 outputs → Stage 3 → Stage 4 → Stage 5 → Stage 6 → **Stage 7**
- **Scope**: 1-2 priority buckets (e.g., 18-33s, 33-60s) with 10 videos each
- **Benefits**:
  - Validates end-to-end pipeline integration
  - Tests with real ML patterns (realistic centroids, feature distributions)
  - Catches data quality issues

**Test Data Structure**:
```
tests/fixtures/
├── synthetic/                           # Phase 1: Controlled edge case testing
│   ├── bucket_0-3s_10videos/           # Edge case: single window, no Phase 2
│   │   ├── rf_video_analysis.json
│   │   └── hook_kmeans_analysis.json
│   ├── bucket_3-9s_10videos/           # Edge case: 2 windows, 9 possible paths
│   ├── bucket_9-13s_10videos/          # Edge case: middle_aggregate
│   ├── bucket_18-33s_20videos/         # Standard case: 6 windows, 729 paths
│   └── ... (all 8 buckets)
├── real/                                # Phase 2: Integration testing
│   └── bucket_18-33s_stage6_outputs/   # Real Stage 6 outputs (13 JSONs)
│       ├── rf_video_analysis.json
│       ├── hook_rf_analysis.json
│       └── ... (13 files from real ML pipeline)
```

---

**Q6.2: LLM Call Mocking - Test without real API calls?**

**Answer**: ✅ **Hybrid approach: Mock for synthetic tests, Live API for real data tests**

**Rationale**:
- **Synthetic tests (Phase 1)**: Mock LLM responses (fast, deterministic, no API cost)
- **Real data tests (Phase 2)**: Live API calls (validates actual LLM behavior, ~$0.10 per test run)

**Implementation**: Environment variable flag
```python
# config/test_config.py
TEST_MODE = os.environ.get("TEST_MODE", "mock")  # "mock" or "live"

if TEST_MODE == "mock":
    # Use pre-recorded JSON responses
    client = MockAnthropicClient(fixtures_dir="tests/fixtures/llm_responses/")
else:
    # Real API calls
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
```

---

**Q6.3: Output Validation Testing - How to validate non-deterministic LLM outputs?**

**Answer**: ✅ **Controlled fictitious JSONs for schema validation + semantic checks**

**Approach**:
- **Schema validation**: Check required fields exist (deterministic)
- **Semantic validation**: Check mentions of top RF features (heuristic)
- **Controlled inputs**: Fictitious Stage 6 JSONs with obvious patterns

**Example - Controlled Test Case**:
```python
# tests/fixtures/synthetic/obvious_pattern_bucket_18-33s/
# Fictional RF data: eye_contact_rate has HUGE importance (0.95)
# Fictional K-Means: Cluster 0 has eye_contact_rate = 0.95, others = 0.10

# Expected LLM output should mention:
# - "eye_contact_rate" (top RF feature)
# - Cluster 0 defined by high eye contact
# - Recommendations include maintaining high eye contact

# Validation:
assert "eye_contact_rate" in llm_output['clusters'][0]['defining_features']
assert llm_output['clusters'][0]['rf_validation']['top_predictive_features_in_cluster']
```

---

**Q6.4: Edge Case Testing - Which edge cases are priority?**

**Answer**: ✅ **All listed edge cases are priority**

**Priority Edge Cases**:
1. **Bucket 0-3s**: Single window, no Phase 2, simplified summary output
2. **Bucket 3-9s**: 2 windows, minimal cluster paths (9 paths), tests minimum viable Phase 2
3. **Bucket 9-13s/13-18s**: `middle_aggregate` window handling
4. **API Failures**: 429 rate limiting, 503 service unavailable, timeout (90s)
5. **JSON Truncation**: max_tokens exceeded, automatic retry with increased limit
6. **All Windows Fail**: Smart retry exhausted, abort bucket logic
7. **<3 Paths Meet 10% Threshold**: Feature-based fallback report generation
8. **Low Frequency Paths**: All paths <10%, test supplementary_insights generation

**Test Coverage Matrix**: See TestingMethodology_Stage7.md Section 4

---

**For HLD Section**: 8 (Testing Strategy) - Brief overview with reference to TestingMethodology_Stage7.md

**Notes**:
- **Complete testing methodology documented in**: `TestingMethodology_Stage7.md`
- **HLD Section 8 will reference** testing methodology document (token-efficient)
- **Two-phase approach**: Synthetic (rigorous logic validation) → Real (integration validation)
- **All 8 buckets** tested in Phase 1 synthetic tests
- **Priority buckets** (18-33s, 33-60s) tested in Phase 2 with real data

---

**Next Steps**:
1. Create `TestingMethodology_Stage7.md` with:
   - Section 1: Synthetic JSON fixture schemas (8 buckets)
   - Section 2: Real data testing pipeline (Stage 2.6 → Stage 7)
   - Section 3: LLM mocking strategies (mock vs live)
   - Section 4: Edge case test matrix (8 priority scenarios)
   - Section 5: Test execution runbook (pytest commands)
2. HLD Section 8 references TestingMethodology_Stage7.md

#### Q7: [CRITICAL] Stage 6 Input Validation - Malformed or Incomplete JSONs

**Context**: Stage 7 depends on Stage 6 producing 13 valid JSON files per bucket. What if Stage 6 failed partially or produced invalid data?

**Questions**:

1. **Malformed JSON Detection**: What if Stage 6 JSON is syntactically invalid?
   - Example: `hook_rf_analysis.json` has trailing comma, fails `json.loads()`
   - Should Stage 7 pre-flight validation check all 13 JSONs are parseable?
   - Or fail at first load attempt?

2. **Schema Validation**: What if JSON is valid but missing required fields?
   - Example: `hook_kmeans_analysis.json` missing `clusters` field
   - Example: `rf_video_analysis.json` has 5 features instead of 10
   - Should Stage 7 validate schema before Phase 1?

3. **Incomplete Stage 6 Output**: What if only 10 of 13 JSONs exist?
   - Example: Stage 6 crashed after generating video RF + 5 window RFs (6 of 13 files)
   - Should Stage 7 check all expected files exist in pre-flight?
   - What's the error message? "Stage 6 incomplete: Missing 7 of 13 JSONs. Re-run Stage 6."

4. **Data Quality Issues**: What if JSON has invalid data?
   - Example: Feature importance sum ≠ 1.0 (ML bug)
   - Example: Cluster sizes don't sum to total_videos
   - Should Stage 7 validate data integrity or trust Stage 6?

**For HLD Section**: 6.1 (Input Validation - Stage 6 output checks), 6.2 (Error Cases - incomplete/malformed input handling)

**Approved Answers**:

**Q7.1: Malformed JSON Detection** → ✅ **Pre-flight validation - Check all 13 JSONs are parseable**

**Rationale**:
- Fail-fast principle: Detect all malformed JSONs upfront
- Clear error reporting: "3 of 13 JSONs malformed: [list]"
- Consistent with Stage 6 pattern (pre-flight validation of dependencies)

**Q7.2: Schema Validation** → ✅ **Yes - Basic schema validation in pre-flight**

**What to Validate**:
- Required fields present (`clusters`, `feature_importance`, `bucket`, etc.)
- Correct array lengths (10 features for RF, 3 clusters for K-Means)

**What NOT to Validate** (trust Stage 6):
- Feature importance sum = 1.0 (ML detail)
- Centroid value ranges (ML detail)
- Feature name consistency (Stage 6 already normalizes)

**Q7.3: Incomplete Stage 6 Output** → ✅ **Yes - Check all expected files exist in pre-flight**

**Error Message Example**:
```
Stage 6 incomplete: Missing 7 of 13 JSONs:
  - ml_analysis/middle_3_rf_analysis.json
  - ml_analysis/middle_4_rf_analysis.json
  ... and 5 more

Action: Re-run Stage 6 (ML Analysis Generation)
```

**Q7.4: Data Quality Issues** → ✅ **Lightweight validation only - Trust Stage 6 for ML correctness**

**What to Validate** (Critical for Stage 7 logic):
- ✅ Cluster sizes sum to total_videos (critical for path extraction in Phase 2)

**What NOT to Validate** (Trust Stage 6):
- ❌ Feature importance sum = 1.0
- ❌ Centroid value ranges
- ❌ RF accuracy/precision/recall values

**Three-Layer Pre-Flight Validation**:
```python
def validate_stage6_outputs(bucket_path: str, windows: list) -> None:
    """
    Three-layer pre-flight validation of Stage 6 outputs.

    Layer 1: File existence (all 13 JSONs present)
    Layer 2: JSON parseability (all JSONs syntactically valid)
    Layer 3: Schema + critical integrity (required fields + cluster sizes)
    """
    # Layer 1: Check files exist
    expected_files = [
        'ml_analysis/rf_video_analysis.json',
        *[f'ml_analysis/{w}_rf_analysis.json' for w in windows],
        *[f'ml_analysis/{w}_kmeans_analysis.json' for w in windows]
    ]

    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(bucket_path, f))]
    if missing_files:
        raise PreFlightValidationError(
            f"Stage 6 incomplete: Missing {len(missing_files)} of {len(expected_files)} JSONs. "
            f"Re-run Stage 6."
        )

    # Layer 2: Check JSONs parseable
    malformed_files = []
    for file_path in expected_files:
        try:
            with open(os.path.join(bucket_path, file_path), 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            malformed_files.append((file_path, str(e)))

    if malformed_files:
        raise PreFlightValidationError(
            f"Stage 6 validation failed: {len(malformed_files)} JSONs malformed. "
            f"Re-run Stage 6."
        )

    # Layer 3: Schema + cluster integrity
    for window in windows:
        # Validate K-Means JSON
        kmeans_data = load_json(os.path.join(bucket_path, f'ml_analysis/{window}_kmeans_analysis.json'))

        # Check required fields
        required = ['window_type', 'bucket', 'n_clusters', 'clusters']
        missing = [f for f in required if f not in kmeans_data]
        if missing:
            raise ValidationError(f"{window}_kmeans_analysis.json: Missing fields: {missing}")

        # Check 3 clusters
        if len(kmeans_data['clusters']) != 3:
            raise ValidationError(f"{window}_kmeans_analysis.json: Expected 3 clusters, got {len(kmeans_data['clusters'])}")

        # CRITICAL: Check cluster sizes sum to total_videos
        total_videos = kmeans_data['total_videos']
        cluster_sizes = [c['size'] for c in kmeans_data['clusters']]
        if sum(cluster_sizes) != total_videos:
            raise ValidationError(
                f"{window}_kmeans_analysis.json: Cluster sizes {cluster_sizes} sum to {sum(cluster_sizes)}, "
                f"but total_videos is {total_videos}. Stage 6 cluster assignment failed."
            )

        # Validate RF JSON (similar checks)
        rf_data = load_json(os.path.join(bucket_path, f'ml_analysis/{window}_rf_analysis.json'))
        if len(rf_data.get('feature_importance', [])) < 10:
            raise ValidationError(f"{window}_rf_analysis.json: Expected 10 features, got {len(rf_data['feature_importance'])}")

    logger.info("✓ Stage 6 output validation passed: All 13 JSONs valid")
```

**Validation Cost**: ~50-100ms for 13 files (negligible)

---

#### Q8: [CRITICAL] CLI Parameters and Orchestration - How Stage 7 Gets Invoked

**Context**: Stage 6 (MLAnalysisGenerationCHILD.md lines 748-749) receives `--bucket` and `--client` parameters. Does Stage 7 follow same pattern?

**Questions & Approved Answers**:

**Q8.1: CLI Interface - What parameters does Stage 7 receive?**

**Answer**: ✅ **3 parameters: `--stage 7`, `--client`, `--bucket` (identical to Stage 6 pattern)**

**CLI Command**:
```bash
# Stage 7 standalone invocation for testing/development
python run_ml_pipeline.py --stage 7 --client acme --bucket 18-33s

# Stage 6 for comparison (same pattern)
python run_ml_pipeline.py --stage 6 --client acme --bucket 18-33s
```

**Parameter Table**:
| Parameter | Type | Default | Valid Values | Required | Impact | Source |
|-----------|------|---------|--------------|----------|--------|--------|
| `--stage` | int | Required | 7 | Yes | Identifies Stage 7 execution | FoundationCHILD.md Section 4 |
| `--client` | str | Required | Any string | Yes | Determines client directory path | FoundationCHILD.md Section 4 |
| `--bucket` | str | Required | `0-3s`, `3-9s`, `9-13s`, `13-18s`, `18-33s`, `33-60s`, `60-90s`, `90-120s` | Yes | Determines which bucket directory to process | FoundationCHILD.md Section 4 |

**Implementation**:
```python
# run_ml_pipeline.py (existing CLI entry point)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, required=True, choices=[1,2,3,4,5,6,7])
    parser.add_argument('--client', type=str, required=True)
    parser.add_argument('--bucket', type=str, required=True,
                       choices=['0-3s','3-9s','9-13s','13-18s','18-33s','33-60s','60-90s','90-120s'])
    args = parser.parse_args()

    if args.stage == 7:
        from stages.stage7_llm_analysis import run_stage7_llm_analysis
        bucket_path = f'/data/clients/{args.client}/buckets/bucket_{args.bucket}'
        result = run_stage7_llm_analysis(bucket_path, args.bucket)
        return result
```

**Rationale**:
- **Consistency**: All ML stages (4, 5, 6, 7) use identical CLI pattern
- **Single entry point**: Developers use same script for all stages (no need to remember stage-specific scripts)
- **Foundation compliance**: Follows FoundationCHILD.md Section 4 CLI structure
- **Simplicity**: Only 3 parameters needed

---

**Q8.2: Hashtag Parameter - Is it a CLI parameter?**

**Answer**: ❌ **NO CLI parameter - Read from metadata file, default to None if missing**

**Implementation**:
```python
def run_stage7_llm_analysis(bucket_path: str, bucket: str) -> dict:
    """
    Run Stage 7 LLM Analysis.

    Args:
        bucket_path: Absolute path to bucket directory
        bucket: Bucket name (e.g., "18-33s")

    Returns:
        dict: Complete analysis with Phase 1 + Phase 2 outputs
    """
    # Try to read hashtag from metadata file (if exists)
    hashtag = get_hashtag_from_metadata(bucket_path)  # Returns None if missing (acceptable)

    # Run Phase 1 with optional hashtag
    window_analyses = run_phase1_parallel(
        bucket_path=bucket_path,
        bucket=bucket,
        hashtag=hashtag,  # None is acceptable
        window_types=BUCKET_WINDOWS[bucket]
    )

    # Run Phase 2
    synthesis = run_phase2_synthesis(
        window_analyses=window_analyses,
        kmeans_outputs=load_kmeans_outputs(bucket_path),
        rf_video_data=load_rf_video_data(bucket_path),
        bucket=bucket,
        hashtag=hashtag  # None is acceptable
    )

    return synthesis


def get_hashtag_from_metadata(bucket_path: str) -> str | None:
    """
    Read hashtag from Stage 2 metadata file if available.

    Args:
        bucket_path: Absolute path to bucket directory

    Returns:
        str: Hashtag (e.g., "nutrition", "fitness") - NO # prefix
        None: If metadata file missing or hashtag field not present
    """
    metadata_path = os.path.join(bucket_path, 'metadata.json')

    if not os.path.exists(metadata_path):
        logger.debug(f"No metadata.json found at {metadata_path} - hashtag will be None")
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            hashtag = metadata.get('hashtag', None)

            if hashtag:
                logger.info(f"Loaded hashtag from metadata: #{hashtag}")
            else:
                logger.debug("metadata.json exists but no 'hashtag' field - using None")

            return hashtag

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse metadata.json: {e} - hashtag will be None")
        return None
```

**Usage in LLM Prompts**:
```python
# Phase 1 prompt template
def build_phase1_prompt(window_type: str, kmeans_data: dict, rf_data: dict, bucket: str, hashtag: str | None) -> str:
    """Build Phase 1 LLM prompt with optional hashtag context."""

    # Hashtag-aware context
    if hashtag:
        context = f"You are analyzing {window_type} segments from 100 viral videos in the {bucket} duration bucket for #{hashtag}."
    else:
        context = f"You are analyzing {window_type} segments from 100 viral videos in the {bucket} duration bucket."

    prompt = f"""
{context}

Context:
- These are all TOP-PERFORMING videos (high engagement)
- You are identifying DIFFERENT STRATEGIES that all lead to success
...
"""
    return prompt
```

**Rationale**:
1. **Simplicity**: Keeps CLI minimal (3 parameters, not 4)
2. **Non-critical feature**: Hashtag only affects LLM prompt context, not Stage 7 logic
   - No filtering based on hashtag
   - No conditional behavior
   - Just makes recommendations slightly more specific ("use nutrition keywords" vs generic advice)
3. **Metadata is source of truth**: If Stage 2 saved hashtag metadata during video download, Stage 7 reads it
4. **Graceful degradation**: If metadata missing, Stage 7 runs with `hashtag=None` (LLM generates generic recommendations)
5. **Future integration friendly**: When integrated into `rumiai_ml_batch.py`, batch script can write metadata file before Stage 7

**Testing Scenarios**:
```bash
# Scenario 1: Testing with existing metadata
# /data/clients/acme/buckets/bucket_18-33s/metadata.json exists with {"hashtag": "nutrition"}
python run_ml_pipeline.py --stage 7 --client acme --bucket 18-33s
# → Stage 7 loads hashtag="nutrition" from metadata.json

# Scenario 2: Testing without metadata
# /data/clients/acme/buckets/bucket_18-33s/metadata.json does NOT exist
python run_ml_pipeline.py --stage 7 --client acme --bucket 18-33s
# → Stage 7 uses hashtag=None (generic LLM recommendations)

# Scenario 3: Metadata exists but no hashtag field
# metadata.json = {"client": "acme", "created_date": "2025-01-28"}
python run_ml_pipeline.py --stage 7 --client acme --bucket 18-33s
# → Stage 7 uses hashtag=None
```

---

**Q8.3: bucket_path Construction - Same pattern as Stage 6?**

**Answer**: ✅ **YES - Identical to Stage 6**

**Implementation**:
```python
# run_ml_pipeline.py
if args.stage == 7:
    # Construct absolute path (SAME pattern as Stage 6)
    bucket_path = f'/data/clients/{args.client}/buckets/bucket_{args.bucket}'

    # Pass to Stage 7
    result = run_stage7_llm_analysis(bucket_path, args.bucket)


# stage7_llm_analysis.py
def run_stage7_llm_analysis(bucket_path: str, bucket: str) -> dict:
    """
    Args:
        bucket_path: Absolute path (e.g., /data/clients/acme/buckets/bucket_18-33s)
        bucket: Bucket name (e.g., "18-33s")
    """
    # Load Stage 6 inputs using bucket_path
    rf_video_path = os.path.join(bucket_path, 'ml_analysis/rf_video_analysis.json')
    hook_rf_path = os.path.join(bucket_path, 'ml_analysis/hook_rf_analysis.json')
    hook_km_path = os.path.join(bucket_path, 'ml_analysis/hook_kmeans_analysis.json')

    # Create Stage 7 output directory
    llm_output_dir = os.path.join(bucket_path, 'ml_analysis/llm')
    os.makedirs(llm_output_dir, exist_ok=True)

    # ...
```

**Rationale**:
- **Consistency**: All stages use same `bucket_path` construction
- **Foundation standard**: Defined in FoundationCHILD.md Section 2 (Client Architecture)
- **Absolute paths**: Prevents relative path ambiguity

---

**Q8.4: Orchestration - Who calls Stage 7?**

**Answer**: ✅ **Two modes: Standalone (testing) + Integrated (production)**

**Mode 1: Standalone Invocation (Testing/Development)**
```bash
# Developer manually runs Stage 7 after Stage 6 completes
python run_ml_pipeline.py --stage 6 --client acme --bucket 18-33s
# (Stage 6 completes, outputs 13 JSONs)

# Developer manually runs Stage 7
python run_ml_pipeline.py --stage 7 --client acme --bucket 18-33s
# (Stage 7 runs, outputs 8 JSONs)
```

**Use Cases**:
- Testing Stage 7 with existing Stage 6 outputs
- Re-running Stage 7 after prompt changes (without re-running Stage 6)
- Debugging Stage 7 independently
- A/B testing different LLM configurations

**Mode 2: Integrated into `rumiai_ml_batch.py` (Production)**
```python
# rumiai_ml_batch.py (production pipeline orchestrator)
def run_full_ml_pipeline(client_id: str, bucket: str):
    """Run complete ML pipeline (Stages 1-7)."""

    # ... Stages 1-5 complete ...

    # Stage 6: ML Analysis Generation
    stage6_result = run_stage6_ml_analysis(client_id, bucket)
    if stage6_result['exit_code'] != 0:
        raise PipelineError("Stage 6 failed")

    # Stage 7: LLM Analysis (automatically triggered after Stage 6)
    bucket_path = f'/data/clients/{client_id}/buckets/bucket_{bucket}'
    stage7_result = run_stage7_llm_analysis(bucket_path, bucket)
    if stage7_result['exit_code'] != 0:
        raise PipelineError("Stage 7 failed")

    return stage7_result
```

**Use Cases**:
- Production batch processing (process 100s of videos through full pipeline)
- Automated nightly runs
- One-command full pipeline execution

**Rationale**:
- **Flexibility**: Supports both testing (standalone) and production (integrated)
- **Current pattern**: All stages can be run standalone via `run_ml_pipeline.py`
- **Future extensibility**: Integration into `rumiai_ml_batch.py` happens after Stage 7 development is complete

**For HLD Section**: 1.2 (Where This Fits in Pipeline), 4.1 (CLI Parameters)

**Notes**:
- **HLD focuses on standalone mode** (testing/development use case)
- **Integration into `rumiai_ml_batch.py`** documented separately (production deployment docs)
- **No automatic Stage 6 → Stage 7 chaining** in standalone mode (user manually invokes each stage)

---

#### Q9: [HIGH] Phase 2 Cluster Path Extraction - Implementation Details

**Context**: Mother Doc (lines 2852-2898) shows `extract_cluster_paths()` and `analyze_path_frequencies()` functions, but details are unclear.

**Questions & Approved Answers**:

**Q9.1: Video ID Tracking - Are video IDs consistent across windows?**

**Answer**: ✅ **YES - Video IDs are consistent across all windows for the same bucket**

**Stage 6 Implementation Confirmation** (MLAnalysisGenerationCHILD.md lines 514, 522-524):
```python
# Stage 6 generates K-Means JSONs with consistent video IDs
cluster_videos = df_km[cluster_labels == cluster_id]
video_ids = cluster_videos.index.tolist()  # Use DataFrame index as video ID

videos_list.append({
    'video_id': f'video_{video_idx}',  # Format: "video_0", "video_1", etc.
    'distance_to_centroid': float(distance)
})
```

**Why Video IDs Are Consistent**:
1. **Stage 6 uses DataFrame index**: `video_ids = cluster_videos.index.tolist()` (line 514)
2. **All windows process same CSV**: Each window loads `{window}_km_transformed.csv` from Stage 4
3. **Stage 4 guarantees same row order**: All window CSVs derived from same `aggregated_features.csv`
4. **Index is stable**: `video_0` in hook = `video_0` in middle_1 = same physical video

**Assumption**: `video_id` is consistent across all windows (`video_0` in hook_kmeans_analysis.json = `video_0` in middle_1_kmeans_analysis.json = same physical video)

**Stage 7 Implementation**:
```python
def extract_cluster_paths(window_types: list, kmeans_outputs: dict) -> list[dict]:
    """
    Extract cluster paths for all videos across windows.

    Args:
        window_types: List of windows (e.g., ['hook', 'middle_1', ..., 'closing'])
        kmeans_outputs: Dict of K-Means JSONs loaded from Stage 6
                       Format: {window_type: kmeans_analysis_json}
                       Example: {'hook': hook_json, 'middle_1': middle1_json, ...}

    Returns:
        List of dicts with cluster paths:
        [
            {'video_id': 'video_0', 'path': [0, 1, 1, 2, 0, 1]},
            {'video_id': 'video_1', 'path': [1, 0, 0, 1, 1, 0]},
            ...
        ]

    Example Output for bucket 18-33s (100 videos, 6 windows):
        - 100 video paths extracted
        - Each path has 6 positions (one per window)
        - Path [0, 1, 1, 2, 0, 1] means:
          - Hook: Cluster 0
          - Middle_1: Cluster 1
          - Middle_2: Cluster 1
          - Middle_3: Cluster 2
          - Middle_4: Cluster 0
          - Closing: Cluster 1
    """
    # Get all video IDs from first window (any window works - all have same videos)
    first_window = window_types[0]
    all_video_ids = []

    for cluster in kmeans_outputs[first_window]['clusters']:
        for video in cluster['videos']:
            all_video_ids.append(video['video_id'])  # e.g., "video_0", "video_1", ...

    # For each video, build its cluster path across all windows
    video_paths = []
    for video_id in all_video_ids:
        path = []

        for window in window_types:
            # Find which cluster this video belongs to in this window
            cluster_id = find_cluster_for_video(video_id, kmeans_outputs[window])
            path.append(cluster_id)

        video_paths.append({
            'video_id': video_id,
            'path': path  # [0, 1, 1, 2, 0, 1] - cluster ID per window
        })

    return video_paths


def find_cluster_for_video(video_id: str, kmeans_data: dict) -> int:
    """
    Find which cluster a video belongs to in a specific window.

    Args:
        video_id: Video identifier (e.g., "video_0")
        kmeans_data: K-Means analysis JSON for one window

    Returns:
        int: Cluster ID (0, 1, or 2)

    Raises:
        ValueError: If video not found in any cluster
    """
    for cluster in kmeans_data['clusters']:
        for video in cluster['videos']:
            if video['video_id'] == video_id:
                return cluster['cluster_id']

    # Video not found in any cluster - should never happen if Stage 6 worked correctly
    raise ValueError(
        f"Video {video_id} not found in K-Means data for window {kmeans_data['window_type']}. "
        f"Stage 6 cluster assignment may have failed."
    )
```

**Example Execution** (bucket 18-33s, 100 videos):
```python
# Load all K-Means JSONs from Stage 6
kmeans_outputs = {
    'hook': load_json('ml_analysis/hook_kmeans_analysis.json'),
    'middle_1': load_json('ml_analysis/middle_1_kmeans_analysis.json'),
    'middle_2': load_json('ml_analysis/middle_2_kmeans_analysis.json'),
    'middle_3': load_json('ml_analysis/middle_3_kmeans_analysis.json'),
    'middle_4': load_json('ml_analysis/middle_4_kmeans_analysis.json'),
    'closing': load_json('ml_analysis/closing_kmeans_analysis.json')
}

# Extract cluster paths for all 100 videos
video_paths = extract_cluster_paths(
    window_types=['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing'],
    kmeans_outputs=kmeans_outputs
)

# Result: 100 paths
# video_paths[0] = {'video_id': 'video_0', 'path': [0, 1, 1, 2, 0, 1]}
# video_paths[1] = {'video_id': 'video_1', 'path': [1, 0, 0, 1, 1, 0]}
# ...
```

**Edge Case - Video Missing from Window**:
```python
# Scenario: video_42 appears in hook but NOT in middle_1
# This SHOULD NOT happen if Stage 6 worked correctly (all windows process same videos)

try:
    cluster_id = find_cluster_for_video('video_42', kmeans_outputs['middle_1'])
except ValueError as e:
    # Error: "Video video_42 not found in K-Means data for window middle_1"
    logger.error(f"Data integrity issue: {e}")
    # Action: Abort Phase 2, log error, flag bucket for investigation
    raise DataIntegrityError("Stage 6 outputs inconsistent - video missing from window")
```

**Rationale**:
1. **Simplicity**: Direct lookup by video_id string (no complex matching logic needed)
2. **Stage 6 guarantee**: All windows in a bucket process the same set of videos (lines 492-514)
3. **Stage 4 guarantee**: All `{window}_km_transformed.csv` files derived from same `aggregated_features.csv` (same row order)
4. **Verified in code**: Stage 6 HLD explicitly documents this pattern

**For HLD Section**: 2.3 (Detailed Process - Phase 2 cluster path extraction), 5.2 (Output Schema - cluster path format)

---

**Q9.2: Cluster Path Construction - How to build path for each video?**

**Answer**: ✅ **See Q9.1 implementation (lines 1346-1398) - Complete logic provided**

**Summary**: For each video, iterate through all windows and look up which cluster the video belongs to in that window, appending cluster IDs to build the path.

**Implementation Reference** (from Q9.1):
```python
# Lines 1384-1396 from Q9.1
for video_id in all_video_ids:
    path = []

    for window in window_types:
        cluster_id = find_cluster_for_video(video_id, kmeans_outputs[window])
        path.append(cluster_id)

    video_paths.append({
        'video_id': video_id,
        'path': path  # [0, 1, 1, 2, 0, 1]
    })
```

**Path Interpretation Example** (bucket 18-33s, 6 windows):
```
Path: [0, 1, 1, 2, 0, 1]

Position 0 (hook):      Cluster 0
Position 1 (middle_1):  Cluster 1
Position 2 (middle_2):  Cluster 1
Position 3 (middle_3):  Cluster 2
Position 4 (middle_4):  Cluster 0
Position 5 (closing):   Cluster 1

Interpretation: "Video starts with Cluster 0 hook strategy, transitions to Cluster 1 for first two middle segments, switches to Cluster 2 for middle_3, returns to Cluster 0 for middle_4, and closes with Cluster 1 strategy."
```

**For HLD Section**: 2.3 (Detailed Process - Phase 2), 5.2 (Output Schema)

---

**Q9.3: Missing Videos - What if a video appears in hook but not in closing?**

**Answer**: ✅ **Should NOT happen - All windows must have same videos (Stage 6 guarantee)**

**Stage 6 Guarantee** (MLAnalysisGenerationCHILD.md lines 492-514):
- All windows load from same Stage 4 CSVs (`{window}_km_transformed.csv`)
- Stage 4 CSVs all derived from same `aggregated_features.csv` (same rows, same order)
- Stage 6 processes all videos for all windows in a bucket

**If Missing Videos Detected** (Data Integrity Error):
```python
# Q9.1 implementation (lines 1421-1424) already handles this
def find_cluster_for_video(video_id: str, kmeans_data: dict) -> int:
    # ... search logic ...

    # Video not found → Raise error
    raise ValueError(
        f"Video {video_id} not found in K-Means data for window {kmeans_data['window_type']}. "
        f"Stage 6 cluster assignment may have failed."
    )
```

**Error Handling in Phase 2**:
```python
try:
    video_paths = extract_cluster_paths(window_types, kmeans_outputs)
except ValueError as e:
    logger.error(f"Data integrity issue during cluster path extraction: {e}")
    # Abort Phase 2, flag bucket for investigation
    raise DataIntegrityError(
        f"Stage 6 outputs inconsistent: {e}. "
        f"Action: Verify Stage 6 completed successfully, check all {len(window_types)} K-Means JSONs have same video count."
    )
```

**Pre-Flight Validation Prevention** (Q7 - lines 1011-1018):
```python
# Stage 7 pre-flight already validates cluster size consistency
total_videos = kmeans_data['total_videos']
cluster_sizes = [c['size'] for c in kmeans_data['clusters']]
if sum(cluster_sizes) != total_videos:
    raise ValidationError(
        f"Cluster sizes {cluster_sizes} sum to {sum(cluster_sizes)}, "
        f"but total_videos is {total_videos}. Stage 6 cluster assignment failed."
    )
```

**Rationale**:
1. **Stage 6 architectural guarantee**: All windows process same videos (verified in Stage 6 HLD)
2. **Pre-flight catches inconsistencies**: Q7 validation detects cluster size mismatches before Phase 1
3. **Defensive error handling**: Q9.1 `find_cluster_for_video()` raises clear error if video missing
4. **Should be rare**: Only occurs if Stage 6 has a bug or partial file corruption

**Action if Error Occurs**:
1. Log detailed error (which window, which video_id, total videos expected vs actual)
2. Abort Phase 2 (do NOT generate partial winning_formulas.json)
3. Flag bucket for investigation
4. User action: Re-run Stage 6 to regenerate K-Means JSONs

**For HLD Section**: 6.2 (Error Cases - data integrity errors), 6.1 (Input Validation - pre-flight cluster size checks)

---

**Q9.4: Path Frequency Calculation - How to apply 10% threshold?**

**Answer**: ✅ **Count identical paths, calculate percentages, filter ≥10%, sort by frequency, take top 3 (or fallback to feature-based)**

**Complete Implementation**:
```python
def analyze_path_frequencies(video_paths: list[dict], total_videos: int) -> dict:
    """
    Calculate path frequencies, apply 10% threshold, assign confidence levels.

    Args:
        video_paths: List of dicts from extract_cluster_paths()
                    [{'video_id': 'video_0', 'path': [0, 1, 1, 2, 0, 1]}, ...]
        total_videos: Total video count (e.g., 100)

    Returns:
        dict with:
        - 'winning_paths': Top 3 paths meeting 10% threshold (sorted by frequency desc)
        - 'path_stats': All paths with frequencies
        - 'needs_fallback': True if <3 paths meet 10% threshold

    Source: Critique Q5 (lines 301-407) - 10% threshold, confidence levels
    """
    from collections import Counter

    # ===== 1. Convert paths to tuples for counting (lists aren't hashable) =====
    path_tuples = [tuple(vp['path']) for vp in video_paths]

    # ===== 2. Count frequency of each unique path =====
    path_counts = Counter(path_tuples)
    # Example: Counter({(0,1,1,1,2,0): 22, (1,0,0,0,1,2): 18, (2,2,1,0,0,1): 12, ...})

    # ===== 3. Build path statistics with percentages =====
    path_stats = []
    for path_tuple, count in path_counts.items():
        percentage = (count / total_videos) * 100  # e.g., 22 / 100 = 22.0%

        path_stats.append({
            'path': list(path_tuple),  # Convert back to list for JSON serialization
            'frequency': count,
            'percentage': round(percentage, 1),
            'confidence_level': classify_confidence(percentage)
        })

    # ===== 4. Sort by frequency descending =====
    path_stats.sort(key=lambda x: x['frequency'], reverse=True)

    # ===== 5. Filter paths meeting 10% threshold =====
    winning_paths = [p for p in path_stats if p['percentage'] >= 10.0]

    # ===== 6. Check if fallback needed (<3 winning paths) =====
    needs_fallback = len(winning_paths) < 3

    # ===== 7. Take top 3 winning paths (if available) =====
    top_3_paths = winning_paths[:3]  # Take top 3 by frequency

    return {
        'winning_paths': top_3_paths,
        'all_paths': path_stats,  # For debugging/logging
        'needs_fallback': needs_fallback,
        'total_unique_paths': len(path_stats),
        'paths_above_threshold': len(winning_paths)
    }


def classify_confidence(percentage: float) -> str:
    """
    Classify confidence level based on path frequency percentage.

    Args:
        percentage: Path frequency percentage (e.g., 22.0 for 22%)

    Returns:
        str: "very_high", "high", or "moderate"

    Source: Critique Q5 (lines 301-407)
    """
    if percentage >= 20.0:
        return "very_high"
    elif percentage >= 15.0:
        return "high"
    else:  # 10.0-14.9%
        return "moderate"
```

**Example Execution** (bucket 18-33s, 100 videos):
```python
# After extracting cluster paths (Q9.1)
video_paths = extract_cluster_paths(window_types, kmeans_outputs)
# video_paths has 100 entries

# Analyze frequencies
analysis = analyze_path_frequencies(video_paths, total_videos=100)

# Result:
{
    'winning_paths': [
        {'path': [0, 1, 1, 1, 2, 0], 'frequency': 22, 'percentage': 22.0, 'confidence_level': 'very_high'},
        {'path': [1, 0, 0, 0, 1, 2], 'frequency': 18, 'percentage': 18.0, 'confidence_level': 'high'},
        {'path': [2, 2, 1, 0, 0, 1], 'frequency': 12, 'percentage': 12.0, 'confidence_level': 'moderate'}
    ],
    'all_paths': [
        # All 45 unique paths with frequencies (sorted desc)
    ],
    'needs_fallback': False,  # 3 paths meet 10% threshold
    'total_unique_paths': 45,
    'paths_above_threshold': 5  # 5 paths ≥10%, but only top 3 used
}
```

---

**Fallback Logic - If <3 Paths Meet 10% Threshold**:

**Scenario 1**: Only 2 paths meet 10% threshold
```python
analysis = {
    'winning_paths': [
        {'path': [0, 1, 1, 2, 0, 1], 'frequency': 18, 'percentage': 18.0, 'confidence_level': 'high'},
        {'path': [1, 0, 0, 1, 1, 0], 'frequency': 12, 'percentage': 12.0, 'confidence_level': 'moderate'}
    ],
    'needs_fallback': True,  # Only 2 paths
    'paths_above_threshold': 2
}

# Stage 7 Phase 2 action:
if analysis['needs_fallback']:
    # Generate 2 path-based reports + 1 feature-based report
    creative_reports = [
        # Report 1: Path-based (18%)
        build_path_based_report(winning_paths[0], ...),
        # Report 2: Path-based (12%)
        build_path_based_report(winning_paths[1], ...),
        # Report 3: Feature-based fallback (uses top RF features)
        build_feature_based_report(rf_video_data, ...)
    ]
```

**Scenario 2**: 0 paths meet 10% threshold (highly fragmented)
```python
analysis = {
    'winning_paths': [],  # Empty - no paths ≥10%
    'needs_fallback': True,
    'paths_above_threshold': 0,
    'all_paths': [
        {'path': [...], 'frequency': 9, 'percentage': 9.0},  # Highest is 9%
        {'path': [...], 'frequency': 8, 'percentage': 8.0},
        # ... all below 10%
    ]
}

# Stage 7 Phase 2 action:
if len(analysis['winning_paths']) == 0:
    # Generate 3 feature-based reports
    creative_reports = [
        # Report 1: Feature-based (top RF features)
        build_feature_based_report(rf_video_data, cluster_group='high'),
        # Report 2: Feature-based (middle RF features)
        build_feature_based_report(rf_video_data, cluster_group='medium'),
        # Report 3: Feature-based (diverse strategies)
        build_feature_based_report(rf_video_data, cluster_group='diverse')
    ]

    logger.warning(
        f"No cluster paths meet 10% threshold (highest: {analysis['all_paths'][0]['percentage']}%). "
        f"Generating 3 feature-based reports as fallback."
    )
```

---

**Confidence Level Mapping** (Critique Q5 lines 301-407):

| Frequency | Percentage | Confidence Level | Interpretation |
|-----------|-----------|------------------|----------------|
| 20+ videos | ≥20% | `very_high` | Strong pattern, highly reliable |
| 15-19 videos | 15-19.9% | `high` | Clear pattern, reliable |
| 10-14 videos | 10-14.9% | `moderate` | Noticeable pattern, acceptable |
| <10 videos | <10% | N/A (filtered out) | Too rare to include |

---

**Edge Cases**:

**Edge Case 1**: Exactly 3 paths at 10%
```python
# 3 paths with 10 videos each (10%)
analysis = {
    'winning_paths': [
        {'path': [0,1,1,2,0,1], 'frequency': 10, 'percentage': 10.0, 'confidence_level': 'moderate'},
        {'path': [1,0,0,1,1,0], 'frequency': 10, 'percentage': 10.0, 'confidence_level': 'moderate'},
        {'path': [2,2,1,0,0,1], 'frequency': 10, 'percentage': 10.0, 'confidence_level': 'moderate'}
    ],
    'needs_fallback': False  # Exactly 3 paths - no fallback needed
}

# All 3 reports are path-based
```

**Edge Case 2**: Ties at 10% (multiple paths same frequency)
```python
# 5 paths all have 10 videos (10%)
path_stats = [
    {'path': [0,1,1,2,0,1], 'frequency': 10, 'percentage': 10.0},
    {'path': [1,0,0,1,1,0], 'frequency': 10, 'percentage': 10.0},
    {'path': [2,2,1,0,0,1], 'frequency': 10, 'percentage': 10.0},
    {'path': [0,0,2,1,1,1], 'frequency': 10, 'percentage': 10.0},
    {'path': [1,1,0,0,2,2], 'frequency': 10, 'percentage': 10.0}
]

# Solution: Take first 3 in Counter order (arbitrary but deterministic)
winning_paths = path_stats[:3]

# Note: Counter order is insertion order (Python 3.7+), so deterministic within run
# But order may vary between runs - acceptable (all equally valid)
```

**Edge Case 3**: Bucket 3-9s (only 2 windows → 9 possible paths)
```python
# Bucket 3-9s has 2 windows (hook, closing)
# Possible paths: [0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]
# With 100 videos, expect 11-12 videos per path on average (if evenly distributed)

# Likely outcome: Most paths meet 10% threshold
analysis = {
    'winning_paths': [
        {'path': [0, 1], 'frequency': 18, 'percentage': 18.0, 'confidence_level': 'high'},
        {'path': [1, 0], 'frequency': 16, 'percentage': 16.0, 'confidence_level': 'high'},
        {'path': [0, 0], 'frequency': 12, 'percentage': 12.0, 'confidence_level': 'moderate'}
    ],
    'needs_fallback': False,
    'paths_above_threshold': 7  # 7 of 9 paths ≥10%
}

# Top 3 selected from 7 paths above threshold
```

---

**For HLD Section**: 2.3 (Detailed Process - Phase 2 frequency calculation), 5.2 (Output Schema - confidence_level field), 6.2 (Error Cases - fallback logic when <3 paths)

**For HLD Section - Hybrid Output Structure**:
```python
# winning_formulas.json structure
{
    "bucket": "18-33s",
    "total_videos": 100,
    "total_unique_paths": 45,
    "paths_above_threshold": 5,
    "creative_reports": [
        {
            "report_id": 1,
            "type": "path_based",  # or "feature_based" if fallback
            "path": [0, 1, 1, 1, 2, 0],
            "frequency": 22,
            "percentage": 22.0,
            "confidence_level": "very_high",
            "formula_name": "The Hook-to-Middle Momentum",
            "strategy_description": "...",
            "creator_recommendations": [...]
        },
        # ... reports 2-3
    ],
    "supplementary_insights": {
        "universal_principles": [
            # Top 5-7 RF features from video-level RF
            "High eye contact rate (88% vs 45% for top vs bottom performers)",
            "Consistent energy maintenance across windows",
            ...
        ],
        "cross_window_patterns": [
            "Videos with energy_delta >0.3 from hook to closing had 2x engagement",
            ...
        ]
    }
}
```

---
           cluster_id = get_video_cluster(video_id, window_type, kmeans_outputs)
           path.append(cluster_id)
   ```

3. **Missing Videos**: What if a video appears in hook but not in closing?
   - Should all 100 videos appear in all 6 windows?
   - Or can some videos be missing (filtered out during transformation)?
   - If missing, skip that video's path or error?

4. **Path Frequency Calculation**: After extracting all paths, how to apply 10% threshold?
   ```python
   # Example:
   # Path [0,1,1,1,2,0]: 18 videos (18%)  → Include ✅
   # Path [1,0,0,0,1,2]: 12 videos (12%)  → Include ✅
   # Path [2,2,1,0,0,1]: 8 videos (8%)    → Exclude ❌ (below 10%)

   # Filter logic:
   winning_paths = [p for p in all_paths if p['percentage'] >= 10.0]

   # Then limit to top 3 for creative_reports?
   creative_reports = winning_paths[:3]  # Top 3 by frequency
   ```

**For HLD Section**: 2.3 (Detailed Process - Phase 2 cluster path extraction logic), 5.2 (Output Schema - path filtering)

---

## Completeness Check

[Will be filled at end - see Step 6]

## Proceed to Phase 3

[Will be filled at end - see Step 6]
