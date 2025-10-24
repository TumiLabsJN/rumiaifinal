# Stage 7 Bugs & Solutions

**Date Discovered**: 2025-10-24
**Status**: 🔴 BLOCKED - Rate limiting preventing Stage 7 completion
**Affected**: All 3 test buckets (bucket_18-33s, bucket_13-18s, bucket_60-90s)

---

## Executive Summary

Stage 7 (LLM Analysis) fails consistently due to **Anthropic API rate limiting**. Phase 1 runs 6-7 windows in parallel per bucket, causing simultaneous API calls that exceed rate limits. API returns empty responses, leading to JSON parse errors.

**Impact**: Stage 7 cannot complete. All 3 buckets fail at Phase 1 (per-window analysis).

**Quick Fix**: Implement sequential window processing instead of parallel execution.

---

## Bug #1: API Rate Limiting Causing Empty Responses

### Symptoms

**Error Message**:
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
Location: ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py line 277
```

**Console Output**:
```
hook: JSON parse error: Expecting value: line 1 column 1 (char 0)
hook: Retrying after JSON parse error...
middle_1: JSON parse error: Expecting value: line 1 column 1 (char 0)
middle_1: Retrying after JSON parse error...
[... repeats for all windows ...]
✗ hook failed: Expecting value: line 1 column 1 (char 0)
Phase1ExecutionError: Phase 1 incomplete: hook failed after retries
```

**Anthropic Console Evidence**:
- 8 rate-limited requests observed
- Requests blocked before returning valid JSON
- Retry logic triggered but also rate-limited

### Root Cause

**Current Implementation** (stage7_llm_analysis.py lines 140-170):
```python
def run_phase1_parallel(bucket_path, bucket, hashtag, window_types):
    """Run Phase 1 analysis for all windows IN PARALLEL using ThreadPoolExecutor"""

    with ThreadPoolExecutor(max_workers=len(window_types)) as executor:
        futures = {
            executor.submit(analyze_window_with_retry, ...): window
            for window in window_types
        }
        # All 6-7 windows call Anthropic API simultaneously
```

**Problem**:
1. `max_workers=len(window_types)` = 6-7 parallel threads
2. All threads call Anthropic API at the same time
3. API rate limiter blocks requests
4. Empty/error responses returned
5. `json.loads(response_text)` fails with JSONDecodeError
6. Retry logic also hits rate limits

**Why This Happens**:
- Free/Tier 1 API keys have strict rate limits (5-50 requests per minute)
- 6 simultaneous requests exceed limits
- Rate-limited responses don't return valid JSON
- Code expects JSON, gets empty string → parse error

### Evidence

**From Stage 7 Test Run (2025-10-24)**:
- Test started: 12:07 UTC
- Duration: ~2 minutes of continuous retries
- Pattern: All 6 windows (hook, middle_1-4, closing) fail identically
- Anthropic Console: 8 rate-limited requests recorded
- All 3 buckets affected with same error

**Code Path**:
```
run_phase1_parallel() [line 140]
  → ThreadPoolExecutor spawns 6 threads
  → analyze_window_with_retry() [line 232]
    → call_anthropic_api() [line 277]
      → json.loads(response_text)
        → JSONDecodeError (response_text is empty)
```

---

## Solutions

### Solution 1: Sequential Execution (Recommended - Quick Fix)

**Complexity**: Low (1 line change)
**Impact**: Slower execution (6-7 min vs 1 min per bucket) but guaranteed to work
**Trade-off**: 6x longer runtime, but respects rate limits

**Implementation**:

**File**: `ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py`

**Change** (line 145):
```python
# BEFORE:
with ThreadPoolExecutor(max_workers=len(window_types)) as executor:

# AFTER:
with ThreadPoolExecutor(max_workers=1) as executor:  # Sequential execution
```

**Why This Works**:
- Only 1 API call at a time
- Natural 10-20s gap between calls (API response time)
- Stays well within rate limits (even 5 RPM allows 1 request every 12s)

**Estimated Time** (per bucket):
- bucket_18-33s: 6 windows × 15s = ~90s (Phase 1) + 20s (Phase 2) = **110s**
- bucket_13-18s: 3 windows × 15s = ~45s (Phase 1) + 20s (Phase 2) = **65s**
- bucket_60-90s: 7 windows × 15s = ~105s (Phase 1) + 20s (Phase 2) = **125s**
- **Total**: ~5 minutes for all 3 buckets (vs 1-2 min with parallel)

---

### Solution 2: Rate Limiting with Backoff (Better - Requires Code)

**Complexity**: Medium (20-30 lines)
**Impact**: Handles rate limits gracefully, partial parallelism possible
**Trade-off**: More complex, but adaptive

**Implementation**:

**File**: `ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py`

**Add** (after imports):
```python
import time
from anthropic import RateLimitError

class RateLimitHandler:
    def __init__(self, max_parallel=2, delay_between_calls=5):
        self.max_parallel = max_parallel
        self.delay_between_calls = delay_between_calls
        self.last_call_time = 0

    def wait_if_needed(self):
        """Enforce minimum delay between API calls"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.delay_between_calls:
            time.sleep(self.delay_between_calls - elapsed)
        self.last_call_time = time.time()
```

**Modify** `call_anthropic_api()` (line 270-280):
```python
def call_anthropic_api(prompt, max_retries=3):
    """Call Anthropic API with rate limit handling"""

    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()  # Add this

            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except RateLimitError as e:  # Handle rate limit explicitly
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential: 10s, 20s, 30s
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            logger.error(f"API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise
```

**Modify** `run_phase1_parallel()` (line 145):
```python
# Limit to 2 parallel workers with rate limiting
with ThreadPoolExecutor(max_workers=2) as executor:
```

**Why This Works**:
- 2 parallel requests with 5s minimum gap
- Stays within most rate limits (2 req × 12 per min = 24 RPM)
- Exponential backoff on rate limit errors
- Handles transient rate limits gracefully

**Estimated Time** (per bucket):
- bucket_18-33s: 6 windows ÷ 2 parallel × 15s = ~45s (Phase 1) + 20s (Phase 2) = **65s**
- **Total**: ~2-3 minutes for all 3 buckets

---

### Solution 3: Upgrade API Tier (Easiest - If Budget Allows)

**Complexity**: None (just billing change)
**Impact**: Higher rate limits allow parallel execution as designed
**Trade-off**: Monthly cost increase

**Steps**:
1. Log into Anthropic Console
2. Navigate to Settings → Billing
3. Check current tier and limits
4. Upgrade to Tier 2+ for higher rate limits

**Tier Comparison** (approximate):
| Tier | Requests/Min | Cost | Notes |
|------|-------------|------|-------|
| Free | 5 | $0 | Too low for parallel |
| Tier 1 | 50 | Usage-based | May still hit limits with 6 parallel |
| Tier 2 | 500+ | Usage-based | Easily handles 6 parallel |

**After Upgrade**: No code changes needed. Parallel execution will work as designed.

---

## Additional Context for Future CLI Instances

### What Stage 7 Does

Stage 7 generates LLM-powered creative insights using a **two-phase hybrid approach**:

**Phase 1**: Per-window cluster analysis
- Analyzes each temporal window (hook, middle_1-N, closing)
- For each window:
  - Loads K-Means clusters (3 clusters per window)
  - Loads RF feature importance (top 10 features)
  - Calls Anthropic API to interpret clusters and features
  - Generates actionable creator recommendations
- Output: 6-7 JSON files per bucket (one per window)

**Phase 2**: Cross-window synthesis
- Analyzes patterns across all windows
- Loads video-level RF analysis
- Combines Phase 1 insights
- Calls Anthropic API once for synthesis
- Output: 1 winning_formulas.json per bucket

**Total API Calls**: 7-8 per bucket × 3 buckets = **19-24 calls**

### Dependencies

**Stage 7 depends on**:
- ✅ Stage 6 outputs (35 JSON files from ml_analysis/)
  - rf_video_analysis.json (video-level RF)
  - {window}_rf_analysis.json (window-level RF)
  - {window}_kmeans_analysis.json (window-level K-Means)
- ✅ ANTHROPIC_API_KEY environment variable
- ✅ anthropic Python package installed

**Stage 7 produces**:
- Phase 1 outputs: `ml_analysis/llm/{window}_analysis.json` (6-7 per bucket)
- Phase 2 outputs: `ml_analysis/llm/winning_formulas.json` (1 per bucket)
- Combined output: `ml_analysis/llm/complete_analysis_{bucket}.json` (1 per bucket)

### File Locations

**Script**: `ml_pipeline/stage7_llm_analysis/stage7_llm_analysis.py`
**Key Functions**:
- `main()` - Entry point (line 565)
- `run_phase1_parallel()` - Parallel window analysis (line 140)
- `analyze_window_with_retry()` - Single window analysis with retry (line 232)
- `call_anthropic_api()` - API wrapper (line 270)

**Config**: `.env` file contains `ANTHROPIC_API_KEY`

**Test Data**: `data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/`

---

## Test Results (2025-10-24)

**Test Configuration**:
- API Key: Set from `.env`
- Buckets: 3 (bucket_18-33s, bucket_13-18s, bucket_60-90s)
- Expected API calls: 6 + 3 + 7 = 16 (Phase 1) + 3 (Phase 2) = **19 total**

**Actual Results**:
- API calls attempted: 16+ (all Phase 1 windows)
- Rate-limited requests: 8 (observed in Anthropic Console)
- Successful calls: 0
- JSONDecodeError: 100% of attempts
- Duration: ~2 minutes of continuous retries

**Failure Mode**:
- All windows fail at first attempt (rate limited)
- Retry logic triggers
- Retries also rate limited
- After exhausting retries, Phase1ExecutionError raised
- Process continues to next bucket (same result)

---

## Recommendations for Future Testing

### Before Running Stage 7

1. **Check API Tier**:
   ```bash
   # Check Anthropic Console → Settings → Limits
   # Verify: Requests per minute (RPM) limit
   ```

2. **Choose Execution Mode**:
   - If Free/Tier 1: Use Solution 1 (sequential)
   - If Tier 2+: Can use parallel as designed
   - If uncertain: Use Solution 2 (rate limiting with backoff)

3. **Estimate Time**:
   - Sequential (Solution 1): ~5 minutes for 3 buckets
   - Rate-limited parallel (Solution 2): ~2-3 minutes
   - Full parallel (if tier allows): ~60-90 seconds

4. **Monitor Console**:
   - Keep Anthropic Console open
   - Watch for rate-limited requests
   - If rate limits appear, stop and implement Solution 1 or 2

### After Implementing Fix

1. **Test with 1 bucket first**:
   ```bash
   python -c "
   from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main
   main('data/clients/test_final/.../bucket_18-33s', '18-33s', 'test_vitamin')
   "
   ```

2. **Verify outputs**:
   ```bash
   ls data/.../bucket_18-33s/ml_analysis/llm/*.json
   # Expected: 8 files (6 windows + 1 winning_formulas + 1 complete)
   ```

3. **Check Anthropic Console**:
   - Verify: 7 successful requests (6 Phase 1 + 1 Phase 2)
   - No rate-limited requests

4. **Run remaining buckets**:
   - If first bucket succeeds, proceed with bucket_13-18s and bucket_60-90s
   - Expected: 12 more successful requests (5 + 7 Phase 1 + 2 Phase 2)

---

## Known Warnings (Not Bugs)

**Warning**: `Feature X missing distribution data`
- **Expected**: 83.3% of features have distributions (from Stage 6 validation)
- **Cause**: Cross-window delta features (e.g., `energy_progression_slope`) and derived features (e.g., `day_of_week`) lack distribution data per HLD design
- **Impact**: None - LLM handles missing distributions gracefully
- **Action**: Ignore these warnings

**Pattern**:
```
Feature energy_max missing distribution data
Feature shortest_scene missing distribution data
[... many similar warnings ...]
```

**These are NOT errors** - they're informational logs from Stage 6 data loading.

---

## Related Documentation

- **Bug Discovery**: Stage7_Test_Results.md (to be created after fix)
- **Stage 6 Validation**: Stage6_Test_Results.md (83.3% distribution coverage)
- **API Documentation**: https://docs.anthropic.com/en/api/rate-limits
- **Stage 7 HLD**: documentation_migration/FutureDevelopments/ChildDocs/LLMAnalysisCHILD.md
- **General Tests**: RumiGeneralTests.md (Stage 7 section at line 1627)

---

## Quick Reference Commands

**Check if API key is set**:
```bash
echo $ANTHROPIC_API_KEY
# Should output: sk-ant-... (not empty)
```

**Run Stage 7 for one bucket**:
```bash
cd /home/jorge/rumiaifinal
export ANTHROPIC_API_KEY="$(cat .env | grep ANTHROPIC_API_KEY | cut -d '=' -f 2)"
/home/jorge/rumiaifinal/venv/bin/python3 -c "
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main
main('data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s', '18-33s', 'test_vitamin')
"
```

**Validate outputs**:
```bash
BASE="data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets"
for bucket in bucket_18-33s bucket_13-18s bucket_60-90s; do
    echo "$bucket:"
    ls "$BASE/$bucket/ml_analysis/llm/"*.json 2>/dev/null | wc -l
done
# Expected: 8 (18-33s), 5 (13-18s), 9 (60-90s)
```

---

---

## Bug #2: Missing Distribution Data in Stage 6 Output

### Status
🟡 **IDENTIFIED** - Needs further analysis before implementing fix

### Discovery Date
2025-10-24 (during investigation of Bug #1)

### Symptoms

**Warning Messages** (repeated for most features):
```
Feature person_count missing distribution data
Feature energy_variance missing distribution data
Feature energy_max missing distribution data
Feature pitch_scatter_ratio missing distribution data
...
```

**Code Behavior**:
- Stage 7 prompt builder logs warnings for 100% of RF features
- Falls back to UNIMODAL assumption (no bimodal patterns detected)
- Prompt still gets built and sent to Claude API
- Claude responds successfully (1000+ token responses observed)

### Root Cause

**Stage 6 Output Schema Missing `distribution` Field**:

**Current Stage 6 RF Output** (`hook_rf_analysis.json`):
```json
{
  "feature_importance": [
    {
      "feature": "person_count",
      "importance": 0.10038,
      "rank": 1,
      "top_performer_avg": 26.92,
      "bottom_performer_avg": 7.1,
      "gap": 19.82
      // ❌ Missing: "distribution" field
    }
  ]
}
```

**Expected by Stage 7** (per LLMAnalysisCHILDTI.md Section 4.1):
```json
{
  "feature_importance": [
    {
      "feature": "person_count",
      "importance": 0.10038,
      "rank": 1,
      "top_performer_avg": 26.92,
      "bottom_performer_avg": 7.1,
      "gap": 19.82,
      // ✅ Expected: distribution data for bimodal detection
      "distribution": {
        "top_performers": {
          "high_percentage": 0.40,  // % with ≥66th percentile
          "low_percentage": 0.35    // % with <33rd percentile
        },
        "bottom_performers": {
          "high_percentage": 0.18,
          "low_percentage": 0.52
        }
      }
    }
  ]
}
```

### Impact Assessment

**What Distribution Data Enables**:
- **Bimodal pattern detection**: Identifies when TWO strategies work (e.g., brief vs dense word counts)
- **Nuanced recommendations**: "Use Strategy A OR Strategy B" instead of "Use average"
- **Higher quality insights**: Creators understand multiple paths to success

**Current Behavior WITHOUT Distribution Data**:
- ✅ Prompts still get built (fallback to UNIMODAL)
- ✅ Claude still generates responses
- ✅ Recommendations still get created
- ❌ Misses bimodal patterns (e.g., "Use 52 words" instead of "Brief ≤20 OR Dense ≥80")
- ❌ Less actionable recommendations
- ❌ Quality degradation (moderate, not critical)

**Severity**: **MEDIUM** - Not blocking, but reduces output quality

### Evidence

**Test Script Confirmation** (2025-10-24):
```bash
# Tested Anthropic API directly - works perfectly
python test_anthropic_response.py
# Result: ✓ API works, response.content[0].text is valid, JSON parsing succeeds
```

**Anthropic Console Confirmation**:
- 9+ successful API calls observed
- Each returning 1000+ tokens
- No rate limiting errors (after applying max_workers=1 fix)
- HTTP 200 responses
- Billing charges confirm successful completions

**Conclusion**: The API and response extraction work correctly. The issue is NOT with API communication.

### Analysis Required

**Before Implementing Fix**:

1. **Verify Stage 6 Design Intent**:
   - Was `distribution` field intentionally removed?
   - Check Stage 6 HLD (MLAnalysisGenerationCHILD.md) for schema specification
   - Check git history for when distribution was removed (if it ever existed)

2. **Assess Stage 6 Complexity**:
   - How hard is it to add distribution calculation to Stage 6?
   - Does Stage 6 have the raw data needed to compute distributions?
   - What's the computational cost?

3. **Evaluate Stage 7 Workaround**:
   - Can Stage 7 compute distributions from Stage 6 data?
   - Is there enough info in `top_performer_avg`, `bottom_performer_avg`, `gap`?
   - Or does it need raw video-level data (not in Stage 6 output)?

4. **Check Test Data Quality**:
   - Do the 47 videos in bucket_18-33s provide enough signal for bimodal detection?
   - Maybe distributions are only meaningful with larger samples (100+ videos)?

### Possible Solutions

**Option A: Fix Stage 6** (Longer-term, higher quality)
- Add distribution calculation to Stage 6 RF analysis
- Update Stage 6 output schema to include `distribution` field
- Provides full bimodal detection capability
- Requires Stage 6 code changes + re-running Stage 6 on test data

**Option B: Stage 7 Workaround** (Short-term, partial solution)
- Estimate distribution from `top_performer_avg` + `gap`
- Limited accuracy but better than UNIMODAL assumption
- No Stage 6 changes needed

**Option C: Accept Limitation** (Current state)
- Keep current UNIMODAL fallback
- Document as known limitation
- Acceptable if bimodal patterns are rare/not critical

**Option D: Defer to Production** (Wait and see)
- Use current system for test_vitamin dataset (111 videos)
- Re-evaluate with larger production datasets (300+ videos per hashtag)
- Bimodal patterns may be more evident with larger samples

### Recommendation

**Defer decision until analyzing:**
1. Stage 6 HLD/TI schema specifications
2. Git history of distribution field
3. Test data sample size considerations

**For now**: Document as known limitation, continue with existing UNIMODAL fallback.

---

## Bug #3: JSON Parse Errors Despite Successful API Responses

### Status
🔴 **ACTIVE** - Under investigation

### Discovery Date
2025-10-24 (after applying Bug #1 fix)

### Symptoms

**Error Message** (persists even with sequential execution):
```
hook: JSON parse error: Expecting value: line 1 column 1 (char 0)
hook: Retrying after JSON parse error...
hook: JSON parse error: Expecting value: line 1 column 1 (char 0)
hook: Retrying after JSON parse error...
hook: JSON parse error: Expecting value: line 1 column 1 (char 0)
✗ hook failed: Expecting value: line 1 column 1 (char 0)
```

**Paradox**:
- ✅ Anthropic Console shows successful API calls (1000+ token responses)
- ✅ Test script confirms response extraction works (`response.content[0].text`)
- ❌ Stage 7 code reports empty response → JSON parse fails

### Analysis

**What We Know**:
1. API is working (verified via Console + test script)
2. Response structure is correct (`response.content[0].text` is valid)
3. Claude IS generating 1000+ token responses
4. Sequential execution (max_workers=1) has been applied

**What We Don't Know**:
1. **Is Claude returning valid JSON?** (Most likely issue)
2. **Or is Claude returning error messages/explanations?** (Due to malformed prompt from missing distribution data?)
3. **What does the actual 1000-token response contain?**

### Hypothesis

**Claude is returning plain text instead of JSON** because:
- Prompt contains warnings about missing distribution data
- Prompt may be confusing or incomplete
- Claude responds with explanation: "I apologize, but I cannot generate the analysis because several features are missing distribution data..."
- Code tries to parse this plain text as JSON → fails

**Supporting Evidence**:
- ALL windows fail identically (suggests systematic prompt issue, not random)
- Missing distribution data warnings for 100% of features
- Claude successfully responds (1000+ tokens) but code can't parse it

### Next Steps

**To Diagnose**:
1. **Capture actual Claude response** before trying to parse as JSON
2. **Add debug logging**: Print first 500 chars of `response.content[0].text`
3. **Examine one real response** to see if it's JSON or plain text
4. **If plain text**: Fix the prompt to be clearer despite missing distribution data
5. **If valid JSON**: Investigate why parsing fails (encoding issue? hidden characters?)

**Implementation**:
```python
# Add before line 277 (json.loads):
response_text = response.content[0].text
logger.info(f"{window_type}: Response preview: {response_text[:500]}")
logger.info(f"{window_type}: Response length: {len(response_text)} chars")
# Then: analysis = json.loads(response_text)
```

---

## Status Updates

| Date | Status | Action Taken | Result |
|------|--------|--------------|--------|
| 2025-10-24 09:00 | 🔴 BLOCKED | Initial Stage 7 test run | Rate limiting discovered (Bug #1) |
| 2025-10-24 09:30 | ⏳ INVESTIGATING | Applied max_workers=1 fix (Bug #1 Solution 1) | Fix applied, retesting |
| 2025-10-24 09:37 | 🟡 PARTIALLY FIXED | Tested with sequential execution | API calls successful, but JSON parse errors persist (Bug #3) |
| 2025-10-24 09:45 | 🟡 ANALYSIS | Created test_anthropic_response.py | Confirmed API + response extraction work correctly |
| 2025-10-24 09:50 | 🟡 DISCOVERY | Analyzed Stage 6 output schema | Discovered missing distribution data (Bug #2) |
| TBD | ⏳ PENDING | Capture actual Claude response | Need to see what Claude is returning |
| TBD | ⏳ PENDING | Fix prompt or response parsing | Awaiting Bug #3 diagnosis |
| TBD | ✅ RESOLVED | Verify all bugs fixed | Expected: 19/19 calls successful, valid JSON outputs |

---

**Last Updated**: 2025-10-24 10:00
**Next Action**:
1. **Immediate**: Add debug logging to capture actual Claude response (Bug #3)
2. **Short-term**: Analyze Stage 6 schema and decide on distribution data fix (Bug #2)
3. **Final**: Verify full Stage 7 pipeline with all 3 buckets
