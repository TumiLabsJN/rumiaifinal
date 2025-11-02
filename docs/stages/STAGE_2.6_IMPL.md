# Stage 2.6: Content Discovery - Implementation Guide

**Purpose**: Discover content patterns from sample transcripts using LLM (Claude Sonnet)
**Target Audience**: LLM agents fixing bugs or adding features to Stage 2.6
**Related**: [PRODUCTION_FLOW.md Stage 2.6 Contract](../../PRODUCTION_FLOW.md#stage-26-content-discovery)

---

## Quick Reference

- **Entry Point**: `ml_pipeline/stage2_content_analysis/discovery.py::run_discovery_stage()` (line 546-642)
- **Orchestrator Call**: `rumiai_ml_batch.py:862-936`
- **State Tracking**: `.content_analysis_state.json` (one-time execution flag)
- **Average Duration**: ~60-90s for 60 sample transcripts (LLM API call)
- **Bottleneck**: Claude Sonnet API call (single large request)
- **⚠️ CRITICAL**: **ONE-TIME EXECUTION** - Stage runs once, then **BLOCKS** pipeline until manual curation

---

## Input Contract

### Prerequisites
**Required Stages**:
- Stage 2.5 (File Organization) → `selection_manifest.json`
- Stage 2.5.1 (Transcript Validation) → `.validation/transcript_validation_cache.json`

**Input Files**:
```
{analysis_base}/
├── selection_manifest.json          # Created by Stage 2.5 (video IDs per bucket)
└── validation/
    └── transcript_validation_cache.json  # Created by Stage 2.5.1 (valid/invalid flags)

/home/jorge/rumiaifinal/speech_transcriptions/
└── {video_id}_whisper.json          # Created by Stage 2 (Whisper transcripts)
```

**Validation Logic**:
```python
# ml_pipeline/stage2_content_analysis/validation.py:18-86
def validate_discovery_inputs(manifest_path: str, sample_size: int):
    # Check 1: Manifest exists
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Check 2: Manifest has required fields
    manifest = load_json(manifest_path)
    required = ['hashtag', 'selected_buckets', 'videos_by_bucket']
    missing = [f for f in required if f not in manifest]
    if missing:
        raise ValueError(f"Manifest missing fields: {missing}")

    # Check 3: Sample size valid (10-100)
    if not 10 <= sample_size <= 100:
        raise ValueError(f"Sample size {sample_size} out of range [10, 100]")

    # Check 4: At least 3 buckets
    if len(manifest['selected_buckets']) < 3:
        raise ValueError("Need at least 3 winning buckets")

    # Check 5: Each bucket has videos
    for bucket in manifest['selected_buckets']:
        if bucket not in manifest['videos_by_bucket']:
            raise ValueError(f"Bucket {bucket} missing from videos_by_bucket")

        performers = manifest['videos_by_bucket'][bucket]
        if not performers.get('top_performers'):
            raise ValueError(f"Bucket {bucket} has no top performers")

    # Check 6: ANTHROPIC_API_KEY set
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set with: export ANTHROPIC_API_KEY=sk-ant-..."
        )
```

**Failure Modes**:
- **Missing manifest**: Raises `FileNotFoundError` → Stage 2.5 incomplete
- **Missing validation cache**: Raises `FileNotFoundError` → Stage 2.5.1 incomplete
- **Invalid sample size**: Raises `ValueError` → Check CLI args
- **Missing API key**: Raises `ValueError` → Set environment variable

---

## Output Contract

### Files Created
```
{analysis_base}/content_taxonomies/
├── {hashtag}_raw_discovery.json     # LLM-generated taxonomy (7 categories)
└── {hashtag}_taxonomy.json          # 🔴 MANUAL CURATION REQUIRED

{analysis_base}/
└── .content_analysis_state.json     # State tracking (one-time execution)
```

### Output Schema

**{hashtag}_raw_discovery.json**:
```json
{
  "metadata": {
    "hashtag": "#nutrition",
    "sample_size": 60,
    "discovered_at": "2025-01-28T10:30:00Z",
    "client_id": "acme_corp"
  },
  "discovered_patterns": {
    "content_categories": [
      {
        "name": "recipe_tutorial",
        "description": "Step-by-step cooking demonstrations",
        "frequency": 24,
        "percentage": 40.0,
        "example_video_ids": ["7545713916584774968", "7428596413707144481"]
      }
      // ... 3-8 categories total
    ],
    "hook_strategies": [
      {
        "name": "question_hook",
        "description": "Opens with a question to viewer",
        "frequency": 18,
        "percentage": 30.0
      }
      // ... 2-5 strategies total
    ],
    "closing_strategies": [
      {"name": "cta_link", "description": "Directs to link in bio"}
      // ... 2-5 strategies total (no frequency, simple list)
    ],
    "audience_pain_points": [
      {"name": "bloating", "description": "Digestive discomfort"}
      // ... simple list
    ],
    "trending_keywords": [
      {"name": "#guthealth", "description": "Digestive wellness"}
      // ... simple list
    ],
    "engagement_drivers": [
      {"name": "personal_testimony", "description": "Creator's own experience"}
      // ... simple list
    ],
    "content_tactics": [
      {"name": "before_after_reveal", "description": "Transformation showcase"}
      // ... simple list
    ]
  }
}
```

**Validation**: Post-LLM validation checks (line 246-298):
```python
# ml_pipeline/stage2_content_analysis/validation.py:246-298
def validate_discovery_output(raw_taxonomy: Dict[str, Any]):
    required_categories = [
        'content_categories',
        'hook_strategies',
        'closing_strategies',
        'audience_pain_points',
        'trending_keywords',
        'engagement_drivers',
        'content_tactics'
    ]

    # Check all 7 categories exist
    for category in required_categories:
        if category not in raw_taxonomy['discovered_patterns']:
            raise ValueError(f"Missing category: {category}")

        patterns = raw_taxonomy['discovered_patterns'][category]
        if not isinstance(patterns, list) or len(patterns) == 0:
            raise ValueError(f"Category {category} is empty")

    # Validate content_categories and hook_strategies have frequency
    for category in ['content_categories', 'hook_strategies']:
        for pattern in raw_taxonomy['discovered_patterns'][category]:
            if 'frequency' not in pattern:
                raise ValueError(f"{category} pattern missing frequency")
            if pattern['frequency'] < 1:
                raise ValueError(f"{category} frequency must be >= 1")
```

---

## Implementation Details

### Core Functions

| Function | File | Line | Purpose | Calls |
|----------|------|------|---------|-------|
| `run_discovery_stage()` | `discovery.py` | 546-642 | Main entry point | `sample_transcripts_for_discovery()`, `discover_patterns_llm()` |
| `sample_transcripts_for_discovery()` | `discovery.py` | 26-226 | Adaptive sampling (20 per bucket) | `load_json()`, `validate_business_rules_sampling()` |
| `discover_patterns_llm()` | `discovery.py` | 229-508 | LLM pattern extraction | `anthropic.messages.create()` |
| `calculate_percentages()` | `discovery.py` | 510-544 | Add percentage fields | (none) |
| `validate_discovery_inputs()` | `validation.py` | 18-86 | Pre-flight validation | (none) |
| `validate_discovery_output()` | `validation.py` | 246-298 | Post-LLM validation | (none) |
| `load_validation_cache()` | `transcript_validation.py` | 348-407 | Load Stage 2.5.1 cache | `load_json()` |

### Data Flow

```
selection_manifest.json (Stage 2.5)
    ↓ [sample_transcripts_for_discovery()]
Sampled transcripts (60, adaptive per bucket)
    ↓ [discover_patterns_llm()]
Raw taxonomy (LLM response)
    ↓ [calculate_percentages()]
{hashtag}_raw_discovery.json
    ↓ **MANUAL CURATION**
{hashtag}_taxonomy.json
```

### Critical Logic

#### 1. Adaptive Sampling Strategy

**Location**: `discovery.py:26-226`

**Purpose**: Sample 60 transcripts (20 per bucket) with adaptive distribution

**Algorithm**:
```python
# Step 1: Assess valid counts per bucket
for bucket in top_3_buckets:
    valid_ids = [
        vid for vid in top_performers
        if vid in validation_cache and validation_cache[vid]['is_valid']
    ]
    bucket_valid_counts[bucket] = len(valid_ids)

# Step 2: Determine sampling plan
for bucket in top_3_buckets:
    if bucket_valid_counts[bucket] < 20:
        # Weak bucket: take all available
        sampling_plan[bucket] = bucket_valid_counts[bucket]
        shortfall += (20 - bucket_valid_counts[bucket])
    else:
        # Strong bucket: initially 20, may increase
        sampling_plan[bucket] = 20
        surplus_buckets.append(bucket)

# Step 3: Distribute shortfall to surplus buckets
if shortfall > 0 and surplus_buckets:
    extra_per_surplus = shortfall // len(surplus_buckets)
    for bucket in surplus_buckets:
        sampling_plan[bucket] += extra_per_surplus
```

**Edge Cases**:
- **Scenario 1** (All healthy): 46, 48, 44 valid → Sample 20, 20, 20 = 60 ✅
- **Scenario 2** (One weak): 12, 50, 48 valid → Sample 12, 24, 24 = 60 ✅
- **Scenario 3** (Insufficient): 8, 12, 18 valid → Sample 8, 12, 18 = 38 ⚠️ (raises ValueError if <10 total)

#### 2. LLM Pattern Discovery

**Location**: `discovery.py:229-508`

**Purpose**: Extract 7 taxonomy categories using Claude Sonnet

**API Configuration**:
```python
# discovery.py:429-447
response = client.messages.create(
    model="claude-sonnet-4-20250514",  # Most capable model
    max_tokens=8000,                   # Large taxonomy output
    temperature=0.3,                   # Lower = more consistent
    timeout=120.0,                     # 2-minute timeout
    system=system_message,             # Role definition
    messages=[{"role": "user", "content": prompt}]
)
```

**Retry Logic** (3 attempts, exponential backoff):
```python
# discovery.py:448-507
for attempt in range(1, 4):  # 3 attempts
    try:
        response = client.messages.create(...)
        response_text = response.content[0].text

        # Parse JSON
        raw_taxonomy = parse_llm_json(response_text)

        # Validate output
        validate_discovery_output(raw_taxonomy)

        # Add percentages
        raw_taxonomy_with_pct = calculate_percentages(raw_taxonomy, sample_size)

        return raw_taxonomy_with_pct

    except TimeoutError as e:
        if attempt < 3:
            delay = 5 * attempt  # 5s, 10s
            logger.warning(f"Attempt {attempt}/3 timed out, retrying in {delay}s")
            time.sleep(delay)
        else:
            raise TimeoutError("LLM timed out after 3 attempts")

    except ValueError as e:  # Invalid JSON
        if attempt < 3:
            delay = 2 * attempt
            logger.warning(f"Attempt {attempt}/3 invalid JSON, retrying in {delay}s")
            time.sleep(delay)
        else:
            raise ValueError(f"LLM returned invalid JSON after 3 retries: {e}")
```

**Cost**: ~$0.15-0.30 per discovery (60 transcripts, 8000 max tokens)

#### 3. One-Time Execution State Management

**Location**: `rumiai_ml_batch.py:862-936` (orchestrator)

**State File** (`.content_analysis_state.json`):
```json
{
  "discovery_complete": true,
  "taxonomy_curated": false,  // User sets to true after manual edit
  "taxonomy_version": "1.0",
  "discovered_at": "2025-01-28T10:30:00Z"
}
```

**Orchestrator Logic**:
```python
# rumiai_ml_batch.py:885-915
state_file = analysis_base / ".content_analysis_state.json"

if state_file.exists():
    state = load_json(state_file)

    if state.get("discovery_complete") and not state.get("taxonomy_curated"):
        # Discovery done, waiting for manual curation
        logger.info("🔴 Stage 2.6 complete, waiting for manual taxonomy curation")
        logger.info("   Edit {hashtag}_taxonomy.json and set taxonomy_curated=true")
        sys.exit(2)  # Exit code 2: Paused for curation

    if state.get("taxonomy_curated"):
        logger.info("✅ Taxonomy curated, skipping discovery (one-time stage)")
        continue  # Skip to Stage 2.7
else:
    # First run: Execute discovery
    run_discovery_stage(...)

    # Save state
    state = {
        "discovery_complete": true,
        "taxonomy_curated": false,
        "taxonomy_version": "1.0",
        "discovered_at": datetime.utcnow().isoformat()
    }
    save_json(state_file, state)

    logger.info("🔴 Pipeline paused for manual taxonomy curation (~15 min)")
    logger.info("   1. Review {hashtag}_raw_discovery.json")
    logger.info("   2. Edit categories in {hashtag}_taxonomy.json")
    logger.info("   3. Set taxonomy_curated=true in .content_analysis_state.json")
    logger.info("   4. Re-run pipeline to continue with Stage 2.7")
    sys.exit(2)  # Block pipeline
```

---

## Error Handling

### Stage 2.6 Errors

**From orchestrator** (`rumiai_ml_batch.py:862-936`):

| Exception | Cause | Action | Exit Code |
|-----------|-------|--------|-----------|
| `FileNotFoundError` | Manifest or cache missing | Exit pipeline | 1 |
| `ValueError` | Validation failure | Exit pipeline | 1 |
| `TimeoutError` | LLM timeout after 3 retries | Exit pipeline | 8 |
| `RuntimeError` | API authentication | Exit pipeline | 99 |

### Common Failure Scenarios

**Scenario 1**: **Missing validation cache**
- **Cause**: Stage 2.5.1 skipped or failed
- **Detection**: `load_validation_cache()` raises `FileNotFoundError` (line 607-614)
- **Action**: Pipeline exits with error message
- **Recovery**:
  ```bash
  # Re-run Stage 2.5.1
  python -m ml_pipeline.stage2_content_analysis.transcript_validation \
    --client acme --hashtag nutrition
  ```

**Scenario 2**: **Insufficient valid transcripts (<10)**
- **Cause**: Most top performers have music/noise transcripts
- **Detection**: `sample_transcripts_for_discovery()` raises `ValueError` (line 216-220)
- **Action**: Pipeline exits
- **Recovery**:
  - Check transcript validation summary
  - Consider scraping more videos (increase `--video-count`)
  - Review invalid reasons (most common: music, noise)

**Scenario 3**: **LLM returns invalid JSON**
- **Cause**: Claude Sonnet output parsing error
- **Detection**: `parse_llm_json()` fails after 3 retries (line 502-504)
- **Action**: Pipeline exits with ValueError
- **Recovery**:
  - Check API logs for response preview (logged at line 503)
  - Verify prompt hasn't been modified
  - Retry (random LLM variance may succeed)

**Scenario 4**: **LLM timeout (>120s)**
- **Cause**: API slow or overloaded
- **Detection**: `messages.create()` raises `TimeoutError` after 3 retries (line 493-500)
- **Action**: Pipeline exits with TimeoutError
- **Recovery**: Retry later (Anthropic API issue, not code)

---

## Modification Guide

### Adding a New Taxonomy Category

**Scenario**: Add "visual_patterns" as 8th category

**Steps**:

1. **Update prompt** (`discovery.py:262-340`)
   ```python
   # Add new category section to prompt
   prompt = f"""...

   ## CATEGORY 8: Visual Patterns

   Identify recurring VISUAL elements (camera angles, editing styles).

   Examples to show naming style:
   - close_up_hands: Tight shot of hands working
   - dynamic_transitions: Fast cuts between scenes

   ...
   """
   ```

2. **Update output validation** (`validation.py:246-298`)
   ```python
   required_categories = [
       'content_categories',
       'hook_strategies',
       'closing_strategies',
       'audience_pain_points',
       'trending_keywords',
       'engagement_drivers',
       'content_tactics',
       'visual_patterns'  # ADD THIS
   ]
   ```

3. **Update taxonomy schema** (documentation)
   - Update `PRODUCTION_FLOW.md` Stage 2.6 contract
   - Update Stage 2.7 classification prompt to include new category

4. **Test**: Run discovery on test hashtag
   ```bash
   python -c "
   from ml_pipeline.stage2_content_analysis.discovery import run_discovery_stage
   result = run_discovery_stage('test', 'nutrition', 'hashtag')
   print('visual_patterns' in result['discovered_patterns'])
   "
   ```

5. **Downstream impact**:
   - Stage 2.7 classification will need updated prompt
   - Stage 8 reports may need schema changes

---

## Debugging Checklist

**If Stage 2.6 fails**:
- [ ] Check Stage 2.5 completed (`selection_manifest.json` exists)
- [ ] Verify Stage 2.5.1 completed (`.validation/transcript_validation_cache.json` exists)
- [ ] Confirm `ANTHROPIC_API_KEY` environment variable set
- [ ] Check validation cache has valid transcripts (inspect `.validation/transcript_validation_cache.json`)
- [ ] Review sample size (must be 10-100)
- [ ] Check internet connection (LLM API call)
- [ ] Review logs for specific error (timeout, invalid JSON, missing files)

**If discovery returns unexpected patterns**:
- [ ] Review sampled transcripts (check bucket distribution)
- [ ] Verify transcripts are actually valid (not music/noise)
- [ ] Check LLM prompt hasn't been modified
- [ ] Review example_video_ids in output (are they relevant?)
- [ ] Consider increasing sample_size for better coverage

**Manual Curation Checklist**:
- [ ] Review `{hashtag}_raw_discovery.json` output
- [ ] Edit category names for clarity (keep snake_case format)
- [ ] Merge duplicate categories (e.g., "recipe_tutorial" + "cooking_demo")
- [ ] Remove low-frequency noise categories (<5%)
- [ ] Update descriptions for creator clarity
- [ ] Save as `{hashtag}_taxonomy.json`
- [ ] Set `taxonomy_curated=true` in `.content_analysis_state.json`
- [ ] Re-run pipeline to continue with Stage 2.7

---

## Dependencies

### Python Modules
- `anthropic` (>=0.17.0) - Claude API client
- `json` - JSON parsing
- `os` - Environment variables
- `random` - Sampling
- `time` - Retry delays

### Internal Imports
- `.utils` - JSON helpers (`load_json`, `save_json`, `parse_llm_json`)
- `.validation` - Input/output validation
- `.error_handlers` - Graceful error handling
- `.cost_tracking` - Token usage logging
- `.transcript_validation` - Validation cache loading
- `foundation.paths` - Path construction

### External Services
- **Anthropic API**: Requires `ANTHROPIC_API_KEY` environment variable
  - Model: `claude-sonnet-4-20250514`
  - Endpoint: `https://api.anthropic.com/v1/messages`
  - Cost: ~$0.15-0.30 per 60-transcript discovery

---

## Testing

### Test Command
```bash
# Run Stage 2.6 only (assumes Stage 2.5 + 2.5.1 complete)
python -c "
from ml_pipeline.stage2_content_analysis.discovery import run_discovery_stage

result = run_discovery_stage(
    client_id='test',
    hashtag='nutrition',
    analysis_type='hashtag',
    sample_size=60,
    random_seed=42  # Reproducible sampling for testing
)

print(f'Discovered {len(result[\"discovered_patterns\"][\"content_categories\"])} categories')
print(f'Output: content_taxonomies/nutrition_raw_discovery.json')
"
```

### Expected Output
- File: `content_taxonomies/nutrition_raw_discovery.json` with 7 categories
- Duration: ~60-90s (LLM API call)
- Sample size: 60 transcripts (adaptive 20 per bucket)
- State: `.content_analysis_state.json` with `discovery_complete=true`
- Pipeline status: **PAUSED** (exit code 2) for manual curation

### Test Data
- **Minimum**: 3 buckets with 20 valid transcripts each (60 total)
- **Full**: 3 buckets with 40+ valid transcripts each (120+ total)

---

## Performance Characteristics

### Timing Breakdown
- **Sampling**: ~2-5s (loading 60 transcripts from disk)
- **LLM API call**: ~50-80s (Claude Sonnet processing)
- **Post-processing**: ~1-2s (percentage calculation, validation)
- **Total**: ~60-90s per discovery

### Bottlenecks
- **Primary**: Claude Sonnet API call (single large request, not parallelizable)
- **Secondary**: None (sampling is fast)

### Optimization Opportunities
- ❌ Cannot parallelize (single LLM call per hashtag)
- ❌ Cannot reduce sample size (need 60 for balanced representation)
- ✅ Could cache transcripts in memory (saves ~2s on re-runs)
- ✅ Could implement prompt caching (Anthropic feature, 50% cost reduction on retries)

---

## Related Documentation

- **PRODUCTION_FLOW.md**: [Stage 2.6 Contract](../../PRODUCTION_FLOW.md#stage-26-content-discovery)
- **Technical Spec**: [`ContentAnalysisCHILDTI.md`](../../documentation_migration/ContentAnalysisCHILDTI.md)
- **Upstream Stage**: [STAGE_2.5_IMPL.md](STAGE_2.5_IMPL.md) (File Organization)
- **Downstream Stage**: [STAGE_2.7_IMPL.md](STAGE_2.7_IMPL.md) (Classification)
- **Prompt Critique**: [`2.6HashtagCritique.md`](../../documentation_migration/2.6HashtagCritique.md)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-28
**Maintainer**: Update when Stage 2.6 implementation changes
**Source**: Systematic code analysis of `ml_pipeline/stage2_content_analysis/` (10 files, 4491 lines)
