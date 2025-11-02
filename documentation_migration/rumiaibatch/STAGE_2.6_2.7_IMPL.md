# Stage 2.6 & 2.7: Content Analysis - Implementation Guide

**Purpose**: Pattern discovery and video classification using LLM (Claude Sonnet + Haiku)
**Target Audience**: LLM agents fixing bugs, adding features, or modifying content analysis stages
**Related**: [PRODUCTION_FLOW.md Stage 2.6 & 2.7 Contracts](../../PRODUCTION_FLOW.md#stage-26-content-discovery)

**Source**: 100% systematic code reading of 10 modules (4136 production lines)

---

## Quick Reference

### Stage 2.6: Content Discovery
- **Entry Point**: `ml_pipeline/stage2_content_analysis/discovery.py::run_discovery_stage()` (line 546)
- **Orchestrator Call**: `rumiai_ml_batch.py:862-936`
- **State Tracking**: `.content_analysis_state.json` (one-time execution flag)
- **Duration**: ~60-90s for 60 sample transcripts
- **Model**: Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **⚠️ CRITICAL**: **ONE-TIME EXECUTION** → Manual curation → **BLOCKS PIPELINE** (exit code 2)

### Stage 2.7: Content Classification
- **Entry Point**: `ml_pipeline/stage2_content_analysis/classification.py::run_classification_stage()` (line 1031)
- **Orchestrator Call**: `rumiai_ml_batch.py:937-1016`
- **Checkpoint**: `.checkpoints/classification_checkpoint.json` (thread-safe)
- **Duration**: ~5 min sequential, ~2 min parallel (120 videos)
- **Model**: Claude Haiku (`claude-3-haiku-20240307`)
- **Modes**: Sequential (default) or Parallel (env: `ENABLE_PARALLEL_CLASSIFICATION`)

### Module Structure
```
ml_pipeline/stage2_content_analysis/
├── __init__.py (79 lines)           # Public API exports
├── discovery.py (642 lines)         # Stage 2.6 main
├── classification.py (1662 lines)   # Stage 2.7 main (dual-flow)
├── validation.py (365 lines)        # Input/output validation
├── transcript_validation.py (491)   # Stage 2.5.1 (filter music/noise)
├── taxonomy_validation.py (136)     # Curated taxonomy validation
├── checkpoint.py (161 lines)        # Checkpoint/resume logic
├── cost_tracking.py (134 lines)     # API cost logging
├── error_handlers.py (195 lines)    # Retry & error handling
└── utils.py (271 lines)             # JSON, logging, text extraction
```

---

## Table of Contents

1. [Part A: Stage 2.6 Discovery](#part-a-stage-26-discovery)
2. [Part B: Stage 2.7 Classification](#part-b-stage-27-classification)
3. [Shared Modules Reference](#shared-modules-reference)
4. [Data Flow & Architecture](#data-flow--architecture)
5. [Error Handling Matrix](#error-handling-matrix)
6. [Debugging Guide](#debugging-guide)

---

# Part A: Stage 2.6 Discovery

## Overview

**Purpose**: Discover 7 content pattern categories from 60 sample transcripts using Claude Sonnet
**Output**: Raw taxonomy → Manual curation → Curated taxonomy for Stage 2.7

## Input Contract

### Prerequisites
- **Stage 2.5** complete → `selection_manifest.json`
- **Stage 2.5.1** complete → `transcript_validation_cache.json`

### Input Files
```
{analysis_base}/
├── selection_manifest.json          # Video IDs per bucket (Stage 2.5)
└── content_taxonomies/
    └── transcript_validation_cache.json  # Valid/invalid flags (Stage 2.5.1)

/home/jorge/rumiaifinal/speech_transcriptions/
└── {video_id}_whisper.json          # Whisper transcripts (Stage 2)
```

### Validation
**File**: `validation.py::validate_discovery_inputs()` (line 18-83)

```python
# Checks performed:
1. Manifest exists
2. Manifest has required fields: ['hashtag', 'selected_buckets', 'videos_by_bucket']
3. 3 selected buckets present
4. Each bucket has ≥10 top performers
5. Sample size 10-200
6. ANTHROPIC_API_KEY environment variable set
```

## Output Contract

### Files Created
```
{analysis_base}/content_taxonomies/
├── {hashtag}_raw_discovery.json     # LLM output (7 categories)
└── {hashtag}_taxonomy.json          # 🔴 MANUAL CURATION REQUIRED

{analysis_base}/
└── .content_analysis_state.json     # One-time execution state
```

### Output Schema

**{hashtag}_raw_discovery.json**:
```json
{
  "hashtag": "#nutrition",
  "analysis_date": "2025-01-28T10:30:00Z",
  "sample_size": 60,
  "discovered_patterns": {
    "content_categories": [
      {
        "name": "recipe_tutorial",
        "frequency": 24,
        "percentage": 40.0,
        "examples": ["step by step", "here's how"],
        "representative_video_ids": ["video1", "video2"]
      }
    ],
    "hook_strategies": [
      {
        "name": "question_hook",
        "frequency": 18,
        "percentage": 30.0,
        "examples": ["did you know", "have you ever"],
        "representative_video_ids": ["video3", "video4"]
      }
    ],
    "closing_strategies": [
      {"name": "direct_cta", "frequency": 32, "percentage": 53.3, ...}
    ],
    "audience_pain_points": ["bloating", "low energy"],
    "trending_keywords": ["protein intake", "gut health"],
    "engagement_drivers": ["before after reveal"],
    "content_tactics": ["direct to camera", "voiceover"]
  }
}
```

**Validation**: `validation.py::validate_discovery_output()` (line 246-292)

## Core Functions

### 1. run_discovery_stage()
**File**: `discovery.py:546-642`
**Purpose**: Main entry point for Stage 2.6

```python
def run_discovery_stage(
    client_id: str,
    hashtag: str,
    analysis_type: str,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive",
    sample_size: int = 60,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
```

**Flow**:
1. Validate inputs (`validate_discovery_inputs()`)
2. Load validation cache from Stage 2.5.1 (`load_validation_cache()`)
3. Sample transcripts adaptively (`sample_transcripts_for_discovery()`)
4. Run LLM discovery (`discover_patterns_llm()`)
5. Calculate percentages (`calculate_percentages()`)
6. Save raw taxonomy

**Returns**: Raw taxonomy dict

### 2. sample_transcripts_for_discovery()
**File**: `discovery.py:26-226`
**Purpose**: Adaptive sampling - 20 per bucket, compensates for weak buckets

**Algorithm**:
```python
# STEP 1: Assess valid counts per bucket
for bucket in top_3_buckets:
    valid_ids = [vid for vid in top_performers
                 if validation_cache[vid]['is_valid']]
    bucket_valid_counts[bucket] = len(valid_ids)

# STEP 2: Determine sampling plan
for bucket in top_3_buckets:
    if bucket_valid_counts[bucket] < 20:
        sampling_plan[bucket] = bucket_valid_counts[bucket]  # Take all
        shortfall += (20 - bucket_valid_counts[bucket])
    else:
        sampling_plan[bucket] = 20
        surplus_buckets.append(bucket)

# STEP 3: Distribute shortfall to surplus buckets
if shortfall > 0 and surplus_buckets:
    extra_per_surplus = shortfall // len(surplus_buckets)
    for bucket in surplus_buckets:
        sampling_plan[bucket] += extra_per_surplus
```

**Edge Cases**:
- Scenario 1 (healthy): 46, 48, 44 valid → Sample 20, 20, 20 = 60 ✅
- Scenario 2 (one weak): 12, 50, 48 valid → Sample 12, 24, 24 = 60 ✅
- Scenario 3 (insufficient): 8, 12, 18 valid → Sample 8, 12, 18 = 38, raises ValueError if <10

**Minimum Threshold**: <10 total sampled → `ValueError` (line 216)

### 3. discover_patterns_llm()
**File**: `discovery.py:229-508`
**Purpose**: Extract 7 taxonomy categories using Claude Sonnet

**API Configuration**:
```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",  # Upgraded 2025-01-17
    max_tokens=4096,
    temperature=0.3,                     # Lower = consistent
    timeout=120,                         # 2-minute timeout
    system=system_message,
    messages=[{"role": "user", "content": prompt}]
)
```

**Retry Logic** (lines 420-507):
```python
for attempt in range(3):  # 3 attempts total
    try:
        response = client.messages.create(...)
        raw_taxonomy = parse_llm_json(response_text)
        validate_discovery_output(raw_taxonomy)
        return calculate_percentages(raw_taxonomy, sample_size)

    except TimeoutError:
        if attempt < 2:
            delay = [1, 2, 4][attempt]  # Exponential backoff
            time.sleep(delay)
        else:
            raise TimeoutError("Failed after 3 retries")

    except json.JSONDecodeError:
        if attempt < 2:
            delay = [1, 2, 4][attempt]
            time.sleep(delay)
        else:
            raise ValueError("Invalid JSON after 3 retries")
```

**Cost**: ~$0.15-0.30 per discovery (Sonnet pricing, 8K max tokens)

### 4. calculate_percentages()
**File**: `discovery.py:510-544`
**Purpose**: Add percentage fields to patterns post-LLM

```python
for category in ['content_categories', 'hook_strategies']:
    for pattern in raw_taxonomy['discovered_patterns'][category]:
        frequency = pattern['frequency']

        # Validate frequency ≤ sample_size
        if frequency > sample_size:
            raise ValueError(f"Frequency {frequency} exceeds sample_size {sample_size}")

        pattern['percentage'] = round((frequency / sample_size) * 100, 1)
```

## One-Time Execution & Manual Curation

### State Management
**File**: `rumiai_ml_batch.py:862-936` (orchestrator logic)

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
if state_file.exists():
    state = load_json(state_file)

    if state["discovery_complete"] and not state["taxonomy_curated"]:
        # BLOCK PIPELINE - waiting for manual curation
        logger.info("🔴 Stage 2.6 complete, edit {hashtag}_taxonomy.json")
        logger.info("   Set taxonomy_curated=true when done")
        sys.exit(2)  # Exit code 2: Paused

    if state["taxonomy_curated"]:
        logger.info("✅ Taxonomy curated, skipping discovery")
        continue  # Proceed to Stage 2.7
else:
    # First run - execute discovery
    run_discovery_stage(...)
    save_json(state_file, {"discovery_complete": true, "taxonomy_curated": false})
    sys.exit(2)  # BLOCK until manual curation
```

### Manual Curation Steps
1. Review `{hashtag}_raw_discovery.json`
2. Edit categories in `{hashtag}_taxonomy.json`:
   - Merge duplicate categories
   - Remove low-frequency noise (<5%)
   - Clarify definitions for creators
   - Ensure snake_case naming
3. Validate taxonomy: `taxonomy_validation.py::validate_curated_taxonomy()`
4. Set `taxonomy_curated=true` in `.content_analysis_state.json`
5. Re-run pipeline to continue with Stage 2.7

**Typical Duration**: ~15 minutes

---

# Part B: Stage 2.7 Classification

## Overview

**Purpose**: Classify 120 videos using curated taxonomy + Claude Haiku
**Architecture**: **Dual-Flow** (Flow 1: valid transcript, Flow 2: caption-only)
**Output**: 15-field classification JSON per video

## Input Contract

### Prerequisites
- **Stage 2.6** complete → `{hashtag}_taxonomy.json` (manually curated)
- **Stage 2.5** complete → `selection_manifest.json`
- **Stage 2.5.1** complete → `transcript_validation_cache.json`

### Input Files
```
{analysis_base}/
├── content_taxonomies/
│   ├── {hashtag}_taxonomy.json              # Curated taxonomy
│   └── transcript_validation_cache.json     # Valid/invalid flags
└── selection_manifest.json                  # Video IDs

/home/jorge/rumiaifinal/speech_transcriptions/
└── {video_id}_whisper.json                  # Whisper transcripts

{analysis_base}/buckets/bucket_{name}/
└── selected_videos.json                     # Caption/hashtag metadata
```

### Validation
**File**: `validation.py::validate_classification_inputs()` (line 88-164)

```python
# Checks performed:
1. Taxonomy file exists
2. Taxonomy has 7 required fields
3. All taxonomy fields non-empty
4. Semantic categories have name + definition (≥10 chars)
5. Manifest exists
```

## Output Contract

### Files Created
```
{analysis_base}/content_analysis/
├── raw_llm_output/
│   └── bucket_{name}/
│       └── {video_id}_raw.json              # LLM output (before normalization)
└── validated/
    └── bucket_{name}/
        └── {video_id}_content.json          # Final 15-field output
```

### Output Schema (15 fields)

**{video_id}_content.json**:
```json
{
  "video_id": "7545713916584774968",
  "bucket": "18-33s",                        // Added by Python
  "performer_type": "top",                   // Added by Python ("top" or "bottom")
  "taxonomy_version": "stage2.6_output",
  "content_category": "recipe_tutorial",
  "hook_strategy": "question_hook",
  "closing_strategy": "direct_cta",
  "pain_points": ["bloating", "low energy"],
  "keywords": ["protein intake"],
  "engagement_drivers": ["before_after_reveal"],
  "content_tactics": ["direct_to_camera"],
  "caption_analysis": {
    "hook_type": "question",
    "cta_type": "link_in_bio",
    "brand_mention_present": false,
    "influencer_tag_present": false,
    "emoji_usage": "some",
    "caption_length": "short",
    "hashtag_count": 7,                      // Calculated by Python from caption
    "hashtag_placement": "end"
  },
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

**Validation**: `validation.py::validate_classification_output()` (line 298-366)

## Dual-Flow Architecture

### Flow Selection Logic
**File**: `classification.py::classify_single_video_with_save()` (line 598-730)

```python
if manifest and validation_cache:
    # DUAL-FLOW MODE
    bucket, performer_type = get_bucket_for_video(video_id, manifest)
    is_valid = validation_cache[video_id]['is_valid']

    if is_valid:
        # Flow 1: Full classification (transcript + caption)
        llm_output = classify_video_with_transcript(...)
        classification = normalize_classification_schema(
            llm_output, video_id, caption,
            transcript_available=True,
            flow_type="full"
        )
    else:
        # Flow 2: Caption analysis only (no valid transcript)
        llm_output = classify_caption_only(...)
        classification = normalize_classification_schema(
            llm_output, video_id, caption,
            transcript_available=False,
            flow_type="caption_only"
        )

    # Add bucket metadata
    classification['bucket'] = bucket
    classification['performer_type'] = performer_type

    # Save validated output
    save_json(f"{output_dir}/validated/bucket_{bucket}/{video_id}_content.json", classification)
else:
    # LEGACY MODE: Single-flow (backward compatible)
    classification = classify_video_llm(...)
    save_json(f"{output_dir}/{video_id}_content.json", classification)
```

### Flow 1: Full Classification (Valid Transcript)
**File**: `classification.py::classify_video_with_transcript()` (line 1236-1558)

**Model**: Claude Haiku
**Prompt**: Physical zone separation (Zone 1: transcript only, Zone 2: caption only)
**Output**: 13 fields (Python adds hashtag_count)

**Key Features**:
- Zone 1 uses ONLY transcript for content classification
- Zone 2 uses ONLY caption for caption analysis
- Prevents LLM from mixing evidence sources
- Explicit examples of valid/invalid classifications

### Flow 2: Caption-Only Classification (Invalid Transcript)
**File**: `classification.py::classify_caption_only()` (line 1560-1662)

**Model**: Claude Haiku
**Prompt**: Caption analysis only (2 fields: hook_type, cta_type)
**Output**: `caption_analysis` object only (Python fills defaults)

**Schema Normalization**:
```python
# Flow 2 normalized output (line 1214-1232)
{
    "video_id": video_id,
    "taxonomy_version": "none_no_transcript",
    "content_category": None,
    "hook_strategy": None,
    "closing_strategy": None,
    "pain_points": [],
    "keywords": [],
    "engagement_drivers": [],
    "content_tactics": [],
    "caption_analysis": llm_output["caption_analysis"],  # Only 2 fields from LLM
    "confidence": "n/a",
    "transcript_available": False,
    "note": "No valid transcript - caption analysis only"
}
```

## Core Functions

### 1. run_classification_stage()
**File**: `classification.py:1031-1148`
**Purpose**: Main entry point for Stage 2.7

```python
def run_classification_stage(
    client_id: str,
    hashtag: str,
    analysis_type: str,
    analysis_mode: str = "top",
    selection_strategy: str = "contrastive",
    parallel: bool = None,              # Reads ENABLE_PARALLEL_CLASSIFICATION env
    max_workers: int = 5,               # Reads MAX_CLASSIFICATION_WORKERS env
    checkpoint_enabled: bool = True
) -> Dict[str, Any]:
```

**Flow**:
1. Read environment variables (parallel, max_workers)
2. Validate inputs (`validate_classification_inputs()`)
3. Load taxonomy and manifest
4. Load validation cache (enables dual-flow)
5. Initialize Anthropic client
6. Classify all videos (`classify_all_videos()`)

**Returns**: Summary dict with `{mode, total, completed, failed, duration_seconds}`

### 2. classify_all_videos()
**File**: `classification.py:922-1029`
**Purpose**: Orchestrate parallel or sequential classification

```python
# Build caption/hashtag cache (O(1) lookups)
video_data_cache = build_video_data_cache(target_dir)

# Set up checkpoint
if checkpoint_enabled:
    checkpoint_path = f"{target_dir}/.checkpoints/classification_checkpoint.json"

# Execute classification
if parallel:
    results = classify_all_videos_parallel(
        videos, taxonomy, client, output_dir, max_workers, checkpoint_path,
        manifest, validation_cache, video_data_cache
    )
else:
    results = classify_all_videos_sequential(
        videos, taxonomy, client, output_dir, checkpoint_path,
        manifest, validation_cache, video_data_cache
    )
```

### 3. classify_all_videos_sequential()
**File**: `classification.py:733-818`
**Purpose**: Sequential classification with checkpoint/resume

**Features**:
- Loads checkpoint, filters completed videos
- Processes remaining videos one-by-one
- Updates checkpoint after each video (completed or failed)
- Logs extraction stats (cleaned responses vs clean responses)

**Performance**: ~5 minutes for 120 videos

### 4. classify_all_videos_parallel()
**File**: `classification.py:820-920`
**Purpose**: Parallel classification with thread-safe checkpoint

**Features**:
- Uses `ThreadPoolExecutor` with configurable workers
- Thread-safe checkpoint updates (uses `threading.Lock()`)
- Submits all tasks, processes as they complete
- Same checkpoint/resume logic as sequential

**Performance**: ~2 minutes for 120 videos (5 workers)

### 5. build_video_data_cache()
**File**: `classification.py:486-541`
**Purpose**: Build hash map of caption/hashtag data for O(1) lookups

```python
video_data_map = {}

# Load winner_analysis.json to get winning buckets
winner_analysis = load_json(f"{target_dir}/winner_analysis.json")
winning_buckets = winner_analysis['top_3_buckets']

# Load caption/hashtag data from each bucket's selected_videos.json
for bucket_name in winning_buckets:
    selected_videos = load_json(f"{target_dir}/buckets/bucket_{bucket_name}/selected_videos.json")

    for video in selected_videos['videos']:
        video_id = video['id']
        video_data_map[video_id] = {
            'caption': video.get('text', ''),
            'hashtags': video.get('hashtags', []),
            'text_language': video.get('textLanguage', 'unknown')
        }

# Returns: {video_id: {caption, hashtags, text_language}}
```

**Why**: Replaces Stage 2.6.5 extraction (simpler, faster)

### 6. extract_json()
**File**: `classification.py:32-135`
**Purpose**: Extract JSON from LLM response with markdown/extra content

**Algorithm**:
```python
# Step 1: Strip markdown code fences
if text.startswith('```'):
    text = text[3:].lstrip()
    if text.startswith('json'):
        text = text[4:].lstrip()
if text.endswith('```'):
    text = text[:-3].rstrip()

# Step 2: Find first complete JSON object using brace counting
first_brace = text.find('{')
brace_count = 0
for i in range(first_brace, len(text)):
    if text[i] == '{':
        brace_count += 1
    elif text[i] == '}':
        brace_count -= 1
        if brace_count == 0:
            last_brace = i
            break

# Step 3: Extract first complete JSON
json_text = text[first_brace:last_brace + 1]
return json.loads(json_text)
```

**Handles**:
- Markdown code fences (```json ... ```)
- Extra text before/after JSON
- Multiple JSON objects (extracts first complete one)

**Tracking**: Logs extraction stats (_extraction_stats module variable)

---

# Shared Modules Reference

## 1. checkpoint.py (161 lines)

### Purpose
Thread-safe checkpoint/resume for Stage 2.7 classification

### Key Functions

**load_checkpoint()** (line 18-71)
```python
def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    # Returns empty checkpoint if file doesn't exist
    if not os.path.exists(checkpoint_path):
        return {
            "completed": [],
            "failed": [],
            "last_updated": None,
            "stats": {"completed_count": 0, "failed_count": 0}
        }

    # Load and validate structure
    checkpoint = load_json(checkpoint_path)
    if missing := [f for f in ["completed", "failed"] if f not in checkpoint]:
        raise ValueError(f"Checkpoint missing fields: {missing}")

    return checkpoint
```

**save_checkpoint()** (line 73-119)
```python
def save_checkpoint(checkpoint_path: str, checkpoint: Dict[str, Any]):
    # Update timestamp and stats
    checkpoint["last_updated"] = datetime.utcnow().isoformat() + "Z"
    checkpoint["stats"] = {
        "completed_count": len(checkpoint["completed"]),
        "failed_count": len(checkpoint["failed"])
    }

    # Atomic write (temp file + rename)
    temp_path = checkpoint_path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(temp_path, checkpoint_path)  # Atomic
```

**update_checkpoint()** (line 121-161)
```python
def update_checkpoint(checkpoint, video_id, status, checkpoint_path):
    # Status: "completed" or "failed"
    if status == "completed":
        if video_id not in checkpoint["completed"]:
            checkpoint["completed"].append(video_id)
        if video_id in checkpoint["failed"]:
            checkpoint["failed"].remove(video_id)  # Retry success
    else:
        if video_id not in checkpoint["failed"]:
            checkpoint["failed"].append(video_id)

    save_checkpoint(checkpoint_path, checkpoint)
```

---

## 2. cost_tracking.py (134 lines)

### Purpose
Log estimated and actual API costs for transparency

### Key Functions

**log_estimated_cost()** (line 15-63)
```python
def log_estimated_cost(operation: str, video_count=None, sample_size=None) -> float:
    if operation == "discovery":
        estimated_cost = 0.75  # Sonnet, ~50 transcripts
        details = f"Sonnet API call, ~{sample_size or 50} transcripts"

    elif operation == "classification":
        cost_per_video = 0.001  # Haiku, $0.001 per video
        estimated_cost = cost_per_video * (video_count or 120)
        details = f"Haiku API calls, {video_count or 120} videos"

    logger.info(f"💰 Estimated cost for {operation}: ${estimated_cost:.2f} ({details})")
    return estimated_cost
```

**log_actual_cost()** (line 66-134)
```python
def log_actual_cost(response: anthropic.types.Message, model: str, start_time=None) -> float:
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # Pricing as of 2025-01
    pricing = {
        "sonnet": {"input": 15 / 1_000_000, "output": 75 / 1_000_000},
        "haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000}
    }

    cost = (input_tokens * pricing[model]["input"] +
            output_tokens * pricing[model]["output"])

    if start_time:
        latency = time.time() - start_time
        logger.debug(f"💸 API call: ${cost:.4f}, {latency:.2f}s, "
                    f"in: {input_tokens:,}, out: {output_tokens:,}, model: {model}")

    return cost
```

---

## 3. error_handlers.py (195 lines)

### Purpose
Centralized error handling with retry logic

### Key Functions

**handle_missing_input_file()** (line 18-35)
```python
def handle_missing_input_file(file_path: str, stage_name: str):
    raise FileNotFoundError(
        f"❌ Required input not found: {file_path}\n"
        f"This file should have been created by {stage_name}.\n"
        f"Action: Verify {stage_name} completed successfully."
    )
```

**handle_api_timeout_with_retry()** (line 42-85)
```python
def handle_api_timeout_with_retry(
    api_call_func: Callable,
    context: str,
    max_retries: int = 3,
    backoff_delays: list = None
) -> Any:
    if backoff_delays is None:
        backoff_delays = [1, 2, 4]  # Exponential backoff

    for attempt in range(max_retries):
        try:
            return api_call_func()
        except TimeoutError as e:
            if attempt < max_retries - 1:
                delay = backoff_delays[attempt]
                logger.warning(f"⏰ {context} timeout. Retry {attempt + 1}/{max_retries} in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"❌ {context} failed after {max_retries} retries")
                raise
```

**handle_graceful_skip()** (line 127-142)
```python
def handle_graceful_skip(video_id: str, reason: str, error_type: str = "warning"):
    if error_type == "warning":
        logger.warning(f"⚠️  Skipping video {video_id}: {reason}")
    else:
        logger.info(f"ℹ️  Skipping video {video_id}: {reason}")
```

---

## 4. validation.py (365 lines)

### Purpose
Input/output validation for Stages 2.6 and 2.7

### Key Functions (covered in Stage 2.6/2.7 sections above)

---

## 5. transcript_validation.py (491 lines)

### Purpose
**Stage 2.5.1**: Filter music/noise transcripts before discovery

### Key Functions

**is_valid_transcript()** (line 48-142)
```python
def is_valid_transcript(text: str, config: TranscriptFilterConfig = None) -> Tuple[bool, Optional[str]]:
    # Check 0: Simple music filter
    if any(marker in text for marker in ['[Music]', '[MUSIC]', '(upbeat music)', '♪']):
        return False, "contains_music_markers (simple filter)"

    # Check 1: Length
    if len(text) < config.min_length:
        return False, f"too_short ({len(text)} < {config.min_length})"
    if len(text) > config.max_length:
        return False, f"too_long ({len(text)} > {config.max_length})"

    # Check 2: Remove sound markers
    text_no_markers = re.sub(SOUND_MARKERS_PATTERN, '', text, flags=re.IGNORECASE)
    if len(text_no_markers) < config.min_length:
        return False, f"music_only ({len(text_no_markers)} chars after removal)"

    # Check 3: Foreign language (optional)
    # Check 4: Word count
    words = text_no_markers.split()
    if len(words) < config.min_words:
        return False, f"too_few_words ({len(words)} < {config.min_words})"

    # Check 5: Repetitiveness (for longer transcripts)
    if len(words) >= 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < config.min_unique_ratio:
            return False, f"too_repetitive (unique_ratio={unique_ratio:.2f})"

    return True, None
```

**validate_all_transcripts()** (line 145-346)
```python
def validate_all_transcripts(...) -> str:
    # Incremental validation (checks cache version)
    if os.path.exists(cache_path):
        existing_cache = load_json(cache_path)
        if existing_cache['version'] == VALIDATION_CACHE_VERSION:
            cached_ids = set(existing_cache['results'].keys())
            new_ids = set(all_video_ids) - cached_ids
            if not new_ids:
                return cache_path  # Cache complete, skip
            videos_to_validate = list(new_ids)  # Incremental

    # Validate transcripts
    for video_id in videos_to_validate:
        transcript = load_json(f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json")
        is_valid, reason = is_valid_transcript(transcript['text'], config)
        validation_results[video_id] = {
            'is_valid': is_valid,
            'failure_reason': reason,
            'text_length': len(transcript['text']),
            'word_count': len(transcript['text'].split())
        }

    # Save cache
    cache_data = {
        'version': VALIDATION_CACHE_VERSION,
        'hashtag': hashtag,
        'validation_date': datetime.utcnow().isoformat() + "Z",
        'config': asdict(config),
        'stats': {'total': len(all_video_ids), 'valid': ..., 'invalid': ...},
        'results': validation_results
    }
    save_json(cache_path, cache_data)

    return cache_path
```

**run_transcript_validation_stage()** (line 409-491)
- Entry point for Stage 2.5.1
- Enforces minimum 30 valid transcripts
- Returns summary with invalid breakdown

---

## 6. taxonomy_validation.py (136 lines)

### Purpose
Validate manually curated taxonomy before Stage 2.7

### Key Function

**validate_curated_taxonomy()** (line 18-136)
```python
def validate_curated_taxonomy(taxonomy_path: str) -> bool:
    taxonomy = load_json(taxonomy_path)

    # Check all 6 required fields present
    required = ['content_categories', 'hook_strategies', 'audience_pain_points',
                'trending_keywords', 'engagement_drivers', 'content_tactics']
    if missing := [f for f in required if f not in taxonomy]:
        raise ValueError(f"Missing fields: {missing}")

    # Validate semantic categories (content_categories, hook_strategies)
    for category_type in ['content_categories', 'hook_strategies']:
        for cat in taxonomy[category_type]:
            # Check has name + definition
            if 'name' not in cat or 'definition' not in cat:
                raise ValueError(f"{category_type} missing name or definition")

            # Check snake_case
            if not re.match(r'^[a-z0-9_]+$', cat['name']):
                raise ValueError(f"Name must be snake_case: {cat['name']}")

            # Check definition length
            if len(cat['definition']) < 10:
                raise ValueError(f"Definition too short: {cat['definition']}")

        # Check no duplicates
        names = [c['name'] for c in taxonomy[category_type]]
        if duplicates := [n for n in names if names.count(n) > 1]:
            raise ValueError(f"Duplicate names: {set(duplicates)}")

    # Validate simple list categories
    for category_type in ['audience_pain_points', 'trending_keywords',
                          'engagement_drivers', 'content_tactics']:
        items = taxonomy[category_type]
        if not items:
            raise ValueError(f"{category_type} cannot be empty")

        for item in items:
            if not isinstance(item, str):
                raise ValueError(f"{category_type} items must be strings")
            if len(item) < 2:
                raise ValueError(f"Item too short: {item}")

        # Check no duplicates
        if duplicates := [i for i in items if items.count(i) > 1]:
            raise ValueError(f"Duplicates: {set(duplicates)}")

    return True
```

---

## 7. utils.py (271 lines)

### Purpose
Shared utilities: JSON I/O, logging, text extraction

### Key Functions

**parse_llm_json()** (line 17-61)
```python
def parse_llm_json(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    return json.loads(text)
```

**load_json()** (line 63-95)
- Loads JSON with error handling
- Raises FileNotFoundError or json.JSONDecodeError

**save_json()** (line 97-138)
- Atomic write pattern (temp file + rename)
- Creates parent directories
- Prevents partial writes on crash

**extract_transcript_ending()** (line 175-231)
```python
def extract_transcript_ending(text: str, max_words: int = 10) -> str:
    # Normalize whitespace
    words = text.strip().split()

    if len(words) <= max_words:
        return " ".join(words)

    # Extract last N words
    ending_words = words[-max_words:]
    ending_text = " ".join(ending_words)

    # Strip trailing punctuation
    return ending_text.rstrip('.!?…').strip()
```

**extract_transcript_opening()** (line 233-272)
- Same as ending but for first N words
- Used for hook strategy analysis

---

# Data Flow & Architecture

## Complete Pipeline Flow

```
Stage 2.5 (File Organization)
    ↓
    selection_manifest.json
    ↓
Stage 2.5.1 (Transcript Validation) ← NEW
    ↓
    transcript_validation_cache.json (valid/invalid flags)
    ↓
Stage 2.6 (Discovery)
    ├─→ Load validation cache
    ├─→ Sample 60 valid transcripts (adaptive)
    ├─→ Claude Sonnet: discover 7 categories
    └─→ {hashtag}_raw_discovery.json
    ↓
**MANUAL CURATION** (~15 min)
    ↓
    {hashtag}_taxonomy.json
    ↓
Stage 2.7 (Classification)
    ├─→ Load taxonomy + validation cache
    ├─→ Build video data cache (caption/hashtags)
    ├─→ For each video:
    │   ├─→ Check validation_cache[video_id]['is_valid']
    │   ├─→ Flow 1 (valid): classify_video_with_transcript()
    │   └─→ Flow 2 (invalid): classify_caption_only()
    ├─→ Normalize schema (add hashtag_count, bucket, performer_type)
    └─→ Save validated/{video_id}_content.json
```

## File Dependency Graph

```
selection_manifest.json (Stage 2.5)
    ├─→ Stage 2.5.1 (validate transcripts)
    ├─→ Stage 2.6 (sampling)
    └─→ Stage 2.7 (video list + bucket metadata)

transcript_validation_cache.json (Stage 2.5.1)
    ├─→ Stage 2.6 (filter invalid transcripts)
    └─→ Stage 2.7 (dual-flow routing)

{hashtag}_taxonomy.json (Stage 2.6 manual)
    └─→ Stage 2.7 (classification categories)

selected_videos.json (Stage 1, per bucket)
    └─→ Stage 2.7 (caption/hashtag cache)
```

---

# Error Handling Matrix

## Stage 2.6 Errors

| Error Type | Cause | Handled By | Action | Exit Code |
|------------|-------|------------|--------|-----------|
| `FileNotFoundError` | Manifest missing | Orchestrator | Exit pipeline | 1 |
| `FileNotFoundError` | Validation cache missing | Discovery | Exit pipeline | 1 |
| `ValueError` | <10 valid transcripts | Sampling | Exit pipeline | 1 |
| `ValueError` | Sample size out of range | Validation | Exit pipeline | 1 |
| `TimeoutError` | Sonnet timeout >120s | Discovery LLM | Retry 3x, then exit | 8 |
| `json.JSONDecodeError` | Invalid JSON from LLM | Discovery LLM | Retry 3x, then exit | 1 |
| `ValueError` | Invalid discovery schema | Output validation | Exit pipeline | 1 |
| `RuntimeError` | API authentication | Orchestrator | Exit pipeline | 99 |

## Stage 2.7 Errors

| Error Type | Cause | Handled By | Action | Exit Code |
|------------|-------|------------|--------|-----------|
| `FileNotFoundError` | Taxonomy missing | Validation | Exit pipeline | 1 |
| `ValueError` | Taxonomy invalid | Validation | Exit pipeline | 1 |
| `TimeoutError` | Haiku timeout >30s | Classification | Retry 3x, skip video | - |
| `json.JSONDecodeError` | Invalid JSON from LLM | Classification | Retry 3x, skip video | - |
| `ValueError` | Invalid classification schema | Output validation | Skip video | - |
| `Exception` | Any video-level error | Sequential/Parallel | Skip video, continue | - |
| `RuntimeError` | API authentication | Orchestrator | Exit pipeline | 99 |

## Retry Strategies

### Discovery (Stage 2.6)
```python
# 3 attempts, exponential backoff
for attempt in range(3):
    try:
        response = client.messages.create(...)
        return process_response(response)
    except (TimeoutError, json.JSONDecodeError):
        if attempt < 2:
            delay = [1, 2, 4][attempt]  # 1s, 2s, 4s
            time.sleep(delay)
        else:
            raise  # Exit pipeline
```

### Classification (Stage 2.7)
```python
# 3 attempts, exponential backoff, but SKIP on final failure
for attempt in range(3):
    try:
        response = client.messages.create(...)
        return process_response(response)
    except (TimeoutError, json.JSONDecodeError):
        if attempt < 2:
            delay = [1, 2, 4][attempt]
            time.sleep(delay)
        else:
            # Skip video, continue with next
            logger.error(f"Skipping {video_id} after 3 retries")
            return None
```

---

# Debugging Guide

## Stage 2.6 Troubleshooting

### Issue: Discovery fails with "Insufficient transcripts"
**Symptom**: `ValueError: <10 valid transcripts sampled`
**Cause**: Most transcripts are music/noise
**Debug**:
```bash
# Check validation cache
cat {analysis_base}/content_taxonomies/transcript_validation_cache.json | jq '.stats'

# Output shows:
{
  "total": 120,
  "valid": 8,
  "invalid": 112,
  "by_reason": {
    "music_only": 95,
    "too_short": 12,
    "too_repetitive": 5
  }
}
```
**Fix**: Select different hashtag with more spoken content

### Issue: Discovery timeout after 3 retries
**Symptom**: `TimeoutError: LLM timed out after 3 attempts`
**Cause**: Anthropic API slow or overloaded
**Debug**:
```bash
# Check API status
curl https://status.anthropic.com/

# Check logs for timeout duration
grep "Discovery timeout" logs/*.log
```
**Fix**: Retry later (API issue, not code issue)

### Issue: Invalid JSON from LLM
**Symptom**: `ValueError: LLM returned invalid JSON after 3 retries`
**Cause**: Sonnet output parsing error
**Debug**:
```bash
# Check logs for response preview
grep "Response text:" logs/*.log | head -1

# Common issues:
# - Extra text before/after JSON
# - Markdown fences not stripped
# - Malformed JSON structure
```
**Fix**: Check `parse_llm_json()` handles response format, retry (random variance may succeed)

## Stage 2.7 Troubleshooting

### Issue: Classification skipping many videos
**Symptom**: `120 total, 85 completed, 35 failed`
**Cause**: Various video-level errors
**Debug**:
```bash
# Check checkpoint for failed IDs
cat .checkpoints/classification_checkpoint.json | jq '.failed[]'

# Check logs for failure reasons
grep "Failed classification" logs/*.log

# Common reasons:
# - Transcript file missing
# - Caption data missing from cache
# - LLM timeout/invalid JSON
# - Schema validation failure
```
**Fix**: Address most common failure reason first

### Issue: Parallel mode not working
**Symptom**: Still running sequentially despite `ENABLE_PARALLEL_CLASSIFICATION=true`
**Cause**: Environment variable not read correctly
**Debug**:
```python
# Check environment variable
import os
print(os.environ.get('ENABLE_PARALLEL_CLASSIFICATION'))

# Check orchestrator reads it
# rumiai_ml_batch.py line 1071:
parallel = os.environ.get('ENABLE_PARALLEL_CLASSIFICATION', 'false').lower() == 'true'
```
**Fix**:
```bash
export ENABLE_PARALLEL_CLASSIFICATION=true
export MAX_CLASSIFICATION_WORKERS=5
python rumiai_ml_batch.py --client test --target "#nutrition"
```

### Issue: Hashtag count mismatch
**Symptom**: `caption_analysis.hashtag_count` doesn't match actual hashtags
**Cause**: M10 FIX implemented - Python calculates from caption, not LLM
**Debug**:
```python
# Check normalization (classification.py line 1207)
hashtag_count = len([word for word in caption.split() if word.startswith('#')])

# LLM no longer outputs hashtag_count field
# Python adds it during normalization
```
**Fix**: This is correct behavior (deterministic calculation)

## Checkpoint/Resume Testing

### Verify Checkpoint Works
```bash
# 1. Start classification (kill after 10 videos)
python rumiai_ml_batch.py --client test --target "#nutrition"
# Ctrl+C after 10 seconds

# 2. Check checkpoint
cat .checkpoints/classification_checkpoint.json | jq '.stats'
# Output: {"completed_count": 10, "failed_count": 0}

# 3. Resume (should skip completed 10)
python rumiai_ml_batch.py --client test --target "#nutrition"
# Log: "Resuming from checkpoint: 10 completed, 110 remaining"
```

### Force Re-run (Delete Checkpoint)
```bash
rm .checkpoints/classification_checkpoint.json
python rumiai_ml_batch.py --client test --target "#nutrition"
# Starts fresh
```

## Performance Profiling

### Discovery Timing
```bash
# Add timing logs
grep "Discovery" logs/*.log | grep -E "calling|received|complete"

# Expected:
# [10:30:00] Calling Claude Sonnet API for discovery (attempt 1/3)...
# [10:30:55] Received response from Claude Sonnet (8234 chars)  # ~55s
# [10:30:56] ✅ Discovery complete  # Total ~56s
```

### Classification Timing
```bash
# Sequential mode
time python rumiai_ml_batch.py --client test --target "#nutrition"
# Expected: ~5 minutes for 120 videos

# Parallel mode (5 workers)
export ENABLE_PARALLEL_CLASSIFICATION=true
time python rumiai_ml_batch.py --client test --target "#nutrition"
# Expected: ~2 minutes for 120 videos
```

---

## Quick Debugging Commands

```bash
# Check all validation cache stats
jq '.stats' content_taxonomies/transcript_validation_cache.json

# Check discovery output categories
jq '.discovered_patterns | keys' content_taxonomies/*_raw_discovery.json

# Check classification checkpoint
jq '.stats' .checkpoints/classification_checkpoint.json

# Count classified videos per bucket
find content_analysis/validated -name "*_content.json" | wc -l

# Check for missing captions in cache
jq '[.[] | select(.caption == "")] | length' video_data_cache.json

# Find videos with low confidence
jq -s '[.[] | select(.confidence == "low")] | length' content_analysis/validated/bucket_*/*_content.json

# Check extraction stats
grep "Extraction stats" logs/*.log
```

---

## Modification Guide

### Adding a New Taxonomy Category

**Scenario**: Add "visual_patterns" as 8th category to Stage 2.6

**Steps**:

1. **Update discovery prompt** (`discovery.py:262-408`)
   ```python
   prompt += """
   ## CATEGORY 8: Visual Patterns

   Identify VISUAL elements (camera angles, editing styles).
   Examples: close_up_hands, dynamic_transitions
   """
   ```

2. **Update output validation** (`validation.py:267-274`)
   ```python
   required_patterns = [
       'content_categories', 'hook_strategies', 'closing_strategies',
       'audience_pain_points', 'trending_keywords',
       'engagement_drivers', 'content_tactics',
       'visual_patterns'  # ADD
   ]
   ```

3. **Update taxonomy validation** (`taxonomy_validation.py:49-55`)
   ```python
   required_fields = [
       'content_categories', 'hook_strategies', 'audience_pain_points',
       'trending_keywords', 'engagement_drivers', 'content_tactics',
       'visual_patterns'  # ADD
   ]
   ```

4. **Update Stage 2.7 classification prompt** (`classification.py:176-340`)
   - Add visual_patterns to taxonomy section
   - Add field to output schema

5. **Test**:
   ```bash
   python -c "
   from ml_pipeline.stage2_content_analysis import run_discovery_stage
   result = run_discovery_stage('test', 'nutrition', 'hashtag')
   print('visual_patterns' in result['discovered_patterns'])
   "
   ```

### Changing Parallel Workers Default

**File**: `classification.py:1073`

**Before**:
```python
max_workers = int(os.environ.get('MAX_CLASSIFICATION_WORKERS', str(max_workers)))
# Default passed as argument: max_workers=5
```

**After**:
```python
max_workers = int(os.environ.get('MAX_CLASSIFICATION_WORKERS', '10'))  # Change default to 10
```

### Adding New Caption Analysis Field

**Scenario**: Add "video_length" field to caption_analysis

**Steps**:

1. **Update classification output validation** (`validation.py:340-348`)
   ```python
   caption_fields = [
       'hook_type', 'cta_type', 'brand_mention_present',
       'influencer_tag_present', 'emoji_usage', 'caption_length',
       'hashtag_count', 'hashtag_placement',
       'video_length'  # ADD
   ]
   ```

2. **Update classification prompt** (`classification.py:286-318`)
   - Add video_length instructions
   - Add to output format

3. **Update normalization** (`classification.py:1184-1234`)
   ```python
   normalized['caption_analysis']['video_length'] = calculate_video_length(...)
   ```

---

## Related Documentation

- **PRODUCTION_FLOW.md**: [Stage 2.6 Contract](../../PRODUCTION_FLOW.md#stage-26-content-discovery), [Stage 2.7 Contract](../../PRODUCTION_FLOW.md#stage-27-content-classification)
- **Technical Specs**:
  - [`ContentAnalysisCHILDTI.md`](../../documentation_migration/ContentAnalysisCHILDTI.md) - Original specification
- **Upstream Stages**:
  - [STAGE_2.5_IMPL.md](STAGE_2.5_IMPL.md) - File Organization
- **Downstream Stages**:
  - [STAGE_7_IMPL.md](STAGE_7_IMPL.md) - LLM Analysis (consumes classifications)
  - [STAGE_8_IMPL.md](STAGE_8_IMPL.md) - Report Generation (consumes classifications)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-28
**Source**: 100% systematic code reading (4136 production lines across 10 modules)
**Maintainer**: Update when Stage 2.6 or 2.7 implementation changes
