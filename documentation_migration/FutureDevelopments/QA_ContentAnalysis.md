# Clarification Q&A: ContentAnalysis

> **Mother Doc**: ContentAnalysis.md (Stages 2.6 & 2.7)
> **Phase 1**: Critique_ContentAnalysis.md
> **Date**: 2025-10-14
> **Status**: IN PROGRESS

## Questions by Category

### Input/Output Contracts

#### Q1: [CRITICAL] Input Schema - selection_manifest.json
ContentAnalysis.md lines 724-752 show a `selection_manifest.json` structure with fields like `hashtag`, `selected_buckets`, `videos_by_bucket`, etc.

**Question**: Is this the exact production schema for `selection_manifest.json` that Stage 2.5 will produce? Or is this a proposed design that needs to be implemented?

**Answer**: "It is not being created in stage 2.5 MLPlanningv2.md, It would have to be created by Content Analysis"

**For HLD Section**: 5.1 (Input Schema), 3.1 (Input Dependencies), 2.3 (Detailed Process)

**Notes**: This reveals a dependency gap. Content Analysis (Stage 2.6) needs to know which videos Stage 2.5 selected, but Stage 2.5 doesn't currently output a manifest. This means either:
- Option A: Stage 2.6 must read the directory structure created by Stage 2.5 to discover which videos were selected
- Option B: Stage 2.5 needs to be enhanced to output a manifest (requires changes to MLPlanningv2.md)
- Option C: Content Analysis discovers videos independently using same selection logic

Need clarification on which approach to use.

### Dependencies & Integration

#### Q2: [CRITICAL] How does Content Analysis discover which videos were selected?
Based on Q1, Stage 2.5 doesn't output a manifest. Content Analysis needs a way to discover which videos were selected by Stage 2.5.

**Question**: Which approach should Content Analysis use?
- Option A: Read Stage 2.5 directory structure (glob ml_training_data/)
- Option B: Enhance Stage 2.5 to output manifest (requires Stage 2.5 changes)
- Option C: Content Analysis runs its own selection logic (duplicate logic)

**Answer**: "Option B"

**For HLD Section**: 2.2 (Data Flow), 3.1 (Input Dependencies), 3.3 (Cross-Stage Dependencies)

**Notes**: Stage 2.5 will be enhanced to output `selection_manifest.json`. This creates a clean inter-stage contract. Content Analysis HLD must:
- Document the manifest schema as a required input
- Note this is a new Stage 2.5 enhancement (dependency on MLPlanningv2.md changes)
- Include validation that manifest exists before Stage 2.6 runs
- Reference this as a cross-stage dependency in Section 3.3

#### Q3: [CRITICAL] Transcript Input Schema
ContentAnalysis.md lines 399-439 describe transcript files in `/speech_transcriptions/` with structure showing `text`, `segments`, `words` fields.

**Question**: Is this the exact schema that Whisper outputs in production?
1. File naming: Is it always `{video_id}_whisper.json` or variations?
2. Required fields: Are `text`, `segments`, `words` always present or can be null/missing?
3. Text field: Does `text` contain complete transcript or concatenate `segments[].text`?
4. Missing transcripts: What if video has no speech? Empty file or file doesn't exist?
5. File location: Is `/speech_transcriptions/` at repo root?

**Answer**:
1. "Its always that way" (always `{video_id}_whisper.json`)
2. "If there is speech, there will be text, it will always be present"
3. "Complete transcript"
4. "Text outputs blank" (file exists, text is empty string)
5. "Its at the repo root" (`/home/jorge/rumiaifinal/speech_transcriptions/`)

**For HLD Section**: 5.1 (Input Schema), 6.1 (Input Validation), 6.2 (Error Cases)

**Notes**: Input validation must handle:
- Empty `text` field (videos with no speech) - should this skip content analysis or pass empty string to LLM?
- File path construction: `/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json`
- No need to check for missing `text` field - it's always present per Whisper output contract

### Edge Cases & Validation

#### Q4: [CRITICAL] Empty Transcript Handling
Based on Q3, videos with no speech have `text: ""` (empty string).

**Question**: When a video has an empty transcript, what should Content Analysis do?
- Option A: Skip video entirely (don't create classification file)
- Option B: Classify using caption/hashtags only (pass empty transcript to LLM)
- Option C: Create minimal stub classification ("no_speech_detected")

**Answer**: "Option B"

**For HLD Section**: 6.2 (Error Cases), 2.3 (Detailed Process Logic), 5.2 (Output Schema)

**Notes**: Content Analysis must:
- Accept empty transcript strings (no validation error)
- Pass empty transcript to LLM along with caption/hashtags
- Add `transcript_available: false` flag to output schema
- Document in output schema that classifications can be caption/hashtag-only
- Stage 7 can query: "What % of top performers use no-speech strategy?"

Example output for no-speech video:
```json
{
  "video_id": "123",
  "transcript_available": false,
  "content_category": "visual_tutorial",
  "classification_confidence": "medium",
  "note": "Classified using caption and hashtags only"
}
```

#### Q5: [CRITICAL] Caption and Hashtag Input Schema
ContentAnalysis.md lines 443-531 describe using caption (`description`) and hashtag analysis from video metadata.

**Question**: Where exactly do caption and hashtags come from?
1. Source file: Are they in `/insights/{video_id}_temporal_windows_updated.json` or elsewhere?
2. Caption field: What's the exact field path?
3. Hashtag field: What's the exact field path and structure?
4. Missing data: Can description/hashtags be null/empty?

**Answer**:
- Caption and hashtags are NOT in `insights/` (temporal windows)
- They ARE in `/unified_analysis/{video_id}.json` (LOC 7 for description, LOC 16-72 for hashtags)
- If no description, field is empty string
- If no hashtags, field is empty array
- Decision: **Option A** - Content Analysis reads from `unified_analysis/` (no changes to temporal_compute)

**For HLD Section**: 5.1 (Input Schema), 3.1 (Input Dependencies)

**Notes**: Content Analysis input sources:
1. **Transcript**: `/speech_transcriptions/{video_id}_whisper.json` → `text` field
2. **Caption**: `/unified_analysis/{video_id}.json` → `metadata.description` field
3. **Hashtags**: `/unified_analysis/{video_id}.json` → `metadata.hashtags` array (extract `name` field from each object)

Hashtag extraction logic:
```python
hashtags_array = unified_analysis['metadata']['hashtags']
hashtag_names = [h['name'] for h in hashtags_array if h.get('name')]
# Result: ["yonisteam", "womenshealth", "mugwort", ...]
```

### Performance & Scale

[Questions will be filled iteratively]

#### Q6: [CRITICAL] Output Schema - Content Classification Structure
ContentAnalysis.md lines 1052-1059 show example output, and lines 349-394 show a detailed structure.

**Question**: What are ALL the required fields in the content classification output?

**Answer**: After analyzing full schema (349-394) and ranking by importance, decided on **middle-ground schema with observable features**:

**Core fields** (always present):
- `video_id`: String
- `content_category`: String (from taxonomy)
- `hook_strategy`: String (from taxonomy)
- `audience_pain_points`: Array of strings (from taxonomy)
- `trending_keywords`: Array of strings (from taxonomy)
- `engagement_drivers`: Array of strings - observable tactics like "before_after_reveal", "specific_metrics_mentioned"
- `content_tactics`: Array of strings - observable boolean features like "personal_story", "direct_to_camera", "vulnerability_shown" (NOT subjective ratings)
- `confidence`: String - categorical "high"/"medium"/"low"
- `transcript_available`: Boolean
- `note`: String or null

**For HLD Section**: 5.2 (Output Schema), 6.3 (Output Validation)

**Notes**: Replaced subjective "content_strategy" ratings with observable "content_tactics" for:
- Reliability (LLM can detect presence better than rate quality)
- Actionability (creators can replicate tactics)
- Validation (correlate tactics with actual performance data)

Schema locked in for TIER 1 features. Proceeded through TIER 2 & TIER 3 evaluation - skipped both.

**ADDENDUM - Caption Analysis**: After Q6, identified that captions need separate analysis (different structural patterns from transcript). Added caption-specific schema to output.

**Final Complete Output Schema:**
```json
{
  "video_id": "123456789",
  "content_category": "recipe_tutorial",
  "hook_strategy": "problem_solution",
  "audience_pain_points": ["bloating", "low_energy"],
  "trending_keywords": ["protein", "gut_health"],
  "engagement_drivers": [
    "before_after_reveal",
    "specific_metrics_mentioned",
    "relatable_struggle"
  ],
  "content_tactics": [
    "personal_story",
    "direct_to_camera",
    "specific_actionable_steps",
    "vulnerability_shown"
  ],
  "caption_analysis": {
    "caption_hook_type": "statement",
    "caption_cta_type": "link_in_bio",
    "caption_cta_present": true,
    "brand_mention_present": true,
    "influencer_tag_present": true,
    "emoji_usage": "moderate",
    "caption_length": "medium",
    "hashtag_count": 9,
    "hashtag_placement": "end",
    "hashtag_strategy": {
      "broad_count": 2,
      "niche_count": 5,
      "branded_count": 2
    }
  },
  "confidence": "high",
  "transcript_available": true,
  "note": null
}
```

All caption fields are observable and actionable. Hashtag strategy breakdown enables analysis of hashtag mix effectiveness.

### Error Handling

#### Q7: [CRITICAL] LLM API Configuration & Credentials
ContentAnalysis.md lines 960-1006 show code using Anthropic API.

**Question**: What's the exact LLM API configuration?
1. API Provider?
2. Model selection for Stage 2.6 vs Stage 2.7?
3. API key management?
4. Rate limits handling?
5. Retry strategy?
6. Fallback if LLM unavailable?

**Answer**:
1. **API Provider**: Anthropic Claude ✓
2. **Model Selection**:
   - Stage 2.6 (Discovery): Claude 3.5 Sonnet (~$0.75/hashtag one-time)
   - Stage 2.7 (Classification): Claude 3 Haiku (~$0.30/300 videos)
   - Rationale: Sonnet for complex reasoning (pattern discovery), Haiku for repetitive classification (15x cheaper)
   - Note: Can upgrade to Sonnet for Stage 2.7 if Haiku quality insufficient
3. **API Key**: `ANTHROPIC_API_KEY` environment variable ✓
4. **Rate Limits**: No rate limiter needed for MVP (sequential processing @ 1 req/sec well under limits). Add configurable 0.5s inter-request delay as safety buffer.
5. **Retry Strategy**: 3 retries with exponential backoff (1s, 2s, 4s). Fail-fast after 3rd failure (aligns with checkpoint/resume architecture).
6. **LLM Unavailability**: Hard dependency - fail immediately if unavailable (no fallback). Content Analysis required for Stage 7.

**For HLD Section**: 3.4 (External Dependencies), 6.2 (Error Cases), 7.1 (Performance Targets)

**Notes**:
- Anthropic rate limits: Haiku ~50 req/s, Sonnet ~10 req/s
- Our usage: Stage 2.6 = 1 req, Stage 2.7 = 40-300 sequential reqs @ 1/sec
- Retry only on: TimeoutError, APIError, 500 errors (not 400 errors = bad input)
- After 3 failed retries: raise exception, pipeline stops, resume from checkpoint after issue fixed

[Questions will be filled iteratively]

#### Q8: [CRITICAL] Stage 2.6 Discovery - Sampling Strategy
ContentAnalysis.md line 978 mentions sampling "50-100 transcripts" for pattern discovery.

**Question**: What's the exact sampling strategy for Stage 2.6 discovery?
1. Sample size: Exactly how many transcripts?
2. Sampling distribution: How to sample across top 3 buckets?
3. Top/Bottom mix: Sample from top performers only or mix with bottom?

**POINT 1 DECISION - Sample Size:**
**Answer**: 50 transcripts (default, configurable)
- Default: 50 transcripts
- Configurable via parameter for diverse hashtags
- Rationale: Balance between cost, speed, and pattern coverage

**POINT 2 DECISION - Sampling Distribution:**
**Answer**: Option B - Stratified Even Distribution
- Sample ~17 transcripts from each of the top 3 selected buckets
- Example: 17 from bucket 1, 17 from bucket 2, 16 from bucket 3 = 50 total
- Rationale: Ensures taxonomy discovers patterns across all duration ranges (e.g., 90-120s videos may have unique narrative patterns)
- Implementation: `samples_per_bucket = DISCOVERY_SAMPLE_SIZE // 3`

**POINT 3 DECISION - Top/Bottom Mix:**
**Answer**: Option A - Top Performers Only
- Sample all 50 transcripts from top performers only (not bottom performers)
- Goal: Identify what content categories exist in viral videos
- Rationale: Content categories are descriptive, not evaluative. Both top and bottom videos contain same categories (recipe_tutorial, workout_guide, etc.). The contrastive analysis ("What % of top vs bottom use X?") happens in Stage 7 with full dataset.
- Implementation: `samples = random.sample(manifest[bucket]['top_performers'], samples_per_bucket)`

**COMPLETE Q8 SUMMARY:**
- **Sample size**: 50 transcripts (default, configurable)
- **Distribution**: ~17 from each of top 3 buckets (stratified even)
- **Source**: Top performers only

**For HLD Section**: 2.3 (Detailed Process), 7.1 (Performance Targets)

#### Q9: [CRITICAL] Stage 2.7 Classification Scope
ContentAnalysis.md lines 862-879 show options for "Selective Analysis" (40 videos) vs "Full Analysis" (300 videos).

**Question**: For Stage 2.7 Classification, how many videos should we classify per hashtag, and from which performance groups?

**Answer**: Option B - Selective Contrastive (120 videos)
- **Scope**: 20 top performers + 20 bottom performers per bucket × 3 buckets = 120 videos total
- **Cost**: ~$0.12/hashtag with Haiku (vs $0.06 for top-only, $0.30 for full)
- **Rationale**:
  - Enables contrastive analysis in Stage 7 ("Top videos use X pattern 3x more than bottom")
  - Core business value requires proving differentiation, not just describing viral videos
  - Without bottom classifications, can't identify if a pattern is truly a differentiator
  - Cost difference ($0.60 for 10 hashtags) is trivial vs business value
- **Implementation**: Per bucket, classify 20 from `top_performers` + 20 from `bottom_performers`

**For HLD Section**: 2.3 (Detailed Process), 7.1 (Performance Targets), Cost Analysis

#### Q10: [HIGH] Taxonomy Schema - Discovery Output
ContentAnalysis.md lines 112-128 show example taxonomy structure.

**Question**: What's the exact schema for taxonomy JSON files (raw discovery vs curated)?

**Answer**: Option B - Structured with Selective Definitions (Hybrid)

**1. Raw Discovery Output** (`raw_discoveries/{hashtag}_raw.json`) - Stage 2.6 output:
```json
{
  "hashtag": "nutrition",
  "analysis_date": "2025-10-14",
  "sample_size": 50,
  "discovered_patterns": {
    "content_categories": [
      {
        "name": "recipe_tutorial",
        "frequency": 32,
        "percentage": 64,
        "examples": ["protein smoothie recipe", "meal prep guide"],
        "representative_video_ids": ["123", "456"]
      }
    ],
    "hook_strategies": [...],
    "audience_pain_points": [...],
    "trending_keywords": [...],
    "engagement_drivers": [...],
    "content_tactics": [...]
  }
}
```
Purpose: Rich context for manual curation (you need examples/frequency to decide what to keep)

**2. Curated Taxonomy** (`taxonomies/{hashtag}_taxonomy.json`) - After manual curation:
```json
{
  "hashtag": "nutrition",
  "content_categories": [
    {"name": "recipe_tutorial", "definition": "Step-by-step cooking instructions"},
    {"name": "supplement_review", "definition": "Product reviews for supplements"}
  ],
  "hook_strategies": [
    {"name": "problem_solution", "definition": "Starts with problem, promises solution"},
    {"name": "direct_statement", "definition": "Opens with bold declarative fact"}
  ],
  "audience_pain_points": ["bloating", "low_energy"],
  "trending_keywords": ["protein", "gut_health"],
  "engagement_drivers": ["before_after_reveal", "specific_metrics_mentioned"],
  "content_tactics": ["personal_story", "direct_to_camera"]
}
```

**Rule**: Add definitions for semantic categories (content_categories, hook_strategies). Simple lists remain strings (pain_points, keywords, tactics are self-explanatory).

**Validation**: All 6 fields required and non-empty. Categories with definitions must have both name + definition (>10 chars).

**For HLD Section**: 5.1 (Input Schema), 5.2 (Output Schema), 6.1 (Input Validation)

#### Q11: [HIGH] Performance Targets
**Question**: What are the acceptable performance targets for Content Analysis stages?

**Answer**: Pragmatic targets based on LLM API characteristics

**Stage 2.6 Discovery:**
- Expected time: 30 seconds (50 transcripts, Sonnet)
- Warning threshold: 60 seconds (log warning, continue)
- Timeout: 120 seconds (fail, API issue likely)

**Stage 2.7 Classification:**
- Expected time: 5 minutes (120 videos × 2.5s each with Haiku)
- Warning threshold: 10 minutes (log warning, continue)
- Per-video timeout: 30 seconds (retry on failure)
- Overall timeout: 15 minutes (fail, systemic API issue)

**LLM API Configuration:**
- API call timeout: 30 seconds
- Inter-request delay: 0.5 seconds (safety buffer for rate limits)

**Rationale:**
- 4x buffer over expected time accounts for API variance (P99)
- Background processing (not user-facing) - 5 min vs 10 min acceptable
- Fail-fast philosophy: 15+ minutes suggests broken API, better to fail and investigate
- Checkpoint/resume means timeouts don't lose work

**For HLD Section**: 7.1 (Performance Targets), 6.2 (Error Cases - timeout handling)

#### Q12: [HIGH] Output File Paths
**Question**: What are the exact output file paths for Content Analysis?

**Answer**: Follow ML pipeline architecture from MLPlanningv2.md

**File Structure:**
```
/data/clients/{client_id}/hashtags/{cluster_id}/top_contrastive/
├── content_taxonomies/                    # Cluster-level (shared across buckets)
│   ├── {cluster_id}_raw_discovery.json    # Stage 2.6 output
│   └── {cluster_id}_taxonomy.json         # Stage 2.6 curated (after manual curation)
│
└── buckets/
    └── bucket_{duration}/                 # e.g., bucket_33_60s
        ├── content_analysis/              # Stage 2.7 outputs (peer to ml_analysis/)
        │   ├── {video_id}_content.json    # 40 files per bucket (20 top + 20 bottom)
        │   └── ...
        ├── ml_analysis/                   # Stage 3-5 outputs
        └── llm_reports/                   # Stage 7 outputs
```

**Rationale:**
- Taxonomies at cluster level (one per hashtag, shared across all buckets)
- Classifications per bucket (inside each bucket's directory)
- `content_analysis/` is peer to `ml_analysis/` (both are derived analysis outputs)
- File naming: `{video_id}_content.json` follows pattern of `{video_id}_temporal_windows_updated.json`

**Example Paths:**
- Discovery: `/data/clients/acme/hashtags/nutrition/top_contrastive/content_taxonomies/nutrition_taxonomy.json`
- Classification: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_33_60s/content_analysis/7526250443832331550_content.json`

**For HLD Section**: 3.2 (Output Contracts), 8 (File Structure)

### Testing

#### Q13: [HIGH] Testing Strategy
**Question**: What testing approach should we use before running on full 120-video batches?

**Answer**: Four-stage testing progression

**Test 1: Discovery Quality Test (Stage 2.6)**
- Run discovery on 10 sample transcripts (not 50)
- Manually review raw_discovery output
- Validates: LLM pattern detection, JSON structure, taxonomy schema
- Cost: ~$0.15 (small sample)

**Test 2: Classification Quality Test (Stage 2.7)**
- Manually classify 5 videos yourself (ground truth)
- Run Stage 2.7 on same 5 videos
- Compare LLM vs manual classifications
- Validates: Haiku quality, taxonomy interpretation, schema correctness
- Cost: ~$0.005 (5 videos)

**Test 3: End-to-End Integration Test**
- Run full pipeline on 1 bucket with 10 videos (5 top + 5 bottom)
- Validates: File paths, manifest reading, error handling, output formats
- Cost: ~$0.01 (10 videos)

**Test 4: Full Production Run**
- After passing Tests 1-3, run on 120 videos (40 per bucket × 3 buckets)
- Cost: ~$0.12

**For HLD Section**: 8 (Testing Strategy)

## Completeness Check

Can write these HLD sections without TODOs or gaps?

**Section 2 (Architecture & Design):**
- ✅ 2.1: High-level approach - YES (Stage 2.6 discovery + manual curation + Stage 2.7 classification)
- ✅ 2.2: Data flow - YES (Q1-Q5: inputs from manifest/transcripts/captions, outputs to content_analysis/)
- ✅ 2.3: Detailed process - YES (Q8: sampling strategy, Q9: classification scope, Q7: LLM config, Q10: taxonomy schema)

**Section 3 (Dependencies & Integration):**
- ✅ 3.1: Input dependencies - YES (Q1/Q2: Stage 2.5 manifest, Q3: transcripts, Q5: unified_analysis)
- ✅ 3.2: Output contracts - YES (Q12: exact file paths, Q6: output schema)
- ✅ 3.3: Cross-stage dependencies - YES (Stage 2.5 → Content Analysis → Stage 7)
- ✅ 3.4: External dependencies - YES (Q7: Anthropic Claude API, Sonnet/Haiku models, ANTHROPIC_API_KEY)

**Section 5 (Data Schemas):**
- ✅ 5.1: Input schema - YES (Q3: transcript, Q5: caption/hashtags, Q2: manifest, Q10: taxonomy)
- ✅ 5.2: Output schema - YES (Q6: complete classification schema with all fields, Q10: taxonomy schemas)

**Section 6 (Error Handling):**
- ✅ 6.1: Input validation - YES (Q3: transcript validation, Q4: empty handling, Q10: taxonomy validation)
- ✅ 6.2: Error cases - YES (Q7: retry strategy 3x with backoff, Q11: timeouts, Q4: empty transcripts)
- ✅ 6.3: Output validation - YES (Q6: schema validation rules, Q10: taxonomy validation)

**Section 8 (Testing Strategy):**
- ✅ 8.1-8.3: Test cases - YES (Q13: 4-stage testing with discovery, classification, integration, production)

## Proceed to Phase 3

**Ready for HLD Generation**: YES

All critical information gathered. We have:
- 13 questions answered (9 CRITICAL, 4 HIGH)
- Complete input/output schemas
- Exact file paths and dependencies
- Error handling and retry strategies
- Performance targets and timeouts
- Testing approach

**Next Steps**: Proceed to Phase 3 with:
- Phase3_ChildHLDGeneration.md (instructions)
- ChildTemplate.md (HLD template)
- MLPlanningv2.md (mother doc - for Part 1 Foundation context)
- Critique_ContentAnalysis.md (Phase 1 output)
- QA_ContentAnalysis.md (this document)

**Status**: COMPLETE
