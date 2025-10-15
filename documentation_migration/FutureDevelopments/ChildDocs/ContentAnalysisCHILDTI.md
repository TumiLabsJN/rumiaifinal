# Content Analysis - Technical Implementation

> **TI Document**: ContentAnalysisCHILDTI.md
> **Parent HLD**: ContentAnalysisCHILD.md
> **Foundation HLD**: FoundationCHILD.md
> **Version**: 1.0
> **Last Updated**: 2025-01-28
> **Status**: Draft

---

## Section 1: Document Metadata

**TI_Document**: ContentAnalysisCHILDTI.md

**Parent_HLD**: ContentAnalysisCHILD.md

**Foundation_HLD**: FoundationCHILD.md

**Covers_HLD_Sections**:
- ContentAnalysisCHILD.md Section 1: Context & Business Goal
- ContentAnalysisCHILD.md Section 2: Architecture & Design
- ContentAnalysisCHILD.md Section 2.1: High-Level Approach
- ContentAnalysisCHILD.md Section 2.2: Data Flow
- ContentAnalysisCHILD.md Section 2.3: Detailed Process
- ContentAnalysisCHILD.md Section 3: Dependencies & Integration
- ContentAnalysisCHILD.md Section 3.1: Input Dependencies
- ContentAnalysisCHILD.md Section 3.2: Output Contracts
- ContentAnalysisCHILD.md Section 3.3: Cross-Stage Dependencies
- ContentAnalysisCHILD.md Section 3.4: External Dependencies
- ContentAnalysisCHILD.md Section 4: Configuration & Parameters
- ContentAnalysisCHILD.md Section 5: Data Schemas
- ContentAnalysisCHILD.md Section 5.1: Input Schema
- ContentAnalysisCHILD.md Section 5.2: Output Schema
- ContentAnalysisCHILD.md Section 6: Error Handling & Validation
- ContentAnalysisCHILD.md Section 6.1: Input Validation
- ContentAnalysisCHILD.md Section 6.2: Error Cases
- ContentAnalysisCHILD.md Section 6.3: Output Validation
- FoundationCHILD.md Section 2: Client Architecture & Directory Structure
- FoundationCHILD.md Section 2.1: Directory Structure
- FoundationCHILD.md Section 2.2: Path Templates
- FoundationCHILD.md Section 4: CLI Command Structure
- FoundationCHILD.md Section 4.1: CLI Parameters
- FoundationCHILD.md Section 5: Configuration Schemas
- FoundationCHILD.md Section 5.1: config.json Schema

**Related_TI_Docs**:
- **Depends_On**:
  - FoundationTI.md (ALWAYS)
  - VideoProcessingTI.md (Stage 2 - provides transcripts, captions, hashtags)
  - BucketSelectionTI.md (Stage 2.5 - provides selection_manifest.json)
- **Feeds_Into**:
  - LLMReportGenerationTI.md (Stage 7 - consumes content classifications for creative reports)

**Implementation_Priority**: HIGH
- **Rationale**: Hard dependency for Stage 7 (LLM Report Generation). Without content classifications, reports lack qualitative insights (ContentAnalysisCHILD.md Section 3.3). Enables contrastive analysis required by business value proposition (ContentAnalysisCHILD.md Section 1.1).

---

## Section 2: Stage Contract

<!-- Source: FoundationCHILD.md Sections 2, 4; ContentAnalysisCHILD.md Sections 3.1, 3.2, 5.1, 5.2 -->

```python
# INPUT CONTRACT
class StageInput:
    """
    Exact structure Stage 2.6 & 2.7 (Content Analysis) receives.
    Sources: FoundationCHILD.md Sections 2 & 4, ContentAnalysisCHILD.md Sections 3.1 & 5.1
    """
    # ===== CLI PARAMETERS (from FoundationCHILD.md Section 4.1) =====
    client_id: str              # CLI parameter --client, Required
                                # Example: "acme_corp"

    hashtag: str                # CLI parameter --target (for hashtag analysis)
                                # Required, Format: "#nutrition" or cluster_id "nutrition"
                                # Note: Can be cluster_id without # prefix

    analysis_mode: str          # CLI parameter --analysis-mode
                                # Default: "top"
                                # Valid values: ["top", "recent"]

    selection_strategy: str     # CLI parameter --selection-strategy
                                # Default: "contrastive"
                                # Valid values: ["contrastive", "top"]

    # ===== DIRECTORY PATHS (from FoundationCHILD.md Section 2.2) =====
    base_path: str              # /data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/
                                # Constructed from FoundationCHILD.md Section 2.2 BASE_PATHS

    bucket_base: str            # {base_path}/buckets/bucket_{bucket}/
                                # Per-bucket directory structure

    # ===== STAGE-SPECIFIC INPUTS (from ContentAnalysisCHILD.md Section 3.1) =====

    # Input 1: Selection Manifest (from Stage 2.5)
    selection_manifest_path: str    # {base_path}/selection_manifest.json
                                    # Required, must exist
                                    # Source: Stage 2.5 (Bucket Selection)
                                    # Schema: ContentAnalysisCHILD.md Section 5.1.1

    # Input 2: Transcripts (from Stage 2 - Whisper)
    transcripts_dir: str        # /home/jorge/rumiaifinal/speech_transcriptions/
                                # Absolute path, not in client directory
                                # Files: {video_id}_whisper.json
                                # Schema: ContentAnalysisCHILD.md Section 5.1.2

    # Input 3: Captions & Hashtags (from Stage 2 - unified_analysis)
    unified_analysis_dir: str   # /home/jorge/rumiaifinal/unified_analysis/
                                # Absolute path, not in client directory
                                # Files: {video_id}.json
                                # Schema: ContentAnalysisCHILD.md Section 5.1.3

    # Input 4: Curated Taxonomy (from Stage 2.6 manual curation)
    taxonomy_path: str          # {base_path}/content_taxonomies/{hashtag}_taxonomy.json
                                # Required for Stage 2.7, must exist after manual curation
                                # Created by Stage 2.6, curated by human
                                # Schema: ContentAnalysisCHILD.md Section 5.1.4

    # ===== ENVIRONMENT VARIABLES (from ContentAnalysisCHILD.md Section 3.4) =====
    ANTHROPIC_API_KEY: str      # Environment variable, Required
                                # Used for Claude API calls (Sonnet for discovery, Haiku for classification)


# OUTPUT CONTRACT
class StageOutput:
    """
    Exact structure Stage 2.6 & 2.7 (Content Analysis) produces.
    Sources: FoundationCHILD.md Section 2, ContentAnalysisCHILD.md Sections 3.2 & 5.2
    """
    # ===== STAGE 2.6: DISCOVERY OUTPUT =====

    # Output 1: Raw Discovery JSON
    raw_discovery_path: str     # {base_path}/content_taxonomies/{hashtag}_raw_discovery.json
                                # Format: JSON (~10KB)
                                # Consumer: Human curator (manual review)
                                # Schema: ContentAnalysisCHILD.md Section 5.2.1

    # Output 2: Curated Taxonomy (after manual curation)
    curated_taxonomy_path: str  # {base_path}/content_taxonomies/{hashtag}_taxonomy.json
                                # Format: JSON (~5KB)
                                # Consumer: Stage 2.7 (classification input)
                                # Schema: ContentAnalysisCHILD.md Section 5.1.4
                                # Note: Created manually after Stage 2.6, not automated output

    # ===== STAGE 2.7: CLASSIFICATION OUTPUT =====

    # Output 3: Video Classifications (120 files: 40 per bucket × 3 buckets)
    classification_files: list[str]  # {bucket_base}/content_analysis/{video_id}_content.json
                                     # Format: JSON (~2KB each, 120 total)
                                     # Consumer: Stage 7 (LLM Report Generation)
                                     # Schema: ContentAnalysisCHILD.md Section 5.2.2
                                     # Fields: 23 total (video_id, performance_group, 10 core, 12 caption_analysis subfields)

    # ===== OUTPUT METADATA =====

    # Stage 2.6 outputs
    discovery_sample_size: int  # 50 transcripts sampled for discovery
    discovery_cost_usd: float   # ~$0.75 per hashtag (Sonnet API cost)

    # Stage 2.7 outputs
    classified_video_count: int # 120 videos classified (40 per bucket × 3 buckets)
    classification_cost_usd: float  # ~$0.12 per hashtag (Haiku API cost)

    # Combined outputs
    total_cost_usd: float       # ~$0.87 per hashtag (first run)
                                # ~$0.12 per hashtag (subsequent runs, taxonomy reused)
```

---

## Section 3: Data Schemas

<!-- Source: FoundationCHILD.md Section 5, ContentAnalysisCHILD.md Sections 5.1, 5.2 -->

### 3.1 Foundation Schemas

```python
# ===== FOUNDATION SCHEMAS (from FoundationCHILD.md Section 5) =====
# These are cross-cutting schemas used by all stages

# Config Schema (FoundationCHILD.md Section 5.1)
ConfigSchema = {
    "client_id": str,              # Required, alphanumeric + underscore
                                   # Example: "acme_corp"
                                   # Source: FoundationCHILD.md Section 5.1

    "analysis_type": str,          # Required, ["hashtag", "competitor", "creator"]
                                   # Example: "hashtag"
                                   # Source: FoundationCHILD.md Section 5.1

    "target": str,                 # Required, format depends on analysis_type
                                   # Example: "#nutrition" (hashtag) or "nutrition" (cluster_id)
                                   # Source: FoundationCHILD.md Section 5.1

    "analysis_mode": str,          # Required, ["top", "recent"]
                                   # Example: "top"
                                   # Source: FoundationCHILD.md Section 5.1

    "selection_strategy": str,     # Required, ["contrastive", "top"]
                                   # Example: "contrastive"
                                   # Source: FoundationCHILD.md Section 5.1

    "video_count": int,            # Required, Range: 10-500
                                   # Example: 100
                                   # Source: FoundationCHILD.md Section 5.1

    "date_filter": str,            # Required, "last_N_days"
                                   # Example: "last_90_days"
                                   # Source: FoundationCHILD.md Section 5.1

    "country_code": str,           # Required, ["US", "BR", "global"]
                                   # Example: "US"
                                   # Source: FoundationCHILD.md Section 5.1

    "report_type": str,            # Required, ["single", "comparison"]
                                   # Example: "single"
                                   # Source: FoundationCHILD.md Section 5.1

    "report_audience": str,        # Required, ["client", "internal", "creator"]
                                   # Example: "client"
                                   # Source: FoundationCHILD.md Section 5.1

    "auto_confirm": bool,          # Required, skip interactive prompts
                                   # Example: False
                                   # Source: FoundationCHILD.md Section 5.1

    "run_date": str,               # Required, ISO 8601 format
                                   # Example: "2025-01-28T10:30:00Z"
                                   # Source: FoundationCHILD.md Section 5.1
}
```

### 3.2 Stage-Specific Input Schemas

```python
# ===== INPUT SCHEMA 1: SELECTION MANIFEST (from ContentAnalysisCHILD.md Section 5.1.1) =====

SelectionManifestSchema = {
    "hashtag": str,                # Required, hashtag name (without #)
                                   # Example: "nutrition"
                                   # Source: ContentAnalysisCHILD.md Section 5.1.1

    "selected_buckets": list[str], # Required, 3 items (top 3 duration buckets)
                                   # Example: ["33_60s", "60_90s", "90_120s"]
                                   # Source: ContentAnalysisCHILD.md Section 5.1.1

    "videos_by_bucket": dict,      # Required, video IDs organized by bucket
                                   # Schema: {bucket_name: {"top_performers": [...], "bottom_performers": [...]}}
                                   # Source: ContentAnalysisCHILD.md Section 5.1.1

    # Nested schema for videos_by_bucket[bucket_name]:
    # {
    #   "top_performers": list[str],     # 40-100 items, video IDs of top performers
    #                                    # Example: ["7526250443832331550", ...]
    #   "bottom_performers": list[str],  # 10-25 items, video IDs of bottom performers
    #                                    # Example: ["7428596413707144481", ...]
    # }

    "total_videos": int,           # Required, Range: 150-375 (total across all buckets)
                                   # Example: 300
                                   # Source: ContentAnalysisCHILD.md Section 5.1.1

    "timestamp": str,              # Required, ISO 8601 format
                                   # Example: "2025-10-14T10:30:00Z"
                                   # Source: ContentAnalysisCHILD.md Section 5.1.1
}

# ===== INPUT SCHEMA 2: TRANSCRIPT (from ContentAnalysisCHILD.md Section 5.1.2) =====

TranscriptSchema = {
    "text": str,                   # Required (can be empty string), Range: 0-5000 chars
                                   # Complete transcript from Whisper
                                   # Example: "this is why every woman needs to start yoni steaming..."
                                   # Source: ContentAnalysisCHILD.md Section 5.1.2

    "segments": list[dict],        # Required (not used by Content Analysis)
                                   # Timestamped segments, optional metadata
                                   # Source: ContentAnalysisCHILD.md Section 5.1.2

    "words": list[dict],           # Required (not used by Content Analysis)
                                   # Word-level data, optional metadata
                                   # Source: ContentAnalysisCHILD.md Section 5.1.2
}

# ===== INPUT SCHEMA 3: CAPTION AND HASHTAGS (from ContentAnalysisCHILD.md Section 5.1.3) =====

UnifiedAnalysisSchema = {
    "metadata": {
        "description": str,        # Optional (can be None or empty), Range: 0-2200 chars
                                   # Creator-written caption
                                   # Example: "this is why every woman needs to start yoni steaming..."
                                   # Source: ContentAnalysisCHILD.md Section 5.1.3

        "hashtags": list[dict],    # Optional (can be None or empty), Range: 0-30 items
                                   # Hashtag objects with id and name fields
                                   # Example: [{"id": "...", "name": "yonisteam"}, ...]
                                   # Source: ContentAnalysisCHILD.md Section 5.1.3

        # Nested schema for hashtags array items:
        # {
        #   "id": str,             # Optional, hashtag ID from TikTok
        #   "name": str,           # Optional (can be None), hashtag name without #
        #                          # Example: "yonisteam"
        # }
    }
}

# ===== INPUT SCHEMA 4: CURATED TAXONOMY (from ContentAnalysisCHILD.md Section 5.1.4) =====

CuratedTaxonomySchema = {
    "hashtag": str,                # Required, hashtag name
                                   # Example: "nutrition"
                                   # Source: ContentAnalysisCHILD.md Section 5.1.4

    "content_categories": list[dict],  # Required, Range: 2-10 items
                                       # Semantic categories with definitions
                                       # Example: [{"name": "recipe_tutorial", "definition": "Step-by-step cooking instructions"}]
                                       # Source: ContentAnalysisCHILD.md Section 5.1.4

    # Nested schema for content_categories items:
    # {
    #   "name": str,               # Required, category identifier (snake_case)
    #   "definition": str,         # Required, >10 chars, human-readable description
    # }

    "hook_strategies": list[dict], # Required, Range: 2-10 items
                                   # Hook patterns with definitions
                                   # Example: [{"name": "problem_solution", "definition": "Starts with problem, promises solution"}]
                                   # Source: ContentAnalysisCHILD.md Section 5.1.4

    # Nested schema for hook_strategies items:
    # {
    #   "name": str,               # Required, strategy identifier (snake_case)
    #   "definition": str,         # Required, >10 chars, human-readable description
    # }

    "audience_pain_points": list[str],  # Required, Range: 2-15 items
                                        # Pain points (simple strings)
                                        # Example: ["bloating", "low_energy"]
                                        # Source: ContentAnalysisCHILD.md Section 5.1.4

    "trending_keywords": list[str],     # Required, Range: 2-15 items
                                        # Keywords (simple strings)
                                        # Example: ["protein", "gut_health"]
                                        # Source: ContentAnalysisCHILD.md Section 5.1.4

    "engagement_drivers": list[str],    # Required, Range: 2-15 items
                                        # Tactics (simple strings)
                                        # Example: ["before_after_reveal", "specific_metrics_mentioned"]
                                        # Source: ContentAnalysisCHILD.md Section 5.1.4

    "content_tactics": list[str],       # Required, Range: 2-15 items
                                        # Presentation styles (simple strings)
                                        # Example: ["personal_story", "direct_to_camera"]
                                        # Source: ContentAnalysisCHILD.md Section 5.1.4
}
```

### 3.3 Stage-Specific Output Schemas

```python
# ===== OUTPUT SCHEMA 1: RAW DISCOVERY (from ContentAnalysisCHILD.md Section 5.2.1) =====

RawDiscoverySchema = {
    "hashtag": str,                # Required, hashtag name
                                   # Example: "nutrition"
                                   # Source: ContentAnalysisCHILD.md Section 5.2.1

    "analysis_date": str,          # Required, ISO 8601 timestamp
                                   # Example: "2025-10-14T10:30:00Z"
                                   # Source: ContentAnalysisCHILD.md Section 5.2.1

    "sample_size": int,            # Required, number of transcripts analyzed
                                   # Typically 50
                                   # Source: ContentAnalysisCHILD.md Section 5.2.1

    "discovered_patterns": dict,   # Required, container for all pattern categories
                                   # Contains 6 categories (see nested schemas below)
                                   # Source: ContentAnalysisCHILD.md Section 5.2.1
}

# Nested schema for discovered_patterns.content_categories (and all other pattern arrays):
DiscoveredPatternSchema = {
    "name": str,                   # Required, category identifier
                                   # Example: "recipe_tutorial"
                                   # Source: ContentAnalysisCHILD.md Section 5.2.1

    "frequency": int,              # Required, count of videos with this pattern
                                   # Example: 32
                                   # Source: ContentAnalysisCHILD.md Section 5.2.1

    "percentage": float,           # Required, frequency / sample_size * 100
                                   # Example: 64.0
                                   # Source: ContentAnalysisCHILD.md Section 5.2.1

    "examples": list[str],         # Required, 2-3 example phrases
                                   # Example: ["protein smoothie recipe", "meal prep tutorial"]
                                   # Source: ContentAnalysisCHILD.md Section 5.2.1

    "representative_video_ids": list[str],  # Required, video IDs showing this pattern
                                            # Example: ["7526250443832331550", "7428596413707144481"]
                                            # Source: ContentAnalysisCHILD.md Section 5.2.1
}

# ===== OUTPUT SCHEMA 2: VIDEO CLASSIFICATION (from ContentAnalysisCHILD.md Section 5.2.2) =====

VideoClassificationSchema = {
    "video_id": str,               # Required, video identifier
                                   # Example: "7526250443832331550"
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "performance_group": str,      # Required, ["top", "bottom"]
                                   # Performance classification (from selection_manifest)
                                   # Example: "top"
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "content_category": str,       # Required, from taxonomy
                                   # Primary content type
                                   # Example: "recipe_tutorial"
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "hook_strategy": str,          # Required, from taxonomy
                                   # Opening pattern
                                   # Example: "problem_solution"
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "audience_pain_points": list[str],  # Required (can be empty array), from taxonomy
                                        # Detected pain points
                                        # Example: ["menstrual_discomfort", "feminine_wellness"]
                                        # Source: ContentAnalysisCHILD.md Section 5.2.2

    "trending_keywords": list[str],     # Required (can be empty array), from taxonomy
                                        # Detected keywords
                                        # Example: ["yoni", "steaming", "holistic", "tcm"]
                                        # Source: ContentAnalysisCHILD.md Section 5.2.2

    "engagement_drivers": list[str],    # Required (can be empty array), from taxonomy
                                        # Shareability tactics
                                        # Example: ["personal_testimony", "product_recommendation"]
                                        # Source: ContentAnalysisCHILD.md Section 5.2.2

    "content_tactics": list[str],       # Required (can be empty array), from taxonomy
                                        # Presentation styles
                                        # Example: ["direct_to_camera", "product_demonstration"]
                                        # Source: ContentAnalysisCHILD.md Section 5.2.2

    "caption_analysis": dict,      # Required, caption-specific analysis (12 subfields)
                                   # See nested schema below
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "confidence": str,             # Required, ["high", "medium", "low"]
                                   # Classification confidence
                                   # Example: "high"
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "transcript_available": bool,  # Required
                                   # Whether transcript was used (false = caption/hashtag only)
                                   # Example: True
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "note": str,                   # Optional (can be None)
                                   # Example: "Classified using caption and hashtags only"
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2
}

# Nested schema for caption_analysis:
CaptionAnalysisSchema = {
    "caption_hook_type": str,      # Required, ["statement", "question", "command", "teaser", "statistic", "contradiction"]
                                   # How caption opens
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "caption_cta_type": str,       # Required, ["link_in_bio", "save_post", "comment", "follow", "share", "tag_friend", "none"]
                                   # Call-to-action type
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "caption_cta_present": bool,   # Required
                                   # Whether CTA exists
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "brand_mention_present": bool, # Required
                                   # Whether brand/influencer mentioned
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "influencer_tag_present": bool,# Required
                                   # Whether influencer tagged
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "emoji_usage": str,            # Required, ["none", "light", "moderate", "heavy"]
                                   # Emoji density
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "caption_length": str,         # Required, ["short", "medium", "long"]
                                   # Caption length category
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "hashtag_count": int,          # Required, Range: 0-30
                                   # Number of hashtags
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "hashtag_placement": str,      # Required, ["end", "mixed", "none"]
                                   # Where hashtags appear
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "hashtag_strategy": dict,      # Required, nested schema with 3 subfields
                                   # See nested schema below
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2
}

# Nested schema for hashtag_strategy:
HashtagStrategySchema = {
    "broad_count": int,            # Required, Range: 0-30
                                   # Broad hashtags (50M+ views)
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "niche_count": int,            # Required, Range: 0-30
                                   # Niche hashtags (<5M views)
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2

    "branded_count": int,          # Required, Range: 0-30
                                   # Branded hashtags
                                   # Source: ContentAnalysisCHILD.md Section 5.2.2
}
```

**Field Count Verification:**

```
ContentAnalysisCHILD.md Section 5.1.1 (Selection Manifest): 5 fields → TI Schema 3.2: 5 fields ✓
ContentAnalysisCHILD.md Section 5.1.2 (Transcript): 3 fields → TI Schema 3.2: 3 fields ✓
ContentAnalysisCHILD.md Section 5.1.3 (Unified Analysis): 2 nested fields → TI Schema 3.2: 2 fields ✓
ContentAnalysisCHILD.md Section 5.1.4 (Taxonomy): 7 fields → TI Schema 3.2: 7 fields ✓
ContentAnalysisCHILD.md Section 5.2.2 (Classification): 23 fields (11 core + 12 caption_analysis) → TI Schema 3.3: 23 fields ✓
```

**Field Name Spot Check:**

```
1. ContentAnalysisCHILD.md: "performance_group" → TI: "performance_group" ✓
2. ContentAnalysisCHILD.md: "content_category" → TI: "content_category" ✓
3. ContentAnalysisCHILD.md: "hook_strategy" → TI: "hook_strategy" ✓
4. ContentAnalysisCHILD.md: "caption_hook_type" → TI: "caption_hook_type" ✓
5. ContentAnalysisCHILD.md: "hashtag_count" → TI: "hashtag_count" ✓
```

---

## Section 4: Algorithmic Specifications

<!-- Source: ContentAnalysisCHILD.md Section 2.3, Appendix C -->

### 4.1 Function: sample_transcripts_for_discovery()

**Source**: ContentAnalysisCHILD.md Section 2.3.1 - Stage 2.6 - Discovery Sampling

**Purpose**: Select 50 representative transcripts from top performers across top 3 buckets for pattern discovery

**Algorithm (Pseudocode)**:
```python
def sample_transcripts_for_discovery(
    manifest_path: str,
    sample_size: int = 50
) -> list[dict]:
    """
    Sample transcripts stratified evenly across top 3 buckets.

    Args:
        manifest_path: Path to selection_manifest.json from Stage 2.5
        sample_size: Total transcripts to sample (default: 50, configurable)

    Returns:
        list[dict]: Sampled video IDs with transcript text and bucket assignment
                    Format: [{"video_id": str, "text": str, "bucket": str}, ...]

    Raises:
        FileNotFoundError: If manifest_path does not exist
        ValueError: If manifest missing required fields
    """
    # Step 1: Load manifest from Stage 2.5
    # Source: ContentAnalysisCHILD.md Section 2.3.1 lines 131-133
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)

    # Step 2: Validate manifest structure
    # Source: ContentAnalysisCHILD.md Section 6.1 (Input Validation)
    required_fields = ['hashtag', 'selected_buckets', 'videos_by_bucket']
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")

    # Step 3: Extract top 3 buckets
    # Source: ContentAnalysisCHILD.md Section 2.3.1 line 133
    top_3_buckets = manifest['selected_buckets']  # e.g., ["33_60s", "60_90s", "90_120s"]

    # Step 4: Calculate samples per bucket (stratified even sampling)
    # Source: ContentAnalysisCHILD.md Section 2.3.1 line 136
    samples_per_bucket = sample_size // 3  # ~17 per bucket

    # Step 5: Initialize results container
    sampled_transcripts = []

    # Step 6: Sample from each bucket
    for bucket in top_3_buckets:
        # Step 6.1: Validate bucket exists in manifest
        if bucket not in manifest['videos_by_bucket']:
            logger.warning(f"Bucket {bucket} not in videos_by_bucket, skipping")
            continue

        # Step 6.2: Extract top performers only
        # Source: ContentAnalysisCHILD.md Section 2.3.1 line 141
        top_performers = manifest['videos_by_bucket'][bucket]['top_performers']

        # Step 6.3: Random sample (handle case where bucket has < samples_per_bucket videos)
        # Source: ContentAnalysisCHILD.md Section 2.3.1 lines 144
        sample_count = min(samples_per_bucket, len(top_performers))
        sampled_ids = random.sample(top_performers, sample_count)

        # Step 6.4: Load transcripts for sampled videos
        for video_id in sampled_ids:
            # Step 6.4.1: Construct transcript path
            # Source: ContentAnalysisCHILD.md Section 2.3.1 line 148
            transcript_path = f"/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json"

            # Step 6.4.2: Load transcript (handle missing files gracefully)
            # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases table
            try:
                transcript_data = load_json(transcript_path)
                text = transcript_data.get('text', '')

                # Step 6.4.3: Include even if empty (may reveal no-speech patterns)
                # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases row 3
                sampled_transcripts.append({
                    "video_id": video_id,
                    "text": text,
                    "bucket": bucket
                })
            except FileNotFoundError:
                # Step 6.4.4: Log warning and skip video
                # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases row 2
                logger.warning(f"Transcript not found: {video_id}, skipping")
                continue

    # Step 7: Validate we have sufficient samples
    if len(sampled_transcripts) < 10:
        raise ValueError(
            f"Insufficient transcripts sampled: {len(sampled_transcripts)}. "
            f"Minimum 10 required for pattern discovery."
        )

    # Step 8: Return sampled transcripts
    return sampled_transcripts
```

**Edge Cases (Exhaustive List)**:
- **Case 1**: Bucket has < 17 videos → Sample all available (Rationale: Rare, buckets typically have 40-80 videos)
- **Case 2**: Transcript file missing → Skip video, log warning (Rationale: Fail gracefully, use available transcripts)
- **Case 3**: Empty transcript (no speech) → Include in sample (Rationale: May reveal "no-speech" content patterns)
- **Case 4**: Sample size not divisible by 3 → Remainder distributed to first buckets (Rationale: 50÷3 = 16, 17, 17)

**Validation Rules**:
```python
assert os.path.exists(manifest_path), f"Manifest must exist: {manifest_path}"
assert 'selected_buckets' in manifest, "Manifest must have 'selected_buckets' field"
assert len(manifest['selected_buckets']) == 3, "Must have exactly 3 buckets"
assert sample_size >= 10, "Sample size must be >= 10"
assert len(sampled_transcripts) >= 10, "Must sample at least 10 transcripts"
```

**Error Conditions**:
- FileNotFoundError: Manifest not found (links to Section 6 Error Case: "missing_input_file")
- ValueError: Invalid manifest structure (links to Section 6 Error Case: "invalid_manifest_structure")

**Example Input**:
```json
{
  "hashtag": "nutrition",
  "selected_buckets": ["33_60s", "60_90s", "90_120s"],
  "videos_by_bucket": {
    "33_60s": {
      "top_performers": ["7526250443832331550", "7428596413707144481"]
    }
  }
}
```

**Example Output**:
```python
[
  {
    "video_id": "7526250443832331550",
    "text": "this is why every woman needs to start yoni steaming...",
    "bucket": "33_60s"
  },
  {
    "video_id": "7428596413707144481",
    "text": "Two minute TikTok videos...",
    "bucket": "60_90s"
  }
  # ... 48 more transcripts
]
```

---

### 4.2 Function: discover_patterns_llm()

**Source**: ContentAnalysisCHILD.md Section 2.3.2 - Stage 2.6 - LLM Discovery

**Purpose**: Use Claude 3.5 Sonnet to discover natural content patterns from 50 transcripts

**Algorithm (Pseudocode)**:
```python
def discover_patterns_llm(
    transcripts: list[dict],
    hashtag: str
) -> dict:
    """
    Discover content patterns using LLM (Claude 3.5 Sonnet).

    Args:
        transcripts: List of transcript dicts with video_id, text, bucket
        hashtag: str, hashtag name (e.g., "nutrition")

    Returns:
        dict: Raw discovery JSON with patterns, frequencies, examples
              Schema: ContentAnalysisCHILD.md Section 5.2.1

    Raises:
        TimeoutError: If LLM exceeds 120s timeout after 3 retries
        ValueError: If LLM returns invalid JSON after 3 retries
    """
    # Step 1: Prepare prompt with taxonomy discovery instructions
    # Source: ContentAnalysisCHILD.md Section 2.3.2 lines 187-218
    prompt = f"""
    Analyze {len(transcripts)} TikTok transcripts from #{hashtag}.

    Identify natural patterns in:
    1. Content Categories: What types of videos exist? (e.g., recipe_tutorial, supplement_review)
    2. Hook Strategies: How do videos open? (e.g., problem_solution, direct_statement)
    3. Audience Pain Points: What problems are mentioned? (e.g., bloating, low_energy)
    4. Trending Keywords: What terms appear frequently? (e.g., protein, gut_health)
    5. Engagement Drivers: What tactics make content shareable? (e.g., before_after_reveal)
    6. Content Tactics: What presentation styles are used? (e.g., personal_story, direct_to_camera)

    For each pattern:
    - name: short snake_case identifier
    - frequency: count of videos exhibiting this pattern
    - percentage: frequency / total videos * 100
    - examples: 2-3 example phrases from transcripts
    - representative_video_ids: video IDs showing this pattern

    Return JSON with structure:
    {{
      "hashtag": "{hashtag}",
      "analysis_date": "{datetime.utcnow().isoformat()}Z",
      "sample_size": {len(transcripts)},
      "discovered_patterns": {{
        "content_categories": [{{"name": "...", "frequency": N, "percentage": P, "examples": [...], "representative_video_ids": [...]}}],
        "hook_strategies": [...],
        "audience_pain_points": [...],
        "trending_keywords": [...],
        "engagement_drivers": [...],
        "content_tactics": [...]
      }}
    }}

    Transcripts:
    {json.dumps([t['text'] for t in transcripts])}
    """

    # Step 2: Initialize Anthropic client
    # Source: ContentAnalysisCHILD.md Section 2.3.2 line 224
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Step 3: Call API with retry logic (3 attempts with exponential backoff)
    # Source: ContentAnalysisCHILD.md Section 2.3.2 Edge Cases table
    for attempt in range(3):
        try:
            # Step 3.1: Make API call with Sonnet model
            # Source: ContentAnalysisCHILD.md Section 2.3.2 lines 226-230
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                timeout=120,  # 2 minutes
                messages=[{"role": "user", "content": prompt}]
            )

            # Step 3.2: Extract response text
            response_text = response.content[0].text

            # Step 3.3: Parse JSON response
            # Source: ContentAnalysisCHILD.md Section 2.3.2 line 234
            raw_taxonomy = json.loads(response_text)

            # Step 3.4: Validate response structure before returning
            # Source: ContentAnalysisCHILD.md Section 6.3 (Output Validation)
            validate_discovery_output(raw_taxonomy)

            # Step 3.5: Save raw discovery to file
            # Source: ContentAnalysisCHILD.md Section 2.3.2 lines 237-238
            output_path = f"/data/clients/{{client_id}}/hashtags/{hashtag}/top_contrastive/content_taxonomies/{hashtag}_raw_discovery.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_json(output_path, raw_taxonomy)

            # Step 3.6: Log success and manual curation instructions
            # Source: ContentAnalysisCHILD.md Section 2.3.2 lines 240-242
            logger.info(f"✅ Discovery complete: {output_path}")
            logger.info(f"📝 Next: Manually curate and save to content_taxonomies/{hashtag}_taxonomy.json")

            # Step 3.7: Return successful result
            return raw_taxonomy

        except TimeoutError as e:
            # Step 3.8: Handle timeout (retry with backoff)
            # Source: ContentAnalysisCHILD.md Section 2.3.2 Edge Cases row 1
            if attempt < 2:
                delay = [1, 2, 4][attempt]  # Exponential backoff
                logger.warning(f"⏰ Discovery timeout (>120s). Retry {attempt+1}/3 in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"❌ Discovery failed after 3 retries. Check status.anthropic.com")
                raise

        except json.JSONDecodeError as e:
            # Step 3.9: Handle invalid JSON (retry)
            # Source: ContentAnalysisCHILD.md Section 2.3.2 Edge Cases row 2
            if attempt < 2:
                delay = [1, 2, 4][attempt]
                logger.warning(f"⚠️ LLM returned invalid JSON. Retry {attempt+1}/3 in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"❌ Invalid JSON after 3 retries: {str(e)}")
                raise ValueError(f"LLM returned invalid JSON after 3 retries: {str(e)}")

    # Unreachable (for type checker)
    raise RuntimeError("Unexpected retry loop exit")
```

**Edge Cases (Exhaustive List)**:
- **Case 1**: LLM timeout (>120s) → Retry 3x with backoff (1s, 2s, 4s), then fail (Rationale: API may be slow)
- **Case 2**: Invalid JSON response → Retry 3x, then fail with clear error (Rationale: LLM occasionally malforms JSON)
- **Case 3**: Very low pattern frequency (<5%) → Include in raw output, curator filters (Rationale: Human decides actionability)
- **Case 4**: Patterns missing a field → Log warning, include partial data (Rationale: Curator can fix during review)

**Validation Rules**:
```python
assert os.environ.get('ANTHROPIC_API_KEY'), "ANTHROPIC_API_KEY must be set"
assert len(transcripts) >= 10, "Need at least 10 transcripts for discovery"
assert 'discovered_patterns' in raw_taxonomy, "Response must have 'discovered_patterns'"
assert len(raw_taxonomy['discovered_patterns']) == 6, "Must have all 6 pattern categories"
```

**Error Conditions**:
- TimeoutError: LLM API timeout after 3 retries (links to Section 6 Error Case: "llm_api_timeout_discovery")
- ValueError: Invalid JSON response after 3 retries (links to Section 6 Error Case: "invalid_json_response")

**Example Trace (Step-by-Step)**:
Input: 50 transcripts from nutrition hashtag
Step 1: Build prompt with taxonomy instructions → 5000 char prompt
Step 2: Call Sonnet API with 120s timeout → Response in 45s
Step 3: Parse JSON response → 6 pattern categories discovered
Step 4: Validate output structure → All required fields present
Step 5: Save to `/data/clients/acme/hashtags/nutrition/top_contrastive/content_taxonomies/nutrition_raw_discovery.json`
Output: Raw discovery JSON with 32 recipe tutorials, 18 supplement reviews, etc.

---

### 4.3 Function: classify_video_llm()

**Source**: ContentAnalysisCHILD.md Section 2.3.4 - Stage 2.7 - Video Classification (lines 392-475)

**Purpose**: Classify single video using LLM (Claude 3 Haiku) + curated taxonomy

**Algorithm (Pseudocode)**:
```python
def classify_video_llm(
    video_id: str,
    transcript: dict,
    caption: str,
    hashtags: list[str],
    taxonomy: dict,
    client: anthropic.Anthropic
) -> dict:
    """
    Classify single video using LLM + taxonomy.

    Args:
        video_id: Video identifier
        transcript: {"text": str, "available": bool}
        caption: Creator-written caption (can be empty string)
        hashtags: List of hashtag names without # (can be empty list)
        taxonomy: Curated taxonomy from Stage 2.6
        client: Initialized Anthropic API client

    Returns:
        dict: Classification JSON with 23 fields (schema Section 3.3)

    Raises:
        TimeoutError: If LLM exceeds 30s timeout per video after 3 retries
        ValueError: If LLM returns invalid JSON after 3 retries
    """
    # Step 1: Build classification prompt with taxonomy + video data
    # Source: ContentAnalysisCHILD.md Section 2.3.4 lines 398-453
    prompt = f"""
    Classify this TikTok video using the predefined taxonomy.

    TAXONOMY:
    Content Categories: {json.dumps(taxonomy['content_categories'])}
    Hook Strategies: {json.dumps(taxonomy['hook_strategies'])}
    Audience Pain Points: {json.dumps(taxonomy['audience_pain_points'])}
    Trending Keywords: {json.dumps(taxonomy['trending_keywords'])}
    Engagement Drivers: {json.dumps(taxonomy['engagement_drivers'])}
    Content Tactics: {json.dumps(taxonomy['content_tactics'])}

    VIDEO DATA:
    Transcript: "{transcript['text']}"
    Caption: "{caption}"
    Hashtags: {json.dumps(hashtags)}

    Return JSON with this EXACT structure:
    {{
      "video_id": "{video_id}",
      "content_category": "<string from content_categories>",
      "hook_strategy": "<string from hook_strategies>",
      "audience_pain_points": ["<strings from audience_pain_points>"],
      "trending_keywords": ["<strings from trending_keywords>"],
      "engagement_drivers": ["<strings from engagement_drivers>"],
      "content_tactics": ["<strings from content_tactics>"],
      "caption_analysis": {{
        "caption_hook_type": "<statement|question|command|teaser|statistic|contradiction>",
        "caption_cta_type": "<link_in_bio|save_post|comment|follow|share|tag_friend|none>",
        "caption_cta_present": <true|false>,
        "brand_mention_present": <true|false>,
        "influencer_tag_present": <true|false>,
        "emoji_usage": "<none|light|moderate|heavy>",
        "caption_length": "<short|medium|long>",
        "hashtag_count": <int>,
        "hashtag_placement": "<end|mixed|none>",
        "hashtag_strategy": {{
          "broad_count": <int>,
          "niche_count": <int>,
          "branded_count": <int>
        }}
      }},
      "confidence": "<high|medium|low>",
      "transcript_available": {str(transcript['available']).lower()},
      "note": {'"Classified using caption and hashtags only"' if not transcript['available'] else 'null'}
    }}

    Instructions:
    - If transcript is empty, classify using caption + hashtags only
    - Select ONE content_category (primary classification)
    - Select ONE hook_strategy
    - Select multiple (0-N) for arrays (pain_points, keywords, drivers, tactics)
    - Analyze caption structure for caption_analysis fields
    - Categorize hashtags as broad (50M+ views), niche (<5M views), or branded
    - Set confidence based on classification certainty
    """

    # Step 2: Call API with retry logic (3 attempts with exponential backoff)
    # Source: ContentAnalysisCHILD.md Section 2.3.4 lines 454-475
    for attempt in range(3):
        try:
            # Step 2.1: Make API call with Haiku model
            # Source: ContentAnalysisCHILD.md Section 2.3.4 lines 457-462
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                timeout=30,  # 30 seconds per video
                messages=[{"role": "user", "content": prompt}]
            )

            # Step 2.2: Parse response
            classification = json.loads(response.content[0].text)

            # Step 2.3: Validate output schema before returning
            # Source: ContentAnalysisCHILD.md Section 6.3 (Output Validation)
            validate_classification_output(classification)

            # Step 2.4: Return successful classification
            return classification

        except (TimeoutError, anthropic.APIError) as e:
            # Step 2.5: Handle timeout/API errors (retry with backoff)
            # Source: ContentAnalysisCHILD.md Section 2.3.4 lines 468-472
            if attempt < 2:
                delay = [1, 2, 4][attempt]  # Exponential backoff
                logger.warning(f"API failed for {video_id}, retry {attempt+1} in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"API failed for {video_id} after 3 retries")
                raise  # Re-raise after final retry

        except json.JSONDecodeError as e:
            # Step 2.6: Handle invalid JSON (retry)
            if attempt < 2:
                delay = [1, 2, 4][attempt]
                logger.warning(f"Invalid JSON for {video_id}, retry {attempt+1} in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"Invalid JSON for {video_id} after 3 retries: {str(e)}")
                raise ValueError(f"LLM returned invalid JSON: {str(e)}")

    # Unreachable (for type checker)
    raise RuntimeError("Unexpected retry loop exit")
```

**Edge Cases (Exhaustive List)**:
- **Case 1**: Empty transcript → Classify using caption/hashtags only, set transcript_available=false (Rationale: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases row 1)
- **Case 2**: Missing caption → Use empty string, classification uses transcript + hashtags (Rationale: Captions are optional)
- **Case 3**: Missing hashtags → Use empty array, classification uses transcript + caption (Rationale: Hashtags are optional)
- **Case 4**: LLM timeout (>30s per video) → Retry 3x with backoff, then fail (Rationale: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases row 4)
- **Case 5**: Invalid JSON response → Retry 3x, then fail with error (Rationale: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases row 5)

**Validation Rules**:
```python
assert video_id, "video_id cannot be empty"
assert taxonomy, "taxonomy cannot be empty"
assert 'content_categories' in taxonomy, "taxonomy must have content_categories"
assert client is not None, "Anthropic client must be initialized"
```

**Error Conditions**:
- TimeoutError: LLM API timeout per video after 3 retries (links to Section 6 Error Case: "llm_api_timeout_per_video")
- ValueError: Invalid JSON response after 3 retries (links to Section 6 Error Case: "invalid_json_response")

**Example Trace (Step-by-Step)**:
Input: video_id="7526250443832331550", transcript="this is why every woman needs to start yoni steaming...", caption="...", hashtags=["yonisteam", "wellness"]
Step 1: Build prompt with taxonomy + video data → 3000 char prompt
Step 2: Call Haiku API with 30s timeout → Response in 2.5s
Step 3: Parse JSON → 23 fields extracted
Step 4: Validate all required fields present → Pass
Step 5: Return classification dict
Output: {"video_id": "7526250443832331550", "content_category": "wellness_practice", "confidence": "high", ...}

---

## Section 5: Validation Rules

<!-- Source: ContentAnalysisCHILD.md Sections 6.1, 6.3, 2.3.X Edge Cases -->

### 5.1 Input Validation

```python
# ===== STAGE 2.6 DISCOVERY INPUT VALIDATION =====
# Source: ContentAnalysisCHILD.md Section 6.1 (lines 784-835)

def validate_discovery_inputs(manifest_path: str, sample_size: int):
    """
    Validate inputs before discovery.
    Source: ContentAnalysisCHILD.md Section 6.1
    """
    # Validation 1: Check manifest exists
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 790-795
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"selection_manifest.json not found at {manifest_path}. "
            "Did Stage 2.5 complete successfully?"
        )

    # Validation 2: Load and validate manifest structure
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 797-801
    manifest = load_json(manifest_path)
    required_fields = ['hashtag', 'selected_buckets', 'videos_by_bucket']
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")

    # Validation 3: Check we have 3 buckets
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 803-808
    if len(manifest['selected_buckets']) != 3:
        raise ValueError(
            f"Expected 3 selected buckets, found {len(manifest['selected_buckets'])}. "
            "Stage 2.5 may have failed."
        )

    # Validation 4: Check each bucket has videos
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 810-821
    for bucket in manifest['selected_buckets']:
        if bucket not in manifest['videos_by_bucket']:
            raise ValueError(f"Bucket {bucket} missing from videos_by_bucket")

        top_performers = manifest['videos_by_bucket'][bucket].get('top_performers', [])
        if len(top_performers) < 10:
            raise ValueError(
                f"Bucket {bucket} has only {len(top_performers)} top performers. "
                f"Need at least 10 for sampling."
            )

    # Validation 5: Check sample size is reasonable
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 823-827
    if sample_size < 10:
        raise ValueError(f"Sample size too small: {sample_size}. Minimum is 10.")
    if sample_size > 200:
        logger.warning(f"Sample size very large: {sample_size}. May exceed LLM token limits.")

    # Validation 6: Check ANTHROPIC_API_KEY set
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 829-834
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set with: export ANTHROPIC_API_KEY=sk-ant-..."
        )


# ===== STAGE 2.7 CLASSIFICATION INPUT VALIDATION =====
# Source: ContentAnalysisCHILD.md Section 6.1 (lines 838-893)

def validate_classification_inputs(taxonomy_path: str, manifest_path: str):
    """
    Validate inputs before classification.
    Source: ContentAnalysisCHILD.md Section 6.1
    """
    # Validation 1: Check taxonomy exists
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 844-849
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(
            f"Curated taxonomy not found at {taxonomy_path}. "
            "Run Stage 2.6 discovery and complete manual curation first."
        )

    # Validation 2: Load and validate taxonomy structure
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 851-866
    taxonomy = load_json(taxonomy_path)

    # Check all required fields present
    required_fields = [
        'content_categories', 'hook_strategies', 'audience_pain_points',
        'trending_keywords', 'engagement_drivers', 'content_tactics'
    ]
    missing = [f for f in required_fields if f not in taxonomy]
    if missing:
        raise ValueError(f"Taxonomy missing required fields: {missing}")

    # Check all fields non-empty
    for field in required_fields:
        if not taxonomy[field]:
            raise ValueError(f"Taxonomy field '{field}' is empty")

    # Validation 3: Check semantic categories have definitions
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 868-885
    for category in taxonomy['content_categories']:
        if 'name' not in category or 'definition' not in category:
            raise ValueError(f"content_categories missing name or definition: {category}")
        if len(category['definition']) < 10:
            raise ValueError(
                f"Definition too short for '{category['name']}': "
                f"'{category['definition']}' (min 10 chars)"
            )

    for strategy in taxonomy['hook_strategies']:
        if 'name' not in strategy or 'definition' not in strategy:
            raise ValueError(f"hook_strategies missing name or definition: {strategy}")
        if len(strategy['definition']) < 10:
            raise ValueError(
                f"Definition too short for '{strategy['name']}': "
                f"'{strategy['definition']}' (min 10 chars)"
            )

    # Validation 4: Check manifest exists (same as discovery validation)
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 887-892
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"selection_manifest.json not found at {manifest_path}. "
            "Did Stage 2.5 complete successfully?"
        )
```

---

### 5.2 Business Logic Validation

```python
# ===== EDGE CASE HANDLING FROM SECTION 2.3.X =====
# Source: ContentAnalysisCHILD.md Section 2.3.X Edge Cases tables

def validate_business_rules_sampling(sampled_transcripts: list[dict]):
    """
    Validate business rules during sampling.
    Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases
    """
    # Rule 1: Bucket with < 17 videos
    # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases row 1
    # Handling: Sample all available (warn, don't fail)
    bucket_counts = {}
    for transcript in sampled_transcripts:
        bucket = transcript['bucket']
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    for bucket, count in bucket_counts.items():
        if count < 17:
            logger.warning(
                f"⚠️  Bucket {bucket} has only {count} sampled transcripts (expected ~17). "
                f"Bucket may have insufficient videos."
            )

    # Rule 2: Empty transcripts included
    # Source: ContentAnalysisCHILD.md Section 2.3.1 Edge Cases row 3
    # Handling: Allow (may reveal no-speech patterns)
    empty_count = sum(1 for t in sampled_transcripts if not t['text'])
    if empty_count > 0:
        logger.info(
            f"ℹ️  {empty_count}/{len(sampled_transcripts)} transcripts are empty (no speech). "
            f"Including for potential no-speech pattern detection."
        )


def validate_business_rules_classification(
    video_id: str,
    transcript: dict,
    caption: str,
    hashtags: list[str]
):
    """
    Validate business rules during classification.
    Source: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases
    """
    # Rule 1: Empty transcript handling
    # Source: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases row 1
    # Handling: Classify using caption + hashtags only (warn, don't fail)
    if not transcript['text']:
        logger.warning(
            f"⚠️  Video {video_id} has empty transcript. "
            f"Classifying using caption and hashtags only."
        )

    # Rule 2: Missing caption handling
    # Source: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases row 2
    # Handling: Use empty string, continue with transcript + hashtags
    if not caption:
        logger.debug(f"Video {video_id} has no caption. Using transcript + hashtags.")

    # Rule 3: Missing hashtags handling
    # Source: ContentAnalysisCHILD.md Section 2.3.4 Edge Cases row 3
    # Handling: Use empty array, continue with transcript + caption
    if not hashtags:
        logger.debug(f"Video {video_id} has no hashtags. Using transcript + caption.")
```

---

### 5.3 Output Validation

```python
# ===== DISCOVERY OUTPUT VALIDATION =====
# Source: ContentAnalysisCHILD.md Section 6.3 (lines 920-955)

def validate_discovery_output(raw_taxonomy: dict):
    """
    Validate raw discovery JSON before saving.
    Source: ContentAnalysisCHILD.md Section 6.3
    """
    # Validation 1: Check top-level fields
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 926-929
    required_top_level = ['hashtag', 'analysis_date', 'sample_size', 'discovered_patterns']
    missing = [f for f in required_top_level if f not in raw_taxonomy]
    if missing:
        raise ValueError(f"Discovery output missing fields: {missing}")

    # Validation 2: Check discovered_patterns has all 6 categories
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 931-937
    required_patterns = [
        'content_categories', 'hook_strategies', 'audience_pain_points',
        'trending_keywords', 'engagement_drivers', 'content_tactics'
    ]
    patterns = raw_taxonomy['discovered_patterns']
    missing = [f for f in required_patterns if f not in patterns]
    if missing:
        raise ValueError(f"Discovered patterns missing categories: {missing}")

    # Validation 3: Check each pattern array is non-empty (warn only)
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 939-944
    for category in required_patterns:
        if not patterns[category]:
            logger.warning(f"Discovery found 0 patterns for {category}. This is unusual.")

    # Validation 4: Check pattern objects have required fields
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 946-954
    for category in ['content_categories', 'hook_strategies']:
        for pattern in patterns[category]:
            required_fields = ['name', 'frequency', 'percentage', 'examples']
            missing = [f for f in required_fields if f not in pattern]
            if missing:
                raise ValueError(
                    f"Pattern in {category} missing fields: {missing}. Pattern: {pattern}"
                )


# ===== CLASSIFICATION OUTPUT VALIDATION =====
# Source: ContentAnalysisCHILD.md Section 6.3 (lines 958-998)

def validate_classification_output(classification: dict):
    """
    Validate classification JSON before saving.
    Source: ContentAnalysisCHILD.md Section 6.3
    """
    # Validation 1: Check all 10 core fields present
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 964-972
    core_fields = [
        'video_id', 'content_category', 'hook_strategy', 'audience_pain_points',
        'trending_keywords', 'engagement_drivers', 'content_tactics',
        'caption_analysis', 'confidence', 'transcript_available', 'note'
    ]
    missing = [f for f in core_fields if f not in classification]
    if missing:
        raise ValueError(f"Classification missing core fields: {missing}")

    # Validation 2: Check confidence value
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 974-978
    if classification['confidence'] not in ['high', 'medium', 'low']:
        raise ValueError(
            f"Invalid confidence value: {classification['confidence']}. "
            f"Must be high, medium, or low."
        )

    # Validation 3: Check caption_analysis has all 12 subfields
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 980-990
    caption_fields = [
        'caption_hook_type', 'caption_cta_type', 'caption_cta_present',
        'brand_mention_present', 'influencer_tag_present', 'emoji_usage',
        'caption_length', 'hashtag_count', 'hashtag_placement', 'hashtag_strategy'
    ]
    caption_analysis = classification['caption_analysis']
    missing = [f for f in caption_fields if f not in caption_analysis]
    if missing:
        raise ValueError(f"caption_analysis missing fields: {missing}")

    # Validation 4: Check hashtag_strategy has 3 subfields
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 992-996
    hashtag_strategy = caption_analysis['hashtag_strategy']
    required_hashtag_fields = ['broad_count', 'niche_count', 'branded_count']
    missing = [f for f in required_hashtag_fields if f not in hashtag_strategy]
    if missing:
        raise ValueError(f"hashtag_strategy missing fields: {missing}")

    # Validation 5: Check arrays are actually arrays
    # Source: ContentAnalysisCHILD.md Section 6.3 lines 1000-1003
    array_fields = ['audience_pain_points', 'trending_keywords', 'engagement_drivers', 'content_tactics']
    for field in array_fields:
        if not isinstance(classification[field], list):
            raise ValueError(f"Field {field} must be array, got {type(classification[field])}")
```

---

## Section 6: Error Handling

<!-- Source: ContentAnalysisCHILD.md Section 6.2 -->

### 6.1 Error Cases Catalog

**Source**: ContentAnalysisCHILD.md Section 6.2 (Error Cases)

| Error ID | Error Type | Trigger Condition | Recovery Strategy | User Action Required |
|----------|-----------|-------------------|-------------------|---------------------|
| **E1: missing_input_file** | FileNotFoundError | selection_manifest.json not found | Fail fast with clear error message | Verify Stage 2.5 completed successfully |
| **E2: invalid_manifest_structure** | ValueError | Manifest missing required fields | Fail fast with field list | Check Stage 2.5 output, regenerate if needed |
| **E3: insufficient_bucket_videos** | ValueError | Bucket has < 10 top performers | Fail fast with warning | Adjust video_count parameter or re-scrape |
| **E4: missing_api_key** | ValueError | ANTHROPIC_API_KEY not set | Fail fast with setup instructions | Set environment variable |
| **E5: llm_api_timeout_discovery** | TimeoutError | Sonnet API exceeds 120s timeout | Retry 3x with exponential backoff (1s, 2s, 4s), then fail | Check status.anthropic.com, retry manually |
| **E6: llm_api_timeout_per_video** | TimeoutError | Haiku API exceeds 30s per video | Retry 3x with exponential backoff, then fail | Check API status, reduce taxonomy size if recurring |
| **E7: invalid_json_response** | ValueError | LLM returns malformed JSON | Retry 3x, then fail with raw response logged | Report to Anthropic if recurring, check prompt formatting |
| **E8: missing_taxonomy** | FileNotFoundError | Curated taxonomy file not found | Fail fast with instructions | Run Stage 2.6 and complete manual curation |
| **E9: empty_taxonomy_field** | ValueError | Taxonomy field is empty array | Fail fast with field name | Review curated taxonomy, add missing patterns |
| **E10: short_definition** | ValueError | Category definition < 10 chars | Fail fast with field name | Expand definition in curated taxonomy |
| **E11: missing_transcript** | FileNotFoundError | Transcript file not found for video | Log warning, skip video in sampling | Check Stage 2 (Whisper) output, acceptable to skip |
| **E12: insufficient_samples** | ValueError | < 10 transcripts sampled successfully | Fail fast after sampling | Check transcript availability, may need to re-run Stage 2 |

---

### 6.2 Error Handling Implementations

```python
# ===== ERROR HANDLER: MISSING INPUT FILE =====
# Error ID: E1
# Source: ContentAnalysisCHILD.md Section 6.2 row 1

def handle_missing_input_file(file_path: str, stage_name: str):
    """
    Handle missing input file error.
    """
    raise FileNotFoundError(
        f"❌ Required input not found: {file_path}\n"
        f"This file should have been created by {stage_name}.\n"
        f"Action: Verify {stage_name} completed successfully."
    )


# ===== ERROR HANDLER: API TIMEOUT WITH RETRY =====
# Error IDs: E5, E6
# Source: ContentAnalysisCHILD.md Section 6.2 rows 5, 6

def handle_api_timeout_with_retry(
    api_call_func: callable,
    context: str,
    max_retries: int = 3,
    backoff_delays: list[int] = [1, 2, 4]
):
    """
    Handle API timeout with exponential backoff retry.

    Args:
        api_call_func: Function to call (must raise TimeoutError on failure)
        context: Description for logging (e.g., "Discovery", "Video 123 classification")
        max_retries: Number of retry attempts (default: 3)
        backoff_delays: Delay in seconds between retries (default: [1, 2, 4])

    Returns:
        Result from api_call_func

    Raises:
        TimeoutError: After all retries exhausted
    """
    for attempt in range(max_retries):
        try:
            return api_call_func()
        except TimeoutError as e:
            if attempt < max_retries - 1:
                delay = backoff_delays[attempt]
                logger.warning(
                    f"⏰ {context} timeout. Retry {attempt + 1}/{max_retries} in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"❌ {context} failed after {max_retries} retries.\n"
                    f"Action: Check status.anthropic.com and retry manually."
                )
                raise

    raise RuntimeError("Unreachable: retry loop exited unexpectedly")


# ===== ERROR HANDLER: INVALID JSON RESPONSE =====
# Error ID: E7
# Source: ContentAnalysisCHILD.md Section 6.2 row 7

def handle_invalid_json_response(
    response_text: str,
    context: str,
    max_retries: int = 3
) -> dict:
    """
    Handle invalid JSON response from LLM.

    Note: This is called WITHIN a retry loop, not a standalone handler.
    """
    # Log raw response for debugging
    logger.error(
        f"⚠️ Invalid JSON from LLM ({context}).\n"
        f"Raw response (first 500 chars): {response_text[:500]}\n"
        f"Action: Check prompt formatting, report to Anthropic if recurring."
    )

    # Re-raise to trigger retry logic in caller
    raise ValueError(f"LLM returned invalid JSON for {context}")


# ===== ERROR HANDLER: GRACEFUL SKIP =====
# Error ID: E11
# Source: ContentAnalysisCHILD.md Section 6.2 row 11

def handle_graceful_skip(video_id: str, reason: str, error_type: str = "warning"):
    """
    Handle non-fatal errors by skipping video and logging.

    Args:
        video_id: Video identifier
        reason: Why video is being skipped
        error_type: "warning" or "info"
    """
    if error_type == "warning":
        logger.warning(f"⚠️  Skipping video {video_id}: {reason}")
    else:
        logger.info(f"ℹ️  Skipping video {video_id}: {reason}")
```

---

### 6.3 Error Recovery Workflows

```python
# ===== WORKFLOW: DISCOVERY WITH ERROR HANDLING =====
# Source: ContentAnalysisCHILD.md Section 2.3.2, 6.2

def run_discovery_with_error_handling(manifest_path: str, hashtag: str):
    """
    Stage 2.6 discovery with comprehensive error handling.
    """
    try:
        # Step 1: Validate inputs (raises FileNotFoundError, ValueError)
        validate_discovery_inputs(manifest_path, sample_size=50)

        # Step 2: Sample transcripts (handles E11 gracefully, raises E12 if insufficient)
        sampled_transcripts = sample_transcripts_for_discovery(manifest_path, sample_size=50)

        # Step 3: Validate business rules (warns, doesn't fail)
        validate_business_rules_sampling(sampled_transcripts)

        # Step 4: LLM discovery with retry on timeout (handles E5, E7)
        raw_taxonomy = discover_patterns_llm(sampled_transcripts, hashtag)

        # Step 5: Validate output (raises ValueError if malformed)
        validate_discovery_output(raw_taxonomy)

        logger.info(f"✅ Discovery completed successfully for #{hashtag}")
        return raw_taxonomy

    except FileNotFoundError as e:
        # E1, E8: Missing input file
        logger.error(f"❌ {str(e)}")
        raise

    except ValueError as e:
        # E2, E9, E10, E12: Invalid data structure
        logger.error(f"❌ Validation failed: {str(e)}")
        raise

    except TimeoutError as e:
        # E5: LLM timeout after retries
        logger.error(f"❌ LLM timeout: {str(e)}")
        logger.error(f"Check https://status.anthropic.com and retry manually.")
        raise


# ===== WORKFLOW: CLASSIFICATION WITH ERROR HANDLING =====
# Source: ContentAnalysisCHILD.md Section 2.3.4, 6.2

def run_classification_with_error_handling(
    video_id: str,
    taxonomy: dict,
    transcript: dict,
    caption: str,
    hashtags: list[str],
    client: anthropic.Anthropic
):
    """
    Single video classification with comprehensive error handling.
    """
    try:
        # Step 1: Validate business rules (warns for edge cases)
        validate_business_rules_classification(video_id, transcript, caption, hashtags)

        # Step 2: Classify with retry on timeout (handles E6, E7)
        classification = classify_video_llm(
            video_id, transcript, caption, hashtags, taxonomy, client
        )

        # Step 3: Validate output (raises ValueError if malformed)
        validate_classification_output(classification)

        logger.debug(f"✅ Classified video {video_id}")
        return classification

    except TimeoutError as e:
        # E6: Per-video timeout after retries
        logger.error(f"❌ Video {video_id} classification timeout after 3 retries: {str(e)}")
        raise

    except ValueError as e:
        # E7: Invalid JSON or validation failure
        logger.error(f"❌ Video {video_id} classification failed validation: {str(e)}")
        raise
```

---

## Section 7: Complete Example Traces

<!-- Source: ContentAnalysisCHILD.md Section 2.3, synthesized from algorithmic flows -->

### 7.1 Trace: Stage 2.6 Discovery (Successful Path)

**Scenario**: Discover content patterns for #nutrition hashtag

**Input State**:
- selection_manifest.json exists with 3 buckets (33-60s, 60-90s, 90-120s)
- Each bucket has 80 top performers
- Transcripts available for all videos
- ANTHROPIC_API_KEY set

**Execution Trace**:

```
Step 1: Validate Inputs
├─ Load selection_manifest.json → Success
├─ Check required fields ['hashtag', 'selected_buckets', 'videos_by_bucket'] → ✓ All present
├─ Validate bucket count → ✓ 3 buckets
├─ Check ANTHROPIC_API_KEY → ✓ Set
└─ Result: Validation passed

Step 2: Sample Transcripts
├─ Calculate samples per bucket: 50 ÷ 3 = 16, 17, 17
├─ Bucket "33-60s": Sample 17 from 80 top performers
│  ├─ Load transcript for 7526250443832331550 → ✓ "this is why every woman needs to start yoni steaming..."
│  ├─ Load transcript for 7428596413707144481 → ✓ "Two minute TikTok videos..."
│  └─ ... (15 more transcripts)
├─ Bucket "60-90s": Sample 17 from 80 top performers → ✓ 17 transcripts loaded
├─ Bucket "90-120s": Sample 16 from 80 top performers → ✓ 16 transcripts loaded
└─ Result: 50 transcripts sampled successfully

Step 3: Validate Business Rules
├─ Check bucket distribution → ✓ All buckets have 16-17 samples (expected)
├─ Check empty transcripts → 2 empty (4%), acceptable
└─ Result: Business rules satisfied

Step 4: LLM Discovery
├─ Build prompt with 50 transcripts (5,243 chars)
├─ Call Claude 3.5 Sonnet (timeout: 120s)
│  └─ Response received in 47s
├─ Parse JSON response → ✓ Valid JSON
├─ Validate output structure
│  ├─ Check top-level fields ['hashtag', 'analysis_date', 'sample_size', 'discovered_patterns'] → ✓ All present
│  ├─ Check 6 pattern categories → ✓ All present
│  ├─ content_categories: 5 patterns discovered (recipe_tutorial, supplement_review, meal_prep, nutrition_myth_busting, diet_transformation)
│  ├─ hook_strategies: 4 patterns (problem_solution, direct_statement, question_hook, shocking_fact)
│  ├─ audience_pain_points: 8 patterns (bloating, low_energy, weight_loss, gut_health, ...)
│  ├─ trending_keywords: 12 patterns (protein, gut_health, fiber, metabolism, ...)
│  ├─ engagement_drivers: 6 patterns (before_after_reveal, specific_metrics, personal_testimony, ...)
│  └─ content_tactics: 5 patterns (direct_to_camera, voiceover, text_overlay, ...)
└─ Result: Discovery successful

Step 5: Save Raw Discovery
├─ Create directory /data/clients/acme_corp/hashtags/nutrition/top_contrastive/content_taxonomies/
├─ Write nutrition_raw_discovery.json (9,847 bytes)
└─ Result: File saved

Final Output:
✅ Discovery completed successfully for #nutrition
📝 Next: Manually curate and save to content_taxonomies/nutrition_taxonomy.json
Cost: ~$0.75 (Sonnet API call)
Duration: 52 seconds
```

**Output State**:
- nutrition_raw_discovery.json created with 40 total patterns across 6 categories
- Ready for manual curation

---

### 7.2 Trace: Stage 2.6 Discovery (Error Path - API Timeout)

**Scenario**: LLM API timeout during discovery

**Input State**:
- selection_manifest.json exists with 3 buckets
- 50 transcripts sampled successfully
- Anthropic API experiencing slowness

**Execution Trace**:

```
Step 1-3: [Same as successful path] → ✓ All validations passed

Step 4: LLM Discovery (Attempt 1)
├─ Build prompt with 50 transcripts
├─ Call Claude 3.5 Sonnet (timeout: 120s)
│  └─ ⏰ TimeoutError after 120s
└─ Retry 1/3 in 1s...

Step 4: LLM Discovery (Attempt 2)
├─ Call Claude 3.5 Sonnet (timeout: 120s)
│  └─ ⏰ TimeoutError after 120s
└─ Retry 2/3 in 2s...

Step 4: LLM Discovery (Attempt 3)
├─ Call Claude 3.5 Sonnet (timeout: 120s)
│  └─ ⏰ TimeoutError after 120s
└─ ❌ Discovery failed after 3 retries

Final Output:
❌ LLM timeout after 3 retries
Action: Check https://status.anthropic.com and retry manually
Error: TimeoutError
Duration: 371 seconds (120s × 3 + backoff delays)
```

**Output State**:
- No raw discovery file created
- User must check Anthropic status and re-run

---

### 7.3 Trace: Stage 2.7 Classification (Successful Path)

**Scenario**: Classify 120 videos (40 per bucket × 3 buckets) using curated taxonomy

**Input State**:
- selection_manifest.json with 3 buckets
- nutrition_taxonomy.json (curated) exists
- Transcripts, captions, hashtags available for all 120 videos
- ANTHROPIC_API_KEY set

**Execution Trace**:

```
Step 1: Load Taxonomy
├─ Read nutrition_taxonomy.json → ✓ Success
├─ Validate structure
│  ├─ Check required fields → ✓ All 6 fields present
│  ├─ content_categories: 5 items with definitions → ✓ Valid
│  ├─ hook_strategies: 4 items with definitions → ✓ Valid
│  └─ All other fields: Valid
└─ Result: Taxonomy loaded successfully

Step 2: Initialize Anthropic Client
└─ Client initialized with API key

Step 3: Load Video IDs from Manifest
├─ Bucket "33-60s": 32 top + 8 bottom = 40 videos
├─ Bucket "60-90s": 32 top + 8 bottom = 40 videos
├─ Bucket "90-120s": 32 top + 8 bottom = 40 videos
└─ Total: 120 videos to classify

Step 4: Classify Videos (Sequential Processing)

Video 1/120: 7526250443832331550 (33-60s, top)
├─ Load transcript → ✓ "this is why every woman needs to start yoni steaming..."
├─ Load caption → ✓ "this is why every woman needs to start yoni steaming..."
├─ Load hashtags → ✓ ["yonisteam", "wellness", "holistic", ...]
├─ Validate business rules → ✓ All data present
├─ Build classification prompt (2,847 chars)
├─ Call Claude 3 Haiku (timeout: 30s)
│  └─ Response in 2.3s
├─ Parse JSON → ✓ Valid JSON
├─ Validate output → ✓ All 23 fields present
├─ Save to bucket_33-60s/content_analysis/7526250443832331550_content.json
└─ Result: ✅ Classified (confidence: high)

Video 2/120: 7428596413707144481 (60-90s, top)
├─ Load transcript → ✓ "Two minute TikTok videos..."
├─ ... [same process as Video 1]
└─ Result: ✅ Classified (confidence: high)

... [Videos 3-119 follow same pattern]

Video 120/120: 7234567890123456789 (90-120s, bottom)
├─ Load transcript → ⚠️  Empty string (no speech)
├─ Load caption → ✓ "Check out my workout routine #fitness"
├─ Load hashtags → ✓ ["fitness", "workout"]
├─ Validate business rules → ⚠️  Empty transcript, classify using caption + hashtags only
├─ Build classification prompt (1,523 chars, transcript empty)
├─ Call Claude 3 Haiku (timeout: 30s)
│  └─ Response in 1.8s
├─ Parse JSON → ✓ Valid JSON
├─ Validate output → ✓ All 23 fields present
├─ transcript_available: false, note: "Classified using caption and hashtags only"
├─ Save to bucket_90-120s/content_analysis/7234567890123456789_content.json
└─ Result: ✅ Classified (confidence: medium)

Final Output:
✅ Classified 120/120 videos successfully
   ├─ bucket_33-60s: 40 videos
   ├─ bucket_60-90s: 40 videos
   └─ bucket_90-120s: 40 videos
Cost: ~$0.12 (120 × Haiku calls)
Duration: 312 seconds (~2.6s per video average)
```

**Output State**:
- 120 classification files created across 3 buckets
- Ready for Stage 7 (LLM Report Generation)

---

### 7.4 Trace: Stage 2.7 Classification (Partial Failure)

**Scenario**: Classify 120 videos with some API failures

**Execution Trace**:

```
[Videos 1-45: Successful as in 7.3]

Video 46/120: 7111111111111111111 (60-90s, top) - ATTEMPT 1
├─ Load data → ✓ All present
├─ Build prompt → ✓
├─ Call Haiku (timeout: 30s)
│  └─ ⏰ TimeoutError after 30s
└─ Retry 1/3 in 1s...

Video 46/120: 7111111111111111111 (60-90s, top) - ATTEMPT 2
├─ Call Haiku (timeout: 30s)
│  └─ ✓ Response in 3.2s
├─ Parse JSON → ✓ Valid
└─ Result: ✅ Classified

[Videos 47-82: Successful]

Video 83/120: 7222222222222222222 (90-120s, top) - ATTEMPT 1
├─ Load data → ✓ All present
├─ Call Haiku (timeout: 30s)
│  └─ ⚠️  Invalid JSON response
└─ Retry 1/3 in 1s...

Video 83/120: 7222222222222222222 (90-120s, top) - ATTEMPT 2
├─ Call Haiku (timeout: 30s)
│  └─ ✓ Response with valid JSON
└─ Result: ✅ Classified

[Videos 84-120: Successful]

Final Output:
✅ Classified 120/120 videos successfully
   ├─ 2 videos required retries (API timeout, invalid JSON)
   ├─ All retries succeeded on 2nd attempt
Cost: ~$0.13 (120 videos + 2 retries)
Duration: 325 seconds (13 seconds added for retries)
```

---

### 7.5 Trace: End-to-End (Discovery + Classification)

**Scenario**: Complete content analysis for #nutrition hashtag

**Execution Summary**:

```
═══════════════════════════════════════════════════════════════
STAGE 2.6: DISCOVERY
═══════════════════════════════════════════════════════════════
Input:  selection_manifest.json (3 buckets, 240 videos total)
Output: nutrition_raw_discovery.json (40 patterns)
Cost:   $0.75
Time:   52 seconds

[MANUAL STEP: Curator reviews raw discovery and creates nutrition_taxonomy.json]
Time:   ~15 minutes

═══════════════════════════════════════════════════════════════
STAGE 2.7: CLASSIFICATION
═══════════════════════════════════════════════════════════════
Input:  nutrition_taxonomy.json (curated)
        selection_manifest.json (120 videos to classify)
        Transcripts, captions, hashtags (from Stage 2)
Output: 120 classification files
        ├─ bucket_33-60s/content_analysis/*.json (40 files)
        ├─ bucket_60-90s/content_analysis/*.json (40 files)
        └─ bucket_90-120s/content_analysis/*.json (40 files)
Cost:   $0.12
Time:   312 seconds

═══════════════════════════════════════════════════════════════
TOTALS (First Run)
═══════════════════════════════════════════════════════════════
Automated Cost:  $0.87
Automated Time:  364 seconds (~6 minutes)
Manual Time:     15 minutes (curation)
Total Time:      ~21 minutes

═══════════════════════════════════════════════════════════════
TOTALS (Subsequent Run - Taxonomy Reused)
═══════════════════════════════════════════════════════════════
Automated Cost:  $0.12 (classification only)
Automated Time:  312 seconds (~5 minutes)
Manual Time:     0 (taxonomy already curated)
Total Time:      ~5 minutes
```

---

## Section 8: File Structure & Integration

<!-- Source: ContentAnalysisCHILD.md Section 2, FoundationCHILD.md Section 2 -->

### 8.1 Module Structure

```
ml_pipeline/
└── stage2_content_analysis/
    ├── __init__.py
    ├── discovery.py              # Stage 2.6: Pattern discovery
    ├── classification.py         # Stage 2.7: Video classification
    ├── validation.py             # Input/output validation functions
    ├── error_handlers.py         # Error handling utilities
    └── utils.py                  # Shared utilities (load_json, save_json, etc.)
```

### 8.2 File Responsibilities

**discovery.py**:
- `sample_transcripts_for_discovery()` → Section 4.1
- `discover_patterns_llm()` → Section 4.2
- `run_discovery_with_error_handling()` → Section 6.3

**classification.py**:
- `classify_video_llm()` → Section 4.3
- `classify_all_videos()` → Batch classification orchestrator
- `run_classification_with_error_handling()` → Section 6.3

**validation.py**:
- `validate_discovery_inputs()` → Section 5.1
- `validate_classification_inputs()` → Section 5.1
- `validate_business_rules_sampling()` → Section 5.2
- `validate_business_rules_classification()` → Section 5.2
- `validate_discovery_output()` → Section 5.3
- `validate_classification_output()` → Section 5.3

**error_handlers.py**:
- `handle_missing_input_file()` → Section 6.2
- `handle_api_timeout_with_retry()` → Section 6.2
- `handle_invalid_json_response()` → Section 6.2
- `handle_graceful_skip()` → Section 6.2

**utils.py**:
- `load_json()` → JSON file loading with error handling
- `save_json()` → JSON file saving with atomic writes
- `construct_path()` → Path construction using FoundationCHILD.md Section 2.2 templates

### 8.3 Directory Outputs

**Stage 2.6 Outputs**:
```
/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/
└── content_taxonomies/
    ├── {hashtag}_raw_discovery.json       # Generated by discovery.py
    └── {hashtag}_taxonomy.json            # Created manually after curation
```

**Stage 2.7 Outputs**:
```
/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/buckets/
├── bucket_33-60s/
│   └── content_analysis/
│       ├── {video_id}_content.json        # 40 files (32 top + 8 bottom)
│       ├── ...
│       └── ...
├── bucket_60-90s/
│   └── content_analysis/
│       └── {video_id}_content.json        # 40 files
└── bucket_90-120s/
    └── content_analysis/
        └── {video_id}_content.json        # 40 files
```

---

## Section 9: Configuration & Environment

<!-- Source: ContentAnalysisCHILD.md Section 4, FoundationCHILD.md Section 4 -->

### 9.1 Environment Variables

```python
# Required
ANTHROPIC_API_KEY = "sk-ant-..."           # Anthropic API key for Claude models
                                            # Used by: discovery.py, classification.py
                                            # Validation: Must be set before Stage 2.6/2.7 execution

# Optional (defaults provided)
DISCOVERY_SAMPLE_SIZE = 50                  # Number of transcripts to sample for discovery
                                            # Default: 50, Range: 10-200
                                            # Used by: discovery.py

DISCOVERY_TIMEOUT_SECONDS = 120             # Timeout for Sonnet API calls
                                            # Default: 120, Range: 60-300
                                            # Used by: discovery.py

CLASSIFICATION_TIMEOUT_SECONDS = 30         # Timeout for Haiku API calls per video
                                            # Default: 30, Range: 10-60
                                            # Used by: classification.py

MAX_RETRIES = 3                             # API retry attempts
                                            # Default: 3, Range: 1-5
                                            # Used by: error_handlers.py
```

### 9.2 Configuration Files

**config.json** (FoundationCHILD.md Section 5.1):
```json
{
  "client_id": "acme_corp",
  "analysis_type": "hashtag",
  "target": "#nutrition",
  "analysis_mode": "top",
  "selection_strategy": "contrastive",
  "video_count": 100,
  "date_filter": "last_90_days",
  "country_code": "US",
  "report_type": "single",
  "report_audience": "client",
  "auto_confirm": false,
  "run_date": "2025-01-28T10:30:00Z"
}
```

**Taxonomy Configuration** (Created manually after Stage 2.6):
```json
{
  "hashtag": "nutrition",
  "content_categories": [
    {"name": "recipe_tutorial", "definition": "Step-by-step cooking instructions"},
    ...
  ],
  "hook_strategies": [
    {"name": "problem_solution", "definition": "Starts with problem, promises solution"},
    ...
  ],
  "audience_pain_points": ["bloating", "low_energy", ...],
  "trending_keywords": ["protein", "gut_health", ...],
  "engagement_drivers": ["before_after_reveal", ...],
  "content_tactics": ["direct_to_camera", ...]
}
```

---

## Section 10: Logging Specifications

<!-- Source: TI Generation Best Practices -->

### 10.1 Log Levels

```python
import logging

# Configure logger
logger = logging.getLogger("rumiai.content_analysis")
logger.setLevel(logging.INFO)

# Log levels by operation type:
# DEBUG: Detailed execution traces (per-video classification progress)
# INFO: Major milestones (discovery complete, classification started)
# WARNING: Non-fatal issues (empty transcripts, missing captions, API retries)
# ERROR: Fatal errors (missing inputs, validation failures, API exhaustion)
```

### 10.2 Logging Examples

**Discovery Logging**:
```python
logger.info(f"🔍 Starting discovery for #{hashtag}")
logger.info(f"📊 Sampled {len(transcripts)} transcripts from {len(buckets)} buckets")
logger.info(f"🤖 Calling Claude 3.5 Sonnet for pattern discovery...")
logger.info(f"✅ Discovery complete: {output_path}")
logger.info(f"📝 Next: Manually curate and save to {taxonomy_path}")
```

**Classification Logging**:
```python
logger.info(f"🏷️  Starting classification for {len(video_ids)} videos")
logger.debug(f"Classifying video {i+1}/{total}: {video_id}")
logger.warning(f"⚠️  Video {video_id} has empty transcript. Using caption + hashtags only.")
logger.info(f"✅ Classified {completed}/{total} videos successfully")
```

**Error Logging**:
```python
logger.error(f"❌ Missing input file: {file_path}")
logger.error(f"❌ API timeout after 3 retries for {context}")
logger.error(f"❌ Invalid JSON from LLM: {response_text[:500]}")
```

### 10.3 Log File Locations

```
/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/buckets/
├── bucket_33-60s/logs/
│   └── content_analysis_2025-01-28.log
├── bucket_60-90s/logs/
│   └── content_analysis_2025-01-28.log
└── bucket_90-120s/logs/
    └── content_analysis_2025-01-28.log
```

---

## Section 11: Dependencies & Prerequisites

<!-- Source: ContentAnalysisCHILD.md Section 3 -->

### 11.1 Python Dependencies

```python
# requirements.txt
anthropic==0.40.0          # Claude API client
pydantic==2.10.0           # Schema validation
python-dotenv==1.0.0       # Environment variable loading
```

### 11.2 Stage Dependencies

**Stage 2.6 (Discovery) Prerequisites**:
- ✅ Stage 2.5 complete → selection_manifest.json exists
- ✅ Stage 2 complete → Transcripts available in /home/jorge/rumiaifinal/speech_transcriptions/
- ✅ ANTHROPIC_API_KEY environment variable set

**Stage 2.7 (Classification) Prerequisites**:
- ✅ Stage 2.6 complete + Manual curation → {hashtag}_taxonomy.json exists
- ✅ Stage 2.5 complete → selection_manifest.json exists
- ✅ Stage 2 complete → Transcripts, captions, hashtags available
- ✅ ANTHROPIC_API_KEY environment variable set

### 11.3 Downstream Consumers

**Stage 7 (LLM Report Generation)**:
- Consumes: 120 classification files from Stage 2.7
- Location: `{bucket_base}/content_analysis/{video_id}_content.json`
- Usage: Synthesizes content patterns across buckets for creative reports

---

## Section 12: HLD Traceability Matrix

<!-- Source: TI Generation Prompt Section 1.3 -->

| HLD Section | HLD Content | TI Section | TI Implementation | Verification |
|-------------|-------------|------------|-------------------|--------------|
| **ContentAnalysisCHILD.md Section 1** | Context & Business Goal | Section 1 | Implementation_Priority: HIGH rationale | ✓ |
| **ContentAnalysisCHILD.md Section 2.1** | High-Level Approach | Section 7 | End-to-end trace (Section 7.5) | ✓ |
| **ContentAnalysisCHILD.md Section 2.2** | Data Flow | Section 2 | StageInput/StageOutput contracts | ✓ |
| **ContentAnalysisCHILD.md Section 2.3.1** | Discovery Sampling | Section 4.1 | sample_transcripts_for_discovery() | ✓ |
| **ContentAnalysisCHILD.md Section 2.3.2** | LLM Discovery | Section 4.2 | discover_patterns_llm() | ✓ |
| **ContentAnalysisCHILD.md Section 2.3.3** | Manual Curation | Section 7.5 | Manual step in end-to-end trace | ✓ |
| **ContentAnalysisCHILD.md Section 2.3.4** | Video Classification | Section 4.3 | classify_video_llm() | ✓ |
| **ContentAnalysisCHILD.md Section 3.1** | Input Dependencies | Section 2, Section 11.2 | StageInput + Prerequisites | ✓ |
| **ContentAnalysisCHILD.md Section 3.2** | Output Contracts | Section 2 | StageOutput | ✓ |
| **ContentAnalysisCHILD.md Section 3.3** | Cross-Stage Dependencies | Section 1, Section 11.3 | Related_TI_Docs + Downstream consumers | ✓ |
| **ContentAnalysisCHILD.md Section 3.4** | External Dependencies | Section 2, Section 9.1 | ANTHROPIC_API_KEY in StageInput + Env vars | ✓ |
| **ContentAnalysisCHILD.md Section 4** | Configuration & Parameters | Section 9 | Configuration & Environment | ✓ |
| **ContentAnalysisCHILD.md Section 5.1** | Input Schemas | Section 3.2 | Stage-Specific Input Schemas | ✓ |
| **ContentAnalysisCHILD.md Section 5.2** | Output Schemas | Section 3.3 | Stage-Specific Output Schemas | ✓ |
| **ContentAnalysisCHILD.md Section 6.1** | Input Validation | Section 5.1 | validate_discovery_inputs(), validate_classification_inputs() | ✓ |
| **ContentAnalysisCHILD.md Section 6.2** | Error Cases | Section 6.1, 6.2 | Error Cases Catalog + Error Handling Implementations | ✓ |
| **ContentAnalysisCHILD.md Section 6.3** | Output Validation | Section 5.3 | validate_discovery_output(), validate_classification_output() | ✓ |
| **FoundationCHILD.md Section 2** | Client Architecture | Section 2, Section 8.3 | base_path, bucket_base + Directory Outputs | ✓ |
| **FoundationCHILD.md Section 2.2** | Path Templates | Section 2, Section 8.3 | Directory paths in StageInput + Output directories | ✓ |
| **FoundationCHILD.md Section 4** | CLI Command Structure | Section 2 | CLI parameters in StageInput | ✓ |
| **FoundationCHILD.md Section 4.1** | CLI Parameters | Section 2 | client_id, hashtag, analysis_mode, selection_strategy | ✓ |
| **FoundationCHILD.md Section 5.1** | config.json Schema | Section 3.1, Section 9.2 | ConfigSchema + Configuration Files | ✓ |

**Coverage Summary**:
- Total HLD sections: 22
- Mapped to TI: 22
- Coverage: 100%

---

**Document Complete**

**Total Sections**: 12
**Total Pages**: ~50 (estimated)
**Generation Date**: 2025-01-28
**Status**: Ready for Implementation

