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
    transcripts_dir: str        # {RUMIAI_ROOT}/speech_transcriptions/
                                # Default: /home/jorge/rumiaifinal/speech_transcriptions/
                                # Override via RUMIAI_ROOT env var (Section 9.1)
                                # Absolute path, not in client directory
                                # Files: {video_id}_whisper.json
                                # Schema: ContentAnalysisCHILD.md Section 5.1.2

    # Input 3: Captions & Hashtags (from Stage 2 - unified_analysis)
    unified_analysis_dir: str   # {RUMIAI_ROOT}/unified_analysis/
                                # Default: /home/jorge/rumiaifinal/unified_analysis/
                                # Override via RUMIAI_ROOT env var (Section 9.1)
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
                                     # Fields: 12 total

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

    # ===== FILE OWNERSHIP & LIFECYCLE =====
    # Source: m11 Enhancement - Clarify output ownership

    # Stage 2.6 owns:
    #   - {hashtag}_raw_discovery.json (created, never modified after creation)
    #   - {hashtag}_taxonomy.json (created after manual curation, versioned)

    # Stage 2.7 owns:
    #   - {video_id}_content.json (created, may be overwritten on re-run)
    #   - classification_checkpoint.json (created, updated during classification)

    # Downstream consumers (Stage 7):
    #   - Reads classification files (read-only access)
    #   - Does NOT modify Stage 2.6/2.7 outputs

    # Lifecycle:
    #   - Files persist until full pipeline completes
    #   - Manual deletion required for cleanup (no auto-deletion)
    #   - Re-running Stage 2.7 overwrites classification files (use checkpoint to resume)
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

    # percentage removed - calculated by Python post-processing (see Section 4.2.5)

    "examples": list[str],         # Required, 2-3 example phrases
                                   # Example: ["protein smoothie recipe", "meal prep tutorial"]
                                   # Source: ContentAnalysisCHILD.md Section 5.2.1

    "representative_video_ids": list[str],  # Required, video IDs showing this pattern
                                            # Example: ["7526250443832331550", "7428596413707144481"]
                                            # Source: ContentAnalysisCHILD.md Section 5.2.1
}

# ===== OUTPUT SCHEMA 2: VIDEO CLASSIFICATION (from ContentAnalysisCHILD.md Section 5.2.2) =====
# Updated: 2025-01-28 - Reflects refined schema from 2.7ClassificationCritique.md

VideoClassificationSchema = {
    "video_id": str,               # Required, video identifier
                                   # Example: "7526250443832331550"
                                   # Source: 2.7ClassificationCritique.md Section 9

    "taxonomy_version": str,       # Required, always "stage2.6_output"
                                   # Links classification to taxonomy source
                                   # Example: "stage2.6_output"
                                   # Source: 2.7ClassificationCritique.md Section 9

    "content_category": str,       # Required, from taxonomy
                                   # Primary content type
                                   # Example: "wellness_practice"
                                   # Source: 2.7ClassificationCritique.md Section 3

    "hook_strategy": str,          # Required, from taxonomy
                                   # Opening pattern
                                   # Example: "personal_story"
                                   # Source: 2.7ClassificationCritique.md Section 3

    "pain_points": list[str],      # Required (can be empty array), from taxonomy
                                   # Detected pain points
                                   # Example: ["gut_health", "bloating"]
                                   # Source: 2.7ClassificationCritique.md Section 4
                                   # Note: Renamed from audience_pain_points in refined schema

    "keywords": list[str],         # Required (can be empty array), from taxonomy
                                   # Detected keywords
                                   # Example: ["probiotics", "gut_protocol"]
                                   # Source: 2.7ClassificationCritique.md Section 4
                                   # Note: Renamed from trending_keywords in refined schema

    "engagement_drivers": list[str],  # Required (can be empty array), from taxonomy
                                      # Shareability tactics
                                      # Example: ["personal_story", "before_after"]
                                      # Source: 2.7ClassificationCritique.md Section 4

    "content_tactics": list[str],     # Required (can be empty array), from taxonomy
                                      # Presentation styles
                                      # Example: ["direct_address"]
                                      # Source: 2.7ClassificationCritique.md Section 4

    "caption_analysis": dict,      # Required, caption-specific analysis (8 subfields)
                                   # See nested schema below
                                   # Source: 2.7ClassificationCritique.md Section 6
                                   # Note: Simplified from 13 to 8 subfields in refined schema

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
# Updated: 2025-01-28 - Reflects refined schema from 2.7ClassificationCritique.md Section 6
CaptionAnalysisSchema = {
    "hook_type": str,              # Required, ["statement", "question", "command", "teaser"]
                                   # How caption opens (simplified from 6 to 4 types)
                                   # Source: 2.7ClassificationCritique.md Section 6
                                   # Note: Removed "statistic" and "contradiction" types

    "cta_type": str,               # Required, ["link_in_bio", "save_post", "comment", "follow", "share", "tag_friend", "none"]
                                   # Call-to-action type
                                   # Source: 2.7ClassificationCritique.md Section 6
                                   # Note: Renamed from "caption_cta_type"

    "brand_mention_present": bool, # Required
                                   # Whether brand/product mentioned
                                   # Source: 2.7ClassificationCritique.md Section 6

    "influencer_tag_present": bool,# Required
                                   # Whether influencer tagged
                                   # Source: 2.7ClassificationCritique.md Section 6

    "emoji_usage": str,            # Required, ["none", "some", "many"]
                                   # Emoji density (simplified from 4 to 3 levels)
                                   # Source: 2.7ClassificationCritique.md Section 6
                                   # Note: "none" (0), "some" (1-4), "many" (5+)

    "caption_length": str,         # Required, ["short", "long"]
                                   # Caption length category (simplified from 3 to 2 levels)
                                   # Source: 2.7ClassificationCritique.md Section 6
                                   # Note: "short" (<100 chars), "long" (100+ chars)

    "hashtag_count": int,          # Required, Range: 0-30
                                   # Number of hashtags
                                   # Source: 2.7ClassificationCritique.md Section 6

    "hashtag_placement": str,      # Required, ["end", "mixed", "none"]
                                   # Where hashtags appear
                                   # Source: 2.7ClassificationCritique.md Section 6
}

# Note: hashtag_strategy nested object REMOVED in refined schema
# Rationale (from 2.7ClassificationCritique.md Section 7):
# - Hashtag broad/niche/branded categorization removed (heuristic without view count data)
# - Only objective, countable fields retained (hashtag_count, hashtag_placement)
```

**Field Count Verification:**

```
ContentAnalysisCHILD.md Section 5.1.1 (Selection Manifest): 5 fields → TI Schema 3.2: 5 fields ✓
ContentAnalysisCHILD.md Section 5.1.2 (Transcript): 3 fields → TI Schema 3.2: 3 fields ✓
ContentAnalysisCHILD.md Section 5.1.3 (Unified Analysis): 2 nested fields → TI Schema 3.2: 2 fields ✓
ContentAnalysisCHILD.md Section 5.1.4 (Taxonomy): 7 fields → TI Schema 3.2: 7 fields ✓
ContentAnalysisCHILD.md Section 5.2.2 (Classification): 12 fields (6 core + 1 caption_analysis object with 8 subfields + 5 metadata) → TI Schema 3.3: 12 fields ✓ (refined from 23)
Note: Field names updated - pain_points (was audience_pain_points), keywords (was trending_keywords)
```

**Field Name Spot Check:**

```
1. ContentAnalysisCHILD.md: "content_category" → TI: "content_category" ✓
2. ContentAnalysisCHILD.md: "hook_strategy" → TI: "hook_strategy" ✓
3. ContentAnalysisCHILD.md: "pain_points" → TI: "pain_points" ✓ (updated from audience_pain_points)
4. ContentAnalysisCHILD.md: "keywords" → TI: "keywords" ✓ (updated from trending_keywords)
5. ContentAnalysisCHILD.md: "caption_analysis.hook_type" → TI: "hook_type" ✓ (updated from caption_hook_type)
6. ContentAnalysisCHILD.md: "caption_analysis.cta_type" → TI: "cta_type" ✓ (updated from caption_cta_type)
7. ContentAnalysisCHILD.md: "hashtag_count" → TI: "hashtag_count" ✓
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
            # Note: Uses RUMIAI_ROOT from Section 9.1 for portability
            RUMIAI_ROOT = os.environ.get('RUMIAI_ROOT', '/home/jorge/rumiaifinal')
            transcript_path = f"{RUMIAI_ROOT}/speech_transcriptions/{video_id}_whisper.json"

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
    # Source: 2.6HashtagCritique.md - Final Prompt (After All Decisions)
    # Note: System message configured separately in API call
    system_message = """You are an expert content analyst specializing in short-form video patterns. Identify recurring patterns in the transcripts based on frequency and evidence. Be objective and data-driven: report patterns that actually appear in the data, not prescriptive advice. Patterns should be actionable for content creators but grounded in observed behavior."""

    prompt = f"""Analyze the following {len(transcripts)} video transcripts from the #{hashtag} hashtag.

Your task is to identify recurring content patterns across 6 categories. Focus on patterns that appear in AT LEAST 10% of videos (minimum 3 videos). Do not create patterns for isolated or single-video elements.

Patterns should be specific and actionable so content creators can replicate them.

---

## CATEGORY 1: Content Categories

Identify the TYPES of videos that exist in this hashtag. Focus on the primary content format or purpose.

Examples to show naming style only (these are NOT limits):
- instructional_walkthrough: Teaching or how-to format
- personal_perspective: Opinion, review, or commentary
- narrative_content: Story-driven or journey-based

Create NEW category names based on patterns you observe in THIS specific hashtag (#{hashtag}). Do not limit yourself to these examples - they only demonstrate the naming format (snake_case, 2-4 words, descriptive).

Discover the categories that reflect actual content patterns in these transcripts. Typically this will be 3-8 categories, but return only as many as you genuinely observe - do not force patterns to reach a target number.

---

## CATEGORY 2: Hook Strategies

Identify HOW videos open. Analyze the OPENING PHRASE (first 5-10 words of each transcript) to detect attention-grabbing techniques.

Examples to show naming style only (these are NOT limits):
- question_hook: Opens with a question
- direct_address: Speaks directly to viewer
- surprising_claim: Unexpected statement or fact

Create NEW hook strategy names based on opening patterns you observe in THIS specific hashtag (#{hashtag}). Focus on the rhetorical technique, not specific content. Typically this will be 2-5 hook strategies, but return only as many as you genuinely observe.

---

## CATEGORIES 3-6: Simple Lists

For Categories 3-6 below: Extract phrases (2-4 words) that appear or are IMPLIED in at least 10% of videos (minimum 3). Phrases can be verbatim quotes OR interpretations of what's shown/discussed. Return as simple string lists.

GROUNDING RULE: Every term you list must be traceable to specific transcripts. If you cannot point to at least 3 transcripts showing this pattern, do not include it.

### CATEGORY 3: Audience Pain Points

Identify PROBLEMS, STRUGGLES, or UNMET NEEDS mentioned or implied. Include:
- Explicit problems stated ("I have bloating")
- Implied problems from solutions shown ("I started doing X and Y went away" → Y is pain point)
- Challenges discussed

### CATEGORY 4: Trending Keywords

Identify TOPICS, METHODS, SOLUTIONS, or CONCEPTS mentioned or implied (excluding problems from Category 3). Include:
- Specific terms used repeatedly
- Methods or practices discussed
- Solutions or approaches mentioned

### CATEGORY 5: Engagement Drivers

Identify CONTENT FEATURES or TECHNIQUES that make content compelling (not topics). Include:
- Storytelling devices mentioned or used
- Proof elements described ("I show before/after photos")
- Engagement tactics visible in how creators speak

### CATEGORY 6: Content Tactics

Identify PRESENTATION STYLES or FORMATS mentioned or implied. Include:
- Delivery methods described or evident from speech patterns
- Visual approaches mentioned ("I'm going to show you on screen")
- Structural formats implied by how content flows

---

## OUTPUT FORMAT

Return your analysis as valid JSON with this exact structure:

{{
  "hashtag": "{hashtag}",
  "analysis_date": "{datetime.utcnow().isoformat()}Z",
  "sample_size": {len(transcripts)},
  "discovered_patterns": {{
    "content_categories": [
      {{
        "name": "instructional_walkthrough",
        "frequency": 28,
        "examples": ["step by step tutorial", "here's how to make"],
        "representative_video_ids": ["7526250443832331550", "7428596413707144481"]
      }}
    ],
    "hook_strategies": [
      {{
        "name": "question_hook",
        "frequency": 15,
        "examples": ["did you know that", "have you ever wondered"],
        "representative_video_ids": ["7526250443832331550", "7428596413707144481"]
      }}
    ],
    "audience_pain_points": ["chronic bloating", "low energy"],
    "trending_keywords": ["protein intake", "gut health"],
    "engagement_drivers": ["before after reveal"],
    "content_tactics": ["direct to camera", "voiceover"]
  }}
}}

Requirements:
- Categories 1-2: Provide 2-3 examples and 2-3 representative_video_ids per pattern
- DO NOT include a "percentage" field - this will be calculated automatically by Python post-processing
- Categories 3-6: Simple string lists only (no objects, no extra fields)

---

## FINAL INSTRUCTIONS

1. Analyze ALL {len(transcripts)} transcripts carefully - every pattern must be grounded in observed data
2. Return only patterns you genuinely observe - if a category has fewer patterns, that's acceptable
3. Use descriptive snake_case names (short but clear, 2-4 words)
4. Return valid JSON only - no commentary, explanations, or additional text outside the JSON structure

DO NOT make up patterns to fill categories. Quality over quantity.

---

TRANSCRIPTS:

{json.dumps([{{'video_id': t['video_id'], 'text': t['text']} for t in transcripts], indent=2)}
"""

    # Step 2: Initialize Anthropic client
    # Source: ContentAnalysisCHILD.md Section 2.3.2 line 224
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Step 3: Call API with retry logic (3 attempts with exponential backoff)
    # Source: ContentAnalysisCHILD.md Section 2.3.2 Edge Cases table
    for attempt in range(3):
        try:
            # Step 3.1: Make API call with Sonnet model
            # Source: 2.6HashtagCritique.md - Final Prompt (System + User messages)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                timeout=120,  # 2 minutes
                system=system_message,
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

### 4.2.5 Function: calculate_percentages()

**Purpose**: Calculate percentage field for discovery patterns after LLM response

**Algorithm (Pseudocode)**:
```python
def calculate_percentages(raw_taxonomy: dict, sample_size: int) -> dict:
    """
    Add percentage field to discovery patterns post-LLM.

    Args:
        raw_taxonomy: Raw discovery JSON from LLM (without percentage fields)
        sample_size: Number of transcripts analyzed (e.g., 50)

    Returns:
        dict: Discovery JSON with percentage fields added

    Raises:
        ValueError: If frequency > sample_size (data integrity check)
    """
    # Step 1: Process Categories 1-2 (content_categories, hook_strategies)
    # These have frequency field, simple lists don't
    for category in ['content_categories', 'hook_strategies']:
        for pattern in raw_taxonomy['discovered_patterns'][category]:
            frequency = pattern['frequency']

            # Step 1.1: Validate frequency doesn't exceed sample size
            if frequency > sample_size:
                raise ValueError(
                    f"Pattern '{pattern['name']}' frequency ({frequency}) "
                    f"exceeds sample size ({sample_size}). LLM hallucination detected."
                )

            # Step 1.2: Calculate percentage (round to 1 decimal place)
            pattern['percentage'] = round((frequency / sample_size) * 100, 1)

    # Step 2: Return enriched taxonomy
    return raw_taxonomy
```

**Validation Rules**:
```python
assert frequency <= sample_size, "Frequency cannot exceed sample size"
assert 0 <= percentage <= 100, "Percentage must be 0-100"
```

**Example**:
Input (from LLM):
```json
{
  "content_categories": [
    {"name": "recipe_tutorial", "frequency": 32, "examples": [...]}
  ]
}
```

Output (after calculate_percentages with sample_size=50):
```json
{
  "content_categories": [
    {"name": "recipe_tutorial", "frequency": 32, "percentage": 64.0, "examples": [...]}
  ]
}
```

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
        dict: Classification JSON with 12 fields (schema Section 3.3, refined prompt)

    Raises:
        TimeoutError: If LLM exceeds 30s timeout per video after 3 retries
        ValueError: If LLM returns invalid JSON after 3 retries
    """
    # Step 1: Build classification prompt with taxonomy + video data
    # Source: 2.7ClassificationCritique.md - Final Refined Prompt
    # Note: This is the refined prompt after systematic critique (Sections 1-10)

    # Build system message
    system_message = """You are an expert content classifier specializing in short-form video analysis. Your task is to accurately classify videos using a predefined taxonomy that was empirically discovered from real video data in this hashtag (Stage 2.6).

Be objective and evidence-based: select classifications that best match the video content based on transcript, caption, and hashtags. Use taxonomy categories EXACTLY as defined - do not reinterpret or expand their meaning. When evidence is ambiguous, note lower confidence rather than forcing a classification."""

    # Build main prompt
    transcript_text = transcript['text'] if transcript['available'] else "(No transcript available - classify using caption and hashtags)"

    prompt = f"""## ZONE 1: TAXONOMY & CORE CLASSIFICATION

### Provided Taxonomy

**Category 1: Content Categories** (Single Selection)
{json.dumps(taxonomy['content_categories'], indent=2)}

**Category 2: Hook Strategies** (Single Selection)
{json.dumps(taxonomy['hook_strategies'], indent=2)}

**Category 3: Audience Pain Points** (Multiple Selection)
{json.dumps(taxonomy['audience_pain_points'], indent=2)}

**Category 4: Trending Keywords** (Multiple Selection)
{json.dumps(taxonomy['trending_keywords'], indent=2)}

**Category 5: Engagement Drivers** (Multiple Selection)
{json.dumps(taxonomy['engagement_drivers'], indent=2)}

**Category 6: Content Tactics** (Multiple Selection)
{json.dumps(taxonomy['content_tactics'], indent=2)}

---

### Video Data

**Video ID**: {video_id}

**Transcript**:
{transcript_text}

**Caption**:
{caption}

**Hashtags**:
{json.dumps(hashtags)}

---

### Classification Task

Select the best-matching categories from the taxonomy above:

**Categories 1-2: Single Selection (REQUIRED)**

**Content Category**: Select exactly ONE category that best describes the primary content format.

**Hook Strategy**: Select exactly ONE strategy that best describes how the video opens.

**IMPORTANT**: You MUST copy the category name EXACTLY as written in the taxonomy. Do not paraphrase, abbreviate, or modify the string. Mismatched spelling or underscores will cause system errors.

**If no perfect match exists**: Select the closest matching category from the taxonomy. Set confidence=low and document the mismatch in the note field (e.g., "Video is comedy skit, closest match is wellness_practice").

**String Matching**: Copy category names character-for-character from taxonomy above.

**Categories 3-6: Multiple Selection (0-N)**

Select ALL applicable items from the taxonomy that are clearly present in the video:

- **Audience Pain Points**: Problems explicitly stated OR strongly implied from solutions discussed
- **Trending Keywords**: Topics/methods explicitly mentioned OR clearly central to the content
- **Engagement Drivers**: Tactics described OR evident from how creator speaks
- **Content Tactics**: Presentation styles explicitly mentioned OR observable from transcript patterns

**GROUNDING RULE**: Only select items that are:
1. **Explicitly mentioned**: Can quote a direct phrase (e.g., "I had bloating")
2. **Strongly implied**: Clear evidence from context (e.g., "I started X and it went away" → X addresses implied problem)

If uncertain whether implication is strong enough, do NOT select. Empty arrays `[]` are acceptable.

**Evidence requirement**: For each selection, you should be able to explain WHY it's selected with specific evidence from transcript or caption.

---

## ZONE 2: CAPTION & HASHTAG ANALYSIS

Analyze caption structure and hashtag strategy as a secondary task after completing Zone 1.

**Caption Hook Type**: How does the caption open? (first 5-10 words)
- statement: Declarative ("This changed my life", "Best product ever")
- question: Interrogative ("Did you know?", "Have you tried?")
- command: Imperative ("Try this now", "Follow for more")
- teaser: Creates curiosity ("You won't believe…", "Wait till the end")

**Call-to-Action**:
- cta_type: link_in_bio, save_post, comment, follow, share, tag_friend, none
- brand_mention_present: Does caption mention a brand/product? (true/false)
- influencer_tag_present: Does caption tag another creator? (true/false)

**Caption Metrics** (simplified levels):
- emoji_usage: none (0), some (1-4), many (5+)
- caption_length: short (<100 chars), long (100+ chars)

**Hashtag Analysis**:
- hashtag_count: Total number of hashtags (integer)
- hashtag_placement: end (all at end), mixed (throughout caption), none

**Note**: Do not attempt to categorize hashtags as broad/niche/branded - this requires view count data not available.

---

## ZONE 3: OUTPUT & CONFIDENCE

### Evidence Handling & Fallback Logic

**Hook Strategy** (required single selection):
- **Primary**: Use transcript opening (first 5-10 words spoken)
- **Fallback**: If transcript empty, use caption opening (first 5-10 words written)
- **Caveat**: Caption opening may not reflect actual video opening - this is acceptable as "best available evidence"

**Content Category** (required single selection):
- **Primary**: Classify from full transcript + caption alignment
- **Fallback**: If transcript empty, classify from caption + hashtags only

**Note Field** (dynamic context for low-confidence scenarios):
- Empty transcript → "Classified from caption/hashtags only - no transcript available"
- Conflicting evidence → "Transcript suggests X, caption suggests Y - selected X (transcript priority)"
- Forced match → "No perfect taxonomy match, selected closest: [category_name]"
- Multiple issues → Combine messages: "No transcript + forced match to [category]"

**Evidence Priority** (transparent but enforced):
When evidence conflicts: transcript > caption > hashtags. Document conflicts in note field.
When evidence is weak: still make best-effort classification, but set confidence=low and explain in note.

---

### Confidence Assessment

Assign confidence based on two factors: (1) How well video matches taxonomy, (2) Quality of evidence

**high**:
- Video clearly matches selected categories (no ambiguity in taxonomy fit)
- Strong evidence from transcript and/or caption
- All selections can be justified with explicit phrases or strong implications

**medium**:
- Video partially matches taxonomy OR selection required inference
- Evidence from transcript OR caption, but not both aligning
- Some selections based on reasonable but not explicit evidence

**low**:
- Forced match for required categories (no perfect taxonomy fit)
- Limited evidence (empty transcript, minimal caption)
- Selections based on weak inference or hashtags alone

**Tie-breakers**:
- If transcript unavailable but caption is rich → can be medium (not automatically low)
- If perfect taxonomy match but only hashtags available → medium (good match, weak evidence)

---

### Output Format

Return a single JSON object with ALL 12 fields present. Do not add fields beyond this schema.

**Required fields** (must be non-null):
- video_id: String (provided in input)
- taxonomy_version: Always use "stage2.6_output"
- content_category: String (exactly one from taxonomy)
- hook_strategy: String (exactly one from taxonomy)
- confidence: "high"|"medium"|"low"
- transcript_available: true|false
- note: String with explanation OR null if high confidence

**Multi-select fields** (empty arrays [] allowed):
- pain_points: Array of strings from taxonomy ([] if none apply)
- keywords: Array of strings from taxonomy ([] if none apply)
- engagement_drivers: Array of strings from taxonomy ([] if none apply)
- content_tactics: Array of strings from taxonomy ([] if none apply)

**Caption analysis object** (all 8 subfields required):
- caption_analysis: {{
    hook_type, cta_type, brand_mention_present, influencer_tag_present,
    emoji_usage, caption_length, hashtag_count, hashtag_placement
  }}

**JSON FORMATTING RULES**:
- Use lowercase true/false for booleans (not True/False)
- Always include note field (use null if not needed, don't omit)
- Empty arrays should be [] (not null)
- Copy string values exactly (including underscores and capitalization)
- No additional fields beyond this schema

---

### FINAL INSTRUCTIONS

Before submitting, verify all requirements are met:

**Critical Requirements (System Errors)**
✓ **Exact Strings**: Copy category names character-for-character from taxonomy (e.g., "wellness_practice" NOT "wellness")
   - Mismatched spelling or underscores will cause system error
✓ **Complete Schema**: All 12 fields present (see Output Format section)
✓ **JSON Only**: No text outside JSON structure

**Classification Quality**
✓ **Evidence-Based**: All selections traceable to transcript/caption/hashtags - do not invent patterns
   - Quality over quantity: empty arrays [] better than wrong selections
✓ **Closest Match**: If perfect taxonomy match unclear, select closest category and set confidence=low
✓ **Note Field**: Explain when confidence=low (forced match, missing transcript, conflicts)
✓ **Evidence Priority**: transcript > caption > hashtags (see Zone 3)

Your classifications feed Stage 7 contrastive analysis - accuracy is critical.
"""

    # Step 2: Call API with retry logic (3 attempts with exponential backoff)
    # Source: ContentAnalysisCHILD.md Section 2.3.4 lines 454-475
    for attempt in range(3):
        try:
            # Step 2.1: Make API call with Haiku model
            # Source: ContentAnalysisCHILD.md Section 2.3.4 lines 457-462
            # Updated: Uses system message from refined prompt (2.7ClassificationCritique.md)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                timeout=30,  # 30 seconds per video
                system=system_message,
                messages=[{"role": "user", "content": prompt}]
            )

            # Step 2.2: Parse response
            classification = json.loads(response.content[0].text)

            # Step 2.3: Validate output schema before returning
            # Source: ContentAnalysisCHILD.md Section 6.3 (Output Validation)
            validate_classification_output(classification)

            # Step 2.4: Return successful classification
            return classification

        except anthropic.RateLimitError as e:
            # Step 2.5a: Handle rate limit errors (429) - delegate to rate limit handler
            # Source: Error E13, Section 6.2
            # Note: Use handle_api_rate_limit() wrapper for batch operations
            if attempt < 2:
                delay = [2, 4, 8][attempt]  # Longer backoff for rate limits
                logger.warning(f"Rate limit hit for {video_id}, retry {attempt+1} in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"Rate limit exceeded for {video_id} after 3 retries")
                raise  # Re-raise after final retry

        except (TimeoutError, anthropic.APIError) as e:
            # Step 2.5b: Handle timeout/API errors (retry with backoff)
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
Step 1: Build prompt with taxonomy + video data → 3000 char prompt (refined 3-zone structure)
Step 2: Call Haiku API with 30s timeout + system message → Response in 2.5s
Step 3: Parse JSON → 12 fields extracted (simplified schema)
Step 4: Validate all required fields present → Pass
Step 5: Return classification dict
Output: {"video_id": "7526250443832331550", "taxonomy_version": "stage2.6_output", "content_category": "wellness_practice", "confidence": "high", ...}

---

### 4.4 Function: classify_all_videos()

**Source**: M2 Enhancement - Configurable Parallel Classification

**Purpose**: Orchestrate classification of multiple videos with configurable sequential or parallel execution

**Algorithm (Pseudocode)**:
```python
def classify_all_videos(
    videos: list[str],
    taxonomy: dict,
    client: anthropic.Anthropic,
    manifest_path: str,
    bucket_base: str,
    parallel: bool = False,
    max_workers: int = 5
) -> dict:
    """
    Classify all videos from manifest with configurable execution mode.

    Args:
        videos: List of video IDs to classify
        taxonomy: Curated taxonomy from Stage 2.6
        client: Initialized Anthropic API client
        manifest_path: Path to selection_manifest.json
        bucket_base: Base path for bucket output directories
        parallel: Enable parallel processing (default: False for sequential)
        max_workers: Concurrent workers if parallel=True (default: 5)

    Returns:
        dict: {
            "total": 120,
            "completed": 120,
            "failed": 0,
            "mode": "sequential" or "parallel",
            "duration_seconds": 312.5
        }

    Raises:
        ValueError: If videos list is empty or taxonomy invalid
    """
    # Step 1: Validate inputs
    if not videos:
        raise ValueError("videos list cannot be empty")
    if not taxonomy:
        raise ValueError("taxonomy cannot be empty")

    # Step 2: Load manifest to get video metadata
    manifest = load_json(manifest_path)

    # Step 3: Choose execution mode based on parallel flag
    # Source: Section 9.1 ENABLE_PARALLEL_CLASSIFICATION env var
    start_time = time.time()

    if parallel:
        logger.info(f"🚀 Starting parallel classification: {len(videos)} videos, {max_workers} workers")
        results = classify_all_videos_parallel(
            videos, taxonomy, client, manifest, bucket_base, max_workers
        )
    else:
        logger.info(f"📋 Starting sequential classification: {len(videos)} videos")
        results = classify_all_videos_sequential(
            videos, taxonomy, client, manifest, bucket_base
        )

    # Step 4: Calculate summary statistics
    duration = time.time() - start_time
    summary = {
        "total": len(videos),
        "completed": results["completed"],
        "failed": results["failed"],
        "mode": "parallel" if parallel else "sequential",
        "duration_seconds": round(duration, 2)
    }

    logger.info(
        f"✅ Classification complete: {summary['completed']}/{summary['total']} videos "
        f"({summary['mode']} mode, {summary['duration_seconds']}s)"
    )

    return summary
```

---

### 4.4.1 Function: classify_all_videos_sequential()

**Purpose**: Sequential classification (one video at a time) - default behavior

**Algorithm (Pseudocode)**:
```python
def classify_all_videos_sequential(
    videos: list[str],
    taxonomy: dict,
    client: anthropic.Anthropic,
    manifest: dict,
    bucket_base: str
) -> dict:
    """
    Classify videos sequentially (one at a time).

    Returns:
        dict: {"completed": int, "failed": int, "failed_ids": list[str]}
    """
    completed = 0
    failed = 0
    failed_ids = []

    for i, video_id in enumerate(videos):
        try:
            # Step 1: Load video data (transcript, caption, hashtags)
            transcript = load_transcript(video_id)
            caption, hashtags = load_caption_and_hashtags(video_id)

            # Step 2: Classify video
            classification = classify_video_llm(
                video_id, transcript, caption, hashtags, taxonomy, client
            )

            # Step 3: Validate output
            validate_classification_output(classification)

            # Step 4: Save to bucket directory
            output_path = f"{bucket_base}/content_analysis/{video_id}_content.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_json(output_path, classification)

            completed += 1
            logger.debug(f"✅ Classified {i+1}/{len(videos)}: {video_id}")

        except Exception as e:
            failed += 1
            failed_ids.append(video_id)
            logger.error(f"❌ Failed {i+1}/{len(videos)}: {video_id} - {str(e)}")

    return {"completed": completed, "failed": failed, "failed_ids": failed_ids}
```

**Performance**: ~5 seconds per video = 600 seconds for 120 videos (10 minutes)

---

### 4.4.2 Function: classify_all_videos_parallel()

**Purpose**: Parallel classification with controlled concurrency for rate limit safety

**Algorithm (Pseudocode)**:
```python
def classify_all_videos_parallel(
    videos: list[str],
    taxonomy: dict,
    client: anthropic.Anthropic,
    manifest: dict,
    bucket_base: str,
    max_workers: int = 5
) -> dict:
    """
    Classify videos in parallel with controlled concurrency.

    Args:
        max_workers: Concurrent API calls (default: 5)
                     Conservative to avoid rate limits (50 req/min for Haiku)
                     Max recommended: 10 (leaves headroom for rate limits)

    Returns:
        dict: {"completed": int, "failed": int, "failed_ids": list[str]}

    Notes:
        - Uses ThreadPoolExecutor for I/O-bound API calls
        - Rate limiting handled per-video via handle_api_rate_limit() (Section 6.2)
        - Each worker processes one video at a time
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed = 0
    failed = 0
    failed_ids = []

    # Step 1: Prepare work items (video ID + metadata)
    work_items = []
    for video_id in videos:
        transcript = load_transcript(video_id)
        caption, hashtags = load_caption_and_hashtags(video_id)
        work_items.append({
            "video_id": video_id,
            "transcript": transcript,
            "caption": caption,
            "hashtags": hashtags
        })

    # Step 2: Submit all tasks to thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all classification tasks
        future_to_video = {
            executor.submit(
                classify_single_video_with_save,
                item, taxonomy, client, bucket_base
            ): item["video_id"]
            for item in work_items
        }

        # Step 3: Process results as they complete
        for future in as_completed(future_to_video):
            video_id = future_to_video[future]
            try:
                result = future.result()  # Blocks until this video completes
                completed += 1
                logger.debug(f"✅ Classified ({completed}/{len(videos)}): {video_id}")

            except Exception as e:
                failed += 1
                failed_ids.append(video_id)
                logger.error(f"❌ Failed classification: {video_id} - {str(e)}")

    return {"completed": completed, "failed": failed, "failed_ids": failed_ids}


def classify_single_video_with_save(
    item: dict,
    taxonomy: dict,
    client: anthropic.Anthropic,
    bucket_base: str
) -> dict:
    """
    Helper function for parallel execution: classify + save.

    Args:
        item: {"video_id": str, "transcript": dict, "caption": str, "hashtags": list}

    Returns:
        dict: Classification result

    Raises:
        Exception: Any classification or save error (caught by parent)
    """
    # Step 1: Classify video (includes retry logic from Section 4.3)
    classification = classify_video_llm(
        item["video_id"],
        item["transcript"],
        item["caption"],
        item["hashtags"],
        taxonomy,
        client
    )

    # Step 2: Validate output
    validate_classification_output(classification)

    # Step 3: Save to bucket directory (thread-safe, each video has unique file)
    output_path = f"{bucket_base}/content_analysis/{item['video_id']}_content.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(output_path, classification)

    return classification
```

**Performance**: ~5 seconds per video ÷ 5 workers = 120 seconds for 120 videos (2 minutes)

**Speedup**: 5x faster than sequential (10 min → 2 min)

**Safety**:
- Rate limiting handled by `handle_api_rate_limit()` (Section 6.2, Error E13)
- Conservative max_workers=5 leaves headroom (50 req/min limit ÷ 5 workers = 10 req/min per worker)
- Each video has independent retry logic from Section 4.3

**Trade-offs**:
- ✅ Pro: 5x speedup for 120 videos
- ✅ Pro: Still safe with rate limits (conservative concurrency)
- ❌ Con: More complex error handling (per-thread exceptions)
- ❌ Con: Higher memory usage (5 concurrent API requests)

---

### 4.5 Checkpoint/Resume Functions

**Source**: M3 Enhancement - Checkpoint/Resume for Stage 2.7

**Purpose**: Enable resumption of interrupted classification runs without reprocessing completed videos

**Checkpoint Format**:
```json
{
    "completed": ["7526250443832331550", "7428596413707144481", ...],
    "failed": ["7111111111111111111"],
    "last_updated": "2025-01-28T10:30:00Z",
    "total_videos": 120,
    "stats": {
        "completed_count": 80,
        "failed_count": 1
    }
}
```

---

### 4.5.1 Function: load_checkpoint()

**Algorithm (Pseudocode)**:
```python
def load_checkpoint(checkpoint_path: str) -> dict:
    """
    Load classification checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint JSON file

    Returns:
        dict: Checkpoint data, or empty checkpoint if file doesn't exist

    Raises:
        ValueError: If checkpoint file is malformed
    """
    # Step 1: Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        # Return empty checkpoint for first run
        logger.info("No checkpoint found, starting fresh classification")
        return {
            "completed": [],
            "failed": [],
            "last_updated": None,
            "total_videos": 0,
            "stats": {"completed_count": 0, "failed_count": 0}
        }

    # Step 2: Load checkpoint file
    try:
        checkpoint = load_json(checkpoint_path)

        # Step 3: Validate checkpoint structure
        required_fields = ["completed", "failed"]
        missing = [f for f in required_fields if f not in checkpoint]
        if missing:
            raise ValueError(f"Checkpoint missing required fields: {missing}")

        # Step 4: Log resumption info
        completed_count = len(checkpoint["completed"])
        failed_count = len(checkpoint["failed"])
        logger.info(
            f"📂 Resuming from checkpoint: {completed_count} completed, "
            f"{failed_count} failed, last updated {checkpoint.get('last_updated', 'unknown')}"
        )

        return checkpoint

    except json.JSONDecodeError as e:
        raise ValueError(f"Checkpoint file is malformed: {str(e)}")
```

**Validation Rules**:
```python
assert "completed" in checkpoint, "Checkpoint must have 'completed' field"
assert "failed" in checkpoint, "Checkpoint must have 'failed' field"
assert isinstance(checkpoint["completed"], list), "'completed' must be array"
assert isinstance(checkpoint["failed"], list), "'failed' must be array"
```

---

### 4.5.2 Function: save_checkpoint()

**Algorithm (Pseudocode)**:
```python
def save_checkpoint(checkpoint_path: str, checkpoint: dict):
    """
    Save classification checkpoint to disk (atomic write).

    Args:
        checkpoint_path: Path to checkpoint JSON file
        checkpoint: Checkpoint data to save

    Raises:
        IOError: If checkpoint cannot be written
    """
    # Step 1: Update timestamp
    checkpoint["last_updated"] = datetime.utcnow().isoformat() + "Z"

    # Step 2: Update stats
    checkpoint["stats"] = {
        "completed_count": len(checkpoint["completed"]),
        "failed_count": len(checkpoint["failed"])
    }

    # Step 3: Atomic write (write to temp file, then rename)
    # Source: Prevents corruption if process killed during write
    temp_path = checkpoint_path + ".tmp"
    try:
        # Write to temp file
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Atomic rename (replaces old checkpoint)
        os.replace(temp_path, checkpoint_path)

        logger.debug(f"💾 Checkpoint saved: {checkpoint['stats']['completed_count']} completed")

    except Exception as e:
        # Clean up temp file if write failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise IOError(f"Failed to save checkpoint: {str(e)}")
```

**Atomic Write Guarantee**:
- Uses temp file + `os.replace()` for atomic operation
- If process killed during write, old checkpoint remains intact
- Prevents checkpoint corruption

---

### 4.5.3 Function: update_checkpoint()

**Algorithm (Pseudocode)**:
```python
def update_checkpoint(
    checkpoint: dict,
    video_id: str,
    status: str,
    checkpoint_path: str
):
    """
    Update checkpoint after processing single video.

    Args:
        checkpoint: Current checkpoint dict
        video_id: Video just processed
        status: "completed" or "failed"
        checkpoint_path: Path to save updated checkpoint

    Raises:
        ValueError: If status is invalid
    """
    # Step 1: Validate status
    if status not in ["completed", "failed"]:
        raise ValueError(f"Invalid status: {status}. Must be 'completed' or 'failed'")

    # Step 2: Update checkpoint data
    if status == "completed":
        if video_id not in checkpoint["completed"]:
            checkpoint["completed"].append(video_id)
        # Remove from failed list if previously failed (retry success)
        if video_id in checkpoint["failed"]:
            checkpoint["failed"].remove(video_id)
    else:  # failed
        if video_id not in checkpoint["failed"]:
            checkpoint["failed"].append(video_id)
        # Don't add to completed list

    # Step 3: Save updated checkpoint
    save_checkpoint(checkpoint_path, checkpoint)
```

**Edge Cases**:
- **Case 1**: Video already in completed list → Skip (idempotent)
- **Case 2**: Video fails after previous success → Move from completed to failed
- **Case 3**: Checkpoint save fails → Raise error, don't continue (fail-fast)

---

### 4.5.4 Integration: Update classify_all_videos_sequential()

**Modified Algorithm**:
```python
def classify_all_videos_sequential(
    videos: list[str],
    taxonomy: dict,
    client: anthropic.Anthropic,
    manifest: dict,
    bucket_base: str,
    checkpoint_path: str = None  # NEW PARAMETER
) -> dict:
    """
    Classify videos sequentially with checkpoint/resume support.

    Args:
        checkpoint_path: Path to checkpoint file (optional)
                        If None, no checkpointing (backward compatible)

    Returns:
        dict: {"completed": int, "failed": int, "failed_ids": list[str]}
    """
    # Step 1: Load checkpoint (if enabled)
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        # Filter out already-completed videos
        remaining_videos = [v for v in videos if v not in checkpoint["completed"]]
        logger.info(
            f"📋 Sequential mode with checkpoints: "
            f"{len(checkpoint['completed'])} already completed, "
            f"{len(remaining_videos)} remaining"
        )
    else:
        remaining_videos = videos
        checkpoint = None

    completed = 0
    failed = 0
    failed_ids = []

    # Step 2: Process remaining videos
    for i, video_id in enumerate(remaining_videos):
        try:
            # Load video data
            transcript = load_transcript(video_id)
            caption, hashtags = load_caption_and_hashtags(video_id)

            # Classify video
            classification = classify_video_llm(
                video_id, transcript, caption, hashtags, taxonomy, client
            )

            # Validate output
            validate_classification_output(classification)

            # Save classification
            output_path = f"{bucket_base}/content_analysis/{video_id}_content.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_json(output_path, classification)

            # Update checkpoint (if enabled)
            if checkpoint_path:
                update_checkpoint(checkpoint, video_id, "completed", checkpoint_path)

            completed += 1
            logger.debug(f"✅ Classified {i+1}/{len(remaining_videos)}: {video_id}")

        except Exception as e:
            # Update checkpoint for failure (if enabled)
            if checkpoint_path:
                update_checkpoint(checkpoint, video_id, "failed", checkpoint_path)

            failed += 1
            failed_ids.append(video_id)
            logger.error(f"❌ Failed {i+1}/{len(remaining_videos)}: {video_id} - {str(e)}")

    # Step 3: Return results
    total_completed = completed + (len(checkpoint["completed"]) if checkpoint else 0)
    return {
        "completed": total_completed,
        "failed": failed,
        "failed_ids": failed_ids
    }
```

**Backward Compatibility**:
- `checkpoint_path=None` → No checkpointing (original behavior)
- `checkpoint_path="/path/to/checkpoint.json"` → Enable checkpointing

---

### 4.5.5 Integration: Update classify_all_videos_parallel()

**Modified Algorithm**:
```python
# ... (existing parallel implementation from Section 4.5.5)
```

---

### 4.6 Cost Tracking Functions

**Source**: M5 Enhancement - Cost Tracking and Logging

**Purpose**: Provide visibility into API costs for monitoring and budgeting

**Pricing (as of 2025-01)**:
- Claude 3.5 Sonnet: $15/1M input tokens, $75/1M output tokens
- Claude 3 Haiku: $0.25/1M input tokens, $1.25/1M output tokens

---

### 4.6.1 Function: log_estimated_cost()

**Algorithm (Pseudocode)**:
```python
def log_estimated_cost(
    operation: str,
    video_count: int = None,
    sample_size: int = None
):
    """
    Log estimated API costs for transparency.

    Args:
        operation: "discovery" or "classification"
        video_count: Number of videos (for classification)
        sample_size: Number of transcripts (for discovery)

    Returns:
        float: Estimated cost in USD
    """
    # Step 1: Calculate estimated costs
    if operation == "discovery":
        # Discovery uses Sonnet for 50 transcripts
        # Estimated: ~85K input tokens, ~2.5K output tokens
        estimated_cost = 0.75  # Fixed per hashtag
        details = f"Sonnet API call, ~{sample_size or 50} transcripts"

    elif operation == "classification":
        # Classification uses Haiku, ~450 input + ~180 output tokens per video
        cost_per_video = 0.001  # $0.001 per video
        estimated_cost = cost_per_video * (video_count or 120)
        details = f"Haiku API calls, {video_count or 120} videos"

    else:
        raise ValueError(f"Unknown operation: {operation}")

    # Step 2: Log cost estimate
    logger.info(
        f"💰 Estimated cost for {operation}: ${estimated_cost:.2f} ({details})"
    )

    return estimated_cost
```

**Example Usage**:
```python
# In discover_patterns_llm() (Section 4.2):
log_estimated_cost("discovery", sample_size=len(transcripts))

# In classify_all_videos() (Section 4.4):
log_estimated_cost("classification", video_count=len(videos))
```

**Example Log Output**:
```
[2025-01-28 10:30:00] [INFO] 💰 Estimated cost for discovery: $0.75 (Sonnet API call, ~50 transcripts)
[2025-01-28 10:35:00] [INFO] 💰 Estimated cost for classification: $0.12 (Haiku API calls, 120 videos)
```

---

### 4.6.2 Function: log_actual_cost()

**Algorithm (Pseudocode)**:
```python
def log_actual_cost(
    response: anthropic.types.Message,
    model: str,
    start_time: float = None
):
    """
    Log actual API cost from response usage data with latency.

    Args:
        response: Anthropic API response object
        model: "sonnet" or "haiku"
        start_time: Optional start time (from time.time()) for latency calculation

    Returns:
        float: Actual cost in USD
    """
    # Step 1: Calculate latency if start_time provided
    latency = None
    if start_time:
        latency = time.time() - start_time

    # Step 2: Extract token usage from response
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # Step 3: Calculate actual cost based on model pricing
    pricing = {
        "sonnet": {
            "input": 15 / 1_000_000,   # $15 per 1M tokens
            "output": 75 / 1_000_000    # $75 per 1M tokens
        },
        "haiku": {
            "input": 0.25 / 1_000_000,  # $0.25 per 1M tokens
            "output": 1.25 / 1_000_000  # $1.25 per 1M tokens
        }
    }

    cost = (input_tokens * pricing[model]["input"] +
            output_tokens * pricing[model]["output"])

    # Step 4: Log actual cost with token details and latency
    if latency:
        logger.debug(
            f"💸 API call: ${cost:.4f}, {latency:.2f}s, "
            f"in: {input_tokens:,} tokens, out: {output_tokens:,} tokens, model: {model}"
        )
    else:
        logger.debug(
            f"💸 API call cost: ${cost:.4f} "
            f"(in: {input_tokens:,} tokens, out: {output_tokens:,} tokens, model: {model})"
        )

    return cost
```

**Integration Example**:
```python
# In discover_patterns_llm() (Section 4.2):
start_time = time.time()
response = client.messages.create(...)
log_actual_cost(response, "sonnet", start_time)

# In classify_video_llm() (Section 4.3):
start_time = time.time()
response = client.messages.create(...)
log_actual_cost(response, "haiku", start_time)
```

**Example Log Output**:
```
[2025-01-28 10:30:45] [DEBUG] 💸 API call: $0.7523, 47.32s, in: 84,532 tokens, out: 2,418 tokens, model: sonnet
[2025-01-28 10:35:12] [DEBUG] 💸 API call: $0.0012, 2.51s, in: 451 tokens, out: 183 tokens, model: haiku
```

---

### 4.6.3 Integration: Update discover_patterns_llm()

**Add cost and latency logging before and after API call**:
```python
def discover_patterns_llm(transcripts: list[dict], hashtag: str) -> dict:
    # ... (existing setup from Section 4.2)

    # NEW: Log estimated cost before API call
    log_estimated_cost("discovery", sample_size=len(transcripts))

    # NEW: Start timing for latency measurement
    start_time = time.time()

    # Make API call
    response = client.messages.create(...)

    # NEW: Log actual cost and latency after API call
    log_actual_cost(response, "sonnet", start_time)

    # ... (rest of existing implementation)
```

---

### 4.6.4 Integration: Update classify_video_llm()

**Add cost and latency logging after API call**:
```python
def classify_video_llm(...) -> dict:
    # ... (existing setup from Section 4.3)

    # NEW: Start timing for latency measurement
    start_time = time.time()

    # Make API call
    response = client.messages.create(...)

    # NEW: Log actual cost and latency after API call
    log_actual_cost(response, "haiku", start_time)

    # ... (rest of existing implementation)
```

---

### 4.6.5 Integration: Update classify_all_videos()

**Add total cost logging**:
```python
def classify_all_videos(...) -> dict:
    # ... (existing setup from Section 4.4)

    # NEW: Log estimated total cost before processing
    estimated_cost = log_estimated_cost("classification", video_count=len(videos))

    # Process videos (sequential or parallel)
    if parallel:
        results = classify_all_videos_parallel(...)
    else:
        results = classify_all_videos_sequential(...)

    # NEW: Log completion with cost reminder
    logger.info(
        f"✅ Classification complete: {results['completed']}/{len(videos)} videos "
        f"(estimated cost: ${estimated_cost:.2f})"
    )

    # ... (rest of existing implementation)
```

---

### 4.7 Taxonomy Validation Functions

**Source**: M7 Enhancement - Manual Curation Validation

**Purpose**: Validate manually curated taxonomy before Stage 2.7 classification to catch errors (typos, missing fields, invalid structure)

---

### 4.7.1 Function: validate_curated_taxonomy()

**Algorithm (Pseudocode)**:
```python
def validate_curated_taxonomy(taxonomy_path: str) -> bool:
    """
    Validate manually curated taxonomy file.

    Args:
        taxonomy_path: Path to curated taxonomy JSON

    Returns:
        bool: True if valid, raises ValueError if invalid

    Raises:
        FileNotFoundError: If taxonomy file missing
        ValueError: If validation fails with specific error message
    """
    import re

    # Step 1: Check file exists
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

    # Step 2: Load and parse JSON
    try:
        taxonomy = load_json(taxonomy_path)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")

    # Step 3: Check all required top-level fields present
    required_fields = [
        'content_categories', 'hook_strategies', 'audience_pain_points',
        'trending_keywords', 'engagement_drivers', 'content_tactics'
    ]
    missing = [f for f in required_fields if f not in taxonomy]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Step 4: Validate semantic categories (categories 1-2)
    for category_type in ['content_categories', 'hook_strategies']:
        categories = taxonomy[category_type]

        # Check non-empty
        if not categories or len(categories) == 0:
            raise ValueError(f"{category_type} cannot be empty (minimum 1 category)")

        # Check reasonable count
        if len(categories) > 15:
            logger.warning(f"{category_type} has {len(categories)} items (recommended: 2-10)")

        for i, cat in enumerate(categories):
            # Check has name and definition
            if 'name' not in cat or 'definition' not in cat:
                raise ValueError(f"{category_type}[{i}] missing 'name' or 'definition'")

            # Check name is snake_case
            if not re.match(r'^[a-z0-9_]+$', cat['name']):
                raise ValueError(
                    f"{category_type}[{i}] name '{cat['name']}' must be snake_case "
                    f"(lowercase letters, numbers, underscores only)"
                )

            # Check definition not too short
            if len(cat['definition']) < 10:
                raise ValueError(
                    f"{category_type}[{i}] '{cat['name']}' definition too short: "
                    f"'{cat['definition']}' (minimum 10 chars)"
                )

        # Check for duplicate names
        names = [c['name'] for c in categories]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            raise ValueError(f"{category_type} has duplicate names: {set(duplicates)}")

    # Step 5: Validate simple list categories (categories 3-6)
    for category_type in ['audience_pain_points', 'trending_keywords',
                          'engagement_drivers', 'content_tactics']:
        items = taxonomy[category_type]

        # Check is array
        if not isinstance(items, list):
            raise ValueError(f"{category_type} must be array, got {type(items)}")

        # Check non-empty
        if not items or len(items) == 0:
            raise ValueError(f"{category_type} cannot be empty (minimum 1 item)")

        # Check reasonable count
        if len(items) > 20:
            logger.warning(f"{category_type} has {len(items)} items (recommended: 2-15)")

        # Check all items are strings
        for i, item in enumerate(items):
            if not isinstance(item, str):
                raise ValueError(f"{category_type}[{i}] must be string, got {type(item)}")

            # Check not too short
            if len(item) < 2:
                raise ValueError(f"{category_type}[{i}] too short: '{item}' (minimum 2 chars)")

        # Check for duplicates
        duplicates = [item for item in items if items.count(item) > 1]
        if duplicates:
            raise ValueError(f"{category_type} has duplicate items: {set(duplicates)}")

    # Step 6: All validations passed
    logger.info(
        f"✅ Taxonomy validation passed: {taxonomy_path}\n"
        f"   - {len(taxonomy['content_categories'])} content categories\n"
        f"   - {len(taxonomy['hook_strategies'])} hook strategies\n"
        f"   - {len(taxonomy['audience_pain_points'])} pain points\n"
        f"   - {len(taxonomy['trending_keywords'])} keywords\n"
        f"   - {len(taxonomy['engagement_drivers'])} engagement drivers\n"
        f"   - {len(taxonomy['content_tactics'])} content tactics"
    )

    return True
```

**Validation Rules Summary**:
```python
# Semantic categories (1-2):
assert len(categories) >= 1, "Must have at least 1 category"
assert 'name' in cat and 'definition' in cat, "Must have name and definition"
assert re.match(r'^[a-z0-9_]+$', cat['name']), "Name must be snake_case"
assert len(cat['definition']) >= 10, "Definition minimum 10 chars"
assert no_duplicates(names), "No duplicate category names"

# Simple lists (3-6):
assert isinstance(items, list), "Must be array"
assert len(items) >= 1, "Must have at least 1 item"
assert all(isinstance(i, str) for i in items), "All items must be strings"
assert all(len(i) >= 2 for i in items), "Items minimum 2 chars"
assert no_duplicates(items), "No duplicate items"
```

**Example Usage**:
```bash
# After manual curation, validate before Stage 2.7:
python -c "from validation import validate_curated_taxonomy; validate_curated_taxonomy('/path/nutrition_taxonomy.json')"
```

**Example Success Output**:
```
✅ Taxonomy validation passed: /data/clients/acme/hashtags/nutrition/top_contrastive/content_taxonomies/nutrition_taxonomy.json
   - 5 content categories
   - 4 hook strategies
   - 8 pain points
   - 12 keywords
   - 6 engagement drivers
   - 5 content tactics
```

**Example Error Output**:
```
ValueError: content_categories[2] name 'Recipe Tutorial' must be snake_case (lowercase letters, numbers, underscores only)
```

---

### 4.7.2 Manual Curation Instructions

**Added to Section 4.2 (discover_patterns_llm) output notes**:

After Stage 2.6 discovery completes, manually curate the taxonomy:

**Steps**:
1. **Open raw discovery** in text editor (VS Code, Sublime, etc.)
   ```bash
   code /data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/content_taxonomies/{hashtag}_raw_discovery.json
   ```

2. **Review and refine patterns**:
   - **Remove** rare patterns (< 10% frequency)
   - **Merge** similar categories (e.g., "recipe" + "cooking_tutorial" → "recipe_tutorial")
   - **Rename** unclear category names (must be snake_case)
   - **Expand** short definitions (minimum 10 chars)
   - **Remove** duplicates

3. **Save as curated taxonomy**:
   ```bash
   # Save to: {hashtag}_taxonomy.json (same directory)
   ```

4. **Validate before Stage 2.7**:
   ```bash
   python -c "from validation import validate_curated_taxonomy; validate_curated_taxonomy('/path/{hashtag}_taxonomy.json')"
   ```

5. **If validation fails**:
   - Read error message carefully (shows exact issue)
   - Fix in text editor
   - Re-validate until passes

**Common Mistakes to Avoid**:
- ❌ Category names with spaces or capitals (use snake_case: "recipe_tutorial" not "Recipe Tutorial")
- ❌ Empty arrays `[]` (must have at least 1 item per category)
- ❌ Definitions too short (minimum 10 chars)
- ❌ Duplicate category names or items
- ❌ Typos in category names (will cause classification errors in Stage 2.7)

---

### 4.5.5 Integration: Update classify_all_videos_parallel()

**Modified Algorithm**:
```python
def classify_all_videos_parallel(
    videos: list[str],
    taxonomy: dict,
    client: anthropic.Anthropic,
    manifest: dict,
    bucket_base: str,
    max_workers: int = 5,
    checkpoint_path: str = None  # NEW PARAMETER
) -> dict:
    """
    Classify videos in parallel with checkpoint/resume support.

    Args:
        checkpoint_path: Path to checkpoint file (optional)

    Returns:
        dict: {"completed": int, "failed": int, "failed_ids": list[str]}

    Notes:
        - Checkpoint updated after each video completes (thread-safe)
        - Uses lock for checkpoint updates to avoid race conditions
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Step 1: Load checkpoint and filter completed videos
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        remaining_videos = [v for v in videos if v not in checkpoint["completed"]]
        checkpoint_lock = threading.Lock()  # Thread-safe checkpoint updates
        logger.info(
            f"🚀 Parallel mode with checkpoints: "
            f"{len(checkpoint['completed'])} already completed, "
            f"{len(remaining_videos)} remaining"
        )
    else:
        remaining_videos = videos
        checkpoint = None
        checkpoint_lock = None

    completed = 0
    failed = 0
    failed_ids = []

    # Step 2: Prepare work items
    work_items = []
    for video_id in remaining_videos:
        transcript = load_transcript(video_id)
        caption, hashtags = load_caption_and_hashtags(video_id)
        work_items.append({
            "video_id": video_id,
            "transcript": transcript,
            "caption": caption,
            "hashtags": hashtags
        })

    # Step 3: Submit all tasks to thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(
                classify_single_video_with_save,
                item, taxonomy, client, bucket_base
            ): item["video_id"]
            for item in work_items
        }

        # Step 4: Process results as they complete
        for future in as_completed(future_to_video):
            video_id = future_to_video[future]
            try:
                result = future.result()
                completed += 1

                # Update checkpoint (thread-safe)
                if checkpoint_path:
                    with checkpoint_lock:
                        update_checkpoint(checkpoint, video_id, "completed", checkpoint_path)

                logger.debug(f"✅ Classified ({completed}/{len(remaining_videos)}): {video_id}")

            except Exception as e:
                failed += 1
                failed_ids.append(video_id)

                # Update checkpoint for failure (thread-safe)
                if checkpoint_path:
                    with checkpoint_lock:
                        update_checkpoint(checkpoint, video_id, "failed", checkpoint_path)

                logger.error(f"❌ Failed classification: {video_id} - {str(e)}")

    # Step 5: Return results
    total_completed = completed + (len(checkpoint["completed"]) if checkpoint else 0)
    return {
        "completed": total_completed,
        "failed": failed,
        "failed_ids": failed_ids
    }
```

**Thread Safety**:
- Uses `threading.Lock()` for checkpoint updates
- Ensures only one thread updates checkpoint at a time
- Prevents race conditions in parallel mode

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
            required_fields = ['name', 'frequency', 'examples']
            missing = [f for f in required_fields if f not in pattern]
            if missing:
                raise ValueError(
                    f"Pattern in {category} missing fields: {missing}. Pattern: {pattern}"
                )


# ===== CLASSIFICATION OUTPUT VALIDATION =====
# Source: 2.7ClassificationCritique.md - Section 9 (Refined Schema)
# Updated: Reflects refined prompt with 12 fields (simplified from 23 fields)

def validate_classification_output(classification: dict):
    """
    Validate classification JSON before saving.
    Source: 2.7ClassificationCritique.md Section 9 - Final Schema
    """
    # Validation 1: Check all 12 top-level fields present
    # Source: 2.7ClassificationCritique.md Section 9 - Required fields
    required_fields = [
        'video_id', 'taxonomy_version', 'content_category', 'hook_strategy',
        'pain_points', 'keywords', 'engagement_drivers', 'content_tactics',
        'caption_analysis', 'confidence', 'transcript_available', 'note'
    ]
    missing = [f for f in required_fields if f not in classification]
    if missing:
        raise ValueError(f"Classification missing required fields: {missing}")

    # Validation 2: Check taxonomy_version is correct
    # Source: 2.7ClassificationCritique.md Section 9
    if classification['taxonomy_version'] != 'stage2.6_output':
        raise ValueError(
            f"Invalid taxonomy_version: {classification['taxonomy_version']}. "
            f"Must be 'stage2.6_output'."
        )

    # Validation 3: Check confidence value
    # Source: 2.7ClassificationCritique.md Section 5 - Confidence Assessment
    if classification['confidence'] not in ['high', 'medium', 'low']:
        raise ValueError(
            f"Invalid confidence value: {classification['confidence']}. "
            f"Must be high, medium, or low."
        )

    # Validation 4: Check caption_analysis has all 8 subfields (simplified from 12)
    # Source: 2.7ClassificationCritique.md Section 6 - Caption Analysis Fields
    caption_fields = [
        'hook_type', 'cta_type', 'brand_mention_present', 'influencer_tag_present',
        'emoji_usage', 'caption_length', 'hashtag_count', 'hashtag_placement'
    ]
    caption_analysis = classification['caption_analysis']
    missing = [f for f in caption_fields if f not in caption_analysis]
    if missing:
        raise ValueError(f"caption_analysis missing fields: {missing}")

    # Validation 5: Check arrays are actually arrays
    # Note: Field names changed from audience_pain_points → pain_points, trending_keywords → keywords
    array_fields = ['pain_points', 'keywords', 'engagement_drivers', 'content_tactics']
    for field in array_fields:
        if not isinstance(classification[field], list):
            raise ValueError(f"Field {field} must be array, got {type(classification[field])}")

    # Validation 6: Check boolean fields are booleans
    # Source: 2.7ClassificationCritique.md Section 9 - JSON Formatting Rules
    boolean_fields = ['transcript_available']
    caption_boolean_fields = ['brand_mention_present', 'influencer_tag_present']
    for field in boolean_fields:
        if not isinstance(classification[field], bool):
            raise ValueError(f"Field {field} must be boolean, got {type(classification[field])}")
    for field in caption_boolean_fields:
        if not isinstance(caption_analysis[field], bool):
            raise ValueError(f"caption_analysis.{field} must be boolean, got {type(caption_analysis[field])}")
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
| **E13: api_rate_limit** | RateLimitError | Exceeded Anthropic API rate limits (429) | Retry 5x with exponential backoff (1s, 2s, 4s, 8s, 16s), then fail | Reduce batch size or wait, check status.anthropic.com |

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


# ===== ERROR HANDLER: API RATE LIMITING =====
# Error ID: E13
# Source: ContentAnalysisCHILDTI_CRITIQUE.md C2

def handle_api_rate_limit(
    api_call_func: callable,
    context: str,
    max_retries: int = 5,
    initial_backoff: float = 1.0
):
    """
    Handle API rate limit errors (429) with exponential backoff.

    Anthropic API Limits (as of 2025-01):
    - Claude Haiku: 50 requests/minute
    - Claude Sonnet: 50 requests/minute

    Args:
        api_call_func: Function to call (must raise RateLimitError on 429)
        context: Description for logging (e.g., "Video 123 classification")
        max_retries: Number of retry attempts (default: 5)
        initial_backoff: Initial backoff delay in seconds (default: 1.0)

    Returns:
        Result from api_call_func

    Raises:
        RateLimitError: After all retries exhausted
    """
    import anthropic

    backoff = initial_backoff

    for attempt in range(max_retries):
        try:
            return api_call_func()
        except anthropic.RateLimitError as e:  # 429 error
            if attempt < max_retries - 1:
                logger.warning(
                    f"⚠️ Rate limit hit for {context}. "
                    f"Retry {attempt + 1}/{max_retries} in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            else:
                logger.error(
                    f"❌ Rate limit exceeded for {context} after {max_retries} retries."
                )
                raise

    raise RuntimeError("Unreachable: retry loop exited unexpectedly")
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

        # Step 6: Calculate percentages (post-process LLM output)
        raw_taxonomy = calculate_percentages(raw_taxonomy, len(sampled_transcripts))

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

### 6.4 Rollback/Cleanup Procedures

**Source**: m4 Enhancement - Rollback procedures for Stage 2.7 failures

**Purpose**: Handle interrupted or failed classification runs

#### Option A: Resume from Checkpoint (Recommended)

If Stage 2.7 fails midway, use checkpoint to resume:

```bash
# Checkpoint automatically tracks progress
# Simply re-run classification - it will skip completed videos
python run_stage_2_7.py --hashtag nutrition

# Check checkpoint status
cat /data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/.checkpoints/classification_checkpoint.json
```

**When to use**: Always use this option (requires checkpoint_path parameter in Section 4.5)

#### Option B: Clean Restart

If checkpoint is corrupted or full restart needed:

```bash
# 1. Identify partial classifications
ls {bucket_base}/content_analysis/ | wc -l  # Count existing files

# 2. Remove all classification files
rm -rf {bucket_base}/content_analysis/*.json

# 3. Remove checkpoint
rm /data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/.checkpoints/classification_checkpoint.json

# 4. Re-run classification from scratch
python run_stage_2_7.py --hashtag nutrition
```

**When to use**: Checkpoint corrupted OR need to change taxonomy and reclassify all videos

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
├─ Validate output → ✓ All 12 fields present (refined schema)
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
├─ Validate output → ✓ All 12 fields present (refined schema)
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
- `calculate_percentages()` → Section 4.2.5
- `run_discovery_with_error_handling()` → Section 6.3

**classification.py**:
- `classify_video_llm()` → Section 4.3 (single video classification)
- `classify_all_videos()` → Section 4.4 (batch orchestrator with sequential/parallel modes)
- `classify_all_videos_sequential()` → Section 4.4.1 (sequential mode, default)
- `classify_all_videos_parallel()` → Section 4.4.2 (parallel mode, optional)
- `classify_single_video_with_save()` → Section 4.4.2 (helper for parallel execution)
- `load_checkpoint()` → Section 4.5.1 (load checkpoint from disk)
- `save_checkpoint()` → Section 4.5.2 (save checkpoint atomically)
- `update_checkpoint()` → Section 4.5.3 (update after single video)
- `log_estimated_cost()` → Section 4.6.1 (log estimated API costs)
- `log_actual_cost()` → Section 4.6.2 (log actual API costs from response)
- `run_classification_with_error_handling()` → Section 6.3

**validation.py**:
- `validate_discovery_inputs()` → Section 5.1
- `validate_classification_inputs()` → Section 5.1
- `validate_business_rules_sampling()` → Section 5.2
- `validate_business_rules_classification()` → Section 5.2
- `validate_discovery_output()` → Section 5.3
- `validate_classification_output()` → Section 5.3
- `validate_curated_taxonomy()` → Section 4.7.1 (validate manually curated taxonomy)

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
/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/
├── .checkpoints/
│   └── classification_checkpoint.json     # Resume state (Section 4.5)
│                                          # Format: {"completed": [...], "failed": [...]}
│                                          # Updated after each video
└── buckets/
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

**Checkpoint Behavior**:
- Created automatically when classification starts (if checkpoint_path provided)
- Updated after each video completes (atomic write)
- Enables resume if process interrupted
- Can be deleted to force fresh classification

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
RUMIAI_ROOT = "/home/jorge/rumiaifinal"    # Root directory for RumiAI installation
                                            # Default: "/home/jorge/rumiaifinal"
                                            # Used by: All stages for locating transcripts, unified_analysis, etc.
                                            # Override: export RUMIAI_ROOT=/custom/path
                                            # Note: Makes document portable across machines/users

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

ENABLE_PARALLEL_CLASSIFICATION = False      # Enable parallel video classification
                                            # Default: False (sequential mode)
                                            # Set to True for 5x speedup (10 min → 2 min for 120 videos)
                                            # Used by: classification.py
                                            # Note: Requires careful rate limit management

MAX_CLASSIFICATION_WORKERS = 5              # Concurrent workers for parallel classification
                                            # Default: 5, Range: 1-10
                                            # Conservative setting to avoid rate limits (50 req/min)
                                            # Used by: classification.py (only if ENABLE_PARALLEL_CLASSIFICATION=True)
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

### 10.4 LLM API Call Logging

**Source**: M8 Enhancement - Comprehensive LLM API logging specification

**Purpose**: Track API costs, latency, and token usage for monitoring and debugging

---

#### 10.4.1 Discovery (Stage 2.6) API Call Logging

**Before API call** (INFO level):
```python
logger.info(f"💰 Estimated cost for discovery: $0.75 (Sonnet API call, ~50 transcripts)")
```

**After API call** (DEBUG level):
```python
logger.debug(f"💸 API call: $0.7523, 47.32s, in: 84,532 tokens, out: 2,418 tokens, model: sonnet")
```

**Complete Flow Example**:
```
[2025-01-28 10:30:00] [INFO] 🔍 Starting discovery for #nutrition
[2025-01-28 10:30:05] [INFO] 📊 Sampled 50 transcripts from 3 buckets
[2025-01-28 10:30:05] [INFO] 💰 Estimated cost for discovery: $0.75 (Sonnet API call, ~50 transcripts)
[2025-01-28 10:30:05] [INFO] 🤖 Calling Claude 3.5 Sonnet for pattern discovery...
[2025-01-28 10:30:52] [DEBUG] 💸 API call: $0.7523, 47.32s, in: 84,532 tokens, out: 2,418 tokens, model: sonnet
[2025-01-28 10:30:52] [INFO] ✅ Discovery complete: /data/.../nutrition_raw_discovery.json
```

---

#### 10.4.2 Classification (Stage 2.7) API Call Logging

**Before batch classification** (INFO level):
```python
logger.info(f"💰 Estimated cost for classification: $0.12 (Haiku API calls, 120 videos)")
```

**Per-video API call** (DEBUG level):
```python
logger.debug(f"💸 API call: $0.0012, 2.51s, in: 451 tokens, out: 183 tokens, model: haiku")
```

**After batch classification** (INFO level):
```python
logger.info(f"✅ Classification complete: 120/120 videos (estimated cost: $0.12)")
```

**Complete Flow Example**:
```
[2025-01-28 10:35:00] [INFO] 🏷️  Starting classification for 120 videos
[2025-01-28 10:35:00] [INFO] 💰 Estimated cost for classification: $0.12 (Haiku API calls, 120 videos)
[2025-01-28 10:35:00] [INFO] 📋 Sequential mode with checkpoints: 0 already completed, 120 remaining
[2025-01-28 10:35:03] [DEBUG] 💸 API call: $0.0012, 2.51s, in: 451 tokens, out: 183 tokens, model: haiku
[2025-01-28 10:35:03] [DEBUG] ✅ Classified 1/120: 7526250443832331550
[2025-01-28 10:35:06] [DEBUG] 💸 API call: $0.0011, 2.48s, in: 442 tokens, out: 179 tokens, model: haiku
[2025-01-28 10:35:06] [DEBUG] ✅ Classified 2/120: 7428596413707144481
... (118 more videos)
[2025-01-28 10:41:12] [INFO] ✅ Classification complete: 120/120 videos (estimated cost: $0.12)
```

---

#### 10.4.3 API Call Metrics Summary

**Logged Metrics** (all in DEBUG level):
| Metric | Format | Example | Purpose |
|--------|--------|---------|---------|
| **Cost** | `$0.0012` | `$0.7523` | Track actual API spending |
| **Latency** | `47.32s` | `2.51s` | Debug slow API calls |
| **Input Tokens** | `84,532 tokens` | `451 tokens` | Understand prompt size |
| **Output Tokens** | `2,418 tokens` | `183 tokens` | Monitor response length |
| **Model** | `sonnet`, `haiku` | `sonnet` | Differentiate API calls |

**Log Format** (standardized):
```
💸 API call: ${cost:.4f}, {latency:.2f}s, in: {input_tokens:,} tokens, out: {output_tokens:,} tokens, model: {model}
```

---

#### 10.4.4 Monitoring and Alerting

**Key Metrics to Monitor**:
1. **Average latency per model**:
   - Sonnet (discovery): Expected 40-60s, Alert if > 90s
   - Haiku (classification): Expected 2-5s, Alert if > 10s

2. **Cost accumulation**:
   - Discovery: Expected $0.75/hashtag, Alert if > $1.50
   - Classification: Expected $0.001/video, Alert if > $0.005/video

3. **Token usage trends**:
   - Discovery: Expected 80-90K input tokens
   - Classification: Expected 400-500 input tokens per video

**Example Monitoring Query** (log aggregation):
```bash
# Calculate average latency for Haiku calls
grep "💸 API call" content_analysis.log | grep "model: haiku" | awk -F', ' '{print $2}' | awk '{sum+=$1; n++} END {print sum/n "s"}'

# Calculate total cost for session
grep "💸 API call" content_analysis.log | awk -F'$' '{print $2}' | awk -F',' '{sum+=$1} END {print "$" sum}'
```

---

## Section 11: Dependencies & Prerequisites

<!-- Source: ContentAnalysisCHILD.md Section 3 -->

### 11.1 Python Dependencies

**Python Version**: 3.9+ (tested on 3.9, 3.10, 3.11)

```python
# requirements.txt
anthropic==0.40.0          # Claude API client
                           # Compatibility: 0.40.x - 0.42.x (API stable)
                           # Breaking change: 0.43.0+ changes Message API

pydantic==2.10.0           # Schema validation
                           # Compatibility: 2.8.x - 2.10.x
                           # Note: Pydantic 1.x NOT supported (major API changes)

python-dotenv==1.0.0       # Environment variable loading
                           # Compatibility: 1.0.x
                           # Any 1.x version compatible (stable API)
```

**Compatibility Notes**:
- **Critical**: anthropic library must be < 0.43.0 (API breaking change)
- **Python 3.8**: Not tested, may work but not recommended
- **Python 3.12+**: Not tested, likely compatible
- **OS**: Linux, macOS, Windows (WSL recommended for Windows)

**Update Strategy**:
```bash
# Safe update (patch versions only)
pip install --upgrade anthropic~=0.40.0 pydantic~=2.10.0

# Check for breaking changes before major updates
pip list --outdated
```

**Language Limitation** (m12):
- **English only**: System assumes English transcripts
- Whisper (Stage 2) transcribes in English
- Taxonomy patterns are English-based (e.g., "pain_points", "keywords")
- LLM prompts are in English
- Non-English videos: Will produce inaccurate classifications (not supported)

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

## Section 12: Glossary

<!-- Source: m9 Enhancement - Define domain-specific terms -->

### 12.1 RumiAI Pipeline Terms

| Term | Definition |
|------|------------|
| **Bucket** | Duration-based grouping of videos (e.g., "bucket_33-60s" contains videos 33-60 seconds long). Used to group similar-length videos for analysis. |
| **Bucket Base** | Base directory path for a bucket's outputs: `/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive/buckets/bucket_{duration}/` |
| **Contrastive Analysis** | Comparison of top performers (80th percentile) vs bottom performers (20th percentile) to identify success patterns. Selection strategy for Stage 2.5. |
| **Temporal Window** | Time segment within a video (e.g., Hook: 0-3s, Middle: 3-57s, Closing: 57-60s). Used in Stage 2 ML analysis to extract time-based features. |
| **Top Performers** | Videos in the 80th percentile by engagement score within a duration bucket. Primary focus for pattern discovery. |
| **Bottom Performers** | Videos in the 20th percentile by engagement score. Used for contrastive comparison (not included in Stage 2.6 discovery). |

### 12.2 Stage-Specific Terms

| Term | Definition |
|------|------------|
| **Discovery (Stage 2.6)** | Pattern discovery phase using LLM (Claude Sonnet) to identify common content patterns across 50 sampled top-performer transcripts. |
| **Classification (Stage 2.7)** | Video classification phase using LLM (Claude Haiku) to label each video with discovered patterns from curated taxonomy. |
| **Curated Taxonomy** | Manually refined taxonomy after Stage 2.6 raw discovery. Human curator removes rare patterns, merges similar ones, and fixes naming. |
| **Raw Discovery** | Initial LLM output from Stage 2.6 before manual curation. Contains all discovered patterns with frequencies and examples. |
| **Taxonomy** | Structured classification system with 6 categories: content_categories, hook_strategies, audience_pain_points, trending_keywords, engagement_drivers, content_tactics. |

### 12.3 ML & Data Terms

| Term | Definition |
|------|------------|
| **Transcript** | Speech-to-text output from Whisper (Stage 2). Contains all spoken words in the video. Source: `/home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json` |
| **Caption** | Creator-written text accompanying TikTok video. May include hashtags, CTAs, and emojis. Source: unified_analysis. |
| **Hashtags** | Keywords from video caption (e.g., ["fitness", "workout"]). Used for video discovery and classification context. |
| **Engagement Score** | Composite metric combining likes, comments, shares, and views. Used to rank videos as top/bottom performers. |
| **Selection Manifest** | Output from Stage 2.5 containing selected video IDs grouped by bucket and performance tier. Input for Stage 2.6/2.7. |

### 12.4 Technical Terms

| Term | Definition |
|------|------------|
| **Checkpoint** | JSON file tracking completed/failed video classifications. Enables resume after interruption. Location: `.checkpoints/classification_checkpoint.json` |
| **Sequential Mode** | Default classification mode processing one video at a time (10 min for 120 videos). Safe, simple, no concurrency. |
| **Parallel Mode** | Optional classification mode using ThreadPoolExecutor with 5 workers (2 min for 120 videos). 5x speedup with rate limit safety. |
| **Snake_case** | Naming convention with lowercase letters, numbers, and underscores only (e.g., "recipe_tutorial"). Required for taxonomy category names. |
| **Latency** | Time in seconds for API call to complete (e.g., 47.32s for Sonnet discovery, 2.51s for Haiku classification per video). |

### 12.5 Model Terms

| Term | Definition |
|------|------------|
| **Claude 3.5 Sonnet** | Large language model used for Stage 2.6 discovery. Higher quality, slower, more expensive ($15/1M input tokens). |
| **Claude 3 Haiku** | Faster language model used for Stage 2.7 classification. Lower cost ($0.25/1M input tokens), sufficient for classification task. |
| **Token** | Unit of text for LLM pricing. Roughly 0.75 words = 1 token. Used to calculate API costs and validate prompt sizes. |
| **Input Tokens** | Tokens in the prompt sent to LLM (transcript + taxonomy + instructions). |
| **Output Tokens** | Tokens in the LLM response (classification JSON or discovered patterns JSON). |

### 12.6 Acronyms

| Acronym | Full Term | Usage |
|---------|-----------|-------|
| **TI** | Technical Implementation | This document type |
| **HLD** | High-Level Design | Parent design document (ContentAnalysisCHILD.md) |
| **LLM** | Large Language Model | Claude Sonnet/Haiku |
| **API** | Application Programming Interface | Anthropic Claude API |
| **JSON** | JavaScript Object Notation | Data format for all inputs/outputs |
| **CTA** | Call To Action | Caption element prompting user action |

---

## Section 13: HLD Traceability Matrix

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

