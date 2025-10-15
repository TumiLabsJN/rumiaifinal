# ContentAnalysis.md - Independent Content Intelligence System
## Brainstorm
**Captions/Hashtag**
Stage 3.2.5 selects videos from /insights, and distributes them to the unified buckets. 

A. Add to Temporal_compute analysis: 
   - Captions
   - All hashtags used

Then, we can refactor part of stage 3.2.5 , to also extract captions and hashtags used and pass them to a new .JSON which Content analysis will use

**Transcriptions**
For Speech Transcription, its current output: rumiaifinal\speech_transcriptions

We need to find a way to move the transcriptions to their relevant bucket duration file (Right now we just move temporal)

### Roadmap
1. After you run your tests, make a copy of the transcripts and place them in /home/jorge/rumiaifinal/ContentAnalysis/



## Overview

**Purpose**: Analyze video content (transcripts, captions, hashtags) to extract content intelligence insights independent of the ML pipeline constraints.

**Key Insight**: We have rich text data (transcripts, descriptions, hashtags) but feeding them to RF/KMeans requires lossy transformations (TF-IDF, embeddings). Solution: Create a separate, semi-structured analysis layer that preserves the full semantic richness of content.

**Content Sources:**
1. **Transcripts** - Speech-to-text from Whisper (what's being said in the video)
2. **Description** - Creator-written captions/video descriptions (how creators position their content)
3. **Hashtag Analysis** - Hashtag strategy metadata (what topics/communities creators target)

## System Design Philosophy

### Structure Spectrum
- **ML Pipeline**: Highly structured (numerical features, fixed schema, normalized ranges)
- **Content Analysis**: Semi-structured (flexible schema, qualitative insights, human-readable)
- **Raw Transcripts**: Unstructured (free-form text)

### Benefits of Semi-Structured Approach
- ✅ Flexible schema - can adapt as we learn
- ✅ Rich categorization - strings, arrays, nested objects
- ✅ Human-readable - easy to understand and act on
- ✅ Queryable - can search/filter/group
- ✅ No normalization constraints - preserves nuanced insights

## Taxonomy-Based Methodology

### Core Principle: Hashtag-Specific Taxonomies

**Why Not Universal Taxonomy?**
Each hashtag represents a **self-contained content ecosystem** with unique:
- Language patterns (#fitness vs #nutrition vs #beauty)
- Hook strategies (wellness uses "secret", tech uses "hack")
- Audience signals (different demographics, pain points)
- Content categories (WIEIAD works for food, not tech tutorials)

**Approach**: Train independent taxonomies per hashtag, similar to how K-means models use separate scalers per duration bucket.

---

## Three-Step Workflow

### **Step 1: Train Each Hashtag (Automated)**

Download and process videos from target hashtag:

```python
# Already in pipeline - Stage 3.2.5
videos = apify.scrape_hashtag("#nutrition", count=300)
for video in videos:
    transcript = whisper.transcribe(video)
    # Store in /ContentAnalysis/nutrition/transcripts/
```

**Output**: 300 videos per hashtag with transcripts, captions, hashtag metadata.

---

### **Step 2: Map Taxonomy for Each Hashtag (Hybrid)**

#### **2A: Automated Pattern Discovery (LLM-Based)**

Use LLM to analyze 50-100 transcripts and identify natural patterns:

```python
def discover_taxonomy(hashtag, transcripts):
    """
    LLM analyzes transcripts to discover patterns automatically
    """
    prompt = f"""
    Analyze {len(transcripts)} transcripts from #{hashtag} on TikTok.

    Identify natural patterns in:
    1. Content Categories (what types of videos exist?)
    2. Hook Strategies (how do videos start?)
    3. Audience Pain Points (what problems are mentioned?)
    4. Engagement Drivers (what makes content shareable?)
    5. Trending Keywords (what terms appear frequently?)

    Return JSON with discovered patterns including:
    - Pattern name
    - Frequency (how many videos exhibit this pattern)
    - Representative examples
    """

    taxonomy = claude_api.analyze(prompt)
    return taxonomy
```

**Output Example**:
```json
{
  "hashtag": "nutrition",
  "discovered_patterns": {
    "content_categories": [
      {"name": "what_i_eat_in_a_day", "frequency": 45, "examples": ["..."]},
      {"name": "recipe_tutorial", "frequency": 32, "examples": ["..."]},
      {"name": "supplement_review", "frequency": 18, "examples": ["..."]},
      {"name": "videos_mentioning_coffee", "frequency": 8, "examples": ["..."]}
    ],
    "hook_strategies": [
      {"name": "direct_statement", "frequency": 38, "examples": ["..."]},
      {"name": "problem_solution", "frequency": 27, "examples": ["..."]}
    ]
  }
}
```

✅ **This step is fully automated** - no human reading 300 transcripts.

---

#### **2B: Human Curation (Business Decisions)**

**Why Human Review is Required:**

Unlike numerical ML (where accuracy/loss guide optimization), content taxonomy quality is **subjective** and requires business judgment:

| Decision Type | Question | Why It Can't Be Automated |
|---------------|----------|---------------------------|
| **Actionability** | Is "videos_mentioning_coffee" useful? | Depends on business goals (competitive analysis vs creator coaching) |
| **Granularity** | Keep "recipe_tutorial" or split into "recipe_under_60s" vs "recipe_over_60s"? | Depends on report complexity needs |
| **Terminology** | Use "narrative_arc_transformation" or "before_after_story"? | Depends on audience (creators understand simpler terms) |
| **Signal vs Noise** | Is this pattern meaningful or coincidental? | Requires domain expertise |

**Human Curation Workflow**:

```python
# 1. LLM discovers patterns (automated)
raw_taxonomy = discover_taxonomy("#nutrition", transcripts)

# 2. Human reviews discovered patterns
"""
Discovered Categories:
☐ what_i_eat_in_a_day (45 videos) → KEEP (high frequency, actionable)
☐ recipe_tutorial (32 videos) → KEEP (high frequency, actionable)
☐ supplement_review (18 videos) → KEEP (actionable for affiliates)
☐ macro_breakdown (12 videos) → REMOVE (too niche, <10% threshold)
☐ videos_with_coffee_mentions (8 videos) → REMOVE (not actionable)
☐ grocery_haul (7 videos) → REMOVE (too few examples)
"""

# 3. Save curated taxonomy (manual decision)
curated_taxonomy = {
  "hashtag": "nutrition",
  "content_categories": [
    "what_i_eat_in_a_day",
    "recipe_tutorial",
    "supplement_review"
  ],
  "hook_strategies": [
    "direct_statement",
    "problem_solution"
  ],
  "audience_pain_points": [
    "bloating",
    "low_energy",
    "food_prep_fatigue"
  ],
  "trending_keywords": [
    "protein",
    "gut_health",
    "metabolism"
  ]
}

save_json("taxonomies/nutrition_taxonomy.json", curated_taxonomy)
```

**Cost**: One-time per hashtag (like training a scaler once, then reusing it).

---

### **Step 3: Apply Taxonomy to Videos (Automated)**

Once taxonomy is defined, classification is fully automated:

```python
def analyze_video(video_id, transcript, hashtag):
    """
    Classify video using saved hashtag-specific taxonomy
    (Similar to scaler.transform() using saved parameters)
    """
    # Load saved taxonomy (like loading a scaler)
    taxonomy = load_json(f"taxonomies/{hashtag}_taxonomy.json")

    # LLM classifies video using taxonomy
    prompt = f"""
    Classify this video using these predefined categories:

    Content Categories: {taxonomy['content_categories']}
    Hook Strategies: {taxonomy['hook_strategies']}
    Audience Pain Points: {taxonomy['audience_pain_points']}

    Transcript: {transcript}

    Return JSON with classifications.
    """

    analysis = claude_api.analyze(prompt)

    # Save to separate location (not mixed with ML features)
    save_json(f"content_analysis/{hashtag}/{video_id}.json", analysis)

    return analysis
```

✅ **No human in the loop** - consistent categorization across all videos.

---

## Automation Summary

| Step | Automated? | Human Effort Required |
|------|-----------|------------------------|
| **1. Download videos & transcripts** | ✅ Fully automated | None |
| **2A. Pattern discovery (LLM)** | ✅ Fully automated | None |
| **2B. Pattern curation** | ❌ **Manual** | - Select actionable categories<br>- Define granularity<br>- Choose creator-friendly terminology<br>- Filter noise |
| **3. Apply taxonomy to videos** | ✅ Fully automated | None |

**Key Insight**: Human curation (Step 2B) is the only manual step, and it's a **one-time cost per hashtag** that ensures business-relevant insights.

---

## File Structure

```
ContentAnalysis/
├── taxonomies/                      # Saved taxonomies (like scalers)
│   ├── nutrition_taxonomy.json      # Curated, production-ready
│   ├── fitness_taxonomy.json
│   └── beauty_taxonomy.json
│
├── raw_discoveries/                 # LLM outputs before curation
│   ├── nutrition_raw.json           # For reference/versioning
│   └── fitness_raw.json
│
├── training_transcripts/            # Sample transcripts used for discovery
│   ├── nutrition/
│   │   └── [18 sample transcripts]
│   └── fitness/
│       └── [sample transcripts]
│
└── content_analysis/                # Final video classifications
    ├── nutrition/
    │   └── {video_id}_content.json
    └── fitness/
        └── {video_id}_content.json
```

---

## Integration with ML Pipeline

### Current Flow (RF/K-means):
```
Videos → ML Services → temporal_windows_updated.json (numerical features)
```

### New Parallel Flow (Content Analysis):
```
Videos → Transcripts + Captions + Hashtags → content_analysis.json (qualitative insights)
```

**Key Design Decision**: Content analysis runs **independently** and outputs to **separate JSON files**. This avoids:
- Mixing numerical and qualitative data
- Breaking existing ML pipeline
- Forcing normalization on text data

### Example Integration in Stage 3.2.5:

```python
# After selecting videos for buckets
for video in selected_videos:
    # Existing: Load temporal features
    temporal_features = load_json(f"insights/{video.id}_temporal_windows_updated.json")

    # New: Load content analysis (if exists)
    content_analysis = load_json(f"content_analysis/{hashtag}/{video.id}_content.json")

    # Combine in reporting (but store separately)
    bucket_data[video.id] = {
        "temporal_features": temporal_features,
        "content_analysis": content_analysis  # Optional, for richer reporting
    }
```

---

## Semi-Automation Opportunity (After 2-3 Hashtags)

Once you've curated taxonomies for 2-3 hashtags, you'll notice **patterns in your curation decisions**:

```python
def auto_filter_taxonomy(raw_taxonomy):
    """
    Apply learned heuristics from previous curation rounds
    """
    filtered = {}

    for category in raw_taxonomy['content_categories']:
        # Rule 1: Must have >10% frequency (learned from experience)
        if category['frequency'] < 10:
            continue

        # Rule 2: No brand-specific categories (business rule)
        if "brand" in category['name'].lower():
            continue

        # Rule 3: Avoid hyper-specific temporal splits (granularity rule)
        if any(term in category['name'] for term in ["under_", "over_", "between_"]):
            continue

        filtered[category['name']] = category

    return filtered

# Still requires final human review, but reduces workload 70%
```

**Result**: After 3-4 hashtags, curation time drops from **2 hours → 30 minutes per hashtag**.

---

## Phase 3: Analysis Framework Implementation
**Target Output Structure**:
```json
{
  "video_id": "123456789",
  "transcript_analysis": {
    "content_category": {
      "primary": "fitness_transformation",
      "secondary": ["motivation", "personal_story"],
      "confidence": "high"
    },
    "hook_strategy": {
      "type": "vulnerable_confession",
      "effectiveness": "high",
      "pattern": "problem_reveal_to_solution"
    },
    "audience_signals": {
      "primary_demographic": "women_18_35",
      "experience_level": "beginner_friendly",
      "pain_points": ["body_confidence", "motivation"],
      "aspirations": ["transformation", "lifestyle_change"]
    },
    "content_strategy": {
      "authenticity": "very_high",
      "relatability": "high",
      "educational_value": "medium",
      "entertainment_factor": "high"
    },
    "engagement_drivers": [
      "before_after_reveal",
      "relatable_struggle",
      "specific_metrics",
      "inspirational_tone"
    ],
    "viral_indicators": {
      "emotional_arc": "struggle_to_triumph",
      "shareability_factors": ["inspiration", "relatability"],
      "trending_keywords": ["transformation", "journey"],
      "call_to_action_strength": "medium"
    },
    "market_intelligence": {
      "topic_saturation": "medium",
      "competition_density": "high",
      "content_gap_opportunities": ["specific_workout_plans"],
      "seasonal_relevance": "new_year_peak"
    }
  }
}
```

## Content Data Sources

### 1. Transcripts
**Location**: `/speech_transcriptions/`
**Overview**: 160+ transcript files ready for analysis
**Source**: Whisper speech-to-text service

**File Format**: JSON files from Whisper
```
/home/jorge/rumiaifinal/speech_transcriptions/
├── 126301987701056_whisper.json
├── 595997271203511_whisper.json
├── 7480428850522950920_whisper.json
└── ... (160 total files)
```

**JSON Structure**:
```json
{
  "text": "Full transcript text for analysis...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 6.8,
      "text": "Segment text with timestamps...",
      "words": [
        {
          "word": "So",
          "start": 0.06,
          "end": 0.22,
          "confidence": 0.515381
        }
      ]
    }
  ]
}
```

**Key Fields for Analysis**:
- `"text"`: Complete transcript (primary content for pattern discovery)
- `"segments"`: Timestamped chunks (useful for hook analysis - first 3 seconds)
- `"words"`: Word-level data with confidence scores (quality filtering)

---

### 2. Description (Captions)
**Source**: Apify TikTok scraper
**Field**: `description` from video metadata
**Overview**: Creator-written captions that appear below the video

**Data Type**: String (50-150 characters typical, up to 2,200 max)

**What It Contains**:
- Hook/headline (attention grabber)
- Key details/value proposition
- Call-to-action (Save, Comment, Follow, Link in bio)
- Hashtags (3-10 typical)
- Emojis (visual emphasis)

**Real Examples**:

```json
{
  "video_id": "7480428850522950920",
  "description": "Best protein shake recipe! 🍓 High protein, low calorie, tastes amazing. Link in bio for full recipe 👆 #nutrition #protein #healthyrecipes #fitnessmotivation #weightloss"
}
```

```json
{
  "video_id": "595997271203511",
  "description": "10 min ab workout 🔥 no equipment needed! Save this for later 💪 #fitness #workout #abs #homeworkout #fitnessmotivation"
}
```

```json
{
  "video_id": "126301987701056",
  "description": "How I got 100K followers in 30 days (no paid ads) 📈 Full strategy in comments 👇 #socialmedia #marketing #entrepreneur #growthhacking #tiktokgrowth"
}
```

**Why ContentAnalysis vs ML Features?**
- **Rich semantic content**: "Best protein shake recipe" conveys much more than TF-IDF word counts
- **Strategic messaging**: CTA patterns ("Save this", "Link in bio") are qualitative insights
- **Creator intent**: How creators position their content reveals strategy
- **Lossy transformation problem**: Converting to 50-100 numerical features loses nuanced meaning

**Analysis Opportunities**:
- Hook pattern identification (question, statement, teaser, command)
- CTA effectiveness by type (save, comment, link, follow)
- Hashtag strategy correlation with performance
- Emoji usage patterns in viral content
- Caption length vs engagement analysis
- Positioning strategy (educational, inspirational, promotional)

---

### 3. Hashtag Analysis
**Source**: Apify TikTok scraper (processed)
**Field**: `hashtag_analysis` object from video metadata
**Overview**: Categorized hashtag strategy metadata

**Data Type**: Object (nested structure with counts and classifications)

**Structure**:
```json
{
  "video_id": "7480428850522950920",
  "hashtag_analysis": {
    "total_count": 5,
    "branded_hashtags": ["#fitnessmotivation"],
    "community_hashtags": ["#nutrition", "#protein", "#healthyrecipes"],
    "trending_hashtags": ["#weightloss"],
    "hashtag_count": 5,
    "branded_hashtag_count": 1,
    "community_hashtag_count": 3,
    "trending_hashtag_count": 1
  }
}
```

**Why ContentAnalysis vs ML Features?**
- **Categorical data**: Hashtag types are qualitative (branded vs community vs trending)
- **Strategic insight**: Which hashtag mix drives engagement is a content pattern, not a numerical feature
- **Context-dependent**: #fitness means different things in different niches
- **List/array data**: Multiple hashtags per video don't normalize well to single features

**Analysis Opportunities**:
- Optimal hashtag mix (branded vs community vs trending ratio)
- Hashtag saturation analysis (overused vs underutilized)
- Hashtag strategy by content type (educational vs promotional)
- Trending hashtag timing (early adoption vs late adoption)
- Community hashtag clustering (which combos work together)
- Branded hashtag effectiveness by niche

### Simple Training Method
**Direct Repository Access**:
```bash
# Read transcript directly from repo
cat /home/jorge/rumiaifinal/speech_transcriptions/{video_id}_whisper.json

# Extract just the text field for analysis
jq -r '.text' /home/jorge/rumiaifinal/speech_transcriptions/*.json
```

**Advantages**:
- ✅ 160 transcripts immediately available
- ✅ No export/preparation needed
- ✅ Rich metadata included (timestamps, confidence)
- ✅ Diverse content already collected
- ✅ Production-ready data format

## Transcript Delivery Methods

### Option 1: Direct Repository Reading (Recommended)
**Approach**: Read transcript files directly from `/speech_transcriptions/`
```bash
# Random sampling for diversity
ls /speech_transcriptions/ | shuf | head -50
```

**Pros**:
- No file preparation needed
- 160 transcripts immediately available
- Rich JSON structure with metadata
- Proven data quality (production transcripts)

**Cons**:
- Need to extract text field from JSON
- File access required

### Option 2: Database Export
**Approach**: Export transcripts from existing speech data
```sql
SELECT video_id, transcript_text, views, engagement_rate
FROM video_analysis
WHERE transcript_text IS NOT NULL
ORDER BY RANDOM()
LIMIT 50;
```

**Pros**:
- Automated data collection
- Can include performance metrics
- Representative sampling

**Cons**:
- Requires database access
- May hit message length limits

### Option 3: Streaming Analysis
**Approach**: Feed transcripts one-by-one with interactive analysis
```
1. Send transcript #1 → Get initial patterns
2. Send transcript #2 → Refine patterns
3. Continue iteratively → Build comprehensive taxonomy
```

**Pros**:
- Interactive pattern discovery
- Can adjust approach based on findings
- No length limitations

**Cons**:
- Time-intensive process
- Requires more coordination

### Option 4: Hybrid Approach (Recommended)
**Phase 1**: Start with 10 diverse transcripts via streaming
**Phase 2**: Batch upload 40 more based on initial pattern insights
**Phase 3**: Interactive refinement of discovered categories

## Implementation Strategy

### Step 1: Sample Collection
**Target Diversity**:
- **Topics**: Fitness, cooking, tech, lifestyle, business, education
- **Creator Types**: Micro-influencers, established creators, brands
- **Performance Levels**: Viral (>1M views), average (10K-100K), low (<10K)
- **Content Styles**: Educational, entertainment, promotional, personal
- **Demographics**: Various age groups, genders, regions

### Step 2: Pattern Discovery Session
**Process**:
1. Feed Claude transcripts with minimal context
2. Ask for natural pattern identification
3. Iteratively refine category systems
4. Build comprehensive taxonomy

### Step 3: Framework Development
**Deliverables**:
- Content categorization system
- Analysis output schema
- Implementation guidelines
- Quality assurance criteria

### Step 4: Validation & Refinement
**Testing**:
- Apply framework to new transcripts
- Validate category accuracy
- Refine based on edge cases
- Optimize for actionable insights

## Success Metrics

### Discovery Phase
- ✅ Identify 10-15 distinct content categories
- ✅ Map 5-10 common hook strategies
- ✅ Define 8-12 audience targeting signals
- ✅ Catalog 6-8 content strategy archetypes

### Implementation Phase
- ✅ Consistent categorization across test videos
- ✅ Actionable insights for creators
- ✅ Queryable content intelligence database
- ✅ Integration with existing video analysis pipeline

## Future Applications

### Creator Tools
- **Content Strategy Recommendations**: Based on successful patterns
- **Audience Targeting Insights**: Language cues for demographic optimization
- **Trend Analysis**: Emerging topics and declining themes
- **Competitive Intelligence**: Content gap identification

### Platform Intelligence
- **Viral Content Prediction**: Pattern-based success indicators
- **Content Curation**: Semantic search and recommendation
- **Market Analysis**: Topic saturation and opportunity mapping
- **Creator Development**: Personalized improvement suggestions

## Pipeline Integration & Architecture

### Integration Point: ML Training Pipeline

**Start Location:** Stage 2.5 (after bucket selection in MLPlanningv2.md)

**Why Start Here?**
- ✅ Videos already filtered to top 3 performing duration buckets
- ✅ Only analyze videos that matter (efficient use of LLM API)
- ✅ Clean separation from existing ML pipeline
- ✅ Natural checkpoint after bucket selection

---

### Detailed Pipeline Flow

```
Stage 2: ML Processing (Existing)
    ├── Generate insights → insights/{video_id}_temporal_windows_updated.json
    └── Generate transcripts → speech_transcriptions/{video_id}_whisper.json
    ↓
Stage 2.5: Bucket Selection (Existing)
    ├── Select top 3 duration buckets
    ├── Move temporal features → ml_training_data/{hashtag}/bucket_{duration}/
    └── (NEW) Save selection manifest → ml_training_data/{hashtag}/selection_manifest.json
    ↓
Stage 2.6: Content Analysis Discovery (NEW - One-Time per Hashtag)
    ├── Check if taxonomy exists (taxonomies/{hashtag}_taxonomy.json)
    ├── If NOT exists:
    │   ├── Sample 50-100 transcripts from selected buckets
    │   ├── LLM pattern discovery (automated)
    │   └── Save raw patterns → raw_discoveries/{hashtag}_raw.json
    ├── Human curation (manual, one-time)
    └── Save curated taxonomy → taxonomies/{hashtag}_taxonomy.json
    ↓
Stage 2.7: Content Analysis Application (NEW - Every Run)
    ├── Load saved taxonomy
    ├── Classify videos in top 3 buckets
    │   ├── Load transcript from speech_transcriptions/
    │   ├── Load caption/hashtags from insights/
    │   └── LLM classification using taxonomy
    └── Save classifications → content_analysis/{hashtag}/{bucket}/{video_id}_content.json
    ↓
Stage 3-6: ML Training (Existing - Unchanged)
    ├── Random Forest training
    └── K-means clustering
    ↓
Stage 7: LLM Report Generation (ENHANCED)
    ├── Load ML insights (feature importance)
    ├── Load content insights (NEW)
    └── Generate combined creative reports
```

---

### Selection Manifest (Stage 2.5 Enhancement)

**Purpose:** Provides video list for Content Analysis without file searching

**Location:** `ml_training_data/{hashtag}/selection_manifest.json`

**Structure:**
```json
{
  "hashtag": "nutrition",
  "selected_buckets": ["33_60s", "60_90s", "90_120s"],
  "videos_by_bucket": {
    "33_60s": {
      "top_performers": ["video_id_1", "video_id_2", ...],
      "bottom_performers": ["video_id_50", ...]
    },
    "60_90s": {
      "top_performers": [...],
      "bottom_performers": [...]
    },
    "90_120s": {
      "top_performers": [...],
      "bottom_performers": [...]
    }
  },
  "total_videos": 300,
  "timestamp": "2025-10-13T10:30:00Z"
}
```

**Why This Matters:**
- Content Analysis reads this manifest (no glob searching)
- Knows exactly which videos to analyze
- Can sample strategically (e.g., 20 top performers per bucket)

---

### Directory Structure (Full System)

```
rumiaifinal/
├── ContentAnalysis/                          # NEW - Content Analysis Module
│   ├── taxonomies/                           # Curated, production-ready
│   │   ├── nutrition_taxonomy.json
│   │   └── fitness_taxonomy.json
│   │
│   ├── raw_discoveries/                      # LLM outputs before curation
│   │   ├── nutrition_raw.json
│   │   └── fitness_raw.json
│   │
│   ├── training_transcripts/                 # Sample transcripts for discovery
│   │   └── nutrition/
│   │       └── [18 sample transcripts]
│   │
│   └── content_analysis/                     # Final classifications (parallel to ml_training_data)
│       └── nutrition/
│           ├── bucket_33_60s/
│           │   ├── video_123_content.json
│           │   └── video_456_content.json
│           ├── bucket_60_90s/
│           │   └── video_789_content.json
│           └── bucket_90_120s/
│               └── video_abc_content.json
│
├── ml_training_data/                         # EXISTING - ML Pipeline
│   └── nutrition/
│       ├── selection_manifest.json           # ENHANCED - Added by Stage 2.5
│       ├── bucket_33_60s/
│       │   ├── top_performers/
│       │   │   └── {video_id}_temporal_windows_updated.json
│       │   └── bottom_performers/
│       └── bucket_60_90s/
│
├── insights/                                 # EXISTING - Temporal features + captions
│   └── {video_id}_temporal_windows_updated.json
│
├── speech_transcriptions/                    # EXISTING - Whisper outputs
│   └── {video_id}_whisper.json
│
└── scripts/                                  # SCRIPTS
    ├── rumiai_runner.py                      # Stage 1-2 (existing)
    ├── ml_training_pipeline.py               # Stage 2.5-6 (existing)
    ├── content_analysis_discovery.py         # Stage 2.6 (NEW)
    ├── content_analysis_application.py       # Stage 2.7 (NEW)
    └── llm_report_generator.py               # Stage 7 (enhanced)
```

---

## LLM API Usage & Costs

### Stage 2.6: Discovery (One-Time per Hashtag)

**API Call:**
```python
# Analyze 50-100 transcripts to discover patterns
prompt = """
Analyze 50 transcripts from #nutrition.
Identify patterns in content categories, hook strategies, etc.
Return JSON.
"""
```

**Cost Estimate:**
- Input: ~50,000 tokens (50 transcripts × 1,000 tokens)
- Output: ~2,000 tokens (JSON response)
- Model: Claude 3.5 Sonnet
- **Cost: ~$0.75 per hashtag (one-time)**

---

### Stage 2.7: Application (Per Video)

**API Call:**
```python
# Classify each video using saved taxonomy
prompt = f"""
Classify this video using predefined taxonomy:

Content Categories: {taxonomy['content_categories']}
Hook Strategies: {taxonomy['hook_strategies']}

Transcript: {transcript}
Caption: {caption}
Hashtags: {hashtags}

Return JSON classification.
"""
```

**Cost Estimate (Per Video):**
- Input: ~2,000 tokens (transcript + taxonomy)
- Output: ~300 tokens (classification)
- Model: **Claude 3 Haiku** (recommended for classification)
- **Cost: ~$0.001 per video (0.1 cent)**

---

### Total Cost Breakdown

#### Option 1: Selective Analysis (Recommended for MVP)

**Configuration:**
- Analyze: 40 videos per hashtag (20 top + 20 bottom per bucket × top 3 buckets)
- Model: Claude 3 Haiku

| Stage | Cost per Hashtag | Frequency |
|-------|------------------|-----------|
| **2.6: Discovery** | $0.75 | One-time |
| **2.7: Application (40 videos)** | $0.04 | Every run |
| **Total (first run)** | **$0.79** | - |
| **Total (subsequent runs)** | **$0.04** | - |

**For 10 Hashtags:**
- First run: $7.90
- Subsequent runs: $0.40

---

#### Option 2: Full Analysis (Higher Quality)

**Configuration:**
- Analyze: All 300 videos in top 3 buckets
- Model: Claude 3.5 Sonnet

| Stage | Cost per Hashtag | Frequency |
|-------|------------------|-----------|
| **2.6: Discovery** | $0.75 | One-time |
| **2.7: Application (300 videos)** | $2.40 | Every run |
| **Total (first run)** | **$3.15** | - |
| **Total (subsequent runs)** | **$2.40** | - |

**For 10 Hashtags:**
- First run: $31.50
- Subsequent runs: $24.00

---

### Cost Optimization Strategies

#### 1. Selective Classification (87% Reduction)
```python
# Only classify representative samples
videos_to_classify = (
    manifest['videos_by_bucket'][bucket]['top_performers'][:20] +
    manifest['videos_by_bucket'][bucket]['bottom_performers'][:20]
)
# Cost: 40 videos instead of 300
```

#### 2. Use Haiku for Classification (90% Reduction)
```python
# Discovery: Sonnet (needs reasoning)
# Application: Haiku (simple classification)
model = "claude-3-haiku-20240307"
```

#### 3. Caching (Avoid Re-Work)
```python
# Check if video already classified
cache_path = f"content_analysis/{hashtag}/{bucket}/{video_id}_content.json"
if os.path.exists(cache_path):
    return load_json(cache_path)  # No API call
```

---

### Recommended Configuration (MVP)

```python
# Stage 2.7 configuration
CONFIG = {
    "videos_per_bucket": 40,          # 20 top + 20 bottom
    "model": "claude-3-haiku",        # Fast, cheap
    "batch_size": 1,                  # Per-video classification
    "enable_caching": True            # Avoid re-classification
}

# Total cost: ~$0.04 per hashtag per run (~$0.40 for 10 hashtags)
```

**Why This Works:**
- 40 videos provide sufficient pattern representation
- Haiku is accurate for structured classification
- Caching makes re-runs nearly free
- Cost is negligible vs client value

---

### Implementation Scripts

#### Stage 2.6: Discovery Script

**Location:** `scripts/content_analysis_discovery.py`

```python
import anthropic
import json
import os

def discover_taxonomy(hashtag, manifest_path):
    """
    Discover content patterns from sample transcripts
    One-time operation per hashtag
    """
    # 1. Check if taxonomy already exists
    taxonomy_path = f"ContentAnalysis/taxonomies/{hashtag}_taxonomy.json"
    if os.path.exists(taxonomy_path):
        print(f"✅ Taxonomy already exists: {taxonomy_path}")
        return

    # 2. Load manifest
    manifest = load_json(manifest_path)

    # 3. Sample transcripts (20 per bucket for diversity)
    transcripts = []
    for bucket in manifest['selected_buckets']:
        video_ids = manifest['videos_by_bucket'][bucket]['top_performers'][:20]
        for video_id in video_ids:
            transcript = load_json(f"speech_transcriptions/{video_id}_whisper.json")
            transcripts.append(transcript['text'])

    # 4. LLM pattern discovery
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""
    Analyze {len(transcripts)} TikTok transcripts from #{hashtag}.

    Identify natural patterns in:
    1. Content Categories
    2. Hook Strategies
    3. Audience Pain Points
    4. Engagement Drivers
    5. Trending Keywords

    Return JSON with discovered patterns (name, frequency, examples).
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_taxonomy = json.loads(response.content[0].text)

    # 5. Save raw discovery
    save_json(f"ContentAnalysis/raw_discoveries/{hashtag}_raw.json", raw_taxonomy)

    print(f"✅ Discovery complete: ContentAnalysis/raw_discoveries/{hashtag}_raw.json")
    print("📝 Next: Manually curate and save to taxonomies/{hashtag}_taxonomy.json")

# CLI Usage:
# python scripts/content_analysis_discovery.py \
#   --hashtag nutrition \
#   --manifest ml_training_data/nutrition/selection_manifest.json
```

---

#### Stage 2.7: Application Script

**Location:** `scripts/content_analysis_application.py`

```python
import anthropic
import json
import os
from glob import glob

def classify_video(video_id, transcript, caption, hashtags, taxonomy, client):
    """
    Classify single video using LLM + taxonomy
    """
    prompt = f"""
    Classify this TikTok video using the predefined taxonomy.

    TAXONOMY:
    Content Categories: {json.dumps(taxonomy['content_categories'])}
    Hook Strategies: {json.dumps(taxonomy['hook_strategies'])}
    Pain Points: {json.dumps(taxonomy['audience_pain_points'])}

    VIDEO DATA:
    Transcript: {transcript}
    Caption: {caption}
    Hashtags: {json.dumps(hashtags)}

    Return JSON:
    {{
      "video_id": "{video_id}",
      "content_category": "...",
      "hook_strategy": "...",
      "audience_pain_points": [],
      "trending_keywords": [],
      "confidence": "high/medium/low"
    }}
    """

    response = client.messages.create(
        model="claude-3-haiku-20240307",  # Cheap, fast
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)

def apply_taxonomy(hashtag, manifest_path):
    """
    Apply saved taxonomy to classify all videos in top 3 buckets
    """
    # 1. Check taxonomy exists
    taxonomy_path = f"ContentAnalysis/taxonomies/{hashtag}_taxonomy.json"
    if not os.path.exists(taxonomy_path):
        print(f"⚠️  No taxonomy found. Run discovery first.")
        return

    taxonomy = load_json(taxonomy_path)
    manifest = load_json(manifest_path)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # 2. Classify videos
    for bucket in manifest['selected_buckets']:
        # Select 20 top + 20 bottom (selective analysis)
        videos = (
            manifest['videos_by_bucket'][bucket]['top_performers'][:20] +
            manifest['videos_by_bucket'][bucket]['bottom_performers'][:20]
        )

        for video_id in videos:
            # Check cache
            output_path = f"ContentAnalysis/content_analysis/{hashtag}/{bucket}/{video_id}_content.json"
            if os.path.exists(output_path):
                print(f"⏭️  Skipping {video_id} (cached)")
                continue

            # Load data
            transcript = load_json(f"speech_transcriptions/{video_id}_whisper.json")
            insights = load_json(f"insights/{video_id}_temporal_windows_updated.json")

            # Classify
            classification = classify_video(
                video_id=video_id,
                transcript=transcript['text'],
                caption=insights['metadata']['description'],
                hashtags=insights['metadata']['hashtag_analysis'],
                taxonomy=taxonomy,
                client=client
            )

            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_json(output_path, classification)
            print(f"✅ Classified: {video_id}")

# CLI Usage:
# python scripts/content_analysis_application.py \
#   --hashtag nutrition \
#   --manifest ml_training_data/nutrition/selection_manifest.json
```

---

### Graceful Degradation

**Content Analysis is Optional:**

```python
# In ml_training_pipeline.py (orchestrator)

# Stage 2.7: Content Analysis (optional)
taxonomy_exists = os.path.exists(f"ContentAnalysis/taxonomies/{hashtag}_taxonomy.json")

if taxonomy_exists:
    logger.info("Running content analysis...")
    run_content_analysis_application(hashtag, manifest_path)
else:
    logger.warning(f"No taxonomy for {hashtag}. Skipping content analysis.")
    logger.info("ML training will proceed without content insights.")

# Continue to Stage 3 (ML training proceeds regardless)
```

**Why?** ML training doesn't depend on content analysis. It's an enhancement, not a blocker.

---

## Technical Considerations

### Data Privacy
- Anonymize personal information in transcripts
- Focus on content patterns, not individual identification
- Respect creator privacy and platform terms

### Scalability
- Design for batch processing of large transcript collections
- LLM API rate limits: Handle with retries and exponential backoff
- Caching prevents redundant classifications

### Integration
- Independent processing pipeline (no ML feature constraints)
- JSON output format for easy database storage
- Parallel to ML pipeline (doesn't block existing flow)

---

**Next Steps**: Implement Stage 2.6 discovery script for initial hashtag taxonomy generation.