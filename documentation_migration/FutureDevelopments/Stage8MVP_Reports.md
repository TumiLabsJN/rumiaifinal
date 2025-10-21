# Stage 8 MVP: Report Template Structures

**Purpose**: Centralized template structure definitions for all Stage 8 PDF reports

**Parent Document**: Stage8MVP.md

**Status**: 2 of 4 templates complete (Hashtag → Client, Hashtag → Creator)

---

## Template Structure Overview

| # | Report Type | Audience | Status | Source |
|---|-------------|----------|--------|--------|
| 1 | Hashtag → Client | Tumi Labs Clients | ✅ **COMPLETE** | MLCreativeReports.md |
| 2 | Hashtag → Creator | Content Creators | ✅ **COMPLETE** | Stage8Planning.md section 1.1 |
| 3 | Handle/Single Competitor → Client | Tumi Labs Clients | ⏸️ **TODO** | To be designed |
| 4 | Handle/Multiple Competitor → Client | Tumi Labs Clients | ⏸️ **TODO** | To be designed |

---

## 1. Hashtag → Client (Executive Report)

**Audience**: Tumi Labs clients (business owners)

**Purpose**: Prove ML sophistication, reduce anxiety, provide creator sourcing strategy

**Deliverable**: 1 PDF per hashtag analysis

**Format**: 3-page PDF (desktop-first, mobile-tested)

**Reading Time**: 5-7 minutes (scannable in 2 minutes)

---

### Input Data Sources

- Stage 1: `winner_analysis.json`, `cluster_analytics.json`
- Stage 6: `rf_video_analysis.json`, `kmeans_analysis.json`
- Stage 7: `winning_formulas.json` (all 3 buckets)

---

### Page 1: Scale of Analysis

**Purpose**: Show the business owner how comprehensive the analysis is

**Header Section**:
```
#nutrition Hashtag Analysis
Analysis Period: Past 2-3 months
Videos Analyzed: 480
Analysis Mode: Top performers (engagement-based)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Hashtag | Config | CLI parameter `--hashtag` | String | "#nutrition" |
| Videos Analyzed | Stage 1 | `total_videos_analyzed` in winner_analysis.json | Integer | 480 |
| Analysis Mode | Config | Mapped from CLI parameter `--mode`:<br>• `top` → "Top performers (engagement-based)"<br>• `contrastive` → "Contrastive analysis (top vs bottom performers)" | String | "Top performers (engagement-based)" |

**Decision**: ✅ Always display "Past 2-3 months" regardless of actual `--date-filter` parameter for marketing consistency and perceived recency.

---

**What We Analyzed**:
```
Total Video Duration: 6.2 hours of content
Duration Range: 0-120 seconds (8 distinct buckets)
Content Elements Tracked: 60+ features per video

Analysis Method:
Multi-dimensional machine learning and AI content analysis:

• Visual & Behavioral Pattern Recognition - Advanced ML analyzed 60+ features
  per video (eye contact, pacing, energy levels, scene transitions, gesture
  frequency) to identify what separates top performers from bottom performers

• Content & Messaging Intelligence - AI-powered analysis of video transcripts
  and captions identified trending hook strategies, audience pain points,
  keywords, and engagement tactics unique to #nutrition content

• Formula Discovery - K-Means clustering revealed 3-5 distinct creative strategies
  per video length, validated by Random Forest classification models

Result: 9 proven formulas combining both "how to present" (visuals, pacing)
and "what to say" (hooks, messaging) for complete creative guidance.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Total Video Duration | Stage 1 | Calculated: sum of all video durations from metadata | String | "6.2 hours of content" |
| Hashtag (in Content & Messaging text) | Config | CLI parameter `--hashtag` | String | "#nutrition" |

**Decision**: ✅ Use integrated description showing both quantitative ML (Random Forest, K-Means) and qualitative content analysis (AI transcript analysis) without separating into explicit dual tracks.

---

### Page 2: Hashtag Intelligence Dashboard

**Purpose**: Show market landscape and what type of content creators to focus on

---

#### Section 1: Duration Distribution (What's Being Posted)

```
[Horizontal bar chart showing % of videos per bucket]

0-3s:   ████ 8%
3-9s:   ████████ 12%
9-13s:  ██████████ 15%
13-18s: ████████████████ 22%  ← HIGH VOLUME
18-33s: ████████████████████ 28%  ← HIGHEST VOLUME
33-60s: ██████████ 12%
60-90s: ██ 2%
90-120s: █ 1%

Key Insight: 65% of #nutrition content is 13-33s
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| % per bucket (all 8 rows) | Stage 1 | Calculated from `bucket_distribution` in winner_analysis.json | Integer (%) | 8, 12, 15, 22, 28, 12, 2, 1 |
| Key Insight % | Stage 1 | Calculated: sum of top buckets (13-18s + 18-33s) | Integer (%) | 65 |
| Hashtag (in Key Insight) | Config | CLI parameter `--hashtag` | String | "#nutrition" |

---

#### Section 2: Performance by Duration (What Performs Best)

```
0-3s:    125K avg views  ⭐⭐
3-9s:    210K avg views  ⭐⭐
9-13s:   380K avg views  ⭐⭐⭐
13-18s:  520K avg views  ⭐⭐⭐⭐
18-33s:  490K avg views  ⭐⭐⭐⭐
33-60s:  310K avg views  ⭐⭐⭐
60-90s:  180K avg views  ⭐⭐
90-120s: 95K avg views   ⭐

Sweet Spot: 13-33s (highest views + sufficient volume)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Avg views per bucket (all 8 rows) | Stage 1 | `avg_views` per bucket in winner_analysis.json | Integer (formatted with K/M) | 125K, 210K, 380K, 520K, etc. |
| Star ratings (all 8 rows) | Calculated | Based on view performance tier (highest = 4 stars, lowest = 1 star) | String (emoji) | ⭐⭐⭐⭐ |
| Sweet Spot range | Stage 1 | Calculated: buckets with highest avg views + sufficient volume | String | "13-33s" |

**Decision**: ✅ Show raw average view counts (not engagement % or normalized scores) for transparency and concreteness.

---

#### Section 3: Creator Profile Priorities (Where to Focus Hiring)

```
TIER 1 (Immediate Sourcing):
• 13-18s Creators (highest performance: 520K avg views)
• 18-33s Creators (strong performance + volume: 490K avg views)
• 33-60s Creators (proven success: 310K avg views)

Note: These are the 3 winning buckets where top performers cluster most.
Your creative reports focus exclusively on these durations.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Tier 1 bucket ranges (3 buckets) | Stage 1 | Top 3 buckets from `winning_buckets` in winner_analysis.json | String (array) | "13-18s", "18-33s", "33-60s" |
| Avg views per winning bucket | Stage 1 | `avg_views` for each winning bucket | Integer (formatted with K/M) | 520K, 490K, 310K |
| Performance labels | Calculated | Based on bucket ranking (highest, strong, proven) | String | "highest performance", "strong performance + volume", "proven success" |

**Decision**: ✅ Keep only Creator Profile Priorities section. Removed redundant sections:
- Content Saturation (not actionable, redundant)
- Trend Direction (too risky to fabricate M-o-M trends)
- Creator Recommendations with specific hiring quantities (too prescriptive)

---

### Page 3: Your Creative Reports

**Purpose**: Show what reports were delivered

**Report Distribution**:

Your content creators will receive 9 creative strategy reports tailored to the #nutrition hashtag:

**Duration Bucket 13-18s:**
  • Formula 1: The Question Hook Formula
  • Formula 2: The Fast-Paced Product Demo
  • Formula 3: The Myth-Busting Reveal

**Duration Bucket 18-33s:**
  • Formula 4: The Transformation Story
  • Formula 5: The Ingredient Deep-Dive
  • Formula 6: The Side-by-Side Comparison

**Duration Bucket 33-60s:**
  • Formula 7: The Step-by-Step Tutorial
  • Formula 8: The Expert Interview Format
  • Formula 9: The Before-After Journey

Each 2-page report includes:
  • Proof with numbers (engagement differences)
  • Second-by-second execution guide
  • Pre-post checklist

**Would you like to review a sample report?** Contact us at [email]

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Hashtag (in intro text) | Config | CLI parameter `--hashtag` | String | "#nutrition" |
| Duration Bucket ranges (3 buckets) | Stage 1 | Top 3 buckets from `winning_buckets` in winner_analysis.json | String (array) | "13-18s", "18-33s", "33-60s" |
| Formula names (9 formulas) | Stage 7 | `pattern_name` from winning_formulas.json (3 per bucket) | String (array) | "The Question Hook Formula", etc. |

**Decision**: ✅ Minimal Page 3 with report distribution list and sample report offer. "How to Use These Reports", "What Makes These Reports Effective", and "Next Steps" sections removed (onboarding material, not recurring report content).

---

### Summary of Resolved Design Decisions

| Issue | Decision | Rationale |
|-------|----------|-----------|
| **Analysis Period** | Always show "Past 2-3 months" | Marketing consistency, perceived recency |
| **ML Method Description** | Integrated dual-track (ML + Content Analysis) | Showcases sophistication, balances completeness and brevity |
| **Engagement Metrics** | Raw average view counts | Honest, concrete, defensible |
| **Content Saturation** | Remove section entirely | Not actionable, redundant with Creator Priorities |
| **Trend Direction** | Remove section entirely | Too risky to fabricate M-o-M trends |
| **Creator Recommendations** | Keep Section 3 only, remove Section 6 | Eliminates redundancy, avoids prescriptiveness |
| **Page 3 Scope** | Minimal - report list + sample offer only | Non-repetitive, executive-focused, scalable |

---

## 2. Hashtag → Creator (Content Creator Report)

**Audience**: Content creators (affiliates)

**Purpose**: Deliver actionable creative formulas with proof and execution steps

**Deliverable**: 9 PDFs per hashtag (3 buckets × 3 formulas each)

**Format**: 2-page PDF (**MOBILE-OPTIMIZED** - minimum 12pt body, 16pt+ headings, portrait layout)

**Reading Time**: 2-3 minutes

---

### Input Data Sources

- Stage 7: `winning_formulas.json` (3 creative reports per bucket)

---

### Report Characteristics

- **Catchy**: Grab attention with proof and numbers immediately
- **Evidence-based**: Show performance data upfront (not buried at end)
- **Actionable**: Dead simple to implement (copy-paste execution guide)
- **Mobile-optimized**: Designed for phone/tablet viewing
- **Scannable**: Find what you need in seconds
- **Consistent**: All 9 reports follow same 2-page structure

---

### Page 1: "Why This Works" (Hook with Proof + Pattern)

---

#### Header Section

```
Pattern Name: "The Question Hook Formula"
Duration: 18-33s | Hashtag: #nutrition | Confidence: 87%
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Pattern Name | Stage 7 | `pattern_name` in winning_formulas.json | String | "The Question Hook Formula" |
| Duration | Stage 7 | `bucket_range` in winning_formulas.json | String | "18-33s" |
| Hashtag | Config | CLI parameter `--hashtag` | String | "#nutrition" |
| Confidence | Stage 7 | `confidence_score` in winning_formulas.json | Integer (0-100%) | 87 |

---

#### The Proof (Numbers First)

```
📊 Videos using this pattern: 8.4% avg engagement
   Videos NOT using this: 3.1% avg engagement
   → 2.7x MORE ENGAGEMENT

[QR CODE]
Scan to watch: Top Performer Using This Pattern (520K views)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Pattern engagement % | Calculated | Industry Benchmark Mapping (see method below) | Float (%) | 8.4 |
| Non-pattern engagement % | Calculated | Industry Benchmark Mapping (see method below) | Float (%) | 3.1 |
| Engagement multiplier | Calculated | Pattern % ÷ Non-pattern % | Float (ratio) | 2.7 |
| QR Code (Top Performer) | Stage 2 + 7 | Video URL from top cluster in Stage 2 metadata, mapped to this formula | QR Code Image | Links to TikTok video |
| Example video views | Stage 2 | `view_count` from top performer video metadata | Integer (formatted with K/M) | 520K |

**Engagement Rate Calculation Method** (Industry Benchmark Mapping):

**Step 1: Identify Performance Tiers from View Data**
- Analyze avg view counts per bucket from Stage 1 `winner_analysis.json`
- Rank buckets by performance: Top (highest views), Middle, Bottom (lowest views)

**Step 2: Map to Industry Engagement Benchmarks**
- **Top performers** (highest view bucket): 6-9% engagement rate
- **Middle performers**: 4-6% engagement rate
- **Bottom performers**: 2-4% engagement rate

**Step 3: Apply to Pattern-Specific Groups**
- Videos using this pattern (from top bucket): Assign high-end benchmark (e.g., 8.4%)
- Videos NOT using this pattern (from bottom/middle): Assign low-end benchmark (e.g., 3.1%)
- Calculate multiplier: 8.4% / 3.1% = 2.7x

**Rationale**: TikTok industry standard engagement rates are 3-9% for good content. We map our view performance tiers to these established benchmarks to provide realistic, defensible engagement estimates.

**Note**: These are estimated engagement rates based on view performance and industry benchmarks, not directly measured engagement data.

---

#### Contrastive Analysis (Do This vs Don't)

```
Top performers do THIS:
✅ Ask question in first 2s (avg 3.2 questions in hook)
✅ Show product by 5 seconds (immediate visual payoff)
✅ Use 5-7 text overlays (keep attention with text)

[QR CODE]
Scan to watch: Bottom Performer - Don't Do This (95K views)

Bottom performers do THIS:
❌ Generic opening/statement (0.8 questions avg)
❌ Product reveal after 10+ seconds (viewers already scrolled)
❌ No text overlays (viewers get bored/confused)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Top behaviors (3-5 items) | Stage 7 | `top_performer_behaviors` in winning_formulas.json | String (array) | ["Ask question in first 2s...", "Show product by 5 seconds...", etc.] |
| Bottom behaviors (3-5 items) | Stage 7 | `bottom_performer_behaviors` in winning_formulas.json | String (array) | ["Generic opening/statement...", "Product reveal after 10+ seconds...", etc.] |
| QR Code (Bottom Performer) | Stage 2 + 7 | Video URL from bottom cluster in Stage 2 metadata | QR Code Image | Links to TikTok video |
| Bottom video views | Stage 2 | `view_count` from bottom performer video metadata | Integer (formatted with K/M) | 95K |

---

#### Pattern Summary (3-Step Overview)

```
1️⃣ Hook (0-3s): Ask compelling question
2️⃣ Show (3-15s): Reveal product + explain benefit
3️⃣ Prove (15-33s): Demonstrate result + CTA

[VISUAL: Simple timeline graphic with 3 boxes]
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Step 1 (Hook) description | Stage 7 | `hook_strategy` in winning_formulas.json | String | "Ask compelling question" |
| Step 2 (Middle) description | Stage 7 | `middle_strategy` in winning_formulas.json | String | "Reveal product + explain benefit" |
| Step 3 (Closing) description | Stage 7 | `closing_strategy` in winning_formulas.json | String | "Demonstrate result + CTA" |
| Timing ranges (all 3 steps) | Stage 7 | `temporal_window_timings` in winning_formulas.json | String (array) | ["0-3s", "3-15s", "15-33s"] |

---

### Page 2: "How to Execute" (Copy-Paste Implementation)

---

#### Second-by-Second Timeline (Literal Script)

```
⏱️ 0-2 seconds: THE QUESTION
  Say: "Did you know [surprising fact about topic]?"
  Visual: Your face, direct to camera
  Text overlay: The question (animated in)

⏱️ 3-5 seconds: SHOW THE THING
  Visual: Close-up of product/ingredient
  Text overlay: Product name
  Say: "This is [product name]"

⏱️ 6-15 seconds: EXPLAIN WHY IT MATTERS
  Say: Main benefit (problem it solves)
  Visual: Product in context (using it, showing it)
  Text overlay: 2-3 key benefit points

⏱️ 16-30 seconds: PROOF/DEMONSTRATION
  Visual: Before/after OR step-by-step demo
  Text overlay: Results/transformation
  Say: Why it works scientifically/logically

⏱️ 31-33 seconds: CALL TO ACTION
  Say: "Save this!" or "Try it yourself!"
  Visual: End card or gesture pointing to save button
  Text overlay: Simple CTA
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Entire timeline (5+ segments) | Stage 7 | `second_by_second_script` in winning_formulas.json | Object (array of segments) | [{timing: "0-2s", label: "THE QUESTION", say: "...", visual: "...", text_overlay: "..."},...] |

**Note**: The timeline structure (timing ranges, segment labels, Say/Visual/Text overlay instructions) all come from Stage 7 LLM analysis. The example shows a typical 5-segment structure, but actual segments vary per pattern.

---

#### Pre-Post Checklist

```
✓ CHECKLIST BEFORE POSTING
□ Question in first 2 seconds?
□ Product visible by 5 seconds?
□ 5-7 text overlays placed?
□ 2-3 scene changes in middle?
□ Clear CTA at end?
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Checklist items (5-7 items) | Stage 7 | `verification_checklist` in winning_formulas.json | String (array) | ["Question in first 2 seconds?", "Product visible by 5 seconds?", etc.] |

---

### Mobile Optimization Requirements

**CRITICAL for Template A (Content Creator reports):**

- **Font sizes**: Minimum 12pt body text, 16pt+ headings
- **Layout**: Portrait orientation (8.5" × 11" or smaller)
- **Single-column**: No multi-column layouts (hard to read on phones)
- **Touch-friendly spacing**: Minimum 0.25" margins, adequate line spacing
- **Visual hierarchy**: Clear section breaks, generous whitespace
- **Testing**: MUST validate on actual iPhone + Android devices before finalizing

---

### QR Code Implementation (Decision: Issue 1 Resolved)

**Decision**: Each creator report includes **2 QR codes** linking to real TikTok video examples

**QR Code Placement**:
1. **QR Code 1** (Top Performer): After "The Proof" section
   - Label: "Scan to watch: Top Performer Using This Pattern (520K views)"
   - Links to: Top cluster video from Stage 2 analysis
2. **QR Code 2** (Bottom Performer): In "Contrastive Analysis" section
   - Label: "Scan to watch: Bottom Performer - Don't Do This (95K views)"
   - Links to: Bottom cluster video from Stage 2 analysis

**Video Selection Criteria**:
- **Priority**: Newest videos from analysis (reduces deletion risk)
- **Source**: Stage 2 Apify video URLs (`video_url` field from metadata)
- **Cluster mapping**: Top cluster videos for QR 1, bottom cluster videos for QR 2
- **Stability preference**: If available, prefer videos from accounts with 100K+ followers

**Technical Requirements**:
- QR code size: ~1 inch × 1 inch (easily scannable on mobile)
- QR code generation library: Python `qrcode` package
- Error correction level: Medium (allows ~15% damage tolerance)
- File size impact: ~5KB per QR code (~10KB total for 2 codes)

**Graceful Degradation**:
- Text descriptions remain primary content (QR codes are supplementary)
- If TikTok video deleted, text still provides value
- QR codes can be refreshed in updated report versions

**MVP Impact**:
- **Task 2.6**: QR code generation (+1 day effort)
- **Task 5.8**: Map Stage 2 video URLs to Stage 7 formulas (+0.5 days effort)
- **Total**: +1.5 days to MVP timeline

---

## 3. Handle/Single Competitor → Client

**Status**: ⏸️ **TO BE DESIGNED**

**Audience**: Tumi Labs clients (business owners)

**Purpose**: Competitive intelligence - benchmark 1 competitor vs client baseline

**Deliverable**: 1 PDF comparing 1 competitor vs client

**Format**: TBD (3-4 pages?)

---

### Placeholder Content Ideas

**Potential Sections** (to be finalized):
- Competitor overview (posting frequency, top buckets, avg performance)
- Creative patterns competitor uses (top 3 formulas from Stage 7)
- Benchmarking vs client (performance gaps, opportunities)
- Hashtag strategy analysis (which hashtags competitor wins with)

---

### Input Data Sources (Confirmed)

- Competitor Stage 7: `winning_formulas.json`
- Competitor Stage 6: `rf_video_analysis.json`, `kmeans_analysis.json`
- Competitor Stage 1: `winner_analysis.json` (bucket distribution)
- Competitor metadata: Handle, posting frequency, top hashtags
- Client baseline: All of above for client (for benchmarking)

---

### Key Design Decisions Locked

- ✅ Audience: Tumi Labs clients (business owners)
- ✅ Requires client baseline for benchmarking
- ✅ Uses full Stages 1-7 pipeline (same as hashtag analysis)
- ⏸️ Page count: TBD (3-4 pages)
- ⏸️ Detailed structure: TBD (needs design session)

---

## 4. Handle/Multiple Competitor → Client

**Status**: ⏸️ **TO BE DESIGNED**

**Audience**: Tumi Labs clients (business owners)

**Purpose**: Multi-competitor comparison - side-by-side benchmarking

**Deliverable**: 1 PDF comparing multiple competitors vs client (side-by-side)

**Format**: TBD (4-5 pages?)

---

### Placeholder Content Ideas

**Potential Sections** (to be finalized):
- Performance leaderboard (rank all competitors + client)
- Bucket strategy comparison (which durations each competitor focuses on)
- Creative formula comparison (best-in-class formulas across competitors)
- Hashtag strategy comparison (diversification analysis)
- Key takeaways (what client should adopt from competitors)

---

### Input Data Sources (Confirmed)

- Multiple competitor Stage 7 outputs
- Multiple competitor Stage 6 outputs
- Multiple competitor Stage 1 outputs
- Client baseline (for benchmarking)

---

### Key Design Decisions Locked

- ✅ Audience: Tumi Labs clients (business owners)
- ✅ Requires client baseline for benchmarking
- ✅ Uses full Stages 1-7 pipeline per competitor
- ⏸️ Page count: TBD (4-5 pages)
- ⏸️ Detailed structure: TBD (needs design session)

---

## Next Steps

1. ✅ **Task 0.1**: Hashtag → Client structure (COMPLETE)
2. ✅ **Task 0.2**: Hashtag → Creator structure (COMPLETE)
3. ⏸️ **Task 0.3**: Design Handle/Single Competitor → Client structure (0.75 days)
4. ⏸️ **Task 0.4**: Design Handle/Multiple Competitor → Client structure (0.5 days)

**Critical Path Blocker**: Tasks 0.3 and 0.4 must be completed before designer can start building PDF templates.

---

## Resolved Issues

### ✅ Issue 1: Dynamic Fields Documentation (RESOLVED)

**Problem**: Template structures showed examples but didn't clearly document which fields are dynamic vs static, nor how to extract dynamic data from JSON sources.

**Solution Implemented**: Alternative 2 - Clean Template + Separate Data Mapping Tables

**Approach**:
- Templates show clean, realistic examples (designer-friendly)
- Each major section followed by "Dynamic Fields" table documenting:
  - Template Field name
  - Source (Stage 1/7, Config, Calculated)
  - JSON Field/Calculation method
  - Data Type
  - Example value
- Tables appear immediately after each section (contextual, scannable)
- Only dynamic fields documented (focused on implementation needs)

**Status**: ✅ **COMPLETE** - Both Hashtag → Client and Hashtag → Creator templates now have comprehensive data mapping tables for all sections.

---

**Status**: ✅ **2 of 4 templates complete** - Ready to design competitor report structures (Tasks 0.3, 0.4)

---

## Content Analysis Data Capabilities (Stage 2.6 & 2.7)

**Last Updated**: 2025-10-21
**Source Documents**: ContentAnalysisCHILD.md, ContentAnalysisCHILDTI.md
**Purpose**: Document what qualitative and quantitative data is available from Content Analysis for report generation

---

### Overview: What Content Analysis Provides

Content Analysis (Stages 2.6 & 2.7) extracts **video-level qualitative patterns** from TikTok content using LLM-powered classification:

**Stage 2.6 (Discovery)**:
- Samples 50 transcripts from top performers across 3 winning buckets
- Uses Claude 3.5 Sonnet to discover natural content patterns
- Outputs raw taxonomy with 6 pattern categories
- Requires manual human curation before classification

**Stage 2.7 (Classification)**:
- Classifies 120 videos (40 per bucket × 3 buckets) using curated taxonomy
- Uses Claude 3 Haiku for fast, low-cost classification
- Outputs 12 classification fields per video (refined schema)
- Provides both video content patterns AND caption/hashtag analysis

---

### Critical Limitation: No Temporal Breakdown

**IMPORTANT**: Content Analysis provides **VIDEO-LEVEL** classifications, NOT second-by-second breakdowns.

**What we have**:
- ✅ "This video uses 'problem_solution' hook strategy" (entire video)
- ✅ "This video includes 'personal_testimony' tactic" (present somewhere in video)
- ✅ "This video mentions 'gut_health' keyword" (mentioned at some point)

**What we DON'T have**:
- ❌ "The 'problem_solution' hook occurs at 0-3 seconds" (no timestamp data)
- ❌ "Personal testimony appears at 10-15 seconds" (no temporal alignment)
- ❌ "Gut health keyword is mentioned at second 8" (no time-based tracking)

**Data Source Evidence**:
- Input: Whisper transcript is **"complete transcript (can be empty)"** - no segment timestamps (ContentAnalysisCHILD.md Section 5.1.2, lines 796-803)
- Processing: LLM analyzes **full transcript + caption + hashtags** as single unit (ContentAnalysisCHILD.md Section 2.3.4)
- Output: Classification fields have **no timestamp metadata** (ContentAnalysisCHILD.md Section 5.2.2)

---

### Available Qualitative Data (12 Fields per Video)

**Source**: ContentAnalysisCHILDTI.md Section 3.3 (VideoClassificationSchema)

#### Core Content Classifications (6 fields)

| Field | Type | Description | Example Values | Notes |
|-------|------|-------------|----------------|-------|
| `content_category` | String (single) | Primary content type/format | "wellness_practice", "recipe_tutorial", "supplement_review" | From curated taxonomy, 1 per video |
| `hook_strategy` | String (single) | Opening pattern/technique | "problem_solution", "direct_statement", "question_hook" | From curated taxonomy, 1 per video |
| `pain_points` | Array of strings | Problems/struggles addressed | ["bloating", "low_energy", "menstrual_discomfort"] | 0-N selections from taxonomy |
| `keywords` | Array of strings | Topics/methods/solutions mentioned | ["protein", "gut_health", "holistic"] | 0-N selections from taxonomy |
| `engagement_drivers` | Array of strings | Shareability tactics used | ["before_after_reveal", "personal_testimony", "product_recommendation"] | 0-N selections from taxonomy |
| `content_tactics` | Array of strings | Presentation styles/formats | ["personal_story", "direct_to_camera", "vulnerability_shown"] | 0-N selections from taxonomy |

**Classification Logic**:
- **Categories 1-2** (content_category, hook_strategy): **Single selection required** - every video gets exactly one
- **Categories 3-6** (pain_points, keywords, engagement_drivers, content_tactics): **Multiple selection allowed** - can be empty arrays `[]` if none apply

**Grounding Rule** (from 2.7ClassificationCritique.md):
- Items selected only if **explicitly mentioned** (direct quote) OR **strongly implied** (clear contextual evidence)
- Empty arrays acceptable when patterns not present
- Quality over quantity - no forced selections

---

#### Caption & Hashtag Analysis (8 subfields)

**Source**: ContentAnalysisCHILDTI.md Section 3.3 (CaptionAnalysisSchema)

All fields nested under `caption_analysis` object:

| Field | Type | Description | Possible Values | Notes |
|-------|------|-------------|-----------------|-------|
| `hook_type` | String | How caption opens (first 5-10 words) | "statement", "question", "command", "teaser" | Simplified from 6 to 4 types in refined schema |
| `cta_type` | String | Call-to-action type | "link_in_bio", "save_post", "comment", "follow", "share", "tag_friend", "none" | Renamed from caption_cta_type |
| `brand_mention_present` | Boolean | Brand/product mentioned in caption | true, false | Objective detection |
| `influencer_tag_present` | Boolean | Another creator tagged in caption | true, false | Objective detection |
| `emoji_usage` | String | Emoji density | "none" (0), "some" (1-4), "many" (5+) | Simplified from 4 to 3 levels |
| `caption_length` | String | Caption length category | "short" (<100 chars), "long" (100+ chars) | Simplified from 3 to 2 levels |
| `hashtag_count` | Integer | Number of hashtags | 0-30 | Directly countable |
| `hashtag_placement` | String | Where hashtags appear | "end" (all at end), "mixed" (throughout), "none" | Observable pattern |

**Fallback Logic** (from 2.7ClassificationCritique.md Zone 3):
- **Hook Strategy** (required field): Primary = transcript opening (first 5-10 words spoken), Fallback = caption opening if transcript empty
- **Caption hook_type**: Always from caption text (not video speech)
- **Distinction**: `hook_strategy` (content pattern) vs `caption_analysis.hook_type` (caption structure)

---

#### Metadata Fields (3 fields)

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `taxonomy_version` | String | Links classification to taxonomy source | Always "stage2.6_output" |
| `confidence` | String | Classification confidence level | "high", "medium", "low" |
| `transcript_available` | Boolean | Whether transcript was used | true, false |
| `note` | String (nullable) | Explanation for edge cases | "Classified using caption and hashtags only" (if transcript empty), null (if high confidence) |

**Confidence Assessment Criteria** (from 2.7ClassificationCritique.md Zone 3):
- **high**: Clear taxonomy match + strong evidence from transcript/caption + all selections explicitly justified
- **medium**: Partial taxonomy match OR inference required OR evidence from transcript OR caption (not both aligning)
- **low**: Forced match (no perfect fit) OR limited evidence (empty transcript, minimal caption) OR weak inference

---

### Taxonomy Structure (6 Categories)

**Source**: ContentAnalysisCHILD.md Section 5.1.4 (Curated Taxonomy Schema)

The curated taxonomy defines all possible classification values:

#### Categories with Definitions (Semantic Categories 1-2)

**Format**: Array of objects with `name` and `definition` fields

```json
{
  "content_categories": [
    {"name": "recipe_tutorial", "definition": "Step-by-step cooking instructions"},
    {"name": "wellness_practice", "definition": "Holistic health routines or rituals"}
  ],
  "hook_strategies": [
    {"name": "problem_solution", "definition": "Starts with problem, promises solution"},
    {"name": "question_hook", "definition": "Opens with interrogative to create curiosity"}
  ]
}
```

**Validation Rules**:
- Names must be snake_case (lowercase, numbers, underscores only)
- Definitions minimum 10 characters
- No duplicates
- Range: 2-10 items per category (recommended)

#### Simple Lists (Categories 3-6)

**Format**: Array of strings (no definitions)

```json
{
  "audience_pain_points": ["bloating", "low_energy", "menstrual_discomfort"],
  "trending_keywords": ["protein", "gut_health", "fiber", "metabolism"],
  "engagement_drivers": ["before_after_reveal", "specific_metrics_mentioned", "personal_testimony"],
  "content_tactics": ["personal_story", "direct_to_camera", "voiceover", "text_overlay"]
}
```

**Validation Rules**:
- All items must be strings
- Minimum 2 characters per item
- No duplicates
- Range: 2-15 items per category (recommended)

---

### Data Flow: Discovery → Curation → Classification

**Stage 2.6 Discovery Process**:
1. Sample 50 transcripts (stratified: ~17 per bucket from top performers)
2. LLM (Claude 3.5 Sonnet) discovers patterns across 6 categories
3. Output: `{hashtag}_raw_discovery.json` with frequencies, examples, representative video IDs
4. **Manual Step**: Human curator reviews, removes rare patterns (<10% frequency), merges similar ones, fixes naming
5. Save curated taxonomy: `{hashtag}_taxonomy.json`

**Stage 2.7 Classification Process**:
1. Load curated taxonomy + selection manifest (120 video IDs to classify)
2. For each video:
   - Load transcript (Whisper), caption, hashtags (unified_analysis)
   - LLM (Claude 3 Haiku) classifies using taxonomy as fixed options
   - Validate output (12 required fields)
   - Save: `{bucket}/content_analysis/{video_id}_content.json`
3. Output: 120 classification files (40 per bucket × 3 buckets)

**Cost & Performance**:
- Discovery: ~$0.75 per hashtag (Sonnet), 52 seconds, one-time per hashtag
- Classification: ~$0.001 per video (Haiku), 2.5 seconds per video, 120 videos = ~$0.12 total
- Total first run: ~$0.87 + 15 min manual curation
- Subsequent runs (taxonomy reused): ~$0.12 only

---

### Integration with Temporal Windows (RumiAI Stage 2 ML)

**CRITICAL**: Content Analysis (qualitative) and Temporal Windows (quantitative) are **separate data sources** with different granularity.

#### Temporal Windows Data (From SystemArchitecturev2.md)

**What it provides**: 50+ quantitative metrics per temporal segment

**Temporal Structure**:
- **Hook (0-3s)**: word_count, person_count, energy_level, close_ratio, has_greeting, joy_ratio, eye_contact_rate
- **Middle segments** (varies by duration): 3-5 segments depending on video length, same 50+ features each
- **Closing (last 3s)**: has_speech_cta, energy_max, plus all standard features

**Data Type**: Quantitative metrics (integers, floats, booleans)
**Timing**: Precise timestamps (0-3s, 3-10s, etc.)

#### Content Analysis Data (From Stage 2.6/2.7)

**What it provides**: Qualitative content patterns

**Coverage**: Entire video (no segment breakdown)

**Data Type**: Categorical strings, arrays of strings
**Timing**: None (video-level only)

---

### Realistic Report Alternatives Given Data Constraints

Based on available data, here are the realistic options for creator reports:

#### Alternative 1: Pattern Blueprint (3-Phase Structure) ← RECOMMENDED

**Approach**: Combine video-level content patterns with temporal window boundaries

**Structure**:
- **Phase 1: HOOK (0-3s)** - Content pattern (e.g., "problem_solution") + Execution metrics (word_count, energy, close_ratio)
- **Phase 2: MIDDLE (3s to last 3s)** - Content elements to include (e.g., "personal_testimony", "gut_health") + Execution standards (scene changes, energy)
- **Phase 3: CLOSING (last 3s)** - CTA pattern (e.g., "link_in_bio") + Execution metrics (energy_max, has_speech_cta)

**Data Sources**:
- Pattern strategies: `content_category`, `hook_strategy`, `pain_points`, `keywords`, `engagement_drivers`, `content_tactics` from Content Analysis
- Execution metrics: Temporal window aggregates (avg word_count, avg energy, avg close_ratio) from RumiAI Stage 2
- Timing boundaries: Temporal window structure (0-3s, middle, last 3s)

**Pros**:
- Data-honest (uses video-level patterns correctly)
- Still actionable (3 clear phases)
- Combines qualitative + quantitative

**Cons**:
- Middle section is vague on exact timing ("include somewhere in middle")
- Not true "second-by-second" timeline

---

#### Alternative 2: Checklist-First (Pattern Execution Guide)

**Approach**: Focus on WHAT to include, not WHEN

**Structure**:
- **Content Requirements**: List of patterns to include (from Content Analysis)
- **Execution Standards**: Metrics for Hook (0-3s) and Closing (last 3s) from Temporal Windows
- **Verification Checklist**: Binary checks for pattern presence

**Data Sources**:
- Content requirements: All 6 Content Analysis fields
- Execution standards: Temporal window metrics for hook and closing only
- No middle timing guidance

**Pros**:
- Honest about data limitations
- Actionable checklist format
- Combines qualitative + quantitative

**Cons**:
- No timeline structure
- May feel too vague for creators wanting step-by-step guidance

---

#### Alternative 3: Hybrid Timeline with Acknowledged Gaps

**Approach**: Provide timing for what we know (hook, closing), acknowledge flexibility for middle

**Structure**:
- **0-3s: HOOK** - Pattern + metrics (precise)
- **3s to last 3s: BUILD & PROVE** - Content elements list with ⚠️ note: "Our data shows WHAT to include, not exact timing"
- **Last 3s: CLOSING** - CTA + metrics (precise)

**Data Sources**:
- Same as Alternative 1, but with explicit disclaimer

**Pros**:
- Transparent about limitations
- Precise where we have data
- Honest about flexibility

**Cons**:
- Warning note may reduce creator confidence
- Still vague on middle timing

---

### Recommendation for Stage 8 Implementation

**Use Alternative 1: Pattern Blueprint (3-Phase Structure)**

**Rationale**:
1. **Data integrity**: Uses Content Analysis (video-level) and Temporal Windows (segment-level) correctly without inventing precision we don't have
2. **Actionable**: Creators get clear structure (Hook → Middle → Close) with specific patterns and metrics
3. **Mobile-friendly**: Concise, scannable format fits 2-page PDF constraint
4. **Honest marketing**: Can call it "Pattern Execution Blueprint" instead of "second-by-second timeline"

**Impact on Template Fields (Section 2: Hashtag → Creator)**:

**CURRENT (in document)**:
- Line 418: `second_by_second_script` from Stage 7 (5+ segments with precise timing)

**UPDATED (recommended)**:
- Replace with: `pattern_blueprint` with 3 phases (Hook, Middle, Closing)
- Each phase combines:
  - Content pattern (from Content Analysis: `hook_strategy`, `keywords`, `engagement_drivers`, etc.)
  - Execution metrics (from Temporal Windows: `word_count`, `energy_level`, `close_ratio`, etc.)
  - Timing boundary (from Temporal Window structure: "0-3s", "3s to last 3s", "last 3s")

**Example JSON Structure for Stage 7 Output**:
```json
{
  "pattern_blueprint": {
    "hook": {
      "timing": "0-3s",
      "content_pattern": {
        "strategy": "problem_solution",
        "description": "State the problem clearly"
      },
      "execution_metrics": {
        "word_count_avg": 12.5,
        "energy_level_avg": 0.47,
        "close_ratio_avg": 0.82
      }
    },
    "middle": {
      "timing": "3s to last 3s",
      "content_elements": [
        {"type": "keyword", "value": "gut_health", "description": "Mention gut health benefits"},
        {"type": "tactic", "value": "personal_testimony", "description": "Share personal experience"},
        {"type": "driver", "value": "before_after_reveal", "description": "Show transformation"}
      ],
      "execution_standards": {
        "scene_changes_per_10s": 2.3,
        "energy_level_avg": 0.36,
        "element_count_avg": 27
      }
    },
    "closing": {
      "timing": "last 3s",
      "content_pattern": {
        "cta_type": "link_in_bio",
        "description": "Direct viewers to link"
      },
      "execution_metrics": {
        "energy_max_avg": 0.91,
        "has_speech_cta": true
      }
    }
  }
}
```

---

### Next Steps for Implementation

1. **Stage 7 LLM Prompt Update**: Modify report generation prompt to output `pattern_blueprint` structure (3 phases) instead of `second_by_second_script` (5+ segments)
2. **Template Field Update**: Update Section 2 (Hashtag → Creator, Page 2) to reflect 3-phase structure
3. **Data Mapping Table**: Add new dynamic fields table for pattern_blueprint extraction
4. **Issue 3 Resolution**: Mark CreatorContentCritique.md Issue 3 as RESOLVED with Alternative 1 decision

---

**End of Content Analysis Data Capabilities Documentation**
