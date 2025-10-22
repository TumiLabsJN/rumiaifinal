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

### Summary of Design Decisions (Issues 1-6 Resolved)

| Issue | Decision | Impact on Template | Location |
|-------|----------|-------------------|----------|
| **Issue 1: Visual Examples** | 2 QR Codes (top + bottom performer) | QR codes on Page 1 after "The Proof" and in "Contrastive Analysis" | Line 598+ (QR Code Implementation) |
| **Issue 2: Confidence Score** | Remove entirely | Header shows 3 fields (Pattern, Duration, Hashtag) - no confidence | Line 296+ (Header Section) |
| **Issue 3: Timeline Detail** | 3-Phase Pattern Blueprint | Page 2 structure: Hook (0-3s), Middle (flexible), Closing (last 3s) | Line 401+ (Structure Decision) |
| **Issue 4: Timeline Structure** | Fixed 3-phase for ALL buckets | Same structure for 13-18s, 18-33s, 33-60s videos | Line 407+ (Structure Applied) |
| **Issue 5: Checklist Length** | 5-7 items (pattern-specific) | Pre-Post Checklist with 6 items grouped by phase | Line 545+ (Pre-Post Checklist) |
| **Issue 6: Pattern Naming** | Data-driven from Content Analysis | Pattern names generated from cluster's dominant Content Analysis fields | Line 301+ (Pattern Naming Strategy) |

**Result**: Consistent, data-honest, mobile-optimized 2-page PDF template applicable to all 9 creative formulas.

---

### Page 1: "Why This Works" (Hook with Proof + Pattern)

---

#### Header Section

```
Pattern Name: "The Question Hook Formula"
Duration: 18-33s | Hashtag: #nutrition
```

**Design Decision**: Confidence score removed (Issue 2 resolution)
- **Rationale**: Stage 7 already filters low-confidence patterns (<70%), so all patterns in reports are validated
- **Simplified header**: 3 elements instead of 4 (cleaner, more scannable on mobile)
- **Implicit trust model**: "If it's in the report, it's proven" - no need for creators to question reliability

**Design Decision**: Pattern naming strategy (Issue 6 resolution)
- **Approach**: Data-driven names generated from Content Analysis labels
- **Format**: "The [Primary Distinctive Element] [Secondary Element OR 'Formula']"
- **Generation**: Stage 7 LLM analyzes cluster's dominant Content Analysis fields and constructs readable name
- **Rationale**:
  - Data integrity: Names directly reflect what Content Analysis found in cluster
  - Consistency: Repeatable across different hashtags
  - No hallucination: Can't invent promises not backed by data
  - Professional: Sounds proven, not clickbait

**Naming Logic**:
- Identify most common `hook_strategy`, `engagement_drivers`, or `content_category` in cluster
- Translate technical labels to readable format:
  - `hook_strategy="question_hook"` → "The Question Hook Formula"
  - `engagement_driver="before_after_reveal"` → "The Before-After Transformation"
  - `hook_strategy="problem_solution"` + `content_category="recipe_tutorial"` → "The Problem-Solution Recipe"
- Ensure uniqueness within bucket (no duplicate names)
- Keep 3-6 words (mobile-friendly)

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Pattern Name | Stage 7 | `pattern_name` in winning_formulas.json (LLM-generated from Content Analysis) | String | "The Question Hook Formula" |
| Duration | Stage 7 | `bucket_range` in winning_formulas.json | String | "18-33s" |
| Hashtag | Config | CLI parameter `--hashtag` | String | "#nutrition" |

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

**Structure Decision**: 3-Phase Pattern Blueprint (applies to ALL duration buckets)

**Why This Structure**:
- **Data Limitation**: Content Analysis provides VIDEO-LEVEL qualitative data (no second-by-second timestamps for middle content)
- **Temporal Alignment**: RumiAI's data structure is 3 segments (0-3s hook, middle segments, last 3s closing)
- **Consistency**: All 9 reports use identical structure - creators learn once, apply everywhere
- **Honest Guidance**: Precise timing where we have data (hook, closing), flexible where we don't (middle)

**Structure Applied to All Buckets**:
- 13-18s bucket: Hook (0-3s), Middle (3-15s), Closing (last 3s)
- 18-33s bucket: Hook (0-3s), Middle (3-30s), Closing (last 3s)
- 33-60s bucket: Hook (0-3s), Middle (3-57s), Closing (last 3s)

**Implementation**: 1 template design, 1 Stage 7 LLM prompt (reused for all buckets)

---

#### Pattern Execution Blueprint

**CONTENT FORMAT**
```
Format: Recipe Tutorial
Step-by-step instructional content showing "how to make" or "how to prepare"
```

**How Content Category is Assigned**:
- Stage 7 selects the **most common** `content_category` from videos in each cluster
- No rotation forcing - if all 3 clusters in a bucket are "Recipe Tutorial", then all 3 formulas show "Recipe Tutorial"
- Total distribution across 9 reports may vary (e.g., 5 recipe + 2 wellness + 2 supplement)
- This is data-honest: reflects what actually separates top performers in each cluster

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Format name | Stage 7 | `content_category` in winning_formulas.json (most common in cluster) | String | "Recipe Tutorial" |
| Format description | Stage 7 | `content_category_description` (LLM-generated) | String | "Step-by-step instructional content..." |

---

**⏱️ PHASE 1: HOOK (0-3 seconds)**

```
Strategy: Ask question about audience pain point
Example: "Did you know bloating might be a gut health issue?"

Execution:
• 10-15 words maximum
• Face visible, direct to camera (close-up)
• High energy from start (enthusiastic tone)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Strategy name | Stage 7 | `hook_strategy` from Content Analysis | String | "problem_solution" |
| Strategy description | Stage 7 | LLM interpretation of hook_strategy | String | "Ask question about audience pain point" |
| Example phrase | Stage 7 | LLM-generated example using `pain_points` + `keywords` | String | "Did you know bloating..." |
| Word count | Temporal Windows | Avg `word_count` from 0-3s window (top performers) | Integer | 10-15 |
| Visual direction | Temporal Windows | Based on `close_ratio` from 0-3s window | String | "Face visible, direct to camera" |
| Energy description | Temporal Windows | Based on `energy_level` from 0-3s window | String | "High energy" |

---

**⏱️ PHASE 2: BUILD & PROVE (3s to last 3s - flexible timing)**

```
💡 Include all these elements in whatever order flows naturally:

Content Checklist:
□ Mention "gut health" and "protein" keywords
□ Share personal testimony ("This is what worked for me...")
□ Show before/after comparison
□ Demonstrate product use

Execution Standards:
• Fast pacing: 2-3 scene changes per 10 seconds
• Use 5-7 text overlays throughout (highlight key points)
• Maintain moderate energy (don't drop off mid-video)
• Product clearly visible in shots
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Keywords (checklist items) | Stage 7 | `keywords` array from Content Analysis | Array of strings | ["gut health", "protein"] |
| Engagement drivers (checklist items) | Stage 7 | `engagement_drivers` array from Content Analysis | Array of strings | ["personal_testimony", "before_after_reveal"] |
| Pain points (optional checklist) | Stage 7 | `pain_points` array from Content Analysis | Array of strings | ["bloating", "low energy"] |
| Content tactics (style notes) | Stage 7 | `content_tactics` array from Content Analysis | Array of strings | ["direct_to_camera", "product_demonstration"] |
| Scene changes rate | Temporal Windows | Avg `scene_count` per middle segment ÷ duration | Float | 2.3 scene changes per 10s |
| Text overlay count | Temporal Windows | Sum of `text_overlay_count` across middle segments | Integer | 5-7 total |
| Energy standard | Temporal Windows | Avg `energy_level` from middle segments | String | "Moderate energy (0.35+)" |

---

**⏱️ PHASE 3: CLOSING (Last 3 seconds)**

```
CTA: "Link in bio!" or "Save this for later!"

Execution:
• Peak energy (most enthusiastic moment of entire video)
• Point to save button or bio link (gesture/visual cue)
• Hold final frame 1-2 seconds (give viewers time to click)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| CTA type | Stage 7 | `caption_analysis.cta_type` from Content Analysis | String | "link_in_bio" |
| CTA example phrase | Stage 7 | LLM-generated example based on cta_type | String | "Link in bio!" |
| Peak energy note | Temporal Windows | Based on `energy_max` from last 3s window | String | "Peak energy (0.9+)" |
| Visual cue | Temporal Windows | Based on presence of gestures/visual elements | String | "Point to save button" |

---

**CAPTION STRUCTURE** (Don't skip this!)

```
[Question that matches your video opening]
[1-2 sentence description or teaser]
[Call-to-action: "Link in bio!" or "Save this!"]
#keyword1 #keyword2 #keyword3

Details:
• Start caption with question (matches video hook)
• Keep short: <100 characters before hashtags
• Use 1-4 emojis (not excessive)
• Include 5-10 relevant hashtags
• Place hashtags at END (not mixed into text)
• Always include clear CTA
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Hook type | Stage 7 | `caption_analysis.hook_type` | String | "question" |
| Caption length | Stage 7 | `caption_analysis.caption_length` | String | "short" |
| Emoji usage | Stage 7 | `caption_analysis.emoji_usage` | String | "some" (1-4) |
| Hashtag count | Stage 7 | `caption_analysis.hashtag_count` | Integer | 7 |
| Hashtag placement | Stage 7 | `caption_analysis.hashtag_placement` | String | "end" |
| CTA type | Stage 7 | `caption_analysis.cta_type` | String | "link_in_bio" |

---

#### Pre-Post Checklist

**Design Decision**: Medium Checklist (5-7 Items, Pattern-Specific)

**Why This Length**:
- **UX best practice**: 5-7 items is optimal for checklist compliance (research-backed sweet spot)
- **Pattern-specific value**: Each item checks unique elements of this formula, not generic criteria
- **Balances quality with usability**: Catches pattern-breaking mistakes without overwhelming creators
- **Mobile-optimized**: 5-7 items fit comfortably on phone screen with readable 12pt font
- **Realistic compliance**: Creators will check 5-7 items (1-2 min) but would skip 10+ items
- **Maps to 3-phase structure**: 1-2 items for Hook, 2-3 items for Middle, 1-2 items for Closing/Caption

**Structure**: 6-item checklist aligned to 3-Phase Blueprint

```
✓ CHECKLIST BEFORE POSTING

Hook:
□ Hook strategy used in first 3s? (problem-solution question)

Middle Content:
□ Keywords mentioned? (gut health, protein)
□ Engagement tactics included? (personal testimony, before/after)
□ Execution standards met? (5-7 text overlays, fast pacing)

Closing + Caption:
□ CTA at end?
□ Caption structure followed? (question hook + CTA + hashtags at end)
```

**Item Count**: 5-7 items per report (varies slightly per pattern)

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Checklist items (5-7 total) | Stage 7 | `verification_checklist` in winning_formulas.json | String (array) | ["Hook strategy used in first 3s?", "Keywords mentioned?", etc.] |
| Items grouped by phase | Stage 7 | Optional: `checklist_grouped` (Hook, Middle, Closing sections) | Object | {hook: [...], middle: [...], closing: [...]} |

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

### QR Code Implementation

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

**Status**: ✅ **COMPLETE**

**Audience**: Tumi Labs clients (business owners)

**Purpose**: Competitive intelligence - benchmark 1 competitor vs client baseline

**Deliverable**: 1 PDF comparing 1 competitor vs client

**Format**: 4-page PDF (desktop-optimized, executive-focused)

**Reading Time**: 8-10 minutes (scannable in 3 minutes)

---

### Input Data Sources

- Competitor Stage 7: `winning_formulas.json`
- Competitor Stage 6: `rf_video_analysis.json`, `kmeans_analysis.json`
- Competitor Stage 1: `winner_analysis.json` (bucket distribution)
- Competitor Stage 2: Video metadata (URLs, view counts, hashtags, timestamps)
- Competitor Stage 2.7: `content_analysis` outputs (content categories, hook strategies)
- Client baseline: All of above for client (for benchmarking)
- Config: CLI parameters (`--competitor`, `--client`, `--analysis-period`)

---

### Design Decisions Locked

- ✅ Page count: 4 pages
- ✅ Analysis period: Last 90 days
- ✅ Hashtag depth: Top 10 hashtags
- ✅ Content category: Competitor only (no side-by-side)
- ✅ QR codes: 1 code (competitor's top video)
- ✅ Data type: Single snapshot analysis
- ✅ Comparison approach: Competitor focus with client baseline context

---

### Page 1: Competitive Overview & Posting Activity

**Purpose**: Establish analysis scope, show competitor's posting behavior

---

#### Header Section

```
Competitive Intelligence Report
Competitor: @rival_brand
Client Baseline: @acme_nutrition
Analysis Period: Last 90 days
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor handle | Config | CLI parameter `--competitor` | String | "@rival_brand" |
| Client handle | Config | CLI parameter `--client` | String | "@acme_nutrition" |
| Analysis period | Config | CLI parameter `--analysis-period` (default "Last 90 days") | String | "Last 90 days" |

---

#### Analysis Scope

```
COMPETITOR ANALYSIS SCOPE:
Videos Analyzed: 127
Total Video Duration: 42 minutes
Duration Range: 0-120 seconds (8 distinct buckets)
Content Elements Tracked: 60+ features per video

CLIENT BASELINE:
Videos Analyzed: 89
Total Video Duration: 28 minutes
Duration Range: 0-120 seconds (8 distinct buckets)

Analysis Method:
Multi-dimensional machine learning and AI content analysis:

• Visual & Behavioral Pattern Recognition - Advanced ML analyzed 60+ features
  per video (eye contact, pacing, energy levels, scene transitions, gesture
  frequency) to identify what separates top performers from average content

• Content & Messaging Intelligence - AI-powered analysis of video transcripts
  and captions identified trending hook strategies, audience pain points,
  keywords, and engagement tactics unique to this competitor's content

• Competitive Pattern Discovery - K-Means clustering revealed 3-5 distinct creative
  strategies per video length, validated by Random Forest classification models

Result: Comprehensive competitive intelligence covering posting behavior, content
strategy, creative patterns, and strategic opportunities.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor videos analyzed | Stage 1 (Comp) | `total_videos_analyzed` in winner_analysis.json | Integer | 127 |
| Competitor total duration | Stage 2 (Comp) | Sum of all video durations from metadata | String | "42 minutes" |
| Client videos analyzed | Stage 1 (Client) | `total_videos_analyzed` in winner_analysis.json | Integer | 89 |
| Client total duration | Stage 2 (Client) | Sum of all video durations from metadata | String | "28 minutes" |

---

#### Posting Activity Intelligence

```
POSTING FREQUENCY:
Competitor: 14 videos per week (average over last 90 days)
Client: 10 videos per week (average over last 90 days)
→ Competitor posts 40% more frequently

POSTING CONSISTENCY:
Competitor: High (posts 12-16 videos weekly, low variance)
Client: Moderate (posts 7-13 videos weekly, medium variance)

CONTENT VELOCITY:
Recent 30 days: 16 videos/week (competitor accelerating)
Prior 60 days: 13 videos/week
→ 23% increase in posting rate

ANALYSIS PERIOD COVERAGE:
Competitor: 127 videos analyzed (from 180 total posted in 90 days)
Client: 89 videos analyzed (from 120 total posted in 90 days)
Coverage: Top 70% of content by engagement
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor posting frequency | Stage 2 (Comp) | Count videos in last 90 days ÷ 13 weeks | Float | 14 videos/week |
| Client posting frequency | Stage 2 (Client) | Count videos in last 90 days ÷ 13 weeks | Float | 10 videos/week |
| Posting frequency gap | Calculated | (Comp freq - Client freq) / Client freq × 100% | Integer (%) | 40% |
| Competitor consistency | Calculated | Weekly variance (Low/Medium/High based on std deviation) | String | "High" |
| Client consistency | Calculated | Weekly variance (Low/Medium/High based on std deviation) | String | "Moderate" |
| Recent velocity (30 days) | Stage 2 (Comp) | Count videos in last 30 days ÷ 4.3 weeks | Float | 16 videos/week |
| Prior velocity (60 days) | Stage 2 (Comp) | Count videos in days 31-90 ÷ 8.6 weeks | Float | 13 videos/week |
| Velocity change | Calculated | (Recent - Prior) / Prior × 100% | Integer (%) | 23% |
| Competitor total posted | Stage 2 (Comp) | Count all videos in 90-day period | Integer | 180 |
| Competitor analyzed | Stage 1 (Comp) | `total_videos_analyzed` | Integer | 127 |
| Client total posted | Stage 2 (Client) | Count all videos in 90-day period | Integer | 120 |
| Client analyzed | Stage 1 (Client) | `total_videos_analyzed` | Integer | 89 |
| Coverage description | Config | Based on `--mode` (e.g., "Top 70% by engagement") | String | "Top 70% of content by engagement" |

---

### Page 2: Content Strategy & Hashtag Intelligence

**Purpose**: Show where competitor focuses content efforts and hashtag strategy

---

#### Section 1: Bucket Strategy Comparison

```
WHERE COMPETITOR FOCUSES CONTENT:

[Horizontal bar chart showing % of videos per bucket]

Competitor Distribution:
0-3s:   ██ 3%
3-9s:   ████ 8%
9-13s:  ████████ 12%
13-18s: ████████████ 18%  ← MODERATE VOLUME
18-33s: ████████████████████ 32%  ← HIGH VOLUME
33-60s: ██████████████ 22%  ← MODERATE VOLUME
60-90s: ███ 4%
90-120s: █ 1%

Key Insight: Competitor focuses 52% of content in 18-33s + 33-60s buckets


WHERE YOU FOCUS CONTENT:

Client Distribution:
0-3s:   ████ 6%
3-9s:   ████████ 12%
9-13s:  ██████████ 15%
13-18s: ████████████████ 25%  ← HIGH VOLUME
18-33s: ██████████████ 22%  ← MODERATE VOLUME
33-60s: ████████ 13%
60-90s: ████ 6%
90-120s: █ 1%

Key Insight: You focus 47% of content in 13-18s + 18-33s buckets


STRATEGIC DIFFERENCES:
→ Competitor invests heavily in 33-60s content (22% vs your 13%)
→ You focus more on 13-18s content (25% vs competitor's 18%)
→ Competitor produces 2x more long-form content (60s+)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor % per bucket (8 rows) | Stage 1 (Comp) | `bucket_distribution` in winner_analysis.json | Integer (%) | 3, 8, 12, 18, 32, 22, 4, 1 |
| Client % per bucket (8 rows) | Stage 1 (Client) | `bucket_distribution` in winner_analysis.json | Integer (%) | 6, 12, 15, 25, 22, 13, 6, 1 |
| Competitor key insight | Calculated | Sum of top 2 buckets percentages + bucket names | String | "52% of content in 18-33s + 33-60s" |
| Client key insight | Calculated | Sum of top 2 buckets percentages + bucket names | String | "47% of content in 13-18s + 18-33s" |
| Strategic differences (3 items) | Calculated | Identify largest percentage gaps between competitor and client | String (array) | ["Competitor invests heavily in 33-60s...", etc.] |

---

#### Section 2: Bucket Performance Comparison

```
COMPETITOR PERFORMANCE BY DURATION:

13-18s:  580K avg views  ⭐⭐⭐⭐
18-33s:  620K avg views  ⭐⭐⭐⭐⭐  ← BEST BUCKET
33-60s:  490K avg views  ⭐⭐⭐⭐
(Other buckets: 150K-380K avg views)

Competitor's Sweet Spot: 18-33s (highest performance + high volume)


YOUR PERFORMANCE BY DURATION:

13-18s:  520K avg views  ⭐⭐⭐⭐
18-33s:  460K avg views  ⭐⭐⭐⭐
33-60s:  410K avg views  ⭐⭐⭐
(Other buckets: 120K-340K avg views)

Your Sweet Spot: 13-18s (highest performance + high volume)


PERFORMANCE GAPS:

13-18s:  -60K gap  (Competitor: 580K vs You: 520K = -12% performance gap)
18-33s:  -160K gap (Competitor: 620K vs You: 460K = -35% performance gap) ⚠️ BIGGEST GAP
33-60s:  -80K gap  (Competitor: 490K vs You: 410K = -19% performance gap)

Key Insight: Competitor outperforms in all major buckets, with largest gap in 18-33s
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor avg views per bucket | Stage 1 (Comp) | `avg_views` per bucket in winner_analysis.json | Integer (formatted with K) | 580K, 620K, 490K |
| Competitor star ratings | Calculated | Based on view performance tier (5 stars = highest, 1 star = lowest) | String (emoji) | ⭐⭐⭐⭐⭐ |
| Competitor sweet spot | Stage 1 (Comp) | Bucket with highest avg_views + sufficient volume | String | "18-33s" |
| Client avg views per bucket | Stage 1 (Client) | `avg_views` per bucket in winner_analysis.json | Integer (formatted with K) | 520K, 460K, 410K |
| Client star ratings | Calculated | Based on view performance tier | String (emoji) | ⭐⭐⭐⭐ |
| Client sweet spot | Stage 1 (Client) | Bucket with highest avg_views + sufficient volume | String | "13-18s" |
| Performance gap (per bucket) | Calculated | Competitor avg_views - Client avg_views | Integer (formatted with K) | -60K, -160K, -80K |
| Performance gap % (per bucket) | Calculated | (Comp views - Client views) / Client views × 100% | Integer (%) | -12%, -35%, -19% |
| Biggest gap bucket | Calculated | Bucket with largest absolute gap | String | "18-33s" |

---

#### Section 3: Top Hashtags Competitor Uses

```
TOP 10 HASHTAGS:

1. #nutrition        (82% of videos)
2. #healthylifestyle (68% of videos)
3. #wellness         (54% of videos)
4. #guthealth        (47% of videos)
5. #protein          (43% of videos)
6. #healthyeating    (38% of videos)
7. #fitfood          (32% of videos)
8. #cleaneating      (28% of videos)
9. #nutritionist     (24% of videos)
10. #healthyliving   (21% of videos)

HASHTAG STRATEGY SUMMARY:

Total unique hashtags: 28
Average hashtags per video: 9
Top 5 hashtags appear in 73% of content (focused strategy)

Strategy Type: Diversified (uses 28 hashtags across content)
Concentration: Top 5 hashtags dominate, but long tail of 23 secondary hashtags
Branded hashtags: None detected
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Top 10 hashtags (list) | Stage 2 (Comp) | Aggregate hashtag frequency from metadata, rank by frequency | String (array) | ["#nutrition", "#healthylifestyle", ...] |
| Hashtag usage % (10 values) | Calculated | (Videos with hashtag / Total videos) × 100% | Integer (%) | 82, 68, 54, 47, 43, 38, 32, 28, 24, 21 |
| Total unique hashtags | Calculated | Count distinct hashtags across all videos | Integer | 28 |
| Avg hashtags per video | Calculated | Total hashtag instances / Total videos | Integer | 9 |
| Top 5 concentration % | Calculated | Avg usage % of top 5 hashtags | Integer (%) | 73 |
| Strategy type | Calculated | If unique hashtags > 20: "Diversified", else "Focused" | String | "Diversified" |
| Branded hashtags | Calculated | Detect hashtags with brand name or custom branded tags | String | "None detected" or list |

---

### Page 3: Creative Pattern Analysis

**Purpose**: Show competitor's winning formulas and content approach

---

#### Section 1: Winning Formulas

```
COMPETITOR'S TOP CREATIVE FORMULAS:

Formula 1: "The Question Hook Recipe Tutorial" (18-33s bucket)
• Engagement: 8.2% avg
• Usage: 24% of competitor's 18-33s content uses this pattern
• Pattern: Opens with question about ingredient/health concern, demonstrates
  recipe step-by-step, shows final result with nutritional benefits

Formula 2: "The Before-After Transformation" (33-60s bucket)
• Engagement: 7.8% avg
• Usage: 31% of competitor's 33-60s content uses this pattern
• Pattern: Shows client/self before state, explains intervention/product,
  reveals after results with testimonial

Formula 3: "The Myth-Busting Reveal" (13-18s bucket)
• Engagement: 7.4% avg
• Usage: 28% of competitor's 13-18s content uses this pattern
• Pattern: States common myth/misconception, explains why it's wrong,
  provides correct information with source/credentials

Formula 4: "The Ingredient Deep-Dive" (18-33s bucket)
• Engagement: 7.1% avg
• Usage: 19% of competitor's 18-33s content uses this pattern
• Pattern: Focuses on single ingredient, explains health benefits,
  shows multiple uses/recipes incorporating it

Formula 5: "The Quick Win Tutorial" (13-18s bucket)
• Engagement: 6.9% avg
• Usage: 22% of competitor's 13-18s content uses this pattern
• Pattern: Promises fast result, demonstrates simple technique,
  provides immediate takeaway viewers can replicate
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Formula names (5 formulas) | Stage 7 (Comp) | `pattern_name` from winning_formulas.json (top 5 by engagement) | String (array) | ["The Question Hook Recipe Tutorial", ...] |
| Formula buckets (5 values) | Stage 7 (Comp) | `bucket_range` per formula | String (array) | ["18-33s", "33-60s", "13-18s", ...] |
| Formula engagement (5 values) | Stage 7 (Comp) | Industry benchmark mapping or performance metric | Float (%) | 8.2, 7.8, 7.4, 7.1, 6.9 |
| Formula usage % (5 values) | Calculated | (Videos using this pattern / Total videos in bucket) × 100% | Integer (%) | 24, 31, 28, 19, 22 |
| Formula descriptions (5 items) | Stage 7 (Comp) | LLM-generated summary from pattern characteristics | String (array) | ["Opens with question about ingredient...", ...] |

---

#### Section 2: Pattern Versatility & Content Mix

```
PATTERN VERSATILITY:

Total distinct formulas: 9 (across all winning buckets)
Formula rotation: High (competitor uses 5-9 different patterns per bucket)
Pattern repetition rate: 24% (avg % of content using single most-used formula)

Insight: Competitor diversifies creative approach, avoiding pattern fatigue


CONTENT CATEGORY MIX:

Recipe Tutorial:        38% of content
Wellness Practice:      28% of content
Supplement Review:      17% of content
Expert Interview:       12% of content
Personal Testimony:     5% of content

Dominant Format: Recipe Tutorial (38% of content)
Secondary Format: Wellness Practice (28% of content)

Strategy: Competitor focuses on instructional content (recipe + wellness = 66%)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Total distinct formulas | Stage 7 (Comp) | Count unique formulas across all buckets in winning_formulas.json | Integer | 9 |
| Formula rotation level | Calculated | If formulas > 6: "High", 4-6: "Medium", <4: "Low" | String | "High" |
| Pattern repetition rate | Calculated | Avg of highest formula usage % across buckets | Integer (%) | 24 |
| Content categories (5 types) | Stage 2.7 (Comp) | Aggregate `content_category` field, count frequency | String (array) | ["Recipe Tutorial", "Wellness Practice", ...] |
| Category percentages (5 values) | Calculated | (Videos with category / Total videos) × 100% | Integer (%) | 38, 28, 17, 12, 5 |
| Dominant format | Calculated | Content category with highest % | String | "Recipe Tutorial" |
| Secondary format | Calculated | Content category with 2nd highest % | String | "Wellness Practice" |
| Strategy insight | Calculated | Sum related categories, identify theme | String | "Competitor focuses on instructional content..." |

---

#### Section 3: Hook Strategy Distribution

```
OPENING PATTERNS COMPETITOR USES:

Question Hook:          42% of content (most common)
Problem-Solution:       31% of content
Direct Statement:       18% of content
Teaser/Curiosity Gap:   9% of content

Dominant Hook: Question Hook (42% of content)

Example Question Hooks from Competitor:
• "Did you know this common food is destroying your gut?"
• "Want to know the secret ingredient nutritionists use?"
• "Ever wonder why you're always bloated after meals?"

Strategy: Competitor leads with questions to create immediate curiosity and engagement
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Hook strategies (4 types) | Stage 2.7 (Comp) | Aggregate `hook_strategy` field, count frequency | String (array) | ["Question Hook", "Problem-Solution", ...] |
| Hook percentages (4 values) | Calculated | (Videos with hook type / Total videos) × 100% | Integer (%) | 42, 31, 18, 9 |
| Dominant hook | Calculated | Hook strategy with highest % | String | "Question Hook" |
| Example hooks (3 items) | Stage 2.7 (Comp) | Sample actual transcript openings from videos using dominant hook | String (array) | ["Did you know this common food...", ...] |
| Strategy insight | Manual | Interpretation of hook distribution | String | "Competitor leads with questions to create..." |

---

#### Section 4: Visual Example

```
[QR CODE - 1" x 1"]

Scan to watch: Competitor's Top Performing Video
Video: 820K views | Duration: 22s (18-33s bucket)
Formula: "The Question Hook Recipe Tutorial"
Hashtags: #nutrition #guthealth #protein #healthyeating

What to observe:
• Question hook in first 2 seconds ("Did you know...")
• Product reveal by second 5
• 8 text overlays throughout video
• Fast pacing (3 scene changes per 10 seconds)
• Clear CTA at end ("Link in bio!")
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| QR Code image | Stage 2 (Comp) | Generate QR code from top performer video URL | QR Code Image | Links to TikTok video |
| Video views | Stage 2 (Comp) | `view_count` from top performer video metadata | Integer (formatted with K) | 820K |
| Video duration | Stage 2 (Comp) | `duration` from video metadata | Integer (seconds) | 22s |
| Video bucket | Stage 1 (Comp) | Bucket classification from duration | String | "18-33s" |
| Formula used | Stage 7 (Comp) | Map video to formula from winning_formulas.json | String | "The Question Hook Recipe Tutorial" |
| Hashtags | Stage 2 (Comp) | `video_hashtags` from metadata (first 4) | String (array) | ["#nutrition", "#guthealth", ...] |
| Observation notes (4-5 items) | Stage 7 (Comp) | Pattern characteristics from formula analysis | String (array) | ["Question hook in first 2 seconds", ...] |

**Video Selection Criteria**:
- **Priority 1**: Highest view count from winning bucket
- **Priority 2**: Newest video (if multiple high performers - reduces deletion risk)
- **Priority 3**: Prefer videos from accounts with 100K+ followers (stability)

---

### Page 4: Strategic Intelligence & Recommendations

**Purpose**: Provide actionable insights and prioritized recommendations

---

#### Section 1: Audience Targeting Intelligence

```
WHAT PAIN POINTS COMPETITOR ADDRESSES:

Top Pain Points Mentioned:
1. Bloating/digestive issues (mentioned in 52% of content)
2. Low energy/fatigue (mentioned in 38% of content)
3. Weight management struggles (mentioned in 31% of content)
4. Inflammation concerns (mentioned in 24% of content)
5. Gut health problems (mentioned in 22% of content)

Insight: Competitor focuses heavily on digestive health and energy (90% of content
addresses at least one of these issues)


WHAT TOPICS/KEYWORDS COMPETITOR DOMINATES:

Top Keywords:
1. "gut health" (appears in 68% of content)
2. "protein" (appears in 54% of content)
3. "anti-inflammatory" (appears in 42% of content)
4. "metabolism" (appears in 36% of content)
5. "fiber" (appears in 31% of content)

Insight: Competitor owns the "gut health + protein" conversation in this niche


ENGAGEMENT DRIVERS COMPETITOR LEVERAGES:

Top Tactics:
1. Before/after reveals (used in 47% of content) - Highest engagement driver
2. Personal testimony (used in 41% of content)
3. Specific metrics mentioned (used in 38% of content) - e.g., "Lost 15 lbs"
4. Product recommendations (used in 34% of content)
5. Expert credentials shown (used in 28% of content)

Insight: Competitor builds trust through transformation proof and personal stories
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Pain points (5 items) | Stage 2.7 (Comp) | Aggregate `pain_points` array, count frequency | String (array) | ["Bloating/digestive issues", "Low energy/fatigue", ...] |
| Pain point % (5 values) | Calculated | (Videos mentioning pain point / Total videos) × 100% | Integer (%) | 52, 38, 31, 24, 22 |
| Pain point insight | Calculated | Identify dominant themes, calculate combined coverage | String | "Competitor focuses heavily on digestive health..." |
| Keywords (5 items) | Stage 2.7 (Comp) | Aggregate `keywords` array, count frequency | String (array) | ["gut health", "protein", ...] |
| Keyword % (5 values) | Calculated | (Videos mentioning keyword / Total videos) × 100% | Integer (%) | 68, 54, 42, 36, 31 |
| Keyword insight | Calculated | Identify keyword clusters/themes | String | "Competitor owns the 'gut health + protein' conversation" |
| Engagement drivers (5 items) | Stage 2.7 (Comp) | Aggregate `engagement_drivers` array, count frequency | String (array) | ["Before/after reveals", "Personal testimony", ...] |
| Driver % (5 values) | Calculated | (Videos using driver / Total videos) × 100% | Integer (%) | 47, 41, 38, 34, 28 |
| Driver insight | Calculated | Identify trust-building patterns | String | "Competitor builds trust through transformation proof..." |

---

#### Section 2: Strategic Gaps & Opportunities

```
TIER 1: IMMEDIATE ACTION (Biggest Competitive Gaps)

Gap 1: Content Volume in 33-60s Bucket
• Competitor posts 22% of content here vs your 13% (69% more volume)
• Competitor averages 490K views in this bucket vs your 410K
• Opportunity: Increase 33-60s content from 13% to 20% of output
• Expected Impact: Close 80K avg view gap in this bucket

Gap 2: "Question Hook" Strategy Adoption
• Competitor uses question hooks in 42% of content vs your 18%
• Question hooks drive 8.2% avg engagement for competitor
• Opportunity: Increase question hook usage from 18% to 35% of content
• Expected Impact: Improve hook engagement by estimated 1.5-2 percentage points

Gap 3: Hashtag Diversification
• Competitor uses 28 unique hashtags vs your 12
• Competitor's top hashtag (#healthylifestyle) reaches audiences you're missing
• Opportunity: Add 10-15 secondary hashtags from competitor's strategy
• Expected Impact: Expand reach to new audience segments


TIER 2: OPTIMIZATION OPPORTUNITIES (Tactical Improvements)

Improvement 1: Text Overlay Density
• Competitor averages 8 text overlays per video vs your 4
• Correlates with higher engagement in competitor's content
• Action: Increase text overlays to 6-8 per video (gradual ramp-up)

Improvement 2: Product Reveal Timing
• Competitor shows product by second 5 vs your average of second 11
• Earlier reveals maintain viewer attention in first critical seconds
• Action: Move product reveal to 3-7 second range in new content

Improvement 3: Before/After Transformation Tactic
• Competitor uses in 47% of content vs your 15%
• Drives highest engagement among competitor's tactics
• Action: Incorporate before/after reveals in 35% of content (especially 33-60s videos)


TIER 3: MAINTAIN YOUR STRENGTHS (Where You Win)

Strength 1: 13-18s Bucket Performance
• You average 520K views in 13-18s (vs competitor's 580K = only -12% gap)
• You allocate more volume to this bucket (25% vs competitor's 18%)
• Strategy: Maintain 13-18s focus - this is your competitive advantage bucket

Strength 2: Posting Consistency in Short-Form
• Your 13-18s content performs well with consistent output
• Competitor is weaker in shorter durations (3-13s buckets)
• Strategy: Defend this positioning - don't abandon short-form content
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| **TIER 1 GAPS** | | | | |
| Gap 1 title | Calculated | Identify largest bucket volume difference | String | "Content Volume in 33-60s Bucket" |
| Gap 1 metrics (4 lines) | Calculated | Competitor % vs Client %, view comparison, opportunity, impact | String (array) | ["Competitor posts 22%...", "Competitor averages 490K...", ...] |
| Gap 2 title | Calculated | Identify largest hook strategy difference | String | "Question Hook Strategy Adoption" |
| Gap 2 metrics (4 lines) | Calculated | Hook usage %, engagement, opportunity, impact | String (array) | ["Competitor uses question hooks in 42%...", ...] |
| Gap 3 title | Manual | Based on hashtag analysis | String | "Hashtag Diversification" |
| Gap 3 metrics (3 lines) | Calculated | Unique hashtag count comparison, top hashtag, opportunity | String (array) | ["Competitor uses 28 unique hashtags...", ...] |
| **TIER 2 IMPROVEMENTS** | | | | |
| Improvement 1-3 titles | Calculated | Identify quantitative behavior differences | String (array) | ["Text Overlay Density", "Product Reveal Timing", ...] |
| Improvement metrics | Calculated | Competitor metric vs Client metric + action recommendation | String (array per improvement) | ["Competitor averages 8 text overlays...", ...] |
| **TIER 3 STRENGTHS** | | | | |
| Strength 1-2 titles | Calculated | Identify buckets/patterns where Client outperforms or matches closely | String (array) | ["13-18s Bucket Performance", "Posting Consistency..."] |
| Strength metrics | Calculated | Client performance, comparison to competitor, defensive strategy | String (array per strength) | ["You average 520K views...", ...] |

---

#### Section 3: Untapped Opportunities

```
SUCCESSFUL PATTERNS COMPETITOR HASN'T ADOPTED:

From broader #nutrition hashtag analysis, these high-performing patterns are
NOT present in competitor's content (opportunity for you to differentiate):

1. "The Side-by-Side Comparison" (18-33s bucket, 7.9% avg engagement)
   • Competitor uses in <5% of content
   • You could own this pattern in the niche

2. "The Expert Interview Format" (33-60s bucket, 7.2% avg engagement)
   • Competitor uses in only 12% of 33-60s content
   • Room for you to establish authority through expert collaborations

3. "The Ingredient Shock Hook" (13-18s bucket, 7.6% avg engagement)
   • Competitor doesn't use shock/surprise hooks often
   • Opportunity to stand out with bold claims (backed by science)

Insight: While competitor dominates volume and consistency, there are creative
patterns from the broader market that they haven't fully exploited. You can
differentiate by owning these patterns.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Untapped patterns (3 items) | Stage 7 (Hashtag) + Stage 7 (Comp) | Identify high-performing hashtag formulas with low competitor usage | String (array) | ["The Side-by-Side Comparison", "The Expert Interview Format", ...] |
| Pattern buckets (3 values) | Stage 7 (Hashtag) | Bucket range for each untapped pattern | String (array) | ["18-33s", "33-60s", "13-18s"] |
| Pattern engagement (3 values) | Stage 7 (Hashtag) | Engagement metric for pattern from hashtag analysis | Float (%) | 7.9, 7.2, 7.6 |
| Competitor usage (3 values) | Calculated | % of competitor's content using this pattern | String (array) | ["<5%", "12%", "doesn't use often"] |
| Opportunity notes (3 items) | Manual | Strategic differentiation positioning | String (array) | ["You could own this pattern...", ...] |
| Overall insight | Manual | Summary of differentiation opportunity | String | "While competitor dominates volume..." |

**Data Requirement**: This section requires BOTH competitor Stage 7 analysis AND hashtag Stage 7 analysis to identify gaps.

---

#### Section 4: Next Steps

```
RECOMMENDED ACTIONS:

Immediate (Next 30 Days):
□ Increase posting frequency from 10 to 12-13 videos/week (close volume gap)
□ Shift 5-8% of content to 33-60s bucket (from 13% to 18-20% allocation)
□ Incorporate question hooks in 25% of new content (up from 18%)
□ Add 5 new hashtags from competitor's top 10 to your rotation

60-90 Day Roadmap:
□ Test "before/after reveal" tactic in 30% of content (up from 15%)
□ Increase text overlay density to 6-8 per video (from current 4 average)
□ Move product reveals earlier (3-7 second range vs current 11 second avg)
□ Experiment with 2-3 untapped formulas competitor hasn't adopted

Ongoing Monitoring:
□ Track competitor's posting frequency monthly (detect acceleration/slowdown)
□ Monitor new hashtags competitor adopts (update your strategy)
□ Analyze new formulas competitor tests (learn from their experiments)

Want execution guides for competitor's top formulas?
Contact us to receive creator-ready reports with step-by-step implementation
for your content team.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Immediate actions (4 items) | Calculated | Derived from Tier 1 gaps with specific targets | String (array) | ["Increase posting frequency from 10 to 12-13...", ...] |
| 60-90 day actions (4 items) | Calculated | Derived from Tier 2 improvements with test approach | String (array) | ["Test 'before/after reveal' tactic...", ...] |
| Ongoing monitoring (3 items) | Manual | Strategic tracking recommendations | String (array) | ["Track competitor's posting frequency...", ...] |

---

### Mobile Optimization Requirements

**NOT REQUIRED for Template C (Client reports are desktop-focused)**

- Clients will review on desktop/laptop (executive context)
- Standard PDF formatting: 8.5" × 11" letter size, landscape or portrait
- Minimum font sizes: 10pt body text acceptable (not 12pt like creator reports)
- Multi-column layouts allowed (unlike creator reports)

---

### Summary of Key Design Patterns Reused

| Pattern | Original Template | How It's Reused in Template C |
|---------|------------------|------------------------------|
| **Scale of Analysis** | Template A (Hashtag → Client) | Analysis scope section (Page 1) |
| **Bucket Distribution Chart** | Template A (Hashtag → Client) | Bucket strategy comparison (Page 2) |
| **Performance Tiers** | Template A (Hashtag → Client) | Bucket performance comparison (Page 2) |
| **Tiered Recommendations** | Template A (Hashtag → Client) | Strategic gaps section (Page 4: Tier 1/2/3) |
| **QR Code Visual Proof** | Template B (Hashtag → Creator) | Competitor's top video example (Page 3) |
| **Formula List** | Template A (Page 3) | Winning formulas section (Page 3) |
| **Soft CTA** | Template A (Page 3) | Next steps section (Page 4) |
| **Aggregated Insights** | Template B (Checklist) | Audience targeting intelligence (Page 4) |

---

### Data Extraction Requirements for extract_competitor_data.py

**New Calculations Required**:

1. **Posting Frequency Metrics**
   - Videos per week (last 90 days, last 30 days, prior 60 days)
   - Posting consistency (weekly variance analysis)
   - Content velocity change percentage

2. **Hashtag Aggregation**
   - Top 10 hashtags by frequency
   - Total unique hashtags count
   - Average hashtags per video
   - Top 5 concentration percentage

3. **Performance Gap Calculations**
   - Bucket-level view gaps (competitor - client)
   - Bucket-level percentage gaps
   - Identify biggest gap bucket

4. **Content Analysis Aggregations**
   - Content category distribution (from Stage 2.7)
   - Hook strategy distribution (from Stage 2.7)
   - Pain points frequency (from Stage 2.7)
   - Keywords frequency (from Stage 2.7)
   - Engagement drivers frequency (from Stage 2.7)

5. **Pattern Versatility Metrics**
   - Total distinct formulas count
   - Formula rotation level classification
   - Pattern repetition rate

6. **Untapped Opportunities Identification**
   - Compare hashtag analysis formulas vs competitor formulas
   - Identify high-performing hashtag patterns with low competitor usage

---

**Status**: ✅ **COMPLETE** - Template structure finalized with all dynamic field mappings

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

1. **Stage 7 LLM Prompt Update**: Modify report generation prompt to output `pattern_blueprint` structure (3 phases) instead of traditional segment-by-segment breakdown
2. **Template Field Update**: Section 2 (Hashtag → Creator, Page 2) already reflects 3-phase structure (see lines 381-506)
3. **Data Mapping Tables**: Already complete with dynamic fields for all 3 phases (see lines 396-504)
4. **Content Category Assignment**: Stage 7 should select most common `content_category` per cluster (no rotation forcing)

**Implementation Status**: ✅ Template structure finalized and documented

---

**End of Content Analysis Data Capabilities Documentation**
