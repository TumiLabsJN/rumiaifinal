# Stage 8 MVP: Report Template Structures

**Purpose**: Centralized template structure definitions for all Stage 8 PDF reports

**Parent Document**: Stage8MVP.md

**Status**: 4 of 4 templates complete - Section 0 complete, ready for design + development

---

## 1. Hashtag → Client (Executive Report)
```
**Audience**: Tumi Labs clients (business owners)

**Purpose**: Prove ML sophistication, reduce anxiety, provide creator sourcing strategy

**Deliverable**: 1 PDF per hashtag analysis

**Format**: 3-page PDF (desktop-first, mobile-tested)

**Reading Time**: 5-7 minutes (scannable in 2 minutes)
```
---

### Page 1: Scale of Analysis

**Purpose**: Show the business owner how comprehensive the analysis is

---

#### Header Section

```
#vitamin Hashtag Analysis
Analysis Period: Past 2-3 months
Videos Analyzed: 1,826

WINNING CONTENT FORMATS:
Our analysis identified 3 highest-performing duration ranges:
• 18-33 seconds: 43% of top performers
• 13-18 seconds: 12% of top performers
• 60-90 seconds: 11% of top performers

DEEP ANALYSIS:
88 top-performing videos + 23 moderate-performing videos analyzed
using 60+ ML features per video (computer vision, audio analysis, NLP)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Hashtag | Cluster Config | `/config/hashtag_clusters/{target}.json` → `primary_hashtag` | String | "#vitamin" | ✅ **Report 1 Header** |
| Analysis Period | Static | Fixed string: "Past 2-3 months" | String | "Past 2-3 months" | ✅ **Report 1 Header** |
| Videos Analyzed | Cluster Analytics | `/data/clients/{client}/hashtag/{target}/cluster_analytics.json` → `scrape_summary.total_scraped_videos` | Integer | 1,826 | ✅ **Report 1 Header** |
| Winning Buckets (3) | Winner Analysis | `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets` array | Array[String] | ["18-33s", "13-18s", "60-90s"] | ✅ **Report 1 Header** |
| Winning Bucket %s (3) | Winner Analysis | Same file → `top_100_distribution[bucket]` for each winning bucket | Array[Integer] | [43, 12, 11] | ✅ **Report 1 Header** |
| Top Performers Count | Selection Manifest | `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/selection_manifest.json` → Sum all `top_performers` array lengths | Integer | 88 | ✅ **Report 1 Header** |
| Bottom Performers Count | Selection Manifest | Same file → Sum all `bottom_performers` array lengths | Integer | 23 | ✅ **Report 1 Header** |

---

#### Analysis Scope & Methodology

```
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
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Hashtag (in Content & Messaging text) | Cluster Config | `/config/hashtag_clusters/{target}.json` → `primary_hashtag` | String | "#vitamin" | ✅ **Report 1 Header** |

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
Your Top 3 Performing Durations:

Duration | Avg Views  | Avg Engagement | Rating
---------|------------|----------------|------------
18-33s   | 490K       | 1.4%           | ⭐⭐⭐⭐⭐  ← BEST
13-18s   | 520K       | 1.2%           | ⭐⭐⭐⭐
60-90s   | 310K       | 1.3%           | ⭐⭐⭐

These 3 durations represent 75.9% of top-performing #nutrition content.
Your 9 creative reports focus exclusively on these high-opportunity durations.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Winning bucket ranges (3 rows) | Stage 1 | `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets` array | Array[String] | ["18-33s", "13-18s", "60-90s"] | ✅ **Report 1 Header** |
| Avg views per winning bucket (3 rows) | Calculated | `calculate_avg_views_per_bucket()` for each winning bucket: load `selected_videos.json` → filter `is_top_performer == true` → average `playCount` → format with K/M suffix (from Section 0.5.6) | Integer (formatted with K/M) | 1.9M, 2.1M, 980K | ✅ **This session** |
| Avg engagement per winning bucket (3 rows) | Calculated | For each winning bucket: load top performer video IDs from `selected_videos.json` → for each, load `/unified_analysis/{video_id}.json` → `metadata` → call `calculate_engagement_metrics()` → average all rates (from Section 0.5.5) | Float (%) | 15.2, 12.8, 10.5 | ✅ **This session** |
| Star ratings (3 rows) | Calculated | Sort 3 winning buckets by `avg_engagement` DESC (primary), then `avg_views` DESC (secondary) → assign 5 stars (rank 1), 4 stars (rank 2), 3 stars (rank 3) | String (emoji) | ⭐⭐⭐⭐⭐, ⭐⭐⭐⭐, ⭐⭐⭐ | ✅ **This session** |
| Top bucket label | Calculated | Bucket ranked #1 from Field 4 (highest engagement + views) gets "← BEST" label, others blank | String | "← BEST", "", "" | ✅ **This session** |
| Coverage percentage | Calculated | Load `winner_analysis.json` → for each bucket in `top_3_buckets`, sum counts from `top_100_distribution` → divide by total of all buckets → multiply by 100 → round to 1 decimal | Float (%) | 75.9 | ✅ **This session** |
| Hashtag (in description) | Config | `/config/hashtag_clusters/{target}.json` → `primary_hashtag` | String | "#nutrition" | ✅ **Report 1 Header** |

**Decision**: ✅ Show engagement metrics ONLY for 3 winning buckets (Option A - data availability constraint). Cannot calculate engagement for non-winning buckets as `selection_manifest.json` only contains video IDs for top 3 buckets.

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
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Tier 1 bucket ranges (3 buckets) | Stage 1 | `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets` array | Array[String] | ["18-33s", "13-18s", "60-90s"] | ✅ **Report 1 Header** |
| Avg views per winning bucket | Calculated | `calculate_avg_views_per_bucket()` for each winning bucket: load `selected_videos.json` → filter `is_top_performer == true` → average `playCount` → format with K/M suffix (from Section 0.5.6) | Integer (formatted with K/M) | 1.9M, 575K, 2.1M | ✅ **This session** |
| Performance labels | Calculated | Based on Section 2 star rating rank (engagement + views): Rank 1 (5 stars) = "highest performance", Rank 2 (4 stars) = "strong performance + volume", Rank 3 (3 stars) = "proven success" | String | "highest performance", "strong performance + volume", "proven success" | ✅ **This session** |


---

#### Section 4: Content Intelligence

**Purpose**: Show what types of content perform best in this hashtag

```
CONTENT INTELLIGENCE:

What Type of Content Wins:
1. Recipe Tutorial (38% of top performers)
2. Wellness Practice (28% of top performers)
3. Supplement Review (22% of top performers)

What Hooks Capture Attention:
• Problem-Solution approach (42% of winning videos)
• Question Hook (35% of winning videos)
• Direct Statement (23% of winning videos)

What Topics Resonate:
Top Keywords: #guthealth, #protein, #antiinflammatory, #metabolism
Pain Points Addressed: Bloating (48%), Low Energy (42%), Inflammation (38%)

What Drives Engagement:
• Before/After Reveal (45% of top performers use this)
• Personal Testimony (38% of top performers)
• Specific Metrics Mentioned (52% of top performers)

Caption Strategy That Works:
• Optimal Hashtag Count: 7 hashtags
• Caption Length: Short captions (<100 characters) in 68% of winners
• Emoji Usage: Light emoji use (1-4 emojis) in 72% of winners
• Call-to-Action: "Link in bio" CTA in 58% of top performers
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Top 3 content categories | Stage 2.7 | For each of 120 winning videos, load `{bucket_base}/content_analysis/{video_id}_content.json` → `content_category` field → aggregate all values, rank by frequency, take top 3 with percentages: `(count / 120) × 100%` (from Section 0.5.1) | String (array) with percentages | ["Recipe Tutorial: 38%", "Wellness Practice: 28%", "Supplement Review: 22%"] | ✅ **This session** |
| Top 3 hook strategies | Stage 2.7 | For each of 120 winning videos, load content.json → `hook_strategy` field → aggregate, rank by frequency, take top 3 with percentages (from Section 0.5.1) | String (array) with percentages | ["Problem-Solution: 42%", "Question Hook: 35%", "Direct Statement: 23%"] | ✅ **This session** |
| Top 4 keywords | Stage 2.7 | For each of 120 winning videos, load content.json → `keywords` array → flatten all arrays, aggregate unique keywords, rank by frequency, take top 4 (from Section 0.5.1) | String (array) | ["#guthealth", "#protein", "#antiinflammatory", "#metabolism"] | ✅ **This session** |
| Top 3 pain points | Stage 2.7 | For each of 120 winning videos, load content.json → `pain_points` array → flatten, aggregate, rank, take top 3 with percentages (from Section 0.5.1) | String (array) with percentages | ["Bloating: 48%", "Low Energy: 42%", "Inflammation: 38%"] | ✅ **This session** |
| Top 3 engagement drivers | Stage 2.7 | For each of 120 winning videos, load content.json → `engagement_drivers` array → flatten, aggregate, rank, take top 3 with percentages (from Section 0.5.1) | String (array) with percentages | ["Before/After Reveal: 45%", "Personal Testimony: 38%", "Specific Metrics Mentioned: 52%"] | ✅ **This session** |
| Optimal hashtag count | Stage 2.7 | For each of 120 winning videos, load content.json → `caption_analysis.hashtag_count` → calculate mean → round to nearest integer (from Section 0.5.1) | Integer | 7 | ✅ **This session** |
| ~~Hashtag strategy breakdown~~ | ~~Stage 2.7~~ | ~~`caption_analysis.hashtag_strategy.niche_count` and `broad_count`~~ | ~~String~~ | ~~"5 niche, 2 broad"~~ | ❌ **REMOVED FROM SCHEMA** (ContentAnalysisCHILDTI.md lines 533-537: hashtag_strategy object removed, only hashtag_count and hashtag_placement retained) |
| Caption length winner | Stage 2.7 | For each of 120 winning videos, load content.json → `caption_analysis.caption_length` (values: "short" <100 chars or "long" 100+ chars) → find most common value with percentage (from Section 0.5.1) | String with percentage | "Short captions (<100 chars): 68%" | ✅ **This session** |
| Emoji usage winner | Stage 2.7 | For each of 120 winning videos, load content.json → `caption_analysis.emoji_usage` (values: "none" (0), "some" (1-4), "many" (5+)) → find most common value with percentage (from Section 0.5.1) | String with percentage | "Some emoji use (1-4 emojis): 72%" | ✅ **This session** |
| Top CTA type | Stage 2.7 | For each of 120 winning videos, load content.json → `caption_analysis.cta_type` (values: "link_in_bio", "save_post", "comment", "follow", "share", "tag_friend", "none") → aggregate, rank, take top 1 with percentage (from Section 0.5.1) | String with percentage | "Link in bio: 58%" | ✅ **This session** |

#### Section 4: Quantitative Intelligence

**Duration Bucket [BUCKET_1_NAME]:**
  • Formula 1: [BUCKET_1_FORMULA_1_NAME]
  • Formula 2: [BUCKET_1_FORMULA_2_NAME]
  • Formula 3: [BUCKET_1_FORMULA_3_NAME]

**Duration Bucket [BUCKET_2_NAME]:**
  • Formula 4: [BUCKET_2_FORMULA_1_NAME]
  • Formula 5: [BUCKET_2_FORMULA_2_NAME]
  • Formula 6: [BUCKET_2_FORMULA_3_NAME]

**Duration Bucket [BUCKET_3_NAME]:**
  • Formula 7: [BUCKET_3_FORMULA_1_NAME]
  • Formula 8: [BUCKET_3_FORMULA_2_NAME]
  • Formula 9: [BUCKET_3_FORMULA_3_NAME]


**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Duration Bucket names (3 buckets) | Stage 1 | `/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets` array | Array[String] | ["18-33s", "13-18s", "60-90s"] | ✅ **Verified** |
| Formula names per bucket (9 total) | Stage 7 | For each winning bucket in `top_3_buckets`: `/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/buckets/bucket_{bucket_name}/ml_analysis/llm/winning_formulas.json` → `creative_reports[0-2].formula_name` → Extract 3 formula names per bucket | Array[String] per bucket | Bucket 1: ["The Silent-to-Vocal Engagement Journey", "The Visual Storytelling Formula", "The Vocal Variety Formula"] | ✅ **Verified** |

**Notes**:
- Bucket names are dynamically populated from `winner_analysis.json` (same source as Section 3)
- Each bucket always has exactly 3 formulas (guaranteed by Stage 7 output schema)
- Formula names are LLM-generated and unique per bucket/hashtag combination
- Formula naming convention typically follows: "The [Pattern] Formula" or "The [Pattern] Journey"

---

#### What Each Report Contains

Each 2-page report includes:
  • Proof with numbers (engagement differences)
  • Second-by-second execution guide
  • Pre-post checklist

**Would you like to review a sample report?** Contact us at [email]

**Decision**: ✅ Minimal Page 3 with report distribution list and sample report offer. "How to Use These Reports", "What Makes These Reports Effective", and "Next Steps" sections removed (onboarding material, not recurring report content).

---

## 2. Hashtag → Creator (Content Creator Report)

```
**Audience**: Content creators (affiliates)

**Purpose**: Deliver actionable creative formulas with proof and execution steps

**Deliverable**: 9 PDFs per hashtag (3 buckets × 3 formulas each)

**Format**: 2-page PDF (**MOBILE-OPTIMIZED** - minimum 12pt body, 16pt+ headings, portrait layout)

**Reading Time**: 2-3 minutes
```

---

### Page 1: "Why This Works" (Hook with Proof + Pattern)

---

#### Header Section

```
Pattern Name: "The Question Hook Formula"
Duration: 18-33s | Hashtag: #nutrition
```
**Hashtag top 3 Performing Durations:**
_If you'd like the creative reports from other durations, please let us know!_

Duration | Avg Views  | Avg Engagement | Rating
---------|------------|----------------|------------
18-33s   | 490K       | 1.4%           | ⭐⭐⭐⭐⭐  ← BEST
13-18s   | 520K       | 1.2%           | ⭐⭐⭐⭐
60-90s   | 310K       | 1.3%           | ⭐⭐⭐

These 3 durations represent 75.9% of top-performing #nutrition content.

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Hashtag | Cluster Config | `/config/hashtag_clusters/{target}.json` → `primary_hashtag` | String | "#vitamin" | ✅ **Report 1 Header** |
| Duration | Winner Analysis | `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets[i]` where i=0,1,2 (9 PDFs: 3 buckets × 3 formulas, each PDF maps to one winning bucket) | String | "18-33s" | ✅ **Report 1 Header** |
| Pattern Name | Stage 7 | `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/buckets/bucket_{name}/ml_analysis/llm/winning_formulas.json` → `creative_reports[j].formula_name` where j=0,1,2 (3 formulas per bucket) | String | "The Question Hook Formula" | ⚠️ **NOT VERIFIED** (Stage 7 future work) |
| Winning bucket ranges (3 rows) | Stage 1 | `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets` array | Array[String] | ["18-33s", "13-18s", "60-90s"] | ✅ **Report 1 Header** |
| Avg views per winning bucket (3 rows) | Calculated | `calculate_avg_views_per_bucket()` for each winning bucket: load `selected_videos.json` → filter `is_top_performer == true` → average `playCount` → format with K/M suffix (from Section 0.5.6) | Integer (formatted with K/M) | 1.9M, 2.1M, 980K | ✅ **This session** |
| Avg engagement per winning bucket (3 rows) | Calculated | For each winning bucket: load top performer video IDs from `selected_videos.json` → for each, load `/unified_analysis/{video_id}.json` → `metadata` → call `calculate_engagement_metrics()` → average all rates (from Section 0.5.5) | Float (%) | 15.2, 12.8, 10.5 | ✅ **This session** |
| Star ratings (3 rows) | Calculated | Sort 3 winning buckets by `avg_engagement` DESC (primary), then `avg_views` DESC (secondary) → assign 5 stars (rank 1), 4 stars (rank 2), 3 stars (rank 3) | String (emoji) | ⭐⭐⭐⭐⭐, ⭐⭐⭐⭐, ⭐⭐⭐ | ✅ **This session** |
| Top bucket label | Calculated | Bucket ranked #1 from Field 4 (highest engagement + views) gets "← BEST" label, others blank | String | "← BEST", "", "" | ✅ **This session** |
| Coverage percentage | Calculated | Load `winner_analysis.json` → for each bucket in `top_3_buckets`, sum counts from `top_100_distribution` → divide by total of all buckets → multiply by 100 → round to 1 decimal | Float (%) | 75.9 | ✅ **This session** |
| Hashtag (in description) | Config | `/config/hashtag_clusters/{target}.json` → `primary_hashtag` | String | "#nutrition" | ✅ **Report 1 Header** |

---


#### The Proof (Real Performance Data)

```
📊 PERFORMANCE COMPARISON:

Videos using these patterns (Top Cluster):
• Average Views: 620K+
• Average Engagement: 1.2%

Videos NOT using this pattern (Bottom Cluster):
• Average Views: 380K
• Average Engagement: 0.8%

RESULTS:
→ 1.6x MORE VIEWS (63% higher reach)
→ 1.5x MORE ENGAGEMENT (50% higher resonance)

```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Top cluster avg views | Stage 7 + Stage 6 + Stage 2 + Section 0.5.8 | **Function**: `calculate_proof_metrics_bucket_scoped(bucket_path, bucket_name,
formula_cluster_id)` (Section 0.5.8) → returns `top_cluster.avg_views`. **Process**: Load K-means clusters → get winning cluster video IDs → load
`selection_manifest.json` → get `videos_by_bucket[bucket_name].top_performers[]` → filter videos in cluster AND in bucket → average `playCount` → format with K/M
suffix. **Bucket-scoped**: Only includes videos from THIS bucket (e.g., 18-33s) | Integer (formatted with K/M) | 620K | ⚠️ **AWAITING STAGE 6 + STAGE 7** (Section)|
| Top cluster avg engagement | Stage 2 + Section 0.5.8 | **Function**: `calculate_proof_metrics_bucket_scoped(bucket_path, bucket_name, formula_cluster_id)` (Section
0.5.8) → returns `top_cluster.avg_engagement`. **Process**: For videos in winning cluster AND in this bucket: Map fields → apply `calculate_engagement_metrics()`
(Section 0.5.5) → average all rates. **Bucket-scoped**: Only 18-33s videos using pattern | Float (%) | 1.2 | ⚠️ **AWAITING STAGE 6 + STAGE 7** (Section 0.5.8) |
| Bottom cluster avg views | Stage 7 + Stage 6 + Stage 2 + Section 0.5.8 | **Function**: `calculate_proof_metrics_bucket_scoped(bucket_path, bucket_name,
formula_cluster_id)` (Section 0.5.8) → returns `bottom_cluster.avg_views`. **Process**: Load K-means clusters → get video IDs NOT in winning cluster → load
`selection_manifest.json` → get `videos_by_bucket[bucket_name].top_performers[]` → filter videos NOT in cluster AND in bucket → average `playCount` → format with K/M.
**Bucket-scoped**: Only 18-33s videos NOT using pattern | Integer (formatted with K/M) | 380K | ⚠️ **AWAITING STAGE 6 + STAGE 7** (Section 0.5.8) |
| Bottom cluster avg engagement | Stage 2 + Section 0.5.8 | **Function**: `calculate_proof_metrics_bucket_scoped(bucket_path, bucket_name, formula_cluster_id)`
(Section 0.5.8) → returns `bottom_cluster.avg_engagement`. **Process**: For videos NOT in winning cluster AND in this bucket: Map fields → apply
`calculate_engagement_metrics()` → average all rates. **Bucket-scoped**: Only 18-33s videos NOT using pattern | Float (%) | 0.8 | ⚠️ **AWAITING STAGE 6 + STAGE 7**
(Section 0.5.8) |
| View multiplier | Calculated | `Field #1 / Field #4` → Example: `620,000 / 380,000 = 1.6x` → Format as ratio with 1 decimal | Float (ratio) | 1.6x | ✅ **This session** |
| Engagement multiplier | Calculated | `Field #2 / Field #5` → Example: `1.2 / 0.8 = 1.5x` → Format as ratio with 1 decimal | Float (ratio) | 1.5x | ✅ **This session** |
| View percentage increase | Calculated | `((Field #1 - Field #4) / Field #4) × 100%` → Example: `((620K - 380K) / 380K) × 100% = 63%` → Round to integer | Integer (%) | 63 | ✅ **This session** |
| Engagement percentage increase | Calculated | `((Field #2 - Field #5) / Field #5) × 100%` → Example: `((1.2 - 0.8) / 0.8) × 100% = 50%` → Round to integer | Integer (%) | 50 | ✅ **This session** |

**Calculation Method** (Real Engagement from Apify Data):
```
Uses `calculate_engagement_metrics()` function (Section 0.5.5 in Stage8MVP.md) to calculate real engagement rates from actual TikTok interaction data:
- **Data Source**: `unified_analysis/{video_id}.json` → `metadata` (lines 8-12)
- **Formula**: `(likes + comments + shares + saves) / views × 100%`
- **Applied to**: All videos in top cluster (pattern users) and bottom cluster (non-pattern users)
- **Output**: Real measured engagement rates, not estimates

**Benefit**: Data integrity and transparency - shows actual engagement performance, not industry benchmark estimates.
```

---

### Page 2: "How to Execute" (Copy-Paste Implementation)

#### Freestyle Tips
For those who'd rather mix and match what works best with their style!

**Pick ONE from each category below:**

**MOST VIRAL CONTENT TYPES**
```
□ [Content Category 1]
   Description: [Auto-generated description]

□ [Content Category 2]
   Description: [Auto-generated description]

□ [Content Category 3]
   Description: [Auto-generated description]
```

**MOST USED ENGAGEMENT DRIVERS**
```
□ [Engagement Driver 1]
□ [Engagement Driver 2]
□ [Engagement Driver 3]
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Content Categories (Top 3) | Stage 7 | **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "content_category", n=3, "top")` (Section 0.5.1.1) → Returns array of top 3 category names | Array of strings | ["recipe_tutorial", "wellness_practice", "supplement_review"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| Content Category Descriptions | Stage 2.6 Taxonomy | **Function**: `get_descriptions_from_taxonomy(category_names, taxonomy_type)` (Section 0.5.1.2) → Read curated taxonomy file → Extract `description` for each category in Top 3 → Source: `/config/taxonomies/content_category.json` → Returns array of descriptions | Array of strings | ["Step-by-step instructional content...", "Health routines...", "Product review..."] | ✅ **FUNCTION READY** (Section 0.5.1.2) |
| Engagement Drivers (Top 3) | Stage 7 | **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "engagement_drivers", n=3, "top")` (Section 0.5.1.1) → Returns snake_case names → **Display Format**: Convert to title case (e.g., "before_after_reveal" → "Before/After Reveal") | Array of strings | ["personal_testimony", "before_after_reveal", "product_demonstration"] → Display: ["Personal Testimony", "Before/After Reveal", "Product Demonstration"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |

**Note**: All 3 bucket reports (33-60s, 60-90s, 90-120s) show the SAME Top 3 options - no rotation. Creators choose which fits their content best.

---

**⏱️ PHASE 1: HOOK (0-3 seconds)**

```
Strategy: Pick ONE from Top 3 below:

□ [Hook Strategy 1]
   Description: [Auto-generated description]

□ [Hook Strategy 2]
   Description: [Auto-generated description]

□ [Hook Strategy 3]
   Description: [Auto-generated description]
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Hook Strategies (Top 3) | Stage 7 | **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "hook_strategy", n=3, "top")` (Section 0.5.1.1) → Returns array of top 3 hook strategy names | Array of strings | ["question_hook", "problem_solution", "shocking_fact"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| Hook Strategy Descriptions | Stage 2.6 Taxonomy | **Function**: `get_descriptions_from_taxonomy(strategy_names, taxonomy_type)` (Section 0.5.1.2) → Read curated taxonomy file → Extract `description` for each strategy in Top 3 → Source: `/config/taxonomies/hook_strategy.json` → Returns array of descriptions | Array of strings | ["Opens with a question...", "Starts with problem...", "Begins with surprising..."] | ✅ **FUNCTION READY** (Section 0.5.1.2) |

---

**⏱️ PHASE 2: BUILD & PROVE (3s to last 3s - flexible timing)**

```
💡 Pick and choose what fits your content naturally:

Content Checklist:

Pain Points (address 1-2 from Top 5):
□ [Pain Point 1]
□ [Pain Point 2]
□ [Pain Point 3]
□ [Pain Point 4]
□ [Pain Point 5]

Keywords (mention 2-3 from Top 8):
□ [Keyword 1]
□ [Keyword 2]
□ [Keyword 3]
□ [Keyword 4]
□ [Keyword 5]
□ [Keyword 6]
□ [Keyword 7]
□ [Keyword 8]

Content Tactics (use 1-2 from Top 4):
□ [Tactic 1]
□ [Tactic 2]
□ [Tactic 3]
□ [Tactic 4]

Top Things to Do
□ [Supplementary Insight 1]
□ [Supplementary Insight 2]
□ [Supplementary Insight 3]
□ [Supplementary Insight 4]
□ [Supplementary Insight 5]

```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Pain Points (Top 5) | Stage 7 | **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "pain_points", n=5, "top")` (Section 0.5.1.1) → Returns array of top 5 pain point names | Array of strings | ["bloating", "low_energy", "weight_loss", "gut_health", "brain_fog"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| Keywords (Top 8) | Stage 7 | **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "keywords", n=8, "top")` (Section 0.5.1.1) → Returns array of top 8 keyword names | Array of strings | ["protein", "gut_health", "fiber", "probiotics", "metabolism", "holistic", "meal_prep", "supplements"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| Content Tactics (Top 4) | Stage 7 | **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "content_tactics", n=4, "top")` (Section 0.5.1.1) → Returns array of top 4 tactic names | Array of strings | ["direct_to_camera", "voiceover", "text_overlay", "product_showcase"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| Supplementary Insights (Top 5) | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `supplementary_insights.universal_principles` → Take first 5 items from array | Array of strings | ["middle_3_eye_contact_rate: 0.57 in top vs 0.43 in bottom (gap: 0.14)", "middle_1_energy_variance: 0.00 in top vs 0.00 in bottom (gap: 0.00)", "middle_3_energy_variance: 0.00 in top vs 0.00 in bottom (gap: 0.00)", "middle_3_energy_level: 0.10 in top vs 0.06 in bottom (gap: 0.04)", "hook_eye_contact_rate: 0.51 in top vs 0.63 in bottom (gap: 0.11)"] | ⚠️ **READY** (Stage 7 data exists) |

---

**⏱️ PHASE 3: CLOSING (Last 3 seconds)**

```
CTA: Pick ONE from Top 3 below:

□ [CTA Type 1]
   Description: [Auto-generated description]

□ [CTA Type 2]
   Description: [Auto-generated description]

□ [CTA Type 3]
   Description: [Auto-generated description]
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| CTA Types (Top 3) | Stage 7 | **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "caption_cta_type", n=3, "top")` (Section 0.5.1.1) → Returns array of top 3 CTA type names | Array of strings | ["link_in_bio", "save_post", "comment"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| CTA Type Descriptions | Stage 2.6 Taxonomy | **Function**: `get_descriptions_from_taxonomy(cta_names, taxonomy_type)` (Section 0.5.1.2) → Read curated taxonomy file → Extract `description` for each CTA in Top 3 → Source: `/config/taxonomies/caption_cta_type.json` → Returns array of descriptions | Array of strings | ["Direct viewers to link in bio", "Encourage saving post for later", "Ask viewers to comment"] | ✅ **FUNCTION READY** (Section 0.5.1.2) |

---



**CAPTION STRUCTURE** (Don't skip this!)

```
WINNING CAPTION PATTERNS:

1. Opening Hook (Top 3):
   • {{hook_type_1}} ({{hook_pct_1}}% of winning videos)
     Example: "Did you know that..."
   • {{hook_type_2}} ({{hook_pct_2}}% of winning videos)
     Example: "Here's why you need..."
   • {{hook_type_3}} ({{hook_pct_3}}% of winning videos)
     Example: "Try this simple..."

2. Call-to-Action (Top 3):
   • {{cta_type_1}} ({{cta_pct_1}}% of winning videos)
     Example: "Link in bio for details!"
   • {{cta_type_2}} ({{cta_pct_2}}% of winning videos)
     Example: "Save this for later!"
   • {{cta_type_3}} ({{cta_pct_3}}% of winning videos)
     Example: "Comment your experience!"

3. Hashtag Strategy:
   • Use {{avg_hashtag_count}} hashtags (average from winning cluster)
   • Place at end of caption for best performance

```

**Dynamic Fields**:
| # | Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|---|----------------|--------|------------------------|-----------|---------|-----------|
| 1 | Hook Type 1 | Stage 7 | **Function**: `get_top_n_from_field(aggregated, field="caption_hook_type", n=3)` → Returns top 3 hook types with percentages. **Source**: `aggregate_content_classifications()` (Section 0.5.1) → aggregated["caption_hook_type"] Counter | String | "question" | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 2 | Hook Type 1 % | Stage 7 | From Counter result: `(count / total) × 100` | Integer (%) | 45 | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 3 | Hook Type 2 | Stage 7 | Same as Field #1, 2nd most common | String | "statement" | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 4 | Hook Type 2 % | Stage 7 | Same as Field #2 | Integer (%) | 32 | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 5 | Hook Type 3 | Stage 7 | Same as Field #1, 3rd most common | String | "command" | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 6 | Hook Type 3 % | Stage 7 | Same as Field #2 | Integer (%) | 18 | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 7 | CTA Type 1 | Stage 7 | **Function**: `get_top_n_from_field(aggregated, field="caption_cta_type", n=3)` → Returns top 3 CTA types with percentages. **Source**: Same aggregation function | String | "link_in_bio" | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 8 | CTA Type 1 % | Stage 7 | From Counter result | Integer (%) | 67 | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 9 | CTA Type 2 | Stage 7 | Same as Field #7, 2nd most common | String | "save_post" | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 10 | CTA Type 2 % | Stage 7 | Same as Field #8 | Integer (%) | 21 | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 11 | CTA Type 3 | Stage 7 | Same as Field #7, 3rd most common | String | "comment" | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 12 | CTA Type 3 % | Stage 7 | Same as Field #8 | Integer (%) | 9 | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| 13 | Avg Hashtag Count | Stage 7 | **Function**: From `aggregate_content_classifications()` (Section 0.5.1) → `hashtag_count_stats["mean"]` → Round to nearest integer | Integer | 7 | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |

---

#### Ready Templates
Proven to work through our Machine Learning and analysis of over 500+ videos!

**Template 1**
Name: [Formula Name 1]
Hook: [Hook Step 1]
Middle: [Middle Step 1]
Closing: [Closing Step 1]

**Template 2**
Name: [Formula Name 2]
Hook: [Hook Step 2]
Middle: [Middle Step 2]
Closing: [Closing Step 2]

**Template 3**
Name: [Formula Name 3]
Hook: [Hook Step 3]
Middle: [Middle Step 3]
Closing: [Closing Step 3]

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Formula Name 1 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[0].formula_name` | String | "The Silent-to-Vocal Engagement Journey" | ⚠️ **READY** (Stage 7 data exists) |
| Hook Step 1 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[0].step_by_step_template[]` → Filter for line starting with "Hook" | String | "Hook (0-3s): Strong eye contact (0.77), prominent face presence (0.42), establish direct connection" | ⚠️ **READY** (Stage 7 data exists) |
| Middle Step 1 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[0].step_by_step_template[]` → Filter for first line starting with "Middle" | String | "Middle_1 (3-6s): Transition to pure visual storytelling (0.00 speech), let visuals speak" | ⚠️ **READY** (Stage 7 data exists) |
| Closing Step 1 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[0].step_by_step_template[]` → Filter for line starting with "Closing" | String | "Closing (23-26s): Visual-first silent closer, minimal verbal content (0.09), indirect gaze (0.19)" | ⚠️ **READY** (Stage 7 data exists) |
| Formula Name 2 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[1].formula_name` | String | "The Visual Storytelling Formula" | ⚠️ **READY** (Stage 7 data exists) |
| Hook Step 2 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[1].step_by_step_template[]` → Filter for line starting with "Hook" | String | "Hook: Use multiple visual angles or dynamic elements to create immediate visual interest" | ⚠️ **READY** (Stage 7 data exists) |
| Middle Step 2 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[1].step_by_step_template[]` → Filter for first line starting with "Middle" | String | "Middle: Maintain visual variety with strategic scene transitions and visual enhancements" | ⚠️ **READY** (Stage 7 data exists) |
| Closing Step 2 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[1].step_by_step_template[]` → Filter for line starting with "Closing" | String | "Closing: Return to primary visual focus while maintaining dynamic elements" | ⚠️ **READY** (Stage 7 data exists) |
| Formula Name 3 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[2].formula_name` | String | "The Vocal Variety Formula" | ⚠️ **READY** (Stage 7 data exists) |
| Hook Step 3 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[2].step_by_step_template[]` → Filter for line starting with "Hook" | String | "Hook: Establish vocal tone with clear articulation and moderate pacing" | ⚠️ **READY** (Stage 7 data exists) |
| Middle Step 3 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[2].step_by_step_template[]` → Filter for first line starting with "Middle" | String | "Middle: Use strategic pauses and vocal variety for emphasis and engagement" | ⚠️ **READY** (Stage 7 data exists) |
| Closing Step 3 | Stage 7 | `/ml_analysis/llm/winning_formulas.json` → `creative_reports[2].step_by_step_template[]` → Filter for line starting with "Closing" | String | "Closing: Maintain vocal energy while delivering clear call-to-action" | ⚠️ **READY** (Stage 7 data exists) |


**Phase 4: Examples**
[QR CODE]
Scan to watch: Top Performer Using this reports' patterns (620K views, 1.4% engagement)

[QR CODE]
Scan to watch: Bottom Performer - Don't Do This (95K views)

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|--------------|--------|------------------------|-----------|---------|-----------|
| QR Code (Top Performer) | Stage 2 + Section 0.5.2 | **Function**: `select_qr_code_videos(bucket_path, bucket_name)` (Section 0.5.2) → returns `top_performer.url` → generate QR code PNG. **Process**: Load `selection_manifest.json` → get `videos_by_bucket[bucket_name].top_performers[]` → load `selected_videos.json` → select max by `(playCount, createTime)` → use `webVideoUrl` | QR Code Image | https://www.tiktok.com/@agitthaaii/video/7545713916584774968 | ✅ **READY** (Section 0.5.2) |
| QR Code (Bottom Performer) | Stage 2 + Section 0.5.2 | **Function**: `select_qr_code_videos(bucket_path, bucket_name)` (Section 0.5.2) → returns `bottom_performer.url` → generate QR code PNG. **Process**: Load `selection_manifest.json` → get `videos_by_bucket[bucket_name].bottom_performers[]` → load `selected_videos.json` → select max by `(playCount, createTime)` → use `webVideoUrl` | QR Code Image | https://www.tiktok.com/@ahealthydoseofash/video/7560886598309612814 | ✅ **READY** (Section 0.5.2) |
| Bottom video views | Stage 2 | From selected bottom QR video (Field #3): `playCount` → format with K/M suffix | Integer (formatted with K/M) | 95K | ✅ **This session** |
| Example video views | Stage 2 | From selected QR video (Field #11): `playCount` → format with K/M suffix | Integer (formatted with K/M) | 620K | ✅ **This session** |
| Example video engagement | Stage 2 + Function | From selected QR video (Field #11): Map fields (`playCount`, `diggCount`, `commentCount`, `shareCount`, `collectCount`) → apply `calculate_engagement_metrics()` | Float (%) | 1.4 | ✅ **This session** |

**Design Decisions**
QR codes at the end - Maybe creators look at QR code, get distracted and dont finish studying report.
QR Codes from top performers of the video bucket duration


---

### QR Code Implementation

**Decision**: Each creator report includes **2 QR codes** linking to real TikTok video examples

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

## 3. Handle/Single Competitor → Client (Deep Dive Report)


**Status**: 🔄 **IN PROGRESS** - Redesigned to match Report 4 structure
```
**Audience**: Tumi Labs clients (business owners)

**Purpose**: Deep dive competitive intelligence on 1 competitor

**Deliverable**: 1 PDF analyzing 1 competitor

**Format**: 3-page PDF (desktop-optimized, executive-focused)

**Reading Time**: 6-8 minutes (scannable in 2 minutes)

**Design Philosophy**: Simplified version of Report 4 (Multi-Competitor) adapted for single competitor analysis
```

### Page 1: Executive Overview

**Purpose**: Establish analysis scope and show what competitor creates

---

#### Header Section

```
Deep Dive: @drinkpoppi
Analysis Period: Last 90 days
Videos Analyzed: 127
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handle | Config | `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/config.json` → `target` (includes @ symbol) | String | "@drinkpoppi" | ✅ **This session** |
| Analysis period | Static | Fixed string: "Last 90 days" | String | "Last 90 days" | ✅ **This session** |
| Videos analyzed | Function | `calculate_videos_analyzed(client_id, competitor_handle)` (Stage8MVP.md:1352) → Sums `selected_count` from winning buckets' `selected_videos.json` files | Integer | 127 | ✅ **This session** |

---

### Page 2: Content Strategy Analysis

**Purpose**: Show where competitor focuses content and how it performs

---

#### Section 1: Duration Distribution

```
WHERE @DRINKPOPPI FOCUSES CONTENT:

[Horizontal bar chart showing % of videos per bucket]

0-3s:   ██ 3%
3-9s:   ████ 8%
9-13s:  ████████ 12%
13-18s: ████████████ 18%
18-33s: ████████████████████ 32%  ← PRIMARY FOCUS
33-60s: ██████████████ 22%
60-90s: ███ 4%
90-120s: █ 1%

```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handle (section title) | Config | `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/config.json` → `target` | String | "@drinkpoppi" | ✅ **This session** |
| % per bucket (8 rows) | Function | `calculate_bucket_distribution(winner_analysis_path)` (Stage8MVP.md:1650) → Reads `top_100_distribution` from `winner_analysis.json` and calculates percentages for all 8 buckets | Integer (%) | 8, 36, 18, 5, 15, 11, 6, 0 | ✅ **This session** |
| Primary focus bucket | Calculated | Bucket with highest percentage from `calculate_bucket_distribution()` output → Simple: `max(bucket_percentages, key=bucket_percentages.get)` | String | "3-9s" | ✅ **This session** |
| Competitor handle (Key Insight) | Config | Same as above | String | "@drinkpoppi" | ✅ **This session** |

---

#### Section 2: Performance by Duration

```
WHAT PERFORMS BEST FOR @DRINKPOPPI:

Top 3 Performing Durations:

Duration | Avg Views | Avg Engagement | Rating
---------|-----------|----------------|--------
18-33s   | 620K      | 1.5%           | ⭐⭐⭐⭐⭐  ← SWEET SPOT
13-18s   | 580K      | 1.3%           | ⭐⭐⭐⭐
33-60s   | 490K      | 1.4%           | ⭐⭐⭐⭐

Sweet Spot: 18-33s (highest views + highest engagement + high volume)

These 3 durations represent 72% of @drinkpoppi's content output.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handle (section title) | Config | `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/config.json` → `target` | String | "@drinkpoppi" | ✅ **This session** |
| Top 3 buckets | Winner Analysis | `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets` array | Array[String] | ["3-9s", "9-13s", "18-33s"] | ✅ **This session** |
| Avg views per bucket (3 rows) | Function | `calculate_competitor_bucket_avg_views(client_id, competitor_handle, bucket_name)` (Stage8MVP.md:2029) → format with K/M suffix | Integer (formatted) | 620K, 580K, 490K | ✅ **This session** |
| Avg engagement per bucket (3 rows) | Function | `calculate_competitor_bucket_avg_engagement(client_id, competitor_handle, bucket_name)` (Stage8MVP.md:2162) | Float (%) | 1.5, 1.3, 1.4 | ✅ **This session** |
| Star ratings (3 rows) | Inline Calculation | `rank_competitor_top_buckets()` (Stage8MVP.md:3715) → Returns list with `stars` field per bucket | String (emoji) | ⭐⭐⭐⭐⭐, ⭐⭐⭐⭐, ⭐⭐⭐⭐ | ✅ **This session** |
| Sweet spot bucket | Inline Calculation | `rank_competitor_top_buckets()` (Stage8MVP.md:3715) → Get bucket where `is_sweet_spot == True` | String | "3-9s" | ✅ **This session** |
| Coverage percentage | Inline Calculation | `calculate_top_3_coverage(bucket_percentages, top_3_buckets)` (Stage8MVP.md:3843) → Sum percentages of top 3 buckets | Integer (%) | 69 | ✅ **This session** |
| Competitor handle (closing line) | Config | Same as above | String | "@drinkpoppi" | ✅ **This session** |

---

#### Section 3: Posting Frequency & Consistency

```
POSTING ACTIVITY:
@drinkpoppi posts 14 videos per week on average
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handle | Config | `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/config.json` → `target` | String | "@drinkpoppi" | ✅ **This session** |
| Posting frequency | Function | `calculate_posting_frequency(client_id, competitor_handle)` (Stage8MVP.md:1249) → Sums videos from `top_100_distribution` / weeks in analysis period | Float | 7.6 videos/week | ✅ **This session** |

---

### Page 3: Creative Intelligence & Patterns

**Purpose**: Show what and how competitor creates content

---

#### Section 1: Content DNA (What They Make)

**Purpose**: Show content types and engagement drivers

```
WHAT @DRINKPOPPI CREATES:

Top Content Categories (across all durations):
1. Recipe Tutorial (38% of content) - Step-by-step cooking instructions
2. Wellness Practice (28% of content) - Daily health routines and habits
3. Supplement Review (17% of content) - Product recommendations and reviews
4. Expert Interview (12% of content) - Professional perspectives
5. Personal Testimony (5% of content) - Personal success stories

Top Engagement Drivers:
• Before/After Reveal (45% of videos) - Visual transformations
• Specific Metrics (42% of videos) - "Lost 15 lbs in 30 days"
• Personal Testimony (38% of videos) - "This worked for me..."
• Expert Credentials (28% of videos) - "Registered nutritionist here..."

```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handle | Config | `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/config.json` → `target` | String | "@drinkpoppi" | ✅ **This session** |
| Top content categories (5 types) | Stage 2.7 | Aggregate across all buckets: **Base Function**: `aggregate_content_classifications()` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(field="content_category", n=5)` (Section 0.5.1.1) | Array[String] with % | ["Recipe Tutorial (38%)", ...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| Content category descriptions | Stage 2.6 | From taxonomy: `get_descriptions_from_taxonomy(categories)` | Array[String] | ["Step-by-step cooking instructions", ...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.6 TAXONOMY** |
| Top engagement drivers (4 items) | Stage 2.7 | Aggregate across all buckets: **Base Function**: `aggregate_content_classifications()` → **Wrapper**: `get_top_n_from_field(field="engagement_drivers", n=4)` | Array[String] with % | ["Before/After Reveal (45%)", ...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| Engagement driver descriptions | Inline Calculation | `format_engagement_driver_description()` (Stage8MVP.md:3882) → Maps engagement drivers to human-readable descriptions | Array[String] | ["Visual transformations", '"Lost 15 lbs in 30 days"', ...] | ✅ **This session** |

---

#### Section 2: Execution Playbook (How They Make It)

**Purpose**: Show hook strategies, pain points, keywords, and content tactics

```
HOW @DRINKPOPPI EXECUTES:

Top Hook Strategies:
1. Question Hook (42% of videos) - Opens with engaging question
2. Problem-Solution (31% of videos) - Identifies pain point, offers solution
3. Direct Statement (18% of videos) - Bold claim or fact
4. Curiosity Gap (9% of videos) - Creates mystery or intrigue

Top CTA Strategies:
1. Link in Bio (38% of videos) - Directs viewers to profile link
2. Follow for More (32% of videos) - Encourages account following
3. Save This Post (21% of videos) - Prompts content bookmarking
4. Tag a Friend (9% of videos) - Drives viral sharing

Pain Points Addressed:
• Bloating/Digestive Issues (48% of videos)
• Low Energy/Fatigue (42% of videos)
• Weight Management (38% of videos)
• Inflammation (32% of videos)
• Gut Health (28% of videos)

Top Keywords:
#guthealth, #protein, #antiinflammatory, #metabolism, #fiber

Content Tactics:
• Direct-to-Camera (52% of videos)
• Voiceover + B-roll (31% of videos)
• Text-Heavy Overlays (24% of videos)
• Product Demonstration (18% of videos)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handle (section title) | Config | Same as Section 1 | String | "@drinkpoppi" | ✅ **This session** |
| Top hook strategies (4 types) | Stage 2.7 | Aggregate: **Base Function**: `aggregate_content_classifications()` → **Wrapper**: `get_top_n_from_field(field="hook_strategy", n=4)` | Array[String] with % | ["Question Hook (42%)", ...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** Section 0.5.1.1 (line 413)|
| Hook strategy descriptions | Function | `get_descriptions_from_taxonomy(hook_names, "hook_strategy")` (Stage8MVP.md:527) → Looks up descriptions from Stage 2.6 taxonomy files | Array[String] | ["Opens with engaging question", ...] | ✅ **This session** |
| Top CTA strategies (4 types) | Stage 2.7 | Aggregate across all buckets: **Base Function**: `aggregate_content_classifications()` (Section 0.5.1) → Extract `caption_analysis.cta_type` Counter → Get top 4 most common with percentages | Array[String] with % | ["Link in Bio (38%)", "Follow for More (32%)", ...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| CTA strategy descriptions | Function | `get_descriptions_from_taxonomy(cta_names, "cta_strategy")` (Stage8MVP.md:527) → Looks up descriptions from Stage 2.6 taxonomy files | Array[String] | ["Directs viewers to profile link", "Encourages account following", ...] | ✅ **This session** |
| Top pain points (5 items) | Stage 2.7 | Aggregate: **Base Function**: `aggregate_content_classifications()` → **Wrapper**: `get_top_n_from_field(field="pain_points", n=5)` | Array[String] with % | ["Bloating/Digestive Issues (48%)", ...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| Top keywords (5 items) | Stage 2.7 | Aggregate: **Base Function**: `aggregate_content_classifications()` → **Wrapper**: `get_top_n_from_field(field="keywords", n=5)` | Array[String] | ["#guthealth", "#protein", ...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
| Content tactics (4 items) | Stage 2.7 | Aggregate: **Base Function**: `aggregate_content_classifications()` → **Wrapper**: `get_top_n_from_field(field="content_tactics", n=4)` | Array[String] with % | ["Direct-to-Camera (52%)", ...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |

---

#### Section 3: Hashtag Strategy

**Purpose**: Show hashtag usage patterns

```
HASHTAG STRATEGY:

Top 10 Hashtags:
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

Total unique hashtags: 28
Average hashtags per video: 9
Strategy Type: Diversified (28 hashtags across content)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Top 10 hashtags | Function | `extract_hashtag_analysis()` (Stage8MVP.md Section 0.5.3) → Returns `top_10` with hashtag names | Array[String] | ["#nutrition", "#healthylifestyle", ...] | ✅ **This session** |
| Hashtag usage % (10 values) | Function | `extract_hashtag_analysis()` (Stage8MVP.md Section 0.5.3) → Returns `top_10` with percentages | Array[Integer] (%) | [82, 68, 54, 47, 43, 38, 32, 28, 24, 21] | ✅ **This session** |
| Total unique hashtags | Function | `extract_hashtag_analysis()` (Stage8MVP.md Section 0.5.3) → Returns `total_unique_hashtags` | Integer | 28 | ✅ **This session** |
| Avg hashtags per video | Function | `extract_hashtag_analysis()` (Stage8MVP.md Section 0.5.3) → Returns `avg_hashtags_per_video` | Integer | 9 | ✅ **This session** |
| Strategy type | Inline Calculation | `determine_hashtag_strategy_type(total_unique_hashtags)` (Stage8MVP.md:4052) → Returns "Diversified" if > 20, else "Focused" | String | "Diversified" | ✅ **This session** |

---

#### Section 4: Caption Strategy

**Purpose**: Show caption formatting and CTA patterns

```
CAPTION STRATEGY:

Avg Hashtag Count:        12 hashtags per video
Top CTA Type:             "Follow me" (52% of videos)

```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Avg hashtag count | Function | `aggregate_content_classifications()` (Stage8MVP.md:169) → Returns `hashtag_count_stats['mean']` from aggregated `caption_analysis.hashtag_count` | Integer | 12 | ✅ **This session** |
| Top CTA type | Function | `aggregate_content_classifications()` (Stage8MVP.md:169) → Returns most common from `caption_cta_type` Counter + calculate percentage | String with % | "Follow me (52%)" | ✅ **This session** |

---

#### Section 5: Content Sourcing Strategy

**Purpose**: Identify affiliate partnerships and content sourcing

```
CONTENT SOURCING:

Original Content: 58% (no affiliate mentions)
Reposted/Affiliate Content: 42% (contains @mentions or repost indicators)

Top Affiliate Contributors:
1. @fitnessguru123     (18% of videos - 54 mentions)
2. @healthcoach_jane   (12% of videos - 36 mentions)
3. @nutritionpro       (8% of videos - 24 mentions)
4. @wellnesswarrior    (5% of videos - 15 mentions)
5. @cleaneatingclub    (4% of videos - 12 mentions)

Total unique @mentions: 47

```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Original content % | Inline Calculation | `calculate_original_content_percentage(repost_rate)` (Stage8MVP.md:4105) → Returns `100 - repost_rate` | Integer (%) | 58 | ✅ **This session** |
| Reposted/Affiliate % | Function | `extract_mention_analysis()` (Stage8MVP.md Section 0.5.4) → Returns `repost_rate` | Integer (%) | 42 | ✅ **This session** |
| Top affiliates (5 items) | Function | `extract_mention_analysis()` (Stage8MVP.md Section 0.5.4) → Returns `top_10_mentions` (take first 5) | Array[Object] | [{handle: "@fitnessguru123", percentage: 18, count: 54}, ...] | ✅ **This session** |
| Total unique mentions | Function | `extract_mention_analysis()` (Stage8MVP.md Section 0.5.4) → Returns `total_unique_mentions` | Integer | 47 | ✅ **This session** |
| Videos with mentions | Function | `extract_mention_analysis()` (Stage8MVP.md Section 0.5.4) → Returns `videos_with_mentions` | Integer | 126 | ✅ **This session** |
| Total videos | Function | `calculate_videos_analyzed()` (Stage8MVP.md:1352) | Integer | 87 | ✅ **This session** |
| Mention rate | Function | `extract_mention_analysis()` (Stage8MVP.md Section 0.5.4) → Returns `mention_rate` | Integer (%) | 42 | ✅ **This session** |

---

#### Section 6: Creative Formulas

**Duration Bucket [BUCKET_1_NAME]:**
  • Formula 1: [BUCKET_1_FORMULA_1_NAME]
  • Formula 2: [BUCKET_1_FORMULA_2_NAME]
  • Formula 3: [BUCKET_1_FORMULA_3_NAME]

**Duration Bucket [BUCKET_2_NAME]:**
  • Formula 4: [BUCKET_2_FORMULA_1_NAME]
  • Formula 5: [BUCKET_2_FORMULA_2_NAME]
  • Formula 6: [BUCKET_2_FORMULA_3_NAME]
  
**Duration Bucket [BUCKET_3_NAME]:**
  • Formula 7: [BUCKET_3_FORMULA_1_NAME]
  • Formula 8: [BUCKET_3_FORMULA_2_NAME]
  • Formula 9: [BUCKET_3_FORMULA_3_NAME]


**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Duration Bucket names (3 buckets) | Stage 1 | `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets` array | Array[String] | ["18-33s", "13-18s", "60-90s"] | ✅ **Verified** |
| Formula names per bucket (9 total) | Stage 7 | For each winning bucket in `top_3_buckets`: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/buckets/bucket_{bucket_name}/ml_analysis/llm/winning_formulas.json` → `creative_reports[0-2].formula_name` → Extract 3 formula names per bucket | Array[String] per bucket | Bucket 1: ["The Silent-to-Vocal Engagement Journey", "The Visual Storytelling Formula", "The Vocal Variety Formula"] | ✅ **Verified** |

**Notes**:
- Bucket names are dynamically populated from `winner_analysis.json` (same source as Section 3)
- Each bucket always has exactly 3 formulas (guaranteed by Stage 7 output schema)
- Formula names are LLM-generated and unique per bucket/hashtag combination
- Formula naming convention typically follows: "The [Pattern] Formula" or "The [Pattern] Journey"


#### Section 7: Videos

**Purpose**: Provide visual proof of top-performing videos per competitor
(BRAINSTORM) - SAMPLE BELOW. Need to update!

```
TOP PERFORMING VIDEOS BY COMPETITOR:

@wellness_pro (Market Leader):
[QR CODE - 1" x 1"]

📹 Video Stats: 820K views | 1.5% engagement | 45s duration
Formula: "The Transformation Journey" (33-60s bucket)
Hashtags: #wellness #guthealth #transformation #healthylifestyle

Key Pattern Elements:
• Before/after reveal in first 5 seconds (immediate hook)
• Personal testimony with specific metrics ("Lost 15 lbs in 30 days")
• Expert credentials mentioned (registered nutritionist)
• Product shown at mid-video (second 22)
• 8 text overlays throughout (retention tactic)


```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| QR codes

**Video Selection Criteria**:
- **Priority 1**: Highest view count from winning bucket per competitor
- **Priority 2**: Newest video (if multiple high performers - reduces deletion risk)
- **Priority 3**: Videos from accounts with 100K+ followers (stability)

### Summary of Report 3 Structure

**3-Page Report** (streamlined from original 4 pages):

**Page 1: Executive Overview**
- Header Section (competitor name, analysis period, videos analyzed)
- Analysis Scope (methodology description)

**Page 2: Content Strategy Analysis**
- Section 1: Duration Distribution (where they focus)
- Section 2: Performance by Duration (what performs best)
- Section 3: Posting Frequency (simplified one-line)

**Page 3: Creative Intelligence & Patterns**
- Section 1: Content DNA (what they make)
- Section 2: Execution Playbook (how they make it)
- Section 3: Hashtag Strategy
- Section 4: Caption Strategy
- Section 5: Content Sourcing Strategy

---

### Data Extraction Requirements

Same functions as Report 4, but applied to single competitor:
- `extract_hashtag_analysis()` (Section 0.5.3)
- `extract_mention_analysis()` (Section 0.5.4)
- `aggregate_content_classifications()` (Section 0.5.1)
- `get_top_n_from_field()` (Section 0.5.1.1)
- `calculate_avg_views_per_bucket()`
- `calculate_avg_engagement_per_bucket()`

---

**Status**: ✅ **COMPLETE** - Report 3 redesigned as simplified version of Report 4 for single competitor

---
## 4. Handle/Multiple Competitor → Client (Market Intelligence Report)
```
**Status**: ✅ **COMPLETE**

**Audience**: Tumi Labs clients (business owners)

**Purpose**: Multi-competitor market intelligence - understand competitive landscape

**Deliverable**: 1 PDF analyzing 2-5 competitors (market intelligence only, no client comparison)

**Format**: 4-page PDF (desktop-optimized, executive-focused)

**Reading Time**: 10-12 minutes (scannable in 3 minutes)
```
---

### Page 1: Market Overview & Performance Rankings

**Purpose**: Show competitive landscape scale and performance rankings

---

#### Header Section

```
Market Intelligence Report
Competitors Analyzed: @rival_brand, @wellness_pro, @fitness_guru
Analysis Period: Last 90 days
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handles (2-5) | Config | Array of `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/config.json` → `target` fields (one per competitor) | String (array) | ["@drinkpoppi", "@nike", "@wellness_pro"] | ✅ **This session** |
| Analysis period | Static | Fixed string: "Last 90 days" | String | "Last 90 days" | ✅ **Report 1 Header** |

---

#### Performance Rankings

```
COMPETITOR PERFORMANCE RANKING:

Rank | Handle            | Avg Views | Avg Engagement | Posting Freq | Videos Analyzed
-----|-------------------|-----------|----------------|--------------|----------------
1    | @wellness_pro     | 580K      | 1.4%           | 16/week      | 145
2    | @rival_brand      | 520K      | 1.3%           | 14/week      | 127
3    | @fitness_guru     | 480K      | 1.2%           | 11/week      | 98

Market Leader: @wellness_pro (580K avg views, 1.4% engagement, highest posting frequency)
```

**Dynamic Fields**:
| # | Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|---|----------------|--------|------------------------|-----------|---------|-----------|
| 1 | Competitor handles (all) | Directory Structure | Directory names under `/data/clients/{client}/competitors/` → Add @ prefix for display. Actual example: `["hankandroy", "nike", "vitalproteins"]` → Display: `["@hankandroy", "@nike", "@vitalproteins"]` | String (array) | ["@hankandroy", "@nike", "@vitalproteins"] | ✅ **This session** |
| 2 | Avg Views (per competitor) | Winner Analysis + Selected Videos | Per competitor: Calculate avg `playCount` per winning bucket from `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/buckets/bucket_{name}/selected_videos.json` → Use first `top_count` videos only → Weighted average across 3 buckets. Formula: `Σ(bucket_avg × top_count) / Σ(top_count)` | Integer (K/M formatted) | 580K, 520K, 480K | ✅ **This session** |
| 3 | Posting Freq (per competitor) | Winner Analysis + Config | Per competitor: `winner_analysis.json → sum(top_100_distribution.values())` ÷ `(config.json → date_filter days ÷ 7)`. Verified example: Drinkpoppi = 98 videos / 13 weeks = 7.5 videos/week. Confidence: "exact" if sum < 100, "minimum" if sum = 100 | Float | 7.5, 11.2, 14.0 videos/week | ✅ **This session** |
| 4 | Videos Analyzed (per competitor) | Selected Videos (All Comp) | Per competitor: Sum `selected_count` across all winning buckets from `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/buckets/bucket_{name}/selected_videos.json` (typically 3 buckets) | Integer | 145, 127, 98 | ✅ **Report 1 Header** |
| 5 | Avg Engagement (per competitor) | Unified Analysis (All Comp) | Per competitor: For each winning bucket → load top performer video IDs from `selected_videos.json` → for each video, load `/buckets/{bucket}/analysis/unified_analysis/{video_id}.json → metadata` → call `calculate_engagement_metrics()` (Section 0.5.5) → average across all top performers from 3 buckets. Formula: `(likes + comments + shares + saves) / views × 100%`. Fields used: `metadata.{playCount, diggCount, commentCount, shareCount, collectCount}`. **Scope limitation**: Top performers only, not all posted videos | Float (%) | 1.4%, 1.3%, 1.2% | ✅ **This session** |
| 6 | Market leader | Calculated | Competitor with highest composite score. Formula: `(avg_views / max_views × 100) + avg_engagement`. Logic: Same as Report 1 "← BEST" bucket selection (highest engagement + views). Normalize views to 0-100 scale, add to engagement %. Highest score wins | String | "@nike" | ✅ **This session** |

---

#### Analysis Scope

```
ANALYSIS SCOPE PER COMPETITOR:

@wellness_pro:
• Videos Analyzed: 145
• Duration Range: 0-120 seconds (8 buckets)
• Content Elements Tracked: 60+ features per video

@rival_brand:
• Videos Analyzed: 127
• Duration Range: 0-120 seconds (8 buckets)
• Content Elements Tracked: 60+ features per video

@fitness_guru:
• Videos Analyzed: 98
• Duration Range: 0-120 seconds (8 buckets)
• Content Elements Tracked: 60+ features per video

Analysis Method:
Multi-dimensional machine learning and AI content analysis applied to each
competitor's content to identify patterns, strategies, and creative formulas.
```

**Dynamic Fields**:
| # | Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|---|----------------|--------|------------------------|-----------|---------|-----------|
| 1 | Videos analyzed (per competitor) | Selected Videos (All Comp) | Per competitor: Sum `selected_count` across winning buckets. Uses `calculate_videos_analyzed()` (Stage8MVP.md Section 0.5.6) | Integer | 145, 127, 98 | ✅ **Performance Rankings Field #4** |

---

### Page 2: Content Strategy Comparison

**Purpose**: Show where each competitor focuses content and what performs best

---

#### Section 1: Bucket Distribution Comparison

```
WHERE EACH COMPETITOR FOCUSES CONTENT:

Duration   | @wellness_pro | @rival_brand | @fitness_guru | Market Pattern
-----------|---------------|--------------|---------------|----------------
0-3s       | 2%            | 3%           | 5%            | Low volume
3-9s       | 5%            | 8%           | 10%           | Low volume
9-13s      | 8%            | 12%          | 14%           | Growing volume
13-18s     | 15%           | 18%          | 22%           | Moderate volume
18-33s     | 28% 🟢        | 32% 🟢       | 26% 🟢        | HIGH VOLUME
33-60s     | 30% 🟢        | 22% 🟢       | 18%           | High volume
60-90s     | 9%            | 4%           | 4%            | Low volume
90-120s    | 3%            | 1%           | 1%            | Very low volume

🟢 = High volume focus (>20% of content)
```

**Dynamic Fields**:
| # | Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|---|----------------|--------|------------------------|-----------|---------|-----------|
| 1 | Bucket % (per competitor × 8 buckets) | Winner Analysis | Per competitor: `winner_analysis.json → top_100_distribution` → Calculate `(count / total) × 100` for each of 8 buckets. Returns 8 percentages. Uses `calculate_bucket_distribution()` (Stage8MVP.md Section 0.5.6 Function 6). Verified: Drinkpoppi = `{0-3s: 8%, 3-9s: 36%, 9-13s: 18%, 13-18s: 5%, 18-33s: 15%, 33-60s: 11%, 60-90s: 6%, 90-120s: 0%}` | Integer (%) per bucket | 2%, 5%, 8%, 15%, 28%, 30%, 9%, 3% | ✅ **This session** |
| 2 | High volume markers (per competitor) | Calculated | Per competitor: For each bucket from Field #1, flag `True` if `percentage > 20`, else `False`. Display: append " 🟢" if True. Uses `calculate_high_volume_markers()` (Section 0.5.6 Function 7). Threshold: 20% = 1.6x uniform distribution. Verified: Drinkpoppi 3-9s = True (36% > 20%) | Boolean per bucket | True/False per cell | ✅ **This session** |
| 3 | Market pattern (per bucket) | Calculated | Aggregate: For each bucket, calculate average % across all competitors → categorize: ≥25% = "HIGH VOLUME", 20-24% = "High volume", 15-19% = "Moderate volume", 10-14% = "Growing volume", <10% = "Low volume". Uses `calculate_market_patterns()` (Section 0.5.6 Function 8). Verified with template examples | String per bucket | "Low volume", "HIGH VOLUME", etc. | ✅ **This session** |


---

#### Section 2: Performance by Duration

```
PERFORMANCE BY COMPETITOR (VIEWS / ENGAGEMENT):
Shows each competitor's top 3 performing durations (winning buckets)

Duration   | @wellness_pro (V/E) | @rival_brand (V/E) | @fitness_guru (V/E) | Best Performer
-----------|---------------------|-------------------|--------------------|-----------------
9-13s      | 420K / 1.2% 👑      | —                 | —                  | @wellness_pro
13-18s     | 580K / 1.3% 👑      | 580K / 1.4% 👑    | 490K / 1.2% 👑     | @rival_brand (engagement wins tie)
18-33s     | 620K / 1.5% 👑      | 620K / 1.4% 👑    | 510K / 1.3% 👑     | @wellness_pro (engagement wins tie)
33-60s     | 590K / 1.4% 👑      | 490K / 1.3% 👑    | 450K / 1.2% 👑     | @wellness_pro (views + engagement)
60-90s     | —                   | —                 | 320K / 1.1% 👑     | @fitness_guru

👑 = Winning bucket for this competitor (top 3 performing durations)
— = Not a winning bucket (engagement data not available)

COMPETITOR WINNING BUCKETS:
• @wellness_pro: 9-13s, 13-18s, 18-33s, 33-60s (showing top 3 only)
• @rival_brand: 13-18s, 18-33s, 33-60s
• @fitness_guru: 13-18s, 18-33s, 60-90s

Note: Each competitor analyzed independently - only their top 3 performing buckets shown
(Option B: Data availability constraint - can only show engagement for winning buckets)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Unique buckets (union of all winning buckets) | Stage 1 (All Comp) | Union of `top_3_buckets` across all competitors. Per competitor: load `/data/clients/{client}/competitors/{target}/{analysis_dir}/winner_analysis.json` → `top_3_buckets` → combine all arrays → deduplicate → sort by duration order. Uses `get_unique_winning_buckets(client_id, competitors)` (Stage8MVP.md Function 9) | String (array) | ["9-13s", "13-18s", "18-33s", "33-60s", "60-90s"] | ✅ **This session** |
| Avg views (per competitor, winning buckets only) | Stage 1 (All Comp) | Per competitor, per bucket: Check if bucket in `top_3_buckets` (if not, return None → display "—"). If yes, load `/buckets/bucket_{name}/selected_videos.json` → first `top_count` videos → calculate `sum(playCount) / top_count`. Uses `calculate_competitor_bucket_avg_views(client_id, competitor_handle, bucket_name)` (Stage8MVP.md Function 10) | Integer (formatted with K) or "—" | 420K, 580K, "—", etc. | ✅ **This session** |
| Avg engagement (per competitor, winning buckets only) | Unified Analysis (All Comp) | Per competitor, per bucket: Check if bucket in `top_3_buckets` (if not, return None → display "—"). If yes, load top performers from `selected_videos.json` → for each video, load `unified_analysis/{video_id}.json → metadata` → call `calculate_engagement_metrics(metadata)` (Section 0.5.5) → average all engagement rates. Formula: `(likes + comments + shares + saves) / views × 100%`. Uses `calculate_competitor_bucket_avg_engagement(client_id, competitor_handle, bucket_name)` (Stage8MVP.md Function 11) | Float (%) or "—" | 1.2%, 1.3%, "—", etc. | ✅ **This session** |
| Winning bucket markers (per competitor) | Stage 1 (All Comp) | Per competitor, per bucket: Check if `bucket_name in top_3_buckets`. Uses `is_winning_bucket(client_id, competitor_handle, bucket_name)` (Stage8MVP.md Function 13). Display: `"👑" if is_winning_bucket() returns True else ""` | String (emoji) or blank | 👑 or blank | ✅ **This session** |
| Best performer (per bucket) | Calculated | Per bucket: Collect all competitors who have this bucket in `top_3_buckets` → for each, get avg views (Function 10) and avg engagement (Function 11) → calculate composite score: `(views / max_views × 100) + engagement` → return competitor with highest score. Tie-breaking: If composite tied, winner determined by engagement. Display includes tie notation if needed. Uses `calculate_bucket_best_performer(client_id, competitors, bucket_name)` (Stage8MVP.md Function 12) | String per bucket | "@wellness_pro", "@rival_brand (engagement wins tie)", "—" | ✅ **This session** |
| Competitor winning bucket lists (3 per competitor) | Stage 1 (All Comp) | Per competitor: Get list of top 3 winning buckets. Uses `get_competitor_winning_buckets(client_id, competitor_handle)` (Stage8MVP.md Function 14) → returns `top_3_buckets` array from `winner_analysis.json`. Display format: Comma-separated list (e.g., "9-13s, 13-18s, 18-33s") or bullet list per competitor | String (array per competitor) | ["9-13s", "13-18s", "18-33s"] | ✅ **This session** |

**Field Removed**:
- ~~**Performance insights (4 items)**~~: Removed due to same issues as "Key Insights" in Performance Rankings section - requires natural language generation, assumes consensus patterns exist, may fail with diverse competitor strategies. All relevant data is already visible in the Performance by Duration table above.

**Option B Implementation Note**: Each competitor is analyzed independently and has their own top 3 winning buckets. The table shows the UNION of all winning buckets (typically 3-6 unique buckets), with "—" for competitors who don't have a particular bucket in their winning 3.

---

#### Section 3: Posting Frequency & Consistency

```
POSTING ACTIVITY (Last 90 Days):

Competitor        | Posting Freq
------------------|-------------
@wellness_pro     | 16/week
@rival_brand      | 14/week
@fitness_guru     | 11/week

MARKET AVERAGE:
• Market average: 13.7 videos/week
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Posting freq (per competitor) | Winner Analysis + Config (All Comp) | Per competitor: Sum `top_100_distribution.values()` from `winner_analysis.json` → divide by weeks from `config.json → date_filter`. Uses `calculate_posting_frequency(client_id, competitor_handle)` (Stage8MVP.md Function 2 - updated with dynamic path discovery) | Float | 16, 14, 11 | ✅ **This session** |
| Market average | Calculated | Calculate mean of all competitor posting frequencies: `sum(frequencies) / len(competitors)`. **Inline logic** (no function needed - simple average calculation). Example: (16 + 14 + 11) / 3 = 13.7 | Float | 13.7 | ✅ **This session** |

**Fields Removed**:
- ~~**Consistency (per competitor)**~~: Removed - requires weekly variance analysis of video timestamps which adds complexity without significant value. Posting frequency already provides sufficient insight into competitor activity levels.
- ~~**Recent velocity (per competitor)**~~: Removed - requires parsing individual video timestamps to filter last 30 days. The Trend field already captures whether activity is accelerating/stable/declining without needing the exact velocity number.
- ~~**Trend (per competitor)**~~: Removed - depends on recent velocity field (which was removed). Calculating trend requires comparing different time windows which needs timestamp parsing. Posting frequency provides sufficient activity insight without trend analysis.

---

### Page 3: Creative Intelligence & Patterns

Purpose: Show winning content patterns and execution tactics per bucket


#### Section 1: Content DNA (What They Make)

Purpose: Show winning content types, engagement drivers, and production tactics by bucket

TOP CONTENT CATEGORIES
```
@wellness_pro

  📊 33-60s Bucket (620K avg views)
  1. Content Category 1
  2. Content Category 2

  📊 60-90s Bucket (580K avg views)
  1. Content Category 1
  2. Content Category 2

  📊 90-120s Bucket (540K avg views)
  1. Content Category 1
  2. Content Category 2

@rival_brand:

  📊 33-60s Bucket (550K avg views)
  1. Content Category 1
  2. Content Category 2

  📊 60-90s Bucket (520K avg views)
  1. Content Category 1
  2. Content Category 2

  📊 90-120s Bucket (490K avg views)
  1. Content Category 1
  2. Content Category 2

@fitness_guru:

  📊 33-60s Bucket (510K avg views)
  1. Content Category 1
  2. Content Category 2

  📊 60-90s Bucket (480K avg views)
  1. Content Category 1
  2. Content Category 2

  📊 90-120s Bucket (450K avg views)
  1. Content Category 1
  2. Content Category 2
```

TOP ENGAGEMENT DRIVERS
```
@wellness_pro

  📊 33-60s Bucket
  1. Engagement Driver 1
  2. Engagement Driver 2

  📊 60-90s Bucket
  1. Engagement Driver 1
  2. Engagement Driver 2

  📊 90-120s Bucket
  1. Engagement Driver 1
  2. Engagement Driver 2

@rival_brand:

  📊 33-60s Bucket
  1. Engagement Driver 1
  2. Engagement Driver 2

  📊 60-90s Bucket
  1. Engagement Driver 1
  2. Engagement Driver 2

  📊 90-120s Bucket
  1. Engagement Driver 1
  2. Engagement Driver 2

@fitness_guru:

  📊 33-60s Bucket
  1. Engagement Driver 1
  2. Engagement Driver 2

  📊 60-90s Bucket
  1. Engagement Driver 1
  2. Engagement Driver 2

  📊 90-120s Bucket
  1. Engagement Driver 1
  2. Engagement Driver 2
```


**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handles | Config | CLI parameter `--competitors` | Array[String] | ["@wellness_pro", "@rival_brand", "@fitness_guru"] | ✅ **Page 1 Header Section** |
| Winning buckets (per competitor) | Winner Analysis | Per competitor: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/winner_analysis.json` → `top_100_distribution` keys sorted by video count (top 3 buckets) | Array[String] per competitor | ["33-60s", "60-90s", "90-120s"] | ⚠️ **NOT VERIFIED** |
| Avg views (per bucket, per competitor) | Selected Videos | Per competitor, per bucket: Calculate average of `selected_videos.json → videos[0:top_count].playCount`. Path: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/buckets/bucket_{name}/selected_videos.json` | Integer (K/M formatted) per bucket | 620K, 580K, 540K | ⚠️ **NOT VERIFIED** |
| Top 2 Content Categories (per bucket, per competitor) | Stage 2.7 | Per competitor, per bucket: **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "content_category", n=2, "top")` (Section 0.5.1.1) → Path: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/buckets/bucket_{name}/` | Array[String] per bucket | ["recipe_tutorial", "wellness_practice"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| Top 2 Engagement Drivers (per bucket, per competitor) | Stage 2.7 | Per competitor, per bucket: **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "engagement_drivers", n=2, "top")` (Section 0.5.1.1) → Returns snake_case → **Display**: Convert to title case | Array[String] per bucket | ["personal_testimony", "before_after_reveal"] → Display: ["Personal Testimony", "Before/After Reveal"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |


**Aggregation Note**: Unlike Report 2 (which aggregates across all buckets and shows same Top N in each bucket report), Report 4 calls aggregation functions **per winning bucket** (3 buckets × N competitors) to show bucket-specific competitive patterns for each competitor.

---

#### Section 2: Execution Playbook (How They Make It)

**Purpose**: Show hook strategies, CTAs, pain points addressed, and keywords per bucket
HOOK STRATEGIES
```
@wellness_pro

  📊 33-60s Bucket
  1. Hook Strategy 1
  2. Hook Strategy 2

  📊 60-90s Bucket
  1. Hook Strategy 1
  2. Hook Strategy 2

  📊 90-120s Bucket
  1. Hook Strategy 1
  2. Hook Strategy 2

@rival_brand:

  📊 33-60s Bucket
  1. Hook Strategy 1
  2. Hook Strategy 2

  📊 60-90s Bucket
  1. Hook Strategy 1
  2. Hook Strategy 2

  📊 90-120s Bucket
  1. Hook Strategy 1
  2. Hook Strategy 2

@fitness_guru:

  📊 33-60s Bucket
  1. Hook Strategy 1
  2. Hook Strategy 2

  📊 60-90s Bucket
  1. Hook Strategy 1
  2. Hook Strategy 2

  📊 90-120s Bucket
  1. Hook Strategy 1
  2. Hook Strategy 2
```

CTA STRATEGIES
```

@wellness_pro

  📊 33-60s Bucket
  1. CTA Strategy 1
  2. CTA Strategy 2

  📊 60-90s Bucket
  1. CTA Strategy 1
  2. CTA Strategy 2

  📊 90-120s Bucket
  1. CTA Strategy 1
  2. CTA Strategy 2

@rival_brand:

  📊 33-60s Bucket
  1. CTA Strategy 1
  2. CTA Strategy 2

  📊 60-90s Bucket
  1. CTA Strategy 1
  2. CTA Strategy 2

  📊 90-120s Bucket
  1. CTA Strategy 1
  2. CTA Strategy 2

@fitness_guru:

  📊 33-60s Bucket
  1. CTA Strategy 1
  2. CTA Strategy 2

  📊 60-90s Bucket
  1. CTA Strategy 1
  2. CTA Strategy 2

  📊 90-120s Bucket
  1. CTA Strategy 1
  2. CTA Strategy 2
```

PAIN POINTS
```

@wellness_pro

  📊 33-60s Bucket
  1. Pain Point 1
  2. Pain Point 2
  3. Pain Point 3

  📊 60-90s Bucket
  1. Pain Point 1
  2. Pain Point 2
  3. Pain Point 3

  📊 90-120s Bucket
  1. Pain Point 1
  2. Pain Point 2
  3. Pain Point 3

@rival_brand:

  📊 33-60s Bucket
  1. Pain Point 1
  2. Pain Point 2
  3. Pain Point 3

  📊 60-90s Bucket
  1. Pain Point 1
  2. Pain Point 2
  3. Pain Point 3

  📊 90-120s Bucket
  1. Pain Point 1
  2. Pain Point 2
  3. Pain Point 3

@fitness_guru:

  📊 33-60s Bucket
  1. Pain Point 1
  2. Pain Point 2
  3. Pain Point 3

  📊 60-90s Bucket
  1. Pain Point 1
  2. Pain Point 2
  3. Pain Point 3

  📊 90-120s Bucket
  1. Pain Point 1
  2. Pain Point 2
  3. Pain Point 3
```

KEYWORDS
```
@wellness_pro

  📊 33-60s Bucket
  1. Keyword 1
  2. Keyword 2
  3. Keyword 3

  📊 60-90s Bucket
  1. Keyword 1
  2. Keyword 2
  3. Keyword 3

  📊 90-120s Bucket
  1. Keyword 1
  2. Keyword 2
  3. Keyword 3

@rival_brand:

  📊 33-60s Bucket
  1. Keyword 1
  2. Keyword 2
  3. Keyword 3

  📊 60-90s Bucket
  1. Keyword 1
  2. Keyword 2
  3. Keyword 3

  📊 90-120s Bucket
  1. Keyword 1
  2. Keyword 2
  3. Keyword 3

@fitness_guru:

  📊 33-60s Bucket
  1. Keyword 1
  2. Keyword 2
  3. Keyword 3

  📊 60-90s Bucket
  1. Keyword 1
  2. Keyword 2
  3. Keyword 3

  📊 90-120s Bucket
  1. Keyword 1
  2. Keyword 2
  3. Keyword 3
```

CONTENT TACTICS
```
@wellness_pro

  📊 33-60s Bucket
  1. Content Tactic 1
  2. Content Tactic 2

  📊 60-90s Bucket
  1. Content Tactic 1
  2. Content Tactic 2

  📊 90-120s Bucket
  1. Content Tactic 1
  2. Content Tactic 2

@rival_brand:

  📊 33-60s Bucket
  1. Content Tactic 1
  2. Content Tactic 2

  📊 60-90s Bucket
  1. Content Tactic 1
  2. Content Tactic 2

  📊 90-120s Bucket
  1. Content Tactic 1
  2. Content Tactic 2

@fitness_guru:

  📊 33-60s Bucket
  1. Content Tactic 1
  2. Content Tactic 2

  📊 60-90s Bucket
  1. Content Tactic 1
  2. Content Tactic 2

  📊 90-120s Bucket
  1. Content Tactic 1
  2. Content Tactic 2
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handles | Config | CLI parameter `--competitors` | Array[String] | ["@wellness_pro", "@rival_brand", "@fitness_guru"] | ✅ **Page 1 Header Section** |
| Top 2 Hook Strategies (per bucket, per competitor) | Stage 2.7 | Per competitor, per bucket: **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "hook_strategy", n=2, "top")` (Section 0.5.1.1) | Array[String] per bucket | ["question_hook", "problem_solution"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| Top 2 CTA Strategies (per bucket, per competitor) | Stage 2.7 | Per competitor, per bucket: **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → Extract `caption_analysis.cta_type` from classifications → Find top 2 most common CTA types | Array[String] per bucket | ["link_in_bio", "save_post"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1) |
| Top 3 Pain Points (per bucket, per competitor) | Stage 2.7 | Per competitor, per bucket: **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "pain_points", n=3, "top")` (Section 0.5.1.1) | Array[String] per bucket | ["bloating", "low_energy", "weight_loss"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| Top 3 Keywords (per bucket, per competitor) | Stage 2.7 | Per competitor, per bucket: **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "keywords", n=3, "top")` (Section 0.5.1.1) | Array[String] per bucket | ["protein", "gut_health", "fiber"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |
| Top 2 Content Tactics (per bucket, per competitor) | Stage 2.7 | Per competitor, per bucket: **Base Function**: `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → **Wrapper**: `get_top_n_from_field(bucket_path, "content_tactics", n=2, "top")` (Section 0.5.1.1) | Array[String] per bucket | ["direct_to_camera", "voiceover"] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1.1) |

**Aggregation Note**: All fields use per-bucket aggregation (3 buckets × N competitors) to show bucket-specific patterns for competitive analysis.

---

#### Section 3: Hashtag Strategy Comparison

```
HASHTAG STRATEGY BY COMPETITOR:

Metric                    | @wellness_pro | @rival_brand | @fitness_guru
--------------------------|---------------|--------------|---------------
Total unique hashtags     | 42            | 28           | 35
Avg hashtags per video    | 11            | 9            | 10
Top 5 concentration       | 65%           | 73%          | 68%
Strategy Type             | Diversified   | Focused      | Diversified

TOP 5 HASHTAGS PER COMPETITOR:

@wellness_pro:
1. #wellness (78%)
2. #healthylifestyle (68%)
3. #nutrition (62%)
4. #guthealth (55%)
5. #holistichealth (48%)

@rival_brand:
1. #nutrition (82%)
2. #healthylifestyle (68%)
3. #wellness (54%)
4. #guthealth (47%)
5. #protein (43%)

@fitness_guru:
1. #fitness (76%)
2. #supplements (65%)
3. #nutrition (58%)
4. #healthylifestyle (52%)
5. #protein (45%)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handles | Config | CLI parameter `--competitors` | Array[String] | ["@wellness_pro", "@rival_brand", "@fitness_guru"] | ✅ **Page 1 Header Section** |
| Total unique hashtags (per competitor) | Selected Videos (All Comp) | Per competitor: **Function**: `extract_hashtag_analysis(client_id, competitor_handle)` (Section 0.5.3) → Loads `selected_videos.json` from each winning bucket → Extracts `videos[].hashtags[].name` from top performers → Returns `total_unique_hashtags`. Path: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/buckets/bucket_{name}/selected_videos.json` | Integer per competitor | 42, 28, 35 | ✅ **This session** |
| Avg hashtags per video (per competitor) | Selected Videos (All Comp) | Per competitor: **Function**: `extract_hashtag_analysis(client_id, competitor_handle)` (Section 0.5.3) → Returns `avg_hashtags_per_video`. Calculation: `total_hashtags / total_videos` across all winning buckets (top performers only) | Integer per competitor | 11, 9, 10 | ✅ **This session** |
| Top 5 concentration (per competitor) | Selected Videos (All Comp) | Per competitor: **Function**: `extract_hashtag_analysis(client_id, competitor_handle)` (Section 0.5.3) → Returns `top_5_concentration`. Calculation: `(top_5_occurrences / total_occurrences) × 100%`. Measures strategic focus: >70% = focused strategy, <70% = diversified strategy | Integer (%) per competitor | 65, 73, 68 | ✅ **This session** |
| Strategy Type (per competitor) | Calculated | Per competitor: Use `top_5_concentration` from Field #4 → If > 70%: "Focused", else: "Diversified". **Inline logic** (no function needed - simple conditional) | String per competitor | "Diversified", "Focused", "Diversified" | ✅ **This session** |
| Top 5 hashtags with usage % (per competitor) | Selected Videos (All Comp) | Per competitor: **Function**: `extract_hashtag_analysis(client_id, competitor_handle)` (Section 0.5.3) → Returns `top_10_hashtags`[:5]. Each item includes `tag`, `usage_pct`, and `video_count` | Array[Object] per competitor | [{"tag": "#wellness", "pct": 78}, ...] | ✅ **This session** |

**Aggregation Note**: Section 3 aggregates hashtag data **across all winning buckets** (not per-bucket) to show overall hashtag strategy per competitor. Data source is `selected_videos.json → videos[].hashtags[]` for top performers across all 3 winning buckets.

---

#### Section 4: Caption Strategy Comparison

**Purpose**: Compare caption formatting and CTA strategies across multiple competitors

```
CAPTION STRATEGY ACROSS COMPETITORS:

Metric                  | @wellness_pro | @rival_brand | @fitness_guru
------------------------|---------------|--------------|---------------
Avg Hashtag Count       | 11            | 9            | 10
Top CTA Type            | Link bio (68%)| Link bio (72%)| Follow (58%)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handles | Config | CLI parameter `--competitors` | Array[String] | ["@wellness_pro", "@rival_brand", "@fitness_guru"] | ✅ **Page 1 Header Section** |
| Avg hashtag count (per competitor) | Stage 2.7 (All Comp) | Per competitor, across all winning buckets: For each bucket, call `aggregate_content_classifications(bucket_path, "top")` (Section 0.5.1) → Extract `hashtag_count_stats.mean` and `total_videos` → Calculate weighted average: `Σ(bucket_mean × bucket_video_count) / Σ(bucket_video_count)`. **Inline logic** (no new function needed - simple weighted average calculation) | Integer per competitor | 11, 9, 10 | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1) |
| Top CTA (per competitor) | Stage 2.7 (All Comp) | Per competitor, across all winning buckets: Call `aggregate_content_classifications(bucket_path, "top")` for each bucket → Aggregate `caption_cta_type` Counter objects → Find mode (most common value) → Calculate % = (mode_count / total_videos) × 100% | String with % per competitor | "link_in_bio (68%)", "link_in_bio (72%)", "Follow (58%)" | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** (Section 0.5.1) |

**Aggregation Note**: Section 4 aggregates caption data **across all winning buckets** (not per-bucket) to show overall caption strategy per competitor. Uses `aggregate_content_classifications()` from Section 0.5.1 which already provides caption field aggregations.

**Note**: Caption analysis fields are universal (standardized enums/integers), so they're directly comparable across all competitors without taxonomy dependency.

---

#### Section 5: Content Sourcing Strategy

**Purpose**: Show each competitor's affiliate partnerships and content sourcing approach

```
CONTENT SOURCING STRATEGY COMPARISON:

Metric                          | @wellness_pro | @rival_brand | @fitness_guru
--------------------------------|---------------|--------------|---------------
UGC/Affiliate Content %         | 28%           | 42%          | 15%
Own Content %                   | 72%           | 58%          | 85%
Total Unique Affiliate Partners | 22            | 47           | 12


═══════════════════════════════════════════════════════════════
@wellness_pro
═══════════════════════════════════════════════════════════════

Top Affiliate Contributors:
1. @holistichealth_coach  (12% of videos - 36 mentions)
2. @wellness_collective   (8% of videos - 24 mentions)
3. @naturalremedies       (5% of videos - 15 mentions)
4. @ayurveda_lifestyle    (3% of videos - 9 mentions)

---

═══════════════════════════════════════════════════════════════
@rival_brand
═══════════════════════════════════════════════════════════════

Top Affiliate Contributors:
1. @fitnessguru123       (18% of videos - 54 mentions)
2. @healthcoach_jane     (12% of videos - 36 mentions)
3. @nutritionpro         (8% of videos - 24 mentions)
4. @wellnesswarrior      (5% of videos - 15 mentions)
5. @cleaneatingclub      (4% of videos - 12 mentions)

---

═══════════════════════════════════════════════════════════════
@fitness_guru
═══════════════════════════════════════════════════════════════

Top Affiliate Contributors:
1. @transformationclub    (8% of videos - 24 mentions)
2. @fitnesstips_daily     (4% of videos - 12 mentions)
3. @workout_motivation    (3% of videos - 9 mentions)

═══════════════════════════════════════════════════════════════
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor handles (table columns) | Config | CLI parameter `--competitors` | Array[String] | ["@wellness_pro", "@rival_brand", "@fitness_guru"] |
| UGC/Affiliate Content % (per competitor) | Function Output | Per competitor across all winning buckets: `extract_mention_analysis(manifest_path)` (Section 0.5.4) → Returns `repost_rate` field | Integer (%) per competitor | 28, 42, 15 |
| Own Content % (per competitor) | Calculated | Per competitor: `100% - repost_rate` | Integer (%) per competitor | 72, 58, 85 |
| Total Unique Affiliate Partners (per competitor) | Function Output | Per competitor across all winning buckets: `extract_mention_analysis(manifest_path)` (Section 0.5.4) → Returns `total_unique_mentions` field | Integer per competitor | 22, 47, 12 |
| Top affiliate contributors (per competitor, 3-5 items) | Function Output | Per competitor across all winning buckets: `extract_mention_analysis(manifest_path)` (Section 0.5.4) → Returns `top_10_mentions` array. Display top 3-5 items in report | Array[Object] per competitor | [{"handle": "@fitnessguru123", "percentage": 18.0, "mention_count": 54}, ...] |

**Data Source**:
- `selected_videos.json` → `videos[].text` per competitor (caption text)
- Regex extraction: `re.findall(r'@(\w+)', caption)`
- Repost indicators: ["repost", "via", "credit", "by", "from"]

**Implementation**: See Stage8MVP.md Section 0.5.4 for `extract_mention_analysis()` function


#### Section 5: Top Things To Do (Quantitative)
---

**[BUCKET_1_NAME]** (e.g., 18-33s)

**Top Things to Do - [COMPETITOR_1_HANDLE]**
□ [BUCKET_1_COMP_1_INSIGHT_1]
□ [BUCKET_1_COMP_1_INSIGHT_2]
□ [BUCKET_1_COMP_1_INSIGHT_3]
□ [BUCKET_1_COMP_1_INSIGHT_4]
□ [BUCKET_1_COMP_1_INSIGHT_5]

**Top Things to Do - [COMPETITOR_2_HANDLE]** *(if applicable)*
□ [BUCKET_1_COMP_2_INSIGHT_1]
□ [BUCKET_1_COMP_2_INSIGHT_2]
□ [BUCKET_1_COMP_2_INSIGHT_3]
□ [BUCKET_1_COMP_2_INSIGHT_4]
□ [BUCKET_1_COMP_2_INSIGHT_5]

**Top Things to Do - [COMPETITOR_3_HANDLE]** *(if applicable)*
□ [BUCKET_1_COMP_3_INSIGHT_1]
□ [BUCKET_1_COMP_3_INSIGHT_2]
□ [BUCKET_1_COMP_3_INSIGHT_3]
□ [BUCKET_1_COMP_3_INSIGHT_4]
□ [BUCKET_1_COMP_3_INSIGHT_5]

---

**[BUCKET_2_NAME]** *(if applicable)*

**Top Things to Do - [COMPETITOR_1_HANDLE]** *(if applicable for this bucket)*
□ [BUCKET_2_COMP_1_INSIGHT_1]
□ [BUCKET_2_COMP_1_INSIGHT_2]
□ [BUCKET_2_COMP_1_INSIGHT_3]
□ [BUCKET_2_COMP_1_INSIGHT_4]
□ [BUCKET_2_COMP_1_INSIGHT_5]

**Top Things to Do - [COMPETITOR_2_HANDLE]** *(if applicable for this bucket)*
□ [BUCKET_2_COMP_2_INSIGHT_1]
□ [BUCKET_2_COMP_2_INSIGHT_2]
□ [BUCKET_2_COMP_2_INSIGHT_3]
□ [BUCKET_2_COMP_2_INSIGHT_4]
□ [BUCKET_2_COMP_2_INSIGHT_5]

**Top Things to Do - [COMPETITOR_3_HANDLE]** *(if applicable for this bucket)*
□ [BUCKET_2_COMP_3_INSIGHT_1]
□ [BUCKET_2_COMP_3_INSIGHT_2]
□ [BUCKET_2_COMP_3_INSIGHT_3]
□ [BUCKET_2_COMP_3_INSIGHT_4]
□ [BUCKET_2_COMP_3_INSIGHT_5]

---

**[BUCKET_3_NAME]** *(if applicable)*

**Top Things to Do - [COMPETITOR_1_HANDLE]** *(if applicable for this bucket)*
□ [BUCKET_3_COMP_1_INSIGHT_1]
□ [BUCKET_3_COMP_1_INSIGHT_2]
□ [BUCKET_3_COMP_1_INSIGHT_3]
□ [BUCKET_3_COMP_1_INSIGHT_4]
□ [BUCKET_3_COMP_1_INSIGHT_5]

**Top Things to Do - [COMPETITOR_2_HANDLE]** *(if applicable for this bucket)*
□ [BUCKET_3_COMP_2_INSIGHT_1]
□ [BUCKET_3_COMP_2_INSIGHT_2]
□ [BUCKET_3_COMP_2_INSIGHT_3]
□ [BUCKET_3_COMP_2_INSIGHT_4]
□ [BUCKET_3_COMP_2_INSIGHT_5]

**Top Things to Do - [COMPETITOR_3_HANDLE]** *(if applicable for this bucket)*
□ [BUCKET_3_COMP_3_INSIGHT_1]
□ [BUCKET_3_COMP_3_INSIGHT_2]
□ [BUCKET_3_COMP_3_INSIGHT_3]
□ [BUCKET_3_COMP_3_INSIGHT_4]
□ [BUCKET_3_COMP_3_INSIGHT_5]

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Bucket names (1-3) | Stage 1 | For each competitor: `/data/clients/{client}/competitors/{competitor_handle}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets` → Merge all competitors' winning buckets → Find common buckets across 2+ competitors → Return unique bucket list (sorted by frequency) | Array[String] | ["18-33s", "13-18s", "60-90s"] | ✅ **Verified** |
| Competitor handles per bucket | Calculated | For each bucket in common_buckets: Check each competitor's `top_3_buckets` → If bucket exists in competitor's list, include competitor → Return list of applicable competitor handles per bucket | Nested Array | Bucket "18-33s": ["@drinkpoppi", "@nike", "@vitalproteins"], Bucket "13-18s": ["@drinkpoppi", "@vitalproteins"] | ✅ **Verified** |
| Supplementary Insights per competitor per bucket | Stage 7 | For each (bucket, competitor) pair: `/data/clients/{client}/competitors/{competitor_handle}/{mode}_{strategy}/buckets/bucket_{bucket_name}/ml_analysis/llm/winning_formulas.json` → `supplementary_insights.universal_principles[0-4]` → Only extract if competitor has this bucket in their `top_3_buckets` | Nested Array of Strings | Bucket "18-33s", Competitor "@drinkpoppi": ["middle_3_eye_contact_rate: 0.57 in top vs 0.43 in bottom (gap: 0.14)", "middle_1_energy_variance: 0.00 in top vs 0.00 in bottom (gap: 0.00)", "middle_3_energy_variance: 0.00 in top vs 0.00 in bottom (gap: 0.00)", "middle_3_energy_level: 0.10 in top vs 0.06 in bottom (gap: 0.04)", "hook_eye_contact_rate: 0.51 in top vs 0.63 in bottom (gap: 0.11)"] | ⚠️ **READY** (Stage 7 data exists) |

### Page 4: Visual Examples

**Purpose**: Provide visual proof of top-performing videos per competitor

---

#### Section 1: Visual Proof (Top Performers)

```
TOP PERFORMING VIDEOS BY COMPETITOR:

@wellness_pro (Market Leader):
[QR CODE - 1" x 1"]

📹 Video Stats: 820K views | 1.5% engagement | 45s duration
Formula: "The Transformation Journey" (33-60s bucket)
Hashtags: #wellness #guthealth #transformation #healthylifestyle

Key Pattern Elements:
• Before/after reveal in first 5 seconds (immediate hook)
• Personal testimony with specific metrics ("Lost 15 lbs in 30 days")
• Expert credentials mentioned (registered nutritionist)
• Product shown at mid-video (second 22)
• 8 text overlays throughout (retention tactic)


@rival_brand:
[QR CODE - 1" x 1"]

📹 Video Stats: 720K views | 1.4% engagement | 22s duration
Formula: "The Question Hook Recipe Tutorial" (18-33s bucket)
Hashtags: #nutrition #guthealth #recipe #protein

Key Pattern Elements:
• Question hook in first 2 seconds ("Did you know this ingredient...")
• Fast pacing (3 scene changes per 10 seconds)
• Recipe step-by-step demonstration (actionable)
• Nutritional benefit callouts with text overlays
• Clear CTA at end ("Link in bio for full recipe!")


@fitness_guru:
[QR CODE - 1" x 1"]

📹 Video Stats: 650K views | 1.3% engagement | 16s duration
Formula: "The Quick Win Tutorial" (13-18s bucket)
Hashtags: #fitness #supplements #protein #healthytips

Key Pattern Elements:
• Quick win promise in opening ("Fix bloating in 5 minutes")
• Direct-to-camera style (personal connection)
• Simple 3-step technique (easy to replicate)
• Personal testimony embedded ("This worked for me")
• Time pressure CTA ("Try it today!")
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| QR codes (3 competitors) | Stage 2 (All Comp) | Generate from top performer video URLs | QR Code Image (array) | 3 QR codes |
| Video stats (per competitor) | Stage 2 (All Comp) | `view_count`, `duration` from top performer metadata | Object per competitor | {views, duration} |
| Video engagement (per competitor) | Calculated | `calculate_engagement_metrics()` for top performer video | Float (%) per competitor | 1.5, 1.4, 1.3 |
| Formula names (per competitor) | Stage 7 (All Comp) | `pattern_name` mapped to top video | String per competitor | "The Transformation Journey", etc. |
| Video buckets (per competitor) | Stage 1 (All Comp) | Bucket classification from duration | String per competitor | "33-60s", "18-33s", "13-18s" |
| Hashtags (per competitor) | Stage 2 (All Comp) | `video_hashtags` from metadata (first 4) | String (array per competitor) | ["#wellness", "#guthealth", ...] |
| Key pattern elements (5 per video) | Stage 7 (All Comp) | Pattern characteristics from formula analysis | String (array per competitor) | ["Before/after reveal in first 5 seconds", ...] |

**Video Selection Criteria**:
- **Priority 1**: Highest view count from winning bucket per competitor
- **Priority 2**: Newest video (if multiple high performers - reduces deletion risk)
- **Priority 3**: Videos from accounts with 100K+ followers (stability)

---

### Data Extraction Requirements for extract_competitor_data.py

**New Functions Required** (extends single competitor extraction):

### 1. Multi-Competitor Performance Rankings

**Function**: `calculate_competitor_rankings(competitors_data_list)`

**Process**:
1. Calculate weighted avg_views for all competitors
2. Extract posting frequency (videos/week) for all
3. Calculate total duration for all
4. Sort by avg_views (descending)
5. Generate ranking table

**Outputs**:
- Rankings table (all competitors sorted)
- Market leader identification

---

### 2. Bucket Distribution Matrix

**Function**: `build_bucket_distribution_matrix(competitors_data_list)`

**Process**:
1. Extract bucket_distribution for all competitors (8 buckets each)
2. Build matrix: rows = 8 buckets, columns = competitors
3. Flag high-volume buckets (>20%) per competitor
4. Calculate market patterns (avg % per bucket)
5. Generate strategic insights

**Outputs**:
- Distribution matrix (8 × N competitors)
- High-volume markers per competitor
- Market pattern categories per bucket
- Strategic insights (3-4 items)

---

### 3. Performance Matrix

**Function**: `build_performance_matrix(competitors_data_list)`

**Process**:
1. Extract avg_views per bucket for all competitors
2. Build matrix: rows = 8 buckets, columns = competitors
3. Identify best performer per bucket
4. Flag top-tier performances (>550K)

**Outputs**:
- Performance matrix (8 × N competitors)
- Best performer per bucket
- Performance insights (4 items)

---

### 4. Content Category Matrix

**Function**: `build_category_matrix(competitors_data_list)`

**Process**:
1. Aggregate content_category from Stage 2.7 for all competitors
2. Calculate % distribution for top 5 categories per competitor
3. Build matrix: rows = 5 categories, columns = competitors
4. Identify market leader per category
5. Generate strategic positioning statements

**Outputs**:
- Category matrix (5 × N competitors)
- Market leader per category
- Strategic positioning per competitor
- Market patterns (3 items)

---

### 5. Best Formulas Extraction (Multi-Competitor)

**Function**: `extract_best_formulas_multi(competitors_data_list, top_n=2)`

**Process**:
1. Load Stage 7 winning_formulas.json for all competitors
2. Select top N formulas per competitor (by cluster size or frequency)
3. Extract: pattern_name, bucket, usage %, pattern description, key elements
4. No engagement metrics (not available)

**Outputs**:
- Top N formulas per competitor (N × competitors)
- Formula metadata per formula

---

### 6. Pain Points & Keywords Cross-Analysis

**Function**: `analyze_pain_points_keywords_multi(competitors_data_list)`

**Process**:
1. Aggregate pain_points and keywords from Stage 2.7 for all competitors
2. Calculate % coverage per competitor per pain point/keyword
3. Identify top 5 pain points and top 5 keywords market-wide
4. Build matrices, identify market leaders per topic
5. Calculate market prevalence (avg % across competitors)

**Outputs**:
- Pain points matrix (5 × N competitors) with market prevalence
- Keywords matrix (5 × N competitors) with market leaders
- Audience insights (4 items)
- Keyword insights (4 items)

---

### 7. Engagement Tactics Matrix

**Function**: `build_engagement_tactics_matrix(competitors_data_list)`

**Process**:
1. Aggregate engagement_drivers from Stage 2.7 for all competitors
2. Calculate % usage per competitor per tactic
3. Identify top 5 tactics market-wide
4. Build matrix, identify most frequent user per tactic

**Outputs**:
- Tactics matrix (5 × N competitors)
- Most used by (per tactic)
- Tactics insights (5 items)

---

### Mobile Optimization Requirements

**NOT REQUIRED for Template 4** (Client reports are desktop-focused)

- Standard PDF formatting: 8.5" × 11" letter size
- Minimum font sizes: 10pt body text acceptable
- Multi-column layouts allowed

---

### Summary of Template 4 vs Template 3

| Feature | Template 3 (Single Competitor) | Template 4 (Multiple Competitors) |
|---------|-------------------------------|----------------------------------|
| **Competitors analyzed** | 1 | 2-5 |
| **Client comparison** | Yes (gap analysis) | No (pure market intelligence) |
| **Recommendations** | Yes (3-tier action plan) | No (intelligence only) |
| **Page count** | 4 pages | 4 pages |
| **Format** | Competitor vs Client | Multi-competitor comparison |
| **QR codes** | 1 (competitor top video) | 3 (top video per competitor) |
| **Use case** | Deep-dive on 1 competitor | Market landscape overview |

---

**Status**: ✅ **COMPLETE** - Template structure finalized with all dynamic field mappings

---

## Next Steps

1. ✅ **Task 0.1**: Hashtag → Client structure (COMPLETE)
2. ✅ **Task 0.2**: Hashtag → Creator structure (COMPLETE)
3. ✅ **Task 0.3**: Design Handle/Single Competitor → Client structure (COMPLETE)
4. ✅ **Task 0.4**: Design Handle/Multiple Competitor → Client structure (COMPLETE)

**Section 0 Status**: ✅ **COMPLETE** - All 4 template structures finalized

**Next Actions**:
- Section 1-2: Designer can start building 4 PDF templates (11 days)
- Section 3: Developer can start data extraction scripts (3.25 days)
- These can run **in parallel**

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

**Status**: ✅ **COMPLETE** - All templates now have comprehensive data mapping tables for all sections.

---

**Status**: ✅ **4 of 4 templates complete** - Section 0 complete, ready for parallel design + development work

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
4. **Honest marketing**: Uses phase-based structure (Hook → Middle → Closing) instead of claiming precise "second-by-second timeline"

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
