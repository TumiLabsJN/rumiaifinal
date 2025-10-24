# Stage 8 MVP: Report Template Structures

**Purpose**: Centralized template structure definitions for all Stage 8 PDF reports

**Parent Document**: Stage8MVP.md

**Status**: 4 of 4 templates complete - Section 0 complete, ready for design + development

---

## Template Structure Overview

| # | Report Type | Audience | Status | Source |
|---|-------------|----------|--------|--------|
| 1 | Hashtag → Client | Tumi Labs Clients | ✅ **COMPLETE** | MLCreativeReports.md |
| 2 | Hashtag → Creator | Content Creators | ✅ **COMPLETE** | Stage8Planning.md section 1.1 |
| 3 | Handle/Single Competitor → Client | Tumi Labs Clients | ✅ **COMPLETE** | This document (Page 1-4 structure) |
| 4 | Handle/Multiple Competitor → Client | Tumi Labs Clients | ✅ **COMPLETE** | This document (Page 1-4 structure) |

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
- Stage 2: Video metadata (`views`, `likes`, `comments`, `shares`, `saves`) from `unified_analysis/{video_id}.json` → `metadata` (lines 8-12) for engagement calculation
- Stage 6: `rf_video_analysis.json`, `kmeans_analysis.json`
- Stage 7: `winning_formulas.json` (all 3 buckets)

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

**Decision**: ✅ Always display "Past 2-3 months" regardless of actual `--date-filter` parameter for marketing consistency and perceived recency.

**Decision**: ✅ Display total scraped videos (1,826) instead of selected videos (111) for marketing impact. The full funnel (winning formats + deep analysis counts) provides context and shows rigor without misleading.

**Decision**: ✅ Show all 3 winning buckets with percentages (not just top bucket) to demonstrate comprehensive pattern identification.

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

**Decision**: ✅ Keep only Creator Profile Priorities section. Removed redundant sections:
- Content Saturation (not actionable, redundant)
- Trend Direction (too risky to fabricate M-o-M trends)
- Creator Recommendations with specific hiring quantities (too prescriptive)

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

**Data Source**:
- Stage 2.7 Content Analysis classifications from 120 winning videos (40 per bucket × 3 winning buckets)
- Aggregated using `aggregate_content_classifications()` function (see Stage8MVP.md Section 0.5.1)
- Percentages calculated as: `(videos_with_feature / total_winning_videos) × 100%`

**Decision**: ✅ Compact dashboard format - fits on Page 2, provides actionable qualitative insights for creator briefs without overwhelming client. Shows top 3-4 items per category for scannability.

---

### Page 3: Your Creative Reports

**Purpose**: Show what reports were delivered

---

#### Report Distribution

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

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Hashtag (in intro text) | Config | `/config/hashtag_clusters/{target}.json` → `primary_hashtag` | String | "#nutrition" | ✅ **Report 1 Header** |
| Duration Bucket ranges (3 buckets) | Stage 1 | `/data/clients/{client}/hashtag/{target}/{mode}_{strategy}/winner_analysis.json` → `top_3_buckets` array | Array[String] | ["18-33s", "13-18s", "60-90s"] | ✅ **Report 1 Header** |
| Formula names (9 formulas) | Stage 8 LLM (future) | For each winning bucket: `aggregate_content_classifications()` → `generate_formula_names_llm()` (LLM-generated via Section 0.5.6 - TO BE DOCUMENTED) → 3 names per bucket, 9 total | Array[String] | ["The Question Hook Formula", "The Transformation Story", ...] | ⚠️ **FUTURE WORK** (Section 0.5.6 LLM function not documented) |

---

#### What Each Report Contains

Each 2-page report includes:
  • Proof with numbers (engagement differences)
  • Second-by-second execution guide
  • Pre-post checklist

**Would you like to review a sample report?** Contact us at [email]

**Decision**: ✅ Minimal Page 3 with report distribution list and sample report offer. "How to Use These Reports", "What Makes These Reports Effective", and "Next Steps" sections removed (onboarding material, not recurring report content).

---

### Data Extraction Requirements for extract_client_data.py

**New Calculations Required**:

1. **Total Video Duration Calculation**
   - **Source**: Stage 1 metadata (video durations)
   - **Process**:
     1. Load all video metadata from Stage 1 analysis
     2. Sum all video durations (in seconds)
     3. Convert to hours: `total_seconds / 3600`
     4. Format as string: "6.2 hours of content"
   - **Output**: Formatted duration string

2. **Bucket Distribution Calculation**
   - **Source**: Stage 1 `winner_analysis.json` → `bucket_distribution`
   - **Process**:
     1. Load bucket_distribution object (contains video counts per bucket)
     2. Calculate total videos across all buckets
     3. For each bucket: `(bucket_count / total_videos) × 100%`
     4. Round to nearest integer
   - **Outputs**:
     - Percentage per bucket (8 values)
     - Key insight percentage (sum of top 2 buckets)

3. **Performance Star Rating Logic**
   - **Source**: Stage 1 `winner_analysis.json` → `avg_views` per winning bucket
   - **Process**:
     1. Load avg_views for 3 winning buckets only
     2. Identify min and max avg_views across 3 winning buckets
     3. Assign star ratings based on relative performance:
        - Top performer (highest views + engagement): ⭐⭐⭐⭐⭐ (5 stars)
        - Second performer: ⭐⭐⭐⭐ (4 stars)
        - Third performer: ⭐⭐⭐ (3 stars)
   - **Output**: Star emoji strings for 3 winning buckets only
   - **Note**: Only winning buckets shown due to data availability constraint (Option A)

4. **Sweet Spot Identification**
   - **Source**: Stage 1 `winner_analysis.json` → `avg_views` + `bucket_distribution` for winning buckets
   - **Process**:
     1. Load avg_views for 3 winning buckets
     2. Load bucket_distribution percentages for winning buckets
     3. Identify the top performing winning bucket with:
        - Highest avg_views or engagement
        - Strong volume (high % of total content)
     4. Return as single bucket label (e.g., "18-33s")
   - **Output**: Sweet spot duration range string (single bucket from winning 3)
   - **Note**: Simplified for Option A - focuses on single best bucket rather than range
   - **Example Logic**:
     ```python
     def identify_sweet_spot(winning_buckets):
         # winning_buckets is array of 3 buckets with avg_views and engagement
         # Sort by engagement (primary) or combined score
         best_bucket = max(winning_buckets, key=lambda x: x['avg_engagement'])

         return best_bucket['bucket_range']  # e.g., "18-33s"
     ```

5. **Performance Label Assignment**
   - **Source**: Winning bucket rankings (Stage 1)
   - **Process**:
     1. Identify top 3 winning buckets
     2. Assign labels based on rank:
        - Rank 1 (highest): "highest performance"
        - Rank 2: "strong performance + volume"
        - Rank 3: "proven success"
   - **Output**: Performance label strings for Tier 1 section

6. **Formula Name Extraction**
   - **Source**: Stage 7 `winning_formulas.json` → `pattern_name` field
   - **Process**:
     1. Load winning_formulas.json
     2. For each of top 3 buckets, extract 3 formula names
     3. Maintain bucket grouping (formulas 1-3 for bucket 1, 4-6 for bucket 2, etc.)
     4. Total: 9 formula names
   - **Output**: Array of 9 formula names organized by bucket

7. **Content Intelligence Aggregation (Section 4)**
   - **Source**: Stage 2.7 Content Analysis classifications (120 winning videos: 40 per bucket × 3 buckets)
   - **Process**:
     1. Load `selection_manifest.json` to get all winning video IDs (top performers across 3 buckets)
     2. For each winning video ID, load `{bucket}/content_analysis/{video_id}_content.json`
     3. Aggregate classifications using `aggregate_content_classifications()` (see Stage8MVP.md Section 0.5.1)
     4. Calculate frequencies for each category:
        - **Content Categories**: Count `content_category` field (single selection per video)
        - **Hook Strategies**: Count `hook_strategy` field (single selection per video)
        - **Keywords**: Aggregate `keywords` array (multiple per video)
        - **Pain Points**: Aggregate `pain_points` array (multiple per video)
        - **Engagement Drivers**: Aggregate `engagement_drivers` array (multiple per video)
        - **Caption Strategy**: Extract from `caption_analysis` object (6 fields)
     5. Calculate percentages: `(videos_with_feature / 120) × 100%`
     6. Rank by frequency and select top N items per category
   - **Outputs**:
     - Top 3 content categories with percentages (e.g., "Recipe Tutorial: 38%")
     - Top 3 hook strategies with percentages (e.g., "Problem-Solution: 42%")
     - Top 4 keywords (e.g., "#guthealth, #protein, #antiinflammatory, #metabolism")
     - Top 3 pain points with percentages (e.g., "Bloating: 48%")
     - Top 3 engagement drivers with percentages (e.g., "Before/After Reveal: 45%")
     - **Caption strategy** (5 fields):
       - Optimal hashtag count: mean of `hashtag_count` (e.g., "7 hashtags")
       - Hashtag breakdown: mean of `hashtag_strategy.niche_count` and `broad_count` (e.g., "5 niche, 2 broad")
       - Caption length winner: most common `caption_length` value (e.g., "Short: 68%")
       - Emoji usage winner: most common `emoji_usage` value (e.g., "Light (1-4): 72%")
       - Top CTA type: most common `caption_cta_type` value (e.g., "Link in bio: 58%")
   - **Example Implementation**:
     ```python
     from collections import Counter
     import numpy as np

     def aggregate_content_intelligence(selection_manifest_path, bucket_paths):
         # Load all winning video IDs
         manifest = load_json(selection_manifest_path)
         winning_video_ids = []
         for bucket, videos in manifest['videos_by_bucket'].items():
             winning_video_ids.extend(videos.get('top_performers', []))

         # Aggregate classifications
         content_categories = []
         hook_strategies = []
         keywords_list = []
         pain_points_list = []
         engagement_drivers_list = []

         # Caption strategy aggregation
         hashtag_counts = []
         niche_counts = []
         broad_counts = []
         caption_lengths = []
         emoji_usages = []
         cta_types = []

         for video_id in winning_video_ids:
             # Find and load content analysis file
             content_path = find_content_analysis_file(video_id, bucket_paths)
             data = load_json(content_path)

             # Extract core content fields
             content_categories.append(data.get('content_category'))
             hook_strategies.append(data.get('hook_strategy'))
             keywords_list.extend(data.get('keywords', []))
             pain_points_list.extend(data.get('pain_points', []))
             engagement_drivers_list.extend(data.get('engagement_drivers', []))

             # Extract caption strategy fields
             caption_analysis = data.get('caption_analysis', {})
             hashtag_counts.append(caption_analysis.get('hashtag_count', 0))
             hashtag_strategy = caption_analysis.get('hashtag_strategy', {})
             niche_counts.append(hashtag_strategy.get('niche_count', 0))
             broad_counts.append(hashtag_strategy.get('broad_count', 0))
             caption_lengths.append(caption_analysis.get('caption_length'))
             emoji_usages.append(caption_analysis.get('emoji_usage'))
             cta_types.append(caption_analysis.get('caption_cta_type'))

         total_videos = len(winning_video_ids)

         # Calculate top items with percentages
         return {
             'top_content_categories': format_top_n(Counter(content_categories), 3, total_videos),
             'top_hook_strategies': format_top_n(Counter(hook_strategies), 3, total_videos),
             'top_keywords': [k for k, _ in Counter(keywords_list).most_common(4)],
             'top_pain_points': format_top_n(Counter(pain_points_list), 3, total_videos),
             'top_engagement_drivers': format_top_n(Counter(engagement_drivers_list), 3, total_videos),

             # Caption strategy
             'optimal_hashtag_count': round(np.mean(hashtag_counts)),
             'hashtag_breakdown': f"{round(np.mean(niche_counts))} niche, {round(np.mean(broad_counts))} broad",
             'caption_length_winner': format_top_1(Counter(caption_lengths), total_videos),
             'emoji_usage_winner': format_top_1(Counter(emoji_usages), total_videos),
             'top_cta_type': format_top_1(Counter(cta_types), total_videos)
         }

     def format_top_n(counter, n, total):
         return [
             f"{item}: {round((count/total)*100)}%"
             for item, count in counter.most_common(n)
         ]

     def format_top_1(counter, total):
         if not counter:
             return "N/A"
         item, count = counter.most_common(1)[0]
         return f"{item}: {round((count/total)*100)}%"
     ```

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
| **Content Intelligence** | Compact dashboard (Option 1) - Section 4 Page 2 | Actionable qualitative insights, fits existing structure, scannable format |

---

## 2. Hashtag → Creator (Content Creator Report)

**Audience**: Content creators (affiliates)

**Purpose**: Deliver actionable creative formulas with proof and execution steps

**Deliverable**: 9 PDFs per hashtag (3 buckets × 3 formulas each)

**Format**: 2-page PDF (**MOBILE-OPTIMIZED** - minimum 12pt body, 16pt+ headings, portrait layout)

**Reading Time**: 2-3 minutes

---

### Input Data Sources

- Stage 2: Video metadata (`views`, `likes`, `comments`, `shares`, `saves`) from `unified_analysis/{video_id}.json` → `metadata` (lines 8-12) for engagement calculation
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
Duration: 18-33s | Hashtag: #nutrition
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Hashtag | Cluster Config | `/config/hashtag_clusters/{target}.json` → `primary_hashtag` | String | "#vitamin" | ✅ **Report 1** |
| Duration | Stage 7 | `winning_formulas.json` → `bucket_range` for this formula | String | "18-33s" | ✅ **Report 1** |
| Pattern Name | Stage 7 | `winning_formulas.json` → `pattern_name` (LLM-generated from Content Analysis) | String | "The Question Hook Formula" | ⚠️ **NOT VERIFIED** |

---

#### The Proof (Real Performance Data)

```
📊 PERFORMANCE COMPARISON:

Videos using this pattern (Top Cluster):
• Average Views: 620K
• Average Engagement: 1.2% (7,440 interactions per video)

Videos NOT using this pattern (Bottom Cluster):
• Average Views: 380K
• Average Engagement: 0.8% (3,040 interactions per video)

RESULTS:
→ 1.6x MORE VIEWS (63% higher reach)
→ 1.5x MORE ENGAGEMENT (50% higher resonance)

[QR CODE]
Scan to watch: Top Performer Using This Pattern (620K views, 1.4% engagement)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Top cluster avg views | Stage 1 + 6 | Avg `view_count` for videos in top cluster | Integer (formatted with K) | 620K |
| Top cluster avg engagement | Calculated | Avg `calculate_engagement_metrics()` for top cluster videos | Float (%) | 1.2 |
| Top cluster interactions | Calculated | Avg engagement × Avg views | Integer | 7,440 |
| Bottom cluster avg views | Stage 1 + 6 | Avg `view_count` for videos in bottom cluster | Integer (formatted with K) | 380K |
| Bottom cluster avg engagement | Calculated | Avg `calculate_engagement_metrics()` for bottom cluster videos | Float (%) | 0.8 |
| Bottom cluster interactions | Calculated | Avg engagement × Avg views | Integer | 3,040 |
| View multiplier | Calculated | Top views / Bottom views | Float (ratio) | 1.6x |
| Engagement multiplier | Calculated | Top engagement / Bottom engagement | Float (ratio) | 1.5x |
| View percentage increase | Calculated | ((Top - Bottom) / Bottom) × 100% | Integer (%) | 63 |
| Engagement percentage increase | Calculated | ((Top - Bottom) / Bottom) × 100% | Integer (%) | 50 |
| QR Code (Top Performer) | Stage 2 + 7 | Video URL from top cluster in Stage 2 metadata, mapped to this formula | QR Code Image | Links to TikTok video |
| Example video views | Stage 2 | `view_count` from top performer video metadata | Integer (formatted with K/M) | 620K |
| Example video engagement | Calculated | `calculate_engagement_metrics()` for top performer video | Float (%) | 1.4 |

**Calculation Method** (Real Engagement from Apify Data):

Uses `calculate_engagement_metrics()` function (Section 0.5.5 in Stage8MVP.md) to calculate real engagement rates from actual TikTok interaction data:
- **Data Source**: `unified_analysis/{video_id}.json` → `metadata` (lines 8-12)
- **Formula**: `(likes + comments + shares + saves) / views × 100%`
- **Applied to**: All videos in top cluster (pattern users) and bottom cluster (non-pattern users)
- **Output**: Real measured engagement rates, not estimates

**Benefit**: Data integrity and transparency - shows actual engagement performance, not industry benchmark estimates.

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

### Data Extraction Requirements for extract_creator_data.py

**New Calculations Required**:

1. **Pattern Performance Comparison (Top vs Bottom Cluster)**
   - **Source**: Stage 6 K-means cluster analysis + Stage 1 metadata
   - **Purpose**: Calculate "Videos using this pattern vs Videos NOT using this pattern" comparison
   - **Process**:
     1. Load Stage 6 K-means cluster assignments for the formula's bucket
     2. Identify top cluster (videos using this pattern) vs bottom cluster (videos not using pattern)
     3. For each cluster, calculate avg_views from video metadata
     4. Calculate multiplier: `top_cluster_avg / bottom_cluster_avg`
     5. Format multiplier: "1.6x MORE VIEWS"
   - **Outputs**:
     - Top cluster avg views (formatted with K/M)
     - Bottom cluster avg views (formatted with K/M)
     - View multiplier (e.g., "1.6x", "2.7x")
   - **Example Implementation**:
     ```python
     def calculate_pattern_performance(bucket_data, formula_cluster_id):
         # Get videos in top cluster (using pattern)
         top_cluster_videos = [v for v in bucket_data if v['cluster_id'] == formula_cluster_id]
         top_avg_views = sum([v['views'] for v in top_cluster_videos]) / len(top_cluster_videos)

         # Get videos in bottom cluster (not using pattern)
         bottom_cluster_videos = [v for v in bucket_data if v['cluster_id'] != formula_cluster_id]
         bottom_avg_views = sum([v['views'] for v in bottom_cluster_videos]) / len(bottom_cluster_videos)

         # Calculate multiplier
         multiplier = top_avg_views / bottom_avg_views

         return {
             'top_cluster_avg_views': format_views(top_avg_views),  # "620K"
             'bottom_cluster_avg_views': format_views(bottom_avg_views),  # "380K"
             'multiplier': f"{multiplier:.1f}x"  # "1.6x"
         }
     ```

2. **Contrastive Behavioral Analysis (Top vs Bottom Behaviors)**
   - **Source**: Stage 7 `winning_formulas.json` + Stage 2.7 Content Analysis
   - **Purpose**: Extract 3-5 specific behaviors that differentiate top from bottom performers
   - **Process**:
     1. Load formula data from Stage 7 (contains top cluster characteristics)
     2. Load Content Analysis data for videos in top vs bottom clusters
     3. Identify differentiating behaviors (quantitative + qualitative):
        - Quantitative: From temporal windows (e.g., "3.2 questions in hook" vs "0.8 questions")
        - Qualitative: From Content Analysis (e.g., "uses problem_solution hook" vs "uses direct_statement")
     4. Format as ✅ vs ❌ checklist (3-5 items each)
   - **Outputs**:
     - Top performer behaviors array (3-5 items with metrics)
     - Bottom performer behaviors array (3-5 items with metrics)
   - **Data Sources**:
     - Temporal windows: `hook_word_count`, `energy_level`, `text_overlay_count`, `scene_change_count`, etc.
     - Content Analysis: `hook_strategy`, `content_tactics`, `engagement_drivers`

3. **3-Phase Pattern Blueprint Extraction**
   - **Source**: Stage 7 `winning_formulas.json` + Stage 2.7 Content Analysis + Temporal Windows
   - **Purpose**: Generate Hook/Middle/Closing execution guide with specific instructions
   - **Process**:
     1. **Phase 1 (Hook 0-3s)**:
        - Extract `hook_strategy` from Content Analysis (e.g., "problem_solution")
        - Extract hook-specific temporal metrics (word_count, energy, close_ratio)
        - Generate instruction: "Use problem_solution hook (avg 3.2 questions in first 2s)"
     2. **Phase 2 (Middle - flexible timing)**:
        - Extract `pain_points`, `keywords`, `engagement_drivers`, `content_tactics` from Content Analysis
        - Format as checklist: "Include these elements in natural order"
        - Extract aggregated middle metrics (scene_changes, text_overlays, energy)
     3. **Phase 3 (Closing - last 3s)**:
        - Extract `cta_type` from Content Analysis caption_analysis
        - Extract closing temporal metrics (energy_max, has_speech_cta)
        - Generate instruction: "End with link_in_bio CTA"
   - **Outputs**:
     - Phase 1 instructions (1-2 sentences + metrics)
     - Phase 2 checklist (4-6 content elements) + execution standards
     - Phase 3 instructions (1-2 sentences + metrics)

4. **Pattern Name Generation**
   - **Source**: Stage 7 formula metadata OR Stage 2.7 Content Analysis dominant patterns
   - **Options**:
     - **Option A**: Stage 7 already provides `pattern_name` field → Use directly
     - **Option B**: Generate from Content Analysis fields (e.g., `hook_strategy` + `content_category` = "The Problem-Solution Recipe Tutorial")
   - **Process** (if Option B):
     1. Extract dominant `content_category` (e.g., "recipe_tutorial")
     2. Extract dominant `hook_strategy` (e.g., "problem_solution")
     3. Format as: "The [Hook Strategy] [Content Category]"
     4. Clean up naming (title case, readable)
   - **Output**: Pattern name string (e.g., "The Question Hook Formula")

5. **QR Code Generation & Video Mapping**
   - **Source**: Stage 2 Apify metadata + Stage 6 cluster assignments
   - **Purpose**: Generate 2 QR codes (top performer + bottom performer examples)
   - **Process**:
     1. Identify top cluster video with highest views (newest if multiple)
     2. Identify bottom cluster video with sufficient views for contrast
     3. Extract video URLs from Stage 2 metadata
     4. Extract video view counts for labels
     5. Generate QR codes using Python `qrcode` library
   - **Video Selection Criteria**:
     - Priority 1: Highest views in cluster
     - Priority 2: Newest video (timestamp) - reduces deletion risk
     - Priority 3: Videos from accounts with 100K+ followers (stability)
   - **Outputs**:
     - Top performer: QR code image + URL + view count + video_id
     - Bottom performer: QR code image + URL + view count + video_id
   - **Technical Specs**:
     - QR size: 1" × 1" (easily scannable on mobile)
     - Error correction: Medium (~15% damage tolerance)
     - File size: ~5KB per code

6. **Pre-Post Checklist Generation (5-7 Items)**
   - **Source**: Formula-specific behaviors from Phase 1/2/3 extraction
   - **Process**:
     1. Extract 1-2 critical hook behaviors (from Phase 1)
     2. Extract 2-3 critical middle behaviors (from Phase 2)
     3. Extract 1-2 critical closing behaviors (from Phase 3)
     4. Format as checkbox items with specific metrics
     5. Total: 5-7 items
   - **Example Output**:
     ```
     □ Question in first 2 seconds? (Pattern avg: 3.2 questions)
     □ Product visible by 5 seconds? (90% of top performers show by second 5)
     □ 5-7 text overlays placed? (Pattern avg: 6 overlays)
     □ 2-3 scene changes in middle? (Pattern avg: 2.8 changes)
     □ Energy increases in closing? (Pattern: 0.85 → 0.92 energy spike)
     □ Clear CTA at end? (85% use link_in_bio CTA)
     ```
   - **Grouping**: Checklist organized by phase (Hook → Middle → Closing)

7. **Caption Structure Extraction**
   - **Source**: Stage 2.7 Content Analysis → `caption_analysis` object
   - **Process**:
     1. Extract `hook_type` (e.g., "question")
     2. Extract `cta_type` (e.g., "link_in_bio")
     3. Extract `emoji_usage` (e.g., "some")
     4. Extract `hashtag_count` average (e.g., 7)
     5. Extract `hashtag_placement` (e.g., "end")
     6. Format as structured guidance
   - **Output**:
     ```
     Hook: Start with question ("Did you know...")
     Body: 2-3 sentences explaining benefit
     CTA: Link in bio reference
     Emojis: Use 3-5 emojis throughout (not excessive)
     Hashtags: 7 hashtags at end (mix of broad + niche)
     ```

---

## 3. Handle/Single Competitor → Client

**Status**: ✅ **COMPLETE**

**Audience**: Tumi Labs clients (business owners)

**Purpose**: Competitive intelligence on 1 competitor

**Deliverable**: 1 PDF analyzing 1 competitor

**Format**: 4-page PDF (desktop-optimized, executive-focused)

**Reading Time**: 8-10 minutes (scannable in 3 minutes)

---

### Input Data Sources

- Competitor Stage 7: `winning_formulas.json`
- Competitor Stage 6: `rf_video_analysis.json`, `kmeans_analysis.json`
- Competitor Stage 1: `winner_analysis.json` (bucket distribution)
- Competitor Stage 2: Video metadata (URLs, view counts, hashtags, timestamps, `likes`, `comments`, `shares`, `saves` for engagement calculation)
- Competitor Stage 2.7: `content_analysis` outputs (content categories, hook strategies)
- Config: CLI parameters (`--competitor`, `--analysis-period`)

---

### Design Decisions Locked

- ✅ Page count: 4 pages
- ✅ Analysis period: Last 90 days
- ✅ Hashtag depth: Top 10 hashtags
- ✅ Content category: Competitor only (no side-by-side)
- ✅ QR codes: 1 code (competitor's top video)
- ✅ Data type: Single snapshot analysis
- ✅ Comparison approach: Competitor focus only (no client comparison)

---

### Page 1: Competitive Overview & Posting Activity

**Purpose**: Establish analysis scope, show competitor's posting behavior

---

#### Header Section

```
Competitive Intelligence Report
Competitor: @rival_brand
Analysis Period: Last 90 days
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handle | Config | `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/config.json` → `target` (includes @ symbol) | String | "@drinkpoppi" | ✅ **This session** |
| Analysis period | Static | Fixed string: "Last 90 days" | String | "Last 90 days" | ✅ **Report 1 Header** |

---

#### Analysis Scope

```
ANALYSIS SCOPE:
Videos Analyzed: 127
Total Video Duration: 42 minutes
Duration Range: 0-120 seconds (8 distinct buckets)
Content Elements Tracked: 60+ features per video

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
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Videos analyzed | Selection Manifest | `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/selection_manifest.json` → Sum all `top_performers` + `bottom_performers` | Integer | 127 | ✅ **Report 1 Header** |
| Total duration | Temporal Windows | For each video ID in selection_manifest, load temporal_windows_updated.json → metadata.duration, sum and convert to minutes | String | "42 minutes" | ⚠️ **NOT VERIFIED** |

---

#### Posting Activity Intelligence

```
POSTING FREQUENCY:
14 videos per week (average over last 90 days)

POSTING CONSISTENCY:
High (posts 12-16 videos weekly, low variance)

CONTENT VELOCITY:
Recent 30 days: 16 videos/week (accelerating)
Prior 60 days: 13 videos/week
→ 23% increase in posting rate

ANALYSIS PERIOD COVERAGE:
127 videos analyzed (from 180 total posted in 90 days)
Coverage: Top 70% of content by engagement
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Posting frequency | Stage 2 | Count videos in last 90 days ÷ 13 weeks | Float | 14 videos/week | ⚠️ **NOT VERIFIED** |
| Posting consistency | Calculated | Weekly variance (Low/Medium/High based on std deviation) | String | "High" | ⚠️ **NOT VERIFIED** |
| Recent velocity (30 days) | Stage 2 | Count videos in last 30 days ÷ 4.3 weeks | Float | 16 videos/week | ⚠️ **NOT VERIFIED** |
| Prior velocity (60 days) | Stage 2 | Count videos in days 31-90 ÷ 8.6 weeks | Float | 13 videos/week | ⚠️ **NOT VERIFIED** |
| Velocity change | Calculated | (Recent - Prior) / Prior × 100% | Integer (%) | 23% | ⚠️ **NOT VERIFIED** |
| Total posted | Stage 2 | Count all videos in 90-day period | Integer | 180 | ⚠️ **NOT VERIFIED** |
| Videos analyzed | Selection Manifest | Sum all `top_performers` + `bottom_performers` | Integer | 127 | ✅ **Report 1 Header** |
| Coverage description | Config | Based on `--mode` (e.g., "Top 70% by engagement") | String | "Top 70% of content by engagement" | ⚠️ **NOT VERIFIED** |

---

### Page 2: Content Strategy & Hashtag Intelligence

**Purpose**: Show where competitor focuses content efforts and hashtag strategy

---

#### Section 1: Bucket Strategy (Content Distribution)

```
CONTENT DISTRIBUTION BY DURATION:

[Horizontal bar chart showing % of videos per bucket]

0-3s:   ██ 3%
3-9s:   ████ 8%
9-13s:  ████████ 12%
13-18s: ████████████ 18%  ← MODERATE VOLUME
18-33s: ████████████████████ 32%  ← HIGH VOLUME
33-60s: ██████████████ 22%  ← MODERATE VOLUME
60-90s: ███ 4%
90-120s: █ 1%

Key Insight: 52% of content concentrated in 18-33s + 33-60s buckets
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| % per bucket (8 rows) | Stage 1 | `bucket_distribution` in winner_analysis.json | Integer (%) | 3, 8, 12, 18, 32, 22, 4, 1 | ⚠️ **NOT VERIFIED** |
| Key insight | Calculated | Sum of top 2 buckets percentages + bucket names | String | "52% of content in 18-33s + 33-60s" | ⚠️ **NOT VERIFIED** |

---

#### Section 2: Bucket Performance (Top 3 Winning Buckets)

```
PERFORMANCE BY DURATION:

Duration | Avg Views | Avg Engagement | Rating
---------|-----------|----------------|--------
13-18s   | 580K      | 1.3%           | ⭐⭐⭐⭐
18-33s   | 620K      | 1.5%           | ⭐⭐⭐⭐⭐  ← SWEET SPOT
33-60s   | 490K      | 1.4%           | ⭐⭐⭐⭐

(Other buckets: 150K-380K views)

Sweet Spot: 18-33s (highest views + highest engagement + high volume)

KEY INSIGHT:
→ 18-33s duration delivers best performance (620K avg views, 1.5% engagement)
→ Consistent engagement rates across top 3 buckets (1.3-1.5%)
→ Strong content resonance in mid-length formats (13-60s)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Avg views per bucket | Stage 1 | `avg_views` per bucket in winner_analysis.json | Integer (formatted with K) | 580K, 620K, 490K | ⚠️ **NOT VERIFIED** |
| Avg engagement per bucket | Calculated | Avg `calculate_engagement_metrics()` per bucket | Float (%) | 1.3, 1.5, 1.4 | ⚠️ **NOT VERIFIED** |
| Star ratings | Calculated | Based on view + engagement performance (5 stars = high both) | String (emoji) | ⭐⭐⭐⭐⭐ | ⚠️ **NOT VERIFIED** |
| Sweet spot | Calculated | Bucket with highest views + highest engagement + high volume | String | "18-33s" | ⚠️ **NOT VERIFIED** |

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

#### Section 4: Caption Strategy Intelligence

**Purpose**: Analyze competitor's caption formatting and CTA strategies

```
COMPETITOR'S CAPTION STRATEGY:

Avg Hashtag Count:        12 hashtags per video
Hashtag Strategy:         8 broad, 4 niche (broad-reach focused)
Caption Length:           Long captions (68% of content >100 chars)
Emoji Usage:              Heavy emoji use (45% use 5+ emojis)
Top CTA Type:             "Follow me" (52% of videos)

STRATEGIC APPROACH:
→ Broad-reach hashtag strategy (8 broad hashtags prioritize discovery)
→ Longer captions suggest storytelling/educational approach
→ Heavy emoji use indicates engaging, personality-driven style
→ "Follow" CTA prioritizes audience building over link clicks
→ Overall strategy: Growth-focused (maximize followers and reach)

CAPTION FORMULA PATTERN:
Typical structure: [Engaging opener with emojis] + [Long-form story/explanation] +
[12 hashtags: 8 broad for reach, 4 niche for targeting] + ["Follow for more!" CTA]
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor avg hashtag count | Stage 2.7 (Comp) | Mean of `caption_analysis.hashtag_count` from competitor's winning videos | Integer | 12 |
| Competitor hashtag breakdown | Stage 2.7 (Comp) | Mean of `caption_analysis.hashtag_strategy.niche_count` and `broad_count` | String | "8 broad, 4 niche" |
| Competitor hashtag focus | Calculated | Identify whether broad or niche dominates | String | "broad-reach focused" |
| Competitor caption length winner | Stage 2.7 (Comp) | Most common `caption_analysis.caption_length` value with % | String with % | "Long (68%)" |
| Competitor emoji usage winner | Stage 2.7 (Comp) | Most common `caption_analysis.emoji_usage` value with % | String with % | "Heavy (5+) 45%" |
| Competitor top CTA | Stage 2.7 (Comp) | Most common `caption_analysis.caption_cta_type` with % | String with % | "Follow (52%)" |
| Strategic approach insights (4-5 items) | Calculated | Interpret competitor's caption strategy characteristics | String (array) | ["Broad-reach hashtag strategy...", "Longer captions suggest...", etc.] |
| Overall strategy summary | Calculated | High-level strategy label based on caption patterns | String | "Growth-focused (maximize followers and reach)" |
| Caption formula pattern | Calculated | Describe typical caption structure based on aggregated data | String | "[Engaging opener...] + [Long-form story...] + [12 hashtags...] + [CTA]" |

**Data Source**:
- Stage 2.7 Content Analysis classifications from competitor's winning videos (40 per bucket × 3 buckets = 120 videos)
- Aggregated using `aggregate_content_intelligence()` function

**Note**: Caption analysis provides competitive intelligence about how competitor formats captions, uses hashtags, and drives engagement. No client comparison - pure competitor analysis.

---

#### Section 5: Content Sourcing Strategy

**Purpose**: Identify affiliate partnerships and repost content strategy

```
CONTENT SOURCING STRATEGY:

Original Content: 58% (no affiliate mentions or repost indicators)
Reposted/Affiliate Content: 42% (contains repost indicators)

TOP AFFILIATE CONTRIBUTORS:
1. @fitnessguru123     (18% of videos - 54 mentions)
2. @healthcoach_jane   (12% of videos - 36 mentions)
3. @nutritionpro       (8% of videos - 24 mentions)
4. @wellnesswarrior    (5% of videos - 15 mentions)
5. @cleaneatingclub    (4% of videos - 12 mentions)

Total unique @mentions: 47
Videos with @mentions: 126 out of 300 (42%)

STRATEGY INSIGHT:
Competitor leverages affiliate network heavily - 42% of content is reposted
from 5 core partners. This allows them to maintain high posting frequency
(14 videos/week) with reduced production costs.

YOUR OPPORTUNITY:
Build similar affiliate partnerships to increase content volume without
proportional production cost increase. Consider developing UGC (user-generated
content) network with micro-influencers in your niche.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Original content % | Calculated | 100% - repost_rate | Integer (%) | 58 |
| Reposted/Affiliate content % | Calculated | `repost_rate` from extract_mention_analysis() | Integer (%) | 42 |
| Top affiliate contributors (5-10 items) | Stage 2 (Comp) | `top_10_mentions` from extract_mention_analysis() | Array of objects | [{handle: "@fitnessguru123", percentage: 18, mention_count: 54}, ...] |
| Total unique mentions | Calculated | `total_unique_mentions` from extract_mention_analysis() | Integer | 47 |
| Videos with mentions | Calculated | `videos_with_mentions` from extract_mention_analysis() | Integer | 126 |
| Total videos analyzed | Stage 1 (Comp) | Count from selection_manifest.json | Integer | 300 |
| Mention rate | Calculated | `mention_rate` from extract_mention_analysis() | Integer (%) | 42 |
| Competitor posting frequency | Stage 2 (Comp) | From posting activity analysis | Float | 14 videos/week |
| Strategy insight | Calculated | Synthesize based on repost_rate and top affiliates | String | "Competitor leverages affiliate network heavily..." |
| Opportunity recommendation | Manual | Strategic recommendation based on competitor's approach | String | "Build similar affiliate partnerships..." |

**Data Source**:
- `unified_analysis/{video_id}.json` → `metadata.description` (caption text)
- Regex extraction: `re.findall(r'@(\w+)', caption)`
- Repost indicators: ["repost", "via", "credit", "by", "from"]

**Implementation**: See Stage8MVP.md Section 0.5.4 for `extract_mention_analysis()` function

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

#### Section 2: Content Strategy Profile

**Purpose**: Show competitor's content approach, messaging themes, and audience targeting

```
COMPETITOR'S CONTENT STRATEGY:

Top Content Types:
1. Recipe Tutorial (38% of content) - Step-by-step cooking instructions
2. Wellness Practice (28% of content) - Daily health routines and habits
3. Supplement Review (17% of content) - Product recommendations and reviews
4. Expert Interview (12% of content) - Professional perspectives
5. Personal Testimony (5% of content) - Personal success stories

Top Hook Strategies:
1. Question Hook (42% of videos) - Opens with engaging question
2. Problem-Solution (31% of videos) - Identifies pain point, offers solution
3. Direct Statement (18% of videos) - Bold claim or fact
4. Curiosity Gap (9% of videos) - Creates mystery or intrigue

Audience Pain Points Addressed:
• Bloating/Digestive Issues (48% of videos)
• Low Energy/Fatigue (42% of videos)
• Weight Management (38% of videos)
• Inflammation (32% of videos)
• Gut Health (28% of videos)

Top Keywords Used:
#guthealth, #protein, #antiinflammatory, #metabolism, #fiber

Top Engagement Drivers:
• Before/After Reveal (45% of videos) - Visual transformations
• Specific Metrics (42% of videos) - "Lost 15 lbs in 30 days"
• Personal Testimony (38% of videos) - "This worked for me..."
• Expert Credentials (28% of videos) - "Registered nutritionist here..."

---

Note: Content categories and patterns discovered from competitor's content using
AI analysis. Categories reflect competitor-specific content approach and may differ
from other creators' content taxonomies.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Top content categories (5 types) | Stage 2.7 (Comp) | Aggregate `content_category` field, rank by frequency, top 5 with descriptions | String (array) with descriptions | ["Recipe Tutorial (38%) - Step-by-step...", ...] |
| Top hook strategies (4 types) | Stage 2.7 (Comp) | Aggregate `hook_strategy` field, rank by frequency, top 4 with descriptions | String (array) with descriptions | ["Question Hook (42%) - Opens with...", ...] |
| Top pain points (5 items) | Stage 2.7 (Comp) | Aggregate `pain_points` array, rank by frequency, top 5 with % | String (array) with % | ["Bloating/Digestive Issues (48%)", ...] |
| Top keywords (5 items) | Stage 2.7 (Comp) | Aggregate `keywords` array, rank by frequency, top 5 | String (array) | ["#guthealth", "#protein", ...] |
| Top engagement drivers (4 items) | Stage 2.7 (Comp) | Aggregate `engagement_drivers` array, rank by frequency, top 4 with % and descriptions | String (array) with % | ["Before/After Reveal (45%) - Visual...", ...] |

---

#### Section 3: Pattern Versatility

```
CREATIVE APPROACH:

Total distinct formulas: 9 (across all winning buckets)
Formula rotation: High (competitor uses 5-9 different patterns per bucket)
Pattern repetition rate: 24% (avg % of content using single most-used formula)

Insight: Competitor diversifies creative approach, avoiding pattern fatigue

Content Focus: Instructional content dominates (recipe + wellness = 66%)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Total distinct formulas | Stage 7 (Comp) | Count unique formulas across all buckets in winning_formulas.json | Integer | 9 |
| Formula rotation level | Calculated | If formulas > 6: "High", 4-6: "Medium", <4: "Low" | String | "High" |
| Pattern repetition rate | Calculated | Avg of highest formula usage % across buckets | Integer (%) | 24 |
| Content focus insight | Calculated | Sum related categories from Section 2, identify theme | String | "Instructional content dominates..." |

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
   - **Source**: `selection_manifest.json` (for selected video IDs) + `unified_analysis/{video_id}.json` (for hashtag data)
   - **Process**:
     1. Load `selection_manifest.json` from pipeline run
     2. Extract all selected video IDs from `videos_by_bucket` (top_performers + bottom_performers across all buckets)
     3. For each selected video ID, load `unified_analysis/{video_id}.json` → `metadata.hashtags` array
     4. Extract hashtag names: `[h['name'] for h in hashtags_array if h.get('name')]`
     5. Aggregate frequency counter across all selected videos
     6. Calculate top 10 hashtags by usage count
     7. Calculate percentage: `(videos_with_hashtag / total_selected_videos) × 100%`
   - **Outputs**:
     - Top 10 hashtags with usage percentages
     - Total unique hashtags count
     - Average hashtags per video
     - Top 5 concentration percentage (avg usage % of top 5)
   - **Example Implementation**:
     ```python
     from collections import Counter

     def extract_hashtag_analysis(manifest_path):
         # Load manifest to get selected video IDs
         manifest = load_json(manifest_path)
         selected_video_ids = []
         for bucket, videos in manifest['videos_by_bucket'].items():
             selected_video_ids.extend(videos.get('top_performers', []))
             selected_video_ids.extend(videos.get('bottom_performers', []))

         # Extract hashtags from selected videos only
         hashtag_counter = Counter()
         total_hashtag_count = 0

         for video_id in selected_video_ids:
             unified_path = f"/unified_analysis/{video_id}.json"
             data = load_json(unified_path)
             hashtags_array = data.get('metadata', {}).get('hashtags', [])
             hashtag_names = [h['name'] for h in hashtags_array if h.get('name')]
             hashtag_counter.update(hashtag_names)
             total_hashtag_count += len(hashtag_names)

         # Calculate stats
         top_10 = hashtag_counter.most_common(10)
         total_videos = len(selected_video_ids)

         return {
             'top_10': [(f"#{tag}", round((count/total_videos)*100)) for tag, count in top_10],
             'total_unique_hashtags': len(hashtag_counter),
             'avg_hashtags_per_video': round(total_hashtag_count / total_videos)
         }
     ```

3. **@Mention Extraction (Affiliate/Repost Analysis)**
   - **Source**: `selection_manifest.json` (for selected video IDs) + `unified_analysis/{video_id}.json` (for caption data)
   - **Purpose**: Identify affiliate partnerships and repost content strategy
   - **Process**:
     1. Load `selection_manifest.json` from pipeline run
     2. Extract all selected video IDs from `videos_by_bucket`
     3. For each selected video ID, load `unified_analysis/{video_id}.json` → `metadata.description`
     4. Extract @mentions using regex: `re.findall(r'@(\w+)', caption)`
     5. Detect repost indicators in caption text: ["repost", "via", "credit", "by", "from"]
     6. Aggregate @mention frequency counter across all selected videos
     7. Calculate top 10 most-mentioned handles
     8. Calculate repost rate (videos with repost indicators / total videos)
   - **Outputs**:
     - Top 10 @mentions with percentages
     - Total unique @mentions count
     - Videos with @mentions count and percentage
     - Videos with repost indicators count and percentage (repost rate)
   - **Strategic Value**:
     - Reveals if competitor leverages affiliate/UGC network vs creating original content
     - Identifies top affiliate partners (e.g., "@fitnessguru123 in 18% of videos")
     - Shows content sourcing strategy (e.g., "42% reposted content, 58% original")
   - **Example Implementation**:
     ```python
     import re
     from collections import Counter

     def extract_mention_analysis(manifest_path):
         # Load manifest to get selected video IDs
         manifest = load_json(manifest_path)
         selected_video_ids = []
         for bucket, videos in manifest['videos_by_bucket'].items():
             selected_video_ids.extend(videos.get('top_performers', []))
             selected_video_ids.extend(videos.get('bottom_performers', []))

         # Extract @mentions from captions
         mention_counter = Counter()
         videos_with_mentions = 0
         repost_indicators = ['repost', 'via', 'credit', 'by', 'from']
         videos_with_reposts = 0

         for video_id in selected_video_ids:
             unified_path = f"/unified_analysis/{video_id}.json"
             data = load_json(unified_path)

             # Get caption/description
             caption = data.get('metadata', {}).get('description', '')
             if not caption:
                 continue

             # Extract @mentions using regex (TikTok handle format)
             mentions = re.findall(r'@(\w+)', caption)

             if mentions:
                 videos_with_mentions += 1
                 mention_counter.update(mentions)

                 # Check for repost indicators
                 caption_lower = caption.lower()
                 if any(indicator in caption_lower for indicator in repost_indicators):
                     videos_with_reposts += 1

         # Calculate stats
         top_10_mentions = mention_counter.most_common(10)
         total_videos = len(selected_video_ids)

         return {
             'top_10_mentions': [
                 {
                     'handle': f"@{handle}",
                     'mention_count': count,
                     'percentage': round((count / total_videos) * 100, 1)
                 }
                 for handle, count in top_10_mentions
             ],
             'total_unique_mentions': len(mention_counter),
             'videos_with_mentions': videos_with_mentions,
             'mention_rate': round((videos_with_mentions / total_videos) * 100, 1),
             'videos_with_repost_indicators': videos_with_reposts,
             'repost_rate': round((videos_with_reposts / total_videos) * 100, 1)
         }
     ```
   - **Report Section** (Optional - can be added to Page 2 or 3):
     ```
     CONTENT SOURCING STRATEGY:

     Original Content: 58% (no affiliate mentions or repost indicators)
     Reposted/Affiliate Content: 42% (contains repost indicators)

     TOP AFFILIATE CONTRIBUTORS:
     1. @fitnessguru123     (18% of videos - 54 mentions)
     2. @healthcoach_jane   (12% of videos - 36 mentions)
     3. @nutritionpro       (8% of videos - 24 mentions)
     4. @wellnesswarrior    (5% of videos - 15 mentions)
     5. @cleaneatingclub    (4% of videos - 12 mentions)

     STRATEGY INSIGHT:
     Competitor leverages affiliate network heavily - 42% of content is reposted
     from 5 core partners. This allows them to maintain high posting frequency
     (14 videos/week) with reduced production costs.

     YOUR OPPORTUNITY:
     Build similar affiliate partnerships to increase content volume without
     proportional production cost increase.
     ```

4. **Performance Gap Calculations**
   - Bucket-level view gaps (competitor - client)
   - Bucket-level percentage gaps
   - Identify biggest gap bucket

5. **Content Analysis Aggregations**
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

## 4. Handle/Multiple Competitor → Client (Market Intelligence Report)

**Status**: ✅ **COMPLETE**

**Audience**: Tumi Labs clients (business owners)

**Purpose**: Multi-competitor market intelligence - understand competitive landscape

**Deliverable**: 1 PDF analyzing 2-5 competitors (market intelligence only, no client comparison)

**Format**: 4-page PDF (desktop-optimized, executive-focused)

**Reading Time**: 10-12 minutes (scannable in 3 minutes)

---

### Input Data Sources

- Competitor Stage 1: `winner_analysis.json` (bucket distribution, avg views per bucket)
- Competitor Stage 2: Video metadata (posting frequency, total duration, hashtags, @mentions, `views`, `likes`, `comments`, `shares`, `saves` for engagement calculation)
- Competitor Stage 2.7: Content Analysis (categories, hooks, pain points, keywords, engagement drivers)
- Competitor Stage 6: `kmeans_analysis.json` (pattern diversity)
- Competitor Stage 7: `winning_formulas.json` (top formulas per competitor)
- Config: CLI parameters (`--competitors` list, `--analysis-period`)

---

### Design Decisions Locked

- ✅ Pure market intelligence (no client comparison)
- ✅ Competitors only: 2-5 handles analyzed
- ✅ Analysis period: Last 90 days (matches single competitor report)
- ✅ No engagement metrics (not available)
- ✅ Focus: What competitors are doing (not what client should do)

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

Rank | Handle            | Avg Views | Avg Engagement | Posting Freq | Videos Analyzed | Total Duration
-----|-------------------|-----------|----------------|--------------|-----------------|----------------
1    | @wellness_pro     | 580K      | 1.4%           | 16/week      | 145             | 68 minutes
2    | @rival_brand      | 520K      | 1.3%           | 14/week      | 127             | 42 minutes
3    | @fitness_guru     | 480K      | 1.2%           | 11/week      | 98              | 31 minutes

Market Leader: @wellness_pro (580K avg views, 1.4% engagement, highest posting frequency)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Competitor handles (all) | Config | Array of `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/config.json` → `target` fields (one per competitor) | String (array) | ["@drinkpoppi", "@nike", "@wellness_pro"] | ✅ **This session** |
| Avg Views (per competitor) | Stage 1 (All Comp) | Weighted avg of `avg_views` across all buckets | Integer (formatted with K) | 580K, 520K, 480K | ⚠️ **NOT VERIFIED** |
| Avg Engagement (per competitor) | Calculated | Avg `calculate_engagement_metrics()` across all videos | Float (%) | 1.4, 1.3, 1.2 | ⚠️ **NOT VERIFIED** |
| Posting Freq (per competitor) | Stage 2 (All Comp) | Count videos in 90 days ÷ 13 weeks | Float | 16, 14, 11 | ⚠️ **NOT VERIFIED** |
| Videos Analyzed (per competitor) | Selection Manifest (All Comp) | Per competitor: `/data/clients/{client}/competitors/{target}/{mode}_{strategy}/selection_manifest.json` → Sum all `top_performers` + `bottom_performers` | Integer | 145, 127, 98 | ✅ **Report 1 Header** |
| Total Duration (per competitor) | Temporal Windows (All Comp) | Per competitor: Sum video durations from temporal_windows_updated.json | String | "68 minutes", "42 minutes", "31 minutes" | ⚠️ **NOT VERIFIED** |
| Market leader | Calculated | Competitor with highest avg_views + engagement | String | "@wellness_pro" | ⚠️ **NOT VERIFIED** |

---

#### Analysis Scope

```
ANALYSIS SCOPE PER COMPETITOR:

@wellness_pro:
• Videos Analyzed: 145
• Total Duration: 68 minutes of content
• Duration Range: 0-120 seconds (8 buckets)
• Content Elements Tracked: 60+ features per video

@rival_brand:
• Videos Analyzed: 127
• Total Duration: 42 minutes of content
• Duration Range: 0-120 seconds (8 buckets)
• Content Elements Tracked: 60+ features per video

@fitness_guru:
• Videos Analyzed: 98
• Total Duration: 31 minutes of content
• Duration Range: 0-120 seconds (8 buckets)
• Content Elements Tracked: 60+ features per video

Analysis Method:
Multi-dimensional machine learning and AI content analysis applied to each
competitor's content to identify patterns, strategies, and creative formulas.
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Videos analyzed (per competitor) | Stage 1 (All Comp) | `total_videos_analyzed` | Integer | 145, 127, 98 |
| Total duration (per competitor) | Stage 2 (All Comp) | Sum of all video durations, formatted | String | "68 minutes", "42 minutes", "31 minutes" |

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

KEY MARKET INSIGHTS:
• All competitors focus heavily on 18-33s bucket (26-32% allocation)
• @wellness_pro invests most in long-form 33-60s content (30% vs 18-22%)
• @fitness_guru spreads content across mid-length 13-33s (48% combined)
• Market consensus: 18-33s is the primary battleground bucket
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Bucket % (per competitor × 8 buckets) | Stage 1 (All Comp) | `bucket_distribution` in winner_analysis.json | Integer (%) matrix | 2, 5, 8, 15, 28, 30, 9, 3 (per competitor) |
| High volume markers (per competitor) | Calculated | Flag buckets >20% per competitor | Boolean matrix | True/False per cell |
| Market pattern (per bucket) | Calculated | Categorize volume level based on competitor averages | String per bucket | "Low volume", "HIGH VOLUME", etc. |
| Key insights (3-4 items) | Calculated | Identify trends, differences, consensus patterns | String (array) | ["All competitors focus heavily on 18-33s...", ...] |

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

PERFORMANCE INSIGHTS:
• All competitors share 13-18s and 18-33s as winning buckets (market consensus)
• @wellness_pro excels in 9-13s content (unique winning bucket)
• @fitness_guru dominates 60-90s long-form (unique winning bucket)
• Common ground: 18-33s appears in all 3 competitors' winning buckets
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Unique buckets (union of all winning buckets) | Stage 1 (All Comp) | Union of `winning_buckets` across all competitors | String (array) | ["9-13s", "13-18s", "18-33s", "33-60s", "60-90s"] |
| Avg views (per competitor, winning buckets only) | Stage 1 (All Comp) | `avg_views` for each competitor's winning buckets | Integer (formatted with K) or "—" | 420K, 580K, "—", etc. |
| Avg engagement (per competitor, winning buckets only) | Calculated | Avg `calculate_engagement_metrics()` for winning buckets or "—" | Float (%) or "—" | 1.2, 1.3, "—", etc. |
| Winning bucket markers (per competitor) | Stage 1 (All Comp) | 👑 if bucket is in competitor's winning 3 | String (emoji) or blank | 👑 or blank |
| Best performer (per bucket) | Calculated | Among competitors with data, who has max views + engagement | String per bucket | "@wellness_pro", "@rival_brand (engagement wins tie)", etc. |
| Competitor winning bucket lists (3 per competitor) | Stage 1 (All Comp) | Top 3 buckets from `winning_buckets` in winner_analysis.json per competitor | String (array per competitor) | ["9-13s", "13-18s", "18-33s"] |
| Performance insights (4 items) | Calculated | Identify shared buckets, unique buckets, dominance patterns | String (array) | ["All competitors share 13-18s...", ...] |

**Option B Implementation Note**: Each competitor is analyzed independently and has their own top 3 winning buckets. The table shows the UNION of all winning buckets (typically 3-6 unique buckets), with "—" for competitors who don't have a particular bucket in their winning 3.

---

#### Section 3: Posting Frequency & Consistency

```
POSTING ACTIVITY (Last 90 Days):

Competitor        | Posting Freq | Consistency | Recent Velocity | Trend
------------------|--------------|-------------|-----------------|-------
@wellness_pro     | 16/week      | High        | 18/week (last 30d) | ↑ Accelerating
@rival_brand      | 14/week      | High        | 14/week (last 30d) | → Stable
@fitness_guru     | 11/week      | Moderate    | 12/week (last 30d) | ↑ Slight increase

MARKET VELOCITY:
• @wellness_pro is accelerating content production (+13% in last 30 days)
• @rival_brand maintains consistent 14 videos/week output
• @fitness_guru showing modest growth trajectory
• Market average: 13.7 videos/week
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Posting freq (per competitor) | Stage 2 (All Comp) | Count videos in 90 days ÷ 13 weeks | Float | 16, 14, 11 |
| Consistency (per competitor) | Calculated | Weekly variance analysis (Low/Moderate/High) | String | "High", "High", "Moderate" |
| Recent velocity (per competitor) | Stage 2 (All Comp) | Count videos in last 30 days ÷ 4.3 weeks | Float | 18, 14, 12 |
| Trend (per competitor) | Calculated | Compare recent velocity vs 90-day average | String | "↑ Accelerating", "→ Stable", "↑ Slight increase" |
| Market average | Calculated | Mean of all competitor posting frequencies | Float | 13.7 |

---

### Page 3: Creative Intelligence & Patterns

**Purpose**: Show winning formulas and content strategies from each competitor

---

#### Section 1: Top Creative Formulas (Best 2 Per Competitor)

```
WINNING FORMULAS BY COMPETITOR:

@wellness_pro (Market Leader - 580K avg views):
Formula 1: "The Transformation Journey" (33-60s bucket)
• Usage: 30% of competitor's 33-60s content uses this pattern
• Pattern: Before state → intervention/product → after results with specific metrics
• Key elements: Personal proof, before/after reveal, testimonial

Formula 2: "The Expert Interview Format" (18-33s bucket)
• Usage: 24% of competitor's 18-33s content uses this pattern
• Pattern: Question nutritionist/expert → get authoritative answers → actionable takeaways
• Key elements: Credentials shown, third-party validation, Q&A structure

@rival_brand:
Formula 1: "The Question Hook Recipe Tutorial" (18-33s bucket)
• Usage: 28% of competitor's 18-33s content uses this pattern
• Pattern: Question about ingredient/concern → demonstrate recipe → show nutritional benefits
• Key elements: Curiosity hook, step-by-step instruction, actionable content

Formula 2: "The Myth-Busting Reveal" (13-18s bucket)
• Usage: 25% of competitor's 13-18s content uses this pattern
• Pattern: State common myth → explain why it's wrong → provide correct information with source
• Key elements: Contrarian positioning, educational, credibility markers

@fitness_guru:
Formula 1: "The Quick Win Tutorial" (13-18s bucket)
• Usage: 32% of competitor's 13-18s content uses this pattern
• Pattern: Promise fast result → demonstrate simple technique → provide immediate takeaway
• Key elements: Low effort perception, high value, easy to replicate

Formula 2: "The Ingredient Deep-Dive" (18-33s bucket)
• Usage: 22% of competitor's 18-33s content uses this pattern
• Pattern: Focus on single ingredient → explain health benefits → show multiple uses/recipes
• Key elements: Educational depth, multiple applications, authority building
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor handles | Config | CLI parameter `--competitors` | String (array) | ["@wellness_pro", "@rival_brand", "@fitness_guru"] |
| Competitor avg views | Stage 1 (All Comp) | Weighted avg across buckets | Integer (formatted with K) | 580K, 520K, 480K |
| Formula names (2 per competitor) | Stage 7 (All Comp) | `pattern_name` from winning_formulas.json (top 2) | String (array per competitor) | ["The Transformation Journey", "The Expert Interview Format"] |
| Formula buckets | Stage 7 (All Comp) | `bucket_range` per formula | String per formula | "33-60s", "18-33s", etc. |
| Usage % | Calculated | (Videos using pattern / Total videos in bucket) × 100% | Integer (%) per formula | 30, 24, 28, 25, 32, 22 |
| Pattern descriptions | Stage 7 (All Comp) | LLM-generated summary from pattern characteristics | String per formula | "Before state → intervention..." |
| Key elements | Stage 7 (All Comp) | Extract 3 defining characteristics per formula | String (array, 3 items per formula) | ["Personal proof", "before/after reveal", "testimonial"] |

---

#### Section 2: Content Strategy Profiles

**Purpose**: Show each competitor's content approach, discovered independently

```
CONTENT STRATEGY BY COMPETITOR:

═══════════════════════════════════════════════════════════════
@wellness_pro
═══════════════════════════════════════════════════════════════

Top Content Types:
1. Wellness Practice (35% of content) - Daily health routines and holistic habits
2. Recipe Tutorial (25% of content) - Step-by-step healthy cooking instructions
3. Expert Interview (18% of content) - Professional nutritionist perspectives
4. Supplement Review (15% of content) - Product recommendations and analysis
5. Personal Testimony (7% of content) - Personal health journey stories

Top Hook Strategies:
1. Problem-Solution (35% of videos) - Identifies pain point, offers solution
2. Question Hook (38% of videos) - Opens with engaging health question
3. Direct Statement (20% of videos) - Bold health claim or fact
4. Curiosity Gap (7% of videos) - Creates intrigue or mystery

Audience Pain Points Addressed:
• Gut Health/Digestion (52% of videos)
• Low Energy/Fatigue (45% of videos)
• Inflammation (38% of videos)
• Weight Management (32% of videos)

Top Keywords: #guthealth, #wellness, #holistic, #naturalhealing, #inflammation

Top Engagement Drivers:
• Personal Testimony (45%) - "This worked for me..."
• Expert Credentials (42%) - "As a nutritionist..."
• Before/After Reveal (38%) - Visual transformations

Strategic Positioning: "The Wellness Authority" - Focuses on holistic health practices
with expert validation. Emphasizes lifestyle changes over quick fixes.

---

═══════════════════════════════════════════════════════════════
@rival_brand
═══════════════════════════════════════════════════════════════

Top Content Types:
1. Recipe Tutorial (38% of content) - Step-by-step cooking demonstrations
2. Ingredient Deep-Dive (32% of content) - Science behind specific ingredients
3. Product Demonstration (18% of content) - How products work and benefits
4. Nutrition Myth-Busting (12% of content) - Debunking common misconceptions

Top Hook Strategies:
1. Question Hook (42% of videos) - Engaging questions about food/nutrition
2. Problem-Solution (31% of videos) - Identifies issue, provides recipe solution
3. Direct Statement (18% of videos) - Nutritional facts and claims
4. Curiosity Gap (9% of videos) - Mysterious ingredient or method

Audience Pain Points Addressed:
• Bloating/Digestive Issues (58% of videos)
• Low Energy (48% of videos)
• Weight Loss (42% of videos)
• Food Sensitivities (35% of videos)

Top Keywords: #protein, #guthealth, #antiinflammatory, #metabolism, #cleaneating

Top Engagement Drivers:
• Before/After Reveal (52%) - Recipe transformations and results
• Specific Metrics (48%) - "30g of protein in this meal"
• Product Features (38%) - Ingredient highlights

Strategic Positioning: "The Recipe Educator" - Emphasizes practical cooking solutions
with scientific backing. Content is highly actionable and tutorial-focused.

---

═══════════════════════════════════════════════════════════════
@fitness_guru
═══════════════════════════════════════════════════════════════

Top Content Types:
1. Workout Tutorial (42% of content) - Exercise demonstrations and form tips
2. Transformation Story (26% of content) - Personal and client success stories
3. Supplement Review (18% of content) - Product recommendations for fitness
4. Nutrition Advice (14% of content) - Diet tips for performance and recovery

Top Hook Strategies:
1. Direct Statement (42% of videos) - Bold fitness/nutrition claims
2. Question Hook (35% of videos) - Fitness and diet questions
3. Problem-Solution (15% of videos) - Fitness challenges and solutions
4. Before/After Hook (8% of videos) - Transformation teasers

Audience Pain Points Addressed:
• Weight Management/Fat Loss (65% of videos)
• Muscle Gain (48% of videos)
• Low Energy (42% of videos)
• Performance/Strength (38% of videos)

Top Keywords: #fitness, #transformation, #weightloss, #musclegain, #workoutroutine

Top Engagement Drivers:
• Before/After Reveal (68%) - Body transformation visuals
• Specific Metrics (55%) - "Lost 15 lbs in 30 days"
• Personal Testimony (52%) - Personal fitness journey

Strategic Positioning: "The Transformation Motivator" - Focuses on results and
personal proof. Balances workout content with nutrition guidance for complete fitness.

═══════════════════════════════════════════════════════════════

Note: Each competitor analyzed independently with AI-discovered content taxonomy.
Categories reflect each competitor's unique content focus and are not directly
comparable due to different content specializations (wellness vs recipes vs fitness).
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor handles | Config | CLI parameter `--competitors` | String (array) | ["@wellness_pro", "@rival_brand", "@fitness_guru"] |
| Top content types (5 per competitor) | Stage 2.7 (All Comp) | Aggregate `content_category` per competitor, rank by frequency, top 5 with descriptions | String (array per competitor) with % | ["Wellness Practice (35%) - Daily health...", ...] |
| Top hook strategies (4 per competitor) | Stage 2.7 (All Comp) | Aggregate `hook_strategy` per competitor, rank by frequency, top 4 with descriptions | String (array per competitor) with % | ["Problem-Solution (35%) - Identifies...", ...] |
| Top pain points (4 per competitor) | Stage 2.7 (All Comp) | Aggregate `pain_points` array per competitor, top 4 with % | String (array per competitor) with % | ["Gut Health/Digestion (52%)", ...] |
| Top keywords (5 per competitor) | Stage 2.7 (All Comp) | Aggregate `keywords` array per competitor, top 5 | String (array per competitor) | ["#guthealth", "#wellness", ...] |
| Top engagement drivers (3 per competitor) | Stage 2.7 (All Comp) | Aggregate `engagement_drivers` array per competitor, top 3 with % | String (array per competitor) with % | ["Personal Testimony (45%)", ...] |
| Strategic positioning (per competitor) | Calculated | Synthesize positioning statement from dominant content types | String per competitor | "The Wellness Authority - Focuses on..." |

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

HASHTAG STRATEGY INSIGHTS:
• All competitors use #nutrition and #healthylifestyle in top 5 (market standard)
• @wellness_pro has most diversified approach (42 unique hashtags)
• @rival_brand shows most focused strategy (73% top 5 concentration)
• #guthealth appears in top 5 for 2 of 3 competitors (trending topic)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Hashtag metrics (per competitor) | Stage 2 (All Comp) | Extract from hashtag aggregation function | Object per competitor | {unique, avg_per_video, top5_conc, strategy_type} |
| Top 5 hashtags (per competitor) | Stage 2 (All Comp) | Aggregate hashtag frequency, rank by frequency, top 5 | Array of objects per competitor | [{tag, usage_percent}, ...] |
| Hashtag insights (4 items) | Calculated | Identify common patterns, diversity differences, trends | String (array) | ["All competitors use #nutrition...", ...] |

---

#### Section 4: Caption Strategy Comparison

**Purpose**: Compare caption formatting and CTA strategies across multiple competitors

```
CAPTION STRATEGY ACROSS COMPETITORS:

Metric                  | @wellness_pro | @rival_brand | @fitness_guru | Market Pattern
------------------------|---------------|--------------|---------------|----------------
Avg Hashtag Count       | 11            | 9            | 10            | Moderate (10 avg)
Hashtag Strategy        | 7 broad, 4 niche | 5 niche, 4 broad | 6 broad, 4 niche | Balanced approach
Caption Length          | Long (65%)    | Short (82%)  | Short (75%)   | Short dominates (74% avg)
Emoji Usage             | Some (68%)    | Some (72%)   | Many (58%)    | Light-to-moderate (66% avg)
Top CTA Type            | Follow (48%)  | Link bio (68%)| Follow (55%)  | Link bio consensus (57% avg)

CAPTION STRATEGY INSIGHTS:
• Caption length consensus: 74% of market uses short captions (<100 chars)
• CTA strategy split: 57% prioritize "link in bio", 43% drive followers
• Hashtag count: Consistent across competitors (9-11 hashtags per video)
• Emoji usage: Light-to-moderate dominates (66% use 1-4 emojis)
• @rival_brand shows most conversion-focused strategy (short captions + link bio CTA)

STRATEGIC DIFFERENTIATION:
• @wellness_pro: Longer captions + follower growth focus
• @rival_brand: Conversion-optimized (short + link bio)
• @fitness_guru: Engagement-heavy (more emojis + follower CTAs)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Avg hashtag count (per competitor) | Stage 2.7 (All Comp) | Mean of `caption_analysis.hashtag_count` per competitor | Integer per competitor | 11, 9, 10 |
| Hashtag breakdown (per competitor) | Stage 2.7 (All Comp) | Mean of `caption_analysis.hashtag_strategy.niche_count` and `broad_count` | String per competitor | "7 broad, 4 niche", "5 niche, 4 broad", etc. |
| Caption length winner (per competitor) | Stage 2.7 (All Comp) | Most common `caption_analysis.caption_length` with % | String with % per competitor | "Long (65%)", "Short (82%)", "Short (75%)" |
| Emoji usage winner (per competitor) | Stage 2.7 (All Comp) | Most common `caption_analysis.emoji_usage` with % | String with % per competitor | "Some (68%)", "Some (72%)", "Many (58%)" |
| Top CTA (per competitor) | Stage 2.7 (All Comp) | Most common `caption_analysis.caption_cta_type` with % | String with % per competitor | "Follow (48%)", "Link bio (68%)", "Follow (55%)" |
| Market pattern avg hashtag count | Calculated | Mean across all competitors | Integer | 10 |
| Market pattern hashtag strategy | Calculated | Describe dominant approach | String | "Balanced approach" |
| Market pattern caption length | Calculated | Most common caption_length across competitors with % | String with % | "Short dominates (74% avg)" |
| Market pattern emoji usage | Calculated | Most common emoji_usage across competitors with % | String with % | "Light-to-moderate (66% avg)" |
| Market pattern CTA | Calculated | Most common caption_cta_type across competitors with % | String with % | "Link bio consensus (57% avg)" |
| Caption insights (3 items) | Calculated | Identify market patterns, consensus, outliers | String (array) | ["Caption length consensus...", "CTA strategy split...", "Hashtag count consistent..."] |
| Strategic differentiation (3 items) | Calculated | Describe each competitor's unique caption approach | String (array) | ["@wellness_pro: Longer captions...", "@rival_brand: Conversion-optimized...", etc.] |

**Data Source**:
- Stage 2.7 Content Analysis classifications from all competitors' winning videos (40 per bucket × 3 buckets × N competitors)
- Aggregated using `aggregate_content_intelligence()` function per competitor
- Market patterns calculated by aggregating across all competitors

**Note**: Caption analysis fields are universal (standardized enums/integers), so they're directly comparable across all competitors without taxonomy dependency.

---

#### Section 5: Content Sourcing Strategy

**Purpose**: Show each competitor's affiliate partnerships and content sourcing approach

```
CONTENT SOURCING STRATEGY BY COMPETITOR:

═══════════════════════════════════════════════════════════════
@wellness_pro
═══════════════════════════════════════════════════════════════

Original Content: 72% (no affiliate mentions or repost indicators)
Reposted/Affiliate Content: 28% (contains repost indicators)

Top Affiliate Contributors:
1. @holistichealth_coach  (12% of videos - 36 mentions)
2. @wellness_collective   (8% of videos - 24 mentions)
3. @naturalremedies       (5% of videos - 15 mentions)
4. @ayurveda_lifestyle    (3% of videos - 9 mentions)

Total unique @mentions: 22
Sourcing Strategy: Mostly original content with selective affiliate partnerships

---

═══════════════════════════════════════════════════════════════
@rival_brand
═══════════════════════════════════════════════════════════════

Original Content: 58% (no affiliate mentions or repost indicators)
Reposted/Affiliate Content: 42% (contains repost indicators)

Top Affiliate Contributors:
1. @fitnessguru123       (18% of videos - 54 mentions)
2. @healthcoach_jane     (12% of videos - 36 mentions)
3. @nutritionpro         (8% of videos - 24 mentions)
4. @wellnesswarrior      (5% of videos - 15 mentions)
5. @cleaneatingclub      (4% of videos - 12 mentions)

Total unique @mentions: 47
Sourcing Strategy: Heavy affiliate network - 42% reposted content from 5 core partners

---

═══════════════════════════════════════════════════════════════
@fitness_guru
═══════════════════════════════════════════════════════════════

Original Content: 85% (no affiliate mentions or repost indicators)
Reposted/Affiliate Content: 15% (contains repost indicators)

Top Affiliate Contributors:
1. @transformationclub    (8% of videos - 24 mentions)
2. @fitnesstips_daily     (4% of videos - 12 mentions)
3. @workout_motivation    (3% of videos - 9 mentions)

Total unique @mentions: 12
Sourcing Strategy: Predominantly original content creator

═══════════════════════════════════════════════════════════════

MARKET INSIGHTS:
• Sourcing approach varies significantly: 15% - 42% reposted/affiliate content
• @rival_brand relies most heavily on affiliate network (42% reposted)
• @fitness_guru creates most original content (85% original)
• Affiliate partnerships enable higher posting frequency without proportional cost increase
• Average affiliate network size: 27 unique @mentions per competitor
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Competitor handles | Config | CLI parameter `--competitors` | String (array) | ["@wellness_pro", "@rival_brand", "@fitness_guru"] |
| Original content % (per competitor) | Calculated | 100% - repost_rate per competitor | Integer (%) per competitor | 72, 58, 85 |
| Reposted/Affiliate % (per competitor) | Calculated | `repost_rate` from extract_mention_analysis() per competitor | Integer (%) per competitor | 28, 42, 15 |
| Top affiliate contributors (per competitor) | Stage 2 (All Comp) | `top_10_mentions` from extract_mention_analysis(), show top 3-5 per competitor | Array per competitor | [{handle, percentage, mention_count}, ...] |
| Total unique mentions (per competitor) | Calculated | `total_unique_mentions` from extract_mention_analysis() per competitor | Integer per competitor | 22, 47, 12 |
| Sourcing strategy label (per competitor) | Calculated | Classify based on repost_rate: <20% "Predominantly original", 20-40% "Selective partnerships", >40% "Heavy affiliate network" | String per competitor | "Mostly original...", "Heavy affiliate...", etc. |
| Market insight - variance | Calculated | Range of repost rates across competitors | String | "15% - 42% reposted/affiliate content" |
| Market insight - highest reliance | Calculated | Competitor with max repost_rate | String | "@rival_brand relies most heavily (42%)" |
| Market insight - most original | Calculated | Competitor with min repost_rate | String | "@fitness_guru creates most original (85%)" |
| Market insight - avg network size | Calculated | Mean of total_unique_mentions across competitors | Integer | 27 |

**Data Source**:
- `unified_analysis/{video_id}.json` → `metadata.description` per competitor
- Regex extraction: `re.findall(r'@(\w+)', caption)`
- Repost indicators: ["repost", "via", "credit", "by", "from"]

**Implementation**: See Stage8MVP.md Section 0.5.4 for `extract_mention_analysis()` function

---

### Page 4: Audience Targeting & Visual Examples

**Purpose**: Show what topics competitors address and provide visual proof

---

#### Section 1: Pain Points & Keywords Analysis

```
WHAT PAIN POINTS COMPETITORS ADDRESS:

Pain Point               | @wellness_pro | @rival_brand | @fitness_guru | Market Prevalence
-------------------------|---------------|--------------|---------------|-------------------
Bloating/digestive issues| 52%           | 48%          | 35%           | High (45% avg)
Low energy/fatigue       | 38%           | 42%          | 55% 🥇        | High (45% avg)
Weight management        | 31%           | 28%          | 45%           | Moderate (35% avg)
Inflammation             | 24%           | 32%          | 22%           | Moderate (26% avg)
Gut health problems      | 48%           | 52% 🥇       | 18%           | Moderate (39% avg)

🥇 = Highest focus on this pain point

AUDIENCE INSIGHTS:
• Energy and digestion issues dominate across all competitors (35-55% coverage)
• @fitness_guru focuses most on energy/fatigue (55% of content)
• @rival_brand leads in gut health messaging (52% of content)
• Weight management is secondary focus for all (28-45%)


WHAT TOPICS/KEYWORDS COMPETITORS DOMINATE:

Keyword              | @wellness_pro | @rival_brand | @fitness_guru | Market Leader
---------------------|---------------|--------------|---------------|---------------
"gut health"         | 65%           | 68% 🥇       | 42%           | @rival_brand
"protein"            | 48%           | 54%          | 62% 🥇        | @fitness_guru
"anti-inflammatory"  | 52% 🥇        | 42%          | 28%           | @wellness_pro
"metabolism"         | 36%           | 38%          | 48%           | @fitness_guru
"fiber"              | 42%           | 31%          | 22%           | @wellness_pro

🥇 = Dominates this keyword

KEYWORD INSIGHTS:
• @rival_brand owns the "gut health" conversation (68% vs 42-65%)
• @fitness_guru leads in "protein" messaging (62% vs 48-54%)
• @wellness_pro focuses on "anti-inflammatory" content (52% vs 28-42%)
• All competitors emphasize "gut health" and "protein" (top 2 keywords market-wide)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Pain points (5 items) | Stage 2.7 (All Comp) | Aggregate `pain_points` array, count frequency per competitor | String (array) | ["Bloating/digestive issues", "Low energy/fatigue", ...] |
| Pain point % (per competitor × 5 pain points) | Calculated | (Videos mentioning pain point / Total videos) × 100% per competitor | Integer (%) matrix | 52, 38, 31, 24, 48 (per competitor) |
| Market prevalence (per pain point) | Calculated | Average % across all competitors + category (High/Moderate/Low) | String per pain point | "High (45% avg)", "Moderate (35% avg)", etc. |
| Audience insights (4 items) | Calculated | Identify dominant themes, leaders, secondary focuses | String (array) | ["Energy and digestion issues dominate...", ...] |
| Keywords (5 items) | Stage 2.7 (All Comp) | Aggregate `keywords` array, count frequency per competitor | String (array) | ["gut health", "protein", ...] |
| Keyword % (per competitor × 5 keywords) | Calculated | (Videos mentioning keyword / Total videos) × 100% per competitor | Integer (%) matrix | 65, 48, 52, 36, 42 (per competitor) |
| Market leader (per keyword) | Calculated | Competitor with max % per keyword | String per keyword | "@rival_brand", "@fitness_guru", etc. |
| Keyword insights (4 items) | Calculated | Identify ownership patterns, emphasis, market-wide trends | String (array) | ["@rival_brand owns the 'gut health' conversation...", ...] |

---

#### Section 2: Engagement Tactics Comparison

```
WHAT TACTICS COMPETITORS USE TO DRIVE ENGAGEMENT:

Tactic                      | @wellness_pro | @rival_brand | @fitness_guru | Most Used By
----------------------------|---------------|--------------|---------------|---------------
Before/after reveals        | 47%           | 42%          | 38%           | @wellness_pro
Personal testimony          | 41%           | 28%          | 52% 🥇        | @fitness_guru
Specific metrics mentioned  | 38%           | 45% 🥇       | 35%           | @rival_brand
Product recommendations     | 34%           | 38%          | 42%           | @fitness_guru
Expert credentials shown    | 42% 🥇        | 18%          | 12%           | @wellness_pro

🥇 = Uses tactic most frequently

ENGAGEMENT TACTICS INSIGHTS:
• Before/after reveals popular across all competitors (38-47% usage)
• @fitness_guru relies heavily on personal testimony (52% vs 28-41%)
• @wellness_pro differentiates with expert credentials (42% vs 12-18%)
• Product recommendations common across all (34-42%)
• @rival_brand emphasizes specific metrics/numbers (45% of content)
```

**Dynamic Fields**:
| Template Field | Source | JSON Field/Calculation | Data Type | Example |
|----------------|--------|------------------------|-----------|---------|
| Engagement tactics (5 items) | Stage 2.7 (All Comp) | Aggregate `engagement_drivers` array, count frequency | String (array) | ["Before/after reveals", "Personal testimony", ...] |
| Tactic % (per competitor × 5 tactics) | Calculated | (Videos using tactic / Total videos) × 100% per competitor | Integer (%) matrix | 47, 41, 38, 34, 42 (per competitor) |
| Most used by (per tactic) | Calculated | Competitor with max % per tactic | String per tactic | "@wellness_pro", "@fitness_guru", etc. |
| Tactics insights (5 items) | Calculated | Identify popularity, differentiation, commonalities | String (array) | ["Before/after reveals popular across...", ...] |

---

#### Section 3: Visual Proof (Top Performers)

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
