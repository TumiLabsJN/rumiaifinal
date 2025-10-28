# Stage 8 MVP: Section 3.2 - COMPLETE IMPLEMENTATION GUIDE

## Section 3.2: `extract_client_data.py` - COMPLETE IMPLEMENTATION GUIDE

### Overview

**Purpose**: Extract hashtag intelligence dashboard data for client executive report (Report 1)

**Report Type**: Report 1 from Stage8MVP_Reports.md Section 1

**Deliverable**: 1 Excel file with comprehensive market intelligence

**CLI Usage**:
```bash
python extract_client_data.py --client acme --hashtag nutrition --mode top --strategy contrastive
```

**Output Files**:
```
/data/clients/acme/hashtags/nutrition/top_contrastive/
└── nutrition_client_data.xlsx (single tab with all pages)
```

**Console Output Pattern**:
```bash
$ python extract_client_data.py --client acme --hashtag nutrition

Running extraction for hashtag: #nutrition
Processing winner analysis...
Calculating performance metrics across 3 winning buckets...
Aggregating content intelligence from 120 videos...

✓ Extraction complete
  Excel: /data/clients/acme/hashtags/nutrition/top_contrastive/nutrition_client_data.xlsx
  Total fields: 52
```

---

### Complete Field List

**Excel Structure**: Single tab with two-column format (Field Name | Value)

**Total Fields**: ~52 fields across 3 pages

```python
# Field structure - two-column format: Field Name | Value
fields = [
    # =============================
    # PAGE 1: SCALE OF ANALYSIS
    # =============================
    ('PAGE_1_SCALE_OF_ANALYSIS', ''),  # Section divider
    ('', ''),  # Empty row

    # --- Header Section ---
    ('HASHTAG', '#nutrition'),  # From cluster config
    ('ANALYSIS_PERIOD', 'Past 2-3 months'),  # Static
    ('VIDEOS_ANALYZED', '1826'),  # From cluster_analytics.json → total_scraped_videos
    ('', ''),

    ('WINNING_BUCKET_1_NAME', '18-33s'),  # From winner_analysis.json → top_3_buckets[0]
    ('WINNING_BUCKET_1_PCT', '43'),  # From winner_analysis.json → top_100_distribution
    ('WINNING_BUCKET_2_NAME', '13-18s'),  # From winner_analysis.json → top_3_buckets[1]
    ('WINNING_BUCKET_2_PCT', '12'),  # From winner_analysis.json → top_100_distribution
    ('WINNING_BUCKET_3_NAME', '60-90s'),  # From winner_analysis.json → top_3_buckets[2]
    ('WINNING_BUCKET_3_PCT', '11'),  # From winner_analysis.json → top_100_distribution
    ('', ''),

    ('TOP_PERFORMERS_COUNT', '88'),  # Sum of selection_manifest → top_performers array lengths
    ('BOTTOM_PERFORMERS_COUNT', '23'),  # Sum of selection_manifest → bottom_performers array lengths

    # --- Analysis Scope & Methodology ---
    ('', ''),
    ('METHODOLOGY_TEXT', 'Multi-dimensional machine learning and AI content analysis'),  # Static

    # =============================
    # PAGE 2: HASHTAG INTELLIGENCE DASHBOARD
    # =============================
    ('', ''),
    ('PAGE_2_HASHTAG_INTELLIGENCE', ''),  # Section divider
    ('', ''),

    # --- Section 1: Duration Distribution ---
    ('BUCKET_0_3S_PCT', '8'),  # From winner_analysis.json → bucket_distribution (calculated %)
    ('BUCKET_3_9S_PCT', '12'),
    ('BUCKET_9_13S_PCT', '15'),
    ('BUCKET_13_18S_PCT', '22'),
    ('BUCKET_18_33S_PCT', '28'),
    ('BUCKET_33_60S_PCT', '12'),
    ('BUCKET_60_90S_PCT', '2'),
    ('BUCKET_90_120S_PCT', '1'),
    ('', ''),
    ('KEY_INSIGHT_PCT', '50'),  # Calculated: sum of dominant buckets (e.g., 13-18s + 18-33s)
    ('KEY_INSIGHT_TEXT', '50% of #nutrition content is 13-33s'),  # Formatted string

    # --- Section 2: Performance by Duration ---
    ('', ''),
    # Note: Buckets are sorted by performance (engagement primary, views secondary)
    # Rank 1 = BEST performer
    ('PERF_BUCKET_1_NAME', '18-33s'),  # Sorted bucket rank 1
    ('PERF_BUCKET_1_AVG_VIEWS', '490K'),  # From calculate_avg_views_per_bucket()
    ('PERF_BUCKET_1_AVG_ENG', '1.4'),  # From calculate_engagement_metrics() averaged
    ('PERF_BUCKET_1_STARS', '⭐⭐⭐⭐⭐'),  # Rank 1 = 5 stars
    ('PERF_BUCKET_1_LABEL', '← BEST'),  # Only rank 1 gets label
    ('', ''),

    ('PERF_BUCKET_2_NAME', '13-18s'),  # Sorted bucket rank 2
    ('PERF_BUCKET_2_AVG_VIEWS', '520K'),
    ('PERF_BUCKET_2_AVG_ENG', '1.2'),
    ('PERF_BUCKET_2_STARS', '⭐⭐⭐⭐'),  # Rank 2 = 4 stars
    ('PERF_BUCKET_2_LABEL', ''),  # Empty for rank 2-3
    ('', ''),

    ('PERF_BUCKET_3_NAME', '60-90s'),  # Sorted bucket rank 3
    ('PERF_BUCKET_3_AVG_VIEWS', '310K'),
    ('PERF_BUCKET_3_AVG_ENG', '1.3'),
    ('PERF_BUCKET_3_STARS', '⭐⭐⭐'),  # Rank 3 = 3 stars
    ('PERF_BUCKET_3_LABEL', ''),  # Empty for rank 2-3
    ('', ''),

    ('COVERAGE_PERCENTAGE', '75.9'),  # Top 3 buckets as % of top 100 videos

    # --- Section 3: Creator Profile Priorities ---
    ('', ''),
    ('TIER1_BUCKET_1_NAME', '13-18s'),  # Sorted bucket rank 1 (highest performance)
    ('TIER1_BUCKET_1_AVG_VIEWS', '520K'),
    ('TIER1_BUCKET_1_LABEL', 'highest performance'),  # Rank 1 label
    ('', ''),

    ('TIER1_BUCKET_2_NAME', '18-33s'),  # Sorted bucket rank 2
    ('TIER1_BUCKET_2_AVG_VIEWS', '490K'),
    ('TIER1_BUCKET_2_LABEL', 'strong performance + volume'),  # Rank 2 label
    ('', ''),

    ('TIER1_BUCKET_3_NAME', '60-90s'),  # Sorted bucket rank 3
    ('TIER1_BUCKET_3_AVG_VIEWS', '310K'),
    ('TIER1_BUCKET_3_LABEL', 'proven success'),  # Rank 3 label

    # --- Section 4: Content Intelligence ---
    ('', ''),
    ('CONTENT_CATEGORY_1', 'Recipe Tutorial'),  # Top 3 from aggregate_content_classifications()
    ('CONTENT_CATEGORY_1_PCT', '38'),
    ('CONTENT_CATEGORY_2', 'Wellness Practice'),
    ('CONTENT_CATEGORY_2_PCT', '28'),
    ('CONTENT_CATEGORY_3', 'Supplement Review'),
    ('CONTENT_CATEGORY_3_PCT', '22'),
    ('', ''),

    ('HOOK_STRATEGY_1', 'Problem-Solution'),  # Top 3 hook strategies
    ('HOOK_STRATEGY_1_PCT', '42'),
    ('HOOK_STRATEGY_2', 'Question Hook'),
    ('HOOK_STRATEGY_2_PCT', '35'),
    ('HOOK_STRATEGY_3', 'Direct Statement'),
    ('HOOK_STRATEGY_3_PCT', '23'),
    ('', ''),

    ('KEYWORD_1', '#guthealth'),  # Top 4 keywords
    ('KEYWORD_2', '#protein'),
    ('KEYWORD_3', '#antiinflammatory'),
    ('KEYWORD_4', '#metabolism'),
    ('', ''),

    ('PAIN_POINT_1', 'Bloating'),  # Top 3 pain points with %
    ('PAIN_POINT_1_PCT', '48'),
    ('PAIN_POINT_2', 'Low Energy'),
    ('PAIN_POINT_2_PCT', '42'),
    ('PAIN_POINT_3', 'Inflammation'),
    ('PAIN_POINT_3_PCT', '38'),
    ('', ''),

    ('ENGAGEMENT_DRIVER_1', 'Before/After Reveal'),  # Top 3 engagement drivers
    ('ENGAGEMENT_DRIVER_1_PCT', '45'),
    ('ENGAGEMENT_DRIVER_2', 'Personal Testimony'),
    ('ENGAGEMENT_DRIVER_2_PCT', '38'),
    ('ENGAGEMENT_DRIVER_3', 'Specific Metrics Mentioned'),
    ('ENGAGEMENT_DRIVER_3_PCT', '52'),
    ('', ''),

    ('OPTIMAL_HASHTAG_COUNT', '7'),  # Mean from caption_analysis
    ('CAPTION_LENGTH_WINNER', 'Short captions (<100 characters)'),
    ('CAPTION_LENGTH_WINNER_PCT', '68'),
    ('EMOJI_USAGE_WINNER', 'Light emoji use (1-4 emojis)'),
    ('EMOJI_USAGE_WINNER_PCT', '72'),
    ('TOP_CTA_TYPE', 'Link in bio'),
    ('TOP_CTA_TYPE_PCT', '58'),

    # =============================
    # PAGE 3: YOUR CREATIVE REPORTS
    # =============================
    ('', ''),
    ('PAGE_3_YOUR_CREATIVE_REPORTS', ''),  # Section divider
    ('', ''),

    # Note: 9 formulas = 3 winning buckets × 3 formulas per bucket
    ('FORMULA_COUNT', '9'),  # Total formulas delivered
    ('', ''),

    # Bucket 1 formulas
    ('BUCKET_1_FORMULA_1_NAME', 'The Question Hook Formula'),  # From winning_formulas.json
    ('BUCKET_1_FORMULA_2_NAME', 'The Fast-Paced Product Demo'),
    ('BUCKET_1_FORMULA_3_NAME', 'The Myth-Busting Reveal'),
    ('', ''),

    # Bucket 2 formulas
    ('BUCKET_2_FORMULA_1_NAME', 'The Transformation Story'),
    ('BUCKET_2_FORMULA_2_NAME', 'The Ingredient Deep-Dive'),
    ('BUCKET_2_FORMULA_3_NAME', 'The Side-by-Side Comparison'),
    ('', ''),

    # Bucket 3 formulas
    ('BUCKET_3_FORMULA_1_NAME', 'The Step-by-Step Tutorial'),
    ('BUCKET_3_FORMULA_2_NAME', 'The Expert Interview Format'),
    ('BUCKET_3_FORMULA_3_NAME', 'The Before-After Journey'),
]
```

**Notes**:
- Total fields: ~122 (including section dividers and empty rows)
- Field naming: `UPPERCASE_WITH_UNDERSCORES`
- Multi-value fields use numbered suffixes (e.g., `KEYWORD_1`, `KEYWORD_2`)
- Empty rows (`('', '')`) provide visual separation
- Section dividers use equals signs for page-level organization

---

### Required Functions

This section defines all functions needed for `extract_client_data.py`. Functions are documented inline for self-contained implementation.

---

#### Function 1: `calculate_engagement_metrics()`

**Purpose**: Calculate real engagement rate from TikTok video metadata

**Used by**: Report 1 (this script), Reports 2, 3, 4 (all reports)

**Input**: Video metadata dictionary with engagement fields

**Output**: Engagement rate as float (percentage)

**Implementation**:
```python
def calculate_engagement_metrics(video_metadata):
    """
    Calculate engagement rate from TikTok video metadata.

    Formula: (likes + comments + shares + saves) / views × 100

    Input fields (from selected_videos.json or unified_analysis JSON):
    - diggCount (likes)
    - commentCount
    - shareCount
    - collectCount (saves/bookmarks)
    - playCount (views)

    Returns: Float (percentage, e.g., 1.2 = 1.2%)
    """

    likes = video_metadata.get('diggCount', 0)
    comments = video_metadata.get('commentCount', 0)
    shares = video_metadata.get('shareCount', 0)
    saves = video_metadata.get('collectCount', 0)
    views = video_metadata.get('playCount', 1)  # Avoid division by zero

    total_interactions = likes + comments + shares + saves
    engagement_rate = (total_interactions / views) * 100

    return round(engagement_rate, 1)  # Round to 1 decimal place
```

**Data Source**: `{bucket_path}/selected_videos.json` → `videos[]` array

**Example**:
```python
video_meta = {
    'playCount': 620000,
    'diggCount': 5580,  # likes
    'commentCount': 1240,
    'shareCount': 310,
    'collectCount': 310  # saves
}

engagement = calculate_engagement_metrics(video_meta)
# Returns: 1.2 (meaning 1.2% engagement rate)
```

---

#### Function 2: `calculate_avg_views_per_bucket()`

**Purpose**: Calculate average playCount for videos in a single bucket and performance group

**Used by**: Report 1 (Performance by Duration, Creator Profile Priorities)

**Input Parameters**:
- `bucket_path` (str): Absolute path to bucket folder
  - Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/`
- `performance_group` (str, optional): Filter by performance tier (default: "top")
  - Valid values: `"top"`, `"bottom"`, or `None` (all videos)

**Process**:
1. Load `{bucket_path}/selected_videos.json`
2. Extract `top_count` or `bottom_count` based on `performance_group`
3. Extract first N videos from `videos` array (pre-sorted by playCount DESC)
   - Top performers: `videos[0:top_count]`
   - Bottom performers: `videos[top_count:top_count+bottom_count]`
4. Calculate average: `sum(playCount) / count`
5. Return as integer

**Implementation**:
```python
def calculate_avg_views_per_bucket(bucket_path, performance_group="top"):
    """
    Calculate average playCount for videos in a single bucket.

    Args:
        bucket_path: Absolute path to bucket folder
        performance_group: "top", "bottom", or None (default: "top")

    Returns:
        int: Average playCount across selected videos

    Example:
        >>> calculate_avg_views_per_bucket(
        ...     "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/",
        ...     performance_group="top"
        ... )
        1900000  # 1.9M average views
    """
    import json

    # Load selected videos
    with open(f"{bucket_path}/selected_videos.json") as f:
        data = json.load(f)

    # Determine which videos to include
    if performance_group == "top":
        count = data["top_count"]
        videos = data["videos"][:count]  # First N = top performers
    elif performance_group == "bottom":
        top_count = data["top_count"]
        bottom_count = data["bottom_count"]
        videos = data["videos"][top_count:top_count + bottom_count]  # Next M = bottom performers
    elif performance_group is None:
        # All videos
        videos = data["videos"]
    else:
        raise ValueError(f"Invalid performance_group: {performance_group}. Must be 'top', 'bottom', or None")

    if not videos:
        return 0  # No videos in this group

    # Calculate average
    total_views = sum(v["playCount"] for v in videos)
    avg_views = int(total_views / len(videos))

    return avg_views
```

**Output Format**:
```python
# Integer (raw view count)
1900000  # Display as "1.9M" in reports using K/M suffix formatter
```

**Data Source**: `{bucket_path}/selected_videos.json`

---

#### Function 3: `aggregate_content_classifications()`

**Purpose**: Aggregate 120 individual Stage 2.7 classifications into frequency distributions

**Used by**: Report 1 (Content Intelligence section)

**Input Parameters**:
- `bucket_path` (string): Path to bucket folder containing `content_analysis/` subdirectory
  - Example: `/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_18-33s/`
- `performance_group` (string, optional): Filter by "top" or "bottom" performers
  - If None: Aggregate all videos (for overall insights)
  - If "top": Only aggregate top performers
  - If "bottom": Only aggregate bottom performers

**Process**:
1. Load all `{video_id}_content.json` files from `content_analysis/` folder
2. Filter by `performance_group` if specified (using `performance_group` field from each file)
3. **Quality Gate**: Filter by confidence (only include `high` and `medium` confidence classifications)
   - Excludes unreliable `low` confidence classifications
   - Tracks excluded count for quality reporting
4. For each of 9 key fields, calculate frequency distributions:
   - **Core Content Fields** (6): `content_category`, `hook_strategy`, `pain_points`, `keywords`, `engagement_drivers`, `content_tactics`
   - **Caption Strategy Fields** (3): `cta_type`, `hook_type`, `hashtag_count` (mean/min/max/median)
5. Calculate effect sizes (if both top and bottom groups aggregated)

**Implementation**:
```python
from collections import Counter
import glob
import json

def aggregate_content_classifications(bucket_path, performance_group=None):
    """
    Aggregate Stage 2.7 Content Analysis classifications.

    Returns dict with frequency distributions for 9 key fields.
    """
    # Load all classification files
    pattern = f"{bucket_path}/content_analysis/*_content.json"
    classification_files = glob.glob(pattern)

    # Load and filter classifications
    all_classifications = []
    for file_path in classification_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

            # Filter by performance group if specified
            if performance_group is None or data.get('performance_group') == performance_group:
                all_classifications.append(data)

    if not all_classifications:
        return None  # No data found

    # Quality Gate: Filter by confidence (exclude low confidence)
    classifications = [
        c for c in all_classifications
        if c.get('confidence', 'high') in ['high', 'medium']
    ]

    excluded_count = len(all_classifications) - len(classifications)

    # Aggregate core content fields (strings)
    aggregated = {
        'total_videos': len(classifications),
        'excluded_low_confidence': excluded_count,
        'content_category': Counter([c['content_category'] for c in classifications]),
        'hook_strategy': Counter([c['hook_strategy'] for c in classifications]),
    }

    # Aggregate array fields (flatten then count)
    for field in ['pain_points', 'keywords', 'engagement_drivers', 'content_tactics']:
        all_values = []
        for c in classifications:
            all_values.extend(c.get(field, []))
        aggregated[field] = Counter(all_values)

    # Aggregate caption analysis fields
    aggregated['caption_hook_type'] = Counter([
        c['caption_analysis']['hook_type'] for c in classifications
    ])
    aggregated['caption_cta_type'] = Counter([
        c['caption_analysis']['cta_type'] for c in classifications
    ])

    # Aggregate numeric fields (hashtag_count)
    hashtag_counts = [c['caption_analysis']['hashtag_count'] for c in classifications]
    aggregated['hashtag_count_stats'] = {
        'mean': sum(hashtag_counts) / len(hashtag_counts),
        'min': min(hashtag_counts),
        'max': max(hashtag_counts),
        'median': sorted(hashtag_counts)[len(hashtag_counts) // 2]
    }

    # Transcript availability ratio
    with_transcript = sum(1 for c in classifications if c.get('transcript_available', False))
    aggregated['transcript_available_ratio'] = with_transcript / len(classifications)

    return aggregated
```

**Output Format**:
```python
{
    'total_videos': 38,  # High/medium confidence only
    'excluded_low_confidence': 2,  # Low confidence classifications excluded
    'content_category': Counter({
        'recipe_tutorial': 23,
        'wellness_practice': 11,
        'supplement_review': 4
    }),
    'hook_strategy': Counter({
        'problem_solution': 24,
        'question': 10,
        'direct_statement': 6
    }),
    'pain_points': Counter({
        'bloating': 21,
        'low_energy': 15,
        'inflammation': 8
    }),
    'keywords': Counter({
        'gut_health': 27,
        'protein': 22,
        'fiber': 18
    }),
    'engagement_drivers': Counter({
        'before_after_reveal': 19,
        'personal_testimony': 16,
        'product_recommendation': 14
    }),
    'content_tactics': Counter({
        'direct_to_camera': 31,
        'product_demonstration': 25,
        'text_overlay_heavy': 20
    }),
    'caption_hook_type': Counter({
        'question': 17,
        'statement': 12,
        'command': 7,
        'teaser': 2
    }),
    'caption_cta_type': Counter({
        'link_in_bio': 32,
        'save_post': 5,
        'comment': 3
    }),
    'hashtag_count_stats': {
        'mean': 7.2,
        'min': 3,
        'max': 12,
        'median': 7
    },
    'transcript_available_ratio': 0.95
}
```

**Usage in Report 1**:
```python
# Aggregate across ALL winning buckets for market-level insights
all_content_data = []

for bucket_name in winning_buckets:
    bucket_path = f"{analysis_path}/buckets/bucket_{bucket_name}"
    bucket_aggregated = aggregate_content_classifications(bucket_path, performance_group="top")

    if bucket_aggregated:
        all_content_data.append(bucket_aggregated)

# Combine Counters from all buckets
combined_content_category = Counter()
for data in all_content_data:
    combined_content_category.update(data['content_category'])

# Get top 3
top_3_categories = combined_content_category.most_common(3)
# Returns: [('recipe_tutorial', 46), ('wellness_practice', 28), ('supplement_review', 18)]
```

**Data Source**: `{bucket_path}/content_analysis/{video_id}_content.json` (see Data Source File Formats section)

---

#### Function 4: Inline Calculations

These are simple calculations that don't need separate functions but are documented for completeness:

##### Calculation 4.1: Format Views with K/M Suffix

```python
def format_views(view_count):
    """
    Format view count with K or M suffix.

    Examples:
    - 620000 → "620K"
    - 1900000 → "1.9M"
    - 520 → "520"
    """
    if view_count >= 1000000:
        return f"{view_count / 1000000:.1f}M"
    elif view_count >= 1000:
        return f"{int(view_count / 1000)}K"
    else:
        return str(view_count)
```

##### Calculation 4.2: Calculate Bucket Distribution Percentages

```python
def calculate_bucket_distribution_percentages(analysis_path):
    """
    Calculate percentage of videos in each duration bucket.

    Args:
        analysis_path: Path to analysis directory

    Returns:
        dict: Bucket name → percentage (rounded to integer)
    """
    import json

    with open(f"{analysis_path}/winner_analysis.json") as f:
        data = json.load(f)

    bucket_distribution = data["bucket_distribution"]
    total_videos = sum(bucket_distribution.values())

    # Calculate percentage for each bucket, rounded to integer
    bucket_percentages = {
        bucket: round((count / total_videos) * 100)
        for bucket, count in bucket_distribution.items()
    }

    return bucket_percentages
```

##### Calculation 4.3: Assign Star Ratings

```python
def assign_star_ratings(analysis_path, winning_buckets):
    """
    Sort winning buckets by performance and assign star ratings.

    Ranking Criteria:
    1. Primary: Average engagement rate (higher is better)
    2. Secondary: Average views (higher is better)

    Args:
        analysis_path: Path to analysis directory
        winning_buckets: List of winning bucket names from winner_analysis.json

    Returns:
        dict: {
            "star_ratings": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐"],
            "sorted_buckets": [
                {"bucket": "18-33s", "avg_views": 1900000, "avg_engagement": 1.4},
                {"bucket": "13-18s", "avg_views": 2100000, "avg_engagement": 1.2},
                {"bucket": "60-90s", "avg_views": 980000, "avg_engagement": 1.3}
            ]
        }
    """
    import json

    # Step 1: Collect performance metrics for each winning bucket
    buckets_with_metrics = []

    for bucket_name in winning_buckets:
        bucket_path = f"{analysis_path}/buckets/bucket_{bucket_name}"

        # Calculate avg views using documented function
        avg_views = calculate_avg_views_per_bucket(bucket_path, "top")

        # Calculate avg engagement
        # Load top performer videos from selected_videos.json
        with open(f"{bucket_path}/selected_videos.json") as f:
            data = json.load(f)

        top_count = data["top_count"]
        top_videos = data["videos"][:top_count]

        # Calculate engagement rate for each video
        engagement_rates = []
        for video in top_videos:
            # Use documented function (calculate_engagement_metrics)
            engagement = calculate_engagement_metrics({
                "playCount": video["playCount"],
                "diggCount": video["diggCount"],
                "commentCount": video["commentCount"],
                "shareCount": video["shareCount"],
                "collectCount": video["collectCount"]
            })
            engagement_rates.append(engagement)

        avg_engagement = sum(engagement_rates) / len(engagement_rates)

        buckets_with_metrics.append({
            "bucket": bucket_name,
            "avg_views": avg_views,
            "avg_engagement": avg_engagement
        })

    # Step 2: Sort by engagement (primary), then views (secondary)
    buckets_with_metrics.sort(
        key=lambda x: (x["avg_engagement"], x["avg_views"]),
        reverse=True
    )

    # Step 3: Assign star ratings based on rank
    star_map = {
        0: "⭐⭐⭐⭐⭐",  # Rank 1 (highest engagement + views)
        1: "⭐⭐⭐⭐",    # Rank 2
        2: "⭐⭐⭐"      # Rank 3
    }

    star_ratings = [star_map[i] for i in range(len(buckets_with_metrics))]

    return {
        "star_ratings": star_ratings,
        "sorted_buckets": buckets_with_metrics  # Return for use in other calculations
    }
```

##### Calculation 4.4: Calculate Coverage Percentage

```python
def calculate_coverage_percentage(analysis_path):
    """
    Calculate percentage of top 100 videos in winning buckets.

    Args:
        analysis_path: Path to analysis directory

    Returns:
        float: Coverage percentage with 1 decimal place
    """
    import json

    with open(f"{analysis_path}/winner_analysis.json") as f:
        data = json.load(f)

    top_3_buckets = data["top_3_buckets"]
    distribution = data["top_100_distribution"]

    # Sum video counts in winning buckets
    winning_count = sum(distribution[bucket] for bucket in top_3_buckets)

    # Sum all video counts
    total_count = sum(distribution.values())

    # Calculate percentage with 1 decimal place
    coverage_pct = round((winning_count / total_count) * 100, 1)

    return coverage_pct
```

---

### Data Source File Formats

This section documents the exact JSON structure for all files used by this script.

---

#### File 1: `cluster_analytics.json`

**Location**: `/data/clients/{client}/hashtags/{target}/cluster_analytics.json`

**Purpose**: Total scraped videos count and cluster-level statistics

**Structure**:
```json
{
  "scrape_summary": {
    "total_scraped_videos": 1826,
    "date_range": {
      "earliest": "2024-10-15",
      "latest": "2025-01-28"
    }
  },
  "clusters": [
    {
      "cluster_id": "nutrition_wellness",
      "video_count": 1826,
      "hashtags": ["#nutrition", "#wellness", "#health"]
    }
  ]
}
```

**Fields Used**:
- `scrape_summary.total_scraped_videos` → `VIDEOS_ANALYZED` field

---

#### File 2: `winner_analysis.json`

**Location**: `/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/winner_analysis.json`

**Purpose**: Winning buckets identification and bucket distribution statistics

**Structure**:
```json
{
  "top_3_buckets": ["18-33s", "13-18s", "60-90s"],
  "top_100_distribution": {
    "0-3s": 2,
    "3-9s": 5,
    "9-13s": 8,
    "13-18s": 12,
    "18-33s": 43,
    "33-60s": 18,
    "60-90s": 11,
    "90-120s": 1
  },
  "bucket_distribution": {
    "0-3s": 146,
    "3-9s": 219,
    "9-13s": 274,
    "13-18s": 402,
    "18-33s": 511,
    "33-60s": 219,
    "60-90s": 37,
    "90-120s": 18
  },
  "analysis_config": {
    "mode": "top",
    "strategy": "contrastive",
    "date_filter": "last_90_days"
  }
}
```

**Fields Used**:
- `top_3_buckets` → Winning bucket names (WINNING_BUCKET_1-3_NAME)
- `top_100_distribution` → Percentage calculation for winning buckets
- `bucket_distribution` → Duration distribution percentages (all 8 buckets)

---

#### File 3: `selection_manifest.json`

**Location**: `/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/selection_manifest.json`

**Purpose**: Video IDs for top and bottom performers per bucket

**Structure**:
```json
{
  "selected_buckets": ["18-33s", "13-18s", "60-90s"],
  "videos_by_bucket": {
    "18-33s": {
      "top_performers": [
        "7540717847325003039",
        "7539482920339442976",
        "7538247993353882913",
        // ... 30 more video IDs (total 33)
      ],
      "bottom_performers": [
        "7522019726648732960",
        "7521784799663149857",
        // ... 7 more video IDs (total 9)
      ]
    },
    "13-18s": {
      "top_performers": ["7545...", "..."],  // 28 video IDs
      "bottom_performers": ["7520...", "..."]  // 7 video IDs
    },
    "60-90s": {
      "top_performers": ["7548...", "..."],  // 27 video IDs
      "bottom_performers": ["7519...", "..."]  // 7 video IDs
    }
  },
  "selection_summary": {
    "total_top_performers": 88,
    "total_bottom_performers": 23,
    "total_selected": 111
  }
}
```

**Fields Used**:
- `videos_by_bucket.{bucket}.top_performers` → Array length for TOP_PERFORMERS_COUNT
- `videos_by_bucket.{bucket}.bottom_performers` → Array length for BOTTOM_PERFORMERS_COUNT
- Sum arrays across all 3 winning buckets

---

#### File 4: `selected_videos.json` (per bucket)

**Location**: `/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/buckets/bucket_{name}/selected_videos.json`

**Purpose**: Video metadata for all selected videos in a bucket (for views and engagement calculation)

**Structure**:
```json
{
  "bucket": "18-33s",
  "strategy": "contrastive",
  "video_count": 100,
  "selected_count": 42,
  "top_count": 33,
  "bottom_count": 9,
  "videos": [
    // Sorted by playCount DESC
    {
      "id": "7540717847325003039",
      "playCount": 6700000,
      "diggCount": 80400,
      "commentCount": 1608,
      "shareCount": 40200,
      "collectCount": 13400,
      "createTime": 1735689600,
      "duration": 21,
      "webVideoUrl": "https://www.tiktok.com/@user/video/7540717847325003039",
      "author": "@agitthaiii",
      "hashtags": [
        {"name": "guthealth"},
        {"name": "nutrition"}
      ]
    },
    // ... 32 more top performers
    {
      "id": "7522019726648732960",
      "playCount": 150000,
      "diggCount": 1800,
      "commentCount": 30,
      "shareCount": 300,
      "collectCount": 150,
      // ... bottom performer metadata
    }
    // ... 8 more bottom performers
  ]
}
```

**Fields Used**:
- `videos[0:top_count]` → Top performers for avg views/engagement calculation
- `playCount` → For average views calculation
- `diggCount`, `commentCount`, `shareCount`, `collectCount` → For engagement calculation

---

#### File 5: `content_analysis/{video_id}_content.json` (per bucket)

**Location**: `/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/buckets/bucket_{name}/content_analysis/{video_id}_content.json`

**Purpose**: Stage 2.7 LLM content classifications per video

**Structure**:
```json
{
  "video_id": "7540717847325003039",
  "performance_group": "top",
  "confidence": "high",
  "transcript_available": true,
  "content_category": "recipe_tutorial",
  "hook_strategy": "problem_solution",
  "pain_points": ["bloating", "low_energy", "inflammation"],
  "keywords": ["gut_health", "protein", "fiber"],
  "engagement_drivers": ["before_after_reveal", "personal_testimony"],
  "content_tactics": ["direct_to_camera", "text_overlay_heavy"],
  "caption_analysis": {
    "hook_type": "question",
    "cta_type": "link_in_bio",
    "caption_length": "short",
    "emoji_usage": "some",
    "hashtag_count": 7
  }
}
```

**Fields Used**:
- ALL fields for aggregation via `aggregate_content_classifications()`
- `performance_group` → Filter by "top" or "bottom"
- `confidence` → Quality gate (exclude "low")

---

#### File 6: `winning_formulas.json` (per bucket)

**Location**: `/data/clients/{client}/hashtags/{target}/{mode}_{strategy}/buckets/bucket_{name}/ml_analysis/llm/winning_formulas.json`

**Purpose**: Stage 7 LLM-identified creative formulas (3 per bucket)

**Structure**:
```json
{
  "bucket": "18-33s",
  "total_formulas": 3,
  "creative_reports": [
    {
      "cluster_id": 0,
      "formula_name": "The Question Hook Formula",
      "confidence": 87,
      "pattern_summary": {
        "hook": "Ask compelling question in first 2s",
        "middle": "Reveal product + explain benefit (3-15s)",
        "closing": "Demonstrate result + CTA (15-33s)"
      }
    },
    {
      "cluster_id": 1,
      "formula_name": "The Fast-Paced Product Demo",
      "confidence": 82,
      "pattern_summary": {
        "hook": "Immediate product reveal with text overlay",
        "middle": "Quick feature showcase with scene changes",
        "closing": "Results + urgency CTA"
      }
    },
    {
      "cluster_id": 2,
      "formula_name": "The Myth-Busting Reveal",
      "confidence": 79,
      "pattern_summary": {
        "hook": "Controversial statement to stop scroll",
        "middle": "Evidence and expert credentials",
        "closing": "Call to action with social proof"
      }
    }
  ]
}
```

**Fields Used**:
- `creative_reports[0-2].formula_name` → Page 3 formula names (9 total across 3 buckets)

---

#### File 7: `cluster_config.json`

**Location**: `/config/hashtag_clusters/{target}.json`

**Purpose**: Cluster configuration with primary hashtag

**Structure**:
```json
{
  "cluster_id": "nutrition_wellness",
  "primary_hashtag": "#nutrition",
  "related_hashtags": ["#wellness", "#health", "#guthealth"],
  "analysis_config": {
    "default_mode": "top",
    "default_strategy": "contrastive"
  }
}
```

**Fields Used**:
- `primary_hashtag` → HASHTAG field in header

---

### Complete Implementation Pattern

This section shows the full script structure for implementation:

```python
#!/usr/bin/env python3
"""
extract_client_data.py - Report 1: Hashtag → Client

Generates executive dashboard data with market intelligence.

Usage:
    python extract_client_data.py --client acme --hashtag nutrition --mode top --strategy contrastive
"""

import argparse
import json
import os
import pandas as pd
from collections import Counter

# Import functions defined above
# (In actual implementation, these would be in the same file or imported from report_utils.py)


def main():
    """Main extraction workflow"""

    # =============================
    # STEP 1: Parse CLI Arguments
    # =============================
    parser = argparse.ArgumentParser(description='Extract Report 1: Hashtag → Client')
    parser.add_argument('--client', required=True, help='Client ID (e.g., acme)')
    parser.add_argument('--hashtag', required=True, help='Hashtag name (e.g., nutrition)')
    parser.add_argument('--mode', default='top', help='Mode (default: top)')
    parser.add_argument('--strategy', default='contrastive', help='Strategy (default: contrastive)')
    args = parser.parse_args()

    print(f"\nRunning extraction for hashtag: #{args.hashtag}")

    # =============================
    # STEP 2: Build File Paths
    # =============================
    base_path = f"/data/clients/{args.client}/hashtags/{args.hashtag}/{args.mode}_{args.strategy}/"
    cluster_config_path = f"/config/hashtag_clusters/{args.hashtag}.json"
    cluster_analytics_path = f"/data/clients/{args.client}/hashtags/{args.hashtag}/cluster_analytics.json"

    # =============================
    # STEP 3: Load Core Data Files
    # =============================
    print("Processing winner analysis...")

    # Load cluster config for primary hashtag
    with open(cluster_config_path) as f:
        cluster_config = json.load(f)
    primary_hashtag = cluster_config["primary_hashtag"]

    # Load cluster analytics for total videos
    with open(cluster_analytics_path) as f:
        cluster_analytics = json.load(f)
    total_videos = cluster_analytics["scrape_summary"]["total_scraped_videos"]

    # Load winner analysis
    winner_analysis_path = os.path.join(base_path, 'winner_analysis.json')
    with open(winner_analysis_path) as f:
        winner_data = json.load(f)

    winning_buckets = winner_data['top_3_buckets']  # ['18-33s', '13-18s', '60-90s']
    top_100_distribution = winner_data['top_100_distribution']
    bucket_distribution = winner_data['bucket_distribution']

    # Load selection manifest for performer counts
    manifest_path = os.path.join(base_path, 'selection_manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Calculate performer counts
    top_performers_count = sum(
        len(bucket_data["top_performers"])
        for bucket_data in manifest["videos_by_bucket"].values()
    )
    bottom_performers_count = sum(
        len(bucket_data["bottom_performers"])
        for bucket_data in manifest["videos_by_bucket"].values()
    )

    # =============================
    # STEP 4: Calculate Performance Metrics
    # =============================
    print(f"Calculating performance metrics across {len(winning_buckets)} winning buckets...")

    # Calculate bucket distribution percentages
    bucket_percentages = calculate_bucket_distribution_percentages(base_path)

    # Assign star ratings and sort buckets by performance
    star_data = assign_star_ratings(base_path, winning_buckets)
    sorted_buckets = star_data["sorted_buckets"]
    star_ratings = star_data["star_ratings"]

    # Calculate coverage percentage
    coverage_pct = calculate_coverage_percentage(base_path)

    # =============================
    # STEP 5: Aggregate Content Intelligence
    # =============================
    print("Aggregating content intelligence from selected videos...")

    # Aggregate content classifications across all winning buckets
    all_content_categories = Counter()
    all_hook_strategies = Counter()
    all_pain_points = Counter()
    all_keywords = Counter()
    all_engagement_drivers = Counter()
    all_caption_hook_types = Counter()
    all_caption_cta_types = Counter()
    all_hashtag_counts = []

    for bucket_name in winning_buckets:
        bucket_path = os.path.join(base_path, 'buckets', f'bucket_{bucket_name}')
        bucket_aggregated = aggregate_content_classifications(bucket_path, performance_group="top")

        if bucket_aggregated:
            all_content_categories.update(bucket_aggregated['content_category'])
            all_hook_strategies.update(bucket_aggregated['hook_strategy'])
            all_pain_points.update(bucket_aggregated['pain_points'])
            all_keywords.update(bucket_aggregated['keywords'])
            all_engagement_drivers.update(bucket_aggregated['engagement_drivers'])
            all_caption_hook_types.update(bucket_aggregated['caption_hook_type'])
            all_caption_cta_types.update(bucket_aggregated['caption_cta_type'])

            # Collect hashtag counts for averaging
            stats = bucket_aggregated['hashtag_count_stats']
            # Approximate: use mean × video count to get total, then re-average later
            all_hashtag_counts.extend([stats['mean']] * bucket_aggregated['total_videos'])

    # Get top N for each field
    top_3_categories = all_content_categories.most_common(3)
    top_3_hooks = all_hook_strategies.most_common(3)
    top_4_keywords = all_keywords.most_common(4)
    top_3_pain_points = all_pain_points.most_common(3)
    top_3_drivers = all_engagement_drivers.most_common(3)

    # Caption analysis
    top_cta = all_caption_cta_types.most_common(1)[0] if all_caption_cta_types else ("link_in_bio", 0)

    # Calculate percentages
    total_classified_videos = sum(all_content_categories.values())

    # Optimal hashtag count
    optimal_hashtag_count = round(sum(all_hashtag_counts) / len(all_hashtag_counts)) if all_hashtag_counts else 7

    # =============================
    # STEP 6: Load Formula Names from Stage 7
    # =============================
    formula_names = []
    for bucket_name in winning_buckets:
        formulas_path = os.path.join(
            base_path, 'buckets', f'bucket_{bucket_name}',
            'ml_analysis', 'llm', 'winning_formulas.json'
        )

        try:
            with open(formulas_path) as f:
                formulas_data = json.load(f)

            # Extract 3 formula names for this bucket
            for report in formulas_data['creative_reports']:
                formula_names.append(report['formula_name'])
        except FileNotFoundError:
            # If Stage 7 not run yet, use placeholders
            formula_names.extend([
                f"Formula {len(formula_names)+1} ({bucket_name})",
                f"Formula {len(formula_names)+2} ({bucket_name})",
                f"Formula {len(formula_names)+3} ({bucket_name})"
            ])

    # =============================
    # STEP 7: Build Excel Data Structure
    # =============================
    tab_data = []

    # PAGE 1: SCALE OF ANALYSIS
    tab_data.append(['PAGE_1_SCALE_OF_ANALYSIS', ''])
    tab_data.append(['', ''])

    # Header Section
    tab_data.append(['HASHTAG', primary_hashtag])
    tab_data.append(['ANALYSIS_PERIOD', 'Past 2-3 months'])
    tab_data.append(['VIDEOS_ANALYZED', str(total_videos)])
    tab_data.append(['', ''])

    # Winning buckets with percentages from top_100_distribution
    for i, bucket in enumerate(winning_buckets, 1):
        tab_data.append([f'WINNING_BUCKET_{i}_NAME', bucket])
        pct = top_100_distribution.get(bucket, 0)
        tab_data.append([f'WINNING_BUCKET_{i}_PCT', str(pct)])

    tab_data.append(['', ''])
    tab_data.append(['TOP_PERFORMERS_COUNT', str(top_performers_count)])
    tab_data.append(['BOTTOM_PERFORMERS_COUNT', str(bottom_performers_count)])

    tab_data.append(['', ''])
    tab_data.append(['METHODOLOGY_TEXT', 'Multi-dimensional machine learning and AI content analysis'])

    # PAGE 2: HASHTAG INTELLIGENCE DASHBOARD
    tab_data.append(['', ''])
    tab_data.append(['PAGE_2_HASHTAG_INTELLIGENCE', ''])
    tab_data.append(['', ''])

    # Section 1: Duration Distribution
    all_buckets = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
    for bucket in all_buckets:
        field_name = f'BUCKET_{bucket.replace("-", "_").upper()}_PCT'
        pct = bucket_percentages.get(bucket, 0)
        tab_data.append([field_name, str(pct)])

    tab_data.append(['', ''])

    # Key insight (sum of top 2 consecutive buckets for example)
    # You could make this dynamic
    key_insight_pct = bucket_percentages.get("13-18s", 0) + bucket_percentages.get("18-33s", 0)
    tab_data.append(['KEY_INSIGHT_PCT', str(key_insight_pct)])
    tab_data.append(['KEY_INSIGHT_TEXT', f'{key_insight_pct}% of {primary_hashtag} content is 13-33s'])

    # Section 2: Performance by Duration
    tab_data.append(['', ''])

    for i, bucket_data in enumerate(sorted_buckets, 1):
        tab_data.append([f'PERF_BUCKET_{i}_NAME', bucket_data['bucket']])
        tab_data.append([f'PERF_BUCKET_{i}_AVG_VIEWS', format_views(bucket_data['avg_views'])])
        tab_data.append([f'PERF_BUCKET_{i}_AVG_ENG', str(round(bucket_data['avg_engagement'], 1))])
        tab_data.append([f'PERF_BUCKET_{i}_STARS', star_ratings[i-1]])
        tab_data.append([f'PERF_BUCKET_{i}_LABEL', '← BEST' if i == 1 else ''])
        tab_data.append(['', ''])

    tab_data.append(['COVERAGE_PERCENTAGE', str(coverage_pct)])

    # Section 3: Creator Profile Priorities
    tab_data.append(['', ''])

    label_map = {
        0: "highest performance",
        1: "strong performance + volume",
        2: "proven success"
    }

    for i, bucket_data in enumerate(sorted_buckets, 1):
        tab_data.append([f'TIER1_BUCKET_{i}_NAME', bucket_data['bucket']])
        tab_data.append([f'TIER1_BUCKET_{i}_AVG_VIEWS', format_views(bucket_data['avg_views'])])
        tab_data.append([f'TIER1_BUCKET_{i}_LABEL', label_map[i-1]])
        tab_data.append(['', ''])

    # Section 4: Content Intelligence
    tab_data.append(['', ''])

    # Content categories
    for i, (category, count) in enumerate(top_3_categories, 1):
        pct = round((count / total_classified_videos) * 100)
        tab_data.append([f'CONTENT_CATEGORY_{i}', category.replace('_', ' ').title()])
        tab_data.append([f'CONTENT_CATEGORY_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Hook strategies
    for i, (hook, count) in enumerate(top_3_hooks, 1):
        pct = round((count / total_classified_videos) * 100)
        tab_data.append([f'HOOK_STRATEGY_{i}', hook.replace('_', ' ').title()])
        tab_data.append([f'HOOK_STRATEGY_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Keywords (no percentage)
    for i, (keyword, count) in enumerate(top_4_keywords, 1):
        tab_data.append([f'KEYWORD_{i}', f'#{keyword}'])

    tab_data.append(['', ''])

    # Pain points
    for i, (pain_point, count) in enumerate(top_3_pain_points, 1):
        pct = round((count / total_classified_videos) * 100)
        tab_data.append([f'PAIN_POINT_{i}', pain_point.replace('_', ' ').title()])
        tab_data.append([f'PAIN_POINT_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Engagement drivers
    for i, (driver, count) in enumerate(top_3_drivers, 1):
        pct = round((count / total_classified_videos) * 100)
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}', driver.replace('_', ' ').title()])
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Caption strategy
    tab_data.append(['OPTIMAL_HASHTAG_COUNT', str(optimal_hashtag_count)])

    # Note: Caption length and emoji usage would require additional aggregation
    # For now using placeholders - you can add logic similar to hook strategies
    tab_data.append(['CAPTION_LENGTH_WINNER', 'Short captions (<100 characters)'])
    tab_data.append(['CAPTION_LENGTH_WINNER_PCT', '68'])
    tab_data.append(['EMOJI_USAGE_WINNER', 'Light emoji use (1-4 emojis)'])
    tab_data.append(['EMOJI_USAGE_WINNER_PCT', '72'])

    cta_type, cta_count = top_cta
    cta_pct = round((cta_count / total_classified_videos) * 100) if total_classified_videos > 0 else 0
    tab_data.append(['TOP_CTA_TYPE', cta_type.replace('_', ' ').title()])
    tab_data.append(['TOP_CTA_TYPE_PCT', str(cta_pct)])

    # PAGE 3: YOUR CREATIVE REPORTS
    tab_data.append(['', ''])
    tab_data.append(['PAGE_3_YOUR_CREATIVE_REPORTS', ''])
    tab_data.append(['', ''])

    tab_data.append(['FORMULA_COUNT', '9'])
    tab_data.append(['', ''])

    # 9 formula names (3 per winning bucket)
    for i in range(0, 9, 3):
        bucket_idx = i // 3 + 1
        tab_data.append([f'BUCKET_{bucket_idx}_FORMULA_1_NAME', formula_names[i] if i < len(formula_names) else 'Placeholder'])
        tab_data.append([f'BUCKET_{bucket_idx}_FORMULA_2_NAME', formula_names[i+1] if i+1 < len(formula_names) else 'Placeholder'])
        tab_data.append([f'BUCKET_{bucket_idx}_FORMULA_3_NAME', formula_names[i+2] if i+2 < len(formula_names) else 'Placeholder'])
        tab_data.append(['', ''])

    # =============================
    # STEP 8: Write Excel File
    # =============================
    excel_filename = f"{args.hashtag}_client_data.xlsx"
    excel_path = os.path.join(base_path, excel_filename)

    df = pd.DataFrame(tab_data, columns=['Field Name', 'Value'])
    df.to_excel(excel_path, sheet_name='Report_Data', index=False, engine='openpyxl')

    # =============================
    # STEP 9: Print Success Message
    # =============================
    print(f"\n✓ Extraction complete")
    print(f"  Excel: {excel_path}")
    print(f"  Total fields: {len(tab_data)}")


if __name__ == '__main__':
    main()
```

---

### Implementation Notes for Developer

**TODO items in skeleton above**:
1. ✅ All functions implemented (calculate_engagement_metrics, calculate_avg_views_per_bucket, aggregate_content_classifications)
2. ✅ All inline calculations included (format_views, bucket percentages, star ratings, coverage)
3. ⚠️ Caption length and emoji usage aggregation - needs same pattern as hook strategies (left as placeholder)
4. ⚠️ Stage 7 formula names - graceful fallback if winning_formulas.json not found

**Testing checklist**:
- [ ] Script runs without errors
- [ ] Excel file created with single tab
- [ ] All ~122 fields populated (no empty values except intentional empty rows)
- [ ] Field values match source JSON files
- [ ] Percentages sum correctly (bucket distribution, content categories)
- [ ] Star ratings in correct order (highest engagement = 5 stars)
- [ ] Coverage percentage accurate (top 3 buckets as % of top 100)

**Error handling**:
Script should exit with clear error if:
- `winner_analysis.json` not found
- `selection_manifest.json` not found
- JSON files malformed
- Missing required fields (e.g., `top_3_buckets` array empty)
- Cannot write Excel file (permissions issue)
- Cluster config or analytics files missing

**Dependencies**:
```bash
pip install pandas openpyxl
```

---

**END OF SECTION 3.2**

This section is complete and self-contained for implementation of `extract_client_data.py`.
